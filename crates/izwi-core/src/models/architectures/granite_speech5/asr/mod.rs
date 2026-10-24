//! Native Granite Speech 5.0 ASR and speech translation.
//!
//! Granite Speech 5.0 keeps the tensor layout of Granite Speech 4.1 2B (a
//! conformer encoder, a window Q-Former projector and a Granite language
//! model) but its language model is YaRN-extended to 16k positions and it
//! has its own prompt contract. It is a separate family: nothing here calls
//! the 4.1 module, which stays as it shipped.
//!
//! A request runs as one atomic invocation on a lifecycle-owned paged KV
//! lease; long audio is cut into chunks by the generic ASR route. The 4.1
//! retained-decode route is uncertified (GRN-8) and is not carried over.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use candle_core::DType;
use serde_json::json;
use tracing::info;

use crate::backends::{DeviceKind, DeviceProfile};
use crate::catalog::ModelVariant;
use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, CapabilityStateDescriptorV2, CheckpointPolicy, InferenceStateContract,
    InvocationLeaseScope, InvocationStageWorkspace, InvocationStateCapacity,
    InvocationWorkspaceDomain, InvocationWorkspaceProfile, InvocationWorkspaceSet, PlacementPolicy,
    PrefixPolicy, RetainedStateCapability, StateClock, StateDType, StateDomainId, StateDomainSpec,
    StateGroupId, StateGroupSpec, StateScope, WorkspaceFormula, CURRENT_INFERENCE_STATE_ABI,
};
use crate::models::architectures::qwen3::core::{
    qwen3_decoder_cache_domain, Qwen3DecoderCacheGeometry,
};
use crate::models::shared::attention::paged::default_kv_page_size;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

mod config;
mod encoder;
mod gguf;
mod language_model;
mod precision;
mod preprocessor;
mod projector;
mod prompt;
#[cfg(test)]
mod reference_tests;
mod runtime;
mod transcript;

pub use config::{
    load_granite_speech5_chat_template, GraniteSpeech5AudioProcessorConfig, GraniteSpeech5Config,
    GraniteSpeech5EncoderConfig, GraniteSpeech5GenerationConfig, GraniteSpeech5MelSpecConfig,
    GraniteSpeech5ProcessorConfig, GraniteSpeech5ProjectorConfig, GraniteSpeech5TextConfig,
    GraniteSpeech5TokenizerConfig,
};
pub(crate) use gguf::granite_speech5_mmproj_safetensors_name;
pub(crate) use precision::{
    granite_speech5_load_representation, granite_speech5_tensor_is_f16_encoder_projection,
    resolve_granite_speech5_encoder_precision, select_granite_speech5_dtype,
    GraniteSpeech5LoadRepresentation,
};
pub use preprocessor::{GraniteSpeech5AudioFeatures, GraniteSpeech5Preprocessor};
pub use prompt::{
    classify_granite_speech5_prompt, normalize_granite_speech5_keywords,
    render_granite_speech5_prompt, resolve_granite_speech5_source_language,
    resolve_granite_speech5_translation, GraniteSpeech5Language, GraniteSpeech5Prompt,
    GraniteSpeech5PromptOptions, GraniteSpeech5PromptOverride, GraniteSpeech5PromptTokenizer,
    GraniteSpeech5RenderedPrompt, GraniteSpeech5SpecialTokens, GraniteSpeech5Task,
    GRANITE_SPEECH5_KEYWORD_ASR_INSTRUCTION, GRANITE_SPEECH5_MAX_KEYWORDS,
    GRANITE_SPEECH5_MAX_KEYWORD_CHARS, GRANITE_SPEECH5_PUNCTUATED_ASR_INSTRUCTION,
    GRANITE_SPEECH5_RAW_ASR_INSTRUCTION, GRANITE_SPEECH5_SOURCE_LANGUAGES,
    GRANITE_SPEECH5_TRANSLATION_TARGETS,
};
pub use runtime::{
    GraniteSpeech5AudioStats, GraniteSpeech5Generation, GraniteSpeech5GenerationStats,
    GraniteSpeech5Runtime,
};
pub use transcript::clean_granite_speech5_output;

/// Files both releases' loaders read besides the weights.
/// `generation_config.json` is optional.
const CONFIG_ARTIFACTS: &[&str] = &[
    "chat_template.jinja",
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
];
const WEIGHTS_FILE: &str = "model.safetensors";
/// IBM's GGUF release: the Q8_0 language model, and the F16 audio encoder
/// and projector.
pub const GRANITE_SPEECH5_GGUF_LM_FILE: &str = "granite-speech-5.0-2b-Q8_0.gguf";
pub const GRANITE_SPEECH5_GGUF_MMPROJ_FILE: &str = "mmproj-model-f16.gguf";

/// Where a Granite Speech 5.0 checkpoint keeps its weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraniteSpeech5Weights {
    /// The safetensors release: one file.
    Safetensors(PathBuf),
    /// The GGUF release: the language model, and the encoder and projector.
    Gguf {
        language_model: PathBuf,
        mmproj: PathBuf,
    },
}

impl GraniteSpeech5Weights {
    pub fn files(&self) -> Vec<PathBuf> {
        match self {
            Self::Safetensors(path) => vec![path.clone()],
            Self::Gguf {
                language_model,
                mmproj,
            } => vec![language_model.clone(), mmproj.clone()],
        }
    }
}

/// Most audio one invocation covers: IBM recommends segments of about two
/// minutes or less. Longer audio is chunked by the generic ASR route.
pub const GRANITE_SPEECH5_MAX_AUDIO_SECONDS: f32 = 120.0;
/// Audio tokens per second: 50 stacked frames per second through a 15-frame
/// window that emits 3 queries.
pub const GRANITE_SPEECH5_AUDIO_TOKENS_PER_SECOND: f32 = 10.0;
/// The automatic decode budget: tokens per 30 s of audio. IBM's library uses
/// 200; Japanese and Mandarin output run about 1.1 tokens per character with
/// the shipped tokenizer, so 5.0 allows half as much again.
const AUTO_TOKENS_PER_WINDOW: f64 = 300.0;
const AUTO_WINDOW_SECONDS: f64 = 30.0;
/// The admitted ceiling of the automatic budget.
pub const GRANITE_SPEECH5_AUTO_MAX_TOKENS: usize = 2048;
const DEFAULT_MAX_NEW_TOKENS: usize = 768;

static NEXT_MODEL_IDENTITY: AtomicU64 = AtomicU64::new(1);

/// The automatic decode budget for `audio_seconds` of audio,
/// `ceil(300 * max(1, secs / 30))`. Callers bound it by the request ceiling
/// and the invocation context.
pub fn granite_speech5_auto_max_tokens_for_duration(audio_seconds: f32) -> usize {
    let windows = if audio_seconds.is_finite() {
        (f64::from(audio_seconds) / AUTO_WINDOW_SECONDS).max(1.0)
    } else {
        1.0
    };
    (AUTO_TOKENS_PER_WINDOW * windows).ceil() as usize
}

/// What a request asks of Granite Speech 5.0 beyond its audio.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GraniteSpeech5RequestOptions {
    /// The spoken language, when the caller knows it.
    pub language: Option<String>,
    /// A free-form prompt: a Granite instruction, a `keywords:` list, or
    /// context text (ignored).
    pub prompt: Option<String>,
    /// Translate into this language instead of transcribing.
    pub translate_to: Option<String>,
    /// Terms to bias recognition toward.
    pub keywords: Vec<String>,
}

impl GraniteSpeech5RequestOptions {
    /// The prompt options these request options select, validated against
    /// 5.0's languages, translation pairs and keyword limits.
    pub fn prompt_options(&self) -> Result<GraniteSpeech5PromptOptions> {
        let source = resolve_granite_speech5_source_language(self.language.as_deref())?;
        let task = match self
            .translate_to
            .as_deref()
            .map(str::trim)
            .filter(|target| !target.is_empty())
        {
            Some(target) => GraniteSpeech5Task::Translate {
                target: resolve_granite_speech5_translation(source, target)?,
            },
            None => GraniteSpeech5Task::Transcribe,
        };
        Ok(GraniteSpeech5PromptOptions {
            task,
            source_language: source.map(|language| language.code().to_string()),
            punctuate: true,
            keywords: self.keywords.clone(),
            custom_prompt: self.prompt.clone(),
        })
    }
}

/// Reject a request Granite Speech 5.0 cannot serve (an unsupported
/// language, translation pair or keyword list) before it is admitted.
pub fn validate_granite_speech5_request(options: &GraniteSpeech5RequestOptions) -> Result<()> {
    let prompt = options.prompt_options()?;
    // Renders with the published audio token; the tokenizer is not needed to
    // validate the instruction.
    render_granite_speech5_prompt("<|audio|>", &prompt).map(|_| ())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeech5GenerationOptions {
    pub max_new_tokens: usize,
    pub stop_token_ids: Vec<u32>,
    pub stop_sequences: Vec<String>,
}

impl Default for GraniteSpeech5GenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: DEFAULT_MAX_NEW_TOKENS,
            stop_token_ids: Vec::new(),
            stop_sequences: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraniteSpeech5Transcription {
    pub text: String,
    /// The output's language: the translation target, or the spoken
    /// language the request named.
    pub language: Option<String>,
    pub diagnostics: Option<serde_json::Value>,
}

/// The invocation state a load publishes for its atomic ASR graphs.
#[derive(Debug, Clone)]
pub(crate) struct GraniteSpeech5PhysicalStateSpec {
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) invocation: InferenceStateContract,
}

pub struct GraniteSpeech5AsrModel {
    model_identity: u64,
    model_dir: PathBuf,
    device: DeviceProfile,
    dtype: DType,
    config: GraniteSpeech5Config,
    processor: GraniteSpeech5ProcessorConfig,
    chat_template: String,
    prompt_tokenizer: GraniteSpeech5PromptTokenizer,
    preprocessor: GraniteSpeech5Preprocessor,
    runtime: GraniteSpeech5Runtime,
}

impl GraniteSpeech5AsrModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        let weights = ensure_granite_speech5_artifacts(model_dir, variant)?;
        let config = GraniteSpeech5Config::load(model_dir)?;
        config.validate()?;
        if config.tie_word_embeddings() {
            return Err(Error::ModelLoadError(
                "Granite Speech 5.0 ships an untied lm_head, but the config ties the output head to the embedding"
                    .to_string(),
            ));
        }
        let processor = GraniteSpeech5ProcessorConfig::load(model_dir)?;
        let generation = GraniteSpeech5GenerationConfig::load(model_dir, &config)?;
        let tokenizer_config = GraniteSpeech5TokenizerConfig::load(model_dir)?;
        let chat_template = load_granite_speech5_chat_template(model_dir)?;
        let prompt_tokenizer = GraniteSpeech5PromptTokenizer::load(
            model_dir,
            &config,
            &processor,
            &tokenizer_config,
            generation.eos_token_id,
            generation.pad_token_id,
        )?;
        let preprocessor = GraniteSpeech5Preprocessor::new(processor.clone())?;
        let dtype = select_granite_speech5_dtype(&device, config.target_dtype_hint())?;
        let encoder_precision = resolve_granite_speech5_encoder_precision(&device);
        let runtime =
            GraniteSpeech5Runtime::load(&weights, &config, &device, dtype, &encoder_precision)?;
        info!(
            "Loaded Granite Speech 5.0 ASR from {:?} on {:?} with dtype {:?}; encoder: {}; LM: {}",
            model_dir,
            device.kind,
            dtype,
            runtime.encoder_precision().reason,
            runtime.lm_precision().reason,
        );
        Ok(Self {
            model_identity: NEXT_MODEL_IDENTITY.fetch_add(1, Ordering::Relaxed),
            model_dir: model_dir.to_path_buf(),
            device,
            dtype,
            config,
            processor,
            chat_template,
            prompt_tokenizer,
            preprocessor,
            runtime,
        })
    }

    pub fn config(&self) -> &GraniteSpeech5Config {
        &self.config
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn max_audio_seconds_hint(&self) -> Option<f32> {
        Some(GRANITE_SPEECH5_MAX_AUDIO_SECONDS)
    }

    /// Decoder-prompt tokens added per second of audio.
    pub(crate) fn audio_token_rate(&self) -> f32 {
        let frames_per_second = self.processor.sample_rate() as f32
            / self
                .processor
                .audio_processor
                .melspec_kwargs
                .hop_length
                .max(1) as f32
            / 2.0;
        frames_per_second * self.config.queries_per_window() as f32 / self.config.window_size as f32
    }

    pub fn build_prompt(
        &self,
        options: &GraniteSpeech5PromptOptions,
    ) -> Result<GraniteSpeech5Prompt> {
        self.prompt_tokenizer.build_prompt(options)
    }

    /// The invocation state for the loaded ASR capability's stage graphs:
    /// each must be the single atomic stage, which leases one paged decoder
    /// cache of the model's full context.
    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<GraniteSpeech5PhysicalStateSpec> {
        let invocation = granite_speech5_invocation_contract(
            &self.config,
            self.runtime.kv_dtype(),
            default_kv_page_size(),
        )?;
        granite_speech5_physical_state_spec(
            stage_graphs,
            invocation,
            self.config.text_config.max_position_embeddings,
        )
    }

    /// Transcribe or translate one invocation of audio on `cache`. With
    /// `on_delta`, text streams as it decodes.
    pub(crate) fn transcribe_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        request: &GraniteSpeech5RequestOptions,
        options: GraniteSpeech5GenerationOptions,
        cache: &mut PhysicalPagedKvCache,
        on_delta: Option<&mut dyn FnMut(&str)>,
    ) -> Result<GraniteSpeech5Transcription> {
        let started = Instant::now();
        let prompt_options = request.prompt_options()?;
        let prompt = self.build_prompt(&prompt_options)?;
        let features = self.preprocessor.prepare(audio, sample_rate)?;
        if features.audio_seconds > GRANITE_SPEECH5_MAX_AUDIO_SECONDS {
            return Err(Error::InvalidInput(format!(
                "Granite Speech 5.0 transcribes up to {GRANITE_SPEECH5_MAX_AUDIO_SECONDS:.0}s per invocation, got {:.1}s",
                features.audio_seconds
            )));
        }
        let mel_prepare = started.elapsed();
        let (audio_embeds, audio_stats) = self.runtime.audio_embeddings(&features)?;
        let mut decode = |ids: &[u32]| self.prompt_tokenizer.decode(ids);
        let mut no_delta = |_delta: &str| {};
        let emit_deltas = on_delta.is_some();
        let on_delta = on_delta.unwrap_or(&mut no_delta);
        let generation = self.runtime.generate(
            &prompt,
            self.prompt_tokenizer.special_tokens(),
            &audio_embeds,
            options.max_new_tokens,
            &options.stop_token_ids,
            &options.stop_sequences,
            &mut decode,
            cache,
            on_delta,
            emit_deltas,
        )?;
        let language = match prompt.task {
            GraniteSpeech5Task::Translate { target } => Some(target.code().to_string()),
            GraniteSpeech5Task::Transcribe => prompt_options.source_language.clone(),
        };
        let diagnostics = self.diagnostics(
            &features,
            &prompt,
            &generation,
            audio_stats,
            mel_prepare,
            started.elapsed(),
        );
        Ok(GraniteSpeech5Transcription {
            text: clean_granite_speech5_output(&generation.text),
            language,
            diagnostics: Some(diagnostics),
        })
    }

    pub fn diagnostics_summary(&self) -> serde_json::Value {
        json!({
            "family": "granite_speech5_asr",
            "dtype": format!("{:?}", self.dtype),
            "device_kind": format!("{:?}", self.device.kind),
            "sample_rate": self.processor.sample_rate(),
            "n_mels": self.processor.audio_processor.melspec_kwargs.n_mels,
            "audio_token_index": self.config.audio_token_index,
            "max_position_embeddings": self.config.text_config.max_position_embeddings,
            "chat_template_bytes": self.chat_template.len(),
            "max_audio_seconds": GRANITE_SPEECH5_MAX_AUDIO_SECONDS,
            "encoder_weight_precision": self.runtime.encoder_precision().diagnostics(),
            "lm_precision": self.runtime.lm_precision().diagnostics(),
            "execution": self.runtime.execution_diagnostics(),
        })
    }

    fn diagnostics(
        &self,
        features: &GraniteSpeech5AudioFeatures,
        prompt: &GraniteSpeech5Prompt,
        generation: &GraniteSpeech5Generation,
        audio: GraniteSpeech5AudioStats,
        mel_prepare: Duration,
        total: Duration,
    ) -> serde_json::Value {
        let stats = &generation.stats;
        let mut execution = self.runtime.execution_diagnostics();
        execution["encoder_weight_precision"] = self.runtime.encoder_precision().diagnostics();
        execution["lm_precision"] = self.runtime.lm_precision().diagnostics();
        execution["stop_check_interval"] = json!(stats.stop_check_interval);
        json!({
            "family": "granite_speech5_asr",
            "dtype": format!("{:?}", self.dtype),
            "device_kind": format!("{:?}", self.device.kind),
            "audio_seconds": features.audio_seconds,
            "sample_rate": features.sample_rate,
            "prompt": {
                "task": prompt.task.name(),
                "instruction": prompt.instruction,
                "prompt_ignored": prompt.prompt_ignored,
                "keywords": prompt.keyword_count,
                "prompt_tokens": stats.prompt_tokens,
            },
            "audio": {
                "audio_tokens": stats.audio_tokens,
                "mel_frames": features.mel_frames,
                "encoder_frames": audio.encoder_frames,
                "conformer_context_size": audio.conformer_context_size,
                "conformer_layers": audio.conformer_layers,
                "qformer_windows": audio.qformer_windows,
                "qformer_window_size": audio.qformer_window_size,
                "qformer_queries_per_window": audio.qformer_queries_per_window,
                "qformer_layers": audio.qformer_layers,
            },
            "projected_audio_tokens": stats.audio_tokens,
            "prompt_tokens": stats.prompt_tokens,
            "generated_tokens": stats.generated_tokens,
            "stop_reason": stats.stop_reason,
            "stop_token": stats.stop_token,
            "decode": {
                "generated_tokens": stats.generated_tokens,
                "max_new_tokens": stats.max_new_tokens,
                "stop_reason": stats.stop_reason,
                "stop_token": stats.stop_token,
            },
            "execution": execution,
            "timings_ms": {
                "mel_prepare": duration_ms(mel_prepare),
                "audio_input_upload": duration_ms(audio.upload),
                "audio_encoder": duration_ms(audio.encoder),
                "audio_projector": duration_ms(audio.projector),
                "prefill": duration_ms(stats.prefill),
                "decode": duration_ms(stats.decode),
                "model_total": duration_ms(total),
            },
        })
    }

    #[cfg(test)]
    pub(crate) fn model_identity(&self) -> u64 {
        self.model_identity
    }

    #[cfg(test)]
    pub(crate) fn runtime(&self) -> &GraniteSpeech5Runtime {
        &self.runtime
    }

    #[cfg(test)]
    pub(crate) fn special_tokens(&self) -> &GraniteSpeech5SpecialTokens {
        self.prompt_tokenizer.special_tokens()
    }

    #[cfg(test)]
    pub(crate) fn preprocessor(&self) -> &GraniteSpeech5Preprocessor {
        &self.preprocessor
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// The checkpoint's weights file, once every file the loader reads exists.
pub fn ensure_granite_speech5_artifacts(
    model_dir: &Path,
    variant: ModelVariant,
) -> Result<GraniteSpeech5Weights> {
    let weights = if variant.is_granite_speech5_gguf() {
        GraniteSpeech5Weights::Gguf {
            language_model: model_dir.join(GRANITE_SPEECH5_GGUF_LM_FILE),
            mmproj: model_dir.join(GRANITE_SPEECH5_GGUF_MMPROJ_FILE),
        }
    } else {
        GraniteSpeech5Weights::Safetensors(model_dir.join(WEIGHTS_FILE))
    };
    let configs = CONFIG_ARTIFACTS.iter().map(|file| model_dir.join(file));
    for path in configs.chain(weights.files()) {
        if !path.is_file() {
            return Err(Error::ModelNotFound(format!(
                "Missing Granite Speech 5.0 artifact {}",
                path.display()
            )));
        }
    }
    Ok(weights)
}

fn granite_speech5_invocation_contract(
    config: &GraniteSpeech5Config,
    dtype: DType,
    preferred_page_tokens: usize,
) -> Result<InferenceStateContract> {
    let text = &config.text_config;
    let domain_id = StateDomainId::new(1);
    let mut domain = qwen3_decoder_cache_domain(Qwen3DecoderCacheGeometry {
        domain: domain_id,
        clock: StateClock::DecoderTokens,
        num_layers: text.num_hidden_layers,
        num_query_heads: text.num_attention_heads,
        num_kv_heads: text.num_key_value_heads,
        key_head_dim: config.decoder_head_dim(),
        value_head_dim: config.decoder_head_dim(),
        sliding_window: None,
        storage_dtype: dtype,
        preferred_page_tokens,
        prefix: PrefixPolicy::Disabled,
    })?;
    domain.header.scope = StateScope::Invocation;
    domain.header.prefix = PrefixPolicy::Disabled;
    domain.header.checkpoint = CheckpointPolicy::None;
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: vec![StateDomainSpec::PagedAttention(domain)],
        groups: vec![StateGroupSpec {
            id: StateGroupId::new(1),
            domains: vec![domain_id],
            prefix_shareable: false,
        }],
    };
    contract.validate()?;
    Ok(contract)
}

fn granite_speech5_physical_state_spec(
    stage_graphs: &[&[StageDescriptor]],
    invocation: InferenceStateContract,
    max_context_tokens: usize,
) -> Result<GraniteSpeech5PhysicalStateSpec> {
    if stage_graphs.is_empty() || max_context_tokens == 0 {
        return Err(Error::ModelLoadError(
            "Granite Speech 5.0 invocation state requires stages and a non-zero text context"
                .into(),
        ));
    }
    // One unbatched stage per graph: the scalar ASR stage, which plans
    // long audio's chunks inside its dispatch. Sequence stages would need
    // retained decoder state this family does not author.
    let invocation_only = |stages: &[StageDescriptor]| {
        stages.len() == 1
            && stages[0].batch_mode == crate::engine::NativeBatchMode::None
            && !matches!(
                stages[0].selector,
                crate::engine::StageWorkSelector::SequencePrefill
                    | crate::engine::StageWorkSelector::SequenceDecode
                    | crate::engine::StageWorkSelector::PreSequencePreparation
            )
    };
    if !stage_graphs.iter().all(|stages| invocation_only(stages)) {
        return Err(Error::ModelLoadError(
            "Granite Speech 5.0 ASR runs one unbatched invocation stage per graph".into(),
        ));
    }
    let max_tokens = u64::try_from(max_context_tokens)
        .map_err(|_| Error::ModelLoadError("Granite Speech 5.0 context exceeds u64".into()))?;
    let max_domain_id = invocation
        .domains
        .iter()
        .map(|domain| domain.id().get())
        .max()
        .ok_or_else(|| {
            Error::ModelLoadError("Granite Speech 5.0 invocation contract is empty".into())
        })?;
    let mut profiles = Vec::with_capacity(stage_graphs.len());
    for stages in stage_graphs {
        let stage = &stages[0];
        let mut domains = invocation
            .domains
            .iter()
            .cloned()
            .map(|state| {
                Ok(InvocationWorkspaceDomain::State {
                    placement: state.header().placement,
                    formula: WorkspaceFormula {
                        fixed_bytes: paged_invocation_bytes(&state, max_tokens)?,
                        dimensions: vec![],
                        terms: vec![],
                    },
                    state,
                    capacity: InvocationStateCapacity::decoder_context(max_tokens)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if stage.max_workspace_bytes > 0 {
            domains.push(InvocationWorkspaceDomain::Scratch {
                id: StateDomainId::new(max_domain_id.checked_add(1).ok_or_else(|| {
                    Error::ModelLoadError("Granite Speech 5.0 scratch domain id overflow".into())
                })?),
                placement: PlacementPolicy::BackendLocal,
                alignment_bytes: 64,
                zero_on_release: false,
                formula: WorkspaceFormula {
                    fixed_bytes: stage.max_workspace_bytes,
                    dimensions: vec![],
                    terms: vec![],
                },
            });
        }
        profiles.push(InvocationWorkspaceProfile {
            stage_graph_fingerprint: stage_graph_fingerprint(stages)?,
            stages: vec![InvocationStageWorkspace {
                stage: stage.id,
                lease_scope: InvocationLeaseScope::PerRow,
                groups: invocation.groups.clone(),
                domains,
            }],
        });
    }
    profiles.sort_unstable_by_key(|profile| profile.stage_graph_fingerprint);
    profiles.dedup();
    let descriptor = CapabilityStateDescriptorV2 {
        abi: CURRENT_INFERENCE_STATE_ABI,
        retained: RetainedStateCapability::Stateless,
        invocation: InvocationWorkspaceSet::Bounded { profiles },
    };
    for stages in stage_graphs {
        descriptor.validate_against_stages(stages)?;
    }
    Ok(GraniteSpeech5PhysicalStateSpec {
        descriptor,
        invocation,
    })
}

fn paged_invocation_bytes(state: &StateDomainSpec, max_tokens: u64) -> Result<u64> {
    let StateDomainSpec::PagedAttention(spec) = state else {
        return Err(Error::ModelLoadError(
            "Granite Speech 5.0 invocation workspace is not paged attention".into(),
        ));
    };
    let overflow = || Error::ModelLoadError("Granite Speech 5.0 invocation bytes overflow".into());
    let page_tokens = u64::from(spec.page_size.preferred_tokens).max(1);
    let rounded_tokens = max_tokens
        .div_ceil(page_tokens)
        .checked_mul(page_tokens)
        .ok_or_else(overflow)?;
    let elements_per_token = spec.layers.iter().try_fold(0_u64, |total, layer| {
        u64::from(layer.kv_heads)
            .checked_mul(u64::from(layer.key_head_dim) + u64::from(layer.value_head_dim))
            .and_then(|layer| total.checked_add(layer))
            .ok_or_else(overflow)
    })?;
    let element_bytes = spec
        .accepted_dtypes
        .iter()
        .map(|dtype| match dtype {
            StateDType::F32 => Ok(4_u64),
            StateDType::F16 | StateDType::Bf16 => Ok(2_u64),
            StateDType::I64 | StateDType::I8 | StateDType::Q4 => Err(Error::ModelLoadError(
                "Granite Speech 5.0 invocation paging requires a dense KV dtype".into(),
            )),
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .min()
        .ok_or_else(|| Error::ModelLoadError("Granite Speech 5.0 KV dtype set is empty".into()))?;
    elements_per_token
        .checked_mul(rounded_tokens)
        .and_then(|elements| elements.checked_mul(element_bytes))
        .ok_or_else(overflow)
}

/// The backend a loaded device runs on, for adapter checks.
pub(crate) fn granite_speech5_backend(device: &DeviceProfile) -> crate::backends::BackendKind {
    match device.kind {
        DeviceKind::Cpu => crate::backends::BackendKind::Cpu,
        DeviceKind::Metal => crate::backends::BackendKind::Metal,
        DeviceKind::Cuda => crate::backends::BackendKind::Cuda,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{ExecutionMode, ExecutionProfile, StageId, StageWorkSelector};

    fn published_config() -> GraniteSpeech5Config {
        GraniteSpeech5Config::load(&config::tests::fixture_dir()).unwrap()
    }

    fn atomic_stage(id: u32) -> StageDescriptor {
        let mut profile = ExecutionProfile::fail_closed(
            crate::backends::BackendKind::Cpu,
            None,
            ExecutionMode::Atomic,
        );
        profile.max_batch_size = 1;
        let mut stage = StageDescriptor::from_execution_profile(
            StageId::new(id),
            "asr.granite_speech5.atomic",
            &profile,
            crate::engine::NativeBatchMode::None,
        );
        stage.selector = StageWorkSelector::Atomic;
        stage
    }

    /// Granite Speech 5.0 is its own architecture: neither module may call
    /// the other, so a change to one cannot alter the other's behaviour.
    #[test]
    fn granite_speech_4_1_and_5_0_modules_never_import_each_other() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/models/architectures");
        fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    sources(&path, out);
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    out.push(path);
                }
            }
        }
        for (module, forbidden) in [
            ("granite_speech5", concat!("granite_speech", "::")),
            ("granite_speech", concat!("granite_speech", "5")),
        ] {
            let mut files = Vec::new();
            sources(&root.join(module), &mut files);
            assert!(!files.is_empty(), "{module}");
            for file in files {
                let source = std::fs::read_to_string(&file).unwrap();
                assert!(
                    !source.contains(forbidden),
                    "{} references {forbidden}",
                    file.display()
                );
            }
        }
    }

    #[test]
    fn invocation_state_leases_the_full_16k_context_of_paged_kv() {
        let config = published_config();
        let invocation =
            granite_speech5_invocation_contract(&config, DType::F16, default_kv_page_size())
                .unwrap();
        let StateDomainSpec::PagedAttention(domain) = &invocation.domains[0] else {
            panic!("paged attention");
        };
        assert_eq!(domain.layers.len(), 40);
        assert!(domain.layers.iter().all(|layer| layer.kv_heads == 4
            && layer.key_head_dim == 128
            && layer.value_head_dim == 128));
        assert_eq!(domain.header.scope, StateScope::Invocation);
        // 40 layers x 4 KV heads x (128 + 128) x 2 bytes x 16,384 tokens.
        assert_eq!(
            paged_invocation_bytes(&invocation.domains[0], 16_384).unwrap(),
            40 * 4 * 256 * 2 * 16_384
        );

        let stage = atomic_stage(0);
        let spec = granite_speech5_physical_state_spec(
            &[std::slice::from_ref(&stage)],
            invocation,
            config.text_config.max_position_embeddings,
        )
        .unwrap();
        assert_eq!(spec.descriptor.retained, RetainedStateCapability::Stateless);
    }

    #[test]
    fn only_single_invocation_graphs_are_accepted() {
        let config = published_config();
        let invocation =
            granite_speech5_invocation_contract(&config, DType::F32, default_kv_page_size())
                .unwrap();
        let mut sequence = atomic_stage(0);
        sequence.selector = StageWorkSelector::SequenceDecode;
        assert!(granite_speech5_physical_state_spec(
            &[std::slice::from_ref(&sequence)],
            invocation.clone(),
            16_384
        )
        .is_err());
        assert!(granite_speech5_physical_state_spec(&[], invocation.clone(), 16_384).is_err());
        // The scalar adapter's stage selects any work.
        let mut scalar = atomic_stage(0);
        scalar.selector = StageWorkSelector::Any;
        assert!(granite_speech5_physical_state_spec(
            &[std::slice::from_ref(&scalar)],
            invocation,
            16_384
        )
        .is_ok());
    }

    #[test]
    fn artifacts_require_each_releases_weights_and_the_loader_configs() {
        let dir = std::env::temp_dir().join(format!("granite-speech5-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        for file in CONFIG_ARTIFACTS {
            std::fs::write(dir.join(file), b"{}").unwrap();
        }
        let safetensors = ModelVariant::GraniteSpeech502B;
        let gguf = ModelVariant::GraniteSpeech502BGguf;
        for (variant, expected) in [(safetensors, "model.safetensors"), (gguf, "Q8_0.gguf")] {
            let missing = ensure_granite_speech5_artifacts(&dir, variant)
                .unwrap_err()
                .to_string();
            assert!(missing.contains(expected), "{missing}");
        }
        std::fs::write(dir.join(WEIGHTS_FILE), b"").unwrap();
        assert_eq!(
            ensure_granite_speech5_artifacts(&dir, safetensors).unwrap(),
            GraniteSpeech5Weights::Safetensors(dir.join(WEIGHTS_FILE))
        );
        // The GGUF release needs both of its files.
        std::fs::write(dir.join(GRANITE_SPEECH5_GGUF_LM_FILE), b"").unwrap();
        let missing = ensure_granite_speech5_artifacts(&dir, gguf)
            .unwrap_err()
            .to_string();
        assert!(
            missing.contains(GRANITE_SPEECH5_GGUF_MMPROJ_FILE),
            "{missing}"
        );
        std::fs::write(dir.join(GRANITE_SPEECH5_GGUF_MMPROJ_FILE), b"").unwrap();
        assert_eq!(
            ensure_granite_speech5_artifacts(&dir, gguf)
                .unwrap()
                .files(),
            vec![
                dir.join(GRANITE_SPEECH5_GGUF_LM_FILE),
                dir.join(GRANITE_SPEECH5_GGUF_MMPROJ_FILE)
            ]
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn auto_budget_covers_dense_scripts_and_grows_with_duration() {
        assert_eq!(granite_speech5_auto_max_tokens_for_duration(4.0), 300);
        assert_eq!(granite_speech5_auto_max_tokens_for_duration(30.0), 300);
        assert_eq!(granite_speech5_auto_max_tokens_for_duration(120.0), 1200);
        assert_eq!(granite_speech5_auto_max_tokens_for_duration(f32::NAN), 300);
        // A full invocation of audio and its budget fit the 16k context.
        let audio_tokens =
            (GRANITE_SPEECH5_MAX_AUDIO_SECONDS * GRANITE_SPEECH5_AUDIO_TOKENS_PER_SECOND) as usize;
        assert!(audio_tokens + GRANITE_SPEECH5_AUTO_MAX_TOKENS + 64 < 16_384);
    }

    #[test]
    fn request_options_select_the_task_and_validate_before_admission() {
        let translate = GraniteSpeech5RequestOptions {
            language: Some("en".into()),
            translate_to: Some("ja".into()),
            keywords: vec!["Izwi".into()],
            ..Default::default()
        };
        let options = translate.prompt_options().unwrap();
        assert_eq!(
            options.task,
            GraniteSpeech5Task::Translate {
                target: GraniteSpeech5Language::Japanese
            }
        );
        assert!(validate_granite_speech5_request(&translate).is_ok());

        for invalid in [
            GraniteSpeech5RequestOptions {
                language: Some("ko".into()),
                ..Default::default()
            },
            GraniteSpeech5RequestOptions {
                language: Some("fr".into()),
                translate_to: Some("de".into()),
                ..Default::default()
            },
            GraniteSpeech5RequestOptions {
                keywords: vec!["<|audio|>".into()],
                ..Default::default()
            },
            GraniteSpeech5RequestOptions {
                translate_to: Some("en".into()),
                prompt: Some("transcribe the speech to text.".into()),
                ..Default::default()
            },
        ] {
            assert!(
                validate_granite_speech5_request(&invalid).is_err(),
                "{invalid:?}"
            );
        }
        // Blank values mean "not set".
        assert_eq!(
            GraniteSpeech5RequestOptions {
                translate_to: Some("  ".into()),
                ..Default::default()
            }
            .prompt_options()
            .unwrap()
            .task,
            GraniteSpeech5Task::Transcribe
        );
    }
}
