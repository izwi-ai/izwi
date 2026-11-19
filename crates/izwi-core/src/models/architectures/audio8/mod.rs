//! Audio8 TTS (`Edge0/Audio8-TTS-Realistic-Air`): a DualAR model with a
//! 44.1 kHz codec.
//!
//! A slow transformer emits one semantic token per codec frame and a fast
//! transformer the frame's ten codebooks; the codec turns frames into audio
//! and a reference clip into frames. With a reference clip and its
//! transcript the model speaks in that voice; with none it speaks in a voice
//! of its own, which the request's seed makes repeatable.
//!
//! The family is its own: its model code started as a copy of the Fish S2
//! family's, whose design the checkpoint follows, and changes wherever the
//! checkpoint does (see `docs/dev/audio8.md`). Sharing code with that
//! family would let a change for one silently alter the other, so nothing
//! here imports another family: it may use `kernels`, `models::shared`,
//! `kv`, `backends`, `engine` and `audio`. The isolation test at the bottom
//! of this file holds that line.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use candle_core::{DType, IndexOp, Tensor};
use serde::Serialize;

use sampling::Audio8SamplingDistribution;

use crate::backends::{DTypeSelection, DeviceProfile};
use crate::catalog::ModelVariant;
use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::kernels::numeric_check::{
    load_with_precision_fallback, run_numeric_self_check, xorshift_values, NumericSelfCheckReport,
    PrecisionLoad, F16_SAFE_MAX_ABS,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

pub mod artifacts;
mod batch;
mod budget;
pub mod codec;
pub mod config;
pub mod contracts;
pub mod dac;
pub mod fast;
mod physical;
mod reference_cache;
mod retained;
mod rotary;
mod sampling;
pub mod slow;
pub mod tokenizer;
pub mod weights;

pub use artifacts::Audio8ArtifactManifest;
pub use budget::auto_frame_budget;
pub use codec::{Audio8CodecArtifact, Audio8CodecSupport};
pub use config::{Audio8AudioDecoderConfig, Audio8Config, Audio8TextConfig};
pub use contracts::{
    build_semantic_allowed_mask, remap_audio8_key, semantic_code_from_token_id, semantic_token_id,
    Audio8DacContract, Audio8PromptTensorShape,
};
pub use dac::{Audio8DacConfig, Audio8DacDecoder};
pub(crate) use fast::audio8_fast_graph_device_bytes;
pub use fast::{Audio8FastConfig, Audio8FastDecoder, Audio8GeneratedFrame, Audio8Sampler};
pub(crate) use physical::{Audio8PhysicalStateSpec, AUDIO8_SLOW_STATE_GROUP};
#[allow(unused_imports)]
pub(crate) use retained::{
    Audio8PreparationMemory, Audio8PreparedArtifact, Audio8RetainedCheckpoint, Audio8RetainedState,
    Audio8RetainedStep,
};
pub use slow::{Audio8SlowConfig, Audio8SlowOutput, Audio8SlowTransformer};
pub use tokenizer::{
    Audio8ConditioningPrompt, Audio8PromptTokenizer, Audio8SpecialTokens, Audio8VqCodes,
};
pub use weights::{Audio8TensorSpec, Audio8WeightIndex, Audio8Weights};

pub(crate) use reference_cache::AUDIO8_REFERENCE_CACHE_BYTES;

pub(crate) const AUDIO8_TTS_PREPARATION_STAGE: &str = "tts.prepare.audio8";
pub(crate) const AUDIO8_TTS_PREFILL_STAGE: &str = "tts.prefill.audio8";
pub(crate) const AUDIO8_TTS_DECODE_STAGE: &str = "tts.decode.audio8";
pub(crate) const AUDIO8_AUDIO_CHUNK_FRAMES: usize = 16;
pub(crate) const AUDIO8_FIRST_AUDIO_FRAMES: usize = 4;
pub(crate) const AUDIO8_TTS_AUDIO_DECODE_STAGE: &str = "tts.audio_decode.audio8";
pub(crate) const AUDIO8_TTS_FINALIZE_STAGE: &str = "tts.codec.audio8.scalar";

pub struct Audio8TtsModel {
    model_identity: u64,
    artifact_fingerprint: Option<String>,
    variant: ModelVariant,
    config: Audio8Config,
    artifacts: Audio8ArtifactManifest,
    codec: Audio8CodecArtifact,
    runtime: Option<Audio8NativeRuntime>,
    reference_cache: reference_cache::ReferenceCodeCache,
}

struct Audio8NativeRuntime {
    tokenizer: Audio8PromptTokenizer,
    slow: Audio8SlowTransformer,
    fast: Audio8FastDecoder,
    dac: Audio8DacDecoder,
    semantic_allowed_mask: Vec<bool>,
    /// The semantic masks on the device, built at the first batched draw.
    semantic_masks: std::sync::OnceLock<Audio8SemanticMasks>,
    dtype: DType,
    /// The transformers' dtype after the load-time check, and why.
    precision: PrecisionLoad,
}

#[derive(Debug, Clone)]
pub struct Audio8Reference {
    pub audio_samples: Vec<f32>,
    pub sample_rate: u32,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Audio8GenerationParams {
    pub max_frames: usize,
    pub temperature: f32,
    pub top_p: f32,
    /// Zero disables the top-k cap. The checkpoint's default is 50.
    pub top_k: usize,
    /// Reproducible Rust sampling seed; fast-codebook sampling uses seed + 1.
    pub seed: u64,
    /// Retry recent semantic repeats using the reference's high-temperature policy.
    /// A zero temperature remains greedy and bypasses this retry.
    pub repetition_aware: bool,
}

#[derive(Debug, Clone)]
pub struct Audio8GenerationOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub frames_generated: usize,
    pub diagnostics: Audio8TtsGenerationDiagnostics,
}

#[derive(Debug, Clone, Serialize)]
pub struct Audio8TtsDiagnostics {
    pub model_family: &'static str,
    pub variant: String,
    pub native_runtime_loaded: bool,
    pub sample_rate: u32,
    pub num_codebooks: usize,
    pub codebook_size: usize,
    pub semantic_start_token_id: u32,
    pub semantic_end_token_id: u32,
    pub max_seq_len: usize,
    pub codec_support: &'static str,
    /// The slow/fast transformers' load precision and self-check;
    /// `None` without the native runtime.
    pub precision: Option<serde_json::Value>,
    /// The fast AR's CUDA frame graphs: why they are off, or their captures,
    /// replays, fallbacks and graphed and eager frames; `None` without the
    /// native runtime.
    pub fast_graphs: Option<serde_json::Value>,
}

/// Model-authored terminal reason. Exhausting a budget is incomplete speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Audio8StopReason {
    #[serde(rename = "im_end")]
    EndOfSpeech,
    #[serde(rename = "max_frames")]
    FrameLimit,
}

impl Audio8StopReason {
    pub(crate) fn require_complete(self) -> Result<()> {
        match self {
            Self::EndOfSpeech => Ok(()),
            Self::FrameLimit => Err(Error::InferenceError(
                "Audio8 generation incomplete: frame_limit reached before end of speech".into(),
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Audio8TtsGenerationDiagnostics {
    pub model_family: &'static str,
    pub sample_rate: u32,
    pub prompt_tokens: usize,
    pub max_frames: usize,
    pub frames_generated: usize,
    pub stop_reason: Audio8StopReason,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    pub repetition_aware: bool,
    pub reference_encode_ms: f32,
    pub prompt_build_ms: f32,
    pub slow_prefill_ms: f32,
    pub ar_decode_ms: f32,
    pub dac_decode_ms: f32,
    pub total_model_ms: f32,
}

const AUDIO8_DTYPE_ENV: &str = "IZWI_AUDIO8_DTYPE";
const RAS_WIN_SIZE: usize = 10;
const RAS_HIGH_TEMP: f32 = 1.0;
const RAS_HIGH_TOP_P: f32 = 0.9;
static NEXT_AUDIO8_MODEL_IDENTITY: AtomicU64 = AtomicU64::new(1);

impl Audio8TtsModel {
    /// Metadata-only fixture for request preparation and admission tests.
    /// Native inference remains unavailable, as for `load_metadata`.
    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        Self {
            model_identity: 1,
            artifact_fingerprint: None,
            variant: ModelVariant::Audio8TtsRealisticAir,
            config: config::current_config(),
            artifacts: Audio8ArtifactManifest {
                model_dir: Default::default(),
                shard_files: Vec::new(),
                tensor_count: 0,
                slow_tensor_count: 0,
                fast_tensor_count: 0,
                codec_path: Default::default(),
            },
            codec: Audio8CodecArtifact {
                path: Default::default(),
                support: codec::Audio8CodecSupport::NativePthStateDict,
            },
            runtime: None,
            reference_cache: Default::default(),
        }
    }

    pub fn load_metadata(model_dir: &Path, variant: ModelVariant) -> Result<Self> {
        if variant != ModelVariant::Audio8TtsRealisticAir {
            return Err(Error::InvalidInput(format!(
                "Unsupported Audio8 TTS variant: {variant}"
            )));
        }
        let config = Audio8Config::load(model_dir)?;
        let artifacts = Audio8ArtifactManifest::load(model_dir)?;
        let codec = Audio8CodecArtifact::load(model_dir)?;
        Ok(Self {
            model_identity: next_audio8_model_identity()?,
            artifact_fingerprint: None,
            variant,
            config,
            artifacts,
            codec,
            runtime: None,
            reference_cache: Default::default(),
        })
    }

    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        let mut model = Self::load_metadata(model_dir, variant)?;
        model.artifact_fingerprint = Some(model.artifacts.content_fingerprint()?);
        model.runtime = Some(Audio8NativeRuntime::load(
            model_dir,
            &model.config,
            &model.codec,
            device,
        )?);
        Ok(model)
    }

    pub fn artifact_fingerprint(&self) -> Option<&str> {
        self.artifact_fingerprint.as_deref()
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    pub fn config(&self) -> &Audio8Config {
        &self.config
    }

    pub fn artifacts(&self) -> &Audio8ArtifactManifest {
        &self.artifacts
    }

    pub fn codec(&self) -> &Audio8CodecArtifact {
        &self.codec
    }

    pub fn diagnostics(&self) -> Audio8TtsDiagnostics {
        let sample_rate = self
            .runtime
            .as_ref()
            .map(|runtime| runtime.dac.config().sample_rate)
            .unwrap_or_else(|| Audio8DacConfig::current().sample_rate);
        Audio8TtsDiagnostics {
            model_family: "audio8_tts",
            variant: self.variant.to_string(),
            native_runtime_loaded: self.runtime.is_some(),
            sample_rate,
            num_codebooks: self.config.num_codebooks,
            codebook_size: self.config.codebook_size,
            semantic_start_token_id: self.config.semantic_start_token_id,
            semantic_end_token_id: self.config.semantic_end_token_id,
            max_seq_len: self.config.max_seq_len,
            codec_support: audio8_codec_support_name(self.codec.support),
            precision: self
                .runtime
                .as_ref()
                .map(|runtime| runtime.precision.diagnostics()),
            fast_graphs: self
                .runtime
                .as_ref()
                .map(|runtime| runtime.fast.graphs_diagnostics()),
        }
    }

    pub fn generate_speech(
        &self,
        _text: &str,
        _reference: Option<Audio8Reference>,
        _params: Audio8GenerationParams,
    ) -> Result<Audio8GenerationOutput> {
        Err(Error::InferenceError(
            "Audio8 generation requires lifecycle-owned slow and fast physical page leases"
                .to_string(),
        ))
    }

    /// Bytes of one retained state element at the precision the model runs.
    pub(crate) fn state_element_bytes(&self) -> usize {
        self.runtime
            .as_ref()
            .map_or(4, |runtime| runtime.dtype.size_in_bytes())
    }

    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<Audio8PhysicalStateSpec> {
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            Error::ModelLoadError(
                "Audio8 TTS Realistic Air metadata was loaded without native inference modules"
                    .to_string(),
            )
        })?;
        physical::audio8_physical_state_spec(&self.config, runtime.dtype, stage_graphs)
    }

    pub(crate) fn generate_speech_physical(
        &self,
        text: &str,
        reference: Option<Audio8Reference>,
        params: Audio8GenerationParams,
        slow_cache: &mut PhysicalPagedKvCache,
        fast_cache: &mut PhysicalPagedKvCache,
    ) -> Result<Audio8GenerationOutput> {
        self.codec.ensure_native_supported()?;
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            Error::ModelLoadError(
                "Audio8 TTS Realistic Air metadata was loaded without native inference modules"
                    .to_string(),
            )
        })?;
        runtime.generate_speech_physical(
            &self.config,
            text,
            reference,
            params,
            slow_cache,
            fast_cache,
        )
    }
}

/// Slow-stack positions the load-time self-check runs.
const AUDIO8_PROBE_TOKENS: usize = 16;

/// The load-time self-check : see
/// [`audio8_numeric_probe`]. Every probed tensor must be finite and below
/// the F16-safe range.
fn audio8_numeric_self_check(
    slow: &Audio8SlowTransformer,
    fast: &Audio8FastDecoder,
    device: &candle_core::Device,
) -> NumericSelfCheckReport {
    run_numeric_self_check("Audio8", F16_SAFE_MAX_ABS, || {
        audio8_numeric_probe(slow, fast, device)
    })
}

/// Xorshift text token ids (below the semantic range) for
/// [`AUDIO8_PROBE_TOKENS`] positions through every slow layer, on a
/// scratch cache in the weights' dtype, and the semantic head; then one
/// fast-codebook frame from the last slow hidden state with an xorshift
/// semantic code. Returns the slow hidden states and logits and the fast
/// logits of every depth.
fn audio8_numeric_probe(
    slow: &Audio8SlowTransformer,
    fast: &Audio8FastDecoder,
    device: &candle_core::Device,
) -> Result<Vec<(String, Tensor)>> {
    let cfg = slow.config();
    let text_tokens = cfg.semantic_start_token_id.max(1);
    let draws = xorshift_values(AUDIO8_PROBE_TOKENS + 1, 0xF15A, 1.0);
    let pick = |draw: f32, limit: u32| ((draw.abs() * limit as f32) as u32).min(limit - 1);
    let ids = draws[..AUDIO8_PROBE_TOKENS]
        .iter()
        .map(|draw| pick(*draw, text_tokens))
        .collect::<Vec<_>>();
    let mut values = vec![ids];
    values.extend((0..cfg.num_codebooks).map(|_| vec![0u32; AUDIO8_PROBE_TOKENS]));
    let prompt = tokenizer::Audio8ConditioningPrompt {
        values,
        vq_mask: vec![false; AUDIO8_PROBE_TOKENS],
        prompt_length: AUDIO8_PROBE_TOKENS,
    };
    let embeds = slow.embed_prompt(&prompt)?;
    let dtype = embeds.dtype();
    let mut slow_cache = PhysicalPagedKvCache::scratch_for_probe(
        device,
        dtype,
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.head_dim,
        AUDIO8_PROBE_TOKENS,
    )?;
    let output = slow.forward_embeds(&embeds, 0, &mut slow_cache, true)?;
    let fast_cfg = fast.config();
    let mut fast_cache = PhysicalPagedKvCache::scratch_for_probe(
        device,
        dtype,
        fast_cfg.num_hidden_layers,
        fast_cfg.num_key_value_heads,
        fast_cfg.head_dim,
        fast_cfg.num_codebooks,
    )?;
    let last = output.hidden_states.narrow(1, AUDIO8_PROBE_TOKENS - 1, 1)?;
    let semantic_code = pick(draws[AUDIO8_PROBE_TOKENS], fast_cfg.codebook_size as u32);
    let fast_logits = fast.numeric_probe(&last, semantic_code, &mut fast_cache)?;
    Ok(vec![
        ("slow_hidden".to_string(), output.hidden_states),
        ("slow_logits".to_string(), output.logits),
        ("fast_logits".to_string(), fast_logits),
    ])
}

fn next_audio8_model_identity() -> Result<u64> {
    NEXT_AUDIO8_MODEL_IDENTITY
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| Error::ModelLoadError("Audio8 model identity space exhausted".into()))
}

impl Audio8NativeRuntime {
    fn load(
        model_dir: &Path,
        config: &Audio8Config,
        codec: &Audio8CodecArtifact,
        device: DeviceProfile,
    ) -> Result<Self> {
        codec.ensure_native_supported()?;
        let tokenizer = Audio8PromptTokenizer::load(model_dir, config)?;
        let dtype_override = std::env::var(AUDIO8_DTYPE_ENV).ok();
        let selection = DTypeSelection {
            dtype: Audio8Weights::resolve_dtype(model_dir, &device, dtype_override.as_deref())?,
            reason: match dtype_override.as_deref().map(str::trim) {
                Some(raw) if !raw.is_empty() => format!("{AUDIO8_DTYPE_ENV}={raw}").into(),
                _ => format!("Audio8 {:?} dtype policy", device.kind).into(),
            },
        };
        let eos = tokenizer.specials().eos;
        // On Metal a dtype below F32 must keep the trunk's activations finite
        // and in range (norm gains can be large), or the slow and fast
        // transformers reload in F32.
        let ((slow, mut fast, device), precision) = load_with_precision_fallback(
            "Audio8",
            &selection,
            device.kind.is_metal(),
            DType::F32,
            |dtype| -> Result<_> {
                let weights = Audio8Weights::load_with_dtype(model_dir, device.clone(), dtype)?;
                let mut slow = Audio8SlowTransformer::load(
                    Audio8SlowConfig::from_config(config)?,
                    weights.var_builder(),
                )?;
                slow.configure_semantic_head(eos)?;
                let fast = Audio8FastDecoder::load(
                    Audio8FastConfig::from_config(config)?,
                    weights.var_builder(),
                )?;
                Ok((slow, fast, weights.device().clone()))
            },
            |(slow, fast, device)| audio8_numeric_self_check(slow, fast, device),
        )?;
        tracing::info!(
            "Audio8 precision: {:?} ({})",
            precision.loaded,
            precision.reason
        );
        // A fast frame's input is the slow transformer's hidden state.
        fast.resolve_graphs(precision.loaded);
        let codec_weights = codec.load_weights(&device, DType::F32)?;
        let dac = Audio8DacDecoder::load(Audio8DacConfig::current(), codec_weights.var_builder())?;
        let semantic_allowed_mask = slow.semantic_allowed_mask(eos)?;

        Ok(Self {
            tokenizer,
            slow,
            fast,
            dac,
            semantic_allowed_mask,
            semantic_masks: std::sync::OnceLock::new(),
            dtype: precision.loaded,
            precision,
        })
    }

    fn generate_speech_physical(
        &self,
        config: &Audio8Config,
        text: &str,
        reference: Option<Audio8Reference>,
        params: Audio8GenerationParams,
        slow_cache: &mut PhysicalPagedKvCache,
        fast_cache: &mut PhysicalPagedKvCache,
    ) -> Result<Audio8GenerationOutput> {
        if text.trim().is_empty() {
            return Err(Error::InvalidInput(
                "Audio8 TTS text input cannot be empty".to_string(),
            ));
        }
        if let Some(reference) = &reference {
            if reference.text.trim().is_empty() {
                return Err(Error::InvalidInput(
                    "Audio8 TTS reference_text cannot be empty".to_string(),
                ));
            }
            if reference.audio_samples.is_empty() {
                return Err(Error::InvalidInput(
                    "Audio8 TTS reference_audio cannot be empty".to_string(),
                ));
            }
        }
        params.validate()?;

        let total_started = Instant::now();
        let started = Instant::now();
        let reference_codes = reference
            .as_ref()
            .map(|reference| {
                self.dac
                    .encode_reference_audio(&reference.audio_samples, reference.sample_rate)
            })
            .transpose()?;
        let reference_encode_ms = elapsed_ms(started);

        let started = Instant::now();
        let prompt = self.tokenizer.build_speech_prompt(
            config,
            reference
                .as_ref()
                .map(|reference| reference.text.as_str())
                .zip(reference_codes),
            text,
        )?;
        let prompt_build_ms = elapsed_ms(started);
        if prompt.prompt_length >= config.max_seq_len {
            return Err(Error::InvalidInput(format!(
                "Audio8 prompt length {} exceeds max_seq_len {}",
                prompt.prompt_length, config.max_seq_len
            )));
        }

        let max_frames = effective_frame_budget(
            prompt.prompt_length,
            config.max_seq_len,
            slow_cache.capacity_tokens(),
            params.max_frames,
        )?;

        let mut semantic_sampler = Audio8SemanticSampler::from_params(&params);
        let mut fast_sampler = Audio8Sampler::with_top_k(
            params.temperature,
            params.top_p,
            params.top_k,
            params.seed.wrapping_add(1),
        );
        let im_end_token_id = self.tokenizer.specials().eos;

        if slow_cache.context_len() != 0 {
            return Err(Error::InvalidInput(format!(
                "Audio8 slow invocation cache must begin empty, got context {}",
                slow_cache.context_len()
            )));
        }
        let started = Instant::now();
        let mut slow_output = self.slow.forward_prompt(&prompt, slow_cache, false)?;
        let slow_prefill_ms = elapsed_ms(started);
        let mut generated_codebooks = vec![Vec::new(); config.num_codebooks];
        let mut recent_semantic_tokens = Vec::with_capacity(RAS_WIN_SIZE);
        let mut stop_reason = Audio8StopReason::FrameLimit;

        let started = Instant::now();
        for frame_index in 0..max_frames {
            let allow_im_end = generated_codebooks
                .first()
                .map(|codebook| !codebook.is_empty())
                .unwrap_or(false);
            let semantic_index = sample_semantic_token(
                &slow_output.logits,
                &self.semantic_allowed_mask,
                self.slow.eos_logit_index(im_end_token_id),
                allow_im_end,
                &recent_semantic_tokens,
                &mut semantic_sampler,
            )?;
            let semantic_token_id = self.slow.token_id_from_logit(semantic_index)?;
            if semantic_token_id == im_end_token_id {
                stop_reason = Audio8StopReason::EndOfSpeech;
                break;
            }

            let frame = self.fast.generate_frame(
                semantic_token_id,
                &slow_output.hidden_states,
                &mut fast_sampler,
                fast_cache,
            )?;
            append_generated_frame(&mut generated_codebooks, &frame)?;

            retained::record_recent_semantic(
                &mut recent_semantic_tokens,
                semantic_index,
                frame_index + 1,
            );

            if frame_index + 1 == max_frames {
                break;
            }
            let frame_prompt = generated_frame_prompt(config.num_codebooks, &frame)?;
            let frame_embeds = self.slow.embed_prompt(&frame_prompt)?;
            let start_pos = slow_cache.context_len();
            slow_output = self
                .slow
                .forward_embeds(&frame_embeds, start_pos, slow_cache, false)?;
        }
        let ar_decode_ms = elapsed_ms(started);
        stop_reason.require_complete()?;

        let frames_generated = generated_codebooks.first().map(Vec::len).unwrap_or(0);
        if frames_generated == 0 {
            return Err(Error::InferenceError(
                "Audio8 generation produced no audio frames".to_string(),
            ));
        }

        let started = Instant::now();
        let samples = self.dac.decode_vq_codes(&Audio8VqCodes {
            codebooks: generated_codebooks,
        })?;
        let dac_decode_ms = elapsed_ms(started);
        if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
            return Err(Error::InferenceError(
                "Audio8 DAC produced empty or non-finite audio".to_string(),
            ));
        }

        let sample_rate = self.dac.config().sample_rate;
        Ok(Audio8GenerationOutput {
            samples,
            sample_rate,
            frames_generated,
            diagnostics: Audio8TtsGenerationDiagnostics {
                model_family: "audio8_tts",
                sample_rate,
                prompt_tokens: prompt.prompt_length,
                max_frames,
                frames_generated,
                stop_reason,
                temperature: params.temperature,
                top_p: params.top_p,
                top_k: params.top_k,
                seed: params.seed,
                repetition_aware: params.repetition_aware,
                reference_encode_ms,
                prompt_build_ms,
                slow_prefill_ms,
                ar_decode_ms,
                dac_decode_ms,
                total_model_ms: elapsed_ms(total_started),
            },
        })
    }
}

impl Default for Audio8GenerationParams {
    fn default() -> Self {
        Self {
            max_frames: ModelVariant::AUDIO8_MAX_OUTPUT_FRAMES,
            // The checkpoint's `generation_config.json`.
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            seed: 0,
            repetition_aware: true,
        }
    }
}

impl Audio8GenerationParams {
    pub fn validate(&self) -> Result<()> {
        if self.max_frames == 0 {
            return Err(Error::InvalidInput(
                "Audio8 max_frames must be greater than zero".into(),
            ));
        }
        sampling::validate_policy(self.temperature, self.top_p)
    }
}

#[derive(Clone)]
struct Audio8SemanticSampler {
    sampler: Audio8Sampler,
    repetition_aware: bool,
}

impl Audio8SemanticSampler {
    fn from_params(params: &Audio8GenerationParams) -> Self {
        Self {
            sampler: Audio8Sampler::with_top_k(
                params.temperature,
                params.top_p,
                params.top_k,
                params.seed,
            ),
            repetition_aware: params.repetition_aware,
        }
    }

    #[cfg(test)]
    fn new(temperature: f32, top_p: f32, seed: u64) -> Self {
        Self::from_params(&Audio8GenerationParams {
            temperature,
            top_p,
            seed,
            ..Default::default()
        })
    }
}

impl Audio8SemanticSampler {
    /// The candidates' top-p a distribution must cover: the request's, or
    /// the repetition-aware retry's when it is higher.
    fn required_top_p(&self) -> f32 {
        if self.repetition_aware && self.sampler.temperature > 0.0 {
            self.sampler.top_p.max(RAS_HIGH_TOP_P)
        } else {
            self.sampler.top_p
        }
    }

    /// One semantic token from `distribution`, with the repetition-aware
    /// retry over the last [`RAS_WIN_SIZE`] tokens of `previous`.
    fn draw(
        &mut self,
        distribution: &Audio8SamplingDistribution,
        im_end_token_id: u32,
        previous: &[u32],
    ) -> Result<u32> {
        let normal = self.sampler.sample(distribution)?;
        if self.repetition_aware && self.sampler.temperature > 0.0 {
            // The reference draws both policies each frame, then chooses the retry
            // only when a semantic token repeats in the last ten frames. RNG
            // algorithms differ from Torch; this records and replays the
            // native Rust policy.
            let high =
                self.sampler
                    .sample_with_policy(distribution, RAS_HIGH_TEMP, RAS_HIGH_TOP_P)?;
            let recent = &previous[previous.len().saturating_sub(RAS_WIN_SIZE)..];
            if normal != im_end_token_id && recent.contains(&normal) {
                return Ok(high);
            }
        }
        Ok(normal)
    }
}

/// The semantic masks as one `[2, vocab]` U8 device tensor (row 0 the
/// semantic mask, row 1 the first frame's, without `<|im_end|>`), uploaded
/// once per model, with the first frame's host mask and both masks' allowed
/// counts.
struct Audio8SemanticMasks {
    device: Tensor,
    first_frame: Vec<bool>,
    allowed: [usize; 2],
}

impl Audio8NativeRuntime {
    fn semantic_masks(
        &self,
        device: &candle_core::Device,
        im_end_token_id: u32,
    ) -> Result<&Audio8SemanticMasks> {
        if let Some(masks) = self.semantic_masks.get() {
            return Ok(masks);
        }
        let mut first_frame = self.semantic_allowed_mask.clone();
        if let Some(allowed) = first_frame.get_mut(im_end_token_id as usize) {
            *allowed = false;
        }
        let values = self
            .semantic_allowed_mask
            .iter()
            .chain(&first_frame)
            .map(|allowed| u8::from(*allowed))
            .collect::<Vec<_>>();
        let count = |mask: &[bool]| mask.iter().filter(|allowed| **allowed).count();
        let masks = Audio8SemanticMasks {
            device: Tensor::from_vec(values, (2, first_frame.len()), device)?,
            allowed: [count(&self.semantic_allowed_mask), count(&first_frame)],
            first_frame,
        };
        let _ = self.semantic_masks.set(masks);
        self.semantic_masks
            .get()
            .ok_or_else(|| Error::InferenceError("Audio8 semantic masks were not kept".into()))
    }
}

/// One row of a step's semantic draw.
struct Audio8SemanticRow<'a> {
    logits: &'a Tensor,
    allow_im_end: bool,
    previous: &'a [u32],
    sampler: &'a mut Audio8SemanticSampler,
}

/// `IZWI_AUDIO8_BATCHED_SAMPLING` (default on): draw a step's semantic and
/// fast codes for every row from one device pass and one readback each.
fn audio8_batched_sampling_enabled() -> bool {
    static ENABLED: crate::env::ReadOnce<bool> =
        crate::env::ReadOnce::new(|| crate::env::flag_or("IZWI_AUDIO8_BATCHED_SAMPLING", true));
    ENABLED.get()
}

/// Whether semantic rows draw from batched device candidates: CUDA and
/// Metal, where the per-row path planned device candidates too.
fn semantic_rows_on_device(logits: &Tensor) -> bool {
    (logits.device().is_cuda() || logits.device().is_metal()) && audio8_batched_sampling_enabled()
}

/// Every row's semantic token, as [`sample_semantic_token`] draws it. With
/// `device_candidates`, the rows' bounded candidates (and each raw row's
/// finiteness count) come from one device pass and one readback, under
/// masks uploaded once per model; the per-row path read back twice per row
/// and copied the vocabulary-sized mask for a first frame. Each row's
/// candidates are bit-identical to the per-row path's, and its own sampler
/// draws from them with the same host code, so every row's token and RNG
/// stream are what it gets alone.
fn sample_semantic_rows(
    runtime: &Audio8NativeRuntime,
    rows: &mut [Audio8SemanticRow<'_>],
    device_candidates: bool,
) -> Result<Vec<u32>> {
    let im_end = runtime
        .slow
        .eos_logit_index(runtime.tokenizer.specials().eos);
    if !device_candidates || rows.is_empty() {
        return rows
            .iter_mut()
            .map(|row| {
                sample_semantic_token(
                    row.logits,
                    &runtime.semantic_allowed_mask,
                    im_end,
                    row.allow_im_end,
                    row.previous,
                    row.sampler,
                )
            })
            .collect();
    }
    let vocab = runtime.semantic_allowed_mask.len();
    let mut logit_rows = Vec::with_capacity(rows.len());
    for row in rows.iter() {
        sampling::validate_policy(row.sampler.sampler.temperature, row.sampler.sampler.top_p)?;
        let logits = last_logits_row(row.logits)?;
        if logits.dim(0)? != vocab {
            return Err(Error::InferenceError(format!(
                "Audio8 semantic mask length {vocab} does not match logits length {}",
                logits.dim(0)?,
            )));
        }
        logit_rows.push(logits);
    }
    let stacked = Tensor::stack(&logit_rows, 0)?.to_dtype(DType::F32)?;
    let finite = stacked
        .eq(&stacked)?
        .broadcast_mul(&stacked.abs()?.le(f32::MAX as f64)?)?
        .to_dtype(DType::U32)?
        .sum(1)?
        .to_dtype(DType::F32)?;
    let masks = runtime.semantic_masks(stacked.device(), im_end)?;
    let selector = rows
        .iter()
        .map(|row| u32::from(!row.allow_im_end))
        .collect::<Vec<_>>();
    let row_masks = masks.device.index_select(
        &Tensor::from_vec(selector.clone(), rows.len(), stacked.device())?,
        0,
    )?;
    let (finite, candidates) = crate::models::shared::sampling::masked_nucleus_candidate_rows(
        &stacked,
        &row_masks,
        vec![finite],
    )?;
    let mut tokens = Vec::with_capacity(rows.len());
    for (index, (row, candidates)) in rows.iter_mut().zip(candidates).enumerate() {
        if finite[index] as usize != vocab {
            return Err(Audio8SamplingDistribution::non_finite_row_error(
                &logit_rows[index],
                "semantic",
            )?);
        }
        let first_frame = selector[index] as usize;
        let mask = if first_frame == 1 {
            &masks.first_frame
        } else {
            &runtime.semantic_allowed_mask
        };
        let distribution = match candidates.and_then(|candidates| {
            Audio8SamplingDistribution::from_device_candidates(
                candidates,
                masks.allowed[first_frame],
                row.sampler.sampler.top_k,
                row.sampler.required_top_p(),
            )
        }) {
            Some(distribution) => distribution,
            None => Audio8SamplingDistribution::from_host_row(
                &logit_rows[index],
                Some(mask),
                "semantic",
            )?,
        };
        tokens.push(row.sampler.draw(&distribution, im_end, row.previous)?);
    }
    Ok(tokens)
}

fn sample_semantic_token(
    logits: &Tensor,
    allowed_mask: &[bool],
    im_end_token_id: u32,
    allow_im_end: bool,
    previous_semantic_tokens: &[u32],
    sampler: &mut Audio8SemanticSampler,
) -> Result<u32> {
    sampling::validate_policy(sampler.sampler.temperature, sampler.sampler.top_p)?;
    let row = last_logits_row(logits)?;
    if row.dim(0)? != allowed_mask.len() {
        return Err(Error::InferenceError(format!(
            "Audio8 semantic mask length {} does not match logits length {}",
            allowed_mask.len(),
            row.dim(0)?,
        )));
    }
    let mut first_frame_mask;
    let sampling_mask = if allow_im_end {
        allowed_mask
    } else {
        first_frame_mask = allowed_mask.to_vec();
        if let Some(allowed) = first_frame_mask.get_mut(im_end_token_id as usize) {
            *allowed = false;
        }
        &first_frame_mask
    };
    let distribution = Audio8SamplingDistribution::from_logits(
        &row,
        Some(sampling_mask),
        sampler.sampler.top_k,
        sampler.required_top_p(),
        "semantic",
    )?;
    sampler.draw(&distribution, im_end_token_id, previous_semantic_tokens)
}

fn last_logits_row(logits: &Tensor) -> Result<Tensor> {
    match logits.rank() {
        3 => {
            let (batch, seq_len, _) = logits.dims3()?;
            if batch != 1 || seq_len == 0 {
                return Err(Error::InferenceError(format!(
                    "Audio8 semantic logits require one non-empty sequence, got {:?}",
                    logits.dims()
                )));
            }
            logits.i((0, seq_len - 1)).map_err(Error::from)
        }
        2 => logits.i(0).map_err(Error::from),
        1 => Ok(logits.clone()),
        _ => Err(Error::InferenceError(format!(
            "Audio8 expected semantic logits rank 1, 2, or 3, got {:?}",
            logits.dims()
        ))),
    }
}

fn append_generated_frame(
    generated_codebooks: &mut [Vec<u32>],
    frame: &Audio8GeneratedFrame,
) -> Result<()> {
    if frame.codebooks.len() != generated_codebooks.len() {
        return Err(Error::InferenceError(format!(
            "Audio8 fast decoder returned {} codebooks, expected {}",
            frame.codebooks.len(),
            generated_codebooks.len()
        )));
    }
    for (row, code) in generated_codebooks.iter_mut().zip(&frame.codebooks) {
        row.push(*code);
    }
    Ok(())
}

fn generated_frame_prompt(
    num_codebooks: usize,
    frame: &Audio8GeneratedFrame,
) -> Result<Audio8ConditioningPrompt> {
    if frame.codebooks.len() != num_codebooks {
        return Err(Error::InferenceError(format!(
            "Audio8 frame has {} codebooks, expected {num_codebooks}",
            frame.codebooks.len()
        )));
    }
    let mut values = vec![vec![0u32; 1]; num_codebooks + 1];
    values[0][0] = frame.semantic_token_id;
    for (idx, code) in frame.codebooks.iter().enumerate() {
        values[idx + 1][0] = *code;
    }
    Ok(Audio8ConditioningPrompt {
        values,
        vq_mask: vec![true],
        prompt_length: 1,
    })
}

pub(crate) fn effective_frame_budget(
    prompt_tokens: usize,
    model_context: usize,
    cache_capacity: usize,
    requested: usize,
) -> Result<usize> {
    let context = model_context.min(cache_capacity);
    let available = context.checked_sub(prompt_tokens).filter(|available| *available > 0)
        .ok_or_else(|| Error::ContextLengthExceeded(format!(
            "Audio8 prompt length {prompt_tokens} leaves no output room in effective context {context}",
        )))?;
    // Reserve one context position beyond the full output budget for EOS. Never
    // silently shrink the admitted output budget after exact tokenization.
    if requested == 0 || requested >= available {
        return Err(Error::ContextLengthExceeded(format!(
            "Audio8 segment does not fit effective context: prompt_tokens={prompt_tokens}, requested_frames={requested}, eos_margin=1, context={context}; split the text segment or shorten the voice reference before generating audio",
        )));
    }
    Ok(requested)
}

fn elapsed_ms(started: Instant) -> f32 {
    started.elapsed().as_secs_f32() * 1000.0
}

fn audio8_codec_support_name(support: Audio8CodecSupport) -> &'static str {
    match support {
        Audio8CodecSupport::NativePthStateDict => "native_pth_state_dict",
    }
}

#[cfg(test)]
mod reference_tests;
#[cfg(test)]
mod smoke;

#[cfg(test)]
mod streaming_smoke;

#[cfg(test)]
mod precision_tests {
    use super::*;
    use crate::kernels::numeric_check::xorshift_tensor;
    use candle_core::Device;
    use candle_nn::VarBuilder;
    use std::collections::BTreeMap;

    /// A tiny two-layer slow transformer and one-layer fast decoder with
    /// xorshift weights and every RMSNorm gain set to `norm_gain`, in `dtype`.
    fn tiny_random_parts(
        device: &Device,
        dtype: DType,
        norm_gain: f32,
    ) -> (Audio8SlowTransformer, Audio8FastDecoder) {
        let slow_cfg = Audio8SlowConfig {
            vocab_size: 32,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            max_seq_len: 32,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            codebook_size: 8,
            num_codebooks: 2,
            semantic_start_token_id: 20,
            semantic_end_token_id: 27,
            attention_qkv_bias: true,
        };
        let fast_cfg = Audio8FastConfig {
            input_hidden_size: 4,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            codebook_size: 8,
            num_codebooks: 3,
            semantic_start_token_id: 20,
            semantic_end_token_id: 27,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
        };
        let mut shapes = BTreeMap::new();
        shapes.insert("embed_tokens.weight".to_string(), vec![32, 4]);
        shapes.insert("codebook_embeddings.weight".to_string(), vec![16, 4]);
        shapes.insert("fast_project_in.weight".to_string(), vec![4, 4]);
        shapes.insert("fast_project_in.bias".to_string(), vec![4]);
        shapes.insert("fast_embeddings.weight".to_string(), vec![8, 4]);
        shapes.insert("fast_output.weight".to_string(), vec![8, 4]);
        let layer = |prefix: &str, shapes: &mut BTreeMap<String, Vec<usize>>| {
            for (name, shape) in [
                ("self_attn.qkv_proj.weight", vec![8, 4]),
                ("self_attn.o_proj.weight", vec![4, 4]),
                ("mlp.gate_proj.weight", vec![8, 4]),
                ("mlp.up_proj.weight", vec![8, 4]),
                ("mlp.down_proj.weight", vec![4, 8]),
            ] {
                shapes.insert(format!("{prefix}.{name}"), shape);
            }
        };
        layer("layers.0", &mut shapes);
        layer("layers.1", &mut shapes);
        layer("fast_layers.0", &mut shapes);
        let mut tensors = shapes
            .into_iter()
            .enumerate()
            .map(|(index, (name, shape))| {
                let tensor = xorshift_tensor(&shape, 0xF500 + index as u64, 0.5, device).unwrap();
                (name, tensor)
            })
            .collect::<std::collections::HashMap<_, _>>();
        let gain = |dim: usize| Tensor::full(norm_gain, dim, device).unwrap();
        for name in [
            "norm.weight",
            "layers.0.input_layernorm.weight",
            "layers.0.post_attention_layernorm.weight",
            "layers.1.input_layernorm.weight",
            "layers.1.post_attention_layernorm.weight",
            "fast_norm.weight",
            "fast_layers.0.input_layernorm.weight",
            "fast_layers.0.post_attention_layernorm.weight",
        ] {
            tensors.insert(name.to_string(), gain(4));
        }
        for name in [
            "layers.0.self_attn.qkv_proj.bias",
            "layers.1.self_attn.qkv_proj.bias",
        ] {
            tensors.insert(
                name.to_string(),
                xorshift_tensor(&[8], 0xB1A5, 0.1, device).unwrap(),
            );
        }
        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        let mut slow = Audio8SlowTransformer::load(slow_cfg, vb.clone()).unwrap();
        slow.configure_semantic_head(1).unwrap();
        let fast = Audio8FastDecoder::load(fast_cfg, vb).unwrap();
        (slow, fast)
    }

    /// The probe runs every slow layer and the semantic head, then a
    /// fast frame; on a healthy tiny model it passes in F16 and tracks F32.
    #[test]
    fn audio8_probe_passes_in_f16_and_tracks_f32() {
        let device = Device::Cpu;
        let report = |dtype| {
            let (slow, fast) = tiny_random_parts(&device, dtype, 1.0);
            audio8_numeric_self_check(&slow, &fast, &device)
        };
        let full = report(DType::F32);
        let half = report(DType::F16);
        assert!(full.passed, "{}", full.reason);
        assert!(half.passed, "{}", half.reason);
        let names = half
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, ["slow_hidden", "slow_logits", "fast_logits"]);
        for (half, full) in half.tensors.iter().zip(&full.tensors) {
            assert!(
                (half.max_abs - full.max_abs).abs() <= 0.05 * full.max_abs.max(1.0),
                "{}: F16 max |x| {} vs F32 {}",
                half.name,
                half.max_abs,
                full.max_abs
            );
        }
    }

    /// When large norm gains push F16 activations out of range, the
    /// self-check fails and the transformers reload in F32; healthy gains
    /// keep F16.
    #[test]
    fn audio8_self_check_falls_back_to_f32_when_f16_activations_overflow() {
        let device = Device::Cpu;
        let selection = DTypeSelection {
            dtype: DType::F16,
            reason: "test policy".into(),
        };
        for (norm_gain, expected) in [(1.0f32, DType::F16), (3.0e3, DType::F32)] {
            let (_, precision) = load_with_precision_fallback(
                "tiny Audio8",
                &selection,
                true,
                DType::F32,
                |dtype| Ok::<_, Error>(tiny_random_parts(&device, dtype, norm_gain)),
                |(slow, fast)| audio8_numeric_self_check(slow, fast, &device),
            )
            .unwrap();
            assert_eq!(precision.loaded, expected, "{}", precision.reason);
            assert_eq!(precision.fell_back(), expected == DType::F32);
        }
    }

    /// The probe runs on Metal in F16 and passes on the healthy
    /// tiny model, tracking CPU F32.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_audio8_probe_passes_in_f16() {
        let Some((metal, _lease)) =
            crate::backends::metal_test_device_or_skip("metal_audio8_probe_passes_in_f16")
        else {
            return;
        };
        let (slow, fast) = tiny_random_parts(&metal, DType::F16, 1.0);
        let report = audio8_numeric_self_check(&slow, &fast, &metal);
        assert!(report.passed, "{}", report.reason);
        let (slow, fast) = tiny_random_parts(&Device::Cpu, DType::F32, 1.0);
        let cpu = audio8_numeric_self_check(&slow, &fast, &Device::Cpu);
        for (metal, cpu) in report.tensors.iter().zip(&cpu.tensors) {
            assert!(
                (metal.max_abs - cpu.max_abs).abs() <= 0.05 * cpu.max_abs.max(1.0),
                "{}: Metal F16 {} vs CPU F32 {}",
                metal.name,
                metal.max_abs,
                cpu.max_abs
            );
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn full_output_budget_and_eos_must_fit_exact_context() {
        assert_eq!(
            super::effective_frame_budget(224, 8192, 256, 31).unwrap(),
            31
        );
        for requested in [32, 512] {
            let error = super::effective_frame_budget(224, 8192, 256, requested).unwrap_err();
            assert!(error
                .to_string()
                .contains("segment does not fit effective context"));
            assert_eq!(error.kind(), crate::error::ErrorKind::ContextLengthExceeded);
        }
        let error = super::effective_frame_budget(256, 8192, 256, 1).unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::ContextLengthExceeded);
    }

    #[test]
    fn frame_limit_is_incomplete_speech_while_eos_is_success() {
        use super::Audio8StopReason;
        Audio8StopReason::EndOfSpeech.require_complete().unwrap();
        assert!(Audio8StopReason::FrameLimit
            .require_complete()
            .unwrap_err()
            .to_string()
            .contains("generation incomplete: frame_limit"));
        assert_eq!(
            serde_json::to_string(&Audio8StopReason::EndOfSpeech).unwrap(),
            "\"im_end\""
        );
        assert_eq!(
            serde_json::to_string(&Audio8StopReason::FrameLimit).unwrap(),
            "\"max_frames\""
        );
    }

    use super::*;
    use candle_core::Device;

    #[test]
    fn generation_params_default_to_the_checkpoints_generation_config() {
        let params = Audio8GenerationParams::default();
        assert_eq!(params.max_frames, ModelVariant::AUDIO8_MAX_OUTPUT_FRAMES);
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_p, 0.9);
        assert_eq!(params.top_k, 50);
        assert!(params.repetition_aware);
    }

    #[test]
    fn semantic_sampler_obeys_allowed_mask_for_argmax() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![100.0f32, 2.0, 3.0, 4.0], (1, 1, 4), &device).unwrap();
        let allowed = vec![false, true, false, true];
        let mut sampler = Audio8SemanticSampler::new(0.0, 1.0, 0);
        let token = sample_semantic_token(&logits, &allowed, 1, true, &[], &mut sampler).unwrap();
        assert_eq!(token, 3);
    }

    #[test]
    fn semantic_sampler_masks_im_end_before_first_audio_frame() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.0f32, 100.0, 98.0, 4.0], (1, 1, 4), &device).unwrap();
        let allowed = vec![false, true, true, true];
        let mut sampler = Audio8SemanticSampler::new(0.0, 1.0, 0);
        let token = sample_semantic_token(&logits, &allowed, 1, false, &[], &mut sampler).unwrap();
        assert_eq!(token, 2);
    }

    #[test]
    fn semantic_sampler_allows_im_end_after_audio_frame() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.0f32, 100.0, 98.0, 4.0], (1, 1, 4), &device).unwrap();
        let allowed = vec![false, true, true, true];
        let mut sampler = Audio8SemanticSampler::new(0.0, 1.0, 0);
        let token = sample_semantic_token(&logits, &allowed, 1, true, &[2], &mut sampler).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn generated_frame_prompt_has_the_reference_codebook_shape() {
        let frame = Audio8GeneratedFrame {
            semantic_token_id: 42,
            codebooks: vec![4, 5, 6],
        };
        let prompt = generated_frame_prompt(3, &frame).unwrap();
        assert_eq!(prompt.prompt_length, 1);
        assert_eq!(prompt.vq_mask, vec![true]);
        assert_eq!(prompt.values, vec![vec![42], vec![4], vec![5], vec![6]]);
    }
}

#[cfg(test)]
mod isolation_tests {
    use std::path::{Path, PathBuf};

    /// No source file of the family names another architecture, by an
    /// absolute path or a relative one.
    #[test]
    fn the_family_imports_no_other_architecture() {
        let root =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/models/architectures/audio8");
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
        let mut files = Vec::new();
        sources(&root, &mut files);
        assert!(files.len() >= 10);
        // Every sibling family, so a relative `super::super::<family>::`
        // path is caught as well.
        let others = std::fs::read_dir(root.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter_map(|path| Some(path.file_stem()?.to_str()?.to_string()))
            .filter(|name| name != "mod" && name != "audio8")
            .collect::<Vec<_>>();
        assert!(others.iter().any(|name| name == "fish_s2"));
        let needle = concat!("architectures", "::");
        let own = concat!("architectures", "::", "audio8");
        for file in files {
            let source = std::fs::read_to_string(&file).unwrap();
            for (offset, _) in source.match_indices(needle) {
                assert!(
                    source[offset..].starts_with(own),
                    "{} names another architecture at byte {offset}",
                    file.display()
                );
            }
            // `super::super::` would climb into `architectures` itself.
            assert!(
                !source.contains(concat!("super::super::super::", "architectures")),
                "{}",
                file.display()
            );
            for other in &others {
                let path = format!("{other}::");
                for (offset, _) in source.match_indices(&path) {
                    let before = source[..offset].chars().next_back();
                    assert!(
                        before.is_some_and(|c| c.is_alphanumeric() || c == '_'),
                        "{} names the {other} family at byte {offset}",
                        file.display()
                    );
                }
            }
        }
    }
}
