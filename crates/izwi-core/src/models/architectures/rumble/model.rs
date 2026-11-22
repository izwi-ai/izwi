//! The loaded Rumble model: checkpoint load, one-invocation transcription
//! and the state a load publishes for its ASR graphs.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use candle_core::{DType, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use serde_json::json;
use tracing::info;

use crate::backends::DeviceProfile;
use crate::catalog::ModelVariant;
use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::kernels::buffer_pool::maybe_init_global_buffer_pool;
use crate::kernels::numeric_check::{
    load_with_precision_fallback, run_numeric_self_check, xorshift_tensor, NumericSelfCheckReport,
    PrecisionLoad, F16_SAFE_MAX_ABS,
};
use crate::kv::v2::{
    stage_graph_fingerprint, CapabilityStateDescriptorV2, CheckpointPolicy, InferenceStateContract,
    InvocationLeaseScope, InvocationStageWorkspace, InvocationStateCapacity,
    InvocationWorkspaceDomain, InvocationWorkspaceProfile, InvocationWorkspaceSet, PlacementPolicy,
    PrefixPolicy, RetainedStateCapability, StateDType, StateDomainId, StateDomainSpec, StateScope,
    WorkspaceFormula, CURRENT_INFERENCE_STATE_ABI,
};
use crate::models::architectures::qwen3::core::{Qwen3Model, Qwen3WeightLayout};
use crate::models::architectures::qwen3::decode_graph::RUMBLE_DECODE_GRAPHS_ENV;
use crate::models::shared::attention::paged::default_kv_page_size;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::memory::metal::metal_pool_for_device;
use crate::models::shared::weights::staged;
use crate::tokenizer::IncrementalDecoder;

use super::audio_tower::AudioTower;
use super::checkpoint::{
    check_rumble_census, ensure_rumble_artifacts, read_safetensors_header, rumble_tensor_plan,
    RUMBLE_AUDIO_TOWER_PREFIX, RUMBLE_LANGUAGE_MODEL_PREFIX, RUMBLE_MODEL_PREFIX,
    RUMBLE_PROJECTOR_PREFIX,
};
use super::config::{RumbleConfig, RumblePreprocessorConfig};
use super::frontend::{
    RumbleAudioFeatures, RumbleFrontend, RUMBLE_AUDIO_TOKENS_PER_SECOND, RUMBLE_MAX_AUDIO_SECONDS,
};
use super::precision::rumble_weight_precisions;
use super::prompt::{
    build_rumble_prompt, RumblePrompt, RumblePromptOptions, RumbleTranscriptStyle,
    RUMBLE_MAX_CONTEXT_TOKENS, RUMBLE_PROMPT_OVERHEAD_TOKENS,
};
use super::tokenizer::RumbleTokenizer;
use super::transcript::{split_rumble_output, RumbleOutput};

/// Decode budget of one invocation when the request names none: the
/// checkpoint's `generation_config.json`.
pub const RUMBLE_DEFAULT_MAX_NEW_TOKENS: usize = 512;
/// The longest sequence one invocation decodes: the prompt's markers and
/// context text, one 30 s window of audio tokens (cut at 95% of the window
/// by the chunk planner, so the window itself is the bound) and the decode
/// budget. The checkpoint's native 65,536-token context would size every
/// invocation's KV for requests that need about a thousand tokens.
pub const RUMBLE_SEQUENCE_TOKENS: usize = 1_536;

/// Mel frames the audio tower's load-time probe runs (two conv chunks).
const AUDIO_PROBE_FRAMES: usize = 200;
/// Token positions the decoder's load-time probe runs.
const TEXT_PROBE_TOKENS: usize = 16;

static NEXT_MODEL_IDENTITY: AtomicU64 = AtomicU64::new(1);

/// What a request asks of Rumble beyond its audio.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RumbleRequestOptions {
    /// The spoken language, when the caller knows it.
    pub language: Option<String>,
    /// Context text (names, terms) for the system turn.
    pub prompt: Option<String>,
    pub style: RumbleTranscriptStyle,
}

impl RumbleRequestOptions {
    pub fn prompt_options(&self) -> RumblePromptOptions {
        RumblePromptOptions {
            language: self.language.clone(),
            context: self.prompt.clone(),
            style: self.style,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RumbleGenerationOptions {
    pub max_new_tokens: usize,
    pub stop_token_ids: Vec<u32>,
    pub stop_sequences: Vec<String>,
}

impl Default for RumbleGenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: RUMBLE_DEFAULT_MAX_NEW_TOKENS,
            stop_token_ids: Vec::new(),
            stop_sequences: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RumbleTranscription {
    pub text: String,
    /// The forced language, or the one the model named.
    pub language: Option<String>,
    pub diagnostics: Option<serde_json::Value>,
}

/// The state a load publishes for its ASR graphs: retained decoder pages for
/// the retained sequence graph (when it is offered), and invocation pages
/// for the atomic graph.
#[derive(Debug, Clone)]
pub(crate) struct RumblePhysicalStateSpec {
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) retained: Option<InferenceStateContract>,
    /// The longest retained sequence.
    pub(crate) retained_max_tokens: Option<usize>,
    pub(crate) invocation: InferenceStateContract,
    /// Decoder tokens one invocation may occupy.
    pub(crate) sequence_tokens: usize,
}

/// One request's audio after the tower and projector: `[1, tokens, hidden]`
/// in the decoder's dtype.
pub(crate) struct RumbleAudioEmbeddings {
    pub(crate) embeddings: Tensor,
    pub(crate) upload_ms: f64,
    pub(crate) encode_ms: f64,
}

pub struct RumbleAsrModel {
    model_identity: u64,
    model_dir: PathBuf,
    device: DeviceProfile,
    config: RumbleConfig,
    tokenizer: RumbleTokenizer,
    frontend: RumbleFrontend,
    audio_tower: AudioTower,
    text_model: Qwen3Model,
    audio_precision: PrecisionLoad,
    text_precision: PrecisionLoad,
    retained_decode: bool,
}

impl RumbleAsrModel {
    pub fn load(model_dir: &Path, _variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        let weights = ensure_rumble_artifacts(model_dir)?;
        let config = RumbleConfig::load(model_dir)?;
        let preprocessor = RumblePreprocessorConfig::load(model_dir)?;
        if preprocessor.feature_size != config.audio_config.num_mel_bins {
            return Err(Error::ModelLoadError(format!(
                "Rumble preprocessor emits {} mel bins, but the audio tower takes {}",
                preprocessor.feature_size, config.audio_config.num_mel_bins
            )));
        }
        check_rumble_census(
            &rumble_tensor_plan(&config),
            &read_safetensors_header(&weights)?,
        )?;
        let tokenizer = RumbleTokenizer::load(model_dir, &config)?;
        let frontend = RumbleFrontend::new(&preprocessor)?;
        let (text_selection, audio_selection) = rumble_weight_precisions(&device)?;

        // Shared scratch pools are initialized up front so their first use
        // does not land in a prefill or decode step.
        let _ = maybe_init_global_buffer_pool(&device.device);
        let _ = metal_pool_for_device(&device.device);

        let var_builder = |dtype: DType| -> Result<VarBuilder<'static>> {
            // SAFETY: the mapped checkpoint is a read-only file.
            let vb = unsafe {
                staged::metal_synced_safetensors(
                    std::slice::from_ref(&weights),
                    dtype,
                    &device.device,
                )?
            };
            Ok(vb.pp(RUMBLE_MODEL_PREFIX))
        };
        // A lowered dtype must pass its probe, or that component alone
        // reloads in F32.
        let check = !device.kind.is_cpu();
        let audio_config = config.audio_config.clone();
        let (audio_tower, audio_precision) = load_with_precision_fallback(
            "Rumble audio tower",
            &audio_selection,
            check,
            DType::F32,
            |dtype| -> Result<AudioTower> {
                let vb = var_builder(dtype)?;
                AudioTower::load(
                    audio_config.clone(),
                    vb.pp(RUMBLE_AUDIO_TOWER_PREFIX),
                    vb.pp(RUMBLE_PROJECTOR_PREFIX),
                )
            },
            |tower| audio_self_check(tower, audio_selection.dtype, &device.device),
        )?;
        let trunk_config = config.trunk_config()?;
        let (mut text_model, text_precision) = load_with_precision_fallback(
            "Rumble decoder",
            &text_selection,
            check,
            DType::F32,
            |dtype| -> Result<Qwen3Model> {
                Qwen3Model::load_with_layout(
                    trunk_config.clone(),
                    var_builder(dtype)?,
                    Qwen3WeightLayout {
                        model_prefix: RUMBLE_LANGUAGE_MODEL_PREFIX,
                        lm_head_prefix: None,
                    },
                )
            },
            text_self_check,
        )?;
        info!(
            "Loaded Rumble ASR from {:?} on {:?} (audio {:?}: {}; decoder {:?}: {})",
            model_dir,
            device.kind,
            audio_precision.loaded,
            audio_precision.reason,
            text_precision.loaded,
            text_precision.reason
        );
        let retained_decode = super::retained::rumble_retained_decode_enabled(device.kind.into());
        // Only the retained route decodes through the trunk's batched step,
        // the one its CUDA graphs replay.
        if retained_decode {
            text_model.enable_asr_decode_graph("Rumble", RUMBLE_DECODE_GRAPHS_ENV);
        }
        Ok(Self {
            model_identity: NEXT_MODEL_IDENTITY.fetch_add(1, Ordering::Relaxed),
            model_dir: model_dir.to_path_buf(),
            device,
            config,
            tokenizer,
            frontend,
            audio_tower,
            text_model,
            audio_precision,
            text_precision,
            retained_decode,
        })
    }

    /// The decoder's CUDA decode graphs, or why this load has none.
    pub(crate) fn decode_graph_diagnostics(&self) -> serde_json::Value {
        self.text_model.decode_graph_diagnostics()
    }

    /// Whether this load offers the retained sequence route.
    pub(crate) fn retained_decode(&self) -> bool {
        self.retained_decode
    }

    /// This load with the retained route forced on or off, whatever its
    /// backend's default.
    #[cfg(test)]
    pub(crate) fn with_retained_decode(mut self, retained_decode: bool) -> Self {
        self.retained_decode = retained_decode;
        self
    }

    pub fn config(&self) -> &RumbleConfig {
        &self.config
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub(crate) fn model_identity(&self) -> u64 {
        self.model_identity
    }

    pub(crate) fn device(&self) -> &DeviceProfile {
        &self.device
    }

    pub(crate) fn tokenizer(&self) -> &RumbleTokenizer {
        &self.tokenizer
    }

    pub(crate) fn frontend(&self) -> &RumbleFrontend {
        &self.frontend
    }

    pub(crate) fn text_model(&self) -> &Qwen3Model {
        &self.text_model
    }

    pub(crate) fn audio_tower(&self) -> &AudioTower {
        &self.audio_tower
    }

    /// The decoder's loaded dtype: its KV pages and the audio embeddings it
    /// reads use it.
    pub fn text_dtype(&self) -> DType {
        self.text_precision.loaded
    }

    pub fn audio_dtype(&self) -> DType {
        self.audio_precision.loaded
    }

    pub fn max_audio_seconds_hint(&self) -> Option<f32> {
        Some(RUMBLE_MAX_AUDIO_SECONDS)
    }

    /// Decoder-prompt tokens added per second of audio.
    pub(crate) fn audio_token_rate(&self) -> f32 {
        RUMBLE_AUDIO_TOKENS_PER_SECOND
    }

    /// Decoder tokens one invocation may occupy.
    pub(crate) fn sequence_tokens(&self) -> usize {
        RUMBLE_SEQUENCE_TOKENS.min(self.config.text_config.max_position_embeddings)
    }

    /// The decoder's paged KV contract with invocation scope.
    pub(crate) fn invocation_contract(&self) -> Result<InferenceStateContract> {
        let mut contract = self.text_model.managed_inference_state_contract(
            StateDomainId::new(1),
            self.text_dtype(),
            default_kv_page_size(),
            // Audio-derived prompts are never shared between requests.
            false,
        )?;
        for domain in &mut contract.domains {
            let StateDomainSpec::PagedAttention(domain) = domain else {
                return Err(Error::ModelLoadError(
                    "Rumble decoder state must be paged attention".into(),
                ));
            };
            domain.header.scope = StateScope::Invocation;
            domain.header.prefix = PrefixPolicy::Disabled;
            domain.header.checkpoint = CheckpointPolicy::None;
        }
        for group in &mut contract.groups {
            group.prefix_shareable = false;
        }
        contract.validate()?;
        Ok(contract)
    }

    /// The state for the loaded ASR capability's stage graphs: retained
    /// decoder pages for a retained sequence graph, and for the atomic graph
    /// one paged decoder cache of [`Self::sequence_tokens`] per invocation.
    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<RumblePhysicalStateSpec> {
        let invocation = self.invocation_contract()?;
        let retained = rumble_retained_contract(invocation.clone())?;
        rumble_physical_state_spec(stage_graphs, retained, invocation, self.sequence_tokens())
    }

    /// Host features for one invocation of audio.
    pub(crate) fn prepare_features(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<RumbleAudioFeatures> {
        self.frontend.prepare(audio, sample_rate)
    }

    /// Run the audio tower and projector over one request's features: the
    /// batched pass with one row, so a request encodes the same way alone
    /// as beside others.
    pub(crate) fn audio_embeddings(
        &self,
        features: &RumbleAudioFeatures,
    ) -> Result<RumbleAudioEmbeddings> {
        let upload_started = Instant::now();
        let input = self.upload_chunk_rows(&[features])?;
        let upload_ms = elapsed_ms(upload_started);
        let encode_started = Instant::now();
        let embeddings = self
            .encode_chunk_rows(&input, &[features])?
            .pop()
            .ok_or_else(|| Error::InferenceError("Rumble audio tower lost its row".into()))?;
        Ok(RumbleAudioEmbeddings {
            embeddings,
            upload_ms,
            encode_ms: elapsed_ms(encode_started),
        })
    }

    /// Run the audio tower and projector over several requests' features in
    /// one pass: each row's `[1, audio_tokens, hidden]` embeddings at the
    /// decoder's dtype, in order. Every row's whole conv chunks go up in
    /// one upload and none reads another's frames.
    pub(crate) fn audio_embeddings_batch(
        &self,
        rows: &[&RumbleAudioFeatures],
    ) -> Result<Vec<Tensor>> {
        let input = self.upload_chunk_rows(rows)?;
        self.encode_chunk_rows(&input, rows)
    }

    /// Every row's features as one `[all chunks, chunk_frames, n_mels]`
    /// tensor at the tower's dtype. A row's features already hold whole
    /// conv chunks (the padded frames are the mel of the silence its
    /// waveform was padded with), so the rows concatenate on the host.
    fn upload_chunk_rows(&self, rows: &[&RumbleAudioFeatures]) -> Result<Tensor> {
        let chunk_frames = self.audio_tower.chunk_frames()?;
        let n_mels = self.audio_tower.mel_bins();
        let mut chunks = 0usize;
        for (row, features) in rows.iter().enumerate() {
            let row_chunks = features.mel_frames.div_ceil(chunk_frames);
            if features.mel_frames == 0
                || features.n_mels != n_mels
                || features.padded_frames != row_chunks * chunk_frames
                || features.flat.len() != features.padded_frames * n_mels
            {
                return Err(Error::InvalidInput(format!(
                    "Rumble audio row {row} holds {} of {} frames x {} mel bins; the tower takes whole {chunk_frames}-frame chunks of {n_mels} bins",
                    features.mel_frames, features.padded_frames, features.n_mels
                )));
            }
            chunks += row_chunks;
        }
        if chunks == 0 {
            return Err(Error::InvalidInput(
                "No audio features available for ASR batch".into(),
            ));
        }
        let input = if let [features] = rows {
            Tensor::from_slice(
                &features.flat,
                (chunks, chunk_frames, n_mels),
                &self.device.device,
            )?
        } else {
            let mut flat = Vec::with_capacity(chunks * chunk_frames * n_mels);
            for features in rows {
                flat.extend_from_slice(&features.flat);
            }
            Tensor::from_vec(flat, (chunks, chunk_frames, n_mels), &self.device.device)?
        };
        input.to_dtype(self.audio_dtype()).map_err(Error::from)
    }

    /// The tower and projector over [`Self::upload_chunk_rows`]' tensor:
    /// each row's embeddings at the decoder's dtype.
    fn encode_chunk_rows(
        &self,
        input: &Tensor,
        rows: &[&RumbleAudioFeatures],
    ) -> Result<Vec<Tensor>> {
        let lengths = rows
            .iter()
            .map(|features| features.mel_frames)
            .collect::<Vec<_>>();
        let encoded = self.audio_tower.forward_chunk_rows(input, &lengths)?;
        if encoded.len() != rows.len() {
            return Err(Error::InferenceError(
                "Rumble audio tower returned the wrong row count".into(),
            ));
        }
        encoded
            .into_iter()
            .zip(rows)
            .map(|(embeddings, features)| {
                if embeddings.dim(1)? != features.audio_tokens {
                    return Err(Error::InferenceError(format!(
                        "Rumble audio tower emitted {} tokens, expected {}",
                        embeddings.dim(1)?,
                        features.audio_tokens
                    )));
                }
                if embeddings.dtype() != self.text_dtype() {
                    return embeddings.to_dtype(self.text_dtype()).map_err(Error::from);
                }
                Ok(embeddings)
            })
            .collect()
    }

    pub(crate) fn build_prompt(
        &self,
        audio_tokens: usize,
        options: &RumblePromptOptions,
    ) -> Result<RumblePrompt> {
        build_rumble_prompt(&self.tokenizer, audio_tokens, options)
    }

    /// The prompt's embeddings with the audio placeholders replaced by
    /// `audio`: `[1, prompt tokens, hidden]`.
    pub(crate) fn prompt_embeddings(
        &self,
        prompt: &RumblePrompt,
        audio: &Tensor,
    ) -> Result<Tensor> {
        let (rows, tokens, _) = audio.dims3()?;
        if rows != 1 || tokens != prompt.audio_tokens {
            return Err(Error::InvalidInput(format!(
                "Rumble prompt holds {} audio placeholders, but the audio has shape {:?}",
                prompt.audio_tokens,
                audio.dims()
            )));
        }
        let ids = Tensor::from_slice(&prompt.ids, (1, prompt.ids.len()), &self.device.device)?;
        let embeds = self.text_model.embeddings(&ids)?;
        let after = prompt.audio_start + prompt.audio_tokens;
        let tail = prompt.ids.len() - after;
        Tensor::cat(
            &[
                embeds.narrow(1, 0, prompt.audio_start)?,
                audio.clone(),
                embeds.narrow(1, after, tail)?,
            ],
            1,
        )
        .map_err(Error::from)
    }

    /// Transcribe one invocation of audio on `cache`. With `on_delta`,
    /// transcript text streams as it decodes.
    pub(crate) fn transcribe_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        request: &RumbleRequestOptions,
        options: RumbleGenerationOptions,
        cache: &mut PhysicalPagedKvCache,
        mut on_delta: Option<&mut dyn FnMut(&str)>,
    ) -> Result<RumbleTranscription> {
        let started = Instant::now();
        if cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "Rumble invocation cache must start at context zero".into(),
            ));
        }
        let features = self.prepare_features(audio, sample_rate)?;
        let audio_embeddings = self.audio_embeddings(&features)?;
        let prompt = self.build_prompt(features.audio_tokens, &request.prompt_options())?;
        // The budget is narrowed to the room the prompt leaves in the
        // invocation context, which the cache was leased for.
        let room = self.sequence_tokens().saturating_sub(prompt.ids.len());
        if room == 0 {
            return Err(Error::InvalidInput(format!(
                "Rumble prompt ({} tokens) leaves no decode room in the invocation context ({})",
                prompt.ids.len(),
                self.sequence_tokens()
            )));
        }
        let max_new_tokens = options.max_new_tokens.clamp(1, room);

        let prefill_started = Instant::now();
        let embeds = self.prompt_embeddings(&prompt, &audio_embeddings.embeddings)?;
        let mut logits = self
            .text_model
            .forward_managed_last_logits_with_embeds(&embeds, 0, cache, None)?;
        drop(embeds);
        let prefill_ms = elapsed_ms(prefill_started);
        let mut position = prompt.ids.len();
        if cache.context_len() != position {
            return Err(Error::InferenceError(format!(
                "Rumble prefill committed {} tokens instead of {position}",
                cache.context_len()
            )));
        }

        let decode_started = Instant::now();
        let mut stop_ids = self.tokenizer.specials().stop_ids();
        stop_ids.extend(options.stop_token_ids.iter().copied());
        let mut generated = Vec::with_capacity(max_new_tokens.min(256));
        let mut stream = RumbleTextStream::new(&self.tokenizer, prompt.forced_language.is_some());
        let mut stop_reason = "max_tokens";
        let mut stop_token = None;
        loop {
            let next = last_row_argmax(&logits)?;
            if stop_ids.contains(&next) {
                stop_reason = "stop_token";
                stop_token = Some(next);
                break;
            }
            generated.push(next);
            let delta = stream.push(next)?;
            if !delta.is_empty() {
                if let Some(on_delta) = on_delta.as_deref_mut() {
                    on_delta(&delta);
                }
            }
            if !options.stop_sequences.is_empty()
                && options
                    .stop_sequences
                    .iter()
                    .any(|stop| !stop.is_empty() && stream.text().contains(stop.as_str()))
            {
                stop_reason = "stop_sequence";
                break;
            }
            if generated.len() >= max_new_tokens {
                break;
            }
            let input = Tensor::from_slice(&[next], (1, 1), &self.device.device)?;
            logits = self.text_model.forward_managed(&input, position, cache)?;
            position += 1;
        }
        let tail = stream.finish()?;
        if !tail.is_empty() {
            if let Some(on_delta) = on_delta {
                on_delta(&tail);
            }
        }
        let decode_ms = elapsed_ms(decode_started);

        let RumbleOutput { language, text } = split_rumble_output(
            &self.tokenizer,
            &generated,
            prompt.forced_language.as_deref(),
        )?;
        let text = trim_at_stop_sequence(text, &options.stop_sequences);
        let diagnostics = json!({
            "family": "rumble_asr",
            "device_kind": format!("{:?}", self.device.kind),
            "dtype": format!("{:?}", self.text_dtype()),
            "audio_dtype": format!("{:?}", self.audio_dtype()),
            "audio_tf32": self.audio_dtype() == DType::F32
                && self.device.device.is_cuda()
                && super::precision::audio_tf32_requested(),
            "audio_seconds": features.audio_seconds,
            "input_sample_rate": features.input_sample_rate,
            "input_samples": features.input_samples,
            "padded_samples": features.padded_samples,
            "mel_frames": features.mel_frames,
            "audio_tokens": features.audio_tokens,
            "prompt_tokens": prompt.ids.len(),
            "context_tokens": prompt.context_tokens,
            "forced_language": prompt.forced_language,
            // The transcript's language, forced or detected: the server
            // reports it when the request named none.
            "language": language,
            "style": {
                "digits": request.style.digits,
                "fillers": request.style.fillers,
                "verbatim": request.style.verbatim,
            },
            "generated_tokens": generated.len(),
            "max_new_tokens": max_new_tokens,
            "stop_reason": stop_reason,
            "stop_token": stop_token,
            "decode": {
                "generated_tokens": generated.len(),
                "max_new_tokens": max_new_tokens,
                "stop_reason": stop_reason,
                "stop_token": stop_token,
            },
            "timings_ms": {
                "resample": features.resample_ms,
                "mel": features.mel_ms,
                "audio_input_upload": audio_embeddings.upload_ms,
                "audio_encoder": audio_embeddings.encode_ms,
                "prefill": prefill_ms,
                "decode": decode_ms,
                "model_total": elapsed_ms(started),
            },
        });
        Ok(RumbleTranscription {
            text,
            language,
            diagnostics: Some(diagnostics),
        })
    }

    pub fn diagnostics_summary(&self) -> serde_json::Value {
        let audio = &self.config.audio_config;
        let text = &self.config.text_config;
        json!({
            "family": "rumble_asr",
            "device_kind": format!("{:?}", self.device.kind),
            "dtype": format!("{:?}", self.text_dtype()),
            "audio_dtype": format!("{:?}", self.audio_dtype()),
            "audio_tf32": self.audio_dtype() == DType::F32
                && self.device.device.is_cuda()
                && super::precision::audio_tf32_requested(),
            "checkpoint_format": "safetensors",
            "audio_layers": audio.encoder_layers,
            "decoder_layers": text.num_hidden_layers,
            "vocab_size": text.vocab_size,
            "tied_head": true,
            "native_context_tokens": text.max_position_embeddings,
            "sequence_tokens": self.sequence_tokens(),
            "max_audio_seconds": RUMBLE_MAX_AUDIO_SECONDS,
            "audio_precision": self.audio_precision.diagnostics(),
            "text_precision": self.text_precision.diagnostics(),
        })
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

/// The argmax of the last row of `[1, sequence, vocab]` logits.
pub(crate) fn last_row_argmax(logits: &Tensor) -> Result<u32> {
    let (rows, sequence, _) = logits.dims3()?;
    if rows != 1 || sequence == 0 {
        return Err(Error::InferenceError(format!(
            "Rumble decode output expects [1, sequence, vocab], got {:?}",
            logits.dims()
        )));
    }
    let row = logits.i((0, sequence - 1))?;
    // A NaN row has no argmax worth decoding: fail the request instead.
    let id = row
        .to_dtype(DType::F32)?
        .argmax(D::Minus1)?
        .to_scalar::<u32>()?;
    Ok(id)
}

/// Cut `text` before the first stop sequence it contains.
fn trim_at_stop_sequence(text: String, stop_sequences: &[String]) -> String {
    let cut = stop_sequences
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop.as_str()))
        .min();
    match cut {
        Some(index) => text[..index].trim_end().to_string(),
        None => text,
    }
}

/// Streams transcript text as ids arrive: nothing before `<asr_text>` when
/// the model names the language itself, and never a control token.
pub(crate) struct RumbleTextStream<'a> {
    tokenizer: &'a RumbleTokenizer,
    decoder: IncrementalDecoder,
    in_transcript: bool,
    text: String,
}

impl<'a> RumbleTextStream<'a> {
    pub(crate) fn new(tokenizer: &'a RumbleTokenizer, forced_language: bool) -> Self {
        Self {
            tokenizer,
            decoder: IncrementalDecoder::new(true),
            in_transcript: forced_language,
            text: String::new(),
        }
    }

    pub(crate) fn push(&mut self, id: u32) -> Result<String> {
        if !self.in_transcript {
            if id == self.tokenizer.specials().asr_text {
                self.in_transcript = true;
            }
            return Ok(String::new());
        }
        let delta = self.tokenizer.decode_incrementally(&mut self.decoder, id)?;
        self.text.push_str(&delta);
        Ok(delta)
    }

    pub(crate) fn finish(&mut self) -> Result<String> {
        let delta = self
            .tokenizer
            .finish_incremental_decode(&mut self.decoder)?;
        self.text.push_str(&delta);
        Ok(delta)
    }

    /// Transcript text streamed so far.
    pub(crate) fn text(&self) -> &str {
        &self.text
    }
}

fn audio_self_check(
    tower: &AudioTower,
    dtype: DType,
    device: &candle_core::Device,
) -> NumericSelfCheckReport {
    run_numeric_self_check(
        format!("Rumble audio tower {dtype:?}"),
        F16_SAFE_MAX_ABS,
        || {
            let features = xorshift_tensor(
                &[1, AUDIO_PROBE_FRAMES, tower.mel_bins()],
                0x52B1,
                1.5,
                device,
            )?
            .to_dtype(dtype)?;
            let output = tower
                .forward_feature_batch(&features, &[AUDIO_PROBE_FRAMES])?
                .into_padded();
            Ok::<_, Error>(vec![("audio_tower".to_string(), output)])
        },
    )
}

fn text_self_check(model: &Qwen3Model) -> NumericSelfCheckReport {
    run_numeric_self_check("Rumble decoder", F16_SAFE_MAX_ABS, || {
        model.stateless_numeric_probe(TEXT_PROBE_TOKENS, 0x52B2)
    })
}

/// One unbatched invocation stage: the atomic graph every request runs.
fn rumble_atomic_graph(stages: &[StageDescriptor]) -> bool {
    stages.len() == 1
        && stages[0].batch_mode == crate::engine::NativeBatchMode::None
        && !matches!(
            stages[0].selector,
            crate::engine::StageWorkSelector::SequencePrefill
                | crate::engine::StageWorkSelector::SequenceDecode
                | crate::engine::StageWorkSelector::PreSequencePreparation
        )
}

/// The retained route's decoder pages: the invocation contract's layout,
/// retained across scheduler quanta with transactional checkpoints.
fn rumble_retained_contract(
    mut contract: InferenceStateContract,
) -> Result<InferenceStateContract> {
    for domain in &mut contract.domains {
        let StateDomainSpec::PagedAttention(domain) = domain else {
            return Err(Error::ModelLoadError(
                "Rumble retained state must be paged attention".into(),
            ));
        };
        domain.header.scope = StateScope::Retained;
        domain.header.prefix = PrefixPolicy::Disabled;
        domain.header.checkpoint = CheckpointPolicy::Transactional;
    }
    contract.validate()?;
    Ok(contract)
}

/// The retained sequence graph a load may seal: scalar exact or ragged
/// batched encoder preparation, scalar exact or (on CUDA) ragged batched
/// prefill, and continuous ragged decode priced per row.
fn rumble_retained_graph(stages: &[StageDescriptor]) -> bool {
    use crate::engine::{ConcurrencyClass, NativeBatchMode, StageShapePolicy, StageWorkSelector};
    let scalar = |stage: &StageDescriptor| {
        stage.batch_mode == NativeBatchMode::None
            && stage.shape_policy == StageShapePolicy::Exact
            && stage.concurrency == ConcurrencyClass::Exclusive
            && stage.max_batch_size == 1
    };
    // The encoder runs one request at a time, or several as a ragged batch
    // with no padding.
    let ragged_batch = |stage: &StageDescriptor| {
        stage.batch_mode == NativeBatchMode::Static
            && stage.shape_policy == StageShapePolicy::Ragged
            && stage.concurrency == ConcurrencyClass::Batchable
            && stage.max_batch_size > 1
            && stage.max_padding_basis_points == 0
    };
    let preparation = stages.iter().any(|stage| {
        stage.selector == StageWorkSelector::PreSequencePreparation
            && (scalar(stage) || ragged_batch(stage))
    });
    let prefill = stages.iter().any(|stage| {
        stage.selector == StageWorkSelector::SequencePrefill
            && (scalar(stage)
                || (ragged_batch(stage)
                    && stage.name == super::retained::RUMBLE_STATIC_PREFILL_STAGE))
    });
    let decode = stages.iter().any(|stage| {
        let batch_workspace = u64::try_from(stage.max_batch_size)
            .ok()
            .and_then(|width| stage.workspace_per_row_bytes.checked_mul(width));
        stage.selector == StageWorkSelector::SequenceDecode
            && stage.batch_mode == NativeBatchMode::Continuous
            && stage.shape_policy == StageShapePolicy::Ragged
            && stage.concurrency == ConcurrencyClass::Batchable
            && stage.max_batch_size > 0
            && stage.workspace_base_bytes == 0
            && stage.workspace_per_row_bytes > 0
            && stage.workspace_per_work_unit_bytes == 0
            && batch_workspace.is_some_and(|bytes| stage.max_workspace_bytes >= bytes)
    });
    stages.len() == 3 && preparation && prefill && decode
}

fn rumble_physical_state_spec(
    stage_graphs: &[&[StageDescriptor]],
    retained: InferenceStateContract,
    invocation: InferenceStateContract,
    sequence_tokens: usize,
) -> Result<RumblePhysicalStateSpec> {
    if stage_graphs.is_empty() || sequence_tokens == 0 {
        return Err(Error::ModelLoadError(
            "Rumble state requires stages and a non-zero decoder context".into(),
        ));
    }
    let atomic_graphs = stage_graphs
        .iter()
        .filter(|stages| rumble_atomic_graph(stages))
        .count();
    let retained_graphs = stage_graphs
        .iter()
        .filter(|stages| {
            stages.iter().any(|stage| {
                matches!(
                    stage.selector,
                    crate::engine::StageWorkSelector::SequencePrefill
                        | crate::engine::StageWorkSelector::SequenceDecode
                )
            })
        })
        .collect::<Vec<_>>();
    // A retained graph needs the atomic graph beside it (long audio); with
    // the retained route off, every graph is atomic and no retained state
    // is authored.
    let valid = if retained_graphs.is_empty() {
        atomic_graphs == stage_graphs.len()
    } else {
        retained_graphs
            .iter()
            .all(|stages| rumble_retained_graph(stages))
            && atomic_graphs > 0
            && retained_graphs.len() + atomic_graphs == stage_graphs.len()
    };
    if !valid {
        return Err(Error::ModelLoadError(
            "Rumble ASR runs its atomic graph, beside at most a retained graph of scalar or ragged-batch preparation, scalar or ragged-batch prefill and continuous ragged decode"
                .into(),
        ));
    }
    let uses_retained = !retained_graphs.is_empty();
    let max_tokens = u64::try_from(sequence_tokens)
        .map_err(|_| Error::ModelLoadError("Rumble context exceeds u64".into()))?;
    let max_domain_id = invocation
        .domains
        .iter()
        .map(|domain| domain.id().get())
        .max()
        .ok_or_else(|| Error::ModelLoadError("Rumble invocation contract is empty".into()))?;
    let mut profiles = Vec::with_capacity(stage_graphs.len());
    for stages in stage_graphs {
        let atomic = rumble_atomic_graph(stages);
        let mut ordered = stages.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|stage| stage.id);
        let mut invocation_stages = Vec::with_capacity(ordered.len());
        for (index, stage) in ordered.into_iter().enumerate() {
            // Only the atomic graph leases invocation pages; the retained
            // graph's stages lease scratch beside the retained pages.
            let mut domains = if atomic {
                invocation
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
                    .collect::<Result<Vec<_>>>()?
            } else {
                Vec::new()
            };
            if stage.max_workspace_bytes > 0 {
                let scratch_bytes = if stage.workspace_per_row_bytes > 0 {
                    stage.workspace_per_row_bytes
                } else {
                    stage.max_workspace_bytes
                };
                let scratch_id = u32::try_from(index + 1)
                    .ok()
                    .and_then(|offset| max_domain_id.checked_add(offset))
                    .ok_or_else(|| {
                        Error::ModelLoadError("Rumble scratch domain id overflow".into())
                    })?;
                domains.push(InvocationWorkspaceDomain::Scratch {
                    id: StateDomainId::new(scratch_id),
                    placement: PlacementPolicy::BackendLocal,
                    alignment_bytes: 64,
                    zero_on_release: false,
                    formula: WorkspaceFormula {
                        fixed_bytes: scratch_bytes,
                        dimensions: vec![],
                        terms: vec![],
                    },
                });
            }
            invocation_stages.push(InvocationStageWorkspace {
                stage: stage.id,
                // A batched encoder pass leases its scratch once for the
                // batch, at the stage's whole-batch ceiling.
                lease_scope: if stage.selector
                    == crate::engine::StageWorkSelector::PreSequencePreparation
                    && stage.batch_mode == crate::engine::NativeBatchMode::Static
                    && stage.workspace_per_row_bytes == 0
                {
                    InvocationLeaseScope::PerStageBatch
                } else {
                    InvocationLeaseScope::PerRow
                },
                groups: if atomic {
                    invocation.groups.clone()
                } else {
                    Vec::new()
                },
                domains,
            });
        }
        profiles.push(InvocationWorkspaceProfile {
            stage_graph_fingerprint: stage_graph_fingerprint(stages)?,
            stages: invocation_stages,
        });
    }
    profiles.sort_unstable_by_key(|profile| profile.stage_graph_fingerprint);
    profiles.dedup();
    let descriptor = CapabilityStateDescriptorV2 {
        abi: CURRENT_INFERENCE_STATE_ABI,
        retained: if uses_retained {
            RetainedStateCapability::Managed {
                contract: retained.clone(),
            }
        } else {
            RetainedStateCapability::Stateless
        },
        invocation: InvocationWorkspaceSet::Bounded { profiles },
    };
    for stages in stage_graphs {
        descriptor.validate_against_stages(stages)?;
    }
    Ok(RumblePhysicalStateSpec {
        descriptor,
        retained: uses_retained.then_some(retained),
        retained_max_tokens: uses_retained.then_some(sequence_tokens),
        invocation,
        sequence_tokens,
    })
}

/// The physical state a Rumble load authors for `stage_graphs`, on the
/// synthetic checkpoint's decoder geometry, so a bundle test can publish and
/// seal exactly what a load would.
#[cfg(test)]
pub(crate) fn rumble_test_physical_state_spec(
    stage_graphs: &[&[StageDescriptor]],
) -> Result<RumblePhysicalStateSpec> {
    let checkpoint = super::synthetic::SyntheticRumbleCheckpoint::write();
    RumbleAsrModel::load(
        checkpoint.path(),
        crate::catalog::ModelVariant::Rumble2Flash,
        crate::backends::DeviceProfile::cpu(),
    )?
    .physical_state_spec(stage_graphs)
}

/// Bytes of one invocation's paged decoder KV at `max_tokens`, in whole
/// pages.
pub(crate) fn paged_invocation_bytes(state: &StateDomainSpec, max_tokens: u64) -> Result<u64> {
    let StateDomainSpec::PagedAttention(spec) = state else {
        return Err(Error::ModelLoadError(
            "Rumble invocation workspace is not paged attention".into(),
        ));
    };
    let overflow = || Error::ModelLoadError("Rumble invocation bytes overflow".into());
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
                "Rumble invocation paging requires a dense KV dtype".into(),
            )),
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .min()
        .ok_or_else(|| Error::ModelLoadError("Rumble KV dtype set is empty".into()))?;
    elements_per_token
        .checked_mul(rounded_tokens)
        .and_then(|elements| elements.checked_mul(element_bytes))
        .ok_or_else(overflow)
}

/// The longest prompt a request can build: its markers, the context text
/// and one window of audio tokens.
pub(crate) fn rumble_max_prompt_tokens() -> usize {
    RUMBLE_PROMPT_OVERHEAD_TOKENS
        + RUMBLE_MAX_CONTEXT_TOKENS
        + (RUMBLE_MAX_AUDIO_SECONDS * RUMBLE_AUDIO_TOKENS_PER_SECOND).ceil() as usize
}
