//! Main Voxtral Realtime model implementation.

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_core::{DType, IndexOp, Tensor, D};
use candle_nn::{Module, VarBuilder};
use tracing::info;

use crate::audio::{MelConfig, MelNorm, MelScale, MelSpectrogram};
use crate::backends::{BackendKind, DTypeSelection, DTypeSelectionRequest, DeviceProfile};
use crate::catalog::ModelFamily;
use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::kernels::numeric_check::{
    load_with_precision_fallback, run_numeric_self_check, xorshift_tensor, NumericSelfCheckReport,
    PrecisionLoad, F16_SAFE_MAX_ABS,
};
use crate::kv::CacheDomainId;
use crate::models::architectures::qwen3::core::build_rope_cache;
use crate::models::architectures::voxtral::lm::{VoxtralAdaScales, VoxtralLM};
use crate::models::architectures::voxtral::{
    voxtral_invocation_contract, voxtral_physical_state_spec, voxtral_realtime_physical_state_spec,
    voxtral_retained_contract, VoxtralPhysicalStateSpec, VoxtralRealtimePhysicalStateSpec,
};
use crate::models::shared::attention::flash::{
    flash_attention_compiled, flash_attention_requested, try_fused_self_attention,
    try_fused_self_attention_with_options, CudaFlashAttentionOptions,
};
use crate::models::shared::attention::paged::default_kv_page_size;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::attention::physical::PhysicalPagedKvCheckpoint;
use crate::models::shared::config::checkpoint_dtype_from_config_json;
use crate::models::shared::weights::staged;

use super::super::layers::exact_gelu;
use super::audio::{AudioLanguageAdapter, TimeEmbedding};
use super::config::VoxtralConfig;
use super::streaming::{
    VoxtralRealtimeHostCheckpoint, VoxtralRealtimeResourceUsage, VoxtralRealtimeState,
};
use super::tokenizer::{AudioConfig, VoxtralTokenizer};

static NEXT_VOXTRAL_MODEL_LOAD_NONCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VoxtralRealtimePreparationMode {
    Push,
    Finish,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VoxtralRealtimePreparationGeometry {
    pub(crate) source_samples: usize,
    pub(crate) resampled_samples: usize,
    pub(crate) padded_samples: usize,
    pub(crate) mel_frames: usize,
    pub(crate) conv1_frames: usize,
    pub(crate) conv2_frames: usize,
    pub(crate) pooled_frames: usize,
    pub(crate) stable_frames: usize,
    pub(crate) embedding_elements: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VoxtralRealtimePreparationBatchGeometry {
    pub(crate) width: usize,
    pub(crate) padded_mel_frames: usize,
    pub(crate) padded_conv_frames: usize,
    pub(crate) materialized_tensor_elements_per_row: u64,
    pub(crate) workspace_per_row_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VoxtralRealtimePreparationStageSeal {
    pub(crate) max_source_samples: usize,
    pub(crate) max_work_units: u64,
    pub(crate) max_materialized_tensor_elements_per_row: u64,
    pub(crate) max_workspace_bytes: u64,
}

/// Finite per-session authorization covering the worst preparation
/// transaction: retained old source, the owned ingress packet, replacement
/// cumulative source, old/new embedding overlap, and encoder scratch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VoxtralRealtimeStreamPeakReservation {
    pub(crate) max_source_samples: usize,
    pub(crate) max_host_bytes: u64,
    pub(crate) max_tensor_bytes: u64,
    pub(crate) max_preparation_scratch_bytes: u64,
    pub(crate) max_committed_host_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VoxtralRealtimePreparedResourceUsage {
    pub(crate) host_bytes: u64,
    pub(crate) tensor_bytes: u64,
}

pub(crate) struct VoxtralRealtimePreparationBatchRow<'a> {
    pub(crate) state: &'a VoxtralRealtimeState,
    pub(crate) appended_samples: &'a [f32],
    pub(crate) sample_rate: u32,
    pub(crate) mode: VoxtralRealtimePreparationMode,
}

#[derive(Debug, Clone)]
pub(crate) struct VoxtralRealtimePreparedAudio {
    model_load_nonce: u64,
    state_id: u64,
    expected_source_samples: usize,
    expected_source_identity: Arc<Vec<f32>>,
    preparation_generation: u64,
    source_sample_rate: u32,
    source_samples: Arc<Vec<f32>>,
    mode: VoxtralRealtimePreparationMode,
    geometry: VoxtralRealtimePreparationGeometry,
    embeddings: Tensor,
}

/// Voxtral Realtime Model
pub struct VoxtralRealtimeModel {
    model_load_nonce: u64,
    device: DeviceProfile,
    dtype: DType,
    tokenizer: VoxtralTokenizer,
    config: VoxtralConfig,
    whisper_encoder: WhisperEncoder,
    audio_adapter: AudioLanguageAdapter,
    language_model: VoxtralLM,
    /// Per-layer Ada RMSNorm gains for the model's fixed delay condition.
    ada_scales: VoxtralAdaScales,
    mel: MelSpectrogram,
    global_log_mel_max: Option<f32>,
    num_delay_tokens: usize,
    streaming_left_pad_tokens: usize,
    audio_length_per_tok: usize,
    layout: VoxtralRealtimeLayout,
    realtime_max_source_samples: usize,
    max_decoded_token_bytes: usize,
    /// The dtype this model loaded with, why, and its self-check.
    precision: PrecisionLoad,
}

/// LM positions the load-time self-check runs.
const VOXTRAL_PROBE_LM_TOKENS: usize = 16;

/// The components [`voxtral_numeric_probe`] runs.
struct VoxtralProbeParts<'a> {
    encoder: &'a WhisperEncoder,
    adapter: &'a AudioLanguageAdapter,
    pool_size: usize,
    language_model: &'a VoxtralLM,
    ada: &'a VoxtralAdaScales,
    n_mels: usize,
}

/// Xorshift log-mel (the normalized log-mel lies in about [-1.5, 1.5]) for
/// [`VOXTRAL_PROBE_LM_TOKENS`] LM positions through the encoder, the pooling
/// adapter and every LM layer and the head, on a scratch cache with K/V in
/// `dtype`. Returns the encoder output, the adapter output and the logits.
fn voxtral_numeric_probe(
    parts: VoxtralProbeParts<'_>,
    dtype: DType,
    device: &candle_core::Device,
) -> Result<Vec<(String, Tensor)>> {
    let frames = VOXTRAL_PROBE_LM_TOKENS
        .checked_mul(2 * parts.pool_size.max(1))
        .ok_or_else(|| Error::ModelLoadError("Voxtral probe overflow".into()))?;
    let mel = xorshift_tensor(&[1, frames, parts.n_mels], 0x7A11, 1.5, device)?.to_dtype(dtype)?;
    let encoded = parts.encoder.forward(&mel)?;
    let pooled = pool_audio_embeddings_by_block(&encoded, parts.pool_size)?;
    let embeds = parts
        .adapter
        .forward(&pooled)?
        .to_dtype(parts.language_model.embedding_dtype())?;
    let mut cache = parts
        .language_model
        .scratch_probe_cache(dtype, embeds.dim(1)?)?;
    let logits = parts.language_model.forward_managed_with_embeds(
        &embeds,
        0,
        &mut cache,
        None,
        Some(parts.ada),
    )?;
    Ok(vec![
        ("encoder".to_string(), encoded),
        ("adapter".to_string(), embeds),
        ("logits".to_string(), logits),
    ])
}

#[derive(Debug, Clone)]
pub(crate) struct VoxtralRealtimeCheckpoint {
    state_id: u64,
    quantum_nonce: u64,
    arena_id: crate::kv::KvArenaId,
    cache_view_id: u64,
    payload: Option<VoxtralRealtimeCheckpointPayload>,
}

#[derive(Debug, Clone)]
struct VoxtralRealtimeCheckpointPayload {
    host: VoxtralRealtimeHostCheckpoint,
    cache: PhysicalPagedKvCheckpoint,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VoxtralRealtimeStep {
    pub(crate) delta: String,
    pub(crate) text: String,
    pub(crate) tokens_generated: usize,
    pub(crate) finished: bool,
}

pub(crate) struct VoxtralRealtimeDecodeBatchRow<'a> {
    pub(crate) state: &'a mut VoxtralRealtimeState,
    pub(crate) cache: &'a mut PhysicalPagedKvCache,
}

#[derive(Debug, Clone)]
pub struct VoxtralTranscriptionOutput {
    pub text: String,
    pub language: Option<String>,
    pub diagnostics: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
struct VoxtralRuntimeSidecarConfig {
    default_num_delay_tokens: Option<usize>,
    downsample_factor: Option<usize>,
    audio_length_per_tok: Option<usize>,
}

#[derive(Debug, Clone, Default)]
struct VoxtralRuntimeConfig {
    default_num_delay_tokens: Option<usize>,
    downsample_factor: Option<usize>,
    audio_length_per_tok: Option<usize>,
    checkpoint_dtype: Option<DType>,
}

/// CPU and Metal encode a whole offline request with dense encoder
/// attention, so they cap it at 1024 audio tokens: 975 after the 49 padding
/// tokens, or 78 s of audio.
const PORTABLE_OFFLINE_AUDIO_FRAME_LIMIT: usize = 1024;

/// Longest audio a pipeline should send in one offline invocation, such as
/// one diarization chunk. 30 s is 424 audio tokens, well inside the portable
/// cap, and matches the offline route's 30 s chunks.
pub(crate) const VOXTRAL_REALTIME_OFFLINE_AUDIO_SECONDS_HINT: f32 = 30.0;

/// Longest realtime session when the checkpoint sets no `chunk_length_s`.
/// The published streaming checkpoint leaves it null, as mistral_common
/// requires in streaming mode, so the session length is an izwi policy.
/// Every push re-encodes the whole stream with dense encoder attention and
/// the preparation seal charges that quadratic workspace, so raise this only
/// once the encoder runs incrementally.
const VOXTRAL_REALTIME_DEFAULT_MAX_SESSION_SECONDS: f32 = 30.0;

/// Audio tokens one offline request may decode. Audio past the limit is an
/// error, never a silently truncated transcript.
fn voxtral_realtime_offline_frame_limit(
    backend: BackendKind,
    audio_frames: usize,
    model_context_limit: usize,
) -> Result<usize> {
    let (limit, bound) = if backend == BackendKind::Cuda {
        if model_context_limit == 0 {
            return Err(Error::ModelLoadError(
                "Voxtral Realtime reported a zero physical context".into(),
            ));
        }
        (model_context_limit, "loaded model context")
    } else {
        (
            PORTABLE_OFFLINE_AUDIO_FRAME_LIMIT,
            "CPU/Metal offline limit",
        )
    };
    if audio_frames > limit {
        return Err(Error::InvalidInput(format!(
            "Voxtral Realtime needs {audio_frames} audio tokens for this audio, exceeding its {limit}-token {bound}; split the audio into chunks of at most {VOXTRAL_REALTIME_OFFLINE_AUDIO_SECONDS_HINT} s"
        )));
    }
    Ok(audio_frames)
}

impl VoxtralRealtimeModel {
    /// Load model from directory. On Metal a dtype below F32 must pass
    /// [`Self::numeric_self_check`], or the model reloads in F32.
    pub fn load(model_dir: &Path, device: DeviceProfile, prefix_reuse: bool) -> Result<Self> {
        let config_str = read_voxtral_config(model_dir)?;
        let runtime_config = load_voxtral_runtime_config(model_dir)?;
        let selection = voxtral_precision(
            &device,
            voxtral_checkpoint_dtype(&config_str, &runtime_config),
        );
        let check = device.kind.is_metal();
        let (mut model, precision) = load_with_precision_fallback(
            "Voxtral",
            &selection,
            check,
            DType::F32,
            |dtype| Self::load_with_dtype(model_dir, device.clone(), prefix_reuse, dtype),
            Self::numeric_self_check,
        )?;
        info!(
            "Voxtral Realtime precision: {:?} ({})",
            precision.loaded, precision.reason
        );
        model.precision = precision;
        Ok(model)
    }

    fn load_with_dtype(
        model_dir: &Path,
        device: DeviceProfile,
        prefix_reuse: bool,
        dtype: DType,
    ) -> Result<Self> {
        let config_str = read_voxtral_config(model_dir)?;
        let config: VoxtralConfig = serde_json::from_str(&config_str)
            .map_err(|e| Error::ModelLoadError(format!("Failed to parse config: {}", e)))?;
        let runtime_config = load_voxtral_runtime_config(model_dir)?;
        let checkpoint_dtype = voxtral_checkpoint_dtype(&config_str, &runtime_config);
        let geometry = realtime_audio_geometry(&config, &runtime_config);
        let audio_length_per_tok = geometry.audio_length_per_tok;

        // Setup audio processing
        let audio_cfg = config.audio_config();
        let mel_cfg = MelConfig {
            sample_rate: audio_cfg.sampling_rate,
            n_fft: audio_cfg.window_size,
            win_length: None,
            hop_length: audio_cfg.hop_length,
            n_mels: audio_cfg.num_mel_bins,
            f_min: 0.0,
            f_max: 8000.0,
            normalize: audio_cfg.global_log_mel_max.is_none(),
            mel_scale: MelScale::Slaney,
            mel_norm: MelNorm::Slaney,
        };
        let mel = MelSpectrogram::new(mel_cfg)?;

        // Setup tokenizer
        let tokenizer = VoxtralTokenizer::load(model_dir, geometry.audio)?;
        let num_delay_tokens = checked_delay_tokens(tokenizer.audio_config(), &runtime_config)?;
        let max_decoded_token_bytes = tokenizer.max_decoded_token_bytes()?;
        let streaming_left_pad_tokens = tokenizer.audio_config().streaming_left_pad_tokens;
        let layout = VoxtralRealtimeLayout::new(
            &config,
            tokenizer.audio_config(),
            geometry.block_pool_size,
            geometry.audio_length_per_tok,
        )?;
        let realtime_max_source_samples = realtime_source_sample_ceiling(&config, &layout)?;

        // Load weights - clone device to a local binding for lifetime
        let device_clone = device.clone();
        let vb = load_weights(model_dir, dtype, &device_clone)?;

        // Load components
        // Note: Checkpoint uses mm_streams_embeddings.embedding_module prefix for audio components
        let whisper_prefix = "mm_streams_embeddings.embedding_module.whisper_encoder";
        let whisper_encoder = WhisperEncoder::load_voxtral(&audio_cfg, vb.pp(whisper_prefix))?;

        let hidden_size = audio_cfg.d_model * layout.pool_size;
        let audio_adapter = AudioLanguageAdapter::load(
            hidden_size,
            config.text_config().hidden_size,
            vb.pp("mm_streams_embeddings.embedding_module.audio_language_projection"),
        )?;

        // Language model uses root-level layers.* and norm (Mistral-style)
        let language_model =
            VoxtralLM::load(config.text_config().into(), vb.clone(), prefix_reuse)?;

        let time_embedding =
            TimeEmbedding::new(config.text_config().hidden_size, 10000.0, &device.device)?;
        // The delay condition is fixed for the loaded model: compute the LM's
        // Ada RMSNorm gains once instead of in every layer of every step.
        let time = Tensor::from_vec(vec![num_delay_tokens as f32], (1,), &device.device)?
            .to_dtype(dtype)?;
        let t_cond = time_embedding
            .forward(&time)?
            .to_dtype(language_model.embedding_dtype())?;
        let ada_scales = language_model.ada_scales(&t_cond)?;

        info!(
            "Loaded Voxtral Realtime model on {:?} with dtype {:?} (checkpoint_dtype={:?})",
            device.kind, dtype, checkpoint_dtype
        );

        Ok(Self {
            model_load_nonce: NEXT_VOXTRAL_MODEL_LOAD_NONCE.fetch_add(1, Ordering::Relaxed),
            device,
            dtype,
            tokenizer,
            config,
            whisper_encoder,
            audio_adapter,
            language_model,
            ada_scales,
            mel,
            global_log_mel_max: audio_cfg.global_log_mel_max,
            num_delay_tokens,
            streaming_left_pad_tokens,
            audio_length_per_tok,
            layout,
            realtime_max_source_samples,
            max_decoded_token_bytes,
            precision: PrecisionLoad::unchecked(&DTypeSelection {
                dtype,
                reason: "set by VoxtralRealtimeModel::load".into(),
            }),
        })
    }

    /// The load-time self-check: xorshift log-mel for
    /// [`VOXTRAL_PROBE_LM_TOKENS`] LM positions through the encoder, the
    /// pooling adapter and every LM layer and the head (on a scratch cache in
    /// the model's K/V dtype). Encoder, adapter and logits must be finite
    /// and below the F16-safe range.
    fn numeric_self_check(&self) -> NumericSelfCheckReport {
        run_numeric_self_check(
            format!("Voxtral {:?}", self.dtype),
            F16_SAFE_MAX_ABS,
            || {
                voxtral_numeric_probe(
                    VoxtralProbeParts {
                        encoder: &self.whisper_encoder,
                        adapter: &self.audio_adapter,
                        pool_size: self.layout.pool_size,
                        language_model: &self.language_model,
                        ada: &self.ada_scales,
                        n_mels: self.mel.config().n_mels,
                    },
                    self.dtype,
                    &self.device.device,
                )
            },
        )
    }

    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<VoxtralPhysicalStateSpec> {
        let invocation = voxtral_invocation_contract(
            &self.language_model,
            self.dtype,
            default_kv_page_size(),
            &[CacheDomainId::new(1)],
        )?;
        let attention_window_tokens = self
            .language_model
            .physical_context_limit()
            .ok_or_else(|| Error::ModelLoadError("Voxtral has no context limit".into()))?;
        let rotating_capacity_tokens = attention_window_tokens
            .checked_add(default_kv_page_size())
            .ok_or_else(|| Error::ModelLoadError("Voxtral cache capacity overflow".into()))?;
        voxtral_physical_state_spec(stage_graphs, invocation, rotating_capacity_tokens)
    }

    pub(crate) fn realtime_physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<VoxtralRealtimePhysicalStateSpec> {
        let retained = voxtral_retained_contract(voxtral_invocation_contract(
            &self.language_model,
            self.dtype,
            default_kv_page_size(),
            &[CacheDomainId::new(1)],
        )?)?;
        let attention_window_tokens = self
            .language_model
            .physical_context_limit()
            .ok_or_else(|| Error::ModelLoadError("Voxtral has no context limit".into()))?;
        let rotating_capacity_tokens = attention_window_tokens
            .checked_add(default_kv_page_size())
            .ok_or_else(|| Error::ModelLoadError("Voxtral cache capacity overflow".into()))?;
        voxtral_realtime_physical_state_spec(stage_graphs, retained, rotating_capacity_tokens)
    }

    pub(crate) fn start_realtime_state(&self, language: Option<&str>) -> VoxtralRealtimeState {
        VoxtralRealtimeState::new(language)
    }

    pub(crate) fn realtime_max_output_steps(&self) -> Result<usize> {
        self.language_model
            .model_context_limit()
            .filter(|limit| *limit > 0)
            .ok_or_else(|| Error::ModelLoadError("Voxtral has no non-zero context limit".into()))
    }

    pub(crate) fn realtime_stream_resource_usage(
        &self,
        state: &VoxtralRealtimeState,
    ) -> Result<VoxtralRealtimeResourceUsage> {
        state.resource_usage()
    }

    pub(crate) fn begin_realtime_quantum(
        &self,
        state: &mut VoxtralRealtimeState,
        cache: &PhysicalPagedKvCache,
    ) -> Result<VoxtralRealtimeCheckpoint> {
        if state.active_quantum.is_some() {
            return Err(Error::InferenceError(
                "a Voxtral realtime quantum is already active".into(),
            ));
        }
        let expected_cursor = if state.prompt_initialized {
            state.next_audio_frame
        } else {
            0
        };
        if cache.context_len() != expected_cursor {
            return Err(Error::InferenceError(format!(
                "Voxtral realtime host/KV cursors disagree: host {expected_cursor}, cache {}",
                cache.context_len()
            )));
        }
        let quantum_nonce = state.next_quantum_nonce;
        let next_quantum_nonce = state.next_quantum_nonce.checked_add(1).ok_or_else(|| {
            Error::InferenceError("Voxtral realtime quantum nonce overflow".into())
        })?;
        state.bind_cache_authority(cache.sequence_authority())?;
        state.next_quantum_nonce = next_quantum_nonce;
        state.active_quantum = Some(quantum_nonce);
        state.active_cache_arena = Some(cache.arena().id());
        state.active_cache_view_id = Some(cache.view_id());
        Ok(VoxtralRealtimeCheckpoint {
            state_id: state.state_id,
            quantum_nonce,
            arena_id: cache.arena().id(),
            cache_view_id: cache.view_id(),
            payload: Some(VoxtralRealtimeCheckpointPayload {
                host: state.checkpoint(),
                cache: cache.logical_checkpoint(),
            }),
        })
    }

    pub(crate) fn commit_realtime_quantum(
        &self,
        state: &mut VoxtralRealtimeState,
        cache: &PhysicalPagedKvCache,
        checkpoint: &mut VoxtralRealtimeCheckpoint,
    ) -> Result<()> {
        self.validate_realtime_checkpoint(state, cache, checkpoint)?;
        if cache.context_len() != state.next_audio_frame {
            return Err(Error::InferenceError(format!(
                "Voxtral realtime host/KV cursors disagree at commit: host {}, cache {}",
                state.next_audio_frame,
                cache.context_len()
            )));
        }
        checkpoint.payload.take().ok_or_else(|| {
            Error::InferenceError("Voxtral realtime checkpoint was already consumed".into())
        })?;
        state.active_quantum = None;
        state.active_cache_arena = None;
        state.active_cache_view_id = None;
        Ok(())
    }

    pub(crate) fn rollback_realtime_quantum(
        &self,
        state: &mut VoxtralRealtimeState,
        cache: &mut PhysicalPagedKvCache,
        checkpoint: &mut VoxtralRealtimeCheckpoint,
    ) -> Result<()> {
        self.validate_realtime_checkpoint(state, cache, checkpoint)?;
        let payload = checkpoint.payload.as_ref().ok_or_else(|| {
            Error::InferenceError("Voxtral realtime checkpoint was already consumed".into())
        })?;
        cache.restore_logical_checkpoint(payload.cache.clone())?;
        let payload = checkpoint.payload.take().ok_or_else(|| {
            Error::InferenceError("Voxtral realtime checkpoint was already consumed".into())
        })?;
        state.restore_checkpoint(payload.host);
        state.active_quantum = None;
        state.active_cache_arena = None;
        state.active_cache_view_id = None;
        Ok(())
    }

    fn validate_realtime_checkpoint(
        &self,
        state: &VoxtralRealtimeState,
        cache: &PhysicalPagedKvCache,
        checkpoint: &VoxtralRealtimeCheckpoint,
    ) -> Result<()> {
        if checkpoint.state_id != state.state_id
            || state.active_quantum != Some(checkpoint.quantum_nonce)
            || state.active_cache_arena != Some(checkpoint.arena_id)
            || state.active_cache_view_id != Some(checkpoint.cache_view_id)
            || checkpoint.arena_id != cache.arena().id()
            || checkpoint.cache_view_id != cache.view_id()
            || checkpoint.payload.is_none()
        {
            return Err(Error::InferenceError(
                "Voxtral realtime checkpoint is foreign, stale, or out of order".into(),
            ));
        }
        Ok(())
    }

    /// Accept source-rate samples and refresh the stable causal encoder prefix.
    /// The host mutation is atomic even before an executor-owned KV quantum is
    /// opened, so a failed preparation never consumes input.
    pub(crate) fn push_realtime_samples(
        &self,
        state: &mut VoxtralRealtimeState,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<usize> {
        if samples.is_empty() {
            if state.input_closed || state.finished {
                return Err(Error::InvalidInput(
                    "Voxtral realtime input is already closed or finished".into(),
                ));
            }
            return Ok(0);
        }
        let prepared = self
            .prepare_realtime_audio_batch(&[VoxtralRealtimePreparationBatchRow {
                state,
                appended_samples: samples,
                sample_rate,
                mode: VoxtralRealtimePreparationMode::Push,
            }])?
            .pop()
            .ok_or_else(|| Error::InferenceError("Voxtral preparation omitted its row".into()))?;
        self.install_realtime_audio_preparation(state, prepared)
    }

    /// Apply one input chunk inside a checkpoint opened by
    /// `begin_realtime_quantum`. The handler owns the final cancellation fence
    /// and must commit or roll back the checkpoint after this returns.
    pub(crate) fn apply_realtime_push_physical(
        &self,
        state: &mut VoxtralRealtimeState,
        cache: &mut PhysicalPagedKvCache,
        samples: &[f32],
        sample_rate: u32,
        max_output_steps: usize,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Vec<VoxtralRealtimeStep>> {
        if samples.is_empty() {
            return Err(Error::InvalidInput(
                "Voxtral realtime push requires a non-empty audio chunk".into(),
            ));
        }
        self.ensure_active_realtime_quantum(state, cache)?;
        Self::check_realtime_cancelled(should_cancel)?;
        self.push_realtime_samples(state, samples, sample_rate)?;
        Self::check_realtime_cancelled(should_cancel)?;
        self.drain_realtime_steps(state, cache, max_output_steps, should_cancel)
    }

    /// Close input and apply the alignment/right-padding tail exactly once.
    pub(crate) fn finish_realtime_input(&self, state: &mut VoxtralRealtimeState) -> Result<usize> {
        if state.final_padding_applied {
            return Ok(0);
        }
        if state.finished {
            state.input_closed = true;
            state.final_padding_applied = true;
            return Ok(0);
        }
        let sample_rate = state
            .source_sample_rate
            .unwrap_or(self.mel.config().sample_rate as u32);
        let prepared = self
            .prepare_realtime_audio_batch(&[VoxtralRealtimePreparationBatchRow {
                state,
                appended_samples: &[],
                sample_rate,
                mode: VoxtralRealtimePreparationMode::Finish,
            }])?
            .pop()
            .ok_or_else(|| Error::InferenceError("Voxtral preparation omitted its row".into()))?;
        self.install_realtime_audio_preparation(state, prepared)
    }

    /// Apply input closure inside a checkpoint opened by
    /// `begin_realtime_quantum`. The handler owns the final cancellation fence
    /// and must commit or roll back the checkpoint after this returns.
    pub(crate) fn apply_realtime_finish_physical(
        &self,
        state: &mut VoxtralRealtimeState,
        cache: &mut PhysicalPagedKvCache,
        max_output_steps: usize,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Vec<VoxtralRealtimeStep>> {
        self.ensure_active_realtime_quantum(state, cache)?;
        Self::check_realtime_cancelled(should_cancel)?;
        self.finish_realtime_input(state)?;
        Self::check_realtime_cancelled(should_cancel)?;
        self.drain_realtime_steps(state, cache, max_output_steps, should_cancel)
    }

    fn ensure_active_realtime_quantum(
        &self,
        state: &VoxtralRealtimeState,
        cache: &PhysicalPagedKvCache,
    ) -> Result<()> {
        if state.active_quantum.is_none() {
            return Err(Error::InferenceError(
                "Voxtral realtime apply requires an active checkpoint".into(),
            ));
        }
        if !state.active_cache_matches(cache.arena().id(), cache.view_id()) {
            return Err(Error::InferenceError(
                "Voxtral realtime apply received a cache outside its active checkpoint".into(),
            ));
        }
        let expected_cursor = if state.prompt_initialized {
            state.next_audio_frame
        } else {
            0
        };
        if cache.context_len() != expected_cursor {
            return Err(Error::InferenceError(format!(
                "Voxtral realtime host/KV cursors disagree before apply: host {expected_cursor}, cache {}",
                cache.context_len()
            )));
        }
        Ok(())
    }

    fn finish_realtime_quantum<T>(
        &self,
        state: &mut VoxtralRealtimeState,
        cache: &mut PhysicalPagedKvCache,
        checkpoint: &mut VoxtralRealtimeCheckpoint,
        result: Result<T>,
    ) -> Result<T> {
        match result {
            Ok(output) => {
                self.commit_realtime_quantum(state, cache, checkpoint)?;
                Ok(output)
            }
            Err(error) => {
                self.rollback_realtime_quantum(state, cache, checkpoint)
                    .map_err(|rollback| {
                        Error::InferenceError(format!(
                            "{error}; Voxtral realtime rollback also failed: {rollback}"
                        ))
                    })?;
                Err(error)
            }
        }
    }

    fn drain_realtime_steps(
        &self,
        state: &mut VoxtralRealtimeState,
        cache: &mut PhysicalPagedKvCache,
        max_output_steps: usize,
        should_cancel: &mut dyn FnMut() -> bool,
    ) -> Result<Vec<VoxtralRealtimeStep>> {
        let mut steps = Vec::with_capacity(max_output_steps.min(64));
        while steps.len() < max_output_steps && !state.finished {
            Self::check_realtime_cancelled(should_cancel)?;
            let step = if state.prompt_initialized {
                self.decode_realtime_step_unchecked(state, cache)?
            } else {
                self.prefill_realtime_unchecked(state, cache)?
            };
            let Some(step) = step else {
                break;
            };
            let finished = step.finished;
            steps.push(step);
            if finished {
                break;
            }
        }
        Ok(steps)
    }

    fn check_realtime_cancelled(should_cancel: &mut dyn FnMut() -> bool) -> Result<()> {
        if should_cancel() {
            Err(Error::Cancelled(
                "Voxtral realtime quantum cancelled".into(),
            ))
        } else {
            Ok(())
        }
    }

    /// Prefill the transcription prompt once enough audio frames are stable.
    pub(crate) fn prefill_realtime_physical(
        &self,
        state: &mut VoxtralRealtimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Option<VoxtralRealtimeStep>> {
        let mut checkpoint = self.begin_realtime_quantum(state, cache)?;
        let result = self.prefill_realtime_unchecked(state, cache);
        self.finish_realtime_quantum(state, cache, &mut checkpoint, result)
    }

    fn prefill_realtime_unchecked(
        &self,
        state: &mut VoxtralRealtimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Option<VoxtralRealtimeStep>> {
        if state.prompt_initialized {
            return Err(Error::InvalidInput(
                "Voxtral realtime prompt is already initialized".into(),
            ));
        }
        if cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "Voxtral realtime prompt requires an empty retained cache".into(),
            ));
        }
        let prompt_tokens = self.tokenizer.build_transcription_prompt()?;
        let prompt_len = prompt_tokens.len();
        if state.prepared_audio_frames < prompt_len {
            return Ok(None);
        }
        let audio_embeds = state.audio_embeds.as_ref().ok_or_else(|| {
            Error::InferenceError("Voxtral realtime audio embeddings are unavailable".into())
        })?;
        let prompt_tensor = Tensor::from_vec(prompt_tokens, (1, prompt_len), &self.device.device)?;
        let text_embeds = self.language_model.embeddings(&prompt_tensor)?;
        let mut audio_prompt = audio_embeds.narrow(1, 0, prompt_len)?;
        if audio_prompt.dtype() != text_embeds.dtype() {
            audio_prompt = audio_prompt.to_dtype(text_embeds.dtype())?;
        }
        let prompt_embeds = audio_prompt.broadcast_add(&text_embeds)?;
        let logits = self.language_model.forward_managed_with_embeds(
            &prompt_embeds,
            0,
            cache,
            None,
            Some(&self.ada_scales),
        )?;
        let next_logits = logits.i((0, logits.dim(1)? - 1))?;
        state.prompt_initialized = true;
        state.next_audio_frame = prompt_len;
        self.accept_realtime_prediction(state, argmax(&next_logits)?)
            .map(Some)
    }

    pub(crate) fn realtime_prompt_cache_append(
        &self,
        state: &VoxtralRealtimeState,
    ) -> Result<Option<usize>> {
        if state.prompt_initialized || state.finished {
            return Ok(None);
        }
        let prompt_len = self.tokenizer.build_transcription_prompt()?.len();
        Ok((state.prepared_audio_frames >= prompt_len).then_some(prompt_len))
    }

    pub(crate) fn realtime_decode_ready(&self, state: &VoxtralRealtimeState) -> bool {
        state.prompt_initialized
            && !state.finished
            && state.pending_input_token.is_some()
            && state.next_audio_frame < state.prepared_audio_frames
    }

    /// Prompt prefill inside an executor-owned outer checkpoint.
    pub(crate) fn prefill_realtime_in_quantum(
        &self,
        state: &mut VoxtralRealtimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<VoxtralRealtimeStep> {
        self.ensure_active_realtime_quantum(state, cache)?;
        self.prefill_realtime_unchecked(state, cache)?
            .ok_or_else(|| {
                Error::InvalidInput("Voxtral realtime prompt prefill is not ready".into())
            })
    }

    /// Tensor-free terminal transition after all prepared audio was consumed.
    pub(crate) fn complete_realtime_in_quantum(
        &self,
        state: &mut VoxtralRealtimeState,
        cache: &PhysicalPagedKvCache,
    ) -> Result<VoxtralRealtimeStep> {
        self.ensure_active_realtime_quantum(state, cache)?;
        if !state.input_closed || state.next_audio_frame < state.prepared_audio_frames {
            return Err(Error::InvalidInput(
                "Voxtral realtime completion requires closed, exhausted input".into(),
            ));
        }
        self.finish_realtime_text(state)
    }

    /// Advance one retained decoder token against one prepared audio frame.
    pub(crate) fn decode_realtime_step_physical(
        &self,
        state: &mut VoxtralRealtimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Option<VoxtralRealtimeStep>> {
        let mut checkpoint = self.begin_realtime_quantum(state, cache)?;
        let result = self.decode_realtime_step_unchecked(state, cache);
        self.finish_realtime_quantum(state, cache, &mut checkpoint, result)
    }

    /// Advance one ready token for each retained row. Every row must already
    /// own an active outer quantum; the caller remains responsible for sealing
    /// or rolling that quantum back after this method returns.
    pub(crate) fn decode_realtime_step_batch(
        &self,
        rows: &mut [VoxtralRealtimeDecodeBatchRow<'_>],
    ) -> Result<Vec<VoxtralRealtimeStep>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        if rows.len() == 1 {
            self.ensure_active_realtime_quantum(rows[0].state, rows[0].cache)?;
            return self
                .decode_realtime_step_unchecked(rows[0].state, rows[0].cache)?
                .map(|step| vec![step])
                .ok_or_else(|| {
                    Error::InvalidInput(
                        "Voxtral realtime decode batch row is not ready to advance".into(),
                    )
                });
        }

        for (row_index, row) in rows.iter().enumerate() {
            self.ensure_active_realtime_quantum(row.state, row.cache)?;
            if !row.state.prompt_initialized
                || row.state.finished
                || row.state.next_audio_frame >= row.state.prepared_audio_frames
                || row.state.pending_input_token.is_none()
                || row.cache.context_len() != row.state.next_audio_frame
            {
                return Err(Error::InvalidInput(format!(
                    "Voxtral realtime decode batch row {row_index} is not ready to advance"
                )));
            }
            row.state
                .next_audio_frame
                .checked_add(1)
                .ok_or_else(|| Error::InvalidInput("Voxtral realtime cursor overflow".into()))?;
        }

        let host_checkpoints = rows
            .iter()
            .map(|row| row.state.checkpoint())
            .collect::<Vec<_>>();
        let cache_checkpoints = rows
            .iter()
            .map(|row| row.cache.logical_checkpoint())
            .collect::<Vec<_>>();
        let execution = (|| -> Result<Vec<VoxtralRealtimeStep>> {
            let mut embed_rows = Vec::with_capacity(rows.len());
            let mut positions = Vec::with_capacity(rows.len());
            for row in rows.iter() {
                let input_token = row
                    .state
                    .pending_input_token
                    .expect("validated pending token");
                let input = Tensor::from_vec(vec![input_token], (1, 1), &self.device.device)?;
                let text_embed = self.language_model.embeddings(&input)?;
                let audio_embeds = row.state.audio_embeds.as_ref().ok_or_else(|| {
                    Error::InferenceError(
                        "Voxtral realtime audio embeddings are unavailable".into(),
                    )
                })?;
                let mut audio_step = audio_embeds.narrow(1, row.state.next_audio_frame, 1)?;
                if audio_step.dtype() != text_embed.dtype() {
                    audio_step = audio_step.to_dtype(text_embed.dtype())?;
                }
                let step_embeds = audio_step.broadcast_add(&text_embed)?;
                embed_rows.push(step_embeds);
                positions.push(row.state.next_audio_frame);
            }
            let embeds = Tensor::cat(&embed_rows.iter().collect::<Vec<_>>(), 0)?;
            let output = {
                let mut caches = rows
                    .iter_mut()
                    .map(|row| &mut *row.cache)
                    .collect::<Vec<_>>();
                self.language_model
                    .forward_managed_decode_batch_with_embeds(
                        &embeds,
                        &positions,
                        &mut caches,
                        Some(&self.ada_scales),
                    )?
            };
            let (batch, sequence, _) = output.dims3()?;
            if batch != rows.len() || sequence != 1 {
                return Err(Error::InferenceError(
                    "Voxtral realtime decode batch returned incompatible logits".into(),
                ));
            }
            let sampled = (0..batch)
                .map(|row| -> Result<u32> {
                    let logits = output.i((row, 0))?;
                    argmax(&logits)
                })
                .collect::<Result<Vec<_>>>()?;
            for row in rows.iter_mut() {
                row.state.next_audio_frame += 1;
            }
            rows.iter_mut()
                .zip(sampled)
                .map(|(row, token)| self.accept_realtime_prediction(row.state, token))
                .collect()
        })();
        match execution {
            Ok(output) => Ok(output),
            Err(error) => {
                let mut rollback_error = None;
                for ((row, host), cache) in
                    rows.iter_mut().zip(host_checkpoints).zip(cache_checkpoints)
                {
                    row.state.restore_checkpoint(host);
                    if let Err(rollback) = row.cache.restore_logical_checkpoint(cache) {
                        rollback_error.get_or_insert(rollback);
                    }
                }
                if let Some(rollback) = rollback_error {
                    Err(Error::InferenceError(format!(
                        "Voxtral realtime decode batch failed: {error}; rollback also failed: {rollback}"
                    )))
                } else {
                    Err(error)
                }
            }
        }
    }

    fn decode_realtime_step_unchecked(
        &self,
        state: &mut VoxtralRealtimeState,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Option<VoxtralRealtimeStep>> {
        if !state.prompt_initialized {
            return Err(Error::InvalidInput(
                "Voxtral realtime decode requires prompt prefill".into(),
            ));
        }
        if state.finished {
            return Ok(None);
        }
        if state.next_audio_frame >= state.prepared_audio_frames {
            if state.input_closed {
                return self.finish_realtime_text(state).map(Some);
            }
            return Ok(None);
        }
        let input_token = state.pending_input_token.ok_or_else(|| {
            Error::InferenceError("Voxtral realtime decoder lost its pending token".into())
        })?;
        if cache.context_len() != state.next_audio_frame {
            return Err(Error::InvalidInput(format!(
                "Voxtral realtime host/KV cursors disagree: frame {}, cache {}",
                state.next_audio_frame,
                cache.context_len()
            )));
        }
        let audio_embeds = state.audio_embeds.as_ref().ok_or_else(|| {
            Error::InferenceError("Voxtral realtime audio embeddings are unavailable".into())
        })?;
        let input_tensor = Tensor::from_vec(vec![input_token], (1, 1), &self.device.device)?;
        let text_embed = self.language_model.embeddings(&input_tensor)?;
        let mut audio_step = audio_embeds.narrow(1, state.next_audio_frame, 1)?;
        if audio_step.dtype() != text_embed.dtype() {
            audio_step = audio_step.to_dtype(text_embed.dtype())?;
        }
        let step_embeds = audio_step.broadcast_add(&text_embed)?;
        let logits = self.language_model.forward_managed_with_embeds(
            &step_embeds,
            state.next_audio_frame,
            cache,
            None,
            Some(&self.ada_scales),
        )?;
        let next_logits = logits.i((0, logits.dim(1)? - 1))?;
        state.next_audio_frame = state
            .next_audio_frame
            .checked_add(1)
            .ok_or_else(|| Error::InvalidInput("Voxtral realtime cursor overflow".into()))?;
        self.accept_realtime_prediction(state, argmax(&next_logits)?)
            .map(Some)
    }

    /// Transcribe audio (non-streaming)
    pub fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        _language: Option<&str>,
    ) -> Result<String> {
        let _ = (audio, sample_rate, _language);
        Err(Error::InferenceError(
            "Voxtral Realtime requires a lifecycle-owned physical invocation cache".into(),
        ))
    }

    pub fn transcribe_with_details(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<VoxtralTranscriptionOutput> {
        let _ = (audio, sample_rate, language);
        Err(Error::InferenceError(
            "Voxtral Realtime requires a lifecycle-owned physical invocation cache".into(),
        ))
    }

    pub fn transcribe_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        _language: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let _ = (audio, sample_rate, _language, on_delta);
        Err(Error::InferenceError(
            "Voxtral Realtime requires a lifecycle-owned physical invocation cache".into(),
        ))
    }

    pub(crate) fn transcribe_with_details_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<VoxtralTranscriptionOutput> {
        self.transcribe_impl(audio, sample_rate, language, None, cache)
    }

    pub(crate) fn transcribe_with_callback_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<String> {
        Ok(self
            .transcribe_impl(audio, sample_rate, language, Some(on_delta), cache)?
            .text)
    }

    fn transcribe_impl(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        mut on_delta: Option<&mut dyn FnMut(&str)>,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<VoxtralTranscriptionOutput> {
        if cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "Voxtral Realtime invocation cache must start empty".into(),
            ));
        }
        let cache_diagnostics = self.cache_diagnostics(cache);
        let input_samples = audio.len();
        let total_start = Instant::now();
        let preprocess_start = Instant::now();
        // Resample to 16kHz if needed
        let audio = if sample_rate != 16_000 {
            resample_audio(audio, sample_rate, 16_000)?
        } else {
            audio.to_vec()
        };

        let backend = BackendKind::from(self.device.kind);
        let model_context_limit = self
            .language_model
            .model_context_limit()
            .ok_or_else(|| Error::ModelLoadError("Voxtral has no context limit".into()))?;
        let attention_window_tokens = self
            .language_model
            .physical_context_limit()
            .ok_or_else(|| Error::ModelLoadError("Voxtral has no attention window".into()))?;
        // Reject over-long audio before the encoder runs over all of it.
        let (left_pad, right_pad) = offline_streaming_padding_samples(
            audio.len(),
            self.layout.raw_audio_length_per_tok,
            self.layout.left_pad_tokens,
            self.layout.right_pad_tokens,
        );
        let predicted_audio_frames = left_pad
            .checked_add(audio.len())
            .and_then(|samples| samples.checked_add(right_pad))
            .ok_or_else(|| Error::InvalidInput("Voxtral audio length overflow".into()))?
            / self.layout.raw_audio_length_per_tok.max(1);
        voxtral_realtime_offline_frame_limit(backend, predicted_audio_frames, model_context_limit)?;

        let padded_audio = self.pad_offline_streaming_audio(&audio);

        // Compute mel spectrogram
        let mut mel_spec = self.mel.compute(&padded_audio)?;
        drop_last_mel_frame_for_voxtral(&mut mel_spec);
        if let Some(max_val) = self.global_log_mel_max {
            normalize_log_mel_with_max(&mut mel_spec, max_val);
        }
        let n_mels = self.mel.config().n_mels;
        let frames = mel_spec.len();

        let mel = mel_frames_to_tensor(&mel_spec, n_mels, &self.device.device, self.dtype)?;
        let preprocess_duration = preprocess_start.elapsed();

        let encoder_start = Instant::now();
        // Process through whisper encoder
        let audio_embeds = self.whisper_encoder.forward(&mel)?;

        // Apply pooling
        let audio_embeds = self.pool_audio_embeddings(&audio_embeds)?;

        // Project to language model dimension
        let audio_embeds = self.audio_adapter.forward(&audio_embeds)?;
        let encoder_adapter_duration = encoder_start.elapsed();

        let audio_frames = audio_embeds.dim(1)?;
        if audio_frames == 0 {
            let timings = VoxtralRealtimeTimings {
                preprocess: preprocess_duration,
                encoder_adapter: encoder_adapter_duration,
                total: total_start.elapsed(),
                ..Default::default()
            };
            self.log_transcription_timings(&timings, 0, 0, 0);
            return Ok(VoxtralTranscriptionOutput {
                text: String::new(),
                language: language.map(str::to_string),
                diagnostics: Some(self.execution_diagnostics(
                    input_samples,
                    audio.len(),
                    padded_audio.len(),
                    frames,
                    audio_frames,
                    0,
                    0,
                    0,
                    cache_diagnostics,
                    Some(timings),
                )),
            });
        }

        // Generate
        let prompt_tokens = self.tokenizer.build_transcription_prompt()?;
        let prompt_len = prompt_tokens.len();
        let max_frames =
            voxtral_realtime_offline_frame_limit(backend, audio_frames, model_context_limit)?;
        if prompt_len > max_frames {
            return Err(Error::InferenceError(format!(
                "Voxtral prompt length ({prompt_len}) exceeds available audio frames ({max_frames})"
            )));
        }
        let page_tokens = default_kv_page_size();
        let resident_tokens = cache
            .capacity_tokens()
            .saturating_sub(cache.window_start().div_euclid(page_tokens) * page_tokens);
        let required_resident_tokens = attention_window_tokens
            .checked_add(page_tokens)
            .ok_or_else(|| Error::ModelLoadError("Voxtral cache capacity overflow".into()))?;
        if resident_tokens < required_resident_tokens {
            return Err(Error::InvalidInput(format!(
                "Voxtral Realtime needs {required_resident_tokens} resident cache tokens for rotating attention, but its invocation lease has {resident_tokens}"
            )));
        }
        let mut generated = Vec::new();
        let mut assembled = String::new();
        let specials = self.tokenizer.specials().clone();

        let lm_prefill_start = Instant::now();
        let prompt_tensor =
            Tensor::from_vec(prompt_tokens.clone(), (1, prompt_len), &self.device.device)?;
        let text_embeds = self.language_model.embeddings(&prompt_tensor)?;
        let mut audio_prompt = audio_embeds.narrow(1, 0, prompt_len)?;
        if audio_prompt.dtype() != text_embeds.dtype() {
            audio_prompt = audio_prompt.to_dtype(text_embeds.dtype())?;
        }
        let prompt_embeds = audio_prompt.broadcast_add(&text_embeds)?;
        let logits = self.language_model.forward_managed_with_embeds(
            &prompt_embeds,
            0,
            cache,
            None,
            Some(&self.ada_scales),
        )?;
        let next_logits = logits.i((0, logits.dim(1)? - 1))?;
        let mut next = argmax(&next_logits)?;
        let lm_prefill_duration = lm_prefill_start.elapsed();
        let mut frame_idx = prompt_len;
        let decode_start = Instant::now();

        loop {
            if next == specials.eos || Some(next) == specials.end_audio {
                break;
            }

            append_generated_token(
                &self.tokenizer,
                &mut generated,
                &mut assembled,
                next,
                &mut on_delta,
            )?;

            if frame_idx >= max_frames {
                break;
            }

            let input_tensor = Tensor::from_vec(vec![next], (1, 1), &self.device.device)?;
            let text_embed = self.language_model.embeddings(&input_tensor)?;
            let mut audio_step = audio_embeds.narrow(1, frame_idx, 1)?;
            if audio_step.dtype() != text_embed.dtype() {
                audio_step = audio_step.to_dtype(text_embed.dtype())?;
            }
            let step_embeds = audio_step.broadcast_add(&text_embed)?;

            let logits = self.language_model.forward_managed_with_embeds(
                &step_embeds,
                frame_idx,
                cache,
                None,
                Some(&self.ada_scales),
            )?;
            let next_logits = logits.i((0, logits.dim(1)? - 1))?;
            next = argmax(&next_logits)?;
            frame_idx += 1;
        }
        flush_generated_text(&self.tokenizer, &generated, &mut assembled, &mut on_delta)?;
        let token_decode_duration = decode_start.elapsed();
        let timings = VoxtralRealtimeTimings {
            preprocess: preprocess_duration,
            encoder_adapter: encoder_adapter_duration,
            lm_prefill: lm_prefill_duration,
            token_decode: token_decode_duration,
            total: total_start.elapsed(),
        };
        self.log_transcription_timings(&timings, generated.len(), prompt_len, max_frames);

        Ok(VoxtralTranscriptionOutput {
            text: assembled.trim().to_string(),
            language: language.map(str::to_string),
            diagnostics: Some(self.execution_diagnostics(
                input_samples,
                audio.len(),
                padded_audio.len(),
                frames,
                audio_frames,
                max_frames,
                prompt_len,
                generated.len(),
                cache_diagnostics,
                Some(timings),
            )),
        })
    }

    fn pad_offline_streaming_audio(&self, audio: &[f32]) -> Vec<f32> {
        let (left_pad, right_pad) = offline_streaming_padding_samples(
            audio.len(),
            self.layout.raw_audio_length_per_tok,
            self.layout.left_pad_tokens,
            self.layout.right_pad_tokens,
        );
        let mut padded = Vec::with_capacity(left_pad + audio.len() + right_pad);
        padded.extend(std::iter::repeat_n(0.0, left_pad));
        padded.extend_from_slice(audio);
        padded.extend(std::iter::repeat_n(0.0, right_pad));
        padded
    }

    pub(crate) fn realtime_preparation_geometry(
        &self,
        state: &VoxtralRealtimeState,
        appended_samples: usize,
        sample_rate: u32,
        mode: VoxtralRealtimePreparationMode,
    ) -> Result<VoxtralRealtimePreparationGeometry> {
        self.validate_realtime_preparation_row(state, appended_samples, sample_rate, mode)?;
        let source_samples = state
            .source_samples
            .len()
            .checked_add(appended_samples)
            .ok_or_else(|| Error::InvalidInput("Voxtral realtime audio length overflow".into()))?;
        realtime_preparation_geometry_for(source_samples, sample_rate, &self.layout, mode)
    }

    pub(crate) fn realtime_preparation_batch_geometry(
        &self,
        rows: &[VoxtralRealtimePreparationGeometry],
    ) -> Result<VoxtralRealtimePreparationBatchGeometry> {
        let width = rows.len();
        if width == 0 {
            return Err(Error::InvalidInput(
                "Voxtral preparation batch must contain at least one row".into(),
            ));
        }
        let padded_mel_frames = rows.iter().map(|row| row.mel_frames).max().unwrap_or(0);
        let padded_conv_frames = rows.iter().map(|row| row.conv2_frames).max().unwrap_or(0);
        if padded_mel_frames == 0 || padded_conv_frames == 0 {
            return Err(Error::InvalidInput(
                "Voxtral preparation rows must produce non-empty encoder inputs".into(),
            ));
        }
        let materialized_tensor_elements_per_row = realtime_materialized_elements_for(
            &self.layout,
            padded_mel_frames,
            padded_conv_frames,
        )?;
        let workspace_per_row_bytes = realtime_workspace_bytes_for(
            &self.layout,
            padded_mel_frames,
            padded_conv_frames,
            self.dtype.size_in_bytes(),
        )?;
        Ok(VoxtralRealtimePreparationBatchGeometry {
            width,
            padded_mel_frames,
            padded_conv_frames,
            materialized_tensor_elements_per_row,
            workspace_per_row_bytes,
        })
    }

    pub(crate) fn realtime_preparation_stage_seal(
        &self,
    ) -> Result<VoxtralRealtimePreparationStageSeal> {
        realtime_preparation_stage_seal_for(
            &self.layout,
            self.realtime_max_source_samples,
            self.dtype.size_in_bytes(),
        )
    }

    pub(crate) fn realtime_stream_peak_reservation(
        &self,
    ) -> Result<VoxtralRealtimeStreamPeakReservation> {
        let seal = self.realtime_preparation_stage_seal()?;
        let source_bytes = u64::try_from(seal.max_source_samples)
            .ok()
            .and_then(|samples| samples.checked_mul(std::mem::size_of::<f32>() as u64))
            .ok_or_else(|| {
                Error::ModelLoadError("Voxtral source-byte ceiling overflowed".into())
            })?;
        let geometry = realtime_preparation_geometry_for(
            seal.max_source_samples,
            self.layout.target_rate,
            &self.layout,
            VoxtralRealtimePreparationMode::Finish,
        )?;
        let embedding_bytes = geometry
            .embedding_elements
            .checked_mul(self.dtype.size_in_bytes() as u64)
            .ok_or_else(|| Error::ModelLoadError("Voxtral embedding ceiling overflowed".into()))?;
        let frontend_host_bytes = checked_realtime_frontend_host_peak_bytes(
            source_bytes,
            geometry.resampled_samples,
            geometry.padded_samples,
            geometry.mel_frames,
            self.mel.config().n_mels,
            self.mel.config().n_fft,
        )?;
        let (committed_text_host_bytes, transactional_text_host_bytes) =
            checked_realtime_text_host_bytes(geometry.pooled_frames, self.max_decoded_token_bytes)?;
        let max_committed_host_bytes = source_bytes
            .checked_add(committed_text_host_bytes)
            .ok_or_else(|| {
                Error::ModelLoadError("Voxtral committed host peak overflowed".into())
            })?;
        let (max_host_bytes, max_tensor_bytes) = checked_realtime_stream_peak_bytes(
            frontend_host_bytes,
            transactional_text_host_bytes,
            embedding_bytes,
            seal.max_workspace_bytes,
        )?;
        Ok(VoxtralRealtimeStreamPeakReservation {
            max_source_samples: seal.max_source_samples,
            max_host_bytes,
            max_tensor_bytes,
            max_preparation_scratch_bytes: seal.max_workspace_bytes,
            max_committed_host_bytes,
        })
    }

    pub(crate) fn realtime_preparation_geometry_for_source_samples(
        &self,
        source_samples: usize,
        sample_rate: u32,
        mode: VoxtralRealtimePreparationMode,
    ) -> Result<VoxtralRealtimePreparationGeometry> {
        if source_samples == 0 || source_samples > self.realtime_max_source_samples {
            return Err(Error::InvalidInput(
                "Voxtral cumulative source samples are outside the sealed realtime range".into(),
            ));
        }
        realtime_preparation_geometry_for(source_samples, sample_rate, &self.layout, mode)
    }

    pub(crate) fn realtime_prepared_resource_usage(
        &self,
        geometry: VoxtralRealtimePreparationGeometry,
    ) -> Result<VoxtralRealtimePreparedResourceUsage> {
        let source_bytes = u64::try_from(geometry.source_samples)
            .ok()
            .and_then(|samples| samples.checked_mul(std::mem::size_of::<f32>() as u64))
            .ok_or_else(|| Error::Overloaded("Voxtral prepared host usage overflowed".into()))?;
        let (text_bytes, _) =
            checked_realtime_text_host_bytes(geometry.pooled_frames, self.max_decoded_token_bytes)?;
        let host_bytes = source_bytes
            .checked_add(text_bytes)
            .ok_or_else(|| Error::Overloaded("Voxtral prepared host usage overflowed".into()))?;
        let tensor_bytes = geometry
            .embedding_elements
            .checked_mul(self.dtype.size_in_bytes() as u64)
            .ok_or_else(|| Error::Overloaded("Voxtral prepared tensor usage overflowed".into()))?;
        Ok(VoxtralRealtimePreparedResourceUsage {
            host_bytes,
            tensor_bytes,
        })
    }

    pub(crate) fn prepare_realtime_audio_batch(
        &self,
        rows: &[VoxtralRealtimePreparationBatchRow<'_>],
    ) -> Result<Vec<VoxtralRealtimePreparedAudio>> {
        if rows.is_empty() {
            return Err(Error::InvalidInput(
                "Voxtral preparation batch must contain at least one row".into(),
            ));
        }
        let mut inputs = Vec::with_capacity(rows.len());
        let mut geometries = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            if rows[..index]
                .iter()
                .any(|previous| previous.state.state_id == row.state.state_id)
            {
                return Err(Error::InvalidInput(
                    "Voxtral preparation batch contains the same stream more than once".into(),
                ));
            }
            let geometry = self.realtime_preparation_geometry(
                row.state,
                row.appended_samples.len(),
                row.sample_rate,
                row.mode,
            )?;
            let mut source = Vec::with_capacity(geometry.source_samples);
            source.extend_from_slice(row.state.source_samples.as_slice());
            source.extend_from_slice(row.appended_samples);
            let target_rate = self.mel.config().sample_rate as u32;
            let audio = if row.sample_rate == target_rate {
                Cow::Borrowed(source.as_slice())
            } else {
                Cow::Owned(resample_audio(&source, row.sample_rate, target_rate)?)
            };
            if audio.len() != geometry.resampled_samples {
                return Err(Error::InferenceError(
                    "Voxtral resampler violated preparation geometry".into(),
                ));
            }
            let left_pad = self
                .layout
                .raw_audio_length_per_tok
                .checked_mul(self.layout.left_pad_tokens)
                .ok_or_else(|| Error::InvalidInput("Voxtral left padding overflow".into()))?;
            let right_pad = geometry
                .padded_samples
                .checked_sub(left_pad)
                .and_then(|len| len.checked_sub(audio.len()))
                .ok_or_else(|| {
                    Error::InferenceError("Voxtral padding geometry underflow".into())
                })?;
            let mut padded = Vec::with_capacity(geometry.padded_samples);
            padded.extend(std::iter::repeat_n(0.0, left_pad));
            padded.extend_from_slice(&audio);
            padded.extend(std::iter::repeat_n(0.0, right_pad));
            let mut mel_spec = self.mel.compute(&padded)?;
            drop_last_mel_frame_for_voxtral(&mut mel_spec);
            if let Some(max_val) = self.global_log_mel_max {
                normalize_log_mel_with_max(&mut mel_spec, max_val);
            }
            if mel_spec.len() != geometry.mel_frames {
                return Err(Error::InferenceError(format!(
                    "Voxtral mel frontend produced {} frames, expected {}",
                    mel_spec.len(),
                    geometry.mel_frames
                )));
            }
            inputs.push((Arc::new(source), mel_spec));
            geometries.push(geometry);
        }

        let batch_geometry = self.realtime_preparation_batch_geometry(&geometries)?;
        let mut mel_rows = Vec::with_capacity(rows.len());
        for (_, mel_spec) in &inputs {
            let mel = mel_frames_to_tensor(
                mel_spec,
                self.mel.config().n_mels,
                &self.device.device,
                self.dtype,
            )?;
            let padding = batch_geometry
                .padded_mel_frames
                .checked_sub(mel.dim(1)?)
                .ok_or_else(|| Error::InferenceError("Voxtral mel padding underflow".into()))?;
            let mel = if padding == 0 {
                mel
            } else {
                let zeros = Tensor::zeros(
                    (1, padding, self.mel.config().n_mels),
                    self.dtype,
                    &self.device.device,
                )?;
                Tensor::cat(&[&mel, &zeros], 1)?
            };
            mel_rows.push(mel);
        }
        let encoder = if mel_rows.len() == 1 {
            self.whisper_encoder.forward(&mel_rows[0])?
        } else {
            let refs = mel_rows.iter().collect::<Vec<_>>();
            let mel = Tensor::cat(&refs, 0)?;
            let lengths = geometries
                .iter()
                .map(|geometry| geometry.mel_frames)
                .collect::<Vec<_>>();
            self.whisper_encoder.forward_valid_lengths(&mel, &lengths)?
        };

        let mut prepared = Vec::with_capacity(rows.len());
        for (index, row) in rows.iter().enumerate() {
            let geometry = geometries[index];
            let encoded = encoder
                .i(index)?
                .unsqueeze(0)?
                .narrow(1, 0, geometry.conv2_frames)?;
            let pooled = self.pool_audio_embeddings(&encoded)?;
            let embeddings = self.audio_adapter.forward(&pooled)?;
            if embeddings.dim(1)? != geometry.pooled_frames {
                return Err(Error::InferenceError(
                    "Voxtral adapter violated preparation geometry".into(),
                ));
            }
            prepared.push(VoxtralRealtimePreparedAudio {
                model_load_nonce: self.model_load_nonce,
                state_id: row.state.state_id,
                expected_source_samples: row.state.source_samples.len(),
                expected_source_identity: row.state.source_samples.clone(),
                preparation_generation: row.state.preparation_generation,
                source_sample_rate: row.sample_rate,
                source_samples: inputs[index].0.clone(),
                mode: row.mode,
                geometry,
                embeddings,
            });
        }
        Ok(prepared)
    }

    pub(crate) fn install_realtime_audio_preparation(
        &self,
        state: &mut VoxtralRealtimeState,
        prepared: VoxtralRealtimePreparedAudio,
    ) -> Result<usize> {
        if prepared.model_load_nonce != self.model_load_nonce
            || prepared.state_id != state.state_id
            || prepared.expected_source_samples != state.source_samples.len()
            || !Arc::ptr_eq(&prepared.expected_source_identity, &state.source_samples)
            || prepared.preparation_generation != state.preparation_generation
            || state
                .source_sample_rate
                .is_some_and(|rate| rate != prepared.source_sample_rate)
            || state.input_closed
            || state.finished
        {
            return Err(Error::InferenceError(
                "Voxtral prepared audio is foreign, stale, or out of order".into(),
            ));
        }
        let stable_frames = prepared.geometry.stable_frames;
        if stable_frames < state.prepared_audio_frames || stable_frames < state.next_audio_frame {
            return Err(Error::InferenceError(format!(
                "Voxtral causal encoder prefix regressed from {} to {stable_frames} frames",
                state.prepared_audio_frames
            )));
        }
        let added = stable_frames - state.prepared_audio_frames;
        state.preparation_generation =
            state.preparation_generation.checked_add(1).ok_or_else(|| {
                Error::InferenceError("Voxtral preparation generation overflow".into())
            })?;
        state.source_sample_rate = Some(prepared.source_sample_rate);
        state.source_samples = prepared.source_samples;
        state.audio_embeds = Some(prepared.embeddings);
        state.prepared_audio_frames = stable_frames;
        if prepared.mode == VoxtralRealtimePreparationMode::Finish {
            state.input_closed = true;
            state.final_padding_applied = true;
        }
        Ok(added)
    }

    fn validate_realtime_preparation_row(
        &self,
        state: &VoxtralRealtimeState,
        appended_samples: usize,
        sample_rate: u32,
        mode: VoxtralRealtimePreparationMode,
    ) -> Result<()> {
        if state.input_closed || state.finished || state.final_padding_applied {
            return Err(Error::InvalidInput(
                "Voxtral realtime input is already closed or finished".into(),
            ));
        }
        if sample_rate == 0 {
            return Err(Error::InvalidInput(
                "Voxtral realtime sample rate must be non-zero".into(),
            ));
        }
        if state
            .source_sample_rate
            .is_some_and(|rate| rate != sample_rate)
        {
            return Err(Error::InvalidInput(
                "Voxtral realtime sample rate changed within one stream".into(),
            ));
        }
        if mode == VoxtralRealtimePreparationMode::Push && appended_samples == 0 {
            return Err(Error::InvalidInput(
                "Voxtral realtime push requires non-empty audio".into(),
            ));
        }
        let total = state
            .source_samples
            .len()
            .checked_add(appended_samples)
            .ok_or_else(|| Error::InvalidInput("Voxtral realtime audio length overflow".into()))?;
        validate_realtime_sample_ceiling(
            total,
            sample_rate,
            self.layout.target_rate,
            self.realtime_max_source_samples,
        )?;
        Ok(())
    }

    fn accept_realtime_prediction(
        &self,
        state: &mut VoxtralRealtimeState,
        token: u32,
    ) -> Result<VoxtralRealtimeStep> {
        let specials = self.tokenizer.specials();
        if token == specials.eos || Some(token) == specials.end_audio {
            state.pending_input_token = None;
            return self.finish_realtime_text(state);
        }
        let mut no_callback = None;
        let delta = append_generated_token(
            &self.tokenizer,
            &mut state.generated,
            &mut state.assembled,
            token,
            &mut no_callback,
        )?;
        state.pending_input_token = Some(token);
        Ok(self.realtime_step(state, delta))
    }

    /// Finish the stream, flushing any held-back partial character into the
    /// final delta.
    fn finish_realtime_text(
        &self,
        state: &mut VoxtralRealtimeState,
    ) -> Result<VoxtralRealtimeStep> {
        let mut no_callback = None;
        let delta = flush_generated_text(
            &self.tokenizer,
            &state.generated,
            &mut state.assembled,
            &mut no_callback,
        )?;
        state.finished = true;
        Ok(self.realtime_step(state, delta))
    }

    fn realtime_step(&self, state: &VoxtralRealtimeState, delta: String) -> VoxtralRealtimeStep {
        VoxtralRealtimeStep {
            delta,
            text: state.text().to_string(),
            tokens_generated: state.generated.len(),
            finished: state.finished,
        }
    }

    fn cache_diagnostics(&self, cache: &PhysicalPagedKvCache) -> serde_json::Value {
        serde_json::json!({
            "page_size": cache.arena().config().page_tokens,
            "dense_decode_enabled": false,
            "dense_decode_max_tokens": 0,
            "kv_quantization": "none"
        })
    }

    fn execution_diagnostics(
        &self,
        input_samples: usize,
        resampled_samples: usize,
        padded_samples: usize,
        mel_frames: usize,
        audio_frames: usize,
        decode_frames: usize,
        prompt_tokens: usize,
        generated_tokens: usize,
        cache: serde_json::Value,
        timings: Option<VoxtralRealtimeTimings>,
    ) -> serde_json::Value {
        serde_json::json!({
            "model_family": "voxtral",
            "audio": {
                "input_samples": input_samples,
                "resampled_sample_rate": 16000,
                "resampled_samples": resampled_samples,
                "padded_samples": padded_samples,
                "mel_frames": mel_frames,
                "audio_frames": audio_frames
            },
            "decode": {
                "frame_synchronous": true,
                "decode_frames": decode_frames,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens
            },
            "execution": {
                "device_kind": format!("{:?}", self.device.kind),
                "dtype": format!("{:?}", self.dtype),
                "precision": self.precision.diagnostics(),
                "flash_attention_requested": flash_attention_requested(),
                "flash_attention_compiled": flash_attention_compiled(),
                "cache": cache,
                "timings_ms": timings.map(VoxtralRealtimeTimings::to_json)
            },
            "config": {
                "num_delay_tokens": self.num_delay_tokens,
                "streaming_left_pad_tokens": self.streaming_left_pad_tokens,
                "offline_left_pad_tokens": self.layout.left_pad_tokens,
                "streaming_right_pad_tokens": self.layout.right_pad_tokens,
                "raw_audio_length_per_tok": self.layout.raw_audio_length_per_tok,
                "block_pool_size": self.layout.pool_size,
                "audio_length_per_tok": self.audio_length_per_tok
            }
        })
    }

    fn log_transcription_timings(
        &self,
        timings: &VoxtralRealtimeTimings,
        generated_tokens: usize,
        prompt_tokens: usize,
        decode_frames: usize,
    ) {
        info!(
            "Voxtral Realtime timings: generated_tokens={}, prompt_tokens={}, decode_frames={}, preprocess={:.2}ms, encoder_adapter={:.2}ms, lm_prefill={:.2}ms, token_decode={:.2}ms, total={:.2}ms",
            generated_tokens,
            prompt_tokens,
            decode_frames,
            duration_ms(timings.preprocess),
            duration_ms(timings.encoder_adapter),
            duration_ms(timings.lm_prefill),
            duration_ms(timings.token_decode),
            duration_ms(timings.total)
        );
    }

    /// Pool audio embeddings by block_size
    fn pool_audio_embeddings(&self, audio_embeds: &Tensor) -> Result<Tensor> {
        pool_audio_embeddings_by_block(audio_embeds, self.layout.pool_size)
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct VoxtralRealtimeTimings {
    preprocess: Duration,
    encoder_adapter: Duration,
    lm_prefill: Duration,
    token_decode: Duration,
    total: Duration,
}

impl VoxtralRealtimeTimings {
    fn to_json(self) -> serde_json::Value {
        serde_json::json!({
            "preprocess": duration_ms(self.preprocess),
            "encoder_adapter": duration_ms(self.encoder_adapter),
            "lm_prefill": duration_ms(self.lm_prefill),
            "token_decode": duration_ms(self.token_decode),
            "total": duration_ms(self.total)
        })
    }
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn pool_audio_embeddings_by_block(audio_embeds: &Tensor, pool_size: usize) -> Result<Tensor> {
    let (bsz, seq_len, hidden) = audio_embeds.dims3()?;
    let pool_size = pool_size.max(1);

    // Ensure seq_len is divisible by pool_size, keeping the left padding/prefix
    // alignment intact and dropping only trailing frames.
    let new_len = seq_len / pool_size;
    let truncated_len = new_len * pool_size;

    if truncated_len < seq_len {
        let audio_embeds = audio_embeds.narrow(1, 0, truncated_len)?;
        let reshaped = audio_embeds.reshape((bsz, new_len, hidden * pool_size))?;
        Ok(reshaped)
    } else {
        let reshaped = audio_embeds.reshape((bsz, new_len, hidden * pool_size))?;
        Ok(reshaped)
    }
}

/// One stem convolution. The zero padding is applied explicitly before the
/// convolution (whose own padding is 0), so it can be one-sided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Conv1dGeometry {
    kernel: usize,
    stride: usize,
    left_padding: usize,
    right_padding: usize,
    dilation: usize,
}

impl Conv1dGeometry {
    /// Symmetric padding of one frame, as in the classic Whisper stem.
    fn symmetric(kernel: usize, stride: usize) -> Self {
        Self {
            kernel,
            stride,
            left_padding: 1,
            right_padding: 1,
            dilation: 1,
        }
    }

    /// Causal padding as in HF `VoxtralRealtimeCausalConv1d` and vLLM
    /// `WhisperCausalConv1d`: `(k - 1) * d + 1 - stride` zeros on the left
    /// only (here `d = 1`), so output `j` reads input frames no later than
    /// `(j + 1) * stride - 1`, the end of its own stride block.
    fn causal(kernel: usize, stride: usize) -> Self {
        Self {
            kernel,
            stride,
            left_padding: kernel.saturating_sub(stride),
            right_padding: 0,
            dilation: 1,
        }
    }
}

/// The encoder's two-convolution stem, shared by the loaded encoder and the
/// weights-free realtime layout. Voxtral Realtime's stem is causal; the
/// classic Whisper branch keeps symmetric padding.
fn conv_stem_geometry(
    cfg: &super::config::AudioEncoderConfig,
    is_voxtral: bool,
) -> (Conv1dGeometry, Conv1dGeometry) {
    let conv = if is_voxtral {
        Conv1dGeometry::causal
    } else {
        Conv1dGeometry::symmetric
    };
    (
        conv(cfg.conv1_kernel_size, cfg.conv1_stride),
        conv(cfg.conv2_kernel_size, cfg.conv2_stride),
    )
}

/// Apply one stem convolution with its explicit zero padding.
fn conv1d_with_geometry(
    conv: &candle_nn::Conv1d,
    spec: Conv1dGeometry,
    x: &Tensor,
) -> Result<Tensor> {
    let x = if spec.left_padding == 0 && spec.right_padding == 0 {
        x.clone()
    } else {
        x.pad_with_zeros(D::Minus1, spec.left_padding, spec.right_padding)?
    };
    conv.forward(&x).map_err(Error::from)
}

fn conv_output_length(input: usize, spec: Conv1dGeometry) -> Result<usize> {
    if spec.kernel == 0 || spec.stride == 0 || spec.dilation == 0 {
        return Err(Error::ModelLoadError(
            "Voxtral encoder has invalid convolution geometry".into(),
        ));
    }
    let padded = input
        .checked_add(spec.left_padding)
        .and_then(|length| length.checked_add(spec.right_padding))
        .ok_or_else(|| Error::InvalidInput("Voxtral convolution length overflow".into()))?;
    let receptive = spec
        .dilation
        .checked_mul(spec.kernel - 1)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| Error::InvalidInput("Voxtral convolution extent overflow".into()))?;
    if padded < receptive {
        return Ok(0);
    }
    Ok((padded - receptive) / spec.stride + 1)
}

fn validate_valid_lengths(batch: usize, frames: usize, lengths: &[usize]) -> Result<()> {
    if lengths.len() != batch
        || lengths
            .iter()
            .any(|length| *length == 0 || *length > frames)
    {
        return Err(Error::InvalidInput(format!(
            "Voxtral valid lengths {:?} do not describe batch {batch} padded to {frames}",
            lengths
        )));
    }
    Ok(())
}

fn frame_valid_mask(lengths: &[usize], frames: usize, reference: &Tensor) -> Result<Tensor> {
    let mut mask = Vec::with_capacity(lengths.len() * frames);
    for length in lengths {
        mask.extend((0..frames).map(|frame| u8::from(frame < *length) as f32));
    }
    Tensor::from_vec(mask, (lengths.len(), frames, 1), reference.device())?
        .to_dtype(reference.dtype())
        .map_err(Error::from)
}

fn channel_valid_mask(lengths: &[usize], frames: usize, reference: &Tensor) -> Result<Tensor> {
    frame_valid_mask(lengths, frames, reference)?
        .transpose(1, 2)
        .map_err(Error::from)
}

/// Whisper encoder for audio processing
pub struct WhisperEncoder {
    conv1: candle_nn::Conv1d,
    conv2: candle_nn::Conv1d,
    layers: Vec<WhisperEncoderLayer>,
    ln_post: Option<candle_nn::LayerNorm>,
    ln_post_rms: Option<candle_nn::RmsNorm>,
    embed_positions: Option<Tensor>,
    is_causal: bool,
    conv1_spec: Conv1dGeometry,
    conv2_spec: Conv1dGeometry,
}

impl WhisperEncoder {
    /// Load for standard Whisper format
    pub fn load(cfg: &super::config::AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Self::load_internal(cfg, vb, false)
    }

    /// Load for Voxtral checkpoint format (different tensor naming)
    pub fn load_voxtral(cfg: &super::config::AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Self::load_internal(cfg, vb, true)
    }

    fn load_internal(
        cfg: &super::config::AudioEncoderConfig,
        vb: VarBuilder,
        is_voxtral: bool,
    ) -> Result<Self> {
        let (conv1_spec, conv2_spec) = conv_stem_geometry(cfg, is_voxtral);
        // Padding is applied by `conv1d_with_geometry`, never by the conv.
        let conv1_config = candle_nn::Conv1dConfig {
            stride: conv1_spec.stride,
            padding: 0,
            groups: 1,
            dilation: conv1_spec.dilation,
            ..Default::default()
        };
        let conv2_config = candle_nn::Conv1dConfig {
            stride: conv2_spec.stride,
            padding: 0,
            groups: 1,
            dilation: conv2_spec.dilation,
            ..Default::default()
        };

        // Voxtral uses conv_layers.0.conv and conv_layers.1.conv
        let (conv1, conv2) = if is_voxtral {
            let conv1 = candle_nn::conv1d(
                cfg.num_mel_bins,
                cfg.d_model,
                cfg.conv1_kernel_size,
                conv1_config,
                vb.pp("conv_layers.0.conv"),
            )?;
            let conv2 = candle_nn::conv1d(
                cfg.d_model,
                cfg.d_model,
                cfg.conv2_kernel_size,
                conv2_config,
                vb.pp("conv_layers.1.conv"),
            )?;
            (conv1, conv2)
        } else {
            let conv1 = candle_nn::conv1d(
                cfg.num_mel_bins,
                cfg.d_model,
                cfg.conv1_kernel_size,
                conv1_config,
                vb.pp("conv1"),
            )?;
            let conv2 = candle_nn::conv1d(
                cfg.d_model,
                cfg.d_model,
                cfg.conv2_kernel_size,
                conv2_config,
                vb.pp("conv2"),
            )?;
            (conv1, conv2)
        };

        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for i in 0..cfg.encoder_layers {
            // Voxtral uses transformer.layers.{i} instead of layers.{i}
            let layer_vb = if is_voxtral {
                vb.pp(format!("transformer.layers.{i}"))
            } else {
                vb.pp(format!("layers.{i}"))
            };
            layers.push(WhisperEncoderLayer::load(cfg, layer_vb, is_voxtral)?);
        }

        let (ln_post, ln_post_rms) = if is_voxtral {
            // Voxtral uses transformer.norm (RMSNorm)
            let norm = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("transformer.norm"))?;
            (None, Some(norm))
        } else {
            let norm = candle_nn::layer_norm(cfg.d_model, 1e-5, vb.pp("ln_post"))?;
            (Some(norm), None)
        };

        let use_rope_positions = is_voxtral && cfg.pos_embed.eq_ignore_ascii_case("rope");
        let embed_positions = if use_rope_positions {
            None
        } else {
            Some(build_sinusoidal_positions(
                cfg.max_source_positions,
                cfg.d_model,
                vb.device(),
            )?)
        };

        Ok(Self {
            conv1,
            conv2,
            layers,
            ln_post,
            ln_post_rms,
            embed_positions,
            is_causal: cfg.is_causal,
            conv1_spec,
            conv2_spec,
        })
    }

    pub fn forward(&self, input_features: &Tensor) -> Result<Tensor> {
        // input_features: [batch, frames, n_mels]
        // Transpose to [batch, n_mels, frames] for conv1d
        let x = input_features.transpose(1, 2)?;

        // Conv layers with gelu
        let x = gelu(&conv1d_with_geometry(&self.conv1, self.conv1_spec, &x)?)?;
        let x = gelu(&conv1d_with_geometry(&self.conv2, self.conv2_spec, &x)?)?;

        // Transpose back: [batch, hidden, frames] -> [batch, frames, hidden]
        let x = x.transpose(1, 2)?;

        // Add absolute positional embeddings for classic Whisper. Voxtral Realtime uses RoPE
        // inside each causal attention layer instead.
        let seq_len = x.dim(1)?;
        let x = if let Some(embed_positions) = &self.embed_positions {
            let pos_embed = embed_positions.narrow(0, 0, seq_len)?;
            let pos_embed = pos_embed.unsqueeze(0)?.broadcast_as(x.shape())?;
            x.broadcast_add(&pos_embed)?
        } else {
            x
        };

        // Transformer layers
        let mut x = x;
        let tables = EncoderForwardTables::default();
        for layer in &self.layers {
            x = layer.forward(&x, self.is_causal, &tables)?;
        }

        // Final layer norm
        if let Some(ln_post_rms) = &self.ln_post_rms {
            ln_post_rms.forward(&x)
        } else {
            self.ln_post.as_ref().unwrap().forward(&x)
        }
        .map_err(|e| Error::InferenceError(e.to_string()))
    }

    fn forward_valid_lengths(
        &self,
        input_features: &Tensor,
        valid_lengths: &[usize],
    ) -> Result<Tensor> {
        let (batch, frames, _) = input_features.dims3()?;
        validate_valid_lengths(batch, frames, valid_lengths)?;
        let input_mask = frame_valid_mask(valid_lengths, frames, input_features)?;
        let x = input_features.broadcast_mul(&input_mask)?.transpose(1, 2)?;

        let conv1_lengths = valid_lengths
            .iter()
            .map(|length| conv_output_length(*length, self.conv1_spec))
            .collect::<Result<Vec<_>>>()?;
        let x = gelu(&conv1d_with_geometry(&self.conv1, self.conv1_spec, &x)?)?;
        let conv1_frames = x.dim(2)?;
        let conv1_mask = channel_valid_mask(&conv1_lengths, conv1_frames, &x)?;
        let x = x.broadcast_mul(&conv1_mask)?;

        let conv2_lengths = conv1_lengths
            .iter()
            .map(|length| conv_output_length(*length, self.conv2_spec))
            .collect::<Result<Vec<_>>>()?;
        let x = gelu(&conv1d_with_geometry(&self.conv2, self.conv2_spec, &x)?)?;
        let conv2_frames = x.dim(2)?;
        let conv2_channel_mask = channel_valid_mask(&conv2_lengths, conv2_frames, &x)?;
        let x = x.broadcast_mul(&conv2_channel_mask)?.transpose(1, 2)?;
        let frame_mask = frame_valid_mask(&conv2_lengths, conv2_frames, &x)?;

        let mut x = if let Some(embed_positions) = &self.embed_positions {
            let pos_embed = embed_positions.narrow(0, 0, conv2_frames)?;
            let pos_embed = pos_embed.unsqueeze(0)?.broadcast_as(x.shape())?;
            x.broadcast_add(&pos_embed)?.broadcast_mul(&frame_mask)?
        } else {
            x
        };
        let tables = EncoderForwardTables::default();
        for layer in &self.layers {
            x = layer.forward_valid_lengths(
                &x,
                self.is_causal,
                &conv2_lengths,
                &frame_mask,
                &tables,
            )?;
        }
        let x = if let Some(ln_post_rms) = &self.ln_post_rms {
            ln_post_rms.forward(&x)?
        } else {
            self.ln_post.as_ref().unwrap().forward(&x)?
        };
        x.broadcast_mul(&frame_mask).map_err(Error::from)
    }

    /// Forward only conv layers (for realtime processing)
    pub fn forward_conv(&self, mel_features: &[Tensor]) -> Result<Tensor> {
        if mel_features.is_empty() {
            return Err(Error::InvalidInput(
                "Voxtral realtime forward_conv requires at least one mel feature tensor"
                    .to_string(),
            ));
        }

        let mut outputs = Vec::with_capacity(mel_features.len());

        for mel in mel_features {
            // mel: [n_mels, seq_len] -> [1, n_mels, seq_len]
            let x = prepare_realtime_conv_input(mel)?;
            let x = gelu(&conv1d_with_geometry(&self.conv1, self.conv1_spec, &x)?)?;
            let x = gelu(&conv1d_with_geometry(&self.conv2, self.conv2_spec, &x)?)?;
            // [1, hidden, seq_len] -> [hidden, seq_len]
            let x = x.squeeze(0)?;
            outputs.push(x);
        }

        // Concatenate along sequence dimension
        let outputs_refs: Vec<&Tensor> = outputs.iter().collect();
        Tensor::cat(&outputs_refs, 1).map_err(|e| Error::InferenceError(e.to_string()))
    }
}

fn prepare_realtime_conv_input(mel: &Tensor) -> Result<Tensor> {
    mel.dims2().map_err(Error::from)?;
    mel.unsqueeze(0).map_err(Error::from)
}

fn build_sinusoidal_positions(
    max_len: usize,
    hidden: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let half_hidden = hidden / 2;
    let log_theta = (10000f32).ln() / (half_hidden as f32 - 1.0);
    let inv_freq: Vec<f32> = (0..half_hidden)
        .map(|i| (-log_theta * i as f32).exp())
        .collect();

    let mut pos_embed_data = Vec::with_capacity(max_len * hidden);
    for pos in 0..max_len {
        for timescale in inv_freq.iter().take(half_hidden) {
            pos_embed_data.push((pos as f32 * timescale).sin());
        }
        for timescale in inv_freq.iter().take(half_hidden) {
            pos_embed_data.push((pos as f32 * timescale).cos());
        }
    }

    Tensor::from_vec(pos_embed_data, (max_len, hidden), device).map_err(Error::from)
}

/// Whisper encoder layer
struct WhisperEncoderLayer {
    self_attn_layer_norm: Option<candle_nn::LayerNorm>,
    final_layer_norm: Option<candle_nn::LayerNorm>,
    self_attn_rms_norm: Option<candle_nn::RmsNorm>,
    final_rms_norm: Option<candle_nn::RmsNorm>,
    self_attn: WhisperAttention,
    fc1: Option<candle_nn::Linear>,
    fc2: Option<candle_nn::Linear>,
    ffn_w1: Option<candle_nn::Linear>,
    ffn_w2: Option<candle_nn::Linear>,
    ffn_w3: Option<candle_nn::Linear>,
    is_voxtral: bool,
}

impl WhisperEncoderLayer {
    pub fn load(
        cfg: &super::config::AudioEncoderConfig,
        vb: VarBuilder,
        is_voxtral: bool,
    ) -> Result<Self> {
        let (self_attn_layer_norm, self_attn_rms_norm, self_attn, final_layer_norm, final_rms_norm) =
            if is_voxtral {
                let norm = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("attention_norm"))?;
                let attn = WhisperAttention::load_voxtral(cfg, vb.pp("attention"))?;
                let ffn_norm = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("ffn_norm"))?;
                (None, Some(norm), attn, None, Some(ffn_norm))
            } else {
                let norm = candle_nn::layer_norm(cfg.d_model, 1e-5, vb.pp("self_attn_layer_norm"))?;
                let attn = WhisperAttention::load(cfg, vb.pp("self_attn"))?;
                let ffn_norm = candle_nn::layer_norm(cfg.d_model, 1e-5, vb.pp("final_layer_norm"))?;
                (Some(norm), None, attn, Some(ffn_norm), None)
            };

        let (fc1, fc2, ffn_w1, ffn_w2, ffn_w3) = if is_voxtral {
            // Voxtral audio uses SwiGLU with a biased down projection.
            let w1 = candle_nn::linear_no_bias(
                cfg.d_model,
                cfg.encoder_ffn_dim,
                vb.pp("feed_forward.w1"),
            )?;
            let w2 = candle_nn::linear(cfg.encoder_ffn_dim, cfg.d_model, vb.pp("feed_forward.w2"))?;
            let w3 = candle_nn::linear_no_bias(
                cfg.d_model,
                cfg.encoder_ffn_dim,
                vb.pp("feed_forward.w3"),
            )?;
            (None, None, Some(w1), Some(w2), Some(w3))
        } else {
            let fc1 = candle_nn::linear(cfg.d_model, cfg.encoder_ffn_dim, vb.pp("fc1"))?;
            let fc2 = candle_nn::linear(cfg.encoder_ffn_dim, cfg.d_model, vb.pp("fc2"))?;
            (Some(fc1), Some(fc2), None, None, None)
        };

        Ok(Self {
            self_attn_layer_norm,
            self_attn_rms_norm,
            self_attn,
            final_layer_norm,
            final_rms_norm,
            fc1,
            fc2,
            ffn_w1,
            ffn_w2,
            ffn_w3,
            is_voxtral,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        is_causal: bool,
        tables: &EncoderForwardTables,
    ) -> Result<Tensor> {
        let residual = x;
        let x = if self.is_voxtral {
            self.self_attn_rms_norm.as_ref().unwrap().forward(x)?
        } else {
            self.self_attn_layer_norm.as_ref().unwrap().forward(x)?
        };
        let x = self.self_attn.forward(&x, is_causal, tables)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = if self.is_voxtral {
            self.final_rms_norm.as_ref().unwrap().forward(&x)?
        } else {
            self.final_layer_norm.as_ref().unwrap().forward(&x)?
        };

        // FFN: Voxtral uses SwiGLU (w1, w2, w3), standard uses GELU (fc1, fc2)
        let x = if self.is_voxtral {
            let w1_out = self.ffn_w1.as_ref().unwrap().forward(&x)?;
            let w3_out = self.ffn_w3.as_ref().unwrap().forward(&x)?;
            // SwiGLU: silu(w1) * w3
            let silu_w1 = candle_nn::ops::silu(&w1_out)?;
            let gated = silu_w1.broadcast_mul(&w3_out)?;
            self.ffn_w2.as_ref().unwrap().forward(&gated)?
        } else {
            let x = self.fc1.as_ref().unwrap().forward(&x)?;
            let x = gelu(&x)?;
            self.fc2.as_ref().unwrap().forward(&x)?
        };

        residual
            .broadcast_add(&x)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }

    fn forward_valid_lengths(
        &self,
        x: &Tensor,
        is_causal: bool,
        valid_lengths: &[usize],
        frame_mask: &Tensor,
        tables: &EncoderForwardTables,
    ) -> Result<Tensor> {
        let residual = x.broadcast_mul(frame_mask)?;
        let normalized = if self.is_voxtral {
            self.self_attn_rms_norm
                .as_ref()
                .unwrap()
                .forward(&residual)?
        } else {
            self.self_attn_layer_norm
                .as_ref()
                .unwrap()
                .forward(&residual)?
        }
        .broadcast_mul(frame_mask)?;
        let attention = self
            .self_attn
            .forward_valid_lengths(&normalized, is_causal, valid_lengths, frame_mask, tables)?
            .broadcast_mul(frame_mask)?;
        let x = residual
            .broadcast_add(&attention)?
            .broadcast_mul(frame_mask)?;

        let residual = &x;
        let normalized = if self.is_voxtral {
            self.final_rms_norm.as_ref().unwrap().forward(&x)?
        } else {
            self.final_layer_norm.as_ref().unwrap().forward(&x)?
        }
        .broadcast_mul(frame_mask)?;
        let feed_forward = if self.is_voxtral {
            let w1 = self.ffn_w1.as_ref().unwrap().forward(&normalized)?;
            let w3 = self.ffn_w3.as_ref().unwrap().forward(&normalized)?;
            let gated = candle_nn::ops::silu(&w1)?.broadcast_mul(&w3)?;
            self.ffn_w2.as_ref().unwrap().forward(&gated)?
        } else {
            let hidden = gelu(&self.fc1.as_ref().unwrap().forward(&normalized)?)?;
            self.fc2.as_ref().unwrap().forward(&hidden)?
        }
        .broadcast_mul(frame_mask)?;
        residual
            .broadcast_add(&feed_forward)?
            .broadcast_mul(frame_mask)
            .map_err(Error::from)
    }
}

/// Whisper attention with optional causal masking
struct WhisperAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    out_proj: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    use_rope_positions: bool,
    rope_theta: f64,
    sliding_window: Option<usize>,
}

impl WhisperAttention {
    /// Load for standard Whisper format  
    pub fn load(cfg: &super::config::AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Self::load_internal(cfg, vb, false)
    }

    /// Load for Voxtral checkpoint format
    pub fn load_voxtral(cfg: &super::config::AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Self::load_internal(cfg, vb, true)
    }

    fn load_internal(
        cfg: &super::config::AudioEncoderConfig,
        vb: VarBuilder,
        is_voxtral: bool,
    ) -> Result<Self> {
        // Use explicit head_dim from config if available, otherwise compute
        let head_dim = if cfg.head_dim > 0 {
            cfg.head_dim
        } else {
            cfg.d_model / cfg.encoder_attention_heads
        };
        let qkv_proj_dim = cfg.encoder_attention_heads * head_dim;

        let (q_proj, k_proj, v_proj, out_proj) = if is_voxtral {
            // Voxtral uses wq, wk, wv with shape [n_heads * head_dim, d_model]
            // Note: wk has no bias, others have bias
            let q = candle_nn::linear(cfg.d_model, qkv_proj_dim, vb.pp("wq"))?;
            let k = candle_nn::linear_no_bias(cfg.d_model, qkv_proj_dim, vb.pp("wk"))?;
            let v = candle_nn::linear(cfg.d_model, qkv_proj_dim, vb.pp("wv"))?;
            let out = candle_nn::linear(qkv_proj_dim, cfg.d_model, vb.pp("wo"))?;
            (q, k, v, out)
        } else {
            // Standard Whisper uses q_proj, k_proj, v_proj, out_proj
            let q = candle_nn::linear(cfg.d_model, qkv_proj_dim, vb.pp("q_proj"))?;
            let k = candle_nn::linear(cfg.d_model, qkv_proj_dim, vb.pp("k_proj"))?;
            let v = candle_nn::linear(cfg.d_model, qkv_proj_dim, vb.pp("v_proj"))?;
            let out = candle_nn::linear(qkv_proj_dim, cfg.d_model, vb.pp("out_proj"))?;
            (q, k, v, out)
        };

        let scale = (head_dim as f64).powf(-0.5);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: cfg.encoder_attention_heads,
            head_dim,
            scale,
            use_rope_positions: is_voxtral && cfg.pos_embed.eq_ignore_ascii_case("rope"),
            rope_theta: cfg.rope_theta,
            sliding_window: (cfg.is_causal && cfg.sliding_window > 0).then_some(cfg.sliding_window),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        is_causal: bool,
        tables: &EncoderForwardTables,
    ) -> Result<Tensor> {
        let (bsz, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention
        let mut q = q.reshape((bsz, seq_len, self.num_heads, self.head_dim))?;
        let mut k = k.reshape((bsz, seq_len, self.num_heads, self.head_dim))?;
        let v = v
            .reshape((bsz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        if self.use_rope_positions {
            let (cos, sin) = tables.rope(seq_len, self.head_dim, self.rope_theta, &q)?;
            q = apply_interleaved_rotary_emb(&q, &cos, &sin)?;
            k = apply_interleaved_rotary_emb(&k, &cos, &sin)?;
        }

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;

        if self.sliding_window.is_none() {
            if let Some(fused_out) =
                try_fused_self_attention(&q, &k, &v, None, self.head_dim, is_causal)?
            {
                let attn_output = fused_out.transpose(1, 2)?.reshape((
                    bsz,
                    seq_len,
                    self.num_heads * self.head_dim,
                ))?;
                return self
                    .out_proj
                    .forward(&attn_output)
                    .map_err(|e| Error::InferenceError(e.to_string()));
            }
        }
        if is_causal
            && self.sliding_window.is_some()
            && q.device().is_cuda()
            && flash_attention_requested()
        {
            let cuda_options = voxtral_realtime_cuda_sliding_flash_options(self.sliding_window);
            if let Some(fused_out) = try_fused_self_attention_with_options(
                &q,
                &k,
                &v,
                None,
                self.head_dim,
                true,
                cuda_options,
            )? {
                let attn_output = fused_out.transpose(1, 2)?.reshape((
                    bsz,
                    seq_len,
                    self.num_heads * self.head_dim,
                ))?;
                return self
                    .out_proj
                    .forward(&attn_output)
                    .map_err(|e| Error::InferenceError(e.to_string()));
            }
        }

        // Scaled dot-product attention
        let q = q.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;
        let k = k.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;
        let v = v.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;

        let attn_weights = q.matmul(&k.transpose(1, 2)?)?;
        let attn_scale = attention_scale_tensor(self.scale, &attn_weights)?;
        let attn_weights = attn_weights.broadcast_mul(&attn_scale)?;

        // Apply causal/sliding mask if needed
        let attn_weights = if is_causal {
            let mask = tables.causal_mask(seq_len, self.sliding_window, &attn_weights)?;
            attn_weights.broadcast_add(&mask)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax(&attn_weights, 2)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output
            .reshape((bsz, self.num_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((bsz, seq_len, self.num_heads * self.head_dim))?;

        self.out_proj
            .forward(&attn_output)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }

    fn forward_valid_lengths(
        &self,
        x: &Tensor,
        is_causal: bool,
        valid_lengths: &[usize],
        frame_mask: &Tensor,
        tables: &EncoderForwardTables,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        validate_valid_lengths(batch, seq_len, valid_lengths)?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        let mut q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let mut k = k.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let v = v
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        if self.use_rope_positions {
            let (cos, sin) = tables.rope(seq_len, self.head_dim, self.rope_theta, &q)?;
            q = apply_interleaved_rotary_emb(&q, &cos, &sin)?;
            k = apply_interleaved_rotary_emb(&k, &cos, &sin)?;
        }
        let q = q
            .transpose(1, 2)?
            .reshape((batch * self.num_heads, seq_len, self.head_dim))?;
        let k = k
            .transpose(1, 2)?
            .reshape((batch * self.num_heads, seq_len, self.head_dim))?;
        let v = v.reshape((batch * self.num_heads, seq_len, self.head_dim))?;
        let weights = q.matmul(&k.transpose(1, 2)?)?;
        let scale = attention_scale_tensor(self.scale, &weights)?;
        let weights = weights.broadcast_mul(&scale)?;
        let mask = tables.valid_attention_mask(
            valid_lengths,
            self.num_heads,
            seq_len,
            is_causal,
            self.sliding_window,
            &weights,
        )?;
        let weights = candle_nn::ops::softmax(&weights.broadcast_add(&mask)?, 2)?;
        let output = weights
            .matmul(&v)?
            .reshape((batch, self.num_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;
        self.out_proj
            .forward(&output)?
            .broadcast_mul(frame_mask)
            .map_err(Error::from)
    }
}

/// Attention inputs that depend only on the sequence geometry, built at most
/// once per encoder forward and shared by every layer. Each layer used to
/// rebuild them: the RoPE cos/sin tables (host angles, upload, device
/// `cos`/`sin`) and an O(seq²) host mask uploaded per layer (32 layers; a
/// 30 s window is a 1500 x 1500 mask). Every entry is keyed by its full
/// geometry and rebuilt on any mismatch, so a cache hit returns exactly what
/// the layer would have built.
#[derive(Default)]
struct EncoderForwardTables {
    rope: RefCell<Option<(RopeTableKey, (Tensor, Tensor))>>,
    causal_mask: RefCell<Option<(CausalMaskKey, Tensor)>>,
    valid_mask: RefCell<Option<(ValidMaskKey, Tensor)>>,
    builds: Cell<usize>,
}

type RopeTableKey = (usize, usize, u64, DType, candle_core::DeviceLocation);
type CausalMaskKey = (usize, Option<usize>, DType, candle_core::DeviceLocation);
type ValidMaskKey = (
    Vec<usize>,
    usize,
    usize,
    bool,
    Option<usize>,
    DType,
    candle_core::DeviceLocation,
);

impl EncoderForwardTables {
    fn cached<K: PartialEq, V: Clone>(
        &self,
        slot: &RefCell<Option<(K, V)>>,
        key: K,
        build: impl FnOnce() -> Result<V>,
    ) -> Result<V> {
        if let Some((cached_key, value)) = slot.borrow().as_ref() {
            if *cached_key == key {
                return Ok(value.clone());
            }
        }
        let value = build()?;
        self.builds.set(self.builds.get() + 1);
        *slot.borrow_mut() = Some((key, value.clone()));
        Ok(value)
    }

    fn rope(
        &self,
        seq_len: usize,
        head_dim: usize,
        rope_theta: f64,
        reference: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let device = reference.device();
        let key = (
            seq_len,
            head_dim,
            rope_theta.to_bits(),
            reference.dtype(),
            device.location(),
        );
        self.cached(&self.rope, key, || {
            build_rope_cache(seq_len, head_dim, 0, rope_theta, device, reference.dtype())
        })
    }

    fn causal_mask(
        &self,
        seq_len: usize,
        sliding_window: Option<usize>,
        reference: &Tensor,
    ) -> Result<Tensor> {
        let key = (
            seq_len,
            sliding_window,
            reference.dtype(),
            reference.device().location(),
        );
        self.cached(&self.causal_mask, key, || {
            create_causal_mask(
                seq_len,
                sliding_window,
                reference.device(),
                reference.dtype(),
            )
        })
    }

    fn valid_attention_mask(
        &self,
        valid_lengths: &[usize],
        heads: usize,
        seq_len: usize,
        is_causal: bool,
        sliding_window: Option<usize>,
        reference: &Tensor,
    ) -> Result<Tensor> {
        let key = (
            valid_lengths.to_vec(),
            heads,
            seq_len,
            is_causal,
            sliding_window,
            reference.dtype(),
            reference.device().location(),
        );
        self.cached(&self.valid_mask, key, || {
            create_valid_attention_mask(
                valid_lengths,
                heads,
                seq_len,
                is_causal,
                sliding_window,
                reference.device(),
                reference.dtype(),
            )
        })
    }
}

fn create_causal_mask(
    seq_len: usize,
    sliding_window: Option<usize>,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        let earliest = sliding_window
            .filter(|window| *window > 0)
            .map(|window| i.saturating_add(1).saturating_sub(window))
            .unwrap_or(0);
        for j in 0..seq_len {
            if j > i || j < earliest {
                mask[i * seq_len + j] = f32::MIN;
            }
        }
    }
    let mask_tensor = Tensor::from_vec(mask, (seq_len, seq_len), device)?;
    mask_tensor
        .to_dtype(dtype)
        .map_err(|e| Error::InferenceError(e.to_string()))
}

fn create_valid_attention_mask(
    valid_lengths: &[usize],
    heads: usize,
    seq_len: usize,
    is_causal: bool,
    sliding_window: Option<usize>,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Tensor> {
    let rows = valid_lengths
        .len()
        .checked_mul(heads)
        .ok_or_else(|| Error::InvalidInput("Voxtral attention batch overflow".into()))?;
    let matrix = seq_len
        .checked_mul(seq_len)
        .ok_or_else(|| Error::InvalidInput("Voxtral attention extent overflow".into()))?;
    let mut mask = vec![
        f32::MIN;
        rows.checked_mul(matrix).ok_or_else(|| {
            Error::InvalidInput("Voxtral attention mask allocation overflow".into())
        })?
    ];
    for (batch, valid) in valid_lengths.iter().copied().enumerate() {
        for head in 0..heads {
            let row_base = (batch * heads + head) * matrix;
            for query in 0..seq_len {
                if query >= valid {
                    // Keep softmax finite; the query output is zeroed immediately after attention.
                    mask[row_base + query * seq_len] = 0.0;
                    continue;
                }
                let earliest = if is_causal {
                    sliding_window
                        .filter(|window| *window > 0)
                        .map(|window| query.saturating_add(1).saturating_sub(window))
                        .unwrap_or(0)
                } else {
                    0
                };
                let latest = if is_causal { query + 1 } else { valid };
                for key in earliest..latest.min(valid) {
                    mask[row_base + query * seq_len + key] = 0.0;
                }
            }
        }
    }
    Tensor::from_vec(mask, (rows, seq_len, seq_len), device)?
        .to_dtype(dtype)
        .map_err(Error::from)
}

fn voxtral_realtime_cuda_sliding_flash_options(
    sliding_window: Option<usize>,
) -> CudaFlashAttentionOptions<'static> {
    CudaFlashAttentionOptions {
        window_size_left: sliding_window.map(|window| window.saturating_sub(1)),
        ..CudaFlashAttentionOptions::default()
    }
}

fn attention_scale_tensor(scale: f64, reference: &Tensor) -> Result<Tensor> {
    Tensor::from_vec(vec![scale as f32], (1, 1, 1), reference.device())?
        .to_dtype(reference.dtype())
        .map_err(Error::from)
}

fn apply_interleaved_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let bsz = x.dim(0)?;
    let seq_len = x.dim(1)?;
    let heads = x.dim(2)?;
    let head_dim = x.dim(3)?;
    let half_dim = head_dim / 2;
    let x = x.reshape((bsz, seq_len, heads, half_dim, 2))?;
    let x1 = x.narrow(4, 0, 1)?.squeeze(4)?;
    let x2 = x.narrow(4, 1, 1)?.squeeze(4)?;

    let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(2)?;
    let rot1 = x1
        .broadcast_mul(&cos)?
        .broadcast_sub(&x2.broadcast_mul(&sin)?)?;
    let rot2 = x1
        .broadcast_mul(&sin)?
        .broadcast_add(&x2.broadcast_mul(&cos)?)?;
    let rot1 = rot1.unsqueeze(4)?;
    let rot2 = rot2.unsqueeze(4)?;
    Tensor::cat(&[rot1, rot2], 4)?
        .reshape((bsz, seq_len, heads, head_dim))
        .map_err(Error::from)
}

fn mel_frames_to_tensor(
    mel_spec: &[Vec<f32>],
    n_mels: usize,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Tensor> {
    let frames = mel_spec.len();
    let mut flat = Vec::with_capacity(frames * n_mels);
    for (idx, frame) in mel_spec.iter().enumerate() {
        if frame.len() != n_mels {
            return Err(Error::InferenceError(format!(
                "Mel frame {idx} has {} bins, expected {n_mels}",
                frame.len()
            )));
        }
        flat.extend_from_slice(frame);
    }

    Tensor::from_vec(flat, (1, frames, n_mels), device)?
        .to_dtype(dtype)
        .map_err(Error::from)
}

fn normalize_log_mel_with_max(log_mel: &mut [Vec<f32>], max_val: f32) {
    let clamp_min = max_val - 8.0;
    for frame in log_mel.iter_mut() {
        for value in frame.iter_mut() {
            if *value < clamp_min {
                *value = clamp_min;
            }
            *value = (*value + 4.0) / 4.0;
        }
    }
}

fn drop_last_mel_frame_for_voxtral(log_mel: &mut Vec<Vec<f32>>) {
    log_mel.pop();
}

fn append_generated_token(
    tokenizer: &VoxtralTokenizer,
    generated: &mut Vec<u32>,
    assembled: &mut String,
    token: u32,
    on_delta: &mut Option<&mut dyn FnMut(&str)>,
) -> Result<String> {
    // Extend the transcript to its last complete character: a character
    // whose bytes span several tokens appears once its last byte arrives.
    generated.push(token);
    let decoded = tokenizer.decode_text_prefix(generated)?;
    Ok(advance_assembled_text(assembled, decoded, on_delta))
}

/// Decode every generated byte once generation ends, so a trailing partial
/// character surfaces as U+FFFD instead of vanishing; returns the appended
/// text.
fn flush_generated_text(
    tokenizer: &VoxtralTokenizer,
    generated: &[u32],
    assembled: &mut String,
    on_delta: &mut Option<&mut dyn FnMut(&str)>,
) -> Result<String> {
    let decoded = tokenizer.decode_text_final(generated)?;
    Ok(advance_assembled_text(assembled, decoded, on_delta))
}

/// Replace the transcript with `decoded`, report the appended text to
/// `on_delta` one character at a time, and return it.
fn advance_assembled_text(
    assembled: &mut String,
    decoded: String,
    on_delta: &mut Option<&mut dyn FnMut(&str)>,
) -> String {
    let delta = text_delta(assembled, &decoded);
    if let Some(on_delta) = on_delta.as_deref_mut() {
        for ch in delta.chars() {
            let mut buf = [0u8; 4];
            on_delta(ch.encode_utf8(&mut buf));
        }
    }
    *assembled = decoded;
    delta
}

fn argmax(logits: &Tensor) -> Result<u32> {
    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (batch, _vocab) = logits.dims2()?;
            if batch != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected batched Voxtral logits for argmax: expected batch=1, got {batch}"
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected Voxtral logits rank for argmax: {rank}"
            )));
        }
    };
    if logits.dim(0)? == 0 {
        return Err(Error::InferenceError(
            "Voxtral logits are empty".to_string(),
        ));
    }

    let idx = logits.argmax(D::Minus1)?;
    let idx = if idx.rank() == 0 {
        idx
    } else {
        idx.squeeze(0)?
    };
    crate::models::shared::telemetry::record_dtype_cast();
    crate::models::shared::telemetry::record_host_read(DType::U32, 1);
    idx.to_dtype(DType::U32)?
        .to_scalar::<u32>()
        .map_err(Error::from)
}

fn text_delta(previous: &str, current: &str) -> String {
    if let Some(delta) = current.strip_prefix(previous) {
        return delta.to_string();
    }
    let common = previous
        .chars()
        .zip(current.chars())
        .take_while(|(a, b)| a == b)
        .count();
    current.chars().skip(common).collect()
}

fn resample_audio(audio: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == 0 || to_rate == 0 {
        return Err(Error::InvalidInput(format!(
            "Invalid audio sample rate for resampling: {from_rate} -> {to_rate}"
        )));
    }
    if audio.is_empty() {
        return Ok(Vec::new());
    }
    if from_rate == to_rate {
        return Ok(audio.to_vec());
    }

    let new_len = resampled_length(audio.len(), from_rate, to_rate)?;
    let ratio = to_rate as f64 / from_rate as f64;
    let mut resampled = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_idx = i as f64 / ratio;
        let src_idx_floor = src_idx.floor() as usize;
        let src_idx_ceil = (src_idx_floor + 1).min(audio.len() - 1);
        let frac = src_idx - src_idx_floor as f64;

        let val = audio[src_idx_floor] as f64 * (1.0 - frac) + audio[src_idx_ceil] as f64 * frac;
        resampled.push(val as f32);
    }

    Ok(resampled)
}

fn resampled_length(samples: usize, from_rate: u32, to_rate: u32) -> Result<usize> {
    if from_rate == 0 || to_rate == 0 {
        return Err(Error::InvalidInput(format!(
            "Invalid audio sample rate for resampling: {from_rate} -> {to_rate}"
        )));
    }
    let scaled = (samples as u128)
        .checked_mul(to_rate as u128)
        .ok_or_else(|| Error::InvalidInput("Voxtral resampled length overflow".into()))?
        / from_rate as u128;
    usize::try_from(scaled)
        .map_err(|_| Error::InvalidInput("Voxtral resampled length exceeds usize".into()))
}

fn validate_realtime_sample_ceiling(
    source_samples: usize,
    source_rate: u32,
    target_rate: u32,
    ceiling: usize,
) -> Result<usize> {
    let resampled_samples = resampled_length(source_samples, source_rate, target_rate)?;
    if source_samples > ceiling || resampled_samples > ceiling {
        return Err(Error::InvalidInput(format!(
            "Voxtral realtime stream has {source_samples} source samples ({resampled_samples} at {target_rate} Hz), exceeding the {ceiling}-sample realtime session ceiling"
        )));
    }
    Ok(resampled_samples)
}

/// Weights-free geometry of realtime preparation, derived at load from the
/// published `params.json`, the `config.json` sidecar and the `tekken.json`
/// audio metadata. Preparation, the stage seal and stream reservations all
/// read it, so the published configs can be checked without weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VoxtralRealtimeLayout {
    target_rate: u32,
    hop_length: usize,
    n_mels: usize,
    raw_audio_length_per_tok: usize,
    left_pad_tokens: usize,
    right_pad_tokens: usize,
    pool_size: usize,
    text_dim: usize,
    conv1: Conv1dGeometry,
    conv2: Conv1dGeometry,
    encoder_hidden: usize,
    encoder_ffn: usize,
    encoder_heads: usize,
    encoder_sliding_window: Option<usize>,
}

impl VoxtralRealtimeLayout {
    fn new(
        config: &VoxtralConfig,
        audio: &AudioConfig,
        pool_size: usize,
        audio_length_per_tok: usize,
    ) -> Result<Self> {
        let encoder = config.audio_config();
        let target_rate = u32::try_from(encoder.sampling_rate)
            .map_err(|_| Error::ModelLoadError("Voxtral sampling rate exceeds u32".into()))?;
        let (conv1, conv2) = conv_stem_geometry(&encoder, true);
        let layout = Self {
            target_rate,
            hop_length: encoder.hop_length,
            n_mels: encoder.num_mel_bins,
            raw_audio_length_per_tok: audio.raw_audio_length_per_tok(),
            // Offline audio carries the same left silence as a stream
            // (mistral_common `_get_streaming_pad`); the 39-token prompt then
            // covers 32 silent tokens plus the first 7 audio tokens.
            left_pad_tokens: audio.streaming_left_pad_tokens,
            right_pad_tokens: audio.offline_right_pad_tokens(),
            pool_size,
            text_dim: config.text_dim,
            conv1,
            conv2,
            encoder_hidden: encoder.d_model,
            encoder_ffn: encoder.encoder_ffn_dim,
            encoder_heads: encoder.encoder_attention_heads,
            encoder_sliding_window: (encoder.is_causal && encoder.sliding_window > 0)
                .then_some(encoder.sliding_window),
        };
        layout.validate_frame_contract(audio_length_per_tok)?;
        Ok(layout)
    }

    /// One audio token must cover exactly the samples and mel frames the
    /// checkpoint was trained on: `hop * s1 * s2 * pool` samples
    /// (`raw_audio_length_per_tok`, 160 * 1 * 2 * 4 = 1280) and
    /// `s1 * s2 * pool` mel frames (`audio_length_per_tok`, 8). A stem that
    /// disagrees runs the decoder at the wrong audio rate, so it fails the load.
    fn validate_frame_contract(&self, audio_length_per_tok: usize) -> Result<()> {
        let mel_frames_per_token = self
            .conv1
            .stride
            .checked_mul(self.conv2.stride)
            .and_then(|stride| stride.checked_mul(self.pool_size));
        let samples_per_token =
            mel_frames_per_token.and_then(|frames| frames.checked_mul(self.hop_length));
        let (Some(mel_frames_per_token), Some(samples_per_token)) =
            (mel_frames_per_token, samples_per_token)
        else {
            return Err(Error::ModelLoadError(
                "Voxtral encoder stem geometry overflows".into(),
            ));
        };
        if mel_frames_per_token != audio_length_per_tok
            || samples_per_token != self.raw_audio_length_per_tok
        {
            return Err(Error::ModelLoadError(format!(
                "Voxtral encoder stem (hop {}, conv strides {} and {}, pooling {}) gives {mel_frames_per_token} mel frames and {samples_per_token} samples per audio token, but the checkpoint expects {audio_length_per_tok} and {}",
                self.hop_length,
                self.conv1.stride,
                self.conv2.stride,
                self.pool_size,
                self.raw_audio_length_per_tok
            )));
        }
        Ok(())
    }
}

fn realtime_preparation_geometry_for(
    source_samples: usize,
    source_rate: u32,
    layout: &VoxtralRealtimeLayout,
    mode: VoxtralRealtimePreparationMode,
) -> Result<VoxtralRealtimePreparationGeometry> {
    if layout.hop_length == 0 || layout.n_mels == 0 || layout.pool_size == 0 || layout.text_dim == 0
    {
        return Err(Error::ModelLoadError(
            "Voxtral realtime preparation has zero-sized model geometry".into(),
        ));
    }
    let resampled_samples = resampled_length(source_samples, source_rate, layout.target_rate)?;
    let left_pad = layout
        .raw_audio_length_per_tok
        .checked_mul(layout.left_pad_tokens)
        .ok_or_else(|| Error::InvalidInput("Voxtral left padding overflow".into()))?;
    let right_pad = if mode == VoxtralRealtimePreparationMode::Finish {
        offline_streaming_padding_samples(
            resampled_samples,
            layout.raw_audio_length_per_tok,
            0,
            layout.right_pad_tokens,
        )
        .1
    } else {
        0
    };
    let padded_samples = left_pad
        .checked_add(resampled_samples)
        .and_then(|length| length.checked_add(right_pad))
        .ok_or_else(|| Error::InvalidInput("Voxtral padded audio length overflow".into()))?;
    let mel_frames = padded_samples / layout.hop_length;
    let conv1_frames = conv_output_length(mel_frames, layout.conv1)?;
    let conv2_frames = conv_output_length(conv1_frames, layout.conv2)?;
    let pooled_frames = conv2_frames / layout.pool_size;
    let stable_frames = if mode == VoxtralRealtimePreparationMode::Finish {
        pooled_frames
    } else {
        pooled_frames.saturating_sub(layout.right_pad_tokens)
    };
    let embedding_elements = u64::try_from(pooled_frames)
        .ok()
        .and_then(|frames| {
            u64::try_from(layout.text_dim)
                .ok()
                .and_then(|dim| frames.checked_mul(dim))
        })
        .ok_or_else(|| Error::InvalidInput("Voxtral embedding geometry overflow".into()))?;
    Ok(VoxtralRealtimePreparationGeometry {
        source_samples,
        resampled_samples,
        padded_samples,
        mel_frames,
        conv1_frames,
        conv2_frames,
        pooled_frames,
        stable_frames,
        embedding_elements,
    })
}

fn realtime_materialized_elements_for(
    layout: &VoxtralRealtimeLayout,
    mel_frames: usize,
    conv_frames: usize,
) -> Result<u64> {
    checked_materialized_elements(
        mel_frames,
        conv_output_length(mel_frames, layout.conv1)?,
        conv_frames,
        layout.n_mels,
        layout.encoder_hidden,
        layout.text_dim,
    )
}

fn realtime_workspace_bytes_for(
    layout: &VoxtralRealtimeLayout,
    mel_frames: usize,
    conv_frames: usize,
    dtype_bytes: usize,
) -> Result<u64> {
    checked_workspace_bytes(
        realtime_materialized_elements_for(layout, mel_frames, conv_frames)?,
        conv_frames,
        layout.encoder_hidden,
        layout.encoder_ffn,
        layout.encoder_heads,
        layout.encoder_sliding_window,
        dtype_bytes,
    )
}

/// The worst preparation transaction a loaded model admits: one row holding
/// the whole session ceiling, finished with its right padding.
fn realtime_preparation_stage_seal_for(
    layout: &VoxtralRealtimeLayout,
    max_source_samples: usize,
    dtype_bytes: usize,
) -> Result<VoxtralRealtimePreparationStageSeal> {
    let max = realtime_preparation_geometry_for(
        max_source_samples,
        layout.target_rate,
        layout,
        VoxtralRealtimePreparationMode::Finish,
    )?;
    Ok(VoxtralRealtimePreparationStageSeal {
        max_source_samples,
        max_work_units: u64::try_from(max_source_samples).map_err(|_| {
            Error::ModelLoadError("Voxtral realtime work ceiling exceeds u64".into())
        })?,
        max_materialized_tensor_elements_per_row: realtime_materialized_elements_for(
            layout,
            max.mel_frames,
            max.conv2_frames,
        )?,
        max_workspace_bytes: realtime_workspace_bytes_for(
            layout,
            max.mel_frames,
            max.conv2_frames,
            dtype_bytes,
        )?,
    })
}

/// Longest cumulative source, in samples at the model rate, that one
/// realtime session may prepare: the checkpoint's `chunk_length_s` when it
/// sets one, otherwise the izwi session policy, bounded by the model context.
fn realtime_source_sample_ceiling(
    config: &VoxtralConfig,
    layout: &VoxtralRealtimeLayout,
) -> Result<usize> {
    let seconds = config
        .multimodal
        .whisper_model_args
        .encoder_args
        .audio_encoding_args
        .chunk_length_s
        .unwrap_or(VOXTRAL_REALTIME_DEFAULT_MAX_SESSION_SECONDS);
    if !seconds.is_finite() || seconds <= 0.0 || layout.target_rate == 0 {
        return Err(Error::ModelLoadError(
            "Voxtral realtime session length must be finite and positive".into(),
        ));
    }
    let policy_samples = (seconds as f64 * layout.target_rate as f64).ceil();
    if policy_samples > usize::MAX as f64 {
        return Err(Error::ModelLoadError(
            "Voxtral realtime sample policy exceeds usize".into(),
        ));
    }
    let model_samples = config
        .model_max_length
        .checked_mul(layout.raw_audio_length_per_tok)
        .and_then(|total| {
            layout
                .raw_audio_length_per_tok
                .checked_mul(
                    layout
                        .left_pad_tokens
                        .saturating_add(layout.right_pad_tokens)
                        .saturating_add(1),
                )
                .and_then(|padding| total.checked_sub(padding))
        })
        .ok_or_else(|| {
            Error::ModelLoadError("Voxtral realtime context geometry is invalid".into())
        })?;
    let ceiling = (policy_samples as usize).min(model_samples);
    if ceiling == 0 {
        return Err(Error::ModelLoadError(
            "Voxtral realtime sample ceiling is zero".into(),
        ));
    }
    Ok(ceiling)
}

fn checked_materialized_elements(
    mel_frames: usize,
    conv1_frames: usize,
    conv_frames: usize,
    n_mels: usize,
    hidden: usize,
    text_dim: usize,
) -> Result<u64> {
    let mel = mel_frames.checked_mul(n_mels);
    let conv = conv1_frames
        .checked_add(conv_frames)
        .and_then(|frames| frames.checked_mul(hidden));
    let adapted = conv_frames.checked_mul(text_dim);
    mel.and_then(|mel| conv.and_then(|conv| mel.checked_add(conv)))
        .and_then(|value| adapted.and_then(|adapted| value.checked_add(adapted)))
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| Error::ModelLoadError("Voxtral materialization ceiling overflow".into()))
}

fn checked_realtime_stream_peak_bytes(
    frontend_host_bytes: u64,
    transactional_text_host_bytes: u64,
    embedding_bytes: u64,
    preparation_scratch_bytes: u64,
) -> Result<(u64, u64)> {
    let host = frontend_host_bytes
        .checked_add(transactional_text_host_bytes)
        .ok_or_else(|| {
            Error::ModelLoadError("Voxtral transactional host peak overflowed".into())
        })?;
    // Tensor checkpoints are shallow handles, but their old allocation stays
    // live until commit while the replacement embedding and scratch coexist.
    let tensor = embedding_bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(preparation_scratch_bytes))
        .ok_or_else(|| {
            Error::ModelLoadError("Voxtral transactional tensor peak overflowed".into())
        })?;
    Ok((host, tensor))
}

fn checked_realtime_frontend_host_peak_bytes(
    source_bytes: u64,
    resampled_samples: usize,
    padded_samples: usize,
    mel_frames: usize,
    n_mels: usize,
    n_fft: usize,
) -> Result<u64> {
    let f32_bytes = std::mem::size_of::<f32>() as u64;
    let bytes = |elements: usize, element_bytes: u64| {
        u64::try_from(elements)
            .ok()
            .and_then(|elements| elements.checked_mul(element_bytes))
    };
    // Old source + owned append equals one complete stream, and the newly
    // assembled source is a second. The remaining terms mirror the actual
    // host frontend: optional resample, padded PCM, reflect padding, FFT frame,
    // power spectrum, flat log-mel, and its frame-vector conversion.
    let source_overlap = source_bytes.checked_mul(2);
    let resampled = bytes(resampled_samples, f32_bytes);
    let padded = bytes(padded_samples, f32_bytes);
    let reflected = padded_samples
        .checked_add(n_fft)
        .and_then(|elements| bytes(elements, f32_bytes));
    let fft_frame = bytes(
        n_fft,
        std::mem::size_of::<rustfft::num_complex::Complex<f32>>() as u64,
    );
    let power = bytes(n_fft / 2 + 1, f32_bytes);
    let mel_elements = mel_frames.checked_mul(n_mels);
    let flat_mel = mel_elements.and_then(|elements| bytes(elements, f32_bytes));
    let nested_mel = flat_mel.and_then(|data| {
        bytes(mel_frames, std::mem::size_of::<Vec<f32>>() as u64)
            .and_then(|headers| data.checked_add(headers))
    });
    [
        source_overlap,
        resampled,
        padded,
        reflected,
        fft_frame,
        power,
        flat_mel,
        nested_mel,
    ]
    .into_iter()
    .try_fold(0u64, |total, value| total.checked_add(value?))
    .ok_or_else(|| Error::ModelLoadError("Voxtral frontend host peak overflowed".into()))
}

fn checked_realtime_text_host_bytes(
    max_generated_tokens: usize,
    max_decoded_token_bytes: usize,
) -> Result<(u64, u64)> {
    // Realtime prefill/decode produces at most one prediction per prepared
    // pooled audio frame across the entire stream.
    let token_capacity = max_generated_tokens
        .checked_next_power_of_two()
        .ok_or_else(|| {
            Error::ModelLoadError("Voxtral generated-token capacity overflowed".into())
        })?;
    let generated = u64::try_from(token_capacity)
        .ok()
        .and_then(|tokens| tokens.checked_mul(std::mem::size_of::<u32>() as u64))
        .ok_or_else(|| Error::ModelLoadError("Voxtral generated-token bytes overflowed".into()))?;
    let assembled_len = max_generated_tokens
        .checked_mul(max_decoded_token_bytes)
        .ok_or_else(|| Error::ModelLoadError("Voxtral assembled-text length overflowed".into()))?;
    let assembled_capacity = assembled_len
        .checked_next_power_of_two()
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| {
            Error::ModelLoadError("Voxtral assembled-text capacity overflowed".into())
        })?;
    let committed = generated
        .checked_add(assembled_capacity)
        .ok_or_else(|| Error::ModelLoadError("Voxtral committed text bytes overflowed".into()))?;
    let transactional = generated
        .checked_mul(2)
        .and_then(|bytes| {
            assembled_capacity
                .checked_mul(3)
                .and_then(|text| bytes.checked_add(text))
        })
        .ok_or_else(|| {
            Error::ModelLoadError("Voxtral transactional text bytes overflowed".into())
        })?;
    Ok((committed, transactional))
}

#[allow(clippy::too_many_arguments)]
fn checked_workspace_bytes(
    materialized: u64,
    conv_frames: usize,
    hidden: usize,
    ffn: usize,
    heads: usize,
    _sliding_window: Option<usize>,
    dtype_bytes: usize,
) -> Result<u64> {
    let attention = conv_frames
        // The current valid-length path materializes both dense attention
        // weights and a dense mask even when the mask describes a sliding
        // window. Do not charge sparse-window geometry until the kernel is
        // actually sparse.
        .checked_mul(conv_frames)
        .and_then(|value| value.checked_mul(heads));
    let activations = conv_frames.checked_mul(
        hidden
            .checked_mul(8)
            .and_then(|value| ffn.checked_mul(3).and_then(|ffn| value.checked_add(ffn)))
            .ok_or_else(|| Error::ModelLoadError("Voxtral activation geometry overflow".into()))?,
    );
    let elements = attention
        .and_then(|attention| {
            activations.and_then(|activations| attention.checked_add(activations))
        })
        .and_then(|value| u64::try_from(value).ok())
        .and_then(|value| value.checked_add(materialized))
        .ok_or_else(|| Error::ModelLoadError("Voxtral workspace geometry overflow".into()))?;
    elements
        .checked_mul(u64::try_from(dtype_bytes.max(4)).unwrap_or(u64::MAX))
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or_else(|| Error::ModelLoadError("Voxtral workspace byte ceiling overflow".into()))
}

fn offline_streaming_padding_samples(
    sample_len: usize,
    raw_audio_length_per_tok: usize,
    left_pad_tokens: usize,
    right_pad_tokens: usize,
) -> (usize, usize) {
    let token_samples = raw_audio_length_per_tok.max(1);
    let left_pad = token_samples.saturating_mul(left_pad_tokens);
    let alignment_pad = (token_samples - (sample_len % token_samples)) % token_samples;
    let right_pad = alignment_pad.saturating_add(token_samples.saturating_mul(right_pad_tokens));
    (left_pad, right_pad)
}

/// The safetensors files `VoxtralRealtimeModel::load` maps: the Mistral
/// `consolidated.safetensors` when present, else the HF shards of
/// `model.safetensors.index.json`, else `model.safetensors`. Load admission
/// prices the same files (`runtime/lifecycle/asr_memory.rs`).
pub(crate) fn voxtral_realtime_weight_files(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let consolidated_path = model_dir.join("consolidated.safetensors");
    if consolidated_path.exists() {
        return Ok(vec![consolidated_path]);
    }

    let index_path = model_dir.join("model.safetensors.index.json");
    if !index_path.exists() {
        return Ok(vec![model_dir.join("model.safetensors")]);
    }
    let index_data = std::fs::read_to_string(&index_path)
        .map_err(|e| Error::ModelLoadError(format!("Failed to read index: {}", e)))?;
    let index: serde_json::Value = serde_json::from_str(&index_data)
        .map_err(|e| Error::ModelLoadError(format!("Failed to parse index: {}", e)))?;

    let weight_map = index
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| Error::InvalidInput("Invalid weight map".to_string()))?;

    let mut shard_files: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    shard_files.sort();
    shard_files.dedup();

    Ok(shard_files.iter().map(|f| model_dir.join(f)).collect())
}

fn load_weights<'a>(
    model_dir: &'a Path,
    dtype: DType,
    device: &'a DeviceProfile,
) -> Result<VarBuilder<'a>> {
    let files = voxtral_realtime_weight_files(model_dir)?;
    info!(
        "Loading Voxtral weights from {} safetensors file(s)",
        files.len()
    );
    unsafe {
        staged::metal_staged_safetensors(&files, dtype, &device.device)
            .map_err(|e| Error::ModelLoadError(format!("Failed to load weights: {}", e)))
    }
}

/// Voxtral's own `params.json` when present, else the HF `config.json`.
fn read_voxtral_config(model_dir: &Path) -> Result<String> {
    let config_path = if model_dir.join("params.json").exists() {
        model_dir.join("params.json")
    } else {
        model_dir.join("config.json")
    };
    std::fs::read_to_string(&config_path)
        .map_err(|e| Error::ModelLoadError(format!("Failed to read config: {}", e)))
}

fn voxtral_checkpoint_dtype(
    config_str: &str,
    runtime_config: &VoxtralRuntimeConfig,
) -> Option<DType> {
    checkpoint_dtype_from_config_json(config_str).or(runtime_config.checkpoint_dtype)
}

/// The weight dtype `VoxtralRealtimeModel::load` selects for the checkpoint
/// in `model_dir` on `device`. Load admission prices the checkpoint at this
/// dtype (`runtime/lifecycle/asr_memory.rs`).
pub(crate) fn voxtral_realtime_weight_dtype(
    model_dir: &Path,
    device: &DeviceProfile,
) -> Result<DType> {
    let config_str = read_voxtral_config(model_dir)?;
    let runtime_config = load_voxtral_runtime_config(model_dir)?;
    Ok(select_voxtral_dtype(
        device,
        voxtral_checkpoint_dtype(&config_str, &runtime_config),
    ))
}

fn load_voxtral_runtime_config(model_dir: &Path) -> Result<VoxtralRuntimeConfig> {
    let sidecar_path = model_dir.join("config.json");
    if !sidecar_path.exists() {
        return Ok(VoxtralRuntimeConfig::default());
    }

    let sidecar_str = std::fs::read_to_string(&sidecar_path).map_err(|e| {
        Error::ModelLoadError(format!("Failed to read Voxtral config.json sidecar: {}", e))
    })?;
    let sidecar: VoxtralRuntimeSidecarConfig = serde_json::from_str(&sidecar_str).map_err(|e| {
        Error::ModelLoadError(format!(
            "Failed to parse Voxtral config.json sidecar: {}",
            e
        ))
    })?;

    Ok(VoxtralRuntimeConfig {
        default_num_delay_tokens: sidecar.default_num_delay_tokens,
        downsample_factor: sidecar.downsample_factor,
        audio_length_per_tok: sidecar.audio_length_per_tok,
        checkpoint_dtype: checkpoint_dtype_from_config_json(&sidecar_str),
    })
}

/// The `params.json` and `config.json` half of the realtime geometry. The
/// tokenizer merges the `tekken.json` audio metadata into `audio` at load.
struct VoxtralRealtimeAudioGeometry {
    audio: AudioConfig,
    block_pool_size: usize,
    audio_length_per_tok: usize,
}

fn realtime_audio_geometry(
    config: &VoxtralConfig,
    runtime: &VoxtralRuntimeConfig,
) -> VoxtralRealtimeAudioGeometry {
    let num_delay_tokens = runtime
        .default_num_delay_tokens
        .unwrap_or_else(|| config.num_delay_tokens())
        .max(1);
    let block_pool_size = runtime
        .downsample_factor
        .unwrap_or_else(|| config.block_pool_size())
        .max(1);
    let audio_length_per_tok = runtime
        .audio_length_per_tok
        .unwrap_or(block_pool_size.saturating_mul(2))
        .max(1);
    let encoder = config.audio_config();
    VoxtralRealtimeAudioGeometry {
        audio: AudioConfig {
            sampling_rate: encoder.sampling_rate,
            frame_rate: config.frame_rate(),
            window_size: encoder.window_size,
            hop_length: encoder.hop_length,
            num_mel_bins: encoder.num_mel_bins,
            n_delay_tokens: num_delay_tokens,
            ..AudioConfig::default()
        },
        block_pool_size,
        audio_length_per_tok,
    }
}

/// The delay the tokenizer derived from `tekken.json`'s
/// `transcription_delay_ms` (mistral_common's source of truth). HF's
/// `config.json` repeats it as `default_num_delay_tokens`; a checkpoint whose
/// two copies disagree is ambiguous, so it fails the load.
fn checked_delay_tokens(audio: &AudioConfig, runtime: &VoxtralRuntimeConfig) -> Result<usize> {
    match runtime.default_num_delay_tokens {
        Some(sidecar) if sidecar != audio.n_delay_tokens => Err(Error::ModelLoadError(format!(
            "Voxtral tekken.json transcription delay gives {} delay tokens, but config.json default_num_delay_tokens is {sidecar}",
            audio.n_delay_tokens
        ))),
        _ => Ok(audio.n_delay_tokens),
    }
}

fn select_voxtral_dtype(device: &DeviceProfile, checkpoint_dtype: Option<DType>) -> DType {
    voxtral_precision(device, checkpoint_dtype).dtype
}

/// Voxtral's dtype on `device`, and why: a BF16 checkpoint takes the Metal
/// BF16-checkpoint LM tier (`backends/device.rs`), `IZWI_METAL_DTYPE`
/// replaces it, CUDA follows the checkpoint and CPU runs F32.
fn voxtral_precision(device: &DeviceProfile, checkpoint_dtype: Option<DType>) -> DTypeSelection {
    let mut request = DTypeSelectionRequest::new(None).with_model_family(ModelFamily::Voxtral);
    if let Some(dtype) = checkpoint_dtype {
        request = request.with_checkpoint_dtype(dtype);
    }
    device.resolve_dtype(request)
}

/// Exact (erf) GELU, as HF and vLLM apply after each stem convolution.
fn gelu(x: &Tensor) -> Result<Tensor> {
    exact_gelu(x).map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::super::config::VoxtralConfig;
    use super::super::tokenizer::{merge_tekken_audio_metadata, AudioConfig};
    use super::{
        apply_interleaved_rotary_emb, argmax, attention_scale_tensor,
        checked_realtime_frontend_host_peak_bytes, checked_realtime_stream_peak_bytes,
        checked_realtime_text_host_bytes, checked_workspace_bytes, conv1d_with_geometry,
        conv_output_length, create_valid_attention_mask, drop_last_mel_frame_for_voxtral,
        load_voxtral_runtime_config, mel_frames_to_tensor, normalize_log_mel_with_max,
        offline_streaming_padding_samples, pool_audio_embeddings_by_block,
        prepare_realtime_conv_input, realtime_audio_geometry, realtime_preparation_geometry_for,
        realtime_preparation_stage_seal_for, realtime_source_sample_ceiling, resample_audio,
        select_voxtral_dtype, text_delta, validate_realtime_sample_ceiling,
        voxtral_realtime_cuda_sliding_flash_options, voxtral_realtime_offline_frame_limit,
        Conv1dGeometry, VoxtralRealtimeLayout, VoxtralRealtimePreparationMode,
    };
    use crate::backends::{BackendKind, DeviceCapabilities, DeviceKind, DeviceProfile};
    use candle_core::{DType, Device, Tensor};
    use std::path::PathBuf;
    use uuid::Uuid;

    use super::super::tokenizer::{published_byte_ids, published_excerpt_tokenizer};
    use super::{
        append_generated_token, checked_delay_tokens, flush_generated_text, VoxtralRuntimeConfig,
    };
    use super::{PORTABLE_OFFLINE_AUDIO_FRAME_LIMIT, VOXTRAL_REALTIME_OFFLINE_AUDIO_SECONDS_HINT};

    /// Sharing one table set across the encoder layers gives bit-identical
    /// output to rebuilding the RoPE tables and masks in every layer (the
    /// pre-change behaviour), with one build of each instead of one per layer.
    #[test]
    fn encoder_layers_share_rope_and_masks_bit_identically() {
        use super::{frame_valid_mask, EncoderForwardTables, WhisperEncoder};
        let device = Device::Cpu;
        let cfg = super::super::config::AudioEncoderConfig {
            d_model: 32,
            encoder_layers: 3,
            encoder_attention_heads: 4,
            encoder_ffn_dim: 64,
            num_mel_bins: 8,
            max_source_positions: 64,
            window_size: 400,
            hop_length: 160,
            sampling_rate: 16_000,
            is_causal: true,
            conv1_kernel_size: 3,
            conv1_stride: 1,
            conv2_kernel_size: 3,
            conv2_stride: 2,
            global_log_mel_max: None,
            head_dim: 8,
            rope_theta: 1_000_000.0,
            pos_embed: "rope".into(),
            sliding_window: 5,
        };
        let vars = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vars, DType::F32, &device);
        let encoder = WhisperEncoder::load_voxtral(&cfg, vb).unwrap();
        let mut state = 0x9B05_688C_2B3E_6C1Fu64;
        let values = (0..2 * 23 * 32)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect::<Vec<_>>();
        let x = Tensor::from_vec(values, (2, 23, 32), &device).unwrap();
        let bits = |t: &Tensor| {
            t.flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };

        let shared = EncoderForwardTables::default();
        let (mut fresh_x, mut shared_x) = (x.clone(), x.clone());
        let mut fresh_builds = 0;
        for layer in &encoder.layers {
            let fresh = EncoderForwardTables::default();
            fresh_x = layer.forward(&fresh_x, true, &fresh).unwrap();
            fresh_builds += fresh.builds.get();
            shared_x = layer.forward(&shared_x, true, &shared).unwrap();
        }
        assert_eq!(bits(&fresh_x), bits(&shared_x));
        assert_eq!(fresh_builds, 2 * cfg.encoder_layers);
        assert_eq!(shared.builds.get(), 2);

        let lengths = [23usize, 11];
        let frame_mask = frame_valid_mask(&lengths, 23, &x).unwrap();
        let shared = EncoderForwardTables::default();
        let (mut fresh_x, mut shared_x) = (x.clone(), x.clone());
        for layer in &encoder.layers {
            let fresh = EncoderForwardTables::default();
            fresh_x = layer
                .forward_valid_lengths(&fresh_x, true, &lengths, &frame_mask, &fresh)
                .unwrap();
            shared_x = layer
                .forward_valid_lengths(&shared_x, true, &lengths, &frame_mask, &shared)
                .unwrap();
        }
        assert_eq!(bits(&fresh_x), bits(&shared_x));
        assert_eq!(shared.builds.get(), 2);

        // A different geometry in the same table set rebuilds instead of
        // returning a stale entry.
        let short = x.narrow(1, 0, 7).unwrap();
        let expected = encoder.layers[0]
            .forward(&short, true, &EncoderForwardTables::default())
            .unwrap();
        let reused = EncoderForwardTables::default();
        encoder.layers[0].forward(&x, true, &reused).unwrap();
        let actual = encoder.layers[0].forward(&short, true, &reused).unwrap();
        assert_eq!(bits(&expected), bits(&actual));
        assert_eq!(reused.builds.get(), 4);
    }

    /// Published configs of mistralai/Voxtral-Mini-4B-Realtime-2602 at
    /// revision 2769294d: `params.json` and `config.json` verbatim, plus a
    /// `tekken.json` excerpt (config, audio, named specials, byte tokens).
    fn published_fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/voxtral_realtime_2602")
    }

    /// The weights-free half of `VoxtralRealtimeModel::load` over the
    /// published configs; only the Tekken vocabulary and weights are skipped.
    fn published_realtime_layout() -> (VoxtralConfig, AudioConfig, VoxtralRealtimeLayout) {
        let dir = published_fixture_dir();
        let params = std::fs::read_to_string(dir.join("params.json")).unwrap();
        let config: VoxtralConfig = serde_json::from_str(&params).unwrap();
        let runtime = load_voxtral_runtime_config(&dir).unwrap();
        let geometry = realtime_audio_geometry(&config, &runtime);
        let tekken = std::fs::read_to_string(dir.join("tekken_audio.json")).unwrap();
        let audio = merge_tekken_audio_metadata(geometry.audio, &tekken).unwrap();
        let layout = VoxtralRealtimeLayout::new(
            &config,
            &audio,
            geometry.block_pool_size,
            geometry.audio_length_per_tok,
        )
        .unwrap();
        (config, audio, layout)
    }

    #[test]
    fn published_realtime_config_seals_without_chunk_length() {
        let (config, audio, layout) = published_realtime_layout();
        let encoding = &config
            .multimodal
            .whisper_model_args
            .encoder_args
            .audio_encoding_args;
        // mistral_common requires chunk_length_s to be None in streaming mode.
        assert_eq!(encoding.chunk_length_s, None);
        assert_eq!(encoding.transcription_format, "streaming");
        assert_eq!(audio.streaming_left_pad_tokens, 32);
        assert_eq!(audio.n_delay_tokens, 6);
        // tekken.json's 480 ms delay agrees with config.json's default.
        let runtime = load_voxtral_runtime_config(&published_fixture_dir()).unwrap();
        assert_eq!(checked_delay_tokens(&audio, &runtime).unwrap(), 6);

        let ceiling = realtime_source_sample_ceiling(&config, &layout).unwrap();
        assert_eq!(ceiling, 480_000);

        let seal =
            realtime_preparation_stage_seal_for(&layout, ceiling, DType::BF16.size_in_bytes())
                .unwrap();
        assert_eq!(seal.max_source_samples, 480_000);
        assert_eq!(seal.max_work_units, 480_000);
        assert!(seal.max_materialized_tensor_elements_per_row > 0);
        assert!(seal.max_workspace_bytes > 0);
        assert!(seal.max_workspace_bytes < 4 << 30);
    }

    /// pooled = ceil(N / 1280) + 49: 32 left-pad tokens, 17 right-pad tokens
    /// (6 delay + 1 + 10 buffer) and the alignment to whole tokens.
    #[test]
    fn published_offline_geometry_matches_reference_table() {
        let (_, _, layout) = published_realtime_layout();
        // (samples, padded, mel = conv1, conv2, pooled)
        let table = [
            (1, 64_000, 400, 200, 50),
            (16_000, 79_360, 496, 248, 62),
            (16_001, 79_360, 496, 248, 62),
            (160_000, 222_720, 1_392, 696, 174),
            (480_000, 542_720, 3_392, 1_696, 424),
        ];
        for (samples, padded, mel, conv2, pooled) in table {
            let geometry = realtime_preparation_geometry_for(
                samples,
                16_000,
                &layout,
                VoxtralRealtimePreparationMode::Finish,
            )
            .unwrap();
            assert_eq!(geometry.padded_samples, padded, "{samples} samples");
            assert_eq!(geometry.mel_frames, mel, "{samples} samples");
            assert_eq!(geometry.conv1_frames, mel, "{samples} samples");
            assert_eq!(geometry.conv2_frames, conv2, "{samples} samples");
            assert_eq!(geometry.pooled_frames, pooled, "{samples} samples");
            assert_eq!(pooled, samples.div_ceil(1_280) + 49);
            assert_eq!(geometry.stable_frames, pooled);
            // HF generates one token per audio token after the 39-token prompt.
            assert!(pooled > 39);
        }
    }

    #[test]
    fn frame_contract_accepts_published_stem_and_rejects_stride_two() {
        let (config, audio, layout) = published_realtime_layout();
        assert_eq!(layout.conv1, Conv1dGeometry::causal(3, 1));
        assert_eq!(layout.conv2, Conv1dGeometry::causal(3, 2));
        assert_eq!(layout.raw_audio_length_per_tok, 1_280);
        assert_eq!(layout.left_pad_tokens, 32);
        assert_eq!(layout.right_pad_tokens, 17);
        layout.validate_frame_contract(8).unwrap();

        // The stride-2 stem ran the decoder at 6.25 Hz instead of 12.5 Hz.
        let mut stride_two = layout;
        stride_two.conv1 = Conv1dGeometry::symmetric(3, 2);
        let error = stride_two.validate_frame_contract(8).unwrap_err();
        assert!(format!("{error}").contains("16 mel frames and 2560 samples"));

        // Load runs the same check against the sidecar's audio_length_per_tok.
        assert!(VoxtralRealtimeLayout::new(&config, &audio, 4, 16).is_err());
    }

    #[test]
    fn realtime_ceiling_honours_chunk_length_and_model_context() {
        let (mut config, _, layout) = published_realtime_layout();
        let encoding = &mut config
            .multimodal
            .whisper_model_args
            .encoder_args
            .audio_encoding_args;
        encoding.chunk_length_s = Some(10.0);
        assert_eq!(
            realtime_source_sample_ceiling(&config, &layout).unwrap(),
            160_000
        );

        // A context of 100 tokens leaves room for 100 - pads - 1 tokens of audio.
        config.model_max_length = 100;
        let audio_tokens = 100 - layout.left_pad_tokens - layout.right_pad_tokens - 1;
        assert_eq!(
            realtime_source_sample_ceiling(&config, &layout).unwrap(),
            audio_tokens * layout.raw_audio_length_per_tok
        );

        config
            .multimodal
            .whisper_model_args
            .encoder_args
            .audio_encoding_args
            .chunk_length_s = Some(f32::NAN);
        assert!(realtime_source_sample_ceiling(&config, &layout).is_err());
    }

    #[test]
    fn realtime_stream_peak_covers_transactional_overlap() {
        assert_eq!(
            checked_realtime_stream_peak_bytes(80, 12, 24, 16).unwrap(),
            (92, 64)
        );
        assert!(checked_realtime_stream_peak_bytes(u64::MAX, 1, 1, 1).is_err());
        assert!(checked_realtime_stream_peak_bytes(1, 1, u64::MAX, 1).is_err());
    }

    #[test]
    fn realtime_frontend_peak_accounts_for_host_stft_and_mel_buffers() {
        let peak = checked_realtime_frontend_host_peak_bytes(40, 10, 20, 4, 3, 8).unwrap();
        let expected = 80 // source/checkpoint/ingress/replacement relation
            + 40 // resampled
            + 80 // padded PCM
            + 112 // reflection-padded PCM
            + 64 // complex FFT frame
            + 20 // power spectrum
            + 48 // flat mel
            + 48 + 4 * std::mem::size_of::<Vec<f32>>() as u64; // nested mel
        assert_eq!(peak, expected);

        assert_eq!(checked_realtime_text_host_bytes(8, 3).unwrap(), (64, 160));
    }

    #[test]
    fn voxtral_dense_cuda_uses_checkpoint_dtype() {
        let profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: DeviceCapabilities {
                supports_bf16: true,
                supports_f16: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        assert_eq!(select_voxtral_dtype(&profile, Some(DType::F16)), DType::F16);
    }

    #[test]
    fn cuda_offline_frames_use_the_full_model_context() {
        assert_eq!(
            voxtral_realtime_offline_frame_limit(BackendKind::Cuda, 131_072, 131_072).unwrap(),
            131_072
        );
        assert!(voxtral_realtime_offline_frame_limit(BackendKind::Cuda, 131_073, 131_072).is_err());
    }

    #[test]
    fn portable_offline_frame_cap_is_a_hard_error() {
        for backend in [BackendKind::Cpu, BackendKind::Metal] {
            assert_eq!(
                voxtral_realtime_offline_frame_limit(backend, 1024, 8192).unwrap(),
                1024
            );
            for frames in [1025, 8192] {
                let error = voxtral_realtime_offline_frame_limit(backend, frames, 8192)
                    .expect_err("audio past the portable cap must not be truncated");
                assert!(matches!(error, crate::error::Error::InvalidInput(_)));
                assert!(format!("{error}").contains("1024-token CPU/Metal offline limit"));
            }
        }
    }

    #[test]
    fn voxtral_realtime_flash_options_match_causal_mask_window_width() {
        let options = voxtral_realtime_cuda_sliding_flash_options(Some(76));
        assert_eq!(options.window_size_left, Some(75));
        assert_eq!(options.window_size_right, None);
        assert!(options.alibi_slopes.is_none());

        let single_token = voxtral_realtime_cuda_sliding_flash_options(Some(1));
        assert_eq!(single_token.window_size_left, Some(0));
    }

    #[test]
    fn voxtral_runtime_config_reads_hf_sidecar_values() {
        let temp_dir =
            std::env::temp_dir().join(format!("izwi-voxtral-config-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir).unwrap();
        std::fs::write(
            temp_dir.join("config.json"),
            r#"{
                "default_num_delay_tokens": 6,
                "downsample_factor": 4,
                "audio_length_per_tok": 8,
                "dtype": "bfloat16"
            }"#,
        )
        .unwrap();

        let config = load_voxtral_runtime_config(&temp_dir).unwrap();

        assert_eq!(config.default_num_delay_tokens, Some(6));
        assert_eq!(config.downsample_factor, Some(4));
        assert_eq!(config.audio_length_per_tok, Some(8));
        assert_eq!(config.checkpoint_dtype, Some(DType::BF16));

        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn offline_streaming_padding_matches_mistral_token_padding_policy() {
        let (left, right) = offline_streaming_padding_samples(16_001, 1_280, 32, 17);

        assert_eq!(left, 32 * 1_280);
        assert_eq!(right, 639 + 17 * 1_280);
        assert_eq!((left + 16_001 + right) % 1_280, 0);
    }

    #[test]
    fn text_delta_preserves_word_boundary_space() {
        assert_eq!(text_delta("hello", "hello world"), " world");
    }

    #[test]
    fn realtime_conv_input_keeps_mels_as_channels() {
        let device = Device::Cpu;
        let mel = Tensor::zeros((80, 7), DType::F32, &device).unwrap();

        let conv_input = prepare_realtime_conv_input(&mel).unwrap();

        assert_eq!(conv_input.dims(), &[1, 80, 7]);
    }

    #[test]
    fn voxtral_mel_tensor_preserves_frame_major_layout() {
        let device = Device::Cpu;
        let mel = vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let tensor = mel_frames_to_tensor(&mel, 3, &device, DType::F32).unwrap();

        assert_eq!(tensor.dims(), &[1, 2, 3]);
        assert_eq!(
            tensor.to_vec3::<f32>().unwrap(),
            vec![vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]]
        );
    }

    #[test]
    fn voxtral_log_mel_normalization_uses_configured_global_max() {
        let mut mel = vec![vec![-10.0f32, -6.0, 2.0]];

        normalize_log_mel_with_max(&mut mel, 1.5);

        assert_eq!(mel, vec![vec![(-6.5 + 4.0) / 4.0, -0.5, 1.5]]);
    }

    #[test]
    fn voxtral_mel_drops_final_stft_frame() {
        let mut mel = vec![vec![1.0f32], vec![2.0], vec![3.0]];

        drop_last_mel_frame_for_voxtral(&mut mel);

        assert_eq!(mel, vec![vec![1.0f32], vec![2.0]]);
    }

    #[test]
    fn voxtral_audio_pooling_preserves_left_prefix_frames() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            (1, 5, 2),
            &device,
        )
        .unwrap();

        let pooled = pool_audio_embeddings_by_block(&x, 4).unwrap();

        assert_eq!(pooled.dims(), &[1, 1, 8]);
        assert_eq!(
            pooled.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn voxtral_audio_rope_uses_interleaved_pairs() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 10.0, 2.0, 20.0], (1, 1, 1, 4), &device).unwrap();
        let cos = Tensor::from_vec(vec![0.5f32, 0.25], (1, 2), &device).unwrap();
        let sin = Tensor::from_vec(vec![0.1f32, 0.2], (1, 2), &device).unwrap();

        let rotated = apply_interleaved_rotary_emb(&x, &cos, &sin).unwrap();

        assert_eq!(
            rotated.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![
                1.0 * 0.5 - 10.0 * 0.1,
                1.0 * 0.1 + 10.0 * 0.5,
                2.0 * 0.25 - 20.0 * 0.2,
                2.0 * 0.2 + 20.0 * 0.25,
            ]
        );
    }

    #[test]
    fn voxtral_audio_attention_scale_matches_activation_dtype() {
        let device = Device::Cpu;
        let reference = Tensor::zeros((1, 1, 1), DType::F16, &device).unwrap();

        let scale = attention_scale_tensor(0.125, &reference).unwrap();

        assert_eq!(scale.dtype(), DType::F16);
        assert_eq!(scale.dims(), &[1, 1, 1]);
    }

    #[test]
    fn voxtral_resample_audio_handles_empty_input() {
        assert_eq!(
            resample_audio(&[], 48_000, 16_000).unwrap(),
            Vec::<f32>::new()
        );
    }

    #[test]
    fn voxtral_low_rate_source_ceiling_is_checked_after_resampling() {
        let source_ceiling = 16_000;
        let source_samples = 16_000;

        let error = validate_realtime_sample_ceiling(source_samples, 8_000, 16_000, source_ceiling)
            .expect_err("target-domain expansion must not bypass the loaded ceiling");

        assert!(format!("{error}").contains("32000 at 16000 Hz"));
    }

    #[test]
    fn voxtral_workspace_charges_dense_attention_despite_sliding_mask() {
        let frames = 1_024;
        let heads = 16;
        let workspace = checked_workspace_bytes(0, frames, 1, 1, heads, Some(64), 2).unwrap();
        let dense_weights_and_mask = 2u64 * frames as u64 * frames as u64 * heads as u64 * 4;

        assert!(workspace >= dense_weights_and_mask);
    }

    #[test]
    fn voxtral_two_conv_valid_lengths_are_exact_for_unequal_rows() {
        let conv1 = Conv1dGeometry::causal(3, 1);
        let conv2 = Conv1dGeometry::causal(3, 2);
        assert_eq!((conv1.left_padding, conv1.right_padding), (2, 0));
        assert_eq!((conv2.left_padding, conv2.right_padding), (1, 0));
        // conv1 keeps every mel frame; conv2 keeps floor(frames / 2).
        for (mel, conv2_frames) in [(9, 4), (8, 4), (5, 2), (4, 2), (1, 0)] {
            let conv1_frames = conv_output_length(mel, conv1).unwrap();
            assert_eq!(conv1_frames, mel);
            assert_eq!(
                conv_output_length(conv1_frames, conv2).unwrap(),
                conv2_frames
            );
        }
    }

    #[test]
    fn voxtral_causal_conv_matches_geometry_and_never_looks_ahead() {
        let device = Device::Cpu;
        for spec in [Conv1dGeometry::causal(3, 1), Conv1dGeometry::causal(3, 2)] {
            let conv = candle_nn::Conv1d::new(
                Tensor::ones((1, 1, spec.kernel), DType::F32, &device).unwrap(),
                None,
                candle_nn::Conv1dConfig {
                    stride: spec.stride,
                    ..Default::default()
                },
            );
            let frames = 11;
            let base = (0..frames).map(|v| v as f32 + 1.0).collect::<Vec<_>>();
            let run = |input: &[f32]| {
                let x = Tensor::from_vec(input.to_vec(), (1, 1, frames), &device).unwrap();
                conv1d_with_geometry(&conv, spec, &x)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            };
            let output = run(&base);
            assert_eq!(output.len(), conv_output_length(frames, spec).unwrap());
            // Output 0 sees only zeros on the left plus the first input frame(s).
            let first_inputs = &base[..spec.kernel - spec.left_padding];
            assert_eq!(output[0], first_inputs.iter().sum::<f32>());

            for t in 0..frames {
                let mut perturbed = base.clone();
                perturbed[t] += 100.0;
                let changed = run(&perturbed);
                for (index, (before, after)) in output.iter().zip(&changed).enumerate() {
                    // Output `index` ends at input frame (index + 1) * stride - 1.
                    let last_input = (index + 1) * spec.stride - 1;
                    if last_input < t {
                        assert_eq!(before, after, "output {index} saw later frame {t}");
                    } else if index == t / spec.stride {
                        assert_ne!(before, after, "output {index} ignored its frame {t}");
                    }
                }
            }
        }
    }

    #[test]
    fn voxtral_valid_attention_mask_isolates_poison_padding() {
        let device = Device::Cpu;
        let mask = create_valid_attention_mask(&[2, 4], 1, 4, true, Some(2), &device, DType::F32)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        assert_eq!(mask[0][0][0], 0.0);
        assert_eq!(mask[0][1][0], 0.0);
        assert_eq!(mask[0][1][1], 0.0);
        assert_eq!(mask[0][1][2], f32::MIN);
        assert_eq!(mask[1][3][1], f32::MIN);
        assert_eq!(mask[1][3][2], 0.0);
        assert_eq!(mask[1][3][3], 0.0);
    }

    #[test]
    fn voxtral_push_and_finish_geometry_partition_the_tail() {
        let conv = Conv1dGeometry::causal(3, 2);
        let layout = VoxtralRealtimeLayout {
            target_rate: 16_000,
            hop_length: 160,
            n_mels: 80,
            raw_audio_length_per_tok: 1_280,
            left_pad_tokens: 1,
            right_pad_tokens: 3,
            pool_size: 4,
            text_dim: 16,
            conv1: conv,
            conv2: conv,
            encoder_hidden: 8,
            encoder_ffn: 16,
            encoder_heads: 2,
            encoder_sliding_window: None,
        };
        let geometry = |mode| realtime_preparation_geometry_for(3_201, 16_000, &layout, mode);
        let push = geometry(VoxtralRealtimePreparationMode::Push).unwrap();
        let finish = geometry(VoxtralRealtimePreparationMode::Finish).unwrap();
        assert!(finish.padded_samples > push.padded_samples);
        assert!(finish.pooled_frames >= push.pooled_frames);
        assert_eq!(push.stable_frames, push.pooled_frames.saturating_sub(3));
        assert_eq!(finish.stable_frames, finish.pooled_frames);
        assert_eq!(
            finish,
            geometry(VoxtralRealtimePreparationMode::Finish).unwrap()
        );
    }

    #[test]
    fn voxtral_resample_audio_rejects_zero_sample_rates() {
        let err = resample_audio(&[0.0], 0, 16_000)
            .expect_err("zero input sample rate should be rejected");

        assert!(format!("{err}").contains("Invalid audio sample rate"));
    }

    #[test]
    fn voxtral_argmax_selects_from_half_logits_on_device() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.1f32, 0.7, -0.2], (3,), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        assert_eq!(argmax(&logits).unwrap(), 1);
    }

    #[test]
    fn voxtral_argmax_rejects_batched_rank2_logits() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.1f32; 8], (2, 4), &device).unwrap();
        let err = argmax(&logits).expect_err("rank2 batch > 1 should be rejected");

        assert!(format!("{err}").contains("Unexpected batched Voxtral logits"));
    }

    #[test]
    fn voxtral_argmax_accepts_singleton_rank2_logits() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.1f32, -0.2, 0.9, 0.4], (1, 4), &device).unwrap();

        assert_eq!(argmax(&logits).unwrap(), 2);
    }

    #[test]
    fn voxtral_argmax_rejects_empty_logits() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(Vec::<f32>::new(), (0,), &device).unwrap();
        let err = argmax(&logits).expect_err("empty logits should be rejected");

        assert!(format!("{err}").contains("Voxtral logits are empty"));
    }

    #[test]
    fn delay_tokens_must_match_the_config_sidecar() {
        let audio = AudioConfig {
            n_delay_tokens: 6,
            ..AudioConfig::default()
        };
        let sidecar = |delay| VoxtralRuntimeConfig {
            default_num_delay_tokens: delay,
            ..VoxtralRuntimeConfig::default()
        };
        assert_eq!(checked_delay_tokens(&audio, &sidecar(None)).unwrap(), 6);
        assert_eq!(checked_delay_tokens(&audio, &sidecar(Some(6))).unwrap(), 6);
        let error = checked_delay_tokens(&audio, &sidecar(Some(5))).unwrap_err();
        assert!(format!("{error}").contains("default_num_delay_tokens is 5"));
    }

    #[test]
    fn generated_text_streams_split_characters_and_flushes_partial_ones() {
        let tokenizer = published_excerpt_tokenizer();
        let mut deltas = Vec::new();
        let mut record = |delta: &str| deltas.push(delta.to_string());
        let mut on_delta: Option<&mut dyn FnMut(&str)> = Some(&mut record);
        let mut generated = Vec::new();
        let mut assembled = String::new();
        // "你" (E4 BD A0), then the first byte of another CJK character.
        let mut appended = Vec::new();
        for token in published_byte_ids(&["你".as_bytes(), &[0xE5]].concat()) {
            appended.push(
                append_generated_token(
                    &tokenizer,
                    &mut generated,
                    &mut assembled,
                    token,
                    &mut on_delta,
                )
                .unwrap(),
            );
        }
        assert_eq!(appended, ["", "", "你", ""]);
        assert_eq!(assembled, "你");

        let flushed =
            flush_generated_text(&tokenizer, &generated, &mut assembled, &mut on_delta).unwrap();
        assert_eq!(flushed, "\u{FFFD}");
        assert_eq!(assembled, "你\u{FFFD}");
        assert_eq!(deltas, ["你", "\u{FFFD}"]);
    }

    #[test]
    fn offline_audio_hint_fits_the_portable_frame_cap() {
        let (_, _, layout) = published_realtime_layout();
        let frames = |seconds: f32| {
            realtime_preparation_geometry_for(
                (seconds * 16_000.0) as usize,
                16_000,
                &layout,
                VoxtralRealtimePreparationMode::Finish,
            )
            .unwrap()
            .pooled_frames
        };
        assert_eq!(frames(VOXTRAL_REALTIME_OFFLINE_AUDIO_SECONDS_HINT), 424);
        // At 12.5 Hz the 1024-token cap is reached at 78 s of audio.
        assert_eq!(frames(78.0), PORTABLE_OFFLINE_AUDIO_FRAME_LIMIT);
        assert!(frames(78.1) > PORTABLE_OFFLINE_AUDIO_FRAME_LIMIT);
    }

    /// The Voxtral load-time probe on a tiny model: encoder, adapter and
    /// logits finite, and the F16 run within F16 rounding of F32.
    fn voxtral_probe_outputs(
        weights: &std::collections::HashMap<String, Tensor>,
        device: &Device,
        dtype: DType,
    ) -> Vec<(String, Vec<f32>)> {
        use super::super::audio::{AudioLanguageAdapter, TimeEmbedding};
        use super::{voxtral_numeric_probe, VoxtralProbeParts, WhisperEncoder};
        use crate::kernels::numeric_check::host_f32;
        use crate::models::architectures::voxtral::lm::tests::tiny_ada_model_with_dtype;

        let cfg = voxtral_probe_encoder_config();
        let tensors = weights
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.to_device(device).unwrap()))
            .collect();
        let vb = candle_nn::VarBuilder::from_tensors(tensors, dtype, device);
        let encoder = WhisperEncoder::load_voxtral(&cfg, vb.pp("encoder")).unwrap();
        let adapter = AudioLanguageAdapter::load(4 * 32, 4, vb.pp("adapter")).unwrap();
        let lm = tiny_ada_model_with_dtype(device, 2, dtype);
        let t_cond = TimeEmbedding::new(4, 10_000.0, device)
            .unwrap()
            .forward(&Tensor::new(&[6f32], device).unwrap())
            .unwrap()
            .to_dtype(lm.embedding_dtype())
            .unwrap();
        let ada = lm.ada_scales(&t_cond).unwrap();
        voxtral_numeric_probe(
            VoxtralProbeParts {
                encoder: &encoder,
                adapter: &adapter,
                pool_size: 4,
                language_model: &lm,
                ada: &ada,
                n_mels: 8,
            },
            dtype,
            device,
        )
        .unwrap()
        .into_iter()
        .map(|(name, tensor)| (name, host_f32(&tensor).unwrap()))
        .collect()
    }

    fn voxtral_probe_encoder_config() -> super::super::config::AudioEncoderConfig {
        super::super::config::AudioEncoderConfig {
            d_model: 32,
            encoder_layers: 2,
            encoder_attention_heads: 4,
            encoder_ffn_dim: 64,
            num_mel_bins: 8,
            max_source_positions: 512,
            window_size: 400,
            hop_length: 160,
            sampling_rate: 16_000,
            is_causal: true,
            conv1_kernel_size: 3,
            conv1_stride: 1,
            conv2_kernel_size: 3,
            conv2_stride: 2,
            global_log_mel_max: None,
            head_dim: 8,
            rope_theta: 1_000_000.0,
            pos_embed: "rope".into(),
            sliding_window: 16,
        }
    }

    /// Random encoder and adapter weights for [`voxtral_probe_outputs`].
    fn voxtral_probe_weights() -> std::collections::HashMap<String, Tensor> {
        use super::super::audio::AudioLanguageAdapter;
        use super::WhisperEncoder;

        let cfg = voxtral_probe_encoder_config();
        let vars = candle_nn::VarMap::new();
        let host = candle_nn::VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu);
        WhisperEncoder::load_voxtral(&cfg, host.pp("encoder")).unwrap();
        AudioLanguageAdapter::load(4 * 32, 4, host.pp("adapter")).unwrap();
        let data = vars.data().lock().unwrap();
        data.iter()
            .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
            .collect()
    }

    fn assert_voxtral_probe_tracks_f32(device: &Device, dtype: DType) {
        use crate::kernels::numeric_check::relative_l2;
        let weights = voxtral_probe_weights();
        let low = voxtral_probe_outputs(&weights, device, dtype);
        let reference = voxtral_probe_outputs(&weights, &Device::Cpu, DType::F32);
        assert_eq!(low.len(), 3);
        for ((name, low), (_, reference)) in low.iter().zip(&reference) {
            assert!(low.iter().all(|value| value.is_finite()), "{name}");
            let rel = relative_l2(low, reference);
            assert!(rel < 2e-2, "{name}: {dtype:?} vs F32 relative L2 {rel:e}");
        }
        // 16 LM positions from the 128-frame window.
        assert_eq!(low[2].1.len(), 16 * 16);
    }

    #[test]
    fn voxtral_numeric_probe_runs_encoder_adapter_and_lm_on_cpu() {
        assert_voxtral_probe_tracks_f32(&Device::Cpu, DType::F32);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_voxtral_numeric_probe_in_f16_tracks_cpu_f32() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "metal_voxtral_numeric_probe_in_f16_tracks_cpu_f32",
        ) else {
            return;
        };
        assert_voxtral_probe_tracks_f32(&device, DType::F16);
    }
}
