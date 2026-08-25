//! Main Voxtral Realtime model implementation.

use std::borrow::Cow;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_core::{DType, IndexOp, Tensor, D};
use candle_nn::{Module, VarBuilder};
use tracing::info;

use crate::audio::{MelConfig, MelNorm, MelScale, MelSpectrogram};
use crate::backends::{BackendKind, DeviceProfile};
use crate::catalog::ModelFamily;
use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::kv::CacheDomainId;
use crate::models::architectures::qwen3::core::build_rope_cache;
use crate::models::architectures::voxtral::lm::VoxtralLM;
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
    time_embedding: TimeEmbedding,
    mel: MelSpectrogram,
    global_log_mel_max: Option<f32>,
    num_delay_tokens: usize,
    streaming_left_pad_tokens: usize,
    offline_left_pad_tokens: usize,
    streaming_right_pad_tokens: usize,
    raw_audio_length_per_tok: usize,
    block_pool_size: usize,
    audio_length_per_tok: usize,
    realtime_max_source_samples: Option<usize>,
    max_decoded_token_bytes: usize,
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

const PORTABLE_OFFLINE_AUDIO_FRAME_LIMIT: usize = 1024;

fn voxtral_realtime_offline_frame_limit(
    backend: BackendKind,
    audio_frames: usize,
    model_context_limit: usize,
) -> Result<usize> {
    if backend != BackendKind::Cuda {
        return Ok(audio_frames.min(PORTABLE_OFFLINE_AUDIO_FRAME_LIMIT));
    }
    if model_context_limit == 0 {
        return Err(Error::ModelLoadError(
            "Voxtral Realtime reported a zero physical context".into(),
        ));
    }
    if audio_frames > model_context_limit {
        return Err(Error::InvalidInput(format!(
            "Voxtral Realtime produced {audio_frames} audio tokens, exceeding its {model_context_limit}-token loaded model context"
        )));
    }
    Ok(audio_frames)
}

impl VoxtralRealtimeModel {
    /// Load model from directory
    pub fn load(model_dir: &Path, device: DeviceProfile) -> Result<Self> {
        // Try params.json (Voxtral format) first, then config.json (standard)
        let config_path = if model_dir.join("params.json").exists() {
            model_dir.join("params.json")
        } else {
            model_dir.join("config.json")
        };
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::ModelLoadError(format!("Failed to read config: {}", e)))?;
        let config: VoxtralConfig = serde_json::from_str(&config_str)
            .map_err(|e| Error::ModelLoadError(format!("Failed to parse config: {}", e)))?;
        let runtime_config = load_voxtral_runtime_config(model_dir)?;
        let checkpoint_dtype =
            checkpoint_dtype_from_config_json(&config_str).or(runtime_config.checkpoint_dtype);
        let num_delay_tokens = runtime_config
            .default_num_delay_tokens
            .unwrap_or_else(|| config.num_delay_tokens())
            .max(1);
        let block_pool_size = runtime_config
            .downsample_factor
            .unwrap_or_else(|| config.block_pool_size())
            .max(1);
        let audio_length_per_tok = runtime_config
            .audio_length_per_tok
            .unwrap_or(block_pool_size.saturating_mul(2))
            .max(1);

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
        let audio_config = AudioConfig {
            sampling_rate: audio_cfg.sampling_rate,
            frame_rate: config.frame_rate(),
            window_size: audio_cfg.window_size,
            hop_length: audio_cfg.hop_length,
            num_mel_bins: audio_cfg.num_mel_bins,
            n_delay_tokens: num_delay_tokens,
            ..AudioConfig::default()
        };
        let tokenizer = VoxtralTokenizer::load(model_dir, audio_config)?;
        let max_decoded_token_bytes = tokenizer.max_decoded_token_bytes()?;
        let streaming_left_pad_tokens = tokenizer.audio_config().streaming_left_pad_tokens;
        let offline_left_pad_tokens =
            offline_left_pad_tokens_for_generation(streaming_left_pad_tokens, num_delay_tokens);
        let streaming_right_pad_tokens = tokenizer.audio_config().offline_right_pad_tokens();
        let raw_audio_length_per_tok = tokenizer.audio_config().raw_audio_length_per_tok();
        let realtime_max_source_samples = realtime_source_sample_ceiling(
            &config,
            &audio_cfg,
            raw_audio_length_per_tok,
            offline_left_pad_tokens,
            streaming_right_pad_tokens,
        )?;

        let dtype = select_voxtral_dtype(&device, checkpoint_dtype);

        // Load weights - clone device to a local binding for lifetime
        let device_clone = device.clone();
        let vb = load_weights(model_dir, dtype, &device_clone)?;

        // Load components
        // Note: Checkpoint uses mm_streams_embeddings.embedding_module prefix for audio components
        let whisper_prefix = "mm_streams_embeddings.embedding_module.whisper_encoder";
        let whisper_encoder = WhisperEncoder::load_voxtral(&audio_cfg, vb.pp(whisper_prefix))?;

        let hidden_size = audio_cfg.d_model * block_pool_size;
        let audio_adapter = AudioLanguageAdapter::load(
            hidden_size,
            config.text_config().hidden_size,
            vb.pp("mm_streams_embeddings.embedding_module.audio_language_projection"),
        )?;

        // Language model uses root-level layers.* and norm (Mistral-style)
        let language_model = VoxtralLM::load(config.text_config().into(), vb.clone())?;

        let time_embedding =
            TimeEmbedding::new(config.text_config().hidden_size, 10000.0, &device.device)?;

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
            time_embedding,
            mel,
            global_log_mel_max: audio_cfg.global_log_mel_max,
            num_delay_tokens,
            streaming_left_pad_tokens,
            offline_left_pad_tokens,
            streaming_right_pad_tokens,
            raw_audio_length_per_tok,
            block_pool_size,
            audio_length_per_tok,
            realtime_max_source_samples,
            max_decoded_token_bytes,
        })
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
        let mut prompt_tokens = self.tokenizer.build_transcription_prompt()?;
        prompt_tokens.truncate(voxtral_generation_prefix_len(prompt_tokens.len()));
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
        let t_cond = self.realtime_time_condition(&prompt_embeds)?;
        let logits = self.language_model.forward_managed_with_embeds(
            &prompt_embeds,
            0,
            cache,
            None,
            Some(&t_cond),
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
        let prompt_len =
            voxtral_generation_prefix_len(self.tokenizer.build_transcription_prompt()?.len());
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
        state.finished = true;
        Ok(self.realtime_step(state, String::new()))
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
            let mut condition_rows = Vec::with_capacity(rows.len());
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
                condition_rows.push(self.realtime_time_condition(&step_embeds)?);
                embed_rows.push(step_embeds);
                positions.push(row.state.next_audio_frame);
            }
            let embeds = Tensor::cat(&embed_rows.iter().collect::<Vec<_>>(), 0)?;
            let conditions = Tensor::stack(&condition_rows.iter().collect::<Vec<_>>(), 0)?;
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
                        Some(&conditions),
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
                state.finished = true;
                return Ok(Some(self.realtime_step(state, String::new())));
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
        let t_cond = self.realtime_time_condition(&step_embeds)?;
        let logits = self.language_model.forward_managed_with_embeds(
            &step_embeds,
            state.next_audio_frame,
            cache,
            None,
            Some(&t_cond),
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
        if backend == BackendKind::Cuda {
            let (left_pad, right_pad) = offline_streaming_padding_samples(
                audio.len(),
                self.raw_audio_length_per_tok,
                self.offline_left_pad_tokens,
                self.streaming_right_pad_tokens,
            );
            let predicted_audio_frames = left_pad
                .checked_add(audio.len())
                .and_then(|samples| samples.checked_add(right_pad))
                .ok_or_else(|| Error::InvalidInput("Voxtral audio length overflow".into()))?
                / self.raw_audio_length_per_tok.max(1);
            voxtral_realtime_offline_frame_limit(
                backend,
                predicted_audio_frames,
                model_context_limit,
            )?;
        }

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

        // Apply time conditioning
        let time_tensor = Tensor::from_vec(
            vec![self.num_delay_tokens as f32],
            (1,),
            &self.device.device,
        )?
        .to_dtype(self.dtype)?;
        let t_cond = self.time_embedding.forward(&time_tensor)?;

        // Generate
        let mut prompt_tokens = self.tokenizer.build_transcription_prompt()?;
        prompt_tokens.truncate(voxtral_generation_prefix_len(prompt_tokens.len()));
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
        let prompt_t_cond = if t_cond.dtype() == prompt_embeds.dtype() {
            t_cond.clone()
        } else {
            t_cond.to_dtype(prompt_embeds.dtype())?
        };
        let logits = self.language_model.forward_managed_with_embeds(
            &prompt_embeds,
            0,
            cache,
            None,
            Some(&prompt_t_cond),
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
            let step_t_cond = if t_cond.dtype() == step_embeds.dtype() {
                t_cond.clone()
            } else {
                t_cond.to_dtype(step_embeds.dtype())?
            };

            let logits = self.language_model.forward_managed_with_embeds(
                &step_embeds,
                frame_idx,
                cache,
                None,
                Some(&step_t_cond),
            )?;
            let next_logits = logits.i((0, logits.dim(1)? - 1))?;
            next = argmax(&next_logits)?;
            frame_idx += 1;
        }
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
            self.raw_audio_length_per_tok,
            self.offline_left_pad_tokens,
            self.streaming_right_pad_tokens,
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
        realtime_preparation_geometry_for(
            source_samples,
            sample_rate,
            self.mel.config().sample_rate as u32,
            self.mel.config().hop_length,
            self.mel.config().n_mels,
            self.raw_audio_length_per_tok,
            self.offline_left_pad_tokens,
            self.streaming_right_pad_tokens,
            self.block_pool_size,
            self.config.text_dim,
            self.whisper_encoder.conv1_spec,
            self.whisper_encoder.conv2_spec,
            mode,
        )
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
        let materialized_tensor_elements_per_row =
            self.realtime_materialized_elements(padded_mel_frames, padded_conv_frames)?;
        let workspace_per_row_bytes =
            self.realtime_workspace_bytes(padded_mel_frames, padded_conv_frames)?;
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
        let max_source_samples = self.realtime_max_source_samples.ok_or_else(|| {
            Error::ModelLoadError(
                "Voxtral realtime preparation is disabled because chunk_length_s is absent".into(),
            )
        })?;
        let max = realtime_preparation_geometry_for(
            max_source_samples,
            self.mel.config().sample_rate as u32,
            self.mel.config().sample_rate as u32,
            self.mel.config().hop_length,
            self.mel.config().n_mels,
            self.raw_audio_length_per_tok,
            self.offline_left_pad_tokens,
            self.streaming_right_pad_tokens,
            self.block_pool_size,
            self.config.text_dim,
            self.whisper_encoder.conv1_spec,
            self.whisper_encoder.conv2_spec,
            VoxtralRealtimePreparationMode::Finish,
        )?;
        Ok(VoxtralRealtimePreparationStageSeal {
            max_source_samples,
            max_work_units: u64::try_from(max_source_samples).map_err(|_| {
                Error::ModelLoadError("Voxtral realtime work ceiling exceeds u64".into())
            })?,
            max_materialized_tensor_elements_per_row: self
                .realtime_materialized_elements(max.mel_frames, max.conv2_frames)?,
            max_workspace_bytes: self.realtime_workspace_bytes(max.mel_frames, max.conv2_frames)?,
        })
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
            self.mel.config().sample_rate as u32,
            self.mel.config().sample_rate as u32,
            self.mel.config().hop_length,
            self.mel.config().n_mels,
            self.raw_audio_length_per_tok,
            self.offline_left_pad_tokens,
            self.streaming_right_pad_tokens,
            self.block_pool_size,
            self.config.text_dim,
            self.whisper_encoder.conv1_spec,
            self.whisper_encoder.conv2_spec,
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
        if source_samples == 0
            || source_samples > self.realtime_preparation_stage_seal()?.max_source_samples
        {
            return Err(Error::InvalidInput(
                "Voxtral cumulative source samples are outside the sealed realtime range".into(),
            ));
        }
        realtime_preparation_geometry_for(
            source_samples,
            sample_rate,
            self.mel.config().sample_rate as u32,
            self.mel.config().hop_length,
            self.mel.config().n_mels,
            self.raw_audio_length_per_tok,
            self.offline_left_pad_tokens,
            self.streaming_right_pad_tokens,
            self.block_pool_size,
            self.config.text_dim,
            self.whisper_encoder.conv1_spec,
            self.whisper_encoder.conv2_spec,
            mode,
        )
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
                .raw_audio_length_per_tok
                .checked_mul(self.offline_left_pad_tokens)
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
        if self.realtime_max_source_samples.is_none() {
            return Err(Error::ModelLoadError(
                "Voxtral realtime preparation requires a finite chunk_length_s".into(),
            ));
        }
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
        let target_rate = self.mel.config().sample_rate as u32;
        let ceiling = self.realtime_max_source_samples.unwrap();
        validate_realtime_sample_ceiling(total, sample_rate, target_rate, ceiling)?;
        Ok(())
    }

    fn realtime_materialized_elements(&self, mel_frames: usize, conv_frames: usize) -> Result<u64> {
        let conv1_frames = conv_output_length(mel_frames, self.whisper_encoder.conv1_spec)?;
        checked_materialized_elements(
            mel_frames,
            conv1_frames,
            conv_frames,
            self.mel.config().n_mels,
            self.whisper_encoder.hidden_size,
            self.config.text_dim,
        )
    }

    fn realtime_workspace_bytes(&self, mel_frames: usize, conv_frames: usize) -> Result<u64> {
        checked_workspace_bytes(
            self.realtime_materialized_elements(mel_frames, conv_frames)?,
            conv_frames,
            self.whisper_encoder.hidden_size,
            self.whisper_encoder.ffn_dim,
            self.whisper_encoder.num_heads,
            self.whisper_encoder.sliding_window,
            self.dtype.size_in_bytes(),
        )
    }

    fn realtime_time_condition(&self, reference: &Tensor) -> Result<Tensor> {
        let time = Tensor::from_vec(
            vec![self.num_delay_tokens as f32],
            (1,),
            &self.device.device,
        )?
        .to_dtype(self.dtype)?;
        let condition = self.time_embedding.forward(&time)?;
        if condition.dtype() == reference.dtype() {
            Ok(condition)
        } else {
            condition.to_dtype(reference.dtype()).map_err(Error::from)
        }
    }

    fn accept_realtime_prediction(
        &self,
        state: &mut VoxtralRealtimeState,
        token: u32,
    ) -> Result<VoxtralRealtimeStep> {
        let specials = self.tokenizer.specials();
        if token == specials.eos || Some(token) == specials.end_audio {
            state.pending_input_token = None;
            state.finished = true;
            return Ok(self.realtime_step(state, String::new()));
        }
        let previous = state.assembled.clone();
        let mut no_callback = None;
        append_generated_token(
            &self.tokenizer,
            &mut state.generated,
            &mut state.assembled,
            token,
            &mut no_callback,
        )?;
        state.pending_input_token = Some(token);
        let delta = text_delta(&previous, &state.assembled);
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
                "flash_attention_requested": flash_attention_requested(),
                "flash_attention_compiled": flash_attention_compiled(),
                "cache": cache,
                "timings_ms": timings.map(VoxtralRealtimeTimings::to_json)
            },
            "config": {
                "num_delay_tokens": self.num_delay_tokens,
                "streaming_left_pad_tokens": self.streaming_left_pad_tokens,
                "offline_left_pad_tokens": self.offline_left_pad_tokens,
                "streaming_right_pad_tokens": self.streaming_right_pad_tokens,
                "raw_audio_length_per_tok": self.raw_audio_length_per_tok,
                "block_pool_size": self.block_pool_size,
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
        pool_audio_embeddings_by_block(audio_embeds, self.block_pool_size)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Conv1dGeometry {
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

fn conv_output_length(input: usize, spec: Conv1dGeometry) -> Result<usize> {
    if spec.kernel == 0 || spec.stride == 0 || spec.dilation == 0 {
        return Err(Error::ModelLoadError(
            "Voxtral encoder has invalid convolution geometry".into(),
        ));
    }
    let padded =
        input
            .checked_add(spec.padding.checked_mul(2).ok_or_else(|| {
                Error::InvalidInput("Voxtral convolution padding overflow".into())
            })?)
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
    hidden_size: usize,
    ffn_dim: usize,
    num_heads: usize,
    sliding_window: Option<usize>,
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
        let conv1_config = candle_nn::Conv1dConfig {
            stride: cfg.conv1_stride,
            padding: 1,
            groups: 1,
            dilation: 1,
            ..Default::default()
        };
        let conv2_config = candle_nn::Conv1dConfig {
            stride: cfg.conv2_stride,
            padding: 1,
            groups: 1,
            dilation: 1,
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
            conv1_spec: Conv1dGeometry {
                kernel: cfg.conv1_kernel_size,
                stride: cfg.conv1_stride,
                padding: 1,
                dilation: 1,
            },
            conv2_spec: Conv1dGeometry {
                kernel: cfg.conv2_kernel_size,
                stride: cfg.conv2_stride,
                padding: 1,
                dilation: 1,
            },
            hidden_size: cfg.d_model,
            ffn_dim: cfg.encoder_ffn_dim,
            num_heads: cfg.encoder_attention_heads,
            sliding_window: (cfg.is_causal && cfg.sliding_window > 0).then_some(cfg.sliding_window),
        })
    }

    pub fn forward(&self, input_features: &Tensor) -> Result<Tensor> {
        // input_features: [batch, frames, n_mels]
        // Transpose to [batch, n_mels, frames] for conv1d
        let x = input_features.transpose(1, 2)?;

        // Conv layers with gelu
        let x = self.conv1.forward(&x)?;
        let x = gelu(&x)?;
        let x = self.conv2.forward(&x)?;
        let x = gelu(&x)?;

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
        for layer in &self.layers {
            x = layer.forward(&x, self.is_causal)?;
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
        let x = gelu(&self.conv1.forward(&x)?)?;
        let conv1_frames = x.dim(2)?;
        let conv1_mask = channel_valid_mask(&conv1_lengths, conv1_frames, &x)?;
        let x = x.broadcast_mul(&conv1_mask)?;

        let conv2_lengths = conv1_lengths
            .iter()
            .map(|length| conv_output_length(*length, self.conv2_spec))
            .collect::<Result<Vec<_>>>()?;
        let x = gelu(&self.conv2.forward(&x)?)?;
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
        for layer in &self.layers {
            x = layer.forward_valid_lengths(&x, self.is_causal, &conv2_lengths, &frame_mask)?;
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
            let x = self.conv1.forward(&x)?;
            let x = gelu(&x)?;
            let x = self.conv2.forward(&x)?;
            let x = gelu(&x)?;
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

    pub fn forward(&self, x: &Tensor, is_causal: bool) -> Result<Tensor> {
        let residual = x;
        let x = if self.is_voxtral {
            self.self_attn_rms_norm.as_ref().unwrap().forward(x)?
        } else {
            self.self_attn_layer_norm.as_ref().unwrap().forward(x)?
        };
        let x = self.self_attn.forward(&x, is_causal)?;
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
            .forward_valid_lengths(&normalized, is_causal, valid_lengths, frame_mask)?
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

    pub fn forward(&self, x: &Tensor, is_causal: bool) -> Result<Tensor> {
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
            let (cos, sin) = build_rope_cache(
                seq_len,
                self.head_dim,
                0,
                self.rope_theta,
                x.device(),
                q.dtype(),
            )?;
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
            let mask = create_causal_mask(
                seq_len,
                self.sliding_window,
                x.device(),
                attn_weights.dtype(),
            )?;
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
            let (cos, sin) = build_rope_cache(
                seq_len,
                self.head_dim,
                0,
                self.rope_theta,
                x.device(),
                q.dtype(),
            )?;
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
        let mask = create_valid_attention_mask(
            valid_lengths,
            self.num_heads,
            seq_len,
            is_causal,
            self.sliding_window,
            x.device(),
            weights.dtype(),
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

fn voxtral_generation_prefix_len(standard_prompt_len: usize) -> usize {
    standard_prompt_len.saturating_sub(1).max(1)
}

fn append_generated_token(
    tokenizer: &VoxtralTokenizer,
    generated: &mut Vec<u32>,
    assembled: &mut String,
    token: u32,
    on_delta: &mut Option<&mut dyn FnMut(&str)>,
) -> Result<()> {
    generated.push(token);
    let decoded = tokenizer.decode_text(generated)?;
    let delta = text_delta(assembled, &decoded);
    if let Some(on_delta) = on_delta.as_deref_mut() {
        for ch in delta.chars() {
            let mut buf = [0u8; 4];
            on_delta(ch.encode_utf8(&mut buf));
        }
    }
    *assembled = decoded;
    Ok(())
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
            "Voxtral realtime stream has {source_samples} source samples ({resampled_samples} at {target_rate} Hz), exceeding its loaded ceiling"
        )));
    }
    Ok(resampled_samples)
}

#[allow(clippy::too_many_arguments)]
fn realtime_preparation_geometry_for(
    source_samples: usize,
    source_rate: u32,
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
    mode: VoxtralRealtimePreparationMode,
) -> Result<VoxtralRealtimePreparationGeometry> {
    if hop_length == 0 || n_mels == 0 || pool_size == 0 || text_dim == 0 {
        return Err(Error::ModelLoadError(
            "Voxtral realtime preparation has zero-sized model geometry".into(),
        ));
    }
    let resampled_samples = resampled_length(source_samples, source_rate, target_rate)?;
    let left_pad = raw_audio_length_per_tok
        .checked_mul(left_pad_tokens)
        .ok_or_else(|| Error::InvalidInput("Voxtral left padding overflow".into()))?;
    let right_pad = if mode == VoxtralRealtimePreparationMode::Finish {
        offline_streaming_padding_samples(
            resampled_samples,
            raw_audio_length_per_tok,
            0,
            right_pad_tokens,
        )
        .1
    } else {
        0
    };
    let padded_samples = left_pad
        .checked_add(resampled_samples)
        .and_then(|length| length.checked_add(right_pad))
        .ok_or_else(|| Error::InvalidInput("Voxtral padded audio length overflow".into()))?;
    let mel_frames = padded_samples / hop_length;
    let conv1_frames = conv_output_length(mel_frames, conv1)?;
    let conv2_frames = conv_output_length(conv1_frames, conv2)?;
    let pooled_frames = conv2_frames / pool_size;
    let stable_frames = if mode == VoxtralRealtimePreparationMode::Finish {
        pooled_frames
    } else {
        pooled_frames.saturating_sub(right_pad_tokens)
    };
    let embedding_elements = u64::try_from(pooled_frames)
        .ok()
        .and_then(|frames| {
            u64::try_from(text_dim)
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

fn realtime_source_sample_ceiling(
    config: &VoxtralConfig,
    audio: &super::config::AudioEncoderConfig,
    raw_audio_length_per_tok: usize,
    left_pad_tokens: usize,
    right_pad_tokens: usize,
) -> Result<Option<usize>> {
    let Some(seconds) = config
        .multimodal
        .whisper_model_args
        .encoder_args
        .audio_encoding_args
        .chunk_length_s
    else {
        return Ok(None);
    };
    if !seconds.is_finite() || seconds <= 0.0 || audio.sampling_rate == 0 {
        return Err(Error::ModelLoadError(
            "Voxtral chunk_length_s must be finite and positive".into(),
        ));
    }
    let policy_samples = (seconds as f64 * audio.sampling_rate as f64).ceil();
    if policy_samples > usize::MAX as f64 {
        return Err(Error::ModelLoadError(
            "Voxtral realtime sample policy exceeds usize".into(),
        ));
    }
    let model_samples = config
        .model_max_length
        .checked_mul(raw_audio_length_per_tok)
        .and_then(|total| {
            raw_audio_length_per_tok
                .checked_mul(
                    left_pad_tokens
                        .saturating_add(right_pad_tokens)
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
    Ok(Some(ceiling))
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

fn offline_left_pad_tokens_for_generation(
    streaming_left_pad_tokens: usize,
    num_delay_tokens: usize,
) -> usize {
    let generation_prefix_tokens = streaming_left_pad_tokens.saturating_add(num_delay_tokens);
    generation_prefix_tokens
        .saturating_mul(2)
        .max(streaming_left_pad_tokens)
}

fn load_weights<'a>(
    model_dir: &'a Path,
    dtype: DType,
    device: &'a DeviceProfile,
) -> Result<VarBuilder<'a>> {
    // Voxtral uses consolidated.safetensors (single file)
    let consolidated_path = model_dir.join("consolidated.safetensors");
    if consolidated_path.exists() {
        info!("Loading Voxtral from consolidated.safetensors");
        unsafe {
            return VarBuilder::from_mmaped_safetensors(
                &[consolidated_path],
                dtype,
                &device.device,
            )
            .map_err(|e| Error::ModelLoadError(format!("Failed to load weights: {}", e)));
        }
    }

    let index_path = model_dir.join("model.safetensors.index.json");

    if index_path.exists() {
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

        let shard_paths: Vec<std::path::PathBuf> =
            shard_files.iter().map(|f| model_dir.join(f)).collect();

        info!("Loading Voxtral with {} shard files", shard_paths.len());

        unsafe {
            VarBuilder::from_mmaped_safetensors(&shard_paths, dtype, &device.device)
                .map_err(|e| Error::ModelLoadError(format!("Failed to load shards: {}", e)))
        }
    } else {
        let weights_path = model_dir.join("model.safetensors");
        unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device.device)
                .map_err(|e| Error::ModelLoadError(format!("Failed to load weights: {}", e)))
        }
    }
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

fn select_voxtral_dtype(device: &DeviceProfile, checkpoint_dtype: Option<DType>) -> DType {
    device.select_model_dtype_with_checkpoint(ModelFamily::Voxtral, checkpoint_dtype)
}

/// GELU activation function
fn gelu(x: &Tensor) -> Result<Tensor> {
    let coeff = 0.044715f32;
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(candle_core::DType::F32)?;
    let x3 = x_f32.powf(3.0)?;
    let coeff_t = Tensor::from_vec(vec![coeff], (1,), x.device())?;
    let x3 = x3.broadcast_mul(&coeff_t)?;
    let sqrt_t = Tensor::from_vec(vec![sqrt_2_over_pi], (1,), x.device())?;
    let inner = (&x_f32 + x3)?.broadcast_mul(&sqrt_t)?;
    let tanh = inner.tanh()?;
    let one = Tensor::from_vec(vec![1.0f32], (1,), x.device())?;
    let half = Tensor::from_vec(vec![0.5f32], (1,), x.device())?;
    let out = x_f32.broadcast_mul(&one.broadcast_add(&tanh)?)?;
    let out = out.broadcast_mul(&half)?;
    out.to_dtype(dtype)
        .map_err(|e| Error::InferenceError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::{
        apply_interleaved_rotary_emb, argmax, attention_scale_tensor,
        checked_realtime_frontend_host_peak_bytes, checked_realtime_stream_peak_bytes,
        checked_realtime_text_host_bytes, checked_workspace_bytes, conv_output_length,
        create_valid_attention_mask, drop_last_mel_frame_for_voxtral, load_voxtral_runtime_config,
        mel_frames_to_tensor, normalize_log_mel_with_max, offline_left_pad_tokens_for_generation,
        offline_streaming_padding_samples, pool_audio_embeddings_by_block,
        prepare_realtime_conv_input, realtime_preparation_geometry_for, resample_audio,
        select_voxtral_dtype, text_delta, validate_realtime_sample_ceiling,
        voxtral_generation_prefix_len, voxtral_realtime_cuda_sliding_flash_options,
        voxtral_realtime_offline_frame_limit, Conv1dGeometry, VoxtralRealtimePreparationMode,
    };
    use crate::backends::{BackendKind, DeviceCapabilities, DeviceKind, DeviceProfile};
    use candle_core::{DType, Device, Tensor};
    use uuid::Uuid;

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
    fn portable_offline_frame_limit_is_unchanged() {
        for backend in [BackendKind::Cpu, BackendKind::Metal] {
            assert_eq!(
                voxtral_realtime_offline_frame_limit(backend, 8192, 8192).unwrap(),
                1024
            );
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
    fn voxtral_offline_padding_covers_generation_prefix() {
        assert_eq!(offline_left_pad_tokens_for_generation(32, 6), 76);

        let (left, _right) = offline_streaming_padding_samples(
            16_001,
            1_280,
            offline_left_pad_tokens_for_generation(32, 6),
            17,
        );

        assert_eq!(left, 76 * 1_280);
    }

    #[test]
    fn voxtral_generation_uses_reference_prefix_length() {
        assert_eq!(voxtral_generation_prefix_len(39), 38);
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
        let conv = Conv1dGeometry {
            kernel: 3,
            stride: 2,
            padding: 1,
            dilation: 1,
        };
        assert_eq!(conv_output_length(9, conv).unwrap(), 5);
        assert_eq!(conv_output_length(5, conv).unwrap(), 3);
        assert_eq!(conv_output_length(8, conv).unwrap(), 4);
        assert_eq!(conv_output_length(4, conv).unwrap(), 2);
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
        let conv = Conv1dGeometry {
            kernel: 3,
            stride: 2,
            padding: 1,
            dilation: 1,
        };
        let push = realtime_preparation_geometry_for(
            3_201,
            16_000,
            16_000,
            160,
            80,
            1_280,
            1,
            3,
            4,
            16,
            conv,
            conv,
            VoxtralRealtimePreparationMode::Push,
        )
        .unwrap();
        let finish = realtime_preparation_geometry_for(
            3_201,
            16_000,
            16_000,
            160,
            80,
            1_280,
            1,
            3,
            4,
            16,
            conv,
            conv,
            VoxtralRealtimePreparationMode::Finish,
        )
        .unwrap();
        assert!(finish.padded_samples > push.padded_samples);
        assert!(finish.pooled_frames >= push.pooled_frames);
        assert_eq!(push.stable_frames, push.pooled_frames.saturating_sub(3));
        assert_eq!(finish.stable_frames, finish.pooled_frames);
        assert_eq!(
            finish,
            realtime_preparation_geometry_for(
                3_201,
                16_000,
                16_000,
                160,
                80,
                1_280,
                1,
                3,
                4,
                16,
                conv,
                conv,
                VoxtralRealtimePreparationMode::Finish,
            )
            .unwrap()
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
}
