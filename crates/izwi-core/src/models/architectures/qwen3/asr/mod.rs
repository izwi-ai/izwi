//! Native Qwen speech-family loader and inference shared with Qwen3-ForcedAligner.

mod audio;
mod config;
mod tokenizer;
mod word_units;

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use std::{fs, io::Read};

use candle_core::quantized::gguf_file::Value as GgufValue;
use candle_core::{DType, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use serde::Deserialize;
use serde_json::json;
use tracing::{debug, info, warn};

use crate::audio::{MelConfig, MelNorm, MelScale, MelSpectrogram};
use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, StateComponentValue, TensorStateArena,
};
use crate::backends::{
    backend_kind_for_device, DTypeSelection, DTypeSelectionRequest, DeviceKind, DeviceProfile,
    MetalDTypeOverride,
};
use crate::catalog::ModelFamily;
use crate::engine::{StageDescriptor, StageWorkSelector, WorkCost};
use crate::error::{Error, Result};
use crate::kernels::buffer_pool::maybe_init_global_buffer_pool;
use crate::kernels::numeric_check::{
    load_with_precision_fallback, run_numeric_self_check, xorshift_tensor, NumericSelfCheckReport,
    PrecisionLoad, F16_SAFE_MAX_ABS,
};
use crate::kv::v2::{
    stage_graph_fingerprint, BoundedShape, CapabilityStateDescriptorV2, CheckpointPolicy,
    InvocationLeaseScope, InvocationStageWorkspace, InvocationStateCapacity,
    InvocationWorkspaceDomain, InvocationWorkspaceProfile, InvocationWorkspaceSet, PlacementPolicy,
    PrefixPolicy, RetainedStateCapability, ShapeAxis, ShapeDimension, ShapeExtent, StateClock,
    StateComponentId, StateDType, StateDomainHeader, StateDomainId, StateDomainSpec, StateScope,
    TensorComponentSpec, TensorRole, TensorStateDomainSpec, WorkspaceFormula,
    CURRENT_INFERENCE_STATE_ABI,
};
use crate::kv::{InferenceStateCapability, InferenceStateContractProvider};
use crate::model::ModelVariant;
use crate::models::architectures::qwen3::core::{
    qwen3_runtime_profile_delta, qwen3_runtime_profile_snapshot, qwen3_runtime_profiling_enabled,
    Qwen3Config, Qwen3Model, Qwen3RuntimeProfileSnapshot, RopeScalingConfig,
};
use crate::models::shared::attention::flash::{
    flash_attention_compiled, flash_attention_requested,
};
use crate::models::shared::attention::paged::default_kv_page_size;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::memory::accounting::{
    deep_copy_tensor_storage, TensorStorageAccounting,
};
use crate::models::shared::memory::metal::metal_pool_for_device;
use crate::models::shared::state::exact_stage_scratch_domain;
use crate::models::shared::weights::gguf::{var_builder_from_gguf_filtered, GgufLoader};
use crate::models::shared::weights::staged;
use crate::runtime::asr_automatic_context as qwen3_asr_automatic_context;

use audio::AudioTower;
pub(crate) use audio::{
    qwen3_audio_cnn_chunk_bytes, qwen3_audio_cnn_slice_chunks, QWEN3_ASR_AUDIO_POSITIONS,
};
use config::Qwen3AsrConfig;
use tokenizer::{AsrTokenizer, SpecialTokenIds};
pub(crate) use word_units::{alignment_units, AlignmentUnit};

const QWEN3_ASR_PREPARED_INPUT_DOMAIN: StateDomainId = StateDomainId::new(2);
static NEXT_QWEN3_ASR_AUDIO_PREPARATION_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Deserialize)]
struct PreprocessorConfig {
    #[serde(default)]
    feature_size: usize,
    #[serde(default)]
    n_fft: usize,
    #[serde(default)]
    hop_length: usize,
    #[serde(default)]
    n_samples: usize,
    #[serde(default)]
    nb_max_frames: usize,
}

pub struct Qwen3AsrModel {
    audio_preparation_id: u64,
    device: DeviceProfile,
    audio_dtype: DType,
    text_dtype: DType,
    checkpoint_format: &'static str,
    gguf_qmatmul_text_enabled: bool,
    is_forced_aligner: bool,
    timestamp_token_id: Option<u32>,
    timestamp_segment_time_ms: Option<u32>,
    tokenizer: AsrTokenizer,
    specials: SpecialTokenIds,
    audio_tower: AudioTower,
    text_model: Qwen3Model,
    text_context_tokens: Option<usize>,
    mel: MelSpectrogram,
    preprocessor: PreprocessorConfig,
    /// The audio tower's and the text model's loaded dtype, why, and their
    /// load-time probes.
    audio_precision: PrecisionLoad,
    text_precision: PrecisionLoad,
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen3AsrPhysicalStateSpec {
    pub(crate) retained: crate::kv::v2::InferenceStateContract,
    pub(crate) retained_max_tokens: usize,
    /// Context one sequence or invocation may occupy. It bounds the prepared
    /// inputs and the invocation KV, so load publishes no larger context.
    pub(crate) sequence_tokens: usize,
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) invocation: crate::kv::v2::InferenceStateContract,
}

/// How load bounds Qwen3-ASR decoder state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qwen3AsrContextBound {
    /// The operator configured the context: state keeps the model's native
    /// bound and the shared fitter applies the configured value.
    Configured,
    /// The automatic ASR ceiling, with retained KV for this many concurrent
    /// sequences.
    Automatic { retained_rows: usize },
}

/// Chunk planners cut audio at 95% of the solved invocation limit
/// (`NativeExecutor::asr_chunk_plan`), so a full chunk needs a limit of
/// `chunk / 0.95` seconds.
const QWEN3_ASR_PLANNED_CHUNK_FRACTION: f32 = 0.95;

/// Tokens one Qwen3-ASR invocation is planned with (QASR-1). No request
/// decodes more than one chunk at a time: input longer than the long-form
/// chunk is split, and the planners fit each chunk with a text-prompt
/// reserve, its audio tokens and a decode reserve. This is the smallest
/// context at which every planner still admits a full chunk.
fn qwen3_asr_sequence_ceiling(audio_token_rate: f32, max_chunk_secs: f32) -> usize {
    // Preparation plans single-pass input with the default generation budget;
    // chunk execution decodes at most the transcription cap.
    let decode_reserve = crate::models::registry::NativeAsrGenerationOptions::default()
        .max_new_tokens
        .max(DEFAULT_TRANSCRIBE_MAX_NEW_TOKENS);
    crate::runtime::asr_sequence_ceiling(
        crate::runtime::ASR_TEXT_PROMPT_TOKEN_RESERVE,
        audio_token_rate,
        max_chunk_secs / QWEN3_ASR_PLANNED_CHUNK_FRACTION,
        decode_reserve,
    )
}

fn qwen3_asr_state_dtype(dtype: DType) -> Result<StateDType> {
    match dtype {
        DType::F32 => Ok(StateDType::F32),
        DType::F16 => Ok(StateDType::F16),
        DType::BF16 => Ok(StateDType::Bf16),
        other => Err(Error::ModelLoadError(format!(
            "Qwen3 ASR retained prepared inputs do not support {other:?}"
        ))),
    }
}

fn qwen3_asr_retained_state_contract(
    mut contract: crate::kv::v2::InferenceStateContract,
    max_sequence: usize,
    hidden: usize,
    dtype: DType,
) -> Result<crate::kv::v2::InferenceStateContract> {
    let max_sequence = u64::try_from(max_sequence)
        .map_err(|_| Error::ModelLoadError("Qwen3 ASR context exceeds u64".to_string()))?;
    let hidden = u64::try_from(hidden)
        .map_err(|_| Error::ModelLoadError("Qwen3 ASR hidden size exceeds u64".to_string()))?;
    for domain in &mut contract.domains {
        if let StateDomainSpec::PagedAttention(domain) = domain {
            domain.header.prefix = PrefixPolicy::Disabled;
            domain.header.checkpoint = CheckpointPolicy::Transactional;
        }
    }
    for group in &mut contract.groups {
        group.domains.push(QWEN3_ASR_PREPARED_INPUT_DOMAIN);
        group.prefix_shareable = false;
    }
    contract
        .domains
        .push(StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: StateDomainHeader {
                id: QWEN3_ASR_PREPARED_INPUT_DOMAIN,
                scope: StateScope::Retained,
                clock: StateClock::DecoderTokens,
                placement: PlacementPolicy::BackendLocal,
                prefix: PrefixPolicy::Disabled,
                checkpoint: CheckpointPolicy::Transactional,
            },
            components: vec![
                TensorComponentSpec {
                    id: StateComponentId::new(1),
                    role: TensorRole::RetainedEmbedding,
                    shape: BoundedShape {
                        dimensions: vec![
                            ShapeDimension {
                                axis: ShapeAxis::Batch,
                                extent: ShapeExtent::Fixed { value: 1 },
                            },
                            ShapeDimension {
                                axis: ShapeAxis::Sequence,
                                extent: ShapeExtent::RuntimeBounded {
                                    min: 1,
                                    max: max_sequence,
                                },
                            },
                            ShapeDimension {
                                axis: ShapeAxis::Hidden,
                                extent: ShapeExtent::Fixed { value: hidden },
                            },
                        ],
                    },
                    accepted_dtypes: vec![qwen3_asr_state_dtype(dtype)?],
                },
                TensorComponentSpec {
                    id: StateComponentId::new(2),
                    role: TensorRole::Control,
                    shape: BoundedShape {
                        dimensions: vec![
                            ShapeDimension {
                                axis: ShapeAxis::Channels,
                                extent: ShapeExtent::Fixed { value: 3 },
                            },
                            ShapeDimension {
                                axis: ShapeAxis::Sequence,
                                extent: ShapeExtent::RuntimeBounded {
                                    min: 1,
                                    max: max_sequence,
                                },
                            },
                        ],
                    },
                    accepted_dtypes: vec![StateDType::I64],
                },
            ],
        }));
    contract.validate()?;
    Ok(contract)
}

impl Qwen3AsrModel {
    pub(crate) fn audio_preparation_row_geometry(
        &self,
        input_samples: usize,
        input_sample_rate: u32,
    ) -> Result<Qwen3AsrAudioPreparationGeometry> {
        if self.is_forced_aligner {
            return Err(Error::InvalidInput(
                "Qwen3-ForcedAligner has no transcription audio preparation stage".into(),
            ));
        }
        let mel = self.mel.config();
        let frontend = qwen3_asr_frontend_geometry(
            input_samples,
            input_sample_rate,
            QWEN3_ASR_SAMPLE_RATE,
            mel.n_fft,
            mel.hop_length,
            self.device.device.is_cuda(),
            self.preprocessor.nb_max_frames,
        )?;
        if frontend.mel_frames == 0 {
            return Err(Error::InvalidInput("Empty audio input".into()));
        }
        let audio_tokens = self.audio_tower.output_lengths(&[frontend.mel_frames])?[0];
        if audio_tokens == 0 {
            return Err(Error::InvalidInput(
                "Qwen3 ASR audio tower geometry produced no tokens".into(),
            ));
        }
        let (_, _, _, _, _, output_dim) = self.audio_tower.preparation_dimensions();
        let mel_elements = checked_product_u64(
            &[frontend.mel_frames, mel.n_mels],
            "Qwen3 ASR row mel geometry",
        )?;
        let output_elements =
            checked_product_u64(&[audio_tokens, output_dim], "Qwen3 ASR row output geometry")?;
        let useful_tensor_elements = mel_elements
            .checked_add(output_elements)
            .ok_or_else(|| Error::Overloaded("Qwen3 ASR row tensor geometry overflow".into()))?;
        let retained_artifact_bytes =
            output_elements
                .checked_mul(u64::try_from(self.text_dtype.size_in_bytes()).map_err(|_| {
                    Error::Overloaded("Qwen3 ASR text dtype width exceeds u64".into())
                })?)
                .ok_or_else(|| Error::Overloaded("Qwen3 ASR retained audio overflow".into()))?;
        Ok(Qwen3AsrAudioPreparationGeometry {
            input_samples,
            input_sample_rate,
            resampled_samples: frontend.resampled_samples,
            mel_frames: frontend.mel_frames,
            audio_tokens,
            useful_tensor_elements,
            retained_artifact_bytes,
        })
    }

    pub(crate) fn audio_preparation_batch_geometry(
        &self,
        rows: &[Qwen3AsrAudioPreparationGeometry],
    ) -> Result<Qwen3AsrAudioPreparationBatchGeometry> {
        if rows.is_empty() {
            return Err(Error::InvalidInput(
                "Qwen3 ASR audio preparation batch is empty".into(),
            ));
        }
        let (n_mels, downsample_hidden, d_model, ffn_dim, attention_heads, output_dim) =
            self.audio_tower.preparation_dimensions();
        let (chunk_input, attention_window) = self.audio_tower.preparation_chunk_geometry()?;
        let max_mel_frames = rows.iter().map(|row| row.mel_frames).max().unwrap_or(0);
        let total_mel_frames = rows.iter().try_fold(0usize, |total, row| {
            total
                .checked_add(row.mel_frames)
                .ok_or_else(|| Error::Overloaded("Qwen3 ASR total mel frames overflow".into()))
        })?;
        let max_audio_tokens = rows.iter().map(|row| row.audio_tokens).max().unwrap_or(0);
        let total_audio_tokens = rows.iter().try_fold(0usize, |total, row| {
            total
                .checked_add(row.audio_tokens)
                .ok_or_else(|| Error::Overloaded("Qwen3 ASR total audio tokens overflow".into()))
        })?;
        let total_useful_tensor_elements = rows.iter().try_fold(0u64, |total, row| {
            total
                .checked_add(row.useful_tensor_elements)
                .ok_or_else(|| {
                    Error::Overloaded("Qwen3 ASR useful tensor elements overflow".into())
                })
        })?;
        let padded_mel_elements_per_row =
            checked_product_u64(&[max_mel_frames, n_mels], "Qwen3 ASR padded mel row")?;
        let padded_mel_elements = padded_mel_elements_per_row
            .checked_mul(
                u64::try_from(rows.len())
                    .map_err(|_| Error::Overloaded("Qwen3 ASR batch width exceeds u64".into()))?,
            )
            .ok_or_else(|| Error::Overloaded("Qwen3 ASR padded mel batch overflow".into()))?;
        let padded_output_elements_per_row = checked_product_u64(
            &[max_audio_tokens, output_dim],
            "Qwen3 ASR padded output row",
        )?;
        let padded_output_elements = padded_output_elements_per_row
            .checked_mul(
                u64::try_from(rows.len())
                    .map_err(|_| Error::Overloaded("Qwen3 ASR batch width exceeds u64".into()))?,
            )
            .ok_or_else(|| Error::Overloaded("Qwen3 ASR padded output batch overflow".into()))?;

        let chunks = rows.iter().try_fold(0usize, |total, row| {
            total
                .checked_add(row.mel_frames.div_ceil(chunk_input))
                .ok_or_else(|| Error::Overloaded("Qwen3 ASR chunk count overflow".into()))
        })?;
        let mut conv_elements = 0u64;
        let mut freq = n_mels;
        let mut time = chunk_input;
        for _ in 0..3 {
            freq = freq.div_ceil(2);
            time = time.div_ceil(2);
            conv_elements = conv_elements
                .checked_add(checked_product_u64(
                    &[chunks, downsample_hidden, freq, time],
                    "Qwen3 ASR convolution geometry",
                )?)
                .ok_or_else(|| {
                    Error::Overloaded("Qwen3 ASR convolution workspace overflow".into())
                })?;
        }
        let packed_elements = checked_product_u64(
            &[total_audio_tokens, d_model],
            "Qwen3 ASR packed encoder geometry",
        )?;
        let qkv_elements = packed_elements
            .checked_mul(3)
            .ok_or_else(|| Error::Overloaded("Qwen3 ASR QKV workspace overflow".into()))?;
        let attention_elements = checked_product_u64(
            &[total_audio_tokens, attention_window, attention_heads],
            "Qwen3 ASR attention geometry",
        )?;
        let ffn_elements =
            checked_product_u64(&[total_audio_tokens, ffn_dim, 2], "Qwen3 ASR FFN geometry")?;
        let audio_dtype_bytes = u64::try_from(self.audio_dtype.size_in_bytes())
            .map_err(|_| Error::Overloaded("Qwen3 ASR audio dtype width exceeds u64".into()))?;
        let text_dtype_bytes = u64::try_from(self.text_dtype.size_in_bytes())
            .map_err(|_| Error::Overloaded("Qwen3 ASR text dtype width exceeds u64".into()))?;
        let host_elements = rows.iter().try_fold(padded_mel_elements, |total, row| {
            let row_elements =
                checked_product_u64(&[row.resampled_samples], "Qwen3 ASR resample geometry")?
                    .checked_add(checked_product_u64(
                        &[row.mel_frames, n_mels],
                        "Qwen3 ASR host mel geometry",
                    )?)
                    .ok_or_else(|| Error::Overloaded("Qwen3 ASR host frontend overflow".into()))?;
            total
                .checked_add(row_elements)
                .ok_or_else(|| Error::Overloaded("Qwen3 ASR host workspace overflow".into()))
        })?;
        let device_audio_elements = padded_mel_elements
            .checked_add(conv_elements)
            .and_then(|value| value.checked_add(packed_elements))
            .and_then(|value| value.checked_add(qkv_elements))
            .and_then(|value| value.checked_add(attention_elements))
            .and_then(|value| value.checked_add(ffn_elements))
            .ok_or_else(|| Error::Overloaded("Qwen3 ASR device workspace overflow".into()))?;
        let host_bytes = host_elements
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .ok_or_else(|| Error::Overloaded("Qwen3 ASR host workspace bytes overflow".into()))?;
        let device_bytes = device_audio_elements
            .checked_mul(audio_dtype_bytes)
            .and_then(|value| {
                padded_output_elements
                    .checked_mul(text_dtype_bytes)
                    .and_then(|output| value.checked_add(output))
            })
            .ok_or_else(|| Error::Overloaded("Qwen3 ASR device workspace bytes overflow".into()))?;
        let workspace_bytes = host_bytes
            .checked_add(device_bytes)
            .ok_or_else(|| Error::Overloaded("Qwen3 ASR aggregate workspace overflow".into()))?;
        Ok(Qwen3AsrAudioPreparationBatchGeometry {
            rows: rows.len(),
            max_mel_frames,
            total_mel_frames,
            padded_mel_elements,
            padded_mel_elements_per_row,
            max_audio_tokens,
            total_audio_tokens,
            total_useful_tensor_elements,
            padded_output_elements,
            padded_output_elements_per_row,
            workspace_bytes,
        })
    }

    pub(crate) fn audio_preparation_row_cost(
        &self,
        row: &Qwen3AsrAudioPreparationGeometry,
    ) -> WorkCost {
        WorkCost::new(row.mel_frames as u64, row.useful_tensor_elements, 0)
    }

    pub(crate) fn audio_preparation_retained_tensor_bytes(
        &self,
        row: &Qwen3AsrAudioPreparationGeometry,
    ) -> u64 {
        row.retained_artifact_bytes
    }

    pub(crate) fn audio_preparation_row_cost_for_batch(
        &self,
        row_index: usize,
        rows: &[Qwen3AsrAudioPreparationGeometry],
        batch: &Qwen3AsrAudioPreparationBatchGeometry,
    ) -> Result<WorkCost> {
        let row = rows.get(row_index).ok_or_else(|| {
            Error::InvalidInput("Qwen3 ASR preparation row index is out of range".into())
        })?;
        if rows.len() != batch.rows {
            return Err(Error::InvalidInput(
                "Qwen3 ASR preparation cost rows disagree with batch geometry".into(),
            ));
        }
        // Allocate the aggregate workspace deterministically: equal integer
        // shares plus one byte for each leading remainder row. Summing the row
        // costs therefore reconstructs the exact physical envelope.
        let width = u64::try_from(batch.rows)
            .map_err(|_| Error::Overloaded("Qwen3 ASR batch width exceeds u64".into()))?;
        let workspace = batch.workspace_bytes / width
            + u64::from((row_index as u64) < batch.workspace_bytes % width);
        Ok(WorkCost::new(
            row.mel_frames as u64,
            row.useful_tensor_elements,
            workspace,
        ))
    }

    pub(crate) fn audio_preparation_batch_cost(
        &self,
        geometry: &Qwen3AsrAudioPreparationBatchGeometry,
    ) -> Result<WorkCost> {
        Ok(WorkCost::new(
            geometry.total_mel_frames as u64,
            geometry.total_useful_tensor_elements,
            geometry.workspace_bytes,
        ))
    }

    pub(crate) fn audio_preparation_max_batch_workspace_bytes(
        &self,
        max_batch_size: usize,
    ) -> Result<u64> {
        if max_batch_size == 0 {
            return Err(Error::InvalidInput(
                "Qwen3 ASR preparation batch width must be non-zero".into(),
            ));
        }
        let mel = self.mel.config();
        let max_frames = qwen3_asr_effective_max_frames(
            self.device.device.is_cuda(),
            false,
            self.preprocessor.nb_max_frames,
            QWEN3_ASR_SAMPLE_RATE as usize,
            mel.hop_length,
        );
        if max_frames == 0 {
            return Err(Error::ModelLoadError(
                "Qwen3 ASR preprocessor has no bounded maximum frame count".into(),
            ));
        }
        let (_, _, _, _, _, output_dim) = self.audio_tower.preparation_dimensions();
        let audio_tokens = self.audio_tower.output_lengths(&[max_frames])?[0];
        let retained_artifact_bytes = checked_product_u64(
            &[audio_tokens, output_dim, self.text_dtype.size_in_bytes()],
            "Qwen3 ASR maximum retained audio",
        )?;
        let max_samples = max_frames
            .checked_mul(mel.hop_length)
            .and_then(|samples| samples.checked_add(mel.n_fft))
            .ok_or_else(|| Error::Overloaded("Qwen3 ASR maximum samples overflow".into()))?;
        let row = Qwen3AsrAudioPreparationGeometry {
            input_samples: max_samples,
            input_sample_rate: QWEN3_ASR_SAMPLE_RATE,
            resampled_samples: max_samples,
            mel_frames: max_frames,
            audio_tokens,
            useful_tensor_elements: checked_product_u64(
                &[max_frames, mel.n_mels],
                "Qwen3 ASR maximum mel geometry",
            )?
            .checked_add(checked_product_u64(
                &[audio_tokens, output_dim],
                "Qwen3 ASR maximum output geometry",
            )?)
            .ok_or_else(|| Error::Overloaded("Qwen3 ASR maximum row geometry overflow".into()))?,
            retained_artifact_bytes,
        };
        self.audio_preparation_batch_geometry(&vec![row; max_batch_size])
            .map(|geometry| geometry.workspace_bytes)
    }

    pub(crate) fn audio_preparation_stage_seal(
        &self,
        backend: crate::backends::BackendKind,
        max_batch_size: usize,
    ) -> Result<Qwen3AsrAudioPreparationStageSeal> {
        let loaded_backend = backend_kind_for_device(&self.device.device);
        if loaded_backend != backend {
            return Err(Error::ModelLoadError(format!(
                "Qwen3 ASR audio preparation backend mismatch: model={loaded_backend:?}, adapter={backend:?}"
            )));
        }
        Ok(Qwen3AsrAudioPreparationStageSeal {
            backend,
            audio_dtype: format!("{:?}", self.audio_dtype).to_ascii_lowercase(),
            text_dtype: format!("{:?}", self.text_dtype).to_ascii_lowercase(),
            max_batch_size,
            max_workspace_bytes: self
                .audio_preparation_max_batch_workspace_bytes(max_batch_size)?,
        })
    }

    /// Target semantic contract for the shared Qwen3 decoder.
    ///
    /// This is kept separate from the advertised loaded-model capability until
    /// the ASR engine route supplies its transaction-owned reservation.
    pub(crate) fn managed_inference_state_contract(
        &self,
    ) -> Result<crate::kv::v2::InferenceStateContract> {
        let max_sequence = self.text_context_tokens.ok_or_else(|| {
            Error::ModelLoadError(
                "Qwen3 ASR text config is missing max_position_embeddings; prepared decoder inputs cannot be bounded"
                    .to_string(),
            )
        })?;
        self.managed_inference_state_contract_for_sequence(max_sequence)
    }

    fn managed_inference_state_contract_for_sequence(
        &self,
        max_sequence: usize,
    ) -> Result<crate::kv::v2::InferenceStateContract> {
        let contract = self.text_model.managed_inference_state_contract(
            crate::kv::v2::StateDomainId::new(1),
            self.text_dtype,
            default_kv_page_size(),
            // Managed prefix reuse is chat-task-gated; ASR never shares.
            false,
        )?;
        qwen3_asr_retained_state_contract(
            contract,
            max_sequence,
            self.text_model.hidden_size(),
            self.text_dtype,
        )
    }

    /// Automatic per-sequence context: one planned long-form chunk, which
    /// also bounds the single-pass route because longer input is chunked.
    fn automatic_asr_sequence_tokens(&self) -> usize {
        let long_form_secs = izwi_asr_toolkit::AsrLongFormConfig::default().hard_max_chunk_secs;
        let max_chunk_secs = self
            .max_audio_seconds_hint()
            .map_or(long_form_secs, |hint| hint.min(long_form_secs));
        qwen3_asr_sequence_ceiling(self.audio_token_rate(), max_chunk_secs)
    }

    /// Author the retained streaming state and invocation-only offline state
    /// for the exact ASR graphs frozen by the loaded adapter.
    ///
    /// Qwen3-ASR's autoregressive decoder gets one isolated paged cache per
    /// physical row of an atomic graph. Native streaming continues to use the
    /// scheduler-owned retained cache. Qwen3-ForcedAligner uses its cacheless
    /// NAR head and therefore seals an explicitly stateless descriptor.
    ///
    /// The model's native context (65,536 tokens) would size both KV arenas
    /// and every prepared-input row for requests that need about 1K tokens,
    /// so the automatic bound is one planned chunk per sequence (QASR-1,
    /// ADM-6). A configured context keeps the native bound.
    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
        bound: Qwen3AsrContextBound,
    ) -> Result<Qwen3AsrPhysicalStateSpec> {
        if self.is_forced_aligner {
            return Err(Error::ModelLoadError(
                "Qwen3-ForcedAligner does not own autoregressive ASR state".to_string(),
            ));
        }
        let native_tokens = self.text_context_tokens.ok_or_else(|| {
            Error::ModelLoadError(
                "Qwen3 ASR text config is missing max_position_embeddings; an exact invocation KV capacity cannot be sealed"
                    .to_string(),
            )
        })?;
        let (sequence_tokens, retained_max_tokens) = match bound {
            Qwen3AsrContextBound::Configured => (native_tokens, native_tokens),
            Qwen3AsrContextBound::Automatic { retained_rows } => qwen3_asr_automatic_context(
                native_tokens,
                self.automatic_asr_sequence_tokens(),
                retained_rows,
            ),
        };
        qwen3_asr_physical_state_spec(
            &self.managed_inference_state_contract_for_sequence(sequence_tokens)?,
            sequence_tokens,
            retained_max_tokens,
            stage_graphs,
        )
    }
}

fn qwen3_asr_physical_state_spec(
    retained: &crate::kv::v2::InferenceStateContract,
    max_tokens: usize,
    retained_max_tokens: usize,
    stage_graphs: &[&[StageDescriptor]],
) -> Result<Qwen3AsrPhysicalStateSpec> {
    if stage_graphs.is_empty() {
        return Err(Error::ModelLoadError(
            "Qwen3 ASR physical state has no execution graph".to_string(),
        ));
    }
    let max_tokens = u64::try_from(max_tokens)
        .map_err(|_| Error::ModelLoadError("Qwen3 ASR context exceeds u64".to_string()))?;
    if max_tokens == 0 {
        return Err(Error::ModelLoadError(
            "Qwen3 ASR physical state requires a non-zero text context".to_string(),
        ));
    }

    let mut invocation_contract = retained.clone();
    invocation_contract
        .domains
        .retain(|domain| domain.id() != QWEN3_ASR_PREPARED_INPUT_DOMAIN);
    for group in &mut invocation_contract.groups {
        group
            .domains
            .retain(|domain| *domain != QWEN3_ASR_PREPARED_INPUT_DOMAIN);
    }
    let StateDomainSpec::PagedAttention(invocation_domain) = &mut invocation_contract.domains[0]
    else {
        return Err(Error::ModelLoadError(
            "Qwen3 ASR KV contract is not paged attention".to_string(),
        ));
    };
    invocation_domain.header.scope = StateScope::Invocation;
    invocation_domain.header.prefix = PrefixPolicy::Disabled;
    invocation_domain.header.checkpoint = CheckpointPolicy::None;
    invocation_contract.groups[0].prefix_shareable = false;
    invocation_contract.validate()?;
    let invocation_domain = invocation_contract.domains[0].clone();
    let invocation_group = invocation_contract.groups[0].clone();
    let max_invocation_domain_id = invocation_domain.id().get();
    let physical_bytes = qwen3_asr_invocation_workspace_bytes(&invocation_domain, max_tokens)?;
    let decoder_capacity = InvocationStateCapacity::decoder_context(max_tokens)?;

    let mut profiles = Vec::with_capacity(stage_graphs.len());
    let mut has_invocation_workspace = false;
    for stages in stage_graphs {
        let mut invocation_stages = stages
            .iter()
            .enumerate()
            .map(|(index, stage)| {
                let owns_invocation_state = stage.selector == StageWorkSelector::Atomic;
                let lease_scope = if owns_invocation_state || stage.max_batch_size == 1 {
                    InvocationLeaseScope::PerRow
                } else {
                    InvocationLeaseScope::PerStageBatch
                };
                let mut domains = owns_invocation_state
                    .then(|| InvocationWorkspaceDomain::State {
                        state: invocation_domain.clone(),
                        capacity: decoder_capacity,
                        placement: invocation_domain.header().placement,
                        formula: WorkspaceFormula {
                            fixed_bytes: physical_bytes,
                            dimensions: vec![],
                            terms: vec![],
                        },
                    })
                    .into_iter()
                    .collect::<Vec<_>>();
                let scratch_id = max_invocation_domain_id
                    .checked_add(u32::try_from(index + 1).map_err(|_| {
                        Error::ModelLoadError("Qwen3 ASR stage count exceeds u32".into())
                    })?)
                    .ok_or_else(|| {
                        Error::ModelLoadError("Qwen3 ASR scratch domain id overflow".into())
                    })?;
                if let Some(scratch) =
                    exact_stage_scratch_domain(stage, StateDomainId::new(scratch_id), lease_scope)?
                {
                    domains.push(scratch);
                }
                has_invocation_workspace |= !domains.is_empty();
                Ok(InvocationStageWorkspace {
                    stage: stage.id,
                    lease_scope,
                    groups: owns_invocation_state
                        .then(|| invocation_group.clone())
                        .into_iter()
                        .collect(),
                    domains,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        invocation_stages.sort_unstable_by_key(|stage| stage.stage);
        profiles.push(InvocationWorkspaceProfile {
            stage_graph_fingerprint: stage_graph_fingerprint(stages)?,
            stages: invocation_stages,
        });
    }
    profiles.sort_unstable_by_key(|profile| profile.stage_graph_fingerprint);
    profiles.dedup();
    let invocation = if has_invocation_workspace {
        InvocationWorkspaceSet::Bounded { profiles }
    } else {
        InvocationWorkspaceSet::None {
            stage_graph_fingerprints: profiles
                .into_iter()
                .map(|profile| profile.stage_graph_fingerprint)
                .collect(),
        }
    };
    let descriptor = CapabilityStateDescriptorV2 {
        abi: CURRENT_INFERENCE_STATE_ABI,
        retained: RetainedStateCapability::Managed {
            contract: retained.clone(),
        },
        invocation,
    };
    for stages in stage_graphs {
        descriptor.validate_against_stages(stages)?;
    }
    Ok(Qwen3AsrPhysicalStateSpec {
        retained: retained.clone(),
        retained_max_tokens,
        sequence_tokens: usize::try_from(max_tokens).map_err(|_| {
            Error::ModelLoadError("Qwen3 ASR sequence context exceeds usize".to_string())
        })?,
        descriptor,
        invocation: invocation_contract,
    })
}

fn qwen3_asr_invocation_workspace_bytes(domain: &StateDomainSpec, max_tokens: u64) -> Result<u64> {
    let StateDomainSpec::PagedAttention(domain) = domain else {
        return Err(Error::ModelLoadError(
            "Qwen3 ASR invocation state is not paged attention".to_string(),
        ));
    };
    let page_tokens = u64::from(domain.page_size.preferred_tokens);
    let rounded_tokens = max_tokens
        .checked_add(page_tokens.saturating_sub(1))
        .and_then(|tokens| tokens.checked_div(page_tokens))
        .and_then(|pages| pages.checked_mul(page_tokens))
        .ok_or_else(|| {
            Error::ModelLoadError("Qwen3 ASR invocation token capacity overflow".to_string())
        })?;
    let elements_per_token = domain.layers.iter().try_fold(0_u64, |total, layer| {
        let per_layer = u64::from(layer.kv_heads)
            .checked_mul(u64::from(layer.key_head_dim) + u64::from(layer.value_head_dim))
            .ok_or_else(|| Error::ModelLoadError("Qwen3 ASR KV geometry overflow".to_string()))?;
        total
            .checked_add(per_layer)
            .ok_or_else(|| Error::ModelLoadError("Qwen3 ASR KV geometry overflow".to_string()))
    })?;
    let elements = elements_per_token
        .checked_mul(rounded_tokens)
        .ok_or_else(|| Error::ModelLoadError("Qwen3 ASR KV capacity overflow".to_string()))?;
    domain
        .accepted_dtypes
        .iter()
        .map(|dtype| match dtype {
            StateDType::F32 => elements.checked_mul(4),
            StateDType::F16 | StateDType::Bf16 => elements.checked_mul(2),
            StateDType::I64 => elements.checked_mul(8),
            StateDType::I8 => Some(elements),
            StateDType::Q4 => elements.checked_add(1).map(|value| value / 2),
        })
        .collect::<Option<Vec<_>>>()
        .and_then(|bytes| bytes.into_iter().min())
        .ok_or_else(|| {
            Error::ModelLoadError(
                "Qwen3 ASR KV dtype capacity overflow or missing dtype".to_string(),
            )
        })
}

impl InferenceStateContractProvider for Qwen3AsrModel {
    fn inference_state_contract(&self) -> Result<InferenceStateCapability> {
        // The forced aligner runs one cacheless forward per request, as its
        // `ForcedAlignment` adapter declares.
        if self.is_forced_aligner {
            return Ok(InferenceStateCapability::Stateless);
        }
        Ok(InferenceStateCapability::Managed(
            self.managed_inference_state_contract()?,
        ))
    }
}

pub struct AsrDecodeState {
    cache: Option<PhysicalPagedKvCache>,
    physical_tensor_sequence: Option<PhysicalStateSequenceId>,
    /// Immutable decoder input assembled from text and audio embeddings.
    /// Scheduler spans borrow slices from this tensor; processed spans are
    /// never concatenated or retained as decoder activations.
    prepared_prompt_embeddings: Option<Tensor>,
    /// Immutable multimodal RoPE positions paired with the prepared prompt.
    prepared_position_ids: Option<Tensor>,
    /// Exact scheduler-visible cursor through the prepared multimodal prompt.
    prefill_progress: usize,
    /// Model output awaiting sampling inside the current executor quantum.
    /// This slot is always empty before an incremental state is yielded.
    unconsumed_output: Option<Tensor>,
    pos: usize,
    /// Token emitted by the previous quantum but not yet appended to KV.
    /// Deferring the write keeps physical progress aligned with the
    /// scheduler's exact input range and transaction receipt.
    pending_token: Option<u32>,
    language: Option<String>,
    generated_ids: Vec<u32>,
    visible_generated_ids: Vec<u32>,
    decoded_visible_tokens: usize,
    tokens_since_resync: usize,
    assembled: String,
    diagnostics: Qwen3AsrDiagnostics,
    profile_start: Qwen3RuntimeProfileSnapshot,
    stop_tokens: Vec<u32>,
    max_new_tokens: usize,
    finished: bool,
}

pub(crate) struct AsrDecodeCheckpoint {
    cache: PhysicalPagedKvCache,
    physical_tensor_sequence: Option<PhysicalStateSequenceId>,
    prepared_prompt_embeddings: Option<Tensor>,
    prepared_position_ids: Option<Tensor>,
    prefill_progress: usize,
    unconsumed_output: Option<Tensor>,
    pos: usize,
    pending_token: Option<u32>,
    generated_ids: Vec<u32>,
    visible_generated_ids: Vec<u32>,
    decoded_visible_tokens: usize,
    tokens_since_resync: usize,
    assembled: String,
    diagnostics: Qwen3AsrDiagnostics,
    finished: bool,
}

impl AsrDecodeState {
    pub(crate) fn prefill_progress(&self) -> usize {
        self.prefill_progress
    }

    pub(crate) fn prefill_token_count(&self) -> usize {
        self.diagnostics.prompt_tokens
    }

    pub(crate) fn uses_managed_kv(&self) -> bool {
        self.cache.is_some()
    }

    pub(crate) fn take_managed_write_completions(
        &mut self,
    ) -> Vec<Arc<crate::backends::kv::KvWriteBatchCompletion>> {
        self.cache
            .as_mut()
            .map(PhysicalPagedKvCache::take_completed_writes)
            .unwrap_or_default()
    }

    pub(crate) fn sequence_position(&self) -> usize {
        self.pos
    }

    pub(crate) fn install_managed_reservation(
        &mut self,
        cache: PhysicalPagedKvCache,
    ) -> Result<()> {
        let checkpoint = self.begin_managed_quantum(cache)?;
        drop(checkpoint);
        Ok(())
    }

    pub(crate) fn begin_managed_quantum(
        &mut self,
        cache: PhysicalPagedKvCache,
    ) -> Result<AsrDecodeCheckpoint> {
        let Some(current) = &self.cache else {
            return Err(Error::InferenceError(
                "an invocation-scoped Qwen3 ASR decode cannot switch to retained KV".to_string(),
            ));
        };
        if current.arena().id() != cache.arena().id()
            || current.arena().config().group != cache.arena().config().group
        {
            return Err(Error::InferenceError(
                "a Qwen3 ASR session cannot switch managed KV authority".to_string(),
            ));
        }
        if cache.context_len() != self.pos {
            return Err(Error::InferenceError(format!(
                "managed Qwen3 ASR reservation starts at {}, but decode state is at {}",
                cache.context_len(),
                self.pos
            )));
        }
        let checkpoint = AsrDecodeCheckpoint {
            cache: self
                .cache
                .replace(cache)
                .expect("validated managed ASR cache"),
            physical_tensor_sequence: self.physical_tensor_sequence,
            prepared_prompt_embeddings: self.prepared_prompt_embeddings.clone(),
            prepared_position_ids: self.prepared_position_ids.clone(),
            prefill_progress: self.prefill_progress,
            unconsumed_output: self.unconsumed_output.clone(),
            pos: self.pos,
            pending_token: self.pending_token,
            generated_ids: self.generated_ids.clone(),
            visible_generated_ids: self.visible_generated_ids.clone(),
            decoded_visible_tokens: self.decoded_visible_tokens,
            tokens_since_resync: self.tokens_since_resync,
            assembled: self.assembled.clone(),
            diagnostics: self.diagnostics.clone(),
            finished: self.finished,
        };
        Ok(checkpoint)
    }

    pub(crate) fn rollback_managed_quantum(&mut self, checkpoint: AsrDecodeCheckpoint) {
        self.cache = Some(checkpoint.cache);
        self.physical_tensor_sequence = checkpoint.physical_tensor_sequence;
        self.prepared_prompt_embeddings = checkpoint.prepared_prompt_embeddings;
        self.prepared_position_ids = checkpoint.prepared_position_ids;
        self.prefill_progress = checkpoint.prefill_progress;
        self.unconsumed_output = checkpoint.unconsumed_output;
        self.pos = checkpoint.pos;
        self.pending_token = checkpoint.pending_token;
        self.generated_ids = checkpoint.generated_ids;
        self.visible_generated_ids = checkpoint.visible_generated_ids;
        self.decoded_visible_tokens = checkpoint.decoded_visible_tokens;
        self.tokens_since_resync = checkpoint.tokens_since_resync;
        self.assembled = checkpoint.assembled;
        self.diagnostics = checkpoint.diagnostics;
        self.finished = checkpoint.finished;
    }

    pub(crate) fn bind_tensor_sequence(&mut self, sequence: u64) -> Result<()> {
        let sequence = PhysicalStateSequenceId::new(sequence)?;
        if self
            .physical_tensor_sequence
            .is_some_and(|current| current != sequence)
        {
            return Err(Error::InferenceError(
                "Qwen3 ASR prepared-input tensor sequence identity changed".to_string(),
            ));
        }
        self.physical_tensor_sequence = Some(sequence);
        Ok(())
    }

    pub(crate) fn restore_prepared_tensor_state(&mut self, arena: &TensorStateArena) -> Result<()> {
        let sequence = self.physical_tensor_sequence.ok_or_else(|| {
            Error::InferenceError(
                "Qwen3 ASR retained state has no prepared-input tensor sequence".to_string(),
            )
        })?;
        let snapshot = arena
            .read(sequence, QWEN3_ASR_PREPARED_INPUT_DOMAIN)?
            .ok_or_else(|| {
                Error::InferenceError(
                    "Qwen3 ASR prepared-input tensor state has no committed snapshot".to_string(),
                )
            })?;
        if snapshot.components.len() != 2
            || snapshot.components[0].component != StateComponentId::new(1)
            || snapshot.components[1].component != StateComponentId::new(2)
            || snapshot.cursor != self.pos as u64
        {
            return Err(Error::InferenceError(
                "Qwen3 ASR prepared-input tensor snapshot is incomplete or stale".to_string(),
            ));
        }
        self.prepared_prompt_embeddings = snapshot.components[0].tensor.clone();
        self.prepared_position_ids = snapshot.components[1].tensor.clone();
        Ok(())
    }

    pub(crate) fn stage_prepared_tensor_state(
        &mut self,
        arena: &TensorStateArena,
        transaction: u64,
    ) -> Result<()> {
        let transaction = PhysicalStateTransactionId::new(transaction)?;
        let expected_cursor = arena
            .read_transaction_base(transaction, QWEN3_ASR_PREPARED_INPUT_DOMAIN)?
            .map_or(0, |snapshot| snapshot.cursor);
        let target_cursor = self
            .cache
            .as_ref()
            .ok_or_else(|| {
                Error::InferenceError(
                    "Qwen3 ASR retained prepared inputs have no managed KV cursor".to_string(),
                )
            })?
            .context_len() as u64;
        if target_cursor != self.pos as u64 {
            return Err(Error::InferenceError(
                "Qwen3 ASR prepared-input and managed-KV cursors diverged".to_string(),
            ));
        }
        arena.stage_replace(
            transaction,
            QWEN3_ASR_PREPARED_INPUT_DOMAIN,
            expected_cursor,
            target_cursor,
            vec![
                StateComponentValue {
                    component: StateComponentId::new(1),
                    tensor: self.prepared_prompt_embeddings.clone(),
                },
                StateComponentValue {
                    component: StateComponentId::new(2),
                    tensor: self.prepared_position_ids.clone(),
                },
            ],
        )?;
        self.prepared_prompt_embeddings = None;
        self.prepared_position_ids = None;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct AsrDecodeStep {
    pub delta: String,
    pub text: String,
    pub tokens_generated: usize,
    pub finished: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsrTranscriptionOutput {
    pub text: String,
    pub language: Option<String>,
    pub diagnostics: Option<serde_json::Value>,
}

/// One raw row for the independently schedulable Qwen3 ASR audio tower.
#[derive(Clone, Copy)]
pub(crate) struct Qwen3AsrAudioBatchRow<'a> {
    pub(crate) audio: &'a [f32],
    pub(crate) sample_rate: u32,
}

/// Immutable, exact-row output of the Qwen3 ASR audio tower.
///
/// It contains no decoder KV and may therefore be retained or discarded
/// independently of a decoder transaction. The engine must account
/// [`Self::resident_tensor_bytes`] while retaining this artifact.
#[derive(Clone)]
pub(crate) struct Qwen3AsrPreparedAudio {
    preparation_id: u64,
    audio_embeddings: Tensor,
    frontend: Qwen3AsrPreparedFrontend,
}

impl Qwen3AsrPreparedAudio {
    #[cfg(test)]
    pub(crate) fn for_test(tokens: usize, width: usize) -> Result<Self> {
        if tokens == 0 || width == 0 {
            return Err(Error::InvalidInput(
                "test prepared audio dimensions must be non-zero".into(),
            ));
        }
        Ok(Self {
            preparation_id: 0,
            audio_embeddings: Tensor::zeros(
                (1, tokens, width),
                DType::F32,
                &candle_core::Device::Cpu,
            )?,
            frontend: Qwen3AsrPreparedFrontend::default(),
        })
    }

    pub(crate) fn audio_tokens(&self) -> Result<usize> {
        self.audio_embeddings.dim(1).map_err(Error::from)
    }

    pub(crate) fn resident_tensor_bytes(&self) -> Result<u64> {
        let mut accounting = TensorStorageAccounting::default();
        accounting
            .add_tensor(&self.audio_embeddings)
            .ok_or_else(|| {
                Error::Overloaded("Qwen3 ASR prepared audio tensor accounting overflow".to_string())
            })?;
        Ok(accounting.bytes())
    }
}

#[derive(Debug, Clone, Default)]
struct Qwen3AsrPreparedFrontend {
    input_sample_rate: u32,
    input_samples: usize,
    resampled_sample_rate: u32,
    resampled_samples: usize,
    mel_frames: usize,
    mel_frames_before_drop: usize,
    mel_frames_before_truncate: usize,
    mel_last_frame_dropped: bool,
    mel_frames_truncated: bool,
    resample_ms: f64,
    mel_ms: f64,
    mel_flatten_upload_ms: f64,
    audio_encode_ms: f64,
    frontend_total_ms: f64,
}

#[derive(Debug, Clone, Default)]
struct Qwen3AsrTimingDiagnostics {
    resample_ms: f64,
    mel_ms: f64,
    mel_flatten_upload_ms: f64,
    audio_encode_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    total_ms: f64,
}

#[derive(Debug, Clone, Default)]
struct Qwen3AsrProfileDiagnostics {
    qwen3_profile_enabled: bool,
    qmatmul_calls: u64,
    qmatmul_ms: f64,
    qmatmul_input_casts: u64,
    qmatmul_input_cast_ms: f64,
    qmatmul_output_casts: u64,
    qmatmul_output_cast_ms: f64,
    lm_head_calls: u64,
    lm_head_ms: f64,
    silu_mul_fused_calls: u64,
    silu_mul_fallback_calls: u64,
    argmax_calls: u64,
    argmax_ms: f64,
}

impl Qwen3AsrProfileDiagnostics {
    fn update_from_runtime_delta(&mut self, delta: Qwen3RuntimeProfileSnapshot) {
        self.qmatmul_calls = delta.qmatmul_calls;
        self.qmatmul_ms = ns_to_ms(delta.qmatmul_ns);
        self.qmatmul_input_casts = delta.qmatmul_input_casts;
        self.qmatmul_input_cast_ms = ns_to_ms(delta.qmatmul_input_cast_ns);
        self.qmatmul_output_casts = delta.qmatmul_output_casts;
        self.qmatmul_output_cast_ms = ns_to_ms(delta.qmatmul_output_cast_ns);
        self.lm_head_calls = delta.lm_head_calls;
        self.lm_head_ms = ns_to_ms(delta.lm_head_ns);
        self.silu_mul_fused_calls = delta.silu_mul_fused_calls;
        self.silu_mul_fallback_calls = delta.silu_mul_fallback_calls;
    }
}

#[derive(Debug, Clone, Default)]
struct Qwen3AsrExecutionDiagnostics {
    device_kind: String,
    audio_dtype: String,
    text_dtype: String,
    checkpoint_format: String,
    flash_attention_requested: bool,
    flash_attention_compiled: bool,
    kv_page_size: usize,
    dense_decode_enabled: bool,
    dense_decode_max_tokens: usize,
    gguf_qmatmul_text_enabled: bool,
    text_projection_backend: String,
    text_projection_quantized: bool,
    qmatmul_projection_count: usize,
    dense_projection_count: usize,
    dense_bias_projection_count: usize,
    audio_precision: serde_json::Value,
    text_precision: serde_json::Value,
}

#[derive(Debug, Clone, Default)]
struct Qwen3AsrDiagnostics {
    input_sample_rate: u32,
    input_samples: usize,
    resampled_sample_rate: u32,
    resampled_samples: usize,
    mel_frames: usize,
    mel_frames_before_drop: usize,
    mel_frames_before_truncate: usize,
    mel_last_frame_dropped: bool,
    mel_frames_truncated: bool,
    audio_tokens: usize,
    prompt_tokens: usize,
    system_prompt_requested: bool,
    system_prompt_token_count: usize,
    forced_language: Option<String>,
    forced_language_token_count: usize,
    asr_text_token_available: bool,
    asr_text_token_appended: bool,
    max_new_tokens: usize,
    generated_tokens: usize,
    execution: Qwen3AsrExecutionDiagnostics,
    profile: Qwen3AsrProfileDiagnostics,
    timings: Qwen3AsrTimingDiagnostics,
}

impl Qwen3AsrDiagnostics {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "model_family": "qwen3_asr",
            "audio": {
                "input_sample_rate": self.input_sample_rate,
                "input_samples": self.input_samples,
                "resampled_sample_rate": self.resampled_sample_rate,
                "resampled_samples": self.resampled_samples,
                "mel_frames": self.mel_frames,
                "mel_frames_before_drop": self.mel_frames_before_drop,
                "mel_frames_before_truncate": self.mel_frames_before_truncate,
                "mel_last_frame_dropped": self.mel_last_frame_dropped,
                "mel_frames_truncated": self.mel_frames_truncated,
                "audio_tokens": self.audio_tokens,
            },
            "prompt": {
                "prompt_tokens": self.prompt_tokens,
                "system_prompt_requested": self.system_prompt_requested,
                "system_prompt_token_count": self.system_prompt_token_count,
                "forced_language": self.forced_language,
                "forced_language_token_count": self.forced_language_token_count,
                "asr_text_token_available": self.asr_text_token_available,
                "asr_text_token_appended": self.asr_text_token_appended,
            },
            "decode": {
                "max_new_tokens": self.max_new_tokens,
                "generated_tokens": self.generated_tokens,
            },
            "execution": {
                "device_kind": &self.execution.device_kind,
                "audio_dtype": &self.execution.audio_dtype,
                "text_dtype": &self.execution.text_dtype,
                "checkpoint_format": &self.execution.checkpoint_format,
                "flash_attention_requested": self.execution.flash_attention_requested,
                "flash_attention_compiled": self.execution.flash_attention_compiled,
                "kv_page_size": self.execution.kv_page_size,
                "dense_decode_enabled": self.execution.dense_decode_enabled,
                "dense_decode_max_tokens": self.execution.dense_decode_max_tokens,
                "gguf_qmatmul_text_enabled": self.execution.gguf_qmatmul_text_enabled,
                "text_projection_backend": &self.execution.text_projection_backend,
                "text_projection_quantized": self.execution.text_projection_quantized,
                "qmatmul_projection_count": self.execution.qmatmul_projection_count,
                "dense_projection_count": self.execution.dense_projection_count,
                "dense_bias_projection_count": self.execution.dense_bias_projection_count,
                "audio_precision": &self.execution.audio_precision,
                "text_precision": &self.execution.text_precision,
                "mrope_position_readbacks_total": super::core::qwen3_mrope_position_readbacks(),
                "sequential_rope_host_tables_total": super::core::qwen3_sequential_rope_host_tables(),
            },
            "timings_ms": {
                "resample": self.timings.resample_ms,
                "mel": self.timings.mel_ms,
                "mel_flatten_upload": self.timings.mel_flatten_upload_ms,
                "audio_encode": self.timings.audio_encode_ms,
                "prefill": self.timings.prefill_ms,
                "decode": self.timings.decode_ms,
                "model_total": self.timings.total_ms,
            },
            "profile": {
                "qwen3_profile_enabled": self.profile.qwen3_profile_enabled,
                "qmatmul_calls": self.profile.qmatmul_calls,
                "qmatmul_ms": self.profile.qmatmul_ms,
                "qmatmul_input_casts": self.profile.qmatmul_input_casts,
                "qmatmul_input_cast_ms": self.profile.qmatmul_input_cast_ms,
                "qmatmul_output_casts": self.profile.qmatmul_output_casts,
                "qmatmul_output_cast_ms": self.profile.qmatmul_output_cast_ms,
                "lm_head_calls": self.profile.lm_head_calls,
                "lm_head_ms": self.profile.lm_head_ms,
                "silu_mul_fused_calls": self.profile.silu_mul_fused_calls,
                "silu_mul_fallback_calls": self.profile.silu_mul_fallback_calls,
                "argmax_calls": self.profile.argmax_calls,
                "argmax_ms": self.profile.argmax_ms,
            }
        })
    }
}

const DEFAULT_TRANSCRIBE_MAX_NEW_TOKENS: usize = 512;
const ASR_INCREMENTAL_DECODE_RESYNC_TOKENS: usize = 32;
const QWEN3_ASR_CUDA_MAX_AUDIO_SECONDS: usize = 1_200;
/// One forced-alignment window on every backend (D1, as upstream). Its
/// forward working set is reserved per window by the forced-alignment
/// envelope; the convolution stem runs by slices and the portable prefill by
/// query blocks, so the window fits on CPU and Metal as well as CUDA.
pub(crate) const QWEN3_ALIGNER_MAX_AUDIO_SECONDS: usize = 180;
/// Alignable words (or CJK characters) one aligner window accepts per second
/// of audio. Fast speech runs about 5 per second; the bound keeps the prompt
/// (about 3.5 tokens per word) proportional to the encoded window.
const QWEN3_ALIGNER_MAX_WORDS_PER_SECOND: f64 = 10.0;
/// Word allowance for very short windows (a single short utterance).
const QWEN3_ALIGNER_MIN_WORD_ALLOWANCE: usize = 32;
const QWEN3_ASR_SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qwen3AsrFrontendGeometry {
    resampled_samples: usize,
    mel_frames_before_drop: usize,
    mel_frames_before_truncate: usize,
    mel_frames: usize,
    mel_last_frame_dropped: bool,
}

/// Exact scheduler-visible geometry for one Qwen3-ASR audio preparation row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen3AsrAudioPreparationGeometry {
    pub(crate) input_samples: usize,
    pub(crate) input_sample_rate: u32,
    pub(crate) resampled_samples: usize,
    pub(crate) mel_frames: usize,
    pub(crate) audio_tokens: usize,
    pub(crate) useful_tensor_elements: u64,
    pub(crate) retained_artifact_bytes: u64,
}

/// Exact padded/packed geometry of one native audio-tower call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen3AsrAudioPreparationBatchGeometry {
    pub(crate) rows: usize,
    pub(crate) max_mel_frames: usize,
    pub(crate) total_mel_frames: usize,
    pub(crate) padded_mel_elements: u64,
    pub(crate) padded_mel_elements_per_row: u64,
    pub(crate) max_audio_tokens: usize,
    pub(crate) total_audio_tokens: usize,
    pub(crate) total_useful_tensor_elements: u64,
    pub(crate) padded_output_elements: u64,
    pub(crate) padded_output_elements_per_row: u64,
    pub(crate) workspace_bytes: u64,
}

/// Load-time identity and capacity of the native audio preparation stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Qwen3AsrAudioPreparationStageSeal {
    pub(crate) backend: crate::backends::BackendKind,
    pub(crate) audio_dtype: String,
    pub(crate) text_dtype: String,
    pub(crate) max_batch_size: usize,
    pub(crate) max_workspace_bytes: u64,
}

fn checked_product_u64(values: &[usize], label: &str) -> Result<u64> {
    values.iter().try_fold(1u64, |product, value| {
        product
            .checked_mul(
                u64::try_from(*value)
                    .map_err(|_| Error::Overloaded(format!("{label} exceeds u64")))?,
            )
            .ok_or_else(|| Error::Overloaded(format!("{label} overflow")))
    })
}

fn qwen3_asr_resampled_sample_count(
    input_samples: usize,
    src_rate: u32,
    dst_rate: u32,
) -> Result<usize> {
    // Keep this ordering identical to `resample`: an identity conversion does
    // not inspect rates. Sharing this helper prevents admission geometry from
    // drifting from the physical front end.
    if src_rate == dst_rate {
        return Ok(input_samples);
    }
    if src_rate == 0 || dst_rate == 0 {
        return Err(Error::InvalidInput(format!(
            "Invalid sample rate for resampling: {src_rate} -> {dst_rate}"
        )));
    }
    Ok(crate::audio::target_sample_count(
        input_samples,
        src_rate,
        dst_rate,
    ))
}

fn qwen3_asr_mel_frame_count(samples: usize, n_fft: usize, hop_length: usize) -> Result<usize> {
    if hop_length == 0 {
        return Err(Error::InvalidInput(
            "Qwen3 ASR mel hop length must be greater than zero".to_string(),
        ));
    }
    let center_padding = (n_fft / 2)
        .checked_mul(2)
        .ok_or_else(|| Error::Overloaded("Qwen3 ASR mel center padding overflow".to_string()))?;
    let padded = samples.checked_add(center_padding).ok_or_else(|| {
        Error::Overloaded("Qwen3 ASR mel padded sample count overflow".to_string())
    })?;
    if padded < n_fft {
        return Ok(1);
    }
    (padded - n_fft)
        .checked_div(hop_length)
        .and_then(|frames| frames.checked_add(1))
        .ok_or_else(|| Error::Overloaded("Qwen3 ASR mel frame count overflow".to_string()))
}

/// The front end's sample and frame counts for `input_samples`. The mel drops
/// the last centre-padded frame, as [`qwen3_log_mel`] does.
fn qwen3_asr_frontend_geometry(
    input_samples: usize,
    input_sample_rate: u32,
    output_sample_rate: u32,
    n_fft: usize,
    hop_length: usize,
    cuda_device_observed: bool,
    configured_max_frames: usize,
) -> Result<Qwen3AsrFrontendGeometry> {
    let resampled_samples =
        qwen3_asr_resampled_sample_count(input_samples, input_sample_rate, output_sample_rate)?;
    let mel_frames_before_drop = qwen3_asr_mel_frame_count(resampled_samples, n_fft, hop_length)?;
    let mel_last_frame_dropped = mel_frames_before_drop > 0;
    let mel_frames_before_truncate =
        mel_frames_before_drop.saturating_sub(usize::from(mel_last_frame_dropped));
    let mel_frames = qwen3_asr_admit_frames(
        mel_frames_before_truncate,
        cuda_device_observed,
        false,
        configured_max_frames,
        usize::try_from(output_sample_rate)
            .map_err(|_| Error::InvalidInput("Qwen3 ASR sample rate exceeds usize".to_string()))?,
        hop_length,
    )?;
    Ok(Qwen3AsrFrontendGeometry {
        resampled_samples,
        mel_frames_before_drop,
        mel_frames_before_truncate,
        mel_frames,
        mel_last_frame_dropped,
    })
}

fn qwen3_asr_effective_max_frames(
    cuda_device_observed: bool,
    forced_aligner: bool,
    configured_max_frames: usize,
    sample_rate: usize,
    hop_length: usize,
) -> usize {
    let seconds = if forced_aligner {
        QWEN3_ALIGNER_MAX_AUDIO_SECONDS
    } else if cuda_device_observed {
        QWEN3_ASR_CUDA_MAX_AUDIO_SECONDS
    } else {
        return configured_max_frames;
    };
    seconds
        .saturating_mul(sample_rate.max(1))
        .checked_div(hop_length.max(1))
        .unwrap_or(0)
}

/// Whisper log-mel features as `WhisperFeatureExtractor` computes them for
/// Qwen3-ASR and the forced aligner: the centre-padded STFT without its last
/// frame, then clamped to `max - 8` and scaled, with the max taken over the
/// kept frames. Returns frame-major features and their frame count, one
/// frame per hop of audio (`samples / hop` for an even FFT size).
fn qwen3_log_mel(mel: &MelSpectrogram, samples: &[f32]) -> Result<(Vec<f32>, usize)> {
    debug_assert!(!mel.config().normalize);
    let (mut flat, frames) = mel.compute_flat(samples)?;
    let kept = frames.saturating_sub(1);
    flat.truncate(kept * mel.config().n_mels);
    MelSpectrogram::whisper_normalize_flat(&mut flat);
    Ok((flat, kept))
}

/// Most alignable words one aligner window of `audio_seconds` accepts.
pub(crate) fn qwen3_aligner_max_words(audio_seconds: f64) -> usize {
    let proportional = if audio_seconds.is_finite() && audio_seconds > 0.0 {
        (audio_seconds * QWEN3_ALIGNER_MAX_WORDS_PER_SECOND).ceil() as usize
    } else {
        0
    };
    proportional.max(QWEN3_ALIGNER_MIN_WORD_ALLOWANCE)
}

fn qwen3_asr_admit_frames(
    frames: usize,
    cuda_device_observed: bool,
    forced_aligner: bool,
    configured_max_frames: usize,
    sample_rate: usize,
    hop_length: usize,
) -> Result<usize> {
    let maximum = qwen3_asr_effective_max_frames(
        cuda_device_observed,
        forced_aligner,
        configured_max_frames,
        sample_rate,
        hop_length,
    );
    if maximum == 0 || frames <= maximum {
        return Ok(frames);
    }
    // The aligner times the whole reference text against the audio it
    // encodes, so a truncated mel would squeeze every later word into the
    // window. Over-limit alignment is rejected on every backend; callers
    // window the audio by `max_audio_seconds_hint`.
    if forced_aligner {
        return Err(Error::InvalidInput(format!(
            "Qwen3 forced alignment input has {frames} mel frames, exceeding the {maximum}-frame ({:.1} s) window; split the audio into windows of at most that length",
            maximum as f64 * hop_length.max(1) as f64 / sample_rate.max(1) as f64
        )));
    }
    if cuda_device_observed {
        return Err(Error::InvalidInput(format!(
            "Qwen3 ASR CUDA transcription input has {frames} mel frames, exceeding the explicit {maximum}-frame limit"
        )));
    }
    Ok(maximum)
}

impl Qwen3AsrModel {
    /// The loaded text decoder, for trunk tests on real weights.
    #[cfg(test)]
    pub(crate) fn into_text_model(self) -> Qwen3Model {
        self.text_model
    }

    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        let gguf_path = resolve_qwen_asr_gguf_path(model_dir, variant);
        let gguf_loader = if let Some(path) = gguf_path.as_ref() {
            Some(GgufLoader::from_path_with_backend(
                path,
                backend_kind_for_device(&device.device),
            )?)
        } else {
            None
        };
        let config = if let Some(loader) = gguf_loader.as_ref() {
            parse_qwen3_asr_config_from_gguf(loader, model_dir, variant)?
        } else {
            let config_path = model_dir.join("config.json");
            let config_str = fs::read_to_string(config_path)?;
            serde_json::from_str(&config_str)?
        };
        let model_family = config.model_family();
        let is_forced_aligner = model_family == ModelFamily::Qwen3ForcedAligner;
        let (timestamp_token_id, timestamp_segment_time_ms) = forced_aligner_timestamp_config(
            is_forced_aligner,
            config.timestamp_token_id,
            config.timestamp_segment_time,
        )?;

        let mut text_cfg = config.thinker_config.text_config.clone();
        let inferred_lm_head_size = if let Some(gguf_path) = gguf_path.as_ref() {
            infer_lm_head_size_from_gguf(gguf_path, Some("thinker.lm_head.weight"))?.or(
                infer_lm_head_size_from_gguf(gguf_path, Some("lm_head.weight"))?,
            )
        } else {
            infer_lm_head_size_from_checkpoint(model_dir, Some("thinker.lm_head.weight"))?.or(
                infer_lm_head_size_from_checkpoint(model_dir, Some("lm_head.weight"))?,
            )
        };
        if let Some(inferred_lm_head_size) = inferred_lm_head_size {
            if inferred_lm_head_size != text_cfg.vocab_size {
                debug!(
                    "Overriding Qwen speech-family lm_head output size from {} to {}",
                    text_cfg.vocab_size, inferred_lm_head_size
                );
                text_cfg.lm_head_size = Some(inferred_lm_head_size);
            }
        } else if let Some(classify_num) = config.thinker_config.classify_num {
            if classify_num != text_cfg.vocab_size {
                text_cfg.lm_head_size = Some(classify_num);
            }
        }

        let tokenizer = if let Some(loader) = gguf_loader.as_ref() {
            AsrTokenizer::load_from_gguf(model_dir, loader, text_cfg.vocab_size)?
        } else {
            AsrTokenizer::load(model_dir, text_cfg.vocab_size)?
        };
        let specials = tokenizer.specials().clone();

        let preprocessor: PreprocessorConfig = if let Some(loader) = gguf_loader.as_ref() {
            preprocessor_config_from_gguf(loader)?
        } else {
            let path = model_dir.join("preprocessor_config.json");
            let data = fs::read_to_string(path)?;
            serde_json::from_str(&data)?
        };

        let mel_cfg = MelConfig {
            sample_rate: 16_000,
            n_fft: preprocessor.n_fft.max(1),
            win_length: None,
            hop_length: preprocessor.hop_length.max(1),
            n_mels: preprocessor.feature_size.max(1),
            f_min: 0.0,
            f_max: 8_000.0,
            // Normalized in `qwen3_log_mel`, after the last frame is dropped.
            normalize: false,
            mel_scale: MelScale::Slaney,
            mel_norm: MelNorm::Slaney,
        };
        let mel = MelSpectrogram::new(mel_cfg)?;

        // Quantized text checkpoints are trained/evaluated in bf16 and can
        // degrade when forced through fp32 dequant paths. CUDA audio kernels are
        // stricter around f32 boundaries, so select audio/text dtypes separately.
        let is_gguf = gguf_loader.is_some();
        let checkpoint_format = if is_gguf { "gguf" } else { "safetensors" };
        let is_quantized = is_gguf || config.is_mlx_quantized();
        if !is_gguf && is_quantized {
            validate_quantization_config(&config)?;
        }
        let (text_selection, audio_selection) = qwen3_asr_weight_precisions(
            &device,
            model_family,
            is_gguf,
            is_quantized,
            config.thinker_config.dtype.as_deref(),
        )?;

        // Pre-initialize shared scratch pools up front to keep first-use
        // allocation costs out of ASR prefill/decode hot loops.
        let _ = maybe_init_global_buffer_pool(&device.device);
        let _ = metal_pool_for_device(&device.device);

        let audio_cfg = config.thinker_config.audio_config.clone();
        let text_context_tokens = text_cfg.context_length();
        let gguf_qmatmul_text_enabled = is_gguf && qwen3_asr_use_gguf_qmatmul_text(&device.device);
        // On Metal a lowered audio or text dtype must pass its own load-time
        // probe, or that component alone reloads in F32 (report 13).
        let check = device.kind.is_metal();
        let (audio_tower, audio_precision, text_model, text_precision) =
            if let Some(gguf_path) = gguf_path.as_ref() {
                let loader = gguf_loader.as_ref().ok_or_else(|| {
                    Error::ModelLoadError("Qwen ASR GGUF loader missing for GGUF path".to_string())
                })?;
                let tensor_prefix = qwen3_asr_gguf_tensor_prefix(loader);
                load_qwen3_asr_components(
                    &audio_selection,
                    &text_selection,
                    check,
                    &device.device,
                    |audio_dtype| {
                        let vb_audio = var_builder_from_gguf_filtered(
                            gguf_path,
                            audio_dtype,
                            &device.device,
                            |name| qwen3_asr_gguf_audio_tensor_matches(name, tensor_prefix),
                        )?;
                        let vb_audio = if tensor_prefix.is_empty() {
                            vb_audio
                        } else {
                            vb_audio.pp(tensor_prefix)
                        };
                        AudioTower::load(audio_cfg.clone(), vb_audio.pp("audio_tower"))
                    },
                    |text_dtype| {
                        if gguf_qmatmul_text_enabled {
                            return Qwen3Model::load_gguf_text(
                                text_cfg.clone(),
                                loader,
                                &device.device,
                                text_dtype,
                                tensor_prefix,
                            );
                        }
                        let vb_text = var_builder_from_gguf_filtered(
                            gguf_path,
                            text_dtype,
                            &device.device,
                            |name| qwen3_asr_gguf_text_tensor_matches(name, tensor_prefix),
                        )?;
                        let vb_text = if tensor_prefix.is_empty() {
                            vb_text
                        } else {
                            vb_text.pp(tensor_prefix)
                        };
                        Qwen3Model::load(text_cfg.clone(), vb_text)
                    },
                )?
            } else {
                let shard_paths = resolve_safetensors_shards(model_dir)?;
                let var_builder = |dtype: DType| -> Result<VarBuilder<'static>> {
                    // SAFETY: the mapped shards are read-only checkpoint files.
                    let vb = unsafe {
                        staged::metal_synced_safetensors(&shard_paths, dtype, &device.device)?
                    };
                    Ok(
                        if vb.contains_tensor("thinker.audio_tower.conv2d1.weight") {
                            vb.pp("thinker")
                        } else {
                            vb
                        },
                    )
                };
                load_qwen3_asr_components(
                    &audio_selection,
                    &text_selection,
                    check,
                    &device.device,
                    |audio_dtype| {
                        AudioTower::load(
                            audio_cfg.clone(),
                            var_builder(audio_dtype)?.pp("audio_tower"),
                        )
                    },
                    |text_dtype| Qwen3Model::load(text_cfg.clone(), var_builder(text_dtype)?),
                )?
            };
        let (audio_dtype, text_dtype) = (audio_precision.loaded, text_precision.loaded);

        info!(
            "Loaded Qwen speech-family model on {:?} (audio_dtype={:?}: {}; text_dtype={:?}: {})",
            device.kind, audio_dtype, audio_precision.reason, text_dtype, text_precision.reason
        );

        Ok(Self {
            audio_preparation_id: NEXT_QWEN3_ASR_AUDIO_PREPARATION_ID
                .fetch_add(1, Ordering::Relaxed),
            device,
            audio_dtype,
            text_dtype,
            checkpoint_format,
            gguf_qmatmul_text_enabled,
            is_forced_aligner,
            timestamp_token_id,
            timestamp_segment_time_ms,
            tokenizer,
            specials,
            audio_tower,
            text_model,
            text_context_tokens,
            mel,
            preprocessor,
            audio_precision,
            text_precision,
        })
    }

    fn execution_diagnostics_for_physical(
        &self,
        cache: &PhysicalPagedKvCache,
    ) -> Qwen3AsrExecutionDiagnostics {
        self.execution_diagnostics_with_cache_geometry(
            cache.arena().config().page_tokens as usize,
            0,
        )
    }

    fn execution_diagnostics_with_cache_geometry(
        &self,
        page_size: usize,
        dense_decode_max_tokens: usize,
    ) -> Qwen3AsrExecutionDiagnostics {
        let projection_diagnostics = self.text_model.projection_diagnostics();
        let text_projection_quantized = projection_diagnostics.quantized_projection_count > 0;
        let text_projection_backend = if text_projection_quantized {
            "qmatmul"
        } else {
            "dense_linear"
        };

        Qwen3AsrExecutionDiagnostics {
            device_kind: format!("{:?}", self.device.kind),
            audio_dtype: format!("{:?}", self.audio_dtype),
            text_dtype: format!("{:?}", self.text_dtype),
            checkpoint_format: self.checkpoint_format.to_string(),
            flash_attention_requested: flash_attention_requested(),
            flash_attention_compiled: flash_attention_compiled(),
            kv_page_size: page_size,
            dense_decode_enabled: dense_decode_max_tokens > 0,
            dense_decode_max_tokens,
            gguf_qmatmul_text_enabled: self.gguf_qmatmul_text_enabled,
            text_projection_backend: text_projection_backend.to_string(),
            text_projection_quantized,
            qmatmul_projection_count: projection_diagnostics.quantized_projection_count,
            dense_projection_count: projection_diagnostics.dense_projection_count,
            dense_bias_projection_count: projection_diagnostics.dense_bias_projection_count,
            audio_precision: self.audio_precision.diagnostics(),
            text_precision: self.text_precision.diagnostics(),
        }
    }

    pub fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<String> {
        self.transcribe_with_prompt(audio, sample_rate, language, None)
    }

    pub fn transcribe_with_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        system_prompt: Option<&str>,
    ) -> Result<String> {
        let _ = (audio, sample_rate, language, system_prompt);
        Err(Error::InferenceError(
            "Qwen3 ASR transcription requires a lifecycle-owned physical invocation cache"
                .to_string(),
        ))
    }

    pub fn transcribe_with_details(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<AsrTranscriptionOutput> {
        self.transcribe_with_details_and_prompt(audio, sample_rate, language, None)
    }

    pub fn transcribe_with_details_and_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        system_prompt: Option<&str>,
    ) -> Result<AsrTranscriptionOutput> {
        let _ = (audio, sample_rate, language, system_prompt);
        Err(Error::InferenceError(
            "Qwen3 ASR transcription requires a lifecycle-owned physical invocation cache"
                .to_string(),
        ))
    }

    /// Upper-bound hint for chunk sizing in long-form ASR orchestration.
    pub fn max_audio_seconds_hint(&self) -> Option<f32> {
        let sample_rate = self.mel.config().sample_rate.max(1) as f32;
        let maximum_frames = qwen3_asr_effective_max_frames(
            self.device.device.is_cuda(),
            self.is_forced_aligner,
            self.preprocessor.nb_max_frames,
            self.mel.config().sample_rate,
            self.mel.config().hop_length,
        );
        if maximum_frames > 0 {
            // One mel frame per hop of audio: the window is exactly the frame
            // limit's audio (180 s for the aligner).
            let hop = self.mel.config().hop_length.max(1);
            return Some(maximum_frames as f32 * hop as f32 / sample_rate);
        }
        if self.preprocessor.n_samples > 0 {
            return Some(self.preprocessor.n_samples as f32 / sample_rate);
        }
        None
    }

    /// Decoder-prompt tokens added per second of audio: the audio tower emits
    /// 13 tokens per 100 mel frames (16 kHz at the configured hop → 160 fps).
    pub(crate) fn audio_token_rate(&self) -> f32 {
        let frames_per_sec =
            QWEN3_ASR_SAMPLE_RATE as f32 / self.mel.config().hop_length.max(1) as f32;
        13.0 * frames_per_sec / 100.0
    }

    /// Resolve the exact text-decoder input span before scheduler admission.
    /// The physical KV allocator must see the CNN-downsampled audio-token
    /// count, not a text-length estimate, or its block table and receipts will
    /// disagree with the prefill that the model actually executes.
    pub(crate) fn incremental_prompt_token_count(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        system_prompt: Option<&str>,
    ) -> Result<usize> {
        if self.is_forced_aligner {
            return Err(Error::InvalidInput(
                "Qwen3-ForcedAligner models do not support transcription decode state.".to_string(),
            ));
        }
        // Center padding makes the mel frame count a pure function of sample
        // count and configuration. Admission therefore does not resample the
        // waveform or compute a mel tensor that the encoder would discard.
        let mel = self.mel.config();
        let geometry = qwen3_asr_frontend_geometry(
            audio.len(),
            sample_rate,
            QWEN3_ASR_SAMPLE_RATE,
            mel.n_fft,
            mel.hop_length,
            self.device.device.is_cuda(),
            self.preprocessor.nb_max_frames,
        )?;
        if geometry.mel_frames == 0 {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }
        let audio_tokens = self.audio_tower.output_lengths(&[geometry.mel_frames])?[0];
        if audio_tokens == 0 {
            return Err(Error::InvalidInput(
                "Qwen3 ASR audio produced no decoder input tokens".to_string(),
            ));
        }
        Ok(self
            .build_prompt(audio_tokens, language, system_prompt)?
            .ids
            .len())
    }

    pub fn transcribe_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        self.transcribe_with_callback_and_prompt(audio, sample_rate, language, None, on_delta)
    }

    pub fn transcribe_with_callback_and_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        system_prompt: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let _ = (audio, sample_rate, language, system_prompt, on_delta);
        Err(Error::InferenceError(
            "Qwen3 ASR transcription requires a lifecycle-owned physical invocation cache"
                .to_string(),
        ))
    }

    /// Run one offline ASR invocation against a load-owned exclusive page
    /// lease. The caller retains the lease so it can validate and fence every
    /// write before returning the slot to the invocation pool.
    pub(crate) fn transcribe_with_details_and_prompt_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        system_prompt: Option<&str>,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<AsrTranscriptionOutput> {
        self.transcribe_impl_physical(audio, sample_rate, language, system_prompt, None, cache)
    }

    pub(crate) fn transcribe_with_callback_and_prompt_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        system_prompt: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<String> {
        Ok(self
            .transcribe_impl_physical(
                audio,
                sample_rate,
                language,
                system_prompt,
                Some(on_delta),
                cache,
            )?
            .text)
    }

    fn transcribe_impl_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        system_prompt: Option<&str>,
        on_delta: Option<&mut dyn FnMut(&str)>,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<AsrTranscriptionOutput> {
        if cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "Qwen3 ASR invocation cache must start at context zero".to_string(),
            ));
        }
        let mut state = self.start_decode_with_prompt_invocation(
            audio,
            sample_rate,
            language,
            system_prompt,
            DEFAULT_TRANSCRIBE_MAX_NEW_TOKENS,
            cache,
        )?;
        self.finish_transcription(&mut state, language, on_delta, Some(cache))
    }

    fn finish_transcription(
        &self,
        state: &mut AsrDecodeState,
        language: Option<&str>,
        mut on_delta: Option<&mut dyn FnMut(&str)>,
        mut invocation_cache: Option<&mut PhysicalPagedKvCache>,
    ) -> Result<AsrTranscriptionOutput> {
        let decode_started = Instant::now();
        loop {
            let step = if on_delta.is_some() {
                self.decode_step_inner(state, true, invocation_cache.as_deref_mut())?
            } else {
                self.decode_step_inner(state, false, invocation_cache.as_deref_mut())?
            };
            if !step.delta.is_empty() {
                if let Some(on_delta) = on_delta.as_deref_mut() {
                    for ch in step.delta.chars() {
                        let mut buf = [0u8; 4];
                        on_delta(ch.encode_utf8(&mut buf));
                    }
                }
            }
            if step.finished {
                state.diagnostics.timings.decode_ms = elapsed_ms(decode_started);
                state.diagnostics.timings.total_ms += state.diagnostics.timings.decode_ms;
                state.diagnostics.generated_tokens = step.tokens_generated;
                if state.diagnostics.profile.qwen3_profile_enabled {
                    state.diagnostics.profile.update_from_runtime_delta(
                        qwen3_runtime_profile_delta(state.profile_start),
                    );
                }
                // `step.text` is already parsed; parse the raw output once so
                // the detected language survives and a transcript that starts
                // with "language ..." is not stripped twice.
                let (detected_language, text) = parse_asr_output(&state.assembled, language);
                return Ok(AsrTranscriptionOutput {
                    text,
                    language: detected_language,
                    diagnostics: Some(state.diagnostics.to_json()),
                });
            }
        }
    }

    /// Start incremental ASR decode with scheduler-owned authoritative pages.
    ///
    /// The reservation must be a fresh ASR session covering the exact prompt
    /// quantum. Later decode quanta replace its block-table view through
    /// [`AsrDecodeState::install_managed_reservation`]. Prefix reuse is
    /// deliberately rejected because audio-derived embeddings are not
    /// represented by text token IDs.
    pub(crate) fn start_decode_with_prompt_managed(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        system_prompt: Option<&str>,
        max_new_tokens: usize,
        cache: PhysicalPagedKvCache,
    ) -> Result<AsrDecodeState> {
        if cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "Qwen3 ASR managed cache must start with an empty multimodal context".to_string(),
            ));
        }
        let mut state = self.begin_resumable_prefill_managed(
            audio,
            sample_rate,
            language,
            system_prompt,
            max_new_tokens,
            cache,
        )?;
        let prompt_tokens = state.prefill_token_count();
        self.continue_resumable_prefill(&mut state, 0, prompt_tokens)?;
        Ok(state)
    }

    /// Prepare immutable multimodal decoder inputs without writing model KV.
    ///
    /// The scheduler may subsequently commit exact monotonic spans with
    /// [`Self::continue_resumable_prefill`]. The physical cache must be empty:
    /// ASR audio embeddings make token-only prefix reconstruction unsafe.
    pub(crate) fn begin_resumable_prefill_managed(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        system_prompt: Option<&str>,
        max_new_tokens: usize,
        cache: PhysicalPagedKvCache,
    ) -> Result<AsrDecodeState> {
        if cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "Qwen3 ASR resumable prefill requires empty physical KV".to_string(),
            ));
        }
        let execution = self.execution_diagnostics_for_physical(&cache);
        self.prepare_decode_state(
            audio,
            sample_rate,
            language,
            system_prompt,
            max_new_tokens,
            Some(cache),
            execution,
        )
    }

    /// Commit one exact scheduler-owned span of the prepared multimodal
    /// prompt. Intermediate spans update KV only; logits become visible only
    /// after the final span commits.
    pub(crate) fn continue_resumable_prefill(
        &self,
        state: &mut AsrDecodeState,
        span_start: usize,
        span_end: usize,
    ) -> Result<bool> {
        continue_asr_resumable_prefill(&self.text_model, state, span_start, span_end)
    }

    fn start_decode_with_prompt_invocation(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        system_prompt: Option<&str>,
        max_new_tokens: usize,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<AsrDecodeState> {
        let execution = self.execution_diagnostics_for_physical(cache);
        let mut state = self.prepare_decode_state(
            audio,
            sample_rate,
            language,
            system_prompt,
            max_new_tokens,
            None,
            execution,
        )?;
        complete_asr_invocation_prefill(&self.text_model, &mut state, cache)?;
        Ok(state)
    }

    /// Run the independently schedulable audio front end for a ragged group.
    ///
    /// Resampling and mel extraction remain host-side per row. Their exact
    /// frame-major outputs are padded once for a single device upload, then the
    /// audio tower packs all logical rows into one convolution/encoder pass.
    /// No decoder KV is allocated or mutated by this operation.
    pub(crate) fn prepare_audio_tower_batch(
        &self,
        rows: &[Qwen3AsrAudioBatchRow<'_>],
    ) -> Result<Vec<Qwen3AsrPreparedAudio>> {
        if self.is_forced_aligner {
            return Err(Error::InvalidInput(
                "Qwen3-ForcedAligner models do not support transcription audio preparation."
                    .to_string(),
            ));
        }
        if rows.is_empty() {
            return Err(Error::InvalidInput(
                "Qwen3 ASR audio preparation requires at least one row".to_string(),
            ));
        }
        let batch_started = Instant::now();

        struct HostFrontendRow {
            flat: Vec<f32>,
            frontend: Qwen3AsrPreparedFrontend,
        }

        let mel_config = self.mel.config();
        let n_mels = mel_config.n_mels;
        let mut host_rows = Vec::with_capacity(rows.len());
        for row in rows {
            let geometry = qwen3_asr_frontend_geometry(
                row.audio.len(),
                row.sample_rate,
                QWEN3_ASR_SAMPLE_RATE,
                mel_config.n_fft,
                mel_config.hop_length,
                self.device.device.is_cuda(),
                self.preprocessor.nb_max_frames,
            )?;
            if geometry.mel_frames == 0 {
                return Err(Error::InvalidInput("Empty audio input".to_string()));
            }

            let resample_started = Instant::now();
            let resampled = if row.sample_rate != QWEN3_ASR_SAMPLE_RATE {
                resample(row.audio, row.sample_rate, QWEN3_ASR_SAMPLE_RATE)?
            } else {
                row.audio.to_vec()
            };
            let resample_ms = elapsed_ms(resample_started);
            if resampled.len() != geometry.resampled_samples {
                return Err(Error::InferenceError(format!(
                    "Qwen3 ASR resample geometry drifted: predicted={}, actual={}",
                    geometry.resampled_samples,
                    resampled.len()
                )));
            }

            let mel_started = Instant::now();
            let (mut flat, actual_frames) = qwen3_log_mel(&self.mel, &resampled)?;
            if actual_frames != geometry.mel_frames_before_truncate {
                return Err(Error::InferenceError(format!(
                    "Qwen3 ASR mel geometry drifted: predicted={}, actual={actual_frames}",
                    geometry.mel_frames_before_truncate
                )));
            }
            flat.truncate(
                geometry
                    .mel_frames
                    .checked_mul(n_mels)
                    .ok_or_else(|| Error::Overloaded("Qwen3 ASR mel size overflow".to_string()))?,
            );
            let mel_ms = elapsed_ms(mel_started);
            host_rows.push(HostFrontendRow {
                flat,
                frontend: Qwen3AsrPreparedFrontend {
                    input_sample_rate: row.sample_rate,
                    input_samples: row.audio.len(),
                    resampled_sample_rate: QWEN3_ASR_SAMPLE_RATE,
                    resampled_samples: geometry.resampled_samples,
                    mel_frames: geometry.mel_frames,
                    mel_frames_before_drop: geometry.mel_frames_before_drop,
                    mel_frames_before_truncate: geometry.mel_frames_before_truncate,
                    mel_last_frame_dropped: geometry.mel_last_frame_dropped,
                    mel_frames_truncated: geometry.mel_frames_before_truncate > geometry.mel_frames,
                    resample_ms,
                    mel_ms,
                    ..Default::default()
                },
            });
        }

        let max_frames = host_rows
            .iter()
            .map(|row| row.frontend.mel_frames)
            .max()
            .unwrap_or(0);
        let row_stride = max_frames
            .checked_mul(n_mels)
            .ok_or_else(|| Error::Overloaded("Qwen3 ASR padded mel row overflow".to_string()))?;
        let padded_elements = rows
            .len()
            .checked_mul(row_stride)
            .ok_or_else(|| Error::Overloaded("Qwen3 ASR padded mel batch overflow".to_string()))?;
        let mut padded_flat = Vec::new();
        padded_flat
            .try_reserve_exact(padded_elements)
            .map_err(|error| {
                Error::Overloaded(format!(
                    "Qwen3 ASR padded mel batch allocation failed: {error}"
                ))
            })?;
        padded_flat.resize(padded_elements, 0.0f32);
        for (index, row) in host_rows.iter().enumerate() {
            let offset = index * row_stride;
            let end = offset + row.flat.len();
            padded_flat[offset..end].copy_from_slice(&row.flat);
        }
        let feature_lens = host_rows
            .iter()
            .map(|row| row.frontend.mel_frames)
            .collect::<Vec<_>>();

        let upload_started = Instant::now();
        let features = Tensor::from_vec(
            padded_flat,
            (rows.len(), max_frames, n_mels),
            &self.device.device,
        )?
        .to_dtype(self.audio_dtype)?;
        let upload_ms = elapsed_ms(upload_started);

        let encode_started = Instant::now();
        let encoded = self
            .audio_tower
            .forward_feature_batch(&features, &feature_lens)?;
        let expected_output_lengths = self.audio_tower.output_lengths(&feature_lens)?;
        if encoded.output_lengths() != expected_output_lengths {
            return Err(Error::InferenceError(format!(
                "Qwen3 ASR audio tower output geometry drifted: predicted={expected_output_lengths:?}, actual={:?}",
                encoded.output_lengths()
            )));
        }
        let mut exact_rows = Vec::with_capacity(rows.len());
        for index in 0..rows.len() {
            let mut row = encoded.row(index)?;
            if row.dtype() != self.text_dtype {
                row = row.to_dtype(self.text_dtype)?;
            }
            // Isolate every retained row in exact-sized persistent storage.
            // Metal pool bins can otherwise make even B1 backing larger than
            // the logical artifact admitted by the scheduler.
            exact_rows.push(deep_copy_tensor_storage(&row)?);
        }
        let encode_ms = elapsed_ms(encode_started);
        let frontend_total_ms = elapsed_ms(batch_started);

        Ok(host_rows
            .into_iter()
            .zip(exact_rows)
            .map(|(mut row, audio_embeddings)| {
                row.frontend.mel_flatten_upload_ms = upload_ms;
                row.frontend.audio_encode_ms = encode_ms;
                row.frontend.frontend_total_ms = frontend_total_ms;
                Qwen3AsrPreparedAudio {
                    preparation_id: self.audio_preparation_id,
                    audio_embeddings,
                    frontend: row.frontend,
                }
            })
            .collect())
    }

    /// Materialize decoder input artifacts from an already encoded audio row.
    /// The prepared audio remains immutable and can be retried after a rejected
    /// cache reservation without repeating the audio tower.
    pub(crate) fn begin_resumable_prefill_managed_from_prepared_audio(
        &self,
        prepared: &Qwen3AsrPreparedAudio,
        language: Option<&str>,
        system_prompt: Option<&str>,
        max_new_tokens: usize,
        cache: PhysicalPagedKvCache,
    ) -> Result<AsrDecodeState> {
        if cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "Qwen3 ASR resumable prefill requires empty physical KV".to_string(),
            ));
        }
        let execution = self.execution_diagnostics_for_physical(&cache);
        self.prepare_decode_state_from_prepared_audio(
            prepared,
            language,
            system_prompt,
            max_new_tokens,
            Some(cache),
            execution,
        )
    }

    pub(crate) fn start_decode_with_prompt_managed_from_prepared_audio(
        &self,
        prepared: &Qwen3AsrPreparedAudio,
        language: Option<&str>,
        system_prompt: Option<&str>,
        max_new_tokens: usize,
        cache: PhysicalPagedKvCache,
    ) -> Result<AsrDecodeState> {
        let mut state = self.begin_resumable_prefill_managed_from_prepared_audio(
            prepared,
            language,
            system_prompt,
            max_new_tokens,
            cache,
        )?;
        let prompt_tokens = state.prefill_token_count();
        self.continue_resumable_prefill(&mut state, 0, prompt_tokens)?;
        Ok(state)
    }

    fn prepare_decode_state(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        system_prompt: Option<&str>,
        max_new_tokens: usize,
        cache: Option<PhysicalPagedKvCache>,
        execution: Qwen3AsrExecutionDiagnostics,
    ) -> Result<AsrDecodeState> {
        let rows = [Qwen3AsrAudioBatchRow { audio, sample_rate }];
        let prepared = self
            .prepare_audio_tower_batch(&rows)?
            .pop()
            .ok_or_else(|| {
                Error::InferenceError(
                    "Qwen3 ASR scalar audio preparation returned no row".to_string(),
                )
            })?;
        self.prepare_decode_state_from_prepared_audio(
            &prepared,
            language,
            system_prompt,
            max_new_tokens,
            cache,
            execution,
        )
    }

    fn prepare_decode_state_from_prepared_audio(
        &self,
        prepared: &Qwen3AsrPreparedAudio,
        language: Option<&str>,
        system_prompt: Option<&str>,
        max_new_tokens: usize,
        cache: Option<PhysicalPagedKvCache>,
        execution: Qwen3AsrExecutionDiagnostics,
    ) -> Result<AsrDecodeState> {
        if prepared.preparation_id != self.audio_preparation_id {
            return Err(Error::InvalidInput(
                "Qwen3 ASR prepared audio belongs to a different loaded model".to_string(),
            ));
        }
        let profile_start = qwen3_runtime_profile_snapshot();
        let profile_enabled = qwen3_runtime_profiling_enabled();
        let total_started = Instant::now();
        let audio_len = prepared.audio_tokens()?;

        let prompt = self.build_prompt(audio_len, language, system_prompt)?;
        if let Some(context_tokens) = self.text_context_tokens {
            let required = prompt.ids.len().saturating_add(max_new_tokens.max(1));
            if required > context_tokens {
                return Err(Error::InvalidInput(format!(
                    "Qwen3 ASR prompt plus decode budget ({required}) exceeds model context ({context_tokens})"
                )));
            }
        }
        let input_ids = Tensor::from_vec(
            prompt.ids.clone(),
            (1, prompt.ids.len()),
            &self.device.device,
        )?;
        let max_new_tokens = max_new_tokens.max(1);
        let (prepared_prompt_embeddings, prepared_position_ids) = self
            .prepare_audio_prompt_embeddings(
                &input_ids,
                &prepared.audio_embeddings,
                prompt.audio_pad_start,
                prompt.audio_pad_len,
            )?;
        let frontend = &prepared.frontend;
        let diagnostics = Qwen3AsrDiagnostics {
            input_sample_rate: frontend.input_sample_rate,
            input_samples: frontend.input_samples,
            resampled_sample_rate: frontend.resampled_sample_rate,
            resampled_samples: frontend.resampled_samples,
            mel_frames: frontend.mel_frames,
            mel_frames_before_drop: frontend.mel_frames_before_drop,
            mel_frames_before_truncate: frontend.mel_frames_before_truncate,
            mel_last_frame_dropped: frontend.mel_last_frame_dropped,
            mel_frames_truncated: frontend.mel_frames_truncated,
            audio_tokens: audio_len,
            prompt_tokens: prompt.ids.len(),
            system_prompt_requested: prompt.system_prompt_requested,
            system_prompt_token_count: prompt.system_prompt_token_count,
            forced_language: prompt.forced_language,
            forced_language_token_count: prompt.forced_language_token_count,
            asr_text_token_available: prompt.asr_text_token_available,
            asr_text_token_appended: prompt.asr_text_token_appended,
            max_new_tokens,
            generated_tokens: 0,
            execution,
            profile: Qwen3AsrProfileDiagnostics {
                qwen3_profile_enabled: profile_enabled,
                ..Default::default()
            },
            timings: Qwen3AsrTimingDiagnostics {
                resample_ms: frontend.resample_ms,
                mel_ms: frontend.mel_ms,
                mel_flatten_upload_ms: frontend.mel_flatten_upload_ms,
                audio_encode_ms: frontend.audio_encode_ms,
                prefill_ms: 0.0,
                decode_ms: 0.0,
                total_ms: frontend.frontend_total_ms + elapsed_ms(total_started),
            },
        };

        Ok(AsrDecodeState {
            cache,
            physical_tensor_sequence: None,
            prepared_prompt_embeddings: Some(prepared_prompt_embeddings),
            prepared_position_ids,
            prefill_progress: 0,
            unconsumed_output: None,
            pos: 0,
            pending_token: None,
            language: language.map(ToString::to_string),
            generated_ids: Vec::new(),
            visible_generated_ids: Vec::new(),
            decoded_visible_tokens: 0,
            tokens_since_resync: 0,
            assembled: String::new(),
            diagnostics,
            profile_start,
            stop_tokens: collect_stop_token_ids(&self.specials),
            max_new_tokens,
            finished: false,
        })
    }

    fn finalized_text(&self, state: &AsrDecodeState) -> String {
        let (_, text) = parse_asr_output(&state.assembled, state.language.as_deref());
        text
    }

    fn refresh_full_decoded_text(&self, state: &mut AsrDecodeState) -> Result<()> {
        state.assembled = self.decode_generated_untrimmed(&state.generated_ids)?;
        state.visible_generated_ids = state
            .generated_ids
            .iter()
            .copied()
            .filter(|id| !is_special_generation_token(&self.specials, *id))
            .collect();
        state.decoded_visible_tokens = state.visible_generated_ids.len();
        state.tokens_since_resync = 0;
        Ok(())
    }

    fn decode_incremental_delta(&self, state: &mut AsrDecodeState) -> Result<String> {
        if state.visible_generated_ids.len() <= state.decoded_visible_tokens {
            return Ok(String::new());
        }

        let new_ids = &state.visible_generated_ids[state.decoded_visible_tokens..];
        let mut delta = self.tokenizer.decode_text(new_ids)?;
        state.decoded_visible_tokens = state.visible_generated_ids.len();
        state.tokens_since_resync = state.tokens_since_resync.saturating_add(new_ids.len());

        if state.tokens_since_resync >= ASR_INCREMENTAL_DECODE_RESYNC_TOKENS {
            let full = self.tokenizer.decode_text(&state.visible_generated_ids)?;
            delta = text_delta(&state.assembled, &full);
            state.assembled = full;
            state.tokens_since_resync = 0;
            return Ok(delta);
        }

        state.assembled.push_str(&delta);
        Ok(delta)
    }

    pub fn decode_step(&self, state: &mut AsrDecodeState) -> Result<AsrDecodeStep> {
        self.decode_step_inner(state, true, None)
    }

    fn decode_step_without_delta(&self, state: &mut AsrDecodeState) -> Result<AsrDecodeStep> {
        self.decode_step_inner(state, false, None)
    }

    fn decode_step_inner(
        &self,
        state: &mut AsrDecodeState,
        emit_delta: bool,
        invocation_cache: Option<&mut PhysicalPagedKvCache>,
    ) -> Result<AsrDecodeStep> {
        if state.finished || state.generated_ids.len() >= state.max_new_tokens {
            state.finished = true;
            self.refresh_full_decoded_text(state)?;
            return Ok(AsrDecodeStep {
                delta: String::new(),
                text: self.finalized_text(state),
                tokens_generated: state.generated_ids.len(),
                finished: true,
            });
        }

        if let Some(pending) = state.pending_token.take() {
            let next_tensor = Tensor::from_vec(vec![pending], (1, 1), &self.device.device)?;
            state.unconsumed_output = Some(if let Some(cache) = invocation_cache {
                self.text_model
                    .forward_managed(&next_tensor, state.pos, cache)?
            } else {
                self.text_model.forward_managed(
                    &next_tensor,
                    state.pos,
                    state.cache.as_mut().ok_or_else(|| {
                        Error::InferenceError(
                            "Qwen3 ASR decode state has no physical cache authority".to_string(),
                        )
                    })?,
                )?
            });
            state.pos += 1;
        }

        let argmax_started = state
            .diagnostics
            .profile
            .qwen3_profile_enabled
            .then(Instant::now);
        let next = take_quantum_argmax(&mut state.unconsumed_output)?;
        if let Some(started) = argmax_started {
            state.diagnostics.profile.argmax_calls =
                state.diagnostics.profile.argmax_calls.saturating_add(1);
            state.diagnostics.profile.argmax_ms += elapsed_ms(started);
        }
        self.finish_decode_sample(state, next, emit_delta)
    }

    fn finish_decode_sample(
        &self,
        state: &mut AsrDecodeState,
        next: u32,
        emit_delta: bool,
    ) -> Result<AsrDecodeStep> {
        if state.stop_tokens.contains(&next) {
            state.finished = true;
            self.refresh_full_decoded_text(state)?;
            return Ok(AsrDecodeStep {
                delta: String::new(),
                text: self.finalized_text(state),
                tokens_generated: state.generated_ids.len(),
                finished: true,
            });
        }

        state.generated_ids.push(next);
        state.pending_token = Some(next);
        if !is_special_generation_token(&self.specials, next) {
            state.visible_generated_ids.push(next);
        }
        let delta = if emit_delta {
            self.decode_incremental_delta(state)?
        } else {
            String::new()
        };

        if state.generated_ids.len() >= state.max_new_tokens {
            state.finished = true;
            self.refresh_full_decoded_text(state)?;
        }

        Ok(AsrDecodeStep {
            delta,
            text: if state.finished {
                self.finalized_text(state)
            } else if emit_delta {
                state.assembled.trim().to_string()
            } else {
                String::new()
            },
            tokens_generated: state.generated_ids.len(),
            finished: state.finished,
        })
    }

    /// Decode one scheduled token for a changing set of independent retained
    /// ASR sessions. Rows may have different physical positions, but a native
    /// multi-row call requires their page tables to share the Qwen KV arena.
    /// A one-row batch deliberately uses the scalar managed forward as the
    /// compatibility fallback.
    pub fn decode_step_batch(
        &self,
        states: &mut [&mut AsrDecodeState],
    ) -> Result<Vec<AsrDecodeStep>> {
        if states.is_empty() {
            return Ok(Vec::new());
        }
        for state in states.iter() {
            if state.finished || state.generated_ids.len() >= state.max_new_tokens {
                return Err(Error::InvalidInput(
                    "continuous Qwen3 ASR batch contains a terminal decode state".to_string(),
                ));
            }
            if state.prefill_progress != state.prefill_token_count()
                || state.prepared_prompt_embeddings.is_some()
                || state.prepared_position_ids.is_some()
                || state.unconsumed_output.is_some()
                || state.pending_token.is_none()
                || state.cache.is_none()
            {
                return Err(Error::InferenceError(
                    "continuous Qwen3 ASR decode requires a completed prefill and one scheduled input token per retained row"
                        .to_string(),
                ));
            }
        }

        let argmax_started = Instant::now();
        let output =
            forward_asr_pending_decode_batch(&self.text_model, &self.device.device, states)?;
        let sampled = batched_decode_argmax(&output)?;
        let argmax_ms = elapsed_ms(argmax_started);
        let mut steps = Vec::with_capacity(states.len());
        for (state, next) in states.iter_mut().zip(sampled) {
            if state.diagnostics.profile.qwen3_profile_enabled {
                state.diagnostics.profile.argmax_calls =
                    state.diagnostics.profile.argmax_calls.saturating_add(1);
                state.diagnostics.profile.argmax_ms += argmax_ms;
            }
            steps.push(self.finish_decode_sample(state, next, true)?);
        }
        Ok(steps)
    }

    pub fn supports_resumable_prefill(&self) -> bool {
        !self.is_forced_aligner
    }

    pub fn supports_continuous_decode_batch(&self) -> bool {
        !self.is_forced_aligner
    }

    pub fn continuous_decode_is_tensor_batched(&self) -> bool {
        !self.is_forced_aligner
    }

    pub fn continuous_decode_batch_workspace_per_row_bytes(&self) -> Result<u64> {
        u64::try_from(self.text_model.hidden_size())
            .ok()
            .and_then(|hidden| {
                hidden.checked_mul(u64::try_from(self.text_dtype.size_in_bytes()).ok()?)
            })
            .ok_or_else(|| {
                Error::Overloaded(
                    "Qwen3 ASR continuous decode workspace estimate overflow".to_string(),
                )
            })
    }

    /// Forced alignment of `reference_text`, split into word units for
    /// `language` ([`alignment_units`]). Returns each unit's source text,
    /// punctuation included, with its start and end in milliseconds.
    pub fn force_align(
        &self,
        audio: &[f32],
        sample_rate: u32,
        reference_text: &str,
        language: Option<&str>,
    ) -> Result<Vec<(String, u32, u32)>> {
        let units = alignment_units(reference_text, language);
        let timings = self.force_align_units(audio, sample_rate, &units)?;
        Ok(units
            .iter()
            .zip(timings)
            .map(|(unit, (start, end))| (unit.surface(reference_text).to_string(), start, end))
            .collect())
    }

    /// Forced alignment of word units: one `(start_ms, end_ms)` per unit,
    /// the reference's values (`parse_timestamp`). Words may have
    /// `start == end`, and nothing replaces implausible timings: callers that
    /// want a fallback check the timings themselves and say so.
    pub fn force_align_units(
        &self,
        audio: &[f32],
        sample_rate: u32,
        units: &[AlignmentUnit],
    ) -> Result<Vec<(u32, u32)>> {
        if !self.is_forced_aligner {
            return Err(Error::InvalidInput(
                "Forced alignment requires a Qwen3-ForcedAligner model; Qwen3 ASR does not expose an autoregressive alignment capability"
                    .to_string(),
            ));
        }
        self.force_align_with_nar_head(audio, sample_rate, units)
    }

    fn force_align_with_nar_head(
        &self,
        audio: &[f32],
        sample_rate: u32,
        units: &[AlignmentUnit],
    ) -> Result<Vec<(u32, u32)>> {
        let audio = if sample_rate != 16_000 {
            resample(audio, sample_rate, 16_000)?
        } else {
            audio.to_vec()
        };

        if units.is_empty() {
            return Err(Error::InvalidInput(
                "Reference text produced no alignable words".to_string(),
            ));
        }
        let words = units
            .iter()
            .map(|unit| unit.text.clone())
            .collect::<Vec<_>>();
        let audio_seconds = audio.len() as f64 / f64::from(QWEN3_ASR_SAMPLE_RATE);
        let max_words = qwen3_aligner_max_words(audio_seconds);
        if words.len() > max_words {
            return Err(Error::InvalidInput(format!(
                "Forced alignment text has {} words for {audio_seconds:.1} s of audio; one window aligns at most {max_words}",
                words.len()
            )));
        }

        let (flat, frames) = qwen3_log_mel(&self.mel, &audio)?;
        qwen3_asr_admit_frames(
            frames,
            self.device.device.is_cuda(),
            true,
            self.preprocessor.nb_max_frames,
            self.mel.config().sample_rate,
            self.mel.config().hop_length,
        )?;
        if frames == 0 {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }
        let n_mels = self.mel.config().n_mels;

        let mel = Tensor::from_vec(flat, (frames, n_mels), &self.device.device)?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .to_dtype(self.audio_dtype)?;

        let feature_lens = vec![frames];
        let mut audio_embeds = self.audio_tower.forward(&mel, Some(&feature_lens))?;
        if audio_embeds.dtype() != self.text_dtype {
            audio_embeds = audio_embeds.to_dtype(self.text_dtype)?;
        }
        let audio_len = audio_embeds.dim(1)?;

        let prompt = self.build_forced_aligner_prompt(audio_len, &words)?;
        let timestamp_positions: Vec<usize> = prompt
            .ids
            .iter()
            .enumerate()
            .filter_map(|(idx, token_id)| (*token_id == prompt.timestamp_token_id).then_some(idx))
            .collect();
        if timestamp_positions.is_empty() {
            return Err(Error::InferenceError(
                "Forced aligner prompt contains no timestamp markers".to_string(),
            ));
        }

        let input_ids = Tensor::from_vec(
            prompt.ids.clone(),
            (1, prompt.ids.len()),
            &self.device.device,
        )?;

        let logits = self.timestamp_logits(
            &input_ids,
            &audio_embeds,
            prompt.audio_pad_start,
            prompt.audio_pad_len,
            &timestamp_positions,
        )?;

        let segment_time_ms = self.timestamp_segment_time_ms.ok_or_else(|| {
            Error::InvalidInput("Forced aligner model has no timestamp_segment_time".to_string())
        })?;
        let classes = timestamp_argmax(&logits, &timestamp_positions)?;
        if classes.len() != words.len() * 2 {
            return Err(Error::InferenceError(format!(
                "Forced aligner prompt has {} timestamp markers for {} words",
                classes.len(),
                words.len()
            )));
        }
        Ok(reference_word_timings(&classes, segment_time_ms))
    }

    fn build_forced_aligner_prompt(
        &self,
        audio_len: usize,
        words: &[String],
    ) -> Result<ForcedAlignerPrompt> {
        let timestamp_token_id = self
            .specials
            .timestamp
            .or(self.timestamp_token_id)
            .ok_or_else(|| {
                Error::InvalidInput(
                    "Forced aligner model is missing timestamp token metadata".to_string(),
                )
            })?;

        let mut ids = Vec::new();
        ids.push(self.specials.audio_start);
        let audio_pad_start = ids.len();
        ids.extend(std::iter::repeat_n(self.specials.audio_token, audio_len));
        ids.push(self.specials.audio_end);

        for word in words {
            ids.extend(self.tokenizer.encode_text(word)?);
            ids.push(timestamp_token_id);
            ids.push(timestamp_token_id);
        }

        Ok(ForcedAlignerPrompt {
            ids,
            audio_pad_start,
            audio_pad_len: audio_len,
            timestamp_token_id,
        })
    }

    /// Raw output for [`parse_asr_output`]. Like upstream's
    /// `batch_decode(skip_special_tokens=True)`, it keeps the non-special
    /// `<asr_text>` marker that separates `language X` from the transcript.
    fn decode_generated_untrimmed(&self, tokens: &[u32]) -> Result<String> {
        decode_raw_asr_output(&self.tokenizer, &self.specials, tokens)
    }

    /// The aligner's classifier logits at the prompt's timestamp rows only,
    /// `[1, timestamp_positions.len(), classes]` (QFA-9).
    fn timestamp_logits(
        &self,
        input_ids: &Tensor,
        audio_embeds: &Tensor,
        audio_pad_start: usize,
        audio_pad_len: usize,
        timestamp_positions: &[usize],
    ) -> Result<Tensor> {
        let (embeds, position_ids) = self.prepare_audio_prompt_embeddings(
            input_ids,
            audio_embeds,
            audio_pad_start,
            audio_pad_len,
        )?;
        self.text_model.forward_stateless_rows_with_embeds(
            &embeds,
            0,
            position_ids.as_ref(),
            timestamp_positions,
        )
    }

    fn prepare_audio_prompt_embeddings(
        &self,
        input_ids: &Tensor,
        audio_embeds: &Tensor,
        audio_pad_start: usize,
        audio_pad_len: usize,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let embeds = self.text_model.embeddings(input_ids)?;
        let seq_len = embeds.dim(1)?;
        let model_audio_len = audio_embeds.dim(1)?;
        if audio_pad_len == 0 {
            return Err(Error::InvalidInput(
                "Audio placeholder length must be at least 1".to_string(),
            ));
        }
        if model_audio_len != audio_pad_len {
            return Err(Error::InvalidInput(format!(
                "Audio placeholder mismatch: prompt has {audio_pad_len}, embeddings have {model_audio_len}"
            )));
        }

        if audio_pad_start + audio_pad_len > seq_len {
            return Err(Error::InvalidInput(
                "Audio placeholder span is out of prompt bounds".to_string(),
            ));
        }

        // Replace the contiguous <|audio_pad|> span with projected audio embeddings.
        let before = if audio_pad_start > 0 {
            embeds.narrow(1, 0, audio_pad_start)?
        } else {
            Tensor::zeros((1, 0, embeds.dim(2)?), embeds.dtype(), embeds.device())?
        };

        let after_start = audio_pad_start + audio_pad_len;
        let after = if after_start < seq_len {
            embeds.narrow(1, after_start, seq_len - after_start)?
        } else {
            Tensor::zeros((1, 0, embeds.dim(2)?), embeds.dtype(), embeds.device())?
        };

        let embeds = Tensor::cat(&[before, audio_embeds.clone(), after], 1)?;

        let position_ids = if self.text_model.uses_mrope() {
            Some(self.build_position_ids(
                embeds.dim(1)?,
                0,
                Some((audio_pad_start, audio_pad_len)),
            )?)
        } else {
            None
        };
        Ok((embeds, position_ids))
    }

    fn build_prompt(
        &self,
        audio_len: usize,
        language: Option<&str>,
        system_prompt: Option<&str>,
    ) -> Result<PromptTokens> {
        build_qwen_asr_prompt_tokens(&self.specials, audio_len, language, system_prompt, |text| {
            self.tokenizer.encode_text(text)
        })
    }

    fn build_position_ids(
        &self,
        seq_len: usize,
        start_pos: usize,
        audio_span: Option<(usize, usize)>,
    ) -> Result<Tensor> {
        let positions = build_mrope_positions(seq_len, start_pos, audio_span);

        let mut data = Vec::with_capacity(3 * seq_len);
        for _axis in 0..3 {
            data.extend_from_slice(&positions);
        }

        Tensor::from_vec(data, (3, seq_len), &self.device.device).map_err(Error::from)
    }
}

fn continue_asr_resumable_prefill(
    text_model: &Qwen3Model,
    state: &mut AsrDecodeState,
    span_start: usize,
    span_end: usize,
) -> Result<bool> {
    let prompt_tokens = state.prefill_token_count();
    if state.prefill_progress != span_start
        || span_start >= span_end
        || span_end > prompt_tokens
        || state.finished
        || state.unconsumed_output.is_some()
        || state.pending_token.is_some()
        || !state.generated_ids.is_empty()
    {
        return Err(Error::InvalidInput(format!(
            "Qwen3 ASR resumable prefill span [{span_start},{span_end}) is incompatible with cursor {} and prompt length {prompt_tokens}",
            state.prefill_progress
        )));
    }
    let physical_start = state
        .cache
        .as_ref()
        .ok_or_else(|| {
            Error::InferenceError(
                "Qwen3 ASR resumable prefill has no retained cache authority".to_string(),
            )
        })?
        .context_len();
    if state.pos != physical_start || physical_start != span_start {
        return Err(Error::InferenceError(format!(
            "Qwen3 ASR resumable prefill physical cursor {physical_start} is incompatible with logical span [{span_start},{span_end})"
        )));
    }

    let prepared = state.prepared_prompt_embeddings.as_ref().ok_or_else(|| {
        Error::InferenceError(
            "Qwen3 ASR resumable prefill has no immutable prompt embeddings".to_string(),
        )
    })?;
    let (batch, prepared_tokens, _hidden) = prepared.dims3()?;
    if batch != 1 || prepared_tokens != prompt_tokens {
        return Err(Error::InferenceError(format!(
            "Qwen3 ASR prepared prompt shape {:?} does not match {prompt_tokens} admitted tokens",
            prepared.dims()
        )));
    }
    let span_len = span_end - span_start;
    let span_embeddings = prepared.narrow(1, span_start, span_len)?;
    let span_position_ids = state
        .prepared_position_ids
        .as_ref()
        .map(|positions| positions.narrow(1, span_start, span_len))
        .transpose()?;
    let complete = span_end == prompt_tokens;
    let prefill_started = Instant::now();
    let output = {
        let cache = state.cache.as_mut().expect("validated retained ASR cache");
        if complete {
            Some(text_model.forward_managed_last_logits_with_embeds(
                &span_embeddings,
                span_start,
                cache,
                span_position_ids.as_ref(),
            )?)
        } else {
            text_model.forward_managed_prefill_only_with_embeds(
                &span_embeddings,
                span_start,
                cache,
                span_position_ids.as_ref(),
            )?;
            None
        }
    };
    let physical_end = state
        .cache
        .as_ref()
        .expect("validated retained ASR cache")
        .context_len();
    if physical_end != span_end {
        return Err(Error::InferenceError(format!(
            "Qwen3 ASR resumable prefill committed physical cursor {physical_end} instead of {span_end}"
        )));
    }

    let prefill_ms = elapsed_ms(prefill_started);
    state.diagnostics.timings.prefill_ms += prefill_ms;
    state.diagnostics.timings.total_ms += prefill_ms;
    state.pos = span_end;
    state.prefill_progress = span_end;
    state.unconsumed_output = output;
    if complete {
        state.prepared_prompt_embeddings = None;
        state.prepared_position_ids = None;
    }
    Ok(complete)
}

fn complete_asr_invocation_prefill(
    text_model: &Qwen3Model,
    state: &mut AsrDecodeState,
    cache: &mut PhysicalPagedKvCache,
) -> Result<()> {
    let prompt_tokens = state.prefill_token_count();
    if cache.context_len() != 0
        || state.cache.is_some()
        || state.prefill_progress != 0
        || state.pos != 0
        || state.unconsumed_output.is_some()
    {
        return Err(Error::InferenceError(
            "Qwen3 ASR invocation prefill must start from empty state".to_string(),
        ));
    }
    let prepared = state.prepared_prompt_embeddings.as_ref().ok_or_else(|| {
        Error::InferenceError("Qwen3 ASR invocation has no prepared prompt".to_string())
    })?;
    let position_ids = state.prepared_position_ids.as_ref();
    let prefill_started = Instant::now();
    let output =
        text_model.forward_managed_last_logits_with_embeds(prepared, 0, cache, position_ids)?;
    if cache.context_len() != prompt_tokens {
        return Err(Error::InferenceError(format!(
            "Qwen3 ASR invocation prefill committed physical cursor {} instead of {prompt_tokens}",
            cache.context_len()
        )));
    }
    let prefill_ms = elapsed_ms(prefill_started);
    state.diagnostics.timings.prefill_ms += prefill_ms;
    state.diagnostics.timings.total_ms += prefill_ms;
    state.pos = prompt_tokens;
    state.prefill_progress = prompt_tokens;
    state.unconsumed_output = Some(output);
    state.prepared_prompt_embeddings = None;
    state.prepared_position_ids = None;
    Ok(())
}

fn forward_asr_pending_decode_batch(
    text_model: &Qwen3Model,
    device: &candle_core::Device,
    states: &mut [&mut AsrDecodeState],
) -> Result<Tensor> {
    if states.is_empty() {
        return Err(Error::InvalidInput(
            "Qwen3 ASR decode batch must contain at least one row".to_string(),
        ));
    }
    let pending_tokens = states
        .iter()
        .map(|state| {
            state.pending_token.ok_or_else(|| {
                Error::InferenceError(
                    "continuous Qwen3 ASR row has no scheduled input token".to_string(),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let positions = states
        .iter()
        .map(|state| {
            state.pos.checked_add(1).ok_or_else(|| {
                Error::InferenceError("Qwen3 ASR decode position overflow".to_string())
            })?;
            Ok(state.pos)
        })
        .collect::<Result<Vec<_>>>()?;
    if states.iter().any(|state| state.cache.is_none()) {
        return Err(Error::InferenceError(
            "continuous Qwen3 ASR row has no retained cache authority".to_string(),
        ));
    }
    let input_ids = Tensor::from_vec(pending_tokens, (states.len(), 1), device)?;
    let output = if states.len() == 1 {
        let state = &mut *states[0];
        text_model.forward_managed(
            &input_ids,
            positions[0],
            state.cache.as_mut().expect("validated retained ASR cache"),
        )?
    } else {
        let mut caches = states
            .iter_mut()
            .map(|state| state.cache.as_mut().expect("validated retained ASR cache"))
            .collect::<Vec<_>>();
        text_model.forward_managed_decode_batch(&input_ids, &positions, &mut caches)?
    };
    for state in states.iter_mut() {
        state.pending_token = None;
        state.pos += 1;
    }
    Ok(output)
}

fn batched_decode_argmax(output: &Tensor) -> Result<Vec<u32>> {
    let (batch, sequence, _width) = output.dims3()?;
    if batch == 0 || sequence == 0 {
        return Err(Error::InferenceError(format!(
            "Qwen3 ASR batched decode output expects [batch,sequence,width], got {:?}",
            output.dims()
        )));
    }
    output
        .narrow(1, sequence - 1, 1)?
        .squeeze(1)?
        .argmax(D::Minus1)?
        .to_dtype(DType::U32)?
        .to_vec1::<u32>()
        .map_err(Error::from)
}

/// The thinker dtype hint a GGUF checkpoint implies: the published GGUFs are
/// converted from BF16 weights and carry no dtype of their own.
pub(crate) const QWEN3_ASR_GGUF_CHECKPOINT_DTYPE: &str = "bf16";

pub(crate) fn resolve_qwen_asr_gguf_path(
    model_dir: &Path,
    variant: ModelVariant,
) -> Option<PathBuf> {
    if let Some(filename) = qwen_asr_gguf_filename(variant) {
        let candidate = model_dir.join(filename);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    [
        model_dir.join("qwen3_asr_0.6b_q8_0.gguf"),
        model_dir.join("qwen3_asr_1.7b_q8_0.gguf"),
    ]
    .into_iter()
    .find(|candidate| candidate.exists())
}

fn qwen_asr_gguf_filename(variant: ModelVariant) -> Option<&'static str> {
    match variant {
        ModelVariant::Qwen3Asr06BGguf => Some("qwen3_asr_0.6b_q8_0.gguf"),
        ModelVariant::Qwen3Asr17BGguf => Some("qwen3_asr_1.7b_q8_0.gguf"),
        _ => None,
    }
}

fn parse_qwen3_asr_config_from_gguf(
    loader: &GgufLoader,
    model_dir: &Path,
    variant: ModelVariant,
) -> Result<Qwen3AsrConfig> {
    let rope_theta = required_f64_metadata(loader, "qwen3_asr.text.rope_theta")?;
    let mrope_section = required_usize_array_metadata(loader, "qwen3_asr.text.mrope_section")?;

    let text_config = Qwen3Config {
        hidden_size: required_usize_metadata(loader, "qwen3_asr.text.hidden_size")?,
        intermediate_size: required_usize_metadata(loader, "qwen3_asr.text.intermediate_size")?,
        num_attention_heads: required_usize_metadata(loader, "qwen3_asr.text.num_attention_heads")?,
        num_hidden_layers: required_usize_metadata(loader, "qwen3_asr.text.num_hidden_layers")?,
        num_key_value_heads: required_usize_metadata(loader, "qwen3_asr.text.num_key_value_heads")?,
        max_position_embeddings: Some(resolve_qwen3_asr_context(
            optional_usize_metadata(loader, "qwen3_asr.text.context_length")
                .or_else(|| optional_usize_metadata(loader, "qwen3_asr.context_length")),
            qwen3_asr_sidecar_context(model_dir)?,
            variant,
        )?),
        head_dim: Some(required_usize_metadata(loader, "qwen3_asr.text.head_dim")?),
        rms_norm_eps: required_f64_metadata(loader, "qwen3_asr.text.rms_norm_eps")?,
        rope_theta,
        vocab_size: required_usize_metadata(loader, "qwen3_asr.text.vocab_size")?,
        lm_head_size: None,
        // The published ASR checkpoints tie the head to the embedding (and
        // store it twice, byte for byte); the loader keeps one copy.
        tie_word_embeddings: loader
            .get_metadata_bool("qwen3_asr.text.tie_word_embeddings")
            .unwrap_or(false),
        rope_scaling: Some(RopeScalingConfig {
            rope_type: Some("mrope".to_string()),
            rope_theta: Some(rope_theta),
            mrope_interleaved: Some(true),
            interleaved: Some(true),
            mrope_section: Some(mrope_section),
        }),
        sliding_window: None,
        use_sliding_window: false,
        ada_rms_norm_t_cond: false,
        ada_rms_norm_t_cond_dim: 0,
        num_experts: None,
        num_experts_per_tok: None,
        moe_intermediate_size: None,
        norm_topk_prob: None,
    };

    let audio_config = config::AudioConfig {
        d_model: required_usize_metadata(loader, "qwen3_asr.audio.d_model")?,
        encoder_attention_heads: required_usize_metadata(
            loader,
            "qwen3_asr.audio.encoder_attention_heads",
        )?,
        encoder_ffn_dim: required_usize_metadata(loader, "qwen3_asr.audio.encoder_ffn_dim")?,
        encoder_layers: required_usize_metadata(loader, "qwen3_asr.audio.encoder_layers")?,
        num_mel_bins: required_usize_metadata(loader, "qwen3_asr.audio.num_mel_bins")?,
        downsample_hidden_size: infer_downsample_hidden_size(loader)?,
        output_dim: required_usize_metadata(loader, "qwen3_asr.audio.output_dim")?,
        conv_chunksize: optional_usize_metadata(loader, "qwen3_asr.audio.conv_chunksize"),
        n_window: optional_usize_metadata(loader, "qwen3_asr.audio.n_window"),
        n_window_infer: optional_usize_metadata(loader, "qwen3_asr.audio.n_window_infer"),
    };

    Ok(Qwen3AsrConfig {
        model_type: loader.get_metadata_string("general.architecture"),
        quantization: None,
        quantization_config: None,
        thinker_config: config::ThinkerConfig {
            model_type: loader.get_metadata_string("general.architecture"),
            classify_num: None,
            audio_config,
            text_config,
            audio_start_token_id: optional_u32_metadata(loader, "qwen3_asr.audio_start_token_id"),
            audio_end_token_id: optional_u32_metadata(loader, "qwen3_asr.audio_end_token_id"),
            audio_token_id: optional_u32_metadata(loader, "qwen3_asr.audio_token_id"),
            dtype: Some(QWEN3_ASR_GGUF_CHECKPOINT_DTYPE.to_string()),
        },
        support_languages: Vec::new(),
        timestamp_token_id: None,
        timestamp_segment_time: None,
    })
}

fn qwen3_asr_sidecar_context(model_dir: &Path) -> Result<Option<usize>> {
    let path = model_dir.join("config.json");
    if !path.exists() {
        return Ok(None);
    }
    let data = fs::read_to_string(&path).map_err(|error| {
        Error::ModelLoadError(format!(
            "failed to read Qwen3 ASR sidecar {}: {error}",
            path.display()
        ))
    })?;
    let value: serde_json::Value = serde_json::from_str(&data).map_err(|error| {
        Error::ModelLoadError(format!(
            "failed to parse Qwen3 ASR sidecar {}: {error}",
            path.display()
        ))
    })?;
    Ok(value
        .pointer("/thinker_config/text_config/max_position_embeddings")
        .or_else(|| value.pointer("/text_config/max_position_embeddings"))
        .or_else(|| value.pointer("/max_position_embeddings"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|tokens| usize::try_from(tokens).ok()))
}

fn resolve_qwen3_asr_context(
    embedded: Option<usize>,
    sidecar: Option<usize>,
    variant: ModelVariant,
) -> Result<usize> {
    let manifest = variant.artifact_context_manifest();
    let manifest_tokens = manifest.map(|entry| entry.native_context_tokens);
    let selected = embedded.or(sidecar).or(manifest_tokens).ok_or_else(|| {
        Error::ModelLoadError(
            "Qwen3 ASR text context is absent from GGUF metadata, config.json, and the variant artifact manifest"
                .to_string(),
        )
    })?;
    if selected == 0 {
        return Err(Error::ModelLoadError(
            "Qwen3 ASR text context must be positive".to_string(),
        ));
    }
    for (source, candidate) in [("GGUF metadata", embedded), ("config.json", sidecar)] {
        if let Some(candidate) = candidate {
            if candidate != selected {
                return Err(Error::ModelLoadError(format!(
                    "Qwen3 ASR context conflict: {source} declares {candidate} tokens but another authoritative source declares {selected}"
                )));
            }
        }
    }
    if let Some(manifest) = manifest {
        if manifest.native_context_tokens != selected {
            return Err(Error::ModelLoadError(format!(
                "Qwen3 ASR context conflict: artifact manifest {} declares {} tokens but artifact metadata declares {selected}",
                manifest.evidence, manifest.native_context_tokens
            )));
        }
    }
    Ok(selected)
}

fn qwen3_asr_gguf_tensor_prefix(loader: &GgufLoader) -> &str {
    if loader.has_tensor("thinker.model.embed_tokens.weight")
        || loader.has_tensor("thinker.audio_tower.conv2d1.weight")
    {
        "thinker"
    } else {
        ""
    }
}

fn qwen3_asr_gguf_audio_tensor_matches(name: &str, prefix: &str) -> bool {
    if prefix.is_empty() {
        name.starts_with("audio_tower.")
    } else {
        name.starts_with(&format!("{prefix}.audio_tower."))
    }
}

fn qwen3_asr_gguf_text_tensor_matches(name: &str, prefix: &str) -> bool {
    if prefix.is_empty() {
        name.starts_with("model.") || name == "lm_head.weight"
    } else {
        name.starts_with(&format!("{prefix}.model.")) || name == format!("{prefix}.lm_head.weight")
    }
}

fn qwen3_asr_gguf_qmatmul_text_policy(override_enabled: Option<bool>) -> bool {
    override_enabled.unwrap_or(true)
}

pub(crate) fn qwen3_asr_use_gguf_qmatmul_text(_device: &candle_core::Device) -> bool {
    let override_enabled = crate::env::flag("IZWI_QWEN3_ASR_GGUF_QMATMUL_TEXT");
    qwen3_asr_gguf_qmatmul_text_policy(override_enabled)
}

fn preprocessor_config_from_gguf(loader: &GgufLoader) -> Result<PreprocessorConfig> {
    let feature_size = required_usize_metadata(loader, "qwen3_asr.audio.num_mel_bins")?;
    let n_fft = optional_usize_metadata(loader, "qwen3_asr.audio.n_fft").unwrap_or(400);
    let hop_length = optional_usize_metadata(loader, "qwen3_asr.audio.hop_length").unwrap_or(160);
    let nb_max_frames = optional_usize_metadata(loader, "qwen3_asr.audio.max_source_positions")
        .map(|value| value.saturating_mul(8))
        .unwrap_or(0);
    let n_samples = if nb_max_frames > 0 {
        nb_max_frames.saturating_mul(hop_length)
    } else {
        0
    };

    Ok(PreprocessorConfig {
        feature_size,
        n_fft,
        hop_length,
        n_samples,
        nb_max_frames,
    })
}

fn infer_downsample_hidden_size(loader: &GgufLoader) -> Result<usize> {
    for tensor_name in [
        "thinker.audio_tower.conv2d1.weight",
        "audio_tower.conv2d1.weight",
    ] {
        if let Some(shape) = loader.tensor_shape(tensor_name) {
            if let Some(channels) = shape.first().copied() {
                return Ok(channels);
            }
        }
    }

    Err(Error::ModelLoadError(
        "Unable to infer qwen3_asr.audio.downsample_hidden_size from GGUF tensors".to_string(),
    ))
}

/// The audio tower and the text model, each read at its selected dtype by
/// `load_audio` / `load_text`. With `check` (Metal), a dtype below F32 must
/// pass that component's load-time probe, or the component alone reloads in
/// F32 (report 13), for GGUF and safetensors checkpoints alike. Admission
/// prices the selected dtypes, so a reload asks the load's
/// [`admit_precision_reload`] gate for its memory
/// (`load_with_precision_fallback` does), which first synchronizes the device
/// so Candle frees the dropped component's buffers.
fn load_qwen3_asr_components(
    audio_selection: &DTypeSelection,
    text_selection: &DTypeSelection,
    check: bool,
    device: &candle_core::Device,
    mut load_audio: impl FnMut(DType) -> Result<AudioTower>,
    mut load_text: impl FnMut(DType) -> Result<Qwen3Model>,
) -> Result<(AudioTower, PrecisionLoad, Qwen3Model, PrecisionLoad)> {
    let (audio_tower, audio_precision) = load_with_precision_fallback(
        "Qwen3-ASR audio tower",
        audio_selection,
        check,
        DType::F32,
        &mut load_audio,
        |tower| qwen3_asr_audio_self_check(tower, audio_selection.dtype, device),
    )?;
    let (text_model, text_precision) = load_with_precision_fallback(
        "Qwen3-ASR text",
        text_selection,
        check,
        DType::F32,
        &mut load_text,
        qwen3_asr_text_self_check,
    )?;
    Ok((audio_tower, audio_precision, text_model, text_precision))
}

fn resolve_safetensors_shards(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index_data = fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_data)?;
        let weight_map = index
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| {
                Error::InvalidInput("Invalid model.safetensors.index.json format".to_string())
            })?;

        let mut shard_files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        shard_files.sort();
        shard_files.dedup();
        let shard_paths: Vec<PathBuf> = shard_files.iter().map(|f| model_dir.join(f)).collect();
        info!(
            "Loading sharded ASR model with {} shard files",
            shard_paths.len()
        );
        return Ok(shard_paths);
    }

    let weights_path = model_dir.join("model.safetensors");
    if !weights_path.exists() {
        return Err(Error::ModelNotFound(format!(
            "Missing ASR checkpoint file: {}",
            weights_path.display()
        )));
    }
    Ok(vec![weights_path])
}

/// How [`Qwen3AsrModel::load`] reads a safetensors checkpoint: the shards,
/// the text and audio-tower dtypes it resolves on the device, and the bit width
/// of MLX-packed weights. Load admission prices the checkpoint with it
/// (`runtime/lifecycle/asr_memory.rs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Qwen3AsrSafetensorsLoad {
    pub(crate) shards: Vec<PathBuf>,
    pub(crate) text_dtype: DType,
    pub(crate) audio_dtype: DType,
    /// Bits per packed element of an MLX-quantized checkpoint, as its config
    /// names them.
    pub(crate) mlx_bits: Option<u64>,
}

/// [`Qwen3AsrSafetensorsLoad`] for the checkpoint in `model_dir` on `device`,
/// or `None` when the loader would read a GGUF checkpoint instead or the
/// config or weights are missing (the loader reports those).
pub(crate) fn qwen3_asr_safetensors_load(
    model_dir: &Path,
    variant: ModelVariant,
    device: &DeviceProfile,
) -> Result<Option<Qwen3AsrSafetensorsLoad>> {
    let config_path = model_dir.join("config.json");
    if resolve_qwen_asr_gguf_path(model_dir, variant).is_some() || !config_path.is_file() {
        return Ok(None);
    }
    let config: Qwen3AsrConfig = serde_json::from_str(&fs::read_to_string(config_path)?)?;
    let shards = match resolve_safetensors_shards(model_dir) {
        Ok(shards) => shards,
        Err(Error::ModelNotFound(_)) => return Ok(None),
        Err(error) => return Err(error),
    };
    let quantized = config.is_mlx_quantized();
    let (text_dtype, audio_dtype) = qwen3_asr_weight_dtypes(
        device,
        config.model_family(),
        false,
        quantized,
        config.thinker_config.dtype.as_deref(),
    )?;
    Ok(Some(Qwen3AsrSafetensorsLoad {
        shards,
        text_dtype,
        audio_dtype,
        mlx_bits: if quantized { config.mlx_bits() } else { None },
    }))
}

/// The text and audio-tower weight dtypes the Qwen3-ASR loader selects on
/// `device`, honouring `IZWI_QWEN_ASR_TEXT_DTYPE` (or `IZWI_QWEN_DTYPE`) and
/// `IZWI_QWEN_ASR_AUDIO_DTYPE`. `checkpoint_dtype` is the thinker config's dtype
/// hint. Load admission prices the checkpoint with the same selection
/// (`runtime/lifecycle/asr_memory.rs`).
pub(crate) fn qwen3_asr_weight_dtypes(
    device: &DeviceProfile,
    model_family: ModelFamily,
    is_gguf: bool,
    is_quantized: bool,
    checkpoint_dtype: Option<&str>,
) -> Result<(DType, DType)> {
    qwen3_asr_weight_precisions(
        device,
        model_family,
        is_gguf,
        is_quantized,
        checkpoint_dtype,
    )
    .map(|(text, audio)| (text.dtype, audio.dtype))
}

/// [`qwen3_asr_weight_dtypes`] with the reason for each choice.
fn qwen3_asr_weight_precisions(
    device: &DeviceProfile,
    model_family: ModelFamily,
    is_gguf: bool,
    is_quantized: bool,
    checkpoint_dtype: Option<&str>,
) -> Result<(DTypeSelection, DTypeSelection)> {
    let text_dtype_override = std::env::var("IZWI_QWEN_ASR_TEXT_DTYPE")
        .ok()
        .or_else(|| std::env::var("IZWI_QWEN_DTYPE").ok())
        .as_deref()
        .map(str::trim)
        .filter(|raw| !raw.is_empty())
        .map(str::to_string);
    let text = if let Some(raw) = text_dtype_override.as_deref() {
        DTypeSelection {
            dtype: device.select_model_dtype_checked(model_family, Some(raw), "Qwen ASR text")?,
            reason: format!("text dtype override {raw:?}").into(),
        }
    } else if is_quantized {
        let requested = parse_asr_dtype(checkpoint_dtype).unwrap_or(DType::BF16);
        let selected = select_quantized_text_dtype(device, model_family, requested);
        debug!(
            "Qwen speech-family quantized dtype selection: requested={:?}, selected={:?} on {:?}",
            requested, selected.dtype, device.kind
        );
        selected
    } else if device.kind.is_metal()
        && model_family == ModelFamily::Qwen3ForcedAligner
        && MetalDTypeOverride::from_env() != MetalDTypeOverride::F32
    {
        // D8: the aligner reads one 80 ms timestamp class per boundary and
        // never decodes, so F16 text moved no boundary on test.wav or a 175 s
        // cut against F32 (Metal, 2026-10-15) and halves its residency. The
        // load-time probe reloads the text at F32 if F16 fails.
        DTypeSelection {
            dtype: DType::F16,
            reason: "Metal runs dense aligner text at F16 behind the load-time probe".into(),
        }
    } else if device.kind.is_metal() {
        // This Qwen speech-family text decode path can drift badly on Metal in fp16 on longer
        // utterances. Prefer f32 unless the user explicitly overrides it.
        DTypeSelection {
            dtype: DType::F32,
            reason: "Metal keeps dense Qwen ASR text in F32".into(),
        }
    } else {
        DTypeSelection {
            dtype: device.select_model_dtype(model_family, None),
            reason: format!("{:?} default text dtype", device.kind).into(),
        }
    };
    let text_dtype = text.dtype;
    let audio_dtype_override = std::env::var("IZWI_QWEN_ASR_AUDIO_DTYPE").ok();
    let audio = match audio_dtype_override.as_deref().map(str::trim) {
        Some(raw) if !raw.is_empty() => {
            let requested =
                device.select_model_dtype_checked(model_family, Some(raw), "Qwen ASR audio")?;
            if is_gguf && requested != text_dtype {
                warn!(
                    "Qwen ASR GGUF requested audio dtype {:?} differs from text dtype {:?}; loading a second GGUF VarBuilder",
                    requested, text_dtype
                );
            }
            DTypeSelection {
                dtype: requested,
                reason: format!("audio dtype override {raw:?}").into(),
            }
        }
        _ if device.kind.is_metal()
            && model_family == ModelFamily::Qwen3ForcedAligner
            && MetalDTypeOverride::from_env() != MetalDTypeOverride::F32 =>
        {
            // The aligner's audio tower only feeds boundary classes (D8):
            // F16 moved no word boundary over 5,278 words (fox, test,
            // diarization, a 5,128-word ceo.mp3; dense and 4-bit, Metal,
            // 2026-10-19) and halves its residency. The load-time probe
            // reloads the tower at F32 if F16 fails.
            DTypeSelection {
                dtype: DType::F16,
                reason: "Metal runs the aligner's audio tower at F16 behind the load-time probe"
                    .into(),
            }
        }
        _ => DTypeSelection {
            dtype: qwen3_asr_default_audio_dtype(device.kind, is_gguf, text_dtype),
            reason: if is_gguf && !device.kind.is_cuda() {
                "the GGUF audio tower runs at the text dtype; it falls back to F32 alone if its probe fails".into()
            } else {
                format!("{:?} default audio dtype", device.kind).into()
            },
        },
    };
    Ok((text, audio))
}

/// Activations for quantized Qwen ASR text weights. Metal takes the
/// BF16-checkpoint LM tier (F16 behind the load-time probe; BF16 on Apple9+
/// once Metal paged attention takes BF16) unless `IZWI_METAL_DTYPE=f32`.
fn select_quantized_text_dtype(
    device: &DeviceProfile,
    model_family: ModelFamily,
    requested: DType,
) -> DTypeSelection {
    match device.kind {
        DeviceKind::Metal if MetalDTypeOverride::from_env() == MetalDTypeOverride::F32 => {
            DTypeSelection {
                dtype: DType::F32,
                reason: "IZWI_METAL_DTYPE=f32 keeps quantized-weight activations in F32".into(),
            }
        }
        DeviceKind::Metal => device.metal_bf16_checkpoint_lm_tier(),
        DeviceKind::Cpu => DTypeSelection {
            dtype: DType::F32,
            reason: "CPU runs quantized-weight activations in F32".into(),
        },
        DeviceKind::Cuda => device.resolve_dtype(
            DTypeSelectionRequest::new(None)
                .with_model_family(model_family)
                .with_checkpoint_dtype(requested)
                .with_quantized(true),
        ),
    }
}

/// Feature frames the audio tower's load-time probe runs (two of its CNN
/// chunks at the default window).
const QWEN3_ASR_AUDIO_PROBE_FRAMES: usize = 200;
/// Token positions the text model's load-time probe runs.
const QWEN3_ASR_TEXT_PROBE_TOKENS: usize = 16;

/// The audio tower's load-time probe: xorshift log-mel features through the
/// tower in its dtype; the output must be finite and in the F16-safe range.
fn qwen3_asr_audio_self_check(
    tower: &AudioTower,
    dtype: DType,
    device: &candle_core::Device,
) -> NumericSelfCheckReport {
    run_numeric_self_check(
        format!("Qwen3-ASR audio tower {dtype:?}"),
        F16_SAFE_MAX_ABS,
        || {
            let n_mels = tower.mel_bins();
            let features = xorshift_tensor(
                &[1, QWEN3_ASR_AUDIO_PROBE_FRAMES, n_mels],
                0x0A5D,
                1.5,
                device,
            )?
            .to_dtype(dtype)?;
            let output = tower
                .forward_feature_batch(&features, &[QWEN3_ASR_AUDIO_PROBE_FRAMES])?
                .into_padded();
            Ok::<_, Error>(vec![("audio_tower".to_string(), output)])
        },
    )
}

/// The text model's load-time probe: xorshift-chosen token ids through every
/// layer and the head (stateless attention); the logits must be finite and in
/// the F16-safe range.
fn qwen3_asr_text_self_check(model: &Qwen3Model) -> NumericSelfCheckReport {
    run_numeric_self_check("Qwen3-ASR text", F16_SAFE_MAX_ABS, || {
        model.stateless_numeric_probe(QWEN3_ASR_TEXT_PROBE_TOKENS, 0x7E47)
    })
}

fn qwen3_asr_default_audio_dtype(
    device_kind: DeviceKind,
    is_gguf: bool,
    text_dtype: DType,
) -> DType {
    if is_gguf && device_kind.is_cuda() {
        return DType::F32;
    }
    if is_gguf {
        return text_dtype;
    }
    DType::F32
}

fn infer_lm_head_size_from_gguf(
    gguf_path: &Path,
    tensor_name: Option<&str>,
) -> Result<Option<usize>> {
    let tensor_name = tensor_name.unwrap_or("lm_head.weight");
    let loader = GgufLoader::from_path(gguf_path)?;
    Ok(loader
        .tensor_shape(tensor_name)
        .and_then(|shape| shape.first().copied()))
}

fn required_usize_metadata(loader: &GgufLoader, key: &str) -> Result<usize> {
    loader
        .metadata_value(key)
        .and_then(gguf_to_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))
}

fn optional_usize_metadata(loader: &GgufLoader, key: &str) -> Option<usize> {
    loader
        .metadata_value(key)
        .and_then(gguf_to_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn optional_u32_metadata(loader: &GgufLoader, key: &str) -> Option<u32> {
    loader
        .metadata_value(key)
        .and_then(gguf_to_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn required_f64_metadata(loader: &GgufLoader, key: &str) -> Result<f64> {
    loader
        .metadata_value(key)
        .and_then(gguf_to_f64)
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))
}

fn required_usize_array_metadata(loader: &GgufLoader, key: &str) -> Result<Vec<usize>> {
    let value = loader
        .metadata_value(key)
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))?;
    let GgufValue::Array(values) = value else {
        return Err(Error::ModelLoadError(format!(
            "Expected GGUF metadata array for {key}"
        )));
    };

    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let raw = gguf_to_u64(value).ok_or_else(|| {
            Error::ModelLoadError(format!("Expected integer metadata values for {key}"))
        })?;
        let parsed = usize::try_from(raw).map_err(|_| {
            Error::ModelLoadError(format!("GGUF metadata value out of range for {key}: {raw}"))
        })?;
        out.push(parsed);
    }
    Ok(out)
}

fn gguf_to_u64(value: &GgufValue) -> Option<u64> {
    match value {
        GgufValue::U64(v) => Some(*v),
        GgufValue::I64(v) => Some(*v as u64),
        GgufValue::U32(v) => Some(*v as u64),
        GgufValue::I32(v) => Some(*v as u64),
        GgufValue::U16(v) => Some(*v as u64),
        GgufValue::I16(v) => Some(*v as u64),
        GgufValue::U8(v) => Some(*v as u64),
        GgufValue::I8(v) => Some(*v as u64),
        _ => None,
    }
}

fn gguf_to_f64(value: &GgufValue) -> Option<f64> {
    match value {
        GgufValue::F64(v) => Some(*v),
        GgufValue::F32(v) => Some(*v as f64),
        GgufValue::U64(v) => Some(*v as f64),
        GgufValue::I64(v) => Some(*v as f64),
        GgufValue::U32(v) => Some(*v as f64),
        GgufValue::I32(v) => Some(*v as f64),
        GgufValue::U16(v) => Some(*v as f64),
        GgufValue::I16(v) => Some(*v as f64),
        GgufValue::U8(v) => Some(*v as f64),
        GgufValue::I8(v) => Some(*v as f64),
        _ => None,
    }
}

const MAX_SAFE_TENSORS_HEADER_SIZE: usize = 100_000_000;

fn infer_lm_head_size_from_checkpoint(
    model_dir: &Path,
    tensor_name: Option<&str>,
) -> Result<Option<usize>> {
    let tensor_name = tensor_name.unwrap_or("lm_head.weight");
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index_data = fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_data)?;
        let weight_map = index
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| {
                Error::InvalidInput("Invalid model.safetensors.index.json format".to_string())
            })?;
        let Some(shard_name) = weight_map.get(tensor_name).and_then(|v| v.as_str()) else {
            return Ok(None);
        };
        let shard_path = model_dir.join(shard_name);
        return infer_tensor_first_dim_from_safetensors(&shard_path, tensor_name);
    }

    let weights_path = model_dir.join("model.safetensors");
    if !weights_path.exists() {
        return Ok(None);
    }
    infer_tensor_first_dim_from_safetensors(&weights_path, tensor_name)
}

fn infer_tensor_first_dim_from_safetensors(
    safetensors_path: &Path,
    tensor_name: &str,
) -> Result<Option<usize>> {
    let shape = match tensor_shape_from_safetensors_header(safetensors_path, tensor_name)? {
        Some(shape) => shape,
        None => return Ok(None),
    };
    Ok(shape.first().copied())
}

fn tensor_shape_from_safetensors_header(
    safetensors_path: &Path,
    tensor_name: &str,
) -> Result<Option<Vec<usize>>> {
    let mut file = fs::File::open(safetensors_path)?;

    let mut n_buf = [0u8; 8];
    file.read_exact(&mut n_buf)?;
    let header_len_u64 = u64::from_le_bytes(n_buf);
    let header_len: usize = header_len_u64
        .try_into()
        .map_err(|_| Error::InvalidInput("Invalid safetensors header length".to_string()))?;
    if header_len > MAX_SAFE_TENSORS_HEADER_SIZE {
        return Err(Error::InvalidInput(format!(
            "Safetensors header too large: {header_len}"
        )));
    }

    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;

    let metadata: serde_json::Value = serde_json::from_slice(&header_buf)?;
    let Some(tensor_entry) = metadata.get(tensor_name) else {
        return Ok(None);
    };
    let Some(shape) = tensor_entry.get("shape").and_then(|shape| shape.as_array()) else {
        return Ok(None);
    };

    let dims = shape
        .iter()
        .map(|dim| dim.as_u64().map(|value| value as usize))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "Invalid shape metadata for tensor {tensor_name} in {}",
                safetensors_path.display()
            ))
        })?;
    Ok(Some(dims))
}

/// The forced aligner's timestamp token and the milliseconds one timestamp
/// class stands for. The reference reads both from the config
/// (`Qwen3ForcedAligner.__init__`), so an aligner checkpoint without them is
/// refused instead of timed with a guessed class length. `(None, None)` for
/// Qwen3-ASR.
fn forced_aligner_timestamp_config(
    is_forced_aligner: bool,
    timestamp_token_id: Option<u32>,
    timestamp_segment_time: Option<usize>,
) -> Result<(Option<u32>, Option<u32>)> {
    if !is_forced_aligner {
        return Ok((None, None));
    }
    let token = timestamp_token_id.ok_or_else(|| {
        Error::ModelLoadError("Forced aligner config has no timestamp_token_id".to_string())
    })?;
    let segment_ms = timestamp_segment_time
        .filter(|ms| *ms > 0)
        .and_then(|ms| u32::try_from(ms).ok())
        .ok_or_else(|| {
            Error::ModelLoadError(
                "Forced aligner config has no positive timestamp_segment_time".to_string(),
            )
        })?;
    Ok((Some(token), Some(segment_ms)))
}

/// The reference `parse_timestamp`: each timestamp class times the segment
/// time, repaired by `fix_timestamp`, as one `(start_ms, end_ms)` per word
/// (two classes per word). The values are returned as they are: a word may
/// have `start == end`.
fn reference_word_timings(classes: &[u32], segment_time_ms: u32) -> Vec<(u32, u32)> {
    let timestamps = classes
        .iter()
        .map(|class| f64::from(*class) * f64::from(segment_time_ms))
        .collect::<Vec<_>>();
    fix_timestamp_sequence(&timestamps)
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[start, end]| (start, end))
        .collect()
}

/// The reference `fix_timestamp`: keep the longest non-decreasing
/// subsequence and repair the rest from its neighbours, in Python float
/// arithmetic, truncating like `int()`.
fn fix_timestamp_sequence(data: &[f64]) -> Vec<u32> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }

    let mut dp = vec![1usize; n];
    let mut parent = vec![usize::MAX; n];
    for i in 1..n {
        for j in 0..i {
            if data[j] <= data[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
        }
    }

    let mut max_idx = 0usize;
    for i in 1..n {
        if dp[i] > dp[max_idx] {
            max_idx = i;
        }
    }

    let mut lis_indices = Vec::new();
    let mut idx = max_idx;
    while idx != usize::MAX {
        lis_indices.push(idx);
        idx = parent[idx];
    }
    lis_indices.reverse();

    let mut is_normal = vec![false; n];
    for idx in lis_indices {
        is_normal[idx] = true;
    }

    let mut result = data.to_vec();
    let mut i = 0usize;
    while i < n {
        if is_normal[i] {
            i += 1;
            continue;
        }

        let mut j = i;
        while j < n && !is_normal[j] {
            j += 1;
        }
        let anomaly_count = j - i;

        if anomaly_count <= 2 {
            let left_val = (0..i).rev().find(|&k| is_normal[k]).map(|k| result[k]);
            let right_val = (j..n).find(|&k| is_normal[k]).map(|k| result[k]);

            for (offset, value) in result[i..j].iter_mut().enumerate() {
                let k = i + offset;
                *value = match (left_val, right_val) {
                    (None, Some(right)) => right,
                    (Some(left), None) => left,
                    (Some(left), Some(right)) => {
                        let left_distance = k as isize - (i as isize - 1);
                        let right_distance = j as isize - k as isize;
                        if left_distance <= right_distance {
                            left
                        } else {
                            right
                        }
                    }
                    (None, None) => *value,
                };
            }
        } else {
            let left_val = (0..i).rev().find(|&k| is_normal[k]).map(|k| result[k]);
            let right_val = (j..n).find(|&k| is_normal[k]).map(|k| result[k]);

            match (left_val, right_val) {
                (Some(left), Some(right)) => {
                    let step = (right - left) / (anomaly_count as f64 + 1.0);
                    for (offset, value) in result[i..j].iter_mut().enumerate() {
                        *value = left + step * (offset as f64 + 1.0);
                    }
                }
                (Some(left), None) => {
                    for value in &mut result[i..j] {
                        *value = left;
                    }
                }
                (None, Some(right)) => {
                    for value in &mut result[i..j] {
                        *value = right;
                    }
                }
                (None, None) => {}
            }
        }

        i = j;
    }

    result
        .into_iter()
        .map(|value| value.max(0.0) as u32)
        .collect()
}

fn validate_quantization_config(config: &Qwen3AsrConfig) -> Result<()> {
    let Some(quant) = config.mlx_quantization() else {
        return Ok(());
    };

    if let Some(mode) = quant.get("mode").and_then(|v| v.as_str()) {
        if mode != "affine" {
            return Err(Error::InvalidInput(format!(
                "Unsupported MLX quantization mode '{mode}'. Only affine is supported."
            )));
        }
    }

    if let Some(bits) = quant.get("bits").and_then(|v| v.as_u64()) {
        if bits == 0 || bits > 8 {
            return Err(Error::InvalidInput(format!(
                "Unsupported MLX quantization bit-width {bits}. Only 1-8 bits are supported."
            )));
        }
    }

    Ok(())
}

struct PromptTokens {
    ids: Vec<u32>,
    audio_pad_start: usize,
    audio_pad_len: usize,
    system_prompt_requested: bool,
    system_prompt_token_count: usize,
    forced_language: Option<String>,
    forced_language_token_count: usize,
    asr_text_token_available: bool,
    asr_text_token_appended: bool,
}

struct ForcedAlignerPrompt {
    ids: Vec<u32>,
    audio_pad_start: usize,
    audio_pad_len: usize,
    timestamp_token_id: u32,
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn ns_to_ms(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}

fn build_qwen_asr_prompt_tokens(
    specials: &SpecialTokenIds,
    audio_len: usize,
    language: Option<&str>,
    system_prompt: Option<&str>,
    mut encode_text: impl FnMut(&str) -> Result<Vec<u32>>,
) -> Result<PromptTokens> {
    // Match the upstream Qwen speech-family prompt contract:
    // <|im_start|>system\n
    // [optional system prompt text]
    // <|im_end|>\n
    // <|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n
    // <|im_start|>assistant\n
    // If language is explicitly forced, append: "language {Lang}<asr_text>".
    let system_prompt = non_empty_trimmed(system_prompt);
    let forced_language = forced_language_name(language);
    let mut ids = Vec::new();

    ids.push(specials.im_start);
    ids.extend(encode_text("system\n")?);
    let system_prompt_token_count = if let Some(prompt) = system_prompt {
        let prompt_tokens = encode_text(prompt)?;
        let count = prompt_tokens.len();
        ids.extend(prompt_tokens);
        count
    } else {
        0
    };
    ids.push(specials.im_end);
    ids.extend(encode_text("\n")?);

    ids.push(specials.im_start);
    ids.extend(encode_text("user\n")?);
    ids.push(specials.audio_start);

    let audio_pad_start = ids.len();
    ids.extend(std::iter::repeat_n(specials.audio_token, audio_len));

    ids.push(specials.audio_end);
    ids.push(specials.im_end);
    ids.extend(encode_text("\n")?);
    ids.push(specials.im_start);
    ids.extend(encode_text("assistant\n")?);

    let mut forced_language_token_count = 0usize;
    let mut asr_text_token_appended = false;
    if let Some(lang) = forced_language.as_deref() {
        let language_prefix = encode_text("language ")?;
        forced_language_token_count += language_prefix.len();
        ids.extend(language_prefix);

        let language_tokens = encode_text(lang)?;
        forced_language_token_count += language_tokens.len();
        ids.extend(language_tokens);

        if let Some(asr_text) = specials.asr_text {
            ids.push(asr_text);
            forced_language_token_count += 1;
            asr_text_token_appended = true;
        }
    }

    Ok(PromptTokens {
        ids,
        audio_pad_start,
        audio_pad_len: audio_len,
        system_prompt_requested: system_prompt.is_some(),
        system_prompt_token_count,
        forced_language,
        forced_language_token_count,
        asr_text_token_available: specials.asr_text.is_some(),
        asr_text_token_appended,
    })
}

fn non_empty_trimmed(value: Option<&str>) -> Option<&str> {
    let trimmed = value?.trim();
    (!trimmed.is_empty()).then_some(trimmed)
}

fn parse_asr_dtype(dtype: Option<&str>) -> Option<DType> {
    match dtype.map(|d| d.trim().to_ascii_lowercase()) {
        Some(d) if d == "bfloat16" || d == "bf16" => Some(DType::BF16),
        Some(d) if d == "float16" || d == "f16" || d == "fp16" => Some(DType::F16),
        Some(d) if d == "float32" || d == "f32" || d == "fp32" => Some(DType::F32),
        _ => None,
    }
}

fn normalized_language_name(language: &str) -> String {
    let lang = language.trim();
    if lang.eq_ignore_ascii_case("auto") {
        return "Auto".to_string();
    }
    if matches!(lang.to_ascii_lowercase().as_str(), "en" | "eng") {
        return "English".to_string();
    }

    let mut out = String::with_capacity(lang.len());
    let mut new_word = true;
    for ch in lang.chars() {
        if ch.is_ascii_alphabetic() {
            if new_word {
                out.push(ch.to_ascii_uppercase());
                new_word = false;
            } else {
                out.push(ch.to_ascii_lowercase());
            }
        } else {
            out.push(ch);
            new_word = ch == ' ' || ch == '-' || ch == '_';
        }
    }
    out
}

fn forced_language_name(language: Option<&str>) -> Option<String> {
    let lang = language?.trim();
    if lang.is_empty() || lang.eq_ignore_ascii_case("auto") {
        return None;
    }
    Some(normalized_language_name(lang))
}

fn parse_asr_output(raw: &str, user_language: Option<&str>) -> (Option<String>, String) {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return (forced_language_name(user_language), String::new());
    }

    if let Some(language) = forced_language_name(user_language) {
        if let Some(rest) = trimmed.strip_prefix("language ") {
            if let Some((_, text)) = split_language_prefixed_output(rest) {
                return (Some(language), text);
            }
        }
        if let Some(text) = strip_forced_language_echo(trimmed, &language) {
            return (Some(language), text.to_string());
        }
        return (Some(language), trimmed.to_string());
    }

    let asr_text_token = "<asr_text>";
    if let Some((meta_part, text_part)) = trimmed.split_once(asr_text_token) {
        let meta_trimmed = meta_part.trim();
        let text_trimmed = text_part.trim();
        if meta_trimmed.to_ascii_lowercase().contains("language none") {
            return if text_trimmed.is_empty() {
                (None, String::new())
            } else {
                (None, text_trimmed.to_string())
            };
        }

        for line in meta_trimmed.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(value) = line.strip_prefix("language ") {
                let value = value.trim();
                if !value.is_empty() {
                    return (
                        Some(normalized_language_name(value)),
                        text_trimmed.to_string(),
                    );
                }
            }
        }

        return (None, text_trimmed.to_string());
    }

    if let Some(rest) = trimmed.strip_prefix("language ") {
        if let Some((language, text)) = split_language_prefixed_output(rest) {
            return (language, text);
        }
    }

    (None, trimmed.to_string())
}

fn strip_forced_language_echo<'a>(text: &'a str, language: &str) -> Option<&'a str> {
    let mut split_idx = None;
    let mut prev_is_lower = false;
    for (idx, ch) in text.char_indices() {
        if prev_is_lower && ch.is_ascii_uppercase() {
            split_idx = Some(idx);
            break;
        }
        prev_is_lower = ch.is_lowercase();
    }
    let split_idx = split_idx?;
    let prefix = text[..split_idx]
        .chars()
        .map(|ch| if ch == '\u{261}' { 'g' } else { ch })
        .collect::<String>()
        .to_ascii_lowercase();
    let language = language.to_ascii_lowercase();
    (prefix.chars().count() >= 3 && language.ends_with(&prefix)).then_some(text[split_idx..].trim())
}

fn split_language_prefixed_output(rest: &str) -> Option<(Option<String>, String)> {
    let rest = rest.trim();
    if rest.is_empty() {
        return None;
    }
    if rest.eq_ignore_ascii_case("none") {
        return Some((None, String::new()));
    }

    // Preferred split: explicit whitespace between language and transcript.
    if let Some((lang, text)) = rest.split_once(char::is_whitespace) {
        let lang = lang.trim();
        if !lang.is_empty()
            && lang.chars().all(|ch| ch.is_ascii_alphabetic() || ch == '-')
            && !has_lower_to_upper_transition(lang)
        {
            return Some((
                Some(normalized_language_name(lang)),
                text.trim().to_string(),
            ));
        }
    }

    // Fallback for GGUF decode where `<asr_text>` is not in tokenizer vocab and
    // the language token glues directly to text, e.g. "EnglishThe quick ...".
    let chars = rest.char_indices().peekable();
    let mut split_idx = None;
    let mut prev_is_lower = false;
    for (idx, ch) in chars {
        if idx == 0 {
            prev_is_lower = ch.is_ascii_lowercase();
            continue;
        }
        if prev_is_lower && ch.is_ascii_uppercase() {
            split_idx = Some(idx);
            break;
        }
        prev_is_lower = ch.is_ascii_lowercase();
    }
    let split_idx = split_idx?;
    let lang = rest[..split_idx].trim();
    if lang.is_empty() || !lang.chars().all(|ch| ch.is_ascii_alphabetic() || ch == '-') {
        return None;
    }
    let text = rest[split_idx..].trim().to_string();
    Some((Some(normalized_language_name(lang)), text))
}

fn has_lower_to_upper_transition(value: &str) -> bool {
    let mut prev_is_lower = false;
    for ch in value.chars() {
        if prev_is_lower && ch.is_ascii_uppercase() {
            return true;
        }
        prev_is_lower = ch.is_ascii_lowercase();
    }
    false
}

fn build_mrope_positions(
    seq_len: usize,
    start_pos: usize,
    audio_span: Option<(usize, usize)>,
) -> Vec<i64> {
    if let Some((audio_start, audio_len)) = audio_span {
        let mut pos = Vec::with_capacity(seq_len);
        let mut st = 0usize;
        let mut st_idx = start_pos as i64;

        if audio_start > 0 && audio_start <= seq_len {
            let text_len = audio_start - st;
            for i in 0..text_len {
                pos.push(st_idx + i as i64);
            }
            st = audio_start;
            st_idx += text_len as i64;
        }

        if audio_len > 0 && st < seq_len {
            let audio_take = audio_len.min(seq_len - st);
            for i in 0..audio_take {
                pos.push(st_idx + i as i64);
            }
            st += audio_take;
            st_idx += audio_take as i64;
        }

        if st < seq_len {
            let tail = seq_len - st;
            for i in 0..tail {
                pos.push(st_idx + i as i64);
            }
        }

        pos
    } else {
        (start_pos..start_pos + seq_len).map(|p| p as i64).collect()
    }
}

/// Raw output for [`parse_asr_output`]. Like upstream's
/// `batch_decode(skip_special_tokens=True)`, it keeps the non-special
/// `<asr_text>` marker that separates `language X` from the transcript.
fn decode_raw_asr_output(
    tokenizer: &AsrTokenizer,
    specials: &SpecialTokenIds,
    tokens: &[u32],
) -> Result<String> {
    let filtered: Vec<u32> = tokens
        .iter()
        .copied()
        .filter(|id| specials.asr_text == Some(*id) || !is_special_generation_token(specials, *id))
        .collect();
    tokenizer.decode_text(&filtered)
}

fn is_special_generation_token(specials: &SpecialTokenIds, id: u32) -> bool {
    if id == specials.im_start
        || id == specials.im_end
        || id == specials.audio_start
        || id == specials.audio_end
        || id == specials.audio_token
        || id == specials.pad
        || id == specials.eos
        || specials.eos_alt == Some(id)
    {
        return true;
    }
    if let Some(asr_text) = specials.asr_text {
        if id == asr_text {
            return true;
        }
    }
    if let Some(id0) = specials.fim_prefix {
        if id == id0 {
            return true;
        }
    }
    if let Some(id0) = specials.fim_middle {
        if id == id0 {
            return true;
        }
    }
    if let Some(id0) = specials.fim_suffix {
        if id == id0 {
            return true;
        }
    }
    if let Some(id0) = specials.fim_pad {
        if id == id0 {
            return true;
        }
    }
    false
}

/// The canonical resampler (soxr HQ response), so a direct caller at
/// another rate gets the front end production audio gets (QFA-4).
fn resample(audio: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    qwen3_asr_resampled_sample_count(audio.len(), src_rate, dst_rate)?;
    crate::audio::resample_mono_high_quality(audio, src_rate, dst_rate)
}

fn argmax(logits: &Tensor) -> Result<u32> {
    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (batch, _vocab) = logits.dims2()?;
            if batch != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected batched logits for argmax: expected batch=1, got {batch}"
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected logits rank for argmax: {rank}"
            )))
        }
    };
    let idx = logits.argmax(D::Minus1)?;
    let idx = if idx.rank() == 0 {
        idx
    } else {
        idx.squeeze(0)?
    };
    idx.to_dtype(DType::U32)?
        .to_scalar::<u32>()
        .map_err(Error::from)
}

fn take_quantum_argmax(output: &mut Option<Tensor>) -> Result<u32> {
    let output = output.take().ok_or_else(|| {
        Error::InferenceError("Qwen3 ASR decode quantum has no unconsumed model output".to_string())
    })?;
    let (batch, sequence, _width) = output.dims3()?;
    if batch != 1 || sequence == 0 {
        return Err(Error::InferenceError(format!(
            "Qwen3 ASR decode output expects [1,sequence,width], got {:?}",
            output.dims()
        )));
    }
    argmax(&output.i((0, sequence - 1))?)
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

fn collect_stop_token_ids(specials: &SpecialTokenIds) -> Vec<u32> {
    let mut stop_ids = vec![specials.im_end, specials.eos];
    if let Some(alt) = specials.eos_alt {
        if alt != specials.im_end && alt != specials.eos {
            stop_ids.push(alt);
        }
    }
    stop_ids
}

/// The class of each timestamp row. `logits` holds one row per entry of
/// `positions` (the prompt positions they were scored at, for errors).
fn timestamp_argmax(logits: &Tensor, positions: &[usize]) -> Result<Vec<u32>> {
    if positions.is_empty() {
        return Ok(Vec::new());
    }
    let dims = logits.dims();
    if dims.len() != 3 || dims[0] != 1 || dims[1] != positions.len() {
        return Err(Error::InferenceError(format!(
            "Forced aligner timestamp argmax expects logits shape [1, {}, classes], got {dims:?}",
            positions.len()
        )));
    }
    let token_logits = logits.i(0)?;
    // A NaN row argmaxes to class 0, which would read as a fabricated 0 ms
    // timestamp; a device fault must surface as an error instead.
    let rows = token_logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    if let Some(row) = rows
        .iter()
        .position(|row| row.iter().any(|value| !value.is_finite()))
    {
        return Err(Error::InferenceError(format!(
            "Forced aligner produced non-finite timestamp logits at prompt position {} \
             (timestamp {row} of {})",
            positions[row],
            positions.len()
        )));
    }
    let indices = token_logits.argmax(D::Minus1)?.to_dtype(DType::U32)?;
    indices.to_vec1::<u32>().map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;

    use crate::backends::kv::{CpuKvArena, KvArena, KvArenaConfig, KvLayerConfig};
    use crate::backends::state::{
        negotiate_state_plan, StateBackendPlanRequest, TensorStateCapacity,
    };
    use crate::backends::BackendKind;
    use crate::backends::DeviceSelector;
    use crate::engine::{
        ConcurrencyClass, ExecutionDomain, MembershipSafePoint, ModelInstanceId, NativeBatchMode,
        OutputVisibility, StageId, StageProgressKind, StageShapePolicy, StageWorkSelector,
    };
    use crate::kv::v2::{InvocationWorkspaceSet, RetainedStateCapability};
    use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
    use crate::models::architectures::qwen3::core::tiny_qwen3_model_for_test;

    fn test_managed_arena() -> (Arc<dyn KvArena>, Vec<KvLayerBinding>) {
        let binding = KvLayerBinding {
            model_layer: 0,
            physical_layer: 0,
        };
        let arena = CpuKvArena::new(KvArenaConfig {
            id: KvArenaId {
                model_instance: ModelInstanceId::new(193),
                backend: BackendKind::Cpu,
                device_ordinal: None,
                generation: 1,
            },
            group: KvGroupId::new(0),
            page_tokens: 2,
            capacity_pages: 24,
            growth: None,
            dtype: DType::F32,
            layers: vec![KvLayerConfig {
                binding,
                num_kv_heads: 1,
                key_head_dim: 2,
                value_head_dim: 2,
            }],
        })
        .expect("ASR test arena");
        (Arc::new(arena), vec![binding])
    }

    fn test_managed_cache(
        arena: Arc<dyn KvArena>,
        bindings: Vec<KvLayerBinding>,
        first_page: u32,
        context_len: usize,
    ) -> PhysicalPagedKvCache {
        let blocks = (first_page..first_page + 4)
            .map(|index| CacheBlockRef {
                arena: arena.id(),
                group: arena.config().group,
                index,
                slot_generation: 1,
            })
            .collect();
        PhysicalPagedKvCache::new(arena, bindings, blocks, context_len)
            .expect("ASR test managed cache")
    }

    fn test_decode_state(
        cache: PhysicalPagedKvCache,
        prepared_prompt_embeddings: Option<Tensor>,
        prompt_tokens: usize,
        prefill_progress: usize,
        pos: usize,
        pending_token: Option<u32>,
    ) -> AsrDecodeState {
        AsrDecodeState {
            cache: Some(cache),
            physical_tensor_sequence: None,
            prepared_prompt_embeddings,
            prepared_position_ids: None,
            prefill_progress,
            unconsumed_output: None,
            pos,
            pending_token,
            language: None,
            generated_ids: Vec::new(),
            visible_generated_ids: Vec::new(),
            decoded_visible_tokens: 0,
            tokens_since_resync: 0,
            assembled: String::new(),
            diagnostics: Qwen3AsrDiagnostics {
                prompt_tokens,
                max_new_tokens: 8,
                ..Default::default()
            },
            profile_start: Qwen3RuntimeProfileSnapshot::default(),
            stop_tokens: vec![u32::MAX],
            max_new_tokens: 8,
            finished: false,
        }
    }

    fn assert_tensor_close(lhs: &Tensor, rhs: &Tensor) {
        assert_eq!(lhs.dims(), rhs.dims());
        let lhs = lhs
            .to_dtype(DType::F32)
            .expect("lhs dtype")
            .flatten_all()
            .expect("lhs flatten")
            .to_vec1::<f32>()
            .expect("lhs values");
        let rhs = rhs
            .to_dtype(DType::F32)
            .expect("rhs dtype")
            .flatten_all()
            .expect("rhs flatten")
            .to_vec1::<f32>()
            .expect("rhs values");
        for (idx, (lhs, rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() <= 1e-5,
                "tensor mismatch at {idx}: {lhs} != {rhs}"
            );
        }
    }

    #[test]
    fn gguf_context_falls_back_to_revisioned_variant_manifest() {
        assert_eq!(
            resolve_qwen3_asr_context(None, None, ModelVariant::Qwen3Asr06BGguf).unwrap(),
            65_536
        );
        assert_eq!(
            resolve_qwen3_asr_context(None, Some(65_536), ModelVariant::Qwen3Asr17BGguf).unwrap(),
            65_536
        );
    }

    #[test]
    fn gguf_context_rejects_conflicting_artifact_sources() {
        let error =
            resolve_qwen3_asr_context(Some(32_768), Some(32_768), ModelVariant::Qwen3Asr06BGguf)
                .unwrap_err();
        assert!(error.to_string().contains("context conflict"));
        assert!(error.to_string().contains("Qwen/Qwen3-ASR-0.6B"));
    }
    /// D1: one 180 s aligner window on every backend, rejected (not
    /// truncated) above it; the configured 30 s ASR frame count no longer
    /// bounds the aligner on CPU/Metal.
    #[test]
    fn forced_aligner_takes_one_180_s_window_on_every_backend() {
        let configured = 3_000;
        for cuda in [false, true] {
            assert_eq!(
                qwen3_asr_effective_max_frames(cuda, true, configured, 16_000, 160),
                18_000
            );
            assert_eq!(
                qwen3_asr_admit_frames(3_001, cuda, true, configured, 16_000, 160).unwrap(),
                3_001
            );
            assert_eq!(
                qwen3_asr_admit_frames(18_000, cuda, true, configured, 16_000, 160).unwrap(),
                18_000
            );
            let error = qwen3_asr_admit_frames(18_001, cuda, true, configured, 16_000, 160)
                .expect_err("the aligner must reject, not truncate, over-limit audio");
            assert!(matches!(error, Error::InvalidInput(_)), "{error}");
            assert!(error.to_string().contains("180.0 s"), "{error}");
        }
    }

    /// With the last centre-padded frame dropped, 180.0 s encodes exactly the
    /// 18,000-frame limit, so the window hint is 180.0 s; one more hop is
    /// rejected.
    #[test]
    fn forced_aligner_window_hint_fits_the_frame_limit() {
        let frames = |samples: usize| {
            qwen3_asr_frontend_geometry(samples, 16_000, 16_000, 400, 160, false, 0)
                .unwrap()
                .mel_frames
        };
        let window = 180 * 16_000;
        assert_eq!(frames(window), 18_000);
        assert_eq!(frames(window + 159), 18_000);
        assert_eq!(frames(window + 160), 18_001);
        assert_eq!(frames(30 * 16_000), 3_000);
        for cuda in [false, true] {
            for (samples, admitted) in [(window, true), (window + 160, false)] {
                assert_eq!(
                    qwen3_asr_admit_frames(frames(samples), cuda, true, 3_000, 16_000, 160).is_ok(),
                    admitted
                );
            }
        }
    }

    #[test]
    fn forced_aligner_word_bound_is_proportional_to_the_window() {
        assert_eq!(
            qwen3_aligner_max_words(0.0),
            QWEN3_ALIGNER_MIN_WORD_ALLOWANCE
        );
        assert_eq!(
            qwen3_aligner_max_words(1.0),
            QWEN3_ALIGNER_MIN_WORD_ALLOWANCE
        );
        assert_eq!(qwen3_aligner_max_words(30.0), 300);
        assert_eq!(qwen3_aligner_max_words(180.0), 1_800);
        assert_eq!(
            qwen3_aligner_max_words(f64::NAN),
            QWEN3_ALIGNER_MIN_WORD_ALLOWANCE
        );
    }

    #[test]
    fn cuda_audio_limits_expand_portable_transcription_truncation() {
        let configured = 3_000;
        assert_eq!(
            qwen3_asr_admit_frames(3_001, false, false, configured, 16_000, 160).unwrap(),
            configured
        );
        assert_eq!(
            qwen3_asr_effective_max_frames(true, false, configured, 16_000, 160),
            120_000
        );
        assert_eq!(
            qwen3_asr_admit_frames(120_000, true, false, configured, 16_000, 160).unwrap(),
            120_000
        );
        assert!(qwen3_asr_admit_frames(120_001, true, false, configured, 16_000, 160).is_err());
        assert!(qwen3_asr_admit_frames(18_001, true, true, configured, 16_000, 160).is_err());
    }

    fn physical_spec_stage(id: u32, progress: StageProgressKind) -> StageDescriptor {
        StageDescriptor {
            id: StageId::new(id),
            name: format!("qwen3-asr-{id}"),
            selector: if progress == StageProgressKind::Atomic {
                StageWorkSelector::Atomic
            } else {
                StageWorkSelector::SequencePrefill
            },
            domain: ExecutionDomain::ExecutionGroup,
            progress,
            concurrency: ConcurrencyClass::Exclusive,
            physical_launch_policy: crate::engine::PhysicalLaunchPolicy::ExecutionGroupExclusive,
            batch_mode: NativeBatchMode::None,
            max_batch_size: 1,
            max_work_units: 1,
            workspace_base_bytes: 0,
            workspace_per_row_bytes: 0,
            workspace_per_work_unit_bytes: 0,
            max_workspace_bytes: 0,
            max_padding_basis_points: 0,
            max_formation_delay: Duration::ZERO,
            shape_policy: StageShapePolicy::Exact,
            membership_safe_point: MembershipSafePoint::OperationBoundary,
            output_visibility: OutputVisibility::AfterQuantumCommit,
            retained_state_selections: None,
        }
    }

    #[test]
    fn qwen_asr_physical_spec_separates_offline_and_streaming_state() {
        let offline = [physical_spec_stage(1, StageProgressKind::Atomic)];
        let streaming = [physical_spec_stage(2, StageProgressKind::Iterative)];
        let mut preparation_stage = physical_spec_stage(3, StageProgressKind::Atomic);
        preparation_stage.selector = StageWorkSelector::PreSequencePreparation;
        preparation_stage.batch_mode = NativeBatchMode::Static;
        preparation_stage.concurrency = ConcurrencyClass::Batchable;
        preparation_stage.max_batch_size = 4;
        preparation_stage.max_workspace_bytes = 128;
        let preparation = [preparation_stage];
        let graphs = vec![
            offline.as_slice(),
            streaming.as_slice(),
            preparation.as_slice(),
        ];
        let retained =
            qwen3_asr_retained_state_contract(crate::kv::test_contract(), 4096, 256, DType::F32)
                .expect("retained ASR state");
        let spec = qwen3_asr_physical_state_spec(&retained, 4096, 4096, &graphs)
            .expect("physical ASR spec");

        assert!(matches!(
            spec.descriptor.retained,
            RetainedStateCapability::Managed { .. }
        ));
        assert_eq!(spec.retained.domains.len(), 2);
        assert!(matches!(
            spec.retained.domains[1],
            StateDomainSpec::Tensor(_)
        ));
        assert_eq!(spec.invocation.domains.len(), 1);
        assert!(matches!(
            spec.invocation.domains[0],
            StateDomainSpec::PagedAttention(_)
        ));
        assert_eq!(
            spec.invocation.groups[0].domains,
            vec![StateDomainId::new(1)]
        );
        let InvocationWorkspaceSet::Bounded { profiles } = &spec.descriptor.invocation else {
            panic!("offline Qwen3 ASR must declare invocation pages");
        };
        assert_eq!(profiles.len(), graphs.len());
        for graph in graphs {
            let profile = profiles
                .iter()
                .find(|profile| {
                    profile.stage_graph_fingerprint == stage_graph_fingerprint(graph).unwrap()
                })
                .expect("exact graph profile");
            let state_count = profile
                .stages
                .iter()
                .flat_map(|stage| stage.domains.iter())
                .filter(|domain| matches!(domain, InvocationWorkspaceDomain::State { .. }))
                .count();
            let scratch_count = profile
                .stages
                .iter()
                .flat_map(|stage| stage.domains.iter())
                .filter(|domain| matches!(domain, InvocationWorkspaceDomain::Scratch { .. }))
                .count();
            let owns_invocation = graph
                .iter()
                .any(|stage| stage.selector == StageWorkSelector::Atomic);
            assert_eq!(state_count, usize::from(owns_invocation));
            assert_eq!(
                scratch_count,
                graph
                    .iter()
                    .filter(|stage| stage.max_workspace_bytes > 0)
                    .count()
            );
        }
    }

    #[test]
    fn qwen_asr_automatic_context_is_one_planned_chunk_per_sequence() {
        // 128-token text reserve, ceil(30 s / 0.95 * 13) audio tokens and the
        // 768-token planning decode reserve.
        let ceiling = qwen3_asr_sequence_ceiling(13.0, 30.0);
        assert_eq!(ceiling, 128 + 411 + 768);
        assert_eq!(
            qwen3_asr_automatic_context(65_536, ceiling, 8),
            (ceiling, 8 * ceiling)
        );
        // Many CUDA rows or a small native context never exceed the native bound.
        assert_eq!(
            qwen3_asr_automatic_context(65_536, ceiling, 96),
            (ceiling, 65_536)
        );
        assert_eq!(
            qwen3_asr_automatic_context(1_024, ceiling, 8),
            (1_024, 1_024)
        );
        assert_eq!(
            qwen3_asr_automatic_context(65_536, ceiling, 0),
            (ceiling, ceiling)
        );
    }

    #[test]
    fn qwen_asr_ceiling_admits_a_full_chunk_on_every_planner() {
        let ceiling = qwen3_asr_sequence_ceiling(13.0, 30.0);
        let chunk = vec![0.0_f32; 30 * 16_000];
        // Preparation plans with the default generation budget, execution
        // with the transcription cap; both must keep a 30 s input whole.
        for decode_budget in [
            crate::models::registry::NativeAsrGenerationOptions::default().max_new_tokens,
            DEFAULT_TRANSCRIBE_MAX_NEW_TOKENS,
        ] {
            let limit = crate::runtime::compose_invocation_audio_limit(
                Some(120.0),
                Some(13.0),
                Some(ceiling),
                |_| decode_budget,
            )
            .expect("the ceiling admits a planned chunk");
            assert!(
                !crate::engine::qwen3_asr_requires_long_form(&chunk, 16_000, limit),
                "a 30 s input must stay one chunk with a {decode_budget}-token budget"
            );
        }

        // The exact 30 s prompt plus the 512-token decode cap fits a sequence.
        let frontend =
            qwen3_asr_frontend_geometry(30 * 16_000, 16_000, 16_000, 400, 160, false, 12_000)
                .unwrap();
        let audio_tokens = audio::get_cnn_output_lengths(&[frontend.mel_frames])[0];
        // 3,000 frames, as WhisperFeatureExtractor gives (QFA-4).
        assert_eq!(audio_tokens, 390);
        let prompt = build_qwen_asr_prompt_tokens(
            &test_specials(Some(7)),
            audio_tokens,
            Some("english"),
            None,
            encode_prompt_piece_for_test,
        )
        .unwrap();
        assert!(prompt.ids.len() + DEFAULT_TRANSCRIBE_MAX_NEW_TOKENS <= ceiling);
    }

    #[test]
    fn qwen_asr_automatic_ceiling_shrinks_state_for_shipped_geometry() {
        let ceiling = qwen3_asr_sequence_ceiling(13.0, 30.0);
        // Qwen3-ASR 0.6B and 1.7B share the decoder KV geometry: 28 layers
        // of 8 KV heads at head_dim 128, stored as F16 on Metal.
        let domain = StateDomainSpec::PagedAttention(
            crate::models::architectures::qwen3::core::qwen3_decoder_cache_domain(
                crate::models::architectures::qwen3::core::Qwen3DecoderCacheGeometry {
                    domain: StateDomainId::new(1),
                    clock: StateClock::DecoderTokens,
                    num_layers: 28,
                    num_query_heads: 16,
                    num_kv_heads: 8,
                    key_head_dim: 128,
                    value_head_dim: 128,
                    sliding_window: None,
                    storage_dtype: DType::F16,
                    preferred_page_tokens: 64,
                    prefix: PrefixPolicy::Disabled,
                },
            )
            .unwrap(),
        );
        let token_bytes = 28 * 8 * (128 + 128) * 2;
        let kv_bytes =
            |tokens: usize| qwen3_asr_invocation_workspace_bytes(&domain, tokens as u64).unwrap();
        // 7 GiB at the native context; 1,344 page-rounded tokens at the ceiling.
        assert_eq!(kv_bytes(65_536), 65_536 * token_bytes);
        assert_eq!(kv_bytes(ceiling), 1_344 * token_bytes);
        assert!(kv_bytes(ceiling) * 48 < kv_bytes(65_536));
        // Eight retained rows still need under a sixth of the native pool.
        let (_, retained) = qwen3_asr_automatic_context(65_536, ceiling, 8);
        assert!(kv_bytes(retained) * 6 < kv_bytes(65_536));

        // Prepared-input rows: F16 embeddings plus I64 MRoPE positions, for
        // the 0.6B (hidden 1024) and 1.7B (hidden 2048) decoders.
        for hidden in [1_024_usize, 2_048] {
            let row_bytes = |max_sequence: usize| {
                let contract = qwen3_asr_retained_state_contract(
                    crate::kv::test_contract(),
                    max_sequence,
                    hidden,
                    DType::F16,
                )
                .unwrap();
                negotiate_state_plan(
                    &contract,
                    &StateBackendPlanRequest {
                        backend: BackendKind::Cpu,
                        device_ordinal: None,
                        page_tokens_hint: Some(16),
                        storage_dtype_hint: Some(StateDType::F16),
                    },
                )
                .unwrap()
                .non_paged[0]
                    .maximum_bytes()
            };
            let expected = |tokens: usize| (tokens * (hidden * 2 + 3 * 8)) as u64;
            assert_eq!(row_bytes(ceiling), expected(ceiling));
            assert_eq!(row_bytes(65_536), expected(65_536));
            assert!(row_bytes(ceiling) * 50 < row_bytes(65_536));
        }
    }

    #[test]
    fn qwen_asr_physical_spec_bounds_invocation_by_the_sequence_context() {
        let offline = [physical_spec_stage(1, StageProgressKind::Atomic)];
        let graphs = vec![offline.as_slice()];
        let ceiling = qwen3_asr_sequence_ceiling(13.0, 30.0);
        let (sequence, retained_max) = qwen3_asr_automatic_context(65_536, ceiling, 8);
        let retained = qwen3_asr_retained_state_contract(
            crate::kv::test_contract(),
            sequence,
            256,
            DType::F32,
        )
        .expect("retained ASR state");
        let spec = qwen3_asr_physical_state_spec(&retained, sequence, retained_max, &graphs)
            .expect("physical ASR spec");

        assert_eq!(spec.sequence_tokens, ceiling);
        assert_eq!(spec.retained_max_tokens, 8 * ceiling);
        assert_eq!(
            spec.descriptor
                .capacity_axis_bounds(crate::kv::v2::StateCapacityAxis::DecoderContext),
            Some((1, ceiling as u64))
        );
    }

    #[test]
    fn qwen_asr_prepared_input_contract_accounts_embeddings_and_i64_mrope_exactly() {
        let contract =
            qwen3_asr_retained_state_contract(crate::kv::test_contract(), 8, 4, DType::F32)
                .expect("retained ASR state");
        let StateDomainSpec::Tensor(prepared) = &contract.domains[1] else {
            panic!("prepared inputs must be tensor state");
        };
        assert_eq!(prepared.components.len(), 2);
        assert_eq!(
            prepared.components[0].accepted_dtypes,
            vec![StateDType::F32]
        );
        assert_eq!(
            prepared.components[1].accepted_dtypes,
            vec![StateDType::I64]
        );

        let plan = negotiate_state_plan(
            &contract,
            &StateBackendPlanRequest {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                page_tokens_hint: Some(16),
                storage_dtype_hint: Some(StateDType::F16),
            },
        )
        .expect("CPU composite state plan");
        assert_eq!(plan.non_paged.len(), 1);
        // [1, 8, 4] F32 embeddings plus [3, 8] I64 MRoPE.
        assert_eq!(plan.non_paged[0].maximum_bytes(), 128 + 192);
        let capacity = TensorStateCapacity::for_plan(&plan, 1, 1).expect("tensor capacity");
        assert_eq!(capacity.per_sequence_bytes(), 320);
        assert_eq!(capacity.authorized_bytes(), 640);
    }

    #[test]
    fn qwen_asr_prepared_inputs_roundtrip_through_transactional_tensor_state() {
        let contract =
            qwen3_asr_retained_state_contract(crate::kv::test_contract(), 4, 4, DType::F32)
                .expect("retained ASR state");
        let plan = negotiate_state_plan(
            &contract,
            &StateBackendPlanRequest {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                page_tokens_hint: Some(16),
                storage_dtype_hint: Some(StateDType::F16),
            },
        )
        .expect("CPU composite state plan");
        let capacity = TensorStateCapacity::for_plan(&plan, 1, 1).expect("tensor capacity");
        let tensor_arena = TensorStateArena::new(Arc::new(plan), capacity, Device::Cpu)
            .expect("prepared-input tensor arena");
        let sequence = PhysicalStateSequenceId::new(7).unwrap();
        tensor_arena.register(sequence).unwrap();

        let embeddings = Tensor::from_vec(
            (0..16).map(|value| value as f32).collect::<Vec<_>>(),
            (1, 4, 4),
            &Device::Cpu,
        )
        .unwrap();
        let mrope = Tensor::from_vec(
            (0..12).map(i64::from).collect::<Vec<_>>(),
            (3, 4),
            &Device::Cpu,
        )
        .unwrap();
        let (kv_arena, bindings) = test_managed_arena();
        let mut state = test_decode_state(
            test_managed_cache(kv_arena, bindings, 0, 2),
            Some(embeddings.clone()),
            4,
            2,
            2,
            None,
        );
        state.prepared_position_ids = Some(mrope.clone());
        state.bind_tensor_sequence(sequence.get()).unwrap();

        let transaction = PhysicalStateTransactionId::new(11).unwrap();
        tensor_arena.begin(transaction, sequence).unwrap();
        state
            .stage_prepared_tensor_state(&tensor_arena, transaction.get())
            .unwrap();
        assert!(state.prepared_prompt_embeddings.is_none());
        assert!(state.prepared_position_ids.is_none());
        tensor_arena.commit(transaction, 2).unwrap();

        state.restore_prepared_tensor_state(&tensor_arena).unwrap();
        assert_tensor_close(
            state.prepared_prompt_embeddings.as_ref().unwrap(),
            &embeddings,
        );
        assert_eq!(
            state
                .prepared_position_ids
                .as_ref()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<i64>()
                .unwrap(),
            mrope.flatten_all().unwrap().to_vec1::<i64>().unwrap()
        );

        let aborted = PhysicalStateTransactionId::new(12).unwrap();
        tensor_arena.begin(aborted, sequence).unwrap();
        state
            .stage_prepared_tensor_state(&tensor_arena, aborted.get())
            .unwrap();
        tensor_arena.abort(aborted).unwrap();
        state.restore_prepared_tensor_state(&tensor_arena).unwrap();
        assert_tensor_close(
            state.prepared_prompt_embeddings.as_ref().unwrap(),
            &embeddings,
        );
    }

    #[test]
    fn qwen_asr_resumable_prefill_matches_monolithic_final_logits() {
        let device = Device::Cpu;
        let model = tiny_qwen3_model_for_test(&device);
        let prompt = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &device).unwrap();
        let prepared = model.embeddings(&prompt).unwrap();
        let (arena, bindings) = test_managed_arena();
        let mut full = test_decode_state(
            test_managed_cache(arena.clone(), bindings.clone(), 0, 0),
            Some(prepared.clone()),
            4,
            0,
            0,
            None,
        );
        let mut chunked = test_decode_state(
            test_managed_cache(arena, bindings, 4, 0),
            Some(prepared),
            4,
            0,
            0,
            None,
        );

        assert!(continue_asr_resumable_prefill(&model, &mut full, 0, 4).unwrap());
        assert!(!continue_asr_resumable_prefill(&model, &mut chunked, 0, 2).unwrap());
        assert_eq!(chunked.prefill_progress, 2);
        assert_eq!(chunked.pos, 2);
        assert_eq!(chunked.cache.as_ref().unwrap().context_len(), 2);
        assert!(chunked.unconsumed_output.is_none());
        assert_eq!(
            chunked.prepared_prompt_embeddings.as_ref().unwrap().dims(),
            &[1, 4, 4]
        );

        assert!(continue_asr_resumable_prefill(&model, &mut chunked, 2, 4).unwrap());
        assert!(chunked.prepared_prompt_embeddings.is_none());
        assert!(chunked.prepared_position_ids.is_none());
        let full_output = full.unconsumed_output.as_ref().unwrap();
        let chunked_output = chunked.unconsumed_output.as_ref().unwrap();
        // Prefill projects only the last prompt row.
        assert_eq!(full_output.dim(1).unwrap(), 1);
        assert_eq!(chunked_output.dim(1).unwrap(), 1);
        assert_tensor_close(full_output, chunked_output);
        assert_eq!(
            full.cache.as_mut().unwrap().take_completed_writes().len(),
            1
        );
        assert_eq!(
            chunked
                .cache
                .as_mut()
                .unwrap()
                .take_completed_writes()
                .len(),
            2
        );
    }

    #[test]
    fn qwen_asr_resumable_prefill_rejects_non_monotonic_spans_without_progress() {
        let device = Device::Cpu;
        let model = tiny_qwen3_model_for_test(&device);
        let prompt = Tensor::from_vec(vec![0u32, 1, 2, 3], (1, 4), &device).unwrap();
        let prepared = model.embeddings(&prompt).unwrap();
        let (arena, bindings) = test_managed_arena();
        let mut state = test_decode_state(
            test_managed_cache(arena, bindings, 0, 0),
            Some(prepared),
            4,
            0,
            0,
            None,
        );

        let error = continue_asr_resumable_prefill(&model, &mut state, 1, 2)
            .expect_err("prefill may not skip its first span");
        assert!(error.to_string().contains("incompatible with cursor 0"));
        assert_eq!(state.prefill_progress, 0);
        assert_eq!(state.pos, 0);
        assert_eq!(state.cache.as_ref().unwrap().context_len(), 0);

        assert!(!continue_asr_resumable_prefill(&model, &mut state, 0, 2).unwrap());
        let error = continue_asr_resumable_prefill(&model, &mut state, 3, 4)
            .expect_err("prefill may not leave a cursor gap");
        assert!(error.to_string().contains("incompatible with cursor 2"));
        assert_eq!(state.prefill_progress, 2);
        assert_eq!(state.pos, 2);
        assert_eq!(state.cache.as_ref().unwrap().context_len(), 2);
        assert!(state.unconsumed_output.is_none());
    }

    #[test]
    fn qwen_asr_first_middle_and_final_prefill_transactions_roll_back_exactly() {
        let device = Device::Cpu;
        let model = tiny_qwen3_model_for_test(&device);
        let prompt = Tensor::from_vec(vec![0u32, 1, 2, 3, 4], (1, 5), &device).unwrap();
        let prepared = model.embeddings(&prompt).unwrap();
        let (arena, bindings) = test_managed_arena();
        let mut state = test_decode_state(
            test_managed_cache(arena.clone(), bindings.clone(), 0, 0),
            Some(prepared),
            5,
            0,
            0,
            None,
        );

        for (span_index, (start, end, expected_complete)) in
            [(0, 2, false), (2, 4, false), (4, 5, true)]
                .into_iter()
                .enumerate()
        {
            let baseline_embeddings = state
                .prepared_prompt_embeddings
                .as_ref()
                .expect("uncommitted span retains immutable prompt input")
                .clone();
            let first_page = 4 + u32::try_from(span_index).unwrap() * 4;
            let checkpoint = state
                .begin_managed_quantum(test_managed_cache(
                    arena.clone(),
                    bindings.clone(),
                    first_page,
                    start,
                ))
                .unwrap();

            assert_eq!(
                continue_asr_resumable_prefill(&model, &mut state, start, end).unwrap(),
                expected_complete
            );
            assert_eq!(state.prefill_progress, end);
            assert_eq!(state.pos, end);
            assert_eq!(state.cache.as_ref().unwrap().context_len(), end);

            state.rollback_managed_quantum(checkpoint);
            assert_eq!(state.prefill_progress, start);
            assert_eq!(state.pos, start);
            assert_eq!(state.cache.as_ref().unwrap().context_len(), start);
            assert!(state.unconsumed_output.is_none());
            assert_tensor_close(
                state.prepared_prompt_embeddings.as_ref().unwrap(),
                &baseline_embeddings,
            );

            let checkpoint = state
                .begin_managed_quantum(test_managed_cache(
                    arena.clone(),
                    bindings.clone(),
                    first_page,
                    start,
                ))
                .unwrap();
            assert_eq!(
                continue_asr_resumable_prefill(&model, &mut state, start, end).unwrap(),
                expected_complete
            );
            drop(checkpoint);
            assert_eq!(
                state.cache.as_mut().unwrap().take_completed_writes().len(),
                1
            );
            assert_eq!(state.prefill_progress, end);
            assert_eq!(state.pos, end);
            assert_eq!(state.unconsumed_output.is_some(), expected_complete);
        }

        assert!(state.prepared_prompt_embeddings.is_none());
        assert!(state.prepared_position_ids.is_none());
    }

    #[test]
    fn qwen_asr_managed_quantum_rollback_restores_cache_cursor_and_pending_token() {
        let (arena, bindings) = test_managed_arena();
        let mut state = test_decode_state(
            test_managed_cache(arena.clone(), bindings.clone(), 0, 4),
            None,
            4,
            4,
            4,
            Some(7),
        );
        state.generated_ids.push(7);
        let old_next_slot = state
            .cache
            .as_ref()
            .unwrap()
            .slots_for_append(4, 1)
            .unwrap()[0];
        let replacement = test_managed_cache(arena, bindings, 4, 4);

        let checkpoint = state.begin_managed_quantum(replacement).unwrap();
        let replacement_next_slot = state
            .cache
            .as_ref()
            .unwrap()
            .slots_for_append(4, 1)
            .unwrap()[0];
        assert_ne!(old_next_slot, replacement_next_slot);
        state.pos = 5;
        state.prefill_progress = 5;
        state.pending_token = None;
        state.generated_ids.push(6);
        state.finished = true;

        state.rollback_managed_quantum(checkpoint);
        assert_eq!(state.pos, 4);
        assert_eq!(state.prefill_progress, 4);
        assert_eq!(state.pending_token, Some(7));
        assert_eq!(state.generated_ids, vec![7]);
        assert!(!state.finished);
        assert_eq!(state.cache.as_ref().unwrap().context_len(), 4);
        assert_eq!(
            state
                .cache
                .as_ref()
                .unwrap()
                .slots_for_append(4, 1)
                .unwrap()[0],
            old_next_slot
        );
    }

    #[test]
    fn qwen_asr_ragged_decode_batch_matches_scalar_and_isolates_row_state() {
        let device = Device::Cpu;
        let model = tiny_qwen3_model_for_test(&device);
        let (arena, bindings) = test_managed_arena();
        let mut serial_a_cache = test_managed_cache(arena.clone(), bindings.clone(), 0, 0);
        let mut serial_b_cache = test_managed_cache(arena.clone(), bindings.clone(), 4, 0);
        let mut batch_a_cache = test_managed_cache(arena.clone(), bindings.clone(), 8, 0);
        let mut batch_b_cache = test_managed_cache(arena, bindings, 12, 0);
        let prompt_a = Tensor::from_vec(vec![0u32, 1], (1, 2), &device).unwrap();
        let prompt_b = Tensor::from_vec(vec![2u32, 3, 4], (1, 3), &device).unwrap();
        for cache in [&mut serial_a_cache, &mut batch_a_cache] {
            model.forward_managed(&prompt_a, 0, cache).unwrap();
            assert_eq!(cache.take_completed_writes().len(), 1);
        }
        for cache in [&mut serial_b_cache, &mut batch_b_cache] {
            model.forward_managed(&prompt_b, 0, cache).unwrap();
            assert_eq!(cache.take_completed_writes().len(), 1);
        }
        let mut serial_a = test_decode_state(serial_a_cache, None, 2, 2, 2, Some(5));
        let mut serial_b = test_decode_state(serial_b_cache, None, 3, 3, 3, Some(6));
        let mut batch_a = test_decode_state(batch_a_cache, None, 2, 2, 2, Some(5));
        let mut batch_b = test_decode_state(batch_b_cache, None, 3, 3, 3, Some(6));

        let scalar_a = {
            let mut rows = [&mut serial_a];
            forward_asr_pending_decode_batch(&model, &device, &mut rows).unwrap()
        };
        let scalar_b = {
            let mut rows = [&mut serial_b];
            forward_asr_pending_decode_batch(&model, &device, &mut rows).unwrap()
        };
        let batched = {
            let mut rows = [&mut batch_a, &mut batch_b];
            forward_asr_pending_decode_batch(&model, &device, &mut rows).unwrap()
        };

        assert_tensor_close(&scalar_a, &batched.i(0).unwrap().unsqueeze(0).unwrap());
        assert_tensor_close(&scalar_b, &batched.i(1).unwrap().unsqueeze(0).unwrap());
        assert_eq!(batch_a.pos, 3);
        assert_eq!(batch_b.pos, 4);
        assert_eq!(batch_a.cache.as_ref().unwrap().context_len(), 3);
        assert_eq!(batch_b.cache.as_ref().unwrap().context_len(), 4);
        assert_eq!(batch_a.pending_token, None);
        assert_eq!(batch_b.pending_token, None);
        assert!(batch_a.generated_ids.is_empty());
        assert!(batch_b.generated_ids.is_empty());
        let completion_a = batch_a.cache.as_mut().unwrap().take_completed_writes();
        let completion_b = batch_b.cache.as_mut().unwrap().take_completed_writes();
        assert_eq!(completion_a.len(), 1);
        assert_eq!(completion_b.len(), 1);
        assert!(Arc::ptr_eq(&completion_a[0], &completion_b[0]));
    }

    #[test]
    fn qwen_asr_mixed_batch_rollback_restores_only_the_cancelled_row() {
        let device = Device::Cpu;
        let model = tiny_qwen3_model_for_test(&device);
        let (arena, bindings) = test_managed_arena();
        let mut a_cache = test_managed_cache(arena.clone(), bindings.clone(), 0, 0);
        let mut b_cache = test_managed_cache(arena.clone(), bindings.clone(), 4, 0);
        let prompt_a = Tensor::from_vec(vec![0u32, 1], (1, 2), &device).unwrap();
        let prompt_b = Tensor::from_vec(vec![2u32, 3, 4], (1, 3), &device).unwrap();
        model.forward_managed(&prompt_a, 0, &mut a_cache).unwrap();
        model.forward_managed(&prompt_b, 0, &mut b_cache).unwrap();
        assert_eq!(a_cache.take_completed_writes().len(), 1);
        assert_eq!(b_cache.take_completed_writes().len(), 1);

        let mut cancelled = test_decode_state(a_cache, None, 2, 2, 2, Some(5));
        let mut live = test_decode_state(b_cache, None, 3, 3, 3, Some(6));
        let cancelled_checkpoint = cancelled
            .begin_managed_quantum(test_managed_cache(arena.clone(), bindings.clone(), 8, 2))
            .unwrap();
        let live_checkpoint = live
            .begin_managed_quantum(test_managed_cache(arena, bindings, 12, 3))
            .unwrap();

        let output = {
            let mut rows = [&mut cancelled, &mut live];
            forward_asr_pending_decode_batch(&model, &device, &mut rows).unwrap()
        };
        assert_eq!(output.dims()[0], 2);

        // Model execution was shared, but cancellation resolves under each
        // row's independent checkpoint before the engine publishes receipts.
        cancelled.rollback_managed_quantum(cancelled_checkpoint);
        drop(live_checkpoint);

        assert_eq!(cancelled.pos, 2);
        assert_eq!(cancelled.prefill_progress, 2);
        assert_eq!(cancelled.pending_token, Some(5));
        assert_eq!(cancelled.cache.as_ref().unwrap().context_len(), 2);
        assert!(cancelled
            .cache
            .as_mut()
            .unwrap()
            .take_completed_writes()
            .is_empty());

        assert_eq!(live.pos, 4);
        assert_eq!(live.prefill_progress, 3);
        assert_eq!(live.pending_token, None);
        assert_eq!(live.cache.as_ref().unwrap().context_len(), 4);
        assert_eq!(
            live.cache.as_mut().unwrap().take_completed_writes().len(),
            1
        );
    }

    #[test]
    fn qwen_asr_decode_batch_validation_preserves_all_rows() {
        let device = Device::Cpu;
        let model = tiny_qwen3_model_for_test(&device);
        let (arena, bindings) = test_managed_arena();
        let mut a = test_decode_state(
            test_managed_cache(arena.clone(), bindings.clone(), 0, 2),
            None,
            2,
            2,
            2,
            Some(5),
        );
        let mut b = test_decode_state(
            test_managed_cache(arena, bindings, 4, 3),
            None,
            3,
            3,
            3,
            None,
        );
        let mut rows = [&mut a, &mut b];
        let error = forward_asr_pending_decode_batch(&model, &device, &mut rows)
            .expect_err("missing row token must fail before any forward");
        assert!(error.to_string().contains("no scheduled input token"));
        assert_eq!(a.pending_token, Some(5));
        assert_eq!(b.pending_token, None);
        assert_eq!(a.pos, 2);
        assert_eq!(b.pos, 3);
        assert_eq!(a.cache.as_ref().unwrap().context_len(), 2);
        assert_eq!(b.cache.as_ref().unwrap().context_len(), 3);
    }

    #[test]
    fn gguf_qmatmul_text_defaults_on_with_env_override() {
        assert!(qwen3_asr_gguf_qmatmul_text_policy(None));
        assert!(qwen3_asr_gguf_qmatmul_text_policy(Some(true)));
        assert!(!qwen3_asr_gguf_qmatmul_text_policy(Some(false)));
    }

    #[test]
    fn collect_stop_token_ids_deduplicates_alt_eos() {
        let specials = SpecialTokenIds {
            im_start: 1,
            im_end: 2,
            audio_start: 3,
            audio_end: 4,
            audio_token: 5,
            timestamp: Some(12),
            asr_text: Some(7),
            fim_prefix: Some(8),
            fim_middle: Some(9),
            fim_suffix: Some(10),
            fim_pad: Some(11),
            eos: 6,
            eos_alt: Some(6),
            pad: 0,
        };
        let stop_ids = collect_stop_token_ids(&specials);
        assert_eq!(stop_ids, vec![2, 6]);
    }

    fn test_specials(asr_text: Option<u32>) -> SpecialTokenIds {
        SpecialTokenIds {
            im_start: 1,
            im_end: 2,
            audio_start: 3,
            audio_end: 4,
            audio_token: 5,
            timestamp: Some(12),
            asr_text,
            fim_prefix: Some(8),
            fim_middle: Some(9),
            fim_suffix: Some(10),
            fim_pad: Some(11),
            eos: 6,
            eos_alt: Some(13),
            pad: 0,
        }
    }

    fn encode_prompt_piece_for_test(text: &str) -> Result<Vec<u32>> {
        match text {
            "system\n" => Ok(vec![20]),
            "\n" => Ok(vec![21]),
            "user\n" => Ok(vec![22]),
            "assistant\n" => Ok(vec![23]),
            "language " => Ok(vec![24]),
            "English" => Ok(vec![25]),
            "spell Izwi correctly" => Ok(vec![26, 27, 28]),
            other => Err(Error::TokenizationError(format!(
                "unexpected test prompt piece: {other}"
            ))),
        }
    }

    #[test]
    fn qwen_asr_prompt_inserts_system_prompt_and_asr_text_marker() {
        let prompt = build_qwen_asr_prompt_tokens(
            &test_specials(Some(7)),
            2,
            Some("english"),
            Some("  spell Izwi correctly  "),
            encode_prompt_piece_for_test,
        )
        .expect("prompt");

        assert_eq!(
            prompt.ids,
            vec![1, 20, 26, 27, 28, 2, 21, 1, 22, 3, 5, 5, 4, 2, 21, 1, 23, 24, 25, 7]
        );
        assert_eq!(prompt.audio_pad_start, 10);
        assert_eq!(prompt.audio_pad_len, 2);
        assert!(prompt.system_prompt_requested);
        assert_eq!(prompt.system_prompt_token_count, 3);
        assert_eq!(prompt.forced_language.as_deref(), Some("English"));
        assert_eq!(prompt.forced_language_token_count, 3);
        assert!(prompt.asr_text_token_available);
        assert!(prompt.asr_text_token_appended);
    }

    #[test]
    fn qwen_asr_prompt_preserves_markerless_forced_language_fallback() {
        let prompt = build_qwen_asr_prompt_tokens(
            &test_specials(None),
            1,
            Some("english"),
            Some("   "),
            encode_prompt_piece_for_test,
        )
        .expect("prompt");

        assert_eq!(
            prompt.ids,
            vec![1, 20, 2, 21, 1, 22, 3, 5, 4, 2, 21, 1, 23, 24, 25]
        );
        assert_eq!(prompt.audio_pad_start, 7);
        assert!(!prompt.system_prompt_requested);
        assert_eq!(prompt.system_prompt_token_count, 0);
        assert_eq!(prompt.forced_language.as_deref(), Some("English"));
        assert_eq!(prompt.forced_language_token_count, 2);
        assert!(!prompt.asr_text_token_available);
        assert!(!prompt.asr_text_token_appended);
    }

    #[test]
    fn qwen_asr_diagnostics_json_shape_is_stable() {
        let diagnostics = Qwen3AsrDiagnostics {
            input_sample_rate: 44_100,
            input_samples: 44_100,
            resampled_sample_rate: 16_000,
            resampled_samples: 16_000,
            mel_frames: 100,
            mel_frames_before_drop: 121,
            mel_frames_before_truncate: 120,
            mel_last_frame_dropped: true,
            mel_frames_truncated: true,
            audio_tokens: 50,
            prompt_tokens: 64,
            system_prompt_requested: true,
            system_prompt_token_count: 4,
            forced_language: Some("English".to_string()),
            forced_language_token_count: 3,
            asr_text_token_available: true,
            asr_text_token_appended: true,
            max_new_tokens: 128,
            generated_tokens: 12,
            execution: Qwen3AsrExecutionDiagnostics {
                device_kind: "Cuda".to_string(),
                audio_dtype: "F16".to_string(),
                text_dtype: "F16".to_string(),
                checkpoint_format: "gguf".to_string(),
                flash_attention_requested: true,
                flash_attention_compiled: true,
                kv_page_size: 256,
                dense_decode_enabled: true,
                dense_decode_max_tokens: 1024,
                gguf_qmatmul_text_enabled: true,
                text_projection_backend: "qmatmul".to_string(),
                text_projection_quantized: true,
                qmatmul_projection_count: 197,
                dense_projection_count: 0,
                dense_bias_projection_count: 0,
                audio_precision: serde_json::json!({ "loaded_dtype": "F16" }),
                text_precision: serde_json::json!({ "loaded_dtype": "F16" }),
            },
            profile: Qwen3AsrProfileDiagnostics {
                qwen3_profile_enabled: true,
                qmatmul_calls: 10,
                qmatmul_ms: 6.5,
                qmatmul_input_casts: 4,
                qmatmul_input_cast_ms: 0.75,
                qmatmul_output_casts: 4,
                qmatmul_output_cast_ms: 0.5,
                lm_head_calls: 2,
                lm_head_ms: 1.25,
                silu_mul_fused_calls: 3,
                silu_mul_fallback_calls: 1,
                argmax_calls: 2,
                argmax_ms: 0.2,
            },
            timings: Qwen3AsrTimingDiagnostics {
                resample_ms: 1.0,
                mel_ms: 2.0,
                mel_flatten_upload_ms: 0.25,
                audio_encode_ms: 3.0,
                prefill_ms: 4.0,
                decode_ms: 5.0,
                total_ms: 15.0,
            },
        }
        .to_json();

        assert_eq!(diagnostics["model_family"], "qwen3_asr");
        assert_eq!(diagnostics["audio"]["mel_frames"], 100);
        assert_eq!(diagnostics["audio"]["mel_frames_before_drop"], 121);
        assert_eq!(diagnostics["audio"]["mel_last_frame_dropped"], true);
        assert_eq!(diagnostics["prompt"]["system_prompt_requested"], true);
        assert_eq!(diagnostics["prompt"]["asr_text_token_appended"], true);
        assert_eq!(diagnostics["decode"]["generated_tokens"], 12);
        assert_eq!(diagnostics["execution"]["device_kind"], "Cuda");
        assert_eq!(diagnostics["execution"]["checkpoint_format"], "gguf");
        assert_eq!(
            diagnostics["execution"]["text_projection_backend"],
            "qmatmul"
        );
        assert_eq!(diagnostics["execution"]["qmatmul_projection_count"], 197);
        assert_eq!(diagnostics["timings_ms"]["mel_flatten_upload"], 0.25);
        assert_eq!(diagnostics["profile"]["qwen3_profile_enabled"], true);
        assert_eq!(diagnostics["profile"]["qmatmul_calls"], 10);
        assert_eq!(diagnostics["profile"]["silu_mul_fused_calls"], 3);
        assert_eq!(diagnostics["profile"]["argmax_calls"], 2);
        assert_eq!(diagnostics["timings_ms"]["model_total"], 15.0);
    }

    #[test]
    fn forced_language_name_ignores_auto_and_empty() {
        assert_eq!(forced_language_name(None), None);
        assert_eq!(forced_language_name(Some("")), None);
        assert_eq!(forced_language_name(Some("Auto")), None);
        assert_eq!(
            forced_language_name(Some("english")),
            Some("English".to_string())
        );
        assert_eq!(
            forced_language_name(Some("en")),
            Some("English".to_string())
        );
    }

    #[test]
    fn gguf_output_splits_on_the_asr_text_marker_like_upstream() {
        let tokenizer = super::tokenizer::gguf_shaped_test_tokenizer(Some("qwen3_asr")).unwrap();
        let specials = tokenizer.specials().clone();
        let marker = specials.asr_text.expect("<asr_text> resolved");
        let generate = |meta: &str, text: &str| {
            let mut ids = tokenizer.encode_text(meta).unwrap();
            ids.push(marker);
            ids.extend(tokenizer.encode_text(text).unwrap());
            ids.push(specials.eos);
            decode_raw_asr_output(&tokenizer, &specials, &ids).unwrap()
        };
        // Expected values follow qwen_asr.inference.utils.parse_asr_output.
        let cases = [
            ("language Chinese", "你好", Some("Chinese"), "你好"),
            ("language English", "and then", Some("English"), "and then"),
            ("language English", "123", Some("English"), "123"),
            ("language None", "", None, ""),
        ];
        for (meta, text, language, expected) in cases {
            let raw = generate(meta, text);
            let (parsed_language, parsed) = parse_asr_output(&raw, None);
            assert_eq!(parsed_language.as_deref(), language, "{raw}");
            assert_eq!(parsed, expected, "{raw}");
        }
        // Forced language: the prompt ends in "language German<asr_text>", so
        // the model emits text only and it is returned as-is.
        let mut ids = tokenizer.encode_text("Hallo Welt").unwrap();
        ids.push(specials.eos);
        let raw = decode_raw_asr_output(&tokenizer, &specials, &ids).unwrap();
        let (language, text) = parse_asr_output(&raw, Some("german"));
        assert_eq!(language.as_deref(), Some("German"));
        assert_eq!(text, "Hallo Welt");
    }

    #[test]
    fn parse_asr_output_extracts_detected_language_and_text() {
        let (language, text) = parse_asr_output(
            "language Chinese<asr_text>甚至出现交易几乎停滞的情况。",
            None,
        );
        assert_eq!(language.as_deref(), Some("Chinese"));
        assert_eq!(text, "甚至出现交易几乎停滞的情况。");
    }

    #[test]
    fn parse_asr_output_respects_forced_language() {
        let (language, text) = parse_asr_output("hello world", Some("english"));
        assert_eq!(language.as_deref(), Some("English"));
        assert_eq!(text, "hello world");
    }

    #[test]
    fn parse_asr_output_respects_forced_language_and_strips_prefix() {
        let (language, text) =
            parse_asr_output("language EnglishThe quick brown fox.", Some("english"));
        assert_eq!(language.as_deref(), Some("English"));
        assert_eq!(text, "The quick brown fox.");
    }

    #[test]
    fn parse_asr_output_strips_partial_confusable_forced_language_echo() {
        let (language, text) = parse_asr_output("ɡlishThe quick brown fox.", Some("english"));
        assert_eq!(language.as_deref(), Some("English"));
        assert_eq!(text, "The quick brown fox.");
    }

    #[test]
    fn parse_asr_output_handles_empty_audio_marker() {
        let (language, text) = parse_asr_output("language None<asr_text>", None);
        assert_eq!(language, None);
        assert!(text.is_empty());
    }

    #[test]
    fn parse_asr_output_handles_markerless_language_prefix_with_space() {
        let (language, text) = parse_asr_output("language English The quick brown fox.", None);
        assert_eq!(language.as_deref(), Some("English"));
        assert_eq!(text, "The quick brown fox.");
    }

    #[test]
    fn parse_asr_output_handles_markerless_language_prefix_without_space() {
        let (language, text) = parse_asr_output("language EnglishThe quick brown fox.", None);
        assert_eq!(language.as_deref(), Some("English"));
        assert_eq!(text, "The quick brown fox.");
    }

    #[test]
    fn parse_asr_dtype_handles_common_aliases() {
        assert_eq!(parse_asr_dtype(Some("bf16")), Some(DType::BF16));
        assert_eq!(parse_asr_dtype(Some("bfloat16")), Some(DType::BF16));
        assert_eq!(parse_asr_dtype(Some("fp16")), Some(DType::F16));
        assert_eq!(parse_asr_dtype(Some("float32")), Some(DType::F32));
        assert_eq!(parse_asr_dtype(Some("unknown")), None);
    }

    #[test]
    fn qwen_asr_cuda_gguf_audio_defaults_to_f32() {
        assert_eq!(
            qwen3_asr_default_audio_dtype(DeviceKind::Cuda, true, DType::BF16),
            DType::F32
        );
        assert_eq!(
            qwen3_asr_default_audio_dtype(DeviceKind::Cuda, true, DType::F16),
            DType::F32
        );
        assert_eq!(
            qwen3_asr_default_audio_dtype(DeviceKind::Cuda, false, DType::BF16),
            DType::F32
        );
    }

    #[test]
    fn qwen_asr_non_cuda_gguf_audio_policy_is_unchanged() {
        assert_eq!(
            qwen3_asr_default_audio_dtype(DeviceKind::Metal, true, DType::F16),
            DType::F16
        );
        assert_eq!(
            qwen3_asr_default_audio_dtype(DeviceKind::Cpu, true, DType::F32),
            DType::F32
        );
    }

    fn qwen3_test_mel(n_mels: usize) -> MelSpectrogram {
        MelSpectrogram::new(MelConfig {
            sample_rate: QWEN3_ASR_SAMPLE_RATE as usize,
            n_fft: 400,
            win_length: None,
            hop_length: 160,
            n_mels,
            f_min: 0.0,
            f_max: 8_000.0,
            normalize: false,
            mel_scale: MelScale::Slaney,
            mel_norm: MelNorm::Slaney,
        })
        .expect("Qwen3 mel frontend")
    }

    /// QFA-4: 24 kHz fox.wav (3.6 s) gives WhisperFeatureExtractor's 360
    /// frames, not the 361 centre-padded STFT frames.
    #[test]
    fn qwen3_mel_keeps_the_whisper_frame_count() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/fox.wav");
        let (audio, rate) =
            crate::audio::decode_audio_bytes_to_mono(&std::fs::read(path).expect("fox.wav"))
                .expect("decode fox.wav");
        assert_eq!((audio.len(), rate), (86_400, 24_000));
        let resampled = resample(&audio, rate, QWEN3_ASR_SAMPLE_RATE).expect("resample");
        let mel = qwen3_test_mel(128);
        let (flat, frames) = qwen3_log_mel(&mel, &resampled).expect("mel");
        assert_eq!(frames, 360);
        assert_eq!(flat.len(), 360 * 128);
        let geometry = qwen3_asr_frontend_geometry(
            audio.len(),
            rate,
            QWEN3_ASR_SAMPLE_RATE,
            400,
            160,
            false,
            0,
        )
        .unwrap();
        assert_eq!(geometry.mel_frames, 360);
        assert_eq!(geometry.mel_frames_before_drop, 361);
    }

    /// The `max - 8` floor comes from the kept frames only: a burst at the
    /// very end is loudest in the dropped frame, which must not raise the
    /// floor of the others.
    #[test]
    fn qwen3_mel_floor_ignores_the_dropped_frame() {
        let mut audio = vec![0.0f32; 16_000];
        for (idx, sample) in audio[15_960..].iter_mut().enumerate() {
            *sample = if idx % 2 == 0 { 0.9 } else { -0.9 };
        }
        let mel = qwen3_test_mel(16);
        let (raw, frames) = mel.compute_flat(&audio).expect("raw mel");
        let max = |values: &[f32]| values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let max_all = max(&raw);
        let max_kept = max(&raw[..(frames - 1) * 16]);
        assert!(max_all > max_kept + 1.0, "{max_all} vs {max_kept}");
        let (features, kept) = qwen3_log_mel(&mel, &audio).expect("features");
        assert_eq!(kept, frames - 1);
        let floor = features.iter().copied().fold(f32::INFINITY, f32::min);
        assert_eq!(floor, (max_kept - 8.0 + 4.0) / 4.0);
    }

    #[test]
    fn exact_frontend_geometry_matches_resample_and_mel_boundaries() {
        let mel = MelSpectrogram::new(MelConfig {
            sample_rate: QWEN3_ASR_SAMPLE_RATE as usize,
            n_fft: 400,
            win_length: None,
            hop_length: 160,
            n_mels: 8,
            f_min: 0.0,
            f_max: 8_000.0,
            normalize: false,
            mel_scale: MelScale::Slaney,
            mel_norm: MelNorm::Slaney,
        })
        .expect("test mel frontend");

        for sample_rate in [8_000u32, QWEN3_ASR_SAMPLE_RATE, 48_000] {
            for input_samples in [0usize, 1, 159, 160, 161, 319, 320, 321, 479, 480, 481] {
                let audio = (0..input_samples)
                    .map(|index| (index as f32 * 0.017).sin())
                    .collect::<Vec<_>>();
                let resampled =
                    resample(&audio, sample_rate, QWEN3_ASR_SAMPLE_RATE).expect("resample");
                let (_, actual_frames) = mel.compute_flat(&resampled).expect("mel frames");
                let (_, kept_frames) = qwen3_log_mel(&mel, &resampled).expect("kept frames");

                let geometry = qwen3_asr_frontend_geometry(
                    input_samples,
                    sample_rate,
                    QWEN3_ASR_SAMPLE_RATE,
                    mel.config().n_fft,
                    mel.config().hop_length,
                    false,
                    0,
                )
                .expect("frontend geometry");
                assert_eq!(geometry.resampled_samples, resampled.len());
                assert_eq!(geometry.mel_frames_before_drop, actual_frames);
                assert_eq!(geometry.mel_frames, kept_frames);
                assert_eq!(kept_frames, actual_frames - 1);
            }
        }
    }

    #[test]
    fn exact_frontend_geometry_matches_odd_fft_center_padding() {
        let mel = MelSpectrogram::new(MelConfig {
            sample_rate: QWEN3_ASR_SAMPLE_RATE as usize,
            n_fft: 5,
            win_length: None,
            hop_length: 2,
            n_mels: 2,
            f_min: 0.0,
            f_max: 8_000.0,
            normalize: false,
            mel_scale: MelScale::Slaney,
            mel_norm: MelNorm::Slaney,
        })
        .expect("odd FFT mel frontend");
        for samples in 0usize..=8 {
            let audio = vec![0.25f32; samples];
            let (_, actual) = mel.compute_flat(&audio).expect("mel frames");
            assert_eq!(
                qwen3_asr_mel_frame_count(samples, 5, 2).expect("frame geometry"),
                actual
            );
        }
    }

    #[test]
    fn prepared_audio_tensor_accounting_is_exact_and_clone_safe() {
        let padded =
            Tensor::zeros((3, 20, 4), DType::F32, &Device::Cpu).expect("padded batch embeddings");
        let exact_row = padded
            .i(1)
            .and_then(|row| row.narrow(0, 0, 13))
            .and_then(|row| row.unsqueeze(0))
            .and_then(|row| deep_copy_tensor_storage(&row))
            .expect("compact prepared row");
        let artifact = Qwen3AsrPreparedAudio {
            preparation_id: 7,
            audio_embeddings: exact_row,
            frontend: Qwen3AsrPreparedFrontend::default(),
        };
        let retained = artifact.clone();
        assert_eq!(artifact.audio_tokens().expect("tokens"), 13);
        assert_eq!(artifact.resident_tensor_bytes().expect("bytes"), 13 * 4 * 4);
        assert_tensor_close(&artifact.audio_embeddings, &retained.audio_embeddings);
    }

    #[test]
    fn resample_preserves_identity_when_rates_match() {
        let input = vec![0.0f32, 0.25, -0.5, 1.0];
        let output = resample(&input, 16_000, 16_000).expect("resample");
        assert_eq!(output, input);
    }

    #[test]
    fn resample_uses_expected_duration_for_up_and_down_sampling() {
        let input = (0..100).map(|idx| idx as f32 / 100.0).collect::<Vec<_>>();
        let down = resample(&input, 100, 50).expect("downsample");
        let up = resample(&input, 50, 100).expect("upsample");

        assert_eq!(down.len(), 50);
        assert_eq!(up.len(), 200);
        assert!(down.iter().all(|value| value.is_finite()));
        assert!(up.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn resample_rejects_zero_sample_rates() {
        let input = vec![0.0f32, 1.0];
        let err = resample(&input, 0, 16_000).expect_err("zero source rate");
        assert!(format!("{err}").contains("Invalid sample rate"));

        let err = resample(&input, 16_000, 0).expect_err("zero destination rate");
        assert!(format!("{err}").contains("Invalid sample rate"));
    }

    #[test]
    fn text_delta_finds_suffix_when_prefix_changes() {
        assert_eq!(text_delta("Hello", "Hello world"), " world");
        assert_eq!(text_delta("abcd", "abXY"), "XY");
    }

    #[test]
    fn fix_timestamp_sequence_recovers_monotonicity() {
        let repaired = fix_timestamp_sequence(&[100.0, 80.0, 120.0, 60.0, 140.0]);
        assert_eq!(repaired.len(), 5);
        for pair in repaired.windows(2) {
            assert!(pair[0] <= pair[1]);
        }
        // Three or more anomalies are interpolated and truncated like int()
        // (the reference `fix_timestamp` gives the same).
        assert_eq!(
            fix_timestamp_sequence(&[0.0, 90.0, 80.0, 70.0, 60.0, 100.0]),
            [0, 90, 92, 95, 97, 100]
        );
    }

    /// QFA-5: izwi's post-processing of the reference's raw timestamp
    /// classes gives exactly the reference's word timings, zero-length words
    /// and LIS repairs included. Classes and `fixed_ms` from qwen-asr 0.0.6
    /// on test.wav (windowed and official encoder) and fox.wav (windowed),
    /// `01-aligner-reference-{test,fox}.json` of the 2026-10-14 audit.
    #[test]
    fn post_processing_returns_the_reference_timings() {
        const TEST_WINDOWED: [u32; 134] = [
            2, 4, 4, 7, 7, 11, 11, 15, 15, 20, 21, 22, 22, 30, 32, 33, 33, 42, 42, 44, 44, 46, 53,
            55, 55, 68, 69, 71, 72, 76, 76, 78, 78, 83, 83, 92, 92, 94, 94, 98, 106, 109, 109, 113,
            113, 115, 115, 120, 120, 123, 124, 134, 135, 142, 151, 160, 160, 163, 163, 167, 167,
            170, 170, 174, 174, 183, 185, 187, 187, 195, 195, 204, 204, 210, 212, 214, 214, 221,
            221, 224, 224, 231, 234, 235, 235, 241, 241, 249, 251, 258, 257, 259, 259, 266, 266,
            268, 268, 276, 276, 285, 286, 287, 287, 290, 290, 296, 299, 301, 301, 304, 304, 308,
            308, 316, 316, 321, 321, 329, 331, 333, 333, 337, 338, 348, 347, 350, 350, 350, 350,
            357, 357, 359, 359, 369,
        ];
        const TEST_WINDOWED_FIXED_MS: [u32; 134] = [
            160, 320, 320, 560, 560, 880, 880, 1200, 1200, 1600, 1680, 1760, 1760, 2400, 2560,
            2640, 2640, 3360, 3360, 3520, 3520, 3680, 4240, 4400, 4400, 5440, 5520, 5680, 5760,
            6080, 6080, 6240, 6240, 6640, 6640, 7360, 7360, 7520, 7520, 7840, 8480, 8720, 8720,
            9040, 9040, 9200, 9200, 9600, 9600, 9840, 9920, 10720, 10800, 11360, 12080, 12800,
            12800, 13040, 13040, 13360, 13360, 13600, 13600, 13920, 13920, 14640, 14800, 14960,
            14960, 15600, 15600, 16320, 16320, 16800, 16960, 17120, 17120, 17680, 17680, 17920,
            17920, 18480, 18720, 18800, 18800, 19280, 19280, 19920, 20080, 20640, 20640, 20720,
            20720, 21280, 21280, 21440, 21440, 22080, 22080, 22800, 22880, 22960, 22960, 23200,
            23200, 23680, 23920, 24080, 24080, 24320, 24320, 24640, 24640, 25280, 25280, 25680,
            25680, 26320, 26480, 26640, 26640, 26960, 27040, 27840, 27840, 28000, 28000, 28000,
            28000, 28560, 28560, 28720, 28720, 29520,
        ];
        // The official (whole-clip encoder) classes differ in five places.
        let mut test_official = TEST_WINDOWED;
        let mut test_official_fixed_ms = TEST_WINDOWED_FIXED_MS;
        for (idx, class, fixed) in [
            (9, 21, 1680),
            (26, 68, 5440),
            (111, 307, 24560),
            (123, 347, 27760),
            (124, 347, 27760),
        ] {
            test_official[idx] = class;
            test_official_fixed_ms[idx] = fixed;
        }
        let fox_windowed = [
            3, 4, 4, 7, 7, 13, 13, 19, 20, 25, 25, 28, 28, 29, 29, 35, 35, 41,
        ];
        let fox_windowed_fixed_ms = [
            240, 320, 320, 560, 560, 1040, 1040, 1520, 1600, 2000, 2000, 2240, 2240, 2320, 2320,
            2800, 2800, 3280,
        ];
        for (classes, fixed_ms) in [
            (&TEST_WINDOWED[..], &TEST_WINDOWED_FIXED_MS[..]),
            (&test_official[..], &test_official_fixed_ms[..]),
            (&fox_windowed[..], &fox_windowed_fixed_ms[..]),
        ] {
            let expected = fixed_ms
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&[start, end]| (start, end))
                .collect::<Vec<_>>();
            assert_eq!(reference_word_timings(classes, 80), expected);
        }
        // test.wav's word 63 is zero-length in the reference.
        assert_eq!(
            reference_word_timings(&TEST_WINDOWED, 80)[63],
            (28_000, 28_000)
        );
    }

    #[test]
    fn aligner_checkpoints_must_declare_their_timestamp_config() {
        assert_eq!(
            forced_aligner_timestamp_config(true, Some(151_705), Some(80)).unwrap(),
            (Some(151_705), Some(80))
        );
        assert_eq!(
            forced_aligner_timestamp_config(false, None, None).unwrap(),
            (None, None)
        );
        for (token, segment) in [
            (None, Some(80)),
            (Some(151_705), None),
            (Some(151_705), Some(0)),
        ] {
            assert!(matches!(
                forced_aligner_timestamp_config(true, token, segment),
                Err(Error::ModelLoadError(_))
            ));
        }
    }

    #[test]
    fn qwen_asr_gguf_filename_matches_new_variants() {
        assert_eq!(
            qwen_asr_gguf_filename(ModelVariant::Qwen3Asr06BGguf),
            Some("qwen3_asr_0.6b_q8_0.gguf")
        );
        assert_eq!(
            qwen_asr_gguf_filename(ModelVariant::Qwen3Asr17BGguf),
            Some("qwen3_asr_1.7b_q8_0.gguf")
        );
        assert_eq!(
            qwen_asr_gguf_filename(ModelVariant::WhisperLargeV3Turbo),
            None
        );
    }

    #[test]
    fn qwen_asr_gguf_audio_tensor_filter_matches_prefix_style() {
        assert!(qwen3_asr_gguf_audio_tensor_matches(
            "thinker.audio_tower.conv2d1.weight",
            "thinker"
        ));
        assert!(!qwen3_asr_gguf_audio_tensor_matches(
            "thinker.model.layers.0.self_attn.q_proj.weight",
            "thinker"
        ));
        assert!(qwen3_asr_gguf_audio_tensor_matches(
            "audio_tower.conv2d1.weight",
            ""
        ));
        assert!(!qwen3_asr_gguf_audio_tensor_matches(
            "model.layers.0.self_attn.q_proj.weight",
            ""
        ));
    }

    #[test]
    fn qwen_asr_gguf_text_tensor_filter_matches_prefix_style() {
        assert!(qwen3_asr_gguf_text_tensor_matches(
            "thinker.model.layers.0.self_attn.q_proj.weight",
            "thinker"
        ));
        assert!(qwen3_asr_gguf_text_tensor_matches(
            "thinker.lm_head.weight",
            "thinker"
        ));
        assert!(!qwen3_asr_gguf_text_tensor_matches(
            "thinker.audio_tower.conv2d1.weight",
            "thinker"
        ));
        assert!(qwen3_asr_gguf_text_tensor_matches(
            "model.layers.0.self_attn.q_proj.weight",
            ""
        ));
        assert!(qwen3_asr_gguf_text_tensor_matches("lm_head.weight", ""));
        assert!(!qwen3_asr_gguf_text_tensor_matches(
            "audio_tower.conv2d1.weight",
            ""
        ));
    }

    #[test]
    fn argmax_selects_max_from_rank1_logits() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::from_vec(vec![0.1f32, 3.2, -1.0, 2.7], 4, &device).expect("logits");
        let idx = argmax(&logits).expect("argmax");
        assert_eq!(idx, 1);
    }

    #[test]
    fn sampling_consumes_asr_quantum_output_without_changing_argmax() {
        let device = candle_core::Device::Cpu;
        let output = Tensor::from_vec(vec![0.1f32, 3.2, -1.0, 2.7], (1, 1, 4), &device).unwrap();
        let expected = argmax(&output.i((0, 0)).unwrap()).unwrap();
        let mut unconsumed = Some(output);

        assert_eq!(take_quantum_argmax(&mut unconsumed).unwrap(), expected);
        assert!(unconsumed.is_none());
        assert!(take_quantum_argmax(&mut unconsumed).is_err());
    }

    #[test]
    fn argmax_rejects_rank2_batched_logits() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::from_vec(vec![0.1f32; 8], (2, 4), &device).expect("logits");
        let err = argmax(&logits).expect_err("argmax should reject batched rank2");
        assert!(format!("{err}").contains("Unexpected batched logits for argmax"));
    }

    #[test]
    fn batched_decode_argmax_selects_each_rows_last_sequence_output() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::from_vec(
            vec![
                9.0f32, 1.0, 0.0, 0.0, 7.0, 2.0, // row 0, last token => 1
                0.0, 8.0, 1.0, 6.0, 0.0, 2.0, // row 1, last token => 0
            ],
            (2, 2, 3),
            &device,
        )
        .expect("logits");

        assert_eq!(batched_decode_argmax(&logits).unwrap(), vec![1, 0]);
    }

    #[test]
    fn timestamp_argmax_reads_one_class_per_scored_row() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::from_vec(
            vec![
                0.3f32, 1.4, 0.2, 0.0, //
                0.1, 0.2, 0.9, 0.0, //
                0.0, 0.2, 0.1, 2.0,
            ],
            (1, 3, 4),
            &device,
        )
        .expect("logits");

        let indices = timestamp_argmax(&logits, &[20, 7, 31]).expect("argmax");

        assert_eq!(indices, vec![1, 2, 3]);
    }

    #[test]
    fn timestamp_argmax_rejects_a_row_count_mismatch() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::from_vec(vec![0.1f32; 8], (1, 2, 4), &device).expect("logits");
        let err = timestamp_argmax(&logits, &[2])
            .expect_err("one position cannot describe two scored rows");

        assert!(format!("{err}").contains("[1, 1, classes]"), "{err}");
    }

    #[test]
    fn timestamp_argmax_rejects_non_finite_logits() {
        let device = candle_core::Device::Cpu;
        for poison in [f32::NAN, f32::INFINITY] {
            let mut values = vec![0.1f32; 12];
            values[2] = 0.9;
            values[5] = poison;
            let logits = Tensor::from_vec(values, (1, 3, 4), &device).expect("logits");

            let err = timestamp_argmax(&logits, &[4, 5, 6])
                .expect_err("a non-finite timestamp row must not become class 0");
            assert!(format!("{err}").contains("non-finite"), "{err}");
            assert!(format!("{err}").contains("prompt position 5"), "{err}");
        }
    }

    pub(super) fn local_models_root() -> PathBuf {
        std::env::var("IZWI_MODELS_DIR")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                dirs::data_local_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("izwi")
                    .join("models")
            })
    }

    /// The aligner against qwen-asr 0.0.6 on fox.wav and test.wav: the same
    /// word units, and word boundaries within 40 ms on average of the
    /// reference's windowed-encoder run (the form izwi implements).
    /// `IZWI_ALIGNER_REFERENCE_DIR` holds the audit's
    /// `01-aligner-reference-{fox,test}.json`; `IZWI_ALIGNER_DEVICE` picks
    /// cpu (default) or metal; `IZWI_ALIGNER_DUMP` names a file for the
    /// timings.
    ///
    /// Every boundary must also be within 160 ms, with the production decode
    /// or the reference's own 16 kHz audio (`IZWI_ALIGNER_AUDIO_DIR`). Both
    /// measure all equal on CPU: the canonical resampler matches soxr_hq. It
    /// once passed 7.3-8 kHz whole and moved test.wav's `AIautonomous` end by
    /// 800 ms (mean 15.5 ms over 134 boundaries).
    #[test]
    #[ignore = "requires the local Qwen3-ForcedAligner-0.6B checkpoint and IZWI_ALIGNER_REFERENCE_DIR"]
    fn forced_aligner_matches_the_reference_timings() {
        #[cfg(feature = "metal")]
        let _lease = crate::backends::metal_test_device_or_skip(
            "forced_aligner_matches_the_reference_timings",
        );
        let reference_dir = PathBuf::from(
            std::env::var("IZWI_ALIGNER_REFERENCE_DIR").expect("IZWI_ALIGNER_REFERENCE_DIR"),
        );
        let device_name = std::env::var("IZWI_ALIGNER_DEVICE").unwrap_or_else(|_| "cpu".into());
        let device = DeviceSelector::detect_with_preference(Some(&device_name)).expect("device");
        let model = Qwen3AsrModel::load(
            &local_models_root().join("Qwen3-ForcedAligner-0.6B"),
            ModelVariant::Qwen3ForcedAligner06B,
            device,
        )
        .expect("load aligner");
        let data = std::env::var("IZWI_ALIGNER_AUDIO_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data"));
        let mut dump = serde_json::Map::new();
        let mut off_target = Vec::new();
        for name in ["fox", "test"] {
            let path = reference_dir.join(format!("01-aligner-reference-{name}.json"));
            let reference: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(path).expect("reference")).unwrap();
            let text = reference["input"]["transcript"].as_str().unwrap();
            let language = reference["input"]["language"].as_str();
            let bytes = fs::read(data.join(format!("{name}.wav"))).expect("audio");
            // The production decode, which resamples to 16 kHz itself.
            let (audio, rate) =
                crate::runtime::audio_io::decode_audio_bytes_canonical(&bytes).expect("decode");
            let units = alignment_units(text, language);
            let timings = model
                .force_align_units(&audio, rate, &units)
                .expect("align");
            dump.insert(name.into(), json!(timings));
            for variant in ["windowed", "official"] {
                let words = reference["variants"][variant]["words"].as_array().unwrap();
                let texts = words
                    .iter()
                    .map(|word| word["text"].as_str().unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(
                    units
                        .iter()
                        .map(|unit| unit.text.as_str())
                        .collect::<Vec<_>>(),
                    texts,
                    "{name}: word units"
                );
                let ms = |value: &serde_json::Value| (value.as_f64().unwrap() * 1000.0).round();
                let deltas = words
                    .iter()
                    .zip(&timings)
                    .flat_map(|(word, (start, end))| {
                        [
                            (ms(&word["start"]) - f64::from(*start)).abs(),
                            (ms(&word["end"]) - f64::from(*end)).abs(),
                        ]
                    })
                    .collect::<Vec<_>>();
                let mean = deltas.iter().sum::<f64>() / deltas.len() as f64;
                let max = deltas.iter().copied().fold(0.0, f64::max);
                eprintln!(
                    "{device_name} {name} vs {variant}: mean |d| {mean:.1} ms, max {max} ms, \
                     {} of {} boundaries differ",
                    deltas.iter().filter(|delta| **delta > 0.0).count(),
                    deltas.len()
                );
                if variant == "windowed" && (mean > 40.0 || max > 160.0) {
                    off_target.push(format!("{name}: mean {mean:.1} ms, max {max} ms"));
                }
            }
        }
        if let Ok(path) = std::env::var("IZWI_ALIGNER_DUMP") {
            fs::write(path, serde_json::Value::Object(dump).to_string()).expect("dump");
        }
        assert!(off_target.is_empty(), "{off_target:?}");
    }

    #[test]
    #[ignore = "requires local Qwen3-ForcedAligner checkpoint"]
    fn forced_aligner_local_checkpoint_loads_and_aligns() {
        let model_dir = local_models_root().join("Qwen3-ForcedAligner-0.6B");
        if !model_dir.join("model.safetensors").exists() {
            eprintln!(
                "Skipping forced aligner local checkpoint test, model not found at {}",
                model_dir.display()
            );
            return;
        }

        let device = DeviceSelector::detect_with_preference(Some("cpu")).expect("cpu device");
        let model = Qwen3AsrModel::load(&model_dir, ModelVariant::Qwen3ForcedAligner06B, device)
            .expect("forced aligner should load");
        let audio = vec![0f32; 16_000];
        let alignment = model
            .force_align(&audio, 16_000, "hello world", None)
            .expect("forced align should run");
        assert!(
            !alignment.is_empty(),
            "forced aligner should return timestamps"
        );
    }

    /// fox.wav gives a 77-token prompt (77 % 32 = 13), which hit Candle's
    /// causal Metal SDPA over-read in the dense text model. Metal must match
    /// CPU per word boundary.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires the local Qwen3-ForcedAligner-0.6B checkpoint and a Metal device"]
    fn forced_aligner_metal_matches_cpu() {
        let _lease = crate::backends::metal_test_device_or_skip("forced_aligner_metal_matches_cpu");
        const MAX_BOUNDARY_DRIFT_MS: u32 = 80;
        let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
        let bytes = std::fs::read(data.join("fox.wav")).expect("fox.wav");
        let (audio, sample_rate) =
            crate::audio::decode_audio_bytes_to_mono(&bytes).expect("decode fox.wav");
        let text = std::fs::read_to_string(data.join("fox.md")).expect("fox.md");
        let text = text.trim();

        let dir_name = "Qwen3-ForcedAligner-0.6B";
        let variant = ModelVariant::Qwen3ForcedAligner06B;
        let model_dir = local_models_root().join(dir_name);
        if !model_dir.join("model.safetensors").exists() {
            eprintln!("Skipping {dir_name}: not found at {}", model_dir.display());
            return;
        }
        // One model resident at a time.
        let align_on = |device: &str| {
            let device = DeviceSelector::detect_with_preference(Some(device)).expect(device);
            let model = Qwen3AsrModel::load(&model_dir, variant, device).expect("load");
            model
                .force_align(&audio, sample_rate, text, None)
                .expect("force_align")
        };
        let cpu = align_on("cpu");
        let metal = align_on("metal");
        eprintln!("{dir_name} cpu:   {cpu:?}\n{dir_name} metal: {metal:?}");

        assert_eq!(cpu.len(), metal.len(), "{dir_name}: word count");
        for (c, m) in cpu.iter().zip(&metal) {
            assert_eq!(c.0, m.0, "{dir_name}: word order");
            let drift = c.1.abs_diff(m.1).max(c.2.abs_diff(m.2));
            assert!(
                drift <= MAX_BOUNDARY_DRIFT_MS,
                "{dir_name}: '{}' Metal {:?} vs CPU {:?}",
                c.0,
                (m.1, m.2),
                (c.1, c.2)
            );
        }
    }
}

#[cfg(all(test, feature = "metal"))]
mod metal_f16_text_tests {
    use super::tests::local_models_root;
    use super::*;
    use crate::backends::DeviceSelector;

    /// D8 gate: the dense aligner's F16 Metal text moves at most 2 of
    /// test.wav's word boundaries by more than 80 ms against F32 text
    /// (measured 2026-10-15: none moved, here and on a 175 s cut).
    #[test]
    #[ignore = "requires the local Qwen3-ForcedAligner-0.6B checkpoint and a Metal device"]
    fn dense_aligner_f16_text_keeps_test_wav_boundaries() {
        let _lease = crate::backends::metal_test_device_or_skip(
            "dense_aligner_f16_text_keeps_test_wav_boundaries",
        );
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data");
        let bytes = std::fs::read(data.join("test.wav")).expect("test.wav");
        let (audio, rate) =
            crate::audio::decode_audio_bytes_to_mono(&bytes).expect("decode test.wav");
        let text = std::fs::read_to_string(data.join("test.md")).expect("test.md");
        let align = |pin_f32: bool| {
            if pin_f32 {
                std::env::set_var("IZWI_METAL_DTYPE", "f32");
            }
            let device = DeviceSelector::detect_with_preference(Some("metal")).expect("metal");
            let model = Qwen3AsrModel::load(
                &local_models_root().join("Qwen3-ForcedAligner-0.6B"),
                ModelVariant::Qwen3ForcedAligner06B,
                device,
            );
            std::env::remove_var("IZWI_METAL_DTYPE");
            let model = model.expect("load");
            let dtype = model.text_dtype;
            let words = model
                .force_align(&audio, rate, text.trim(), None)
                .expect("align");
            (dtype, words)
        };
        let (f16_dtype, f16) = align(false);
        let (f32_dtype, f32) = align(true);
        assert_eq!((f16_dtype, f32_dtype), (DType::F16, DType::F32));
        assert_eq!(f16.len(), f32.len());
        let boundaries = f16
            .iter()
            .zip(&f32)
            .flat_map(|(a, b)| [a.1.abs_diff(b.1), a.2.abs_diff(b.2)])
            .collect::<Vec<_>>();
        let moved = boundaries.iter().filter(|delta| **delta > 80).count();
        eprintln!(
            "F16 vs F32 text: {moved}/{} boundaries moved > 80 ms, max {} ms",
            boundaries.len(),
            boundaries.iter().max().copied().unwrap_or(0)
        );
        assert!(moved <= 2, "{moved} boundaries moved > 80 ms");
    }
}

#[cfg(test)]
mod precision_tests {
    use candle_core::{DType, Device};

    use super::{
        load_qwen3_asr_components, qwen3_asr_audio_self_check, qwen3_asr_text_self_check,
        qwen3_asr_weight_precisions, select_quantized_text_dtype, Qwen3AsrModel,
    };
    use crate::backends::{DTypeSelection, DeviceCapabilities, DeviceKind, DeviceProfile};
    use crate::catalog::ModelFamily;
    use crate::kernels::numeric_check::PrecisionLoad;
    use crate::kv::{InferenceStateCapability, InferenceStateContractProvider};
    use crate::models::architectures::qwen3::core::tiny_qwen3_model_for_test;

    fn selection(dtype: DType) -> DTypeSelection {
        DTypeSelection {
            dtype,
            reason: "test selection".into(),
        }
    }

    /// A Qwen3 speech-family model over the tiny audio tower and text model.
    fn tiny_model(is_forced_aligner: bool) -> Qwen3AsrModel {
        let device = Device::Cpu;
        let tokenizer = super::tokenizer::gguf_shaped_test_tokenizer(Some("qwen3_asr")).unwrap();
        let specials = tokenizer.specials().clone();
        let mel = super::MelSpectrogram::new(super::MelConfig {
            sample_rate: 16_000,
            n_fft: 400,
            win_length: None,
            hop_length: 160,
            n_mels: 8,
            f_min: 0.0,
            f_max: 8_000.0,
            normalize: false,
            mel_scale: super::MelScale::Slaney,
            mel_norm: super::MelNorm::Slaney,
        })
        .unwrap();
        Qwen3AsrModel {
            audio_preparation_id: 0,
            device: DeviceProfile::cpu(),
            audio_dtype: DType::F32,
            text_dtype: DType::F32,
            checkpoint_format: "safetensors",
            gguf_qmatmul_text_enabled: false,
            is_forced_aligner,
            timestamp_token_id: None,
            timestamp_segment_time_ms: None,
            tokenizer,
            specials,
            audio_tower: super::audio::tiny_audio_tower_for_test(&device),
            text_model: tiny_qwen3_model_for_test(&device),
            text_context_tokens: Some(128),
            mel,
            preprocessor: serde_json::from_str("{}").unwrap(),
            audio_precision: PrecisionLoad::unchecked(&selection(DType::F32)),
            text_precision: PrecisionLoad::unchecked(&selection(DType::F32)),
        }
    }

    /// QFA-7 / ADM-11: both checkpoint formats load through one helper, so
    /// a safetensors checkpoint's lowered tier (the 4-bit aligner's F16 text
    /// on Metal) runs its load-time probe, as the GGUF one always did. F32
    /// is the reference and is never probed; off Metal nothing is.
    #[test]
    fn lowered_tiers_run_their_load_time_probe_on_metal() {
        let device = Device::Cpu;
        for check in [true, false] {
            let mut audio_loads = Vec::new();
            let mut text_loads = Vec::new();
            let (_, audio, _, text) = load_qwen3_asr_components(
                &selection(DType::F32),
                &selection(DType::F16),
                check,
                &device,
                |dtype| {
                    audio_loads.push(dtype);
                    Ok(super::audio::tiny_audio_tower_for_test(&device))
                },
                |dtype| {
                    text_loads.push(dtype);
                    Ok(tiny_qwen3_model_for_test(&device))
                },
            )
            .unwrap();
            assert_eq!(audio_loads, [DType::F32]);
            // A passing probe keeps the tier: one load, no F32 reload.
            assert_eq!(text_loads, [DType::F16]);
            assert!(audio.self_check.is_none());
            assert_eq!(text.loaded, DType::F16);
            match text.self_check {
                Some(report) => assert!(check && report.passed, "{}", report.reason),
                None => assert!(!check),
            }
        }
    }

    /// A failed text probe asks the load's reload gate for the F32 text's
    /// memory after dropping the F16 text and before reloading; a refusal
    /// stops the load without reloading.
    #[test]
    fn a_failed_probe_admits_the_f32_reload_before_loading_it() {
        use crate::kernels::numeric_check::with_precision_reload_gate;
        use crate::models::architectures::qwen3::core::tiny_non_finite_qwen3_model_for_test;
        use std::sync::{Arc, Mutex};

        let device = Device::Cpu;
        for admit in [true, false] {
            let events = Arc::new(Mutex::new(Vec::new()));
            let gate_events = events.clone();
            let gate = Box::new(move |label: &str, dtype: DType| {
                gate_events
                    .lock()
                    .unwrap()
                    .push(format!("admit {label} {dtype:?}"));
                if admit {
                    Ok(())
                } else {
                    Err(crate::error::Error::ModelLoadError("refused".into()))
                }
            });
            let outcome = with_precision_reload_gate(gate, || {
                load_qwen3_asr_components(
                    &selection(DType::F32),
                    &selection(DType::F16),
                    true,
                    &device,
                    |_| Ok(super::audio::tiny_audio_tower_for_test(&device)),
                    |dtype| {
                        events.lock().unwrap().push(format!("load text {dtype:?}"));
                        Ok(if dtype == DType::F16 {
                            tiny_non_finite_qwen3_model_for_test(&device)
                        } else {
                            tiny_qwen3_model_for_test(&device)
                        })
                    },
                )
            });
            let mut expected = vec!["load text F16", "admit Qwen3-ASR text F32"];
            match outcome {
                Ok((_, audio, _, text)) => {
                    assert!(admit);
                    expected.push("load text F32");
                    assert!(text.fell_back() && !audio.fell_back());
                }
                Err(error) => {
                    assert!(!admit);
                    assert!(error.to_string().contains("refused"), "{error}");
                }
            }
            assert_eq!(*events.lock().unwrap(), expected);
        }
    }

    /// QFA-11: the aligner's cacheless forward publishes no KV, matching its
    /// stateless `ForcedAlignment` adapter; Qwen3-ASR keeps its managed
    /// contract.
    #[test]
    fn the_forced_aligner_reports_a_stateless_contract() {
        let aligner = tiny_model(true).inference_state_contract().unwrap();
        assert!(matches!(aligner, InferenceStateCapability::Stateless));
        let asr = tiny_model(false).inference_state_contract().unwrap();
        assert!(matches!(asr, InferenceStateCapability::Managed(_)));
    }

    #[test]
    fn quantized_text_on_metal_takes_the_lm_tier_unless_f32_is_pinned() {
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let metal = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Metal,
            capabilities: DeviceCapabilities {
                supports_f16: true,
                ..Default::default()
            },
            memory_pool: None,
        };
        std::env::remove_var("IZWI_METAL_DTYPE");
        let tier = select_quantized_text_dtype(&metal, ModelFamily::Qwen3Asr, DType::BF16);
        assert_eq!(tier.dtype, DType::F16);
        assert!(tier.reason.contains("LM tier"), "{}", tier.reason);
        std::env::set_var("IZWI_METAL_DTYPE", "f32");
        let pinned = select_quantized_text_dtype(&metal, ModelFamily::Qwen3Asr, DType::BF16);
        std::env::remove_var("IZWI_METAL_DTYPE");
        assert_eq!(pinned.dtype, DType::F32);
        assert_eq!(
            select_quantized_text_dtype(&DeviceProfile::cpu(), ModelFamily::Qwen3Asr, DType::BF16)
                .dtype,
            DType::F32
        );
    }

    /// D8: dense aligner text and every aligner's audio tower take F16 on
    /// Metal (probed at load) unless `IZWI_METAL_DTYPE=f32`; dense Qwen3-ASR
    /// stays F32.
    #[test]
    fn dense_aligner_text_takes_f16_on_metal_unless_f32_is_pinned() {
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let metal = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Metal,
            capabilities: DeviceCapabilities {
                supports_f16: true,
                ..Default::default()
            },
            memory_pool: None,
        };
        let precisions = |family, quantized| {
            qwen3_asr_weight_precisions(&metal, family, false, quantized, Some("bfloat16")).unwrap()
        };
        let text = |family| precisions(family, false).0;
        std::env::remove_var("IZWI_METAL_DTYPE");
        let aligner = text(ModelFamily::Qwen3ForcedAligner);
        assert_eq!(aligner.dtype, DType::F16);
        assert!(aligner.reason.contains("probe"), "{}", aligner.reason);
        assert_eq!(text(ModelFamily::Qwen3Asr).dtype, DType::F32);
        // The aligner's audio tower takes F16 too, dense or MLX 4-bit; dense
        // Qwen3-ASR keeps an F32 tower.
        for quantized in [false, true] {
            let audio = precisions(ModelFamily::Qwen3ForcedAligner, quantized).1;
            assert_eq!(audio.dtype, DType::F16, "quantized={quantized}");
            assert!(audio.reason.contains("probe"), "{}", audio.reason);
        }
        assert_eq!(precisions(ModelFamily::Qwen3Asr, false).1.dtype, DType::F32);
        std::env::set_var("IZWI_METAL_DTYPE", "f32");
        let pinned = text(ModelFamily::Qwen3ForcedAligner);
        let pinned_audio = precisions(ModelFamily::Qwen3ForcedAligner, false).1;
        std::env::remove_var("IZWI_METAL_DTYPE");
        assert_eq!(pinned.dtype, DType::F32);
        assert_eq!(pinned_audio.dtype, DType::F32);
        assert_eq!(
            qwen3_asr_weight_precisions(
                &DeviceProfile::cpu(),
                ModelFamily::Qwen3ForcedAligner,
                false,
                false,
                Some("bfloat16")
            )
            .unwrap()
            .0
            .dtype,
            DType::F32
        );
    }

    #[test]
    fn audio_and_text_probes_run_the_real_forward_paths() {
        let device = Device::Cpu;
        let tower = super::audio::tiny_audio_tower_for_test(&device);
        let report = qwen3_asr_audio_self_check(&tower, DType::F32, &device);
        assert!(report.passed, "{}", report.reason);
        assert_eq!(report.tensors[0].name, "audio_tower");

        let model = crate::models::architectures::qwen3::core::tiny_qwen3_model_for_test(&device);
        let report = qwen3_asr_text_self_check(&model);
        assert!(report.passed, "{}", report.reason);
        assert_eq!(report.tensors[0].name, "logits");
    }
}
