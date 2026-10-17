//! Load admission for the ASR families on every backend: the residency plan of
//! the ASR admission framework (`AsrResidencyPlan`). Each family prices the
//! header census of the files its loader reads (tensor name, stored dtype and
//! shape from the safetensors, GGUF or PyTorch headers, never the payloads) in
//! the form that loader leaves resident on the selected backend. The dtypes come
//! from the loader's own resolvers, so a dtype policy change reprices admission
//! without a table to update here. One plan answers first admission and the
//! Metal OOM retry on CPU, Metal and CUDA alike; the catalog's
//! `memory_required_gb` stays a display figure for these variants.
//!
//! A plan prices, on the load backend's domain (host memory on CPU, unified
//! memory on Metal, device memory on CUDA):
//! - Resident weights: every tensor the loader reads, in its resident form (kept
//!   quantized, dequantized or cast), plus the copies it makes (Granite's F16 LM
//!   head, LFM's second Q6_K token-embedding read, Qwen3-ASR's separate Q8_0
//!   `lm_head` beside its dense embedding) and the tables it computes. Tensors it
//!   never reads are left out. On Metal, a buffer a device kernel writes (a cast,
//!   a concatenation, any op output) is sized the way Candle 0.11's pool
//!   allocates it, rounded up to the next power of two (`candle-core`
//!   `metal_backend/device.rs`, `buf_size`). Host uploads keep their exact size.
//! - The load peak: resident plus everything alive while loading. Candle keeps a
//!   Metal shared buffer it released, such as the stored-dtype upload of a cast
//!   tensor, until the next device synchronization. The PyTorch loaders and
//!   Candle's safetensors backend never synchronize before the load fence, so
//!   all their uploads stay until then; every GGUF dequantization
//!   synchronizes. Whisper, Granite and Voxtral read through
//!   `staged::metal_staged_safetensors`, which on Metal casts on the host,
//!   uploads exact sizes and synchronizes every
//!   `METAL_LOAD_SYNC_INTERVAL_BYTES` of uploads ([`Ledger::staged`]). Freed private-pool
//!   intermediates, such as the parts of a fused projection, are reused by later
//!   allocations, so only their high-water mark counts. CPU and CUDA release a
//!   buffer when it drops.
//! - CUDA host staging, which is the whole CUDA host authorization: the largest
//!   per-tensor host copy (NeMo archives stream to disk, so a first load adds
//!   no extraction buffer).
//! - The Nemotron model memo bound, resident on the load device.
//!
//! Approximations, each conservative unless noted:
//! - On CUDA every tensor is priced as staged through one host copy. Candle
//!   copies a memory-mapped safetensors payload only when it is misaligned.
//! - Candle's pool can serve a later, smaller allocation from a larger freed
//!   buffer, so a Metal resident can occupy more than priced (not
//!   conservative). The peak is unaffected because that buffer was already
//!   counted.
//! - Derived tensors under about 1 MiB per family (concatenated norm gains,
//!   scalars, decode masks, the LFM conformer BatchNorm folds) are not priced.
//! - A first NeMo load has no checkpoint header to read yet; its weights are
//!   priced from the archive's payload size (`price_nemo_first_load`).
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use candle_core::quantized::GgmlDType;
use candle_core::DType;

use super::{
    census_checkpoint_file, CensusFile, CensusTensor, ModelMemoryEstimate, ModelResourcePlan,
    StoredDType,
};
use crate::backends::{BackendKind, DeviceProfile};
use crate::catalog::ModelFamily;
use crate::engine::ResourceAmount;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::granite_speech::asr::{
    ensure_granite_speech_artifacts, granite_load_representation,
    granite_tensor_is_f16_encoder_projection, resolve_granite_encoder_projection_precision,
    select_granite_speech_dtype, GraniteLoadRepresentation, GraniteSpeechConfig,
};
use crate::models::architectures::lfm25_audio::lfm25_audio_load_layout;
use crate::models::architectures::nemotron::asr::{
    nemotron_checkpoint_paths, nemotron_tensor_is_f16_projection, resolve_nemotron_asr_dtype,
    resolve_nemotron_projection_precision, NEMOTRON_MODEL_MEMO_MAX_BYTES,
};
use crate::models::architectures::parakeet::asr::{
    parakeet_checkpoint_paths, parakeet_tensor_is_f16_projection,
    resolve_parakeet_projection_precision, PARAKEET_WEIGHT_DTYPE,
};
use crate::models::architectures::qwen3::asr::{
    qwen3_asr_use_gguf_qmatmul_text, qwen3_asr_weight_dtypes, resolve_qwen_asr_gguf_path,
    QWEN3_ASR_AUDIO_POSITIONS, QWEN3_ASR_GGUF_CHECKPOINT_DTYPE,
};
use crate::models::architectures::qwen3::core::{
    qwen3_dense_projection_fusion_enabled, qwen3_quantized_projection_grouping_for_hidden_size,
};
use crate::models::architectures::vibevoice::asr::{
    resolve_vibevoice_tokenizer_projection_precision, vibevoice_asr_audio_dtype,
    vibevoice_asr_weight_dtype, vibevoice_tensor_is_f16_tokenizer_projection,
    vibevoice_tensor_is_language_model,
};
use crate::models::architectures::voxtral::realtime::model::{
    voxtral_realtime_weight_dtype, voxtral_realtime_weight_files,
};
use crate::models::architectures::whisper::asr::{whisper_weight_dtype, whisper_weight_files};
use crate::models::shared::weights::gguf::safetensors_weight_files;
use crate::models::shared::weights::staged::METAL_LOAD_SYNC_INTERVAL_BYTES;

/// Candle 0.11's CUDA `QTensor` allocation pads every quantized buffer by this
/// many rows of blocks (`quantized/cuda.rs`, `MATRIX_ROW_PADDING`).
const CUDA_QTENSOR_ROW_PADDING: u64 = 512;

/// One tensor's load-time copies when a first NeMo load has no tensor list to
/// read: four times the largest tensor of either published NeMo ASR checkpoint
/// (Nemotron's 33.5 MB prediction embedding; Parakeet's largest is 21 MB).
const NEMO_UNLISTED_TENSOR_BYTES: u64 = 128 * 1024 * 1024;

fn overflow() -> Error {
    Error::ModelLoadError("ASR residency plan overflow".into())
}

pub(super) fn sum(values: &[u64]) -> Result<u64> {
    values
        .iter()
        .try_fold(0u64, |total, value| total.checked_add(*value))
        .ok_or_else(overflow)
}

fn dense_bytes(elements: u64, dtype: DType) -> Result<u64> {
    elements
        .checked_mul(dtype.size_in_bytes() as u64)
        .ok_or_else(overflow)
}

/// Admission for a load of the ASR `variant` checkpoint at `model_path` on
/// `device`, or `None` for a variant without a residency plan (or with weight
/// files missing, which the loader reports): such loads keep the generic
/// estimate.
pub(super) fn resource_plan(
    variant: ModelVariant,
    model_path: &Path,
    device: &DeviceProfile,
) -> Result<Option<ModelResourcePlan>> {
    let Some(plan) = residency_plan(variant, model_path, device)? else {
        return Ok(None);
    };
    tracing::info!(
        model = %variant,
        backend = ?plan.backend,
        logical_resident_bytes = plan.logical_resident_bytes,
        resident_bytes = plan.resident_bytes,
        memo_bytes = plan.memo_bytes,
        load_peak_bytes = plan.load_peak_bytes,
        host_staging_bytes = plan.host_staging_bytes,
        "Priced ASR load from its residency plan"
    );
    plan.resource_plan().map(Some)
}

/// Whether `variant` is an ASR family this module prices. Its state fits
/// spend live headroom, not only the ledger's, so the requests the state
/// serves can still run beside it.
pub(super) fn is_asr_variant(variant: ModelVariant) -> bool {
    matches!(
        variant,
        ModelVariant::WhisperLargeV3Turbo
            | ModelVariant::Qwen3Asr06BGguf
            | ModelVariant::Qwen3Asr17BGguf
            | ModelVariant::ParakeetTdt06BV3
            | ModelVariant::Nemotron35AsrStreaming06B
            | ModelVariant::VibeVoiceAsr
            | ModelVariant::VoxtralMini4BRealtime2602
            | ModelVariant::Lfm25Audio15BGguf
    ) || variant.family() == ModelFamily::GraniteSpeechAsr
}

pub(super) fn residency_plan(
    variant: ModelVariant,
    model_path: &Path,
    device: &DeviceProfile,
) -> Result<Option<AsrResidencyPlan>> {
    let backend = BackendKind::from(device.kind);
    match variant {
        ModelVariant::WhisperLargeV3Turbo => whisper_plan(model_path, device, backend),
        ModelVariant::Qwen3Asr06BGguf | ModelVariant::Qwen3Asr17BGguf => {
            qwen3_asr_plan(variant, model_path, device, backend)
        }
        ModelVariant::ParakeetTdt06BV3 => parakeet_plan(variant, model_path, device, backend),
        ModelVariant::Nemotron35AsrStreaming06B => {
            nemotron_plan(variant, model_path, device, backend)
        }
        // Plus and base share the loader; the census carries their differences
        // (base: no layer-3 concatenation in the projector, an untied LM head).
        _ if variant.family() == ModelFamily::GraniteSpeechAsr => {
            granite_plan(model_path, device, backend)
        }
        ModelVariant::VibeVoiceAsr => vibevoice_plan(model_path, device, backend),
        ModelVariant::VoxtralMini4BRealtime2602 => voxtral_plan(model_path, device, backend),
        ModelVariant::Lfm25Audio15BGguf => lfm25_audio_plan(model_path, backend),
        _ => Ok(None),
    }
}

/// What a load of one ASR checkpoint allocates on its backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AsrResidencyPlan {
    backend: BackendKind,
    /// Resident weights at their exact sizes.
    logical_resident_bytes: u64,
    /// Resident weights as the backend allocates them.
    resident_bytes: u64,
    /// Bounded model memo, resident on the load device and filled lazily.
    memo_bytes: u64,
    /// High-water mark of the load domain while the weights load, resident
    /// weights included and the memo excluded.
    load_peak_bytes: u64,
    /// CUDA only: high-water mark of host staging.
    host_staging_bytes: u64,
}

impl AsrResidencyPlan {
    /// This plan, with a load peak that also covers `fallback`: a loader
    /// that checks a lower-precision form at load and reloads densely when
    /// the check fails passes through both.
    pub(super) fn with_fallback_peak(mut self, fallback: &Self) -> Self {
        self.load_peak_bytes = self.load_peak_bytes.max(fallback.load_peak_bytes);
        self.host_staging_bytes = self.host_staging_bytes.max(fallback.host_staging_bytes);
        self
    }

    pub(super) fn from_ledger(ledger: Ledger, memo_bytes: u64) -> Result<Self> {
        let ledger = ledger.finish()?;
        Ok(Self {
            backend: ledger.backend,
            logical_resident_bytes: ledger.logical,
            resident_bytes: ledger.resident,
            memo_bytes,
            load_peak_bytes: ledger.peak,
            host_staging_bytes: ledger.host_peak,
        })
    }

    pub(super) fn estimate(&self) -> Result<ModelMemoryEstimate> {
        Ok(ModelMemoryEstimate {
            resident_bytes: sum(&[self.resident_bytes, self.memo_bytes])?,
            load_peak_bytes: sum(&[self.load_peak_bytes, self.memo_bytes])?,
        })
    }

    pub(super) fn resource_plan(&self) -> Result<ModelResourcePlan> {
        let mut plan = super::model_resource_plan(self.backend, self.estimate()?);
        if self.backend == BackendKind::Cuda {
            // The device holds the weights; the host only stages them.
            plan.load_authorization.host_bytes = ResourceAmount::Known(self.host_staging_bytes);
        }
        Ok(plan)
    }
}

/// How a loader reads a tensor's payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Source {
    /// From a memory-mapped safetensors file.
    Mapped,
    /// Through a host buffer (a PyTorch zip entry).
    Copied,
}

/// What one load step allocates beyond what the ledger already holds.
#[derive(Debug, Default)]
struct Step {
    logical: u64,
    /// Allocated bytes the step leaves resident.
    resident: u64,
    /// Load-domain bytes the step has alive at its high point, its new
    /// resident bytes included. Host buffers count on CPU and Metal (one
    /// memory); Metal's reusable private-pool intermediates do not.
    high: u64,
    /// Metal: buffers the step releases that Candle keeps until the next
    /// device synchronization (released shared buffers).
    held_until_sync: u64,
    /// Metal: private-pool intermediates, reusable by later steps.
    intermediates: u64,
    /// CUDA: host bytes alive at the step's high point.
    host: u64,
    /// Metal: load-domain bytes the step has alive when it synchronizes the
    /// device, which drops every released buffer still held.
    before_sync: Option<u64>,
}

/// Accumulates a load's steps into an upper bound of its peak that holds for
/// any order of the steps (the loaders' order is theirs to change): every step
/// may run last, beside every resident byte and whatever earlier steps left
/// for Candle to release at the next synchronization.
#[derive(Debug)]
pub(super) struct Ledger {
    backend: BackendKind,
    logical: u64,
    resident: u64,
    /// Buffers kept alive across steps ([`Self::hold`]), and their maximum.
    live: u64,
    max_live: u64,
    /// Metal: high-water mark of reusable private-pool intermediates.
    scratch: u64,
    /// Metal: bytes held until a synchronization by steps that do not
    /// synchronize first: they accumulate until the load fence.
    held_unsynced: u64,
    /// Metal: the most any one synchronizing step (or a release) leaves held;
    /// the next synchronizing step drops it.
    held_max: u64,
    /// Metal: per synchronizing step, what it leaves held and what it has
    /// alive at its synchronization (beside what an earlier step left held).
    synced: Vec<(u64, u64)>,
    /// Largest step high point beyond the step's own resident bytes, over
    /// steps that do not synchronize, and over steps that do.
    extra_max: u64,
    extra_max_synced: u64,
    /// Largest step high point beyond the step's own resident and held bytes.
    extra_beyond_held_max: u64,
    /// Metal: the most a synchronizing step has alive at its synchronization.
    before_sync_max: u64,
    /// Metal: the most held by a synchronizing step's predecessor (another
    /// step, or a release) beside its own synchronization.
    at_sync_max: u64,
    /// Load-domain buffers freed before any weight loads.
    prelude: u64,
    host_peak: u64,
    peak: u64,
    /// Metal: the loader reads through `staged::metal_staged_safetensors`,
    /// which converts each tensor on the host, uploads it at its exact size,
    /// and synchronizes the device every this many uploaded bytes.
    sync_interval: Option<u64>,
    /// Metal, staged: the most one step that does not synchronize itself
    /// leaves held.
    held_step_max: u64,
}

impl Ledger {
    pub(super) fn new(backend: BackendKind) -> Self {
        Self {
            backend,
            logical: 0,
            resident: 0,
            live: 0,
            max_live: 0,
            scratch: 0,
            held_unsynced: 0,
            held_max: 0,
            synced: Vec::new(),
            extra_max: 0,
            extra_max_synced: 0,
            extra_beyond_held_max: 0,
            before_sync_max: 0,
            at_sync_max: 0,
            prelude: 0,
            host_peak: 0,
            peak: 0,
            sync_interval: None,
            held_step_max: 0,
        }
    }

    /// A ledger for a loader that reads through
    /// `staged::metal_staged_safetensors`, which is Candle's own backend
    /// except on Metal.
    pub(super) fn staged(backend: BackendKind) -> Self {
        let mut ledger = Self::new(backend);
        if ledger.metal() {
            ledger.sync_interval = Some(METAL_LOAD_SYNC_INTERVAL_BYTES);
        }
        ledger
    }

    /// Metal: the loader converts on the host and uploads exact sizes.
    fn host_cast(&self) -> bool {
        self.sync_interval.is_some()
    }

    fn metal(&self) -> bool {
        self.backend == BackendKind::Metal
    }

    /// The size Candle allocates for `bytes` written by a device kernel.
    fn kernel(&self, bytes: u64) -> Result<u64> {
        if self.metal() {
            bytes.checked_next_power_of_two().ok_or_else(overflow)
        } else {
            Ok(bytes)
        }
    }

    fn apply(&mut self, step: Step) -> Result<()> {
        self.logical = sum(&[self.logical, step.logical])?;
        self.resident = sum(&[self.resident, step.resident])?;
        let extra = step.high.checked_sub(step.resident).ok_or_else(overflow)?;
        self.extra_beyond_held_max = self
            .extra_beyond_held_max
            .max(extra.saturating_sub(step.held_until_sync));
        self.scratch = self.scratch.max(step.intermediates);
        self.host_peak = self.host_peak.max(step.host);
        match step.before_sync.filter(|_| self.metal()) {
            Some(before_sync) => {
                self.extra_max_synced = self.extra_max_synced.max(extra);
                self.before_sync_max = self.before_sync_max.max(before_sync);
                self.held_max = self.held_max.max(step.held_until_sync);
                self.synced.push((step.held_until_sync, before_sync));
            }
            None => {
                self.extra_max = self.extra_max.max(extra);
                self.held_unsynced = sum(&[self.held_unsynced, step.held_until_sync])?;
                self.held_step_max = self.held_step_max.max(step.held_until_sync);
            }
        }
        Ok(())
    }

    /// Keep `bytes` alive, not resident, until [`Self::release`].
    fn hold(&mut self, bytes: u64) -> Result<()> {
        self.live = sum(&[self.live, bytes])?;
        self.max_live = self.max_live.max(self.live);
        Ok(())
    }

    fn release(&mut self, bytes: u64) -> Result<()> {
        self.live = self.live.checked_sub(bytes).ok_or_else(overflow)?;
        if self.metal() {
            // Held like a synchronizing step's buffers: the next
            // synchronization drops it.
            self.held_max = self.held_max.max(bytes);
            self.at_sync_max = self.at_sync_max.max(sum(&[bytes, self.before_sync_max])?);
        }
        Ok(())
    }

    /// A host buffer filled and freed before any weight loads (NeMo
    /// extraction).
    fn host_buffer(&mut self, bytes: u64) {
        if self.backend == BackendKind::Cuda {
            self.host_peak = self.host_peak.max(bytes);
        } else {
            self.prelude = self.prelude.max(bytes);
        }
    }

    /// A tensor read through a Candle `VarBuilder` at `chain[0]` (uploaded at
    /// its stored dtype, then cast on the device when they differ; a staged
    /// Metal loader casts on the host and uploads the result) and rewritten by
    /// device ops into each later dtype of `chain`, the last of which stays
    /// resident. Every form stays alive until the step ends.
    pub(super) fn var(
        &mut self,
        tensor: &CensusTensor,
        source: Source,
        chain: &[DType],
    ) -> Result<()> {
        let (&first, rest) = chain.split_first().ok_or_else(overflow)?;
        let stored = tensor.stored_bytes;
        let elements = tensor.elements();
        let cast = tensor.dtype != StoredDType::Dense(first);
        // Allocated bytes of each form; the first is the upload itself (exact
        // size) unless a device cast wrote it.
        let first_exact = dense_bytes(elements, first)?;
        if self.host_cast() {
            return self.staged_var(tensor, first_exact, cast, rest);
        }
        let mut forms = vec![if cast {
            self.kernel(first_exact)?
        } else {
            first_exact
        }];
        let mut logical = first_exact;
        for &dtype in rest {
            logical = dense_bytes(elements, dtype)?;
            forms.push(self.kernel(logical)?);
        }
        let resident = forms.pop().ok_or_else(overflow)?;
        let earlier = sum(&forms)?;
        let source_copy = if cast { stored } else { 0 };
        let step = match self.backend {
            BackendKind::Cpu => Step {
                logical,
                resident,
                high: sum(&[resident, source_copy, earlier])?,
                ..Step::default()
            },
            BackendKind::Cuda => Step {
                logical,
                resident,
                high: sum(&[resident, source_copy, earlier])?,
                host: stored,
                ..Step::default()
            },
            BackendKind::Metal => {
                // The shared upload is released when a cast or a later form
                // replaces it; cast-written earlier forms are private.
                let upload_released = if cast || !rest.is_empty() { stored } else { 0 };
                let private_earlier = if cast {
                    earlier
                } else {
                    earlier.saturating_sub(stored)
                };
                let host = if source == Source::Copied { stored } else { 0 };
                Step {
                    logical,
                    resident,
                    high: sum(&[resident, upload_released, host])?,
                    held_until_sync: upload_released,
                    intermediates: private_earlier,
                    ..Step::default()
                }
            }
        };
        self.apply(step)
    }

    /// Metal, staged: [`Self::var`] for a tensor whose host copy (stored
    /// bytes) is converted on the host when `cast` and uploaded at `upload`
    /// bytes; later forms of the chain are device ops, as there.
    fn staged_var(
        &mut self,
        tensor: &CensusTensor,
        upload: u64,
        cast: bool,
        rest: &[DType],
    ) -> Result<()> {
        let stored = tensor.stored_bytes;
        let elements = tensor.elements();
        let host = sum(&[stored, if cast { upload } else { 0 }])?;
        let mut logical = upload;
        let mut forms = vec![upload];
        for &dtype in rest {
            logical = dense_bytes(elements, dtype)?;
            forms.push(self.kernel(logical)?);
        }
        let resident = forms.pop().ok_or_else(overflow)?;
        // The upload is released once a later form replaces it; the forms
        // between are private.
        let released = if rest.is_empty() { 0 } else { upload };
        let private = sum(&forms)?.saturating_sub(released);
        self.apply(Step {
            logical,
            resident,
            high: sum(&[resident, released, host])?,
            held_until_sync: released,
            intermediates: private,
            ..Step::default()
        })
    }

    /// `VarBuilder` tensors read at `dtype` and joined by `Tensor::cat` into
    /// one resident tensor, with `zeros` extra elements the loader synthesizes
    /// (a zero bias part); the parts are dropped.
    fn fused(
        &mut self,
        parts: &[&CensusTensor],
        source: Source,
        dtype: DType,
        zeros: u64,
    ) -> Result<()> {
        let zero_bytes = dense_bytes(zeros, dtype)?;
        let mut exact = vec![zero_bytes];
        let mut private = vec![self.kernel(zero_bytes)?];
        let mut uploads = Vec::with_capacity(parts.len());
        let mut cast_sources = vec![0];
        let host_cast = self.host_cast();
        let mut host_copies = vec![0];
        for part in parts {
            let bytes = dense_bytes(part.elements(), dtype)?;
            exact.push(bytes);
            if host_cast {
                // Uploaded at `dtype`, exact size, after a host copy and cast.
                uploads.push(bytes);
                host_copies.push(sum(&[part.stored_bytes, bytes])?);
                continue;
            }
            uploads.push(part.stored_bytes);
            if part.dtype != StoredDType::Dense(dtype) {
                private.push(self.kernel(bytes)?);
                cast_sources.push(part.stored_bytes);
            }
        }
        let largest_host_copy = host_copies.iter().copied().max().unwrap_or(0);
        let logical = sum(&exact)?;
        let largest_upload = uploads.iter().copied().max().unwrap_or(0);
        let largest_cast_source = cast_sources.iter().copied().max().unwrap_or(0);
        let step = match self.backend {
            // Each part's source while it is cast, then every part beside the
            // concatenation.
            BackendKind::Cpu | BackendKind::Cuda => Step {
                logical,
                resident: logical,
                high: sum(&[logical, logical.max(largest_cast_source)])?,
                host: if self.backend == BackendKind::Cuda {
                    largest_upload
                } else {
                    0
                },
                ..Step::default()
            },
            BackendKind::Metal => {
                let resident = self.kernel(logical)?;
                let uploads = sum(&uploads)?;
                let host = if host_cast {
                    largest_host_copy
                } else if source == Source::Copied {
                    largest_upload
                } else {
                    0
                };
                Step {
                    logical,
                    resident,
                    high: sum(&[resident, uploads, host])?,
                    held_until_sync: uploads,
                    intermediates: sum(&private)?,
                    ..Step::default()
                }
            }
        };
        self.apply(step)
    }

    /// A GGUF tensor kept quantized (`QTensor` / `QMatMul`), read through a
    /// host buffer.
    fn gguf_stored(&mut self, tensor: &CensusTensor) -> Result<()> {
        let stored = tensor.stored_bytes;
        self.apply(if self.backend == BackendKind::Cuda {
            let resident = sum(&[stored, cuda_qtensor_padding(tensor)?])?;
            Step {
                logical: stored,
                resident,
                high: resident,
                host: stored,
                ..Step::default()
            }
        } else {
            // The read buffer beside the `QTensor`'s own copy (CPU) or upload.
            Step {
                logical: stored,
                resident: stored,
                high: sum(&[stored, stored])?,
                ..Step::default()
            }
        })
    }

    /// A GGUF tensor dequantized (`QTensor::dequantize`, which yields F32) and
    /// cast to `dtype`. The `QTensor` lives until the cast is done.
    fn gguf_dense(&mut self, tensor: &CensusTensor, dtype: DType) -> Result<()> {
        let stored = tensor.stored_bytes;
        let f32_bytes = dense_bytes(tensor.elements(), DType::F32)?;
        let logical = dense_bytes(tensor.elements(), dtype)?;
        let cast = dtype != DType::F32;
        let step = match self.backend {
            // The read buffer beside the `QTensor`'s copy; then the `QTensor`,
            // its F32 dequantization and the cast.
            BackendKind::Cpu => {
                let cast_copy = if cast { logical } else { 0 };
                Step {
                    logical,
                    resident: if cast { logical } else { f32_bytes },
                    high: sum(&[stored, stored])?.max(sum(&[stored, f32_bytes, cast_copy])?),
                    ..Step::default()
                }
            }
            // Candle dequantizes on the host (`quantized/metal.rs`): the
            // upload and a pooled blit copy of it, a device synchronization,
            // the host F32 result and its upload (the resident tensor unless a
            // cast replaces it), then the cast.
            BackendKind::Metal => {
                let blit = self.kernel(stored)?;
                let resident = if cast {
                    self.kernel(logical)?
                } else {
                    f32_bytes
                };
                let held = sum(&[stored, blit])?;
                Step {
                    logical,
                    resident,
                    // The host F32 buffer is freed before the cast allocates.
                    high: sum(&[
                        held,
                        f32_bytes,
                        f32_bytes.max(if cast { resident } else { 0 }),
                    ])?,
                    held_until_sync: sum(&[held, if cast { f32_bytes } else { 0 }])?,
                    before_sync: Some(held),
                    ..Step::default()
                }
            }
            // Quantized types dequantize on the device; F32, F16 and BF16
            // sources round-trip through the host (`quantized/cuda.rs`).
            BackendKind::Cuda => {
                let host_dequantized = matches!(
                    tensor.dtype,
                    StoredDType::Ggml(GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16)
                );
                let cast_copy = if cast { logical } else { 0 };
                Step {
                    logical,
                    resident: if cast { logical } else { f32_bytes },
                    high: sum(&[stored, cuda_qtensor_padding(tensor)?, f32_bytes, cast_copy])?,
                    host: if host_dequantized {
                        sum(&[stored, f32_bytes])?
                    } else {
                        stored
                    },
                    ..Step::default()
                }
            }
        };
        self.apply(step)
    }

    /// Metal: quantized projections packed into one `QTensor`. Each part is
    /// uploaded and read back through a pooled blit copy (each read-back
    /// synchronizes), concatenated on the host, and uploaded again.
    fn gguf_packed(&mut self, parts: &[&CensusTensor]) -> Result<()> {
        let stored = sum(&parts
            .iter()
            .map(|part| part.stored_bytes)
            .collect::<Vec<_>>())?;
        let mut blit = 0;
        for part in parts {
            blit = blit.max(self.kernel(part.stored_bytes)?);
        }
        let parts_and_blit = sum(&[stored, blit])?;
        self.apply(Step {
            logical: stored,
            resident: stored,
            // The parts, the last blit copy, the host concatenation and the
            // packed upload.
            high: sum(&[parts_and_blit, stored, stored])?,
            held_until_sync: parts_and_blit,
            before_sync: Some(parts_and_blit),
            ..Step::default()
        })
    }

    /// A tensor the loader computes on the device: its kernel-written result
    /// stays resident, and the kernel-written `intermediates` (exact bytes) are
    /// freed once it is built.
    fn derived(&mut self, elements: u64, dtype: DType, intermediates: &[u64]) -> Result<()> {
        let logical = dense_bytes(elements, dtype)?;
        let resident = self.kernel(logical)?;
        let mut allocated = Vec::with_capacity(intermediates.len());
        for bytes in intermediates {
            allocated.push(self.kernel(*bytes)?);
        }
        let intermediates = sum(&allocated)?;
        self.apply(if self.metal() {
            Step {
                logical,
                resident,
                high: resident,
                intermediates,
                ..Step::default()
            }
        } else {
            Step {
                logical,
                resident,
                high: sum(&[resident, intermediates])?,
                ..Step::default()
            }
        })
    }

    /// A table the loader builds on the host and uploads at its exact size.
    fn uploaded(&mut self, elements: u64, dtype: DType) -> Result<()> {
        let bytes = dense_bytes(elements, dtype)?;
        self.apply(match self.backend {
            BackendKind::Cpu => Step {
                logical: bytes,
                resident: bytes,
                high: bytes,
                ..Step::default()
            },
            BackendKind::Metal => Step {
                logical: bytes,
                resident: bytes,
                high: sum(&[bytes, bytes])?,
                ..Step::default()
            },
            BackendKind::Cuda => Step {
                logical: bytes,
                resident: bytes,
                high: bytes,
                host: bytes,
                ..Step::default()
            },
        })
    }

    /// A weight read on the host at its stored dtype, converted there to
    /// `dtype` and uploaded once at its exact size (`F16WeightLinear`): the
    /// host read and its converted copy are alive beside the upload.
    pub(super) fn host_converted(&mut self, tensor: &CensusTensor, dtype: DType) -> Result<()> {
        self.host_converted_through(tensor, None, dtype)
    }

    /// [`Self::host_converted`] for a weight the loader reads at an
    /// `intermediate` host dtype first (an MLX-quantized Qwen3-TTS text
    /// embedding is dequantized at F32, then converted). The read converts
    /// the stored copy and drops it, so the intermediate is alive with the
    /// stored copy while it is read and with the converted form after: the
    /// larger pair, not all three.
    pub(super) fn host_converted_through(
        &mut self,
        tensor: &CensusTensor,
        intermediate: Option<DType>,
        dtype: DType,
    ) -> Result<()> {
        let bytes = dense_bytes(tensor.elements(), dtype)?;
        let host = match intermediate {
            Some(form) => {
                let intermediate = dense_bytes(tensor.elements(), form)?;
                sum(&[tensor.stored_bytes, intermediate])?.max(sum(&[intermediate, bytes])?)
            }
            None => sum(&[tensor.stored_bytes, bytes])?,
        };
        self.host_staged(bytes, host)
    }

    /// [`Self::host_converted`] for a weight converted straight from its
    /// memory map (Qwen3-TTS's float text embedding): the converted copy is
    /// the only host allocation, the mapped checkpoint pages being file
    /// backed.
    pub(super) fn host_converted_in_place(
        &mut self,
        tensor: &CensusTensor,
        dtype: DType,
    ) -> Result<()> {
        let bytes = dense_bytes(tensor.elements(), dtype)?;
        self.host_staged(bytes, bytes)
    }

    /// A weight of `bytes` uploaded once from `host` bytes of host buffers.
    fn host_staged(&mut self, bytes: u64, host: u64) -> Result<()> {
        self.apply(match self.backend {
            BackendKind::Cpu | BackendKind::Metal => Step {
                logical: bytes,
                resident: bytes,
                high: sum(&[host, bytes])?,
                ..Step::default()
            },
            BackendKind::Cuda => Step {
                logical: bytes,
                resident: bytes,
                high: bytes,
                host,
                ..Step::default()
            },
        })
    }

    /// The most alive at a synchronization: the step's own buffers beside
    /// what another step, or a release, left held.
    fn at_sync_max(&self) -> Result<u64> {
        let mut top = [(0u64, usize::MAX); 2];
        for (index, (held, _)) in self.synced.iter().enumerate() {
            if *held > top[0].0 {
                top = [(*held, index), top[0]];
            } else if *held > top[1].0 {
                top[1] = (*held, index);
            }
        }
        let mut most = self.at_sync_max;
        for (index, (_, before_sync)) in self.synced.iter().enumerate() {
            let other = if top[0].1 == index {
                top[1].0
            } else {
                top[0].0
            };
            most = most.max(sum(&[other, *before_sync])?);
        }
        Ok(most)
    }

    fn finish(mut self) -> Result<Self> {
        let base = sum(&[self.resident, self.max_live])?;
        self.peak = if self.metal() {
            let base = sum(&[base, self.scratch])?;
            if self.held_unsynced > 0 {
                // Nothing synchronizes before the load fence: every released
                // upload is still held when the last step runs. A staged
                // loader synchronizes right after the upload that brings its
                // uploads since the last synchronization to the interval, and
                // a step releases only uploads it made before its release.
                // So the steps released since the last synchronization made
                // less than an interval of uploads after it, except the one
                // step that straddles it.
                let held_unsynced = match self.sync_interval {
                    Some(interval) => self
                        .held_unsynced
                        .min(sum(&[interval, self.held_step_max])?),
                    None => self.held_unsynced,
                };
                let held = sum(&[held_unsynced, self.held_max])?;
                let extra = self.extra_beyond_held_max.max(self.before_sync_max);
                sum(&[base, held, extra])?
            } else {
                // A synchronizing step drops what the previous one left held;
                // the others run beside it.
                let unsynced = sum(&[self.held_max, self.extra_max])?;
                let extra = self.extra_max_synced.max(unsynced).max(self.at_sync_max()?);
                sum(&[base, extra])?
            }
        } else {
            sum(&[base, self.extra_max.max(self.extra_max_synced)])?
        }
        .max(self.prelude);
        Ok(self)
    }
}

fn cuda_qtensor_padding(tensor: &CensusTensor) -> Result<u64> {
    let StoredDType::Ggml(dtype) = tensor.dtype else {
        return Ok(0);
    };
    CUDA_QTENSOR_ROW_PADDING
        .checked_mul(dtype.type_size() as u64)
        .map(|bytes| bytes / dtype.block_size() as u64)
        .ok_or_else(overflow)
}

/// The census of `paths`, or `None` when one is missing (the loader reports
/// that). A file that is not a readable checkpoint is an error.
pub(super) fn census_of(paths: &[PathBuf]) -> Result<Option<Vec<CensusFile>>> {
    if paths.iter().any(|path| !path.is_file()) {
        return Ok(None);
    }
    paths
        .iter()
        .map(|path| {
            census_checkpoint_file(path)?.ok_or_else(|| {
                Error::ModelLoadError(format!("Cannot read the tensor list of {}", path.display()))
            })
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

pub(super) fn by_name(files: &[CensusFile]) -> BTreeMap<&str, &CensusTensor> {
    files
        .iter()
        .flat_map(|file| file.tensors.iter())
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect()
}

fn tensor<'a>(tensors: &BTreeMap<&str, &'a CensusTensor>, name: &str) -> Result<&'a CensusTensor> {
    tensors
        .get(name)
        .copied()
        .ok_or_else(|| Error::ModelLoadError(format!("Checkpoint is missing tensor `{name}`")))
}

fn dim(tensor: &CensusTensor, index: usize) -> Result<u64> {
    tensor
        .shape
        .get(index)
        .map(|dim| *dim as u64)
        .ok_or_else(|| {
            Error::ModelLoadError(format!("Tensor `{}` has no dimension {index}", tensor.name))
        })
}

fn whisper_plan(
    model_path: &Path,
    device: &DeviceProfile,
    backend: BackendKind,
) -> Result<Option<AsrResidencyPlan>> {
    let Some(files) = census_of(&whisper_weight_files(model_path)?)? else {
        return Ok(None);
    };
    // F16 on Metal by default. A failed load-time self-check reloads in F32
    // after dropping the F16 model; that rare fallback is not priced here
    // (as for Parakeet and Nemotron), and live-memory admission sees it.
    price_whisper(&files, backend, whisper_weight_dtype(device)?).map(Some)
}

/// Whisper (`whisper/model.rs`): every tensor through a `VarBuilder` at
/// `dtype`, except the encoder positional table, which the loader computes on
/// the device (F32 sin/cos, cast to `dtype`; the checkpoint's copy gives its
/// shape), and each self-attention's Q/K/V weights and Q/V biases, which it
/// concatenates into one projection with a zero K bias.
fn price_whisper(
    files: &[CensusFile],
    backend: BackendKind,
    dtype: DType,
) -> Result<AsrResidencyPlan> {
    const ENCODER_POSITIONS: &str = "model.encoder.embed_positions.weight";
    let tensors = by_name(files);
    let mut ledger = Ledger::staged(backend);
    let mut fused = BTreeSet::new();
    for name in tensors.keys() {
        let Some(attention) = name.strip_suffix(".q_proj.weight") else {
            continue;
        };
        if !attention.ends_with(".self_attn") {
            continue;
        }
        let part = |suffix: &str| tensor(&tensors, &format!("{attention}.{suffix}"));
        let weights = [
            part("q_proj.weight")?,
            part("k_proj.weight")?,
            part("v_proj.weight")?,
        ];
        ledger.fused(&weights, Source::Mapped, dtype, 0)?;
        let biases = [part("q_proj.bias")?, part("v_proj.bias")?];
        ledger.fused(&biases, Source::Mapped, dtype, biases[0].elements())?;
        fused.extend(weights.iter().chain(&biases).map(|t| t.name.as_str()));
    }
    for (name, tensor) in &tensors {
        if *name != ENCODER_POSITIONS && !fused.contains(name) {
            ledger.var(tensor, Source::Mapped, &[dtype])?;
        }
    }
    let positions = tensor(&tensors, ENCODER_POSITIONS)?;
    let elements = positions.elements();
    // scaled_time, sin and cos are [rows, channels / 2] F32; a cast replaces
    // their F32 concatenation.
    let half = dense_bytes(elements / 2, DType::F32)?;
    let mut intermediates = vec![half, half, half];
    if dtype != DType::F32 {
        intermediates.push(dense_bytes(elements, DType::F32)?);
    }
    ledger.derived(elements, dtype, &intermediates)?;
    AsrResidencyPlan::from_ledger(ledger, 0)
}

/// The switches the Qwen3-ASR GGUF loader resolves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Qwen3AsrLoad {
    text_dtype: DType,
    audio_dtype: DType,
    /// Text projections and `lm_head` stay quantized `QMatMul`s.
    qmatmul_text: bool,
    /// Q/K/V and gate/up are packed into one quantized projection each.
    packed_projections: bool,
}

fn qwen3_asr_plan(
    variant: ModelVariant,
    model_path: &Path,
    device: &DeviceProfile,
    backend: BackendKind,
) -> Result<Option<AsrResidencyPlan>> {
    let Some(path) = resolve_qwen_asr_gguf_path(model_path, variant) else {
        return Ok(None);
    };
    let Some(files) = census_of(&[path])? else {
        return Ok(None);
    };
    let (text_dtype, audio_dtype) = qwen3_asr_weight_dtypes(
        device,
        ModelFamily::Qwen3Asr,
        true,
        true,
        Some(QWEN3_ASR_GGUF_CHECKPOINT_DTYPE),
    )?;
    let tensors = by_name(&files);
    let prefix = qwen3_asr_prefix(&tensors);
    let hidden_size = dim(
        tensor(&tensors, &format!("{prefix}model.embed_tokens.weight"))?,
        1,
    )?;
    let load = Qwen3AsrLoad {
        text_dtype,
        audio_dtype,
        qmatmul_text: qwen3_asr_use_gguf_qmatmul_text(&device.device),
        packed_projections: qwen3_quantized_projection_grouping_for_hidden_size(
            &device.device,
            hidden_size as usize,
        ),
    };
    price_qwen3_asr(&files, backend, load).map(Some)
}

fn qwen3_asr_prefix(tensors: &BTreeMap<&str, &CensusTensor>) -> &'static str {
    if tensors.keys().any(|name| name.starts_with("thinker.")) {
        "thinker."
    } else {
        ""
    }
}

/// Qwen3-ASR GGUF (`qwen3/asr/mod.rs`): the audio tower dequantized to the
/// audio dtype; the text embedding and norms dequantized to the text dtype; the
/// text projections and the separate `lm_head` kept quantized (unless the
/// `QMatMul` text path is off); plus the F32 audio position table built on the
/// host.
fn price_qwen3_asr(
    files: &[CensusFile],
    backend: BackendKind,
    load: Qwen3AsrLoad,
) -> Result<AsrResidencyPlan> {
    const PROJECTIONS: [&str; 7] = [
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
    ];
    let tensors = by_name(files);
    let prefix = qwen3_asr_prefix(&tensors);
    let audio = format!("{prefix}audio_tower.");
    let text = format!("{prefix}model.");
    let lm_head = format!("{prefix}lm_head.weight");
    let mut ledger = Ledger::new(backend);
    let mut packed = BTreeSet::new();
    if load.qmatmul_text && load.packed_projections {
        for name in tensors.keys() {
            let Some(layer) = name.strip_suffix(".self_attn.q_proj.weight") else {
                continue;
            };
            if !layer.starts_with(&text) {
                continue;
            }
            for group in [
                ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"].as_slice(),
                ["mlp.gate_proj", "mlp.up_proj"].as_slice(),
            ] {
                let parts = group
                    .iter()
                    .map(|part| tensor(&tensors, &format!("{layer}.{part}.weight")))
                    .collect::<Result<Vec<_>>>()?;
                ledger.gguf_packed(&parts)?;
                packed.extend(parts.iter().map(|part| part.name.as_str()));
            }
        }
    }
    for (name, tensor) in &tensors {
        if packed.contains(name) {
            continue;
        }
        if name.starts_with(&audio) {
            ledger.gguf_dense(tensor, load.audio_dtype)?;
        } else if name.starts_with(&text) || *name == lm_head {
            let projection = *name == lm_head || PROJECTIONS.iter().any(|p| name.ends_with(p));
            if load.qmatmul_text && projection {
                ledger.gguf_stored(tensor)?;
            } else {
                ledger.gguf_dense(tensor, load.text_dtype)?;
            }
        }
    }
    let d_model = dim(tensor(&tensors, &format!("{audio}ln_post.weight"))?, 0)?;
    let positions = (QWEN3_ASR_AUDIO_POSITIONS as u64)
        .checked_mul(d_model)
        .ok_or_else(overflow)?;
    ledger.uploaded(positions, DType::F32)?;
    AsrResidencyPlan::from_ledger(ledger, 0)
}

/// The NeMo ASR families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NemoFamily {
    Parakeet,
    Nemotron,
}

impl NemoFamily {
    fn is_f16_projection(self, name: &str) -> bool {
        match self {
            Self::Parakeet => parakeet_tensor_is_f16_projection(name),
            Self::Nemotron => nemotron_tensor_is_f16_projection(name),
        }
    }

    fn memo_bytes(self) -> u64 {
        match self {
            Self::Parakeet => 0,
            Self::Nemotron => NEMOTRON_MODEL_MEMO_MAX_BYTES,
        }
    }
}

fn parakeet_plan(
    variant: ModelVariant,
    model_path: &Path,
    device: &DeviceProfile,
    backend: BackendKind,
) -> Result<Option<AsrResidencyPlan>> {
    let paths = parakeet_checkpoint_paths(model_path, variant)?;
    let f16_projections = resolve_parakeet_projection_precision(device).dtype == DType::F16;
    // The loader reads the extracted checkpoint, or its zip-repacked copy when
    // Candle cannot parse the original.
    for checkpoint in [&paths.checkpoint_path, &paths.repacked_checkpoint_path] {
        if checkpoint.is_file() {
            if let Some(file) = census_checkpoint_file(checkpoint)? {
                return price_nemo(
                    &[file],
                    backend,
                    PARAKEET_WEIGHT_DTYPE,
                    NemoFamily::Parakeet,
                    f16_projections,
                )
                .map(Some);
            }
        }
    }
    // Without a tensor list the first load is priced at F32 throughout, an
    // over-estimate when the projections load in F16.
    if paths.checkpoint_path.is_file() {
        // An unreadable checkpoint is repacked on disk before it loads.
        let payload = std::fs::metadata(&paths.checkpoint_path)?.len();
        return price_nemo_first_load(backend, payload, PARAKEET_WEIGHT_DTYPE, 0, 0).map(Some);
    }
    if !paths.nemo_path.is_file() {
        return Ok(None);
    }
    // First load: the archive is streamed to disk through small buffers.
    let payload = nemo_payload_bytes(&paths.nemo_path)?;
    price_nemo_first_load(backend, payload, PARAKEET_WEIGHT_DTYPE, 0, 0).map(Some)
}

fn nemotron_plan(
    variant: ModelVariant,
    model_path: &Path,
    device: &DeviceProfile,
    backend: BackendKind,
) -> Result<Option<AsrResidencyPlan>> {
    let paths = nemotron_checkpoint_paths(model_path, variant)?;
    let dtype = resolve_nemotron_asr_dtype(device)?.dtype;
    let f16_projections = resolve_nemotron_projection_precision(device, dtype).dtype == DType::F16;
    if paths.checkpoint_path.is_file() {
        if let Some(file) = census_checkpoint_file(&paths.checkpoint_path)? {
            return price_nemo(
                &[file],
                backend,
                dtype,
                NemoFamily::Nemotron,
                f16_projections,
            )
            .map(Some);
        }
    }
    if !paths.nemo_path.is_file() {
        return Ok(None);
    }
    // First load: the archive is streamed to disk through small buffers.
    // Without a tensor list it is priced at `dtype` throughout, an
    // over-estimate when the projections load in F16.
    let payload = nemo_payload_bytes(&paths.nemo_path)?;
    let memo = NemoFamily::Nemotron.memo_bytes();
    price_nemo_first_load(backend, payload, dtype, 0, memo).map(Some)
}

/// Parakeet and Nemotron (`VarBuilder::from_pth`): each tensor is read into a
/// host buffer, uploaded, and cast to `dtype` on the device. BatchNorm step
/// counters are never read, and the featurizer tensors go back to the host.
/// Parakeet adds each depthwise convolution with its BatchNorm folded in;
/// Nemotron on CUDA adds transposed copies of its LSTM weights. With
/// `f16_projections` the family's encoder projections are read and converted
/// to F16 on the host and uploaded once (weight-only F16 on Metal).
fn price_nemo(
    files: &[CensusFile],
    backend: BackendKind,
    dtype: DType,
    family: NemoFamily,
    f16_projections: bool,
) -> Result<AsrResidencyPlan> {
    let mut ledger = Ledger::new(backend);
    for tensor in files.iter().flat_map(|file| file.tensors.iter()) {
        let name = tensor.name.as_str();
        if name.ends_with(".num_batches_tracked") || name.starts_with("preprocessor.") {
            continue;
        }
        if f16_projections && family.is_f16_projection(name) {
            ledger.host_converted(tensor, DType::F16)?;
        } else {
            ledger.var(tensor, Source::Copied, &[dtype])?;
        }
        match family {
            NemoFamily::Parakeet if name.ends_with(".conv.depthwise_conv.weight") => {
                let channels = dim(tensor, 0)?;
                ledger.derived(sum(&[tensor.elements(), channels])?, dtype, &[])?;
            }
            NemoFamily::Nemotron
                if backend == BackendKind::Cuda
                    && (name.contains(".lstm.weight_ih_l")
                        || name.contains(".lstm.weight_hh_l")) =>
            {
                ledger.derived(tensor.elements(), dtype, &[])?;
            }
            _ => {}
        }
    }
    AsrResidencyPlan::from_ledger(ledger, family.memo_bytes())
}

/// A NeMo load before its checkpoint has been extracted, so there is no tensor
/// list to read: the F32 checkpoint is priced from the archive's `payload`
/// (which also holds the small config and tokenizer members) cast to `dtype`,
/// with one tensor's copies bounded by [`NEMO_UNLISTED_TENSOR_BYTES`], after
/// an `extraction` host buffer that is freed before any weight loads.
fn price_nemo_first_load(
    backend: BackendKind,
    payload: u64,
    dtype: DType,
    extraction: u64,
    memo_bytes: u64,
) -> Result<AsrResidencyPlan> {
    let weights = payload
        .checked_div(DType::F32.size_in_bytes() as u64)
        .and_then(|elements| elements.checked_mul(dtype.size_in_bytes() as u64))
        .ok_or_else(overflow)?;
    let mut ledger = Ledger::new(backend);
    ledger.host_buffer(extraction);
    // One tensor's host copy (which on the CPU is the weight itself) and the
    // stored-dtype upload a cast replaces.
    let cast_source = if dtype != DType::F32 {
        NEMO_UNLISTED_TENSOR_BYTES
    } else {
        0
    };
    ledger.apply(match backend {
        BackendKind::Cpu => Step {
            logical: weights,
            resident: weights,
            high: sum(&[weights, cast_source])?,
            ..Step::default()
        },
        BackendKind::Metal => Step {
            logical: weights,
            resident: weights,
            high: sum(&[weights, NEMO_UNLISTED_TENSOR_BYTES, cast_source])?,
            ..Step::default()
        },
        BackendKind::Cuda => Step {
            logical: weights,
            resident: weights,
            high: sum(&[weights, cast_source])?,
            host: NEMO_UNLISTED_TENSOR_BYTES,
            ..Step::default()
        },
    })?;
    AsrResidencyPlan::from_ledger(ledger, memo_bytes)
}

/// The uncompressed payload of a `.nemo` archive, a tar that some checkpoints
/// gzip, read without decompressing it. A gzip trailer stores the payload size
/// modulo 2^32 (ISIZE). The payload is the smallest value with that residue
/// that is no smaller than the archive less deflate's bounded expansion (about
/// 5 bytes per 64 KiB stored block). A weights tarball barely compresses.
fn nemo_payload_bytes(path: &Path) -> Result<u64> {
    let mut file = std::fs::File::open(path)?;
    let archive = file.metadata()?.len();
    let mut magic = [0u8; 2];
    if archive < 18 || file.read_exact(&mut magic).is_err() || magic != [0x1f, 0x8b] {
        return Ok(archive);
    }
    file.seek(SeekFrom::End(-4))?;
    let mut residue = [0u8; 4];
    file.read_exact(&mut residue)?;
    let floor = archive - archive / 512;
    let mut payload = u64::from(u32::from_le_bytes(residue));
    while payload < floor {
        payload = payload.checked_add(1 << 32).ok_or_else(overflow)?;
    }
    Ok(payload)
}

/// The switches the Granite Speech loader resolves.
#[derive(Debug, Clone, Copy, PartialEq)]
struct GraniteLoad {
    dtype: DType,
    /// The encoder and projector's dtype (F32 under weight-only F16).
    audio_dtype: DType,
    /// Encoder projections load as weight-only F16.
    f16_encoder_projections: bool,
    representation: GraniteLoadRepresentation,
    residual_multiplier: f32,
    tied_lm_head: bool,
    /// (encoder layers, attention context) of the per-layer U32 distance table.
    encoder_attention_dists: (usize, usize),
}

fn granite_plan(
    model_path: &Path,
    device: &DeviceProfile,
    backend: BackendKind,
) -> Result<Option<AsrResidencyPlan>> {
    let Ok(shards) = ensure_granite_speech_artifacts(model_path) else {
        return Ok(None);
    };
    let Some(files) = census_of(&shards)? else {
        return Ok(None);
    };
    let config = GraniteSpeechConfig::load(model_path)?;
    let dtype = select_granite_speech_dtype(device, config.target_dtype_hint())?;
    let f16_encoder_projections =
        resolve_granite_encoder_projection_precision(device).dtype == DType::F16;
    let load = GraniteLoad {
        dtype,
        audio_dtype: if f16_encoder_projections {
            DType::F32
        } else {
            dtype
        },
        f16_encoder_projections,
        representation: granite_load_representation(&device.device, dtype),
        residual_multiplier: config.text_config.residual_multiplier,
        tied_lm_head: config.tie_word_embeddings(),
        encoder_attention_dists: (
            config.encoder_config.num_layers,
            config.encoder_config.context_size,
        ),
    };
    price_granite(&files, backend, &load).map(Some)
}

/// Granite Speech (`granite_speech/asr/runtime.rs`): every tensor through a
/// `VarBuilder` at the base dtype except BatchNorm step counters, with the LM's
/// representation switches applied: F16 (and fused) Q/K/V and gate/up, o_proj
/// and down_proj read at F16 (when selective) and then pre-scaled by the
/// residual multiplier, and
/// an F16 copy of a tied embedding as the LM head. The encoder and projector
/// read at the audio dtype, with the encoder projections converted to F16 on
/// the host under weight-only F16. Each encoder layer adds a host-built U32
/// attention distance table.
fn price_granite(
    files: &[CensusFile],
    backend: BackendKind,
    load: &GraniteLoad,
) -> Result<AsrResidencyPlan> {
    const EMBEDDING: &str = "language_model.model.embed_tokens.weight";
    const LM_HEAD: &str = "language_model.lm_head.weight";
    let representation = load.representation;
    let dtype = load.dtype;
    let half_if = |enabled: bool| if enabled { DType::F16 } else { dtype };
    let prescaled = representation.residual_prescale && load.residual_multiplier != 1.0;
    // Read at F16 when selective, then pre-scaled in that dtype.
    let residual_chain = |f16: bool| {
        let read = half_if(f16);
        if prescaled {
            vec![read, read]
        } else {
            vec![read]
        }
    };
    let tensors = by_name(files);
    let mut ledger = Ledger::staged(backend);
    let mut grouped = BTreeSet::new();
    for name in tensors.keys() {
        if !name.starts_with("language_model.") {
            continue;
        }
        let groups = if let Some(attention) = name.strip_suffix(".q_proj.weight") {
            (
                attention,
                ["q_proj", "k_proj", "v_proj"].as_slice(),
                half_if(representation.f16_qkv),
                representation.qkv_fusion,
            )
        } else if let Some(mlp) = name.strip_suffix(".gate_proj.weight") {
            (
                mlp,
                ["gate_proj", "up_proj"].as_slice(),
                half_if(representation.f16_mlp),
                representation.gate_up_fusion,
            )
        } else {
            continue;
        };
        let (stem, parts, part_dtype, fuse) = groups;
        let parts = parts
            .iter()
            .map(|part| tensor(&tensors, &format!("{stem}.{part}.weight")))
            .collect::<Result<Vec<_>>>()?;
        if fuse {
            ledger.fused(&parts, Source::Mapped, part_dtype, 0)?;
        } else {
            for part in &parts {
                ledger.var(part, Source::Mapped, &[part_dtype])?;
            }
        }
        grouped.extend(parts.iter().map(|part| part.name.as_str()));
    }
    for (name, tensor) in &tensors {
        if grouped.contains(name) || name.ends_with(".num_batches_tracked") {
            continue;
        }
        if !name.starts_with("language_model.") {
            if load.f16_encoder_projections && granite_tensor_is_f16_encoder_projection(name) {
                ledger.host_converted(tensor, DType::F16)?;
            } else {
                ledger.var(tensor, Source::Mapped, &[load.audio_dtype])?;
            }
        } else if name.ends_with(".o_proj.weight") {
            let chain = residual_chain(representation.f16_attention_output);
            ledger.var(tensor, Source::Mapped, &chain)?;
        } else if name.ends_with(".down_proj.weight") {
            ledger.var(
                tensor,
                Source::Mapped,
                &residual_chain(representation.f16_mlp),
            )?;
        } else if *name == LM_HEAD {
            ledger.var(
                tensor,
                Source::Mapped,
                &[half_if(representation.f16_lm_head)],
            )?;
        } else {
            ledger.var(tensor, Source::Mapped, &[dtype])?;
            if *name == EMBEDDING && load.tied_lm_head && representation.f16_lm_head {
                ledger.derived(tensor.elements(), DType::F16, &[])?;
            }
        }
    }
    let (layers, context) = load.encoder_attention_dists;
    let dists = (layers as u64)
        .checked_mul(context as u64)
        .and_then(|n| n.checked_mul(context as u64))
        .ok_or_else(overflow)?;
    ledger.uploaded(dists, DType::U32)?;
    AsrResidencyPlan::from_ledger(ledger, 0)
}

fn vibevoice_plan(
    model_path: &Path,
    device: &DeviceProfile,
    backend: BackendKind,
) -> Result<Option<AsrResidencyPlan>> {
    let Ok(paths) = safetensors_weight_files(model_path) else {
        return Ok(None);
    };
    let Some(files) = census_of(&paths)? else {
        return Ok(None);
    };
    let lm_dtype = vibevoice_asr_weight_dtype(device)?;
    let load = VibeVoiceLoad {
        lm_dtype,
        audio_dtype: vibevoice_asr_audio_dtype(device, lm_dtype),
        f16_tokenizer_projections: resolve_vibevoice_tokenizer_projection_precision(device).dtype
            == DType::F16,
        fused_projections: qwen3_dense_projection_fusion_enabled(&device.device),
        tts: false,
        staged: false,
    };
    price_vibevoice(&files, backend, &load).map(Some)
}

/// The TTS-only acoustic decoder, which VibeVoice-ASR does not load.
const VIBEVOICE_ASR_UNLOADED_DECODER: &str = "model.acoustic_tokenizer.decoder.";

/// How `VibeVoiceAsrModel::load` (or, with `tts`, `VibeVoiceTtsModel::load`)
/// represents the checkpoint.
#[derive(Debug, Clone, Copy)]
pub(super) struct VibeVoiceLoad {
    /// The decoder LM's dtype.
    lm_dtype: DType,
    /// The tokenizers' and connectors' dtype.
    audio_dtype: DType,
    /// The tokenizer encoders' FFN linears load as weight-only F16.
    f16_tokenizer_projections: bool,
    /// Dense projection fusion (`qwen3/core.rs`).
    fused_projections: bool,
    /// The TTS model, which loads the acoustic decoder and the prediction
    /// head the ASR model skips.
    tts: bool,
    /// The loader reads through `staged::metal_staged_safetensors`.
    staged: bool,
}

impl VibeVoiceLoad {
    /// `VibeVoiceTtsModel::load`: every tensor at `dtype`.
    pub(super) fn tts(dtype: DType, fused_projections: bool) -> Self {
        Self {
            lm_dtype: dtype,
            audio_dtype: dtype,
            f16_tokenizer_projections: false,
            fused_projections,
            tts: true,
            staged: true,
        }
    }
}

/// VibeVoice-ASR (`vibevoice/asr.rs`): every tensor through a `VarBuilder`,
/// the decoder LM at its dtype and the audio path (tokenizer encoders and
/// connectors) at the audio dtype, with the tokenizer encoders' FFN weights
/// converted to F16 on the host under weight-only F16. The acoustic decoder
/// (ASR loads the acoustic tokenizer encoder-only) and a TTS checkpoint's
/// prediction head and speech scaling factors are skipped. With dense
/// projection fusion (`qwen3/core.rs`), each LM layer's
/// Q/K/V weights and biases and its gate/up weights are concatenated.
pub(super) fn price_vibevoice(
    files: &[CensusFile],
    backend: BackendKind,
    load: &VibeVoiceLoad,
) -> Result<AsrResidencyPlan> {
    let dtype = load.lm_dtype;
    let fused_projections = load.fused_projections;
    let tensors = by_name(files);
    let mut ledger = if load.staged {
        Ledger::staged(backend)
    } else {
        Ledger::new(backend)
    };
    let mut grouped = BTreeSet::new();
    if fused_projections {
        for name in tensors.keys() {
            let Some(layer) = name.strip_suffix(".self_attn.q_proj.weight") else {
                continue;
            };
            if !layer.starts_with("model.language_model.layers.") {
                continue;
            }
            for group in [
                [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ]
                .as_slice(),
                [
                    "self_attn.q_proj.bias",
                    "self_attn.k_proj.bias",
                    "self_attn.v_proj.bias",
                ]
                .as_slice(),
                ["mlp.gate_proj.weight", "mlp.up_proj.weight"].as_slice(),
            ] {
                let parts = group
                    .iter()
                    .map(|part| tensor(&tensors, &format!("{layer}.{part}")))
                    .collect::<Result<Vec<_>>>()?;
                ledger.fused(&parts, Source::Mapped, dtype, 0)?;
                grouped.extend(parts.iter().map(|part| part.name.as_str()));
            }
        }
    }
    for (name, tensor) in &tensors {
        let tts_only = !load.tts
            && (name.starts_with("model.prediction_head.")
                || name.starts_with(VIBEVOICE_ASR_UNLOADED_DECODER)
                || matches!(
                    *name,
                    "model.speech_scaling_factor" | "model.speech_bias_factor"
                ));
        if tts_only || grouped.contains(name) {
            continue;
        }
        if vibevoice_tensor_is_language_model(name) {
            ledger.var(tensor, Source::Mapped, &[dtype])?;
        } else if load.f16_tokenizer_projections
            && vibevoice_tensor_is_f16_tokenizer_projection(name)
        {
            ledger.host_converted(tensor, DType::F16)?;
        } else {
            ledger.var(tensor, Source::Mapped, &[load.audio_dtype])?;
        }
    }
    AsrResidencyPlan::from_ledger(ledger, 0)
}

fn voxtral_plan(
    model_path: &Path,
    device: &DeviceProfile,
    backend: BackendKind,
) -> Result<Option<AsrResidencyPlan>> {
    let Some(files) = census_of(&voxtral_realtime_weight_files(model_path)?)? else {
        return Ok(None);
    };
    let dtype = voxtral_realtime_weight_dtype(model_path, device)?;
    price_voxtral(&files, backend, dtype).map(Some)
}

/// Voxtral realtime (`voxtral/realtime/model.rs`): every tensor through a
/// `VarBuilder` at `dtype`, the LM head tied to the embedding without a copy.
fn price_voxtral(
    files: &[CensusFile],
    backend: BackendKind,
    dtype: DType,
) -> Result<AsrResidencyPlan> {
    let mut ledger = Ledger::staged(backend);
    for tensor in by_name(files).values() {
        ledger.var(tensor, Source::Mapped, &[dtype])?;
    }
    AsrResidencyPlan::from_ledger(ledger, 0)
}

/// The four LFM2.5-Audio GGUFs' censuses.
#[derive(Debug)]
struct Lfm25AudioCensus {
    main: CensusFile,
    mmproj: CensusFile,
    tokenizer: CensusFile,
    vocoder: CensusFile,
}

fn lfm25_audio_plan(model_path: &Path, backend: BackendKind) -> Result<Option<AsrResidencyPlan>> {
    let Ok(layout) = lfm25_audio_load_layout(model_path) else {
        return Ok(None);
    };
    let paths = [
        layout.main.clone(),
        layout.mmproj.clone(),
        layout.tokenizer.clone(),
        layout.vocoder.clone(),
    ];
    let Some(files) = census_of(&paths)? else {
        return Ok(None);
    };
    let [main, mmproj, tokenizer, vocoder]: [CensusFile; 4] =
        files.try_into().map_err(|_| overflow())?;
    let census = Lfm25AudioCensus {
        main,
        mmproj,
        tokenizer,
        vocoder,
    };
    price_lfm25_audio(&census, backend, layout.rope_tables).map(Some)
}

/// LFM2.5-Audio (`lfm25_audio/`, `lfm2/backbone.rs`): all four GGUFs load on
/// every backend. Quantized projections stay quantized; F32 tensors and the
/// embedding tables (the main backbone's `token_embd`, every loaded mmproj
/// tensor, the vocoder's audio and depth embeddings) are dequantized to F32,
/// the dtype the loader hard-codes (there is no dtype resolver to call). The
/// detokenizer backbone runs on vocoder embeddings and skips its `token_embd`.
/// The main backbone has no `output.weight`, so its head reads `token_embd` a
/// second time as a quantized projection, and keeps its first `token_embd`
/// read alive until it finishes loading. Each backbone also
/// precomputes F32 RoPE tables over its full context (`rope_tables`).
fn price_lfm25_audio(
    census: &Lfm25AudioCensus,
    backend: BackendKind,
    rope_tables: [(usize, usize); 2],
) -> Result<AsrResidencyPlan> {
    const MMPROJ_UNREAD: [&str; 3] = [
        "a.position_embd.weight",
        "a.embd_to_logits.weight",
        "a.position_embd_norm.weight",
    ];
    const VOCODER_UNREAD: [&str; 2] = [
        "audio_embedding.to_logits.weight",
        "audio_embedding.embedding_norm.weight",
    ];
    let mut ledger = Ledger::new(backend);
    price_lfm2_backbone(&mut ledger, &census.main, true)?;
    price_lfm2_backbone(&mut ledger, &census.tokenizer, false)?;
    for tensor in &census.mmproj.tensors {
        if !MMPROJ_UNREAD.contains(&tensor.name.as_str()) {
            ledger.gguf_dense(tensor, DType::F32)?;
        }
    }
    for tensor in &census.vocoder.tensors {
        let name = tensor.name.as_str();
        if VOCODER_UNREAD.contains(&name) {
            continue;
        }
        let embedding = name.ends_with(".embedding.weight") || name == "emb.emb.weight";
        if gguf_quantized(tensor) && !embedding {
            ledger.gguf_stored(tensor)?;
        } else {
            ledger.gguf_dense(tensor, DType::F32)?;
        }
    }
    for table in rope_tables {
        price_lfm2_rope_table(&mut ledger, table)?;
    }
    AsrResidencyPlan::from_ledger(ledger, 0)
}

fn gguf_quantized(tensor: &CensusTensor) -> bool {
    !matches!(
        tensor.dtype,
        StoredDType::Ggml(GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16)
    )
}

/// One LFM2 backbone GGUF (`lfm2/backbone.rs`): quantized projections stay
/// quantized and F32 tensors load dense. With `embeds_tokens`, token rows are
/// gathered from the quantized `token_embd` itself; a backbone without
/// `output.weight` uses that same tensor as its LM head, so it is read and
/// held once either way.
fn price_lfm2_backbone(
    ledger: &mut Ledger,
    backbone: &CensusFile,
    embeds_tokens: bool,
) -> Result<()> {
    let tensors = by_name(std::slice::from_ref(backbone));
    let embedding = tensor(&tensors, "token_embd.weight")?;
    for (name, tensor) in &tensors {
        if *name == "token_embd.weight" {
            continue;
        }
        if gguf_quantized(tensor) {
            ledger.gguf_stored(tensor)?;
        } else {
            ledger.gguf_dense(tensor, DType::F32)?;
        }
    }
    let tied_head =
        !tensors.contains_key("output.weight") && !tensors.contains_key("dense_2.weight");
    if embeds_tokens || tied_head {
        ledger.gguf_stored(embedding)?;
    }
    Ok(())
}

/// An LFM2 backbone's RoPE tables over `(positions, head_dim)`: `cos` and
/// `sin` (`[positions, head_dim / 2]` F32) built on the host and uploaded,
/// then concatenated on the device for the fused path.
fn price_lfm2_rope_table(ledger: &mut Ledger, (positions, head_dim): (usize, usize)) -> Result<()> {
    let half = (positions as u64)
        .checked_mul(head_dim as u64 / 2)
        .ok_or_else(overflow)?;
    ledger.uploaded(half, DType::F32)?;
    ledger.uploaded(half, DType::F32)?;
    ledger.derived(sum(&[half, half])?, DType::F32, &[])
}

/// LFM2.5-1.2B chat (`lfm2/chat.rs`): one GGUF on the LFM2 backbone loader,
/// with RoPE tables over the capped chat context. `None` when the checkpoint
/// is missing (the loader reports that).
pub(super) fn lfm2_chat_resource_plan(
    variant: ModelVariant,
    model_path: &Path,
    backend: BackendKind,
) -> Result<Option<ModelResourcePlan>> {
    let Ok(layout) =
        crate::models::architectures::lfm2::chat::lfm2_chat_load_layout(model_path, variant)
    else {
        return Ok(None);
    };
    let Some(files) = census_of(std::slice::from_ref(&layout.gguf))? else {
        return Ok(None);
    };
    let plan = price_lfm2_chat(&files[0], backend, layout.rope_table)?;
    tracing::info!(
        model = %variant,
        backend = ?plan.backend,
        logical_resident_bytes = plan.logical_resident_bytes,
        resident_bytes = plan.resident_bytes,
        load_peak_bytes = plan.load_peak_bytes,
        host_staging_bytes = plan.host_staging_bytes,
        "Priced LFM2.5 chat load from its residency plan"
    );
    plan.resource_plan().map(Some)
}

fn price_lfm2_chat(
    gguf: &CensusFile,
    backend: BackendKind,
    rope_table: (usize, usize),
) -> Result<AsrResidencyPlan> {
    let mut ledger = Ledger::new(backend);
    price_lfm2_backbone(&mut ledger, gguf, true)?;
    price_lfm2_rope_table(&mut ledger, rope_table)?;
    AsrResidencyPlan::from_ledger(ledger, 0)
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::backends::{DeviceCapabilities, DeviceKind};
    use crate::models::architectures::qwen3::core::{
        qwen3_dense_projection_fusion_policy, qwen3_quantized_projection_grouping_policy,
    };

    const GIB: f64 = (1u64 << 30) as f64;
    const BACKENDS: [BackendKind; 3] = [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda];

    /// A committed header census: `[name, dtype, shape]`, or
    /// `[pattern, dtype, shape, first, count]` for a run of tensors whose names
    /// differ in one index (`{i}`), with each file's exact payload total.
    pub(in super::super) fn fixture(json: &str) -> Vec<CensusFile> {
        let doc: serde_json::Value = serde_json::from_str(json).unwrap();
        let files = doc["files"].as_array().unwrap();
        files
            .iter()
            .map(|file| {
                let gguf = file["format"] == "gguf";
                let mut tensors = Vec::new();
                for entry in file["tensors"].as_array().unwrap() {
                    let entry = entry.as_array().unwrap();
                    let name = entry[0].as_str().unwrap();
                    let dtype = stored_dtype(entry[1].as_str().unwrap(), gguf);
                    let shape = entry[2]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|dim| dim.as_u64().unwrap() as usize)
                        .collect::<Vec<_>>();
                    let (first, count) = match entry.len() {
                        5 => (entry[3].as_u64().unwrap(), entry[4].as_u64().unwrap()),
                        _ => (0, 1),
                    };
                    for index in first..first + count {
                        let name = name.replace("{i}", &index.to_string());
                        tensors.push(census_tensor(name, dtype, shape.clone()));
                    }
                }
                assert_eq!(tensors.len() as u64, file["tensor_count"].as_u64().unwrap());
                assert_eq!(
                    tensors.iter().map(|t| t.stored_bytes).sum::<u64>(),
                    file["total_bytes"].as_u64().unwrap(),
                    "{}",
                    file["path"]
                );
                CensusFile {
                    path: PathBuf::from(file["path"].as_str().unwrap()),
                    tensors,
                }
            })
            .collect()
    }

    fn stored_dtype(name: &str, gguf: bool) -> StoredDType {
        if gguf {
            StoredDType::Ggml(match name {
                "F32" => GgmlDType::F32,
                "F16" => GgmlDType::F16,
                "BF16" => GgmlDType::BF16,
                "Q8_0" => GgmlDType::Q8_0,
                "Q4_0" => GgmlDType::Q4_0,
                "Q4_K" => GgmlDType::Q4K,
                "Q6_K" => GgmlDType::Q6K,
                other => panic!("unexpected GGUF dtype {other}"),
            })
        } else {
            StoredDType::Dense(match name {
                "F32" => DType::F32,
                "F16" => DType::F16,
                "BF16" => DType::BF16,
                "I64" => DType::I64,
                other => panic!("unexpected dtype {other}"),
            })
        }
    }

    fn census_tensor(name: String, dtype: StoredDType, shape: Vec<usize>) -> CensusTensor {
        let elements = shape.iter().product::<usize>() as u64;
        let stored_bytes = match dtype {
            StoredDType::Dense(dtype) => elements * dtype.size_in_bytes() as u64,
            StoredDType::Ggml(dtype) => {
                elements / dtype.block_size() as u64 * dtype.type_size() as u64
            }
            StoredDType::Unsupported => unreachable!(),
        };
        CensusTensor {
            name,
            dtype,
            shape,
            stored_bytes,
        }
    }

    macro_rules! census {
        ($file:literal) => {
            fixture(include_str!(concat!(
                "../../../tests/fixtures/asr_residency/",
                $file
            )))
        };
    }

    /// A device profile standing in for `backend`'s (dtype resolvers read only
    /// the kind and capabilities): CUDA reports BF16 and F16, Metal F16.
    pub(in super::super) fn profile(backend: BackendKind) -> DeviceProfile {
        let kind = match backend {
            BackendKind::Cpu => DeviceKind::Cpu,
            BackendKind::Metal => DeviceKind::Metal,
            BackendKind::Cuda => DeviceKind::Cuda,
        };
        DeviceProfile {
            device: candle_core::Device::Cpu,
            kind,
            capabilities: DeviceCapabilities {
                supports_bf16: kind == DeviceKind::Cuda,
                supports_f16: kind != DeviceKind::Cpu,
                has_unified_memory: kind == DeviceKind::Metal,
                ..DeviceCapabilities::default()
            },
            memory_pool: None,
        }
    }

    pub(in super::super) fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn whisper(backend: BackendKind) -> AsrResidencyPlan {
        let dtype = whisper_weight_dtype(&profile(backend)).unwrap();
        price_whisper(&census!("whisper_large_v3_turbo.json"), backend, dtype).unwrap()
    }

    fn qwen3_asr(files: &[CensusFile], backend: BackendKind) -> AsrResidencyPlan {
        let (text_dtype, audio_dtype) = qwen3_asr_weight_dtypes(
            &profile(backend),
            ModelFamily::Qwen3Asr,
            true,
            true,
            Some(QWEN3_ASR_GGUF_CHECKPOINT_DTYPE),
        )
        .unwrap();
        let tensors = by_name(files);
        let hidden = dim(tensors["thinker.model.embed_tokens.weight"], 1).unwrap();
        let load = Qwen3AsrLoad {
            text_dtype,
            audio_dtype,
            qmatmul_text: qwen3_asr_use_gguf_qmatmul_text(&candle_core::Device::Cpu),
            packed_projections: qwen3_quantized_projection_grouping_policy(
                backend == BackendKind::Metal,
                hidden as usize,
                None,
            ),
        };
        price_qwen3_asr(files, backend, load).unwrap()
    }

    fn parakeet(backend: BackendKind) -> AsrResidencyPlan {
        let f16_projections =
            resolve_parakeet_projection_precision(&profile(backend)).dtype == DType::F16;
        price_nemo(
            &census!("parakeet_tdt_0_6b_v3.json"),
            backend,
            PARAKEET_WEIGHT_DTYPE,
            NemoFamily::Parakeet,
            f16_projections,
        )
        .unwrap()
    }

    fn nemotron(backend: BackendKind) -> AsrResidencyPlan {
        let profile = profile(backend);
        let dtype = resolve_nemotron_asr_dtype(&profile).unwrap().dtype;
        let f16_projections =
            resolve_nemotron_projection_precision(&profile, dtype).dtype == DType::F16;
        price_nemo(
            &census!("nemotron_3_5_asr_streaming_0_6b.json"),
            backend,
            dtype,
            NemoFamily::Nemotron,
            f16_projections,
        )
        .unwrap()
    }

    /// The Granite representation each backend resolves by default: Metal
    /// takes the F16 LM projections and head copy and the residual
    /// pre-scaling for F32 weights (`granite_speech/asr/runtime.rs` policies);
    /// every backend fuses Q/K/V and gate/up.
    fn granite_representation(backend: BackendKind) -> GraniteLoadRepresentation {
        let metal = backend == BackendKind::Metal;
        GraniteLoadRepresentation {
            f16_lm_head: metal,
            f16_qkv: metal,
            f16_mlp: metal,
            f16_attention_output: metal,
            qkv_fusion: true,
            gate_up_fusion: true,
            residual_prescale: metal,
        }
    }

    fn granite(backend: BackendKind) -> AsrResidencyPlan {
        price_granite(
            &census!("granite_speech_4_1_2b_plus.json"),
            backend,
            &granite_load(backend),
        )
        .unwrap()
    }

    fn granite_load(backend: BackendKind) -> GraniteLoad {
        let dtype = select_granite_speech_dtype(&profile(backend), Some("bfloat16")).unwrap();
        // Metal loads the encoder projections as weight-only F16.
        let f16_encoder_projections = backend == BackendKind::Metal;
        GraniteLoad {
            dtype,
            audio_dtype: if f16_encoder_projections {
                DType::F32
            } else {
                dtype
            },
            f16_encoder_projections,
            representation: granite_representation(backend),
            // The published granite-speech-4.1-2b-plus config.json.
            residual_multiplier: 0.22,
            tied_lm_head: true,
            encoder_attention_dists: (16, 200),
        }
    }

    fn vibevoice(backend: BackendKind) -> AsrResidencyPlan {
        price_vibevoice(
            &census!("vibevoice_asr.json"),
            backend,
            &vibevoice_load(backend),
        )
        .unwrap()
    }

    fn vibevoice_load(backend: BackendKind) -> VibeVoiceLoad {
        let profile = profile(backend);
        let lm_dtype = vibevoice_asr_weight_dtype(&profile).unwrap();
        VibeVoiceLoad {
            lm_dtype,
            audio_dtype: vibevoice_asr_audio_dtype(&profile, lm_dtype),
            f16_tokenizer_projections: resolve_vibevoice_tokenizer_projection_precision(&profile)
                .dtype
                == DType::F16,
            fused_projections: qwen3_dense_projection_fusion_policy(
                backend == BackendKind::Metal,
                None,
            ),
            tts: false,
            staged: false,
        }
    }

    fn voxtral(backend: BackendKind) -> AsrResidencyPlan {
        let config_dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/voxtral_realtime_2602");
        let dtype = voxtral_realtime_weight_dtype(&config_dir, &profile(backend)).unwrap();
        price_voxtral(
            &census!("voxtral_mini_4b_realtime_2602.json"),
            backend,
            dtype,
        )
        .unwrap()
    }

    /// The published GGUF metadata: both LFM2 backbones precompute RoPE tables
    /// over 128,000 positions; head dims 2048 / 32 and 512 / 16.
    const LFM_ROPE_TABLES: [(usize, usize); 2] = [(128_000, 64), (128_000, 32)];

    /// LFM2.5-1.2B chat prices the quantized checkpoint once (its Q6_K
    /// `token_embd` serves the row gather and the tied LM head) plus RoPE
    /// tables over the 32K chat context.
    /// The generic estimate priced the GGUF bytes alone (728.5 MB on CPU and
    /// Metal) and the 2 GiB catalog figure on CUDA.
    #[test]
    fn lfm25_chat_prices_the_shared_quantized_table_and_rope_tables() {
        let files = census!("lfm2_5_1_2b_instruct_q4_k_m.json");
        let gguf = &files[0];
        assert_eq!(gguf.tensors.len(), 148);
        let stored: u64 = gguf.tensors.iter().map(|tensor| tensor.stored_bytes).sum();
        assert_eq!(stored, 728_509_440);
        let rope = 32_768 * 32 * 4 * 4;
        // The quantized table is held once, for the gather and the tied head.
        let logical = stored + rope;
        for backend in BACKENDS {
            let plan = price_lfm2_chat(gguf, backend, (32_768, 64)).unwrap();
            assert_eq!(plan.logical_resident_bytes, logical, "{backend:?}");
            assert!(plan.resident_bytes >= logical, "{backend:?}");
            assert!(
                plan.resident_bytes < logical + logical / 16,
                "{backend:?}: {}",
                plan.resident_bytes
            );
            assert!(plan.load_peak_bytes >= plan.resident_bytes, "{backend:?}");
            // Metal holds each load's upload copies until its fence; live
            // Metal loads peaked at 2.6 GB process footprint, runtime and the
            // KV arena included.
            let peak_bound = if backend == BackendKind::Metal { 2.5 } else { 1.5 };
            assert!(
                gib(plan.load_peak_bytes) < peak_bound,
                "{backend:?}: {}",
                plan.load_peak_bytes
            );
        }
    }

    fn lfm25_audio(backend: BackendKind) -> AsrResidencyPlan {
        let files = census!("lfm2_5_audio_1_5b_q4_0.json");
        let [main, mmproj, tokenizer, vocoder]: [CensusFile; 4] = files.try_into().unwrap();
        let census = Lfm25AudioCensus {
            main,
            mmproj,
            tokenizer,
            vocoder,
        };
        price_lfm25_audio(&census, backend, LFM_ROPE_TABLES).unwrap()
    }

    pub(in super::super) fn gib(bytes: u64) -> f64 {
        bytes as f64 / GIB
    }

    pub(in super::super) fn stored(
        files: &[CensusFile],
        keep: impl Fn(&CensusTensor) -> bool,
    ) -> u64 {
        files
            .iter()
            .flat_map(|file| file.tensors.iter())
            .filter(|tensor| keep(tensor))
            .map(|tensor| tensor.stored_bytes)
            .sum()
    }

    fn largest(files: &[CensusFile]) -> u64 {
        files
            .iter()
            .flat_map(|file| file.tensors.iter())
            .map(|tensor| tensor.stored_bytes)
            .max()
            .unwrap()
    }

    pub(in super::super) fn assert_gib(label: &str, bytes: u64, expected: f64) {
        let actual = gib(bytes);
        assert!(
            (actual - expected).abs() <= 0.015,
            "{label}: {actual:.3} GiB, expected {expected:.2} GiB"
        );
    }

    /// Per variant and backend (CPU, Metal, CUDA): resident weights at their
    /// exact size, and as allocated on Metal. The exact figures are the
    /// admission report's table B (per-variant dtype sources); Metal's
    /// allocator sizes are computed from the census here (the report gave
    /// only rough figures, and its Whisper 3.80 GiB left out the fused Q/K/V,
    /// which the Whisper audit's 4.38 GB included). Whisper, Granite and
    /// Voxtral cast on the host on Metal, so only their device-written
    /// tensors (fusions, pre-scales, computed tables) round up.
    #[test]
    fn plans_match_the_per_backend_residency_table() {
        let _env = env_lock();
        let qwen06 = census!("qwen3_asr_0_6b_q8_0.json");
        let qwen17 = census!("qwen3_asr_1_7b_q8_0.json");
        type Row<'a> = (
            &'a str,
            Box<dyn Fn(BackendKind) -> AsrResidencyPlan + 'a>,
            [f64; 4],
        );
        let rows: Vec<Row> = vec![
            (
                "Parakeet-TDT-0.6B-v3",
                Box::new(parakeet),
                [2.34, 1.20, 1.20, 2.34],
            ),
            (
                "Nemotron-3.5-ASR-0.6B",
                Box::new(nemotron),
                [2.38, 1.24, 1.24, 1.20],
            ),
            (
                "Whisper-L-v3-Turbo",
                Box::new(whisper),
                [3.01, 1.51, 1.74, 1.51],
            ),
            (
                "Qwen3-ASR-0.6B",
                Box::new(|b| qwen3_asr(&qwen06, b)),
                [1.86, 1.23, 1.55, 1.57],
            ),
            (
                "Qwen3-ASR-1.7B",
                Box::new(|b| qwen3_asr(&qwen17, b)),
                [4.05, 2.87, 3.30, 3.47],
            ),
            (
                "Granite-Speech-Plus",
                Box::new(granite),
                [7.87, 4.78, 5.06, 3.93],
            ),
            (
                "VibeVoice-ASR",
                Box::new(vibevoice),
                [31.03, 15.72, 28.35, 15.52],
            ),
            (
                "Voxtral-Mini-4B-RT",
                Box::new(voxtral),
                [16.50, 8.25, 8.25, 8.25],
            ),
            // The main backbone gathers token rows from its quantized table
            // (shared with its tied head) instead of a 0.5 GiB F32 copy.
            (
                "LFM2.5-Audio-1.5B",
                Box::new(lfm25_audio),
                [1.46, 1.46, 1.46, 1.46],
            ),
        ];
        for (label, plan, [cpu, metal_exact, metal_allocated, cuda]) in rows {
            let [on_cpu, on_metal, on_cuda] = BACKENDS.map(&plan);
            eprintln!(
                "{label:22} resident (load peak) GiB: CPU {:.2} ({:.2}), Metal {:.2} exact / {:.2} allocated ({:.2}), CUDA {:.2} ({:.2}, host {:.2})",
                gib(on_cpu.resident_bytes),
                gib(on_cpu.load_peak_bytes),
                gib(on_metal.logical_resident_bytes),
                gib(on_metal.resident_bytes),
                gib(on_metal.load_peak_bytes),
                gib(on_cuda.resident_bytes),
                gib(on_cuda.load_peak_bytes),
                gib(on_cuda.host_staging_bytes),
            );
            assert_gib(&format!("{label} CPU"), on_cpu.resident_bytes, cpu);
            assert_eq!(on_cpu.resident_bytes, on_cpu.logical_resident_bytes);
            assert_gib(
                &format!("{label} Metal"),
                on_metal.logical_resident_bytes,
                metal_exact,
            );
            assert_gib(
                &format!("{label} Metal allocated"),
                on_metal.resident_bytes,
                metal_allocated,
            );
            assert_gib(&format!("{label} CUDA"), on_cuda.resident_bytes, cuda);
            for plan in [on_cpu, on_metal, on_cuda] {
                assert!(plan.load_peak_bytes >= plan.resident_bytes, "{label}");
                assert!(
                    plan.resident_bytes >= plan.logical_resident_bytes,
                    "{label}"
                );
            }
        }
    }

    /// The census totals pin the exact figures independently of the plan.
    #[test]
    fn exact_residency_follows_the_loaders_dtype_and_skips() {
        let _env = env_lock();
        // Parakeet: every F32 tensor but the featurizer, on every backend,
        // plus each depthwise convolution with its BatchNorm folded in. Metal
        // stores the encoder projections in F16: the subsampling output
        // projection and, per layer, four FFN, five attention and two
        // pointwise-conv weights.
        let files = census!("parakeet_tdt_0_6b_v3.json");
        let weights = stored(&files, |t| {
            t.dtype == StoredDType::Dense(DType::F32) && !t.name.starts_with("preprocessor.")
        });
        let folds = 24 * (1024 * 9 + 1024) * 4;
        for backend in [BackendKind::Cpu, BackendKind::Cuda] {
            assert_eq!(parakeet(backend).logical_resident_bytes, weights + folds);
        }
        let projections = 1024 * 4096 + 24 * (4 * 4096 * 1024 + 5 * 1024 * 1024 + 3 * 1024 * 1024);
        assert_eq!(
            parakeet(BackendKind::Metal).logical_resident_bytes,
            weights + folds - projections * 2
        );
        // About 2.5 GB of weights become about 1.3 GB.
        assert_eq!(weights, 2_508_229_144);
        assert_eq!(weights - projections * 2, 1_291_880_984);
        // Nemotron: F32 on CPU and Metal; BF16 on CUDA plus transposed copies
        // of the two LSTM layers' input and hidden weights. Metal stores the
        // encoder projections in F16, as for Parakeet (a 4352-wide
        // subsampling output projection).
        let files = census!("nemotron_3_5_asr_streaming_0_6b.json");
        let weights = stored(&files, |t| !t.name.starts_with("preprocessor."));
        assert_eq!(nemotron(BackendKind::Cpu).resident_bytes, weights);
        let projections = 1024 * 4352 + 24 * (4 * 4096 * 1024 + 5 * 1024 * 1024 + 3 * 1024 * 1024);
        assert_eq!(
            nemotron(BackendKind::Metal).logical_resident_bytes,
            weights - projections * 2
        );
        assert_eq!(
            nemotron(BackendKind::Cuda).resident_bytes,
            weights / 2 + 4 * 2560 * 640 * 2
        );
        for backend in BACKENDS {
            let plan = nemotron(backend);
            assert_eq!(plan.memo_bytes, NEMOTRON_MODEL_MEMO_MAX_BYTES);
            assert_eq!(
                plan.estimate().unwrap().resident_bytes,
                plan.resident_bytes + NEMOTRON_MODEL_MEMO_MAX_BYTES
            );
        }
        // Whisper: F16 on CUDA and Metal, F32 on CPU. The computed encoder
        // table replaces the checkpoint's; each fused self-attention adds a
        // zero K bias.
        let files = census!("whisper_large_v3_turbo.json");
        let f16 = stored(&files, |_| true);
        let zero_biases = 36 * 1280;
        assert_eq!(
            whisper(BackendKind::Cpu).resident_bytes,
            2 * f16 + 4 * zero_biases
        );
        assert_eq!(
            whisper(BackendKind::Cuda).resident_bytes,
            f16 + 2 * zero_biases
        );
        assert_eq!(
            whisper(BackendKind::Metal).logical_resident_bytes,
            f16 + 2 * zero_biases
        );
        // VibeVoice and Voxtral keep their BF16 checkpoint as is on CUDA.
        let files = census!("vibevoice_asr.json");
        let asr_loaded = |t: &CensusTensor| !t.name.starts_with(VIBEVOICE_ASR_UNLOADED_DECODER);
        assert_eq!(
            vibevoice(BackendKind::Cuda).resident_bytes,
            stored(&files, asr_loaded)
        );
        assert_eq!(
            vibevoice(BackendKind::Cpu).resident_bytes,
            2 * stored(&files, asr_loaded)
        );
        // The acoustic decoder is TTS-only: about 0.64 GiB of BF16 not loaded.
        let decoder = stored(&files, |t| !asr_loaded(t));
        assert!(decoder > 600 << 20 && decoder < 700 << 20, "{decoder}");
        let files = census!("voxtral_mini_4b_realtime_2602.json");
        assert_eq!(
            voxtral(BackendKind::Cuda).resident_bytes,
            stored(&files, |_| true)
        );
        assert_eq!(
            voxtral(BackendKind::Metal).logical_resident_bytes,
            stored(&files, |_| true)
        );
        // Granite on Metal: the encoder's linear and pointwise-conv weights
        // are weight-only F16, 437M parameters (16 layers of FFN, attention
        // and pointwise convs, plus the input and CTC projections).
        let files = census!("granite_speech_4_1_2b_plus.json");
        let encoder_projections = stored(&files, |t| {
            granite_tensor_is_f16_encoder_projection(&t.name)
        }) / 2;
        assert_eq!(
            encoder_projections,
            16 * (2 * 2048 * 1024 + 2 * 1024 * 1024 + 4096 * 1024 + 4 * 4096 * 1024)
                + 1024 * 160
                + 2 * 348 * 1024
        );
        let dense_encoder = {
            let mut load = granite_load(BackendKind::Metal);
            load.audio_dtype = DType::F32;
            load.f16_encoder_projections = false;
            price_granite(&files, BackendKind::Metal, &load)
                .unwrap()
                .logical_resident_bytes
        };
        assert_eq!(
            dense_encoder - granite(BackendKind::Metal).logical_resident_bytes,
            encoder_projections * 2
        );
        // Granite on CUDA: BF16 without BatchNorm step counters, plus the
        // encoder's U32 attention distance tables.
        let files = census!("granite_speech_4_1_2b_plus.json");
        let bf16 = stored(&files, |t| t.dtype == StoredDType::Dense(DType::BF16));
        assert_eq!(
            granite(BackendKind::Cuda).resident_bytes,
            bf16 + 16 * 200 * 200 * 4
        );
    }

    /// On Metal Candle rounds every kernel-written buffer up to a power of
    /// two and keeps a cast tensor's stored-dtype upload until the load fence;
    /// host uploads keep their size.
    #[test]
    fn metal_prices_allocator_rounding_and_uploads_held_until_the_fence() {
        let _env = env_lock();
        let tensor = |name: &str, dtype, shape: Vec<usize>| {
            census_tensor(name.into(), StoredDType::Dense(dtype), shape)
        };
        // An F16 tensor cast to 6.6 MB of F32 takes an 8 MiB buffer; an F32
        // upload of the same tensor keeps its size.
        let f16 = tensor("w", DType::F16, vec![1280, 1280]);
        let f32 = tensor("w", DType::F32, vec![1280, 1280]);
        for (source, expected) in [(&f16, 8 << 20), (&f32, 1280 * 1280 * 4)] {
            let mut ledger = Ledger::new(BackendKind::Metal);
            ledger.var(source, Source::Mapped, &[DType::F32]).unwrap();
            let plan = AsrResidencyPlan::from_ledger(ledger, 0).unwrap();
            assert_eq!(plan.logical_resident_bytes, 1280 * 1280 * 4);
            assert_eq!(plan.resident_bytes, expected);
        }
        let mut ledger = Ledger::new(BackendKind::Cpu);
        ledger.var(&f16, Source::Mapped, &[DType::F32]).unwrap();
        assert_eq!(
            AsrResidencyPlan::from_ledger(ledger, 0)
                .unwrap()
                .resident_bytes,
            1280 * 1280 * 4
        );
        // Candle's safetensors backend never synchronizes before the fence:
        // a plain ledger holds every cast tensor's upload beside the
        // residents. The staged loaders (Whisper, Granite, Voxtral)
        // synchronize every interval of uploads instead, so their staging is
        // bounded by the interval and their largest tensors, not the
        // checkpoint.
        for (label, plan, files) in [
            (
                "Whisper",
                whisper(BackendKind::Metal),
                census!("whisper_large_v3_turbo.json"),
            ),
            (
                "Granite",
                granite(BackendKind::Metal),
                census!("granite_speech_4_1_2b_plus.json"),
            ),
            (
                "Voxtral",
                voxtral(BackendKind::Metal),
                census!("voxtral_mini_4b_realtime_2602.json"),
            ),
        ] {
            let uploads = stored(&files, |t| !t.name.ends_with("num_batches_tracked"));
            let staging = plan.load_peak_bytes - plan.resident_bytes;
            // Each bound term is at most an F32 copy of the largest tensor.
            let bound = METAL_LOAD_SYNC_INTERVAL_BYTES + 4 * 2 * largest(&files);
            assert!(
                staging < uploads && staging <= bound,
                "{label}: staging {staging}, uploads {uploads}, bound {bound}"
            );
        }
        // Whisper's live Metal load reached about 6.3 GB with the process
        // when every upload was held to the fence (5.61 GiB priced).
        assert_gib(
            "Whisper Metal load peak",
            whisper(BackendKind::Metal).load_peak_bytes,
            WHISPER_METAL_LOAD_PEAK_GIB,
        );
        assert_gib(
            "Granite Metal load peak",
            granite(BackendKind::Metal).load_peak_bytes,
            GRANITE_METAL_LOAD_PEAK_GIB,
        );
    }

    /// Staged: 5.61 and 10.11 GiB when every upload was held to the fence.
    /// Whisper's was 4.13 GiB staged in F32, before it loaded in F16.
    const WHISPER_METAL_LOAD_PEAK_GIB: f64 = 2.14;
    /// Granite's was 7.30 GiB before its encoder loaded weight-only F16.
    const GRANITE_METAL_LOAD_PEAK_GIB: f64 = 6.49;

    /// A staged Metal loader casts on the host, so its uploads keep their
    /// exact sizes, and it synchronizes every interval of uploads, so the
    /// held staging stops growing with the checkpoint.
    #[test]
    fn staged_metal_loads_upload_exact_sizes_and_hold_one_interval() {
        let tensor = |dtype, rows: usize| {
            census_tensor("w".into(), StoredDType::Dense(dtype), vec![rows, 1280])
        };
        // 6.6 MB of F32 from F16: an 8 MiB buffer when cast on the device,
        // its exact size when cast on the host.
        let f16 = tensor(DType::F16, 1280);
        let mut ledger = Ledger::staged(BackendKind::Metal);
        ledger.var(&f16, Source::Mapped, &[DType::F32]).unwrap();
        let plan = AsrResidencyPlan::from_ledger(ledger, 0).unwrap();
        assert_eq!(plan.resident_bytes, 1280 * 1280 * 4);
        // A chain's device steps still round and release the upload: 128
        // F16-then-pre-scaled 8 MiB projections release 1 GiB in all.
        let projection = tensor(DType::BF16, 4096);
        let projection_bytes = 4096 * 1280 * 2;
        let peak = |mut ledger: Ledger| {
            for _ in 0..128 {
                ledger
                    .var(&projection, Source::Mapped, &[DType::F16, DType::F16])
                    .unwrap();
            }
            AsrResidencyPlan::from_ledger(ledger, 0).unwrap()
        };
        let unsynced = peak(Ledger::new(BackendKind::Metal));
        let staged = peak(Ledger::staged(BackendKind::Metal));
        assert_eq!(staged.resident_bytes, unsynced.resident_bytes);
        let held = |plan: AsrResidencyPlan| plan.load_peak_bytes - plan.resident_bytes;
        assert!(held(unsynced) >= 128 * projection_bytes);
        // The interval, the step straddling the synchronization, and one
        // step's host copy and cast beside it.
        let bound = METAL_LOAD_SYNC_INTERVAL_BYTES + projection_bytes + 2 * projection_bytes;
        assert!(
            held(staged) <= bound,
            "staged held {} > bound {bound}",
            held(staged)
        );
        assert!(held(staged) >= METAL_LOAD_SYNC_INTERVAL_BYTES);
        // CPU and CUDA keep Candle's backend.
        for backend in [BackendKind::Cpu, BackendKind::Cuda] {
            assert_eq!(peak(Ledger::staged(backend)), peak(Ledger::new(backend)));
        }
    }

    /// CUDA prices the device from the plan and authorizes only host staging
    /// on the host, never the catalog's figure.
    #[test]
    fn cuda_plans_price_the_device_and_stage_only_on_the_host() {
        let _env = env_lock();
        let qwen06 = census!("qwen3_asr_0_6b_q8_0.json");
        let plans = [
            (ModelVariant::ParakeetTdt06BV3, parakeet(BackendKind::Cuda)),
            (
                ModelVariant::Nemotron35AsrStreaming06B,
                nemotron(BackendKind::Cuda),
            ),
            (
                ModelVariant::WhisperLargeV3Turbo,
                whisper(BackendKind::Cuda),
            ),
            (
                ModelVariant::Qwen3Asr06BGguf,
                qwen3_asr(&qwen06, BackendKind::Cuda),
            ),
            (
                ModelVariant::GraniteSpeech412BPlus,
                granite(BackendKind::Cuda),
            ),
            (ModelVariant::VibeVoiceAsr, vibevoice(BackendKind::Cuda)),
            (
                ModelVariant::VoxtralMini4BRealtime2602,
                voxtral(BackendKind::Cuda),
            ),
            (
                ModelVariant::Lfm25Audio15BGguf,
                lfm25_audio(BackendKind::Cuda),
            ),
        ];
        for (variant, plan) in plans {
            let catalog = super::super::model_memory_estimate(variant);
            let resource = plan.resource_plan().unwrap();
            let device = plan.estimate().unwrap();
            assert_eq!(
                resource.load_authorization.device_bytes,
                ResourceAmount::Known(device.load_peak_bytes)
            );
            assert_eq!(
                resource.resident_authorization.device_bytes,
                ResourceAmount::Known(device.resident_bytes)
            );
            assert_eq!(
                resource.load_authorization.host_bytes,
                ResourceAmount::Known(plan.host_staging_bytes)
            );
            assert!(
                device.load_peak_bytes < catalog.load_peak_bytes,
                "{variant}"
            );
            assert!(plan.host_staging_bytes < 1100 << 20, "{variant}");
            assert!(
                plan.host_staging_bytes < device.load_peak_bytes,
                "{variant}"
            );
        }
        // The largest tensor's host copy: VibeVoice's untied LM head.
        let files = census!("vibevoice_asr.json");
        assert_eq!(
            vibevoice(BackendKind::Cuda).host_staging_bytes,
            largest(&files)
        );
        // Parakeet's 12 GiB catalog figure (device and host) was 5x its weights.
        assert_gib(
            "Parakeet CUDA",
            parakeet(BackendKind::Cuda).load_peak_bytes,
            2.34,
        );
    }

    /// A dtype policy change reprices admission: the plans take the loaders'
    /// resolvers, overrides included.
    #[test]
    fn plans_follow_the_loader_dtype_overrides() {
        let _env = env_lock();
        struct Restore(&'static str);
        impl Drop for Restore {
            fn drop(&mut self) {
                std::env::remove_var(self.0);
            }
        }
        let default = vibevoice(BackendKind::Cuda).resident_bytes;
        std::env::set_var("IZWI_VIBEVOICE_ASR_DTYPE", "f32");
        let _restore = Restore("IZWI_VIBEVOICE_ASR_DTYPE");
        assert_eq!(vibevoice(BackendKind::Cuda).resident_bytes, 2 * default);
        // CPU resolves F32 whatever is requested; Metal honours the override
        // for the LM and keeps the audio path F32 with weight-only F16
        // tokenizer FFN weights (BF16 and F16 store the same bytes); the
        // TTS-only acoustic decoder is not loaded at all.
        std::env::set_var("IZWI_VIBEVOICE_ASR_DTYPE", "bf16");
        let files = census!("vibevoice_asr.json");
        let all = stored(&files, |t| {
            !t.name.starts_with(VIBEVOICE_ASR_UNLOADED_DECODER)
        });
        let lm = stored(&files, |t| vibevoice_tensor_is_language_model(&t.name));
        let projections = stored(&files, |t| {
            vibevoice_tensor_is_f16_tokenizer_projection(&t.name)
        });
        assert_eq!(vibevoice(BackendKind::Cpu).resident_bytes, 2 * all);
        assert_eq!(
            vibevoice(BackendKind::Metal).logical_resident_bytes,
            lm + 2 * (all - lm) - projections
        );
        // IZWI_VIBEVOICE_ASR_DTYPE=f32 keeps every Metal weight F32.
        std::env::set_var("IZWI_VIBEVOICE_ASR_DTYPE", "f32");
        assert_eq!(
            vibevoice(BackendKind::Metal).logical_resident_bytes,
            2 * all
        );
        // Whisper's Metal F16 default and its F32 escape hatches.
        let _restore_whisper = Restore("IZWI_WHISPER_DTYPE");
        let _restore_metal = Restore("IZWI_METAL_DTYPE");
        let f16 = whisper(BackendKind::Metal).logical_resident_bytes;
        let f32 = whisper(BackendKind::Cpu).resident_bytes;
        assert!(f16 < f32);
        std::env::set_var("IZWI_METAL_DTYPE", "f32");
        assert_eq!(whisper(BackendKind::Metal).logical_resident_bytes, f32);
        std::env::remove_var("IZWI_METAL_DTYPE");
        std::env::set_var("IZWI_WHISPER_DTYPE", "f32");
        assert_eq!(whisper(BackendKind::Metal).logical_resident_bytes, f32);
    }

    /// A Granite checkpoint with an untied LM head (the base variant) loads
    /// it from the checkpoint, in F16 on Metal, and makes no embedding copy.
    #[test]
    fn granite_prices_a_tied_head_copy_or_an_untied_head() {
        let tensor = |name: &str, shape: Vec<usize>| {
            census_tensor(name.into(), StoredDType::Dense(DType::BF16), shape)
        };
        let embedding = tensor("language_model.model.embed_tokens.weight", vec![1000, 64]);
        let head = tensor("language_model.lm_head.weight", vec![1000, 64]);
        let price = |tensors: Vec<CensusTensor>, tied| {
            let files = [CensusFile {
                path: PathBuf::from("model.safetensors"),
                tensors,
            }];
            let load = GraniteLoad {
                dtype: DType::F32,
                audio_dtype: DType::F32,
                f16_encoder_projections: true,
                representation: granite_representation(BackendKind::Metal),
                residual_multiplier: 0.22,
                tied_lm_head: tied,
                encoder_attention_dists: (0, 0),
            };
            price_granite(&files, BackendKind::Metal, &load)
                .unwrap()
                .logical_resident_bytes
        };
        let embedding_f32 = 1000 * 64 * 4;
        let head_f16 = 1000 * 64 * 2;
        assert_eq!(
            price(vec![embedding.clone()], true),
            embedding_f32 + head_f16
        );
        assert_eq!(
            price(vec![embedding, head], false),
            embedding_f32 + head_f16
        );
    }

    /// The CPU resolves the Granite switches the plan prices for the CPU.
    #[test]
    fn granite_cpu_representation_matches_the_loader() {
        let _env = env_lock();
        assert_eq!(
            granite_load_representation(&candle_core::Device::Cpu, DType::F32),
            granite_representation(BackendKind::Cpu)
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn granite_metal_representation_matches_the_loader() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "granite_metal_representation_matches_the_loader",
        ) else {
            return;
        };
        let _env = env_lock();
        assert_eq!(
            granite_load_representation(&device, DType::F32),
            granite_representation(BackendKind::Metal)
        );
    }

    pub(in super::super) fn temp_dir(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "izwi-asr-residency-{label}-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Writes a small BF16 VibeVoice-shaped `model.safetensors`.
    pub(in super::super) fn write_vibevoice_checkpoint(dir: &Path) -> u64 {
        let device = candle_core::Device::Cpu;
        let tensors = std::collections::HashMap::from([
            (
                "model.language_model.embed_tokens.weight".to_string(),
                candle_core::Tensor::zeros((512, 64), DType::BF16, &device).unwrap(),
            ),
            (
                "lm_head.weight".to_string(),
                candle_core::Tensor::zeros((512, 64), DType::BF16, &device).unwrap(),
            ),
            (
                "model.acoustic_tokenizer.encoder.head.weight".to_string(),
                candle_core::Tensor::zeros((32, 64), DType::BF16, &device).unwrap(),
            ),
        ]);
        candle_core::safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();
        (2 * 512 * 64 + 32 * 64) * 2
    }

    /// Admission reads the checkpoint the loader reads, and the Metal OOM
    /// retry judges the same plan.
    #[test]
    fn admission_and_the_oom_retry_price_the_same_plan() {
        let _env = env_lock();
        let dir = temp_dir("retry");
        let bf16 = write_vibevoice_checkpoint(&dir);
        let performance = crate::performance::PerformanceConfig::default();
        for backend in BACKENDS {
            let device = profile(backend);
            let plan = resource_plan(ModelVariant::VibeVoiceAsr, &dir, &device)
                .unwrap()
                .expect("VibeVoice has a residency plan");
            let admitted = super::super::load_domain_bytes(plan.load_authorization, backend);
            let retry = super::super::retry_load_peak_bytes(
                ModelVariant::VibeVoiceAsr,
                backend,
                &performance,
                Some((&dir, &device)),
            );
            assert_eq!(admitted, ResourceAmount::Known(retry), "{backend:?}");
            let catalog = super::super::model_memory_estimate(ModelVariant::VibeVoiceAsr);
            assert!(retry < catalog.load_peak_bytes);
            if backend == BackendKind::Cuda {
                assert_eq!(
                    plan.resident_authorization.device_bytes,
                    ResourceAmount::Known(bf16)
                );
                assert_eq!(
                    plan.load_authorization.host_bytes,
                    ResourceAmount::Known(512 * 64 * 2)
                );
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Variants without a plan, and checkpoints whose files are missing (the
    /// loader reports those), keep the generic estimate.
    #[test]
    fn variants_without_a_plan_keep_the_generic_estimate() {
        let _env = env_lock();
        let dir = temp_dir("fallback");
        let device = profile(BackendKind::Cuda);
        for variant in [ModelVariant::Kokoro82M, ModelVariant::Qwen306B] {
            assert!(residency_plan(variant, &dir, &device).unwrap().is_none());
        }
        for variant in [
            ModelVariant::VibeVoiceAsr,
            ModelVariant::WhisperLargeV3Turbo,
            ModelVariant::Qwen3Asr06BGguf,
            ModelVariant::ParakeetTdt06BV3,
            ModelVariant::Nemotron35AsrStreaming06B,
            ModelVariant::GraniteSpeech412BPlus,
            ModelVariant::VoxtralMini4BRealtime2602,
            ModelVariant::Lfm25Audio15BGguf,
        ] {
            assert!(
                residency_plan(variant, &dir, &device).unwrap().is_none(),
                "{variant}"
            );
        }
        let performance = crate::performance::PerformanceConfig::default();
        assert_eq!(
            super::super::retry_load_peak_bytes(
                ModelVariant::Kokoro82M,
                BackendKind::Metal,
                &performance,
                Some((&dir, &device)),
            ),
            super::super::model_memory_estimate(ModelVariant::Kokoro82M).load_peak_bytes
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// The loaded runtime's admission takes the ASR plan on its backend.
    #[test]
    fn runtime_admission_prices_asr_loads_by_their_plan() {
        let _authority = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let _env = env_lock();
        let models_dir = temp_dir("runtime");
        let checkpoint = models_dir.join("VibeVoice-ASR");
        std::fs::create_dir_all(&checkpoint).unwrap();
        write_vibevoice_checkpoint(&checkpoint);
        let runtime = crate::runtime::service::RuntimeService::new(crate::config::EngineConfig {
            models_dir: models_dir.clone(),
            backend: crate::backends::BackendPreference::Cpu,
            ..crate::config::EngineConfig::default()
        })
        .unwrap();
        let admitted = runtime
            .model_lifecycle
            .model_resource_plan(ModelVariant::VibeVoiceAsr, &checkpoint)
            .unwrap();
        let device = runtime
            .model_lifecycle
            .backend_router
            .context()
            .device
            .clone();
        let plan = resource_plan(ModelVariant::VibeVoiceAsr, &checkpoint, &device)
            .unwrap()
            .unwrap();
        assert_eq!(admitted, plan);
        // F32 on the CPU: twice the BF16 checkpoint, not its stored bytes.
        let inventory = super::super::portable_model_memory_estimate(
            BackendKind::Cpu,
            ModelVariant::VibeVoiceAsr,
            &checkpoint,
        )
        .unwrap();
        assert_eq!(
            admitted.resident_authorization.host_bytes,
            ResourceAmount::Known(2 * inventory.resident_bytes)
        );
        drop(runtime);
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    /// A gzip `.nemo` reports its payload through the ISIZE trailer; a plain
    /// tar is its own payload.
    #[test]
    fn nemo_payload_reads_the_gzip_trailer() {
        use std::io::Write;
        let dir = temp_dir("nemo");
        let payload = vec![7u8; 3 << 20];
        let gzip = dir.join("model.nemo");
        let mut encoder = flate2::write::GzEncoder::new(
            std::fs::File::create(&gzip).unwrap(),
            flate2::Compression::fast(),
        );
        encoder.write_all(&payload).unwrap();
        encoder.finish().unwrap();
        // Highly compressible: far smaller than its payload.
        assert!(std::fs::metadata(&gzip).unwrap().len() < 1 << 20);
        assert_eq!(nemo_payload_bytes(&gzip).unwrap(), 3 << 20);
        let tar = dir.join("plain.nemo");
        std::fs::write(&tar, &payload).unwrap();
        assert_eq!(nemo_payload_bytes(&tar).unwrap(), 3 << 20);
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Before extraction a NeMo load is priced from the archive payload, after
    /// any host extraction buffer is freed.
    #[test]
    fn first_nemo_loads_price_the_payload_and_the_extraction_buffer() {
        let payload: u64 = 2_509_332_480;
        let extraction = payload.next_power_of_two();
        let cpu =
            price_nemo_first_load(BackendKind::Cpu, payload, DType::F32, extraction, 0).unwrap();
        assert_eq!(cpu.resident_bytes, payload);
        // Extraction precedes the weights: the peak is the larger of the two.
        assert_eq!(cpu.load_peak_bytes, extraction);
        let cuda = price_nemo_first_load(BackendKind::Cuda, payload, DType::BF16, 0, 7).unwrap();
        assert_eq!(cuda.resident_bytes, payload / 2);
        assert_eq!(cuda.memo_bytes, 7);
        assert_eq!(cuda.host_staging_bytes, NEMO_UNLISTED_TENSOR_BYTES);
    }
}
