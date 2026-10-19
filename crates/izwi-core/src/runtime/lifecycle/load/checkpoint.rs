//! Checkpoint census and model memory estimates: what a load must reserve
//! before its weights are instantiated.

use super::*;

pub(super) const QWEN38_PORTABLE_CONVERSION_SCRATCH_BYTES: u64 = 1024 * 1024 * 1024;

pub(super) const QWEN38_CUDA_HOST_CONVERSION_SCRATCH_BYTES: u64 = 8 * 1024 * 1024 * 1024;

pub(super) const QWEN38_CUDA_DEVICE_CONVERSION_SCRATCH_BYTES: u64 = 256 * 1024 * 1024;

/// Static estimate of the pinned Qwen3.8-27B checkpoint on `backend`, priced
/// by the Qwen3.8 residency policy over the checkpoint's tensor list (the
/// same pricing load admission applies to a real checkpoint). The CUDA arm
/// is not the CUDA admission path: without the device it cannot run the
/// native block-FP8 gates, so it prices Q8_0 (an upper bound).
pub(super) fn qwen38_representation_memory_estimate(
    backend: BackendKind,
    performance: &crate::performance::PerformanceConfig,
) -> ModelMemoryEstimate {
    qwen38_memory::static_estimate(backend, performance)
        .expect("the pinned Qwen3.8 inventory prices without overflow")
}

pub(super) fn model_memory_estimate(variant: ModelVariant) -> ModelMemoryEstimate {
    const GIB: u64 = 1024 * 1024 * 1024;

    let inference_bytes = (variant.memory_required_gb() as f64 * GIB as f64).ceil() as u64;
    let base = match variant {
        // The 5 GiB catalog value describes total inference memory, including
        // request-scoped activations and audio workspace that the coordinator
        // reserves separately. The GGUF loader retains about 2.25 GiB of
        // quantized/dequantized tensors; 3 GiB covers model-owned load overlap,
        // tokenizer metadata, allocator alignment, and steady residency.
        ModelVariant::Lfm25Audio15BGguf => ModelMemoryEstimate {
            load_peak_bytes: 3 * GIB,
            resident_bytes: 3 * GIB,
        },
        _ => ModelMemoryEstimate {
            load_peak_bytes: inference_bytes,
            resident_bytes: inference_bytes,
        },
    };
    let memo_bytes = match variant {
        ModelVariant::Nemotron35AsrStreaming06B => {
            crate::models::architectures::nemotron::asr::NEMOTRON_MODEL_MEMO_MAX_BYTES
        }
        ModelVariant::Kokoro82M => {
            crate::models::architectures::kokoro::KOKORO_MODEL_MEMO_MAX_BYTES
        }
        _ => 0,
    };
    ModelMemoryEstimate {
        load_peak_bytes: base
            .load_peak_bytes
            .checked_add(memo_bytes)
            .expect("catalog model load estimate overflowed"),
        resident_bytes: base
            .resident_bytes
            .checked_add(memo_bytes)
            .expect("catalog model residency estimate overflowed"),
    }
}

/// The residency plan first admission prices for a checkpoint-planned
/// family (Fish, ASR, TTS, LFM2.5 chat), or `None` for one without a plan. The Metal OOM
/// retry judges futility by the same plan.
pub(super) fn checkpoint_resource_plan(
    variant: ModelVariant,
    model_path: &Path,
    device: &crate::backends::DeviceProfile,
    granite: &crate::config::GraniteSpeechOptions,
) -> Result<Option<ModelResourcePlan>> {
    if variant == ModelVariant::FishAudioS2Pro {
        let memory = crate::models::architectures::fish_s2::weights::fish_s2_model_memory(
            model_path, device,
        )?;
        return fish_s2_resource_plan(BackendKind::from(device.kind), memory).map(Some);
    }
    // ASR, TTS, LFM2.5 chat and Qwen3.5 / Qwen3 chat (CPU and Metal) price what
    // their loader allocates; the catalog figure stays a display value for
    // them.
    if let Some(plan) = asr_memory::resource_plan(variant, model_path, device, granite)? {
        return Ok(Some(plan));
    }
    if variant.family() == crate::catalog::ModelFamily::Lfm2Chat {
        return asr_memory::lfm2_chat_resource_plan(
            variant,
            model_path,
            BackendKind::from(device.kind),
        );
    }
    if variant.family() == crate::catalog::ModelFamily::Qwen35Chat {
        return asr_memory::qwen35_chat_resource_plan(
            variant,
            model_path,
            BackendKind::from(device.kind),
        );
    }
    if variant.family() == crate::catalog::ModelFamily::Qwen3Chat {
        return asr_memory::qwen3_chat_resource_plan(variant, model_path, device);
    }
    tts_memory::resource_plan(variant, model_path, device)
}

/// Static load peak that decides whether a Metal OOM retry can fit. The
/// catalog's `memory_required_gb` is one coarse figure for every backend (80
/// GiB for Qwen3.8, 140 GiB for Qwen3.6, sized for the CPU F32 expansion);
/// these checkpoints use the per-backend representation estimate that load
/// admission prices instead. A planned checkpoint (`checkpoint`: its path and
/// the load device; Fish, ASR and TTS) is judged by the plan first admission
/// priced ([`checkpoint_resource_plan`], with the engine's Granite overrides).
pub(super) fn retry_load_peak_bytes(
    variant: ModelVariant,
    backend: BackendKind,
    performance: &crate::performance::PerformanceConfig,
    checkpoint: Option<(
        &Path,
        &crate::backends::DeviceProfile,
        &crate::config::GraniteSpeechOptions,
    )>,
) -> u64 {
    let catalog = || model_memory_estimate(variant).load_peak_bytes;
    if let Some((model_path, device, granite)) = checkpoint {
        if let Ok(Some(plan)) = checkpoint_resource_plan(variant, model_path, device, granite) {
            if let ResourceAmount::Known(bytes) =
                load_domain_bytes(plan.load_authorization, backend)
            {
                return bytes;
            }
        }
    }
    match variant {
        ModelVariant::Qwen3827BFp8
            if !crate::models::architectures::qwen38::native::synthetic_geometry_enabled() =>
        {
            qwen38_representation_memory_estimate(backend, performance).load_peak_bytes
        }
        ModelVariant::Qwen36Moe35BA3BFp8
            if !crate::models::architectures::qwen36moe::native::synthetic_geometry_enabled() =>
        {
            let peak = qwen36moe_memory::resource_plan(backend, performance)
                .map(|plan| load_domain_bytes(plan.load_authorization, backend));
            match peak {
                Ok(ResourceAmount::Known(bytes)) => bytes,
                _ => catalog(),
            }
        }
        _ => catalog(),
    }
}

pub(super) fn collect_checkpoint_files(path: &Path, depth: usize, files: &mut Vec<PathBuf>) -> Result<()> {
    if path.is_file() {
        files.push(path.to_path_buf());
        return Ok(());
    }
    if depth == 0 || !path.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let child = entry.path();
        if entry.file_type()?.is_dir() {
            collect_checkpoint_files(&child, depth - 1, files)?;
        } else {
            files.push(child);
        }
    }
    Ok(())
}

/// How a checkpoint stores one tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StoredDType {
    /// A safetensors or PyTorch tensor, which Candle loads as a dense tensor.
    Dense(candle_core::DType),
    /// A GGUF tensor. Candle loads every GGUF tensor, F32 ones included, as a
    /// `QTensor`.
    Ggml(candle_core::quantized::GgmlDType),
    /// A safetensors dtype Candle cannot load: only its stored bytes are known.
    Unsupported,
}

/// One tensor as its checkpoint header records it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CensusTensor {
    pub(super) name: String,
    pub(super) dtype: StoredDType,
    pub(super) shape: Vec<usize>,
    /// Payload bytes in the file.
    pub(super) stored_bytes: u64,
}

impl CensusTensor {
    pub(super) fn elements(&self) -> u64 {
        u64::try_from(
            self.shape
                .iter()
                .try_fold(1_usize, |acc, &dim| acc.checked_mul(dim))
                .unwrap_or(usize::MAX),
        )
        .unwrap_or(u64::MAX)
    }
}

/// The header census of one checkpoint file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct CensusFile {
    pub(super) path: PathBuf,
    pub(super) tensors: Vec<CensusTensor>,
}

/// Header census of the checkpoint files under a model path: the tensor list
/// of every readable checkpoint, without materializing any payload.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct CheckpointCensus {
    pub(super) files: Vec<CensusFile>,
    /// Size of the largest container whose tensors cannot be listed (a `.nemo`
    /// archive, or a PTH file Candle cannot parse).
    pub(super) container_fallback_bytes: u64,
}

impl CheckpointCensus {
    pub(super) fn tensors(&self) -> impl Iterator<Item = &CensusTensor> {
        self.files.iter().flat_map(|file| file.tensors.iter())
    }

    pub(super) fn inventory(&self) -> Option<TensorStorageInventory> {
        if self.files.is_empty() {
            // The compressed artifact size is a conservative lower bound until
            // the extracted checkpoint header exists.
            let bytes = self.container_fallback_bytes;
            return (bytes > 0).then_some(TensorStorageInventory {
                total_bytes: bytes,
                largest_tensor_bytes: bytes,
                largest_tensor_elements: 0,
                tensor_count: 1,
            });
        }
        Some(
            self.tensors()
                .fold(TensorStorageInventory::default(), |inv, tensor| {
                    inv.push(tensor.stored_bytes, tensor.elements())
                }),
        )
    }
}

/// The header census of one checkpoint file, or `None` for a file that is not
/// a checkpoint or a PTH file Candle cannot parse.
pub(super) fn census_checkpoint_file(file: &Path) -> Result<Option<CensusFile>> {
    let tensors = match file.extension().and_then(|value| value.to_str()) {
        Some("gguf") => crate::models::shared::weights::gguf::GgufLoader::from_path(file)?
            .tensor_census()?
            .into_iter()
            .map(|tensor| CensusTensor {
                name: tensor.name,
                dtype: StoredDType::Ggml(tensor.dtype),
                shape: tensor.shape,
                stored_bytes: tensor.bytes,
            })
            .collect(),
        Some("safetensors") => {
            // SAFETY: Candle owns the read-only mapping for the lifetime of
            // the parsed tensor views read below.
            let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(file) }?;
            tensors
                .tensors()
                .into_iter()
                .map(|(name, view)| {
                    let stored_bytes = u64::try_from(view.data().len()).map_err(|_| {
                        Error::ModelLoadError("safetensors tensor size exceeds u64".into())
                    })?;
                    Ok(CensusTensor {
                        name,
                        dtype: candle_core::DType::try_from(view.dtype())
                            .map_or(StoredDType::Unsupported, StoredDType::Dense),
                        shape: view.shape().to_vec(),
                        stored_bytes,
                    })
                })
                .collect::<Result<Vec<_>>>()?
        }
        Some("pth" | "ckpt") => {
            let Ok(infos) = candle_core::pickle::read_pth_tensor_info(file, false, None) else {
                return Ok(None);
            };
            infos
                .into_iter()
                .map(|info| {
                    let elements =
                        u64::try_from(info.layout.shape().elem_count()).unwrap_or(u64::MAX);
                    CensusTensor {
                        name: info.name,
                        dtype: StoredDType::Dense(info.dtype),
                        shape: info.layout.shape().dims().to_vec(),
                        stored_bytes: elements.saturating_mul(info.dtype.size_in_bytes() as u64),
                    }
                })
                .collect()
        }
        _ => return Ok(None),
    };
    Ok(Some(CensusFile {
        path: file.to_path_buf(),
        tensors,
    }))
}

pub(super) fn checkpoint_census(path: &Path) -> Result<CheckpointCensus> {
    let mut files = Vec::new();
    collect_checkpoint_files(path, 3, &mut files)?;
    let mut census = CheckpointCensus::default();
    for file in files {
        let extension = file.extension().and_then(|value| value.to_str());
        let container = match extension {
            // NeMo is a tar container (gzip-compressed for some checkpoints).
            Some("nemo") => true,
            _ => match census_checkpoint_file(&file)? {
                Some(parsed) => {
                    census.files.push(parsed);
                    false
                }
                None => matches!(extension, Some("pth" | "ckpt")),
            },
        };
        if container {
            census.container_fallback_bytes =
                census.container_fallback_bytes.max(file.metadata()?.len());
        }
    }
    Ok(census)
}

pub(super) fn checkpoint_tensor_inventory(path: &Path) -> Result<Option<TensorStorageInventory>> {
    Ok(checkpoint_census(path)?.inventory())
}

/// Per-tensor instantiation slack: device buffer page rounding, storage
/// headers, and allocator slack for every materialized weight tensor. Only
/// becomes material at MoE scale (thousands of same-sized expert tensors);
/// negligible for the few hundred tensors of a dense checkpoint.
pub(super) const PER_TENSOR_INSTANTIATION_SLACK_BYTES: u64 = 32 * 1024;

/// Worst-case load-time destination expansion of one tensor: a quantized
/// source dequantizes to F32 (4 bytes/element) and the source bytes stay
/// alive while the destination is built.
pub(super) const SINGLE_TENSOR_F32_DESTINATION_BYTES_PER_ELEMENT: u64 = 4;

pub(super) fn estimate_from_tensor_inventory(
    catalog: ModelMemoryEstimate,
    inventory: Option<TensorStorageInventory>,
) -> Result<ModelMemoryEstimate> {
    let Some(inventory) = inventory else {
        return Ok(catalog);
    };
    if inventory.total_bytes == 0 {
        return Ok(catalog);
    }
    // Size-class bound for one tensor's working copy (the historic term).
    let single_tensor_working_copy = inventory
        .largest_tensor_bytes
        .checked_next_power_of_two()
        .unwrap_or(inventory.largest_tensor_bytes);
    // Conversion bound for a dequantizing loader: source bytes plus the F32
    // destination of the largest tensor. Dominates the size-class bound only
    // when the source is sub-F32 (quantized or half-precision checkpoints).
    let single_tensor_conversion_bound = inventory.largest_tensor_bytes.saturating_add(
        inventory
            .largest_tensor_elements
            .saturating_mul(SINGLE_TENSOR_F32_DESTINATION_BYTES_PER_ELEMENT),
    );
    // Allocation-count bound: instantiation overhead of the whole checkpoint.
    // A MoE-shaped checkpoint (many same-sized expert tensors) collapses the
    // largest-tensor terms; this term keeps its load scratch reserved.
    let whole_checkpoint_instantiation = inventory
        .tensor_count
        .saturating_mul(PER_TENSOR_INSTANTIATION_SLACK_BYTES);
    let load_scratch_bytes = single_tensor_working_copy
        .max(single_tensor_conversion_bound)
        .max(whole_checkpoint_instantiation);
    let load_peak_bytes = inventory
        .total_bytes
        .checked_add(load_scratch_bytes)
        .ok_or_else(|| Error::ModelLoadError("portable model load estimate overflow".into()))?;
    Ok(ModelMemoryEstimate {
        load_peak_bytes,
        resident_bytes: inventory.total_bytes,
    })
}

pub(super) fn portable_model_memory_estimate(
    backend: BackendKind,
    variant: ModelVariant,
    model_path: &Path,
) -> Result<ModelMemoryEstimate> {
    if variant == ModelVariant::Qwen3827BFp8 {
        return Ok(qwen38_representation_memory_estimate(
            backend,
            &crate::performance::PerformanceConfig::default(),
        ));
    }
    let catalog = model_memory_estimate(variant);
    estimate_from_tensor_inventory(catalog, checkpoint_tensor_inventory(model_path)?)
}

pub(super) fn model_resource_plan(backend: BackendKind, estimate: ModelMemoryEstimate) -> ModelResourcePlan {
    let mut resident_authorization = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => {
            resident_authorization.host_bytes = ResourceAmount::Known(estimate.resident_bytes);
        }
        BackendKind::Metal => {
            resident_authorization.unified_bytes = ResourceAmount::Known(estimate.resident_bytes);
        }
        BackendKind::Cuda => {
            resident_authorization.device_bytes = ResourceAmount::Known(estimate.resident_bytes);
        }
    }

    let mut load_authorization = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => {
            load_authorization.host_bytes = ResourceAmount::Known(estimate.load_peak_bytes);
        }
        BackendKind::Metal => {
            load_authorization.unified_bytes = ResourceAmount::Known(estimate.load_peak_bytes);
        }
        BackendKind::Cuda => {
            load_authorization.device_bytes = ResourceAmount::Known(estimate.load_peak_bytes);
        }
    }
    if backend == BackendKind::Cuda {
        // CUDA loaders materialize host-side artifact/tensor state before or
        // while copying the resident weights to the device. Authorize both
        // peaks up front; the host component is shed after publication.
        load_authorization.host_bytes = ResourceAmount::Known(estimate.load_peak_bytes);
    }

    ModelResourcePlan {
        load_authorization,
        resident_authorization,
        deferred_resident_authorization: ResourceVector::zero(),
    }
}

/// Fixture-mode qwen38 estimate (synthetic geometry opt-in): derived from the
/// actual checkpoint inventory instead of the pinned 27B constants, while
/// reserving the same portable conversion scratch the pinned estimate carries.
pub(super) fn qwen38_synthetic_fixture_estimate(model_path: &Path) -> Result<ModelMemoryEstimate> {
    let overflow = || Error::ModelLoadError("Qwen3.8 fixture memory estimate overflow".into());
    let Some(inventory) = checkpoint_tensor_inventory(model_path)? else {
        return Err(Error::ModelLoadError(
            "Synthetic Qwen3.8 fixture has no readable tensor inventory".into(),
        ));
    };
    // The portable load path materializes expanded F32 projections from the
    // FP8/BF16 source bytes; a 4x envelope covers the resident expansion and
    // the portable conversion scratch bounds the load peak like the pinned
    // estimate does.
    let resident_bytes = inventory.total_bytes.checked_mul(4).ok_or_else(overflow)?;
    let load_peak_bytes = resident_bytes
        .checked_add(QWEN38_PORTABLE_CONVERSION_SCRATCH_BYTES)
        .and_then(|bytes| bytes.checked_add(inventory.largest_tensor_bytes.next_power_of_two()))
        .ok_or_else(overflow)?;
    Ok(ModelMemoryEstimate {
        load_peak_bytes,
        resident_bytes,
    })
}

pub(super) fn fish_s2_resource_plan(
    backend: BackendKind,
    memory: crate::models::architectures::fish_s2::weights::FishS2ModelMemory,
) -> Result<ModelResourcePlan> {
    let mut plan = model_resource_plan(
        backend,
        ModelMemoryEstimate {
            load_peak_bytes: memory.load_peak_bytes,
            resident_bytes: memory.resident_bytes,
        },
    );
    if backend == BackendKind::Cuda {
        plan.load_authorization.host_bytes =
            ResourceAmount::Known(memory.cuda_host_load_peak_bytes);
    }
    // Reference VQ codes are immutable CPU data owned by this model instance,
    // even when weights reside on CUDA or Metal. Reserve their complete bounded
    // allocation before load and keep it charged until model unload.
    let reference_cache = ResourceVector {
        host_bytes: ResourceAmount::Known(
            crate::models::architectures::fish_s2::FISH_S2_REFERENCE_CACHE_BYTES,
        ),
        ..ResourceVector::zero()
    };
    plan.load_authorization = plan.load_authorization.checked_add(reference_cache)?;
    plan.resident_authorization = plan.resident_authorization.checked_add(reference_cache)?;
    // Cache entries are populated lazily. Keep their promise unmaterialized so
    // state fitting cannot consume this headroom before the first reference is
    // encoded. As with lazy graph caches, this remains conservative once filled.
    plan.deferred_resident_authorization = reference_cache;
    Ok(plan)
}
