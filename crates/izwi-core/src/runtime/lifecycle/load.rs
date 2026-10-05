use std::collections::{HashMap, HashSet};
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use futures::FutureExt;
use tracing::info;

use crate::backends::kv::managed_kv_backend_compiled;
use crate::backends::BackendKind;
use crate::config::ContextLengthPreference;
use crate::engine::{
    AdapterInstanceId, CacheMode, ModelInstanceId, ReservationClass, ReservationOwner,
    ResourceAmount, ResourceLease, ResourceVector, StageDescriptor,
};
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, CapabilityStateDescriptorV2, InferenceStateContract,
    InvocationStateCapacity, InvocationWorkspaceBindingV2, InvocationWorkspaceDomain,
    InvocationWorkspaceKeyV2, InvocationWorkspaceRuntimeV2, InvocationWorkspaceSet,
    RetainedStateCapability, RetainedStateRuntimeV2, RetainedStateUseV2, StateCapacityAxis,
    StateDomainId, StateDomainSpec, StateScope,
};
use crate::kv::InferenceStateContractProvider;
use crate::model::ModelVariant;
use crate::models::architectures::fish_s2::FishS2PhysicalStateSpec;
use crate::models::registry::NativeAsrModel;
use crate::models::shared::memory::metal::MetalPoolManager;
use crate::models::shared::weights::gguf::TensorStorageInventory;
use crate::runtime::adapters::{
    CapabilityKind, LoadedExecutionContract, LoadedModelBundleDraft, LoadedStatePublication,
};
use crate::runtime::lifecycle::controller::{
    ModelLifecycleController, SharedLoadFailure, SharedLoadOutcome,
};
use crate::runtime::service::RuntimeService;

#[path = "qwen35moe_memory.rs"]
mod qwen35moe_memory;
#[path = "qwen38_memory.rs"]
mod qwen38_memory;

pub(super) fn now_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

fn is_metal_command_buffer_oom(error: &Error) -> bool {
    let message = error.to_string().to_ascii_lowercase();
    message.contains("metal error")
        && (message.contains("insufficient memory")
            || message.contains("commandbuffercallbackerroroutofmemory")
            || message.contains("kiogpucommandbuffercallbackerroroutofmemory"))
}

fn select_lru_eviction_candidate(
    resident_variants: &[ModelVariant],
    requested_variant: ModelVariant,
    active_variants: &HashSet<ModelVariant>,
    pinned_variants: &HashSet<ModelVariant>,
    last_used: &HashMap<ModelVariant, u64>,
) -> Option<ModelVariant> {
    resident_variants
        .iter()
        .copied()
        .filter(|variant| {
            *variant != requested_variant
                && !active_variants.contains(variant)
                && !pinned_variants.contains(variant)
        })
        .min_by(|left, right| {
            last_used
                .get(left)
                .copied()
                .unwrap_or(0)
                .cmp(&last_used.get(right).copied().unwrap_or(0))
                .then_with(|| left.to_string().cmp(&right.to_string()))
        })
}

fn residency_budget_has_capacity(
    resident_variants: &[ModelVariant],
    requested_variant: ModelVariant,
    max_loaded_models: usize,
) -> bool {
    resident_variants.contains(&requested_variant) || resident_variants.len() < max_loaded_models
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoadedAsrStatePublicationRoute {
    Qwen3,
    VibeVoice,
    Whisper,
    Parakeet,
    NemotronOffline,
    LegacyCache,
}

fn loaded_asr_state_publication_route(variant: ModelVariant) -> LoadedAsrStatePublicationRoute {
    match variant.family() {
        crate::catalog::ModelFamily::Qwen3Asr => LoadedAsrStatePublicationRoute::Qwen3,
        crate::catalog::ModelFamily::VibeVoiceAsr => LoadedAsrStatePublicationRoute::VibeVoice,
        crate::catalog::ModelFamily::WhisperAsr => LoadedAsrStatePublicationRoute::Whisper,
        crate::catalog::ModelFamily::ParakeetAsr => LoadedAsrStatePublicationRoute::Parakeet,
        crate::catalog::ModelFamily::NemotronAsr => LoadedAsrStatePublicationRoute::NemotronOffline,
        _ => LoadedAsrStatePublicationRoute::LegacyCache,
    }
}

fn uses_asr_model_registry(variant: ModelVariant) -> bool {
    matches!(
        variant.family(),
        crate::catalog::ModelFamily::ParakeetAsr
            | crate::catalog::ModelFamily::WhisperAsr
            | crate::catalog::ModelFamily::Qwen3Asr
            | crate::catalog::ModelFamily::VibeVoiceAsr
            | crate::catalog::ModelFamily::NemotronAsr
            | crate::catalog::ModelFamily::GraniteSpeechAsr
            | crate::catalog::ModelFamily::Qwen3ForcedAligner
    )
}

fn kokoro_effective_context_tokens(
    model_context_tokens: usize,
    portable_context_tokens: usize,
) -> Result<u64> {
    let effective = model_context_tokens.min(portable_context_tokens);
    if effective == 0 {
        return Err(Error::ModelLoadError(
            "Kokoro effective context must be greater than zero".into(),
        ));
    }
    u64::try_from(effective)
        .map_err(|_| Error::ModelLoadError("Kokoro effective context exceeds u64".into()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PortableInvocationContextIntent {
    Automatic { ceiling: u64 },
    Explicit { selected: u64 },
}

fn portable_invocation_context_intent(
    preference: ContextLengthPreference,
    published_context: Option<usize>,
    maximum: u64,
) -> PortableInvocationContextIntent {
    match preference.explicit_tokens() {
        Some(tokens) => PortableInvocationContextIntent::Explicit {
            selected: maximum.min(tokens as u64),
        },
        None => PortableInvocationContextIntent::Automatic {
            ceiling: published_context
                .and_then(|tokens| u64::try_from(tokens).ok())
                .map_or(maximum, |tokens| maximum.min(tokens)),
        },
    }
}

fn automatic_state_group_budget(available_bytes: u64, remaining_groups: u64) -> u64 {
    available_bytes / remaining_groups.max(1)
}

fn portable_context_reserve_bytes(variant: ModelVariant, configured_reserve_bytes: u64) -> u64 {
    const GIB: u64 = 1024 * 1024 * 1024;

    // Synthetic fixture loads must not inherit the pinned 27B catalog
    // inference size: their actual residency is tiny, so subtracting it from
    // the 80 GiB catalog value would reserve an absurd request-scoped budget
    // and collapse the portable context fit.
    if variant == ModelVariant::Qwen3827BFp8
        && crate::models::architectures::qwen38::native::synthetic_geometry_enabled()
    {
        return configured_reserve_bytes;
    }
    if variant == ModelVariant::Qwen36Moe35BA3BFp8
        && crate::models::architectures::qwen35moe::native::synthetic_geometry_enabled()
    {
        return configured_reserve_bytes;
    }
    let total_inference_bytes = (variant.memory_required_gb() as f64 * GIB as f64).ceil() as u64;
    let resident_bytes = model_memory_estimate(variant).resident_bytes;
    configured_reserve_bytes.saturating_add(total_inference_bytes.saturating_sub(resident_bytes))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ManagedChatCapacityPolicy {
    /// `None` delegates staged transaction width to the engine-wide
    /// `max_staged_transactions` setting.
    staged_transaction_rows: Option<u32>,
    /// CUDA Qwen3.8 and Qwen3.5/3.6-MoE fit logical context against resident
    /// device headroom; this is independent of staged transaction width.
    fit_cuda_resident_context: bool,
}

fn managed_chat_capacity_policy(
    variant: ModelVariant,
    backend: BackendKind,
) -> ManagedChatCapacityPolicy {
    let family = variant.family();
    ManagedChatCapacityPolicy {
        staged_transaction_rows: None,
        fit_cuda_resident_context: matches!(
            family,
            crate::catalog::ModelFamily::Qwen38Chat | crate::catalog::ModelFamily::Qwen35MoeChat
        ) && backend == BackendKind::Cuda,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelResourcePlan {
    /// Maximum simultaneous memory authorized before physical instantiation.
    load_authorization: ResourceVector,
    /// Long-lived memory retained after publication completes.
    resident_authorization: ResourceVector,
    /// Resident capacity reserved for lazy model-owned allocations. It remains
    /// a pending claim while physical state is fitted before first inference.
    deferred_resident_authorization: ResourceVector,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelMemoryEstimate {
    /// Maximum model-owned memory while tensors are being instantiated.
    load_peak_bytes: u64,
    /// Long-lived model-owned memory after publication.
    resident_bytes: u64,
}

const QWEN38_FP8_ELEMENTS: u64 = 24_699_207_680;
const QWEN38_BF16_ELEMENTS: u64 = 3_082_220_272;
const QWEN38_PORTABLE_CONVERSION_SCRATCH_BYTES: u64 = 1024 * 1024 * 1024;
const QWEN38_CUDA_HOST_CONVERSION_SCRATCH_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const QWEN38_CUDA_DEVICE_CONVERSION_SCRATCH_BYTES: u64 = 256 * 1024 * 1024;
const QWEN38_Q8_0_BLOCK_ELEMENTS: u64 = 32;
const QWEN38_Q8_0_BLOCK_BYTES: u64 = 34;

fn qwen38_representation_memory_estimate(backend: BackendKind) -> ModelMemoryEstimate {
    let resident_bytes = match backend {
        BackendKind::Cpu => QWEN38_FP8_ELEMENTS
            .checked_add(QWEN38_BF16_ELEMENTS)
            .and_then(|elements| elements.checked_mul(4))
            .expect("Qwen3.8 F32 residency is a compile-time bounded value"),
        BackendKind::Metal => QWEN38_FP8_ELEMENTS
            .checked_add(QWEN38_BF16_ELEMENTS)
            .and_then(|elements| elements.checked_mul(2))
            .expect("Qwen3.8 F16 residency is a compile-time bounded value"),
        BackendKind::Cuda => {
            // Every source FP8 matrix uses 128x128 blocks, so its element count
            // is independently divisible by Q8_0's 32-element block width.
            // Keep ceil division here so a future inventory remains
            // conservative if that checkpoint invariant changes.
            let q8_blocks = QWEN38_FP8_ELEMENTS
                .checked_add(QWEN38_Q8_0_BLOCK_ELEMENTS - 1)
                .expect("Qwen3.8 Q8_0 block rounding is a compile-time bounded value")
                / QWEN38_Q8_0_BLOCK_ELEMENTS;
            let q8_bytes = q8_blocks
                .checked_mul(QWEN38_Q8_0_BLOCK_BYTES)
                .expect("Qwen3.8 Q8_0 residency is a compile-time bounded value");
            let bf16_bytes = QWEN38_BF16_ELEMENTS
                .checked_mul(2)
                .expect("Qwen3.8 BF16 residency is a compile-time bounded value");
            q8_bytes
                .checked_add(bf16_bytes)
                .expect("Qwen3.8 CUDA residency is a compile-time bounded value")
        }
    };
    let load_peak_bytes = match backend {
        BackendKind::Cpu | BackendKind::Metal => resident_bytes
            .checked_add(QWEN38_PORTABLE_CONVERSION_SCRATCH_BYTES)
            .expect("Qwen3.8 expanded load peak is a compile-time bounded value"),
        // Host-side F32 decode/conversion has a separate authorization. Device
        // load admission adds only the bounded QTensor quantize/upload overlap.
        BackendKind::Cuda => resident_bytes
            .checked_add(QWEN38_CUDA_DEVICE_CONVERSION_SCRATCH_BYTES)
            .expect("Qwen3.8 CUDA load peak is a compile-time bounded value"),
    };
    ModelMemoryEstimate {
        load_peak_bytes,
        resident_bytes,
    }
}

fn model_memory_estimate(variant: ModelVariant) -> ModelMemoryEstimate {
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

fn collect_checkpoint_files(path: &Path, depth: usize, files: &mut Vec<PathBuf>) -> Result<()> {
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

fn checkpoint_tensor_inventory(path: &Path) -> Result<Option<TensorStorageInventory>> {
    let mut files = Vec::new();
    collect_checkpoint_files(path, 3, &mut files)?;
    let mut found = false;
    let mut container_fallback = 0_u64;
    let mut aggregate = TensorStorageInventory::default();
    for file in files {
        let extension = file.extension().and_then(|value| value.to_str());
        let inventory = match extension {
            Some("gguf") => {
                let loader = crate::models::shared::weights::gguf::GgufLoader::from_path(&file)?;
                Some(loader.tensor_storage_inventory()?)
            }
            Some("safetensors") => {
                // SAFETY: Candle owns the read-only mapping for the lifetime of
                // the parsed tensor views returned below.
                let tensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(&file) }?;
                Some(tensors.tensors().into_iter().try_fold(
                    TensorStorageInventory::default(),
                    |inv, (_, tensor)| {
                        let bytes = u64::try_from(tensor.data().len()).map_err(|_| {
                            Error::ModelLoadError("safetensors tensor size exceeds u64".into())
                        })?;
                        let elements = u64::try_from(
                            tensor
                                .shape()
                                .iter()
                                .try_fold(1_usize, |acc: usize, &dim| acc.checked_mul(dim))
                                .unwrap_or(usize::MAX),
                        )
                        .map_err(|_| {
                            Error::ModelLoadError(
                                "safetensors tensor element count exceeds u64".into(),
                            )
                        })?;
                        Ok::<_, Error>(inv.push(bytes, elements))
                    },
                )?)
            }
            Some("pth" | "ckpt") => {
                let parsed = candle_core::pickle::read_pth_tensor_info(&file, false, None)
                    .ok()
                    .map(|infos| {
                        infos
                            .into_iter()
                            .fold(TensorStorageInventory::default(), |inv, info| {
                                let elements = u64::try_from(info.layout.shape().elem_count())
                                    .unwrap_or(u64::MAX);
                                let bytes =
                                    elements.saturating_mul(info.dtype.size_in_bytes() as u64);
                                inv.push(bytes, elements)
                            })
                    });
                if parsed.is_none() {
                    container_fallback = container_fallback.max(file.metadata()?.len());
                }
                parsed
            }
            // NeMo is a zip container. Its compressed artifact size is a
            // conservative lower bound until the repacked PTH header exists.
            Some("nemo") => {
                let bytes = file.metadata()?.len();
                container_fallback = container_fallback.max(bytes);
                None
            }
            _ => None,
        };
        if let Some(file_inventory) = inventory {
            found = true;
            aggregate = aggregate.merge(file_inventory);
        }
    }
    if !found && container_fallback > 0 {
        return Ok(Some(TensorStorageInventory {
            total_bytes: container_fallback,
            largest_tensor_bytes: container_fallback,
            largest_tensor_elements: 0,
            tensor_count: 1,
        }));
    }
    Ok(found.then_some(aggregate))
}

/// Per-tensor instantiation slack: device buffer page rounding, storage
/// headers, and allocator slack for every materialized weight tensor. Only
/// becomes material at MoE scale (thousands of same-sized expert tensors);
/// negligible for the few hundred tensors of a dense checkpoint.
const PER_TENSOR_INSTANTIATION_SLACK_BYTES: u64 = 32 * 1024;
/// Worst-case load-time destination expansion of one tensor: a quantized
/// source dequantizes to F32 (4 bytes/element) and the source bytes stay
/// alive while the destination is built.
const SINGLE_TENSOR_F32_DESTINATION_BYTES_PER_ELEMENT: u64 = 4;

fn estimate_from_tensor_inventory(
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

fn portable_model_memory_estimate(
    backend: BackendKind,
    variant: ModelVariant,
    model_path: &Path,
) -> Result<ModelMemoryEstimate> {
    if variant == ModelVariant::Qwen3827BFp8 {
        return Ok(qwen38_representation_memory_estimate(backend));
    }
    let catalog = model_memory_estimate(variant);
    estimate_from_tensor_inventory(catalog, checkpoint_tensor_inventory(model_path)?)
}

fn model_resource_plan(backend: BackendKind, estimate: ModelMemoryEstimate) -> ModelResourcePlan {
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

fn qwen38_resource_plan(backend: BackendKind) -> ModelResourcePlan {
    let estimate = qwen38_representation_memory_estimate(backend);
    let mut plan = model_resource_plan(backend, estimate);
    if backend == BackendKind::Cuda {
        // CUDA retains Q8_0 projections plus the checkpoint's BF16 tensors.
        // Host memory only holds the callback-scoped shard/dequantization and
        // requantization window, not a second full resident representation.
        plan.load_authorization.host_bytes =
            ResourceAmount::Known(QWEN38_CUDA_HOST_CONVERSION_SCRATCH_BYTES);
    }
    plan
}

/// Fixture-mode qwen38 estimate (synthetic geometry opt-in): derived from the
/// actual checkpoint inventory instead of the pinned 27B constants, while
/// reserving the same portable conversion scratch the pinned estimate carries.
fn qwen38_synthetic_fixture_estimate(model_path: &Path) -> Result<ModelMemoryEstimate> {
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

fn fish_s2_resource_plan(
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

#[derive(Debug, Clone)]
struct InvocationAllocationV2 {
    adapter_instance: AdapterInstanceId,
    key: InvocationWorkspaceKeyV2,
    domain: InvocationWorkspaceDomain,
    slot_count: u32,
}

fn invalid_invocation_publication(message: impl Into<String>) -> Error {
    Error::ModelLoadError(message.into())
}

fn validate_physical_publication_backing(
    descriptor: &CapabilityStateDescriptorV2,
    executions: &[LoadedExecutionContract],
    retained: Option<&RetainedStateRuntimeV2>,
    retained_uses: &HashMap<[u8; 32], RetainedStateUseV2>,
) -> Result<()> {
    let execution_graphs = executions
        .iter()
        .map(|execution| stage_graph_fingerprint(&execution.stages))
        .collect::<Result<HashSet<_>>>()?;
    match (&descriptor.retained, retained) {
        (RetainedStateCapability::Stateless, None) if retained_uses.is_empty() => Ok(()),
        (RetainedStateCapability::Managed { contract }, Some(retained))
            if contract.fingerprint()? == retained.state_plan_v2().contract_fingerprint
                && retained_uses.keys().copied().collect::<HashSet<_>>() == execution_graphs =>
        {
            Ok(())
        }
        (RetainedStateCapability::Stateless, None) => Err(invalid_invocation_publication(
            "invocation-only publication cannot declare retained graph mappings",
        )),
        (RetainedStateCapability::Stateless, Some(_)) => Err(invalid_invocation_publication(
            "invocation-only publication unexpectedly owns retained backing",
        )),
        (RetainedStateCapability::Managed { .. }, None) => Err(invalid_invocation_publication(
            "retained invocation publication is missing physical retained backing",
        )),
        (RetainedStateCapability::Managed { .. }, Some(_)) => Err(invalid_invocation_publication(
            "retained invocation publication has mismatched backing or graph mappings",
        )),
    }
}

/// Resolve one capability-authored invocation descriptor into exact physical
/// allocations. This is deliberately model-neutral: graph, stage, domain, and
/// concurrency identities all come from the sealed execution/state contracts.
fn plan_invocation_allocations(
    descriptor: &CapabilityStateDescriptorV2,
    invocation_contract: &InferenceStateContract,
    executions: &[LoadedExecutionContract],
) -> Result<Vec<InvocationAllocationV2>> {
    if executions.is_empty() {
        return Err(invalid_invocation_publication(
            "physical invocation publication has no execution stage graphs",
        ));
    }
    invocation_contract.validate()?;

    let mut contract_domains = HashMap::new();
    for domain in &invocation_contract.domains {
        if domain.scope() != StateScope::Invocation {
            return Err(invalid_invocation_publication(
                "physical invocation contract contains retained state",
            ));
        }
        if contract_domains.insert(domain.id(), domain).is_some() {
            return Err(invalid_invocation_publication(
                "physical invocation contract repeats a state domain",
            ));
        }
    }

    let mut executions_by_graph = HashMap::new();
    for execution in executions {
        let graph = stage_graph_fingerprint(&execution.stages)?;
        match executions_by_graph.entry(graph) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(execution);
            }
            std::collections::hash_map::Entry::Occupied(entry) => {
                let current = entry.get();
                if current.adapter_instance_id != execution.adapter_instance_id
                    || current.stages != execution.stages
                {
                    return Err(invalid_invocation_publication(
                        "one invocation stage graph maps to multiple loaded adapters",
                    ));
                }
            }
        }
    }

    let InvocationWorkspaceSet::Bounded { profiles } = &descriptor.invocation else {
        return Err(invalid_invocation_publication(
            "physical invocation publication has no bounded workspace profiles",
        ));
    };
    let execution_graphs = executions_by_graph.keys().copied().collect::<HashSet<_>>();
    let profile_graphs = profiles
        .iter()
        .map(|profile| profile.stage_graph_fingerprint)
        .collect::<HashSet<_>>();
    if profile_graphs.len() != profiles.len() || profile_graphs != execution_graphs {
        return Err(invalid_invocation_publication(
            "physical invocation profiles do not map exactly to the loaded stage graphs",
        ));
    }

    let mut mapped_domains: HashMap<StateDomainId, &StateDomainSpec> = HashMap::new();
    let mut physical_keys = HashSet::new();
    let mut allocations = Vec::new();
    for profile in profiles {
        let execution = executions_by_graph
            .get(&profile.stage_graph_fingerprint)
            .copied()
            .ok_or_else(|| {
                invalid_invocation_publication(
                    "physical invocation profile lost its loaded stage graph",
                )
            })?;
        descriptor.validate_against_stages(&execution.stages)?;
        for workspace in &profile.stages {
            let stage = execution
                .stages
                .iter()
                .find(|candidate| candidate.id == workspace.stage)
                .ok_or_else(|| {
                    invalid_invocation_publication(
                        "physical invocation profile lost its execution stage",
                    )
                })?;
            let slot_count = workspace.slot_count(stage.max_batch_size)?;
            for domain in &workspace.domains {
                let InvocationWorkspaceDomain::State {
                    state, capacity, ..
                } = domain
                else {
                    // Scratch is accounted by the stage workspace formula and
                    // has no persistent typed backing to allocate here.
                    continue;
                };
                let capacity_matches = matches!(state, StateDomainSpec::PagedAttention(_))
                    && capacity.paged_max_tokens().is_some()
                    || matches!(
                        (state, capacity),
                        (
                            StateDomainSpec::StaticAttention(_)
                                | StateDomainSpec::Tensor(_)
                                | StateDomainSpec::Append(_)
                                | StateDomainSpec::Ring(_)
                                | StateDomainSpec::StaticTensor(_),
                            InvocationStateCapacity::SemanticBounded
                        )
                    );
                if !capacity_matches {
                    return Err(invalid_invocation_publication(
                        "physical invocation descriptor uses a capacity incompatible with its state domain",
                    ));
                }
                match mapped_domains.entry(state.id()) {
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(state);
                    }
                    std::collections::hash_map::Entry::Occupied(entry) if *entry.get() == state => {
                    }
                    std::collections::hash_map::Entry::Occupied(_) => {
                        return Err(invalid_invocation_publication(
                            "one invocation domain has inconsistent physical definitions",
                        ));
                    }
                }
                let key = InvocationWorkspaceKeyV2 {
                    stage_graph: profile.stage_graph_fingerprint,
                    stage: workspace.stage,
                    domain: state.id(),
                };
                if !physical_keys.insert(key) {
                    return Err(invalid_invocation_publication(
                        "physical invocation publication repeats a graph/stage/domain mapping",
                    ));
                }
                allocations.push(InvocationAllocationV2 {
                    adapter_instance: execution.adapter_instance_id,
                    key,
                    domain: domain.clone(),
                    slot_count,
                });
            }
        }
    }

    if mapped_domains.len() != contract_domains.len()
        || mapped_domains.iter().any(|(id, state)| {
            contract_domains
                .get(id)
                .is_none_or(|contract_state| *contract_state != *state)
        })
    {
        return Err(invalid_invocation_publication(
            "physical invocation descriptor and contract have missing or extra domain mappings",
        ));
    }
    if allocations.is_empty() {
        return Err(invalid_invocation_publication(
            "physical invocation publication has no typed state domains",
        ));
    }
    Ok(allocations)
}

fn validate_scratch_only_invocation_publication(
    descriptor: &CapabilityStateDescriptorV2,
    executions: &[LoadedExecutionContract],
) -> Result<()> {
    if executions.is_empty() {
        return Err(invalid_invocation_publication(
            "scratch-only invocation publication has no execution stage graphs",
        ));
    }
    let execution_graphs = executions
        .iter()
        .map(|execution| stage_graph_fingerprint(&execution.stages))
        .collect::<Result<HashSet<_>>>()?;
    let profile_graphs = match &descriptor.invocation {
        InvocationWorkspaceSet::None {
            stage_graph_fingerprints,
        } => stage_graph_fingerprints
            .iter()
            .copied()
            .collect::<HashSet<_>>(),
        InvocationWorkspaceSet::Bounded { profiles } => {
            if profiles.iter().any(|profile| {
                profile.stages.iter().any(|stage| {
                    stage
                        .domains
                        .iter()
                        .any(|domain| matches!(domain, InvocationWorkspaceDomain::State { .. }))
                })
            }) {
                return Err(invalid_invocation_publication(
                    "scratch-only invocation publication contains typed state",
                ));
            }
            profiles
                .iter()
                .map(|profile| profile.stage_graph_fingerprint)
                .collect::<HashSet<_>>()
        }
    };
    if profile_graphs != execution_graphs {
        return Err(invalid_invocation_publication(
            "scratch-only invocation profiles do not map exactly to the loaded stage graphs",
        ));
    }
    for execution in executions {
        descriptor.validate_against_stages(&execution.stages)?;
    }
    Ok(())
}

impl ModelLifecycleController {
    /// Seal each advertised Fish route against its own adapter identity. Slow
    /// retained state is model-scoped and reused; invocation backing is per adapter.
    async fn load_fish_s2_state_publications(
        &self,
        model_instance_id: ModelInstanceId,
        bundle_draft: &LoadedModelBundleDraft,
        max_sequence_tokens: usize,
        capacity: &crate::runtime::fish_capacity::ResolvedFishServingCapacity,
        physical_state_spec: impl Fn(&[&[StageDescriptor]]) -> Result<FishS2PhysicalStateSpec>,
    ) -> Result<HashMap<CapabilityKind, LoadedStatePublication>> {
        let variant = ModelVariant::FishAudioS2Pro;
        let mut publications = HashMap::new();
        let mut retained_runtime: Option<Arc<crate::engine::ManagedKvModelRuntime>> = None;
        for capability in [CapabilityKind::Tts, CapabilityKind::StreamingTts] {
            let contracts = bundle_draft.execution_contracts(capability)?;
            let stage_graphs = contracts
                .iter()
                .map(|contract| contract.stages.as_ref())
                .collect::<Vec<_>>();
            let physical_spec = physical_state_spec(&stage_graphs)?;
            let retained_contract = physical_spec.retained.as_ref().ok_or_else(|| {
                Error::ModelLoadError(
                    "Fish S2 TTS normal graph did not publish retained state".into(),
                )
            })?;
            // Fit and allocate slow state once. Refitting after its allocation
            // would incorrectly price another copy against remaining headroom.
            let retained = match &retained_runtime {
                Some(retained) => {
                    if retained.state_plan_v2().contract_fingerprint
                        != retained_contract.fingerprint()?
                    {
                        return Err(Error::ModelLoadError(
                            "Fish capabilities disagree on retained state".into(),
                        ));
                    }
                    retained.clone()
                }
                None => {
                    let retained = self
                        .core_engine
                        .load_managed_model_state_with_row_limits(
                            model_instance_id,
                            retained_contract,
                            Some(max_sequence_tokens),
                            capacity.active_rows,
                            capacity.staged_rows,
                        )
                        .await?;
                    retained_runtime = Some(retained.clone());
                    retained
                }
            };
            self.model_registry
                .publish_effective_context(variant, retained.logical_token_reach())?;
            crate::runtime::rollout::validate_managed_state_plan_eligibility(
                variant,
                capability,
                retained.state_plan_v2(),
            )?;
            let retained_uses = contracts
                .iter()
                .map(|contract| {
                    let graph = stage_graph_fingerprint(&contract.stages)?;
                    let retained_use = match contract.execution_profile.cache_mode {
                        CacheMode::ExternalPaged => RetainedStateUseV2::ExternalPaged,
                        CacheMode::None => RetainedStateUseV2::Inactive,
                    };
                    Ok((graph, retained_use))
                })
                .collect::<Result<HashMap<_, _>>>()?;
            let publication = self
                .load_invocation_workspace_publication(
                    model_instance_id,
                    &contracts,
                    physical_spec.descriptor,
                    &physical_spec.invocation,
                    Some(retained.into()),
                    retained_uses,
                )
                .await?;
            publications.insert(capability, publication);
        }
        Ok(publications)
    }

    fn load_scratch_only_workspace_publication(
        &self,
        executions: &[LoadedExecutionContract],
        descriptor: CapabilityStateDescriptorV2,
        retained: Option<RetainedStateRuntimeV2>,
        retained_uses: HashMap<[u8; 32], RetainedStateUseV2>,
    ) -> Result<LoadedStatePublication> {
        validate_physical_publication_backing(
            &descriptor,
            executions,
            retained.as_ref(),
            &retained_uses,
        )?;
        validate_scratch_only_invocation_publication(&descriptor, executions)?;
        Ok(LoadedStatePublication::PhysicalV2 {
            descriptor,
            retained,
            retained_uses,
            invocation_workspace: InvocationWorkspaceRuntimeV2::default(),
        })
    }

    fn fit_invocation_decoder_context(
        &self,
        variant: ModelVariant,
        descriptor: &mut CapabilityStateDescriptorV2,
        invocation_contract: &InferenceStateContract,
        executions: &[LoadedExecutionContract],
        remaining_automatic_state_groups: u64,
    ) -> Result<()> {
        let safety_reserve_bytes =
            portable_context_reserve_bytes(variant, self.config.portable_context_reserve_bytes);
        let backend = self.backend_router.context().backend_kind;
        if backend == BackendKind::Cuda {
            return Ok(());
        }
        let axis_bounds = descriptor.capacity_axis_bounds(StateCapacityAxis::DecoderContext);
        let ResourceAmount::Known(headroom_bytes) = self
            .coordinator
            .resource_authority()
            .planning_headroom_bytes(backend)?
        else {
            let Some((_, maximum)) = axis_bounds else {
                return Ok(());
            };
            let fallback = self
                .config
                .max_sequence_length
                .explicit_tokens()
                .map_or(maximum.min(4_096), |tokens| maximum.min(tokens as u64));
            descriptor.resolve_capacity_axis(StateCapacityAxis::DecoderContext, fallback)?;
            self.model_registry
                .publish_effective_context(variant, fallback)?;
            return Ok(());
        };
        let budget = headroom_bytes.saturating_sub(safety_reserve_bytes);
        let automatic_budget =
            automatic_state_group_budget(budget, remaining_automatic_state_groups);
        let required = |tokens: Option<u64>| -> Result<u64> {
            let mut candidate = descriptor.clone();
            if let Some(tokens) = tokens {
                candidate.resolve_capacity_axis(StateCapacityAxis::DecoderContext, tokens)?;
            }
            plan_invocation_allocations(&candidate, invocation_contract, executions)?
                .into_iter()
                .try_fold(0_u64, |total, allocation| {
                    total
                        .checked_add(
                            allocation
                                .domain
                                .maximum_bytes()?
                                .checked_mul(u64::from(allocation.slot_count))
                                .ok_or_else(|| {
                                    Error::ModelLoadError(
                                        "invocation context byte plan overflow".into(),
                                    )
                                })?,
                        )
                        .ok_or_else(|| {
                            Error::ModelLoadError("invocation context byte plan overflow".into())
                        })
                })
        };
        let Some((minimum, maximum)) = axis_bounds else {
            let bytes = required(None)?;
            if bytes > budget {
                return Err(Error::ModelLoadError(format!(
                    "portable fixed invocation state does not fit: state_bytes={bytes}, planning_headroom={headroom_bytes}, safety_reserve={safety_reserve_bytes}"
                )));
            }
            return Ok(());
        };
        // The invocation context ceiling is the model-authored maximum; the
        // memory-budgeted binary search below decides how much of it fits.
        let intent = portable_invocation_context_intent(
            self.config.max_sequence_length,
            self.model_registry.effective_context(variant),
            maximum,
        );
        let selected = if let PortableInvocationContextIntent::Explicit { selected } = intent {
            let bytes = required(Some(selected))?;
            if bytes > budget {
                return Err(Error::ModelLoadError(format!(
                    "explicit portable context does not fit: context={selected}, state_bytes={bytes}, planning_headroom={headroom_bytes}, safety_reserve={safety_reserve_bytes}"
                )));
            }
            selected
        } else {
            let PortableInvocationContextIntent::Automatic { ceiling } = intent else {
                unreachable!("portable invocation context intent was matched above")
            };
            if ceiling < minimum {
                return Err(Error::ModelLoadError(format!(
                    "published automatic context is below the invocation minimum: context={ceiling}, minimum={minimum}"
                )));
            }
            let minimum_bytes = required(Some(minimum))?;
            if minimum_bytes > budget {
                return Err(Error::ModelLoadError(format!(
                    "portable invocation context minimum does not fit: context={minimum}, state_bytes={minimum_bytes}, planning_headroom={headroom_bytes}, safety_reserve={safety_reserve_bytes}"
                )));
            }
            let (mut low, mut high) = (minimum, ceiling);
            while low < high {
                let middle = low + (high - low).div_ceil(2);
                if required(Some(middle))? <= automatic_budget {
                    low = middle;
                } else {
                    high = middle - 1;
                }
            }
            low
        };
        descriptor.resolve_capacity_axis(StateCapacityAxis::DecoderContext, selected)?;
        let published = self
            .model_registry
            .effective_context(variant)
            .map_or(selected, |current| selected.min(current as u64));
        self.model_registry
            .publish_effective_context(variant, published)?;
        Ok(())
    }

    async fn load_invocation_workspace_publication(
        &self,
        model_instance_id: ModelInstanceId,
        executions: &[LoadedExecutionContract],
        descriptor: CapabilityStateDescriptorV2,
        invocation_contract: &InferenceStateContract,
        retained: Option<RetainedStateRuntimeV2>,
        retained_uses: HashMap<[u8; 32], RetainedStateUseV2>,
    ) -> Result<LoadedStatePublication> {
        self.load_invocation_workspace_publication_with_remaining_groups(
            model_instance_id,
            executions,
            descriptor,
            invocation_contract,
            retained,
            retained_uses,
            1,
        )
        .await
    }

    async fn load_invocation_workspace_publication_with_remaining_groups(
        &self,
        model_instance_id: ModelInstanceId,
        executions: &[LoadedExecutionContract],
        mut descriptor: CapabilityStateDescriptorV2,
        invocation_contract: &InferenceStateContract,
        retained: Option<RetainedStateRuntimeV2>,
        retained_uses: HashMap<[u8; 32], RetainedStateUseV2>,
        remaining_automatic_state_groups: u64,
    ) -> Result<LoadedStatePublication> {
        let variant = self
            .resident_variant_for_instance(model_instance_id)
            .ok_or_else(|| {
                Error::ModelLoadError(
                    "invocation state lost its authoritative model generation".into(),
                )
            })?;
        self.fit_invocation_decoder_context(
            variant,
            &mut descriptor,
            invocation_contract,
            executions,
            remaining_automatic_state_groups,
        )?;
        validate_physical_publication_backing(
            &descriptor,
            executions,
            retained.as_ref(),
            &retained_uses,
        )?;
        let allocations =
            plan_invocation_allocations(&descriptor, invocation_contract, executions)?;
        let mut bindings = Vec::with_capacity(allocations.len());
        for allocation in allocations {
            let backing = self
                .core_engine
                .resolve_and_load_invocation_workspace(
                    model_instance_id,
                    allocation.adapter_instance,
                    allocation.key.stage_graph,
                    allocation.key.stage,
                    invocation_contract,
                    &allocation.domain,
                    allocation.slot_count,
                )
                .await?;
            bindings.push(InvocationWorkspaceBindingV2 {
                key: allocation.key,
                backing,
            });
        }
        Ok(LoadedStatePublication::PhysicalV2 {
            descriptor,
            retained,
            retained_uses,
            invocation_workspace: InvocationWorkspaceRuntimeV2::new(bindings)?,
        })
    }

    pub(super) async fn touch_model_usage(&self, variant: ModelVariant) {
        let mut last_used = self.model_last_used.lock().await;
        last_used.insert(variant, now_unix_millis());
    }

    /// Mark an explicitly loaded resident as pinned: it survives budget and
    /// memory-pressure eviction (and the idle-TTL reaper) until an explicit
    /// unload. Job auto-loads never call this.
    pub(super) async fn pin_model(&self, variant: ModelVariant) {
        let mut pinned = self.pinned_variants.lock().await;
        if pinned.insert(variant) {
            info!(model = %variant, "Pinning explicitly loaded model against eviction");
        }
    }

    pub(super) async fn unpin_model(&self, variant: ModelVariant) {
        self.pinned_variants.lock().await.remove(&variant);
    }

    pub(super) async fn pinned_model_variants(&self) -> HashSet<ModelVariant> {
        self.pinned_variants.lock().await.clone()
    }

    pub(super) async fn forget_model_usage(&self, variant: ModelVariant) {
        let mut last_used = self.model_last_used.lock().await;
        last_used.remove(&variant);
    }

    async fn known_resident_variants(&self) -> Vec<ModelVariant> {
        let mut variants = self.authoritative_resident_variants();
        variants.extend(self.model_manager.resident_variants().await);
        variants.sort_by_key(|variant| variant.to_string());
        variants.dedup();
        variants
    }

    pub(super) async fn ensure_model_budget_before_load(
        &self,
        requested_variant: ModelVariant,
        max_loaded_models: Option<usize>,
    ) -> Result<()> {
        let Some(max_loaded_models) = max_loaded_models else {
            return Ok(());
        };

        loop {
            let resident_variants = self.known_resident_variants().await;
            if residency_budget_has_capacity(
                &resident_variants,
                requested_variant,
                max_loaded_models,
            ) {
                return Ok(());
            }

            let mut active_variants = self.core_engine.active_model_variants().await;
            active_variants.extend(
                resident_variants
                    .iter()
                    .copied()
                    .filter(|variant| self.model_manager.active_residency_leases(*variant) > 0),
            );
            let mut ready_variants = Vec::with_capacity(resident_variants.len());
            for variant in &resident_variants {
                if self.resident_phase(*variant)
                    == Some(crate::runtime::lifecycle::controller::ResidentPhase::Ready)
                    || self.model_manager.is_ready(*variant).await
                {
                    ready_variants.push(*variant);
                }
            }
            let last_used = self.model_last_used.lock().await.clone();
            let pinned_variants = self.pinned_variants.lock().await.clone();
            let Some(victim) = select_lru_eviction_candidate(
                &ready_variants,
                requested_variant,
                &active_variants,
                &pinned_variants,
                &last_used,
            ) else {
                return Err(Error::ModelLoadError(format!(
                    "Cannot load {requested_variant}: the {max_loaded_models}-model residency budget is full and no resident model is idle and available for eviction; unload a model to make room"
                )));
            };

            info!(
                requested_variant = %requested_variant,
                victim = %victim,
                max_loaded_models,
                "Evicting idle model before loading its replacement"
            );
            self.unload_model_locked(victim).await?;
        }
    }

    fn model_resource_plan(
        &self,
        variant: ModelVariant,
        model_path: &Path,
    ) -> Result<ModelResourcePlan> {
        let backend = self.backend_router.context().backend_kind;
        if variant == ModelVariant::Qwen3827BFp8 {
            if backend == BackendKind::Cuda {
                return qwen38_memory::resource_plan(
                    model_path,
                    &self.backend_router.context().device.device,
                    &self.config.performance,
                );
            }
            if crate::models::architectures::qwen38::native::synthetic_geometry_enabled() {
                // Fixture load (benchmark/CI): price the actual checkpoint
                // instead of the pinned 27B constants.
                let estimate = qwen38_synthetic_fixture_estimate(model_path)?;
                return Ok(model_resource_plan(backend, estimate));
            }
            return Ok(qwen38_resource_plan(backend));
        }
        if variant == ModelVariant::Qwen36Moe35BA3BFp8 {
            if crate::models::architectures::qwen35moe::native::synthetic_geometry_enabled() {
                // Fixture load (benchmark/CI): price the actual checkpoint
                // instead of the pinned 35B constants.
                let estimate = qwen35moe_memory::synthetic_fixture_estimate(model_path)?;
                return Ok(model_resource_plan(backend, estimate));
            }
            return qwen35moe_memory::resource_plan(backend);
        }
        if variant == ModelVariant::FishAudioS2Pro {
            let memory = crate::models::architectures::fish_s2::weights::fish_s2_model_memory(
                model_path,
                &self.backend_router.context().device,
            )?;
            return fish_s2_resource_plan(backend, memory);
        }
        let estimate = if backend == BackendKind::Cuda {
            model_memory_estimate(variant)
        } else {
            portable_model_memory_estimate(backend, variant, model_path)?
        };
        Ok(model_resource_plan(backend, estimate))
    }

    async fn ensure_portable_memory_before_load(
        &self,
        requested_variant: ModelVariant,
        required_bytes: u64,
    ) -> Result<()> {
        let backend = self.backend_router.context().backend_kind;
        if backend == BackendKind::Cuda {
            return Ok(());
        }
        let mut pooled_scratch_flushed = false;
        loop {
            let ResourceAmount::Known(planning_headroom) = self
                .coordinator
                .resource_authority()
                .planning_headroom_bytes(backend)?
            else {
                return Ok(());
            };
            // Metal loads allocate the full working set at the load fence, so
            // the pre-load check must also respect probed live availability —
            // static ledger estimates alone under-count GGUF materialization
            // and pooled scratch. An unavailable live probe never blocks a
            // load; planning headroom remains the contract then.
            let live_headroom = self
                .coordinator
                .resource_authority()
                .live_preload_headroom_bytes(backend)?;
            let headroom = live_headroom
                .map(|live| planning_headroom.min(live))
                .unwrap_or(planning_headroom);
            if required_bytes <= headroom {
                return Ok(());
            }
            let resident_variants = self.known_resident_variants().await;
            let mut active_variants = self.core_engine.active_model_variants().await;
            active_variants.extend(
                resident_variants
                    .iter()
                    .copied()
                    .filter(|variant| self.model_manager.active_residency_leases(*variant) > 0),
            );
            let last_used = self.model_last_used.lock().await.clone();
            let pinned_variants = self.pinned_variants.lock().await.clone();
            let Some(victim) = select_lru_eviction_candidate(
                &resident_variants,
                requested_variant,
                &active_variants,
                &pinned_variants,
                &last_used,
            ) else {
                if backend == BackendKind::Metal && !pooled_scratch_flushed {
                    // The pool retains released activation scratch until the
                    // last model unloads, which can hide gigabytes of
                    // reclaimable headroom from the live sample. Flushing it
                    // is safe (only unchecked-out buffers are dropped) and
                    // beats failing a load that would fit. One flush per
                    // admission decision; persistent shortage still errors.
                    pooled_scratch_flushed = true;
                    info!(
                        requested_variant = %requested_variant,
                        planning_headroom,
                        live_headroom,
                        "Reclaiming pooled Metal scratch before rejecting the load"
                    );
                    MetalPoolManager::global().clear_all();
                    self.coordinator
                        .resource_authority()
                        .refresh_physical_capacity_after_release();
                    continue;
                }
                return Err(Error::ModelLoadError(format!(
                    "Cannot fit {requested_variant} model tensors before state allocation: load_peak_bytes={required_bytes}, planning_headroom={planning_headroom}, live_headroom={live_headroom:?}, backend={backend:?}; no idle resident model is available for eviction; unload a model to make room"
                )));
            };
            info!(
                requested_variant = %requested_variant,
                victim = %victim,
                required_bytes,
                planning_headroom = headroom,
                "Evicting idle model before portable tensor allocation"
            );
            self.unload_model_locked(victim).await?;
        }
    }

    async fn reserve_model_resources(
        &self,
        requested_variant: ModelVariant,
        load_authorization: ResourceVector,
    ) -> Result<ResourceLease> {
        loop {
            match self.coordinator.resource_authority().reserve(
                ReservationOwner::new(ReservationClass::Model, requested_variant.to_string()),
                load_authorization,
            ) {
                Ok(lease) => return Ok(lease),
                Err(resource_error @ Error::Overloaded(_)) => {
                    let resident_variants = self.known_resident_variants().await;
                    let mut active_variants = self.core_engine.active_model_variants().await;
                    active_variants.extend(resident_variants.iter().copied().filter(|variant| {
                        self.model_manager.active_residency_leases(*variant) > 0
                    }));
                    let mut ready_variants = Vec::new();
                    for variant in &resident_variants {
                        if self.resident_phase(*variant)
                            == Some(crate::runtime::lifecycle::controller::ResidentPhase::Ready)
                            || self.model_manager.is_ready(*variant).await
                        {
                            ready_variants.push(*variant);
                        }
                    }
                    let last_used = self.model_last_used.lock().await.clone();
                    let pinned_variants = self.pinned_variants.lock().await.clone();
                    let Some(victim) = select_lru_eviction_candidate(
                        &ready_variants,
                        requested_variant,
                        &active_variants,
                        &pinned_variants,
                        &last_used,
                    ) else {
                        return Err(Error::ModelLoadError(format!(
                            "Cannot reserve memory for {requested_variant}: {resource_error}; no idle resident model is available for eviction; unload a model to make room"
                        )));
                    };
                    info!(
                        requested_variant = %requested_variant,
                        victim = %victim,
                        "Evicting idle model to satisfy the physical memory budget"
                    );
                    self.unload_model_locked(victim).await?;
                }
                Err(err) => return Err(err),
            }
        }
    }

    /// Whether a load failure must trigger the Metal command-buffer OOM
    /// ladder (pooled-scratch flush and one retry before poisoning). Real
    /// Metal backends classify by backend kind; CPU test harnesses opt in
    /// through the injection flag so the ladder stays regression-tested
    /// without a Metal device.
    fn load_failure_is_metal_command_buffer_oom(&self, error: &Error) -> bool {
        if !is_metal_command_buffer_oom(error) {
            return false;
        }
        if self.backend_router.context().backend_kind == BackendKind::Metal {
            return true;
        }
        #[cfg(test)]
        {
            if self.load_test_metal_oom_ladder.load(std::sync::atomic::Ordering::Acquire) {
                return true;
            }
        }
        false
    }

    async fn run_load_transaction_locked(
        &self,
        variant: ModelVariant,
        max_loaded_models: Option<usize>,
        generation: u64,
    ) -> Result<()> {
        let outcome = self
            .run_load_attempt_locked(variant, max_loaded_models, generation)
            .await;
        let Err(error) = outcome else {
            return Ok(());
        };
        if !self.load_failure_is_metal_command_buffer_oom(&error) {
            return Err(error);
        }
        if self.coordinator.resource_authority().poison_reason().is_some() {
            // The failed attempt could not roll back cleanly; retrying over
            // dirty residency state would compound the damage. Fail closed.
            return Err(error);
        }
        MetalPoolManager::global().clear_all();

        // A retry only makes sense if the flush actually freed enough pooled
        // scratch to fit the model's load peak. Re-probe live headroom after
        // the flush; when it is still clearly below the model's static load
        // peak, a second full load is guaranteed to hit the same command-buffer
        // OOM and would burn another GGUF read + F16 materialization + fence
        // (tens of seconds) before poisoning. Skip straight to the poison.
        let backend = self.backend_router.context().backend_kind;
        let required_bytes = model_memory_estimate(variant).load_peak_bytes;
        self.coordinator
            .resource_authority()
            .refresh_physical_capacity_after_release();
        let retry_futile = match self
            .coordinator
            .resource_authority()
            .live_preload_headroom_bytes(backend)
        {
            Ok(Some(live_headroom)) => live_headroom < required_bytes,
            // No trustworthy live sample: preserve the original single-retry
            // behavior rather than treating the unknown as unlimited.
            Ok(None) | Err(_) => false,
        };
        if retry_futile {
            self.coordinator.resource_authority().poison(format!(
                "Metal command-buffer OOM while loading {variant}: live device headroom after a pooled-scratch flush is still below the model's load peak (load_peak_bytes={required_bytes}); not retrying a doomed load: {error}"
            ));
            self.drain_unpinned_residents_for_recovery().await;
            return Err(error);
        }

        info!(
            model = %variant,
            %error,
            "Metal command-buffer OOM while loading; flushed pooled Metal scratch and retrying the load once"
        );
        let retry = self
            .run_load_attempt_locked(variant, max_loaded_models, generation)
            .await;
        if let Err(retry_error) = &retry {
            if self.load_failure_is_metal_command_buffer_oom(retry_error) {
                self.coordinator.resource_authority().poison(format!(
                    "Metal command-buffer OOM while loading {variant} persisted after a pooled-scratch flush and one retry: {retry_error}"
                ));
                // The authority now rejects every new reservation. Evicting
                // unpinned idle residents lets the device drain so the
                // poison clears without operator action; pinned residents
                // keep their protection and hold the poison until unloaded.
                self.drain_unpinned_residents_for_recovery().await;
            }
        }
        retry
    }

    async fn run_load_attempt_locked(
        &self,
        variant: ModelVariant,
        max_loaded_models: Option<usize>,
        generation: u64,
    ) -> Result<()> {
        if self.resident_phase(variant)
            == Some(crate::runtime::lifecycle::controller::ResidentPhase::Ready)
        {
            return Ok(());
        }
        #[cfg(test)]
        self.maybe_panic_during_load();
        #[cfg(test)]
        self.maybe_fail_load_with_metal_oom()?;

        let load_started = Instant::now();
        let resolved = self.resolve_model_load(variant).await?;
        let acquired = self.acquire_model_artifacts(resolved).await?;
        let artifacts_ms = load_started.elapsed().as_secs_f64() * 1000.0;
        let admission_started = Instant::now();

        self.ensure_model_budget_before_load(variant, max_loaded_models)
            .await?;
        let resource_plan = self.model_resource_plan(variant, &acquired.model_path)?;
        let portable_load_bytes = match self.backend_router.context().backend_kind {
            BackendKind::Cpu => resource_plan.load_authorization.host_bytes,
            BackendKind::Metal => resource_plan.load_authorization.unified_bytes,
            BackendKind::Cuda => ResourceAmount::Known(0),
        };
        if let ResourceAmount::Known(required_bytes) = portable_load_bytes {
            self.ensure_portable_memory_before_load(variant, required_bytes)
                .await?;
        }
        let resource_lease = self
            .reserve_model_resources(variant, resource_plan.load_authorization)
            .await?;
        let model_instance_id = self.install_loading_slot(variant, resource_lease)?;
        if model_instance_id != crate::engine::ModelInstanceId::new(generation) {
            let error = Error::ModelLoadError(format!(
                "model {variant} loading slot does not match generation {generation}"
            ));
            if let Err(rollback_error) = self.rollback_model_locked(variant).await {
                self.mark_slot_cleanup_required(variant);
                tracing::error!(
                    model = %variant,
                    error = %rollback_error,
                    "Mismatched model generation rollback failed"
                );
            }
            return Err(error);
        }

        let publication = async {
            let admission_ms = admission_started.elapsed().as_secs_f64() * 1000.0;
            // Adapter factories are one-shot. Freeze their exact identities
            // and selectable stage graphs before model-derived state planning
            // or any physical state allocation can occur.
            let bundle_draft =
                self.draft_loaded_model_bundle(variant, model_instance_id)?;
            // This is the first operation allowed to allocate model tensors;
            // the peak host/device authorization and authoritative Loading slot
            // are both installed above.
            let weights_started = Instant::now();
            let instantiated = self.instantiate_model(acquired).await?;
            let weights_ms = weights_started.elapsed().as_secs_f64() * 1000.0;
            let preparation_started = Instant::now();
            if variant.family() == crate::catalog::ModelFamily::Qwen3Asr {
                let loaded = self
                    .model_registry
                    .get_loading_asr(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "instantiated Qwen3 ASR model {variant} is missing before adapter sealing"
                        ))
                    })?;
                let NativeAsrModel::Qwen3(model) = loaded.as_ref() else {
                    return Err(Error::ModelLoadError(format!(
                        "instantiated model {variant} does not expose Qwen3 ASR geometry"
                    )));
                };
                // The lifecycle slot is still Loading, so no request can bind
                // this model while its exact backend/dtype/width geometry is
                // being frozen into the already-selected adapter identity.
                bundle_draft.seal_qwen3_asr_audio_preparation(model)?;
            } else if variant.family() == crate::catalog::ModelFamily::WhisperAsr {
                let loaded = self
                    .model_registry
                    .get_loading_asr(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "instantiated Whisper ASR model {variant} is missing before adapter sealing"
                        ))
                    })?;
                let NativeAsrModel::WhisperTurbo(model) = loaded.as_ref() else {
                    return Err(Error::ModelLoadError(format!(
                        "instantiated model {variant} does not expose Whisper ASR geometry"
                    )));
                };
                bundle_draft.seal_whisper_audio_preparation(model)?;
            } else if variant.family() == crate::catalog::ModelFamily::VibeVoiceAsr {
                let loaded = self
                    .model_registry
                    .get_loading_asr(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "instantiated VibeVoice ASR model {variant} is missing before adapter sealing"
                        ))
                    })?;
                let NativeAsrModel::VibeVoice(model) = loaded.as_ref() else {
                    return Err(Error::ModelLoadError(format!(
                        "instantiated model {variant} does not expose VibeVoice ASR geometry"
                    )));
                };
                bundle_draft.seal_vibevoice_asr_preparation(model)?;
            } else if variant.family() == crate::catalog::ModelFamily::GraniteSpeechAsr {
                let loaded = self
                    .model_registry
                    .get_loading_asr(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "instantiated Granite Speech ASR model {variant} is missing before adapter sealing"
                        ))
                    })?;
                let NativeAsrModel::GraniteSpeech(model) = loaded.as_ref() else {
                    return Err(Error::ModelLoadError(format!(
                        "instantiated model {variant} does not expose Granite Speech ASR geometry"
                    )));
                };
                bundle_draft.seal_granite_speech_asr_preparation(model)?;
            } else if variant.family() == crate::catalog::ModelFamily::Lfm25Audio {
                let model = self
                    .model_registry
                    .get_loading_lfm25_audio_lease(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "instantiated LFM2.5 Audio model {variant} is missing before ASR adapter sealing"
                        ))
                    })?;
                bundle_draft.seal_lfm25_audio_asr_preparation(&model)?;
                bundle_draft.seal_lfm25_audio_tts_preparation(&model)?;
            } else if variant.family() == crate::catalog::ModelFamily::Voxtral
                && self
                    .adapter_registry
                    .require(CapabilityKind::RealtimeAsr, variant)
                    .is_ok()
            {
                let model = self
                    .model_registry
                    .get_loading_voxtral(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "instantiated Voxtral realtime model {variant} is missing before adapter sealing"
                        ))
                    })?;
                bundle_draft.seal_voxtral_realtime_preparation(model.as_ref())?;
            }
            // Metal records many tensor uploads asynchronously. Flush them
            // before publishing the model so an allocation failure is owned by
            // this load transaction rather than a later state/request fence.
            let fence_started = Instant::now();
            self.core_engine.synchronize_worker_device().await?;
            let upload_fence_ms = fence_started.elapsed().as_secs_f64() * 1000.0;
            self.publish_loaded_model(instantiated).await?;
            // Model tensors are now visible in the backend provider's used
            // memory. Reconcile the lease before reserving any retained or
            // invocation state so live headroom does not charge the model once
            // through the provider and again as unmaterialized ledger work.
            self.finalize_slot_materialization_with_pending(
                variant,
                resource_plan.resident_authorization,
                resource_plan.deferred_resident_authorization,
            )?;
            let backend = self.backend_router.context().backend_kind;
            // Resolve physical state from the exact loaded chat implementation
            // without wrapping or replacing the already-selected adapter.
            let state_started = Instant::now();
            let mut state_publications = HashMap::new();
            if let Some(loaded) = self.model_registry.get_chat(variant).await {
                // Freeze exact MTP/collation geometry before deriving any
                // selectable stage graphs or planning physical state.
                bundle_draft.seal_chat_workspace(
                    loaded.continuous_decode_batch_workspace_per_row_bytes()?,
                )?;
                let decode_workspace_reserve_bytes = bundle_draft
                    .execution_contracts(CapabilityKind::Chat)?
                    .iter()
                    .flat_map(|contract| contract.stages.iter())
                    .map(|stage| stage.max_workspace_bytes)
                    .max()
                    .unwrap_or(0);
                let loaded_cache = loaded.inference_state_contract()?;
                loaded_cache.validate()?;
                let publication = match &loaded_cache {
                    crate::kv::InferenceStateCapability::Managed(contract) => {
                        if !managed_kv_backend_compiled(backend) {
                            return Err(Error::ModelLoadError(format!(
                                "loaded model {variant} publishes managed KV, but the {backend:?} build has no direct paged-attention runtime"
                            )));
                        }
                        let capacity_policy = managed_chat_capacity_policy(variant, backend);
                        let physical = self
                            .core_engine
                            .load_managed_model_cache_with_capacity_policy(
                                model_instance_id,
                                &loaded_cache,
                                Some(loaded.max_context_tokens()?),
                                capacity_policy.staged_transaction_rows,
                                capacity_policy.fit_cuda_resident_context,
                                decode_workspace_reserve_bytes,
                            )
                            .await?;
                        let physical = physical.ok_or_else(|| {
                            Error::ModelLoadError(
                                "managed state allocation returned no physical runtime".to_string(),
                            )
                        })?;
                        self.model_registry.publish_effective_context(
                            variant,
                            physical.maximum_sequence_tokens(),
                        )?;
                        crate::runtime::rollout::validate_managed_state_plan_eligibility(
                            variant,
                            CapabilityKind::Chat,
                            physical.state_plan_v2(),
                        )?;
                        Some(LoadedStatePublication::ManagedV2 {
                            contract: contract.clone(),
                            physical,
                        })
                    }
                    crate::kv::InferenceStateCapability::Stateless => None,
                };
                if let Some(publication) = publication {
                    state_publications.insert(CapabilityKind::Chat, publication);
                }
            }
            if variant.family() == crate::catalog::ModelFamily::GraniteSpeechAsr {
                if !managed_kv_backend_compiled(backend) {
                    return Err(Error::ModelLoadError(format!(
                        "loaded model {variant} requires physical Granite Speech invocation state, but the {backend:?} build has no direct paged-attention runtime"
                    )));
                }
                let model = self
                    .model_registry
                    .get_loading_asr(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "loaded Granite Speech model {variant} is missing from the registry"
                        ))
                    })?;
                for capability in [
                    CapabilityKind::Asr,
                    CapabilityKind::SpeakerAttributedAsr,
                ] {
                    if self.adapter_registry.require(capability, variant).is_err() {
                        continue;
                    }
                    let contracts = bundle_draft.execution_contracts(capability)?;
                    let stage_graphs = contracts
                        .iter()
                        .map(|contract| contract.stages.as_ref())
                        .collect::<Vec<_>>();
                    let physical_spec =
                        model.granite_speech_physical_state_spec(&stage_graphs)?;
                    let (retained, retained_uses) = if capability == CapabilityKind::Asr {
                        let retained_contract = physical_spec.retained.as_ref().ok_or_else(|| {
                            Error::ModelLoadError(
                                "Granite Speech ASR graph did not author retained decoder state"
                                    .into(),
                            )
                        })?;
                        let retained_max_tokens = physical_spec
                            .retained_max_tokens
                            .ok_or_else(|| {
                                Error::ModelLoadError(
                                    "Granite Speech retained decoder has no context bound".into(),
                                )
                            })?;
                        let retained = self
                            .core_engine
                            .load_managed_model_state_with_portable_copies(
                                model_instance_id,
                                retained_contract,
                                Some(retained_max_tokens),
                                2,
                            )
                            .await?;
                        self.model_registry.publish_effective_context(
                            variant,
                            retained.logical_token_reach(),
                        )?;
                        crate::runtime::rollout::validate_managed_state_plan_eligibility(
                            variant,
                            CapabilityKind::Asr,
                            retained.state_plan_v2(),
                        )?;
                        let uses = contracts
                            .iter()
                            .map(|contract| {
                                let graph = stage_graph_fingerprint(&contract.stages)?;
                                let use_kind = match contract.execution_profile.mode {
                                    crate::engine::ExecutionMode::Sequence => {
                                        RetainedStateUseV2::ExternalPaged
                                    }
                                    crate::engine::ExecutionMode::Atomic => {
                                        RetainedStateUseV2::Inactive
                                    }
                                    _ => {
                                        return Err(Error::ModelLoadError(
                                            "Granite Speech ASR graph has an incompatible retained-state profile"
                                                .into(),
                                        ));
                                    }
                                };
                                Ok((graph, use_kind))
                            })
                            .collect::<Result<HashMap<_, _>>>()?;
                        (Some(retained.into()), uses)
                    } else {
                        if physical_spec.retained.is_some()
                            || physical_spec.retained_max_tokens.is_some()
                        {
                            return Err(Error::ModelLoadError(
                                "speaker-attributed Granite Speech graph unexpectedly requested retained state"
                                    .into(),
                            ));
                        }
                        (None, HashMap::new())
                    };
                    let publication = self
                        .load_invocation_workspace_publication_with_remaining_groups(
                            model_instance_id,
                            &contracts,
                            physical_spec.descriptor,
                            &physical_spec.invocation,
                            retained,
                            retained_uses,
                            if capability == CapabilityKind::Asr { 2 } else { 1 },
                        )
                        .await?;
                    state_publications.insert(capability, publication);
                }
            }
            if self
                .adapter_registry
                .require(CapabilityKind::Asr, variant)
                .is_ok()
                && variant.family() != crate::catalog::ModelFamily::GraniteSpeechAsr
            {
                if variant.family() == crate::catalog::ModelFamily::Voxtral {
                    if !managed_kv_backend_compiled(backend) {
                        return Err(Error::ModelLoadError(format!(
                            "loaded model {variant} requires physical ASR invocation state, but the {backend:?} build has no direct paged-attention runtime"
                        )));
                    }
                    let loaded = self
                        .model_registry
                        .get_loading_voxtral(variant)
                        .await
                        .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "loaded Voxtral model {variant} is missing from the registry"
                        ))
                        })?;
                    let contracts = bundle_draft.execution_contracts(CapabilityKind::Asr)?;
                    let stage_graphs = contracts
                        .iter()
                        .map(|contract| contract.stages.as_ref())
                        .collect::<Vec<_>>();
                    let physical_spec = loaded.physical_state_spec(&stage_graphs)?;
                    let publication = self
                        .load_invocation_workspace_publication(
                            model_instance_id,
                            &contracts,
                            physical_spec.descriptor,
                            &physical_spec.invocation,
                            None,
                            HashMap::new(),
                        )
                        .await?;
                    state_publications.insert(CapabilityKind::Asr, publication);
                } else if let Some(loaded) =
                    self.model_registry.get_loading_asr(variant).await
                {
                    let loaded_cache = loaded.inference_state_contract()?;
                    loaded_cache.validate()?;
                    let publication_route = loaded_asr_state_publication_route(variant);
                    if publication_route == LoadedAsrStatePublicationRoute::Qwen3 {
                        if !managed_kv_backend_compiled(backend) {
                            return Err(Error::ModelLoadError(format!(
                                "loaded model {variant} requires physical ASR state, but the {backend:?} build has no direct paged-attention runtime"
                            )));
                        }
                        let contracts = bundle_draft.execution_contracts(CapabilityKind::Asr)?;
                        let stage_graphs = contracts
                            .iter()
                            .map(|contract| contract.stages.as_ref())
                            .collect::<Vec<_>>();
                        let physical_spec = loaded.qwen3_physical_state_spec(&stage_graphs)?;
                        let retained_max_tokens = physical_spec.retained_max_tokens;
                        let physical = self
                            .core_engine
                            .load_managed_model_state_with_portable_copies(
                                model_instance_id,
                                &physical_spec.retained,
                                Some(retained_max_tokens),
                                2,
                            )
                            .await?;
                        self.model_registry.publish_effective_context(
                            variant,
                            physical.logical_token_reach(),
                        )?;
                        crate::runtime::rollout::validate_managed_state_plan_eligibility(
                            variant,
                            CapabilityKind::Asr,
                            physical.state_plan_v2(),
                        )?;
                        let retained_uses = contracts
                            .iter()
                            .map(|contract| {
                                let graph = stage_graph_fingerprint(&contract.stages)?;
                                let retained_use = match contract.execution_profile.cache_mode {
                                    CacheMode::ExternalPaged
                                        if contract.execution_profile.cache_namespace.is_some()
                                            && contract.execution_profile.kv_dtype != "none" =>
                                    {
                                        RetainedStateUseV2::ExternalPaged
                                    }
                                    CacheMode::None
                                        if contract.execution_profile.cache_namespace.is_none()
                                            && contract.execution_profile.kv_dtype == "none" =>
                                    {
                                        RetainedStateUseV2::Inactive
                                    }
                                    _ => {
                                        return Err(Error::ModelLoadError(
                                            "Qwen3 ASR graph has an incompatible retained-state profile"
                                                .to_string(),
                                        ));
                                    }
                                };
                                Ok((graph, retained_use))
                            })
                            .collect::<Result<HashMap<_, _>>>()?;
                        let publication = self
                            .load_invocation_workspace_publication(
                                model_instance_id,
                                &contracts,
                                physical_spec.descriptor,
                                &physical_spec.invocation,
                                Some(physical.into()),
                                retained_uses,
                            )
                            .await?;
                        state_publications.insert(
                            CapabilityKind::Asr,
                            publication,
                        );
                    } else if publication_route == LoadedAsrStatePublicationRoute::VibeVoice {
                        if !managed_kv_backend_compiled(backend) {
                            return Err(Error::ModelLoadError(format!(
                                "loaded model {variant} requires physical ASR invocation state, but the {backend:?} build has no direct paged-attention runtime"
                            )));
                        }
                        let contracts = bundle_draft.execution_contracts(CapabilityKind::Asr)?;
                        let stage_graphs = contracts
                            .iter()
                            .map(|contract| contract.stages.as_ref())
                            .collect::<Vec<_>>();
                        let physical_spec = loaded.vibevoice_physical_state_spec(&stage_graphs)?;
                        let retained_contract = physical_spec.retained.as_ref().ok_or_else(|| {
                            Error::ModelLoadError(
                                "VibeVoice ASR normal graph did not author retained decoder state"
                                    .into(),
                            )
                        })?;
                        let retained_max_tokens = physical_spec.retained_max_tokens.ok_or_else(|| {
                            Error::ModelLoadError(
                                "VibeVoice ASR retained decoder state has no context bound".into(),
                            )
                        })?;
                        let retained = self
                            .core_engine
                            .load_managed_model_state_with_portable_copies(
                                model_instance_id,
                                retained_contract,
                                Some(retained_max_tokens),
                                2,
                            )
                            .await?;
                        self.model_registry.publish_effective_context(
                            variant,
                            retained.logical_token_reach(),
                        )?;
                        crate::runtime::rollout::validate_managed_state_plan_eligibility(
                            variant,
                            CapabilityKind::Asr,
                            retained.state_plan_v2(),
                        )?;
                        let retained_uses = contracts
                            .iter()
                            .map(|contract| {
                                let graph = stage_graph_fingerprint(&contract.stages)?;
                                let retained_use = match contract.execution_profile.mode {
                                    crate::engine::ExecutionMode::Sequence => {
                                        RetainedStateUseV2::ExternalPaged
                                    }
                                    crate::engine::ExecutionMode::Atomic => {
                                        RetainedStateUseV2::Inactive
                                    }
                                    _ => {
                                        return Err(Error::ModelLoadError(
                                            "VibeVoice ASR graph has an incompatible retained-state profile"
                                                .into(),
                                        ));
                                    }
                                };
                                Ok((graph, retained_use))
                            })
                            .collect::<Result<HashMap<_, _>>>()?;
                        let publication = self
                            .load_invocation_workspace_publication(
                                model_instance_id,
                                &contracts,
                                physical_spec.descriptor,
                                &physical_spec.invocation,
                                Some(retained.into()),
                                retained_uses,
                            )
                            .await?;
                        state_publications.insert(CapabilityKind::Asr, publication);
                    } else if publication_route == LoadedAsrStatePublicationRoute::Whisper {
                        if !managed_kv_backend_compiled(backend) {
                            return Err(Error::ModelLoadError(format!(
                                "loaded model {variant} requires physical Whisper ASR invocation state, but the {backend:?} build has no direct paged-attention runtime"
                            )));
                        }
                        let contracts = bundle_draft.execution_contracts(CapabilityKind::Asr)?;
                        let stage_graphs = contracts
                            .iter()
                            .map(|contract| contract.stages.as_ref())
                            .collect::<Vec<_>>();
                        let physical_spec = loaded.whisper_physical_state_spec(&stage_graphs)?;
                        let retained = self
                            .core_engine
                            .load_composite_retained_state(
                                model_instance_id,
                                &physical_spec.retained,
                                physical_spec.retained_static_domain,
                                Some(physical_spec.retained_max_tokens),
                            )
                            .await?;
                        let retained_uses = contracts
                            .iter()
                            .map(|contract| {
                                let graph = stage_graph_fingerprint(&contract.stages)?;
                                let retained_use = if contract.execution_profile.mode
                                    == crate::engine::ExecutionMode::Sequence
                                {
                                    RetainedStateUseV2::ExternalPagedStatic
                                } else if contract.execution_profile.mode
                                    == crate::engine::ExecutionMode::Atomic
                                {
                                    RetainedStateUseV2::Inactive
                                } else {
                                    return Err(Error::ModelLoadError(
                                        "Whisper graph has an incompatible retained-state profile"
                                            .into(),
                                    ));
                                };
                                Ok((graph, retained_use))
                            })
                            .collect::<Result<HashMap<_, _>>>()?;
                        let publication = self
                            .load_invocation_workspace_publication(
                                model_instance_id,
                                &contracts,
                                physical_spec.descriptor,
                                &physical_spec.invocation,
                                Some(retained.into()),
                                retained_uses,
                            )
                            .await?;
                        state_publications.insert(CapabilityKind::Asr, publication);
                    } else if publication_route == LoadedAsrStatePublicationRoute::Parakeet {
                        let contracts = bundle_draft.execution_contracts(CapabilityKind::Asr)?;
                        let stage_graphs = contracts
                            .iter()
                            .map(|contract| contract.stages.as_ref())
                            .collect::<Vec<_>>();
                        let physical_spec = loaded.parakeet_physical_state_spec(&stage_graphs)?;
                        let retained_contract = physical_spec.retained.as_ref().ok_or_else(|| {
                            Error::ModelLoadError(
                                "Parakeet sequence graph did not author retained recurrent state"
                                    .into(),
                            )
                        })?;
                        let retained = self
                            .core_engine
                            .load_retained_tensor_state(
                                model_instance_id,
                                retained_contract,
                                self.realtime_asr_sequence_capacity,
                            )
                            .await?;
                        let effective_context = self.config.portable_context_ceiling();
                        self.model_registry.publish_effective_context(
                            variant,
                            u64::try_from(effective_context).map_err(|_| {
                                Error::ModelLoadError(
                                    "Parakeet effective context exceeds u64".into(),
                                )
                            })?,
                        )?;
                        let retained_uses = contracts
                            .iter()
                            .map(|contract| {
                                let graph = stage_graph_fingerprint(&contract.stages)?;
                                if contract.execution_profile.mode
                                    != crate::engine::ExecutionMode::Sequence
                                {
                                    return Err(Error::ModelLoadError(
                                        "Parakeet retained graph has an incompatible execution mode"
                                            .into(),
                                    ));
                                }
                                Ok((graph, RetainedStateUseV2::Inactive))
                            })
                            .collect::<Result<HashMap<_, _>>>()?;
                        if physical_spec.invocation.is_some() {
                            return Err(Error::ModelLoadError(
                                "Parakeet sequence graph unexpectedly published invocation predictor state"
                                    .into(),
                            ));
                        }
                        let publication = self.load_scratch_only_workspace_publication(
                            &contracts,
                            physical_spec.descriptor,
                            Some(retained.into()),
                            retained_uses,
                        )?;
                        state_publications.insert(CapabilityKind::Asr, publication);
                    } else if publication_route
                        == LoadedAsrStatePublicationRoute::NemotronOffline
                    {
                        let contracts = bundle_draft.execution_contracts(CapabilityKind::Asr)?;
                        let stage_graphs = contracts
                            .iter()
                            .map(|contract| contract.stages.as_ref())
                            .collect::<Vec<_>>();
                        let physical_spec =
                            loaded.nemotron_offline_physical_state_spec(&stage_graphs)?;
                        let publication = self
                            .load_invocation_workspace_publication(
                                model_instance_id,
                                &contracts,
                                physical_spec.descriptor,
                                &physical_spec.invocation,
                                None,
                                HashMap::new(),
                            )
                            .await?;
                        state_publications.insert(CapabilityKind::Asr, publication);
                    } else if let crate::kv::InferenceStateCapability::Managed(contract) =
                        &loaded_cache
                    {
                        return Err(Error::ModelLoadError(format!(
                            "loaded non-Qwen ASR model {variant} still publishes legacy managed state with {} domains",
                            contract.domains.len()
                        )));
                    }
                }
            }
            if self
                .adapter_registry
                .require(CapabilityKind::RealtimeAsr, variant)
                .is_ok()
            {
                let contracts =
                    bundle_draft.execution_contracts(CapabilityKind::RealtimeAsr)?;
                let stage_graphs = contracts
                    .iter()
                    .map(|contract| contract.stages.as_ref())
                    .collect::<Vec<_>>();
                if variant.family() == crate::catalog::ModelFamily::Voxtral {
                    if !managed_kv_backend_compiled(backend) {
                        return Err(Error::ModelLoadError(format!(
                            "loaded model {variant} requires physical realtime ASR state, but the {backend:?} build has no direct paged-attention runtime"
                        )));
                    }
                    let model = self
                        .model_registry
                        .get_loading_voxtral(variant)
                        .await
                        .ok_or_else(|| {
                            Error::ModelLoadError(format!(
                                "loaded Voxtral realtime ASR model {variant} is missing from the registry"
                            ))
                        })?;
                    let physical_spec = model.realtime_physical_state_spec(&stage_graphs)?;
                    let retained = self
                        .core_engine
                        .load_managed_model_state(
                            model_instance_id,
                            &physical_spec.retained,
                            Some(physical_spec.retained_max_tokens),
                        )
                        .await?;
                    let retained_uses = contracts
                        .iter()
                        .map(|contract| {
                            Ok((
                                stage_graph_fingerprint(&contract.stages)?,
                                RetainedStateUseV2::ExternalPaged,
                            ))
                        })
                        .collect::<Result<HashMap<_, _>>>()?;
                    state_publications.insert(
                        CapabilityKind::RealtimeAsr,
                        LoadedStatePublication::PhysicalV2 {
                            descriptor: physical_spec.descriptor,
                            retained: Some(retained.into()),
                            retained_uses,
                            invocation_workspace: InvocationWorkspaceRuntimeV2::default(),
                        },
                    );
                } else {
                    let model = self
                        .model_registry
                        .get_loading_asr(variant)
                        .await
                        .ok_or_else(|| {
                            Error::ModelLoadError(format!(
                                "loaded realtime ASR model {variant} is missing from the registry"
                            ))
                        })?;
                    let physical_spec = model.realtime_physical_state_spec(&stage_graphs)?;
                    let retained = self
                        .core_engine
                        .load_retained_tensor_state(
                            model_instance_id,
                            &physical_spec.retained,
                            self.realtime_asr_sequence_capacity,
                        )
                        .await?;
                    let retained_uses = contracts
                        .iter()
                        .map(|contract| {
                            Ok((
                                stage_graph_fingerprint(&contract.stages)?,
                                RetainedStateUseV2::ExternalTensor,
                            ))
                        })
                        .collect::<Result<HashMap<_, _>>>()?;
                    state_publications.insert(
                        CapabilityKind::RealtimeAsr,
                        LoadedStatePublication::PhysicalV2 {
                            descriptor: physical_spec.descriptor,
                            retained: Some(retained.into()),
                            retained_uses,
                            invocation_workspace: InvocationWorkspaceRuntimeV2::default(),
                        },
                    );
                }
            }
            if variant.family() == crate::catalog::ModelFamily::SortformerDiarization {
                let model = self
                    .model_registry
                    .get_diarization(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "loaded Sortformer model {variant} is missing from the registry"
                        ))
                    })?;
                let contracts =
                    bundle_draft.execution_contracts(CapabilityKind::Diarization)?;
                let stage_graphs = contracts
                    .iter()
                    .map(|contract| contract.stages.as_ref())
                    .collect::<Vec<_>>();
                let physical_spec = model.physical_state_spec(&stage_graphs)?;
                let publication = self
                    .load_invocation_workspace_publication(
                        model_instance_id,
                        &contracts,
                        physical_spec.descriptor,
                        &physical_spec.invocation,
                        None,
                        HashMap::new(),
                    )
                    .await?;
                state_publications.insert(CapabilityKind::Diarization, publication);
            }
            if variant.family() == crate::catalog::ModelFamily::Lfm25Audio {
                if !managed_kv_backend_compiled(backend) {
                    return Err(Error::ModelLoadError(format!(
                        "loaded model {variant} requires physical LFM2.5 Audio invocation state, but the {backend:?} build has no direct paged-attention runtime"
                    )));
                }
                let model = self
                    .model_registry
                    .get_loading_lfm25_audio_lease(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "loaded LFM2.5 Audio model {variant} is missing from the registry"
                        ))
                    })?;
                for capability in [
                    CapabilityKind::Asr,
                    CapabilityKind::Tts,
                    CapabilityKind::AudioChat,
                    CapabilityKind::SpeechToSpeech,
                ] {
                    let remaining_automatic_state_groups = match capability {
                        CapabilityKind::Asr => 5,
                        CapabilityKind::Tts => 3,
                        CapabilityKind::AudioChat => 2,
                        CapabilityKind::SpeechToSpeech => 1,
                        _ => unreachable!("LFM2.5 Audio capability list is closed above"),
                    };
                    let contracts = bundle_draft.execution_contracts(capability)?;
                    let stage_graphs = contracts
                        .iter()
                        .map(|contract| contract.stages.as_ref())
                        .collect::<Vec<_>>();
                    if capability == CapabilityKind::Asr {
                        let physical_spec = model.retained_asr_state_spec(&stage_graphs)?;
                        let retained_max_tokens = physical_spec.retained_max_tokens;
                        let retained = self
                            .core_engine
                            .load_managed_model_state_with_portable_copies(
                                model_instance_id,
                                &physical_spec.retained,
                                Some(retained_max_tokens),
                                2,
                            )
                            .await?;
                        self.model_registry.publish_effective_context(
                            variant,
                            retained.logical_token_reach(),
                        )?;
                        crate::runtime::rollout::validate_managed_state_plan_eligibility(
                            variant,
                            CapabilityKind::Asr,
                            retained.state_plan_v2(),
                        )?;
                        let retained_uses = contracts
                            .iter()
                            .map(|contract| {
                                let graph = stage_graph_fingerprint(&contract.stages)?;
                                let retained_use = match contract.execution_profile.mode {
                                    crate::engine::ExecutionMode::Sequence => {
                                        RetainedStateUseV2::ExternalPaged
                                    }
                                    crate::engine::ExecutionMode::Atomic => {
                                        RetainedStateUseV2::Inactive
                                    }
                                    _ => {
                                        return Err(Error::ModelLoadError(
                                            "LFM2.5 Audio ASR graph has an incompatible retained-state profile"
                                                .into(),
                                        ));
                                    }
                                };
                                Ok((graph, retained_use))
                            })
                            .collect::<Result<HashMap<_, _>>>()?;
                        let main_invocation = physical_spec.main_invocation.as_ref().ok_or_else(|| {
                            Error::ModelLoadError(
                                "LFM2.5 Audio ASR retained topology is missing long-form main invocation state"
                                    .into(),
                            )
                        })?;
                        let publication = self
                            .load_invocation_workspace_publication_with_remaining_groups(
                                model_instance_id,
                                &contracts,
                                physical_spec.descriptor,
                                main_invocation,
                                Some(retained.into()),
                                retained_uses,
                                remaining_automatic_state_groups,
                            )
                            .await?;
                        state_publications.insert(capability, publication);
                        continue;
                    }
                    if capability == CapabilityKind::Tts {
                        let physical_spec = model.retained_tts_state_spec(&stage_graphs)?;
                        let retained_max_tokens = self
                            .model_registry
                            .effective_context(variant)
                            .map_or(physical_spec.retained_max_tokens, |tokens| {
                                physical_spec.retained_max_tokens.min(tokens)
                            });
                        let retained = self
                            .core_engine
                            .load_managed_model_state(
                                model_instance_id,
                                &physical_spec.retained,
                                Some(retained_max_tokens),
                            )
                            .await?;
                        self.model_registry.publish_effective_context(
                            variant,
                            retained.logical_token_reach(),
                        )?;
                        crate::runtime::rollout::validate_managed_state_plan_eligibility(
                            variant,
                            CapabilityKind::Tts,
                            retained.state_plan_v2(),
                        )?;
                        let retained_uses = contracts
                            .iter()
                            .map(|contract| {
                                Ok((
                                    stage_graph_fingerprint(&contract.stages)?,
                                    RetainedStateUseV2::ExternalPaged,
                                ))
                            })
                            .collect::<Result<HashMap<_, _>>>()?;
                        let depthformer = physical_spec.depthformer_invocation.as_ref().ok_or_else(
                            || {
                                Error::ModelLoadError(
                                    "LFM2.5 Audio TTS topology is missing Depthformer invocation state"
                                        .into(),
                                )
                            },
                        )?;
                        let publication = self
                            .load_invocation_workspace_publication_with_remaining_groups(
                                model_instance_id,
                                &contracts,
                                physical_spec.descriptor,
                                depthformer,
                                Some(retained.into()),
                                retained_uses,
                                remaining_automatic_state_groups,
                            )
                            .await?;
                        state_publications.insert(capability, publication);
                        continue;
                    }
                    let mode = if capability == CapabilityKind::Asr {
                        crate::models::architectures::lfm25_audio::physical::Lfm25AudioStateMode::MainOnly
                    } else {
                        crate::models::architectures::lfm25_audio::physical::Lfm25AudioStateMode::MainAndDepthformer
                    };
                    let physical_spec = model.physical_state_spec(mode, &stage_graphs)?;
                    let publication = self
                        .load_invocation_workspace_publication_with_remaining_groups(
                            model_instance_id,
                            &contracts,
                            physical_spec.descriptor,
                            &physical_spec.invocation,
                            None,
                            HashMap::new(),
                            remaining_automatic_state_groups,
                        )
                        .await?;
                    state_publications.insert(capability, publication);
                }
            }
            if variant.family() == crate::catalog::ModelFamily::Qwen3Tts {
                let model = self
                    .model_registry
                    .get_qwen_tts(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "loaded Qwen3 TTS model {variant} is missing from the registry"
                        ))
                    })?;
                for capability in [CapabilityKind::Tts, CapabilityKind::StreamingTts] {
                    if self.adapter_registry.require(capability, variant).is_err() {
                        continue;
                    }
                    let contracts = bundle_draft.execution_contracts(capability)?;
                    let stage_graphs = contracts
                        .iter()
                        .map(|contract| contract.stages.as_ref())
                        .collect::<Vec<_>>();
                    let physical_spec = model.physical_state_spec(&stage_graphs)?;
                    if !managed_kv_backend_compiled(backend) {
                        return Err(Error::ModelLoadError(format!(
                            "loaded model {variant} requires physical TTS state, but the {backend:?} build has no direct paged-attention runtime"
                        )));
                    }
                    let retained_max_tokens = self
                        .model_registry
                        .effective_context(variant)
                        .map_or(physical_spec.retained_max_tokens, |tokens| {
                            physical_spec.retained_max_tokens.min(tokens)
                        });
                    let retained = self
                        .core_engine
                        .load_managed_model_state(
                            model_instance_id,
                            &physical_spec.retained,
                            Some(retained_max_tokens),
                        )
                        .await?;
                    self.model_registry.publish_effective_context(
                        variant,
                        retained.logical_token_reach(),
                    )?;
                    crate::runtime::rollout::validate_managed_state_plan_eligibility(
                        variant,
                        capability,
                        retained.state_plan_v2(),
                    )?;
                    let retained_uses = contracts
                        .iter()
                        .map(|contract| {
                            let graph = stage_graph_fingerprint(&contract.stages)?;
                            let retained_use = match contract.execution_profile.cache_mode {
                                CacheMode::ExternalPaged
                                    if contract.execution_profile.cache_namespace.is_some()
                                        && contract.execution_profile.kv_dtype != "none" =>
                                {
                                    RetainedStateUseV2::ExternalPaged
                                }
                                CacheMode::None
                                    if contract.execution_profile.cache_namespace.is_none()
                                        && contract.execution_profile.kv_dtype == "none" =>
                                {
                                    RetainedStateUseV2::Inactive
                                }
                                _ => {
                                    return Err(Error::ModelLoadError(
                                        "Qwen3 TTS retained-state graph has an incompatible cache profile"
                                            .to_string(),
                                    ));
                                }
                            };
                            Ok((graph, retained_use))
                        })
                        .collect::<Result<HashMap<_, _>>>()?;
                    let publication = self
                        .load_invocation_workspace_publication_with_remaining_groups(
                            model_instance_id,
                            &contracts,
                            physical_spec.descriptor,
                            &physical_spec.predictor_contract,
                            Some(retained.into()),
                            retained_uses,
                            if capability == CapabilityKind::Tts
                                && self
                                    .adapter_registry
                                    .require(CapabilityKind::StreamingTts, variant)
                                    .is_ok()
                            {
                                3
                            } else {
                                1
                            },
                        )
                        .await?;
                    state_publications.insert(capability, publication);
                }
            }
            if variant.family() == crate::catalog::ModelFamily::VibeVoiceTts {
                if !managed_kv_backend_compiled(backend) {
                    return Err(Error::ModelLoadError(format!(
                        "loaded model {variant} requires physical TTS invocation state, but the {backend:?} build has no direct paged-attention runtime"
                    )));
                }
                let model = self
                    .model_registry
                    .get_loading_vibevoice_tts(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "loaded VibeVoice TTS model {variant} is missing from the registry"
                        ))
                    })?;
                let contracts = bundle_draft.execution_contracts(CapabilityKind::Tts)?;
                let stage_graphs = contracts
                    .iter()
                    .map(|contract| contract.stages.as_ref())
                    .collect::<Vec<_>>();
                let physical_spec = model.physical_state_spec(&stage_graphs)?;
                let retained_contract = physical_spec.retained.as_ref().ok_or_else(|| {
                    Error::ModelLoadError(
                        "VibeVoice TTS normal graph did not publish retained state".into(),
                    )
                })?;
                let retained_max_tokens = physical_spec.retained_max_tokens;
                let retained = self
                    .core_engine
                    .load_managed_model_state(
                        model_instance_id,
                        retained_contract,
                        retained_max_tokens,
                    )
                    .await?;
                self.model_registry.publish_effective_context(
                    variant,
                    retained.logical_token_reach(),
                )?;
                crate::runtime::rollout::validate_managed_state_plan_eligibility(
                    variant,
                    CapabilityKind::Tts,
                    retained.state_plan_v2(),
                )?;
                let retained_uses = contracts
                    .iter()
                    .map(|contract| {
                        let graph = stage_graph_fingerprint(&contract.stages)?;
                        let retained_use = match contract.execution_profile.cache_mode {
                            CacheMode::ExternalPaged => RetainedStateUseV2::ExternalPaged,
                            CacheMode::None => RetainedStateUseV2::Inactive,
                        };
                        Ok((graph, retained_use))
                    })
                    .collect::<Result<HashMap<_, _>>>()?;
                let publication = self
                    .load_invocation_workspace_publication(
                        model_instance_id,
                        &contracts,
                        physical_spec.descriptor,
                        &physical_spec.invocation,
                        Some(retained.into()),
                        retained_uses,
                    )
                    .await?;
                state_publications.insert(CapabilityKind::Tts, publication);
            }
            if variant.family() == crate::catalog::ModelFamily::FishS2Tts {
                if !managed_kv_backend_compiled(backend) {
                    return Err(Error::ModelLoadError(format!(
                        "loaded model {variant} requires physical TTS invocation state, but the {backend:?} build has no direct paged-attention runtime"
                    )));
                }
                let model = self
                    .model_registry
                    .get_loading_fish_s2_tts(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "loaded Fish S2 TTS model {variant} is missing from the registry"
                        ))
                    })?;
                let headroom = match self.coordinator.resource_authority().planning_headroom_bytes(backend)? {
                    ResourceAmount::Known(bytes) => Some(bytes),
                    _ => None,
                };
                let native_batching = match std::env::var("IZWI_FISH_NATIVE_BATCHING").as_deref() {
                    Ok("1" | "true") => true,
                    Ok("0" | "false") | Err(_) => false,
                    Ok(value) => return Err(Error::ConfigError(format!("invalid IZWI_FISH_NATIVE_BATCHING={value}; expected true or false"))),
                };
                let capacity = crate::runtime::fish_capacity::ResolvedFishServingCapacity::resolve(
                    model.config(), &self.config, headroom, native_batching,
                )?;
                tracing::info!(?capacity, ?backend, "Sealing Fish serving capacity; device throughput qualification is separate");
                bundle_draft.seal_fish_capacity(capacity.clone())?;
                state_publications.extend(self.load_fish_s2_state_publications(
                    model_instance_id, &bundle_draft,
                    self.config.max_sequence_length.explicit_tokens().unwrap_or(model.config().max_seq_len).min(model.config().max_seq_len),
                    &capacity,
                    |graphs| model.physical_state_spec(graphs),
                ).await?);
            }
            if variant.family() == crate::catalog::ModelFamily::VoxtralTts {
                if !managed_kv_backend_compiled(backend) {
                    return Err(Error::ModelLoadError(format!(
                        "loaded model {variant} requires physical TTS invocation state, but the {backend:?} build has no direct paged-attention runtime"
                    )));
                }
                let model = self
                    .model_registry
                    .get_loading_voxtral_tts(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "loaded Voxtral TTS model {variant} is missing from the registry"
                        ))
                    })?;
                let contracts = bundle_draft.execution_contracts(CapabilityKind::Tts)?;
                let stage_graphs = contracts
                    .iter()
                    .map(|contract| contract.stages.as_ref())
                    .collect::<Vec<_>>();
                let physical_spec = model.physical_state_spec(&stage_graphs)?;
                let publication = self
                    .load_invocation_workspace_publication(
                        model_instance_id,
                        &contracts,
                        physical_spec.descriptor,
                        &physical_spec.invocation,
                        None,
                        HashMap::new(),
                    )
                    .await?;
                state_publications.insert(CapabilityKind::Tts, publication);
            }
            if variant.family() == crate::catalog::ModelFamily::KokoroTts {
                let model = self
                    .model_registry
                    .get_loading_kokoro(variant)
                    .await
                    .ok_or_else(|| {
                        Error::ModelLoadError(format!(
                            "loaded Kokoro TTS model {variant} is missing from the registry"
                        ))
                    })?;
                let effective_context = kokoro_effective_context_tokens(
                    model.config().context_length(),
                    self.config.portable_context_ceiling(),
                )?;
                self.model_registry
                    .publish_effective_context(variant, effective_context)?;
            }
            let state_allocation_ms = state_started.elapsed().as_secs_f64() * 1000.0;
            let binding_started = Instant::now();
            self.bind_loaded_model_bundle_draft(
                bundle_draft,
                variant,
                model_instance_id,
                state_publications,
            )?;
            // Install the legacy manager projection before the authoritative
            // commit. Inference pins consult the slot, so no caller can observe
            // Ready while this await is still in progress.
            self.model_manager.mark_loaded(variant).await;
            self.mark_slot_ready_for_instance(variant, model_instance_id)?;
            if uses_asr_model_registry(variant) {
                // This is the registry's external publication barrier. Every
                // adapter seal, backend fence, state plan, and bundle binding
                // has committed and the authoritative slot is already Ready.
                // The safe transient is Ready-but-hidden; an ASR handle must
                // never be visible while its slot is still Loading.
                self.model_registry.publish_asr_ready(variant).await?;
            }
            if variant.family() == crate::catalog::ModelFamily::Voxtral {
                self.model_registry.publish_voxtral_ready(variant).await?;
            }
            if variant.family() == crate::catalog::ModelFamily::Lfm25Audio {
                self.model_registry
                    .publish_lfm25_audio_ready(variant)
                    .await?;
            }
            if variant.family() == crate::catalog::ModelFamily::VibeVoiceTts {
                self.model_registry
                    .publish_vibevoice_tts_ready(variant)
                    .await?;
            }
            if variant.family() == crate::catalog::ModelFamily::FishS2Tts {
                self.model_registry
                    .publish_fish_s2_tts_ready(variant)
                    .await?;
            }
            if variant.family() == crate::catalog::ModelFamily::KokoroTts {
                self.model_registry.publish_kokoro_ready(variant).await?;
            }
            if variant.family() == crate::catalog::ModelFamily::VoxtralTts {
                self.model_registry
                    .publish_voxtral_tts_ready(variant)
                    .await?;
            }
            self.touch_model_usage(variant).await;
            info!(
                model = %variant,
                generation,
                artifacts_ms,
                admission_ms,
                weights_ms,
                upload_fence_ms,
                state_allocation_ms,
                binding_publication_ms = binding_started.elapsed().as_secs_f64() * 1000.0,
                preparation_ms = preparation_started.elapsed().as_secs_f64() * 1000.0,
                ready_ms = load_started.elapsed().as_secs_f64() * 1000.0,
                "Model load reached physical Ready"
            );
            Ok(())
        }
        .await;

        if let Err(error) = publication {
            let metal_oom = self.load_failure_is_metal_command_buffer_oom(&error);
            if let Err(rollback_error) = self.rollback_model_locked(variant).await {
                self.mark_slot_cleanup_required(variant);
                tracing::error!(
                    model = %variant,
                    error = %rollback_error,
                    "Model load rollback failed"
                );
                if metal_oom {
                    // Dirty residency state after a device-fatal OOM: new
                    // physical work must fail closed until recreation, so a
                    // later attempt cannot be blamed for earlier queued work.
                    self.coordinator.resource_authority().poison(format!(
                        "Metal command-buffer OOM while loading {variant} and rollback failed: {error}"
                    ));
                }
            }
            return Err(error);
        }

        Ok(())
    }

    pub(crate) fn spawn_load_transaction(
        self: &Arc<Self>,
        variant: ModelVariant,
        max_loaded_models: Option<usize>,
        leader: crate::runtime::lifecycle::controller::LoadLeader,
    ) -> tokio::task::JoinHandle<()> {
        let controller = self.clone();
        tokio::spawn(async move {
            // Publication of both the Ready slot and the shared terminal
            // outcome is one mutation-gated transaction. Explicit unload can
            // neither erase a successful load before waiters are notified nor
            // observe a half-published failure rollback.
            let _mutation_guard = controller.mutation_gate.lock().await;
            if !controller.is_current_load_generation_locked(variant, leader.generation) {
                return;
            }
            let _coordinator_load = match controller
                .coordinator
                .begin_model_load(format!("model-load:{variant}"))
            {
                Ok(load) => load,
                Err(error) => {
                    controller.finish_load_locked(
                        variant,
                        leader.generation,
                        &leader.completion,
                        SharedLoadOutcome::Failed(SharedLoadFailure::from_error(error)),
                    );
                    return;
                }
            };
            let outcome = match AssertUnwindSafe(controller.run_load_transaction_locked(
                variant,
                max_loaded_models,
                leader.generation,
            ))
            .catch_unwind()
            .await
            {
                Ok(Ok(())) => SharedLoadOutcome::Ready,
                Ok(Err(error)) => SharedLoadOutcome::Failed(SharedLoadFailure::from_error(error)),
                Err(payload) => {
                    let message = if let Some(message) = payload.downcast_ref::<&str>() {
                        (*message).to_string()
                    } else if let Some(message) = payload.downcast_ref::<String>() {
                        message.clone()
                    } else {
                        "unknown model load panic".to_string()
                    };
                    if let Err(error) = controller.rollback_model_after_panic_locked(variant).await
                    {
                        tracing::error!(model = %variant, %error, "Panicked model load rollback failed");
                    }
                    SharedLoadOutcome::Failed(SharedLoadFailure::ModelLoad(format!(
                        "model load task panicked: {message}"
                    )))
                }
            };
            controller.finish_load_locked(variant, leader.generation, &leader.completion, outcome);
        })
    }
}

impl RuntimeService {
    pub(crate) async fn load_model_for_inference(
        &self,
        variant: ModelVariant,
    ) -> Result<crate::model::ModelResidencyLease> {
        loop {
            if let Some(lease) = self.model_lifecycle.try_acquire_ready_lease(variant) {
                self.model_lifecycle.touch_model_usage(variant).await;
                return Ok(lease);
            }

            let (waiter, leader) = self.model_lifecycle.join_or_start_load(variant);
            if let Some(leader) = leader {
                let _load_task = self.model_lifecycle.spawn_load_transaction(
                    variant,
                    self.max_loaded_models,
                    leader,
                );
            }
            waiter.wait().await?;
        }
    }

    /// Load a model without retaining an inference pin. The explicit load
    /// entry (admin API, preload list) marks the resident as pinned so it
    /// survives budget and memory-pressure eviction until explicitly
    /// unloaded; job auto-loads use `load_model_for_inference` directly and
    /// stay transient.
    pub async fn load_model(&self, variant: ModelVariant) -> Result<()> {
        drop(self.load_model_for_inference(variant).await?);
        self.model_lifecycle.pin_model(variant).await;
        Ok(())
    }

    /// Variants explicitly loaded (pinned) in this server session.
    pub async fn pinned_model_variants(&self) -> std::collections::HashSet<ModelVariant> {
        self.model_lifecycle.pinned_model_variants().await
    }

    async fn ensure_model_budget_before_load(&self, requested_variant: ModelVariant) -> Result<()> {
        let _mutation_guard = self.model_lifecycle.mutation_gate.lock().await;
        self.model_lifecycle
            .ensure_model_budget_before_load(requested_variant, self.max_loaded_models)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::{
        automatic_state_group_budget, estimate_from_tensor_inventory, fish_s2_resource_plan,
        is_metal_command_buffer_oom, kokoro_effective_context_tokens, now_unix_millis,
        loaded_asr_state_publication_route, managed_chat_capacity_policy, model_memory_estimate,
        model_resource_plan, plan_invocation_allocations, portable_context_reserve_bytes,
        portable_invocation_context_intent,
        qwen38_representation_memory_estimate, qwen38_resource_plan, residency_budget_has_capacity,
        select_lru_eviction_candidate, validate_scratch_only_invocation_publication,
        LoadedAsrStatePublicationRoute, ModelMemoryEstimate, PortableInvocationContextIntent,
        QWEN38_BF16_ELEMENTS, QWEN38_CUDA_DEVICE_CONVERSION_SCRATCH_BYTES,
        QWEN38_CUDA_HOST_CONVERSION_SCRATCH_BYTES, QWEN38_FP8_ELEMENTS,
        QWEN38_PORTABLE_CONVERSION_SCRATCH_BYTES, QWEN38_Q8_0_BLOCK_BYTES,
        QWEN38_Q8_0_BLOCK_ELEMENTS,
    };
    use crate::backends::kv::managed_kv_backend_compiled;
    use crate::backends::{BackendKind, BackendPreference};
    use crate::config::{ContextLengthPreference, EngineConfig};
    use crate::engine::{
        AdapterAbiRevision, AdapterInstanceId, CapacitySource, ConcurrencyClass, ExecutionGroupId,
        ExecutionMode, ExecutionProfile, ModelInstanceId, NativeBatchMode,
        PhysicalCapacityProvider, PhysicalCapacitySnapshot, ReservationClass, ReservationOwner,
        ResourceAmount, ResourceAuthority, ResourceLease, ResourceVector, StageDescriptor, StageId,
        StageWorkSelector,
    };
    use crate::error::Error;
    use crate::kv::v2::{
        stage_graph_fingerprint, test_contract, BoundedShape, CapabilityStateDescriptorV2,
        CheckpointPolicy, InvocationLeaseScope, InvocationStageWorkspace, InvocationStateCapacity,
        InvocationWorkspaceDomain, InvocationWorkspaceProfile, InvocationWorkspaceSet,
        PrefixPolicy, RetainedStateCapability, ShapeAxis, ShapeDimension, ShapeExtent,
        StateComponentId, StateDType, StateDomainId, StateDomainSpec, StateScope,
        StaticTensorDomainSpec, TensorComponentSpec, TensorRole, WorkspaceFormula,
        CURRENT_INFERENCE_STATE_ABI,
    };
    use crate::model::ModelVariant;
    use crate::models::architectures::fish_s2::weights::FishS2ModelMemory;
    use crate::models::shared::weights::gguf::TensorStorageInventory;
    use crate::runtime::adapters::{
        CapabilityKind, LoadedExecutionContract, RuntimeAdapterRegistry,
    };
    use crate::runtime::lifecycle::controller::{
        ResidentPhase, SharedLoadFailure, SharedLoadOutcome,
    };
    use crate::runtime::service::RuntimeService;
    use std::collections::{HashMap, HashSet};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex as StdMutex};
    use std::time::{Duration, Instant};
    use tokio::sync::{oneshot, Barrier};
    use uuid::Uuid;

    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn fish_s2_load_publishes_and_seals_both_tts_capabilities() {
        use crate::models::architectures::fish_s2::{fish_s2_physical_state_spec, FishS2TtsModel};
        use crate::runtime::adapters::LoadedModelBundleDraft;
        use crate::runtime::adapters::StreamingRequirements;
        // Serialize against OOM-ladder poison windows and load-transaction
        // reservations on the process-global resource authority.
        let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let directory =
            std::env::temp_dir().join(format!("izwi-fish-state-publication-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&directory).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: directory.clone(),
            backend: BackendPreference::Cpu,
            max_sequence_length: ContextLengthPreference::explicit(32).unwrap(),
            max_retained_sequences: 2,
            max_staged_transactions: 2,
            ..Default::default()
        })
        .unwrap();
        let variant = ModelVariant::FishAudioS2Pro;
        let authority = vector_authority(all_memory_capacity(1024));
        let lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "fish-load-publication"),
                ResourceVector::zero(),
            )
            .unwrap();
        let instance = runtime
            .model_lifecycle
            .install_loading_slot(variant, lease)
            .unwrap();
        let registry = RuntimeAdapterRegistry::built_in();
        let draft = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(991),
            instance,
            variant,
            BackendKind::Cpu,
        )
        .unwrap();
        // Use the model-authored physical contract with small neural geometry;
        // no weights or inference mocks are needed to exercise publication.
        let mut config = FishS2TtsModel::for_test().config().clone();
        config.text_config.num_hidden_layers = 1;
        config.text_config.hidden_size = 128;
        config.text_config.num_attention_heads = 1;
        config.text_config.num_key_value_heads = 1;
        config.text_config.head_dim = Some(128);
        config.audio_decoder_config.num_hidden_layers = 1;
        config.audio_decoder_config.hidden_size = 128;
        config.audio_decoder_config.num_attention_heads = 1;
        config.audio_decoder_config.num_key_value_heads = 1;
        config.audio_decoder_config.head_dim = Some(128);
        let mut capacity =
            crate::runtime::fish_capacity::ResolvedFishServingCapacity::scalar().unwrap();
        capacity.ar_rows = 2;
        capacity.codec_rows = 2;
        capacity.active_rows = 2;
        capacity.staged_rows = 2;
        capacity.ar_workspace_per_row = 8192;
        capacity.prefill_workspace_per_token = 8192;
        capacity.codec_workspace_per_row = 8192;
        capacity.native_batching = true;
        draft.seal_fish_capacity(capacity.clone()).unwrap();
        let publications = runtime
            .model_lifecycle
            .load_fish_s2_state_publications(instance, &draft, 32, &capacity, |graphs| {
                fish_s2_physical_state_spec(&config, candle_core::DType::F32, graphs)
            })
            .await
            .unwrap();
        assert_eq!(publications.len(), 2);
        let bundle = draft
            .seal(publications)
            .expect("all advertised capabilities must seal");
        let mut bindings = Vec::new();
        for capability in [CapabilityKind::Tts, CapabilityKind::StreamingTts] {
            let binding = bundle
                .capability_binding_for_streaming(capability, StreamingRequirements::NONE)
                .unwrap();
            assert_eq!(binding.execution.model_instance_id, instance);
            assert_eq!(
                binding
                    .execution
                    .stages
                    .iter()
                    .find(|stage| stage.selector == StageWorkSelector::SequenceDecode)
                    .unwrap()
                    .max_batch_size,
                2
            );
            assert!(binding.state.managed_kv_runtime().is_some());
            assert!(binding
                .execution
                .stages
                .iter()
                .any(|stage| stage.selector == StageWorkSelector::SequenceAudioDecode));
            bindings.push(binding);
        }
        assert_ne!(
            bindings[0].execution.adapter_instance_id,
            bindings[1].execution.adapter_instance_id
        );
        assert!(Arc::ptr_eq(
            &bindings[0].state.managed_kv_runtime().unwrap(),
            &bindings[1].state.managed_kv_runtime().unwrap()
        ));
        drop(bindings);
        drop(bundle);
        drop(runtime);
        std::fs::remove_dir_all(directory).unwrap();
    }

    fn invocation_execution(max_batch_size: usize) -> LoadedExecutionContract {
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let metadata = *RuntimeAdapterRegistry::built_in()
            .require(CapabilityKind::Tts, variant)
            .unwrap();
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Atomic);
        profile.max_batch_size = max_batch_size;
        profile.concurrency = if max_batch_size > 1 {
            ConcurrencyClass::Batchable
        } else {
            ConcurrencyClass::Exclusive
        };
        let mut stage = StageDescriptor::from_execution_profile(
            StageId::new(1),
            "test.atomic.scalar",
            &profile,
            NativeBatchMode::None,
        );
        stage.selector = StageWorkSelector::Atomic;
        stage.max_workspace_bytes = 0;
        stage.validate().unwrap();
        LoadedExecutionContract {
            execution_group_id: ExecutionGroupId::new(1),
            model_instance_id: ModelInstanceId::new(2),
            adapter_instance_id: AdapterInstanceId::new(3),
            adapter_abi_revision: AdapterAbiRevision::new(4),
            metadata,
            execution_profile: profile,
            stages: Arc::from([stage]),
        }
    }

    #[test]
    fn portable_tensor_inventory_replaces_catalog_guess_and_accounts_load_overlap() {
        let catalog = ModelMemoryEstimate {
            load_peak_bytes: 12 * 1024 * 1024 * 1024,
            resident_bytes: 12 * 1024 * 1024 * 1024,
        };
        // F32-shaped largest tensor: 160 MB storage over 40M elements.
        let estimate = estimate_from_tensor_inventory(
            catalog,
            Some(TensorStorageInventory {
                total_bytes: 2_400_000_000,
                largest_tensor_bytes: 160_000_000,
                largest_tensor_elements: 40_000_000,
                tensor_count: 400,
            }),
        )
        .unwrap();
        assert_eq!(estimate.resident_bytes, 2_400_000_000);
        // Scratch = max(next_pow2(160M) = 268,435,456; 160M + 40M*4 = 320M;
        // 400 * 32 KiB = 13,107,200) = 320,000,000: the F32 conversion bound
        // of the largest tensor exceeds the historic size-class proxy.
        assert_eq!(estimate.load_peak_bytes, 2_720_000_000);
        assert_eq!(
            estimate_from_tensor_inventory(catalog, None).unwrap(),
            catalog
        );
        assert_eq!(
            estimate_from_tensor_inventory(catalog, Some(TensorStorageInventory::default()),)
                .unwrap(),
            catalog
        );
    }

    #[test]
    fn moe_shaped_inventory_reserves_whole_checkpoint_instantiation_scratch() {
        let catalog = ModelMemoryEstimate {
            load_peak_bytes: 40 * 1024 * 1024 * 1024,
            resident_bytes: 40 * 1024 * 1024 * 1024,
        };
        // MoE shape: tens of thousands of same-sized expert tensors collapse
        // the largest-tensor terms; only the allocation-count term keeps the
        // instantiation scratch reserved.
        let inventory = TensorStorageInventory {
            total_bytes: 30_000_000_000,
            largest_tensor_bytes: 20_971_520,
            largest_tensor_elements: 5_242_880,
            tensor_count: 18_432,
        };
        let estimate = estimate_from_tensor_inventory(catalog, Some(inventory)).unwrap();
        assert_eq!(estimate.resident_bytes, 30_000_000_000);
        assert_eq!(estimate.load_peak_bytes, 30_603_979_776);
        // The historic formula would have reserved only the size-class proxy
        // of one expert tensor (30,033,554,432) — under-reserving by the
        // whole-checkpoint instantiation term the fix exists for.
        let historic_load_peak = inventory.total_bytes
            + inventory
                .largest_tensor_bytes
                .checked_next_power_of_two()
                .unwrap();
        assert_eq!(historic_load_peak, 30_033_554_432);
        assert!(estimate.load_peak_bytes > historic_load_peak);
    }

    #[test]
    fn quantized_largest_tensor_reserves_f32_conversion_scratch() {
        let catalog = ModelMemoryEstimate {
            load_peak_bytes: 1024 * 1024 * 1024,
            resident_bytes: 1024 * 1024 * 1024,
        };
        // Q4_0-shaped largest tensor: 1M elements over 562.5 KB of storage.
        // A dequantizing loader holds the source while building the F32
        // destination (4 MB), which the size-class proxy under-covers.
        let estimate = estimate_from_tensor_inventory(
            catalog,
            Some(TensorStorageInventory {
                total_bytes: 5_000_000,
                largest_tensor_bytes: 562_500,
                largest_tensor_elements: 1_000_000,
                tensor_count: 10,
            }),
        )
        .unwrap();
        assert_eq!(estimate.load_peak_bytes, 9_562_500);
    }

    #[test]
    fn qwen38_resource_plan_prices_backend_resident_representations() {
        let cpu = qwen38_representation_memory_estimate(BackendKind::Cpu);
        let metal = qwen38_representation_memory_estimate(BackendKind::Metal);
        let cuda = qwen38_representation_memory_estimate(BackendKind::Cuda);
        let elements = QWEN38_FP8_ELEMENTS + QWEN38_BF16_ELEMENTS;
        let q8_0_bytes =
            QWEN38_FP8_ELEMENTS.div_ceil(QWEN38_Q8_0_BLOCK_ELEMENTS) * QWEN38_Q8_0_BLOCK_BYTES;

        assert_eq!(cpu.resident_bytes, elements * 4);
        assert_eq!(metal.resident_bytes, elements * 2);
        assert_eq!(cuda.resident_bytes, q8_0_bytes + QWEN38_BF16_ELEMENTS * 2);
        assert_eq!(cuda.resident_bytes, 32_407_348_704);
        assert_eq!(
            cuda.load_peak_bytes,
            cuda.resident_bytes + QWEN38_CUDA_DEVICE_CONVERSION_SCRATCH_BYTES
        );
        assert_eq!(cuda.load_peak_bytes, 32_675_784_160);
        assert_eq!(
            cpu.load_peak_bytes - cpu.resident_bytes,
            QWEN38_PORTABLE_CONVERSION_SCRATCH_BYTES
        );
        assert_eq!(
            metal.load_peak_bytes - metal.resident_bytes,
            QWEN38_PORTABLE_CONVERSION_SCRATCH_BYTES
        );
        let cuda_plan = qwen38_resource_plan(BackendKind::Cuda);
        assert_eq!(
            cuda_plan.load_authorization.host_bytes,
            ResourceAmount::Known(QWEN38_CUDA_HOST_CONVERSION_SCRATCH_BYTES)
        );
        assert_eq!(
            cuda_plan.load_authorization.device_bytes,
            ResourceAmount::Known(cuda.load_peak_bytes)
        );
        assert_eq!(
            cuda_plan.resident_authorization.device_bytes,
            ResourceAmount::Known(cuda.resident_bytes)
        );
    }

    #[test]
    fn qwen38_uses_engine_staged_width_while_cuda_retains_context_fitting() {
        let cuda_qwen38 =
            managed_chat_capacity_policy(ModelVariant::Qwen3827BFp8, BackendKind::Cuda);
        assert_eq!(cuda_qwen38.staged_transaction_rows, None);
        assert!(cuda_qwen38.fit_cuda_resident_context);

        let metal_qwen38 =
            managed_chat_capacity_policy(ModelVariant::Qwen3827BFp8, BackendKind::Metal);
        assert_eq!(metal_qwen38.staged_transaction_rows, None);
        assert!(!metal_qwen38.fit_cuda_resident_context);

        let cuda_qwen35 =
            managed_chat_capacity_policy(ModelVariant::Qwen359BGguf, BackendKind::Cuda);
        assert_eq!(cuda_qwen35.staged_transaction_rows, None);
        assert!(!cuda_qwen35.fit_cuda_resident_context);

        // The Qwen3.5/3.6 MoE family fits its invocation-state arena against
        // CUDA resident headroom like Qwen3.8: its authored-context arena
        // otherwise cannot coexist with the resident representation on an
        // 80 GB device.
        let cuda_qwen36_moe =
            managed_chat_capacity_policy(ModelVariant::Qwen36Moe35BA3BFp8, BackendKind::Cuda);
        assert_eq!(cuda_qwen36_moe.staged_transaction_rows, None);
        assert!(cuda_qwen36_moe.fit_cuda_resident_context);

        let metal_qwen36_moe =
            managed_chat_capacity_policy(ModelVariant::Qwen36Moe35BA3BFp8, BackendKind::Metal);
        assert!(!metal_qwen36_moe.fit_cuda_resident_context);
    }

    #[test]
    fn qwen38_cuda_q8_0_plan_fits_reported_l40s_available_bytes() {
        const REPORTED_L40S_AVAILABLE_BYTES: u64 = 47_196_667_904;

        let plan = qwen38_resource_plan(BackendKind::Cuda);
        let ResourceAmount::Known(device_load_peak_bytes) = plan.load_authorization.device_bytes
        else {
            panic!("Qwen3.8 CUDA device load peak must be known")
        };
        let old_expanded_load_peak_bytes = (QWEN38_FP8_ELEMENTS + QWEN38_BF16_ELEMENTS) * 2
            + QWEN38_PORTABLE_CONVERSION_SCRATCH_BYTES;
        assert_eq!(
            plan.load_authorization.host_bytes,
            ResourceAmount::Known(8_589_934_592)
        );
        assert_eq!(device_load_peak_bytes, 32_675_784_160);
        assert_eq!(old_expanded_load_peak_bytes, 56_636_597_728);
        assert!(device_load_peak_bytes < REPORTED_L40S_AVAILABLE_BYTES);
        assert!(old_expanded_load_peak_bytes > REPORTED_L40S_AVAILABLE_BYTES);

        let l40s_capacity = ResourceVector {
            host_bytes: ResourceAmount::Known(QWEN38_CUDA_HOST_CONVERSION_SCRATCH_BYTES),
            device_bytes: ResourceAmount::Known(REPORTED_L40S_AVAILABLE_BYTES),
            ..ResourceVector::zero()
        };
        let authority = vector_authority(l40s_capacity);
        let lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "qwen38-cuda-q8_0"),
                plan.load_authorization,
            )
            .expect("Q8_0 CUDA representation should fit reported L40S headroom");
        drop(lease);

        let old_expanded = ResourceVector {
            host_bytes: ResourceAmount::Known(QWEN38_CUDA_HOST_CONVERSION_SCRATCH_BYTES),
            device_bytes: ResourceAmount::Known(old_expanded_load_peak_bytes),
            ..ResourceVector::zero()
        };
        let error = vector_authority(l40s_capacity)
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "qwen38-cuda-expanded"),
                old_expanded,
            )
            .expect_err("the old expanded CUDA peak must not fit reported L40S headroom");
        assert!(matches!(error, Error::Overloaded(_)));

        let insufficient_host = ResourceVector {
            host_bytes: ResourceAmount::Known(QWEN38_CUDA_HOST_CONVERSION_SCRATCH_BYTES - 1),
            device_bytes: ResourceAmount::Known(REPORTED_L40S_AVAILABLE_BYTES),
            ..ResourceVector::zero()
        };
        let error = vector_authority(insufficient_host)
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "qwen38-cuda-host-scratch"),
                plan.load_authorization,
            )
            .expect_err("CUDA load must reserve the complete 8 GiB host conversion window");
        assert!(matches!(error, Error::Overloaded(_)));
    }

    #[test]
    fn metal_command_buffer_oom_is_backend_fatal_but_other_errors_are_not() {
        assert!(is_metal_command_buffer_oom(&Error::InferenceError(
            "Metal error Command buffer had following error: Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)".into(),
        )));
        assert!(!is_metal_command_buffer_oom(&Error::ModelLoadError(
            "portable context minimum does not fit".into(),
        )));
    }

    fn invocation_contract(domain_count: u32) -> crate::kv::v2::InferenceStateContract {
        let mut contract = test_contract();
        let StateDomainSpec::PagedAttention(first) = &mut contract.domains[0] else {
            unreachable!()
        };
        first.header.scope = StateScope::Invocation;
        first.header.prefix = PrefixPolicy::Disabled;
        first.header.checkpoint = CheckpointPolicy::None;
        first.accepted_dtypes = vec![StateDType::F32];
        contract.groups[0].prefix_shareable = false;
        for id in 2..=domain_count {
            let mut state = contract.domains[0].clone();
            let StateDomainSpec::PagedAttention(domain) = &mut state else {
                unreachable!()
            };
            domain.header.id = StateDomainId::new(id);
            contract.domains.push(state);
        }
        contract.groups[0].domains = (1..=domain_count).map(StateDomainId::new).collect();
        contract.validate().unwrap();
        contract
    }

    fn invocation_descriptor(
        execution: &LoadedExecutionContract,
        contract: &crate::kv::v2::InferenceStateContract,
        lease_scope: InvocationLeaseScope,
    ) -> CapabilityStateDescriptorV2 {
        let domains = contract
            .domains
            .iter()
            .map(|state| InvocationWorkspaceDomain::State {
                state: state.clone(),
                capacity: if matches!(state, StateDomainSpec::PagedAttention(_)) {
                    InvocationStateCapacity::PagedTokens { max_tokens: 16 }
                } else {
                    InvocationStateCapacity::SemanticBounded
                },
                placement: state.header().placement,
                formula: WorkspaceFormula {
                    fixed_bytes: 1024 * 1024,
                    dimensions: vec![],
                    terms: vec![],
                },
            })
            .collect();
        CapabilityStateDescriptorV2 {
            abi: CURRENT_INFERENCE_STATE_ABI,
            retained: RetainedStateCapability::Stateless,
            invocation: InvocationWorkspaceSet::Bounded {
                profiles: vec![InvocationWorkspaceProfile {
                    stage_graph_fingerprint: stage_graph_fingerprint(&execution.stages).unwrap(),
                    stages: vec![InvocationStageWorkspace {
                        stage: execution.stages[0].id,
                        lease_scope,
                        groups: contract.groups.clone(),
                        domains,
                    }],
                }],
            },
        }
    }

    fn mixed_invocation_contract() -> crate::kv::v2::InferenceStateContract {
        let mut contract = invocation_contract(1);
        let mut header = contract.domains[0].header().clone();
        header.id = StateDomainId::new(2);
        contract
            .domains
            .push(StateDomainSpec::StaticTensor(StaticTensorDomainSpec {
                header,
                components: vec![TensorComponentSpec {
                    id: StateComponentId::new(1),
                    role: TensorRole::EncoderMemory,
                    shape: BoundedShape {
                        dimensions: vec![
                            ShapeDimension {
                                axis: ShapeAxis::Sequence,
                                extent: ShapeExtent::RuntimeBounded { min: 1, max: 16 },
                            },
                            ShapeDimension {
                                axis: ShapeAxis::Hidden,
                                extent: ShapeExtent::Fixed { value: 8 },
                            },
                        ],
                    },
                    accepted_dtypes: vec![StateDType::F32],
                }],
            }));
        contract.groups[0].domains.push(StateDomainId::new(2));
        contract.validate().unwrap();
        contract
    }

    #[test]
    fn managed_capability_cache_truth_tracks_compiled_direct_kernels() {
        assert!(managed_kv_backend_compiled(BackendKind::Cpu));
        assert_eq!(
            managed_kv_backend_compiled(BackendKind::Metal),
            cfg!(feature = "metal")
        );
        assert_eq!(
            managed_kv_backend_compiled(BackendKind::Cuda),
            cfg!(feature = "cuda")
        );
    }

    #[test]
    fn whisper_asr_selects_physical_invocation_publication() {
        assert_eq!(
            loaded_asr_state_publication_route(ModelVariant::WhisperLargeV3Turbo),
            LoadedAsrStatePublicationRoute::Whisper
        );
        assert_eq!(
            loaded_asr_state_publication_route(ModelVariant::VibeVoiceAsr),
            LoadedAsrStatePublicationRoute::VibeVoice
        );
        assert_eq!(
            loaded_asr_state_publication_route(ModelVariant::ParakeetTdt06BV3),
            LoadedAsrStatePublicationRoute::Parakeet
        );
        assert_eq!(
            loaded_asr_state_publication_route(ModelVariant::Nemotron35AsrStreaming06B),
            LoadedAsrStatePublicationRoute::NemotronOffline
        );
    }

    #[test]
    fn generic_invocation_planner_allocates_every_domain_at_exact_row_concurrency() {
        let execution = invocation_execution(3);
        let contract = invocation_contract(2);
        let descriptor = invocation_descriptor(&execution, &contract, InvocationLeaseScope::PerRow);

        let allocations =
            plan_invocation_allocations(&descriptor, &contract, &[execution]).unwrap();
        assert_eq!(allocations.len(), 2);
        assert!(allocations
            .iter()
            .all(|allocation| allocation.slot_count == 3));
        assert_eq!(
            allocations
                .iter()
                .map(|allocation| allocation.key.domain)
                .collect::<Vec<_>>(),
            vec![StateDomainId::new(1), StateDomainId::new(2)]
        );
    }

    #[test]
    fn generic_invocation_planner_accepts_mixed_paged_and_semantic_domains() {
        let execution = invocation_execution(2);
        let contract = mixed_invocation_contract();
        let descriptor = invocation_descriptor(&execution, &contract, InvocationLeaseScope::PerRow);

        let allocations =
            plan_invocation_allocations(&descriptor, &contract, &[execution]).unwrap();
        assert_eq!(allocations.len(), 2);
        assert!(allocations
            .iter()
            .all(|allocation| allocation.slot_count == 2));
        assert!(matches!(
            allocations[0].domain,
            InvocationWorkspaceDomain::State {
                state: StateDomainSpec::PagedAttention(_),
                capacity: InvocationStateCapacity::PagedTokens { .. },
                ..
            }
        ));
        assert!(matches!(
            allocations[1].domain,
            InvocationWorkspaceDomain::State {
                state: StateDomainSpec::StaticTensor(_),
                capacity: InvocationStateCapacity::SemanticBounded,
                ..
            }
        ));
    }

    #[test]
    fn generic_invocation_planner_rejects_missing_extra_and_foreign_graph_mappings() {
        let execution = invocation_execution(1);
        let contract = invocation_contract(2);
        let descriptor =
            invocation_descriptor(&execution, &contract, InvocationLeaseScope::PerStageBatch);

        let mut missing_contract = contract.clone();
        missing_contract.domains.pop();
        missing_contract.groups[0].domains.pop();
        missing_contract.validate().unwrap();
        assert!(plan_invocation_allocations(
            &descriptor,
            &missing_contract,
            std::slice::from_ref(&execution),
        )
        .is_err());

        let mut missing_descriptor = descriptor.clone();
        let InvocationWorkspaceSet::Bounded { profiles } = &mut missing_descriptor.invocation
        else {
            unreachable!()
        };
        profiles[0].stages[0].domains.pop();
        profiles[0].stages[0].groups[0].domains.pop();
        assert!(plan_invocation_allocations(
            &missing_descriptor,
            &contract,
            std::slice::from_ref(&execution),
        )
        .is_err());

        let foreign_execution = invocation_execution(2);
        assert!(plan_invocation_allocations(
            &descriptor,
            &contract,
            &[execution, foreign_execution]
        )
        .is_err());
    }

    #[test]
    fn scratch_only_publication_accepts_stage_scratch_and_rejects_typed_state() {
        let mut execution = invocation_execution(1);
        let mut stage = execution.stages[0].clone();
        stage.max_workspace_bytes = 4096;
        stage.validate().unwrap();
        execution.stages = Arc::from([stage]);
        let descriptor =
            CapabilityStateDescriptorV2::stateless_for_stage_graphs(&[execution.stages.as_ref()])
                .unwrap();
        validate_scratch_only_invocation_publication(&descriptor, std::slice::from_ref(&execution))
            .unwrap();

        let contract = invocation_contract(1);
        let typed = invocation_descriptor(&execution, &contract, InvocationLeaseScope::PerRow);
        let error =
            validate_scratch_only_invocation_publication(&typed, std::slice::from_ref(&execution))
                .unwrap_err();
        assert!(error.to_string().contains("contains typed state"));
    }

    fn one_byte_host_reservation() -> ResourceVector {
        ResourceVector {
            host_bytes: ResourceAmount::Known(1),
            ..ResourceVector::zero()
        }
    }

    #[derive(Debug)]
    struct TestCapacityProvider;

    impl PhysicalCapacityProvider for TestCapacityProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            let capacity = ResourceVector {
                host_bytes: ResourceAmount::Known(1024),
                ..ResourceVector::zero()
            };
            PhysicalCapacitySnapshot {
                capacity,
                available: capacity,
                source: CapacitySource::Test,
            }
        }
    }

    #[derive(Debug)]
    struct VectorCapacityProvider {
        snapshot: PhysicalCapacitySnapshot,
    }

    impl PhysicalCapacityProvider for VectorCapacityProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            self.snapshot
        }
    }

    fn all_memory_capacity(bytes: u64) -> ResourceVector {
        ResourceVector {
            host_bytes: ResourceAmount::Known(bytes),
            device_bytes: ResourceAmount::Known(bytes),
            unified_bytes: ResourceAmount::Known(bytes),
            ..ResourceVector::zero()
        }
    }

    fn vector_authority(capacity: ResourceVector) -> Arc<ResourceAuthority> {
        Arc::new(ResourceAuthority::new(Arc::new(VectorCapacityProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity,
                available: capacity,
                source: CapacitySource::Test,
            },
        })))
    }

    fn isolated_resource_lease(key: &str) -> (Arc<ResourceAuthority>, ResourceLease) {
        let authority = Arc::new(ResourceAuthority::new(Arc::new(TestCapacityProvider)));
        let lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, key),
                one_byte_host_reservation(),
            )
            .expect("test resource reservation");
        (authority, lease)
    }

    #[test]
    fn select_lru_eviction_candidate_skips_requested_and_active_models() {
        let resident_variants = vec![
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            ModelVariant::Qwen359BGguf,
            ModelVariant::Kokoro82M,
        ];
        let requested_variant = ModelVariant::Kokoro82M;
        let active_variants = HashSet::from([ModelVariant::Qwen359BGguf]);
        let pinned_variants = HashSet::new();
        let last_used = HashMap::from([
            (ModelVariant::Qwen3Tts12Hz06BCustomVoice, 10_u64),
            (ModelVariant::Qwen359BGguf, 5_u64),
            (ModelVariant::Kokoro82M, 20_u64),
        ]);

        let candidate = select_lru_eviction_candidate(
            &resident_variants,
            requested_variant,
            &active_variants,
            &pinned_variants,
            &last_used,
        );

        assert_eq!(candidate, Some(ModelVariant::Qwen3Tts12Hz06BCustomVoice));
    }

    #[test]
    fn select_lru_eviction_candidate_skips_pinned_models() {
        let resident_variants = vec![
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            ModelVariant::Qwen359BGguf,
        ];
        let requested_variant = ModelVariant::Kokoro82M;
        let active_variants = HashSet::new();
        let pinned_variants = HashSet::from([ModelVariant::Qwen359BGguf]);
        // The pinned model is older than the unpinned one but must not be
        // selected.
        let last_used = HashMap::from([
            (ModelVariant::Qwen3Tts12Hz06BCustomVoice, 10_u64),
            (ModelVariant::Qwen359BGguf, 5_u64),
        ]);

        let candidate = select_lru_eviction_candidate(
            &resident_variants,
            requested_variant,
            &active_variants,
            &pinned_variants,
            &last_used,
        );

        assert_eq!(candidate, Some(ModelVariant::Qwen3Tts12Hz06BCustomVoice));

        // When every resident is pinned there is no candidate.
        let all_pinned = HashSet::from([
            ModelVariant::Qwen359BGguf,
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
        ]);
        assert_eq!(
            select_lru_eviction_candidate(
                &resident_variants,
                requested_variant,
                &active_variants,
                &all_pinned,
                &last_used,
            ),
            None
        );
    }

    #[test]
    fn residency_budget_requires_space_before_loading_a_replacement() {
        let resident_variants = vec![ModelVariant::Kokoro82M];

        assert!(!residency_budget_has_capacity(
            &resident_variants,
            ModelVariant::Qwen359BGguf,
            1,
        ));
        assert!(residency_budget_has_capacity(
            &resident_variants,
            ModelVariant::Kokoro82M,
            1,
        ));
        assert!(residency_budget_has_capacity(
            &resident_variants,
            ModelVariant::Qwen359BGguf,
            2,
        ));
    }

    #[test]
    fn model_load_resource_plan_authorizes_backend_specific_peaks() {
        let estimate = ModelMemoryEstimate {
            load_peak_bytes: 96,
            resident_bytes: 64,
        };

        let cpu = model_resource_plan(BackendKind::Cpu, estimate);
        assert_eq!(
            cpu.load_authorization,
            ResourceVector {
                host_bytes: ResourceAmount::Known(96),
                ..ResourceVector::zero()
            }
        );
        assert_eq!(
            cpu.resident_authorization,
            ResourceVector {
                host_bytes: ResourceAmount::Known(64),
                ..ResourceVector::zero()
            }
        );

        let metal = model_resource_plan(BackendKind::Metal, estimate);
        assert_eq!(
            metal.load_authorization,
            ResourceVector {
                unified_bytes: ResourceAmount::Known(96),
                ..ResourceVector::zero()
            }
        );
        assert_eq!(
            metal.resident_authorization,
            ResourceVector {
                unified_bytes: ResourceAmount::Known(64),
                ..ResourceVector::zero()
            }
        );

        let cuda = model_resource_plan(BackendKind::Cuda, estimate);
        assert_eq!(
            cuda.load_authorization,
            ResourceVector {
                host_bytes: ResourceAmount::Known(96),
                device_bytes: ResourceAmount::Known(96),
                ..ResourceVector::zero()
            }
        );
        assert_eq!(
            cuda.resident_authorization,
            ResourceVector {
                device_bytes: ResourceAmount::Known(64),
                ..ResourceVector::zero()
            }
        );
    }

    #[test]
    fn fish_s2_cuda_weight_lease_leaves_room_for_separately_charged_state() {
        const GIB: u64 = 1024 * 1024 * 1024;
        // Pinned checkpoint header arithmetic is verified in the Fish weight
        // tests. This regression checks the resource-authority wiring, without
        // claiming that every request or native-context cache fits this device.
        let memory = FishS2ModelMemory {
            resident_bytes: 10_712_372_164,
            load_peak_bytes: 12_307_518_404,
            cuda_host_load_peak_bytes: 9_944_351_744,
        };
        let plan = fish_s2_resource_plan(BackendKind::Cuda, memory).unwrap();
        assert_eq!(
            plan.resident_authorization,
            ResourceVector {
                device_bytes: ResourceAmount::Known(memory.resident_bytes),
                host_bytes: ResourceAmount::Known(
                    crate::models::architectures::fish_s2::FISH_S2_REFERENCE_CACHE_BYTES
                ),
                ..ResourceVector::zero()
            }
        );
        let authority = vector_authority(ResourceVector {
            host_bytes: ResourceAmount::Known(32 * GIB),
            device_bytes: ResourceAmount::Known(24 * GIB),
            ..ResourceVector::zero()
        });
        let state = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Cache, "fish-s2-state"),
                ResourceVector {
                    device_bytes: ResourceAmount::Known(4 * GIB),
                    ..ResourceVector::zero()
                },
            )
            .unwrap();
        let weights = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "fish-s2-weights"),
                plan.load_authorization,
            )
            .expect("the weight peak and an independent state lease should fit");
        drop(weights);
        let old_catalog_plan = model_resource_plan(
            BackendKind::Cuda,
            ModelMemoryEstimate {
                load_peak_bytes: 24 * GIB,
                resident_bytes: 24 * GIB,
            },
        );
        let error = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "fish-s2-flat-catalog"),
                old_catalog_plan.load_authorization,
            )
            .expect_err("the old flat 24 GiB lease left no room for state");
        assert!(matches!(error, Error::Overloaded(_)));
        drop(state);
    }

    #[test]
    fn fish_s2_lazy_reference_cache_keeps_headroom_reserved_after_weight_load() {
        let cache = crate::models::architectures::fish_s2::FISH_S2_REFERENCE_CACHE_BYTES;
        let host = |n| ResourceVector {
            host_bytes: ResourceAmount::Known(n),
            ..ResourceVector::zero()
        };
        let plan = fish_s2_resource_plan(
            BackendKind::Cpu,
            FishS2ModelMemory {
                resident_bytes: 100,
                load_peak_bytes: 200,
                cuda_host_load_peak_bytes: 0,
            },
        )
        .unwrap();
        assert_eq!(plan.deferred_resident_authorization, host(cache));
        let authority = Arc::new(ResourceAuthority::new(Arc::new(VectorCapacityProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity: host(cache + 1000),
                available: host(cache + 200),
                source: CapacitySource::Test,
            },
        })));
        let mut lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "fish-weights-and-reference-cache"),
                plan.load_authorization,
            )
            .unwrap();
        lease
            .reconcile_materialized(
                plan.resident_authorization
                    .checked_sub(plan.deferred_resident_authorization)
                    .unwrap(),
            )
            .unwrap();
        lease.resize(plan.resident_authorization).unwrap();
        assert!(authority
            .reserve(
                ReservationOwner::new(ReservationClass::Cache, "oversized-state"),
                host(201)
            )
            .is_err());
        let state = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Cache, "fitted-state"),
                host(200),
            )
            .unwrap();
        drop(state);
        drop(lease);
        assert_eq!(authority.snapshot().reserved, ResourceVector::zero());
    }

    #[test]
    fn fish_s2_portable_weight_plan_charges_the_expanded_representation() {
        let cpu_memory = FishS2ModelMemory {
            resident_bytes: 19_836_076_996,
            load_peak_bytes: 22_228_796_356,
            cuda_host_load_peak_bytes: 9_944_351_744,
        };
        let cpu = fish_s2_resource_plan(BackendKind::Cpu, cpu_memory).unwrap();
        assert_eq!(
            cpu.resident_authorization.host_bytes,
            ResourceAmount::Known(
                cpu_memory.resident_bytes
                    + crate::models::architectures::fish_s2::FISH_S2_REFERENCE_CACHE_BYTES
            )
        );
        assert_eq!(
            cpu.load_authorization.host_bytes,
            ResourceAmount::Known(
                cpu_memory.load_peak_bytes
                    + crate::models::architectures::fish_s2::FISH_S2_REFERENCE_CACHE_BYTES
            )
        );
        assert_eq!(
            cpu.load_authorization.device_bytes,
            ResourceAmount::Known(0)
        );
        let half_memory = FishS2ModelMemory {
            resident_bytes: 10_712_372_164,
            load_peak_bytes: 12_307_518_404,
            ..cpu_memory
        };
        let metal = fish_s2_resource_plan(BackendKind::Metal, half_memory).unwrap();
        assert_eq!(
            metal.load_authorization.unified_bytes,
            ResourceAmount::Known(half_memory.load_peak_bytes)
        );
        assert_eq!(
            metal.resident_authorization.unified_bytes,
            ResourceAmount::Known(half_memory.resident_bytes)
        );
        assert_eq!(
            metal.load_authorization.host_bytes,
            ResourceAmount::Known(
                crate::models::architectures::fish_s2::FISH_S2_REFERENCE_CACHE_BYTES
            )
        );
    }

    #[test]
    fn lfm25_audio_model_memory_excludes_request_scoped_inference_workspace() {
        const GIB: u64 = 1024 * 1024 * 1024;

        assert_eq!(ModelVariant::Lfm25Audio15BGguf.memory_required_gb(), 5.0);
        assert_eq!(
            model_memory_estimate(ModelVariant::Lfm25Audio15BGguf),
            ModelMemoryEstimate {
                load_peak_bytes: 3 * GIB,
                resident_bytes: 3 * GIB,
            }
        );
    }

    #[test]
    fn bounded_model_memos_are_part_of_resident_authorization() {
        const GIB: u64 = 1024 * 1024 * 1024;

        let nemotron = model_memory_estimate(ModelVariant::Nemotron35AsrStreaming06B);
        assert_eq!(
            nemotron.resident_bytes,
            6 * GIB + crate::models::architectures::nemotron::asr::NEMOTRON_MODEL_MEMO_MAX_BYTES
        );
        let kokoro = model_memory_estimate(ModelVariant::Kokoro82M);
        assert_eq!(
            kokoro.resident_bytes,
            2 * GIB + crate::models::architectures::kokoro::KOKORO_MODEL_MEMO_MAX_BYTES
        );
        assert_eq!(nemotron.load_peak_bytes, nemotron.resident_bytes);
        assert_eq!(kokoro.load_peak_bytes, kokoro.resident_bytes);
    }

    #[test]
    fn kokoro_effective_context_respects_model_and_runtime_limits() {
        assert_eq!(kokoro_effective_context_tokens(512, 1_024).unwrap(), 512);
        assert_eq!(kokoro_effective_context_tokens(512, 256).unwrap(), 256);
        assert!(kokoro_effective_context_tokens(0, 1_024).is_err());
        assert!(kokoro_effective_context_tokens(512, 0).is_err());
    }

    #[test]
    fn automatic_invocation_context_treats_published_reach_as_a_ceiling() {
        assert_eq!(
            portable_invocation_context_intent(
                ContextLengthPreference::Auto,
                Some(128_000),
                131_072
            ),
            PortableInvocationContextIntent::Automatic { ceiling: 128_000 }
        );
        assert_eq!(
            portable_invocation_context_intent(
                ContextLengthPreference::Auto,
                Some(196_608),
                65_536
            ),
            PortableInvocationContextIntent::Automatic { ceiling: 65_536 }
        );
    }

    #[test]
    fn automatic_state_groups_share_remaining_headroom() {
        assert_eq!(automatic_state_group_budget(5_000, 5), 1_000);
        assert_eq!(automatic_state_group_budget(5_000, 2), 2_500);
        assert_eq!(automatic_state_group_budget(5_000, 1), 5_000);
        assert_eq!(automatic_state_group_budget(5_000, 0), 5_000);
    }

    #[test]
    fn lfm25_audio_context_fit_preserves_request_workspace_and_safety_headroom() {
        const GIB: u64 = 1024 * 1024 * 1024;

        assert_eq!(
            portable_context_reserve_bytes(ModelVariant::Lfm25Audio15BGguf, GIB),
            3 * GIB
        );
        assert_eq!(
            portable_context_reserve_bytes(ModelVariant::Kokoro82M, GIB),
            GIB
        );
    }

    #[test]
    fn explicit_invocation_context_remains_mandatory_after_state_publication() {
        let explicit = ContextLengthPreference::explicit(32_768).unwrap();
        assert_eq!(
            portable_invocation_context_intent(explicit, Some(8_192), 65_536),
            PortableInvocationContextIntent::Explicit { selected: 32_768 }
        );
    }

    #[test]
    fn lfm25_audio_cold_load_fits_with_separately_reserved_request_workspace() {
        const GIB: u64 = 1024 * 1024 * 1024;

        let authority = vector_authority(ResourceVector {
            unified_bytes: ResourceAmount::Known(5 * GIB),
            ..ResourceVector::zero()
        });
        let _request = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Request, "lfm-request"),
                ResourceVector {
                    unified_bytes: ResourceAmount::Known(2 * GIB),
                    ..ResourceVector::zero()
                },
            )
            .expect("request workspace should fit");
        let plan = model_resource_plan(
            BackendKind::Metal,
            model_memory_estimate(ModelVariant::Lfm25Audio15BGguf),
        );

        let _model = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "lfm-model"),
                plan.load_authorization,
            )
            .expect("model-owned memory should not include request workspace twice");
    }

    #[test]
    fn cuda_load_is_rejected_when_only_device_peak_has_capacity() {
        let plan = model_resource_plan(
            BackendKind::Cuda,
            ModelMemoryEstimate {
                load_peak_bytes: 64,
                resident_bytes: 64,
            },
        );
        let capacity = ResourceVector {
            host_bytes: ResourceAmount::Known(63),
            device_bytes: ResourceAmount::Known(64),
            ..ResourceVector::zero()
        };
        let authority = vector_authority(capacity);

        assert!(matches!(
            authority.reserve(
                ReservationOwner::new(ReservationClass::Model, "cuda-host-peak"),
                plan.load_authorization,
            ),
            Err(Error::Overloaded(_))
        ));
        assert_eq!(authority.snapshot().reservations, 0);
        assert_eq!(authority.snapshot().reserved, ResourceVector::zero());
    }

    #[tokio::test]
    async fn published_models_retain_only_backend_residency_authorization() {
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-resource-finalize-test-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;

        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let plan = model_resource_plan(
                backend,
                ModelMemoryEstimate {
                    load_peak_bytes: 64,
                    resident_bytes: 64,
                },
            );
            let authority = vector_authority(all_memory_capacity(1024));
            let resource_lease = authority
                .reserve(
                    ReservationOwner::new(
                        ReservationClass::Model,
                        format!("{backend:?}-publication"),
                    ),
                    plan.load_authorization,
                )
                .expect("peak load authorization");
            runtime
                .model_lifecycle
                .install_loading_slot(variant, resource_lease)
                .expect("loading slot");

            runtime
                .model_lifecycle
                .finalize_slot_materialization(variant, plan.resident_authorization)
                .expect("publication resource finalization");
            assert_eq!(
                authority.snapshot().reserved,
                plan.resident_authorization,
                "{backend:?} retained the wrong resident authorization"
            );

            assert!(runtime.model_lifecycle.remove_resident_slot(variant));
            assert_eq!(authority.snapshot().reserved, ResourceVector::zero());
        }

        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[tokio::test]
    async fn deferred_model_graph_capacity_remains_pending_during_state_fitting() {
        let models_dir =
            std::env::temp_dir().join(format!("izwi-deferred-graphs-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..Default::default()
        })
        .unwrap();
        let authority = Arc::new(ResourceAuthority::new(Arc::new(VectorCapacityProvider {
            snapshot: PhysicalCapacitySnapshot {
                capacity: all_memory_capacity(1024),
                available: all_memory_capacity(100),
                source: CapacitySource::Test,
            },
        })));
        let bytes = |n| ResourceVector {
            device_bytes: ResourceAmount::Known(n),
            ..ResourceVector::zero()
        };
        let variant = ModelVariant::Qwen3827BFp8;
        let lease = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "weights-and-graphs"),
                bytes(64),
            )
            .unwrap();
        runtime
            .model_lifecycle
            .install_loading_slot(variant, lease)
            .unwrap();
        runtime
            .model_lifecycle
            .finalize_slot_materialization_with_pending(variant, bytes(64), bytes(16))
            .unwrap();
        assert!(matches!(
            authority.reserve(
                ReservationOwner::new(ReservationClass::Model, "oversized-state"),
                bytes(85)
            ),
            Err(Error::Overloaded(_))
        ));
        let state = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "fitted-state"),
                bytes(84),
            )
            .unwrap();
        assert!(runtime.model_lifecycle.remove_resident_slot(variant));
        drop(state);
        assert_eq!(authority.snapshot().reserved, ResourceVector::zero());
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn explicit_unload_supersedes_registered_load_before_spawn() {
        // Serialize the load transaction's residency reservation on the
        // process-global authority.
        let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-pre-gate-load-unload-test-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;

        // Register the generation but deliberately delay spawning its detached
        // transaction. This is the exact window where unload used to return
        // before the stale task acquired the gate and published the model.
        let (waiter, leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let leader = leader.expect("registered load leader");
        let stale_generation = leader.generation;

        runtime
            .unload_model(variant)
            .await
            .expect("explicit unload supersedes pending load");
        {
            let _mutation_guard = runtime.model_lifecycle.mutation_gate.lock().await;
            runtime.model_lifecycle.finish_load_locked(
                variant,
                leader.generation,
                &leader.completion,
                SharedLoadOutcome::Ready,
            );
        }
        let error = tokio::time::timeout(Duration::from_secs(1), waiter.wait())
            .await
            .expect("superseded waiter timed out")
            .expect_err("superseded load must fail");
        assert!(matches!(
            error,
            Error::Cancelled(message)
                if message.contains("superseded by explicit unload")
        ));

        let stale_task = runtime.model_lifecycle.spawn_load_transaction(
            variant,
            runtime.max_loaded_models,
            leader,
        );
        tokio::time::timeout(Duration::from_secs(1), stale_task)
            .await
            .expect("stale detached load timed out")
            .expect("stale detached load task");

        assert_eq!(runtime.model_lifecycle.resident_phase(variant), None);
        assert!(!runtime.model_manager.is_ready(variant).await);
        assert_eq!(runtime.coordinator.snapshot().active_model_loads, 0);

        // Removal of the stale registration must allow a later request to own
        // a new generation rather than coalescing with cancelled work.
        let (retry_waiter, retry_leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let retry_leader = retry_leader.expect("new load generation after unload");
        assert_ne!(retry_leader.generation, stale_generation);
        {
            let _mutation_guard = runtime.model_lifecycle.mutation_gate.lock().await;
            runtime.model_lifecycle.finish_load_locked(
                variant,
                retry_leader.generation,
                &retry_leader.completion,
                SharedLoadOutcome::Failed(SharedLoadFailure::Cancelled("test cleanup".to_string())),
            );
        }
        assert!(matches!(
            retry_waiter.wait().await,
            Err(Error::Cancelled(_))
        ));

        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unload_all_supersedes_every_registered_load_before_spawn() {
        // Serialize the load transaction's residency reservation on the
        // process-global authority (the GGUF variant alone claims ~2.25 GiB).
        let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-pre-gate-load-unload-all-test-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variants = [ModelVariant::Kokoro82M, ModelVariant::Qwen359BGguf];
        let mut registrations = Vec::new();
        for variant in variants {
            let (waiter, leader) = runtime.model_lifecycle.join_or_start_load(variant);
            registrations.push((variant, waiter, leader.expect("registered load leader")));
        }

        assert_eq!(
            runtime
                .unload_all_models()
                .await
                .expect("unload-all supersedes pending loads"),
            0
        );

        for (variant, waiter, leader) in registrations {
            let error = tokio::time::timeout(Duration::from_secs(1), waiter.wait())
                .await
                .expect("superseded unload-all waiter timed out")
                .expect_err("superseded load must fail");
            assert!(matches!(
                error,
                Error::Cancelled(message)
                    if message.contains("superseded by explicit unload-all")
            ));
            let stale_task = runtime.model_lifecycle.spawn_load_transaction(
                variant,
                runtime.max_loaded_models,
                leader,
            );
            tokio::time::timeout(Duration::from_secs(1), stale_task)
                .await
                .expect("stale unload-all load timed out")
                .expect("stale unload-all task");
            assert_eq!(runtime.model_lifecycle.resident_phase(variant), None);
            assert!(!runtime.model_manager.is_ready(variant).await);
        }
        assert_eq!(runtime.coordinator.snapshot().active_model_loads, 0);

        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ready_outcome_is_published_before_explicit_unload_enters_the_gate() {
        // Serialize the load transaction's residency reservation on the
        // process-global authority.
        let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-load-publication-race-test-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;
        let resources = one_byte_host_reservation();
        let (authority, resource_lease) = isolated_resource_lease("publication-race");
        let installed_instance = runtime
            .model_lifecycle
            .install_loading_slot(variant, resource_lease)
            .expect("loading slot");
        runtime
            .model_lifecycle
            .finalize_slot_materialization(variant, resources)
            .expect("materialized slot");
        let (waiter, leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let leader = leader.expect("load leader");

        let publication_reached = Arc::new(Barrier::new(2));
        let publication_release = Arc::new(Barrier::new(2));
        let events = Arc::new(StdMutex::new(Vec::new()));
        let controller = runtime.model_lifecycle.clone();
        let reached = publication_reached.clone();
        let release = publication_release.clone();
        let publication_events = events.clone();
        let publisher = tokio::spawn(async move {
            let _mutation_guard = controller.mutation_gate.lock().await;
            controller.model_manager.mark_loaded(variant).await;
            controller.mark_slot_ready(variant).expect("ready slot");
            reached.wait().await;
            release.wait().await;
            controller.finish_load_locked(
                variant,
                leader.generation,
                &leader.completion,
                SharedLoadOutcome::Ready,
            );
            publication_events
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .push("outcome");
        });

        publication_reached.wait().await;
        let lease = runtime
            .model_lifecycle
            .try_acquire_ready_lease(variant)
            .expect("ready instance lease");
        assert_eq!(lease.model_instance_id(), Some(installed_instance));
        drop(lease);
        let bundle = runtime
            .model_lifecycle
            .try_get_ready_bundle(variant)
            .expect("ready execution bundle");
        assert_eq!(bundle.model_instance_id(), installed_instance);
        assert_eq!(bundle.model_variant(), variant);
        assert_eq!(
            bundle.execution_group_id(),
            runtime.coordinator.execution_group_id()
        );
        assert!(bundle.adapter_count() > 0);
        let unload_controller = runtime.model_lifecycle.clone();
        let unload_events = events.clone();
        let mut unload = tokio::spawn(async move {
            unload_controller
                .unload_model_detached(variant)
                .await
                .expect("explicit unload");
            unload_events
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .push("unload");
        });
        assert!(
            tokio::time::timeout(Duration::from_millis(25), &mut unload)
                .await
                .is_err(),
            "unload must not cross the gate before the load outcome is published"
        );

        publication_release.wait().await;
        waiter.wait().await.expect("shared ready outcome");
        publisher.await.expect("publisher task");
        unload.await.expect("unload task");
        assert_eq!(
            *events.lock().unwrap_or_else(|poison| poison.into_inner()),
            vec!["outcome", "unload"]
        );
        assert_eq!(runtime.model_lifecycle.resident_phase(variant), None);
        assert_eq!(authority.snapshot().reservations, 0);

        std::fs::remove_dir_all(models_dir).unwrap();
    }

    // The OOM-ladder cases and every other test that reserves against the
    // process-global authority serialize on the shared
    // crate::engine::resources::test_support::authority_test_lock().

    struct ClearAuthorityPoisonOnDrop(Arc<ResourceAuthority>);

    impl Drop for ClearAuthorityPoisonOnDrop {
        fn drop(&mut self) {
            self.0.clear_poison();
        }
    }

    // The ladder lock deliberately spans every await in these tests: it
    // serializes the process-global resource authority against the other
    // Metal OOM ladder cases, not shared state inside this runtime.
    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn metal_oom_load_is_flushed_and_retried_once_before_poisoning() {
        let _ladder_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-load-oom-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;
        runtime.model_lifecycle.set_load_test_metal_ooms(1);

        let (waiter, leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let _load_task = runtime.model_lifecycle.spawn_load_transaction(
            variant,
            runtime.max_loaded_models,
            leader.expect("load leader"),
        );
        // The first attempt fails with the injected command-buffer OOM; the
        // wrapper must flush pooled scratch and retry once. The retry is not
        // injected, so it fails with an ordinary missing-model error instead
        // of the OOM — proving the attempt actually re-ran.
        let error = tokio::time::timeout(Duration::from_secs(2), waiter.wait())
            .await
            .expect("retry outcome timed out")
            .expect_err("missing model artifacts must fail");
        assert!(
            !format!("{error}").contains("kIOGPUCommandBufferCallbackErrorOutOfMemory"),
            "the retried attempt must not surface the injected OOM: {error}"
        );
        assert!(
            runtime
                .coordinator
                .resource_authority()
                .poison_reason()
                .is_none(),
            "a single OOM must not poison the authority"
        );
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn persisted_metal_oom_after_the_retry_poisons_the_authority() {
        let _ladder_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-load-oom2-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;
        runtime.model_lifecycle.set_load_test_metal_ooms(2);
        // A resident blocks the recovery drain, so the poison is observable
        // after the failed load. Unpinned residents would be evicted and the
        // authority would recover before the outcome lands.
        runtime
            .model_manager
            .mark_loaded(ModelVariant::WhisperLargeV3Turbo)
            .await;
        runtime
            .model_lifecycle
            .pin_model(ModelVariant::WhisperLargeV3Turbo)
            .await;

        let (waiter, leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let _load_task = runtime.model_lifecycle.spawn_load_transaction(
            variant,
            runtime.max_loaded_models,
            leader.expect("load leader"),
        );
        let error = tokio::time::timeout(Duration::from_secs(2), waiter.wait())
            .await
            .expect("poison outcome timed out")
            .expect_err("persisted OOM must fail");
        assert!(
            format!("{error}").contains("kIOGPUCommandBufferCallbackErrorOutOfMemory"),
            "the injected OOM must surface after the retry: {error}"
        );
        let reason = runtime
            .coordinator
            .resource_authority()
            .poison_reason()
            .expect("persisted OOM must poison the authority");
        assert!(
            reason.contains("persisted after a pooled-scratch flush and one retry"),
            "unexpected poison reason: {reason}"
        );
        drop(ClearAuthorityPoisonOnDrop(
            runtime.coordinator.resource_authority(),
        ));
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn poisoned_authority_recovers_when_the_device_fully_drains() {
        let _ladder_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-poison-recover-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            model_keep_alive_secs: 1,
            ..EngineConfig::default()
        })
        .unwrap();
        let authority = runtime.coordinator.resource_authority();
        authority.poison("simulated backend-fatal error");
        runtime
            .model_manager
            .mark_loaded(ModelVariant::GraniteSpeech412BPlus)
            .await;
        runtime
            .model_lifecycle
            .model_last_used
            .lock()
            .await
            .insert(
                ModelVariant::GraniteSpeech412BPlus,
                now_unix_millis().saturating_sub(10_000),
            );

        // The unpinned resident is the only thing keeping the device from
        // draining; reaping it via the standard unload path must clear the
        // poison.
        let reaped = runtime.reap_idle_models().await;
        assert_eq!(reaped, vec![ModelVariant::GraniteSpeech412BPlus]);
        assert!(
            authority.poison_reason().is_none(),
            "a fully drained device must recover from poison"
        );

        // A pinned resident blocks the drain, so the poison persists.
        authority.poison("simulated backend-fatal error");
        runtime
            .model_manager
            .mark_loaded(ModelVariant::WhisperLargeV3Turbo)
            .await;
        runtime
            .model_lifecycle
            .pin_model(ModelVariant::WhisperLargeV3Turbo)
            .await;
        runtime
            .model_lifecycle
            .model_last_used
            .lock()
            .await
            .insert(
                ModelVariant::WhisperLargeV3Turbo,
                now_unix_millis().saturating_sub(10_000),
            );
        assert!(runtime.reap_idle_models().await.is_empty());
        assert!(
            authority.poison_reason().is_some(),
            "a pinned resident must keep the poison until it is unloaded"
        );
        drop(ClearAuthorityPoisonOnDrop(authority));
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn detached_load_panic_rolls_back_before_publishing_failure() {
        // Serialize the load transaction's residency reservation on the
        // process-global authority.
        let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-load-panic-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;
        let (authority, resource_lease) = isolated_resource_lease("load-panic");
        runtime
            .model_lifecycle
            .install_loading_slot(variant, resource_lease)
            .expect("loading slot");
        runtime.model_manager.mark_loaded(variant).await;
        runtime.model_lifecycle.set_load_test_panics(1);

        let (waiter, leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let _load_task = runtime.model_lifecycle.spawn_load_transaction(
            variant,
            runtime.max_loaded_models,
            leader.expect("load leader"),
        );
        let error = tokio::time::timeout(Duration::from_secs(1), waiter.wait())
            .await
            .expect("panic outcome timed out")
            .expect_err("injected load panic must fail");
        assert!(
            matches!(error, Error::ModelLoadError(message) if message.contains("injected model load panic"))
        );
        assert_eq!(runtime.model_lifecycle.resident_phase(variant), None);
        assert!(!runtime.model_manager.is_ready(variant).await);
        assert_eq!(authority.snapshot().reservations, 0);

        let (cleanup_waiter, cleanup_leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let cleanup_leader = cleanup_leader.expect("failed generation must be removable");
        {
            let _mutation_guard = runtime.model_lifecycle.mutation_gate.lock().await;
            runtime.model_lifecycle.finish_load_locked(
                variant,
                cleanup_leader.generation,
                &cleanup_leader.completion,
                SharedLoadOutcome::Failed(SharedLoadFailure::ModelLoad("test cleanup".to_string())),
            );
        }
        assert!(cleanup_waiter.wait().await.is_err());

        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[tokio::test]
    async fn idle_reaper_unloads_idle_transient_models_but_preserves_pins() {
        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-reap-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            model_keep_alive_secs: 1,
            ..EngineConfig::default()
        })
        .unwrap();
        runtime
            .model_manager
            .mark_loaded(ModelVariant::GraniteSpeech412BPlus)
            .await;
        runtime
            .model_manager
            .mark_loaded(ModelVariant::WhisperLargeV3Turbo)
            .await;
        runtime
            .model_lifecycle
            .pin_model(ModelVariant::WhisperLargeV3Turbo)
            .await;

        let stale = now_unix_millis().saturating_sub(10_000);
        {
            let mut last_used = runtime.model_lifecycle.model_last_used.lock().await;
            last_used.insert(ModelVariant::GraniteSpeech412BPlus, stale);
            last_used.insert(ModelVariant::WhisperLargeV3Turbo, stale);
        }

        let reaped = runtime.reap_idle_models().await;
        assert_eq!(reaped, vec![ModelVariant::GraniteSpeech412BPlus]);

        let residents = runtime.model_manager.resident_variants().await;
        assert!(!residents.contains(&ModelVariant::GraniteSpeech412BPlus));
        assert!(residents.contains(&ModelVariant::WhisperLargeV3Turbo));
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[tokio::test]
    async fn idle_reaper_is_disabled_without_keep_alive() {
        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-reap-off-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            model_keep_alive_secs: 0,
            ..EngineConfig::default()
        })
        .unwrap();
        runtime
            .model_manager
            .mark_loaded(ModelVariant::GraniteSpeech412BPlus)
            .await;
        runtime
            .model_lifecycle
            .model_last_used
            .lock()
            .await
            .insert(
                ModelVariant::GraniteSpeech412BPlus,
                now_unix_millis().saturating_sub(10_000),
            );

        assert!(runtime.reap_idle_models().await.is_empty());
        assert!(runtime
            .model_manager
            .resident_variants()
            .await
            .contains(&ModelVariant::GraniteSpeech412BPlus));
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[tokio::test]
    async fn residency_budget_preserves_pinned_residents_and_reports_make_room() {        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-residency-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            max_loaded_models: Some(1),
            ..EngineConfig::default()
        })
        .unwrap();
        runtime
            .model_manager
            .mark_loaded(ModelVariant::GraniteSpeech412BPlus)
            .await;
        runtime
            .model_lifecycle
            .pin_model(ModelVariant::GraniteSpeech412BPlus)
            .await;

        let error = runtime
            .ensure_model_budget_before_load(ModelVariant::WhisperLargeV3Turbo)
            .await
            .expect_err("a pinned resident must not be silently evicted");
        assert!(
            error.to_string().contains("unload a model to make room"),
            "unexpected error: {error}"
        );

        assert!(runtime
            .model_manager
            .resident_variants()
            .await
            .contains(&ModelVariant::GraniteSpeech412BPlus));
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[tokio::test]
    async fn residency_budget_evicts_granite_before_loading_another_asr_model() {
        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-residency-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            max_loaded_models: Some(1),
            ..EngineConfig::default()
        })
        .unwrap();
        runtime
            .model_manager
            .mark_loaded(ModelVariant::GraniteSpeech412BPlus)
            .await;

        runtime
            .ensure_model_budget_before_load(ModelVariant::WhisperLargeV3Turbo)
            .await
            .unwrap();

        assert!(runtime.model_manager.resident_variants().await.is_empty());
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn cold_model_load_is_rejected_before_artifact_work_during_drain() {
        // Serialize the load transaction's residency reservation on the
        // process-global authority.
        let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir =
            std::env::temp_dir().join(format!("izwi-runtime-load-drain-test-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        runtime.coordinator.begin_drain();

        assert!(matches!(
            runtime
                .load_model_for_inference(ModelVariant::Kokoro82M)
                .await,
            Err(Error::Overloaded(_))
        ));
        assert!(runtime.model_manager.resident_variants().await.is_empty());

        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn cancelled_waiter_keeps_shared_load_accounted_and_visible_to_drain() {
        // Serialize the load transaction's residency reservation on the
        // process-global authority.
        let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-cancelled-load-test-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;
        let resources = one_byte_host_reservation();
        let (authority, resource_lease) = isolated_resource_lease("cancelled-load-test");
        runtime
            .model_lifecycle
            .install_loading_slot(variant, resource_lease)
            .expect("loading slot");

        let (first_waiter, leader) = runtime.model_lifecycle.join_or_start_load(variant);
        let leader = leader.expect("first waiter leads the load");
        let (second_waiter, second_leader) = runtime.model_lifecycle.join_or_start_load(variant);
        assert!(second_leader.is_none(), "second waiter must coalesce");

        let first_task = tokio::spawn(first_waiter.wait());
        let second_task = tokio::spawn(second_waiter.wait());
        let (started_tx, started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let loader_calls = Arc::new(AtomicUsize::new(0));
        let controller = runtime.model_lifecycle.clone();
        let calls = loader_calls.clone();
        let transaction = tokio::spawn(async move {
            let _mutation_guard = controller.mutation_gate.lock().await;
            let _coordinator_load = controller
                .coordinator
                .begin_model_load("cancelled-shared-load")
                .expect("model load admission");
            calls.fetch_add(1, Ordering::AcqRel);
            let _ = started_tx.send(());
            let _ = release_rx.await;
            controller
                .finalize_slot_materialization(variant, resources)
                .expect("materialized lease");
            controller.model_manager.mark_loaded(variant).await;
            controller.mark_slot_ready(variant).expect("ready slot");
            controller.finish_load_locked(
                variant,
                leader.generation,
                &leader.completion,
                SharedLoadOutcome::Ready,
            );
        });

        started_rx.await.expect("fake load started");
        first_task.abort();
        assert!(first_task
            .await
            .expect_err("first waiter should be cancelled")
            .is_cancelled());
        assert_eq!(loader_calls.load(Ordering::Acquire), 1);
        assert_eq!(
            runtime.model_lifecycle.resident_phase(variant),
            Some(ResidentPhase::Loading)
        );
        assert_eq!(runtime.coordinator.snapshot().active_model_loads, 1);
        assert_eq!(authority.snapshot().reservations, 1);

        runtime.begin_drain();
        assert!(
            tokio::time::timeout(
                Duration::from_millis(25),
                runtime
                    .coordinator
                    .wait_for_idle(Instant::now() + Duration::from_secs(1)),
            )
            .await
            .is_err(),
            "drain must still observe the detached load"
        );

        release_tx.send(()).expect("release fake load");
        second_task
            .await
            .expect("second waiter join")
            .expect("coalesced waiter succeeds");
        transaction.await.expect("fake load transaction");
        runtime
            .coordinator
            .wait_for_idle(Instant::now() + Duration::from_secs(1))
            .await
            .expect("drain after load completion");
        assert_eq!(
            runtime.model_lifecycle.resident_phase(variant),
            Some(ResidentPhase::Ready)
        );
        assert!(runtime.model_manager.is_ready(variant).await);

        assert_eq!(
            runtime.unload_all_models().await.expect("shutdown unload"),
            1
        );
        assert_eq!(runtime.model_lifecycle.resident_phase(variant), None);
        assert_eq!(authority.snapshot().reservations, 0);
        std::fs::remove_dir_all(models_dir).unwrap();
    }

    #[tokio::test]
    async fn failed_publication_rolls_back_slot_before_releasing_lease() {
        let models_dir = std::env::temp_dir().join(format!(
            "izwi-runtime-load-rollback-test-{}",
            Uuid::new_v4()
        ));
        std::fs::create_dir_all(&models_dir).unwrap();
        let runtime = RuntimeService::new(EngineConfig {
            models_dir: models_dir.clone(),
            backend: BackendPreference::Cpu,
            ..EngineConfig::default()
        })
        .unwrap();
        let variant = ModelVariant::Kokoro82M;
        let (authority, resource_lease) = isolated_resource_lease("rollback-test");
        runtime
            .model_lifecycle
            .install_loading_slot(variant, resource_lease)
            .expect("loading slot");

        assert_eq!(
            runtime.model_lifecycle.resident_phase(variant),
            Some(ResidentPhase::Loading)
        );
        assert_eq!(authority.snapshot().reservations, 1);
        let _mutation_guard = runtime.model_lifecycle.mutation_gate.lock().await;
        runtime
            .model_lifecycle
            .rollback_model_locked(variant)
            .await
            .expect("rollback");

        assert_eq!(runtime.model_lifecycle.resident_phase(variant), None);
        assert!(!runtime.model_manager.is_ready(variant).await);
        assert_eq!(authority.snapshot().reservations, 0);
        std::fs::remove_dir_all(models_dir).unwrap();
    }
}
