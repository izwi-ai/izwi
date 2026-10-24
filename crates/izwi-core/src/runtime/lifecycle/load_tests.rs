use super::{
    admit_asr_request_floor, asr_state_rows, automatic_state_group_budget,
    estimate_from_tensor_inventory, fish_s2_resource_plan, is_metal_command_buffer_oom,
    kokoro_effective_context_tokens, loaded_asr_state_publication_route,
    managed_chat_capacity_policy, materialization_tolerance_bytes, model_memory_estimate,
    model_resource_plan, now_unix_millis, plan_invocation_allocations,
    portable_context_reserve_bytes, portable_invocation_context_intent, precision_reload_gate,
    qwen38_representation_memory_estimate, reconcile_observed_growth,
    residency_budget_has_capacity, select_lru_eviction_candidate, state_fit_budgets,
    validate_scratch_only_invocation_publication, DeviceUsageSample,
    LoadedAsrStatePublicationRoute, ModelMemoryEstimate, PortableInvocationContextIntent,
    PrecisionReloadRequest, QWEN38_CUDA_DEVICE_CONVERSION_SCRATCH_BYTES,
    QWEN38_CUDA_HOST_CONVERSION_SCRATCH_BYTES, QWEN38_PORTABLE_CONVERSION_SCRATCH_BYTES,
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

/// A Granite Speech load without resumable prefill publishes invocation
/// state only, and the bundle seals it: its ASR capability offers no
/// retained graph (GRN-8), so no graph declares retained cache state beside
/// the stateless publication, and speaker attribution publishes as before.
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn granite_speech_load_without_resumable_prefill_seals_invocation_only_state() {
    use crate::models::architectures::granite_speech::asr::{
        granite_speech_test_physical_state_spec, GraniteSpeechAsrModel,
    };
    use crate::runtime::adapters::{
        LoadedModelBundleDraft, LoadedStatePublication, StreamingRequirements,
    };
    const { assert!(!GraniteSpeechAsrModel::RESUMABLE_PREFILL) };
    // Serialize against OOM-ladder poison windows and load-transaction
    // reservations on the process-global resource authority.
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let directory =
        std::env::temp_dir().join(format!("izwi-granite-state-publication-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&directory).unwrap();
    let runtime = RuntimeService::new(EngineConfig {
        models_dir: directory.clone(),
        backend: BackendPreference::Cpu,
        max_sequence_length: ContextLengthPreference::explicit(32).unwrap(),
        ..Default::default()
    })
    .unwrap();
    let variant = ModelVariant::GraniteSpeech412BPlus;
    let authority = vector_authority(all_memory_capacity(1024));
    let lease = authority
        .reserve(
            ReservationOwner::new(ReservationClass::Model, "granite-load-publication"),
            ResourceVector::zero(),
        )
        .unwrap();
    let instance = runtime
        .model_lifecycle
        .install_loading_slot(variant, lease)
        .unwrap();
    let draft = LoadedModelBundleDraft::build(
        &RuntimeAdapterRegistry::built_in(),
        ExecutionGroupId::new(992),
        instance,
        variant,
        BackendKind::Cpu,
    )
    .unwrap();
    let publications = runtime
        .model_lifecycle
        .load_granite_speech_state_publications(
            variant,
            instance,
            &draft,
            granite_speech_test_physical_state_spec,
        )
        .await
        .unwrap();
    assert_eq!(publications.len(), 2);
    for capability in [CapabilityKind::Asr, CapabilityKind::SpeakerAttributedAsr] {
        let LoadedStatePublication::PhysicalV2 {
            descriptor,
            retained,
            retained_uses,
            ..
        } = &publications[&capability]
        else {
            panic!("Granite Speech {capability:?} must publish physical state");
        };
        assert!(retained.is_none(), "{capability:?}");
        assert!(retained_uses.is_empty(), "{capability:?}");
        assert!(
            matches!(descriptor.retained, RetainedStateCapability::Stateless),
            "{capability:?}"
        );
    }
    let bundle = draft
        .seal(publications)
        .expect("a Granite Speech load without resumable prefill must seal");

    // Every ASR requirement, ordinary or long-form, with or without
    // transport output, binds the atomic graph and its invocation state.
    for streaming in [
        StreamingRequirements::NONE,
        StreamingRequirements::transport_only(),
        StreamingRequirements::NONE.with_asr_long_form(true),
        StreamingRequirements::transport_only().with_asr_long_form(true),
    ] {
        let contract = bundle
            .contract_for_streaming(CapabilityKind::Asr, streaming)
            .unwrap();
        assert_eq!(contract.execution_profile.mode, ExecutionMode::Atomic);
        let binding = bundle
            .capability_binding_for_streaming(CapabilityKind::Asr, streaming)
            .unwrap();
        assert_eq!(binding.execution, contract.adapter_binding().unwrap());
        binding
            .state
            .validate_against(BackendKind::Cpu, &binding.execution)
            .unwrap();
        assert!(binding.state.managed_kv_runtime().is_none());
    }
    let attributed = bundle
        .capability_binding_for_streaming(
            CapabilityKind::SpeakerAttributedAsr,
            StreamingRequirements::NONE,
        )
        .unwrap();
    attributed
        .state
        .validate_against(BackendKind::Cpu, &attributed.execution)
        .unwrap();
    assert_eq!(
        attributed.execution.stages[0].selector,
        StageWorkSelector::Pipeline { ordinal: None }
    );
    drop(attributed);
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

/// The header census lists every tensor with its stored dtype, shape and
/// payload bytes, and the inventory folds the same records.
#[test]
fn checkpoint_census_records_every_tensor_header() {
    use super::{checkpoint_census, checkpoint_tensor_inventory, StoredDType};
    let dir = std::env::temp_dir().join(format!("izwi-census-{}", Uuid::new_v4()));
    std::fs::create_dir_all(dir.join("shards")).unwrap();
    let device = candle_core::Device::Cpu;
    let zeros = |shape: &[usize], dtype| candle_core::Tensor::zeros(shape, dtype, &device).unwrap();
    candle_core::safetensors::save(
        &HashMap::from([
            ("embed".to_string(), zeros(&[300, 16], candle_core::DType::BF16)),
            ("norm".to_string(), zeros(&[16], candle_core::DType::F32)),
        ]),
        dir.join("shards/model.safetensors"),
    )
    .unwrap();
    std::fs::write(dir.join("model.nemo"), vec![0u8; 4096]).unwrap();

    let census = checkpoint_census(&dir).unwrap();
    assert_eq!(census.files.len(), 1);
    assert_eq!(census.container_fallback_bytes, 4096);
    let mut tensors = census.files[0]
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), t.dtype, t.shape.clone(), t.stored_bytes))
        .collect::<Vec<_>>();
    tensors.sort_by_key(|tensor| tensor.0);
    assert_eq!(
        tensors,
        [
            ("embed", StoredDType::Dense(candle_core::DType::BF16), vec![300, 16], 9600),
            ("norm", StoredDType::Dense(candle_core::DType::F32), vec![16], 64),
        ]
    );
    assert_eq!(
        checkpoint_tensor_inventory(&dir).unwrap(),
        Some(TensorStorageInventory {
            total_bytes: 9664,
            largest_tensor_bytes: 9600,
            largest_tensor_elements: 4800,
            tensor_count: 2,
        })
    );
    // Without a readable checkpoint the container size stands in.
    std::fs::remove_dir_all(dir.join("shards")).unwrap();
    assert_eq!(
        checkpoint_tensor_inventory(&dir).unwrap(),
        Some(TensorStorageInventory {
            total_bytes: 4096,
            largest_tensor_bytes: 4096,
            largest_tensor_elements: 0,
            tensor_count: 1,
        })
    );
    std::fs::remove_dir_all(dir).unwrap();
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

/// Element census of the pinned Qwen3.8-27B checkpoint by resident class,
/// computed independently of the residency policy's tensor list.
mod qwen38_27b {
    /// Block-FP8 projections: 48 DeltaNet layers x 382,730,240, 16
    /// attention layers x 372,244,480, and the MTP layer's 372,244,480.
    pub const TRUNK_FP8: u64 = 48 * 382_730_240 + 16 * 372_244_480;
    pub const MTP_FP8: u64 = 372_244_480;
    /// Embedding and LM head (248,320 x 5,120 each).
    pub const EMBEDDING: u64 = 1_271_398_400;
    pub const LM_HEAD: u64 = 1_271_398_400;
    /// DeltaNet a/b projections, kept in the activation dtype.
    pub const TRUNK_DENSE: u64 = 48 * 2 * 48 * 5_120;
    /// Norm gains (F32 since C2) and DeltaNet math/conv (F32).
    pub const TRUNK_F32: u64 =
        5_120 + 64 * 2 * 5_120 + 16 * 2 * 256 + 48 * (128 + 48 + 48 + 10_240 * 4);
    pub const MTP_DENSE: u64 = 5_120 * 10_240;
    pub const MTP_F32: u64 = 5 * 5_120 + 2 * 256;
    /// F32 block scales of raw block FP8 (one per 128x128 block).
    pub const TRUNK_SCALES: u64 = 48 * 23_360 + 16 * 22_720;
    pub const MTP_SCALES: u64 = 22_720;
    /// The source checkpoint's BF16 elements also include a 460,730,096
    /// element vision tower the text runtime never loads.
    pub const VISION: u64 = 460_730_096;
}

fn qwen38_resource_plan(backend: BackendKind) -> super::ModelResourcePlan {
    super::qwen38_memory::static_resource_plan(
        backend,
        &crate::performance::PerformanceConfig::default(),
    )
    .unwrap()
}

#[test]
fn qwen38_resource_plan_prices_backend_resident_representations() {
    use qwen38_27b::*;
    let default = crate::performance::PerformanceConfig::default();
    let estimate = |backend| qwen38_representation_memory_estimate(backend, &default);
    let (cpu, metal, cuda) = (
        estimate(BackendKind::Cpu),
        estimate(BackendKind::Metal),
        estimate(BackendKind::Cuda),
    );
    let fp8 = TRUNK_FP8 + MTP_FP8;
    assert_eq!(fp8, 24_699_207_680);
    let dense = EMBEDDING + LM_HEAD + TRUNK_DENSE + MTP_DENSE;
    let f32_bytes = (TRUNK_F32 + MTP_F32) * 4;
    // The old static figure priced every BF16 source element (vision
    // included) at two bytes, and the F32 gains at two as well.
    assert_eq!(
        dense + TRUNK_F32 + MTP_F32 + VISION,
        3_082_220_272,
        "census accounts for every BF16 checkpoint element"
    );

    // The CPU keeps projections and the LM head as Q8_0 (34 bytes per 32
    // elements) and the rest F32; `expanded` expands everything to F32.
    assert_eq!(
        cpu.resident_bytes,
        (fp8 + LM_HEAD) / 32 * 34 + (dense - LM_HEAD) * 4 + f32_bytes
    );
    assert_eq!(cpu.resident_bytes, 32_994_136_064);
    let mut cpu_expanded = crate::performance::PerformanceConfig::default();
    cpu_expanded.cpu.projection_backend = crate::performance::CpuProjectionBackend::Expanded;
    assert_eq!(
        qwen38_representation_memory_estimate(BackendKind::Cpu, &cpu_expanded).resident_bytes,
        109_282_791_424
    );
    assert_eq!((fp8 + dense) * 4 + f32_bytes, 109_282_791_424);
    // Metal keeps raw block FP8 (bytes + F32 block scales) with F16 dense
    // tensors and the 1 GiB F16 prefill scratch; a build without the
    // Metal kernels expands to F16.
    let scratch =
        crate::models::architectures::qwen38::residency::METAL_FP8_PREFILL_DEQUANT_BYTES;
    let metal_native = fp8 + (TRUNK_SCALES + MTP_SCALES) * 4 + dense * 2 + f32_bytes + scratch;
    let metal_expanded = (fp8 + dense) * 2 + f32_bytes;
    if cfg!(feature = "metal") {
        assert_eq!(metal.resident_bytes, metal_native);
        assert_eq!(metal.resident_bytes, 31_027_303_168);
        // Fits a 48 GB Mac's 32.4 GiB budget (with a little context);
        // a 32 GB Mac's 21.6 GiB budget refuses it.
        const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
        assert!((metal.load_peak_bytes as f64) < 32.4 * GIB);
        assert!((metal.load_peak_bytes as f64) > 21.6 * GIB);
        let plan = qwen38_resource_plan(BackendKind::Metal);
        assert_eq!(
            plan.deferred_resident_authorization.unified_bytes,
            ResourceAmount::Known(scratch)
        );
    } else {
        assert_eq!(metal.resident_bytes, metal_expanded);
    }
    assert_eq!(metal_expanded, 54_646_738_944);
    // CUDA's static arm prices Q8_0 (34 bytes per 32 elements), plus the
    // F32 rows the DeltaNet alpha/beta router GEMV keeps beside the
    // dense pair.
    assert_eq!(
        cuda.resident_bytes,
        fp8 / 32 * 34 + dense * 2 + f32_bytes + TRUNK_DENSE * 4
    );
    assert_eq!(cuda.resident_bytes, 31_585_603_584);
    assert_eq!(
        cuda.load_peak_bytes,
        cuda.resident_bytes + QWEN38_CUDA_DEVICE_CONVERSION_SCRATCH_BYTES
    );
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

    // Admission follows the loader's MTP rule: without the head its
    // projections, fc and norms are not charged.
    let mut no_mtp = crate::performance::PerformanceConfig::default();
    no_mtp.cuda.mtp = crate::performance::OptimizationMode::Off;
    let mtp_metal = if cfg!(feature = "metal") {
        MTP_FP8 + MTP_SCALES * 4 + MTP_DENSE * 2 + MTP_F32 * 4
    } else {
        (MTP_FP8 + MTP_DENSE) * 2 + MTP_F32 * 4
    };
    assert_eq!(
        qwen38_representation_memory_estimate(BackendKind::Metal, &no_mtp).resident_bytes,
        metal.resident_bytes - mtp_metal
    );
    // `metal.projection_backend = "expanded"` prices the F16 expansion.
    let mut expanded = crate::performance::PerformanceConfig::default();
    expanded.metal.projection_backend = crate::performance::MetalProjectionBackend::Expanded;
    assert_eq!(
        qwen38_representation_memory_estimate(BackendKind::Metal, &expanded).resident_bytes,
        metal_expanded
    );
    // Off CUDA the CUDA master switch does not govern MTP; on CUDA it does.
    let mut cuda_off = crate::performance::PerformanceConfig::default();
    cuda_off.cuda.mode = crate::performance::OptimizationMode::Off;
    assert_eq!(
        qwen38_representation_memory_estimate(BackendKind::Metal, &cuda_off),
        metal
    );
    // The master switch also turns the alpha/beta GEMV (and its rows) off.
    assert_eq!(
        qwen38_representation_memory_estimate(BackendKind::Cuda, &cuda_off).resident_bytes,
        cuda.resident_bytes - MTP_FP8 / 32 * 34 - MTP_DENSE * 2 - MTP_F32 * 4 - TRUNK_DENSE * 4
    );
}

#[test]
fn qwen38_uses_engine_staged_width_while_cuda_retains_context_fitting() {
    use crate::engine::CudaResidentPool;
    let cuda_qwen38 =
        managed_chat_capacity_policy(ModelVariant::Qwen3827BFp8, BackendKind::Cuda);
    assert_eq!(cuda_qwen38.staged_transaction_rows, None);
    assert_eq!(cuda_qwen38.cuda_resident_pool, CudaResidentPool::RetainedRows);

    let metal_qwen38 =
        managed_chat_capacity_policy(ModelVariant::Qwen3827BFp8, BackendKind::Metal);
    assert_eq!(metal_qwen38.staged_transaction_rows, None);
    assert_eq!(metal_qwen38.cuda_resident_pool, CudaResidentPool::Disabled);

    let cuda_qwen35 =
        managed_chat_capacity_policy(ModelVariant::Qwen359BGguf, BackendKind::Cuda);
    assert_eq!(cuda_qwen35.staged_transaction_rows, None);
    assert_eq!(cuda_qwen35.cuda_resident_pool, CudaResidentPool::Disabled);

    // The Qwen3.5/3.6 MoE family fits its invocation-state arena against
    // CUDA resident headroom like Qwen3.8: its authored-context arena
    // otherwise cannot coexist with the resident representation on an
    // 80 GB device.
    let cuda_qwen36_moe =
        managed_chat_capacity_policy(ModelVariant::Qwen36Moe35BA3BFp8, BackendKind::Cuda);
    assert_eq!(cuda_qwen36_moe.staged_transaction_rows, None);
    assert_eq!(cuda_qwen36_moe.cuda_resident_pool, CudaResidentPool::RetainedRows);

    let metal_qwen36_moe =
        managed_chat_capacity_policy(ModelVariant::Qwen36Moe35BA3BFp8, BackendKind::Metal);
    assert_eq!(metal_qwen36_moe.cuda_resident_pool, CudaResidentPool::Disabled);

    // LFM2.5 chat fits one materialized context on CUDA and keeps the
    // portable fit elsewhere; families without a resident fit stay off.
    for variant in [
        ModelVariant::Lfm2512BInstructGguf,
        ModelVariant::Lfm2512BThinkingGguf,
    ] {
        assert_eq!(
            managed_chat_capacity_policy(variant, BackendKind::Cuda).cuda_resident_pool,
            CudaResidentPool::OneContext
        );
        for backend in [BackendKind::Cpu, BackendKind::Metal] {
            assert_eq!(
                managed_chat_capacity_policy(variant, backend).cuda_resident_pool,
                CudaResidentPool::Disabled
            );
        }
    }
}

#[test]
fn qwen36_config_state_bytes_match_its_contract() {
    use super::{hybrid_recurrent_state_bytes_per_sequence, qwen36moe_state_geometry};
    use crate::models::architectures::qwen36moe::{
        cache::qwen35_composite_cache_contract_with_mtp, native::pinned_native_config,
        native_model::qwen35_text_config_from_native,
    };
    let native = pinned_native_config();
    let from_config =
        hybrid_recurrent_state_bytes_per_sequence(qwen36moe_state_geometry(&native.text))
            .unwrap();
    // The bytes the managed planner authorizes per row for the real
    // contract builder, so the pre-load formula cannot drift from it.
    let contract = qwen35_composite_cache_contract_with_mtp(
        &qwen35_text_config_from_native(&native.text),
        candle_core::DType::BF16,
        64,
        false,
        false,
    )
    .unwrap();
    let plan = crate::backends::state::negotiate_state_plan(
        &contract,
        &crate::backends::state::StateBackendPlanRequest {
            backend: BackendKind::Cpu,
            device_ordinal: None,
            page_tokens_hint: Some(64),
            storage_dtype_hint: None,
        },
    )
    .unwrap();
    let (_, tensor) = crate::engine::plan_managed_state_capacity(
        &plan,
        ModelInstanceId::new(36),
        crate::engine::ManagedStateCapacityRequest {
            total_paged_pages: 64,
            logical_token_reach: None,
            retained_sequence_rows: 1,
            staged_transaction_rows: 1,
        },
    )
    .unwrap();
    assert_eq!(from_config, tensor.unwrap().per_sequence_bytes());
    assert_eq!(from_config, 65_863_680);
}

#[test]
fn cuda_state_floor_refuses_an_unfittable_plan_before_weights() {
    use super::{admit_cuda_state_floor, CudaStateFloor};
    const GIB: u64 = 1024 * 1024 * 1024;
    const H100_BYTES: u64 = 85_017_624_576;
    const QWEN38_ROW_BYTES: u64 = 156_893_184;
    let plan = qwen38_resource_plan(BackendKind::Cuda);
    let ResourceAmount::Known(weights) = plan.resident_authorization.device_bytes else {
        panic!("Qwen3.8 CUDA residency must be known");
    };
    let floor = |rows, automatic_rows| CudaStateFloor {
        retained_rows: rows,
        staged_rows: rows.min(8),
        snapshot_rows: 0,
        per_sequence_bytes: QWEN38_ROW_BYTES,
        automatic_rows,
    };
    let admit = |floor, capacity| {
        admit_cuda_state_floor(
            ModelVariant::Qwen3827BFp8,
            &plan,
            floor,
            GIB,
            ResourceAmount::Known(capacity),
        )
    };

    // The decode-width floor fits an H100 and joins the load authorization.
    let authorization = admit(floor(8, true), H100_BYTES).unwrap();
    assert_eq!(
        authorization.device_bytes,
        ResourceAmount::Known(weights + 16 * QWEN38_ROW_BYTES + GIB)
    );
    assert_eq!(authorization.host_bytes, plan.load_authorization.host_bytes);

    // The pre-fix automatic rows, now explicit, can never fit.
    let legacy = CudaStateFloor {
        retained_rows: 316,
        staged_rows: 316,
        snapshot_rows: 0,
        per_sequence_bytes: QWEN38_ROW_BYTES,
        automatic_rows: false,
    };
    let message = admit(legacy, H100_BYTES).unwrap_err().to_string();
    assert!(message.contains("before loading Qwen3.8"), "{message}");
    assert!(
        message.contains("minimum_state_bytes=99156492288"),
        "{message}"
    );
    assert!(message.contains("IZWI_MAX_RETAINED_SEQUENCES"), "{message}");

    // Automatic rows on a GPU smaller than the weights name the staged knob.
    let message = admit(floor(8, true), 24 * GIB).unwrap_err().to_string();
    assert!(
        message.contains("IZWI_MAX_STAGED_TRANSACTIONS"),
        "{message}"
    );
    assert!(
        message.contains("per_sequence_state_bytes=156893184"),
        "{message}"
    );

    // Unknown capacity defers to live admission, still pricing the floor.
    let authorization = admit_cuda_state_floor(
        ModelVariant::Qwen3827BFp8,
        &plan,
        floor(8, true),
        GIB,
        ResourceAmount::Unknown,
    )
    .unwrap();
    assert_eq!(
        authorization.device_bytes,
        ResourceAmount::Known(weights + 16 * QWEN38_ROW_BYTES + GIB)
    );

    // With prefix reuse engaged, the floor also holds the snapshot budget.
    let shared = CudaStateFloor {
        snapshot_rows: 8,
        ..floor(8, true)
    };
    let authorization = admit(shared, H100_BYTES).unwrap();
    assert_eq!(
        authorization.device_bytes,
        ResourceAmount::Known(weights + 24 * QWEN38_ROW_BYTES + GIB)
    );
}

#[test]
fn qwen38_cuda_q8_0_plan_fits_reported_l40s_available_bytes() {
    const REPORTED_L40S_AVAILABLE_BYTES: u64 = 47_196_667_904;

    let plan = qwen38_resource_plan(BackendKind::Cuda);
    let ResourceAmount::Known(device_load_peak_bytes) = plan.load_authorization.device_bytes
    else {
        panic!("Qwen3.8 CUDA device load peak must be known")
    };
    // Every FP8 and BF16 source element at two bytes, plus the portable
    // conversion scratch: the old fully expanded CUDA plan.
    let old_expanded_load_peak_bytes =
        (24_699_207_680 + 3_082_220_272) * 2 + QWEN38_PORTABLE_CONVERSION_SCRATCH_BYTES;
    assert_eq!(
        plan.load_authorization.host_bytes,
        ResourceAmount::Known(8_589_934_592)
    );
    assert_eq!(device_load_peak_bytes, 31_854_039_040);
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
fn metal_oom_retry_judges_hybrid_checkpoints_by_their_backend_load_peak() {
    use super::retry_load_peak_bytes;
    const GIB: u64 = 1024 * 1024 * 1024;
    let _env_guard = crate::env_test_lock()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let performance = crate::performance::PerformanceConfig::default();
    let peak = |variant, backend| retry_load_peak_bytes(variant, backend, &performance, None);

    // Qwen3.8 on Metal (raw block FP8, about 29.9 GiB at load; 51.9 GiB
    // expanded to F16) sits far below the 80 GiB catalog figure that used
    // to veto every retry on a 96 GB Mac.
    let qwen38_metal = peak(ModelVariant::Qwen3827BFp8, BackendKind::Metal);
    assert_eq!(
        qwen38_metal,
        qwen38_representation_memory_estimate(BackendKind::Metal, &performance).load_peak_bytes
    );
    assert!(qwen38_metal < 54 * GIB);
    assert!(peak(ModelVariant::Qwen36Moe35BA3BFp8, BackendKind::Metal) < 140 * GIB);
    // Variants without a representation plan keep the catalog estimate.
    assert_eq!(
        peak(ModelVariant::Kokoro82M, BackendKind::Metal),
        model_memory_estimate(ModelVariant::Kokoro82M).load_peak_bytes
    );
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
fn parakeet_load_route_publishes_long_form_predictor_beside_retained_state() {
    use crate::kv::v2::{InvocationStateBackingKindV2, InvocationWorkspaceBindingV2};
    use crate::models::architectures::parakeet::asr::{
        parakeet_physical_state_spec, PARAKEET_PREDICTOR_STATE_DOMAIN,
    };
    use crate::runtime::adapters::{
        LoadedModelBundleDraft, LoadedStatePublication, RuntimeAdapterRegistry,
        StreamingRequirements,
    };

    let model = ModelInstanceId::new(91);
    let draft = LoadedModelBundleDraft::build(
        &RuntimeAdapterRegistry::built_in(),
        ExecutionGroupId::new(7),
        model,
        crate::model::ModelVariant::ParakeetTdt06BV3,
        BackendKind::Cpu,
    )
    .unwrap();
    let contracts = draft.execution_contracts(CapabilityKind::Asr).unwrap();
    let graphs = contracts
        .iter()
        .map(|contract| contract.stages.as_ref())
        .collect::<Vec<_>>();
    let spec = parakeet_physical_state_spec(&graphs).unwrap();
    let invocation = spec
        .invocation
        .as_ref()
        .expect("long-form predictor contract");
    let mut core = crate::engine::EngineCore::new(crate::engine::EngineCoreConfig {
        backend: BackendKind::Cpu,
        ..crate::engine::EngineCoreConfig::default()
    })
    .unwrap();
    let retained = core
        .load_retained_tensor_state(model, spec.retained.as_ref().unwrap(), 1)
        .unwrap();
    let retained_uses = super::parakeet_retained_uses(&contracts).unwrap();
    let retained: crate::kv::v2::RetainedStateRuntimeV2 = retained.into();
    super::validate_physical_publication_backing(
        &spec.descriptor,
        &contracts,
        Some(&retained),
        &retained_uses,
    )
    .unwrap();

    // Only the two atomic long-form graphs (with and without transport
    // output) allocate typed state: one predictor slot each.
    let allocations =
        plan_invocation_allocations(&spec.descriptor, invocation, &contracts).unwrap();
    assert_eq!(allocations.len(), 2);
    let mut bindings = Vec::new();
    for allocation in allocations {
        assert_eq!(allocation.key.domain, PARAKEET_PREDICTOR_STATE_DOMAIN);
        assert_eq!(allocation.slot_count, 1);
        // Four [1, 640] LSTM tensors plus one [1, 1, 640] output, F32.
        assert_eq!(allocation.domain.maximum_bytes().unwrap(), 5 * 640 * 4);
        let backing = core
            .resolve_and_load_invocation_workspace(
                model,
                allocation.adapter_instance,
                allocation.key.stage_graph,
                allocation.key.stage,
                invocation,
                &allocation.domain,
                allocation.slot_count,
            )
            .unwrap();
        bindings.push(InvocationWorkspaceBindingV2 {
            key: allocation.key,
            backing,
        });
    }
    let bundle = draft
        .seal(HashMap::from([(
            CapabilityKind::Asr,
            LoadedStatePublication::PhysicalV2 {
                descriptor: spec.descriptor,
                retained: Some(retained),
                retained_uses,
                invocation_workspace: crate::kv::v2::InvocationWorkspaceRuntimeV2::new(bindings)
                    .unwrap(),
            },
        )]))
        .unwrap();

    let long_form = StreamingRequirements::NONE.with_asr_long_form(true);
    let contract = bundle
        .contract_for_streaming(CapabilityKind::Asr, long_form)
        .unwrap();
    let stage = contract
        .adapter_binding()
        .unwrap()
        .stage_for_work(&crate::engine::WorkUnit::AtomicJob {
            kind: "diarization.transcribe".into(),
        })
        .unwrap()
        .id;
    let binding = bundle
        .capability_binding_for_streaming(CapabilityKind::Asr, long_form)
        .unwrap();
    binding
        .state
        .validate_against(BackendKind::Cpu, &binding.execution)
        .unwrap();
    let mut leases = binding
        .state
        .lease_complete_invocation_workspace_set(stage)
        .unwrap();
    let predictor = leases
        .lease_exact_kind_mut(InvocationStateBackingKindV2::Tensor)
        .unwrap()
        .typed_mut::<crate::engine::InvocationTensorLease>()
        .unwrap();
    assert_eq!(predictor.domain(), PARAKEET_PREDICTOR_STATE_DOMAIN);
    leases.release().unwrap();

    // Ordinary requests keep the retained sequence graph and lease nothing.
    let normal = bundle
        .contract_for_streaming(CapabilityKind::Asr, StreamingRequirements::NONE)
        .unwrap();
    assert_eq!(normal.execution_profile.mode, ExecutionMode::Sequence);
    let normal_binding = bundle
        .capability_binding_for_streaming(CapabilityKind::Asr, StreamingRequirements::NONE)
        .unwrap();
    for stage in normal.stages.iter() {
        assert!(normal_binding
            .state
            .lease_complete_invocation_workspace_set(stage.id)
            .unwrap()
            .domains()
            .next()
            .is_none());
    }
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
        metal_device_load_peak_bytes: 10_712_372_164,
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
            metal_device_load_peak_bytes: 100,
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
        metal_device_load_peak_bytes: 19_836_076_996,
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
        metal_device_load_peak_bytes: 10_712_372_164,
        ..cpu_memory
    };
    // On Metal the unified authorization books what the GPU allocates; the
    // host staging (1.6 GB) is checked against host memory instead.
    let metal = fish_s2_resource_plan(BackendKind::Metal, half_memory).unwrap();
    assert_eq!(
        metal.load_authorization.unified_bytes,
        ResourceAmount::Known(half_memory.metal_device_load_peak_bytes)
    );
    assert_eq!(
        metal.host_staging_bytes,
        half_memory.load_peak_bytes - half_memory.metal_device_load_peak_bytes
    );
    assert_eq!(cpu.host_staging_bytes, 0);
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
fn portable_context_reserve_is_the_configured_reserve_when_residency_is_the_catalog_figure() {
    const GIB: u64 = 1024 * 1024 * 1024;

    assert_eq!(
        portable_context_reserve_bytes(ModelVariant::Kokoro82M, GIB),
        GIB
    );
}

/// Truth reconciliation (ADM-8): growth beyond the authorization plus the
/// tolerance fails the load; growth inside it, or matched by leases
/// reserved over the same interval (the load's own state, a concurrent
/// request), passes and reports the observed bytes.
#[test]
fn observed_growth_beyond_the_authorization_fails_the_load() {
    const MIB: u64 = 1024 * 1024;
    let variant = ModelVariant::WhisperLargeV3Turbo;
    let sample = |observed, reserved| DeviceUsageSample { observed, reserved };
    let authorized = 4 * 1024 * MIB;
    let tolerance = materialization_tolerance_bytes(authorized);
    assert_eq!(tolerance, 512 * MIB);
    assert_eq!(materialization_tolerance_bytes(MIB), 256 * MIB);

    let before = sample(100 * MIB, authorized);
    let within = sample(100 * MIB + authorized + tolerance, authorized);
    assert_eq!(
        reconcile_observed_growth(variant, "weight load", before, within, authorized).unwrap(),
        authorized + tolerance
    );
    let over = sample(within.observed + 1, authorized);
    let error = reconcile_observed_growth(variant, "weight load", before, over, authorized)
        .unwrap_err()
        .to_string();
    assert!(error.contains("under-prices"), "{error}");

    // State leases reserved between the samples authorize their growth.
    let state_before = sample(0, 1_000 * MIB);
    let state_after = sample(1_200 * MIB, 2_000 * MIB);
    assert!(reconcile_observed_growth(
        variant,
        "state allocation",
        state_before,
        state_after,
        0
    )
    .is_ok());
    let state_unauthorized = sample(1_200 * MIB, 1_000 * MIB);
    assert!(reconcile_observed_growth(
        variant,
        "state allocation",
        state_before,
        state_unauthorized,
        0
    )
    .is_err());
    // A release elsewhere (observed shrinking) never fails a load.
    assert_eq!(
        reconcile_observed_growth(variant, "weight load", sample(MIB, 0), sample(0, 0), 0)
            .unwrap(),
        0
    );
}

/// The Granite shape from a live Metal run: after the weights land, live
/// headroom (0.8 GB) is below the 1 GiB reserve while planning headroom
/// holds gigabytes. The minimum state must still be admitted (against
/// planning); live memory caps only growth above it.
#[test]
fn asr_state_floor_is_admitted_against_planning_when_live_undercounts() {
    const GIB: u64 = 1024 * 1024 * 1024;
    let reserve = GIB;
    let (floor, growth) =
        state_fit_budgets(5 * GIB, ResourceAmount::Known(796_545_836), reserve);
    assert_eq!(floor, 4 * GIB);
    assert_eq!(growth, 0);
    assert!(
        10_485_760 <= floor,
        "a one-page minimum fits the floor budget"
    );
    let (floor, growth) = state_fit_budgets(5 * GIB, ResourceAmount::Known(3 * GIB), reserve);
    assert_eq!((floor, growth), (4 * GIB, 2 * GIB));
    // Live never raises growth above the planning floor.
    let (floor, growth) = state_fit_budgets(2 * GIB, ResourceAmount::Known(9 * GIB), reserve);
    assert_eq!((floor, growth), (GIB, GIB));
    assert_eq!(
        state_fit_budgets(2 * GIB, ResourceAmount::Unknown, reserve),
        (GIB, GIB)
    );
}

/// ASR rows: chat's automatic rows (320 on an H100) collapse to 8, never
/// below one decode batch; operator rows stay as configured.
/// On CPU and Metal the engine rows stay at the portable width (TTS and
/// ASR fit their state on them), and a chat model stages one step's
/// launch at its own certified decode width (EXE-F7).
#[test]
fn portable_chat_models_stage_rows_at_their_decode_width() {
    use crate::runtime::adapters::LoadedModelBundleDraft;
    let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(8, 1).unwrap();
    let staged = |runtime: &RuntimeService, variant: ModelVariant| {
        let draft = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(992),
            ModelInstanceId::new(1),
            variant,
            BackendKind::Cpu,
        )
        .unwrap();
        draft.seal_chat_workspace(8_192).unwrap();
        runtime
            .model_lifecycle
            .portable_chat_staged_rows(&draft, BackendKind::Cpu)
            .unwrap()
    };
    let automatic = RuntimeService::new(EngineConfig {
        backend: BackendPreference::Cpu,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(automatic.core_engine.config().max_staged_transactions, 2);
    assert_eq!(
        staged(&automatic, ModelVariant::Lfm2512BInstructGguf),
        Some(8)
    );
    assert_eq!(staged(&automatic, ModelVariant::Qwen306B), Some(2));

    let explicit = RuntimeService::new(EngineConfig {
        backend: BackendPreference::Cpu,
        max_staged_transactions: 3,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(staged(&explicit, ModelVariant::Lfm2512BInstructGguf), None);
}

#[test]
fn asr_state_rows_come_from_the_asr_policy_not_chat_rows() {
    assert_eq!(asr_state_rows(true, 320, 4), (8, 4));
    assert_eq!(asr_state_rows(true, 320, 16), (16, 16));
    assert_eq!(asr_state_rows(true, 4, 4), (4, 4));
    assert_eq!(asr_state_rows(false, 320, 32), (320, 32));
    assert_eq!(asr_state_rows(true, 0, 0), (1, 1));
}

/// The request floor joins the load authorization before eviction: the
/// authorization is the larger of the weight peak and resident weights
/// plus the floor.
#[test]
fn asr_request_floor_joins_the_load_authorization() {
    let estimate = ModelMemoryEstimate {
        load_peak_bytes: 1_000,
        resident_bytes: 800,
    };
    let plan = model_resource_plan(BackendKind::Metal, estimate);
    let unified = |authorization: ResourceVector| authorization.unified_bytes;
    assert_eq!(
        unified(admit_asr_request_floor(&plan, BackendKind::Metal, 100).unwrap()),
        ResourceAmount::Known(1_000)
    );
    assert_eq!(
        unified(admit_asr_request_floor(&plan, BackendKind::Metal, 500).unwrap()),
        ResourceAmount::Known(1_300)
    );
    let cuda = model_resource_plan(BackendKind::Cuda, estimate);
    let authorization = admit_asr_request_floor(&cuda, BackendKind::Cuda, 500).unwrap();
    assert_eq!(authorization.device_bytes, ResourceAmount::Known(1_300));
    assert_eq!(authorization.host_bytes, cuda.load_authorization.host_bytes);
}

/// On a 16 GB M1 Pro-like Metal budget (0.9 × the 12.7 GB working set),
/// a state fit that spends everything it may see after the model's
/// weights and the held request envelope still leaves one full-chunk
/// envelope free once the hold is released (ADM-1, ADM-4).
#[test]
fn state_fits_leave_one_request_envelope_on_a_16_gb_metal_budget() {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const BUDGET: u64 = 11_441_804_076;
    let unified = |bytes| ResourceVector {
        unified_bytes: ResourceAmount::Known(bytes),
        ..ResourceVector::zero()
    };
    // Metal-allocated resident weights from the residency plans.
    for (variant, resident_gib) in [
        (ModelVariant::Qwen3Asr06BGguf, 1.54),
        (ModelVariant::Qwen3Asr17BGguf, 3.30),
        (ModelVariant::Lfm25Audio15BGguf, 2.08),
        (ModelVariant::GraniteSpeech412BPlus, 5.87),
    ] {
        let authority = Arc::new(ResourceAuthority::new_advisory_shared_host_unified(
            Arc::new(VectorCapacityProvider {
                snapshot: PhysicalCapacitySnapshot {
                    capacity: unified(BUDGET),
                    available: unified(BUDGET),
                    source: CapacitySource::Test,
                },
            }),
        ));
        let resident = (resident_gib * GIB) as u64;
        let _model = authority
            .reserve_with_initial_materialized(
                ReservationOwner::new(ReservationClass::Model, "weights"),
                unified(resident),
                unified(resident),
            )
            .unwrap();
        let envelope = super::super::asr_envelope::deferred_request_envelope(
            variant,
            BackendKind::Metal,
            1,
        )
        .unwrap()
        .unwrap()
        .total_bytes()
        .unwrap();
        let hold = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Model, "envelope"),
                unified(envelope),
            )
            .unwrap();
        let ResourceAmount::Known(fit) = authority
            .state_fit_headroom_bytes(BackendKind::Metal)
            .unwrap()
        else {
            panic!("known headroom");
        };
        let _state = authority
            .reserve(
                ReservationOwner::new(ReservationClass::Cache, "greedy-state"),
                unified(fit),
            )
            .unwrap();
        drop(hold);
        let ResourceAmount::Known(left) = authority
            .planning_headroom_bytes(BackendKind::Metal)
            .unwrap()
        else {
            panic!("known headroom");
        };
        assert!(left >= envelope, "{variant}: {left} < {envelope}");
        assert!(fit > 0, "{variant}");
    }
}

/// A loading thread's precision reload waits on its gate until the load
/// task answers, and sees the task's refusal.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_precision_reload_gate_blocks_until_the_load_task_answers() {
    let (mut gate, mut requests) = precision_reload_gate(candle_core::Device::Cpu);
    let loader = tokio::task::spawn_blocking(move || {
        let admitted = gate("text", candle_core::DType::F32);
        let refused = gate("text", candle_core::DType::F32);
        (admitted, refused)
    });
    for answer in [Ok(()), Err(Error::ModelLoadError("no room".into()))] {
        let request = requests.recv().await.expect("reload request");
        assert_eq!(
            (request.component.as_str(), request.dtype),
            ("text", candle_core::DType::F32)
        );
        request.reply.send(answer).unwrap();
    }
    let (admitted, refused) = loader.await.unwrap();
    admitted.unwrap();
    assert!(refused.unwrap_err().to_string().contains("no room"));
    // A gate whose load task is gone refuses instead of blocking.
    let (mut gate, requests) = precision_reload_gate(candle_core::Device::Cpu);
    drop(requests);
    let orphaned = tokio::task::spawn_blocking(move || gate("text", candle_core::DType::F32));
    assert!(orphaned.await.unwrap().is_err());
}

/// A failed probe's reload takes a second reservation: the loading
/// slot's authorization grows from the lowered load's peak to the reload
/// plan's before the reload allocates, and the reload plan then prices
/// the load. A reload that cannot fit is refused with the shortfall and
/// leaves the authorization as admitted.
#[tokio::test]
async fn a_failed_probe_grows_the_load_authorization_to_the_reload_plan() {
    let models_dir = std::env::temp_dir().join(format!("izwi-precision-reload-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&models_dir).unwrap();
    let runtime = RuntimeService::new(EngineConfig {
        models_dir: models_dir.clone(),
        backend: BackendPreference::Cpu,
        ..Default::default()
    })
    .unwrap();
    let plan = |load_peak_bytes, resident_bytes| {
        model_resource_plan(
            BackendKind::Cpu,
            ModelMemoryEstimate {
                load_peak_bytes,
                resident_bytes,
            },
        )
    };
    let (lowered, reload, too_large) = (plan(256, 128), plan(512, 384), plan(2048, 1024));
    let authority = vector_authority(all_memory_capacity(1024));
    let variant = ModelVariant::Qwen3ForcedAligner06B;
    let lease = authority
        .reserve(
            ReservationOwner::new(ReservationClass::Model, "precision-reload"),
            lowered.load_authorization,
        )
        .unwrap();
    let lifecycle = &runtime.model_lifecycle;
    lifecycle.install_loading_slot(variant, lease).unwrap();
    let (reply, _) = oneshot::channel();
    let request = PrecisionReloadRequest {
        component: "text".into(),
        dtype: candle_core::DType::F32,
        reply,
    };
    let grown = lifecycle
        .grow_load_authorization_for_reload(variant, &lowered, reload, &request)
        .await
        .unwrap();
    assert_eq!(grown, reload);
    assert_eq!(authority.snapshot().reserved, reload.load_authorization);
    let error = lifecycle
        .grow_load_authorization_for_reload(variant, &reload, too_large, &request)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        error.contains(&format!(
            "Cannot reload {variant} text at F32 after its load-time probe failed"
        )),
        "{error}"
    );
    assert_eq!(authority.snapshot().reserved, reload.load_authorization);
    assert!(lifecycle.remove_resident_slot(variant));
    std::fs::remove_dir_all(models_dir).unwrap();
}

/// The weight lease records the smaller of the estimate and the observed
/// growth: priced bytes the loader never allocated stay pending.
#[tokio::test]
async fn observed_weight_growth_is_recorded_as_materialized() {
    let models_dir =
        std::env::temp_dir().join(format!("izwi-observed-weights-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&models_dir).unwrap();
    let runtime = RuntimeService::new(EngineConfig {
        models_dir: models_dir.clone(),
        backend: BackendPreference::Cpu,
        ..Default::default()
    })
    .unwrap();
    let authority = vector_authority(all_memory_capacity(1024));
    let bytes = |n| ResourceVector {
        unified_bytes: ResourceAmount::Known(n),
        ..ResourceVector::zero()
    };
    let variant = ModelVariant::WhisperLargeV3Turbo;
    let lease = authority
        .reserve(
            ReservationOwner::new(ReservationClass::Model, "observed-weights"),
            bytes(256),
        )
        .unwrap();
    runtime
        .model_lifecycle
        .install_loading_slot(variant, lease)
        .unwrap();
    runtime
        .model_lifecycle
        .finalize_slot_materialization_with_pending(
            variant,
            bytes(200),
            ResourceVector::zero(),
            Some((BackendKind::Metal, 150)),
        )
        .unwrap();
    let snapshot = authority.snapshot();
    assert_eq!(snapshot.reserved, bytes(200));
    assert_eq!(snapshot.materialized, bytes(150));
    assert_eq!(snapshot.pending, bytes(50));
    assert!(runtime.model_lifecycle.remove_resident_slot(variant));
    std::fs::remove_dir_all(models_dir).unwrap();
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
        .finalize_slot_materialization_with_pending(variant, bytes(64), bytes(16), None)
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

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn idle_reaper_unloads_idle_transient_models_but_preserves_pins() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
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
    // Loaded on request without a pin: kept while idle.
    runtime
        .model_manager
        .mark_loaded(ModelVariant::Kokoro82M)
        .await;
    runtime
        .model_lifecycle
        .keep_model(ModelVariant::Kokoro82M)
        .await;

    let stale = now_unix_millis().saturating_sub(10_000);
    {
        let mut last_used = runtime.model_lifecycle.model_last_used.lock().await;
        last_used.insert(ModelVariant::GraniteSpeech412BPlus, stale);
        last_used.insert(ModelVariant::WhisperLargeV3Turbo, stale);
        last_used.insert(ModelVariant::Kokoro82M, stale);
    }

    let reaped = runtime.reap_idle_models().await;
    assert_eq!(reaped, vec![ModelVariant::GraniteSpeech412BPlus]);

    let residents = runtime.model_manager.resident_variants().await;
    assert!(!residents.contains(&ModelVariant::GraniteSpeech412BPlus));
    assert!(residents.contains(&ModelVariant::WhisperLargeV3Turbo));
    assert!(residents.contains(&ModelVariant::Kokoro82M));
    std::fs::remove_dir_all(models_dir).unwrap();
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn pins_apply_to_loaded_models_and_unpinning_keeps_them_loaded() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let models_dir = std::env::temp_dir().join(format!("izwi-runtime-pin-test-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&models_dir).unwrap();
    let runtime = RuntimeService::new(EngineConfig {
        models_dir: models_dir.clone(),
        backend: BackendPreference::Cpu,
        ..EngineConfig::default()
    })
    .unwrap();
    let variant = ModelVariant::Kokoro82M;

    assert!(!runtime.pin_resident_model(variant).await, "not loaded");
    runtime.model_manager.mark_loaded(variant).await;
    assert!(runtime.pin_resident_model(variant).await);
    assert!(runtime.pinned_model_variants().await.contains(&variant));

    assert!(runtime.unpin_resident_model(variant).await);
    assert!(!runtime.pinned_model_variants().await.contains(&variant));
    assert!(runtime
        .model_lifecycle
        .kept_model_variants()
        .await
        .contains(&variant));
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
    let message = error.to_string();
    assert!(
        message.contains("Granite Speech 4.1 2B Plus is loaded explicitly or serving requests: unload it to make room")
            && message.contains("IZWI_MAX_LOADED_MODELS"),
        "unexpected error: {error}"
    );
    assert_eq!(error.kind(), crate::error::ErrorKind::CapacityExhausted);

    assert!(runtime
        .model_manager
        .resident_variants()
        .await
        .contains(&ModelVariant::GraniteSpeech412BPlus));
    std::fs::remove_dir_all(models_dir).unwrap();
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn residency_budget_evicts_granite_before_loading_another_asr_model() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
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

/// SPK-F7: a realtime voice session holds its models through
/// `SessionModelLeases`. The models are never pinned, a held model is not
/// evicted to make room for another, and it becomes evictable once the
/// session drops its leases.
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn session_model_leases_protect_without_pinning() {
    // Serialize the load transaction's residency reservation on the
    // process-global authority.
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let models_dir = std::env::temp_dir().join(format!(
        "izwi-runtime-session-lease-test-{}",
        Uuid::new_v4()
    ));
    std::fs::create_dir_all(&models_dir).unwrap();
    let runtime = RuntimeService::new(EngineConfig {
        models_dir: models_dir.clone(),
        backend: BackendPreference::Cpu,
        max_loaded_models: Some(1),
        ..EngineConfig::default()
    })
    .unwrap();
    let variant = ModelVariant::Kokoro82M;
    let (_authority, resource_lease) = isolated_resource_lease("session-lease");
    runtime
        .model_lifecycle
        .install_loading_slot(variant, resource_lease)
        .expect("loading slot");
    runtime
        .model_lifecycle
        .finalize_slot_materialization(variant, one_byte_host_reservation())
        .expect("materialized slot");
    runtime.model_manager.mark_loaded(variant).await;
    runtime
        .model_lifecycle
        .mark_slot_ready(variant)
        .expect("ready slot");

    let leases = super::SessionModelLeases::default();
    leases.hold(&runtime, variant).await.expect("first turn");
    leases
        .clone()
        .hold(&runtime, variant)
        .await
        .expect("next turn");
    assert!(!runtime.pinned_model_variants().await.contains(&variant));
    assert_eq!(runtime.model_manager.active_residency_leases(variant), 1);

    let error = runtime
        .ensure_model_budget_before_load(ModelVariant::WhisperLargeV3Turbo)
        .await
        .expect_err("a model held by a session is not evicted");
    assert!(
        error
            .to_string()
            .contains("Kokoro 82M is loaded explicitly or serving requests: unload it"),
        "{error}"
    );
    assert_eq!(
        runtime.model_manager.resident_variants().await,
        vec![variant]
    );

    drop(leases);
    assert_eq!(runtime.model_manager.active_residency_leases(variant), 0);
    runtime
        .ensure_model_budget_before_load(ModelVariant::WhisperLargeV3Turbo)
        .await
        .expect("an unpinned model is evictable after the session");
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

/// A unified-memory machine whose provider samples live memory and
/// observes device usage (Metal's `currentAllocatedSize`).
#[derive(Debug)]
struct ObservedUnifiedProvider {
    capacity: u64,
    available: u64,
    observed: u64,
}

impl PhysicalCapacityProvider for ObservedUnifiedProvider {
    fn snapshot(&self) -> PhysicalCapacitySnapshot {
        let unified = |bytes| ResourceVector {
            unified_bytes: ResourceAmount::Known(bytes),
            ..ResourceVector::zero()
        };
        PhysicalCapacitySnapshot {
            capacity: unified(self.capacity),
            available: unified(self.available),
            source: CapacitySource::Test,
        }
    }

    fn observed_device_usage_bytes(&self) -> Option<u64> {
        Some(self.observed)
    }
}

/// The final Metal run's Qwen3-TTS-0.6B-Base load (about 11.4 GB of
/// planning capacity, 10 GB live, 2.67 GB of weights) reserves its one
/// request slot's envelope: CustomVoice's live envelope plus a 30 s
/// reference encode on the shipped encoder. The reference term was
/// 47 GB and refused the load with the generic capacity error; a
/// refusal now names what the envelope needs and what is available.
#[test]
fn qwen_base_request_envelope_fits_beside_its_weights() {
    use crate::models::architectures::qwen3::tts::speech_tokenizer::{
        reference_encode_workspace_upper_bound_bytes, EncoderConfig,
        QWEN3_TTS_12HZ_ENCODER_CONFIG_JSON,
    };
    let encoder: EncoderConfig =
        serde_json::from_str(QWEN3_TTS_12HZ_ENCODER_CONFIG_JSON).unwrap();
    let reference =
        reference_encode_workspace_upper_bound_bytes(&encoder, 24_000, 1_920, 30, 4).unwrap();
    // CustomVoice's envelope from the same run (no reference encode).
    let envelope = super::super::asr_envelope::AsrRequestEnvelope {
        host_bytes: 46_377_456,
        accelerator_bytes: 471_076_352 + reference,
    };
    /// Metal as the server builds it: a configured planning budget
    /// beside a live sample.
    #[derive(Debug)]
    struct BudgetedUnifiedProvider;
    impl PhysicalCapacityProvider for BudgetedUnifiedProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            let unified = |bytes| ResourceVector {
                unified_bytes: ResourceAmount::Known(bytes),
                ..ResourceVector::zero()
            };
            PhysicalCapacitySnapshot {
                capacity: unified(16_000_000_000),
                available: unified(10_000_000_000),
                source: CapacitySource::MetalWorkingSet,
            }
        }
        fn configured_budget(&self) -> Option<ResourceVector> {
            Some(ResourceVector {
                host_bytes: ResourceAmount::Known(11_400_000_000),
                unified_bytes: ResourceAmount::Known(11_400_000_000),
                ..ResourceVector::zero()
            })
        }
        fn observed_device_usage_bytes(&self) -> Option<u64> {
            Some(2_668_085_248)
        }
    }
    let authority = Arc::new(ResourceAuthority::new_advisory_shared_host_unified(
        Arc::new(BudgetedUnifiedProvider),
    ));
    let weights = ResourceVector {
        unified_bytes: ResourceAmount::Known(2_667_798_788),
        ..ResourceVector::zero()
    };
    let _weights = authority
        .reserve_with_initial_materialized(
            ReservationOwner::new(ReservationClass::Model, "weights"),
            weights,
            weights,
        )
        .unwrap();
    authority
        .reserve(
            ReservationOwner::new(ReservationClass::Model, "envelope"),
            super::request_envelope_resources(BackendKind::Metal, envelope).unwrap(),
        )
        .unwrap_or_else(|error| panic!("reference {reference} bytes: {error}"));

    let refusal = super::request_envelope_shortfall(
        ModelVariant::Qwen3Tts12Hz06BBase,
        BackendKind::Metal,
        envelope,
        1_000_000_000,
    )
    .unwrap();
    assert!(matches!(refusal, crate::error::Error::Overloaded(_)));
    let message = refusal.to_string();
    assert!(message.contains("needs "), "{message}");
    assert!(message.contains("1.00 GB available"), "{message}");
}

const GIB: u64 = 1024 * 1024 * 1024;
/// What load admission charged Qwen3-TTS 0.6B's weights at 0adabf16.
const TTS_CHARGED_WEIGHT_BYTES: u64 = 2_493_807_108;

/// An engine core beside weights the ledger charged at
/// [`TTS_CHARGED_WEIGHT_BYTES`] while the device observer reports
/// `observed` bytes in use, and the lease that holds those weights.
fn tts_state_core(
    variant: ModelVariant,
    capacity: u64,
    available: u64,
    observed: u64,
) -> crate::error::Result<(crate::engine::EngineCore, ResourceLease)> {
    let authority = Arc::new(ResourceAuthority::new_advisory_shared_host_unified(
        Arc::new(ObservedUnifiedProvider {
            capacity,
            available,
            observed,
        }),
    ));
    let weights = ResourceVector {
        unified_bytes: ResourceAmount::Known(TTS_CHARGED_WEIGHT_BYTES),
        ..ResourceVector::zero()
    };
    let weights = authority.reserve_with_initial_materialized(
        ReservationOwner::new(ReservationClass::Model, variant.to_string()),
        weights,
        weights,
    )?;
    assert!(
        authority.untracked_device_bytes(BackendKind::Cpu).unwrap() > 0,
        "observed usage must exceed what the ledger materialized"
    );
    let mut config = crate::engine::EngineCoreConfig {
        backend: BackendKind::Cpu,
        max_retained_sequences: 1,
        max_staged_transactions: 1,
        ..Default::default()
    };
    config.portable_context_auto = true;
    let core = crate::engine::EngineCore::new_with_managed_authority_for_test(config, authority)?;
    Ok((core, weights))
}

/// Load TTS retained state through the engine's real load path the way
/// `load_model` does for an unobserved (non-ASR) load, beside weights the
/// ledger charged at [`TTS_CHARGED_WEIGHT_BYTES`] while the device
/// observer reports `observed` bytes in use.
fn load_tts_retained_state(
    variant: ModelVariant,
    contract: &crate::kv::v2::InferenceStateContract,
    maximum_tokens: usize,
    capacity: u64,
    available: u64,
    observed: u64,
) -> crate::error::Result<u64> {
    let (mut core, _weights) = tts_state_core(variant, capacity, available, observed)?;
    let runtime = core.load_managed_model_state(
        ModelInstanceId::new(4_101),
        contract,
        Some(maximum_tokens),
        super::tts_state_minimum_tokens(variant),
        crate::engine::StateFitReserve::for_observed_load(false),
        0,
    )?;
    Ok(runtime.logical_token_reach())
}

/// A2: portable chat arenas are fully resident from load, so fitting the
/// model's whole advertised context pinned gigabytes for an idle model.
/// The automatic fit is capped; a zero cap restores the full context.
#[test]
fn portable_auto_chat_context_is_capped() -> crate::error::Result<()> {
    for (cap, expected) in [(4_096_usize, 4_096_u64), (0, 16_384)] {
        let authority = Arc::new(ResourceAuthority::new_advisory_shared_host_unified(
            Arc::new(ObservedUnifiedProvider {
                capacity: 64 * GIB,
                available: 64 * GIB,
                observed: 0,
            }),
        ));
        let mut config = crate::engine::EngineCoreConfig {
            backend: BackendKind::Cpu,
            max_retained_sequences: 1,
            max_staged_transactions: 1,
            portable_auto_context_cap_tokens: cap,
            ..Default::default()
        };
        config.portable_context_auto = true;
        let mut core =
            crate::engine::EngineCore::new_with_managed_authority_for_test(config, authority)?;
        let runtime = core
            .load_managed_model_cache(
                ModelInstanceId::new(4_102),
                &crate::kv::InferenceStateCapability::Managed(crate::kv::test_contract()),
                Some(16_384),
            )?
            .expect("managed runtime");
        assert_eq!(runtime.logical_token_reach(), expected, "cap {cap}");
    }
    Ok(())
}

/// A multi-user `max_num_seqs` plans the capped context once per sequence
/// (at most 8,192 tokens each), within what memory holds; one sequence keeps
/// the single-user pool. Each request may still use the model's context.
#[test]
fn portable_multi_user_chat_pool_plans_a_context_per_sequence() -> crate::error::Result<()> {
    for (max_num_seqs, cap, expected) in [
        (None, 4_096_usize, 4_096_u64),
        (Some(1), 4_096, 4_096),
        (Some(4), 4_096, 16_384),
        (Some(4), 32_768, 32_768),
        (Some(8), 32_768, 65_536),
    ] {
        let authority = Arc::new(ResourceAuthority::new_advisory_shared_host_unified(
            Arc::new(ObservedUnifiedProvider {
                capacity: 64 * GIB,
                available: 64 * GIB,
                observed: 0,
            }),
        ));
        let mut config = crate::engine::EngineCoreConfig {
            backend: BackendKind::Cpu,
            max_retained_sequences: max_num_seqs.unwrap_or(1),
            max_staged_transactions: 1,
            portable_auto_context_cap_tokens: cap,
            max_num_seqs,
            ..Default::default()
        };
        config.portable_context_auto = true;
        let mut core =
            crate::engine::EngineCore::new_with_managed_authority_for_test(config, authority)?;
        let runtime = core
            .load_managed_model_cache(
                ModelInstanceId::new(4_103),
                &crate::kv::InferenceStateCapability::Managed(crate::kv::test_contract()),
                Some(131_072),
            )?
            .expect("managed runtime");
        assert_eq!(
            runtime.logical_token_reach(),
            expected,
            "{max_num_seqs:?} cap {cap}"
        );
    }
    Ok(())
}

/// R-1: Qwen3-TTS retained state collapsed to one 64-token page on
/// Metal, cutting every CustomVoice request at 54 frames and refusing
/// every clone. The fit must reach every request's prompt plus its
/// automatic budget whatever live memory reads, spend live growth above
/// it without reserving the unobserved weight under-pricing a second
/// time, and refuse with a 503 naming the shortfall when planning
/// headroom cannot hold the floor.
#[test]
fn qwen3_tts_effective_context_covers_prompt_and_frame_budget_under_low_live_headroom() {
    use crate::models::architectures::qwen3::tts::{
        qwen3_tts_auto_max_frames_for_text, qwen3_tts_auto_max_output_frames,
        qwen3_tts_inference_state_contract, qwen3_tts_retained_talker_contract,
        tests::cache_test_config,
    };
    let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
    let config = cache_test_config();
    let full = qwen3_tts_inference_state_contract(
        &config,
        candle_core::DType::F32,
        candle_core::DType::F32,
        crate::models::shared::attention::paged::default_kv_page_size(),
    )
    .unwrap();
    let contract =
        qwen3_tts_retained_talker_contract(&full, &config, candle_core::DType::F32).unwrap();
    // A clone prompt with the longest reference: role and codec-prefix
    // tokens plus 320 reference codes.
    let max_clone_prefill = 9 + 320 + 1;
    let paragraph = "Good morning. Today we are testing a speech system that reads \
            longer passages aloud, and we want to hear whether every sentence comes out \
            clearly without the voice drifting, rushing, or stopping before the final word.";
    assert!(qwen3_tts_auto_max_frames_for_text(paragraph) < qwen3_tts_auto_max_output_frames());
    let required = (max_clone_prefill + qwen3_tts_auto_max_output_frames()) as u64;
    let maximum = 3_072;
    let budget = 11 * GIB + 440 * 1024 * 1024;
    let observed = 6_000_000_000;

    // Live memory is far below planning and the observed usage leaves
    // nothing above the measured untracked bytes: the floor holds.
    let reach = load_tts_retained_state(variant, &contract, maximum, budget, GIB / 2, observed)
        .unwrap();
    assert!(reach >= required, "effective context {reach} < {required}");
    assert!(reach < maximum as u64, "live memory must still cap growth");

    // With 3 GiB live, the configured reserve leaves growth for the full
    // context; the unobserved weight under-pricing is not reserved again.
    let reach = load_tts_retained_state(variant, &contract, maximum, budget, 3 * GIB, observed)
        .unwrap();
    assert_eq!(reach, maximum as u64);

    // A floor planning headroom cannot hold refuses the load (503).
    let tight = TTS_CHARGED_WEIGHT_BYTES + GIB + 64 * 1024 * 1024;
    match load_tts_retained_state(variant, &contract, maximum, tight, tight, observed) {
        Err(Error::Overloaded(message)) => {
            assert!(
                message.starts_with("not enough memory for this model's minimum")
                    && message.contains(" safety reserve, ")
                    && !message.contains("planning_headroom="),
                "{message}"
            )
        }
        other => panic!("expected a capacity refusal, got {other:?}"),
    }
}

/// S3: a CPU or Metal load must find its residents, its request envelope
/// and the state its fits keep free before the weights load, or its load
/// peak, whichever is larger. Unknown residency leaves the peak alone.
#[test]
fn portable_preload_holds_the_state_floor_beside_the_residents() {
    let plan = |domain: fn(u64) -> ResourceVector| super::ModelResourcePlan {
        resident_authorization: domain(10_000),
        load_authorization: domain(12_000),
        ..model_resource_plan(
            BackendKind::Metal,
            ModelMemoryEstimate {
                load_peak_bytes: 0,
                resident_bytes: 0,
            },
        )
    };
    let unified = |bytes| ResourceVector {
        unified_bytes: ResourceAmount::Known(bytes),
        ..ResourceVector::zero()
    };
    let host = |bytes| ResourceVector {
        host_bytes: ResourceAmount::Known(bytes),
        ..ResourceVector::zero()
    };
    let metal = plan(unified);
    assert_eq!(
        super::portable_preload_bytes(&metal, BackendKind::Metal, 12_000, 0, 1_000).unwrap(),
        12_000
    );
    assert_eq!(
        super::portable_preload_bytes(&metal, BackendKind::Metal, 12_000, 500, 3_000).unwrap(),
        13_500
    );
    let cpu = plan(host);
    assert_eq!(
        super::portable_preload_bytes(&cpu, BackendKind::Cpu, 12_000, 0, 2_500).unwrap(),
        12_500
    );
    // The Metal plan has no host residency to stand on.
    assert_eq!(
        super::portable_preload_bytes(&metal, BackendKind::Cpu, 12_000, 0, 9_999).unwrap(),
        12_000
    );
}

/// S2: Qwen3-TTS binds one retained arena for its Tts and StreamingTts
/// capabilities. The second capability gets the runtime the first fit
/// bound; refitting it against the headroom that arena had already taken
/// refused its 2,524-token floor after the weights had loaded. The first
/// fit leaves the predictor workspaces loaded after it free.
#[test]
fn a_second_capability_reuses_the_bound_retained_runtime() {
    use crate::models::architectures::qwen3::tts::{
        qwen3_tts_inference_state_contract, qwen3_tts_retained_talker_contract,
        tests::cache_test_config,
    };
    let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
    let config = cache_test_config();
    let full = qwen3_tts_inference_state_contract(
        &config,
        candle_core::DType::F32,
        candle_core::DType::F32,
        crate::models::shared::attention::paged::default_kv_page_size(),
    )
    .unwrap();
    let contract =
        qwen3_tts_retained_talker_contract(&full, &config, candle_core::DType::F32).unwrap();
    let budget = 11 * GIB + 440 * 1024 * 1024;
    let load = |core: &mut crate::engine::EngineCore, later_minimum_bytes| {
        core.load_managed_model_state(
            ModelInstanceId::new(4_102),
            &contract,
            Some(1 << 20),
            super::tts_state_minimum_tokens(variant),
            crate::engine::StateFitReserve::for_observed_load(false),
            later_minimum_bytes,
        )
    };

    // Live memory lets the first fit grow to the planning floor budget.
    let (mut core, _weights) = tts_state_core(variant, budget, budget, 6_000_000_000).unwrap();
    let first = load(&mut core, 0).unwrap();
    let second = load(&mut core, 0).unwrap();
    assert!(Arc::ptr_eq(&first, &second));

    // Holding back a later workspace makes the first fit smaller.
    let (mut core, _weights) = tts_state_core(variant, budget, budget, 6_000_000_000).unwrap();
    let held = load(&mut core, 2 * GIB).unwrap();
    assert!(
        held.logical_token_reach() < first.logical_token_reach(),
        "{} !< {}",
        held.logical_token_reach(),
        first.logical_token_reach()
    );
}

/// Phase 0 state-ceiling contract: every TTS family whose load fits
/// retained state guarantees one full request (its longest prompt plus
/// its largest automatic output budget), and families sized elsewhere
/// ask for no floor.
#[test]
fn tts_state_floor_covers_one_full_request_per_family() {
    use crate::catalog::ModelFamily;
    use crate::models::architectures::qwen3::tts::qwen3_tts_auto_max_output_frames;
    use crate::models::architectures::vibevoice::tts::vibevoice_tts_auto_max_frames_for_text;
    let flood = "word ".repeat(20_000);
    let floor = |variant| super::tts_state_minimum_tokens(variant) as usize;
    for variant in ModelVariant::all().iter().copied() {
        if !variant.is_enabled() || !variant.is_tts() {
            continue;
        }
        let required = match variant.family() {
            // 320 reference codes plus role and codec-prefix tokens.
            ModelFamily::Qwen3Tts => 330 + qwen3_tts_auto_max_output_frames(),
            // A 12 s voice reference (90 latents) plus its markers.
            ModelFamily::VibeVoiceTts => 96 + vibevoice_tts_auto_max_frames_for_text(&flood),
            // A preset voice prompt plus CUDA's frame maximum.
            ModelFamily::VoxtralTts => 256 + ModelVariant::VOXTRAL_TTS_CUDA_MAX_OUTPUT_FRAMES,
            _ => {
                assert_eq!(floor(variant), 0, "{variant}");
                continue;
            }
        };
        assert!(
            floor(variant) >= required,
            "{variant}: floor {} < {required}",
            floor(variant)
        );
    }
}

/// R-1 for VibeVoice: every request returned 400 against a 64-token
/// context. Both CFG domains must reach a prompt plus the largest
/// automatic budget.
#[test]
fn vibevoice_tts_effective_context_covers_prompt_and_frame_budget_under_low_live_headroom() {
    use crate::models::architectures::vibevoice::tts::vibevoice_tts_auto_max_frames_for_text;
    let variant = ModelVariant::VibeVoice15BTts;
    let contract =
        crate::models::architectures::vibevoice::vibevoice_tts_retained_test_contract(
            candle_core::DType::F32,
        );
    // System prompt, a 12 s voice reference (90 latents plus markers)
    // and the target text that fills the largest automatic budget.
    let longest = "word ".repeat(1_000);
    let required = (30 + 96 + 200 + vibevoice_tts_auto_max_frames_for_text(&longest)) as u64;
    let maximum = 1_536;
    let budget = 11 * GIB + 440 * 1024 * 1024;
    let reach =
        load_tts_retained_state(variant, &contract, maximum, budget, GIB / 2, 6_000_000_000)
            .unwrap();
    assert!(reach >= required, "effective context {reach} < {required}");
    assert!(reach < maximum as u64, "live memory must still cap growth");
}

fn shortfall(
    required_gb: u64,
    ledger_gb: u64,
    live_gb: Option<u64>,
    held_gb: u64,
    unevictable: Vec<ModelVariant>,
) -> super::admission::LoadShortfall {
    const GB: u64 = 1_000_000_000;
    super::admission::LoadShortfall {
        backend: BackendKind::Metal,
        admission: crate::engine::MetalLoadAdmission::Advisory,
        required_bytes: required_gb * GB,
        ledger_headroom: ledger_gb * GB,
        ledger_ceiling: Some(11 * GB),
        live_headroom: live_gb.map(|gb| gb * GB),
        held_by_residents: held_gb * GB,
        unevictable,
        state_floor_bytes: 0,
    }
}

/// A refusal names the pinned or busy models only when unloading them
/// could cover the gap, and otherwise says what is actually short.
#[test]
fn load_refusals_name_their_cause() {
    let variant = ModelVariant::Qwen354BGguf;
    let pinned = vec![ModelVariant::Kokoro82M, ModelVariant::WhisperLargeV3Turbo];

    let message = shortfall(6, 2, Some(3), 5, pinned.clone()).message(variant);
    assert_eq!(
        message,
        "Cannot load Qwen3.5 4B GGUF: needs 6.00 GB, 2.00 GB available; Kokoro 82M and \
         Whisper Large v3 Turbo are loaded explicitly or serving requests and hold 5.00 GB: \
         unload one of them to make room"
    );

    // Unloading both would leave 2 + 1 GB: not enough, so the budget is named.
    let message = shortfall(12, 2, None, 1, pinned).message(variant);
    assert!(
        message.ends_with(
            "; the model is larger than the 11.00 GB memory budget \
             (IZWI_METAL_MEMORY_BUDGET_BYTES, by default 90% of the GPU working set)"
        ),
        "{message}"
    );

    // The GPU working set binds below the ledger.
    let message = shortfall(6, 9, Some(4), 0, Vec::new()).message(variant);
    assert!(
        message.starts_with("Cannot load Qwen3.5 4B GGUF: needs 6.00 GB, 4.00 GB available")
            && message.ends_with("GPU working set (this Mac's limit for GPU memory) has free"),
        "{message}"
    );

    // Strict host gating blames other apps and names the switch.
    let mut strict = shortfall(6, 9, Some(4), 0, Vec::new());
    strict.admission = crate::engine::MetalLoadAdmission::Strict;
    let message = strict.message(variant);
    assert!(
        message.contains("other apps are using it (IZWI_METAL_LOAD_ADMISSION=strict)"),
        "{message}"
    );

    // Requests and state hold the budget: nothing to unload.
    let message = shortfall(6, 2, None, 0, Vec::new()).message(variant);
    assert!(
        message.ends_with("held by running requests and model state: retry when they finish"),
        "{message}"
    );

    // S3: the state a load keeps free beside its weights is named (Fish S2
    // Pro on a 16 GB Mac: 10.74 GB of weights and a 1.07 GB reserve).
    let mut fish = shortfall(12, 12, Some(11), 0, Vec::new());
    fish.state_floor_bytes = 1_070_000_000;
    let message = fish.message(ModelVariant::FishAudioS2Pro);
    assert!(
        message.starts_with(
            "Cannot load Fish Audio S2 Pro: needs 12.00 GB, 11.00 GB available (of which \
             1.07 GB is the state and safety reserve the model needs beside its weights)"
        ),
        "{message}"
    );
}

/// Host staging (Fish's staged copies beside its GPU peak) gates a Metal
/// load only under strict admission; advisory and off admission check the
/// GPU peak alone.
#[test]
fn host_staging_gates_only_strict_metal_loads() {
    use crate::engine::MetalLoadAdmission;
    let gated = |admission| super::admission::gated_load_bytes(admission, 10_712, 1_595);
    assert_eq!(gated(MetalLoadAdmission::Strict), 12_307);
    assert_eq!(gated(MetalLoadAdmission::Advisory), 10_712);
    assert_eq!(gated(MetalLoadAdmission::Off), 10_712);
}

#[test]
fn unevictable_residents_are_pinned_or_busy_and_never_the_request() {
    let residents = [
        ModelVariant::Kokoro82M,
        ModelVariant::WhisperLargeV3Turbo,
        ModelVariant::Qwen354BGguf,
        ModelVariant::ParakeetTdt06BV3,
    ];
    let active = HashSet::from([ModelVariant::WhisperLargeV3Turbo]);
    let pinned = HashSet::from([ModelVariant::Kokoro82M, ModelVariant::Qwen354BGguf]);
    assert_eq!(
        super::admission::unevictable_residents(
            &residents,
            ModelVariant::Qwen354BGguf,
            &active,
            &pinned,
        ),
        vec![ModelVariant::Kokoro82M, ModelVariant::WhisperLargeV3Turbo]
    );
}
