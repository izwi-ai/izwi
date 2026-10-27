use super::*;
use crate::engine::{EngineCore, EngineCoreConfig};
use crate::kv::InferenceStateCapability;
use crate::runtime::adapters::ExecutionTargetKind;
use crate::runtime::adapters::SequenceExecutionMode;

fn chat_adapter_metadata(variant: ModelVariant) -> AdapterMetadata {
    AdapterMetadata {
        id: "test.chat.adapter",
        capability: CapabilityKind::Chat,
        model_variant: variant,
        streaming_mode: StreamingMode::Chunked,
        execution_target: ExecutionTargetKind::TokenEngine,
        sequence_execution: SequenceExecutionMode::Always,
        state_requirement: InferenceStateRequirement::Retained,
    }
}

fn install_test_preparation_seals(
    draft: &LoadedModelBundleDraft,
    backend: BackendKind,
    max_batch_size: usize,
) {
    for execution in draft.capabilities.values() {
        execution
            .install_test_preparation_seal(backend, max_batch_size)
            .unwrap_or_else(|error| {
                panic!(
                    "failed to install test preparation seal for {} {:?}: {error}",
                    execution.metadata().model_variant,
                    execution.metadata().capability
                )
            });
    }
}

#[test]
fn stateful_chat_models_with_batch_paths_select_the_continuous_stage() {
    let continuous = ContinuousPhysicalChatAdapterFactory;
    let scalar = ScalarExecutionAdapterFactory;

    for variant in [
        ModelVariant::Qwen3827BFp8,
        ModelVariant::Qwen3508BGguf,
        ModelVariant::Qwen36Moe35BA3BFp8,
        ModelVariant::Lfm2512BInstructGguf,
    ] {
        let metadata = chat_adapter_metadata(variant);
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            assert!(
                continuous.supports(metadata, backend),
                "{variant}/{backend:?}"
            );
            assert!(!scalar.supports(metadata, backend), "{variant}/{backend:?}");
        }
    }
}

#[test]
fn managed_qwen_publication_seals_a_physical_v2_runtime() {
    let registry = RuntimeAdapterRegistry::built_in();
    let model_instance = ModelInstanceId::new(8);
    let state_contract = crate::kv::test_contract();
    let capability = InferenceStateCapability::Managed(state_contract.clone());
    let mut core = EngineCore::new(EngineCoreConfig {
        backend: BackendKind::Cpu,
        max_blocks: 4,
        block_size: 32,
        ..EngineCoreConfig::default()
    })
    .unwrap();
    let physical = core
        .load_managed_model_cache(model_instance, &capability, None)
        .unwrap()
        .expect("physical managed runtime");
    let draft = LoadedModelBundleDraft::build(
        &registry,
        ExecutionGroupId::new(3),
        model_instance,
        ModelVariant::Qwen306B,
        BackendKind::Cpu,
    )
    .unwrap();
    install_test_preparation_seals(&draft, BackendKind::Cpu, 1);
    let bundle = draft
        .seal(HashMap::from([(
            CapabilityKind::Chat,
            LoadedStatePublication::ManagedV2 {
                contract: state_contract,
                physical: physical.clone(),
            },
        )]))
        .unwrap();

    let binding = bundle
        .capability_binding_for_streaming(CapabilityKind::Chat, StreamingRequirements::NONE)
        .unwrap();
    assert_eq!(binding.execution.model_instance_id, model_instance);
    let runtime = binding.state;
    assert!(runtime.managed_kv_runtime().is_some());
    let inactive = CapabilityStateRuntimeV2::managed(
        ManagedCapabilityRuntimeV2::seal(
            BackendKind::Cpu,
            &binding.execution,
            runtime.descriptor.clone(),
            physical.clone(),
            RetainedStateUseV2::Inactive,
        )
        .unwrap(),
    );
    assert!(inactive.managed_kv_runtime().is_none());
    assert_ne!(inactive.id, runtime.id);
    assert_eq!(
        runtime
            .managed_kv_runtime()
            .expect("managed backing")
            .state_plan_v2()
            .id,
        physical.state_plan_v2().id
    );
    assert_eq!(
        bundle
            .contract(CapabilityKind::Chat, false)
            .unwrap()
            .execution_profile
            .cache_mode,
        CacheMode::ExternalPaged
    );
    let mut request = crate::engine::EngineCoreRequest::chat(vec![])
        .with_model_variant(ModelVariant::Qwen306B);
    request.bind_model_instance(model_instance).unwrap();
    request.bind_execution_adapter(binding.execution).unwrap();
    request
        .bind_v2_state_runtime(runtime.clone(), runtime.state_fingerprint, BackendKind::Cpu)
        .unwrap();
    assert!(request.v2_state_runtime().is_some());
}

#[test]
fn sealed_adapter_bundle_does_not_pin_an_idle_physical_generation() {
    let registry = RuntimeAdapterRegistry::built_in();
    let model_instance = ModelInstanceId::new(80);
    let state_contract = crate::kv::test_contract();
    let mut core = EngineCore::new(EngineCoreConfig {
        backend: BackendKind::Cpu,
        max_blocks: 4,
        block_size: 32,
        ..EngineCoreConfig::default()
    })
    .unwrap();
    let physical = core
        .load_managed_model_cache(
            model_instance,
            &InferenceStateCapability::Managed(state_contract.clone()),
            None,
        )
        .unwrap()
        .expect("physical managed runtime");
    let draft = LoadedModelBundleDraft::build(
        &registry,
        ExecutionGroupId::new(30),
        model_instance,
        ModelVariant::Qwen306B,
        BackendKind::Cpu,
    )
    .unwrap();
    install_test_preparation_seals(&draft, BackendKind::Cpu, 1);
    let bundle = draft
        .seal(HashMap::from([(
            CapabilityKind::Chat,
            LoadedStatePublication::ManagedV2 {
                contract: state_contract,
                physical: physical.clone(),
            },
        )]))
        .unwrap();
    let binding = bundle
        .capability_binding_for_streaming(CapabilityKind::Chat, StreamingRequirements::NONE)
        .unwrap();
    let runtime = binding.state;
    drop(physical);

    assert!(core
        .unload_managed_model_cache(model_instance)
        .expect("idle physical generation unload"));
    assert!(runtime.managed_kv_runtime().is_none());
    assert!(runtime
        .validate_against(BackendKind::Cpu, &binding.execution)
        .is_err());
}

#[test]
fn qwen_asr_direct_binding_fails_closed_before_model_geometry_is_sealed() {
    let registry = RuntimeAdapterRegistry::built_in();
    let model_instance = ModelInstanceId::new(81);
    let state_contract = crate::kv::test_contract();
    let capability = InferenceStateCapability::Managed(state_contract.clone());
    let mut core = EngineCore::new(EngineCoreConfig {
        max_blocks: 4,
        block_size: 32,
        ..EngineCoreConfig::default()
    })
    .unwrap();
    let physical = core
        .load_managed_model_cache(model_instance, &capability, None)
        .unwrap()
        .expect("physical managed runtime");
    let error = LoadedModelBundle::bind_with_state_publications(
        &registry,
        ExecutionGroupId::new(31),
        model_instance,
        ModelVariant::Qwen3Asr06BGguf,
        BackendKind::Cpu,
        HashMap::from([(
            CapabilityKind::Asr,
            LoadedStatePublication::ManagedV2 {
                contract: state_contract,
                physical,
            },
        )]),
    )
    .expect_err("Qwen3 ASR direct binding must not invent unloaded model geometry");
    assert!(error.to_string().contains("audio preparation is sealed"));
}

#[test]
fn stateful_qwen_capability_fails_closed_without_physical_publication() {
    let registry = RuntimeAdapterRegistry::built_in();
    let draft = LoadedModelBundleDraft::build(
        &registry,
        ExecutionGroupId::new(3),
        ModelInstanceId::new(10),
        ModelVariant::Qwen306B,
        BackendKind::Cpu,
    )
    .unwrap();
    install_test_preparation_seals(&draft, BackendKind::Cpu, 1);
    let error = draft
        .seal(HashMap::new())
        .expect_err("stateful chat must not seal without physical state");
    assert!(error
        .to_string()
        .contains("requires an explicit load-sealed ABI-v2 state publication"));
}

#[test]
fn lfm2_chat_fails_closed_without_managed_state_publication() {
    let registry = RuntimeAdapterRegistry::built_in();
    let draft = LoadedModelBundleDraft::build(
        &registry,
        ExecutionGroupId::new(3),
        ModelInstanceId::new(11),
        ModelVariant::Lfm2512BInstructGguf,
        BackendKind::Cpu,
    )
    .unwrap();
    install_test_preparation_seals(&draft, BackendKind::Cpu, 1);
    let error = draft
        .seal(HashMap::new())
        .expect_err("LFM2 chat must not seal without managed physical state");
    assert!(error
        .to_string()
        .contains("requires an explicit load-sealed ABI-v2 state publication"));
}

#[test]
fn v2_state_publication_is_preserved_without_legacy_fallback() {
    let registry = RuntimeAdapterRegistry::built_in();
    let variant = ModelVariant::Qwen3TtsTokenizer12Hz;
    let capability = CapabilityKind::Tokenizer;
    let compatibility = LoadedModelBundle::bind(
        &registry,
        ExecutionGroupId::new(3),
        ModelInstanceId::new(10),
        variant,
        BackendKind::Cpu,
    )
    .unwrap();
    let offline = compatibility
        .contract_for_streaming(capability, StreamingRequirements::NONE)
        .unwrap()
        .stages;
    let transport = compatibility
        .contract_for_streaming(capability, StreamingRequirements::transport_only())
        .unwrap()
        .stages;
    let descriptor =
        crate::kv::v2::CapabilityStateDescriptorV2::stateless_for_stage_graphs_test(&[
            &offline, &transport,
        ]);
    let bundle = LoadedModelBundle::bind_with_state_publications(
        &registry,
        ExecutionGroupId::new(3),
        ModelInstanceId::new(10),
        variant,
        BackendKind::Cpu,
        HashMap::from([(capability, LoadedStatePublication::V2(descriptor.clone()))]),
    )
    .unwrap();

    let binding = bundle
        .capability_binding_for_streaming(capability, StreamingRequirements::NONE)
        .unwrap();
    let runtime = binding.state;
    assert_eq!(runtime.descriptor, descriptor);
    runtime
        .validate_against(BackendKind::Cpu, &binding.execution)
        .unwrap();
}

#[test]
fn v2_runtime_reuses_one_seal_for_identical_selectable_stage_graphs() {
    let registry = RuntimeAdapterRegistry::built_in();
    let variant = ModelVariant::Qwen3TtsTokenizer12Hz;
    let capability = CapabilityKind::Tokenizer;
    let compatibility = LoadedModelBundle::bind(
        &registry,
        ExecutionGroupId::new(3),
        ModelInstanceId::new(11),
        variant,
        BackendKind::Cpu,
    )
    .unwrap();
    let offline = compatibility
        .contract_for_streaming(capability, StreamingRequirements::NONE)
        .unwrap();
    let descriptor =
        crate::kv::v2::CapabilityStateDescriptorV2::stateless_for_stages_test(&offline.stages);

    let bundle = LoadedModelBundle::bind_with_state_publications(
        &registry,
        ExecutionGroupId::new(3),
        ModelInstanceId::new(11),
        variant,
        BackendKind::Cpu,
        HashMap::from([(capability, LoadedStatePublication::V2(descriptor))]),
    )
    .unwrap();
    let offline = bundle
        .capability_binding_for_streaming(capability, StreamingRequirements::NONE)
        .unwrap();
    let transport = bundle
        .capability_binding_for_streaming(capability, StreamingRequirements::transport_only())
        .unwrap();
    assert_eq!(
        offline.state.state_fingerprint,
        transport.state.state_fingerprint
    );
}

#[test]
fn stateful_capability_cannot_publish_ready_without_physical_backing() {
    let registry = RuntimeAdapterRegistry::built_in();
    let variant = ModelVariant::Nemotron35AsrStreaming06B;
    let error = LoadedModelBundle::bind(
        &registry,
        ExecutionGroupId::new(3),
        ModelInstanceId::new(14),
        variant,
        BackendKind::Cpu,
    )
    .expect_err("Nemotron realtime must fail closed without physical publication");
    assert!(error
        .to_string()
        .contains("requires an explicit load-sealed ABI-v2 state publication"));
}

#[test]
fn nemotron_realtime_factory_authors_split_retained_stages() {
    let draft = LoadedModelBundleDraft::build(
        &RuntimeAdapterRegistry::built_in_with_execution_limits(4, 4).unwrap(),
        ExecutionGroupId::new(3),
        ModelInstanceId::new(15),
        ModelVariant::Nemotron35AsrStreaming06B,
        BackendKind::Cpu,
    )
    .unwrap();
    let contracts = draft
        .execution_contracts(CapabilityKind::RealtimeAsr)
        .unwrap();

    assert!(!contracts.is_empty());
    for contract in contracts {
        assert_eq!(contract.adapter_abi_revision, NEMOTRON_REALTIME_ADAPTER_ABI);
        assert_eq!(contract.execution_profile.mode, ExecutionMode::Realtime);
        assert_eq!(contract.execution_profile.cache_mode, CacheMode::None);
        assert_eq!(
            contract.execution_profile.decode_batch,
            NativeBatchMode::Continuous
        );
        assert_eq!(contract.execution_profile.max_batch_size, 4);
        assert_eq!(contract.stages.len(), 3);
        assert_eq!(contract.stages[0].selector, StageWorkSelector::Atomic);
        assert_eq!(
            contract.stages[1].selector,
            StageWorkSelector::RealtimePreparation
        );
        assert_eq!(contract.stages[1].batch_mode, NativeBatchMode::None);
        assert_eq!(
            contract.stages[2].selector,
            StageWorkSelector::RealtimeDecodeContinuation
        );
        assert_eq!(contract.stages[2].batch_mode, NativeBatchMode::Continuous);
        assert_eq!(contract.stages[2].max_batch_size, 4);
        assert_eq!(
            contract.stages[0].max_workspace_bytes,
            crate::models::architectures::nemotron::asr::NEMOTRON_REALTIME_STAGE_WORKSPACE_BYTES
        );
        assert_eq!(
            contract.stages[2].max_workspace_bytes,
            4 * crate::models::architectures::nemotron::asr::NEMOTRON_REALTIME_STAGE_WORKSPACE_BYTES
        );
    }
}

#[test]
fn voxtral_realtime_factory_is_published_only_for_its_exact_capability() {
    assert!(built_in_loaded_adapter_factories()
        .iter()
        .any(|factory| factory.id() == "builtin.voxtral_realtime.physical_paged"));
}

#[test]
fn voxtral_realtime_sealed_graph_is_exact_on_supported_backends() {
    let metadata = AdapterMetadata {
        id: "test.voxtral.realtime",
        capability: CapabilityKind::RealtimeAsr,
        model_variant: ModelVariant::VoxtralMini4BRealtime2602,
        streaming_mode: StreamingMode::Realtime,
        execution_target: ExecutionTargetKind::RealtimeRunner,
        sequence_execution: SequenceExecutionMode::None,
        state_requirement: InferenceStateRequirement::RetainedAndInvocation,
    };
    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        let adapter = VoxtralRealtimeExecutionAdapter::new(
            ExecutionGroupId::new(7),
            ModelInstanceId::new(9),
            metadata,
            backend,
            3,
        );
        adapter
            .install_preparation_seal(
                crate::models::architectures::voxtral::realtime::VoxtralRealtimePreparationStageSeal {
                    max_source_samples: 32_000,
                    max_work_units: 32_000,
                    max_materialized_tensor_elements_per_row: 1_000_000,
                    max_workspace_bytes: 4_000_000,
                },
            )
            .expect("finite test seal");
        let contract = adapter
            .contract(StreamingRequirements::native(true))
            .expect("sealed Voxtral graph");
        assert_eq!(contract.adapter_abi_revision, VOXTRAL_REALTIME_ADAPTER_ABI);
        assert_eq!(contract.execution_profile.max_batch_size, 3);
        assert_eq!(contract.stages.len(), 4);
        assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::Static);
        assert_eq!(contract.stages[0].shape_policy, StageShapePolicy::Padded);
        assert_eq!(contract.stages[0].max_workspace_bytes, 12_000_000);
        assert_eq!(contract.stages[1].max_batch_size, 1);
        assert_eq!(contract.stages[2].batch_mode, NativeBatchMode::Continuous);
        assert_eq!(contract.stages[2].shape_policy, StageShapePolicy::Ragged);
        assert_eq!(contract.stages[3].max_workspace_bytes, 0);
        crate::models::architectures::voxtral::authenticate_voxtral_realtime_execution_binding(
            &contract.adapter_binding().expect("binding"),
        )
        .expect("exact Voxtral authentication");
    }
}

#[test]
fn managed_v2_publication_requires_a_physical_runtime_before_ready() {
    let registry = RuntimeAdapterRegistry::built_in();
    let draft = LoadedModelBundleDraft::build(
        &registry,
        ExecutionGroupId::new(3),
        ModelInstanceId::new(12),
        ModelVariant::Qwen306B,
        BackendKind::Cpu,
    )
    .unwrap();
    install_test_preparation_seals(&draft, BackendKind::Cpu, 1);
    let stages = draft
        .execution_contracts(CapabilityKind::Chat)
        .unwrap()
        .remove(0)
        .stages;
    let descriptor = CapabilityStateDescriptorV2::managed_for_stages_test(
        crate::kv::v2::test_contract(),
        &stages,
    );

    let error = draft
        .seal(HashMap::from([(
            CapabilityKind::Chat,
            LoadedStatePublication::V2(descriptor),
        )]))
        .expect_err("managed v2 metadata alone must not publish Ready");
    assert!(error.to_string().contains("requires physical backing"));
}

#[test]
fn stateless_v2_rejects_execution_that_declares_model_owned_cache() {
    let registry = RuntimeAdapterRegistry::built_in();
    let variant = ModelVariant::Qwen3508BGguf;
    let draft = LoadedModelBundleDraft::build(
        &registry,
        ExecutionGroupId::new(3),
        ModelInstanceId::new(13),
        variant,
        BackendKind::Cpu,
    )
    .unwrap();
    install_test_preparation_seals(&draft, BackendKind::Cpu, 1);
    let stages = draft
        .execution_contracts(CapabilityKind::Chat)
        .unwrap()
        .remove(0)
        .stages;
    let descriptor = CapabilityStateDescriptorV2::stateless_for_stages_test(&stages);

    let error = draft
        .seal(HashMap::from([(
            CapabilityKind::Chat,
            LoadedStatePublication::V2(descriptor),
        )]))
        .expect_err("stateless v2 must not relabel a retained sequence cache");
    assert!(
        error
            .to_string()
            .contains("requiring retained inference state"),
        "unexpected error: {error}"
    );
}

#[test]
fn sealed_state_runtime_is_scoped_to_one_capability_descriptor() {
    let registry = RuntimeAdapterRegistry::built_in();
    let bundle = LoadedModelBundle::bind(
        &registry,
        ExecutionGroupId::new(4),
        ModelInstanceId::new(11),
        ModelVariant::Kokoro82M,
        BackendKind::Cpu,
    )
    .unwrap();

    let tts = bundle
        .capability_binding_for_streaming(CapabilityKind::Tts, StreamingRequirements::NONE)
        .unwrap();
    let streaming_tts = bundle
        .capability_binding_for_streaming(
            CapabilityKind::StreamingTts,
            StreamingRequirements::NONE,
        )
        .unwrap();
    assert_ne!(
        tts.execution.capability_id,
        streaming_tts.execution.capability_id
    );
    assert_ne!(tts.state.id, streaming_tts.state.id);
}

#[test]
fn cache_truth_for_an_unregistered_capability_is_rejected() {
    let state_publications = HashMap::from([(
        CapabilityKind::Asr,
        LoadedStatePublication::V2(CapabilityStateDescriptorV2::stateless_for_test()),
    )]);

    let error = LoadedModelBundle::bind_with_state_publications(
        &RuntimeAdapterRegistry::built_in(),
        ExecutionGroupId::new(5),
        ModelInstanceId::new(12),
        ModelVariant::Qwen306B,
        BackendKind::Cpu,
        state_publications,
    )
    .expect_err("an unmatched cache declaration must fail closed");

    assert!(error.to_string().contains("unregistered capabilities"));
    assert!(error.to_string().contains("asr"));
}

#[derive(Debug)]
struct TestStaticTtsFactory {
    id: &'static str,
    model_variant: ModelVariant,
}

impl LoadedExecutionAdapterFactory for TestStaticTtsFactory {
    fn id(&self) -> &'static str {
        self.id
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::Static
    }

    fn supports(&self, metadata: AdapterMetadata, _backend_kind: BackendKind) -> bool {
        metadata.model_variant == self.model_variant
            && matches!(
                metadata.capability,
                CapabilityKind::Tts | CapabilityKind::StreamingTts
            )
    }

    fn create(
        &self,
        context: LoadedAdapterFactoryContext,
        metadata: AdapterMetadata,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        Ok(Arc::new(StaticTtsExecutionAdapter::new(
            context.execution_group_id,
            context.model_instance_id,
            metadata,
            context.backend_kind,
            context.max_tensor_batch_size,
            context.request_parallelism,
        )))
    }
}

#[derive(Debug)]
struct TestScalarFactory {
    excluded_model_variant: ModelVariant,
}

impl LoadedExecutionAdapterFactory for TestScalarFactory {
    fn id(&self) -> &'static str {
        "test.scalar"
    }

    fn batch_mode(&self) -> NativeBatchMode {
        NativeBatchMode::None
    }

    fn supports(&self, metadata: AdapterMetadata, backend_kind: BackendKind) -> bool {
        metadata.model_variant != self.excluded_model_variant
            && ScalarExecutionAdapterFactory.supports(metadata, backend_kind)
    }

    fn create(
        &self,
        context: LoadedAdapterFactoryContext,
        metadata: AdapterMetadata,
    ) -> Result<Arc<dyn LoadedExecutionAdapter>> {
        Ok(Arc::new(ScalarExecutionAdapter::new(
            context.execution_group_id,
            context.model_instance_id,
            metadata,
            context.backend_kind,
            context.request_parallelism,
        )))
    }
}

#[test]
fn every_supported_model_capability_authors_an_exact_width_one_contract() {
    let registry = RuntimeAdapterRegistry::built_in();

    for (index, variant) in ModelVariant::all().iter().copied().enumerate() {
        let instance = ModelInstanceId::new(index as u64 + 1);
        let draft = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(7),
            instance,
            variant,
            BackendKind::Cpu,
        )
        .unwrap_or_else(|error| panic!("failed to build {variant}: {error}"));
        install_test_preparation_seals(&draft, BackendKind::Cpu, 1);
        let metadata = registry.capabilities_for(variant);

        assert_eq!(draft.capabilities.len(), metadata.len(), "{variant}");
        for metadata in metadata {
            let execution = draft.capabilities.get(&metadata.capability).unwrap();
            let contract = execution
                .contract(StreamingRequirements::NONE)
                .unwrap_or_else(|error| panic!("failed to contract {variant}: {error}"));
            assert_eq!(contract.execution_group_id, ExecutionGroupId::new(7));
            assert_eq!(contract.model_instance_id, instance);
            assert_eq!(contract.metadata, metadata);
            let factory = registry
                .loaded_adapter_factory(metadata, BackendKind::Cpu)
                .unwrap();
            if factory.id() == "builtin.scalar" {
                assert_eq!(contract.adapter_abi_revision, SCALAR_ADAPTER_ABI);
                assert_eq!(contract.stages.len(), 1);
                assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::None);
            } else if factory.id() == LoadedAdapterKind::FishS2Tts.factory_id() {
                // The factory implements native batching, but an unqualified
                // scalar load must publish the scalar graph truthfully.
                assert_ne!(contract.adapter_abi_revision, SCALAR_ADAPTER_ABI);
                assert_eq!(contract.execution_profile.max_batch_size, 1);
                assert!(contract
                    .stages
                    .iter()
                    .all(|stage| stage.batch_mode == NativeBatchMode::None));
            } else {
                assert_ne!(contract.adapter_abi_revision, SCALAR_ADAPTER_ABI);
                assert!(
                    contract
                        .stages
                        .iter()
                        .any(|stage| stage.batch_mode == factory.batch_mode()),
                    "{variant} {:?} factory {} did not publish {:?}",
                    metadata.capability,
                    factory.id(),
                    factory.batch_mode()
                );
            }
            assert!(contract
                .stages
                .iter()
                .all(|stage| stage.max_batch_size == 1));
            assert_eq!(contract.execution_profile.max_batch_size, 1);
            assert!(contract.execution_profile.resolved_from_loaded_model);

            let transport = execution
                .contract(StreamingRequirements::transport_only())
                .unwrap_or_else(|error| {
                    panic!("failed transport-only contract for {variant}: {error}")
                });
            assert_eq!(transport.metadata, metadata);

            let native_streaming = execution.contract(StreamingRequirements::native(true));
            if metadata.streaming_mode == StreamingMode::None {
                assert!(
                    native_streaming.is_err(),
                    "{variant} {:?} unexpectedly advertised native streaming",
                    metadata.capability
                );
            } else {
                native_streaming.unwrap_or_else(|error| {
                    panic!("failed native-streaming contract for {variant}: {error}")
                });
            }
        }
    }
}

#[test]
fn every_stateful_capability_fails_closed_without_physical_publication() {
    let registry = RuntimeAdapterRegistry::built_in();
    let backends = [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda];

    for backend in backends {
        for (index, variant) in ModelVariant::all().iter().copied().enumerate() {
            let draft = LoadedModelBundleDraft::build(
                &registry,
                ExecutionGroupId::new(11),
                ModelInstanceId::new(index as u64 + 1),
                variant,
                backend,
            )
            .unwrap_or_else(|error| {
                panic!("failed to build {variant} for {backend:?}: {error}")
            });
            install_test_preparation_seals(&draft, backend, 1);

            for execution in draft.capabilities.values() {
                let metadata = execution.metadata();
                let sealed = LoadedCapabilityDescriptor::new(execution.clone(), None, backend);
                if metadata.state_requirement == InferenceStateRequirement::Stateless {
                    sealed.unwrap_or_else(|error| {
                        panic!(
                            "stateless {variant} {:?} failed to seal for {backend:?}: {error}",
                            metadata.capability
                        )
                    });
                } else {
                    let error = sealed.expect_err(&format!(
                        "stateful {variant} {:?} sealed without physical state for {backend:?}",
                        metadata.capability
                    ));
                    assert!(
                        error.to_string().contains(
                            "requires an explicit load-sealed ABI-v2 state publication"
                        ),
                        "unexpected fail-closed error for {variant} {:?} on {backend:?}: {error}",
                        metadata.capability
                    );
                }
            }
        }
    }
}

#[test]
fn loaded_launch_policy_matrix_is_group_exclusive_without_concurrency_evidence() {
    let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(1, 3).unwrap();

    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        for (index, variant) in ModelVariant::all().iter().copied().enumerate() {
            let draft = LoadedModelBundleDraft::build(
                &registry,
                ExecutionGroupId::new(41),
                ModelInstanceId::new(index as u64 + 1),
                variant,
                backend,
            )
            .unwrap_or_else(|error| {
                panic!("failed to build {variant} for {backend:?}: {error}")
            });
            install_test_preparation_seals(&draft, backend, 1);
            for execution in draft.capabilities.values() {
                let metadata = execution.metadata();
                for contract in
                    loaded_execution_contracts(execution.as_ref()).unwrap_or_else(|error| {
                        panic!(
                            "failed launch contract for {variant} {:?} on {backend:?}: {error}",
                            metadata.capability
                        )
                    })
                {
                    let expected = PhysicalLaunchPolicy::ExecutionGroupExclusive;
                    assert_eq!(
                        contract.execution_profile.physical_launch_policy, expected,
                        "{variant} {:?} on {backend:?}",
                        metadata.capability
                    );
                    assert!(contract
                        .stages
                        .iter()
                        .all(|stage| stage.physical_launch_policy == expected));
                }
            }
        }
    }
}

/// An operator's explicit width reaches a chat model above its family's
/// certified width (CPU Qwen3 is certified at 2); an automatic or
/// `max_num_seqs`-derived width still clamps to it.
#[test]
fn an_explicit_width_runs_above_the_certified_chat_width() {
    let decode_width = |explicit: bool| {
        let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(2, 1)
            .unwrap()
            .with_chat_tensor_batch_size(6)
            .with_explicit_tensor_width(explicit);
        let draft = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(46),
            ModelInstanceId::new(1),
            ModelVariant::Qwen306B,
            BackendKind::Cpu,
        )
        .unwrap();
        install_test_preparation_seals(&draft, BackendKind::Cpu, 2);
        draft
            .execution_contracts(CapabilityKind::Chat)
            .unwrap()
            .iter()
            .flat_map(|contract| contract.stages.to_vec())
            .map(|stage| stage.max_batch_size)
            .max()
            .unwrap()
    };
    assert_eq!(decode_width(false), 2);
    assert_eq!(decode_width(true), 6);
}

/// Built as the runtime builds it on each backend, a chat model's
/// stages run at most its family's certified width and every other
/// capability at the engine rows' width (EXE-F7).
#[test]
fn loaded_models_clamp_to_their_family_certified_width() {
    let widest = |variant: ModelVariant, backend: BackendKind| {
        let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(
            crate::catalog::engine_row_width(backend, 8),
            1,
        )
        .unwrap()
        .with_chat_tensor_batch_size(8);
        let draft = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(45),
            ModelInstanceId::new(1),
            variant,
            backend,
        )
        .unwrap();
        // Audio preparation seals carry the engine rows' width; the chat
        // workspace seal has none.
        install_test_preparation_seals(
            &draft,
            backend,
            crate::catalog::engine_row_width(backend, 8),
        );
        draft
            .capabilities
            .keys()
            .flat_map(|capability| draft.execution_contracts(*capability).unwrap())
            .flat_map(|contract| contract.stages.to_vec())
            .map(|stage| stage.max_batch_size)
            .max()
            .unwrap()
    };
    for backend in [BackendKind::Cpu, BackendKind::Metal] {
        assert_eq!(widest(ModelVariant::Lfm2512BInstructGguf, backend), 8);
        assert_eq!(widest(ModelVariant::Qwen306B, backend), 2);
        assert_eq!(widest(ModelVariant::WhisperLargeV3Turbo, backend), 2);
    }
    assert_eq!(widest(ModelVariant::Qwen3508BGguf, BackendKind::Metal), 8);
    assert_eq!(widest(ModelVariant::Qwen3508BGguf, BackendKind::Cpu), 2);
    assert_eq!(widest(ModelVariant::Qwen306B, BackendKind::Cuda), 8);
    assert_eq!(
        widest(ModelVariant::WhisperLargeV3Turbo, BackendKind::Cuda),
        8
    );
}

#[test]
fn missing_concurrency_evidence_keeps_whisper_decoder_scalar_while_encoder_is_static() {
    let request_parallelism = 4;
    let registry =
        RuntimeAdapterRegistry::built_in_with_execution_limits(2, request_parallelism).unwrap();
    let whisper = LoadedModelBundleDraft::build(
        &registry,
        ExecutionGroupId::new(44),
        ModelInstanceId::new(1),
        ModelVariant::WhisperLargeV3Turbo,
        BackendKind::Cpu,
    )
    .unwrap();
    install_test_preparation_seals(&whisper, BackendKind::Cpu, 2);
    let whisper = whisper
        .execution_contracts(CapabilityKind::Asr)
        .unwrap()
        .remove(0);
    assert_eq!(whisper.execution_profile.max_batch_size, 2);
    assert_eq!(
        whisper.execution_profile.concurrency,
        ConcurrencyClass::Exclusive
    );
    assert_eq!(
        whisper.execution_profile.physical_launch_policy,
        PhysicalLaunchPolicy::ExecutionGroupExclusive
    );
    assert_eq!(whisper.stages[0].batch_mode, NativeBatchMode::Static);
    assert_eq!(whisper.stages[0].max_batch_size, 2);
    assert_eq!(whisper.stages[0].concurrency, ConcurrencyClass::Batchable);
    assert!(whisper.stages[1..].iter().all(|stage| {
        stage.batch_mode == NativeBatchMode::None
            && stage.max_batch_size == 1
            && stage.concurrency == ConcurrencyClass::Exclusive
            && stage.shape_policy == StageShapePolicy::Exact
            && stage.physical_launch_policy == PhysicalLaunchPolicy::ExecutionGroupExclusive
    }));

    for (index, variant) in [ModelVariant::Kokoro82M].into_iter().enumerate() {
        let tts = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(45),
            ModelInstanceId::new(index as u64 + 2),
            variant,
            BackendKind::Cpu,
        )
        .unwrap();
        for contract in tts.execution_contracts(CapabilityKind::Tts).unwrap() {
            assert_eq!(
                contract.execution_profile.physical_launch_policy,
                PhysicalLaunchPolicy::ExecutionGroupExclusive
            );
            if contract
                .stages
                .iter()
                .any(|stage| stage.batch_mode == NativeBatchMode::Static)
            {
                assert_eq!(contract.execution_profile.max_batch_size, 2);
                assert_eq!(
                    contract.execution_profile.concurrency,
                    ConcurrencyClass::Batchable
                );
                assert!(contract.execution_profile.capabilities().native_batch);
            } else {
                assert_eq!(contract.execution_profile.max_batch_size, 1);
                assert_eq!(
                    contract.execution_profile.concurrency,
                    ConcurrencyClass::Exclusive
                );
                assert!(!contract.execution_profile.capabilities().native_batch);
            }
        }
    }

    let qwen_tts = LoadedModelBundleDraft::build(
        &registry,
        ExecutionGroupId::new(45),
        ModelInstanceId::new(3),
        ModelVariant::Qwen3Tts12Hz06BCustomVoice,
        BackendKind::Cpu,
    )
    .unwrap();
    for contract in qwen_tts.execution_contracts(CapabilityKind::Tts).unwrap() {
        assert_eq!(contract.execution_profile.max_batch_size, 2);
        assert_eq!(
            contract.execution_profile.concurrency,
            ConcurrencyClass::Batchable
        );
        assert_eq!(
            contract.execution_profile.decode_batch,
            NativeBatchMode::Continuous
        );
        assert_eq!(contract.stages[0].max_batch_size, 1);
        assert_eq!(contract.stages[0].concurrency, ConcurrencyClass::Exclusive);
    }
}

#[test]
fn audio_adapters_remain_scalar_except_for_exact_family_native_opt_ins() {
    let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(8, 4).unwrap();
    let audio_capability = |capability| {
        matches!(
            capability,
            CapabilityKind::Asr
                | CapabilityKind::SpeakerAttributedAsr
                | CapabilityKind::RealtimeAsr
                | CapabilityKind::Tts
                | CapabilityKind::StreamingTts
        )
    };

    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        for (index, variant) in ModelVariant::all().iter().copied().enumerate() {
            let draft = LoadedModelBundleDraft::build(
                &registry,
                ExecutionGroupId::new(46),
                ModelInstanceId::new(index as u64 + 1),
                variant,
                backend,
            )
            .unwrap_or_else(|error| {
                panic!("failed to build {variant} for {backend:?}: {error}")
            });
            install_test_preparation_seals(&draft, backend, 8);
            for execution in draft
                .capabilities
                .values()
                .filter(|execution| audio_capability(execution.metadata().capability))
            {
                for contract in
                    loaded_execution_contracts(execution.as_ref()).unwrap_or_else(|error| {
                        panic!(
                            "failed audio contract for {variant} {:?} on {backend:?}: {error}",
                            execution.metadata().capability
                        )
                    })
                {
                    let qwen3_asr = execution.metadata().capability == CapabilityKind::Asr
                        && variant.family() == crate::catalog::ModelFamily::Qwen3Asr;
                    let vibevoice_asr = execution.metadata().capability == CapabilityKind::Asr
                        && variant.family() == crate::catalog::ModelFamily::VibeVoiceAsr;
                    let granite_asr = execution.metadata().capability == CapabilityKind::Asr
                        && variant.family() == crate::catalog::ModelFamily::GraniteSpeechAsr;
                    let whisper_asr = execution.metadata().capability == CapabilityKind::Asr
                        && variant.family() == crate::catalog::ModelFamily::WhisperAsr;
                    let lfm25_audio_asr = execution.metadata().capability
                        == CapabilityKind::Asr
                        && variant.family() == crate::catalog::ModelFamily::Lfm25Audio;
                    let qwen3_tts = matches!(
                        execution.metadata().capability,
                        CapabilityKind::Tts | CapabilityKind::StreamingTts
                    ) && variant.family()
                        == crate::catalog::ModelFamily::Qwen3Tts;
                    let kokoro_tts = execution.metadata().capability == CapabilityKind::Tts
                        && variant.family() == crate::catalog::ModelFamily::KokoroTts;
                    let fish_s2_tts = FishS2PhysicalTtsAdapterFactory
                        .supports(execution.metadata(), BackendKind::Cpu);
                    let retained_tts = execution.metadata().capability == CapabilityKind::Tts
                        && matches!(
                            variant.family(),
                            crate::catalog::ModelFamily::Lfm25Audio
                                | crate::catalog::ModelFamily::VibeVoiceTts
                                | crate::catalog::ModelFamily::VoxtralTts
                                | crate::catalog::ModelFamily::ChatterboxTts
                                | crate::catalog::ModelFamily::Maya1Tts
                                | crate::catalog::ModelFamily::Maya2Tts
                        );
                    let parakeet_asr = execution.metadata().capability == CapabilityKind::Asr
                        && variant.family() == crate::catalog::ModelFamily::ParakeetAsr;
                    let nemotron_realtime = execution.metadata().capability
                        == CapabilityKind::RealtimeAsr
                        && variant.family() == crate::catalog::ModelFamily::NemotronAsr;
                    let voxtral_realtime = execution.metadata().capability
                        == CapabilityKind::RealtimeAsr
                        && variant == ModelVariant::VoxtralMini4BRealtime2602;
                    if (qwen3_asr
                        || vibevoice_asr
                        || granite_asr
                        || lfm25_audio_asr
                        || qwen3_tts
                        || retained_tts
                        || parakeet_asr
                        || voxtral_realtime
                        || nemotron_realtime)
                        && contract.execution_profile.mode == ExecutionMode::Sequence
                    {
                        assert_eq!(
                            contract.execution_profile.decode_batch,
                            NativeBatchMode::Continuous
                        );
                        assert_eq!(contract.execution_profile.max_batch_size, 8);
                        assert!(contract.execution_profile.capabilities().native_batch);
                    } else if voxtral_realtime {
                        assert_eq!(contract.execution_profile.max_batch_size, 8);
                        assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::Static);
                        assert_eq!(contract.stages[2].batch_mode, NativeBatchMode::Continuous);
                    } else if nemotron_realtime {
                        assert_eq!(contract.execution_profile.max_batch_size, 8);
                        assert_eq!(contract.stages[1].batch_mode, NativeBatchMode::None);
                        assert_eq!(contract.stages[2].batch_mode, NativeBatchMode::Continuous);
                    } else if whisper_asr
                        && contract.execution_profile.mode == ExecutionMode::Sequence
                    {
                        assert_eq!(contract.execution_profile.max_batch_size, 8);
                        assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::Static);
                        assert_eq!(contract.stages[0].max_batch_size, 8);
                        assert!(contract.stages[1..]
                            .iter()
                            .all(|stage| stage.batch_mode == NativeBatchMode::None));
                    } else if kokoro_tts
                        && contract
                            .stages
                            .iter()
                            .any(|stage| stage.batch_mode == NativeBatchMode::Static)
                    {
                        assert_eq!(contract.execution_profile.mode, ExecutionMode::Atomic);
                        assert_eq!(contract.execution_profile.max_batch_size, 8);
                        let static_stage = contract
                            .stages
                            .iter()
                            .find(|stage| stage.batch_mode == NativeBatchMode::Static)
                            .unwrap();
                        assert_eq!(static_stage.shape_policy, StageShapePolicy::Ragged);
                    } else if fish_s2_tts
                        && contract.execution_profile.mode == ExecutionMode::Sequence
                    {
                        assert_eq!(contract.execution_profile.max_batch_size, 1);
                        assert_eq!(
                            contract.execution_profile.concurrency,
                            ConcurrencyClass::Batchable
                        );
                        assert!(!contract.execution_profile.capabilities().native_batch);
                        assert!(contract.stages.iter().all(|stage| {
                            stage.batch_mode == NativeBatchMode::None
                                && stage.max_batch_size == 1
                                && stage.concurrency == ConcurrencyClass::Exclusive
                        }));
                    } else {
                        assert_eq!(contract.execution_profile.max_batch_size, 1);
                        assert_eq!(
                            contract.execution_profile.concurrency,
                            ConcurrencyClass::Exclusive
                        );
                        assert!(!contract.execution_profile.capabilities().native_batch);
                        assert!(contract.stages.iter().all(|stage| {
                            stage.batch_mode == NativeBatchMode::None
                                && stage.max_batch_size == 1
                                && stage.concurrency == ConcurrencyClass::Exclusive
                        }));
                    }
                }
            }
        }
    }
}

#[test]
fn loaded_contracts_reject_policy_without_evidence_and_stage_profile_mismatch() {
    let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(1, 3).unwrap();
    let unknown = LoadedModelBundleDraft::build(
        &registry,
        ExecutionGroupId::new(42),
        ModelInstanceId::new(1),
        ModelVariant::Kokoro82M,
        BackendKind::Cpu,
    )
    .unwrap();
    let base = unknown
        .execution_contracts(CapabilityKind::Tts)
        .unwrap()
        .remove(0);
    assert_eq!(
        base.execution_profile.concurrency,
        ConcurrencyClass::Batchable
    );
    assert_eq!(
        base.execution_profile.physical_launch_policy,
        PhysicalLaunchPolicy::ExecutionGroupExclusive
    );

    for policy in [
        PhysicalLaunchPolicy::ModelExclusive,
        PhysicalLaunchPolicy::concurrent(3).unwrap(),
    ] {
        let mut contract = base.clone();
        contract.execution_profile.physical_launch_policy = policy;
        contract.stages = contract
            .stages
            .iter()
            .cloned()
            .map(|mut stage| {
                stage.physical_launch_policy = policy;
                stage
            })
            .collect::<Vec<_>>()
            .into();
        let error = contract
            .validate_physical_launch_policy()
            .expect_err("unsupported model policy must fail closed");
        assert!(error
            .to_string()
            .contains("no production concurrency evidence is available"));
    }

    let mut mismatch = base;
    let mut stages = mismatch.stages.to_vec();
    stages[0].physical_launch_policy = PhysicalLaunchPolicy::ModelExclusive;
    mismatch.stages = stages.into();
    let error = mismatch
        .validate_physical_launch_policy()
        .expect_err("stage/profile launch-policy mismatch must fail closed");
    assert!(error.to_string().contains("stage launch policy"));
}

#[test]
fn whisper_metadata_and_adapter_abi_cannot_manufacture_concurrent_decode_policy() {
    let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(1, 3).unwrap();
    let metadata = *registry
        .require(CapabilityKind::Asr, ModelVariant::WhisperLargeV3Turbo)
        .unwrap();
    let whisper = WhisperAsrExecutionAdapter::new(
        ExecutionGroupId::new(43),
        ModelInstanceId::new(2),
        metadata,
        BackendKind::Cpu,
        3,
    );
    whisper
        .audio_preparation
        .set(
            crate::models::architectures::whisper::asr::WhisperAudioPreparationStageSeal {
                backend: BackendKind::Cpu,
                dtype: "f32".into(),
                max_batch_size: 3,
                max_workspace_bytes: 1024,
            },
        )
        .unwrap();
    let mut manufactured = whisper.contract(StreamingRequirements::NONE).unwrap();
    assert_eq!(manufactured.adapter_abi_revision, WHISPER_ASR_ADAPTER_ABI);
    assert_eq!(
        manufactured.metadata.model_variant,
        ModelVariant::WhisperLargeV3Turbo
    );
    assert_eq!(manufactured.metadata.capability, CapabilityKind::Asr);

    let concurrent = PhysicalLaunchPolicy::concurrent(3).unwrap();
    manufactured.execution_profile.max_batch_size = 3;
    manufactured.execution_profile.concurrency = ConcurrencyClass::Batchable;
    manufactured.execution_profile.physical_launch_policy = concurrent;
    manufactured.stages = manufactured
        .stages
        .iter()
        .cloned()
        .map(|mut stage| {
            stage.max_batch_size = 3;
            stage.max_work_units = 3;
            stage.concurrency = ConcurrencyClass::Batchable;
            stage.shape_policy = StageShapePolicy::Independent;
            stage.physical_launch_policy = concurrent;
            stage
        })
        .collect::<Vec<_>>()
        .into();

    let error = manufactured
        .validate_physical_launch_policy()
        .expect_err("metadata and adapter ABI are not concurrency evidence");
    assert!(error
        .to_string()
        .contains("no production concurrency evidence is available"));
}

#[test]
fn scalar_adapters_remain_exact_width_one_without_concurrency_evidence() {
    let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(1, 3).unwrap();

    for (index, variant) in ModelVariant::all().iter().copied().enumerate() {
        let draft = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(9),
            ModelInstanceId::new(index as u64 + 1),
            variant,
            BackendKind::Cpu,
        )
        .unwrap_or_else(|error| panic!("failed to build {variant}: {error}"));
        for metadata in registry.capabilities_for(variant) {
            if registry
                .loaded_adapter_factory(metadata, BackendKind::Cpu)
                .unwrap()
                .id()
                != "builtin.scalar"
            {
                continue;
            }
            let contract = draft
                .execution_contracts(metadata.capability)
                .unwrap_or_else(|error| panic!("failed to contract {variant}: {error}"));
            for contract in contract {
                assert_eq!(contract.adapter_abi_revision, SCALAR_ADAPTER_ABI);
                assert_eq!(contract.execution_profile.max_batch_size, 1);
                assert_eq!(
                    contract.execution_profile.concurrency,
                    ConcurrencyClass::Exclusive
                );
                assert_eq!(
                    contract.execution_profile.physical_launch_policy,
                    PhysicalLaunchPolicy::ExecutionGroupExclusive
                );
                assert_eq!(contract.stages[0].max_batch_size, 1);

                assert_eq!(
                    contract.stages[0].shape_policy,
                    crate::engine::StageShapePolicy::Exact
                );
                assert_eq!(
                    contract.stages[0].physical_launch_policy,
                    PhysicalLaunchPolicy::ExecutionGroupExclusive
                );
            }
        }
    }

    let metal = LoadedModelBundle::bind(
        &registry,
        ExecutionGroupId::new(9),
        ModelInstanceId::new(999),
        ModelVariant::Kokoro82M,
        BackendKind::Metal,
    )
    .unwrap();
    let contract = metal.contract(CapabilityKind::StreamingTts, false).unwrap();
    assert_eq!(contract.execution_profile.max_batch_size, 1);
    assert_eq!(
        contract.execution_profile.concurrency,
        ConcurrencyClass::Exclusive
    );

    let cuda = LoadedModelBundleDraft::build(
        &registry,
        ExecutionGroupId::new(10),
        ModelInstanceId::new(1_000),
        ModelVariant::Lfm2512BThinkingGguf,
        BackendKind::Cuda,
    )
    .unwrap();
    install_test_preparation_seals(&cuda, BackendKind::Cuda, 1);
    let contract = cuda
        .execution_contracts(CapabilityKind::Chat)
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(contract.execution_profile.max_batch_size, 1);
    assert_eq!(contract.stages[0].max_batch_size, 1);
    assert_eq!(
        contract.execution_profile.concurrency,
        ConcurrencyClass::Batchable
    );
    assert_eq!(contract.stages.len(), 2);
    assert_eq!(contract.stages[1].batch_mode, NativeBatchMode::Continuous);
    assert_eq!(contract.stages[1].max_batch_size, 1);
}

#[test]
fn voxtral_streaming_binds_to_its_exact_token_engine_adapter() {
    let variant = ModelVariant::VoxtralMini4BRealtime2602;
    let draft = LoadedModelBundleDraft::build(
        &RuntimeAdapterRegistry::built_in(),
        ExecutionGroupId::new(7),
        ModelInstanceId::new(1),
        variant,
        BackendKind::Cpu,
    )
    .unwrap();
    install_test_preparation_seals(&draft, BackendKind::Cpu, 1);

    let contract = draft
        .execution_contracts(CapabilityKind::Asr)
        .unwrap()
        .into_iter()
        .find(|contract| {
            contract.stages[0].output_visibility == OutputVisibility::IncrementalCommitted
        })
        .expect("streaming Voxtral contract");
    assert_eq!(
        contract.metadata.execution_target,
        ExecutionTargetKind::TokenEngine
    );
    assert_eq!(contract.metadata.streaming_mode, StreamingMode::Chunked);
    assert!(contract.execution_profile.resolved_from_loaded_model);
    assert_eq!(
        contract.stages[0].output_visibility,
        OutputVisibility::IncrementalCommitted
    );
    let error = draft
        .seal(HashMap::new())
        .expect_err("Voxtral must not seal without physical invocation state");
    assert!(error
        .to_string()
        .contains("requires an explicit load-sealed ABI-v2 state publication"));
}

#[test]
fn offline_asr_transport_progress_does_not_require_native_streaming() {
    let draft = LoadedModelBundleDraft::build(
        &RuntimeAdapterRegistry::built_in(),
        ExecutionGroupId::new(7),
        ModelInstanceId::new(1),
        ModelVariant::ParakeetTdt06BV3,
        BackendKind::Cpu,
    )
    .unwrap();
    let execution = draft.capabilities.get(&CapabilityKind::Asr).unwrap();
    assert!(execution
        .contract(StreamingRequirements::native(true))
        .is_err());
    let transport = execution
        .contract(StreamingRequirements::transport_only())
        .expect("offline ASR must expose atomic executor progress");
    assert_eq!(transport.metadata.streaming_mode, StreamingMode::None);
    assert_eq!(transport.execution_profile.mode, ExecutionMode::Sequence);
    assert_eq!(transport.execution_profile.cache_mode, CacheMode::None);
    assert_eq!(transport.stages.len(), 3);
    assert_eq!(transport.stages[0].batch_mode, NativeBatchMode::None);
    assert_eq!(transport.stages[1].batch_mode, NativeBatchMode::Static);
    assert_eq!(transport.stages[2].batch_mode, NativeBatchMode::Continuous);
    assert_eq!(
        transport.stages[2].retained_state_selections.as_deref(),
        Some(
            &[ClockedStateSelection::new(
                crate::models::architectures::parakeet::asr::PARAKEET_PREDICTOR_STATE_GROUP,
                StateClock::DecoderTokens,
            )
            .unwrap()][..]
        )
    );
}

#[test]
fn every_pipeline_selectable_asr_publishes_an_atomic_long_form_graph() {
    // The diarization pipeline loads its ASR long-form on every backend
    // and refuses a contract without an atomic stage, so every variant the
    // pipeline resolver accepts must publish one: in the sealed bundle and
    // in the weight-free probe the pipeline runs before diarizing.
    let registry = RuntimeAdapterRegistry::built_in();
    let mut selectable = Vec::new();
    for &variant in ModelVariant::all() {
        if crate::catalog::resolve_diarization_asr_model_variant(Some(variant.dir_name()))
            != Ok(variant)
        {
            continue;
        }
        selectable.push(variant);
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            assert!(
                registry
                    .supports_atomic_long_form(CapabilityKind::Asr, variant, backend)
                    .unwrap_or_else(|error| panic!("{variant}/{backend:?}: {error}")),
                "{variant}/{backend:?} probe has no atomic long-form graph"
            );
            let draft = LoadedModelBundleDraft::build(
                &registry,
                ExecutionGroupId::new(7),
                ModelInstanceId::new(1),
                variant,
                backend,
            )
            .unwrap();
            install_test_preparation_seals(&draft, backend, 1);
            let contract = draft
                .capabilities
                .get(&CapabilityKind::Asr)
                .unwrap()
                .contract(StreamingRequirements::NONE.with_asr_long_form(true))
                .unwrap();
            assert!(
                contract_supports_atomic_blocking(&contract),
                "{variant}/{backend:?} long-form contract has no atomic stage"
            );
            // The pipeline transcribes only from an invocation workspace.
            assert!(
                matches!(
                    contract.metadata.state_requirement,
                    InferenceStateRequirement::Invocation
                        | InferenceStateRequirement::RetainedAndInvocation
                ),
                "{variant}/{backend:?} ASR is not invocation-backed"
            );
            assert!(
                contract
                    .adapter_binding()
                    .unwrap()
                    .stage_for_work(&crate::engine::WorkUnit::AtomicJob {
                        kind: "diarization.transcribe".into(),
                    })
                    .is_ok(),
                "{variant}/{backend:?} cannot place a diarization transcription"
            );
        }
    }
    // Every native ASR family and Voxtral is selectable by its catalog id
    // (no variant escapes the check through an unparseable name), and no
    // audio-chat model is: the pipeline cannot hold one as its ASR.
    let expected = ModelVariant::all()
        .iter()
        .copied()
        .filter(|variant| variant.is_asr() || variant.is_voxtral())
        .collect::<Vec<_>>();
    assert_eq!(selectable, expected);
    assert!(selectable.contains(&ModelVariant::ParakeetTdt06BV3));
    assert!(selectable.contains(&ModelVariant::VoxtralMini4BRealtime2602));
    assert!(!selectable.contains(&ModelVariant::Lfm25Audio15BGguf));
}

#[test]
fn every_asr_stream_route_binds_to_its_loaded_contract() {
    let registry = RuntimeAdapterRegistry::built_in();
    let mut offline_only = Vec::new();
    for &variant in ModelVariant::all() {
        let Ok(metadata) = registry.require(CapabilityKind::Asr, variant) else {
            continue;
        };
        if metadata.execution_target != ExecutionTargetKind::TokenEngine {
            continue;
        }
        let draft = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(7),
            ModelInstanceId::new(1),
            variant,
            BackendKind::Cpu,
        )
        .unwrap();
        install_test_preparation_seals(&draft, BackendKind::Cpu, 1);
        let execution = draft.capabilities.get(&CapabilityKind::Asr).unwrap();
        // The requirements the runtime binds for a stream=true request.
        let native = crate::runtime::asr::asr_streams_natively(&registry, variant);
        let streaming = if native {
            StreamingRequirements::native(true)
        } else {
            offline_only.push(variant);
            StreamingRequirements::transport_only()
        };
        for asr_long_form in [false, true] {
            let result = execution.contract(streaming.with_asr_long_form(asr_long_form));
            assert!(
                result.is_ok(),
                "{variant} stream (native {native}, long-form {asr_long_form}): {:?}",
                result.err()
            );
        }
    }
    assert!(offline_only.contains(&ModelVariant::ParakeetTdt06BV3));
    assert!(offline_only.contains(&ModelVariant::Nemotron35AsrStreaming06B));
}

#[test]
fn lfm2_sequence_chat_remains_quantum_committed_when_streaming() {
    let draft = LoadedModelBundleDraft::build(
        &RuntimeAdapterRegistry::built_in(),
        ExecutionGroupId::new(7),
        ModelInstanceId::new(1),
        ModelVariant::Lfm2512BThinkingGguf,
        BackendKind::Cpu,
    )
    .unwrap();
    install_test_preparation_seals(&draft, BackendKind::Cpu, 1);
    let adapter = draft.capabilities.get(&CapabilityKind::Chat).unwrap();

    let non_streaming = adapter.contract(StreamingRequirements::NONE).unwrap();
    assert_eq!(
        non_streaming.execution_profile.mode,
        ExecutionMode::Sequence
    );
    assert!(non_streaming
        .stages
        .iter()
        .all(|stage| stage.output_visibility == OutputVisibility::AfterQuantumCommit));

    let streaming = adapter
        .contract(StreamingRequirements::native(true))
        .unwrap();
    assert_eq!(streaming.execution_profile.mode, ExecutionMode::Sequence);
    assert!(streaming
        .stages
        .iter()
        .all(|stage| stage.output_visibility == OutputVisibility::AfterQuantumCommit));
}

#[test]
fn sequence_chat_remains_quantum_committed_when_streaming() {
    let draft = LoadedModelBundleDraft::build(
        &RuntimeAdapterRegistry::built_in(),
        ExecutionGroupId::new(7),
        ModelInstanceId::new(1),
        ModelVariant::Qwen3508BGguf,
        BackendKind::Cpu,
    )
    .unwrap();
    install_test_preparation_seals(&draft, BackendKind::Cpu, 1);

    let streaming = draft
        .capabilities
        .get(&CapabilityKind::Chat)
        .unwrap()
        .contract(StreamingRequirements::native(true))
        .unwrap();
    assert_eq!(streaming.execution_profile.mode, ExecutionMode::Sequence);
    assert!(streaming
        .stages
        .iter()
        .all(|stage| { stage.output_visibility == OutputVisibility::AfterQuantumCommit }));
}

#[test]
fn adapter_instances_are_distinct_across_capabilities_and_loads() {
    let registry = RuntimeAdapterRegistry::built_in();
    let first = LoadedModelBundle::bind(
        &registry,
        ExecutionGroupId::new(1),
        ModelInstanceId::new(1),
        ModelVariant::Kokoro82M,
        BackendKind::Metal,
    )
    .expect("first bundle");
    let second = LoadedModelBundle::bind(
        &registry,
        ExecutionGroupId::new(1),
        ModelInstanceId::new(2),
        ModelVariant::Kokoro82M,
        BackendKind::Metal,
    )
    .expect("second bundle");

    let first_tts = first
        .capability_binding_for_streaming(CapabilityKind::Tts, StreamingRequirements::NONE)
        .expect("first tts")
        .execution
        .adapter_instance_id;
    let first_streaming_tts = first
        .capability_binding_for_streaming(
            CapabilityKind::StreamingTts,
            StreamingRequirements::NONE,
        )
        .expect("first streaming tts")
        .execution
        .adapter_instance_id;
    let second_tts = second
        .capability_binding_for_streaming(CapabilityKind::Tts, StreamingRequirements::NONE)
        .expect("second tts")
        .execution
        .adapter_instance_id;

    assert_ne!(first_tts, first_streaming_tts);
    assert_ne!(first_tts, second_tts);
}

#[test]
fn bundle_draft_preserves_the_exact_adapter_identity_through_state_seal() {
    let registry = RuntimeAdapterRegistry::built_in();
    let draft = LoadedModelBundleDraft::build(
        &registry,
        ExecutionGroupId::new(12),
        ModelInstanceId::new(91),
        ModelVariant::Kokoro82M,
        BackendKind::Cpu,
    )
    .unwrap();
    let contracts = draft.execution_contracts(CapabilityKind::Tts).unwrap();
    let adapter = contracts[0].adapter_instance_id;
    assert!(contracts
        .iter()
        .all(|contract| contract.adapter_instance_id == adapter));

    let bundle = draft.seal(HashMap::new()).unwrap();
    let sealed = bundle
        .contract_for_streaming(CapabilityKind::Tts, StreamingRequirements::NONE)
        .unwrap();
    assert_eq!(sealed.adapter_instance_id, adapter);
}

#[test]
fn replacing_the_scalar_factory_adds_an_optimized_model_without_bundle_branching() {
    let variant = ModelVariant::Kokoro82M;
    let mut registry = RuntimeAdapterRegistry::built_in();
    registry.loaded_adapter_factories.retain(|factory| {
        !matches!(
            factory.id(),
            "builtin.scalar" | "builtin.kokoro_tts.tensor_static"
        )
    });
    registry
        .loaded_adapter_factories
        .push(Arc::new(TestScalarFactory {
            excluded_model_variant: variant,
        }));
    registry
        .loaded_adapter_factories
        .push(Arc::new(TestStaticTtsFactory {
            id: "test.kokoro.tensor_static",
            model_variant: variant,
        }));
    registry.validate_loaded_adapter_factories().unwrap();
    let draft = LoadedModelBundleDraft::build(
        &registry,
        ExecutionGroupId::new(1),
        ModelInstanceId::new(2),
        variant,
        BackendKind::Cpu,
    )
    .unwrap();
    let contract = draft
        .capabilities
        .get(&CapabilityKind::Tts)
        .unwrap()
        .contract(StreamingRequirements::NONE)
        .unwrap();

    assert_eq!(contract.adapter_abi_revision, STATIC_TENSOR_ADAPTER_ABI);
    assert!(contract
        .stages
        .iter()
        .any(|stage| stage.batch_mode == NativeBatchMode::Static));
    assert!(registry
        .static_tensor_batch_variants(BackendKind::Cpu)
        .contains(&variant));
}

#[test]
fn kokoro_factory_publishes_stateless_ragged_static_generation() {
    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(4, 2).unwrap();
        let draft = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(1),
            ModelInstanceId::new(2),
            ModelVariant::Kokoro82M,
            backend,
        )
        .unwrap();
        let contract = draft
            .capabilities
            .get(&CapabilityKind::Tts)
            .unwrap()
            .contract(StreamingRequirements::NONE)
            .unwrap();

        assert_eq!(contract.adapter_abi_revision, STATIC_TENSOR_ADAPTER_ABI);
        assert_eq!(contract.execution_profile.mode, ExecutionMode::Atomic);
        assert_eq!(contract.execution_profile.cache_mode, CacheMode::None);
        assert_eq!(contract.execution_profile.max_batch_size, 4);
        assert_eq!(contract.stages.len(), 3);
        let preparation = contract
            .stages
            .iter()
            .find(|stage| stage.selector == StageWorkSelector::PreSequencePreparation)
            .unwrap();
        assert_eq!(preparation.batch_mode, NativeBatchMode::None);
        assert_eq!(preparation.max_batch_size, 1);
        let static_stage = contract
            .stages
            .iter()
            .find(|stage| stage.batch_mode == NativeBatchMode::Static)
            .unwrap();
        assert_eq!(static_stage.selector, StageWorkSelector::Atomic);
        assert_eq!(static_stage.shape_policy, StageShapePolicy::Ragged);
        assert_eq!(static_stage.max_batch_size, 4);
    }
}

#[test]
fn missing_loaded_factory_fails_closed() {
    let variant = ModelVariant::Kokoro82M;
    let mut registry = RuntimeAdapterRegistry::built_in();
    registry.loaded_adapter_factories.retain(|factory| {
        !matches!(
            factory.id(),
            "builtin.scalar" | "builtin.kokoro_tts.tensor_static"
        )
    });
    let metadata = *registry.require(CapabilityKind::Tts, variant).unwrap();

    let error = registry
        .loaded_adapter_factory(metadata, BackendKind::Cpu)
        .expect_err("every loaded capability requires exactly one factory");

    let message = error.to_string();
    assert!(message.contains("has no execution adapter factory"));
    assert!(message.contains(&variant.to_string()));
}

#[test]
fn overlapping_loaded_factories_fail_closed() {
    let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
    let mut registry = RuntimeAdapterRegistry::built_in_with_execution_limits(2, 1).unwrap();
    registry
        .loaded_adapter_factories
        .push(Arc::new(TestStaticTtsFactory {
            id: "test.overlapping.tensor_static",
            model_variant: variant,
        }));

    let error = LoadedModelBundle::bind(
        &registry,
        ExecutionGroupId::new(1),
        ModelInstanceId::new(2),
        variant,
        BackendKind::Cpu,
    )
    .expect_err("ambiguous factories must not depend on registration order");

    assert!(error.to_string().contains("matches both"));
    assert!(error.to_string().contains("test.overlapping.tensor_static"));
}

#[test]
fn qwen_tts_factory_binds_every_capability_to_physical_sequence_stages() {
    let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
    let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(4, 1).unwrap();
    let draft = LoadedModelBundleDraft::build(
        &registry,
        ExecutionGroupId::new(1),
        ModelInstanceId::new(2),
        variant,
        BackendKind::Metal,
    )
    .unwrap();

    let tts = draft.capabilities.get(&CapabilityKind::Tts).unwrap();
    let physical = tts.contract(StreamingRequirements::NONE).unwrap();
    assert_eq!(physical.adapter_abi_revision, CONTINUOUS_TTS_ADAPTER_ABI);
    assert_eq!(physical.execution_profile.mode, ExecutionMode::Sequence);
    assert_eq!(physical.execution_profile.prefill, PrefillMode::Incremental);
    assert_eq!(
        physical.execution_profile.cache_mode,
        CacheMode::ExternalPaged
    );
    assert_eq!(physical.execution_profile.max_batch_size, 4);
    assert_eq!(
        physical.execution_profile.decode_batch,
        NativeBatchMode::Continuous
    );
    assert_eq!(physical.stages.len(), 2);
    assert_eq!(
        physical.stages[0].selector,
        StageWorkSelector::SequencePrefill
    );
    assert_eq!(physical.stages[0].batch_mode, NativeBatchMode::None);
    assert_eq!(physical.stages[0].max_batch_size, 1);
    assert_eq!(physical.stages[0].concurrency, ConcurrencyClass::Exclusive);
    assert_eq!(
        physical.stages[0].max_workspace_bytes,
        STATIC_TTS_MAX_BATCH_WORKSPACE_BYTES
    );
    assert_eq!(
        physical.stages[1].selector,
        StageWorkSelector::SequenceDecode
    );
    assert_eq!(physical.stages[1].batch_mode, NativeBatchMode::Continuous);
    assert_eq!(physical.stages[1].concurrency, ConcurrencyClass::Batchable);
    assert_eq!(
        physical.stages[1].max_workspace_bytes,
        STATIC_TTS_MAX_BATCH_WORKSPACE_BYTES
    );
    let production_output_bytes = u64::try_from(ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES)
        .unwrap()
        * 1_920
        * std::mem::size_of::<f32>() as u64;
    assert!(physical.stages[1].max_workspace_bytes >= production_output_bytes * 1_024);

    let streaming = tts.contract(StreamingRequirements::native(true)).unwrap();
    assert_eq!(streaming.adapter_abi_revision, CONTINUOUS_TTS_ADAPTER_ABI);
    assert_eq!(
        streaming.execution_profile.prefill_batch,
        NativeBatchMode::None
    );
    assert_eq!(streaming.execution_profile.max_batch_size, 4);
    assert_eq!(streaming.stages[0].batch_mode, NativeBatchMode::None);
    assert_eq!(streaming.stages[1].batch_mode, NativeBatchMode::Continuous);

    let streaming_capability = draft
        .capabilities
        .get(&CapabilityKind::StreamingTts)
        .unwrap()
        .contract(StreamingRequirements::NONE)
        .unwrap();
    assert_eq!(
        streaming_capability.adapter_abi_revision,
        CONTINUOUS_TTS_ADAPTER_ABI
    );
    assert_eq!(
        streaming_capability.execution_profile.cache_mode,
        CacheMode::ExternalPaged
    );
}

#[test]
fn vibevoice_tts_preparation_uses_the_authenticated_operation_boundary() {
    let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(2, 1).unwrap();
    let draft = LoadedModelBundleDraft::build(
        &registry,
        ExecutionGroupId::new(1),
        ModelInstanceId::new(2),
        ModelVariant::VibeVoice15BTts,
        BackendKind::Cpu,
    )
    .unwrap();
    let contract = draft
        .capabilities
        .get(&CapabilityKind::Tts)
        .unwrap()
        .contract(StreamingRequirements::NONE)
        .unwrap();

    assert_eq!(
        contract.stages[0].selector,
        StageWorkSelector::PreSequencePreparation
    );
    assert_eq!(
        contract.stages[0].membership_safe_point,
        MembershipSafePoint::OperationBoundary
    );
    assert_eq!(
        contract.stages[1].retained_state_selections.as_deref(),
        Some(&[][..])
    );
    assert_eq!(
        contract.stages[2].retained_state_selections.as_deref(),
        Some(
            [ClockedStateSelection::new(
                crate::models::architectures::vibevoice::VIBEVOICE_TTS_TOKENIZER_GROUP,
                StateClock::CodecFrames,
            )
            .unwrap()]
            .as_slice()
        )
    );
}

#[test]
fn qwen_tts_physical_sequence_is_enabled_on_cpu_metal_and_cuda() {
    let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
    let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(4, 1).unwrap();
    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        let draft = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(1),
            ModelInstanceId::new(2),
            variant,
            backend,
        )
        .unwrap();

        let contract = draft
            .capabilities
            .get(&CapabilityKind::Tts)
            .unwrap()
            .contract(StreamingRequirements::NONE)
            .unwrap();
        assert_eq!(contract.adapter_abi_revision, CONTINUOUS_TTS_ADAPTER_ABI);
        assert_eq!(contract.execution_profile.backend, backend);
        assert_eq!(contract.execution_profile.max_batch_size, 4);
        assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::None);
        assert_eq!(contract.stages[1].batch_mode, NativeBatchMode::Continuous);
        assert_eq!(
            contract.execution_profile.cache_mode,
            CacheMode::ExternalPaged
        );
    }
}

#[test]
fn continuous_chat_workspace_is_sealed_before_publication_and_scales_to_loaded_geometry() {
    // Full-size hybrid verification and graph storage exceed the former
    // 16 MiB batch ceiling even for one row. Also cover small/disabled
    // feature configurations without imposing a production-model minimum.
    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        for width in [1, 8] {
            for accelerator_bytes in [8_192, 512 * 1024 * 1024] {
                let adapter = ContinuousChatExecutionAdapter::new(
                    ExecutionGroupId::new(1),
                    ModelInstanceId::new(2),
                    *RuntimeAdapterRegistry::built_in()
                        .require(CapabilityKind::Chat, ModelVariant::Qwen3827BFp8)
                        .unwrap(),
                    backend,
                    width,
                    1,
                );
                assert!(adapter.contract(StreamingRequirements::NONE).is_err());
                adapter.seal_chat_workspace(accelerator_bytes).unwrap();
                adapter.seal_chat_workspace(accelerator_bytes).unwrap();
                assert!(adapter.seal_chat_workspace(accelerator_bytes + 1).is_err());
                let row = crate::engine::WorkCost::with_workspace(
                    1,
                    1,
                    continuous_chat_workspace_per_row(accelerator_bytes).unwrap(),
                );
                let row_bytes = row.workspace.workspace_bytes().unwrap();
                assert!(
                    row_bytes > accelerator_bytes,
                    "host collation must be included"
                );
                for streaming in [
                    StreamingRequirements::NONE,
                    StreamingRequirements::native(true),
                ] {
                    let contract = adapter.contract(streaming).unwrap();
                    let decode = &contract.stages[1];
                    assert_eq!(decode.workspace_per_row_bytes, row_bytes);
                    assert_eq!(decode.max_workspace_bytes, row_bytes * width as u64);
                    let budget = crate::engine::BatchBudget {
                        max_rows: width,
                        max_workspace_bytes: decode.max_workspace_bytes,
                        ..crate::engine::BatchBudget::width_one()
                    };
                    let mut accumulated = crate::engine::WorkCost::default();
                    for rows in 0..width {
                        assert!(budget.admits(rows, accumulated, row));
                        accumulated = accumulated.checked_add(row).unwrap();
                    }
                    assert!(!budget.admits(width, accumulated, row));
                    let oversized =
                        crate::engine::WorkCost::new(1, 1, decode.max_workspace_bytes + 1);
                    assert!(!budget.admits(0, crate::engine::WorkCost::default(), oversized));
                }
            }
        }
    }
}

#[test]
fn continuous_chat_workspace_rejects_row_and_batch_overflow() {
    let adapter = ContinuousChatExecutionAdapter::new(
        ExecutionGroupId::new(1),
        ModelInstanceId::new(2),
        *RuntimeAdapterRegistry::built_in()
            .require(CapabilityKind::Chat, ModelVariant::Qwen3827BFp8)
            .unwrap(),
        BackendKind::Cuda,
        8,
        1,
    );
    assert!(adapter.seal_chat_workspace(u64::MAX).is_err());
    assert!(adapter.seal_chat_workspace(u64::MAX / 2).is_err());
    assert!(adapter.contract(StreamingRequirements::NONE).is_err());
}

#[test]
fn qwen_chat_native_factory_publishes_scalar_prefill_and_ragged_decode() {
    let variant = ModelVariant::Qwen306B;
    let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(8, 1).unwrap();
    let draft = LoadedModelBundleDraft::build(
        &registry,
        ExecutionGroupId::new(1),
        ModelInstanceId::new(2),
        variant,
        BackendKind::Cuda,
    )
    .unwrap();

    install_test_preparation_seals(&draft, BackendKind::Cuda, 8);
    let contract = draft
        .capabilities
        .get(&CapabilityKind::Chat)
        .unwrap()
        .contract(StreamingRequirements::native(true))
        .unwrap();
    assert_eq!(contract.adapter_abi_revision, CONTINUOUS_TENSOR_ADAPTER_ABI);
    assert_eq!(contract.execution_profile.mode, ExecutionMode::Sequence);
    assert_eq!(
        contract.execution_profile.prefill_batch,
        NativeBatchMode::None
    );
    assert_eq!(
        contract.execution_profile.decode_batch,
        NativeBatchMode::Continuous
    );
    assert_eq!(contract.execution_profile.max_batch_size, 8);
    assert_eq!(contract.stages.len(), 2);
    assert_eq!(
        contract.stages[0].selector,
        StageWorkSelector::SequencePrefill
    );
    assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::None);
    assert_eq!(contract.stages[0].max_batch_size, 1);
    assert_eq!(
        contract.stages[1].selector,
        StageWorkSelector::SequenceDecode
    );
    assert_eq!(contract.stages[1].batch_mode, NativeBatchMode::Continuous);
    assert_eq!(contract.stages[1].max_batch_size, 8);
    assert_eq!(contract.stages[1].max_work_units, 8);
    assert!(contract
        .stages
        .iter()
        .all(|stage| { stage.output_visibility == OutputVisibility::AfterQuantumCommit }));
    assert_eq!(
        contract.stages[1].max_workspace_bytes,
        8 * continuous_chat_workspace_per_row(8_192)
            .unwrap()
            .workspace_bytes()
            .unwrap()
    );
}

#[test]
fn fish_s2_factory_binds_both_tts_capabilities_to_native_streaming_stages() {
    let registry = RuntimeAdapterRegistry::built_in();
    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        let draft = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(1),
            ModelInstanceId::new(2),
            ModelVariant::FishAudioS2Pro,
            backend,
        )
        .unwrap();
        for capability in [CapabilityKind::Tts, CapabilityKind::StreamingTts] {
            let adapter = draft.capabilities.get(&capability).unwrap();
            assert!(FishS2PhysicalTtsAdapterFactory.supports(adapter.metadata(), backend));
            for streaming in [
                StreamingRequirements::NONE,
                StreamingRequirements::native(true),
            ] {
                let contract = adapter.contract(streaming).unwrap();
                assert_eq!(contract.metadata.capability, capability);
                assert_eq!(contract.adapter_abi_revision, FISH_S2_TTS_ADAPTER_ABI);
                assert_eq!(contract.execution_profile.backend, backend);
                assert_eq!(contract.execution_profile.mode, ExecutionMode::Sequence);
                assert_eq!(
                    contract.execution_profile.cache_mode,
                    CacheMode::ExternalPaged
                );
                assert_eq!(
                    contract
                        .stages
                        .iter()
                        .map(|stage| stage.selector)
                        .collect::<Vec<_>>(),
                    vec![
                        StageWorkSelector::PreSequencePreparation,
                        StageWorkSelector::SequencePrefill,
                        StageWorkSelector::SequenceDecode,
                        StageWorkSelector::SequenceFinalize,
                        StageWorkSelector::SequenceAudioDecode,
                    ]
                );
                assert!(contract.stages.iter().all(|stage| {
                    stage.output_visibility == OutputVisibility::AfterQuantumCommit
                }));
            }
        }
    }
}

#[test]
fn fish_capacity_seals_both_routes_and_rejects_rebinding() {
    let registry = RuntimeAdapterRegistry::built_in();
    let draft = LoadedModelBundleDraft::build(
        &registry,
        ExecutionGroupId::new(17),
        ModelInstanceId::new(19),
        ModelVariant::FishAudioS2Pro,
        BackendKind::Cuda,
    )
    .unwrap();
    let mut capacity =
        crate::runtime::fish_capacity::ResolvedFishServingCapacity::scalar().unwrap();
    capacity.ar_rows = 7;
    capacity.codec_rows = 7;
    capacity.active_rows = 9;
    capacity.staged_rows = 7;
    capacity.native_batching = true;
    draft.seal_fish_capacity(capacity.clone()).unwrap();
    draft.seal_fish_capacity(capacity.clone()).unwrap();
    for capability in [CapabilityKind::Tts, CapabilityKind::StreamingTts] {
        let contracts = draft.execution_contracts(capability).unwrap();
        for contract in contracts {
            assert_eq!(contract.execution_profile.max_batch_size, 7);
            let decode = contract
                .stages
                .iter()
                .find(|stage| stage.selector == StageWorkSelector::SequenceDecode)
                .unwrap();
            assert_eq!(decode.batch_mode, NativeBatchMode::Continuous);
            assert_eq!(decode.max_batch_size, 7);
            assert_eq!(
                decode.max_workspace_bytes,
                decode.workspace_per_row_bytes * 7
            );
            assert_eq!(contract.stages[0].max_batch_size, 1);
            let prefill = contract
                .stages
                .iter()
                .find(|stage| stage.selector == StageWorkSelector::SequencePrefill)
                .unwrap();
            assert_eq!(prefill.batch_mode, NativeBatchMode::Static);
            assert_eq!(prefill.max_work_units, capacity.prefill_tokens);
            let codec = contract
                .stages
                .iter()
                .find(|stage| stage.selector == StageWorkSelector::SequenceAudioDecode)
                .unwrap();
            assert_eq!(codec.batch_mode, NativeBatchMode::Static);
            assert_eq!(codec.max_batch_size, 7);
            assert_eq!(codec.max_workspace_bytes, codec.workspace_per_row_bytes * 7);
        }
    }
    capacity.ar_rows = 6;
    assert!(draft.seal_fish_capacity(capacity).is_err());
}

#[test]
fn fish_s2_loaded_contracts_have_only_retained_state_and_codec_finalization() {
    let registry = RuntimeAdapterRegistry::built_in();
    let metadata = *registry
        .require(CapabilityKind::Tts, ModelVariant::FishAudioS2Pro)
        .unwrap();
    let adapter = FishS2TtsExecutionAdapter::new(
        ExecutionGroupId::new(1),
        ModelInstanceId::new(2),
        metadata,
        BackendKind::Cpu,
    );

    let contracts = loaded_execution_contracts(&adapter).unwrap();
    assert!(contracts
        .iter()
        .any(|contract| contract.execution_profile.mode == ExecutionMode::Sequence));
    assert!(contracts
        .iter()
        .all(|contract| contract.execution_profile.mode != ExecutionMode::Atomic));
    let retained = contracts
        .iter()
        .find(|contract| contract.execution_profile.mode == ExecutionMode::Sequence)
        .unwrap();
    assert_eq!(retained.stages.len(), 5);
    let finalize = retained
        .stages
        .iter()
        .find(|stage| stage.selector == StageWorkSelector::SequenceFinalize)
        .unwrap();
    assert_eq!(finalize.id, StageId::new(3));
    assert_eq!(finalize.max_batch_size, 1);
    assert_eq!(
        finalize.output_visibility,
        OutputVisibility::AfterQuantumCommit
    );
    assert!(finalize
        .retained_state_selections
        .as_ref()
        .is_none_or(Vec::is_empty));
}

#[test]
fn qwen_asr_native_factory_publishes_incremental_scalar_prefill_and_ragged_decode() {
    let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(8, 1).unwrap();

    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        let metadata = *registry
            .require(CapabilityKind::Asr, ModelVariant::Qwen3Asr06BGguf)
            .unwrap();
        let adapter = ContinuousAsrExecutionAdapter::new(
            ExecutionGroupId::new(61),
            ModelInstanceId::new(62),
            metadata,
            backend,
            8,
        );
        assert!(adapter.contract(StreamingRequirements::NONE).is_err());
        let long_form = adapter
            .contract(StreamingRequirements::NONE.with_asr_long_form(true))
            .unwrap();
        assert_eq!(long_form.stages.len(), 1);
        assert_eq!(long_form.stages[0].name, "asr.long_form.chunked");
        assert_eq!(long_form.stages[0].progress, StageProgressKind::Iterative);
        adapter
            .install_audio_preparation_seal(
                crate::models::architectures::qwen3::asr::Qwen3AsrAudioPreparationStageSeal {
                    backend,
                    audio_dtype: "f32".into(),
                    text_dtype: "f32".into(),
                    max_batch_size: 8,
                    max_workspace_bytes: 64 * 1024 * 1024,
                },
            )
            .unwrap();
        for streaming in [
            StreamingRequirements::NONE,
            StreamingRequirements::native(true),
        ] {
            let contract = adapter.contract(streaming).unwrap();
            assert_eq!(contract.adapter_abi_revision, CONTINUOUS_ASR_ADAPTER_ABI);
            assert_eq!(contract.execution_profile.mode, ExecutionMode::Sequence);
            assert_eq!(contract.execution_profile.prefill, PrefillMode::Incremental);
            assert_eq!(
                contract.execution_profile.prefill_batch,
                NativeBatchMode::None
            );
            assert_eq!(
                contract.execution_profile.decode_batch,
                NativeBatchMode::Continuous
            );
            assert_eq!(contract.execution_profile.max_batch_size, 8);
            assert!(contract.execution_profile.recompute_safe);
            assert!(contract.execution_profile.cache_release_safe);
            assert!(!contract.execution_profile.prefix_reuse_safe);
            assert_eq!(contract.stages.len(), 3);
            assert_eq!(
                contract.stages[0].selector,
                StageWorkSelector::PreSequencePreparation
            );
            assert_eq!(contract.stages[0].name, "asr.encoder.audio");
            assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::Static);
            assert_eq!(contract.stages[0].max_batch_size, 8);
            assert_eq!(contract.stages[0].max_workspace_bytes, 64 * 1024 * 1024);
            assert_eq!(
                contract.stages[1].selector,
                StageWorkSelector::SequencePrefill
            );
            assert_eq!(contract.stages[1].batch_mode, NativeBatchMode::None);
            assert_eq!(contract.stages[1].max_batch_size, 1);
            assert_eq!(
                contract.stages[2].selector,
                StageWorkSelector::SequenceDecode
            );
            assert_eq!(contract.stages[2].batch_mode, NativeBatchMode::Continuous);
            assert_eq!(contract.stages[2].max_batch_size, 8);
            assert_eq!(contract.stages[2].max_work_units, 8);
            assert_eq!(
                contract.stages[2].max_workspace_bytes,
                CONTINUOUS_ASR_MAX_BATCH_WORKSPACE_BYTES
            );
        }
    }
}

#[test]
fn continuous_chat_stage_reserves_the_bounded_solo_speculation_quantum() {
    let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(1, 1).unwrap();
    let draft = LoadedModelBundleDraft::build(
        &registry,
        ExecutionGroupId::new(1),
        ModelInstanceId::new(2),
        ModelVariant::Qwen3827BFp8,
        BackendKind::Cpu,
    )
    .unwrap();
    install_test_preparation_seals(&draft, BackendKind::Cpu, 1);
    let contract = draft
        .capabilities
        .get(&CapabilityKind::Chat)
        .unwrap()
        .contract(StreamingRequirements::native(true))
        .unwrap();

    assert_eq!(contract.stages[1].max_batch_size, 1);
    assert_eq!(
        contract.stages[1].max_work_units,
        CONTINUOUS_CHAT_MAX_DECODE_QUANTUM
    );
}

#[test]
fn lfm25_audio_tts_factory_publishes_sealed_continuous_decode_graph() {
    let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(8, 1).unwrap();
    let scalar = ScalarExecutionAdapterFactory;
    let factory = Lfm25AudioPhysicalTtsAdapterFactory;
    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        let metadata = *registry
            .require(CapabilityKind::Tts, ModelVariant::Lfm25Audio15BGguf)
            .unwrap();
        assert!(factory.supports(metadata, backend));
        assert!(!scalar.supports(metadata, backend));
        let adapter = Lfm25AudioTtsExecutionAdapter::new(
            ExecutionGroupId::new(73),
            ModelInstanceId::new(74),
            metadata,
            backend,
            8,
        );
        assert!(adapter.contract(StreamingRequirements::NONE).is_err());
        adapter.install_test_preparation_seal(backend, 1).unwrap();
        let contract = adapter.contract(StreamingRequirements::NONE).unwrap();
        assert_eq!(contract.adapter_abi_revision, LFM25_AUDIO_TTS_ADAPTER_ABI);
        assert_eq!(contract.execution_profile.mode, ExecutionMode::Sequence);
        assert_eq!(
            contract.execution_profile.prefill_batch,
            NativeBatchMode::Static
        );
        assert_eq!(
            contract.execution_profile.decode_batch,
            NativeBatchMode::Continuous
        );
        assert_eq!(contract.stages.len(), 3);
        assert_eq!(contract.stages[0].name, "tts.prepare.lfm25_audio");
        assert_eq!(
            contract.stages[1].name,
            "tts.prefill.lfm25_audio.tensor_static"
        );
        assert_eq!(
            contract.stages[2].name,
            "tts.decode.lfm25_audio.tensor_continuous"
        );
        assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::None);
        assert_eq!(contract.stages[1].batch_mode, NativeBatchMode::Static);
        assert_eq!(contract.stages[2].batch_mode, NativeBatchMode::Continuous);
        assert_eq!(contract.stages[0].max_batch_size, 1);
        assert_eq!(contract.stages[1].max_batch_size, 8);
        assert_eq!(contract.stages[2].max_batch_size, 8);
        assert!(contract
            .stages
            .iter()
            .all(|stage| { stage.output_visibility == OutputVisibility::AfterQuantumCommit }));
    }
}

#[test]
fn lfm25_audio_asr_factory_publishes_sealed_retained_graph_and_atomic_long_form() {
    let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(8, 1).unwrap();
    let scalar = ScalarExecutionAdapterFactory;
    let factory = Lfm25AudioPhysicalAsrAdapterFactory;

    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        let metadata = *registry
            .require(CapabilityKind::Asr, ModelVariant::Lfm25Audio15BGguf)
            .unwrap();
        assert!(factory.supports(metadata, backend));
        assert!(!scalar.supports(metadata, backend));

        let adapter = Lfm25AudioAsrExecutionAdapter::new(
            ExecutionGroupId::new(71),
            ModelInstanceId::new(72),
            metadata,
            backend,
            8,
        );
        let missing = adapter.contract(StreamingRequirements::NONE).unwrap_err();
        assert!(missing.to_string().contains("preparation is sealed"));

        let long = adapter
            .contract(StreamingRequirements::NONE.with_asr_long_form(true))
            .unwrap();
        assert_eq!(long.stages.len(), 1);
        assert_eq!(long.stages[0].name, "asr.long_form.lfm25_audio.chunked");
        assert_eq!(long.stages[0].progress, StageProgressKind::Iterative);
        assert_eq!(
            long.stages[0].membership_safe_point,
            MembershipSafePoint::QuantumBoundary
        );
        assert_eq!(long.stages[0].shape_policy, StageShapePolicy::Exact);
        assert_eq!(
            long.stages[0].output_visibility,
            OutputVisibility::AfterQuantumCommit
        );

        adapter.install_test_preparation_seal(backend, 8).unwrap();
        let contract = adapter.contract(StreamingRequirements::NONE).unwrap();
        assert_eq!(contract.adapter_abi_revision, LFM25_AUDIO_ASR_ADAPTER_ABI);
        assert_eq!(contract.execution_profile.mode, ExecutionMode::Sequence);
        assert_eq!(contract.execution_profile.prefill, PrefillMode::Incremental);
        assert_eq!(
            contract.execution_profile.prefill_batch,
            NativeBatchMode::Static
        );
        assert_eq!(
            contract.execution_profile.decode_batch,
            NativeBatchMode::Continuous
        );
        assert_eq!(contract.stages.len(), 3);

        let preparation = &contract.stages[0];
        assert_eq!(preparation.name, "asr.encoder.lfm25_audio");
        assert_eq!(preparation.progress, StageProgressKind::Atomic);
        assert_eq!(preparation.batch_mode, NativeBatchMode::None);
        assert_eq!(preparation.max_batch_size, 1);
        assert_eq!(preparation.concurrency, ConcurrencyClass::Exclusive);
        assert_eq!(preparation.shape_policy, StageShapePolicy::Exact);
        assert_eq!(preparation.max_work_units, 1_024);
        assert_eq!(preparation.max_workspace_bytes, 96 * 1024 * 1024);

        let prefill = &contract.stages[1];
        assert_eq!(prefill.name, "asr.prefill.lfm25_audio.tensor_static");
        assert_eq!(prefill.progress, StageProgressKind::Iterative);
        assert_eq!(prefill.batch_mode, NativeBatchMode::Static);
        assert_eq!(prefill.max_batch_size, 8);
        assert_eq!(prefill.concurrency, ConcurrencyClass::Batchable);
        assert_eq!(prefill.shape_policy, StageShapePolicy::Ragged);
        assert_eq!(prefill.max_padding_basis_points, 0);

        let decode = &contract.stages[2];
        assert_eq!(decode.name, "asr.decode.lfm25_audio.continuous");
        assert_eq!(decode.progress, StageProgressKind::Iterative);
        assert_eq!(decode.batch_mode, NativeBatchMode::Continuous);
        assert_eq!(decode.max_batch_size, 8);
        assert_eq!(decode.concurrency, ConcurrencyClass::Batchable);
        assert_eq!(decode.shape_policy, StageShapePolicy::Ragged);
        // The model-derived per-row decode budget, for every row.
        assert_eq!(decode.max_workspace_bytes, 8 * 16 * 1024 * 1024);
        assert!(contract
            .stages
            .iter()
            .all(|stage| stage.output_visibility == OutputVisibility::AfterQuantumCommit));
    }
}

/// Granite Speech 5.0 runs on the scalar adapter, whose ASR graph is the
/// chunked atomic stage for every request: a long file dispatches one chunk
/// at a time (it used to hold the device for the whole file), and the normal
/// and long-form requirements seal the same graph, so the load keeps one
/// invocation pool. Nemotron's scalar ASR keeps its single-dispatch stage.
#[test]
fn granite_speech5_scalar_asr_dispatches_one_chunk_at_a_time() {
    let registry = RuntimeAdapterRegistry::built_in();
    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        let draft = LoadedModelBundleDraft::build(
            &registry,
            ExecutionGroupId::new(7),
            ModelInstanceId::new(1),
            ModelVariant::GraniteSpeech502B,
            backend,
        )
        .unwrap();
        let adapter = draft.capabilities.get(&CapabilityKind::Asr).unwrap();
        let normal = adapter.contract(StreamingRequirements::NONE).unwrap();
        let long = adapter
            .contract(StreamingRequirements::NONE.with_asr_long_form(true))
            .unwrap();
        for contract in [&normal, &long] {
            assert_eq!(contract.execution_profile.mode, ExecutionMode::Atomic);
            assert_eq!(contract.stages.len(), 1);
            let stage = &contract.stages[0];
            assert_eq!(stage.name, "asr.long_form.chunked", "{backend:?}");
            assert_eq!(stage.selector, StageWorkSelector::Atomic);
            assert_eq!(stage.progress, StageProgressKind::Iterative);
            assert_eq!(
                stage.membership_safe_point,
                MembershipSafePoint::QuantumBoundary
            );
        }
        assert_eq!(
            crate::kv::v2::stage_graph_fingerprint(&normal.stages).unwrap(),
            crate::kv::v2::stage_graph_fingerprint(&long.stages).unwrap(),
        );
    }
    let nemotron = LoadedModelBundleDraft::build(
        &registry,
        ExecutionGroupId::new(7),
        ModelInstanceId::new(2),
        ModelVariant::Nemotron35AsrStreaming06B,
        BackendKind::Cpu,
    )
    .unwrap();
    let scalar = nemotron
        .capabilities
        .get(&CapabilityKind::Asr)
        .unwrap()
        .contract(StreamingRequirements::NONE)
        .unwrap();
    assert_eq!(scalar.stages[0].name, "asr.scalar");
    assert_eq!(scalar.stages[0].progress, StageProgressKind::Atomic);
}

#[test]
fn whisper_normal_and_long_form_graphs_are_distinct_and_scalar() {
    let registry = RuntimeAdapterRegistry::built_in();
    let metadata = *registry
        .require(CapabilityKind::Asr, ModelVariant::WhisperLargeV3Turbo)
        .unwrap();
    let adapter = WhisperAsrExecutionAdapter::new(
        ExecutionGroupId::new(1),
        ModelInstanceId::new(2),
        metadata,
        BackendKind::Cpu,
        4,
    );
    adapter
        .audio_preparation
        .set(
            crate::models::architectures::whisper::asr::WhisperAudioPreparationStageSeal {
                backend: BackendKind::Cpu,
                dtype: "f32".into(),
                max_batch_size: 4,
                max_workspace_bytes: 1024,
            },
        )
        .unwrap();
    let normal = adapter.contract(StreamingRequirements::NONE).unwrap();
    assert_eq!(normal.execution_profile.mode, ExecutionMode::Sequence);
    assert_eq!(normal.execution_profile.max_batch_size, 4);
    assert_eq!(normal.execution_profile.decode_batch, NativeBatchMode::None);
    assert_eq!(normal.stages[0].name, "asr.encoder.whisper");
    assert_eq!(normal.stages[0].batch_mode, NativeBatchMode::Static);
    assert_eq!(
        normal.stages[1].selector,
        StageWorkSelector::SequencePrefill
    );
    assert_eq!(normal.stages[2].selector, StageWorkSelector::SequenceDecode);
    assert_eq!(normal.stages[2].batch_mode, NativeBatchMode::None);

    let long = adapter
        .contract(StreamingRequirements::NONE.with_asr_long_form(true))
        .unwrap();
    assert_eq!(long.execution_profile.mode, ExecutionMode::Atomic);
    assert_eq!(long.stages.len(), 1);
    // One chunk per dispatch: the job yields between chunks.
    assert_eq!(long.stages[0].name, "asr.long_form.chunked");
    assert_eq!(long.stages[0].selector, StageWorkSelector::Atomic);
    assert_eq!(long.stages[0].progress, StageProgressKind::Iterative);
    assert_eq!(
        long.stages[0].membership_safe_point,
        MembershipSafePoint::QuantumBoundary
    );
    // The load seals both graphs into one physical state spec.
    let config = candle_transformers::models::whisper::Config {
        num_mel_bins: 80,
        max_source_positions: 33,
        d_model: 16,
        encoder_attention_heads: 4,
        encoder_layers: 2,
        vocab_size: 128,
        max_target_positions: 65,
        decoder_attention_heads: 4,
        decoder_layers: 2,
        suppress_tokens: vec![],
    };
    crate::models::architectures::whisper::whisper_physical_state_spec(
        &config,
        candle_core::DType::F32,
        BackendKind::Cpu,
        &[&normal.stages, &long.stages],
    )
    .expect("the load authenticates the normal and chunked long-form graphs");
}

#[test]
fn parakeet_long_form_graph_is_one_atomic_stage_beside_the_sequence_graph() {
    // The diarization pipeline loads ASR with the long-form flag and
    // refuses a contract without an atomic stage; Parakeet must publish
    // that graph while ordinary requests keep the retained sequence graph.
    let registry = RuntimeAdapterRegistry::built_in();
    let metadata = *registry
        .require(CapabilityKind::Asr, ModelVariant::ParakeetTdt06BV3)
        .unwrap();
    let adapter = ParakeetAsrExecutionAdapter::new(
        ExecutionGroupId::new(1),
        ModelInstanceId::new(2),
        metadata,
        BackendKind::Cpu,
        4,
    );
    let normal = adapter.contract(StreamingRequirements::NONE).unwrap();
    assert_eq!(normal.execution_profile.mode, ExecutionMode::Sequence);
    assert_eq!(normal.stages[0].name, "asr.encoder.parakeet");
    assert!(normal
        .stages
        .iter()
        .all(|stage| stage.selector != StageWorkSelector::Atomic));

    let long = adapter
        .contract(StreamingRequirements::NONE.with_asr_long_form(true))
        .unwrap();
    assert_eq!(long.execution_profile.mode, ExecutionMode::Atomic);
    assert_eq!(long.execution_profile.max_batch_size, 1);
    assert_eq!(long.stages.len(), 1);
    assert_eq!(long.stages[0].name, "asr.long_form.atomic");
    assert_eq!(long.stages[0].selector, StageWorkSelector::Atomic);
    assert_eq!(long.stages[0].batch_mode, NativeBatchMode::None);
    assert!(long
        .adapter_binding()
        .unwrap()
        .stage_for_work(&crate::engine::WorkUnit::AtomicJob {
            kind: "diarization.transcribe".into(),
        })
        .is_ok());
}

/// A Granite Speech ASR adapter with its retained sequence route on. The
/// route is dormant until Granite resumes prefill (GRN-8); these tests keep
/// its graph honest for that day.
fn granite_retained_route_adapter(
    metadata: AdapterMetadata,
    backend: BackendKind,
    max_batch_size: usize,
) -> GraniteSpeechAsrExecutionAdapter {
    let mut adapter = GraniteSpeechAsrExecutionAdapter::new(
        ExecutionGroupId::new(1),
        ModelInstanceId::new(2),
        metadata,
        backend,
        max_batch_size,
    );
    adapter.retained_route = true;
    adapter
}

#[test]
fn granite_speech_without_resumable_prefill_offers_only_its_atomic_graph() {
    let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(8, 1).unwrap();
    let draft = LoadedModelBundleDraft::build(
        &registry,
        ExecutionGroupId::new(1),
        ModelInstanceId::new(2),
        ModelVariant::GraniteSpeech412BPlus,
        BackendKind::Metal,
    )
    .unwrap();
    // Every selectable graph is known when the bundle is drafted, before the
    // weights load: none waits on the preparation seal.
    let contracts = draft.execution_contracts(CapabilityKind::Asr).unwrap();
    let atomic = draft.capabilities[&CapabilityKind::Asr]
        .contract(StreamingRequirements::NONE.with_asr_long_form(true))
        .unwrap();
    for contract in &contracts {
        assert_eq!(contract.execution_profile.mode, ExecutionMode::Atomic);
        assert_eq!(contract.execution_profile.cache_mode, CacheMode::None);
        assert!(contract.execution_profile.cache_namespace.is_none());
        assert_eq!(contract.execution_profile.kv_dtype, "none");
        assert_eq!(contract.stages.len(), 1);
        assert_eq!(contract.stages[0].selector, StageWorkSelector::Atomic);
        assert!(contract.stages.iter().all(|stage| !matches!(
            stage.selector,
            StageWorkSelector::SequencePrefill | StageWorkSelector::SequenceDecode
        )));
    }
    // An ordinary requirement re-seals the long-form graph exactly.
    let normal = draft.capabilities[&CapabilityKind::Asr]
        .contract(StreamingRequirements::NONE)
        .unwrap();
    assert_eq!(
        stage_graph_fingerprint(&normal.stages).unwrap(),
        stage_graph_fingerprint(&atomic.stages).unwrap()
    );
    // The model's preparation seal still installs, and changes no graph.
    install_test_preparation_seals(&draft, BackendKind::Metal, 8);
    assert_eq!(
        draft
            .execution_contracts(CapabilityKind::Asr)
            .unwrap()
            .len(),
        contracts.len()
    );
    assert_eq!(
        stage_graph_fingerprint(
            &draft.capabilities[&CapabilityKind::Asr]
                .contract(StreamingRequirements::NONE)
                .unwrap()
                .stages
        )
        .unwrap(),
        stage_graph_fingerprint(&atomic.stages).unwrap()
    );
}

#[test]
fn granite_speech_normal_graph_has_static_preparation_continuous_decode_and_atomic_fallback() {
    let registry = RuntimeAdapterRegistry::built_in();
    let metadata = *registry
        .require(CapabilityKind::Asr, ModelVariant::GraniteSpeech412BPlus)
        .unwrap();
    let adapter = granite_retained_route_adapter(metadata, BackendKind::Cpu, 8);
    adapter
        .install_test_preparation_seal(BackendKind::Cpu, 8)
        .unwrap();
    let sealed = adapter.seal.get().unwrap();
    assert_eq!(
        sealed.preparation_max_batch_materialized_tensor_elements,
        8 * 2_000_000
    );

    let normal = adapter.contract(StreamingRequirements::NONE).unwrap();
    assert_eq!(normal.adapter_abi_revision, GRANITE_SPEECH_ASR_ADAPTER_ABI);
    assert_eq!(normal.execution_profile.mode, ExecutionMode::Sequence);
    assert_eq!(normal.execution_profile.prefill, PrefillMode::Incremental);
    assert_eq!(
        normal.execution_profile.decode_batch,
        NativeBatchMode::Continuous
    );
    assert_eq!(normal.execution_profile.max_batch_size, 8);
    assert_eq!(
        normal.execution_profile.concurrency,
        ConcurrencyClass::Batchable
    );
    assert_eq!(normal.stages.len(), 3);
    assert_eq!(
        normal.stages[0].selector,
        StageWorkSelector::PreSequencePreparation
    );
    assert_eq!(
        normal.stages[1].selector,
        StageWorkSelector::SequencePrefill
    );
    assert_eq!(normal.stages[2].selector, StageWorkSelector::SequenceDecode);
    assert_eq!(normal.stages[0].name, "asr.encoder.granite_speech");
    assert_eq!(normal.stages[0].batch_mode, NativeBatchMode::Static);
    assert_eq!(normal.stages[0].shape_policy, StageShapePolicy::Padded);
    assert_eq!(normal.stages[0].max_batch_size, 8);
    assert_eq!(normal.stages[0].concurrency, ConcurrencyClass::Batchable);
    assert_eq!(normal.stages[0].max_work_units, 8 * 10_000);
    assert_eq!(normal.stages[0].workspace_per_row_bytes, 0);
    assert_eq!(normal.stages[0].max_workspace_bytes, 8 * 1024 * 1024 * 1024);
    assert_eq!(normal.stages[1].batch_mode, NativeBatchMode::None);
    assert_eq!(normal.stages[1].max_batch_size, 1);
    assert_eq!(normal.stages[1].concurrency, ConcurrencyClass::Exclusive);
    assert_eq!(normal.stages[2].batch_mode, NativeBatchMode::Continuous);
    assert_eq!(normal.stages[2].max_batch_size, 8);
    assert_eq!(normal.stages[2].concurrency, ConcurrencyClass::Batchable);
    assert_eq!(normal.stages[2].shape_policy, StageShapePolicy::Ragged);
    assert_eq!(normal.stages[2].max_work_units, 8);
    let decode_workspace = continuous_asr_workspace_per_row_bytes(8_192).unwrap();
    assert_eq!(normal.stages[2].max_workspace_bytes, 8 * decode_workspace);
    assert_eq!(normal.stages[2].workspace_per_row_bytes, decode_workspace);
    assert!(normal.stages[0].max_work_units > 0);
    assert!(normal.stages[0].max_workspace_bytes > 0);

    let long = adapter
        .contract(StreamingRequirements::NONE.with_asr_long_form(true))
        .unwrap();
    assert_eq!(long.execution_profile.mode, ExecutionMode::Atomic);
    assert_eq!(long.stages.len(), 1);
    assert_eq!(long.stages[0].selector, StageWorkSelector::Atomic);
    assert_eq!(long.execution_profile.cache_mode, CacheMode::None);
}

#[test]
fn granite_speech_native_batch_contract_is_backend_truthful() {
    let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(8, 1).unwrap();
    let metadata = *registry
        .require(CapabilityKind::Asr, ModelVariant::GraniteSpeech412BPlus)
        .unwrap();

    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        let adapter = granite_retained_route_adapter(metadata, backend, 8);
        adapter.install_test_preparation_seal(backend, 8).unwrap();
        let contract = adapter.contract(StreamingRequirements::NONE).unwrap();

        assert_eq!(contract.execution_profile.backend, backend);
        assert_eq!(
            contract.execution_profile.prefill_batch,
            NativeBatchMode::None
        );
        assert_eq!(
            contract.execution_profile.decode_batch,
            NativeBatchMode::Continuous
        );
        assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::Static);
        assert_eq!(contract.stages[0].shape_policy, StageShapePolicy::Padded);
        assert_eq!(contract.stages[0].max_batch_size, 8);
        assert_eq!(contract.stages[0].workspace_per_row_bytes, 0);
        assert_eq!(
            contract.stages[0].max_workspace_bytes,
            8 * 1024 * 1024 * 1024
        );
        assert_eq!(contract.stages[1].concurrency, ConcurrencyClass::Exclusive);
        assert_eq!(contract.stages[2].concurrency, ConcurrencyClass::Batchable);
        let decode_workspace = continuous_asr_workspace_per_row_bytes(8_192).unwrap();
        assert_eq!(contract.stages[2].workspace_per_row_bytes, decode_workspace);
        assert_eq!(contract.stages[2].max_workspace_bytes, 8 * decode_workspace);

        // Long-form runs one chunk per dispatch.
        let long = adapter
            .contract(StreamingRequirements::NONE.with_asr_long_form(true))
            .unwrap();
        assert_eq!(long.stages[0].name, "asr.granite.long_form.chunked");
        assert_eq!(long.stages[0].progress, StageProgressKind::Iterative);
    }
}

#[test]
fn granite_speech_width_one_keeps_scalar_preparation() {
    let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(1, 1).unwrap();
    let metadata = *registry
        .require(CapabilityKind::Asr, ModelVariant::GraniteSpeech412BPlus)
        .unwrap();
    let adapter = granite_retained_route_adapter(metadata, BackendKind::Cpu, 1);
    adapter
        .install_test_preparation_seal(BackendKind::Cpu, 1)
        .unwrap();

    let contract = adapter.contract(StreamingRequirements::NONE).unwrap();
    assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::None);
    assert_eq!(contract.stages[0].shape_policy, StageShapePolicy::Exact);
    assert_eq!(contract.stages[0].max_batch_size, 1);
    assert_eq!(contract.stages[0].concurrency, ConcurrencyClass::Exclusive);
    assert_eq!(contract.stages[1].batch_mode, NativeBatchMode::None);
    assert_eq!(contract.stages[2].batch_mode, NativeBatchMode::Continuous);
    assert_eq!(contract.stages[2].max_batch_size, 1);
}

#[test]
fn granite_speech_normal_graph_fails_closed_before_preparation_is_sealed() {
    let registry = RuntimeAdapterRegistry::built_in();
    let metadata = *registry
        .require(CapabilityKind::Asr, ModelVariant::GraniteSpeech412BPlus)
        .unwrap();
    let adapter = granite_retained_route_adapter(metadata, BackendKind::Cpu, 8);

    let error = adapter.contract(StreamingRequirements::NONE).unwrap_err();
    assert!(error.to_string().contains("preparation is sealed"));
    assert!(adapter
        .contract(StreamingRequirements::NONE.with_asr_long_form(true))
        .is_ok());
}

#[test]
fn granite_speech_speaker_attribution_authenticates_as_pipeline() {
    let registry = RuntimeAdapterRegistry::built_in();
    let metadata = *registry
        .require(
            CapabilityKind::SpeakerAttributedAsr,
            ModelVariant::GraniteSpeech412BPlus,
        )
        .unwrap();
    let adapter = ScalarExecutionAdapter::new(
        ExecutionGroupId::new(1),
        ModelInstanceId::new(2),
        metadata,
        BackendKind::Cpu,
        1,
    );
    let contract = adapter.contract(StreamingRequirements::NONE).unwrap();
    assert_eq!(contract.stages.len(), 1);
    assert_eq!(
        contract.stages[0].selector,
        StageWorkSelector::Pipeline { ordinal: None }
    );
    assert_eq!(contract.stages[0].batch_mode, NativeBatchMode::None);
    assert_eq!(contract.stages[0].shape_policy, StageShapePolicy::Exact);
}

#[test]
fn vibevoice_normal_graph_has_scalar_preparation_and_ragged_decode() {
    let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(4, 1).unwrap();
    let metadata = *registry
        .require(CapabilityKind::Asr, ModelVariant::VibeVoiceAsr)
        .unwrap();
    let adapter = VibeVoiceAsrExecutionAdapter::new(
        ExecutionGroupId::new(1),
        ModelInstanceId::new(2),
        metadata,
        BackendKind::Cpu,
        4,
    );
    adapter
        .seal
        .set(VibeVoiceAsrExecutionSeal {
            preparation:
                crate::models::architectures::vibevoice::asr::VibeVoiceAsrPreparationStageSeal {
                    backend: BackendKind::Cpu,
                    dtype: "f32".into(),
                    max_work_units: 1_500,
                    max_workspace_bytes: 64 * 1024 * 1024,
                },
            prefill_max_batch_work_units: 4 * 1_500,
            prefill_max_batch_workspace_bytes: 4 * 64 * 1024 * 1024,
            decode_workspace_per_row_bytes: 8_192,
        })
        .unwrap();

    let normal = adapter.contract(StreamingRequirements::NONE).unwrap();
    assert_eq!(normal.adapter_abi_revision, VIBEVOICE_ASR_ADAPTER_ABI);
    assert_eq!(normal.execution_profile.mode, ExecutionMode::Sequence);
    assert_eq!(
        normal.execution_profile.prefill_batch,
        NativeBatchMode::Static
    );
    assert_eq!(
        normal.execution_profile.decode_batch,
        NativeBatchMode::Continuous
    );
    assert_eq!(normal.stages.len(), 3);
    assert_eq!(normal.stages[0].name, "asr.encoder.vibevoice");
    assert_eq!(normal.stages[0].batch_mode, NativeBatchMode::None);
    assert_eq!(normal.stages[0].shape_policy, StageShapePolicy::Exact);
    assert_eq!(normal.stages[0].concurrency, ConcurrencyClass::Exclusive);
    assert_eq!(normal.stages[0].max_batch_size, 1);
    assert_eq!(normal.stages[0].max_work_units, 1_500);
    assert_eq!(normal.stages[1].name, "asr.prefill.tensor_static");
    assert_eq!(normal.stages[1].batch_mode, NativeBatchMode::Static);
    assert_eq!(normal.stages[1].shape_policy, StageShapePolicy::Padded);
    assert_eq!(normal.stages[1].concurrency, ConcurrencyClass::Batchable);
    assert_eq!(normal.stages[1].max_batch_size, 4);
    assert_eq!(normal.stages[1].max_work_units, 4 * 1_500);
    assert_eq!(normal.stages[1].workspace_per_row_bytes, 64 * 1024 * 1024);
    assert_eq!(
        normal.stages[1].retained_state_selections.as_deref(),
        Some(
            [ClockedStateSelection::new(
                crate::models::architectures::vibevoice::VIBEVOICE_ASR_TOKENIZER_GROUP,
                StateClock::AudioSamples,
            )
            .unwrap()]
            .as_slice()
        )
    );
    assert_eq!(normal.stages[1].max_workspace_bytes, 4 * 64 * 1024 * 1024);
    assert_eq!(normal.stages[2].name, "asr.decode.tensor_continuous");
    assert_eq!(
        normal.stages[2].retained_state_selections.as_deref(),
        Some(&[][..])
    );
    assert_eq!(normal.stages[2].batch_mode, NativeBatchMode::Continuous);
    assert_eq!(normal.stages[2].shape_policy, StageShapePolicy::Ragged);
    assert_eq!(normal.stages[2].max_batch_size, 4);
    let decode_workspace = continuous_asr_workspace_per_row_bytes(8_192).unwrap();
    assert_eq!(normal.stages[2].workspace_per_row_bytes, decode_workspace);
    assert_eq!(normal.stages[2].max_workspace_bytes, 4 * decode_workspace);

    let legacy = adapter
        .contract(StreamingRequirements::NONE.with_asr_long_form(true))
        .unwrap();
    assert_eq!(legacy.execution_profile.mode, ExecutionMode::Atomic);
    assert_eq!(legacy.stages.len(), 1);
    assert_eq!(legacy.stages[0].name, "asr.scalar");
    assert_eq!(legacy.stages[0].selector, StageWorkSelector::Atomic);
}

#[test]
fn vibevoice_prefill_keeps_width_one_scalar_fallback() {
    let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(1, 1).unwrap();
    let metadata = *registry
        .require(CapabilityKind::Asr, ModelVariant::VibeVoiceAsr)
        .unwrap();
    let adapter = VibeVoiceAsrExecutionAdapter::new(
        ExecutionGroupId::new(1),
        ModelInstanceId::new(2),
        metadata,
        BackendKind::Cpu,
        1,
    );
    adapter
        .install_test_preparation_seal(BackendKind::Cpu, 1)
        .unwrap();

    let contract = adapter.contract(StreamingRequirements::NONE).unwrap();

    assert_eq!(
        contract.execution_profile.prefill_batch,
        NativeBatchMode::None
    );
    assert_eq!(contract.stages[1].batch_mode, NativeBatchMode::None);
    assert_eq!(contract.stages[1].shape_policy, StageShapePolicy::Exact);
    assert_eq!(contract.stages[1].concurrency, ConcurrencyClass::Exclusive);
    assert_eq!(contract.stages[1].max_batch_size, 1);
}

/// VXT-1: the registry advertised DirectModel while the loaded adapter
/// publishes only engine sequence stages, so the direct route's
/// `AtomicJob` matched no stage and every request failed. The contract
/// target must be the route `RuntimeService` takes, every unit of work
/// that route schedules must find its stage, and the retained context
/// is bounded by one request, not the LM's 128,000 positions (VXT-2).
#[test]
fn voxtral_tts_contract_target_is_the_route_taken_and_serves_its_scheduled_work() {
    use crate::engine::{InputRange, SequencePhase, WorkUnit};
    use crate::kv::v2::{
        test_contract, CheckpointPolicy, InvocationWorkspaceDomain, InvocationWorkspaceSet,
        RetainedStateCapability, StateDomainSpec, StateScope,
    };
    use crate::models::architectures::voxtral::tts::model::voxtral_tts_retained_max_tokens;
    use crate::models::architectures::voxtral::tts::VOXTRAL_TTS_STATE_CONTEXT_TOKENS;
    let variant = ModelVariant::Voxtral4BTts2603;
    let registry = RuntimeAdapterRegistry::built_in();
    let metadata = *registry.require(CapabilityKind::Tts, variant).unwrap();
    assert_eq!(
        metadata.execution_target,
        crate::runtime::tts::tts_route_target(variant, false)
    );
    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        let contract = VoxtralTtsExecutionAdapter::new(
            ExecutionGroupId::new(91),
            ModelInstanceId::new(92),
            metadata,
            backend,
            2,
        )
        .contract(StreamingRequirements::NONE)
        .unwrap();
        assert_eq!(
            contract.metadata.execution_target,
            metadata.execution_target
        );
        let binding = contract.adapter_binding().unwrap();
        let sequence = |phase| WorkUnit::SequenceStep {
            phase,
            input: InputRange::new(0, 8).unwrap(),
            max_output_steps: 1,
            auxiliary_state: None,
        };
        for work in [
            WorkUnit::PreSequencePreparation {
                kind: "tts.prepare.voxtral".into(),
            },
            sequence(SequencePhase::Prefill),
            sequence(SequencePhase::Decode),
            WorkUnit::SequenceFinalize {
                max_output_steps: 1,
            },
        ] {
            binding
                .stage_for_work(&work)
                .unwrap_or_else(|error| panic!("{backend:?} {work:?}: {error}"));
        }
        assert!(binding
            .stage_for_work(&WorkUnit::AtomicJob { kind: "tts".into() })
            .is_err());

        let mut invocation = test_contract();
        for domain in &mut invocation.domains {
            let StateDomainSpec::PagedAttention(domain) = domain else {
                panic!("test contract is paged");
            };
            domain.header.scope = StateScope::Invocation;
            domain.header.checkpoint = CheckpointPolicy::None;
        }
        let spec = crate::models::architectures::voxtral::voxtral_tts_physical_state_spec(
            &[contract.stages.as_ref()],
            invocation,
            voxtral_tts_retained_max_tokens(128_000),
        )
        .unwrap();
        assert_eq!(spec.retained_max_tokens, VOXTRAL_TTS_STATE_CONTEXT_TOKENS);
        assert!(matches!(
            spec.descriptor.retained,
            RetainedStateCapability::Managed { .. }
        ));
        let InvocationWorkspaceSet::Bounded { profiles } = &spec.descriptor.invocation else {
            panic!("bounded scratch profiles");
        };
        assert!(profiles
            .iter()
            .flat_map(|profile| profile.stages.iter())
            .flat_map(|stage| stage.domains.iter())
            .all(|domain| matches!(domain, InvocationWorkspaceDomain::Scratch { .. })));
    }
}

#[test]
fn voxtral_tts_factory_publishes_retained_batched_graph_on_all_backends() {
    let registry = RuntimeAdapterRegistry::built_in_with_execution_limits(8, 1).unwrap();
    let metadata = *registry
        .require(CapabilityKind::Tts, ModelVariant::Voxtral4BTts2603)
        .unwrap();
    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        let adapter = VoxtralTtsExecutionAdapter::new(
            ExecutionGroupId::new(91),
            ModelInstanceId::new(92),
            metadata,
            backend,
            8,
        );
        let contract = adapter.contract(StreamingRequirements::NONE).unwrap();
        assert_eq!(contract.adapter_abi_revision, VOXTRAL_TTS_ADAPTER_ABI);
        assert_eq!(
            contract.execution_profile.cache_mode,
            CacheMode::ExternalPaged
        );
        assert_eq!(
            contract.execution_profile.prefill_batch,
            NativeBatchMode::Static
        );
        assert_eq!(
            contract.execution_profile.decode_batch,
            NativeBatchMode::Continuous
        );
        assert_eq!(contract.stages[1].shape_policy, StageShapePolicy::Ragged);
        assert_eq!(contract.stages[1].max_padding_basis_points, 0);
        assert_eq!(contract.stages.len(), 4);
        assert_eq!(
            contract.stages[3].selector,
            StageWorkSelector::SequenceFinalize
        );
        assert_eq!(contract.stages[3].batch_mode, NativeBatchMode::None);
        assert_eq!(contract.stages[3].max_batch_size, 1);
        assert!(contract
            .stages
            .iter()
            .all(|stage| { stage.output_visibility == OutputVisibility::AfterQuantumCommit }));
    }
}
