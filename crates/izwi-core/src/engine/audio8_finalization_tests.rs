use super::*;
use crate::engine::{ExecutionAdapterBinding, StageDescriptor, StageWorkSelector};
use crate::models::architectures::audio8::{
    Audio8GenerationParams, Audio8PreparedArtifact, Audio8TtsModel,
};
use crate::models::registry::Audio8TtsModelLease;
use crate::runtime::{CapabilityKind, LoadedModelBundleDraft, RuntimeAdapterRegistry};

fn prepared_request(frames: usize) -> EngineCoreRequest {
    let mut request = EngineCoreRequest::tts("Tell me something funny, I stay laughing")
        .with_model_variant(ModelVariant::Audio8TtsRealisticAir);
    request
        .install_audio8_tts_execution_model(
            ModelVariant::Audio8TtsRealisticAir,
            Audio8TtsModelLease::for_test(Audio8TtsModel::for_test()),
            Audio8PreparedArtifact::test_prompt(11, 32),
            Audio8GenerationParams {
                max_frames: frames,
                ..Default::default()
            },
            8192,
        )
        .unwrap();
    request.validate_execution_preparation().unwrap();
    request
}

fn loaded_binding(backend: BackendKind) -> (ExecutionAdapterBinding, StageDescriptor) {
    let draft = LoadedModelBundleDraft::build(
        &RuntimeAdapterRegistry::built_in(),
        ExecutionGroupId::new(1),
        ModelInstanceId::new(2),
        ModelVariant::Audio8TtsRealisticAir,
        backend,
    )
    .unwrap();
    let contract = draft
        .execution_contracts(CapabilityKind::Tts)
        .unwrap()
        .into_iter()
        .find(|contract| contract.execution_profile.mode == ExecutionMode::Sequence)
        .unwrap();
    let stage = contract
        .stages
        .iter()
        .find(|stage| stage.selector == StageWorkSelector::SequenceAudioDecode)
        .unwrap()
        .clone();
    (contract.adapter_binding().unwrap(), stage)
}

fn finalize_cost(
    request: &EngineCoreRequest,
    stage: Option<&StageDescriptor>,
    backend: BackendKind,
) -> Result<WorkCost> {
    EngineCore::work_cost(
        request,
        &WorkUnit::SequenceAudioDecode { max_frames: 1 },
        stage,
        backend,
    )
}

/// A codec row's sealed share of its audio quantum, never below its own
/// history and PCM.
fn quantum_row_bytes(stage: &StageDescriptor, frames: usize) -> u64 {
    stage.workspace_per_row_bytes.max(
        crate::models::architectures::audio8::codec::streaming_row_workspace_bytes(frames).unwrap(),
    )
}

#[test]
fn audio8_preparation_binds_then_prices_scalar_codec_on_each_backend() {
    for backend in [BackendKind::Cpu, BackendKind::Cuda, BackendKind::Metal] {
        for frames in [1, 512, ModelVariant::AUDIO8_MAX_OUTPUT_FRAMES] {
            let mut request = prepared_request(frames);
            let (binding, stage) = loaded_binding(backend);
            assert_eq!(stage.batch_mode, NativeBatchMode::None);
            assert!(request.execution_adapter_binding().is_none());
            assert!(request.prepared_stage_cost(stage.id).is_none());
            request.bind_execution_adapter(binding).unwrap();
            request.validate_execution_preparation().unwrap();
            let cost = finalize_cost(&request, Some(&stage), backend).unwrap();
            // A row's sealed share of its audio quantum.
            let bytes = quantum_row_bytes(&stage, 1);
            let mut expected = ResourceVector::zero();
            match backend {
                BackendKind::Cpu => expected.host_bytes = ResourceAmount::Known(bytes),
                BackendKind::Cuda => expected.device_bytes = ResourceAmount::Known(bytes),
                BackendKind::Metal => expected.unified_bytes = ResourceAmount::Known(bytes),
            }
            assert_eq!(cost, WorkCost::with_workspace(1, 1, expected));
            assert!(bytes <= stage.max_workspace_bytes);
            assert!(request.prepared_stage_cost(stage.id).is_none());
        }
    }
}

#[test]
fn audio8_codec_cost_rejects_missing_binding_and_foreign_stage() {
    let mut request = prepared_request(512);
    let (binding, stage) = loaded_binding(BackendKind::Cuda);
    assert!(finalize_cost(&request, Some(&stage), BackendKind::Cuda).is_err());
    request.bind_execution_adapter(binding.clone()).unwrap();
    assert!(finalize_cost(&request, None, BackendKind::Cuda).is_err());
    let foreign_stage = binding
        .stages
        .iter()
        .find(|stage| stage.selector == StageWorkSelector::SequenceDecode)
        .unwrap();
    assert!(finalize_cost(&request, Some(foreign_stage), BackendKind::Cuda).is_err());
    let mut changed_stage = stage.clone();
    changed_stage.name.push_str(".foreign");
    assert!(finalize_cost(&request, Some(&changed_stage), BackendKind::Cuda).is_err());
}

#[test]
fn audio8_codec_cost_rejects_stale_prepared_geometry() {
    let mut request = prepared_request(512);
    let (binding, stage) = loaded_binding(BackendKind::Cuda);
    request.bind_execution_adapter(binding).unwrap();
    request.params.max_tokens = 1;
    assert!(finalize_cost(&request, Some(&stage), BackendKind::Cuda).is_err());
}

#[test]
fn audio8_codec_cost_cannot_exceed_loaded_stage_ceiling() {
    let mut request = prepared_request(512);
    let (mut binding, mut stage) = loaded_binding(BackendKind::Cuda);
    stage.max_workspace_bytes = 1;
    stage.workspace_per_row_bytes = 1;
    binding.stages = binding
        .stages
        .iter()
        .map(|entry| {
            if entry.id == stage.id {
                stage.clone()
            } else {
                entry.clone()
            }
        })
        .collect::<Vec<_>>()
        .into();
    request.bind_execution_adapter(binding).unwrap();
    assert!(finalize_cost(&request, Some(&stage), BackendKind::Cuda).is_err());
}

#[test]
fn audio8_scalar_codec_remains_invalid_as_a_prepared_tensor_stage() {
    let mut request = prepared_request(512);
    let (binding, stage) = loaded_binding(BackendKind::Cpu);
    request.bind_execution_adapter(binding).unwrap();
    let error = request
        .install_prepared_stage_cost(stage.id, WorkCost::new(1, 1, 1))
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("invalid prepared tensor-stage cost"));
}

#[test]
fn audio8_codec_cost_rejects_unsupported_frame_and_work_bounds() {
    let (binding, stage) = loaded_binding(BackendKind::Cpu);
    let mut request = prepared_request(ModelVariant::AUDIO8_MAX_OUTPUT_FRAMES + 1);
    request.bind_execution_adapter(binding.clone()).unwrap();
    assert!(finalize_cost(&request, Some(&stage), BackendKind::Cpu).is_err());
    let mut request = prepared_request(128);
    request.bind_execution_adapter(binding).unwrap();
    assert!(EngineCore::work_cost(
        &request,
        &WorkUnit::SequenceAudioDecode {
            max_frames: crate::models::architectures::audio8::AUDIO8_CODEC_QUANTUM_FRAMES + 1,
        },
        Some(&stage),
        BackendKind::Cpu,
    )
    .is_err());
}

#[test]
fn audio8_decode_and_other_models_keep_generic_work_cost() {
    let mut request = prepared_request(128);
    let (binding, _) = loaded_binding(BackendKind::Cpu);
    let stage = binding
        .stages
        .iter()
        .find(|stage| stage.selector == StageWorkSelector::SequenceDecode)
        .unwrap()
        .clone();
    request.bind_execution_adapter(binding).unwrap();
    let work = WorkUnit::SequenceStep {
        phase: SequencePhase::Decode,
        input: crate::engine::InputRange::new(0, 1).unwrap(),
        max_output_steps: 1,
        auxiliary_state: None,
    };
    let cost = EngineCore::work_cost(&request, &work, Some(&stage), BackendKind::Cpu).unwrap();
    assert_eq!(
        cost,
        WorkCost::new(
            1,
            1,
            stage.workspace_per_row_bytes + stage.workspace_per_work_unit_bytes
        )
    );
    let request = EngineCoreRequest::tts("Other model").with_model_variant(ModelVariant::Kokoro82M);
    let cost = finalize_cost(&request, None, BackendKind::Cpu).unwrap();
    assert_eq!(cost, WorkCost::new(1, 1, 0));
}

#[test]
fn audio8_exact_prompt_fits_admission_generation_and_codec_budget() {
    let mut request = EngineCoreRequest::tts("A context-fitted request")
        .with_model_variant(ModelVariant::Audio8TtsRealisticAir);
    request
        .install_audio8_tts_execution_model(
            ModelVariant::Audio8TtsRealisticAir,
            Audio8TtsModelLease::for_test(Audio8TtsModel::for_test()),
            Audio8PreparedArtifact::test_prompt(11, 224),
            Audio8GenerationParams {
                max_frames: 31,
                ..Default::default()
            },
            256,
        )
        .unwrap();
    request.validate_execution_preparation().unwrap();
    assert_eq!(request.params.max_tokens, 31);
    assert_eq!(
        request
            .audio8_tts_generation_params_for_executor()
            .unwrap()
            .unwrap()
            .max_frames,
        31
    );
    let (binding, stage) = loaded_binding(BackendKind::Cuda);
    request.bind_execution_adapter(binding).unwrap();
    request.validate_execution_preparation().unwrap();
    let cost = finalize_cost(&request, Some(&stage), BackendKind::Cuda).unwrap();
    let mut workspace = ResourceVector::zero();
    workspace.device_bytes = ResourceAmount::Known(quantum_row_bytes(&stage, 1));
    assert_eq!(cost, WorkCost::with_workspace(1, 1, workspace));
}

#[test]
fn audio8_terminal_flush_never_reserves_whole_waveform_codec_work() {
    let mut request = prepared_request(ModelVariant::AUDIO8_MAX_OUTPUT_FRAMES);
    let (binding, _) = loaded_binding(BackendKind::Cuda);
    let stage = binding
        .stages
        .iter()
        .find(|s| s.selector == StageWorkSelector::SequenceFinalize)
        .unwrap()
        .clone();
    request.bind_execution_adapter(binding).unwrap();
    let cost = EngineCore::work_cost(
        &request,
        &WorkUnit::SequenceFinalize {
            max_output_steps: 1,
        },
        Some(&stage),
        BackendKind::Cuda,
    )
    .unwrap();
    assert_eq!(cost, WorkCost::with_workspace(1, 1, ResourceVector::zero()));
    assert_eq!(stage.max_workspace_bytes, 0);
    assert!(EngineCore::work_cost(
        &request,
        &WorkUnit::SequenceFinalize {
            max_output_steps: 2
        },
        Some(&stage),
        BackendKind::Cuda
    )
    .is_err());
}

#[test]
fn audio8_exact_preparation_rejects_output_shrink_without_installing_artifact() {
    let mut request = EngineCoreRequest::tts("An oversized segment")
        .with_model_variant(ModelVariant::Audio8TtsRealisticAir);
    let error = request
        .install_audio8_tts_execution_model(
            ModelVariant::Audio8TtsRealisticAir,
            Audio8TtsModelLease::for_test(Audio8TtsModel::for_test()),
            Audio8PreparedArtifact::test_prompt(11, 224),
            Audio8GenerationParams {
                max_frames: 512,
                ..Default::default()
            },
            256,
        )
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("segment does not fit effective context"));
    assert!(request
        .prepared_audio8_tts_artifact_for_executor()
        .unwrap()
        .is_none());
}
