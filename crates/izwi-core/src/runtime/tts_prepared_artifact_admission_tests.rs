//! A TTS preparation that leaves conditioning resident on the device must
//! price it into the request's execution admission. On CUDA every Chatterbox,
//! VibeVoice, Voxtral and LFM2.5-Audio request was refused with "resource
//! resize would shrink authorization below materialized usage" because the
//! execution estimate priced only the request input.
use super::*;
use crate::catalog::ModelVariant;
use crate::runtime::adapters::{CapabilityKind, LoadedModelBundleDraft, RuntimeAdapterRegistry};
use crate::runtime::service::{add_prepared_tts_artifact_to_admission, fish_s2_artifact_resources};

const INPUT: u64 = 4096;
const CONDITIONING: u64 = 3 * 1024 * 1024;
const WORKSPACE: u64 = 64 * 1024 * 1024;

#[derive(Debug)]
struct Capacity(ResourceVector);
impl PhysicalCapacityProvider for Capacity {
    fn snapshot(&self) -> PhysicalCapacitySnapshot {
        PhysicalCapacitySnapshot {
            capacity: self.0,
            available: self.0,
            source: CapacitySource::Test,
        }
    }
}

fn coordinator(backend: BackendKind) -> Arc<InferenceCoordinator> {
    let capacity = match backend {
        BackendKind::Cpu => ResourceVector {
            host_bytes: ResourceAmount::Known(1 << 32),
            ..ResourceVector::zero()
        },
        BackendKind::Metal => ResourceVector {
            unified_bytes: ResourceAmount::Known(1 << 32),
            ..ResourceVector::zero()
        },
        BackendKind::Cuda => ResourceVector {
            host_bytes: ResourceAmount::Known(1 << 32),
            device_bytes: ResourceAmount::Known(1 << 32),
            ..ResourceVector::zero()
        },
    };
    Arc::new(InferenceCoordinator::with_resource_authority(
        backend,
        1,
        4,
        Arc::new(ResourceAuthority::new(Arc::new(Capacity(capacity)))),
    ))
}

/// Host bytes plus resident tensors in the backend's own domains.
fn artifact_resources(backend: BackendKind, host: u64, accelerator: u64) -> ResourceVector {
    let mut resources = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => resources.host_bytes = ResourceAmount::Known(host + accelerator),
        BackendKind::Metal => resources.unified_bytes = ResourceAmount::Known(host + accelerator),
        BackendKind::Cuda => {
            resources.host_bytes = ResourceAmount::Known(host);
            resources.device_bytes = ResourceAmount::Known(accelerator);
        }
    }
    resources
}

fn spec(resources: ResourceVector) -> JobSpec {
    JobSpec {
        request_id: "tts-prepared-artifact".into(),
        lane: CoordinatorLane::Atomic,
        priority: Priority::default(),
        workload_class: WorkloadClass::Interactive,
        deadline: None,
        resources,
    }
}

fn work() -> WorkUnit {
    WorkUnit::PreSequencePreparation {
        kind: "tts.prepare.chatterbox".into(),
    }
}

fn contract(c: &InferenceCoordinator, backend: BackendKind) -> LoadedExecutionContract {
    LoadedModelBundleDraft::build(
        &RuntimeAdapterRegistry::built_in(),
        c.execution_group_id(),
        ModelInstanceId::new(91),
        ModelVariant::Chatterbox500M,
        backend,
    )
    .unwrap()
    .execution_contracts(CapabilityKind::Tts)
    .unwrap()
    .into_iter()
    .find(|contract| contract.execution_profile.mode == crate::engine::ExecutionMode::Sequence)
    .expect("Chatterbox publishes a sequence preparation graph")
}

/// The Chatterbox preparation shape: an input-sized lease, a preparation
/// lease with room for the conditioning, then one preparation row whose
/// artifact keeps `CONDITIONING` bytes resident.
async fn committed_preparation(
    c: &Arc<InferenceCoordinator>,
    backend: BackendKind,
) -> (JobResourceObservation, PreparationAdmissionBridge) {
    let initial = c
        .admit_observed(
            spec(fish_s2_artifact_resources(backend, INPUT).unwrap()),
            JobResourceObservation::host(INPUT),
        )
        .await
        .unwrap();
    let bridge = c.bridge_preparation_admission(initial).unwrap();
    let job = c
        .admit_observed_from_preparation(
            bridge,
            spec(artifact_resources(backend, INPUT, WORKSPACE)),
            JobResourceObservation::host(INPUT),
        )
        .await
        .unwrap();
    let contract = contract(c, backend);
    let mut workspace = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => workspace.host_bytes = ResourceAmount::Known(WORKSPACE),
        BackendKind::Metal => workspace.unified_bytes = ResourceAmount::Known(WORKSPACE),
        BackendKind::Cuda => workspace.device_bytes = ResourceAmount::Known(WORKSPACE),
    }
    let row = c
        .seal_preparation_row(
            job,
            &contract,
            &work(),
            WorkCost::with_workspace(128, 128, workspace),
            128,
            PreparationCancellation::default(),
        )
        .unwrap();
    let retained = JobResourceObservation {
        host_bytes: INPUT,
        accelerator_bytes: CONDITIONING,
    };
    let mut outcomes = c
        .run_loaded_native_preparation_batch(vec![row], contract, work(), move |_| {
            Ok(vec![Ok(PreparationArtifact {
                value: (),
                retained,
            })])
        })
        .await
        .unwrap();
    let PreparationRowOutcome::Committed { artifact, bridge } = outcomes.pop().unwrap() else {
        panic!("expected the preparation to commit")
    };
    (artifact.retained, bridge)
}

#[tokio::test]
async fn unpriced_resident_conditioning_is_refused_at_execution_admission() {
    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        let c = coordinator(backend);
        let (retained, bridge) = committed_preparation(&c, backend).await;
        // The generic request estimate: the input alone.
        let execution = spec(fish_s2_artifact_resources(backend, INPUT).unwrap());
        let failure = c
            .admit_observed_from_preparation(bridge, execution, retained)
            .await
            .expect_err("an execution lease below the resident conditioning");
        assert!(
            failure
                .error
                .to_string()
                .contains("would shrink authorization below materialized usage"),
            "{backend:?}: {}",
            failure.error
        );
    }
}

#[tokio::test]
async fn priced_resident_conditioning_is_admitted_exactly() {
    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        let c = coordinator(backend);
        let (retained, bridge) = committed_preparation(&c, backend).await;
        let mut execution = spec(fish_s2_artifact_resources(backend, INPUT).unwrap());
        add_prepared_tts_artifact_to_admission(backend, &mut execution, 0, CONDITIONING).unwrap();
        execution.lane = CoordinatorLane::Resumable;
        let job = c
            .admit_observed_from_preparation(bridge, execution, retained)
            .await
            .unwrap_or_else(|failure| panic!("{backend:?}: {}", failure.error));
        let expected = artifact_resources(backend, INPUT, CONDITIONING);
        assert_eq!(
            c.resource_authority().snapshot().reserved,
            expected,
            "{backend:?}"
        );
        drop(job);
        assert_eq!(c.resource_authority().snapshot().reservations, 0);
    }
}

#[test]
fn prepared_artifact_pricing_adds_host_and_device_shares() {
    let mut execution = spec(fish_s2_artifact_resources(BackendKind::Cuda, INPUT).unwrap());
    add_prepared_tts_artifact_to_admission(BackendKind::Cuda, &mut execution, 10, 20).unwrap();
    assert_eq!(
        execution.resources.host_bytes,
        ResourceAmount::Known(INPUT + 10)
    );
    assert_eq!(execution.resources.device_bytes, ResourceAmount::Known(20));
    let mut metal = spec(fish_s2_artifact_resources(BackendKind::Metal, INPUT).unwrap());
    add_prepared_tts_artifact_to_admission(BackendKind::Metal, &mut metal, 10, 20).unwrap();
    assert_eq!(
        metal.resources.unified_bytes,
        ResourceAmount::Known(INPUT + 30)
    );
}
