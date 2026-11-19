//! Exercise Audio8's real loaded preparation stage without weights or a CUDA device.
use super::*;
use crate::catalog::ModelVariant;
use crate::models::architectures::audio8::Audio8PreparedArtifact;
use crate::runtime::adapters::{CapabilityKind, LoadedModelBundleDraft, RuntimeAdapterRegistry};
use crate::runtime::service::{
    add_audio8_artifact_to_admission, audio8_artifact_resources, audio8_preparation_resources,
};

const INPUT: u64 = 4096;
const HOST_WORK: u64 = 8192;
const DEVICE_WORK: u64 = 16384;

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
fn coordinator(backend: BackendKind, host: u64, device: u64) -> Arc<InferenceCoordinator> {
    let capacity = match backend {
        BackendKind::Cpu | BackendKind::Metal => audio8_artifact_resources(backend, host).unwrap(),
        BackendKind::Cuda => ResourceVector {
            host_bytes: ResourceAmount::Known(host),
            device_bytes: ResourceAmount::Known(device),
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
fn spec(resources: ResourceVector) -> JobSpec {
    JobSpec {
        request_id: "audio8-reservation-test".into(),
        lane: CoordinatorLane::Atomic,
        priority: Priority::default(),
        workload_class: WorkloadClass::Interactive,
        deadline: None,
        resources,
    }
}
fn work() -> WorkUnit {
    WorkUnit::PreSequencePreparation {
        kind: "tts.prepare.audio8".into(),
    }
}
fn contract(c: &InferenceCoordinator, backend: BackendKind) -> LoadedExecutionContract {
    // Preparation needs the real adapter's stage geometry, but performs no
    // sequence execution or physical state allocation. Obtain its draft graph
    // before the model loader seals the ABI-v2 state publication.
    LoadedModelBundleDraft::build(
        &RuntimeAdapterRegistry::built_in(),
        c.execution_group_id(),
        ModelInstanceId::new(77),
        ModelVariant::Audio8TtsRealisticAir,
        backend,
    )
    .unwrap()
    .execution_contracts(CapabilityKind::Tts)
    .unwrap()
    .into_iter()
    .find(|contract| contract.execution_profile.mode == crate::engine::ExecutionMode::Sequence)
    .expect("Audio8 adapter publishes a retained sequence preparation graph")
}
async fn preparation(c: &Arc<InferenceCoordinator>, backend: BackendKind, host: u64) -> JobLease {
    let initial = c
        .admit_observed(
            spec(audio8_artifact_resources(backend, INPUT).unwrap()),
            JobResourceObservation::host(INPUT),
        )
        .await
        .unwrap();
    let bridge = c.bridge_preparation_admission(initial).unwrap();
    c.admit_observed_from_preparation(
        bridge,
        spec(audio8_preparation_resources(backend, INPUT, host).unwrap()),
        JobResourceObservation::host(INPUT),
    )
    .await
    .unwrap()
}
fn row(
    c: &Arc<InferenceCoordinator>,
    contract: &LoadedExecutionContract,
    job: JobLease,
    backend: BackendKind,
    cancellation: PreparationCancellation,
) -> PreparationBatchRow {
    let mut workspace = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => workspace.host_bytes = ResourceAmount::Known(DEVICE_WORK),
        BackendKind::Metal => workspace.unified_bytes = ResourceAmount::Known(DEVICE_WORK),
        BackendKind::Cuda => workspace.device_bytes = ResourceAmount::Known(DEVICE_WORK),
    }
    c.seal_preparation_row(
        job,
        contract,
        &work(),
        WorkCost::with_workspace(128, 128, workspace),
        128,
        cancellation,
    )
    .unwrap()
}
fn assert_released(c: &InferenceCoordinator) {
    assert_eq!(c.resource_authority().snapshot().reservations, 0);
    assert_eq!(c.snapshot().active_jobs, 0);
    assert_eq!(c.snapshot().active_preparation_bridges, 0);
    assert_eq!(c.snapshot().active_executions, 0);
}

#[tokio::test]
async fn audio8_cuda_old_host_authorization_reproduces_error_at_real_commit() {
    let backend = BackendKind::Cuda;
    let c = coordinator(backend, 1 << 30, 1 << 30);
    let artifact = Audio8PreparedArtifact::test_prompt(11, 128);
    let bytes = artifact.retained_bytes().unwrap();
    assert!(bytes > 0);
    let job = preparation(&c, backend, 0).await;
    let contract = contract(&c, backend);
    let row = row(
        &c,
        &contract,
        job,
        backend,
        PreparationCancellation::default(),
    );
    let result = c
        .run_loaded_native_preparation_batch(vec![row], contract, work(), move |_| {
            Ok(vec![Ok(PreparationArtifact {
                value: artifact,
                retained: JobResourceObservation::host(INPUT + bytes),
            })])
        })
        .await
        .unwrap();
    let PreparationRowOutcome::Failed(error) = &result[0] else {
        panic!("expected strict commit failure")
    };
    assert!(error
        .to_string()
        .contains("materialized resource usage exceeds its authorized reservation"));
    assert!(error.to_string().contains("host_bytes"));
    drop(result);
    assert_released(&c);
}

#[tokio::test]
async fn audio8_preparation_commits_and_transfers_exact_artifact_in_every_domain() {
    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        let c = coordinator(backend, 1 << 30, DEVICE_WORK);
        let artifact = Audio8PreparedArtifact::test_prompt(11, 129);
        let bytes = artifact.retained_bytes().unwrap();
        let weak = Arc::downgrade(&artifact);
        let job = preparation(&c, backend, HOST_WORK + bytes).await;
        // CUDA batch must fit with exactly one workspace allowance: no job-level duplicate.
        assert_eq!(job.spec.resources.device_bytes, ResourceAmount::Known(0));
        let contract = contract(&c, backend);
        let row = row(
            &c,
            &contract,
            job,
            backend,
            PreparationCancellation::default(),
        );
        let mut result = c
            .run_loaded_native_preparation_batch(vec![row], contract, work(), move |_| {
                Ok(vec![Ok(PreparationArtifact {
                    value: artifact,
                    retained: JobResourceObservation::host(INPUT + bytes),
                })])
            })
            .await
            .unwrap();
        let PreparationRowOutcome::Committed { artifact, bridge } = result.pop().unwrap() else {
            panic!("expected artifact commit")
        };
        let mut execution = spec(audio8_artifact_resources(backend, INPUT).unwrap());
        let mut observed = JobResourceObservation::host(INPUT);
        add_audio8_artifact_to_admission(backend, &mut execution, &mut observed, bytes).unwrap();
        assert_eq!(observed, artifact.retained);
        execution.lane = CoordinatorLane::Resumable;
        let job = c
            .admit_observed_from_preparation(bridge, execution, artifact.retained)
            .await
            .unwrap();
        assert_eq!(
            c.resource_authority().snapshot().reserved,
            audio8_artifact_resources(backend, INPUT + bytes).unwrap()
        );
        assert_eq!(Arc::strong_count(&artifact.value), 1);
        drop(artifact);
        assert!(weak.upgrade().is_none());
        drop(job);
        assert_released(&c);
    }
}

#[tokio::test]
async fn audio8_preparation_rejects_host_and_device_pressure_before_operation() {
    let backend = BackendKind::Cuda;
    let c = coordinator(backend, INPUT + HOST_WORK - 1, 1 << 30);
    let initial = c
        .admit_observed(
            spec(audio8_artifact_resources(backend, INPUT).unwrap()),
            JobResourceObservation::host(INPUT),
        )
        .await
        .unwrap();
    let bridge = c.bridge_preparation_admission(initial).unwrap();
    let failure = c
        .admit_observed_from_preparation(
            bridge,
            spec(audio8_preparation_resources(backend, INPUT, HOST_WORK).unwrap()),
            JobResourceObservation::host(INPUT),
        )
        .await
        .unwrap_err();
    assert!(matches!(failure.error, Error::Overloaded(_)));
    assert_eq!(c.resource_authority().snapshot().reservations, 1);
    drop(failure);
    assert_released(&c);

    let c = coordinator(backend, 1 << 30, DEVICE_WORK - 1);
    let job = preparation(&c, backend, HOST_WORK).await;
    let contract = contract(&c, backend);
    let row = row(
        &c,
        &contract,
        job,
        backend,
        PreparationCancellation::default(),
    );
    let result = c
        .run_loaded_native_preparation_batch::<(), _>(vec![row], contract, work(), |_| {
            panic!("device admission must reject before the codec operation")
        })
        .await;
    assert!(matches!(result, Err(Error::Overloaded(_))));
    assert_released(&c);
}

#[tokio::test]
async fn audio8_failed_execution_handoff_keeps_artifact_and_original_authorization() {
    let backend = BackendKind::Cuda;
    let c = coordinator(backend, 1 << 30, 1 << 30);
    let artifact = Audio8PreparedArtifact::test_prompt(11, 128);
    let bytes = artifact.retained_bytes().unwrap();
    let job = preparation(&c, backend, HOST_WORK + bytes).await;
    let contract = contract(&c, backend);
    let row = row(
        &c,
        &contract,
        job,
        backend,
        PreparationCancellation::default(),
    );
    let mut result = c
        .run_loaded_native_preparation_batch(vec![row], contract, work(), move |_| {
            Ok(vec![Ok(PreparationArtifact {
                value: artifact,
                retained: JobResourceObservation::host(INPUT + bytes),
            })])
        })
        .await
        .unwrap();
    let PreparationRowOutcome::Committed { artifact, bridge } = result.pop().unwrap() else {
        panic!("expected commit")
    };
    let reserved = c.resource_authority().snapshot().reserved;
    let failure = c
        .admit_observed_from_preparation(
            bridge,
            spec(audio8_artifact_resources(backend, INPUT).unwrap()),
            artifact.retained,
        )
        .await
        .unwrap_err();
    assert!(matches!(failure.error, Error::InvalidInput(_)));
    assert_eq!(c.resource_authority().snapshot().reserved, reserved);
    assert_eq!(Arc::strong_count(&artifact.value), 1);
    drop(artifact);
    drop(failure);
    assert_released(&c);
}

#[tokio::test]
async fn audio8_preparation_failure_cancellation_and_timeout_release_ownership() {
    for mode in 0..3 {
        let backend = BackendKind::Cuda;
        let c = coordinator(backend, 1 << 30, 1 << 30);
        let artifact = Audio8PreparedArtifact::test_prompt(11, 128);
        let bytes = artifact.retained_bytes().unwrap();
        let weak = Arc::downgrade(&artifact);
        let job = preparation(&c, backend, HOST_WORK + bytes).await;
        let contract = contract(&c, backend);
        let cancellation = PreparationCancellation::default();
        let row = row(&c, &contract, job, backend, cancellation.clone());
        let result = c
            .run_loaded_native_preparation_batch(vec![row], contract, work(), move |_| match mode {
                0 => Err(Error::InferenceError("synthetic codec failure".into())),
                1 => {
                    cancellation.cancel();
                    Ok(vec![Ok(PreparationArtifact {
                        value: artifact,
                        retained: JobResourceObservation::host(INPUT + bytes),
                    })])
                }
                _ => Err(Error::Timeout("audio8-reservation-test".into())),
            })
            .await
            .unwrap();
        if mode == 1 {
            assert!(matches!(result[0], PreparationRowOutcome::Cancelled));
        } else {
            assert!(matches!(result[0], PreparationRowOutcome::Failed(_)));
        }
        drop(result);
        assert!(weak.upgrade().is_none());
        assert_released(&c);
    }
}

#[test]
fn audio8_host_resource_helpers_check_overflow_and_zero_artifact() {
    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        assert_eq!(
            audio8_artifact_resources(backend, 0).unwrap(),
            ResourceVector::zero()
        );
        assert!(matches!(
            audio8_preparation_resources(backend, u64::MAX, 1),
            Err(Error::Overloaded(_))
        ));
    }
}

struct AssertAuthorizedOnDrop(Arc<InferenceCoordinator>);
impl Drop for AssertAuthorizedOnDrop {
    fn drop(&mut self) {
        assert!(
            self.0.resource_authority().snapshot().reservations > 0,
            "physical inputs must be freed before their authorization"
        );
    }
}

#[tokio::test]
async fn audio8_owned_inputs_keep_authorization_through_sealing_and_operation_failures() {
    use crate::runtime::service::PreparationOwnedInputs;
    let backend = BackendKind::Cuda;
    for execute in [false, true] {
        let c = coordinator(backend, 1 << 30, 1 << 30);
        let job = preparation(&c, backend, HOST_WORK).await;
        let owner = PreparationOwnedInputs::new(AssertAuthorizedOnDrop(c.clone()), job.clone());
        // Simulate consuming row seal/dispatch rejecting and dropping its job.
        drop(job);
        if execute {
            let result: Result<()> = owner.run(|inputs| {
                drop(inputs);
                Err(Error::InferenceError("preparation failed".into()))
            });
            assert!(result.is_err());
        } else {
            drop(owner);
        }
        assert_released(&c);
    }
}

#[tokio::test]
async fn audio8_failed_handoff_drops_inputs_before_bridge() {
    use crate::runtime::service::release_failed_preparation;
    let backend = BackendKind::Cuda;
    let c = coordinator(backend, 1 << 30, 1 << 30);
    let job = preparation(&c, backend, HOST_WORK).await;
    let bridge = c.bridge_preparation_admission(job).unwrap();
    let mut execution = spec(audio8_artifact_resources(backend, INPUT).unwrap());
    execution.deadline = Some(Instant::now()); // Rejected identity/deadline still returns its bridge.
    let failure = c
        .admit_observed_from_preparation(bridge, execution, JobResourceObservation::host(INPUT))
        .await
        .unwrap_err();
    let _error = release_failed_preparation(failure, AssertAuthorizedOnDrop(c.clone()));
    assert_released(&c);
}
