use super::*;
use crate::backends::{BackendCapabilities, BackendContext, BackendSelectionSource};
use crate::engine::{
    ExecutionDisposition, ExecutorOutput, FinishReason as ExecutionFinishReason,
    OutputProcessor,
};
use crate::runtime::broker::{InferenceBroker, InferenceBrokerMode};

#[test]
fn granite_speech_deadline_buckets_separate_unbounded_and_materially_different_budgets() {
    let now = Instant::now();
    assert_eq!(granite_speech_deadline_budget_bucket(None, now), None);
    assert_eq!(
        granite_speech_deadline_budget_bucket(Some(now + Duration::from_millis(199)), now),
        Some(1)
    );
    assert_eq!(
        granite_speech_deadline_budget_bucket(Some(now + Duration::from_millis(200)), now),
        Some(2)
    );
}

#[test]
fn direct_execution_capacity_obeys_rollout_mode() {
    assert_eq!(
        effective_physical_execution_parallelism(
            crate::config::PhysicalExecutionMode::Serial,
            4,
            8,
        ),
        1
    );
    assert_eq!(
        effective_physical_execution_parallelism(
            crate::config::PhysicalExecutionMode::Shadow,
            4,
            8,
        ),
        1
    );
    assert_eq!(
        effective_physical_execution_parallelism(
            crate::config::PhysicalExecutionMode::Concurrent,
            4,
            8,
        ),
        8
    );
}

#[tokio::test]
async fn runtime_chat_handle_cancellation_is_idempotent_and_teardown_is_explicit() {
    let cancellation = Arc::new(AtomicBool::new(false));
    let cancellation_wakeup = Arc::new(Notify::new());
    let wakeup_observer = cancellation_wakeup.clone();
    let (_event_sender, events) = mpsc::channel(1);
    let (teardown_sender, teardown) = oneshot::channel();
    let handle = RuntimeChatInvocation {
        request_id: "serving-request".into(),
        cancellation: cancellation.clone(),
        cancellation_wakeup,
        events,
        teardown: Some(teardown),
        cancel_on_drop: true,
    };

    assert!(handle.request_cancel());
    assert!(!handle.request_cancel());
    tokio::time::timeout(Duration::from_millis(50), wakeup_observer.notified())
        .await
        .expect("cancellation wakeup must persist until the driver observes it");
    teardown_sender
        .send(RuntimeChatTeardown {
            request_id: "serving-request".into(),
            disposition: RuntimeChatTeardownDisposition::Cancelled,
        })
        .unwrap();

    let confirmed = handle.wait_for_teardown().await.unwrap();
    assert_eq!(confirmed.request_id, "serving-request");
    assert_eq!(
        confirmed.disposition,
        RuntimeChatTeardownDisposition::Cancelled
    );
    assert!(cancellation.load(Ordering::Acquire));
}

#[test]
fn dropping_runtime_chat_handle_requests_cancellation() {
    let cancellation = Arc::new(AtomicBool::new(false));
    let (_event_sender, events) = mpsc::channel(1);
    let (_teardown_sender, teardown) = oneshot::channel();
    let handle = RuntimeChatInvocation {
        request_id: "disconnected-request".into(),
        cancellation: cancellation.clone(),
        cancellation_wakeup: Arc::new(Notify::new()),
        events,
        teardown: Some(teardown),
        cancel_on_drop: true,
    };

    drop(handle);

    assert!(cancellation.load(Ordering::Acquire));
}

fn terminal_output(reason: ExecutionFinishReason) -> EngineOutput {
    OutputProcessor::new(24_000).process_execution(
        ExecutorOutput::terminal("terminal-request".to_string()),
        &ExecutionDisposition::Finished(reason),
        7,
        std::time::Duration::ZERO,
    )
}

#[test]
fn stream_output_order_rejects_wrong_identity_reordering_and_truncation() {
    let request_id = "ordered-stream";
    let mut order = StreamOutputOrder::new(EngineStreamPolicy::FailOnFull);
    let first = StreamingOutput::new(request_id.to_string(), 0, vec![0.0], 24_000);
    order.observe(request_id, &first).unwrap();

    let duplicate = StreamingOutput::new(request_id.to_string(), 0, vec![0.0], 24_000);
    assert!(order
        .observe(request_id, &duplicate)
        .unwrap_err()
        .to_string()
        .contains("not greater"));
    let wrong_request = StreamingOutput::new("stale".to_string(), 1, vec![0.0], 24_000);
    assert!(order
        .observe(request_id, &wrong_request)
        .unwrap_err()
        .to_string()
        .contains("carried request ID"));
    assert!(order
        .require_final(request_id)
        .unwrap_err()
        .to_string()
        .contains("without a final marker"));

    let gap = StreamingOutput::new(request_id.to_string(), 4, Vec::new(), 0);
    assert!(order
        .observe(request_id, &gap)
        .unwrap_err()
        .to_string()
        .contains("did not match expected 1"));

    // Gaps remain valid only for an explicitly lossy DropNewest transport,
    // while every observed sequence must still advance monotonically.
    let mut order = StreamOutputOrder::new(EngineStreamPolicy::DropNewest);
    order.observe(request_id, &first).unwrap();
    let mut final_output = StreamingOutput::new(request_id.to_string(), 4, Vec::new(), 0);
    final_output.is_final = true;
    order.observe(request_id, &final_output).unwrap();
    order.require_final(request_id).unwrap();

    let after_final = StreamingOutput::new(request_id.to_string(), 5, vec![0.0], 24_000);
    assert!(order
        .observe(request_id, &after_final)
        .unwrap_err()
        .to_string()
        .contains("after its final marker"));
}

#[test]
fn residency_binding_carries_exact_instance_and_rejects_wrong_variant() {
    let residency = crate::model::ModelResidency::default();
    let instance = crate::engine::ModelInstanceId::new(17);
    let lease = residency.acquire_instance_lease(ModelVariant::Kokoro82M, instance);
    let adapters = RuntimeAdapterRegistry::built_in();
    let bundle = LoadedModelBundle::bind(
        &adapters,
        crate::engine::ExecutionGroupId::new(3),
        instance,
        ModelVariant::Kokoro82M,
        BackendKind::Cpu,
    )
    .expect("loaded bundle");
    let mut request =
        EngineCoreRequest::tts("bind me").with_model_variant(ModelVariant::Kokoro82M);

    bind_request_to_residency(&mut request, Some(&lease), Some(&bundle), false)
        .expect("matching residency");
    assert_eq!(request.model_instance_id(), Some(instance));
    assert!(request.execution_adapter_binding().is_some());
    assert!(request.v2_state_runtime().is_some());

    let mut wrong = EngineCoreRequest::tts("wrong").with_model_variant(ModelVariant::Qwen306B);
    assert!(bind_request_to_residency(&mut wrong, Some(&lease), Some(&bundle), false).is_err());
    assert_eq!(wrong.model_instance_id(), None);

    let mut missing_bundle =
        EngineCoreRequest::tts("missing").with_model_variant(ModelVariant::Kokoro82M);
    assert!(bind_request_to_residency(&mut missing_bundle, Some(&lease), None, false).is_err());
    assert_eq!(missing_bundle.model_instance_id(), None);
}

#[test]
fn residency_binding_uses_one_descriptor_for_execution_and_state_truth() {
    let residency = crate::model::ModelResidency::default();
    let instance = crate::engine::ModelInstanceId::new(19);
    let variant = ModelVariant::Kokoro82M;
    let lease = residency.acquire_instance_lease(variant, instance);
    let bundle = LoadedModelBundle::bind(
        &RuntimeAdapterRegistry::built_in(),
        crate::engine::ExecutionGroupId::new(3),
        instance,
        variant,
        BackendKind::Cpu,
    )
    .expect("loaded bundle");
    let mut request = EngineCoreRequest::tts("state").with_model_variant(variant);

    bind_request_to_residency(&mut request, Some(&lease), Some(&bundle), false)
        .expect("matching loaded capability descriptor");

    let execution = request
        .execution_adapter_binding()
        .expect("execution binding");
    let state = request.v2_state_runtime().expect("state binding");
    assert_eq!(execution.model_instance_id, instance);
    state.validate_against(BackendKind::Cpu, execution).unwrap();
}

#[test]
fn residency_binding_preserves_v2_state_without_opaque_fallback() {
    let residency = crate::model::ModelResidency::default();
    let instance = crate::engine::ModelInstanceId::new(20);
    let variant = ModelVariant::Kokoro82M;
    let lease = residency.acquire_instance_lease(variant, instance);
    let registry = RuntimeAdapterRegistry::built_in();
    let compatibility = LoadedModelBundle::bind(
        &registry,
        crate::engine::ExecutionGroupId::new(3),
        instance,
        variant,
        BackendKind::Cpu,
    )
    .expect("compatibility bundle");
    let offline = compatibility
        .contract_for_streaming(CapabilityKind::Tts, StreamingRequirements::NONE)
        .expect("offline TTS contract")
        .stages;
    let transport = compatibility
        .contract_for_streaming(CapabilityKind::Tts, StreamingRequirements::transport_only())
        .expect("transport TTS contract")
        .stages;
    let native = compatibility
        .contract_for_streaming(
            CapabilityKind::Tts,
            StreamingRequirements {
                transport_output: false,
                model_native: true,
                asr_long_form: false,
            },
        )
        .expect("native TTS contract")
        .stages;
    let native_transport = compatibility
        .contract_for_streaming(CapabilityKind::Tts, StreamingRequirements::native(true))
        .expect("native transport TTS contract")
        .stages;
    let descriptor =
        crate::kv::v2::CapabilityStateDescriptorV2::stateless_for_stage_graphs_test(&[
            &offline,
            &transport,
            &native,
            &native_transport,
        ]);
    let bundle = LoadedModelBundle::bind_with_state_publications(
        &registry,
        crate::engine::ExecutionGroupId::new(3),
        instance,
        variant,
        BackendKind::Cpu,
        HashMap::from([(
            CapabilityKind::Tts,
            crate::runtime::adapters::LoadedStatePublication::V2(descriptor.clone()),
        )]),
    )
    .expect("v2 loaded bundle");
    let mut request = EngineCoreRequest::tts("stateless").with_model_variant(variant);
    bind_request_to_residency(&mut request, Some(&lease), Some(&bundle), false)
        .expect("matching v2 loaded capability descriptor");

    assert_eq!(request.v2_state_descriptor(), Some(&descriptor));
    assert_eq!(
        request.v2_state_fingerprint(),
        Some(descriptor.fingerprint(&offline).unwrap())
    );
    assert!(request.v2_state_runtime().is_some());
    let direct = loaded_contract_for_residency(
        &lease,
        Some(&bundle),
        CapabilityKind::Tts,
        false,
        false,
        crate::engine::ExecutionGroupId::new(3),
        BackendKind::Cpu,
        None,
    )
    .expect("direct runners may use a load-sealed stateless v2 runtime");
    assert_eq!(direct.model_instance_id, instance);
}

#[test]
fn direct_loaded_contract_requires_exact_generation_group_backend_and_target() {
    let residency = crate::model::ModelResidency::default();
    let instance = crate::engine::ModelInstanceId::new(21);
    let group = crate::engine::ExecutionGroupId::new(8);
    let lease = residency.acquire_instance_lease(ModelVariant::Kokoro82M, instance);
    let adapters = RuntimeAdapterRegistry::built_in();
    let bundle = LoadedModelBundle::bind(
        &adapters,
        group,
        instance,
        ModelVariant::Kokoro82M,
        BackendKind::Cpu,
    )
    .unwrap();

    let contract = loaded_contract_for_residency(
        &lease,
        Some(&bundle),
        CapabilityKind::StreamingTts,
        false,
        false,
        group,
        BackendKind::Cpu,
        Some(ExecutionTargetKind::DirectModel),
    )
    .unwrap();
    assert_eq!(contract.model_instance_id, instance);
    assert_eq!(contract.execution_group_id, group);

    assert!(loaded_contract_for_residency(
        &lease,
        Some(&bundle),
        CapabilityKind::StreamingTts,
        false,
        false,
        group,
        BackendKind::Cpu,
        Some(ExecutionTargetKind::TokenEngine),
    )
    .is_err());
    assert!(loaded_contract_for_residency(
        &lease,
        Some(&bundle),
        CapabilityKind::StreamingTts,
        false,
        false,
        crate::engine::ExecutionGroupId::new(group.get() + 1),
        BackendKind::Cpu,
        Some(ExecutionTargetKind::DirectModel),
    )
    .is_err());
}

async fn pending_streaming_guard_fixture(
    runtime: &RuntimeService,
    request_id: &str,
    deadline: Option<Instant>,
) -> (
    EngineCoreRequest,
    PendingRequestGuard,
    oneshot::Receiver<Result<EngineOutput>>,
    ModelVariant,
) {
    let residency_variant = ModelVariant::Kokoro82M;
    let mut request = EngineCoreRequest::tts("streaming callback fixture")
        .with_model_variant(residency_variant)
        .with_deadline(deadline);
    request.id = request_id.to_string();
    request.prompt_tokens = vec![1];
    request.streaming = true;
    let observation_request = request.clone();
    let (spec, observation) = runtime
        .coordinator_job_for_request(&request)
        .expect("job shape");
    let job = runtime
        .coordinator
        .admit_observed(spec, observation)
        .await
        .expect("job admission");
    let (registration_id, receiver) = runtime
        .register_waiter(request_id)
        .await
        .expect("waiter registration");
    let session = runtime
        .core_engine
        .add_request_with_session(request)
        .await
        .expect("engine admission");
    runtime
        .bind_waiter(request_id, registration_id, session.epoch)
        .await
        .expect("waiter binding");
    let residency_lease = runtime
        .model_manager
        .acquire_residency_lease(residency_variant);
    let guard = PendingRequestGuard::new(
        session,
        runtime.core_engine.clone(),
        runtime.completion_waiters.clone(),
        registration_id,
        runtime.telemetry.clone(),
        job,
        Some(residency_lease),
    );

    (observation_request, guard, receiver, residency_variant)
}

struct DriverHarness {
    stream: mpsc::Sender<StreamingOutput>,
    completion: oneshot::Sender<Result<EngineOutput>>,
    events: mpsc::Receiver<Result<RuntimeChatInvocationEvent>>,
    teardown: oneshot::Receiver<RuntimeChatTeardown>,
}

/// A streaming chat driver over an admitted fixture session, fed by hand
/// through its engine stream and completion channels.
async fn chat_driver_harness(
    runtime: &RuntimeService,
    request_id: &str,
    event_capacity: usize,
    slow_consumer_timeout: Duration,
) -> DriverHarness {
    let (_request, guard, _receiver, _variant) =
        pending_streaming_guard_fixture(runtime, request_id, None).await;
    let (stream, stream_rx) = mpsc::channel(64);
    let (completion, completion_rx) = oneshot::channel();
    let (event_sender, events) = mpsc::channel(event_capacity);
    let (teardown_sender, teardown) = oneshot::channel();
    let driver = RuntimeChatInvocationDriver {
        request_id: request_id.to_string(),
        deadline: None,
        observation_context: RuntimeObservationContext::default(),
        admission_ms: None,
        telemetry: runtime.telemetry.clone(),
        cancellation: Arc::new(AtomicBool::new(false)),
        cancellation_wakeup: Arc::new(Notify::new()),
        events: event_sender,
        teardown: Some(teardown_sender),
        completion: completion_rx,
        stream: Some(stream_rx),
        stream_order: Some(StreamOutputOrder::new(EngineStreamPolicy::FailOnFull)),
        streamed_text: String::new(),
        pending_delta: None,
        slow_consumer_timeout,
        guard,
    };
    tokio::spawn(driver.drive());
    DriverHarness {
        stream,
        completion,
        events,
        teardown,
    }
}

fn text_chunk(request_id: &str, sequence: usize, text: &str) -> StreamingOutput {
    let mut chunk = StreamingOutput::new(request_id.to_string(), sequence, Vec::new(), 0);
    chunk.text = Some(text.to_string());
    chunk
}

#[tokio::test]
async fn chat_driver_coalesces_deltas_for_a_paused_consumer() {
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let request_id = "paused-consumer";
    let mut harness =
        chat_driver_harness(&runtime, request_id, 2, Duration::from_secs(30)).await;
    let words = (0..40)
        .map(|index| format!("w{index} "))
        .collect::<Vec<_>>();
    for (sequence, word) in words.iter().enumerate() {
        harness
            .stream
            .send(text_chunk(request_id, sequence, word))
            .await
            .expect("driver drains the engine stream");
    }
    let mut final_chunk =
        StreamingOutput::new(request_id.to_string(), words.len(), Vec::new(), 0);
    final_chunk.is_final = true;
    harness.stream.send(final_chunk).await.unwrap();
    drop(harness.stream);
    let _ = harness
        .completion
        .send(Ok(terminal_output(ExecutionFinishReason::Completed)));

    // The consumer stays paused far past the 2-slot channel's capacity.
    tokio::time::sleep(Duration::from_millis(50)).await;
    let mut text = String::new();
    let mut deltas = 0;
    let generation = loop {
        match harness.events.recv().await.expect("terminal event") {
            Ok(RuntimeChatInvocationEvent::TextDelta { text: delta, .. }) => {
                deltas += 1;
                text.push_str(&delta);
            }
            Ok(RuntimeChatInvocationEvent::Completed(generation)) => break generation,
            Err(error) => panic!("paused consumer failed the invocation: {error}"),
        }
    };
    assert_eq!(text, words.concat());
    assert_eq!(generation.text, words.concat());
    assert!(deltas < words.len(), "deltas were coalesced, not queued");
    assert!(harness.events.recv().await.is_none());
    assert_eq!(
        harness.teardown.await.expect("teardown").disposition,
        RuntimeChatTeardownDisposition::Completed
    );
}

/// A model that fails mid-stream ends its stream without a final marker;
/// the invocation reports the model's error, not the missing marker.
#[tokio::test]
async fn chat_driver_reports_the_model_error_of_a_failed_stream() {
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let request_id = "failed-stream";
    let mut harness =
        chat_driver_harness(&runtime, request_id, 4, Duration::from_secs(30)).await;
    harness
        .stream
        .send(text_chunk(request_id, 0, "partial"))
        .await
        .unwrap();
    let _ = harness.completion.send(Err(Error::InferenceError(
        "model rejected the answer".into(),
    )));
    drop(harness.stream);

    assert!(matches!(
        harness.events.recv().await,
        Some(Ok(RuntimeChatInvocationEvent::TextDelta { .. }))
    ));
    match harness.events.recv().await {
        Some(Err(Error::InferenceError(message))) => {
            assert_eq!(message, "model rejected the answer")
        }
        other => panic!("expected the model error, got {other:?}"),
    }
    assert_eq!(
        harness.teardown.await.expect("teardown").disposition,
        RuntimeChatTeardownDisposition::Failed
    );
}

#[tokio::test]
async fn chat_driver_fails_a_consumer_stalled_past_the_timeout() {
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let request_id = "stalled-consumer";
    let mut harness =
        chat_driver_harness(&runtime, request_id, 1, Duration::from_millis(50)).await;
    for sequence in 0..4 {
        harness
            .stream
            .send(text_chunk(request_id, sequence, "x"))
            .await
            .unwrap();
    }

    let teardown = tokio::time::timeout(Duration::from_secs(5), &mut harness.teardown)
        .await
        .expect("a stalled consumer is failed after its timeout")
        .expect("teardown");
    assert_eq!(teardown.disposition, RuntimeChatTeardownDisposition::Failed);
    assert!(matches!(
        harness.events.recv().await,
        Some(Ok(RuntimeChatInvocationEvent::TextDelta { .. }))
    ));
}

#[test]
fn runtime_completion_preserves_typed_terminal_failures() {
    assert!(matches!(
        runtime_completion(terminal_output(ExecutionFinishReason::Cancelled)),
        Err(Error::Cancelled(request_id)) if request_id == "terminal-request"
    ));
    assert!(matches!(
        runtime_completion(terminal_output(ExecutionFinishReason::TimedOut)),
        Err(Error::Timeout(request_id)) if request_id == "terminal-request"
    ));
}

#[tokio::test]
async fn duplicate_waiter_registration_preserves_original_owner() {
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let (original_registration, original) = runtime
        .register_waiter("same-request")
        .await
        .expect("first waiter");
    let duplicate = runtime.register_waiter("same-request").await;

    assert!(matches!(duplicate, Err(Error::InvalidInput(_))));
    assert_eq!(runtime.completion_waiters.lock().await.len(), 1);
    assert!(runtime
        .completion_waiters
        .lock()
        .await
        .contains_key("same-request"));
    drop(original);
    runtime
        .remove_waiter("same-request", original_registration)
        .await;
}

#[tokio::test]
async fn delayed_waiter_cleanup_cannot_remove_a_reused_request_registration() {
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let (old_registration, old_receiver) = runtime
        .register_waiter("reused-request")
        .await
        .expect("old waiter");
    runtime
        .remove_waiter("reused-request", old_registration)
        .await;
    drop(old_receiver);

    let (new_registration, _new_receiver) = runtime
        .register_waiter("reused-request")
        .await
        .expect("new waiter");
    assert_ne!(old_registration, new_registration);

    assert!(
        !remove_waiter_registration(
            runtime.completion_waiters.as_ref(),
            "reused-request",
            old_registration,
        )
        .await
    );
    assert_eq!(
        runtime
            .completion_waiters
            .lock()
            .await
            .get("reused-request")
            .map(|waiter| waiter.registration_id),
        Some(new_registration)
    );
}

#[tokio::test]
async fn dropped_waiter_guard_cleans_an_unbound_registration() {
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let (registration_id, receiver) = runtime
        .register_waiter("cancelled-before-admission")
        .await
        .expect("waiter registration");
    let guard = WaiterRegistrationGuard::new(
        "cancelled-before-admission".to_string(),
        registration_id,
        runtime.completion_waiters.clone(),
    );

    drop(guard);
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if !runtime
                .completion_waiters
                .lock()
                .await
                .contains_key("cancelled-before-admission")
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("dropped guard did not remove its exact registration");
    assert!(receiver.await.is_err());
}

/// The fault-injection hook fails its own runtime only: the resource
/// authority is shared by every runtime in the process, and a probe test that
/// poisoned it failed unrelated tests running alongside.
#[test]
fn injected_device_fault_fails_only_its_own_runtime() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let faulted = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let neighbour = RuntimeService::new(EngineConfig::default()).expect("runtime");
    assert!(faulted.latch_device_fault_for_test(&Error::InferenceError(
        "DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, \"an illegal memory access was encountered\")"
            .into(),
    )));
    assert!(faulted.health().is_failed());
    assert!(neighbour.health().is_healthy(), "{:?}", neighbour.health());
}

#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropped_pending_guard_reclaims_admitted_session_and_job() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    runtime.ensure_step_driver_started().await;
    let mut request = EngineCoreRequest::tts("cancel between admission and waiter binding")
        .with_model_variant(ModelVariant::Kokoro82M);
    request.id = "cancel-during-waiter-binding".to_string();
    request.prompt_tokens = vec![1];
    let request_id = request.id.clone();
    let (spec, observation) = runtime
        .coordinator_job_for_request(&request)
        .expect("job shape");
    let job = runtime
        .coordinator
        .admit_observed(spec, observation)
        .await
        .expect("job admission");
    let (registration_id, receiver) = runtime
        .register_waiter(&request_id)
        .await
        .expect("waiter registration");
    let abandoned_session = runtime
        .core_engine
        .add_request_with_session(request)
        .await
        .expect("engine admission");
    let residency_variant = ModelVariant::Kokoro82M;
    let residency_lease = runtime
        .model_manager
        .acquire_residency_lease(residency_variant);
    assert_eq!(
        runtime
            .model_manager
            .active_residency_leases(residency_variant),
        1
    );
    let waiter_lock = runtime.completion_waiters.lock().await;
    let guard = PendingRequestGuard::new(
        abandoned_session.clone(),
        runtime.core_engine.clone(),
        runtime.completion_waiters.clone(),
        registration_id,
        runtime.telemetry.clone(),
        job,
        Some(residency_lease),
    );

    drop(guard);
    tokio::task::yield_now().await;
    assert_eq!(
        runtime
            .model_manager
            .active_residency_leases(residency_variant),
        1,
        "cancellation cleanup must retain the model residency lease"
    );
    drop(waiter_lock);
    assert!(receiver.await.is_err());

    let mut replacement = EngineCoreRequest::tts("reuse cancelled binding request id")
        .with_model_variant(ModelVariant::Kokoro82M);
    replacement.id = request_id.clone();
    replacement.prompt_tokens = vec![2];
    let replacement_session = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if let Ok(session) = runtime
                .core_engine
                .add_request_with_session(replacement.clone())
                .await
            {
                break session;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("dropped pending guard kept the exact session fenced");
    assert_ne!(replacement_session.epoch, abandoned_session.epoch);
    // The cleanup task lifts the session fence first and releases the job
    // and the residency lease just after, so a replacement can be admitted
    // in between: wait for the release rather than race it.
    tokio::time::timeout(Duration::from_secs(5), async {
        while runtime.coordinator_snapshot().active_jobs != 0
            || runtime
                .model_manager
                .active_residency_leases(residency_variant)
                != 0
        {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("dropped pending guard released its job and residency lease");
    runtime.core_engine.abort_all_requests().await;
}

#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropped_pending_guard_retains_ownership_while_exact_abort_waits_for_core_step_lock() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let mut request = EngineCoreRequest::tts("cancel during an in-flight engine step");
    request.id = "cancel-during-engine-step".to_string();
    request.prompt_tokens = vec![1];
    request.model_variant = Some(ModelVariant::Kokoro82M);
    let request_id = request.id.clone();
    let (spec, observation) = runtime
        .coordinator_job_for_request(&request)
        .expect("job shape");
    let job = runtime
        .coordinator
        .admit_observed(spec, observation)
        .await
        .expect("job admission");
    let (registration_id, receiver) = runtime
        .register_waiter(&request_id)
        .await
        .expect("waiter registration");
    let session = runtime
        .core_engine
        .add_request_with_session(request)
        .await
        .expect("engine admission");
    let residency_variant = ModelVariant::Kokoro82M;
    let residency_lease = runtime
        .model_manager
        .acquire_residency_lease(residency_variant);
    let guard = PendingRequestGuard::new(
        session,
        runtime.core_engine.clone(),
        runtime.completion_waiters.clone(),
        registration_id,
        runtime.telemetry.clone(),
        job,
        Some(residency_lease),
    );

    let (step_entered_tx, step_entered_rx) = oneshot::channel();
    let (release_step_tx, release_step_rx) = oneshot::channel();
    let engine = runtime.core_engine.clone();
    let step_lock = tokio::spawn(async move {
        engine
            .hold_core_step_lock_for_test(step_entered_tx, release_step_rx)
            .await;
    });
    step_entered_rx.await.expect("step lock was not acquired");

    drop(guard);
    assert!(tokio::time::timeout(Duration::from_secs(1), receiver)
        .await
        .expect("cleanup did not remove its waiter before exact abort")
        .is_err());
    assert_eq!(runtime.coordinator_snapshot().active_jobs, 1);
    assert_eq!(
        runtime
            .model_manager
            .active_residency_leases(residency_variant),
        1,
        "exact abort must retain residency while an engine step owns the core lock"
    );

    release_step_tx.send(()).expect("release step lock");
    step_lock.await.expect("step-lock task");
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if runtime.coordinator_snapshot().active_jobs == 0
                && runtime
                    .model_manager
                    .active_residency_leases(residency_variant)
                    == 0
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("exact abort did not release request ownership after the core lock became safe");
    runtime.core_engine.abort_all_requests().await;
}

#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn nonstreaming_engine_admission_returns_at_deadline_while_core_lock_is_held() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let (step_entered_tx, step_entered_rx) = oneshot::channel();
    let (release_step_tx, release_step_rx) = oneshot::channel();
    let engine = runtime.core_engine.clone();
    let step_lock = tokio::spawn(async move {
        engine
            .hold_core_step_lock_for_test(step_entered_tx, release_step_rx)
            .await;
    });
    step_entered_rx.await.expect("step lock was not acquired");

    let request_id = "nonstreaming-admission-deadline".to_string();
    let deadline = Instant::now() + Duration::from_millis(25);
    let mut request = EngineCoreRequest::tts("bounded Engine admission")
        .with_model_variant(ModelVariant::Kokoro82M)
        .with_deadline(Some(deadline));
    request.id = request_id.clone();
    request.prompt_tokens = vec![1];
    let (spec, observation) = runtime
        .coordinator_job_for_request(&request)
        .expect("job shape");
    let job = runtime
        .coordinator
        .admit_observed(spec, observation)
        .await
        .expect("job admission");
    let err = tokio::time::timeout(
        Duration::from_secs(1),
        runtime.await_engine_admission_for_job(
            &job,
            runtime.core_engine.add_request_with_session(request),
        ),
    )
    .await
    .expect("Engine admission waited for the core lock past its deadline")
    .expect_err("expired Engine admission unexpectedly succeeded");
    assert!(
        matches!(err, Error::Timeout(ref id) if id == &request_id),
        "expected request deadline timeout, got {err:?}"
    );
    drop(job);
    assert!(
        !step_lock.is_finished(),
        "the core lock was released too early"
    );
    assert_eq!(runtime.coordinator_snapshot().active_jobs, 0);
    assert!(!runtime
        .completion_waiters
        .lock()
        .await
        .contains_key(&request_id));

    release_step_tx.send(()).expect("release step lock");
    step_lock.await.expect("step-lock task");
    assert_eq!(
        runtime.core_engine.request_session_key(&request_id).await,
        None
    );
}

#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_engine_admission_returns_at_deadline_while_core_lock_is_held() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let (step_entered_tx, step_entered_rx) = oneshot::channel();
    let (release_step_tx, release_step_rx) = oneshot::channel();
    let engine = runtime.core_engine.clone();
    let step_lock = tokio::spawn(async move {
        engine
            .hold_core_step_lock_for_test(step_entered_tx, release_step_rx)
            .await;
    });
    step_entered_rx.await.expect("step lock was not acquired");

    let request_id = "streaming-admission-deadline".to_string();
    let deadline = Instant::now() + Duration::from_millis(25);
    let residency_variant = ModelVariant::FishAudioS2Pro;
    let mut request = EngineCoreRequest::tts("bounded streaming Engine admission")
        .with_model_variant(residency_variant)
        .with_deadline(Some(deadline));
    request.id = request_id.clone();
    request.prompt_tokens = vec![1];
    request.streaming = true;
    request.mark_audio_streaming_only();
    // The model-bound fixture has already completed preparation. This test
    // isolates deadline handling while admission waits for the core lock.
    request
        .install_fish_s2_tts_execution_model(
            residency_variant,
            crate::models::registry::FishS2TtsModelLease::for_test(
                crate::models::architectures::fish_s2::FishS2TtsModel::for_test(),
            ),
            crate::models::architectures::fish_s2::FishS2PreparedArtifact::test_prompt(11, 16),
            FishS2GenerationParams {
                max_frames: 32,
                ..Default::default()
            },
            128,
        )
        .unwrap();
    let (spec, observation) = runtime
        .coordinator_job_for_request(&request)
        .expect("job shape");
    let job = runtime
        .coordinator
        .admit_observed(spec, observation)
        .await
        .expect("job admission");

    let residency_lease = runtime
        .model_manager
        .acquire_residency_lease(residency_variant);
    let err = tokio::time::timeout(
        Duration::from_secs(1),
        runtime.run_streaming_request_after_admission(
            request,
            |_| std::future::ready(Ok(())),
            job,
            Some(residency_lease),
            true,
        ),
    )
    .await
    .expect("streaming Engine admission waited for the core lock past its deadline")
    .expect_err("expired streaming Engine admission unexpectedly succeeded");
    assert!(
        matches!(err, Error::Timeout(ref id) if id == &request_id),
        "expected admission timeout, got {err:?}"
    );
    assert!(
        !step_lock.is_finished(),
        "the core lock was released too early"
    );
    assert_eq!(runtime.coordinator_snapshot().active_jobs, 0);
    assert_eq!(
        runtime
            .model_manager
            .active_residency_leases(residency_variant),
        0
    );
    assert!(!runtime
        .completion_waiters
        .lock()
        .await
        .contains_key(&request_id));

    release_step_tx.send(()).expect("release step lock");
    step_lock.await.expect("step-lock task");
    assert_eq!(
        runtime.core_engine.request_session_key(&request_id).await,
        None
    );
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn fish_streaming_runs_model_preparation_before_engine_admission() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    for model_streaming_required in [true, false] {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let mut request = EngineCoreRequest::tts("prepare streaming Fish")
            .with_model_variant(ModelVariant::FishAudioS2Pro);
        request.streaming = true;
        request.mark_audio_streaming_only();
        let request_id = request.id.clone();
        let (spec, observation) = runtime.coordinator_job_for_request(&request).unwrap();
        let job = runtime
            .coordinator
            .admit_observed(spec, observation)
            .await
            .unwrap();
        let callback_invoked = Arc::new(AtomicBool::new(false));
        let callback_observer = callback_invoked.clone();

        // No weights are installed: reaching Fish preparation must fail here,
        // before a request without its model lease can enter the executor.
        let error = tokio::time::timeout(
            Duration::from_secs(1),
            runtime.run_streaming_request_after_admission(
                request,
                move |_| {
                    callback_observer.store(true, Ordering::Release);
                    std::future::ready(Ok(()))
                },
                job,
                None,
                model_streaming_required,
            ),
        )
        .await
        .expect("missing Fish model should fail before execution")
        .expect_err("streaming Fish must prepare its model");
        assert!(
            matches!(error, Error::ModelNotFound(ref message) if message.contains("Fish S2 TTS")),
            "expected the Fish preparation failure, got {error:?}"
        );
        assert!(!callback_invoked.load(Ordering::Acquire));
        assert_eq!(runtime.coordinator_snapshot().active_jobs, 0);
        assert_eq!(
            runtime.core_engine.request_session_key(&request_id).await,
            None
        );
        assert!(!runtime
            .completion_waiters
            .lock()
            .await
            .contains_key(&request_id));
    }
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn shared_preparation_preserves_fish_model_artifact_and_streaming_budget() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    use crate::models::architectures::fish_s2::{FishS2PreparedArtifact, FishS2TtsModel};
    use crate::models::registry::FishS2TtsModelLease;

    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let model = FishS2TtsModelLease::for_test(FishS2TtsModel::for_test());
    let model_identity = model.model_arc();
    let artifact = FishS2PreparedArtifact::test_prompt(11, 16);
    let mut request = EngineCoreRequest::tts("already prepared streaming Fish")
        .with_model_variant(ModelVariant::FishAudioS2Pro);
    request.streaming = true;
    request.mark_audio_streaming_only();
    request
        .install_fish_s2_tts_execution_model(
            ModelVariant::FishAudioS2Pro,
            model,
            artifact.clone(),
            FishS2GenerationParams {
                max_frames: 32,
                ..Default::default()
            },
            128,
        )
        .unwrap();
    let request_id = request.id.clone();
    let queue_capacity = request.audio_stream_engine_queue_capacity;
    let (spec, observation) = runtime.coordinator_job_for_request(&request).unwrap();
    let job = runtime
        .coordinator
        .admit_observed(spec, observation)
        .await
        .unwrap();

    let (prepared, job) = runtime
        .prepare_request_for_binding(request, job, None)
        .await
        .unwrap();
    let lease = prepared
        .prepared_fish_s2_tts_model_lease_for_executor()
        .unwrap()
        .unwrap();
    assert!(Arc::ptr_eq(&model_identity, &lease.model_arc()));
    assert!(Arc::ptr_eq(
        &artifact,
        &prepared
            .prepared_fish_s2_tts_artifact_for_executor()
            .unwrap()
            .unwrap()
    ));
    assert_eq!(prepared.id, request_id);
    assert_eq!(prepared.params.max_tokens, 32);
    assert_eq!(
        prepared
            .fish_s2_tts_generation_params_for_executor()
            .unwrap()
            .unwrap()
            .max_frames,
        32
    );
    assert!(prepared.streaming);
    assert!(!prepared.collect_audio_samples);
    assert_eq!(prepared.audio_stream_engine_queue_capacity, queue_capacity);
    assert_eq!(job.spec.request_id, request_id);
    drop(job);
    assert_eq!(runtime.coordinator_snapshot().active_jobs, 0);
}

#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn expired_streaming_deadline_does_not_invoke_callback() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let request_id = "expired-streaming-callback";
    let deadline = Instant::now() + Duration::from_millis(50);
    let (observation_request, mut guard, receiver, residency_variant) =
        pending_streaming_guard_fixture(&runtime, request_id, Some(deadline)).await;
    tokio::time::sleep_until(deadline.into()).await;

    let callback_invoked = Arc::new(AtomicBool::new(false));
    let callback_observer = callback_invoked.clone();
    let mut callback = move |_| {
        callback_observer.store(true, Ordering::Release);
        std::future::ready(Ok(()))
    };
    let chunk = StreamingOutput::new(request_id.to_string(), 0, vec![0.0], 24_000);

    let err = runtime
        .deliver_streaming_chunk_before_deadline(
            &mut callback,
            chunk,
            Some(deadline),
            request_id,
            &observation_request,
            &mut guard,
        )
        .await
        .expect_err("expired callback unexpectedly ran");
    assert!(matches!(err, Error::Timeout(id) if id == request_id));
    assert!(
        !callback_invoked.load(Ordering::Acquire),
        "an expired request invoked synchronous callback code"
    );
    assert!(tokio::time::timeout(Duration::from_secs(1), receiver)
        .await
        .expect("deadline cleanup did not remove its exact waiter")
        .is_err());
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if runtime.coordinator_snapshot().active_jobs == 0
                && runtime
                    .model_manager
                    .active_residency_leases(residency_variant)
                    == 0
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("deadline cleanup did not release request ownership");
}

#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hung_streaming_callback_is_bounded_by_absolute_request_deadline() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let request_id = "hung-streaming-callback-deadline";
    let deadline = Instant::now() + Duration::from_millis(25);
    let (observation_request, mut guard, receiver, residency_variant) =
        pending_streaming_guard_fixture(&runtime, request_id, Some(deadline)).await;
    let chunk = StreamingOutput::new(request_id.to_string(), 0, vec![0.0], 24_000);
    let mut callback = |_| std::future::pending::<Result<()>>();

    let err = tokio::time::timeout(
        Duration::from_secs(1),
        runtime.deliver_streaming_chunk_before_deadline(
            &mut callback,
            chunk,
            Some(deadline),
            request_id,
            &observation_request,
            &mut guard,
        ),
    )
    .await
    .expect("hung callback outlived the absolute request deadline")
    .expect_err("hung callback unexpectedly succeeded");
    assert!(matches!(err, Error::Timeout(id) if id == request_id));
    assert!(tokio::time::timeout(Duration::from_secs(1), receiver)
        .await
        .expect("deadline cleanup did not remove its exact waiter")
        .is_err());
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if runtime.coordinator_snapshot().active_jobs == 0
                && runtime
                    .model_manager
                    .active_residency_leases(residency_variant)
                    == 0
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("deadline cleanup did not release request ownership");
}

#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_callback_failure_returns_while_exact_abort_waits_for_core_step_lock() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let request_id = "streaming-callback-failure-during-step";
    let (observation_request, mut guard, receiver, residency_variant) =
        pending_streaming_guard_fixture(&runtime, request_id, None).await;
    let chunk = StreamingOutput::new(request_id.to_string(), 0, vec![0.0], 24_000);
    let (step_entered_tx, step_entered_rx) = oneshot::channel();
    let (release_step_tx, release_step_rx) = oneshot::channel();
    let engine = runtime.core_engine.clone();
    let step_lock = tokio::spawn(async move {
        engine
            .hold_core_step_lock_for_test(step_entered_tx, release_step_rx)
            .await;
    });
    step_entered_rx.await.expect("step lock was not acquired");
    let mut callback = |_| {
        std::future::ready(Err(Error::InferenceError(
            "streaming callback failed".to_string(),
        )))
    };

    let err = tokio::time::timeout(
        Duration::from_secs(1),
        runtime.deliver_streaming_chunk_before_deadline(
            &mut callback,
            chunk,
            None,
            request_id,
            &observation_request,
            &mut guard,
        ),
    )
    .await
    .expect("callback failure waited for the in-flight core step")
    .expect_err("failing callback unexpectedly succeeded");
    assert!(err.to_string().contains("streaming callback failed"));
    assert!(tokio::time::timeout(Duration::from_secs(1), receiver)
        .await
        .expect("detached cleanup did not remove its exact waiter")
        .is_err());
    assert_eq!(runtime.coordinator_snapshot().active_jobs, 1);
    assert_eq!(
        runtime
            .model_manager
            .active_residency_leases(residency_variant),
        1,
        "detached exact abort released residency while the core lock was held"
    );

    release_step_tx.send(()).expect("release step lock");
    step_lock.await.expect("step-lock task");
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if runtime.coordinator_snapshot().active_jobs == 0
                && runtime
                    .model_manager
                    .active_residency_leases(residency_variant)
                    == 0
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("exact abort did not release ownership after the core lock became safe");
}

#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn aborted_detached_cleanup_retains_request_ownership_fail_closed() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let mut request = EngineCoreRequest::tts("abort detached cancellation cleanup");
    request.id = "abort-detached-cleanup".to_string();
    request.prompt_tokens = vec![1];
    request.model_variant = Some(ModelVariant::Kokoro82M);
    let request_id = request.id.clone();
    let (spec, observation) = runtime
        .coordinator_job_for_request(&request)
        .expect("job shape");
    let isolated_coordinator = Arc::new(InferenceCoordinator::new(
        runtime.backend_context().backend_kind,
        1,
        1,
    ));
    let job = isolated_coordinator
        .admit_observed(spec, observation)
        .await
        .expect("isolated job admission");
    let (registration_id, receiver) = runtime
        .register_waiter(&request_id)
        .await
        .expect("waiter registration");
    let session = runtime
        .core_engine
        .add_request_with_session(request)
        .await
        .expect("engine admission");
    let residency_variant = ModelVariant::Kokoro82M;
    let residency_lease = runtime
        .model_manager
        .acquire_residency_lease(residency_variant);
    let ownership = DeferredRequestOwnership::new(Some(job), Some(residency_lease));
    let waiter_lock = runtime.completion_waiters.lock().await;
    let cleanup = tokio::spawn(cleanup_pending_request(
        session.clone(),
        runtime.core_engine.clone(),
        runtime.completion_waiters.clone(),
        registration_id,
        runtime.telemetry.clone(),
        ownership,
    ));

    tokio::task::yield_now().await;
    cleanup.abort();
    assert!(cleanup
        .await
        .expect_err("cleanup task unexpectedly completed")
        .is_cancelled());
    assert_eq!(isolated_coordinator.snapshot().active_jobs, 1);
    assert_eq!(
        runtime
            .model_manager
            .active_residency_leases(residency_variant),
        1,
        "aborting detached cleanup must retain model residency fail-closed"
    );

    drop(waiter_lock);
    assert!(
        remove_waiter_registration(
            runtime.completion_waiters.as_ref(),
            &request_id,
            registration_id,
        )
        .await
    );
    assert!(receiver.await.is_err());
    assert!(runtime
        .core_engine
        .abort_request_session(&session)
        .await
        .expect("manual exact abort"));
    assert_eq!(isolated_coordinator.snapshot().active_jobs, 1);
    assert_eq!(
        runtime
            .model_manager
            .active_residency_leases(residency_variant),
        1,
        "cancelled cleanup ownership is intentionally unrecoverable"
    );
}

#[test]
fn retained_chat_observation_uses_allocated_capacities_and_media_sources() {
    use crate::models::shared::chat::{ChatMediaInput, ChatMediaKind, ChatRole};

    let mut content = String::with_capacity(64);
    content.push('x');
    let mut messages = Vec::with_capacity(8);
    messages.push(ChatMessage {
        role: ChatRole::User,
        content,
    });

    let mut source = String::with_capacity(128);
    source.push_str("image.png");
    let mut media_inputs = Vec::with_capacity(3);
    media_inputs.push(ChatMediaInput {
        kind: ChatMediaKind::Image,
        source,
    });
    let config = ChatRequestConfig {
        media_inputs,
        ..ChatRequestConfig::default()
    };
    let correlation_id = {
        let mut value = String::with_capacity(32);
        value.push_str("correlation");
        value
    };

    let retained = retained_chat_preparation_input_bytes(
        &messages,
        messages.capacity(),
        &config,
        &GenerationParams::default(),
        Some(&correlation_id),
    )
    .unwrap();
    let minimum = messages.capacity() * std::mem::size_of::<ChatMessage>()
        + 64
        + config.media_inputs.capacity() * std::mem::size_of::<ChatMediaInput>()
        + 128
        + 32;

    assert_eq!(retained, minimum);
    assert!(retained > messages[0].content.len() + config.media_inputs[0].source.len());
}

#[test]
fn speech_to_speech_preparation_accounts_allocated_capacities() {
    use crate::models::shared::chat::ChatRole;

    let mut content = String::with_capacity(64);
    content.push_str("hello");
    let mut messages = Vec::with_capacity(8);
    messages.push(ChatMessage {
        role: ChatRole::User,
        content,
    });
    let mut speaker = String::with_capacity(32);
    speaker.push_str("voice");
    let params = GenerationParams {
        speaker: Some(speaker),
        ..GenerationParams::default()
    };

    let retained = retained_speech_to_speech_preparation_input_bytes(
        4,
        &messages,
        messages.capacity(),
        &params,
        Some("system"),
        Some("correlation"),
    )
    .expect("retained input");
    let minimum = 4
        + messages.capacity() * std::mem::size_of::<ChatMessage>()
        + 64
        + 32
        + "system".len()
        + "correlation".len();

    assert_eq!(retained, minimum);
}

#[tokio::test]
async fn runtime_routes_terminal_before_releasing_exact_session_fence() {
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let mut request = EngineCoreRequest::tts("expired terminal routing")
        .with_model_variant(ModelVariant::Kokoro82M);
    request.id = "runtime-terminal-routing".to_string();
    request.prompt_tokens = vec![1];

    let (registration, completion) = runtime
        .register_waiter(&request.id)
        .await
        .expect("waiter registration");
    runtime
        .core_engine
        .add_request(request.clone())
        .await
        .expect("request admission");
    let session = runtime
        .core_engine
        .request_session_key(&request.id)
        .await
        .expect("request session");
    runtime
        .bind_waiter(&request.id, registration, session.epoch)
        .await
        .expect("waiter session binding");
    assert!(
        runtime
            .core_engine
            .set_request_hard_deadline_for_test(
                &request.id,
                Instant::now() - Duration::from_millis(1),
            )
            .await
    );

    let outputs = runtime
        .core_engine
        .step_for_dispatch()
        .await
        .expect("terminal step");
    assert_eq!(outputs.len(), 1);
    assert!(outputs[0].is_finished);
    assert!(
        runtime
            .core_engine
            .add_request(request.clone())
            .await
            .is_err(),
        "the request ID must remain fenced while its terminal output awaits runtime routing"
    );

    route_terminal_output(
        runtime.core_engine.as_ref(),
        runtime.completion_waiters.as_ref(),
        runtime.telemetry.as_ref(),
        outputs.into_iter().next().unwrap(),
    )
    .await;
    assert!(matches!(
        completion.await.expect("completion channel"),
        Err(Error::Timeout(request_id)) if request_id == request.id
    ));

    runtime
        .core_engine
        .add_request(request)
        .await
        .expect("the request ID must be reusable after routing acknowledgement");
    runtime.core_engine.abort_all_requests().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fast_terminal_can_route_before_runtime_binds_atomic_admission_session() {
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let mut request = EngineCoreRequest::tts("fast terminal waiter binding")
        .with_model_variant(ModelVariant::Kokoro82M);
    request.id = "runtime-fast-terminal-binding".to_string();
    request.prompt_tokens = vec![1];

    let (registration, completion) = runtime
        .register_waiter(&request.id)
        .await
        .expect("waiter registration");
    let session = runtime
        .core_engine
        .add_request_with_session(request.clone())
        .await
        .expect("atomic request admission");
    assert!(
        runtime
            .core_engine
            .set_request_hard_deadline_for_test(
                &request.id,
                Instant::now() - Duration::from_millis(1),
            )
            .await
    );
    let output = runtime
        .core_engine
        .step_for_dispatch()
        .await
        .expect("fast terminal step")
        .into_iter()
        .next()
        .expect("terminal output");
    assert_eq!(output.sequence_id, session.epoch);

    let engine = runtime.core_engine.clone();
    let waiters = runtime.completion_waiters.clone();
    let telemetry = runtime.telemetry.clone();
    let routing = tokio::spawn(async move {
        route_terminal_output(
            engine.as_ref(),
            waiters.as_ref(),
            telemetry.as_ref(),
            output,
        )
        .await;
    });
    // Let routing observe the intentionally unbound registration first.
    tokio::task::yield_now().await;
    runtime
        .bind_waiter(&request.id, registration, session.epoch)
        .await
        .expect("waiter session binding");
    tokio::time::timeout(Duration::from_secs(1), routing)
        .await
        .expect("terminal routing waited forever for binding")
        .expect("routing task panicked");
    assert!(matches!(
        completion.await.expect("completion channel"),
        Err(Error::Timeout(request_id)) if request_id == request.id
    ));
}

#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stale_pending_guard_cleanup_cannot_abort_reused_session() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let mut old_request =
        EngineCoreRequest::tts("old exact session").with_model_variant(ModelVariant::Kokoro82M);
    old_request.id = "runtime-stale-pending-guard".to_string();
    old_request.prompt_tokens = vec![1];
    let request_id = old_request.id.clone();
    let (spec, observation) = runtime
        .coordinator_job_for_request(&old_request)
        .expect("job shape");
    let job = runtime
        .coordinator
        .admit_observed(spec, observation)
        .await
        .expect("job admission");
    let (registration, completion) = runtime
        .register_waiter(&request_id)
        .await
        .expect("waiter registration");
    let old_session = runtime
        .core_engine
        .add_request_with_session(old_request)
        .await
        .expect("old request admission");
    runtime
        .bind_waiter(&request_id, registration, old_session.epoch)
        .await
        .expect("old waiter binding");
    let guard = PendingRequestGuard::new(
        old_session.clone(),
        runtime.core_engine.clone(),
        runtime.completion_waiters.clone(),
        registration,
        runtime.telemetry.clone(),
        job,
        None,
    );

    assert!(runtime
        .core_engine
        .abort_request_session(&old_session)
        .await
        .expect("old exact abort"));
    let old_terminal = runtime
        .core_engine
        .step_for_dispatch()
        .await
        .expect("old cancellation step")
        .into_iter()
        .next()
        .expect("old cancellation output");
    route_terminal_output(
        runtime.core_engine.as_ref(),
        runtime.completion_waiters.as_ref(),
        runtime.telemetry.as_ref(),
        old_terminal,
    )
    .await;
    assert!(matches!(
        completion.await.expect("completion channel"),
        Err(Error::Cancelled(id)) if id == request_id
    ));

    let mut replacement = EngineCoreRequest::tts("replacement exact session")
        .with_model_variant(ModelVariant::Kokoro82M);
    replacement.id = request_id.clone();
    replacement.prompt_tokens = vec![2];
    let replacement_session = runtime
        .core_engine
        .add_request_with_session(replacement)
        .await
        .expect("replacement admission");
    assert_ne!(replacement_session.epoch, old_session.epoch);

    // The stale fallback runs after public-ID reuse. It must target only
    // the old epoch and leave the replacement request untouched.
    drop(guard);
    tokio::time::timeout(Duration::from_secs(1), async {
        while runtime.coordinator_snapshot().active_jobs != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("stale pending cleanup did not finish");
    assert_eq!(
        runtime.core_engine.request_session_key(&request_id).await,
        Some(replacement_session)
    );
    runtime.core_engine.abort_all_requests().await;
}

#[tokio::test]
async fn stale_terminal_output_cannot_steal_a_later_session_waiter() {
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let mut request = EngineCoreRequest::tts("stale terminal routing")
        .with_model_variant(ModelVariant::Kokoro82M);
    request.id = "runtime-stale-terminal-routing".to_string();
    request.prompt_tokens = vec![1];
    runtime
        .core_engine
        .add_request(request.clone())
        .await
        .expect("old request admission");
    assert!(
        runtime
            .core_engine
            .set_request_hard_deadline_for_test(
                &request.id,
                Instant::now() - Duration::from_millis(1),
            )
            .await
    );
    let old_output = runtime
        .core_engine
        .step_for_dispatch()
        .await
        .expect("old terminal step")
        .into_iter()
        .next()
        .expect("old terminal output");

    let later_epoch = old_output
        .sequence_id
        .checked_add(1)
        .expect("session epoch");
    let (later_registration, mut later_completion) = runtime
        .register_waiter(&request.id)
        .await
        .expect("later waiter registration");
    runtime
        .bind_waiter(&request.id, later_registration, later_epoch)
        .await
        .expect("later waiter binding");

    route_terminal_output(
        runtime.core_engine.as_ref(),
        runtime.completion_waiters.as_ref(),
        runtime.telemetry.as_ref(),
        old_output,
    )
    .await;

    assert!(matches!(
        later_completion.try_recv(),
        Err(tokio::sync::oneshot::error::TryRecvError::Empty)
    ));
    let waiters = runtime.completion_waiters.lock().await;
    let later = waiters.get(&request.id).expect("later waiter retained");
    assert_eq!(later.registration_id, later_registration);
    assert_eq!(later.session_epoch, Some(later_epoch));
    drop(waiters);
    runtime.remove_waiter(&request.id, later_registration).await;
}

#[test]
fn explicit_cuda_mismatch_gets_cuda_specific_error() {
    let context = BackendContext::new(
        BackendPreference::Cuda,
        BackendSelectionSource::Config,
        BackendCapabilities {
            cpu_compiled: true,
            metal_compiled: false,
            cuda_compiled: true,
        },
        DeviceProfile::cpu(),
        "Requested cuda backend fell back to cpu",
    );

    let err = RuntimeService::ensure_requested_backend_available(&context).unwrap_err();
    let message = err.to_string();

    assert!(message.contains("CUDA backend was requested"));
    assert!(message.contains("selected backend is `cpu`"));
    assert!(message.contains("no usable CUDA device"));
}

#[test]
fn assigned_cpu_runtime_uses_exact_production_selection() {
    let runtime =
        RuntimeService::new_assigned(EngineConfig::default(), RuntimeDeviceAssignment::Cpu)
            .expect("assigned CPU runtime");

    let context = runtime.backend_context();
    assert_eq!(context.backend_kind, BackendKind::Cpu);
    assert_eq!(context.preference, BackendPreference::Cpu);
    assert!(context.reason.contains("exact cpu device"));
}

#[tokio::test]
async fn runtime_concurrency_metrics_preserve_real_width_and_recovery_counts() {
    use crate::engine::metrics::{
        record_capacity_replay, record_capacity_suspension, record_engine_model_call,
        EngineModelCall,
    };
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let before = runtime.engine_telemetry_snapshot().await;
    record_engine_model_call(EngineModelCall::NativeTensor {
        mode: crate::engine::NativeBatchMode::Continuous,
        rows: 3,
    });
    record_capacity_suspension();
    record_capacity_replay(17);
    let after = runtime.engine_telemetry_snapshot().await;
    assert!(
        after
            .model_tensor_batch_width_counts
            .get(&3)
            .copied()
            .unwrap_or(0)
            > before
                .model_tensor_batch_width_counts
                .get(&3)
                .copied()
                .unwrap_or(0)
    );
    assert!(after.capacity_suspensions_total > before.capacity_suspensions_total);
    assert!(after.capacity_replay_tokens_total >= before.capacity_replay_tokens_total + 17);
    let json = serde_json::to_value(&after).expect("serialize concurrency metrics");
    assert_eq!(
        json["model_tensor_batch_width_counts"]["3"],
        serde_json::json!(after.model_tensor_batch_width_counts[&3])
    );
    assert_eq!(
        json["capacity_replay_tokens_total"],
        serde_json::json!(after.capacity_replay_tokens_total)
    );
    let payload = runtime.telemetry_prometheus().await;
    assert!(payload
        .contains("izwi_engine_executor_model_tensor_batch_width_calls_total{width=\"3\"}"));
    assert!(payload.contains("# TYPE izwi_engine_scheduler_capacity_suspensions_total counter"));
    assert!(
        payload.contains("# TYPE izwi_engine_scheduler_capacity_replay_tokens_total counter")
    );
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn runtime_prometheus_includes_engine_metric_values() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");

    let payload = runtime.telemetry_prometheus().await;

    // Every histogram in the whole exposition is well formed, and the
    // engine's step, batch and inter-token families are present.
    crate::runtime::prometheus::check_histograms(&payload).expect("valid histograms");
    for family in [
        "izwi_engine_inter_token_latency_seconds",
        "izwi_engine_step_host_seconds",
        "izwi_engine_batch_rows",
        "izwi_engine_batch_tokens",
        "izwi_engine_model_calls_per_step",
        "izwi_engine_mixed_rows",
        "izwi_engine_mixed_tokens",
        "izwi_engine_device_idle_seconds",
        "izwi_engine_device_queue_wait_seconds",
        "izwi_request_duration_seconds",
        "izwi_request_ttft_seconds",
        "izwi_request_queue_wait_seconds",
    ] {
        assert!(
            payload.contains(&format!("# TYPE {family} histogram")),
            "{family}"
        );
    }
    // Every TYPE line names a family once.
    let mut types = payload
        .lines()
        .filter(|line| line.starts_with("# TYPE "))
        .collect::<Vec<_>>();
    let total = types.len();
    types.sort_unstable();
    types.dedup();
    assert_eq!(types.len(), total, "duplicate metric families");
    assert!(payload.contains("izwi_engine_scheduler_queue_depth"));
    assert!(payload.contains("izwi_engine_scheduler_running_requests"));
    assert!(payload
        .contains("izwi_engine_kv_cache_allocated_blocks{accounting=\"physical_pages\"}"));
    assert!(payload
        .contains("izwi_engine_kv_cache_utilization_ratio{accounting=\"physical_pages\"}"));
    assert!(payload.contains(
        "izwi_engine_kv_cache_memory_capacity_bytes{accounting=\"resident_paged_plus_authorized_tensor\"}"
    ));
    assert!(payload.contains("allocated physical KV-cache pages"));
    assert!(payload
        .contains("Resident managed KV pages plus authorized retained tensor-state bytes"));
    assert!(!payload.contains("izwi_engine_kv_cache_soft_max_blocks"));
    assert!(!payload.contains("izwi_engine_kv_cache_copy_on_write_splits_total"));
    assert!(payload.contains("# TYPE izwi_engine_device_ops_off_thread_total counter"));
    assert!(payload.contains("izwi_engine_stream_backpressure_total"));
    assert!(payload.contains("izwi_engine_stream_checkpoints_committed_total"));
    assert!(payload.contains("izwi_engine_stream_checkpoint_rejections_total"));
    assert!(payload.contains("izwi_engine_stream_delivery_failures_total"));
    assert!(payload.contains("izwi_engine_executor_tensor_batches_total"));
    assert!(payload.contains("izwi_engine_executor_request_parallel_batches_total"));
    assert!(payload.contains("izwi_engine_executor_tensor_batch_max_width"));
    assert!(payload.contains("izwi_engine_executor_tensor_static_batches_total"));
    assert!(payload.contains("izwi_engine_executor_tensor_continuous_batches_total"));
    assert!(payload.contains("izwi_engine_executor_tensor_continuous_multirow_batches_total"));
    assert!(payload.contains("izwi_engine_executor_model_decode_calls_total"));
    assert!(payload.contains("izwi_engine_executor_model_tensor_batches_total"));
    assert!(payload.contains("izwi_engine_executor_model_tensor_multirow_calls_total"));
    assert!(payload.contains("izwi_engine_executor_model_tensor_batch_width_calls_total"));
    assert!(payload.contains("izwi_engine_scheduler_capacity_suspensions_total"));
    assert!(payload.contains("izwi_engine_scheduler_capacity_replay_tokens_total"));
    for kind in ["replay", "replay_capped", "recompute", "abort"] {
        assert!(payload.contains(&format!(
            "izwi_engine_scheduler_preemptions_total{{kind=\"{kind}\"}}"
        )));
    }
    assert!(payload.contains("# TYPE izwi_engine_scheduler_kv_blocked_requests gauge"));
    assert!(payload.contains("izwi_engine_kv_cache_prefix_query_tokens_total"));
    assert!(payload.contains("izwi_engine_kv_cache_prefix_hit_tokens_total"));
    assert!(payload.contains("izwi_engine_executor_model_tensor_batch_rows_total"));
    assert!(payload.contains("izwi_engine_executor_model_tensor_batch_max_width"));
    assert!(payload.contains("izwi_engine_executor_model_scalar_row_dispatches_total"));
    assert!(payload.contains("izwi_engine_executor_continuous_envelope_scalar_fallbacks_total"));
    assert!(payload.contains("izwi_engine_executor_physical_batch_rejections_total"));
    assert!(payload
        .contains("izwi_engine_executor_dispatch_state_rows_total{state=\"not_started\"}"));
    assert!(
        payload.contains("izwi_engine_executor_failure_origin_rows_total{origin=\"model\"}")
    );
    assert!(payload
        .contains("izwi_engine_executor_deadline_phase_rows_total{phase=\"dispatch_wait\"}"));
    assert!(payload.contains(
        "izwi_engine_executor_batch_workspace_domain_bytes_total{domain=\"device\"}"
    ));
    assert!(payload.contains("izwi_engine_executor_tensor_batch_fill_ratio"));
    assert!(payload.contains("izwi_engine_executor_tensor_batch_padding_ratio"));
    assert!(payload.contains("izwi_engine_executor_physical_execution_mode{mode=\"serial\"}"));
    assert!(payload.contains("izwi_engine_executor_physical_execution_cap"));
    assert!(payload.contains("izwi_engine_executor_physical_dispatches_in_flight"));
    assert!(payload.contains("izwi_engine_executor_physical_dispatch_seconds_total"));
    assert!(payload.contains(
        "izwi_engine_executor_physical_fallbacks_total{reason=\"uncertified_profile\"}"
    ));
    assert!(payload
        .contains("izwi_engine_executor_physical_defers_total{reason=\"workspace_capacity\"}"));
    assert!(payload.contains(
        "izwi_engine_executor_physical_workspace_high_water_bytes{domain=\"device\"}"
    ));
    assert!(payload.contains("izwi_engine_executor_physical_batch_fill_ratio"));
    assert!(payload.contains("# TYPE izwi_inference_coordinator_active_jobs gauge"));
    assert!(payload.contains("# TYPE izwi_inference_coordinator_admitted_total counter"));
    assert!(payload.contains("izwi_inference_coordinator_reserved_memory_bytes"));
    assert!(payload.contains("izwi_inference_coordinator_reserved_host_memory_bytes"));
    assert!(payload.contains("izwi_inference_coordinator_reserved_device_memory_bytes"));
    assert!(payload.contains("izwi_inference_coordinator_reserved_unified_memory_bytes"));
    assert!(payload.contains("izwi_inference_coordinator_materialized_device_memory_bytes"));
    assert!(payload.contains("izwi_inference_coordinator_pending_device_memory_bytes"));
    assert!(payload.contains("izwi_inference_coordinator_untracked_device_memory_bytes"));
    assert!(payload.contains("izwi_inference_coordinator_draining 0"));
    assert!(payload.contains("izwi_inference_coordinator_poisoned 0"));

    let snapshot = runtime.telemetry_snapshot().await;
    assert_eq!(
        snapshot.coordinator.reserved_memory_bytes,
        snapshot
            .coordinator
            .reserved_host_memory_bytes
            .saturating_add(snapshot.coordinator.reserved_device_memory_bytes)
            .saturating_add(snapshot.coordinator.reserved_unified_memory_bytes)
    );
    assert!(snapshot.coordinator.capacity >= 1);
    assert_eq!(snapshot.coordinator.active_jobs, 0);
    assert_eq!(snapshot.coordinator.active_preparation_bridges, 0);
    assert_eq!(snapshot.coordinator.active_model_loads, 0);
    assert_eq!(snapshot.coordinator.active_executions, 0);
    assert_eq!(snapshot.coordinator.admitted_total, 0);
    assert_eq!(snapshot.coordinator.rejected_total, 0);
    assert_eq!(snapshot.coordinator.expired_total, 0);
    assert!(!snapshot.coordinator.draining);
    assert!(!snapshot.coordinator.poisoned);
    assert!(snapshot.engine.physical_execution.effective_cap >= 1);
    let serialized = serde_json::to_value(&snapshot).expect("serialize runtime telemetry");
    let effective_mode = serialized["engine"]["physical_execution"]["effective_mode"]
        .as_str()
        .expect("bounded effective physical execution mode");
    assert!(["serial", "shadow", "concurrent"].contains(&effective_mode));
    assert_eq!(
        snapshot.engine.kv_cache.memory_accounting,
        "resident_paged_plus_authorized_tensor"
    );
    assert_eq!(snapshot.engine.kv_cache.totals.models, 0);
    assert_eq!(
        snapshot.engine.kv_cache.totals.coordinator.capacity_pages,
        0
    );
}

#[tokio::test]
async fn runtime_drain_is_observable_and_completes_when_idle() {
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");

    runtime
        .wait_for_drain(Duration::from_millis(50))
        .await
        .unwrap();

    assert!(runtime.is_draining());
    assert!(runtime.telemetry_snapshot().await.coordinator.draining);
    assert!(runtime
        .telemetry_prometheus()
        .await
        .contains("izwi_inference_coordinator_draining 1"));
}

#[test]
fn managed_kv_prometheus_projections_use_physical_arenas() {
    use crate::engine::{
        ManagedKvArenaRuntimeSnapshot, ManagedKvCoordinatorSnapshot,
        ManagedKvModelRuntimeSnapshot, ManagedKvOperationSnapshot, ModelInstanceId,
    };

    let arena = |allocated_pages| ManagedKvArenaRuntimeSnapshot {
        generation: 0,
        group_id: 1,
        domain_id: 1,
        device_ordinal: Some(0),
        page_tokens: 16,
        storage_dtype: "f16".to_string(),
        token_capacity: 160,
        bytes_per_page: 128,
        physical_bytes: 1_280,
        prefix_capacity_pages: 0,
        coordinator: ManagedKvCoordinatorSnapshot {
            capacity_pages: 10,
            allocated_pages,
            free_pages: 10 - allocated_pages,
            ..ManagedKvCoordinatorSnapshot::default()
        },
        operations: ManagedKvOperationSnapshot::default(),
    };
    let model = |model_instance, backend, allocated_pages| ManagedKvModelRuntimeSnapshot {
        model_instance: ModelInstanceId::new(model_instance),
        plan_fingerprint: format!("plan-{model_instance}"),
        state_plan_v2_fingerprint: format!("state-plan-v2-{model_instance}"),
        backend,
        device_ordinal: Some(0),
        resident_paged_bytes: 1_280,
        authorized_tensor_bytes: 0,
        materialized_tensor_bytes: 0,
        physical_bytes: 1_280,
        registered_sessions: 1,
        single_sequence_token_capacity: 160,
        aggregate_token_capacity: 160,
        full_context_sequence_capacity: 1,
        retained_sequence_capacity: None,
        staged_transaction_capacity: None,
        tensor_bytes_per_sequence: None,
        prefix_snapshot_capacity: None,
        prefix_snapshot_budget_bytes: 0,
        prefix_snapshots: 0,
        prefix_snapshot_bytes: 0,
        incremental_claim_sessions: 0,
        arenas: vec![arena(allocated_pages)],
    };
    let snapshot = crate::engine::ManagedKvRuntimeSnapshot {
        models: vec![
            model(1, BackendKind::Cpu, 3),
            model(2, BackendKind::Metal, 4),
            model(3, BackendKind::Cuda, 5),
        ],
        ..Default::default()
    };

    assert_eq!(managed_kv_used_bytes(&snapshot), 12 * 128);
    assert_eq!(managed_kv_device_pages(&snapshot), 9);

    // Each arena is its own series; pools are never summed across models.
    let mut payload = String::new();
    push_managed_kv_arena_metrics(&mut payload, &snapshot, |instance| match instance.get() {
        2 => "qwen\"metal".to_string(),
        other => format!("instance-{other}"),
    });
    assert!(payload.contains("# TYPE izwi_engine_kv_cache_model_allocated_pages gauge"));
    assert!(payload.contains(
        "izwi_engine_kv_cache_model_allocated_pages{model=\"instance-1\",backend=\"cpu\",domain=\"1\",storage_format=\"f16\"} 3"
    ));
    assert!(payload.contains(
        "izwi_engine_kv_cache_model_capacity_pages{model=\"qwen\\\"metal\",backend=\"metal\",domain=\"1\",storage_format=\"f16\"} 10"
    ));
    assert!(payload.contains(
        "izwi_engine_kv_cache_model_admission_available_pages{model=\"instance-3\",backend=\"cuda\""
    ));
}

#[tokio::test]
async fn streaming_requests_are_validated_as_streaming_by_broker() {
    let mut runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    runtime.inference_broker = InferenceBroker::with_mode(InferenceBrokerMode::On);
    let request =
        EngineCoreRequest::asr("audio").with_model_variant(ModelVariant::WhisperLargeV3Turbo);

    let err = runtime
        .run_streaming_request(request, |_| std::future::ready(Ok(())))
        .await
        .expect_err("batch-only ASR should be rejected before streaming execution");

    assert!(err.to_string().contains("not streaming execution"));

    let snapshot = runtime.telemetry_snapshot().await;
    assert_eq!(snapshot.observability.stage_observations_total, 1);
    assert_eq!(snapshot.observability.stage_failures_total, 1);
    assert_eq!(
        snapshot.observability.recent_stage_samples[0]
            .context
            .pipeline_stage
            .as_deref(),
        Some("runtime.routing")
    );
    assert_eq!(
        snapshot.observability.recent_stage_samples[0]
            .error_kind
            .as_deref(),
        Some("routing_validation_failed")
    );
}

#[tokio::test]
async fn transport_streaming_requests_can_validate_as_offline_broker_execution() {
    let mut runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    runtime.inference_broker = InferenceBroker::with_mode(InferenceBrokerMode::Shadow);
    let mut request =
        EngineCoreRequest::asr("audio").with_model_variant(ModelVariant::ParakeetTdt06BV3);
    request.streaming = true;

    runtime
        .observe_broker_request_with_streaming_required(&request, false)
        .expect("transport streaming should validate as offline ASR execution");

    let snapshot = runtime.telemetry_snapshot().await;
    assert_eq!(snapshot.broker.shadow_requests, 1);
    assert_eq!(snapshot.broker.route_decisions, 1);
    assert_eq!(snapshot.broker.validation_failures, 0);
    assert_eq!(snapshot.observability.stage_observations_total, 1);
    assert_eq!(snapshot.observability.stage_failures_total, 0);
    assert_eq!(
        snapshot.observability.recent_stage_samples[0]
            .context
            .execution_target
            .as_deref(),
        Some("TokenEngine")
    );
    assert_eq!(
        snapshot.observability.recent_stage_samples[0]
            .context
            .streaming_mode
            .as_deref(),
        Some("None")
    );
}

#[test]
fn parakeet_decode_budget_tracks_emittable_labels() {
    let variant = ModelVariant::ParakeetTdt06BV3;
    // 29.7 s at 16 kHz: 372 encoder frames × 10 labels.
    let max_emittable =
        crate::models::architectures::parakeet::asr::parakeet_max_emittable_tokens(
            475_200, 16_000, 10,
        )
        .unwrap();
    assert_eq!(max_emittable, 3_720);
    let request = |max_tokens: usize, auto: bool| {
        let mut request = EngineCoreRequest::asr("audio").with_model_variant(variant);
        request.params.max_tokens = max_tokens;
        request.asr_auto_max_tokens = auto;
        install_parakeet_decode_budget(&mut request, max_emittable).unwrap();
        request.params.max_tokens
    };

    assert_eq!(request(2048, true), max_emittable);
    assert_eq!(request(5, false), 5);
    assert_eq!(request(50_000, false), max_emittable);
}

async fn admitted_asr_job(runtime: &RuntimeService, request: &EngineCoreRequest) -> JobLease {
    let (spec, observation) = runtime
        .coordinator_job_for_request(request)
        .expect("ASR job");
    runtime
        .coordinator
        .admit_observed(spec, observation)
        .await
        .expect("ASR admission")
}

fn ten_second_wav(samples: impl FnOnce(usize) -> Vec<f32>) -> Vec<u8> {
    use crate::runtime::asr_no_speech::tests::{wav_bytes, SAMPLE_RATE};
    wav_bytes(&samples(SAMPLE_RATE as usize * 10), SAMPLE_RATE)
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn silent_asr_request_is_answered_before_route_preparation() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let request = EngineCoreRequest::asr_bytes(ten_second_wav(|len| vec![0.0; len]))
        .with_model_variant(ModelVariant::WhisperLargeV3Turbo);
    let job = admitted_asr_job(&runtime, &request).await;

    // No model is loaded, so route preparation would fail: the gate must
    // answer before it.
    let output = runtime
        .run_request_after_admission(request, job, None)
        .await
        .expect("gated transcript");

    assert_eq!(output.text.as_deref(), Some(""));
    assert!(output.is_finished);
    assert_eq!(output.input_audio_duration_secs, Some(10.0));
    let diagnostics = output.asr_diagnostics.expect("gate diagnostics");
    assert_eq!(diagnostics["no_speech_gate"]["gated"], true);
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn noise_only_streaming_asr_request_emits_a_terminal_empty_final() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    use crate::runtime::asr_no_speech::tests::pink_noise;

    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let mut request =
        EngineCoreRequest::asr_bytes(ten_second_wav(|len| pink_noise(len, 0.05, 4)))
            .with_model_variant(ModelVariant::Qwen3Asr17BGguf);
    request.streaming = true;
    let job = admitted_asr_job(&runtime, &request).await;

    let mut chunks = Vec::new();
    let output = runtime
        .run_streaming_request_after_admission(
            request,
            |chunk| {
                chunks.push(chunk);
                std::future::ready(Ok(()))
            },
            job,
            None,
            true,
        )
        .await
        .expect("gated stream");

    assert_eq!(chunks.len(), 1, "exactly one terminal marker");
    assert!(chunks[0].is_final);
    assert!(chunks[0].text.is_none());
    assert_eq!(output.text.as_deref(), Some(""));
    assert_eq!(output.input_audio_duration_secs, Some(10.0));
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn no_speech_gate_passes_speech_and_transducer_requests_through() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");

    let parakeet = EngineCoreRequest::asr_bytes(ten_second_wav(|len| vec![0.0; len]))
        .with_model_variant(ModelVariant::ParakeetTdt06BV3);
    let job = admitted_asr_job(&runtime, &parakeet).await;
    let (parakeet, gate) = runtime
        .asr_no_speech_gate(parakeet, &job)
        .await
        .expect("transducer gate decision");
    assert!(gate.is_none(), "transducers transcribe silence as empty");
    assert!(parakeet.prepared_asr_audio_for_executor().unwrap().is_none());

    let fox = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/fox.wav"),
    )
    .expect("speech fixture");
    let whisper =
        EngineCoreRequest::asr_bytes(fox).with_model_variant(ModelVariant::WhisperLargeV3Turbo);
    let job = admitted_asr_job(&runtime, &whisper).await;
    let (whisper, gate) = runtime
        .asr_no_speech_gate(whisper, &job)
        .await
        .expect("speech gate decision");
    assert!(gate.is_none(), "speech must reach the model");
    assert!(
        whisper.prepared_asr_audio_for_executor().unwrap().is_none(),
        "the gate's decode is transient; route preparation owns prepared audio"
    );
}

#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn whisper_transport_streaming_preserves_normal_and_long_form_routes() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let mut runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    runtime.inference_broker = InferenceBroker::with_mode(InferenceBrokerMode::Shadow);
    let variant = ModelVariant::WhisperLargeV3Turbo;

    let mut normal = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
    normal.streaming = true;
    normal
        .install_prepared_asr_audio(variant, vec![0.0; 160], 16_000)
        .unwrap();
    normal
        .install_prepared_sequence_input_tokens(16, 448)
        .unwrap();
    normal
        .install_prepared_whisper_window(
            variant,
            Arc::new(
                crate::models::architectures::whisper::asr::WhisperPreparedWindow::for_test(
                    4, 2, 8,
                )
                .unwrap(),
            ),
        )
        .unwrap();
    let (normal_spec, normal_observation) = runtime
        .coordinator_job_for_request(&normal)
        .expect("normal streaming job");
    let normal_job = runtime
        .coordinator
        .admit_observed(normal_spec, normal_observation)
        .await
        .expect("normal streaming admission");
    let (normal, normal_job) = runtime
        .prepare_asr_shape_for_binding(normal, normal_job, None)
        .await
        .expect("shared streaming preparation must preserve normal shape");
    drop(normal_job);
    assert_eq!(
        coordinator_lane_for_request(&normal),
        CoordinatorLane::Resumable
    );
    runtime
        .observe_broker_request_with_streaming_required(&normal, false)
        .expect("transport streaming must retain Whisper normal offline execution");

    let mut long_form = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
    long_form.streaming = true;
    long_form
        .install_prepared_asr_audio(variant, vec![0.0; 160], 16_000)
        .unwrap();
    long_form
        .install_prepared_asr_long_form_atomic(
            crate::engine::PlannedAsrChunks::unexecuted_for_tests(),
        )
        .unwrap();
    let (long_spec, long_observation) = runtime
        .coordinator_job_for_request(&long_form)
        .expect("long-form streaming job");
    let long_job = runtime
        .coordinator
        .admit_observed(long_spec, long_observation)
        .await
        .expect("long-form streaming admission");
    let (long_form, long_job) = runtime
        .prepare_asr_shape_for_binding(long_form, long_job, None)
        .await
        .expect("shared streaming preparation must preserve long-form shape");
    drop(long_job);
    assert_eq!(
        coordinator_lane_for_request(&long_form),
        CoordinatorLane::Atomic
    );
    runtime
        .observe_broker_request_with_streaming_required(&long_form, false)
        .expect("transport streaming must retain Whisper long-form offline execution");
}

#[tokio::test]
async fn voice_runtime_events_record_stage_observations() {
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");

    runtime.record_voice_session_started();

    let snapshot = runtime.telemetry_snapshot().await;
    assert_eq!(snapshot.voice.sessions_started, 1);
    assert_eq!(snapshot.observability.stage_observations_total, 1);
    assert_eq!(
        snapshot.observability.recent_stage_samples[0]
            .context
            .route_source
            .as_deref(),
        Some("RealtimeVoice")
    );
    assert_eq!(
        snapshot.observability.recent_stage_samples[0]
            .context
            .pipeline_stage
            .as_deref(),
        Some("voice.session_started")
    );
}

#[tokio::test]
async fn direct_capability_observation_records_broker_telemetry() {
    let mut runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    runtime.inference_broker = InferenceBroker::with_mode(InferenceBrokerMode::Shadow);

    runtime
        .observe_broker_capability_request(
            CapabilityKind::Tts,
            Some(ModelVariant::Kokoro82M),
            true,
        )
        .expect("direct capability observation should validate");

    let snapshot = runtime.telemetry_snapshot().await;
    assert_eq!(snapshot.broker.shadow_requests, 1);
    assert_eq!(snapshot.broker.execution_requests, 0);
    assert_eq!(snapshot.broker.route_decisions, 1);
    assert_eq!(snapshot.broker.validation_failures, 0);
    assert_eq!(snapshot.observability.stage_observations_total, 1);
    assert_eq!(snapshot.observability.stage_failures_total, 0);
    assert_eq!(
        snapshot.observability.recent_stage_samples[0]
            .context
            .pipeline_stage
            .as_deref(),
        Some("runtime.routing")
    );
    assert_eq!(
        snapshot.observability.recent_stage_samples[0]
            .context
            .model_variant
            .as_deref(),
        Some(ModelVariant::Kokoro82M.dir_name())
    );
}

#[tokio::test]
async fn batch_pipeline_observation_records_pipeline_telemetry() {
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");

    runtime.record_batch_asr_pipeline_job();
    runtime.record_batch_tts_pipeline_job();

    let snapshot = runtime.telemetry_snapshot().await;
    assert_eq!(snapshot.pipelines.batch_asr_transcriptions, 1);
    assert_eq!(snapshot.pipelines.batch_tts_speech, 1);
    assert_eq!(snapshot.pipelines.stages_recorded, 8);
}

/// Admitted CUDA requests hold no device headroom while they wait:
/// sixteen fit beside 256 MiB of free device memory (each used to claim
/// 64 MiB of it for its whole life, so the fifth was refused).
#[test]
fn concurrent_cuda_requests_do_not_claim_device_headroom() {
    use crate::engine::{
        CapacitySource, PhysicalCapacityProvider, PhysicalCapacitySnapshot, ReservationClass,
        ReservationOwner, ResourceAuthority,
    };
    #[derive(Debug)]
    struct FixedProvider(PhysicalCapacitySnapshot);
    impl PhysicalCapacityProvider for FixedProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            self.0
        }
    }
    let authority = Arc::new(ResourceAuthority::new(Arc::new(FixedProvider(
        PhysicalCapacitySnapshot {
            capacity: ResourceVector {
                host_bytes: ResourceAmount::Known(64 << 30),
                device_bytes: ResourceAmount::Known(24 << 30),
                ..ResourceVector::zero()
            },
            available: ResourceVector {
                host_bytes: ResourceAmount::Known(32 << 30),
                device_bytes: ResourceAmount::Known(256 << 20),
                ..ResourceVector::zero()
            },
            source: CapacitySource::Test,
        },
    ))));
    let requests = (0..16)
        .map(|index| {
            authority.reserve(
                ReservationOwner::new(ReservationClass::Request, format!("request-{index}")),
                transient_resources(BackendKind::Cuda, 4096),
            )
        })
        .collect::<Result<Vec<_>>>();
    assert!(requests.is_ok(), "{:?}", requests.err());
}

#[test]
fn transient_estimates_are_fully_known_for_every_backend() {
    const BASE_WORKSPACE_BYTES: u64 = 64 * 1024 * 1024;
    const INPUT_BYTES: usize = 1024;
    let host_preparation_bytes = BASE_WORKSPACE_BYTES + (INPUT_BYTES as u64 * 8);
    let cpu = transient_resources(BackendKind::Cpu, INPUT_BYTES);
    let metal = transient_resources(BackendKind::Metal, INPUT_BYTES);
    let cuda = transient_resources(BackendKind::Cuda, INPUT_BYTES);

    assert!(cpu.is_fully_known());
    assert!(metal.is_fully_known());
    assert!(cuda.is_fully_known());
    assert_eq!(
        cpu.host_bytes,
        ResourceAmount::Known(host_preparation_bytes)
    );
    assert_eq!(
        metal.unified_bytes,
        ResourceAmount::Known(host_preparation_bytes)
    );
    assert_eq!(
        cuda.host_bytes,
        ResourceAmount::Known(host_preparation_bytes)
    );
    // Device scratch is charged per dispatched row, not per request.
    assert_eq!(cuda.device_bytes, ResourceAmount::Known(0));
}

#[test]
fn media_preparation_estimates_map_to_physical_backend_domains() {
    let estimate = crate::models::architectures::qwen35::Qwen35MediaResourceEstimate {
        host_bytes: 300,
        backend_tensor_bytes: 700,
    };
    let cpu = media_preparation_resources(BackendKind::Cpu, estimate).unwrap();
    let metal = media_preparation_resources(BackendKind::Metal, estimate).unwrap();
    let cuda = media_preparation_resources(BackendKind::Cuda, estimate).unwrap();

    assert_eq!(cpu.host_bytes, ResourceAmount::Known(1_000));
    assert_eq!(metal.unified_bytes, ResourceAmount::Known(1_000));
    assert_eq!(cuda.host_bytes, ResourceAmount::Known(300));
    assert_eq!(cuda.device_bytes, ResourceAmount::Known(700));
    assert!(cpu.is_fully_known());
    assert!(metal.is_fully_known());
    assert!(cuda.is_fully_known());
}

#[test]
fn lfm25_audio_tts_preparation_charges_prompt_and_backend_workspace_exactly() {
    let messages = vec![
        ChatMessage {
            role: crate::models::shared::chat::ChatRole::System,
            content: String::from("system"),
        },
        ChatMessage {
            role: crate::models::shared::chat::ChatRole::User,
            content: String::from("speak this"),
        },
    ];
    let expected_host = messages.len() * std::mem::size_of::<ChatMessage>()
        + messages
            .iter()
            .map(|message| message.content.capacity())
            .sum::<usize>();
    assert_eq!(
        retained_lfm25_audio_tts_artifact_host_bytes(&messages).unwrap(),
        expected_host as u64
    );

    let cpu = lfm25_audio_tts_preparation_workspace(BackendKind::Cpu, 17);
    let metal = lfm25_audio_tts_preparation_workspace(BackendKind::Metal, 17);
    let cuda = lfm25_audio_tts_preparation_workspace(BackendKind::Cuda, 17);
    assert_eq!(cpu.host_bytes, ResourceAmount::Known(17));
    assert_eq!(metal.unified_bytes, ResourceAmount::Known(17));
    assert_eq!(cuda.device_bytes, ResourceAmount::Known(17));
    assert!(cpu.is_fully_known());
    assert!(metal.is_fully_known());
    assert!(cuda.is_fully_known());
}

#[test]
fn audio_decode_workspace_maps_to_host_or_unified_memory() {
    let cpu = audio_decode_resources(BackendKind::Cpu);
    let metal = audio_decode_resources(BackendKind::Metal);
    let cuda = audio_decode_resources(BackendKind::Cuda);

    assert_eq!(
        cpu.host_bytes,
        ResourceAmount::Known(AUDIO_DECODE_WORKSPACE_BYTES)
    );
    assert_eq!(
        metal.unified_bytes,
        ResourceAmount::Known(AUDIO_DECODE_WORKSPACE_BYTES)
    );
    assert_eq!(
        cuda.host_bytes,
        ResourceAmount::Known(AUDIO_DECODE_WORKSPACE_BYTES)
    );
    assert_eq!(cuda.device_bytes, ResourceAmount::Known(0));
    assert!(task_decodes_audio(TaskType::ASR));
    assert!(task_decodes_audio(TaskType::SpeechToSpeech));
    assert!(!task_decodes_audio(TaskType::TTS));
    assert!(!task_decodes_audio(TaskType::Chat));
}

#[test]
fn direct_audio_jobs_include_decoder_workspace_before_admission() {
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let backend = runtime.backend_context().backend_kind;
    let input_bytes = 4096;
    let context = RuntimeRequestContext::default();
    let expected = transient_resources(backend, input_bytes)
        .checked_add(audio_decode_resources(backend))
        .expect("resource estimate");
    let spec = runtime
        .coordinator_job_for_audio_input(
            "direct-asr-audio",
            CoordinatorLane::Atomic,
            context,
            input_bytes,
        )
        .expect("direct audio job");

    assert_eq!(spec.resources, expected);
}

#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "current_thread")]
async fn preparation_copy_yields_between_bounded_quanta() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    use std::sync::atomic::AtomicBool;

    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let input = vec![7_u8; PREPARATION_COPY_QUANTUM_BYTES * 3 + 17];
    let spec = runtime
        .coordinator_job_for_audio_input(
            "yielding-audio-copy",
            CoordinatorLane::Atomic,
            RuntimeRequestContext::default(),
            input.len(),
        )
        .expect("audio job");
    let job = runtime.coordinator.admit(spec).await.expect("admission");
    let peer_ran = Arc::new(AtomicBool::new(false));
    let task_peer_ran = peer_ran.clone();
    let peer = tokio::spawn(async move {
        task_peer_ran.store(true, Ordering::Release);
    });

    let copied = copy_preparation_bytes(&job, &input, "test audio")
        .await
        .expect("copy");
    assert!(
        peer_ran.load(Ordering::Acquire),
        "a multi-quantum copy must yield to another Tokio task"
    );
    let mut utf8_input = "x".repeat(PREPARATION_COPY_QUANTUM_BYTES - 1);
    utf8_input.push('é');
    let utf8_copy = copy_preparation_string(&job, &utf8_input, "test base64")
        .await
        .expect("UTF-8 copy");

    assert_eq!(copied, input);
    assert_eq!(utf8_copy, utf8_input);
    peer.await.expect("peer task");
}

#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "current_thread")]
async fn preparation_copy_stops_at_absolute_deadline() {
    let _authority_lock = crate::engine::resources::test_support::authority_test_lock()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let deadline = Instant::now() + Duration::from_millis(50);
    let context = RuntimeRequestContext::default().with_deadline(deadline);
    let spec = runtime
        .coordinator_job_for_audio_input(
            "expired-audio-copy",
            CoordinatorLane::Atomic,
            context,
            8,
        )
        .expect("audio job");
    let job = runtime.coordinator.admit(spec).await.expect("admission");
    tokio::time::sleep(Duration::from_millis(60)).await;

    let error = copy_preparation_bytes(&job, &[0_u8; 8], "expired audio")
        .await
        .expect_err("copy must honor its absolute deadline");

    assert!(matches!(error, Error::Timeout(request_id) if request_id == "expired-audio-copy"));
}

#[test]
fn core_audio_requests_include_decoder_workspace_in_job_authorization() {
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let backend = runtime.backend_context().backend_kind;

    let asr = EngineCoreRequest::asr("AAAA");
    let speech_to_speech = EngineCoreRequest::speech_to_speech("AAAA");
    let plain_tts = EngineCoreRequest::tts("hello");
    let mut reference_tts = EngineCoreRequest::tts("hello");
    reference_tts.reference_audio = Some("AAAA".to_string());
    reference_tts.reference_text = Some("reference transcript".to_string());
    let mut canonical_reference_tts = reference_tts.clone();
    canonical_reference_tts
        .canonicalize_direct_payloads(runtime.config.portable_context_ceiling())
        .expect("canonical TTS reference");
    assert!(canonical_reference_tts.reference_audio.is_none());
    assert!(canonical_reference_tts.has_tts_reference_for_execution());

    for request in [
        &asr,
        &speech_to_speech,
        &reference_tts,
        &canonical_reference_tts,
    ] {
        let input_bytes =
            retained_engine_request_input_bytes(request).expect("retained request input");
        let expected = transient_resources(backend, input_bytes)
            .checked_add(audio_decode_resources(backend))
            .expect("resource estimate");
        let (spec, _) = runtime
            .coordinator_job_for_request(request)
            .expect("coordinator job");

        assert_eq!(spec.resources, expected);
    }

    let plain_tts_input =
        retained_engine_request_input_bytes(&plain_tts).expect("retained request input");
    let expected_plain_tts = transient_resources(backend, plain_tts_input);
    let (plain_tts_spec, _) = runtime
        .coordinator_job_for_request(&plain_tts)
        .expect("coordinator job");
    assert_eq!(plain_tts_spec.resources, expected_plain_tts);
}

#[test]
fn max_num_seqs_sizes_every_automatic_limit() {
    let _guard = crate::env_test_lock().lock().expect("env lock poisoned");
    if std::env::var_os("IZWI_ASR_REALTIME_MAX_SESSIONS").is_some() {
        return;
    }
    let single = RuntimeService::new(EngineConfig {
        backend: crate::backends::BackendPreference::Cpu,
        ..Default::default()
    })
    .unwrap();
    let concurrency = single.resolved_concurrency();
    assert_eq!(
        concurrency.profile,
        crate::config::ConcurrencyProfile::SingleUser
    );
    assert_eq!(concurrency.max_num_seqs, None);
    assert_eq!(concurrency.scheduler_rows, 8);
    assert_eq!(concurrency.realtime_asr_sessions, 4);

    let multi = RuntimeService::new(EngineConfig {
        backend: crate::backends::BackendPreference::Cpu,
        max_num_seqs: Some(5),
        ..Default::default()
    })
    .unwrap();
    let concurrency = multi.resolved_concurrency();
    assert_eq!(
        concurrency.profile,
        crate::config::ConcurrencyProfile::MultiUser
    );
    assert_eq!(concurrency.max_num_seqs, Some(5));
    assert_eq!(concurrency.scheduler_rows, 5);
    assert_eq!(concurrency.retained_sequences, 5);
    assert_eq!(concurrency.tensor_width, 5);
    assert_eq!(concurrency.realtime_asr_sessions, 5);
    assert_eq!(multi.config().max_retained_sequences, 5);
    // The engine width is the knob's; each chat family still caps it.
    assert_eq!(multi.adapter_registry.max_chat_tensor_batch_size(), 5);

    // Explicit limits keep their values.
    let explicit = RuntimeService::new(EngineConfig {
        backend: crate::backends::BackendPreference::Cpu,
        max_num_seqs: Some(5),
        max_scheduler_batch_size: 3,
        max_batch_size: crate::config::BatchSizePreference::fixed(2).unwrap(),
        ..Default::default()
    })
    .unwrap();
    let concurrency = explicit.resolved_concurrency();
    assert_eq!(concurrency.scheduler_rows, 3);
    assert_eq!(concurrency.retained_sequences, 5);
    assert_eq!(concurrency.tensor_width, 2);

    // `max_num_seqs` keeps speech models at their certified CPU width; an
    // explicit width runs them (and chat) as asked, above the CPU ceiling.
    assert_eq!(multi.resolved_concurrency().non_chat_row_width, 2);
    let wide = RuntimeService::new(EngineConfig {
        backend: crate::backends::BackendPreference::Cpu,
        max_batch_size: crate::config::BatchSizePreference::fixed(12).unwrap(),
        ..Default::default()
    })
    .unwrap();
    let concurrency = wide.resolved_concurrency();
    assert_eq!(concurrency.tensor_width, 12);
    assert_eq!(concurrency.non_chat_row_width, 12);
    assert_eq!(wide.tts_row_width(), 12);
}

#[test]
fn automatic_staged_rows_resolve_to_the_step_launch_width() {
    // CUDA: width 8, one candidate, 316 memory-derived rows on an H100.
    assert_eq!(automatic_staged_transaction_rows(8, 1, 316), 8);
    // Shadow/concurrent candidates each hold their own launched rows.
    assert_eq!(automatic_staged_transaction_rows(8, 4, 316), 32);
    // Staged rows never exceed retained rows.
    assert_eq!(automatic_staged_transaction_rows(8, 1, 4), 4);

    let automatic = RuntimeService::new(EngineConfig {
        backend: crate::backends::BackendPreference::Cpu,
        max_retained_sequences: 0,
        max_staged_transactions: 0,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(automatic.config().max_retained_sequences, 8);
    // Engine rows and non-chat adapters stay at the portable width; chat
    // runs at the engine width, clamped per family, and sizes its own
    // staged rows.
    assert_eq!(automatic.adapter_registry.max_tensor_batch_size(), 2);
    assert_eq!(automatic.tts_row_width(), 2);
    assert_eq!(automatic.adapter_registry.max_chat_tensor_batch_size(), 8);
    assert_eq!(automatic.config().max_staged_transactions, 2);
    let core = automatic.core_engine.config();
    assert_eq!(core.max_staged_transactions, 2);
    assert!(core.automatic_row_limits.retained_sequences);
    assert!(core.automatic_row_limits.staged_transactions);

    let explicit = RuntimeService::new(EngineConfig {
        backend: crate::backends::BackendPreference::Cpu,
        max_staged_transactions: 3,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(explicit.config().max_staged_transactions, 3);
    let core = explicit.core_engine.config();
    assert_eq!(core.max_staged_transactions, 3);
    assert!(!core.automatic_row_limits.staged_transactions);

    // An operator's narrower width narrows the TTS rows too.
    let single = RuntimeService::new(EngineConfig {
        backend: crate::backends::BackendPreference::Cpu,
        max_batch_size: crate::config::BatchSizePreference::fixed(1).unwrap(),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(single.tts_row_width(), 1);
}

/// Unset step sizes take the backend's automatic budget and chunk; an
/// operator's values reach the engine core unchanged.
#[test]
fn step_budget_and_prefill_chunk_resolve_per_backend() {
    let mut automatic = EngineCoreConfig {
        backend: BackendKind::Cpu,
        ..Default::default()
    };
    automatic.resolve_step_sizes(0, 0);
    let service = RuntimeService::new(EngineConfig {
        backend: crate::backends::BackendPreference::Cpu,
        ..Default::default()
    })
    .unwrap();
    let core = service.core_engine.config();
    assert_eq!(core.max_tokens_per_step, automatic.max_tokens_per_step);
    assert_eq!(
        core.chunked_prefill_threshold,
        automatic.chunked_prefill_threshold
    );

    let explicit = RuntimeService::new(EngineConfig {
        backend: crate::backends::BackendPreference::Cpu,
        max_tokens_per_step: 640,
        chunked_prefill_threshold: 96,
        ..Default::default()
    })
    .unwrap();
    let core = explicit.core_engine.config();
    assert_eq!(core.max_tokens_per_step, 640);
    assert_eq!(core.chunked_prefill_threshold, 96);
}

#[test]
fn fish_streaming_output_admission_is_bounded_independently_of_length() {
    let runtime = RuntimeService::new(EngineConfig {
        backend: crate::backends::BackendPreference::Cpu,
        ..Default::default()
    })
    .unwrap();
    let mut short =
        EngineCoreRequest::tts("hello").with_model_variant(ModelVariant::FishAudioS2Pro);
    short.params.max_tokens = 32;
    let mut long = short.clone();
    long.params.max_tokens = ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES;
    let whole_short = runtime
        .coordinator_job_for_request(&short)
        .unwrap()
        .0
        .resources;
    let whole_long = runtime
        .coordinator_job_for_request(&long)
        .unwrap()
        .0
        .resources;
    assert_ne!(whole_short, whole_long);
    short.mark_audio_streaming_only();
    long.mark_audio_streaming_only();
    let stream_short = runtime
        .coordinator_job_for_request(&short)
        .unwrap()
        .0
        .resources;
    let stream_long = runtime
        .coordinator_job_for_request(&long)
        .unwrap()
        .0
        .resources;
    assert_eq!(stream_short, stream_long);
    assert_ne!(stream_long, whole_long);
    let before = long.streaming_audio_buffer_bytes(16 * 2048).unwrap();
    long.audio_stream_external_queue_capacity = 128;
    let after = long.streaming_audio_buffer_bytes(16 * 2048).unwrap();
    assert_eq!(
        after - before,
        128 * (16 * 2048 * 4 + long.id.len() as u64 + 512)
    );
    assert_ne!(
        runtime
            .coordinator_job_for_request(&long)
            .unwrap()
            .0
            .resources,
        stream_long
    );
    long.audio_stream_engine_queue_capacity = Some(17);
    assert_eq!(CoreEngine::streaming_queue_capacity(&long), 17);
    long.audio_stream_external_queue_capacity = usize::MAX;
    assert!(long.streaming_audio_buffer_bytes(16 * 2048).is_err());
}

#[test]
fn asr_task_options_count_toward_the_retained_request() {
    let request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]);
    let before = retained_engine_request_input_bytes(&request).unwrap();
    let task = crate::engine::AsrTaskOptions::new(
        Some("English".to_string()),
        vec!["Izwi".to_string(), "Granite".to_string()],
    );
    let task_bytes = task.translate_to.as_ref().unwrap().capacity()
        + task.keywords.capacity() * std::mem::size_of::<String>()
        + task.keywords.iter().map(String::capacity).sum::<usize>();
    let request = request.with_asr_task(task);
    let after = retained_engine_request_input_bytes(&request).unwrap();
    assert_eq!(after - before, task_bytes);
    assert_eq!(request.asr_task.retained_bytes(), task_bytes);
}

#[test]
fn direct_qwen_asr_job_accounts_the_prepared_audio_artifact_exactly() {
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let backend = runtime.backend_context().backend_kind;
    let variant = ModelVariant::Qwen3Asr06BGguf;
    let mut request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
    let before = retained_engine_request_input_bytes(&request).unwrap();
    request
        .install_prepared_asr_audio(variant, vec![0.0; 257], 16_000)
        .unwrap();
    request
        .install_prepared_sequence_input_tokens(32, 4096)
        .unwrap();
    let after = retained_engine_request_input_bytes(&request).unwrap();
    assert_eq!(after - before, 257 * std::mem::size_of::<f32>());

    let (spec, observation) = runtime
        .coordinator_job_for_request(&request)
        .expect("prepared Qwen ASR coordinator job");
    assert_eq!(observation.host_bytes, after as u64);
    assert_eq!(
        Some(spec.resources),
        asr_transient_resources(backend, variant, after, Some(257.0 / 16_000.0)).unwrap()
    );
}

/// KOK-8: an engine Kokoro request reserves its widest prepared chunk's
/// workspace and the joined output of its chunks at the planning pace,
/// not 4,096 frames and the 50-bins-per-token output of its whole text.
#[test]
fn kokoro_engine_reservation_follows_the_prepared_chunks() {
    use crate::models::architectures::kokoro::admission::kokoro_chunk_price;
    use crate::models::architectures::kokoro::KokoroPreparedRequest;
    let chunk = |text: &str| KokoroPreparedRequest {
        source_text: text.into(),
        requested_speaker: None,
        requested_language: None,
        requested_speed: 1.0,
        phonemes: "p".into(),
        token_ids: vec![1; 8],
        ref_style: candle_core::Tensor::zeros(
            (1, 4),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap(),
        speed: 1.0,
        rng_seed: None,
    };
    let prepared = crate::engine::KokoroPreparedText {
        source_text: "ab".into(),
        requested_speaker: None,
        requested_language: None,
        requested_speed: 1.0,
        chunks: vec![chunk(&"a".repeat(100)), chunk(&"b".repeat(40))],
    };
    // 64 frames per chunk plus four per character.
    let wide = kokoro_chunk_price(64 + 400, BackendKind::Metal).unwrap();
    let narrow = kokoro_chunk_price(64 + 160, BackendKind::Metal).unwrap();
    let output = (wide.output_samples + narrow.output_samples + 960) * 4;
    assert_eq!(
        kokoro_synthesis_resources(BackendKind::Metal, &prepared)
            .unwrap()
            .unified_bytes,
        ResourceAmount::Known(output + wide.host_bytes + wide.accelerator_tensor_bytes)
    );
}

/// TPIPE-24: unset speech sampling keeps each family's reference
/// default instead of the shared chat defaults, and request fields
/// override it.
#[test]
fn speech_family_params_keep_reference_defaults_until_the_request_sets_them() {
    let unset = GenerationParams::default();
    let vibevoice = vibevoice_tts_request_params(&unset, "Hello there.", 10);
    assert_eq!((vibevoice.cfg_scale, vibevoice.diffusion_steps), (1.3, 10));
    let fish = fish_s2_request_params(&unset, "request-a");
    assert_eq!((fish.temperature, fish.top_p, fish.top_k), (0.7, 0.7, 30));
    assert_eq!(fish.seed, EngineCoreRequest::chat_request_seed("request-a"));
    let voxtral = voxtral_tts_request_params(&unset, "request-a", 4096);
    assert_eq!(
        (voxtral.temperature, voxtral.top_p, voxtral.top_k),
        (0.0, 1.0, 0)
    );
    assert_ne!(voxtral.seed, 0, "retried frames redraw the same noise");

    let mut set = GenerationParams::default();
    set.speech.cfg_scale = Some(2.0);
    set.speech.diffusion_steps = Some(20);
    set.speech.temperature = Some(0.5);
    set.speech.top_p = Some(0.9);
    set.speech.top_k = Some(12);
    set.speech.seed = Some(99);
    let vibevoice = vibevoice_tts_request_params(&set, "Hello there.", 10);
    assert_eq!((vibevoice.cfg_scale, vibevoice.diffusion_steps), (2.0, 20));
    let fish = fish_s2_request_params(&set, "request-a");
    assert_eq!(
        (fish.temperature, fish.top_p, fish.top_k, fish.seed),
        (0.5, 0.9, 12, 99)
    );
    let voxtral = voxtral_tts_request_params(&set, "request-a", 4096);
    assert_eq!(
        (
            voxtral.temperature,
            voxtral.top_p,
            voxtral.top_k,
            voxtral.seed
        ),
        (0.5, 0.9, 12, 99)
    );
}

/// Engine output errors keep their kind: a request beyond its fitted
/// state is a capacity refusal (503), a malformed one an input error
/// (400); only untyped failures stay inference errors (500), and an
/// inference error is not prefixed twice.
#[test]
fn engine_output_errors_keep_their_kind() {
    let failed = |error: String| {
        let mut output = terminal_output(ExecutionFinishReason::Completed);
        output.error = Some(error);
        runtime_completion(output).unwrap_err()
    };
    let capacity =
        Error::Overloaded("physical paged append ends at 324, beyond capacity 320".into());
    assert!(matches!(
        failed(capacity.to_string()),
        Error::Overloaded(detail) if detail.ends_with("beyond capacity 320")
    ));
    assert!(matches!(
        failed(Error::InvalidInput("bad audio".into()).to_string()),
        Error::InvalidInput(detail) if detail == "bad audio"
    ));
    assert!(matches!(
        failed(Error::Backpressure("slow".into()).to_string()),
        Error::Backpressure(_)
    ));
    assert!(matches!(
        failed("kernel failed".into()),
        Error::InferenceError(detail) if detail == "kernel failed"
    ));
    let device = Error::InferenceError("unsupported const-set f64".into());
    assert_eq!(
        failed(device.to_string()).to_string(),
        "Inference error: unsupported const-set f64"
    );
}

/// ASR requests are priced by the model's envelope over their audio, not
/// by a multiple of their input: an hour of decoded audio costs one
/// chunk's working set beside the retained samples (PIPE-15).
#[test]
fn asr_requests_price_the_model_envelope_instead_of_an_input_multiple() {
    const HOUR_SAMPLES: usize = 3_600 * 16_000;
    let input = HOUR_SAMPLES * std::mem::size_of::<f32>();
    for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
        let generic = transient_resources(backend, input);
        for variant in [
            ModelVariant::Qwen3Asr06BGguf,
            ModelVariant::WhisperLargeV3Turbo,
            ModelVariant::Nemotron35AsrStreaming06B,
        ] {
            let hour = asr_transient_resources(backend, variant, input, Some(3_600.0))
                .unwrap()
                .unwrap();
            let chunk = asr_transient_resources(backend, variant, input, None)
                .unwrap()
                .unwrap();
            assert_eq!(hour, chunk, "{variant} {backend:?}");
            assert!(
                hour.workspace_bytes().unwrap() < generic.workspace_bytes().unwrap(),
                "{variant} {backend:?}"
            );
            let bytes = |resources: ResourceVector| match backend {
                BackendKind::Cpu => resources.host_bytes,
                BackendKind::Metal => resources.unified_bytes,
                BackendKind::Cuda => resources.host_bytes,
            };
            assert!(
                matches!(bytes(hour), ResourceAmount::Known(value) if value > input as u64)
            );
        }
    }
    assert_eq!(
        asr_transient_resources(BackendKind::Cpu, ModelVariant::Kokoro82M, 1, None).unwrap(),
        None
    );
}

#[test]
fn direct_whisper_asr_job_accounts_prepared_audio_and_encoder_artifact_exactly() {
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let backend = runtime.backend_context().backend_kind;
    let variant = ModelVariant::WhisperLargeV3Turbo;
    let mut request = EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant);
    request
        .install_prepared_asr_audio(variant, vec![0.0; 257], 16_000)
        .unwrap();
    request
        .install_prepared_sequence_input_tokens(32, 448)
        .unwrap();
    let artifact = Arc::new(
        crate::models::architectures::whisper::asr::WhisperPreparedWindow::for_test(7, 4, 16)
            .unwrap(),
    );
    let encoder_bytes = artifact.resident_tensor_bytes().unwrap();
    request
        .install_prepared_whisper_window(variant, artifact)
        .unwrap();

    let host_bytes = retained_engine_request_input_bytes(&request).unwrap();
    let expected =
        asr_transient_resources(backend, variant, host_bytes, Some(257.0 / 16_000.0))
            .unwrap()
            .unwrap()
            .checked_add(asr_encoder_retained_resources(backend, 0, encoder_bytes).unwrap())
            .unwrap();
    let (spec, observation) = runtime
        .coordinator_job_for_request(&request)
        .expect("prepared Whisper ASR coordinator job");

    assert_eq!(observation.host_bytes, host_bytes as u64);
    assert_eq!(spec.resources, expected);
}

#[test]
fn engine_requests_use_truthful_controller_lanes() {
    let mut qwen_tts = EngineCoreRequest::tts("hello");
    qwen_tts.model_variant = Some(ModelVariant::Qwen3Tts12Hz06BBase);
    assert_eq!(
        coordinator_lane_for_request(&qwen_tts),
        CoordinatorLane::Resumable
    );

    let mut offline_asr = EngineCoreRequest::asr("audio");
    offline_asr.model_variant = Some(ModelVariant::Qwen3Asr06BGguf);
    assert_eq!(
        coordinator_lane_for_request(&offline_asr),
        CoordinatorLane::Atomic
    );
    offline_asr.streaming = true;
    assert_eq!(
        coordinator_lane_for_request(&offline_asr),
        CoordinatorLane::Atomic
    );
    offline_asr
        .install_prepared_asr_audio(ModelVariant::Qwen3Asr06BGguf, vec![0.0; 16_000], 16_000)
        .unwrap();
    offline_asr
        .install_prepared_asr_long_form_atomic(
            crate::engine::PlannedAsrChunks::unexecuted_for_tests(),
        )
        .unwrap();
    assert_eq!(
        coordinator_lane_for_request(&offline_asr),
        CoordinatorLane::Atomic
    );
    offline_asr.workload_class = WorkloadClass::Realtime;
    assert_eq!(
        coordinator_lane_for_request(&offline_asr),
        CoordinatorLane::Realtime
    );

    let whisper_variant = ModelVariant::WhisperLargeV3Turbo;
    let mut whisper = EngineCoreRequest::asr("audio").with_model_variant(whisper_variant);
    assert_eq!(
        coordinator_lane_for_request(&whisper),
        CoordinatorLane::Atomic
    );
    whisper
        .install_prepared_sequence_input_tokens(16, 448)
        .unwrap();
    assert_eq!(
        coordinator_lane_for_request(&whisper),
        CoordinatorLane::Resumable
    );
    let mut whisper_long = EngineCoreRequest::asr("audio").with_model_variant(whisper_variant);
    whisper_long
        .install_prepared_asr_long_form_atomic(
            crate::engine::PlannedAsrChunks::unexecuted_for_tests(),
        )
        .unwrap();
    assert_eq!(
        coordinator_lane_for_request(&whisper_long),
        CoordinatorLane::Atomic
    );

    let vibe_variant = ModelVariant::VibeVoiceAsr;
    let mut vibe = EngineCoreRequest::asr("audio").with_model_variant(vibe_variant);
    assert_eq!(coordinator_lane_for_request(&vibe), CoordinatorLane::Atomic);
    vibe.install_prepared_sequence_input_tokens(32, 4_096)
        .unwrap();
    assert_eq!(
        coordinator_lane_for_request(&vibe),
        CoordinatorLane::Resumable
    );
    let mut vibe_long = EngineCoreRequest::asr("audio").with_model_variant(vibe_variant);
    vibe_long
        .install_prepared_asr_long_form_atomic(
            crate::engine::PlannedAsrChunks::unexecuted_for_tests(),
        )
        .unwrap();
    assert_eq!(
        coordinator_lane_for_request(&vibe_long),
        CoordinatorLane::Atomic
    );

    let lfm_variant = ModelVariant::Lfm25Audio15BGguf;
    let mut lfm = EngineCoreRequest::asr("audio").with_model_variant(lfm_variant);
    assert_eq!(coordinator_lane_for_request(&lfm), CoordinatorLane::Atomic);
    lfm.install_prepared_sequence_input_tokens(32, 4_096)
        .unwrap();
    assert_eq!(
        coordinator_lane_for_request(&lfm),
        CoordinatorLane::Resumable
    );
    let mut lfm_long = EngineCoreRequest::asr("audio").with_model_variant(lfm_variant);
    lfm_long
        .install_prepared_asr_long_form_atomic(
            crate::engine::PlannedAsrChunks::unexecuted_for_tests(),
        )
        .unwrap();
    assert_eq!(
        coordinator_lane_for_request(&lfm_long),
        CoordinatorLane::Atomic
    );
}

#[test]
fn qwen_tts_coordinator_reserves_request_max_output_exactly() {
    let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
    let backend = runtime.backend_context().backend_kind;
    let mut request = EngineCoreRequest::tts("hello");
    request.model_variant = Some(ModelVariant::Qwen3Tts12Hz06BBase);
    let output_bytes = (request.qwen_tts_generation_params().max_frames as u64)
        * 1_920
        * std::mem::size_of::<f32>() as u64;
    let mut output = ResourceVector::zero();
    match backend {
        BackendKind::Metal => output.unified_bytes = ResourceAmount::Known(output_bytes),
        BackendKind::Cpu | BackendKind::Cuda => {
            output.host_bytes = ResourceAmount::Known(output_bytes)
        }
    }
    let mut plain = request.clone();
    plain.model_variant = None;
    let (plain_spec, _) = runtime.coordinator_job_for_request(&plain).unwrap();
    let expected = plain_spec.resources.checked_add(output).unwrap();
    let (spec, _) = runtime.coordinator_job_for_request(&request).unwrap();
    assert_eq!(spec.resources, expected);
}
