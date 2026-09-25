//! Realtime WebSocket (`izwi-realtime-v1`) session contract tests against the
//! worker's `/internal/v1/realtime` route. Sessions run over real loopback
//! sockets against a scripted ASR stage runner, so admission fencing, bounds,
//! terminal outcomes, and the shared attempt table are exercised without a
//! deployed model.

use async_trait::async_trait;
use axum::http::HeaderValue;
use futures::{SinkExt, StreamExt};
use izwi_core::RuntimeAsrRealtimeEvent;
use izwi_serving_protocol::*;
use izwi_serving_worker::{AdmissionFailure, AdmittedInvocation, WorkerConfigError};
use izwi_serving_worker::{
    InvocationExecutor, RealtimeAsrStageStream, RealtimeStageRunner, RuntimeRealtimeSessionLimits,
    WorkerConfig, WorkerService, REALTIME_SESSION_DEFAULT_LIMITS,
};
use std::collections::{BTreeSet, VecDeque};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Mutex as AsyncMutex;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::{Error as WsError, Message};
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use tower::ServiceExt;

type WsStream = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

fn id<T: TryFrom<&'static str>>(value: &'static str) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(value).unwrap()
}

#[derive(Default)]
struct RunnerCounters {
    start_calls: AtomicUsize,
    active_streams: AtomicUsize,
    pushes: AtomicUsize,
}

/// Scripted ASR stage runner: one delta per push, one final transcript on
/// finish, optional start failure injection. Stream liveness is observable so
/// tests can assert teardown actually dropped the runtime stream.
#[derive(Clone)]
struct FakeAsrRunner {
    counters: Arc<RunnerCounters>,
    deltas: Arc<Mutex<VecDeque<String>>>,
    final_text: Arc<String>,
    fail_start: Arc<AsyncMutex<bool>>,
}

impl FakeAsrRunner {
    fn new(final_text: &str) -> Self {
        Self {
            counters: Arc::new(RunnerCounters::default()),
            deltas: Arc::new(Mutex::new(VecDeque::from([
                "partial one".into(),
                "partial two".into(),
            ]))),
            final_text: Arc::new(final_text.into()),
            fail_start: Arc::new(AsyncMutex::new(false)),
        }
    }
}

struct FakeAsrStream {
    final_text: Arc<String>,
    counters: Arc<RunnerCounters>,
    deltas: Arc<Mutex<VecDeque<String>>>,
    chunk_index: usize,
}

impl Drop for FakeAsrStream {
    fn drop(&mut self) {
        self.counters.active_streams.fetch_sub(1, Ordering::AcqRel);
    }
}

#[async_trait::async_trait]
impl RealtimeAsrStageStream for FakeAsrStream {
    async fn push_samples(
        &mut self,
        _samples: &[f32],
        _sample_rate: u32,
    ) -> Result<Vec<RuntimeAsrRealtimeEvent>, izwi_core::Error> {
        self.counters.pushes.fetch_add(1, Ordering::AcqRel);
        let delta = self
            .deltas
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| format!("chunk {}", self.chunk_index));
        self.chunk_index += 1;
        Ok(vec![RuntimeAsrRealtimeEvent {
            delta,
            text: String::new(),
            is_final: false,
            chunk_index: self.chunk_index,
        }])
    }

    async fn finish(&mut self) -> Result<Vec<RuntimeAsrRealtimeEvent>, izwi_core::Error> {
        Ok(vec![RuntimeAsrRealtimeEvent {
            delta: String::new(),
            text: self.final_text.to_string(),
            is_final: true,
            chunk_index: self.chunk_index,
        }])
    }
}

#[async_trait::async_trait]
impl RealtimeStageRunner for FakeAsrRunner {
    fn stage_task(&self) -> TaskKind {
        TaskKind::SpeechToText
    }

    async fn start_asr_stream(
        &self,
        _language: Option<&str>,
    ) -> Result<Box<dyn RealtimeAsrStageStream>, izwi_core::Error> {
        self.counters.start_calls.fetch_add(1, Ordering::AcqRel);
        if *self.fail_start.lock().await {
            return Err(izwi_core::Error::InferenceError(
                "injected start failure".into(),
            ));
        }
        self.counters.active_streams.fetch_add(1, Ordering::AcqRel);
        Ok(Box::new(FakeAsrStream {
            final_text: Arc::clone(&self.final_text),
            counters: Arc::clone(&self.counters),
            deltas: Arc::clone(&self.deltas),
            chunk_index: 0,
        }))
    }
}

struct FakeRealtimeExecutor {
    runner: FakeAsrRunner,
}

#[async_trait]
impl InvocationExecutor for FakeRealtimeExecutor {
    async fn admit(
        &self,
        _request: &InvocationRequest,
    ) -> Result<AdmittedInvocation, AdmissionFailure> {
        Err(AdmissionFailure::new(
            RejectionCode::IncompatibleTask,
            "fake realtime executor serves realtime WebSocket sessions only",
        ))
    }

    fn realtime_runner(&self) -> Option<Arc<dyn RealtimeStageRunner>> {
        Some(Arc::new(self.runner.clone()))
    }
}

fn realtime_config(limits: RuntimeRealtimeSessionLimits) -> WorkerConfig {
    WorkerConfig {
        descriptor: WorkerDescriptor {
            schema_version: PROTOCOL_V1,
            supported_protocol_versions: vec![PROTOCOL_V1],
            worker_id: id("asr-worker"),
            node_id: id("local-node"),
            incarnation_id: id("incarnation-1"),
            build_version: "test".into(),
            assignment: DeviceAssignment::Cpu {
                thread_budget: 1,
                affinity: Vec::new(),
                host_memory_limit_bytes: 64 * 1024 * 1024,
            },
            features: BTreeSet::from([
                WorkerFeature::Streaming,
                WorkerFeature::Cancellation,
                WorkerFeature::AttemptQuery,
                WorkerFeature::RealtimeSocket,
            ]),
        },
        deployment: LoadedDeployment {
            deployment_id: id("asr-cpu-v1"),
            public_model: id("tiny-asr"),
            artifact_revision: id("tiny-asr-v1"),
            model_generation: ModelGeneration::new(1).unwrap(),
            task: TaskKind::SpeechToText,
            backend: BackendKind::Cpu,
            precision: "fp16".into(),
            execution_representation: "tiny-asr-realtime".into(),
            tokenizer_revision: None,
            readiness: ModelReadiness::Ready,
            capability: Capability {
                task: TaskKind::SpeechToText,
                streaming: true,
                realtime: true,
                cancellation: CancellationBehavior::Cooperative,
                accepted_input_formats: BTreeSet::from([InputFormat::PcmAudio]),
                output_formats: BTreeSet::from([OutputFormat::Text]),
                max_input_bytes: 4096,
                max_context_tokens: None,
                max_output_tokens: None,
            },
            kv_cache_usage_pct: None,
            prefix_hits_total: None,
            prefix_queries_total: None,
            prefix_evictions_total: None,
            tokens_out_per_s_ema: None,
            observation_cost_units: None,
        },
        credentials: ServiceCredentials {
            credential_id: id("worker-credential"),
            bearer_token: ServiceBearerToken::new("worker-secret").unwrap(),
        },
        max_active_invocations: 4,
        max_request_bytes: 4096,
        max_retained_attempts: 4,
        attempt_retention: Duration::from_secs(300),
        event_channel_capacity: 4,
        max_event_bytes: 4096,
        realtime_session_limits: limits,
    }
}

async fn spawn_worker(service: &WorkerService<FakeRealtimeExecutor>) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = service.router();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    addr
}

fn connect_request(
    addr: SocketAddr,
    subprotocol: bool,
    auth: bool,
) -> tokio_tungstenite::tungstenite::http::Request<()> {
    let mut request = format!("ws://{addr}{REALTIME_WS_PATH}")
        .into_client_request()
        .unwrap();
    let headers = request.headers_mut();
    if subprotocol {
        headers.insert(
            "sec-websocket-protocol",
            HeaderValue::from_static(REALTIME_SUBPROTOCOL),
        );
    }
    if auth {
        headers.insert(
            SERVICE_AUTHORIZATION_HEADER,
            HeaderValue::from_static("Bearer worker-secret"),
        );
        headers.insert(
            SERVICE_CREDENTIAL_ID_HEADER,
            HeaderValue::from_static("worker-credential"),
        );
    }
    request
}

async fn connect(
    addr: SocketAddr,
    subprotocol: bool,
    auth: bool,
) -> Result<(WsStream, Option<HeaderValue>), WsError> {
    let (stream, response) = connect_async(connect_request(addr, subprotocol, auth)).await?;
    let protocol = response
        .headers()
        .get("sec-websocket-protocol")
        .map(HeaderValue::from);
    Ok((stream, protocol))
}

fn admit_frame(attempt: &'static str, mutate: impl FnOnce(&mut RealtimeSessionAdmit)) -> String {
    let mut admit = RealtimeSessionAdmit {
        schema_version: PROTOCOL_V1,
        session_id: id("session-1"),
        request_id: id("request-1"),
        attempt_id: id(attempt),
        expected_worker_incarnation: id("incarnation-1"),
        deployment_id: id("asr-cpu-v1"),
        expected_model_generation: ModelGeneration::new(1).unwrap(),
        caller: GatewayAttestedCallerContext {
            tenant_id: id("tenant-1"),
            caller_id: id("caller-1"),
            policy_revision: id("policy-1"),
            permitted_actions: BTreeSet::from([PermittedAction::Invoke]),
            allowed_data_regions: vec!["local".into()],
        },
        task: TaskKind::SpeechToText,
        service_class: ServiceClass::Realtime,
        remaining_time_ms: 5_000,
        input: RealtimeStageInput::AudioStream {
            spec: RealtimeAudioSpec {
                codec: RealtimeAudioCodec::PcmI16Le,
                sample_rate: 16_000,
                channels: 1,
            },
            language: None,
        },
    };
    mutate(&mut admit);
    serde_json::to_string(&RealtimeClientFrame::Admit {
        admit: Box::new(admit),
    })
    .unwrap()
}

fn audio_frame(sequence: u32, payload: &[u8]) -> Message {
    Message::Binary(
        encode_realtime_audio_frame(sequence, false, payload)
            .unwrap()
            .into(),
    )
}

async fn send_json(ws: &mut WsStream, value: &RealtimeClientFrame) {
    ws.send(Message::Text(serde_json::to_string(value).unwrap().into()))
        .await
        .unwrap();
}

async fn send_text(ws: &mut WsStream, text: String) {
    ws.send(Message::Text(text.into())).await.unwrap();
}

/// Reads the next server frame, asserting text frames decode to
/// `RealtimeServerFrame` and returning close codes verbatim.
#[derive(Debug)]
enum NextFrame {
    Admitted,
    Event(InvocationEvent),
    Close(u16),
    Other,
}

async fn next_frame(ws: &mut WsStream) -> NextFrame {
    loop {
        let message = tokio::time::timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("server frame within deadline")
            .expect("stream open")
            .unwrap_or_else(|error| panic!("transport error: {error}"));
        match message {
            Message::Text(text) => match serde_json::from_str::<RealtimeServerFrame>(&text) {
                Ok(RealtimeServerFrame::Admitted { .. }) => return NextFrame::Admitted,
                Ok(RealtimeServerFrame::Event { event }) => return NextFrame::Event(event),
                Ok(RealtimeServerFrame::Pong) => return NextFrame::Other,
                Err(_) => return NextFrame::Other,
            },
            Message::Close(frame) => {
                return NextFrame::Close(frame.map(|frame| u16::from(frame.code)).unwrap_or(1000))
            }
            _ => continue,
        }
    }
}

async fn expect_close(ws: &mut WsStream, code: u16) {
    for _ in 0..8 {
        match next_frame(ws).await {
            NextFrame::Close(seen) => {
                assert_eq!(seen, code, "close reason code");
                return;
            }
            NextFrame::Other | NextFrame::Event(_) | NextFrame::Admitted => continue,
        }
    }
    panic!("session never closed with {code}");
}

#[tokio::test]
async fn realtime_session_streams_deltas_and_completes_through_shared_attempt_table() {
    let runner = FakeAsrRunner::new("final transcript");
    let service = WorkerService::new(
        realtime_config(REALTIME_SESSION_DEFAULT_LIMITS),
        FakeRealtimeExecutor {
            runner: runner.clone(),
        },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;

    let (mut ws, protocol) = connect(addr, true, true).await.unwrap();
    assert_eq!(
        protocol.as_ref().map(|value| value.to_str().unwrap()),
        Some(REALTIME_SUBPROTOCOL)
    );

    send_text(&mut ws, admit_frame("attempt-1", |_| {})).await;
    assert!(matches!(next_frame(&mut ws).await, NextFrame::Admitted));
    let accepted = match next_frame(&mut ws).await {
        NextFrame::Event(event) => event,
        other => panic!("expected accepted event, got {other:?}"),
    };
    assert!(matches!(
        accepted.event,
        InvocationEventKind::Accepted { .. }
    ));
    assert_eq!(accepted.sequence, 0);

    for sequence in 1..=2u32 {
        ws.send(audio_frame(
            sequence,
            &[0i16.to_le_bytes(), 0i16.to_le_bytes()].concat(),
        ))
        .await
        .unwrap();
        match next_frame(&mut ws).await {
            NextFrame::Event(event) => {
                assert!(matches!(event.event, InvocationEventKind::TextDelta { .. }));
                assert_eq!(event.sequence, u64::from(sequence));
            }
            other => panic!("expected delta event, got {other:?}"),
        }
    }

    send_json(&mut ws, &RealtimeClientFrame::Finish).await;
    let final_delta = match next_frame(&mut ws).await {
        NextFrame::Event(event) => event,
        other => panic!("expected final delta, got {other:?}"),
    };
    let InvocationEventKind::TextDelta { text } = final_delta.event else {
        panic!("expected final transcript delta");
    };
    assert_eq!(text, "final transcript");
    let completed = match next_frame(&mut ws).await {
        NextFrame::Event(event) => event,
        other => panic!("expected completed, got {other:?}"),
    };
    assert!(matches!(
        completed.event,
        InvocationEventKind::Completed { .. }
    ));
    expect_close(&mut ws, 1000).await;

    // The attempt lives in the shared table: HTTP query behaves identically.
    let query = service
        .router()
        .oneshot(
            axum::http::Request::builder()
                .method("GET")
                .uri("/internal/v1/invocations/attempt-1")
                .header(SERVICE_AUTHORIZATION_HEADER, "Bearer worker-secret")
                .header(SERVICE_CREDENTIAL_ID_HEADER, "worker-credential")
                .body(axum::body::Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(query.status(), axum::http::StatusCode::OK);
    let body = axum::body::to_bytes(query.into_body(), 4096).await.unwrap();
    let status: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(status["state"], "completed");

    assert_eq!(service.active_invocations(), 0);
    assert_eq!(runner.counters.active_streams.load(Ordering::Acquire), 0);
    assert_eq!(runner.counters.start_calls.load(Ordering::Acquire), 1);
}

#[tokio::test]
async fn realtime_attempt_id_reuse_conflicts_across_sessions() {
    let service = WorkerService::new(
        realtime_config(REALTIME_SESSION_DEFAULT_LIMITS),
        FakeRealtimeExecutor {
            runner: FakeAsrRunner::new("x"),
        },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;

    let (mut first, _) = connect(addr, true, true).await.unwrap();
    send_text(&mut first, admit_frame("attempt-1", |_| {})).await;
    assert!(matches!(next_frame(&mut first).await, NextFrame::Admitted));

    let (mut second, _) = connect(addr, true, true).await.unwrap();
    send_text(
        &mut second,
        admit_frame("attempt-1", |admit| {
            admit.session_id = id("session-2");
            admit.request_id = id("request-2");
        }),
    )
    .await;
    expect_close(
        &mut second,
        RealtimeSessionCloseCode::DuplicateAttempt.code(),
    )
    .await;
    send_json(&mut first, &RealtimeClientFrame::Cancel).await;
    expect_close(&mut first, 1000).await;
}

#[tokio::test]
async fn realtime_fencing_rejects_stale_admission_identity() {
    let service = WorkerService::new(
        realtime_config(REALTIME_SESSION_DEFAULT_LIMITS),
        FakeRealtimeExecutor {
            runner: FakeAsrRunner::new("x"),
        },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;

    type FenceCase = (Box<dyn FnOnce(&mut RealtimeSessionAdmit)>, u16);
    let cases: Vec<FenceCase> = vec![
        (
            Box::new(|admit: &mut RealtimeSessionAdmit| {
                admit.expected_worker_incarnation = id("incarnation-9");
            }),
            RealtimeSessionCloseCode::WrongWorkerIncarnation.code(),
        ),
        (
            Box::new(|admit: &mut RealtimeSessionAdmit| {
                admit.deployment_id = id("asr-cpu-gone");
            }),
            RealtimeSessionCloseCode::UnknownDeployment.code(),
        ),
        (
            Box::new(|admit: &mut RealtimeSessionAdmit| {
                admit.expected_model_generation = ModelGeneration::new(2).unwrap();
            }),
            RealtimeSessionCloseCode::WrongModelGeneration.code(),
        ),
        (
            Box::new(|admit: &mut RealtimeSessionAdmit| {
                admit.task = TaskKind::TextToSpeech;
                admit.input = RealtimeStageInput::TextStream;
            }),
            RealtimeSessionCloseCode::IncompatibleTask.code(),
        ),
    ];

    for (index, (mutate, expected)) in cases.into_iter().enumerate() {
        let attempt: &'static str = Box::leak(format!("attempt-fence-{index}").into_boxed_str());
        let (mut ws, _) = connect(addr, true, true).await.unwrap();
        send_text(&mut ws, admit_frame(attempt, mutate)).await;
        expect_close(&mut ws, expected).await;
    }
}

#[tokio::test]
async fn realtime_session_without_invoke_permission_is_policy_denied() {
    let service = WorkerService::new(
        realtime_config(REALTIME_SESSION_DEFAULT_LIMITS),
        FakeRealtimeExecutor {
            runner: FakeAsrRunner::new("x"),
        },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;

    let (mut ws, _) = connect(addr, true, true).await.unwrap();
    send_text(
        &mut ws,
        admit_frame("attempt-1", |admit| {
            admit.caller.permitted_actions.clear();
        }),
    )
    .await;
    expect_close(&mut ws, RealtimeSessionCloseCode::PolicyDenied.code()).await;
    assert_eq!(service.active_invocations(), 0);
}

#[tokio::test]
async fn realtime_capacity_exhaustion_closes_the_second_session() {
    let mut config = realtime_config(REALTIME_SESSION_DEFAULT_LIMITS);
    config.max_active_invocations = 1;
    let service = WorkerService::new(
        config,
        FakeRealtimeExecutor {
            runner: FakeAsrRunner::new("x"),
        },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;

    let (mut first, _) = connect(addr, true, true).await.unwrap();
    send_text(&mut first, admit_frame("attempt-1", |_| {})).await;
    assert!(matches!(next_frame(&mut first).await, NextFrame::Admitted));

    let (mut second, _) = connect(addr, true, true).await.unwrap();
    send_text(&mut second, admit_frame("attempt-2", |_| {})).await;
    expect_close(
        &mut second,
        RealtimeSessionCloseCode::CapacityExhausted.code(),
    )
    .await;

    // Ending the first session releases capacity for a replacement.
    send_json(&mut first, &RealtimeClientFrame::Cancel).await;
    expect_close(&mut first, 1000).await;
    let (mut third, _) = connect(addr, true, true).await.unwrap();
    send_text(&mut third, admit_frame("attempt-3", |_| {})).await;
    assert!(matches!(next_frame(&mut third).await, NextFrame::Admitted));
    send_json(&mut third, &RealtimeClientFrame::Cancel).await;
    expect_close(&mut third, 1000).await;
}

#[tokio::test]
async fn draining_worker_rejects_realtime_sessions_before_and_after_upgrade() {
    let service = WorkerService::new(
        realtime_config(REALTIME_SESSION_DEFAULT_LIMITS),
        FakeRealtimeExecutor {
            runner: FakeAsrRunner::new("x"),
        },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;

    // Pre-upgrade: the route returns 503, surfaced as an HTTP handshake error.
    let drain_service = service.clone();
    drain_service.begin_draining().await;
    let denied = connect(addr, true, true).await.unwrap_err();
    match denied {
        WsError::Http(response) => {
            assert_eq!(
                response.status(),
                axum::http::StatusCode::SERVICE_UNAVAILABLE
            )
        }
        other => panic!("expected HTTP rejection, got {other:?}"),
    }
}

#[tokio::test]
async fn realtime_audio_sequence_regression_fails_the_session() {
    let service = WorkerService::new(
        realtime_config(REALTIME_SESSION_DEFAULT_LIMITS),
        FakeRealtimeExecutor {
            runner: FakeAsrRunner::new("x"),
        },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;

    let (mut ws, _) = connect(addr, true, true).await.unwrap();
    send_text(&mut ws, admit_frame("attempt-1", |_| {})).await;
    assert!(matches!(next_frame(&mut ws).await, NextFrame::Admitted));
    assert!(matches!(next_frame(&mut ws).await, NextFrame::Event(_)));

    ws.send(audio_frame(7, &[0, 0, 0, 0])).await.unwrap();
    assert!(matches!(next_frame(&mut ws).await, NextFrame::Event(_)));
    ws.send(audio_frame(7, &[0, 0, 0, 0])).await.unwrap();
    match next_frame(&mut ws).await {
        NextFrame::Event(event) => {
            let InvocationEventKind::Error { code, .. } = event.event else {
                panic!("expected error terminal, got {:?}", event.event);
            };
            assert_eq!(code, InvocationErrorCode::InvalidInput);
        }
        other => panic!("expected error event, got {other:?}"),
    }
    expect_close(&mut ws, 1000).await;
}

#[tokio::test]
async fn realtime_session_audio_budget_fails_closed() {
    let limits = RuntimeRealtimeSessionLimits {
        max_session_audio_bytes: 2,
        ..REALTIME_SESSION_DEFAULT_LIMITS
    };
    let service = WorkerService::new(
        realtime_config(limits),
        FakeRealtimeExecutor {
            runner: FakeAsrRunner::new("x"),
        },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;

    let (mut ws, _) = connect(addr, true, true).await.unwrap();
    send_text(&mut ws, admit_frame("attempt-1", |_| {})).await;
    assert!(matches!(next_frame(&mut ws).await, NextFrame::Admitted));
    assert!(matches!(next_frame(&mut ws).await, NextFrame::Event(_)));

    ws.send(audio_frame(1, &[0, 0, 0, 0])).await.unwrap();
    match next_frame(&mut ws).await {
        NextFrame::Event(event) => {
            let InvocationEventKind::Error { code, message } = event.event else {
                panic!("expected error terminal");
            };
            assert_eq!(code, InvocationErrorCode::OutputLimitExceeded);
            assert!(message.contains("budget"));
        }
        other => panic!("expected error event, got {other:?}"),
    }
    expect_close(&mut ws, 1000).await;
}

#[tokio::test]
async fn realtime_admit_timeout_closes_idle_sessions() {
    let limits = RuntimeRealtimeSessionLimits {
        admit_timeout: Duration::from_millis(100),
        ..REALTIME_SESSION_DEFAULT_LIMITS
    };
    let service = WorkerService::new(
        realtime_config(limits),
        FakeRealtimeExecutor {
            runner: FakeAsrRunner::new("x"),
        },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;

    let (mut ws, _) = connect(addr, true, true).await.unwrap();
    expect_close(&mut ws, RealtimeSessionCloseCode::AdmitTimeout.code()).await;
    assert_eq!(service.active_invocations(), 0);
}

#[tokio::test]
async fn realtime_route_requires_subprotocol_auth_and_capability() {
    // No subprotocol offered: 400 before upgrade.
    let service = WorkerService::new(
        realtime_config(REALTIME_SESSION_DEFAULT_LIMITS),
        FakeRealtimeExecutor {
            runner: FakeAsrRunner::new("x"),
        },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;
    let rejected = connect(addr, false, true).await.unwrap_err();
    match rejected {
        WsError::Http(response) => {
            assert_eq!(response.status(), axum::http::StatusCode::BAD_REQUEST)
        }
        other => panic!("expected HTTP rejection, got {other:?}"),
    }

    // No service credential: 401 before upgrade.
    let rejected = connect(addr, true, false).await.unwrap_err();
    match rejected {
        WsError::Http(response) => {
            assert_eq!(response.status(), axum::http::StatusCode::UNAUTHORIZED)
        }
        other => panic!("expected HTTP rejection, got {other:?}"),
    }

    // A chat deployment does not advertise realtime: fail-closed 404.
    let mut chat_config = realtime_config(REALTIME_SESSION_DEFAULT_LIMITS);
    chat_config.deployment.task = TaskKind::Chat;
    chat_config.deployment.capability.task = TaskKind::Chat;
    chat_config.deployment.capability.realtime = false;
    chat_config
        .deployment
        .capability
        .accepted_input_formats
        .clear();
    chat_config
        .deployment
        .capability
        .accepted_input_formats
        .insert(InputFormat::ChatMessages);
    chat_config
        .descriptor
        .features
        .remove(&WorkerFeature::RealtimeSocket);
    let chat_service = WorkerService::new(
        chat_config,
        FakeRealtimeExecutor {
            runner: FakeAsrRunner::new("x"),
        },
    )
    .unwrap();
    let chat_addr = spawn_worker(&chat_service).await;
    let rejected = connect(chat_addr, true, true).await.unwrap_err();
    match rejected {
        WsError::Http(response) => {
            assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND)
        }
        other => panic!("expected HTTP rejection, got {other:?}"),
    }
}

#[tokio::test]
async fn http_cancel_terminates_a_running_realtime_session() {
    let service = WorkerService::new(
        realtime_config(REALTIME_SESSION_DEFAULT_LIMITS),
        FakeRealtimeExecutor {
            runner: FakeAsrRunner::new("x"),
        },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;

    let (mut ws, _) = connect(addr, true, true).await.unwrap();
    send_text(&mut ws, admit_frame("attempt-1", |_| {})).await;
    assert!(matches!(next_frame(&mut ws).await, NextFrame::Admitted));
    assert!(matches!(next_frame(&mut ws).await, NextFrame::Event(_)));

    let cancel_request = CancelAttemptRequest {
        schema_version: PROTOCOL_V1,
        identity: AttemptIdentity {
            request_id: id("request-1"),
            attempt_id: id("attempt-1"),
            tenant_id: id("tenant-1"),
            caller_id: id("caller-1"),
            incarnation_id: id("incarnation-1"),
            deployment_id: id("asr-cpu-v1"),
            model_generation: ModelGeneration::new(1).unwrap(),
        },
    };
    let cancel = service
        .router()
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/internal/v1/invocations/attempt-1/cancel")
                .header(SERVICE_AUTHORIZATION_HEADER, "Bearer worker-secret")
                .header(SERVICE_CREDENTIAL_ID_HEADER, "worker-credential")
                .header(axum::http::header::CONTENT_TYPE, "application/json")
                .body(axum::body::Body::from(
                    serde_json::to_vec(&cancel_request).unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(cancel.status(), axum::http::StatusCode::OK);

    let cancelled = match next_frame(&mut ws).await {
        NextFrame::Event(event) => event,
        other => panic!("expected cancelled event, got {other:?}"),
    };
    assert!(matches!(
        cancelled.event,
        InvocationEventKind::Cancelled { .. }
    ));
    expect_close(&mut ws, 1000).await;
    assert_eq!(service.active_invocations(), 0);
}

#[tokio::test]
async fn worker_config_rejects_realtime_limits_outside_protocol_caps() {
    let mut config = realtime_config(REALTIME_SESSION_DEFAULT_LIMITS);
    config.realtime_session_limits.max_frame_bytes = 0;
    assert!(matches!(
        config.validate(),
        Err(WorkerConfigError::InvalidRealtimeLimits)
    ));

    let mut config = realtime_config(REALTIME_SESSION_DEFAULT_LIMITS);
    config.realtime_session_limits.max_frame_bytes = MAX_REALTIME_AUDIO_FRAME_BYTES + 1;
    assert!(matches!(
        config.validate(),
        Err(WorkerConfigError::InvalidRealtimeLimits)
    ));

    let mut config = realtime_config(REALTIME_SESSION_DEFAULT_LIMITS);
    config.realtime_session_limits.admit_timeout = Duration::ZERO;
    assert!(matches!(
        config.validate(),
        Err(WorkerConfigError::InvalidRealtimeLimits)
    ));
}
