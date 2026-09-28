//! Realtime WebSocket (`izwi-realtime-v1`) session contract tests against the
//! worker's `/internal/v1/realtime` route. Sessions run over real loopback
//! sockets against a scripted ASR stage runner, so admission fencing, bounds,
//! terminal outcomes, and the shared attempt table are exercised without a
//! deployed model.

use async_trait::async_trait;
use axum::http::HeaderValue;
use futures::{SinkExt, StreamExt};
use izwi_core::{AudioChunk, RuntimeAsrRealtimeEvent};
use izwi_serving_protocol::*;
use izwi_serving_worker::{AdmissionFailure, AdmittedInvocation, WorkerConfigError};
use izwi_serving_worker::{
    InvocationExecutor, RealtimeAsrStageStream, RealtimeStageRunner, RealtimeTtsStageStream,
    RuntimeRealtimeSessionLimits, WorkerConfig, WorkerService, REALTIME_SESSION_DEFAULT_LIMITS,
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
            kv_host_pages: None,
            kv_demotions_total: None,
            kv_promotions_total: None,
            kv_promotion_latency_avg_seconds: None,
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

async fn spawn_worker<E: InvocationExecutor>(service: &WorkerService<E>) -> SocketAddr {
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
    Binary {
        sequence: u32,
        is_final: bool,
        payload: Vec<u8>,
    },
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
            Message::Binary(bytes) => match decode_realtime_audio_frame(&bytes) {
                Ok((header, payload)) => {
                    return NextFrame::Binary {
                        sequence: header.sequence,
                        is_final: header.is_final,
                        payload: payload.to_vec(),
                    }
                }
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
            NextFrame::Other
            | NextFrame::Event(_)
            | NextFrame::Admitted
            | NextFrame::Binary { .. } => continue,
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
    let InvocationEventKind::TextDelta { text, .. } = final_delta.event else {
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

// ---------------------------------------------------------------------------
// TTS-stream stage (protocol minor 2): text in, synthesized audio out.
// ---------------------------------------------------------------------------

struct TtsChunk {
    samples: Vec<f32>,
    is_final: bool,
}

fn tts_chunk(samples: &[f32]) -> TtsChunk {
    TtsChunk {
        samples: samples.to_vec(),
        is_final: false,
    }
}

fn tts_final_chunk() -> TtsChunk {
    TtsChunk {
        samples: Vec::new(),
        is_final: true,
    }
}

#[derive(Default)]
struct TtsRunnerCounters {
    start_calls: AtomicUsize,
    active_streams: AtomicUsize,
}

/// Scripted TTS stage runner: chunks are yielded on finish-commit, with an
/// optional injected start or mid-synthesis failure. Stream liveness is
/// observable so tests can assert teardown actually dropped the stream.
#[derive(Clone)]
struct FakeTtsRunner {
    counters: Arc<TtsRunnerCounters>,
    chunks: Arc<Mutex<VecDeque<TtsChunk>>>,
    fail_start: Arc<AsyncMutex<bool>>,
    error_after_chunks: Arc<AtomicUsize>,
    chunk_delay: Arc<AsyncMutex<Duration>>,
}

impl FakeTtsRunner {
    /// Default script: two audio chunks followed by the runtime's terminal
    /// empty final chunk.
    fn new() -> Self {
        Self {
            counters: Arc::new(TtsRunnerCounters::default()),
            chunks: Arc::new(Mutex::new(VecDeque::from([
                tts_chunk(&[0.5, -0.25, 0.125]),
                tts_chunk(&[-0.75, 0.5]),
                tts_final_chunk(),
            ]))),
            fail_start: Arc::new(AsyncMutex::new(false)),
            error_after_chunks: Arc::new(AtomicUsize::new(usize::MAX)),
            chunk_delay: Arc::new(AsyncMutex::new(Duration::ZERO)),
        }
    }

    fn with_chunks(chunks: Vec<TtsChunk>) -> Self {
        Self {
            chunks: Arc::new(Mutex::new(chunks.into())),
            ..Self::new()
        }
    }

    async fn fail_start(&self) {
        *self.fail_start.lock().await = true;
    }

    fn fail_after(&self, chunks: usize) {
        self.error_after_chunks.store(chunks, Ordering::Release);
    }

    async fn set_chunk_delay(&self, delay: Duration) {
        *self.chunk_delay.lock().await = delay;
    }
}

struct FakeTtsStream {
    counters: Arc<TtsRunnerCounters>,
    chunks: Arc<Mutex<VecDeque<TtsChunk>>>,
    error_after_chunks: Arc<AtomicUsize>,
    chunk_delay: Arc<AsyncMutex<Duration>>,
    emitted: usize,
}

impl Drop for FakeTtsStream {
    fn drop(&mut self) {
        self.counters.active_streams.fetch_sub(1, Ordering::AcqRel);
    }
}

#[async_trait::async_trait]
impl RealtimeTtsStageStream for FakeTtsStream {
    async fn next_chunk(&mut self) -> Result<Option<AudioChunk>, izwi_core::Error> {
        let delay = *self.chunk_delay.lock().await;
        if !delay.is_zero() {
            tokio::time::sleep(delay).await;
        }
        let next = self.chunks.lock().unwrap().pop_front();
        match next {
            Some(chunk) => {
                self.emitted += 1;
                if self.emitted > self.error_after_chunks.load(Ordering::Acquire) {
                    return Err(izwi_core::Error::InferenceError(
                        "injected synthesis failure".into(),
                    ));
                }
                Ok(Some(AudioChunk {
                    request_id: "fake-tts".into(),
                    sequence: self.emitted,
                    samples: chunk.samples,
                    sample_rate: 24_000,
                    is_final: chunk.is_final,
                    stats: None,
                }))
            }
            None => Ok(None),
        }
    }
}

#[async_trait::async_trait]
impl RealtimeStageRunner for FakeTtsRunner {
    fn stage_task(&self) -> TaskKind {
        TaskKind::TextToSpeech
    }

    async fn start_asr_stream(
        &self,
        _language: Option<&str>,
    ) -> Result<Box<dyn RealtimeAsrStageStream>, izwi_core::Error> {
        Err(izwi_core::Error::ConfigError(
            "fake TTS runner does not serve ASR".into(),
        ))
    }

    fn output_audio_spec(&self) -> Option<RealtimeAudioSpec> {
        Some(RealtimeAudioSpec {
            codec: RealtimeAudioCodec::PcmI16Le,
            sample_rate: 24_000,
            channels: 1,
        })
    }

    async fn start_tts_stream(
        &self,
        _text: String,
        _deadline: std::time::Instant,
    ) -> Result<Box<dyn RealtimeTtsStageStream>, izwi_core::Error> {
        self.counters.start_calls.fetch_add(1, Ordering::AcqRel);
        if *self.fail_start.lock().await {
            return Err(izwi_core::Error::InferenceError(
                "injected start failure".into(),
            ));
        }
        self.counters.active_streams.fetch_add(1, Ordering::AcqRel);
        Ok(Box::new(FakeTtsStream {
            counters: Arc::clone(&self.counters),
            chunks: Arc::clone(&self.chunks),
            error_after_chunks: Arc::clone(&self.error_after_chunks),
            chunk_delay: Arc::clone(&self.chunk_delay),
            emitted: 0,
        }))
    }
}

struct FakeTtsRealtimeExecutor {
    runner: FakeTtsRunner,
}

#[async_trait]
impl InvocationExecutor for FakeTtsRealtimeExecutor {
    async fn admit(
        &self,
        _request: &InvocationRequest,
    ) -> Result<AdmittedInvocation, AdmissionFailure> {
        Err(AdmissionFailure::new(
            RejectionCode::IncompatibleTask,
            "fake TTS executor serves realtime WebSocket sessions only",
        ))
    }

    fn realtime_runner(&self) -> Option<Arc<dyn RealtimeStageRunner>> {
        Some(Arc::new(self.runner.clone()))
    }
}

fn tts_config(limits: RuntimeRealtimeSessionLimits) -> WorkerConfig {
    let mut config = realtime_config(limits);
    config.deployment.task = TaskKind::TextToSpeech;
    config.deployment.capability.task = TaskKind::TextToSpeech;
    config.deployment.capability.accepted_input_formats.clear();
    config
        .deployment
        .capability
        .accepted_input_formats
        .insert(InputFormat::Text);
    config.deployment.capability.output_formats.clear();
    config
        .deployment
        .capability
        .output_formats
        .insert(OutputFormat::PcmAudio);
    config.deployment.execution_representation = "tiny-tts-realtime".into();
    config
}

fn tts_admit_frame(
    attempt: &'static str,
    mutate: impl FnOnce(&mut RealtimeSessionAdmit),
) -> String {
    admit_frame(attempt, |admit| {
        admit.deployment_id = id("asr-cpu-v1");
        admit.task = TaskKind::TextToSpeech;
        admit.input = RealtimeStageInput::TextStream;
        mutate(admit);
    })
}

fn pcm_i16_bytes(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|sample| {
            let quantized = (sample.clamp(-1.0, 1.0) * 32_767.0) as i16;
            quantized.to_le_bytes()
        })
        .collect()
}

async fn admit_tts_session(ws: &mut WsStream, attempt: &'static str) {
    send_text(ws, tts_admit_frame(attempt, |_| {})).await;
    let admitted = next_frame(ws).await;
    let NextFrame::Admitted = admitted else {
        panic!("expected admitted frame, got {admitted:?}");
    };
    let accepted = match next_frame(ws).await {
        NextFrame::Event(event) => event,
        other => panic!("expected accepted event, got {other:?}"),
    };
    assert!(matches!(
        accepted.event,
        InvocationEventKind::Accepted { .. }
    ));
}

/// Drives the happy path: two text frames, finish, then reads every frame the
/// worker emits until the session closes. Returns the binary frames in order.
async fn drive_tts_session_to_completion(
    ws: &mut WsStream,
    attempt: &'static str,
) -> Vec<NextFrame> {
    admit_tts_session(ws, attempt).await;
    send_json(
        ws,
        &RealtimeClientFrame::Input {
            text: "Hello ".into(),
        },
    )
    .await;
    send_json(
        ws,
        &RealtimeClientFrame::Input {
            text: "world".into(),
        },
    )
    .await;
    send_json(ws, &RealtimeClientFrame::Finish).await;

    let mut frames = Vec::new();
    for _ in 0..16 {
        match next_frame(ws).await {
            NextFrame::Close(code) => {
                assert_eq!(code, 1000, "orderly session close");
                frames.push(NextFrame::Close(code));
                return frames;
            }
            frame @ NextFrame::Binary { .. } | frame @ NextFrame::Event(_) => {
                frames.push(frame);
            }
            other => panic!("unexpected frame: {other:?}"),
        }
    }
    panic!("session never reached a terminal outcome");
}

#[tokio::test]
async fn tts_session_streams_audio_frames_and_completes() {
    let runner = FakeTtsRunner::new();
    let service = WorkerService::new(
        tts_config(REALTIME_SESSION_DEFAULT_LIMITS),
        FakeTtsRealtimeExecutor {
            runner: runner.clone(),
        },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;

    let (mut ws, _) = connect(addr, true, true).await.unwrap();
    let frames = drive_tts_session_to_completion(&mut ws, "attempt-1").await;

    // Payload frames carry the i16 quantization of the scripted samples; the
    // terminal empty chunk becomes the zero-payload final-flagged frame.
    let mut binary = Vec::new();
    for frame in &frames {
        if let NextFrame::Binary {
            sequence,
            is_final,
            payload,
        } = frame
        {
            binary.push(((*sequence, *is_final), payload.clone()));
        }
    }
    assert_eq!(binary.len(), 3, "two payload frames plus the final frame");
    assert_eq!(binary[0].0, (1, false));
    assert_eq!(binary[0].1, pcm_i16_bytes(&[0.5, -0.25, 0.125]));
    assert_eq!(binary[1].0, (2, false));
    assert_eq!(binary[1].1, pcm_i16_bytes(&[-0.75, 0.5]));
    assert_eq!(binary[2].0, (3, true));
    assert!(binary[2].1.is_empty(), "terminal frame is zero-payload");

    // The event stream carries exactly the terminal event here: the Accepted
    // event was consumed by the admit helper above.
    let events: Vec<&InvocationEvent> = frames
        .iter()
        .filter_map(|frame| match frame {
            NextFrame::Event(event) => Some(event),
            _ => None,
        })
        .collect();
    assert_eq!(events.len(), 1, "only the terminal event remains");
    let InvocationEventKind::Completed { finish_reason, .. } = events[0].event else {
        panic!("expected completed terminal");
    };
    assert_eq!(finish_reason, FinishReason::Stop);

    assert_eq!(runner.counters.start_calls.load(Ordering::Acquire), 1);
    assert_eq!(runner.counters.active_streams.load(Ordering::Acquire), 0);
    assert_eq!(service.active_invocations(), 0);

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
}

#[tokio::test]
async fn tts_admitted_frame_announces_the_output_audio_spec() {
    let service = WorkerService::new(
        tts_config(REALTIME_SESSION_DEFAULT_LIMITS),
        FakeTtsRealtimeExecutor {
            runner: FakeTtsRunner::new(),
        },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;

    let (mut ws, _) = connect(addr, true, true).await.unwrap();
    send_text(&mut ws, tts_admit_frame("attempt-1", |_| {})).await;
    // Read the raw Admitted JSON to inspect output_audio directly.
    let message = tokio::time::timeout(Duration::from_secs(5), ws.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    let Message::Text(text) = message else {
        panic!("expected text frame");
    };
    let frame: RealtimeServerFrame = serde_json::from_str(&text).unwrap();
    let RealtimeServerFrame::Admitted { output_audio, .. } = frame else {
        panic!("expected admitted frame");
    };
    let spec = output_audio.expect("TTS sessions announce their output spec");
    assert_eq!(spec.codec, RealtimeAudioCodec::PcmI16Le);
    assert_eq!(spec.sample_rate, 24_000);
    assert_eq!(spec.channels, 1);
}

#[tokio::test]
async fn tts_session_empty_finish_completes_without_audio() {
    let runner = FakeTtsRunner::new();
    let service = WorkerService::new(
        tts_config(REALTIME_SESSION_DEFAULT_LIMITS),
        FakeTtsRealtimeExecutor { runner },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;

    let (mut ws, _) = connect(addr, true, true).await.unwrap();
    admit_tts_session(&mut ws, "attempt-1").await;
    send_json(&mut ws, &RealtimeClientFrame::Finish).await;
    let completed = match next_frame(&mut ws).await {
        NextFrame::Event(event) => event,
        other => panic!("expected completed, got {other:?}"),
    };
    assert!(matches!(
        completed.event,
        InvocationEventKind::Completed { .. }
    ));
    expect_close(&mut ws, 1000).await;
    assert_eq!(service.active_invocations(), 0);
}

#[tokio::test]
async fn tts_session_rejects_input_after_finish() {
    let service = WorkerService::new(
        tts_config(REALTIME_SESSION_DEFAULT_LIMITS),
        FakeTtsRealtimeExecutor {
            runner: FakeTtsRunner::new(),
        },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;

    let (mut ws, _) = connect(addr, true, true).await.unwrap();
    admit_tts_session(&mut ws, "attempt-1").await;
    send_json(&mut ws, &RealtimeClientFrame::Input { text: "hi".into() }).await;
    send_json(&mut ws, &RealtimeClientFrame::Finish).await;
    send_json(
        &mut ws,
        &RealtimeClientFrame::Input {
            text: "late".into(),
        },
    )
    .await;
    // Synthesis may already be streaming when the late input lands, so audio
    // frames may precede the failure.
    loop {
        match next_frame(&mut ws).await {
            NextFrame::Binary { .. } => continue,
            NextFrame::Event(event) => {
                let InvocationEventKind::Error { code, message } = event.event else {
                    panic!("expected error terminal");
                };
                assert_eq!(code, InvocationErrorCode::InvalidInput);
                assert!(message.contains("after finish"));
                break;
            }
            other => panic!("expected error event, got {other:?}"),
        }
    }
    expect_close(&mut ws, 1000).await;
}

#[tokio::test]
async fn tts_session_enforces_the_text_budget() {
    let limits = RuntimeRealtimeSessionLimits {
        max_session_audio_bytes: 4,
        ..REALTIME_SESSION_DEFAULT_LIMITS
    };
    let service = WorkerService::new(
        tts_config(limits),
        FakeTtsRealtimeExecutor {
            runner: FakeTtsRunner::new(),
        },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;

    let (mut ws, _) = connect(addr, true, true).await.unwrap();
    admit_tts_session(&mut ws, "attempt-1").await;
    send_json(
        &mut ws,
        &RealtimeClientFrame::Input {
            text: "12345".into(),
        },
    )
    .await;
    match next_frame(&mut ws).await {
        NextFrame::Event(event) => {
            let InvocationEventKind::Error { code, message } = event.event else {
                panic!("expected error terminal");
            };
            assert_eq!(code, InvocationErrorCode::InvalidInput);
            assert!(message.contains("text budget"));
        }
        other => panic!("expected error event, got {other:?}"),
    }
    expect_close(&mut ws, 1000).await;
}

#[tokio::test]
async fn tts_session_cancel_mid_synthesis_confirms_teardown() {
    let runner = FakeTtsRunner::with_chunks(vec![tts_chunk(&[0.5]), tts_final_chunk()]);
    runner.set_chunk_delay(Duration::from_millis(200)).await;
    let service = WorkerService::new(
        tts_config(REALTIME_SESSION_DEFAULT_LIMITS),
        FakeTtsRealtimeExecutor {
            runner: runner.clone(),
        },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;

    let (mut ws, _) = connect(addr, true, true).await.unwrap();
    admit_tts_session(&mut ws, "attempt-1").await;
    send_json(&mut ws, &RealtimeClientFrame::Input { text: "hi".into() }).await;
    send_json(&mut ws, &RealtimeClientFrame::Finish).await;
    send_json(&mut ws, &RealtimeClientFrame::Cancel).await;

    let cancelled = loop {
        match next_frame(&mut ws).await {
            NextFrame::Event(event) => match event.event {
                InvocationEventKind::Cancelled { .. } => break event,
                _ => continue,
            },
            other => panic!("expected cancelled event, got {other:?}"),
        }
    };
    assert_eq!(cancelled.sequence, 1);
    expect_close(&mut ws, 1000).await;
    assert_eq!(service.active_invocations(), 0);
    // Dropping the synthesis stream is the teardown confirmation.
    assert_eq!(runner.counters.active_streams.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn tts_session_http_cancel_reaches_the_session() {
    let runner = FakeTtsRunner::with_chunks(vec![tts_chunk(&[0.5]), tts_final_chunk()]);
    runner.set_chunk_delay(Duration::from_millis(200)).await;
    let service = WorkerService::new(
        tts_config(REALTIME_SESSION_DEFAULT_LIMITS),
        FakeTtsRealtimeExecutor {
            runner: runner.clone(),
        },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;

    let (mut ws, _) = connect(addr, true, true).await.unwrap();
    admit_tts_session(&mut ws, "attempt-1").await;
    send_json(&mut ws, &RealtimeClientFrame::Input { text: "hi".into() }).await;
    send_json(&mut ws, &RealtimeClientFrame::Finish).await;

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

    loop {
        match next_frame(&mut ws).await {
            NextFrame::Event(event) => match event.event {
                InvocationEventKind::Cancelled { .. } => break,
                _ => continue,
            },
            other => panic!("expected cancelled event, got {other:?}"),
        }
    }
    expect_close(&mut ws, 1000).await;
    assert_eq!(service.active_invocations(), 0);
    assert_eq!(runner.counters.active_streams.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn tts_session_splits_oversized_chunks_across_frames() {
    // A 3-sample chunk is 6 bytes; a 4-byte frame cap splits it in two, and
    // the final flag rides only the last frame of the terminal emission.
    let limits = RuntimeRealtimeSessionLimits {
        max_frame_bytes: 4,
        ..REALTIME_SESSION_DEFAULT_LIMITS
    };
    let runner = FakeTtsRunner::with_chunks(vec![tts_chunk(&[0.5, -0.25, 0.125])]);
    let service =
        WorkerService::new(tts_config(limits), FakeTtsRealtimeExecutor { runner }).unwrap();
    let addr = spawn_worker(&service).await;

    let (mut ws, _) = connect(addr, true, true).await.unwrap();
    admit_tts_session(&mut ws, "attempt-1").await;
    send_json(&mut ws, &RealtimeClientFrame::Input { text: "hi".into() }).await;
    send_json(&mut ws, &RealtimeClientFrame::Finish).await;

    let mut binary = Vec::new();
    loop {
        match next_frame(&mut ws).await {
            NextFrame::Binary {
                sequence,
                is_final,
                payload,
            } => binary.push(((sequence, is_final), payload)),
            NextFrame::Event(event) => match event.event {
                InvocationEventKind::Completed { .. } => break,
                _ => continue,
            },
            other => panic!("unexpected frame: {other:?}"),
        }
    }
    expect_close(&mut ws, 1000).await;
    let expected = pcm_i16_bytes(&[0.5, -0.25, 0.125]);
    assert_eq!(binary.len(), 3, "6-byte chunk splits into 4+2, then final");
    assert_eq!(binary[0].0, (1, false));
    assert_eq!(binary[0].1, expected[0..4]);
    assert_eq!(binary[1].0, (2, false));
    assert_eq!(binary[1].1, expected[4..6]);
    assert_eq!(binary[2].0, (3, true));
    assert!(binary[2].1.is_empty());
}

#[tokio::test]
async fn tts_session_fails_when_synthesis_fails() {
    let runner = FakeTtsRunner::new();
    runner.fail_after(1);
    let service = WorkerService::new(
        tts_config(REALTIME_SESSION_DEFAULT_LIMITS),
        FakeTtsRealtimeExecutor {
            runner: runner.clone(),
        },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;

    let (mut ws, _) = connect(addr, true, true).await.unwrap();
    admit_tts_session(&mut ws, "attempt-1").await;
    send_json(&mut ws, &RealtimeClientFrame::Input { text: "hi".into() }).await;
    send_json(&mut ws, &RealtimeClientFrame::Finish).await;

    // One frame may arrive before the injected failure surfaces.
    let mut saw_frame = false;
    loop {
        match next_frame(&mut ws).await {
            NextFrame::Binary { .. } => saw_frame = true,
            NextFrame::Event(event) => match event.event {
                InvocationEventKind::Error { code, .. } => {
                    assert_eq!(code, InvocationErrorCode::ExecutionFailed);
                    break;
                }
                InvocationEventKind::Completed { .. } => {
                    panic!("failed synthesis must not complete")
                }
                _ => continue,
            },
            other => panic!("unexpected frame: {other:?}"),
        }
    }
    let _ = saw_frame;
    expect_close(&mut ws, 1000).await;
    assert_eq!(service.active_invocations(), 0);
    assert_eq!(runner.counters.active_streams.load(Ordering::Acquire), 0);
}

#[tokio::test]
async fn tts_start_failure_fails_the_session() {
    let runner = FakeTtsRunner::new();
    runner.fail_start().await;
    let service = WorkerService::new(
        tts_config(REALTIME_SESSION_DEFAULT_LIMITS),
        FakeTtsRealtimeExecutor { runner },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;

    let (mut ws, _) = connect(addr, true, true).await.unwrap();
    admit_tts_session(&mut ws, "attempt-1").await;
    send_json(&mut ws, &RealtimeClientFrame::Input { text: "hi".into() }).await;
    send_json(&mut ws, &RealtimeClientFrame::Finish).await;
    match next_frame(&mut ws).await {
        NextFrame::Event(event) => {
            let InvocationEventKind::Error { code, .. } = event.event else {
                panic!("expected error terminal");
            };
            assert_eq!(code, InvocationErrorCode::ExecutionFailed);
        }
        other => panic!("expected error event, got {other:?}"),
    }
    expect_close(&mut ws, 1000).await;
    assert_eq!(service.active_invocations(), 0);
}

#[tokio::test]
async fn tts_admit_on_an_asr_worker_is_incompatible() {
    let service = WorkerService::new(
        realtime_config(REALTIME_SESSION_DEFAULT_LIMITS),
        FakeRealtimeExecutor {
            runner: FakeAsrRunner::new("x"),
        },
    )
    .unwrap();
    let addr = spawn_worker(&service).await;

    let (mut ws, _) = connect(addr, true, true).await.unwrap();
    send_text(&mut ws, tts_admit_frame("attempt-1", |_| {})).await;
    expect_close(&mut ws, RealtimeSessionCloseCode::IncompatibleTask.code()).await;
}

#[tokio::test]
async fn asr_session_rejects_text_input_frames() {
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
    send_json(&mut ws, &RealtimeClientFrame::Input { text: "hi".into() }).await;
    match next_frame(&mut ws).await {
        NextFrame::Event(event) => {
            let InvocationEventKind::Error { code, message } = event.event else {
                panic!("expected error terminal");
            };
            assert_eq!(code, InvocationErrorCode::InvalidInput);
            assert!(message.contains("speech_to_text"));
        }
        other => panic!("expected error event, got {other:?}"),
    }
    expect_close(&mut ws, 1000).await;
}
