//! A real worker for tests: [`WorkerService`] and its router on a loopback
//! socket, driven by a scripted runtime instead of a model.
//!
//! Everything a client or gateway sees (authentication, admission and
//! fencing, attempt bookkeeping, NDJSON framing, slow-consumer handling, the
//! realtime socket) is the production worker's code. Only the runtime behind
//! [`InvocationExecutor`] and [`RealtimeStageRunner`] is scripted, so a test
//! can make it slow, silent, or lose an invocation, but cannot make the
//! worker break the wire contract.

use crate::{
    AdmissionFailure, AdmittedExecution, AdmittedInvocation, ExecutionEvent, ExecutionFailure,
    ExecutionTeardown, InvocationExecutor, RealtimeAsrStageStream, RealtimeStageRunner,
    RealtimeTtsStageStream, RuntimeRealtimeSessionLimits, WorkerConfig, WorkerService,
    DEFAULT_ATTEMPT_RETENTION, DEFAULT_EVENT_CHANNEL_CAPACITY, DEFAULT_MAX_EVENT_BYTES,
    DEFAULT_SLOW_CONSUMER_TIMEOUT, REALTIME_SESSION_DEFAULT_LIMITS,
};
use async_trait::async_trait;
use izwi_core::{AudioChunk, RuntimeAsrRealtimeEvent};
use izwi_serving_protocol::*;
use std::{
    collections::BTreeSet,
    net::SocketAddr,
    pin::Pin,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    task::{Context, Poll, Waker},
    time::Duration,
};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::sync::watch;

/// What the scripted runtime does with an admitted chat invocation. Every
/// script waits `output_cadence` before its first output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerScript {
    /// One delta of `output_text`, then completion after another cadence.
    Complete,
    /// No output until the invocation is cancelled.
    Hang,
    /// One delta of `output_text`, then no further output.
    TextDeltaThenHang,
    /// `count` deltas of `text_bytes` bytes each, then completion.
    ManyTextDeltas { count: usize, text_bytes: usize },
    /// Each text as its own delta, then completion.
    TextDeltas(Vec<String>),
    /// One delta of `output_text`, then the model fails with `code`.
    TextDeltaThenFail(InvocationErrorCode),
    /// An empty text delta every `output_cadence`: progress without output.
    EmptyDeltaTrickle,
    /// No output, and the runtime cannot confirm teardown: the worker ends
    /// the stream without a terminal event and keeps the capacity.
    LoseAfterAccept,
    /// One delta of `output_text`, then as [`Self::LoseAfterAccept`].
    LoseAfterPartial,
}

/// Scripted realtime ASR stage: one transcript delta per audio push
/// and one final transcript on finish.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestAsrStage {
    pub delta_per_frame: String,
    pub final_text: String,
    /// Delay before each push's delta.
    pub push_cadence: Duration,
    /// When set, the worker's open connections are cut once this many audio
    /// pushes arrived: owner loss with no terminal event.
    pub disconnect_after_frames: Option<usize>,
    /// When set, the final transcript ends with the number of samples
    /// pushed, so a test can check every frame reached the runtime however
    /// the worker grouped frames into pushes.
    pub report_pushed_samples: bool,
}

impl Default for TestAsrStage {
    fn default() -> Self {
        Self {
            delta_per_frame: "partial ".into(),
            final_text: "test realtime transcript".into(),
            push_cadence: Duration::from_millis(5),
            disconnect_after_frames: None,
            report_pushed_samples: false,
        }
    }
}

/// Scripted realtime TTS stage: one finish-commit synthesizes `chunk_count`
/// audio chunks of `chunk_bytes` PCM bytes (see [`tts_chunk_bytes`]), then a
/// final empty chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestTtsStage {
    pub chunk_bytes: usize,
    pub chunk_count: usize,
    /// Delay before each chunk.
    pub push_cadence: Duration,
    pub output_sample_rate: u32,
    /// When set, the worker's open connections are cut once this many
    /// chunks were synthesized: owner loss with no terminal event.
    pub disconnect_after_frames: Option<usize>,
}

impl Default for TestTtsStage {
    fn default() -> Self {
        Self {
            chunk_bytes: 64,
            chunk_count: 3,
            push_cadence: Duration::from_millis(5),
            output_sample_rate: 24_000,
            disconnect_after_frames: None,
        }
    }
}

/// The PCM bytes the scripted TTS stage synthesizes for chunk `index`.
pub fn tts_chunk_bytes(index: usize, chunk_bytes: usize) -> Vec<u8> {
    (0..chunk_bytes)
        .map(|position| ((index * 31 + position * 7) % 251 + 1) as u8)
        .collect()
}

/// Every minor-4 chat feature.
pub fn minor4_chat_features() -> BTreeSet<WorkerFeature> {
    BTreeSet::from([
        WorkerFeature::ChatSeed,
        WorkerFeature::ChatStopSequences,
        WorkerFeature::ChatTopK,
        WorkerFeature::ChatPenalties,
        WorkerFeature::ChatLogprobs,
        WorkerFeature::ChatJsonObject,
        WorkerFeature::ChatContextPolicy,
        WorkerFeature::ChatTools,
        WorkerFeature::ChatSkipSpecialTokens,
    ])
}

#[derive(Debug, Clone)]
pub struct TestWorkerConfig {
    pub worker_id: WorkerId,
    pub node_id: NodeId,
    pub incarnation_id: IncarnationId,
    pub deployment_id: DeploymentId,
    pub public_model: ModelAlias,
    pub model_generation: ModelGeneration,
    pub credentials: ServiceCredentials,
    pub max_active_invocations: usize,
    pub max_request_bytes: usize,
    pub max_retained_attempts: usize,
    pub max_output_tokens: u32,
    pub event_channel_capacity: usize,
    pub output_text: String,
    pub output_cadence: Duration,
    /// How long confirmed teardown takes after a cancellation.
    pub cancellation_delay: Duration,
    /// How long runtime admission takes; past the request's queue wait the
    /// worker answers that admission is unknown.
    pub admission_delay: Duration,
    pub script: WorkerScript,
    /// `false` keeps the deployment loading: invocations are refused.
    pub ready: bool,
    /// Serve a realtime speech_to_text deployment instead of chat.
    pub realtime: Option<TestAsrStage>,
    /// Serve a realtime text_to_speech deployment instead of chat. Exclusive
    /// with `realtime`.
    pub realtime_tts: Option<TestTtsStage>,
    /// Realtime frames queued toward the gateway before the worker treats it
    /// as gone.
    pub realtime_max_outbound_queue: usize,
    /// Input tokens served from the prefix cache, reported on completion.
    pub usage_cached_input_tokens: Option<u64>,
    /// Logprob entries the runtime attaches to every text delta.
    pub delta_logprobs: Option<Vec<TokenLogprob>>,
    /// The protocol version the worker implements. Before minor 4 it models
    /// an older worker: set `chat_features` to what that minor had.
    pub protocol_version: SchemaVersion,
    /// Chat control features the worker advertises.
    pub chat_features: BTreeSet<WorkerFeature>,
}

impl Default for TestWorkerConfig {
    fn default() -> Self {
        fn id<T: TryFrom<&'static str>>(value: &'static str) -> T
        where
            T::Error: std::fmt::Debug,
        {
            T::try_from(value).expect("static test identity")
        }
        Self {
            worker_id: id("test-worker-1"),
            node_id: id("test-node-1"),
            incarnation_id: id("test-incarnation-1"),
            deployment_id: id("test-chat-v1"),
            public_model: id("test-chat"),
            model_generation: ModelGeneration::new(1).expect("non-zero"),
            credentials: ServiceCredentials {
                credential_id: id("test-credential-1"),
                bearer_token: ServiceBearerToken::new("test-secret-token")
                    .expect("static test token"),
            },
            max_active_invocations: 1,
            max_request_bytes: 1024 * 1024,
            max_retained_attempts: 64,
            max_output_tokens: 1024,
            event_channel_capacity: DEFAULT_EVENT_CHANNEL_CAPACITY,
            output_text: "deterministic test response".into(),
            output_cadence: Duration::from_millis(5),
            cancellation_delay: Duration::from_millis(25),
            admission_delay: Duration::ZERO,
            script: WorkerScript::Complete,
            ready: true,
            realtime: None,
            realtime_tts: None,
            realtime_max_outbound_queue: REALTIME_SESSION_DEFAULT_LIMITS.max_outbound_queue,
            usage_cached_input_tokens: None,
            delta_logprobs: None,
            protocol_version: PROTOCOL_V1,
            chat_features: minor4_chat_features(),
        }
    }
}

impl TestWorkerConfig {
    fn worker_config(&self) -> std::io::Result<WorkerConfig> {
        if self.realtime.is_some() && self.realtime_tts.is_some() {
            return Err(std::io::Error::other(
                "realtime and realtime_tts stages are mutually exclusive",
            ));
        }
        let (task, input, output) = if self.realtime.is_some() {
            (
                TaskKind::SpeechToText,
                InputFormat::PcmAudio,
                OutputFormat::Text,
            )
        } else if self.realtime_tts.is_some() {
            (
                TaskKind::TextToSpeech,
                InputFormat::Text,
                OutputFormat::PcmAudio,
            )
        } else {
            (
                TaskKind::Chat,
                InputFormat::ChatMessages,
                OutputFormat::Text,
            )
        };
        let realtime = task != TaskKind::Chat;
        let mut features = BTreeSet::from([
            WorkerFeature::Streaming,
            WorkerFeature::Cancellation,
            WorkerFeature::AttemptQuery,
        ]);
        if realtime {
            features.insert(WorkerFeature::RealtimeSocket);
        } else {
            features.extend(self.chat_features.iter().copied());
        }
        let config = WorkerConfig {
            descriptor: WorkerDescriptor {
                schema_version: self.protocol_version,
                supported_protocol_versions: vec![self.protocol_version],
                worker_id: self.worker_id.clone(),
                node_id: self.node_id.clone(),
                incarnation_id: self.incarnation_id.clone(),
                build_version: "scripted-test-runtime".into(),
                assignment: DeviceAssignment::Cpu {
                    thread_budget: 1,
                    affinity: Vec::new(),
                    host_memory_limit_bytes: 64 * 1024 * 1024,
                },
                features,
            },
            deployment: LoadedDeployment {
                deployment_id: self.deployment_id.clone(),
                public_model: self.public_model.clone(),
                artifact_revision: ArtifactRevision::new("test-artifact-v1")
                    .expect("static identity"),
                model_generation: self.model_generation,
                task,
                backend: BackendKind::Cpu,
                precision: "test".into(),
                execution_representation: if realtime {
                    "scripted-realtime".into()
                } else {
                    "scripted-text".into()
                },
                tokenizer_revision: None,
                readiness: if self.ready {
                    ModelReadiness::Ready
                } else {
                    ModelReadiness::Loading
                },
                capability: Capability {
                    task,
                    streaming: true,
                    realtime,
                    cancellation: CancellationBehavior::Cooperative,
                    accepted_input_formats: BTreeSet::from([input]),
                    output_formats: BTreeSet::from([output]),
                    max_input_bytes: self.max_request_bytes as u64,
                    max_context_tokens: (!realtime).then_some(4096),
                    max_output_tokens: (!realtime).then_some(self.max_output_tokens),
                },
                kv_cache_usage_pct: None,
                prefix_hits_total: None,
                prefix_queries_total: None,
                prefix_evictions_total: None,
                kv_host_pages: None,
                kv_demotions_total: None,
                kv_promotions_total: None,
                kv_promotion_latency_avg_seconds: None,
                tokens_out_per_s_ema: None,
                observation_cost_units: None,
            },
            credentials: self.credentials.clone(),
            max_active_invocations: self.max_active_invocations,
            max_request_bytes: self.max_request_bytes,
            max_retained_attempts: self.max_retained_attempts,
            attempt_retention: DEFAULT_ATTEMPT_RETENTION,
            event_channel_capacity: self.event_channel_capacity,
            max_event_bytes: DEFAULT_MAX_EVENT_BYTES,
            slow_consumer_timeout: DEFAULT_SLOW_CONSUMER_TIMEOUT,
            realtime_session_limits: RuntimeRealtimeSessionLimits {
                max_outbound_queue: self.realtime_max_outbound_queue,
                ..REALTIME_SESSION_DEFAULT_LIMITS
            },
        };
        config.validate().map_err(std::io::Error::other)?;
        Ok(config)
    }
}

/// What the scripted runtime saw, and the switch that cuts connections.
#[derive(Default)]
struct Observed {
    admitted: AtomicU64,
    invocations: Mutex<Vec<InvocationRequest>>,
    /// The `traceparent` of every request the worker received, in order.
    traceparents: Mutex<Vec<String>>,
}

/// A real worker on a loopback socket. Dropping it stops accepting
/// connections and cuts the open ones.
pub struct TestWorker {
    address: SocketAddr,
    config: TestWorkerConfig,
    service: WorkerService<ScriptedExecutor>,
    observed: Arc<Observed>,
    server: tokio::task::JoinHandle<()>,
    _cuts: Arc<watch::Sender<u64>>,
}

impl std::fmt::Debug for TestWorker {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TestWorker")
            .field("address", &self.address)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl TestWorker {
    pub async fn spawn(config: TestWorkerConfig) -> std::io::Result<Self> {
        Self::spawn_at(config, SocketAddr::from(([127, 0, 0, 1], 0))).await
    }

    /// Like [`Self::spawn`], at a fixed address: a test can approve the
    /// endpoint first and start the worker behind it later.
    pub async fn spawn_at(config: TestWorkerConfig, address: SocketAddr) -> std::io::Result<Self> {
        let worker_config = config.worker_config()?;
        let observed = Arc::new(Observed::default());
        let cuts = Arc::new(watch::channel(0_u64).0);
        let executor = ScriptedExecutor {
            config: config.clone(),
            observed: Arc::clone(&observed),
            cuts: Arc::clone(&cuts),
        };
        let service = WorkerService::new(worker_config, executor).map_err(std::io::Error::other)?;
        let listener = tokio::net::TcpListener::bind(address).await?;
        let address = listener.local_addr()?;
        let listener = CuttableListener {
            inner: listener,
            cuts: cuts.subscribe(),
        };
        let seen = Arc::clone(&observed);
        let router = service.router().layer(axum::middleware::from_fn(
            move |request: axum::extract::Request, next: axum::middleware::Next| {
                if let Some(value) = request
                    .headers()
                    .get(TRACEPARENT_HEADER)
                    .and_then(|value| value.to_str().ok())
                {
                    seen.traceparents
                        .lock()
                        .expect("test worker record poisoned")
                        .push(value.to_string());
                }
                next.run(request)
            },
        ));
        let server = tokio::spawn(async move {
            let _ = axum::serve(listener, router).await;
        });
        Ok(Self {
            address,
            config,
            service,
            observed,
            server,
            _cuts: cuts,
        })
    }

    pub fn endpoint(&self) -> String {
        format!("http://{}/", self.address)
    }

    pub fn config(&self) -> &TestWorkerConfig {
        &self.config
    }

    /// Chat invocations the runtime admitted since spawn: which worker
    /// actually served traffic.
    pub fn served_invocations(&self) -> u64 {
        self.observed.admitted.load(Ordering::SeqCst)
    }

    /// Invocations holding worker capacity, through confirmed teardown.
    pub fn active_invocations(&self) -> usize {
        self.service.active_invocations()
    }

    /// The last chat invocation the runtime admitted, as decoded from the
    /// wire.
    pub fn last_invocation(&self) -> Option<InvocationRequest> {
        self.observed
            .invocations
            .lock()
            .expect("test worker record poisoned")
            .last()
            .cloned()
    }

    /// The `traceparent` headers the worker received, oldest first.
    pub fn received_traceparents(&self) -> Vec<String> {
        self.observed
            .traceparents
            .lock()
            .expect("test worker record poisoned")
            .clone()
    }

    /// The time budget an admitted attempt arrived with.
    pub fn attempt_remaining_time_ms(&self, attempt_id: &AttemptId) -> Option<u64> {
        self.observed
            .invocations
            .lock()
            .expect("test worker record poisoned")
            .iter()
            .find(|invocation| invocation.attempt_id == *attempt_id)
            .map(|invocation| invocation.remaining_time_ms)
    }
}

impl Drop for TestWorker {
    fn drop(&mut self) {
        self.server.abort();
        cut_connections(&self._cuts);
    }
}

fn cut_connections(cuts: &watch::Sender<u64>) {
    cuts.send_modify(|generation| *generation += 1);
}

/// Accepts loopback connections that a later cut severs, as if the worker
/// process died under them. Connections accepted after a cut are unaffected.
struct CuttableListener {
    inner: tokio::net::TcpListener,
    cuts: watch::Receiver<u64>,
}

impl axum::serve::Listener for CuttableListener {
    type Io = CuttableStream;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            match self.inner.accept().await {
                Ok((stream, address)) => {
                    let signal = Arc::new(CutSignal::default());
                    let mut cuts = self.cuts.clone();
                    let accepted_at = *cuts.borrow_and_update();
                    let watched = Arc::clone(&signal);
                    tokio::spawn(async move {
                        let _ = cuts.wait_for(|generation| *generation > accepted_at).await;
                        watched.cut();
                    });
                    return (CuttableStream { stream, signal }, address);
                }
                Err(_) => tokio::time::sleep(Duration::from_millis(5)).await,
            }
        }
    }

    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        self.inner.local_addr()
    }
}

/// Set once a connection is cut; wakes whichever task waits on either half.
#[derive(Default)]
struct CutSignal {
    cut: AtomicBool,
    read_waker: Mutex<Option<Waker>>,
    write_waker: Mutex<Option<Waker>>,
}

impl CutSignal {
    fn cut(&self) {
        self.cut.store(true, Ordering::SeqCst);
        for slot in [&self.read_waker, &self.write_waker] {
            if let Some(waker) = slot.lock().expect("cut waker poisoned").take() {
                waker.wake();
            }
        }
    }

    /// Registers the waiting task first, so a cut that lands after the check
    /// still wakes it.
    fn is_cut(&self, slot: &Mutex<Option<Waker>>, cx: &Context<'_>) -> bool {
        *slot.lock().expect("cut waker poisoned") = Some(cx.waker().clone());
        self.cut.load(Ordering::SeqCst)
    }
}

struct CuttableStream {
    stream: tokio::net::TcpStream,
    signal: Arc<CutSignal>,
}

impl CuttableStream {
    /// Ends the peer's view of the connection (FIN) and fails this side, as
    /// a dead worker process would.
    fn sever(&mut self, cx: &mut Context<'_>) -> std::io::Error {
        let _ = Pin::new(&mut self.stream).poll_shutdown(cx);
        std::io::Error::from(std::io::ErrorKind::ConnectionReset)
    }
}

impl AsyncRead for CuttableStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let this = self.get_mut();
        if this.signal.is_cut(&this.signal.read_waker, cx) {
            return Poll::Ready(Err(this.sever(cx)));
        }
        Pin::new(&mut this.stream).poll_read(cx, buf)
    }
}

impl AsyncWrite for CuttableStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        let this = self.get_mut();
        if this.signal.is_cut(&this.signal.write_waker, cx) {
            return Poll::Ready(Err(this.sever(cx)));
        }
        Pin::new(&mut this.stream).poll_write(cx, buf)
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        let this = self.get_mut();
        if this.signal.is_cut(&this.signal.write_waker, cx) {
            return Poll::Ready(Err(this.sever(cx)));
        }
        Pin::new(&mut this.stream).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.get_mut().stream).poll_shutdown(cx)
    }
}

struct ScriptedExecutor {
    config: TestWorkerConfig,
    observed: Arc<Observed>,
    cuts: Arc<watch::Sender<u64>>,
}

#[async_trait]
impl InvocationExecutor for ScriptedExecutor {
    async fn admit(
        &self,
        request: &InvocationRequest,
    ) -> Result<AdmittedInvocation, AdmissionFailure> {
        if self.config.realtime.is_some() || self.config.realtime_tts.is_some() {
            return Err(AdmissionFailure::new(
                RejectionCode::IncompatibleTask,
                "a realtime test worker serves realtime sessions only",
            ));
        }
        self.observed.admitted.fetch_add(1, Ordering::SeqCst);
        self.observed
            .invocations
            .lock()
            .expect("test worker record poisoned")
            .push(request.clone());
        tokio::time::sleep(self.config.admission_delay).await;
        Ok(AdmittedInvocation::new(Box::new(ScriptedExecution {
            config: self.config.clone(),
            logprobs: core_logprobs(self.config.delta_logprobs.as_deref()),
            step: 0,
            cancelled: AtomicBool::new(false),
        })))
    }

    fn realtime_runner(&self) -> Option<Arc<dyn RealtimeStageRunner>> {
        if let Some(stage) = &self.config.realtime {
            return Some(Arc::new(ScriptedAsrRunner {
                stage: stage.clone(),
                cuts: Arc::clone(&self.cuts),
            }));
        }
        self.config.realtime_tts.as_ref().map(|stage| {
            Arc::new(ScriptedTtsRunner {
                stage: stage.clone(),
                cuts: Arc::clone(&self.cuts),
            }) as Arc<dyn RealtimeStageRunner>
        })
    }
}

fn core_logprobs(entries: Option<&[TokenLogprob]>) -> Vec<izwi_core::engine::TokenLogprob> {
    entries
        .unwrap_or_default()
        .iter()
        .map(|entry| izwi_core::engine::TokenLogprob {
            token: entry.token.clone(),
            logprob: entry.logprob,
            bytes: entry.bytes.clone(),
            top_logprobs: entry
                .top_logprobs
                .iter()
                .map(|top| izwi_core::engine::TopTokenLogprob {
                    token: top.token.clone(),
                    logprob: top.logprob,
                    bytes: top.bytes.clone(),
                })
                .collect(),
        })
        .collect()
}

struct ScriptedExecution {
    config: TestWorkerConfig,
    logprobs: Vec<izwi_core::engine::TokenLogprob>,
    step: usize,
    cancelled: AtomicBool,
}

impl ScriptedExecution {
    fn delta(&self, text: String) -> Option<ExecutionEvent> {
        Some(ExecutionEvent::TextDelta {
            text,
            logprobs: self.logprobs.clone(),
        })
    }

    fn completed(&self, output_tokens: u64) -> Option<ExecutionEvent> {
        Some(ExecutionEvent::Completed {
            text: None,
            finish_reason: FinishReason::Stop,
            input_tokens: 1,
            output_tokens,
            cached_input_tokens: self.config.usage_cached_input_tokens,
        })
    }
}

#[async_trait]
impl AdmittedExecution for ScriptedExecution {
    async fn next_event(&mut self) -> Option<ExecutionEvent> {
        let step = self.step;
        self.step += 1;
        if step == 0 {
            tokio::time::sleep(self.config.output_cadence).await;
        }
        match (&self.config.script, step) {
            (WorkerScript::Complete, 0) => self.delta(self.config.output_text.clone()),
            (WorkerScript::Complete, 1) => {
                tokio::time::sleep(self.config.output_cadence).await;
                self.completed(3)
            }
            (WorkerScript::Hang, _) | (WorkerScript::TextDeltaThenHang, 1..) => {
                std::future::pending().await
            }
            (
                WorkerScript::TextDeltaThenHang
                | WorkerScript::LoseAfterPartial
                | WorkerScript::TextDeltaThenFail(_),
                0,
            ) => self.delta(self.config.output_text.clone()),
            (WorkerScript::TextDeltaThenFail(code), 1) => {
                Some(ExecutionEvent::Failed(ExecutionFailure {
                    code: *code,
                    message: "scripted model failure".into(),
                }))
            }
            (WorkerScript::ManyTextDeltas { count, text_bytes }, step) if step < *count => {
                self.delta("x".repeat(*text_bytes))
            }
            (WorkerScript::ManyTextDeltas { count, .. }, step) if step == *count => {
                self.completed(u64::try_from(*count).unwrap_or(u64::MAX))
            }
            (WorkerScript::TextDeltas(texts), step) if step < texts.len() => {
                self.delta(texts[step].clone())
            }
            (WorkerScript::TextDeltas(texts), step) if step == texts.len() => {
                self.completed(u64::try_from(texts.len()).unwrap_or(u64::MAX))
            }
            (WorkerScript::EmptyDeltaTrickle, step) => {
                if step > 0 {
                    tokio::time::sleep(self.config.output_cadence).await;
                }
                self.delta(String::new())
            }
            _ => None,
        }
    }

    fn request_cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    async fn wait_for_teardown(self: Box<Self>) -> ExecutionTeardown {
        if matches!(
            self.config.script,
            WorkerScript::LoseAfterAccept | WorkerScript::LoseAfterPartial
        ) {
            return ExecutionTeardown::Unconfirmed;
        }
        if matches!(self.config.script, WorkerScript::TextDeltaThenFail(_)) {
            return ExecutionTeardown::Failed;
        }
        if self.cancelled.load(Ordering::SeqCst) {
            tokio::time::sleep(self.config.cancellation_delay).await;
            return ExecutionTeardown::Cancelled;
        }
        ExecutionTeardown::Completed
    }
}

struct ScriptedAsrRunner {
    stage: TestAsrStage,
    cuts: Arc<watch::Sender<u64>>,
}

#[async_trait]
impl RealtimeStageRunner for ScriptedAsrRunner {
    fn stage_task(&self) -> TaskKind {
        TaskKind::SpeechToText
    }

    async fn start_asr_stream(
        &self,
        _language: Option<&str>,
    ) -> Result<Box<dyn RealtimeAsrStageStream>, izwi_core::Error> {
        Ok(Box::new(ScriptedAsrStream {
            stage: self.stage.clone(),
            cuts: Arc::clone(&self.cuts),
            frames: 0,
            samples: 0,
        }))
    }
}

struct ScriptedAsrStream {
    stage: TestAsrStage,
    cuts: Arc<watch::Sender<u64>>,
    frames: usize,
    samples: usize,
}

#[async_trait]
impl RealtimeAsrStageStream for ScriptedAsrStream {
    async fn push_samples(
        &mut self,
        samples: &[f32],
        _sample_rate: u32,
    ) -> Result<Vec<RuntimeAsrRealtimeEvent>, izwi_core::Error> {
        self.frames += 1;
        self.samples += samples.len();
        if self
            .stage
            .disconnect_after_frames
            .is_some_and(|frames| self.frames >= frames)
        {
            cut_connections(&self.cuts);
            return std::future::pending().await;
        }
        tokio::time::sleep(self.stage.push_cadence).await;
        Ok(vec![RuntimeAsrRealtimeEvent {
            delta: self.stage.delta_per_frame.clone(),
            text: String::new(),
            is_final: false,
            chunk_index: self.frames,
        }])
    }

    async fn finish(&mut self) -> Result<Vec<RuntimeAsrRealtimeEvent>, izwi_core::Error> {
        let text = if self.stage.report_pushed_samples {
            format!("{} after {} samples", self.stage.final_text, self.samples)
        } else {
            self.stage.final_text.clone()
        };
        Ok(vec![RuntimeAsrRealtimeEvent {
            delta: String::new(),
            text,
            is_final: true,
            chunk_index: self.frames,
        }])
    }
}

struct ScriptedTtsRunner {
    stage: TestTtsStage,
    cuts: Arc<watch::Sender<u64>>,
}

#[async_trait]
impl RealtimeStageRunner for ScriptedTtsRunner {
    fn stage_task(&self) -> TaskKind {
        TaskKind::TextToSpeech
    }

    async fn start_asr_stream(
        &self,
        _language: Option<&str>,
    ) -> Result<Box<dyn RealtimeAsrStageStream>, izwi_core::Error> {
        Err(izwi_core::Error::ConfigError(
            "stage does not serve realtime ASR sessions".into(),
        ))
    }

    fn output_audio_spec(&self) -> Option<RealtimeAudioSpec> {
        Some(RealtimeAudioSpec {
            codec: RealtimeAudioCodec::PcmI16Le,
            sample_rate: self.stage.output_sample_rate,
            channels: 1,
        })
    }

    async fn start_tts_stream(
        &self,
        _text: String,
        _options: &crate::RealtimeTtsOptions,
        _deadline: std::time::Instant,
    ) -> Result<Box<dyn RealtimeTtsStageStream>, izwi_core::Error> {
        Ok(Box::new(ScriptedTtsStream {
            stage: self.stage.clone(),
            cuts: Arc::clone(&self.cuts),
            chunks: 0,
        }))
    }
}

struct ScriptedTtsStream {
    stage: TestTtsStage,
    cuts: Arc<watch::Sender<u64>>,
    chunks: usize,
}

#[async_trait]
impl RealtimeTtsStageStream for ScriptedTtsStream {
    async fn next_chunk(&mut self) -> Result<Option<AudioChunk>, izwi_core::Error> {
        let index = self.chunks;
        if index > self.stage.chunk_count {
            return Ok(None);
        }
        self.chunks += 1;
        if self.stage.disconnect_after_frames == Some(index) {
            cut_connections(&self.cuts);
            return std::future::pending().await;
        }
        tokio::time::sleep(self.stage.push_cadence).await;
        let request_id = "scripted-tts".to_string();
        let chunk = if index == self.stage.chunk_count {
            AudioChunk::final_chunk(request_id, index, Vec::new())
        } else {
            let samples = pcm_samples(&tts_chunk_bytes(index, self.stage.chunk_bytes));
            AudioChunk::new(request_id, index, samples)
        };
        Ok(Some(chunk.with_sample_rate(self.stage.output_sample_rate)))
    }
}

/// Samples the worker quantizes back to exactly `bytes` (PCM i16 LE). Half
/// a step away from zero survives the worker's truncating conversion.
fn pcm_samples(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let value = f32::from(i16::from_le_bytes(*pair));
            (value + 0.5_f32.copysign(value)) / 32_767.0
        })
        .collect()
}
