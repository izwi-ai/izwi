//! Private, versioned HTTP worker boundary for Izwi inference runtimes.
//!
//! The HTTP layer owns bounded request parsing, attempt fencing, and transport-level
//! concurrency. An [`InvocationExecutor`] owns authoritative runtime admission. An
//! invocation is never advertised as accepted until both layers have admitted it.

use async_stream::stream;
use async_trait::async_trait;
use axum::{
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, Path, Request as AxumRequest, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use izwi_core::{ManagedKvRuntimeSnapshot, RuntimeHealth, RuntimeTelemetrySnapshot};
use izwi_serving_protocol::*;
use std::{
    collections::{HashMap, VecDeque},
    convert::Infallible,
    fmt::Write as _,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, MutexGuard, PoisonError,
    },
    time::Duration,
};
use tokio::sync::{mpsc, watch, OwnedRwLockReadGuard, OwnedSemaphorePermit, RwLock, Semaphore};
use tokio::time::Instant;
use tracing::Instrument;

mod realtime;
mod runtime;
#[cfg(feature = "test-support")]
pub mod test_support;

pub use realtime::{RuntimeRealtimeSessionLimits, REALTIME_SESSION_DEFAULT_LIMITS};
pub use runtime::{
    admission_rejection_code, execution_error_code, warm_up_asr_runtime, warm_up_chat_runtime,
    warm_up_tts_runtime, RuntimeChatExecutor, RuntimeRealtimeAsrExecutor,
    RuntimeRealtimeTtsExecutor,
};

pub const DEFAULT_MAX_REQUEST_BYTES: usize = 1024 * 1024;
pub const DEFAULT_MAX_RETAINED_ATTEMPTS: usize = 1024;
pub const DEFAULT_ATTEMPT_RETENTION: Duration = Duration::from_secs(300);
pub const DEFAULT_EVENT_CHANNEL_CAPACITY: usize = 4;
pub const DEFAULT_MAX_EVENT_BYTES: usize = 1024 * 1024;
pub const DEFAULT_SLOW_CONSUMER_TIMEOUT: Duration = izwi_core::DEFAULT_STREAM_SLOW_CONSUMER_TIMEOUT;
pub const WORKER_METRICS_PATH: &str = "/internal/v1/metrics/prometheus";
const EVENT_ENVELOPE_ALLOWANCE: usize = 1024;
const MAX_PROMETHEUS_RESPONSE_BYTES: usize = 8192;

/// Immutable worker identity and bounded local resource policy.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub descriptor: WorkerDescriptor,
    pub deployment: LoadedDeployment,
    pub credentials: ServiceCredentials,
    pub max_active_invocations: usize,
    pub max_request_bytes: usize,
    pub max_retained_attempts: usize,
    pub attempt_retention: Duration,
    pub event_channel_capacity: usize,
    pub max_event_bytes: usize,
    /// How long a streaming reader may leave text unaccepted before the
    /// invocation is cancelled. Until then a lagging reader's deltas are
    /// coalesced, never dropped.
    pub slow_consumer_timeout: Duration,
    /// Per-realtime-session transport bounds; always within the protocol's
    /// hard caps. Ignored by deployments that do not serve realtime.
    pub realtime_session_limits: RuntimeRealtimeSessionLimits,
}

impl WorkerConfig {
    pub fn validate(&self) -> Result<(), WorkerConfigError> {
        if self.descriptor.schema_version.major != PROTOCOL_V1.major
            || !self
                .descriptor
                .supported_protocol_versions
                .iter()
                .any(|version| version.major == PROTOCOL_V1.major)
        {
            return Err(WorkerConfigError::ProtocolVersion);
        }
        if self.descriptor.worker_id.as_str().is_empty()
            || self.descriptor.incarnation_id.as_str().is_empty()
        {
            return Err(WorkerConfigError::Identity);
        }
        if self.descriptor.assignment.backend() != self.deployment.backend {
            return Err(WorkerConfigError::BackendMismatch);
        }
        match self.deployment.task {
            TaskKind::Chat => {
                if self.deployment.capability.task != TaskKind::Chat
                    || !self
                        .deployment
                        .capability
                        .accepted_input_formats
                        .contains(&InputFormat::ChatMessages)
                    || !self
                        .deployment
                        .capability
                        .output_formats
                        .contains(&OutputFormat::Text)
                {
                    return Err(WorkerConfigError::UnsupportedDeployment);
                }
            }
            TaskKind::SpeechToText => {
                if self.deployment.capability.task != TaskKind::SpeechToText
                    || !self.deployment.capability.realtime
                    || !self
                        .deployment
                        .capability
                        .accepted_input_formats
                        .contains(&InputFormat::PcmAudio)
                    || !self
                        .deployment
                        .capability
                        .output_formats
                        .contains(&OutputFormat::Text)
                    || !self
                        .descriptor
                        .features
                        .contains(&WorkerFeature::RealtimeSocket)
                {
                    return Err(WorkerConfigError::UnsupportedDeployment);
                }
            }
            TaskKind::TextToSpeech => {
                if self.deployment.capability.task != TaskKind::TextToSpeech
                    || !self.deployment.capability.realtime
                    || !self
                        .deployment
                        .capability
                        .accepted_input_formats
                        .contains(&InputFormat::Text)
                    || !self
                        .deployment
                        .capability
                        .output_formats
                        .contains(&OutputFormat::PcmAudio)
                    || !self
                        .descriptor
                        .features
                        .contains(&WorkerFeature::RealtimeSocket)
                {
                    return Err(WorkerConfigError::UnsupportedDeployment);
                }
            }
            TaskKind::Unknown => return Err(WorkerConfigError::UnsupportedDeployment),
        }
        if self.max_active_invocations == 0 {
            return Err(WorkerConfigError::ZeroActiveCapacity);
        }
        if self.realtime_session_limits.validate().is_err() {
            return Err(WorkerConfigError::InvalidRealtimeLimits);
        }
        if self.max_request_bytes == 0 {
            return Err(WorkerConfigError::ZeroRequestLimit);
        }
        if self.max_retained_attempts < self.max_active_invocations {
            return Err(WorkerConfigError::AttemptRetention);
        }
        if self.attempt_retention < Duration::from_secs(1)
            || self.attempt_retention > Duration::from_secs(86_400)
        {
            return Err(WorkerConfigError::AttemptRetentionWindow);
        }
        // Accepted plus one text delta plus one terminal event must never block
        // completion on a slow or disconnected transport consumer.
        if self.event_channel_capacity < 3 {
            return Err(WorkerConfigError::EventChannelCapacity);
        }
        if self.max_event_bytes < EVENT_ENVELOPE_ALLOWANCE * 2 {
            return Err(WorkerConfigError::EventLimit);
        }
        if self.slow_consumer_timeout.is_zero() {
            return Err(WorkerConfigError::SlowConsumerTimeout);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorkerConfigError {
    #[error("worker must advertise private protocol v1")]
    ProtocolVersion,
    #[error("worker identity fields must be non-empty")]
    Identity,
    #[error("device assignment and deployment backend differ")]
    BackendMismatch,
    #[error("this worker slice supports exactly one chat deployment")]
    UnsupportedDeployment,
    #[error("max_active_invocations must be non-zero")]
    ZeroActiveCapacity,
    #[error("max_request_bytes must be non-zero")]
    ZeroRequestLimit,
    #[error("attempt retention must cover every active invocation")]
    AttemptRetention,
    #[error("attempt retention window must be between one second and one day")]
    AttemptRetentionWindow,
    #[error("event channel capacity must hold accepted, delta, and terminal events")]
    EventChannelCapacity,
    #[error("slow_consumer_timeout must be non-zero")]
    SlowConsumerTimeout,
    #[error("max_event_bytes must be at least 2048")]
    EventLimit,
    #[error("realtime session limits must be non-zero and within the protocol caps")]
    InvalidRealtimeLimits,
}

/// A rejection returned before runtime ownership has been accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionFailure {
    pub code: RejectionCode,
    pub message: String,
    pub retry_after_ms: Option<u64>,
}

impl AdmissionFailure {
    pub fn new(code: RejectionCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            retry_after_ms: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionFailure {
    pub code: InvocationErrorCode,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionEvent {
    /// DS9.3: a text delta with its per-token logprob entries (empty unless
    /// the request asked for logprobs).
    TextDelta {
        text: String,
        logprobs: Vec<izwi_core::engine::TokenLogprob>,
    },
    Completed {
        /// Non-streaming runtimes return their one bounded text value here.
        /// Streaming adapters leave this empty after forwarding deltas.
        text: Option<String>,
        finish_reason: FinishReason,
        input_tokens: u64,
        output_tokens: u64,
        /// DS9.1: input tokens served from the managed prefix cache. Always
        /// a subset of `input_tokens`; `None` when the runtime did not
        /// measure prefix reuse.
        cached_input_tokens: Option<u64>,
    },
    Failed(ExecutionFailure),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionTeardown {
    Completed,
    Cancelled,
    Failed,
    /// Physical teardown could not be proven; capacity must remain held.
    Unconfirmed,
}

/// Opaque admitted runtime execution retained through confirmed teardown.
#[async_trait]
pub trait AdmittedExecution: Send + 'static {
    /// Return the next bounded output event. `None` is not teardown proof.
    async fn next_event(&mut self) -> Option<ExecutionEvent>;

    /// Cooperatively signal cancellation without claiming execution stopped.
    fn request_cancel(&self);

    /// Resolve only after native completion or exact-session cleanup.
    async fn wait_for_teardown(self: Box<Self>) -> ExecutionTeardown;
}

/// Runtime ownership obtained atomically before the HTTP worker emits Accepted.
pub struct AdmittedInvocation {
    execution: Box<dyn AdmittedExecution>,
}

impl AdmittedInvocation {
    pub fn new(execution: Box<dyn AdmittedExecution>) -> Self {
        Self { execution }
    }

    fn into_execution(self) -> Box<dyn AdmittedExecution> {
        self.execution
    }
}

/// Adapter from the private protocol into an inference runtime.
///
/// `admit` must return only after the runtime owns all authoritative request,
/// model-residency, scheduler, and physical-capacity leases needed by the
/// invocation. The returned execution must retain ownership until native
/// execution has completed or teardown has been confirmed. The worker never
/// drops an in-flight admission future merely because its HTTP wait expires.
#[async_trait]
pub trait InvocationExecutor: Send + Sync + 'static {
    async fn admit(
        &self,
        request: &InvocationRequest,
    ) -> Result<AdmittedInvocation, AdmissionFailure>;

    /// Engine-owned runtime telemetry when the executor embeds a serving
    /// runtime. Surfaced on the worker's metrics endpoint so serving evidence
    /// (managed-KV prefix and tensor-snapshot counters) is observable in the
    /// gateway/worker topology. Executors without an engine return `None`.
    async fn runtime_telemetry(&self) -> Option<RuntimeTelemetrySnapshot> {
        None
    }

    /// Physical execution health of the embedded runtime: admission poison
    /// and latched fatal device faults. Status reports it instead of the
    /// static startup readiness. Executors without an engine stay healthy.
    fn runtime_health(&self) -> RuntimeHealth {
        RuntimeHealth::Healthy
    }

    /// Realtime stage runner when the executor embeds a runtime able to serve
    /// realtime WebSocket sessions (`izwi-realtime-v1`). Executors without
    /// realtime support return `None`, and the realtime socket route fails
    /// closed rather than advertising an unusable surface.
    fn realtime_runner(&self) -> Option<std::sync::Arc<dyn RealtimeStageRunner>> {
        None
    }
}

/// Per-session realtime ASR stream handle produced by a
/// [`RealtimeStageRunner`]. Every push is one input quantum: cancellation
/// lands between pushes, never mid-decode. Dropping the handle releases the
/// stage's engine leases — the worker's teardown confirmation for the ASR
/// stage.
#[async_trait::async_trait]
pub trait RealtimeAsrStageStream: Send {
    /// Pushes one frame of audio at the stream's sample rate.
    async fn push_samples(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<izwi_core::RuntimeAsrRealtimeEvent>, izwi_core::Error>;

    /// Finalizes the stream and returns the final transcript events.
    async fn finish(&mut self)
        -> Result<Vec<izwi_core::RuntimeAsrRealtimeEvent>, izwi_core::Error>;
}

/// Per-session realtime TTS stream handle produced by a
/// [`RealtimeStageRunner`]. Synthesis runs against the worker's runtime and
/// audio chunks arrive on the handle until the stream ends; dropping the
/// handle aborts the synthesis and releases the stage's runtime leases — the
/// worker's teardown confirmation for the TTS stage.
#[async_trait::async_trait]
pub trait RealtimeTtsStageStream: Send {
    /// Receives the next synthesized audio chunk; `Ok(None)` ends the stream.
    async fn next_chunk(&mut self) -> Result<Option<izwi_core::AudioChunk>, izwi_core::Error>;
}

/// How a realtime TTS session speaks: the text-stream admission's voice.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RealtimeTtsOptions {
    pub speaker: Option<String>,
    pub language: Option<String>,
}

/// Execution surface for one realtime stage on this worker's runtime.
#[async_trait::async_trait]
pub trait RealtimeStageRunner: Send + Sync + 'static {
    /// The stage task this runner serves.
    fn stage_task(&self) -> TaskKind;

    /// Starts a realtime ASR stream for the deployed variant.
    async fn start_asr_stream(
        &self,
        language: Option<&str>,
    ) -> Result<Box<dyn RealtimeAsrStageStream>, izwi_core::Error>;

    /// The output audio spec announced in the `Admitted` frame for stages
    /// that emit audio (TTS-stream). `None` for input-only stages.
    fn output_audio_spec(&self) -> Option<RealtimeAudioSpec> {
        None
    }

    /// Starts realtime TTS synthesis of one piece of the session's text,
    /// spoken with `options`. Synthesis must stop by the given deadline; the
    /// session's own cancellation ladder remains authoritative.
    async fn start_tts_stream(
        &self,
        text: String,
        options: &RealtimeTtsOptions,
        deadline: std::time::Instant,
    ) -> Result<Box<dyn RealtimeTtsStageStream>, izwi_core::Error> {
        let _ = (text, options, deadline);
        Err(izwi_core::Error::ConfigError(
            "stage does not serve realtime TTS sessions".into(),
        ))
    }
}

#[derive(Clone)]
struct AttemptRecord {
    identity: AttemptIdentity,
    digest: RequestDigest,
    state: AttemptState,
    last_sequence: Option<u64>,
    cancel_requested: bool,
    cancel: Option<watch::Sender<bool>>,
    evict_after: Option<Instant>,
}

/// Locks the attempt table or an admission slot. A panic while one was held
/// must not turn every later request on this worker into a panic, so a
/// poisoned lock is recovered with whatever state the interrupted section
/// left, the policy the gateway registry and tenant limits follow too.
fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[derive(Default)]
struct AttemptTable {
    records: HashMap<AttemptId, AttemptRecord>,
    order: VecDeque<AttemptId>,
}

struct WorkerState<E> {
    config: WorkerConfig,
    executor: Arc<E>,
    capacity: Arc<Semaphore>,
    admission_gate: Arc<RwLock<()>>,
    attempts: Mutex<AttemptTable>,
    status_sequence: AtomicU64,
    draining: AtomicBool,
    metrics: WorkerMetrics,
}

#[derive(Clone, Default)]
struct WorkerMetrics {
    inner: Arc<WorkerMetricCounters>,
}

#[derive(Default)]
struct WorkerMetricCounters {
    admitted: AtomicU64,
    rejected: AtomicU64,
    auth_rejections: AtomicU64,
    body_limit_rejections: AtomicU64,
    admission_unknown: AtomicU64,
    completed: AtomicU64,
    failed: AtomicU64,
    cancelled: AtomicU64,
    unconfirmed_teardown: AtomicU64,
    queue_wait_observations: AtomicU64,
    queue_wait_micros: AtomicU64,
    execution_observations: AtomicU64,
    execution_micros: AtomicU64,
    cancellation_requests: AtomicU64,
    cancellation_to_stop_observations: AtomicU64,
    cancellation_to_stop_micros: AtomicU64,
    event_delivery_failures: AtomicU64,
    tokens_out_ema: TokensOutEma,
}

/// Exponentially weighted output tokens-per-second over completed
/// invocations. A statistical routing signal, so relaxed ordering and a
/// benign last-writer race between concurrent completions are acceptable.
/// A recorded rate is strictly positive, which makes zero bits mean "no
/// observation yet" without a separate flag.
#[derive(Default)]
struct TokensOutEma {
    bits: AtomicU64,
}

impl TokensOutEma {
    const ALPHA: f64 = 0.3;

    fn record(&self, output_tokens: u64, elapsed: Duration) {
        let secs = elapsed.as_secs_f64();
        if output_tokens == 0 || !secs.is_finite() || secs <= 0.0 {
            return;
        }
        let sample = output_tokens as f64 / secs;
        if !sample.is_finite() {
            return;
        }
        let mut current = self.bits.load(Ordering::Relaxed);
        loop {
            let next = match f64::from_bits(current) {
                0.0 => sample,
                previous => Self::ALPHA * sample + (1.0 - Self::ALPHA) * previous,
            };
            match self.bits.compare_exchange_weak(
                current,
                next.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }

    fn load(&self) -> Option<f64> {
        match f64::from_bits(self.bits.load(Ordering::Relaxed)) {
            0.0 => None,
            value if value.is_finite() => Some(value),
            _ => None,
        }
    }
}

impl WorkerMetrics {
    fn add_duration(counter: &AtomicU64, duration: Duration) {
        counter.fetch_add(
            u64::try_from(duration.as_micros()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
    }

    fn record_queue_wait(&self, duration: Duration) {
        self.inner
            .queue_wait_observations
            .fetch_add(1, Ordering::Relaxed);
        Self::add_duration(&self.inner.queue_wait_micros, duration);
    }

    fn record_execution(&self, duration: Duration) {
        self.inner
            .execution_observations
            .fetch_add(1, Ordering::Relaxed);
        Self::add_duration(&self.inner.execution_micros, duration);
    }

    fn record_cancellation_started(&self) {
        self.inner
            .cancellation_requests
            .fetch_add(1, Ordering::Relaxed);
    }

    fn record_cancellation_stopped(&self, duration: Duration) {
        self.inner
            .cancellation_to_stop_observations
            .fetch_add(1, Ordering::Relaxed);
        Self::add_duration(&self.inner.cancellation_to_stop_micros, duration);
    }

    fn render_prometheus<E>(
        &self,
        state: &WorkerState<E>,
        engine_telemetry: Option<&RuntimeTelemetrySnapshot>,
    ) -> String {
        let active = state.config.max_active_invocations - state.capacity.available_permits();
        let retained_attempts = lock_recover(&state.attempts).records.len();
        let mut output = String::with_capacity(MAX_PROMETHEUS_RESPONSE_BYTES);
        macro_rules! metric {
            ($name:literal, $kind:literal, $help:literal, $value:expr) => {{
                let _ = writeln!(output, concat!("# HELP ", $name, " ", $help));
                let _ = writeln!(output, concat!("# TYPE ", $name, " ", $kind));
                let _ = writeln!(output, concat!($name, " {}"), $value);
            }};
        }
        metric!(
            "izwi_worker_active_invocations",
            "gauge",
            "Currently admitted invocations.",
            active
        );
        metric!(
            "izwi_worker_queued_invocations",
            "gauge",
            "Currently queued invocations.",
            0
        );
        metric!(
            "izwi_worker_retained_sessions",
            "gauge",
            "Currently retained stateful sessions.",
            0
        );
        metric!(
            "izwi_worker_retained_attempts",
            "gauge",
            "Bounded retained attempt records.",
            retained_attempts
        );
        metric!(
            "izwi_worker_draining",
            "gauge",
            "Whether new admission is disabled for drain.",
            u8::from(state.draining.load(Ordering::Relaxed))
        );
        metric!(
            "izwi_worker_admitted_total",
            "counter",
            "Runtime admissions accepted.",
            self.inner.admitted.load(Ordering::Relaxed)
        );
        metric!(
            "izwi_worker_rejected_total",
            "counter",
            "Requests rejected before acceptance.",
            self.inner.rejected.load(Ordering::Relaxed)
        );
        metric!(
            "izwi_worker_auth_rejections_total",
            "counter",
            "Private endpoint authentication rejections.",
            self.inner.auth_rejections.load(Ordering::Relaxed)
        );
        metric!(
            "izwi_worker_body_limit_rejections_total",
            "counter",
            "Private request body-size rejections.",
            self.inner.body_limit_rejections.load(Ordering::Relaxed)
        );
        metric!(
            "izwi_worker_admission_unknown_total",
            "counter",
            "Admission waits whose ownership outcome was initially unknown.",
            self.inner.admission_unknown.load(Ordering::Relaxed)
        );
        metric!(
            "izwi_worker_completed_total",
            "counter",
            "Invocations completed after confirmed teardown.",
            self.inner.completed.load(Ordering::Relaxed)
        );
        metric!(
            "izwi_worker_failed_total",
            "counter",
            "Invocations failed after confirmed teardown.",
            self.inner.failed.load(Ordering::Relaxed)
        );
        metric!(
            "izwi_worker_cancelled_total",
            "counter",
            "Invocations cancelled after confirmed teardown.",
            self.inner.cancelled.load(Ordering::Relaxed)
        );
        metric!(
            "izwi_worker_unconfirmed_teardown_total",
            "counter",
            "Invocations retaining capacity because teardown is unconfirmed.",
            self.inner.unconfirmed_teardown.load(Ordering::Relaxed)
        );
        metric!(
            "izwi_worker_queue_wait_observations_total",
            "counter",
            "Admission wait observations.",
            self.inner.queue_wait_observations.load(Ordering::Relaxed)
        );
        metric!(
            "izwi_worker_queue_wait_microseconds_total",
            "counter",
            "Accumulated runtime admission wait.",
            self.inner.queue_wait_micros.load(Ordering::Relaxed)
        );
        metric!(
            "izwi_worker_execution_observations_total",
            "counter",
            "Confirmed execution duration observations.",
            self.inner.execution_observations.load(Ordering::Relaxed)
        );
        metric!(
            "izwi_worker_execution_microseconds_total",
            "counter",
            "Accumulated admitted execution duration.",
            self.inner.execution_micros.load(Ordering::Relaxed)
        );
        metric!(
            "izwi_worker_cancellation_requests_total",
            "counter",
            "Executions asked to cancel.",
            self.inner.cancellation_requests.load(Ordering::Relaxed)
        );
        metric!(
            "izwi_worker_cancellation_to_stop_observations_total",
            "counter",
            "Confirmed cancellation-to-stop observations.",
            self.inner
                .cancellation_to_stop_observations
                .load(Ordering::Relaxed)
        );
        metric!(
            "izwi_worker_cancellation_to_stop_microseconds_total",
            "counter",
            "Accumulated cancellation-to-confirmed-stop duration.",
            self.inner
                .cancellation_to_stop_micros
                .load(Ordering::Relaxed)
        );
        metric!(
            "izwi_worker_event_delivery_failures_total",
            "counter",
            "Oversized or unavailable event deliveries.",
            self.inner.event_delivery_failures.load(Ordering::Relaxed)
        );
        if let Some(telemetry) = engine_telemetry {
            // Managed-KV serving evidence from the embedded engine (DS1.4
            // process-level surfacing; per-deployment attribution stays with
            // the engine snapshot). Names mirror the server's engine metrics.
            let counters = &telemetry.engine.kv_cache.counters;
            metric!(
                "izwi_engine_kv_cache_hits_total",
                "counter",
                "Managed-KV committed prefix hits.",
                counters.prefix_hits
            );
            metric!(
                "izwi_engine_kv_cache_misses_total",
                "counter",
                "Managed-KV committed prefix misses.",
                counters.prefix_misses
            );
            metric!(
                "izwi_engine_kv_cache_evictions_total",
                "counter",
                "Managed-KV committed prefix evictions.",
                counters.prefix_evictions
            );
            metric!(
                "izwi_engine_kv_cache_reused_tokens_total",
                "counter",
                "Tokens served from committed prefix state.",
                counters.reused_tokens
            );
            metric!(
                "izwi_engine_kv_cache_avoided_prefill_tokens_total",
                "counter",
                "Prefill tokens avoided by committed prefix reuse.",
                counters.avoided_prefill_tokens
            );
            metric!(
                "izwi_engine_tensor_snapshot_publishes_total",
                "counter",
                "Committed tensor snapshots published for cross-request fork.",
                counters.tensor_snapshot_publishes
            );
            metric!(
                "izwi_engine_tensor_snapshot_attaches_total",
                "counter",
                "Committed tensor snapshots attached by a fork.",
                counters.tensor_snapshot_attaches
            );
            metric!(
                "izwi_engine_tensor_snapshot_truncations_total",
                "counter",
                "Paged matches truncated back to snapshot-backed boundaries.",
                counters.tensor_snapshot_truncations
            );
            metric!(
                "izwi_engine_tensor_snapshot_evictions_total",
                "counter",
                "Committed tensor snapshots evicted from the snapshot index.",
                counters.tensor_snapshot_evictions
            );
            metric!(
                "izwi_engine_kv_cache_host_pages",
                "gauge",
                "Host-resident offloaded KV pages currently held by the DS4 host pool.",
                counters.kv_host_pages
            );
            metric!(
                "izwi_engine_kv_cache_demotions_total",
                "counter",
                "Pages demoted from device arenas to the DS4 host pool.",
                counters.demotions_total
            );
            metric!(
                "izwi_engine_kv_cache_promotions_total",
                "counter",
                "Pages promoted from the DS4 host pool back into device arenas.",
                counters.promotions_total
            );
        }
        debug_assert!(output.len() <= MAX_PROMETHEUS_RESPONSE_BYTES);
        output
    }
}

impl<E> WorkerState<E> {
    fn remove_record(table: &mut AttemptTable, attempt_id: &AttemptId) {
        table.records.remove(attempt_id);
        if let Some(index) = table.order.iter().position(|id| id == attempt_id) {
            table.order.remove(index);
        }
    }

    fn record_is_expired(record: &AttemptRecord, now: Instant) -> bool {
        record.evict_after.is_some_and(|deadline| deadline <= now)
    }

    fn evict_expired_record(table: &mut AttemptTable, now: Instant) -> bool {
        let Some(index) = table.order.iter().position(|attempt_id| {
            table
                .records
                .get(attempt_id)
                .and_then(|record| record.evict_after)
                .is_some_and(|deadline| deadline <= now)
        }) else {
            return false;
        };
        let evicted = table.order.remove(index).expect("known retention index");
        table.records.remove(&evicted);
        true
    }

    fn authenticate(&self, headers: &HeaderMap) -> bool {
        let bearer = headers
            .get(SERVICE_AUTHORIZATION_HEADER)
            .and_then(|value| value.to_str().ok());
        let credential = headers
            .get(SERVICE_CREDENTIAL_ID_HEADER)
            .and_then(|value| value.to_str().ok());
        let presented_token = bearer.and_then(|value| {
            value
                .strip_prefix(SERVICE_AUTH_SCHEME)
                .and_then(|value| value.strip_prefix(' '))
        });
        credential == Some(self.config.credentials.credential_id.as_str())
            && presented_token.is_some_and(|token| {
                self.config
                    .credentials
                    .bearer_token
                    .matches_presented(token)
            })
    }

    fn reserve_attempt(&self, request: &InvocationRequest) -> ReserveAttempt {
        let mut table = lock_recover(&self.attempts);
        Self::reserve_new_attempt(
            &mut table,
            AttemptIdentity::from(request),
            request.request_digest.clone(),
            self.config.max_retained_attempts,
        )
    }

    /// Realtime sessions reserve attempts in the same table with the same
    /// fencing identity model, so `query_attempt`/`cancel_attempt` over HTTP
    /// behave identically for them.
    fn reserve_realtime_session(&self, admit: &RealtimeSessionAdmit) -> ReserveAttempt {
        let mut table = lock_recover(&self.attempts);
        Self::reserve_new_attempt(
            &mut table,
            AttemptIdentity {
                request_id: admit.request_id.clone(),
                attempt_id: admit.attempt_id.clone(),
                tenant_id: admit.caller.tenant_id.clone(),
                caller_id: admit.caller.caller_id.clone(),
                incarnation_id: admit.expected_worker_incarnation.clone(),
                deployment_id: admit.deployment_id.clone(),
                model_generation: admit.expected_model_generation,
            },
            realtime::realtime_attempt_digest(admit),
            self.config.max_retained_attempts,
        )
    }

    fn reserve_new_attempt(
        table: &mut AttemptTable,
        identity: AttemptIdentity,
        digest: RequestDigest,
        max_retained_attempts: usize,
    ) -> ReserveAttempt {
        if table
            .records
            .get(&identity.attempt_id)
            .is_some_and(|record| Self::record_is_expired(record, Instant::now()))
        {
            Self::remove_record(table, &identity.attempt_id);
        }
        if let Some(existing) = table.records.get(&identity.attempt_id) {
            return if existing.identity == identity && existing.digest == digest {
                ReserveAttempt::AlreadyOwned
            } else {
                ReserveAttempt::Conflict
            };
        }
        while table.records.len() >= max_retained_attempts {
            if !Self::evict_expired_record(table, Instant::now()) {
                return ReserveAttempt::Full;
            }
        }
        table.order.push_back(identity.attempt_id.clone());
        table.records.insert(
            identity.attempt_id.clone(),
            AttemptRecord {
                identity,
                digest,
                state: AttemptState::Queued,
                last_sequence: None,
                cancel_requested: false,
                cancel: None,
                evict_after: None,
            },
        );
        ReserveAttempt::Reserved
    }

    fn remove_reservation(&self, attempt_id: &AttemptId) {
        let mut table = lock_recover(&self.attempts);
        Self::remove_record(&mut table, attempt_id);
    }

    fn install_admission(&self, attempt_id: &AttemptId, cancel: watch::Sender<bool>) -> bool {
        let mut table = lock_recover(&self.attempts);
        let record = table
            .records
            .get_mut(attempt_id)
            .expect("reserved attempt must exist through admission");
        record.state = AttemptState::Admitted;
        record.last_sequence = Some(0);
        record.cancel = Some(cancel);
        record.cancel_requested
    }

    fn update_attempt(&self, attempt_id: &AttemptId, state: AttemptState, sequence: Option<u64>) {
        let mut table = lock_recover(&self.attempts);
        if let Some(record) = table.records.get_mut(attempt_id) {
            record.state = state;
            if let Some(sequence) = sequence {
                record.last_sequence = Some(sequence);
            }
            if state.is_terminal() {
                record.cancel = None;
                record.evict_after = Instant::now().checked_add(self.config.attempt_retention);
            }
        }
    }

    fn mark_cancellation_requested(&self, attempt_id: &AttemptId) {
        let mut table = lock_recover(&self.attempts);
        if let Some(record) = table.records.get_mut(attempt_id) {
            record.cancel_requested = true;
            record.state = AttemptState::CancellationRequested;
        }
    }

    fn insert_cancel_tombstone(&self, identity: AttemptIdentity) {
        let mut table = lock_recover(&self.attempts);
        if table.records.contains_key(&identity.attempt_id) {
            return;
        }
        while table.records.len() >= self.config.max_retained_attempts {
            if !Self::evict_expired_record(&mut table, Instant::now()) {
                return;
            }
        }
        let attempt_id = identity.attempt_id.clone();
        table.order.push_back(attempt_id.clone());
        table.records.insert(
            attempt_id,
            AttemptRecord {
                identity,
                digest: RequestDigest::new("cancel-tombstone").expect("static identity"),
                state: AttemptState::CancellationRequested,
                last_sequence: None,
                cancel_requested: true,
                cancel: None,
                evict_after: Instant::now().checked_add(self.config.attempt_retention),
            },
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReserveAttempt {
    Reserved,
    AlreadyOwned,
    Conflict,
    Full,
}

struct AdmissionHandoff {
    result: Result<AdmittedInvocation, AdmissionFailure>,
    permit: OwnedSemaphorePermit,
    admission_guard: OwnedRwLockReadGuard<()>,
    execution_started_at: Instant,
}

#[derive(Default)]
struct AdmissionSlot {
    handoff: Option<AdmissionHandoff>,
    abandoned: bool,
}

/// Cloneable owner of the worker router and lifecycle state.
pub struct WorkerService<E> {
    state: Arc<WorkerState<E>>,
}

impl<E> Clone for WorkerService<E> {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

impl<E: InvocationExecutor> WorkerService<E> {
    pub fn new(config: WorkerConfig, executor: E) -> Result<Self, WorkerConfigError> {
        config.validate()?;
        let capacity = Arc::new(Semaphore::new(config.max_active_invocations));
        Ok(Self {
            state: Arc::new(WorkerState {
                config,
                executor: Arc::new(executor),
                capacity,
                admission_gate: Arc::new(RwLock::new(())),
                attempts: Mutex::new(AttemptTable::default()),
                status_sequence: AtomicU64::new(0),
                draining: AtomicBool::new(false),
                metrics: WorkerMetrics::default(),
            }),
        })
    }

    pub fn router(&self) -> Router {
        let max_request_bytes = self.state.config.max_request_bytes;
        Router::new()
            .route(WORKER_DESCRIPTOR_PATH, get(descriptor::<E>))
            .route(WORKER_STATUS_PATH, get(status::<E>))
            .route(WORKER_METRICS_PATH, get(metrics::<E>))
            .route(INVOCATIONS_PATH, post(invoke::<E>))
            .route(REALTIME_WS_PATH, get(realtime::realtime_socket::<E>))
            .route(
                &format!("{INVOCATIONS_PATH}/{{attempt_id}}"),
                get(query_attempt::<E>),
            )
            .route(
                &format!("{INVOCATIONS_PATH}/{{attempt_id}}/cancel"),
                post(cancel_attempt::<E>),
            )
            .layer(DefaultBodyLimit::max(max_request_bytes))
            .layer(middleware::from_fn_with_state(
                Arc::clone(&self.state),
                observe_worker_body_limit,
            ))
            .layer(middleware::from_fn(continue_trace))
            .with_state(Arc::clone(&self.state))
    }

    pub async fn begin_draining(&self) {
        let _gate = self.state.admission_gate.write().await;
        self.state.draining.store(true, Ordering::Release);
    }

    /// Request cooperative cancellation for every currently admitted attempt.
    /// Capacity remains held until each execution reports confirmed teardown.
    pub fn request_cancel_all(&self) -> usize {
        let cancellations = {
            let mut table = lock_recover(&self.state.attempts);
            table
                .records
                .values_mut()
                .filter_map(|record| {
                    if record.state.is_terminal() || record.cancel_requested {
                        return None;
                    }
                    record.cancel_requested = true;
                    record.state = AttemptState::CancellationRequested;
                    record.cancel.clone()
                })
                .collect::<Vec<_>>()
        };
        let requested = cancellations.len();
        for cancellation in cancellations {
            let _ = cancellation.send(true);
        }
        requested
    }

    pub fn active_invocations(&self) -> usize {
        self.state.config.max_active_invocations - self.state.capacity.available_permits()
    }

    pub fn runtime_health(&self) -> RuntimeHealth {
        self.state.executor.runtime_health()
    }
}

async fn observe_worker_body_limit<E: InvocationExecutor>(
    State(state): State<Arc<WorkerState<E>>>,
    request: AxumRequest,
    next: Next,
) -> Response {
    let response = next.run(request).await;
    if response.status() == StatusCode::PAYLOAD_TOO_LARGE {
        state
            .metrics
            .inner
            .body_limit_rejections
            .fetch_add(1, Ordering::Relaxed);
        state.metrics.inner.rejected.fetch_add(1, Ordering::Relaxed);
    }
    response
}

async fn descriptor<E: InvocationExecutor>(
    State(state): State<Arc<WorkerState<E>>>,
    headers: HeaderMap,
) -> Response {
    if !state.authenticate(&headers) {
        state
            .metrics
            .inner
            .auth_rejections
            .fetch_add(1, Ordering::Relaxed);
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let peer = peer_protocol_version(&headers);
    Json(state.config.descriptor.for_peer(peer)).into_response()
}

/// The protocol version a gateway declared in [`PROTOCOL_VERSION_HEADER`],
/// or [`UNDECLARED_PEER_VERSION`] for gateways from before the header.
fn peer_protocol_version(headers: &HeaderMap) -> SchemaVersion {
    headers
        .get(PROTOCOL_VERSION_HEADER)
        .and_then(|value| value.to_str().ok())
        .and_then(SchemaVersion::parse)
        .unwrap_or(UNDECLARED_PEER_VERSION)
}

/// Map managed-KV engine counters onto the deployment's optional routing
/// signals. Usage follows the engine's own Prometheus utilization ratio
/// (allocated over capacity pages); queries are hits plus misses because the
/// engine counts lookups as one or the other.
fn apply_kv_routing_signals(
    deployment: &mut LoadedDeployment,
    kv_cache: &ManagedKvRuntimeSnapshot,
) {
    let coordinator = &kv_cache.totals.coordinator;
    if coordinator.capacity_pages > 0 {
        deployment.kv_cache_usage_pct =
            Some((coordinator.allocated_pages as f64 / coordinator.capacity_pages as f64) * 100.0);
    }
    deployment.prefix_hits_total = Some(kv_cache.counters.prefix_hits);
    deployment.prefix_queries_total = Some(
        kv_cache
            .counters
            .prefix_hits
            .saturating_add(kv_cache.counters.prefix_misses),
    );
    deployment.prefix_evictions_total = Some(kv_cache.counters.prefix_evictions);
    // DS4 hierarchical offload signals: absent on engines without the feature
    // compiled, zero on engines with the feature but no host budget.
    deployment.kv_host_pages = Some(kv_cache.counters.kv_host_pages);
    deployment.kv_demotions_total = Some(kv_cache.counters.demotions_total);
    deployment.kv_promotions_total = Some(kv_cache.counters.promotions_total);
    deployment.kv_promotion_latency_avg_seconds = if kv_cache.counters.promotions_total > 0 {
        Some(
            kv_cache.counters.promotion_latency_ns_total as f64
                / kv_cache.counters.promotions_total as f64
                / 1e9,
        )
    } else {
        Some(0.0)
    };
}

async fn status<E: InvocationExecutor>(
    State(state): State<Arc<WorkerState<E>>>,
    headers: HeaderMap,
) -> Response {
    if !state.authenticate(&headers) {
        state
            .metrics
            .inner
            .auth_rejections
            .fetch_add(1, Ordering::Relaxed);
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let active = state.config.max_active_invocations - state.capacity.available_permits();
    let draining = state.draining.load(Ordering::Acquire);
    let health = state.executor.runtime_health();
    let mut deployment = state.config.deployment.clone();
    // A poisoned runtime rejects every invocation, so it must never be
    // advertised as ready: a permanent failure fails the whole process, and
    // a drain-recoverable poison fails only the deployment's readiness.
    let process_state = if health.is_failed() {
        WorkerProcessState::Failed
    } else if draining {
        WorkerProcessState::Draining
    } else {
        WorkerProcessState::Running
    };
    if !health.is_healthy() {
        deployment.readiness = ModelReadiness::Failed;
    } else if draining {
        deployment.readiness = ModelReadiness::Draining;
    }
    // Optional routing signals: populated only while the executor embeds an
    // engine that reports them. Absence stays "signal unavailable" for the
    // gateway, matching the additive protocol contract.
    let engine_telemetry = state.executor.runtime_telemetry().await;
    if let Some(telemetry) = engine_telemetry.as_ref() {
        apply_kv_routing_signals(&mut deployment, &telemetry.engine.kv_cache);
    }
    deployment.tokens_out_per_s_ema = state.metrics.inner.tokens_out_ema.load();
    // The worker's permit model prices one admission credit per concurrent
    // invocation, matching `outstanding_cost_units: active` below.
    deployment.observation_cost_units = Some(1);
    Json(WorkerStatus {
        schema_version: PROTOCOL_V1,
        worker_id: state.config.descriptor.worker_id.clone(),
        node_id: state.config.descriptor.node_id.clone(),
        incarnation_id: state.config.descriptor.incarnation_id.clone(),
        status_sequence: state.status_sequence.fetch_add(1, Ordering::Relaxed) + 1,
        process_state,
        deployments: vec![deployment],
        capacity: CapacitySnapshot {
            max_active_invocations: state.config.max_active_invocations as u32,
            active_invocations: active as u32,
            max_queued_invocations: 0,
            queued_invocations: 0,
            max_sessions: 0,
            reserved_sessions: 0,
            available_admission_credits: state.capacity.available_permits() as u32,
            outstanding_cost_units: active as u64,
        },
    })
    .into_response()
}

async fn metrics<E: InvocationExecutor>(
    State(state): State<Arc<WorkerState<E>>>,
    headers: HeaderMap,
) -> Response {
    if !state.authenticate(&headers) {
        state
            .metrics
            .inner
            .auth_rejections
            .fetch_add(1, Ordering::Relaxed);
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let engine_telemetry = state.executor.runtime_telemetry().await;
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        state
            .metrics
            .render_prometheus(&state, engine_telemetry.as_ref()),
    )
        .into_response()
}

/// The worker's JSON log line shape (`IZWI_LOG_FORMAT=json`), matching the
/// server's: event fields at the top level plus the current span and the
/// span list, so the W3C trace ids reach every line logged in a request.
pub fn json_log_layer<S, W>(writer: W) -> impl tracing_subscriber::Layer<S>
where
    S: tracing::Subscriber + for<'span> tracing_subscriber::registry::LookupSpan<'span>,
    W: for<'writer> tracing_subscriber::fmt::MakeWriter<'writer> + Send + Sync + 'static,
{
    tracing_subscriber::fmt::layer()
        .json()
        .flatten_event(true)
        .with_current_span(true)
        .with_span_list(true)
        .with_writer(writer)
}

/// Continue the caller's W3C trace. Every request runs in a span carrying
/// the trace id, this worker's span id and the caller's span id, so the
/// worker's logs join the gateway's under one trace id; a request without a
/// valid `traceparent` starts a trace.
async fn continue_trace(mut request: AxumRequest, next: Next) -> Response {
    let incoming = request
        .headers()
        .get(TRACEPARENT_HEADER)
        .and_then(|value| value.to_str().ok());
    let (trace, parent) = TraceContext::continue_from(incoming);
    let span = tracing::info_span!(
        "worker_request",
        method = %request.method(),
        path = %request.uri().path(),
        trace_id = %trace.trace_id(),
        span_id = %trace.span_id(),
        parent_span_id = tracing::field::Empty,
    );
    if let Some(parent) = parent {
        span.record("parent_span_id", trace::hex(&parent));
    }
    // Status polls are frequent; only work-carrying requests log at info.
    let work =
        request.method() == axum::http::Method::POST || request.uri().path() == REALTIME_WS_PATH;
    request.extensions_mut().insert(trace);
    let started = Instant::now();
    async move {
        let response = next.run(request).await;
        let status = response.status().as_u16();
        let latency_ms = started.elapsed().as_millis() as u64;
        if work {
            tracing::info!(status, latency_ms, "worker request completed");
        } else {
            tracing::debug!(status, latency_ms, "worker request completed");
        }
        response
    }
    .instrument(span)
    .await
}

async fn invoke<E: InvocationExecutor>(
    State(state): State<Arc<WorkerState<E>>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let received_at = Instant::now();
    if !state.authenticate(&headers) {
        state
            .metrics
            .inner
            .auth_rejections
            .fetch_add(1, Ordering::Relaxed);
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let request: InvocationRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(_) => {
            state.metrics.inner.rejected.fetch_add(1, Ordering::Relaxed);
            return StatusCode::BAD_REQUEST.into_response();
        }
    };
    if request.schema_version.major != PROTOCOL_V1.major {
        return counted_rejection(
            &state,
            &request,
            StatusCode::CONFLICT,
            RejectionCode::UnsupportedProtocolVersion,
            "unsupported protocol major version",
            None,
        );
    }
    if let Err(error) = request.validate() {
        return counted_rejection(
            &state,
            &request,
            StatusCode::BAD_REQUEST,
            RejectionCode::InvalidRequest,
            error.to_string(),
            None,
        );
    }
    // A control this worker did not advertise is refused, never ignored.
    let InvocationInput::Chat { parameters, .. } = &request.input;
    if let Some(parameter) = parameters.first_unsupported(&state.config.descriptor.features) {
        return counted_rejection(
            &state,
            &request,
            StatusCode::BAD_REQUEST,
            RejectionCode::InvalidRequest,
            format!("this worker does not support `{parameter}`"),
            None,
        );
    }
    if !request
        .caller
        .permitted_actions
        .contains(&PermittedAction::Invoke)
    {
        return counted_rejection(
            &state,
            &request,
            StatusCode::FORBIDDEN,
            RejectionCode::PolicyDenied,
            "gateway caller context does not permit invocation",
            None,
        );
    }
    if state.draining.load(Ordering::Acquire) {
        return counted_rejection(
            &state,
            &request,
            StatusCode::SERVICE_UNAVAILABLE,
            RejectionCode::WorkerDraining,
            "worker is draining",
            None,
        );
    }
    if request.expected_worker_incarnation != state.config.descriptor.incarnation_id {
        return counted_rejection(
            &state,
            &request,
            StatusCode::CONFLICT,
            RejectionCode::WrongWorkerIncarnation,
            "worker incarnation changed",
            None,
        );
    }
    if request.deployment_id != state.config.deployment.deployment_id {
        return counted_rejection(
            &state,
            &request,
            StatusCode::NOT_FOUND,
            RejectionCode::UnknownDeployment,
            "deployment is not loaded",
            None,
        );
    }
    if request.expected_model_generation != state.config.deployment.model_generation {
        return counted_rejection(
            &state,
            &request,
            StatusCode::CONFLICT,
            RejectionCode::WrongModelGeneration,
            "model generation changed",
            None,
        );
    }
    if request.task != TaskKind::Chat || request.input.task() != TaskKind::Chat {
        return counted_rejection(
            &state,
            &request,
            StatusCode::UNPROCESSABLE_ENTITY,
            RejectionCode::IncompatibleTask,
            "task is incompatible with deployment",
            None,
        );
    }
    if request.service_class == ServiceClass::Realtime
        && !state.config.deployment.capability.realtime
    {
        return counted_rejection(
            &state,
            &request,
            StatusCode::UNPROCESSABLE_ENTITY,
            RejectionCode::IncompatibleTask,
            "deployment does not support realtime execution",
            None,
        );
    }
    if state.config.deployment.readiness != ModelReadiness::Ready {
        return counted_rejection(
            &state,
            &request,
            StatusCode::SERVICE_UNAVAILABLE,
            RejectionCode::ModelNotReady,
            "deployment is not ready",
            None,
        );
    }
    let input_bytes = match &request.input {
        InvocationInput::Chat { input, .. } => input
            .messages
            .iter()
            .try_fold(0u64, |total, message| {
                total.checked_add(message.content.len() as u64)
            })
            .unwrap_or(u64::MAX),
    };
    if input_bytes > state.config.deployment.capability.max_input_bytes
        || usize::try_from(request.output_limits.max_bytes).map_or(true, |limit| {
            limit.saturating_add(EVENT_ENVELOPE_ALLOWANCE) > state.config.max_event_bytes
        })
        || state
            .config
            .deployment
            .capability
            .max_output_tokens
            .is_some_and(|limit| request.output_limits.max_tokens > limit)
        || !state
            .config
            .deployment
            .capability
            .output_formats
            .contains(&request.requested_output_format)
    {
        return counted_rejection(
            &state,
            &request,
            StatusCode::BAD_REQUEST,
            RejectionCode::InvalidRequest,
            "requested input or output limits exceed worker capability",
            None,
        );
    }

    // Serialize the transition from accepting work to draining with runtime
    // admission. A holder either observes draining or transfers authoritative
    // ownership before `begin_draining` can return.
    let admission_guard = Arc::clone(&state.admission_gate).read_owned().await;
    if state.draining.load(Ordering::Acquire) {
        return counted_rejection(
            &state,
            &request,
            StatusCode::SERVICE_UNAVAILABLE,
            RejectionCode::WorkerDraining,
            "worker is draining",
            None,
        );
    }

    match state.reserve_attempt(&request) {
        ReserveAttempt::Reserved => {}
        ReserveAttempt::AlreadyOwned => {
            state.metrics.inner.rejected.fetch_add(1, Ordering::Relaxed);
            return (
                StatusCode::CONFLICT,
                "attempt is already owned; acceptance is unknown, query or cancel it",
            )
                .into_response();
        }
        ReserveAttempt::Conflict => {
            return counted_rejection(
                &state,
                &request,
                StatusCode::CONFLICT,
                RejectionCode::DuplicateAttemptConflict,
                "attempt identity was reused with different content",
                None,
            )
        }
        ReserveAttempt::Full => {
            return counted_rejection(
                &state,
                &request,
                StatusCode::TOO_MANY_REQUESTS,
                RejectionCode::CapacityExhausted,
                "attempt retention is exhausted",
                None,
            )
        }
    }

    let permit = match Arc::clone(&state.capacity).try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            state.remove_reservation(&request.attempt_id);
            return counted_rejection(
                &state,
                &request,
                StatusCode::TOO_MANY_REQUESTS,
                RejectionCode::CapacityExhausted,
                "worker capacity is exhausted",
                Some(25),
            );
        }
    };

    let remaining_before_admission =
        Duration::from_millis(request.remaining_time_ms).saturating_sub(received_at.elapsed());
    let admission_budget =
        remaining_before_admission.min(Duration::from_millis(request.max_queue_wait_ms));
    let admission_slot = Arc::new(Mutex::new(AdmissionSlot::default()));
    let admission_ready = Arc::new(tokio::sync::Notify::new());
    let admission_wait = admission_ready.notified();
    tokio::pin!(admission_wait);
    let admission_state = Arc::clone(&state);
    let admission_request = request.clone();
    let slot_for_admission = Arc::clone(&admission_slot);
    let ready_for_admission = Arc::clone(&admission_ready);
    tokio::spawn(
        async move {
            let result = admission_state.executor.admit(&admission_request).await;
            admission_state
                .metrics
                .record_queue_wait(received_at.elapsed());
            if result.is_ok() {
                admission_state
                    .metrics
                    .inner
                    .admitted
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                admission_state
                    .metrics
                    .inner
                    .rejected
                    .fetch_add(1, Ordering::Relaxed);
            }
            let handoff = AdmissionHandoff {
                result,
                permit,
                admission_guard,
                execution_started_at: Instant::now(),
            };
            let abandoned_handoff = {
                let mut slot = lock_recover(&slot_for_admission);
                if slot.abandoned {
                    Some(handoff)
                } else {
                    slot.handoff = Some(handoff);
                    None
                }
            };
            ready_for_admission.notify_one();
            if let Some(handoff) = abandoned_handoff {
                settle_abandoned_admission(admission_state, admission_request, handoff).await;
            }
        }
        .in_current_span(),
    );
    let handoff = match tokio::time::timeout(admission_budget, &mut admission_wait).await {
        Ok(()) => lock_recover(&admission_slot)
            .handoff
            .take()
            .expect("admission notification carries ownership"),
        Err(_) => {
            // Resolve the timeout/completion race while holding the slot. If
            // admission already published ownership, it wins. Otherwise mark
            // the receiver abandoned before returning the uncertain response.
            let completed = {
                let mut slot = lock_recover(&admission_slot);
                if let Some(handoff) = slot.handoff.take() {
                    Some(handoff)
                } else {
                    slot.abandoned = true;
                    None
                }
            };
            if let Some(handoff) = completed {
                handoff
            } else {
                // The runtime may already own an exact Engine session. Continue
                // admission under the permit; the detached path cancels and fences
                // any later success. This is intentionally not a typed rejection.
                state.mark_cancellation_requested(&request.attempt_id);
                state
                    .metrics
                    .inner
                    .admission_unknown
                    .fetch_add(1, Ordering::Relaxed);
                return (
                    StatusCode::GATEWAY_TIMEOUT,
                    Json(InvocationAdmissionUnknown {
                        schema_version: PROTOCOL_V1,
                        request_id: request.request_id.clone(),
                        attempt_id: request.attempt_id.clone(),
                        message: "runtime admission outcome is unknown; query or cancel this \
                                  attempt"
                            .into(),
                    }),
                )
                    .into_response();
            }
        }
    };
    let admitted = match handoff.result {
        Ok(admitted) => admitted,
        Err(failure) => {
            drop(handoff.admission_guard);
            drop(handoff.permit);
            state.remove_reservation(&request.attempt_id);
            return rejection(
                &request,
                rejection_status(failure.code),
                failure.code,
                bounded_message(failure.message),
                failure.retry_after_ms,
            );
        }
    };
    let permit = handoff.permit;
    let admission_guard = handoff.admission_guard;
    let execution_started_at = handoff.execution_started_at;
    let execution = admitted.into_execution();
    let (cancel_tx, cancel_rx) = watch::channel(false);
    let cancel_was_requested = state.install_admission(&request.attempt_id, cancel_tx.clone());
    if cancel_was_requested {
        let _ = cancel_tx.send(true);
    }

    let (tx, mut rx) = mpsc::channel::<Bytes>(state.config.event_channel_capacity);
    let remaining_time =
        Duration::from_millis(request.remaining_time_ms).saturating_sub(received_at.elapsed());
    tokio::spawn(
        run_invocation(
            Arc::clone(&state),
            request,
            permit,
            execution,
            cancel_rx,
            remaining_time,
            execution_started_at,
            tx,
        )
        .in_current_span(),
    );
    drop(admission_guard);
    let output = stream! {
        while let Some(bytes) = rx.recv().await {
            yield Ok::<Bytes, Infallible>(bytes);
        }
    };
    Response::builder()
        .status(StatusCode::OK)
        .header(axum::http::header::CONTENT_TYPE, NDJSON_MEDIA_TYPE)
        .body(Body::from_stream(output))
        .expect("static worker response")
}

// Streaming invocation handler: every parameter is a distinct admitted-run
// handle; grouping them would only add indirection in the hot serving path.
#[allow(clippy::too_many_arguments)]
async fn run_invocation<E: InvocationExecutor>(
    state: Arc<WorkerState<E>>,
    request: InvocationRequest,
    permit: OwnedSemaphorePermit,
    mut execution: Box<dyn AdmittedExecution>,
    mut cancel: watch::Receiver<bool>,
    remaining_time: Duration,
    execution_started_at: Instant,
    tx: mpsc::Sender<Bytes>,
) {
    let accepted = InvocationEvent {
        schema_version: PROTOCOL_V1,
        request_id: request.request_id.clone(),
        attempt_id: request.attempt_id.clone(),
        sequence: 0,
        event: InvocationEventKind::Accepted {
            worker_id: state.config.descriptor.worker_id.clone(),
            node_id: state.config.descriptor.node_id.clone(),
            incarnation_id: state.config.descriptor.incarnation_id.clone(),
            deployment_id: state.config.deployment.deployment_id.clone(),
            model_generation: state.config.deployment.model_generation,
        },
    };
    let accepted_delivered = matches!(
        try_send_event(&state.config, &state.metrics, &tx, accepted),
        EventSendResult::Sent
    );
    state.update_attempt(&request.attempt_id, AttemptState::Running, Some(0));

    let deadline = tokio::time::sleep(remaining_time);
    tokio::pin!(deadline);
    let mut sequence = 1;
    let mut output_bytes = 0usize;
    let mut timed_out = false;
    let mut cancellation_requested = !accepted_delivered || *cancel.borrow();
    let mut cancellation_started_at = None;
    let mut terminal = None;
    let mut outbox = TextOutbox::new(state.config.slow_consumer_timeout);
    let mut consumer_stalled = false;
    if cancellation_requested {
        mark_execution_cancellation(&state.metrics, &mut cancellation_started_at);
        execution.request_cancel();
        state.update_attempt(&request.attempt_id, AttemptState::ExecutionStopping, None);
    }

    while terminal.is_none() && !cancellation_requested {
        let stall_deadline = outbox.stall_deadline();
        // Every arm yields the delivery result of any text it emitted; a
        // failed delivery stops the invocation below.
        let delivery = tokio::select! {
            biased;
            () = &mut deadline, if !timed_out => {
                timed_out = true;
                cancellation_requested = true;
                mark_execution_cancellation(&state.metrics, &mut cancellation_started_at);
                execution.request_cancel();
                state.update_attempt(
                    &request.attempt_id,
                    AttemptState::ExecutionStopping,
                    None,
                );
                EventSendResult::Sent
            }
            () = tx.closed(), if !cancellation_requested => {
                cancellation_requested = true;
                mark_execution_cancellation(&state.metrics, &mut cancellation_started_at);
                execution.request_cancel();
                state.update_attempt(
                    &request.attempt_id,
                    AttemptState::ExecutionStopping,
                    None,
                );
                EventSendResult::Sent
            }
            changed = cancel.changed(), if !cancellation_requested => {
                if changed.is_ok() && *cancel.borrow() {
                    cancellation_requested = true;
                    mark_execution_cancellation(&state.metrics, &mut cancellation_started_at);
                    execution.request_cancel();
                    state.update_attempt(
                        &request.attempt_id,
                        AttemptState::ExecutionStopping,
                        None,
                    );
                }
                EventSendResult::Sent
            }
            _ = async { tx.reserve().await.is_ok() }, if outbox.has_pending() => {
                outbox.flush(&state, &request, &tx, &mut sequence)
            }
            () = async {
                match stall_deadline {
                    Some(deadline) => tokio::time::sleep_until(deadline).await,
                    None => std::future::pending().await,
                }
            }, if stall_deadline.is_some() => {
                consumer_stalled = true;
                EventSendResult::Unavailable
            }
            event = execution.next_event() => {
                match event {
                    Some(ExecutionEvent::TextDelta { text, logprobs })
                        if !cancellation_requested =>
                    {
                        output_bytes = output_bytes.saturating_add(text.len());
                        if output_bytes as u64 > request.output_limits.max_bytes {
                            terminal = Some(TerminalEvent::Failed(ExecutionFailure {
                                code: InvocationErrorCode::OutputLimitExceeded,
                                message: "runtime output exceeded the requested byte limit".into(),
                            }));
                            cancellation_requested = true;
                            mark_execution_cancellation(&state.metrics, &mut cancellation_started_at);
                            execution.request_cancel();
                            state.update_attempt(
                                &request.attempt_id,
                                AttemptState::ExecutionStopping,
                                None,
                            );
                            continue;
                        }
                        outbox.offer(&state, &request, &tx, &mut sequence, text, logprobs)
                    }
                    Some(ExecutionEvent::TextDelta { .. }) => EventSendResult::Sent,
                    Some(ExecutionEvent::Completed {
                        text,
                        finish_reason,
                        input_tokens,
                        output_tokens,
                        cached_input_tokens,
                    }) => {
                        let mut delivery = EventSendResult::Sent;
                        if let Some(text) = text.filter(|text| !text.is_empty()) {
                            output_bytes = output_bytes.saturating_add(text.len());
                            if output_bytes as u64 > request.output_limits.max_bytes {
                                terminal = Some(TerminalEvent::Failed(ExecutionFailure {
                                    code: InvocationErrorCode::OutputLimitExceeded,
                                    message: "runtime output exceeded the requested byte limit".into(),
                                }));
                                mark_execution_cancellation(&state.metrics, &mut cancellation_started_at);
                                execution.request_cancel();
                                continue;
                            }
                            delivery =
                                outbox.offer(&state, &request, &tx, &mut sequence, text, Vec::new());
                        }
                        state
                            .metrics
                            .inner
                            .tokens_out_ema
                            .record(output_tokens, execution_started_at.elapsed());
                        if delivery == EventSendResult::Sent {
                            terminal = Some(TerminalEvent::Completed {
                                finish_reason,
                                usage: Usage {
                                    input_tokens,
                                    output_tokens,
                                    cached_tokens: cached_input_tokens,
                                },
                            });
                        }
                        delivery
                    }
                    Some(ExecutionEvent::Failed(failure)) => {
                        terminal = Some(TerminalEvent::Failed(failure));
                        EventSendResult::Sent
                    }
                    None => break,
                }
            }
        };
        match delivery {
            EventSendResult::Sent => {}
            EventSendResult::Oversized => {
                terminal = Some(TerminalEvent::Failed(ExecutionFailure {
                    code: InvocationErrorCode::OutputLimitExceeded,
                    message: "encoded output event exceeded the worker limit".into(),
                }));
                cancellation_requested = true;
                mark_execution_cancellation(&state.metrics, &mut cancellation_started_at);
                execution.request_cancel();
            }
            // The reader is gone, stalled past the slow-consumer timeout, or
            // owes more text than one event can carry.
            EventSendResult::Unavailable => {
                cancellation_requested = true;
                mark_execution_cancellation(&state.metrics, &mut cancellation_started_at);
                execution.request_cancel();
            }
        }
    }

    // A terminal runtime event and a deadline are not teardown proof. Retain
    // worker capacity until the admitted runtime owner confirms cleanup.
    let teardown = execution.wait_for_teardown().await;
    if teardown == ExecutionTeardown::Unconfirmed {
        state
            .metrics
            .inner
            .unconfirmed_teardown
            .fetch_add(1, Ordering::Relaxed);
        state.update_attempt(&request.attempt_id, AttemptState::ExecutionStopping, None);
        // Fail closed: losing capacity is safer than advertising a credit
        // while native work may still exist in this incarnation.
        std::mem::forget(permit);
        return;
    }
    state
        .metrics
        .record_execution(execution_started_at.elapsed());
    if let Some(started_at) = cancellation_started_at {
        state
            .metrics
            .record_cancellation_stopped(started_at.elapsed());
    }
    let terminal = if timed_out {
        TerminalEvent::Failed(ExecutionFailure {
            code: InvocationErrorCode::DeadlineExceeded,
            message: "invocation deadline elapsed; execution teardown is confirmed".into(),
        })
    } else if cancellation_requested || teardown == ExecutionTeardown::Cancelled {
        TerminalEvent::Cancelled
    } else if teardown == ExecutionTeardown::Failed {
        match terminal {
            Some(TerminalEvent::Failed(failure)) => TerminalEvent::Failed(failure),
            _ => TerminalEvent::Failed(ExecutionFailure {
                code: InvocationErrorCode::ExecutionFailed,
                message: "runtime execution ended without successful teardown".into(),
            }),
        }
    } else {
        terminal.unwrap_or_else(|| {
            TerminalEvent::Failed(ExecutionFailure {
                code: InvocationErrorCode::Internal,
                message: "runtime execution ended without a terminal event".into(),
            })
        })
    };
    // A reader that already stalled past its budget gets no further wait.
    let publish_deadline = if consumer_stalled {
        Instant::now()
    } else {
        outbox.publish_deadline()
    };
    publish_terminal(
        &state,
        &request,
        &tx,
        &mut outbox,
        sequence,
        terminal,
        publish_deadline,
    )
    .await;
}

async fn settle_abandoned_admission<E: InvocationExecutor>(
    state: Arc<WorkerState<E>>,
    request: InvocationRequest,
    handoff: AdmissionHandoff,
) {
    let AdmissionHandoff {
        result,
        permit,
        admission_guard,
        execution_started_at,
    } = handoff;
    let Ok(admitted) = result else {
        state.remove_reservation(&request.attempt_id);
        drop(admission_guard);
        return;
    };
    let execution = admitted.into_execution();
    let cancellation_started_at = Instant::now();
    state.metrics.record_cancellation_started();
    execution.request_cancel();
    state.update_attempt(&request.attempt_id, AttemptState::ExecutionStopping, None);
    drop(admission_guard);
    match execution.wait_for_teardown().await {
        ExecutionTeardown::Completed => {
            state
                .metrics
                .inner
                .completed
                .fetch_add(1, Ordering::Relaxed);
            state.update_attempt(&request.attempt_id, AttemptState::Completed, None)
        }
        ExecutionTeardown::Cancelled => {
            state
                .metrics
                .inner
                .cancelled
                .fetch_add(1, Ordering::Relaxed);
            state.update_attempt(&request.attempt_id, AttemptState::Cancelled, None)
        }
        ExecutionTeardown::Failed => {
            state.metrics.inner.failed.fetch_add(1, Ordering::Relaxed);
            state.update_attempt(&request.attempt_id, AttemptState::Failed, None)
        }
        ExecutionTeardown::Unconfirmed => {
            state
                .metrics
                .inner
                .unconfirmed_teardown
                .fetch_add(1, Ordering::Relaxed);
            std::mem::forget(permit);
            return;
        }
    }
    state
        .metrics
        .record_execution(execution_started_at.elapsed());
    state
        .metrics
        .record_cancellation_stopped(cancellation_started_at.elapsed());
    drop(permit);
}

fn mark_execution_cancellation(metrics: &WorkerMetrics, started_at: &mut Option<Instant>) {
    if started_at.is_none() {
        *started_at = Some(Instant::now());
        metrics.record_cancellation_started();
    }
}

/// DS9.3: map core logprob entries onto the protocol wire shape.
fn protocol_logprobs(
    entries: &[izwi_core::engine::TokenLogprob],
) -> Option<Vec<izwi_serving_protocol::TokenLogprob>> {
    if entries.is_empty() {
        return None;
    }
    Some(
        entries
            .iter()
            .map(|entry| izwi_serving_protocol::TokenLogprob {
                token: entry.token.clone(),
                logprob: entry.logprob,
                bytes: entry.bytes.clone(),
                top_logprobs: entry
                    .top_logprobs
                    .iter()
                    .map(|top| izwi_serving_protocol::TopTokenLogprob {
                        token: top.token.clone(),
                        logprob: top.logprob,
                        bytes: top.bytes.clone(),
                    })
                    .collect(),
            })
            .collect(),
    )
}

/// Text deltas on their way to the NDJSON reader. While the reader lags (the
/// bounded event channel is full), deltas and their logprobs merge, in
/// order, into one pending delta that is flushed as a single event when the
/// reader frees a slot. A slow reader therefore costs bytes, not queued
/// events, and only a reader stalled past the slow-consumer timeout, or one
/// owing more text than one event can carry, stops the invocation.
struct TextOutbox {
    pending: Option<PendingDelta>,
    slow_consumer_timeout: Duration,
}

struct PendingDelta {
    text: String,
    logprobs: Vec<izwi_core::engine::TokenLogprob>,
    /// Upper bound on the encoded size of the merged delta.
    encoded_bytes: usize,
    since: Instant,
}

impl TextOutbox {
    fn new(slow_consumer_timeout: Duration) -> Self {
        Self {
            pending: None,
            slow_consumer_timeout,
        }
    }

    fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// When a reader that has not accepted the pending delta is stalled.
    fn stall_deadline(&self) -> Option<Instant> {
        self.pending
            .as_ref()
            .map(|pending| pending.since + self.slow_consumer_timeout)
    }

    /// How long the terminal event may wait for the reader.
    fn publish_deadline(&self) -> Instant {
        self.stall_deadline()
            .unwrap_or_else(|| Instant::now() + self.slow_consumer_timeout)
    }

    /// Send a delta now, or merge it into the pending delta while the reader
    /// lags. `run_invocation` is the channel's only sender, so free capacity
    /// observed here is still free when the event is sent.
    fn offer<E>(
        &mut self,
        state: &WorkerState<E>,
        request: &InvocationRequest,
        tx: &mpsc::Sender<Bytes>,
        sequence: &mut u64,
        text: String,
        logprobs: Vec<izwi_core::engine::TokenLogprob>,
    ) -> EventSendResult {
        if self.pending.is_none() && tx.capacity() > 0 {
            return send_text_delta(state, request, tx, sequence, text, logprobs);
        }
        if tx.is_closed() {
            return EventSendResult::Unavailable;
        }
        let encoded_bytes = encoded_delta_bytes(&text, &logprobs);
        let pending = self.pending.get_or_insert_with(|| PendingDelta {
            text: String::new(),
            logprobs: Vec::new(),
            encoded_bytes: 0,
            since: Instant::now(),
        });
        pending.text.push_str(&text);
        pending.logprobs.extend(logprobs);
        pending.encoded_bytes = pending.encoded_bytes.saturating_add(encoded_bytes);
        if pending
            .encoded_bytes
            .saturating_add(EVENT_ENVELOPE_ALLOWANCE)
            > state.config.max_event_bytes
        {
            state
                .metrics
                .inner
                .event_delivery_failures
                .fetch_add(1, Ordering::Relaxed);
            return EventSendResult::Unavailable;
        }
        EventSendResult::Sent
    }

    /// Send the pending delta if the reader has room; otherwise keep it.
    fn flush<E>(
        &mut self,
        state: &WorkerState<E>,
        request: &InvocationRequest,
        tx: &mpsc::Sender<Bytes>,
        sequence: &mut u64,
    ) -> EventSendResult {
        if tx.capacity() == 0 {
            return EventSendResult::Unavailable;
        }
        match self.pending.take() {
            Some(pending) => {
                send_text_delta(state, request, tx, sequence, pending.text, pending.logprobs)
            }
            None => EventSendResult::Sent,
        }
    }
}

/// Encoded size of one delta's payload, an upper bound on what it adds to a
/// merged delta.
fn encoded_delta_bytes(text: &str, logprobs: &[izwi_core::engine::TokenLogprob]) -> usize {
    serde_json::to_vec(&InvocationEventKind::TextDelta {
        text: text.to_owned(),
        logprobs: protocol_logprobs(logprobs),
    })
    .map_or(usize::MAX, |encoded| encoded.len())
}

fn send_text_delta<E>(
    state: &WorkerState<E>,
    request: &InvocationRequest,
    tx: &mpsc::Sender<Bytes>,
    sequence: &mut u64,
    text: String,
    logprobs: Vec<izwi_core::engine::TokenLogprob>,
) -> EventSendResult {
    let event = InvocationEvent {
        schema_version: PROTOCOL_V1,
        request_id: request.request_id.clone(),
        attempt_id: request.attempt_id.clone(),
        sequence: *sequence,
        event: InvocationEventKind::TextDelta {
            text,
            logprobs: protocol_logprobs(&logprobs),
        },
    };
    let result = try_send_event(&state.config, &state.metrics, tx, event);
    if result == EventSendResult::Sent {
        state.update_attempt(&request.attempt_id, AttemptState::Running, Some(*sequence));
        *sequence = sequence.saturating_add(1);
    }
    result
}

enum TerminalEvent {
    Completed {
        finish_reason: FinishReason,
        usage: Usage,
    },
    Cancelled,
    Failed(ExecutionFailure),
}

/// Publish the terminal event after any pending text, waiting for the
/// reader until `deadline`. A terminal event is never dropped just because
/// the channel is momentarily full, and never overtakes text.
async fn publish_terminal<E>(
    state: &WorkerState<E>,
    request: &InvocationRequest,
    tx: &mpsc::Sender<Bytes>,
    outbox: &mut TextOutbox,
    mut sequence: u64,
    terminal: TerminalEvent,
    deadline: Instant,
) {
    if outbox.has_pending() {
        let _ = tokio::time::timeout_at(deadline, tx.reserve()).await;
        let _ = outbox.flush(state, request, tx, &mut sequence);
    }
    let (attempt_state, event) = match terminal {
        TerminalEvent::Completed {
            finish_reason,
            usage,
        } => {
            state
                .metrics
                .inner
                .completed
                .fetch_add(1, Ordering::Relaxed);
            (
                AttemptState::Completed,
                InvocationEventKind::Completed {
                    finish_reason,
                    usage: Some(usage),
                },
            )
        }
        TerminalEvent::Cancelled => {
            state
                .metrics
                .inner
                .cancelled
                .fetch_add(1, Ordering::Relaxed);
            (
                AttemptState::Cancelled,
                InvocationEventKind::Cancelled {
                    reason: Some("requested".into()),
                },
            )
        }
        TerminalEvent::Failed(failure) => {
            state.metrics.inner.failed.fetch_add(1, Ordering::Relaxed);
            (
                AttemptState::Failed,
                InvocationEventKind::Error {
                    code: failure.code,
                    message: bounded_message(failure.message),
                },
            )
        }
    };
    let event = InvocationEvent {
        schema_version: PROTOCOL_V1,
        request_id: request.request_id.clone(),
        attempt_id: request.attempt_id.clone(),
        sequence,
        event,
    };
    let _ = tokio::time::timeout_at(deadline, tx.reserve()).await;
    let published = !outbox.has_pending()
        && matches!(
            try_send_event(&state.config, &state.metrics, tx, event),
            EventSendResult::Sent
        );
    state.update_attempt(
        &request.attempt_id,
        attempt_state,
        published.then_some(sequence),
    );
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventSendResult {
    Sent,
    Oversized,
    Unavailable,
}

fn try_send_event(
    config: &WorkerConfig,
    metrics: &WorkerMetrics,
    tx: &mpsc::Sender<Bytes>,
    event: InvocationEvent,
) -> EventSendResult {
    let Ok(mut encoded) = serde_json::to_vec(&event) else {
        metrics
            .inner
            .event_delivery_failures
            .fetch_add(1, Ordering::Relaxed);
        return EventSendResult::Oversized;
    };
    if encoded.len().saturating_add(1) > config.max_event_bytes {
        metrics
            .inner
            .event_delivery_failures
            .fetch_add(1, Ordering::Relaxed);
        return EventSendResult::Oversized;
    }
    encoded.push(b'\n');
    match tx.try_send(Bytes::from(encoded)) {
        Ok(()) => EventSendResult::Sent,
        Err(_) => {
            metrics
                .inner
                .event_delivery_failures
                .fetch_add(1, Ordering::Relaxed);
            EventSendResult::Unavailable
        }
    }
}

async fn query_attempt<E: InvocationExecutor>(
    State(state): State<Arc<WorkerState<E>>>,
    Path(raw_attempt_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    if !state.authenticate(&headers) {
        state
            .metrics
            .inner
            .auth_rejections
            .fetch_add(1, Ordering::Relaxed);
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let Ok(attempt_id) = AttemptId::new(raw_attempt_id) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    let mut table = lock_recover(&state.attempts);
    let Some(record) = table.records.get(&attempt_id) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    if WorkerState::<E>::record_is_expired(record, Instant::now()) {
        let identity = record.identity.clone();
        WorkerState::<E>::remove_record(&mut table, &attempt_id);
        return (
            StatusCode::GONE,
            Json(AttemptQueryResponse {
                schema_version: PROTOCOL_V1,
                worker_id: state.config.descriptor.worker_id.clone(),
                identity,
                state: AttemptState::Expired,
                last_sequence: None,
            }),
        )
            .into_response();
    }
    Json(AttemptQueryResponse {
        schema_version: PROTOCOL_V1,
        worker_id: state.config.descriptor.worker_id.clone(),
        identity: record.identity.clone(),
        state: record.state,
        last_sequence: record.last_sequence,
    })
    .into_response()
}

async fn cancel_attempt<E: InvocationExecutor>(
    State(state): State<Arc<WorkerState<E>>>,
    Path(attempt_id): Path<String>,
    headers: HeaderMap,
    Json(request): Json<CancelAttemptRequest>,
) -> Response {
    if !state.authenticate(&headers) {
        state
            .metrics
            .inner
            .auth_rejections
            .fetch_add(1, Ordering::Relaxed);
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let Ok(path_attempt_id) = AttemptId::new(attempt_id) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    if request.schema_version.major != PROTOCOL_V1.major
        || path_attempt_id != request.identity.attempt_id
        || request.identity.incarnation_id != state.config.descriptor.incarnation_id
    {
        return StatusCode::CONFLICT.into_response();
    }
    let (disposition, cancel) = {
        let mut table = lock_recover(&state.attempts);
        let expired = table
            .records
            .get(&request.identity.attempt_id)
            .filter(|record| record.identity == request.identity)
            .is_some_and(|record| WorkerState::<E>::record_is_expired(record, Instant::now()));
        if expired {
            WorkerState::<E>::remove_record(&mut table, &request.identity.attempt_id);
            return Json(CancelAttemptResponse {
                schema_version: PROTOCOL_V1,
                worker_id: state.config.descriptor.worker_id.clone(),
                identity: request.identity,
                disposition: CancelDisposition::Expired,
            })
            .into_response();
        }
        let Some(record) = table
            .records
            .get_mut(&request.identity.attempt_id)
            .filter(|record| record.identity == request.identity)
        else {
            drop(table);
            state.insert_cancel_tombstone(request.identity.clone());
            return Json(CancelAttemptResponse {
                schema_version: PROTOCOL_V1,
                worker_id: state.config.descriptor.worker_id.clone(),
                identity: request.identity,
                disposition: CancelDisposition::Unknown,
            })
            .into_response();
        };
        if record.state.is_terminal() {
            (CancelDisposition::AlreadyTerminal, None)
        } else if record.cancel_requested {
            (CancelDisposition::AlreadyRequested, None)
        } else {
            record.cancel_requested = true;
            record.state = AttemptState::CancellationRequested;
            (CancelDisposition::Requested, record.cancel.clone())
        }
    };

    let disposition = if let Some(cancel) = cancel {
        if cancel.send(true).is_ok() {
            state.update_attempt(
                &request.identity.attempt_id,
                AttemptState::ExecutionStopping,
                None,
            );
            disposition
        } else {
            CancelDisposition::Unknown
        }
    } else {
        disposition
    };
    Json(CancelAttemptResponse {
        schema_version: PROTOCOL_V1,
        worker_id: state.config.descriptor.worker_id.clone(),
        identity: request.identity,
        disposition,
    })
    .into_response()
}

fn counted_rejection<E>(
    state: &WorkerState<E>,
    request: &InvocationRequest,
    status: StatusCode,
    code: RejectionCode,
    message: impl Into<String>,
    retry_after_ms: Option<u64>,
) -> Response {
    state.metrics.inner.rejected.fetch_add(1, Ordering::Relaxed);
    rejection(request, status, code, message, retry_after_ms)
}

fn rejection(
    request: &InvocationRequest,
    status: StatusCode,
    code: RejectionCode,
    message: impl Into<String>,
    retry_after_ms: Option<u64>,
) -> Response {
    let mut rejection = InvocationRejection::new(
        request.request_id.clone(),
        request.attempt_id.clone(),
        code.for_peer(request.schema_version),
        bounded_message(message.into()),
    );
    rejection.retry_after_ms = retry_after_ms;
    (status, Json(rejection)).into_response()
}

fn rejection_status(code: RejectionCode) -> StatusCode {
    match code {
        RejectionCode::Unauthenticated => StatusCode::UNAUTHORIZED,
        RejectionCode::Unauthorized | RejectionCode::PolicyDenied => StatusCode::FORBIDDEN,
        RejectionCode::UnsupportedProtocolVersion
        | RejectionCode::WrongWorkerIncarnation
        | RejectionCode::WrongModelGeneration
        | RejectionCode::DuplicateAttemptConflict => StatusCode::CONFLICT,
        RejectionCode::UnknownDeployment => StatusCode::NOT_FOUND,
        RejectionCode::IncompatibleTask => StatusCode::UNPROCESSABLE_ENTITY,
        RejectionCode::ModelNotReady | RejectionCode::WorkerDraining => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        RejectionCode::CapacityExhausted | RejectionCode::QueueWaitExceeded => {
            StatusCode::TOO_MANY_REQUESTS
        }
        RejectionCode::InvalidRequest | RejectionCode::ContextLengthExceeded => {
            StatusCode::BAD_REQUEST
        }
        // A worker never sends a code it does not know.
        RejectionCode::Unknown => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn bounded_message(mut message: String) -> String {
    const MAX_MESSAGE_BYTES: usize = 1024;
    if message.len() <= MAX_MESSAGE_BYTES {
        return message;
    }
    let mut end = MAX_MESSAGE_BYTES;
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    message.truncate(end);
    message
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::to_bytes, http::Request};
    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::ServiceExt;

    fn id<T: TryFrom<&'static str>>(value: &'static str) -> T
    where
        T::Error: std::fmt::Debug,
    {
        T::try_from(value).unwrap()
    }

    fn credentials() -> ServiceCredentials {
        ServiceCredentials {
            credential_id: id("worker-credential"),
            bearer_token: ServiceBearerToken::new("worker-secret").unwrap(),
        }
    }

    fn config() -> WorkerConfig {
        WorkerConfig {
            descriptor: WorkerDescriptor {
                schema_version: PROTOCOL_V1,
                supported_protocol_versions: vec![PROTOCOL_V1],
                worker_id: id("cpu-worker"),
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
                ]),
            },
            deployment: LoadedDeployment {
                deployment_id: id("lfm-cpu-v1"),
                public_model: id("tiny-lfm"),
                artifact_revision: id("tiny-fixture-v1"),
                model_generation: ModelGeneration::new(1).unwrap(),
                task: TaskKind::Chat,
                backend: BackendKind::Cpu,
                precision: "f32".into(),
                execution_representation: "tiny-lfm".into(),
                tokenizer_revision: None,
                readiness: ModelReadiness::Ready,
                capability: Capability {
                    task: TaskKind::Chat,
                    streaming: true,
                    realtime: false,
                    cancellation: CancellationBehavior::Cooperative,
                    accepted_input_formats: BTreeSet::from([InputFormat::ChatMessages]),
                    output_formats: BTreeSet::from([OutputFormat::Text]),
                    max_input_bytes: 4096,
                    max_context_tokens: Some(32),
                    max_output_tokens: Some(32),
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
            credentials: credentials(),
            max_active_invocations: 1,
            max_request_bytes: 4096,
            max_retained_attempts: 4,
            attempt_retention: Duration::from_secs(300),
            event_channel_capacity: 4,
            max_event_bytes: 4096,
            slow_consumer_timeout: DEFAULT_SLOW_CONSUMER_TIMEOUT,
            realtime_session_limits: REALTIME_SESSION_DEFAULT_LIMITS,
        }
    }

    fn invocation() -> InvocationRequest {
        InvocationRequest {
            schema_version: PROTOCOL_V1,
            request_id: id("request-1"),
            attempt_id: id("attempt-1"),
            expected_worker_incarnation: id("incarnation-1"),
            deployment_id: id("lfm-cpu-v1"),
            expected_model_generation: ModelGeneration::new(1).unwrap(),
            caller: GatewayAttestedCallerContext {
                tenant_id: id("tenant-1"),
                caller_id: id("caller-1"),
                policy_revision: id("policy-1"),
                permitted_actions: BTreeSet::from([PermittedAction::Invoke]),
                allowed_data_regions: vec!["local".into()],
            },
            task: TaskKind::Chat,
            service_class: ServiceClass::Interactive,
            remaining_time_ms: 5_000,
            max_queue_wait_ms: 1_000,
            output_limits: OutputLimits {
                max_tokens: 8,
                max_bytes: 1024,
            },
            requested_output_format: OutputFormat::Text,
            session_id: None,
            request_digest: id("sha256:test"),
            input: InvocationInput::Chat {
                input: ChatInput {
                    messages: vec![ChatMessage {
                        role: ChatRole::User,
                        content: "hello".into(),
                    }],
                },
                parameters: ChatParameters::default(),
            },
        }
    }

    fn authorized_request(method: &str, uri: &str, body: Body) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header(
                SERVICE_AUTHORIZATION_HEADER,
                format!("{SERVICE_AUTH_SCHEME} worker-secret"),
            )
            .header(SERVICE_CREDENTIAL_ID_HEADER, "worker-credential")
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(body)
            .unwrap()
    }

    struct ScriptExecutor {
        events: Mutex<Option<VecDeque<ExecutionEvent>>>,
        teardown: ExecutionTeardown,
        cancel_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl InvocationExecutor for ScriptExecutor {
        async fn admit(
            &self,
            _request: &InvocationRequest,
        ) -> Result<AdmittedInvocation, AdmissionFailure> {
            let events = self.events.lock().unwrap().take().unwrap();
            Ok(AdmittedInvocation::new(Box::new(ScriptExecution {
                events,
                teardown: self.teardown,
                cancel_calls: Arc::clone(&self.cancel_calls),
            })))
        }
    }

    struct ScriptExecution {
        events: VecDeque<ExecutionEvent>,
        teardown: ExecutionTeardown,
        cancel_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl AdmittedExecution for ScriptExecution {
        async fn next_event(&mut self) -> Option<ExecutionEvent> {
            self.events.pop_front()
        }

        fn request_cancel(&self) {
            self.cancel_calls.fetch_add(1, Ordering::Relaxed);
        }

        async fn wait_for_teardown(self: Box<Self>) -> ExecutionTeardown {
            self.teardown
        }
    }

    #[tokio::test]
    async fn accepted_stream_is_ordered_and_terminal_after_teardown() {
        let executor = ScriptExecutor {
            events: Mutex::new(Some(VecDeque::from([
                ExecutionEvent::TextDelta {
                    text: "tiny response".into(),
                    logprobs: Vec::new(),
                },
                ExecutionEvent::Completed {
                    text: None,
                    finish_reason: FinishReason::Stop,
                    input_tokens: 2,
                    output_tokens: 2,
                    cached_input_tokens: None,
                },
            ]))),
            teardown: ExecutionTeardown::Completed,
            cancel_calls: Arc::new(AtomicUsize::new(0)),
        };
        let service = WorkerService::new(config(), executor).unwrap();
        let response = service
            .router()
            .oneshot(authorized_request(
                "POST",
                INVOCATIONS_PATH,
                Body::from(serde_json::to_vec(&invocation()).unwrap()),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        let events = body
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice::<InvocationEvent>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(events.len(), 3);
        assert!(matches!(
            events[0].event,
            InvocationEventKind::Accepted { .. }
        ));
        assert!(matches!(
            events[1].event,
            InvocationEventKind::TextDelta { .. }
        ));
        assert!(matches!(
            events[2].event,
            InvocationEventKind::Completed { .. }
        ));
        assert_eq!(service.active_invocations(), 0);
    }

    /// DP-F11: a gateway from before minor 4 sends no version header and
    /// cannot decode minor-4 features, so the worker leaves them out for it.
    #[tokio::test]
    async fn descriptor_leaves_out_features_the_gateway_cannot_decode() {
        let mut config = config();
        config.descriptor.features.insert(WorkerFeature::ChatSeed);
        let service = WorkerService::new(
            config,
            HealthExecutor {
                health: Arc::new(Mutex::new(RuntimeHealth::Healthy)),
            },
        )
        .unwrap();
        let descriptor = |version: Option<&str>| {
            let mut request = authorized_request("GET", WORKER_DESCRIPTOR_PATH, Body::empty());
            if let Some(version) = version {
                request
                    .headers_mut()
                    .insert(PROTOCOL_VERSION_HEADER, version.parse().unwrap());
            }
            let router = service.router();
            async move {
                let response = router.oneshot(request).await.unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
                serde_json::from_slice::<serde_json::Value>(&body).unwrap()
            }
        };
        for (version, advertised) in [
            (None, false),
            (Some("1.3"), false),
            (Some("not-a-version"), false),
            (Some("1.4"), true),
            (Some("1.9"), true),
        ] {
            let features = descriptor(version).await["features"].clone();
            assert_eq!(
                features
                    .as_array()
                    .unwrap()
                    .contains(&serde_json::json!("chat_seed")),
                advertised,
                "{version:?}: {features}"
            );
            assert!(features
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("streaming")));
        }
    }

    struct OverflowExecutor;

    #[async_trait]
    impl InvocationExecutor for OverflowExecutor {
        async fn admit(
            &self,
            _request: &InvocationRequest,
        ) -> Result<AdmittedInvocation, AdmissionFailure> {
            Err(AdmissionFailure::new(
                RejectionCode::ContextLengthExceeded,
                "This model's maximum context length is 32 tokens.",
            ))
        }
    }

    async fn rejection_for(
        service: &WorkerService<OverflowExecutor>,
        request: &InvocationRequest,
    ) -> (StatusCode, InvocationRejection) {
        let response = service
            .router()
            .oneshot(authorized_request(
                "POST",
                INVOCATIONS_PATH,
                Body::from(serde_json::to_vec(request).unwrap()),
            ))
            .await
            .unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        (status, serde_json::from_slice(&body).unwrap())
    }

    /// A control the worker does not advertise is refused, never ignored;
    /// a rejection code a gateway cannot decode becomes the older one.
    #[tokio::test]
    async fn invocations_honour_advertised_controls_and_the_gateway_minor() {
        let service = WorkerService::new(config(), OverflowExecutor).unwrap();
        let mut seeded = invocation();
        let InvocationInput::Chat { parameters, .. } = &mut seeded.input;
        parameters.seed = Some(7);
        let (status, rejection) = rejection_for(&service, &seeded).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(rejection.code, RejectionCode::InvalidRequest);
        assert!(
            rejection.message.contains("`seed`"),
            "{}",
            rejection.message
        );

        let (status, rejection) = rejection_for(&service, &invocation()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(rejection.code, RejectionCode::ContextLengthExceeded);
        let mut minor3 = invocation();
        minor3.schema_version = SchemaVersion::new(1, 3);
        let (status, rejection) = rejection_for(&service, &minor3).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(rejection.code, RejectionCode::InvalidRequest);
        assert_eq!(
            rejection.message,
            "This model's maximum context length is 32 tokens."
        );
    }

    struct HealthExecutor {
        health: Arc<Mutex<RuntimeHealth>>,
    }

    #[async_trait]
    impl InvocationExecutor for HealthExecutor {
        async fn admit(
            &self,
            _request: &InvocationRequest,
        ) -> Result<AdmittedInvocation, AdmissionFailure> {
            Err(AdmissionFailure::new(
                RejectionCode::CapacityExhausted,
                "health-only executor",
            ))
        }

        fn runtime_health(&self) -> RuntimeHealth {
            self.health.lock().unwrap().clone()
        }
    }

    async fn worker_status<E: InvocationExecutor>(service: &WorkerService<E>) -> WorkerStatus {
        let response = service
            .router()
            .oneshot(authorized_request("GET", WORKER_STATUS_PATH, Body::empty()))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    #[tokio::test]
    async fn status_reports_runtime_health_instead_of_static_readiness() {
        let health = Arc::new(Mutex::new(RuntimeHealth::Healthy));
        let service = WorkerService::new(
            config(),
            HealthExecutor {
                health: Arc::clone(&health),
            },
        )
        .unwrap();
        let status = worker_status(&service).await;
        assert_eq!(status.process_state, WorkerProcessState::Running);
        assert_eq!(status.deployments[0].readiness, ModelReadiness::Ready);

        *health.lock().unwrap() = RuntimeHealth::Degraded {
            reason: "Metal command buffer OOM".into(),
        };
        let status = worker_status(&service).await;
        assert_eq!(status.process_state, WorkerProcessState::Running);
        assert_eq!(status.deployments[0].readiness, ModelReadiness::Failed);

        *health.lock().unwrap() = RuntimeHealth::Failed {
            reason: "fatal device fault".into(),
        };
        service.begin_draining().await;
        let status = worker_status(&service).await;
        assert_eq!(status.process_state, WorkerProcessState::Failed);
        assert_eq!(status.deployments[0].readiness, ModelReadiness::Failed);
    }

    #[tokio::test]
    async fn status_fails_after_the_real_runtime_latches_a_device_fault() {
        let models_dir = tempfile::tempdir().unwrap();
        let runtime = Arc::new(
            izwi_core::RuntimeService::new_assigned(
                izwi_core::EngineConfig {
                    models_dir: models_dir.path().to_path_buf(),
                    ..izwi_core::EngineConfig::default()
                },
                izwi_core::backends::RuntimeDeviceAssignment::Cpu,
            )
            .unwrap(),
        );
        let service = WorkerService::new(
            config(),
            RuntimeChatExecutor::new(
                Arc::clone(&runtime),
                izwi_core::ModelVariant::Lfm2512BInstructGguf,
                true,
            ),
        )
        .unwrap();
        let status = worker_status(&service).await;
        assert_eq!(status.process_state, WorkerProcessState::Running);
        assert_eq!(status.deployments[0].readiness, ModelReadiness::Ready);
        assert!(service.runtime_health().is_healthy());

        assert!(runtime.latch_device_fault_for_test(&izwi_core::Error::InferenceError(
            "DriverError(CUDA_ERROR_ILLEGAL_ADDRESS, \"an illegal memory access was encountered\")"
                .into(),
        )));
        let status = worker_status(&service).await;
        assert_eq!(status.process_state, WorkerProcessState::Failed);
        assert_eq!(status.deployments[0].readiness, ModelReadiness::Failed);
        assert!(service.runtime_health().is_failed());
    }

    #[tokio::test]
    async fn worker_metrics_are_authenticated_bounded_and_fixed_cardinality() {
        let executor = ScriptExecutor {
            events: Mutex::new(Some(VecDeque::from([ExecutionEvent::Completed {
                text: Some("tiny response".into()),
                finish_reason: FinishReason::Stop,
                input_tokens: 2,
                output_tokens: 2,
                cached_input_tokens: None,
            }]))),
            teardown: ExecutionTeardown::Completed,
            cancel_calls: Arc::new(AtomicUsize::new(0)),
        };
        let service = WorkerService::new(config(), executor).unwrap();

        let denied = service
            .router()
            .oneshot(
                Request::builder()
                    .uri(WORKER_METRICS_PATH)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);

        let response = service
            .router()
            .oneshot(authorized_request(
                "POST",
                INVOCATIONS_PATH,
                Body::from(serde_json::to_vec(&invocation()).unwrap()),
            ))
            .await
            .unwrap();
        let _ = to_bytes(response.into_body(), 16 * 1024).await.unwrap();

        let scrape = service
            .router()
            .oneshot(authorized_request(
                "GET",
                WORKER_METRICS_PATH,
                Body::empty(),
            ))
            .await
            .unwrap();
        assert_eq!(scrape.status(), StatusCode::OK);
        assert_eq!(
            scrape.headers()[axum::http::header::CONTENT_TYPE],
            "text/plain; version=0.0.4; charset=utf-8"
        );
        let body = to_bytes(scrape.into_body(), MAX_PROMETHEUS_RESPONSE_BYTES)
            .await
            .unwrap();
        let body = std::str::from_utf8(&body).unwrap();
        assert!(body.len() <= MAX_PROMETHEUS_RESPONSE_BYTES);
        assert!(body.contains("izwi_worker_active_invocations 0\n"));
        assert!(body.contains("izwi_worker_retained_attempts 1\n"));
        assert!(body.contains("izwi_worker_admitted_total 1\n"));
        assert!(body.contains("izwi_worker_completed_total 1\n"));
        assert!(body.contains("izwi_worker_auth_rejections_total 1\n"));
        assert!(body.contains("izwi_worker_queue_wait_observations_total 1\n"));
        assert!(body.contains("izwi_worker_execution_observations_total 1\n"));
        for private_value in [
            "cpu-worker",
            "local-node",
            "incarnation-1",
            "lfm-cpu-v1",
            "tiny-lfm",
            "tenant-1",
            "caller-1",
            "worker-secret",
        ] {
            assert!(!body.contains(private_value));
        }
        assert!(body
            .lines()
            .filter(|line| !line.starts_with('#'))
            .all(|line| !line.contains('{')));
    }

    fn ndjson_events(body: &[u8]) -> Vec<InvocationEvent> {
        body.split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice::<InvocationEvent>(line).unwrap())
            .collect()
    }

    #[tokio::test]
    async fn paused_reader_receives_the_full_text_once_it_resumes() {
        let cancel_calls = Arc::new(AtomicUsize::new(0));
        let words = (0..64)
            .map(|index| format!("w{index} "))
            .collect::<Vec<_>>();
        let mut events = words
            .iter()
            .map(|word| ExecutionEvent::TextDelta {
                text: word.clone(),
                logprobs: Vec::new(),
            })
            .collect::<VecDeque<_>>();
        events.push_back(ExecutionEvent::Completed {
            text: None,
            finish_reason: FinishReason::Stop,
            input_tokens: 1,
            output_tokens: 64,
            cached_input_tokens: None,
        });
        let executor = ScriptExecutor {
            events: Mutex::new(Some(events)),
            teardown: ExecutionTeardown::Completed,
            cancel_calls: Arc::clone(&cancel_calls),
        };
        let mut worker_config = config();
        worker_config.event_channel_capacity = 3;
        let service = WorkerService::new(worker_config, executor).unwrap();
        let response = service
            .router()
            .oneshot(authorized_request(
                "POST",
                INVOCATIONS_PATH,
                Body::from(serde_json::to_vec(&invocation()).unwrap()),
            ))
            .await
            .unwrap();
        // The reader stays paused while the execution emits far more deltas
        // than the three-slot channel holds and completes.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(cancel_calls.load(Ordering::Acquire), 0);

        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let events = ndjson_events(&body);
        let text = events
            .iter()
            .filter_map(|event| match &event.event {
                InvocationEventKind::TextDelta { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(text, words.concat());
        assert!(events.len() < words.len() + 2, "deltas were coalesced");
        assert!(matches!(
            events.last().unwrap().event,
            InvocationEventKind::Completed { .. }
        ));
        assert!(events
            .windows(2)
            .all(|pair| pair[1].sequence == pair[0].sequence + 1));
        assert_eq!(cancel_calls.load(Ordering::Acquire), 0);
        assert_eq!(service.active_invocations(), 0);
    }

    struct StallingExecutor {
        deltas: usize,
        cancel_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl InvocationExecutor for StallingExecutor {
        async fn admit(
            &self,
            _request: &InvocationRequest,
        ) -> Result<AdmittedInvocation, AdmissionFailure> {
            Ok(AdmittedInvocation::new(Box::new(StallingExecution {
                remaining: self.deltas,
                cancel_calls: Arc::clone(&self.cancel_calls),
            })))
        }
    }

    /// Emits its deltas, then keeps generating (never finishes on its own).
    struct StallingExecution {
        remaining: usize,
        cancel_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl AdmittedExecution for StallingExecution {
        async fn next_event(&mut self) -> Option<ExecutionEvent> {
            if self.remaining == 0 {
                return std::future::pending().await;
            }
            self.remaining -= 1;
            Some(ExecutionEvent::TextDelta {
                text: "x".into(),
                logprobs: Vec::new(),
            })
        }

        fn request_cancel(&self) {
            self.cancel_calls.fetch_add(1, Ordering::Relaxed);
        }

        async fn wait_for_teardown(self: Box<Self>) -> ExecutionTeardown {
            ExecutionTeardown::Cancelled
        }
    }

    #[tokio::test]
    async fn reader_paused_past_the_slow_consumer_timeout_cancels_the_invocation() {
        let cancel_calls = Arc::new(AtomicUsize::new(0));
        let mut worker_config = config();
        worker_config.event_channel_capacity = 3;
        worker_config.slow_consumer_timeout = Duration::from_millis(50);
        let service = WorkerService::new(
            worker_config,
            StallingExecutor {
                deltas: 8,
                cancel_calls: Arc::clone(&cancel_calls),
            },
        )
        .unwrap();
        let response = service
            .router()
            .oneshot(authorized_request(
                "POST",
                INVOCATIONS_PATH,
                Body::from(serde_json::to_vec(&invocation()).unwrap()),
            ))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while cancel_calls.load(Ordering::Acquire) == 0 || service.active_invocations() != 0 {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("a reader stalled past the timeout cancels the invocation");
        assert_eq!(cancel_calls.load(Ordering::Acquire), 1);
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        assert!(ndjson_events(&body)
            .iter()
            .all(|event| !matches!(event.event, InvocationEventKind::Completed { .. })));
    }

    #[tokio::test]
    async fn unconfirmed_teardown_never_publishes_terminal_or_releases_capacity() {
        let executor = ScriptExecutor {
            events: Mutex::new(Some(VecDeque::new())),
            teardown: ExecutionTeardown::Unconfirmed,
            cancel_calls: Arc::new(AtomicUsize::new(0)),
        };
        let service = WorkerService::new(config(), executor).unwrap();
        let request = invocation();
        let attempt_id = request.attempt_id.clone();
        let response = service
            .router()
            .oneshot(authorized_request(
                "POST",
                INVOCATIONS_PATH,
                Body::from(serde_json::to_vec(&request).unwrap()),
            ))
            .await
            .unwrap();
        let body = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        let events = body
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice::<InvocationEvent>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].event,
            InvocationEventKind::Accepted { .. }
        ));
        assert_eq!(service.active_invocations(), 1);

        let query = service
            .router()
            .oneshot(authorized_request(
                "GET",
                &format!("{INVOCATIONS_PATH}/{attempt_id}"),
                Body::empty(),
            ))
            .await
            .unwrap();
        let body = to_bytes(query.into_body(), 4096).await.unwrap();
        let attempt: AttemptQueryResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(attempt.state, AttemptState::ExecutionStopping);
    }

    struct LateAdmissionExecutor {
        release: Arc<tokio::sync::Notify>,
        completed: Arc<AtomicBool>,
        cancel_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl InvocationExecutor for LateAdmissionExecutor {
        async fn admit(
            &self,
            _request: &InvocationRequest,
        ) -> Result<AdmittedInvocation, AdmissionFailure> {
            self.release.notified().await;
            self.completed.store(true, Ordering::Release);
            Ok(AdmittedInvocation::new(Box::new(ScriptExecution {
                events: VecDeque::new(),
                teardown: ExecutionTeardown::Cancelled,
                cancel_calls: Arc::clone(&self.cancel_calls),
            })))
        }
    }

    #[tokio::test]
    async fn expired_admission_wait_retains_ownership_and_is_not_a_rejection() {
        let release = Arc::new(tokio::sync::Notify::new());
        let completed = Arc::new(AtomicBool::new(false));
        let cancel_calls = Arc::new(AtomicUsize::new(0));
        let service = WorkerService::new(
            config(),
            LateAdmissionExecutor {
                release: Arc::clone(&release),
                completed: Arc::clone(&completed),
                cancel_calls: Arc::clone(&cancel_calls),
            },
        )
        .unwrap();
        let mut request = invocation();
        request.max_queue_wait_ms = 1;
        let response = service
            .router()
            .oneshot(authorized_request(
                "POST",
                INVOCATIONS_PATH,
                Body::from(serde_json::to_vec(&request).unwrap()),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        assert!(serde_json::from_slice::<InvocationRejection>(&body).is_err());
        let unknown: InvocationAdmissionUnknown = serde_json::from_slice(&body).unwrap();
        assert_eq!(unknown.attempt_id, request.attempt_id);
        assert!(!completed.load(Ordering::Acquire));
        assert_eq!(service.active_invocations(), 1);

        release.notify_one();
        tokio::time::timeout(Duration::from_secs(1), async {
            while service.active_invocations() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(completed.load(Ordering::Acquire));
        assert_eq!(cancel_calls.load(Ordering::Acquire), 1);
    }

    struct DelayedExecutor {
        entered: Arc<tokio::sync::Notify>,
        release_admission: Arc<tokio::sync::Notify>,
        finish_execution: Arc<tokio::sync::Notify>,
        cancel_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl InvocationExecutor for DelayedExecutor {
        async fn admit(
            &self,
            _request: &InvocationRequest,
        ) -> Result<AdmittedInvocation, AdmissionFailure> {
            self.entered.notify_one();
            self.release_admission.notified().await;
            Ok(AdmittedInvocation::new(Box::new(DelayedExecution {
                finish_execution: Arc::clone(&self.finish_execution),
                cancel_calls: Arc::clone(&self.cancel_calls),
            })))
        }
    }

    struct DelayedExecution {
        finish_execution: Arc<tokio::sync::Notify>,
        cancel_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl AdmittedExecution for DelayedExecution {
        async fn next_event(&mut self) -> Option<ExecutionEvent> {
            self.finish_execution.notified().await;
            None
        }

        fn request_cancel(&self) {
            self.cancel_calls.fetch_add(1, Ordering::Relaxed);
        }

        async fn wait_for_teardown(self: Box<Self>) -> ExecutionTeardown {
            self.finish_execution.notified().await;
            ExecutionTeardown::Cancelled
        }
    }

    #[tokio::test]
    async fn cancel_before_admission_is_preserved_and_capacity_waits_for_teardown() {
        let entered = Arc::new(tokio::sync::Notify::new());
        let release_admission = Arc::new(tokio::sync::Notify::new());
        let finish_execution = Arc::new(tokio::sync::Notify::new());
        let cancel_calls = Arc::new(AtomicUsize::new(0));
        let service = WorkerService::new(
            config(),
            DelayedExecutor {
                entered: Arc::clone(&entered),
                release_admission: Arc::clone(&release_admission),
                finish_execution: Arc::clone(&finish_execution),
                cancel_calls: Arc::clone(&cancel_calls),
            },
        )
        .unwrap();
        let request = invocation();
        let identity = AttemptIdentity::from(&request);
        let invoke_router = service.router();
        let invoke_task = tokio::spawn(async move {
            invoke_router
                .oneshot(authorized_request(
                    "POST",
                    INVOCATIONS_PATH,
                    Body::from(serde_json::to_vec(&request).unwrap()),
                ))
                .await
                .unwrap()
        });
        tokio::time::timeout(Duration::from_secs(1), entered.notified())
            .await
            .unwrap();
        let cancel_path = format!("{INVOCATIONS_PATH}/{}/cancel", identity.attempt_id);
        let cancel_response = service
            .router()
            .oneshot(authorized_request(
                "POST",
                &cancel_path,
                Body::from(
                    serde_json::to_vec(&CancelAttemptRequest {
                        schema_version: PROTOCOL_V1,
                        identity,
                    })
                    .unwrap(),
                ),
            ))
            .await
            .unwrap();
        let body = to_bytes(cancel_response.into_body(), 4096).await.unwrap();
        let cancelled: CancelAttemptResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(cancelled.disposition, CancelDisposition::Requested);

        release_admission.notify_one();
        let response = invoke_task.await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        tokio::time::timeout(Duration::from_secs(1), async {
            while cancel_calls.load(Ordering::Acquire) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(service.active_invocations(), 1);

        finish_execution.notify_one();
        let _ = to_bytes(response.into_body(), 16 * 1024).await.unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while service.active_invocations() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        let scrape = service
            .router()
            .oneshot(authorized_request(
                "GET",
                WORKER_METRICS_PATH,
                Body::empty(),
            ))
            .await
            .unwrap();
        let body = to_bytes(scrape.into_body(), MAX_PROMETHEUS_RESPONSE_BYTES)
            .await
            .unwrap();
        let body = std::str::from_utf8(&body).unwrap();
        assert!(body.contains("izwi_worker_cancellation_requests_total 1\n"));
        assert!(body.contains("izwi_worker_cancelled_total 1\n"));
        assert!(body.contains("izwi_worker_cancellation_to_stop_observations_total 1\n"));
    }

    #[tokio::test]
    async fn request_body_limit_is_enforced_before_json_allocation() {
        let mut worker_config = config();
        worker_config.max_request_bytes = 64;
        let executor = ScriptExecutor {
            events: Mutex::new(Some(VecDeque::new())),
            teardown: ExecutionTeardown::Completed,
            cancel_calls: Arc::new(AtomicUsize::new(0)),
        };
        let service = WorkerService::new(worker_config, executor).unwrap();
        let response = service
            .router()
            .oneshot(authorized_request(
                "POST",
                INVOCATIONS_PATH,
                Body::from(serde_json::to_vec(&invocation()).unwrap()),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let scrape = service
            .router()
            .oneshot(authorized_request(
                "GET",
                WORKER_METRICS_PATH,
                Body::empty(),
            ))
            .await
            .unwrap();
        let body = to_bytes(scrape.into_body(), MAX_PROMETHEUS_RESPONSE_BYTES)
            .await
            .unwrap();
        let body = std::str::from_utf8(&body).unwrap();
        assert!(body.contains("izwi_worker_body_limit_rejections_total 1\n"));
        assert!(body.contains("izwi_worker_rejected_total 1\n"));
    }

    /// HYG-R4: a panic while the attempt table was held must not take the
    /// worker down with it; later requests recover the lock.
    #[tokio::test]
    async fn a_poisoned_attempt_table_keeps_serving() {
        let executor = ScriptExecutor {
            events: Mutex::new(Some(VecDeque::new())),
            teardown: ExecutionTeardown::Completed,
            cancel_calls: Arc::new(AtomicUsize::new(0)),
        };
        let service = WorkerService::new(config(), executor).unwrap();
        let state = Arc::clone(&service.state);
        let poisoner = std::thread::spawn(move || {
            let _table = state.attempts.lock().unwrap();
            panic!("poison the attempt table");
        });
        assert!(poisoner.join().is_err());
        assert!(service.state.attempts.is_poisoned());

        assert_eq!(service.request_cancel_all(), 0);
        let response = service
            .router()
            .oneshot(authorized_request(
                "POST",
                INVOCATIONS_PATH,
                Body::from(serde_json::to_vec(&invocation()).unwrap()),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn attested_caller_without_invoke_permission_is_rejected_before_admission() {
        let executor = ScriptExecutor {
            events: Mutex::new(Some(VecDeque::new())),
            teardown: ExecutionTeardown::Completed,
            cancel_calls: Arc::new(AtomicUsize::new(0)),
        };
        let service = WorkerService::new(config(), executor).unwrap();
        let mut request = invocation();
        request.caller.permitted_actions.clear();
        let response = service
            .router()
            .oneshot(authorized_request(
                "POST",
                INVOCATIONS_PATH,
                Body::from(serde_json::to_vec(&request).unwrap()),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let rejection: InvocationRejection = serde_json::from_slice(&body).unwrap();
        assert_eq!(rejection.code, RejectionCode::PolicyDenied);
        assert_eq!(service.active_invocations(), 0);
    }

    #[tokio::test]
    async fn drain_waits_for_in_progress_admission_then_rejects_new_work() {
        let entered = Arc::new(tokio::sync::Notify::new());
        let release_admission = Arc::new(tokio::sync::Notify::new());
        let finish_execution = Arc::new(tokio::sync::Notify::new());
        let cancel_calls = Arc::new(AtomicUsize::new(0));
        let service = WorkerService::new(
            config(),
            DelayedExecutor {
                entered: Arc::clone(&entered),
                release_admission: Arc::clone(&release_admission),
                finish_execution: Arc::clone(&finish_execution),
                cancel_calls: Arc::clone(&cancel_calls),
            },
        )
        .unwrap();
        let first_request = invocation();
        let invoke_router = service.router();
        let invoke_task = tokio::spawn(async move {
            invoke_router
                .oneshot(authorized_request(
                    "POST",
                    INVOCATIONS_PATH,
                    Body::from(serde_json::to_vec(&first_request).unwrap()),
                ))
                .await
                .unwrap()
        });
        tokio::time::timeout(Duration::from_secs(1), entered.notified())
            .await
            .unwrap();

        let drain_service = service.clone();
        let drain_task = tokio::spawn(async move { drain_service.begin_draining().await });
        tokio::task::yield_now().await;
        assert!(!drain_task.is_finished());

        release_admission.notify_one();
        let first_response = invoke_task.await.unwrap();
        assert_eq!(first_response.status(), StatusCode::OK);
        tokio::time::timeout(Duration::from_secs(1), drain_task)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(service.request_cancel_all(), 1);
        tokio::time::timeout(Duration::from_secs(1), async {
            while cancel_calls.load(Ordering::Acquire) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(service.request_cancel_all(), 0);
        assert_eq!(service.active_invocations(), 1);

        let mut second_request = invocation();
        second_request.request_id = id("request-2");
        second_request.attempt_id = id("attempt-2");
        second_request.request_digest = id("sha256:test-2");
        let response = service
            .router()
            .oneshot(authorized_request(
                "POST",
                INVOCATIONS_PATH,
                Body::from(serde_json::to_vec(&second_request).unwrap()),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), 4096).await.unwrap();
        let rejection: InvocationRejection = serde_json::from_slice(&body).unwrap();
        assert_eq!(rejection.code, RejectionCode::WorkerDraining);

        drop(first_response);
        finish_execution.notify_one();
        tokio::time::timeout(Duration::from_secs(1), async {
            while service.active_invocations() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn duplicate_owned_attempt_is_an_uncertain_response_not_a_rejection() {
        let entered = Arc::new(tokio::sync::Notify::new());
        let release_admission = Arc::new(tokio::sync::Notify::new());
        let finish_execution = Arc::new(tokio::sync::Notify::new());
        let service = WorkerService::new(
            config(),
            DelayedExecutor {
                entered: Arc::clone(&entered),
                release_admission: Arc::clone(&release_admission),
                finish_execution: Arc::clone(&finish_execution),
                cancel_calls: Arc::new(AtomicUsize::new(0)),
            },
        )
        .unwrap();
        let request = invocation();
        let first_router = service.router();
        let first_body = serde_json::to_vec(&request).unwrap();
        let first_task = tokio::spawn(async move {
            first_router
                .oneshot(authorized_request(
                    "POST",
                    INVOCATIONS_PATH,
                    Body::from(first_body),
                ))
                .await
                .unwrap()
        });
        tokio::time::timeout(Duration::from_secs(1), entered.notified())
            .await
            .unwrap();
        release_admission.notify_one();
        let first_response = first_task.await.unwrap();

        let duplicate = service
            .router()
            .oneshot(authorized_request(
                "POST",
                INVOCATIONS_PATH,
                Body::from(serde_json::to_vec(&request).unwrap()),
            ))
            .await
            .unwrap();
        assert_eq!(duplicate.status(), StatusCode::CONFLICT);
        let body = to_bytes(duplicate.into_body(), 4096).await.unwrap();
        assert!(serde_json::from_slice::<InvocationRejection>(&body).is_err());

        drop(first_response);
        finish_execution.notify_one();
        tokio::time::timeout(Duration::from_secs(1), async {
            while service.active_invocations() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn dropping_http_stream_requests_cancel_and_holds_capacity_until_teardown() {
        let entered = Arc::new(tokio::sync::Notify::new());
        let release_admission = Arc::new(tokio::sync::Notify::new());
        let finish_execution = Arc::new(tokio::sync::Notify::new());
        let cancel_calls = Arc::new(AtomicUsize::new(0));
        let service = WorkerService::new(
            config(),
            DelayedExecutor {
                entered: Arc::clone(&entered),
                release_admission: Arc::clone(&release_admission),
                finish_execution: Arc::clone(&finish_execution),
                cancel_calls: Arc::clone(&cancel_calls),
            },
        )
        .unwrap();
        let router = service.router();
        let task = tokio::spawn(async move {
            router
                .oneshot(authorized_request(
                    "POST",
                    INVOCATIONS_PATH,
                    Body::from(serde_json::to_vec(&invocation()).unwrap()),
                ))
                .await
                .unwrap()
        });
        tokio::time::timeout(Duration::from_secs(1), entered.notified())
            .await
            .unwrap();
        release_admission.notify_one();
        let response = task.await.unwrap();
        assert_eq!(service.active_invocations(), 1);
        drop(response);

        tokio::time::timeout(Duration::from_secs(1), async {
            while cancel_calls.load(Ordering::Acquire) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(service.active_invocations(), 1);

        finish_execution.notify_one();
        tokio::time::timeout(Duration::from_secs(1), async {
            while service.active_invocations() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn cancel_tombstone_expires_after_the_declared_retention_window() {
        let mut worker_config = config();
        worker_config.attempt_retention = Duration::from_secs(1);
        let service = WorkerService::new(
            worker_config,
            ScriptExecutor {
                events: Mutex::new(Some(VecDeque::new())),
                teardown: ExecutionTeardown::Completed,
                cancel_calls: Arc::new(AtomicUsize::new(0)),
            },
        )
        .unwrap();
        let request = invocation();
        service
            .state
            .insert_cancel_tombstone(AttemptIdentity::from(&request));
        assert_eq!(
            service.state.reserve_attempt(&request),
            ReserveAttempt::Conflict
        );

        tokio::time::advance(Duration::from_secs(1)).await;
        assert_eq!(
            service.state.reserve_attempt(&request),
            ReserveAttempt::Reserved
        );
    }

    #[test]
    fn tokens_out_ema_ignores_empty_and_starts_from_the_first_sample() {
        let ema = TokensOutEma::default();
        assert_eq!(ema.load(), None);
        ema.record(0, Duration::from_secs(1));
        assert_eq!(ema.load(), None);
        ema.record(10, Duration::from_secs(0));
        assert_eq!(ema.load(), None);

        ema.record(10, Duration::from_secs(1));
        assert_eq!(ema.load(), Some(10.0));
        ema.record(20, Duration::from_secs(1));
        let updated = ema.load().unwrap();
        assert!((updated - (TokensOutEma::ALPHA * 20.0 + 0.7 * 10.0)).abs() < 1e-9);
    }

    #[test]
    fn kv_routing_signals_follow_engine_counters_and_zero_capacity_stays_unavailable() {
        let mut kv_cache = ManagedKvRuntimeSnapshot::default();
        kv_cache.totals.coordinator.capacity_pages = 8;
        kv_cache.totals.coordinator.allocated_pages = 2;
        kv_cache.counters.prefix_hits = 5;
        kv_cache.counters.prefix_misses = 3;
        kv_cache.counters.prefix_evictions = 1;

        let mut deployment = config().deployment;
        apply_kv_routing_signals(&mut deployment, &kv_cache);
        assert_eq!(deployment.kv_cache_usage_pct, Some(25.0));
        assert_eq!(deployment.prefix_hits_total, Some(5));
        assert_eq!(deployment.prefix_queries_total, Some(8));
        assert_eq!(deployment.prefix_evictions_total, Some(1));

        let mut empty = ManagedKvRuntimeSnapshot::default();
        empty.counters.prefix_hits = 2;
        empty.counters.prefix_misses = 1;
        let mut deployment = config().deployment;
        apply_kv_routing_signals(&mut deployment, &empty);
        assert_eq!(deployment.kv_cache_usage_pct, None);
        assert_eq!(deployment.prefix_queries_total, Some(3));
    }
}
