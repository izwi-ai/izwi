//! Production-ready inference engine following vLLM architecture patterns.
//!
//! This module implements a high-throughput audio inference engine with:
//! - Request scheduling with FCFS/priority policies
//! - Continuous batching for improved throughput
//! - Paged KV-cache memory management
//! - Streaming output support
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                         Engine                                   │
//! │  ┌──────────────┐  ┌───────────┐  ┌──────────────────────────┐ │
//! │  │   Request    │  │           │  │      Engine Core          │ │
//! │  │  Processor   │──│ Scheduler │──│  ┌────────────────────┐  │ │
//! │  │              │  │           │  │  │  Model Executor    │  │ │
//! │  └──────────────┘  └───────────┘  │  │  (Native Rust)     │  │ │
//! │                                    │  └────────────────────┘  │ │
//! │  ┌──────────────┐                 │  ┌────────────────────┐  │ │
//! │  │   Output     │◄────────────────│  │  KV Cache Manager  │  │ │
//! │  │  Processor   │                 │  └────────────────────┘  │ │
//! │  └──────────────┘                 └──────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

mod cache;
mod config;
mod core;
pub mod execution;
mod executor;
mod kv_cache;
mod metal_kv_cache;
pub mod metrics;
mod output;
mod request;
pub mod resources;
mod scheduler;
pub mod signal_frontend;
mod types;

pub use config::EngineCoreConfig;
pub use core::EngineCore;
pub use execution::{
    BatchDispatch, BatchDispatchKind, BatchKey, CacheMode, CancellationGranularity,
    ConcurrencyClass, ExecutionCapabilities, ExecutionDisposition, ExecutionFailure, ExecutionMode,
    ExecutionPlan, ExecutionProfile, ExecutionReport, ExecutionState, ExecutionTracker,
    FailureKind, FailureScope, FinishReason, HealthImpact, InputRange, NativeBatchMode, PlanId,
    PrefillMode, RetryDisposition, SequencePhase, SessionEpoch, SessionKey, TerminalOutcome,
    WorkUnit, YieldReason,
};
pub use executor::{
    ExecutorOutput, ExecutorStepResult, ModelExecutor, ModelSessionResult, WorkerConfig,
    REQUEST_DEADLINE_EXCEEDED,
};
pub use kv_cache::{
    BlockAllocator, CacheResidency, KVCacheConfig as KVConfig, KVCacheManager, KVCacheStats,
    PinnedBlockHandle,
};
pub use metrics::{
    engine_metric_catalog, engine_request_parallel_batches_total, engine_stream_backpressure_total,
    engine_tensor_batch_max_width, engine_tensor_batches_total, prometheus_engine_metric_name,
    prometheus_engine_metric_type, BenchmarkResult, EngineMetricDescriptor, MetricsCollector,
    MetricsSnapshot, ENGINE_EXECUTOR_REQUEST_PARALLEL_BATCHES_TOTAL,
    ENGINE_EXECUTOR_TENSOR_BATCHES_TOTAL, ENGINE_EXECUTOR_TENSOR_BATCH_MAX_WIDTH,
    ENGINE_KV_CACHE_ALLOCATED_BLOCKS, ENGINE_KV_CACHE_CHURN_RATIO,
    ENGINE_KV_CACHE_COPY_ON_WRITE_SPLITS_TOTAL, ENGINE_KV_CACHE_EVICTIONS_TOTAL,
    ENGINE_KV_CACHE_FREE_BLOCKS, ENGINE_KV_CACHE_GPU_RESIDENT_BLOCKS, ENGINE_KV_CACHE_HITS_TOTAL,
    ENGINE_KV_CACHE_MEMORY_CAPACITY_BYTES, ENGINE_KV_CACHE_MEMORY_USED_BYTES,
    ENGINE_KV_CACHE_MISSES_TOTAL, ENGINE_KV_CACHE_PINNED_BLOCKS,
    ENGINE_KV_CACHE_PREFIX_REUSE_BLOCKS_TOTAL, ENGINE_KV_CACHE_SHARED_PREFIXES,
    ENGINE_KV_CACHE_SOFT_MAX_BLOCKS, ENGINE_KV_CACHE_UTILIZATION_RATIO, ENGINE_METRIC_CATALOG,
    ENGINE_SCHEDULER_PREEMPTIONS_TOTAL, ENGINE_SCHEDULER_QUEUE_DEPTH,
    ENGINE_SCHEDULER_RUNNING_REQUESTS, ENGINE_SCHEDULER_STEP_TOKENS_TOTAL,
    ENGINE_STREAM_BACKPRESSURE_TOTAL,
};
pub use output::{AsrProgress, AsrProgressPhase, OutputProcessor, StreamingOutput};
pub use request::{
    AsrEngineInput, AudioChatEngineInput, ChatEngineInput, EngineAudioInput, EngineCoreRequest,
    EngineStreamPolicy, EngineTask, RequestProcessor, RequestStatus, TtsEngineInput, WorkloadClass,
};
pub use resources::{
    CapacitySource, PhysicalCapacityProvider, PhysicalCapacitySnapshot, ReservationClass,
    ReservationId, ReservationOwner, ResourceAmount, ResourceAuthority, ResourceAuthoritySnapshot,
    ResourceEstimate, ResourceLease, ResourceLedger, ResourceReservation, ResourceVector,
};
pub use scheduler::{ScheduleResult, Scheduler, SchedulerConfig, SchedulingPolicy};
pub use types::FinishReason as OutputFinishReason;
pub use types::{
    AudioOutput, EngineMetrics, EngineOutput, GenerationParams, Priority, RequestId, SequenceId,
    TaskType,
};

use crate::error::Result;
use crate::model::ModelVariant;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Notify, RwLock};
use tracing::{debug, info, warn};

/// Main inference engine - the primary interface for audio generation.
///
/// The engine orchestrates all components and provides both synchronous
/// and asynchronous interfaces for inference.
struct RequestControl {
    session_epoch: SequenceId,
    cancellation: Arc<std::sync::atomic::AtomicBool>,
    model_variant: Option<ModelVariant>,
}

struct CompletionMailbox {
    registration_id: u64,
    session_epoch: Option<SequenceId>,
    sender: oneshot::Sender<EngineOutput>,
}

struct CompletionRegistration<'a> {
    engine: &'a Engine,
    request_id: RequestId,
    registration_id: u64,
}

impl Drop for CompletionRegistration<'_> {
    fn drop(&mut self) {
        let mut mailboxes = self
            .engine
            .completion_mailboxes
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if mailboxes
            .get(&self.request_id)
            .is_some_and(|mailbox| mailbox.registration_id == self.registration_id)
        {
            mailboxes.remove(&self.request_id);
        }
    }
}

pub struct Engine {
    /// Engine core handles the actual inference loop
    core: Arc<RwLock<EngineCore>>,
    /// Request processor validates and preprocesses inputs
    request_processor: RequestProcessor,
    /// Output processor formats results for clients
    output_processor: OutputProcessor,
    /// Configuration
    config: EngineCoreConfig,
    /// Whether the engine is running
    running: std::sync::atomic::AtomicBool,
    /// Metrics collector
    metrics: Arc<RwLock<EngineMetrics>>,
    /// Event-driven wakeup for run-loop when new requests arrive.
    wake_notify: Arc<Notify>,
    /// Session-fenced cooperative cancellation signals available without the core lock.
    request_controls: std::sync::Mutex<HashMap<RequestId, RequestControl>>,
    /// Exact-session terminal outputs for synchronous public callers.
    completion_mailboxes: std::sync::Mutex<HashMap<RequestId, CompletionMailbox>>,
    /// Distinguishes a cancelled registration from a later reuse of the public ID.
    next_completion_registration: std::sync::atomic::AtomicU64,
}

impl Engine {
    fn queue_capacity_from_env(key: &str) -> Option<usize> {
        std::env::var(key)
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
    }

    fn streaming_queue_capacity(request: &EngineCoreRequest) -> usize {
        let default_capacity = match request.task_type {
            TaskType::TTS => 8usize,
            // Unified speech-to-speech emits bursty interleaved text and audio
            // chunks, so it needs a deeper queue than plain TTS.
            TaskType::SpeechToSpeech => 64usize,
            // ASR can emit per-character deltas in streaming mode.
            // Use a deeper default queue to absorb bursty decode emission.
            TaskType::ASR => 4096usize,
            TaskType::Chat => 64usize,
        };

        let task_override = match request.task_type {
            TaskType::TTS | TaskType::SpeechToSpeech => {
                Self::queue_capacity_from_env("IZWI_STREAM_AUDIO_QUEUE_CAPACITY")
            }
            TaskType::ASR | TaskType::Chat => {
                Self::queue_capacity_from_env("IZWI_STREAM_TEXT_QUEUE_CAPACITY")
            }
        };

        task_override
            .or_else(|| Self::queue_capacity_from_env("IZWI_STREAM_QUEUE_CAPACITY"))
            .unwrap_or(default_capacity)
    }

    /// Create a new inference engine with the given configuration.
    pub fn new(config: EngineCoreConfig) -> Result<Self> {
        let worker_config = WorkerConfig::from(&config);
        Self::new_with_worker(config, worker_config)
    }

    /// Create a new inference engine with explicit worker configuration.
    pub fn new_with_worker(config: EngineCoreConfig, worker_config: WorkerConfig) -> Result<Self> {
        info!("Initializing inference engine");

        let core = EngineCore::new_with_worker(config.clone(), worker_config)?;
        let request_processor = RequestProcessor::new(config.clone());
        let output_processor = OutputProcessor::new(config.sample_rate);

        Ok(Self {
            core: Arc::new(RwLock::new(core)),
            request_processor,
            output_processor,
            config,
            running: std::sync::atomic::AtomicBool::new(false),
            metrics: Arc::new(RwLock::new(EngineMetrics::default())),
            wake_notify: Arc::new(Notify::new()),
            request_controls: std::sync::Mutex::new(HashMap::new()),
            completion_mailboxes: std::sync::Mutex::new(HashMap::new()),
            next_completion_registration: std::sync::atomic::AtomicU64::new(1),
        })
    }

    fn register_completion_mailbox(
        &self,
        request_id: RequestId,
    ) -> Result<(CompletionRegistration<'_>, oneshot::Receiver<EngineOutput>)> {
        use std::collections::hash_map::Entry;
        use std::sync::atomic::Ordering;

        let registration_id = self
            .next_completion_registration
            .fetch_add(1, Ordering::Relaxed);
        let (sender, receiver) = oneshot::channel();
        let mut mailboxes = self
            .completion_mailboxes
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        match mailboxes.entry(request_id.clone()) {
            Entry::Occupied(_) => {
                return Err(crate::error::Error::InvalidInput(format!(
                    "Request {request_id} already has a completion waiter"
                )));
            }
            Entry::Vacant(entry) => {
                entry.insert(CompletionMailbox {
                    registration_id,
                    session_epoch: None,
                    sender,
                });
            }
        }
        drop(mailboxes);

        Ok((
            CompletionRegistration {
                engine: self,
                request_id,
                registration_id,
            },
            receiver,
        ))
    }

    fn bind_completion_mailbox(
        &self,
        request_id: &RequestId,
        registration_id: u64,
        session_epoch: SequenceId,
    ) {
        let mut mailboxes = self
            .completion_mailboxes
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mailbox = mailboxes
            .get_mut(request_id)
            .filter(|mailbox| mailbox.registration_id == registration_id)
            .expect("completion registration must remain live while its request is admitted");
        mailbox.session_epoch = Some(session_epoch);
    }

    fn take_completion_sender(
        &self,
        session: &SessionKey,
    ) -> Option<oneshot::Sender<EngineOutput>> {
        let mut mailboxes = self
            .completion_mailboxes
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let owns_session = mailboxes
            .get(&session.request_id)
            .is_some_and(|mailbox| mailbox.session_epoch == Some(session.epoch));
        owns_session
            .then(|| mailboxes.remove(&session.request_id))
            .flatten()
            .map(|mailbox| mailbox.sender)
    }

    fn resolve_generation_output(
        request_id: &RequestId,
        output: EngineOutput,
    ) -> Result<EngineOutput> {
        if output.request_id != *request_id {
            return Err(crate::error::Error::InferenceError(format!(
                "Completion mailbox for {request_id} received output for {}",
                output.request_id
            )));
        }
        if output.finish_reason == Some(types::FinishReason::Aborted) {
            return Err(crate::error::Error::Cancelled(request_id.clone()));
        }
        if let Some(err) = output.error.clone() {
            return Err(crate::error::Error::InferenceError(err));
        }
        Ok(output)
    }

    async fn add_request_with_completion(
        &self,
        request: EngineCoreRequest,
        completion_registration: Option<u64>,
    ) -> Result<RequestId> {
        // Validate and preprocess
        let mut processed = self.request_processor.process(request)?;
        let request_id = processed.id.clone();
        let model_variant = processed.model_variant;
        let cancellation = Arc::new(std::sync::atomic::AtomicBool::new(false));
        processed.set_cancellation_signal(cancellation.clone());

        // Add to engine core. The core write lock also makes binding a pending
        // completion registration atomic with respect to every engine step.
        let mut core = self.core.write().await;
        core.add_request(processed)?;
        let session = core.get_session_key(&request_id).ok_or_else(|| {
            crate::error::Error::InferenceError(format!(
                "request {request_id} is missing its scheduler session"
            ))
        })?;
        self.request_controls
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(
                request_id.clone(),
                RequestControl {
                    session_epoch: session.epoch,
                    cancellation,
                    model_variant,
                },
            );
        if let Some(registration_id) = completion_registration {
            self.bind_completion_mailbox(&request_id, registration_id, session.epoch);
        }
        self.wake_notify.notify_one();

        debug!("Added request {} to engine", request_id);
        Ok(request_id)
    }

    /// Add a request to the engine for processing.
    ///
    /// The request will be validated, preprocessed, and added to the scheduler's
    /// waiting queue. Returns a request ID that can be used to track the request.
    pub async fn add_request(&self, request: EngineCoreRequest) -> Result<RequestId> {
        self.add_request_with_completion(request, None).await
    }

    /// Generate audio synchronously (blocking until complete).
    ///
    /// This is a convenience method that adds a request and waits for completion.
    pub async fn generate(&self, request: EngineCoreRequest) -> Result<EngineOutput> {
        let request_id = request.id.clone();
        let (registration, mut completion) =
            self.register_completion_mailbox(request_id.clone())?;
        self.add_request_with_completion(request, Some(registration.registration_id))
            .await?;
        let mut idle_backoff_ms = 1u64;

        // Run steps until this request completes
        loop {
            let outputs = tokio::select! {
                biased;
                completion = &mut completion => {
                    let output = completion.map_err(|_| {
                        crate::error::Error::InferenceError(format!(
                            "Completion mailbox for {request_id} closed before delivery"
                        ))
                    })?;
                    return Self::resolve_generation_output(&request_id, output);
                }
                outputs = self.step() => outputs?,
            };
            let step_was_idle = outputs.is_empty();

            // Check if request is still in the system
            let core = self.core.read().await;
            if !core.has_request(&request_id) && !core.has_pending_terminal_output(&request_id) {
                drop(core);
                return match completion.try_recv() {
                    Ok(output) => Self::resolve_generation_output(&request_id, output),
                    Err(oneshot::error::TryRecvError::Closed) => {
                        Err(crate::error::Error::InferenceError(format!(
                            "Completion mailbox for {request_id} closed before delivery"
                        )))
                    }
                    Err(oneshot::error::TryRecvError::Empty) => {
                        Err(crate::error::Error::InferenceError(format!(
                            "Request {request_id} was removed unexpectedly"
                        )))
                    }
                };
            }
            drop(core);

            if step_was_idle {
                tokio::select! {
                    biased;
                    completion = &mut completion => {
                        let output = completion.map_err(|_| {
                            crate::error::Error::InferenceError(format!(
                                "Completion mailbox for {request_id} closed before delivery"
                            ))
                        })?;
                        return Self::resolve_generation_output(&request_id, output);
                    }
                    _ = self.wake_notify.notified() => {},
                    _ = tokio::time::sleep(tokio::time::Duration::from_millis(idle_backoff_ms)) => {},
                }
                idle_backoff_ms = idle_backoff_ms.saturating_mul(2).min(50);
            } else {
                idle_backoff_ms = 1;
            }
        }
    }

    /// Generate audio with streaming output.
    ///
    /// Returns a channel receiver that will receive audio chunks as they're generated.
    pub async fn generate_streaming(
        &self,
        request: EngineCoreRequest,
    ) -> Result<(RequestId, mpsc::Receiver<StreamingOutput>)> {
        let capacity = Self::streaming_queue_capacity(&request);
        let (tx, rx) = mpsc::channel(capacity);
        let request_id = request.id.clone();

        // Add request with streaming callback
        let mut streaming_request = request;
        streaming_request.streaming = true;
        streaming_request.streaming_tx = Some(tx);

        self.add_request(streaming_request).await?;

        Ok((request_id, rx))
    }

    /// Execute one step of the inference loop.
    ///
    /// This is the core loop that:
    /// 1. Schedules requests (decides what to process this step)
    /// 2. Runs forward pass on scheduled requests
    /// 3. Processes outputs (sampling, stop conditions)
    ///
    /// Returns outputs for any completed or streaming requests.
    pub async fn step(&self) -> Result<Vec<EngineOutput>> {
        let mut core = self.core.write().await;
        let outputs = core.step().await?;

        // Keep every await before terminal dispatch. Once a completion sender
        // is notified, routing and exact-session acknowledgement must finish
        // synchronously so cancelling a competing generate future cannot leave
        // the delivery half-committed.
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_steps += 1;
            metrics.requests_processed += outputs.len() as u64;
        }

        if outputs.iter().any(|output| output.is_finished) {
            for output in &outputs {
                if output.is_finished {
                    let session = SessionKey::new(output.request_id.clone(), output.sequence_id);
                    if let Some(sender) = self.take_completion_sender(&session) {
                        // A dropped receiver is a completed routing attempt: no
                        // live caller remains, so the public ID must not stay
                        // fenced forever.
                        let _ = sender.send(output.clone());
                    }

                    // Acknowledge only after the exact-session mailbox has been
                    // routed (or the output has been placed in this step's
                    // return batch for callers without a mailbox).
                    if !core.acknowledge_terminal_output(&session) {
                        warn!(
                            request_id = %session.request_id,
                            session_epoch = session.epoch,
                            "Terminal output had no matching delivery fence"
                        );
                    }

                    let mut controls = self
                        .request_controls
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner());
                    let owns_output = controls
                        .get(&output.request_id)
                        .is_some_and(|control| control.session_epoch == output.sequence_id);
                    if owns_output {
                        controls.remove(&output.request_id);
                    }
                }
            }
        }

        Ok(outputs)
    }

    /// Run the engine continuously, processing requests as they arrive.
    ///
    /// This should be called in a separate task. It will run until `stop()` is called.
    pub async fn run(&self) -> Result<()> {
        use std::sync::atomic::Ordering;

        self.running.store(true, Ordering::SeqCst);
        info!("Engine started");
        let mut idle_backoff_ms = 1u64;

        while self.running.load(Ordering::SeqCst) {
            // Check if there are requests to process
            let has_work = {
                let core = self.core.read().await;
                core.has_pending_work()
            };

            if has_work {
                match self.step().await {
                    Ok(outputs) if outputs.is_empty() => {
                        tokio::select! {
                            _ = self.wake_notify.notified() => {},
                            _ = tokio::time::sleep(tokio::time::Duration::from_millis(idle_backoff_ms)) => {},
                        }
                        idle_backoff_ms = idle_backoff_ms.saturating_mul(2).min(50);
                    }
                    Ok(_) => idle_backoff_ms = 1,
                    Err(e) => {
                        warn!("Engine step error: {}", e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(idle_backoff_ms))
                            .await;
                        idle_backoff_ms = idle_backoff_ms.saturating_mul(2).min(50);
                    }
                }
            } else {
                // Event-driven wait to avoid hot polling on local/edge devices.
                tokio::select! {
                    _ = self.wake_notify.notified() => {},
                    _ = tokio::time::sleep(tokio::time::Duration::from_millis(50)) => {},
                }
                idle_backoff_ms = 1;
            }
        }

        info!("Engine stopped");
        Ok(())
    }

    /// Stop the engine.
    pub fn stop(&self) {
        use std::sync::atomic::Ordering;
        self.running.store(false, Ordering::SeqCst);
        self.wake_notify.notify_waiters();
    }

    /// Check if the engine is running.
    pub fn is_running(&self) -> bool {
        use std::sync::atomic::Ordering;
        self.running.load(Ordering::SeqCst)
    }

    /// Get engine metrics.
    pub async fn metrics(&self) -> EngineMetrics {
        self.metrics.read().await.clone()
    }

    /// Get current configuration.
    pub fn config(&self) -> &EngineCoreConfig {
        &self.config
    }

    /// Abort a specific request.
    pub async fn abort_request(&self, request_id: &RequestId) -> Result<bool> {
        if let Some(control) = self
            .request_controls
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(request_id)
        {
            control
                .cancellation
                .store(true, std::sync::atomic::Ordering::Release);
        }
        let mut core = self.core.write().await;
        let aborted = core.abort_request(request_id).await;
        drop(core);
        if aborted {
            self.request_controls
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .remove(request_id);
            self.wake_notify.notify_one();
        }
        Ok(aborted)
    }

    /// Read the session-fenced identity for the active request ID.
    pub async fn request_session_key(&self, request_id: &RequestId) -> Option<SessionKey> {
        self.core.read().await.get_session_key(request_id)
    }

    /// Abort only the request incarnation named by `session`.
    pub async fn abort_request_session(&self, session: &SessionKey) -> Result<bool> {
        if let Some(control) = self
            .request_controls
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(&session.request_id)
        {
            if control.session_epoch == session.epoch {
                control
                    .cancellation
                    .store(true, std::sync::atomic::Ordering::Release);
            }
        }
        let mut core = self.core.write().await;
        let aborted = core.abort_request_session(session).await;
        drop(core);
        if aborted {
            let mut controls = self
                .request_controls
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if controls
                .get(&session.request_id)
                .is_some_and(|control| control.session_epoch == session.epoch)
            {
                controls.remove(&session.request_id);
            }
            drop(controls);
            self.wake_notify.notify_one();
        }
        Ok(aborted)
    }

    /// Abort all requests currently routed to a specific model variant.
    pub async fn abort_requests_for_variant(&self, variant: ModelVariant) -> Vec<RequestId> {
        {
            let controls = self
                .request_controls
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            for control in controls.values() {
                if control.model_variant == Some(variant) {
                    control
                        .cancellation
                        .store(true, std::sync::atomic::Ordering::Release);
                }
            }
        }
        let mut core = self.core.write().await;
        let aborted = core.abort_requests_for_variant(variant).await;
        self.request_controls
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .retain(|_, control| control.model_variant != Some(variant));
        if !aborted.is_empty() {
            self.wake_notify.notify_one();
        }
        aborted
    }

    /// Abort every request currently tracked by the engine.
    pub async fn abort_all_requests(&self) -> Vec<RequestId> {
        {
            let controls = self
                .request_controls
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            for control in controls.values() {
                control
                    .cancellation
                    .store(true, std::sync::atomic::Ordering::Release);
            }
        }
        let mut core = self.core.write().await;
        let aborted = core.abort_all_requests().await;
        self.request_controls
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clear();
        if !aborted.is_empty() {
            self.wake_notify.notify_one();
        }
        aborted
    }

    /// Check if a request is still tracked by the engine core.
    pub async fn has_request(&self, request_id: &RequestId) -> bool {
        let core = self.core.read().await;
        core.has_request(request_id)
    }

    /// Get model variants currently referenced by active engine requests.
    pub async fn active_model_variants(&self) -> HashSet<ModelVariant> {
        let core = self.core.read().await;
        core.active_model_variants()
    }

    /// Get the number of pending requests.
    pub async fn pending_requests(&self) -> usize {
        let core = self.core.read().await;
        core.pending_request_count()
    }

    /// Get the number of running requests.
    pub async fn running_requests(&self) -> usize {
        let core = self.core.read().await;
        core.running_request_count()
    }

    /// Get KV cache statistics.
    pub async fn kv_cache_stats(&self) -> KVCacheStats {
        let core = self.core.read().await;
        core.kv_cache_stats()
    }

    /// Check if scheduler currently has runnable or queued work.
    pub async fn has_pending_work(&self) -> bool {
        let core = self.core.read().await;
        core.has_pending_work()
    }
}

#[cfg(test)]
mod tests {
    use super::scheduler::ScheduledRequest;
    use super::*;
    use crate::backends::BackendKind;
    use crate::error::Error;

    struct EndlessSequenceExecutor;

    impl EndlessSequenceExecutor {
        fn outputs(scheduled: &[ScheduledRequest]) -> Vec<ExecutorStepResult> {
            scheduled
                .iter()
                .map(|entry| {
                    ExecutorStepResult::new(
                        entry,
                        ExecutorOutput {
                            request_id: entry.request_id.clone(),
                            audio: None,
                            text: None,
                            input_transcription: None,
                            tokens_processed: usize::from(entry.is_prefill) * entry.num_tokens,
                            tokens_generated: usize::from(!entry.is_prefill),
                            finished: false,
                            phase_timing_override: None,
                            asr_diagnostics: None,
                            error: None,
                        },
                    )
                })
                .collect()
        }
    }

    impl ModelExecutor for EndlessSequenceExecutor {
        fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
            let mut profile = ExecutionProfile::fail_closed(
                BackendKind::Cpu,
                request.model_variant,
                ExecutionMode::Sequence,
            );
            profile.prefill = PrefillMode::Full;
            Some(profile)
        }

        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(Self::outputs(scheduled))
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(Self::outputs(scheduled))
        }

        fn is_ready(&self) -> bool {
            true
        }

        fn initialize(&mut self) -> Result<()> {
            Ok(())
        }

        fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }

        fn cleanup_request(&self, _request_id: &str) -> executor::CacheReleaseReport {
            executor::CacheReleaseReport::confirmed(1)
        }
    }

    struct ImmediateTerminalExecutor {
        max_batch_width: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl ImmediateTerminalExecutor {
        fn new(max_batch_width: Arc<std::sync::atomic::AtomicUsize>) -> Self {
            Self { max_batch_width }
        }

        fn outputs(&self, scheduled: &[ScheduledRequest]) -> Vec<ExecutorStepResult> {
            use std::sync::atomic::Ordering;

            self.max_batch_width
                .fetch_max(scheduled.len(), Ordering::Relaxed);
            scheduled
                .iter()
                .map(|entry| {
                    ExecutorStepResult::new(
                        entry,
                        ExecutorOutput {
                            request_id: entry.request_id.clone(),
                            audio: None,
                            text: Some(format!("done-{}", entry.request_id)),
                            input_transcription: None,
                            tokens_processed: entry.num_tokens.max(1),
                            tokens_generated: 1,
                            finished: true,
                            phase_timing_override: None,
                            asr_diagnostics: None,
                            error: None,
                        },
                    )
                    .with_dispatch(BatchDispatch::new(
                        BatchDispatchKind::TensorStatic,
                        scheduled.len(),
                    ))
                })
                .collect()
        }
    }

    impl ModelExecutor for ImmediateTerminalExecutor {
        fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
            let mut profile = ExecutionProfile::fail_closed(
                BackendKind::Cpu,
                request.model_variant,
                ExecutionMode::Sequence,
            );
            profile.prefill = PrefillMode::Full;
            profile.prefill_batch = NativeBatchMode::Static;
            profile.decode_batch = NativeBatchMode::Static;
            profile.concurrency = ConcurrencyClass::Batchable;
            profile.max_batch_size = 8;
            profile.resolved_from_loaded_model = true;
            Some(profile)
        }

        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(self.outputs(scheduled))
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(self.outputs(scheduled))
        }

        fn is_ready(&self) -> bool {
            true
        }

        fn initialize(&mut self) -> Result<()> {
            Ok(())
        }

        fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }

        fn cleanup_request(&self, _request_id: &str) -> executor::CacheReleaseReport {
            executor::CacheReleaseReport::confirmed(1)
        }
    }

    fn immediate_terminal_request(id: &str) -> EngineCoreRequest {
        let mut request = EngineCoreRequest::tts(format!("terminal output for {id}"));
        request.id = id.to_string();
        request.prompt_tokens = vec![1];
        request
    }

    fn engine_with_test_executor(executor: Box<dyn ModelExecutor>) -> Engine {
        let config = EngineCoreConfig::default();
        let core = EngineCore::new_with_unified_executor(
            config.clone(),
            executor::UnifiedExecutor::new_for_test(executor),
        )
        .unwrap();
        Engine {
            core: Arc::new(RwLock::new(core)),
            request_processor: RequestProcessor::new(config.clone()),
            output_processor: OutputProcessor::new(config.sample_rate),
            config,
            running: std::sync::atomic::AtomicBool::new(false),
            metrics: Arc::new(RwLock::new(EngineMetrics::default())),
            wake_notify: Arc::new(Notify::new()),
            request_controls: std::sync::Mutex::new(HashMap::new()),
            completion_mailboxes: std::sync::Mutex::new(HashMap::new()),
            next_completion_registration: std::sync::atomic::AtomicU64::new(1),
        }
    }

    #[tokio::test]
    async fn test_engine_creation() {
        let config = EngineCoreConfig::default();
        let engine = Engine::new(config);
        assert!(engine.is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_generate_callers_receive_their_own_batched_terminal_output() {
        let max_batch_width = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let engine = Arc::new(engine_with_test_executor(Box::new(
            ImmediateTerminalExecutor::new(max_batch_width.clone()),
        )));

        // Hold the core until both generate futures have installed their
        // mailboxes and queued admission. Tokio's fair write lock then admits
        // both requests before either caller can enter its first engine step.
        let admission_gate = engine.core.write().await;
        let first_engine = engine.clone();
        let first = tokio::spawn(async move {
            first_engine
                .generate(immediate_terminal_request("generate-first"))
                .await
        });
        let second_engine = engine.clone();
        let second = tokio::spawn(async move {
            second_engine
                .generate(immediate_terminal_request("generate-second"))
                .await
        });
        tokio::time::timeout(tokio::time::Duration::from_secs(1), async {
            loop {
                let mailbox_count = engine
                    .completion_mailboxes
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .len();
                if mailbox_count == 2 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("both generate callers must register before admission opens");
        drop(admission_gate);

        let first = tokio::time::timeout(tokio::time::Duration::from_secs(1), first)
            .await
            .expect("first generate timed out")
            .expect("first generate task panicked")
            .expect("first generation failed");
        let second = tokio::time::timeout(tokio::time::Duration::from_secs(1), second)
            .await
            .expect("second generate timed out")
            .expect("second generate task panicked")
            .expect("second generation failed");

        assert_eq!(first.request_id, "generate-first");
        assert_eq!(first.text.as_deref(), Some("done-generate-first"));
        assert_eq!(second.request_id, "generate-second");
        assert_eq!(second.text.as_deref(), Some("done-generate-second"));
        assert_eq!(
            max_batch_width.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "the regression requires both terminal outputs in one shared step"
        );
        assert!(
            engine
                .completion_mailboxes
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .is_empty(),
            "terminal routing must consume both exact-session mailboxes"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_dispatches_batched_terminal_outputs_to_registered_mailboxes() {
        let max_batch_width = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let engine = Arc::new(engine_with_test_executor(Box::new(
            ImmediateTerminalExecutor::new(max_batch_width.clone()),
        )));
        let first_request = immediate_terminal_request("run-first");
        let second_request = immediate_terminal_request("run-second");

        let (first_registration, first_completion) = engine
            .register_completion_mailbox(first_request.id.clone())
            .unwrap();
        engine
            .add_request_with_completion(
                first_request,
                Some(first_registration.registration_id),
            )
            .await
            .unwrap();
        let (second_registration, second_completion) = engine
            .register_completion_mailbox(second_request.id.clone())
            .unwrap();
        engine
            .add_request_with_completion(
                second_request,
                Some(second_registration.registration_id),
            )
            .await
            .unwrap();

        let run_engine = engine.clone();
        let runner = tokio::spawn(async move { run_engine.run().await });
        let (first, second) = tokio::time::timeout(
            tokio::time::Duration::from_secs(1),
            async { tokio::join!(first_completion, second_completion) },
        )
        .await
        .expect("run loop did not route both completions");
        engine.stop();
        tokio::time::timeout(tokio::time::Duration::from_secs(1), runner)
            .await
            .expect("run loop did not stop")
            .expect("run task panicked")
            .expect("run loop failed");

        let first = first.expect("first mailbox closed");
        let second = second.expect("second mailbox closed");
        assert_eq!(first.request_id, "run-first");
        assert_eq!(second.request_id, "run-second");
        assert_eq!(
            max_batch_width.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "run must route every output from the shared terminal batch"
        );

        // Exact-session acknowledgement happens after the mailboxes are routed,
        // so the public ID is reusable once delivery completes.
        engine
            .add_request(immediate_terminal_request("run-first"))
            .await
            .expect("delivered session must release its public ID fence");
        engine.abort_all_requests().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn direct_generate_returns_cancelled_after_exact_abort() {
        let engine = Arc::new(engine_with_test_executor(Box::new(EndlessSequenceExecutor)));
        let mut request = EngineCoreRequest::tts("cancel direct generation");
        request.id = "direct-generate-abort".to_string();
        request.prompt_tokens = vec![1];
        request.params.max_tokens = usize::MAX;
        let request_id = request.id.clone();
        let generating_engine = engine.clone();
        let generating = tokio::spawn(async move { generating_engine.generate(request).await });

        let session = tokio::time::timeout(tokio::time::Duration::from_secs(1), async {
            loop {
                if let Some(session) = engine.request_session_key(&request_id).await {
                    break session;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("request was not admitted");
        assert!(engine.abort_request_session(&session).await.unwrap());

        let result = tokio::time::timeout(tokio::time::Duration::from_secs(1), generating)
            .await
            .expect("generate did not observe cancellation")
            .expect("generate task panicked");
        assert!(matches!(
            result,
            Err(Error::Cancelled(id)) if id == request_id
        ));
    }

    #[test]
    fn speech_to_speech_streaming_queue_defaults_deeper_than_tts() {
        let tts_request = EngineCoreRequest::tts("hello");
        let speech_to_speech_request = EngineCoreRequest::speech_to_speech("audio");

        assert_eq!(Engine::streaming_queue_capacity(&tts_request), 8);
        assert_eq!(
            Engine::streaming_queue_capacity(&speech_to_speech_request),
            64
        );
    }

    #[test]
    fn asr_streaming_queue_default_handles_character_level_deltas() {
        let asr_request = EngineCoreRequest::asr("audio");
        assert_eq!(Engine::streaming_queue_capacity(&asr_request), 4096);
    }

    #[tokio::test]
    async fn bulk_abort_signals_and_removes_matching_request_controls() {
        use std::sync::atomic::Ordering;

        let engine = Engine::new(EngineCoreConfig::default()).unwrap();
        let mut first = EngineCoreRequest::tts("first");
        first.id = "first".to_string();
        first.model_variant = Some(ModelVariant::Qwen3Tts12Hz06BBase);
        let mut second = EngineCoreRequest::tts("second");
        second.id = "second".to_string();
        second.model_variant = Some(ModelVariant::Qwen3Tts12Hz06BCustomVoice);
        engine.add_request(first).await.unwrap();
        engine.add_request(second).await.unwrap();

        let (first_signal, second_signal) = {
            let controls = engine
                .request_controls
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            (
                controls["first"].cancellation.clone(),
                controls["second"].cancellation.clone(),
            )
        };

        assert_eq!(
            engine
                .abort_requests_for_variant(ModelVariant::Qwen3Tts12Hz06BBase)
                .await,
            vec!["first".to_string()]
        );
        assert!(first_signal.load(Ordering::Acquire));
        assert!(!second_signal.load(Ordering::Acquire));
        {
            let controls = engine
                .request_controls
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            assert!(!controls.contains_key("first"));
            assert!(controls.contains_key("second"));
        }

        assert_eq!(
            engine.abort_all_requests().await,
            vec!["second".to_string()]
        );
        assert!(second_signal.load(Ordering::Acquire));
        assert!(engine
            .request_controls
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .is_empty());
    }
}
