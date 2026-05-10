//! Runtime service orchestrator.

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use futures::FutureExt;
use tokio::sync::{broadcast, oneshot, Mutex, Notify, RwLock};
use tokio::task::yield_now;
use tracing::{debug, error, info_span};

use crate::artifacts::{DownloadProgress, ModelLifecycleSnapshot, ModelManager};
use crate::audio::{AudioCodec, AudioEncoder, StreamingConfig};
use crate::backends::{BackendPreference, BackendRouter, BackendSelectionSource, DeviceProfile};
use crate::catalog::{ModelInfo, ModelVariant};
use crate::config::EngineConfig;
use crate::engine::{
    engine_stream_backpressure_total, Engine as CoreEngine, EngineCoreConfig, EngineCoreRequest,
    EngineOutput, StreamingOutput, WorkerConfig, ENGINE_KV_CACHE_ALLOCATED_BLOCKS,
    ENGINE_KV_CACHE_EVICTIONS_TOTAL, ENGINE_KV_CACHE_HITS_TOTAL, ENGINE_KV_CACHE_MISSES_TOTAL,
    ENGINE_KV_CACHE_PREFIX_REUSE_BLOCKS_TOTAL, ENGINE_SCHEDULER_QUEUE_DEPTH,
    ENGINE_SCHEDULER_RUNNING_REQUESTS, ENGINE_STREAM_BACKPRESSURE_TOTAL,
};
use crate::error::{Error, Result};
use crate::model::ModelResidencyLease;
use crate::runtime::adapters::CapabilityKind;
use crate::runtime::adapters::RuntimeAdapterRegistry;
use crate::runtime::broker::{
    InferenceBroker, InferenceBrokerObservation, InferenceBrokerSnapshot,
};
use crate::runtime::pipeline::{PipelineExecutor, PipelineGraph};
use crate::runtime::telemetry::{
    push_engine_metric, EngineRuntimeTelemetrySnapshot, RuntimeTelemetryCollector,
    RuntimeTelemetrySnapshot,
};
use crate::runtime_models::ModelRegistry;
use crate::tokenizer::Tokenizer;

fn panic_payload_to_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        return (*msg).to_string();
    }
    if let Some(msg) = payload.downcast_ref::<String>() {
        return msg.clone();
    }
    "unknown panic payload".to_string()
}

/// Main inference engine runtime.
pub struct RuntimeService {
    pub(crate) config: EngineConfig,
    pub(crate) backend_router: BackendRouter,
    pub(crate) inference_broker: InferenceBroker,
    pub(crate) adapter_registry: RuntimeAdapterRegistry,
    pub(crate) model_manager: Arc<ModelManager>,
    pub(crate) model_registry: Arc<ModelRegistry>,
    pub(crate) tokenizer: RwLock<Option<Tokenizer>>,
    pub(crate) codec: RwLock<AudioCodec>,
    #[allow(dead_code)]
    pub(crate) streaming_config: StreamingConfig,
    pub(crate) core_engine: Arc<CoreEngine>,
    telemetry: Arc<RuntimeTelemetryCollector>,
    completion_waiters: Arc<Mutex<HashMap<String, oneshot::Sender<Result<EngineOutput>>>>>,
    step_driver_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
    step_driver_wakeup: Arc<Notify>,
    step_driver_started: AtomicBool,
    pub(crate) loaded_tts_variant: RwLock<Option<ModelVariant>>,
    pub(crate) max_loaded_models: Option<usize>,
    pub(crate) model_last_used: Arc<Mutex<HashMap<ModelVariant, u64>>>,
    pub(crate) device: DeviceProfile,
}

struct PendingRequestGuard {
    request_id: String,
    core_engine: Arc<CoreEngine>,
    completion_waiters: Arc<Mutex<HashMap<String, oneshot::Sender<Result<EngineOutput>>>>>,
    active: bool,
}

impl PendingRequestGuard {
    fn new(
        request_id: String,
        core_engine: Arc<CoreEngine>,
        completion_waiters: Arc<Mutex<HashMap<String, oneshot::Sender<Result<EngineOutput>>>>>,
    ) -> Self {
        Self {
            request_id,
            core_engine,
            completion_waiters,
            active: true,
        }
    }

    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for PendingRequestGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        let request_id = self.request_id.clone();
        let engine = self.core_engine.clone();
        let waiters = self.completion_waiters.clone();

        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let mut guard = waiters.lock().await;
                guard.remove(&request_id);
                drop(guard);

                let _ = engine.abort_request(&request_id).await;
            });
        }
    }
}

impl RuntimeService {
    pub fn backend_context(&self) -> crate::backends::BackendContext {
        self.backend_router.context().clone()
    }

    fn ensure_requested_backend_available(
        backend_context: &crate::backends::BackendContext,
    ) -> Result<()> {
        if backend_context.matches_preference() {
            return Ok(());
        }

        Err(Error::InferenceError(
            requested_backend_unavailable_message(backend_context),
        ))
    }

    /// Create a new inference engine.
    pub fn new(config: EngineConfig) -> Result<Self> {
        configure_runtime_threading(config.num_threads.max(1));
        let model_manager = Arc::new(ModelManager::new(config.clone())?);

        let backend_context =
            BackendRouter::resolve_context(config.backend, BackendSelectionSource::Config);
        let device = backend_context.device.clone();
        Self::ensure_requested_backend_available(&backend_context)?;
        let selected_backend_kind = backend_context.backend_kind;

        let model_registry = Arc::new(ModelRegistry::new(
            config.models_dir.clone(),
            device.clone(),
        ));

        let mut core_config = EngineCoreConfig::for_qwen3_tts();
        core_config.models_dir = config.models_dir.clone();
        core_config.max_batch_size = config.max_batch_size.max(1);
        core_config.max_seq_len = config.max_sequence_length.max(1);
        core_config.backend = selected_backend_kind;
        core_config.num_threads = config.num_threads.max(1);
        core_config.block_size = config.kv_page_size.max(1);
        core_config.kv_cache_dtype = config.kv_cache_dtype.clone();

        let mut worker_config = WorkerConfig::from(&core_config);
        worker_config.models_dir = config.models_dir.clone();
        worker_config.kv_cache_dtype = config.kv_cache_dtype.clone();
        worker_config.kv_page_size = config.kv_page_size.max(1);
        worker_config.model_registry = Some(model_registry.clone());
        worker_config.backend = selected_backend_kind;
        worker_config.backend_context = backend_context.clone();
        let core_engine = Arc::new(CoreEngine::new_with_worker(core_config, worker_config)?);

        Ok(Self {
            config,
            backend_router: BackendRouter::from_context(backend_context),
            inference_broker: InferenceBroker::from_env(),
            adapter_registry: RuntimeAdapterRegistry::built_in(),
            model_manager,
            model_registry,
            tokenizer: RwLock::new(None),
            codec: RwLock::new(AudioCodec::new()),
            streaming_config: StreamingConfig::default(),
            core_engine,
            telemetry: Arc::new(RuntimeTelemetryCollector::new(2048)),
            completion_waiters: Arc::new(Mutex::new(HashMap::new())),
            step_driver_task: Mutex::new(None),
            step_driver_wakeup: Arc::new(Notify::new()),
            step_driver_started: AtomicBool::new(false),
            loaded_tts_variant: RwLock::new(None),
            max_loaded_models: positive_usize_env("IZWI_MAX_LOADED_MODELS"),
            model_last_used: Arc::new(Mutex::new(HashMap::new())),
            device,
        })
    }

    /// Get reference to model manager.
    pub fn model_manager(&self) -> &Arc<ModelManager> {
        &self.model_manager
    }

    /// List available models.
    pub async fn list_models(&self) -> Vec<ModelInfo> {
        self.model_manager.list_models().await
    }

    /// Get explicit artifact and residency state for a specific model.
    pub async fn model_lifecycle_snapshot(
        &self,
        variant: ModelVariant,
    ) -> Option<ModelLifecycleSnapshot> {
        self.model_manager.lifecycle_snapshot(variant).await
    }

    /// Get explicit artifact and residency states for all known models.
    pub async fn model_lifecycle_snapshots(&self) -> Vec<ModelLifecycleSnapshot> {
        self.model_manager.lifecycle_snapshots().await
    }

    /// Snapshot of inference broker rollout state.
    pub(crate) fn inference_broker_snapshot(&self) -> InferenceBrokerSnapshot {
        self.inference_broker.snapshot()
    }

    /// Acquire a model residency lease for active runtime work.
    ///
    /// Phase 4 keeps this as observable scaffolding; unload/eviction enforcement
    /// is introduced only after direct model paths are fully wrapped.
    pub(crate) fn acquire_model_residency_lease(
        &self,
        variant: ModelVariant,
    ) -> ModelResidencyLease {
        self.model_manager.acquire_residency_lease(variant)
    }

    pub(crate) fn active_model_residency_leases(&self, variant: ModelVariant) -> usize {
        self.model_manager.active_residency_leases(variant)
    }

    fn observe_broker_request(&self, request: &EngineCoreRequest) -> Result<()> {
        let Some(observation) = self
            .inference_broker
            .observe_engine_request(request, &self.adapter_registry)
        else {
            return Ok(());
        };

        self.record_broker_observation(observation)
    }

    pub(crate) fn observe_broker_capability_request(
        &self,
        capability: CapabilityKind,
        model_variant: Option<ModelVariant>,
        streaming_required: bool,
    ) -> Result<()> {
        let Some(observation) = self.inference_broker.observe_capability_request(
            capability,
            model_variant,
            streaming_required,
            &self.adapter_registry,
        ) else {
            return Ok(());
        };

        self.record_broker_observation(observation)
    }

    fn record_broker_observation(&self, observation: InferenceBrokerObservation) -> Result<()> {
        if observation.shadow_enabled {
            self.telemetry.record_broker_shadow_request();
        }
        if observation.execution_enabled {
            self.telemetry.record_broker_execution_request();
        }

        if let Some(message) = observation.validation_error {
            self.telemetry.record_broker_validation_failure();
            if observation.execution_enabled {
                return Err(Error::InvalidInput(message));
            }
            debug!(
                capability = ?observation.capability,
                model_variant = ?observation.model_variant,
                "Inference broker shadow validation failed: {message}"
            );
        }

        Ok(())
    }

    /// Download a model.
    pub async fn download_model(&self, variant: ModelVariant) -> Result<()> {
        self.model_manager.download_model(variant).await?;
        Ok(())
    }

    /// Spawn a non-blocking background download.
    pub async fn spawn_download(
        &self,
        variant: ModelVariant,
    ) -> Result<broadcast::Receiver<DownloadProgress>> {
        self.model_manager.spawn_download(variant).await
    }

    /// Check if a download is active.
    pub async fn is_download_active(&self, variant: ModelVariant) -> bool {
        self.model_manager.is_download_active(variant).await
    }

    /// Get runtime configuration.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Get codec sample rate.
    pub async fn sample_rate(&self) -> u32 {
        self.codec.read().await.sample_rate()
    }

    /// Create audio encoder.
    pub async fn audio_encoder(&self) -> AudioEncoder {
        let codec = self.codec.read().await;
        AudioEncoder::new(codec.sample_rate(), 1)
    }

    /// Get available speakers for loaded TTS model.
    pub async fn available_speakers(&self) -> Result<Vec<String>> {
        let variant = (*self.loaded_tts_variant.read().await)
            .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;
        let _lease = self.acquire_model_residency_lease(variant);

        match variant.family() {
            crate::catalog::ModelFamily::Qwen3Tts => {
                let model = self
                    .model_registry
                    .get_qwen_tts(variant)
                    .await
                    .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;
                Ok(model.available_speakers().into_iter().cloned().collect())
            }
            crate::catalog::ModelFamily::KokoroTts => {
                let model = self
                    .model_registry
                    .get_kokoro(variant)
                    .await
                    .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;
                model.available_speakers()
            }
            crate::catalog::ModelFamily::Lfm25Audio => Ok(
                crate::models::architectures::lfm25_audio::LFM25_AUDIO_BUILT_IN_SPEAKERS
                    .iter()
                    .map(|speaker| (*speaker).to_string())
                    .collect(),
            ),
            _ => Err(Error::InferenceError(format!(
                "Model {variant} does not expose TTS speakers"
            ))),
        }
    }

    async fn ensure_step_driver_started(&self) {
        let mut guard = self.step_driver_task.lock().await;
        let restart_needed = match guard.as_ref() {
            Some(handle) if !handle.is_finished() => false,
            Some(_) => true,
            None => true,
        };

        if !restart_needed {
            self.step_driver_started.store(true, Ordering::Release);
            return;
        }

        if guard.is_some() {
            self.telemetry.record_worker_restart();
        }

        let engine = self.core_engine.clone();
        let waiters = self.completion_waiters.clone();
        let telemetry = self.telemetry.clone();
        let wakeup = self.step_driver_wakeup.clone();
        let task = tokio::spawn(async move {
            let mut idle_backoff_ms = 1u64;
            loop {
                let step_result = std::panic::AssertUnwindSafe(engine.step())
                    .catch_unwind()
                    .await;
                match step_result {
                    Ok(Ok(outputs)) => {
                        if outputs.is_empty() {
                            if engine.has_pending_work().await {
                                idle_backoff_ms = 1;
                                tokio::task::yield_now().await;
                                continue;
                            }
                            let sleep_for = tokio::time::Duration::from_millis(idle_backoff_ms);
                            tokio::select! {
                                _ = tokio::time::sleep(sleep_for) => {}
                                _ = wakeup.notified() => {}
                            }
                            idle_backoff_ms = (idle_backoff_ms.saturating_mul(2)).min(50);
                            continue;
                        }
                        idle_backoff_ms = 1;

                        for output in outputs {
                            if !output.is_finished {
                                continue;
                            }
                            telemetry.record_request_finished(&output).await;

                            let waiter = {
                                let mut w = waiters.lock().await;
                                w.remove(&output.request_id)
                            };

                            if let Some(tx) = waiter {
                                if let Some(err) = output.error.clone() {
                                    let _ = tx.send(Err(Error::InferenceError(err)));
                                } else {
                                    let _ = tx.send(Ok(output));
                                }
                            }
                        }
                    }
                    Ok(Err(err)) => {
                        let mut w = waiters.lock().await;
                        let pending: Vec<_> = w.drain().collect();
                        drop(w);
                        telemetry.record_forced_failures(pending.len());
                        for (_, tx) in pending {
                            let _ = tx.send(Err(Error::InferenceError(err.to_string())));
                        }
                        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
                    }
                    Err(payload) => {
                        let panic_message = panic_payload_to_string(payload.as_ref());
                        telemetry.record_worker_panic();
                        let mut w = waiters.lock().await;
                        let pending: Vec<_> = w.drain().collect();
                        drop(w);
                        telemetry.record_forced_failures(pending.len());
                        for (_, tx) in pending {
                            let _ = tx.send(Err(Error::InferenceError(format!(
                                "Engine worker panicked: {}",
                                panic_message
                            ))));
                        }
                        error!(
                            "Engine step worker panicked ({}); continuing with isolated loop",
                            panic_message
                        );
                        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
                    }
                }
            }
        });

        *guard = Some(task);
        self.step_driver_started.store(true, Ordering::Release);
    }

    async fn register_waiter(&self, request_id: &str) -> oneshot::Receiver<Result<EngineOutput>> {
        let (tx, rx) = oneshot::channel();
        let mut waiters = self.completion_waiters.lock().await;
        waiters.insert(request_id.to_string(), tx);
        rx
    }

    async fn remove_waiter(&self, request_id: &str) {
        let mut waiters = self.completion_waiters.lock().await;
        waiters.remove(request_id);
    }

    async fn await_completion(
        &self,
        request_id: &str,
        rx: oneshot::Receiver<Result<EngineOutput>>,
    ) -> Result<EngineOutput> {
        rx.await.map_err(|_| {
            Error::InferenceError(format!(
                "Request {} completion channel closed unexpectedly",
                request_id
            ))
        })?
    }

    pub(crate) async fn run_request(&self, request: EngineCoreRequest) -> Result<EngineOutput> {
        self.observe_broker_request(&request)?;
        let _residency_lease = request
            .model_variant
            .map(|variant| self.acquire_model_residency_lease(variant));
        self.ensure_step_driver_started().await;

        let span = info_span!(
            "runtime_request",
            request_id = %request.id,
            correlation_id = ?request.correlation_id,
            task = ?request.task_type,
            streaming = false
        );
        let _entered = span.enter();

        let request_id = request.id.clone();
        let completion_rx = self.register_waiter(&request_id).await;

        if let Err(err) = self.core_engine.add_request(request).await {
            self.remove_waiter(&request_id).await;
            return Err(err);
        }
        self.telemetry.record_request_queued().await;
        self.step_driver_wakeup.notify_one();

        let mut guard = PendingRequestGuard::new(
            request_id.clone(),
            self.core_engine.clone(),
            self.completion_waiters.clone(),
        );
        let output = self.await_completion(&request_id, completion_rx).await?;
        guard.disarm();
        Ok(output)
    }

    pub(crate) async fn run_streaming_request<F, Fut>(
        &self,
        mut request: EngineCoreRequest,
        mut on_chunk: F,
    ) -> Result<EngineOutput>
    where
        F: FnMut(StreamingOutput) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        request.streaming = true;
        self.observe_broker_request(&request)?;
        let _residency_lease = request
            .model_variant
            .map(|variant| self.acquire_model_residency_lease(variant));
        self.ensure_step_driver_started().await;

        let span = info_span!(
            "runtime_request",
            request_id = %request.id,
            correlation_id = ?request.correlation_id,
            task = ?request.task_type,
            streaming = true
        );
        let _entered = span.enter();

        let request_id = request.id.clone();
        let mut completion_rx = self.register_waiter(&request_id).await;
        let (stream_request_id, mut stream_rx) =
            match self.core_engine.generate_streaming(request).await {
                Ok(v) => v,
                Err(err) => {
                    self.remove_waiter(&request_id).await;
                    return Err(err);
                }
            };
        self.telemetry.record_request_queued().await;
        self.step_driver_wakeup.notify_one();
        debug_assert_eq!(stream_request_id, request_id);
        let mut guard = PendingRequestGuard::new(
            stream_request_id.clone(),
            self.core_engine.clone(),
            self.completion_waiters.clone(),
        );
        let mut completion_result: Option<EngineOutput> = None;

        loop {
            tokio::select! {
                maybe_chunk = stream_rx.recv() => {
                    let Some(chunk) = maybe_chunk else {
                        break;
                    };

                    if chunk.request_id != stream_request_id {
                        continue;
                    }

                    if let Err(err) = on_chunk(chunk).await {
                        self.remove_waiter(&stream_request_id).await;
                        let _ = self.core_engine.abort_request(&stream_request_id).await;
                        return Err(err);
                    }
                }
                completion = &mut completion_rx, if completion_result.is_none() => {
                    let completion = completion.map_err(|_| {
                        Error::InferenceError(format!(
                            "Request {} completion channel closed unexpectedly",
                            stream_request_id
                        ))
                    })?;

                    match completion {
                        Ok(output) => {
                            completion_result = Some(output);
                        }
                        Err(err) => {
                            // If engine worker panics, fail fast so streaming callers
                            // don't hang waiting for a chunk channel that may never close.
                            let _ = self.core_engine.abort_request(&stream_request_id).await;
                            return Err(err);
                        }
                    }
                }
            }
        }

        let output = if let Some(output) = completion_result {
            output
        } else {
            self.await_completion(&stream_request_id, completion_rx)
                .await?
        };
        guard.disarm();
        // Allow pending tasks to progress before returning to upper layers.
        yield_now().await;
        Ok(output)
    }

    /// Snapshot of runtime/engine telemetry (queue/prefill/decode/worker health).
    pub async fn telemetry_snapshot(&self) -> RuntimeTelemetrySnapshot {
        let mut snapshot = self.telemetry.snapshot().await;
        snapshot.engine = self.engine_telemetry_snapshot().await;
        snapshot
    }

    /// Prometheus exposition format telemetry payload.
    pub async fn telemetry_prometheus(&self) -> String {
        let mut payload = self.telemetry.prometheus().await;
        self.push_engine_prometheus_metrics(&mut payload).await;
        payload
    }

    async fn engine_telemetry_snapshot(&self) -> EngineRuntimeTelemetrySnapshot {
        let queue_depth = self.core_engine.pending_requests().await as u64;
        let running_requests = self.core_engine.running_requests().await as u64;
        let kv_cache = self.core_engine.kv_cache_stats().await;
        let stream_backpressure_total = engine_stream_backpressure_total();
        let kv_cache_hits_total = kv_cache.telemetry.shared_prefix_hits;
        let kv_cache_misses_total = kv_cache
            .telemetry
            .total_allocations
            .saturating_sub(kv_cache_hits_total);

        EngineRuntimeTelemetrySnapshot {
            scheduler_queue_depth: queue_depth,
            scheduler_running_requests: running_requests,
            kv_cache_hits_total,
            kv_cache_misses_total,
            kv_cache_evictions_total: kv_cache.telemetry.total_frees,
            kv_cache_allocated_blocks: kv_cache.allocated_blocks as u64,
            kv_cache_prefix_reuse_blocks_total: kv_cache_hits_total,
            stream_backpressure_total,
        }
    }

    async fn push_engine_prometheus_metrics(&self, payload: &mut String) {
        let snapshot = self.engine_telemetry_snapshot().await;
        push_engine_metric(
            payload,
            ENGINE_SCHEDULER_QUEUE_DEPTH,
            snapshot.scheduler_queue_depth,
        );
        push_engine_metric(
            payload,
            ENGINE_SCHEDULER_RUNNING_REQUESTS,
            snapshot.scheduler_running_requests,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_CACHE_HITS_TOTAL,
            snapshot.kv_cache_hits_total,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_CACHE_MISSES_TOTAL,
            snapshot.kv_cache_misses_total,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_CACHE_EVICTIONS_TOTAL,
            snapshot.kv_cache_evictions_total,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_CACHE_ALLOCATED_BLOCKS,
            snapshot.kv_cache_allocated_blocks,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_CACHE_PREFIX_REUSE_BLOCKS_TOTAL,
            snapshot.kv_cache_prefix_reuse_blocks_total,
        );
        push_engine_metric(
            payload,
            ENGINE_STREAM_BACKPRESSURE_TOTAL,
            snapshot.stream_backpressure_total,
        );
    }

    pub fn record_voice_session_started(&self) {
        self.telemetry.record_voice_session_started();
    }

    pub fn record_voice_session_closed(&self) {
        self.telemetry.record_voice_session_closed();
    }

    pub fn record_voice_interruption(&self) {
        self.telemetry.record_voice_interruption();
    }

    pub fn record_voice_barge_in(&self) {
        self.telemetry.record_voice_barge_in();
    }

    pub fn record_modular_voice_pipeline_turn(&self) {
        let graph = PipelineGraph::modular_voice_turn();
        let summary = PipelineExecutor.execute_contract(&graph);
        self.telemetry.record_pipeline_execution(&summary);
    }

    pub fn record_unified_voice_pipeline_turn(&self) {
        let graph = PipelineGraph::unified_voice_turn();
        let summary = PipelineExecutor.execute_contract(&graph);
        self.telemetry.record_pipeline_execution(&summary);
    }

    pub(crate) fn record_diarization_transcript_pipeline(&self, enable_llm_refinement: bool) {
        let graph = PipelineGraph::diarization_transcript(enable_llm_refinement);
        let summary = PipelineExecutor.execute_contract(&graph);
        self.telemetry.record_pipeline_execution(&summary);
    }
}

fn configure_runtime_threading(num_threads: usize) {
    let value = num_threads.max(1).to_string();
    for key in [
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ] {
        if std::env::var(key).is_err() {
            std::env::set_var(key, &value);
        }
    }
    debug!("Configured runtime threading hints to {} threads", value);
}

fn positive_usize_env(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn requested_backend_unavailable_message(
    backend_context: &crate::backends::BackendContext,
) -> String {
    let requested = backend_context.preference.as_str();
    let selected = backend_context.backend_kind.as_str();

    if backend_context.preference == BackendPreference::Cuda {
        let detail = if backend_context.capabilities.cuda_compiled {
            "CUDA support is compiled in, but no usable CUDA device was selected"
        } else {
            "this runtime is not compiled with CUDA support"
        };

        return format!(
            "CUDA backend was requested, but the selected backend is `{selected}`. {detail}. Use `izwi status --detailed` or `/v1/health` to inspect CUDA runtime diagnostics."
        );
    }

    format!(
        "Requested backend `{requested}` is not available on this runtime (selected `{selected}`)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{BackendCapabilities, BackendContext, BackendSelectionSource};
    use crate::runtime::broker::{InferenceBroker, InferenceBrokerMode};

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

    #[tokio::test]
    async fn runtime_prometheus_includes_engine_metric_values() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");

        let payload = runtime.telemetry_prometheus().await;

        assert!(payload.contains("izwi_engine_scheduler_queue_depth"));
        assert!(payload.contains("izwi_engine_scheduler_running_requests"));
        assert!(payload.contains("izwi_engine_kv_cache_allocated_blocks"));
        assert!(payload.contains("izwi_engine_stream_backpressure_total"));
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
        assert_eq!(snapshot.broker.validation_failures, 0);
    }
}
