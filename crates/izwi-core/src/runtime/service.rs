//! Runtime service orchestrator.

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::FutureExt;
use tokio::sync::{broadcast, oneshot, Mutex, Notify, RwLock};
use tokio::task::yield_now;
use tracing::{debug, error, info_span};

use crate::artifacts::{DownloadProgress, ModelLifecycleSnapshot, ModelManager};
use crate::audio::{AudioCodec, AudioEncoder, StreamingConfig};
use crate::backends::{
    BackendKind, BackendPreference, BackendRouter, BackendSelectionSource, DeviceProfile,
};
use crate::catalog::{ModelInfo, ModelVariant};
use crate::config::EngineConfig;
use crate::engine::{
    engine_stream_backpressure_total, Engine as CoreEngine, EngineCoreConfig, EngineCoreRequest,
    EngineOutput, ResourceAmount, ResourceVector, SessionKey, StreamingOutput, TaskType,
    WorkerConfig, WorkloadClass, ENGINE_KV_CACHE_ALLOCATED_BLOCKS, ENGINE_KV_CACHE_CHURN_RATIO,
    ENGINE_KV_CACHE_COPY_ON_WRITE_SPLITS_TOTAL, ENGINE_KV_CACHE_EVICTIONS_TOTAL,
    ENGINE_KV_CACHE_FREE_BLOCKS, ENGINE_KV_CACHE_GPU_RESIDENT_BLOCKS, ENGINE_KV_CACHE_HITS_TOTAL,
    ENGINE_KV_CACHE_MEMORY_CAPACITY_BYTES, ENGINE_KV_CACHE_MEMORY_USED_BYTES,
    ENGINE_KV_CACHE_MISSES_TOTAL, ENGINE_KV_CACHE_PINNED_BLOCKS,
    ENGINE_KV_CACHE_PREFIX_REUSE_BLOCKS_TOTAL, ENGINE_KV_CACHE_SHARED_PREFIXES,
    ENGINE_KV_CACHE_SOFT_MAX_BLOCKS, ENGINE_KV_CACHE_UTILIZATION_RATIO,
    ENGINE_SCHEDULER_QUEUE_DEPTH, ENGINE_SCHEDULER_RUNNING_REQUESTS,
    ENGINE_STREAM_BACKPRESSURE_TOTAL, REQUEST_DEADLINE_EXCEEDED,
};
use crate::error::{Error, Result};
use crate::model::ModelResidencyLease;
use crate::runtime::adapters::CapabilityKind;
use crate::runtime::adapters::RuntimeAdapterRegistry;
use crate::runtime::broker::{
    InferenceBroker, InferenceBrokerObservation, InferenceBrokerSnapshot,
};
use crate::runtime::coordinator::{
    CoordinatorLane, CoordinatorSnapshot, InferenceCoordinator, JobLease, JobSpec,
};
use crate::runtime::pipeline::{PipelineExecutor, PipelineGraph};
use crate::runtime::routing::RouteSource;
use crate::runtime::telemetry::{
    push_engine_metric, push_engine_metric_f64, EngineKvCacheRuntimeSnapshot,
    EngineRuntimeTelemetrySnapshot, RuntimeObservationContext, RuntimeStageObservation,
    RuntimeStageOutcome, RuntimeStageOutputCounters, RuntimeStageTiming, RuntimeTelemetryCollector,
    RuntimeTelemetrySnapshot,
};
use crate::runtime_models::{LoadedModelDiagnostics, ModelRegistry};
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

fn reported_gpu_resident_blocks(backend_kind: BackendKind, logical_blocks: u64) -> u64 {
    if backend_kind == BackendKind::Cpu {
        0
    } else {
        logical_blocks
    }
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
    pub(crate) coordinator: Arc<InferenceCoordinator>,
    telemetry: Arc<RuntimeTelemetryCollector>,
    completion_waiters: Arc<Mutex<HashMap<String, oneshot::Sender<Result<EngineOutput>>>>>,
    step_driver_task: Mutex<Option<tokio::task::JoinHandle<()>>>,
    step_driver_wakeup: Arc<Notify>,
    step_driver_started: AtomicBool,
    pub(crate) loaded_tts_variant: RwLock<Option<ModelVariant>>,
    pub(crate) max_loaded_models: Option<usize>,
    pub(crate) model_last_used: Arc<Mutex<HashMap<ModelVariant, u64>>>,
    pub(crate) model_load_lock: Mutex<()>,
    pub(crate) device: DeviceProfile,
}

struct PendingRequestGuard {
    session: SessionKey,
    core_engine: Arc<CoreEngine>,
    completion_waiters: Arc<Mutex<HashMap<String, oneshot::Sender<Result<EngineOutput>>>>>,
    telemetry: Arc<RuntimeTelemetryCollector>,
    job: Option<JobLease>,
    active: bool,
}

impl PendingRequestGuard {
    fn new(
        session: SessionKey,
        core_engine: Arc<CoreEngine>,
        completion_waiters: Arc<Mutex<HashMap<String, oneshot::Sender<Result<EngineOutput>>>>>,
        telemetry: Arc<RuntimeTelemetryCollector>,
        job: JobLease,
    ) -> Self {
        Self {
            session,
            core_engine,
            completion_waiters,
            telemetry,
            job: Some(job),
            active: true,
        }
    }

    fn disarm(&mut self) {
        self.active = false;
        self.job.take();
    }
}

impl Drop for PendingRequestGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }

        let session = self.session.clone();
        let engine = self.core_engine.clone();
        let waiters = self.completion_waiters.clone();
        let telemetry = self.telemetry.clone();
        let job = self.job.take();

        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let mut guard = waiters.lock().await;
                guard.remove(&session.request_id);
                drop(guard);

                let _ = engine.abort_request_session(&session).await;
                telemetry
                    .record_request_cancelled(&session.request_id)
                    .await;
                drop(job);
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
        let execution_parallelism = worker_config.request_parallelism;
        let core_engine = Arc::new(CoreEngine::new_with_worker(core_config, worker_config)?);
        let coordinator = Arc::new(InferenceCoordinator::new(
            selected_backend_kind,
            execution_parallelism,
            config.max_batch_size.max(1).saturating_mul(16).max(64),
        ));

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
            coordinator,
            telemetry: Arc::new(RuntimeTelemetryCollector::new(2048)),
            completion_waiters: Arc::new(Mutex::new(HashMap::new())),
            step_driver_task: Mutex::new(None),
            step_driver_wakeup: Arc::new(Notify::new()),
            step_driver_started: AtomicBool::new(false),
            loaded_tts_variant: RwLock::new(None),
            max_loaded_models: positive_usize_env("IZWI_MAX_LOADED_MODELS"),
            model_last_used: Arc::new(Mutex::new(HashMap::new())),
            model_load_lock: Mutex::new(()),
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

    pub fn record_stage_observation(&self, observation: RuntimeStageObservation) {
        self.telemetry.record_stage_observation(observation);
    }

    fn observe_broker_request(&self, request: &EngineCoreRequest) -> Result<()> {
        self.observe_broker_request_with_streaming_required(request, request.streaming)
    }

    fn observe_broker_request_with_streaming_required(
        &self,
        request: &EngineCoreRequest,
        streaming_required: bool,
    ) -> Result<()> {
        let Some(observation) = self
            .inference_broker
            .observe_engine_request_with_streaming_required(
                request,
                streaming_required,
                &self.adapter_registry,
                &self.backend_router,
            )
        else {
            return Ok(());
        };

        self.record_broker_observation(observation)
    }

    fn observe_broker_request_with_transport_streaming(
        &self,
        request: &EngineCoreRequest,
    ) -> Result<()> {
        self.observe_broker_request_with_streaming_required(request, false)
    }

    pub(crate) fn observe_broker_capability_request(
        &self,
        capability: CapabilityKind,
        model_variant: Option<ModelVariant>,
        streaming_required: bool,
    ) -> Result<()> {
        let Some(observation) = self.inference_broker.observe_capability_request(
            RouteSource::InternalRuntime,
            capability,
            model_variant,
            streaming_required,
            &self.adapter_registry,
            &self.backend_router,
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
        if observation.routing_decision.is_some() {
            self.telemetry.record_broker_route_decision();
        }

        if let Some(message) = observation.validation_error {
            self.telemetry.record_broker_validation_failure();
            self.telemetry.record_stage_observation(
                RuntimeStageObservation::new(
                    RuntimeObservationContext {
                        route_source: Some(format!("{:?}", observation.source)),
                        capability: Some(format!("{:?}", observation.capability)),
                        model_variant: observation
                            .model_variant
                            .map(|variant| variant.dir_name().to_string()),
                        pipeline_stage: Some("runtime.routing".to_string()),
                        ..RuntimeObservationContext::default()
                    },
                    RuntimeStageOutcome::Failed,
                )
                .with_error_kind("routing_validation_failed"),
            );
            if observation.execution_enabled {
                return Err(Error::InvalidInput(message));
            }
            debug!(
                source = ?observation.source,
                capability = ?observation.capability,
                model_variant = ?observation.model_variant,
                "Inference broker shadow validation failed: {message}"
            );
        } else if let Some(decision) = observation.routing_decision {
            self.telemetry
                .record_stage_observation(RuntimeStageObservation::new(
                    RuntimeObservationContext {
                        route_source: Some(format!("{:?}", observation.source)),
                        capability: Some(format!("{:?}", observation.capability)),
                        model_variant: Some(decision.selected_model_variant.dir_name().to_string()),
                        backend_kind: Some(decision.backend_kind.as_str().to_string()),
                        execution_target: Some(format!(
                            "{:?}",
                            decision.execution_plan.execution_target
                        )),
                        streaming_mode: Some(format!(
                            "{:?}",
                            decision.execution_plan.streaming_mode
                        )),
                        pipeline_stage: Some("runtime.routing".to_string()),
                        ..RuntimeObservationContext::default()
                    },
                    RuntimeStageOutcome::Observed,
                ));
            debug!(
                source = ?observation.source,
                capability = ?observation.capability,
                requested_model_variant = ?observation.model_variant,
                selected_model_variant = ?decision.selected_model_variant,
                execution_target = ?decision.execution_plan.execution_target,
                backend_kind = ?decision.backend_kind,
                "Inference broker route decision recorded"
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
            crate::catalog::ModelFamily::VoxtralTts => {
                let model = self
                    .model_registry
                    .get_voxtral_tts(variant)
                    .await
                    .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;
                Ok(model.available_speakers())
            }
            crate::catalog::ModelFamily::VibeVoiceTts => {
                let model = self
                    .model_registry
                    .get_vibevoice_tts(variant)
                    .await
                    .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;
                Ok(model.available_speakers())
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

    /// Machine-readable diagnostics for the currently loaded direct TTS model.
    pub async fn loaded_tts_model_diagnostics(&self) -> Option<serde_json::Value> {
        let variant = (*self.loaded_tts_variant.read().await)?;
        match variant.family() {
            crate::catalog::ModelFamily::Qwen3Tts => {
                let model = self.model_registry.get_qwen_tts(variant).await?;
                serde_json::to_value(model.diagnostics()).ok()
            }
            crate::catalog::ModelFamily::VibeVoiceTts => {
                let model = self.model_registry.get_vibevoice_tts(variant).await?;
                serde_json::to_value(model.diagnostics()).ok()
            }
            crate::catalog::ModelFamily::FishS2Tts => {
                let model = self.model_registry.get_fish_s2_tts(variant).await?;
                serde_json::to_value(model.diagnostics()).ok()
            }
            _ => None,
        }
    }

    /// Registry-backed diagnostics for native model handles loaded in memory.
    pub async fn loaded_model_diagnostics(&self) -> Vec<LoadedModelDiagnostics> {
        self.model_registry.loaded_model_diagnostics().await
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
        let coordinator = self.coordinator.clone();
        let waiters = self.completion_waiters.clone();
        let telemetry = self.telemetry.clone();
        let wakeup = self.step_driver_wakeup.clone();
        let task = tokio::spawn(async move {
            let mut idle_backoff_ms = 1u64;
            loop {
                if !engine.has_pending_work().await {
                    let sleep_for = tokio::time::Duration::from_millis(idle_backoff_ms);
                    tokio::select! {
                        _ = tokio::time::sleep(sleep_for) => {}
                        _ = wakeup.notified() => {}
                    }
                    idle_backoff_ms = (idle_backoff_ms.saturating_mul(2)).min(50);
                    continue;
                }
                let _execution = match coordinator.acquire_execution(None).await {
                    Ok(lease) => lease,
                    Err(err) => {
                        error!("Inference coordinator closed: {err}");
                        tokio::task::yield_now().await;
                        continue;
                    }
                };
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
                                    let runtime_error = if err == REQUEST_DEADLINE_EXCEEDED {
                                        Error::Timeout(output.request_id.clone())
                                    } else {
                                        Error::InferenceError(err)
                                    };
                                    let _ = tx.send(Err(runtime_error));
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
                        let request_ids: Vec<_> =
                            pending.iter().map(|(id, _)| id.as_str()).collect();
                        telemetry.record_forced_failures(request_ids).await;
                        let _ = engine.abort_all_requests().await;
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
                        let request_ids: Vec<_> =
                            pending.iter().map(|(id, _)| id.as_str()).collect();
                        telemetry.record_forced_failures(request_ids).await;
                        let _ = engine.abort_all_requests().await;
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

    async fn register_waiter(
        &self,
        request_id: &str,
    ) -> Result<oneshot::Receiver<Result<EngineOutput>>> {
        use std::collections::hash_map::Entry;

        let (tx, rx) = oneshot::channel();
        let mut waiters = self.completion_waiters.lock().await;
        match waiters.entry(request_id.to_string()) {
            Entry::Vacant(entry) => {
                entry.insert(tx);
                Ok(rx)
            }
            Entry::Occupied(_) => Err(Error::InvalidInput(format!(
                "request {request_id} already has a completion waiter"
            ))),
        }
    }

    async fn remove_waiter(&self, request_id: &str) {
        let mut waiters = self.completion_waiters.lock().await;
        waiters.remove(request_id);
    }

    async fn await_completion(
        &self,
        request_id: &str,
        rx: oneshot::Receiver<Result<EngineOutput>>,
        deadline: Option<std::time::Instant>,
    ) -> Result<EngineOutput> {
        let completion = async {
            rx.await.map_err(|_| {
                Error::InferenceError(format!(
                    "Request {} completion channel closed unexpectedly",
                    request_id
                ))
            })?
        };
        match deadline {
            Some(deadline) => tokio::time::timeout_at(deadline.into(), completion)
                .await
                .map_err(|_| Error::Timeout(request_id.to_string()))?,
            None => completion.await,
        }
    }

    fn engine_observation_context(
        &self,
        request: &EngineCoreRequest,
        streaming: bool,
    ) -> RuntimeObservationContext {
        RuntimeObservationContext {
            route_source: Some(format!("{:?}", RouteSource::InternalEngine)),
            capability: Some(capability_name_for_task(request.task_type).to_string()),
            model_variant: request
                .model_variant
                .map(|variant| variant.dir_name().to_string()),
            backend_kind: Some(
                self.backend_router
                    .default_backend()
                    .kind()
                    .as_str()
                    .to_string(),
            ),
            pipeline_stage: Some(if streaming {
                "engine.streaming_request".to_string()
            } else {
                "engine.request".to_string()
            }),
            workload_class: Some(request.workload_class.as_str().to_string()),
            request_id: Some(request.id.clone()),
            correlation_id: request.correlation_id.clone(),
            ..RuntimeObservationContext::default()
        }
    }

    fn record_engine_output_observation(
        &self,
        request: &EngineCoreRequest,
        output: &EngineOutput,
        streaming: bool,
    ) {
        let mut timing = RuntimeStageTiming {
            admission_ms: request.admission_ms,
            total_ms: Some(output.generation_time.as_secs_f64() * 1000.0),
            ..RuntimeStageTiming::default()
        };
        if let Some(latency) = output.latency_breakdown.as_ref() {
            timing.queue_wait_ms = Some(latency.queue_wait_ms);
            timing.media_decode_ms = latency.media_decode_ms;
            timing.normalization_ms = latency.normalization_ms;
            timing.prefill_ms = Some(latency.prefill_ms);
            timing.decode_ms = Some(latency.decode_ms);
            timing.ttft_ms = latency.ttft_ms;
            timing.sampling_ms = latency.sampling_ms;
            timing.codec_ms = latency.codec_ms;
            timing.postprocess_ms = latency.postprocess_ms;
            timing.total_ms = Some(latency.total_ms);
        }

        let outcome = if output.error.is_some() {
            RuntimeStageOutcome::Failed
        } else {
            RuntimeStageOutcome::Completed
        };
        let mut observation = RuntimeStageObservation::new(
            self.engine_observation_context(request, streaming),
            outcome,
        );
        observation.timing = timing;
        observation.outputs = RuntimeStageOutputCounters {
            prompt_tokens: Some(output.token_stats.prompt_tokens as u64),
            generated_tokens: Some(output.token_stats.generated_tokens as u64),
            audio_samples: Some(output.audio.samples.len() as u64),
            transcript_chars: output.text.as_ref().map(|text| text.chars().count() as u64),
            stop_reason: output.finish_reason.map(|reason| format!("{reason:?}")),
            ..RuntimeStageOutputCounters::default()
        };
        if let Some(error) = output.error.as_ref() {
            observation.error_kind = Some(error.clone());
        }
        self.telemetry.record_stage_observation(observation);
    }

    fn record_engine_error_observation(
        &self,
        request: &EngineCoreRequest,
        streaming: bool,
        error_kind: impl Into<String>,
    ) {
        let mut observation = RuntimeStageObservation::new(
            self.engine_observation_context(request, streaming),
            RuntimeStageOutcome::Failed,
        )
        .with_error_kind(error_kind);
        observation.timing.admission_ms = request.admission_ms;
        self.telemetry.record_stage_observation(observation);
    }

    fn coordinator_job_for_request(&self, request: &EngineCoreRequest) -> JobSpec {
        let input_bytes = request
            .audio_bytes
            .as_ref()
            .map(Vec::len)
            .or_else(|| request.audio_input.as_ref().map(String::len))
            .or_else(|| request.text.as_ref().map(String::len))
            .or_else(|| {
                request.chat_messages.as_ref().map(|messages| {
                    messages
                        .iter()
                        .map(|message| message.content.len())
                        .sum::<usize>()
                })
            })
            .unwrap_or_default() as u64;
        let estimated_bytes = (64 * 1024 * 1024u64).saturating_add(input_bytes.saturating_mul(8));
        let mut resources = ResourceVector::zero();
        match self.backend_router.context().backend_kind {
            BackendKind::Cpu => resources.host_bytes = ResourceAmount::Known(estimated_bytes),
            BackendKind::Metal => resources.unified_bytes = ResourceAmount::Known(estimated_bytes),
            BackendKind::Cuda => resources.device_bytes = ResourceAmount::Known(estimated_bytes),
        }
        JobSpec {
            request_id: request.id.clone(),
            lane: CoordinatorLane::Resumable,
            priority: request.priority,
            workload_class: request.workload_class,
            deadline: request.deadline,
            resources,
        }
    }

    pub(crate) async fn run_request(&self, request: EngineCoreRequest) -> Result<EngineOutput> {
        self.observe_broker_request(&request)?;
        let job = self
            .coordinator
            .admit(self.coordinator_job_for_request(&request))
            .await?;
        let observation_request = request.clone();
        let _residency_lease = request
            .model_variant
            .map(|variant| self.acquire_model_residency_lease(variant));
        self.ensure_step_driver_started().await;

        let span = info_span!(
            "runtime_request",
            request_id = %request.id,
            correlation_id = ?request.correlation_id,
            task = ?request.task_type,
            workload_class = ?request.workload_class,
            streaming = false
        );
        let _entered = span.enter();

        let request_id = request.id.clone();
        let completion_rx = self.register_waiter(&request_id).await?;

        if let Err(err) = self.core_engine.add_request(request).await {
            self.remove_waiter(&request_id).await;
            self.record_engine_error_observation(&observation_request, false, err.to_string());
            return Err(err);
        }
        self.telemetry.record_request_queued(&request_id).await;
        self.step_driver_wakeup.notify_one();

        let Some(session) = self.core_engine.request_session_key(&request_id).await else {
            self.remove_waiter(&request_id).await;
            let _ = self.core_engine.abort_request(&request_id).await;
            return Err(Error::InferenceError(format!(
                "request {request_id} is missing its scheduler session"
            )));
        };

        let mut guard = PendingRequestGuard::new(
            session,
            self.core_engine.clone(),
            self.completion_waiters.clone(),
            self.telemetry.clone(),
            job,
        );
        let completion = self
            .await_completion(&request_id, completion_rx, observation_request.deadline)
            .await;
        match completion.as_ref() {
            Ok(output) => {
                self.record_engine_output_observation(&observation_request, output, false)
            }
            Err(err) => {
                self.record_engine_error_observation(&observation_request, false, err.to_string())
            }
        }
        let output = completion?;
        guard.disarm();
        Ok(output)
    }

    pub(crate) async fn run_streaming_request<F, Fut>(
        &self,
        request: EngineCoreRequest,
        on_chunk: F,
    ) -> Result<EngineOutput>
    where
        F: FnMut(StreamingOutput) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        self.run_streaming_request_with_broker_streaming(request, on_chunk, true)
            .await
    }

    pub(crate) async fn run_transport_streaming_request<F, Fut>(
        &self,
        request: EngineCoreRequest,
        on_chunk: F,
    ) -> Result<EngineOutput>
    where
        F: FnMut(StreamingOutput) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        self.run_streaming_request_with_broker_streaming(request, on_chunk, false)
            .await
    }

    async fn run_streaming_request_with_broker_streaming<F, Fut>(
        &self,
        mut request: EngineCoreRequest,
        mut on_chunk: F,
        broker_streaming_required: bool,
    ) -> Result<EngineOutput>
    where
        F: FnMut(StreamingOutput) -> Fut,
        Fut: Future<Output = Result<()>>,
    {
        request.streaming = true;
        if request.workload_class == WorkloadClass::Online {
            request.workload_class = WorkloadClass::Streaming;
        }
        if broker_streaming_required {
            self.observe_broker_request(&request)?;
        } else {
            self.observe_broker_request_with_transport_streaming(&request)?;
        }
        let job = self
            .coordinator
            .admit(self.coordinator_job_for_request(&request))
            .await?;
        let observation_request = request.clone();
        let _residency_lease = request
            .model_variant
            .map(|variant| self.acquire_model_residency_lease(variant));
        self.ensure_step_driver_started().await;

        let span = info_span!(
            "runtime_request",
            request_id = %request.id,
            correlation_id = ?request.correlation_id,
            task = ?request.task_type,
            workload_class = ?request.workload_class,
            streaming = true
        );
        let _entered = span.enter();

        let request_id = request.id.clone();
        let mut completion_rx = self.register_waiter(&request_id).await?;
        let (stream_request_id, mut stream_rx) = match self
            .core_engine
            .generate_streaming(request)
            .await
        {
            Ok(v) => v,
            Err(err) => {
                self.remove_waiter(&request_id).await;
                self.record_engine_error_observation(&observation_request, true, err.to_string());
                return Err(err);
            }
        };
        self.telemetry.record_request_queued(&request_id).await;
        self.step_driver_wakeup.notify_one();
        debug_assert_eq!(stream_request_id, request_id);
        let Some(session) = self
            .core_engine
            .request_session_key(&stream_request_id)
            .await
        else {
            self.remove_waiter(&stream_request_id).await;
            let _ = self.core_engine.abort_request(&stream_request_id).await;
            return Err(Error::InferenceError(format!(
                "request {stream_request_id} is missing its scheduler session"
            )));
        };
        let mut guard = PendingRequestGuard::new(
            session,
            self.core_engine.clone(),
            self.completion_waiters.clone(),
            self.telemetry.clone(),
            job,
        );
        let mut completion_result: Option<EngineOutput> = None;
        let deadline = observation_request.deadline;
        let deadline_wait = async move {
            match deadline {
                Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
                None => std::future::pending::<()>().await,
            }
        };
        tokio::pin!(deadline_wait);

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
                        self.record_engine_error_observation(
                            &observation_request,
                            true,
                            err.to_string(),
                        );
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
                            self.record_engine_error_observation(
                                &observation_request,
                                true,
                                err.to_string(),
                            );
                            return Err(err);
                        }
                    }
                }
                _ = &mut deadline_wait => {
                    self.record_engine_error_observation(
                        &observation_request,
                        true,
                        "request deadline exceeded",
                    );
                    return Err(Error::Timeout(stream_request_id));
                }
            }
        }

        let output = if let Some(output) = completion_result {
            output
        } else {
            match self
                .await_completion(
                    &stream_request_id,
                    completion_rx,
                    observation_request.deadline,
                )
                .await
            {
                Ok(output) => output,
                Err(err) => {
                    self.record_engine_error_observation(
                        &observation_request,
                        true,
                        err.to_string(),
                    );
                    return Err(err);
                }
            }
        };
        self.record_engine_output_observation(&observation_request, &output, true);
        guard.disarm();
        // Allow pending tasks to progress before returning to upper layers.
        yield_now().await;
        Ok(output)
    }

    /// Snapshot of runtime/engine telemetry (queue/prefill/decode/worker health).
    pub async fn telemetry_snapshot(&self) -> RuntimeTelemetrySnapshot {
        let mut snapshot = self.telemetry.snapshot().await;
        snapshot.engine = self.engine_telemetry_snapshot().await;
        snapshot.coordinator = self.coordinator.snapshot();
        snapshot.models = self.loaded_model_diagnostics().await;
        snapshot
    }

    /// Prometheus exposition format telemetry payload.
    pub async fn telemetry_prometheus(&self) -> String {
        let mut payload = self.telemetry.prometheus().await;
        self.push_engine_prometheus_metrics(&mut payload).await;
        self.push_coordinator_prometheus_metrics(&mut payload);
        payload
    }

    pub fn coordinator_snapshot(&self) -> CoordinatorSnapshot {
        self.coordinator.snapshot()
    }

    pub fn is_draining(&self) -> bool {
        self.coordinator.is_draining()
    }

    pub fn begin_drain(&self) {
        self.coordinator.begin_drain();
        self.step_driver_wakeup.notify_waiters();
    }

    pub async fn wait_for_drain(&self, timeout: Duration) -> Result<()> {
        self.begin_drain();
        self.coordinator
            .wait_for_idle(Instant::now() + timeout)
            .await
    }

    async fn engine_telemetry_snapshot(&self) -> EngineRuntimeTelemetrySnapshot {
        let queue_depth = self.core_engine.pending_requests().await as u64;
        let running_requests = self.core_engine.running_requests().await as u64;
        let kv_cache = self.core_engine.kv_cache_stats().await;
        let stream_backpressure_total = engine_stream_backpressure_total();
        let kv_cache_hits_total = kv_cache.telemetry.shared_prefix_hits;
        let kv_cache_misses_total = kv_cache.telemetry.shared_prefix_misses;
        let backend_kind = self.backend_context().backend_kind;
        let kv_cache_snapshot = EngineKvCacheRuntimeSnapshot {
            block_accounting: "logical",
            memory_accounting: "estimated_from_config",
            total_blocks: kv_cache.total_blocks as u64,
            soft_max_blocks: kv_cache.soft_max_blocks as u64,
            allocated_blocks: kv_cache.allocated_blocks as u64,
            free_blocks: kv_cache.free_blocks as u64,
            block_size: kv_cache.block_size as u64,
            dtype_bytes: kv_cache.dtype_bytes as u64,
            block_memory_bytes: kv_cache.block_memory_bytes as u64,
            memory_used_bytes: kv_cache.memory_used_bytes as u64,
            memory_capacity_bytes: kv_cache.memory_capacity_bytes as u64,
            utilization_ratio: kv_cache.utilization(),
            gpu_resident_blocks: reported_gpu_resident_blocks(
                backend_kind,
                kv_cache.gpu_resident_blocks as u64,
            ),
            pinned_blocks: kv_cache.pinned_blocks as u64,
            shared_prefixes: kv_cache.shared_prefixes as u64,
            total_allocations: kv_cache.telemetry.total_allocations,
            total_frees: kv_cache.telemetry.total_frees,
            shared_prefix_hits: kv_cache.telemetry.shared_prefix_hits,
            shared_prefix_misses: kv_cache.telemetry.shared_prefix_misses,
            shared_prefix_blocks_reused: kv_cache.telemetry.shared_prefix_blocks_reused,
            persistent_prefix_evictions: kv_cache.telemetry.persistent_prefix_evictions,
            copy_on_write_splits: kv_cache.telemetry.copy_on_write_splits,
            last_churn_ratio: kv_cache.telemetry.last_churn_ratio,
        };

        EngineRuntimeTelemetrySnapshot {
            scheduler_queue_depth: queue_depth,
            scheduler_running_requests: running_requests,
            kv_cache_hits_total,
            kv_cache_misses_total,
            kv_cache_evictions_total: kv_cache.telemetry.persistent_prefix_evictions,
            kv_cache_allocated_blocks: kv_cache.allocated_blocks as u64,
            kv_cache_prefix_reuse_blocks_total: kv_cache.telemetry.shared_prefix_blocks_reused,
            stream_backpressure_total,
            kv_cache: kv_cache_snapshot,
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
            ENGINE_KV_CACHE_FREE_BLOCKS,
            snapshot.kv_cache.free_blocks,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_CACHE_SOFT_MAX_BLOCKS,
            snapshot.kv_cache.soft_max_blocks,
        );
        push_engine_metric_f64(
            payload,
            ENGINE_KV_CACHE_UTILIZATION_RATIO,
            snapshot.kv_cache.utilization_ratio,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_CACHE_MEMORY_USED_BYTES,
            snapshot.kv_cache.memory_used_bytes,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_CACHE_MEMORY_CAPACITY_BYTES,
            snapshot.kv_cache.memory_capacity_bytes,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_CACHE_SHARED_PREFIXES,
            snapshot.kv_cache.shared_prefixes,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_CACHE_PREFIX_REUSE_BLOCKS_TOTAL,
            snapshot.kv_cache_prefix_reuse_blocks_total,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_CACHE_COPY_ON_WRITE_SPLITS_TOTAL,
            snapshot.kv_cache.copy_on_write_splits,
        );
        push_engine_metric_f64(
            payload,
            ENGINE_KV_CACHE_CHURN_RATIO,
            snapshot.kv_cache.last_churn_ratio,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_CACHE_GPU_RESIDENT_BLOCKS,
            snapshot.kv_cache.gpu_resident_blocks,
        );
        push_engine_metric(
            payload,
            ENGINE_KV_CACHE_PINNED_BLOCKS,
            snapshot.kv_cache.pinned_blocks,
        );
        push_engine_metric(
            payload,
            ENGINE_STREAM_BACKPRESSURE_TOTAL,
            snapshot.stream_backpressure_total,
        );
    }

    fn push_coordinator_prometheus_metrics(&self, payload: &mut String) {
        let snapshot = self.coordinator.snapshot();
        payload.push_str(&format!(
            "# TYPE izwi_inference_coordinator_capacity gauge\n\
izwi_inference_coordinator_capacity {}\n\
# TYPE izwi_inference_coordinator_active_jobs gauge\n\
izwi_inference_coordinator_active_jobs {}\n\
# TYPE izwi_inference_coordinator_active_executions gauge\n\
izwi_inference_coordinator_active_executions {}\n\
# TYPE izwi_inference_coordinator_reserved_memory_bytes gauge\n\
izwi_inference_coordinator_reserved_memory_bytes {}\n\
# TYPE izwi_inference_coordinator_admitted_total counter\n\
izwi_inference_coordinator_admitted_total {}\n\
# TYPE izwi_inference_coordinator_rejected_total counter\n\
izwi_inference_coordinator_rejected_total {}\n\
# TYPE izwi_inference_coordinator_expired_total counter\n\
izwi_inference_coordinator_expired_total {}\n\
# TYPE izwi_inference_coordinator_draining gauge\n\
izwi_inference_coordinator_draining {}\n",
            snapshot.capacity,
            snapshot.active_jobs,
            snapshot.active_executions,
            snapshot.reserved_memory_bytes,
            snapshot.admitted_total,
            snapshot.rejected_total,
            snapshot.expired_total,
            u8::from(snapshot.draining),
        ));
    }

    pub fn record_voice_session_started(&self) {
        self.telemetry.record_voice_session_started();
        self.record_voice_stage_observation("voice.session_started");
    }

    pub fn record_voice_session_closed(&self) {
        self.telemetry.record_voice_session_closed();
        self.record_voice_stage_observation("voice.session_closed");
    }

    pub fn record_voice_interruption(&self) {
        self.telemetry.record_voice_interruption();
        self.record_voice_stage_observation("voice.interruption");
    }

    pub fn record_voice_barge_in(&self) {
        self.telemetry.record_voice_barge_in();
        self.record_voice_stage_observation("voice.barge_in");
    }

    pub fn record_voice_stream_backpressure(&self) {
        self.telemetry.record_voice_stream_backpressure();
        self.record_voice_stage_observation("voice.stream_backpressure");
    }

    pub fn record_transcription_stream_backpressure(&self) {
        self.telemetry.record_transcription_stream_backpressure();
        self.telemetry
            .record_stage_observation(RuntimeStageObservation::new(
                RuntimeObservationContext {
                    route_source: Some("realtime_transcription".to_string()),
                    capability: Some("asr".to_string()),
                    pipeline_kind: Some("realtime_transcription".to_string()),
                    pipeline_stage: Some("transcription.stream_backpressure".to_string()),
                    ..RuntimeObservationContext::default()
                },
                RuntimeStageOutcome::Observed,
            ));
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

    pub fn record_batch_asr_pipeline_job(&self) {
        let graph = PipelineGraph::batch_asr_transcription();
        let summary = PipelineExecutor.execute_contract(&graph);
        self.telemetry.record_pipeline_execution(&summary);
    }

    pub fn record_batch_tts_pipeline_job(&self) {
        let graph = PipelineGraph::batch_tts_speech();
        let summary = PipelineExecutor.execute_contract(&graph);
        self.telemetry.record_pipeline_execution(&summary);
    }

    fn record_voice_stage_observation(&self, pipeline_stage: &'static str) {
        self.telemetry
            .record_stage_observation(RuntimeStageObservation::new(
                RuntimeObservationContext {
                    route_source: Some(format!("{:?}", RouteSource::RealtimeVoice)),
                    capability: Some(format!("{:?}", CapabilityKind::SpeechToSpeech)),
                    pipeline_kind: Some("realtime_voice".to_string()),
                    pipeline_stage: Some(pipeline_stage.to_string()),
                    ..RuntimeObservationContext::default()
                },
                RuntimeStageOutcome::Observed,
            ));
    }
}

fn capability_name_for_task(task_type: TaskType) -> &'static str {
    match task_type {
        TaskType::TTS => "tts",
        TaskType::ASR => "asr",
        TaskType::Chat => "chat",
        TaskType::SpeechToSpeech => "speech_to_speech",
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

    #[tokio::test]
    async fn duplicate_waiter_registration_preserves_original_owner() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");
        let original = runtime
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
        runtime.remove_waiter("same-request").await;
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

    #[tokio::test]
    async fn runtime_prometheus_includes_engine_metric_values() {
        let runtime = RuntimeService::new(EngineConfig::default()).expect("runtime");

        let payload = runtime.telemetry_prometheus().await;

        assert!(payload.contains("izwi_engine_scheduler_queue_depth"));
        assert!(payload.contains("izwi_engine_scheduler_running_requests"));
        assert!(payload.contains("izwi_engine_kv_cache_allocated_blocks"));
        assert!(payload.contains("izwi_engine_kv_cache_soft_max_blocks"));
        assert!(payload.contains("izwi_engine_kv_cache_utilization_ratio"));
        assert!(payload.contains("izwi_engine_kv_cache_copy_on_write_splits_total"));
        assert!(payload.contains("allocated logical KV-cache blocks"));
        assert!(payload.contains("Estimated KV-cache bytes"));
        assert!(payload.contains("izwi_engine_stream_backpressure_total"));
        assert!(payload.contains("# TYPE izwi_inference_coordinator_active_jobs gauge"));
        assert!(payload.contains("# TYPE izwi_inference_coordinator_admitted_total counter"));
        assert!(payload.contains("izwi_inference_coordinator_reserved_memory_bytes"));
        assert!(payload.contains("izwi_inference_coordinator_draining 0"));

        let snapshot = runtime.telemetry_snapshot().await;
        assert_eq!(snapshot.coordinator, runtime.coordinator_snapshot());
        assert!(snapshot.engine.kv_cache.total_blocks > 0);
        assert!(snapshot.engine.kv_cache.block_size > 0);
        assert_eq!(snapshot.engine.kv_cache.block_accounting, "logical");
        assert_eq!(
            snapshot.engine.kv_cache.memory_accounting,
            "estimated_from_config"
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
    fn cpu_backend_never_reports_logical_blocks_as_gpu_resident() {
        assert_eq!(reported_gpu_resident_blocks(BackendKind::Cpu, 17), 0);
        assert_eq!(reported_gpu_resident_blocks(BackendKind::Cuda, 17), 17);
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
}
