//! Runtime service orchestrator.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use futures::FutureExt;
use serde::Serialize;
use tokio::sync::{broadcast, oneshot, Mutex, Notify, RwLock};
use tokio::task::yield_now;
use tracing::{debug, error, info_span};

use crate::artifacts::{DownloadProgress, ModelManager};
use crate::audio::{AudioCodec, AudioEncoder, StreamingConfig};
use crate::backends::{
    BackendPreference, BackendRouter, BackendSelectionSource, DeviceKind, DeviceProfile,
};
use crate::catalog::{ModelInfo, ModelVariant};
use crate::config::EngineConfig;
use crate::engine::{
    Engine as CoreEngine, EngineCoreConfig, EngineCoreRequest, EngineOutput, StreamingOutput,
    WorkerConfig,
};
use crate::error::{Error, Result};
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

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeTelemetrySnapshot {
    pub uptime_secs: f64,
    pub requests_queued: u64,
    pub requests_completed: u64,
    pub requests_failed: u64,
    pub requests_active: u64,
    pub worker_restarts: u64,
    pub worker_panics: u64,
    pub queue_wait_ms_avg: f64,
    pub queue_wait_ms_p50: f64,
    pub queue_wait_ms_p95: f64,
    pub prefill_ms_avg: f64,
    pub prefill_ms_p50: f64,
    pub prefill_ms_p95: f64,
    pub decode_ms_avg: f64,
    pub decode_ms_p50: f64,
    pub decode_ms_p95: f64,
    pub end_to_end_ms_avg: f64,
    pub end_to_end_ms_p50: f64,
    pub end_to_end_ms_p95: f64,
}

#[derive(Debug)]
struct RuntimeTelemetryCollector {
    start_time: Instant,
    max_samples: usize,
    requests_queued: AtomicU64,
    requests_completed: AtomicU64,
    requests_failed: AtomicU64,
    requests_active: AtomicU64,
    worker_restarts: AtomicU64,
    worker_panics: AtomicU64,
    queue_wait_ms_samples: Mutex<VecDeque<f64>>,
    prefill_ms_samples: Mutex<VecDeque<f64>>,
    decode_ms_samples: Mutex<VecDeque<f64>>,
    end_to_end_ms_samples: Mutex<VecDeque<f64>>,
}

impl RuntimeTelemetryCollector {
    fn new(max_samples: usize) -> Self {
        Self {
            start_time: Instant::now(),
            max_samples: max_samples.max(64),
            requests_queued: AtomicU64::new(0),
            requests_completed: AtomicU64::new(0),
            requests_failed: AtomicU64::new(0),
            requests_active: AtomicU64::new(0),
            worker_restarts: AtomicU64::new(0),
            worker_panics: AtomicU64::new(0),
            queue_wait_ms_samples: Mutex::new(VecDeque::with_capacity(max_samples.max(64))),
            prefill_ms_samples: Mutex::new(VecDeque::with_capacity(max_samples.max(64))),
            decode_ms_samples: Mutex::new(VecDeque::with_capacity(max_samples.max(64))),
            end_to_end_ms_samples: Mutex::new(VecDeque::with_capacity(max_samples.max(64))),
        }
    }

    async fn record_request_queued(&self) {
        self.requests_queued.fetch_add(1, Ordering::Relaxed);
        self.requests_active.fetch_add(1, Ordering::Relaxed);
    }

    async fn record_request_finished(&self, output: &EngineOutput) {
        self.requests_completed.fetch_add(1, Ordering::Relaxed);
        if output.error.is_some() {
            self.requests_failed.fetch_add(1, Ordering::Relaxed);
        }
        self.requests_active.fetch_sub(1, Ordering::Relaxed);

        if let Some(latency) = output.latency_breakdown.as_ref() {
            Self::push_sample(
                &self.queue_wait_ms_samples,
                self.max_samples,
                latency.queue_wait_ms,
            )
            .await;
            Self::push_sample(
                &self.prefill_ms_samples,
                self.max_samples,
                latency.prefill_ms,
            )
            .await;
            Self::push_sample(&self.decode_ms_samples, self.max_samples, latency.decode_ms).await;
            Self::push_sample(
                &self.end_to_end_ms_samples,
                self.max_samples,
                latency.total_ms,
            )
            .await;
        } else {
            Self::push_sample(
                &self.end_to_end_ms_samples,
                self.max_samples,
                output.generation_time.as_secs_f64() * 1000.0,
            )
            .await;
        }
    }

    fn record_forced_failures(&self, count: usize) {
        if count == 0 {
            return;
        }
        let count_u64 = count as u64;
        self.requests_completed
            .fetch_add(count_u64, Ordering::Relaxed);
        self.requests_failed.fetch_add(count_u64, Ordering::Relaxed);
        let _ = self
            .requests_active
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(count_u64))
            });
    }

    fn record_worker_restart(&self) {
        self.worker_restarts.fetch_add(1, Ordering::Relaxed);
    }

    fn record_worker_panic(&self) {
        self.worker_panics.fetch_add(1, Ordering::Relaxed);
    }

    async fn snapshot(&self) -> RuntimeTelemetrySnapshot {
        let queue = self.queue_wait_ms_samples.lock().await.clone();
        let prefill = self.prefill_ms_samples.lock().await.clone();
        let decode = self.decode_ms_samples.lock().await.clone();
        let end_to_end = self.end_to_end_ms_samples.lock().await.clone();

        RuntimeTelemetrySnapshot {
            uptime_secs: self.start_time.elapsed().as_secs_f64(),
            requests_queued: self.requests_queued.load(Ordering::Relaxed),
            requests_completed: self.requests_completed.load(Ordering::Relaxed),
            requests_failed: self.requests_failed.load(Ordering::Relaxed),
            requests_active: self.requests_active.load(Ordering::Relaxed),
            worker_restarts: self.worker_restarts.load(Ordering::Relaxed),
            worker_panics: self.worker_panics.load(Ordering::Relaxed),
            queue_wait_ms_avg: mean(&queue),
            queue_wait_ms_p50: percentile(&queue, 0.50),
            queue_wait_ms_p95: percentile(&queue, 0.95),
            prefill_ms_avg: mean(&prefill),
            prefill_ms_p50: percentile(&prefill, 0.50),
            prefill_ms_p95: percentile(&prefill, 0.95),
            decode_ms_avg: mean(&decode),
            decode_ms_p50: percentile(&decode, 0.50),
            decode_ms_p95: percentile(&decode, 0.95),
            end_to_end_ms_avg: mean(&end_to_end),
            end_to_end_ms_p50: percentile(&end_to_end, 0.50),
            end_to_end_ms_p95: percentile(&end_to_end, 0.95),
        }
    }

    async fn prometheus(&self) -> String {
        let snapshot = self.snapshot().await;
        format!(
            "# TYPE izwi_requests_queued_total counter\nizwi_requests_queued_total {}\n\
# TYPE izwi_requests_completed_total counter\nizwi_requests_completed_total {}\n\
# TYPE izwi_requests_failed_total counter\nizwi_requests_failed_total {}\n\
# TYPE izwi_requests_active gauge\nizwi_requests_active {}\n\
# TYPE izwi_worker_restarts_total counter\nizwi_worker_restarts_total {}\n\
# TYPE izwi_worker_panics_total counter\nizwi_worker_panics_total {}\n\
# TYPE izwi_latency_queue_wait_ms gauge\nizwi_latency_queue_wait_ms{{quantile=\"avg\"}} {:.6}\nizwi_latency_queue_wait_ms{{quantile=\"p50\"}} {:.6}\nizwi_latency_queue_wait_ms{{quantile=\"p95\"}} {:.6}\n\
# TYPE izwi_latency_prefill_ms gauge\nizwi_latency_prefill_ms{{quantile=\"avg\"}} {:.6}\nizwi_latency_prefill_ms{{quantile=\"p50\"}} {:.6}\nizwi_latency_prefill_ms{{quantile=\"p95\"}} {:.6}\n\
# TYPE izwi_latency_decode_ms gauge\nizwi_latency_decode_ms{{quantile=\"avg\"}} {:.6}\nizwi_latency_decode_ms{{quantile=\"p50\"}} {:.6}\nizwi_latency_decode_ms{{quantile=\"p95\"}} {:.6}\n\
# TYPE izwi_latency_end_to_end_ms gauge\nizwi_latency_end_to_end_ms{{quantile=\"avg\"}} {:.6}\nizwi_latency_end_to_end_ms{{quantile=\"p50\"}} {:.6}\nizwi_latency_end_to_end_ms{{quantile=\"p95\"}} {:.6}\n",
            snapshot.requests_queued,
            snapshot.requests_completed,
            snapshot.requests_failed,
            snapshot.requests_active,
            snapshot.worker_restarts,
            snapshot.worker_panics,
            snapshot.queue_wait_ms_avg,
            snapshot.queue_wait_ms_p50,
            snapshot.queue_wait_ms_p95,
            snapshot.prefill_ms_avg,
            snapshot.prefill_ms_p50,
            snapshot.prefill_ms_p95,
            snapshot.decode_ms_avg,
            snapshot.decode_ms_p50,
            snapshot.decode_ms_p95,
            snapshot.end_to_end_ms_avg,
            snapshot.end_to_end_ms_p50,
            snapshot.end_to_end_ms_p95,
        )
    }

    async fn push_sample(buffer: &Mutex<VecDeque<f64>>, max_samples: usize, value: f64) {
        let mut guard = buffer.lock().await;
        if guard.len() >= max_samples {
            guard.pop_front();
        }
        guard.push_back(value.max(0.0));
    }
}

fn mean(values: &VecDeque<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn percentile(values: &VecDeque<f64>, q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = values.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len().saturating_sub(1)) as f64 * q.clamp(0.0, 1.0)) as usize;
    sorted[idx]
}

/// Main inference engine runtime.
pub struct RuntimeService {
    pub(crate) config: EngineConfig,
    pub(crate) backend_router: BackendRouter,
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
    fn ensure_requested_backend_available(
        preference: BackendPreference,
        detected: DeviceKind,
    ) -> Result<()> {
        let matched = match preference {
            BackendPreference::Auto => true,
            BackendPreference::Cpu => detected == DeviceKind::Cpu,
            BackendPreference::Metal => detected == DeviceKind::Metal,
            BackendPreference::Cuda => detected == DeviceKind::Cuda,
        };

        if matched {
            return Ok(());
        }

        Err(Error::InferenceError(format!(
            "Requested backend `{}` is not available on this runtime (detected `{}`)",
            preference.as_str(),
            match detected {
                DeviceKind::Cpu => "cpu",
                DeviceKind::Metal => "metal",
                DeviceKind::Cuda => "cuda",
            }
        )))
    }

    /// Create a new inference engine.
    pub fn new(config: EngineConfig) -> Result<Self> {
        configure_runtime_threading(config.num_threads.max(1));
        let model_manager = Arc::new(ModelManager::new(config.clone())?);

        let backend_context =
            BackendRouter::resolve_context(config.backend, BackendSelectionSource::Config);
        let device = backend_context.device.clone();
        Self::ensure_requested_backend_available(config.backend, device.kind)?;
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
        let core_engine = Arc::new(CoreEngine::new_with_worker(core_config, worker_config)?);

        Ok(Self {
            config,
            backend_router: BackendRouter::from_context(backend_context),
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
            crate::catalog::ModelFamily::Lfm2Audio => {
                let model = self
                    .model_registry
                    .get_lfm2(variant)
                    .await
                    .ok_or_else(|| Error::InferenceError("No TTS model loaded".to_string()))?;
                Ok(model.available_voices())
            }
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
        self.ensure_step_driver_started().await;

        request.streaming = true;
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
        self.telemetry.snapshot().await
    }

    /// Prometheus exposition format telemetry payload.
    pub async fn telemetry_prometheus(&self) -> String {
        self.telemetry.prometheus().await
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
