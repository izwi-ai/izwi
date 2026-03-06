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

mod config;
mod core;
mod executor;
mod kv_cache;
mod metal_kv_cache;
pub mod metrics;
mod output;
mod request;
mod scheduler;
pub mod signal_frontend;
mod types;

pub use config::EngineCoreConfig;
pub use core::EngineCore;
pub use executor::{ExecutorOutput, ModelExecutor, WorkerConfig};
pub use kv_cache::{
    BlockAllocator, CacheResidency, KVCacheConfig as KVConfig, KVCacheManager, PinnedBlockHandle,
};
pub use metrics::{BenchmarkResult, MetricsCollector, MetricsSnapshot};
pub use output::{OutputProcessor, StreamingOutput};
pub use request::{EngineCoreRequest, RequestProcessor, RequestStatus};
pub use scheduler::{ScheduleResult, Scheduler, SchedulerConfig, SchedulingPolicy};
pub use types::{
    AudioOutput, EngineMetrics, EngineOutput, GenerationParams, RequestId, SequenceId,
};

use crate::error::Result;
use crate::model::ModelVariant;
use std::sync::Arc;
use tokio::sync::{mpsc, Notify, RwLock};
use tracing::{debug, info, warn};

use self::types::TaskType;

/// Main inference engine - the primary interface for audio generation.
///
/// The engine orchestrates all components and provides both synchronous
/// and asynchronous interfaces for inference.
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
            TaskType::TTS | TaskType::SpeechToSpeech => 8usize,
            TaskType::ASR | TaskType::Chat => 64usize,
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
        })
    }

    /// Add a request to the engine for processing.
    ///
    /// The request will be validated, preprocessed, and added to the scheduler's
    /// waiting queue. Returns a request ID that can be used to track the request.
    pub async fn add_request(&self, request: EngineCoreRequest) -> Result<RequestId> {
        // Validate and preprocess
        let processed = self.request_processor.process(request)?;
        let request_id = processed.id.clone();

        // Add to engine core
        let mut core = self.core.write().await;
        core.add_request(processed)?;
        self.wake_notify.notify_one();

        debug!("Added request {} to engine", request_id);
        Ok(request_id)
    }

    /// Generate audio synchronously (blocking until complete).
    ///
    /// This is a convenience method that adds a request and waits for completion.
    pub async fn generate(&self, request: EngineCoreRequest) -> Result<EngineOutput> {
        let request_id = self.add_request(request).await?;

        // Run steps until this request completes
        loop {
            let outputs = self.step().await?;

            for output in outputs {
                if output.request_id == request_id && output.is_finished {
                    if let Some(err) = output.error.clone() {
                        return Err(crate::error::Error::InferenceError(err));
                    }
                    return Ok(output);
                }
            }

            // Check if request is still in the system
            let core = self.core.read().await;
            if !core.has_request(&request_id) {
                return Err(crate::error::Error::InferenceError(format!(
                    "Request {} was removed unexpectedly",
                    request_id
                )));
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

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_steps += 1;
            metrics.requests_processed += outputs.len() as u64;
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

        while self.running.load(Ordering::SeqCst) {
            // Check if there are requests to process
            let has_work = {
                let core = self.core.read().await;
                core.has_pending_work()
            };

            if has_work {
                if let Err(e) = self.step().await {
                    warn!("Engine step error: {}", e);
                }
            } else {
                // Event-driven wait to avoid hot polling on local/edge devices.
                tokio::select! {
                    _ = self.wake_notify.notified() => {},
                    _ = tokio::time::sleep(tokio::time::Duration::from_millis(50)) => {},
                }
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
        let mut core = self.core.write().await;
        Ok(core.abort_request(request_id).await)
    }

    /// Abort all requests currently routed to a specific model variant.
    pub async fn abort_requests_for_variant(&self, variant: ModelVariant) -> Vec<RequestId> {
        let mut core = self.core.write().await;
        core.abort_requests_for_variant(variant).await
    }

    /// Check if a request is still tracked by the engine core.
    pub async fn has_request(&self, request_id: &RequestId) -> bool {
        let core = self.core.read().await;
        core.has_request(request_id)
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

    /// Check if scheduler currently has runnable or queued work.
    pub async fn has_pending_work(&self) -> bool {
        let core = self.core.read().await;
        core.has_pending_work()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_engine_creation() {
        let config = EngineCoreConfig::default();
        let engine = Engine::new(config);
        assert!(engine.is_ok());
    }
}
