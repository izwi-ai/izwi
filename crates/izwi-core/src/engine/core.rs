//! Engine core - the central orchestrator for inference.
//!
//! The engine core coordinates:
//! - Request scheduling
//! - Model execution
//! - KV cache management
//! - Output processing

use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tracing::{debug, info};

use super::config::EngineCoreConfig;
use super::executor::{ExecutorOutput, UnifiedExecutor, WorkerConfig};
use super::kv_cache::{KVCacheConfig, KVCacheManager, KVCacheStats};
use super::metal_kv_cache::{MetalKVCacheConfig, MetalKVCacheManager};
use super::output::OutputProcessor;
use super::request::{EngineCoreRequest, RequestStatus};
use super::scheduler::{Scheduler, SchedulerConfig};
use super::types::{AudioOutput, EngineOutput, LatencyBreakdown, RequestId};
use crate::backends::{kv_dtype_bytes, BackendKind, BackendRouter, BackendSelectionSource};
use crate::error::{Error, Result};
use crate::model::ModelVariant;

enum KvCacheBackend {
    Standard(KVCacheManager),
    Metal(MetalKVCacheManager),
}

impl KvCacheBackend {
    fn new(config: &EngineCoreConfig) -> Result<Self> {
        let backend_context =
            BackendRouter::resolve_context_for_kind(config.backend, BackendSelectionSource::Config);
        let is_metal = backend_context.backend_kind == BackendKind::Metal;
        // Keep Metal KV manager on its tuned F32 layout unless explicit int8 KV is requested.
        let dtype_bytes = kv_dtype_bytes(&config.kv_cache_dtype, is_metal);
        let kv_config = KVCacheConfig {
            num_layers: 24,
            num_heads: 16,
            head_dim: 64,
            block_size: config.block_size,
            max_blocks: config.max_blocks,
            dtype_bytes,
        };

        if is_metal && kv_config.dtype_bytes == 4 {
            let profile = backend_context.device.clone();
            if profile.kind.is_metal() {
                let mut metal_config = MetalKVCacheConfig::default();
                metal_config.base_config = kv_config.clone();
                let manager = MetalKVCacheManager::new(metal_config, profile)?;
                return Ok(Self::Metal(manager));
            }
        }

        Ok(Self::Standard(KVCacheManager::new(kv_config)))
    }

    fn inner(&self) -> &KVCacheManager {
        match self {
            Self::Standard(manager) => manager,
            Self::Metal(manager) => &manager.inner,
        }
    }

    fn inner_mut(&mut self) -> &mut KVCacheManager {
        match self {
            Self::Standard(manager) => manager,
            Self::Metal(manager) => &mut manager.inner,
        }
    }

    fn maintenance(&mut self) -> Result<()> {
        if let Self::Metal(manager) = self {
            manager.maintenance()?;
        }
        Ok(())
    }

    fn compact_shared_prefixes(&mut self) {
        self.inner_mut().compact_shared_prefixes();
    }

    fn stats(&self) -> KVCacheStats {
        self.inner().stats()
    }
}

#[derive(Debug, Clone, Default)]
struct RequestPhaseTiming {
    first_scheduled_at: Option<Instant>,
    queue_wait_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    prefill_steps: u32,
    decode_steps: u32,
}

/// The engine core - manages the inference loop.
pub struct EngineCore {
    /// Configuration
    config: EngineCoreConfig,
    /// Request scheduler
    scheduler: Scheduler,
    /// KV cache manager
    kv_cache: KvCacheBackend,
    /// Model executor
    executor: UnifiedExecutor,
    /// Output processor
    output_processor: OutputProcessor,
    /// Active requests (by ID)
    requests: HashMap<RequestId, EngineCoreRequest>,
    /// Request start times (for timing)
    request_start_times: HashMap<RequestId, Instant>,
    /// Per-request phase timing accumulated by scheduler steps.
    request_phase_timings: HashMap<RequestId, RequestPhaseTiming>,
    /// Whether the engine has been initialized
    initialized: bool,
    /// Step counter for periodic cache housekeeping.
    maintenance_steps: u64,
}

impl EngineCore {
    fn build_compatible_subbatches<'a>(
        request_refs: &'a [&'a EngineCoreRequest],
        scheduled: &[super::scheduler::ScheduledRequest],
    ) -> Vec<(
        Vec<&'a EngineCoreRequest>,
        Vec<super::scheduler::ScheduledRequest>,
    )> {
        if request_refs.is_empty() || scheduled.is_empty() {
            return Vec::new();
        }

        let mut request_by_id = HashMap::with_capacity(request_refs.len());
        for req in request_refs {
            request_by_id.insert(req.id.as_str(), *req);
        }

        let mut groups: Vec<(
            (super::types::TaskType, Option<crate::model::ModelVariant>),
            Vec<super::scheduler::ScheduledRequest>,
        )> = Vec::new();

        for item in scheduled {
            let Some(req) = request_by_id.get(item.request_id.as_str()) else {
                continue;
            };
            let key = (req.task_type, req.model_variant);
            if let Some((_, bucket)) = groups.iter_mut().find(|(group_key, _)| *group_key == key) {
                bucket.push(item.clone());
            } else {
                groups.push((key, vec![item.clone()]));
            }
        }

        let mut outputs = Vec::new();
        for (_, bucket) in groups {
            let mut bucket_refs = Vec::with_capacity(bucket.len());
            let mut seen = HashSet::new();
            for item in &bucket {
                if !seen.insert(item.request_id.as_str()) {
                    continue;
                }
                if let Some(req) = request_by_id.get(item.request_id.as_str()) {
                    bucket_refs.push(*req);
                }
            }
            outputs.push((bucket_refs, bucket));
        }

        outputs
    }

    fn should_microbatch_decode(scheduled: &[super::scheduler::ScheduledRequest]) -> bool {
        scheduled.len() > 1 && scheduled.iter().any(|entry| entry.num_tokens > 1)
    }

    fn merge_audio_output(
        existing: Option<AudioOutput>,
        current: Option<AudioOutput>,
    ) -> Option<AudioOutput> {
        match (existing, current) {
            (None, None) => None,
            (Some(existing), None) => Some(existing),
            (None, Some(current)) => Some(current),
            (Some(mut existing), Some(current)) => {
                if existing.sample_rate != current.sample_rate {
                    return Some(current);
                }
                if current.samples.is_empty() {
                    return Some(existing);
                }
                if existing.samples.is_empty() {
                    return Some(current);
                }

                let looks_cumulative = current.samples.len() >= existing.samples.len()
                    && current
                        .samples
                        .iter()
                        .zip(existing.samples.iter())
                        .all(|(cur, prev)| cur == prev);

                if looks_cumulative {
                    Some(current)
                } else {
                    existing.append(&current);
                    Some(existing)
                }
            }
        }
    }

    fn merge_executor_output(
        existing: Option<ExecutorOutput>,
        current: ExecutorOutput,
    ) -> ExecutorOutput {
        let Some(mut merged) = existing else {
            return current;
        };

        if merged.request_id != current.request_id {
            return current;
        }
        if merged.finished || merged.error.is_some() {
            return merged;
        }

        let ExecutorOutput {
            request_id: _,
            audio,
            text,
            tokens_processed,
            tokens_generated,
            finished,
            error,
        } = current;

        merged.audio = Self::merge_audio_output(merged.audio.take(), audio);
        if text.is_some() {
            merged.text = text;
        }
        merged.tokens_processed = merged.tokens_processed.saturating_add(tokens_processed);
        merged.tokens_generated = merged.tokens_generated.saturating_add(tokens_generated);
        merged.finished |= finished;
        if error.is_some() {
            merged.error = error;
        }

        merged
    }

    async fn execute_decode_subbatch(
        &self,
        request_refs: &[&EngineCoreRequest],
        scheduled: &[super::scheduler::ScheduledRequest],
    ) -> Result<Vec<ExecutorOutput>> {
        if !Self::should_microbatch_decode(scheduled) {
            return self.executor.execute_decode(request_refs, scheduled).await;
        }

        let mut remaining_tokens: Vec<usize> = scheduled
            .iter()
            .map(|entry| entry.num_tokens.max(1))
            .collect();
        let mut merged_outputs: Vec<Option<ExecutorOutput>> = vec![None; scheduled.len()];

        let request_idx_by_id: HashMap<&str, usize> = scheduled
            .iter()
            .enumerate()
            .map(|(idx, entry)| (entry.request_id.as_str(), idx))
            .collect();

        while remaining_tokens.iter().any(|remaining| *remaining > 0) {
            let mut round_indices = Vec::new();
            let mut round_schedule = Vec::new();

            for (idx, entry) in scheduled.iter().enumerate() {
                if remaining_tokens[idx] == 0 {
                    continue;
                }
                let mut single = entry.clone();
                single.num_tokens = 1;
                round_indices.push(idx);
                round_schedule.push(single);
            }

            if round_schedule.is_empty() {
                break;
            }

            let round_outputs = self
                .executor
                .execute_decode(request_refs, &round_schedule)
                .await?;
            let mut seen = vec![false; scheduled.len()];

            for output in round_outputs {
                let Some(&idx) = request_idx_by_id.get(output.request_id.as_str()) else {
                    continue;
                };
                if remaining_tokens[idx] == 0 {
                    continue;
                }

                seen[idx] = true;
                let combined = Self::merge_executor_output(merged_outputs[idx].take(), output);
                if combined.finished || combined.error.is_some() {
                    remaining_tokens[idx] = 0;
                } else {
                    remaining_tokens[idx] = remaining_tokens[idx].saturating_sub(1);
                }
                merged_outputs[idx] = Some(combined);
            }

            for idx in round_indices {
                if seen[idx] {
                    continue;
                }

                remaining_tokens[idx] = 0;
                let fallback = ExecutorOutput::error(
                    scheduled[idx].request_id.clone(),
                    "Decode micro-batch round did not return an output for a scheduled request",
                );
                merged_outputs[idx] = Some(Self::merge_executor_output(
                    merged_outputs[idx].take(),
                    fallback,
                ));
            }
        }

        Ok(merged_outputs
            .into_iter()
            .enumerate()
            .map(|(idx, output)| {
                output.unwrap_or_else(|| {
                    ExecutorOutput::error(
                        scheduled[idx].request_id.clone(),
                        "Decode micro-batch did not produce any output",
                    )
                })
            })
            .collect())
    }

    /// Create a new engine core.
    pub fn new(config: EngineCoreConfig) -> Result<Self> {
        let worker_config = WorkerConfig::from(&config);
        Self::new_with_worker(config, worker_config)
    }

    /// Create a new engine core with an explicit worker configuration.
    pub fn new_with_worker(config: EngineCoreConfig, worker_config: WorkerConfig) -> Result<Self> {
        info!("Creating engine core");

        let executor = UnifiedExecutor::new_native(worker_config);
        Self::new_with_executor(config, executor)
    }

    fn new_with_executor(config: EngineCoreConfig, executor: UnifiedExecutor) -> Result<Self> {
        // Create scheduler
        let scheduler_config = SchedulerConfig::from(&config);
        let scheduler = Scheduler::new(scheduler_config);

        // Create KV cache manager
        let kv_cache = KvCacheBackend::new(&config)?;

        // Create output processor
        let output_processor =
            OutputProcessor::new(config.sample_rate).with_chunk_size(config.streaming_chunk_size);

        Ok(Self {
            config,
            scheduler,
            kv_cache,
            executor,
            output_processor,
            requests: HashMap::new(),
            request_start_times: HashMap::new(),
            request_phase_timings: HashMap::new(),
            initialized: false,
            maintenance_steps: 0,
        })
    }

    #[cfg(test)]
    pub(crate) fn new_with_unified_executor(
        config: EngineCoreConfig,
        executor: UnifiedExecutor,
    ) -> Result<Self> {
        info!("Creating engine core");
        Self::new_with_executor(config, executor)
    }

    /// Initialize the engine core.
    pub async fn initialize(&mut self) -> Result<()> {
        if self.initialized {
            return Ok(());
        }

        info!("Initializing engine core");

        // Initialize executor backend
        self.executor.initialize().await?;

        self.initialized = true;
        info!("Engine core initialized");

        Ok(())
    }

    /// Add a request to the engine.
    pub fn add_request(&mut self, request: EngineCoreRequest) -> Result<()> {
        let request_id = request.id.clone();

        if self.requests.contains_key(&request_id) {
            return Err(Error::InvalidInput(format!(
                "Request {} already exists",
                request_id
            )));
        }

        // Add to scheduler
        self.scheduler.add_request(&request);

        // Track request
        self.requests.insert(request_id.clone(), request);
        self.request_start_times
            .insert(request_id.clone(), Instant::now());
        self.request_phase_timings
            .insert(request_id.clone(), RequestPhaseTiming::default());

        debug!(
            request_id = %request_id,
            correlation_id = ?self.requests.get(&request_id).and_then(|req| req.correlation_id.as_deref()),
            "Added request to engine core"
        );

        Ok(())
    }

    /// Execute one step of the inference loop.
    ///
    /// The step consists of:
    /// 1. Schedule - select requests to process
    /// 2. Execute - run forward pass
    /// 3. Process - handle outputs, check stop conditions
    pub async fn step(&mut self) -> Result<Vec<EngineOutput>> {
        // Ensure initialized
        if !self.initialized {
            self.initialize().await?;
        }

        // Phase 1: Schedule
        self.kv_cache.maintenance()?;
        self.maintenance_steps = self.maintenance_steps.saturating_add(1);
        if self.maintenance_steps % 64 == 0 {
            self.kv_cache.compact_shared_prefixes();
        }
        let schedule_result = self.scheduler.schedule(self.kv_cache.inner_mut());

        if !schedule_result.has_work() {
            return Ok(Vec::new());
        }

        debug!(
            "Scheduled {} prefill, {} decode requests",
            schedule_result.prefill_requests.len(),
            schedule_result.decode_requests.len()
        );

        let prefill_scheduled = schedule_result.prefill_requests.clone();
        let decode_scheduled = schedule_result.decode_requests.clone();
        let now = Instant::now();

        // Capture queue wait for first scheduling event.
        for scheduled in decode_scheduled.iter().chain(prefill_scheduled.iter()) {
            let request_id = &scheduled.request_id;
            let timing = self
                .request_phase_timings
                .entry(request_id.clone())
                .or_default();
            if timing.first_scheduled_at.is_none() {
                timing.first_scheduled_at = Some(now);
                if let Some(started) = self.request_start_times.get(request_id) {
                    timing.queue_wait_ms = started.elapsed().as_secs_f64() * 1000.0;
                }
            }
        }

        let prefill_request_refs: Vec<&EngineCoreRequest> = prefill_scheduled
            .iter()
            .filter_map(|s| self.requests.get(&s.request_id))
            .collect();
        let decode_request_refs: Vec<&EngineCoreRequest> = decode_scheduled
            .iter()
            .filter_map(|s| self.requests.get(&s.request_id))
            .collect();

        if prefill_request_refs.is_empty() && decode_request_refs.is_empty() {
            return Ok(Vec::new());
        }

        // Phase 2: Execute decode/prefill. On Metal we prefer sequential execution
        // to reduce device contention and thermal spikes on local machines.
        let run_decode = async {
            if decode_request_refs.is_empty() || decode_scheduled.is_empty() {
                return Ok((Vec::new(), std::time::Duration::ZERO));
            }
            let started = Instant::now();
            let sub_batches =
                Self::build_compatible_subbatches(&decode_request_refs, &decode_scheduled);
            let mut outputs = Vec::new();
            for (refs, batch) in sub_batches {
                outputs.extend(self.execute_decode_subbatch(&refs, &batch).await?);
            }
            Ok::<_, Error>((outputs, started.elapsed()))
        };
        let run_prefill = async {
            if prefill_request_refs.is_empty() || prefill_scheduled.is_empty() {
                return Ok((Vec::new(), std::time::Duration::ZERO));
            }
            let started = Instant::now();
            let sub_batches =
                Self::build_compatible_subbatches(&prefill_request_refs, &prefill_scheduled);
            let mut outputs = Vec::new();
            for (refs, batch) in sub_batches {
                outputs.extend(self.executor.execute_prefill(&refs, &batch).await?);
            }
            Ok::<_, Error>((outputs, started.elapsed()))
        };

        let (mut decode_outputs, decode_elapsed, mut prefill_outputs, prefill_elapsed) =
            if self.config.backend == BackendKind::Metal
                && !decode_request_refs.is_empty()
                && !prefill_request_refs.is_empty()
            {
                let (decode_outputs, decode_elapsed) = run_decode.await?;
                let (prefill_outputs, prefill_elapsed) = run_prefill.await?;
                (
                    decode_outputs,
                    decode_elapsed,
                    prefill_outputs,
                    prefill_elapsed,
                )
            } else {
                let (decode_result, prefill_result) = tokio::join!(run_decode, run_prefill);
                let (decode_outputs, decode_elapsed) = decode_result?;
                let (prefill_outputs, prefill_elapsed) = prefill_result?;
                (
                    decode_outputs,
                    decode_elapsed,
                    prefill_outputs,
                    prefill_elapsed,
                )
            };

        let decode_step_ms = decode_elapsed.as_secs_f64() * 1000.0;
        let prefill_step_ms = prefill_elapsed.as_secs_f64() * 1000.0;
        let decode_ids: HashSet<RequestId> = decode_scheduled
            .iter()
            .map(|s| s.request_id.clone())
            .collect();
        let prefill_ids: HashSet<RequestId> = prefill_scheduled
            .iter()
            .map(|s| s.request_id.clone())
            .collect();

        for request_id in &decode_ids {
            let timing = self
                .request_phase_timings
                .entry(request_id.clone())
                .or_default();
            timing.decode_ms += decode_step_ms;
            timing.decode_steps = timing.decode_steps.saturating_add(1);
        }
        for request_id in &prefill_ids {
            let timing = self
                .request_phase_timings
                .entry(request_id.clone())
                .or_default();
            timing.prefill_ms += prefill_step_ms;
            timing.prefill_steps = timing.prefill_steps.saturating_add(1);
        }

        decode_outputs.append(&mut prefill_outputs);
        let executor_outputs = decode_outputs;

        // Phase 3: Process outputs
        let mut outputs = Vec::new();

        for exec_output in executor_outputs {
            let request_id = exec_output.request_id.clone();

            // Get timing info
            let generation_time = self
                .request_start_times
                .get(&request_id)
                .map(|t| t.elapsed())
                .unwrap_or_default();

            // Get sequence ID from scheduler
            let sequence_id = self.scheduler.get_sequence_id(&request_id).unwrap_or(0);

            let step_time_ms = if decode_ids.contains(&request_id) {
                decode_step_ms
            } else {
                prefill_step_ms
            };

            self.scheduler.update_after_step(
                &request_id,
                exec_output.tokens_processed,
                exec_output.tokens_generated,
                Vec::new(),
                step_time_ms,
            );

            // Process output
            let mut engine_output =
                self.output_processor
                    .process(exec_output.clone(), sequence_id, generation_time);
            if exec_output.finished {
                if let Some((_, total_generated)) = self.scheduler.get_running_info(&request_id) {
                    let resolved_total = total_generated.max(engine_output.num_tokens);
                    engine_output.num_tokens = resolved_total;
                    engine_output.token_stats.generated_tokens = resolved_total;
                }
            }
            if let Some(phase) = self.request_phase_timings.get(&request_id).cloned() {
                engine_output.token_stats.prefill_time_ms = phase.prefill_ms as f32;
                engine_output.token_stats.decode_time_ms = phase.decode_ms as f32;
                if phase.decode_ms > 0.0 {
                    engine_output.token_stats.tokens_per_second =
                        (engine_output.token_stats.generated_tokens as f64 * 1000.0
                            / phase.decode_ms) as f32;
                }
                engine_output.latency_breakdown = Some(LatencyBreakdown {
                    queue_wait_ms: phase.queue_wait_ms,
                    prefill_ms: phase.prefill_ms,
                    decode_ms: phase.decode_ms,
                    total_ms: generation_time.as_secs_f64() * 1000.0,
                    prefill_steps: phase.prefill_steps,
                    decode_steps: phase.decode_steps,
                });
            }

            // Update scheduler state
            if exec_output.finished {
                self.executor.cleanup_request(&request_id).await;
                self.scheduler
                    .finish_request(&request_id, self.kv_cache.inner_mut());
                self.requests.remove(&request_id);
                self.request_start_times.remove(&request_id);
                self.request_phase_timings.remove(&request_id);
                debug!("Finished request {}", request_id);
            }

            outputs.push(engine_output);
        }

        Ok(outputs)
    }

    /// Check if there's pending work.
    pub fn has_pending_work(&self) -> bool {
        self.scheduler.has_pending_work()
    }

    /// Check if a request exists.
    pub fn has_request(&self, request_id: &RequestId) -> bool {
        self.requests.contains_key(request_id)
    }

    /// Get request status.
    pub fn get_request_status(&self, request_id: &RequestId) -> Option<RequestStatus> {
        self.scheduler.get_status(request_id)
    }

    /// Abort a request.
    pub async fn abort_request(&mut self, request_id: &RequestId) -> bool {
        let existed = self.scheduler.has_request(request_id);
        let removed_running = self
            .scheduler
            .abort_request(request_id, self.kv_cache.inner_mut());
        if removed_running || (existed && !self.scheduler.has_request(request_id)) {
            self.executor.cleanup_request(request_id).await;
            self.requests.remove(request_id);
            self.request_start_times.remove(request_id);
            self.request_phase_timings.remove(request_id);
            debug!("Aborted request {}", request_id);
            true
        } else {
            false
        }
    }

    /// Abort all active requests that target a specific model variant.
    pub async fn abort_requests_for_variant(&mut self, variant: ModelVariant) -> Vec<RequestId> {
        let request_ids: Vec<RequestId> = self
            .requests
            .iter()
            .filter_map(|(request_id, request)| {
                if request.model_variant == Some(variant) {
                    Some(request_id.clone())
                } else {
                    None
                }
            })
            .collect();

        let mut aborted = Vec::with_capacity(request_ids.len());
        for request_id in request_ids {
            if self.abort_request(&request_id).await {
                aborted.push(request_id);
            }
        }
        aborted
    }

    /// Get number of pending (waiting) requests.
    pub fn pending_request_count(&self) -> usize {
        self.scheduler.waiting_count()
    }

    /// Get number of running requests.
    pub fn running_request_count(&self) -> usize {
        self.scheduler.running_count()
    }

    /// Get KV cache statistics.
    pub fn kv_cache_stats(&self) -> super::kv_cache::KVCacheStats {
        self.kv_cache.stats()
    }

    /// Get configuration.
    pub fn config(&self) -> &EngineCoreConfig {
        &self.config
    }

    /// Shutdown the engine core.
    pub async fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down engine core");

        // Abort all pending requests
        let request_ids: Vec<_> = self.requests.keys().cloned().collect();
        for id in request_ids {
            self.abort_request(&id).await;
        }

        // Shutdown executor
        self.executor.shutdown().await?;

        self.initialized = false;
        info!("Engine core shutdown complete");

        Ok(())
    }
}

impl Drop for EngineCore {
    fn drop(&mut self) {
        // Note: We can't do async cleanup in drop, so we just log
        if self.initialized {
            debug!("EngineCore dropped while still initialized");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::executor::{ExecutorOutput, ModelExecutor};
    use super::super::scheduler::ScheduledRequest;
    use super::super::types::{AudioOutput, Priority};
    use super::*;
    use crate::model::ModelVariant;
    use std::sync::{Arc, Mutex};

    struct MockExecutor {
        initialized: bool,
        cleanup_calls: Arc<Mutex<Vec<String>>>,
    }

    impl MockExecutor {
        fn new(cleanup_calls: Arc<Mutex<Vec<String>>>) -> Self {
            Self {
                initialized: false,
                cleanup_calls,
            }
        }

        fn build_outputs(scheduled: &[ScheduledRequest]) -> Vec<ExecutorOutput> {
            scheduled
                .iter()
                .map(|entry| ExecutorOutput {
                    request_id: entry.request_id.clone(),
                    audio: None,
                    text: None,
                    tokens_processed: entry.num_tokens.max(1),
                    tokens_generated: usize::from(!entry.is_prefill),
                    finished: false,
                    error: None,
                })
                .collect()
        }
    }

    impl ModelExecutor for MockExecutor {
        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorOutput>> {
            Ok(Self::build_outputs(scheduled))
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorOutput>> {
            Ok(Self::build_outputs(scheduled))
        }

        fn is_ready(&self) -> bool {
            self.initialized
        }

        fn initialize(&mut self) -> Result<()> {
            self.initialized = true;
            Ok(())
        }

        fn shutdown(&mut self) -> Result<()> {
            self.initialized = false;
            Ok(())
        }

        fn cleanup_request(&self, request_id: &str) {
            if let Ok(mut calls) = self.cleanup_calls.lock() {
                calls.push(request_id.to_string());
            }
        }
    }

    struct TraceDecodeExecutor {
        decode_calls: Arc<Mutex<Vec<Vec<(String, usize)>>>>,
    }

    impl TraceDecodeExecutor {
        fn new(decode_calls: Arc<Mutex<Vec<Vec<(String, usize)>>>>) -> Self {
            Self { decode_calls }
        }
    }

    impl ModelExecutor for TraceDecodeExecutor {
        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorOutput>> {
            Ok(scheduled
                .iter()
                .map(|entry| ExecutorOutput {
                    request_id: entry.request_id.clone(),
                    audio: Some(AudioOutput::empty(24_000)),
                    text: None,
                    tokens_processed: entry.num_tokens.max(1),
                    tokens_generated: 0,
                    finished: false,
                    error: None,
                })
                .collect())
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorOutput>> {
            if let Ok(mut calls) = self.decode_calls.lock() {
                calls.push(
                    scheduled
                        .iter()
                        .map(|entry| (entry.request_id.clone(), entry.num_tokens))
                        .collect(),
                );
            }
            Ok(scheduled
                .iter()
                .map(|entry| ExecutorOutput {
                    request_id: entry.request_id.clone(),
                    audio: Some(AudioOutput::empty(24_000)),
                    text: Some(format!("step-{}", entry.request_id)),
                    tokens_processed: entry.num_tokens.max(1),
                    tokens_generated: entry.num_tokens.max(1),
                    finished: false,
                    error: None,
                })
                .collect())
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
    }

    #[test]
    fn test_engine_core_creation() {
        let config = EngineCoreConfig::default();
        let core = EngineCore::new(config);
        assert!(core.is_ok());
    }

    #[tokio::test]
    async fn test_add_request() {
        let config = EngineCoreConfig::default();
        let mut core = EngineCore::new(config).unwrap();

        let request = EngineCoreRequest::tts("Hello, world!");
        let result = core.add_request(request);
        assert!(result.is_ok());
        assert_eq!(core.pending_request_count(), 1);
    }

    #[tokio::test]
    async fn test_step_preserves_executor_state_for_preempted_request() {
        let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
        let executor =
            UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls.clone())));

        let config = EngineCoreConfig {
            max_batch_size: 2,
            max_tokens_per_step: 8,
            min_tokens_per_step: 1,
            block_size: 1,
            max_blocks: 1,
            scheduling_policy: super::super::scheduler::SchedulingPolicy::Priority,
            enable_chunked_prefill: false,
            enable_preemption: true,
            enable_adaptive_batching: false,
            backend: BackendKind::Cpu,
            ..Default::default()
        };
        let mut core = EngineCore::new_with_unified_executor(config, executor).unwrap();

        let mut low = EngineCoreRequest::tts("low-priority");
        low.id = "low-priority".to_string();
        low.prompt_tokens = vec![1];
        low.priority = Priority::Low;
        core.add_request(low).unwrap();
        let _ = core.step().await.unwrap();

        let mut high = EngineCoreRequest::tts("high-priority");
        high.id = "high-priority".to_string();
        high.prompt_tokens = vec![1];
        high.priority = Priority::High;
        core.add_request(high).unwrap();
        let _ = core.step().await.unwrap();

        let calls = cleanup_calls.lock().unwrap().clone();
        assert!(
            !calls.iter().any(|id| id == "low-priority"),
            "preempted request should preserve executor decode state for resume"
        );
        assert_eq!(
            core.get_request_status(&"low-priority".to_string()),
            Some(RequestStatus::Running)
        );
    }

    #[tokio::test]
    async fn test_abort_requests_for_variant_only_aborts_matching_requests() {
        let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
        let executor =
            UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls.clone())));
        let config = EngineCoreConfig::default();
        let mut core = EngineCore::new_with_unified_executor(config, executor).unwrap();

        let mut req_a = EngineCoreRequest::tts("variant-a");
        req_a.id = "req-a".to_string();
        req_a.model_variant = Some(ModelVariant::Qwen34BGguf);

        let mut req_b = EngineCoreRequest::tts("variant-b");
        req_b.id = "req-b".to_string();
        req_b.model_variant = Some(ModelVariant::Qwen38BGguf);

        core.add_request(req_a).unwrap();
        core.add_request(req_b).unwrap();

        let aborted = core
            .abort_requests_for_variant(ModelVariant::Qwen34BGguf)
            .await;
        assert_eq!(aborted, vec!["req-a".to_string()]);
        assert!(!core.has_request(&"req-a".to_string()));
        assert!(core.has_request(&"req-b".to_string()));

        let calls = cleanup_calls.lock().unwrap().clone();
        assert!(calls.iter().any(|id| id == "req-a"));
        assert!(!calls.iter().any(|id| id == "req-b"));
    }

    #[test]
    fn test_merge_executor_output_replaces_cumulative_audio_snapshots() {
        let first = ExecutorOutput {
            request_id: "req-a".to_string(),
            audio: Some(AudioOutput::new(vec![0.1, 0.2], 24_000)),
            text: Some("hello".to_string()),
            tokens_processed: 1,
            tokens_generated: 1,
            finished: false,
            error: None,
        };
        let second = ExecutorOutput {
            request_id: "req-a".to_string(),
            audio: Some(AudioOutput::new(vec![0.1, 0.2, 0.3], 24_000)),
            text: Some("hello world".to_string()),
            tokens_processed: 1,
            tokens_generated: 1,
            finished: true,
            error: None,
        };

        let merged = EngineCore::merge_executor_output(Some(first), second);
        let audio = merged.audio.expect("merged audio");
        assert_eq!(audio.samples, vec![0.1, 0.2, 0.3]);
        assert_eq!(merged.text.as_deref(), Some("hello world"));
        assert_eq!(merged.tokens_processed, 2);
        assert_eq!(merged.tokens_generated, 2);
        assert!(merged.finished);
    }

    #[tokio::test]
    async fn test_execute_decode_subbatch_round_robins_multi_token_requests() {
        let decode_calls = Arc::new(Mutex::new(Vec::new()));
        let executor =
            UnifiedExecutor::new_for_test(Box::new(TraceDecodeExecutor::new(decode_calls.clone())));

        let config = EngineCoreConfig::default();
        let core = EngineCore::new_with_unified_executor(config, executor).unwrap();

        let mut req_a = EngineCoreRequest::tts("a");
        req_a.id = "req-a".to_string();
        let mut req_b = EngineCoreRequest::tts("b");
        req_b.id = "req-b".to_string();
        let req_refs = vec![&req_a, &req_b];

        let scheduled = vec![
            ScheduledRequest {
                request_id: req_a.id.clone(),
                sequence_id: 0,
                num_tokens: 3,
                is_prefill: false,
                block_ids: Vec::new(),
                num_computed_tokens: 0,
            },
            ScheduledRequest {
                request_id: req_b.id.clone(),
                sequence_id: 1,
                num_tokens: 2,
                is_prefill: false,
                block_ids: Vec::new(),
                num_computed_tokens: 0,
            },
        ];

        let outputs = core
            .execute_decode_subbatch(&req_refs, &scheduled)
            .await
            .unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].request_id, "req-a");
        assert_eq!(outputs[0].tokens_processed, 3);
        assert_eq!(outputs[0].tokens_generated, 3);
        assert_eq!(outputs[1].request_id, "req-b");
        assert_eq!(outputs[1].tokens_processed, 2);
        assert_eq!(outputs[1].tokens_generated, 2);

        let calls = decode_calls.lock().unwrap().clone();
        assert_eq!(calls.len(), 3);
        assert_eq!(
            calls[0],
            vec![("req-a".to_string(), 1), ("req-b".to_string(), 1)]
        );
        assert_eq!(
            calls[1],
            vec![("req-a".to_string(), 1), ("req-b".to_string(), 1)]
        );
        assert_eq!(calls[2], vec![("req-a".to_string(), 1)]);
    }
}
