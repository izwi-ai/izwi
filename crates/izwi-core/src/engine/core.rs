//! Engine core - the central orchestrator for inference.
//!
//! The engine core coordinates:
//! - Request scheduling
//! - Model execution
//! - KV cache management
//! - Output processing

use std::collections::{HashMap, HashSet};
use std::time::Instant;
use tracing::{debug, info, warn};

use super::config::EngineCoreConfig;
use super::execution::{
    BatchKey, ExecutionDisposition, ExecutionFailure, ExecutionMode, ExecutionPlan,
    ExecutionProfile, ExecutionReport, ExecutionState, ExecutionTracker, PrefillMode, WorkUnit,
};
use super::executor::{
    ExecutorOutput, ExecutorStepResult, UnifiedExecutor, WorkerConfig, REQUEST_DEADLINE_EXCEEDED,
};
use super::kv_cache::{KVCacheConfig, KVCacheManager, KVCacheStats};
use super::metal_kv_cache::{MetalKVCacheConfig, MetalKVCacheManager};
use super::output::OutputProcessor;
use super::request::{EngineCoreRequest, RequestStatus};
use super::scheduler::{Scheduler, SchedulerConfig};
use super::types::{AudioOutput, EngineOutput, LatencyBreakdown, RequestId};
use super::ResourceVector;
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
    media_decode_ms: Option<f64>,
    normalization_ms: Option<f64>,
    prefill_ms: f64,
    decode_ms: f64,
    sampling_ms: Option<f64>,
    codec_ms: Option<f64>,
    postprocess_ms: Option<f64>,
    first_output_ms: Option<f64>,
    prefill_steps: u32,
    decode_steps: u32,
}

fn merge_optional_phase_ms(target: &mut Option<f64>, value: Option<f64>) {
    if let Some(value) = value {
        let value = value.max(0.0);
        *target = Some(target.unwrap_or(0.0) + value);
    }
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
    /// Per-session lifecycle and active-plan fence.
    execution_trackers: HashMap<RequestId, ExecutionTracker>,
    /// Plans prepared under the core lock and awaiting one validated result.
    active_plans: HashMap<u64, ExecutionPlan>,
    /// Whether the engine has been initialized
    initialized: bool,
    /// Step counter for periodic cache housekeeping.
    maintenance_steps: u64,
}

impl EngineCore {
    async fn refresh_scheduler_execution_profiles(&mut self) {
        let requests: Vec<_> = self.requests.values().cloned().collect();
        for request in requests {
            let Some(epoch) = self.scheduler.get_sequence_id(&request.id) else {
                continue;
            };
            let profile = self
                .executor
                .execution_profile(&request)
                .await
                .unwrap_or_else(|| {
                    ExecutionProfile::fail_closed(
                        self.config.backend,
                        request.model_variant,
                        ExecutionMode::Atomic,
                    )
                });
            self.scheduler.update_execution_profile(
                &super::SessionKey::new(request.id.clone(), epoch),
                &profile,
            );
        }
    }

    async fn begin_execution_plan(
        &mut self,
        scheduled: &super::scheduler::ScheduledRequest,
    ) -> Result<()> {
        let request = self
            .requests
            .get(&scheduled.request_id)
            .cloned()
            .ok_or_else(|| {
                Error::InferenceError(format!(
                    "scheduled request {} is missing from the engine",
                    scheduled.request_id
                ))
            })?;
        let profile = self
            .executor
            .execution_profile(&request)
            .await
            .unwrap_or_else(|| {
                ExecutionProfile::fail_closed(
                    self.config.backend,
                    request.model_variant,
                    ExecutionMode::Atomic,
                )
            });
        let work = match profile.mode {
            ExecutionMode::Sequence | ExecutionMode::Realtime
                if scheduled.is_prefill && profile.prefill == PrefillMode::Full =>
            {
                WorkUnit::SequenceStep {
                    phase: super::SequencePhase::Prefill,
                    input: super::InputRange {
                        start: 0,
                        end: request.num_prompt_tokens(),
                    },
                    max_output_steps: 1,
                }
            }
            ExecutionMode::Sequence | ExecutionMode::Realtime => scheduled.work.clone(),
            ExecutionMode::Atomic | ExecutionMode::Artifact => WorkUnit::AtomicJob {
                kind: format!("{:?}", request.task_type).to_ascii_lowercase(),
            },
            ExecutionMode::Pipeline => WorkUnit::PipelineStage {
                name: format!("{:?}", request.task_type).to_ascii_lowercase(),
                ordinal: 0,
            },
        };
        let work_kind = match &work {
            WorkUnit::SequenceStep { phase, .. } => format!("{phase:?}").to_ascii_lowercase(),
            WorkUnit::AtomicJob { kind } => kind.clone(),
            WorkUnit::PipelineStage { name, ordinal } => format!("{name}:{ordinal}"),
        };
        let plan = ExecutionPlan {
            plan_id: scheduled.plan_id,
            session: scheduled.session_key(),
            work,
            batch_key: BatchKey {
                backend: profile.backend,
                model_variant: profile.model_variant,
                task_type: request.task_type,
                work_kind,
                compute_dtype: profile.compute_dtype,
                kv_dtype: profile.kv_dtype,
                cache_namespace: profile
                    .cache_namespace
                    .unwrap_or_else(|| "none".to_string()),
                adapter_id: None,
            },
            estimate: ResourceVector::zero(),
            reservation: None,
        };

        let tracker = self
            .execution_trackers
            .entry(scheduled.request_id.clone())
            .or_insert_with(|| ExecutionTracker::new(plan.session.clone()));
        if tracker.session() != &plan.session {
            return Err(Error::InferenceError(format!(
                "request {} already has a different active session",
                scheduled.request_id
            )));
        }
        if tracker.state() == ExecutionState::Queued {
            tracker.transition(ExecutionState::Admitted)?;
        }
        let running_state = match plan.work {
            WorkUnit::SequenceStep {
                phase: super::SequencePhase::Prefill,
                ..
            } => ExecutionState::Prefilling,
            WorkUnit::SequenceStep {
                phase: super::SequencePhase::Decode,
                ..
            } => ExecutionState::Decoding,
            WorkUnit::AtomicJob { .. } => ExecutionState::AtomicRunning,
            WorkUnit::PipelineStage { .. } => ExecutionState::PipelineRunning,
        };
        if tracker.state() != running_state {
            tracker.transition(running_state)?;
        }
        tracker.begin_plan(&plan)?;
        if self.active_plans.insert(plan.plan_id, plan).is_some() {
            return Err(Error::InferenceError(format!(
                "execution plan {} was prepared twice",
                scheduled.plan_id
            )));
        }
        Ok(())
    }

    fn report_from_result(result: &ExecutorStepResult) -> ExecutionReport {
        let output = &result.output;
        ExecutionReport {
            plan_id: result.plan_id,
            session: result.session.clone(),
            input_consumed: output.tokens_processed,
            output_produced: output.tokens_generated,
            observed_resources: ResourceVector::zero(),
            elapsed: std::time::Duration::ZERO,
            safe_point: result.safe_point,
            disposition: result.disposition.clone(),
        }
    }

    fn commit_executor_result(
        &mut self,
        mut result: ExecutorStepResult,
        step_time_ms: f64,
    ) -> ExecutorOutput {
        let Some(plan) = self.active_plans.remove(&result.plan_id) else {
            return ExecutorOutput::error(
                result.session.request_id,
                "executor returned an inactive or already committed plan",
            );
        };
        let report = Self::report_from_result(&result);
        let commit_result = self
            .execution_trackers
            .get_mut(&plan.session.request_id)
            .ok_or_else(|| {
                Error::InferenceError("execution tracker is missing for active plan".to_string())
            })
            .and_then(|tracker| tracker.commit(&plan, &report));

        if let Err(err) = commit_result {
            let failure_report = ExecutionReport {
                plan_id: plan.plan_id,
                session: plan.session.clone(),
                input_consumed: 0,
                output_produced: 0,
                observed_resources: ResourceVector::zero(),
                elapsed: std::time::Duration::ZERO,
                safe_point: true,
                disposition: ExecutionDisposition::Failed(ExecutionFailure::invalid_output(
                    err.to_string(),
                )),
            };
            if let Some(tracker) = self.execution_trackers.get_mut(&plan.session.request_id) {
                let _ = tracker.commit(&plan, &failure_report);
            }
            return ExecutorOutput::error(
                plan.session.request_id,
                format!("invalid executor result: {err}"),
            );
        }

        self.scheduler.update_after_step(
            &plan.session.request_id,
            result.output.tokens_processed,
            result.output.tokens_generated,
            Vec::new(),
            step_time_ms,
        );
        result.output.request_id = plan.session.request_id;
        result.output
    }

    fn build_compatible_subbatches<'a>(
        &self,
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

        let mut groups: Vec<(BatchKey, Vec<super::scheduler::ScheduledRequest>)> = Vec::new();

        for item in scheduled {
            if !request_by_id.contains_key(item.request_id.as_str()) {
                continue;
            }
            let Some(key) = self
                .active_plans
                .get(&item.plan_id)
                .map(|plan| plan.batch_key.clone())
            else {
                warn!(
                    plan_id = item.plan_id,
                    request_id = %item.request_id,
                    "Skipping scheduled work without an active execution plan"
                );
                continue;
            };
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
            input_transcription,
            tokens_processed,
            tokens_generated,
            finished,
            phase_timing_override,
            asr_diagnostics,
            error,
        } = current;

        merged.audio = Self::merge_audio_output(merged.audio.take(), audio);
        if text.is_some() {
            merged.text = text;
        }
        if input_transcription.is_some() {
            merged.input_transcription = input_transcription;
        }
        merged.tokens_processed = merged.tokens_processed.saturating_add(tokens_processed);
        merged.tokens_generated = merged.tokens_generated.saturating_add(tokens_generated);
        merged.finished |= finished;
        merged.phase_timing_override = phase_timing_override.or(merged.phase_timing_override);
        merged.asr_diagnostics = asr_diagnostics.or(merged.asr_diagnostics);
        if error.is_some() {
            merged.error = error;
        }

        merged
    }

    fn has_user_visible_output(exec_output: &ExecutorOutput) -> bool {
        exec_output.tokens_generated > 0
            || exec_output
                .text
                .as_ref()
                .is_some_and(|text| !text.is_empty())
            || exec_output
                .audio
                .as_ref()
                .is_some_and(|audio| !audio.samples.is_empty())
    }

    fn failed_step_result(
        scheduled: &super::scheduler::ScheduledRequest,
        message: impl Into<String>,
    ) -> ExecutorStepResult {
        ExecutorStepResult::new(
            scheduled,
            ExecutorOutput::error(scheduled.request_id.clone(), message),
        )
    }

    fn reconcile_executor_outputs(
        phase: &str,
        scheduled: &[super::scheduler::ScheduledRequest],
        result: Result<Vec<ExecutorStepResult>>,
    ) -> Vec<ExecutorStepResult> {
        let expected: HashSet<_> = scheduled
            .iter()
            .map(|entry| (entry.plan_id, entry.session_key()))
            .collect();
        let outputs = match result {
            Ok(outputs) => outputs,
            Err(err) => {
                return scheduled
                    .iter()
                    .map(|entry| {
                        Self::failed_step_result(entry, format!("{phase} executor failed: {err}"))
                    })
                    .collect();
            }
        };

        let mut by_transaction = HashMap::new();
        let mut duplicates = HashSet::new();
        for mut result in outputs {
            let key = (result.plan_id, result.session.clone());
            if !expected.contains(&key) {
                warn!(
                    phase,
                    plan_id = result.plan_id,
                    request_id = %result.session.request_id,
                    session_epoch = result.session.epoch,
                    "Ignoring executor output for an unknown or stale transaction"
                );
                continue;
            }
            if result.output.request_id != result.session.request_id {
                result.output = ExecutorOutput::error(
                    result.session.request_id.clone(),
                    format!("{phase} executor output request ID did not match its session"),
                );
            } else if result.output.error.is_some() {
                result.output.finished = true;
            }
            if by_transaction.insert(key.clone(), result).is_some() {
                duplicates.insert(key);
            }
        }

        scheduled
            .iter()
            .map(|entry| {
                let key = (entry.plan_id, entry.session_key());
                if duplicates.contains(&key) {
                    return Self::failed_step_result(
                        entry,
                        format!("{phase} executor returned duplicate outputs"),
                    );
                }
                by_transaction.remove(&key).unwrap_or_else(|| {
                    Self::failed_step_result(
                        entry,
                        format!("{phase} executor did not return a scheduled output"),
                    )
                })
            })
            .collect()
    }

    async fn execute_decode_subbatch(
        &self,
        request_refs: &[&EngineCoreRequest],
        scheduled: &[super::scheduler::ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        // A scheduler plan is dispatched exactly once. Reusing one plan ID for
        // internal one-token rounds makes completion fencing ambiguous; native
        // continuous batching will expose child quanta explicitly in Phase 8.
        let result = self.executor.execute_decode(request_refs, scheduled).await;
        Ok(Self::reconcile_executor_outputs(
            "decode", scheduled, result,
        ))
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
            execution_trackers: HashMap::new(),
            active_plans: HashMap::new(),
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
        self.refresh_scheduler_execution_profiles().await;
        self.kv_cache.maintenance()?;
        self.maintenance_steps = self.maintenance_steps.saturating_add(1);
        if self.maintenance_steps % 64 == 0 {
            self.kv_cache.compact_shared_prefixes();
        }
        let schedule_result = self.scheduler.schedule(self.kv_cache.inner_mut());

        let mut preemption_failures = Vec::new();
        for session in &schedule_result.preempted_requests {
            let release = self.executor.cleanup_session(session).await;
            if !release.confirmed {
                // The scheduler already rolled logical progress back. If the
                // executor cannot prove physical state release, abort this
                // incarnation rather than risk stale-cache recomputation.
                self.scheduler
                    .abort_request(&session.request_id, self.kv_cache.inner_mut());
                self.requests.remove(&session.request_id);
                self.request_start_times.remove(&session.request_id);
                self.request_phase_timings.remove(&session.request_id);
                preemption_failures.push(ExecutorOutput::error(
                    session.request_id.clone(),
                    "executor could not confirm physical cache release during preemption",
                ));
            }
            if self
                .execution_trackers
                .get(&session.request_id)
                .is_some_and(|tracker| tracker.session() == session)
            {
                self.execution_trackers.remove(&session.request_id);
            }
            self.active_plans.retain(|_, plan| plan.session != *session);
            self.request_phase_timings
                .entry(session.request_id.clone())
                .and_modify(|timing| *timing = RequestPhaseTiming::default());
        }

        let expired_sequence_ids: HashMap<_, _> = schedule_result
            .expired_requests
            .iter()
            .map(|request| (request.request_id.clone(), request.sequence_id))
            .collect();
        let mut terminal_outputs =
            Vec::with_capacity(schedule_result.expired_requests.len() + preemption_failures.len());
        terminal_outputs.extend(preemption_failures);
        for request in &schedule_result.expired_requests {
            terminal_outputs.push(ExecutorOutput::error(
                request.request_id.clone(),
                REQUEST_DEADLINE_EXCEEDED,
            ));
        }

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
        for scheduled in decode_scheduled.iter().chain(prefill_scheduled.iter()) {
            self.begin_execution_plan(scheduled).await?;
        }
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

        if prefill_request_refs.is_empty()
            && decode_request_refs.is_empty()
            && terminal_outputs.is_empty()
        {
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
                self.build_compatible_subbatches(&decode_request_refs, &decode_scheduled);
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
                self.build_compatible_subbatches(&prefill_request_refs, &prefill_scheduled);
            let mut outputs = Vec::new();
            for (refs, batch) in sub_batches {
                let result = self.executor.execute_prefill(&refs, &batch).await;
                outputs.extend(Self::reconcile_executor_outputs("prefill", &batch, result));
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
        let executor_results = decode_outputs;
        let mut executor_outputs =
            Vec::with_capacity(executor_results.len() + terminal_outputs.len());
        for result in executor_results {
            let step_time_ms = if decode_ids.contains(&result.session.request_id) {
                decode_step_ms
            } else {
                prefill_step_ms
            };
            executor_outputs.push(self.commit_executor_result(result, step_time_ms));
        }
        executor_outputs.append(&mut terminal_outputs);

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
            let generation_time_ms = generation_time.as_secs_f64() * 1000.0;

            // Get sequence ID from scheduler
            let sequence_id = expired_sequence_ids
                .get(&request_id)
                .copied()
                .or_else(|| self.scheduler.get_sequence_id(&request_id))
                .unwrap_or(0);

            if let Some(override_timing) = exec_output.phase_timing_override.as_ref() {
                let phase = self
                    .request_phase_timings
                    .entry(request_id.clone())
                    .or_default();
                merge_optional_phase_ms(
                    &mut phase.media_decode_ms,
                    override_timing.media_decode_ms,
                );
                merge_optional_phase_ms(
                    &mut phase.normalization_ms,
                    override_timing.normalization_ms,
                );
                if let Some(prefill_ms) = override_timing.prefill_ms {
                    phase.prefill_ms = prefill_ms.max(0.0);
                }
                if let Some(decode_ms) = override_timing.decode_ms {
                    phase.decode_ms = decode_ms.max(0.0);
                }
                merge_optional_phase_ms(&mut phase.sampling_ms, override_timing.sampling_ms);
                merge_optional_phase_ms(&mut phase.codec_ms, override_timing.codec_ms);
                merge_optional_phase_ms(&mut phase.postprocess_ms, override_timing.postprocess_ms);
                if let Some(prefill_steps) = override_timing.prefill_steps {
                    phase.prefill_steps = prefill_steps;
                }
                if let Some(decode_steps) = override_timing.decode_steps {
                    phase.decode_steps = decode_steps;
                }
                if let Some(first_output_ms_since_start) =
                    override_timing.first_output_ms_since_start
                {
                    let request_relative_first_output = (phase.queue_wait_ms
                        + first_output_ms_since_start.max(0.0))
                    .min(generation_time_ms);
                    phase
                        .first_output_ms
                        .get_or_insert(request_relative_first_output);
                }
            }

            // Process output
            let mut engine_output =
                self.output_processor
                    .process(exec_output.clone(), sequence_id, generation_time);
            engine_output.token_stats.prompt_tokens = self
                .requests
                .get(&request_id)
                .map(|request| request.num_prompt_tokens())
                .unwrap_or(engine_output.token_stats.prompt_tokens);
            if exec_output.finished {
                if let Some((_, total_generated)) = self.scheduler.get_running_info(&request_id) {
                    let resolved_total = total_generated.max(engine_output.num_tokens);
                    engine_output.num_tokens = resolved_total;
                    engine_output.token_stats.generated_tokens = resolved_total;
                }
            }
            if Self::has_user_visible_output(&exec_output) {
                if let Some(phase) = self.request_phase_timings.get_mut(&request_id) {
                    phase.first_output_ms.get_or_insert(generation_time_ms);
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
                    media_decode_ms: phase.media_decode_ms,
                    normalization_ms: phase.normalization_ms,
                    prefill_ms: phase.prefill_ms,
                    decode_ms: phase.decode_ms,
                    sampling_ms: phase.sampling_ms,
                    codec_ms: phase.codec_ms,
                    postprocess_ms: phase.postprocess_ms,
                    ttft_ms: phase.first_output_ms,
                    total_ms: generation_time_ms,
                    prefill_steps: phase.prefill_steps,
                    decode_steps: phase.decode_steps,
                });
            }

            // Update scheduler state
            if exec_output.finished {
                if let Some(session) = self
                    .execution_trackers
                    .get(&request_id)
                    .map(|tracker| tracker.session().clone())
                {
                    self.executor.cleanup_session(&session).await;
                } else {
                    self.executor.cleanup_request(&request_id).await;
                }
                self.scheduler
                    .finish_request(&request_id, self.kv_cache.inner_mut());
                self.requests.remove(&request_id);
                self.request_start_times.remove(&request_id);
                self.request_phase_timings.remove(&request_id);
                self.execution_trackers.remove(&request_id);
                self.active_plans
                    .retain(|_, plan| plan.session.request_id != request_id);
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

    /// Get the set of model variants currently referenced by active engine requests.
    pub fn active_model_variants(&self) -> HashSet<ModelVariant> {
        self.requests
            .values()
            .filter_map(|request| request.model_variant)
            .collect()
    }

    /// Get request status.
    pub fn get_request_status(&self, request_id: &RequestId) -> Option<RequestStatus> {
        self.scheduler.get_status(request_id)
    }

    /// Return the current scheduler incarnation for a public request ID.
    pub fn get_session_key(&self, request_id: &RequestId) -> Option<super::SessionKey> {
        self.scheduler
            .get_sequence_id(request_id)
            .map(|epoch| super::SessionKey::new(request_id.clone(), epoch))
    }

    /// Abort only if the caller still owns the exact request incarnation.
    pub async fn abort_request_session(&mut self, session: &super::SessionKey) -> bool {
        if self.scheduler.get_sequence_id(&session.request_id) != Some(session.epoch) {
            return false;
        }
        self.abort_request(&session.request_id).await
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
            self.execution_trackers.remove(request_id);
            self.active_plans
                .retain(|_, plan| plan.session.request_id != *request_id);
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

    /// Abort every request tracked by the core and release executor state.
    pub async fn abort_all_requests(&mut self) -> Vec<RequestId> {
        let request_ids: Vec<_> = self.requests.keys().cloned().collect();
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
    use super::super::executor::{
        ExecutorOutput, ExecutorPhaseTiming, ExecutorStepResult, ModelExecutor,
    };
    use super::super::scheduler::ScheduledRequest;
    use super::super::types::{AudioOutput, Priority};
    use super::*;
    use crate::model::ModelVariant;
    use crate::models::shared::chat::{ChatMessage, ChatRole};
    use std::sync::{Arc, Mutex};

    fn scheduled_prefill(request_id: &str, sequence_id: u64) -> ScheduledRequest {
        ScheduledRequest {
            plan_id: sequence_id + 1,
            request_id: request_id.to_string(),
            sequence_id,
            num_tokens: 1,
            is_prefill: true,
            block_ids: Vec::new(),
            num_computed_tokens: 0,
            work: crate::engine::WorkUnit::SequenceStep {
                phase: crate::engine::SequencePhase::Prefill,
                input: crate::engine::InputRange { start: 0, end: 1 },
                max_output_steps: 1,
            },
        }
    }

    fn wrap_outputs(
        scheduled: &[ScheduledRequest],
        outputs: Vec<ExecutorOutput>,
    ) -> Vec<ExecutorStepResult> {
        assert_eq!(scheduled.len(), outputs.len());
        scheduled
            .iter()
            .zip(outputs)
            .map(|(scheduled, output)| ExecutorStepResult::new(scheduled, output))
            .collect()
    }

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
                    input_transcription: None,
                    tokens_processed: entry.num_tokens.max(1),
                    tokens_generated: usize::from(!entry.is_prefill),
                    finished: false,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: None,
                })
                .collect()
        }
    }

    impl ModelExecutor for MockExecutor {
        fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
            let mut profile = ExecutionProfile::fail_closed(
                BackendKind::Cpu,
                request.model_variant,
                ExecutionMode::Sequence,
            );
            profile.prefill = PrefillMode::Full;
            profile.recompute_safe = true;
            profile.cache_release_safe = true;
            Some(profile)
        }

        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(wrap_outputs(scheduled, Self::build_outputs(scheduled)))
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(wrap_outputs(scheduled, Self::build_outputs(scheduled)))
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

        fn cleanup_request(&self, request_id: &str) -> super::super::executor::CacheReleaseReport {
            if let Ok(mut calls) = self.cleanup_calls.lock() {
                calls.push(request_id.to_string());
            }
            super::super::executor::CacheReleaseReport::confirmed(1)
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
        ) -> Result<Vec<ExecutorStepResult>> {
            let outputs = scheduled
                .iter()
                .map(|entry| ExecutorOutput {
                    request_id: entry.request_id.clone(),
                    audio: Some(AudioOutput::empty(24_000)),
                    text: None,
                    input_transcription: None,
                    tokens_processed: entry.num_tokens.max(1),
                    tokens_generated: 0,
                    finished: false,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: None,
                })
                .collect();
            Ok(wrap_outputs(scheduled, outputs))
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            if let Ok(mut calls) = self.decode_calls.lock() {
                calls.push(
                    scheduled
                        .iter()
                        .map(|entry| (entry.request_id.clone(), entry.num_tokens))
                        .collect(),
                );
            }
            let outputs = scheduled
                .iter()
                .map(|entry| ExecutorOutput {
                    request_id: entry.request_id.clone(),
                    audio: Some(AudioOutput::empty(24_000)),
                    text: Some(format!("step-{}", entry.request_id)),
                    input_transcription: None,
                    tokens_processed: entry.num_tokens.max(1),
                    tokens_generated: entry.num_tokens.max(1),
                    finished: false,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: None,
                })
                .collect();
            Ok(wrap_outputs(scheduled, outputs))
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

    struct ImmediateFinishExecutor;

    impl ModelExecutor for ImmediateFinishExecutor {
        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            let outputs = scheduled
                .iter()
                .map(|entry| ExecutorOutput {
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
                })
                .collect();
            Ok(wrap_outputs(scheduled, outputs))
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            _scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(Vec::new())
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

    struct SelectiveFailureExecutor;

    impl ModelExecutor for SelectiveFailureExecutor {
        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            if scheduled.iter().any(|entry| entry.request_id == "bad") {
                return Err(Error::InferenceError("isolated failure".to_string()));
            }
            let outputs = scheduled
                .iter()
                .map(|entry| ExecutorOutput {
                    request_id: entry.request_id.clone(),
                    audio: None,
                    text: Some("ok".to_string()),
                    input_transcription: None,
                    tokens_processed: entry.num_tokens,
                    tokens_generated: 1,
                    finished: true,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: None,
                })
                .collect();
            Ok(wrap_outputs(scheduled, outputs))
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            _scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(Vec::new())
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

    struct PhaseTimingExecutor;

    impl ModelExecutor for PhaseTimingExecutor {
        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            let outputs = scheduled
                .iter()
                .map(|entry| ExecutorOutput {
                    request_id: entry.request_id.clone(),
                    audio: None,
                    text: Some("done".to_string()),
                    input_transcription: None,
                    tokens_processed: entry.num_tokens.max(1),
                    tokens_generated: 1,
                    finished: true,
                    phase_timing_override: Some(ExecutorPhaseTiming {
                        media_decode_ms: Some(12.5),
                        sampling_ms: Some(1.25),
                        ..ExecutorPhaseTiming::default()
                    }),
                    asr_diagnostics: None,
                    error: None,
                })
                .collect();
            Ok(wrap_outputs(scheduled, outputs))
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            _scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(Vec::new())
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
    async fn test_step_clears_executor_state_for_recompute_preemption() {
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
            calls.iter().any(|id| id == "low-priority"),
            "recompute preemption must clear stale executor decode state"
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

    #[tokio::test]
    async fn stale_session_cannot_abort_reused_request_id() {
        let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls)));
        let mut core =
            EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
        let mut first = EngineCoreRequest::tts("first");
        first.id = "reused".to_string();
        core.add_request(first).unwrap();
        let first_session = core.get_session_key(&"reused".to_string()).unwrap();
        assert!(core.abort_request_session(&first_session).await);

        let mut second = EngineCoreRequest::tts("second");
        second.id = "reused".to_string();
        core.add_request(second).unwrap();
        let second_session = core.get_session_key(&"reused".to_string()).unwrap();
        assert_ne!(first_session.epoch, second_session.epoch);
        assert!(!core.abort_request_session(&first_session).await);
        assert!(core.has_request(&"reused".to_string()));
    }

    #[test]
    fn test_merge_executor_output_replaces_cumulative_audio_snapshots() {
        let first = ExecutorOutput {
            request_id: "req-a".to_string(),
            audio: Some(AudioOutput::new(vec![0.1, 0.2], 24_000)),
            text: Some("hello".to_string()),
            input_transcription: None,
            tokens_processed: 1,
            tokens_generated: 1,
            finished: false,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        };
        let second = ExecutorOutput {
            request_id: "req-a".to_string(),
            audio: Some(AudioOutput::new(vec![0.1, 0.2, 0.3], 24_000)),
            text: Some("hello world".to_string()),
            input_transcription: Some("hello there".to_string()),
            tokens_processed: 1,
            tokens_generated: 1,
            finished: true,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        };

        let merged = EngineCore::merge_executor_output(Some(first), second);
        let audio = merged.audio.expect("merged audio");
        assert_eq!(audio.samples, vec![0.1, 0.2, 0.3]);
        assert_eq!(merged.input_transcription.as_deref(), Some("hello there"));
        assert_eq!(merged.text.as_deref(), Some("hello world"));
        assert_eq!(merged.tokens_processed, 2);
        assert_eq!(merged.tokens_generated, 2);
        assert!(merged.finished);
    }

    #[test]
    fn executor_output_reconciliation_rejects_duplicates_unknowns_and_missing_ids() {
        let scheduled = vec![scheduled_prefill("req-a", 0), scheduled_prefill("req-b", 1)];
        let mut executor_outputs = wrap_outputs(
            &scheduled[..1],
            MockExecutor::build_outputs(&scheduled[..1]),
        );
        executor_outputs.push(executor_outputs[0].clone());
        let mut unknown = executor_outputs[0].clone();
        unknown.plan_id = 999;
        unknown.session = crate::engine::SessionKey::new("unknown".to_string(), 999);
        unknown.output.request_id = "unknown".to_string();
        executor_outputs.push(unknown);

        let reconciled =
            EngineCore::reconcile_executor_outputs("prefill", &scheduled, Ok(executor_outputs));

        assert_eq!(
            reconciled
                .iter()
                .map(|result| result.output.request_id.as_str())
                .collect::<Vec<_>>(),
            vec!["req-a", "req-b"]
        );
        assert!(reconciled.iter().all(|result| result.output.finished));
        assert!(reconciled
            .iter()
            .all(|result| result.output.error.as_deref().is_some()));
        assert!(reconciled[0]
            .output
            .error
            .as_deref()
            .unwrap()
            .contains("duplicate"));
        assert!(reconciled[1]
            .output
            .error
            .as_deref()
            .unwrap()
            .contains("did not return"));
    }

    #[tokio::test]
    async fn test_execute_decode_subbatch_dispatches_each_plan_exactly_once() {
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
                plan_id: 1,
                request_id: req_a.id.clone(),
                sequence_id: 0,
                num_tokens: 3,
                is_prefill: false,
                block_ids: Vec::new(),
                num_computed_tokens: 0,
                work: crate::engine::WorkUnit::SequenceStep {
                    phase: crate::engine::SequencePhase::Decode,
                    input: crate::engine::InputRange { start: 0, end: 3 },
                    max_output_steps: 3,
                },
            },
            ScheduledRequest {
                plan_id: 2,
                request_id: req_b.id.clone(),
                sequence_id: 1,
                num_tokens: 2,
                is_prefill: false,
                block_ids: Vec::new(),
                num_computed_tokens: 0,
                work: crate::engine::WorkUnit::SequenceStep {
                    phase: crate::engine::SequencePhase::Decode,
                    input: crate::engine::InputRange { start: 0, end: 2 },
                    max_output_steps: 2,
                },
            },
        ];

        let outputs = core
            .execute_decode_subbatch(&req_refs, &scheduled)
            .await
            .unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].output.request_id, "req-a");
        assert_eq!(outputs[0].output.tokens_processed, 3);
        assert_eq!(outputs[0].output.tokens_generated, 3);
        assert_eq!(outputs[1].output.request_id, "req-b");
        assert_eq!(outputs[1].output.tokens_processed, 2);
        assert_eq!(outputs[1].output.tokens_generated, 2);

        let calls = decode_calls.lock().unwrap().clone();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0],
            vec![("req-a".to_string(), 3), ("req-b".to_string(), 2)]
        );
    }

    #[tokio::test]
    async fn executor_progress_is_validated_before_scheduler_mutation() {
        let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls)));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 8,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        let mut request = EngineCoreRequest::tts("transaction");
        request.id = "transaction".to_string();
        request.prompt_tokens = vec![1];
        core.add_request(request).unwrap();
        let scheduled = core
            .scheduler
            .schedule(core.kv_cache.inner_mut())
            .prefill_requests
            .remove(0);
        core.begin_execution_plan(&scheduled).await.unwrap();

        let mut invalid = ExecutorStepResult::new(
            &scheduled,
            ExecutorOutput {
                request_id: scheduled.request_id.clone(),
                audio: None,
                text: None,
                input_transcription: None,
                tokens_processed: scheduled.num_tokens + 1,
                tokens_generated: 0,
                finished: false,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
            },
        );
        invalid.disposition = ExecutionDisposition::Progress;
        let output = core.commit_executor_result(invalid, 1.0);

        assert!(output.finished);
        assert!(output
            .error
            .as_deref()
            .is_some_and(|message| message.contains("beyond the scheduled quantum")));
        assert_eq!(
            core.scheduler.get_running_info(&scheduled.request_id),
            Some((0, 0)),
            "invalid progress must not reach scheduler accounting"
        );
    }

    #[tokio::test]
    async fn full_prefill_plan_covers_the_model_owned_prompt_operation() {
        let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls)));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 4,
                enable_chunked_prefill: true,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        let mut request = EngineCoreRequest::tts("full prefill");
        request.id = "full-prefill".to_string();
        request.prompt_tokens = (0..16).collect();
        core.add_request(request).unwrap();
        let scheduled = core
            .scheduler
            .schedule(core.kv_cache.inner_mut())
            .prefill_requests
            .remove(0);
        assert!(scheduled.num_tokens < 16);
        core.begin_execution_plan(&scheduled).await.unwrap();
        let plan = core.active_plans.get(&scheduled.plan_id).unwrap();
        assert!(matches!(
            plan.work,
            WorkUnit::SequenceStep {
                input: super::super::InputRange { start: 0, end: 16 },
                max_output_steps: 1,
                ..
            }
        ));

        let result = ExecutorStepResult::new(
            &scheduled,
            ExecutorOutput {
                request_id: scheduled.request_id.clone(),
                audio: None,
                text: None,
                input_transcription: None,
                tokens_processed: 16,
                tokens_generated: 1,
                finished: false,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
            },
        );
        let output = core.commit_executor_result(result, 1.0);
        assert!(output.error.is_none());
        assert_eq!(
            core.scheduler.get_running_info(&scheduled.request_id),
            Some((16, 1))
        );
    }

    #[tokio::test]
    async fn valid_executor_progress_commits_exactly_once() {
        let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls)));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 8,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        let mut request = EngineCoreRequest::tts("transaction");
        request.id = "valid-transaction".to_string();
        request.prompt_tokens = vec![1];
        core.add_request(request).unwrap();
        let scheduled = core
            .scheduler
            .schedule(core.kv_cache.inner_mut())
            .prefill_requests
            .remove(0);
        core.begin_execution_plan(&scheduled).await.unwrap();
        let result = ExecutorStepResult::new(
            &scheduled,
            MockExecutor::build_outputs(std::slice::from_ref(&scheduled)).remove(0),
        );

        let output = core.commit_executor_result(result.clone(), 1.0);
        assert!(output.error.is_none());
        assert_eq!(
            core.scheduler.get_running_info(&scheduled.request_id),
            Some((scheduled.num_tokens, 0))
        );

        let duplicate = core.commit_executor_result(result, 1.0);
        assert!(duplicate
            .error
            .as_deref()
            .is_some_and(|message| message.contains("already committed")));
        assert_eq!(
            core.scheduler.get_running_info(&scheduled.request_id),
            Some((scheduled.num_tokens, 0)),
            "duplicate results must not advance progress twice"
        );
    }

    #[tokio::test]
    async fn unresolved_executor_is_planned_as_atomic_and_must_finish() {
        let executor = UnifiedExecutor::new_for_test(Box::new(ImmediateFinishExecutor));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        let mut request = EngineCoreRequest::tts("atomic");
        request.id = "atomic".to_string();
        request.prompt_tokens = vec![1];
        core.add_request(request).unwrap();
        let scheduled = core
            .scheduler
            .schedule(core.kv_cache.inner_mut())
            .prefill_requests
            .remove(0);
        core.begin_execution_plan(&scheduled).await.unwrap();
        assert!(matches!(
            core.active_plans
                .get(&scheduled.plan_id)
                .map(|plan| &plan.work),
            Some(WorkUnit::AtomicJob { .. })
        ));

        let result = ExecutorStepResult::new(
            &scheduled,
            ExecutorOutput {
                request_id: scheduled.request_id.clone(),
                audio: None,
                text: None,
                input_transcription: None,
                tokens_processed: 1,
                tokens_generated: 0,
                finished: false,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
            },
        );
        let output = core.commit_executor_result(result, 1.0);
        assert!(output
            .error
            .as_deref()
            .is_some_and(|message| message.contains("atomic execution must finish")));
        assert_eq!(
            core.scheduler.get_running_info(&scheduled.request_id),
            Some((0, 0))
        );
    }

    #[tokio::test]
    async fn executor_subbatch_failure_is_isolated_and_reconciled() {
        let executor = UnifiedExecutor::new_for_test(Box::new(SelectiveFailureExecutor));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 2,
                ..Default::default()
            },
            executor,
        )
        .unwrap();

        let mut bad = EngineCoreRequest::tts("bad");
        bad.id = "bad".to_string();
        bad.prompt_tokens = vec![1];
        bad.model_variant = Some(ModelVariant::Qwen34BGguf);
        let mut good = EngineCoreRequest::tts("good");
        good.id = "good".to_string();
        good.prompt_tokens = vec![1];
        good.model_variant = Some(ModelVariant::Qwen38BGguf);
        core.add_request(bad).unwrap();
        core.add_request(good).unwrap();

        let outputs = core.step().await.unwrap();

        assert_eq!(outputs.len(), 2);
        assert!(outputs
            .iter()
            .any(|output| output.request_id == "bad" && output.error.is_some()));
        assert!(outputs
            .iter()
            .any(|output| output.request_id == "good" && output.error.is_none()));
        assert!(!core.has_request(&"bad".to_string()));
        assert!(!core.has_request(&"good".to_string()));
    }

    #[tokio::test]
    async fn hard_deadline_returns_one_terminal_output_with_original_sequence() {
        let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
        let executor =
            UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls.clone())));
        let mut core =
            EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
        let mut request = EngineCoreRequest::tts("expired")
            .with_deadline(Some(Instant::now() - std::time::Duration::from_millis(1)));
        request.id = "expired".to_string();
        core.add_request(request).unwrap();

        let outputs = core.step().await.unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].request_id, "expired");
        assert_eq!(outputs[0].sequence_id, 0);
        assert!(outputs[0].is_finished);
        assert_eq!(outputs[0].num_tokens, 0);
        assert_eq!(outputs[0].error.as_deref(), Some(REQUEST_DEADLINE_EXCEEDED));
        assert!(!core.has_request(&"expired".to_string()));
        assert_eq!(
            cleanup_calls
                .lock()
                .unwrap()
                .iter()
                .filter(|id| id.as_str() == "expired")
                .count(),
            1,
            "deadline cleanup must occur exactly once"
        );
    }

    #[tokio::test]
    async fn test_step_records_exact_prompt_tokens_and_ttft() {
        let executor = UnifiedExecutor::new_for_test(Box::new(ImmediateFinishExecutor));
        let config = EngineCoreConfig {
            enable_chunked_prefill: false,
            block_size: 1,
            max_blocks: 8,
            ..Default::default()
        };
        let mut core = EngineCore::new_with_unified_executor(config, executor).unwrap();

        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".to_string(),
        }]);
        request.id = "chat-req".to_string();
        request.prompt_tokens = vec![11, 22, 33, 44];

        core.add_request(request).unwrap();
        let outputs = core.step().await.unwrap();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].token_stats.prompt_tokens, 4);
        let latency = outputs[0]
            .latency_breakdown
            .as_ref()
            .expect("latency breakdown");
        assert!(latency.ttft_ms.is_some());
        assert!(latency.ttft_ms.unwrap() >= 0.0);
    }

    #[tokio::test]
    async fn test_step_preserves_optional_executor_phase_timings() {
        let executor = UnifiedExecutor::new_for_test(Box::new(PhaseTimingExecutor));
        let config = EngineCoreConfig {
            enable_chunked_prefill: false,
            block_size: 1,
            max_blocks: 8,
            ..Default::default()
        };
        let mut core = EngineCore::new_with_unified_executor(config, executor).unwrap();

        let mut request = EngineCoreRequest::tts("phase timing");
        request.id = "phase-req".to_string();
        request.prompt_tokens = vec![1, 2, 3];

        core.add_request(request).unwrap();
        let outputs = core.step().await.unwrap();

        let latency = outputs[0]
            .latency_breakdown
            .as_ref()
            .expect("latency breakdown");
        assert_eq!(latency.media_decode_ms, Some(12.5));
        assert_eq!(latency.sampling_ms, Some(1.25));
        assert!(
            latency.prefill_ms >= 0.0,
            "scheduler prefill timing remains available"
        );
        assert_eq!(latency.decode_ms, 0.0);
    }
}
