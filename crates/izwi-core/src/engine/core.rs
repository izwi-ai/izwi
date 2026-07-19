//! Engine core - the central orchestrator for inference.
//!
//! The engine core coordinates:
//! - Request scheduling
//! - Model execution
//! - KV cache management
//! - Output processing

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use super::config::EngineCoreConfig;
use super::execution::{
    BatchDispatch, BatchKey, CacheMode, ExecutionDisposition, ExecutionFailure, ExecutionMode,
    ExecutionPlan, ExecutionProfile, ExecutionReport, ExecutionState, ExecutionTracker,
    FinishReason as ExecutionFinishReason, NativeBatchMode, PrefillMode, RetryDisposition,
    WorkUnit,
};
#[cfg(test)]
use super::executor::REQUEST_DEADLINE_EXCEEDED;
use super::executor::{
    CacheReleaseReport, ExecutorOutput, ExecutorStepResult, UnifiedExecutor, WorkerConfig,
};
use super::kv_cache::{KVCacheConfig, KVCacheManager, KVCacheStats};
use super::metal_kv_cache::{MetalKVCacheConfig, MetalKVCacheManager};
use super::metrics::record_engine_batch_dispatch;
use super::output::OutputProcessor;
use super::request::{EngineCoreRequest, RequestStatus};
use super::scheduler::{BeginTerminalRelease, Scheduler, SchedulerConfig, TerminalReleaseCause};
use super::types::{AudioOutput, EngineOutput, LatencyBreakdown, RequestId};
use super::{ResourceAmount, ResourceVector};
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

#[derive(Debug)]
struct CommittedExecutorOutput {
    session: super::SessionKey,
    output: ExecutorOutput,
    disposition: ExecutionDisposition,
}

#[derive(Debug, Clone, Copy)]
struct LifecycleRetryPolicy {
    max_execution_retries: u32,
    execution_backoff_base: Duration,
    execution_backoff_max: Duration,
    cleanup_backoff_base: Duration,
    cleanup_backoff_max: Duration,
    cleanup_budget_per_step: usize,
}

impl Default for LifecycleRetryPolicy {
    fn default() -> Self {
        Self {
            max_execution_retries: 3,
            execution_backoff_base: Duration::from_millis(5),
            execution_backoff_max: Duration::from_millis(200),
            cleanup_backoff_base: Duration::from_millis(25),
            cleanup_backoff_max: Duration::from_secs(1),
            cleanup_budget_per_step: 16,
        }
    }
}

impl LifecycleRetryPolicy {
    fn exponential_delay(base: Duration, max: Duration, attempt: u32) -> Duration {
        let exponent = attempt.saturating_sub(1).min(20);
        base.saturating_mul(1u32 << exponent).min(max)
    }

    fn execution_delay(self, attempt: u32) -> Duration {
        Self::exponential_delay(
            self.execution_backoff_base,
            self.execution_backoff_max,
            attempt,
        )
    }

    fn cleanup_delay(self, attempt: u32) -> Duration {
        Self::exponential_delay(self.cleanup_backoff_base, self.cleanup_backoff_max, attempt)
    }
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
    /// Durable typed terminal events created outside a committed executor
    /// result (for example cancellation, deadline expiry, or failed
    /// preemption) and delivered by the next successful engine step.
    pending_terminal_outputs: VecDeque<CommittedExecutorOutput>,
    /// Consecutive retryable executor failures for each exact session.
    execution_retry_attempts: HashMap<super::SessionKey, u32>,
    retry_policy: LifecycleRetryPolicy,
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
        if scheduled.is_prefill
            && profile.prefill == PrefillMode::Full
            && (scheduled.num_computed_tokens != 0
                || scheduled.num_tokens < request.num_prompt_tokens())
        {
            return Err(Error::InferenceError(format!(
                "full-prefill request {} was scheduled as a partial prompt quantum",
                scheduled.request_id
            )));
        }
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
        let estimate = if profile.cache_mode == CacheMode::None {
            ResourceVector::zero()
        } else {
            let bytes_per_block =
                self.config.kv_cache_memory_bytes() / self.config.max_blocks.max(1);
            let estimated_bytes = scheduled
                .block_ids
                .len()
                .checked_mul(bytes_per_block)
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| Error::Overloaded("cache plan estimate overflow".to_string()))?;
            ResourceVector {
                kv_bytes: ResourceAmount::Known(estimated_bytes),
                ..ResourceVector::zero()
            }
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
            batch_mode: if scheduled.is_prefill {
                profile.prefill_batch
            } else {
                profile.decode_batch
            },
            max_batch_size: profile.max_batch_size.max(1),
            estimate,
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
            observed_resources: result.observed_resources,
            dispatch: result.dispatch,
            elapsed: std::time::Duration::ZERO,
            safe_point: result.safe_point,
            disposition: result.disposition.clone(),
            output_finished: output.finished,
            output_has_error: output.error.is_some(),
        }
    }

    async fn commit_executor_result(
        &mut self,
        mut result: ExecutorStepResult,
        step_time_ms: f64,
    ) -> Option<CommittedExecutorOutput> {
        let Some(plan) = self.active_plans.remove(&result.plan_id) else {
            warn!(
                plan_id = result.plan_id,
                request_id = %result.session.request_id,
                session_epoch = result.session.epoch,
                "Ignoring executor result for an inactive or already committed plan"
            );
            return None;
        };

        let retry_attempt = match &mut result.disposition {
            ExecutionDisposition::Failed(failure) if failure.retry != RetryDisposition::Never => {
                let attempts = self
                    .execution_retry_attempts
                    .entry(plan.session.clone())
                    .or_default();
                *attempts = attempts.saturating_add(1);
                if *attempts > self.retry_policy.max_execution_retries {
                    let message = format!(
                        "executor retry budget exhausted after {} attempts: {}",
                        self.retry_policy.max_execution_retries, failure.message
                    );
                    failure.retry = RetryDisposition::Never;
                    failure.message = message.clone();
                    result.output.finished = true;
                    result.output.error = Some(message);
                    None
                } else {
                    Some(*attempts)
                }
            }
            _ => {
                self.execution_retry_attempts.remove(&plan.session);
                None
            }
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
                dispatch: BatchDispatch::serial(),
                elapsed: std::time::Duration::ZERO,
                safe_point: true,
                disposition: ExecutionDisposition::Failed(ExecutionFailure::invalid_output(
                    err.to_string(),
                )),
                output_finished: true,
                output_has_error: true,
            };
            if let Some(tracker) = self.execution_trackers.get_mut(&plan.session.request_id) {
                let _ = tracker.commit(&plan, &failure_report);
            }
            let message = format!("invalid executor result: {err}");
            return Some(CommittedExecutorOutput {
                session: plan.session.clone(),
                output: ExecutorOutput::error(plan.session.request_id, message.clone()),
                disposition: ExecutionDisposition::Failed(ExecutionFailure::invalid_output(
                    message,
                )),
            });
        }

        match &result.disposition {
            ExecutionDisposition::Failed(failure)
                if failure.retry == RetryDisposition::RetrySameSession =>
            {
                let attempt = retry_attempt.unwrap_or(1);
                let retry_at = Instant::now() + self.retry_policy.execution_delay(attempt);
                if self
                    .scheduler
                    .defer_execution_retry(&plan.session, retry_at)
                {
                    return None;
                }
                let message = "scheduler rejected a same-session execution retry";
                return Some(CommittedExecutorOutput {
                    session: plan.session.clone(),
                    output: ExecutorOutput::error(
                        plan.session.request_id.clone(),
                        message.to_string(),
                    ),
                    disposition: ExecutionDisposition::Failed(ExecutionFailure::invalid_output(
                        message,
                    )),
                });
            }
            ExecutionDisposition::Failed(failure)
                if failure.retry == RetryDisposition::Recompute =>
            {
                let release = self.executor.cleanup_session(&plan.session).await;
                if release.confirmed
                    && self
                        .scheduler
                        .restart_request_for_recompute(&plan.session, self.kv_cache.inner_mut())
                {
                    let attempt = retry_attempt.unwrap_or(1);
                    let retry_at = Instant::now() + self.retry_policy.execution_delay(attempt);
                    if !self
                        .scheduler
                        .defer_execution_retry(&plan.session, retry_at)
                    {
                        let message = "scheduler rejected a deferred recompute retry";
                        return Some(CommittedExecutorOutput {
                            session: plan.session.clone(),
                            output: ExecutorOutput::error(plan.session.request_id.clone(), message),
                            disposition: ExecutionDisposition::Failed(
                                ExecutionFailure::invalid_output(message),
                            ),
                        });
                    }
                    self.execution_trackers.remove(&plan.session.request_id);
                    self.active_plans
                        .retain(|_, active| active.session != plan.session);
                    self.request_phase_timings
                        .entry(plan.session.request_id)
                        .and_modify(|timing| *timing = RequestPhaseTiming::default());
                    return None;
                }

                let message = if release.confirmed {
                    "scheduler rejected an execution recompute retry"
                } else {
                    "executor could not confirm physical cache release for recompute retry"
                };
                return Some(CommittedExecutorOutput {
                    session: plan.session.clone(),
                    output: ExecutorOutput::error(
                        plan.session.request_id.clone(),
                        message.to_string(),
                    ),
                    disposition: ExecutionDisposition::Failed(ExecutionFailure::invalid_output(
                        message,
                    )),
                });
            }
            ExecutionDisposition::Progress
            | ExecutionDisposition::Yielded(_)
            | ExecutionDisposition::Finished(_) => {
                self.scheduler.update_after_step(
                    &plan.session.request_id,
                    result.output.tokens_processed,
                    result.output.tokens_generated,
                    Vec::new(),
                    step_time_ms,
                );
            }
            ExecutionDisposition::Failed(_) => {}
        }
        result.output.request_id = plan.session.request_id.clone();
        Some(CommittedExecutorOutput {
            session: plan.session,
            output: result.output,
            disposition: result.disposition,
        })
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

        let mut groups: Vec<(
            BatchKey,
            NativeBatchMode,
            usize,
            Vec<super::scheduler::ScheduledRequest>,
        )> = Vec::new();

        for item in scheduled {
            if !request_by_id.contains_key(item.request_id.as_str()) {
                continue;
            }
            let Some((key, batch_mode, max_batch_size)) =
                self.active_plans.get(&item.plan_id).map(|plan| {
                    (
                        plan.batch_key.clone(),
                        plan.batch_mode,
                        plan.max_batch_size.max(1),
                    )
                })
            else {
                warn!(
                    plan_id = item.plan_id,
                    request_id = %item.request_id,
                    "Skipping scheduled work without an active execution plan"
                );
                continue;
            };
            if batch_mode == NativeBatchMode::None && max_batch_size == 1 {
                groups.push((key, batch_mode, 1, vec![item.clone()]));
            } else if let Some((_, _, _, bucket)) =
                groups
                    .iter_mut()
                    .find(|(group_key, group_mode, group_max, bucket)| {
                        *group_key == key
                            && *group_mode == batch_mode
                            && *group_max == max_batch_size
                            && bucket.len() < max_batch_size
                    })
            {
                bucket.push(item.clone());
            } else {
                groups.push((key, batch_mode, max_batch_size, vec![item.clone()]));
            }
        }

        let mut outputs = Vec::new();
        for (_, _, _, bucket) in groups {
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
        // internal one-token rounds makes completion fencing ambiguous. A future
        // continuous adapter must expose child quanta explicitly.
        let result = self.executor.execute_decode(request_refs, scheduled).await;
        if let Ok(outputs) = &result {
            if let Some(first) = outputs.first() {
                record_engine_batch_dispatch(first.dispatch);
            }
        }
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
            pending_terminal_outputs: VecDeque::new(),
            execution_retry_attempts: HashMap::new(),
            retry_policy: LifecycleRetryPolicy::default(),
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
    pub fn add_request(&mut self, mut request: EngineCoreRequest) -> Result<()> {
        let request_id = request.id.clone();

        if self.requests.contains_key(&request_id) {
            return Err(Error::InvalidInput(format!(
                "Request {} already exists",
                request_id
            )));
        }

        // Chat prompt tokens drive scheduler/KV accounting and must come from
        // exact model preparation, never from the public mutable request fields.
        request.seal_execution_preparation()?;
        request.enforce_chat_context_window(self.config.max_seq_len)?;

        // Add to scheduler. A public ID remains unavailable while an expired
        // incarnation has logical cache quarantined behind unconfirmed
        // executor cleanup.
        if !self.scheduler.add_request(&request) {
            return Err(Error::InvalidInput(format!(
                "Request {} already exists or is awaiting cache cleanup",
                request_id
            )));
        }

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

    fn terminal_release_cause(disposition: &ExecutionDisposition) -> Option<TerminalReleaseCause> {
        match disposition {
            ExecutionDisposition::Finished(ExecutionFinishReason::Completed) => {
                Some(TerminalReleaseCause::Completed)
            }
            ExecutionDisposition::Finished(ExecutionFinishReason::Cancelled) => {
                Some(TerminalReleaseCause::Cancelled)
            }
            ExecutionDisposition::Finished(ExecutionFinishReason::TimedOut) => {
                Some(TerminalReleaseCause::TimedOut)
            }
            ExecutionDisposition::Finished(ExecutionFinishReason::Rejected)
            | ExecutionDisposition::Failed(ExecutionFailure {
                retry: RetryDisposition::Never,
                ..
            }) => Some(TerminalReleaseCause::Failed),
            _ => None,
        }
    }

    fn clear_exact_execution_state(&mut self, session: &super::SessionKey) {
        if self
            .execution_trackers
            .get(&session.request_id)
            .is_some_and(|tracker| tracker.session() == session)
        {
            self.execution_trackers.remove(&session.request_id);
        }
        self.active_plans.retain(|_, plan| plan.session != *session);
        self.execution_retry_attempts.remove(session);
    }

    fn record_unconfirmed_cleanup(&mut self, session: &super::SessionKey) {
        let attempt = self
            .scheduler
            .pending_cleanup_attempts(session)
            .unwrap_or_default()
            .saturating_add(1);
        let retry_at = Instant::now() + self.retry_policy.cleanup_delay(attempt);
        self.scheduler.record_cleanup_retry(session, retry_at);
        if attempt == 1 || attempt.is_power_of_two() {
            warn!(
                request_id = %session.request_id,
                session_epoch = session.epoch,
                cleanup_attempt = attempt,
                retry_in_ms = self.retry_policy.cleanup_delay(attempt).as_millis(),
                "Executor could not confirm exact-session cleanup; cache remains quarantined"
            );
        }
    }

    async fn attempt_pending_release_cleanup(&mut self, session: &super::SessionKey) {
        let Some(confirmation_required) = self
            .scheduler
            .pending_release_confirmation_required(session)
        else {
            return;
        };
        let release = self.executor.cleanup_session(session).await;
        if release.confirmed || !confirmation_required {
            self.scheduler
                .confirm_session_release(session, self.kv_cache.inner_mut());
        } else {
            self.record_unconfirmed_cleanup(session);
        }
    }

    async fn begin_terminal_release(
        &mut self,
        session: &super::SessionKey,
        cause: TerminalReleaseCause,
    ) {
        if matches!(
            self.scheduler.begin_terminal_release(session, cause),
            BeginTerminalRelease::Started { .. }
        ) {
            self.attempt_pending_release_cleanup(session).await;
        }
        self.clear_exact_execution_state(session);
    }

    async fn reconcile_due_cleanup(&mut self) {
        let sessions = self
            .scheduler
            .due_cleanup_sessions(Instant::now(), self.retry_policy.cleanup_budget_per_step);
        for session in sessions {
            self.attempt_pending_release_cleanup(&session).await;
        }
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
        self.reconcile_due_cleanup().await;
        let schedule_result = self.scheduler.schedule(self.kv_cache.inner_mut());

        // Deadline expiry removes a request from runnable scheduler state but
        // deliberately retains its logical cache allocation. Reconcile the
        // exact executor session before any newly scheduled work can execute;
        // only a confirmed physical cleanup permits logical block reuse.
        for expired in &schedule_result.expired_requests {
            let session = expired.session_key();
            self.attempt_pending_release_cleanup(&session).await;
        }

        for session in &schedule_result.preempted_requests {
            let release = self.executor.cleanup_session(session).await;
            if release.confirmed {
                if !self
                    .scheduler
                    .confirm_preemption(session, self.kv_cache.inner_mut())
                {
                    let quarantined = self.scheduler.quarantine_rejected_confirmed_preemption(
                        session,
                        self.kv_cache.inner_mut(),
                    );
                    warn!(
                        request_id = %session.request_id,
                        session_epoch = session.epoch,
                        quarantined,
                        "Scheduler rejected confirmed preemption"
                    );
                    if quarantined {
                        self.requests.remove(&session.request_id);
                        let message = "scheduler could not commit an executor-confirmed preemption";
                        self.pending_terminal_outputs
                            .push_back(CommittedExecutorOutput {
                                session: session.clone(),
                                output: ExecutorOutput::error(session.request_id.clone(), message),
                                disposition: ExecutionDisposition::Failed(
                                    ExecutionFailure::invalid_output(message),
                                ),
                            });
                    }
                }
            } else {
                self.scheduler.quarantine_failed_preemption(session);
                self.record_unconfirmed_cleanup(session);
                self.requests.remove(&session.request_id);
                let message = "executor could not confirm physical cache release during preemption";
                self.pending_terminal_outputs
                    .push_back(CommittedExecutorOutput {
                        session: session.clone(),
                        output: ExecutorOutput::error(session.request_id.clone(), message),
                        disposition: ExecutionDisposition::Failed(
                            ExecutionFailure::invalid_output(message),
                        ),
                    });
            }
            self.clear_exact_execution_state(session);
            self.request_phase_timings
                .entry(session.request_id.clone())
                .and_modify(|timing| *timing = RequestPhaseTiming::default());
        }

        for request in &schedule_result.expired_requests {
            self.pending_terminal_outputs
                .push_back(CommittedExecutorOutput {
                    session: request.session_key(),
                    output: ExecutorOutput::terminal(request.request_id.clone()),
                    disposition: ExecutionDisposition::Finished(ExecutionFinishReason::TimedOut),
                });
        }

        if !schedule_result.has_execution_work() && self.pending_terminal_outputs.is_empty() {
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
            && self.pending_terminal_outputs.is_empty()
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
                if let Ok(executor_outputs) = &result {
                    if let Some(first) = executor_outputs.first() {
                        record_engine_batch_dispatch(first.dispatch);
                    }
                }
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
            Vec::with_capacity(executor_results.len() + self.pending_terminal_outputs.len());
        for result in executor_results {
            let step_time_ms = if decode_ids.contains(&result.session.request_id) {
                decode_step_ms
            } else {
                prefill_step_ms
            };
            if let Some(committed) = self.commit_executor_result(result, step_time_ms).await {
                executor_outputs.push(committed);
            }
        }
        // Terminal events are a durable outbox until all fallible work for the
        // step has completed. Draining earlier can lose an abort/deadline event
        // when maintenance or execution-plan preparation returns an error.
        executor_outputs.extend(self.pending_terminal_outputs.drain(..));

        // Phase 3: Process outputs
        let mut outputs = Vec::new();

        for committed in executor_outputs {
            let CommittedExecutorOutput {
                session,
                output: exec_output,
                disposition,
            } = committed;
            let request_id = exec_output.request_id.clone();

            // Get timing info
            let generation_time = self
                .request_start_times
                .get(&request_id)
                .map(|t| t.elapsed())
                .unwrap_or_default();
            let generation_time_ms = generation_time.as_secs_f64() * 1000.0;

            let sequence_id = session.epoch;

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
            let mut engine_output = self.output_processor.process_execution(
                exec_output.clone(),
                &disposition,
                sequence_id,
                generation_time,
            );
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

            // Update scheduler state only from the authoritative disposition.
            if let Some(cause) = Self::terminal_release_cause(&disposition) {
                self.begin_terminal_release(&session, cause).await;
                self.requests.remove(&request_id);
                self.request_start_times.remove(&request_id);
                self.request_phase_timings.remove(&request_id);
                self.clear_exact_execution_state(&session);
                debug!("Finished request {}", request_id);
            }

            outputs.push(engine_output);
        }

        Ok(outputs)
    }

    /// Confirm that a terminal output has been routed outside the core.
    ///
    /// The exact scheduler session remains fenced after [`Self::step`] returns
    /// its terminal output. Callers must acknowledge that same session only
    /// after placing the output in their delivery channel or return batch.
    pub fn acknowledge_terminal_output(&mut self, session: &super::SessionKey) -> bool {
        self.scheduler.mark_terminal_delivered(session)
    }

    /// Cancel a synchronous caller's exact session after its completion
    /// receiver is abandoned. There is no consumer for the cancellation event,
    /// so discard that exact terminal output and mark delivery complete while
    /// retaining the scheduler quarantine until executor cleanup is confirmed.
    pub(crate) async fn abandon_request_session(&mut self, session: &super::SessionKey) -> bool {
        let aborted = self.abort_request_session(session).await;
        let has_queued_terminal = self
            .pending_terminal_outputs
            .iter()
            .any(|pending| pending.session == *session);
        if !aborted && !has_queued_terminal {
            return false;
        }
        self.pending_terminal_outputs
            .retain(|pending| pending.session != *session);
        self.acknowledge_terminal_output(session);
        true
    }

    /// Retry cleanup for an abandoned exact session and return the bounded
    /// delay before another attempt. `None` means the quarantine is gone.
    pub(crate) async fn retry_abandoned_session_cleanup(
        &mut self,
        session: &super::SessionKey,
    ) -> Option<Duration> {
        self.attempt_pending_release_cleanup(session).await;
        self.scheduler
            .pending_cleanup_attempts(session)
            .map(|attempt| self.retry_policy.cleanup_delay(attempt.max(1)))
    }

    pub(crate) fn abandoned_session_cleanup_delay(
        &self,
        session: &super::SessionKey,
    ) -> Option<Duration> {
        self.scheduler
            .pending_cleanup_attempts(session)
            .map(|attempt| self.retry_policy.cleanup_delay(attempt.max(1)))
    }

    /// Check if there's pending work.
    pub fn has_pending_work(&self) -> bool {
        !self.pending_terminal_outputs.is_empty()
            || self.scheduler.has_pending_work()
            || self.scheduler.has_due_cleanup(Instant::now())
    }

    /// Check if a request exists.
    pub fn has_request(&self, request_id: &RequestId) -> bool {
        self.requests.contains_key(request_id)
    }

    pub(crate) fn has_pending_terminal_output(&self, request_id: &RequestId) -> bool {
        self.pending_terminal_outputs
            .iter()
            .any(|committed| committed.session.request_id == *request_id)
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

    #[cfg(test)]
    pub(crate) fn set_hard_deadline_for_test(
        &mut self,
        request_id: &RequestId,
        deadline: Instant,
    ) -> bool {
        self.scheduler
            .set_hard_deadline_for_test(request_id, deadline)
    }

    /// Abort only if the caller still owns the exact request incarnation.
    pub async fn abort_request_session(&mut self, session: &super::SessionKey) -> bool {
        if self.scheduler.get_sequence_id(&session.request_id) != Some(session.epoch) {
            return false;
        }

        self.begin_terminal_release(session, TerminalReleaseCause::Cancelled)
            .await;
        self.requests.remove(&session.request_id);
        self.clear_exact_execution_state(session);
        self.pending_terminal_outputs
            .push_back(CommittedExecutorOutput {
                session: session.clone(),
                output: ExecutorOutput::terminal(session.request_id.clone()),
                disposition: ExecutionDisposition::Finished(ExecutionFinishReason::Cancelled),
            });
        debug!(
            request_id = %session.request_id,
            session_epoch = session.epoch,
            "Queued exact-session cancellation"
        );
        true
    }

    /// Abort a request.
    pub async fn abort_request(&mut self, request_id: &RequestId) -> bool {
        let Some(session) = self.get_session_key(request_id) else {
            return false;
        };
        self.abort_request_session(&session).await
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

    /// Purge reusable executor cache state owned by one model variant.
    pub async fn purge_model_cache(&mut self, variant: ModelVariant) -> CacheReleaseReport {
        self.executor.purge_model_cache(variant).await
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

        // A successful executor shutdown is the final physical-release fence:
        // no backend session can still reference the quarantined logical cache.
        self.scheduler
            .force_release_all_after_executor_shutdown(self.kv_cache.inner_mut());
        self.requests.clear();
        self.request_start_times.clear();
        self.request_phase_timings.clear();
        self.execution_trackers.clear();
        self.active_plans.clear();
        self.pending_terminal_outputs.clear();
        self.execution_retry_attempts.clear();

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
    use super::super::types::{AudioOutput, Priority, TaskType};
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

    struct DeadlineCleanupExecutor {
        events: Arc<Mutex<Vec<String>>>,
        confirm_cleanup: bool,
    }

    impl DeadlineCleanupExecutor {
        fn new(events: Arc<Mutex<Vec<String>>>, confirm_cleanup: bool) -> Self {
            Self {
                events,
                confirm_cleanup,
            }
        }

        fn execute_phase(
            &self,
            phase: &str,
            scheduled: &[ScheduledRequest],
        ) -> Vec<ExecutorStepResult> {
            let mut outputs = Vec::with_capacity(scheduled.len());
            for entry in scheduled {
                self.events
                    .lock()
                    .unwrap()
                    .push(format!("execute-{phase}:{}", entry.request_id));
                outputs.push(ExecutorStepResult::new(
                    entry,
                    ExecutorOutput {
                        request_id: entry.request_id.clone(),
                        audio: None,
                        text: None,
                        input_transcription: None,
                        tokens_processed: entry.num_tokens,
                        tokens_generated: 0,
                        finished: false,
                        phase_timing_override: None,
                        asr_diagnostics: None,
                        error: None,
                    },
                ));
            }
            outputs
        }
    }

    impl ModelExecutor for DeadlineCleanupExecutor {
        fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
            let mut profile = ExecutionProfile::fail_closed(
                BackendKind::Cpu,
                request.model_variant,
                ExecutionMode::Sequence,
            );
            profile.prefill = PrefillMode::Full;
            profile.cache_mode = super::super::execution::CacheMode::ExternalPaged;
            profile.cache_release_safe = true;
            profile.prefix_reuse_safe = true;
            profile.resolved_from_loaded_model = true;
            Some(profile)
        }

        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(self.execute_phase("prefill", scheduled))
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            Ok(self.execute_phase("decode", scheduled))
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

        fn cleanup_session(
            &self,
            session: &super::super::SessionKey,
        ) -> super::super::executor::CacheReleaseReport {
            self.events
                .lock()
                .unwrap()
                .push(format!("cleanup:{}:{}", session.request_id, session.epoch));
            if self.confirm_cleanup {
                super::super::executor::CacheReleaseReport::confirmed(1)
            } else {
                super::super::executor::CacheReleaseReport::unconfirmed()
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

    struct CancelledExecutor;

    impl ModelExecutor for CancelledExecutor {
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
            Ok(scheduled
                .iter()
                .map(|entry| {
                    ExecutorStepResult::from_session(
                        entry,
                        super::super::executor::ModelSessionResult::cancelled(
                            ExecutorOutput::cancelled(entry.request_id.clone()),
                        ),
                    )
                })
                .collect())
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
    async fn cancelled_execution_surfaces_aborted_terminal_output() {
        let executor = UnifiedExecutor::new_for_test(Box::new(CancelledExecutor));
        let mut core =
            EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
        let mut request = EngineCoreRequest::tts("cancelled");
        request.id = "cancelled".to_string();
        request.prompt_tokens = vec![1];
        core.add_request(request).unwrap();

        let outputs = core.step().await.unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_finished);
        assert_eq!(
            outputs[0].finish_reason,
            Some(super::super::types::FinishReason::Aborted)
        );
        assert_eq!(outputs[0].error.as_deref(), Some("request cancelled"));
        assert!(!core.has_request(&"cancelled".to_string()));
    }

    #[tokio::test]
    async fn waiting_and_running_aborts_emit_one_cancelled_output_each() {
        let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
        let executor =
            UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls.clone())));
        let mut core =
            EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();

        let mut waiting = EngineCoreRequest::tts("waiting");
        waiting.id = "abort-waiting".to_string();
        waiting.prompt_tokens = vec![1];
        core.add_request(waiting).unwrap();
        let waiting_session = core.get_session_key(&"abort-waiting".to_string()).unwrap();
        assert!(core.abort_request_session(&waiting_session).await);
        let waiting_outputs = core.step().await.unwrap();
        assert_eq!(waiting_outputs.len(), 1);
        assert_eq!(waiting_outputs[0].sequence_id, waiting_session.epoch);
        assert_eq!(
            waiting_outputs[0].finish_reason,
            Some(super::super::types::FinishReason::Aborted)
        );

        let mut running = EngineCoreRequest::tts("running");
        running.id = "abort-running".to_string();
        running.prompt_tokens = vec![1];
        core.add_request(running).unwrap();
        let progress = core.step().await.unwrap();
        assert_eq!(progress.len(), 1);
        assert!(!progress[0].is_finished);
        let running_session = core.get_session_key(&"abort-running".to_string()).unwrap();
        assert!(core.abort_request_session(&running_session).await);
        let running_outputs = core.step().await.unwrap();
        assert_eq!(running_outputs.len(), 1);
        assert_eq!(running_outputs[0].sequence_id, running_session.epoch);
        assert_eq!(
            running_outputs[0].finish_reason,
            Some(super::super::types::FinishReason::Aborted)
        );
        assert!(core.step().await.unwrap().is_empty());

        let calls = cleanup_calls.lock().unwrap();
        assert_eq!(
            calls
                .iter()
                .filter(|id| id.as_str() == "abort-waiting")
                .count(),
            1
        );
        assert_eq!(
            calls
                .iter()
                .filter(|id| id.as_str() == "abort-running")
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn unconfirmed_cancel_cleanup_is_retried_and_keeps_the_id_fenced() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let executor = UnifiedExecutor::new_for_test(Box::new(DeadlineCleanupExecutor::new(
            events.clone(),
            false,
        )));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 1,
                block_size: 1,
                max_blocks: 1,
                enable_adaptive_batching: false,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        core.retry_policy.cleanup_backoff_base = Duration::ZERO;
        core.retry_policy.cleanup_backoff_max = Duration::ZERO;

        let mut request = EngineCoreRequest::tts("retry cleanup");
        request.id = "cleanup-fence".to_string();
        request.prompt_tokens = vec![1];
        core.add_request(request.clone()).unwrap();
        core.step().await.unwrap();
        let session = core.get_session_key(&request.id).unwrap();
        assert!(core.abort_request_session(&session).await);
        assert!(core.add_request(request.clone()).is_err());

        let outputs = core.step().await.unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(
            outputs[0].finish_reason,
            Some(super::super::types::FinishReason::Aborted)
        );
        assert!(core.step().await.unwrap().is_empty());
        let cleanup_attempts = events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| event.starts_with("cleanup:cleanup-fence:"))
            .count();
        assert!(cleanup_attempts >= 2, "cleanup was not retried");
        assert_eq!(core.kv_cache_stats().allocated_blocks, 1);
        assert!(core.add_request(request).is_err());
    }

    #[tokio::test]
    async fn aborted_terminal_output_survives_plan_failure_until_successful_delivery() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let executor =
            UnifiedExecutor::new_for_test(Box::new(DeadlineCleanupExecutor::new(events, true)));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 1,
                min_tokens_per_step: 1,
                block_size: 1,
                max_blocks: 4,
                enable_chunked_prefill: false,
                enable_adaptive_batching: false,
                backend: BackendKind::Cpu,
                ..Default::default()
            },
            executor,
        )
        .unwrap();

        let mut aborted = EngineCoreRequest::tts("durable terminal outbox");
        aborted.id = "durable-abort".to_string();
        aborted.prompt_tokens = vec![1];
        core.add_request(aborted.clone()).unwrap();
        core.step().await.unwrap();
        assert_eq!(core.kv_cache_stats().allocated_blocks, 1);

        let aborted_session = core.get_session_key(&aborted.id).unwrap();
        assert!(core.abort_request_session(&aborted_session).await);
        assert_eq!(core.kv_cache_stats().allocated_blocks, 0);
        assert!(core.has_pending_terminal_output(&aborted.id));
        assert!(core.add_request(aborted.clone()).is_err());

        let mut plan_fault = EngineCoreRequest::tts("force plan validation failure");
        plan_fault.id = "plan-fault".to_string();
        plan_fault.prompt_tokens = vec![2];
        core.add_request(plan_fault.clone()).unwrap();
        let plan_fault_session = core.get_session_key(&plan_fault.id).unwrap();
        core.execution_trackers.insert(
            plan_fault.id.clone(),
            ExecutionTracker::new(super::super::SessionKey::new(
                plan_fault.id.clone(),
                plan_fault_session.epoch + 1,
            )),
        );

        let error = core.step().await.expect_err("plan preparation must fail");
        assert!(error
            .to_string()
            .contains("already has a different active session"));
        assert!(core.has_pending_terminal_output(&aborted.id));
        assert_eq!(
            core.scheduler
                .pending_release_confirmation_required(&aborted_session),
            Some(true)
        );
        assert!(core.add_request(aborted.clone()).is_err());

        core.execution_trackers.remove(&plan_fault.id);
        assert!(core
            .scheduler
            .release_execution_quantum_for_retry(&plan_fault_session));
        let retry_outputs = core.step().await.unwrap();
        let aborted_outputs: Vec<_> = retry_outputs
            .iter()
            .filter(|output| output.request_id == aborted.id)
            .collect();
        assert_eq!(aborted_outputs.len(), 1);
        assert_eq!(aborted_outputs[0].sequence_id, aborted_session.epoch);
        assert_eq!(
            aborted_outputs[0].finish_reason,
            Some(super::super::types::FinishReason::Aborted)
        );
        assert!(!core.has_pending_terminal_output(&aborted.id));
        assert_eq!(
            core.scheduler
                .pending_release_confirmation_required(&aborted_session),
            Some(true),
            "the delivered batch remains fenced until its outer consumer routes it"
        );
        assert!(core.add_request(aborted.clone()).is_err());
        assert!(core.acknowledge_terminal_output(&aborted_session));
        assert_eq!(
            core.scheduler
                .pending_release_confirmation_required(&aborted_session),
            None
        );

        let later_outputs = core.step().await.unwrap();
        assert!(later_outputs
            .iter()
            .all(|output| output.request_id != aborted.id));
        core.add_request(aborted).unwrap();
        assert_ne!(
            core.get_session_key(&"durable-abort".to_string())
                .unwrap()
                .epoch,
            aborted_session.epoch
        );
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
        assert!(
            core.add_request(second.clone()).is_err(),
            "public ID must remain fenced until cancellation is delivered"
        );
        let outputs = core.step().await.unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].sequence_id, first_session.epoch);
        assert_eq!(
            outputs[0].finish_reason,
            Some(super::super::types::FinishReason::Aborted)
        );
        assert_eq!(outputs[0].error.as_deref(), Some("request cancelled"));

        assert!(
            core.add_request(second.clone()).is_err(),
            "returning a terminal output must not acknowledge delivery inside the core"
        );
        assert!(core.acknowledge_terminal_output(&first_session));
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
    fn compatible_subbatches_respect_declared_mode_and_width() {
        fn install_plan(
            core: &mut EngineCore,
            scheduled: &ScheduledRequest,
            mode: NativeBatchMode,
            max_batch_size: usize,
        ) {
            core.active_plans.insert(
                scheduled.plan_id,
                ExecutionPlan {
                    plan_id: scheduled.plan_id,
                    session: scheduled.session_key(),
                    work: scheduled.work.clone(),
                    batch_key: BatchKey {
                        backend: BackendKind::Cpu,
                        model_variant: None,
                        task_type: TaskType::TTS,
                        work_kind: "tts".to_string(),
                        compute_dtype: "f32".to_string(),
                        kv_dtype: "none".to_string(),
                        cache_namespace: "none".to_string(),
                        adapter_id: None,
                    },
                    batch_mode: mode,
                    max_batch_size,
                    estimate: ResourceVector::zero(),
                },
            );
        }

        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
            Mutex::new(Vec::new()),
        ))));
        let mut core = EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor)
            .expect("core");
        let mut requests = Vec::new();
        let mut scheduled = Vec::new();
        for index in 0..3 {
            let id = format!("batch-{index}");
            let mut request = EngineCoreRequest::tts("hello");
            request.id = id.clone();
            requests.push(request);
            scheduled.push(scheduled_prefill(&id, index));
        }
        let refs = requests.iter().collect::<Vec<_>>();

        for item in &scheduled {
            install_plan(&mut core, item, NativeBatchMode::None, 1);
        }
        let serial = core.build_compatible_subbatches(&refs, &scheduled);
        assert_eq!(serial.len(), 3);
        assert!(serial.iter().all(|(_, batch)| batch.len() == 1));

        core.active_plans.clear();
        for item in &scheduled {
            install_plan(&mut core, item, NativeBatchMode::None, 2);
        }
        let request_parallel = core.build_compatible_subbatches(&refs, &scheduled);
        assert_eq!(
            request_parallel
                .iter()
                .map(|(_, batch)| batch.len())
                .collect::<Vec<_>>(),
            vec![2, 1]
        );

        core.active_plans.clear();
        for item in &scheduled {
            install_plan(&mut core, item, NativeBatchMode::Static, 2);
        }
        let static_batches = core.build_compatible_subbatches(&refs, &scheduled);
        assert_eq!(
            static_batches
                .iter()
                .map(|(_, batch)| batch.len())
                .collect::<Vec<_>>(),
            vec![2, 1]
        );
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
        let output = core
            .commit_executor_result(invalid, 1.0)
            .await
            .expect("invalid result must emit a terminal output")
            .output;

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

    fn retryable_step_result(
        scheduled: &ScheduledRequest,
        retry: RetryDisposition,
    ) -> ExecutorStepResult {
        ExecutorStepResult::from_session(
            scheduled,
            super::super::executor::ModelSessionResult {
                output: ExecutorOutput {
                    request_id: scheduled.request_id.clone(),
                    audio: None,
                    text: None,
                    input_transcription: None,
                    tokens_processed: 0,
                    tokens_generated: 0,
                    finished: false,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: Some("transient backend failure".to_string()),
                },
                disposition: ExecutionDisposition::Failed(ExecutionFailure {
                    kind: super::super::execution::FailureKind::Backend,
                    scope: super::super::execution::FailureScope::Row,
                    retry,
                    health: super::super::execution::HealthImpact::Degraded,
                    message: "transient backend failure".to_string(),
                }),
                safe_point: true,
            },
        )
    }

    #[tokio::test]
    async fn same_session_retry_releases_quantum_without_committing_progress() {
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
            Mutex::new(Vec::new()),
        ))));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 8,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        core.retry_policy.execution_backoff_base = Duration::ZERO;
        core.retry_policy.execution_backoff_max = Duration::ZERO;
        let mut request = EngineCoreRequest::tts("retry");
        request.id = "same-session-retry".to_string();
        request.prompt_tokens = vec![1];
        core.add_request(request).unwrap();
        core.refresh_scheduler_execution_profiles().await;
        let scheduled = core
            .scheduler
            .schedule(core.kv_cache.inner_mut())
            .prefill_requests
            .remove(0);
        let session = scheduled.session_key();
        core.begin_execution_plan(&scheduled).await.unwrap();

        let emitted = core
            .commit_executor_result(
                retryable_step_result(&scheduled, RetryDisposition::RetrySameSession),
                1.0,
            )
            .await;
        assert!(
            emitted.is_none(),
            "a retry must not terminalize the request"
        );
        assert_eq!(
            core.scheduler.get_running_info(&scheduled.request_id),
            Some((0, 0))
        );

        let mut retry_schedule = core.scheduler.schedule(core.kv_cache.inner_mut());
        assert_eq!(
            retry_schedule.prefill_requests.len(),
            1,
            "same-session retry was not rescheduled: {retry_schedule:?}"
        );
        let retry = retry_schedule.prefill_requests.remove(0);
        assert_eq!(retry.session_key(), session);
        assert_ne!(retry.plan_id, scheduled.plan_id);
    }

    #[tokio::test]
    async fn retry_budget_exhaustion_terminalizes_the_exact_session() {
        let executor = UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(Arc::new(
            Mutex::new(Vec::new()),
        ))));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 8,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        core.retry_policy.max_execution_retries = 2;
        core.retry_policy.execution_backoff_base = Duration::ZERO;
        core.retry_policy.execution_backoff_max = Duration::ZERO;
        let mut request = EngineCoreRequest::tts("retry budget");
        request.id = "retry-budget".to_string();
        request.prompt_tokens = vec![1];
        core.add_request(request).unwrap();
        core.refresh_scheduler_execution_profiles().await;

        let mut scheduled = core
            .scheduler
            .schedule(core.kv_cache.inner_mut())
            .prefill_requests
            .remove(0);
        let session = scheduled.session_key();
        for attempt in 1..=3 {
            core.begin_execution_plan(&scheduled).await.unwrap();
            let committed = core
                .commit_executor_result(
                    retryable_step_result(&scheduled, RetryDisposition::RetrySameSession),
                    1.0,
                )
                .await;
            if attempt <= 2 {
                assert!(committed.is_none());
                scheduled = core
                    .scheduler
                    .schedule(core.kv_cache.inner_mut())
                    .prefill_requests
                    .remove(0);
                assert_eq!(scheduled.session_key(), session);
            } else {
                let committed = committed.expect("retry budget must terminalize");
                assert_eq!(committed.session, session);
                assert!(committed
                    .output
                    .error
                    .as_deref()
                    .is_some_and(|message| message.contains("retry budget exhausted")));
                assert!(matches!(
                    committed.disposition,
                    ExecutionDisposition::Failed(ExecutionFailure {
                        retry: RetryDisposition::Never,
                        ..
                    })
                ));
            }
        }
    }

    #[tokio::test]
    async fn recompute_retry_releases_physical_and_logical_session_state() {
        let cleanup_calls = Arc::new(Mutex::new(Vec::new()));
        let executor =
            UnifiedExecutor::new_for_test(Box::new(MockExecutor::new(cleanup_calls.clone())));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 8,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        core.retry_policy.execution_backoff_base = Duration::ZERO;
        core.retry_policy.execution_backoff_max = Duration::ZERO;
        let mut request = EngineCoreRequest::tts("recompute");
        request.id = "recompute-retry".to_string();
        request.prompt_tokens = vec![1];
        core.add_request(request).unwrap();
        core.refresh_scheduler_execution_profiles().await;

        let prefill = core
            .scheduler
            .schedule(core.kv_cache.inner_mut())
            .prefill_requests
            .remove(0);
        let session = prefill.session_key();
        core.begin_execution_plan(&prefill).await.unwrap();
        core.commit_executor_result(
            ExecutorStepResult::new(
                &prefill,
                MockExecutor::build_outputs(std::slice::from_ref(&prefill)).remove(0),
            ),
            1.0,
        )
        .await
        .expect("prefill progress must be emitted");

        let decode = core
            .scheduler
            .schedule(core.kv_cache.inner_mut())
            .decode_requests
            .remove(0);
        core.begin_execution_plan(&decode).await.unwrap();
        let emitted = core
            .commit_executor_result(
                retryable_step_result(&decode, RetryDisposition::Recompute),
                1.0,
            )
            .await;
        assert!(
            emitted.is_none(),
            "recompute must not terminalize the request"
        );
        assert!(cleanup_calls
            .lock()
            .unwrap()
            .iter()
            .any(|request_id| request_id == "recompute-retry"));
        assert_eq!(
            core.scheduler.get_running_info(&decode.request_id),
            Some((0, 0))
        );
        assert!(!core.execution_trackers.contains_key(&decode.request_id));

        let retry = core
            .scheduler
            .schedule(core.kv_cache.inner_mut())
            .prefill_requests
            .remove(0);
        assert_eq!(retry.session_key(), session);
        assert_ne!(retry.plan_id, decode.plan_id);
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
        core.refresh_scheduler_execution_profiles().await;
        let scheduled = core
            .scheduler
            .schedule(core.kv_cache.inner_mut())
            .prefill_requests
            .remove(0);
        assert_eq!(scheduled.num_computed_tokens, 0);
        assert_eq!(scheduled.num_tokens, 16);
        assert_eq!(
            scheduled.block_ids.len(),
            core.kv_cache.inner().blocks_for_tokens(16)
        );
        let mut partial = scheduled.clone();
        partial.num_tokens = 4;
        assert!(core.begin_execution_plan(&partial).await.is_err());
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
        let output = core
            .commit_executor_result(result, 1.0)
            .await
            .expect("valid progress must emit an output")
            .output;
        assert!(output.error.is_none());
        assert_eq!(
            core.scheduler.get_running_info(&scheduled.request_id),
            Some((16, 1))
        );
    }

    #[tokio::test]
    async fn cache_plan_estimate_and_executor_observation_remain_distinct() {
        let executor = UnifiedExecutor::new_for_test(Box::new(DeadlineCleanupExecutor::new(
            Arc::new(Mutex::new(Vec::new())),
            true,
        )));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                block_size: 1,
                max_blocks: 8,
                ..Default::default()
            },
            executor,
        )
        .unwrap();
        let mut request = EngineCoreRequest::tts("cache contract");
        request.id = "cache-contract".to_string();
        request.prompt_tokens = vec![1, 2];
        core.add_request(request).unwrap();
        core.refresh_scheduler_execution_profiles().await;
        let scheduled = core
            .scheduler
            .schedule(core.kv_cache.inner_mut())
            .prefill_requests
            .remove(0);
        core.begin_execution_plan(&scheduled).await.unwrap();
        assert!(matches!(
            core.active_plans[&scheduled.plan_id].estimate.kv_bytes,
            ResourceAmount::Known(bytes) if bytes > 0
        ));

        let observed = ResourceVector {
            kv_bytes: ResourceAmount::Known(123),
            ..ResourceVector::zero()
        };
        let result = ExecutorStepResult::new(
            &scheduled,
            ExecutorOutput {
                request_id: scheduled.request_id.clone(),
                audio: None,
                text: None,
                input_transcription: None,
                tokens_processed: scheduled.num_tokens,
                tokens_generated: 0,
                finished: false,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
            },
        )
        .with_observed_resources(observed);
        let report = EngineCore::report_from_result(&result);
        assert_eq!(report.observed_resources, observed);
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

        let output = core
            .commit_executor_result(result.clone(), 1.0)
            .await
            .expect("valid progress must emit an output")
            .output;
        assert!(output.error.is_none());
        assert_eq!(
            core.scheduler.get_running_info(&scheduled.request_id),
            Some((scheduled.num_tokens, 0))
        );

        assert!(core.commit_executor_result(result, 1.0).await.is_none());
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
        let output = core
            .commit_executor_result(result, 1.0)
            .await
            .expect("invalid atomic result must emit an error")
            .output;
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
    async fn deadline_cleanup_precedes_newly_scheduled_execution() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let executor = UnifiedExecutor::new_for_test(Box::new(DeadlineCleanupExecutor::new(
            events.clone(),
            true,
        )));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 1,
                min_tokens_per_step: 1,
                block_size: 1,
                max_blocks: 2,
                enable_chunked_prefill: false,
                enable_adaptive_batching: false,
                backend: BackendKind::Cpu,
                ..Default::default()
            },
            executor,
        )
        .unwrap();

        let mut expired = EngineCoreRequest::tts("expires after prefill");
        expired.id = "expired-running".to_string();
        expired.prompt_tokens = vec![1];
        core.add_request(expired).unwrap();
        core.step().await.unwrap();
        assert!(core.scheduler.set_hard_deadline_for_test(
            &"expired-running".to_string(),
            Instant::now() - std::time::Duration::from_millis(1),
        ));

        let mut next = EngineCoreRequest::tts("new work");
        next.id = "new-work".to_string();
        next.prompt_tokens = vec![2];
        core.add_request(next).unwrap();
        core.step().await.unwrap();

        let events = events.lock().unwrap();
        let cleanup_index = events
            .iter()
            .position(|event| event.starts_with("cleanup:expired-running:"))
            .expect("expired session cleanup was not requested");
        let execution_index = events
            .iter()
            .position(|event| event == "execute-prefill:new-work")
            .expect("new work was not executed");
        assert!(
            cleanup_index < execution_index,
            "new work executed before expired cache cleanup: {events:?}"
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| event.starts_with("cleanup:expired-running:"))
                .count(),
            1,
            "expired cleanup must not be repeated during terminal processing"
        );
    }

    #[tokio::test]
    async fn unconfirmed_deadline_cleanup_quarantines_scarce_logical_block() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let executor = UnifiedExecutor::new_for_test(Box::new(DeadlineCleanupExecutor::new(
            events.clone(),
            false,
        )));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_batch_size: 1,
                max_tokens_per_step: 1,
                min_tokens_per_step: 1,
                block_size: 1,
                max_blocks: 1,
                enable_chunked_prefill: false,
                enable_prefix_caching: true,
                enable_adaptive_batching: false,
                backend: BackendKind::Cpu,
                ..Default::default()
            },
            executor,
        )
        .unwrap();

        let mut expired = EngineCoreRequest::tts("owns the only block");
        expired.id = "expired-running".to_string();
        expired.prompt_tokens = vec![1];
        core.add_request(expired).unwrap();
        core.step().await.unwrap();
        assert_eq!(core.kv_cache_stats().allocated_blocks, 1);
        assert!(core.scheduler.set_hard_deadline_for_test(
            &"expired-running".to_string(),
            Instant::now() - std::time::Duration::from_millis(1),
        ));

        let mut waiting = EngineCoreRequest::tts("waits for capacity");
        waiting.id = "waiting".to_string();
        waiting.prompt_tokens = vec![1];
        core.add_request(waiting).unwrap();
        let outputs = core.step().await.unwrap();

        assert!(outputs.iter().any(|output| {
            output.request_id == "expired-running"
                && output.error.as_deref() == Some(REQUEST_DEADLINE_EXCEEDED)
        }));
        assert_eq!(core.kv_cache_stats().allocated_blocks, 1);
        assert_eq!(core.pending_request_count(), 1);
        assert!(core.step().await.unwrap().is_empty());
        assert!(events
            .lock()
            .unwrap()
            .iter()
            .all(|event| event != "execute-prefill:waiting"));

        let mut reused = EngineCoreRequest::tts("must remain fenced");
        reused.id = "expired-running".to_string();
        reused.prompt_tokens = vec![3];
        assert!(core.add_request(reused).is_err());
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
        request.model_variant = Some(ModelVariant::Qwen306B);
        request
            .install_chat_execution_preparation(ModelVariant::Qwen306B, vec![11, 22, 33, 44], None)
            .unwrap();

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

    #[test]
    fn core_rejects_unprepared_and_mismatched_chat_accounting() {
        let executor = UnifiedExecutor::new_for_test(Box::new(ImmediateFinishExecutor));
        let mut core =
            EngineCore::new_with_unified_executor(EngineCoreConfig::default(), executor).unwrap();
        let mut request = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);
        request.prompt_tokens = vec![11, 22, 33, 44];

        let error = core
            .add_request(request)
            .expect_err("public prompt tokens must not authorize scheduler accounting");
        assert!(error
            .to_string()
            .contains("missing exact model prompt preparation"));

        let mut mismatched = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "Hello".to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);
        mismatched
            .install_chat_execution_preparation(ModelVariant::Qwen306B, vec![11, 22], None)
            .unwrap();
        mismatched.prompt_tokens[0] = 99;
        let error = core
            .add_request(mismatched)
            .expect_err("mutated exact preparation must not reach the scheduler");
        assert!(error
            .to_string()
            .contains("changed after exact prompt preparation"));
    }

    #[test]
    fn core_enforces_exact_chat_context_after_preparation() {
        let executor = UnifiedExecutor::new_for_test(Box::new(ImmediateFinishExecutor));
        let mut core = EngineCore::new_with_unified_executor(
            EngineCoreConfig {
                max_seq_len: 4,
                ..EngineCoreConfig::default()
            },
            executor,
        )
        .unwrap();

        let mut full = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "full".to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);
        full.install_chat_execution_preparation(ModelVariant::Qwen306B, vec![1, 2, 3, 4], None)
            .unwrap();
        assert!(core
            .add_request(full)
            .expect_err("a full context must leave no output allocation")
            .to_string()
            .contains("leaves no output capacity"));

        let mut bounded = EngineCoreRequest::chat(vec![ChatMessage {
            role: ChatRole::User,
            content: "bounded".to_string(),
        }])
        .with_model_variant(ModelVariant::Qwen306B);
        bounded.id = "bounded-context".to_string();
        bounded.params.max_tokens = 100;
        bounded
            .install_chat_execution_preparation(ModelVariant::Qwen306B, vec![1, 2], None)
            .unwrap();
        core.add_request(bounded).unwrap();
        assert_eq!(core.requests["bounded-context"].params.max_tokens, 2);
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
