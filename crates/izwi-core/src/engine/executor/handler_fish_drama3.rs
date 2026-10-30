//! Fish Audio Drama 3 execution (§7.1, §7.4): S2's route shape on the
//! family's own model loop.
//!
//! * **Prefill** runs exactly one cell of the family's grid per quantum,
//!   whatever span the scheduler offers, and reports that cell as its
//!   progress: the trunk over the cell's rows, then the tower teacher-forced
//!   over them. A request's cells are never batched with another's, so
//!   other users' cells and frames interleave between them.
//! * **Decode** makes one frame per row per quantum across users, every row
//!   alone through each kernel. The draw after the budget's last frame must
//!   be eos: a `FrameLimit` stop fails the request, never a cut-off take.
//! * **Audio decode** runs committed frames through the codec in batched
//!   quanta (the first after 4 frames, then 16 at a time), reserving each
//!   stream's output credit first and yielding under back-pressure.
//! * **Finalize** flushes the stream; its PCM is frames × 2,048 samples.
//!
//! Every model quantum runs inside the family's quantum checkpoint, and the
//! trunk's DeltaNet tensors move through the engine's transactional tensor
//! arena: restored before the quantum, staged into the plan's transaction
//! after it. The checkpoint and every codec candidate wait in the
//! family's pending coordinator for the core's commit or abort, so an
//! aborted quantum restores the request bit for bit. Cancellation and the
//! deadline are checked before and after every model call: between prefill
//! cells, decode quanta and codec quanta.

use std::sync::Arc;
use std::time::Instant;

use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::models::architectures::fish_drama3::model::{
    FISH_DRAMA3_AUDIO_CHUNK_FRAMES, FISH_DRAMA3_FIRST_AUDIO_FRAMES,
};
use crate::models::architectures::fish_drama3::physical::{TOWER_KV_DOMAIN, TRUNK_KV_DOMAIN};
use crate::models::architectures::fish_drama3::{
    FishDrama3AudioStream, FishDrama3QuantumCheckpoint, FishDrama3Step, FishDrama3StopReason,
};

use super::super::request::EngineCoreRequest;
use super::super::scheduler::ScheduledRequest;
use super::super::types::AudioOutput;
use super::super::SessionKey;
use super::handler_tts::retained_tts_batch_model_call;
use super::state::ActiveFishDrama3TtsDecode;
use super::{ExecutorOutput, ExecutorStateLease, ModelSessionResult, NativeExecutor};

/// If a later row fails, every candidate already handed to the coordinator
/// is discarded before the physical invocation reports its failure.
struct FishDrama3PendingBatchGuard<'a> {
    coordinator: &'a super::FishDrama3StateCoordinator,
    staged: Vec<(crate::engine::PlanId, SessionKey)>,
    armed: bool,
}

impl Drop for FishDrama3PendingBatchGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            for (plan, session) in &self.staged {
                super::PendingQuantumFinalizer::discard(self.coordinator, *plan, session);
            }
        }
    }
}

/// One row of a model quantum: its checked-out state and the quantum's
/// rollback point until the coordinator holds it.
struct FishDrama3Row<'a> {
    index: usize,
    lease: Option<ExecutorStateLease<'a, ActiveFishDrama3TtsDecode>>,
    checkpoint: Option<FishDrama3QuantumCheckpoint>,
    fresh: bool,
}

impl FishDrama3Row<'_> {
    fn active_mut(&mut self) -> Result<&mut ActiveFishDrama3TtsDecode> {
        self.lease
            .as_mut()
            .ok_or_else(|| Error::InferenceError("Fish Audio Drama 3 row lost its lease".into()))?
            .require_state_mut()
    }

    /// Restores the row to its state before the quantum; a first quantum
    /// has none, so its state is dropped.
    fn rollback(&mut self) -> Result<()> {
        let Some(checkpoint) = self.checkpoint.take() else {
            return Ok(());
        };
        let Some(lease) = self.lease.as_mut() else {
            return Ok(());
        };
        if self.fresh {
            lease.discard_state();
            self.fresh = false;
            return Ok(());
        }
        let active = lease.require_state_mut()?;
        let model = active.model.clone();
        model.abort_quantum(&mut active.state, checkpoint)?;
        lease.mark_clean();
        Ok(())
    }
}

impl Drop for FishDrama3Row<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.rollback() {
            tracing::error!(%error, "Fish Audio Drama 3 rollback failed; state remains fenced")
        }
    }
}

fn row_terminal(request: &EngineCoreRequest, dispatched: bool) -> Option<ModelSessionResult> {
    if request.is_cancelled() {
        let output = ExecutorOutput::cancelled(request.id.clone());
        return Some(if dispatched {
            ModelSessionResult::cancelled(output)
        } else {
            ModelSessionResult::cancelled_before_dispatch(output)
        });
    }
    if request
        .deadline
        .is_none_or(|deadline| Instant::now() < deadline)
    {
        return None;
    }
    let mut result = ModelSessionResult::sequence(ExecutorOutput::error(
        request.id.clone(),
        Error::Timeout(request.id.clone()).to_string(),
    ));
    result.disposition =
        crate::engine::ExecutionDisposition::Finished(crate::engine::FinishReason::TimedOut);
    result.provenance = crate::engine::OutcomeProvenance::deadline(
        if dispatched {
            crate::engine::DeadlinePhase::ModelExecution
        } else {
            crate::engine::DeadlinePhase::DispatchWait
        },
        if dispatched {
            crate::engine::DispatchState::Started
        } else {
            crate::engine::DispatchState::NotStarted
        },
    );
    Some(result)
}

fn check_request(request: &EngineCoreRequest) -> Result<()> {
    if request.is_cancelled() {
        return Err(Error::Cancelled(request.id.clone()));
    }
    if request
        .deadline
        .is_some_and(|deadline| Instant::now() >= deadline)
    {
        return Err(Error::Timeout(request.id.clone()));
    }
    Ok(())
}

fn timing(
    active: &ActiveFishDrama3TtsDecode,
    codec_ms: f64,
    first_audio_ms: Option<f64>,
) -> super::ExecutorPhaseTiming {
    super::ExecutorPhaseTiming {
        prefill_ms: Some(active.prefill_ms),
        decode_ms: Some(active.decode_ms),
        codec_ms: Some(codec_ms),
        prefill_steps: Some(u32::try_from(active.prefill_steps).unwrap_or(u32::MAX)),
        decode_steps: Some(u32::try_from(active.decode_steps).unwrap_or(u32::MAX)),
        first_output_ms_since_start: first_audio_ms,
        ..Default::default()
    }
}

fn empty_output(request: &EngineCoreRequest, sample_rate: u32) -> ExecutorOutput {
    ExecutorOutput {
        request_id: request.id.clone(),
        audio: Some(AudioOutput::empty(sample_rate)),
        text: None,
        input_transcription: None,
        tokens_processed: 0,
        tokens_generated: 0,
        finished: false,
        phase_timing_override: None,
        asr_diagnostics: None,
        error: None,
        logprobs: Vec::new(),
    }
}

/// Committed frames the codec has not decoded yet, and the codec quantum
/// they call for: the first audio after 4 frames, then 16 at a time, and
/// whatever remains once the request stops.
fn audio_due(frames: usize, stream: &FishDrama3AudioStream, finished: bool) -> Option<usize> {
    let pending = frames.saturating_sub(stream.decoded_frames());
    let threshold = if stream.decoded_frames() == 0 {
        FISH_DRAMA3_FIRST_AUDIO_FRAMES
    } else {
        FISH_DRAMA3_AUDIO_CHUNK_FRAMES
    };
    (pending >= threshold || (finished && pending > 0))
        .then(|| pending.min(FISH_DRAMA3_AUDIO_CHUNK_FRAMES))
}

impl NativeExecutor {
    pub(super) fn fish_drama3_tts_request_with_managed_cache(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        retained: Option<super::RetainedRowManagedState>,
    ) -> Result<ModelSessionResult> {
        self.fish_drama3_tts_batch_with_managed(
            &[request],
            std::slice::from_ref(scheduled),
            vec![retained],
        )?
        .pop()
        .ok_or_else(|| {
            Error::InferenceError("Fish Audio Drama 3 scalar quantum produced no result".into())
        })
    }

    /// One model quantum: a prefill cell of one request, or a frame for
    /// every decode row.
    pub(super) fn fish_drama3_tts_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        mut managed: Vec<Option<super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        if requests.is_empty()
            || requests.len() != scheduled.len()
            || managed.len() != scheduled.len()
        {
            return Err(Error::InvalidInput(
                "Fish Audio Drama 3 batch row counts differ".into(),
            ));
        }
        let is_prefill = scheduled[0].is_prefill;
        if is_prefill && requests.len() != 1 {
            return Err(Error::InvalidInput(
                "Fish Audio Drama 3 prefill cells are never batched across requests".into(),
            ));
        }
        let model = requests[0]
            .prepared_fish_drama3_tts_model_lease_for_executor()?
            .ok_or_else(|| Error::InferenceError("Fish Audio Drama 3 batch lost model".into()))?;
        let variant = Self::resolve_variant(requests[0])?;
        if variant.family() != ModelFamily::FishDrama3Tts {
            return Err(Error::InvalidInput(
                "foreign Fish Audio Drama 3 request".into(),
            ));
        }
        let sample_rate = model.codec().sample_rate();
        let mut outputs: Vec<Option<ModelSessionResult>> =
            (0..requests.len()).map(|_| None).collect();
        let mut rows = Vec::with_capacity(requests.len());
        for (index, (request, scheduled_row)) in requests.iter().zip(scheduled).enumerate() {
            if scheduled_row.request_id != request.id
                || scheduled_row.is_prefill != is_prefill
                || Self::resolve_variant(request)? != variant
                || request
                    .execution_adapter_binding()
                    .map(|binding| binding.adapter_instance_id)
                    != requests[0]
                        .execution_adapter_binding()
                        .map(|binding| binding.adapter_instance_id)
            {
                return Err(Error::InvalidInput(
                    "Fish Audio Drama 3 batch crossed request, phase or adapter identity".into(),
                ));
            }
            let row_model = request
                .prepared_fish_drama3_tts_model_lease_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError("Fish Audio Drama 3 batch row lost model".into())
                })?;
            if !Arc::ptr_eq(&row_model.model_arc(), &model.model_arc()) {
                return Err(Error::InvalidInput(
                    "Fish Audio Drama 3 batch crossed model generation".into(),
                ));
            }
            if let Some(terminal) = row_terminal(request, false) {
                outputs[index] = Some(terminal);
                continue;
            }
            let mut retained = managed[index].take().ok_or_else(|| {
                Error::InferenceError("Fish Audio Drama 3 batch lost managed state".into())
            })?;
            let trunk = retained
                .take_paged_domain(TRUNK_KV_DOMAIN, true)?
                .expect("required Drama 3 trunk cache");
            // A final one-row prompt cell appends nothing to the tower, so
            // the quantum holds no tower view.
            let tower = retained.take_paged_domain(TOWER_KV_DOMAIN, false)?;
            retained.ensure_all_paged_consumed()?;
            let tensor = retained.tensor_state.clone().ok_or_else(|| {
                Error::InferenceError(
                    "Fish Audio Drama 3 row lost its tensor-state reservation".into(),
                )
            })?;
            let runtime = request.managed_cache_runtime().ok_or_else(|| {
                Error::InferenceError("Fish Audio Drama 3 row lost its managed runtime".into())
            })?;
            let arena = runtime.tensor_state().ok_or_else(|| {
                Error::InferenceError("Fish Audio Drama 3 row lost its tensor arena".into())
            })?;
            let mut lease = ExecutorStateLease::checkout(
                &self.fish_drama3_tts_decode_states,
                scheduled_row.session_key(),
                variant,
                "Fish Audio Drama 3 batch",
            )?;
            if lease.state().is_some_and(|active| {
                active.variant != variant
                    || !Arc::ptr_eq(&active.model.model_arc(), &model.model_arc())
            }) {
                return Err(Error::InvalidInput(
                    "Fish Audio Drama 3 retained state crossed model".into(),
                ));
            }
            let fresh = lease.state().is_none();
            if fresh {
                if !is_prefill || scheduled_row.num_computed_tokens != 0 {
                    return Err(Error::InferenceError(
                        "Fish Audio Drama 3 lost its state before its first prefill cell".into(),
                    ));
                }
                let artifact = request
                    .prepared_fish_drama3_tts_artifact_for_executor()?
                    .ok_or_else(|| {
                        Error::InferenceError("Fish Audio Drama 3 lost its prompt".into())
                    })?;
                let params = request
                    .fish_drama3_tts_generation_params_for_executor()?
                    .ok_or_else(|| {
                        Error::InferenceError(
                            "Fish Audio Drama 3 lost its generation params".into(),
                        )
                    })?;
                let tower = tower.ok_or_else(|| {
                    Error::InferenceError(
                        "Fish Audio Drama 3 first prefill cell lost its tower view".into(),
                    )
                })?;
                let positions = usize::try_from(runtime.maximum_sequence_tokens())
                    .map_err(|_| Error::InvalidInput("Drama 3 context exceeds usize".into()))?;
                let state =
                    model.new_prepared_request(&artifact, params, trunk, tower, positions)?;
                lease.install_state(ActiveFishDrama3TtsDecode {
                    variant,
                    model: model.clone(),
                    state,
                    stream: FishDrama3AudioStream::new(),
                    last_frames: 0,
                    stream_sequence: 0,
                    audio_samples: Vec::new(),
                    total_audio_samples: 0,
                    collect_audio_samples: request.collect_audio_samples,
                    prefill_ms: 0.0,
                    decode_ms: 0.0,
                    prefill_steps: 0,
                    decode_steps: 0,
                    codec_ms: 0.0,
                    execution_started: Instant::now(),
                    first_audio_ms: None,
                })?;
            } else {
                let active = lease.require_state_mut()?;
                model.rebind_caches(&mut active.state, trunk, tower)?;
            }
            let active = lease.require_state_mut()?;
            let trunk_state = active.state.trunk_state_mut();
            trunk_state.bind_tensor_sequence(tensor.sequence)?;
            trunk_state.restore_tensor_state(arena)?;
            let cursor = if is_prefill {
                active.state.prefilled()
            } else {
                active.state.trunk_kv_len()
            };
            if cursor != scheduled_row.num_computed_tokens
                || is_prefill == active.state.is_prefilled()
                || active.state.is_finished()
            {
                return Err(Error::InferenceError(format!(
                    "Fish Audio Drama 3 scheduler cursor {} disagrees with the request's {cursor}",
                    scheduled_row.num_computed_tokens
                )));
            }
            let checkpoint = model.begin_quantum(&mut active.state)?;
            lease.mark_dirty();
            rows.push(FishDrama3Row {
                index,
                lease: Some(lease),
                checkpoint: Some(checkpoint),
                fresh,
            });
        }

        let mut pending_guard = FishDrama3PendingBatchGuard {
            coordinator: &self.fish_drama3_pending,
            staged: Vec::new(),
            armed: true,
        };
        if !rows.is_empty() {
            let width = rows.len();
            let started = Instant::now();
            let steps = if is_prefill {
                let row = &mut rows[0];
                let span = scheduled[row.index].num_tokens;
                let state = &mut row.active_mut()?.state;
                let cell = model.prefill_cell(state).ok_or_else(|| {
                    Error::InferenceError("Fish Audio Drama 3 prompt is already prefilled".into())
                })?;
                // The family's grid fixes the cell; a scheduler span shorter
                // than it has no exact cut.
                if cell.len() > span {
                    return Err(Error::InvalidInput(format!(
                        "Fish Audio Drama 3 prefill cell {cell:?} exceeds the scheduled span of \
                         {span} rows"
                    )));
                }
                vec![Self::run_blocking(|| model.prefill_span(state))?]
            } else {
                let mut states = Vec::with_capacity(width);
                for row in &mut rows {
                    let active = row
                        .lease
                        .as_mut()
                        .expect("armed Drama 3 lease")
                        .require_state_mut()?;
                    states.push(&mut active.state);
                }
                Self::run_blocking(|| model.decode_quantum(&mut states))?
            };
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            if steps.len() != width {
                return Err(Error::InferenceError(
                    "Fish Audio Drama 3 batch returned the wrong width".into(),
                ));
            }
            if let Some(call) = retained_tts_batch_model_call(
                if is_prefill {
                    crate::engine::NativeBatchMode::None
                } else {
                    crate::engine::NativeBatchMode::Continuous
                },
                width,
            ) {
                crate::engine::metrics::record_engine_model_call(call);
            }
            for (row, step) in rows.iter_mut().zip(steps) {
                let index = row.index;
                let request = requests[index];
                let scheduled_row = &scheduled[index];
                let runtime = request.managed_cache_runtime().ok_or_else(|| {
                    Error::InferenceError("Fish Audio Drama 3 row lost its managed runtime".into())
                })?;
                let arena = runtime.tensor_state().ok_or_else(|| {
                    Error::InferenceError("Fish Audio Drama 3 row lost its tensor arena".into())
                })?;
                if let Some(terminal) = row_terminal(request, true) {
                    row.rollback()?;
                    outputs[index] = Some(terminal);
                    continue;
                }
                let active = row.active_mut()?;
                // The DeltaNet tensors go to the plan's transaction; the
                // state stays drained until the next quantum restores them.
                active
                    .state
                    .trunk_state_mut()
                    .stage_tensor_state(arena, scheduled_row.plan_id)?;
                let (trunk_writes, tower_writes) = active.state.take_write_completions();
                let completions = trunk_writes.into_iter().chain(tower_writes).collect();
                let (appended, generated) = match step {
                    FishDrama3Step::Prefill { consumed, .. } => {
                        active.prefill_ms += elapsed_ms;
                        active.prefill_steps += 1;
                        (consumed, 0)
                    }
                    FishDrama3Step::Frame { frames } => {
                        active.decode_ms += elapsed_ms / width as f64;
                        active.decode_steps += 1;
                        (1, frames.saturating_sub(active.last_frames))
                    }
                    FishDrama3Step::Finished { reason, .. } => {
                        active.decode_ms += elapsed_ms / width as f64;
                        active.decode_steps += 1;
                        if reason == FishDrama3StopReason::FrameLimit {
                            // A cut-off take is an error, never success.
                            let error = reason
                                .require_complete()
                                .expect_err("a frame limit is incomplete");
                            row.rollback()?;
                            outputs[index] = Some(ModelSessionResult::sequence(
                                ExecutorOutput::error(request.id.clone(), error.to_string()),
                            ));
                            continue;
                        }
                        (0, 0)
                    }
                };
                let active = row.active_mut()?;
                let finished = active.state.is_finished();
                let due = (!is_prefill)
                    .then(|| audio_due(active.state.frames(), &active.stream, finished))
                    .flatten();
                let output = ExecutorOutput {
                    request_id: request.id.clone(),
                    audio: Some(AudioOutput::new(Vec::new(), sample_rate)),
                    text: None,
                    input_transcription: None,
                    tokens_processed: appended,
                    tokens_generated: generated,
                    finished: false,
                    phase_timing_override: Some(timing(active, 0.0, active.first_audio_ms)),
                    asr_diagnostics: None,
                    error: None,
                    logprobs: Vec::new(),
                };
                let result = match due {
                    Some(max_frames) => ModelSessionResult::yielded(
                        output,
                        crate::engine::YieldReason::AwaitingAudioDecode { max_frames },
                    ),
                    None if finished => ModelSessionResult::yielded(
                        output,
                        crate::engine::YieldReason::AwaitingFinalization,
                    ),
                    None => ModelSessionResult::sequence(output),
                };
                let lease = row.lease.take().expect("armed Drama 3 lease");
                let checkpoint = row.checkpoint.take().expect("armed Drama 3 checkpoint");
                self.fish_drama3_pending.stage_quantum(
                    scheduled_row.plan_id,
                    scheduled_row.session_key(),
                    lease,
                    checkpoint,
                    row.fresh,
                )?;
                pending_guard
                    .staged
                    .push((scheduled_row.plan_id, scheduled_row.session_key()));
                outputs[index] = Some(
                    result
                        .with_managed_cache_completions(completions)
                        .with_managed_cache_append(appended)
                        .requiring_pending_quantum(),
                );
            }
        }
        let result = outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| {
                    Error::InferenceError("Fish Audio Drama 3 batch lost an output".into())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        pending_guard.armed = false;
        Ok(result)
    }

    pub(super) fn fish_drama3_tts_audio_decode_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ModelSessionResult> {
        self.fish_drama3_tts_audio_decode_batch(&[request], std::slice::from_ref(scheduled))?
            .pop()
            .ok_or_else(|| {
                Error::InferenceError("Fish Audio Drama 3 codec lost its scalar output".into())
            })
    }

    /// One codec quantum for every row, on committed frames only, in one
    /// batched codec call (batched output equals scalar output).
    pub(super) fn fish_drama3_tts_audio_decode_batch(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ModelSessionResult>> {
        if requests.is_empty() || requests.len() != scheduled.len() {
            return Err(Error::InvalidInput(
                "Fish Audio Drama 3 codec batch row counts differ".into(),
            ));
        }
        let model = requests[0]
            .prepared_fish_drama3_tts_model_lease_for_executor()?
            .ok_or_else(|| {
                Error::InferenceError("Fish Audio Drama 3 codec batch lost model".into())
            })?;
        let sample_rate = model.codec().sample_rate();
        let samples_per_frame = model.codec().samples_per_frame()?;
        let mut outputs: Vec<Option<ModelSessionResult>> =
            (0..requests.len()).map(|_| None).collect();
        let mut rows = Vec::new();
        for (index, (request, scheduled_row)) in requests.iter().zip(scheduled).enumerate() {
            let crate::engine::WorkUnit::SequenceAudioDecode { max_frames } = scheduled_row.work
            else {
                return Err(Error::InvalidInput(
                    "Fish Audio Drama 3 codec batch requires audio work".into(),
                ));
            };
            if request.id != scheduled_row.request_id
                || request
                    .execution_adapter_binding()
                    .map(|binding| binding.adapter_instance_id)
                    != requests[0]
                        .execution_adapter_binding()
                        .map(|binding| binding.adapter_instance_id)
            {
                return Err(Error::InvalidInput(
                    "Fish Audio Drama 3 codec batch crossed adapter identity".into(),
                ));
            }
            if let Some(terminal) = row_terminal(request, false) {
                outputs[index] = Some(terminal);
                continue;
            }
            let variant = Self::resolve_variant(request)?;
            let row_model = request
                .prepared_fish_drama3_tts_model_lease_for_executor()?
                .ok_or_else(|| {
                    Error::InferenceError("Fish Audio Drama 3 codec row lost model".into())
                })?;
            let mut lease = ExecutorStateLease::checkout(
                &self.fish_drama3_tts_decode_states,
                scheduled_row.session_key(),
                variant,
                "Fish Audio Drama 3 codec batch",
            )?;
            let active = lease.require_state_mut()?;
            if active.variant != variant
                || !Arc::ptr_eq(&active.model.model_arc(), &model.model_arc())
                || !Arc::ptr_eq(&row_model.model_arc(), &model.model_arc())
            {
                return Err(Error::InvalidInput(
                    "Fish Audio Drama 3 codec batch crossed model generation".into(),
                ));
            }
            let max_frames = max_frames.min(FISH_DRAMA3_AUDIO_CHUNK_FRAMES);
            // A stream reserves the quantum's PCM before the codec runs and
            // waits for its consumer when the queue is full.
            let output_credit = if Self::stream_sender(request).is_some() {
                let bytes = max_frames
                    .checked_mul(samples_per_frame * std::mem::size_of::<f32>())
                    .ok_or_else(|| {
                        Error::Overloaded("Fish Audio Drama 3 output credit overflow".into())
                    })?;
                let Some(credit) = request.try_reserve_audio_output(bytes)? else {
                    outputs[index] = Some(ModelSessionResult::yielded(
                        empty_output(request, sample_rate),
                        crate::engine::YieldReason::AwaitingAudioOutput { max_frames },
                    ));
                    continue;
                };
                Some(credit)
            } else {
                None
            };
            rows.push((index, lease, max_frames, output_credit));
        }
        let mut pending_guard = FishDrama3PendingBatchGuard {
            coordinator: &self.fish_drama3_pending,
            staged: Vec::new(),
            armed: true,
        };
        if !rows.is_empty() {
            let started = Instant::now();
            let live = rows.iter().map(|row| requests[row.0]).collect::<Vec<_>>();
            let candidates = {
                let mut inputs = Vec::with_capacity(rows.len());
                for (_, lease, max_frames, _) in &mut rows {
                    let active: &ActiveFishDrama3TtsDecode = lease.require_state_mut()?;
                    inputs.push((&active.state, &active.stream, *max_frames));
                }
                let check = || live.iter().try_for_each(|request| check_request(request));
                Self::run_blocking(|| model.stage_audio_batch(&inputs, &check))?
            };
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            if candidates.len() != rows.len() {
                return Err(Error::InferenceError(
                    "Fish Audio Drama 3 codec returned the wrong width".into(),
                ));
            }
            for ((index, mut lease, _, _credit), candidate) in rows.into_iter().zip(candidates) {
                let request = requests[index];
                let scheduled_row = &scheduled[index];
                if let Some(terminal) = row_terminal(request, true) {
                    // Only a candidate stream was written: dropping it and
                    // the clean lease keeps the committed state.
                    outputs[index] = Some(terminal);
                    continue;
                }
                let active = lease.require_state_mut()?;
                let mut sequence = active.stream_sequence;
                if let Some(tx) = Self::stream_sender(request) {
                    Self::stream_audio_with_policy(
                        &tx,
                        request.stream_policy,
                        &request.id,
                        &mut sequence,
                        candidate.samples().to_vec(),
                        sample_rate,
                        false,
                    )?;
                }
                let decoded = active.stream.decoded_frames() + candidate.frames();
                let pending = active.state.frames().saturating_sub(decoded);
                let finished = active.state.is_finished();
                let first_audio_ms = active
                    .first_audio_ms
                    .or_else(|| Some(active.execution_started.elapsed().as_secs_f64() * 1000.0));
                let output_timing = timing(active, active.codec_ms + elapsed_ms, first_audio_ms);
                self.fish_drama3_pending.stage_codec(
                    scheduled_row.plan_id,
                    scheduled_row.session_key(),
                    lease,
                    super::FishDrama3CodecCommit {
                        candidate,
                        stream_sequence: sequence,
                        codec_ms: elapsed_ms,
                        first_audio_ms,
                    },
                )?;
                let mut output = empty_output(request, sample_rate);
                output.phase_timing_override = Some(output_timing);
                let reason = if pending > 0 {
                    crate::engine::YieldReason::AwaitingAudioDecode {
                        max_frames: pending.min(FISH_DRAMA3_AUDIO_CHUNK_FRAMES),
                    }
                } else if finished {
                    crate::engine::YieldReason::AwaitingFinalization
                } else {
                    crate::engine::YieldReason::QuantumExhausted
                };
                pending_guard
                    .staged
                    .push((scheduled_row.plan_id, scheduled_row.session_key()));
                outputs[index] =
                    Some(ModelSessionResult::yielded(output, reason).requiring_pending_quantum());
            }
        }
        let result = outputs
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| {
                    Error::InferenceError("Fish Audio Drama 3 codec batch lost an output".into())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        pending_guard.armed = false;
        Ok(result)
    }

    /// Flushes a stopped request's stream: every committed frame decoded,
    /// its PCM exactly frames × 2,048, the final marker sent.
    pub(super) fn fish_drama3_tts_finalize_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
    ) -> Result<ModelSessionResult> {
        if !matches!(
            scheduled.work,
            crate::engine::WorkUnit::SequenceFinalize {
                max_output_steps: 1
            }
        ) || scheduled.is_prefill
        {
            return Err(Error::InvalidInput(
                "Fish Audio Drama 3 codec requires an exact sequence-finalize quantum".into(),
            ));
        }
        let variant = Self::resolve_variant(request)?;
        if variant.family() != ModelFamily::FishDrama3Tts {
            return Err(Error::InvalidInput(
                "foreign Fish Audio Drama 3 codec request".into(),
            ));
        }
        let model = request
            .prepared_fish_drama3_tts_model_lease_for_executor()?
            .ok_or_else(|| {
                Error::InferenceError("Fish Audio Drama 3 codec lost model residency".into())
            })?;
        let mut lease = ExecutorStateLease::checkout(
            &self.fish_drama3_tts_decode_states,
            scheduled.session_key(),
            variant,
            "Fish Audio Drama 3 codec",
        )?;
        let active = lease.require_state_mut()?;
        if active.variant != variant || !Arc::ptr_eq(&active.model.model_arc(), &model.model_arc())
        {
            return Err(Error::InferenceError(
                "Fish Audio Drama 3 codec crossed its model fence".into(),
            ));
        }
        if active.state.frames() == 0
            || !active.state.is_finished()
            || active.stream.decoded_frames() != active.state.frames()
        {
            return Err(Error::InferenceError(
                "Fish Audio Drama 3 finalization has undecoded frames".into(),
            ));
        }
        check_request(request)?;
        let tail = model.finalize_audio(&active.state, &mut active.stream, &|| {
            check_request(request)
        })?;
        let sample_rate = model.codec().sample_rate();
        active.total_audio_samples += tail.len();
        if active.collect_audio_samples {
            active.audio_samples.extend_from_slice(&tail);
        }
        let output_timing = timing(active, active.codec_ms, active.first_audio_ms);
        if let Some(tx) = Self::stream_sender(request) {
            Self::stream_audio_with_policy(
                &tx,
                request.stream_policy,
                &request.id,
                &mut active.stream_sequence,
                tail,
                sample_rate,
                true,
            )?;
        }
        let audio = if active.collect_audio_samples {
            AudioOutput::new(std::mem::take(&mut active.audio_samples), sample_rate)
        } else {
            AudioOutput::streamed(active.total_audio_samples, sample_rate)
        };
        lease.release()?;
        Ok(ModelSessionResult::sequence(ExecutorOutput {
            request_id: request.id.clone(),
            audio: Some(audio),
            text: None,
            input_transcription: None,
            tokens_processed: 0,
            tokens_generated: 0,
            finished: true,
            phase_timing_override: Some(output_timing),
            asr_diagnostics: None,
            error: None,
            logprobs: Vec::new(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_quanta_follow_the_first_audio_and_chunk_cadence() {
        let fresh = FishDrama3AudioStream::new();
        assert_eq!(audio_due(3, &fresh, false), None);
        assert_eq!(audio_due(4, &fresh, false), Some(4));
        assert_eq!(audio_due(40, &fresh, false), Some(16));
        // A stopped request decodes whatever remains.
        assert_eq!(audio_due(2, &fresh, true), Some(2));
        assert_eq!(audio_due(0, &fresh, true), None);
    }
}
