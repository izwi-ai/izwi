//! LFM3-Audio LiveASR sessions as engine realtime rows. A push's preparation
//! hears its 16 kHz audio and opens the first whole chunk's block; a block
//! prefills alone (the pending embedding and the chunk's encoder tokens) and
//! writes its text one token per decode quantum, the language model batched
//! across sessions. One operation feeds every chunk its audio completes. A
//! finish hears the segment's last audio, and the session ends when its last
//! block closes. The ShortConv ring stays in the session's host state; only
//! the language model's pages live in the managed arena.

use super::super::state::{ActiveLfm3AudioRealtime, PendingLfm3AudioRealtimeQuantum};
use super::*;
use crate::engine::{InputRange, RealtimeOperationId};
use crate::models::architectures::lfm25_audio::state::{
    Lfm25AudioRetainedCheckpoint, Lfm25AudioRetainedMode, Lfm25AudioRetainedState,
};
use crate::models::architectures::lfm3_audio::config::INPUT_SAMPLE_RATE;
use crate::models::architectures::lfm3_audio::model::{Lfm3AudioModel, Lfm3LiveAsr};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

/// One session's quantum in flight, with what a rollback restores.
struct LiveRow<'a> {
    session: SessionKey,
    lease: ExecutorStateLease<'a, ActiveLfm3AudioRealtime>,
    cache: PhysicalPagedKvCache,
    checkpoint: Lfm25AudioRetainedCheckpoint,
    prior_live: Lfm3LiveAsr,
    prior_feed: Option<(candle_core::Tensor, usize)>,
    prior_text_len: usize,
    prior_stream_sequence: usize,
    prior_heard_samples: usize,
    prior_cache_len: usize,
    operation_id: RealtimeOperationId,
    completed: RealtimeSubphase,
    max_cache_append: usize,
    preparation: Option<(RealtimePreparationMode, InputRange)>,
    payload: RealtimeAsrOperationPayload,
}

impl LiveRow<'_> {
    /// Undo the quantum and hand the session back to its store.
    fn abort(mut self) -> Result<()> {
        let active = self.lease.require_state_mut()?;
        active.live = self.prior_live;
        active.feed = self.prior_feed;
        active.text.truncate(self.prior_text_len);
        active.stream_sequence = self.prior_stream_sequence;
        active.heard_samples = self.prior_heard_samples;
        active
            .retained
            .rollback_quantum(&mut self.cache, None, &self.checkpoint)?;
        self.lease.mark_clean();
        self.lease.restore()
    }
}

/// Abort every row of a failed quantum, and drop what it staged to stream.
fn abort_rows(
    requests: &[&EngineCoreRequest],
    rows: Vec<(usize, LiveRow<'_>)>,
    error: Error,
) -> Error {
    let mut failures = Vec::new();
    for (index, row) in rows {
        let _ = requests[index].take_staged_stream_outputs();
        if let Err(rollback) = row.abort() {
            failures.push(rollback.to_string());
        }
    }
    if failures.is_empty() {
        error
    } else {
        Error::InferenceError(format!(
            "{error}; LFM3-Audio realtime rollback also failed: {}",
            failures.join("; ")
        ))
    }
}

impl NativeExecutor {
    pub(super) fn lfm3_audio_realtime_request(
        &self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        managed_state: super::super::RetainedRowManagedState,
    ) -> Result<ModelSessionResult> {
        self.lfm3_audio_realtime_batch_with_managed(
            &[request],
            std::slice::from_ref(scheduled),
            vec![Some(managed_state)],
        )?
        .pop()
        .ok_or_else(|| Error::InferenceError("LFM3-Audio realtime row lost its output".into()))
    }

    /// One quantum for every session row: all preparations, one prefill, or
    /// one decode step each with the language model batched across rows.
    pub(in crate::engine::executor) fn lfm3_audio_realtime_batch_with_managed(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        mut managed: Vec<Option<super::super::RetainedRowManagedState>>,
    ) -> Result<Vec<ModelSessionResult>> {
        if requests.is_empty()
            || requests.len() != scheduled.len()
            || managed.len() != scheduled.len()
        {
            return Err(Error::InvalidInput(
                "LFM3-Audio realtime batch rows do not match".into(),
            ));
        }
        let variant = Self::resolve_variant(requests[0])?;
        let model = self.with_registry(|registry| {
            registry.try_get_lfm25_audio_lease(variant).ok_or_else(|| {
                Error::ModelNotFound(format!("LFM3-Audio model {variant} is not loaded"))
            })
        })?;
        let lfm3 = model.lfm3_audio()?;
        let mut rows = Vec::with_capacity(requests.len());
        for (index, request) in requests.iter().enumerate() {
            let begun = managed[index]
                .take()
                .ok_or_else(|| {
                    Error::InferenceError("LFM3-Audio realtime row lost its managed state".into())
                })
                .and_then(|state| {
                    self.begin_lfm3_live_row(request, &scheduled[index], state, &model, lfm3)
                });
            match begun {
                Ok(row)
                    if rows
                        .first()
                        .is_none_or(|(_, first): &(usize, LiveRow<'_>)| {
                            std::mem::discriminant(&first.completed)
                                == std::mem::discriminant(&row.completed)
                        }) =>
                {
                    rows.push((index, row))
                }
                Ok(row) => {
                    rows.push((index, row));
                    return Err(abort_rows(
                        requests,
                        rows,
                        Error::InvalidInput("LFM3-Audio realtime batch mixed stages".into()),
                    ));
                }
                Err(error) => return Err(abort_rows(requests, rows, error)),
            }
        }
        let nexts = match Self::run_lfm3_live_rows(lfm3, &mut rows) {
            Ok(nexts) => nexts,
            Err(error) => return Err(abort_rows(requests, rows, error)),
        };
        self.seal_lfm3_live_rows(requests, scheduled, rows, nexts)
    }

    /// Check out a session row (starting the session on its first
    /// operation) and open its quantum's checkpoint.
    fn begin_lfm3_live_row<'a>(
        &'a self,
        request: &EngineCoreRequest,
        scheduled: &ScheduledRequest,
        mut managed: super::super::RetainedRowManagedState,
        model: &crate::models::registry::Lfm25AudioModelLease,
        lfm3: &Lfm3AudioModel,
    ) -> Result<LiveRow<'a>> {
        let variant = Self::resolve_variant(request)?;
        if variant.family() != ModelFamily::Lfm3Audio || !request.is_realtime_asr_session() {
            return Err(Error::InvalidInput(
                "LFM3-Audio realtime work crossed its session route".into(),
            ));
        }
        if managed.tensor_state.is_some() {
            return Err(Error::InferenceError(
                "LFM3-Audio realtime row unexpectedly received tensor state".into(),
            ));
        }
        let cache = managed.take_only_paged()?;
        managed.ensure_all_paged_consumed()?;
        let (operation_id, completed, max_cache_append, preparation) = match scheduled.work {
            WorkUnit::RealtimePreparation {
                operation_id,
                mode,
                input,
                max_cache_append,
                ..
            } => (
                operation_id,
                RealtimeSubphase::Preparation,
                max_cache_append,
                Some((mode, input)),
            ),
            WorkUnit::RealtimePromptPrefill {
                operation_id,
                cache_append,
                ..
            } => (
                operation_id,
                RealtimeSubphase::PromptPrefill { cache_append },
                cache_append,
                None,
            ),
            WorkUnit::RealtimeDecodeContinuation {
                operation_id,
                max_cache_append,
                ..
            } => (
                operation_id,
                RealtimeSubphase::DecodeContinuation,
                max_cache_append,
                None,
            ),
            _ => {
                return Err(Error::InvalidInput(
                    "LFM3-Audio LiveASR has no such realtime stage".into(),
                ))
            }
        };
        let payload = request.realtime_asr_operation(operation_id)?;
        let session = scheduled.session_key();
        let mut lease = ExecutorStateLease::checkout(
            &self.lfm3_audio_realtime.states,
            session.clone(),
            variant,
            "LFM3-Audio LiveASR",
        )?;
        if lease.state().is_none() {
            if completed != RealtimeSubphase::Preparation {
                return Err(Error::InferenceError(
                    "LFM3-Audio LiveASR session must start with a preparation".into(),
                ));
            }
            lease.install_state(ActiveLfm3AudioRealtime {
                variant,
                model: model.clone(),
                live: lfm3.start_live_asr(),
                retained: Lfm25AudioRetainedState::new(
                    Lfm25AudioRetainedMode::Asr,
                    lfm3.new_lm_shortconv_state(),
                    0,
                ),
                feed: None,
                text: String::new(),
                stream_sequence: 0,
                heard_samples: 0,
            })?;
        }
        let active = lease.require_state_mut()?;
        if active.variant != variant
            || !std::sync::Arc::ptr_eq(&active.model.model_arc(), &model.model_arc())
        {
            return Err(Error::InferenceError(
                "LFM3-Audio LiveASR session crossed its loaded model".into(),
            ));
        }
        let checkpoint = active.retained.begin_main_quantum(&cache)?;
        let mut row = LiveRow {
            session,
            prior_live: active.live.clone(),
            prior_feed: active.feed.clone(),
            prior_text_len: active.text.len(),
            prior_stream_sequence: active.stream_sequence,
            prior_heard_samples: active.heard_samples,
            prior_cache_len: cache.context_len(),
            lease,
            cache,
            checkpoint,
            operation_id,
            completed,
            max_cache_append,
            preparation,
            payload,
        };
        row.lease.mark_dirty();
        Ok(row)
    }

    /// Run the rows' stage: hear each push, feed one block's prompt span,
    /// or feed every row's last token in one batched forward. Returns each
    /// row's next subphase.
    fn run_lfm3_live_rows(
        lfm3: &Lfm3AudioModel,
        rows: &mut [(usize, LiveRow<'_>)],
    ) -> Result<Vec<Option<RealtimeSubphase>>> {
        let stage = rows
            .first()
            .map(|(_, row)| row.completed)
            .ok_or_else(|| Error::InvalidInput("LFM3-Audio realtime batch is empty".into()))?;
        let mut lasts = Vec::with_capacity(rows.len());
        match stage {
            RealtimeSubphase::Preparation => {
                for (_, row) in rows.iter_mut() {
                    let payload = row.payload.clone();
                    let active = row.lease.require_state_mut()?;
                    match (row.preparation, payload) {
                        (
                            Some((RealtimePreparationMode::Push, input)),
                            RealtimeAsrOperationPayload::Push {
                                samples,
                                sample_rate,
                            },
                        ) if input.start == active.heard_samples
                            && input.len() == samples.len() =>
                        {
                            if sample_rate != INPUT_SAMPLE_RATE {
                                return Err(Error::InvalidInput(format!(
                                    "LFM3-Audio LiveASR hears 16 kHz audio, not {sample_rate} Hz"
                                )));
                            }
                            lfm3.hear_live_asr(&mut active.live, &samples, false)?;
                            active.heard_samples += samples.len();
                        }
                        (
                            Some((RealtimePreparationMode::Finish, _)),
                            RealtimeAsrOperationPayload::Finish,
                        ) => lfm3.hear_live_asr(&mut active.live, &[], true)?,
                        _ => {
                            return Err(Error::InvalidInput(
                                "LFM3-Audio realtime work and operation payload disagree".into(),
                            ))
                        }
                    }
                    lasts.push(None);
                }
            }
            RealtimeSubphase::PromptPrefill { cache_append } => {
                for (_, row) in rows.iter_mut() {
                    let LiveRow { lease, cache, .. } = row;
                    let active = lease.require_state_mut()?;
                    let (input, position) = active.feed.take().ok_or_else(|| {
                        Error::InferenceError("LFM3-Audio prefill has nothing to feed".into())
                    })?;
                    let span = input.dim(1)?;
                    if span != cache_append {
                        return Err(Error::InferenceError(
                            "LFM3-Audio prefill differs from its scheduled span".into(),
                        ));
                    }
                    let hidden = lfm3.lm().forward_embeds_retained(
                        &input,
                        position,
                        cache,
                        active.retained.shortconv_mut(),
                    )?;
                    lasts.push(Some(hidden.narrow(1, span - 1, 1)?.squeeze(1)?));
                }
            }
            RealtimeSubphase::DecodeContinuation => {
                let mut embeds = Vec::with_capacity(rows.len());
                let mut positions = Vec::with_capacity(rows.len());
                let mut shortconvs = Vec::with_capacity(rows.len());
                let mut caches = Vec::with_capacity(rows.len());
                for (_, row) in rows.iter_mut() {
                    let LiveRow { lease, cache, .. } = row;
                    let active = lease.require_state_mut()?;
                    let (embed, position) = active.feed.take().ok_or_else(|| {
                        Error::InferenceError("LFM3-Audio decode has nothing to feed".into())
                    })?;
                    embeds.push(embed);
                    positions.push(position);
                    shortconvs.push(active.retained.shortconv_mut());
                    caches.push(cache);
                }
                let output = lfm3.lm().forward_embeds_retained_batch(
                    &embeds,
                    &positions,
                    &mut shortconvs,
                    &mut caches,
                )?;
                for hidden in output.last_hidden {
                    lasts.push(Some(hidden.squeeze(1)?));
                }
            }
            RealtimeSubphase::Completion => {
                return Err(Error::InvalidInput(
                    "LFM3-Audio LiveASR has no completion stage".into(),
                ))
            }
        }
        rows.iter_mut()
            .zip(lasts)
            .map(|((_, row), last)| {
                let active = row.lease.require_state_mut()?;
                if let Some(last) = last {
                    if let Some(feed) = lfm3.step_live_asr(&mut active.live, &last)? {
                        active.feed = Some(feed);
                        return Ok(Some(RealtimeSubphase::DecodeContinuation));
                    }
                }
                match lfm3.open_live_asr_block(&mut active.live)? {
                    Some((input, position)) => {
                        let cache_append = input.dim(1)?;
                        active.feed = Some((input, position));
                        Ok(Some(RealtimeSubphase::PromptPrefill { cache_append }))
                    }
                    None => Ok(None),
                }
            })
            .collect()
    }

    /// Stream each row's new text, check its KV append, and hand its session
    /// to Core's commit as a pending quantum.
    fn seal_lfm3_live_rows(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
        mut rows: Vec<(usize, LiveRow<'_>)>,
        nexts: Vec<Option<RealtimeSubphase>>,
    ) -> Result<Vec<ModelSessionResult>> {
        let mut sealed = Vec::with_capacity(rows.len());
        let sealing = (|| -> Result<()> {
            for ((index, row), next) in rows.iter_mut().zip(&nexts) {
                let request = requests[*index];
                let expected_append = match row.completed {
                    RealtimeSubphase::PromptPrefill { cache_append } => cache_append,
                    RealtimeSubphase::DecodeContinuation => 1,
                    _ => 0,
                };
                let appended = row
                    .cache
                    .context_len()
                    .checked_sub(row.prior_cache_len)
                    .filter(|appended| {
                        *appended == expected_append && *appended <= row.max_cache_append
                    })
                    .ok_or_else(|| {
                        Error::InferenceError(
                            "LFM3-Audio realtime cache append violated its bound".into(),
                        )
                    })?;
                let active = row.lease.require_state_mut()?;
                let delta = active.live.take_text();
                active.text.push_str(&delta);
                let finished = next.is_none() && active.live.finished();
                if let Some(tx) = Self::stream_sender(request) {
                    if !delta.is_empty() {
                        Self::stream_text_with_policy(
                            &tx,
                            request.stream_policy,
                            &request.id,
                            &mut active.stream_sequence,
                            delta,
                        )?;
                    }
                    if finished {
                        Self::stream_final_marker_with_policy(
                            &tx,
                            request.stream_policy,
                            &request.id,
                            &mut active.stream_sequence,
                        )?;
                    }
                }
                let staged = request.take_staged_stream_outputs()?;
                let input_consumed = match (&row.payload, next) {
                    (RealtimeAsrOperationPayload::Push { samples, .. }, None) => samples.len(),
                    _ => 0,
                };
                let output_steps = usize::from(matches!(
                    row.completed,
                    RealtimeSubphase::PromptPrefill { .. } | RealtimeSubphase::DecodeContinuation
                ));
                sealed.push((
                    staged,
                    RealtimeStageOutcome {
                        plan_id: scheduled[*index].plan_id,
                        operation_id: row.operation_id,
                        completed: row.completed,
                        next: *next,
                        input_consumed,
                        output_steps,
                        cache_append: appended,
                    },
                    active.text.clone(),
                    active.heard_samples,
                    finished,
                ));
            }
            for (_, row) in &rows {
                row.lease.validate_defer()?;
            }
            Ok(())
        })();
        if let Err(error) = sealing {
            return Err(abort_rows(requests, rows, error));
        }
        let mut outputs = Vec::with_capacity(rows.len());
        let mut pending = Vec::with_capacity(rows.len());
        for ((index, mut row), (staged, outcome, text, heard_samples, finished)) in
            rows.into_iter().zip(sealed)
        {
            let completions = row.cache.take_completed_writes();
            let appended = outcome.cache_append;
            let output_steps = outcome.output_steps;
            pending.push((
                outcome.plan_id,
                PendingLfm3AudioRealtimeQuantum {
                    session: row.session,
                    active: row.lease.defer_validated(),
                    cache: row.cache,
                    checkpoint: row.checkpoint,
                    prior_live: row.prior_live,
                    prior_feed: row.prior_feed,
                    prior_text_len: row.prior_text_len,
                    prior_stream_sequence: row.prior_stream_sequence,
                    prior_heard_samples: row.prior_heard_samples,
                    finished,
                },
            ));
            outputs.push(
                ModelSessionResult::sequence(ExecutorOutput {
                    request_id: requests[index].id.clone(),
                    audio: Some(AudioOutput {
                        streamed_samples: None,
                        samples: Vec::new(),
                        sample_rate: INPUT_SAMPLE_RATE,
                        duration_secs: heard_samples as f32 / INPUT_SAMPLE_RATE as f32,
                    }),
                    text: Some(text),
                    input_transcription: None,
                    tokens_processed: 0,
                    tokens_generated: output_steps,
                    finished,
                    phase_timing_override: None,
                    asr_diagnostics: None,
                    error: None,
                    logprobs: Vec::new(),
                })
                .with_staged_stream_outputs(staged)
                .with_managed_cache_completions(completions)
                .with_managed_cache_append(appended)
                .requiring_pending_quantum()
                .with_realtime_stage_outcome(outcome),
            );
        }
        if let Err((error, pending)) = self.lfm3_audio_realtime.register_batch(pending) {
            let mut failures = Vec::new();
            for (_, quantum) in pending {
                let session = quantum.session.clone();
                if let Err(rollback) = self
                    .lfm3_audio_realtime
                    .resolve_pending(quantum, super::super::PendingQuantumDecision::Abort)
                    .and_then(|prepared| {
                        self.lfm3_audio_realtime
                            .replace_in_flight(&session, prepared.replacement)
                    })
                {
                    failures.push(rollback.to_string());
                }
            }
            return Err(if failures.is_empty() {
                error
            } else {
                Error::InferenceError(format!(
                    "{error}; LFM3-Audio realtime rollback also failed: {}",
                    failures.join("; ")
                ))
            });
        }
        Ok(outputs)
    }
}
