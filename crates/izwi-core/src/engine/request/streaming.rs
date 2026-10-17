//! Stream staging: the progress budget, audio output credits and the staging
//! buffer a streaming request fills between commits.

use super::*;

impl StreamProgressBudget {
    pub(in crate::engine) fn new(max_bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            max_bytes: max_bytes.max(1),
            state: Mutex::new(0),
            available: Condvar::new(),
        })
    }

    pub(super) fn reserve(
        self: &Arc<Self>,
        bytes: usize,
        policy: EngineStreamPolicy,
    ) -> Result<Option<StreamProgressPermit>> {
        if bytes > STREAM_PROGRESS_MAX_EVENT_BYTES || bytes > self.max_bytes {
            return Err(Error::Overloaded(format!(
                "stream progress event requires {bytes} bytes, exceeding its bounded budget"
            )));
        }

        let mut used = self
            .state
            .lock()
            .map_err(|_| Error::InferenceError("stream progress budget mutex poisoned".into()))?;
        let admits = |used: usize| {
            used.checked_add(bytes)
                .is_some_and(|next| next <= self.max_bytes)
        };

        match policy {
            EngineStreamPolicy::FailOnFull => {
                if !admits(*used) {
                    record_engine_stream_backpressure();
                    return Err(Error::Overloaded(
                        "stream progress byte budget is full".to_string(),
                    ));
                }
            }
            EngineStreamPolicy::DropNewest => {
                if !admits(*used) {
                    record_engine_stream_backpressure();
                    return Ok(None);
                }
            }
            EngineStreamPolicy::BlockWithDeadline { timeout_ms } => {
                let deadline = checked_stream_backpressure_deadline(timeout_ms)?;
                while !admits(*used) {
                    let now = Instant::now();
                    if now >= deadline {
                        record_engine_stream_backpressure();
                        return Err(Error::Overloaded(
                            "stream progress byte-budget deadline elapsed".to_string(),
                        ));
                    }
                    let wait = deadline.saturating_duration_since(now);
                    let (next, result) = self.available.wait_timeout(used, wait).map_err(|_| {
                        Error::InferenceError("stream progress budget mutex poisoned".into())
                    })?;
                    used = next;
                    if result.timed_out() && !admits(*used) {
                        record_engine_stream_backpressure();
                        return Err(Error::Overloaded(
                            "stream progress byte-budget deadline elapsed".to_string(),
                        ));
                    }
                }
            }
        }

        *used = used.checked_add(bytes).ok_or_else(|| {
            Error::Overloaded("stream progress byte accounting overflowed".to_string())
        })?;
        Ok(Some(StreamProgressPermit {
            budget: self.clone(),
            bytes,
        }))
    }
}

impl Drop for StreamProgressPermit {
    fn drop(&mut self) {
        let Ok(mut used) = self.budget.state.lock() else {
            return;
        };
        *used = used.saturating_sub(self.bytes);
        self.budget.available.notify_all();
    }
}

pub(super) fn committed_audio_budget() -> &'static Arc<StreamProgressBudget> {
    static BUDGET: OnceLock<Arc<StreamProgressBudget>> = OnceLock::new();
    BUDGET.get_or_init(|| StreamProgressBudget::new(64 * 1024 * 1024))
}

impl IncrementalStreamBinding {
    pub(super) fn send(
        &self,
        output: StreamingOutput,
        policy: EngineStreamPolicy,
    ) -> Result<StreamPushOutcome> {
        let payload_bytes = stream_payload_bytes(&output)?;
        let Some(budget_permit) = self.budget.reserve(payload_bytes, policy)? else {
            return Ok(StreamPushOutcome::Dropped);
        };
        let mut progress = FencedStreamProgress {
            batch_id: self.batch_id,
            lane: self.lane.clone(),
            plan_id: self.plan_id,
            session: self.session.clone(),
            output,
            budget_permit,
        };

        match policy {
            EngineStreamPolicy::FailOnFull => self
                .progress_tx
                .try_send(progress)
                .map(|_| StreamPushOutcome::Accepted)
                .map_err(stream_progress_send_error),
            EngineStreamPolicy::DropNewest => match self.progress_tx.try_send(progress) {
                Ok(()) => Ok(StreamPushOutcome::Accepted),
                Err(mpsc::error::TrySendError::Closed(progress)) => Err(
                    stream_progress_send_error(mpsc::error::TrySendError::Closed(progress)),
                ),
                Err(mpsc::error::TrySendError::Full(_)) => {
                    record_engine_stream_backpressure();
                    Ok(StreamPushOutcome::Dropped)
                }
            },
            EngineStreamPolicy::BlockWithDeadline { timeout_ms } => {
                let deadline = checked_stream_backpressure_deadline(timeout_ms)?;
                loop {
                    match self.progress_tx.try_send(progress) {
                        Ok(()) => return Ok(StreamPushOutcome::Accepted),
                        Err(mpsc::error::TrySendError::Closed(progress)) => {
                            return Err(stream_progress_send_error(
                                mpsc::error::TrySendError::Closed(progress),
                            ));
                        }
                        Err(mpsc::error::TrySendError::Full(returned)) => {
                            progress = returned;
                            let now = Instant::now();
                            if now >= deadline {
                                record_engine_stream_backpressure();
                                return Err(Error::Overloaded(
                                    "stream progress queue deadline elapsed".to_string(),
                                ));
                            }
                            std::thread::sleep(
                                deadline
                                    .saturating_duration_since(now)
                                    .min(Duration::from_millis(1)),
                            );
                        }
                    }
                }
            }
        }
    }
}

pub(super) fn stream_progress_send_error(err: mpsc::error::TrySendError<FencedStreamProgress>) -> Error {
    match err {
        mpsc::error::TrySendError::Closed(_) => {
            Error::InferenceError("stream progress channel closed".to_string())
        }
        mpsc::error::TrySendError::Full(_) => {
            record_engine_stream_backpressure();
            Error::Overloaded("stream progress queue is full".to_string())
        }
    }
}

pub(super) fn stream_payload_bytes(output: &StreamingOutput) -> Result<usize> {
    let sample_bytes = output
        .samples
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::Overloaded("stream sample byte size overflowed".to_string()))?;
    output
        .request_id
        .len()
        .checked_add(output.text.as_ref().map_or(0, String::len))
        .and_then(|bytes| bytes.checked_add(sample_bytes))
        .and_then(|bytes| bytes.checked_add(256))
        .ok_or_else(|| Error::Overloaded("stream payload byte size overflowed".to_string()))
}

impl StreamStagingBuffer {
    #[cfg(test)]
    pub(in crate::engine) fn clear(&self) -> Result<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::InferenceError("stream staging mutex poisoned".to_string()))?;
        state.staged.clear();
        state.audio_credits.clear();
        state.staged_bytes = 0;
        state.binding = None;
        Ok(())
    }

    pub(in crate::engine) fn bind_quantum(
        &self,
        batch_id: BatchId,
        lane: BatchLaneKey,
        plan_id: PlanId,
        session: SessionKey,
        visibility: OutputVisibility,
        progress_tx: mpsc::Sender<FencedStreamProgress>,
        budget: Arc<StreamProgressBudget>,
    ) -> Result<StreamBindingGuard> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::InferenceError("stream staging mutex poisoned".to_string()))?;
        if state.binding.is_some() {
            return Err(Error::InferenceError(
                "stream progress binding is already active".to_string(),
            ));
        }
        state.staged.clear();
        state.audio_credits.clear();
        state.staged_bytes = 0;
        if visibility == OutputVisibility::IncrementalCommitted {
            state.binding = Some(IncrementalStreamBinding {
                batch_id,
                lane,
                plan_id,
                session: session.clone(),
                progress_tx,
                budget,
            });
        }
        Ok(StreamBindingGuard {
            buffer: self.clone(),
            plan_id,
            session,
        })
    }

    pub(in crate::engine) fn push_with_policy(
        &self,
        output: StreamingOutput,
        policy: EngineStreamPolicy,
    ) -> Result<StreamPushOutcome> {
        let binding = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| Error::InferenceError("stream staging mutex poisoned".to_string()))?;
            if output.is_final || state.binding.is_none() {
                let payload_bytes = stream_payload_bytes(&output)?;
                let next_bytes =
                    state
                        .staged_bytes
                        .checked_add(payload_bytes)
                        .ok_or_else(|| {
                            Error::Overloaded("stream staging size overflowed".to_string())
                        })?;
                if state.staged.len() >= STREAM_STAGED_MAX_EVENTS
                    || next_bytes > STREAM_PROGRESS_MAX_BUFFERED_BYTES
                {
                    record_engine_stream_backpressure();
                    if policy == EngineStreamPolicy::DropNewest && !output.is_final {
                        return Ok(StreamPushOutcome::Dropped);
                    }
                    return Err(Error::Overloaded(
                        "stream staging outbox exceeded its bounded capacity".to_string(),
                    ));
                }
                if !state.audio_credits.is_empty()
                    && next_bytes
                        > state
                            .audio_credits
                            .iter()
                            .map(|credit| credit.bytes)
                            .sum::<usize>()
                {
                    return Err(Error::Overloaded(
                        "Audio output exceeded its pre-dispatch credit".into(),
                    ));
                }
                state.staged.push(output);
                state.staged_bytes = next_bytes;
                return Ok(StreamPushOutcome::Accepted);
            }
            state.binding.clone().expect("binding checked above")
        };
        binding.send(output, policy)
    }

    pub(in crate::engine) fn take(&self) -> Result<Vec<StreamingOutput>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::InferenceError("stream staging mutex poisoned".to_string()))?;
        state.staged_bytes = 0;
        Ok(std::mem::take(&mut state.staged))
    }

    #[cfg(test)]
    pub(in crate::engine) fn has_incremental_binding(&self) -> bool {
        self.state
            .lock()
            .map(|state| state.binding.is_some())
            .unwrap_or(false)
    }

    pub(super) fn clear_binding(&self, plan_id: PlanId, session: &SessionKey) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if state
            .binding
            .as_ref()
            .is_some_and(|binding| binding.plan_id == plan_id && &binding.session == session)
        {
            state.binding = None;
        }
    }
}

impl Drop for StreamBindingGuard {
    fn drop(&mut self) {
        self.buffer.clear_binding(self.plan_id, &self.session);
    }
}

impl EngineCoreRequest {
    pub(crate) fn mark_audio_streaming_only(&mut self) {
        self.collect_audio_samples = false;
        if self.audio_stream_engine_queue_capacity.is_none() {
            self.audio_stream_engine_queue_capacity =
                Some(super::super::Engine::streaming_queue_capacity(self));
        }
    }

    /// Price all independently queued PCM, not only the credited codec outbox.
    pub(crate) fn streaming_audio_buffer_bytes(&self, max_chunk_samples: usize) -> Result<u64> {
        let engine_rows = self.audio_stream_engine_queue_capacity.ok_or_else(|| {
            Error::InvalidInput(
                "streaming audio queue capacity was not frozen before admission".into(),
            )
        })?;
        let chunk_bytes = max_chunk_samples
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| bytes.checked_add(self.id.len()))
            .and_then(|bytes| bytes.checked_add(512))
            .ok_or_else(|| Error::Overloaded("streaming chunk reservation overflow".into()))?;
        let rows = engine_rows
            .checked_add(self.audio_stream_external_queue_capacity)
            .and_then(|rows| rows.checked_add(3))
            .ok_or_else(|| Error::Overloaded("streaming queue reservation overflow".into()))?;
        let bytes = rows
            .checked_mul(chunk_bytes)
            .and_then(|bytes| bytes.checked_add(STREAM_PROGRESS_MAX_BUFFERED_BYTES))
            .ok_or_else(|| Error::Overloaded("streaming PCM reservation overflow".into()))?;
        u64::try_from(bytes)
            .map_err(|_| Error::Overloaded("streaming PCM reservation exceeds u64".into()))
    }

    /// Nonblocking pre-codec admission. None means this row should yield while
    /// peers progress; no codec history or samples may be materialized yet.
    pub(crate) fn try_reserve_audio_output(
        &self,
        sample_bytes: usize,
    ) -> Result<Option<AudioOutputCredit>> {
        let bytes = sample_bytes
            .checked_add(self.id.len())
            .and_then(|bytes| bytes.checked_add(256))
            .ok_or_else(|| Error::Overloaded("Audio output credit overflow".into()))?;
        let mut state = self
            .stream_staging
            .state
            .lock()
            .map_err(|_| Error::InferenceError("stream staging mutex poisoned".into()))?;
        let budget = state
            .audio_budget
            .get_or_insert_with(|| StreamProgressBudget::new(4 * 1024 * 1024))
            .clone();
        let Some(local) = budget.reserve(bytes, EngineStreamPolicy::DropNewest)? else {
            return Ok(None);
        };
        let Some(global) =
            committed_audio_budget().reserve(bytes, EngineStreamPolicy::DropNewest)?
        else {
            return Ok(None);
        };
        let credit = AudioOutputCredit {
            _permits: Arc::new(vec![local, global]),
            bytes,
        };
        state.audio_credits.push(credit.clone());
        Ok(Some(credit))
    }

    pub(in crate::engine) fn take_audio_output_credits(&self) -> Vec<AudioOutputCredit> {
        let mut state = self
            .stream_staging
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        std::mem::take(&mut state.audio_credits)
    }

    #[cfg(test)]
    pub(in crate::engine) fn begin_stream_staging(&self) -> Result<()> {
        self.stream_staging.clear()
    }

    pub(in crate::engine) fn bind_stream_quantum(
        &self,
        batch_id: BatchId,
        lane: BatchLaneKey,
        plan_id: PlanId,
        session: SessionKey,
        visibility: OutputVisibility,
        progress_tx: mpsc::Sender<FencedStreamProgress>,
        budget: Arc<StreamProgressBudget>,
    ) -> Result<StreamBindingGuard> {
        self.stream_staging.bind_quantum(
            batch_id,
            lane,
            plan_id,
            session,
            visibility,
            progress_tx,
            budget,
        )
    }

    pub(in crate::engine) fn stream_staging_buffer(&self) -> StreamStagingBuffer {
        self.stream_staging.clone()
    }

    pub(in crate::engine) fn take_staged_stream_outputs(&self) -> Result<Vec<StreamingOutput>> {
        self.stream_staging.take()
    }
}
