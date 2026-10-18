//! Cross-request ASR encoder batchers: Qwen3-ASR, Whisper, Granite Speech and
//! VibeVoice encoder work queued by request preparation and run as one batch.

use super::*;

pub(super) type QwenAsrEncoderOutcome = PreparationRowOutcome<Qwen3AsrPreparedAudio>;

pub(super) struct QwenAsrEncoderPending {
    pub(super) job: JobLease,
    pub(super) contract: LoadedExecutionContract,
    pub(super) model: AsrModelLease,
    pub(super) samples: Arc<[f32]>,
    pub(super) sample_rate: u32,
    pub(super) geometry: Qwen3AsrAudioPreparationGeometry,
    pub(super) retained_host_bytes: u64,
    pub(super) cancellation: PreparationCancellation,
    pub(super) response: Option<oneshot::Sender<QwenAsrEncoderOutcome>>,
}

#[derive(Default)]
pub(super) struct QwenAsrEncoderBatcherState {
    pub(super) pending: HashMap<QwenAsrEncoderQueueKey, VecDeque<QwenAsrEncoderPending>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct QwenAsrEncoderQueueKey {
    pub(super) binding: AdapterBindingKey,
    pub(super) mel_frame_bucket: usize,
}

pub(super) struct QwenAsrEncoderBatcher {
    pub(super) coordinator: Arc<InferenceCoordinator>,
    pub(super) state: Mutex<QwenAsrEncoderBatcherState>,
}

pub(super) type WhisperEncoderOutcome = PreparationRowOutcome<WhisperPreparedWindow>;

pub(super) struct WhisperEncoderPending {
    pub(super) job: JobLease,
    pub(super) contract: LoadedExecutionContract,
    pub(super) model: AsrModelLease,
    pub(super) samples: Arc<[f32]>,
    pub(super) sample_rate: u32,
    pub(super) geometry: WhisperWindowPreparationGeometry,
    pub(super) retained_host_bytes: u64,
    pub(super) cancellation: PreparationCancellation,
    pub(super) response: Option<oneshot::Sender<WhisperEncoderOutcome>>,
}

#[derive(Default)]
pub(super) struct WhisperEncoderBatcherState {
    pub(super) pending: HashMap<WhisperEncoderQueueKey, VecDeque<WhisperEncoderPending>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct WhisperEncoderQueueKey {
    pub(super) binding: AdapterBindingKey,
    pub(super) mel_frame_bucket: usize,
}

pub(super) struct WhisperEncoderBatcher {
    pub(super) coordinator: Arc<InferenceCoordinator>,
    pub(super) state: Mutex<WhisperEncoderBatcherState>,
}

pub(super) type VibeVoiceEncoderOutcome = PreparationRowOutcome<VibeVoiceAsrPreparedArtifact>;

pub(super) type GraniteSpeechEncoderOutcome = PreparationRowOutcome<Arc<GraniteSpeechPreparedPromptArtifact>>;

pub(super) struct GraniteSpeechEncoderPending {
    pub(super) job: JobLease,
    pub(super) contract: LoadedExecutionContract,
    pub(super) model: AsrModelLease,
    pub(super) samples: Arc<[f32]>,
    pub(super) sample_rate: u32,
    pub(super) language: Option<String>,
    pub(super) prompt: Option<String>,
    pub(super) geometry: GraniteSpeechPreparedGeometry,
    pub(super) retained_host_bytes: u64,
    pub(super) artifact_host_bytes: u64,
    pub(super) cancellation: PreparationCancellation,
    pub(super) response: Option<oneshot::Sender<GraniteSpeechEncoderOutcome>>,
}

#[derive(Default)]
pub(super) struct GraniteSpeechEncoderBatcherState {
    pub(super) pending: HashMap<GraniteSpeechEncoderQueueKey, VecDeque<GraniteSpeechEncoderPending>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct GraniteSpeechEncoderQueueKey {
    pub(super) binding: AdapterBindingKey,
    pub(super) audio_token_bucket: usize,
    pub(super) prompt_token_bucket: usize,
    pub(super) deadline_budget_bucket: Option<u64>,
}

pub(super) struct GraniteSpeechEncoderBatcher {
    pub(super) coordinator: Arc<InferenceCoordinator>,
    pub(super) state: Mutex<GraniteSpeechEncoderBatcherState>,
}

pub(super) fn granite_speech_deadline_budget_bucket(deadline: Option<Instant>, now: Instant) -> Option<u64> {
    deadline.map(|deadline| {
        u64::try_from(deadline.saturating_duration_since(now).as_millis()).unwrap_or(u64::MAX) / 100
    })
}

pub(super) struct VibeVoiceEncoderPending {
    pub(super) job: JobLease,
    pub(super) contract: LoadedExecutionContract,
    pub(super) model: AsrModelLease,
    pub(super) samples: Arc<[f32]>,
    pub(super) sample_rate: u32,
    pub(super) language: Option<String>,
    pub(super) prompt: Option<String>,
    pub(super) geometry: VibeVoiceAsrPreparedGeometry,
    pub(super) retained_request_host_bytes: u64,
    pub(super) cancellation: PreparationCancellation,
}

pub(super) struct VibeVoiceEncoderBatcher {
    pub(super) coordinator: Arc<InferenceCoordinator>,
}

impl VibeVoiceEncoderBatcher {
    pub(super) fn new(coordinator: Arc<InferenceCoordinator>) -> Self {
        Self { coordinator }
    }

    pub(super) async fn submit(&self, pending: VibeVoiceEncoderPending) -> Result<VibeVoiceEncoderOutcome> {
        let work = WorkUnit::PreSequencePreparation {
            kind: "asr.encoder.vibevoice".into(),
        };
        let cost = pending.geometry.work_cost();
        let cancellation = pending.cancellation.clone();
        let row = self.coordinator.seal_preparation_row(
            pending.job,
            &pending.contract,
            &work,
            cost,
            pending.geometry.embedding_elements,
            pending.cancellation,
        )?;
        let model = pending.model;
        let samples = pending.samples;
        let sample_rate = pending.sample_rate;
        let language = pending.language;
        let prompt = pending.prompt;
        let retained_request_host_bytes = pending.retained_request_host_bytes;
        let mut guard = PreparationCancellationGuard {
            cancellation,
            armed: true,
        };
        let mut outcomes = self
            .coordinator
            .run_loaded_native_preparation_batch(vec![row], pending.contract, work, move |live| {
                if live != [0] {
                    return Err(Error::InferenceError(
                        "VibeVoice scalar preparation received a non-scalar live set".into(),
                    ));
                }
                let artifact = model.prepare_vibevoice_retained_artifact(
                    samples.as_ref(),
                    sample_rate,
                    language.as_deref(),
                    prompt.as_deref(),
                )?;
                let host_bytes = retained_request_host_bytes
                    .checked_add(artifact.resident_host_bytes())
                    .ok_or_else(|| {
                        Error::Overloaded("VibeVoice ASR retained host accounting overflow".into())
                    })?;
                Ok(vec![Ok(PreparationArtifact {
                    retained: JobResourceObservation {
                        host_bytes,
                        accelerator_bytes: artifact.resident_tensor_bytes(),
                    },
                    value: artifact,
                })])
            })
            .await?;
        guard.armed = false;
        outcomes.pop().ok_or_else(|| {
            Error::InferenceError("VibeVoice scalar preparation returned no outcome".into())
        })
    }
}

impl WhisperEncoderBatcher {
    pub(super) fn new(coordinator: Arc<InferenceCoordinator>) -> Self {
        Self {
            coordinator,
            state: Mutex::new(WhisperEncoderBatcherState::default()),
        }
    }

    pub(super) async fn submit(
        self: &Arc<Self>,
        mut pending: WhisperEncoderPending,
    ) -> Result<WhisperEncoderOutcome> {
        let work = WorkUnit::PreSequencePreparation {
            kind: "asr.encoder.whisper".to_string(),
        };
        let binding = pending.contract.adapter_binding()?;
        let stage = binding.stage_for_work(&work)?;
        if stage.name != "asr.encoder.whisper" {
            return Err(Error::InvalidInput(
                "Whisper loaded contract did not select asr.encoder.whisper".into(),
            ));
        }
        let key = WhisperEncoderQueueKey {
            binding: binding.key_for_stage(stage.id)?,
            mel_frame_bucket: pending
                .geometry
                .useful_mel_frames
                .checked_next_power_of_two()
                .ok_or_else(|| Error::Overloaded("Whisper mel-frame bucket overflow".into()))?,
        };
        let max_width = stage.max_batch_size.max(1);
        let formation_delay = stage.max_formation_delay;
        let deadline = pending.job.spec.deadline;
        let cancellation = pending.cancellation.clone();
        let (sender, receiver) = oneshot::channel();
        pending.response = Some(sender);
        let mut immediate = None;
        let first;
        {
            let mut state = self.state.lock().await;
            let queue = state.pending.entry(key.clone()).or_default();
            first = queue.is_empty();
            queue.push_back(pending);
            let pressure = deadline.is_some_and(|deadline| {
                deadline
                    <= Instant::now()
                        .checked_add(formation_delay)
                        .unwrap_or(deadline)
            });
            if queue.len() >= max_width || pressure {
                immediate = Some(queue.drain(..queue.len().min(max_width)).collect());
                if queue.is_empty() {
                    state.pending.remove(&key);
                }
            }
        }
        if let Some(batch) = immediate {
            self.spawn(batch);
        } else if first {
            let batcher = self.clone();
            tokio::spawn(async move {
                yield_now().await;
                if !formation_delay.is_zero() {
                    tokio::time::sleep(formation_delay).await;
                }
                let batch = {
                    let mut state = batcher.state.lock().await;
                    let Some(queue) = state.pending.get_mut(&key) else {
                        return;
                    };
                    let batch = queue
                        .drain(..queue.len().min(max_width))
                        .collect::<Vec<_>>();
                    if queue.is_empty() {
                        state.pending.remove(&key);
                    }
                    batch
                };
                batcher.spawn(batch);
            });
        }
        struct CancelOnDrop(PreparationCancellation, bool);
        impl Drop for CancelOnDrop {
            fn drop(&mut self) {
                if self.1 {
                    self.0.cancel();
                }
            }
        }
        let mut guard = CancelOnDrop(cancellation, true);
        let outcome = receiver.await.map_err(|_| {
            Error::InferenceError("Whisper encoder batch worker stopped before reply".into())
        })?;
        guard.1 = false;
        Ok(outcome)
    }

    pub(super) fn spawn(self: &Arc<Self>, batch: Vec<WhisperEncoderPending>) {
        let this = self.clone();
        tokio::spawn(async move {
            this.run_batch(batch).await;
        });
    }

    pub(super) async fn run_batch(&self, mut rows: Vec<WhisperEncoderPending>) {
        let Some(first) = rows.first() else {
            return;
        };
        let contract = first.contract.clone();
        let model = first.model.clone();
        let geometries = rows.iter().map(|row| row.geometry).collect::<Vec<_>>();
        let batch_geometry = match model.whisper_window_preparation_batch_geometry(&geometries) {
            Ok(value) => value,
            Err(error) => {
                Self::fail(rows, error);
                return;
            }
        };
        let mut sealed = Vec::with_capacity(rows.len());
        let work = WorkUnit::PreSequencePreparation {
            kind: "asr.encoder.whisper".into(),
        };
        for (index, row) in rows.iter().enumerate() {
            let cost = match model.whisper_window_preparation_row_cost_for_batch(
                index,
                &geometries,
                &batch_geometry,
            ) {
                Ok(value) => value,
                Err(error) => {
                    Self::fail(rows, error);
                    return;
                }
            };
            match self.coordinator.seal_preparation_row(
                row.job.clone(),
                &row.contract,
                &work,
                cost,
                batch_geometry.materialized_tensor_elements_per_row,
                row.cancellation.clone(),
            ) {
                Ok(value) => sealed.push(value),
                Err(error) => {
                    Self::fail(rows, error);
                    return;
                }
            }
        }
        let inputs = rows
            .iter()
            .map(|row| {
                (
                    row.samples.clone(),
                    row.sample_rate,
                    row.retained_host_bytes,
                )
            })
            .collect::<Vec<_>>();
        let senders = rows
            .drain(..)
            .map(|mut row| {
                row.response
                    .take()
                    .expect("queued Whisper row has response channel")
            })
            .collect::<Vec<_>>();
        let physical = model.clone();
        let result = self
            .coordinator
            .run_loaded_native_preparation_batch(sealed, contract, work, move |live| {
                let selected = live
                    .iter()
                    .map(|index| WhisperAudioBatchRow {
                        audio: inputs[*index].0.as_ref(),
                        sample_rate: inputs[*index].1,
                    })
                    .collect::<Vec<_>>();
                let prepared = physical.prepare_whisper_window_batch(&selected)?;
                Ok(prepared
                    .into_iter()
                    .zip(live)
                    .map(|(artifact, index)| {
                        Ok(PreparationArtifact {
                            retained: JobResourceObservation {
                                host_bytes: inputs[*index].2,
                                accelerator_bytes: artifact.resident_tensor_bytes()?,
                            },
                            value: artifact,
                        })
                    })
                    .collect::<Vec<Result<PreparationArtifact<WhisperPreparedWindow>>>>())
            })
            .await;
        match result {
            Ok(outcomes) => {
                for (sender, outcome) in senders.into_iter().zip(outcomes) {
                    let _ = sender.send(outcome);
                }
            }
            Err(error) => {
                for sender in senders {
                    let _ = sender.send(PreparationRowOutcome::Failed(Error::InferenceError(
                        error.to_string(),
                    )));
                }
            }
        }
    }

    pub(super) fn fail(rows: Vec<WhisperEncoderPending>, error: Error) {
        for row in rows {
            if let Some(sender) = row.response {
                let _ = sender.send(PreparationRowOutcome::Failed(Error::InferenceError(
                    error.to_string(),
                )));
            }
        }
    }
}

impl GraniteSpeechEncoderBatcher {
    pub(super) fn new(coordinator: Arc<InferenceCoordinator>) -> Self {
        Self {
            coordinator,
            state: Mutex::new(GraniteSpeechEncoderBatcherState::default()),
        }
    }

    pub(super) async fn submit(
        self: &Arc<Self>,
        mut pending: GraniteSpeechEncoderPending,
    ) -> Result<GraniteSpeechEncoderOutcome> {
        let work = WorkUnit::PreSequencePreparation {
            kind: "asr.encoder.granite_speech".into(),
        };
        let binding = pending.contract.adapter_binding()?;
        let stage = binding.stage_for_work(&work)?;
        if stage.name != "asr.encoder.granite_speech" {
            return Err(Error::InvalidInput(
                "Granite Speech loaded contract did not select asr.encoder.granite_speech".into(),
            ));
        }
        let deadline = pending.job.spec.deadline;
        let now = Instant::now();
        let key = GraniteSpeechEncoderQueueKey {
            binding: binding.key_for_stage(stage.id)?,
            audio_token_bucket: pending
                .geometry
                .audio_tokens
                .checked_next_power_of_two()
                .ok_or_else(|| {
                    Error::Overloaded("Granite Speech audio-token bucket overflow".into())
                })?,
            prompt_token_bucket: pending
                .geometry
                .prompt_tokens
                .checked_next_power_of_two()
                .ok_or_else(|| {
                    Error::Overloaded("Granite Speech prompt-token bucket overflow".into())
                })?,
            deadline_budget_bucket: granite_speech_deadline_budget_bucket(deadline, now),
        };
        let max_width = stage.max_batch_size.max(1);
        let formation_delay = stage.max_formation_delay;
        let cancellation = pending.cancellation.clone();
        let (sender, receiver) = oneshot::channel();
        pending.response = Some(sender);

        let mut immediate = None;
        let first;
        {
            let mut state = self.state.lock().await;
            let queue = state.pending.entry(key.clone()).or_default();
            first = queue.is_empty();
            queue.push_back(pending);
            let deadline_pressure = deadline.is_some_and(|deadline| {
                deadline
                    <= Instant::now()
                        .checked_add(formation_delay)
                        .unwrap_or(deadline)
            });
            if queue.len() >= max_width || deadline_pressure {
                immediate = Some(Self::drain(queue, max_width));
                if queue.is_empty() {
                    state.pending.remove(&key);
                }
            }
        }
        if let Some(batch) = immediate {
            self.spawn_batch(batch);
        } else if first {
            let batcher = self.clone();
            tokio::spawn(async move {
                yield_now().await;
                if !formation_delay.is_zero() {
                    let should_wait = {
                        let state = batcher.state.lock().await;
                        state.pending.get(&key).is_some_and(|queue| {
                            !queue.iter().any(|row| {
                                row.job.spec.deadline.is_some_and(|deadline| {
                                    deadline
                                        <= Instant::now()
                                            .checked_add(formation_delay)
                                            .unwrap_or(deadline)
                                })
                            })
                        })
                    };
                    if should_wait {
                        tokio::time::sleep(formation_delay).await;
                    }
                }
                if let Some(batch) = batcher.take_batch(&key, max_width).await {
                    batcher.spawn_batch(batch);
                }
            });
        }

        let mut guard = PreparationCancellationGuard {
            cancellation,
            armed: true,
        };
        let outcome = receiver.await.map_err(|_| {
            Error::InferenceError("Granite Speech encoder batch worker stopped before reply".into())
        })?;
        guard.armed = false;
        Ok(outcome)
    }

    pub(super) fn drain(
        queue: &mut VecDeque<GraniteSpeechEncoderPending>,
        max_width: usize,
    ) -> Vec<GraniteSpeechEncoderPending> {
        queue.drain(..queue.len().min(max_width)).collect()
    }

    pub(super) async fn take_batch(
        &self,
        key: &GraniteSpeechEncoderQueueKey,
        max_width: usize,
    ) -> Option<Vec<GraniteSpeechEncoderPending>> {
        let mut state = self.state.lock().await;
        let queue = state.pending.get_mut(key)?;
        let batch = Self::drain(queue, max_width);
        if queue.is_empty() {
            state.pending.remove(key);
        }
        (!batch.is_empty()).then_some(batch)
    }

    pub(super) fn spawn_batch(self: &Arc<Self>, batch: Vec<GraniteSpeechEncoderPending>) {
        let batcher = self.clone();
        tokio::spawn(async move {
            batcher.run_batch(batch).await;
        });
    }

    pub(super) async fn run_batch(&self, mut batch: Vec<GraniteSpeechEncoderPending>) {
        let now = Instant::now();
        let mut live = Vec::with_capacity(batch.len());
        for mut row in batch.drain(..) {
            let terminal = if row.cancellation.is_cancelled() {
                Some(PreparationRowOutcome::Cancelled)
            } else if row
                .job
                .spec
                .deadline
                .is_some_and(|deadline| deadline <= now)
            {
                Some(PreparationRowOutcome::TimedOut)
            } else {
                None
            };
            if let Some(outcome) = terminal {
                if let Some(response) = row.response.take() {
                    let _ = response.send(outcome);
                }
            } else {
                live.push(row);
            }
        }
        batch = live;
        let Some(first) = batch.first() else {
            return;
        };
        let contract = first.contract.clone();
        let model = first.model.clone();
        let geometries = batch.iter().map(|row| row.geometry).collect::<Vec<_>>();
        let batch_geometry = match model.granite_speech_preparation_batch_geometry(&geometries) {
            Ok(geometry) => geometry,
            Err(error) => {
                Self::fail_batch(batch, error);
                return;
            }
        };
        let work = WorkUnit::PreSequencePreparation {
            kind: "asr.encoder.granite_speech".into(),
        };
        let mut sealed = Vec::with_capacity(batch.len());
        for (index, row) in batch.iter().enumerate() {
            let cost = match model.granite_speech_preparation_row_cost_for_batch(
                index,
                &geometries,
                batch_geometry,
            ) {
                Ok(cost) => cost,
                Err(error) => {
                    Self::fail_batch(batch, error);
                    return;
                }
            };
            match self.coordinator.seal_preparation_row(
                row.job.clone(),
                &row.contract,
                &work,
                cost,
                batch_geometry.materialized_tensor_elements_per_row,
                row.cancellation.clone(),
            ) {
                Ok(seal) => sealed.push(seal),
                Err(error) => {
                    Self::fail_batch(batch, error);
                    return;
                }
            }
        }

        let inputs = batch
            .iter()
            .map(|row| {
                (
                    row.samples.clone(),
                    row.sample_rate,
                    row.language.clone(),
                    row.prompt.clone(),
                    row.geometry,
                    row.retained_host_bytes,
                    row.artifact_host_bytes,
                )
            })
            .collect::<Vec<_>>();
        let senders = batch
            .drain(..)
            .map(|mut row| {
                row.response
                    .take()
                    .expect("queued Granite Speech row has response channel")
            })
            .collect::<Vec<_>>();
        let physical_model = model.clone();
        let result = self
            .coordinator
            .run_loaded_native_preparation_batch(sealed, contract, work, move |live| {
                let selected = live
                    .iter()
                    .map(|index| GraniteSpeechPreparationBatchRow {
                        audio: inputs[*index].0.as_ref(),
                        sample_rate: inputs[*index].1,
                        language: inputs[*index].2.as_deref(),
                        prompt: inputs[*index].3.as_deref(),
                    })
                    .collect::<Vec<_>>();
                let prepared =
                    physical_model.prepare_granite_speech_prompt_artifact_batch(&selected)?;
                if prepared.len() != live.len() {
                    return Err(Error::InferenceError(
                        "Granite Speech preparation batch returned the wrong row count".into(),
                    ));
                }
                Ok(prepared
                    .into_iter()
                    .zip(live.iter().copied())
                    .map(|(artifact, index)| {
                        let geometry = inputs[index].4;
                        let resident_tensor_bytes = artifact.resident_tensor_bytes()?;
                        if artifact.prompt_tokens() != geometry.prompt_tokens
                            || artifact.audio_tokens() != geometry.audio_tokens
                            || artifact.resident_host_bytes() != inputs[index].6
                            || resident_tensor_bytes != geometry.retained_device_bytes
                        {
                            return Err(Error::InferenceError(
                                "Granite Speech batch artifact drifted from admitted geometry"
                                    .into(),
                            ));
                        }
                        Ok(PreparationArtifact {
                            retained: JobResourceObservation {
                                host_bytes: inputs[index].5,
                                accelerator_bytes: resident_tensor_bytes,
                            },
                            value: artifact,
                        })
                    })
                    .collect::<Vec<Result<PreparationArtifact<
                        Arc<GraniteSpeechPreparedPromptArtifact>,
                    >>>>())
            })
            .await;
        match result {
            Ok(outcomes) => {
                for (sender, outcome) in senders.into_iter().zip(outcomes) {
                    let _ = sender.send(outcome);
                }
            }
            Err(error) => {
                let message = error.to_string();
                for sender in senders {
                    let _ = sender.send(PreparationRowOutcome::Failed(Error::InferenceError(
                        message.clone(),
                    )));
                }
            }
        }
    }

    pub(super) fn fail_batch(batch: Vec<GraniteSpeechEncoderPending>, error: Error) {
        let message = error.to_string();
        for row in batch {
            if let Some(response) = row.response {
                let _ = response.send(PreparationRowOutcome::Failed(Error::InferenceError(
                    message.clone(),
                )));
            }
        }
    }
}

impl QwenAsrEncoderBatcher {
    pub(super) fn new(coordinator: Arc<InferenceCoordinator>) -> Self {
        Self {
            coordinator,
            state: Mutex::new(QwenAsrEncoderBatcherState::default()),
        }
    }

    pub(super) async fn submit(
        self: &Arc<Self>,
        pending: QwenAsrEncoderPending,
    ) -> Result<QwenAsrEncoderOutcome> {
        let work = WorkUnit::PreSequencePreparation {
            kind: "asr.encoder.audio".to_string(),
        };
        let binding = pending.contract.adapter_binding()?;
        let stage = binding.stage_for_work(&work)?;
        if stage.name != "asr.encoder.audio" {
            return Err(Error::InvalidInput(
                "Qwen3 ASR loaded contract did not select asr.encoder.audio".to_string(),
            ));
        }
        let key = QwenAsrEncoderQueueKey {
            binding: binding.key_for_stage(stage.id)?,
            // The loaded padded-stage contract allows at most 100% padding.
            // Power-of-two duration buckets keep every admitted row within a
            // factor of two while preserving ragged logical lengths.
            mel_frame_bucket: pending
                .geometry
                .mel_frames
                .checked_next_power_of_two()
                .ok_or_else(|| {
                    Error::Overloaded("Qwen3 ASR mel-frame bucket overflow".to_string())
                })?,
        };
        let max_width = stage.max_batch_size.max(1);
        let formation_delay = stage.max_formation_delay;
        let deadline = pending.job.spec.deadline;
        let cancellation = pending.cancellation.clone();
        let (response, receiver) = oneshot::channel();
        let mut pending = pending;
        pending.response = Some(response);

        let mut immediate = None;
        let first;
        {
            let mut state = self.state.lock().await;
            let queue = state.pending.entry(key.clone()).or_default();
            first = queue.is_empty();
            queue.push_back(pending);
            let deadline_pressure = deadline.is_some_and(|deadline| {
                deadline
                    <= Instant::now()
                        .checked_add(formation_delay)
                        .unwrap_or(deadline)
            });
            if queue.len() >= max_width || deadline_pressure {
                immediate = Some(Self::drain(queue, max_width));
                if queue.is_empty() {
                    state.pending.remove(&key);
                }
            }
        }

        if let Some(batch) = immediate {
            self.spawn_batch(batch);
        } else if first {
            let batcher = self.clone();
            tokio::spawn(async move {
                // A zero-delay stage still yields once so requests already
                // runnable in this executor turn can join without imposing a
                // timer-derived latency floor.
                yield_now().await;
                if !formation_delay.is_zero() {
                    let should_wait = {
                        let state = batcher.state.lock().await;
                        state.pending.get(&key).is_some_and(|queue| {
                            !queue.iter().any(|row| {
                                row.job.spec.deadline.is_some_and(|deadline| {
                                    deadline
                                        <= Instant::now()
                                            .checked_add(formation_delay)
                                            .unwrap_or(deadline)
                                })
                            })
                        })
                    };
                    if should_wait {
                        tokio::time::sleep(formation_delay).await;
                    }
                }
                if let Some(batch) = batcher.take_batch(&key, max_width).await {
                    batcher.spawn_batch(batch);
                }
            });
        }

        let mut guard = PreparationCancellationGuard {
            cancellation,
            armed: true,
        };
        let outcome = receiver.await.map_err(|_| {
            Error::InferenceError("Qwen3 ASR encoder batch worker stopped before reply".to_string())
        })?;
        guard.armed = false;
        Ok(outcome)
    }

    pub(super) fn drain(
        queue: &mut VecDeque<QwenAsrEncoderPending>,
        max_width: usize,
    ) -> Vec<QwenAsrEncoderPending> {
        let width = queue.len().min(max_width);
        queue.drain(..width).collect()
    }

    pub(super) async fn take_batch(
        &self,
        key: &QwenAsrEncoderQueueKey,
        max_width: usize,
    ) -> Option<Vec<QwenAsrEncoderPending>> {
        let mut state = self.state.lock().await;
        let queue = state.pending.get_mut(key)?;
        let batch = Self::drain(queue, max_width);
        if queue.is_empty() {
            state.pending.remove(key);
        }
        (!batch.is_empty()).then_some(batch)
    }

    pub(super) fn spawn_batch(self: &Arc<Self>, batch: Vec<QwenAsrEncoderPending>) {
        let batcher = self.clone();
        tokio::spawn(async move {
            batcher.run_batch(batch).await;
        });
    }

    pub(super) async fn run_batch(&self, mut batch: Vec<QwenAsrEncoderPending>) {
        let Some(first) = batch.first() else {
            return;
        };
        let contract = first.contract.clone();
        let model = first.model.clone();
        let geometries = batch.iter().map(|row| row.geometry).collect::<Vec<_>>();
        let batch_geometry = match model.audio_preparation_batch_geometry(&geometries) {
            Ok(geometry) => geometry,
            Err(error) => {
                Self::fail_batch(batch, error);
                return;
            }
        };
        let materialized_per_row = match batch_geometry
            .padded_mel_elements_per_row
            .checked_add(batch_geometry.padded_output_elements_per_row)
        {
            Some(value) => value,
            None => {
                Self::fail_batch(
                    batch,
                    Error::Overloaded("Qwen3 ASR padded work accounting overflow".to_string()),
                );
                return;
            }
        };
        let work = WorkUnit::PreSequencePreparation {
            kind: "asr.encoder.audio".to_string(),
        };
        let mut sealed = Vec::with_capacity(batch.len());
        for (index, row) in batch.iter().enumerate() {
            let cost = match model.audio_preparation_row_cost_for_batch(
                index,
                &geometries,
                &batch_geometry,
            ) {
                Ok(cost) => cost,
                Err(error) => {
                    Self::fail_batch(batch, error);
                    return;
                }
            };
            match self.coordinator.seal_preparation_row(
                row.job.clone(),
                &row.contract,
                &work,
                cost,
                materialized_per_row,
                row.cancellation.clone(),
            ) {
                Ok(seal) => sealed.push(seal),
                Err(error) => {
                    Self::fail_batch(batch, error);
                    return;
                }
            }
        }
        // The sealed rows now own the only JobLease clones used by the
        // physical transaction. Drop the queue-owned originals before the
        // runner converts successful jobs into unique admission bridges.
        let samples = batch
            .iter()
            .map(|row| {
                (
                    row.samples.clone(),
                    row.sample_rate,
                    row.retained_host_bytes,
                )
            })
            .collect::<Vec<_>>();
        let senders = batch
            .drain(..)
            .map(|mut row| {
                row.response
                    .take()
                    .expect("queued Qwen3 ASR row has a response channel")
            })
            .collect::<Vec<_>>();
        let physical_model = model.clone();
        let result = self
            .coordinator
            .run_loaded_native_preparation_batch(sealed, contract, work, move |live| {
                let selected = live
                    .iter()
                    .map(|index| Qwen3AsrAudioBatchRow {
                        audio: samples[*index].0.as_ref(),
                        sample_rate: samples[*index].1,
                    })
                    .collect::<Vec<_>>();
                let prepared = physical_model.prepare_qwen3_audio_tower_batch(&selected)?;
                Ok(prepared
                    .into_iter()
                    .zip(live.iter())
                    .map(|(artifact, index)| {
                        Ok(PreparationArtifact {
                            retained: JobResourceObservation {
                                host_bytes: samples[*index].2,
                                accelerator_bytes: artifact.resident_tensor_bytes()?,
                            },
                            value: artifact,
                        })
                    })
                    .collect::<Vec<Result<PreparationArtifact<Qwen3AsrPreparedAudio>>>>())
            })
            .await;
        match result {
            Ok(outcomes) => {
                for (sender, outcome) in senders.into_iter().zip(outcomes) {
                    let _ = sender.send(outcome);
                }
            }
            Err(error) => {
                let message = error.to_string();
                for sender in senders {
                    let _ = sender.send(PreparationRowOutcome::Failed(Error::InferenceError(
                        message.clone(),
                    )));
                }
            }
        }
    }

    pub(super) fn fail_batch(batch: Vec<QwenAsrEncoderPending>, error: Error) {
        let message = error.to_string();
        for row in batch {
            if let Some(response) = row.response {
                let _ = response.send(PreparationRowOutcome::Failed(Error::InferenceError(
                    message.clone(),
                )));
            }
        }
    }
}
