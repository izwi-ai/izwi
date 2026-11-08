//! Cross-request ASR encoder batchers: Qwen3-ASR, Whisper, Granite Speech
//! 4.1 and 5.0, VibeVoice, Parakeet and Nemotron encoder work queued by
//! request preparation and run as one batch.

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct QwenAsrEncoderQueueKey {
    pub(super) binding: AdapterBindingKey,
    pub(super) mel_frame_bucket: usize,
}

pub(super) struct QwenAsrEncoderBatcher {
    pub(super) coordinator: Arc<InferenceCoordinator>,
    pub(super) formation: EncoderBatchFormation<QwenAsrEncoderQueueKey, QwenAsrEncoderPending>,
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct WhisperEncoderQueueKey {
    pub(super) binding: AdapterBindingKey,
    pub(super) mel_frame_bucket: usize,
}

pub(super) struct WhisperEncoderBatcher {
    pub(super) coordinator: Arc<InferenceCoordinator>,
    pub(super) formation: EncoderBatchFormation<WhisperEncoderQueueKey, WhisperEncoderPending>,
}

pub(super) type VibeVoiceEncoderOutcome = PreparationRowOutcome<VibeVoiceAsrPreparedArtifact>;

pub(super) type ParakeetEncoderOutcome =
    PreparationRowOutcome<Arc<ParakeetPreparedEncoderArtifact>>;

pub(super) struct ParakeetEncoderPending {
    pub(super) job: JobLease,
    pub(super) contract: LoadedExecutionContract,
    pub(super) model: AsrModelLease,
    /// The model's catalog name, for the batch-width metric.
    pub(super) model_label: &'static str,
    pub(super) backend: BackendKind,
    /// The host log-mel, dropped once the encode has read it.
    pub(super) features: ParakeetHostFeatures,
    pub(super) language: Option<String>,
    /// Host bytes the request keeps after the encode: its input, not the
    /// features.
    pub(super) retained_host_bytes: u64,
    /// The admitted ceiling of the row's encoder output.
    pub(super) retained_device_ceiling: u64,
    pub(super) cancellation: PreparationCancellation,
    pub(super) response: Option<oneshot::Sender<ParakeetEncoderOutcome>>,
}

pub(super) type NemotronEncoderOutcome =
    PreparationRowOutcome<Arc<NemotronPreparedEncoderArtifact>>;

pub(super) struct NemotronEncoderPending {
    pub(super) job: JobLease,
    pub(super) contract: LoadedExecutionContract,
    pub(super) model: AsrModelLease,
    /// The model's catalog name, for the batch-width metric.
    pub(super) model_label: &'static str,
    pub(super) backend: BackendKind,
    /// The host log-mel and prompt, dropped once the encode has read them.
    pub(super) features: NemotronHostFeatures,
    /// Host bytes the request keeps after the encode: its input, not the
    /// features.
    pub(super) retained_host_bytes: u64,
    /// The admitted ceiling of the row's encoder projection.
    pub(super) retained_device_ceiling: u64,
    pub(super) cancellation: PreparationCancellation,
    pub(super) response: Option<oneshot::Sender<NemotronEncoderOutcome>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct NemotronEncoderQueueKey {
    pub(super) binding: AdapterBindingKey,
    /// Encoder frames rounded up to a power of two, so padding a row to
    /// its batch's longest row adds less than its own length.
    pub(super) encoded_len_bucket: usize,
}

pub(super) struct NemotronEncoderBatcher {
    pub(super) coordinator: Arc<InferenceCoordinator>,
    pub(super) formation: EncoderBatchFormation<NemotronEncoderQueueKey, NemotronEncoderPending>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct ParakeetEncoderQueueKey {
    pub(super) binding: AdapterBindingKey,
    /// Encoder frames rounded up to a power of two, so padding a row to
    /// its batch's longest row adds less than its own length.
    pub(super) encoded_len_bucket: usize,
}

pub(super) struct ParakeetEncoderBatcher {
    pub(super) coordinator: Arc<InferenceCoordinator>,
    pub(super) formation: EncoderBatchFormation<ParakeetEncoderQueueKey, ParakeetEncoderPending>,
}

/// Forms one encoder's cross-request batches with at most one batch in
/// flight per key. Rows that arrive while their key's batch runs wait, and
/// form the key's next batch when it completes, so requests that arrive
/// staggered still batch under load. Every batched encoder here forms its
/// batches this way; a queue drained one yield after its first row (the
/// older shape) runs each row as a batch of one behind a busy device.
pub(super) struct EncoderBatchFormation<K, P> {
    /// A key's queue exists exactly while a runner drains it.
    queues: std::sync::Mutex<HashMap<K, VecDeque<P>>>,
}

impl<K: Eq + std::hash::Hash, P> EncoderBatchFormation<K, P> {
    pub(super) fn new() -> Self {
        Self {
            queues: std::sync::Mutex::new(HashMap::new()),
        }
    }

    fn queues(&self) -> std::sync::MutexGuard<'_, HashMap<K, VecDeque<P>>> {
        self.queues
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Queues `row` under `key`. True when the key had no runner: the
    /// caller then starts one with [`Self::run`].
    pub(super) fn push(&self, key: K, row: P) -> bool {
        let mut queues = self.queues();
        match queues.get_mut(&key) {
            Some(queue) => {
                queue.push_back(row);
                false
            }
            None => {
                queues.insert(key, VecDeque::from([row]));
                true
            }
        }
    }

    /// Runs `key`'s batches of at most `max_width` rows one after another
    /// until its queue is empty. Rows already runnable in this executor
    /// turn join the first batch.
    pub(super) async fn run<F, Fut>(&self, key: &K, max_width: usize, mut run_batch: F)
    where
        F: FnMut(Vec<P>) -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        /// A runner that unwinds drops its key's waiting rows, whose
        /// submitters then fail, and frees the key for a new runner.
        struct Abandon<'a, K: Eq + std::hash::Hash, P> {
            formation: &'a EncoderBatchFormation<K, P>,
            key: &'a K,
            armed: bool,
        }

        impl<K: Eq + std::hash::Hash, P> Drop for Abandon<'_, K, P> {
            fn drop(&mut self) {
                if self.armed {
                    let rows = self.formation.queues().remove(self.key);
                    drop(rows);
                }
            }
        }

        let mut abandon = Abandon {
            formation: self,
            key,
            armed: true,
        };
        yield_now().await;
        while let Some(batch) = self.next_batch(key, max_width) {
            run_batch(batch).await;
        }
        abandon.armed = false;
    }

    /// The key's next batch, or `None` after retiring its empty queue.
    fn next_batch(&self, key: &K, max_width: usize) -> Option<Vec<P>> {
        let mut queues = self.queues();
        let queue = queues.get_mut(key)?;
        if queue.is_empty() {
            queues.remove(key);
            return None;
        }
        let width = queue.len().min(max_width.max(1));
        Some(queue.drain(..width).collect())
    }
}

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct GraniteSpeechEncoderQueueKey {
    pub(super) binding: AdapterBindingKey,
    pub(super) audio_token_bucket: usize,
    pub(super) prompt_token_bucket: usize,
    pub(super) deadline_budget_bucket: Option<u64>,
}

pub(super) struct GraniteSpeechEncoderBatcher {
    pub(super) coordinator: Arc<InferenceCoordinator>,
    pub(super) formation:
        EncoderBatchFormation<GraniteSpeechEncoderQueueKey, GraniteSpeechEncoderPending>,
}

pub(super) type GraniteSpeech5EncoderOutcome = PreparationRowOutcome<Arc<GraniteSpeech5PreparedPromptArtifact>>;

pub(super) struct GraniteSpeech5EncoderPending {
    pub(super) job: JobLease,
    pub(super) contract: LoadedExecutionContract,
    pub(super) model: AsrModelLease,
    pub(super) samples: Arc<[f32]>,
    pub(super) sample_rate: u32,
    pub(super) request: GraniteSpeech5RequestOptions,
    pub(super) geometry: GraniteSpeech5PreparedGeometry,
    pub(super) retained_host_bytes: u64,
    pub(super) artifact_host_bytes: u64,
    pub(super) cancellation: PreparationCancellation,
    pub(super) response: Option<oneshot::Sender<GraniteSpeech5EncoderOutcome>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct GraniteSpeech5EncoderQueueKey {
    pub(super) binding: AdapterBindingKey,
    pub(super) audio_token_bucket: usize,
    pub(super) prompt_token_bucket: usize,
    pub(super) deadline_budget_bucket: Option<u64>,
}

pub(super) struct GraniteSpeech5EncoderBatcher {
    pub(super) coordinator: Arc<InferenceCoordinator>,
    pub(super) formation:
        EncoderBatchFormation<GraniteSpeech5EncoderQueueKey, GraniteSpeech5EncoderPending>,
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

/// Whether concurrent single-window Parakeet requests encode as one batch
/// (`IZWI_PARAKEET_ENCODER_BATCHING`, on by default); off, every request
/// encodes alone, as before batching.
fn parakeet_encoder_batching_enabled() -> bool {
    static ENABLED: crate::env::ReadOnce<bool> =
        crate::env::ReadOnce::new(|| crate::env::flag_or("IZWI_PARAKEET_ENCODER_BATCHING", true));
    ENABLED.get()
}

impl ParakeetEncoderBatcher {
    pub(super) fn new(coordinator: Arc<InferenceCoordinator>) -> Self {
        Self {
            coordinator,
            formation: EncoderBatchFormation::new(),
        }
    }

    /// Encodes one request's features: a single-window row waits for its
    /// length bucket's running batch and joins the next one; a longer row,
    /// or any row with batching off, runs as a batch of its own.
    pub(super) async fn submit(
        self: &Arc<Self>,
        mut pending: ParakeetEncoderPending,
    ) -> Result<ParakeetEncoderOutcome> {
        let work = WorkUnit::PreSequencePreparation {
            kind: "asr.encoder.parakeet".into(),
        };
        let binding = pending.contract.adapter_binding()?;
        let stage = binding.stage_for_work(&work)?;
        if stage.name != "asr.encoder.parakeet" {
            return Err(Error::InvalidInput(
                "Parakeet loaded contract did not select asr.encoder.parakeet".into(),
            ));
        }
        let max_width = stage.max_batch_size.max(1);
        let key = (max_width > 1
            && pending.features.single_window()
            && parakeet_encoder_batching_enabled())
        .then(|| {
            Ok::<_, Error>(ParakeetEncoderQueueKey {
                binding: binding.key_for_stage(stage.id)?,
                encoded_len_bucket: pending
                    .features
                    .encoded_len()
                    .checked_next_power_of_two()
                    .ok_or_else(|| Error::Overloaded("Parakeet length bucket overflow".into()))?,
            })
        })
        .transpose()?;
        let cancellation = pending.cancellation.clone();
        let (sender, receiver) = oneshot::channel();
        pending.response = Some(sender);
        let batcher = self.clone();
        match key {
            Some(key) => {
                if self.formation.push(key.clone(), pending) {
                    tokio::spawn(async move {
                        batcher
                            .formation
                            .run(&key, max_width, |batch| batcher.run_batch(batch))
                            .await;
                    });
                }
            }
            None => {
                tokio::spawn(async move { batcher.run_batch(vec![pending]).await });
            }
        }
        let mut guard = PreparationCancellationGuard {
            cancellation,
            armed: true,
        };
        let outcome = receiver.await.map_err(|_| {
            Error::InferenceError("Parakeet encoder batch worker stopped before reply".into())
        })?;
        guard.armed = false;
        Ok(outcome)
    }

    /// Runs one formed batch: rows cancelled or expired while they waited
    /// answer at once, and the rest are priced at the batch's padded
    /// length and encoded in one physical call.
    async fn run_batch(&self, rows: Vec<ParakeetEncoderPending>) {
        let now = Instant::now();
        let rows = rows
            .into_iter()
            .filter_map(|mut row| {
                let terminal = if row.cancellation.is_cancelled() {
                    PreparationRowOutcome::Cancelled
                } else if row
                    .job
                    .spec
                    .deadline
                    .is_some_and(|deadline| deadline <= now)
                {
                    PreparationRowOutcome::TimedOut
                } else {
                    return Some(row);
                };
                if let Some(response) = row.response.take() {
                    let _ = response.send(terminal);
                }
                None
            })
            .collect::<Vec<_>>();
        let Some(first) = rows.first() else {
            return;
        };
        let contract = first.contract.clone();
        let model = first.model.clone();
        let model_label = first.model_label;
        let width = rows.len();
        let padded_frames = rows
            .iter()
            .map(|row| row.features.window_frames())
            .max()
            .unwrap_or(0);
        let work = WorkUnit::PreSequencePreparation {
            kind: "asr.encoder.parakeet".into(),
        };
        // The host stage recorded the request bytes plus its log-mel
        // features, which the encode consumes; each row's commit records
        // only what its artifact keeps (request bytes and the encoder
        // output). Restore the features' claim to pending first: usage may
        // only rise through an observation, and the reservation still covers
        // the features until the batch ends.
        for row in &rows {
            if let Err(error) = row
                .job
                .prepare_materialized_release(JobResourceObservation::host(row.retained_host_bytes))
            {
                Self::fail(rows, &error);
                return;
            }
        }
        let mut sealed = Vec::with_capacity(width);
        for row in &rows {
            let seal =
                parakeet_preparation_row_cost(&row.features, padded_frames, width, row.backend)
                    .and_then(|(cost, materialized)| {
                        self.coordinator.seal_preparation_row(
                            row.job.clone(),
                            &row.contract,
                            &work,
                            cost,
                            materialized,
                            row.cancellation.clone(),
                        )
                    });
            match seal {
                Ok(seal) => sealed.push(seal),
                Err(error) => {
                    Self::fail(rows, &error);
                    return;
                }
            }
        }
        // The sealed rows now own the only job leases the physical call
        // uses; the queued originals drop here, before committed rows
        // become admission bridges.
        let mut senders = Vec::with_capacity(width);
        let mut inputs = Vec::with_capacity(width);
        for mut row in rows {
            senders.push(
                row.response
                    .take()
                    .expect("queued Parakeet row has a response channel"),
            );
            inputs.push(ParakeetEncoderInput {
                request_id: row.job.spec.request_id.clone(),
                deadline: row.job.spec.deadline,
                features: row.features,
                language: row.language,
                cancellation: row.cancellation,
                retained_host_bytes: row.retained_host_bytes,
                retained_device_ceiling: row.retained_device_ceiling,
            });
        }
        let result = self
            .coordinator
            .run_loaded_native_preparation_batch(sealed, contract, work, move |live| {
                crate::engine::metrics::record_asr_preparation_batch_rows(model_label, live.len());
                // Checked before the encode and between a long row's
                // windows, so a disconnected or expired request fails
                // alone and releases the device within one window.
                let checkpoints = live
                    .iter()
                    .map(|index| {
                        let input = &inputs[*index];
                        move || {
                            if input.cancellation.is_cancelled() {
                                return Err(Error::Cancelled(input.request_id.clone()));
                            }
                            if input
                                .deadline
                                .is_some_and(|deadline| Instant::now() >= deadline)
                            {
                                return Err(Error::Timeout(input.request_id.clone()));
                            }
                            Ok(())
                        }
                    })
                    .collect::<Vec<_>>();
                let rows = live
                    .iter()
                    .zip(&checkpoints)
                    .map(|(index, checkpoint)| ParakeetEncoderBatchRow {
                        features: &inputs[*index].features,
                        language: inputs[*index].language.as_deref(),
                        checkpoint,
                    })
                    .collect::<Vec<_>>();
                let prepared = model.prepare_parakeet_encoder_batch(&rows)?;
                if prepared.len() != live.len() {
                    return Err(Error::InferenceError(
                        "Parakeet encoder batch returned the wrong row count".into(),
                    ));
                }
                Ok(prepared
                    .into_iter()
                    .zip(live)
                    .map(|(artifact, index)| {
                        let artifact = Arc::new(artifact?);
                        let accelerator_bytes = artifact.resident_tensor_bytes()?;
                        let ceiling = inputs[*index].retained_device_ceiling;
                        if accelerator_bytes > ceiling {
                            return Err(Error::InferenceError(format!(
                                "Parakeet encoder artifact exceeded its admitted ceiling: {accelerator_bytes} > {ceiling}"
                            )));
                        }
                        Ok(PreparationArtifact {
                            retained: JobResourceObservation {
                                host_bytes: inputs[*index].retained_host_bytes,
                                accelerator_bytes,
                            },
                            value: artifact,
                        })
                    })
                    .collect::<Vec<_>>())
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
                    let _ = sender.send(PreparationRowOutcome::Failed(error.replicate()));
                }
            }
        }
    }

    fn fail(rows: Vec<ParakeetEncoderPending>, error: &Error) {
        for row in rows {
            if let Some(response) = row.response {
                let _ = response.send(PreparationRowOutcome::Failed(error.replicate()));
            }
        }
    }
}

/// A sealed Parakeet row's input to the physical call.
struct ParakeetEncoderInput {
    request_id: String,
    deadline: Option<Instant>,
    features: ParakeetHostFeatures,
    language: Option<String>,
    cancellation: PreparationCancellation,
    retained_host_bytes: u64,
    retained_device_ceiling: u64,
}

/// Prices one Parakeet row in an encoder batch of `rows` rows padded to
/// `padded_frames` encoder frames: its input samples as work, its padded
/// encoder output as materialized tensor work, and its share of the
/// batch's workspace.
fn parakeet_preparation_row_cost(
    features: &ParakeetHostFeatures,
    padded_frames: usize,
    rows: usize,
    backend: BackendKind,
) -> Result<(crate::engine::WorkCost, u64)> {
    use crate::models::architectures::parakeet::asr::{
        parakeet_encode_row_bytes, parakeet_encode_row_elements,
    };
    let bytes = parakeet_encode_row_bytes(features, padded_frames, rows, backend)?;
    let (useful, materialized) =
        parakeet_encode_row_elements(features, padded_frames, rows, backend)?;
    let samples = u64::try_from(features.input_samples())
        .map_err(|_| Error::Overloaded("Parakeet sample work exceeds u64".into()))?;
    let cost = crate::engine::WorkCost::with_workspace(
        samples,
        useful,
        retained_artifact_resources(backend, bytes.host_bytes, bytes.device_bytes()?)?,
    );
    Ok((cost, materialized))
}

/// Whether concurrent Nemotron clips encode as one batch
/// (`IZWI_NEMOTRON_ASR_ENCODER_BATCHING`, on by default); off, every request
/// encodes alone, as before batching.
fn nemotron_encoder_batching_enabled() -> bool {
    static ENABLED: crate::env::ReadOnce<bool> = crate::env::ReadOnce::new(|| {
        crate::env::flag_or("IZWI_NEMOTRON_ASR_ENCODER_BATCHING", true)
    });
    ENABLED.get()
}

impl NemotronEncoderBatcher {
    pub(super) fn new(coordinator: Arc<InferenceCoordinator>) -> Self {
        Self {
            coordinator,
            formation: EncoderBatchFormation::new(),
        }
    }

    /// The queue a row joins: its binding and power-of-two length bucket,
    /// or `None` when it encodes alone (batching off, or a width-1 stage).
    fn queue_key(
        binding: &crate::engine::ExecutionAdapterBinding,
        stage: &crate::engine::StageDescriptor,
        features: &NemotronHostFeatures,
        batching: bool,
    ) -> Result<Option<NemotronEncoderQueueKey>> {
        if stage.max_batch_size <= 1 || !batching {
            return Ok(None);
        }
        Ok(Some(NemotronEncoderQueueKey {
            binding: binding.key_for_stage(stage.id)?,
            encoded_len_bucket: features
                .encoded_len()
                .checked_next_power_of_two()
                .ok_or_else(|| Error::Overloaded("Nemotron length bucket overflow".into()))?,
        }))
    }

    /// Encodes one request's features: a row waits for its length bucket's
    /// running batch and joins the next one; with batching off, it runs as
    /// a batch of its own.
    pub(super) async fn submit(
        self: &Arc<Self>,
        mut pending: NemotronEncoderPending,
    ) -> Result<NemotronEncoderOutcome> {
        let work = WorkUnit::PreSequencePreparation {
            kind: NEMOTRON_ENCODER_STAGE.into(),
        };
        let binding = pending.contract.adapter_binding()?;
        let stage = binding.stage_for_work(&work)?;
        if stage.name != NEMOTRON_ENCODER_STAGE {
            return Err(Error::InvalidInput(format!(
                "Nemotron loaded contract did not select {NEMOTRON_ENCODER_STAGE}"
            )));
        }
        let max_width = stage.max_batch_size.max(1);
        let key = Self::queue_key(
            &binding,
            stage,
            &pending.features,
            nemotron_encoder_batching_enabled(),
        )?;
        let cancellation = pending.cancellation.clone();
        let (sender, receiver) = oneshot::channel();
        pending.response = Some(sender);
        let batcher = self.clone();
        match key {
            Some(key) => {
                if self.formation.push(key.clone(), pending) {
                    tokio::spawn(async move {
                        batcher
                            .formation
                            .run(&key, max_width, |batch| batcher.run_batch(batch))
                            .await;
                    });
                }
            }
            None => {
                tokio::spawn(async move { batcher.run_batch(vec![pending]).await });
            }
        }
        let mut guard = PreparationCancellationGuard {
            cancellation,
            armed: true,
        };
        let outcome = receiver.await.map_err(|_| {
            Error::InferenceError("Nemotron encoder batch worker stopped before reply".into())
        })?;
        guard.armed = false;
        Ok(outcome)
    }

    /// Runs one formed batch: rows cancelled or expired while they waited
    /// answer at once, and the rest are priced at the batch's padded
    /// length and encoded in one physical call.
    async fn run_batch(&self, rows: Vec<NemotronEncoderPending>) {
        let now = Instant::now();
        let rows = rows
            .into_iter()
            .filter_map(|mut row| {
                let terminal = if row.cancellation.is_cancelled() {
                    PreparationRowOutcome::Cancelled
                } else if row
                    .job
                    .spec
                    .deadline
                    .is_some_and(|deadline| deadline <= now)
                {
                    PreparationRowOutcome::TimedOut
                } else {
                    return Some(row);
                };
                if let Some(response) = row.response.take() {
                    let _ = response.send(terminal);
                }
                None
            })
            .collect::<Vec<_>>();
        let Some(first) = rows.first() else {
            return;
        };
        let contract = first.contract.clone();
        let model = first.model.clone();
        let model_label = first.model_label;
        let width = rows.len();
        let padded_frames = rows
            .iter()
            .map(|row| row.features.encoded_len())
            .max()
            .unwrap_or(0);
        let work = WorkUnit::PreSequencePreparation {
            kind: NEMOTRON_ENCODER_STAGE.into(),
        };
        // The host stage recorded the request bytes plus its log-mel
        // features, which the encode consumes; each row's commit records
        // only what its artifact keeps (request bytes and the encoder
        // output). Restore the features' claim to pending first: usage may
        // only rise through an observation, and the reservation still covers
        // the features until the batch ends.
        for row in &rows {
            if let Err(error) = row
                .job
                .prepare_materialized_release(JobResourceObservation::host(row.retained_host_bytes))
            {
                Self::fail(rows, &error);
                return;
            }
        }
        let mut sealed = Vec::with_capacity(width);
        for row in &rows {
            let seal =
                nemotron_preparation_row_cost(&row.features, padded_frames, width, row.backend)
                    .and_then(|(cost, materialized)| {
                        self.coordinator.seal_preparation_row(
                            row.job.clone(),
                            &row.contract,
                            &work,
                            cost,
                            materialized,
                            row.cancellation.clone(),
                        )
                    });
            match seal {
                Ok(seal) => sealed.push(seal),
                Err(error) => {
                    Self::fail(rows, &error);
                    return;
                }
            }
        }
        // The sealed rows now own the only job leases the physical call
        // uses; the queued originals drop here, before committed rows
        // become admission bridges.
        let mut senders = Vec::with_capacity(width);
        let mut inputs = Vec::with_capacity(width);
        for mut row in rows {
            senders.push(
                row.response
                    .take()
                    .expect("queued Nemotron row has a response channel"),
            );
            inputs.push(NemotronEncoderInput {
                request_id: row.job.spec.request_id.clone(),
                deadline: row.job.spec.deadline,
                features: row.features,
                cancellation: row.cancellation,
                retained_host_bytes: row.retained_host_bytes,
                retained_device_ceiling: row.retained_device_ceiling,
            });
        }
        let result = self
            .coordinator
            .run_loaded_native_preparation_batch(sealed, contract, work, move |live| {
                crate::engine::metrics::record_asr_preparation_batch_rows(model_label, live.len());
                // Checked before the encode, so a disconnected or expired
                // request fails alone.
                let checkpoints = live
                    .iter()
                    .map(|index| {
                        let input = &inputs[*index];
                        move || {
                            if input.cancellation.is_cancelled() {
                                return Err(Error::Cancelled(input.request_id.clone()));
                            }
                            if input
                                .deadline
                                .is_some_and(|deadline| Instant::now() >= deadline)
                            {
                                return Err(Error::Timeout(input.request_id.clone()));
                            }
                            Ok(())
                        }
                    })
                    .collect::<Vec<_>>();
                let rows = live
                    .iter()
                    .zip(&checkpoints)
                    .map(|(index, checkpoint)| NemotronEncoderBatchRow {
                        features: &inputs[*index].features,
                        checkpoint,
                    })
                    .collect::<Vec<_>>();
                let prepared = model.prepare_nemotron_encoder_batch(&rows)?;
                if prepared.len() != live.len() {
                    return Err(Error::InferenceError(
                        "Nemotron encoder batch returned the wrong row count".into(),
                    ));
                }
                Ok(prepared
                    .into_iter()
                    .zip(live)
                    .map(|(artifact, index)| {
                        let artifact = Arc::new(artifact?);
                        let accelerator_bytes = artifact.resident_tensor_bytes()?;
                        let ceiling = inputs[*index].retained_device_ceiling;
                        if accelerator_bytes > ceiling {
                            return Err(Error::InferenceError(format!(
                                "Nemotron encoder artifact exceeded its admitted ceiling: {accelerator_bytes} > {ceiling}"
                            )));
                        }
                        Ok(PreparationArtifact {
                            retained: JobResourceObservation {
                                host_bytes: inputs[*index].retained_host_bytes,
                                accelerator_bytes,
                            },
                            value: artifact,
                        })
                    })
                    .collect::<Vec<_>>())
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
                    let _ = sender.send(PreparationRowOutcome::Failed(error.replicate()));
                }
            }
        }
    }

    fn fail(rows: Vec<NemotronEncoderPending>, error: &Error) {
        for row in rows {
            if let Some(response) = row.response {
                let _ = response.send(PreparationRowOutcome::Failed(error.replicate()));
            }
        }
    }
}

/// A sealed Nemotron row's input to the physical call.
struct NemotronEncoderInput {
    request_id: String,
    deadline: Option<Instant>,
    features: NemotronHostFeatures,
    cancellation: PreparationCancellation,
    retained_host_bytes: u64,
    retained_device_ceiling: u64,
}

/// Prices one Nemotron row in an encoder batch of `rows` rows padded to
/// `padded_frames` encoder frames: its input samples as work, its padded
/// encoder output as materialized tensor work, its share of the batch's
/// device peak and its retained projection, and on the host the feature
/// copy its upload reads.
fn nemotron_preparation_row_cost(
    features: &NemotronHostFeatures,
    padded_frames: usize,
    rows: usize,
    backend: BackendKind,
) -> Result<(crate::engine::WorkCost, u64)> {
    use crate::models::architectures::nemotron::asr::{
        nemotron_encoder_batch_row_bytes, nemotron_encoder_projection_bytes,
        nemotron_encoder_row_elements,
    };
    let metal = backend == BackendKind::Metal;
    let encoded_len = features.encoded_len();
    let device_bytes = nemotron_encoder_batch_row_bytes(padded_frames, rows, metal)?
        .checked_add(nemotron_encoder_projection_bytes(encoded_len, metal)?)
        .ok_or_else(|| Error::Overloaded("Nemotron preparation bytes overflowed".into()))?;
    let (useful, materialized) = nemotron_encoder_row_elements(encoded_len, padded_frames)?;
    let samples = u64::try_from(features.input_samples())
        .map_err(|_| Error::Overloaded("Nemotron sample work exceeds u64".into()))?;
    let cost = crate::engine::WorkCost::with_workspace(
        samples,
        useful,
        retained_artifact_resources(backend, features.host_bytes()?, device_bytes)?,
    );
    Ok((cost, materialized))
}

impl WhisperEncoderBatcher {
    pub(super) fn new(coordinator: Arc<InferenceCoordinator>) -> Self {
        Self {
            coordinator,
            formation: EncoderBatchFormation::new(),
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
        let cancellation = pending.cancellation.clone();
        let (sender, receiver) = oneshot::channel();
        pending.response = Some(sender);
        // A row waits for its bucket's running batch and joins the next one,
        // so requests that arrive while the device is busy still batch.
        if self.formation.push(key.clone(), pending) {
            let batcher = self.clone();
            tokio::spawn(async move {
                batcher
                    .formation
                    .run(&key, max_width, |batch| batcher.run_batch(batch))
                    .await;
            });
        }
        let mut guard = PreparationCancellationGuard {
            cancellation,
            armed: true,
        };
        let outcome = receiver.await.map_err(|_| {
            Error::InferenceError("Whisper encoder batch worker stopped before reply".into())
        })?;
        guard.armed = false;
        Ok(outcome)
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
            formation: EncoderBatchFormation::new(),
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
        let cancellation = pending.cancellation.clone();
        let (sender, receiver) = oneshot::channel();
        pending.response = Some(sender);
        // A row waits for its bucket's running batch and joins the next one,
        // so requests that arrive while the device is busy still batch.
        if self.formation.push(key.clone(), pending) {
            let batcher = self.clone();
            tokio::spawn(async move {
                batcher
                    .formation
                    .run(&key, max_width, |batch| batcher.run_batch(batch))
                    .await;
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

impl GraniteSpeech5EncoderBatcher {
    pub(super) fn new(coordinator: Arc<InferenceCoordinator>) -> Self {
        Self {
            coordinator,
            formation: EncoderBatchFormation::new(),
        }
    }

    pub(super) async fn submit(
        self: &Arc<Self>,
        mut pending: GraniteSpeech5EncoderPending,
    ) -> Result<GraniteSpeech5EncoderOutcome> {
        let work = WorkUnit::PreSequencePreparation {
            kind: "asr.encoder.granite_speech5".into(),
        };
        let binding = pending.contract.adapter_binding()?;
        let stage = binding.stage_for_work(&work)?;
        if stage.name != "asr.encoder.granite_speech5" {
            return Err(Error::InvalidInput(
                "Granite Speech 5.0 loaded contract did not select asr.encoder.granite_speech5".into(),
            ));
        }
        let deadline = pending.job.spec.deadline;
        let now = Instant::now();
        let key = GraniteSpeech5EncoderQueueKey {
            binding: binding.key_for_stage(stage.id)?,
            audio_token_bucket: pending
                .geometry
                .audio_tokens
                .checked_next_power_of_two()
                .ok_or_else(|| {
                    Error::Overloaded("Granite Speech 5.0 audio-token bucket overflow".into())
                })?,
            prompt_token_bucket: pending
                .geometry
                .prompt_tokens
                .checked_next_power_of_two()
                .ok_or_else(|| {
                    Error::Overloaded("Granite Speech 5.0 prompt-token bucket overflow".into())
                })?,
            deadline_budget_bucket: granite_speech_deadline_budget_bucket(deadline, now),
        };
        let max_width = stage.max_batch_size.max(1);
        let cancellation = pending.cancellation.clone();
        let (sender, receiver) = oneshot::channel();
        pending.response = Some(sender);
        // A row waits for its bucket's running batch and joins the next one,
        // so requests that arrive while the device is busy still batch.
        if self.formation.push(key.clone(), pending) {
            let batcher = self.clone();
            tokio::spawn(async move {
                batcher
                    .formation
                    .run(&key, max_width, |batch| batcher.run_batch(batch))
                    .await;
            });
        }

        let mut guard = PreparationCancellationGuard {
            cancellation,
            armed: true,
        };
        let outcome = receiver.await.map_err(|_| {
            Error::InferenceError("Granite Speech 5.0 encoder batch worker stopped before reply".into())
        })?;
        guard.armed = false;
        Ok(outcome)
    }

    pub(super) async fn run_batch(&self, mut batch: Vec<GraniteSpeech5EncoderPending>) {
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
        let batch_geometry = match model.granite_speech5_preparation_batch_geometry(&geometries) {
            Ok(geometry) => geometry,
            Err(error) => {
                Self::fail_batch(batch, error);
                return;
            }
        };
        let work = WorkUnit::PreSequencePreparation {
            kind: "asr.encoder.granite_speech5".into(),
        };
        let mut sealed = Vec::with_capacity(batch.len());
        for (index, row) in batch.iter().enumerate() {
            let cost = match model.granite_speech5_preparation_row_cost_for_batch(
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
                    row.request.clone(),
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
                    .map(|index| GraniteSpeech5PreparationBatchRow {
                        audio: inputs[*index].0.as_ref(),
                        sample_rate: inputs[*index].1,
                        request: &inputs[*index].2,
                    })
                    .collect::<Vec<_>>();
                let prepared =
                    physical_model.prepare_granite_speech5_prompt_artifact_batch(&selected)?;
                if prepared.len() != live.len() {
                    return Err(Error::InferenceError(
                        "Granite Speech 5.0 preparation batch returned the wrong row count".into(),
                    ));
                }
                Ok(prepared
                    .into_iter()
                    .zip(live.iter().copied())
                    .map(|(artifact, index)| {
                        let geometry = inputs[index].3;
                        let resident_tensor_bytes = artifact.resident_tensor_bytes()?;
                        if artifact.prompt_tokens() != geometry.prompt_tokens
                            || artifact.audio_tokens() != geometry.audio_tokens
                            || artifact.resident_host_bytes() != inputs[index].5
                            || resident_tensor_bytes != geometry.retained_device_bytes
                        {
                            return Err(Error::InferenceError(
                                "Granite Speech 5.0 batch artifact drifted from admitted geometry"
                                    .into(),
                            ));
                        }
                        Ok(PreparationArtifact {
                            retained: JobResourceObservation {
                                host_bytes: inputs[index].4,
                                accelerator_bytes: resident_tensor_bytes,
                            },
                            value: artifact,
                        })
                    })
                    .collect::<Vec<Result<PreparationArtifact<
                        Arc<GraniteSpeech5PreparedPromptArtifact>,
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

    pub(super) fn fail_batch(batch: Vec<GraniteSpeech5EncoderPending>, error: Error) {
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
            formation: EncoderBatchFormation::new(),
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
        let cancellation = pending.cancellation.clone();
        let (response, receiver) = oneshot::channel();
        let mut pending = pending;
        pending.response = Some(response);
        // A row waits for its bucket's running batch and joins the next one,
        // so requests that arrive while the device is busy still batch.
        if self.formation.push(key.clone(), pending) {
            let batcher = self.clone();
            tokio::spawn(async move {
                batcher
                    .formation
                    .run(&key, max_width, |batch| batcher.run_batch(batch))
                    .await;
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use tokio::sync::Notify;

    use super::EncoderBatchFormation;

    type Batches = Arc<Mutex<Vec<Vec<usize>>>>;

    /// Drains `key` with a runner whose first batch waits for `release`
    /// after signalling `started`; every batch is recorded in order.
    fn start_runner(
        formation: &Arc<EncoderBatchFormation<u32, usize>>,
        key: u32,
        max_width: usize,
        started: &Arc<Notify>,
        release: &Arc<Notify>,
    ) -> (Batches, tokio::task::JoinHandle<()>) {
        let batches = Batches::default();
        let (formation, recorded) = (formation.clone(), batches.clone());
        let (started, release) = (started.clone(), release.clone());
        let runner = tokio::spawn(async move {
            formation
                .run(&key, max_width, |batch| {
                    let first = {
                        let mut recorded = recorded.lock().unwrap();
                        recorded.push(batch);
                        recorded.len() == 1
                    };
                    let (started, release) = (started.clone(), release.clone());
                    async move {
                        if first {
                            started.notify_one();
                            release.notified().await;
                        }
                    }
                })
                .await;
        });
        (batches, runner)
    }

    /// Rows submitted together form one batch, and rows that arrive while
    /// it runs wait and form the next one instead of running alone.
    #[tokio::test]
    async fn rows_arriving_while_a_batch_runs_form_the_next_batch() {
        let formation = Arc::new(EncoderBatchFormation::new());
        let (started, release) = (Arc::new(Notify::new()), Arc::new(Notify::new()));
        assert!(formation.push(7, 0));
        let (batches, runner) = start_runner(&formation, 7, 8, &started, &release);
        for row in 1..5 {
            assert!(!formation.push(7, row));
        }
        started.notified().await;
        for row in 5..8 {
            assert!(!formation.push(7, row));
        }
        release.notify_one();
        runner.await.unwrap();
        assert_eq!(
            *batches.lock().unwrap(),
            vec![vec![0, 1, 2, 3, 4], vec![5, 6, 7]]
        );
        // The drained key is retired: its next row starts a new runner.
        assert!(formation.push(7, 8));
    }

    /// A key's queue drains in batches of at most its width, oldest first,
    /// while another key runs beside it.
    #[tokio::test]
    async fn batches_hold_at_most_the_width_and_keys_run_apart() {
        let formation = Arc::new(EncoderBatchFormation::new());
        let (started, release) = (Arc::new(Notify::new()), Arc::new(Notify::new()));
        assert!(formation.push(1, 0));
        assert!(formation.push(2, 100));
        let (batches, runner) = start_runner(&formation, 1, 4, &started, &release);
        for row in 1..10 {
            assert!(!formation.push(1, row));
        }
        release.notify_one();
        runner.await.unwrap();
        assert_eq!(
            *batches.lock().unwrap(),
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9]]
        );
        let (other, runner) = start_runner(&formation, 2, 4, &started, &release);
        release.notify_one();
        runner.await.unwrap();
        assert_eq!(*other.lock().unwrap(), vec![vec![100]]);
    }

    struct Row(Arc<AtomicUsize>);

    impl Drop for Row {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// A runner that panics drops the rows waiting on its key, so their
    /// submitters fail instead of hanging, and frees the key.
    #[tokio::test]
    async fn a_failed_runner_drops_its_waiting_rows_and_frees_the_key() {
        let formation = Arc::new(EncoderBatchFormation::<u32, Row>::new());
        let dropped = Arc::new(AtomicUsize::new(0));
        let (started, release) = (Arc::new(Notify::new()), Arc::new(Notify::new()));
        assert!(formation.push(3, Row(dropped.clone())));
        let runner = {
            let (formation, started, release) =
                (formation.clone(), started.clone(), release.clone());
            tokio::spawn(async move {
                formation
                    .run(&3, 8, |batch| {
                        let (started, release) = (started.clone(), release.clone());
                        async move {
                            started.notify_one();
                            release.notified().await;
                            drop(batch);
                            panic!("encoder batch failed");
                        }
                    })
                    .await;
            })
        };
        started.notified().await;
        assert!(!formation.push(3, Row(dropped.clone())));
        assert!(!formation.push(3, Row(dropped.clone())));
        release.notify_one();
        assert!(runner.await.is_err());
        assert_eq!(dropped.load(Ordering::SeqCst), 3);
        assert!(formation.push(3, Row(dropped.clone())));
    }

    /// Concurrent Nemotron clips of one power-of-two length bucket share a
    /// queue and form one batch of up to the stage's width; a longer clip
    /// queues apart, and with batching off (or a width-1 stage) every clip
    /// encodes alone.
    #[tokio::test]
    async fn concurrent_nemotron_clips_of_one_bucket_form_one_batch() {
        use crate::catalog::ModelVariant;
        use crate::engine::{ExecutionGroupId, ModelInstanceId, WorkUnit};
        use crate::models::architectures::nemotron::asr::{
            NemotronHostFeatures, NEMOTRON_ENCODER_STAGE,
        };
        use crate::runtime::{CapabilityKind, LoadedModelBundleDraft, RuntimeAdapterRegistry};

        let draft = LoadedModelBundleDraft::build(
            &RuntimeAdapterRegistry::built_in_with_execution_limits(4, 4).unwrap(),
            ExecutionGroupId::new(3),
            ModelInstanceId::new(9),
            ModelVariant::Nemotron35AsrStreaming06B,
            crate::backends::BackendKind::Cpu,
        )
        .unwrap();
        let contract = draft
            .execution_contracts(CapabilityKind::Asr)
            .unwrap()
            .into_iter()
            .find(|contract| contract.stages.len() == 3)
            .expect("the retained-row graph");
        let binding = contract.adapter_binding().unwrap();
        let stage = binding
            .stage_for_work(&WorkUnit::PreSequencePreparation {
                kind: NEMOTRON_ENCODER_STAGE.into(),
            })
            .unwrap();
        assert_eq!(stage.max_batch_size, 4);
        // 10, 8 and 6 s clips (126, 101 and 76 encoder frames) bucket to
        // 128; a 20 s clip (251 frames) to 256.
        let features =
            [1_000usize, 800, 600, 2_000, 900].map(NemotronHostFeatures::silent_for_test);
        let keys = features
            .iter()
            .map(|features| {
                super::NemotronEncoderBatcher::queue_key(&binding, stage, features, true)
                    .unwrap()
                    .expect("batched key")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            keys.iter()
                .map(|key| key.encoded_len_bucket)
                .collect::<Vec<_>>(),
            [128, 128, 128, 256, 128]
        );
        assert!(
            super::NemotronEncoderBatcher::queue_key(&binding, stage, &features[0], false)
                .unwrap()
                .is_none()
        );
        let mut solo_stage = stage.clone();
        solo_stage.max_batch_size = 1;
        assert!(super::NemotronEncoderBatcher::queue_key(
            &binding,
            &solo_stage,
            &features[0],
            true
        )
        .unwrap()
        .is_none());

        let formation = Arc::new(EncoderBatchFormation::new());
        let mut runners = Vec::new();
        for (row, key) in keys.iter().enumerate() {
            if formation.push(key.clone(), row) {
                runners.push(key.clone());
            }
        }
        assert_eq!(runners.len(), 2, "one runner per key");
        let batches = Batches::default();
        for key in runners {
            let recorded = batches.clone();
            formation
                .run(&key, stage.max_batch_size, |batch| {
                    recorded.lock().unwrap().push(batch);
                    async {}
                })
                .await;
        }
        assert_eq!(*batches.lock().unwrap(), vec![vec![0, 1, 2, 4], vec![3]]);
    }
}
