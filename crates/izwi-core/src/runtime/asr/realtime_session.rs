//! Realtime ASR sessions: limits and leases, physical sessions, engine
//! segments, cleanup and the stream's push/finish state.

use super::*;

impl RealtimeAsrSessionLimits {
    /// The engine config's limits, else the environment's, else the
    /// defaults for `backend`.
    #[cfg(test)]
    pub(super) fn resolve(config: RealtimeAsrSessionConfig, backend: BackendKind) -> Result<Self> {
        Self::resolve_for_sequences(config, backend, None)
    }

    /// [`Self::resolve`], with `max_num_seqs` (when set) replacing the
    /// backend's default session quota: the config and environment still win.
    pub(super) fn resolve_for_sequences(
        config: RealtimeAsrSessionConfig,
        backend: BackendKind,
        max_num_seqs: Option<usize>,
    ) -> Result<Self> {
        fn configured<T: Default + PartialEq>(value: Option<T>, field: &str) -> Result<Option<T>> {
            match value {
                Some(value) if value == T::default() => Err(Error::ConfigError(format!(
                    "asr_realtime.{field} must be positive"
                ))),
                value => Ok(value),
            }
        }
        let default_sessions = match (max_num_seqs, backend) {
            (Some(seqs), _) if seqs > 0 => seqs,
            (_, BackendKind::Cuda) => DEFAULT_CUDA_ASR_REALTIME_MAX_SESSIONS,
            (_, BackendKind::Cpu | BackendKind::Metal) => DEFAULT_ASR_REALTIME_MAX_SESSIONS,
        };
        let max_sessions = match configured(config.max_sessions, "max_sessions")? {
            Some(sessions) => sessions,
            None => positive_usize_env_or_default(ENV_ASR_REALTIME_MAX_SESSIONS, default_sessions)?,
        };
        let max_lifetime_secs = match configured(config.max_lifetime_secs, "max_lifetime_secs")? {
            Some(secs) => secs,
            None => positive_u64_env_or_default(
                ENV_ASR_REALTIME_MAX_LIFETIME_SECS,
                DEFAULT_ASR_REALTIME_MAX_LIFETIME_SECS,
            )?,
        };
        let idle_timeout_secs = match configured(config.idle_timeout_secs, "idle_timeout_secs")? {
            Some(secs) => secs,
            None => positive_u64_env_or_default(
                ENV_ASR_REALTIME_IDLE_TIMEOUT_SECS,
                DEFAULT_ASR_REALTIME_IDLE_TIMEOUT_SECS,
            )?,
        };
        Ok(Self {
            max_sessions,
            max_lifetime: Duration::from_secs(max_lifetime_secs),
            idle_timeout: Duration::from_secs(idle_timeout_secs),
        })
    }
}

impl RealtimeAsrSessionPolicy {
    pub(in crate::runtime) fn resolve_for_sequences(
        config: RealtimeAsrSessionConfig,
        backend: BackendKind,
        max_num_seqs: Option<usize>,
    ) -> Result<Self> {
        Self::new(RealtimeAsrSessionLimits::resolve_for_sequences(
            config,
            backend,
            max_num_seqs,
        )?)
    }

    pub(in crate::runtime) fn max_sessions(&self) -> usize {
        self.limits.max_sessions
    }

    pub(super) fn new(limits: RealtimeAsrSessionLimits) -> Result<Self> {
        if limits.max_sessions > Semaphore::MAX_PERMITS {
            return Err(Error::ConfigError(format!(
                "{ENV_ASR_REALTIME_MAX_SESSIONS} cannot exceed {}",
                Semaphore::MAX_PERMITS
            )));
        }
        if Instant::now().checked_add(limits.max_lifetime).is_none() {
            return Err(Error::ConfigError(format!(
                "{ENV_ASR_REALTIME_MAX_LIFETIME_SECS} exceeds the platform clock range"
            )));
        }
        Ok(Self {
            permits: Arc::new(Semaphore::new(limits.max_sessions)),
            limits,
            pushes: Arc::default(),
        })
    }

    pub(in crate::runtime) fn retained_sequence_capacity(&self) -> Result<u32> {
        u32::try_from(self.limits.max_sessions).map_err(|_| {
            Error::ConfigError(
                "realtime ASR session quota exceeds physical sequence capacity".into(),
            )
        })
    }

    pub(super) fn try_acquire(&self) -> Result<Arc<RealtimeAsrSessionLease>> {
        let permit = self
            .permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::Overloaded("realtime ASR session quota is full".to_string()))?;
        Ok(Arc::new(RealtimeAsrSessionLease { _permit: permit }))
    }
}

pub(super) struct RealtimeAsrSessionLease {
    pub(super) _permit: OwnedSemaphorePermit,
}

pub(super) fn positive_usize_env_or_default(name: &str, default: usize) -> Result<usize> {
    match std::env::var(name) {
        Ok(raw) => raw
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| Error::ConfigError(format!("{name} must be a positive integer"))),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(std::env::VarError::NotUnicode(_)) => Err(Error::ConfigError(format!(
            "{name} must contain valid UTF-8"
        ))),
    }
}

pub(super) fn positive_u64_env_or_default(name: &str, default: u64) -> Result<u64> {
    match std::env::var(name) {
        Ok(raw) => raw
            .trim()
            .parse::<u64>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| Error::ConfigError(format!("{name} must be a positive integer"))),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(std::env::VarError::NotUnicode(_)) => Err(Error::ConfigError(format!(
            "{name} must contain valid UTF-8"
        ))),
    }
}

pub(super) struct RealtimeAsrPhysicalSession {
    pub(super) runtime: Arc<RetainedTensorStateRuntimeV2>,
    pub(super) sequence: PhysicalStateSequenceId,
    pub(super) active_transaction: Option<PhysicalStateTransactionId>,
    pub(super) revision: u64,
}

impl RealtimeAsrPhysicalSession {
    pub(super) fn new(runtime: Arc<RetainedTensorStateRuntimeV2>) -> Result<Self> {
        let sequence = runtime.register_sequence()?;
        Ok(Self {
            runtime,
            sequence,
            active_transaction: None,
            revision: 0,
        })
    }

    pub(super) fn begin(&mut self) -> Result<PhysicalStateTransactionId> {
        if self.active_transaction.is_some() {
            return Err(Error::InferenceError(
                "realtime ASR physical transaction is already active".into(),
            ));
        }
        let transaction = self.runtime.begin_transaction(self.sequence)?;
        self.active_transaction = Some(transaction);
        Ok(transaction)
    }

    pub(super) fn commit(&mut self, transaction: PhysicalStateTransactionId, revision: u64) -> Result<()> {
        if self.active_transaction != Some(transaction) {
            return Err(Error::InferenceError(
                "realtime ASR physical transaction identity changed".into(),
            ));
        }
        self.runtime.commit_transaction(transaction, revision)?;
        self.active_transaction = None;
        self.revision = revision;
        Ok(())
    }

    pub(super) fn abort(&mut self) {
        if let Some(transaction) = self.active_transaction.take() {
            let _ = self.runtime.abort_transaction(transaction);
        }
    }
}

impl Drop for RealtimeAsrPhysicalSession {
    fn drop(&mut self) {
        self.abort();
        if let Err(error) = self.runtime.release_sequence(self.sequence) {
            tracing::error!(
                sequence = self.sequence.get(),
                error = %error,
                "failed to release realtime ASR physical sequence"
            );
        }
    }
}

pub(super) struct RuntimeAsrRealtimeState {
    pub(super) native: NativeAsrRealtimeState,
    pub(super) physical: RealtimeAsrPhysicalSession,
}

impl RuntimeAsrRealtimeState {
    pub(super) fn new(
        model: &NativeAsrModel,
        native: NativeAsrRealtimeState,
        runtime: Arc<RetainedTensorStateRuntimeV2>,
    ) -> Result<Self> {
        let mut owner = Self {
            native,
            physical: RealtimeAsrPhysicalSession::new(runtime)?,
        };
        let transaction = owner.physical.begin()?;
        let revision = 1;
        let result = model.stage_realtime_physical_state(
            &mut owner.native,
            &owner.physical.runtime,
            transaction,
            revision,
        );
        if let Err(error) = result {
            owner.physical.abort();
            return Err(error);
        }
        if let Err(error) = owner.physical.commit(transaction, revision) {
            owner.physical.abort();
            return Err(error);
        }
        Ok(owner)
    }

    pub(super) fn transact<T>(
        &mut self,
        model: &NativeAsrModel,
        operation: impl FnOnce(&mut NativeAsrRealtimeState) -> Result<T>,
    ) -> Result<T> {
        let checkpoint = match &self.native {
            NativeAsrRealtimeState::Nemotron(state) => {
                NativeAsrRealtimeState::Nemotron(state.clone())
            }
        };
        let transaction = self.physical.begin()?;
        let result = (|| {
            model.hydrate_realtime_physical_state(
                &mut self.native,
                &self.physical.runtime,
                transaction,
            )?;
            let output = operation(&mut self.native)?;
            let revision = self.physical.revision.checked_add(1).ok_or_else(|| {
                Error::InferenceError("realtime ASR physical revision overflowed".into())
            })?;
            model.stage_realtime_physical_state(
                &mut self.native,
                &self.physical.runtime,
                transaction,
                revision,
            )?;
            self.physical.commit(transaction, revision)?;
            Ok(output)
        })();
        if result.is_err() {
            self.native = checkpoint;
            if let Err(error) = model.clear_realtime_tensor_handles(&mut self.native) {
                tracing::error!(
                    error = %error,
                    "failed to drain Nemotron tensor handles after aborted operation"
                );
            }
            self.physical.abort();
        }
        result
    }

    /// [`Self::transact`] over several streams of one model: every row
    /// hydrates, one operation runs over all of them, then every row stages
    /// and commits. A failure rolls back every row that has not committed;
    /// the rows are untouched unless the commit itself failed part way.
    pub(super) fn transact_batch<T>(
        rows: &mut [&mut Self],
        model: &NativeAsrModel,
        operation: impl FnOnce(&mut [&mut NativeAsrRealtimeState]) -> Result<Vec<T>>,
    ) -> Result<Vec<T>> {
        let checkpoints = rows
            .iter()
            .map(|row| match &row.native {
                NativeAsrRealtimeState::Nemotron(state) => {
                    NativeAsrRealtimeState::Nemotron(state.clone())
                }
            })
            .collect::<Vec<_>>();
        let result = (|| {
            let mut transactions = Vec::with_capacity(rows.len());
            for row in rows.iter_mut() {
                transactions.push(row.physical.begin()?);
            }
            for (row, &transaction) in rows.iter_mut().zip(&transactions) {
                model.hydrate_realtime_physical_state(
                    &mut row.native,
                    &row.physical.runtime,
                    transaction,
                )?;
            }
            let outputs = {
                let mut natives = rows
                    .iter_mut()
                    .map(|row| &mut row.native)
                    .collect::<Vec<_>>();
                operation(&mut natives)?
            };
            if outputs.len() != rows.len() {
                return Err(Error::InferenceError(
                    "realtime ASR batch returned the wrong number of rows".into(),
                ));
            }
            let mut revisions = Vec::with_capacity(rows.len());
            for (row, &transaction) in rows.iter_mut().zip(&transactions) {
                let revision = row.physical.revision.checked_add(1).ok_or_else(|| {
                    Error::InferenceError("realtime ASR physical revision overflowed".into())
                })?;
                model.stage_realtime_physical_state(
                    &mut row.native,
                    &row.physical.runtime,
                    transaction,
                    revision,
                )?;
                revisions.push(revision);
            }
            for ((row, &transaction), revision) in rows.iter_mut().zip(&transactions).zip(revisions)
            {
                row.physical.commit(transaction, revision)?;
            }
            Ok(outputs)
        })();
        if result.is_err() {
            for (row, checkpoint) in rows.iter_mut().zip(checkpoints) {
                if row.physical.active_transaction.is_none() {
                    continue;
                }
                row.native = checkpoint;
                if let Err(error) = model.clear_realtime_tensor_handles(&mut row.native) {
                    tracing::error!(
                        error = %error,
                        "failed to drain Nemotron tensor handles after aborted operation"
                    );
                }
                row.physical.abort();
            }
        }
        result
    }
}

pub(super) struct RuntimeAsrRealtimeResources {
    pub(super) model: Option<Arc<NativeAsrModel>>,
    pub(super) state: Option<Arc<StdMutex<RuntimeAsrRealtimeState>>>,
    pub(super) execution_contract: Option<LoadedExecutionContract>,
    pub(super) residency_lease: Option<ModelResidencyLease>,
    pub(super) job: Option<JobLease>,
    pub(super) session_lease: Option<Arc<RealtimeAsrSessionLease>>,
    pub(super) engine_session: Option<RuntimeEngineRealtimeSession>,
    pub(super) engine_model: Option<EngineRealtimeModel>,
    pub(super) absolute_deadline: Instant,
    pub(super) idle_timeout: Duration,
    pub(super) last_activity: Instant,
    pub(super) active_operations: usize,
    pub(super) closed: bool,
    pub(super) timeout_reason: Option<&'static str>,
}

impl RuntimeAsrRealtimeResources {
    pub(super) fn idle_deadline(&self) -> Instant {
        self.last_activity
            .checked_add(self.idle_timeout)
            .unwrap_or(self.absolute_deadline)
            .min(self.absolute_deadline)
    }

    pub(super) fn next_watchdog_deadline(&self) -> Option<Instant> {
        if self.closed {
            None
        } else if self.active_operations > 0 {
            Some(self.absolute_deadline)
        } else {
            Some(self.idle_deadline())
        }
    }

    pub(super) fn expiration(&self, now: Instant) -> Option<&'static str> {
        realtime_asr_session_expiration(
            now,
            self.absolute_deadline,
            self.idle_deadline(),
            self.active_operations,
        )
    }

    pub(super) fn detach(&mut self) -> Option<RealtimeAsrDetachedResources> {
        if self.closed {
            return None;
        }
        self.closed = true;
        Some(RealtimeAsrDetachedResources {
            state: self.state.take(),
            model: self.model.take(),
            execution_contract: self.execution_contract.take(),
            residency_lease: self.residency_lease.take(),
            job: self.job.take(),
            session_lease: self.session_lease.take(),
            engine_session: self.engine_session.take(),
            engine_model: self.engine_model.take(),
        })
    }

    pub(super) fn detach_expired(&mut self, reason: &'static str) -> Option<RealtimeAsrDetachedResources> {
        self.timeout_reason = Some(reason);
        self.detach()
    }

    pub(super) fn closed_error(&self) -> Error {
        match self.timeout_reason {
            Some(reason) => Error::Timeout(reason.to_string()),
            None => Error::InvalidInput("realtime ASR stream is closed".to_string()),
        }
    }
}

pub(super) struct RealtimeAsrDetachedResources {
    pub(super) state: Option<Arc<StdMutex<RuntimeAsrRealtimeState>>>,
    pub(super) model: Option<Arc<NativeAsrModel>>,
    pub(super) execution_contract: Option<LoadedExecutionContract>,
    pub(super) residency_lease: Option<ModelResidencyLease>,
    pub(super) job: Option<JobLease>,
    pub(super) session_lease: Option<Arc<RealtimeAsrSessionLease>>,
    pub(super) engine_session: Option<RuntimeEngineRealtimeSession>,
    pub(super) engine_model: Option<EngineRealtimeModel>,
}

impl RealtimeAsrDetachedResources {
    pub(super) fn release(mut self) {
        // Physical allocations must disappear before their immutable
        // authorization and quota are released for another session.
        self.state.take();
        self.model.take();
        self.execution_contract.take();
        self.residency_lease.take();
        self.job.take();
        self.session_lease.take();
        self.engine_session.take();
        self.engine_model.take();
    }
}

/// Fail-closed owner for detached Runtime and Engine authorities. If the
/// asynchronous cleanup future is cancelled or its runtime is torn down,
/// dropping this guard deliberately retains every authority. Only an exact
/// Engine cleanup confirmation may disarm it.
pub(super) struct RealtimeAsrCleanupGuard {
    pub(super) cleanup: Option<RealtimeAsrDetachedResources>,
    pub(super) engine_session: Option<RuntimeEngineRealtimeSession>,
    pub(super) disarmed: bool,
}

impl RealtimeAsrCleanupGuard {
    pub(super) fn new(
        cleanup: RealtimeAsrDetachedResources,
        engine_session: RuntimeEngineRealtimeSession,
    ) -> Self {
        Self {
            cleanup: Some(cleanup),
            engine_session: Some(engine_session),
            disarmed: false,
        }
    }

    pub(super) fn engine_session(&self) -> &RuntimeEngineRealtimeSession {
        self.engine_session
            .as_ref()
            .expect("armed realtime cleanup guard must retain its Engine session")
    }

    #[cfg(test)]
    pub(super) fn for_test(cleanup: RealtimeAsrDetachedResources) -> Self {
        Self {
            cleanup: Some(cleanup),
            engine_session: None,
            disarmed: false,
        }
    }

    pub(super) fn release_confirmed(mut self) {
        self.disarmed = true;
        self.engine_session.take();
        if let Some(cleanup) = self.cleanup.take() {
            cleanup.release();
        }
    }
}

impl Drop for RealtimeAsrCleanupGuard {
    fn drop(&mut self) {
        if self.disarmed {
            return;
        }
        if let Some(engine_session) = self.engine_session.take() {
            std::mem::forget(engine_session);
        }
        if let Some(cleanup) = self.cleanup.take() {
            std::mem::forget(cleanup);
        }
    }
}

pub(super) fn schedule_realtime_asr_cleanup(cleanup: Option<RealtimeAsrDetachedResources>) {
    let Some(mut cleanup) = cleanup else {
        return;
    };
    if let Some(engine_session) = cleanup.engine_session.take() {
        let guard = RealtimeAsrCleanupGuard::new(cleanup, engine_session);
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                // Failures are logged and retain the authorities.
                let _ = confirm_realtime_asr_engine_cleanup(guard).await;
            });
            return;
        }
        tracing::error!(
            request_id = %guard.engine_session().handle.request_id(),
            "cannot asynchronously abort Engine realtime ASR session without a Tokio runtime"
        );
        drop(guard);
        return;
    }
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.spawn_blocking(move || cleanup.release());
    } else {
        cleanup.release();
    }
}

/// Releases a detached Engine session's authorities only after the Engine
/// confirms its cleanup. Any failure, or dropping this future, retains them.
pub(super) async fn confirm_realtime_asr_engine_cleanup(guard: RealtimeAsrCleanupGuard) -> Result<()> {
    let cleanup_receipt = match guard
        .engine_session()
        .engine
        .cleanup_realtime_asr_session(&guard.engine_session().handle)
        .await
    {
        Ok(receipt) => receipt,
        Err(error) => {
            tracing::error!(
                request_id = %guard.engine_session().handle.request_id(),
                error = %error,
                "Engine realtime ASR cleanup receipt could not be created; retaining authorities"
            );
            return Err(error);
        }
    };
    if let Err(error) = cleanup_receipt.confirmed().await {
        tracing::error!(
            request_id = %guard.engine_session().handle.request_id(),
            error = %error,
            "Engine realtime ASR cleanup was not confirmed; retaining authorities"
        );
        return Err(error);
    }
    tokio::task::spawn_blocking(move || guard.release_confirmed())
        .await
        .map_err(|error| {
            Error::InferenceError(format!("realtime ASR cleanup release failed: {error}"))
        })
}

/// Like [`schedule_realtime_asr_cleanup`], but completes before returning,
/// so a rollover never holds two segments' physical state at once.
pub(super) async fn release_realtime_asr_cleanup(cleanup: Option<RealtimeAsrDetachedResources>) -> Result<()> {
    let Some(mut cleanup) = cleanup else {
        return Ok(());
    };
    match cleanup.engine_session.take() {
        Some(engine_session) => {
            confirm_realtime_asr_engine_cleanup(RealtimeAsrCleanupGuard::new(
                cleanup,
                engine_session,
            ))
            .await
        }
        None => tokio::task::spawn_blocking(move || cleanup.release())
            .await
            .map_err(|error| {
                Error::InferenceError(format!("realtime ASR cleanup release failed: {error}"))
            }),
    }
}

pub(super) struct RealtimeAsrOperationHandles {
    pub(super) model: Arc<NativeAsrModel>,
    pub(super) state: Arc<StdMutex<RuntimeAsrRealtimeState>>,
    pub(super) execution_contract: LoadedExecutionContract,
    pub(super) job: JobLease,
    pub(super) session_lease: Arc<RealtimeAsrSessionLease>,
    pub(super) _guard: RealtimeAsrOperationGuard,
}

#[derive(Clone)]
pub(super) struct RuntimeEngineRealtimeSession {
    pub(super) engine: Arc<crate::engine::Engine>,
    pub(super) handle: RealtimeAsrSessionHandle,
    pub(super) model: EngineRealtimeModel,
    pub(super) retained_metadata_bytes: u64,
    pub(super) max_cache_append: usize,
}

/// A family's name in an Engine realtime error.
pub(super) fn engine_realtime_family_name(family: ModelFamily) -> &'static str {
    match family {
        ModelFamily::Lfm3Audio => "LFM3-Audio",
        ModelFamily::Voxtral => "Voxtral",
        _ => "this model's",
    }
}

/// Whether `language` asks for English, the one language LFM3-Audio's
/// LiveASR transcribes.
pub(super) fn lfm3_live_asr_language(language: &str) -> bool {
    let language = language.trim().to_ascii_lowercase();
    language.is_empty()
        || language == "en"
        || language == "english"
        || language.starts_with("en-")
        || language.starts_with("en_")
}

/// The loaded model an Engine realtime session runs on. The Engine path
/// serves Voxtral and LFM3-Audio; Nemotron's retained tensor state runs on
/// the direct path (`realtime_asr_execution_path`).
#[derive(Clone)]
pub(super) enum EngineRealtimeModel {
    /// Voxtral bounds every operation's output steps alike.
    Voxtral {
        model: VoxtralModelLease,
        max_output_steps: usize,
    },
    /// LFM3-Audio hears 16 kHz audio and bounds each operation by it; the
    /// lease keeps the model loaded for the session.
    Lfm3Audio { _model: Lfm25AudioModelLease },
}

/// What one Engine realtime operation costs, the retained usage it leaves,
/// and the output steps it may take.
pub(super) struct EngineRealtimeOperationBudget {
    pub(super) preparation_cost: WorkCost,
    pub(super) committed_observation: JobResourceObservation,
    pub(super) max_output_steps: usize,
}

impl RuntimeEngineRealtimeSession {
    /// The budget of a push of `pushed` samples after `committed` ones, or
    /// of the finish (`finish`, nothing pushed).
    pub(super) fn operation_budget(
        &self,
        committed: usize,
        pushed: usize,
        sample_rate: u32,
        finish: bool,
    ) -> Result<EngineRealtimeOperationBudget> {
        match &self.model {
            EngineRealtimeModel::Voxtral {
                model,
                max_output_steps,
            } => {
                // Voxtral recomputes its preparation over the committed prefix.
                let total = committed.checked_add(pushed).ok_or_else(|| {
                    Error::Overloaded("realtime ASR sample clock overflowed".into())
                })?;
                let mode = if finish {
                    VoxtralRealtimePreparationMode::Finish
                } else {
                    VoxtralRealtimePreparationMode::Push
                };
                let geometry = model.realtime_preparation_geometry_for_source_samples(
                    total,
                    sample_rate,
                    mode,
                )?;
                Ok(EngineRealtimeOperationBudget {
                    preparation_cost: voxtral_realtime_preparation_cost(model, geometry)?,
                    committed_observation: voxtral_realtime_committed_observation(
                        self.retained_metadata_bytes,
                        model.realtime_prepared_resource_usage(geometry)?,
                    )?,
                    max_output_steps: *max_output_steps,
                })
            }
            EngineRealtimeModel::Lfm3Audio { .. } => {
                lfm3_audio_realtime_operation_budget(self.retained_metadata_bytes, pushed, finish)
            }
        }
    }
}

/// An LFM3-Audio push hears only its own audio: its cost follows its mel
/// frames, and the session retains its bounded streams whatever it heard.
pub(super) fn lfm3_audio_realtime_operation_budget(
    retained_metadata_bytes: u64,
    pushed: usize,
    finish: bool,
) -> Result<EngineRealtimeOperationBudget> {
    use crate::models::architectures::lfm3_audio::model::lfm3_live_asr_operation_bounds;
    use crate::models::architectures::lfm3_audio::preprocessor::N_MELS;
    use crate::models::architectures::lfm3_audio::{
        LFM3_AUDIO_REALTIME_MAX_PREPARATION_FRAMES,
        LFM3_AUDIO_REALTIME_PREPARATION_WORKSPACE_BYTES, LFM3_AUDIO_REALTIME_STATE_HOST_BYTES,
        LFM3_AUDIO_REALTIME_STATE_TENSOR_BYTES,
    };
    let (frames, max_output_steps) = lfm3_live_asr_operation_bounds(pushed, finish);
    let frames = u64::try_from(frames)
        .ok()
        .filter(|frames| *frames <= LFM3_AUDIO_REALTIME_MAX_PREPARATION_FRAMES)
        .ok_or_else(|| {
            Error::InvalidInput("LFM3-Audio LiveASR push exceeds its preparation bound".into())
        })?;
    Ok(EngineRealtimeOperationBudget {
        preparation_cost: WorkCost::new(
            frames,
            frames * N_MELS as u64,
            LFM3_AUDIO_REALTIME_PREPARATION_WORKSPACE_BYTES,
        ),
        committed_observation: JobResourceObservation::new(
            retained_metadata_bytes
                .checked_mul(2)
                .and_then(|bytes| bytes.checked_add(LFM3_AUDIO_REALTIME_STATE_HOST_BYTES))
                .ok_or_else(|| {
                    Error::Overloaded("LFM3-Audio retained host usage overflowed".into())
                })?,
            LFM3_AUDIO_REALTIME_STATE_TENSOR_BYTES,
        ),
        max_output_steps,
    })
}

/// LFM3-Audio's side of an Engine realtime stream, which outlives its
/// segments: the source audio resampled to the model's 16 kHz, the audio a
/// new segment hears again, and the transcript the segments join into.
pub(super) struct Lfm3RealtimeInput {
    resampler: Option<HighQualityResampler>,
    source_rate: u32,
    source_samples: usize,
    heard_samples: usize,
    /// The last `LFM3_LIVE_ASR_OVERLAP_SECONDS` of 16 kHz audio heard.
    recent: VecDeque<f32>,
    pub(super) transcript: Lfm3LiveTranscript,
}

impl Lfm3RealtimeInput {
    pub(super) const SAMPLE_RATE: u32 =
        crate::models::architectures::lfm3_audio::config::INPUT_SAMPLE_RATE;

    pub(super) fn new() -> Self {
        Self {
            resampler: None,
            source_rate: 0,
            source_samples: 0,
            heard_samples: 0,
            recent: VecDeque::new(),
            transcript: Lfm3LiveTranscript::default(),
        }
    }

    /// Samples one segment hears, its re-heard overlap included.
    pub(super) fn segment_samples() -> usize {
        (LFM3_LIVE_ASR_SEGMENT_SECONDS * Self::SAMPLE_RATE as f32) as usize
    }

    pub(super) fn overlap_samples() -> usize {
        (LFM3_LIVE_ASR_OVERLAP_SECONDS * Self::SAMPLE_RATE as f32) as usize
    }

    /// `samples` at `sample_rate` as the model's 16 kHz audio, as far as the
    /// resampler has produced it.
    pub(super) fn resample(&mut self, samples: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        if self.source_rate == 0 {
            self.resampler = (sample_rate != Self::SAMPLE_RATE)
                .then(|| HighQualityResampler::new(sample_rate, Self::SAMPLE_RATE))
                .transpose()?;
            self.source_rate = sample_rate;
        } else if self.source_rate != sample_rate {
            return Err(Error::InvalidInput(
                "realtime ASR sample rate changed within one stream".into(),
            ));
        }
        self.source_samples = self
            .source_samples
            .checked_add(samples.len())
            .ok_or_else(|| Error::Overloaded("realtime ASR sample clock overflowed".into()))?;
        let mut heard = Vec::new();
        match self.resampler.as_mut() {
            Some(resampler) => resampler.push(samples, &mut heard)?,
            None => heard.extend_from_slice(samples),
        }
        self.heard_samples += heard.len();
        Ok(heard)
    }

    /// The resampler's tail at the stream's end, cut to the stream's exact
    /// 16 kHz length.
    pub(super) fn finish(&mut self) -> Result<Vec<f32>> {
        let Some(mut resampler) = self.resampler.take() else {
            return Ok(Vec::new());
        };
        let target = crate::audio::target_sample_count(
            self.source_samples,
            self.source_rate,
            Self::SAMPLE_RATE,
        );
        let mut tail = Vec::new();
        resampler.finish(&mut tail, target)?;
        tail.truncate(target.saturating_sub(self.heard_samples));
        self.heard_samples += tail.len();
        Ok(tail)
    }

    /// Remember audio the current segment heard.
    pub(super) fn heard(&mut self, samples: &[f32]) {
        let keep = Self::overlap_samples();
        self.recent.extend(samples.iter().copied());
        let excess = self.recent.len().saturating_sub(keep);
        self.recent.drain(..excess);
    }

    /// The audio a new segment hears again.
    pub(super) fn overlap(&self) -> Vec<f32> {
        self.recent.iter().copied().collect()
    }
}

/// Where a realtime session executes, from the capability table. Voxtral
/// publishes a paged managed state runtime, which the Engine's realtime
/// session admission requires; Nemotron publishes a retained tensor state
/// runtime and runs on the direct native streaming path (routing it through
/// Engine admission fails closed with "requires a physical managed state
/// runtime"). A model without a realtime row takes the direct path.
pub(super) fn realtime_asr_execution_path(variant: ModelVariant) -> RealtimeAsrPath {
    capability_row(CapabilityKind::RealtimeAsr, variant)
        .and_then(|row| row.realtime_path)
        .unwrap_or(RealtimeAsrPath::Direct)
}

pub(super) struct RealtimeAsrEngineOperationHandles {
    pub(super) session: RuntimeEngineRealtimeSession,
    pub(super) job: JobLease,
    pub(super) session_lease: Arc<RealtimeAsrSessionLease>,
    pub(super) _guard: RealtimeAsrOperationGuard,
}

pub(super) struct RealtimeAsrOperationGuard {
    pub(super) resources: Arc<StdMutex<RuntimeAsrRealtimeResources>>,
    pub(super) activity: Arc<Notify>,
}

impl Drop for RealtimeAsrOperationGuard {
    fn drop(&mut self) {
        if let Ok(mut resources) = self.resources.lock() {
            resources.active_operations = resources.active_operations.saturating_sub(1);
            if !resources.closed {
                resources.last_activity = Instant::now();
            }
        }
        self.activity.notify_one();
    }
}

/// An Engine realtime session prepares at most `max_samples` of source
/// audio (Voxtral re-encodes its whole prefix on every push), so a longer
/// stream continues in a fresh session. A segment looks for a pause in its
/// final fifth so the boundary falls between words; it rolls over at the
/// hard ceiling only when the speaker never pauses.
pub(super) const REALTIME_SEGMENT_SOFT_ROLLOVER_DIVISOR: usize = 5;

/// Sustained quiet audio that counts as a pause between words.
pub(super) const REALTIME_SEGMENT_PAUSE_MS: u64 = 200;

/// Chunk RMS below which pushed audio counts as quiet (about -40 dBFS).
pub(super) const REALTIME_SEGMENT_QUIET_RMS: f64 = 0.01;

/// What an Engine realtime stream carries from one session to the next.
#[derive(Debug, Default)]
pub(super) struct RealtimeEngineSegment {
    pub(super) language: Option<String>,
    pub(super) prompt: Option<String>,
    pub(super) trailing_quiet_samples: usize,
    pub(super) separator_pending: bool,
}

impl RealtimeEngineSegment {
    pub(super) fn new(language: Option<&str>, prompt: Option<&str>) -> Self {
        Self {
            language: language.map(str::to_string),
            prompt: prompt.map(str::to_string),
            ..Self::default()
        }
    }
}

pub(super) fn realtime_chunk_is_quiet(samples: &[f32]) -> bool {
    if samples.is_empty() {
        return false;
    }
    let energy: f64 = samples
        .iter()
        .map(|sample| f64::from(*sample) * f64::from(*sample))
        .sum();
    (energy / samples.len() as f64).sqrt() < REALTIME_SEGMENT_QUIET_RMS
}

pub(super) fn realtime_trailing_quiet_samples(previous: usize, chunk: &[f32]) -> usize {
    if realtime_chunk_is_quiet(chunk) {
        previous.saturating_add(chunk.len())
    } else {
        0
    }
}

/// Whether an Engine realtime stream must start a new session before it
/// accepts `incoming` more samples.
pub(super) fn realtime_segment_rollover_due(
    committed_samples: usize,
    incoming_samples: usize,
    trailing_quiet_samples: usize,
    max_samples: usize,
    sample_rate: u32,
) -> bool {
    if committed_samples == 0 {
        return false;
    }
    if committed_samples.saturating_add(incoming_samples) > max_samples {
        return true;
    }
    let soft_start = max_samples - max_samples / REALTIME_SEGMENT_SOFT_ROLLOVER_DIVISOR;
    let pause_samples = usize::try_from(u64::from(sample_rate) * REALTIME_SEGMENT_PAUSE_MS / 1_000)
        .unwrap_or(usize::MAX)
        .max(1);
    committed_samples >= soft_start && trailing_quiet_samples >= pause_samples
}

impl RuntimeAsrRealtimeStream {
    pub(super) fn execution_path(&self) -> RealtimeAsrPath {
        realtime_asr_execution_path(self.variant)
    }

    /// Whether the stream has ended: finished, failed, or expired. A closed
    /// stream rejects every later operation, so callers must stop feeding it.
    pub fn is_closed(&self) -> bool {
        self.resources
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .closed
    }

    /// The session-wide authorities a rollover hands to the next segment:
    /// the session quota slot, the absolute lifetime and the idle timeout.
    pub(super) fn segment_authorities(&self) -> Result<(Arc<RealtimeAsrSessionLease>, Instant, Duration)> {
        let resources = self.resources.lock().map_err(|_| {
            Error::InferenceError("realtime ASR resource mutex poisoned".to_string())
        })?;
        if resources.closed {
            return Err(resources.closed_error());
        }
        let session_lease = resources.session_lease.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("realtime ASR session lease is unavailable".into())
        })?;
        Ok((
            session_lease,
            resources.absolute_deadline,
            resources.idle_timeout,
        ))
    }

    pub(super) fn detach(&self) -> Option<RealtimeAsrDetachedResources> {
        let cleanup = self
            .resources
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .detach();
        self.activity.notify_one();
        cleanup
    }

    /// Continues this stream on `next`'s fresh Engine session. The
    /// transcript, chunk numbering and language carry over; the segment
    /// clock restarts. `next` is left holding this stream's closed session.
    pub(super) fn continue_on(&mut self, next: &mut RuntimeAsrRealtimeStream) {
        std::mem::swap(&mut self.resources, &mut next.resources);
        std::mem::swap(&mut self.activity, &mut next.activity);
        std::mem::swap(&mut self.operation_gate, &mut next.operation_gate);
        self.max_samples = next.max_samples;
        self.committed_samples = 0;
        self.engine_sample_rate = None;
        self.segment.trailing_quiet_samples = 0;
        self.segment.separator_pending = true;
    }

    pub(super) fn begin_engine_operation(
        &mut self,
        refresh_activity: bool,
    ) -> Result<RealtimeAsrEngineOperationHandles> {
        let now = Instant::now();
        let mut resources = self.resources.lock().map_err(|_| {
            Error::InferenceError("realtime ASR resource mutex poisoned".to_string())
        })?;
        if resources.closed {
            return Err(resources.closed_error());
        }
        if let Some(reason) = resources.expiration(now) {
            let cleanup = resources.detach_expired(reason);
            drop(resources);
            schedule_realtime_asr_cleanup(cleanup);
            return Err(Error::Timeout(reason.to_string()));
        }
        let session = resources.engine_session.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("Engine realtime ASR session is unavailable".into())
        })?;
        let job = resources.job.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("realtime ASR stream job is unavailable".into())
        })?;
        let session_lease = resources.session_lease.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("realtime ASR session lease is unavailable".into())
        })?;
        resources.active_operations = resources
            .active_operations
            .checked_add(1)
            .ok_or_else(|| Error::Overloaded("realtime ASR operation count overflowed".into()))?;
        if refresh_activity {
            resources.last_activity = now;
        }
        drop(resources);
        if refresh_activity {
            self.activity.notify_one();
        }
        Ok(RealtimeAsrEngineOperationHandles {
            session,
            job,
            session_lease,
            _guard: RealtimeAsrOperationGuard {
                resources: self.resources.clone(),
                activity: self.activity.clone(),
            },
        })
    }

    pub(super) fn map_engine_outputs(
        &mut self,
        outputs: Vec<StreamingOutput>,
    ) -> Vec<RuntimeAsrRealtimeEvent> {
        outputs
            .into_iter()
            .map(|output| {
                let mut delta = output.text.unwrap_or_default();
                if let Some(lfm3) = self.lfm3.as_mut() {
                    // A segment's text joins the transcript past whatever it
                    // repeats of the previous segment's end.
                    delta = lfm3.transcript.push(&delta);
                    if output.is_final {
                        delta.push_str(&lfm3.transcript.flush());
                    }
                } else if self.segment.separator_pending && !delta.is_empty() {
                    // A new segment's decoder starts without the previous
                    // word, so it may not emit the space between them.
                    self.segment.separator_pending = false;
                    if !self.engine_text.is_empty()
                        && !self.engine_text.ends_with(char::is_whitespace)
                        && !delta.starts_with(char::is_whitespace)
                    {
                        delta.insert(0, ' ');
                    }
                }
                self.engine_text_event(delta, output.is_final)
            })
            .collect()
    }

    /// The event that adds `delta` to the stream's transcript.
    pub(super) fn engine_text_event(
        &mut self,
        delta: String,
        is_final: bool,
    ) -> RuntimeAsrRealtimeEvent {
        self.engine_text.push_str(&delta);
        let chunk_index = self.engine_chunk_index;
        self.engine_chunk_index = self.engine_chunk_index.saturating_add(1);
        RuntimeAsrRealtimeEvent {
            delta,
            text: self.engine_text.clone(),
            is_final,
            chunk_index,
        }
    }
    pub(super) fn begin_operation(&mut self, refresh_activity: bool) -> Result<RealtimeAsrOperationHandles> {
        let now = Instant::now();
        let mut resources = self.resources.lock().map_err(|_| {
            Error::InferenceError("realtime ASR resource mutex poisoned".to_string())
        })?;
        if resources.closed {
            return Err(resources.closed_error());
        }
        if let Some(reason) = resources.expiration(now) {
            let cleanup = resources.detach_expired(reason);
            drop(resources);
            schedule_realtime_asr_cleanup(cleanup);
            return Err(Error::Timeout(reason.to_string()));
        }
        let model = resources.model.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("realtime ASR stream model is unavailable".to_string())
        })?;
        let state = resources.state.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("realtime ASR stream state is unavailable".to_string())
        })?;
        let execution_contract =
            resources
                .execution_contract
                .as_ref()
                .cloned()
                .ok_or_else(|| {
                    Error::InferenceError(
                        "realtime ASR stream execution contract is unavailable".to_string(),
                    )
                })?;
        let job = resources.job.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("realtime ASR stream job is unavailable".to_string())
        })?;
        let session_lease = resources.session_lease.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("realtime ASR session lease is unavailable".to_string())
        })?;
        let active_operations = resources.active_operations.checked_add(1).ok_or_else(|| {
            Error::Overloaded("realtime ASR operation count overflowed".to_string())
        })?;
        if refresh_activity {
            resources.last_activity = now;
        }
        resources.active_operations = active_operations;
        drop(resources);
        if refresh_activity {
            self.activity.notify_one();
        }
        Ok(RealtimeAsrOperationHandles {
            model: model.clone(),
            state,
            execution_contract,
            job,
            session_lease,
            _guard: RealtimeAsrOperationGuard {
                resources: self.resources.clone(),
                activity: self.activity.clone(),
            },
        })
    }

    pub(super) fn ensure_open(&mut self) -> Result<()> {
        let now = Instant::now();
        let mut resources = self.resources.lock().map_err(|_| {
            Error::InferenceError("realtime ASR resource mutex poisoned".to_string())
        })?;
        if resources.closed {
            return Err(resources.closed_error());
        }
        if let Some(reason) = resources.expiration(now) {
            let cleanup = resources.detach_expired(reason);
            drop(resources);
            schedule_realtime_asr_cleanup(cleanup);
            return Err(Error::Timeout(reason.to_string()));
        }
        Ok(())
    }

    pub(super) fn close(&self) {
        schedule_realtime_asr_cleanup(self.detach());
    }

    pub(super) fn close_due_to_timeout(&self) {
        let mut resources = self
            .resources
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let reason = resources
            .expiration(Instant::now())
            .unwrap_or("realtime ASR operation timed out");
        let cleanup = resources.detach_expired(reason);
        drop(resources);
        schedule_realtime_asr_cleanup(cleanup);
        self.activity.notify_one();
    }
}

impl Drop for RuntimeAsrRealtimeStream {
    fn drop(&mut self) {
        self.close();
    }
}

pub(super) fn realtime_asr_session_expiration(
    now: Instant,
    absolute_deadline: Instant,
    idle_deadline: Instant,
    active_operations: usize,
) -> Option<&'static str> {
    if now >= absolute_deadline {
        Some("realtime ASR stream exceeded its absolute lifetime")
    } else if active_operations == 0 && now >= idle_deadline {
        Some("realtime ASR stream exceeded its idle timeout")
    } else {
        None
    }
}

pub(super) fn spawn_realtime_asr_watchdog(
    resources: &Arc<StdMutex<RuntimeAsrRealtimeResources>>,
    activity: Arc<Notify>,
) {
    let weak_resources = Arc::downgrade(resources);
    tokio::spawn(async move {
        loop {
            // Register before reading the deadline so activity between the
            // state snapshot and select cannot be lost.
            let notified = activity.notified();
            let Some(resources) = weak_resources.upgrade() else {
                return;
            };
            let deadline = match resources.lock() {
                Ok(resources) => resources.next_watchdog_deadline(),
                Err(_) => return,
            };
            drop(resources);
            let Some(deadline) = deadline else {
                return;
            };

            tokio::select! {
                _ = tokio::time::sleep_until(deadline.into()) => {
                    let Some(resources) = weak_resources.upgrade() else {
                        return;
                    };
                    let mut resources = match resources.lock() {
                        Ok(resources) => resources,
                        Err(_) => return,
                    };
                    if let Some(reason) = resources.expiration(Instant::now()) {
                        let cleanup = resources.detach_expired(reason);
                        drop(resources);
                        schedule_realtime_asr_cleanup(cleanup);
                        return;
                    }
                }
                _ = notified => {}
            }
        }
    });
}

pub(super) fn map_native_realtime_events(events: Vec<NativeAsrRealtimeEvent>) -> Vec<RuntimeAsrRealtimeEvent> {
    events
        .into_iter()
        .map(|event| RuntimeAsrRealtimeEvent {
            delta: event.delta,
            text: event.text,
            is_final: event.is_final,
            chunk_index: event.chunk_index,
        })
        .collect()
}

pub(super) fn realtime_state_observation(state: &NativeAsrRealtimeState) -> Result<JobResourceObservation> {
    let (host_bytes, tensor_bytes) = state.resource_usage().ok_or_else(|| {
        Error::InferenceError("realtime ASR state resource usage is unavailable".to_string())
    })?;
    Ok(JobResourceObservation::new(host_bytes, tensor_bytes))
}

pub(super) fn realtime_state_with_input_observation(
    state: &NativeAsrRealtimeState,
    input_sample_capacity: usize,
) -> Result<JobResourceObservation> {
    let mut observation = realtime_state_observation(state)?;
    let input_bytes = input_sample_capacity
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| Error::Overloaded("realtime ASR input storage overflowed".to_string()))?;
    observation.host_bytes = observation
        .host_bytes
        .checked_add(input_bytes)
        .ok_or_else(|| Error::Overloaded("realtime ASR host usage overflowed".to_string()))?;
    Ok(observation)
}

pub(super) fn validate_realtime_input_copy(samples: usize, max_samples: usize) -> Result<()> {
    if samples > max_samples {
        return Err(Error::InvalidInput(format!(
            "Realtime ASR input chunk of {samples} samples exceeds the stream reservation of {max_samples} samples"
        )));
    }
    Ok(())
}

pub(super) async fn run_realtime_blocking_operation<T, B, F>(
    coordinator: &Arc<InferenceCoordinator>,
    job: &JobLease,
    execution_contract: LoadedExecutionContract,
    operation_gate: Arc<tokio::sync::Mutex<()>>,
    operation_kind: &'static str,
    build_operation: B,
) -> Result<T>
where
    T: Send + 'static,
    B: FnOnce() -> Result<F> + Send,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    let operation_gate = coordinator
        .acquire_job_ordering(job, operation_gate)
        .await?;
    let operation = build_operation()?;
    coordinator
        .run_loaded_blocking_stage(
            job,
            execution_contract,
            WorkUnit::AtomicJob {
                kind: operation_kind.to_string(),
            },
            move || {
                let _operation_gate = operation_gate;
                operation()
            },
        )
        .await
}

pub(super) fn realtime_stream_resource_vector(
    backend: BackendKind,
    host_bytes: u64,
    tensor_bytes: u64,
) -> Result<ResourceVector> {
    let mut resources = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => {
            resources.host_bytes =
                ResourceAmount::Known(host_bytes.checked_add(tensor_bytes).ok_or_else(|| {
                    Error::Overloaded("realtime ASR resource reservation overflowed".to_string())
                })?);
        }
        BackendKind::Metal => {
            resources.unified_bytes =
                ResourceAmount::Known(host_bytes.checked_add(tensor_bytes).ok_or_else(|| {
                    Error::Overloaded("realtime ASR resource reservation overflowed".to_string())
                })?);
        }
        BackendKind::Cuda => {
            resources.host_bytes = ResourceAmount::Known(host_bytes);
            resources.device_bytes = ResourceAmount::Known(tensor_bytes);
        }
    }
    Ok(resources)
}

pub(super) fn voxtral_realtime_preparation_cost(
    model: &VoxtralModelLease,
    geometry: VoxtralRealtimePreparationGeometry,
) -> Result<WorkCost> {
    let batch = model.realtime_preparation_batch_geometry(&[geometry])?;
    let seal = model.realtime_preparation_stage_seal()?;
    voxtral_realtime_preparation_cost_from_geometry(geometry, batch, seal)
}

pub(super) fn voxtral_realtime_preparation_cost_from_geometry(
    geometry: VoxtralRealtimePreparationGeometry,
    batch: VoxtralRealtimePreparationBatchGeometry,
    seal: VoxtralRealtimePreparationStageSeal,
) -> Result<WorkCost> {
    if batch.materialized_tensor_elements_per_row > seal.max_materialized_tensor_elements_per_row
        || batch.workspace_per_row_bytes > seal.max_workspace_bytes
    {
        return Err(Error::InferenceError(
            "Voxtral preparation geometry exceeds its loaded stage seal".into(),
        ));
    }
    Ok(WorkCost::new(
        u64::try_from(geometry.source_samples)
            .map_err(|_| Error::Overloaded("Voxtral preparation work overflowed".into()))?
            .max(1),
        batch.materialized_tensor_elements_per_row,
        batch.workspace_per_row_bytes,
    ))
}

pub(super) fn voxtral_realtime_committed_observation(
    retained_metadata_bytes: u64,
    usage: VoxtralRealtimePreparedResourceUsage,
) -> Result<JobResourceObservation> {
    Ok(JobResourceObservation::new(
        retained_metadata_bytes
            .checked_mul(2)
            .ok_or_else(|| Error::Overloaded("Voxtral retained metadata overflowed".into()))?
            .checked_add(usage.host_bytes)
            .ok_or_else(|| Error::Overloaded("Voxtral retained host usage overflowed".into()))?,
        usage.tensor_bytes,
    ))
}

pub(super) fn add_realtime_stream_reservation(
    base: ResourceVector,
    backend: BackendKind,
    reservation: NativeAsrRealtimeResourceReservation,
) -> Result<ResourceVector> {
    base.checked_add(realtime_stream_resource_vector(
        backend,
        reservation.host_bytes(),
        crate::models::architectures::nemotron::asr::NEMOTRON_REALTIME_STAGE_WORKSPACE_BYTES,
    )?)
}
