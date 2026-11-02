//! Fish Audio Drama 3's model quanta and codec candidates stay invisible
//! until the core's commit decision: the family's own copy of the Fish
//! pending coordinator (`fish_pending.rs`), so S2's stays untouched.
//!
//! A decode or prefill quantum stages the request's quantum checkpoint; a
//! commit accepts it (the family checks its KV lengths), an abort restores
//! every piece of the request bit for bit, and an aborted first quantum
//! drops the request's state, which has nothing earlier to return to. A
//! codec quantum stages its candidate stream; only a commit installs it and
//! counts its PCM.
use super::*;
use crate::models::architectures::fish_drama3::{
    FishDrama3AudioCandidate, FishDrama3QuantumCheckpoint,
};

pub(super) struct FishDrama3CodecCommit {
    pub(super) candidate: FishDrama3AudioCandidate,
    pub(super) stream_sequence: usize,
    pub(super) codec_ms: f64,
    pub(super) first_audio_ms: Option<f64>,
}

enum FishDrama3PendingUpdate {
    Codec(FishDrama3CodecCommit),
    Quantum {
        checkpoint: FishDrama3QuantumCheckpoint,
        /// The request's first quantum: an abort drops its state.
        fresh: bool,
    },
}

struct PendingFishDrama3Row {
    session: SessionKey,
    active: ActiveFishDrama3TtsDecode,
    update: FishDrama3PendingUpdate,
    decision: Option<PendingQuantumDecision>,
}

pub(super) struct FishDrama3StateCoordinator {
    pub(super) states: Arc<ExecutorStateStore<ActiveFishDrama3TtsDecode>>,
    rows: Mutex<HashMap<PlanId, PendingFishDrama3Row>>,
}

/// The most PCM samples `active` may emit: its frame budget at the codec's
/// frame size.
fn sample_budget(active: &ActiveFishDrama3TtsDecode) -> Result<usize> {
    active
        .state
        .params()
        .max_frames
        .checked_mul(active.model.codec().samples_per_frame()?)
        .ok_or_else(|| Error::Overloaded("Fish Audio Drama 3 PCM sample budget overflow".into()))
}

impl FishDrama3StateCoordinator {
    pub(super) fn new() -> Self {
        Self {
            states: Arc::new(Mutex::new(HashMap::new())),
            rows: Mutex::new(HashMap::new()),
        }
    }

    /// Defers a codec quantum's candidate until the core decides.
    pub(super) fn stage_codec(
        &self,
        plan_id: PlanId,
        session: SessionKey,
        mut lease: ExecutorStateLease<'_, ActiveFishDrama3TtsDecode>,
        update: FishDrama3CodecCommit,
    ) -> Result<()> {
        if lease.session != session || !std::ptr::eq(lease.store, self.states.as_ref()) {
            return Err(Error::InferenceError(
                "Fish Audio Drama 3 codec pending lease crossed sessions".into(),
            ));
        }
        let mut rows = self.rows.lock().map_err(|_| {
            Error::InferenceError("Fish Audio Drama 3 pending mutex poisoned".into())
        })?;
        if rows.contains_key(&plan_id) {
            return Err(Error::InferenceError(
                "Fish Audio Drama 3 codec pending plan already exists".into(),
            ));
        }
        let active = lease.require_state_mut()?;
        let budget = sample_budget(active)?;
        let candidate_samples = active
            .total_audio_samples
            .checked_add(update.candidate.samples().len())
            .ok_or_else(|| {
                Error::Overloaded("Fish Audio Drama 3 PCM candidate length overflow".into())
            })?;
        if candidate_samples > budget || active.audio_samples.capacity() > budget {
            return Err(Error::Overloaded(
                "Fish Audio Drama 3 PCM exceeds its sealed sample budget".into(),
            ));
        }
        // The sealed output budget is allocated once: geometric growth per
        // chunk could exceed admission though the final length fits it.
        if active.collect_audio_samples && active.audio_samples.capacity() < budget {
            active
                .audio_samples
                .try_reserve_exact(budget - active.audio_samples.len())
                .map_err(|error| {
                    Error::Overloaded(format!(
                        "Fish Audio Drama 3 PCM retention allocation failed: {error}"
                    ))
                })?;
        }
        let active = lease.defer()?;
        rows.insert(
            plan_id,
            PendingFishDrama3Row {
                session,
                active,
                update: FishDrama3PendingUpdate::Codec(update),
                decision: None,
            },
        );
        Ok(())
    }

    /// Defers a model quantum's checkpoint until the core decides. A
    /// colliding plan or session aborts the quantum at once.
    pub(super) fn stage_quantum(
        &self,
        plan_id: PlanId,
        session: SessionKey,
        mut lease: ExecutorStateLease<'_, ActiveFishDrama3TtsDecode>,
        checkpoint: FishDrama3QuantumCheckpoint,
        fresh: bool,
    ) -> Result<()> {
        let mut rows = self.rows.lock().map_err(|_| {
            Error::InferenceError("Fish Audio Drama 3 pending mutex poisoned".into())
        })?;
        if lease.session != session
            || !std::ptr::eq(lease.store, self.states.as_ref())
            || rows.contains_key(&plan_id)
        {
            if fresh {
                lease.discard_state();
            } else {
                let active = lease.require_state_mut()?;
                let model = active.model.clone();
                model.abort_quantum(&mut active.state, checkpoint)?;
                lease.mark_clean();
            }
            return Err(Error::InferenceError(
                "Fish Audio Drama 3 pending quantum identity collision".into(),
            ));
        }
        let active = lease.defer()?;
        rows.insert(
            plan_id,
            PendingFishDrama3Row {
                session,
                active,
                update: FishDrama3PendingUpdate::Quantum { checkpoint, fresh },
                decision: None,
            },
        );
        Ok(())
    }

    fn restore(&self, row: PendingFishDrama3Row, commit: bool) -> Result<()> {
        let mut states = self
            .states
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if !matches!(states.get(&row.session), Some(ExecutorStateSlot::InFlight { variant }) if *variant == row.active.variant)
        {
            return Err(Error::InferenceError(
                "Fish Audio Drama 3 pending row lost its ownership marker".into(),
            ));
        }
        let mut active = row.active;
        match row.update {
            FishDrama3PendingUpdate::Codec(update) if commit => {
                let samples = active.stream.commit(update.candidate)?;
                active.total_audio_samples += samples.len();
                if active.collect_audio_samples {
                    active.audio_samples.extend_from_slice(&samples);
                }
                active.stream_sequence = update.stream_sequence;
                active.codec_ms += update.codec_ms;
                if active.first_audio_ms.is_none() {
                    active.first_audio_ms = update.first_audio_ms;
                }
            }
            FishDrama3PendingUpdate::Codec(_) => {}
            FishDrama3PendingUpdate::Quantum { checkpoint, fresh } => {
                let model = active.model.clone();
                if commit {
                    model.commit_quantum(&mut active.state, &checkpoint)?;
                    active.last_frames = active.state.frames();
                } else if fresh {
                    states.remove(&row.session);
                    return Ok(());
                } else {
                    model.abort_quantum(&mut active.state, checkpoint)?;
                }
            }
        }
        states.insert(
            row.session,
            ExecutorStateSlot::Ready {
                variant: active.variant,
                state: active,
            },
        );
        Ok(())
    }

    pub(super) fn abort_matching(&self, predicate: impl Fn(&SessionKey) -> bool) -> Result<()> {
        let mut rows = self.rows.lock().map_err(|_| {
            Error::InferenceError("Fish Audio Drama 3 pending mutex poisoned".into())
        })?;
        let ids = rows
            .iter()
            .filter_map(|(id, row)| predicate(&row.session).then_some(*id))
            .collect::<Vec<_>>();
        for id in ids {
            self.restore(rows.remove(&id).expect("pending Drama 3 row exists"), false)?;
        }
        Ok(())
    }

    /// Aborts every pending row of `variant` (an unload purges them).
    pub(super) fn abort_variant(&self, variant: ModelVariant) -> Result<()> {
        let mut rows = self.rows.lock().map_err(|_| {
            Error::InferenceError("Fish Audio Drama 3 pending mutex poisoned".into())
        })?;
        let ids = rows
            .iter()
            .filter_map(|(id, row)| (row.active.variant == variant).then_some(*id))
            .collect::<Vec<_>>();
        for id in ids {
            self.restore(rows.remove(&id).expect("pending Drama 3 row exists"), false)?;
        }
        Ok(())
    }

    pub(super) fn has_prepared(&self, plan_id: PlanId) -> bool {
        self.rows
            .lock()
            .ok()
            .is_some_and(|rows| rows.get(&plan_id).is_some_and(|row| row.decision.is_some()))
    }
}

impl PendingQuantumFinalizer for FishDrama3StateCoordinator {
    fn contains(&self, plan_id: PlanId, session: &SessionKey) -> bool {
        self.rows.lock().ok().is_some_and(|rows| {
            rows.get(&plan_id)
                .is_some_and(|row| &row.session == session && row.decision.is_none())
        })
    }

    fn prepare(
        &self,
        plan_id: PlanId,
        session: &SessionKey,
        decision: PendingQuantumDecision,
    ) -> Result<PendingQuantumFinalizeStatus> {
        let mut rows = self.rows.lock().map_err(|_| {
            Error::InferenceError("Fish Audio Drama 3 pending mutex poisoned".into())
        })?;
        let Some(row) = rows.get_mut(&plan_id) else {
            return Ok(PendingQuantumFinalizeStatus::NotFound);
        };
        if &row.session != session || row.decision.is_some() {
            return Err(Error::InferenceError(
                "Fish Audio Drama 3 prepare crossed session or decision".into(),
            ));
        }
        row.decision = Some(decision);
        Ok(PendingQuantumFinalizeStatus::Finalized)
    }

    fn publish(
        &self,
        plan_id: PlanId,
        session: &SessionKey,
    ) -> Result<PendingQuantumFinalizeStatus> {
        let mut rows = self.rows.lock().map_err(|_| {
            Error::InferenceError("Fish Audio Drama 3 pending mutex poisoned".into())
        })?;
        let Some(row) = rows.get(&plan_id) else {
            return Ok(PendingQuantumFinalizeStatus::NotFound);
        };
        if &row.session != session || row.decision.is_none() {
            return Err(Error::InferenceError(
                "Fish Audio Drama 3 publish requires its exact prepared session".into(),
            ));
        }
        let row = rows.remove(&plan_id).expect("prepared Drama 3 row exists");
        let commit = row.decision == Some(PendingQuantumDecision::Commit);
        self.restore(row, commit)?;
        Ok(PendingQuantumFinalizeStatus::Finalized)
    }

    fn discard(&self, plan_id: PlanId, session: &SessionKey) {
        let mut rows = self.rows.lock().unwrap_or_else(|error| error.into_inner());
        if rows
            .get(&plan_id)
            .is_some_and(|row| &row.session == session)
        {
            let row = rows.remove(&plan_id).expect("pending Drama 3 row exists");
            let _ = self.restore(row, false);
        }
    }
}
