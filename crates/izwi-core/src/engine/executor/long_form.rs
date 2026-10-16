//! Long-form ASR runs one chunk per dispatch. The run between chunks is
//! host state kept here per session; a chunk's new state stays pending until
//! the core commits that dispatch, so a rolled-back dispatch leaves the
//! committed run (and its stream position) untouched.
use super::*;

use super::audio::ChunkedAsrRun;

/// The committed state of one long-form request between its chunks.
#[derive(Clone)]
pub(super) struct LongFormAsrState {
    pub(super) run: ChunkedAsrRun,
    /// Stream sequence of the next staged output.
    pub(super) sequence: usize,
    pub(super) samples: Arc<[f32]>,
    pub(super) sample_rate: u32,
    /// Audio decode time of the first dispatch, reported with the result.
    pub(super) audio_decode_ms: f64,
}

struct PendingChunk {
    session: SessionKey,
    state: LongFormAsrState,
    decision: Option<PendingQuantumDecision>,
}

#[derive(Default)]
pub(super) struct LongFormAsrRuns {
    committed: Mutex<HashMap<SessionKey, LongFormAsrState>>,
    pending: Mutex<HashMap<PlanId, PendingChunk>>,
}

impl LongFormAsrRuns {
    /// The committed run of `session`, if a chunk of it has committed.
    pub(super) fn committed(&self, session: &SessionKey) -> Option<LongFormAsrState> {
        self.committed
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(session)
            .cloned()
    }

    /// Hold `state` until the core commits or aborts `plan_id`.
    pub(super) fn stage(
        &self,
        plan_id: PlanId,
        session: SessionKey,
        state: LongFormAsrState,
    ) -> Result<()> {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if pending.contains_key(&plan_id) {
            return Err(Error::InferenceError(
                "long-form ASR chunk plan was staged twice".into(),
            ));
        }
        pending.insert(
            plan_id,
            PendingChunk {
                session,
                state,
                decision: None,
            },
        );
        Ok(())
    }

    /// Drop every run and staged chunk of `request_id`.
    pub(super) fn release(&self, request_id: &str) {
        self.committed
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .retain(|session, _| session.request_id != request_id);
        self.pending
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .retain(|_, chunk| chunk.session.request_id != request_id);
    }

    /// Drop the run and staged chunks of exactly `session`.
    pub(super) fn release_session(&self, session: &SessionKey) {
        self.committed
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .remove(session);
        self.pending
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .retain(|_, chunk| &chunk.session != session);
    }

    pub(super) fn has_prepared(&self, plan_id: PlanId) -> bool {
        self.pending
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(&plan_id)
            .is_some_and(|chunk| chunk.decision.is_some())
    }
}

impl PendingQuantumFinalizer for LongFormAsrRuns {
    fn contains(&self, plan_id: PlanId, session: &SessionKey) -> bool {
        self.pending
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(&plan_id)
            .is_some_and(|chunk| &chunk.session == session && chunk.decision.is_none())
    }

    fn prepare(
        &self,
        plan_id: PlanId,
        session: &SessionKey,
        decision: PendingQuantumDecision,
    ) -> Result<PendingQuantumFinalizeStatus> {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let Some(chunk) = pending.get_mut(&plan_id) else {
            return Ok(PendingQuantumFinalizeStatus::NotFound);
        };
        if &chunk.session != session || chunk.decision.is_some() {
            return Err(Error::InferenceError(
                "long-form ASR chunk prepare crossed its session or decision".into(),
            ));
        }
        chunk.decision = Some(decision);
        Ok(PendingQuantumFinalizeStatus::Finalized)
    }

    fn publish(
        &self,
        plan_id: PlanId,
        session: &SessionKey,
    ) -> Result<PendingQuantumFinalizeStatus> {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let Some(chunk) = pending.get(&plan_id) else {
            return Ok(PendingQuantumFinalizeStatus::NotFound);
        };
        if &chunk.session != session || chunk.decision.is_none() {
            return Err(Error::InferenceError(
                "long-form ASR chunk publish requires its exact prepared session".into(),
            ));
        }
        let chunk = pending.remove(&plan_id).expect("prepared chunk exists");
        drop(pending);
        if chunk.decision == Some(PendingQuantumDecision::Commit) {
            self.committed
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .insert(chunk.session, chunk.state);
        }
        Ok(PendingQuantumFinalizeStatus::Finalized)
    }

    fn discard(&self, plan_id: PlanId, session: &SessionKey) {
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if pending
            .get(&plan_id)
            .is_some_and(|chunk| &chunk.session == session)
        {
            pending.remove(&plan_id);
        }
    }
}
