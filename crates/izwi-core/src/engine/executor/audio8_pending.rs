//! Audio8 codec candidates remain invisible until the core's commit decision.
use super::*;
use crate::models::architectures::audio8::dac::Audio8DacStreamState;

pub(super) struct Audio8CodecCommit {
    pub(super) codec: Audio8DacStreamState,
    pub(super) samples: Vec<f32>,
    pub(super) stream_sequence: usize,
    pub(super) codec_ms: f64,
    pub(super) first_audio_ms: Option<f64>,
}

enum Audio8PendingUpdate {
    Codec(Audio8CodecCommit),
    Ar(crate::models::architectures::audio8::Audio8RetainedCheckpoint),
}

struct PendingAudio8Codec {
    session: SessionKey,
    active: ActiveAudio8TtsDecode,
    update: Audio8PendingUpdate,
    decision: Option<PendingQuantumDecision>,
}

pub(super) struct Audio8StateCoordinator {
    pub(super) states: Arc<ExecutorStateStore<ActiveAudio8TtsDecode>>,
    rows: Mutex<HashMap<PlanId, PendingAudio8Codec>>,
}

impl Audio8StateCoordinator {
    pub(super) fn new() -> Self {
        Self {
            states: Arc::new(Mutex::new(HashMap::new())),
            rows: Mutex::new(HashMap::new()),
        }
    }

    pub(super) fn stage(
        &self,
        plan_id: PlanId,
        session: SessionKey,
        mut lease: ExecutorStateLease<'_, ActiveAudio8TtsDecode>,
        update: Audio8CodecCommit,
    ) -> Result<()> {
        if lease.session != session || !std::ptr::eq(lease.store, self.states.as_ref()) {
            return Err(Error::InferenceError(
                "Audio8 codec pending lease crossed sessions".into(),
            ));
        }
        let mut rows = self
            .rows
            .lock()
            .map_err(|_| Error::InferenceError("Audio8 pending mutex poisoned".into()))?;
        if rows.contains_key(&plan_id) {
            return Err(Error::InferenceError(
                "Audio8 codec pending plan already exists".into(),
            ));
        }
        let active = lease.require_state_mut()?;
        let sample_budget = active
            .state
            .params()
            .max_frames
            .checked_mul(2048)
            .ok_or_else(|| Error::Overloaded("Audio8 PCM sample budget overflow".into()))?;
        let candidate_samples = active
            .total_audio_samples
            .checked_add(update.samples.len())
            .ok_or_else(|| Error::Overloaded("Audio8 PCM candidate length overflow".into()))?;
        if candidate_samples > sample_budget || active.audio_samples.capacity() > sample_budget {
            return Err(Error::Overloaded(
                "Audio8 PCM exceeds its sealed sample budget".into(),
            ));
        }
        // Allocate the sealed output budget once. Geometric per-chunk growth
        // can exceed admission even when the final logical length fits it.
        if active.collect_audio_samples && active.audio_samples.capacity() < sample_budget {
            active
                .audio_samples
                .try_reserve_exact(sample_budget - active.audio_samples.len())
                .map_err(|error| {
                    Error::Overloaded(format!("Audio8 PCM retention allocation failed: {error}"))
                })?;
        }
        if active.audio_samples.capacity() > sample_budget {
            return Err(Error::Overloaded(
                "Audio8 PCM allocation exceeds its sealed sample budget".into(),
            ));
        }
        let active = lease.defer()?;
        rows.insert(
            plan_id,
            PendingAudio8Codec {
                session,
                active,
                update: Audio8PendingUpdate::Codec(update),
                decision: None,
            },
        );
        Ok(())
    }

    pub(super) fn stage_ar(
        &self,
        plan_id: PlanId,
        session: SessionKey,
        mut lease: ExecutorStateLease<'_, ActiveAudio8TtsDecode>,
        mut checkpoint: crate::models::architectures::audio8::Audio8RetainedCheckpoint,
    ) -> Result<()> {
        let mut rows = self
            .rows
            .lock()
            .map_err(|_| Error::InferenceError("Audio8 pending mutex poisoned".into()))?;
        if lease.session != session
            || !std::ptr::eq(lease.store, self.states.as_ref())
            || rows.contains_key(&plan_id)
        {
            if checkpoint.is_initial() {
                lease.discard_state();
            } else {
                lease
                    .require_state_mut()?
                    .state
                    .rollback_managed_quantum(&mut checkpoint)?;
                lease.mark_clean();
            }
            return Err(Error::InferenceError(
                "Audio8 AR pending identity collision".into(),
            ));
        }
        let active = lease.defer()?;
        rows.insert(
            plan_id,
            PendingAudio8Codec {
                session,
                active,
                update: Audio8PendingUpdate::Ar(checkpoint),
                decision: None,
            },
        );
        Ok(())
    }

    fn restore(&self, row: PendingAudio8Codec, commit: bool) -> Result<()> {
        let mut states = self
            .states
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if !matches!(states.get(&row.session), Some(ExecutorStateSlot::InFlight { variant }) if *variant == row.active.variant)
        {
            return Err(Error::InferenceError(
                "Audio8 pending codec lost its ownership marker".into(),
            ));
        }
        let mut active = row.active;
        match row.update {
            Audio8PendingUpdate::Codec(update) if commit => {
                active.codec = update.codec;
                active.total_audio_samples += update.samples.len();
                if active.collect_audio_samples {
                    active.audio_samples.extend_from_slice(&update.samples);
                }
                active.stream_sequence = update.stream_sequence;
                active.codec_ms += update.codec_ms;
                if active.first_audio_ms.is_none() {
                    active.first_audio_ms = update.first_audio_ms;
                }
            }
            Audio8PendingUpdate::Ar(mut checkpoint) => {
                if commit {
                    active.state.commit_managed_quantum(&mut checkpoint)?;
                    active.last_frames_generated = active.state.frames_generated();
                } else if checkpoint.is_initial() {
                    states.remove(&row.session);
                    return Ok(());
                } else {
                    active.state.rollback_managed_quantum(&mut checkpoint)?;
                }
            }
            Audio8PendingUpdate::Codec(_) => {}
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
        let mut rows = self
            .rows
            .lock()
            .map_err(|_| Error::InferenceError("Audio8 pending mutex poisoned".into()))?;
        let ids = rows
            .iter()
            .filter_map(|(id, row)| predicate(&row.session).then_some(*id))
            .collect::<Vec<_>>();
        for id in ids {
            self.restore(rows.remove(&id).expect("pending Audio8 row exists"), false)?;
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

impl PendingQuantumFinalizer for Audio8StateCoordinator {
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
        let mut rows = self
            .rows
            .lock()
            .map_err(|_| Error::InferenceError("Audio8 pending mutex poisoned".into()))?;
        let Some(row) = rows.get_mut(&plan_id) else {
            return Ok(PendingQuantumFinalizeStatus::NotFound);
        };
        if &row.session != session || row.decision.is_some() {
            return Err(Error::InferenceError(
                "Audio8 codec prepare crossed session or decision".into(),
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
        let mut rows = self
            .rows
            .lock()
            .map_err(|_| Error::InferenceError("Audio8 pending mutex poisoned".into()))?;
        let Some(row) = rows.get(&plan_id) else {
            return Ok(PendingQuantumFinalizeStatus::NotFound);
        };
        if &row.session != session || row.decision.is_none() {
            return Err(Error::InferenceError(
                "Audio8 codec publish requires its exact prepared session".into(),
            ));
        }
        let row = rows.remove(&plan_id).expect("prepared Audio8 row exists");
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
            let row = rows.remove(&plan_id).expect("pending Audio8 row exists");
            let _ = self.restore(row, false);
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::models::architectures::audio8::{Audio8RetainedState, Audio8TtsModel};
    use crate::models::registry::Audio8TtsModelLease;

    pub(crate) fn staged() -> (Audio8StateCoordinator, SessionKey) {
        staged_with_collection(true)
    }

    #[test]
    fn initial_ar_discard_and_rejected_staging_release_unpublished_state() {
        for collision in [false, true] {
            let (coordinator, session) = staged();
            coordinator.discard(17, &session);
            let mut lease = ExecutorStateLease::checkout(
                &coordinator.states,
                session.clone(),
                ModelVariant::Audio8TtsRealisticAir,
                "initial abort",
            )
            .unwrap();
            let (state, checkpoint) = Audio8RetainedState::initial_quantum_for_test();
            lease.require_state_mut().unwrap().state = state;
            lease.mark_dirty();
            if collision {
                let foreign = SessionKey::new(session.request_id.clone(), session.epoch + 1);
                assert!(coordinator
                    .stage_ar(18, foreign, lease, checkpoint)
                    .is_err());
            } else {
                coordinator
                    .stage_ar(18, session.clone(), lease, checkpoint)
                    .unwrap();
                coordinator.discard(18, &session);
            }
            assert!(coordinator.rows.lock().unwrap().is_empty());
            assert!(!coordinator.states.lock().unwrap().contains_key(&session));
        }
    }

    fn staged_with_collection(collect: bool) -> (Audio8StateCoordinator, SessionKey) {
        let coordinator = Audio8StateCoordinator::new();
        let session = SessionKey::new("audio8-pending".into(), 1);
        let mut lease = ExecutorStateLease::checkout(
            &coordinator.states,
            session.clone(),
            ModelVariant::Audio8TtsRealisticAir,
            "Audio8 test",
        )
        .unwrap();
        lease
            .install_state(ActiveAudio8TtsDecode {
                variant: ModelVariant::Audio8TtsRealisticAir,
                model: Audio8TtsModelLease::for_test(Audio8TtsModel::for_test()),
                state: Audio8RetainedState::for_test(),
                last_frames_generated: 0,
                stream_sequence: 3,
                codec: Default::default(),
                audio_samples: if collect { vec![0.25] } else { Vec::new() },
                total_audio_samples: 1,
                collect_audio_samples: collect,
                codec_ms: 2.0,
                execution_started: std::time::Instant::now(),
                first_audio_ms: None,
            })
            .unwrap();
        coordinator
            .stage(
                17,
                session.clone(),
                lease,
                Audio8CodecCommit {
                    codec: Default::default(),
                    samples: vec![0.5, 0.75],
                    stream_sequence: 4,
                    codec_ms: 5.0,
                    first_audio_ms: Some(10.0),
                },
            )
            .unwrap();
        (coordinator, session)
    }

    pub(crate) fn assert_ready(
        coordinator: &Audio8StateCoordinator,
        session: &SessionKey,
        committed: bool,
    ) {
        let mut lease = ExecutorStateLease::checkout(
            &coordinator.states,
            session.clone(),
            ModelVariant::Audio8TtsRealisticAir,
            "Audio8 test",
        )
        .unwrap();
        let state = lease.require_state_mut().unwrap();
        assert!(state.audio_samples.capacity() <= state.state.params().max_frames * 2048);
        assert_eq!(
            state.audio_samples,
            if committed {
                vec![0.25, 0.5, 0.75]
            } else {
                vec![0.25]
            }
        );
        assert_eq!(state.stream_sequence, if committed { 4 } else { 3 });
        assert_eq!(state.codec_ms, if committed { 7.0 } else { 2.0 });
        assert_eq!(state.first_audio_ms, committed.then_some(10.0));
        lease.restore().unwrap();
    }

    #[test]
    fn audio8_stream_only_commit_counts_pcm_without_retaining_waveform_and_abort_is_exact() {
        for commit in [false, true] {
            let (coordinator, session) = staged_with_collection(false);
            coordinator
                .prepare(
                    17,
                    &session,
                    if commit {
                        PendingQuantumDecision::Commit
                    } else {
                        PendingQuantumDecision::Abort
                    },
                )
                .unwrap();
            coordinator.publish(17, &session).unwrap();
            let mut lease = ExecutorStateLease::checkout(
                &coordinator.states,
                session.clone(),
                ModelVariant::Audio8TtsRealisticAir,
                "stream only test",
            )
            .unwrap();
            let active = lease.require_state_mut().unwrap();
            assert_eq!(active.audio_samples.capacity(), 0);
            assert_eq!(active.audio_samples.len(), 0);
            assert_eq!(active.total_audio_samples, if commit { 3 } else { 1 });
            let max = active.state.params().max_frames * 2048;
            active.total_audio_samples = max;
            let error = coordinator
                .stage(
                    18,
                    session.clone(),
                    lease,
                    Audio8CodecCommit {
                        codec: Default::default(),
                        samples: vec![0.0],
                        stream_sequence: 5,
                        codec_ms: 1.0,
                        first_audio_ms: None,
                    },
                )
                .unwrap_err();
            assert!(error.to_string().contains("sealed sample budget"));
            let mut lease = ExecutorStateLease::checkout(
                &coordinator.states,
                session,
                ModelVariant::Audio8TtsRealisticAir,
                "stream only check",
            )
            .unwrap();
            let active = lease.require_state_mut().unwrap();
            assert_eq!(active.total_audio_samples, max);
            assert_eq!(active.audio_samples.capacity(), 0);
            lease.restore().unwrap();
        }
    }

    #[test]
    fn audio8_codec_pcm_budget_is_reserved_once_and_rejects_cumulative_overrun() {
        let (coordinator, session) = staged();
        coordinator
            .prepare(17, &session, PendingQuantumDecision::Commit)
            .unwrap();
        coordinator.publish(17, &session).unwrap();
        let mut lease = ExecutorStateLease::checkout(
            &coordinator.states,
            session.clone(),
            ModelVariant::Audio8TtsRealisticAir,
            "Audio8 test",
        )
        .unwrap();
        let state = lease.require_state_mut().unwrap();
        let budget = state.state.params().max_frames * 2048;
        assert_eq!(state.audio_samples.capacity(), budget);
        let pointer = state.audio_samples.as_ptr();
        coordinator
            .stage(
                18,
                session.clone(),
                lease,
                Audio8CodecCommit {
                    codec: Default::default(),
                    samples: vec![0.0],
                    stream_sequence: 5,
                    codec_ms: 1.0,
                    first_audio_ms: None,
                },
            )
            .unwrap();
        coordinator
            .prepare(18, &session, PendingQuantumDecision::Commit)
            .unwrap();
        coordinator.publish(18, &session).unwrap();
        let mut lease = ExecutorStateLease::checkout(
            &coordinator.states,
            session.clone(),
            ModelVariant::Audio8TtsRealisticAir,
            "Audio8 test",
        )
        .unwrap();
        assert_eq!(
            lease.require_state_mut().unwrap().audio_samples.as_ptr(),
            pointer
        );
        let remaining = budget - lease.require_state_mut().unwrap().audio_samples.len();
        let error = coordinator
            .stage(
                19,
                session.clone(),
                lease,
                Audio8CodecCommit {
                    codec: Default::default(),
                    samples: vec![0.0; remaining + 1],
                    stream_sequence: 6,
                    codec_ms: 1.0,
                    first_audio_ms: None,
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("sealed sample budget"));
        let mut lease = ExecutorStateLease::checkout(
            &coordinator.states,
            session,
            ModelVariant::Audio8TtsRealisticAir,
            "Audio8 test",
        )
        .unwrap();
        let state = lease.require_state_mut().unwrap();
        assert_eq!(state.audio_samples.len(), 4);
        assert_eq!(state.stream_sequence, 5);
        assert_eq!(state.audio_samples.capacity(), budget);
        assert!(!coordinator.contains(19, &lease.session));
        lease.restore().unwrap();
    }

    #[test]
    fn audio8_codec_publication_applies_pcm_and_sequence_once_after_authorization() {
        let (coordinator, session) = staged();
        assert!(coordinator.contains(17, &session));
        coordinator
            .prepare(17, &session, PendingQuantumDecision::Commit)
            .unwrap();
        assert!(matches!(
            coordinator.states.lock().unwrap().get(&session),
            Some(ExecutorStateSlot::InFlight { .. })
        ));
        assert_eq!(
            coordinator.publish(17, &session).unwrap(),
            PendingQuantumFinalizeStatus::Finalized
        );
        assert_ready(&coordinator, &session, true);
        assert_eq!(
            coordinator.publish(17, &session).unwrap(),
            PendingQuantumFinalizeStatus::NotFound
        );
        assert_ready(&coordinator, &session, true);
    }

    #[test]
    fn audio8_codec_abort_and_rejected_commit_restore_unpublished_pcm_and_sequence() {
        for discard in [false, true] {
            let (coordinator, session) = staged();
            coordinator
                .prepare(
                    17,
                    &session,
                    if discard {
                        PendingQuantumDecision::Commit
                    } else {
                        PendingQuantumDecision::Abort
                    },
                )
                .unwrap();
            if discard {
                coordinator.discard(17, &session);
            } else {
                coordinator.publish(17, &session).unwrap();
            }
            assert_ready(&coordinator, &session, false);
        }
    }

    #[test]
    fn audio8_codec_wrong_session_cannot_consume_or_publish_pending_audio() {
        let (coordinator, session) = staged();
        let wrong = SessionKey::new(session.request_id.clone(), session.epoch + 1);
        assert!(coordinator
            .prepare(17, &wrong, PendingQuantumDecision::Commit)
            .is_err());
        coordinator
            .prepare(17, &session, PendingQuantumDecision::Commit)
            .unwrap();
        assert!(coordinator.publish(17, &wrong).is_err());
        coordinator.discard(17, &wrong);
        coordinator.publish(17, &session).unwrap();
        assert_ready(&coordinator, &session, true);
    }

    #[test]
    fn audio8_codec_cleanup_releases_pending_or_prepared_candidate_without_committing() {
        for prepare in [false, true] {
            let (coordinator, session) = staged();
            if prepare {
                coordinator
                    .prepare(17, &session, PendingQuantumDecision::Commit)
                    .unwrap();
            }
            coordinator
                .abort_matching(|candidate| candidate == &session)
                .unwrap();
            assert_ready(&coordinator, &session, false);
            assert!(!coordinator.contains(17, &session));
        }
    }
}
