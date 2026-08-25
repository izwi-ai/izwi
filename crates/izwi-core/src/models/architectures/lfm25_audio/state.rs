//! Dormant retained-state foundation for LFM2.5 Audio.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, TensorStateArena,
};
use crate::error::{Error, Result};
use crate::models::architectures::lfm2::backbone::Lfm2ShortConvRuntimeState;
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PhysicalPagedKvCheckpoint};

static NEXT_LFM25_AUDIO_RETAINED_STATE_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lfm25AudioRetainedMode {
    Asr,
    Tts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lfm25AudioRetainedSubphase {
    Main,
    DepthformerFrame,
}

pub(crate) struct Lfm25AudioRetainedState {
    state_id: u64,
    next_quantum_nonce: u64,
    active_quantum: Option<u64>,
    mode: Lfm25AudioRetainedMode,
    max_depthformer_steps: usize,
    main_position: usize,
    depthformer_step: usize,
    tensor_sequence: Option<PhysicalStateSequenceId>,
    main_view_id: Option<u64>,
    pub(super) shortconv: Lfm2ShortConvRuntimeState,
}

pub(crate) struct Lfm25AudioRetainedCheckpoint {
    state_id: u64,
    quantum_nonce: u64,
    subphase: Lfm25AudioRetainedSubphase,
    main_view_id: u64,
    depthformer_view_id: Option<u64>,
    main_kv: PhysicalPagedKvCheckpoint,
    depthformer_kv: Option<PhysicalPagedKvCheckpoint>,
    shortconv: Lfm2ShortConvRuntimeState,
    main_position: usize,
    depthformer_step: usize,
}

impl Lfm25AudioRetainedState {
    pub(super) fn new(
        mode: Lfm25AudioRetainedMode,
        shortconv: Lfm2ShortConvRuntimeState,
        max_depthformer_steps: usize,
    ) -> Self {
        Self {
            state_id: NEXT_LFM25_AUDIO_RETAINED_STATE_ID.fetch_add(1, Ordering::Relaxed),
            next_quantum_nonce: 1,
            active_quantum: None,
            mode,
            max_depthformer_steps,
            main_position: 0,
            depthformer_step: 0,
            tensor_sequence: None,
            main_view_id: None,
            shortconv,
        }
    }

    pub(crate) const fn main_position(&self) -> usize {
        self.main_position
    }

    pub(crate) const fn depthformer_step(&self) -> usize {
        self.depthformer_step
    }

    pub(crate) fn begin_main_quantum(
        &mut self,
        main: &PhysicalPagedKvCache,
    ) -> Result<Lfm25AudioRetainedCheckpoint> {
        self.begin_quantum(Lfm25AudioRetainedSubphase::Main, main, None)
    }

    pub(crate) fn begin_depthformer_quantum(
        &mut self,
        main: &PhysicalPagedKvCache,
        depthformer: &PhysicalPagedKvCache,
    ) -> Result<Lfm25AudioRetainedCheckpoint> {
        self.begin_quantum(
            Lfm25AudioRetainedSubphase::DepthformerFrame,
            main,
            Some(depthformer),
        )
    }

    fn begin_quantum(
        &mut self,
        subphase: Lfm25AudioRetainedSubphase,
        main: &PhysicalPagedKvCache,
        depthformer: Option<&PhysicalPagedKvCache>,
    ) -> Result<Lfm25AudioRetainedCheckpoint> {
        if self.active_quantum.is_some() {
            return Err(Error::InferenceError(
                "an LFM2.5 Audio retained quantum is already active".into(),
            ));
        }
        let bind_main = self.validate_caches(subphase, main, depthformer)?;
        let nonce =
            begin_quantum_authority(&mut self.active_quantum, &mut self.next_quantum_nonce)?;
        if bind_main {
            self.main_view_id = Some(main.view_id());
        }
        Ok(Lfm25AudioRetainedCheckpoint {
            state_id: self.state_id,
            quantum_nonce: nonce,
            subphase,
            main_view_id: main.view_id(),
            depthformer_view_id: depthformer.map(PhysicalPagedKvCache::view_id),
            main_kv: main.logical_checkpoint(),
            depthformer_kv: depthformer.map(PhysicalPagedKvCache::logical_checkpoint),
            shortconv: self.shortconv.clone(),
            main_position: self.main_position,
            depthformer_step: self.depthformer_step,
        })
    }

    pub(crate) fn commit_quantum(
        &mut self,
        main: &PhysicalPagedKvCache,
        depthformer: Option<&PhysicalPagedKvCache>,
        checkpoint: &Lfm25AudioRetainedCheckpoint,
    ) -> Result<()> {
        self.authenticate(checkpoint, main, depthformer)?;
        if main.context_len() != self.shortconv.cursor() as usize
            || (checkpoint.subphase == Lfm25AudioRetainedSubphase::DepthformerFrame
                && main.context_len() != checkpoint.main_position)
        {
            return Err(Error::InferenceError(
                "LFM2.5 Audio main KV and ShortConv clocks diverged".into(),
            ));
        }
        if depthformer.is_some_and(|cache| cache.context_len() > self.max_depthformer_steps) {
            return Err(Error::InferenceError(
                "LFM2.5 Audio Depthformer clock exceeded its codebook bound".into(),
            ));
        }
        validate_subphase_binding(
            self.mode,
            checkpoint.subphase,
            self.depthformer_step,
            self.max_depthformer_steps,
            depthformer.map(PhysicalPagedKvCache::context_len),
            false,
        )?;
        if checkpoint.subphase == Lfm25AudioRetainedSubphase::Main {
            self.main_position = main.context_len();
        } else if let Some(depthformer) = depthformer {
            self.depthformer_step = depthformer.context_len();
        }
        finish_quantum_authority(&mut self.active_quantum, checkpoint.quantum_nonce)?;
        Ok(())
    }

    pub(crate) fn rollback_quantum(
        &mut self,
        main: &mut PhysicalPagedKvCache,
        mut depthformer: Option<&mut PhysicalPagedKvCache>,
        checkpoint: &Lfm25AudioRetainedCheckpoint,
    ) -> Result<()> {
        self.authenticate(checkpoint, main, depthformer.as_deref())?;
        let mut failures = Vec::new();
        if let Err(error) = main.restore_logical_checkpoint(checkpoint.main_kv.clone()) {
            failures.push(format!("main KV: {error}"));
        }
        if let (Some(cache), Some(saved)) = (
            depthformer.as_deref_mut(),
            checkpoint.depthformer_kv.as_ref(),
        ) {
            if let Err(error) = cache.restore_logical_checkpoint(saved.clone()) {
                failures.push(format!("Depthformer KV: {error}"));
            }
        }
        self.shortconv = checkpoint.shortconv.clone();
        self.main_position = checkpoint.main_position;
        self.depthformer_step = checkpoint.depthformer_step;
        if failures.is_empty() {
            finish_quantum_authority(&mut self.active_quantum, checkpoint.quantum_nonce)?;
            Ok(())
        } else {
            Err(Error::InferenceError(format!(
                "LFM2.5 Audio retained rollback failed: {}",
                failures.join("; ")
            )))
        }
    }

    /// Depthformer is a codebook-local clock. Reset it before opening the
    /// frame quantum; rollback then returns to that zero-step frame boundary.
    pub(crate) fn reset_depthformer_frame(
        &mut self,
        depthformer: &mut PhysicalPagedKvCache,
    ) -> Result<()> {
        if self.mode != Lfm25AudioRetainedMode::Tts || self.active_quantum.is_some() {
            return Err(Error::InferenceError(
                "LFM2.5 Audio Depthformer reset is outside an idle TTS frame boundary".into(),
            ));
        }
        depthformer.reset_invocation()?;
        self.depthformer_step = 0;
        Ok(())
    }

    pub(crate) fn bind_tensor_sequence(&mut self, sequence: u64) -> Result<()> {
        let sequence = PhysicalStateSequenceId::new(sequence)?;
        if self.tensor_sequence.is_some_and(|bound| bound != sequence) {
            return Err(Error::InferenceError(
                "LFM2.5 Audio tensor-state sequence identity changed".into(),
            ));
        }
        self.tensor_sequence = Some(sequence);
        Ok(())
    }

    pub(crate) fn restore_shortconv(&mut self, arena: &TensorStateArena) -> Result<()> {
        let sequence = self.tensor_sequence.ok_or_else(|| {
            Error::InferenceError("LFM2.5 Audio has no tensor-state sequence".into())
        })?;
        self.shortconv.restore(arena, sequence)
    }

    pub(crate) fn stage_shortconv(
        &mut self,
        arena: &TensorStateArena,
        transaction: u64,
    ) -> Result<()> {
        self.shortconv.stage(
            arena,
            PhysicalStateTransactionId::new(transaction)?,
            self.shortconv.cursor(),
        )
    }

    fn validate_caches(
        &self,
        subphase: Lfm25AudioRetainedSubphase,
        main: &PhysicalPagedKvCache,
        depthformer: Option<&PhysicalPagedKvCache>,
    ) -> Result<bool> {
        if main.context_len() != self.main_position
            || self.shortconv.cursor() as usize != self.main_position
        {
            return Err(Error::InferenceError(
                "LFM2.5 Audio retained main clocks are not aligned".into(),
            ));
        }
        validate_subphase_binding(
            self.mode,
            subphase,
            self.depthformer_step,
            self.max_depthformer_steps,
            depthformer.map(PhysicalPagedKvCache::context_len),
            true,
        )?;
        validate_exact_view(self.main_view_id, main.view_id(), "main")
    }

    fn authenticate(
        &self,
        checkpoint: &Lfm25AudioRetainedCheckpoint,
        main: &PhysicalPagedKvCache,
        depthformer: Option<&PhysicalPagedKvCache>,
    ) -> Result<()> {
        if !checkpoint_identity_matches(
            self.state_id,
            self.active_quantum,
            checkpoint.state_id,
            checkpoint.quantum_nonce,
            checkpoint.main_view_id,
            main.view_id(),
            checkpoint.depthformer_view_id,
            depthformer.map(PhysicalPagedKvCache::view_id),
        ) || self.main_view_id != Some(main.view_id())
        {
            return Err(Error::InferenceError(
                "LFM2.5 Audio checkpoint is stale, foreign, or crossed cache authority".into(),
            ));
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn checkpoint_identity_matches(
    state_id: u64,
    active_quantum: Option<u64>,
    checkpoint_state_id: u64,
    checkpoint_nonce: u64,
    checkpoint_main_view: u64,
    main_view: u64,
    checkpoint_depth_view: Option<u64>,
    depth_view: Option<u64>,
) -> bool {
    checkpoint_state_id == state_id
        && active_quantum == Some(checkpoint_nonce)
        && checkpoint_main_view == main_view
        && checkpoint_depth_view == depth_view
}

fn validate_exact_view(bound: Option<u64>, candidate: u64, label: &str) -> Result<bool> {
    match bound {
        Some(expected) if expected != candidate => Err(Error::InferenceError(format!(
            "LFM2.5 Audio {label} cache view identity changed"
        ))),
        Some(_) => Ok(false),
        None => Ok(true),
    }
}

fn begin_quantum_authority(active: &mut Option<u64>, next: &mut u64) -> Result<u64> {
    if active.is_some() {
        return Err(Error::InferenceError(
            "an LFM2.5 Audio retained quantum is already active".into(),
        ));
    }
    let nonce = *next;
    *next = next
        .checked_add(1)
        .ok_or_else(|| Error::InferenceError("LFM2.5 Audio quantum nonce overflowed".into()))?;
    *active = Some(nonce);
    Ok(nonce)
}

fn finish_quantum_authority(active: &mut Option<u64>, nonce: u64) -> Result<()> {
    if *active != Some(nonce) {
        return Err(Error::InferenceError(
            "LFM2.5 Audio quantum authority is stale or foreign".into(),
        ));
    }
    *active = None;
    Ok(())
}

fn validate_subphase_binding(
    mode: Lfm25AudioRetainedMode,
    subphase: Lfm25AudioRetainedSubphase,
    expected_step: usize,
    max_steps: usize,
    cache_step: Option<usize>,
    require_exact_step: bool,
) -> Result<()> {
    match (mode, subphase, cache_step) {
        (_, Lfm25AudioRetainedSubphase::Main, None) => Ok(()),
        (Lfm25AudioRetainedMode::Asr, Lfm25AudioRetainedSubphase::DepthformerFrame, _) => Err(
            Error::InvalidInput("LFM2.5 Audio ASR state cannot enter a Depthformer frame".into()),
        ),
        (Lfm25AudioRetainedMode::Tts, Lfm25AudioRetainedSubphase::DepthformerFrame, Some(step))
            if step > max_steps =>
        {
            Err(Error::InferenceError(
                "LFM2.5 Audio Depthformer clock exceeded its bound".into(),
            ))
        }
        (Lfm25AudioRetainedMode::Tts, Lfm25AudioRetainedSubphase::DepthformerFrame, Some(step))
            if !require_exact_step || step == expected_step =>
        {
            Ok(())
        }
        (_, Lfm25AudioRetainedSubphase::Main, Some(_)) => Err(Error::InvalidInput(
            "LFM2.5 Audio main-only quantum cannot bind Depthformer KV".into(),
        )),
        _ => Err(Error::InferenceError(
            "LFM2.5 Audio Depthformer frame is missing or its clock is not aligned".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        begin_quantum_authority, checkpoint_identity_matches, finish_quantum_authority,
        validate_exact_view, validate_subphase_binding, Lfm25AudioRetainedMode,
        Lfm25AudioRetainedSubphase,
    };

    #[test]
    fn retained_modes_keep_depthformer_authority_explicit() {
        assert_ne!(Lfm25AudioRetainedMode::Asr, Lfm25AudioRetainedMode::Tts);
    }

    #[test]
    fn depthformer_clock_is_independent_bounded_and_mode_authenticated() {
        assert!(validate_subphase_binding(
            Lfm25AudioRetainedMode::Asr,
            Lfm25AudioRetainedSubphase::Main,
            0,
            0,
            None,
            true
        )
        .is_ok());
        assert!(validate_subphase_binding(
            Lfm25AudioRetainedMode::Tts,
            Lfm25AudioRetainedSubphase::Main,
            3,
            8,
            None,
            true
        )
        .is_ok());
        assert!(validate_subphase_binding(
            Lfm25AudioRetainedMode::Tts,
            Lfm25AudioRetainedSubphase::DepthformerFrame,
            3,
            8,
            Some(3),
            true
        )
        .is_ok());
        assert!(validate_subphase_binding(
            Lfm25AudioRetainedMode::Tts,
            Lfm25AudioRetainedSubphase::DepthformerFrame,
            3,
            8,
            Some(9),
            false
        )
        .is_err());
        assert!(validate_subphase_binding(
            Lfm25AudioRetainedMode::Asr,
            Lfm25AudioRetainedSubphase::DepthformerFrame,
            0,
            8,
            Some(0),
            true
        )
        .is_err());
    }

    #[test]
    fn checkpoint_identity_rejects_stale_foreign_and_crossed_views() {
        assert!(checkpoint_identity_matches(
            7,
            Some(3),
            7,
            3,
            11,
            11,
            Some(13),
            Some(13)
        ));
        assert!(!checkpoint_identity_matches(
            7,
            Some(4),
            7,
            3,
            11,
            11,
            Some(13),
            Some(13)
        ));
        assert!(!checkpoint_identity_matches(
            7,
            Some(3),
            8,
            3,
            11,
            11,
            Some(13),
            Some(13)
        ));
        assert!(!checkpoint_identity_matches(
            7,
            Some(3),
            7,
            3,
            11,
            12,
            Some(13),
            Some(13)
        ));
    }

    #[test]
    fn main_prefill_authority_supports_commit_and_retry_after_rejection() {
        let mut active = None;
        let mut next = 1;
        let nonce = begin_quantum_authority(&mut active, &mut next).unwrap();
        assert!(begin_quantum_authority(&mut active, &mut next).is_err());
        assert!(finish_quantum_authority(&mut active, nonce + 1).is_err());
        assert_eq!(active, Some(nonce));
        finish_quantum_authority(&mut active, nonce).unwrap();
        assert_eq!(active, None);

        let retry = begin_quantum_authority(&mut active, &mut next).unwrap();
        finish_quantum_authority(&mut active, retry).unwrap();
    }

    #[test]
    fn cache_view_binding_is_permanent_even_at_the_same_clock() {
        assert_eq!(validate_exact_view(None, 11, "main").unwrap(), true);
        assert_eq!(validate_exact_view(Some(11), 11, "main").unwrap(), false);
        assert!(validate_exact_view(Some(11), 12, "main").is_err());
    }

    #[test]
    fn rejected_begin_leaves_main_authority_unbound_and_retryable() {
        let mut bound = None;
        let mut active = None;
        let mut next = 1;
        assert!(validate_subphase_binding(
            Lfm25AudioRetainedMode::Tts,
            Lfm25AudioRetainedSubphase::DepthformerFrame,
            0,
            8,
            None,
            true,
        )
        .is_err());
        assert_eq!(bound, None);
        assert_eq!(active, None);
        assert_eq!(next, 1);

        let should_bind = validate_exact_view(bound, 11, "main").unwrap();
        let nonce = begin_quantum_authority(&mut active, &mut next).unwrap();
        if should_bind {
            bound = Some(11);
        }
        assert_eq!(bound, Some(11));
        finish_quantum_authority(&mut active, nonce).unwrap();
    }

    #[test]
    fn depthformer_frame_two_accepts_a_fresh_invocation_view() {
        assert!(checkpoint_identity_matches(
            7,
            Some(1),
            7,
            1,
            11,
            11,
            Some(13),
            Some(13)
        ));
        assert!(checkpoint_identity_matches(
            7,
            Some(2),
            7,
            2,
            11,
            11,
            Some(14),
            Some(14)
        ));
        assert!(!checkpoint_identity_matches(
            7,
            Some(2),
            7,
            2,
            11,
            11,
            Some(14),
            Some(13)
        ));
    }
}
