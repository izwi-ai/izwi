//! Dormant retained-state foundation for LFM2.5 Audio.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, TensorStateArena,
};
use crate::error::{Error, Result};
use crate::models::architectures::lfm2::backbone::Lfm2ShortConvRuntimeState;
use crate::models::shared::attention::physical::{
    PhysicalPagedKvCache, PhysicalPagedKvCheckpoint, PhysicalPagedKvSequenceAuthority,
};

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
    main_cache_authority: Option<PhysicalPagedKvSequenceAuthority>,
    pub(super) shortconv: Lfm2ShortConvRuntimeState,
}

pub(crate) struct Lfm25AudioRetainedCheckpoint {
    state_id: u64,
    quantum_nonce: u64,
    subphase: Lfm25AudioRetainedSubphase,
    prior_main_cache_authority: Option<PhysicalPagedKvSequenceAuthority>,
    main_cache_authority: PhysicalPagedKvSequenceAuthority,
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
            main_cache_authority: None,
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
        self.validate_caches(subphase, main, depthformer)?;
        let nonce =
            begin_quantum_authority(&mut self.active_quantum, &mut self.next_quantum_nonce)?;
        Ok(Lfm25AudioRetainedCheckpoint {
            state_id: self.state_id,
            quantum_nonce: nonce,
            subphase,
            prior_main_cache_authority: self.main_cache_authority,
            main_cache_authority: main.sequence_authority(),
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
        install_cache_authority_after_commit(
            &mut self.main_cache_authority,
            checkpoint.prior_main_cache_authority,
            checkpoint.main_cache_authority,
        )?;
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

    pub(crate) fn authenticate_main_quantum(
        &self,
        main: &PhysicalPagedKvCache,
        checkpoint: &Lfm25AudioRetainedCheckpoint,
    ) -> Result<()> {
        if checkpoint.subphase != Lfm25AudioRetainedSubphase::Main {
            return Err(Error::InferenceError(
                "LFM2.5 Audio ASR received a non-main retained checkpoint".into(),
            ));
        }
        self.authenticate(checkpoint, main, None)
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
    ) -> Result<()> {
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
        validate_exact_cache_authority(self.main_cache_authority, main.sequence_authority(), "main")
            .map(|_| ())
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
        ) || checkpoint.main_cache_authority != main.sequence_authority()
            || checkpoint.prior_main_cache_authority != self.main_cache_authority
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

fn validate_exact_cache_authority(
    bound: Option<PhysicalPagedKvSequenceAuthority>,
    candidate: PhysicalPagedKvSequenceAuthority,
    label: &str,
) -> Result<bool> {
    match bound {
        Some(expected) if expected != candidate => Err(Error::InferenceError(format!(
            "LFM2.5 Audio {label} cache sequence authority changed"
        ))),
        Some(_) => Ok(false),
        None => Ok(true),
    }
}

fn install_cache_authority_after_commit(
    bound: &mut Option<PhysicalPagedKvSequenceAuthority>,
    prior: Option<PhysicalPagedKvSequenceAuthority>,
    candidate: PhysicalPagedKvSequenceAuthority,
) -> Result<()> {
    if *bound != prior || prior.is_some_and(|authority| authority != candidate) {
        return Err(Error::InferenceError(
            "LFM2.5 Audio main cache sequence authority changed before commit".into(),
        ));
    }
    if bound.is_none() {
        *bound = Some(candidate);
    }
    Ok(())
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
    use std::sync::Arc;

    use super::{
        begin_quantum_authority, checkpoint_identity_matches, finish_quantum_authority,
        install_cache_authority_after_commit, validate_exact_cache_authority,
        validate_subphase_binding, Lfm25AudioRetainedMode, Lfm25AudioRetainedSubphase,
    };
    use crate::backends::kv::{CpuKvArena, KvArenaConfig, KvLayerConfig};
    use crate::backends::BackendKind;
    use crate::engine::ModelInstanceId;
    use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
    use crate::models::shared::attention::physical::PhysicalPagedKvCache;
    use candle_core::DType;

    fn cache_authority_views() -> (
        PhysicalPagedKvCache,
        PhysicalPagedKvCache,
        PhysicalPagedKvCache,
    ) {
        let arena_id = KvArenaId {
            model_instance: ModelInstanceId::new(25),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let binding = KvLayerBinding {
            model_layer: 0,
            physical_layer: 0,
        };
        let arena = Arc::new(
            CpuKvArena::new(KvArenaConfig {
                id: arena_id,
                group,
                page_tokens: 4,
                capacity_pages: 2,
                growth: None,
                dtype: DType::F32,
                layers: vec![KvLayerConfig {
                    binding,
                    num_kv_heads: 1,
                    key_head_dim: 2,
                    value_head_dim: 2,
                }],
            })
            .expect("test KV arena"),
        );
        let block = |index| CacheBlockRef {
            arena: arena_id,
            group,
            index,
            slot_generation: 1,
        };
        let cache = |blocks| {
            PhysicalPagedKvCache::new(arena.clone(), vec![binding], blocks, 0)
                .expect("test physical cache")
        };
        (
            cache(vec![block(0)]),
            cache(vec![block(0)]),
            cache(vec![block(1)]),
        )
    }

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
    fn reconstructed_views_keep_sequence_authority_but_reject_foreign_sequences() {
        let (first, reconstructed, foreign) = cache_authority_views();
        assert_ne!(first.view_id(), reconstructed.view_id());
        assert_eq!(
            first.sequence_authority(),
            reconstructed.sequence_authority()
        );
        assert_ne!(first.sequence_authority(), foreign.sequence_authority());

        let authority = first.sequence_authority();
        assert!(validate_exact_cache_authority(None, authority, "main").unwrap());
        assert!(!validate_exact_cache_authority(
            Some(authority),
            reconstructed.sequence_authority(),
            "main"
        )
        .unwrap());
        assert!(validate_exact_cache_authority(
            Some(authority),
            foreign.sequence_authority(),
            "main"
        )
        .is_err());
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

        let (main, _, _) = cache_authority_views();
        let authority = main.sequence_authority();
        let should_bind = validate_exact_cache_authority(bound, authority, "main").unwrap();
        let nonce = begin_quantum_authority(&mut active, &mut next).unwrap();
        if should_bind {
            install_cache_authority_after_commit(&mut bound, None, authority).unwrap();
        }
        assert_eq!(bound, Some(authority));
        finish_quantum_authority(&mut active, nonce).unwrap();
    }

    #[test]
    fn first_quantum_rollback_leaves_fresh_sequence_authority_retryable() {
        let (first, _, fresh) = cache_authority_views();
        let first_authority = first.sequence_authority();
        let fresh_authority = fresh.sequence_authority();
        let mut bound = None;

        assert!(validate_exact_cache_authority(bound, first_authority, "main").unwrap());
        // Beginning and then rolling back the first quantum never publishes
        // its provisional sequence authority.
        assert_eq!(bound, None);

        assert!(validate_exact_cache_authority(bound, fresh_authority, "main").unwrap());
        install_cache_authority_after_commit(&mut bound, None, fresh_authority).unwrap();
        assert_eq!(bound, Some(fresh_authority));
        assert!(validate_exact_cache_authority(bound, first_authority, "main").is_err());
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
