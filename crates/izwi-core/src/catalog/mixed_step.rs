//! Certified mixed prefill+decode steps for chat, per model family and
//! backend (Phase 3a).
//!
//! A family whose loaded model can run prefill spans and one-token decode
//! rows in one model call publishes a mixed `chat.step.mixed` stage only
//! where it is certified here: mixed-vs-split logits within the tolerance the
//! changed op shapes allow, teacher-forced greedy agreement, and an
//! interleaved in-process A/B showing the mixed step is not slower. Everyone
//! else keeps the split prefill and decode stages.

use super::ModelFamily;
use crate::backends::BackendKind;

/// Operators override the table with `IZWI_MIXED_STEP=0|1` (A/B runs and a
/// kill switch). A model that cannot run a mixed step never publishes one.
pub(crate) const MIXED_STEP_ENV: &str = "IZWI_MIXED_STEP";

/// Whether `family` is certified for mixed steps on `backend`.
///
/// | Family | CPU | Metal | CUDA |
/// |---|---|---|---|
/// | LFM2 / LFM2.5 | no: parity holds (bit-exact), no measured win | no: parity holds, ITL p95 regresses | no: not run |
/// | Qwen3 | no: no chat checkpoint to measure | no: same | no: not run |
/// | every other chat family | no: no mixed forward | no | no |
///
/// The LFM2 measurements are in `docs/dev/PRODUCTION_SERVING_SUPPORT_MATRIX.md`
/// ("Mixed prefill+decode steps").
pub(crate) const fn certified_mixed_step(_family: ModelFamily, _backend: BackendKind) -> bool {
    false
}

/// Whether a loaded `family` model publishes its mixed stage on `backend`:
/// the certification table, unless `IZWI_MIXED_STEP` overrides it.
pub(crate) fn mixed_step_enabled(family: ModelFamily, backend: BackendKind) -> bool {
    crate::env::flag(MIXED_STEP_ENV).unwrap_or_else(|| certified_mixed_step(family, backend))
}

#[cfg(test)]
mod tests {
    use super::*;

    const BACKENDS: [BackendKind; 3] = [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda];

    #[test]
    fn no_family_is_certified_yet() {
        for backend in BACKENDS {
            for variant in crate::model::ModelVariant::all() {
                assert!(
                    !certified_mixed_step(variant.family(), backend),
                    "{variant:?} {backend:?}"
                );
            }
        }
    }
}
