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
/// | every chat family | no | no | no |
pub(crate) const fn certified_mixed_step(_family: ModelFamily, _backend: BackendKind) -> bool {
    false
}

/// Whether a loaded `family` model publishes its mixed stage on `backend`:
/// the certification table, unless `IZWI_MIXED_STEP` overrides it.
pub(crate) fn mixed_step_enabled(family: ModelFamily, backend: BackendKind) -> bool {
    mixed_step_enabled_with(
        std::env::var(MIXED_STEP_ENV).ok().as_deref(),
        family,
        backend,
    )
}

fn mixed_step_enabled_with(
    override_value: Option<&str>,
    family: ModelFamily,
    backend: BackendKind,
) -> bool {
    match override_value.map(str::trim) {
        Some("1" | "true" | "on") => true,
        Some("0" | "false" | "off") => false,
        Some(value) => {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    value,
                    "Ignoring {MIXED_STEP_ENV} that is not 0 or 1; using the certified table"
                );
            });
            certified_mixed_step(family, backend)
        }
        None => certified_mixed_step(family, backend),
    }
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
                assert!(!mixed_step_enabled_with(None, variant.family(), backend));
            }
        }
    }

    #[test]
    fn the_override_turns_mixed_steps_on_and_off() {
        let (family, backend) = (ModelFamily::Lfm2Chat, BackendKind::Metal);
        for on in ["1", "true", " on "] {
            assert!(mixed_step_enabled_with(Some(on), family, backend), "{on}");
        }
        for off in ["0", "false", "off"] {
            assert!(
                !mixed_step_enabled_with(Some(off), family, backend),
                "{off}"
            );
        }
        assert_eq!(
            mixed_step_enabled_with(Some("maybe"), family, backend),
            certified_mixed_step(family, backend)
        );
    }
}
