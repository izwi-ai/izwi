//! Certified tensor batch widths for continuous chat decode, per model family
//! and backend (EXE-F7).
//!
//! A family's batched decode runs every row through one forward, so the
//! width it may be served at is a property of its kernels on each backend. A
//! width is certified on CPU or Metal when (a) greedy decode at that width
//! picks the width-1 tokens, teacher-forced, within the tolerance the changed
//! op shapes allow, and (b) aggregate decode throughput rises with the width
//! in an interleaved in-process A/B. The evidence is in the ignored
//! real-checkpoint tests named in the table and in
//! `docs/dev/PRODUCTION_SERVING_SUPPORT_MATRIX.md` ("Certified decode batch
//! widths"). Families without that evidence, and every capability other than
//! chat, keep the portable width of two they have always been served at.
//!
//! CUDA is not measured here: its ceiling is the kernel and metadata limit,
//! and its automatic width stays the VRAM-tiered default of eight.

use super::ModelFamily;
use crate::backends::BackendKind;

/// The width every family has been served at on CPU and Metal.
pub(crate) const PORTABLE_BASELINE_WIDTH: usize = 2;

/// The CUDA ceiling: a policy constant, not a kernel or metadata limit and
/// not a measured width.
pub(crate) const CUDA_MAX_WIDTH: usize = 32;

/// The widest decode batch `family` is certified for on `backend`. The
/// engine clamps the automatic and `max_num_seqs` widths to it; an
/// operator's explicit `IZWI_MAX_BATCH_SIZE` may exceed it, with a warning.
///
/// | Family | CPU | Metal | Evidence (ignored real-checkpoint tests) |
/// |---|---|---|---|
/// | LFM2 / LFM2.5 | 8 | 8 | `lfm25_decode_matches_across_batch_widths_on_*`, `lfm25_batch_width_scaling_*` |
/// | Qwen3.5 | 2 | 8 | `qwen35_decode_width_parity_and_throughput_*` |
/// | Qwen3, Qwen3-MoE, Gemma3, Qwen3.6, Qwen3.8 | 2 | 2 | No checkpoint measured |
///
/// Qwen3.5 on the CPU gained nothing past two rows in a repeated run, so it
/// stays at two. Metal's LFM2 small-batch mat-vec serves 2..=8 rows, which
/// bounds that family at eight.
pub(crate) const fn certified_decode_width(family: ModelFamily, backend: BackendKind) -> usize {
    match (backend, family) {
        (BackendKind::Cuda, _) => CUDA_MAX_WIDTH,
        (BackendKind::Cpu | BackendKind::Metal, ModelFamily::Lfm2Chat)
        | (BackendKind::Metal, ModelFamily::Qwen35Chat) => 8,
        _ => PORTABLE_BASELINE_WIDTH,
    }
}

/// The widest certified width of any family on `backend`: the engine-wide
/// tensor width ceiling, and the automatic width on CPU and Metal (each
/// loaded model then clamps to its own family's width).
pub(crate) const fn max_certified_decode_width(backend: BackendKind) -> usize {
    match backend {
        BackendKind::Cuda => CUDA_MAX_WIDTH,
        BackendKind::Metal | BackendKind::Cpu => 8,
    }
}

/// The widest tensor batch an operator may ask for explicitly
/// (`IZWI_MAX_BATCH_SIZE`) on any backend. An explicit width above a family's
/// certified width runs uncertified, with a startup warning, instead of being
/// clamped silently.
pub(crate) const EXPLICIT_WIDTH_CEILING: usize = CUDA_MAX_WIDTH;

/// The rows every capability but chat runs at: the certified engine rows,
/// or an operator's explicit width as asked.
pub(crate) const fn non_chat_row_width(
    backend: BackendKind,
    tensor_width: usize,
    explicit: bool,
) -> usize {
    if explicit {
        tensor_width
    } else {
        engine_row_width(backend, tensor_width)
    }
}

/// The width a model without its own row fit can run at: every capability
/// but chat sizes its state on the engine-wide rows, so on CPU and Metal
/// those rows stay at the portable width (chat sizes its own from its decode
/// width).
pub(crate) const fn engine_row_width(backend: BackendKind, tensor_width: usize) -> usize {
    match backend {
        BackendKind::Cuda => tensor_width,
        BackendKind::Metal | BackendKind::Cpu => {
            if tensor_width < PORTABLE_BASELINE_WIDTH {
                tensor_width
            } else {
                PORTABLE_BASELINE_WIDTH
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BACKENDS: [BackendKind; 3] = [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda];

    #[test]
    fn the_engine_ceiling_is_the_widest_certified_family() {
        for backend in BACKENDS {
            let widest = crate::model::ModelVariant::all()
                .iter()
                .map(|variant| certified_decode_width(variant.family(), backend))
                .max()
                .unwrap();
            assert_eq!(max_certified_decode_width(backend), widest, "{backend:?}");
        }
    }

    #[test]
    fn portable_widths_are_certified_per_family() {
        for backend in [BackendKind::Cpu, BackendKind::Metal] {
            assert_eq!(certified_decode_width(ModelFamily::Lfm2Chat, backend), 8);
            for family in [
                ModelFamily::Qwen3Chat,
                ModelFamily::Gemma3Chat,
                ModelFamily::Qwen35MoeChat,
                ModelFamily::Qwen38Chat,
                ModelFamily::Lfm25Audio,
            ] {
                assert_eq!(
                    certified_decode_width(family, backend),
                    PORTABLE_BASELINE_WIDTH,
                    "{family:?} {backend:?}"
                );
            }
        }
        assert_eq!(
            certified_decode_width(ModelFamily::Qwen35Chat, BackendKind::Metal),
            8
        );
        assert_eq!(
            certified_decode_width(ModelFamily::Qwen35Chat, BackendKind::Cpu),
            PORTABLE_BASELINE_WIDTH
        );
        assert_eq!(
            certified_decode_width(ModelFamily::Gemma3Chat, BackendKind::Cuda),
            CUDA_MAX_WIDTH
        );
    }
}
