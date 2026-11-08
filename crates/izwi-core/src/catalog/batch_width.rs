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
//! chat, keep the portable width of two they have always been served at,
//! except Fish Audio Drama 3, whose rows start at one on CPU and Metal.
//!
//! On CUDA the engine and every capability but chat and ASR decode run the
//! automatic eight. Chat's automatic width there is the CUDA ceiling,
//! clamped per family: Qwen3.5, Qwen3.6, Qwen3.8, LFM2 and LFM3 batch every
//! row's recurrent and conv state in one launch per layer (one launch per row
//! before), and Qwen3.6 and Qwen3.8 run their block-FP8 projections on
//! tensor-core kernels whose cost barely grows with the rows, so their steps
//! stay cheap past eight rows and they take the ceiling; every other family
//! keeps eight. ASR decode takes the same ceiling for the families whose
//! continuous decode steps every row in one launch per op (Qwen3-ASR and
//! VibeVoice-ASR through the Qwen3 core, Parakeet's TDT and Nemotron ASR's
//! RNNT joints, Whisper's decoder with one cross-attention call per layer,
//! Voxtral's offline rows and Granite Speech 4.1's retained rows);
//! their encoders, preparation and prefill stay at the engine rows, and each
//! load fits the width to its memory.

use super::ModelFamily;
use crate::backends::BackendKind;

/// The width every family has been served at on CPU and Metal.
pub(crate) const PORTABLE_BASELINE_WIDTH: usize = 2;

/// The CUDA ceiling: a policy constant, not a kernel or metadata limit and
/// not a measured width.
pub(crate) const CUDA_MAX_WIDTH: usize = 32;

/// The automatic CUDA width of the engine rows, and of chat families not
/// certified for wider CUDA batches.
pub(crate) const CUDA_DEFAULT_WIDTH: usize = 8;

/// The widest decode batch `family` is certified for on `backend`. The
/// engine clamps the automatic and `max_num_seqs` widths to it; an
/// operator's explicit `IZWI_MAX_BATCH_SIZE` may exceed it, with a warning.
///
/// | Family | CPU | Metal | CUDA | Evidence (ignored real-checkpoint tests) |
/// |---|---|---|---|---|
/// | LFM2 / LFM2.5 / LFM3 | 8 | 8 | 32 | `lfm25_decode_matches_across_batch_widths_on_*`, `lfm25_batch_width_scaling_*` |
/// | Qwen3.5 | 2 | 8 | 32 | `qwen35_decode_width_parity_and_throughput_*` |
/// | Qwen3.6, Qwen3.8 | 2 | 2 | 32 | H100 greedy parity and serving matrix (support matrix) |
/// | Qwen3, Qwen3-MoE, Gemma3 | 2 | 2 | 8 | No checkpoint measured |
/// | Fish Audio Drama 3 (TTS) | 1 | 1 | 8 | Defaults until its row-invariance and throughput evidence |
/// | Qwen3-ASR, VibeVoice-ASR, Parakeet, Nemotron ASR (decode only) | 2 | 2 | 32 | Unmeasured past 8. H100 (`e8f189691`): Qwen3-ASR and VibeVoice-ASR scaled 2.4–2.9x from 1 to 8 streams, then stayed flat at the width (Qwen3-ASR ~82 audio-s/s); each decode step runs all rows per launch |
/// | Whisper (decode only) | 2 | 2 | 32 | Unmeasured. Its ~316 MB decoder was launch-bound at one row per step; rows now share each launch, one flash-attention call per layer for cross attention, and a row decodes the same alone as beside others (CPU tiny-model parity test) |
/// | Granite Speech 4.1 (decode only) | 2 | 2 | 32 | Unmeasured past one row. Each request held the device for its whole decode; retained rows now step together (one token upload, one RoPE gather, one paged append and attention per layer, one argmax readback for every row), and the real 2B Plus checkpoint emits the atomic route's tokens through spans, scalar steps and a two-row batch on CPU and Metal (`granite_speech_retained_decode_matches_atomic_on_*`) |
/// | Voxtral (offline decode only) | 2 | 2 | 32 | Unmeasured past one row. H100: an offline request held the device for its whole decode (about 6.3 ms a frame), 9.6 audio-s/s flat from 1 to 32 streams; offline rows now step together (one token upload, LM forward and argmax readback for every row), and a row emits the same alone as beside others (CPU tiny-model test) |
///
/// Qwen3.5 on the CPU gained nothing past two rows in a repeated run, so it
/// stays at two. Metal's LFM2 small-batch mat-vec serves 2..=8 rows, which
/// bounds that family at eight. Drama 3 reads about 6 GB of weights per
/// frame and holds ~37 GiB resident, so it serves one row on CPU and Metal;
/// Metal's second row is an opt-in its load takes from
/// `IZWI_FISH_DRAMA3_NATIVE_BATCHING`, and CUDA keeps the automatic eight.
pub(crate) const fn certified_decode_width(family: ModelFamily, backend: BackendKind) -> usize {
    match (backend, family) {
        (
            BackendKind::Cuda,
            ModelFamily::Qwen35Chat
            | ModelFamily::Qwen35MoeChat
            | ModelFamily::Qwen38Chat
            | ModelFamily::Lfm2Chat
            | ModelFamily::Lfm3Chat
            | ModelFamily::Qwen3Asr
            | ModelFamily::VibeVoiceAsr
            | ModelFamily::ParakeetAsr
            | ModelFamily::NemotronAsr
            | ModelFamily::WhisperAsr
            | ModelFamily::Voxtral
            | ModelFamily::GraniteSpeechAsr,
        ) => CUDA_MAX_WIDTH,
        (BackendKind::Cuda, _) => CUDA_DEFAULT_WIDTH,
        (BackendKind::Cpu | BackendKind::Metal, ModelFamily::Lfm2Chat | ModelFamily::Lfm3Chat)
        | (BackendKind::Metal, ModelFamily::Qwen35Chat) => 8,
        (BackendKind::Cpu | BackendKind::Metal, ModelFamily::FishDrama3Tts) => 1,
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

/// The engine width chat models clamp to their family's certified width:
/// on CUDA with no width fixed (`IZWI_MAX_BATCH_SIZE` or `max_num_seqs`) the
/// CUDA ceiling, otherwise the engine's tensor width.
pub(crate) const fn chat_engine_width(
    backend: BackendKind,
    tensor_width: usize,
    automatic: bool,
) -> usize {
    match backend {
        BackendKind::Cuda if automatic => CUDA_MAX_WIDTH,
        _ => tensor_width,
    }
}

/// The engine width ASR decode clamps to its family's certified width: on
/// CUDA with no width fixed (`IZWI_MAX_BATCH_SIZE` or `max_num_seqs`) the
/// CUDA ceiling, otherwise the non-chat rows.
pub(crate) const fn asr_engine_width(
    backend: BackendKind,
    non_chat_width: usize,
    automatic: bool,
) -> usize {
    match backend {
        BackendKind::Cuda if automatic => CUDA_MAX_WIDTH,
        _ => non_chat_width,
    }
}

/// The continuous decode width an ASR model of `family` loads with before
/// its memory fit: the ASR engine width clamped to the family's certified
/// width, never below the non-chat rows its encoder and prefill run at (an
/// explicit or `max_num_seqs` width therefore applies as it always has).
pub(crate) const fn asr_decode_width(
    family: ModelFamily,
    backend: BackendKind,
    non_chat_width: usize,
    asr_engine_width: usize,
) -> usize {
    let certified = certified_decode_width(family, backend);
    let widened = if asr_engine_width < certified {
        asr_engine_width
    } else {
        certified
    };
    if widened > non_chat_width {
        widened
    } else {
        non_chat_width
    }
}

/// Widths a load fits an ASR decode wider than the engine rows to, widest
/// first.
const ASR_FIT_WIDTHS: [usize; 3] = [CUDA_MAX_WIDTH, 16, CUDA_DEFAULT_WIDTH];

/// The widest of 32, 16 and 8 rows, at most `ceiling` and above `floor`,
/// whose rows of `per_row_bytes` take at most half of `headroom_bytes` (the
/// planning headroom left after the model's resident weights); `floor`, the
/// rows the model ran at before, when none does. The other half stays free
/// for the encoder, prefill and request workspace and for other models.
pub(crate) fn fit_asr_decode_width(
    ceiling: usize,
    floor: usize,
    per_row_bytes: u64,
    headroom_bytes: u64,
) -> usize {
    let budget = headroom_bytes / 2;
    ASR_FIT_WIDTHS
        .into_iter()
        .filter(|width| *width <= ceiling && *width > floor)
        .find(|width| {
            (*width as u64)
                .checked_mul(per_row_bytes)
                .is_some_and(|bytes| bytes <= budget)
        })
        .unwrap_or(floor)
}

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
        // On CUDA the families with batched state kernels take the ceiling;
        // every other family keeps the engine's eight.
        for family in [
            ModelFamily::Qwen35Chat,
            ModelFamily::Qwen35MoeChat,
            ModelFamily::Qwen38Chat,
            ModelFamily::Lfm2Chat,
            ModelFamily::Lfm3Chat,
        ] {
            assert_eq!(
                certified_decode_width(family, BackendKind::Cuda),
                CUDA_MAX_WIDTH,
                "{family:?}"
            );
        }
        assert_eq!(
            certified_decode_width(ModelFamily::Gemma3Chat, BackendKind::Cuda),
            CUDA_DEFAULT_WIDTH
        );
    }

    #[test]
    fn cuda_chat_takes_the_ceiling_only_when_no_width_is_fixed() {
        assert_eq!(
            chat_engine_width(BackendKind::Cuda, 8, true),
            CUDA_MAX_WIDTH
        );
        assert_eq!(chat_engine_width(BackendKind::Cuda, 16, false), 16);
        assert_eq!(chat_engine_width(BackendKind::Metal, 8, true), 8);
        assert_eq!(chat_engine_width(BackendKind::Cpu, 8, true), 8);
    }

    /// The seven ASR families whose decode steps every row per launch take
    /// the CUDA ceiling; every other speech family keeps the engine rows, and
    /// none changes on CPU or Metal.
    #[test]
    fn batched_asr_decode_families_take_the_cuda_ceiling() {
        const WIDE: [ModelFamily; 7] = [
            ModelFamily::Qwen3Asr,
            ModelFamily::VibeVoiceAsr,
            ModelFamily::ParakeetAsr,
            ModelFamily::NemotronAsr,
            ModelFamily::WhisperAsr,
            ModelFamily::Voxtral,
            ModelFamily::GraniteSpeechAsr,
        ];
        for family in WIDE {
            assert_eq!(
                certified_decode_width(family, BackendKind::Cuda),
                CUDA_MAX_WIDTH,
                "{family:?}"
            );
            for backend in [BackendKind::Cpu, BackendKind::Metal] {
                assert_eq!(
                    certified_decode_width(family, backend),
                    PORTABLE_BASELINE_WIDTH,
                    "{family:?} {backend:?}"
                );
            }
        }
        for family in [
            ModelFamily::GraniteSpeech5Asr,
            ModelFamily::NemotronTranscribeAsr,
            ModelFamily::Lfm25Audio,
            ModelFamily::Lfm3Audio,
            ModelFamily::Qwen3Tts,
        ] {
            assert_eq!(
                certified_decode_width(family, BackendKind::Cuda),
                CUDA_DEFAULT_WIDTH,
                "{family:?}"
            );
        }
    }

    #[test]
    fn asr_decode_widens_only_on_automatic_cuda() {
        let decode = |family, backend, non_chat, automatic| {
            asr_decode_width(
                family,
                backend,
                non_chat,
                asr_engine_width(backend, non_chat, automatic),
            )
        };
        // Automatic CUDA: certified families take the ceiling, the rest the
        // engine rows.
        assert_eq!(
            decode(ModelFamily::Qwen3Asr, BackendKind::Cuda, 8, true),
            CUDA_MAX_WIDTH
        );
        assert_eq!(
            decode(ModelFamily::ParakeetAsr, BackendKind::Cuda, 8, true),
            CUDA_MAX_WIDTH
        );
        assert_eq!(
            decode(ModelFamily::WhisperAsr, BackendKind::Cuda, 8, true),
            CUDA_MAX_WIDTH
        );
        assert_eq!(
            decode(ModelFamily::GraniteSpeechAsr, BackendKind::Cuda, 8, true),
            CUDA_MAX_WIDTH
        );
        assert_eq!(
            decode(ModelFamily::GraniteSpeech5Asr, BackendKind::Cuda, 8, true),
            8
        );
        // An explicit or `max_num_seqs` width applies as it always has, above
        // or below a family's certified width.
        for family in [ModelFamily::Qwen3Asr, ModelFamily::WhisperAsr] {
            for width in [4, 16, 32] {
                assert_eq!(decode(family, BackendKind::Cuda, width, false), width);
            }
        }
        // CPU and Metal keep the portable rows (or an explicit width).
        for backend in [BackendKind::Cpu, BackendKind::Metal] {
            for family in [
                ModelFamily::Qwen3Asr,
                ModelFamily::NemotronAsr,
                ModelFamily::WhisperAsr,
            ] {
                assert_eq!(decode(family, backend, 2, true), 2);
                assert_eq!(decode(family, backend, 8, false), 8);
            }
        }
    }

    #[test]
    fn asr_decode_fits_the_widest_width_half_the_headroom_holds() {
        const GIB: u64 = 1024 * 1024 * 1024;
        // Qwen3-ASR 1.7B rows (about 149 MiB each).
        let row = 149 * 1024 * 1024;
        assert_eq!(fit_asr_decode_width(32, 8, row, 70 * GIB), 32);
        assert_eq!(fit_asr_decode_width(32, 8, row, 6 * GIB), 16);
        assert_eq!(fit_asr_decode_width(32, 8, row, 3 * GIB), 8);
        // Below even the floor's rows the model keeps the rows it ran at.
        assert_eq!(fit_asr_decode_width(32, 8, row, GIB), 8);
        assert_eq!(fit_asr_decode_width(32, 8, row, 0), 8);
        // The fit never exceeds the ceiling, and never drops below a floor
        // that is not one of its widths.
        assert_eq!(fit_asr_decode_width(16, 8, row, 70 * GIB), 16);
        assert_eq!(fit_asr_decode_width(32, 12, row, 3 * GIB), 12);
        assert_eq!(fit_asr_decode_width(8, 8, row, 70 * GIB), 8);
        assert_eq!(fit_asr_decode_width(32, 8, u64::MAX, u64::MAX), 8);
    }

    #[test]
    fn fish_drama3_serves_one_row_on_cpu_and_metal() {
        for backend in [BackendKind::Cpu, BackendKind::Metal] {
            assert_eq!(
                certified_decode_width(ModelFamily::FishDrama3Tts, backend),
                1
            );
        }
        assert_eq!(
            certified_decode_width(ModelFamily::FishDrama3Tts, BackendKind::Cuda),
            CUDA_DEFAULT_WIDTH
        );
    }
}
