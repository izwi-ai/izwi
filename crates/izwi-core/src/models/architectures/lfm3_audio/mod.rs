//! LiquidAI LFM3-Audio-2B: speech recognition, speech synthesis and voice
//! chat with one model.
//!
//! The language model is the LFM3 trunk (`lfm3`) on the LFM2 backbone. The
//! causal speech encoder runs on the LFM2 backbone too, under its own tensor
//! names in the mmproj GGUF, and so does LiquidCodec's decoder trunk; the
//! MaskGIT depth decoder and the FSQ/ISTFT stages are this module's own.

pub(crate) mod codec;
pub(crate) mod config;
pub(crate) mod depth;
pub(crate) mod encoder;
pub(crate) mod model;
pub(crate) mod physical;
pub(crate) mod preprocessor;
pub mod prompts;
pub(crate) mod retained;
pub(crate) mod sampling;

pub use model::{lfm3_audio_tts_auto_max_frames, Lfm3AudioModel, Lfm3AudioOutput};
pub use prompts::{
    Lfm3AudioVoice, LFM3_AUDIO_ASR_LANGUAGES, LFM3_AUDIO_TTS_LOCALES, LFM3_AUDIO_VOICES,
};

use crate::engine::{NativeBatchMode, OutputVisibility, StageShapePolicy, StageWorkSelector};
use crate::error::{Error, Result};

/// LiveASR's engine realtime graph: static-batched preparation (mel
/// features and the streaming encoder, ragged since each session hears its
/// own push), one session's chunk prefill at a time, continuous decode
/// batched across sessions, and a scalar completion.
pub(crate) const LFM3_AUDIO_REALTIME_EXECUTION_ABI: u32 = 1;
pub(crate) const LFM3_AUDIO_REALTIME_PREPARATION_STAGE: &str =
    "asr.realtime.lfm3_audio.preparation";
pub(crate) const LFM3_AUDIO_REALTIME_PROMPT_PREFILL_STAGE: &str =
    "asr.realtime.lfm3_audio.chunk_prefill";
pub(crate) const LFM3_AUDIO_REALTIME_DECODE_STAGE: &str = "asr.realtime.lfm3_audio.decode";
pub(crate) const LFM3_AUDIO_REALTIME_COMPLETION_STAGE: &str = "asr.realtime.lfm3_audio.completion";

/// Mel frames one push's preparation may hear: 40.96 s of audio, above a
/// whole segment pushed at once.
pub(crate) const LFM3_AUDIO_REALTIME_MAX_PREPARATION_FRAMES: u64 = 4_096;
/// One push's preparation workspace: its spectrogram and mel features (about
/// 7 MiB for the largest push) and the streaming encoder's activations for
/// one chunk at a time (a few MiB), with ample room.
pub(crate) const LFM3_AUDIO_REALTIME_PREPARATION_WORKSPACE_BYTES: u64 = 64 * 1024 * 1024;
/// A LiveASR session's retained state outside the arena's pages, on the host
/// and in tensors, with a quantum's checkpoint copy: the mel and encoder
/// streams' bounded tails (the encoder's 31-row attention and 2-row
/// ShortConv inputs, under a mebibyte), the chunk embeddings one push
/// queues (about 3 MiB for a whole segment pushed at once), the language
/// model's ShortConv ring and the segment's transcript.
pub(crate) const LFM3_AUDIO_REALTIME_STATE_HOST_BYTES: u64 = 16 * 1024 * 1024;
pub(crate) const LFM3_AUDIO_REALTIME_STATE_TENSOR_BYTES: u64 = 16 * 1024 * 1024;

/// Admit a realtime session only on LFM3-Audio's exact LiveASR graph.
pub(crate) fn authenticate_lfm3_audio_realtime_execution_binding(
    binding: &crate::engine::ExecutionAdapterBinding,
) -> Result<()> {
    if binding.model_variant.family() != crate::catalog::ModelFamily::Lfm3Audio
        || binding.capability_id != "realtime_asr"
        || binding.adapter_abi_revision
            != crate::engine::AdapterAbiRevision::new(LFM3_AUDIO_REALTIME_EXECUTION_ABI)
        || binding.stages.len() != 4
    {
        return Err(Error::InvalidInput(
            "realtime ASR request is not bound to the exact LFM3-Audio execution ABI".into(),
        ));
    }
    let expected = [
        (
            LFM3_AUDIO_REALTIME_PREPARATION_STAGE,
            StageWorkSelector::RealtimePreparation,
            NativeBatchMode::Static,
            StageShapePolicy::Ragged,
        ),
        (
            LFM3_AUDIO_REALTIME_PROMPT_PREFILL_STAGE,
            StageWorkSelector::RealtimePromptPrefill,
            NativeBatchMode::None,
            StageShapePolicy::Exact,
        ),
        (
            LFM3_AUDIO_REALTIME_DECODE_STAGE,
            StageWorkSelector::RealtimeDecodeContinuation,
            NativeBatchMode::Continuous,
            StageShapePolicy::Ragged,
        ),
        (
            LFM3_AUDIO_REALTIME_COMPLETION_STAGE,
            StageWorkSelector::RealtimeCompletion,
            NativeBatchMode::None,
            StageShapePolicy::Exact,
        ),
    ];
    let sealed = binding.stages.iter().zip(expected).enumerate().all(
        |(id, (stage, (name, selector, batch_mode, shape_policy)))| {
            stage.id == crate::engine::StageId::new(id as u32)
                && stage.name == name
                && stage.selector == selector
                && stage.batch_mode == batch_mode
                && stage.shape_policy == shape_policy
                && stage.output_visibility == OutputVisibility::AfterQuantumCommit
        },
    );
    if !sealed {
        return Err(Error::InvalidInput(
            "realtime ASR request crossed the sealed LFM3-Audio stage graph".into(),
        ));
    }
    Ok(())
}
