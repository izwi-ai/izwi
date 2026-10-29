//! CTC word timestamps on the host: the CTC tokenizer and NeMo's
//! multi-speaker forced aligner (`MultiSpeakerSOTWordTimestampAligner` in
//! `speechlm2/parts/ctc_timestamp_utils.py` @ 743beaf4, with vLLM's public
//! `ctc_timestamps` object from `vllm/salm/ctc_timestamps.py`).
//!
//! The neural CTC head (`ctc_head`) produces 40 ms log-probabilities over
//! 23,863 classes (blank = 23,862, the last). This module turns a finished
//! t-SOT transcript plus those log-probabilities and the diarizer's speaker
//! probabilities into word timestamps:
//!
//! 1. [`prepare`] parses the transcript (cleaned of the decoder's other
//!    special tokens by [`clean_sot_transcript`]) into speaker streams,
//!    re-tokenises each with the CTC SentencePiece model ([`CtcTokenizer`])
//!    and lists the columns to gather, so the device never ships the full
//!    vocabulary to the host;
//! 2. [`align`] maps speaker tags to diarizer columns, runs the banded
//!    forced alignment per stream, refines word boundaries against the
//!    diarization segments and returns [`CtcTimestamps`], serialising as
//!    vLLM's `ctc_timestamps` (`words`, `diarization`,
//!    `speaker_tag_to_diarization_speaker`, `error`).
//!
//! Upstream's behaviour is kept to the bit where the platform allows it;
//! the module docs of `align`, `search` and `tokenizer` list the
//! deviations (Thai segmentation, checkpointed backtrace) and the two
//! roundings that follow upstream's arm64 builds.

pub mod align;
mod assignment;
mod search;
mod text;
pub mod tokenizer;

#[cfg(test)]
mod golden_tests;

pub use align::{
    align, diarization_segments, prepare, CtcAlignmentConfig, CtcAlignmentInputs, CtcPrepared,
    CtcSegment, CtcTimestampError, CtcTimestamps, CtcWord, SpeakerActivity,
};
pub use text::clean_sot_transcript;
pub use tokenizer::CtcTokenizer;
