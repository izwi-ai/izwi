//! Maya Research's Maya 2 Native (multilingual streaming TTS), ported
//! natively.
//!
//! Upstream: weights <https://huggingface.co/maya-research/maya2-native> at
//! `1403b841` (Apache-2.0); inference code
//! <https://github.com/MayaResearch/maya2-native> at `bbfdb152` (Apache-2.0);
//! codec <https://huggingface.co/kyutai/mimi> (CC-BY-4.0); speaker encoder
//! <https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb> (Apache-2.0).
//!
//! Text (in one of 12 languages) and a voice → a Qwen3-1.7B backbone that
//! predicts each 80 ms frame's first Mimi codebook (or the end of audio) →
//! a depth decoder that fills in the other 31 → Mimi's decoder, frame by
//! frame, → 24 kHz mono. A voice is one of 20 built-in personas, selected by
//! its name in the text, or a clone of a reference clip: its transcript and
//! Mimi codes ahead of the target text, plus an ECAPA speaker embedding that
//! enters the backbone (an anchor at each `<|audio|>` sentinel) and the depth
//! decoder (its cross-attention memory).

pub mod backbone;
pub mod config;
pub mod depth;
pub mod ecapa;
pub mod guards;
pub mod inputs;
pub mod prompt;
pub mod runtime;
pub mod sampling;
pub mod text;
mod weights;

pub use config::{MAYA2_DEFAULTS, SAMPLE_RATE};
pub use guards::Maya2StopReason;
pub use prompt::{detect_language, resolve_language, Maya2Prompt, Maya2Reference};
pub use runtime::{
    Maya2ClonedVoice, Maya2GenerationOutput, Maya2GenerationParams, Maya2Runtime, Maya2Voice,
};
pub use sampling::Maya2SamplingParams;
pub use weights::published_tensors;
