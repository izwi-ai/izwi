//! LFM3-Audio-2B constants: special tokens, the speech-frame geometry and
//! the published sampling settings (`config.json` and
//! `generation_config.json` of LiquidAI/LFM3-Audio-2B @ 9d49701c).

/// `<|startoftext|>`.
pub(crate) const BOS: u32 = 124_894;
/// `<|endoftext|>`.
pub(crate) const END_OF_TEXT: u32 = 124_895;
/// `<|im_start|>`.
pub(crate) const IM_START: u32 = 124_899;
/// `<|im_end|>`: ends the turn from any text slot.
pub(crate) const IM_END: u32 = 124_900;
/// `<|audio_start|>`: switches generation to speech until an end-of-audio
/// frame.
pub(crate) const AUDIO_START: u32 = 125_012;
/// `<|text_end|>`: closes the text of an interleaved block; outside one it
/// ends the turn.
pub(crate) const TEXT_END: u32 = 125_014;
/// `<|mixed_start|>`: blocks of text and speech.
pub(crate) const MIXED_START: u32 = 125_015;
/// `<|mixed_end|>`: closes LiveASR's final chunk.
pub(crate) const MIXED_END: u32 = 125_016;
/// One encoded input-speech frame.
pub(crate) const AUDIO_PLACEHOLDER: u32 = 125_031;
/// One generated speech frame in a prompt.
pub(crate) const AUDIO_OUT_PLACEHOLDER: u32 = 125_032;

/// Codebooks per speech frame.
pub(crate) const CODEBOOKS: usize = 12;
/// Codes per codebook: the codec's 2,016 FSQ codes plus end-of-audio.
pub(crate) const CODEBOOK_SIZE: usize = 2_017;
/// The end-of-audio code. A frame sampling it in any codebook ends speech.
pub(crate) const AUDIO_EOS_CODE: u32 = 2_016;
/// Codebooks revealed by each MaskGIT pass of the depth decoder.
pub(crate) const MASKGIT_SCHEDULE: [usize; 3] = [4, 4, 4];
/// Text tokens per interleaved block.
pub(crate) const INTERLEAVED_TEXT: usize = 6;
/// Speech frames per interleaved block.
pub(crate) const INTERLEAVED_AUDIO: usize = 12;

/// Input sample rate of the audio encoder.
pub(crate) const INPUT_SAMPLE_RATE: u32 = 16_000;
/// Output sample rate of the LiquidCodec decoder.
pub(crate) const OUTPUT_SAMPLE_RATE: u32 = 24_000;
/// Output samples per speech frame (12.5 frames per second).
pub(crate) const SAMPLES_PER_FRAME: usize = 1_920;
/// Frames per second of speech.
pub(crate) const FRAMES_PER_SECOND: f32 = OUTPUT_SAMPLE_RATE as f32 / SAMPLES_PER_FRAME as f32;
/// Mel frames per encoder token (8 x 10 ms).
pub(crate) const ENCODER_STRIDE: usize = 8;

/// The model card's speech sampling: temperature 1, top-k 128, a
/// repetition penalty of 1.1 over each codebook's last 4,096 frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lfm3AudioSampling {
    pub temperature: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub repetition_window: usize,
}

impl Default for Lfm3AudioSampling {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 128,
            repetition_penalty: 1.1,
            repetition_window: 4_096,
        }
    }
}

/// Text sampling. `generation_config.json` decodes text greedily.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lfm3TextSampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    /// Generated tokens the repetition penalty covers (HF's default 32).
    pub repetition_window: usize,
}

impl Default for Lfm3TextSampling {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            repetition_window: 32,
        }
    }
}
