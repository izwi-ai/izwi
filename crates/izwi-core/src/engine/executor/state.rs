use crate::model::ModelVariant;
use crate::models::architectures::qwen3::tts::TtsDecodeState as QwenTtsDecodeState;
use crate::models::registry::{NativeAsrDecodeState, NativeChatDecodeState};

pub(super) struct ActiveChatDecode {
    pub(super) variant: ModelVariant,
    pub(super) state: NativeChatDecodeState,
    pub(super) prompt_accounted: bool,
    pub(super) last_tokens_generated: usize,
    pub(super) stream_sequence: usize,
}

pub(super) struct ActiveAsrDecode {
    pub(super) variant: ModelVariant,
    pub(super) state: NativeAsrDecodeState,
    pub(super) prompt_accounted: bool,
    pub(super) last_tokens_generated: usize,
    pub(super) stream_sequence: usize,
    pub(super) input_sample_rate: u32,
    pub(super) input_sample_count: usize,
}

pub(super) struct ActiveQwenTtsDecode {
    pub(super) variant: Option<ModelVariant>,
    pub(super) state: QwenTtsDecodeState,
    pub(super) prompt_accounted: bool,
    pub(super) last_frames_generated: usize,
    pub(super) stream_sequence: usize,
    pub(super) audio_samples_accum: Vec<f32>,
}
