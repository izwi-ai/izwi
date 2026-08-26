use std::sync::Arc;
use std::time::Instant;

use crate::model::ModelVariant;
use crate::models::architectures::lfm25_audio::asr_retained::Lfm25AudioAsrRetainedState;
use crate::models::architectures::lfm25_audio::tts_retained::Lfm25AudioTtsRetainedState;
use crate::models::architectures::qwen3::tts::{
    PhysicalTtsDecodeState, PhysicalTtsPrefillState, Qwen3TtsModel,
};
use crate::models::architectures::vibevoice::tts::VibeVoiceTtsRetainedState;
use crate::models::architectures::voxtral::realtime::{
    VoxtralRealtimeCheckpoint, VoxtralRealtimeState,
};
use crate::models::registry::{
    AsrModelLease, Lfm25AudioModelLease, NativeAsrDecodeState, NativeAsrModel,
    NativeChatDecodeState, QwenTtsModelLease, VibeVoiceTtsModelLease, VoxtralModelLease,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

pub(super) struct ActiveChatDecode {
    pub(super) variant: ModelVariant,
    pub(super) state: NativeChatDecodeState,
    pub(super) last_tokens_generated: usize,
    pub(super) stream_sequence: usize,
    /// Text already made authoritative through append-only public deltas.
    /// Tokenizer decoders may normalize or rewrite their cumulative terminal
    /// string, but an SSE delta cannot be retracted after publication.
    pub(super) streamed_text: String,
}

pub(super) struct ActiveAsrDecode {
    pub(super) variant: ModelVariant,
    pub(super) model: Arc<NativeAsrModel>,
    pub(super) _model_lease: AsrModelLease,
    pub(super) state: NativeAsrDecodeState,
    pub(super) last_tokens_generated: usize,
    pub(super) stream_sequence: usize,
    pub(super) input_sample_rate: u32,
    pub(super) input_sample_count: usize,
}

pub(super) struct ActiveLfm25AsrDecode {
    pub(super) variant: ModelVariant,
    pub(super) model: Lfm25AudioModelLease,
    pub(super) state: Lfm25AudioAsrRetainedState,
    pub(super) last_tokens_generated: usize,
    pub(super) stream_sequence: usize,
    pub(super) input_sample_rate: u32,
    pub(super) input_sample_count: usize,
}

pub(super) struct ActiveLfm25TtsDecode {
    pub(super) variant: ModelVariant,
    pub(super) model: Lfm25AudioModelLease,
    pub(super) state: Lfm25AudioTtsRetainedState,
    pub(super) last_tokens_generated: usize,
    pub(super) stream_sequence: usize,
}

pub(super) struct ActiveVibeVoiceTtsDecode {
    pub(super) variant: ModelVariant,
    pub(super) model: VibeVoiceTtsModelLease,
    pub(super) state: VibeVoiceTtsRetainedState,
    pub(super) last_frames_generated: usize,
    pub(super) stream_sequence: usize,
}

pub(super) struct ActiveVoxtralRealtime {
    pub(super) variant: ModelVariant,
    pub(super) model: VoxtralModelLease,
    pub(super) state: VoxtralRealtimeState,
    pub(super) last_tokens_generated: usize,
    pub(super) stream_sequence: usize,
    pub(super) input_sample_rate: u32,
}

/// A Voxtral host/cache transaction that has completed model execution but is
/// not authoritative until EngineCore accepts the matching physical result.
pub(super) struct PendingVoxtralRealtimeQuantum {
    pub(super) session: crate::engine::SessionKey,
    pub(super) active: ActiveVoxtralRealtime,
    pub(super) cache: PhysicalPagedKvCache,
    pub(super) checkpoint: VoxtralRealtimeCheckpoint,
    pub(super) prior_last_tokens_generated: usize,
    pub(super) prior_stream_sequence: usize,
    pub(super) prior_input_sample_rate: u32,
    pub(super) finished: bool,
}

/// A pending quantum whose model checkpoint has been resolved, but whose
/// retained state cannot be published until Core resolves the matching managed
/// KV transaction.
pub(super) struct PreparedVoxtralRealtimeQuantum {
    pub(super) session: crate::engine::SessionKey,
    pub(super) replacement: Option<ActiveVoxtralRealtime>,
}

pub(super) enum QwenTtsPhysicalState {
    Prefill(PhysicalTtsPrefillState),
    Decode(PhysicalTtsDecodeState),
    Transitioning,
}

pub(super) struct ActiveQwenTtsDecode {
    pub(super) variant: Option<ModelVariant>,
    pub(super) model: Arc<Qwen3TtsModel>,
    pub(super) _model_lease: Option<QwenTtsModelLease>,
    pub(super) state: QwenTtsPhysicalState,
    pub(super) last_frames_generated: usize,
    pub(super) stream_sequence: usize,
    pub(super) audio_samples_accum: Vec<f32>,
    pub(super) execution_started: Instant,
    pub(super) normalization_ms: f64,
    pub(super) prefill_ms: f64,
    pub(super) sampling_ms: f64,
    pub(super) decode_ms: f64,
    pub(super) codec_ms: f64,
    pub(super) postprocess_ms: f64,
    pub(super) first_output_ms_since_start: Option<f64>,
    pub(super) prefill_steps: u32,
    pub(super) decode_steps: u32,
}
