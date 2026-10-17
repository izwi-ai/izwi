use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use crate::model::ModelVariant;
use crate::models::architectures::fish_s2::FishS2RetainedState;
use crate::models::architectures::lfm25_audio::asr_retained::Lfm25AudioAsrRetainedState;
use crate::models::architectures::lfm25_audio::tts_retained::Lfm25AudioTtsRetainedState;
use crate::models::architectures::nemotron::asr::{
    NemotronRealtimePreparedChunk, NemotronStreamingState,
};
use crate::models::architectures::parakeet::asr::{
    ParakeetPreparedEncoderArtifact, ParakeetRetainedDecodeState,
};
use crate::models::architectures::qwen3::tts::{
    PhysicalTtsDecodeState, PhysicalTtsPrefillState, Qwen3TtsModel,
};
use crate::models::architectures::vibevoice::tts::VibeVoiceTtsRetainedState;
use crate::models::architectures::voxtral::realtime::{
    VoxtralRealtimeCheckpoint, VoxtralRealtimeState,
};
use crate::models::architectures::voxtral::tts::retained::VoxtralTtsRetainedState;
use crate::models::registry::{
    AsrModelLease, FishS2TtsModelLease, Lfm25AudioModelLease, NativeAsrDecodeState, NativeAsrModel,
    NativeChatDecodeState, NativeChatReplayCheckpoint, QwenTtsModelLease, VibeVoiceTtsModelLease,
    VoxtralModelLease, VoxtralTtsModelLease,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

pub(super) struct SuspendedChatDecode {
    pub(super) variant: ModelVariant,
    pub(super) checkpoint: NativeChatReplayCheckpoint,
    pub(super) last_tokens_generated: usize,
    pub(super) stream_sequence: usize,
    pub(super) streamed_text: String,
    /// DS9.3: logprob entries accumulated across committed quanta.
    pub(super) logprobs: Vec<crate::engine::TokenLogprob>,
    pub(super) stop: Option<super::stop_sequences::StopSequenceFilter>,
}

pub(super) struct ActiveChatDecode {
    pub(super) variant: ModelVariant,
    pub(super) state: NativeChatDecodeState,
    pub(super) last_tokens_generated: usize,
    pub(super) stream_sequence: usize,
    /// Text already made authoritative through append-only public deltas.
    /// Tokenizer decoders may normalize or rewrite their cumulative terminal
    /// string, but an SSE delta cannot be retracted after publication.
    pub(super) streamed_text: String,
    /// DS9.3: logprob entries accumulated across committed quanta, published
    /// with the terminal output.
    pub(super) logprobs: Vec<crate::engine::TokenLogprob>,
    /// The request's stop strings, with any text held back because it may
    /// begin one.
    pub(super) stop: Option<super::stop_sequences::StopSequenceFilter>,
}

impl ActiveChatDecode {
    /// Apply the request's stop strings to one committed step (see
    /// [`StopSequenceFilter::apply`]) and return the logprob entries to
    /// publish with the step's delta.
    ///
    /// [`StopSequenceFilter::apply`]: super::stop_sequences::StopSequenceFilter::apply
    pub(super) fn apply_stop_sequences(
        &mut self,
        step: &mut crate::models::registry::NativeChatDecodeStep,
    ) -> Vec<crate::engine::TokenLogprob> {
        match self.stop.as_mut() {
            Some(stop) => stop.apply(step),
            None => step.logprobs.clone(),
        }
    }

    /// DS9.3: record one committed step's logprob entries and, on the
    /// terminal step, hand back the request's whole list. Streaming does not
    /// change this: SSE deltas carry their own entries, and the terminal
    /// output always carries the complete list.
    pub(super) fn record_step_logprobs(
        &mut self,
        step: &[crate::engine::TokenLogprob],
        finished: bool,
    ) -> Vec<crate::engine::TokenLogprob> {
        accumulate_step_logprobs(&mut self.logprobs, step, finished)
    }
}

pub(super) fn accumulate_step_logprobs(
    accumulated: &mut Vec<crate::engine::TokenLogprob>,
    step: &[crate::engine::TokenLogprob],
    finished: bool,
) -> Vec<crate::engine::TokenLogprob> {
    accumulated.extend_from_slice(step);
    if finished {
        std::mem::take(accumulated)
    } else {
        Vec::new()
    }
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

pub(super) struct ActiveParakeetAsrDecode {
    pub(super) variant: ModelVariant,
    pub(super) model: AsrModelLease,
    pub(super) artifact: Arc<ParakeetPreparedEncoderArtifact>,
    pub(super) state: ParakeetRetainedDecodeState,
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
    /// Samples a streamed request has published so far.
    pub(super) streamed_samples: usize,
}

pub(super) struct ActiveVibeVoiceTtsDecode {
    pub(super) variant: ModelVariant,
    pub(super) model: VibeVoiceTtsModelLease,
    pub(super) state: VibeVoiceTtsRetainedState,
    pub(super) last_frames_generated: usize,
    pub(super) stream_sequence: usize,
    /// Samples a streamed request has published so far.
    pub(super) streamed_samples: usize,
}

pub(super) struct ActiveFishS2TtsDecode {
    pub(super) variant: ModelVariant,
    pub(super) model: FishS2TtsModelLease,
    pub(super) state: FishS2RetainedState,
    pub(super) last_frames_generated: usize,
    pub(super) stream_sequence: usize,
    pub(super) codec: crate::models::architectures::fish_s2::dac::FishS2DacStreamState,
    pub(super) audio_samples: Vec<f32>,
    pub(super) total_audio_samples: usize,
    pub(super) collect_audio_samples: bool,
    pub(super) codec_ms: f64,
    pub(super) execution_started: std::time::Instant,
    pub(super) first_audio_ms: Option<f64>,
}

pub(super) struct ActiveVoxtralTtsDecode {
    pub(super) variant: ModelVariant,
    pub(super) model: VoxtralTtsModelLease,
    pub(super) state: VoxtralTtsRetainedState,
    pub(super) last_frames_generated: usize,
    pub(super) stream_sequence: usize,
    /// Samples a streamed request has published so far.
    pub(super) streamed_samples: usize,
}

pub(super) struct ActiveVoxtralRealtime {
    pub(super) variant: ModelVariant,
    pub(super) model: VoxtralModelLease,
    pub(super) state: VoxtralRealtimeState,
    pub(super) last_tokens_generated: usize,
    pub(super) stream_sequence: usize,
    pub(super) input_sample_rate: u32,
}

#[derive(Clone)]
pub(super) struct ActiveNemotronRealtime {
    pub(super) variant: ModelVariant,
    pub(super) model: AsrModelLease,
    pub(super) state: NemotronStreamingState,
    pub(super) prepared: VecDeque<NemotronRealtimePreparedChunk>,
    pub(super) stream_sequence: usize,
    pub(super) input_sample_rate: u32,
}

pub(super) struct PendingNemotronRealtimeQuantum {
    pub(super) session: crate::engine::SessionKey,
    pub(super) active: ActiveNemotronRealtime,
    pub(super) checkpoint: ActiveNemotronRealtime,
    pub(super) finished: bool,
}

pub(super) struct PreparedNemotronRealtimeQuantum {
    pub(super) session: crate::engine::SessionKey,
    pub(super) replacement: Option<ActiveNemotronRealtime>,
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

#[cfg(test)]
mod tests {
    use super::accumulate_step_logprobs;
    use crate::engine::TokenLogprob;

    fn entry(token: &str) -> TokenLogprob {
        TokenLogprob {
            token: token.to_string(),
            logprob: -0.5,
            bytes: token.as_bytes().to_vec(),
            top_logprobs: Vec::new(),
        }
    }

    #[test]
    fn terminal_step_publishes_every_committed_logprob_entry() {
        let mut accumulated = Vec::new();
        assert!(accumulate_step_logprobs(&mut accumulated, &[entry("a")], false).is_empty());
        assert!(accumulate_step_logprobs(&mut accumulated, &[entry("b")], false).is_empty());
        let terminal = accumulate_step_logprobs(&mut accumulated, &[entry("c")], true);
        let tokens: Vec<&str> = terminal.iter().map(|entry| entry.token.as_str()).collect();
        assert_eq!(tokens, ["a", "b", "c"]);
        assert!(accumulated.is_empty());
    }
}
