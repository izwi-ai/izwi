use std::sync::Arc;
use std::time::Instant;

use crate::model::ModelVariant;
use crate::models::architectures::chatterbox::retained::ChatterboxRetainedState;
use crate::models::architectures::fish_drama3::{FishDrama3AudioStream, FishDrama3RequestState};
use crate::models::architectures::fish_s2::FishS2RetainedState;
use crate::models::architectures::lfm25_audio::asr_retained::Lfm25AudioAsrRetainedState;
use crate::models::architectures::lfm25_audio::tts_retained::Lfm25AudioTtsRetainedState;
use crate::models::architectures::maya1::retained::Maya1RetainedState;
use crate::models::architectures::maya2::retained::Maya2RetainedState;
use crate::models::architectures::nemotron::asr::{
    NemotronPreparedEncoderArtifact, NemotronRetainedDecodeState,
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
    AsrModelLease, ChatterboxTtsModelLease, FishDrama3TtsModelLease, FishS2TtsModelLease,
    Lfm25AudioModelLease, Maya1TtsModelLease, Maya2TtsModelLease, NativeAsrDecodeState,
    NativeAsrModel, NativeChatDecodeState, NativeChatReplayCheckpoint, QwenTtsModelLease,
    VibeVoiceTtsModelLease, VoxtralModelLease, VoxtralTtsModelLease,
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
    pub(super) clock: RetainedAsrClock,
}

/// Wall time a retained ASR row spent in the prefill and decode dispatches
/// that carried it, for its final `asr_diagnostics`. A batched step counts
/// in full for every row it carried.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct RetainedAsrClock {
    prefill_ms: f64,
    prefill_dispatches: usize,
    decode_wall_ms: f64,
    decode_steps: usize,
}

impl RetainedAsrClock {
    pub(super) fn record_prefill(&mut self, elapsed: std::time::Duration) {
        self.prefill_ms += elapsed.as_secs_f64() * 1000.0;
        self.prefill_dispatches += 1;
    }

    pub(super) fn record_decode(&mut self, elapsed: std::time::Duration) {
        self.decode_wall_ms += elapsed.as_secs_f64() * 1000.0;
        self.decode_steps += 1;
    }

    /// The row's final diagnostics: the family's own (if it publishes any)
    /// with the retained route and its dispatch times.
    pub(super) fn diagnostics(&self, family: Option<serde_json::Value>) -> serde_json::Value {
        let mut diagnostics = match family {
            Some(serde_json::Value::Object(map)) => map,
            _ => serde_json::Map::new(),
        };
        diagnostics.insert("route".into(), "retained_sequence".into());
        diagnostics.insert(
            "retained_timings_ms".into(),
            serde_json::json!({
                "prefill": self.prefill_ms,
                "prefill_dispatches": self.prefill_dispatches,
                "decode_wall": self.decode_wall_ms,
                "decode_steps": self.decode_steps,
            }),
        );
        serde_json::Value::Object(diagnostics)
    }
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

pub(super) struct ActiveNemotronAsrDecode {
    pub(super) variant: ModelVariant,
    pub(super) model: AsrModelLease,
    pub(super) artifact: Arc<NemotronPreparedEncoderArtifact>,
    pub(super) state: NemotronRetainedDecodeState,
    pub(super) last_tokens_generated: usize,
    pub(super) stream_sequence: usize,
    pub(super) input_sample_rate: u32,
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

/// One Fish Audio Drama 3 request between quanta: its retained state (the
/// trunk's DeltaNet tensors live in the engine's tensor arena between
/// quanta), its committed codec stream, and what it has emitted.
pub(super) struct ActiveFishDrama3TtsDecode {
    pub(super) variant: ModelVariant,
    pub(super) model: FishDrama3TtsModelLease,
    pub(super) state: FishDrama3RequestState,
    pub(super) stream: FishDrama3AudioStream,
    /// Frames committed by the last decode quantum.
    pub(super) last_frames: usize,
    pub(super) stream_sequence: usize,
    /// A final-only request's PCM, handed over at finalization.
    pub(super) audio_samples: Vec<f32>,
    pub(super) total_audio_samples: usize,
    pub(super) collect_audio_samples: bool,
    pub(super) prefill_ms: f64,
    pub(super) decode_ms: f64,
    pub(super) prefill_steps: usize,
    pub(super) decode_steps: usize,
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

pub(super) struct ActiveChatterboxTtsDecode {
    pub(super) variant: ModelVariant,
    pub(super) model: ChatterboxTtsModelLease,
    pub(super) state: ChatterboxRetainedState,
    /// A streamed request's next chunk sequence and samples sent so far.
    pub(super) stream_sequence: usize,
    pub(super) streamed_samples: usize,
}

pub(super) struct ActiveMaya1TtsDecode {
    pub(super) variant: ModelVariant,
    pub(super) model: Maya1TtsModelLease,
    pub(super) state: Maya1RetainedState,
    /// A streamed request's next chunk sequence.
    pub(super) stream_sequence: usize,
    /// A final-only request's settled samples, handed over at finalize.
    pub(super) rendered: Vec<f32>,
}

pub(super) struct ActiveMaya2TtsDecode {
    pub(super) variant: ModelVariant,
    pub(super) model: Maya2TtsModelLease,
    pub(super) state: Maya2RetainedState,
    /// A streamed request's next chunk sequence.
    pub(super) stream_sequence: usize,
    /// A final-only request's settled samples, handed over at finalize.
    pub(super) rendered: Vec<f32>,
}

pub(super) struct ActiveLfm3AudioDecode {
    pub(super) variant: ModelVariant,
    pub(super) model: Lfm25AudioModelLease,
    pub(super) state: crate::models::architectures::lfm3_audio::retained::Lfm3AudioRetainedState,
    /// A streamed request's next chunk sequence.
    pub(super) stream_sequence: usize,
    /// Samples a streamed request has published so far.
    pub(super) streamed_samples: usize,
    /// A final-only request's samples, handed over when it finishes.
    pub(super) rendered: Vec<f32>,
    /// A transcription's input audio: its sample rate and length.
    pub(super) input_audio: Option<(u32, usize)>,
}

/// One Nemotron 3.5 Transcribe row between quanta: its retained state
/// (Mamba state, perception stream, decode loop) and stream position.
pub(super) struct ActiveNemotronTranscribeDecode {
    pub(super) variant: ModelVariant,
    pub(super) model: AsrModelLease,
    pub(super) state: crate::models::architectures::nemotron_transcribe::NemotronRetainedState,
    /// A streamed request's next chunk sequence.
    pub(super) stream_sequence: usize,
}

/// One LFM3-Audio LiveASR session between quanta: its segment's host
/// state, the language model's ShortConv ring (kept here, never in the
/// arena) and the paged-KV authority, and what the next quantum feeds.
pub(super) struct ActiveLfm3AudioRealtime {
    pub(super) variant: ModelVariant,
    pub(super) model: Lfm25AudioModelLease,
    pub(super) live: crate::models::architectures::lfm3_audio::model::Lfm3LiveAsr,
    pub(super) retained: crate::models::architectures::lfm25_audio::state::Lfm25AudioRetainedState,
    /// The embeddings the next prefill or decode quantum feeds, and where.
    pub(super) feed: Option<(candle_core::Tensor, usize)>,
    /// The segment's text so far.
    pub(super) text: String,
    pub(super) stream_sequence: usize,
    /// Samples heard so far.
    pub(super) heard_samples: usize,
}

/// An LFM3-Audio LiveASR quantum that has run but is not authoritative
/// until EngineCore accepts the matching physical result.
pub(super) struct PendingLfm3AudioRealtimeQuantum {
    pub(super) session: crate::engine::SessionKey,
    pub(super) active: ActiveLfm3AudioRealtime,
    pub(super) cache: PhysicalPagedKvCache,
    pub(super) checkpoint:
        crate::models::architectures::lfm25_audio::state::Lfm25AudioRetainedCheckpoint,
    pub(super) prior_live: crate::models::architectures::lfm3_audio::model::Lfm3LiveAsr,
    pub(super) prior_feed: Option<(candle_core::Tensor, usize)>,
    pub(super) prior_text_len: usize,
    pub(super) prior_stream_sequence: usize,
    pub(super) prior_heard_samples: usize,
    pub(super) finished: bool,
}

pub(super) struct ActiveVoxtralRealtime {
    pub(super) variant: ModelVariant,
    pub(super) model: VoxtralModelLease,
    pub(super) state: VoxtralRealtimeState,
    pub(super) last_tokens_generated: usize,
    pub(super) stream_sequence: usize,
    pub(super) input_sample_rate: u32,
    /// Prefill and decode time of an offline row, for its diagnostics.
    pub(super) clock: RetainedAsrClock,
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

    /// A retained row's diagnostics keep the family's own fields and add the
    /// route and the dispatch times it was carried through.
    #[test]
    fn retained_clock_extends_the_family_diagnostics() {
        let mut clock = super::RetainedAsrClock::default();
        clock.record_prefill(std::time::Duration::from_millis(4));
        clock.record_decode(std::time::Duration::from_millis(2));
        clock.record_decode(std::time::Duration::from_millis(3));
        let diagnostics =
            clock.diagnostics(Some(serde_json::json!({ "model_family": "qwen3_asr" })));
        assert_eq!(diagnostics["model_family"], "qwen3_asr");
        assert_eq!(diagnostics["route"], "retained_sequence");
        let timings = &diagnostics["retained_timings_ms"];
        assert_eq!(timings["prefill_dispatches"], 1);
        assert_eq!(timings["decode_steps"], 2);
        assert!((timings["decode_wall"].as_f64().unwrap() - 5.0).abs() < 1e-9);
        assert!((timings["prefill"].as_f64().unwrap() - 4.0).abs() < 1e-9);
        let bare = super::RetainedAsrClock::default().diagnostics(None);
        assert_eq!(bare["route"], "retained_sequence");
        assert_eq!(bare["retained_timings_ms"]["decode_steps"], 0);
    }
}
