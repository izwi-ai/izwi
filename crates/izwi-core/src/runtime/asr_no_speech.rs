//! Whole-input no-speech gate for ASR decoders that hallucinate on non-speech
//! (PIPE-18).
//!
//! Generative ASR decoders continue a language-model prompt, so silence or
//! noise comes back as a confident short phrase ("you", "Okay.") that their
//! own no-speech heuristics keep. The gate runs once on the canonical decoded
//! audio, before route preparation, for every model family whose capability
//! says it hallucinates on non-speech. It returns an empty transcript only
//! when no VAD frame of the whole input reaches the speech planner's start
//! threshold. It never trims audio: any input with a speech-like frame reaches
//! the model unchanged.

use std::time::Duration;

use izwi_asr_toolkit::{scan_speech_presence, AsrSpeechChunkConfig, SpeechPresenceScan};
use serde_json::json;

use crate::engine::{
    AudioOutput, EngineCoreRequest, EngineOutput, ExecutorOutput, OutputProcessor,
    StreamingOutput, TaskType,
};

/// Whether the gate decides this request: offline ASR on a model family that
/// hallucinates on non-speech, before its route is prepared. Realtime sessions
/// receive audio incrementally and are not gated here.
pub(crate) fn asr_no_speech_gate_applies(request: &EngineCoreRequest) -> bool {
    request.task_type == TaskType::ASR
        && !request.is_realtime_asr_session()
        && request.prepared_asr_execution_shape().is_none()
        && request
            .model_variant
            .is_some_and(|variant| variant.family().hallucinates_on_non_speech())
}

/// A request whose whole input holds no speech.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AsrNoSpeechGate {
    scan: SpeechPresenceScan,
    input_samples: usize,
    sample_rate: u32,
}

impl AsrNoSpeechGate {
    /// Gate decision for canonical decoded audio: `Some` when no VAD frame
    /// reaches the speech planner's start threshold.
    pub(crate) fn evaluate(samples: &[f32], sample_rate: u32) -> Option<Self> {
        let scan = scan_speech_presence(samples, sample_rate, &AsrSpeechChunkConfig::default());
        scan.no_speech.then_some(Self {
            scan,
            input_samples: samples.len(),
            sample_rate,
        })
    }

    pub(crate) fn duration_secs(&self) -> f32 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.input_samples as f32 / self.sample_rate as f32
        }
    }

    pub(crate) fn diagnostics(&self) -> serde_json::Value {
        json!({
            "no_speech_gate": {
                "gated": true,
                "rule": "no_vad_frame_reaches_speech_threshold",
                "threshold": self.scan.threshold,
                "peak_score": self.scan.peak_score,
                "frames_scored": self.scan.frames_scored,
                "input_samples": self.input_samples,
                "sample_rate": self.sample_rate,
                "audio_seconds": self.duration_secs(),
            }
        })
    }

    /// The finished output of a gated request: an empty transcript with the
    /// input duration and the gate's diagnostics.
    pub(crate) fn engine_output(&self, request_id: &str) -> EngineOutput {
        let duration_secs = self.duration_secs();
        let mut output = OutputProcessor::new(self.sample_rate).process(
            ExecutorOutput {
                audio: Some(AudioOutput {
                    samples: Vec::new(),
                    sample_rate: self.sample_rate,
                    duration_secs,
                    streamed_samples: None,
                }),
                text: Some(String::new()),
                asr_diagnostics: Some(self.diagnostics()),
                ..ExecutorOutput::terminal(request_id.to_string())
            },
            0,
            Duration::ZERO,
        );
        output.input_audio_duration_secs = Some(duration_secs);
        output
    }

    /// The terminal stream marker a gated streaming request emits in place of
    /// model output, as the executor's final marker does.
    pub(crate) fn final_stream_chunk(request_id: &str) -> StreamingOutput {
        StreamingOutput::final_chunk(request_id.to_string(), 0, Vec::new(), 0)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::Path;

    use super::{asr_no_speech_gate_applies, AsrNoSpeechGate};
    use crate::catalog::ModelVariant;
    use crate::engine::EngineCoreRequest;
    use crate::runtime::audio_io::decode_audio_bytes_canonical;

    pub(crate) const SAMPLE_RATE: u32 = 16_000;

    /// Deterministic xorshift samples, uniform in `[-1, 1)`.
    fn uniform_noise(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    pub(crate) fn white_noise(len: usize, amplitude: f32, seed: u64) -> Vec<f32> {
        uniform_noise(len, seed)
            .into_iter()
            .map(|sample| sample * amplitude)
            .collect()
    }

    /// Pink noise (Paul Kellet's filter over white noise), scaled to `peak`.
    pub(crate) fn pink_noise(len: usize, peak: f32, seed: u64) -> Vec<f32> {
        let mut b = [0.0f32; 7];
        let mut samples = uniform_noise(len, seed)
            .into_iter()
            .map(|white| {
                b[0] = 0.99886 * b[0] + white * 0.055_517_9;
                b[1] = 0.99332 * b[1] + white * 0.075_075_9;
                b[2] = 0.96900 * b[2] + white * 0.153_852;
                b[3] = 0.86650 * b[3] + white * 0.310_485_6;
                b[4] = 0.55000 * b[4] + white * 0.532_952_2;
                b[5] = -0.7616 * b[5] - white * 0.016_898;
                let pink = b.iter().sum::<f32>() + white * 0.5362;
                b[6] = white * 0.115_926;
                pink
            })
            .collect::<Vec<_>>();
        let max = samples.iter().fold(0.0f32, |max, sample| max.max(sample.abs()));
        for sample in &mut samples {
            *sample *= peak / max;
        }
        samples
    }

    pub(crate) fn wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
        let mut bytes = std::io::Cursor::new(Vec::new());
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::new(&mut bytes, spec).expect("wav writer");
        for sample in samples {
            writer
                .write_sample((sample.clamp(-1.0, 1.0) * 32767.0).round() as i16)
                .expect("wav sample");
        }
        writer.finalize().expect("wav finalize");
        bytes.into_inner()
    }

    /// A repository speech fixture through the production canonical decode.
    fn canonical_fixture(name: &str) -> (Vec<f32>, u32) {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data")
            .join(name);
        let bytes = std::fs::read(&path).expect("speech fixture");
        decode_audio_bytes_canonical(&bytes).expect("canonical decode")
    }

    fn attenuated(samples: &[f32], db: f32) -> Vec<f32> {
        let gain = 10f32.powf(db / 20.0);
        samples.iter().map(|sample| sample * gain).collect()
    }

    #[test]
    fn digital_silence_and_low_level_noise_are_gated() {
        let len = SAMPLE_RATE as usize * 10;
        // Noise-floor levels whose frames stay below the speech threshold
        // (peak frame score <= 0.39 for every seed tried). Louder noise can
        // score single speech-like frames and is left to the model.
        let inputs = [
            ("digital silence", vec![0.0f32; len]),
            ("pink noise", pink_noise(len, 0.02, 4)),
            ("white noise", white_noise(len, 0.005, 4)),
        ];
        for (name, samples) in inputs {
            // Inputs travel as 16-bit WAV, like the live requests.
            let (decoded, sample_rate) =
                decode_audio_bytes_canonical(&wav_bytes(&samples, SAMPLE_RATE))
                    .expect("canonical decode");
            let gate = AsrNoSpeechGate::evaluate(&decoded, sample_rate)
                .unwrap_or_else(|| panic!("{name} must be gated"));
            assert_eq!(gate.duration_secs(), 10.0, "{name}");
            let diagnostics = gate.diagnostics();
            let peak = diagnostics["no_speech_gate"]["peak_score"]
                .as_f64()
                .expect("peak score");
            assert!(peak < 0.5, "{name} peak {peak}");
        }
    }

    #[test]
    fn speech_fixtures_are_never_gated_even_when_quiet() {
        for name in ["fox.wav", "test.wav"] {
            let (samples, sample_rate) = canonical_fixture(name);
            assert_eq!(sample_rate, SAMPLE_RATE);
            for db in [0.0, -20.0, -40.0] {
                assert!(
                    AsrNoSpeechGate::evaluate(&attenuated(&samples, db), sample_rate).is_none(),
                    "{name} at {db} dB must reach the model"
                );
            }
        }
    }

    #[test]
    fn quiet_speech_inside_long_silence_is_not_gated() {
        let (speech, sample_rate) = canonical_fixture("fox.wav");
        let silence = vec![0.0f32; sample_rate as usize * 20];
        let mut samples = silence.clone();
        samples.extend(attenuated(&speech, -20.0));
        samples.extend(silence);

        assert!(AsrNoSpeechGate::evaluate(&samples, sample_rate).is_none());
    }

    #[test]
    fn gate_applies_only_to_offline_asr_on_hallucinating_families() {
        let request = |variant: ModelVariant| {
            EngineCoreRequest::asr_bytes(vec![1, 2, 3]).with_model_variant(variant)
        };
        for variant in [
            ModelVariant::WhisperLargeV3Turbo,
            ModelVariant::Qwen3Asr17BGguf,
            ModelVariant::GraniteSpeech412BPlus,
            ModelVariant::VibeVoiceAsr,
            ModelVariant::VoxtralMini4BRealtime2602,
            ModelVariant::Lfm25Audio15BGguf,
        ] {
            assert!(asr_no_speech_gate_applies(&request(variant)), "{variant}");
        }
        for variant in [
            ModelVariant::ParakeetTdt06BV3,
            ModelVariant::Nemotron35AsrStreaming06B,
        ] {
            assert!(!asr_no_speech_gate_applies(&request(variant)), "{variant}");
        }
        assert!(!asr_no_speech_gate_applies(&EngineCoreRequest::asr_bytes(
            vec![1, 2, 3]
        )));

        let mut realtime = request(ModelVariant::VoxtralMini4BRealtime2602);
        realtime.enable_realtime_asr_ingress().expect("realtime ingress");
        assert!(!asr_no_speech_gate_applies(&realtime));

        let variant = ModelVariant::WhisperLargeV3Turbo;
        let mut routed = request(variant);
        routed
            .install_prepared_asr_audio(variant, vec![0.0; 160], SAMPLE_RATE)
            .expect("prepared audio");
        routed
            .install_prepared_asr_long_form_atomic()
            .expect("prepared route");
        assert!(routed.uses_asr_long_form_atomic());
        assert!(!asr_no_speech_gate_applies(&routed));
    }

    #[test]
    fn gated_output_is_a_finished_empty_transcript_with_input_duration() {
        let gate = AsrNoSpeechGate::evaluate(&vec![0.0; SAMPLE_RATE as usize * 3], SAMPLE_RATE)
            .expect("silence is gated");

        let output = gate.engine_output("gated");
        assert_eq!(output.request_id, "gated");
        assert_eq!(output.text.as_deref(), Some(""));
        assert!(output.is_finished);
        assert!(output.error.is_none());
        assert_eq!(output.input_audio_duration_secs, Some(3.0));
        assert_eq!(output.audio.duration_secs, 3.0);
        assert!(output.audio.samples.is_empty());
        let diagnostics = output.asr_diagnostics.expect("gate diagnostics");
        assert_eq!(diagnostics["no_speech_gate"]["gated"], true);
        assert_eq!(diagnostics["no_speech_gate"]["audio_seconds"], 3.0);

        let marker = AsrNoSpeechGate::final_stream_chunk("gated");
        assert!(marker.is_final);
        assert!(marker.text.is_none());
        assert!(marker.samples.is_empty());
    }
}
