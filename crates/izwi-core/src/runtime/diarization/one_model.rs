//! One-model diarization: a model that transcribes, tags speakers and times
//! its words itself (Nemotron 3.5 Transcribe) diarizes a recording in one
//! transcription, with no diarizer, pipeline ASR or aligner loaded.
//!
//! The transcription's `ctc_timestamps` (NeMo's public object: `words` with
//! the speaker tag that wrote them, the embedded diarizer's `diarization`
//! segments per column, and `speaker_tag_to_diarization_speaker`) become the
//! job's artifacts. Speakers are the diarizer's columns, `SPEAKER_NN` as in
//! the pipeline; a word takes the column its tag was assigned to, and a tag
//! without one leaves its words unattributed.

use std::collections::{BTreeSet, HashMap};

use serde::Deserialize;

use super::{
    build_utterances, format_utterance_transcript, DiarizationAudioInput, TranscriptAudioInput,
    UNKNOWN_SPEAKER,
};
use crate::error::{Error, Result};
use crate::runtime::asr::AsrAudioInput;
use crate::runtime::service::RuntimeService;
use crate::runtime::types::{
    DiarizationConfig, DiarizationPipelineOptions, DiarizationSegment, DiarizationTranscriptResult,
    DiarizationWord,
};
use crate::ModelVariant;

#[derive(Debug, Default, Deserialize)]
struct CtcTimestamps {
    #[serde(default)]
    words: Vec<CtcWord>,
    #[serde(default)]
    diarization: Vec<CtcSegment>,
    #[serde(default)]
    speaker_tag_to_diarization_speaker: HashMap<String, Option<usize>>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CtcWord {
    word: String,
    start: f64,
    end: f64,
    #[serde(default)]
    speaker: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CtcSegment {
    speaker: usize,
    start: f64,
    end: f64,
}

fn speaker_label(column: usize) -> String {
    format!("SPEAKER_{column:02}")
}

/// The job's artifacts from one speaker-tagged transcription: its text and
/// `asr_diagnostics`.
pub(super) fn one_model_transcript(
    variant: ModelVariant,
    text: &str,
    diagnostics: Option<&serde_json::Value>,
    duration_secs: f32,
    config: &DiarizationConfig,
) -> DiarizationTranscriptResult {
    let mut warnings = Vec::new();
    let timestamps = match diagnostics.and_then(|d| d.get("ctc_timestamps")) {
        Some(value) => {
            serde_json::from_value::<CtcTimestamps>(value.clone()).unwrap_or_else(|err| {
                warnings.push(format!("{variant} reported unreadable timestamps: {err}"));
                CtcTimestamps::default()
            })
        }
        None => {
            warnings.push(format!("{variant} reported no word timestamps."));
            CtcTimestamps::default()
        }
    };
    if let Some(error) = &timestamps.error {
        warnings.push(format!(
            "{variant} could not time its words ({error}); speaker segments only."
        ));
    }
    let mut segments = timestamps
        .diarization
        .iter()
        .filter(|segment| segment.end > segment.start)
        .map(|segment| DiarizationSegment {
            speaker: speaker_label(segment.speaker),
            start_secs: segment.start as f32,
            end_secs: segment.end as f32,
            confidence: None,
        })
        .collect::<Vec<_>>();
    segments.sort_by(|a, b| a.start_secs.total_cmp(&b.start_secs));
    let column_of = |tag: Option<&str>| -> Option<usize> {
        timestamps
            .speaker_tag_to_diarization_speaker
            .get(tag?)
            .copied()
            .flatten()
    };
    // Event and style tags are aligned as units but are not words.
    let words = timestamps
        .words
        .iter()
        .filter(|word| !(word.word.starts_with('<') && word.word.ends_with('>')))
        .map(|word| {
            let speaker = column_of(word.speaker.as_deref())
                .map_or_else(|| UNKNOWN_SPEAKER.to_string(), speaker_label);
            let (start_secs, end_secs) = (word.start as f32, word.end as f32);
            let middle = (start_secs + end_secs) / 2.0;
            DiarizationWord {
                overlaps_segment: segments.iter().any(|segment| {
                    segment.speaker == speaker
                        && segment.start_secs <= middle
                        && middle <= segment.end_secs
                }),
                word: word.word.clone(),
                speaker,
                start_secs,
                end_secs,
                speaker_confidence: None,
            }
        })
        .collect::<Vec<_>>();
    let word_texts = words
        .iter()
        .map(|word| word.word.clone())
        .collect::<Vec<_>>();
    let utterances = build_utterances(&words, &word_texts);
    let raw_transcript = if utterances.is_empty() {
        text.to_string()
    } else {
        format_utterance_transcript(&utterances)
    };
    let unattributed_words = words
        .iter()
        .filter(|word| word.speaker == UNKNOWN_SPEAKER)
        .count();
    let speakers = segments
        .iter()
        .map(|segment| segment.speaker.as_str())
        .chain(words.iter().map(|word| word.speaker.as_str()))
        .filter(|speaker| *speaker != UNKNOWN_SPEAKER)
        .collect::<BTreeSet<_>>();
    let speaker_count = speakers.len();
    if config.min_speakers.is_some_and(|min| speaker_count < min) {
        warnings.push(format!(
            "{variant} found {speaker_count} speaker(s), below the requested minimum {}.",
            config.min_speakers.unwrap_or_default()
        ));
    }
    if config.max_speakers.is_some_and(|max| speaker_count > max) {
        warnings.push(format!(
            "{variant} found {speaker_count} speaker(s), above the requested maximum {}.",
            config.max_speakers.unwrap_or_default()
        ));
    }
    DiarizationTranscriptResult {
        alignment_coverage: if words.is_empty() {
            0.0
        } else {
            (words.len() - unattributed_words) as f32 / words.len() as f32
        },
        segments,
        words,
        utterances,
        asr_text: text.to_string(),
        transcript: raw_transcript.clone(),
        raw_transcript,
        duration_secs,
        speaker_count,
        unattributed_words,
        warnings,
    }
}

impl RuntimeService {
    /// Diarizes with `variant` alone: one background transcription with
    /// speaker tags and word timestamps.
    pub(super) async fn diarize_in_one_model(
        &self,
        audio_input: TranscriptAudioInput<'_>,
        variant: ModelVariant,
        config: &DiarizationConfig,
        options: &DiarizationPipelineOptions,
    ) -> Result<DiarizationTranscriptResult> {
        let ensure_not_cancelled = || {
            if options.cancellation.is_cancelled() {
                return Err(Error::Cancelled(format!(
                    "{variant} diarization was cancelled"
                )));
            }
            Ok(())
        };
        ensure_not_cancelled()?;
        let encoded;
        let audio = match audio_input {
            TranscriptAudioInput::Encoded(DiarizationAudioInput::Base64(audio)) => {
                AsrAudioInput::Base64(audio)
            }
            TranscriptAudioInput::Encoded(DiarizationAudioInput::Bytes(audio)) => {
                AsrAudioInput::Bytes(audio)
            }
            // Canonical 16 kHz mono samples go in as a WAV, which the ASR
            // route decodes without resampling.
            TranscriptAudioInput::Samples(samples) => {
                encoded = crate::audio::AudioEncoder::new(16_000, 1)
                    .encode(&samples, crate::audio::AudioFormat::Wav)?;
                AsrAudioInput::Bytes(&encoded)
            }
        };
        let output = self
            .speaker_tagged_asr_output(audio, variant, options.language.as_deref())
            .await?;
        ensure_not_cancelled()?;
        let duration_secs = output
            .input_audio_duration_secs
            .unwrap_or(output.audio.duration_secs);
        let text = output.text.unwrap_or_default();
        if text.trim().is_empty() && output.asr_diagnostics.is_none() {
            return Err(Error::InferenceError(format!(
                "{variant} returned no transcript to diarize"
            )));
        }
        Ok(one_model_transcript(
            variant,
            &text,
            output.asr_diagnostics.as_ref(),
            duration_secs,
            config,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VARIANT: ModelVariant = ModelVariant::Nemotron35Transcribe30BA4BFp8;

    fn diagnostics() -> serde_json::Value {
        serde_json::json!({
            "ctc_timestamps": {
                "words": [
                    {"word": "hello", "start": 0.1, "end": 0.4, "speaker": "0"},
                    {"word": "there", "start": 0.5, "end": 0.8, "speaker": "0"},
                    {"word": "<ev:laughter>", "start": 0.9, "end": 1.0, "speaker": "0"},
                    {"word": "hi", "start": 1.2, "end": 1.4, "speaker": "1"},
                    {"word": "stray", "start": 1.6, "end": 1.8, "speaker": "2"}
                ],
                "diarization": [
                    {"speaker": 3, "start": 0.0, "end": 1.0},
                    {"speaker": 5, "start": 1.1, "end": 1.5}
                ],
                "speaker_tag_to_diarization_speaker": {"0": 3, "1": 5, "2": null},
                "error": null
            }
        })
    }

    #[test]
    fn words_take_their_tags_column_and_events_are_dropped() {
        let result = one_model_transcript(
            VARIANT,
            "hello there hi stray",
            Some(&diagnostics()),
            2.0,
            &DiarizationConfig::default(),
        );
        assert_eq!(result.segments.len(), 2);
        assert_eq!(result.segments[0].speaker, "SPEAKER_03");
        let speakers = result
            .words
            .iter()
            .map(|word| {
                (
                    word.word.as_str(),
                    word.speaker.as_str(),
                    word.overlaps_segment,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            speakers,
            [
                ("hello", "SPEAKER_03", true),
                ("there", "SPEAKER_03", true),
                ("hi", "SPEAKER_05", true),
                ("stray", UNKNOWN_SPEAKER, false),
            ]
        );
        assert_eq!(result.speaker_count, 2);
        assert_eq!(result.unattributed_words, 1);
        assert!((result.alignment_coverage - 0.75).abs() < 1e-6);
        assert!(result.transcript.contains("SPEAKER_03"));
        assert!(result.warnings.is_empty(), "{:?}", result.warnings);
    }

    #[test]
    fn a_failed_alignment_keeps_the_segments_and_says_so() {
        let mut diagnostics = diagnostics();
        diagnostics["ctc_timestamps"]["words"] = serde_json::json!([]);
        diagnostics["ctc_timestamps"]["error"] = serde_json::json!("alignment_failed");
        let config = DiarizationConfig {
            min_speakers: Some(3),
            ..Default::default()
        };
        let result = one_model_transcript(VARIANT, "hello", Some(&diagnostics), 2.0, &config);
        assert_eq!(result.segments.len(), 2);
        assert!(result.words.is_empty());
        assert_eq!(result.transcript, "hello");
        assert_eq!(result.warnings.len(), 2, "{:?}", result.warnings);
        let result =
            one_model_transcript(VARIANT, "hello", None, 2.0, &DiarizationConfig::default());
        assert!(result.segments.is_empty());
        assert_eq!(result.warnings.len(), 1);
    }
}
