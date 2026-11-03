//! Nemotron 3.5 Transcribe translates a long recording in short segments.
//!
//! The model translates sentence-length speech: NVIDIA evaluates its
//! translation on FLEURS and CoVoST 2 only, and its long-form training data is
//! transcription. Given a long recording and the translation instruction it
//! falls back to a transcript in the spoken language. A translation longer
//! than one segment is therefore cut at pauses into segments of at most
//! [`TRANSLATION_SEGMENT_ENV`] seconds, each translated as an ordinary request
//! with the same instruction (they run side by side, as concurrent requests
//! do), and the translations are joined in order. Transcription stays one
//! pass: speaker labels and word timestamps need the whole recording.

// Granite SAA's long-form chunk progress, which any chunked request reports.
use super::speaker_attribution::{
    granite_saa_chunk_progress, granite_saa_complete_progress, granite_saa_processing_progress,
    samples_to_seconds_f64,
};
use super::*;
use crate::runtime::audio_io::CANONICAL_SPEECH_SAMPLE_RATE;
use futures::StreamExt;
use izwi_asr_toolkit::spacing_between;

/// Longest segment of a translation, in seconds; `0` translates in one pass.
/// Read once per process.
pub(crate) const TRANSLATION_SEGMENT_ENV: &str =
    "IZWI_NEMOTRON_TRANSCRIBE_TRANSLATION_SEGMENT_SECONDS";
const DEFAULT_SEGMENT_SECS: f32 = 30.0;
const MIN_SEGMENT_SECS: f32 = 5.0;
/// Segments one translation keeps in flight: the CUDA decode width. More
/// would only queue in the engine.
const SEGMENTS_IN_FLIGHT: usize = 8;
/// A segment's decode budget per second of audio, plus a reserve: several
/// times what dense speech needs in any target language. A row books KV for
/// its whole budget, so it stays near what the audio can need.
const SEGMENT_TOKENS_PER_SECOND: f32 = 16.0;
const SEGMENT_TOKEN_RESERVE: usize = 64;

/// The segment length of `variant`'s `task`, or `None` when it runs in one
/// pass: it is not a Nemotron 3.5 Transcribe translation, or segments are
/// off.
pub(super) fn translation_segment_secs(
    variant: ModelVariant,
    task: &AsrTaskOptions,
) -> Option<f32> {
    static SECS: crate::env::ReadOnce<Option<f32>> = crate::env::ReadOnce::new(|| {
        parse_segment_secs(crate::env::value(TRANSLATION_SEGMENT_ENV).as_deref())
    });
    (variant.family() == ModelFamily::NemotronTranscribeAsr && task.translate_to.is_some())
        .then(|| SECS.get())
        .flatten()
}

/// `raw` as a segment length: unset or unparsable is the default, `0` turns
/// segments off, and a short length is raised to [`MIN_SEGMENT_SECS`].
fn parse_segment_secs(raw: Option<&str>) -> Option<f32> {
    let Some(raw) = raw else {
        return Some(DEFAULT_SEGMENT_SECS);
    };
    match raw.parse::<f32>() {
        Ok(0.0) => None,
        Ok(secs) if secs.is_finite() && secs > 0.0 => Some(secs.max(MIN_SEGMENT_SECS)),
        _ => {
            tracing::warn!(
                "{TRANSLATION_SEGMENT_ENV}={raw:?} is not a length in seconds; using {DEFAULT_SEGMENT_SECS}"
            );
            Some(DEFAULT_SEGMENT_SECS)
        }
    }
}

/// The planner settings for `segment_secs`: no overlap (two translations of
/// the same words cannot be merged), each cut in the longest pause near
/// four fifths of the limit, so a sentence rarely spans a seam.
fn segment_config(segment_secs: f32) -> AsrLongFormConfig {
    AsrLongFormConfig {
        target_chunk_secs: segment_secs * 0.8,
        hard_max_chunk_secs: segment_secs,
        overlap_secs: 0.0,
        silence_search_secs: (segment_secs * 0.15).clamp(2.0, 8.0),
        min_chunk_secs: (segment_secs * 0.25).min(8.0),
        prefer_longest_pause: true,
        ..AsrLongFormConfig::default()
    }
}

/// Segments of at most `segment_secs` that tile `samples`; one when the
/// recording fits.
pub(in crate::runtime) fn plan_translation_segments(
    samples: &[f32],
    sample_rate: u32,
    segment_secs: f32,
) -> Vec<AudioChunk> {
    let mut segments = plan_audio_chunks(samples, sample_rate, &segment_config(segment_secs), None);
    // The planner overlaps chunks by at least one sample; segments meet.
    for index in 1..segments.len() {
        segments[index].start_sample = segments[index - 1].end_sample;
    }
    segments
}

/// A segment's decode budget: the duration's, or the caller's when smaller.
fn segment_max_tokens(seconds: f64, requested: Option<usize>) -> usize {
    let budget = (seconds.max(0.0) as f32 * SEGMENT_TOKENS_PER_SECOND).ceil() as usize
        + SEGMENT_TOKEN_RESERVE;
    requested
        .map_or(budget, |requested| requested.min(budget))
        .max(1)
}

/// `samples` as a 32-bit float WAV, so a segment reaches the engine exactly.
pub(in crate::runtime) fn segment_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let encode = || -> std::result::Result<Vec<u8>, hound::Error> {
        let mut cursor = std::io::Cursor::new(Vec::with_capacity(44 + samples.len() * 4));
        let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
        for &sample in samples {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
        Ok(cursor.into_inner())
    };
    encode().map_err(|error| Error::InferenceError(format!("translation segment WAV: {error}")))
}

/// Appends one segment's translation to `text`; returns the delta a stream
/// emits for it (a space first where the scripts take one), or `None` for an
/// empty segment.
pub(in crate::runtime) fn append_segment(text: &mut String, segment: &str) -> Option<String> {
    let segment = segment.trim();
    if segment.is_empty() {
        return None;
    }
    let delta = format!("{}{segment}", spacing_between(text, segment));
    text.push_str(&delta);
    Some(delta)
}

/// The segments in the long-form `chunking` shape, with the first segment's
/// own diagnostics (its model and decode path).
fn segmented_diagnostics(
    segments: &[AudioChunk],
    sample_rate: u32,
    segment_secs: f32,
    first: Option<serde_json::Value>,
) -> serde_json::Value {
    let config = segment_config(segment_secs);
    let chunks = segments
        .iter()
        .map(|segment| {
            let start = samples_to_seconds_f64(segment.start_sample, sample_rate);
            let end = samples_to_seconds_f64(segment.end_sample, sample_rate);
            serde_json::json!({
                "start_sample": segment.start_sample,
                "end_sample": segment.end_sample,
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": end - start,
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "chunking": {
            "planner": "translation_segments",
            "chunk_count": segments.len(),
            "sample_rate": sample_rate,
            "audio_seconds": samples_to_seconds_f64(
                segments.last().map_or(0, |segment| segment.end_sample),
                sample_rate,
            ),
            "config": {
                "target_chunk_seconds": config.target_chunk_secs,
                "hard_max_chunk_seconds": config.hard_max_chunk_secs,
                "overlap_seconds": config.overlap_secs,
                "silence_search_seconds": config.silence_search_secs,
                "min_chunk_seconds": config.min_chunk_secs,
            },
            "chunks": chunks,
        },
        "first_chunk_diagnostics": first,
    })
}

impl RuntimeService {
    /// A Nemotron 3.5 Transcribe translation of `audio_bytes`, segment by
    /// segment, or `None` when the request runs in one pass (it is no such
    /// translation, or its audio fits one segment). Deltas and chunk progress
    /// come per finished segment, in order; a failed segment fails the
    /// request, and dropping the future cancels every segment in flight.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn translate_in_segments<F, P>(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        task: &AsrTaskOptions,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        on_delta: &mut F,
        on_progress: &mut P,
    ) -> Result<Option<AsrTranscription>>
    where
        F: FnMut(String) + Send,
        P: FnMut(AsrProgress) + Send,
    {
        let Some(segment_secs) = translation_segment_secs(variant, task) else {
            return Ok(None);
        };
        // Refuse a bad request before decoding its audio.
        validate_asr_language(variant, language)?;
        validate_asr_task(variant, language, prompt, task)?;
        AsrAudioInput::Bytes(audio_bytes).validate_retained_size()?;
        let source = audio_bytes.to_vec();
        let (samples, segments) = tokio::task::spawn_blocking(move || -> Result<_> {
            let (samples, sample_rate) = decode_audio_bytes_canonical(&source)?;
            let segments = plan_translation_segments(&samples, sample_rate, segment_secs);
            Ok((samples, segments))
        })
        .await
        .map_err(|error| {
            Error::InferenceError(format!("translation segment planning stopped: {error}"))
        })??;
        if segments.len() < 2 {
            return Ok(None);
        }
        let sample_rate = CANONICAL_SPEECH_SAMPLE_RATE;
        tracing::info!(
            %variant,
            segments = segments.len(),
            segment_secs,
            audio_secs = samples_to_seconds_f64(samples.len(), sample_rate),
            "translating a long recording in segments"
        );
        on_progress(granite_saa_processing_progress(&segments, sample_rate));
        let samples: Arc<[f32]> = samples.into();
        let mut outputs = futures::stream::iter(segments.iter().copied())
            .map(|segment| {
                let samples = samples.clone();
                let task = task.clone();
                async move {
                    let seconds = samples_to_seconds_f64(segment.len_samples(), sample_rate);
                    let wav = segment_wav(
                        &samples[segment.start_sample..segment.end_sample],
                        sample_rate,
                    )?;
                    drop(samples);
                    let admitted = self
                        .build_asr_request(
                            variant,
                            AsrAudioInput::Bytes(&wav),
                            language,
                            prompt,
                            task,
                            Some(segment_max_tokens(seconds, max_tokens)),
                            correlation_id,
                            runtime_context,
                            false,
                            false,
                        )
                        .await?;
                    drop(wav);
                    self.run_admitted_request(admitted).await
                }
            })
            .buffered(SEGMENTS_IN_FLIGHT);
        let mut text = String::new();
        let mut first_diagnostics = None;
        let mut index = 0;
        while let Some(output) = outputs.next().await {
            let output = output?;
            if let Some(delta) = append_segment(
                &mut text,
                &asr_transcript_text(output.text.as_deref().unwrap_or("")),
            ) {
                on_delta(delta);
            }
            if index == 0 {
                first_diagnostics = output.asr_diagnostics;
            }
            on_progress(granite_saa_chunk_progress(
                AsrProgressPhase::ChunkFinished,
                index,
                &segments[index],
                &segments,
                sample_rate,
            ));
            index += 1;
        }
        on_progress(granite_saa_complete_progress(&segments, sample_rate));
        Ok(Some(AsrTranscription {
            text,
            language: asr_response_language(language, task.translate_to.as_deref(), None),
            duration_secs: samples_to_seconds_f64(samples.len(), sample_rate) as f32,
            asr_diagnostics: Some(segmented_diagnostics(
                &segments,
                sample_rate,
                segment_secs,
                first_diagnostics,
            )),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone_with_pauses(seconds: f32, sample_rate: u32) -> Vec<f32> {
        // One second of tone, then half a second of silence, repeated.
        (0..(seconds * sample_rate as f32) as usize)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                if t % 1.5 < 1.0 {
                    0.3 * (t * 440.0 * std::f32::consts::TAU).sin()
                } else {
                    0.0
                }
            })
            .collect()
    }

    #[test]
    fn only_a_nemotron_translation_is_segmented() {
        let translate = AsrTaskOptions::new(Some("de".into()), Vec::new());
        let nemotron = ModelVariant::Nemotron35Transcribe30BA4BFp8;
        assert!(translation_segment_secs(nemotron, &translate).is_some());
        assert_eq!(
            translation_segment_secs(nemotron, &AsrTaskOptions::default()),
            None
        );
        assert_eq!(
            translation_segment_secs(ModelVariant::GraniteSpeech502B, &translate),
            None
        );
    }

    #[test]
    fn the_switch_reads_seconds_and_zero_turns_segments_off() {
        assert_eq!(parse_segment_secs(None), Some(DEFAULT_SEGMENT_SECS));
        assert_eq!(parse_segment_secs(Some("45")), Some(45.0));
        assert_eq!(parse_segment_secs(Some("0")), None);
        assert_eq!(parse_segment_secs(Some("2")), Some(MIN_SEGMENT_SECS));
        assert_eq!(parse_segment_secs(Some("soon")), Some(DEFAULT_SEGMENT_SECS));
        assert_eq!(parse_segment_secs(Some("-3")), Some(DEFAULT_SEGMENT_SECS));
    }

    #[test]
    fn segments_tile_the_recording_within_the_limit_and_cut_in_pauses() {
        let rate = 16_000;
        let samples = tone_with_pauses(130.0, rate);
        let segments = plan_translation_segments(&samples, rate, 30.0);
        assert!(segments.len() >= 5, "{segments:?}");
        assert_eq!(segments[0].start_sample, 0);
        assert_eq!(segments.last().unwrap().end_sample, samples.len());
        for pair in segments.windows(2) {
            assert_eq!(
                pair[0].end_sample, pair[1].start_sample,
                "no gap or overlap: {segments:?}"
            );
        }
        for segment in &segments {
            assert!(segment.len_samples() <= 30 * rate as usize, "{segment:?}");
        }
        // Every inner cut lands in silence.
        for segment in &segments[..segments.len() - 1] {
            let cut = segment.end_sample;
            assert!(samples[cut.saturating_sub(80)..cut + 80]
                .iter()
                .all(|s| *s == 0.0));
        }
        // A recording that fits is one segment.
        assert_eq!(
            plan_translation_segments(&samples[..25 * rate as usize], rate, 30.0).len(),
            1
        );
    }

    #[test]
    fn budgets_follow_the_audio_and_the_caller_cap() {
        assert_eq!(segment_max_tokens(30.0, None), 30 * 16 + 64);
        assert_eq!(segment_max_tokens(30.0, Some(100)), 100);
        assert_eq!(segment_max_tokens(30.0, Some(100_000)), 30 * 16 + 64);
        assert_eq!(segment_max_tokens(0.0, Some(0)), 1);
    }

    #[test]
    fn segments_join_with_a_space_except_between_unspaced_scripts() {
        let mut text = String::new();
        assert_eq!(
            append_segment(&mut text, " Hello there. "),
            Some("Hello there.".into())
        );
        assert_eq!(append_segment(&mut text, "   "), None);
        assert_eq!(
            append_segment(&mut text, "How are you?"),
            Some(" How are you?".into())
        );
        assert_eq!(text, "Hello there. How are you?");
        let mut japanese = String::new();
        append_segment(&mut japanese, "こんにちは。");
        assert_eq!(
            append_segment(&mut japanese, "元気ですか。"),
            Some("元気ですか。".into())
        );
        assert_eq!(japanese, "こんにちは。元気ですか。");
    }

    #[test]
    fn a_segment_wav_decodes_to_the_same_samples() {
        let samples = tone_with_pauses(2.0, 16_000);
        let wav = segment_wav(&samples, 16_000).unwrap();
        let (decoded, rate) = decode_audio_bytes_canonical(&wav).unwrap();
        assert_eq!(rate, 16_000);
        assert_eq!(decoded, samples);
    }
}
