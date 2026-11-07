use std::collections::VecDeque;
use std::ops::Range;
use std::sync::Arc;

use izwi_core::runtime_models::architectures::fish_drama3::chunking::{
    plan_chunk_spans, split_text as fish_drama3_split_text, CHUNK_MAX_SPEAKERS,
    HISTORY_RECENT_TURNS,
};
use izwi_core::text_segments::pack_sentences;
use izwi_core::{
    GenerationRequest, GenerationResult, ModelVariant, RuntimeService, SpeechChunkTurn,
    SpeechFrameCodes, TtsStopReason,
};

use crate::api::speech_takes::SpeechTake;
use crate::batch_runtime::speech_progress::SpeechChunkTurnCheckpoint;

const DEFAULT_CHUNK_MAX_CHARS: usize = 480;
const CHUNK_MAX_CHARS_MIN: usize = 80;
const CHUNK_MAX_CHARS_MAX: usize = 4000;
/// Qwen3-TTS long-form segments: whole sentences packed up to about the
/// medium test paragraph (45 words, 260 characters), which ends on EOS on
/// every backend. A 460-character segment of the 250-word paragraph ran to
/// its 663-frame budget and babbled (WER 0.32 on CPU); the reference model
/// generates sentence by sentence. Text longer than this splits even when
/// one request could hold it.
const QWEN_SEGMENT_MAX_CHARS: usize = 300;
/// Maya1 long-form segments: whole sentences up to about 11 s of speech,
/// inside the 1-14 s clips it was trained on; its automatic budget reaches
/// the per-request cap near 160 characters.
const MAYA1_SEGMENT_MAX_CHARS: usize = 160;
use crate::api::tts_policy::MAYA2_SEGMENT_MAX_CHARS;

/// Silence between two long-form segments: a sentence pause, not the raw
/// back-to-back concatenation that ran sentences together (TPIPE-5).
const SEGMENT_PAUSE_SECS: f32 = 0.12;
/// A short fade on each side of the pause, so no segment ends or starts on
/// a click.
const SEGMENT_FADE_SECS: f32 = 0.01;

fn seconds_to_samples(secs: f32, sample_rate: u32) -> usize {
    (secs * sample_rate as f32).round().max(1.0) as usize
}

/// Samples in the fade on each side of a segment join.
pub(crate) fn segment_fade_samples(sample_rate: u32) -> usize {
    seconds_to_samples(SEGMENT_FADE_SECS, sample_rate)
}

/// Samples of silence between two segments.
pub(crate) fn segment_pause_samples(sample_rate: u32) -> usize {
    seconds_to_samples(SEGMENT_PAUSE_SECS, sample_rate)
}

/// Fade the first samples of a segment in from silence.
pub(crate) fn fade_in_segment(samples: &mut [f32], sample_rate: u32) {
    let fade = segment_fade_samples(sample_rate).min(samples.len());
    for (index, sample) in samples.iter_mut().take(fade).enumerate() {
        *sample *= (index + 1) as f32 / (fade + 1) as f32;
    }
}

/// Fade the last samples of a segment out to silence.
pub(crate) fn fade_out_segment(samples: &mut [f32], sample_rate: u32) {
    let fade = segment_fade_samples(sample_rate).min(samples.len());
    let tail = samples.len() - fade;
    for (index, sample) in samples[tail..].iter_mut().enumerate() {
        *sample *= (fade - index) as f32 / (fade + 1) as f32;
    }
}

/// The pause before a later segment, followed by that segment faded in.
pub(crate) fn pause_then_segment(segment: &[f32], sample_rate: u32) -> Vec<f32> {
    let pause = segment_pause_samples(sample_rate);
    let mut out = Vec::with_capacity(pause + segment.len());
    out.resize(pause, 0.0);
    out.extend_from_slice(segment);
    fade_in_segment(&mut out[pause..], sample_rate);
    out
}

/// Append one long-form segment: the first as is, a later one after a
/// faded-out tail and a short pause, faded in.
pub(crate) fn append_segment(joined: &mut Vec<f32>, segment: &[f32], sample_rate: u32) {
    if joined.is_empty() {
        joined.extend_from_slice(segment);
        return;
    }
    fade_out_segment(joined, sample_rate);
    joined.extend(pause_then_segment(segment, sample_rate));
}

fn chunk_max_chars() -> usize {
    configured_chunk_max_chars().unwrap_or(DEFAULT_CHUNK_MAX_CHARS)
}

fn configured_chunk_max_chars() -> Option<usize> {
    crate::env_config::tuning()
        .tts_long_form_chunk_max_chars
        .map(|value| value.clamp(CHUNK_MAX_CHARS_MIN, CHUNK_MAX_CHARS_MAX))
}

/// The most characters one long-form segment of `variant` packs: the
/// configured cap, else Qwen3-TTS's segment size or the default.
fn segment_max_chars(variant: ModelVariant) -> usize {
    configured_chunk_max_chars().unwrap_or(match variant.family() {
        izwi_core::ModelFamily::Qwen3Tts => QWEN_SEGMENT_MAX_CHARS,
        izwi_core::ModelFamily::Maya1Tts => MAYA1_SEGMENT_MAX_CHARS,
        izwi_core::ModelFamily::Maya2Tts => MAYA2_SEGMENT_MAX_CHARS,
        _ => DEFAULT_CHUNK_MAX_CHARS,
    })
}

/// Whether text under an automatic budget splits into long-form segments:
/// when one request cannot speak it, or (Qwen3-TTS, Maya1, Maya 2 Native)
/// when it is longer than one segment.
fn splits_into_segments(variant: ModelVariant, text: &str) -> bool {
    crate::api::tts_policy::tts_text_exceeds_one_request(variant, text)
        || (matches!(
            variant.family(),
            izwi_core::ModelFamily::Qwen3Tts
                | izwi_core::ModelFamily::Maya1Tts
                | izwi_core::ModelFamily::Maya2Tts
        ) && text.trim().chars().count() > segment_max_chars(variant))
}

/// The inline tags of `chunk` in `variant`'s syntax: Maya1's `<tag>`s,
/// Maya 2 Native's lowercase `[tag]`s.
fn inline_tags(variant: ModelVariant, chunk: &str) -> Vec<&str> {
    if variant.family() == izwi_core::ModelFamily::Maya2Tts {
        let mut tags = Vec::new();
        let mut rest = chunk;
        while let Some(open) = rest.find('[') {
            let after = &rest[open + 1..];
            let Some(close) = after.find(']') else { break };
            let name = &after[..close];
            if !name.is_empty() && name.chars().all(|ch| ch.is_ascii_lowercase() || ch == ' ') {
                tags.push(&rest[open..open + close + 2]);
            }
            rest = &after[close + 1..];
        }
        return tags;
    }
    izwi_core::runtime_models::architectures::maya1::prompt::inline_tags(chunk).collect()
}

/// Merge a segment that holds only inline tags (`<laugh>` or `[laughs]` on a
/// line of its own, or after a sentence's full stop) into its neighbour, so
/// the model performs it beside the words it belongs to rather than in a
/// request of its own.
fn attach_lone_tags(variant: ModelVariant, chunks: Vec<String>) -> Vec<String> {
    let lone = |chunk: &str| {
        let mut rest = chunk.to_string();
        for tag in inline_tags(variant, chunk) {
            rest = rest.replacen(tag, "", 1);
        }
        rest.trim().is_empty()
    };
    let mut merged: Vec<String> = Vec::with_capacity(chunks.len());
    let mut leading = String::new();
    for chunk in chunks {
        if lone(&chunk) {
            match merged.last_mut() {
                Some(previous) => {
                    previous.push(' ');
                    previous.push_str(chunk.trim());
                }
                None => {
                    leading.push_str(chunk.trim());
                    leading.push(' ');
                }
            }
        } else if leading.is_empty() {
            merged.push(chunk);
        } else {
            merged.push(format!("{}{}", std::mem::take(&mut leading), chunk.trim()));
        }
    }
    if !leading.is_empty() {
        merged.push(leading.trim().to_string());
    }
    merged
}

/// A stable, versioned partition of the original UTF-8 input. Audio limits are
/// assigned after partitioning, never used as a switch for complete-text planning.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub(crate) struct SpeechTextPlan {
    pub version: u32,
    pub text_sha256: String,
    pub segments: Vec<std::ops::Range<usize>>,
    pub max_output_frames: Option<usize>,
    /// A chunk session's chunks (Fish Audio Drama 3), one per segment: the
    /// text each speaks as the model's chunker wrote it. Each segment's
    /// range is the input it covers, for progress and recovery; a chunk of
    /// tagged text is rewritten (its pieces re-tagged), so it is not that
    /// slice.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chunks: Vec<String>,
    /// A chunk session's budget per chunk when the client named one
    /// (`max_tokens`, at most 4,096 frames); unset, each chunk takes its
    /// automatic budget. Fish S2 bounds the whole plan instead
    /// (`max_output_frames`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_frames: Option<usize>,
}

/// A chunk session's plan (its chunks and per-chunk budgets).
const CHUNK_SESSION_PLAN_VERSION: u32 = 2;

/// One segment a plan still has to speak: the input bytes it covers and the
/// text the model is given for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PlannedSegment {
    pub range: Range<usize>,
    pub text: String,
}

fn check_speech_text_bytes(text: &str) -> Result<(), izwi_core::Error> {
    let max_bytes = crate::api::tts_policy::tts_max_text_bytes();
    if text.len() > max_bytes {
        return Err(izwi_core::Error::InvalidInput(format!(
            "Speech text exceeds configured limit of {max_bytes} bytes"
        )));
    }
    Ok(())
}

/// Refuse a plan whose PCM would pass the deployment's output limit (or
/// WAV's capacity): `frames` of 2,048 16-bit samples each.
fn check_speech_plan_pcm(frames: usize) -> Result<(), izwi_core::Error> {
    let max_pcm_bytes = crate::env_config::tuning()
        .tts_stream_max_pcm_bytes
        .unwrap_or(u32::MAX as usize - 44)
        .min(u32::MAX as usize - 44);
    if frames
        .checked_mul(2048 * 2)
        .is_none_or(|bytes| bytes > max_pcm_bytes)
    {
        return Err(izwi_core::Error::InvalidInput("Speech plan exceeds configured PCM/WAV output limit; shorten the text or raise the deployment limit within WAV capacity".into()));
    }
    Ok(())
}

impl SpeechTextPlan {
    /// The plan of `variant`'s speech: a chunk session for a model that
    /// continues speech across chunks, else Fish S2's segment plan.
    pub fn for_variant(
        variant: ModelVariant,
        text: &str,
        max_output_frames: usize,
    ) -> Result<Self, izwi_core::Error> {
        if variant.continues_speech_across_chunks() {
            Self::chunk_session(variant, text, max_output_frames)
        } else {
            Self::fish(text, max_output_frames)
        }
    }

    /// Fish Audio Drama 3's chunk session (§3.7, §7.5): the model's own
    /// speaker-aware chunker cuts the text (about 300 bytes a chunk, at most
    /// three speakers, sentence ends first, tags and protected spans kept
    /// whole). Each chunk is one engine request with its own budget: the
    /// client's `max_tokens` (at most 4,096 frames) or the chunk's automatic
    /// one (S2's speaking estimate times 2.5, at most 1,024).
    pub fn chunk_session(
        variant: ModelVariant,
        text: &str,
        max_tokens: usize,
    ) -> Result<Self, izwi_core::Error> {
        use sha2::{Digest, Sha256};
        check_speech_text_bytes(text)?;
        if text.trim().is_empty() {
            return Err(izwi_core::Error::InvalidInput(
                "Speech input is empty".into(),
            ));
        }
        let chunks = plan_chunk_spans(text, false);
        if chunks.len() > 1 && !crate::env_config::tuning().fish_drama3_long_form_enabled {
            return Err(izwi_core::Error::InvalidInput(
                "Fish Audio Drama 3 long-form generation is disabled by this deployment".into(),
            ));
        }
        // Segments tile the input: each runs to its chunk's last byte, the
        // last to the end.
        let mut segments = Vec::with_capacity(chunks.len());
        let mut start = 0;
        for (index, (_, span)) in chunks.iter().enumerate() {
            let end = if index + 1 == chunks.len() {
                text.len()
            } else {
                span.end
            };
            if end <= start {
                return Err(izwi_core::Error::InferenceError(
                    "Speech plan chunks are out of order".into(),
                ));
            }
            segments.push(start..end);
            start = end;
        }
        let chunk_frames = (max_tokens > 0)
            .then(|| {
                crate::api::tts_policy::resolve_tts_output_frames(variant, text, Some(max_tokens))
            })
            .flatten();
        let chunks: Vec<String> = chunks.into_iter().map(|(chunk, _)| chunk).collect();
        let estimated_frames = chunks.iter().try_fold(0usize, |total, chunk| {
            total.checked_add(chunk_frames.unwrap_or_else(|| {
                crate::api::tts_policy::resolve_tts_output_frames(variant, chunk, None)
                    .unwrap_or(ModelVariant::FISH_DRAMA3_MAX_OUTPUT_FRAMES)
            }))
        });
        check_speech_plan_pcm(estimated_frames.ok_or_else(|| {
            izwi_core::Error::InvalidInput("Speech output size overflow".into())
        })?)?;
        Ok(Self {
            version: CHUNK_SESSION_PLAN_VERSION,
            text_sha256: hex::encode(Sha256::digest(text.as_bytes())),
            segments,
            max_output_frames: None,
            chunks,
            chunk_frames,
        })
    }

    /// Whether this plan is a chunk session.
    pub fn is_chunk_session(&self) -> bool {
        !self.chunks.is_empty()
    }

    /// The text each segment speaks, in order.
    pub fn segment_texts(&self, text: &str) -> Vec<String> {
        if self.is_chunk_session() {
            return self.chunks.clone();
        }
        self.segments
            .iter()
            .map(|range| text[range.clone()].to_string())
            .collect()
    }

    /// The segments still to speak once the first `completed` bytes of
    /// `text` are done, in order. A Fish S2 segment that `completed` falls
    /// inside (a halved one whose first part finished) resumes from it; a
    /// chunk that it falls inside is halved again, as it was, and only the
    /// parts after it remain.
    pub(crate) fn queue(
        &self,
        text: &str,
        completed: usize,
    ) -> Result<VecDeque<PlannedSegment>, izwi_core::Error> {
        let mut queue = VecDeque::with_capacity(self.segments.len());
        if !self.is_chunk_session() {
            for range in &self.segments {
                if range.end <= completed {
                    continue;
                }
                let range = range.start.max(completed)..range.end;
                queue.push_back(PlannedSegment {
                    text: text[range.clone()].to_string(),
                    range,
                });
            }
            return Ok(queue);
        }
        if self.chunks.len() != self.segments.len() {
            return Err(izwi_core::Error::InvalidInput(
                "Speech plan chunks do not match its segments".into(),
            ));
        }
        let mut pending: Vec<PlannedSegment> = self
            .segments
            .iter()
            .zip(&self.chunks)
            .rev()
            .map(|(range, chunk)| PlannedSegment {
                range: range.clone(),
                text: chunk.clone(),
            })
            .collect();
        while let Some(segment) = pending.pop() {
            if segment.range.end <= completed {
                continue;
            }
            if segment.range.start >= completed {
                queue.push_back(segment);
                continue;
            }
            let halves = halve_chunk(text, &segment).ok_or_else(|| {
                izwi_core::Error::InvalidInput("Invalid speech recovery text offset".into())
            })?;
            pending.extend(halves.into_iter().rev());
        }
        Ok(queue)
    }

    /// `segment` split in two or more, for a segment whose prompt was too
    /// long for the model: Fish S2's halves at its middle character; a
    /// chunk's cut again by the model's chunker at half its bytes (tags and
    /// protected spans kept whole), each part placed in the chunk's input
    /// range by its share of the text. `None` when it cannot be split.
    pub(crate) fn halve(
        &self,
        text: &str,
        segment: &PlannedSegment,
    ) -> Option<Vec<PlannedSegment>> {
        if self.is_chunk_session() {
            return halve_chunk(text, segment);
        }
        let range = segment.range.clone();
        let count = text[range.clone()].chars().count();
        if count <= 1 {
            return None;
        }
        let middle = range.start + text[range.clone()].char_indices().nth(count / 2)?.0;
        Some(vec![
            PlannedSegment {
                text: text[range.start..middle].to_string(),
                range: range.start..middle,
            },
            PlannedSegment {
                text: text[middle..range.end].to_string(),
                range: middle..range.end,
            },
        ])
    }

    /// The frame budget of a segment speaking `segment`: a chunk session's
    /// per-chunk budget, else the segment's automatic one.
    fn segment_frames(&self, variant: ModelVariant, segment: &str) -> usize {
        self.chunk_frames.unwrap_or_else(|| {
            crate::api::tts_policy::resolve_tts_output_frames(variant, segment, None)
                .unwrap_or(4096)
        })
    }

    pub fn fish(text: &str, max_output_frames: usize) -> Result<Self, izwi_core::Error> {
        use sha2::{Digest, Sha256};
        check_speech_text_bytes(text)?;
        let mut segments = Vec::new();
        let mut start = 0;
        let mut chars = 0;
        let mut boundary = None;
        let mut sentence_boundary = None;
        // Keep Fish segments comfortably below its per-call context/output limit.
        let target = chunk_max_chars().min(480);
        for (offset, ch) in text.char_indices() {
            if chars >= target {
                let end = sentence_boundary
                    .filter(|end| *end > start)
                    .or_else(|| boundary.filter(|end| *end > start))
                    .unwrap_or(offset);
                segments.push(start..end);
                start = end;
                chars = text[start..offset].chars().count();
                boundary = None;
                sentence_boundary = None;
            }
            chars += 1;
            if matches!(ch, '。' | '！' | '？' | '\n' | ';' | '；') {
                sentence_boundary = Some(offset + ch.len_utf8());
            } else if matches!(ch, '.' | '!' | '?')
                && text[offset + ch.len_utf8()..]
                    .chars()
                    .next()
                    .is_none_or(char::is_whitespace)
            {
                let word = text[start..offset].split_whitespace().last().unwrap_or("");
                if ch != '.' || word.chars().count() > 3 {
                    sentence_boundary = Some(offset + ch.len_utf8());
                }
            }
            if ch.is_whitespace() || matches!(ch, '。' | '！' | '？' | ';' | '；') {
                boundary = Some(offset + ch.len_utf8());
            }
        }
        if start < text.len() && !text[start..].trim().is_empty() {
            segments.push(start..text.len());
        } else if let Some(last) = segments.last_mut() {
            last.end = text.len();
        }
        // Blank layout alone is not an inference segment. Byte ranges still
        // identify original content for durable progress and recovery.
        segments.retain(|range| !text[range.clone()].trim().is_empty());
        if segments.len() > 1 && !crate::env_config::tuning().fish_long_form_enabled {
            return Err(izwi_core::Error::InvalidInput(
                "Fish long-form generation is disabled by this deployment".into(),
            ));
        }
        let estimated_frames = segments
            .iter()
            .try_fold(0usize, |total, range| {
                total.checked_add(
                    crate::api::tts_policy::resolve_tts_output_frames(
                        ModelVariant::FishAudioS2Pro,
                        &text[range.clone()],
                        None,
                    )
                    .unwrap_or(4096),
                )
            })
            .ok_or_else(|| izwi_core::Error::InvalidInput("Speech output size overflow".into()))?;
        let estimated_frames = if max_output_frames > 0 {
            estimated_frames.min(max_output_frames)
        } else {
            estimated_frames
        };
        check_speech_plan_pcm(estimated_frames)?;
        Ok(Self {
            version: 1,
            text_sha256: hex::encode(Sha256::digest(text.as_bytes())),
            segments,
            max_output_frames: (max_output_frames > 0).then_some(max_output_frames),
            chunks: Vec::new(),
            chunk_frames: None,
        })
    }
}

/// A chunk cut again by the model's chunker at half its bytes, each part
/// given its share of the chunk's input range (on character boundaries of
/// `source`). Deterministic, so a recovered plan halves a chunk exactly as
/// the attempt before it did.
fn halve_chunk(source: &str, segment: &PlannedSegment) -> Option<Vec<PlannedSegment>> {
    let pieces = fish_drama3_split_text(
        &segment.text,
        (segment.text.len() / 2).max(1),
        CHUNK_MAX_SPEAKERS,
    );
    if pieces.len() < 2 {
        return None;
    }
    let total = pieces.iter().map(String::len).sum::<usize>().max(1);
    let span = segment.range.len();
    let count = pieces.len();
    let mut halves = Vec::with_capacity(count);
    let (mut start, mut cumulative) = (segment.range.start, 0usize);
    for (index, piece) in pieces.into_iter().enumerate() {
        cumulative += piece.len();
        let end = if index + 1 == count {
            segment.range.end
        } else {
            let mut end = segment.range.start + span.checked_mul(cumulative)? / total;
            while !source.is_char_boundary(end) {
                end -= 1;
            }
            end
        };
        if end <= start {
            return None;
        }
        halves.push(PlannedSegment {
            range: start..end,
            text: piece,
        });
        start = end;
    }
    Some(halves)
}

/// A chunk session's history as the server carries it from one chunk
/// request to the next (§7.5, the family's `FishDrama3History` policy): the
/// first chunk that made audio, pinned, and the latest others. Each chunk's
/// prompt holds as many of them as fit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SpeechChunkHistory {
    pinned: Option<SpeechChunkTurn>,
    recent: VecDeque<SpeechChunkTurn>,
}

impl SpeechChunkHistory {
    /// Records a completed chunk; a chunk without frames is not history.
    pub(crate) fn push(&mut self, turn: SpeechChunkTurn) {
        if turn.codes.frames() == 0 {
            return;
        }
        if self.pinned.is_none() {
            self.pinned = Some(turn);
            return;
        }
        if HISTORY_RECENT_TURNS == 0 {
            return;
        }
        if self.recent.len() == HISTORY_RECENT_TURNS {
            self.recent.pop_front();
        }
        self.recent.push_back(turn);
    }

    /// The turns the next chunk's request carries, oldest (pinned) first.
    pub(crate) fn turns(&self) -> Vec<SpeechChunkTurn> {
        self.pinned.iter().chain(&self.recent).cloned().collect()
    }

    /// The history as a durable checkpoint holds it: each turn's codes as
    /// little-endian u16 (every code is below 4,096), base64.
    pub(crate) fn to_checkpoint(
        &self,
    ) -> Result<crate::batch_runtime::speech_progress::SpeechChunkHistoryCheckpoint, izwi_core::Error>
    {
        let turns = self
            .pinned
            .iter()
            .chain(&self.recent)
            .map(encode_chunk_turn)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(crate::batch_runtime::speech_progress::SpeechChunkHistoryCheckpoint { turns })
    }

    /// The history a durable checkpoint holds, checked.
    pub(crate) fn from_checkpoint(
        checkpoint: &crate::batch_runtime::speech_progress::SpeechChunkHistoryCheckpoint,
    ) -> Result<Self, izwi_core::Error> {
        if checkpoint.turns.len() > 1 + HISTORY_RECENT_TURNS {
            return Err(izwi_core::Error::InvalidInput(
                "Invalid speech chunk history: too many turns".into(),
            ));
        }
        let mut history = Self::default();
        for turn in &checkpoint.turns {
            history.push(decode_chunk_turn(turn)?);
        }
        Ok(history)
    }
}

/// A chunk's text and codes as a checkpoint (or a take's record) stores
/// them: the codes `[codebooks][frames]` flattened codebook by codebook as
/// little-endian u16 (every code is below 4,096), base64.
pub(crate) fn encode_chunk_turn(
    turn: &SpeechChunkTurn,
) -> Result<SpeechChunkTurnCheckpoint, izwi_core::Error> {
    use base64::Engine;
    let mut bytes = Vec::with_capacity(turn.codes.frames() * turn.codes.codebooks.len() * 2);
    for code in turn.codes.codebooks.iter().flatten() {
        let code = u16::try_from(*code).map_err(|_| {
            izwi_core::Error::InferenceError(format!(
                "Speech chunk code {code} does not fit its checkpoint"
            ))
        })?;
        bytes.extend_from_slice(&code.to_le_bytes());
    }
    Ok(SpeechChunkTurnCheckpoint {
        text: turn.text.clone(),
        codebooks: turn.codes.codebooks.len(),
        codes: base64::engine::general_purpose::STANDARD.encode(bytes),
    })
}

/// A stored chunk's text and codes, checked: whole frames, at least one.
pub(crate) fn decode_chunk_turn(
    turn: &SpeechChunkTurnCheckpoint,
) -> Result<SpeechChunkTurn, izwi_core::Error> {
    use base64::Engine;
    let invalid = |reason: &str| {
        izwi_core::Error::InvalidInput(format!("Invalid speech chunk history: {reason}"))
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&turn.codes)
        .map_err(|_| invalid("codes are not base64"))?;
    if turn.codebooks == 0 || bytes.is_empty() || bytes.len() % (2 * turn.codebooks) != 0 {
        return Err(invalid("codes do not fill whole frames"));
    }
    let frames = bytes.len() / 2 / turn.codebooks;
    let codes = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u32::from(u16::from_le_bytes(*pair)))
        .collect::<Vec<_>>();
    Ok(SpeechChunkTurn {
        text: turn.text.clone(),
        codes: SpeechFrameCodes::new(codes.chunks_exact(frames).map(<[u32]>::to_vec).collect()),
    })
}

fn should_enable_chunking(variant: ModelVariant, requested_max_tokens: usize) -> bool {
    let Some(model_max_frames) = variant.tts_max_output_frames_hint() else {
        return false;
    };
    requested_max_tokens == 0 || requested_max_tokens >= model_max_frames
}

pub fn split_tts_text_for_long_form(
    variant: ModelVariant,
    requested_max_tokens: usize,
    text: &str,
) -> Vec<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    if variant.plans_own_speech_segments() {
        // Callers accepting work must validate SpeechTextPlan first. This legacy
        // infallible splitter is also used by Studio's text editor.
        return SpeechTextPlan::for_variant(variant, trimmed, requested_max_tokens)
            .map(|plan| plan.segment_texts(trimmed))
            .unwrap_or_else(|_| vec![trimmed.to_string()]);
    }
    if !should_enable_chunking(variant, requested_max_tokens) {
        return vec![trimmed.to_string()];
    }

    // Whole sentences, packed up to the character cap and what one request
    // of this family can speak (TPIPE-5).
    let chunks = pack_sentences(trimmed, segment_max_chars(variant), &|segment| {
        !crate::api::tts_policy::tts_text_exceeds_one_request(variant, segment)
    });

    if chunks.is_empty() {
        vec![trimmed.to_string()]
    } else if matches!(
        variant.family(),
        izwi_core::ModelFamily::Maya1Tts | izwi_core::ModelFamily::Maya2Tts
    ) {
        attach_lone_tags(variant, chunks)
    } else {
        chunks
    }
}

/// Requests for one speech job. An automatic budget (`max_tokens` 0)
/// splits only text one request cannot finish (TPIPE-2), and every request
/// gets the automatic budget of its own text; an explicit budget keeps the
/// caller's split rule.
pub fn expand_generation_requests_for_long_form(
    base_request: &GenerationRequest,
    variant: ModelVariant,
) -> Vec<GenerationRequest> {
    let requested = base_request.config.options.max_tokens;
    let automatic = requested == 0 && !variant.plans_own_speech_segments();
    let chunks = if automatic && !splits_into_segments(variant, &base_request.text) {
        vec![base_request.text.trim().to_string()]
    } else {
        split_tts_text_for_long_form(variant, requested, base_request.text.as_str())
    };
    let budget_for = |request: &mut GenerationRequest| {
        if variant.plans_own_speech_segments() || automatic {
            if let Some(frames) =
                crate::api::tts_policy::resolve_tts_output_frames(variant, &request.text, None)
            {
                request.config.options.max_tokens = frames;
            } else if variant.plans_own_speech_segments() {
                request.config.options.max_tokens = 4096;
            }
        }
    };

    if chunks.len() <= 1 {
        let mut request = base_request.clone();
        if variant.plans_own_speech_segments() && request.config.options.max_tokens != 0 {
            return vec![request];
        }
        budget_for(&mut request);
        return vec![request];
    }

    chunks
        .into_iter()
        .enumerate()
        .map(|(idx, text)| {
            let mut req = base_request.clone();
            req.id = format!("{}:{}", base_request.id, idx + 1);
            req.text = text;
            budget_for(&mut req);
            req
        })
        .collect()
}

/// Leave the budget automatic (`0`) when the client set none and the text is
/// longer than one automatic request can speak, so long form splits it
/// instead of the request being cut at its cap.
pub(crate) fn defer_automatic_budget_to_long_form(
    request: &mut GenerationRequest,
    variant: ModelVariant,
    explicit_frames: Option<usize>,
) {
    if explicit_frames.unwrap_or(0) == 0
        && !variant.plans_own_speech_segments()
        && splits_into_segments(variant, &request.text)
    {
        request.config.options.max_tokens = 0;
    }
}

pub async fn generate_long_form_tts(
    runtime: &Arc<RuntimeService>,
    variant: ModelVariant,
    mut request: GenerationRequest,
) -> Result<GenerationResult, izwi_core::Error> {
    request.model_variant.get_or_insert(variant);
    let retry_runaways = retries_runaway_segments(variant, &request);
    let planned_requests = expand_generation_requests_for_long_form(&request, variant);
    if planned_requests.len() == 1 {
        return runtime
            .generate(planned_requests.into_iter().next().unwrap())
            .await;
    }

    let started = std::time::Instant::now();
    let split_request_count = planned_requests.len();
    let outputs = generate_segments(
        planned_requests,
        long_form_segment_concurrency(runtime),
        retry_runaways,
        &request.id,
        |segment| runtime.generate(segment),
    )
    .await?;
    let mut constituent_diagnostics = Vec::with_capacity(split_request_count);
    let mut merged_samples: Vec<f32> = Vec::new();
    let mut sample_rate: Option<u32> = None;
    let mut total_tokens = 0usize;
    let mut total_time_ms = 0f32;
    let mut stop_reasons = Vec::with_capacity(split_request_count);
    let mut retried_segments = Vec::new();
    let mut budget_segments = Vec::new();

    for (index, (output, retried)) in outputs.into_iter().enumerate() {
        if retried {
            retried_segments.push(index + 1);
        }
        if ran_to_budget(&output) {
            tracing::warn!(
                request_id = %request.id,
                segment = index + 1,
                "Long-form segment ended at its budget"
            );
            budget_segments.push(index + 1);
        }
        stop_reasons.push(output.stop_reason);
        if let Some(existing_rate) = sample_rate {
            if existing_rate != output.sample_rate {
                return Err(izwi_core::Error::InferenceError(format!(
                    "Long-form TTS sample-rate mismatch: {existing_rate} vs {}",
                    output.sample_rate
                )));
            }
        } else {
            sample_rate = Some(output.sample_rate);
        }
        append_segment(&mut merged_samples, &output.samples, output.sample_rate);
        total_tokens = total_tokens.saturating_add(output.total_tokens);
        total_time_ms += output.total_time_ms;
        constituent_diagnostics.push(output.diagnostics);
    }

    let sample_rate = sample_rate.ok_or_else(|| {
        izwi_core::Error::InferenceError("Long-form TTS produced no chunks".to_string())
    })?;

    Ok(GenerationResult {
        request_id: request.id,
        samples: merged_samples,
        sample_rate,
        total_tokens,
        total_time_ms,
        diagnostics: Some(serde_json::json!({
            "timing_basis": "sum_of_engine_executions",
            "split_request_count": split_request_count,
            "request_wall_ms": started.elapsed().as_secs_f64() * 1000.0,
            "constituents": constituent_diagnostics,
            // 1-based segments sampled again, and those that still ended
            // at their budget (the reason for a `length` stop).
            "retried_segments": retried_segments,
            "budget_segments": budget_segments,
        })),
        stop_reason: TtsStopReason::combine(stop_reasons),
    })
}

/// Stream a request into `output` as its long-form segments, one after
/// another, so streamed text is split as it is for a final-only request
/// instead of being cut at one request's budget. Text one request can carry
/// streams as is.
pub(crate) async fn generate_long_form_stream(
    runtime: &Arc<RuntimeService>,
    variant: ModelVariant,
    mut request: GenerationRequest,
    output: tokio::sync::mpsc::Sender<izwi_core::AudioChunk>,
    fallback_sample_rate: u32,
) -> Result<(), izwi_core::Error> {
    request.model_variant.get_or_insert(variant);
    let mut segments = expand_generation_requests_for_long_form(&request, variant);
    if segments.len() == 1 {
        return runtime.generate_streaming(segments.remove(0), output).await;
    }
    let runtime = runtime.clone();
    stream_segments(
        segments,
        &request.id,
        fallback_sample_rate,
        output,
        move |segment, sender| {
            let runtime = runtime.clone();
            async move { runtime.generate_streaming(segment, sender).await }
        },
    )
    .await
}

/// Forward each segment's chunks as one utterance's: renumbered under
/// `request_id`, a later segment opening with the long-form pause and fading
/// in, and only the last segment's final chunk final, carrying the whole
/// request's totals and combined stop reason (an earlier segment's totals go
/// out as a delta).
async fn stream_segments<F, Fut>(
    segments: Vec<GenerationRequest>,
    request_id: &str,
    fallback_sample_rate: u32,
    output: tokio::sync::mpsc::Sender<izwi_core::AudioChunk>,
    generate: F,
) -> Result<(), izwi_core::Error>
where
    F: Fn(GenerationRequest, tokio::sync::mpsc::Sender<izwi_core::AudioChunk>) -> Fut,
    Fut: std::future::Future<Output = Result<(), izwi_core::Error>> + Send + 'static,
{
    let count = segments.len();
    let mut sequence = 0;
    let (mut tokens, mut time_ms) = (0usize, 0f32);
    let mut stop_reasons = Vec::with_capacity(count);
    for (index, segment) in segments.into_iter().enumerate() {
        let last = index + 1 == count;
        let (sender, mut receiver) = tokio::sync::mpsc::channel(32);
        let task = tokio::spawn(generate(segment, sender));
        let _abort = AbortSpeechTask(task.abort_handle());
        let mut pause_pending = index > 0;
        // Deltas this segment has sent so far.
        let (mut sent_tokens, mut sent_ms) = (0usize, 0f32);
        while let Some(mut chunk) = receiver.recv().await {
            if pause_pending && !chunk.samples.is_empty() {
                pause_pending = false;
                let rate = chunk.sample_rate_or(fallback_sample_rate).max(1);
                chunk.samples = pause_then_segment(&chunk.samples, rate);
            }
            match (chunk.is_final, chunk.stats.as_mut()) {
                (false, Some(stats)) => {
                    sent_tokens += stats.tokens_generated;
                    sent_ms += stats.generation_time_ms;
                }
                (true, Some(stats)) => {
                    stop_reasons.push(stats.stop_reason);
                    let (segment_tokens, segment_ms) =
                        (stats.tokens_generated, stats.generation_time_ms);
                    if last {
                        stats.tokens_generated += tokens;
                        stats.generation_time_ms += time_ms;
                        stats.stop_reason = TtsStopReason::combine(stop_reasons.drain(..));
                    } else {
                        stats.tokens_generated = segment_tokens.saturating_sub(sent_tokens);
                        stats.generation_time_ms = (segment_ms - sent_ms).max(0.0);
                        stats.stop_reason = None;
                    }
                    tokens += segment_tokens;
                    time_ms += segment_ms;
                }
                (true, None) => stop_reasons.push(None),
                (false, None) => {}
            }
            chunk.is_final &= last;
            chunk.request_id = request_id.to_string();
            chunk.sequence = sequence;
            sequence += 1;
            output.send(chunk).await.map_err(|_| {
                izwi_core::Error::InferenceError("Speech consumer disconnected".into())
            })?;
        }
        task.await.map_err(|error| {
            izwi_core::Error::InferenceError(format!("Speech segment task failed: {error}"))
        })??;
    }
    Ok(())
}

/// Long-form segments in flight at once: the rows the engine batches TTS at
/// (`RuntimeService::tts_row_width`), so segments synthesize side by side
/// instead of one after another (SPK-F12). More would only queue.
fn long_form_segment_concurrency(runtime: &RuntimeService) -> usize {
    runtime.tts_row_width()
}

/// Generate the segments with up to `concurrency` in flight, returning each
/// output (and whether it was sampled again) in segment order. A sampled
/// Qwen3-TTS segment that ran to its budget without ending speech babbled
/// (TPIPE-27); when `retry_runaways`, it is sampled once more and the second
/// draw kept if it ended.
async fn generate_segments<F, Fut>(
    segments: Vec<GenerationRequest>,
    concurrency: usize,
    retry_runaways: bool,
    request_id: &str,
    generate: F,
) -> Result<Vec<(GenerationResult, bool)>, izwi_core::Error>
where
    F: Fn(GenerationRequest) -> Fut,
    Fut: std::future::Future<Output = Result<GenerationResult, izwi_core::Error>>,
{
    use futures::{StreamExt, TryStreamExt};
    let generate = &generate;
    futures::stream::iter(
        segments
            .into_iter()
            .enumerate()
            .map(|(index, segment)| async move {
                let retry = retry_runaways.then(|| segment.clone());
                let output = generate(segment).await?;
                let Some(retry) = retry.filter(|_| ran_to_budget(&output)) else {
                    return Ok((output, false));
                };
                tracing::warn!(
                    request_id = %request_id,
                    segment = index + 1,
                    "Long-form segment ran to its budget without end of speech; sampling it again"
                );
                let again = generate(retry).await?;
                Ok((if ran_to_budget(&again) { output } else { again }, true))
            }),
    )
    .buffered(concurrency.max(1))
    .try_collect()
    .await
}

/// Whether long form samples a Qwen3-TTS segment again when it runs to its
/// automatic budget: only for unseeded sampling, where a second draw differs
/// (greedy or a fixed seed would repeat the runaway).
fn retries_runaway_segments(variant: ModelVariant, request: &GenerationRequest) -> bool {
    let options = &request.config.options;
    variant.family() == izwi_core::ModelFamily::Qwen3Tts
        && options.max_tokens == 0
        && options.speech.seed.is_none()
        && options
            .speech
            .temperature
            .is_none_or(|temperature| temperature > 0.0)
        && options.speech.top_k != Some(1)
}

fn ran_to_budget(output: &GenerationResult) -> bool {
    output.stop_reason == Some(TtsStopReason::OutputBudget)
}

/// One live inference segment per job. Consumers receive one ordered stream and
/// one terminal total; neither PCM nor reference clones accumulate with duration.
pub(crate) async fn generate_speech_plan_stream(
    state: &crate::state::AppState,
    variant: ModelVariant,
    base: GenerationRequest,
    output: tokio::sync::mpsc::Sender<izwi_core::AudioChunk>,
    workload: izwi_core::WorkloadClass,
) -> Result<(), izwi_core::Error> {
    generate_speech_plan_stream_with_progress(state, variant, base, output, workload, None)
        .await
        .map(|_| ())
}

/// Speaks `base.text` by `variant`'s plan, sending the audio to `output`.
///
/// A chunk session (Fish Audio Drama 3) starts its history from
/// `base.chunk_history`, the chunks of the takes it continues (oldest
/// first), unless a recovered checkpoint already holds it, and returns the
/// chunks its own take keeps ([`SpeechTake`]); any other plan returns none.
pub(crate) async fn generate_speech_plan_stream_with_progress(
    state: &crate::state::AppState,
    variant: ModelVariant,
    mut base: GenerationRequest,
    output: tokio::sync::mpsc::Sender<izwi_core::AudioChunk>,
    workload: izwi_core::WorkloadClass,
    mut progress: Option<&mut crate::api::speech_history::DurableSpeechProgress>,
) -> Result<Option<SpeechTake>, izwi_core::Error> {
    use izwi_core::{AudioChunk, ChunkStats};
    let progress_error = |error: anyhow::Error| izwi_core::Error::InferenceError(error.to_string());
    let expected_fingerprint = state
        .runtime
        .fish_artifact_fingerprint(variant)
        .await
        .ok_or_else(|| {
            izwi_core::Error::ModelLoadError("Fish model fingerprint unavailable".into())
        })?;
    if progress
        .as_deref()
        .and_then(|p| p.artifact_fingerprint())
        .is_some_and(|expected| expected != expected_fingerprint)
    {
        return Err(izwi_core::Error::InvalidInput(
            "Fish model changed during speech recovery".into(),
        ));
    }
    let plan = SpeechTextPlan::for_variant(variant, &base.text, base.config.options.max_tokens)?;
    let mut remaining = plan.max_output_frames.unwrap_or(usize::MAX);
    let mut sequence = progress.as_deref().map_or(0, |p| p.next_sequence());
    let mut segment = progress.as_deref().map_or(0, |p| p.completed_segments());
    let mut tokens = progress.as_deref().map_or(0, |p| p.completed_tokens());
    // A resumed plan cannot see earlier segments' stop reasons.
    let mut stop_reasons = vec![if tokens > 0 {
        None
    } else {
        Some(TtsStopReason::EndOfSpeech)
    }];
    let mut execution_ms = progress
        .as_deref()
        .map_or(0.0, |p| p.completed_execution_ms());
    let mut duration = progress
        .as_deref()
        .map_or(0.0, |p| p.completed_duration_secs());
    remaining = remaining.saturating_sub(tokens);
    let mut rate = None;
    let completed = progress.as_deref().map_or(0, |p| p.completed_text_bytes());
    if completed > base.text.len() || !base.text.is_char_boundary(completed) {
        return Err(izwi_core::Error::InvalidInput(
            "Invalid speech recovery text offset".into(),
        ));
    }
    // A previously completed adaptive split can end inside an original range.
    // Resume only its suffix, never replay the already committed prefix.
    let mut ranges = plan.queue(&base.text, completed)?;
    // A chunk session's earlier chunks, as a recovered plan last recorded
    // them, else those of the takes it continues, pushed as if they were
    // its own earlier chunks: the first stays pinned, the latest follows.
    let previous_turns = std::mem::take(&mut base.chunk_history);
    let mut history = if plan.is_chunk_session() {
        Some(match progress.as_deref().and_then(|p| p.chunk_history()) {
            Some(checkpoint) => SpeechChunkHistory::from_checkpoint(checkpoint)?,
            None => {
                let mut history = SpeechChunkHistory::default();
                for turn in previous_turns {
                    history.push(turn);
                }
                history
            }
        })
    } else if previous_turns.is_empty() {
        None
    } else {
        return Err(izwi_core::Error::InvalidInput(format!(
            "{variant} does not continue earlier takes"
        )));
    };
    // The chunks this take keeps for its record.
    let mut take = if plan.is_chunk_session() {
        Some(match progress.as_deref().and_then(|p| p.take()) {
            Some(stored) => SpeechTake::from_stored(stored)?,
            None => SpeechTake::default(),
        })
    } else {
        None
    };
    while let Some(planned) = ranges.pop_front() {
        let range = planned.range.clone();
        if let Some(progress) = progress.as_deref_mut() {
            progress
                .begin_segment(segment, range.clone(), segment + 1 + ranges.len())
                .await
                .map_err(progress_error)?;
        }
        if output.is_closed() {
            return Err(izwi_core::Error::InferenceError(
                "Speech consumer disconnected".into(),
            ));
        }
        if remaining == 0 {
            return Err(izwi_core::Error::InvalidInput(
                "Speech generation incomplete: output_limit".into(),
            ));
        }
        if state
            .runtime
            .fish_artifact_fingerprint(variant)
            .await
            .as_deref()
            != Some(expected_fingerprint.as_str())
        {
            return Err(izwi_core::Error::InvalidInput(
                "Fish model changed during speech generation".into(),
            ));
        }
        let mut request = base.clone();
        request.id = format!("{}:{}", base.id, segment);
        request.text = planned.text.clone();
        request.config.streaming = true;
        request.config.options.max_tokens =
            plan.segment_frames(variant, &request.text).min(remaining);
        if let Some(history) = &history {
            // Each chunk's prompt holds the earlier chunks with their codes,
            // and chunk i samples from seed + i (the family's session).
            request.chunk_history = history.turns();
            if let Some(seed) = base.config.options.speech.seed {
                request.config.options.speech.seed = Some(seed.wrapping_add(segment as u64));
            }
        }
        let permit = state.acquire_workload_permit(workload).await;
        let tenant = base.runtime_context.tenant_key;
        request.runtime_context = permit.runtime_context();
        request.runtime_context.tenant_key = tenant;
        let (sender, mut receiver) = tokio::sync::mpsc::channel::<AudioChunk>(2);
        let runtime = state.runtime.clone();
        let task = tokio::spawn(async move { runtime.generate_streaming(request, sender).await });
        let _abort = AbortSpeechTask(task.abort_handle());
        // A progress deadline, not a wall clock: a segment fails only when
        // it publishes nothing for this long (TPIPE-6).
        let stall_timeout = crate::env_config::tuning().tts_segment_timeout;
        let mut deadline = tokio::time::Instant::now() + stall_timeout;
        let mut segment_duration = 0.0;
        let mut emitted = false;
        let mut segment_tokens = 0;
        let mut segment_ms = 0.0;
        let mut segment_stop = None;
        let mut segment_codes = None;
        loop {
            // Buffered durable audio is published by its deadline even when
            // the model is between chunks.
            let flush_at = progress.as_deref().and_then(|p| p.flush_deadline());
            let flush_due = async {
                match flush_at {
                    Some(at) => tokio::time::sleep_until(at).await,
                    None => std::future::pending().await,
                }
            };
            let mut chunk = tokio::select! {
                chunk = receiver.recv() => match chunk {
                    Some(chunk) => chunk,
                    None => break,
                },
                _ = tokio::time::sleep_until(deadline) => return Err(izwi_core::Error::Timeout("Speech segment made no progress".into())),
                _ = output.closed() => return Err(izwi_core::Error::InferenceError("Speech consumer disconnected".into())),
                _ = flush_due => {
                    if let Some(progress) = progress.as_deref_mut() {
                        if let Some(published) = progress.flush().await.map_err(progress_error)? {
                            send_published(&output, &mut sequence, published).await?;
                        }
                    }
                    continue;
                }
            };
            deadline = tokio::time::Instant::now() + stall_timeout;
            // The codes a chunk generated stay with the plan; no listener
            // sees them.
            if let Some(codes) = chunk.generated_codes.take() {
                segment_codes = Some(codes);
            }
            if let Some(stats) = &chunk.stats {
                if chunk.is_final {
                    segment_tokens = stats.tokens_generated;
                    segment_ms = stats.generation_time_ms;
                    segment_stop = stats.stop_reason;
                } else {
                    segment_tokens += stats.tokens_generated;
                    segment_ms += stats.generation_time_ms;
                }
            }
            if chunk.samples.is_empty() {
                continue;
            }
            let actual = chunk.sample_rate_or(state.runtime.sample_rate().await);
            if rate.is_some_and(|expected| expected != actual) {
                drop(receiver);
                task.abort();
                let _ = task.await;
                return Err(izwi_core::Error::InferenceError(
                    "Speech segment sample rate changed".into(),
                ));
            }
            rate = Some(actual);
            segment_duration += chunk.samples.len() as f32 / actual as f32;
            chunk.request_id = base.id.clone();
            chunk.is_final = false;
            chunk.stats = None;
            emitted = true;
            // Durable audio reaches listeners only once it is published.
            let ready = match progress.as_deref_mut() {
                Some(progress) => progress
                    .publish_chunk(chunk)
                    .await
                    .map_err(progress_error)?,
                None => vec![chunk],
            };
            for chunk in ready {
                if let Err(error) = send_published(&output, &mut sequence, chunk).await {
                    drop(receiver);
                    task.abort();
                    let _ = task.await;
                    return Err(error);
                }
            }
        }
        let result = task
            .await
            .map_err(|err| izwi_core::Error::InferenceError(err.to_string()))?;
        drop(permit);
        if let Err(error) = result {
            // Exact tokenized reference+text fit is checked inside admitted model
            // preparation. Replan only before this segment has published audio.
            if !emitted && error.kind() == izwi_core::ErrorKind::ContextLengthExceeded {
                if let Some(halves) = plan.halve(&base.text, &planned) {
                    for half in halves.into_iter().rev() {
                        ranges.push_front(half);
                    }
                    continue;
                }
            }
            return Err(error);
        }
        if !emitted || segment_tokens == 0 {
            return Err(izwi_core::Error::InferenceError(
                "Speech segment completed without audio or accounting".into(),
            ));
        }
        if let Some(history) = history.as_mut() {
            let codes = segment_codes.ok_or_else(|| {
                izwi_core::Error::InferenceError(
                    "Speech chunk completed without its generated codes".into(),
                )
            })?;
            let turn = SpeechChunkTurn {
                text: planned.text,
                codes: Arc::unwrap_or_clone(codes),
            };
            if let Some(take) = take.as_mut() {
                take.push(turn.clone());
                if let Some(progress) = progress.as_deref_mut() {
                    progress.record_take(take.to_stored()?);
                }
            }
            history.push(turn);
            if let Some(progress) = progress.as_deref_mut() {
                progress.record_chunk_history(history.to_checkpoint()?);
            }
        }
        if let Some(progress) = progress.as_deref_mut() {
            if let Some(published) = progress.flush().await.map_err(progress_error)? {
                send_published(&output, &mut sequence, published).await?;
            }
            progress
                .complete_segment(range.end, segment_tokens, segment_ms, segment_duration)
                .await
                .map_err(progress_error)?;
        }
        duration += segment_duration;
        stop_reasons.push(segment_stop);
        remaining = remaining.saturating_sub(segment_tokens);
        tokens += segment_tokens;
        execution_ms += segment_ms;
        segment += 1;
    }
    let mut terminal =
        AudioChunk::final_chunk(base.id, sequence, Vec::new()).with_sample_rate(rate.unwrap_or(0));
    terminal.stats = Some(ChunkStats {
        tokens_generated: tokens,
        generation_time_ms: execution_ms,
        rtf: if duration > 0.0 {
            execution_ms / 1000.0 / duration
        } else {
            0.0
        },
        stop_reason: TtsStopReason::combine(stop_reasons),
    });
    output
        .send(terminal)
        .await
        .map_err(|_| izwi_core::Error::InferenceError("Speech consumer disconnected".into()))?;
    Ok(take)
}

/// Send one published chunk under the next output sequence.
async fn send_published(
    output: &tokio::sync::mpsc::Sender<izwi_core::AudioChunk>,
    sequence: &mut usize,
    mut chunk: izwi_core::AudioChunk,
) -> Result<(), izwi_core::Error> {
    chunk.sequence = *sequence;
    output
        .send(chunk)
        .await
        .map_err(|_| izwi_core::Error::InferenceError("Speech consumer disconnected".into()))?;
    *sequence += 1;
    Ok(())
}

/// Dropping a timed-out/cancelled job also cancels its current inference future.
struct AbortSpeechTask(tokio::task::AbortHandle);
impl Drop for AbortSpeechTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Streamed segments reach the consumer as one utterance: numbered in
    /// order under the request's id, the second opening with the pause, only
    /// the last chunk final, and the totals and stop reason of both.
    #[tokio::test]
    async fn streamed_segments_join_into_one_utterance() {
        use izwi_core::{AudioChunk, ChunkStats};
        let stats = |tokens, ms, stop_reason| ChunkStats {
            generation_time_ms: ms,
            tokens_generated: tokens,
            rtf: 0.0,
            stop_reason,
        };
        let segments = vec![
            GenerationRequest::new("first"),
            GenerationRequest::new("second"),
        ];
        let (output, mut received) = tokio::sync::mpsc::channel(16);
        stream_segments(segments, "request", 100, output, move |segment, sender| {
            let first = segment.text == "first";
            async move {
                let mut interim = AudioChunk::new(segment.id.clone(), 0, vec![1.0; 4]);
                interim.stats = Some(stats(3, 30.0, None));
                sender.send(interim).await.unwrap();
                let mut last = AudioChunk::final_chunk(segment.id.clone(), 1, vec![1.0; 2]);
                last.stats = Some(if first {
                    stats(5, 50.0, Some(TtsStopReason::OutputBudget))
                } else {
                    stats(4, 40.0, Some(TtsStopReason::EndOfSpeech))
                });
                sender.send(last).await.unwrap();
                Ok(())
            }
        })
        .await
        .unwrap();

        let mut chunks = Vec::new();
        while let Some(chunk) = received.recv().await {
            chunks.push(chunk);
        }
        assert_eq!(
            chunks.iter().map(|c| c.sequence).collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        assert!(chunks.iter().all(|c| c.request_id == "request"));
        assert_eq!(
            chunks.iter().map(|c| c.is_final).collect::<Vec<_>>(),
            [false, false, false, true]
        );
        // The first segment's rest goes out as a delta: 5 − 3 tokens.
        let delta = chunks[1].stats.as_ref().unwrap();
        assert_eq!((delta.tokens_generated, delta.stop_reason), (2, None));
        let pause = segment_pause_samples(100);
        assert_eq!(chunks[2].samples.len(), pause + 4);
        assert!(chunks[2].samples[..pause]
            .iter()
            .all(|&sample| sample == 0.0));
        let total = chunks[3].stats.as_ref().unwrap();
        assert_eq!(total.tokens_generated, 9);
        assert_eq!(total.generation_time_ms, 90.0);
        assert_eq!(total.stop_reason, Some(TtsStopReason::OutputBudget));
    }

    /// SPK-F12: segments run `concurrency` at a time, and their outputs come
    /// back in segment order however long each takes.
    #[tokio::test]
    async fn long_form_segments_run_concurrently_in_order() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let in_flight = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let segments: Vec<GenerationRequest> = (0..6)
            .map(|index| GenerationRequest::new(format!("segment {index}")))
            .collect();
        let outputs = generate_segments(segments, 3, false, "request", |segment| {
            let (in_flight, peak) = (&in_flight, &peak);
            async move {
                let now = in_flight.fetch_add(1, Ordering::AcqRel) + 1;
                peak.fetch_max(now, Ordering::AcqRel);
                let index: u64 = segment.text["segment ".len()..].parse().unwrap();
                // Later segments finish first.
                tokio::time::sleep(std::time::Duration::from_millis(60 - 10 * index)).await;
                in_flight.fetch_sub(1, Ordering::AcqRel);
                Ok(GenerationResult {
                    request_id: segment.text.clone(),
                    samples: vec![index as f32],
                    sample_rate: 24_000,
                    total_tokens: 1,
                    total_time_ms: 1.0,
                    diagnostics: None,
                    stop_reason: Some(TtsStopReason::EndOfSpeech),
                })
            }
        })
        .await
        .expect("segments");
        let order: Vec<f32> = outputs
            .iter()
            .map(|(output, _)| output.samples[0])
            .collect();
        assert_eq!(order, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(peak.load(Ordering::Acquire), 3, "three segments in flight");
    }

    /// A segment that ran to its budget is sampled again, and the retry is
    /// kept only if it ended speech.
    #[tokio::test]
    async fn runaway_segments_are_sampled_again() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let outputs = generate_segments(
            vec![GenerationRequest::new("a"), GenerationRequest::new("b")],
            2,
            true,
            "request",
            |segment| {
                let call = calls.fetch_add(1, Ordering::AcqRel);
                async move {
                    // "a" runs away on its first draw only.
                    let runaway = segment.text == "a" && call == 0;
                    Ok(GenerationResult {
                        request_id: segment.text,
                        samples: vec![call as f32],
                        sample_rate: 24_000,
                        total_tokens: 1,
                        total_time_ms: 1.0,
                        diagnostics: None,
                        stop_reason: Some(if runaway {
                            TtsStopReason::OutputBudget
                        } else {
                            TtsStopReason::EndOfSpeech
                        }),
                    })
                }
            },
        )
        .await
        .expect("segments");
        assert!(outputs[0].1, "the runaway segment was retried");
        assert!(!ran_to_budget(&outputs[0].0));
        assert!(!outputs[1].1);
    }

    #[test]
    fn fish_plan_preserves_utf8_text_and_short_fast_path() {
        for text in [
            "Hello. A short paragraph!".to_string(),
            "Dr. Smith paid $3.14. 中文🎵 no spaces 日本語\n".repeat(500),
            "字".repeat(4000),
        ] {
            let plan = SpeechTextPlan::fish(&text, 0).unwrap();
            let reconstructed: String = plan
                .segments
                .iter()
                .map(|range| &text[range.clone()])
                .collect();
            assert_eq!(reconstructed, text);
            assert!(plan
                .segments
                .iter()
                .all(|range| text[range.clone()].chars().count() <= 480));
            if text.len() < 480 {
                assert_eq!(plan.segments.len(), 1);
            }
        }
    }

    #[test]
    fn fish_plan_packs_short_sentences_and_prefers_sentence_boundaries() {
        let paragraph = "First sentence. Second sentence. Third sentence.";
        assert_eq!(
            SpeechTextPlan::fish(paragraph, 0).unwrap().segments.len(),
            1
        );
        let text = format!(
            "{} {}",
            "A reasonably sized sentence. ".repeat(14),
            "word ".repeat(60)
        );
        let plan = SpeechTextPlan::fish(&text, 0).unwrap();
        assert!(text[plan.segments[0].clone()].trim_end().ends_with('.'));
    }

    #[test]
    fn fish_plan_never_generates_blank_layout_segments() {
        let text = format!(
            "{}Hello.{}World.{}",
            " ".repeat(1000),
            "\n".repeat(1000),
            " ".repeat(1000)
        );
        let plan = SpeechTextPlan::fish(&text, 0).unwrap();
        let spoken: String = plan
            .segments
            .iter()
            .map(|range| &text[range.clone()])
            .collect();
        assert_eq!(
            spoken.split_whitespace().collect::<String>(),
            "Hello.World."
        );
        assert!(plan
            .segments
            .iter()
            .all(|range| !text[range.clone()].trim().is_empty()));
    }

    #[test]
    fn fish_long_text_splits_independently_of_output_budget() {
        let text = "This sentence must all be spoken. ".repeat(1000);
        let auto = SpeechTextPlan::fish(&text, 0).unwrap();
        let limited = SpeechTextPlan::fish(&text, 2584).unwrap();
        assert!(auto.segments.len() > 1);
        assert_eq!(auto.segments, limited.segments);
        assert_eq!(limited.max_output_frames, Some(2584));
    }

    /// TPIPE-2: an automatic budget never splits long form, so text longer
    /// than one request's 120 s cap was cut at the cap with a 200. Text one
    /// automatic request cannot finish now splits, each request budgeted for
    /// its own text; shorter text stays one request with its automatic
    /// budget; an explicit budget is kept.
    #[test]
    fn automatic_budget_splits_only_text_one_request_cannot_finish() {
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let cap =
            izwi_core::runtime_models::architectures::qwen3::tts::qwen3_tts_auto_max_output_frames(
            );
        let sentence = "The committee reviewed every proposal in detail before the vote. ";
        let long = sentence.repeat(30);
        assert!(crate::api::tts_policy::tts_text_exceeds_one_request(
            variant, &long
        ));
        let mut request = GenerationRequest::new(long.clone()).with_model_variant(variant);
        request.config.options.max_tokens =
            crate::api::tts_policy::resolve_tts_output_frames(variant, &long, None).unwrap();
        assert_eq!(request.config.options.max_tokens, cap);
        defer_automatic_budget_to_long_form(&mut request, variant, None);
        assert_eq!(request.config.options.max_tokens, 0);
        let planned = expand_generation_requests_for_long_form(&request, variant);
        assert!(planned.len() > 1);
        for chunk in &planned {
            let budget = chunk.config.options.max_tokens;
            assert!(budget > 0 && budget < cap, "chunk budget {budget}");
        }

        let short = sentence.repeat(3);
        let mut request = GenerationRequest::new(short.clone()).with_model_variant(variant);
        request.config.options.max_tokens =
            crate::api::tts_policy::resolve_tts_output_frames(variant, &short, None).unwrap();
        let automatic = request.config.options.max_tokens;
        defer_automatic_budget_to_long_form(&mut request, variant, None);
        assert_eq!(request.config.options.max_tokens, automatic);
        assert_eq!(
            expand_generation_requests_for_long_form(&request, variant).len(),
            1
        );

        // An explicit budget is the caller's: no deferral, no split.
        let mut request = GenerationRequest::new(long).with_model_variant(variant);
        request.config.options.max_tokens = 300;
        defer_automatic_budget_to_long_form(&mut request, variant, Some(300));
        let planned = expand_generation_requests_for_long_form(&request, variant);
        assert_eq!(planned.len(), 1);
        assert_eq!(planned[0].config.options.max_tokens, 300);
        // Families without automatic long form never split.
        assert!(!crate::api::tts_policy::tts_text_exceeds_one_request(
            ModelVariant::Kokoro82M,
            &sentence.repeat(30)
        ));
    }

    /// TPIPE-5: short sentences pack into one request instead of one request
    /// per sentence; text longer than one request splits between whole
    /// sentences, and every segment fits one request.
    #[test]
    fn qwen_auto_chunking_packs_whole_sentences_up_to_one_request() {
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let chunks = split_tts_text_for_long_form(
            variant,
            0,
            "Hello world. This is sentence two! Final line?",
        );
        assert_eq!(
            chunks,
            vec!["Hello world. This is sentence two! Final line?"]
        );
        let sentence = "Dr. Smith paid $3.50 at 10:30 for the long and winding trip home. ";
        let text = sentence.repeat(30);
        let chunks = split_tts_text_for_long_form(variant, 0, &text);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(
                chunk.starts_with("Dr. Smith") && chunk.ends_with("home."),
                "{chunk}"
            );
            assert!(!crate::api::tts_policy::tts_text_exceeds_one_request(
                variant, chunk
            ));
        }
    }

    /// TPIPE-27: the 250-word scoring paragraph used to split into 480-char
    /// segments only because it exceeded one request, and one 460-char
    /// segment ran to its 663-frame budget and babbled. Qwen3-TTS segments
    /// are now whole sentences packed up to 300 characters, so it splits
    /// into short segments, each with its own automatic budget, while the
    /// 45-word medium paragraph (EOS on every backend) stays one request.
    #[test]
    fn qwen_long_form_speaks_short_sentence_segments() {
        let _env = crate::test_support::env_lock();
        let variant = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let long = "The old lighthouse stood at the edge of the harbor for more than a century. \
            Every evening, the keeper climbed one hundred and twelve steps to light the lamp, \
            and every morning he climbed them again to put it out. Ships from distant ports \
            used its beam to find the narrow channel between the rocks. In winter, storms \
            pushed waves high enough to wash over the lower windows, and the keeper would sit \
            by the stove, listening to the wind and writing in his logbook. He recorded the \
            weather, the ships that passed, and the small events of each day. One entry \
            describes a fishing boat that lost its engine in a gale and drifted toward the \
            cliffs until the crew saw the light and rowed to safety. Another mentions a family \
            of seals that slept on the breakwater every spring. When the lighthouse was finally \
            automated, the logbooks were sent to the town library, where students still read \
            them today. They show how a single person, doing the same careful work every night, \
            kept hundreds of strangers safe. The building is now a small museum. Visitors can \
            climb the stairs, look out over the water, and imagine what it was like to watch \
            the sea alone for so many years. On clear nights, the automated lamp still turns, \
            sweeping its light across the dark water just as it always has.";
        let mut request = GenerationRequest::new(long).with_model_variant(variant);
        request.config.options.max_tokens =
            crate::api::tts_policy::resolve_tts_output_frames(variant, long, None).unwrap();
        defer_automatic_budget_to_long_form(&mut request, variant, None);
        assert_eq!(request.config.options.max_tokens, 0);
        let planned = expand_generation_requests_for_long_form(&request, variant);
        assert!(planned.len() >= 5, "{} segments", planned.len());
        for segment in &planned {
            assert!(
                segment.text.chars().count() <= QWEN_SEGMENT_MAX_CHARS,
                "{}",
                segment.text
            );
            assert!(segment.text.ends_with('.'), "{}", segment.text);
            let budget = segment.config.options.max_tokens;
            assert!(budget > 0 && budget < 663, "{budget}: {}", segment.text);
        }
        assert_eq!(
            planned
                .iter()
                .map(|segment| segment.text.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            long.split_whitespace().collect::<Vec<_>>().join(" ")
        );

        let medium = "Good morning. Today we are testing a speech synthesis system on a laptop. \
            The goal is to measure how fast it runs, how much memory it uses, and whether \
            every word in this paragraph comes out clearly, without skipping, repeating, \
            or adding anything at the end.";
        let mut request = GenerationRequest::new(medium).with_model_variant(variant);
        request.config.options.max_tokens =
            crate::api::tts_policy::resolve_tts_output_frames(variant, medium, None).unwrap();
        defer_automatic_budget_to_long_form(&mut request, variant, None);
        assert_eq!(
            expand_generation_requests_for_long_form(&request, variant).len(),
            1
        );
    }

    /// A segment that runs to its budget is sampled once more only when a
    /// second draw can differ: unseeded, non-greedy Qwen3-TTS sampling.
    #[test]
    fn only_unseeded_sampled_qwen_segments_are_retried() {
        let qwen = ModelVariant::Qwen3Tts12Hz06BCustomVoice;
        let request = |configure: &dyn Fn(&mut GenerationRequest)| {
            let mut request = GenerationRequest::new("text").with_model_variant(qwen);
            configure(&mut request);
            request
        };
        assert!(retries_runaway_segments(qwen, &request(&|_| {})));
        assert!(!retries_runaway_segments(
            qwen,
            &request(&|r| r.config.options.speech.seed = Some(7))
        ));
        assert!(!retries_runaway_segments(
            qwen,
            &request(&|r| r.config.options.speech.temperature = Some(0.0))
        ));
        assert!(!retries_runaway_segments(
            qwen,
            &request(&|r| r.config.options.speech.top_k = Some(1))
        ));
        assert!(!retries_runaway_segments(
            qwen,
            &request(&|r| r.config.options.max_tokens = 300)
        ));
        assert!(!retries_runaway_segments(
            ModelVariant::VibeVoice15BTts,
            &request(&|_| {})
        ));
    }

    /// Segments join with a faded tail, a 120 ms pause and a faded head, not
    /// back to back.
    #[test]
    fn segments_join_with_a_faded_pause() {
        let rate = 24_000;
        let mut joined = Vec::new();
        append_segment(&mut joined, &[1.0; 480], rate);
        assert_eq!(joined, vec![1.0; 480]);
        append_segment(&mut joined, &[1.0; 480], rate);
        let pause = 2_880;
        assert_eq!(joined.len(), 480 + pause + 480);
        // The first segment fades out over its last 10 ms (240 samples).
        assert_eq!(joined[239], 1.0);
        assert!(joined[479] > 0.0 && joined[479] < 0.01);
        assert!(joined[480..480 + pause].iter().all(|sample| *sample == 0.0));
        // The second fades in over its first 10 ms.
        assert!(joined[480 + pause] > 0.0 && joined[480 + pause] < 0.01);
        assert_eq!(*joined.last().unwrap(), 1.0);
    }

    #[test]
    fn explicit_small_token_budget_disables_long_form_split() {
        let chunks = split_tts_text_for_long_form(
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            256,
            "Sentence one. Sentence two.",
        );
        assert_eq!(chunks.len(), 1);
    }

    /// Maya1 segments are whole sentences of about 11 s at most (it was
    /// trained on 1-14 s clips), each with its own budget, and a tag that
    /// would stand alone joins the sentence before it.
    #[test]
    fn maya1_splits_into_short_sentences_and_keeps_tags_with_words() {
        let variant = ModelVariant::Maya13B;
        let sentence = "The committee reviewed every proposal in detail before the vote. ";
        let short = GenerationRequest::new(sentence.to_string()).with_model_variant(variant);
        assert_eq!(
            expand_generation_requests_for_long_form(&short, variant).len(),
            1
        );

        let long = GenerationRequest::new(sentence.repeat(6)).with_model_variant(variant);
        let planned = expand_generation_requests_for_long_form(&long, variant);
        assert!(planned.len() >= 3, "{}", planned.len());
        for chunk in &planned {
            assert!(chunk.text.chars().count() <= 160, "{}", chunk.text);
            let budget = chunk.config.options.max_tokens;
            assert!(budget > 0 && budget % 7 == 0, "segment budget {budget}");
        }

        let chunks = attach_lone_tags(
            variant,
            vec![
                "<sigh>".to_string(),
                "Well, that went badly.".to_string(),
                "<laugh>".to_string(),
                "Never again.".to_string(),
            ],
        );
        assert_eq!(
            chunks,
            vec![
                "<sigh> Well, that went badly. <laugh>".to_string(),
                "Never again.".to_string()
            ]
        );
        assert_eq!(
            attach_lone_tags(variant, vec!["<gasp>".to_string()]),
            vec!["<gasp>"]
        );
    }

    /// Maya 2 Native speaks long text in pieces of about 20 s (280
    /// characters): Hindi paragraphs split at dandas, and a lone `[laughs]`
    /// stays with its sentence.
    #[test]
    fn maya2_splits_at_dandas_and_keeps_square_tags_with_words() {
        let variant = ModelVariant::Maya2Native25B;
        let sentence = "आपका ऑर्डर कल शाम तक पहुँच जाएगा और डिलीवरी एजेंट आपको फोन करेगा। ";
        let long = GenerationRequest::new(sentence.repeat(8)).with_model_variant(variant);
        let planned = expand_generation_requests_for_long_form(&long, variant);
        assert!(planned.len() >= 2, "{}", planned.len());
        for chunk in &planned {
            assert!(chunk.text.chars().count() <= 280, "{}", chunk.text);
            assert!(chunk.text.trim_end().ends_with('।'), "{}", chunk.text);
        }
        let short = GenerationRequest::new(sentence).with_model_variant(variant);
        assert_eq!(
            expand_generation_requests_for_long_form(&short, variant).len(),
            1
        );

        let chunks = attach_lone_tags(
            variant,
            vec![
                "That went badly.".to_string(),
                "[laughs]".to_string(),
                "Never again.".to_string(),
            ],
        );
        assert_eq!(
            chunks,
            vec![
                "That went badly. [laughs]".to_string(),
                "Never again.".to_string()
            ]
        );
        // Brackets that are not lowercase tags are text.
        assert_eq!(inline_tags(variant, "[Note] a [b c] [x1]"), vec!["[b c]"]);
    }

    /// Chatterbox speaks at most 1,000 speech tokens (40 s) per request, so
    /// text whose automatic budget reaches that splits between sentences
    /// and each segment gets its own budget below the cap.
    #[test]
    fn chatterbox_splits_text_one_request_cannot_finish() {
        let variant = ModelVariant::Chatterbox500M;
        let sentence = "The committee reviewed every proposal in detail before the vote. ";
        let short = sentence.repeat(3);
        assert!(!crate::api::tts_policy::tts_text_exceeds_one_request(
            variant, &short
        ));
        let request = GenerationRequest::new(short).with_model_variant(variant);
        assert_eq!(
            expand_generation_requests_for_long_form(&request, variant).len(),
            1
        );

        let long = sentence.repeat(20);
        assert!(crate::api::tts_policy::tts_text_exceeds_one_request(
            variant, &long
        ));
        let request = GenerationRequest::new(long).with_model_variant(variant);
        let planned = expand_generation_requests_for_long_form(&request, variant);
        assert!(planned.len() > 1);
        for chunk in &planned {
            assert!(chunk.text.starts_with("The committee") && chunk.text.ends_with("vote."));
            let budget = chunk.config.options.max_tokens;
            assert!(
                budget > 0 && budget < ModelVariant::CHATTERBOX_MAX_OUTPUT_FRAMES,
                "segment budget {budget}"
            );
        }
    }

    /// Drama 3's plan is its chunk session: the family's speaker-aware
    /// chunker (not S2's segment plan), one segment per chunk tiling the
    /// input, and its own per-chunk budget. S2's plan serializes as before.
    #[test]
    fn fish_drama3_plans_a_chunk_session_with_its_own_chunker() {
        use izwi_core::runtime_models::architectures::fish_drama3::chunking::plan_chunks;
        let drama3 = ModelVariant::FishAudioDrama3;
        let text = format!(
            "  {}<|speaker:1|> [sighs] {}<|speaker:2|> Short.  ",
            "The first speaker talks for a while here. ".repeat(9),
            "Then the second one answers at some length. ".repeat(8)
        );
        let plan = SpeechTextPlan::for_variant(drama3, &text, 0).unwrap();
        assert!(plan.is_chunk_session());
        assert_eq!(plan.version, CHUNK_SESSION_PLAN_VERSION);
        assert_eq!(plan.chunks, plan_chunks(&text, false));
        assert!(plan.chunks.len() >= 3, "{}", plan.chunks.len());
        assert_ne!(
            plan.chunks,
            SpeechTextPlan::fish(&text, 0).unwrap().segment_texts(&text)
        );
        // Every chunk of tagged text carries its speaker's tag.
        assert!(plan
            .chunks
            .iter()
            .flat_map(|chunk| chunk.split('\n'))
            .all(|line| line.starts_with("<|speaker:")));
        // Segments tile the input, from its first byte to its last.
        assert_eq!(plan.segments.first().unwrap().start, 0);
        assert_eq!(plan.segments.last().unwrap().end, text.len());
        assert!(plan
            .segments
            .windows(2)
            .all(|pair| pair[0].end == pair[1].start && pair[0].start < pair[0].end));
        assert_eq!(plan.segment_texts(&text), plan.chunks);
        // Each chunk's budget: its own automatic one, or the client's per
        // chunk (at most 4,096).
        assert_eq!(plan.max_output_frames, None);
        assert_eq!(plan.chunk_frames, None);
        for chunk in &plan.chunks {
            assert_eq!(
                plan.segment_frames(drama3, chunk),
                crate::api::tts_policy::resolve_tts_output_frames(drama3, chunk, None).unwrap()
            );
        }
        let explicit = SpeechTextPlan::for_variant(drama3, &text, 9_000).unwrap();
        assert_eq!(explicit.chunk_frames, Some(4_096));
        assert_eq!(explicit.segment_frames(drama3, &plan.chunks[0]), 4_096);
        assert!(matches!(
            SpeechTextPlan::for_variant(drama3, " \n ", 0),
            Err(izwi_core::Error::InvalidInput(_))
        ));
        // One short text is one chunk; the splitter Studio uses agrees.
        let short = SpeechTextPlan::for_variant(drama3, "Hello there.", 0).unwrap();
        assert_eq!(short.chunks, ["Hello there."]);
        assert_eq!(short.segments.len(), 1);
        assert_eq!(short.segments[0], 0..12);
        assert_eq!(
            split_tts_text_for_long_form(drama3, 0, &text),
            plan_chunks(text.trim(), false)
        );
        // S2's plan is unchanged and its identity JSON has no chunk fields.
        let fish = SpeechTextPlan::for_variant(ModelVariant::FishAudioS2Pro, &text, 0).unwrap();
        assert_eq!(fish, SpeechTextPlan::fish(&text, 0).unwrap());
        assert!(!fish.is_chunk_session());
        let json = serde_json::to_value(&fish).unwrap();
        let mut keys = json
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        keys.sort();
        assert_eq!(
            keys,
            ["max_output_frames", "segments", "text_sha256", "version"]
        );
        let json = serde_json::to_value(&explicit).unwrap();
        assert_eq!(json["chunk_frames"], 4_096);
        assert_eq!(
            serde_json::from_value::<SpeechTextPlan>(json).unwrap(),
            explicit
        );
    }

    /// A chunk too long for the model is cut again by the family's chunker
    /// (tags kept), its parts tiling its input range; recovery finds the
    /// same parts again from the completed offset alone.
    #[test]
    fn halved_chunks_keep_their_tags_and_resume_where_they_stopped() {
        let drama3 = ModelVariant::FishAudioDrama3;
        let text = format!(
            "<|speaker:3|> {}<|speaker:4|> {}",
            "One clause, then another clause, and more words follow here. ".repeat(4),
            "A reply that is long enough to stand alone as its chunk. ".repeat(5)
        );
        let plan = SpeechTextPlan::for_variant(drama3, &text, 0).unwrap();
        let all = plan.queue(&text, 0).unwrap();
        assert_eq!(all.len(), plan.chunks.len());
        let first = all[0].clone();
        let halves = plan.halve(&text, &first).unwrap();
        assert!(halves.len() >= 2);
        assert_eq!(halves[0].range.start, first.range.start);
        assert_eq!(halves.last().unwrap().range.end, first.range.end);
        assert!(halves
            .windows(2)
            .all(|pair| pair[0].range.end == pair[1].range.start));
        assert!(halves.iter().all(
            |half| half.text.starts_with("<|speaker:3|>") && half.text.len() < first.text.len()
        ));
        // Recovery after the first half: the rest of the halved chunk, then
        // every later chunk.
        let resumed = plan.queue(&text, halves[0].range.end).unwrap();
        assert_eq!(resumed[0], halves[1]);
        assert_eq!(resumed.len(), halves.len() - 1 + all.len() - 1);
        assert_eq!(resumed.back(), all.back());
        // At a chunk boundary: the chunks after it.
        let at_boundary = plan.queue(&text, all[0].range.end).unwrap();
        assert_eq!(
            at_boundary,
            all.iter().skip(1).cloned().collect::<VecDeque<_>>()
        );
        assert!(plan.queue(&text, text.len()).unwrap().is_empty());
        // An offset inside a chunk no halving can produce is refused.
        let whole = "[a long tag stays whole]";
        let tagged = SpeechTextPlan::for_variant(drama3, whole, 0).unwrap();
        let error = tagged.queue(whole, 5).unwrap_err();
        assert!(
            error.to_string().contains("recovery text offset"),
            "{error}"
        );
        // A protected span cannot be cut.
        let tag = PlannedSegment {
            range: 0..20,
            text: "[a long tag stays whole]".into(),
        };
        assert!(plan.halve("[a long tag stays w]", &tag).is_none());
        // S2 halves at its middle character, as before.
        let fish = SpeechTextPlan::fish("abcdef", 0).unwrap();
        let segment = fish.queue("abcdef", 0).unwrap().pop_front().unwrap();
        let halves = fish.halve("abcdef", &segment).unwrap();
        assert_eq!(
            halves
                .iter()
                .map(|h| (h.range.clone(), h.text.as_str()))
                .collect::<Vec<_>>(),
            [(0..3, "abc"), (3..6, "def")]
        );
        assert_eq!(fish.queue("abcdef", 3).unwrap()[0].text, "def");
    }

    /// The history pins the first chunk that made audio and keeps the
    /// latest one, as the family's session does, and survives a durable
    /// checkpoint exactly.
    #[test]
    fn chunk_history_pins_the_first_keeps_the_latest_and_round_trips() {
        let turn = |text: &str, frames: usize, salt: u32| SpeechChunkTurn {
            text: text.into(),
            codes: SpeechFrameCodes::new(
                (0..10u32)
                    .map(|k| {
                        (0..frames as u32)
                            .map(|f| (k * 409 + f * 13 + salt) % 4096)
                            .collect()
                    })
                    .collect(),
            ),
        };
        let mut history = SpeechChunkHistory::default();
        assert!(history.turns().is_empty());
        history.push(turn("one", 0, 0));
        assert!(
            history.turns().is_empty(),
            "a chunk without frames is no history"
        );
        history.push(turn("one", 3, 1));
        history.push(turn("two", 2, 2));
        history.push(turn("three", 4, 3));
        let held = history.turns();
        assert_eq!(
            held.iter().map(|t| t.text.as_str()).collect::<Vec<_>>(),
            ["one", "three"]
        );
        let checkpoint = history.to_checkpoint().unwrap();
        assert_eq!(checkpoint.turns[1].codebooks, 10);
        let json = serde_json::to_value(&checkpoint).unwrap();
        let recovered =
            SpeechChunkHistory::from_checkpoint(&serde_json::from_value(json).unwrap()).unwrap();
        assert_eq!(recovered, history);
        assert_eq!(recovered.turns()[1].codes, turn("three", 4, 3).codes);
        // A damaged checkpoint is refused.
        let mut bad = checkpoint.clone();
        bad.turns[0].codes.push_str("!!");
        assert!(SpeechChunkHistory::from_checkpoint(&bad).is_err());
        let mut ragged = checkpoint.clone();
        ragged.turns[0].codebooks = 7;
        assert!(SpeechChunkHistory::from_checkpoint(&ragged).is_err());
        let mut many = checkpoint;
        many.turns.push(many.turns[0].clone());
        assert!(SpeechChunkHistory::from_checkpoint(&many).is_err());
    }

    #[test]
    fn non_qwen_variant_stays_single_chunk() {
        let chunks =
            split_tts_text_for_long_form(ModelVariant::Kokoro82M, 0, "Sentence one. Sentence two.");
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn qwen_voice_design_and_base_pack_the_same_way() {
        for variant in [
            ModelVariant::Qwen3Tts12Hz17BVoiceDesign,
            ModelVariant::Qwen3Tts12Hz06BBase,
        ] {
            let chunks =
                split_tts_text_for_long_form(variant, 0, &"One sentence here. ".repeat(80));
            assert!(chunks.len() > 1, "{variant}");
            assert!(chunks.iter().all(|chunk| chunk.ends_with("here.")));
        }
    }
}
