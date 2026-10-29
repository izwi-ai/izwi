//! LFM3-Audio LiveASR: English speech transcribed while it is still being
//! spoken, as Liquid's `transcribe_live` does (`modeling_lfm3_audio.py` of
//! LiquidAI/LFM3-Audio-2B @ 9d49701c).
//!
//! The stream's mel features reach the encoder in chunks of 96 frames
//! (0.96 s, `interleaved_n_audio` encoder tokens). Each chunk's embeddings
//! feed the language model after what is pending, the prompt of the context
//! or the previous block's closing `<|text_end|>`, and the model writes the
//! chunk's text greedily until `<|text_end|>`: at most 48 tokens, 96 for the
//! last chunk. Before the last chunk it may not start speech, an interleaved
//! block or the end of the turn.
//!
//! One [`Lfm3LiveAsr`] is one segment: one text context of at most 30 s of
//! audio. A longer stream runs as segments of [`LFM3_LIVE_ASR_SEGMENT_SECONDS`];
//! each new one first hears the previous one's last
//! [`LFM3_LIVE_ASR_OVERLAP_SECONDS`] again, since the model writes a word a
//! chunk or so after it hears it, and [`Lfm3LiveTranscript`] drops the words
//! that overlap repeats. The old segment's last block, which heard audio cut
//! at the boundary (maybe mid-word), is not kept: the overlap covers it, and
//! the new segment hears it whole.

use std::collections::VecDeque;

use super::*;
use crate::models::architectures::lfm3_audio::encoder::Lfm3AudioEncoderStream;
use crate::models::architectures::lfm3_audio::preprocessor::{Lfm3AudioMelStream, HOP, N_MELS};

/// Seconds of audio one segment hears, its re-heard overlap included: under
/// a context's 30 s with room for the last partial chunk.
pub(crate) const LFM3_LIVE_ASR_SEGMENT_SECONDS: f32 = 29.0;
/// Seconds of the previous segment's end a new segment hears again.
pub(crate) const LFM3_LIVE_ASR_OVERLAP_SECONDS: f32 = 2.88;

/// What one push of `samples` 16 kHz samples (or the finish, `is_final`)
/// may take: the mel frames it adds, and the language model steps of the
/// blocks it completes, each a prefill and at most the last block's text
/// budget. Frames held from earlier pushes and the final flush's padding
/// complete at most two chunks more, and the finish adds the last partial
/// one.
pub(crate) fn lfm3_live_asr_operation_bounds(samples: usize, is_final: bool) -> (usize, usize) {
    let frames = samples.div_ceil(HOP).max(1);
    let blocks = frames / CHUNK_FRAMES + 2 + usize::from(is_final);
    (frames, blocks * (1 + FINAL_TEXT_TOKENS))
}

/// Mel frames per chunk: `interleaved_n_audio` encoder tokens.
const CHUNK_FRAMES: usize = INTERLEAVED_AUDIO * ENCODER_STRIDE;
/// Text tokens one chunk may write.
const CHUNK_TEXT_TOKENS: usize = 48;
/// Text tokens the last chunk may write.
const FINAL_TEXT_TOKENS: usize = 96;
/// Encoder tokens one text context hears: 30 s.
const CONTEXT_AUDIO_TOKENS: usize = 375;
/// Transcript words a new segment's repeat is matched against.
const RECENT_WORDS: usize = 16;
/// Ids that may never be written: the input and output speech placeholders.
const PLACEHOLDERS: [u32; 2] = [AUDIO_PLACEHOLDER, AUDIO_OUT_PLACEHOLDER];
/// Ids masked until the last chunk: speech, an interleaved block, the end of
/// the turn (`<|text_end|>` stays, since it closes every block), and the
/// card's `<|mixed_end|>` and `<|endoftext|>`.
const SUPPRESSED_UNTIL_FINAL: [u32; 5] = [AUDIO_START, MIXED_START, IM_END, MIXED_END, END_OF_TEXT];
/// Ids that end the last chunk besides `<|text_end|>`.
const FINAL_STOPS: [u32; 3] = [IM_END, END_OF_TEXT, IM_START];

/// The block being written: its token budget, the tokens written, and
/// whether it is the segment's last.
#[derive(Debug, Clone, Copy)]
struct LiveBlock {
    budget: usize,
    written: usize,
    is_final: bool,
}

/// One segment of a live transcription: the mel and encoder streams, the
/// chunks heard but not yet fed, and where the language model's context and
/// the block being written stand. Tensors are immutable, so a clone is an
/// exact checkpoint of everything but the language model's state.
#[derive(Clone)]
pub(crate) struct Lfm3LiveAsr {
    mel: Lfm3AudioMelStream,
    encoder: Lfm3AudioEncoderStream,
    /// Mel frames not yet a whole chunk, one row per mel band.
    frames: Vec<Vec<f32>>,
    /// Chunks heard and not yet fed: their embeddings (none for a last
    /// chunk that completes no token) and whether each is the last.
    queued: VecDeque<(Option<Tensor>, bool)>,
    /// Fed before the next chunk's audio: the previous block's
    /// `<|text_end|>`; the context's prompt before the first chunk.
    pending: Option<Tensor>,
    /// Positions the language-model context holds.
    position: usize,
    /// Encoder tokens the context has heard.
    context_audio_tokens: usize,
    block: Option<LiveBlock>,
    decoder: IncrementalDecoder,
    /// Text written and not yet taken.
    text: String,
    finished: bool,
}

impl Lfm3LiveAsr {
    /// Whether the segment has heard its last chunk and written its text.
    pub(crate) fn finished(&self) -> bool {
        self.finished
    }

    /// Text written since the last call.
    pub(crate) fn take_text(&mut self) -> String {
        std::mem::take(&mut self.text)
    }
}

impl Lfm3AudioModel {
    pub(crate) fn start_live_asr(&self) -> Lfm3LiveAsr {
        Lfm3LiveAsr {
            mel: Lfm3AudioMelStream::default(),
            encoder: Lfm3AudioEncoderStream::default(),
            frames: vec![Vec::new(); N_MELS],
            queued: VecDeque::new(),
            pending: None,
            position: 0,
            context_audio_tokens: 0,
            block: None,
            decoder: IncrementalDecoder::new(true),
            text: String::new(),
            finished: false,
        }
    }

    /// Hear 16 kHz mono `samples`, the segment's last with `is_final`:
    /// encode every whole chunk they complete (and, at the end, the rest)
    /// and queue it.
    pub(crate) fn hear_live_asr(
        &self,
        live: &mut Lfm3LiveAsr,
        samples: &[f32],
        is_final: bool,
    ) -> Result<()> {
        if live.finished || live.queued.back().is_some_and(|(_, last)| *last) {
            return Err(Error::InvalidInput(
                "LFM3-Audio live transcription has heard its last audio".into(),
            ));
        }
        let (features, count) = self.mel.stream(&mut live.mel, samples, is_final)?;
        for (mel, row) in live.frames.iter_mut().enumerate() {
            row.extend_from_slice(&features[mel * count..][..count]);
        }
        while live.frames[0].len() >= CHUNK_FRAMES {
            let chunk = take_frames(&mut live.frames, CHUNK_FRAMES);
            let audio = self
                .encoder
                .stream(&mut live.encoder, &chunk, CHUNK_FRAMES, false)?;
            live.queued.push_back((audio, false));
        }
        if is_final {
            let frames = live.frames[0].len();
            let chunk = take_frames(&mut live.frames, frames);
            let audio = self
                .encoder
                .stream(&mut live.encoder, &chunk, frames, true)?;
            if live.encoder.tokens() == 0 {
                // A segment that heard no speech token writes nothing.
                live.finished = true;
            } else {
                live.queued.push_back((audio, true));
            }
        }
        Ok(())
    }

    /// Open the next queued chunk's block: what to feed (the pending
    /// embedding and the chunk's audio) at which position, or `None` when no
    /// chunk is queued.
    pub(crate) fn open_live_asr_block(
        &self,
        live: &mut Lfm3LiveAsr,
    ) -> Result<Option<(Tensor, usize)>> {
        if live.block.is_some() {
            return Err(Error::InvalidInput(
                "LFM3-Audio live block opened before the last one closed".into(),
            ));
        }
        let Some((audio, is_final)) = live.queued.pop_front() else {
            return Ok(None);
        };
        let audio_tokens = audio.as_ref().map_or(Ok(0), |audio| audio.dim(1))?;
        if live.context_audio_tokens + audio_tokens > CONTEXT_AUDIO_TOKENS {
            return Err(Error::InvalidInput(
                "an LFM3-Audio live segment hears at most 30 s; start a new one".into(),
            ));
        }
        let pending = match live.pending.take() {
            Some(pending) => pending,
            None => self.live_asr_prompt()?,
        };
        let input = match audio {
            Some(audio) => Tensor::cat(&[pending, audio], 1)?,
            None => pending,
        };
        let position = live.position;
        live.position += input.dim(1)?;
        live.context_audio_tokens += audio_tokens;
        live.block = Some(LiveBlock {
            budget: if is_final {
                FINAL_TEXT_TOKENS
            } else {
                CHUNK_TEXT_TOKENS
            },
            written: 0,
            is_final,
        });
        Ok(Some((input, position)))
    }

    /// The open block's next token, after a feed whose last hidden state is
    /// `last` (`[1, 2048]`): the token's embedding to feed at which
    /// position, or `None` once the block has closed.
    pub(crate) fn step_live_asr(
        &self,
        live: &mut Lfm3LiveAsr,
        last: &Tensor,
    ) -> Result<Option<(Tensor, usize)>> {
        let block = live.block.ok_or_else(|| {
            Error::InvalidInput("LFM3-Audio live step without an open block".into())
        })?;
        let token = if block.written < block.budget {
            let mut scores = self
                .lm
                .project_hidden(last)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec1::<f32>()?;
            for id in PLACEHOLDERS {
                scores[id as usize] = f32::NEG_INFINITY;
            }
            if !block.is_final {
                for id in SUPPRESSED_UNTIL_FINAL {
                    scores[id as usize] = f32::NEG_INFINITY;
                }
            }
            Some(greedy(&scores))
                .filter(|&token| token != TEXT_END)
                .filter(|token| !(block.is_final && FINAL_STOPS.contains(token)))
        } else {
            None
        };
        let Some(token) = token else {
            // The block closes before the next chunk, whether the model
            // wrote `<|text_end|>` or ran out of tokens.
            live.block = None;
            if block.is_final {
                let tail = self
                    .tokenizer
                    .finish_incremental_decode(&mut live.decoder)?;
                live.text.push_str(&tail);
                live.finished = true;
            } else {
                live.pending = Some(self.embed_live_asr_token(TEXT_END)?);
            }
            return Ok(None);
        };
        let text = self
            .tokenizer
            .decode_incrementally(&mut live.decoder, token)?;
        live.text.push_str(&text);
        live.block = Some(LiveBlock {
            written: block.written + 1,
            ..block
        });
        let position = live.position;
        live.position += 1;
        Ok(Some((self.embed_live_asr_token(token)?, position)))
    }

    /// Hear `samples` (the segment's last with `is_final`) and write their
    /// text on `state`, which holds this segment's context.
    pub(crate) fn push_live_asr(
        &self,
        live: &mut Lfm3LiveAsr,
        samples: &[f32],
        is_final: bool,
        state: &mut dyn Lfm3AudioLmState,
    ) -> Result<String> {
        self.hear_live_asr(live, samples, is_final)?;
        while let Some((input, position)) = self.open_live_asr_block(live)? {
            if position == 0 {
                state.reset()?;
            }
            let span = input.dim(1)?;
            let hidden = state.forward(&self.lm, &input, position)?;
            let mut last = hidden.narrow(1, span - 1, 1)?.squeeze(1)?;
            while let Some((embed, position)) = self.step_live_asr(live, &last)? {
                last = state.forward(&self.lm, &embed, position)?.squeeze(1)?;
            }
        }
        Ok(live.take_text())
    }

    fn live_asr_prompt(&self) -> Result<Tensor> {
        let (prompt, _) = self.prompt_embeds(&Lfm3AudioRequest {
            prompt: live_asr_prompt(),
            clips: Vec::new(),
            mode: Lfm3AudioMode::Text,
            text_only: true,
            max_slots: FINAL_TEXT_TOKENS,
            text_sampling: Lfm3TextSampling::default(),
            audio_sampling: Lfm3AudioSampling::default(),
            seed: 0,
        })?;
        Ok(prompt)
    }

    fn embed_live_asr_token(&self, token: u32) -> Result<Tensor> {
        self.lm
            .embed_tokens(&Tensor::from_vec(vec![token], (1, 1), &self.device.device)?)
    }
}

/// A live transcript across segments. A new segment first hears the
/// previous one's end again, and catches up on it a chunk or two later: its
/// text is held until it moves past the words the transcript ends with (or
/// plainly does not repeat them), and only the words after the repeat are
/// kept. A new segment's text is spaced from the last one's.
#[derive(Debug, Clone, Default)]
pub(crate) struct Lfm3LiveTranscript {
    /// The transcript's last words, lowercased without punctuation.
    recent: VecDeque<String>,
    /// The transcript so far, if any, does not end in whitespace.
    open_word: bool,
    separate: bool,
    /// The new segment's text while it may still repeat the transcript.
    held: Option<String>,
}

/// How a new segment's words line up with the transcript's last ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Overlap {
    /// Its first `n` words repeat the transcript up to its end.
    Through(usize),
    /// Its words repeat a stretch of the transcript that has not yet
    /// reached the end.
    Within,
    None,
}

impl Lfm3LiveTranscript {
    /// A new segment starts, hearing the previous one's end again: what an
    /// earlier one still held is settled first.
    pub(crate) fn start_segment(&mut self) -> String {
        let settled = self.release(true);
        self.separate = true;
        self.held = Some(String::new());
        settled
    }

    /// A segment's text: what the transcript gains by it.
    pub(crate) fn push(&mut self, piece: &str) -> String {
        match self.held.as_mut() {
            Some(held) => {
                held.push_str(piece);
                self.release(false)
            }
            None => self.emit(piece),
        }
    }

    /// The stream or the segment ends: settle the text still held.
    pub(crate) fn flush(&mut self) -> String {
        self.release(true)
    }

    /// Emit the held text past its repeat once that can be told: the
    /// complete words after a repeat that reaches the transcript's end, all
    /// of them when they repeat nothing, none while they may still be
    /// catching up. `settle` decides with what is held.
    fn release(&mut self, settle: bool) -> String {
        let Some(held) = self.held.as_ref() else {
            return String::new();
        };
        let starts = word_starts(held);
        let complete = if settle || held.ends_with(char::is_whitespace) {
            starts.len()
        } else {
            starts.len().saturating_sub(1)
        };
        let keys = held
            .split_whitespace()
            .take(complete)
            .map(normalize)
            .collect::<Vec<_>>();
        let from = match overlap(&self.recent, &keys) {
            Overlap::Through(count) if count < complete => Some(starts[count]),
            Overlap::Through(_) | Overlap::Within if settle => None,
            Overlap::None if settle || complete >= 3 => Some(0),
            _ => return String::new(),
        };
        let held = self.held.take().unwrap_or_default();
        match from {
            Some(from) => self.emit(&held[from..]),
            None => String::new(),
        }
    }

    fn emit(&mut self, piece: &str) -> String {
        if piece.is_empty() {
            return String::new();
        }
        let mut piece = piece.to_string();
        if std::mem::take(&mut self.separate)
            && self.open_word
            && !piece.starts_with(char::is_whitespace)
        {
            piece.insert(0, ' ');
        }
        // A word split across pieces is completed by the next one.
        let continues = !piece.starts_with(char::is_whitespace) && self.open_word;
        let mut words = piece.split_whitespace().map(normalize);
        if continues {
            if let (Some(last), Some(rest)) = (self.recent.back_mut(), words.next()) {
                last.push_str(&rest);
            }
        }
        self.recent.extend(words.filter(|word| !word.is_empty()));
        while self.recent.len() > RECENT_WORDS {
            self.recent.pop_front();
        }
        self.open_word = !piece.ends_with(char::is_whitespace);
        piece
    }
}

/// A word lowercased without its punctuation.
fn normalize(word: &str) -> String {
    word.chars()
        .filter(|ch| ch.is_alphanumeric() || *ch == '\'')
        .flat_map(char::to_lowercase)
        .collect()
}

/// The byte offset of each whitespace-separated word of `text`.
fn word_starts(text: &str) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut previous_space = true;
    for (offset, ch) in text.char_indices() {
        let space = ch.is_whitespace();
        if previous_space && !space {
            starts.push(offset);
        }
        previous_space = space;
    }
    starts
}

/// How `keys`, a new segment's words, line up with `recent`: a repeat from
/// the first word, or (two words at least) from the second or third, the
/// first ones being a word the chunk edge cut.
fn overlap(recent: &VecDeque<String>, keys: &[String]) -> Overlap {
    let recent = recent.iter().map(String::as_str).collect::<Vec<_>>();
    for count in (1..=keys.len().min(recent.len())).rev() {
        for skip in (0..=2).filter(|&skip| skip == 0 || count >= 2) {
            if skip + count <= keys.len()
                && recent[recent.len() - count..]
                    .iter()
                    .copied()
                    .eq(keys[skip..skip + count].iter().map(String::as_str))
            {
                return Overlap::Through(skip + count);
            }
        }
    }
    for skip in (0..=2).filter(|&skip| skip < keys.len()) {
        let tail = &keys[skip..];
        if recent
            .windows(tail.len())
            .any(|window| window.iter().copied().eq(tail.iter().map(String::as_str)))
        {
            return Overlap::Within;
        }
    }
    Overlap::None
}

/// The first `count` frames of every mel row, `[N_MELS, count]` mel-major.
fn take_frames(rows: &mut [Vec<f32>], count: usize) -> Vec<f32> {
    let mut chunk = Vec::with_capacity(rows.len() * count);
    for row in rows {
        chunk.extend(row.drain(..count));
    }
    chunk
}

/// The highest score's id; the first on a tie, as `argmax` picks.
fn greedy(scores: &[f32]) -> u32 {
    let mut best = 0;
    for (id, &score) in scores.iter().enumerate() {
        if score > scores[best] {
            best = id;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stitch(old: &[&str], new: &[&str]) -> String {
        let mut transcript = Lfm3LiveTranscript::default();
        let mut text = String::new();
        for piece in old {
            text.push_str(&transcript.push(piece));
        }
        text.push_str(&transcript.start_segment());
        for piece in new {
            text.push_str(&transcript.push(piece));
        }
        text.push_str(&transcript.flush());
        text
    }

    #[test]
    fn a_new_segments_repeat_of_the_transcript_end_is_dropped() {
        let old = ["it's about amplif", "ying it to solve the impossible."];
        let expected = "it's about amplifying it to solve the impossible. The fox.";
        // Caught up within the overlap, after it, or cut mid-word.
        for new in [
            &["it to solve the impossible.", " The fox."][..],
            &[
                "",
                " It's about",
                " amplifying it to",
                " solve the impossible.",
                " The fox.",
            ],
            &["fying it to solve the", " impossible. The fox."],
            &["The impossible. The fox."],
        ] {
            assert_eq!(stitch(&old, new), expected, "{new:?}");
        }
        // Nothing new by the end, and nothing repeated.
        assert_eq!(
            stitch(&old, &[" to solve the"]),
            "it's about amplifying it to solve the impossible."
        );
        assert_eq!(
            stitch(&old, &["Hello there, friend."]),
            "it's about amplifying it to solve the impossible. Hello there, friend."
        );
    }

    #[test]
    fn pieces_join_into_words_and_a_new_segment_is_spaced() {
        let mut transcript = Lfm3LiveTranscript::default();
        let mut text = String::new();
        for piece in ["We're no", " longer amplif", "ying it to solve the"] {
            text.push_str(&transcript.push(piece));
        }
        text.push_str(&transcript.start_segment());
        text.push_str(&transcript.push("to solve the imposs"));
        text.push_str(&transcript.push("ible."));
        text.push_str(&transcript.push(" The fox."));
        text.push_str(&transcript.flush());
        assert_eq!(
            text,
            "We're no longer amplifying it to solve the impossible. The fox."
        );
    }
}
