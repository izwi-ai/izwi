//! LFM3-Audio LiveASR: English speech transcribed while it is still being
//! spoken, as Liquid's `transcribe_live` does (`modeling_lfm3_audio.py` of
//! LiquidAI/LFM3-Audio-2B @ 9d49701c).
//!
//! The stream's mel features reach the encoder in chunks of 96 frames
//! (0.96 s, `interleaved_n_audio` encoder tokens). Each chunk's embeddings
//! feed the language model after what is pending, the prompt of a new
//! context or the previous block's closing `<|text_end|>`, and the model
//! writes the chunk's text greedily until `<|text_end|>`: at most 48 tokens,
//! 96 for the last chunk. Before the last chunk it may not start speech, an
//! interleaved block or the end of the turn.
//!
//! A context hears at most 30 s of audio; the next chunk then starts a new
//! one with the prompt, while the encoder keeps following the stream. The
//! model writes a word a chunk or so after it hears it, and a fresh context
//! writes little for its first chunk, so a new context first hears the last
//! three chunks again: it writes the words the old one had not, and the
//! words it repeats from the transcript's end are dropped.

use std::collections::VecDeque;

use super::*;
use crate::models::architectures::lfm3_audio::encoder::Lfm3AudioEncoderStream;
use crate::models::architectures::lfm3_audio::preprocessor::{Lfm3AudioMelStream, N_MELS};

/// Mel frames per chunk: `interleaved_n_audio` encoder tokens.
const CHUNK_FRAMES: usize = INTERLEAVED_AUDIO * ENCODER_STRIDE;
/// Text tokens one chunk may write.
const CHUNK_TEXT_TOKENS: usize = 48;
/// Text tokens the last chunk may write.
const FINAL_TEXT_TOKENS: usize = 96;
/// Encoder tokens one text context hears: 30 s.
const CONTEXT_AUDIO_TOKENS: usize = 375;
/// Transcript words a new context's repeat is matched against.
const RECENT_WORDS: usize = 16;
/// Chunks a new context hears again before its first new one.
const REHEARD_CHUNKS: usize = 3;
/// Ids that may never be written: the input and output speech placeholders.
const PLACEHOLDERS: [u32; 2] = [AUDIO_PLACEHOLDER, AUDIO_OUT_PLACEHOLDER];
/// Ids masked until the last chunk: speech, an interleaved block, the end of
/// the turn (`<|text_end|>` stays, since it closes every block), and the
/// card's `<|mixed_end|>` and `<|endoftext|>`.
const SUPPRESSED_UNTIL_FINAL: [u32; 5] = [AUDIO_START, MIXED_START, IM_END, MIXED_END, END_OF_TEXT];
/// Ids that end the last chunk besides `<|text_end|>`.
const FINAL_STOPS: [u32; 3] = [IM_END, END_OF_TEXT, IM_START];

/// One live transcription: the mel and encoder streams, the mel frames short
/// of a chunk, and where the language model's context stands.
pub(crate) struct Lfm3LiveAsr {
    mel: Lfm3AudioMelStream,
    encoder: Lfm3AudioEncoderStream,
    /// Mel frames not yet a whole chunk, one row per mel band.
    frames: Vec<Vec<f32>>,
    /// Fed before the next chunk's audio: the prompt of a new context, or
    /// the previous block's `<|text_end|>`.
    pending: Option<Tensor>,
    /// Positions the language-model context holds.
    position: usize,
    /// Encoder tokens the context has heard.
    context_audio_tokens: usize,
    /// The latest chunks' embeddings, which a new context hears again.
    recent_audio: VecDeque<Tensor>,
    /// The transcript's last words, lowercased without punctuation.
    recent: VecDeque<String>,
    /// The transcript so far, if any, does not end in whitespace: a new
    /// context's first words need a space before them.
    open_word: bool,
    separate: bool,
    decoder: IncrementalDecoder,
    finished: bool,
}

impl Lfm3LiveAsr {
    pub(crate) fn finished(&self) -> bool {
        self.finished
    }

    /// Append `piece` to the transcript: after a space when it opens a new
    /// context's text, and into the recent words.
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

impl Lfm3AudioModel {
    pub(crate) fn start_live_asr(&self) -> Lfm3LiveAsr {
        Lfm3LiveAsr {
            mel: Lfm3AudioMelStream::default(),
            encoder: Lfm3AudioEncoderStream::default(),
            frames: vec![Vec::new(); N_MELS],
            pending: None,
            position: 0,
            context_audio_tokens: 0,
            recent_audio: VecDeque::new(),
            recent: VecDeque::new(),
            open_word: false,
            separate: false,
            decoder: IncrementalDecoder::new(true),
            finished: false,
        }
    }

    /// Feed 16 kHz mono `samples` and return the text of every chunk they
    /// complete. `is_final` flushes the stream and closes the transcript.
    pub(crate) fn push_live_asr(
        &self,
        live: &mut Lfm3LiveAsr,
        samples: &[f32],
        is_final: bool,
        state: &mut dyn Lfm3AudioLmState,
    ) -> Result<String> {
        if live.finished {
            return Err(Error::InvalidInput(
                "LFM3-Audio live transcription is finished".into(),
            ));
        }
        let (features, count) = self.mel.stream(&mut live.mel, samples, is_final)?;
        for (mel, row) in live.frames.iter_mut().enumerate() {
            row.extend_from_slice(&features[mel * count..][..count]);
        }
        let mut text = String::new();
        while live.frames[0].len() >= CHUNK_FRAMES {
            let chunk = take_frames(&mut live.frames, CHUNK_FRAMES);
            text.push_str(&self.live_asr_chunk(live, &chunk, CHUNK_FRAMES, false, state)?);
        }
        if is_final {
            let frames = live.frames[0].len();
            let chunk = take_frames(&mut live.frames, frames);
            text.push_str(&self.live_asr_chunk(live, &chunk, frames, true, state)?);
            let tail = self
                .tokenizer
                .finish_incremental_decode(&mut live.decoder)?;
            text.push_str(&live.emit(&tail));
            live.finished = true;
        }
        Ok(text)
    }

    /// One chunk: its audio heard after what is pending, then its text. A
    /// chunk that would take the context past 30 s, or past its room, opens
    /// a new one, which first hears the last chunks again.
    fn live_asr_chunk(
        &self,
        live: &mut Lfm3LiveAsr,
        features: &[f32],
        frames: usize,
        is_final: bool,
        state: &mut dyn Lfm3AudioLmState,
    ) -> Result<String> {
        let audio = self
            .encoder
            .stream(&mut live.encoder, features, frames, is_final)?;
        let audio_tokens = audio.as_ref().map_or(Ok(0), |audio| audio.dim(1))?;
        let budget = if is_final {
            FINAL_TEXT_TOKENS
        } else {
            CHUNK_TEXT_TOKENS
        };
        let room = state.capacity_tokens().min(self.lm_config.context_length);
        let mut text = String::new();
        if live.pending.is_none()
            || live.context_audio_tokens + audio_tokens > CONTEXT_AUDIO_TOKENS
            || live.position + 1 + audio_tokens + budget >= room
        {
            let reheard = if live.pending.is_some() {
                std::mem::take(&mut live.recent_audio)
            } else {
                VecDeque::new()
            };
            let tail = self
                .tokenizer
                .finish_incremental_decode(&mut live.decoder)?;
            text.push_str(&live.emit(&tail));
            live.decoder = IncrementalDecoder::new(true);
            self.open_live_asr_context(live, state)?;
            live.separate = true;
            if !reheard.is_empty() {
                let mut decoder = IncrementalDecoder::new(true);
                let mut heard = String::new();
                for chunk in &reheard {
                    heard.push_str(&self.live_asr_block(
                        live,
                        Some(chunk.clone()),
                        CHUNK_TEXT_TOKENS,
                        false,
                        &mut decoder,
                        state,
                    )?);
                }
                heard.push_str(&self.tokenizer.finish_incremental_decode(&mut decoder)?);
                text.push_str(&live.emit(&after_overlap(&live.recent, &heard)));
            }
            live.recent_audio = reheard;
        }
        let mut decoder = std::mem::replace(&mut live.decoder, IncrementalDecoder::new(true));
        let written =
            self.live_asr_block(live, audio.clone(), budget, is_final, &mut decoder, state);
        live.decoder = decoder;
        text.push_str(&live.emit(&written?));
        if let Some(audio) = audio {
            live.recent_audio.push_back(audio);
            while live.recent_audio.len() > REHEARD_CHUNKS {
                live.recent_audio.pop_front();
            }
        }
        Ok(text)
    }

    /// Start a new text context: empty state and the prompt pending.
    fn open_live_asr_context(
        &self,
        live: &mut Lfm3LiveAsr,
        state: &mut dyn Lfm3AudioLmState,
    ) -> Result<()> {
        state.reset()?;
        let (prompt, _) = self.prompt_embeds(&Lfm3AudioRequest {
            prompt: live_asr_prompt(),
            audio: None,
            mode: Lfm3AudioMode::Text,
            text_only: true,
            max_slots: FINAL_TEXT_TOKENS,
            text_sampling: Lfm3TextSampling::default(),
            audio_sampling: Lfm3AudioSampling::default(),
            seed: 0,
        })?;
        live.pending = Some(prompt);
        live.position = 0;
        live.context_audio_tokens = 0;
        Ok(())
    }

    /// Feed what is pending and `audio`, then write text greedily until
    /// `<|text_end|>` (or a final stop), at most `budget` tokens.
    fn live_asr_block(
        &self,
        live: &mut Lfm3LiveAsr,
        audio: Option<Tensor>,
        budget: usize,
        is_final: bool,
        decoder: &mut IncrementalDecoder,
        state: &mut dyn Lfm3AudioLmState,
    ) -> Result<String> {
        let audio_tokens = audio.as_ref().map_or(Ok(0), |audio| audio.dim(1))?;
        let pending = live.pending.take().ok_or_else(|| {
            Error::InferenceError("LFM3-Audio live transcription lost its context".into())
        })?;
        let input = match audio {
            Some(audio) => Tensor::cat(&[pending, audio], 1)?,
            None => pending,
        };
        let span = input.dim(1)?;
        let room = state.capacity_tokens().min(self.lm_config.context_length);
        if live.position + span + budget >= room {
            return Err(Error::InvalidInput(
                "LFM3-Audio live transcription chunk does not fit its context".into(),
            ));
        }
        let hidden = state.forward(&self.lm, &input, live.position)?;
        live.position += span;
        live.context_audio_tokens += audio_tokens;
        let mut last = hidden.narrow(1, span - 1, 1)?.squeeze(1)?;
        let mut text = String::new();
        for _ in 0..budget {
            let mut scores = self
                .lm
                .project_hidden(&last)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec1::<f32>()?;
            for id in PLACEHOLDERS {
                scores[id as usize] = f32::NEG_INFINITY;
            }
            if !is_final {
                for id in SUPPRESSED_UNTIL_FINAL {
                    scores[id as usize] = f32::NEG_INFINITY;
                }
            }
            let token = greedy(&scores);
            if token == TEXT_END || (is_final && FINAL_STOPS.contains(&token)) {
                break;
            }
            text.push_str(&self.tokenizer.decode_incrementally(decoder, token)?);
            let embed = self.lm.embed_tokens(&Tensor::from_vec(
                vec![token],
                (1, 1),
                &self.device.device,
            )?)?;
            last = state.forward(&self.lm, &embed, live.position)?.squeeze(1)?;
            live.position += 1;
        }
        // The block closes before the next chunk, whether the model wrote
        // `<|text_end|>` or ran out of tokens.
        if !is_final {
            live.pending = Some(self.lm.embed_tokens(&Tensor::from_vec(
                vec![TEXT_END],
                (1, 1),
                &self.device.device,
            )?)?);
        }
        Ok(text)
    }
}

/// A word lowercased without its punctuation.
fn normalize(word: &str) -> String {
    word.chars()
        .filter(|ch| ch.is_alphanumeric() || *ch == '\'')
        .flat_map(char::to_lowercase)
        .collect()
}

/// `heard` after the longest run of its words that repeats the end of
/// `recent`: from its first word, or (two words at least) from its second or
/// third, the first ones being a word the chunk edge cut.
fn after_overlap(recent: &VecDeque<String>, heard: &str) -> String {
    let words = heard.split_whitespace().collect::<Vec<_>>();
    let keys = words.iter().map(|word| normalize(word)).collect::<Vec<_>>();
    let repeats = |skip: usize, count: usize| {
        skip + count <= keys.len()
            && count <= recent.len()
            && recent
                .iter()
                .skip(recent.len() - count)
                .eq(keys[skip..skip + count].iter())
    };
    let repeated = (1..=keys.len().min(recent.len()))
        .rev()
        .find_map(|count| {
            (0..=2)
                .filter(|&skip| skip == 0 || count >= 2)
                .find(|&skip| repeats(skip, count))
                .map(|skip| skip + count)
        })
        .unwrap_or(0);
    words[repeated..].join(" ")
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

    fn recent(text: &str) -> VecDeque<String> {
        text.split_whitespace().map(normalize).collect()
    }

    #[test]
    fn a_new_contexts_repeat_of_the_transcript_end_is_dropped() {
        let tail = recent("it's about amplifying it to solve the");
        assert_eq!(
            after_overlap(&tail, "it to solve the impossible."),
            "impossible."
        );
        assert_eq!(after_overlap(&tail, "The impossible."), "impossible.");
        assert_eq!(after_overlap(&tail, "impossible."), "impossible.");
        assert_eq!(after_overlap(&tail, "solve the"), "");
        assert_eq!(
            after_overlap(&tail, "fying it to solve the impossible."),
            "impossible."
        );
        assert_eq!(
            after_overlap(&VecDeque::new(), "Hello there."),
            "Hello there."
        );
    }

    #[test]
    fn pieces_join_into_words_and_a_new_context_is_spaced() {
        let mut live = Lfm3LiveAsr {
            mel: Lfm3AudioMelStream::default(),
            encoder: Lfm3AudioEncoderStream::default(),
            frames: Vec::new(),
            pending: None,
            position: 0,
            context_audio_tokens: 0,
            recent_audio: VecDeque::new(),
            recent: VecDeque::new(),
            open_word: false,
            separate: false,
            decoder: IncrementalDecoder::new(true),
            finished: false,
        };
        let mut transcript = String::new();
        for piece in ["We're no", " longer amplif", "ying it"] {
            transcript.push_str(&live.emit(piece));
        }
        live.separate = true;
        transcript.push_str(&live.emit("The fox."));
        assert_eq!(transcript, "We're no longer amplifying it The fox.");
        assert_eq!(
            live.recent.iter().map(String::as_str).collect::<Vec<_>>(),
            ["we're", "no", "longer", "amplifying", "it", "the", "fox"]
        );
    }
}
