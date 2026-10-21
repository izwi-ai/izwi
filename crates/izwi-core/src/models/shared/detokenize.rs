//! Incremental detokenization for decoders that stream text token by token.
//!
//! Re-decoding the whole generated sequence on every token is O(n²) over a
//! transcript. [`IncrementalDetokenizer`] generalises the `tokenizers`
//! crate's decode stream to any whole-sequence decode function: each step
//! decodes only a short window that starts one emission back, so the work per
//! token stays bounded however long the transcript grows. The window keeps
//! enough context for decoders that depend on neighbouring tokens
//! (SentencePiece `▁` word boundaries, a leading space stripped from the
//! first piece, `" ."` normalisation), and a step whose text ends in U+FFFD or
//! does not extend the window's previous text is held back until a later
//! token completes it. [`IncrementalDetokenizer::finish`] flushes whatever is
//! still held back, so the concatenated deltas equal one decode of the whole
//! sequence.
//!
//! The one decoder this cannot match is `tokenizers`' `ByteFallback` given
//! byte tokens that never form UTF-8: a later stray byte turns an earlier,
//! already emitted ASCII byte of the same run into U+FFFD. Models emit
//! byte-fallback tokens as whole characters, which stream exactly.

use crate::error::{Error, Result};

/// Window offsets into a caller-owned token sequence. The caller keeps the
/// tokens (every decoder here needs them anyway); the detokenizer keeps only
/// the window boundaries and that window's previous text, so it is cheap to
/// clone into a rollback checkpoint.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct IncrementalDetokenizer {
    /// First token of the decode window.
    prefix_offset: usize,
    /// Tokens before this offset have been emitted.
    read_offset: usize,
    /// Text of `tokens[prefix_offset..read_offset]`.
    prefix_text: String,
    /// Tokens passed to the decode function so far.
    decoded_tokens: u64,
    /// Steps whose window text did not extend the previous text.
    divergences: u64,
}

impl IncrementalDetokenizer {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The text that `tokens`' newest tokens add. `tokens` is the whole
    /// sequence so far: the one passed to the previous call plus any newly
    /// appended tokens. An empty result means the tokens do not yet form
    /// complete text and are held back.
    pub(crate) fn step<T>(
        &mut self,
        tokens: &[T],
        mut decode: impl FnMut(&[T]) -> Result<String>,
    ) -> Result<String> {
        self.check_len(tokens)?;
        if tokens.len() == self.read_offset {
            return Ok(String::new());
        }
        let text = self.decode_window(&tokens[self.prefix_offset..], &mut decode)?;
        if text.len() <= self.prefix_text.len() || text.ends_with(char::REPLACEMENT_CHARACTER) {
            return Ok(String::new());
        }
        let Some(delta) = text.strip_prefix(self.prefix_text.as_str()) else {
            self.divergences += 1;
            return Ok(String::new());
        };
        let delta = delta.to_owned();
        self.prefix_text = self.decode_window(&tokens[self.read_offset..], &mut decode)?;
        self.prefix_offset = self.read_offset;
        self.read_offset = tokens.len();
        Ok(delta)
    }

    /// The text still held back once `tokens` is complete: an incomplete
    /// trailing character decodes as the decode function renders it. After
    /// this call every token counts as emitted.
    pub(crate) fn finish<T>(
        &mut self,
        tokens: &[T],
        mut decode: impl FnMut(&[T]) -> Result<String>,
    ) -> Result<String> {
        self.check_len(tokens)?;
        if tokens.len() == self.read_offset {
            return Ok(String::new());
        }
        let text = self.decode_window(&tokens[self.prefix_offset..], &mut decode)?;
        let delta = match text.strip_prefix(self.prefix_text.as_str()) {
            Some(delta) => delta.to_owned(),
            None => {
                // The emitted text cannot be retracted; append what follows
                // the longest shared prefix.
                self.divergences += 1;
                let shared = self
                    .prefix_text
                    .char_indices()
                    .zip(text.chars())
                    .find(|((_, held), next)| held != next)
                    .map_or(self.prefix_text.len().min(text.len()), |((at, _), _)| at);
                text.get(shared..).unwrap_or_default().to_owned()
            }
        };
        self.prefix_text = text;
        self.read_offset = tokens.len();
        Ok(delta)
    }

    /// Tokens passed to the decode function so far: O(n) for an n-token
    /// sequence, where whole-sequence re-decoding is O(n²).
    pub(crate) fn decoded_tokens(&self) -> u64 {
        self.decoded_tokens
    }

    /// Steps whose decode did not extend the window's previous text. A
    /// prefix-stable decoder never diverges.
    pub(crate) fn divergences(&self) -> u64 {
        self.divergences
    }

    fn check_len<T>(&self, tokens: &[T]) -> Result<()> {
        if tokens.len() < self.read_offset {
            return Err(Error::InferenceError(format!(
                "incremental detokenizer saw {} tokens after emitting {}",
                tokens.len(),
                self.read_offset
            )));
        }
        Ok(())
    }

    fn decode_window<T>(
        &mut self,
        window: &[T],
        decode: &mut impl FnMut(&[T]) -> Result<String>,
    ) -> Result<String> {
        self.decoded_tokens = self.decoded_tokens.saturating_add(window.len() as u64);
        decode(window)
    }
}

/// Deltas of `tokens` decoded one token at a time, then flushed: test oracle
/// for the decoders that adopt [`IncrementalDetokenizer`].
#[cfg(test)]
pub(crate) fn stream_deltas<T: Clone>(
    tokens: &[T],
    mut decode: impl FnMut(&[T]) -> Result<String>,
) -> (Vec<String>, IncrementalDetokenizer) {
    let mut detokenizer = IncrementalDetokenizer::new();
    let mut deltas = Vec::with_capacity(tokens.len() + 1);
    for end in 1..=tokens.len() {
        deltas.push(detokenizer.step(&tokens[..end], &mut decode).unwrap());
    }
    deltas.push(detokenizer.finish(tokens, &mut decode).unwrap());
    (deltas, detokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use tokenizers::Tokenizer as HfTokenizer;

    /// Characters whose UTF-8 spans 1–4 bytes, so byte-level tokens split
    /// them across steps.
    const TEXT_CHARS: &[&str] = &[
        "a", "b", "z", " ", ".", ",", "é", "ß", "你", "好", "世", "界", "😀", "🎉", "ñ", "\n",
    ];

    fn byte_level_tokenizer() -> HfTokenizer {
        let alphabet = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet();
        let mut chars = alphabet.into_iter().collect::<Vec<_>>();
        chars.sort_unstable();
        let vocab = chars
            .iter()
            .enumerate()
            .map(|(id, ch)| format!("{}: {id}", serde_json::json!(ch.to_string())))
            .collect::<Vec<_>>()
            .join(", ");
        let json = format!(
            r#"{{
                "version": "1.0",
                "added_tokens": [],
                "normalizer": null,
                "pre_tokenizer": {{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}},
                "post_processor": null,
                "decoder": {{"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}},
                "model": {{"type": "BPE", "vocab": {{{vocab}}}, "merges": []}}
            }}"#
        );
        json.parse().expect("byte-level fixture tokenizer")
    }

    /// SentencePiece-style tokenizer with the Llama decoder chain: `▁` to
    /// space, byte fallback, fuse, and strip of the first leading space.
    /// Returns the id of `<0x00>`.
    fn sentencepiece_tokenizer() -> (HfTokenizer, u32) {
        let mut pieces = vec![
            "<unk>".to_string(),
            "▁hello".into(),
            "▁world".into(),
            "▁".into(),
            "▁.".into(),
            ".".into(),
            ",".into(),
            "ing".into(),
            "▁go".into(),
            "▁你".into(),
            "好".into(),
            "▁Ça".into(),
            "▁?".into(),
        ];
        pieces.extend((0..=255u8).map(|byte| format!("<0x{byte:02X}>")));
        let vocab = pieces
            .iter()
            .enumerate()
            .map(|(id, piece)| format!("{}: {id}", serde_json::json!(piece)))
            .collect::<Vec<_>>()
            .join(", ");
        let json = format!(
            r#"{{
                "version": "1.0",
                "added_tokens": [],
                "normalizer": null,
                "pre_tokenizer": null,
                "post_processor": null,
                "decoder": {{"type": "Sequence", "decoders": [
                    {{"type": "Replace", "pattern": {{"String": "▁"}}, "content": " "}},
                    {{"type": "ByteFallback"}},
                    {{"type": "Fuse"}},
                    {{"type": "Strip", "content": " ", "start": 1, "stop": 0}}
                ]}},
                "model": {{"type": "WordLevel", "vocab": {{{vocab}}}, "unk_token": "<unk>"}}
            }}"#
        );
        let first_byte = pieces.len() as u32 - 256;
        (
            json.parse().expect("sentencepiece fixture tokenizer"),
            first_byte,
        )
    }

    fn hf_decode(tokenizer: &HfTokenizer) -> impl FnMut(&[u32]) -> Result<String> + '_ {
        |ids| {
            tokenizer
                .decode(ids, true)
                .map_err(|error| Error::TokenizationError(error.to_string()))
        }
    }

    fn random_text(rng: &mut ChaCha8Rng, chars: usize) -> String {
        (0..chars)
            .map(|_| TEXT_CHARS[rng.random_range(0..TEXT_CHARS.len())])
            .collect()
    }

    /// Max tokens one step decodes, from the counter's growth per step.
    fn max_step_work<T: Clone>(
        tokens: &[T],
        mut decode: impl FnMut(&[T]) -> Result<String>,
    ) -> u64 {
        let mut detokenizer = IncrementalDetokenizer::new();
        let mut max_step = 0;
        for end in 1..=tokens.len() {
            let before = detokenizer.decoded_tokens();
            detokenizer.step(&tokens[..end], &mut decode).unwrap();
            max_step = max_step.max(detokenizer.decoded_tokens() - before);
        }
        max_step
    }

    #[test]
    fn byte_level_deltas_concatenate_to_the_full_decode_of_multibyte_text() {
        let tokenizer = byte_level_tokenizer();
        let mut rng = ChaCha8Rng::seed_from_u64(0x8_3);
        for _ in 0..200 {
            let chars = rng.random_range(0..60);
            let text = random_text(&mut rng, chars);
            let ids = tokenizer
                .encode(text.as_str(), false)
                .unwrap()
                .get_ids()
                .to_vec();
            let (deltas, detokenizer) = stream_deltas(&ids, hf_decode(&tokenizer));
            assert_eq!(deltas.concat(), text);
            assert_eq!(hf_decode(&tokenizer)(&ids).unwrap(), text);
            // Every delta is whole characters; nothing waits for the flush.
            assert!(deltas.iter().all(|delta| !delta.contains('\u{FFFD}')));
            assert_eq!(deltas.last().map(String::as_str), Some(""));
            assert_eq!(detokenizer.divergences(), 0);
        }
    }

    #[test]
    fn byte_level_random_ids_including_invalid_utf8_match_the_lossy_full_decode() {
        let tokenizer = byte_level_tokenizer();
        let vocab = tokenizer.get_vocab_size(true) as u32;
        let mut rng = ChaCha8Rng::seed_from_u64(0x8_3_1);
        for _ in 0..300 {
            let ids = (0..rng.random_range(0..80))
                .map(|_| rng.random_range(0..vocab))
                .collect::<Vec<_>>();
            let (deltas, _) = stream_deltas(&ids, hf_decode(&tokenizer));
            assert_eq!(
                deltas.concat(),
                hf_decode(&tokenizer)(&ids).unwrap(),
                "{ids:?}"
            );
        }
    }

    #[test]
    fn sentencepiece_word_boundaries_and_byte_fallback_match_the_full_decode() {
        let (tokenizer, first_byte) = sentencepiece_tokenizer();
        let mut rng = ChaCha8Rng::seed_from_u64(0x8_3_2);
        for _ in 0..300 {
            // Pieces, plus characters outside the vocabulary spelled as
            // byte-fallback tokens the way a SentencePiece model emits them.
            let mut ids = Vec::new();
            for _ in 0..rng.random_range(0..60) {
                if rng.random_bool(0.6) {
                    ids.push(rng.random_range(0..first_byte));
                } else {
                    let ch = TEXT_CHARS[rng.random_range(0..TEXT_CHARS.len())];
                    ids.extend(ch.bytes().map(|byte| first_byte + u32::from(byte)));
                }
            }
            let (deltas, detokenizer) = stream_deltas(&ids, hf_decode(&tokenizer));
            assert_eq!(
                deltas.concat(),
                hf_decode(&tokenizer)(&ids).unwrap(),
                "{ids:?}"
            );
            assert_eq!(detokenizer.divergences(), 0, "{ids:?}");
        }
        // The first piece's leading space is stripped; later ones keep it.
        let hello_world = [1, 2, 4];
        let (deltas, _) = stream_deltas(&hello_world, hf_decode(&tokenizer));
        assert_eq!(deltas, ["hello", " world", " .", ""]);
    }

    #[test]
    fn decode_work_per_token_is_bounded_by_the_window_not_the_sequence() {
        let tokenizer = byte_level_tokenizer();
        let mut rng = ChaCha8Rng::seed_from_u64(0x8_3_3);
        let text = random_text(&mut rng, 4_000);
        let ids = tokenizer
            .encode(text.as_str(), false)
            .unwrap()
            .get_ids()
            .to_vec();
        assert!(ids.len() > 8_000);
        // A 4-byte character spans 4 tokens: the window never needs more
        // than the held-back character plus one emitted character.
        let max_step = max_step_work(&ids, hf_decode(&tokenizer));
        assert!(max_step <= 16, "one step decoded {max_step} tokens");
        let (_, detokenizer) = stream_deltas(&ids, hf_decode(&tokenizer));
        let n = ids.len() as u64;
        assert!(
            detokenizer.decoded_tokens() <= 8 * n,
            "decoded {} tokens for {n}; re-decoding is {}",
            detokenizer.decoded_tokens(),
            n * (n + 1) / 2
        );
    }

    #[test]
    fn rejects_a_sequence_shorter_than_what_it_already_emitted() {
        let mut detokenizer = IncrementalDetokenizer::new();
        let decode = |ids: &[u32]| Ok(ids.iter().map(|id| id.to_string()).collect::<String>());
        assert_eq!(detokenizer.step(&[1, 2], decode).unwrap(), "12");
        assert!(detokenizer.step(&[1], decode).is_err());
        // A repeated call with no new token is a no-op.
        assert_eq!(detokenizer.step(&[1, 2], decode).unwrap(), "");
        assert_eq!(detokenizer.finish(&[1, 2], decode).unwrap(), "");
    }

    #[test]
    fn a_decoder_that_rewrites_earlier_text_is_counted_and_flushed_best_effort() {
        // Not prefix-stable: the text is reversed.
        let decode = |ids: &[u32]| {
            Ok(ids
                .iter()
                .rev()
                .map(|id| id.to_string())
                .collect::<String>())
        };
        let mut detokenizer = IncrementalDetokenizer::new();
        assert_eq!(detokenizer.step(&[1], decode).unwrap(), "1");
        assert_eq!(detokenizer.step(&[1, 2], decode).unwrap(), "");
        assert_eq!(detokenizer.divergences(), 1);
        assert_eq!(detokenizer.finish(&[1, 2], decode).unwrap(), "21");
    }
}
