//! Tokenizer for the Voxtral Realtime model.
//!
//! Voxtral uses Mistral's Tekken tokenizer. The model directory must contain
//! `tekken.json`, which is downloaded alongside `params.json` and
//! `consolidated.safetensors`.

use std::fmt::Display;
use std::path::Path;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use tekken::Tekkenizer;

use crate::error::{Error, Result};

const AUDIO_TOKEN_CANDIDATES: &[&str] = &["[AUDIO]"];
const BEGIN_AUDIO_TOKEN_CANDIDATES: &[&str] = &["[BEGIN_AUDIO]"];
const END_AUDIO_TOKEN_CANDIDATES: &[&str] = &["[END_AUDIO]"];
const UNK_TOKEN_CANDIDATES: &[&str] = &["<unk>", ""];
const BOS_TOKEN_CANDIDATES: &[&str] = &["<s>", "~~"];
const EOS_TOKEN_CANDIDATES: &[&str] = &["</s>", "~~ "];
const PAD_TOKEN_CANDIDATES: &[&str] = &["<pad>", ""];
const TRANSCRIBE_TOKEN_CANDIDATES: &[&str] = &["[TRANSCRIBE]"];
const STREAMING_PAD_TOKEN_CANDIDATES: &[&str] = &["[STREAMING_PAD]"];
const STREAMING_WORD_TOKEN_CANDIDATES: &[&str] = &["[STREAMING_WORD]"];
const TEXT_TO_AUDIO_TOKEN_CANDIDATES: &[&str] = &["[NEXT_AUDIO_TEXT]"];
const AUDIO_TO_TEXT_TOKEN_CANDIDATES: &[&str] = &["[REPEAT_AUDIO_TEXT]"];
const DEFAULT_STREAMING_LEFT_PAD_TOKENS: usize = 32;
pub(crate) const OFFLINE_STREAMING_BUFFER_TOKENS: usize = 10;

/// Special token IDs for Voxtral.
#[derive(Debug, Clone)]
pub struct SpecialTokenIds {
    pub audio: u32,
    pub begin_audio: u32,
    pub end_audio: Option<u32>,
    pub pad: u32,
    pub bos: u32,
    pub eos: u32,
    pub unk: u32,
    pub transcribe: Option<u32>,
    pub streaming_pad: Option<u32>,
    pub streaming_word: Option<u32>,
    pub text_to_audio: Option<u32>,
    pub audio_to_text: Option<u32>,
}

impl SpecialTokenIds {
    fn from_tekkenizer(tokenizer: &Tekkenizer) -> Result<Self> {
        Ok(Self {
            audio: required_control_token(tokenizer, "audio", AUDIO_TOKEN_CANDIDATES)?,
            begin_audio: required_control_token(
                tokenizer,
                "begin_audio",
                BEGIN_AUDIO_TOKEN_CANDIDATES,
            )?,
            end_audio: optional_control_token(tokenizer, END_AUDIO_TOKEN_CANDIDATES),
            pad: required_control_token(tokenizer, "pad", PAD_TOKEN_CANDIDATES)?,
            bos: required_control_token(tokenizer, "bos", BOS_TOKEN_CANDIDATES)?,
            eos: required_control_token(tokenizer, "eos", EOS_TOKEN_CANDIDATES)?,
            unk: required_control_token(tokenizer, "unk", UNK_TOKEN_CANDIDATES)?,
            transcribe: optional_control_token(tokenizer, TRANSCRIBE_TOKEN_CANDIDATES),
            streaming_pad: optional_control_token(tokenizer, STREAMING_PAD_TOKEN_CANDIDATES),
            streaming_word: optional_control_token(tokenizer, STREAMING_WORD_TOKEN_CANDIDATES),
            text_to_audio: optional_control_token(tokenizer, TEXT_TO_AUDIO_TOKEN_CANDIDATES),
            audio_to_text: optional_control_token(tokenizer, AUDIO_TO_TEXT_TOKEN_CANDIDATES),
        })
    }
}

/// Audio configuration for tokenization.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    pub sampling_rate: usize,
    pub frame_rate: f32,
    pub window_size: usize,
    pub hop_length: usize,
    pub num_mel_bins: usize,
    pub n_delay_tokens: usize,
    pub streaming_left_pad_tokens: usize,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sampling_rate: 16_000,
            frame_rate: 12.5,
            window_size: 400,
            hop_length: 160,
            num_mel_bins: 128,
            n_delay_tokens: 0,
            streaming_left_pad_tokens: DEFAULT_STREAMING_LEFT_PAD_TOKENS,
        }
    }
}

impl AudioConfig {
    /// Audio tokens covering `audio_length` samples, as mistral_common's
    /// `num_audio_tokens`: whole mel frames, rounded up to whole tokens. (Its
    /// `ceil(len / hop - 1)` for a partial frame equals `floor(len / hop)`.)
    pub fn num_audio_tokens(&self, audio_length: usize) -> usize {
        let mel_frames = audio_length / self.hop_length.max(1);
        mel_frames.div_ceil(self.audio_length_per_tok().max(1))
    }

    pub fn raw_audio_length_per_tok(&self) -> usize {
        (self.sampling_rate as f32 / self.frame_rate).max(1.0) as usize
    }

    /// Mel frames per audio token (mistral_common `audio_length_per_tok`).
    pub fn audio_length_per_tok(&self) -> usize {
        self.raw_audio_length_per_tok() / self.hop_length.max(1)
    }

    /// Delay tokens for a transcription delay, as mistral_common's
    /// `get_num_delay_tokens`: the delay in samples, `int(ms / 1000 * rate)`,
    /// converted like any audio length. The delay must be a positive whole
    /// number of frames, which mistral_common asserts for streaming configs.
    pub fn delay_tokens_for_ms(&self, transcription_delay_ms: f64) -> Result<usize> {
        let frame_ms = 1000.0 / f64::from(self.frame_rate);
        if !transcription_delay_ms.is_finite()
            || transcription_delay_ms <= 0.0
            || !frame_ms.is_finite()
            || transcription_delay_ms % frame_ms != 0.0
        {
            return Err(Error::ModelLoadError(format!(
                "Voxtral transcription_delay_ms {transcription_delay_ms} is not a positive multiple of the {frame_ms} ms audio frame"
            )));
        }
        let delay_samples = (transcription_delay_ms / 1000.0 * self.sampling_rate as f64) as usize;
        Ok(self.num_audio_tokens(delay_samples))
    }

    pub fn streaming_prompt_pad_tokens(&self) -> usize {
        self.streaming_left_pad_tokens + self.n_delay_tokens
    }

    pub fn offline_right_pad_tokens(&self) -> usize {
        self.n_delay_tokens + 1 + OFFLINE_STREAMING_BUFFER_TOKENS
    }
}

/// Tokenizer for Voxtral Realtime.
pub struct VoxtralTokenizer {
    inner: Tekkenizer,
    bytes: TekkenByteTable,
    specials: SpecialTokenIds,
    audio_config: AudioConfig,
}

impl VoxtralTokenizer {
    pub fn load(model_dir: &Path, audio_config: AudioConfig) -> Result<Self> {
        let tekken_path = model_dir.join("tekken.json");
        let tekken_str = std::fs::read_to_string(&tekken_path)
            .map_err(|err| tokenization_error(format!("read {}", tekken_path.display()), err))?;
        let metadata = TekkenMetadata::parse(&tekken_str)?;
        let audio_config = metadata.merge_audio_config(audio_config)?;
        let bytes = metadata.byte_table()?;
        let inner = Tekkenizer::from_file(&tekken_path)
            .map_err(|err| tokenization_error(format!("load {}", tekken_path.display()), err))?;
        Self::from_parts(inner, bytes, audio_config)
    }

    fn from_parts(
        inner: Tekkenizer,
        bytes: TekkenByteTable,
        audio_config: AudioConfig,
    ) -> Result<Self> {
        if bytes.num_special_tokens as usize != inner.num_special_tokens()
            || bytes.len() + inner.num_special_tokens() != inner.vocab_size()
        {
            return Err(Error::TokenizationError(format!(
                "Voxtral Tekken byte table ({} specials + {} text tokens) disagrees with the tokenizer ({} specials, vocabulary {})",
                bytes.num_special_tokens,
                bytes.len(),
                inner.num_special_tokens(),
                inner.vocab_size()
            )));
        }
        let specials = SpecialTokenIds::from_tekkenizer(&inner)?;
        Ok(Self {
            inner,
            bytes,
            specials,
            audio_config,
        })
    }

    /// Get special token IDs.
    pub fn specials(&self) -> &SpecialTokenIds {
        &self.specials
    }

    /// Get audio configuration.
    pub fn audio_config(&self) -> &AudioConfig {
        &self.audio_config
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Longest UTF-8 text one token can add to a transcript. Decoding is
    /// subadditive across token boundaries, so this bounds any decode.
    pub(crate) fn max_decoded_token_bytes(&self) -> Result<usize> {
        self.bytes.max_decoded_token_bytes().ok_or_else(|| {
            Error::TokenizationError("Voxtral tokenizer has an empty vocabulary".into())
        })
    }

    /// Build the Mistral streaming transcription prefix:
    /// BOS followed by left-padding and delay STREAMING_PAD tokens.
    pub fn build_transcription_prompt(&self) -> Result<Vec<u32>> {
        let streaming_pad = self.specials.streaming_pad.ok_or_else(|| {
            Error::TokenizationError(
                "Voxtral streaming transcription requires [STREAMING_PAD]".to_string(),
            )
        })?;
        let mut tokens = Vec::with_capacity(self.audio_config.streaming_prompt_pad_tokens() + 1);
        tokens.push(self.specials.bos);
        tokens.extend(std::iter::repeat_n(
            streaming_pad,
            self.audio_config.streaming_prompt_pad_tokens(),
        ));
        Ok(tokens)
    }

    /// Text of `tokens` up to the last complete UTF-8 character, skipping
    /// special ids as HF's `skip_special_tokens` does. Bytes of a character
    /// whose tokens are still arriving are held back; an invalid sequence
    /// inside the text becomes U+FFFD, like tiktoken's `errors="replace"`.
    /// Stateless, so a rollback only has to truncate `tokens`.
    pub fn decode_text_prefix(&self, tokens: &[u32]) -> Result<String> {
        Ok(utf8_prefix_lossy(&self.bytes.text_bytes(tokens)?))
    }

    /// Final text of `tokens`: every byte decoded, with an incomplete
    /// trailing character replaced by U+FFFD.
    pub fn decode_text_final(&self, tokens: &[u32]) -> Result<String> {
        Ok(String::from_utf8_lossy(&self.bytes.text_bytes(tokens)?).into_owned())
    }
}

/// Decode `bytes` up to the last complete character. An incomplete trailing
/// sequence is held back; each maximal invalid subpart elsewhere becomes one
/// U+FFFD, the same substitution `String::from_utf8_lossy` makes, so the
/// result is always a prefix of the lossy decode of the same bytes.
fn utf8_prefix_lossy(bytes: &[u8]) -> String {
    let mut text = String::with_capacity(bytes.len());
    let mut rest = bytes;
    loop {
        match std::str::from_utf8(rest) {
            Ok(valid) => {
                text.push_str(valid);
                return text;
            }
            Err(error) => {
                let (valid, after) = rest.split_at(error.valid_up_to());
                text.push_str(std::str::from_utf8(valid).unwrap_or_default());
                let Some(invalid) = error.error_len() else {
                    // Unexpected end of input: wait for the next token.
                    return text;
                };
                text.push(char::REPLACEMENT_CHARACTER);
                rest = &after[invalid..];
            }
        }
    }
}

/// Raw bytes of every Tekken text token. Token `t >= num_special_tokens` is
/// rank `t - num_special_tokens` of `tekken.json`'s `vocab`, whose
/// `token_bytes` are base64. Tekken pieces are bytes, not characters: 1,435
/// of the published model's 130,072 text tokens are not valid UTF-8 alone.
#[derive(Debug, Clone)]
struct TekkenByteTable {
    num_special_tokens: u32,
    /// `offsets[rank]..offsets[rank + 1]` indexes `bytes`.
    offsets: Vec<usize>,
    bytes: Vec<u8>,
}

impl TekkenByteTable {
    /// Build from the first `text_tokens` vocab entries, which must hold
    /// ranks `0..text_tokens` in order (tekken truncates the same way).
    fn new(
        vocab: &[TekkenVocabEntry],
        num_special_tokens: usize,
        text_tokens: usize,
    ) -> Result<Self> {
        let num_special_tokens = u32::try_from(num_special_tokens).map_err(|_| {
            Error::TokenizationError("Voxtral Tekken special-token count exceeds u32".into())
        })?;
        if vocab.len() < text_tokens {
            return Err(Error::TokenizationError(format!(
                "Voxtral tekken.json lists {} vocabulary tokens, fewer than its {text_tokens} text tokens",
                vocab.len()
            )));
        }
        let mut offsets = Vec::with_capacity(text_tokens + 1);
        let mut bytes = Vec::new();
        offsets.push(0);
        for (rank, entry) in vocab.iter().take(text_tokens).enumerate() {
            if entry.rank != rank {
                return Err(Error::TokenizationError(format!(
                    "Voxtral tekken.json vocabulary entry {rank} has rank {}",
                    entry.rank
                )));
            }
            let decoded = BASE64
                .decode(&entry.token_bytes)
                .map_err(|err| tokenization_error(format!("decode Tekken token {rank}"), err))?;
            bytes.extend_from_slice(&decoded);
            offsets.push(bytes.len());
        }
        Ok(Self {
            num_special_tokens,
            offsets,
            bytes,
        })
    }

    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    fn rank_bytes(&self, rank: usize) -> Option<&[u8]> {
        let start = *self.offsets.get(rank)?;
        let end = *self.offsets.get(rank + 1)?;
        self.bytes.get(start..end)
    }

    /// Concatenated bytes of the text tokens in `tokens`; special ids are
    /// skipped.
    fn text_bytes(&self, tokens: &[u32]) -> Result<Vec<u8>> {
        let mut text = Vec::with_capacity(tokens.len() * 4);
        for &token in tokens {
            let Some(rank) = token.checked_sub(self.num_special_tokens) else {
                continue;
            };
            let piece = self.rank_bytes(rank as usize).ok_or_else(|| {
                Error::TokenizationError(format!(
                    "Voxtral token {token} is outside the Tekken vocabulary"
                ))
            })?;
            text.extend_from_slice(piece);
        }
        Ok(text)
    }

    fn max_decoded_token_bytes(&self) -> Option<usize> {
        (0..self.len())
            .filter_map(|rank| self.rank_bytes(rank))
            .map(|piece| String::from_utf8_lossy(piece).len())
            .max()
    }
}

/// The `tekken.json` fields Voxtral reads itself; the Tekkenizer reads the
/// rest.
#[derive(Debug, Default, serde::Deserialize)]
pub(super) struct TekkenMetadata {
    #[serde(default)]
    config: Option<TekkenVocabularyConfig>,
    #[serde(default)]
    vocab: Vec<TekkenVocabEntry>,
    #[serde(default)]
    audio: Option<TekkenAudioConfig>,
}

#[derive(Debug, serde::Deserialize)]
struct TekkenVocabularyConfig {
    default_vocab_size: usize,
    default_num_special_tokens: usize,
}

#[derive(Debug, serde::Deserialize)]
struct TekkenVocabEntry {
    rank: usize,
    token_bytes: String,
}

#[derive(Debug, Default, serde::Deserialize)]
struct TekkenAudioConfig {
    transcription_delay_ms: Option<f64>,
    streaming_n_left_pad_tokens: Option<usize>,
}

impl TekkenMetadata {
    pub(super) fn parse(tekken_json: &str) -> Result<Self> {
        serde_json::from_str(tekken_json)
            .map_err(|err| tokenization_error("parse Tekken metadata", err))
    }

    /// Apply the streaming audio metadata: the left-pad token count and the
    /// delay tokens derived from `transcription_delay_ms`.
    pub(super) fn merge_audio_config(&self, mut audio_config: AudioConfig) -> Result<AudioConfig> {
        let Some(audio) = self.audio.as_ref() else {
            return Ok(audio_config);
        };
        if let Some(left_pad) = audio.streaming_n_left_pad_tokens {
            audio_config.streaming_left_pad_tokens = left_pad;
        }
        if let Some(delay_ms) = audio.transcription_delay_ms {
            audio_config.n_delay_tokens = audio_config.delay_tokens_for_ms(delay_ms)?;
        }
        Ok(audio_config)
    }

    fn byte_table(&self) -> Result<TekkenByteTable> {
        let config = self.config.as_ref().ok_or_else(|| {
            Error::TokenizationError("Voxtral tekken.json has no vocabulary config".into())
        })?;
        let text_tokens = config
            .default_vocab_size
            .checked_sub(config.default_num_special_tokens)
            .ok_or_else(|| {
                Error::TokenizationError(
                    "Voxtral tekken.json has more special tokens than vocabulary".into(),
                )
            })?;
        TekkenByteTable::new(&self.vocab, config.default_num_special_tokens, text_tokens)
    }
}

fn required_control_token(tokenizer: &Tekkenizer, name: &str, candidates: &[&str]) -> Result<u32> {
    for candidate in candidates {
        if let Ok(token_id) = tokenizer.get_control_token(candidate) {
            return Ok(token_id);
        }
    }

    Err(Error::TokenizationError(format!(
        "Voxtral Tekken tokenizer is missing required `{name}` control token; tried {}",
        candidates.join(", ")
    )))
}

fn optional_control_token(tokenizer: &Tekkenizer, candidates: &[&str]) -> Option<u32> {
    candidates
        .iter()
        .find_map(|candidate| tokenizer.get_control_token(candidate).ok())
}

fn tokenization_error(context: impl Display, err: impl Display) -> Error {
    Error::TokenizationError(format!("{context}: {err}"))
}

/// Merge `tekken.json` audio metadata into `audio_config`, as `load` does.
#[cfg(test)]
pub(super) fn merge_tekken_audio_metadata(
    audio_config: AudioConfig,
    tekken_json: &str,
) -> Result<AudioConfig> {
    TekkenMetadata::parse(tekken_json)?.merge_audio_config(audio_config)
}

#[cfg(test)]
pub(super) fn published_tekken_excerpt() -> String {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/voxtral_realtime_2602/tekken_audio.json");
    std::fs::read_to_string(path).expect("published tekken excerpt")
}

/// A tokenizer with the published special-token ranks, byte tokens and audio
/// metadata. The excerpt holds only the 256 byte tokens, so its vocabulary
/// ends there.
#[cfg(test)]
pub(super) fn published_excerpt_tokenizer() -> VoxtralTokenizer {
    let excerpt = published_tekken_excerpt();
    let metadata = TekkenMetadata::parse(&excerpt).expect("tekken metadata");
    let audio_config = metadata
        .merge_audio_config(AudioConfig::default())
        .expect("audio metadata");
    let data: tekken::config::ModelData = serde_json::from_str(&excerpt).expect("tekken excerpt");
    let num_special_tokens = data.config.default_num_special_tokens;
    let text_tokens = data.vocab.len();
    let bytes = TekkenByteTable::new(&metadata.vocab, num_special_tokens, text_tokens)
        .expect("published byte table");
    let inner = Tekkenizer::new(
        data.vocab,
        &data.special_tokens.expect("published special tokens"),
        &data.config.pattern,
        num_special_tokens + text_tokens,
        num_special_tokens,
        tekken::config::TokenizerVersion::V7,
        None,
    )
    .expect("published Tekkenizer");
    VoxtralTokenizer::from_parts(inner, bytes, audio_config).expect("Voxtral tokenizer")
}

/// Text token ids for raw bytes: published byte token `b` is id 1000 + b.
#[cfg(test)]
pub(super) fn published_byte_ids(bytes: &[u8]) -> Vec<u32> {
    bytes.iter().map(|byte| 1000 + u32::from(*byte)).collect()
}

#[cfg(test)]
mod tests {
    use base64::engine::general_purpose::STANDARD as BASE64;
    use base64::Engine;
    use tekken::config::{TokenInfo, TokenizerVersion};
    use tekken::special_tokens::SpecialTokenInfo;

    use super::*;

    fn token(rank: usize, text: &str) -> TokenInfo {
        TokenInfo {
            rank,
            token_bytes: BASE64.encode(text.as_bytes()),
            token_str: Some(text.to_string()),
        }
    }

    fn byte_token(rank: usize) -> TokenInfo {
        let byte = rank as u8;
        TokenInfo {
            rank,
            token_bytes: BASE64.encode([byte]),
            token_str: None,
        }
    }

    fn special(rank: usize, text: &str) -> SpecialTokenInfo {
        SpecialTokenInfo {
            rank,
            token_str: text.to_string(),
            is_control: true,
        }
    }

    fn vocab_entries(vocab: &[TokenInfo]) -> Vec<TekkenVocabEntry> {
        vocab
            .iter()
            .map(|token| TekkenVocabEntry {
                rank: token.rank,
                token_bytes: token.token_bytes.clone(),
            })
            .collect()
    }

    fn test_tokenizer(audio_config: AudioConfig) -> VoxtralTokenizer {
        let mut vocab = (0..=255).map(byte_token).collect::<Vec<_>>();
        vocab.push(token(256, "hello"));
        vocab.push(token(257, " world"));
        let special_tokens = vec![
            special(258, ""),
            special(259, "~~"),
            special(260, "~~ "),
            special(261, "[AUDIO]"),
            special(262, "[BEGIN_AUDIO]"),
            special(263, "[STREAMING_PAD]"),
            special(264, "[STREAMING_WORD]"),
        ];
        let bytes = TekkenByteTable::new(&vocab_entries(&vocab), special_tokens.len(), vocab.len())
            .expect("test byte table");
        let inner = Tekkenizer::new(
            vocab,
            &special_tokens,
            r"(?s).+",
            265,
            special_tokens.len(),
            TokenizerVersion::V13,
            None,
        )
        .expect("test Tekkenizer");
        VoxtralTokenizer::from_parts(inner, bytes, audio_config).expect("Voxtral tokenizer")
    }

    fn published_tokenizer() -> VoxtralTokenizer {
        published_excerpt_tokenizer()
    }

    /// Prefix decodes after each token, as the decode loop sees them.
    fn prefix_decodes(tokenizer: &VoxtralTokenizer, ids: &[u32]) -> Vec<String> {
        (1..=ids.len())
            .map(|len| {
                tokenizer
                    .decode_text_prefix(&ids[..len])
                    .expect("prefix decode")
            })
            .collect()
    }

    #[test]
    fn derives_special_token_ids_from_tekken() {
        let tokenizer = test_tokenizer(AudioConfig::default());

        assert_eq!(tokenizer.specials().audio, 261);
        assert_eq!(tokenizer.specials().begin_audio, 262);
        assert_eq!(tokenizer.specials().streaming_pad, Some(263));
        assert_eq!(tokenizer.specials().streaming_word, Some(264));
        assert_eq!(tokenizer.specials().end_audio, None);
    }

    #[test]
    fn reads_streaming_left_pad_tokens_from_tekken_audio_metadata() {
        let config = TekkenMetadata::parse(
            r#"{
                "audio": {
                    "streaming_n_left_pad_tokens": 12
                }
            }"#,
        )
        .and_then(|metadata| metadata.merge_audio_config(AudioConfig::default()))
        .expect("audio metadata");

        assert_eq!(config.streaming_left_pad_tokens, 12);
        assert_eq!(
            config.streaming_prompt_pad_tokens(),
            12 + config.n_delay_tokens
        );
    }

    #[test]
    fn transcription_prompt_uses_streaming_pad_schedule() {
        let tokenizer = test_tokenizer(AudioConfig {
            n_delay_tokens: 3,
            streaming_left_pad_tokens: 2,
            ..AudioConfig::default()
        });

        let prompt = tokenizer
            .build_transcription_prompt()
            .expect("transcription prompt");

        assert_eq!(
            prompt,
            vec![
                tokenizer.specials().bos,
                tokenizer.specials().streaming_pad.unwrap(),
                tokenizer.specials().streaming_pad.unwrap(),
                tokenizer.specials().streaming_pad.unwrap(),
                tokenizer.specials().streaming_pad.unwrap(),
                tokenizer.specials().streaming_pad.unwrap(),
            ]
        );
    }

    #[test]
    fn published_transcription_prompt_is_bos_and_38_streaming_pads() {
        let tokenizer = published_tokenizer();
        assert_eq!(tokenizer.specials().bos, 1);
        assert_eq!(tokenizer.specials().eos, 2);
        assert_eq!(tokenizer.specials().streaming_pad, Some(32));
        assert_eq!(tokenizer.specials().streaming_word, Some(33));

        // mistral_common: BOS + (32 left-pad + 6 delay) x [STREAMING_PAD]. The
        // model prefills all 39 positions.
        let prompt = tokenizer.build_transcription_prompt().expect("prompt");
        let mut expected = vec![1];
        expected.extend([32; 38]);
        assert_eq!(prompt, expected);
        assert_eq!(prompt.len(), 39);
    }

    #[test]
    fn published_delay_derives_6_delay_and_17_right_pad_tokens() {
        let config = published_tokenizer().audio_config().clone();
        // 480 ms -> 7680 samples -> 48 mel frames -> 6 tokens of 8 frames.
        assert_eq!(config.n_delay_tokens, 6);
        // delay + 1 + 10 buffer tokens, mistral_common `n_right_pad_tokens`.
        assert_eq!(config.offline_right_pad_tokens(), 17);
        assert_eq!(config.raw_audio_length_per_tok(), 1_280);
        assert_eq!(config.audio_length_per_tok(), 8);

        assert_eq!(config.delay_tokens_for_ms(80.0).unwrap(), 1);
        assert_eq!(config.delay_tokens_for_ms(2_400.0).unwrap(), 30);
        // mistral_common asserts the delay is a whole number of 80 ms frames.
        assert!(config.delay_tokens_for_ms(500.0).is_err());
        assert!(config.delay_tokens_for_ms(0.0).is_err());
        assert!(config.delay_tokens_for_ms(f64::NAN).is_err());
    }

    #[test]
    fn published_byte_table_requires_the_full_vocabulary() {
        let metadata = TekkenMetadata::parse(&published_tekken_excerpt()).expect("metadata");
        // default_vocab_size 131072 - 1000 specials: the excerpt's 256 entries
        // cannot stand in for the published table.
        let error = metadata
            .byte_table()
            .expect_err("excerpt is not a vocabulary");
        assert!(format!("{error}").contains("fewer than its 130072 text tokens"));
    }

    #[test]
    fn decode_skips_special_ids_and_rejects_unknown_ids() {
        let tokenizer = published_tokenizer();
        let mut ids = vec![1];
        ids.extend(published_byte_ids(b"hello"));
        ids.extend([33, 32, 2]);

        assert_eq!(tokenizer.decode_text_prefix(&ids).unwrap(), "hello");
        assert_eq!(tokenizer.decode_text_final(&ids).unwrap(), "hello");
        assert!(tokenizer.decode_text_final(&[1000 + 256]).is_err());
        assert_eq!(tokenizer.max_decoded_token_bytes().unwrap(), 3);
    }

    #[test]
    fn prefix_decode_holds_back_a_character_split_across_tokens() {
        let tokenizer = published_tokenizer();
        // "你" is E4 BD A0, one published byte token per byte.
        let ni = published_byte_ids("你".as_bytes());

        assert_eq!(prefix_decodes(&tokenizer, &ni), ["", "", "你"]);
        assert_eq!(tokenizer.decode_text_final(&ni[..1]).unwrap(), "\u{FFFD}");
        assert_eq!(tokenizer.decode_text_final(&ni[..2]).unwrap(), "\u{FFFD}");
        assert_eq!(tokenizer.decode_text_final(&ni).unwrap(), "你");
    }

    #[test]
    fn prefix_decode_holds_back_a_split_four_byte_emoji() {
        let tokenizer = published_tokenizer();
        let mut ids = published_byte_ids(b"ok ");
        ids.extend(published_byte_ids("😀".as_bytes()));

        assert_eq!(
            prefix_decodes(&tokenizer, &ids),
            ["o", "ok", "ok ", "ok ", "ok ", "ok ", "ok 😀"]
        );
        assert_eq!(
            tokenizer.decode_text_final(&ids[..5]).unwrap(),
            "ok \u{FFFD}"
        );
    }

    #[test]
    fn prefix_decode_replaces_interior_invalid_bytes_and_continues() {
        let tokenizer = published_tokenizer();
        let ids = published_byte_ids(b"a\xFFb");
        assert_eq!(
            prefix_decodes(&tokenizer, &ids),
            ["a", "a\u{FFFD}", "a\u{FFFD}b"]
        );

        // A truncated sequence followed by ASCII is one maximal invalid subpart.
        let ids = published_byte_ids(b"\xE4\xBDx");
        assert_eq!(prefix_decodes(&tokenizer, &ids), ["", "", "\u{FFFD}x"]);
        assert_eq!(tokenizer.decode_text_final(&ids).unwrap(), "\u{FFFD}x");
    }

    #[test]
    fn prefix_decode_deltas_are_monotonic_and_end_at_the_final_text() {
        let tokenizer = published_tokenizer();
        let mut ids = vec![1];
        for piece in [
            "the ".as_bytes(),
            "你好".as_bytes(),
            b"\xFF",
            "é😀".as_bytes(),
            b" end\xF0\x9F",
        ] {
            ids.extend(published_byte_ids(piece));
            ids.push(33);
        }

        let prefixes = prefix_decodes(&tokenizer, &ids);
        for pair in prefixes.windows(2) {
            assert!(
                pair[1].starts_with(&pair[0]),
                "{:?} -> {:?}",
                pair[0],
                pair[1]
            );
        }
        let last = prefixes.last().unwrap();
        let final_text = tokenizer.decode_text_final(&ids).unwrap();
        assert!(final_text.starts_with(last.as_str()));
        assert_eq!(final_text, "the 你好\u{FFFD}é😀 end\u{FFFD}");
        assert_eq!(*last, "the 你好\u{FFFD}é😀 end");
    }
}
