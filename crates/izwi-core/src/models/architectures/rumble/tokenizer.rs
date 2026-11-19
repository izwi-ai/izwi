//! Rumble's tokenizer: the published `tokenizer.json` (a pruned, re-indexed
//! Qwen vocabulary) and the special ids the prompt and decode loop use.
//! Every id is resolved by name and checked against `config.json`.

use std::path::Path;

use crate::error::{Error, Result};
use crate::tokenizer::{IncrementalDecoder, Tokenizer};

use super::config::RumbleConfig;

/// The special ids of one loaded checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RumbleSpecialTokens {
    pub im_start: u32,
    pub im_end: u32,
    pub endoftext: u32,
    pub audio_start: u32,
    pub audio_end: u32,
    pub audio_pad: u32,
    /// Ends the `language {Name}` header of the model's output.
    pub asr_text: u32,
    /// Emitted for audio without speech; not part of a transcript.
    pub non_speech: Option<u32>,
    pub digits: u32,
    pub no_digits: u32,
    pub filler: u32,
    pub no_filler: u32,
    pub verbatim: u32,
    pub no_verbatim: u32,
}

impl RumbleSpecialTokens {
    /// Ids that end generation.
    pub fn stop_ids(&self) -> Vec<u32> {
        vec![self.im_end, self.endoftext]
    }

    /// Ids that never contribute transcript text.
    pub fn is_control(&self, id: u32) -> bool {
        id == self.im_start
            || id == self.im_end
            || id == self.endoftext
            || id == self.audio_start
            || id == self.audio_end
            || id == self.audio_pad
            || id == self.asr_text
            || self.non_speech == Some(id)
            || [
                self.digits,
                self.no_digits,
                self.filler,
                self.no_filler,
                self.verbatim,
                self.no_verbatim,
            ]
            .contains(&id)
    }
}

pub struct RumbleTokenizer {
    inner: Tokenizer,
    specials: RumbleSpecialTokens,
    vocab_size: usize,
}

impl RumbleTokenizer {
    pub fn load(model_dir: &Path, config: &RumbleConfig) -> Result<Self> {
        let inner = Tokenizer::from_path_requiring_tokenizer_json(model_dir)?;
        Self::from_tokenizer(inner, config)
    }

    pub(crate) fn from_tokenizer(inner: Tokenizer, config: &RumbleConfig) -> Result<Self> {
        let vocab_size = config.text_config.vocab_size;
        let id_for = |token: &str| -> Result<u32> {
            let id = inner.token_to_id(token).ok_or_else(|| {
                Error::TokenizationError(format!("Rumble tokenizer has no {token} token"))
            })?;
            if id as usize >= vocab_size {
                return Err(Error::TokenizationError(format!(
                    "Rumble token {token} has id {id}, outside the {vocab_size}-token vocabulary"
                )));
            }
            Ok(id)
        };
        let specials = RumbleSpecialTokens {
            im_start: id_for("<|im_start|>")?,
            im_end: id_for("<|im_end|>")?,
            endoftext: id_for("<|endoftext|>")?,
            audio_start: id_for("<|audio_start|>")?,
            audio_end: id_for("<|audio_end|>")?,
            audio_pad: id_for("<|audio_pad|>")?,
            asr_text: id_for("<asr_text>")?,
            non_speech: inner
                .token_to_id("<non_speech>")
                .filter(|id| (*id as usize) < vocab_size),
            digits: id_for("<digits>")?,
            no_digits: id_for("<nodigits>")?,
            filler: id_for("<filler>")?,
            no_filler: id_for("<nofiller>")?,
            verbatim: id_for("<verbatim>")?,
            no_verbatim: id_for("<noverbatim>")?,
        };
        for (name, tokenizer_id, config_id) in [
            ("<|audio_pad|>", specials.audio_pad, config.audio_token_id),
            (
                "<|audio_start|>",
                specials.audio_start,
                config.audio_start_token_id,
            ),
            (
                "<|audio_end|>",
                specials.audio_end,
                config.audio_end_token_id,
            ),
        ] {
            if tokenizer_id != config_id {
                return Err(Error::TokenizationError(format!(
                    "Rumble tokenizer maps {name} to {tokenizer_id}, but config.json says {config_id}"
                )));
            }
        }
        let config_stops = config.eos_token_id.ids();
        if !config_stops.is_empty() {
            let mut expected = specials.stop_ids();
            expected.sort_unstable();
            let mut declared = config_stops.clone();
            declared.sort_unstable();
            if expected != declared {
                return Err(Error::TokenizationError(format!(
                    "Rumble config stops on {config_stops:?}, but the tokenizer's stop tokens are {:?}",
                    specials.stop_ids()
                )));
            }
        }
        Ok(Self {
            inner,
            specials,
            vocab_size,
        })
    }

    pub fn specials(&self) -> &RumbleSpecialTokens {
        &self.specials
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Token ids of plain prompt text.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.inner.encode(text)
    }

    /// Text of transcript ids: control ids and ids outside the vocabulary
    /// are dropped first.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let kept = ids
            .iter()
            .copied()
            .filter(|id| (*id as usize) < self.vocab_size && !self.specials.is_control(*id))
            .collect::<Vec<_>>();
        self.inner.decode(&kept)
    }

    /// The next UTF-8-safe text delta for one transcript id.
    pub fn decode_incrementally(
        &self,
        decoder: &mut IncrementalDecoder,
        id: u32,
    ) -> Result<String> {
        if (id as usize) >= self.vocab_size || self.specials.is_control(id) {
            return Ok(String::new());
        }
        self.inner.decode_incrementally(decoder, id)
    }

    pub fn finish_incremental_decode(&self, decoder: &mut IncrementalDecoder) -> Result<String> {
        self.inner.finish_incremental_decode(decoder)
    }
}
