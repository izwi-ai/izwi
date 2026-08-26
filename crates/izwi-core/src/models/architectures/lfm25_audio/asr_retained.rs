//! Dormant rollback-safe scalar ASR continuation for LFM2.5 Audio.

use std::sync::Arc;

use candle_core::Tensor;

use crate::backends::state::TensorStateArena;
use crate::error::{Error, Result};
use crate::models::architectures::lfm2::backbone::QuantizedLfm2Backbone;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

use super::model::{
    has_token_repetition_loop, is_asr_stop_token as is_stop_token, text_delta,
    trim_repeated_phrase_tail, Lfm25AudioPreparedAsrArtifact,
};
use super::sampling::greedy_from_logits;
use super::state::{Lfm25AudioRetainedCheckpoint, Lfm25AudioRetainedState};
use super::tokenizer::{Lfm25SpecialTokenIds, Lfm25TextTokenizer};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lfm25AudioAsrStopReason {
    StopToken,
    TokenRepetition,
    TextRepetition,
    MaxTokens,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Lfm25AudioAsrPrefillStep {
    pub(crate) consumed_tokens: usize,
    pub(crate) prefill_cursor: usize,
    pub(crate) prompt_tokens: usize,
    pub(crate) complete: bool,
    pub(crate) pending_token: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Lfm25AudioAsrDecodeStep {
    /// Token appended to main KV in this quantum. A terminal stop decision is
    /// a valid zero-append quantum and therefore carries `None`.
    pub(crate) cache_append_token: Option<u32>,
    pub(crate) visible_token: Option<u32>,
    pub(crate) delta: String,
    pub(crate) text: String,
    pub(crate) tokens_generated: usize,
    pub(crate) finished: bool,
    pub(crate) stop_reason: Option<Lfm25AudioAsrStopReason>,
    pub(crate) pending_token: Option<u32>,
}

#[derive(Clone)]
struct Lfm25AudioAsrHostCheckpoint {
    prefill_cursor: usize,
    decode_tokens_appended: usize,
    pending_token: Option<u32>,
    generated_ids: Vec<u32>,
    assembled: String,
    token_repetition_loop: bool,
    text_repetition_loop: bool,
    finished: bool,
    stop_reason: Option<Lfm25AudioAsrStopReason>,
}

pub(crate) struct Lfm25AudioAsrQuantumCheckpoint {
    retained: Lfm25AudioRetainedCheckpoint,
    host: Lfm25AudioAsrHostCheckpoint,
}

pub(crate) struct Lfm25AudioAsrRetainedState {
    artifact: Arc<Lfm25AudioPreparedAsrArtifact>,
    retained: Lfm25AudioRetainedState,
    vocab_limit: usize,
    specials: Lfm25SpecialTokenIds,
    max_new_tokens: usize,
    prefill_cursor: usize,
    decode_tokens_appended: usize,
    pending_token: Option<u32>,
    generated_ids: Vec<u32>,
    assembled: String,
    token_repetition_loop: bool,
    text_repetition_loop: bool,
    finished: bool,
    stop_reason: Option<Lfm25AudioAsrStopReason>,
}

impl Lfm25AudioAsrRetainedState {
    pub(crate) fn new(
        artifact: Arc<Lfm25AudioPreparedAsrArtifact>,
        retained: Lfm25AudioRetainedState,
        expected_model_load_nonce: u64,
        vocab_limit: usize,
        specials: Lfm25SpecialTokenIds,
        requested_max_new_tokens: usize,
        context_limit: usize,
    ) -> Result<Self> {
        if artifact.model_load_nonce() != expected_model_load_nonce
            || vocab_limit == 0
            || retained.main_position() != 0
        {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio retained ASR state has invalid initial model state".into(),
            ));
        }
        let available = context_limit
            .checked_sub(artifact.prompt_tokens)
            .ok_or_else(|| {
                Error::InvalidInput(
                    "LFM2.5 Audio retained ASR prompt exceeds the context limit".into(),
                )
            })?;
        if available == 0 {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio retained ASR prompt leaves no decode capacity".into(),
            ));
        }
        Ok(Self {
            artifact,
            retained,
            vocab_limit,
            specials,
            max_new_tokens: requested_max_new_tokens.max(1).min(available),
            prefill_cursor: 0,
            decode_tokens_appended: 0,
            pending_token: None,
            generated_ids: Vec::new(),
            assembled: String::new(),
            token_repetition_loop: false,
            text_repetition_loop: false,
            finished: false,
            stop_reason: None,
        })
    }

    pub(crate) fn begin_quantum(
        &mut self,
        cache: &PhysicalPagedKvCache,
    ) -> Result<Lfm25AudioAsrQuantumCheckpoint> {
        let host = self.host_checkpoint();
        let retained = self.retained.begin_main_quantum(cache)?;
        Ok(Lfm25AudioAsrQuantumCheckpoint { retained, host })
    }

    pub(crate) fn bind_tensor_sequence(&mut self, sequence: u64) -> Result<()> {
        self.retained.bind_tensor_sequence(sequence)
    }

    pub(crate) fn restore_shortconv(&mut self, arena: &TensorStateArena) -> Result<()> {
        self.retained.restore_shortconv(arena)
    }

    pub(crate) fn stage_shortconv(
        &mut self,
        arena: &TensorStateArena,
        transaction: u64,
    ) -> Result<()> {
        self.retained.stage_shortconv(arena, transaction)
    }

    pub(crate) fn commit_quantum(
        &mut self,
        cache: &PhysicalPagedKvCache,
        checkpoint: &Lfm25AudioAsrQuantumCheckpoint,
    ) -> Result<()> {
        let expected = self
            .prefill_cursor
            .checked_add(self.decode_tokens_appended)
            .ok_or_else(|| Error::InferenceError("LFM2.5 Audio ASR cursor overflowed".into()))?;
        let checkpoint_cursor = checkpoint
            .host
            .prefill_cursor
            .checked_add(checkpoint.host.decode_tokens_appended)
            .ok_or_else(|| {
                Error::InferenceError("LFM2.5 Audio ASR checkpoint cursor overflowed".into())
            })?;
        let pending_was_stop = checkpoint
            .host
            .pending_token
            .is_some_and(|token| is_stop_token(token, &self.specials));
        if cache.context_len() != expected
            || !valid_commit_progress(
                checkpoint_cursor,
                expected,
                self.finished,
                self.stop_reason,
                pending_was_stop,
            )
        {
            return Err(Error::InferenceError(
                "LFM2.5 Audio ASR quantum made no exact retained-cache progress".into(),
            ));
        }
        self.retained
            .commit_quantum(cache, None, &checkpoint.retained)
    }

    pub(crate) fn rollback_quantum(
        &mut self,
        cache: &mut PhysicalPagedKvCache,
        checkpoint: &Lfm25AudioAsrQuantumCheckpoint,
    ) -> Result<()> {
        self.retained
            .rollback_quantum(cache, None, &checkpoint.retained)?;
        self.restore_host(&checkpoint.host);
        Ok(())
    }

    pub(crate) fn prefill_step(
        &mut self,
        backbone: &QuantizedLfm2Backbone,
        cache: &mut PhysicalPagedKvCache,
        checkpoint: &Lfm25AudioAsrQuantumCheckpoint,
        max_tokens: usize,
    ) -> Result<Lfm25AudioAsrPrefillStep> {
        self.authenticate_step(cache, checkpoint)?;
        if self.finished
            || self.pending_token.is_some()
            || self.prefill_cursor >= self.artifact.prompt_tokens
        {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio ASR prefill is already complete".into(),
            ));
        }
        if backbone.hidden_size() != self.artifact.hidden_size()? || max_tokens == 0 {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio ASR prefill quantum has invalid geometry or bound".into(),
            ));
        }
        let remaining = self.artifact.prompt_tokens - self.prefill_cursor;
        let consumed = remaining.min(max_tokens);
        let input = self.artifact.prompt_slice(self.prefill_cursor, consumed)?;
        let hidden = backbone.forward_embeds_retained(
            &input,
            self.prefill_cursor,
            cache,
            &mut self.retained.shortconv,
        )?;
        let next_cursor = self.prefill_cursor + consumed;
        let pending_token = if next_cursor == self.artifact.prompt_tokens {
            let logits = backbone.project_last_hidden(&hidden)?;
            let token = greedy_from_logits(&logits, self.vocab_limit)?;
            Some(token)
        } else {
            None
        };
        self.prefill_cursor = next_cursor;
        self.pending_token = pending_token;
        Ok(Lfm25AudioAsrPrefillStep {
            consumed_tokens: consumed,
            prefill_cursor: next_cursor,
            prompt_tokens: self.artifact.prompt_tokens,
            complete: next_cursor == self.artifact.prompt_tokens,
            pending_token,
        })
    }

    pub(crate) fn decode_step(
        &mut self,
        backbone: &QuantizedLfm2Backbone,
        tokenizer: &Lfm25TextTokenizer,
        cache: &mut PhysicalPagedKvCache,
        checkpoint: &Lfm25AudioAsrQuantumCheckpoint,
    ) -> Result<Lfm25AudioAsrDecodeStep> {
        self.authenticate_step(cache, checkpoint)?;
        if self.finished || self.prefill_cursor != self.artifact.prompt_tokens {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio ASR decode is not active".into(),
            ));
        }
        let appended = self.pending_token.ok_or_else(|| {
            Error::InferenceError("LFM2.5 Audio ASR decode has no staged token".into())
        })?;
        if is_stop_token(appended, &self.specials) {
            self.pending_token = None;
            self.finished = true;
            self.stop_reason = Some(Lfm25AudioAsrStopReason::StopToken);
            return Ok(self.decode_outcome(None, None, String::new()));
        }
        let mut generated_ids = self.generated_ids.clone();
        generated_ids.push(appended);
        let mut assembled = tokenizer.decode_text(&generated_ids)?;
        let token_repetition_loop = has_token_repetition_loop(&generated_ids);
        let text_repetition_loop = if let Some(trimmed) = trim_repeated_phrase_tail(&assembled) {
            assembled = trimmed;
            true
        } else {
            false
        };
        let delta = text_delta(&self.assembled, &assembled);
        let repetition_stop = if text_repetition_loop {
            Some(Lfm25AudioAsrStopReason::TextRepetition)
        } else if token_repetition_loop {
            Some(Lfm25AudioAsrStopReason::TokenRepetition)
        } else {
            None
        };
        if let Some(stop_reason) = repetition_stop {
            self.pending_token = None;
            self.generated_ids = generated_ids;
            self.assembled = assembled;
            self.token_repetition_loop |= token_repetition_loop;
            self.text_repetition_loop |= text_repetition_loop;
            self.finished = true;
            self.stop_reason = Some(stop_reason);
            return Ok(self.decode_outcome(None, Some(appended), delta));
        }
        let position = self
            .artifact
            .prompt_tokens
            .checked_add(self.decode_tokens_appended)
            .ok_or_else(|| Error::InferenceError("LFM2.5 Audio ASR position overflowed".into()))?;
        let token = Tensor::from_vec(vec![appended], (1, 1), self.artifact.device())?;
        let logits = backbone
            .forward_tokens_retained(&token, position, cache, &mut self.retained.shortconv, true)?
            .ok_or_else(|| Error::InferenceError("LFM2.5 Audio ASR logits are absent".into()))?;

        let next_appended = self.decode_tokens_appended.checked_add(1).ok_or_else(|| {
            Error::InferenceError("LFM2.5 Audio ASR decode cursor overflowed".into())
        })?;
        let reached_budget = generated_ids.len() >= self.max_new_tokens;
        let stop_reason = if reached_budget {
            Some(Lfm25AudioAsrStopReason::MaxTokens)
        } else {
            None
        };
        let next_pending = if stop_reason.is_none() {
            Some(greedy_from_logits(&logits, self.vocab_limit)?)
        } else {
            None
        };

        self.decode_tokens_appended = next_appended;
        self.pending_token = next_pending;
        self.generated_ids = generated_ids;
        self.assembled = assembled;
        self.token_repetition_loop |= token_repetition_loop;
        self.text_repetition_loop |= text_repetition_loop;
        self.finished = stop_reason.is_some();
        self.stop_reason = stop_reason;
        Ok(self.decode_outcome(Some(appended), Some(appended), delta))
    }

    pub(crate) const fn prefill_cursor(&self) -> usize {
        self.prefill_cursor
    }

    pub(crate) fn main_position(&self) -> Result<usize> {
        self.prefill_cursor
            .checked_add(self.decode_tokens_appended)
            .ok_or_else(|| Error::InferenceError("LFM2.5 Audio ASR position overflowed".into()))
    }

    pub(crate) fn prompt_tokens(&self) -> usize {
        self.artifact.prompt_tokens
    }

    pub(crate) fn pending_token(&self) -> Option<u32> {
        self.pending_token
    }

    pub(crate) fn text(&self) -> &str {
        self.assembled.trim()
    }

    pub(crate) fn generated_ids(&self) -> &[u32] {
        &self.generated_ids
    }

    pub(crate) const fn finished(&self) -> bool {
        self.finished
    }

    fn host_checkpoint(&self) -> Lfm25AudioAsrHostCheckpoint {
        Lfm25AudioAsrHostCheckpoint {
            prefill_cursor: self.prefill_cursor,
            decode_tokens_appended: self.decode_tokens_appended,
            pending_token: self.pending_token,
            generated_ids: self.generated_ids.clone(),
            assembled: self.assembled.clone(),
            token_repetition_loop: self.token_repetition_loop,
            text_repetition_loop: self.text_repetition_loop,
            finished: self.finished,
            stop_reason: self.stop_reason,
        }
    }

    fn authenticate_step(
        &self,
        cache: &PhysicalPagedKvCache,
        checkpoint: &Lfm25AudioAsrQuantumCheckpoint,
    ) -> Result<()> {
        self.retained
            .authenticate_main_quantum(cache, &checkpoint.retained)?;
        let host = &checkpoint.host;
        if self.prefill_cursor != host.prefill_cursor
            || self.decode_tokens_appended != host.decode_tokens_appended
            || self.pending_token != host.pending_token
            || self.generated_ids != host.generated_ids
            || self.assembled != host.assembled
            || self.token_repetition_loop != host.token_repetition_loop
            || self.text_repetition_loop != host.text_repetition_loop
            || self.finished != host.finished
            || self.stop_reason != host.stop_reason
        {
            return Err(Error::InferenceError(
                "LFM2.5 Audio ASR quantum was already advanced or crossed host state".into(),
            ));
        }
        Ok(())
    }

    fn restore_host(&mut self, checkpoint: &Lfm25AudioAsrHostCheckpoint) {
        self.prefill_cursor = checkpoint.prefill_cursor;
        self.decode_tokens_appended = checkpoint.decode_tokens_appended;
        self.pending_token = checkpoint.pending_token;
        self.generated_ids.clone_from(&checkpoint.generated_ids);
        self.assembled.clone_from(&checkpoint.assembled);
        self.token_repetition_loop = checkpoint.token_repetition_loop;
        self.text_repetition_loop = checkpoint.text_repetition_loop;
        self.finished = checkpoint.finished;
        self.stop_reason = checkpoint.stop_reason;
    }

    fn decode_outcome(
        &self,
        cache_append_token: Option<u32>,
        visible_token: Option<u32>,
        delta: String,
    ) -> Lfm25AudioAsrDecodeStep {
        Lfm25AudioAsrDecodeStep {
            cache_append_token,
            visible_token,
            delta,
            text: self.assembled.trim().to_string(),
            tokens_generated: self.generated_ids.len(),
            finished: self.finished,
            stop_reason: self.stop_reason,
            pending_token: self.pending_token,
        }
    }
}

fn valid_commit_progress(
    checkpoint_cursor: usize,
    current_cursor: usize,
    finished: bool,
    stop_reason: Option<Lfm25AudioAsrStopReason>,
    pending_was_stop: bool,
) -> bool {
    current_cursor > checkpoint_cursor
        || (current_cursor == checkpoint_cursor
            && finished
            && match stop_reason {
                Some(Lfm25AudioAsrStopReason::StopToken) => pending_was_stop,
                Some(
                    Lfm25AudioAsrStopReason::TokenRepetition
                    | Lfm25AudioAsrStopReason::TextRepetition,
                ) => !pending_was_stop,
                _ => false,
            })
}

#[cfg(test)]
mod tests {
    use super::{
        has_token_repetition_loop, text_delta, trim_repeated_phrase_tail, valid_commit_progress,
        Lfm25AudioAsrStopReason,
    };

    #[test]
    fn retained_asr_repetition_rules_match_scalar_thresholds() {
        let mut ids = Vec::new();
        for _ in 0..4 {
            ids.extend(0..8);
        }
        assert!(!has_token_repetition_loop(&ids));
        for _ in 0..2 {
            ids.extend(0..8);
        }
        assert!(has_token_repetition_loop(&ids));
        assert_eq!(
            trim_repeated_phrase_tail("hello world hello world hello world hello world"),
            Some("hello world".into())
        );
    }

    #[test]
    fn retained_asr_delta_handles_prefix_and_rewrite() {
        assert_eq!(text_delta("hello", "hello world"), " world");
        assert_eq!(text_delta("hello x", "hello y"), "y");
    }

    #[test]
    fn terminal_stop_is_the_only_valid_zero_append_decode_quantum() {
        assert!(valid_commit_progress(9, 10, false, None, false));
        assert!(valid_commit_progress(
            9,
            9,
            true,
            Some(Lfm25AudioAsrStopReason::StopToken),
            true
        ));
        assert!(!valid_commit_progress(
            9,
            9,
            true,
            Some(Lfm25AudioAsrStopReason::MaxTokens),
            false
        ));
        assert!(valid_commit_progress(
            9,
            9,
            true,
            Some(Lfm25AudioAsrStopReason::TokenRepetition),
            false
        ));
        assert!(!valid_commit_progress(9, 8, false, None, false));
    }
}
