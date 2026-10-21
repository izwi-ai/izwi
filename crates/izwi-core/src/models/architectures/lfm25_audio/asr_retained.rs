//! Dormant rollback-safe scalar ASR continuation for LFM2.5 Audio.

use std::sync::Arc;

use candle_core::{IndexOp, Tensor};

use crate::backends::state::TensorStateArena;
use crate::error::{Error, Result};
use crate::models::architectures::lfm2::backbone::QuantizedLfm2Backbone;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::detokenize::IncrementalDetokenizer;

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
    pub(crate) finished: bool,
}

#[derive(Debug)]
pub(crate) struct Lfm25AudioAsrPrefillBatch {
    pub(crate) steps: Vec<Lfm25AudioAsrPrefillStep>,
    pub(crate) launch_widths: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Lfm25AudioAsrDecodeStep {
    /// Token appended to main KV in this quantum. `None` is retained only as
    /// a defensive signal; sequence transactions must finish on a prior
    /// authenticated append.
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
    detokenizer: IncrementalDetokenizer,
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
    detokenizer: IncrementalDetokenizer,
    assembled: String,
    token_repetition_loop: bool,
    text_repetition_loop: bool,
    finished: bool,
    stop_reason: Option<Lfm25AudioAsrStopReason>,
}

impl Lfm25AudioAsrRetainedState {
    pub(crate) fn prefill_batch(
        backbone: &QuantizedLfm2Backbone,
        states: &mut [&mut Self],
        caches: &mut [&mut PhysicalPagedKvCache],
        checkpoints: &[&Lfm25AudioAsrQuantumCheckpoint],
        max_tokens: &[usize],
    ) -> Result<Lfm25AudioAsrPrefillBatch> {
        let batch = states.len();
        if batch == 0
            || caches.len() != batch
            || checkpoints.len() != batch
            || max_tokens.len() != batch
        {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio ASR prefill batch rows do not match".into(),
            ));
        }
        // Reuse the span-capable scalar path for the normal singleton batch.
        // It authenticates the same checkpoint and leaves commit/rollback to
        // the caller, while avoiding one backbone launch per prompt token.
        if batch == 1 {
            let step =
                states[0].prefill_step(backbone, caches[0], checkpoints[0], max_tokens[0])?;
            return Ok(Lfm25AudioAsrPrefillBatch {
                steps: vec![step],
                launch_widths: vec![1],
            });
        }
        let mut positions = Vec::with_capacity(batch);
        let mut consumed = Vec::with_capacity(batch);
        let mut inputs = Vec::with_capacity(batch);
        for row in 0..batch {
            states[row].authenticate_step(caches[row], checkpoints[row])?;
            if states[row].finished
                || states[row].pending_token.is_some()
                || states[row].prefill_cursor >= states[row].artifact.prompt_tokens
                || max_tokens[row] == 0
                || backbone.hidden_size() != states[row].artifact.hidden_size()?
            {
                return Err(Error::InvalidInput(
                    "LFM2.5 Audio ASR prefill batch contains an invalid row".into(),
                ));
            }
            let span = max_tokens[row]
                .min(states[row].artifact.prompt_tokens - states[row].prefill_cursor);
            positions.push(states[row].prefill_cursor);
            consumed.push(span);
            inputs.push(
                states[row]
                    .artifact
                    .prompt_slice(states[row].prefill_cursor, span)?,
            );
        }
        let mut shortconv = states
            .iter_mut()
            .map(|state| &mut state.retained.shortconv)
            .collect::<Vec<_>>();
        let output =
            backbone.forward_embeds_retained_batch(&inputs, &positions, &mut shortconv, caches)?;
        drop(shortconv);
        let completing = (0..batch)
            .filter(|row| positions[*row] + consumed[*row] == states[*row].artifact.prompt_tokens)
            .collect::<Vec<_>>();
        let logits = if completing.is_empty() {
            None
        } else {
            let hidden = Tensor::cat(
                &completing
                    .iter()
                    .map(|row| output.last_hidden[*row].clone())
                    .collect::<Vec<_>>(),
                0,
            )?;
            Some(backbone.project_last_hidden(&hidden)?)
        };
        let mut pending = vec![None; batch];
        if let Some(logits) = logits.as_ref() {
            for (logit_row, row) in completing.iter().copied().enumerate() {
                pending[row] = Some(greedy_from_logits(
                    &logits.i(logit_row)?,
                    states[row].vocab_limit,
                )?);
            }
        }
        let mut steps = Vec::with_capacity(batch);
        for row in 0..batch {
            states[row].prefill_cursor += consumed[row];
            let terminal_stop =
                pending[row].is_some_and(|token| is_stop_token(token, &states[row].specials));
            states[row].pending_token = (!terminal_stop).then_some(pending[row]).flatten();
            states[row].finished = terminal_stop;
            states[row].stop_reason = terminal_stop.then_some(Lfm25AudioAsrStopReason::StopToken);
            steps.push(Lfm25AudioAsrPrefillStep {
                consumed_tokens: consumed[row],
                prefill_cursor: states[row].prefill_cursor,
                prompt_tokens: states[row].artifact.prompt_tokens,
                complete: states[row].prefill_cursor == states[row].artifact.prompt_tokens,
                pending_token: states[row].pending_token,
                finished: states[row].finished,
            });
        }
        Ok(Lfm25AudioAsrPrefillBatch {
            steps,
            launch_widths: output.launch_widths,
        })
    }

    pub(crate) fn decode_will_append(&self, _tokenizer: &Lfm25TextTokenizer) -> Result<bool> {
        if self.finished || self.prefill_cursor != self.artifact.prompt_tokens {
            return Ok(false);
        }
        let token = self.pending_token.ok_or_else(|| {
            Error::InferenceError("LFM2.5 Audio ASR decode has no staged token".into())
        })?;
        if is_stop_token(token, &self.specials) {
            return Ok(false);
        }
        Ok(!is_stop_token(token, &self.specials))
    }

    pub(crate) fn decode_append_batch(
        backbone: &QuantizedLfm2Backbone,
        tokenizer: &Lfm25TextTokenizer,
        states: &mut [&mut Self],
        caches: &mut [&mut PhysicalPagedKvCache],
        checkpoints: &[&Lfm25AudioAsrQuantumCheckpoint],
    ) -> Result<Vec<Lfm25AudioAsrDecodeStep>> {
        if states.is_empty() || states.len() != caches.len() || states.len() != checkpoints.len() {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio ASR decode batch rows do not match".into(),
            ));
        }
        let mut tokens = Vec::with_capacity(states.len());
        let mut positions = Vec::with_capacity(states.len());
        let mut next_host = Vec::with_capacity(states.len());
        for row in 0..states.len() {
            states[row].authenticate_step(caches[row], checkpoints[row])?;
            if !states[row].decode_will_append(tokenizer)? {
                return Err(Error::InvalidInput(
                    "LFM2.5 Audio native decode batch contains a zero-append terminal row".into(),
                ));
            }
            let token = states[row].pending_token.expect("append row has a token");
            let mut ids = states[row].generated_ids.clone();
            ids.push(token);
            let token_repetition = has_token_repetition_loop(&ids);
            let text = advance_retained_text(
                &states[row].detokenizer,
                &states[row].assembled,
                &ids,
                token_repetition || ids.len() >= states[row].max_new_tokens,
                |ids| tokenizer.decode_text(ids),
            )?;
            let text_repetition = text.text_repetition;
            let repetition_stop = if text_repetition {
                Some(Lfm25AudioAsrStopReason::TextRepetition)
            } else if token_repetition {
                Some(Lfm25AudioAsrStopReason::TokenRepetition)
            } else {
                None
            };
            tokens.push(token);
            positions.push(states[row].main_position()?);
            next_host.push((ids, text, repetition_stop));
        }
        let mut shortconv = states
            .iter_mut()
            .map(|state| &mut state.retained.shortconv)
            .collect::<Vec<_>>();
        let logits = backbone.forward_token_ids_retained_batch(
            &tokens,
            &positions,
            &mut shortconv,
            caches,
        )?;
        drop(shortconv);
        if logits.dim(0)? != states.len() {
            return Err(Error::InferenceError(
                "LFM2.5 Audio native decode returned the wrong row count".into(),
            ));
        }
        // Finish every fallible per-row calculation before publishing any
        // host state. The caller's armed outer checkpoints can then roll back
        // the already-completed physical cohort without partial host updates.
        let mut post_kernel = Vec::with_capacity(states.len());
        for (row, state) in states.iter().enumerate() {
            let reached_budget = next_host[row].0.len() >= state.max_new_tokens;
            let repetition_stop = next_host[row].2;
            let sampled = if reached_budget || repetition_stop.is_some() {
                None
            } else {
                Some(greedy_from_logits(&logits.i(row)?, state.vocab_limit)?)
            };
            let terminal_stop = sampled.is_some_and(|token| is_stop_token(token, &state.specials));
            let next_pending = (!terminal_stop).then_some(sampled).flatten();
            let appended_count = state.decode_tokens_appended.checked_add(1).ok_or_else(|| {
                Error::InferenceError("LFM2.5 Audio ASR decode cursor overflowed".into())
            })?;
            let stop_reason = if reached_budget {
                Some(Lfm25AudioAsrStopReason::MaxTokens)
            } else {
                repetition_stop.or(terminal_stop.then_some(Lfm25AudioAsrStopReason::StopToken))
            };
            if terminal_stop {
                let (ids, text, _) = &mut next_host[row];
                text.flush(ids, |ids| tokenizer.decode_text(ids))?;
            }
            post_kernel.push((stop_reason, next_pending, appended_count));
        }
        let mut outcomes = Vec::with_capacity(states.len());
        for (row, state) in states.iter_mut().enumerate() {
            let (ids, text, repetition_stop) = std::mem::take(&mut next_host[row]);
            let (stop_reason, next_pending, appended_count) = post_kernel[row];
            let RetainedTextAdvance {
                detokenizer,
                assembled,
                delta,
                ..
            } = text;
            state.decode_tokens_appended = appended_count;
            state.pending_token = next_pending;
            state.generated_ids = ids;
            state.detokenizer = detokenizer;
            state.assembled = assembled;
            state.token_repetition_loop |=
                repetition_stop == Some(Lfm25AudioAsrStopReason::TokenRepetition);
            state.text_repetition_loop |=
                repetition_stop == Some(Lfm25AudioAsrStopReason::TextRepetition);
            state.finished = stop_reason.is_some();
            state.stop_reason = stop_reason;
            outcomes.push(state.decode_outcome(Some(tokens[row]), Some(tokens[row]), delta));
        }
        Ok(outcomes)
    }

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
            detokenizer: IncrementalDetokenizer::new(),
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
        let terminal_stop = pending_token.is_some_and(|token| is_stop_token(token, &self.specials));
        self.pending_token = (!terminal_stop).then_some(pending_token).flatten();
        self.finished = terminal_stop;
        self.stop_reason = terminal_stop.then_some(Lfm25AudioAsrStopReason::StopToken);
        Ok(Lfm25AudioAsrPrefillStep {
            consumed_tokens: consumed,
            prefill_cursor: next_cursor,
            prompt_tokens: self.artifact.prompt_tokens,
            complete: next_cursor == self.artifact.prompt_tokens,
            pending_token: self.pending_token,
            finished: self.finished,
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
            let held_back = self
                .detokenizer
                .finish(&self.generated_ids, |ids| tokenizer.decode_text(ids))?;
            self.assembled.push_str(&held_back);
            self.pending_token = None;
            self.finished = true;
            self.stop_reason = Some(Lfm25AudioAsrStopReason::StopToken);
            return Ok(self.decode_outcome(None, None, held_back));
        }
        let mut generated_ids = self.generated_ids.clone();
        generated_ids.push(appended);
        let token_repetition_loop = has_token_repetition_loop(&generated_ids);
        let mut text = advance_retained_text(
            &self.detokenizer,
            &self.assembled,
            &generated_ids,
            token_repetition_loop || generated_ids.len() >= self.max_new_tokens,
            |ids| tokenizer.decode_text(ids),
        )?;
        let text_repetition_loop = text.text_repetition;
        let repetition_stop = if text_repetition_loop {
            Some(Lfm25AudioAsrStopReason::TextRepetition)
        } else if token_repetition_loop {
            Some(Lfm25AudioAsrStopReason::TokenRepetition)
        } else {
            None
        };
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
        let sampled = if reached_budget || repetition_stop.is_some() {
            None
        } else {
            Some(greedy_from_logits(&logits, self.vocab_limit)?)
        };
        let terminal_stop = sampled.is_some_and(|token| is_stop_token(token, &self.specials));
        let stop_reason = if reached_budget {
            Some(Lfm25AudioAsrStopReason::MaxTokens)
        } else {
            repetition_stop.or(terminal_stop.then_some(Lfm25AudioAsrStopReason::StopToken))
        };
        let next_pending = (!terminal_stop).then_some(sampled).flatten();
        if terminal_stop {
            text.flush(&generated_ids, |ids| tokenizer.decode_text(ids))?;
        }
        let RetainedTextAdvance {
            detokenizer,
            assembled,
            delta,
            ..
        } = text;

        self.decode_tokens_appended = next_appended;
        self.pending_token = next_pending;
        self.generated_ids = generated_ids;
        self.detokenizer = detokenizer;
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
            detokenizer: self.detokenizer.clone(),
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
            || self.detokenizer != host.detokenizer
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
        self.detokenizer.clone_from(&checkpoint.detokenizer);
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

/// Host text after the newest token of `ids`, staged so a failed quantum
/// publishes nothing.
#[derive(Default)]
struct RetainedTextAdvance {
    detokenizer: IncrementalDetokenizer,
    assembled: String,
    delta: String,
    text_repetition: bool,
}

impl RetainedTextAdvance {
    /// Append the text still held back because the decode ends here.
    fn flush(&mut self, ids: &[u32], decode: impl FnMut(&[u32]) -> Result<String>) -> Result<()> {
        if self.text_repetition {
            // The trimmed text already ends before the held-back tail.
            return Ok(());
        }
        let held_back = self.detokenizer.finish(ids, decode)?;
        self.assembled.push_str(&held_back);
        self.delta.push_str(&held_back);
        Ok(())
    }
}

/// Detokenize only the tokens since the last emission and check the
/// repeated-phrase tail. `flush` ends the decode at this token. Re-decoding
/// the whole transcript here made every token O(n).
fn advance_retained_text(
    detokenizer: &IncrementalDetokenizer,
    assembled: &str,
    ids: &[u32],
    flush: bool,
    mut decode: impl FnMut(&[u32]) -> Result<String>,
) -> Result<RetainedTextAdvance> {
    let mut detokenizer = detokenizer.clone();
    let mut delta = detokenizer.step(ids, &mut decode)?;
    if flush {
        delta.push_str(&detokenizer.finish(ids, &mut decode)?);
    }
    let mut candidate = String::with_capacity(assembled.len() + delta.len());
    candidate.push_str(assembled);
    candidate.push_str(&delta);
    if let Some(trimmed) = trim_repeated_phrase_tail(&candidate) {
        return Ok(RetainedTextAdvance {
            detokenizer,
            delta: text_delta(assembled, &trimmed),
            assembled: trimmed,
            text_repetition: true,
        });
    }
    Ok(RetainedTextAdvance {
        detokenizer,
        assembled: candidate,
        delta,
        text_repetition: false,
    })
}

fn valid_commit_progress(
    checkpoint_cursor: usize,
    current_cursor: usize,
    _finished: bool,
    _stop_reason: Option<Lfm25AudioAsrStopReason>,
    _pending_was_stop: bool,
) -> bool {
    current_cursor > checkpoint_cursor
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

    fn byte_decode(ids: &[u32]) -> crate::error::Result<String> {
        let bytes = ids.iter().map(|&id| id as u8).collect::<Vec<_>>();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Feed byte tokens through the retained text advance, flushing at the
    /// last one when `flush_last`.
    fn stream(ids: &[u32], flush_last: bool) -> (Vec<String>, super::RetainedTextAdvance) {
        let mut state = super::RetainedTextAdvance::default();
        let mut deltas = Vec::new();
        for end in 1..=ids.len() {
            let flush = flush_last && end == ids.len();
            state = super::advance_retained_text(
                &state.detokenizer,
                &state.assembled,
                &ids[..end],
                flush,
                byte_decode,
            )
            .unwrap();
            deltas.push(state.delta.clone());
            if state.text_repetition {
                break;
            }
        }
        (deltas, state)
    }

    fn bytes(text: &str) -> Vec<u32> {
        text.bytes().map(u32::from).collect()
    }

    #[test]
    fn retained_text_streams_whole_characters_with_linear_detokenize_work() {
        use rand::{RngExt, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let words = [
            "Größe",
            "你好",
            "😀",
            "—",
            "naïve",
            "text.",
            "über",
            "日本語",
        ];
        let mut rng = ChaCha8Rng::seed_from_u64(0x8_3_7);
        let text = (0..2_000)
            .map(|_| words[rng.random_range(0..words.len())])
            .collect::<Vec<_>>()
            .join(" ");
        let (deltas, state) = stream(&bytes(&text), true);
        assert!(!state.text_repetition);
        assert_eq!(deltas.concat(), text);
        assert_eq!(state.assembled, text);
        assert!(deltas.iter().all(|delta| !delta.contains('\u{FFFD}')));
        let n = text.len() as u64;
        assert!(
            state.detokenizer.decoded_tokens() <= 8 * n,
            "decoded {} tokens for {n}; re-decoding is {}",
            state.detokenizer.decoded_tokens(),
            n * (n + 1) / 2
        );
    }

    #[test]
    fn retained_text_flushes_a_partial_character_and_trims_repeats() {
        // The decode ends two bytes into a 4-byte emoji.
        let ids = &bytes("ok 😀")[..5];
        let (deltas, held) = stream(ids, false);
        assert_eq!(deltas.concat(), "ok ");
        let (deltas, flushed) = stream(ids, true);
        assert_eq!(deltas.concat(), byte_decode(ids).unwrap());
        assert_eq!(flushed.assembled, "ok \u{FFFD}");
        let mut late = held;
        late.flush(ids, byte_decode).unwrap();
        assert_eq!(late.assembled, flushed.assembled);

        let (_, state) = stream(
            &bytes("so it goes so it goes so it goes so it goes so it goes"),
            false,
        );
        assert!(state.text_repetition);
        assert_eq!(state.assembled, "so it goes");
    }

    #[test]
    fn retained_asr_delta_handles_prefix_and_rewrite() {
        assert_eq!(text_delta("hello", "hello world"), " world");
        assert_eq!(text_delta("hello x", "hello y"), "y");
    }

    #[test]
    fn retained_sequence_commit_always_requires_an_authenticated_append() {
        assert!(valid_commit_progress(9, 10, false, None, false));
        assert!(!valid_commit_progress(
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
        assert!(!valid_commit_progress(
            9,
            9,
            true,
            Some(Lfm25AudioAsrStopReason::TokenRepetition),
            false
        ));
        assert!(!valid_commit_progress(9, 8, false, None, false));
    }
}
