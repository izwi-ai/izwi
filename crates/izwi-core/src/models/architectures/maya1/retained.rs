//! Rollback-safe retained Maya1 generation, quantum by quantum.
//!
//! A request's decoder sequence lives in one engine-owned paged cache. Its
//! host state (prefill cursor, sampled tokens, the penalty history, the
//! sampler's stream, the logits of the last forward and the frames so far) is
//! checkpointed with the cache at the start of each quantum, so a failed or
//! cancelled quantum rolls back to exactly where it began. Prefill may arrive
//! in chunks; each decode step samples one token from the pending logits and
//! runs it through the decoder (the stop token needs no forward). Decode rows
//! from many requests share decoder calls, each row getting the logits it
//! would alone ([`Maya1Trunk::forward_batch`]). Audio is decoded in windows
//! between quanta ([`super::stream`]); a failed window discards the request.

use std::sync::Arc;

use super::config::{AUDIO_CODE_START, END_OF_SPEECH, HEAD_FIRST_TOKEN, TOKENS_PER_FRAME};
use super::frames::{unpack_frames, SnacCodes};
use super::noise::Maya1Stream;
use super::sampling::{self, HeadHistory};
use super::snac::SnacDecoder;
use super::stream::{Maya1AudioStream, Maya1Window, MAYA1_WINDOW_PLAN};
use super::trunk::{Maya1ForwardRow, Maya1Trunk};
use super::{Maya1GenerationParams, Maya1StopReason};
use crate::error::{Error, Result};
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PhysicalPagedKvCheckpoint};
use crate::models::shared::noise::SplitMix64;

/// One request, prepared before it reaches the engine: its prompt.
#[derive(Debug)]
pub(crate) struct Maya1PreparedArtifact {
    pub(crate) prompt: Vec<u32>,
    pub(crate) text_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Maya1Phase {
    Prefill,
    Decode,
    /// The decoder has stopped; windows remain.
    Audio,
    Finished,
}

#[derive(Clone)]
struct HostCheckpoint {
    prefill_cursor: usize,
    pending: Option<Vec<f32>>,
    tokens: usize,
    frames: usize,
    rng: SplitMix64,
    phase: Maya1Phase,
    stop_reason: Option<Maya1StopReason>,
}

pub(crate) struct Maya1QuantumCheckpoint {
    cache: PhysicalPagedKvCheckpoint,
    host: HostCheckpoint,
}

pub(crate) struct Maya1RetainedState {
    pub(crate) artifact: Arc<Maya1PreparedArtifact>,
    pub(crate) params: Maya1GenerationParams,
    prefill_cursor: usize,
    /// Head logits of the last forward.
    pending: Option<Vec<f32>>,
    /// Every sampled token, the stop token included.
    tokens: Vec<u32>,
    history: HeadHistory,
    rng: SplitMix64,
    phase: Maya1Phase,
    stop_reason: Option<Maya1StopReason>,
    /// The SNAC codes of every whole frame sampled so far.
    codes: SnacCodes,
    pub(crate) audio: Maya1AudioStream,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Maya1PrefillStep {
    pub(crate) consumed_tokens: usize,
    pub(crate) complete: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Maya1DecodeStep {
    pub(crate) tokens_generated: usize,
    pub(crate) finished: bool,
}

/// One request's state and cache in a batched quantum.
pub(crate) struct Maya1StepRow<'a> {
    pub(crate) state: &'a mut Maya1RetainedState,
    pub(crate) cache: &'a mut PhysicalPagedKvCache,
}

impl Maya1RetainedState {
    pub(crate) fn new(
        artifact: Arc<Maya1PreparedArtifact>,
        params: Maya1GenerationParams,
        context_limit: usize,
    ) -> Result<Self> {
        params.validate()?;
        let required = Self::required_context(artifact.prompt.len(), params.max_new_tokens);
        if artifact.prompt.is_empty() || required > context_limit {
            return Err(Error::InvalidInput(format!(
                "Maya1 request needs {required} positions, beyond its {context_limit}-position context"
            )));
        }
        Ok(Self {
            rng: SplitMix64::for_stream(params.seed, Maya1Stream::Sampling),
            audio: Maya1AudioStream::new(MAYA1_WINDOW_PLAN, params.seed),
            artifact,
            params,
            prefill_cursor: 0,
            pending: None,
            tokens: Vec::new(),
            history: HeadHistory::default(),
            phase: Maya1Phase::Prefill,
            stop_reason: None,
            codes: SnacCodes::default(),
        })
    }

    /// Positions the sequence reaches: the prompt plus every sampled token
    /// but the last.
    pub(crate) fn required_context(prompt_tokens: usize, max_new_tokens: usize) -> usize {
        prompt_tokens + max_new_tokens.saturating_sub(1)
    }

    pub(crate) fn phase(&self) -> Maya1Phase {
        self.phase
    }

    pub(crate) fn prompt_tokens(&self) -> usize {
        self.artifact.prompt.len()
    }

    pub(crate) fn prefill_cursor(&self) -> usize {
        self.prefill_cursor
    }

    pub(crate) fn tokens_generated(&self) -> usize {
        self.tokens.len()
    }

    pub(crate) fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub(crate) fn stop_reason(&self) -> Option<Maya1StopReason> {
        self.stop_reason
    }

    /// Whether the decoder has sampled its last token.
    pub(crate) fn stopped(&self) -> bool {
        matches!(self.phase, Maya1Phase::Audio | Maya1Phase::Finished)
    }

    /// The audio window that can run now, if any.
    pub(crate) fn next_window(&self) -> Option<Maya1Window> {
        self.audio.next_window(self.codes.frames(), self.stopped())
    }

    /// Decode the next window with `snac`; its samples, or `None` when no
    /// window is due.
    pub(crate) fn decode_window(&mut self, snac: &SnacDecoder) -> Result<Option<Vec<f32>>> {
        let Some(window) = self.next_window() else {
            return Ok(None);
        };
        self.audio.decode(snac, &self.codes, window).map(Some)
    }

    /// Positions the cache must hold now.
    fn position(&self) -> usize {
        match self.phase {
            Maya1Phase::Prefill => self.prefill_cursor,
            Maya1Phase::Decode => self.artifact.prompt.len() + self.tokens.len(),
            Maya1Phase::Audio | Maya1Phase::Finished => {
                self.artifact.prompt.len() + self.tokens.len().saturating_sub(1)
            }
        }
    }

    fn host_checkpoint(&self) -> HostCheckpoint {
        HostCheckpoint {
            prefill_cursor: self.prefill_cursor,
            pending: self.pending.clone(),
            tokens: self.tokens.len(),
            frames: self.codes.frames(),
            rng: self.rng.clone(),
            phase: self.phase,
            stop_reason: self.stop_reason,
        }
    }

    fn check_cache(&self, cache: &PhysicalPagedKvCache) -> Result<()> {
        if cache.context_len() != self.position() {
            return Err(Error::InvalidInput(
                "Maya1 retained cache crossed host state".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn begin_quantum(
        &self,
        cache: &PhysicalPagedKvCache,
    ) -> Result<Maya1QuantumCheckpoint> {
        self.check_cache(cache)?;
        Ok(Maya1QuantumCheckpoint {
            cache: cache.logical_checkpoint(),
            host: self.host_checkpoint(),
        })
    }

    pub(crate) fn commit_quantum(
        &self,
        cache: &PhysicalPagedKvCache,
        checkpoint: &Maya1QuantumCheckpoint,
    ) -> Result<()> {
        self.check_cache(cache)?;
        if self.tokens.len() < checkpoint.host.tokens
            || self.prefill_cursor < checkpoint.host.prefill_cursor
        {
            return Err(Error::InferenceError(
                "Maya1 quantum moved its state backwards".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn rollback_quantum(
        &mut self,
        cache: &mut PhysicalPagedKvCache,
        checkpoint: &Maya1QuantumCheckpoint,
    ) -> Result<()> {
        cache.restore_logical_checkpoint(checkpoint.cache.clone())?;
        let host = &checkpoint.host;
        for &token in &self.tokens[host.tokens.min(self.tokens.len())..] {
            if token != END_OF_SPEECH {
                self.history.forget((token - HEAD_FIRST_TOKEN) as usize);
            }
        }
        self.tokens.truncate(host.tokens);
        self.codes = self.codes.slice(0, host.frames);
        self.prefill_cursor = host.prefill_cursor;
        self.pending = host.pending.clone();
        self.rng = host.rng.clone();
        self.phase = host.phase;
        self.stop_reason = host.stop_reason;
        Ok(())
    }

    /// Record a sampled token: the penalty history, and the frame it
    /// completes. The sampler yields only the stop token or audio codes, so
    /// every seventh token before the stop ends a frame.
    fn push_token(&mut self, token: u32) {
        self.tokens.push(token);
        if token == END_OF_SPEECH {
            return;
        }
        self.history.record((token - HEAD_FIRST_TOKEN) as usize);
        if self.tokens.len().is_multiple_of(TOKENS_PER_FRAME) {
            let frame = unpack_frames(&self.tokens[self.tokens.len() - TOKENS_PER_FRAME..]);
            self.codes.coarse.extend(frame.coarse);
            self.codes.medium.extend(frame.medium);
            self.codes.fine.extend(frame.fine);
        }
    }

    pub(crate) fn mark_finished(&mut self) -> Result<()> {
        if self.phase != Maya1Phase::Audio || !self.audio.done() {
            return Err(Error::InferenceError(
                "Maya1 finished before its last audio window".into(),
            ));
        }
        self.phase = Maya1Phase::Finished;
        Ok(())
    }
}

/// Prefill `tokens[i]` more prompt positions of each row, in one call.
pub(crate) fn maya1_prefill_batch(
    trunk: &Maya1Trunk,
    rows: &mut [Maya1StepRow<'_>],
    tokens: &[usize],
) -> Result<Vec<Maya1PrefillStep>> {
    if rows.is_empty() || rows.len() != tokens.len() {
        return Err(Error::InvalidInput(
            "Maya1 prefill batch rows do not match".into(),
        ));
    }
    let mut steps = Vec::with_capacity(rows.len());
    for (row, &count) in rows.iter().zip(tokens) {
        let state = &*row.state;
        let start = state.prefill_cursor;
        if state.phase != Maya1Phase::Prefill
            || count == 0
            || start + count > state.artifact.prompt.len()
        {
            return Err(Error::InvalidInput(format!(
                "Maya1 prefill of {count} tokens at {start} exceeds its {}-token prompt",
                state.artifact.prompt.len()
            )));
        }
        steps.push(Maya1PrefillStep {
            consumed_tokens: count,
            complete: start + count == state.artifact.prompt.len(),
        });
    }
    let artifacts = rows
        .iter()
        .map(|row| row.state.artifact.clone())
        .collect::<Vec<_>>();
    let mut forward = rows
        .iter_mut()
        .zip(&artifacts)
        .zip(&steps)
        .map(|((row, artifact), step)| {
            let start = row.state.prefill_cursor;
            Maya1ForwardRow {
                tokens: &artifact.prompt[start..start + step.consumed_tokens],
                cache: &mut *row.cache,
            }
        })
        .collect::<Vec<_>>();
    let logits = trunk.forward_batch(&mut forward)?;
    drop(forward);
    for ((row, step), logits) in rows.iter_mut().zip(&steps).zip(logits) {
        row.state.prefill_cursor += step.consumed_tokens;
        if step.complete {
            row.state.phase = Maya1Phase::Decode;
            row.state.pending = Some(logits);
        }
    }
    Ok(steps)
}

/// One decode step for each row: sample a token from its pending logits,
/// then run every row that has not stopped through the decoder in one call.
pub(crate) fn maya1_decode_batch(
    trunk: &Maya1Trunk,
    rows: &mut [Maya1StepRow<'_>],
) -> Result<Vec<Maya1DecodeStep>> {
    if rows.is_empty() {
        return Err(Error::InvalidInput("Maya1 decode batch is empty".into()));
    }
    let mut sampled = Vec::with_capacity(rows.len());
    for row in rows.iter_mut() {
        let state = &mut *row.state;
        if state.phase != Maya1Phase::Decode {
            return Err(Error::InvalidInput(
                "Maya1 decode step outside its decode phase".into(),
            ));
        }
        let pending = state.pending.take().ok_or_else(|| {
            Error::InferenceError("Maya1 decode step has no pending logits".into())
        })?;
        let head_row = sampling::sample_next(
            &pending,
            &state.history,
            state.tokens.len(),
            &state.params.sampling,
            &mut state.rng,
        )?;
        let token = HEAD_FIRST_TOKEN + head_row as u32;
        debug_assert!(token == END_OF_SPEECH || token >= AUDIO_CODE_START);
        state.push_token(token);
        let stopped = if token == END_OF_SPEECH {
            Some(Maya1StopReason::EndOfSpeech)
        } else if state.tokens.len() >= state.params.max_new_tokens {
            Some(Maya1StopReason::TokenBudget)
        } else {
            None
        };
        if let Some(reason) = stopped {
            state.stop_reason = Some(reason);
            state.phase = Maya1Phase::Audio;
        }
        sampled.push(token);
    }

    let continuing = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row.state.phase == Maya1Phase::Decode)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if !continuing.is_empty() {
        let tokens = continuing
            .iter()
            .map(|&index| [sampled[index]])
            .collect::<Vec<_>>();
        let mut forward = Vec::with_capacity(continuing.len());
        let mut cursor = 0usize;
        for (index, row) in rows.iter_mut().enumerate() {
            if continuing.get(cursor) != Some(&index) {
                continue;
            }
            forward.push(Maya1ForwardRow {
                tokens: &tokens[cursor],
                cache: &mut *row.cache,
            });
            cursor += 1;
        }
        let logits = trunk.forward_batch(&mut forward)?;
        drop(forward);
        for (&index, logits) in continuing.iter().zip(logits) {
            rows[index].state.pending = Some(logits);
        }
    }
    Ok(rows
        .iter()
        .map(|row| Maya1DecodeStep {
            tokens_generated: row.state.tokens.len(),
            finished: row.state.phase == Maya1Phase::Audio,
        })
        .collect())
}
