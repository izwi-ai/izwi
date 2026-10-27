//! Rollback-safe retained Chatterbox generation, quantum by quantum.
//!
//! A request's T3 sequences live in two engine-owned paged caches (the
//! conditional and, with guidance, the unconditional one). Its host state —
//! prefill cursor, sampled tokens, the sampler's stream and the logits of
//! the last forward — is checkpointed with both caches at the start of each
//! quantum, so a failed or cancelled quantum rolls back to exactly where it
//! began. Prefill may arrive in chunks; each decode step samples one token
//! from the pending logits and runs it through T3 (the last one, a stop,
//! needs no forward). Rows from many requests batch into one T3 call.

use std::sync::Arc;

use candle_core::Tensor;

use super::noise::{NoiseStream, SplitMix64};
use super::sampling;
use super::stream::ChatterboxStream;
use super::t3::{T3ForwardRow, T3Model};
use super::{
    renderable_speech_tokens, streamable_speech_tokens, ChatterboxAcoustic,
    ChatterboxGenerationParams, ChatterboxPreparedVoice, ChatterboxRuntime, ChatterboxStopReason,
};
use crate::error::{Error, Result};
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PhysicalPagedKvCheckpoint};

/// One request, prepared before it reaches the engine: both prefill rows and
/// the voice for the acoustic stage.
pub(crate) struct ChatterboxPreparedArtifact {
    /// [prompt, hidden] each.
    pub(crate) conditional: Tensor,
    pub(crate) unconditional: Option<Tensor>,
    pub(crate) prompt_tokens: usize,
    pub(crate) text_tokens: usize,
    pub(crate) voice: Arc<ChatterboxPreparedVoice>,
    /// Device bytes the prepared rows hold.
    pub(crate) retained_resident_bytes: u64,
}

impl std::fmt::Debug for ChatterboxPreparedArtifact {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ChatterboxPreparedArtifact")
            .field("prompt_tokens", &self.prompt_tokens)
            .field("text_tokens", &self.text_tokens)
            .field("guided", &self.unconditional.is_some())
            .field("retained_resident_bytes", &self.retained_resident_bytes)
            .finish_non_exhaustive()
    }
}

impl ChatterboxRuntime {
    /// Text and voice to prefill rows. `params` decides guidance.
    pub(crate) fn prepare_artifact(
        &self,
        text: &str,
        voice: Arc<ChatterboxPreparedVoice>,
        params: &ChatterboxGenerationParams,
    ) -> Result<ChatterboxPreparedArtifact> {
        params.validate()?;
        let text_ids = self.tokenizer().text_sequence_for(text, params.language)?;
        let conditioning = self.t3().conditioning(&voice.t3, params.exaggeration)?;
        let (conditional, unconditional) =
            self.t3()
                .prefill_embeds(&conditioning, &text_ids, params.sampling.guided())?;
        let prompt_tokens = conditional.dim(0)?;
        let row_bytes = (conditional.elem_count() * conditional.dtype().size_in_bytes()) as u64;
        Ok(ChatterboxPreparedArtifact {
            retained_resident_bytes: row_bytes * if unconditional.is_some() { 2 } else { 1 },
            conditional,
            unconditional,
            prompt_tokens,
            text_tokens: text_ids.len(),
            voice,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChatterboxPhase {
    Prefill,
    Decode,
    Acoustic,
    Finished,
}

#[derive(Clone)]
struct HostCheckpoint {
    prefill_cursor: usize,
    pending: Option<PendingLogits>,
    history_len: usize,
    rng: SplitMix64,
    phase: ChatterboxPhase,
    stop_reason: Option<ChatterboxStopReason>,
}

/// Logits of the last forward, one row per branch.
#[derive(Clone)]
struct PendingLogits {
    conditional: Vec<f32>,
    unconditional: Option<Vec<f32>>,
}

pub(crate) struct ChatterboxQuantumCheckpoint {
    conditional: PhysicalPagedKvCheckpoint,
    unconditional: Option<PhysicalPagedKvCheckpoint>,
    host: HostCheckpoint,
}

pub(crate) struct ChatterboxRetainedState {
    pub(crate) artifact: Arc<ChatterboxPreparedArtifact>,
    pub(crate) params: ChatterboxGenerationParams,
    prefill_cursor: usize,
    pending: Option<PendingLogits>,
    /// The speech start token, then every sampled token.
    history: Vec<u32>,
    rng: SplitMix64,
    phase: ChatterboxPhase,
    stop_reason: Option<ChatterboxStopReason>,
    /// S3Gen and HiFT progress once T3 has stopped, one unit per quantum.
    pub(crate) acoustic: ChatterboxAcoustic,
    /// A streamed request's windows, rendered while T3 runs; `None`
    /// renders the whole utterance once T3 has stopped.
    pub(crate) stream: Option<ChatterboxStream>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChatterboxPrefillStep {
    pub(crate) consumed_tokens: usize,
    pub(crate) complete: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChatterboxDecodeStep {
    /// Speech tokens sampled so far (the stop token included).
    pub(crate) tokens_generated: usize,
    pub(crate) finished: bool,
}

/// One request's state and caches in a batched quantum.
pub(crate) struct ChatterboxStepRow<'a> {
    pub(crate) state: &'a mut ChatterboxRetainedState,
    pub(crate) conditional: &'a mut PhysicalPagedKvCache,
    pub(crate) unconditional: Option<&'a mut PhysicalPagedKvCache>,
}

impl ChatterboxRetainedState {
    pub(crate) fn new(
        artifact: Arc<ChatterboxPreparedArtifact>,
        params: ChatterboxGenerationParams,
        start_speech_token: u32,
        context_limit: usize,
        stream: Option<ChatterboxStream>,
    ) -> Result<Self> {
        params.validate()?;
        if params.sampling.guided() != artifact.unconditional.is_some() {
            return Err(Error::InvalidInput(
                "Chatterbox prepared rows do not match the request's guidance".into(),
            ));
        }
        let required = Self::required_context(artifact.prompt_tokens, params.max_speech_tokens);
        if artifact.prompt_tokens == 0 || required > context_limit {
            return Err(Error::InvalidInput(format!(
                "Chatterbox request needs {required} positions, beyond its {context_limit}-position context"
            )));
        }
        Ok(Self {
            rng: SplitMix64::for_stream(params.seed, NoiseStream::Sampling),
            artifact,
            params,
            prefill_cursor: 0,
            pending: None,
            history: vec![start_speech_token],
            phase: ChatterboxPhase::Prefill,
            stop_reason: None,
            acoustic: ChatterboxAcoustic::Pending,
            stream,
        })
    }

    /// Positions one sequence reaches: the prompt plus every sampled token
    /// but the last.
    pub(crate) fn required_context(prompt_tokens: usize, max_speech_tokens: usize) -> usize {
        prompt_tokens + max_speech_tokens.saturating_sub(1)
    }

    pub(crate) fn phase(&self) -> ChatterboxPhase {
        self.phase
    }

    pub(crate) fn prompt_tokens(&self) -> usize {
        self.artifact.prompt_tokens
    }

    pub(crate) fn prefill_cursor(&self) -> usize {
        self.prefill_cursor
    }

    pub(crate) fn guided(&self) -> bool {
        self.artifact.unconditional.is_some()
    }

    pub(crate) fn tokens_generated(&self) -> usize {
        self.history.len() - 1
    }

    pub(crate) fn stop_reason(&self) -> Option<ChatterboxStopReason> {
        self.stop_reason
    }

    /// Whether T3 has sampled its last token.
    pub(crate) fn stopped(&self) -> bool {
        matches!(
            self.phase,
            ChatterboxPhase::Acoustic | ChatterboxPhase::Finished
        )
    }

    /// The tokens a stream renders, as far as T3 has sampled.
    pub(crate) fn streamable_tokens(&self, model: &T3Model) -> Vec<u32> {
        streamable_speech_tokens(&self.history[1..], model.config())
    }

    /// Positions the caches must hold now.
    fn position(&self) -> usize {
        match self.phase {
            ChatterboxPhase::Prefill => self.prefill_cursor,
            _ => self.artifact.prompt_tokens + self.forwarded_speech_tokens(),
        }
    }

    /// Sampled tokens already run through T3: all but the last, unless the
    /// last one is still to be forwarded (decode keeps it pending).
    fn forwarded_speech_tokens(&self) -> usize {
        match self.phase {
            ChatterboxPhase::Prefill => 0,
            ChatterboxPhase::Decode => self.tokens_generated(),
            ChatterboxPhase::Acoustic | ChatterboxPhase::Finished => {
                self.tokens_generated().saturating_sub(1)
            }
        }
    }

    fn host_checkpoint(&self) -> HostCheckpoint {
        HostCheckpoint {
            prefill_cursor: self.prefill_cursor,
            pending: self.pending.clone(),
            history_len: self.history.len(),
            rng: self.rng.clone(),
            phase: self.phase,
            stop_reason: self.stop_reason,
        }
    }

    fn check_caches(
        &self,
        conditional: &PhysicalPagedKvCache,
        unconditional: Option<&PhysicalPagedKvCache>,
    ) -> Result<()> {
        let position = self.position();
        let matches = conditional.context_len() == position
            && match (self.guided(), unconditional) {
                (true, Some(cache)) => cache.context_len() == position,
                (false, None) => true,
                _ => false,
            };
        if !matches {
            return Err(Error::InvalidInput(
                "Chatterbox retained caches crossed host state".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn begin_quantum(
        &self,
        conditional: &PhysicalPagedKvCache,
        unconditional: Option<&PhysicalPagedKvCache>,
    ) -> Result<ChatterboxQuantumCheckpoint> {
        self.check_caches(conditional, unconditional)?;
        Ok(ChatterboxQuantumCheckpoint {
            conditional: conditional.logical_checkpoint(),
            unconditional: unconditional.map(PhysicalPagedKvCache::logical_checkpoint),
            host: self.host_checkpoint(),
        })
    }

    pub(crate) fn commit_quantum(
        &self,
        conditional: &PhysicalPagedKvCache,
        unconditional: Option<&PhysicalPagedKvCache>,
        checkpoint: &ChatterboxQuantumCheckpoint,
    ) -> Result<()> {
        self.check_caches(conditional, unconditional)?;
        if self.history.len() < checkpoint.host.history_len
            || self.prefill_cursor < checkpoint.host.prefill_cursor
        {
            return Err(Error::InferenceError(
                "Chatterbox quantum moved its state backwards".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn rollback_quantum(
        &mut self,
        conditional: &mut PhysicalPagedKvCache,
        unconditional: Option<&mut PhysicalPagedKvCache>,
        checkpoint: &ChatterboxQuantumCheckpoint,
    ) -> Result<()> {
        conditional.restore_logical_checkpoint(checkpoint.conditional.clone())?;
        match (unconditional, &checkpoint.unconditional) {
            (Some(cache), Some(saved)) => cache.restore_logical_checkpoint(saved.clone())?,
            (None, None) => {}
            _ => {
                return Err(Error::InferenceError(
                    "Chatterbox rollback lost a guidance cache".into(),
                ))
            }
        }
        let host = &checkpoint.host;
        self.prefill_cursor = host.prefill_cursor;
        self.pending = host.pending.clone();
        self.history.truncate(host.history_len);
        self.rng = host.rng.clone();
        self.phase = host.phase;
        self.stop_reason = host.stop_reason;
        Ok(())
    }

    /// The speech tokens S3Gen renders, once T3 has stopped.
    pub(crate) fn renderable_tokens(&self, model: &T3Model) -> Result<Vec<u32>> {
        if !matches!(
            self.phase,
            ChatterboxPhase::Acoustic | ChatterboxPhase::Finished
        ) {
            return Err(Error::InferenceError(
                "Chatterbox speech tokens read before T3 stopped".into(),
            ));
        }
        Ok(renderable_speech_tokens(&self.history[1..], model.config()))
    }

    pub(crate) fn mark_finished(&mut self) -> Result<()> {
        if self.phase != ChatterboxPhase::Acoustic {
            return Err(Error::InferenceError(
                "Chatterbox finished outside its acoustic stage".into(),
            ));
        }
        self.phase = ChatterboxPhase::Finished;
        Ok(())
    }
}

/// Prefill `tokens[i]` more prompt positions of each row, in one T3 call.
pub(crate) fn chatterbox_prefill_batch(
    model: &T3Model,
    rows: &mut [ChatterboxStepRow<'_>],
    tokens: &[usize],
) -> Result<Vec<ChatterboxPrefillStep>> {
    if rows.is_empty() || rows.len() != tokens.len() {
        return Err(Error::InvalidInput(
            "Chatterbox prefill batch rows do not match".into(),
        ));
    }
    let mut forward = Vec::with_capacity(rows.len() * 2);
    let mut steps = Vec::with_capacity(rows.len());
    for (row, &count) in rows.iter_mut().zip(tokens) {
        let state = &*row.state;
        let start = state.prefill_cursor;
        if state.phase != ChatterboxPhase::Prefill
            || count == 0
            || start + count > state.artifact.prompt_tokens
        {
            return Err(Error::InvalidInput(format!(
                "Chatterbox prefill of {count} tokens at {start} exceeds its {}-token prompt",
                state.artifact.prompt_tokens
            )));
        }
        steps.push(ChatterboxPrefillStep {
            consumed_tokens: count,
            complete: start + count == state.artifact.prompt_tokens,
        });
    }
    for (index, row) in rows.iter_mut().enumerate() {
        let count = steps[index].consumed_tokens;
        let start = row.state.prefill_cursor;
        let artifact = &row.state.artifact;
        let conditional = artifact.conditional.narrow(0, start, count)?;
        let unconditional = artifact
            .unconditional
            .as_ref()
            .map(|rows| rows.narrow(0, start, count))
            .transpose()?;
        forward.push(T3ForwardRow {
            embeds: conditional,
            cache: &mut *row.conditional,
        });
        if let Some(embeds) = unconditional {
            let cache = row.unconditional.as_deref_mut().ok_or_else(|| {
                Error::InferenceError("Chatterbox guided row lost its unconditional cache".into())
            })?;
            forward.push(T3ForwardRow { embeds, cache });
        }
    }
    let logits = model.forward(&mut forward)?;
    drop(forward);
    distribute_logits(rows, &logits, |row, index| {
        let step = steps[index];
        row.state.prefill_cursor += step.consumed_tokens;
        if step.complete {
            row.state.phase = ChatterboxPhase::Decode;
        }
        Ok(step.complete)
    })?;
    Ok(steps)
}

/// Copy each row's logits out of a forward's `[rows, vocab]` output, in the
/// order the rows were pushed (conditional, then unconditional when
/// guided), keeping them only where `keep` says the row is now decoding.
fn distribute_logits(
    rows: &mut [ChatterboxStepRow<'_>],
    logits: &Tensor,
    mut keep: impl FnMut(&mut ChatterboxStepRow<'_>, usize) -> Result<bool>,
) -> Result<()> {
    let all = logits.to_vec2::<f32>()?;
    let mut next = 0usize;
    for (index, row) in rows.iter_mut().enumerate() {
        let conditional = all[next].clone();
        next += 1;
        let unconditional = if row.state.guided() {
            next += 1;
            Some(all[next - 1].clone())
        } else {
            None
        };
        if keep(row, index)? {
            row.state.pending = Some(PendingLogits {
                conditional,
                unconditional,
            });
        }
    }
    Ok(())
}

/// One decode step for each row: sample a token from its pending logits,
/// then run every row that has not stopped through T3 in one call.
pub(crate) fn chatterbox_decode_batch(
    model: &T3Model,
    rows: &mut [ChatterboxStepRow<'_>],
) -> Result<Vec<ChatterboxDecodeStep>> {
    if rows.is_empty() {
        return Err(Error::InvalidInput(
            "Chatterbox decode batch is empty".into(),
        ));
    }
    let cfg = model.config();
    let mut sampled = Vec::with_capacity(rows.len());
    for row in rows.iter_mut() {
        let state = &mut *row.state;
        if state.phase != ChatterboxPhase::Decode {
            return Err(Error::InvalidInput(
                "Chatterbox decode step outside its decode phase".into(),
            ));
        }
        let pending = state.pending.take().ok_or_else(|| {
            Error::InferenceError("Chatterbox decode step has no pending logits".into())
        })?;
        let token = sampling::sample_next(
            &pending.conditional,
            pending.unconditional.as_deref(),
            &state.history,
            &state.params.sampling,
            &mut state.rng,
        )?;
        state.history.push(token);
        let stopped = if token == cfg.stop_speech_token {
            Some(ChatterboxStopReason::EndOfSpeech)
        } else if state.tokens_generated() >= state.params.max_speech_tokens {
            Some(ChatterboxStopReason::TokenBudget)
        } else {
            None
        };
        if let Some(reason) = stopped {
            state.stop_reason = Some(reason);
            state.phase = ChatterboxPhase::Acoustic;
        }
        sampled.push(token);
    }

    let continuing = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| row.state.phase == ChatterboxPhase::Decode)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if !continuing.is_empty() {
        let embeds = continuing
            .iter()
            .map(|&index| {
                let position = rows[index].state.tokens_generated();
                model.speech_step_embed(sampled[index], position)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut forward = Vec::with_capacity(continuing.len() * 2);
        let mut cursor = 0usize;
        for (index, row) in rows.iter_mut().enumerate() {
            if continuing.get(cursor) != Some(&index) {
                continue;
            }
            let embed = embeds[cursor].clone();
            cursor += 1;
            forward.push(T3ForwardRow {
                embeds: embed.clone(),
                cache: &mut *row.conditional,
            });
            if row.state.guided() {
                let cache = row.unconditional.as_deref_mut().ok_or_else(|| {
                    Error::InferenceError(
                        "Chatterbox guided row lost its unconditional cache".into(),
                    )
                })?;
                forward.push(T3ForwardRow {
                    embeds: embed,
                    cache,
                });
            }
        }
        let logits = model.forward(&mut forward)?;
        drop(forward);
        let all = logits.to_vec2::<f32>()?;
        let mut next = 0usize;
        for &index in &continuing {
            let state = &mut *rows[index].state;
            let conditional = all[next].clone();
            next += 1;
            let unconditional = if state.guided() {
                next += 1;
                Some(all[next - 1].clone())
            } else {
                None
            };
            state.pending = Some(PendingLogits {
                conditional,
                unconditional,
            });
        }
    }
    Ok(rows
        .iter()
        .map(|row| ChatterboxDecodeStep {
            tokens_generated: row.state.tokens_generated(),
            finished: row.state.phase == ChatterboxPhase::Acoustic,
        })
        .collect())
}
