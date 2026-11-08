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

use candle_core::Tensor;

use super::config::{
    AUDIO_CODE_START, END_OF_SPEECH, HEAD_FIRST_TOKEN, HEAD_ROWS, TOKENS_PER_FRAME,
};
use super::frames::{unpack_frames, SnacCodes};
use super::noise::Maya1Stream;
use super::sampling::{self, HeadHistory, Maya1SamplingParams};
use super::snac::SnacDecoder;
use super::stream::{Maya1AudioStream, Maya1Window, MAYA1_WINDOW_PLAN};
use super::trunk::{Maya1ForwardRow, Maya1Trunk};
use super::{Maya1GenerationParams, Maya1StopReason};
use crate::error::{Error, Result};
use crate::kernels::penalized_topk::{self, CandidateRow};
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

/// What a request's next draw reads, from the forward that ran its last
/// token.
#[derive(Debug, Clone)]
enum Maya1Pending {
    /// The whole head logits.
    Logits(Vec<f32>),
    /// On CUDA, the rows top-k keeps with their tempered scores, from the
    /// row's device candidates ([`sampling::candidate_survivors`]).
    Survivors(Vec<(u32, f32)>),
}

#[derive(Clone)]
struct HostCheckpoint {
    prefill_cursor: usize,
    pending: Option<Maya1Pending>,
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
    /// What the next draw reads from the last forward.
    pending: Option<Maya1Pending>,
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
    prefill_batch(trunk, rows, tokens, device_sampling(trunk))
}

/// One decode step for each row: sample a token from its pending logits,
/// then run every row that has not stopped through the decoder in one call.
pub(crate) fn maya1_decode_batch(
    trunk: &Maya1Trunk,
    rows: &mut [Maya1StepRow<'_>],
) -> Result<Vec<Maya1DecodeStep>> {
    decode_batch(trunk, rows, device_sampling(trunk))
}

/// [`maya1_prefill_batch`], its draws planned from device candidates when
/// `device_candidates`.
fn prefill_batch(
    trunk: &Maya1Trunk,
    rows: &mut [Maya1StepRow<'_>],
    tokens: &[usize],
    device_candidates: bool,
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
    let mut plans = Vec::with_capacity(rows.len());
    let mut forward = Vec::with_capacity(rows.len());
    for ((row, artifact), step) in rows.iter_mut().zip(&artifacts).zip(&steps) {
        let Maya1StepRow { state, cache } = row;
        let state: &Maya1RetainedState = state;
        // The history is empty and nothing sampled: what the first draw
        // reads.
        plans.push(if step.complete {
            plan_row(state, device_candidates)
        } else {
            RowPlan::Skip
        });
        let start = state.prefill_cursor;
        forward.push(Maya1ForwardRow {
            tokens: &artifact.prompt[start..start + step.consumed_tokens],
            cache,
        });
    }
    let pending = forward_pending(trunk, &mut forward, plans)?;
    drop(forward);
    for ((row, step), pending) in rows.iter_mut().zip(&steps).zip(pending) {
        row.state.prefill_cursor += step.consumed_tokens;
        if step.complete {
            row.state.phase = Maya1Phase::Decode;
            row.state.pending = pending;
        }
    }
    Ok(steps)
}

/// Whether a step's draws read device candidates: on CUDA, unless
/// `IZWI_MAYA1_DEVICE_SAMPLING=false` or the kernel's one-time check failed.
fn device_sampling(trunk: &Maya1Trunk) -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    trunk.device().is_cuda()
        && *ENABLED.get_or_init(|| crate::env::flag_or(DEVICE_SAMPLING_ENV, true))
        && penalized_topk::device_path::ready(trunk.device())
}

/// Kill switch for [`device_sampling`].
const DEVICE_SAMPLING_ENV: &str = "IZWI_MAYA1_DEVICE_SAMPLING";

/// How a forward row's logits reach its next draw.
enum RowPlan<'a> {
    /// The row draws nothing next (a prompt chunk before its end).
    Skip,
    /// Its whole head logits.
    Whole,
    /// Its device candidates, `count` of them.
    Candidates {
        row: CandidateRow<'a>,
        count: usize,
        params: Maya1SamplingParams,
    },
}

fn plan_row(state: &Maya1RetainedState, device_candidates: bool) -> RowPlan<'_> {
    let params = state.params.sampling;
    match device_candidates
        .then(|| sampling::device_candidates(&state.history, state.tokens.len(), &params))
        .flatten()
    {
        Some((row, count)) => RowPlan::Candidates { row, count, params },
        None => RowPlan::Whole,
    }
}

/// Run `forward` and turn each row's logits into what its next draw reads
/// (`None` for a skipped row). Candidate rows read back together in one
/// transfer; a row whose candidates cannot prove the draw (non-finite
/// logits, a tie at the cut) or that has no candidates reads its whole
/// logits in a second.
fn forward_pending(
    trunk: &Maya1Trunk,
    forward: &mut [Maya1ForwardRow<'_>],
    plans: Vec<RowPlan<'_>>,
) -> Result<Vec<Option<Maya1Pending>>> {
    let wanted = |plan: &RowPlan<'_>| !matches!(plan, RowPlan::Skip);
    let count = plans
        .iter()
        .filter_map(|plan| match plan {
            RowPlan::Candidates { count, .. } => Some(*count),
            _ => None,
        })
        .max();
    let Some(count) = count else {
        let logits = trunk.forward_batch(forward)?;
        return Ok(plans
            .iter()
            .zip(logits)
            .map(|(plan, logits)| wanted(plan).then_some(Maya1Pending::Logits(logits)))
            .collect());
    };
    let logits = trunk.forward_batch_logits(forward)?;
    let mut pending = vec![None; plans.len()];
    let mut selected = Vec::new();
    let mut candidate_rows = Vec::new();
    for (index, plan) in plans.iter().enumerate() {
        if let RowPlan::Candidates { row, .. } = plan {
            selected.push(index);
            candidate_rows.push(CandidateRow {
                logits_row: index,
                ..row.clone()
            });
        }
    }
    let packed = penalized_topk::candidates_packed(&logits, HEAD_ROWS, &candidate_rows, count)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    for (&index, candidates) in selected.iter().zip(penalized_topk::unpack(&packed, count)) {
        let RowPlan::Candidates { params, .. } = &plans[index] else {
            continue;
        };
        if candidates.non_finite == 0 {
            pending[index] =
                sampling::candidate_survivors(&candidates.values, &candidates.indices, params)
                    .map(Maya1Pending::Survivors);
        }
    }
    let whole = (0..plans.len())
        .filter(|&index| pending[index].is_none() && wanted(&plans[index]))
        .collect::<Vec<_>>();
    penalized_topk::device_path::record(selected.len(), whole.len());
    if !whole.is_empty() {
        let ids = whole.iter().map(|&index| index as u32).collect::<Vec<_>>();
        let ids = Tensor::from_vec(ids, whole.len(), logits.device())?;
        let rows = logits.index_select(&ids, 0)?.to_vec2::<f32>()?;
        for (&index, row) in whole.iter().zip(rows) {
            pending[index] = Some(Maya1Pending::Logits(row));
        }
    }
    Ok(pending)
}

/// [`maya1_decode_batch`], its draws planned from device candidates when
/// `device_candidates`.
fn decode_batch(
    trunk: &Maya1Trunk,
    rows: &mut [Maya1StepRow<'_>],
    device_candidates: bool,
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
        let head_row = match pending {
            Maya1Pending::Logits(logits) => sampling::sample_next(
                &logits,
                &state.history,
                state.tokens.len(),
                &state.params.sampling,
                &mut state.rng,
            )?,
            Maya1Pending::Survivors(survivors) => {
                sampling::sample_survivors(&survivors, &state.params.sampling, &mut state.rng)?
            }
        };
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
        let mut plans = Vec::with_capacity(continuing.len());
        let mut forward = Vec::with_capacity(continuing.len());
        let mut cursor = 0usize;
        for (index, row) in rows.iter_mut().enumerate() {
            if continuing.get(cursor) != Some(&index) {
                continue;
            }
            let Maya1StepRow { state, cache } = row;
            plans.push(plan_row(state, device_candidates));
            forward.push(Maya1ForwardRow {
                tokens: &tokens[cursor],
                cache,
            });
            cursor += 1;
        }
        let pending = forward_pending(trunk, &mut forward, plans)?;
        drop(forward);
        for (&index, pending) in continuing.iter().zip(pending) {
            rows[index].state.pending = pending;
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

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};
    use candle_nn::VarBuilder;

    use super::*;
    use crate::models::architectures::maya1::trunk::tests::{tiny_config, tiny_weights};

    /// Each seed's tokens, its requests prefilled (in two chunks) and
    /// decoded to the end as one batch.
    fn run(trunk: &Maya1Trunk, seeds: &[u64], device_candidates: bool) -> Vec<Vec<u32>> {
        let cfg = trunk.config().clone();
        let prompt = vec![
            128_259u32, 128_000, 40, 7_000, 52_011, 128_009, 128_260, 128_261, 128_257,
        ];
        let artifact = Arc::new(Maya1PreparedArtifact {
            text_tokens: 3,
            prompt: prompt.clone(),
        });
        let mut states = seeds
            .iter()
            .map(|&seed| {
                let params = Maya1GenerationParams {
                    sampling: Default::default(),
                    max_new_tokens: 40,
                    seed,
                };
                Maya1RetainedState::new(artifact.clone(), params, 64).unwrap()
            })
            .collect::<Vec<_>>();
        let mut caches = seeds
            .iter()
            .map(|_| {
                PhysicalPagedKvCache::scratch_for_probe(
                    &Device::Cpu,
                    DType::F32,
                    cfg.num_hidden_layers,
                    cfg.num_key_value_heads,
                    cfg.head_dim,
                    64,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        for span in [5, prompt.len() - 5] {
            let mut rows = states
                .iter_mut()
                .zip(caches.iter_mut())
                .map(|(state, cache)| Maya1StepRow { state, cache })
                .collect::<Vec<_>>();
            let spans = vec![span; rows.len()];
            prefill_batch(trunk, &mut rows, &spans, device_candidates).unwrap();
        }
        loop {
            let mut rows = states
                .iter_mut()
                .zip(caches.iter_mut())
                .filter(|(state, _)| state.phase() == Maya1Phase::Decode)
                .map(|(state, cache)| Maya1StepRow { state, cache })
                .collect::<Vec<_>>();
            if rows.is_empty() {
                break;
            }
            decode_batch(trunk, &mut rows, device_candidates).unwrap();
        }
        states.iter().map(|state| state.tokens().to_vec()).collect()
    }

    /// A request draws the same tokens from device candidates as from its
    /// whole logits, alone and beside others.
    #[test]
    fn device_candidates_decode_the_host_tokens_alone_and_batched() {
        let cfg = tiny_config();
        let vb = VarBuilder::from_tensors(tiny_weights(&cfg), DType::F32, &Device::Cpu);
        let trunk = Maya1Trunk::load(cfg, vb, 64).unwrap();
        let seeds = [3u64, 8, 21];
        let host = seeds
            .iter()
            .map(|&seed| run(&trunk, &[seed], false).remove(0))
            .collect::<Vec<_>>();
        assert!(host.iter().all(|tokens| tokens.len() >= 28));
        assert_ne!(host[0], host[1]);
        assert_eq!(run(&trunk, &seeds, true), host);
        assert_eq!(run(&trunk, &seeds[1..2], true)[0], host[1]);
    }
}
