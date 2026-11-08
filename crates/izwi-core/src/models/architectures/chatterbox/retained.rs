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
use crate::kernels::penalized_topk;
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

/// What the next draw reads from the last forward.
#[derive(Clone)]
enum PendingLogits {
    /// The logits, one row per branch.
    Rows {
        conditional: Vec<f32>,
        unconditional: Option<Vec<f32>>,
    },
    /// On CUDA, the tokens the draw can pick with their processed logits,
    /// from the row's device candidates ([`sampling::candidate_survivors`]).
    Survivors(Vec<(u32, f32)>),
}

/// One request's share of a T3 forward's logits, for [`read_pending`].
struct PendingRequest<'a> {
    guided: bool,
    /// Whether the request draws from this forward.
    wanted: bool,
    params: &'a sampling::T3SamplingParams,
    history: &'a [u32],
}

/// Whether draws read device candidates: on CUDA, unless
/// `IZWI_CHATTERBOX_DEVICE_SAMPLING=false` or the kernel's one-time check
/// failed.
fn device_sampling(model: &T3Model) -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    model.device().is_cuda()
        && *ENABLED.get_or_init(|| crate::env::flag_or("IZWI_CHATTERBOX_DEVICE_SAMPLING", true))
        && penalized_topk::device_path::ready(model.device())
}

/// What each request's next draw reads from a forward's `[pushed rows,
/// vocab]` F32 logits (each request's conditional row, then its
/// unconditional row when guided, in `requests` order); `None` where not
/// `wanted`. With `device_candidates`, the requests read back their
/// candidates together in one transfer; a request they cannot serve (min-p
/// off, non-finite logits, an undecidable min-p cut), or every request
/// without, reads its rows whole.
fn read_pending(
    logits: &Tensor,
    requests: &[PendingRequest<'_>],
    device_candidates: bool,
) -> Result<Vec<Option<PendingLogits>>> {
    let vocab = logits.dim(1)?;
    let mut conditional_rows = Vec::with_capacity(requests.len());
    let mut next = 0usize;
    for request in requests {
        conditional_rows.push(next);
        next += if request.guided { 2 } else { 1 };
    }
    let mut pending = vec![None; requests.len()];
    if device_candidates {
        let histories = requests
            .iter()
            .map(|request| {
                let mut tokens = request
                    .history
                    .iter()
                    .copied()
                    .filter(|&token| (token as usize) < vocab)
                    .collect::<Vec<_>>();
                tokens.sort_unstable();
                tokens.dedup();
                tokens
            })
            .collect::<Vec<_>>();
        let mut selected = Vec::new();
        let mut candidate_rows = Vec::new();
        for (index, request) in requests.iter().enumerate() {
            let row = conditional_rows[index];
            let candidates = request
                .wanted
                .then(|| {
                    sampling::device_candidates(
                        request.params,
                        (row, request.guided.then_some(row + 1)),
                        &histories[index],
                    )
                })
                .flatten();
            if let Some(candidates) = candidates {
                selected.push(index);
                candidate_rows.push(candidates);
            }
        }
        if !selected.is_empty() {
            let count = sampling::DEVICE_CANDIDATES;
            let packed = penalized_topk::candidates_packed(logits, vocab, &candidate_rows, count)?
                .flatten_all()?
                .to_vec1::<u32>()?;
            for (&index, candidates) in selected.iter().zip(penalized_topk::unpack(&packed, count))
            {
                if candidates.non_finite == 0 {
                    pending[index] = sampling::candidate_survivors(
                        &candidates.values,
                        &candidates.indices,
                        requests[index].params,
                        vocab,
                    )
                    .map(PendingLogits::Survivors);
                }
            }
        }
    }
    let whole = (0..requests.len())
        .filter(|&index| requests[index].wanted && pending[index].is_none())
        .collect::<Vec<_>>();
    if device_candidates {
        let wanted = requests.iter().filter(|request| request.wanted).count();
        penalized_topk::device_path::record(wanted, whole.len());
    }
    if whole.is_empty() {
        return Ok(pending);
    }
    // Every row as before, or only the rows of the requests that need them.
    let (host, row_of) = if !device_candidates {
        (logits.to_vec2::<f32>()?, conditional_rows.clone())
    } else {
        let mut ids = Vec::new();
        let mut row_of = vec![0; requests.len()];
        for &index in &whole {
            row_of[index] = ids.len();
            ids.push(conditional_rows[index] as u32);
            if requests[index].guided {
                ids.push(conditional_rows[index] as u32 + 1);
            }
        }
        let count = ids.len();
        let ids = Tensor::from_vec(ids, count, logits.device())?;
        (logits.index_select(&ids, 0)?.to_vec2::<f32>()?, row_of)
    };
    for &index in &whole {
        let row = row_of[index];
        pending[index] = Some(PendingLogits::Rows {
            conditional: host[row].clone(),
            unconditional: requests[index].guided.then(|| host[row + 1].clone()),
        });
    }
    Ok(pending)
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
    let pending = {
        let requests = rows
            .iter()
            .zip(&steps)
            .map(|(row, step)| PendingRequest {
                guided: row.state.guided(),
                // The prompt's end: the first draw reads its logits.
                wanted: step.complete,
                params: &row.state.params.sampling,
                history: &row.state.history,
            })
            .collect::<Vec<_>>();
        read_pending(&logits, &requests, device_sampling(model))?
    };
    for ((row, step), pending) in rows.iter_mut().zip(&steps).zip(pending) {
        row.state.prefill_cursor += step.consumed_tokens;
        if step.complete {
            row.state.phase = ChatterboxPhase::Decode;
            row.state.pending = pending;
        }
    }
    Ok(steps)
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
        let token = match pending {
            PendingLogits::Rows {
                conditional,
                unconditional,
            } => sampling::sample_next(
                &conditional,
                unconditional.as_deref(),
                &state.history,
                &state.params.sampling,
                &mut state.rng,
            )?,
            PendingLogits::Survivors(survivors) => {
                sampling::sample_survivors(&survivors, &mut state.rng)?
            }
        };
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
        // Every continuing row's token at its speech position, gathered in
        // one upload and one lookup per table.
        let (tokens, positions): (Vec<u32>, Vec<usize>) = continuing
            .iter()
            .map(|&index| (sampled[index], rows[index].state.tokens_generated()))
            .unzip();
        let embeds = model.speech_step_embeds(&tokens, &positions)?;
        let mut forward = Vec::with_capacity(continuing.len() * 2);
        let mut cursor = 0usize;
        for (index, row) in rows.iter_mut().enumerate() {
            if continuing.get(cursor) != Some(&index) {
                continue;
            }
            let embed = embeds.narrow(0, cursor, 1)?;
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
        let pending = {
            let requests = continuing
                .iter()
                .map(|&index| {
                    let state = &*rows[index].state;
                    PendingRequest {
                        guided: state.guided(),
                        wanted: true,
                        params: &state.params.sampling,
                        history: &state.history,
                    }
                })
                .collect::<Vec<_>>();
            read_pending(&logits, &requests, device_sampling(model))?
        };
        for (&index, pending) in continuing.iter().zip(pending) {
            rows[index].state.pending = pending;
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

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    /// A forward's logits reach each request's draw the same through device
    /// candidates as whole: guided and unguided requests side by side, one
    /// not drawing, one with min-p off (whole rows on either path).
    #[test]
    fn pending_draws_match_with_and_without_device_candidates() {
        const VOCAB: usize = 8_194;
        let mut source = SplitMix64::new(41);
        let base = sampling::T3SamplingParams {
            temperature: 0.8,
            repetition_penalty: 1.2,
            min_p: 0.05,
            top_p: 1.0,
            cfg_weight: 0.5,
        };
        let unguided = sampling::T3SamplingParams {
            cfg_weight: 0.0,
            top_p: 0.9,
            ..base
        };
        let min_p_off = sampling::T3SamplingParams {
            min_p: 0.0,
            ..unguided
        };
        let params = [base, unguided, base, min_p_off];
        let wanted = [true, true, false, true];
        let histories = (0..4)
            .map(|row| {
                (0..40 + row * 30)
                    .map(|_| (source.next_u64() % VOCAB as u64) as u32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let requests = (0..4)
            .map(|index| PendingRequest {
                guided: params[index].guided(),
                wanted: wanted[index],
                params: &params[index],
                history: &histories[index],
            })
            .collect::<Vec<_>>();
        let rows = requests
            .iter()
            .map(|request| if request.guided { 2 } else { 1 })
            .sum::<usize>();
        let logits = Tensor::from_vec(
            source
                .standard_normals(rows * VOCAB)
                .into_iter()
                .map(|v| v * 3.0)
                .collect::<Vec<_>>(),
            (rows, VOCAB),
            &Device::Cpu,
        )
        .unwrap();
        let draw = |pending: &PendingLogits, index: usize| {
            let mut rng = SplitMix64::new(7 + index as u64);
            let token = match pending {
                PendingLogits::Rows {
                    conditional,
                    unconditional,
                } => sampling::sample_next(
                    conditional,
                    unconditional.as_deref(),
                    &histories[index],
                    &params[index],
                    &mut rng,
                )
                .unwrap(),
                PendingLogits::Survivors(survivors) => {
                    sampling::sample_survivors(survivors, &mut rng).unwrap()
                }
            };
            (token, rng.next_u64())
        };
        let host = read_pending(&logits, &requests, false).unwrap();
        let device = read_pending(&logits, &requests, true).unwrap();
        for index in 0..4 {
            assert_eq!(host[index].is_some(), wanted[index]);
            assert_eq!(device[index].is_some(), wanted[index]);
            if let (Some(host), Some(device)) = (&host[index], &device[index]) {
                assert!(matches!(host, PendingLogits::Rows { .. }));
                assert_eq!(
                    matches!(device, PendingLogits::Survivors(_)),
                    index != 3,
                    "request {index}"
                );
                assert_eq!(draw(host, index), draw(device, index), "request {index}");
            }
        }
    }
}
