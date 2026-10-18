//! Bounded accelerator-side candidate selection for stochastic sampling.
//!
//! Candle's CUDA and Metal argsort implementations are efficient only for
//! bounded rows. This module sorts fixed-size chunks and recursively merges
//! their exact top-k candidates, so a large vocabulary never becomes one
//! oversized device sort or a full host copy.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::VecDeque;
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::{DType, Device, IndexOp, Tensor, D};

use crate::{Error, Result};

use super::chat::ChatGenerationConfig;

pub(crate) const DEVICE_SORT_CHUNK: usize = 1024;
const MAX_EXACT_F32_INTEGER: usize = 1 << 24;
const DEVICE_SAMPLING_TENSOR_CACHE_ENTRIES: usize = 8;
pub const DEVICE_SAMPLING_CANDIDATE_LIMIT: usize = 256;
/// Backward-compatible name for the candidate bound used by the original
/// CUDA-only entry point.
pub const CUDA_SAMPLING_CANDIDATE_LIMIT: usize = DEVICE_SAMPLING_CANDIDATE_LIMIT;

/// Token histories read by the two sampling penalties.
///
/// The single-history entry points apply both penalties to the same tokens.
/// Callers that follow vLLM's semantics pass the prompt and output to the
/// repetition penalty and only the output to the presence penalty, so a
/// presence penalty does not punish `<|im_end|>` or newlines because the
/// prompt contains them.
#[derive(Debug, Clone, Copy)]
pub struct PenaltyHistory<'a> {
    pub repetition: &'a [u32],
    pub presence: &'a [u32],
}

impl<'a> PenaltyHistory<'a> {
    /// Both penalties read `history` (the single-history contract).
    pub fn uniform(history: &'a [u32]) -> Self {
        Self {
            repetition: history,
            presence: history,
        }
    }

    fn is_uniform(&self) -> bool {
        std::ptr::eq(self.repetition, self.presence)
    }
}

/// DS9.2: per-request constrained-decoding runtime. The mask cache is shared
/// across sampler clones; the machine advances per committed token.
pub struct GrammarRuntime {
    tokenizer: std::sync::Arc<crate::tokenizer::Tokenizer>,
    /// Token ids that must remain sampleable regardless of the grammar
    /// (the family's end-of-sequence tokens and configured stops).
    always_allowed: Vec<u32>,
    surfaces: std::sync::Arc<std::sync::OnceLock<Vec<String>>>,
    masks: super::grammar::JsonGrammarMasks,
    machine: super::grammar::JsonGrammarMachine,
}

impl Clone for GrammarRuntime {
    fn clone(&self) -> Self {
        Self {
            tokenizer: self.tokenizer.clone(),
            always_allowed: self.always_allowed.clone(),
            surfaces: self.surfaces.clone(),
            masks: self.masks.clone(),
            machine: self.machine.clone(),
        }
    }
}

impl GrammarRuntime {
    pub(crate) fn new(
        tokenizer: std::sync::Arc<crate::tokenizer::Tokenizer>,
        always_allowed: Vec<u32>,
    ) -> Self {
        Self {
            tokenizer,
            always_allowed,
            surfaces: std::sync::Arc::new(std::sync::OnceLock::new()),
            masks: super::grammar::JsonGrammarMasks::new(),
            machine: super::grammar::JsonGrammarMachine::new(),
        }
    }

    fn surfaces(&self, vocab_size: usize) -> Result<&[String]> {
        if let Some(surfaces) = self.surfaces.get() {
            if surfaces.len() >= vocab_size {
                return Ok(&surfaces[..vocab_size]);
            }
        }
        let surfaces = (0..vocab_size)
            .map(|id| self.tokenizer.decode(&[id as u32]).unwrap_or_default())
            .collect::<Vec<_>>();
        let _ = self.surfaces.set(surfaces);
        Ok(self
            .surfaces
            .get()
            .expect("surfaces just inserted")
            .as_slice())
    }

    /// Sample one token under the grammar's current mask, then advance the
    /// machine with the token's surface text.
    pub(crate) fn sample_token<R: GrammarRng>(
        &mut self,
        logits: &Tensor,
        vocab_size: usize,
        config: &ChatGenerationConfig,
        history: &[u32],
        rng: &mut R,
    ) -> Result<(u32, Option<RawTokenLogprobs>)> {
        self.sample_token_with_history(
            logits,
            vocab_size,
            config,
            PenaltyHistory::uniform(history),
            rng,
        )
    }

    /// [`Self::sample_token`] with separate penalty histories.
    pub(crate) fn sample_token_with_history<R: GrammarRng>(
        &mut self,
        logits: &Tensor,
        vocab_size: usize,
        config: &ChatGenerationConfig,
        history: PenaltyHistory<'_>,
        rng: &mut R,
    ) -> Result<(u32, Option<RawTokenLogprobs>)> {
        let values = read_f32_values_to_host(&chat_logits_row(logits)?)?;
        self.sample_host_values(values, vocab_size, config, history, rng)
    }

    /// [`Self::sample_token_with_history`] on a logits row already read to
    /// the host.
    fn sample_host_values<R: GrammarRng>(
        &mut self,
        mut values: Vec<f32>,
        vocab_size: usize,
        config: &ChatGenerationConfig,
        history: PenaltyHistory<'_>,
        rng: &mut R,
    ) -> Result<(u32, Option<RawTokenLogprobs>)> {
        let surfaces = self.surfaces(vocab_size)?;
        let mask = self.masks.mask_for(&self.machine, surfaces);
        values.truncate(vocab_size.min(values.len()));
        if values.is_empty() {
            return Err(Error::InvalidInput(
                "chat sampler received no in-vocabulary logits".to_string(),
            ));
        }
        // Raw stats (for logprobs) come from the unmasked row.
        let raw_logprobs = if config.logprobs {
            let (logsumexp, top) = raw_logprobs_stats(&values, config.top_logprobs)?;
            Some((logsumexp, top, values.clone()))
        } else {
            None
        };
        for (index, value) in values.iter_mut().enumerate() {
            let allowed = mask.get(index).copied().unwrap_or(false)
                || self.always_allowed.iter().any(|id| *id as usize == index);
            if !allowed {
                *value = f32::NEG_INFINITY;
            }
        }
        let token = sample_from_host_values(values, config, history, rng)?;
        if config.stop_token_ids.contains(&token) || self.always_allowed.contains(&token) {
            // A stop token ends generation; the machine state stays put.
        } else {
            let surface = surfaces
                .get(token as usize)
                .ok_or_else(|| {
                    Error::InferenceError("sampled token outside the grammar vocab".into())
                })?
                .clone();
            self.machine.feed(&surface).map_err(|_| {
                Error::InferenceError("grammar machine rejected a masked-in token".into())
            })?;
        }
        let logprobs = match raw_logprobs {
            Some((logsumexp, top, raw)) => {
                let chosen_raw = raw[token as usize];
                if !chosen_raw.is_finite() {
                    return Err(Error::InferenceError(
                        "sampled token has a non-finite raw logit".into(),
                    ));
                }
                Some(RawTokenLogprobs {
                    token,
                    logprob: chosen_raw - logsumexp,
                    top,
                })
            }
            None => None,
        };
        Ok((token, logprobs))
    }
}

/// Request-owned chat sampler used after a shared tensor forward. Continuous
/// batching shares logits computation, never sampling policy or RNG state.
#[derive(Clone)]
pub struct ChatSampler {
    config: ChatGenerationConfig,
    history: Vec<u32>,
    track_history: bool,
    rng: SimpleRng,
    grammar: Option<GrammarRuntime>,
}

impl ChatSampler {
    pub fn new(config: ChatGenerationConfig, prompt_history: &[u32]) -> Self {
        let track_history =
            config.repetition_penalty > 1.0 || config.presence_penalty.abs() > f32::EPSILON;
        Self {
            rng: SimpleRng::new(config.seed),
            history: if track_history {
                prompt_history.to_vec()
            } else {
                Vec::new()
            },
            config,
            track_history,
            grammar: None,
        }
    }

    /// DS9.2: constrain this request to emit one valid JSON value
    /// (`response_format: json_object`). `always_allowed` lists token ids
    /// (typically end-of-sequence) that must stay sampleable in every state.
    pub fn with_json_object_constraint(
        mut self,
        tokenizer: std::sync::Arc<crate::tokenizer::Tokenizer>,
        always_allowed: Vec<u32>,
    ) -> Self {
        if self.config.constrain_json_object {
            self.grammar = Some(GrammarRuntime::new(tokenizer, always_allowed));
        }
        self
    }

    pub fn sample(&mut self, logits: &Tensor, vocab_size: usize) -> Result<u32> {
        let (token, _) = self.sample_with_logprobs(logits, vocab_size)?;
        Ok(token)
    }

    /// DS9.3: sample a token and, when the request asked for logprobs,
    /// return its raw-distribution logprob plus the top alternatives.
    pub fn sample_with_logprobs(
        &mut self,
        logits: &Tensor,
        vocab_size: usize,
    ) -> Result<(u32, Option<RawTokenLogprobs>)> {
        if let Some(grammar) = self.grammar.as_mut() {
            let (token, logprobs) = grammar.sample_token(
                logits,
                vocab_size,
                &self.config,
                &self.history,
                &mut self.rng,
            )?;
            if self.track_history {
                self.history.push(token);
            }
            return Ok((token, logprobs));
        }
        let (token, logprobs) = sample_chat_token_and_logprobs(
            logits,
            vocab_size,
            &self.config,
            &self.history,
            &mut self.rng,
        )?;
        if self.track_history {
            self.history.push(token);
        }
        Ok((token, logprobs))
    }

    /// DS9.3: whether this request collects per-token logprobs.
    pub fn wants_logprobs(&self) -> bool {
        self.config.logprobs
    }

    pub fn is_configured_stop(&self, token: u32) -> bool {
        self.config.stop_token_ids.contains(&token)
    }

    #[cfg(test)]
    fn history(&self) -> &[u32] {
        &self.history
    }

    /// How [`sample_chat_rows`] reads row `row` of `logits` (already cut to
    /// the vocabulary): the route [`Self::sample_with_logprobs`] would take.
    fn row_sampling(&self, logits: &Tensor, row: usize, vocab_size: usize) -> Result<RowSampling> {
        if self.grammar.is_some() || self.config.logprobs {
            return Ok(RowSampling::Host);
        }
        if is_deterministic_greedy(&self.config) {
            return Ok(RowSampling::Greedy);
        }
        if !(logits.device().is_cuda() || logits.device().is_metal()) {
            return Ok(RowSampling::Host);
        }
        Ok(pending_sampling_candidates(
            &logits.i(row)?,
            vocab_size,
            self.config.top_k,
            self.config.temperature,
            PenaltyHistory::uniform(&self.history),
            self.config.repetition_penalty,
            self.config.presence_penalty,
            None,
        )?
        .map_or(RowSampling::Host, RowSampling::Candidates))
    }

    /// The draw of a row whose logits are already on the host.
    fn sample_host_row(&mut self, values: Vec<f32>, vocab_size: usize) -> Result<SampledToken> {
        if let Some(grammar) = self.grammar.as_mut() {
            return grammar.sample_host_values(
                values,
                vocab_size,
                &self.config,
                PenaltyHistory::uniform(&self.history),
                &mut self.rng,
            );
        }
        if self.config.logprobs {
            return sample_host_logprobs(
                values,
                vocab_size,
                &self.config,
                &self.history,
                &mut self.rng,
            );
        }
        let token = sample_candidates_or_host(
            None,
            || Ok(values),
            vocab_size,
            &self.config,
            &self.history,
            &mut self.rng,
        )?;
        Ok((token, None))
    }

    fn record(&mut self, token: u32) {
        if self.track_history {
            self.history.push(token);
        }
    }
}

/// A row's next token and, when its request asked for them, the token's raw
/// logprobs.
pub type SampledToken = (u32, Option<RawTokenLogprobs>);

/// How one row of a batched decode step is read back.
enum RowSampling {
    /// Deterministic greedy: the row's argmax and its logit.
    Greedy,
    /// Bounded accelerator candidates.
    Candidates(PendingCandidates),
    /// The whole in-vocabulary row: grammar, logprobs and host sampling.
    Host,
}

/// The next token of every row of one decode step's `[B, V]` logits, each
/// drawn by that row's own [`ChatSampler`] exactly as
/// [`ChatSampler::sample_with_logprobs`] draws it (same arithmetic, same RNG
/// draws), with one device-to-host readback for the whole batch instead of
/// one or two per row. Greedy rows read their argmax and its logit,
/// accelerator rows their bounded candidates, and every other row (grammar,
/// logprobs, CPU sampling) its in-vocabulary logits. A row reads again only
/// where the single-row sampler falls back too: a non-finite greedy pick, or
/// candidates that miss the top-p nucleus.
///
/// Only generic tensor ops run on the device, so the path is the same on
/// CPU, Metal and CUDA.
pub fn sample_chat_rows(
    samplers: &mut [&mut ChatSampler],
    logits: &Tensor,
    vocab_size: usize,
) -> Result<Vec<SampledToken>> {
    sample_chat_rows_guarded(samplers, logits, vocab_size, None)?.ok_or_else(|| {
        Error::InferenceError("unguarded chat sampling reported a tripped guard".into())
    })
}

/// [`sample_chat_rows`] that reads `guard`, one F32 flag, in the same
/// transfer and samples nothing (`None`) when it is zero, so a caller can
/// refuse non-finite logits without another synchronization.
pub(crate) fn sample_chat_rows_guarded(
    samplers: &mut [&mut ChatSampler],
    logits: &Tensor,
    vocab_size: usize,
    guard: Option<&Tensor>,
) -> Result<Option<Vec<SampledToken>>> {
    let (rows, width) = logits.dims2()?;
    if rows == 0 || rows != samplers.len() {
        return Err(Error::InvalidInput(format!(
            "batched chat sampling received {rows} logits rows for {} samplers",
            samplers.len()
        )));
    }
    if vocab_size == 0 {
        return Err(Error::InvalidInput(
            "chat sampler received vocab_size=0".to_string(),
        ));
    }
    let vocab = width.min(vocab_size);
    if vocab == 0 {
        return Err(Error::InvalidInput(
            "chat sampler received no in-vocabulary logits".to_string(),
        ));
    }
    let full_logits = logits;
    let logits = logits.narrow(1, 0, vocab)?;
    if vocab > MAX_EXACT_F32_INTEGER {
        // Token ids would not survive the F32 transfer: sample row by row.
        if let Some(guard) = guard {
            if read_f32_values_to_host(&guard.flatten_all()?)?[0] == 0.0 {
                return Ok(None);
            }
        }
        return samplers
            .iter_mut()
            .enumerate()
            .map(|(row, sampler)| sampler.sample_with_logprobs(&logits.i(row)?, vocab_size))
            .collect::<Result<Vec<_>>>()
            .map(Some);
    }
    let plans = samplers
        .iter()
        .enumerate()
        .map(|(row, sampler)| sampler.row_sampling(&logits, row, vocab))
        .collect::<Result<Vec<_>>>()?;
    let mut parts = Vec::with_capacity(rows + 3);
    if let Some(guard) = guard {
        parts.push(guard.to_dtype(DType::F32)?.flatten_all()?);
    }
    let any_greedy = plans.iter().any(|plan| matches!(plan, RowSampling::Greedy));
    if any_greedy {
        let ids = logits.argmax_keepdim(D::Minus1)?;
        // Gather reads contiguous rows; every id is inside the vocabulary.
        let picked = full_logits.contiguous()?.gather(&ids, D::Minus1)?;
        parts.push(ids.to_dtype(DType::F32)?.flatten_all()?);
        parts.push(picked.to_dtype(DType::F32)?.flatten_all()?);
    }
    for (row, plan) in plans.iter().enumerate() {
        match plan {
            RowSampling::Greedy => {}
            RowSampling::Candidates(pending) => parts.push(pending.packed.clone()),
            RowSampling::Host => parts.push(logits.i(row)?.to_dtype(DType::F32)?),
        }
    }
    let packed = read_f32_values_to_host(&Tensor::cat(&parts, 0)?)?;
    let mut cursor = 0usize;
    if guard.is_some() {
        cursor = 1;
        if packed.first().copied() == Some(0.0) {
            return Ok(None);
        }
    }
    let greedy_at = cursor;
    if any_greedy {
        cursor += rows * 2;
    }
    let mut sampled = Vec::with_capacity(rows);
    for (row, (sampler, plan)) in samplers.iter_mut().zip(plans).enumerate() {
        let token = match plan {
            RowSampling::Greedy => {
                let picked = packed[greedy_at + rows + row];
                let token = if picked.is_finite() {
                    packed[greedy_at + row] as u32
                } else {
                    finite_argmax(&read_f32_values_to_host(&logits.i(row)?)?)?
                };
                (token, None)
            }
            RowSampling::Candidates(pending) => {
                let candidates = pending.unpack(take(&packed, &mut cursor, pending.len())?)?;
                let row_logits = logits.i(row)?;
                let token = sample_candidates_or_host(
                    candidates,
                    || read_f32_values_to_host(&row_logits),
                    vocab_size,
                    &sampler.config,
                    &sampler.history,
                    &mut sampler.rng,
                )?;
                (token, None)
            }
            RowSampling::Host => {
                sampler.sample_host_row(take(&packed, &mut cursor, vocab)?.to_vec(), vocab_size)?
            }
        };
        sampler.record(token.0);
        sampled.push(token);
    }
    if cursor != packed.len() {
        return Err(Error::InferenceError(
            "batched sampling readback has the wrong length".into(),
        ));
    }
    Ok(Some(sampled))
}

/// The next `len` values of a packed readback.
fn take<'a>(packed: &'a [f32], cursor: &mut usize, len: usize) -> Result<&'a [f32]> {
    let slice = packed
        .get(*cursor..*cursor + len)
        .ok_or_else(|| Error::InferenceError("batched sampling readback is too short".into()))?;
    *cursor += len;
    Ok(slice)
}

#[derive(Clone)]
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        let seed = if seed == 0 {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_nanos() as u64)
                .unwrap_or(0x9E37_79B9_7F4A_7C15)
        } else {
            seed
        };
        Self {
            state: seed ^ 0xA076_1D64_78BD_642F,
        }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / (u32::MAX as f64 + 1.0)) as f32
    }
}

fn sample_chat_token(
    logits: &Tensor,
    vocab_size: usize,
    config: &ChatGenerationConfig,
    history: &[u32],
    rng: &mut SimpleRng,
) -> Result<u32> {
    sample_chat_token_and_logprobs(logits, vocab_size, config, history, rng).map(|(token, _)| token)
}

/// DS9.3: raw-distribution logprob of the sampled token plus its top
/// alternatives. `None` when the request did not ask for logprobs.
#[derive(Debug, Clone, PartialEq)]
pub struct RawTokenLogprobs {
    pub token: u32,
    pub logprob: f32,
    /// Top-k `(token id, raw logprob)` pairs sorted by descending logprob.
    pub top: Vec<(u32, f32)>,
}

/// DS9.3: resolve raw sampler logprobs into the public per-token payload.
///
/// Token strings use single-token decoding: byte-level pieces that split a
/// multi-byte UTF-8 sequence decode lossily, and `bytes` carries those
/// lossy bytes.
pub fn resolve_token_logprob(
    tokenizer: &crate::tokenizer::Tokenizer,
    raw: &RawTokenLogprobs,
) -> Result<crate::engine::TokenLogprob> {
    fn decode_one(tokenizer: &crate::tokenizer::Tokenizer, id: u32) -> Result<String> {
        tokenizer.decode(&[id])
    }
    let token = decode_one(tokenizer, raw.token)?;
    let mut top_logprobs = Vec::with_capacity(raw.top.len());
    for (id, logprob) in &raw.top {
        let text = decode_one(tokenizer, *id)?;
        top_logprobs.push(crate::engine::TopTokenLogprob {
            bytes: text.as_bytes().to_vec(),
            token: text,
            logprob: *logprob,
        });
    }
    Ok(crate::engine::TokenLogprob {
        bytes: token.as_bytes().to_vec(),
        token,
        logprob: raw.logprob,
        top_logprobs,
    })
}

fn sample_chat_token_and_logprobs(
    logits: &Tensor,
    vocab_size: usize,
    config: &ChatGenerationConfig,
    history: &[u32],
    rng: &mut SimpleRng,
) -> Result<(u32, Option<RawTokenLogprobs>)> {
    if vocab_size == 0 {
        return Err(Error::InvalidInput(
            "chat sampler received vocab_size=0".to_string(),
        ));
    }
    let row = chat_logits_row(logits)?;

    // DS9.3: logprobs need the full raw row on host (logsumexp + top-k over
    // the raw distribution), so requests that ask for them route through the
    // host sampler instead of the device fast paths.
    if config.logprobs {
        return sample_host_logprobs(
            read_f32_values_to_host(&row)?,
            vocab_size,
            config,
            history,
            rng,
        );
    }

    if is_deterministic_greedy(config) {
        return chat_argmax_clamped(&row, vocab_size).map(|token| (token, None));
    }

    let candidates = bounded_device_sampling_candidates(
        &row,
        vocab_size,
        config.top_k,
        config.temperature,
        history,
        config.repetition_penalty,
        config.presence_penalty,
        None,
    )?;
    let token = sample_candidates_or_host(
        candidates,
        || read_f32_values_to_host(&row),
        vocab_size,
        config,
        history,
        rng,
    )?;
    Ok((token, None))
}

/// Greedy decoding that needs no host-side policy: argmax of the raw row.
fn is_deterministic_greedy(config: &ChatGenerationConfig) -> bool {
    config.temperature <= 1e-5
        && (config.repetition_penalty - 1.0).abs() <= f32::EPSILON
        && config.presence_penalty.abs() <= f32::EPSILON
        && config.top_k == 0
        && config.top_p >= 1.0
}

/// The logprobs route on a row already read to the host: sample with the
/// host pipeline and report the raw-distribution logprobs of the token.
fn sample_host_logprobs(
    values: Vec<f32>,
    vocab_size: usize,
    config: &ChatGenerationConfig,
    history: &[u32],
    rng: &mut SimpleRng,
) -> Result<(u32, Option<RawTokenLogprobs>)> {
    let mut sampled = values.clone();
    sampled.truncate(vocab_size.min(sampled.len()));
    if sampled.is_empty() {
        return Err(Error::InvalidInput(
            "chat sampler received no in-vocabulary logits".to_string(),
        ));
    }
    let (logsumexp, top) = raw_logprobs_stats(&sampled, config.top_logprobs)?;
    let token = sample_from_host_values(sampled, config, PenaltyHistory::uniform(history), rng)?;
    let chosen_raw = values
        .get(token as usize)
        .copied()
        .ok_or_else(|| Error::InferenceError("sampled token outside raw row".into()))?;
    if !chosen_raw.is_finite() {
        return Err(Error::InferenceError(
            "sampled token has a non-finite raw logit".into(),
        ));
    }
    Ok((
        token,
        Some(RawTokenLogprobs {
            token,
            logprob: chosen_raw - logsumexp,
            top,
        }),
    ))
}

/// Draw from bounded device candidates when they hold the whole top-p
/// nucleus, else from the full row on the host (`host_values` reads it).
fn sample_candidates_or_host(
    candidates: Option<DeviceSamplingCandidates>,
    host_values: impl FnOnce() -> Result<Vec<f32>>,
    vocab_size: usize,
    config: &ChatGenerationConfig,
    history: &[u32],
    rng: &mut SimpleRng,
) -> Result<u32> {
    if let Some(candidates) = candidates {
        if device_candidates_cover_top_p(&candidates, config.top_p) {
            if let Some(sampled) =
                sample_device_candidates(&candidates, config.top_p, rng.next_f32())
            {
                return Ok(sampled);
            }
        }
    }
    let mut values = host_values()?;
    values.truncate(vocab_size.min(values.len()));
    if values.is_empty() {
        return Err(Error::InvalidInput(
            "chat sampler received no in-vocabulary logits".to_string(),
        ));
    }
    sample_from_host_values(values, config, PenaltyHistory::uniform(history), rng)
}

/// DS9.3: raw-distribution log_softmax statistics over the full row:
/// the logsumexp plus the top-k `(token id, raw logprob)` pairs.
pub(crate) fn raw_logprobs_stats(
    values: &[f32],
    top_logprobs: usize,
) -> Result<(f32, Vec<(u32, f32)>)> {
    let finite = values.iter().filter(|v| v.is_finite());
    let max_logit = finite.clone().fold(f32::NEG_INFINITY, |acc, v| acc.max(*v));
    if !max_logit.is_finite() {
        return Err(Error::InferenceError(
            "chat sampler received no finite logits for logprobs".into(),
        ));
    }
    let logsumexp = max_logit
        + finite
            .clone()
            .map(|v| (v - max_logit).exp())
            .sum::<f32>()
            .ln();
    let mut indexed: Vec<(usize, f32)> = values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(index, v)| (index, *v))
        .collect();
    indexed.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top = indexed
        .into_iter()
        .take(top_logprobs)
        .map(|(index, value)| {
            u32::try_from(index)
                .map(|token| (token, value - logsumexp))
                .map_err(|_| Error::InferenceError("token id exceeds u32".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((logsumexp, top))
}

/// The existing host fallback pipeline, extracted so the logprobs route
/// samples through the exact same math (penalties → temperature → top-k →
/// top-p → draw) on the already-read row.
/// DS9.2: uniform unit draw used by host sampling paths. Abstracted so
/// model families that own an equivalent local RNG (qwen35's decode loop)
/// can drive the shared grammar runtime without changing their state.
pub(crate) trait GrammarRng {
    fn draw_unit(&mut self) -> f32;
}

impl GrammarRng for SimpleRng {
    fn draw_unit(&mut self) -> f32 {
        self.next_f32()
    }
}

fn sample_from_host_values<R: GrammarRng>(
    mut values: Vec<f32>,
    config: &ChatGenerationConfig,
    history: PenaltyHistory<'_>,
    rng: &mut R,
) -> Result<u32> {
    apply_chat_history_penalties(
        &mut values,
        history,
        config.repetition_penalty,
        config.presence_penalty,
    );
    if config.temperature <= 1e-5 {
        return finite_argmax(&values);
    }

    let temperature = config.temperature.max(1e-5);
    for value in &mut values {
        if value.is_finite() {
            *value /= temperature;
        }
    }
    let candidates = ranked_candidates(&values, config.top_k);
    if candidates.is_empty() {
        return finite_argmax(&values);
    }
    let max_logit = candidates
        .iter()
        .map(|index| values[*index])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = candidates
        .iter()
        .map(|index| (*index, (values[*index] - max_logit).exp()))
        .collect::<Vec<_>>();
    let mut sum = probabilities.iter().map(|(_, value)| *value).sum::<f32>();
    if !sum.is_finite() || sum <= 0.0 {
        return finite_argmax(&values);
    }
    for (_, probability) in &mut probabilities {
        *probability /= sum;
    }
    if config.top_p < 1.0 {
        let cutoff = config.top_p.clamp(1e-6, 1.0);
        let mut cumulative = 0.0f32;
        let mut keep = 0usize;
        for (_, probability) in &probabilities {
            cumulative += *probability;
            keep += 1;
            if cumulative >= cutoff {
                break;
            }
        }
        probabilities.truncate(keep.max(1));
        sum = probabilities.iter().map(|(_, value)| *value).sum();
        if sum.is_finite() && sum > 0.0 {
            for (_, probability) in &mut probabilities {
                *probability /= sum;
            }
        }
    }
    let draw = rng.draw_unit();
    let mut cumulative = 0.0f32;
    for (index, probability) in &probabilities {
        cumulative += *probability;
        if draw <= cumulative {
            return u32::try_from(*index)
                .map_err(|_| Error::InferenceError("sampled token exceeds u32".into()));
        }
    }
    probabilities
        .last()
        .map(|(index, _)| *index)
        .ok_or_else(|| Error::InferenceError("chat sampler produced no candidate".into()))
        .and_then(|index| {
            u32::try_from(index)
                .map_err(|_| Error::InferenceError("sampled token exceeds u32".into()))
        })
}

/// Finite candidates by value descending, ties by index ascending (the order
/// a stable sort of the index-ordered candidates gives), cut to `top_k` when
/// it is non-zero. The comparison is total, so selecting the top-k before
/// sorting returns exactly the full sort's prefix without sorting the whole
/// vocabulary.
fn ranked_candidates(values: &[f32], top_k: usize) -> Vec<usize> {
    let mut candidates = values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| value.is_finite().then_some(index))
        .collect::<Vec<_>>();
    let by_value = |left: &usize, right: &usize| {
        values[*right]
            .partial_cmp(&values[*left])
            .unwrap_or(Ordering::Equal)
            .then(left.cmp(right))
    };
    if top_k > 0 && top_k < candidates.len() {
        candidates.select_nth_unstable_by(top_k - 1, by_value);
        candidates.truncate(top_k);
    }
    candidates.sort_by(by_value);
    candidates
}

fn chat_logits_row(logits: &Tensor) -> Result<Tensor> {
    match logits.rank() {
        1 => Ok(logits.clone()),
        2 => {
            let rows = logits.dim(0)?;
            if rows == 0 {
                return Err(Error::InvalidInput(
                    "chat sampler received an empty sequence".into(),
                ));
            }
            logits.i(rows - 1).map_err(Error::from)
        }
        3 => {
            let (batch, sequence, _) = logits.dims3()?;
            if batch != 1 || sequence == 0 {
                return Err(Error::InvalidInput(format!(
                    "chat sampler expected one non-empty row, got {:?}",
                    logits.dims()
                )));
            }
            logits.i((0, sequence - 1)).map_err(Error::from)
        }
        rank => Err(Error::InvalidInput(format!(
            "chat sampler expected rank 1, 2, or 3 logits, got rank {rank}"
        ))),
    }
}

fn chat_argmax_clamped(row: &Tensor, vocab_size: usize) -> Result<u32> {
    let width = row.dim(0)?;
    let row = if vocab_size < width {
        row.narrow(0, 0, vocab_size)?
    } else {
        row.clone()
    };
    if row.dim(0)? == 0 {
        return Err(Error::InvalidInput(
            "chat sampler received no in-vocabulary logits".into(),
        ));
    }
    let selected = row
        .argmax(D::Minus1)?
        .to_dtype(DType::U32)?
        .to_scalar::<u32>()?;
    crate::models::shared::telemetry::record_host_read(DType::U32, 1);
    let selected_value = row
        .i(selected as usize)?
        .to_dtype(DType::F32)?
        .to_scalar::<f32>()?;
    crate::models::shared::telemetry::record_host_read(DType::F32, 1);
    if selected_value.is_finite() {
        return Ok(selected);
    }
    finite_argmax(&read_f32_values_to_host(&row)?)
}

fn read_f32_values_to_host(values: &Tensor) -> Result<Vec<f32>> {
    let values = values.to_dtype(DType::F32)?;
    let element_count = values.elem_count();
    let host_values = values.to_vec1::<f32>()?;
    crate::models::shared::telemetry::record_host_read(DType::F32, element_count);
    Ok(host_values)
}

fn apply_chat_history_penalties(
    values: &mut [f32],
    history: PenaltyHistory<'_>,
    repetition_penalty: f32,
    presence_penalty: f32,
) {
    if !history.is_uniform() {
        apply_split_history_penalties(values, history, repetition_penalty, presence_penalty);
        return;
    }
    let history = history.repetition;
    if history.is_empty()
        || ((repetition_penalty - 1.0).abs() <= f32::EPSILON
            && presence_penalty.abs() <= f32::EPSILON)
    {
        return;
    }
    let mut seen = vec![false; values.len()];
    for token in history {
        if let Some(seen) = seen.get_mut(*token as usize) {
            *seen = true;
        }
    }
    for (index, was_seen) in seen.into_iter().enumerate() {
        if !was_seen || !values[index].is_finite() {
            continue;
        }
        if repetition_penalty > 1.0 {
            if values[index] > 0.0 {
                values[index] /= repetition_penalty;
            } else {
                values[index] *= repetition_penalty;
            }
        }
        values[index] -= presence_penalty;
    }
}

/// Host penalties for separate histories: repetition over
/// `history.repetition`, presence over `history.presence`.
fn apply_split_history_penalties(
    values: &mut [f32],
    history: PenaltyHistory<'_>,
    repetition_penalty: f32,
    presence_penalty: f32,
) {
    let len = values.len();
    let seen = |tokens: &[u32]| {
        let mut seen = vec![false; len];
        for token in tokens {
            if let Some(seen) = seen.get_mut(*token as usize) {
                *seen = true;
            }
        }
        seen
    };
    if repetition_penalty > 1.0 && !history.repetition.is_empty() {
        for (value, was_seen) in values.iter_mut().zip(seen(history.repetition)) {
            if !was_seen || !value.is_finite() {
                continue;
            }
            if *value > 0.0 {
                *value /= repetition_penalty;
            } else {
                *value *= repetition_penalty;
            }
        }
    }
    if presence_penalty.abs() > f32::EPSILON && !history.presence.is_empty() {
        for (value, was_seen) in values.iter_mut().zip(seen(history.presence)) {
            if was_seen && value.is_finite() {
                *value -= presence_penalty;
            }
        }
    }
}

fn finite_argmax(values: &[f32]) -> Result<u32> {
    values
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|left, right| left.1.partial_cmp(right.1).unwrap_or(Ordering::Equal))
        .map(|(index, _)| index)
        .ok_or_else(|| Error::InferenceError("chat sampler found no finite logits".into()))
        .and_then(|index| {
            u32::try_from(index)
                .map_err(|_| Error::InferenceError("sampled token exceeds u32".into()))
        })
}

#[derive(Clone)]
enum SamplingTensorCacheInput {
    Mask(Vec<u8>),
    Indices(Vec<u32>),
    /// `0..len` as U32. Candle builds `arange` on the host, so an uncached
    /// full-vocabulary range uploads ~1 MB per sampled token.
    Arange(u32),
}

struct SamplingTensorCacheEntry {
    device: candle_core::DeviceLocation,
    input: SamplingTensorCacheInput,
    tensor: Tensor,
}

thread_local! {
    static DEVICE_SAMPLING_TENSOR_CACHE: RefCell<VecDeque<SamplingTensorCacheEntry>> =
        const { RefCell::new(VecDeque::new()) };
}

fn cached_sampling_tensor(device: &Device, input: SamplingTensorCacheInput) -> Result<Tensor> {
    DEVICE_SAMPLING_TENSOR_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let location = device.location();
        if let Some(index) = cache.iter().position(|entry| {
            entry.device == location
                && match (&entry.input, &input) {
                    (
                        SamplingTensorCacheInput::Mask(left),
                        SamplingTensorCacheInput::Mask(right),
                    ) => left == right,
                    (
                        SamplingTensorCacheInput::Indices(left),
                        SamplingTensorCacheInput::Indices(right),
                    ) => left == right,
                    (
                        SamplingTensorCacheInput::Arange(left),
                        SamplingTensorCacheInput::Arange(right),
                    ) => left == right,
                    _ => false,
                }
        }) {
            let entry = cache
                .remove(index)
                .expect("located device sampling tensor cache entry");
            let tensor = entry.tensor.clone();
            cache.push_back(entry);
            return Ok(tensor);
        }
        let tensor = match &input {
            SamplingTensorCacheInput::Mask(values) => {
                Tensor::from_vec(values.clone(), values.len(), device)?
            }
            SamplingTensorCacheInput::Indices(values) => {
                Tensor::from_vec(values.clone(), values.len(), device)?
            }
            SamplingTensorCacheInput::Arange(len) => Tensor::arange(0u32, *len, device)?,
        };
        if cache.len() == DEVICE_SAMPLING_TENSOR_CACHE_ENTRIES {
            cache.pop_front();
        }
        cache.push_back(SamplingTensorCacheEntry {
            device: location,
            input,
            tensor: tensor.clone(),
        });
        Ok(tensor)
    })
}

#[derive(Debug, Clone)]
pub struct DeviceSamplingCandidates {
    pub values: Vec<f32>,
    pub indices: Vec<u32>,
    /// Present when candidates were selected without an explicit top-k. The
    /// caller can prove that a top-p cutoff is wholly contained in the bounded
    /// result by comparing cumulative `exp(value - logsumexp)` with top-p.
    pub logsumexp: Option<f32>,
}

pub fn sample_device_candidates(
    candidates: &DeviceSamplingCandidates,
    top_p: f32,
    random_draw: f32,
) -> Option<u32> {
    let (mut probabilities, keep) = candidate_probabilities(candidates, top_p)?;
    probabilities.truncate(keep);
    let kept_sum = probabilities.iter().sum::<f32>();
    if !kept_sum.is_finite() || kept_sum <= 0.0 {
        return None;
    }
    let mut draw = random_draw.clamp(0.0, 1.0 - f32::EPSILON) * kept_sum;
    for (index, probability) in probabilities.iter().enumerate() {
        if draw <= *probability {
            return candidates.indices.get(index).copied();
        }
        draw -= *probability;
    }
    candidates.indices.first().copied()
}

pub fn device_candidates_cover_top_p(candidates: &DeviceSamplingCandidates, top_p: f32) -> bool {
    candidate_probabilities(candidates, top_p).is_some()
}

fn candidate_probabilities(
    candidates: &DeviceSamplingCandidates,
    top_p: f32,
) -> Option<(Vec<f32>, usize)> {
    if candidates.values.is_empty() || candidates.values.len() != candidates.indices.len() {
        return None;
    }
    let probabilities = if let Some(logsumexp) = candidates.logsumexp {
        candidates
            .values
            .iter()
            .map(|value| (*value - logsumexp).exp())
            .collect::<Vec<_>>()
    } else {
        let max = candidates.values[0];
        let mut probabilities = candidates
            .values
            .iter()
            .map(|value| (*value - max).exp())
            .collect::<Vec<_>>();
        let total = probabilities.iter().sum::<f32>();
        if !total.is_finite() || total <= 0.0 {
            return None;
        }
        for probability in &mut probabilities {
            *probability /= total;
        }
        probabilities
    };
    if probabilities.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let cutoff = top_p.clamp(1e-6, 1.0);
    let mut cumulative = 0.0f32;
    let mut keep = 0usize;
    for probability in &probabilities {
        cumulative += *probability;
        keep += 1;
        if cumulative >= cutoff {
            break;
        }
    }
    if candidates.logsumexp.is_some() && cumulative < cutoff {
        return None;
    }
    Some((probabilities, keep.max(1)))
}

#[allow(clippy::too_many_arguments)]
/// Build a bounded, exact candidate set on a supported accelerator.
///
/// CUDA and Metal execute the tensor-side selection. `None` asks the caller to
/// preserve the exact host fallback (for an unsupported device, greedy
/// temperature, or a request outside the bounded contract).
pub fn bounded_device_sampling_candidates(
    logits: &Tensor,
    vocab_limit: usize,
    top_k: usize,
    temperature: f32,
    history: &[u32],
    repetition_penalty: f32,
    presence_penalty: f32,
    allowed_mask: Option<&[bool]>,
) -> Result<Option<DeviceSamplingCandidates>> {
    bounded_device_sampling_candidates_with_history(
        logits,
        vocab_limit,
        top_k,
        temperature,
        PenaltyHistory::uniform(history),
        repetition_penalty,
        presence_penalty,
        allowed_mask,
    )
}

/// [`bounded_device_sampling_candidates`] with separate penalty histories.
#[allow(clippy::too_many_arguments)]
pub fn bounded_device_sampling_candidates_with_history(
    logits: &Tensor,
    vocab_limit: usize,
    top_k: usize,
    temperature: f32,
    history: PenaltyHistory<'_>,
    repetition_penalty: f32,
    presence_penalty: f32,
    allowed_mask: Option<&[bool]>,
) -> Result<Option<DeviceSamplingCandidates>> {
    if !(logits.device().is_cuda() || logits.device().is_metal()) || temperature <= 1e-5 {
        return Ok(None);
    }
    bounded_sampling_candidates(
        logits,
        vocab_limit,
        top_k,
        temperature,
        history,
        repetition_penalty,
        presence_penalty,
        allowed_mask,
    )
}

/// Compatibility entry point for callers that intentionally remain CUDA-only.
/// New accelerator-neutral callers should use
/// [`bounded_device_sampling_candidates`].
#[allow(clippy::too_many_arguments)]
pub fn bounded_cuda_sampling_candidates(
    logits: &Tensor,
    vocab_limit: usize,
    top_k: usize,
    temperature: f32,
    history: &[u32],
    repetition_penalty: f32,
    presence_penalty: f32,
    allowed_mask: Option<&[bool]>,
) -> Result<Option<DeviceSamplingCandidates>> {
    if !logits.device().is_cuda() {
        return Ok(None);
    }
    bounded_sampling_candidates(
        logits,
        vocab_limit,
        top_k,
        temperature,
        PenaltyHistory::uniform(history),
        repetition_penalty,
        presence_penalty,
        allowed_mask,
    )
}

#[allow(clippy::too_many_arguments)]
fn bounded_sampling_candidates(
    logits: &Tensor,
    vocab_limit: usize,
    top_k: usize,
    temperature: f32,
    history: PenaltyHistory<'_>,
    repetition_penalty: f32,
    presence_penalty: f32,
    allowed_mask: Option<&[bool]>,
) -> Result<Option<DeviceSamplingCandidates>> {
    let Some(pending) = pending_sampling_candidates(
        logits,
        vocab_limit,
        top_k,
        temperature,
        history,
        repetition_penalty,
        presence_penalty,
        allowed_mask,
    )?
    else {
        return Ok(None);
    };
    pending.unpack(&read_f32_values_to_host(&pending.packed)?)
}

#[allow(clippy::too_many_arguments)]
fn pending_sampling_candidates(
    logits: &Tensor,
    vocab_limit: usize,
    top_k: usize,
    temperature: f32,
    history: PenaltyHistory<'_>,
    repetition_penalty: f32,
    presence_penalty: f32,
    allowed_mask: Option<&[bool]>,
) -> Result<Option<PendingCandidates>> {
    if temperature <= 1e-5 {
        return Ok(None);
    }
    let row = logits_row(logits)?;
    let vocab = row.dim(0)?.min(vocab_limit);
    if vocab == 0 || vocab > MAX_EXACT_F32_INTEGER || top_k > DEVICE_SORT_CHUNK {
        return Ok(None);
    }
    if allowed_mask.is_some_and(|mask| mask.len() < vocab) {
        return Err(Error::InvalidInput(
            "sampling mask is shorter than the selected vocabulary".into(),
        ));
    }

    let mut adjusted = row.narrow(0, 0, vocab)?.to_dtype(DType::F32)?;
    adjusted = sanitize_finite(&adjusted)?;
    if let Some(mask) = allowed_mask {
        let mask = cached_sampling_tensor(
            adjusted.device(),
            SamplingTensorCacheInput::Mask(
                mask[..vocab]
                    .iter()
                    .map(|allowed| u8::from(*allowed))
                    .collect(),
            ),
        )?;
        let negative = negative_infinity(adjusted.device())?.broadcast_as(adjusted.shape())?;
        adjusted = mask.where_cond(&adjusted, &negative)?;
    }
    adjusted = apply_history_penalties(&adjusted, history, repetition_penalty, presence_penalty)?;
    adjusted = (adjusted / temperature.max(1e-5) as f64)?;

    let requested = if top_k == 0 {
        DEVICE_SAMPLING_CANDIDATE_LIMIT
    } else {
        top_k
    }
    .min(vocab);
    let (values, indices) = bounded_topk(&adjusted, requested)?;
    let logsumexp = if top_k == 0 {
        Some(adjusted.log_sum_exp(D::Minus1)?.reshape(1)?)
    } else {
        None
    };
    let indices_f32 = indices.to_dtype(DType::F32)?;
    let mut packed_parts = vec![&values, &indices_f32];
    if let Some(logsumexp) = &logsumexp {
        packed_parts.push(logsumexp);
    }
    Ok(Some(PendingCandidates {
        packed: Tensor::cat(&packed_parts, 0)?,
        count: values.dim(0)?,
        has_logsumexp: logsumexp.is_some(),
    }))
}

/// One row's bounded candidates, still on the device as one packed F32
/// tensor (values, indices, then the normalizer when present), so a caller
/// can read many rows back in a single transfer.
struct PendingCandidates {
    packed: Tensor,
    count: usize,
    has_logsumexp: bool,
}

impl PendingCandidates {
    fn len(&self) -> usize {
        self.count * 2 + usize::from(self.has_logsumexp)
    }

    /// `None` when a candidate is non-finite: the caller keeps the exact
    /// host fallback.
    fn unpack(&self, packed: &[f32]) -> Result<Option<DeviceSamplingCandidates>> {
        let (values, indices, logsumexp) =
            unpack_sampling_readback(packed, self.count, self.has_logsumexp)?;
        if values.iter().any(|value| !value.is_finite()) {
            return Ok(None);
        }
        Ok(Some(DeviceSamplingCandidates {
            values,
            indices,
            logsumexp,
        }))
    }
}

fn unpack_sampling_readback(
    packed: &[f32],
    candidate_count: usize,
    has_logsumexp: bool,
) -> Result<(Vec<f32>, Vec<u32>, Option<f32>)> {
    let expected = candidate_count
        .checked_mul(2)
        .and_then(|count| count.checked_add(usize::from(has_logsumexp)))
        .ok_or_else(|| Error::InferenceError("sampling readback size overflow".into()))?;
    if packed.len() != expected {
        return Err(Error::InferenceError(format!(
            "sampling readback returned {} values, expected {expected}",
            packed.len()
        )));
    }
    let values = packed[..candidate_count].to_vec();
    let indices = packed[candidate_count..candidate_count * 2]
        .iter()
        .map(|value| *value as u32)
        .collect::<Vec<_>>();
    let logsumexp = has_logsumexp.then(|| packed[candidate_count * 2]);
    Ok((values, indices, logsumexp))
}

fn logits_row(logits: &Tensor) -> Result<Tensor> {
    match logits.rank() {
        1 => Ok(logits.clone()),
        2 if logits.dim(0)? == 1 => logits.get(0).map_err(Error::from),
        rank => Err(Error::InvalidInput(format!(
            "device sampler expected rank-1 logits or one rank-2 row, got rank {rank} shape {:?}",
            logits.dims()
        ))),
    }
}

fn negative_infinity(device: &Device) -> Result<Tensor> {
    Tensor::new(f32::NEG_INFINITY, device).map_err(Error::from)
}

fn sanitize_finite(values: &Tensor) -> Result<Tensor> {
    let not_nan = values.eq(values)?;
    let finite = values.abs()?.le(f32::MAX as f64)?;
    let valid = not_nan.broadcast_mul(&finite)?;
    let negative = negative_infinity(values.device())?.broadcast_as(values.shape())?;
    valid.where_cond(values, &negative).map_err(Error::from)
}

fn apply_history_penalties(
    values: &Tensor,
    history: PenaltyHistory<'_>,
    repetition_penalty: f32,
    presence_penalty: f32,
) -> Result<Tensor> {
    if !history.is_uniform() {
        let mut values = values.clone();
        if repetition_penalty > 1.0 {
            values = apply_history_penalties(
                &values,
                PenaltyHistory::uniform(history.repetition),
                repetition_penalty,
                0.0,
            )?;
        }
        if presence_penalty.abs() > f32::EPSILON {
            values = apply_history_penalties(
                &values,
                PenaltyHistory::uniform(history.presence),
                1.0,
                presence_penalty,
            )?;
        }
        return Ok(values);
    }
    let history = history.repetition;
    if history.is_empty()
        || ((repetition_penalty - 1.0).abs() <= f32::EPSILON
            && presence_penalty.abs() <= f32::EPSILON)
    {
        return Ok(values.clone());
    }
    let indices = distinct_in_vocab(history, values.dim(0)?);
    if indices.is_empty() {
        return Ok(values.clone());
    }
    let indices =
        cached_sampling_tensor(values.device(), SamplingTensorCacheInput::Indices(indices))?;
    let mut replacements = values.index_select(&indices, 0)?;
    if repetition_penalty > 1.0 {
        let positive = replacements.gt(0.0f64)?;
        let divided = (&replacements / repetition_penalty as f64)?;
        let multiplied = (&replacements * repetition_penalty as f64)?;
        replacements = positive.where_cond(&divided, &multiplied)?;
    }
    if presence_penalty.abs() > f32::EPSILON {
        replacements = (replacements - presence_penalty as f64)?;
    }
    values
        .scatter(&indices, &replacements, 0)
        .map_err(Error::from)
}

/// The distinct in-vocabulary tokens of `history`, ascending. Sorting wins
/// for short histories; past about vocab/8 tokens a vocabulary bitmap costs
/// less (a 32k-token history: ~60 µs instead of ~370 µs, every step).
fn distinct_in_vocab(history: &[u32], vocab: usize) -> Vec<u32> {
    if history.len() < vocab / 8 {
        let mut tokens = history
            .iter()
            .copied()
            .filter(|token| (*token as usize) < vocab)
            .collect::<Vec<_>>();
        tokens.sort_unstable();
        tokens.dedup();
        return tokens;
    }
    let mut seen = vec![false; vocab];
    for token in history {
        if let Some(seen) = seen.get_mut(*token as usize) {
            *seen = true;
        }
    }
    seen.into_iter()
        .enumerate()
        .filter_map(|(token, seen)| seen.then_some(token as u32))
        .collect()
}

/// Draw one token on CUDA or Metal and keep it on device: temperature and
/// top-k run on the accelerator, then the Gumbel-max trick picks among the
/// k candidates using `draw` (host uniforms in [0, 1), k values uploaded).
/// Nothing is read back, so a caller can feed the token straight into the
/// next embedding gather and read a whole frame once. Exact sampling from
/// softmax(top-k(logits / temperature)). `None` asks for the host path: other
/// devices, greedy temperature, or top-k 0 / above the device sort bound.
pub fn device_resident_top_k_sample(
    logits: &Tensor,
    vocab_limit: usize,
    top_k: usize,
    temperature: f32,
    draw: &mut dyn FnMut() -> f32,
) -> Result<Option<Tensor>> {
    if !(logits.device().is_cuda() || logits.device().is_metal()) {
        return Ok(None);
    }
    resident_top_k_sample(logits, vocab_limit, top_k, temperature, draw)
}

/// [`device_resident_top_k_sample`] on any device, CPU included: the token
/// stays a `[1]` U32 tensor where the logits live. `None` for greedy
/// temperature or a top-k of 0 / above the device sort bound.
pub fn resident_top_k_sample(
    logits: &Tensor,
    vocab_limit: usize,
    top_k: usize,
    temperature: f32,
    draw: &mut dyn FnMut() -> f32,
) -> Result<Option<Tensor>> {
    if temperature <= 1e-5 || top_k == 0 || top_k > DEVICE_SORT_CHUNK {
        return Ok(None);
    }
    let row = logits_row(logits)?;
    let vocab = row.dim(0)?.min(vocab_limit);
    if vocab == 0 {
        return Ok(None);
    }
    let adjusted = sanitize_finite(&row.narrow(0, 0, vocab)?.to_dtype(DType::F32)?)?;
    let adjusted = (adjusted / temperature as f64)?;
    let k = top_k.min(vocab);
    let gumbel = gumbel_noise(k, draw);
    gumbel_top_k_token(&adjusted, k, &gumbel).map(Some)
}

/// `-ln(-ln(u))` for `count` uniforms, the Gumbel(0, 1) noise of the
/// Gumbel-max trick.
pub(crate) fn gumbel_noise(count: usize, draw: &mut dyn FnMut() -> f32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            let uniform = f64::from(draw()).clamp(1e-12, 1.0 - 1e-12);
            (-(-uniform.ln()).ln()) as f32
        })
        .collect()
}

/// The top-k candidate of `scores` (already scaled by temperature) whose
/// score plus `gumbel` is largest, as a `[1]` U32 token on the scores' device.
fn gumbel_top_k_token(scores: &Tensor, k: usize, gumbel: &[f32]) -> Result<Tensor> {
    let (values, indices) = bounded_topk(scores, k)?;
    let keep = values.dim(0)?;
    let noise = Tensor::from_vec(gumbel[..keep].to_vec(), keep, scores.device())?;
    let pick = values
        .broadcast_add(&noise)?
        .argmax(D::Minus1)?
        .to_dtype(DType::U32)?
        .reshape(1)?;
    indices
        .to_dtype(DType::U32)?
        .index_select(&pick, 0)
        .map_err(Error::from)
}

/// The `k` largest values of a rank-1 `values`, sorted descending, and their
/// U32 indices, computed on the values' device. `None` when `k` is 0 or above
/// the device sort bound.
pub(crate) fn device_top_k(values: &Tensor, k: usize) -> Result<Option<(Tensor, Tensor)>> {
    if k == 0 || k > DEVICE_SORT_CHUNK {
        return Ok(None);
    }
    bounded_topk(values, k).map(Some)
}

fn bounded_topk(values: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    let length = values.dim(0)?;
    if length == 0 || length > u32::MAX as usize || k == 0 || k > DEVICE_SORT_CHUNK {
        return Err(Error::InvalidInput(format!(
            "bounded top-k requires a U32-addressable non-empty row and 1..={DEVICE_SORT_CHUNK} candidates"
        )));
    }
    let mut candidate_values = values.contiguous()?;
    let mut candidate_indices = cached_sampling_tensor(
        values.device(),
        SamplingTensorCacheInput::Arange(length as u32),
    )?;
    loop {
        let length = candidate_values.dim(0)?;
        if length <= DEVICE_SORT_CHUNK {
            let (sorted, order) = candidate_values.sort_last_dim(false)?;
            let keep = k.min(length);
            return Ok((
                sorted.narrow(0, 0, keep)?,
                candidate_indices.gather(&order, 0)?.narrow(0, 0, keep)?,
            ));
        }

        let chunks = length.div_ceil(DEVICE_SORT_CHUNK);
        let padded_len = chunks * DEVICE_SORT_CHUNK;
        if padded_len > length {
            let padding = padded_len - length;
            candidate_values = Tensor::cat(
                &[
                    &candidate_values,
                    &Tensor::full(f32::NEG_INFINITY, padding, values.device())?,
                ],
                0,
            )?;
            candidate_indices = Tensor::cat(
                &[
                    &candidate_indices,
                    &Tensor::full(u32::MAX, padding, values.device())?,
                ],
                0,
            )?;
        }
        let values_2d = candidate_values.reshape((chunks, DEVICE_SORT_CHUNK))?;
        let indices_2d = candidate_indices.reshape((chunks, DEVICE_SORT_CHUNK))?;
        let (sorted, order) = values_2d.sort_last_dim(false)?;
        let keep = k.min(DEVICE_SORT_CHUNK);
        candidate_values = sorted.narrow(1, 0, keep)?.flatten_all()?;
        candidate_indices = indices_2d
            .gather(&order, 1)?
            .narrow(1, 0, keep)?
            .flatten_all()?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unstructured logits: xorshift values in [-4, 4).
    fn xorshift_logits(state: &mut u64, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| {
                *state ^= *state << 13;
                *state ^= *state >> 7;
                *state ^= *state << 17;
                (*state >> 40) as f32 / (1u64 << 24) as f32 * 8.0 - 4.0
            })
            .collect()
    }

    fn sampled(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> ChatGenerationConfig {
        ChatGenerationConfig {
            temperature,
            top_k,
            top_p,
            seed,
            ..ChatGenerationConfig::default()
        }
    }

    /// Steps `samplers` through `steps` batches of `[rows, width]` logits on
    /// `device`, sampling every batch once with [`sample_chat_rows`] and
    /// once row by row with clones through [`ChatSampler::sample_with_logprobs`]
    /// (the single-row path), and asserts both draw the same tokens and
    /// logprobs. Returns the host reads of each batched step.
    fn assert_batched_rows_match_single_rows(
        device: &Device,
        mut samplers: Vec<ChatSampler>,
        width: usize,
        vocab_size: usize,
        steps: usize,
    ) -> Vec<u64> {
        let mut single = samplers.clone();
        let rows = samplers.len();
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut reads = Vec::with_capacity(steps);
        for step in 0..steps {
            let logits = Tensor::from_vec(
                xorshift_logits(&mut state, rows * width),
                (rows, width),
                device,
            )
            .unwrap();
            let mut batch = samplers.iter_mut().collect::<Vec<_>>();
            let before = crate::models::shared::telemetry::thread_host_reads();
            let batched = sample_chat_rows(&mut batch, &logits, vocab_size).unwrap();
            reads.push(crate::models::shared::telemetry::thread_host_reads() - before);
            let expected = single
                .iter_mut()
                .enumerate()
                .map(|(row, sampler)| {
                    sampler
                        .sample_with_logprobs(&logits.i(row).unwrap(), vocab_size)
                        .unwrap()
                })
                .collect::<Vec<_>>();
            assert_eq!(batched, expected, "step {step}");
        }
        for (batched, single) in samplers.iter().zip(&single) {
            assert_eq!(batched.history, single.history);
        }
        reads
    }

    /// Greedy, top-k, top-p, penalised, greedy-with-penalty and logprob
    /// rows over a vocabulary narrower than the logits and wider than one
    /// device sort chunk.
    fn mixed_samplers(include_nucleus_only: bool) -> Vec<ChatSampler> {
        let history = [3, 17, 17, 905];
        let mut configs = vec![
            ChatGenerationConfig::default(),
            sampled(0.9, 40, 1.0, 3),
            sampled(0.7, 64, 0.8, 4),
            ChatGenerationConfig {
                repetition_penalty: 1.3,
                presence_penalty: 0.5,
                ..sampled(0.7, 32, 1.0, 5)
            },
            ChatGenerationConfig {
                repetition_penalty: 1.2,
                ..ChatGenerationConfig::default()
            },
            ChatGenerationConfig {
                logprobs: true,
                top_logprobs: 3,
                ..sampled(0.8, 0, 1.0, 6)
            },
            ChatGenerationConfig {
                logprobs: true,
                top_logprobs: 2,
                ..ChatGenerationConfig::default()
            },
        ];
        if include_nucleus_only {
            configs.push(sampled(1.1, 0, 0.9, 7));
        }
        configs
            .into_iter()
            .map(|config| ChatSampler::new(config, &history))
            .collect()
    }

    fn json_samplers() -> Vec<ChatSampler> {
        let tokenizer = crate::tokenizer::Tokenizer::from_hf_json_bytes(
            br#"{
            "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":{
                "{":0,"\"":1,"a":2,":":3,"1":4,"}":5,",":6,"[":7,"]":8," ":9,"b":10,
                "<eos>":11,"<unk>":12},
                "unk_token":"<unk>"}
        }"#,
        )
        .unwrap();
        let tokenizer = std::sync::Arc::new(tokenizer);
        let json = |config: ChatGenerationConfig| {
            ChatSampler::new(
                ChatGenerationConfig {
                    constrain_json_object: true,
                    ..config
                },
                &[],
            )
            .with_json_object_constraint(tokenizer.clone(), vec![11])
        };
        vec![
            json(sampled(1.0, 0, 1.0, 8)),
            json(ChatGenerationConfig {
                logprobs: true,
                top_logprobs: 2,
                ..sampled(0.9, 4, 1.0, 9)
            }),
            ChatSampler::new(ChatGenerationConfig::default(), &[]),
            ChatSampler::new(sampled(1.0, 3, 1.0, 10), &[]),
        ]
    }

    fn assert_batched_sampling_parity(device: &Device) {
        assert_batched_rows_match_single_rows(device, mixed_samplers(true), 1_500, 1_400, 24);
        assert_batched_rows_match_single_rows(device, json_samplers(), 16, 13, 24);
        let reads =
            assert_batched_rows_match_single_rows(device, mixed_samplers(false), 1_500, 1_400, 24);
        assert!(
            reads.iter().all(|&reads| reads == 1),
            "reads per step {reads:?}"
        );
    }

    #[test]
    fn batched_rows_match_single_row_sampling_with_one_readback_on_cpu() {
        assert_batched_sampling_parity(&Device::Cpu);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn batched_rows_match_single_row_sampling_with_one_readback_on_metal() {
        let Some((device, _serial)) = crate::backends::metal_test_device_or_skip(
            "batched_rows_match_single_row_sampling_with_one_readback_on_metal",
        ) else {
            return;
        };
        assert_batched_sampling_parity(&device);
    }

    #[test]
    fn batched_rows_fall_back_like_single_rows_on_non_finite_greedy_picks() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![f32::NAN, 1.0, 2.0, 0.5, f32::INFINITY, -1.0],
            (2, 3),
            &device,
        )
        .unwrap();
        let mut samplers = vec![ChatSampler::new(ChatGenerationConfig::default(), &[]); 2];
        let mut single = samplers.clone();
        let mut batch = samplers.iter_mut().collect::<Vec<_>>();
        let batched = sample_chat_rows(&mut batch, &logits, 3).unwrap();
        let expected = single
            .iter_mut()
            .enumerate()
            .map(|(row, sampler)| {
                sampler
                    .sample_with_logprobs(&logits.i(row).unwrap(), 3)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(batched, expected);
    }

    #[test]
    fn distinct_in_vocab_is_sorted_deduplicated_and_bounded() {
        let mut state = 0x2545_f491_u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for vocab in [1usize, 8, 64, 1000] {
            // Both sides of the vocab/8 switch, with out-of-vocab ids.
            for len in [0usize, 1, vocab / 8, vocab / 8 + 1, vocab * 3] {
                let history: Vec<u32> = (0..len).map(|_| next() % (vocab as u32 + 5)).collect();
                let reference: Vec<u32> = history
                    .iter()
                    .copied()
                    .filter(|token| (*token as usize) < vocab)
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect();
                assert_eq!(
                    distinct_in_vocab(&history, vocab),
                    reference,
                    "vocab {vocab} len {len}"
                );
            }
        }
    }

    /// The partial top-k selection returns exactly the full stable sort's
    /// prefix, ties and non-finite values included.
    #[test]
    fn ranked_candidates_match_a_full_stable_sort() {
        let mut state = 0x9e37_79b9_u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for trial in 0..200 {
            let len = 1 + (next() % 300) as usize;
            let values: Vec<f32> = (0..len)
                .map(|_| match next() % 10 {
                    // Coarse values force many ties at the top-k boundary.
                    0 => f32::NEG_INFINITY,
                    1 => f32::NAN,
                    _ => (next() % 7) as f32 * 0.5,
                })
                .collect();
            for top_k in [0usize, 1, 2, 5, 50, len, len + 3] {
                let mut reference: Vec<usize> = (0..len)
                    .filter(|index| values[*index].is_finite())
                    .collect();
                reference.sort_by(|left, right| {
                    values[*right]
                        .partial_cmp(&values[*left])
                        .unwrap_or(Ordering::Equal)
                });
                if top_k > 0 {
                    reference.truncate(top_k.min(reference.len()));
                }
                assert_eq!(
                    ranked_candidates(&values, top_k),
                    reference,
                    "trial {trial} top_k {top_k}"
                );
            }
        }
    }

    /// Gumbel-max over the device top-k: the token is the top-k candidate
    /// with the largest score + noise, and only top-k candidates can win.
    #[test]
    fn gumbel_top_k_token_picks_the_noisy_argmax_within_top_k() {
        let device = Device::Cpu;
        let scores = Tensor::from_vec(vec![0.5f32, 3.0, 1.0, 2.5, -4.0], 5, &device).unwrap();
        // Top-3 in score order: indices 1 (3.0), 3 (2.5), 2 (1.0).
        let pick = |noise: [f32; 3]| {
            gumbel_top_k_token(&scores, 3, &noise)
                .unwrap()
                .to_vec1::<u32>()
                .unwrap()[0]
        };
        assert_eq!(pick([0.0, 0.0, 0.0]), 1);
        assert_eq!(pick([0.0, 0.6, 0.0]), 3);
        assert_eq!(pick([0.0, 0.0, 2.1]), 2);
        // Index 0 (0.5) is outside the top 3, whatever the noise.
        assert_ne!(pick([-9.0, -9.0, -9.0]), 0);
        let noise = gumbel_noise(4, &mut || 0.5);
        assert!((noise[0] - 0.366_512_9).abs() < 1e-6, "{noise:?}");
    }

    /// The device-resident sampler runs on Metal and agrees with the CPU
    /// Gumbel-max for the same draws.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_device_resident_top_k_sample_matches_cpu_gumbel_max() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "metal_device_resident_top_k_sample_matches_cpu_gumbel_max",
        ) else {
            return;
        };
        let values = crate::kernels::numeric_check::xorshift_values(2049, 31, 4.0);
        let logits = Tensor::from_vec(values, 2049, &device).unwrap();
        for seed in 0..8u64 {
            let mut state = seed;
            let mut draws = Vec::new();
            let mut draw = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let value = (state >> 40) as f32 / (1u64 << 24) as f32;
                draws.push(value);
                value
            };
            let token = device_resident_top_k_sample(&logits, 2048, 64, 0.8, &mut draw)
                .unwrap()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap()[0];
            let mut replay = draws.into_iter();
            let noise = gumbel_noise(64, &mut || replay.next().unwrap());
            let cpu_scores = (logits
                .to_device(&Device::Cpu)
                .unwrap()
                .narrow(0, 0, 2048)
                .unwrap()
                / 0.8f64)
                .unwrap();
            let expected = gumbel_top_k_token(&cpu_scores, 64, &noise)
                .unwrap()
                .to_vec1::<u32>()
                .unwrap()[0];
            assert_eq!(token, expected, "seed {seed}");
            assert!(token < 2048);
        }
    }

    fn parity_logits(width: usize) -> Vec<f32> {
        (0..width)
            .map(|index| {
                let permuted = (index * 37) % width;
                permuted as f32 * 0.03125 + index as f32 * 0.000_001
            })
            .collect()
    }

    #[test]
    fn logprob_collection_reports_raw_softmax_and_top_k_ordering() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], 4, &device).unwrap();
        let config = ChatGenerationConfig {
            logprobs: true,
            top_logprobs: 3,
            ..ChatGenerationConfig::default()
        };
        let mut sampler = ChatSampler::new(config, &[]);
        let (token, logprobs) = sampler.sample_with_logprobs(&logits, 4).unwrap();
        assert_eq!(token, 3, "greedy argmax over raw logits");
        let logprobs = logprobs.expect("logprobs collected");
        let lse = (1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp() + 4.0f32.exp()).ln();
        assert!((logprobs.logprob - (4.0 - lse)).abs() < 1e-5);
        assert_eq!(
            logprobs.top.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![3, 2, 1]
        );
        for (index, (id, logprob)) in logprobs.top.iter().enumerate() {
            let expected_logit = 4.0 - index as f32;
            assert_eq!(*id as usize, 3 - index);
            assert!((logprob - (expected_logit - lse)).abs() < 1e-5);
        }
    }

    #[test]
    fn logprob_collection_preserves_sampled_tokens_exactly() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.1f32, 0.9, 0.3, 0.7], 4, &device).unwrap();
        let base = ChatGenerationConfig {
            temperature: 1.0,
            top_p: 1.0,
            seed: 97,
            ..ChatGenerationConfig::default()
        };
        let mut plain = ChatSampler::new(base.clone(), &[]);
        let mut measured = ChatSampler::new(
            ChatGenerationConfig {
                logprobs: true,
                top_logprobs: 2,
                ..base
            },
            &[],
        );
        for _ in 0..16 {
            let expected = plain.sample(&logits, 4).unwrap();
            let (actual, _) = measured.sample_with_logprobs(&logits, 4).unwrap();
            assert_eq!(
                actual, expected,
                "logprob collection must not move the draw"
            );
        }
    }

    #[test]
    fn resolve_token_logprob_decodes_single_token_surfaces() {
        let tokenizer = crate::tokenizer::Tokenizer::from_hf_json_bytes(
            br#"{
            "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":{"a":0,"b":1,"c":2,"d":3},"unk_token":"a"}
        }"#,
        )
        .unwrap();
        let raw = RawTokenLogprobs {
            token: 1,
            logprob: -0.5,
            top: vec![(1, -0.5), (0, -1.5)],
        };
        let resolved = resolve_token_logprob(&tokenizer, &raw).unwrap();
        assert_eq!(resolved.token, "b");
        assert_eq!(resolved.bytes, b"b".to_vec());
        assert_eq!(resolved.logprob, -0.5);
        assert_eq!(resolved.top_logprobs.len(), 2);
        assert_eq!(resolved.top_logprobs[0].token, "b");
        assert_eq!(resolved.top_logprobs[1].token, "a");
        assert_eq!(resolved.top_logprobs[1].logprob, -1.5);
    }

    #[test]
    fn chat_sampler_keeps_per_request_seed_and_policy_independent() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.0f32, 0.0, 0.0, 0.0], 4, &device).unwrap();
        let config = ChatGenerationConfig {
            temperature: 1.0,
            top_p: 1.0,
            seed: 17,
            ..ChatGenerationConfig::default()
        };
        let mut first = ChatSampler::new(config.clone(), &[]);
        let mut second = ChatSampler::new(config, &[]);
        let first_tokens = (0..8)
            .map(|_| first.sample(&logits, 4).unwrap())
            .collect::<Vec<_>>();
        let second_tokens = (0..8)
            .map(|_| second.sample(&logits, 4).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(first_tokens, second_tokens);
    }

    #[test]
    fn chat_sampler_applies_history_and_custom_stop_contract() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![4.0f32, 3.0], 2, &device).unwrap();
        let config = ChatGenerationConfig {
            temperature: 0.0,
            repetition_penalty: 2.0,
            stop_token_ids: vec![1],
            ..ChatGenerationConfig::default()
        };
        let mut sampler = ChatSampler::new(config, &[0]);
        let token = sampler.sample(&logits, 2).unwrap();
        assert_eq!(token, 1);
        assert!(sampler.is_configured_stop(token));
        assert_eq!(sampler.history(), &[0, 1]);
    }

    #[test]
    fn chat_sampler_selects_last_sequence_row() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![9.0f32, 0.0, 0.0, 1.0], (1, 2, 2), &device).unwrap();
        let mut sampler = ChatSampler::new(ChatGenerationConfig::default(), &[]);
        assert_eq!(sampler.sample(&logits, 2).unwrap(), 1);
    }

    #[test]
    fn chunked_topk_is_exact_across_chunk_boundaries() {
        let device = Device::Cpu;
        let mut input = (0..2500).map(|index| index as f32).collect::<Vec<_>>();
        input[3] = 9000.0;
        input[1027] = 8000.0;
        input[2499] = 7000.0;
        let tensor = Tensor::from_vec(input, 2500, &device).unwrap();
        let (values, indices) = bounded_topk(&tensor, 3).unwrap();
        assert_eq!(
            values.to_vec1::<f32>().unwrap(),
            vec![9000.0, 8000.0, 7000.0]
        );
        assert_eq!(indices.to_vec1::<u32>().unwrap(), vec![3, 1027, 2499]);
    }

    #[test]
    fn bounded_candidates_match_exact_host_sampling_with_penalties_and_mask() {
        let device = Device::Cpu;
        let width = 2_050;
        let mut values = parity_logits(width);
        values[17] = f32::NAN;
        values[1_027] = f32::INFINITY;
        let mut allowed = vec![true; width];
        allowed[2_049] = false;
        allowed[2_048] = false;
        let history = [2_047, 2_047, 1_500];
        let config = ChatGenerationConfig {
            temperature: 0.7,
            top_p: 0.82,
            top_k: 19,
            repetition_penalty: 1.7,
            presence_penalty: 0.35,
            ..ChatGenerationConfig::default()
        };
        let tensor = Tensor::from_vec(values.clone(), width, &device).unwrap();
        let candidates = bounded_sampling_candidates(
            &tensor,
            width,
            config.top_k,
            config.temperature,
            PenaltyHistory::uniform(&history),
            config.repetition_penalty,
            config.presence_penalty,
            Some(&allowed),
        )
        .unwrap()
        .expect("CPU semantic fixture should execute the device-tensor algorithm");

        for (index, value) in values.iter_mut().enumerate() {
            if !value.is_finite() || !allowed[index] {
                *value = f32::NEG_INFINITY;
            }
        }
        apply_chat_history_penalties(
            &mut values,
            PenaltyHistory::uniform(&history),
            config.repetition_penalty,
            config.presence_penalty,
        );
        for value in &mut values {
            *value /= config.temperature;
        }
        let mut expected_indices = values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| value.is_finite().then_some(index))
            .collect::<Vec<_>>();
        expected_indices.sort_by(|left, right| {
            values[*right]
                .partial_cmp(&values[*left])
                .unwrap_or(Ordering::Equal)
        });
        expected_indices.truncate(config.top_k);
        assert_eq!(
            candidates.indices,
            expected_indices
                .iter()
                .map(|index| *index as u32)
                .collect::<Vec<_>>()
        );
        for (actual, index) in candidates.values.iter().zip(expected_indices) {
            assert!((actual - values[index]).abs() < 1e-5);
        }
        assert!(candidates.logsumexp.is_none());

        let sampling_candidates = bounded_sampling_candidates(
            &tensor,
            width,
            config.top_k,
            config.temperature,
            PenaltyHistory::uniform(&history),
            config.repetition_penalty,
            config.presence_penalty,
            None,
        )
        .unwrap()
        .unwrap();
        for seed in 1..=64 {
            let mut host_rng = SimpleRng::new(seed);
            let host = sample_chat_token(&tensor, width, &config, &history, &mut host_rng).unwrap();
            let mut device_rng = SimpleRng::new(seed);
            let device =
                sample_device_candidates(&sampling_candidates, config.top_p, device_rng.next_f32())
                    .expect("explicit top-k candidates are complete");
            assert_eq!(device, host, "sampling diverged for seed {seed}");
        }
    }

    #[test]
    fn bounded_nucleus_fallback_and_host_read_telemetry_are_exact() {
        let device = Device::Cpu;
        let width = DEVICE_SAMPLING_CANDIDATE_LIMIT * 2;
        let logits = Tensor::zeros(width, DType::F32, &device).unwrap();
        let before_candidates = crate::models::shared::telemetry::snapshot();
        let candidates = bounded_sampling_candidates(
            &logits,
            width,
            0,
            1.0,
            PenaltyHistory::uniform(&[]),
            1.0,
            0.0,
            None,
        )
        .unwrap()
        .unwrap();
        let after_candidates = crate::models::shared::telemetry::snapshot();
        let packed_elements = DEVICE_SAMPLING_CANDIDATE_LIMIT * 2 + 1;
        assert!(after_candidates.host_read_ops_total > before_candidates.host_read_ops_total);
        assert!(
            after_candidates.host_read_bytes_total
                >= before_candidates.host_read_bytes_total + (packed_elements * 4) as u64
        );
        assert!(!device_candidates_cover_top_p(&candidates, 0.75));

        let config = ChatGenerationConfig {
            temperature: 1.0,
            top_p: 0.75,
            seed: 9,
            ..ChatGenerationConfig::default()
        };
        let before_fallback = crate::models::shared::telemetry::snapshot();
        ChatSampler::new(config, &[])
            .sample(&logits, width)
            .unwrap();
        let after_fallback = crate::models::shared::telemetry::snapshot();
        assert!(after_fallback.host_read_ops_total > before_fallback.host_read_ops_total);
        assert!(
            after_fallback.host_read_bytes_total
                >= before_fallback.host_read_bytes_total + (width * 4) as u64
        );

        assert!(
            bounded_device_sampling_candidates(&logits, width, 0, 1.0, &[], 1.0, 0.0, None,)
                .unwrap()
                .is_none()
        );
        assert!(
            bounded_cuda_sampling_candidates(&logits, width, 0, 1.0, &[], 1.0, 0.0, None,)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn greedy_sampling_records_only_the_scalars_materialized_on_host() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.5f32, 3.0, -2.0], 3, &device).unwrap();
        let before = crate::models::shared::telemetry::snapshot();
        let sampled = ChatSampler::new(ChatGenerationConfig::default(), &[])
            .sample(&logits, 3)
            .unwrap();
        let after = crate::models::shared::telemetry::snapshot();
        assert_eq!(sampled, 1);
        assert!(after.host_read_ops_total >= before.host_read_ops_total + 2);
        assert!(after.host_read_bytes_total >= before.host_read_bytes_total + 8);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_bounded_candidates_match_cpu_semantics_if_available() {
        let Some((metal, _lease)) = crate::backends::metal_test_device_or_skip(
            "metal_bounded_candidates_match_cpu_semantics_if_available",
        ) else {
            return;
        };
        let width = 257;
        let values = parity_logits(width);
        let mut allowed = vec![true; width];
        allowed[256] = false;
        let history = [255, 127];
        let cpu_tensor = Tensor::from_vec(values.clone(), width, &Device::Cpu).unwrap();
        let expected = bounded_sampling_candidates(
            &cpu_tensor,
            width,
            0,
            0.8,
            PenaltyHistory::uniform(&history),
            1.5,
            0.2,
            Some(&allowed),
        )
        .unwrap()
        .unwrap();
        let metal_tensor = Tensor::from_vec(values, width, &metal).unwrap();
        let actual = bounded_device_sampling_candidates(
            &metal_tensor,
            width,
            0,
            0.8,
            &history,
            1.5,
            0.2,
            Some(&allowed),
        )
        .unwrap()
        .expect("Metal is an eligible bounded-sampling backend");
        assert_eq!(actual.indices, expected.indices);
        for (actual, expected) in actual.values.iter().zip(&expected.values) {
            assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
        }
        assert!((actual.logsumexp.unwrap() - expected.logsumexp.unwrap()).abs() < 1e-4);
    }

    #[test]
    fn device_penalties_and_masks_match_sampling_contract() {
        let device = Device::Cpu;
        let values = Tensor::from_vec(vec![4.0f32, -2.0, 3.0, f32::NAN], 4, &device).unwrap();
        let values = sanitize_finite(&values).unwrap();
        let values =
            apply_history_penalties(&values, PenaltyHistory::uniform(&[0, 1, 1]), 2.0, 0.5)
                .unwrap();
        let actual = values.to_vec1::<f32>().unwrap();
        assert_eq!(actual[..3], [1.5, -4.5, 3.0]);
        assert_eq!(actual[3], f32::NEG_INFINITY);
    }

    #[test]
    fn split_penalty_histories_match_on_device_and_host() {
        // Repetition reads prompt + output, presence reads output only.
        let logits = vec![4.0f32, -2.0, 3.0, 1.0, f32::NEG_INFINITY];
        let history = [0u32, 1, 3, 1, 2];
        let split = PenaltyHistory {
            repetition: &history,
            presence: &history[3..],
        };
        // Token 0 and 3: repetition only. Token 1: both. Token 2: both.
        let expected = [2.0f32, -4.5, 1.0, 0.5, f32::NEG_INFINITY];

        let mut host = logits.clone();
        apply_chat_history_penalties(&mut host, split, 2.0, 0.5);
        assert_eq!(host, expected);

        let device = Device::Cpu;
        let values = Tensor::from_vec(logits.clone(), logits.len(), &device).unwrap();
        let on_device = apply_history_penalties(&values, split, 2.0, 0.5)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(on_device, expected);

        // A presence-only request touches only the generated tokens.
        let mut presence_only = logits.clone();
        apply_chat_history_penalties(&mut presence_only, split, 1.0, 0.5);
        assert_eq!(presence_only, [4.0, -2.5, 2.5, 1.0, f32::NEG_INFINITY]);
        let on_device = apply_history_penalties(&values, split, 1.0, 0.5)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(on_device, presence_only);
    }

    #[test]
    fn uniform_penalty_history_keeps_the_single_history_contract() {
        let history = [0u32, 2];
        assert!(PenaltyHistory::uniform(&history).is_uniform());
        let split = PenaltyHistory {
            repetition: &history,
            presence: &history[1..],
        };
        assert!(!split.is_uniform());

        let logits = vec![4.0f32, -2.0, 3.0];
        let mut uniform = logits.clone();
        apply_chat_history_penalties(&mut uniform, PenaltyHistory::uniform(&history), 2.0, 0.5);
        assert_eq!(uniform, [1.5, -2.0, 1.0]);
    }

    #[test]
    fn bounded_nucleus_falls_back_until_it_contains_the_exact_cutoff() {
        let complete_logsumexp = (1.0f32 + (-1.0f32).exp() + (-2.0f32).exp()).ln();
        let candidates = DeviceSamplingCandidates {
            values: vec![0.0, -1.0],
            indices: vec![7, 3],
            logsumexp: Some(complete_logsumexp),
        };
        assert!(device_candidates_cover_top_p(&candidates, 0.8));
        assert!(!device_candidates_cover_top_p(&candidates, 0.95));
        assert_eq!(sample_device_candidates(&candidates, 0.8, 0.0), Some(7));

        let explicit_top_k = DeviceSamplingCandidates {
            logsumexp: None,
            ..candidates
        };
        assert!(device_candidates_cover_top_p(&explicit_top_k, 1.0));
    }

    #[test]
    fn packed_candidate_readback_preserves_values_indices_and_normalizer() {
        let (values, indices, logsumexp) =
            unpack_sampling_readback(&[4.5, 3.0, 7.0, 1027.0, 5.25], 2, true).unwrap();
        assert_eq!(values, vec![4.5, 3.0]);
        assert_eq!(indices, vec![7, 1027]);
        assert_eq!(logsumexp, Some(5.25));
        assert!(unpack_sampling_readback(&[1.0], 2, false).is_err());
    }

    #[test]
    fn sampling_tensor_cache_reuses_equal_masks_and_histories() {
        DEVICE_SAMPLING_TENSOR_CACHE.with(|cache| cache.borrow_mut().clear());
        let device = Device::Cpu;
        let first =
            cached_sampling_tensor(&device, SamplingTensorCacheInput::Mask(vec![1, 0, 1])).unwrap();
        let second =
            cached_sampling_tensor(&device, SamplingTensorCacheInput::Mask(vec![1, 0, 1])).unwrap();
        assert_eq!(first.id(), second.id());
        let indices =
            cached_sampling_tensor(&device, SamplingTensorCacheInput::Indices(vec![3, 7])).unwrap();
        assert_eq!(indices.to_vec1::<u32>().unwrap(), vec![3, 7]);
        DEVICE_SAMPLING_TENSOR_CACHE.with(|cache| assert_eq!(cache.borrow().len(), 2));
    }
}
