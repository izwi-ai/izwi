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
/// The largest top-k the chunked device sort serves. Each merge round keeps
/// `k` values of every 1024-wide chunk, so a row of 1025 to 2048 values only
/// shrinks while `2k` is below its length: from `k = 513` a round can leave
/// the length unchanged and the rounds never end. Larger top-k sample on the
/// host.
pub(crate) const DEVICE_TOP_K_LIMIT: usize = DEVICE_SORT_CHUNK / 2;
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

    /// How [`sample_chat_rows`] reads this sampler's row of logits on
    /// `device`: the route [`Self::sample_with_logprobs`] would take. Device
    /// candidates are only requested here, so the step can plan every
    /// requesting row together.
    fn row_plan(&self, device: &Device) -> PlannedReadback<'_> {
        if self.grammar.is_some() || self.config.logprobs {
            return PlannedReadback::None;
        }
        if is_deterministic_greedy(&self.config) {
            return PlannedReadback::Greedy;
        }
        if !(device.is_cuda() || device.is_metal()) {
            return PlannedReadback::None;
        }
        PlannedReadback::Candidates(CandidateRequest {
            top_k: self.config.top_k,
            temperature: self.config.temperature,
            history: PenaltyHistory::uniform(&self.history),
            repetition_penalty: self.config.repetition_penalty,
            presence_penalty: self.config.presence_penalty,
        })
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
    let requests = samplers
        .iter()
        .map(|sampler| sampler.row_plan(logits.device()))
        .collect::<Vec<_>>();
    let candidates = plan_step_candidates(full_logits, vocab, &requests)?;
    let plans = requests
        .iter()
        .zip(candidates)
        .map(|(request, candidates)| match (request, candidates) {
            (PlannedReadback::Greedy, _) => RowSampling::Greedy,
            (_, Some(pending)) => RowSampling::Candidates(pending),
            _ => RowSampling::Host,
        })
        .collect::<Vec<_>>();
    let mut readback = ReadbackParts::default();
    if let Some(guard) = guard {
        readback.push(guard.to_dtype(DType::F32)?.flatten_all()?);
    }
    let any_greedy = plans.iter().any(|plan| matches!(plan, RowSampling::Greedy));
    let greedy_at = readback.len;
    if any_greedy {
        let ids = logits.argmax_keepdim(D::Minus1)?;
        // Gather reads contiguous rows; every id is inside the vocabulary.
        let picked = full_logits.contiguous()?.gather(&ids, D::Minus1)?;
        readback.push(ids.to_dtype(DType::F32)?.flatten_all()?);
        readback.push(picked.to_dtype(DType::F32)?.flatten_all()?);
    }
    let starts = plans
        .iter()
        .enumerate()
        .map(|(row, plan)| {
            Ok(match plan {
                RowSampling::Greedy => greedy_at,
                RowSampling::Candidates(pending) => readback.push_candidates(pending),
                RowSampling::Host => readback.push(logits.i(row)?.to_dtype(DType::F32)?),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let packed = readback.read()?;
    if guard.is_some() && packed.first().copied() == Some(0.0) {
        return Ok(None);
    }
    let mut sampled = Vec::with_capacity(rows);
    for (row, ((sampler, plan), start)) in samplers.iter_mut().zip(plans).zip(starts).enumerate() {
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
                let candidates = pending.unpack(readback_slice(&packed, start, pending.len())?)?;
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
            RowSampling::Host => sampler
                .sample_host_row(readback_slice(&packed, start, vocab)?.to_vec(), vocab_size)?,
        };
        sampler.record(token.0);
        sampled.push(token);
    }
    Ok(Some(sampled))
}

/// One step's device-to-host readback as it is assembled: every part once,
/// in order, and where each lands in the host copy. A batched planning
/// group's candidate pack is pushed by its first row and found by the rest.
#[derive(Default)]
struct ReadbackParts {
    parts: Vec<Tensor>,
    len: usize,
    packs: Vec<(candle_core::TensorId, usize)>,
}

impl ReadbackParts {
    /// Appends a rank-1 F32 `part` and returns where it starts.
    fn push(&mut self, part: Tensor) -> usize {
        let start = self.len;
        self.len += part.elem_count();
        self.parts.push(part);
        start
    }

    /// Where `pending`'s values start, pushing its pack on first use.
    fn push_candidates(&mut self, pending: &PendingCandidates) -> usize {
        let id = pending.packed.id();
        let start = match self.packs.iter().find(|(pushed, _)| *pushed == id) {
            Some(&(_, start)) => start,
            None => {
                let start = self.push(pending.packed.clone());
                self.packs.push((id, start));
                start
            }
        };
        start + pending.offset
    }

    /// Every part in one transfer.
    fn read(self) -> Result<Vec<f32>> {
        if self.parts.is_empty() {
            return Ok(Vec::new());
        }
        let packed = read_f32_values_to_host(&Tensor::cat(&self.parts, 0)?)?;
        if packed.len() != self.len {
            return Err(Error::InferenceError(
                "batched sampling readback has the wrong length".into(),
            ));
        }
        Ok(packed)
    }
}

/// The `len` values of a packed readback that start at `start`.
fn readback_slice(packed: &[f32], start: usize, len: usize) -> Result<&[f32]> {
    packed
        .get(start..start + len)
        .ok_or_else(|| Error::InferenceError("batched sampling readback is too short".into()))
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
    // `ranked_candidates` selects the top entries without sorting the whole
    // vocabulary, in the order a stable full sort gives (ties by index).
    let top = ranked_candidates(values, top_logprobs.max(1))
        .into_iter()
        .take(top_logprobs)
        .map(|index| (index, values[index]))
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
    if !plans_bounded_candidates(vocab, top_k, temperature) {
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

    let (values, indices) = bounded_topk(&adjusted, candidate_count(top_k, vocab))?;
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
        offset: 0,
        count: values.dim(0)?,
        has_logsumexp: logsumexp.is_some(),
    }))
}

/// Whether [`pending_sampling_candidates`] plans candidates for a row of
/// `vocab` in-vocabulary logits, rather than leaving the row to the host.
fn plans_bounded_candidates(vocab: usize, top_k: usize, temperature: f32) -> bool {
    if temperature <= 1e-5 {
        return false;
    }
    vocab != 0 && vocab <= MAX_EXACT_F32_INTEGER && top_k <= DEVICE_TOP_K_LIMIT
}

/// How many candidates a row keeps: its top-k, or the nucleus bound when
/// top-k is off, never more than the vocabulary.
fn candidate_count(top_k: usize, vocab: usize) -> usize {
    if top_k == 0 {
        DEVICE_SAMPLING_CANDIDATE_LIMIT
    } else {
        top_k
    }
    .min(vocab)
}

/// A row's request for bounded device candidates: what
/// [`pending_sampling_candidates`] takes besides the logits.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CandidateRequest<'a> {
    pub(crate) top_k: usize,
    pub(crate) temperature: f32,
    pub(crate) history: PenaltyHistory<'a>,
    pub(crate) repetition_penalty: f32,
    pub(crate) presence_penalty: f32,
}

/// What one row of a step's batched readback carries, as planned by a
/// family's own per-row sampler.
pub(crate) enum PlannedReadback<'a> {
    /// The row's argmax and the logit it selected.
    Greedy,
    /// The row's bounded device candidates, planned together with the
    /// step's other candidate rows.
    Candidates(CandidateRequest<'a>),
    /// Nothing: the row samples on its own (grammar, logprobs, host).
    None,
}

/// A row's share of the batched readback.
pub(crate) enum PlannedValues {
    Greedy {
        index: u32,
        logit: f32,
    },
    /// `None` when a candidate came back non-finite (host fallback).
    Candidates(Option<DeviceSamplingCandidates>),
    None,
}

/// Read every row's planned values of one step's `[rows, width]` logits in
/// a single device-to-host transfer: the argmax and selected logit of each
/// greedy row (over the first `vocab_size` columns), then the candidates of
/// every candidate row, planned together by [`plan_step_candidates`].
/// Families with their own per-row sampler draw from the result exactly as
/// that sampler would.
pub(crate) fn read_planned_rows(
    logits: &Tensor,
    vocab_size: usize,
    plans: Vec<PlannedReadback<'_>>,
) -> Result<Vec<PlannedValues>> {
    let (rows, width) = logits.dims2()?;
    if rows != plans.len() {
        return Err(Error::InvalidInput(format!(
            "batched readback planned {} rows for {rows} logits rows",
            plans.len()
        )));
    }
    let vocab = width.min(vocab_size);
    if vocab == 0 || vocab > MAX_EXACT_F32_INTEGER {
        return Err(Error::InvalidInput(format!(
            "batched readback cannot carry a {vocab}-token vocabulary"
        )));
    }
    let candidates = plan_step_candidates(logits, vocab, &plans)?;
    let greedy_rows = plans
        .iter()
        .enumerate()
        .filter(|(_, plan)| matches!(plan, PlannedReadback::Greedy))
        .map(|(row, _)| row as u32)
        .collect::<Vec<_>>();
    let mut readback = ReadbackParts::default();
    if !greedy_rows.is_empty() {
        let selected = Tensor::from_vec(greedy_rows.clone(), greedy_rows.len(), logits.device())?;
        let greedy = logits.narrow(1, 0, vocab)?.index_select(&selected, 0)?;
        let greedy = greedy.to_dtype(DType::F32)?.contiguous()?;
        let ids = greedy.argmax_keepdim(D::Minus1)?;
        let picked = greedy.gather(&ids, D::Minus1)?;
        readback.push(ids.to_dtype(DType::F32)?.flatten_all()?);
        readback.push(picked.flatten_all()?);
    }
    let starts = candidates
        .iter()
        .map(|pending| {
            pending
                .as_ref()
                .map(|pending| readback.push_candidates(pending))
        })
        .collect::<Vec<_>>();
    let packed = readback.read()?;
    let greedy_count = greedy_rows.len();
    let mut greedy_seen = 0usize;
    let mut values = Vec::with_capacity(plans.len());
    for ((plan, pending), start) in plans.iter().zip(candidates).zip(starts) {
        values.push(match (plan, pending, start) {
            (PlannedReadback::Greedy, _, _) => {
                let index = packed[greedy_seen] as u32;
                let logit = packed[greedy_count + greedy_seen];
                greedy_seen += 1;
                PlannedValues::Greedy { index, logit }
            }
            (_, Some(pending), Some(start)) => PlannedValues::Candidates(
                pending.unpack(readback_slice(&packed, start, pending.len())?)?,
            ),
            // A request outside the bounded contract samples on its own.
            _ => PlannedValues::None,
        });
    }
    Ok(values)
}

/// One row of [`read_masked_planned_rows`]: its plan over the first `vocab`
/// columns of its logits, of which it may pick only the `allowed` ones
/// (every one when `None`).
pub(crate) struct MaskedReadback<'a> {
    pub(crate) vocab: usize,
    pub(crate) allowed: Option<&'a [bool]>,
    pub(crate) plan: PlannedReadback<'a>,
}

/// [`read_planned_rows`] for rows that each read their own vocabulary
/// prefix under their own allowed mask, all in one device-to-host transfer.
/// A disallowed column reads as `-inf`: a greedy row's index is the argmax
/// of its masked prefix, and a candidate row's candidates are bit for bit
/// what [`bounded_device_sampling_candidates`] plans for it alone with that
/// `allowed_mask` (masking before the `-inf` sanitizing or after it gives
/// the same values). Rows with the same prefix and mask are planned
/// together. A row with an empty or too wide prefix reads nothing (`None`).
pub(crate) fn read_masked_planned_rows(
    logits: &Tensor,
    rows: Vec<MaskedReadback<'_>>,
) -> Result<Vec<PlannedValues>> {
    let (row_count, width) = logits.dims2()?;
    if row_count != rows.len() {
        return Err(Error::InvalidInput(format!(
            "masked readback planned {} rows for {row_count} logits rows",
            rows.len()
        )));
    }
    // Rows sharing a prefix and mask, in first-seen order.
    let mut groups: Vec<(usize, Option<&[bool]>, Vec<usize>)> = Vec::new();
    for (row, request) in rows.iter().enumerate() {
        let vocab = request.vocab.min(width);
        if matches!(request.plan, PlannedReadback::None)
            || vocab == 0
            || vocab > MAX_EXACT_F32_INTEGER
        {
            continue;
        }
        let allowed = match request.allowed {
            Some(mask) if mask.len() < vocab => {
                return Err(Error::InvalidInput(
                    "sampling mask is shorter than the selected vocabulary".into(),
                ))
            }
            // Allowing every column changes no value.
            Some(mask) if mask[..vocab].iter().all(|allowed| *allowed) => None,
            Some(mask) => Some(&mask[..vocab]),
            None => None,
        };
        match groups
            .iter_mut()
            .find(|(group_vocab, group_mask, _)| *group_vocab == vocab && *group_mask == allowed)
        {
            Some((_, _, members)) => members.push(row),
            None => groups.push((vocab, allowed, vec![row])),
        }
    }
    let device = logits.device();
    let mut readback = ReadbackParts::default();
    let mut greedy_at = vec![None; row_count];
    let mut pending = (0..row_count).map(|_| None).collect::<Vec<_>>();
    for (vocab, allowed, members) in &groups {
        let block = if members.iter().copied().eq(0..row_count) {
            logits.clone()
        } else {
            let ids = members.iter().map(|&row| row as u32).collect::<Vec<_>>();
            let ids = Tensor::from_vec(ids, members.len(), device)?;
            logits.contiguous()?.index_select(&ids, 0)?
        };
        let mut block = block.narrow(1, 0, *vocab)?.to_dtype(DType::F32)?;
        if let Some(allowed) = allowed {
            let mask = cached_sampling_tensor(
                device,
                SamplingTensorCacheInput::Mask(allowed.iter().map(|a| u8::from(*a)).collect()),
            )?;
            let negative = negative_infinity(device)?.broadcast_as(block.shape())?;
            block = mask
                .broadcast_as(block.shape())?
                .where_cond(&block, &negative)?;
        }
        let greedy = |row: &usize| matches!(rows[*row].plan, PlannedReadback::Greedy);
        if members.iter().any(greedy) {
            let block = block.contiguous()?;
            let ids = block.argmax_keepdim(D::Minus1)?;
            let picked = block.gather(&ids, D::Minus1)?;
            let ids_at = readback.push(ids.to_dtype(DType::F32)?.flatten_all()?);
            let picked_at = readback.push(picked.flatten_all()?);
            for (slot, row) in members.iter().enumerate().filter(|(_, row)| greedy(row)) {
                greedy_at[*row] = Some((ids_at + slot, picked_at + slot));
            }
        }
        let plans = members
            .iter()
            .map(|&row| match &rows[row].plan {
                PlannedReadback::Candidates(request) => PlannedReadback::Candidates(*request),
                _ => PlannedReadback::None,
            })
            .collect::<Vec<_>>();
        if plans
            .iter()
            .any(|plan| matches!(plan, PlannedReadback::Candidates(_)))
        {
            for (&row, planned) in members
                .iter()
                .zip(plan_step_candidates(&block, *vocab, &plans)?)
            {
                pending[row] = planned.map(|planned| {
                    let start = readback.push_candidates(&planned);
                    (planned, start)
                });
            }
        }
    }
    let packed = readback.read()?;
    rows.iter()
        .zip(greedy_at.into_iter().zip(pending))
        .map(|(request, (greedy, pending))| {
            Ok(match (&request.plan, greedy, pending) {
                (PlannedReadback::Greedy, Some((index, logit)), _) => PlannedValues::Greedy {
                    index: packed[index] as u32,
                    logit: packed[logit],
                },
                (PlannedReadback::Candidates(_), _, Some((pending, start))) => {
                    let values = readback_slice(&packed, start, pending.len())?;
                    PlannedValues::Candidates(pending.unpack(values)?)
                }
                _ => PlannedValues::None,
            })
        })
        .collect()
}

/// [`bounded_device_sampling_candidates`] on any device, the CPU included,
/// so a family's batched sampler can be checked against the single-row
/// accelerator path on the CPU.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn bounded_sampling_candidates_on_any_device(
    logits: &Tensor,
    vocab_limit: usize,
    top_k: usize,
    temperature: f32,
    history: &[u32],
    repetition_penalty: f32,
    presence_penalty: f32,
    allowed_mask: Option<&[bool]>,
) -> Result<Option<DeviceSamplingCandidates>> {
    if temperature <= 1e-5 {
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

/// One row's bounded candidates, still on the device as one packed F32
/// tensor (values, indices, then the normalizer when present), so a caller
/// can read many rows back in a single transfer.
pub(crate) struct PendingCandidates {
    /// The row's own pack, or its batched planning group's (one row after
    /// another), which the step reads back once for the whole group.
    packed: Tensor,
    /// Where this row's values start in `packed`.
    offset: usize,
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

/// The bit patterns of one row's planned pack, read without the sampling
/// telemetry (for the batched planner's self-check and its tests).
fn packed_bits(pending: &PendingCandidates) -> Result<Vec<u32>> {
    let packed = pending.packed.to_vec1::<f32>()?;
    Ok(readback_slice(&packed, pending.offset, pending.len())?
        .iter()
        .map(|value| value.to_bits())
        .collect())
}

/// The bounded candidates of every [`PlannedReadback::Candidates`] row of
/// one step's `[rows, width]` logits (over the first `vocab_limit` columns),
/// `None` where [`pending_sampling_candidates`] plans nothing for the row.
fn plan_step_candidates(
    logits: &Tensor,
    vocab_limit: usize,
    plans: &[PlannedReadback<'_>],
) -> Result<Vec<Option<PendingCandidates>>> {
    plan_step_candidates_with(logits, vocab_limit, plans, batched_device_sampling::admit)
}

/// [`plan_step_candidates`] with the batching switch passed in. Two or more
/// rows within the bounded contract are planned in one pass by
/// [`plan_batched_candidates`] when `batched` (asked only then, with the
/// logits' device and the row count) allows it; every other row is planned
/// alone, as before.
fn plan_step_candidates_with(
    logits: &Tensor,
    vocab_limit: usize,
    plans: &[PlannedReadback<'_>],
    batched: impl FnOnce(&Device, usize) -> bool,
) -> Result<Vec<Option<PendingCandidates>>> {
    let (rows, width) = logits.dims2()?;
    if rows != plans.len() {
        return Err(Error::InvalidInput(format!(
            "candidate planning received {} plans for {rows} logits rows",
            plans.len()
        )));
    }
    let vocab = width.min(vocab_limit);
    let batch = plans
        .iter()
        .enumerate()
        .filter_map(|(row, plan)| match plan {
            PlannedReadback::Candidates(request)
                if plans_bounded_candidates(vocab, request.top_k, request.temperature) =>
            {
                Some((row, *request))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    // Flattened penalty entries address the whole `[n, vocab]` block as U32.
    let addressable = batch
        .len()
        .checked_mul(vocab)
        .is_some_and(|len| len <= u32::MAX as usize);
    let mut planned = (0..rows).map(|_| None).collect::<Vec<_>>();
    if batch.len() >= 2 && addressable && batched(logits.device(), batch.len()) {
        for ((row, _), pending) in batch
            .iter()
            .zip(plan_batched_candidates(logits, vocab, &batch)?)
        {
            planned[*row] = Some(pending);
        }
    }
    for (row, plan) in plans.iter().enumerate() {
        if let (PlannedReadback::Candidates(request), None) = (plan, &planned[row]) {
            planned[row] = pending_sampling_candidates(
                &logits.i(row)?,
                vocab_limit,
                request.top_k,
                request.temperature,
                request.history,
                request.repetition_penalty,
                request.presence_penalty,
                None,
            )?;
        }
    }
    Ok(planned)
}

/// The candidates of `rows` (step row and request, each within the bounded
/// contract), in `rows` order, from one planning pass over their
/// `[n, vocab]` logits instead of one per row: one sanitize, two flattened
/// penalty passes, one temperature affine per run of equal temperatures,
/// then per group of equal candidate count and normalizer one row-wise
/// [`bounded_topk_rows`] and one pack. Every op is elementwise or row-wise
/// with the per-row path's arithmetic, so each row's pack is bit-identical
/// to what [`pending_sampling_candidates`] plans for it alone.
fn plan_batched_candidates(
    logits: &Tensor,
    vocab: usize,
    rows: &[(usize, CandidateRequest<'_>)],
) -> Result<Vec<PendingCandidates>> {
    let device = logits.device();
    let group_of = |request: &CandidateRequest<'_>| {
        (candidate_count(request.top_k, vocab), request.top_k == 0)
    };
    // Slots order the rows so every group, and every temperature inside a
    // group, is one contiguous run of rows.
    let mut slots = (0..rows.len()).collect::<Vec<_>>();
    slots.sort_by_key(|&index| {
        let (row, request) = &rows[index];
        (
            group_of(request),
            request.temperature.max(1e-5).to_bits(),
            *row,
        )
    });
    let selected = slots.iter().map(|&index| rows[index].0).collect::<Vec<_>>();
    let logits = if selected.iter().copied().eq(0..logits.dim(0)?) {
        logits.clone()
    } else {
        let ids = selected.iter().map(|&row| row as u32).collect::<Vec<_>>();
        let ids = Tensor::from_vec(ids, selected.len(), device)?;
        logits.contiguous()?.index_select(&ids, 0)?
    };
    let requests = slots
        .iter()
        .map(|&index| &rows[index].1)
        .collect::<Vec<_>>();
    let adjusted = sanitize_finite(&logits.narrow(1, 0, vocab)?.to_dtype(DType::F32)?)?;
    let adjusted = apply_row_history_penalties(&adjusted, &requests)?;

    let mut planned = (0..rows.len()).map(|_| None).collect::<Vec<_>>();
    let mut start = 0;
    while start < requests.len() {
        let group = group_of(requests[start]);
        let len = requests[start..]
            .iter()
            .take_while(|request| group_of(request) == group)
            .count();
        let (count, has_logsumexp) = group;
        let scaled = scale_rows_by_temperature(
            &adjusted.narrow(0, start, len)?,
            &requests[start..start + len],
        )?;
        let (values, indices) = bounded_topk_rows(&scaled, count)?;
        let mut parts = vec![values, indices.to_dtype(DType::F32)?];
        if has_logsumexp {
            parts.push(scaled.log_sum_exp(D::Minus1)?.reshape((len, 1))?);
        }
        let packed = Tensor::cat(&parts, 1)?.flatten_all()?;
        let stride = count * 2 + usize::from(has_logsumexp);
        for (slot, &index) in slots[start..start + len].iter().enumerate() {
            planned[index] = Some(PendingCandidates {
                packed: packed.clone(),
                offset: slot * stride,
                count,
                has_logsumexp,
            });
        }
        start += len;
    }
    planned
        .into_iter()
        .map(|pending| {
            pending.ok_or_else(|| Error::InferenceError("batched planning lost a row".into()))
        })
        .collect()
}

/// The tokens each penalty reads for one row, as [`apply_history_penalties`]
/// decides: `(repetition, presence)`, `None` where that penalty leaves the
/// row alone. A uniform history runs both penalties behind one shared guard;
/// a split history runs each on its own, so a repetition penalty within
/// `f32::EPSILON` of 1 still applies beside a presence penalty only when the
/// history is uniform.
fn penalized_tokens<'a>(
    history: PenaltyHistory<'a>,
    repetition_penalty: f32,
    presence_penalty: f32,
) -> (Option<&'a [u32]>, Option<&'a [u32]>) {
    let repeats = repetition_penalty > 1.0;
    let presents = presence_penalty.abs() > f32::EPSILON;
    if history.is_uniform() {
        let tokens = history.repetition;
        let neutral = (repetition_penalty - 1.0).abs() <= f32::EPSILON && !presents;
        if tokens.is_empty() || neutral {
            return (None, None);
        }
        return (repeats.then_some(tokens), presents.then_some(tokens));
    }
    let repetition = (repeats
        && (repetition_penalty - 1.0).abs() > f32::EPSILON
        && !history.repetition.is_empty())
    .then_some(history.repetition);
    let presence = (presents && !history.presence.is_empty()).then_some(history.presence);
    (repetition, presence)
}

/// [`apply_history_penalties`] for every row of `values` (`[n, vocab]`, row
/// `slot` under `requests[slot]`) in at most two passes over the flattened
/// block: repetition, then presence. Rows never share an entry, and every
/// entry rounds as the per-row scalar affines round it: `x * (1 / penalty)`
/// for a positive `x` (the affine's `+ 0` changes nothing there),
/// `x * penalty + 0` otherwise (the `+ 0` turns a `-0` product into `+0` as
/// the affine does), and `x + (-presence)`, which is `x * 1 - presence`.
fn apply_row_history_penalties(
    values: &Tensor,
    requests: &[&CandidateRequest<'_>],
) -> Result<Tensor> {
    let (rows, vocab) = values.dims2()?;
    let device = values.device();
    let (mut repeated_ids, mut inverses, mut factors) = (Vec::new(), Vec::new(), Vec::new());
    let (mut present_ids, mut negated) = (Vec::new(), Vec::new());
    for (slot, request) in requests.iter().enumerate() {
        let (repeated, present) = penalized_tokens(
            request.history,
            request.repetition_penalty,
            request.presence_penalty,
        );
        let base = (slot * vocab) as u32;
        let repeated_tokens = repeated.map(|tokens| distinct_in_vocab(tokens, vocab));
        if let Some(tokens) = &repeated_tokens {
            let inverse = (1.0 / request.repetition_penalty as f64) as f32;
            repeated_ids.extend(tokens.iter().map(|token| base + token));
            inverses.extend(std::iter::repeat_n(inverse, tokens.len()));
            factors.extend(std::iter::repeat_n(
                request.repetition_penalty,
                tokens.len(),
            ));
        }
        if let Some(history) = present {
            // Over a uniform history both penalties read the same tokens.
            let tokens = match (repeated, repeated_tokens) {
                (Some(repeated), Some(tokens)) if std::ptr::eq(repeated, history) => tokens,
                _ => distinct_in_vocab(history, vocab),
            };
            present_ids.extend(tokens.iter().map(|token| base + token));
            negated.extend(std::iter::repeat_n(-request.presence_penalty, tokens.len()));
        }
    }
    let mut flat = values.flatten_all()?;
    if !repeated_ids.is_empty() {
        let len = repeated_ids.len();
        let ids = Tensor::from_vec(repeated_ids, len, device)?;
        let selected = flat.index_select(&ids, 0)?;
        let positive = selected.gt(0.0f64)?;
        let divided = selected.mul(&Tensor::from_vec(inverses, len, device)?)?;
        let multiplied = (selected.mul(&Tensor::from_vec(factors, len, device)?)? + 0.0)?;
        flat = flat.scatter(&ids, &positive.where_cond(&divided, &multiplied)?, 0)?;
    }
    if !present_ids.is_empty() {
        let len = present_ids.len();
        let ids = Tensor::from_vec(present_ids, len, device)?;
        let selected = flat.index_select(&ids, 0)?;
        let penalized = selected.add(&Tensor::from_vec(negated, len, device)?)?;
        flat = flat.scatter(&ids, &penalized, 0)?;
    }
    flat.reshape((rows, vocab)).map_err(Error::from)
}

/// Each row of `values` divided by its request's temperature through the
/// per-row path's own scalar affine, one launch per run of equal
/// temperatures. Candle's affine is `x * mul + add` (one fused multiply-add
/// on Metal, and on CUDA as nvcc contracts it), so a broadcast multiply by
/// per-row factors could differ from it in the sign of a zero result; the
/// same affine keeps every bit.
fn scale_rows_by_temperature(
    values: &Tensor,
    requests: &[&CandidateRequest<'_>],
) -> Result<Tensor> {
    let mut runs = Vec::new();
    let mut start = 0;
    while start < requests.len() {
        let temperature = requests[start].temperature.max(1e-5);
        let len = requests[start..]
            .iter()
            .take_while(|request| request.temperature.max(1e-5).to_bits() == temperature.to_bits())
            .count();
        runs.push((values.narrow(0, start, len)? / temperature as f64)?);
        start += len;
    }
    Tensor::cat(&runs, 0).map_err(Error::from)
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
    if temperature <= 1e-5 || top_k == 0 || top_k > DEVICE_TOP_K_LIMIT {
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

/// [`resident_top_k_sample`] for every row of `logits` (`[rows, vocab]`) in
/// one pass, all rows sharing `top_k` and `temperature`: the same
/// sanitizing and scaling, row-wise top-k ([`bounded_topk_rows`] equals
/// [`bounded_topk`] row by row), each row's Gumbel noise from its own
/// `draws` closure in the same order, and one argmax per row. Each row's code
/// is the one [`resident_top_k_sample`] picks for that row alone, and the
/// noise for all rows goes up in one upload instead of one per row. Returns
/// a `[rows]` U32 tensor, or `None` where the single-row sampler would (and
/// the caller falls back to it row by row).
pub fn resident_top_k_sample_rows<'d>(
    logits: &Tensor,
    top_k: usize,
    temperature: f32,
    draws: &mut [&mut (dyn FnMut() -> f32 + 'd)],
) -> Result<Option<Tensor>> {
    if temperature <= 1e-5 || top_k == 0 || top_k > DEVICE_TOP_K_LIMIT {
        return Ok(None);
    }
    let (rows, vocab) = logits.dims2()?;
    if rows == 0 || vocab == 0 || draws.len() != rows {
        return Ok(None);
    }
    let adjusted = sanitize_finite(&logits.to_dtype(DType::F32)?)?;
    let adjusted = (adjusted / temperature as f64)?;
    let k = top_k.min(vocab);
    let (values, indices) = bounded_topk_rows(&adjusted, k)?;
    let keep = values.dim(1)?;
    let mut noise = Vec::with_capacity(rows * keep);
    for draw in draws.iter_mut() {
        noise.extend_from_slice(&gumbel_noise(k, &mut **draw)[..keep]);
    }
    let noise = Tensor::from_vec(noise, (rows, keep), logits.device())?;
    let picks = values
        .broadcast_add(&noise)?
        .argmax(D::Minus1)?
        .to_dtype(DType::U32)?
        .reshape((rows, 1))?;
    Ok(Some(
        indices
            .to_dtype(DType::U32)?
            .contiguous()?
            .gather(&picks, 1)?
            .reshape(rows)?,
    ))
}

/// [`resident_top_k_sample_rows`] with every row's Gumbel noise already on
/// the logits' device, `[rows, top_k.min(vocab)]` (each row's
/// [`gumbel_noise`] of that many draws, as that function draws it), and the
/// column ids `0..vocab` (U32) passed in: the same arithmetic with nothing
/// uploaded or cached, so a CUDA graph can capture it and take the noise and
/// ids as inputs. `None` where that function returns `None`.
pub fn resident_top_k_rows_with_noise(
    logits: &Tensor,
    top_k: usize,
    temperature: f32,
    noise: &Tensor,
    column_ids: &Tensor,
) -> Result<Option<Tensor>> {
    if temperature <= 1e-5 || top_k == 0 || top_k > DEVICE_TOP_K_LIMIT {
        return Ok(None);
    }
    let (rows, vocab) = logits.dims2()?;
    if rows == 0 || vocab == 0 {
        return Ok(None);
    }
    let adjusted = sanitize_finite(&logits.to_dtype(DType::F32)?)?;
    let adjusted = (adjusted / temperature as f64)?;
    let (values, indices) = bounded_topk_rows_from(&adjusted, top_k.min(vocab), Some(column_ids))?;
    if noise.dims() != values.dims() {
        return Err(Error::InvalidInput(format!(
            "top-k noise {:?} does not match the candidates {:?}",
            noise.dims(),
            values.dims()
        )));
    }
    let picks = values
        .broadcast_add(noise)?
        .argmax(D::Minus1)?
        .to_dtype(DType::U32)?
        .reshape((rows, 1))?;
    Ok(Some(
        indices
            .to_dtype(DType::U32)?
            .contiguous()?
            .gather(&picks, 1)?
            .reshape(rows)?,
    ))
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
    if k == 0 || k > DEVICE_TOP_K_LIMIT {
        return Ok(None);
    }
    bounded_topk(values, k).map(Some)
}

fn bounded_topk(values: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    let length = values.dim(0)?;
    if length == 0 || length > u32::MAX as usize || k == 0 || k > DEVICE_TOP_K_LIMIT {
        return Err(Error::InvalidInput(format!(
            "bounded top-k requires a U32-addressable non-empty row and 1..={DEVICE_TOP_K_LIMIT} candidates"
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

/// [`bounded_topk`] of every row of `values` (`[rows, length]`) at once:
/// the same 1024-wide chunk sorts, `-inf`/`u32::MAX` padding at each row's
/// end and merge rounds, with all rows' chunks in one sort per round. Sorts
/// and gathers work row by row, so each row's result equals `bounded_topk`
/// of that row alone, ties included.
fn bounded_topk_rows(values: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    bounded_topk_rows_from(values, k, None)
}

/// [`bounded_topk_rows`] whose column ids come from `indices` (`0..length`
/// as U32) when given, not from the per-thread tensor cache: a CUDA graph
/// takes them as an input, since a cached tensor evicted after the capture
/// would free memory the graph still reads.
fn bounded_topk_rows_from(
    values: &Tensor,
    k: usize,
    indices: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (rows, length) = values.dims2()?;
    if rows == 0 || length == 0 || length > u32::MAX as usize || k == 0 || k > DEVICE_TOP_K_LIMIT {
        return Err(Error::InvalidInput(format!(
            "bounded top-k requires U32-addressable non-empty rows and 1..={DEVICE_TOP_K_LIMIT} candidates"
        )));
    }
    let device = values.device();
    let mut candidate_values = values.contiguous()?;
    let indices = match indices {
        Some(indices) if indices.dims() == [length] && indices.dtype() == DType::U32 => {
            indices.clone()
        }
        Some(indices) => {
            return Err(Error::InvalidInput(format!(
                "bounded top-k column ids {:?} {:?} do not cover {length} columns",
                indices.dims(),
                indices.dtype()
            )))
        }
        None => cached_sampling_tensor(device, SamplingTensorCacheInput::Arange(length as u32))?,
    };
    let mut candidate_indices = indices
        .unsqueeze(0)?
        .broadcast_as((rows, length))?
        .contiguous()?;
    loop {
        let length = candidate_values.dim(1)?;
        if length <= DEVICE_SORT_CHUNK {
            let (sorted, order) = candidate_values.sort_last_dim(false)?;
            let keep = k.min(length);
            return Ok((
                sorted.narrow(1, 0, keep)?,
                candidate_indices.gather(&order, 1)?.narrow(1, 0, keep)?,
            ));
        }

        let chunks = length.div_ceil(DEVICE_SORT_CHUNK);
        let padded_len = chunks * DEVICE_SORT_CHUNK;
        if padded_len > length {
            let padding = padded_len - length;
            candidate_values = Tensor::cat(
                &[
                    &candidate_values,
                    &Tensor::full(f32::NEG_INFINITY, (rows, padding), device)?,
                ],
                1,
            )?;
            candidate_indices = Tensor::cat(
                &[
                    &candidate_indices,
                    &Tensor::full(u32::MAX, (rows, padding), device)?,
                ],
                1,
            )?;
        }
        let values_2d = candidate_values.reshape((rows * chunks, DEVICE_SORT_CHUNK))?;
        let indices_2d = candidate_indices.reshape((rows * chunks, DEVICE_SORT_CHUNK))?;
        let (sorted, order) = values_2d.sort_last_dim(false)?;
        let keep = k.min(DEVICE_SORT_CHUNK);
        candidate_values = sorted.narrow(1, 0, keep)?.reshape((rows, chunks * keep))?;
        candidate_indices = indices_2d
            .gather(&order, 1)?
            .narrow(1, 0, keep)?
            .reshape((rows, chunks * keep))?;
    }
}

/// Batched planning of a step's device candidates
/// ([`plan_batched_candidates`]): its kill switch, a one-time bitwise check
/// against per-row planning, and counters for the family diagnostics.
pub(crate) mod batched_device_sampling {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::sync::OnceLock;

    use candle_core::{Device, Tensor};

    use super::{
        packed_bits, plan_step_candidates_with, CandidateRequest, PenaltyHistory, PlannedReadback,
    };
    use crate::{Error, Result};

    /// Kill switch: `false` plans each row's candidates on its own.
    pub(crate) const ENV: &str = "IZWI_BATCHED_DEVICE_SAMPLING";

    static RESOLVED: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    static STEPS: AtomicU64 = AtomicU64::new(0);
    static ROWS: AtomicU64 = AtomicU64::new(0);

    /// Whether a step plans its `rows` candidate rows together (counting it
    /// when it does), resolved once, on the first device that asks.
    pub(super) fn admit(device: &Device, rows: usize) -> bool {
        let admitted = RESOLVED.get_or_init(|| resolve(device)).is_ok();
        if admitted {
            STEPS.fetch_add(1, Relaxed);
            ROWS.fetch_add(rows as u64, Relaxed);
        }
        admitted
    }

    pub(super) fn resolve(device: &Device) -> std::result::Result<(), String> {
        if !crate::env::flag_or(ENV, true) {
            return Err(format!("{ENV}=false"));
        }
        self_check(device).map_err(|error| {
            tracing::warn!(
                %error,
                "batched device sampling self-check failed; planning candidate rows one at a time"
            );
            format!("self-check failed: {error}")
        })
    }

    /// Three rows of coarse logits (ties everywhere) with NaN, ±inf and
    /// signed zeros at penalized tokens, a vocabulary narrower than the
    /// logits, split and uniform histories, planned in mixed groups (top-k
    /// 20, 1 and 0, so the rows are reordered) and in one group (all top-k
    /// 0, three temperatures): every batched pack must equal its per-row
    /// pack bit for bit.
    pub(super) fn self_check(device: &Device) -> Result<()> {
        const ROWS: usize = 3;
        const WIDTH: usize = 5_003;
        const VOCAB: usize = 5_000;
        let logits = (0..ROWS * WIDTH)
            .map(|index| match index % WIDTH {
                5 => f32::NAN,
                6 => f32::INFINITY,
                7 => f32::NEG_INFINITY,
                8 => -0.0,
                9 => 0.0,
                _ => ((index * 7_919) % 37) as f32 * 0.25 - 4.5,
            })
            .collect::<Vec<_>>();
        let logits = Tensor::from_vec(logits, (ROWS, WIDTH), device)?;
        let history = [
            5u32, 6, 7, 8, 9, 13, 1_027, 4_999, 5_001, 13, 2_048, 8, 3_100, 9,
        ];
        let histories = [
            PenaltyHistory::uniform(&history),
            PenaltyHistory {
                repetition: &history,
                presence: &history[8..],
            },
            PenaltyHistory::uniform(&history[8..]),
        ];
        let temperatures = [0.7, 1.0, 1.3];
        let repetition = [1.3, 1.2, 1.0];
        let presence = [0.4, -0.3, 0.6];
        for top_ks in [[20, 1, 0], [0, 0, 0]] {
            let plans = (0..ROWS)
                .map(|row| {
                    PlannedReadback::Candidates(CandidateRequest {
                        top_k: top_ks[row],
                        temperature: temperatures[row],
                        history: histories[row],
                        repetition_penalty: repetition[row],
                        presence_penalty: presence[row],
                    })
                })
                .collect::<Vec<_>>();
            let batched = plan_step_candidates_with(&logits, VOCAB, &plans, |_, _| true)?;
            let per_row = plan_step_candidates_with(&logits, VOCAB, &plans, |_, _| false)?;
            for (row, (batched, per_row)) in batched.iter().zip(&per_row).enumerate() {
                let (Some(batched), Some(per_row)) = (batched, per_row) else {
                    return Err(Error::InferenceError(format!(
                        "row {row} was not planned on both paths"
                    )));
                };
                if packed_bits(batched)? != packed_bits(per_row)? {
                    return Err(Error::InferenceError(format!(
                        "batched candidates of row {row} (top-k {}) differ from per-row planning",
                        top_ks[row]
                    )));
                }
            }
        }
        Ok(())
    }

    /// The resolution and how many steps and rows planned together, for the
    /// admin diagnostics.
    pub(crate) fn summary() -> serde_json::Value {
        let (backend, reason) = match RESOLVED.get() {
            None => ("unresolved", None),
            Some(Ok(())) => ("batched", None),
            Some(Err(reason)) => ("per_row", Some(reason.as_str())),
        };
        serde_json::json!({
            "backend": backend,
            "reason": reason,
            "steps": STEPS.load(Relaxed),
            "rows": ROWS.load(Relaxed),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(seed: u64) -> impl FnMut() -> f32 {
        let mut state = seed | 1;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    /// Each row of the batched sampler picks the code the single-row sampler
    /// picks for that row with the same noise stream, at widths that take one
    /// sort chunk and several (the predictor's 2,048-code heads), with ties.
    #[test]
    fn resident_top_k_rows_match_each_row_alone() {
        let device = Device::Cpu;
        for (rows, vocab, top_k) in [(1, 64, 8), (5, 2048, 50), (8, 3000, 1), (3, 700, 512)] {
            let mut values = xorshift(vocab as u64 * 31 + rows as u64);
            let mut logits: Vec<f32> = (0..rows * vocab).map(|_| values() * 8.0 - 4.0).collect();
            // A tie inside the candidates of row 0.
            logits[3] = logits[5];
            let logits = Tensor::from_vec(logits, (rows, vocab), &device).unwrap();
            let mut batched_draws = (0..rows).map(|row| xorshift(row as u64 + 7)).collect::<Vec<_>>();
            let mut draws = batched_draws
                .iter_mut()
                .map(|draw| draw as &mut dyn FnMut() -> f32)
                .collect::<Vec<_>>();
            let batched = resident_top_k_sample_rows(&logits, top_k, 0.9, &mut draws)
                .unwrap()
                .expect("sampled on the device")
                .to_vec1::<u32>()
                .unwrap();
            for (row, &code) in batched.iter().enumerate() {
                let mut draw = xorshift(row as u64 + 7);
                let alone = resident_top_k_sample(&logits.i(row).unwrap(), vocab, top_k, 0.9, &mut draw)
                    .unwrap()
                    .expect("sampled on the device")
                    .to_vec1::<u32>()
                    .unwrap();
                assert_eq!(code, alone[0], "rows={rows} vocab={vocab} top_k={top_k} row={row}");
            }
        }
    }

    /// Noise drawn up front and passed in picks what the drawing sampler
    /// picks for every row, at one and several sort chunks.
    #[test]
    fn resident_top_k_rows_with_noise_match_the_drawing_sampler() {
        let device = Device::Cpu;
        for (rows, vocab, top_k) in [(4, 64, 8), (6, 2048, 50), (3, 700, 512), (2, 30, 50)] {
            let mut values = xorshift(vocab as u64 * 17 + rows as u64);
            let logits: Vec<f32> = (0..rows * vocab).map(|_| values() * 8.0 - 4.0).collect();
            let logits = Tensor::from_vec(logits, (rows, vocab), &device).unwrap();
            let mut streams = (0..rows)
                .map(|row| xorshift(row as u64 + 11))
                .collect::<Vec<_>>();
            let mut draws = streams
                .iter_mut()
                .map(|draw| draw as &mut dyn FnMut() -> f32)
                .collect::<Vec<_>>();
            let drawn = resident_top_k_sample_rows(&logits, top_k, 1.1, &mut draws)
                .unwrap()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();
            let keep = top_k.min(vocab);
            let noise = (0..rows)
                .flat_map(|row| gumbel_noise(keep, &mut xorshift(row as u64 + 11)))
                .collect::<Vec<_>>();
            let noise = Tensor::from_vec(noise, (rows, keep), &device).unwrap();
            let ids = Tensor::arange(0u32, vocab as u32, &device).unwrap();
            let passed = resident_top_k_rows_with_noise(&logits, top_k, 1.1, &noise, &ids)
                .unwrap()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap();
            assert_eq!(passed, drawn, "rows={rows} vocab={vocab} top_k={top_k}");
        }
    }

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

    /// Rows that share batched planning groups: three top-k 40 rows at two
    /// temperatures (one penalized) and two nucleus-only rows.
    fn grouped_samplers() -> Vec<ChatSampler> {
        let history = [3, 17, 17, 905];
        [
            sampled(0.9, 40, 1.0, 11),
            sampled(0.9, 40, 0.95, 12),
            ChatGenerationConfig {
                repetition_penalty: 1.3,
                presence_penalty: 0.5,
                ..sampled(0.6, 40, 1.0, 13)
            },
            sampled(1.1, 0, 0.9, 14),
            sampled(0.8, 0, 0.5, 15),
        ]
        .into_iter()
        .map(|config| ChatSampler::new(config, &history))
        .collect()
    }

    fn assert_batched_sampling_parity(device: &Device) {
        assert_batched_rows_match_single_rows(device, mixed_samplers(true), 1_500, 1_400, 24);
        assert_batched_rows_match_single_rows(device, grouped_samplers(), 1_500, 1_400, 24);
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

    /// Coarse logits (ties everywhere) with NaN, ±inf, signed zeros and
    /// subnormals on penalized tokens, and a last row that is `-inf` but for
    /// a few tokens, so its top-k holds `-inf` and padding candidates.
    fn planning_logits(rows: usize, width: usize) -> Vec<f32> {
        (0..rows * width)
            .map(|index| {
                let (row, token) = (index / width, index % width);
                if row == rows - 1 && token % 997 != 3 {
                    return f32::NEG_INFINITY;
                }
                match token {
                    5 => f32::NAN,
                    6 => f32::INFINITY,
                    7 => f32::NEG_INFINITY,
                    8 => -0.0,
                    9 => 0.0,
                    10 => -1.0e-40,
                    11 => 1.0e-40,
                    _ => ((index * 7_919 + row * 13) % 41) as f32 * 0.25 - 5.0,
                }
            })
            .collect()
    }

    fn candidates(
        top_k: usize,
        temperature: f32,
        history: PenaltyHistory<'_>,
        repetition_penalty: f32,
        presence_penalty: f32,
    ) -> PlannedReadback<'_> {
        PlannedReadback::Candidates(CandidateRequest {
            top_k,
            temperature,
            history,
            repetition_penalty,
            presence_penalty,
        })
    }

    /// Plans `plans` over [`planning_logits`] in one batched pass and row by
    /// row: the same rows are planned, the per-row packs are each row's own,
    /// and every batched pack equals its per-row pack to the bit. Returns
    /// the batched plan.
    fn assert_batched_packs_match_per_row(
        device: &Device,
        width: usize,
        vocab: usize,
        plans: &[PlannedReadback<'_>],
    ) -> Vec<Option<PendingCandidates>> {
        let logits = Tensor::from_vec(
            planning_logits(plans.len(), width),
            (plans.len(), width),
            device,
        )
        .unwrap();
        let batched = plan_step_candidates_with(&logits, vocab, plans, |_, _| true).unwrap();
        let per_row = plan_step_candidates_with(&logits, vocab, plans, |_, _| false).unwrap();
        for (row, (batched, per_row)) in batched.iter().zip(&per_row).enumerate() {
            match (batched, per_row) {
                (None, None) => {}
                (Some(batched), Some(per_row)) => {
                    assert_eq!(per_row.offset, 0, "row {row}");
                    assert_eq!(per_row.packed.elem_count(), per_row.len(), "row {row}");
                    assert_eq!(
                        (batched.count, batched.has_logsumexp),
                        (per_row.count, per_row.has_logsumexp),
                        "row {row}"
                    );
                    assert_eq!(
                        packed_bits(batched).unwrap(),
                        packed_bits(per_row).unwrap(),
                        "row {row} on {:?}",
                        device.location()
                    );
                }
                _ => panic!("row {row}: only one path planned it"),
            }
        }
        batched
    }

    /// The batched planner's self-check, then three steps: mixed groups and
    /// temperatures over a vocabulary narrower than the logits, with rows
    /// that are not batched in between; rows already in group order (no row
    /// gather); and a ~70k vocabulary that takes three or more sort rounds.
    fn assert_batched_planning(device: &Device) {
        batched_device_sampling::self_check(device).unwrap();
        let prompt = [
            5u32, 6, 7, 8, 9, 10, 11, 17, 17, 1_027, 2_047, 2_049, 2_060, 905,
        ];
        let uniform = PenaltyHistory::uniform(&prompt);
        let split = PenaltyHistory {
            repetition: &prompt,
            presence: &prompt[9..],
        };
        // Within `f32::EPSILON` of 1: applied beside a presence penalty only
        // over a uniform history.
        let near_one = 1.0 + f32::EPSILON;
        let mixed = [
            PlannedReadback::Greedy,
            candidates(20, 0.7, uniform, 1.3, 0.4),
            candidates(20, 0.7, split, 1.2, -0.3),
            PlannedReadback::None,
            candidates(20, 1.1, uniform, near_one, 0.5),
            candidates(0, 0.9, split, near_one, 0.5),
            candidates(0, 1.3, PenaltyHistory::uniform(&[]), 1.5, 0.0),
            candidates(1, 1.0, uniform, 0.8, 0.0),
            // Outside the bounded contract: planned by neither path.
            candidates(2_000, 0.8, uniform, 1.3, 0.0),
            candidates(20, 0.0, uniform, 1.3, 0.0),
            candidates(512, 0.5, uniform, 1.0, 0.2),
            candidates(20, 1.1, split, 1.0, 0.0),
            candidates(20, 0.6, uniform, 1.4, 0.0),
        ];
        let planned = assert_batched_packs_match_per_row(device, 2_100, 2_050, &mixed);
        let pack = |row: usize| planned[row].as_ref().unwrap().packed.id();
        assert!([2, 4, 11, 12].into_iter().all(|row| pack(row) == pack(1)));
        assert_eq!(pack(5), pack(6));
        assert_ne!(pack(1), pack(5));
        assert!(planned[8].is_none() && planned[9].is_none());

        let in_order = [
            candidates(1, 1.0, uniform, 1.3, 0.0),
            candidates(20, 0.7, split, 1.2, 0.4),
            candidates(20, 0.7, uniform, 1.0, 0.0),
            candidates(0, 1.2, uniform, 1.1, -0.2),
        ];
        assert_batched_packs_match_per_row(device, 1_000, 1_000, &in_order);

        let wide_prompt = [5u32, 8, 10, 1_023, 1_024, 40_000, 69_997, 69_999, 8, 70_001];
        let wide_uniform = PenaltyHistory::uniform(&wide_prompt);
        let wide_split = PenaltyHistory {
            repetition: &wide_prompt,
            presence: &wide_prompt[4..],
        };
        let wide = [
            candidates(20, 0.8, wide_split, 1.3, 0.5),
            candidates(0, 1.0, wide_uniform, 1.2, 0.0),
            candidates(20, 1.3, wide_uniform, 1.0, 0.3),
            candidates(1, 0.5, wide_uniform, 1.5, 0.0),
        ];
        assert_batched_packs_match_per_row(device, 70_000, 69_998, &wide);
    }

    #[test]
    fn batched_candidate_planning_matches_per_row_packs_on_cpu() {
        assert_batched_planning(&Device::Cpu);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn batched_candidate_planning_matches_per_row_packs_on_metal() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "batched_candidate_planning_matches_per_row_packs_on_metal",
        ) else {
            return;
        };
        assert_batched_planning(&device);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn batched_candidate_planning_matches_per_row_packs_on_cuda() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        assert_batched_planning(&device);
    }

    /// With the kill switch off a step plans every row alone, and so does a
    /// step with one candidate row: each pack is the row's own tensor,
    /// exactly what `pending_sampling_candidates` plans.
    #[test]
    fn batched_device_sampling_kill_switch_plans_every_row_alone() {
        let _guard = crate::env_test_lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::set_var(batched_device_sampling::ENV, "false");
        let switched_off = batched_device_sampling::resolve(&Device::Cpu);
        std::env::remove_var(batched_device_sampling::ENV);
        assert_eq!(
            switched_off,
            Err(format!("{}=false", batched_device_sampling::ENV))
        );
        assert_eq!(batched_device_sampling::resolve(&Device::Cpu), Ok(()));

        let device = Device::Cpu;
        let history = [5u32, 8, 17, 17, 300];
        let (width, vocab) = (1_500, 1_400);
        let logits = Tensor::from_vec(planning_logits(3, width), (3, width), &device).unwrap();
        let expect_per_row =
            |plans: &[PlannedReadback<'_>], planned: Vec<Option<PendingCandidates>>| {
                for (row, (plan, planned)) in plans.iter().zip(planned).enumerate() {
                    let PlannedReadback::Candidates(request) = plan else {
                        assert!(planned.is_none(), "row {row}");
                        continue;
                    };
                    let planned = planned.unwrap();
                    let alone = pending_sampling_candidates(
                        &logits.i(row).unwrap(),
                        vocab,
                        request.top_k,
                        request.temperature,
                        request.history,
                        request.repetition_penalty,
                        request.presence_penalty,
                        None,
                    )
                    .unwrap()
                    .unwrap();
                    assert_eq!(planned.offset, 0, "row {row}");
                    assert_eq!(planned.packed.elem_count(), planned.len(), "row {row}");
                    assert_eq!(
                        packed_bits(&planned).unwrap(),
                        packed_bits(&alone).unwrap(),
                        "row {row}"
                    );
                }
            };
        let plans = [
            candidates(20, 0.7, PenaltyHistory::uniform(&history), 1.3, 0.4),
            candidates(20, 0.9, PenaltyHistory::uniform(&history), 1.0, 0.0),
            candidates(0, 1.1, PenaltyHistory::uniform(&[]), 1.0, 0.0),
        ];
        let planned = plan_step_candidates_with(&logits, vocab, &plans, |_, _| false).unwrap();
        expect_per_row(&plans, planned);

        let one_row = [
            PlannedReadback::Greedy,
            candidates(20, 0.7, PenaltyHistory::uniform(&history), 1.3, 0.4),
            PlannedReadback::None,
        ];
        let planned = plan_step_candidates_with(&logits, vocab, &one_row, |_, _| {
            panic!("a step with one candidate row never batches")
        })
        .unwrap();
        expect_per_row(&one_row, planned);
    }

    fn candidate_bits(candidates: &Option<DeviceSamplingCandidates>) -> Option<Vec<u32>> {
        candidates.as_ref().map(|candidates| {
            let mut bits = candidates
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>();
            bits.extend(&candidates.indices);
            bits.extend(candidates.logsumexp.map(f32::to_bits));
            bits
        })
    }

    /// Rows with their own vocabulary prefixes and masks, greedy and
    /// candidate rows mixed, in one transfer: each greedy row reads the
    /// argmax of its masked prefix and its logit, and each candidate row
    /// exactly the candidates the single-row accelerator planner gives it
    /// with that mask (one row nearly all `-inf`, so it falls back).
    fn assert_masked_rows_match_single_rows(device: &Device) {
        let (rows, width) = (8, 2_100);
        let logits = Tensor::from_vec(planning_logits(rows, width), (rows, width), device).unwrap();
        let history = [5u32, 6, 7, 8, 9, 17, 1_027, 2_000, 2_049];
        let masked = (0..2_050)
            .map(|token| token < 2_000 || token == 2_049)
            .collect::<Vec<_>>();
        let open = vec![true; 2_000];
        let (eos, all) = (Some(masked.as_slice()), Some(open.as_slice()));
        let uniform = PenaltyHistory::uniform(&history);
        let plan = |row: usize| -> (usize, Option<&[bool]>, PlannedReadback<'_>) {
            match row {
                0 => (2_050, eos, PlannedReadback::Greedy),
                1 => (2_050, eos, candidates(50, 0.9, uniform, 1.05, 0.0)),
                2 => (2_000, all, candidates(50, 0.9, uniform, 1.05, 0.0)),
                3 => (2_000, None, PlannedReadback::Greedy),
                4 => (2_050, eos, candidates(0, 1.2, uniform, 1.3, 0.0)),
                5 => (2_050, eos, PlannedReadback::None),
                6 => (2_000, None, candidates(1, 0.7, uniform, 1.0, 0.0)),
                _ => (2_050, eos, candidates(20, 0.8, uniform, 1.2, 0.0)),
            }
        };
        let read = read_masked_planned_rows(
            &logits,
            (0..rows)
                .map(|row| {
                    let (vocab, allowed, plan) = plan(row);
                    MaskedReadback {
                        vocab,
                        allowed,
                        plan,
                    }
                })
                .collect(),
        )
        .unwrap();
        for (row, value) in read.into_iter().enumerate() {
            let (vocab, allowed, plan) = plan(row);
            let row_logits = logits.i(row).unwrap();
            match (plan, value) {
                (PlannedReadback::Greedy, PlannedValues::Greedy { index, logit }) => {
                    let mut alone = row_logits.narrow(0, 0, vocab).unwrap();
                    if let Some(allowed) = allowed.filter(|mask| mask.iter().any(|a| !a)) {
                        let mask = allowed.iter().map(|a| u8::from(*a)).collect::<Vec<_>>();
                        let mask = Tensor::from_vec(mask, vocab, device).unwrap();
                        let blocked = Tensor::full(f32::NEG_INFINITY, vocab, device).unwrap();
                        alone = mask.where_cond(&alone, &blocked).unwrap();
                    }
                    let expected = alone.argmax(0).unwrap().to_scalar::<u32>().unwrap();
                    let values = alone.to_vec1::<f32>().unwrap();
                    assert_eq!(index, expected, "row {row}");
                    assert_eq!(
                        logit.to_bits(),
                        values[expected as usize].to_bits(),
                        "row {row}"
                    );
                }
                (PlannedReadback::Candidates(request), PlannedValues::Candidates(candidates)) => {
                    let alone = bounded_sampling_candidates(
                        &row_logits,
                        vocab,
                        request.top_k,
                        request.temperature,
                        request.history,
                        request.repetition_penalty,
                        request.presence_penalty,
                        allowed,
                    )
                    .unwrap();
                    assert_eq!(
                        candidate_bits(&candidates),
                        candidate_bits(&alone),
                        "row {row} on {:?}",
                        device.location()
                    );
                    assert_eq!(candidates.is_none(), row == rows - 1, "row {row}");
                }
                (PlannedReadback::None, PlannedValues::None) => {}
                _ => panic!("row {row} read the wrong kind of values"),
            }
        }
    }

    #[test]
    fn masked_rows_read_what_each_row_reads_alone_on_cpu() {
        assert_masked_rows_match_single_rows(&Device::Cpu);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn masked_rows_read_what_each_row_reads_alone_on_metal() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "masked_rows_read_what_each_row_reads_alone_on_metal",
        ) else {
            return;
        };
        assert_masked_rows_match_single_rows(&device);
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

    /// The top logprobs come from a partial selection; they equal a stable
    /// full sort's prefix, ties in index order.
    #[test]
    fn raw_logprob_tops_equal_a_full_sort_prefix() {
        let mut values = crate::kernels::numeric_check::xorshift_values(4_096, 9, 4.0);
        values[17] = values[3];
        values[40] = f32::NEG_INFINITY;
        for k in [0, 1, 5, 20, 4_096] {
            let (_, top) = raw_logprobs_stats(&values, k).unwrap();
            let mut sorted = values
                .iter()
                .enumerate()
                .filter(|(_, value)| value.is_finite())
                .collect::<Vec<_>>();
            sorted.sort_by(|left, right| right.1.partial_cmp(left.1).unwrap());
            let expected = sorted
                .into_iter()
                .take(k)
                .map(|(index, _)| index as u32)
                .collect::<Vec<_>>();
            assert_eq!(
                top.iter().map(|(token, _)| *token).collect::<Vec<_>>(),
                expected,
                "top {k}"
            );
        }
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

    /// Top-k above the device limit used to loop forever once a merge round
    /// could not shrink the row (here 1100 values, two chunks of 600 kept
    /// each); it is refused, and planning sends the row to the host.
    #[test]
    fn device_top_k_refuses_a_k_the_merge_rounds_cannot_finish() {
        let device = Device::Cpu;
        let tensor =
            Tensor::from_vec((0..1100).map(|v| v as f32).collect(), 1100, &device).unwrap();
        assert!(bounded_topk(&tensor, 600).is_err());
        assert!(device_top_k(&tensor, 600).unwrap().is_none());
        assert!(!plans_bounded_candidates(150_000, 600, 0.7));
        let (values, _) = bounded_topk(&tensor, DEVICE_TOP_K_LIMIT).unwrap();
        assert_eq!(values.dim(0).unwrap(), DEVICE_TOP_K_LIMIT);
        assert_eq!(values.to_vec1::<f32>().unwrap()[0], 1099.0);
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
