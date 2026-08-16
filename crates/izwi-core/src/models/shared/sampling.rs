//! Bounded accelerator-side candidate selection for stochastic sampling.
//!
//! Candle's CUDA argsort is efficient only for bounded rows. This module sorts
//! fixed-size chunks and recursively merges their exact top-k candidates, so a
//! large vocabulary never becomes one oversized CUDA sort or a full host copy.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeSet, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::{DType, Device, IndexOp, Tensor, D};

use crate::{Error, Result};

use super::chat::ChatGenerationConfig;

const CUDA_SORT_CHUNK: usize = 1024;
const MAX_EXACT_F32_INTEGER: usize = 1 << 24;
const CUDA_SAMPLING_TENSOR_CACHE_ENTRIES: usize = 8;
pub const CUDA_SAMPLING_CANDIDATE_LIMIT: usize = 256;

/// Request-owned chat sampler used after a shared tensor forward. Continuous
/// batching shares logits computation, never sampling policy or RNG state.
#[derive(Clone)]
pub struct ChatSampler {
    config: ChatGenerationConfig,
    history: Vec<u32>,
    track_history: bool,
    rng: SimpleRng,
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
        }
    }

    pub fn sample(&mut self, logits: &Tensor, vocab_size: usize) -> Result<u32> {
        let token = sample_chat_token(
            logits,
            vocab_size,
            &self.config,
            &self.history,
            &mut self.rng,
        )?;
        if self.track_history {
            self.history.push(token);
        }
        Ok(token)
    }

    pub fn is_configured_stop(&self, token: u32) -> bool {
        self.config.stop_token_ids.contains(&token)
    }

    #[cfg(test)]
    fn history(&self) -> &[u32] {
        &self.history
    }
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
    if vocab_size == 0 {
        return Err(Error::InvalidInput(
            "chat sampler received vocab_size=0".to_string(),
        ));
    }
    let row = chat_logits_row(logits)?;
    let deterministic_greedy = config.temperature <= 1e-5
        && (config.repetition_penalty - 1.0).abs() <= f32::EPSILON
        && config.presence_penalty.abs() <= f32::EPSILON
        && config.top_k == 0
        && config.top_p >= 1.0;
    if deterministic_greedy {
        return chat_argmax_clamped(&row, vocab_size);
    }

    if let Some(candidates) = bounded_cuda_sampling_candidates(
        &row,
        vocab_size,
        config.top_k,
        config.temperature,
        history,
        config.repetition_penalty,
        config.presence_penalty,
        None,
    )? {
        if device_candidates_cover_top_p(&candidates, config.top_p) {
            if let Some(sampled) =
                sample_device_candidates(&candidates, config.top_p, rng.next_f32())
            {
                return Ok(sampled);
            }
        }
    }

    let mut values = row.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    values.truncate(vocab_size.min(values.len()));
    if values.is_empty() {
        return Err(Error::InvalidInput(
            "chat sampler received no in-vocabulary logits".to_string(),
        ));
    }
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
    let mut candidates = values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| value.is_finite().then_some(index))
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return finite_argmax(&values);
    }
    candidates.sort_by(|left, right| {
        values[*right]
            .partial_cmp(&values[*left])
            .unwrap_or(Ordering::Equal)
    });
    if config.top_k > 0 {
        candidates.truncate(config.top_k.min(candidates.len()));
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
    let draw = rng.next_f32();
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
    let selected_value = row
        .i(selected as usize)?
        .to_dtype(DType::F32)?
        .to_scalar::<f32>()?;
    if selected_value.is_finite() {
        return Ok(selected);
    }
    finite_argmax(&row.to_dtype(DType::F32)?.to_vec1::<f32>()?)
}

fn apply_chat_history_penalties(
    values: &mut [f32],
    history: &[u32],
    repetition_penalty: f32,
    presence_penalty: f32,
) {
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
}

struct SamplingTensorCacheEntry {
    device: candle_core::DeviceLocation,
    input: SamplingTensorCacheInput,
    tensor: Tensor,
}

thread_local! {
    static CUDA_SAMPLING_TENSOR_CACHE: RefCell<VecDeque<SamplingTensorCacheEntry>> =
        const { RefCell::new(VecDeque::new()) };
}

fn cached_sampling_tensor(device: &Device, input: SamplingTensorCacheInput) -> Result<Tensor> {
    CUDA_SAMPLING_TENSOR_CACHE.with(|cache| {
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
                    _ => false,
                }
        }) {
            let entry = cache
                .remove(index)
                .expect("located CUDA sampling tensor cache entry");
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
        };
        if cache.len() == CUDA_SAMPLING_TENSOR_CACHE_ENTRIES {
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
    if !logits.device().is_cuda() || temperature <= 1e-5 {
        return Ok(None);
    }
    let row = logits_row(logits)?;
    let vocab = row.dim(0)?.min(vocab_limit);
    if vocab == 0 || vocab > MAX_EXACT_F32_INTEGER || top_k > CUDA_SORT_CHUNK {
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
        CUDA_SAMPLING_CANDIDATE_LIMIT
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
    let packed = Tensor::cat(&packed_parts, 0)?.to_vec1::<f32>()?;
    crate::models::shared::telemetry::record_host_read(DType::F32, packed.len());
    let candidate_count = values.dim(0)?;
    let (values, indices, logsumexp) =
        unpack_sampling_readback(&packed, candidate_count, logsumexp.is_some())?;
    if values.iter().any(|value| !value.is_finite()) {
        return Ok(None);
    }
    Ok(Some(DeviceSamplingCandidates {
        values,
        indices,
        logsumexp,
    }))
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
    history: &[u32],
    repetition_penalty: f32,
    presence_penalty: f32,
) -> Result<Tensor> {
    if history.is_empty()
        || ((repetition_penalty - 1.0).abs() <= f32::EPSILON
            && presence_penalty.abs() <= f32::EPSILON)
    {
        return Ok(values.clone());
    }
    let vocab = values.dim(0)?;
    let indices = history
        .iter()
        .copied()
        .filter(|token| (*token as usize) < vocab)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
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

fn bounded_topk(values: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    let length = values.dim(0)?;
    if length == 0 || length > u32::MAX as usize || k == 0 || k > CUDA_SORT_CHUNK {
        return Err(Error::InvalidInput(format!(
            "bounded top-k requires a U32-addressable non-empty row and 1..={CUDA_SORT_CHUNK} candidates"
        )));
    }
    let mut candidate_values = values.contiguous()?;
    let mut candidate_indices = Tensor::arange(0u32, length as u32, values.device())?;
    loop {
        let length = candidate_values.dim(0)?;
        if length <= CUDA_SORT_CHUNK {
            let (sorted, order) = candidate_values.sort_last_dim(false)?;
            let keep = k.min(length);
            return Ok((
                sorted.narrow(0, 0, keep)?,
                candidate_indices.gather(&order, 0)?.narrow(0, 0, keep)?,
            ));
        }

        let chunks = length.div_ceil(CUDA_SORT_CHUNK);
        let padded_len = chunks * CUDA_SORT_CHUNK;
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
        let values_2d = candidate_values.reshape((chunks, CUDA_SORT_CHUNK))?;
        let indices_2d = candidate_indices.reshape((chunks, CUDA_SORT_CHUNK))?;
        let (sorted, order) = values_2d.sort_last_dim(false)?;
        let keep = k.min(CUDA_SORT_CHUNK);
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
    fn device_penalties_and_masks_match_sampling_contract() {
        let device = Device::Cpu;
        let values = Tensor::from_vec(vec![4.0f32, -2.0, 3.0, f32::NAN], 4, &device).unwrap();
        let values = sanitize_finite(&values).unwrap();
        let values = apply_history_penalties(&values, &[0, 1, 1], 2.0, 0.5).unwrap();
        let actual = values.to_vec1::<f32>().unwrap();
        assert_eq!(actual[..3], [1.5, -4.5, 3.0]);
        assert_eq!(actual[3], f32::NEG_INFINITY);
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
        CUDA_SAMPLING_TENSOR_CACHE.with(|cache| cache.borrow_mut().clear());
        let device = Device::Cpu;
        let first =
            cached_sampling_tensor(&device, SamplingTensorCacheInput::Mask(vec![1, 0, 1])).unwrap();
        let second =
            cached_sampling_tensor(&device, SamplingTensorCacheInput::Mask(vec![1, 0, 1])).unwrap();
        assert_eq!(first.id(), second.id());
        let indices =
            cached_sampling_tensor(&device, SamplingTensorCacheInput::Indices(vec![3, 7])).unwrap();
        assert_eq!(indices.to_vec1::<u32>().unwrap(), vec![3, 7]);
        CUDA_SAMPLING_TENSOR_CACHE.with(|cache| assert_eq!(cache.borrow().len(), 2));
    }
}
