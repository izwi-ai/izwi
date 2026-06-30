use std::cmp::Ordering;

use candle_core::{DType, IndexOp, Tensor, D};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy)]
pub struct Lfm25SamplingConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
}

impl Default for Lfm25SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Lfm25AudioGenerationConfig {
    pub text: Lfm25SamplingConfig,
    pub audio: Lfm25SamplingConfig,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        let seed = if seed == 0 {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|value| value.as_nanos() as u64)
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

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / (u32::MAX as f64 + 1.0)) as f32
    }
}

pub fn sample_from_logits(
    logits: &Tensor,
    vocab_limit: usize,
    config: &Lfm25SamplingConfig,
    rng: &mut SimpleRng,
) -> Result<u32> {
    let logits = logits_row(logits)?;
    let vocab_limit = effective_vocab_limit(&logits, vocab_limit)?;

    if config.temperature <= 1e-5 {
        return greedy_from_logits_row(&logits, vocab_limit);
    }

    let mut values = logits_to_vec(&logits.narrow(0, 0, vocab_limit)?)?;
    let temperature = config.temperature.max(1e-5);
    for value in &mut values {
        if value.is_finite() {
            *value /= temperature;
        }
    }

    let mut candidates: Vec<usize> = values
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| value.is_finite().then_some(idx))
        .collect();
    if candidates.is_empty() {
        return argmax_values(&values);
    }

    if config.top_k > 0 && config.top_k < candidates.len() {
        candidates.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap_or(Ordering::Equal));
        candidates.truncate(config.top_k);
    }

    let max_logit = candidates
        .iter()
        .map(|&idx| values[idx])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = candidates
        .iter()
        .map(|&idx| (idx, (values[idx] - max_logit).exp()))
        .collect();

    let mut sum: f32 = probs.iter().map(|(_, prob)| *prob).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return argmax_values(&values);
    }
    for (_, prob) in &mut probs {
        *prob /= sum;
    }

    if config.top_p < 1.0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        let cutoff = config.top_p.max(1e-6);
        let mut cumulative = 0.0f32;
        let mut keep = 0usize;
        for (_, prob) in &probs {
            cumulative += *prob;
            keep += 1;
            if cumulative >= cutoff {
                break;
            }
        }
        probs.truncate(keep.max(1));
        sum = probs.iter().map(|(_, prob)| *prob).sum();
        if sum > 0.0 {
            for (_, prob) in &mut probs {
                *prob /= sum;
            }
        }
    }

    let sample = rng.next_f32();
    let mut cumulative = 0.0f32;
    for (idx, prob) in &probs {
        cumulative += *prob;
        if sample <= cumulative {
            return Ok(*idx as u32);
        }
    }

    probs
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .map(|(idx, _)| *idx as u32)
        .ok_or_else(|| Error::InferenceError("Failed to sample LFM2.5 Audio token".to_string()))
}

pub fn greedy_from_logits(logits: &Tensor, vocab_limit: usize) -> Result<u32> {
    let logits = logits_row(logits)?;
    let vocab_limit = effective_vocab_limit(&logits, vocab_limit)?;
    greedy_from_logits_row(&logits, vocab_limit)
}

fn logits_row(logits: &Tensor) -> Result<Tensor> {
    match logits.rank() {
        1 => Ok(logits.clone()),
        2 => {
            let (rows, _cols) = logits.dims2()?;
            if rows != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected LFM2.5 Audio logits shape for sampling: {:?}",
                    logits.shape().dims()
                )));
            }
            Ok(logits.i(0)?)
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected LFM2.5 Audio logits rank for sampling: {rank}"
            )));
        }
    }
}

fn logits_to_vec(logits: &Tensor) -> Result<Vec<f32>> {
    logits
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()
        .map_err(Error::from)
}

fn effective_vocab_limit(logits: &Tensor, vocab_limit: usize) -> Result<usize> {
    let vocab = logits.dim(0)?;
    let limit = vocab.min(vocab_limit);
    if limit == 0 {
        return Err(Error::InferenceError(
            "Cannot sample from zero-sized LFM2.5 Audio vocabulary".to_string(),
        ));
    }
    Ok(limit)
}

fn greedy_from_logits_row(logits: &Tensor, vocab_limit: usize) -> Result<u32> {
    if logits.device().is_metal() || logits.device().is_cuda() {
        return argmax_row_device(logits, vocab_limit);
    }
    argmax_row_host(logits, vocab_limit)
}

fn argmax_row_device(logits: &Tensor, vocab_limit: usize) -> Result<u32> {
    let logits = logits.narrow(0, 0, vocab_limit)?;
    let idx = logits.argmax(D::Minus1)?;
    let idx = if idx.rank() == 0 {
        idx
    } else {
        idx.squeeze(0)?
    };
    idx.to_dtype(DType::U32)?
        .to_scalar::<u32>()
        .map_err(Error::from)
}

fn argmax_row_host(logits: &Tensor, vocab_limit: usize) -> Result<u32> {
    let values = logits_to_vec(&logits.narrow(0, 0, vocab_limit)?)?;
    argmax_values(&values)
}

fn argmax_values(values: &[f32]) -> Result<u32> {
    let mut max_idx = None;
    let mut max_value = f32::NEG_INFINITY;

    for (idx, value) in values.iter().enumerate() {
        if value.is_finite() && *value > max_value {
            max_value = *value;
            max_idx = Some(idx);
        }
    }

    max_idx
        .map(|idx| idx as u32)
        .ok_or_else(|| Error::InferenceError("No valid LFM2.5 Audio logits to sample".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_rng_is_repeatable() {
        let mut first = SimpleRng::new(1234);
        let mut second = SimpleRng::new(1234);
        assert_eq!(first.next_u32(), second.next_u32());
        assert_eq!(first.next_u32(), second.next_u32());
    }

    #[test]
    fn sampling_defaults_to_greedy_when_temperature_is_zero() {
        let logits = Tensor::from_vec(vec![0.1f32, 0.9, 0.3], (3,), &candle_core::Device::Cpu)
            .expect("logits");
        let config = Lfm25SamplingConfig::default();
        let token =
            sample_from_logits(&logits, 3, &config, &mut SimpleRng::new(7)).expect("sample token");
        assert_eq!(token, 1);
    }

    #[test]
    fn greedy_sampling_respects_vocab_limit() {
        let logits = Tensor::from_vec(vec![0.1f32, 0.9, 7.0], (3,), &candle_core::Device::Cpu)
            .expect("logits");
        let token = greedy_from_logits(&logits, 2).expect("sample token");
        assert_eq!(token, 1);
    }

    #[test]
    fn non_greedy_sampling_respects_vocab_limit() {
        let logits = Tensor::from_vec(vec![0.1f32, 0.9, 7.0], (3,), &candle_core::Device::Cpu)
            .expect("logits");
        let config = Lfm25SamplingConfig {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
        };
        let token =
            sample_from_logits(&logits, 2, &config, &mut SimpleRng::new(7)).expect("sample token");
        assert_eq!(token, 1);
    }
}
