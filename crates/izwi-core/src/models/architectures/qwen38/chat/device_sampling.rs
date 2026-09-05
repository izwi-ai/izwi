//! Sampling keeps probability rows on the device. Only bounded token/status
//! results cross the device boundary. The caller owns history and RNG commit.
use super::{ChatGenerationConfig, SimpleRng};
use crate::error::{Error, Result};
use crate::kernels::cuda::sampling::{self, SamplingParams};
use crate::models::shared::speculative_sampling::SpeculativeVerification;
use candle_core::{DType, IndexOp, Tensor};
use std::collections::BTreeMap;

fn unit(rng: &mut SimpleRng) -> f32 {
    (rng.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
}

pub(super) fn greedy(logits: &Tensor, vocab: usize) -> Result<Vec<u32>> {
    let (_, width) = logits.dims2()?;
    if vocab == 0 || width < vocab {
        return Err(Error::InvalidInput(
            "invalid device sampling vocabulary".into(),
        ));
    }
    let results = sampling::greedy_rows(&logits.narrow(1, 0, vocab)?)?.to_vec2::<u32>()?;
    results
        .into_iter()
        .map(|row| {
            if row[1] != 1 {
                Err(Error::InferenceError(
                    "No finite Qwen3.8 logits to sample".into(),
                ))
            } else {
                Ok(row[0])
            }
        })
        .collect()
}

pub(super) fn distribution(
    logits: &Tensor,
    vocab: usize,
    config: &ChatGenerationConfig,
    history: &[u32],
) -> Result<Tensor> {
    let (_, width) = logits.dims2()?;
    if vocab == 0 || width < vocab {
        return Err(Error::InvalidInput(
            "invalid device sampling vocabulary".into(),
        ));
    }
    let mut frequencies = BTreeMap::<u32, u32>::new();
    for &token in history {
        if (token as usize) < vocab {
            let count = frequencies.entry(token).or_default();
            *count = count.saturating_add(1);
        }
    }
    let mut counts = Tensor::zeros(vocab, DType::U32, logits.device())?;
    if !frequencies.is_empty() {
        let indices = Tensor::from_vec(
            frequencies.keys().copied().collect::<Vec<_>>(),
            frequencies.len(),
            logits.device(),
        )?;
        let values = Tensor::from_vec(
            frequencies.values().copied().collect::<Vec<_>>(),
            frequencies.len(),
            logits.device(),
        )?;
        counts = counts.index_add(&indices, &values, 0)?;
    }
    let params = SamplingParams {
        temperature: config.temperature,
        repetition_penalty: config.repetition_penalty,
        presence_penalty: config.presence_penalty,
        frequency_penalty: 0.0,
        top_k: config.top_k,
        top_p: config.top_p,
        min_p: 0.0,
    };
    sampling::distributions(&logits.narrow(1, 0, vocab)?, &counts.unsqueeze(0)?, &params)
        .map_err(Error::from)
}

fn sample_at(probabilities: &Tensor, uniform: f32) -> Result<u32> {
    let uniforms = Tensor::from_vec(vec![uniform], 1, probabilities.device())?;
    let result = sampling::sample_rows(probabilities, &uniforms)?.to_vec2::<u32>()?;
    if result[0][1] != 1 {
        return Err(Error::InferenceError(
            "No finite Qwen3.8 sampling distribution".into(),
        ));
    }
    Ok(result[0][0])
}

pub(super) fn propose(
    logits: &Tensor,
    vocab: usize,
    config: &ChatGenerationConfig,
    history: &mut Vec<u32>,
    rng: &mut SimpleRng,
) -> Result<(u32, Tensor)> {
    let probabilities = distribution(logits, vocab, config, history)?;
    let mut staged = rng.clone();
    let token = sample_at(&probabilities, unit(&mut staged))?;
    history.push(token);
    *rng = staged;
    Ok((token, probabilities))
}

pub(super) fn sample(
    logits: &Tensor,
    vocab: usize,
    config: &ChatGenerationConfig,
    history: &[u32],
    rng: &mut SimpleRng,
) -> Result<u32> {
    if config.temperature <= 1e-5
        && config.repetition_penalty <= 1.0
        && config.presence_penalty.abs() <= f32::EPSILON
    {
        return Ok(greedy(&logits.reshape((1, ()))?, vocab)?[0]);
    }
    let probabilities = distribution(&logits.reshape((1, ()))?, vocab, config, history)?;
    let mut staged = rng.clone();
    // Greedy distributions are one-hot and must not consume a draw.
    let uniform = if config.temperature <= 1e-5 {
        0.0
    } else {
        staged.next_f32()
    };
    let token = sample_at(&probabilities, uniform)?;
    *rng = staged;
    Ok(token)
}

pub(super) fn verify_greedy(
    drafts: &[u32],
    target_logits: &Tensor,
    vocab: usize,
    config: &ChatGenerationConfig,
    history: &mut Vec<u32>,
) -> Result<SpeculativeVerification> {
    let mut position_history = history.clone();
    let mut probabilities = Vec::with_capacity(drafts.len() + 1);
    for position in 0..=drafts.len() {
        probabilities.push(distribution(
            &target_logits.i((0, position))?.unsqueeze(0)?,
            vocab,
            config,
            &position_history,
        )?);
        if position < drafts.len() {
            position_history.push(drafts[position]);
        }
    }
    let probabilities = Tensor::cat(&probabilities, 0)?;
    let statuses = sampling::sample_rows(
        &probabilities,
        &Tensor::zeros(drafts.len() + 1, DType::F32, target_logits.device())?,
    )?
    .to_vec2::<u32>()?;
    let tokens = statuses
        .into_iter()
        .map(|row| {
            if row[1] == 1 {
                Ok(row[0])
            } else {
                Err(Error::InferenceError(
                    "No finite Qwen3.8 greedy distribution".into(),
                ))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    crate::models::shared::speculative_sampling::verify_greedy_token_prefix(
        drafts, &tokens, history,
    )
}

pub(super) fn verify(
    drafts: &[u32],
    q: &[Tensor],
    target_logits: &Tensor,
    vocab: usize,
    config: &ChatGenerationConfig,
    history: &mut Vec<u32>,
    rng: &mut SimpleRng,
) -> Result<SpeculativeVerification> {
    if drafts.is_empty() || drafts.len() != q.len() || target_logits.dim(1)? != drafts.len() + 1 {
        return Err(Error::InvalidInput(
            "invalid device speculative block".into(),
        ));
    }
    let mut staged_history = history.clone();
    let mut p = Vec::with_capacity(drafts.len() + 1);
    for position in 0..=drafts.len() {
        p.push(distribution(
            &target_logits.i((0, position))?.unsqueeze(0)?,
            vocab,
            config,
            &staged_history,
        )?);
        if position < drafts.len() {
            staged_history.push(drafts[position]);
        }
    }
    let mut draws_rng = rng.clone();
    let draws = (0..=drafts.len())
        .map(|_| unit(&mut draws_rng))
        .collect::<Vec<_>>();
    let uniforms = draws
        .windows(2)
        .flat_map(|pair| pair.iter().copied())
        .collect::<Vec<_>>();
    let device = target_logits.device();
    let status = sampling::verify_rows(
        &Tensor::cat(&p[..drafts.len()], 0)?,
        &Tensor::cat(q, 0)?,
        &Tensor::from_slice(drafts, drafts.len(), device)?,
        &Tensor::from_vec(uniforms, (drafts.len(), 2), device)?,
    )?
    .to_vec2::<u32>()?;
    let accepted = status
        .iter()
        .position(|row| row[0] == 0)
        .unwrap_or(drafts.len());
    let inspected = (accepted + 1).min(drafts.len());
    if status[..inspected].iter().any(|row| row[2] != 1) {
        return Err(Error::InferenceError(
            "invalid device speculative probabilities".into(),
        ));
    }
    let mut emitted_tokens = drafts[..accepted].to_vec();
    let consumed_draws = if accepted < drafts.len() {
        emitted_tokens.push(status[accepted][1]);
        accepted + 2
    } else {
        emitted_tokens.push(sample_at(&p[drafts.len()], draws[drafts.len()])?);
        drafts.len() + 1
    };
    let mut staged_rng = rng.clone();
    for _ in 0..consumed_draws {
        unit(&mut staged_rng);
    }
    history.extend_from_slice(&emitted_tokens);
    *rng = staged_rng;
    Ok(SpeculativeVerification {
        emitted_tokens,
        accepted_draft_tokens: accepted,
        draft_tokens: drafts.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::shared::speculative_sampling::{
        propose_speculative_draft, verify_speculative_proposals,
    };
    use candle_core::Device;

    #[test]
    fn device_math_matches_shared_proposal_and_pq_verifier_with_penalties_and_rng_commit() {
        let mut accepted_lengths = [false; 3];
        for seed in 1..=192 {
            let config = ChatGenerationConfig {
                temperature: 0.8,
                top_k: 4,
                top_p: 0.92,
                repetition_penalty: 1.15,
                presence_penalty: 0.17,
                seed,
                ..Default::default()
            };
            let initial_history = vec![1, 1, 3];
            let mut device_history = initial_history.clone();
            let mut shared_history = initial_history.clone();
            let mut device_rng = SimpleRng::new(seed);
            let mut shared_rng = device_rng.clone();
            let mut q = Vec::new();
            let mut tokens = Vec::new();
            let mut proposals = Vec::new();
            for logits in [[1.2f32, -0.3, 0.8, 0.1, -2.0], [-0.2, 0.7, 0.9, 1.1, -1.0]] {
                let tensor = Tensor::from_slice(&logits, (1, 5), &Device::Cpu).unwrap();
                let (token, probabilities) =
                    propose(&tensor, 5, &config, &mut device_history, &mut device_rng).unwrap();
                let shared = propose_speculative_draft(
                    &logits,
                    &config,
                    &mut shared_history,
                    &mut shared_rng,
                )
                .unwrap();
                assert_eq!(token, shared.token_id, "proposal seed={seed}");
                assert_eq!(device_rng.state, shared_rng.state);
                tokens.push(token);
                q.push(probabilities);
                proposals.push(shared);
            }
            assert_eq!(device_history, shared_history);
            let rows = vec![
                vec![0.8f32, 1.1, 0.3, -0.5, -2.0],
                vec![1.0, -0.2, 0.7, 0.9, -1.0],
                vec![0.4, 0.8, -0.3, 0.9, -1.0],
            ];
            let target = Tensor::from_vec(
                rows.iter().flatten().copied().collect(),
                (1, 3, 5),
                &Device::Cpu,
            )
            .unwrap();
            let mut device_history = initial_history.clone();
            let mut shared_history = initial_history;
            let actual = verify(
                &tokens,
                &q,
                &target,
                5,
                &config,
                &mut device_history,
                &mut device_rng,
            )
            .unwrap();
            let expected = verify_speculative_proposals(
                &proposals,
                &rows,
                &config,
                &mut shared_history,
                &mut shared_rng,
            )
            .unwrap();
            assert_eq!(actual, expected, "verification seed={seed}");
            assert_eq!(device_history, shared_history);
            assert_eq!(
                device_rng.state, shared_rng.state,
                "only used acceptance/residual/bonus draws commit"
            );
            accepted_lengths[actual.accepted_draft_tokens] = true;
        }
        assert_eq!(accepted_lengths, [true; 3]);
    }

    #[test]
    fn failed_distribution_is_transactional_and_greedy_does_not_consume_rng() {
        let config = ChatGenerationConfig::default();
        let mut rng = SimpleRng::new(42);
        let before = rng.state;
        let logits = Tensor::from_slice(&[2f32, 2.0, -1.0], 3, &Device::Cpu).unwrap();
        assert_eq!(sample(&logits, 3, &config, &[], &mut rng).unwrap(), 0);
        assert_eq!(rng.state, before);
        let bad = Tensor::from_slice(&[f32::NAN, f32::INFINITY], (1, 2), &Device::Cpu).unwrap();
        let mut history = vec![1];
        assert!(propose(&bad, 2, &config, &mut history, &mut rng).is_err());
        assert_eq!(history, vec![1]);
        assert_eq!(rng.state, before);
    }
}
