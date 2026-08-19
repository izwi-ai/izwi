//! Model-neutral verification for speculative token blocks.
//!
//! The verifier operates on host logits so model implementations can own the
//! forward pass and cache transaction. It mirrors [`ChatGenerationConfig`]
//! sampling transforms and commits sampler history and RNG state only after an
//! entire verification operation succeeds.

use std::cmp::Ordering;

use rand::RngCore;

use crate::{Error, Result};

use super::chat::ChatGenerationConfig;

const GREEDY_TEMPERATURE_CUTOFF: f32 = 1e-5;
const F32_UNIT_SCALE: f32 = 1.0 / ((1u32 << 24) as f32);

/// Tokens emitted by one target-model verification pass.
///
/// A rejected block emits the accepted draft prefix followed by one target
/// correction. A fully accepted block emits every draft token followed by one
/// target bonus token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeculativeVerification {
    pub emitted_tokens: Vec<u32>,
    pub accepted_draft_tokens: usize,
    pub draft_tokens: usize,
}

/// One token proposed by a draft model together with the exact sampling
/// distribution that produced it.
///
/// Retaining `q` is required for standard lossless speculative rejection
/// sampling. Treating a sampled proposal as one-hot is valid only for greedy
/// drafting and unnecessarily lowers acceptance for stochastic requests.
#[derive(Debug, Clone)]
pub struct SpeculativeDraft {
    pub token_id: u32,
    distribution: TargetDistribution,
}

/// Sample one draft token using the same transforms as target sampling.
///
/// `history` and `rng` are transactional: neither changes when constructing or
/// sampling the proposal fails.
pub fn propose_speculative_draft<R: RngCore + Clone>(
    draft_logits: &[f32],
    config: &ChatGenerationConfig,
    history: &mut Vec<u32>,
    rng: &mut R,
) -> Result<SpeculativeDraft> {
    let distribution = TargetDistribution::from_logits(draft_logits, config, history)?;
    let mut staged_rng = rng.clone();
    let token_id = if config.temperature <= GREEDY_TEMPERATURE_CUTOFF {
        distribution.argmax()
    } else {
        distribution.sample(next_unit_f32(&mut staged_rng))
    };
    history.push(token_id);
    *rng = staged_rng;
    Ok(SpeculativeDraft {
        token_id,
        distribution,
    })
}

impl SpeculativeVerification {
    pub fn all_drafts_accepted(&self) -> bool {
        self.accepted_draft_tokens == self.draft_tokens
    }

    pub fn emitted_bonus_token(&self) -> bool {
        self.all_drafts_accepted()
    }
}

/// Verify a draft block using the mode implied by the chat sampling policy.
///
/// Zero-temperature policies use deterministic prefix verification. Stochastic
/// policies use lossless rejection sampling against a greedy, one-hot draft
/// distribution. `target_logits` must contain one row per draft token plus one
/// bonus-token row.
pub fn verify_speculative_prefix<R: RngCore + Clone>(
    draft_tokens: &[u32],
    target_logits: &[Vec<f32>],
    config: &ChatGenerationConfig,
    history: &mut Vec<u32>,
    rng: &mut R,
) -> Result<SpeculativeVerification> {
    if config.temperature <= GREEDY_TEMPERATURE_CUTOFF {
        verify_greedy_prefix(draft_tokens, target_logits, config, history)
    } else {
        verify_rejection_sampled_prefix(draft_tokens, target_logits, config, history, rng)
    }
}

/// Verify proposals while retaining their true draft distributions.
///
/// Greedy requests use exact prefix matching. Stochastic requests use the
/// standard lossless acceptance probability `min(1, p(d) / q(d))` and sample
/// a correction from normalized `max(p - q, 0)` after rejection.
pub fn verify_speculative_proposals<R: RngCore + Clone>(
    drafts: &[SpeculativeDraft],
    target_logits: &[Vec<f32>],
    config: &ChatGenerationConfig,
    history: &mut Vec<u32>,
    rng: &mut R,
) -> Result<SpeculativeVerification> {
    let draft_tokens = drafts
        .iter()
        .map(|draft| draft.token_id)
        .collect::<Vec<_>>();
    if config.temperature <= GREEDY_TEMPERATURE_CUTOFF {
        return verify_greedy_prefix(&draft_tokens, target_logits, config, history);
    }
    validate_block(&draft_tokens, target_logits)?;
    for (index, draft) in drafts.iter().enumerate() {
        if draft.distribution.probability(draft.token_id) <= 0.0 {
            return Err(Error::InvalidInput(format!(
                "speculative draft token {} at index {index} has zero proposal probability",
                draft.token_id
            )));
        }
    }

    let mut staged_history = history.clone();
    let mut staged_rng = rng.clone();
    let mut emitted_tokens = Vec::with_capacity(drafts.len() + 1);
    for (index, draft) in drafts.iter().enumerate() {
        let target =
            TargetDistribution::from_logits(&target_logits[index], config, &staged_history)?;
        let p = target.probability(draft.token_id);
        let q = draft.distribution.probability(draft.token_id);
        let acceptance = (p / q).min(1.0);
        if next_unit_f32(&mut staged_rng) < acceptance {
            emitted_tokens.push(draft.token_id);
            staged_history.push(draft.token_id);
            continue;
        }

        let correction =
            target.sample_residual_against(&draft.distribution, next_unit_f32(&mut staged_rng))?;
        emitted_tokens.push(correction);
        staged_history.push(correction);
        *history = staged_history;
        *rng = staged_rng;
        return Ok(SpeculativeVerification {
            emitted_tokens,
            accepted_draft_tokens: index,
            draft_tokens: drafts.len(),
        });
    }

    let bonus =
        TargetDistribution::from_logits(&target_logits[drafts.len()], config, &staged_history)?
            .sample(next_unit_f32(&mut staged_rng));
    emitted_tokens.push(bonus);
    staged_history.push(bonus);
    *history = staged_history;
    *rng = staged_rng;
    Ok(SpeculativeVerification {
        emitted_tokens,
        accepted_draft_tokens: drafts.len(),
        draft_tokens: drafts.len(),
    })
}

/// Deterministically accept the longest draft prefix matching target argmaxes.
///
/// History changes are transactional: an invalid logits block leaves `history`
/// untouched. This path does not consume RNG state.
pub fn verify_greedy_prefix(
    draft_tokens: &[u32],
    target_logits: &[Vec<f32>],
    config: &ChatGenerationConfig,
    history: &mut Vec<u32>,
) -> Result<SpeculativeVerification> {
    validate_block(draft_tokens, target_logits)?;
    let mut staged_history = history.clone();
    let mut emitted_tokens = Vec::with_capacity(draft_tokens.len() + 1);

    for (index, draft_token) in draft_tokens.iter().copied().enumerate() {
        let target =
            TargetDistribution::from_logits(&target_logits[index], config, &staged_history)?
                .argmax();
        if target != draft_token {
            emitted_tokens.push(target);
            staged_history.push(target);
            *history = staged_history;
            return Ok(SpeculativeVerification {
                emitted_tokens,
                accepted_draft_tokens: index,
                draft_tokens: draft_tokens.len(),
            });
        }

        emitted_tokens.push(draft_token);
        staged_history.push(draft_token);
    }

    let bonus = TargetDistribution::from_logits(
        &target_logits[draft_tokens.len()],
        config,
        &staged_history,
    )?
    .argmax();
    emitted_tokens.push(bonus);
    staged_history.push(bonus);
    *history = staged_history;

    Ok(SpeculativeVerification {
        emitted_tokens,
        accepted_draft_tokens: draft_tokens.len(),
        draft_tokens: draft_tokens.len(),
    })
}

/// Losslessly verify a greedy draft block against the target distribution.
///
/// For draft token `d`, the draft distribution is `q(d) = 1`. The token is
/// accepted with probability `p(d)`; on rejection, the correction is sampled
/// from normalized `max(p - q, 0)`. A fully accepted block receives a sample
/// from the final target row. RNG and history are cloned up front and committed
/// together only on success.
pub fn verify_rejection_sampled_prefix<R: RngCore + Clone>(
    draft_tokens: &[u32],
    target_logits: &[Vec<f32>],
    config: &ChatGenerationConfig,
    history: &mut Vec<u32>,
    rng: &mut R,
) -> Result<SpeculativeVerification> {
    validate_block(draft_tokens, target_logits)?;
    let mut staged_history = history.clone();
    let mut staged_rng = rng.clone();
    let mut emitted_tokens = Vec::with_capacity(draft_tokens.len() + 1);

    for (index, draft_token) in draft_tokens.iter().copied().enumerate() {
        let distribution =
            TargetDistribution::from_logits(&target_logits[index], config, &staged_history)?;
        let acceptance_draw = next_unit_f32(&mut staged_rng);
        if acceptance_draw < distribution.probability(draft_token) {
            emitted_tokens.push(draft_token);
            staged_history.push(draft_token);
            continue;
        }

        let correction =
            distribution.sample_without(draft_token, next_unit_f32(&mut staged_rng))?;
        emitted_tokens.push(correction);
        staged_history.push(correction);
        *history = staged_history;
        *rng = staged_rng;
        return Ok(SpeculativeVerification {
            emitted_tokens,
            accepted_draft_tokens: index,
            draft_tokens: draft_tokens.len(),
        });
    }

    let bonus = TargetDistribution::from_logits(
        &target_logits[draft_tokens.len()],
        config,
        &staged_history,
    )?
    .sample(next_unit_f32(&mut staged_rng));
    emitted_tokens.push(bonus);
    staged_history.push(bonus);
    *history = staged_history;
    *rng = staged_rng;

    Ok(SpeculativeVerification {
        emitted_tokens,
        accepted_draft_tokens: draft_tokens.len(),
        draft_tokens: draft_tokens.len(),
    })
}

fn validate_block(draft_tokens: &[u32], target_logits: &[Vec<f32>]) -> Result<()> {
    let expected_rows = draft_tokens
        .len()
        .checked_add(1)
        .ok_or_else(|| Error::InvalidInput("speculative draft length overflow".into()))?;
    if target_logits.len() != expected_rows {
        return Err(Error::InvalidInput(format!(
            "speculative verification requires {expected_rows} target rows for {} draft tokens, got {}",
            draft_tokens.len(),
            target_logits.len()
        )));
    }

    let vocab_size = target_logits
        .first()
        .map(Vec::len)
        .ok_or_else(|| Error::InvalidInput("speculative verification received no rows".into()))?;
    if vocab_size == 0 {
        return Err(Error::InvalidInput(
            "speculative verification received an empty vocabulary".into(),
        ));
    }
    for (index, row) in target_logits.iter().enumerate() {
        if row.len() != vocab_size {
            return Err(Error::InvalidInput(format!(
                "speculative target row {index} has vocabulary {}, expected {vocab_size}",
                row.len()
            )));
        }
    }
    for (index, token) in draft_tokens.iter().copied().enumerate() {
        if token as usize >= vocab_size {
            return Err(Error::InvalidInput(format!(
                "speculative draft token {token} at index {index} is outside vocabulary {vocab_size}"
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct TargetDistribution {
    // Sorted by descending target probability, then ascending token ID.
    candidates: Vec<(u32, f32)>,
}

impl TargetDistribution {
    fn from_logits(logits: &[f32], config: &ChatGenerationConfig, history: &[u32]) -> Result<Self> {
        let mut values = logits.to_vec();
        apply_history_penalties(
            &mut values,
            history,
            config.repetition_penalty,
            config.presence_penalty,
        );

        if config.temperature <= GREEDY_TEMPERATURE_CUTOFF {
            return Ok(Self {
                candidates: vec![(finite_argmax(&values)?, 1.0)],
            });
        }

        let temperature = config.temperature.max(GREEDY_TEMPERATURE_CUTOFF);
        for value in &mut values {
            if value.is_finite() {
                *value /= temperature;
            }
        }

        let mut ranked = values
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .collect::<Vec<_>>();
        if ranked.is_empty() {
            return Err(no_finite_logits_error());
        }
        ranked.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        if config.top_k > 0 {
            ranked.truncate(config.top_k.min(ranked.len()));
        }

        let max_logit = ranked[0].1;
        let mut candidates = ranked
            .into_iter()
            .map(|(token, value)| {
                u32::try_from(token)
                    .map(|token| (token, (value - max_logit).exp()))
                    .map_err(|_| {
                        Error::InvalidInput(
                            "speculative vocabulary contains a token above u32::MAX".into(),
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        normalize(&mut candidates)?;

        if config.top_p < 1.0 {
            let cutoff = config.top_p.clamp(1e-6, 1.0);
            let mut cumulative = 0.0f32;
            let mut keep = 0usize;
            for (_, probability) in &candidates {
                cumulative += *probability;
                keep += 1;
                if cumulative >= cutoff {
                    break;
                }
            }
            candidates.truncate(keep.max(1));
            normalize(&mut candidates)?;
        }

        Ok(Self { candidates })
    }

    fn argmax(&self) -> u32 {
        self.candidates[0].0
    }

    fn probability(&self, token: u32) -> f32 {
        self.candidates
            .iter()
            .find_map(|(candidate, probability)| (*candidate == token).then_some(*probability))
            .unwrap_or(0.0)
    }

    fn sample(&self, draw: f32) -> u32 {
        sample_weighted(
            self.candidates.iter().copied(),
            draw,
            "target distribution is empty",
        )
        .expect("validated target distribution")
    }

    fn sample_without(&self, excluded: u32, draw: f32) -> Result<u32> {
        sample_weighted(
            self.candidates
                .iter()
                .copied()
                .filter(|(token, _)| *token != excluded),
            draw,
            "speculative residual distribution is empty after rejection",
        )
    }

    fn sample_residual_against(&self, draft: &Self, draw: f32) -> Result<u32> {
        sample_weighted(
            self.candidates
                .iter()
                .filter_map(|(token, target_probability)| {
                    let residual = *target_probability - draft.probability(*token);
                    (residual > 0.0).then_some((*token, residual))
                }),
            draw,
            "speculative residual distribution is empty after rejection",
        )
    }
}

fn apply_history_penalties(
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
    let mut selected = None;
    let mut selected_value = f32::NEG_INFINITY;
    for (index, value) in values.iter().copied().enumerate() {
        if value.is_finite() && value > selected_value {
            selected = Some(index);
            selected_value = value;
        }
    }
    selected
        .ok_or_else(no_finite_logits_error)
        .and_then(|index| {
            u32::try_from(index).map_err(|_| {
                Error::InvalidInput("speculative argmax token exceeds u32::MAX".into())
            })
        })
}

fn normalize(candidates: &mut [(u32, f32)]) -> Result<()> {
    let sum = candidates
        .iter()
        .map(|(_, probability)| *probability)
        .sum::<f32>();
    if candidates.is_empty() || !sum.is_finite() || sum <= 0.0 {
        return Err(no_finite_logits_error());
    }
    for (_, probability) in candidates {
        *probability /= sum;
    }
    Ok(())
}

fn sample_weighted(
    candidates: impl Iterator<Item = (u32, f32)>,
    draw: f32,
    empty_message: &'static str,
) -> Result<u32> {
    let candidates = candidates
        .filter(|(_, probability)| probability.is_finite() && *probability > 0.0)
        .collect::<Vec<_>>();
    let sum = candidates
        .iter()
        .map(|(_, probability)| *probability)
        .sum::<f32>();
    if candidates.is_empty() || !sum.is_finite() || sum <= 0.0 {
        return Err(Error::InferenceError(empty_message.into()));
    }

    let threshold = draw.clamp(0.0, 1.0 - f32::EPSILON) * sum;
    let mut cumulative = 0.0f32;
    for (token, probability) in &candidates {
        cumulative += *probability;
        if threshold < cumulative {
            return Ok(*token);
        }
    }
    Ok(candidates.last().expect("non-empty candidates").0)
}

fn next_unit_f32(rng: &mut impl RngCore) -> f32 {
    ((rng.next_u32() >> 8) as f32) * F32_UNIT_SCALE
}

fn no_finite_logits_error() -> Error {
    Error::InferenceError("speculative sampler found no finite target logits".into())
}

#[cfg(test)]
mod tests {
    use rand::{Error as RandError, RngCore};

    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct ScriptedRng {
        draws: Vec<u32>,
        cursor: usize,
    }

    impl ScriptedRng {
        fn new(draws: impl IntoIterator<Item = f32>) -> Self {
            Self {
                draws: draws
                    .into_iter()
                    .map(|draw| {
                        let quantized =
                            (draw.clamp(0.0, 1.0 - f32::EPSILON) * ((1u32 << 24) as f32)) as u32;
                        quantized << 8
                    })
                    .collect(),
                cursor: 0,
            }
        }
    }

    impl RngCore for ScriptedRng {
        fn next_u32(&mut self) -> u32 {
            let value = self.draws.get(self.cursor).copied().unwrap_or(0);
            self.cursor += 1;
            value
        }

        fn next_u64(&mut self) -> u64 {
            (u64::from(self.next_u32()) << 32) | u64::from(self.next_u32())
        }

        fn fill_bytes(&mut self, dest: &mut [u8]) {
            for chunk in dest.chunks_mut(std::mem::size_of::<u32>()) {
                let bytes = self.next_u32().to_le_bytes();
                chunk.copy_from_slice(&bytes[..chunk.len()]);
            }
        }

        fn try_fill_bytes(&mut self, dest: &mut [u8]) -> std::result::Result<(), RandError> {
            self.fill_bytes(dest);
            Ok(())
        }
    }

    fn stochastic_config() -> ChatGenerationConfig {
        ChatGenerationConfig {
            temperature: 1.0,
            ..ChatGenerationConfig::default()
        }
    }

    #[test]
    fn greedy_verification_accepts_prefix_and_emits_bonus() {
        let mut history = vec![3];
        let verification = verify_greedy_prefix(
            &[1, 2],
            &[
                vec![0.0, 3.0, 1.0, -1.0],
                vec![0.0, 1.0, 4.0, -1.0],
                vec![5.0, 1.0, 0.0, -1.0],
            ],
            &ChatGenerationConfig::default(),
            &mut history,
        )
        .unwrap();

        assert_eq!(verification.emitted_tokens, vec![1, 2, 0]);
        assert_eq!(verification.accepted_draft_tokens, 2);
        assert!(verification.all_drafts_accepted());
        assert!(verification.emitted_bonus_token());
        assert_eq!(history, vec![3, 1, 2, 0]);
    }

    #[test]
    fn greedy_verification_emits_first_target_mismatch_after_penalties() {
        let config = ChatGenerationConfig {
            repetition_penalty: 2.0,
            presence_penalty: 0.5,
            ..ChatGenerationConfig::default()
        };
        let mut history = vec![0];
        let verification = verify_greedy_prefix(
            &[0],
            &[vec![4.0, 3.0], vec![0.0, 1.0]],
            &config,
            &mut history,
        )
        .unwrap();

        assert_eq!(verification.emitted_tokens, vec![1]);
        assert_eq!(verification.accepted_draft_tokens, 0);
        assert!(!verification.all_drafts_accepted());
        assert_eq!(history, vec![0, 1]);
    }

    #[test]
    fn rejection_sampling_uses_one_hot_residual_distribution() {
        let mut history = vec![7];
        let mut rng = ScriptedRng::new([0.9, 0.8]);
        let verification = verify_rejection_sampled_prefix(
            &[0],
            &[
                vec![0.2f32.ln(), 0.3f32.ln(), 0.5f32.ln()],
                vec![0.0, 0.0, 0.0],
            ],
            &stochastic_config(),
            &mut history,
            &mut rng,
        )
        .unwrap();

        // q is one-hot on token 0. After rejection the residual weights are
        // [token 1: 0.3, token 2: 0.5]. In descending-probability order,
        // draw 0.8 passes token 2's mass and selects token 1.
        assert_eq!(verification.emitted_tokens, vec![1]);
        assert_eq!(verification.accepted_draft_tokens, 0);
        assert_eq!(history, vec![7, 1]);
        assert_eq!(rng.cursor, 2);
    }

    #[test]
    fn full_stochastic_acceptance_samples_target_bonus() {
        let config = ChatGenerationConfig {
            temperature: 1.0,
            top_k: 1,
            ..ChatGenerationConfig::default()
        };
        let mut history = Vec::new();
        let mut rng = ScriptedRng::new([0.75, 0.25]);
        let verification = verify_rejection_sampled_prefix(
            &[0],
            &[vec![4.0, 1.0], vec![0.0, 5.0]],
            &config,
            &mut history,
            &mut rng,
        )
        .unwrap();

        assert_eq!(verification.emitted_tokens, vec![0, 1]);
        assert!(verification.all_drafts_accepted());
        assert_eq!(history, vec![0, 1]);
        assert_eq!(rng.cursor, 2);
    }

    #[test]
    fn target_distribution_matches_temperature_top_k_and_top_p_contract() {
        let history = vec![0];
        let config = ChatGenerationConfig {
            temperature: 1.0,
            top_k: 2,
            top_p: 0.7,
            repetition_penalty: 2.0,
            presence_penalty: 0.5,
            ..ChatGenerationConfig::default()
        };
        let distribution =
            TargetDistribution::from_logits(&[4.0, 3.0, 2.0], &config, &history).unwrap();

        // Token 0 is adjusted from 4.0 to 1.5. top-k keeps tokens 1 and 2;
        // their softmax mass on token 1 exceeds top-p, leaving it one-hot.
        assert_eq!(distribution.candidates, vec![(1, 1.0)]);

        let cold = TargetDistribution::from_logits(
            &[0.0, 1.0],
            &ChatGenerationConfig {
                temperature: 0.5,
                ..ChatGenerationConfig::default()
            },
            &[],
        )
        .unwrap();
        let warm = TargetDistribution::from_logits(
            &[0.0, 1.0],
            &ChatGenerationConfig {
                temperature: 2.0,
                ..ChatGenerationConfig::default()
            },
            &[],
        )
        .unwrap();
        assert!(cold.probability(1) > warm.probability(1));
    }

    #[test]
    fn failed_verification_rolls_back_history_and_rng() {
        let config = ChatGenerationConfig {
            temperature: 1.0,
            top_k: 1,
            ..ChatGenerationConfig::default()
        };
        let mut history = vec![9];
        let initial_history = history.clone();
        let mut rng = ScriptedRng::new([0.5, 0.5]);
        let initial_rng = rng.clone();

        let error = verify_rejection_sampled_prefix(
            &[0],
            &[vec![3.0, 1.0], vec![f32::NAN, f32::NEG_INFINITY]],
            &config,
            &mut history,
            &mut rng,
        )
        .unwrap_err();

        assert!(error.to_string().contains("no finite target logits"));
        assert_eq!(history, initial_history);
        assert_eq!(rng, initial_rng);
    }

    #[test]
    fn automatic_greedy_mode_does_not_advance_rng() {
        let mut history = Vec::new();
        let mut rng = ScriptedRng::new([0.5]);
        let initial_rng = rng.clone();
        let verification = verify_speculative_prefix(
            &[0],
            &[vec![2.0, 1.0], vec![0.0, 3.0]],
            &ChatGenerationConfig::default(),
            &mut history,
            &mut rng,
        )
        .unwrap();

        assert_eq!(verification.emitted_tokens, vec![0, 1]);
        assert_eq!(rng, initial_rng);
    }

    #[test]
    fn stochastic_proposals_use_p_over_q_and_residual_correction() {
        let config = ChatGenerationConfig {
            temperature: 1.0,
            ..ChatGenerationConfig::default()
        };
        let mut draft_history = Vec::new();
        let mut draft_rng = ScriptedRng::new([0.1]);
        let proposal = propose_speculative_draft(
            &[4.0f32.ln(), 1.0f32.ln()],
            &config,
            &mut draft_history,
            &mut draft_rng,
        )
        .unwrap();
        assert_eq!(proposal.token_id, 0);
        assert_eq!(draft_history, vec![0]);

        // q(0)=0.8 and p(0)=0.4, so the acceptance probability is 0.5.
        // A 0.75 draw rejects token 0; max(p-q, 0) contains only token 1.
        let mut target_history = Vec::new();
        let mut target_rng = ScriptedRng::new([0.75, 0.3]);
        let verification = verify_speculative_proposals(
            &[proposal],
            &[vec![2.0f32.ln(), 3.0f32.ln()], vec![1.0, 0.0]],
            &config,
            &mut target_history,
            &mut target_rng,
        )
        .unwrap();

        assert_eq!(verification.emitted_tokens, vec![1]);
        assert_eq!(verification.accepted_draft_tokens, 0);
        assert_eq!(target_history, vec![1]);
        assert_eq!(target_rng.cursor, 2);
    }

    #[test]
    fn stochastic_proposal_and_target_rng_streams_are_independent() {
        let config = ChatGenerationConfig {
            temperature: 1.0,
            ..ChatGenerationConfig::default()
        };
        let mut draft_history = Vec::new();
        let mut draft_rng = ScriptedRng::new([0.1]);
        let proposal = propose_speculative_draft(
            &[4.0f32.ln(), 1.0f32.ln()],
            &config,
            &mut draft_history,
            &mut draft_rng,
        )
        .unwrap();
        let draft_rng_after_proposal = draft_rng.clone();

        let mut target_history = Vec::new();
        let mut target_rng = ScriptedRng::new([0.1, 0.9]);
        let verification = verify_speculative_proposals(
            &[proposal],
            &[
                vec![9.0f32.ln(), 1.0f32.ln()],
                vec![1.0f32.ln(), 9.0f32.ln()],
            ],
            &config,
            &mut target_history,
            &mut target_rng,
        )
        .unwrap();

        assert_eq!(verification.emitted_tokens, vec![0, 1]);
        assert!(verification.all_drafts_accepted());
        assert_eq!(draft_rng, draft_rng_after_proposal);
        assert_eq!(target_rng.cursor, 2);
    }

    #[test]
    fn malformed_block_does_not_commit_partial_greedy_history() {
        let mut history = vec![4];
        let initial_history = history.clone();
        let error = verify_greedy_prefix(
            &[0],
            &[vec![2.0, 1.0], vec![f32::NAN, f32::INFINITY]],
            &ChatGenerationConfig::default(),
            &mut history,
        )
        .unwrap_err();

        assert!(error.to_string().contains("no finite target logits"));
        assert_eq!(history, initial_history);
    }
}
