//! Sparse expert (MoE) dispatch seam — DS10 groundwork.
//!
//! The routing/dispach/combine logic lives here so that every family using
//! sparse experts shares one implementation, and so expert parallelism later
//! becomes an alternative [`ExpertSet`] implementation rather than a rewrite
//! of the routing math (the vLLM `MoELayer`/`FusedMoE` split). Expert MLP
//! internals stay family-native: a family adapts its own projection machinery
//! to [`ExpertSet`] and calls [`SparseMoeDispatcher::dispatch`] with its
//! router logits.
//!
//! The single-device reference dispatch routes host-side (weights and top-k
//! indices are read to the host once per step) and applies experts with
//! tensor-level gather/scatter on the model device. Per-expert activation
//! counters are the input every production load balancer (EPLB-style) needs;
//! recording them costs one counter array per step.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use candle_core::{D, DType, Tensor};
use candle_nn::ops;

use crate::error::{Error, Result};

/// Resolved sparse-expert geometry for one model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseMoeConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
}

impl SparseMoeConfig {
    pub fn validate(&self) -> Result<()> {
        if self.num_experts == 0 {
            return Err(Error::ModelLoadError(
                "sparse MoE config requires at least one expert".into(),
            ));
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts {
            return Err(Error::ModelLoadError(format!(
                "sparse MoE experts_per_tok {} must be within 1..={}",
                self.num_experts_per_tok, self.num_experts
            )));
        }
        Ok(())
    }
}

/// The expert dimension of a sparse block. Implementations own the expert
/// weights; the dispatcher only addresses experts by index.
pub trait ExpertSet: Send + Sync {
    fn num_experts(&self) -> usize;

    /// Apply expert `expert` to the gathered token rows
    /// `[n_selected, hidden_dim]`, returning `[n_selected, hidden_dim]`.
    fn apply_expert(&self, expert: usize, tokens: &Tensor) -> Result<Tensor>;
}

impl ExpertSet for Vec<Arc<dyn ExpertSet>> {
    fn num_experts(&self) -> usize {
        self.len()
    }

    fn apply_expert(&self, expert: usize, tokens: &Tensor) -> Result<Tensor> {
        let set = self.get(expert).ok_or_else(|| {
            Error::InferenceError(format!("sparse MoE expert {expert} is out of range"))
        })?;
        set.apply_expert(expert, tokens)
    }
}

/// Per-expert selection counters. One array per sparse block; selections are
/// recorded per dispatch step so a step's histogram is the exact routing
/// decision (tokens × experts_per_tok increments).
#[derive(Debug, Default)]
pub struct ExpertActivationCounters {
    counts: Vec<AtomicU64>,
}

impl ExpertActivationCounters {
    pub fn new(num_experts: usize) -> Self {
        Self {
            counts: (0..num_experts).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    pub fn num_experts(&self) -> usize {
        self.counts.len()
    }

    pub fn record(&self, expert: usize, selections: u64) {
        if let Some(counter) = self.counts.get(expert) {
            counter.fetch_add(selections, Ordering::Relaxed);
        }
    }

    pub fn snapshot(&self) -> Vec<u64> {
        self.counts
            .iter()
            .map(|counter| counter.load(Ordering::Relaxed))
            .collect()
    }

    pub fn total_selections(&self) -> u64 {
        self.snapshot().into_iter().sum()
    }
}

/// Reference single-device sparse dispatch: softmax router → top-k → optional
/// in-top-k renormalization → per-expert gather/apply/weighted-scatter.
#[derive(Debug, Clone)]
pub struct SparseMoeDispatcher {
    config: SparseMoeConfig,
    counters: Option<Arc<ExpertActivationCounters>>,
}

impl SparseMoeDispatcher {
    pub fn new(config: SparseMoeConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            counters: None,
        })
    }

    pub fn with_counters(mut self, counters: Arc<ExpertActivationCounters>) -> Self {
        self.counters = Some(counters);
        self
    }

    pub fn config(&self) -> &SparseMoeConfig {
        &self.config
    }

    pub fn counters(&self) -> Option<&Arc<ExpertActivationCounters>> {
        self.counters.as_ref()
    }

    /// Route `hidden` ([num_tokens, hidden_dim]) through the experts using
    /// `router_logits` ([num_tokens, num_experts]); returns the combined
    /// hidden states with the input shape and dtype.
    pub fn dispatch(
        &self,
        hidden: &Tensor,
        router_logits: &Tensor,
        experts: &dyn ExpertSet,
    ) -> Result<Tensor> {
        if experts.num_experts() != self.config.num_experts {
            return Err(Error::InferenceError(format!(
                "sparse MoE dispatcher expects {} experts, expert set holds {}",
                self.config.num_experts,
                experts.num_experts()
            )));
        }
        let (num_tokens, _num_experts) = router_logits.dims2()?;
        let (hidden_tokens, hidden_dim) = hidden.dims2()?;
        if hidden_tokens != num_tokens {
            return Err(Error::InferenceError(format!(
                "sparse MoE hidden {:?} does not align with router logits {:?}",
                hidden.dims(),
                router_logits.dims()
            )));
        }

        let routing_weights = ops::softmax_last_dim(&router_logits.to_dtype(DType::F32)?)?;
        let topk_indices = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.config.num_experts_per_tok)?
            .contiguous()?;
        let topk_weights = routing_weights.gather(&topk_indices, D::Minus1)?;
        let topk_weights = if self.config.norm_topk_prob {
            topk_weights.broadcast_div(&topk_weights.sum_keepdim(D::Minus1)?)?
        } else {
            topk_weights
        };

        let weights = topk_weights.to_vec2::<f32>()?;
        let indices = topk_indices.to_vec2::<u32>()?;
        let mut rows_by_expert: Vec<Vec<u32>> = vec![Vec::new(); self.config.num_experts];
        let mut weights_by_expert: Vec<Vec<f32>> = vec![Vec::new(); self.config.num_experts];
        for (row, (index_row, weight_row)) in indices.iter().zip(&weights).enumerate() {
            for (&expert, &weight) in index_row.iter().zip(weight_row) {
                rows_by_expert[expert as usize].push(row as u32);
                weights_by_expert[expert as usize].push(weight);
            }
        }
        if let Some(counters) = &self.counters {
            for (expert, rows) in rows_by_expert.iter().enumerate() {
                counters.record(expert, rows.len() as u64);
            }
        }

        let mut output = Tensor::zeros((num_tokens, hidden_dim), hidden.dtype(), hidden.device())?;
        for (expert, rows) in rows_by_expert.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let row_ids = Tensor::from_vec(rows.clone(), (rows.len(),), hidden.device())?;
            let selected = hidden.index_select(&row_ids, 0)?;
            let applied = experts.apply_expert(expert, &selected)?;
            if applied.dims() != selected.dims() {
                return Err(Error::InferenceError(format!(
                    "sparse MoE expert {expert} returned {:?} for input {:?}",
                    applied.dims(),
                    selected.dims()
                )));
            }
            let gate = Tensor::from_vec(
                weights_by_expert[expert].clone(),
                (rows.len(), 1),
                hidden.device(),
            )?
            .to_dtype(hidden.dtype())?;
            output = output.index_add(&row_ids, &applied.broadcast_mul(&gate)?, 0)?;
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use std::sync::Mutex;

    /// Identity expert scaled by (expert + 1): output rows are the input rows
    /// times (e + 1), so a whole dispatch is computable in plain f32 math.
    struct ScalingExperts {
        num_experts: usize,
        calls: Mutex<Vec<usize>>,
    }

    impl ScalingExperts {
        fn new(num_experts: usize) -> Self {
            Self {
                num_experts,
                calls: Mutex::new(Vec::new()),
            }
        }

        fn called_experts(&self) -> Vec<usize> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl ExpertSet for ScalingExperts {
        fn num_experts(&self) -> usize {
            self.num_experts
        }

        fn apply_expert(&self, expert: usize, tokens: &Tensor) -> Result<Tensor> {
            self.calls.lock().unwrap().push(expert);
            let scale = Tensor::from_vec(
                vec![(expert + 1) as f32],
                (1,),
                tokens.device(),
            )?
            .reshape((1, 1))?;
            Ok(tokens.broadcast_mul(&scale)?)
        }
    }

    fn softmax_row(values: &[f32]) -> Vec<f32> {
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = values.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|v| v / sum).collect()
    }

    fn reference_dispatch(
        hidden: &[f32],
        logits: &[f32],
        num_tokens: usize,
        hidden_dim: usize,
        num_experts: usize,
        top_k: usize,
        norm_topk_prob: bool,
    ) -> Vec<f32> {
        let mut output = vec![0_f32; num_tokens * hidden_dim];
        for token in 0..num_tokens {
            let probs = softmax_row(&logits[token * num_experts..(token + 1) * num_experts]);
            let mut ranked: Vec<(usize, f32)> =
                probs.iter().copied().enumerate().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            let selected = &ranked[..top_k];
            let norm = if norm_topk_prob {
                selected.iter().map(|(_, w)| *w).sum::<f32>()
            } else {
                1.0
            };
            for &(expert, weight) in selected {
                for dim in 0..hidden_dim {
                    output[token * hidden_dim + dim] +=
                        (weight / norm) * ((expert + 1) as f32) * hidden[token * hidden_dim + dim];
                }
            }
        }
        output
    }

    fn run_dispatch(
        norm_topk_prob: bool,
        top_k: usize,
    ) -> (Vec<f32>, ScalingExperts, Arc<ExpertActivationCounters>) {
        // Distinct logits avoid arg-sort tie ambiguity.
        let logits = vec![1.5_f32, -0.5, 3.0, 0.25, 2.0, -1.0, 0.75, 0.1, 2.5, -2.0, 1.0, 4.0];
        let hidden = vec![1.0_f32, -2.0, 0.5, 3.0, -1.5, 2.5];
        let num_tokens = 3;
        let hidden_dim = 2;
        let num_experts = 4;

        let experts = ScalingExperts::new(num_experts);
        let counters = Arc::new(ExpertActivationCounters::new(num_experts));
        let dispatcher = SparseMoeDispatcher::new(SparseMoeConfig {
            num_experts,
            num_experts_per_tok: top_k,
            norm_topk_prob,
        })
        .unwrap()
        .with_counters(counters.clone());

        let hidden = Tensor::from_vec(hidden.clone(), (num_tokens, hidden_dim), &Device::Cpu)
            .unwrap();
        let logits = Tensor::from_vec(logits.clone(), (num_tokens, num_experts), &Device::Cpu)
            .unwrap();
        let output = dispatcher.dispatch(&hidden, &logits, &experts).unwrap();
        let output = output.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        let output = output.to_vec1::<f32>().unwrap();
        (output, experts, counters)
    }

    #[test]
    fn dispatch_matches_reference_with_renormalized_topk() {
        let (output, experts, counters) = run_dispatch(true, 2);
        let reference = reference_dispatch(
            &[1.0, -2.0, 0.5, 3.0, -1.5, 2.5],
            &[1.5, -0.5, 3.0, 0.25, 2.0, -1.0, 0.75, 0.1, 2.5, -2.0, 1.0, 4.0],
            3,
            2,
            4,
            2,
            true,
        );
        for (actual, expected) in output.iter().zip(&reference) {
            assert!((actual - expected).abs() < 1e-5, "{actual} vs {expected}");
        }
        // Two of four experts saw tokens; every token selected two experts.
        assert_eq!(counters.total_selections(), 6);
        assert_eq!(experts.called_experts().len(), 3);
    }

    #[test]
    fn dispatch_keeps_raw_softmax_weights_without_renormalization() {
        let (output, _experts, counters) = run_dispatch(false, 1);
        let reference = reference_dispatch(
            &[1.0, -2.0, 0.5, 3.0, -1.5, 2.5],
            &[1.5, -0.5, 3.0, 0.25, 2.0, -1.0, 0.75, 0.1, 2.5, -2.0, 1.0, 4.0],
            3,
            2,
            4,
            1,
            false,
        );
        for (actual, expected) in output.iter().zip(&reference) {
            assert!((actual - expected).abs() < 1e-5, "{actual} vs {expected}");
        }
        assert_eq!(counters.total_selections(), 3);
    }

    #[test]
    fn unselected_experts_are_never_applied() {
        let hidden = Tensor::from_vec(vec![1.0_f32, 2.0], (1, 2), &Device::Cpu).unwrap();
        let logits = Tensor::from_vec(vec![5.0_f32, 0.1, 0.2, 0.3, 0.4], (1, 5), &Device::Cpu)
            .unwrap();
        let experts = ScalingExperts::new(5);
        let dispatcher =
            SparseMoeDispatcher::new(SparseMoeConfig {
                num_experts: 5,
                num_experts_per_tok: 1,
                norm_topk_prob: false,
            })
            .unwrap();
        dispatcher.dispatch(&hidden, &logits, &experts).unwrap();
        assert_eq!(experts.called_experts(), vec![0]);
    }

    #[test]
    fn counters_snapshot_records_exact_buckets() {
        let _ = run_dispatch(true, 2);
        // Re-run with access to the dispatcher's own counters to pin bucket
        // arithmetic: 3 tokens × top-2 selections.
        let logits = Tensor::from_vec(
            vec![1.5_f32, -0.5, 3.0, 0.25, 2.0, -1.0, 0.75, 0.1, 2.5, -2.0, 1.0, 4.0],
            (3, 4),
            &Device::Cpu,
        )
        .unwrap();
        let hidden = Tensor::from_vec(vec![1.0_f32, -2.0, 0.5, 3.0, -1.5, 2.5], (3, 2), &Device::Cpu)
            .unwrap();
        let experts = ScalingExperts::new(4);
        let counters = Arc::new(ExpertActivationCounters::new(4));
        let dispatcher = SparseMoeDispatcher::new(SparseMoeConfig {
            num_experts: 4,
            num_experts_per_tok: 2,
            norm_topk_prob: true,
        })
        .unwrap()
        .with_counters(counters.clone());
        dispatcher.dispatch(&hidden, &logits, &experts).unwrap();
        assert_eq!(counters.snapshot().iter().sum::<u64>(), 6);
        assert_eq!(counters.num_experts(), 4);
    }

    #[test]
    fn config_and_geometry_mismatches_fail_closed() {
        assert!(SparseMoeDispatcher::new(SparseMoeConfig {
            num_experts: 0,
            num_experts_per_tok: 1,
            norm_topk_prob: false,
        })
        .is_err());
        assert!(SparseMoeDispatcher::new(SparseMoeConfig {
            num_experts: 4,
            num_experts_per_tok: 5,
            norm_topk_prob: false,
        })
        .is_err());

        let hidden = Tensor::from_vec(vec![1.0_f32, 2.0], (1, 2), &Device::Cpu).unwrap();
        let logits = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], (1, 4), &Device::Cpu).unwrap();
        let dispatcher = SparseMoeDispatcher::new(SparseMoeConfig {
            num_experts: 4,
            num_experts_per_tok: 1,
            norm_topk_prob: false,
        })
        .unwrap();
        // Expert set with the wrong count is rejected before routing.
        let error = dispatcher
            .dispatch(&hidden, &logits, &ScalingExperts::new(3))
            .unwrap_err();
        assert!(format!("{error}").contains("expert set holds 3"));
        // Router/hidden shape disagreement is rejected.
        let short_hidden = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], (3, 1), &Device::Cpu).unwrap();
        let error = dispatcher
            .dispatch(&short_hidden, &logits, &ScalingExperts::new(4))
            .unwrap_err();
        assert!(format!("{error}").contains("does not align"));
    }
}
