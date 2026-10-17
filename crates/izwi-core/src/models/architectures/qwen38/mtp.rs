//! Recurrent multi-token-prediction head for native Qwen3.8 checkpoints.

use std::ops::ControlFlow;
use std::sync::Arc;

use candle_core::{DType, Device, Module, Tensor};

use super::fast_path::{summarize, Qwen38CudaSwitches, Qwen38FastPath};
use super::native::{
    IndexedSafetensors, ProjectionMaterialization, Qwen38MtpInventory, Qwen38NativeConfig,
    QWEN38_MTP_TENSOR_COUNT,
};
use super::residency::Qwen38ResidencyPolicy;
use super::text::{
    load_native_projection, load_native_zero_centered_norm, norm_self_check, resolve_norm_path,
    Qwen38FullAttention, Qwen38Mlp, Qwen38Projection, Qwen38ProjectionRepresentation,
    Qwen38RmsNorm, Qwen38RopeInputs, Qwen38TextModel, Qwen38WeightLoader,
};
use crate::backends::kv::KvWriteCompletionCollector;
use crate::error::{Error, Result};
use crate::kv::KvDecodeBatchMetadata;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

pub(crate) const QWEN38_MTP_MIN_DEPTH: usize = 1;
pub(crate) const QWEN38_MTP_MAX_DEPTH: usize = 3;

/// Validated number of recurrent predictions requested from the one physical
/// Qwen3.8 MTP layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen38MtpDepth(usize);

impl Qwen38MtpDepth {
    pub(crate) fn new(depth: usize) -> Result<Self> {
        if !(QWEN38_MTP_MIN_DEPTH..=QWEN38_MTP_MAX_DEPTH).contains(&depth) {
            return Err(Error::InvalidInput(format!(
                "Qwen3.8 MTP recurrent depth must be in {QWEN38_MTP_MIN_DEPTH}..={QWEN38_MTP_MAX_DEPTH}, got {depth}"
            )));
        }
        Ok(Self(depth))
    }

    pub(crate) const fn get(self) -> usize {
        self.0
    }
}

impl TryFrom<usize> for Qwen38MtpDepth {
    type Error = Error;

    fn try_from(value: usize) -> Result<Self> {
        Self::new(value)
    }
}

/// Per-request latency controller. It compares elapsed time per committed
/// token, including draft, verification and prefix commit. Exploration is
/// bounded to four arms (scalar and depths 1..3), one probe every eight rounds.
/// Scheduler-limited tails do not train the controller.
#[derive(Clone, Debug)]
pub(crate) struct AdaptiveMtp {
    enabled: bool,
    speculation_disabled: bool,
    fixed_depth: usize,
    selected: usize,
    samples: [u32; 4],
    cost_per_token: [f64; 4],
    rounds: u64,
    probe: usize,
}

impl AdaptiveMtp {
    pub(crate) fn new(enabled: bool, starting_depth: usize) -> Self {
        let depth = starting_depth.clamp(1, 3);
        Self {
            enabled,
            speculation_disabled: false,
            fixed_depth: depth,
            selected: depth,
            samples: [0; 4],
            cost_per_token: [0.0; 4],
            rounds: 0,
            probe: 0,
        }
    }

    /// A numerical draft failure disables speculation for the entire request,
    /// including fixed-depth mode and any delayed timing observations.
    pub(crate) fn disable_after_nonfinite_draft(&mut self) {
        self.speculation_disabled = true;
    }

    pub(crate) fn speculation_disabled(&self) -> bool {
        self.speculation_disabled
    }

    /// Roll back timing policy without forgetting a numerical failure observed
    /// either before the checkpoint or in the cancelled quantum.
    pub(crate) fn restore_from_checkpoint(&mut self, checkpoint: Self) {
        let speculation_disabled = self.speculation_disabled || checkpoint.speculation_disabled;
        *self = checkpoint;
        self.speculation_disabled = speculation_disabled;
    }

    pub(crate) fn can_train(&self, budget: usize) -> bool {
        !self.speculation_disabled() && self.enabled && budget >= 4
    }

    pub(crate) fn depth(&self, budget: usize) -> usize {
        if self.speculation_disabled {
            return 0;
        }
        let ceiling = budget.saturating_sub(1).min(3);
        if !self.enabled {
            return self.fixed_depth.min(ceiling);
        }
        let arm = if self.rounds > 0 && self.rounds.is_multiple_of(8) {
            self.probe
        } else {
            self.selected
        };
        arm.min(ceiling)
    }

    pub(crate) fn observe(
        &mut self,
        depth: usize,
        committed: usize,
        elapsed: std::time::Duration,
        budget: usize,
    ) {
        if self.speculation_disabled
            || !self.enabled
            || depth > 3
            || committed == 0
            || budget < 4
            || elapsed.is_zero()
        {
            return;
        }
        let cost = elapsed.as_secs_f64() / committed as f64;
        self.cost_per_token[depth] = if self.samples[depth] == 0 {
            cost
        } else {
            self.cost_per_token[depth] * 0.75 + cost * 0.25
        };
        self.samples[depth] = self.samples[depth].saturating_add(1);
        if self.rounds > 0 && self.rounds.is_multiple_of(8) {
            self.probe = (self.probe + 1) % 4;
        }
        self.rounds = self.rounds.saturating_add(1);
        // Require a 5% advantage before switching to reduce timer noise churn.
        for candidate in 0..4 {
            if self.samples[candidate] > 0
                && self.cost_per_token[candidate] < self.cost_per_token[self.selected] * 0.95
            {
                self.selected = candidate;
            }
        }
    }
}

/// Aligned token embeddings, predecessor hidden states, and rotary positions
/// for one MTP invocation.
pub(crate) struct Qwen38MtpPairBatch {
    embeddings: Tensor,
    hidden_states: Tensor,
    position_ids: Vec<[usize; 3]>,
}

impl Qwen38MtpPairBatch {
    pub(crate) fn new(
        embeddings: Tensor,
        hidden_states: Tensor,
        position_ids: Vec<[usize; 3]>,
    ) -> Result<Self> {
        let embedding_dims = embeddings.dims3().map_err(|_| {
            Error::InvalidInput("Qwen3.8 MTP embeddings must have shape [1, tokens, hidden]".into())
        })?;
        let hidden_dims = hidden_states.dims3().map_err(|_| {
            Error::InvalidInput(
                "Qwen3.8 MTP predecessor hidden states must have shape [1, tokens, hidden]".into(),
            )
        })?;
        if embedding_dims.0 != 1
            || embedding_dims != hidden_dims
            || embedding_dims.1 == 0
            || embedding_dims.1 != position_ids.len()
        {
            return Err(Error::InvalidInput(format!(
                "Qwen3.8 MTP pair mismatch: embeddings {embedding_dims:?}, hidden {hidden_dims:?}, positions {}",
                position_ids.len()
            )));
        }
        Ok(Self {
            embeddings,
            hidden_states,
            position_ids,
        })
    }

    /// Construct the prompt-length shifted MTP prefill used by Qwen3.8.
    ///
    /// For target prompt embeddings `[e0, e1, ..., eN-1]` and target residuals
    /// `[h0, h1, ..., hN-1]`, the MTP inputs are
    /// `[e1, ..., eN-1, e_next]` paired with the unshifted target residuals.
    /// The target positions are retained unchanged. Supplying the target-
    /// produced `e_next` final row keeps the MTP KV cursor at exactly N rather
    /// than silently lagging the target cursor by one token.
    pub(crate) fn shifted_prompt(
        prompt_embeddings: &Tensor,
        target_hidden_states: &Tensor,
        next_token_embedding: &Tensor,
        target_position_ids: &[[usize; 3]],
    ) -> Result<Self> {
        let (batch, prompt_len, hidden) = prompt_embeddings.dims3().map_err(|_| {
            Error::InvalidInput(
                "Qwen3.8 MTP prompt embeddings must have shape [1, tokens, hidden]".into(),
            )
        })?;
        if batch != 1 || prompt_len == 0 {
            return Err(Error::InvalidInput(
                "Qwen3.8 MTP shifted prompt requires one non-empty sequence".into(),
            ));
        }
        if target_hidden_states.dims3()? != (batch, prompt_len, hidden) {
            return Err(Error::InvalidInput(
                "Qwen3.8 MTP target hidden span does not match prompt embeddings".into(),
            ));
        }
        if next_token_embedding.dims3()? != (1, 1, hidden) {
            return Err(Error::InvalidInput(format!(
                "Qwen3.8 MTP next-token embedding must have shape [1, 1, {hidden}]"
            )));
        }
        if target_position_ids.len() != prompt_len {
            return Err(Error::InvalidInput(format!(
                "Qwen3.8 MTP shifted prompt has {prompt_len} tokens but {} positions",
                target_position_ids.len()
            )));
        }
        let shifted_embeddings = if prompt_len == 1 {
            next_token_embedding.clone()
        } else {
            let tail = prompt_embeddings.narrow(1, 1, prompt_len - 1)?;
            Tensor::cat(&[&tail, next_token_embedding], 1)?
        };
        Self::new(
            shifted_embeddings,
            target_hidden_states.clone(),
            target_position_ids.to_vec(),
        )
    }

    pub(crate) fn single(
        embedding: Tensor,
        predecessor_hidden: Tensor,
        position_id: [usize; 3],
    ) -> Result<Self> {
        Self::new(embedding, predecessor_hidden, vec![position_id])
    }

    pub(crate) fn embeddings(&self) -> &Tensor {
        &self.embeddings
    }

    pub(crate) fn hidden_states(&self) -> &Tensor {
        &self.hidden_states
    }

    pub(crate) fn position_ids(&self) -> &[[usize; 3]] {
        &self.position_ids
    }

    pub(crate) fn token_count(&self) -> usize {
        self.position_ids.len()
    }
}

/// Draft tokens together with the normalized MTP hidden row that produced
/// each token through the shared target LM head.
pub(crate) struct Qwen38MtpDraftSequence {
    pub token_ids: Vec<u32>,
    pub lm_head_hidden: Vec<Tensor>,
}

/// The checkpoint-backed Qwen3.8 recurrent MTP head.
///
/// This owns only the 22 `mtp.*` tensors. Token embeddings and the LM head stay
/// owned by [`Qwen38TextModel`] and are accessed through explicit shared-I/O
/// helpers, preserving the checkpoint's non-dedicated-embedding contract.
pub(crate) struct Qwen38MtpHead {
    device: Device,
    projection_representation: Qwen38ProjectionRepresentation,
    hidden_size: usize,
    model_layer: u32,
    pre_fc_norm_embedding: Qwen38RmsNorm,
    pre_fc_norm_hidden: Qwen38RmsNorm,
    fc: Qwen38Projection,
    input_layernorm: Qwen38RmsNorm,
    attention: Qwen38FullAttention,
    post_attention_layernorm: Qwen38RmsNorm,
    mlp: Qwen38Mlp,
    norm: Qwen38RmsNorm,
    /// The plan's activation dtype.
    activation_dtype: DType,
    /// Fused residual add + RMSNorm over the head's own norms.
    fused_norm_path: Qwen38FastPath,
    /// One entry per block-FP8 projection group: packed, or why not.
    fp8_packing: Vec<Qwen38FastPath>,
    /// Metal: the GEMV check outcome of every FP8 projection kind.
    fp8_gemv_checks: Vec<Qwen38FastPath>,
    /// CUDA: the dequantized prefill GEMM of every FP8 projection kind.
    fp8_prefill_gemm_checks: Vec<Qwen38FastPath>,
}

impl Qwen38MtpHead {
    pub(crate) fn load_native(
        tensors: &IndexedSafetensors,
        native: &Qwen38NativeConfig,
        inventory: &Qwen38MtpInventory,
        device: &Device,
        target: ProjectionMaterialization,
    ) -> Result<Self> {
        let performance = crate::performance::PerformanceConfig::default().resolve_env()?;
        Self::load_native_with_performance(
            tensors,
            native,
            inventory,
            device,
            target,
            &performance.cuda,
        )
    }

    pub(crate) fn load_native_with_performance(
        tensors: &IndexedSafetensors,
        native: &Qwen38NativeConfig,
        inventory: &Qwen38MtpInventory,
        device: &Device,
        target: ProjectionMaterialization,
        performance: &crate::performance::CudaPerformanceConfig,
    ) -> Result<Self> {
        let mut config = crate::performance::PerformanceConfig::default();
        config.cuda = performance.clone();
        let policy = Qwen38ResidencyPolicy::resolve(device, target, &config, tensors, native);
        Self::load_native_with_policy(
            tensors,
            native,
            inventory,
            device,
            target,
            performance,
            &policy,
        )
    }

    /// Load under the residency `policy` the text trunk was loaded under: the
    /// head's projections take the same forms and checks.
    pub(crate) fn load_native_with_policy(
        tensors: &IndexedSafetensors,
        native: &Qwen38NativeConfig,
        inventory: &Qwen38MtpInventory,
        device: &Device,
        target: ProjectionMaterialization,
        performance: &crate::performance::CudaPerformanceConfig,
        policy: &Qwen38ResidencyPolicy,
    ) -> Result<Self> {
        if native.mtp.num_hidden_layers != 1 || native.mtp.use_dedicated_embeddings {
            return Err(Error::ModelLoadError(format!(
                "Unsupported Qwen3.8 MTP topology: layers={}, dedicated_embeddings={}",
                native.mtp.num_hidden_layers, native.mtp.use_dedicated_embeddings
            )));
        }
        if inventory.tensor_count() != QWEN38_MTP_TENSOR_COUNT {
            return Err(Error::ModelLoadError(format!(
                "Qwen3.8 MTP inventory has {} tensors, expected {QWEN38_MTP_TENSOR_COUNT}",
                inventory.tensor_count()
            )));
        }
        let cfg = &native.text;
        let hidden = cfg.embedding_length;
        let fused = hidden.checked_mul(2).ok_or_else(|| {
            Error::ModelLoadError("Qwen3.8 MTP fused input width overflow".into())
        })?;
        let block = native.block_fp8.block_shape;
        let prefix = "mtp.layers.0";
        let eps = cfg.attention_layer_norm_rms_epsilon;
        let model_layer = u32::try_from(cfg.block_count)
            .map_err(|_| Error::ModelLoadError("Qwen3.8 MTP layer id exceeds u32".into()))?;
        let loader = Qwen38WeightLoader::new(tensors, policy, device, block, target);
        let mut head = Self {
            device: device.clone(),
            projection_representation: policy.representation(),
            hidden_size: hidden,
            model_layer,
            pre_fc_norm_embedding: load_native_zero_centered_norm(
                tensors,
                "mtp.pre_fc_norm_embedding.weight",
                hidden,
                eps,
                device,
            )?,
            pre_fc_norm_hidden: load_native_zero_centered_norm(
                tensors,
                "mtp.pre_fc_norm_hidden.weight",
                hidden,
                eps,
                device,
            )?,
            fc: load_native_projection(&loader, "mtp.fc.weight", [hidden, fused])?,
            input_layernorm: load_native_zero_centered_norm(
                tensors,
                &format!("{prefix}.input_layernorm.weight"),
                hidden,
                eps,
                device,
            )?,
            attention: Qwen38FullAttention::load_native(&loader, prefix, cfg)?,
            post_attention_layernorm: load_native_zero_centered_norm(
                tensors,
                &format!("{prefix}.post_attention_layernorm.weight"),
                hidden,
                eps,
                device,
            )?,
            mlp: Qwen38Mlp::load_native(&loader, prefix, cfg, performance)?,
            norm: load_native_zero_centered_norm(tensors, "mtp.norm.weight", hidden, eps, device)?,
            activation_dtype: match target {
                ProjectionMaterialization::F32 => DType::F32,
                ProjectionMaterialization::F16 => DType::F16,
                ProjectionMaterialization::BF16 => DType::BF16,
            },
            fused_norm_path: Qwen38FastPath::default(),
            fp8_packing: Vec::new(),
            fp8_gemv_checks: Vec::new(),
            fp8_prefill_gemm_checks: Vec::new(),
        };
        let outcomes = loader.into_outcomes();
        (
            head.fp8_packing,
            head.fp8_gemv_checks,
            head.fp8_prefill_gemm_checks,
        ) = (outcomes.packing, outcomes.gemv, outcomes.prefill_gemm);
        head.resolve_fast_paths(&Qwen38CudaSwitches::for_device(device, performance), false);
        Ok(head)
    }

    pub(crate) fn graph_diagnostics(&self) -> serde_json::Value {
        self.mlp.graph_diagnostics()
    }

    /// Resolve the head's fast paths on its own weights, like
    /// [`Qwen38TextModel::resolve_fast_paths`] does for the trunk.
    pub(crate) fn resolve_fast_paths(&mut self, switches: &Qwen38CudaSwitches, allow_cpu: bool) {
        let dtype = self.activation_dtype;
        let path = resolve_norm_path(switches, &self.device, allow_cpu, dtype, || {
            norm_self_check(&self.input_layernorm, &self.device, dtype)?;
            norm_self_check(self.attention.q_norm(), &self.device, dtype)
        });
        let fused = path.is_fused();
        for norm in [
            &mut self.pre_fc_norm_embedding,
            &mut self.pre_fc_norm_hidden,
            &mut self.input_layernorm,
            &mut self.post_attention_layernorm,
            &mut self.norm,
        ] {
            norm.set_fused(fused);
        }
        self.attention.set_fused_norms(fused);
        self.fused_norm_path = path;
        self.attention
            .resolve_fused_attention(switches, &self.device, dtype, allow_cpu);
    }

    /// Resolve against the portable CPU references (tests only).
    #[cfg(test)]
    pub(crate) fn enable_fast_paths_for_tests(&mut self) {
        self.resolve_fast_paths(&Qwen38CudaSwitches::default(), true);
    }

    /// Fast-path resolutions for the admin diagnostics.
    pub(crate) fn fast_path_diagnostics(&self) -> serde_json::Value {
        let [qk_rope, kv_write, attn_gate] = self.attention.fast_paths();
        serde_json::json!({
            "rms_norm": summarize([&self.fused_norm_path]),
            "qk_rope": summarize([qk_rope]),
            "kv_write": summarize([kv_write]),
            "attn_gate": summarize([attn_gate]),
        })
    }

    /// Packing outcome of every block-FP8 projection group of the head.
    pub(crate) fn fp8_packing(&self) -> &[Qwen38FastPath] {
        &self.fp8_packing
    }

    /// Metal: the GEMV check outcome of every FP8 projection kind.
    pub(crate) fn fp8_gemv_checks(&self) -> &[Qwen38FastPath] {
        &self.fp8_gemv_checks
    }

    /// CUDA: the dequantized prefill GEMM of every FP8 projection kind.
    pub(crate) fn fp8_prefill_gemm_checks(&self) -> &[Qwen38FastPath] {
        &self.fp8_prefill_gemm_checks
    }

    pub(crate) fn projection_representation(&self) -> Qwen38ProjectionRepresentation {
        self.projection_representation
    }

    pub(crate) fn model_layer(&self) -> u32 {
        self.model_layer
    }

    /// Execute one prompt or continuation pair span and commit its MTP KV rows.
    /// The returned tensor is post-`mtp.norm` and must be projected directly by
    /// the target model's shared LM head, without the target output norm.
    pub(crate) fn forward_pairs(
        &self,
        pairs: &Qwen38MtpPairBatch,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        let (_, token_count, hidden) = pairs.embeddings.dims3()?;
        if hidden != self.hidden_size || pairs.hidden_states.dims3()? != (1, token_count, hidden) {
            return Err(Error::InvalidInput(format!(
                "Qwen3.8 MTP pair width {hidden} does not match loaded width {}",
                self.hidden_size
            )));
        }
        let (kv_heads, key_head_dim, value_head_dim) = self.attention.cache_geometry();
        cache.validate_sparse_model_layers(&[(
            self.model_layer,
            kv_heads,
            key_head_dim,
            value_head_dim,
        )])?;

        let embedding = self.pre_fc_norm_embedding.forward(&pairs.embeddings)?;
        let predecessor = self.pre_fc_norm_hidden.forward(&pairs.hidden_states)?;
        let fused = Tensor::cat(&[&embedding, &predecessor], 2)?;
        let hidden_states = self.fc.forward(&fused)?;
        let rope = Qwen38RopeInputs::new(&pairs.position_ids, &self.device, hidden_states.dtype())
            .with_decode_step(cache, std::iter::once(&self.attention))?;
        let mut prepared = cache.prepare_append(cache.context_len(), token_count)?;
        let result = (|| -> Result<Tensor> {
            let normalized = self.input_layernorm.forward(&hidden_states)?;
            let attended =
                self.attention
                    .forward_physical(&normalized, &rope, cache, &mut prepared, 0)?;
            let (residual, normalized) = self
                .post_attention_layernorm
                .add_forward(&hidden_states, &attended)?;
            let mlp = self.mlp.forward(&normalized)?;
            Ok(self.norm.add_forward(&residual, &mlp)?.1)
        })();
        match result {
            Ok(hidden_states) => {
                cache.commit_prepared(prepared)?;
                Ok(hidden_states)
            }
            Err(error) => match cache.abort_prepared(prepared) {
                Ok(()) => Err(error),
                Err(abort) => Err(Error::InferenceError(format!(
                    "Qwen3.8 MTP forward failed: {error}; provisional cache abort also failed: {abort}"
                ))),
            },
        }
    }

    pub(crate) fn forward_step(
        &self,
        token_embedding: Tensor,
        predecessor_hidden: Tensor,
        position_id: [usize; 3],
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Tensor> {
        let pairs = Qwen38MtpPairBatch::single(token_embedding, predecessor_hidden, position_id)?;
        self.forward_pairs(&pairs, cache)
    }

    /// Advance one MTP pair for each independently retained decode row while
    /// sharing the projection, attention, and MLP tensor dimensions.
    pub(crate) fn forward_steps_batch(
        &self,
        token_embeddings: &Tensor,
        predecessor_hidden: &Tensor,
        position_ids: &[[usize; 3]],
        caches: &mut [&mut PhysicalPagedKvCache],
    ) -> Result<Tensor> {
        let (batch_size, token_count, hidden) = token_embeddings.dims3().map_err(|_| {
            Error::InvalidInput(
                "Qwen3.8 MTP batch embeddings must have shape [batch,1,hidden]".into(),
            )
        })?;
        if batch_size == 0
            || token_count != 1
            || hidden != self.hidden_size
            || predecessor_hidden.dims3()? != (batch_size, 1, hidden)
            || position_ids.len() != batch_size
            || caches.len() != batch_size
        {
            return Err(Error::InvalidInput(
                "Qwen3.8 MTP decode batch rows do not match".into(),
            ));
        }
        let (kv_heads, key_head_dim, value_head_dim) = self.attention.cache_geometry();
        for cache in caches.iter() {
            cache.validate_sparse_model_layers(&[(
                self.model_layer,
                kv_heads,
                key_head_dim,
                value_head_dim,
            )])?;
        }
        let start_positions = caches
            .iter()
            .map(|cache| cache.context_len())
            .collect::<Vec<_>>();
        let first = &*caches[0];
        let slots = caches
            .iter()
            .enumerate()
            .map(|(row, cache)| {
                cache
                    .slots_for_append(start_positions[row], 1)
                    .map(|slots| slots[0])
            })
            .collect::<Result<Vec<_>>>()?;
        let lowered = first.arena().lower_slots(&slots)?;
        if lowered.arena_id() != first.arena().id() || lowered.len() != batch_size {
            return Err(Error::InvalidInput(
                "Qwen3.8 MTP batch produced an incompatible slot map".into(),
            ));
        }
        let metadata = KvDecodeBatchMetadata {
            sequences: caches
                .iter()
                .enumerate()
                .map(|(row, cache)| cache.sequence_table(start_positions[row] + 1))
                .collect::<Result<Vec<_>>>()?,
        };
        let mut completions =
            KvWriteCompletionCollector::new(first.arena().config(), lowered.logical_slots())?;

        let execution = (|| -> Result<Tensor> {
            let embedding = self.pre_fc_norm_embedding.forward(token_embeddings)?;
            let predecessor = self.pre_fc_norm_hidden.forward(predecessor_hidden)?;
            let fused = Tensor::cat(&[&embedding, &predecessor], 2)?;
            let hidden_states = self.fc.forward(&fused)?;
            let normalized = self.input_layernorm.forward(&hidden_states)?;
            let cache_refs = caches.iter().map(|cache| &**cache).collect::<Vec<_>>();
            let attended = self.attention.forward_physical_decode_batch(
                &normalized,
                position_ids,
                &cache_refs,
                lowered.as_ref(),
                &metadata,
                &mut completions,
                0,
            )?;
            let (residual, normalized) = self
                .post_attention_layernorm
                .add_forward(&hidden_states, &attended)?;
            let mlp = self.mlp.forward(&normalized)?;
            Ok(self.norm.add_forward(&residual, &mlp)?.1)
        })();
        let hidden_states = match execution {
            Ok(hidden) => hidden,
            Err(error) => {
                return match completions.drain() {
                    Ok(()) => Err(error),
                    Err(drain) => Err(Error::InferenceError(format!(
                        "Qwen3.8 MTP batch failed: {error}; write-fence drain also failed: {drain}"
                    ))),
                }
            }
        };
        let completion = Arc::new(completions.seal()?);
        for (row, cache) in caches.iter_mut().enumerate() {
            cache.commit_shared_completion(start_positions[row], 1, completion.clone())?;
        }
        Ok(hidden_states)
    }

    /// Reuse this one physical head for 1..=3 predictions.
    ///
    /// `first_lm_head_hidden` is normally the final row returned by the shifted
    /// prompt prefill. The selector projects/samples that row. Each non-final
    /// sampled token is embedded and appended with the preceding MTP hidden to
    /// produce the next prediction row.
    /// A selector break discards the whole partial proposal; errors remain
    /// fatal. The caller restores the physical cache checkpoint in either case.
    pub(crate) fn draft_recurrently<S, E>(
        &self,
        first_lm_head_hidden: &Tensor,
        depth: Qwen38MtpDepth,
        continuation_positions: &[[usize; 3]],
        cache: &mut PhysicalPagedKvCache,
        mut select: S,
        mut embed: E,
    ) -> Result<ControlFlow<Error, Qwen38MtpDraftSequence>>
    where
        S: FnMut(usize, &Tensor) -> Result<ControlFlow<Error, u32>>,
        E: FnMut(u32) -> Result<Tensor>,
    {
        if first_lm_head_hidden.dims3()? != (1, 1, self.hidden_size) {
            return Err(Error::InvalidInput(format!(
                "Qwen3.8 MTP recurrent seed must have shape [1, 1, {}]",
                self.hidden_size
            )));
        }
        let continuation_count = depth.get() - 1;
        if continuation_positions.len() != continuation_count {
            return Err(Error::InvalidInput(format!(
                "Qwen3.8 MTP depth {} requires {continuation_count} continuation positions, got {}",
                depth.get(),
                continuation_positions.len()
            )));
        }
        let mut token_ids = Vec::with_capacity(depth.get());
        let mut lm_head_hidden = Vec::with_capacity(depth.get());
        let mut current = first_lm_head_hidden.clone();
        for step in 0..depth.get() {
            let token = match select(step, &current)? {
                ControlFlow::Continue(token) => token,
                ControlFlow::Break(reason) => return Ok(ControlFlow::Break(reason)),
            };
            token_ids.push(token);
            lm_head_hidden.push(current.clone());
            if step < continuation_count {
                current =
                    self.forward_step(embed(token)?, current, continuation_positions[step], cache)?;
            }
        }
        Ok(ControlFlow::Continue(Qwen38MtpDraftSequence {
            token_ids,
            lm_head_hidden,
        }))
    }

    /// Convenience recurrence using the target model's shared embedding and
    /// raw LM head. `select_logits` receives `[1, 1, vocab]` draft logits.
    pub(crate) fn draft_recurrently_with_text<S>(
        &self,
        text: &Qwen38TextModel,
        first_lm_head_hidden: &Tensor,
        depth: Qwen38MtpDepth,
        continuation_positions: &[[usize; 3]],
        cache: &mut PhysicalPagedKvCache,
        mut select_logits: S,
    ) -> Result<ControlFlow<Error, Qwen38MtpDraftSequence>>
    where
        S: FnMut(usize, &Tensor) -> Result<ControlFlow<Error, u32>>,
    {
        self.draft_recurrently(
            first_lm_head_hidden,
            depth,
            continuation_positions,
            cache,
            |step, hidden| {
                let logits = text.project_with_shared_lm_head(hidden)?;
                select_logits(step, &logits)
            },
            |token| text.embed_token_ids(&[token]),
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use candle_core::{DType, Device, Tensor};
    use half::bf16;
    use safetensors::tensor::TensorView;
    use safetensors::Dtype as SafeDType;
    use serde_json::json;

    use super::*;
    use crate::backends::kv::{CpuKvArena, KvArenaConfig, KvLayerConfig};
    use crate::backends::BackendKind;
    use crate::engine::ModelInstanceId;
    use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};
    use crate::models::architectures::qwen38::chat::Qwen38TextConfig;
    use crate::models::architectures::qwen38::native::{
        BlockFp8Config, Qwen38LayerType, Qwen38MtpConfig,
    };

    pub(crate) struct TestDir(PathBuf);

    impl TestDir {
        pub(crate) fn new(label: &str) -> Self {
            // The clock alone is not unique: macOS reports microseconds, so
            // two tests starting together would share (and delete) a fixture.
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "izwi-qwen38-mtp-{label}-{}-{nonce}-{sequence}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        pub(crate) fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    pub(crate) struct FixtureTensor {
        name: String,
        dtype: SafeDType,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    pub(crate) fn tiny_config() -> Qwen38NativeConfig {
        Qwen38NativeConfig {
            text: Qwen38TextConfig {
                architecture: "qwen3_5".into(),
                block_count: 1,
                context_length: 32,
                embedding_length: 4,
                feed_forward_length: 4,
                attention_head_count: 2,
                attention_head_count_kv: 1,
                attention_key_length: 2,
                attention_value_length: 2,
                rope_dimension_sections: vec![1, 0, 0],
                rope_dimension_count: 2,
                rope_freq_base: 10_000.0,
                attention_layer_norm_rms_epsilon: 1e-6,
                ssm_conv_kernel: 1,
                ssm_state_size: 1,
                ssm_group_count: 1,
                ssm_time_step_rank: 1,
                ssm_inner_size: 1,
                full_attention_interval: 1,
            },
            layer_types: vec![Qwen38LayerType::FullAttention],
            vocab_size: 8,
            attn_output_gate: true,
            partial_rotary_factor: 0.5,
            mrope_interleaved: true,
            tie_word_embeddings: false,
            block_fp8: BlockFp8Config {
                block_shape: [2, 2],
            },
            mtp: Qwen38MtpConfig {
                num_hidden_layers: 1,
                use_dedicated_embeddings: false,
            },
        }
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    fn dense(name: impl Into<String>, shape: &[usize], values: Vec<f32>) -> FixtureTensor {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        FixtureTensor {
            name: name.into(),
            dtype: SafeDType::BF16,
            shape: shape.to_vec(),
            data: bf16_bytes(&values),
        }
    }

    fn zero_norm(name: impl Into<String>, width: usize) -> FixtureTensor {
        dense(name, &[width], vec![0.0; width])
    }

    fn push_zero_fp8_projection(
        tensors: &mut Vec<FixtureTensor>,
        name: impl Into<String>,
        shape: [usize; 2],
        block: [usize; 2],
    ) {
        let name = name.into();
        tensors.push(FixtureTensor {
            name: name.clone(),
            dtype: SafeDType::F8_E4M3,
            shape: shape.to_vec(),
            data: vec![0; shape[0] * shape[1]],
        });
        let scale_shape = [shape[0].div_ceil(block[0]), shape[1].div_ceil(block[1])];
        tensors.push(dense(
            name.replace(".weight", ".weight_scale_inv"),
            &scale_shape,
            vec![1.0; scale_shape[0] * scale_shape[1]],
        ));
    }

    /// Deterministic pseudo-random values in [-1, 1).
    fn fixture_uniform(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    /// Roughly Gaussian values with standard deviation `std`.
    fn fixture_normal(seed: u64, len: usize, std: f32) -> Vec<f32> {
        let parts: Vec<Vec<f32>> = (0..4)
            .map(|part| fixture_uniform(seed * 4 + part, len))
            .collect();
        (0..len)
            .map(|i| parts.iter().map(|part| part[i]).sum::<f32>() * std * 0.866)
            .collect()
    }

    /// A block-FP8 projection with mixed bytes and a distinct scale in every
    /// block on both axes: Gaussian weights, scaled per block, quantized to
    /// the checkpoint layout (E4M3FN bytes, BF16 `weight_scale_inv`).
    pub(crate) fn push_random_fp8_projection(
        tensors: &mut Vec<FixtureTensor>,
        name: impl Into<String>,
        shape: [usize; 2],
        block: [usize; 2],
        seed: u64,
    ) {
        let name = name.into();
        let [n, k] = shape;
        let std = 1.0 / (k as f32).sqrt();
        let column_blocks = k.div_ceil(block[1]);
        let values: Vec<f32> = fixture_normal(seed, n * k, std)
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                let index = (i / k / block[0]) * column_blocks + (i % k) / block[1];
                v * (0.6 + 0.13 * (index % 7) as f32)
            })
            .collect();
        assert_eq!(
            block,
            [128, 128],
            "the checkpoint quantizer uses 128x128 blocks"
        );
        let (bytes, scales) =
            crate::kernels::cuda::fp8::quantize_block_fp8(&values, n, k, |v| v).unwrap();
        tensors.push(FixtureTensor {
            name: name.clone(),
            dtype: SafeDType::F8_E4M3,
            shape: shape.to_vec(),
            data: bytes,
        });
        tensors.push(dense(
            name.replace(".weight", ".weight_scale_inv"),
            &[n.div_ceil(block[0]), column_blocks],
            scales,
        ));
    }

    /// A dense BF16 fixture tensor of Gaussian values (tests of other modules).
    pub(crate) fn fixture_dense_for_tests(
        name: &str,
        shape: [usize; 2],
        std: f32,
    ) -> FixtureTensor {
        dense(name, &shape, fixture_normal(77, shape[0] * shape[1], std))
    }

    pub(crate) fn write_fixture(dir: &Path, tensors: &[FixtureTensor], shard: &str) {
        let views = tensors
            .iter()
            .map(|tensor| {
                (
                    tensor.name.clone(),
                    TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.data).unwrap(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        safetensors::serialize_to_file(&views, &None, &dir.join(shard)).unwrap();
        let weight_map = tensors
            .iter()
            .map(|tensor| (tensor.name.clone(), json!(shard)))
            .collect::<serde_json::Map<_, _>>();
        fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({ "weight_map": weight_map })).unwrap(),
        )
        .unwrap();
    }

    /// `config.json` of the FP8-geometry fixture: Qwen3.8's structure at
    /// sizes the 128x128 block-FP8 kernels accept (hidden 256; three
    /// DeltaNet layers with 2 key / 6 value heads of 128 and one attention
    /// layer with 2 query heads of 256 and one KV head; FFN 512; vocab 512;
    /// interleaved [11, 11, 10] M-RoPE over 64 rotary dims; one MTP layer).
    pub(crate) fn fp8_geometry_config_json() -> Vec<u8> {
        let rope = json!({
            "mrope_interleaved": true,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000.0,
            "rope_type": "default"
        });
        let mut text = json!({
                "attention_bias": false,
                "attention_dropout": 0.0,
                "attn_output_gate": true,
                "bos_token_id": 1,
                "dtype": "bfloat16",
                "eos_token_id": 2,
                "full_attention_interval": 4,
                "head_dim": 256,
                "hidden_act": "silu",
                "hidden_size": 256,
                "intermediate_size": 512,
                "layer_types": [
                    "linear_attention", "linear_attention", "linear_attention", "full_attention"
                ],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 6,
                "linear_value_head_dim": 128,
                "mamba_ssm_dtype": "float32",
                "max_position_embeddings": 512,
                "model_type": "qwen3_5_text",
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "num_attention_heads": 2,
                "num_hidden_layers": 4,
                "num_key_value_heads": 1,
                "output_gate_type": "swish",
                "partial_rotary_factor": 0.25,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": false,
                "use_cache": true,
                "vocab_size": 512
        });
        text["rope_parameters"] = rope;
        json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "language_model_only": false,
            "model_type": "qwen3_5",
            "tie_word_embeddings": false,
            "quantization_config": {
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128]
            },
            "text_config": text
        })
        .to_string()
        .into_bytes()
    }

    /// Write the FP8-geometry fixture (config, one shard, index) to `dir`:
    /// every projection a random block-FP8 matrix with distinct block scales
    /// ([`push_random_fp8_projection`]), random zero-centered norm gains and
    /// DeltaNet parameters, and the 22-tensor MTP manifest. Open it with
    /// `Qwen38NativeCheckpoint::open_synthetic` (or with the synthetic
    /// geometry opt-in for production loading).
    pub(crate) fn write_fp8_geometry_checkpoint(dir: &Path) -> Qwen38NativeConfig {
        let raw = fp8_geometry_config_json();
        fs::write(dir.join("config.json"), &raw).unwrap();
        let config = crate::models::architectures::qwen38::native::Qwen38NativeConfig::from_json_synthetic_for_tests(&raw);
        let text = &config.text;
        let (hidden, ff, vocab) = (
            text.embedding_length,
            text.feed_forward_length,
            config.vocab_size,
        );
        let block = config.block_fp8.block_shape;
        let mut seed = 1000u64;
        let mut next = || {
            seed += 1;
            seed
        };
        let norm = |name: String, width: usize, seed: u64| {
            dense(
                name,
                &[width],
                fixture_uniform(seed, width)
                    .iter()
                    .map(|v| v * 0.1)
                    .collect(),
            )
        };
        let mut tensors = vec![
            dense(
                "model.language_model.embed_tokens.weight",
                &[vocab, hidden],
                fixture_normal(next(), vocab * hidden, 1.0),
            ),
            dense(
                "lm_head.weight",
                &[vocab, hidden],
                fixture_normal(next(), vocab * hidden, 0.25),
            ),
            norm("model.language_model.norm.weight".into(), hidden, next()),
        ];
        let conv_dim = text.ssm_state_size * text.ssm_group_count * 2 + text.ssm_inner_size;
        let heads = text.ssm_time_step_rank;
        let q_width = text.attention_head_count * text.attention_key_length * 2;
        let kv_width = text.attention_head_count_kv * text.attention_key_length;
        let attention =
            |tensors: &mut Vec<FixtureTensor>, prefix: &str, next: &mut dyn FnMut() -> u64| {
                let head = text.attention_key_length;
                tensors.push(norm(
                    format!("{prefix}.self_attn.q_norm.weight"),
                    head,
                    next(),
                ));
                tensors.push(norm(
                    format!("{prefix}.self_attn.k_norm.weight"),
                    head,
                    next(),
                ));
                for (name, shape) in [
                    ("q_proj", [q_width, hidden]),
                    ("k_proj", [kv_width, hidden]),
                    ("v_proj", [kv_width, hidden]),
                    (
                        "o_proj",
                        [
                            hidden,
                            text.attention_head_count * text.attention_value_length,
                        ],
                    ),
                ] {
                    push_random_fp8_projection(
                        tensors,
                        format!("{prefix}.self_attn.{name}.weight"),
                        shape,
                        block,
                        next(),
                    );
                }
            };
        let mlp =
            |tensors: &mut Vec<FixtureTensor>, prefix: &str, next: &mut dyn FnMut() -> u64| {
                tensors.push(norm(
                    format!("{prefix}.input_layernorm.weight"),
                    hidden,
                    next(),
                ));
                tensors.push(norm(
                    format!("{prefix}.post_attention_layernorm.weight"),
                    hidden,
                    next(),
                ));
                for (name, shape) in [
                    ("gate_proj", [ff, hidden]),
                    ("up_proj", [ff, hidden]),
                    ("down_proj", [hidden, ff]),
                ] {
                    push_random_fp8_projection(
                        tensors,
                        format!("{prefix}.mlp.{name}.weight"),
                        shape,
                        block,
                        next(),
                    );
                }
            };
        for (index, kind) in config.layer_types.iter().enumerate() {
            let prefix = format!("model.language_model.layers.{index}");
            mlp(&mut tensors, &prefix, &mut next);
            match kind {
                Qwen38LayerType::FullAttention => attention(&mut tensors, &prefix, &mut next),
                Qwen38LayerType::LinearAttention => {
                    let linear = format!("{prefix}.linear_attn");
                    tensors.push(dense(
                        format!("{linear}.A_log"),
                        &[heads],
                        fixture_uniform(next(), heads),
                    ));
                    tensors.push(dense(
                        format!("{linear}.dt_bias"),
                        &[heads],
                        fixture_uniform(next(), heads),
                    ));
                    tensors.push(dense(
                        format!("{linear}.conv1d.weight"),
                        &[conv_dim, 1, text.ssm_conv_kernel],
                        fixture_normal(next(), conv_dim * text.ssm_conv_kernel, 0.4),
                    ));
                    for name in ["in_proj_a", "in_proj_b"] {
                        tensors.push(dense(
                            format!("{linear}.{name}.weight"),
                            &[heads, hidden],
                            fixture_normal(next(), heads * hidden, 1.0 / (hidden as f32).sqrt()),
                        ));
                    }
                    let head_v = text.ssm_inner_size / heads;
                    tensors.push(dense(
                        format!("{linear}.norm.weight"),
                        &[head_v],
                        fixture_uniform(next(), head_v)
                            .iter()
                            .map(|v| 1.0 + 0.1 * v)
                            .collect(),
                    ));
                    for (name, shape) in [
                        ("in_proj_qkv", [conv_dim, hidden]),
                        ("in_proj_z", [text.ssm_inner_size, hidden]),
                        ("out_proj", [hidden, text.ssm_inner_size]),
                    ] {
                        push_random_fp8_projection(
                            &mut tensors,
                            format!("{linear}.{name}.weight"),
                            shape,
                            block,
                            next(),
                        );
                    }
                }
            }
        }
        let prefix = "mtp.layers.0";
        mlp(&mut tensors, prefix, &mut next);
        attention(&mut tensors, prefix, &mut next);
        tensors.push(dense(
            "mtp.fc.weight",
            &[hidden, hidden * 2],
            fixture_normal(
                next(),
                hidden * hidden * 2,
                1.0 / (2.0 * hidden as f32).sqrt(),
            ),
        ));
        for name in [
            "mtp.norm.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
        ] {
            tensors.push(norm(name.into(), hidden, next()));
        }
        write_fixture(dir, &tensors, "model.safetensors");
        config
    }

    pub(crate) fn write_tiny_checkpoint(dir: &Path, config: &Qwen38NativeConfig) {
        let hidden = config.text.embedding_length;
        let ff = config.text.feed_forward_length;
        let q_width = config.text.attention_head_count * config.text.attention_key_length * 2;
        let kv_width = config.text.attention_head_count_kv * config.text.attention_key_length;
        let output_width = config.text.attention_head_count * config.text.attention_value_length;
        let block = config.block_fp8.block_shape;
        let prefix = "mtp.layers.0";
        let mut fc = vec![0.0; hidden * hidden * 2];
        for row in 0..hidden {
            fc[row * hidden * 2 + row] = 1.0;
        }
        let mut tensors = vec![
            dense("mtp.fc.weight", &[hidden, hidden * 2], fc),
            zero_norm("mtp.pre_fc_norm_embedding.weight", hidden),
            zero_norm("mtp.pre_fc_norm_hidden.weight", hidden),
            zero_norm("mtp.norm.weight", hidden),
            zero_norm(format!("{prefix}.input_layernorm.weight"), hidden),
            zero_norm(format!("{prefix}.post_attention_layernorm.weight"), hidden),
            zero_norm(
                format!("{prefix}.self_attn.q_norm.weight"),
                config.text.attention_key_length,
            ),
            zero_norm(
                format!("{prefix}.self_attn.k_norm.weight"),
                config.text.attention_key_length,
            ),
        ];
        for (name, shape) in [
            (format!("{prefix}.mlp.gate_proj.weight"), [ff, hidden]),
            (format!("{prefix}.mlp.up_proj.weight"), [ff, hidden]),
            (format!("{prefix}.mlp.down_proj.weight"), [hidden, ff]),
            (
                format!("{prefix}.self_attn.q_proj.weight"),
                [q_width, hidden],
            ),
            (
                format!("{prefix}.self_attn.k_proj.weight"),
                [kv_width, hidden],
            ),
            (
                format!("{prefix}.self_attn.v_proj.weight"),
                [kv_width, hidden],
            ),
            (
                format!("{prefix}.self_attn.o_proj.weight"),
                [hidden, output_width],
            ),
        ] {
            push_zero_fp8_projection(&mut tensors, name, shape, block);
        }
        assert_eq!(tensors.len(), QWEN38_MTP_TENSOR_COUNT);

        let views = tensors
            .iter()
            .map(|tensor| {
                (
                    tensor.name.clone(),
                    TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.data).unwrap(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        safetensors::serialize_to_file(&views, &None, &dir.join("mtp.safetensors")).unwrap();
        let weight_map = tensors
            .iter()
            .map(|tensor| (tensor.name.clone(), json!("mtp.safetensors")))
            .collect::<serde_json::Map<_, _>>();
        fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({ "weight_map": weight_map })).unwrap(),
        )
        .unwrap();
    }

    pub(crate) fn tiny_cache(config: &Qwen38NativeConfig) -> PhysicalPagedKvCache {
        let arena_id = KvArenaId {
            model_instance: ModelInstanceId::new(81),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let binding = KvLayerBinding {
            model_layer: config.text.block_count as u32,
            physical_layer: 0,
        };
        let arena = Arc::new(
            CpuKvArena::new(KvArenaConfig {
                id: arena_id,
                group,
                page_tokens: 4,
                capacity_pages: 2,
                growth: None,
                dtype: DType::F32,
                layers: vec![KvLayerConfig {
                    binding,
                    num_kv_heads: config.text.attention_head_count_kv as u32,
                    key_head_dim: config.text.attention_key_length as u32,
                    value_head_dim: config.text.attention_value_length as u32,
                }],
            })
            .unwrap(),
        );
        let blocks = (0..2)
            .map(|index| CacheBlockRef {
                arena: arena_id,
                group,
                index,
                slot_generation: 1,
            })
            .collect();
        PhysicalPagedKvCache::new(arena, vec![binding], blocks, 0).unwrap()
    }

    pub(crate) fn tiny_shared_caches(
        config: &Qwen38NativeConfig,
        rows: usize,
    ) -> Vec<PhysicalPagedKvCache> {
        let arena_id = KvArenaId {
            model_instance: ModelInstanceId::new(82),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation: 1,
        };
        let group = KvGroupId::new(1);
        let binding = KvLayerBinding {
            model_layer: config.text.block_count as u32,
            physical_layer: 0,
        };
        let arena = Arc::new(
            CpuKvArena::new(KvArenaConfig {
                id: arena_id,
                group,
                page_tokens: 4,
                capacity_pages: (rows * 2) as u32,
                growth: None,
                dtype: DType::F32,
                layers: vec![KvLayerConfig {
                    binding,
                    num_kv_heads: config.text.attention_head_count_kv as u32,
                    key_head_dim: config.text.attention_key_length as u32,
                    value_head_dim: config.text.attention_value_length as u32,
                }],
            })
            .unwrap(),
        );
        (0..rows)
            .map(|row| {
                let blocks = (row * 2..row * 2 + 2)
                    .map(|index| CacheBlockRef {
                        arena: arena_id,
                        group,
                        index: index as u32,
                        slot_generation: 1,
                    })
                    .collect();
                PhysicalPagedKvCache::new(arena.clone(), vec![binding], blocks, 0).unwrap()
            })
            .collect()
    }

    fn assert_close(actual: &Tensor, expected: &Tensor) {
        let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "value {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn shifted_prompt_is_prompt_length_and_uses_next_token_as_final_embedding() {
        let prompt = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            (1, 2, 4),
            &Device::Cpu,
        )
        .unwrap();
        let hidden = Tensor::from_vec(
            vec![10.0f32, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0],
            (1, 2, 4),
            &Device::Cpu,
        )
        .unwrap();
        let next = Tensor::from_vec(vec![0.0f32, 0.0, 3.0, 0.0], (1, 1, 4), &Device::Cpu).unwrap();
        let positions = [[7, 7, 7], [8, 8, 8]];

        let pairs = Qwen38MtpPairBatch::shifted_prompt(&prompt, &hidden, &next, &positions)
            .expect("shifted prompt");

        assert_eq!(pairs.token_count(), 2);
        assert_eq!(pairs.position_ids(), positions);
        assert_eq!(
            pairs.embeddings().to_vec3::<f32>().unwrap(),
            vec![vec![vec![0.0, 2.0, 0.0, 0.0], vec![0.0, 0.0, 3.0, 0.0]]]
        );
        assert_eq!(
            pairs.hidden_states().to_vec3::<f32>().unwrap(),
            hidden.to_vec3::<f32>().unwrap()
        );
    }

    #[test]
    fn loads_all_mtp_tensors_runs_reference_math_and_reuses_one_layer_three_times() {
        let dir = TestDir::new("head");
        let config = tiny_config();
        write_tiny_checkpoint(dir.path(), &config);
        let tensors = IndexedSafetensors::open(dir.path()).unwrap();
        let inventory = tensors.validate_mtp_tensor_manifest(&config).unwrap();
        let head = Qwen38MtpHead::load_native(
            &tensors,
            &config,
            &inventory,
            &Device::Cpu,
            ProjectionMaterialization::F32,
        )
        .unwrap();
        assert_eq!(head.model_layer(), 1);
        assert_eq!(
            head.projection_representation(),
            Qwen38ProjectionRepresentation::ExpandedF32
        );

        let prompt = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            (1, 2, 4),
            &Device::Cpu,
        )
        .unwrap();
        let predecessor = Tensor::from_vec(vec![9.0f32; 8], (1, 2, 4), &Device::Cpu).unwrap();
        let next = Tensor::from_vec(vec![0.0f32, 0.0, 3.0, 0.0], (1, 1, 4), &Device::Cpu).unwrap();
        let pairs = Qwen38MtpPairBatch::shifted_prompt(
            &prompt,
            &predecessor,
            &next,
            &[[0, 0, 0], [1, 1, 1]],
        )
        .unwrap();
        let shifted = pairs.embeddings().clone();
        let unit_weight = Tensor::ones(4, DType::F32, &Device::Cpu).unwrap();
        let pre_normalized = candle_nn::ops::rms_norm(&shifted, &unit_weight, 1e-6).unwrap();
        let expected = candle_nn::ops::rms_norm(&pre_normalized, &unit_weight, 1e-6).unwrap();
        let mut cache = tiny_cache(&config);

        let output = head.forward_pairs(&pairs, &mut cache).unwrap();

        assert_eq!(cache.context_len(), 2);
        assert_close(&output, &expected);
        let seed = output.narrow(1, 1, 1).unwrap();
        let mut embed_calls = 0usize;
        let drafts = head
            .draft_recurrently(
                &seed,
                Qwen38MtpDepth::new(3).unwrap(),
                &[[2, 2, 2], [3, 3, 3]],
                &mut cache,
                |step, hidden| {
                    assert_eq!(hidden.dims3().unwrap(), (1, 1, 4));
                    Ok(ControlFlow::Continue(3 + step as u32))
                },
                |token| {
                    embed_calls += 1;
                    let mut values = vec![0.0f32; 4];
                    values[token as usize % 4] = 1.0;
                    Tensor::from_vec(values, (1, 1, 4), &Device::Cpu).map_err(Error::from)
                },
            )
            .unwrap()
            .continue_value()
            .expect("complete draft sequence");
        assert_eq!(drafts.token_ids, [3, 4, 5]);
        assert_eq!(drafts.lm_head_hidden.len(), 3);
        assert_eq!(embed_calls, 2);
        assert_eq!(cache.context_len(), 4);
    }

    #[test]
    fn partial_draft_break_discards_sequence_and_backend_error_remains_fatal() {
        let dir = TestDir::new("partial-draft-abort");
        let config = tiny_config();
        write_tiny_checkpoint(dir.path(), &config);
        let tensors = IndexedSafetensors::open(dir.path()).unwrap();
        let inventory = tensors.validate_mtp_tensor_manifest(&config).unwrap();
        let head = Qwen38MtpHead::load_native(
            &tensors,
            &config,
            &inventory,
            &Device::Cpu,
            ProjectionMaterialization::F32,
        )
        .unwrap();
        let mut cache = tiny_cache(&config);
        let input = Tensor::ones((1, 1, 4), DType::F32, &Device::Cpu).unwrap();
        let seed = head
            .forward_step(input.clone(), input.clone(), [0; 3], &mut cache)
            .unwrap();

        for backend_error in [false, true] {
            let checkpoint = cache.logical_checkpoint();
            let mut selected_steps = Vec::new();
            let mut embedded_tokens = Vec::new();
            let outcome = head.draft_recurrently(
                &seed,
                Qwen38MtpDepth::new(3).unwrap(),
                &[[1; 3], [2; 3]],
                &mut cache,
                |step, _| {
                    selected_steps.push(step);
                    if step == 0 {
                        return Ok(ControlFlow::Continue(3));
                    }
                    assert_eq!(step, 1, "selection must stop at the failed draft");
                    if backend_error {
                        Err(Error::InferenceError("backend failure".into()))
                    } else {
                        Ok(ControlFlow::Break(Error::InferenceError(
                            "non-finite draft".into(),
                        )))
                    }
                },
                |token| {
                    embedded_tokens.push(token);
                    Ok(input.clone())
                },
            );
            match (backend_error, outcome) {
                (false, Ok(ControlFlow::Break(Error::InferenceError(reason)))) => {
                    assert_eq!(reason, "non-finite draft");
                }
                (true, Err(Error::InferenceError(reason))) => {
                    assert_eq!(reason, "backend failure");
                }
                _ => panic!("partial proposal must not escape as a completed sequence"),
            }
            assert_eq!(selected_steps, [0, 1]);
            assert_eq!(embedded_tokens, [3]);
            assert_eq!(cache.context_len(), 2, "only step zero appended a KV row");
            // The caller owns rollback of the successful provisional step.
            cache.restore_logical_checkpoint(checkpoint).unwrap();
            assert_eq!(cache.context_len(), 1);
        }
    }

    #[test]
    fn batched_mtp_steps_match_independent_scalar_rows() {
        let dir = TestDir::new("batch-head");
        let config = tiny_config();
        write_tiny_checkpoint(dir.path(), &config);
        let tensors = IndexedSafetensors::open(dir.path()).unwrap();
        let inventory = tensors.validate_mtp_tensor_manifest(&config).unwrap();
        let head = Qwen38MtpHead::load_native(
            &tensors,
            &config,
            &inventory,
            &Device::Cpu,
            ProjectionMaterialization::F32,
        )
        .unwrap();
        let embeddings = Tensor::from_vec(
            vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            (2, 1, 4),
            &Device::Cpu,
        )
        .unwrap();
        let predecessors = Tensor::from_vec(
            vec![3.0f32, 1.0, 2.0, 4.0, 2.0, 5.0, 1.0, 3.0],
            (2, 1, 4),
            &Device::Cpu,
        )
        .unwrap();
        let positions = [[0, 0, 0], [0, 0, 0]];
        let mut scalar_outputs = Vec::new();
        for row in 0..2 {
            let mut cache = tiny_cache(&config);
            scalar_outputs.push(
                head.forward_step(
                    embeddings.narrow(0, row, 1).unwrap(),
                    predecessors.narrow(0, row, 1).unwrap(),
                    positions[row],
                    &mut cache,
                )
                .unwrap(),
            );
            assert_eq!(cache.context_len(), 1);
        }
        let scalar_refs = scalar_outputs.iter().collect::<Vec<_>>();
        let scalar = Tensor::cat(&scalar_refs, 0).unwrap();

        let mut caches = tiny_shared_caches(&config, 2);
        let mut cache_refs = caches.iter_mut().collect::<Vec<_>>();
        let batched = head
            .forward_steps_batch(&embeddings, &predecessors, &positions, &mut cache_refs)
            .unwrap();
        assert_close(&batched, &scalar);
        assert!(caches.iter().all(|cache| cache.context_len() == 1));
    }

    #[test]
    fn recurrent_depth_is_strictly_bounded() {
        assert!(Qwen38MtpDepth::new(0).is_err());
        assert_eq!(Qwen38MtpDepth::new(1).unwrap().get(), 1);
        assert_eq!(Qwen38MtpDepth::new(3).unwrap().get(), 3);
        assert!(Qwen38MtpDepth::new(4).is_err());
    }

    /// The head resolves its own norms on its own weights; with them fused
    /// (CPU reference), the pair and batched paths track the composition.
    #[test]
    fn mtp_norms_resolve_fused_and_track_the_composition() {
        let _guard = crate::models::architectures::qwen38::fast_path::counters_lock();
        let dir = TestDir::new("fused-norms");
        let config = tiny_config();
        write_tiny_checkpoint(dir.path(), &config);
        let tensors = IndexedSafetensors::open(dir.path()).unwrap();
        let inventory = tensors.validate_mtp_tensor_manifest(&config).unwrap();
        let load = || {
            Qwen38MtpHead::load_native(
                &tensors,
                &config,
                &inventory,
                &Device::Cpu,
                ProjectionMaterialization::F16,
            )
            .unwrap()
        };
        let legacy = load();
        let mut fused = load();
        assert_eq!(
            legacy.fast_path_diagnostics()["rms_norm"]["backend"],
            "legacy"
        );
        fused.enable_fast_paths_for_tests();
        let diagnostics = fused.fast_path_diagnostics();
        for name in ["rms_norm", "qk_rope", "kv_write", "attn_gate"] {
            assert_eq!(
                diagnostics[name]["backend"], "fused",
                "{name}: {diagnostics}"
            );
        }

        let values = |seed: f32, rows: usize| {
            Tensor::from_vec(
                (0..rows * 4)
                    .map(|i| ((i as f32 + seed) * 0.754_877_7).sin() * 2.0)
                    .collect::<Vec<_>>(),
                (rows, 4),
                &Device::Cpu,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
        };
        let close = |actual: &Tensor, expected: &Tensor| {
            let host = |t: &Tensor| {
                t.to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
            };
            crate::models::architectures::qwen38::fast_path::compare_values(
                "MTP fused norms",
                &host(actual),
                &host(expected),
                0.02,
                0.03,
            )
            .unwrap();
        };
        let pairs = Qwen38MtpPairBatch::new(
            values(1.0, 2).unsqueeze(0).unwrap(),
            values(2.0, 2).unsqueeze(0).unwrap(),
            vec![[0, 0, 0], [1, 1, 1]],
        )
        .unwrap();
        let expected = legacy
            .forward_pairs(&pairs, &mut tiny_cache(&config))
            .unwrap();
        let actual = fused
            .forward_pairs(&pairs, &mut tiny_cache(&config))
            .unwrap();
        close(&actual, &expected);

        let embeddings = values(3.0, 2).unsqueeze(1).unwrap();
        let predecessors = values(4.0, 2).unsqueeze(1).unwrap();
        let batch = |head: &Qwen38MtpHead| {
            let mut caches = tiny_shared_caches(&config, 2);
            let mut cache_refs = caches.iter_mut().collect::<Vec<_>>();
            head.forward_steps_batch(
                &embeddings,
                &predecessors,
                &[[0, 0, 0], [0, 0, 0]],
                &mut cache_refs,
            )
            .unwrap()
        };
        close(&batch(&fused), &batch(&legacy));
    }
}

#[cfg(test)]
mod adaptive_tests {
    use super::AdaptiveMtp;
    use std::time::Duration;
    #[test]
    fn starts_shallow_explores_bounded_depths_and_selects_elapsed_cost() {
        let mut policy = AdaptiveMtp::new(true, 1);
        assert_eq!(policy.depth(4), 1);
        let mut seen = [false; 4];
        for _ in 0..160 {
            let depth = policy.depth(4);
            seen[depth] = true;
            let committed = depth + 1;
            // Depth two is fastest despite depth three accepting more tokens.
            let cost = [20, 15, 8, 12][depth];
            policy.observe(
                depth,
                committed,
                Duration::from_millis(cost * committed as u64),
                4,
            );
        }
        assert_eq!(seen, [true; 4]);
        assert_eq!(policy.selected, 2);
        assert_eq!(policy.depth(1), 0);
        assert!(policy.depth(2) <= 1);
    }
    #[test]
    fn poor_speculation_selects_scalar_and_opt_out_is_fixed() {
        let mut policy = AdaptiveMtp::new(true, 1);
        for _ in 0..80 {
            let depth = policy.depth(4);
            policy.observe(
                depth,
                1,
                Duration::from_millis(if depth == 0 { 5 } else { 30 }),
                4,
            );
        }
        assert_eq!(policy.selected, 0);
        let mut fixed = AdaptiveMtp::new(false, 3);
        fixed.observe(0, 1, Duration::from_nanos(1), 4);
        assert_eq!(fixed.depth(4), 3);
        assert_eq!(fixed.depth(2), 1);
    }
    #[test]
    fn cancellation_clone_and_scheduler_limited_tails_do_not_change_policy() {
        let base = AdaptiveMtp::new(true, 1);
        let mut cancelled = base.clone();
        cancelled.observe(1, 2, Duration::from_millis(1), 4);
        assert_eq!(base.rounds, 0);
        let mut limited = base.clone();
        limited.observe(0, 1, Duration::from_millis(1), 1);
        assert_eq!(limited.rounds, 0);
    }

    #[test]
    fn numerical_disable_blocks_fixed_depth_probes_and_delayed_observations() {
        for adaptive in [false, true] {
            let mut policy = AdaptiveMtp::new(adaptive, 3);
            assert!(!policy.speculation_disabled());
            assert_eq!(policy.depth(4), 3);
            for _ in 0..8 {
                policy.observe(3, 4, Duration::from_millis(4), 4);
            }
            let before = policy.clone();
            policy.disable_after_nonfinite_draft();
            policy.disable_after_nonfinite_draft();
            assert!(policy.speculation_disabled());
            // Events queued before the failure must not train or re-enable
            // the controller, even across multiple exploration intervals.
            for _ in 0..32 {
                for depth in 0..=3 {
                    policy.observe(depth, depth + 1, Duration::from_nanos(1), 4);
                }
            }
            for budget in [0, 1, 2, 4, usize::MAX] {
                assert_eq!(policy.depth(budget), 0);
                assert!(!policy.can_train(budget));
            }
            assert_eq!(policy.samples, before.samples);
            assert_eq!(policy.cost_per_token, before.cost_per_token);
            assert_eq!(policy.rounds, before.rounds);
            assert_eq!(policy.probe, before.probe);
            assert_eq!(policy.selected, before.selected);
        }
    }

    #[test]
    fn checkpoint_restore_keeps_either_numerical_latch_and_restores_timing_policy() {
        for current_disabled in [false, true] {
            for checkpoint_disabled in [false, true] {
                let mut checkpoint = AdaptiveMtp::new(true, 1);
                checkpoint.observe(1, 2, Duration::from_millis(2), 4);
                if checkpoint_disabled {
                    checkpoint.disable_after_nonfinite_draft();
                }
                let mut current = AdaptiveMtp::new(false, 3);
                if current_disabled {
                    current.disable_after_nonfinite_draft();
                }
                current.restore_from_checkpoint(checkpoint.clone());
                let disabled = current_disabled || checkpoint_disabled;
                assert_eq!(current.speculation_disabled(), disabled);
                assert_eq!(current.enabled, checkpoint.enabled);
                assert_eq!(current.fixed_depth, checkpoint.fixed_depth);
                assert_eq!(current.selected, checkpoint.selected);
                assert_eq!(current.samples, checkpoint.samples);
                assert_eq!(current.cost_per_token, checkpoint.cost_per_token);
                assert_eq!(current.rounds, checkpoint.rounds);
                assert_eq!(current.probe, checkpoint.probe);
                if disabled {
                    current.observe(3, 4, Duration::from_nanos(1), 4);
                    assert_eq!(current.rounds, checkpoint.rounds);
                    assert_eq!(current.depth(4), 0);
                    assert!(!current.can_train(4));
                } else {
                    assert_eq!(current.depth(4), checkpoint.depth(4));
                    assert!(current.can_train(4));
                }
            }
        }
    }
}
