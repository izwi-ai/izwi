//! Recurrent multi-token-prediction head for native Qwen3.8 checkpoints.

use std::sync::Arc;

use candle_core::{Device, Module, Tensor};

use super::native::{
    IndexedSafetensors, ProjectionMaterialization, Qwen38MtpInventory, Qwen38NativeConfig,
    QWEN38_MTP_TENSOR_COUNT,
};
use super::text::{
    load_native_projection, load_native_zero_centered_norm, native_projection_representation,
    Qwen38FullAttention, Qwen38Mlp, Qwen38Projection, Qwen38ProjectionRepresentation,
    Qwen38RmsNorm, Qwen38TextModel,
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
}

impl Qwen38MtpHead {
    pub(crate) fn load_native(
        tensors: &IndexedSafetensors,
        native: &Qwen38NativeConfig,
        inventory: &Qwen38MtpInventory,
        device: &Device,
        target: ProjectionMaterialization,
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
        Ok(Self {
            device: device.clone(),
            projection_representation: native_projection_representation(device, target),
            hidden_size: hidden,
            model_layer,
            pre_fc_norm_embedding: load_native_zero_centered_norm(
                tensors,
                "mtp.pre_fc_norm_embedding.weight",
                hidden,
                eps,
                target,
                device,
            )?,
            pre_fc_norm_hidden: load_native_zero_centered_norm(
                tensors,
                "mtp.pre_fc_norm_hidden.weight",
                hidden,
                eps,
                target,
                device,
            )?,
            fc: load_native_projection(
                tensors,
                "mtp.fc.weight",
                [hidden, fused],
                block,
                target,
                device,
            )?,
            input_layernorm: load_native_zero_centered_norm(
                tensors,
                &format!("{prefix}.input_layernorm.weight"),
                hidden,
                eps,
                target,
                device,
            )?,
            attention: Qwen38FullAttention::load_native(
                tensors, device, prefix, cfg, block, target,
            )?,
            post_attention_layernorm: load_native_zero_centered_norm(
                tensors,
                &format!("{prefix}.post_attention_layernorm.weight"),
                hidden,
                eps,
                target,
                device,
            )?,
            mlp: Qwen38Mlp::load_native(tensors, device, prefix, cfg, block, target)?,
            norm: load_native_zero_centered_norm(
                tensors,
                "mtp.norm.weight",
                hidden,
                eps,
                target,
                device,
            )?,
        })
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
        let mrope_plan = self.attention.prepare_mrope_plan(
            &pairs.position_ids,
            &self.device,
            hidden_states.dtype(),
        )?;
        let mut prepared = cache.prepare_append(cache.context_len(), token_count)?;
        let result = (|| -> Result<Tensor> {
            let residual = hidden_states.clone();
            let normalized = self.input_layernorm.forward(&hidden_states)?;
            let attended = self.attention.forward_physical(
                &normalized,
                &mrope_plan,
                cache,
                &mut prepared,
                0,
            )?;
            let hidden_states = (&residual + &attended)?;
            let residual = hidden_states.clone();
            let normalized = self.post_attention_layernorm.forward(&hidden_states)?;
            let mlp = self.mlp.forward(&normalized)?;
            let hidden_states = (&residual + &mlp)?;
            self.norm.forward(&hidden_states).map_err(Error::from)
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
            let residual = hidden_states.clone();
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
            let hidden_states = residual.broadcast_add(&attended)?;
            let residual = hidden_states.clone();
            let normalized = self.post_attention_layernorm.forward(&hidden_states)?;
            let mlp = self.mlp.forward(&normalized)?;
            let hidden_states = residual.broadcast_add(&mlp)?;
            self.norm.forward(&hidden_states).map_err(Error::from)
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
    pub(crate) fn draft_recurrently<S, E>(
        &self,
        first_lm_head_hidden: &Tensor,
        depth: Qwen38MtpDepth,
        continuation_positions: &[[usize; 3]],
        cache: &mut PhysicalPagedKvCache,
        mut select: S,
        mut embed: E,
    ) -> Result<Qwen38MtpDraftSequence>
    where
        S: FnMut(usize, &Tensor) -> Result<u32>,
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
            let token = select(step, &current)?;
            token_ids.push(token);
            lm_head_hidden.push(current.clone());
            if step < continuation_count {
                current =
                    self.forward_step(embed(token)?, current, continuation_positions[step], cache)?;
            }
        }
        Ok(Qwen38MtpDraftSequence {
            token_ids,
            lm_head_hidden,
        })
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
    ) -> Result<Qwen38MtpDraftSequence>
    where
        S: FnMut(usize, &Tensor) -> Result<u32>,
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
mod tests {
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

    struct TestDir(PathBuf);

    impl TestDir {
        fn new(label: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "izwi-qwen38-mtp-{label}-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    struct FixtureTensor {
        name: String,
        dtype: SafeDType,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    fn tiny_config() -> Qwen38NativeConfig {
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

    fn write_tiny_checkpoint(dir: &Path, config: &Qwen38NativeConfig) {
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

    fn tiny_cache(config: &Qwen38NativeConfig) -> PhysicalPagedKvCache {
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

    fn tiny_shared_caches(config: &Qwen38NativeConfig, rows: usize) -> Vec<PhysicalPagedKvCache> {
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
                    Ok(3 + step as u32)
                },
                |token| {
                    embed_calls += 1;
                    let mut values = vec![0.0f32; 4];
                    values[token as usize % 4] = 1.0;
                    Tensor::from_vec(values, (1, 1, 4), &Device::Cpu).map_err(Error::from)
                },
            )
            .unwrap();
        assert_eq!(drafts.token_ids, [3, 4, 5]);
        assert_eq!(drafts.lm_head_hidden.len(), 3);
        assert_eq!(embed_calls, 2);
        assert_eq!(cache.context_len(), 4);
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
}
