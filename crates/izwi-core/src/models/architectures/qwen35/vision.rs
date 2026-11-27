use candle_core::quantized::gguf_file::Value as GgufValue;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{Conv2d, Conv2dConfig, Embedding, LayerNorm, Linear};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};

use crate::error::{Error, Result};
use crate::models::shared::chat::{ChatMediaInput, ChatMediaKind};
use crate::models::shared::media::{decode_image, fetch_media_bytes, image_media_host_bytes};
use crate::models::shared::weights::gguf::GgufLoader;

const DEFAULT_IMAGE_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const DEFAULT_IMAGE_STD: [f32; 3] = [0.26862955, 0.2613026, 0.2757771];
// The vision implementation materializes dense f32 attention scores for as
// many as 2,560 patches. Reserve both the score and softmax buffers plus qkv
// and allocator headroom before touching untrusted media.
const VISION_BACKEND_ATTENTION_WORKSPACE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const IMAGE_BACKEND_RETAINED_TENSOR_BYTES: u64 = 128 * 1024 * 1024;

pub type Qwen35MediaResourceEstimate = crate::models::shared::media::MediaResourceEstimate;

pub fn media_resource_estimate(
    media_inputs: &[ChatMediaInput],
) -> Result<Qwen35MediaResourceEstimate> {
    let host_bytes = image_media_host_bytes(media_inputs, "Qwen3.5")?;
    let count = u64::try_from(media_inputs.len())
        .map_err(|_| Error::Overloaded("Qwen3.5 media count overflow".to_string()))?;
    let retained_tensor_bytes = count
        .checked_mul(IMAGE_BACKEND_RETAINED_TENSOR_BYTES)
        .ok_or_else(|| Error::Overloaded("Qwen3.5 media tensor workspace overflow".to_string()))?;
    let backend_tensor_bytes = if count == 0 {
        0
    } else {
        VISION_BACKEND_ATTENTION_WORKSPACE_BYTES
            .checked_add(retained_tensor_bytes)
            .ok_or_else(|| {
                Error::Overloaded("Qwen3.5 media tensor workspace overflow".to_string())
            })?
    };
    Ok(Qwen35MediaResourceEstimate {
        host_bytes,
        backend_tensor_bytes,
    })
}

#[derive(Debug, Clone)]
pub struct PreparedVisionInputs {
    pub embeddings: Tensor,
    pub grids: Vec<[usize; 3]>,
    pub token_counts: Vec<usize>,
}

#[derive(Debug, Clone)]
struct Qwen35VisionConfig {
    block_count: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
    num_position_embeddings: usize,
    layer_norm_epsilon: f64,
    projector_uses_gelu: bool,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    min_pixels: usize,
    max_pixels: usize,
}

pub struct Qwen35VisionModel {
    device: Device,
    config: Qwen35VisionConfig,
    patch_embed: PatchEmbed,
    pos_embed: Embedding,
    blocks: Vec<VisionBlock>,
    merger: PatchMerger,
}

struct PatchEmbed {
    proj_t0: Conv2d,
    proj_t1: Conv2d,
    bias: Tensor,
    in_channels: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    hidden_size: usize,
}

impl PatchEmbed {
    fn load(loader: &GgufLoader, cfg: &Qwen35VisionConfig, device: &Device) -> Result<Self> {
        let weight_t0 = load_dense(loader, device, "v.patch_embd.weight", Some(DType::F32))?;
        let weight_t1 = load_dense(loader, device, "v.patch_embd.weight.1", Some(DType::F32))?;
        let bias = load_dense(loader, device, "v.patch_embd.bias", Some(DType::F32))?;
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };

        Ok(Self {
            proj_t0: Conv2d::new(weight_t0, None, conv_cfg),
            proj_t1: Conv2d::new(weight_t1, None, conv_cfg),
            bias,
            in_channels: 3,
            patch_size: cfg.patch_size,
            temporal_patch_size: cfg.temporal_patch_size,
            hidden_size: cfg.hidden_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.reshape((
            (),
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        ))?;
        let xs_t0 = xs.i((.., .., 0, .., ..))?;
        let xs_t1 = xs.i((.., .., 1, .., ..))?;
        let xs = (&self.proj_t0.forward(&xs_t0)? + &self.proj_t1.forward(&xs_t1)?)?;
        let xs = xs.reshape(((), self.hidden_size))?;
        xs.broadcast_add(&self.bias.unsqueeze(0)?)
            .map_err(Error::from)
    }
}

struct VisionAttention {
    qkv: VisionLinear,
    proj: VisionLinear,
    num_heads: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn load(
        loader: &GgufLoader,
        device: &Device,
        prefix: &str,
        cfg: &Qwen35VisionConfig,
    ) -> Result<Self> {
        Ok(Self {
            qkv: load_linear(
                loader,
                device,
                &format!("{prefix}.attn_qkv"),
                vision_linear_dtype(device),
            )?,
            proj: load_linear(
                loader,
                device,
                &format!("{prefix}.attn_out"),
                vision_linear_dtype(device),
            )?,
            num_heads: cfg.num_heads,
            head_dim: cfg.hidden_size / cfg.num_heads,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        let hidden_states = self.qkv.forward(xs)?;
        let qkv = hidden_states
            .reshape((seq_len, 3, self.num_heads, self.head_dim))?
            .permute((1, 0, 2, 3))?;
        let mut q = qkv.i(0)?.squeeze(0)?;
        let mut k = qkv.i(1)?.squeeze(0)?;
        let mut v = qkv.i(2)?.squeeze(0)?;

        let cos = cos.to_dtype(DType::F32)?;
        let sin = sin.to_dtype(DType::F32)?;
        q = q.to_dtype(DType::F32)?;
        k = k.to_dtype(DType::F32)?;
        v = v.to_dtype(DType::F32)?;
        (q, k) = apply_rotary_pos_emb_vision(&q, &k, &cos, &sin)?;

        let mut outputs = Vec::new();
        for window in cu_seqlens.windows(2) {
            let start = window[0];
            let end = window[1];
            if end <= start {
                continue;
            }
            let len = end - start;
            let q_chunk = q.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;
            let k_chunk = k.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;
            let v_chunk = v.narrow(0, start, len)?.transpose(0, 1)?.contiguous()?;

            let q = q_chunk.unsqueeze(0)?.contiguous()?;
            let k = k_chunk.unsqueeze(0)?.contiguous()?;
            let v = v_chunk.unsqueeze(0)?.contiguous()?;
            let k_t = k.transpose(2, 3)?.contiguous()?;
            let attn_weights = (q.matmul(&k_t)? / (self.head_dim as f64).sqrt())?;
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            let chunk_out = attn_weights
                .contiguous()?
                .matmul(&v)?
                .squeeze(0)?
                .transpose(0, 1)?
                .reshape((len, self.num_heads * self.head_dim))?;
            outputs.push(chunk_out.to_dtype(xs.dtype())?);
        }

        let attn_output = Tensor::cat(&outputs.iter().collect::<Vec<_>>(), 0)?;
        self.proj.forward(&attn_output)
    }
}

struct VisionMlp {
    fc1: VisionLinear,
    fc2: VisionLinear,
}

impl VisionMlp {
    fn load(loader: &GgufLoader, device: &Device, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1: load_linear(
                loader,
                device,
                &format!("{prefix}.ffn_up"),
                vision_linear_dtype(device),
            )?,
            fc2: load_linear(
                loader,
                device,
                &format!("{prefix}.ffn_down"),
                vision_linear_dtype(device),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = xs.gelu()?;
        self.fc2.forward(&xs)
    }
}

struct VisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: VisionAttention,
    mlp: VisionMlp,
}

impl VisionBlock {
    fn load(
        loader: &GgufLoader,
        device: &Device,
        layer_idx: usize,
        cfg: &Qwen35VisionConfig,
    ) -> Result<Self> {
        let prefix = format!("v.blk.{layer_idx}");
        Ok(Self {
            norm1: load_layer_norm(
                loader,
                device,
                &format!("{prefix}.ln1"),
                cfg.layer_norm_epsilon,
            )?,
            norm2: load_layer_norm(
                loader,
                device,
                &format!("{prefix}.ln2"),
                cfg.layer_norm_epsilon,
            )?,
            attn: VisionAttention::load(loader, device, &prefix, cfg)?,
            mlp: VisionMlp::load(loader, device, &prefix)?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &[usize],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let normed = self.norm1.forward(xs)?;
        let attn_out = self.attn.forward(&normed, cu_seqlens, cos, sin)?;
        let xs_att = xs.add(&attn_out)?;
        let mlp_out = self.mlp.forward(&self.norm2.forward(&xs_att)?)?;
        xs_att.add(&mlp_out).map_err(Error::from)
    }
}

struct PatchMerger {
    norm: LayerNorm,
    use_postshuffle_norm: bool,
    spatial_merge_unit: usize,
    merged_hidden_size: usize,
    fc1: VisionLinear,
    fc2: VisionLinear,
    use_gelu: bool,
}

impl PatchMerger {
    fn load(loader: &GgufLoader, device: &Device, cfg: &Qwen35VisionConfig) -> Result<Self> {
        let fc1 = load_linear(loader, device, "mm.0", vision_linear_dtype(device))?;
        let fc2 = load_linear(loader, device, "mm.2", vision_linear_dtype(device))?;
        let merged_hidden_size = fc1.in_features()?;
        let norm_weight = load_dense(loader, device, "v.post_ln.weight", Some(DType::F32))?;
        let norm_bias = load_dense(loader, device, "v.post_ln.bias", Some(DType::F32))?;
        let norm_dim = norm_weight.elem_count();
        let use_postshuffle_norm = norm_dim == merged_hidden_size;
        if !use_postshuffle_norm && norm_dim != cfg.hidden_size {
            return Err(Error::ModelLoadError(format!(
                "Unexpected Qwen3.5 projector norm width {norm_dim}; expected {} or {merged_hidden_size}",
                cfg.hidden_size
            )));
        }

        Ok(Self {
            norm: LayerNorm::new(
                norm_weight.reshape((norm_dim,))?,
                norm_bias.reshape((norm_dim,))?,
                cfg.layer_norm_epsilon,
            ),
            use_postshuffle_norm,
            spatial_merge_unit: cfg.spatial_merge_size.pow(2),
            merged_hidden_size,
            fc1,
            fc2,
            use_gelu: cfg.projector_uses_gelu,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let seq_len = xs.dim(0)?;
        if seq_len % self.spatial_merge_unit != 0 {
            return Err(Error::InferenceError(format!(
                "Sequence length {} is not divisible by spatial merge unit {}",
                seq_len, self.spatial_merge_unit
            )));
        }

        let grouped = seq_len / self.spatial_merge_unit;
        let norm_input = if self.use_postshuffle_norm {
            xs.reshape((grouped, self.merged_hidden_size))?
        } else {
            xs.clone()
        };
        let normed = self.norm.forward(&norm_input)?;
        let reshaped = if self.use_postshuffle_norm {
            normed
        } else {
            normed.reshape((grouped, self.merged_hidden_size))?
        };
        let xs = self.fc1.forward(&reshaped)?;
        let xs = if self.use_gelu {
            xs.gelu()?
        } else {
            candle_nn::ops::silu(&xs)?
        };
        self.fc2.forward(&xs)
    }
}

struct VisionRotaryEmbedding {
    inv_freq: Tensor,
}

impl VisionRotaryEmbedding {
    const THETA: f32 = 10000.;

    fn new(dim: usize, device: &Device) -> Result<Self> {
        let inv_freq = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / Self::THETA.powf(i as f32 / dim as f32))
            .collect::<Vec<_>>();
        let inv_freq_len = inv_freq.len();
        Ok(Self {
            inv_freq: Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?,
        })
    }

    fn make_embeds(&self, seqlen: usize) -> Result<Tensor> {
        let seq =
            Tensor::arange(0f32, seqlen as f32, self.inv_freq.device())?.unsqueeze(D::Minus1)?;
        seq.broadcast_matmul(&self.inv_freq).map_err(Error::from)
    }
}

impl Qwen35VisionModel {
    pub fn load(
        loader: &GgufLoader,
        device: &Device,
        expected_text_hidden_size: usize,
    ) -> Result<Self> {
        let config = parse_vision_config(loader)?;
        let pos_weight = load_dense(loader, device, "v.position_embd.weight", Some(DType::F32))?;
        let (num_position_embeddings, hidden_size) = pos_weight.dims2()?;
        if hidden_size != config.hidden_size {
            return Err(Error::ModelLoadError(format!(
                "Qwen3.5 vision position embedding width mismatch: {hidden_size} vs {}",
                config.hidden_size
            )));
        }
        if config.hidden_size != expected_text_hidden_size && loader.has_tensor("mm.2.weight") {
            let projector_out = loader
                .tensor_shape("mm.2.weight")
                .and_then(|shape| shape.first().copied())
                .ok_or_else(|| {
                    Error::ModelLoadError("Qwen3.5 projector mm.2.weight has no shape".into())
                })?;
            if projector_out != expected_text_hidden_size {
                return Err(Error::ModelLoadError(format!(
                    "Qwen3.5 projector output width mismatch: {projector_out} vs expected text hidden size {expected_text_hidden_size}"
                )));
            }
        }

        let patch_embed = PatchEmbed::load(loader, &config, device)?;
        let pos_embed = Embedding::new(pos_weight, hidden_size);
        let mut blocks = Vec::with_capacity(config.block_count);
        for layer_idx in 0..config.block_count {
            blocks.push(VisionBlock::load(loader, device, layer_idx, &config)?);
        }
        let merger = PatchMerger::load(loader, device, &config)?;

        let mut config = config;
        config.num_position_embeddings = num_position_embeddings;
        Ok(Self {
            device: device.clone(),
            config,
            patch_embed,
            pos_embed,
            blocks,
            merger,
        })
    }

    pub fn encode_media(
        &self,
        media_inputs: &[ChatMediaInput],
    ) -> Result<Option<PreparedVisionInputs>> {
        if media_inputs.is_empty() {
            return Ok(None);
        }
        let _ = media_resource_estimate(media_inputs)?;

        let prepared_media = prepare_image_media_with(media_inputs, fetch_media_bytes, |bytes| {
            self.preprocess_image(decode_image(bytes)?)
        })?;
        let mut all_patches = Vec::with_capacity(prepared_media.len());
        let mut grids = Vec::with_capacity(prepared_media.len());
        let mut token_counts = Vec::with_capacity(prepared_media.len());
        for (patches, grid, token_count) in prepared_media {
            all_patches.push(patches);
            grids.push(grid);
            token_counts.push(token_count);
        }

        let patch_refs: Vec<&Tensor> = all_patches.iter().collect();
        let patches = Tensor::cat(&patch_refs, 0)?;
        let grid_flat: Vec<u32> = grids
            .iter()
            .flat_map(|grid| grid.iter().map(|value| *value as u32))
            .collect();
        let grid_thw = Tensor::from_vec(grid_flat, (grids.len(), 3), &self.device)?;
        let embeddings = self.forward(&patches, &grid_thw)?;

        let expected_tokens: usize = token_counts.iter().sum();
        if embeddings.dim(0)? != expected_tokens {
            return Err(Error::InferenceError(format!(
                "Qwen3.5 vision token count mismatch: encoder returned {}, expected {}",
                embeddings.dim(0)?,
                expected_tokens
            )));
        }

        Ok(Some(PreparedVisionInputs {
            embeddings,
            grids,
            token_counts,
        }))
    }

    pub fn spatial_merge_size(&self) -> usize {
        self.config.spatial_merge_size
    }

    fn forward(&self, xs: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        let dtype = self.pos_embed.embeddings().dtype();
        let xs = self.patch_embed.forward(&xs.to_dtype(dtype)?)?;
        let pos_embeds = self.fast_pos_embed_interpolate(grid_thw)?;
        let mut hidden_states = xs.add(&pos_embeds)?;

        let rotary_pos_emb = self.rot_pos_emb(grid_thw)?;
        let seq_len = hidden_states.dim(0)?;
        let rotary_pos_emb = rotary_pos_emb.reshape((seq_len, ()))?;
        let emb = Tensor::cat(&[&rotary_pos_emb, &rotary_pos_emb], D::Minus1)?;
        let cos = emb.cos()?.to_dtype(DType::F32)?;
        let sin = emb.sin()?.to_dtype(DType::F32)?;
        let cu_seqlens = self.build_cu_seqlens(grid_thw)?;

        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states, &cu_seqlens, &cos, &sin)?;
        }

        self.merger.forward(&hidden_states)
    }

    fn fast_pos_embed_interpolate(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let device = self.pos_embed.embeddings().device();
        let dtype = self.pos_embed.embeddings().dtype();
        let grid = grid_thw.to_vec2::<u32>()?;
        let num_grid_per_side =
            (self.config.num_position_embeddings as f64).sqrt().round() as usize;

        let mut idx_lists: [Vec<i64>; 4] = Default::default();
        let mut weight_lists: [Vec<f32>; 4] = Default::default();
        let mut hw_lengths = Vec::with_capacity(grid.len());

        for g in &grid {
            let h = g[1] as usize;
            let w = g[2] as usize;
            hw_lengths.push(h * w);

            let h_vals = linspace_points(h, num_grid_per_side);
            let w_vals = linspace_points(w, num_grid_per_side);

            let h_floor: Vec<usize> = h_vals.iter().map(|v| v.floor() as usize).collect();
            let w_floor: Vec<usize> = w_vals.iter().map(|v| v.floor() as usize).collect();
            let h_ceil: Vec<usize> = h_vals
                .iter()
                .map(|v| (v.ceil() as usize).min(num_grid_per_side - 1))
                .collect();
            let w_ceil: Vec<usize> = w_vals
                .iter()
                .map(|v| (v.ceil() as usize).min(num_grid_per_side - 1))
                .collect();
            let dh: Vec<f32> = h_vals
                .iter()
                .zip(&h_floor)
                .map(|(v, floor)| v - *floor as f32)
                .collect();
            let dw: Vec<f32> = w_vals
                .iter()
                .zip(&w_floor)
                .map(|(v, floor)| v - *floor as f32)
                .collect();

            for ((&hf, &hc), &dh_val) in h_floor.iter().zip(&h_ceil).zip(&dh) {
                for ((&wf, &wc), &dw_val) in w_floor.iter().zip(&w_ceil).zip(&dw) {
                    let base00 = (hf * num_grid_per_side + wf) as i64;
                    let base01 = (hf * num_grid_per_side + wc) as i64;
                    let base10 = (hc * num_grid_per_side + wf) as i64;
                    let base11 = (hc * num_grid_per_side + wc) as i64;

                    idx_lists[0].push(base00);
                    idx_lists[1].push(base01);
                    idx_lists[2].push(base10);
                    idx_lists[3].push(base11);

                    weight_lists[0].push((1.0 - dh_val) * (1.0 - dw_val));
                    weight_lists[1].push((1.0 - dh_val) * dw_val);
                    weight_lists[2].push(dh_val * (1.0 - dw_val));
                    weight_lists[3].push(dh_val * dw_val);
                }
            }
        }

        let idx_tensors = idx_lists
            .iter()
            .map(|idxs| Tensor::from_vec(idxs.clone(), (idxs.len(),), device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let idx_tensor = Tensor::stack(&idx_tensors.iter().collect::<Vec<_>>(), 0)?;

        let weight_tensors = weight_lists
            .iter()
            .map(|weights| Tensor::from_vec(weights.clone(), (weights.len(),), device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let weight_tensor =
            Tensor::stack(&weight_tensors.iter().collect::<Vec<_>>(), 0)?.to_dtype(dtype)?;

        let pos_embeds = self.pos_embed.forward(&idx_tensor)?;
        let pos_embeds = pos_embeds.broadcast_mul(&weight_tensor.unsqueeze(D::Minus1)?)?;
        let pos_embeds = pos_embeds.sum(0)?;

        let mut splits = Vec::with_capacity(hw_lengths.len());
        let mut start = 0;
        for len in hw_lengths {
            splits.push(pos_embeds.narrow(0, start, len)?);
            start += len;
        }

        let mut permuted = Vec::with_capacity(grid.len());
        for (pos_embed, g) in splits.into_iter().zip(&grid) {
            let t = g[0] as usize;
            let h = g[1] as usize;
            let w = g[2] as usize;
            let pos_embed = pos_embed.repeat((t, 1))?;
            let pos_embed = pos_embed.reshape((
                t,
                h / self.config.spatial_merge_size,
                self.config.spatial_merge_size,
                w / self.config.spatial_merge_size,
                self.config.spatial_merge_size,
                self.config.hidden_size,
            ))?;
            let pos_embed = pos_embed
                .permute((0, 1, 3, 2, 4, 5))?
                .reshape((t * h * w, self.config.hidden_size))?;
            permuted.push(pos_embed);
        }

        Tensor::cat(&permuted.iter().collect::<Vec<_>>(), 0).map_err(Error::from)
    }

    fn rot_pos_emb(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let device = self.device.clone();
        let grid = grid_thw.to_vec2::<u32>()?;
        let max_hw = grid
            .iter()
            .flat_map(|values| values[1..3].iter())
            .copied()
            .max()
            .unwrap_or(0) as usize;
        let rotary = VisionRotaryEmbedding::new(
            self.config.hidden_size / self.config.num_heads / 2,
            &device,
        )?;
        let freq_table = rotary.make_embeds(max_hw)?;

        let mut coords = Vec::new();
        for g in &grid {
            let t = g[0] as usize;
            let h = g[1] as usize;
            let w = g[2] as usize;
            let merged_h = h / self.config.spatial_merge_size;
            let merged_w = w / self.config.spatial_merge_size;

            let mut base_coords = Vec::with_capacity(h * w);
            for block_row in 0..merged_h {
                for block_col in 0..merged_w {
                    for inner_row in 0..self.config.spatial_merge_size {
                        for inner_col in 0..self.config.spatial_merge_size {
                            base_coords.push((
                                (block_row * self.config.spatial_merge_size + inner_row) as i64,
                                (block_col * self.config.spatial_merge_size + inner_col) as i64,
                            ));
                        }
                    }
                }
            }

            for _ in 0..t {
                coords.extend(base_coords.iter().copied());
            }
        }

        let total_tokens = coords.len();
        let rows = Tensor::from_vec(
            coords.iter().map(|(row, _)| *row).collect::<Vec<_>>(),
            (total_tokens,),
            &device,
        )?;
        let cols = Tensor::from_vec(
            coords.iter().map(|(_, col)| *col).collect::<Vec<_>>(),
            (total_tokens,),
            &device,
        )?;
        let row_embeds = freq_table.index_select(&rows, 0)?;
        let col_embeds = freq_table.index_select(&cols, 0)?;
        Tensor::stack(&[row_embeds, col_embeds], D::Minus2)?
            .reshape((total_tokens, freq_table.dim(D::Minus1)? * 2))
            .map_err(Error::from)
    }

    fn build_cu_seqlens(&self, grid_thw: &Tensor) -> Result<Vec<usize>> {
        let grid = grid_thw.to_vec2::<u32>()?;
        let mut cu = Vec::with_capacity(grid.iter().map(|g| g[0] as usize).sum::<usize>() + 1);
        cu.push(0);
        let mut acc = 0usize;
        for g in &grid {
            let area = (g[1] * g[2]) as usize;
            for _ in 0..(g[0] as usize) {
                acc += area;
                cu.push(acc);
            }
        }
        Ok(cu)
    }

    fn preprocess_image(&self, image: DynamicImage) -> Result<(Tensor, [usize; 3], usize)> {
        let (height, width) = image.dimensions();
        let factor = self.config.patch_size * self.config.spatial_merge_size;
        let (resized_height, resized_width) = smart_resize(
            height as usize,
            width as usize,
            factor,
            self.config.min_pixels,
            self.config.max_pixels,
        )?;
        let resized = image
            .resize_exact(
                resized_width as u32,
                resized_height as u32,
                FilterType::CatmullRom,
            )
            .to_rgb8();

        let mut frame = vec![0f32; 3 * resized_height * resized_width];
        for (x, y, pixel) in resized.enumerate_pixels() {
            let base = y as usize * resized_width + x as usize;
            for channel in 0..3 {
                let value = pixel[channel] as f32 / 255.0;
                frame[channel * resized_height * resized_width + base] =
                    (value - self.config.image_mean[channel]) / self.config.image_std[channel];
            }
        }

        let mut frames = vec![frame];
        while !frames.len().is_multiple_of(self.config.temporal_patch_size) {
            let last = frames.last().cloned().ok_or_else(|| {
                Error::InvalidInput("Qwen3.5 image preprocessing produced no frames".to_string())
            })?;
            frames.push(last);
        }

        let grid_t = frames.len() / self.config.temporal_patch_size;
        let grid_h = resized_height / self.config.patch_size;
        let grid_w = resized_width / self.config.patch_size;
        let llm_grid_h = grid_h / self.config.spatial_merge_size;
        let llm_grid_w = grid_w / self.config.spatial_merge_size;
        let patch_dim =
            3 * self.config.temporal_patch_size * self.config.patch_size * self.config.patch_size;
        let seq_len = grid_t * grid_h * grid_w;
        let mut flatten = Vec::with_capacity(seq_len * patch_dim);

        for t in 0..grid_t {
            for block_row in 0..llm_grid_h {
                for block_col in 0..llm_grid_w {
                    for inner_row in 0..self.config.spatial_merge_size {
                        for inner_col in 0..self.config.spatial_merge_size {
                            for channel in 0..3 {
                                for temporal in 0..self.config.temporal_patch_size {
                                    let frame =
                                        &frames[t * self.config.temporal_patch_size + temporal];
                                    let patch_row = (block_row * self.config.spatial_merge_size
                                        + inner_row)
                                        * self.config.patch_size;
                                    let patch_col = (block_col * self.config.spatial_merge_size
                                        + inner_col)
                                        * self.config.patch_size;
                                    for patch_r in 0..self.config.patch_size {
                                        let row = patch_row + patch_r;
                                        let base = channel * resized_height * resized_width
                                            + row * resized_width;
                                        for patch_c in 0..self.config.patch_size {
                                            flatten.push(frame[base + patch_col + patch_c]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let patches = Tensor::from_vec(flatten, (seq_len, patch_dim), &self.device)?;
        Ok((
            patches,
            [grid_t, grid_h, grid_w],
            grid_t * llm_grid_h * llm_grid_w,
        ))
    }
}

fn prepare_image_media_with<T, L, E>(
    media_inputs: &[ChatMediaInput],
    mut load: L,
    mut encode: E,
) -> Result<Vec<T>>
where
    L: FnMut(&str) -> Result<Vec<u8>>,
    E: FnMut(&[u8]) -> Result<T>,
{
    let mut prepared = Vec::with_capacity(media_inputs.len());
    for media in media_inputs {
        if media.kind != ChatMediaKind::Image {
            return Err(Error::InvalidInput(
                "Qwen3.5 video inputs are not implemented yet".to_string(),
            ));
        }
        let bytes = load(&media.source)?;
        prepared.push(encode(&bytes)?);
    }
    Ok(prepared)
}

fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    let aspect_ratio = height.max(width) as f64 / height.min(width) as f64;
    if aspect_ratio > 200.0 {
        return Err(Error::InvalidInput(format!(
            "Qwen3.5 image aspect ratio must be smaller than 200, got {aspect_ratio}"
        )));
    }

    let mut h_bar = ((height as f64 / factor as f64).round() as usize).max(1) * factor;
    let mut w_bar = ((width as f64 / factor as f64).round() as usize).max(1) * factor;
    if h_bar * w_bar > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        h_bar = factor.max(((height as f64 / beta / factor as f64).floor() as usize) * factor);
        w_bar = factor.max(((width as f64 / beta / factor as f64).floor() as usize) * factor);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        h_bar = ((height as f64 * beta / factor as f64).ceil() as usize) * factor;
        w_bar = ((width as f64 * beta / factor as f64).ceil() as usize) * factor;
    }
    Ok((h_bar, w_bar))
}

fn linspace_points(steps: usize, num_grid_per_side: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![0.0];
    }
    let max_val = (num_grid_per_side - 1) as f32;
    let step = max_val / (steps.saturating_sub(1)) as f32;
    (0..steps).map(|idx| idx as f32 * step).collect()
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1).map_err(Error::from)
}

fn apply_rotary_pos_emb_vision(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let cos = cos.unsqueeze(D::Minus2)?;
    let sin = sin.unsqueeze(D::Minus2)?;
    let q_embed = (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin)?)?;
    Ok((q_embed, k_embed))
}

fn parse_vision_config(loader: &GgufLoader) -> Result<Qwen35VisionConfig> {
    let patch_size = required_usize(loader, "clip.vision.patch_size")?;
    let spatial_merge_size = required_usize(loader, "clip.vision.spatial_merge_size")?;
    Ok(Qwen35VisionConfig {
        block_count: required_usize(loader, "clip.vision.block_count")?,
        hidden_size: required_usize(loader, "clip.vision.embedding_length")?,
        intermediate_size: required_usize(loader, "clip.vision.feed_forward_length")?,
        num_heads: required_usize(loader, "clip.vision.attention.head_count")?,
        patch_size,
        temporal_patch_size: 2,
        spatial_merge_size,
        num_position_embeddings: 0,
        layer_norm_epsilon: required_f64(loader, "clip.vision.attention.layer_norm_epsilon")?,
        projector_uses_gelu: loader
            .metadata_value("clip.use_gelu")
            .and_then(gguf_to_bool)
            .unwrap_or(true),
        image_mean: optional_f32_array(loader, "clip.vision.image_mean")?
            .unwrap_or(DEFAULT_IMAGE_MEAN),
        image_std: optional_f32_array(loader, "clip.vision.image_std")?
            .unwrap_or(DEFAULT_IMAGE_STD),
        min_pixels: 56 * 56,
        max_pixels: patch_size * patch_size * 2 * 1280,
    })
}

/// The weight dtype of the vision tower's linear layers: F16 on an
/// accelerator, where the tower is stored F16 and F32 would double it
/// (0.66 to 1.33 GB for the 2B/4B projector), F32 on CPU.
fn vision_linear_dtype(device: &Device) -> DType {
    if device.is_cpu() {
        DType::F32
    } else {
        DType::F16
    }
}

/// A vision linear layer: weights kept in [`vision_linear_dtype`] and
/// widened to the activation dtype for each product. The checkpoint stores
/// the tower in F16, so widening is exact and the products match an F32
/// tower bit for bit; only one layer's F32 weight exists at a time, while an
/// image is encoded.
struct VisionLinear(Linear);

impl VisionLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if self.0.weight().dtype() == xs.dtype() {
            return self.0.forward(xs).map_err(Error::from);
        }
        let weight = self.0.weight().to_dtype(xs.dtype())?;
        Linear::new(weight, self.0.bias().cloned())
            .forward(xs)
            .map_err(Error::from)
    }

    fn in_features(&self) -> Result<usize> {
        Ok(self.0.weight().dims2()?.1)
    }
}

fn load_linear(
    loader: &GgufLoader,
    device: &Device,
    prefix: &str,
    dtype: DType,
) -> Result<VisionLinear> {
    let weight = load_dense_via_host(loader, device, &format!("{prefix}.weight"), dtype)?;
    let bias_name = format!("{prefix}.bias");
    // Biases are small and stored F32; they stay F32.
    let bias = if loader.has_tensor(&bias_name) {
        Some(load_dense_via_host(loader, device, &bias_name, DType::F32)?)
    } else {
        None
    };
    Ok(VisionLinear(Linear::new(weight, bias)))
}

/// A tensor converted to `dtype` on the host and uploaded at its exact size:
/// a conversion on the device would hold the F32 expansion beside the
/// result and round the result's buffer up to a power of two.
fn load_dense_via_host(
    loader: &GgufLoader,
    device: &Device,
    name: &str,
    dtype: DType,
) -> Result<Tensor> {
    loader
        .load_qtensor(name, &Device::Cpu)?
        .dequantize(&Device::Cpu)?
        .to_dtype(dtype)?
        .to_device(device)
        .map_err(Error::from)
}

fn load_layer_norm(
    loader: &GgufLoader,
    device: &Device,
    prefix: &str,
    eps: f64,
) -> Result<LayerNorm> {
    let weight = load_dense(
        loader,
        device,
        &format!("{prefix}.weight"),
        Some(DType::F32),
    )?;
    let bias = load_dense(loader, device, &format!("{prefix}.bias"), Some(DType::F32))?;
    Ok(LayerNorm::new(
        weight.reshape((weight.elem_count(),))?,
        bias.reshape((bias.elem_count(),))?,
        eps,
    ))
}

fn load_dense(
    loader: &GgufLoader,
    device: &Device,
    name: &str,
    dtype: Option<DType>,
) -> Result<Tensor> {
    let mut tensor = loader
        .load_qtensor(name, device)?
        .dequantize(device)
        .map_err(Error::from)?;
    if let Some(dtype) = dtype {
        if tensor.dtype() != dtype {
            tensor = tensor.to_dtype(dtype)?;
        }
    }
    Ok(tensor)
}

fn required_usize(loader: &GgufLoader, key: &str) -> Result<usize> {
    loader
        .get_metadata_u64(key)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))
}

fn required_f64(loader: &GgufLoader, key: &str) -> Result<f64> {
    loader
        .metadata_value(key)
        .and_then(gguf_to_f64)
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))
}

fn optional_f32_array(loader: &GgufLoader, key: &str) -> Result<Option<[f32; 3]>> {
    let Some(value) = loader.metadata_value(key) else {
        return Ok(None);
    };
    let GgufValue::Array(items) = value else {
        return Err(Error::ModelLoadError(format!(
            "Expected GGUF array metadata for {key}"
        )));
    };
    if items.len() != 3 {
        return Err(Error::ModelLoadError(format!(
            "Expected 3 values for {key}, found {}",
            items.len()
        )));
    }
    let mut out = [0f32; 3];
    for (idx, item) in items.iter().enumerate() {
        out[idx] = gguf_to_f64(item).ok_or_else(|| {
            Error::ModelLoadError(format!("Invalid floating-point metadata for {key}"))
        })? as f32;
    }
    Ok(Some(out))
}

fn gguf_to_bool(value: &GgufValue) -> Option<bool> {
    match value {
        GgufValue::Bool(value) => Some(*value),
        GgufValue::U8(value) => Some(*value != 0),
        GgufValue::I8(value) => Some(*value != 0),
        _ => None,
    }
}

fn gguf_to_f64(value: &GgufValue) -> Option<f64> {
    match value {
        GgufValue::F64(value) => Some(*value),
        GgufValue::F32(value) => Some(*value as f64),
        GgufValue::U64(value) => Some(*value as f64),
        GgufValue::I64(value) => Some(*value as f64),
        GgufValue::U32(value) => Some(*value as f64),
        GgufValue::I32(value) => Some(*value as f64),
        GgufValue::U16(value) => Some(*value as f64),
        GgufValue::I16(value) => Some(*value as f64),
        GgufValue::U8(value) => Some(*value as f64),
        GgufValue::I8(value) => Some(*value as f64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::shared::media::MAX_MEDIA_INPUTS;
    use std::cell::Cell;

    #[test]
    fn smart_resize_matches_qwen_constraints() {
        let (h, w) = smart_resize(513, 901, 28, 56 * 56, 28 * 28 * 1280).expect("resize");
        assert_eq!(h % 28, 0);
        assert_eq!(w % 28, 0);
        assert!(h * w <= 28 * 28 * 1280);
        assert!(h * w >= 56 * 56);
    }

    #[test]
    fn media_estimate_is_bounded_and_rejects_unsupported_shapes() {
        let image = [ChatMediaInput {
            kind: ChatMediaKind::Image,
            source: "https://example.invalid/image.png".to_string(),
        }];
        let estimate = media_resource_estimate(&image).unwrap();
        // The host half is pinned to its absolute figures in `shared::media`.
        assert_eq!(
            estimate.host_bytes,
            image_media_host_bytes(&image, "Qwen3.5").unwrap()
        );
        assert_eq!(
            estimate.backend_tensor_bytes,
            VISION_BACKEND_ATTENTION_WORKSPACE_BYTES + IMAGE_BACKEND_RETAINED_TENSOR_BYTES
        );

        let too_many = (0..=MAX_MEDIA_INPUTS)
            .map(|idx| ChatMediaInput {
                kind: ChatMediaKind::Image,
                source: format!("image-{idx}.png"),
            })
            .collect::<Vec<_>>();
        assert!(matches!(
            media_resource_estimate(&too_many),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            media_resource_estimate(&[ChatMediaInput {
                kind: ChatMediaKind::Video,
                source: "video.mp4".to_string(),
            }]),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn injectable_media_pipeline_fetches_and_encodes_each_input_once() {
        let media = [ChatMediaInput {
            kind: ChatMediaKind::Image,
            source: "counted-local-image".to_string(),
        }];
        let fetches = Cell::new(0usize);
        let encodes = Cell::new(0usize);
        let prepared = prepare_image_media_with(
            &media,
            |source| {
                assert_eq!(source, "counted-local-image");
                fetches.set(fetches.get() + 1);
                Ok(vec![1, 2, 3])
            },
            |bytes| {
                assert_eq!(bytes, [1, 2, 3]);
                encodes.set(encodes.get() + 1);
                Ok(bytes.len())
            },
        )
        .unwrap();

        assert_eq!(prepared, vec![3]);
        assert_eq!(fetches.get(), 1);
        assert_eq!(encodes.get(), 1);
    }
}
