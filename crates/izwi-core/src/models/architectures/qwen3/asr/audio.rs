//! Audio tower shared by the retained Qwen speech/aligner stack.

use candle_core::{IndexOp, Module, Tensor, D};
use candle_nn::ops;
use candle_nn::{layer_norm, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};

use crate::error::{Error, Result};
use crate::models::architectures::qwen3::asr::config::AudioConfig;
use crate::models::shared::attention::flash::{
    flash_attention_requested, try_fused_self_attention,
};
use crate::models::shared::weights::mlx;

/// Compute output length after feature extraction/downsampling.
/// Matches upstream Qwen speech-family `_get_feat_extract_output_lengths`.
pub fn get_cnn_output_lengths(input_lengths: &[usize]) -> Vec<usize> {
    input_lengths
        .iter()
        .map(|&len| {
            let input_lengths_leave = len % 100;
            let feat_lengths = (input_lengths_leave.saturating_sub(1)) / 2 + 1;
            (((feat_lengths.saturating_sub(1)) / 2 + 1).saturating_sub(1)) / 2
                + 1
                + (len / 100) * 13
        })
        .collect()
}

/// Compute output length after a single conv2d with stride=2, kernel=3, padding=1.
fn conv_output_len(input_len: usize) -> usize {
    (input_len.saturating_sub(1)) / 2 + 1
}

struct SinusoidalPositionEmbedding {
    embedding: Tensor,
}

impl SinusoidalPositionEmbedding {
    fn new(max_len: usize, channels: usize, device: &candle_core::Device) -> Result<Self> {
        let half_channels = channels / 2;
        let log_timescale = (10000f32).ln() / (half_channels as f32 - 1.0);
        let inv_timescales: Vec<f32> = (0..half_channels)
            .map(|i| (-log_timescale * i as f32).exp())
            .collect();

        let mut embedding_data = Vec::with_capacity(max_len * channels);
        for pos in 0..max_len {
            for i in 0..half_channels {
                let timescale = inv_timescales[i];
                embedding_data.push((pos as f32 * timescale).sin());
            }
            for i in 0..half_channels {
                let timescale = inv_timescales[i];
                embedding_data.push((pos as f32 * timescale).cos());
            }
        }

        let embedding = Tensor::from_vec(embedding_data, (max_len, channels), device)?;
        Ok(Self { embedding })
    }

    fn get(&self, seqlen: usize) -> Result<Tensor> {
        Ok(self.embedding.narrow(0, 0, seqlen)?)
    }
}

/// Create attention mask for chunked sequences using cu_seqlens
fn create_chunked_attention_mask(
    seq_len: usize,
    cu_seqlens: &[i64],
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> Result<Tensor> {
    let min_val = f32::MIN;
    let mut mask = vec![min_val; seq_len * seq_len];

    // For each chunk, allow attention within the chunk
    for i in 1..cu_seqlens.len() {
        let start = (cu_seqlens[i - 1].max(0) as usize).min(seq_len);
        let end = (cu_seqlens[i].max(0) as usize).min(seq_len);
        if end <= start {
            continue;
        }
        for row in start..end {
            for col in start..end {
                mask[row * seq_len + col] = 0.0;
            }
        }
    }

    Tensor::from_vec(mask, (seq_len, seq_len), device)?
        .to_dtype(dtype)
        .map_err(|e| crate::error::Error::InferenceError(e.to_string()))
}

struct AudioAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl AudioAttention {
    fn load(cfg: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.d_model / cfg.encoder_attention_heads;
        let q_proj = mlx::load_linear(cfg.d_model, cfg.d_model, vb.pp("q_proj"))?;
        let k_proj = mlx::load_linear(cfg.d_model, cfg.d_model, vb.pp("k_proj"))?;
        let v_proj = mlx::load_linear(cfg.d_model, cfg.d_model, vb.pp("v_proj"))?;
        let out_proj = mlx::load_linear(cfg.d_model, cfg.d_model, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: cfg.encoder_attention_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor, cu_seqlens: &[i64]) -> Result<Tensor> {
        let seq_len = x.dim(1)?;

        let q = self
            .q_proj
            .forward(x)?
            .reshape((1, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((1, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((1, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let mask = create_chunked_attention_mask(seq_len, cu_seqlens, x.device(), q.dtype())?;
        if flash_attention_requested() {
            let sdpa_mask =
                mask.unsqueeze(0)?
                    .unsqueeze(0)?
                    .expand((1, self.num_heads, seq_len, seq_len))?;
            if let Some(fused_out) =
                try_fused_self_attention(&q, &k, &v, Some(&sdpa_mask), self.head_dim, false)?
            {
                let out = fused_out.transpose(1, 2)?.reshape((
                    1,
                    seq_len,
                    self.num_heads * self.head_dim,
                ))?;
                return self.out_proj.forward(&out).map_err(Error::from);
            }
        }

        let q = q.reshape((self.num_heads, seq_len, self.head_dim))?;
        let k = k.reshape((self.num_heads, seq_len, self.head_dim))?;
        let v = v.reshape((self.num_heads, seq_len, self.head_dim))?;

        let scale = (self.head_dim as f64).powf(-0.5);
        let mut attn = q.matmul(&k.transpose(1, 2)?)?;
        attn = attn.broadcast_mul(
            &Tensor::from_vec(vec![scale as f32], (1, 1, 1), attn.device())?
                .to_dtype(attn.dtype())?,
        )?;

        // Apply chunked attention mask
        attn = attn.broadcast_add(&mask.unsqueeze(0)?)?;

        let attn = ops::softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v)?;

        let out = out
            .reshape((1, self.num_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((1, seq_len, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&out).map_err(Error::from)
    }
}

struct AudioEncoderLayer {
    self_attn_layer_norm: LayerNorm,
    self_attn: AudioAttention,
    final_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl AudioEncoderLayer {
    fn load(cfg: &AudioConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn_layer_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let self_attn = AudioAttention::load(cfg, vb.pp("self_attn"))?;
        let final_layer_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("final_layer_norm"))?;
        let fc1 = mlx::load_linear(cfg.d_model, cfg.encoder_ffn_dim, vb.pp("fc1"))?;
        let fc2 = mlx::load_linear(cfg.encoder_ffn_dim, cfg.d_model, vb.pp("fc2"))?;
        Ok(Self {
            self_attn_layer_norm,
            self_attn,
            final_layer_norm,
            fc1,
            fc2,
        })
    }

    fn forward(&self, x: &Tensor, cu_seqlens: &[i64]) -> Result<Tensor> {
        let normed = self.self_attn_layer_norm.forward(x)?;
        let attn = self.self_attn.forward(&normed, cu_seqlens)?;
        let x = x.broadcast_add(&attn)?;

        let normed = self.final_layer_norm.forward(&x)?;
        let hidden = self.fc1.forward(&normed)?;
        let hidden = gelu(&hidden)?;
        let hidden = self.fc2.forward(&hidden)?;
        let x = x.broadcast_add(&hidden)?;

        Ok(x)
    }
}

pub struct AudioTower {
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    conv_out: Linear,
    layers: Vec<AudioEncoderLayer>,
    ln_post: LayerNorm,
    proj1: Linear,
    proj2: Linear,
    pos_embed: SinusoidalPositionEmbedding,
    cfg: AudioConfig,
}

impl AudioTower {
    pub fn load(cfg: AudioConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };

        let conv2d1 =
            mlx::load_conv2d(1, cfg.downsample_hidden_size, 3, conv_cfg, vb.pp("conv2d1"))?;
        let conv2d2 = mlx::load_conv2d(
            cfg.downsample_hidden_size,
            cfg.downsample_hidden_size,
            3,
            conv_cfg,
            vb.pp("conv2d2"),
        )?;
        let conv2d3 = mlx::load_conv2d(
            cfg.downsample_hidden_size,
            cfg.downsample_hidden_size,
            3,
            conv_cfg,
            vb.pp("conv2d3"),
        )?;

        let conv_out = mlx::load_linear_no_bias(
            cfg.downsample_hidden_size * (cfg.num_mel_bins / 8),
            cfg.d_model,
            vb.pp("conv_out"),
        )?;

        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for idx in 0..cfg.encoder_layers {
            layers.push(AudioEncoderLayer::load(
                &cfg,
                vb.pp(format!("layers.{idx}")),
            )?);
        }

        let ln_post = layer_norm(cfg.d_model, 1e-5, vb.pp("ln_post"))?;
        let proj1 = mlx::load_linear(cfg.d_model, cfg.d_model, vb.pp("proj1"))?;
        let proj2 = mlx::load_linear(cfg.d_model, cfg.output_dim, vb.pp("proj2"))?;
        let pos_embed = SinusoidalPositionEmbedding::new(1500, cfg.d_model, vb.device())?;

        Ok(Self {
            conv2d1,
            conv2d2,
            conv2d3,
            conv_out,
            layers,
            ln_post,
            proj1,
            proj2,
            pos_embed,
            cfg,
        })
    }

    pub fn forward(&self, mel: &Tensor, feature_lens: Option<&[usize]>) -> Result<Tensor> {
        let bsz = mel.dim(0)?;
        let total_frames = mel.dim(3)?;

        if bsz == 1 {
            let input_len = feature_lens
                .and_then(|lens| lens.first().copied())
                .unwrap_or(total_frames)
                .min(total_frames);
            if input_len == 0 {
                return Err(crate::error::Error::InvalidInput(
                    "Empty audio feature sequence".to_string(),
                ));
            }
            let sample = mel.i((0, 0))?;
            return self.forward_single_sample(&sample, input_len);
        }

        let mut outputs = Vec::with_capacity(bsz);
        for sample_idx in 0..bsz {
            let input_len = feature_lens
                .and_then(|lens| lens.get(sample_idx).copied())
                .unwrap_or(total_frames)
                .min(total_frames);
            if input_len == 0 {
                return Err(crate::error::Error::InvalidInput(
                    "Empty audio feature sequence".to_string(),
                ));
            }
            let sample = mel.i((sample_idx, 0))?;
            outputs.push(self.forward_single_sample(&sample, input_len)?);
        }

        if outputs.is_empty() {
            return Err(crate::error::Error::InvalidInput(
                "No audio features available for ASR batch".to_string(),
            ));
        }

        let mut max_seq_len = 0usize;
        for output in &outputs {
            max_seq_len = max_seq_len.max(output.dim(1)?);
        }
        let hidden = outputs[0].dim(2)?;
        let mut padded = Vec::with_capacity(outputs.len());
        for output in outputs {
            let seq_len = output.dim(1)?;
            if seq_len < max_seq_len {
                let pad = Tensor::zeros(
                    (1, max_seq_len - seq_len, hidden),
                    output.dtype(),
                    output.device(),
                )?;
                padded.push(Tensor::cat(&[output, pad], 1)?);
            } else {
                padded.push(output);
            }
        }
        let refs: Vec<&Tensor> = padded.iter().collect();
        Tensor::cat(&refs, 0).map_err(crate::error::Error::from)
    }

    fn forward_single_sample(&self, mel_sample: &Tensor, input_len: usize) -> Result<Tensor> {
        let n_mels = mel_sample.dim(0)?;
        let n_window = self.cfg.n_window.unwrap_or(50);
        let n_window_infer = self.cfg.n_window_infer.unwrap_or(800);
        let chunk_input_len = n_window * 2;
        if chunk_input_len == 0 {
            return Err(crate::error::Error::InvalidInput(
                "Invalid audio chunk size".to_string(),
            ));
        }

        // Match upstream: split features into fixed-size chunks before CNN.
        let feature_seq = mel_sample.transpose(0, 1)?; // [frames, n_mels]
        let mut chunk_lengths = Vec::new();
        let mut remaining = input_len;
        while remaining > 0 {
            let take = remaining.min(chunk_input_len);
            chunk_lengths.push(take);
            remaining -= take;
        }

        let mut chunks = Vec::with_capacity(chunk_lengths.len());
        let mut offset = 0usize;
        for &len in &chunk_lengths {
            let chunk = feature_seq.narrow(0, offset, len)?;
            offset += len;
            if len < chunk_input_len {
                let pad = Tensor::zeros(
                    (chunk_input_len - len, n_mels),
                    chunk.dtype(),
                    chunk.device(),
                )?;
                chunks.push(Tensor::cat(&[chunk, pad], 0)?);
            } else {
                chunks.push(chunk);
            }
        }

        let chunk_refs: Vec<&Tensor> = chunks.iter().collect();
        let mut x = Tensor::stack(&chunk_refs, 0)?; // [num_chunks, chunk_input_len, n_mels]
        x = x.transpose(1, 2)?.unsqueeze(1)?; // [num_chunks, 1, n_mels, chunk_input_len]

        x = self.conv2d1.forward(&x)?;
        x = gelu(&x)?;
        x = self.conv2d2.forward(&x)?;
        x = gelu(&x)?;
        x = self.conv2d3.forward(&x)?;
        x = gelu(&x)?;

        let num_chunks = x.dim(0)?;
        let channels = x.dim(1)?;
        let freq = x.dim(2)?;
        let frames = x.dim(3)?;

        // [b, c, f, t] -> [b, t, c, f]
        x = x.transpose(1, 3)?.transpose(2, 3)?;
        x = x.reshape((num_chunks, frames, channels * freq))?;

        x = self.conv_out.forward(&x)?;

        let pos_emb = self.pos_embed.get(x.dim(1)?)?;
        let pos_emb = pos_emb.unsqueeze(0)?.to_dtype(x.dtype())?;
        x = x.broadcast_add(&pos_emb)?;

        // Remove padded chunk tails after CNN and pack chunks back to one sequence.
        let chunk_out_lens = get_cnn_output_lengths(&chunk_lengths);
        let mut packed_chunks = Vec::with_capacity(chunk_out_lens.len());
        for (idx, &len) in chunk_out_lens.iter().enumerate() {
            let keep = len.min(frames);
            if keep == 0 {
                continue;
            }
            let chunk = x.i(idx)?.narrow(0, 0, keep)?;
            packed_chunks.push(chunk);
        }
        let packed_refs: Vec<&Tensor> = packed_chunks.iter().collect();
        let mut x = Tensor::cat(&packed_refs, 0)?.unsqueeze(0)?; // [1, total_frames_after_cnn, d_model]
        let packed_len = x.dim(1)?;

        // Build chunked self-attention windows in the CNN-downsampled domain.
        let cnn_lengths = get_cnn_output_lengths(&[input_len]);
        let max_chunk_after_cnn = get_cnn_output_lengths(&[chunk_input_len])[0].max(1);
        let infer_ratio = (n_window_infer / chunk_input_len).max(1);
        let window_after_cnn = max_chunk_after_cnn * infer_ratio;

        let mut cu_seqlens = vec![0i64];
        for &len in &cnn_lengths {
            let mut rem = len;
            while rem > window_after_cnn {
                cu_seqlens.push(*cu_seqlens.last().unwrap() + window_after_cnn as i64);
                rem -= window_after_cnn;
            }
            if rem > 0 {
                cu_seqlens.push(*cu_seqlens.last().unwrap() + rem as i64);
            }
        }
        let packed_len_i64 = packed_len as i64;
        for v in &mut cu_seqlens {
            if *v > packed_len_i64 {
                *v = packed_len_i64;
            }
        }
        cu_seqlens.dedup();
        if *cu_seqlens.last().unwrap_or(&0) < packed_len_i64 {
            cu_seqlens.push(packed_len_i64);
        }
        if cu_seqlens.len() < 2 {
            cu_seqlens = vec![0, packed_len_i64];
        }

        for layer in &self.layers {
            x = layer.forward(&x, &cu_seqlens)?;
        }

        let x = self.ln_post.forward(&x)?;
        let x = self.proj1.forward(&x)?;
        let x = gelu(&x)?;
        let x = self.proj2.forward(&x)?;
        Ok(x)
    }
}

fn gelu(x: &Tensor) -> Result<Tensor> {
    let coeff = 0.044715f32;
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let dtype = x.dtype();
    let x3 = x.powf(3.0)?;
    let coeff_t = Tensor::from_vec(vec![coeff], (1,), x.device())?.to_dtype(dtype)?;
    let x3 = x3.broadcast_mul(&coeff_t)?;
    let sqrt_t = Tensor::from_vec(vec![sqrt_2_over_pi], (1,), x.device())?.to_dtype(dtype)?;
    let inner = (x + x3)?.broadcast_mul(&sqrt_t)?;
    let tanh = inner.tanh()?;
    let one = Tensor::from_vec(vec![1.0f32], (1,), x.device())?.to_dtype(dtype)?;
    let half = Tensor::from_vec(vec![0.5f32], (1,), x.device())?.to_dtype(dtype)?;
    let out = x.broadcast_mul(&one.broadcast_add(&tanh)?)?;
    let out = out.broadcast_mul(&half)?;
    Ok(out)
}
