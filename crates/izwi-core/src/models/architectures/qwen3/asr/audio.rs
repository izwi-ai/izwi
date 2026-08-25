//! Audio tower shared by the retained Qwen speech/aligner stack.

use candle_core::{IndexOp, Module, Tensor, D};
use candle_nn::ops;
use candle_nn::{layer_norm, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};

use crate::error::{Error, Result};
use crate::models::architectures::qwen3::asr::config::AudioConfig;
use crate::models::shared::attention::flash::{
    try_fused_self_attention, try_fused_varlen_self_attention,
};
use crate::models::shared::telemetry::{
    record_chunk_attention_fused_span, record_chunk_attention_mask_fallback,
    record_chunk_attention_sequence, record_chunk_attention_unfused_span,
};
use crate::models::shared::weights::mlx;

/// Compute output length after feature extraction/downsampling.
/// Matches upstream Qwen speech-family `_get_feat_extract_output_lengths`.
pub fn get_cnn_output_lengths(input_lengths: &[usize]) -> Vec<usize> {
    input_lengths
        .iter()
        .map(|&len| {
            let input_lengths_leave = len % 100;
            conv_stack_output_len(input_lengths_leave) + (len / 100) * 13
        })
        .collect()
}

/// Compute output length after a single conv2d with stride=2, kernel=3, padding=1.
fn conv_output_len(input_len: usize) -> usize {
    if input_len == 0 {
        0
    } else {
        (input_len - 1) / 2 + 1
    }
}

fn conv_stack_output_len(input_len: usize) -> usize {
    conv_output_len(conv_output_len(conv_output_len(input_len)))
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

fn chunk_spans_from_cu_seqlens(seq_len: usize, cu_seqlens: &[i64]) -> Option<Vec<(usize, usize)>> {
    if seq_len == 0 {
        return Some(Vec::new());
    }
    if cu_seqlens.len() < 2 || *cu_seqlens.first()? != 0 {
        return None;
    }

    let mut spans = Vec::with_capacity(cu_seqlens.len() - 1);
    let mut prev = 0usize;
    for &raw_end in cu_seqlens.iter().skip(1) {
        let end = usize::try_from(raw_end).ok()?;
        if end > seq_len || end < prev {
            return None;
        }
        if end > prev {
            spans.push((prev, end));
        }
        prev = end;
    }

    if prev != seq_len || spans.is_empty() {
        return None;
    }

    Some(spans)
}

fn chunk_cu_seqlens_u32(seq_len: usize, spans: &[(usize, usize)]) -> Option<Vec<u32>> {
    if spans.is_empty() {
        return None;
    }

    let mut out = Vec::with_capacity(spans.len() + 1);
    let mut expected_start = 0usize;
    out.push(0);
    for &(start, end) in spans {
        if start != expected_start || end <= start || end > seq_len {
            return None;
        }
        out.push(u32::try_from(end).ok()?);
        expected_start = end;
    }

    if expected_start != seq_len {
        return None;
    }
    Some(out)
}

fn attention_unfused_with_mask(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    num_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let seq_len = q.dim(2)?;
    let q = q.reshape((num_heads, seq_len, head_dim))?;
    let k = k.reshape((num_heads, seq_len, head_dim))?;
    let v = v.reshape((num_heads, seq_len, head_dim))?;

    let mut attn = q.matmul(&k.transpose(1, 2)?)?;
    attn = (attn / (head_dim as f64).sqrt())?;
    attn = attn.broadcast_add(&mask.unsqueeze(0)?)?;

    let attn = ops::softmax(&attn, D::Minus1)?;
    let out = attn.matmul(&v)?;
    out.reshape((1, num_heads, seq_len, head_dim))
        .map_err(Error::from)
}

fn attention_no_mask(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    num_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    if let Ok(out) = ops::sdpa(q, k, v, None, false, scale, 1.0) {
        return Ok(out);
    }

    let seq_len = q.dim(2)?;
    let q = q.reshape((num_heads, seq_len, head_dim))?;
    let k = k.reshape((num_heads, seq_len, head_dim))?;
    let v = v.reshape((num_heads, seq_len, head_dim))?;

    let mut attn = q.matmul(&k.transpose(1, 2)?)?;
    attn = (attn / (head_dim as f64).sqrt())?;
    let attn = ops::softmax(&attn, D::Minus1)?;
    let out = attn.matmul(&v)?;
    out.reshape((1, num_heads, seq_len, head_dim))
        .map_err(Error::from)
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

        // Fast path: execute attention independently per chunk. This avoids building
        // a full block mask and unlocks mask-free fused kernels for each span.
        if let Some(spans) = chunk_spans_from_cu_seqlens(seq_len, cu_seqlens) {
            record_chunk_attention_sequence(spans.len(), seq_len);
            if spans.len() > 1 && q.device().is_cuda() {
                let max_span = spans
                    .iter()
                    .map(|(start, end)| end - start)
                    .max()
                    .unwrap_or(0);
                if let Some(cu_seqlens_u32) = chunk_cu_seqlens_u32(seq_len, &spans) {
                    if let Some(fused) = try_fused_varlen_self_attention(
                        &q,
                        &k,
                        &v,
                        &cu_seqlens_u32,
                        max_span,
                        self.head_dim,
                        false,
                    )? {
                        for _ in &spans {
                            record_chunk_attention_fused_span();
                        }
                        let out = fused.transpose(1, 2)?.reshape((
                            1,
                            seq_len,
                            self.num_heads * self.head_dim,
                        ))?;
                        return self.out_proj.forward(&out).map_err(Error::from);
                    }
                }
            }

            if spans.len() == 1 {
                let (start, end) = spans[0];
                let span = end - start;
                let (q_chunk, k_chunk, v_chunk) = if start == 0 && span == seq_len {
                    (q.contiguous()?, k.contiguous()?, v.contiguous()?)
                } else {
                    (
                        q.narrow(2, start, span)?.contiguous()?,
                        k.narrow(2, start, span)?.contiguous()?,
                        v.narrow(2, start, span)?.contiguous()?,
                    )
                };

                let out = if let Some(fused) = try_fused_self_attention(
                    &q_chunk,
                    &k_chunk,
                    &v_chunk,
                    None,
                    self.head_dim,
                    false,
                )? {
                    record_chunk_attention_fused_span();
                    fused
                } else {
                    record_chunk_attention_unfused_span();
                    attention_no_mask(&q_chunk, &k_chunk, &v_chunk, self.num_heads, self.head_dim)?
                };

                let out =
                    out.transpose(1, 2)?
                        .reshape((1, seq_len, self.num_heads * self.head_dim))?;
                return self.out_proj.forward(&out).map_err(Error::from);
            }

            let mut outputs = Vec::with_capacity(spans.len());
            for (start, end) in spans {
                let span = end - start;
                let q_chunk = q.narrow(2, start, span)?.contiguous()?;
                let k_chunk = k.narrow(2, start, span)?.contiguous()?;
                let v_chunk = v.narrow(2, start, span)?.contiguous()?;

                let out = if let Some(fused) = try_fused_self_attention(
                    &q_chunk,
                    &k_chunk,
                    &v_chunk,
                    None,
                    self.head_dim,
                    false,
                )? {
                    record_chunk_attention_fused_span();
                    fused
                } else {
                    record_chunk_attention_unfused_span();
                    attention_no_mask(&q_chunk, &k_chunk, &v_chunk, self.num_heads, self.head_dim)?
                };
                outputs.push(out);
            }

            let refs: Vec<&Tensor> = outputs.iter().collect();
            let out = Tensor::cat(&refs, 2)?.transpose(1, 2)?.reshape((
                1,
                seq_len,
                self.num_heads * self.head_dim,
            ))?;
            return self.out_proj.forward(&out).map_err(Error::from);
        }

        // Fallback: retain original masked full-sequence behavior if chunk metadata
        // is malformed or incomplete.
        record_chunk_attention_mask_fallback();
        let mask = create_chunked_attention_mask(seq_len, cu_seqlens, x.device(), q.dtype())?;
        let out = attention_unfused_with_mask(&q, &k, &v, &mask, self.num_heads, self.head_dim)?
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

/// Padded audio-tower output with exact logical lengths for every input row.
///
/// Padding is never visible to the text decoder: callers must extract rows
/// through [`Self::row`], which returns the exact `[1, tokens, hidden]` span.
#[derive(Debug)]
pub struct AudioTowerBatchOutput {
    padded: Tensor,
    output_lengths: Vec<usize>,
}

impl AudioTowerBatchOutput {
    pub fn output_lengths(&self) -> &[usize] {
        &self.output_lengths
    }

    pub fn row(&self, index: usize) -> Result<Tensor> {
        let batch = self.padded.dim(0)?;
        let max_tokens = self.padded.dim(1)?;
        let length = *self.output_lengths.get(index).ok_or_else(|| {
            Error::InvalidInput(format!(
                "Qwen3 ASR audio batch row {index} is outside batch size {batch}"
            ))
        })?;
        if index >= batch || length > max_tokens {
            return Err(Error::InferenceError(format!(
                "Qwen3 ASR audio batch metadata is inconsistent for row {index}: length={length}, batch={batch}, padded_tokens={max_tokens}"
            )));
        }
        self.padded
            .i(index)?
            .narrow(0, 0, length)?
            .unsqueeze(0)
            .map_err(Error::from)
    }

    pub fn into_padded(self) -> Tensor {
        self.padded
    }
}

impl AudioTower {
    pub(super) fn preparation_dimensions(&self) -> (usize, usize, usize, usize, usize, usize) {
        (
            self.cfg.num_mel_bins,
            self.cfg.downsample_hidden_size,
            self.cfg.d_model,
            self.cfg.encoder_ffn_dim,
            self.cfg.encoder_attention_heads,
            self.cfg.output_dim,
        )
    }

    pub(super) fn preparation_chunk_geometry(&self) -> Result<(usize, usize)> {
        let chunk_input = self
            .cfg
            .n_window
            .unwrap_or(50)
            .checked_mul(2)
            .ok_or_else(|| Error::InvalidInput("Qwen3 ASR audio chunk size overflow".into()))?;
        if chunk_input == 0 {
            return Err(Error::InvalidInput("Invalid audio chunk size".into()));
        }
        let infer_window = self.cfg.n_window_infer.unwrap_or(800);
        let output_window = conv_stack_output_len(chunk_input)
            .checked_mul((infer_window / chunk_input).max(1))
            .ok_or_else(|| Error::InvalidInput("Audio attention window overflow".into()))?;
        Ok((chunk_input, output_window.max(1)))
    }

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
        let feature_lens = feature_lens
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| vec![total_frames; bsz]);
        let features = mel.narrow(1, 0, 1)?.squeeze(1)?.transpose(1, 2)?;
        Ok(self
            .forward_feature_batch(&features, &feature_lens)?
            .into_padded())
    }

    /// Return the exact number of encoder tokens produced for each feature
    /// length using this tower's configured CNN chunk geometry.
    pub fn output_lengths(&self, feature_lens: &[usize]) -> Result<Vec<usize>> {
        let chunk_input_len = self
            .cfg
            .n_window
            .unwrap_or(50)
            .checked_mul(2)
            .ok_or_else(|| Error::InvalidInput("Audio chunk size overflow".to_string()))?;
        if chunk_input_len == 0 {
            return Err(Error::InvalidInput("Invalid audio chunk size".to_string()));
        }
        feature_lens
            .iter()
            .map(|&length| {
                let full_chunks = length / chunk_input_len;
                let tail = length % chunk_input_len;
                full_chunks
                    .checked_mul(conv_stack_output_len(chunk_input_len))
                    .and_then(|full| full.checked_add(conv_stack_output_len(tail)))
                    .ok_or_else(|| {
                        Error::InvalidInput(
                            "Qwen3 ASR audio output sequence length overflow".to_string(),
                        )
                    })
            })
            .collect()
    }

    /// Encode a padded, frame-major feature batch in one native tower pass.
    ///
    /// `features` is `[batch, max_frames, n_mels]`; `feature_lens` is the
    /// exact logical frame count for each row. All rows are packed before the
    /// convolution and transformer stacks. The attention `cu_seqlens` acts as
    /// a ragged mask, so neither padding nor a neighbouring row can influence
    /// a row's output. Batch size one intentionally retains the established
    /// scalar path as the correctness fallback.
    pub fn forward_feature_batch(
        &self,
        features: &Tensor,
        feature_lens: &[usize],
    ) -> Result<AudioTowerBatchOutput> {
        let (batch, max_frames, n_mels) = features.dims3()?;
        if batch == 0 {
            return Err(Error::InvalidInput(
                "No audio features available for ASR batch".to_string(),
            ));
        }
        if feature_lens.len() != batch {
            return Err(Error::InvalidInput(format!(
                "Qwen3 ASR audio batch has {batch} rows but {} feature lengths",
                feature_lens.len()
            )));
        }
        if n_mels != self.cfg.num_mel_bins {
            return Err(Error::InvalidInput(format!(
                "Qwen3 ASR audio batch has {n_mels} mel bins, expected {}",
                self.cfg.num_mel_bins
            )));
        }
        for (row, &length) in feature_lens.iter().enumerate() {
            if length == 0 || length > max_frames {
                return Err(Error::InvalidInput(format!(
                    "Qwen3 ASR audio row {row} has invalid feature length {length}; padded extent is {max_frames}"
                )));
            }
        }

        if batch == 1 {
            let output = self.forward_single_feature_sequence(&features.i(0)?, feature_lens[0])?;
            let output_len = output.dim(1)?;
            return Ok(AudioTowerBatchOutput {
                padded: output,
                output_lengths: vec![output_len],
            });
        }

        self.forward_ragged_feature_batch(features, feature_lens)
    }

    pub fn forward_feature_sequence(&self, features: &Tensor, input_len: usize) -> Result<Tensor> {
        let (frames, _n_mels) = features.dims2()?;
        let input_len = input_len.min(frames);
        if input_len == 0 {
            return Err(crate::error::Error::InvalidInput(
                "Empty audio feature sequence".to_string(),
            ));
        }
        self.forward_single_feature_sequence(features, input_len)
    }

    fn forward_single_sample(&self, mel_sample: &Tensor, input_len: usize) -> Result<Tensor> {
        let feature_seq = mel_sample.transpose(0, 1)?; // [frames, n_mels]
        self.forward_single_feature_sequence(&feature_seq, input_len)
    }

    fn forward_single_feature_sequence(
        &self,
        feature_seq: &Tensor,
        input_len: usize,
    ) -> Result<Tensor> {
        let n_mels = feature_seq.dim(1)?;
        let n_window = self.cfg.n_window.unwrap_or(50);
        let n_window_infer = self.cfg.n_window_infer.unwrap_or(800);
        let chunk_input_len = n_window * 2;
        if chunk_input_len == 0 {
            return Err(crate::error::Error::InvalidInput(
                "Invalid audio chunk size".to_string(),
            ));
        }

        // Match upstream: split features into fixed-size chunks before CNN.
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
        let chunk_out_lens: Vec<usize> = chunk_lengths
            .iter()
            .copied()
            .map(conv_stack_output_len)
            .collect();
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
        let cnn_lengths = vec![packed_len];
        let max_chunk_after_cnn = conv_stack_output_len(chunk_input_len).max(1);
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

    fn forward_ragged_feature_batch(
        &self,
        features: &Tensor,
        feature_lens: &[usize],
    ) -> Result<AudioTowerBatchOutput> {
        let n_mels = features.dim(2)?;
        let n_window = self.cfg.n_window.unwrap_or(50);
        let n_window_infer = self.cfg.n_window_infer.unwrap_or(800);
        let chunk_input_len = n_window.checked_mul(2).ok_or_else(|| {
            Error::InvalidInput("Qwen3 ASR audio chunk size overflow".to_string())
        })?;
        if chunk_input_len == 0 {
            return Err(Error::InvalidInput("Invalid audio chunk size".to_string()));
        }

        // Validate every row before launching a kernel. The tower is stateless,
        // so a rejected batch leaves no partially prepared row to roll back.
        let mut all_chunk_lengths = Vec::new();
        let mut row_chunk_counts = Vec::with_capacity(feature_lens.len());
        for &input_len in feature_lens {
            let chunks = input_len.div_ceil(chunk_input_len);
            row_chunk_counts.push(chunks);
            let full_chunks = input_len / chunk_input_len;
            all_chunk_lengths.extend(std::iter::repeat_n(chunk_input_len, full_chunks));
            let tail = input_len % chunk_input_len;
            if tail > 0 {
                all_chunk_lengths.push(tail);
            }
        }

        let mut chunks = Vec::with_capacity(all_chunk_lengths.len());
        for (row, (&input_len, &chunk_count)) in
            feature_lens.iter().zip(row_chunk_counts.iter()).enumerate()
        {
            let feature_row = features.i(row)?;
            for chunk in 0..chunk_count {
                let offset = chunk * chunk_input_len;
                let length = (input_len - offset).min(chunk_input_len);
                let values = feature_row.narrow(0, offset, length)?;
                if length == chunk_input_len {
                    chunks.push(values);
                } else {
                    let pad = Tensor::zeros(
                        (chunk_input_len - length, n_mels),
                        values.dtype(),
                        values.device(),
                    )?;
                    chunks.push(Tensor::cat(&[values, pad], 0)?);
                }
            }
        }

        let chunk_refs: Vec<&Tensor> = chunks.iter().collect();
        let mut x = Tensor::stack(&chunk_refs, 0)?; // [all_chunks, chunk_input_len, n_mels]
        x = x.transpose(1, 2)?.unsqueeze(1)?;

        // One convolutional launch sequence serves every request row.
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
        x = x.transpose(1, 3)?.transpose(2, 3)?;
        x = x.reshape((num_chunks, frames, channels * freq))?;
        x = self.conv_out.forward(&x)?;

        // Positional coordinates reset at every CNN chunk, matching the
        // established scalar Qwen audio path exactly.
        let pos_emb = self.pos_embed.get(x.dim(1)?)?;
        let pos_emb = pos_emb.unsqueeze(0)?.to_dtype(x.dtype())?;
        x = x.broadcast_add(&pos_emb)?;

        let chunk_output_lengths: Vec<usize> = all_chunk_lengths
            .iter()
            .copied()
            .map(conv_stack_output_len)
            .collect();
        let mut packed_chunks = Vec::with_capacity(chunk_output_lengths.len());
        for (chunk, &length) in chunk_output_lengths.iter().enumerate() {
            let keep = length.min(frames);
            if keep > 0 {
                packed_chunks.push(x.i(chunk)?.narrow(0, 0, keep)?);
            }
        }
        let packed_refs: Vec<&Tensor> = packed_chunks.iter().collect();
        let mut x = Tensor::cat(&packed_refs, 0)?.unsqueeze(0)?;

        let mut row_output_lengths = Vec::with_capacity(feature_lens.len());
        let mut chunk_cursor = 0usize;
        for &chunk_count in &row_chunk_counts {
            let row_length = chunk_output_lengths[chunk_cursor..chunk_cursor + chunk_count]
                .iter()
                .copied()
                .sum();
            row_output_lengths.push(row_length);
            chunk_cursor += chunk_count;
        }
        let packed_len = x.dim(1)?;
        let planned_len: usize = row_output_lengths.iter().sum();
        if packed_len != planned_len {
            return Err(Error::InferenceError(format!(
                "Qwen3 ASR ragged audio packing mismatch: tensor={packed_len}, plan={planned_len}"
            )));
        }

        // Each row receives its own set of chunked-attention windows. The
        // cumulative boundaries are the ragged attention mask and prohibit
        // cross-request attention while retaining one packed encoder call.
        let max_chunk_after_cnn = conv_stack_output_len(chunk_input_len).max(1);
        let infer_ratio = (n_window_infer / chunk_input_len).max(1);
        let window_after_cnn = max_chunk_after_cnn
            .checked_mul(infer_ratio)
            .ok_or_else(|| Error::InvalidInput("Audio attention window overflow".to_string()))?;
        let mut cu_seqlens = vec![0i64];
        let mut cursor = 0usize;
        for &row_length in &row_output_lengths {
            let mut remaining = row_length;
            while remaining > 0 {
                let span = remaining.min(window_after_cnn);
                cursor = cursor.checked_add(span).ok_or_else(|| {
                    Error::InvalidInput("Audio attention sequence length overflow".to_string())
                })?;
                cu_seqlens.push(i64::try_from(cursor).map_err(|_| {
                    Error::InvalidInput("Audio attention sequence exceeds i64".to_string())
                })?);
                remaining -= span;
            }
        }

        for layer in &self.layers {
            x = layer.forward(&x, &cu_seqlens)?;
        }
        let x = self.ln_post.forward(&x)?;
        let x = self.proj1.forward(&x)?;
        let x = gelu(&x)?;
        let x = self.proj2.forward(&x)?;

        let max_output_len = row_output_lengths.iter().copied().max().unwrap_or(0);
        let hidden = x.dim(2)?;
        let mut padded_rows = Vec::with_capacity(row_output_lengths.len());
        let mut offset = 0usize;
        for &length in &row_output_lengths {
            let row = x.narrow(1, offset, length)?;
            offset += length;
            if length == max_output_len {
                padded_rows.push(row);
            } else {
                let pad = Tensor::zeros(
                    (1, max_output_len - length, hidden),
                    row.dtype(),
                    row.device(),
                )?;
                padded_rows.push(Tensor::cat(&[row, pad], 1)?);
            }
        }
        let row_refs: Vec<&Tensor> = padded_rows.iter().collect();
        let padded = Tensor::cat(&row_refs, 0)?;
        Ok(AudioTowerBatchOutput {
            padded,
            output_lengths: row_output_lengths,
        })
    }
}

fn gelu(x: &Tensor) -> Result<Tensor> {
    x.gelu().map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::shape::ShapeWithOneHole;
    use candle_core::{DType, Device, Tensor};

    fn deterministic_tensor(
        elements: usize,
        shape: impl ShapeWithOneHole,
        device: &Device,
        seed: usize,
    ) -> Tensor {
        let values = (0..elements)
            .map(|index| {
                let centered = ((index + seed * 7) % 29) as f32 - 14.0;
                centered * 0.0075
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, shape, device).expect("deterministic tensor")
    }

    fn tiny_linear(out_dim: usize, in_dim: usize, device: &Device, seed: usize) -> Linear {
        Linear::new(
            deterministic_tensor(out_dim * in_dim, (out_dim, in_dim), device, seed),
            Some(
                deterministic_tensor(out_dim, out_dim, device, seed + 1)
                    .reshape(out_dim)
                    .expect("linear bias"),
            ),
        )
    }

    fn tiny_layer_norm(dim: usize, device: &Device) -> LayerNorm {
        LayerNorm::new(
            Tensor::ones(dim, DType::F32, device).expect("norm weight"),
            Tensor::zeros(dim, DType::F32, device).expect("norm bias"),
            1e-5,
        )
    }

    fn tiny_audio_tower(device: &Device) -> AudioTower {
        let cfg = AudioConfig {
            d_model: 4,
            encoder_attention_heads: 1,
            encoder_ffn_dim: 8,
            encoder_layers: 1,
            num_mel_bins: 8,
            downsample_hidden_size: 2,
            output_dim: 4,
            conv_chunksize: None,
            n_window: Some(50),
            n_window_infer: Some(200),
        };
        let conv_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let conv = |out_channels: usize, in_channels: usize, seed: usize| {
            Conv2d::new(
                deterministic_tensor(
                    out_channels * in_channels * 3 * 3,
                    (out_channels, in_channels, 3, 3),
                    device,
                    seed,
                ),
                Some(deterministic_tensor(
                    out_channels,
                    out_channels,
                    device,
                    seed + 1,
                )),
                conv_cfg,
            )
        };
        let attention = AudioAttention {
            q_proj: tiny_linear(4, 4, device, 10),
            k_proj: tiny_linear(4, 4, device, 12),
            v_proj: tiny_linear(4, 4, device, 14),
            out_proj: tiny_linear(4, 4, device, 16),
            num_heads: 1,
            head_dim: 4,
        };
        let layer = AudioEncoderLayer {
            self_attn_layer_norm: tiny_layer_norm(4, device),
            self_attn: attention,
            final_layer_norm: tiny_layer_norm(4, device),
            fc1: tiny_linear(8, 4, device, 18),
            fc2: tiny_linear(4, 8, device, 20),
        };
        AudioTower {
            conv2d1: conv(2, 1, 1),
            conv2d2: conv(2, 2, 3),
            conv2d3: conv(2, 2, 5),
            conv_out: Linear::new(deterministic_tensor(8, (4, 2), device, 7), None),
            layers: vec![layer],
            ln_post: tiny_layer_norm(4, device),
            proj1: tiny_linear(4, 4, device, 22),
            proj2: tiny_linear(4, 4, device, 24),
            pos_embed: SinusoidalPositionEmbedding::new(1500, 4, device)
                .expect("position embedding"),
            cfg,
        }
    }

    fn ragged_features(lengths: &[usize], n_mels: usize, device: &Device) -> Tensor {
        let max_frames = lengths.iter().copied().max().expect("feature rows");
        let mut values = vec![0.0f32; lengths.len() * max_frames * n_mels];
        for (row, &length) in lengths.iter().enumerate() {
            for frame in 0..length {
                for mel in 0..n_mels {
                    let index = row * max_frames * n_mels + frame * n_mels + mel;
                    values[index] =
                        (((row + 1) * 31 + frame * 7 + mel * 3) % 41) as f32 * 0.01 - 0.2;
                }
            }
        }
        Tensor::from_vec(values, (lengths.len(), max_frames, n_mels), device)
            .expect("ragged features")
    }

    fn assert_close(lhs: &Tensor, rhs: &Tensor) {
        assert_eq!(lhs.dims(), rhs.dims());
        let lhs = lhs
            .flatten_all()
            .expect("lhs flat")
            .to_vec1::<f32>()
            .expect("lhs values");
        let rhs = rhs
            .flatten_all()
            .expect("rhs flat")
            .to_vec1::<f32>()
            .expect("rhs values");
        for (index, (lhs, rhs)) in lhs.iter().zip(rhs.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() <= 2e-5,
                "tensor mismatch at {index}: {lhs} != {rhs}"
            );
        }
    }

    #[test]
    fn cnn_geometry_is_exact_at_upstream_chunk_boundaries() {
        assert_eq!(
            get_cnn_output_lengths(&[0, 1, 8, 99, 100, 101, 199, 200, 201]),
            vec![0, 1, 1, 13, 13, 14, 26, 26, 27]
        );
    }

    #[test]
    fn ragged_audio_batch_matches_scalar_across_chunk_boundaries() {
        let device = Device::Cpu;
        let tower = tiny_audio_tower(&device);
        let lengths = [1usize, 8, 99, 100, 101, 199, 200, 201];
        let features = ragged_features(&lengths, 8, &device);
        let batched = tower
            .forward_feature_batch(&features, &lengths)
            .expect("ragged batch");
        assert_eq!(
            batched.output_lengths(),
            tower.output_lengths(&lengths).expect("output geometry")
        );

        for (row, &length) in lengths.iter().enumerate() {
            let scalar = tower
                .forward_feature_sequence(&features.i(row).expect("feature row"), length)
                .expect("scalar audio tower");
            let exact = batched.row(row).expect("exact batch row");
            assert_close(&exact, &scalar);
        }
    }

    #[test]
    fn one_row_audio_batch_uses_scalar_equivalent_path() {
        let device = Device::Cpu;
        let tower = tiny_audio_tower(&device);
        let lengths = [101usize];
        let features = ragged_features(&lengths, 8, &device);
        let scalar = tower
            .forward_feature_sequence(&features.i(0).expect("feature row"), lengths[0])
            .expect("scalar output");
        let batched = tower
            .forward_feature_batch(&features, &lengths)
            .expect("single-row batch")
            .row(0)
            .expect("single exact row");
        assert_close(&batched, &scalar);
    }

    #[test]
    fn ragged_audio_batch_rejection_is_atomic_and_retryable() {
        let device = Device::Cpu;
        let tower = tiny_audio_tower(&device);
        let lengths = [99usize, 100, 101];
        let features = ragged_features(&lengths, 8, &device);
        let before = tower
            .forward_feature_batch(&features, &lengths)
            .expect("baseline batch");

        let error = tower
            .forward_feature_batch(&features, &[99, 0, 101])
            .expect_err("zero-length row must reject before execution");
        assert!(error.to_string().contains("row 1"));
        let error = tower
            .forward_feature_batch(&features, &[99, 100, 102])
            .expect_err("out-of-bounds row must reject before execution");
        assert!(error.to_string().contains("row 2"));

        let after = tower
            .forward_feature_batch(&features, &lengths)
            .expect("retry batch");
        for row in 0..lengths.len() {
            assert_close(
                &before.row(row).expect("baseline row"),
                &after.row(row).expect("retry row"),
            );
        }
    }

    #[test]
    fn chunk_spans_parse_valid_cu_seqlens() {
        let spans = chunk_spans_from_cu_seqlens(11, &[0, 3, 7, 11]).expect("spans");
        assert_eq!(spans, vec![(0, 3), (3, 7), (7, 11)]);
    }

    #[test]
    fn chunk_spans_reject_malformed_cu_seqlens() {
        assert!(chunk_spans_from_cu_seqlens(8, &[1, 8]).is_none());
        assert!(chunk_spans_from_cu_seqlens(8, &[0, 4, 3, 8]).is_none());
        assert!(chunk_spans_from_cu_seqlens(8, &[0, 4, 7]).is_none());
    }

    #[test]
    fn chunk_cu_seqlens_u32_requires_contiguous_spans() {
        let spans = vec![(0, 3), (3, 7), (7, 11)];
        assert_eq!(chunk_cu_seqlens_u32(11, &spans), Some(vec![0, 3, 7, 11]));

        assert!(chunk_cu_seqlens_u32(11, &[(0, 3), (4, 11)]).is_none());
        assert!(chunk_cu_seqlens_u32(11, &[(0, 3), (3, 12)]).is_none());
        assert!(chunk_cu_seqlens_u32(11, &[]).is_none());
    }

    #[test]
    fn chunkwise_attention_matches_block_mask_attention() {
        let device = Device::Cpu;
        let num_heads = 2usize;
        let seq_len = 5usize;
        let head_dim = 4usize;
        let dtype = DType::F32;

        let q = Tensor::from_vec(
            (0..(num_heads * seq_len * head_dim))
                .map(|v| (v as f32) * 0.01)
                .collect::<Vec<_>>(),
            (1, num_heads, seq_len, head_dim),
            &device,
        )
        .expect("q");
        let k = Tensor::from_vec(
            (0..(num_heads * seq_len * head_dim))
                .map(|v| (v as f32) * 0.013 + 0.2)
                .collect::<Vec<_>>(),
            (1, num_heads, seq_len, head_dim),
            &device,
        )
        .expect("k");
        let v = Tensor::from_vec(
            (0..(num_heads * seq_len * head_dim))
                .map(|v| (v as f32) * 0.017 - 0.1)
                .collect::<Vec<_>>(),
            (1, num_heads, seq_len, head_dim),
            &device,
        )
        .expect("v");
        let cu = vec![0i64, 2, 5];

        let mask = create_chunked_attention_mask(seq_len, &cu, &device, dtype).expect("chunk mask");
        let masked = attention_unfused_with_mask(&q, &k, &v, &mask, num_heads, head_dim)
            .expect("masked attention");

        let spans = chunk_spans_from_cu_seqlens(seq_len, &cu).expect("spans");
        let mut outputs = Vec::with_capacity(spans.len());
        for (start, end) in spans {
            let span = end - start;
            outputs.push(
                attention_no_mask(
                    &q.narrow(2, start, span).expect("q span"),
                    &k.narrow(2, start, span).expect("k span"),
                    &v.narrow(2, start, span).expect("v span"),
                    num_heads,
                    head_dim,
                )
                .expect("chunk attention"),
            );
        }
        let refs: Vec<&Tensor> = outputs.iter().collect();
        let chunked = Tensor::cat(&refs, 2).expect("cat");

        let masked_vals = masked
            .flatten_all()
            .expect("flatten masked")
            .to_vec1::<f32>()
            .expect("masked vals");
        let chunked_vals = chunked
            .flatten_all()
            .expect("flatten chunked")
            .to_vec1::<f32>()
            .expect("chunked vals");
        assert_eq!(masked_vals.len(), chunked_vals.len());
        for (lhs, rhs) in masked_vals.iter().zip(chunked_vals.iter()) {
            assert!((lhs - rhs).abs() < 1e-5);
        }
    }
}
