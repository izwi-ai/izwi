//! Voxtral neural codec decoder.

use std::ops::Range;

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{
    ops, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Embedding, Linear, Module,
    RmsNorm, VarBuilder,
};

use crate::error::{Error, Result};
use crate::models::architectures::qwen3::core::repeat_kv;

use super::super::layers::linear_forward_last_dim;
use super::acoustic::{strip_audio_token_offset, AudioCodeValue};
use super::config::{VoxtralTtsAudioTokenizerArgs, VoxtralTtsConfig};

pub const VOXTRAL_CODEC_CHUNK_FRAMES: usize = 375;

#[derive(Debug, Clone, PartialEq)]
pub struct VoxtralCodecConfig {
    pub channels: usize,
    pub sample_rate: usize,
    pub frame_rate: f32,
    pub patch_size: usize,
    pub decoder_strides: Vec<usize>,
    pub semantic_dim: usize,
    pub acoustic_dim: usize,
    pub latent_dim: usize,
}

impl VoxtralCodecConfig {
    pub fn from_config(config: &VoxtralTtsConfig) -> Result<Self> {
        let tokenizer = &config.multimodal.audio_tokenizer_args;
        Ok(Self {
            channels: tokenizer.channels,
            sample_rate: tokenizer.sampling_rate,
            frame_rate: config.frame_rate(),
            patch_size: tokenizer.pretransform_patch_size,
            decoder_strides: config.decoder_conv_strides()?,
            semantic_dim: tokenizer.semantic_dim,
            acoustic_dim: tokenizer.acoustic_dim,
            latent_dim: tokenizer.semantic_dim + tokenizer.acoustic_dim,
        })
    }

    pub fn downsample_factor(&self) -> Result<usize> {
        let stride_product = self
            .decoder_strides
            .iter()
            .try_fold(1usize, |acc, stride| {
                acc.checked_mul(*stride).ok_or_else(|| {
                    Error::ConfigError("Voxtral codec stride product overflowed".to_string())
                })
            })?;
        self.patch_size.checked_mul(stride_product).ok_or_else(|| {
            Error::ConfigError("Voxtral codec downsample factor overflowed".to_string())
        })
    }

    pub fn samples_for_frames(&self, frames: usize) -> Result<usize> {
        self.downsample_factor()?
            .checked_mul(frames)
            .ok_or_else(|| Error::AudioError("Voxtral codec sample count overflowed".to_string()))
    }

    pub fn frame_count_for_samples_ceil(&self, samples: usize) -> Result<usize> {
        let factor = self.downsample_factor()?;
        Ok((samples + factor - 1) / factor)
    }

    pub fn chunk_ranges(&self, frames: usize) -> Vec<Range<usize>> {
        chunk_ranges(frames, VOXTRAL_CODEC_CHUNK_FRAMES)
    }
}

pub struct VoxtralCodecDecoder {
    quantizer: VoxtralCodecQuantizer,
    decoder_blocks: Vec<VoxtralDecoderBlock>,
    output_proj: VoxtralCausalConv1d,
    config: VoxtralCodecConfig,
}

struct VoxtralCodecQuantizer {
    semantic_codebook: Embedding,
    semantic_dim: usize,
    acoustic_dim: usize,
    acoustic_codebook_size: usize,
    num_codebooks: usize,
    dtype: DType,
}

enum VoxtralDecoderBlock {
    Conv(VoxtralCausalConv1d),
    TransposeConv(VoxtralCausalConvTranspose1d),
    Transformer(VoxtralCodecTransformer),
}

struct VoxtralCausalConv1d {
    conv: Conv1d,
    pad_mode: VoxtralPadMode,
    effective_kernel: usize,
    stride: usize,
    padding_total: usize,
}

struct VoxtralCausalConvTranspose1d {
    conv: ConvTranspose1d,
    left_trim: usize,
    right_trim: usize,
}

struct VoxtralCodecTransformer {
    layers: Vec<VoxtralCodecTransformerBlock>,
}

struct VoxtralCodecTransformerBlock {
    attention: VoxtralCodecAttention,
    feed_forward: VoxtralCodecFeedForward,
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    attention_scale: Option<Tensor>,
    ffn_scale: Option<Tensor>,
}

struct VoxtralCodecAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    sliding_window: usize,
    causal: bool,
    alibi_slopes: Vec<f32>,
}

struct VoxtralCodecFeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VoxtralPadMode {
    Constant,
    Replicate,
}

impl VoxtralCodecDecoder {
    pub fn load(config: &VoxtralTtsConfig, vb: VarBuilder) -> Result<Self> {
        let codec_config = VoxtralCodecConfig::from_config(config)?;
        let args = &config.multimodal.audio_tokenizer_args;
        if args.channels != 1 {
            return Err(Error::ConfigError(format!(
                "Voxtral codec decoder only supports mono output, got {} channels",
                args.channels
            )));
        }

        let kernels = config.decoder_conv_kernels()?;
        let strides = config.decoder_conv_strides()?;
        let transformer_lengths = config.decoder_transformer_lengths()?;
        if kernels.len() != strides.len() || kernels.len() != transformer_lengths.len() {
            return Err(Error::ConfigError(format!(
                "Voxtral codec decoder config length mismatch: kernels={}, strides={}, transformers={}",
                kernels.len(),
                strides.len(),
                transformer_lengths.len()
            )));
        }
        if kernels.is_empty() {
            return Err(Error::ConfigError(
                "Voxtral codec decoder requires at least one decoder stage".to_string(),
            ));
        }

        let quantizer = VoxtralCodecQuantizer::load(config, vb.pp("quantizer"))?;
        let input_proj = VoxtralCausalConv1d::load(
            codec_config.latent_dim,
            args.dim,
            kernels[0],
            strides[0],
            args.conv_weight_norm,
            false,
            VoxtralPadMode::Replicate,
            vb.pp("decoder_blocks.0"),
        )?;

        let mut decoder_blocks = Vec::new();
        decoder_blocks.push(VoxtralDecoderBlock::Conv(input_proj));
        let mut block_idx = 1usize;
        let mut window_size = decoder_initial_window_size(args, &strides)?;
        for (stage_idx, n_layers) in transformer_lengths.iter().copied().enumerate() {
            let transformer = VoxtralCodecTransformer::load(
                args,
                n_layers,
                window_size,
                vb.pp(format!("decoder_blocks.{block_idx}")),
            )?;
            decoder_blocks.push(VoxtralDecoderBlock::Transformer(transformer));
            block_idx += 1;

            if stage_idx + 1 != transformer_lengths.len()
                && (kernels[stage_idx + 1] != 1 || strides[stage_idx + 1] != 1)
            {
                let upsample = VoxtralCausalConvTranspose1d::load(
                    args.dim,
                    args.dim,
                    kernels[stage_idx + 1],
                    strides[stage_idx + 1],
                    args.conv_weight_norm,
                    false,
                    vb.pp(format!("decoder_blocks.{block_idx}")),
                )?;
                decoder_blocks.push(VoxtralDecoderBlock::TransposeConv(upsample));
                if args.half_attn_window_upon_downsampling && strides[stage_idx + 1] > 1 {
                    window_size =
                        window_size
                            .checked_mul(strides[stage_idx + 1])
                            .ok_or_else(|| {
                                Error::ConfigError(
                                    "Voxtral codec attention window overflowed".to_string(),
                                )
                            })?;
                }
                block_idx += 1;
            }
        }

        let output_channels = args
            .pretransform_patch_size
            .checked_mul(args.channels)
            .ok_or_else(|| {
                Error::ConfigError("Voxtral codec output projection size overflowed".to_string())
            })?;
        let output_proj = VoxtralCausalConv1d::load(
            args.dim,
            output_channels,
            args.patch_proj_kernel_size,
            1,
            args.conv_weight_norm,
            false,
            VoxtralPadMode::Replicate,
            vb.pp("output_proj"),
        )?;

        Ok(Self {
            quantizer,
            decoder_blocks,
            output_proj,
            config: codec_config,
        })
    }

    pub fn decode_timeline(&self, timeline: &VoxtralCodecTimeline) -> Result<Vec<f32>> {
        let trimmed = timeline.trim_at_end_audio();
        let frames = trimmed.generated_frame_count();
        if frames == 0 {
            return Ok(Vec::new());
        }
        if trimmed.codebook_count() != self.quantizer.num_codebooks {
            return Err(Error::AudioError(format!(
                "Voxtral codec expected {} codebooks, got {}",
                self.quantizer.num_codebooks,
                trimmed.codebook_count()
            )));
        }

        let mut samples = Vec::new();
        for range in self.config.chunk_ranges(frames) {
            let chunk_frames = range.end - range.start;
            let codes = trimmed.to_unshifted_tensor_range(
                range,
                self.quantizer.semantic_codebook.embeddings().device(),
            )?;
            let audio = self.decode_unshifted_codes(&codes)?;
            let mut chunk = audio
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            trim_samples_to_frames(&mut chunk, chunk_frames, &self.config)?;
            samples.extend(chunk);
        }
        Ok(samples)
    }

    pub fn decode_unshifted_codes(&self, codes: &Tensor) -> Result<Tensor> {
        let emb = self
            .quantizer
            .decode(codes, self.quantizer.dtype())?
            .to_dtype(self.quantizer.dtype())?;
        self.forward_decoder(&emb)
    }

    fn forward_decoder(&self, emb: &Tensor) -> Result<Tensor> {
        let mut hidden = emb.transpose(1, 2)?;
        for block in &self.decoder_blocks {
            match block {
                VoxtralDecoderBlock::Conv(conv) => {
                    hidden = conv.forward(&hidden.transpose(1, 2)?)?.transpose(1, 2)?;
                }
                VoxtralDecoderBlock::TransposeConv(conv) => {
                    hidden = conv.forward(&hidden.transpose(1, 2)?)?.transpose(1, 2)?;
                }
                VoxtralDecoderBlock::Transformer(transformer) => {
                    hidden = transformer.forward(&hidden)?;
                }
            }
        }
        let out = self.output_proj.forward(&hidden.transpose(1, 2)?)?;
        let batch = out.dim(0)?;
        let frames = out.dim(2)?;
        let channels = self.config.channels;
        let patch = self.config.patch_size;
        out.reshape((batch, channels, patch, frames))?
            .transpose(2, 3)?
            .flatten(2, 3)
            .map_err(Error::from)
    }
}

impl VoxtralCodecQuantizer {
    fn load(config: &VoxtralTtsConfig, vb: VarBuilder) -> Result<Self> {
        let tokenizer = &config.multimodal.audio_tokenizer_args;
        let cluster_usage = vb.get(
            (tokenizer.semantic_codebook_size,),
            "semantic_codebook.cluster_usage",
        )?;
        let embedding_sum = vb.get(
            (tokenizer.semantic_codebook_size, tokenizer.semantic_dim),
            "semantic_codebook.embedding_sum",
        )?;
        let dtype = embedding_sum.dtype();
        let usage = cluster_usage.clamp(1e-5, f32::INFINITY)?.unsqueeze(1)?;
        let semantic_embedding = embedding_sum.broadcast_div(&usage)?;
        Ok(Self {
            semantic_codebook: Embedding::new(semantic_embedding, tokenizer.semantic_dim),
            semantic_dim: tokenizer.semantic_dim,
            acoustic_dim: tokenizer.acoustic_dim,
            acoustic_codebook_size: tokenizer.acoustic_codebook_size,
            num_codebooks: 1 + tokenizer.acoustic_dim,
            dtype,
        })
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn decode(&self, codes: &Tensor, dtype: DType) -> Result<Tensor> {
        if codes.rank() != 3 {
            return Err(Error::AudioError(format!(
                "Voxtral codec codes must have shape [B, K, T], got rank {}",
                codes.rank()
            )));
        }
        if codes.dim(1)? != self.num_codebooks {
            return Err(Error::AudioError(format!(
                "Voxtral codec codes have {} codebooks, expected {}",
                codes.dim(1)?,
                self.num_codebooks
            )));
        }

        let semantic_codes = codes.narrow(1, 0, 1)?.squeeze(1)?;
        let semantic = self
            .semantic_codebook
            .forward(&semantic_codes)?
            .transpose(1, 2)?
            .to_dtype(dtype)?;
        let acoustic_codes = codes
            .narrow(1, 1, self.acoustic_dim)?
            .to_dtype(DType::F32)?;
        let acoustic = acoustic_codes
            .broadcast_mul(&Tensor::new(
                2.0f32 / (self.acoustic_codebook_size as f32 - 1.0),
                codes.device(),
            )?)?
            .broadcast_sub(&Tensor::new(1.0f32, codes.device())?)?
            .to_dtype(dtype)?;
        if semantic.dim(1)? != self.semantic_dim {
            return Err(Error::AudioError(format!(
                "Voxtral semantic latent has {} channels, expected {}",
                semantic.dim(1)?,
                self.semantic_dim
            )));
        }
        Tensor::cat(&[semantic, acoustic], 1).map_err(Error::from)
    }
}

impl VoxtralCausalConv1d {
    #[allow(clippy::too_many_arguments)]
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        use_weight_norm: bool,
        use_bias: bool,
        pad_mode: VoxtralPadMode,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride,
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let conv = load_conv1d(
            in_channels,
            out_channels,
            kernel_size,
            use_weight_norm,
            use_bias,
            cfg,
            vb.pp("conv"),
        )?;
        let effective_kernel = kernel_size;
        let padding_total = effective_kernel.saturating_sub(stride);
        Ok(Self {
            conv,
            pad_mode,
            effective_kernel,
            stride,
            padding_total,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input_len = x.dim(2)?;
        let extra = self.extra_padding(input_len);
        let x = pad_1d(x, self.padding_total, extra, self.pad_mode)?;
        self.conv.forward(&x).map_err(Error::from)
    }

    fn extra_padding(&self, input_len: usize) -> usize {
        if input_len == 0 {
            return 0;
        }
        let n_frames = ((input_len as f64 - self.effective_kernel as f64
            + self.padding_total as f64)
            / self.stride as f64
            + 1.0)
            .ceil()
            .max(1.0);
        let target_length = ((n_frames as usize - 1) * self.stride)
            + self.effective_kernel.saturating_sub(self.padding_total);
        target_length.saturating_sub(input_len)
    }
}

impl VoxtralCausalConvTranspose1d {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        use_weight_norm: bool,
        use_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride,
            dilation: 1,
            groups: 1,
        };
        let conv = load_conv_transpose1d(
            in_channels,
            out_channels,
            kernel_size,
            use_weight_norm,
            use_bias,
            cfg,
            vb.pp("conv"),
        )?;
        let total_padding = kernel_size.saturating_sub(stride);
        Ok(Self {
            conv,
            left_trim: 0,
            right_trim: total_padding,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.conv.forward(x)?;
        let out_len = out.dim(2)?;
        let keep = out_len.saturating_sub(self.left_trim + self.right_trim);
        out.narrow(2, self.left_trim, keep).map_err(Error::from)
    }
}

impl VoxtralCodecTransformer {
    fn load(
        args: &VoxtralTtsAudioTokenizerArgs,
        n_layers: usize,
        window_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(n_layers);
        for layer_idx in 0..n_layers {
            layers.push(VoxtralCodecTransformerBlock::load(
                args,
                layer_idx,
                window_size,
                vb.pp(format!("layers.{layer_idx}")),
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut hidden = x.clone();
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        Ok(hidden)
    }
}

impl VoxtralCodecTransformerBlock {
    fn load(
        args: &VoxtralTtsAudioTokenizerArgs,
        layer_idx: usize,
        window_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let attention =
            VoxtralCodecAttention::load(args, layer_idx, window_size, vb.pp("attention"))?;
        let feed_forward = VoxtralCodecFeedForward::load(args, vb.pp("feed_forward"))?;
        let attention_norm = candle_nn::rms_norm(args.dim, args.norm_eps, vb.pp("attention_norm"))?;
        let ffn_norm = candle_nn::rms_norm(args.dim, args.norm_eps, vb.pp("ffn_norm"))?;
        let attention_scale = load_layer_scale(args, vb.clone(), "attention_scale", layer_idx)?;
        let ffn_scale = load_layer_scale(args, vb, "ffn_scale", layer_idx)?;
        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
            attention_scale,
            ffn_scale,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut attn = self.attention.forward(&self.attention_norm.forward(x)?)?;
        if let Some(scale) = &self.attention_scale {
            attn = attn.broadcast_mul(scale)?;
        }
        let hidden = x.broadcast_add(&attn)?;
        let mut ff = self
            .feed_forward
            .forward(&self.ffn_norm.forward(&hidden)?)?;
        if let Some(scale) = &self.ffn_scale {
            ff = ff.broadcast_mul(scale)?;
        }
        hidden.broadcast_add(&ff).map_err(Error::from)
    }
}

impl VoxtralCodecAttention {
    fn load(
        args: &VoxtralTtsAudioTokenizerArgs,
        _layer_idx: usize,
        window_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let q_dim = args.n_heads * args.head_dim;
        let kv_dim = args.n_kv_heads * args.head_dim;
        let wq = candle_nn::linear_no_bias(args.dim, q_dim, vb.pp("wq"))?;
        let wk = candle_nn::linear_no_bias(args.dim, kv_dim, vb.pp("wk"))?;
        let wv = candle_nn::linear_no_bias(args.dim, kv_dim, vb.pp("wv"))?;
        let wo = linear_maybe_bias(q_dim, args.dim, args.use_biases, vb.pp("wo"))?;
        let q_norm = if args.qk_norm {
            Some(candle_nn::rms_norm(
                q_dim,
                args.qk_norm_eps,
                vb.pp("q_norm"),
            )?)
        } else {
            None
        };
        let k_norm = if args.qk_norm {
            Some(candle_nn::rms_norm(
                kv_dim,
                args.qk_norm_eps,
                vb.pp("k_norm"),
            )?)
        } else {
            None
        };
        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            n_heads: args.n_heads,
            n_kv_heads: args.n_kv_heads,
            head_dim: args.head_dim,
            sliding_window: window_size,
            causal: args.causal,
            alibi_slopes: alibi_slopes(args.n_heads),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch = x.dim(0)?;
        let seq_len = x.dim(1)?;
        let q_dim = self.n_heads * self.head_dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let mut q = linear_forward_last_dim(&self.wq, x)?;
        let mut k = linear_forward_last_dim(&self.wk, x)?;
        let v = linear_forward_last_dim(&self.wv, x)?;
        if let Some(norm) = &self.q_norm {
            q = norm.forward(&q)?;
        }
        if let Some(norm) = &self.k_norm {
            k = norm.forward(&k)?;
        }
        let q = q.reshape((batch, seq_len, self.n_heads, self.head_dim))?;
        let k = k.reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?;
        let v = v.reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?;
        let k = repeat_kv(&k, self.n_heads, self.n_kv_heads)?;
        let v = repeat_kv(&v, self.n_heads, self.n_kv_heads)?;
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let q = q.reshape((batch * self.n_heads, seq_len, self.head_dim))?;
        let k = k.reshape((batch * self.n_heads, seq_len, self.head_dim))?;
        let v = v.reshape((batch * self.n_heads, seq_len, self.head_dim))?;

        let mut attn = q.matmul(&k.transpose(1, 2)?)?;
        let scale =
            Tensor::new((self.head_dim as f32).sqrt(), attn.device())?.to_dtype(attn.dtype())?;
        attn = attn.broadcast_div(&scale)?;
        let bias = alibi_attention_bias(
            batch,
            self.n_heads,
            seq_len,
            self.sliding_window,
            self.causal,
            &self.alibi_slopes,
            attn.device(),
            attn.dtype(),
        )?;
        attn = attn.broadcast_add(&bias)?;
        let attn = ops::softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v)?;
        let out = out
            .reshape((batch, self.n_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((batch, seq_len, q_dim))?;
        if out.dim(2)? != q_dim || kv_dim == 0 {
            return Err(Error::InferenceError(
                "Voxtral codec attention produced an invalid shape".to_string(),
            ));
        }
        linear_forward_last_dim(&self.wo, &out)
    }
}

impl VoxtralCodecFeedForward {
    fn load(args: &VoxtralTtsAudioTokenizerArgs, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            w1: candle_nn::linear_no_bias(args.dim, args.hidden_dim, vb.pp("w1"))?,
            w2: linear_maybe_bias(args.hidden_dim, args.dim, args.use_biases, vb.pp("w2"))?,
            w3: candle_nn::linear_no_bias(args.dim, args.hidden_dim, vb.pp("w3"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = linear_forward_last_dim(&self.w1, x)?;
        let up = linear_forward_last_dim(&self.w3, x)?;
        let hidden = ops::silu(&gate)?.broadcast_mul(&up)?;
        linear_forward_last_dim(&self.w2, &hidden)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxtralCodecTimeline {
    pub shifted_codebooks: Vec<Vec<u32>>,
}

impl VoxtralCodecTimeline {
    pub fn new(shifted_codebooks: Vec<Vec<u32>>) -> Result<Self> {
        if shifted_codebooks.is_empty() {
            return Err(Error::AudioError(
                "Voxtral codec timeline must contain at least one codebook".to_string(),
            ));
        }
        let frames = shifted_codebooks[0].len();
        if shifted_codebooks
            .iter()
            .any(|codebook| codebook.len() != frames)
        {
            return Err(Error::AudioError(
                "Voxtral codec codebooks must all have the same frame length".to_string(),
            ));
        }
        Ok(Self { shifted_codebooks })
    }

    pub fn codebook_count(&self) -> usize {
        self.shifted_codebooks.len()
    }

    pub fn generated_frame_count(&self) -> usize {
        self.shifted_codebooks.first().map(Vec::len).unwrap_or(0)
    }

    pub fn audible_frame_count(&self) -> usize {
        self.shifted_codebooks
            .first()
            .and_then(|semantic| {
                semantic
                    .iter()
                    .position(|token| strip_audio_token_offset(*token) == AudioCodeValue::End)
            })
            .unwrap_or_else(|| self.generated_frame_count())
    }

    pub fn trim_at_end_audio(&self) -> Self {
        let frames = self.audible_frame_count();
        let shifted_codebooks = self
            .shifted_codebooks
            .iter()
            .map(|codebook| codebook[..frames].to_vec())
            .collect();
        Self { shifted_codebooks }
    }

    pub fn unshifted_codebooks(&self) -> Vec<Vec<Option<u32>>> {
        self.shifted_codebooks
            .iter()
            .map(|codebook| {
                codebook
                    .iter()
                    .map(|token| match strip_audio_token_offset(*token) {
                        AudioCodeValue::Code(code) => Some(code),
                        AudioCodeValue::Empty | AudioCodeValue::End => None,
                    })
                    .collect()
            })
            .collect()
    }

    fn to_unshifted_tensor_range(&self, range: Range<usize>, device: &Device) -> Result<Tensor> {
        if range.end > self.generated_frame_count() || range.start > range.end {
            return Err(Error::AudioError(format!(
                "Invalid Voxtral codec frame range {}..{} for {} frames",
                range.start,
                range.end,
                self.generated_frame_count()
            )));
        }
        let frames = range.end - range.start;
        let mut values = Vec::with_capacity(self.codebook_count() * frames);
        for codebook in &self.shifted_codebooks {
            for token in &codebook[range.clone()] {
                let code = match strip_audio_token_offset(*token) {
                    AudioCodeValue::Code(code) => code,
                    AudioCodeValue::Empty | AudioCodeValue::End => 0,
                };
                values.push(code);
            }
        }
        Tensor::from_vec(values, (1, self.codebook_count(), frames), device).map_err(Error::from)
    }
}

pub fn chunk_ranges(frames: usize, chunk_frames: usize) -> Vec<Range<usize>> {
    if frames == 0 {
        return Vec::new();
    }
    let chunk_frames = chunk_frames.max(1);
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < frames {
        let end = (start + chunk_frames).min(frames);
        ranges.push(start..end);
        start = end;
    }
    ranges
}

pub fn trim_samples_to_frames(
    samples: &mut Vec<f32>,
    frames: usize,
    config: &VoxtralCodecConfig,
) -> Result<()> {
    let expected = config.samples_for_frames(frames)?;
    if samples.len() > expected {
        samples.truncate(expected);
    }
    Ok(())
}

fn decoder_initial_window_size(
    args: &VoxtralTtsAudioTokenizerArgs,
    decoder_strides: &[usize],
) -> Result<usize> {
    if !args.half_attn_window_upon_downsampling {
        return Ok(args.attn_sliding_window_size.max(1));
    }
    let upsample_factor = decoder_strides
        .iter()
        .copied()
        .filter(|stride| *stride > 1)
        .try_fold(1usize, |acc, stride| {
            acc.checked_mul(stride).ok_or_else(|| {
                Error::ConfigError("Voxtral codec decoder stride product overflowed".to_string())
            })
        })?;
    Ok((args.attn_sliding_window_size / upsample_factor).max(1))
}

fn load_conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    use_weight_norm: bool,
    use_bias: bool,
    cfg: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    if use_weight_norm && vb.contains_tensor("parametrizations.weight.original0") {
        let weight_g = vb.get((out_channels, 1, 1), "parametrizations.weight.original0")?;
        let weight_v = vb.get(
            (out_channels, in_channels, kernel_size),
            "parametrizations.weight.original1",
        )?;
        let norm_v = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
        let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
        let bias = optional_bias(&vb, out_channels, use_bias)?;
        return Ok(Conv1d::new(weight, bias, cfg));
    }
    if vb.contains_tensor("weight") {
        let weight = vb.get((out_channels, in_channels, kernel_size), "weight")?;
        let bias = optional_bias(&vb, out_channels, use_bias)?;
        return Ok(Conv1d::new(weight, bias, cfg));
    }
    Err(Error::ModelLoadError(format!(
        "Voxtral codec conv1d is missing weights; expected weight or parametrizations.weight.original{{0,1}} at {}",
        vb.prefix()
    )))
}

fn load_conv_transpose1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    use_weight_norm: bool,
    use_bias: bool,
    cfg: ConvTranspose1dConfig,
    vb: VarBuilder,
) -> Result<ConvTranspose1d> {
    if use_weight_norm && vb.contains_tensor("parametrizations.weight.original0") {
        let weight_g = vb.get((in_channels, 1, 1), "parametrizations.weight.original0")?;
        let weight_v = vb.get(
            (in_channels, out_channels, kernel_size),
            "parametrizations.weight.original1",
        )?;
        let norm_v = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
        let weight = weight_v.broadcast_mul(&weight_g)?.broadcast_div(&norm_v)?;
        let bias = optional_bias(&vb, out_channels, use_bias)?;
        return Ok(ConvTranspose1d::new(weight, bias, cfg));
    }
    if vb.contains_tensor("weight") {
        let weight = vb.get((in_channels, out_channels, kernel_size), "weight")?;
        let bias = optional_bias(&vb, out_channels, use_bias)?;
        return Ok(ConvTranspose1d::new(weight, bias, cfg));
    }
    Err(Error::ModelLoadError(format!(
        "Voxtral codec conv_transpose1d is missing weights; expected weight or parametrizations.weight.original{{0,1}} at {}",
        vb.prefix()
    )))
}

fn optional_bias(vb: &VarBuilder, dim: usize, use_bias: bool) -> Result<Option<Tensor>> {
    if use_bias && vb.contains_tensor("bias") {
        Ok(Some(vb.get((dim,), "bias")?))
    } else {
        Ok(None)
    }
}

fn linear_maybe_bias(
    in_dim: usize,
    out_dim: usize,
    use_bias: bool,
    vb: VarBuilder,
) -> Result<Linear> {
    if use_bias {
        candle_nn::linear(in_dim, out_dim, vb).map_err(Error::from)
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb).map_err(Error::from)
    }
}

fn load_layer_scale(
    args: &VoxtralTtsAudioTokenizerArgs,
    vb: VarBuilder,
    name: &str,
    layer_idx: usize,
) -> Result<Option<Tensor>> {
    if !args.layer_scale {
        return Ok(None);
    }
    if vb.contains_tensor(name) {
        return Ok(Some(vb.get((args.dim,), name)?));
    }
    let init = args.layer_scale_init.unwrap_or_else(|| {
        if layer_idx < 18 {
            0.1
        } else if layer_idx <= 24 {
            1e-5
        } else {
            1e-6
        }
    });
    let values = vec![init; args.dim];
    Tensor::from_vec(values, (args.dim,), vb.device())
        .map(Some)
        .map_err(Error::from)
}

fn pad_1d(x: &Tensor, left: usize, right: usize, mode: VoxtralPadMode) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    match mode {
        VoxtralPadMode::Constant => x.pad_with_zeros(2, left, right).map_err(Error::from),
        VoxtralPadMode::Replicate => pad_replicate(x, left, right),
    }
}

fn pad_replicate(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let len = x.dim(2)?;
    if len == 0 {
        return x.pad_with_zeros(2, left, right).map_err(Error::from);
    }
    let mut parts = Vec::with_capacity(3);
    if left > 0 {
        let first = x.narrow(2, 0, 1)?;
        parts.push(first.broadcast_as((x.dim(0)?, x.dim(1)?, left))?);
    }
    parts.push(x.clone());
    if right > 0 {
        let last = x.narrow(2, len - 1, 1)?;
        parts.push(last.broadcast_as((x.dim(0)?, x.dim(1)?, right))?);
    }
    Tensor::cat(&parts, 2).map_err(Error::from)
}

fn alibi_slopes(n_heads: usize) -> Vec<f32> {
    fn slopes_power_of_2(n: usize) -> Vec<f32> {
        let ratio = 2.0f32.powf(-8.0 / n as f32);
        (0..n).map(|idx| ratio.powi(idx as i32)).collect()
    }
    if n_heads == 0 {
        return Vec::new();
    }
    if n_heads.is_power_of_two() {
        return slopes_power_of_2(n_heads);
    }
    let base = 1usize << (usize::BITS as usize - 1 - n_heads.leading_zeros() as usize);
    let mut slopes = slopes_power_of_2(base);
    let extras = slopes_power_of_2(base * 2);
    slopes.extend(extras.into_iter().step_by(2).take(n_heads - base));
    slopes
}

#[allow(clippy::too_many_arguments)]
fn alibi_attention_bias(
    batch: usize,
    n_heads: usize,
    seq_len: usize,
    sliding_window: usize,
    causal: bool,
    slopes: &[f32],
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut values = Vec::with_capacity(batch * n_heads * seq_len * seq_len);
    let window_left = sliding_window as isize;
    let window_right = if causal { 0 } else { sliding_window as isize };
    for _batch_idx in 0..batch {
        for head in 0..n_heads {
            let slope = slopes.get(head).copied().unwrap_or(1.0);
            for query in 0..seq_len {
                for key in 0..seq_len {
                    let rel = key as isize - query as isize;
                    let outside = rel < -window_left || rel > window_right;
                    let future = causal && rel > 0;
                    if outside || future {
                        values.push(f32::NEG_INFINITY);
                    } else {
                        values.push(slope * rel as f32);
                    }
                }
            }
        }
    }
    Tensor::from_vec(values, (batch * n_heads, seq_len, seq_len), device)?
        .to_dtype(dtype)
        .map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::architectures::voxtral::tts::acoustic::{
        apply_audio_token_offset, fsq_code_to_unit, AudioSpecialToken,
    };
    use crate::models::architectures::voxtral::tts::config::{fixture_json, VoxtralTtsConfig};
    use candle_core::{Shape, Tensor};
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn derives_codec_shape_from_params() {
        let config = VoxtralTtsConfig::from_json_str(fixture_json()).unwrap();
        let codec = VoxtralCodecConfig::from_config(&config).unwrap();
        assert_eq!(codec.sample_rate, 24_000);
        assert_eq!(codec.patch_size, 240);
        assert_eq!(codec.decoder_strides, vec![1, 2, 2, 2]);
        assert_eq!(codec.downsample_factor().unwrap(), 1920);
        assert_eq!(codec.samples_for_frames(12).unwrap(), 23_040);
        assert_eq!(codec.frame_count_for_samples_ceil(1921).unwrap(), 2);
        assert_eq!(codec.latent_dim, 292);
    }

    #[test]
    fn cuts_timeline_at_end_audio_on_semantic_codebook() {
        let semantic = vec![
            apply_audio_token_offset(3),
            apply_audio_token_offset(4),
            AudioSpecialToken::End.id(),
            apply_audio_token_offset(5),
        ];
        let acoustic = vec![apply_audio_token_offset(1); 4];
        let timeline = VoxtralCodecTimeline::new(vec![semantic, acoustic]).unwrap();
        assert_eq!(timeline.generated_frame_count(), 4);
        assert_eq!(timeline.audible_frame_count(), 2);
        assert_eq!(timeline.trim_at_end_audio().generated_frame_count(), 2);
    }

    #[test]
    fn chunks_like_vllm_omni_decoder_helper() {
        let ranges = chunk_ranges(751, VOXTRAL_CODEC_CHUNK_FRAMES);
        assert_eq!(ranges, vec![0..375, 375..750, 750..751]);
    }

    #[test]
    fn unshifts_special_audio_tokens_to_none() {
        let timeline = VoxtralCodecTimeline::new(vec![vec![0, 1, 2, 8]]).unwrap();
        assert_eq!(
            timeline.unshifted_codebooks(),
            vec![vec![None, None, Some(0), Some(6)]]
        );
    }

    #[test]
    fn codec_quantizer_decodes_semantic_vq_and_acoustic_fsq() {
        let device = Device::Cpu;
        let config = tiny_codec_config();
        let tensors = tiny_codec_tensors(&device);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device).pp("audio_tokenizer");
        let quantizer = VoxtralCodecQuantizer::load(&config, vb.pp("quantizer")).unwrap();
        let codes = Tensor::from_vec(vec![1u32, 2, 0, 2, 1, 0], (1, 3, 2), &device).unwrap();
        let latent = quantizer.decode(&codes, DType::F32).unwrap();
        assert_eq!(latent.dims(), &[1, 4, 2]);
        let values = latent.to_vec3::<f32>().unwrap();
        assert_eq!(values[0][0], vec![2.0, 4.0]);
        assert!((values[0][2][0] - fsq_code_to_unit(0, 3)).abs() < 1e-6);
        assert!((values[0][2][1] - fsq_code_to_unit(2, 3)).abs() < 1e-6);
    }

    #[test]
    fn tiny_codec_decoder_preserves_vllm_sample_count() {
        let device = Device::Cpu;
        let config = tiny_codec_config();
        let tensors = tiny_codec_tensors(&device);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device).pp("audio_tokenizer");
        let decoder = VoxtralCodecDecoder::load(&config, vb).unwrap();
        let semantic = vec![apply_audio_token_offset(1), apply_audio_token_offset(2)];
        let acoustic_a = vec![apply_audio_token_offset(0), apply_audio_token_offset(1)];
        let acoustic_b = vec![apply_audio_token_offset(2), apply_audio_token_offset(0)];
        let timeline = VoxtralCodecTimeline::new(vec![semantic, acoustic_a, acoustic_b]).unwrap();
        let samples = decoder.decode_timeline(&timeline).unwrap();
        assert_eq!(samples.len(), 8);
        assert!(samples.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn tiny_codec_decoder_trims_at_end_audio_before_decoding() {
        let device = Device::Cpu;
        let config = tiny_codec_config();
        let tensors = tiny_codec_tensors(&device);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device).pp("audio_tokenizer");
        let decoder = VoxtralCodecDecoder::load(&config, vb).unwrap();
        let semantic = vec![
            apply_audio_token_offset(1),
            AudioSpecialToken::End.id(),
            apply_audio_token_offset(2),
        ];
        let acoustic_a = vec![apply_audio_token_offset(0); 3];
        let acoustic_b = vec![apply_audio_token_offset(1); 3];
        let timeline = VoxtralCodecTimeline::new(vec![semantic, acoustic_a, acoustic_b]).unwrap();
        let samples = decoder.decode_timeline(&timeline).unwrap();
        assert_eq!(samples.len(), 4);
    }

    #[test]
    fn alibi_bias_masks_future_and_outside_sliding_window() {
        let device = Device::Cpu;
        let bias = alibi_attention_bias(1, 1, 4, 2, true, &[1.0], &device, DType::F32).unwrap();
        let values = bias.to_vec3::<f32>().unwrap();
        assert!(values[0][0][1].is_infinite());
        assert!(values[0][3][0].is_infinite());
        assert_eq!(values[0][3][1], -2.0);
    }

    fn tiny_codec_config() -> VoxtralTtsConfig {
        let mut value: serde_json::Value = serde_json::from_str(fixture_json()).unwrap();
        value["dim"] = json!(4);
        value["hidden_dim"] = json!(8);
        value["n_heads"] = json!(2);
        value["n_kv_heads"] = json!(1);
        value["head_dim"] = json!(2);
        let audio = &mut value["multimodal"]["audio_model_args"];
        audio["semantic_codebook_size"] = json!(4);
        audio["acoustic_codebook_size"] = json!(3);
        audio["n_acoustic_codebook"] = json!(2);
        audio["audio_encoding_args"]["num_codebooks"] = json!(3);
        let tokenizer = &mut value["multimodal"]["audio_tokenizer_args"];
        tokenizer["pretransform_patch_size"] = json!(2);
        tokenizer["patch_proj_kernel_size"] = json!(3);
        tokenizer["semantic_codebook_size"] = json!(4);
        tokenizer["semantic_dim"] = json!(2);
        tokenizer["acoustic_codebook_size"] = json!(3);
        tokenizer["acoustic_dim"] = json!(2);
        tokenizer["conv_weight_norm"] = json!(false);
        tokenizer["attn_sliding_window_size"] = json!(4);
        tokenizer["dim"] = json!(4);
        tokenizer["hidden_dim"] = json!(8);
        tokenizer["head_dim"] = json!(2);
        tokenizer["n_heads"] = json!(2);
        tokenizer["n_kv_heads"] = json!(1);
        tokenizer["qk_norm_eps"] = json!(1e-6);
        tokenizer["qk_norm"] = json!(true);
        tokenizer["norm_eps"] = json!(1e-5);
        tokenizer["layer_scale"] = json!(true);
        tokenizer["layer_scale_init"] = json!(1.0);
        tokenizer["decoder_transformer_lengths_str"] = json!("1,1");
        tokenizer["decoder_convs_kernels_str"] = json!("3,4");
        tokenizer["decoder_convs_strides_str"] = json!("1,2");
        VoxtralTtsConfig::from_json_str(&value.to_string()).unwrap()
    }

    fn tiny_codec_tensors(device: &Device) -> HashMap<String, Tensor> {
        fn zeros(shape: impl Into<Shape>, device: &Device) -> Tensor {
            Tensor::zeros(shape, DType::F32, device).unwrap()
        }
        fn ones(shape: impl Into<Shape>, device: &Device) -> Tensor {
            Tensor::ones(shape, DType::F32, device).unwrap()
        }
        let mut tensors = HashMap::from([
            (
                "audio_tokenizer.quantizer.semantic_codebook.cluster_usage".to_string(),
                ones((4,), device),
            ),
            (
                "audio_tokenizer.quantizer.semantic_codebook.embedding_sum".to_string(),
                Tensor::from_vec(
                    vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                    (4, 2),
                    device,
                )
                .unwrap(),
            ),
            (
                "audio_tokenizer.decoder_blocks.0.conv.weight".to_string(),
                zeros((4, 4, 3), device),
            ),
            (
                "audio_tokenizer.decoder_blocks.2.conv.weight".to_string(),
                zeros((4, 4, 4), device),
            ),
            (
                "audio_tokenizer.output_proj.conv.weight".to_string(),
                zeros((2, 4, 3), device),
            ),
        ]);
        for block in [1usize, 3usize] {
            add_transformer_tensors(&mut tensors, device, block);
        }
        tensors
    }

    fn add_transformer_tensors(
        tensors: &mut HashMap<String, Tensor>,
        device: &Device,
        block: usize,
    ) {
        fn zeros(shape: impl Into<Shape>, device: &Device) -> Tensor {
            Tensor::zeros(shape, DType::F32, device).unwrap()
        }
        fn ones(shape: impl Into<Shape>, device: &Device) -> Tensor {
            Tensor::ones(shape, DType::F32, device).unwrap()
        }
        let prefix = format!("audio_tokenizer.decoder_blocks.{block}.layers.0");
        for (name, tensor) in [
            ("attention.wq.weight", zeros((4, 4), device)),
            ("attention.wk.weight", zeros((2, 4), device)),
            ("attention.wv.weight", zeros((2, 4), device)),
            ("attention.wo.weight", zeros((4, 4), device)),
            ("attention.q_norm.weight", ones((4,), device)),
            ("attention.k_norm.weight", ones((2,), device)),
            ("attention_norm.weight", ones((4,), device)),
            ("ffn_norm.weight", ones((4,), device)),
            ("feed_forward.w1.weight", zeros((8, 4), device)),
            ("feed_forward.w2.weight", zeros((4, 8), device)),
            ("feed_forward.w3.weight", zeros((8, 4), device)),
            ("attention_scale", ones((4,), device)),
            ("ffn_scale", ones((4,), device)),
        ] {
            tensors.insert(format!("{prefix}.{name}"), tensor);
        }
    }
}
