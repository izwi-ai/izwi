//! Fish S2 modified DAC codec.

use candle_core::{DType, IndexOp, Tensor, D};
use candle_nn::{
    ops, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Embedding, LayerNorm,
    Linear, Module, RmsNorm, VarBuilder,
};

use crate::error::{Error, Result};
use crate::models::architectures::fish_s2::codec::fuse_weight_norm_dim0;
use crate::models::architectures::fish_s2::contracts::FishS2DacContract;
use crate::models::architectures::fish_s2::tokenizer::FishS2VqCodes;
use crate::models::architectures::qwen3::core::{build_rope_cache, causal_mask, repeat_kv};

#[derive(Debug, Clone, PartialEq)]
pub struct FishS2DacConfig {
    pub sample_rate: u32,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub downsample_factors: Vec<usize>,
    pub codebook_dim: usize,
    pub semantic_codebook_size: usize,
    pub residual_codebook_size: usize,
    pub residual_codebooks: usize,
    pub transformer_layers: usize,
    pub transformer_heads: usize,
    pub transformer_kv_heads: usize,
    pub transformer_head_dim: usize,
    pub transformer_intermediate: usize,
    pub transformer_window_size: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
}

pub struct FishS2DacDecoder {
    config: FishS2DacConfig,
    quantizer: FishS2DownsampleResidualVectorQuantizer,
    decoder: FishS2DacAudioDecoder,
}

struct FishS2DownsampleResidualVectorQuantizer {
    semantic_quantizer: FishS2ResidualVectorQuantizer,
    residual_quantizer: FishS2ResidualVectorQuantizer,
    post_module: FishS2WindowLimitedTransformer,
    upsample: Vec<FishS2UpsampleBlock>,
}

struct FishS2ResidualVectorQuantizer {
    quantizers: Vec<FishS2VectorQuantizer>,
}

struct FishS2VectorQuantizer {
    in_proj: FishS2CausalConv1d,
    out_proj: FishS2CausalConv1d,
    codebook: Embedding,
}

struct FishS2UpsampleBlock {
    transposed: FishS2CausalConvTranspose1d,
    convnext: FishS2ConvNeXtBlock,
}

struct FishS2ConvNeXtBlock {
    dwconv: FishS2CausalConv1d,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Option<Tensor>,
}

struct FishS2DacAudioDecoder {
    first: FishS2CausalConv1d,
    blocks: Vec<FishS2DecoderBlock>,
    final_snake: FishS2Snake1d,
    final_conv: FishS2CausalConv1d,
}

struct FishS2DecoderBlock {
    snake: FishS2Snake1d,
    transposed: FishS2CausalConvTranspose1d,
    residuals: Vec<FishS2ResidualUnit>,
}

struct FishS2ResidualUnit {
    snake1: FishS2Snake1d,
    conv1: FishS2CausalConv1d,
    snake2: FishS2Snake1d,
    conv2: FishS2CausalConv1d,
    causal: bool,
}

struct FishS2Snake1d {
    alpha: Tensor,
}

struct FishS2CausalConv1d {
    conv: Conv1d,
    effective_kernel: usize,
    stride: usize,
    padding_total: usize,
}

struct FishS2CausalConvTranspose1d {
    conv: ConvTranspose1d,
    left_trim: usize,
    right_trim: usize,
}

struct FishS2WindowLimitedTransformer {
    input_proj: Option<Linear>,
    layers: Vec<FishS2DacTransformerBlock>,
    norm: RmsNorm,
    output_proj: Option<Linear>,
    channels_first: bool,
}

struct FishS2DacTransformerBlock {
    attention: FishS2DacAttention,
    feed_forward: FishS2DacFeedForward,
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    attention_scale: Tensor,
    ffn_scale: Tensor,
}

struct FishS2DacAttention {
    wqkv: Linear,
    wo: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,
    window_size: usize,
}

struct FishS2DacFeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl FishS2DacConfig {
    pub fn current() -> Self {
        let contract = FishS2DacContract::CURRENT;
        Self {
            sample_rate: contract.sample_rate,
            latent_dim: 1024,
            decoder_dim: 1536,
            decoder_rates: contract.decoder_rates.to_vec(),
            downsample_factors: vec![2, 2],
            codebook_dim: 8,
            semantic_codebook_size: contract.semantic_codebook_size,
            residual_codebook_size: contract.residual_codebook_size,
            residual_codebooks: contract.residual_codebooks,
            transformer_layers: 8,
            transformer_heads: 16,
            transformer_kv_heads: 16,
            transformer_head_dim: 64,
            transformer_intermediate: 3072,
            transformer_window_size: 128,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
        }
    }

    pub fn num_codebooks(&self) -> usize {
        1 + self.residual_codebooks
    }

    pub fn samples_per_frame(&self) -> Result<usize> {
        let upsample = self
            .downsample_factors
            .iter()
            .chain(self.decoder_rates.iter())
            .try_fold(1usize, |acc, value| {
                acc.checked_mul(*value).ok_or_else(|| {
                    Error::ConfigError("Fish S2 DAC upsample factor overflowed".to_string())
                })
            })?;
        Ok(upsample)
    }
}

impl FishS2DacDecoder {
    pub fn load(config: FishS2DacConfig, vb: VarBuilder) -> Result<Self> {
        let quantizer = FishS2DownsampleResidualVectorQuantizer::load(&config, vb.pp("quantizer"))?;
        let decoder = FishS2DacAudioDecoder::load(&config, vb.pp("decoder"))?;
        Ok(Self {
            config,
            quantizer,
            decoder,
        })
    }

    pub fn config(&self) -> &FishS2DacConfig {
        &self.config
    }

    pub fn decode_vq_codes(&self, codes: &FishS2VqCodes) -> Result<Vec<f32>> {
        let audio = self.decode_codebooks(&codes.codebooks)?;
        audio
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()
            .map_err(Error::from)
    }

    pub fn decode_codebooks(&self, codebooks: &[Vec<u32>]) -> Result<Tensor> {
        let codes = codebooks_to_tensor(codebooks, &self.config, self.decoder_device()?)?;
        let latents = self.quantizer.decode(&codes, &self.config)?;
        self.decoder.forward(&latents)?.tanh().map_err(Error::from)
    }

    fn decoder_device(&self) -> Result<&candle_core::Device> {
        Ok(self.decoder.final_conv.conv.weight().device())
    }
}

impl FishS2DownsampleResidualVectorQuantizer {
    fn load(config: &FishS2DacConfig, vb: VarBuilder) -> Result<Self> {
        let semantic_quantizer = FishS2ResidualVectorQuantizer::load(
            config.latent_dim,
            1,
            config.semantic_codebook_size,
            config.codebook_dim,
            vb.pp("semantic_quantizer"),
        )?;
        let residual_quantizer = FishS2ResidualVectorQuantizer::load(
            config.latent_dim,
            config.residual_codebooks,
            config.residual_codebook_size,
            config.codebook_dim,
            vb.pp("quantizer"),
        )?;
        let post_module = FishS2WindowLimitedTransformer::load(
            config.latent_dim,
            config.latent_dim,
            config,
            true,
            vb.pp("post_module"),
        )?;

        let dims = std::iter::once(config.latent_dim)
            .chain(std::iter::repeat(config.latent_dim).take(config.downsample_factors.len()))
            .collect::<Vec<_>>();
        let mut upsample = Vec::with_capacity(config.downsample_factors.len());
        for (out_idx, (idx, factor)) in config
            .downsample_factors
            .iter()
            .copied()
            .enumerate()
            .rev()
            .enumerate()
        {
            upsample.push(FishS2UpsampleBlock::load(
                dims[idx + 1],
                dims[idx],
                factor,
                vb.pp(format!("upsample.{out_idx}")),
            )?);
        }

        Ok(Self {
            semantic_quantizer,
            residual_quantizer,
            post_module,
            upsample,
        })
    }

    fn decode(&self, codes: &Tensor, config: &FishS2DacConfig) -> Result<Tensor> {
        if codes.rank() != 3 {
            return Err(Error::AudioError(format!(
                "Fish S2 DAC codes must have shape [B, K, T], got rank {}",
                codes.rank()
            )));
        }
        if codes.dim(1)? != config.num_codebooks() {
            return Err(Error::AudioError(format!(
                "Fish S2 DAC expected {} codebooks, got {}",
                config.num_codebooks(),
                codes.dim(1)?
            )));
        }

        let semantic_codes = codes.narrow(1, 0, 1)?;
        let residual_codes = codes.narrow(1, 1, config.residual_codebooks)?;
        let semantic = self.semantic_quantizer.from_codes(&semantic_codes)?;
        let residual = self.residual_quantizer.from_codes(&residual_codes)?;
        let mut z = semantic.broadcast_add(&residual)?;
        z = self.post_module.forward(&z)?;
        for block in &self.upsample {
            z = block.forward(&z)?;
        }
        Ok(z)
    }
}

impl FishS2ResidualVectorQuantizer {
    fn load(
        input_dim: usize,
        n_codebooks: usize,
        codebook_size: usize,
        codebook_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = vb.pp("quantizers");
        let mut quantizers = Vec::with_capacity(n_codebooks);
        for idx in 0..n_codebooks {
            quantizers.push(FishS2VectorQuantizer::load(
                input_dim,
                codebook_size,
                codebook_dim,
                vb.pp(idx),
            )?);
        }
        Ok(Self { quantizers })
    }

    fn from_codes(&self, codes: &Tensor) -> Result<Tensor> {
        let mut sum: Option<Tensor> = None;
        for (idx, quantizer) in self.quantizers.iter().enumerate() {
            let z_p = quantizer.decode_code(&codes.i((.., idx, ..))?)?;
            let z_q = quantizer.out_proj.forward(&z_p)?;
            sum = Some(match sum {
                Some(current) => current.broadcast_add(&z_q)?,
                None => z_q,
            });
        }
        sum.ok_or_else(|| Error::AudioError("Fish S2 DAC has no quantizers".to_string()))
    }
}

impl FishS2VectorQuantizer {
    fn load(
        input_dim: usize,
        codebook_size: usize,
        codebook_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let in_proj = FishS2CausalConv1d::load(input_dim, codebook_dim, 1, 1, 1, vb.pp("in_proj"))?;
        let out_proj =
            FishS2CausalConv1d::load(codebook_dim, input_dim, 1, 1, 1, vb.pp("out_proj"))?;
        let codebook = candle_nn::embedding(codebook_size, codebook_dim, vb.pp("codebook"))?;
        Ok(Self {
            in_proj,
            out_proj,
            codebook,
        })
    }

    fn decode_code(&self, codes: &Tensor) -> Result<Tensor> {
        let _ = &self.in_proj;
        self.codebook
            .forward(codes)?
            .transpose(1, 2)
            .map_err(Error::from)
    }
}

impl FishS2UpsampleBlock {
    fn load(in_dim: usize, out_dim: usize, factor: usize, vb: VarBuilder) -> Result<Self> {
        let transposed =
            FishS2CausalConvTranspose1d::load(in_dim, out_dim, factor, factor, vb.pp("0"))?;
        let convnext = FishS2ConvNeXtBlock::load(out_dim, vb.pp("1"))?;
        Ok(Self {
            transposed,
            convnext,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.convnext.forward(&self.transposed.forward(x)?)
    }
}

impl FishS2ConvNeXtBlock {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        let dwconv = FishS2CausalConv1d::load_with_groups(dim, dim, 7, 1, 1, dim, vb.pp("dwconv"))?;
        let norm = candle_nn::layer_norm(dim, 1e-6, vb.pp("norm"))?;
        let pwconv1 = candle_nn::linear(dim, dim * 4, vb.pp("pwconv1"))?;
        let pwconv2 = candle_nn::linear(dim * 4, dim, vb.pp("pwconv2"))?;
        let gamma = if vb.contains_tensor("gamma") {
            Some(vb.get((dim,), "gamma")?)
        } else {
            None
        };
        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let mut hidden = self.dwconv.forward(x)?.transpose(1, 2)?;
        hidden = self.norm.forward(&hidden)?;
        hidden = self.pwconv1.forward(&hidden)?.gelu()?;
        hidden = self.pwconv2.forward(&hidden)?;
        if let Some(gamma) = &self.gamma {
            hidden = hidden.broadcast_mul(&gamma.reshape((1, 1, gamma.dim(0)?))?)?;
        }
        hidden = hidden.transpose(1, 2)?;
        residual.broadcast_add(&hidden).map_err(Error::from)
    }
}

impl FishS2DacAudioDecoder {
    fn load(config: &FishS2DacConfig, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("model");
        let first =
            FishS2CausalConv1d::load(config.latent_dim, config.decoder_dim, 7, 1, 1, vb.pp("0"))?;
        let mut blocks = Vec::with_capacity(config.decoder_rates.len());
        for (idx, stride) in config.decoder_rates.iter().copied().enumerate() {
            let in_dim = config.decoder_dim / (1usize << idx);
            let out_dim = config.decoder_dim / (1usize << (idx + 1));
            blocks.push(FishS2DecoderBlock::load(
                in_dim,
                out_dim,
                stride,
                vb.pp(idx + 1),
            )?);
        }
        let final_dim = config.decoder_dim / (1usize << config.decoder_rates.len());
        let final_snake = FishS2Snake1d::load(final_dim, vb.pp(config.decoder_rates.len() + 1))?;
        let final_conv =
            FishS2CausalConv1d::load(final_dim, 1, 7, 1, 1, vb.pp(config.decoder_rates.len() + 2))?;
        Ok(Self {
            first,
            blocks,
            final_snake,
            final_conv,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut hidden = self.first.forward(x)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        self.final_conv.forward(&self.final_snake.forward(&hidden)?)
    }
}

impl FishS2DecoderBlock {
    fn load(in_dim: usize, out_dim: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("block");
        let snake = FishS2Snake1d::load(in_dim, vb.pp("0"))?;
        let transposed =
            FishS2CausalConvTranspose1d::load(in_dim, out_dim, 2 * stride, stride, vb.pp("1"))?;
        let residuals = [1usize, 3, 9]
            .into_iter()
            .enumerate()
            .map(|(idx, dilation)| {
                FishS2ResidualUnit::load(out_dim, dilation, true, vb.pp(idx + 2))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            snake,
            transposed,
            residuals,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut hidden = self.transposed.forward(&self.snake.forward(x)?)?;
        for residual in &self.residuals {
            hidden = residual.forward(&hidden)?;
        }
        Ok(hidden)
    }
}

impl FishS2ResidualUnit {
    fn load(dim: usize, dilation: usize, causal: bool, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("block");
        let snake1 = FishS2Snake1d::load(dim, vb.pp("0"))?;
        let conv1 = FishS2CausalConv1d::load(dim, dim, 7, 1, dilation, vb.pp("1"))?;
        let snake2 = FishS2Snake1d::load(dim, vb.pp("2"))?;
        let conv2 = FishS2CausalConv1d::load(dim, dim, 1, 1, 1, vb.pp("3"))?;
        Ok(Self {
            snake1,
            conv1,
            snake2,
            conv2,
            causal,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv2.forward(
            &self
                .snake2
                .forward(&self.conv1.forward(&self.snake1.forward(x)?)?)?,
        )?;
        let x_len = x.dim(2)?;
        let y_len = y.dim(2)?;
        if x_len > y_len {
            let trim = x_len - y_len;
            let residual = if self.causal {
                x.narrow(2, 0, y_len)?
            } else {
                x.narrow(2, trim / 2, y_len)?
            };
            residual.broadcast_add(&y).map_err(Error::from)
        } else {
            x.broadcast_add(&y).map_err(Error::from)
        }
    }
}

impl FishS2Snake1d {
    fn load(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get((1, channels, 1), "alpha")?;
        Ok(Self { alpha })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape().clone();
        let x = x.flatten_from(2)?;
        let sin = self.alpha.broadcast_mul(&x)?.sin()?;
        let sin_sq = sin.sqr()?;
        let inv_alpha = (&self.alpha + 1e-9)?.recip()?;
        x.broadcast_add(&inv_alpha.broadcast_mul(&sin_sq)?)?
            .reshape(shape)
            .map_err(Error::from)
    }
}

impl FishS2CausalConv1d {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::load_with_groups(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            1,
            vb,
        )
    }

    fn load_with_groups(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups,
            ..Default::default()
        };
        let conv = load_conv1d_weight(
            in_channels,
            out_channels,
            kernel_size,
            groups,
            cfg,
            vb.pp("conv"),
        )?;
        let effective_kernel = (kernel_size - 1) * dilation + 1;
        let padding_total = effective_kernel.saturating_sub(stride);
        Ok(Self {
            conv,
            effective_kernel,
            stride,
            padding_total,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let input_len = x.dim(2)?;
        let extra_padding = get_extra_padding_for_conv1d(
            input_len,
            self.effective_kernel,
            self.stride,
            self.padding_total,
        );
        let x = x.pad_with_zeros(2, self.padding_total, extra_padding)?;
        self.conv.forward(&x).map_err(Error::from)
    }
}

impl FishS2CausalConvTranspose1d {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride,
            dilation: 1,
            groups: 1,
        };
        let conv = load_conv_transpose1d_weight(
            in_channels,
            out_channels,
            kernel_size,
            cfg,
            vb.pp("conv"),
        )?;
        let pad = kernel_size.saturating_sub(stride);
        Ok(Self {
            conv,
            left_trim: 0,
            right_trim: pad,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.conv.forward(x)?;
        let out_len = out.dim(2)?;
        let keep = out_len.saturating_sub(self.left_trim + self.right_trim);
        out.narrow(2, self.left_trim, keep).map_err(Error::from)
    }
}

impl FishS2WindowLimitedTransformer {
    fn load(
        input_dim: usize,
        dim: usize,
        config: &FishS2DacConfig,
        channels_first: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_proj = if input_dim != dim || vb.contains_tensor("input_proj.weight") {
            Some(candle_nn::linear(input_dim, dim, vb.pp("input_proj"))?)
        } else {
            None
        };
        let mut layers = Vec::with_capacity(config.transformer_layers);
        for idx in 0..config.transformer_layers {
            layers.push(FishS2DacTransformerBlock::load(
                config,
                vb.pp(format!("layers.{idx}")),
            )?);
        }
        let norm = candle_nn::rms_norm(dim, config.rms_norm_eps, vb.pp("norm"))?;
        let output_proj = if input_dim != dim || vb.contains_tensor("output_proj.weight") {
            Some(candle_nn::linear(dim, input_dim, vb.pp("output_proj"))?)
        } else {
            None
        };
        Ok(Self {
            input_proj,
            layers,
            norm,
            output_proj,
            channels_first,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut hidden = if self.channels_first {
            x.transpose(1, 2)?
        } else {
            x.clone()
        };
        if let Some(input_proj) = &self.input_proj {
            hidden = input_proj.forward(&hidden)?;
        }
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        hidden = self.norm.forward(&hidden)?;
        if let Some(output_proj) = &self.output_proj {
            hidden = output_proj.forward(&hidden)?;
        }
        if self.channels_first {
            hidden.transpose(1, 2).map_err(Error::from)
        } else {
            Ok(hidden)
        }
    }
}

impl FishS2DacTransformerBlock {
    fn load(config: &FishS2DacConfig, vb: VarBuilder) -> Result<Self> {
        let dim = config.transformer_heads * config.transformer_head_dim;
        Ok(Self {
            attention: FishS2DacAttention::load(config, vb.pp("attention"))?,
            feed_forward: FishS2DacFeedForward::load(config, vb.pp("feed_forward"))?,
            attention_norm: candle_nn::rms_norm(dim, config.rms_norm_eps, vb.pp("attention_norm"))?,
            ffn_norm: candle_nn::rms_norm(dim, config.rms_norm_eps, vb.pp("ffn_norm"))?,
            attention_scale: vb.get((dim,), "attention_layer_scale.gamma")?,
            ffn_scale: vb.get((dim,), "ffn_layer_scale.gamma")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let attn = self
            .attention
            .forward(&self.attention_norm.forward(x)?)?
            .broadcast_mul(&self.attention_scale.reshape((
                1,
                1,
                self.attention_scale.dim(0)?,
            ))?)?;
        let hidden = x.broadcast_add(&attn)?;
        let ff = self
            .feed_forward
            .forward(&self.ffn_norm.forward(&hidden)?)?
            .broadcast_mul(&self.ffn_scale.reshape((1, 1, self.ffn_scale.dim(0)?))?)?;
        hidden.broadcast_add(&ff).map_err(Error::from)
    }
}

impl FishS2DacAttention {
    fn load(config: &FishS2DacConfig, vb: VarBuilder) -> Result<Self> {
        let q_dim = config.transformer_heads * config.transformer_head_dim;
        let kv_dim = config.transformer_kv_heads * config.transformer_head_dim;
        let total = q_dim + 2 * kv_dim;
        Ok(Self {
            wqkv: candle_nn::linear_no_bias(q_dim, total, vb.pp("wqkv"))?,
            wo: candle_nn::linear_no_bias(q_dim, q_dim, vb.pp("wo"))?,
            num_heads: config.transformer_heads,
            num_kv_heads: config.transformer_kv_heads,
            head_dim: config.transformer_head_dim,
            rope_theta: config.rope_theta,
            window_size: config.transformer_window_size.max(1),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let bsz = x.dim(0)?;
        let seq_len = x.dim(1)?;
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let qkv = self.wqkv.forward(x)?;
        let q = qkv
            .narrow(2, 0, q_dim)?
            .reshape((bsz, seq_len, self.num_heads, self.head_dim))?;
        let k = qkv.narrow(2, q_dim, kv_dim)?.reshape((
            bsz,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let v = qkv.narrow(2, q_dim + kv_dim, kv_dim)?.reshape((
            bsz,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        ))?;
        let (cos, sin) = build_rope_cache(
            seq_len,
            self.head_dim,
            0,
            self.rope_theta,
            x.device(),
            x.dtype(),
        )?;
        let q = apply_rope(&q, &cos, &sin)?.transpose(1, 2)?;
        let k = apply_rope(&k, &cos, &sin)?.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let k = repeat_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v = repeat_kv(&v, self.num_heads, self.num_kv_heads)?;
        let q = q.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;
        let k = k.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;
        let v = v.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;

        let mut att = q.matmul(&k.transpose(1, 2)?)?;
        let scale =
            Tensor::new((self.head_dim as f32).sqrt(), att.device())?.to_dtype(att.dtype())?;
        att = att.broadcast_div(&scale)?;
        let mask = windowed_causal_mask(seq_len, self.window_size, att.device(), att.dtype())?;
        att = att.broadcast_add(&mask)?;
        let att = ops::softmax(&att, D::Minus1)?;
        let out = att.matmul(&v)?;
        let out = out
            .reshape((bsz, self.num_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((bsz, seq_len, q_dim))?;
        self.wo.forward(&out).map_err(Error::from)
    }
}

impl FishS2DacFeedForward {
    fn load(config: &FishS2DacConfig, vb: VarBuilder) -> Result<Self> {
        let dim = config.transformer_heads * config.transformer_head_dim;
        Ok(Self {
            w1: candle_nn::linear_no_bias(dim, config.transformer_intermediate, vb.pp("w1"))?,
            w2: candle_nn::linear_no_bias(config.transformer_intermediate, dim, vb.pp("w2"))?,
            w3: candle_nn::linear_no_bias(dim, config.transformer_intermediate, vb.pp("w3"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w1.forward(x)?;
        let up = self.w3.forward(x)?;
        let hidden = ops::silu(&gate)?.broadcast_mul(&up)?;
        self.w2.forward(&hidden).map_err(Error::from)
    }
}

fn codebooks_to_tensor(
    codebooks: &[Vec<u32>],
    config: &FishS2DacConfig,
    device: &candle_core::Device,
) -> Result<Tensor> {
    if codebooks.len() != config.num_codebooks() {
        return Err(Error::AudioError(format!(
            "Fish S2 DAC expected {} codebooks, got {}",
            config.num_codebooks(),
            codebooks.len()
        )));
    }
    let frames = codebooks.first().map(Vec::len).unwrap_or(0);
    if frames == 0 {
        return Err(Error::AudioError(
            "Fish S2 DAC cannot decode empty codebooks".to_string(),
        ));
    }
    if codebooks.iter().any(|row| row.len() != frames) {
        return Err(Error::AudioError(
            "Fish S2 DAC codebook rows must have the same frame count".to_string(),
        ));
    }

    let mut values = Vec::with_capacity(codebooks.len() * frames);
    for (idx, row) in codebooks.iter().enumerate() {
        let max_code = if idx == 0 {
            config.semantic_codebook_size
        } else {
            config.residual_codebook_size
        } as u32;
        for code in row {
            values.push((*code).min(max_code.saturating_sub(1)));
        }
    }
    Tensor::from_vec(values, (1, codebooks.len(), frames), device).map_err(Error::from)
}

fn get_extra_padding_for_conv1d(
    length: usize,
    kernel_size: usize,
    stride: usize,
    padding_total: usize,
) -> usize {
    let n_frames = ((length as f64 - kernel_size as f64 + padding_total as f64) / stride as f64
        + 1.0)
        .ceil()
        .max(1.0);
    let ideal_length = ((n_frames as usize - 1) * stride) + kernel_size - padding_total;
    ideal_length.saturating_sub(length)
}

fn load_conv1d_weight(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    groups: usize,
    cfg: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let in_per_group = in_channels
        .checked_div(groups)
        .ok_or_else(|| Error::ConfigError("Fish S2 DAC invalid conv1d group count".to_string()))?;
    let weight = if vb.contains_tensor("weight") {
        vb.get((out_channels, in_per_group, kernel_size), "weight")?
    } else if vb.contains_tensor("weight_g") && vb.contains_tensor("weight_v") {
        let weight_g = vb.get((out_channels, 1, 1), "weight_g")?;
        let weight_v = vb.get((out_channels, in_per_group, kernel_size), "weight_v")?;
        fuse_weight_norm_dim0(&weight_v, &weight_g)?
    } else if vb.contains_tensor("parametrizations.weight.original0") {
        let weight_g = vb.get((out_channels, 1, 1), "parametrizations.weight.original0")?;
        let weight_v = vb.get(
            (out_channels, in_per_group, kernel_size),
            "parametrizations.weight.original1",
        )?;
        fuse_weight_norm_dim0(&weight_v, &weight_g)?
    } else {
        return Err(Error::ModelLoadError(format!(
            "Fish S2 DAC conv1d missing weight at {}",
            vb.prefix()
        )));
    };
    let bias = if vb.contains_tensor("bias") {
        Some(vb.get((out_channels,), "bias")?)
    } else {
        None
    };
    Ok(Conv1d::new(weight, bias, cfg))
}

fn load_conv_transpose1d_weight(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: ConvTranspose1dConfig,
    vb: VarBuilder,
) -> Result<ConvTranspose1d> {
    let weight = if vb.contains_tensor("weight") {
        vb.get((in_channels, out_channels, kernel_size), "weight")?
    } else if vb.contains_tensor("weight_g") && vb.contains_tensor("weight_v") {
        let weight_g = vb.get((in_channels, 1, 1), "weight_g")?;
        let weight_v = vb.get((in_channels, out_channels, kernel_size), "weight_v")?;
        fuse_weight_norm_dim0(&weight_v, &weight_g)?
    } else if vb.contains_tensor("parametrizations.weight.original0") {
        let weight_g = vb.get((in_channels, 1, 1), "parametrizations.weight.original0")?;
        let weight_v = vb.get(
            (in_channels, out_channels, kernel_size),
            "parametrizations.weight.original1",
        )?;
        fuse_weight_norm_dim0(&weight_v, &weight_g)?
    } else {
        return Err(Error::ModelLoadError(format!(
            "Fish S2 DAC conv_transpose1d missing weight at {}",
            vb.prefix()
        )));
    };
    let bias = if vb.contains_tensor("bias") {
        Some(vb.get((out_channels,), "bias")?)
    } else {
        None
    };
    Ok(ConvTranspose1d::new(weight, bias, cfg))
}

fn apply_rope(x: &Tensor, cos_half: &Tensor, sin_half: &Tensor) -> Result<Tensor> {
    let half_dim = x.dim(3)? / 2;
    let x1 = x.narrow(3, 0, half_dim)?;
    let x2 = x.narrow(3, half_dim, half_dim)?;
    let cos = cos_half.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin_half.unsqueeze(0)?.unsqueeze(2)?;
    let out_first = x1
        .broadcast_mul(&cos)?
        .broadcast_sub(&x2.broadcast_mul(&sin)?)?;
    let out_second = x1
        .broadcast_mul(&sin)?
        .broadcast_add(&x2.broadcast_mul(&cos)?)?;
    Tensor::cat(&[&out_first, &out_second], 3).map_err(Error::from)
}

fn windowed_causal_mask(
    seq_len: usize,
    window_size: usize,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Tensor> {
    if window_size >= seq_len {
        return causal_mask(seq_len, seq_len, 0, device, dtype);
    }
    let mut values = Vec::with_capacity(seq_len * seq_len);
    for row in 0..seq_len {
        let min_col = row.saturating_add(1).saturating_sub(window_size);
        for col in 0..seq_len {
            let allowed = col <= row && col >= min_col;
            values.push(if allowed { 0.0 } else { f32::NEG_INFINITY });
        }
    }
    Tensor::from_vec(values, (1, seq_len, seq_len), device)?
        .to_dtype(dtype)
        .map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Shape};
    use std::collections::HashMap;

    fn tiny_config() -> FishS2DacConfig {
        FishS2DacConfig {
            sample_rate: 100,
            latent_dim: 4,
            decoder_dim: 4,
            decoder_rates: vec![1],
            downsample_factors: vec![1],
            codebook_dim: 2,
            semantic_codebook_size: 8,
            residual_codebook_size: 8,
            residual_codebooks: 1,
            transformer_layers: 1,
            transformer_heads: 2,
            transformer_kv_heads: 2,
            transformer_head_dim: 2,
            transformer_intermediate: 8,
            transformer_window_size: 4,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
        }
    }

    fn tensor(device: &Device, shape: impl Into<Shape>, value: f32) -> Tensor {
        Tensor::full(value, shape, device).unwrap()
    }

    fn insert_conv(
        tensors: &mut HashMap<String, Tensor>,
        device: &Device,
        prefix: &str,
        out_channels: usize,
        in_channels: usize,
        kernel: usize,
        value: f32,
    ) {
        tensors.insert(
            format!("{prefix}.conv.weight"),
            tensor(device, (out_channels, in_channels, kernel), value),
        );
        tensors.insert(
            format!("{prefix}.conv.bias"),
            tensor(device, (out_channels,), 0.0),
        );
    }

    fn insert_trans_conv(
        tensors: &mut HashMap<String, Tensor>,
        device: &Device,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        value: f32,
    ) {
        tensors.insert(
            format!("{prefix}.conv.weight"),
            tensor(device, (in_channels, out_channels, kernel), value),
        );
        tensors.insert(
            format!("{prefix}.conv.bias"),
            tensor(device, (out_channels,), 0.0),
        );
    }

    fn insert_snake(
        tensors: &mut HashMap<String, Tensor>,
        device: &Device,
        prefix: &str,
        channels: usize,
    ) {
        tensors.insert(
            format!("{prefix}.alpha"),
            tensor(device, (1, channels, 1), 1.0),
        );
    }

    fn insert_residual_unit(
        tensors: &mut HashMap<String, Tensor>,
        device: &Device,
        prefix: &str,
        dim: usize,
    ) {
        insert_snake(tensors, device, &format!("{prefix}.block.0"), dim);
        insert_conv(
            tensors,
            device,
            &format!("{prefix}.block.1"),
            dim,
            dim,
            7,
            0.0,
        );
        insert_snake(tensors, device, &format!("{prefix}.block.2"), dim);
        insert_conv(
            tensors,
            device,
            &format!("{prefix}.block.3"),
            dim,
            dim,
            1,
            0.0,
        );
    }

    fn insert_convnext(
        tensors: &mut HashMap<String, Tensor>,
        device: &Device,
        prefix: &str,
        dim: usize,
    ) {
        insert_conv(tensors, device, &format!("{prefix}.dwconv"), dim, 1, 7, 0.0);
        tensors.insert(format!("{prefix}.norm.weight"), tensor(device, (dim,), 1.0));
        tensors.insert(format!("{prefix}.norm.bias"), tensor(device, (dim,), 0.0));
        tensors.insert(
            format!("{prefix}.pwconv1.weight"),
            tensor(device, (dim * 4, dim), 0.0),
        );
        tensors.insert(
            format!("{prefix}.pwconv1.bias"),
            tensor(device, (dim * 4,), 0.0),
        );
        tensors.insert(
            format!("{prefix}.pwconv2.weight"),
            tensor(device, (dim, dim * 4), 0.0),
        );
        tensors.insert(
            format!("{prefix}.pwconv2.bias"),
            tensor(device, (dim,), 0.0),
        );
        tensors.insert(format!("{prefix}.gamma"), tensor(device, (dim,), 1.0));
    }

    fn tiny_decoder(device: &Device) -> FishS2DacDecoder {
        let config = tiny_config();
        let mut tensors = HashMap::new();
        for root in [
            "quantizer.semantic_quantizer.quantizers.0",
            "quantizer.quantizer.quantizers.0",
        ] {
            insert_conv(
                &mut tensors,
                device,
                &format!("{root}.in_proj"),
                2,
                4,
                1,
                0.0,
            );
            insert_conv(
                &mut tensors,
                device,
                &format!("{root}.out_proj"),
                4,
                2,
                1,
                0.01,
            );
            tensors.insert(
                format!("{root}.codebook.weight"),
                tensor(device, (8, 2), 0.5),
            );
        }

        tensors.insert(
            "quantizer.post_module.layers.0.attention.wqkv.weight".to_string(),
            tensor(device, (12, 4), 0.0),
        );
        tensors.insert(
            "quantizer.post_module.layers.0.attention.wo.weight".to_string(),
            tensor(device, (4, 4), 0.0),
        );
        tensors.insert(
            "quantizer.post_module.layers.0.feed_forward.w1.weight".to_string(),
            tensor(device, (8, 4), 0.0),
        );
        tensors.insert(
            "quantizer.post_module.layers.0.feed_forward.w2.weight".to_string(),
            tensor(device, (4, 8), 0.0),
        );
        tensors.insert(
            "quantizer.post_module.layers.0.feed_forward.w3.weight".to_string(),
            tensor(device, (8, 4), 0.0),
        );
        for name in [
            "quantizer.post_module.layers.0.attention_norm.weight",
            "quantizer.post_module.layers.0.ffn_norm.weight",
            "quantizer.post_module.layers.0.attention_layer_scale.gamma",
            "quantizer.post_module.layers.0.ffn_layer_scale.gamma",
            "quantizer.post_module.norm.weight",
        ] {
            tensors.insert(name.to_string(), tensor(device, (4,), 1.0));
        }

        insert_trans_conv(&mut tensors, device, "quantizer.upsample.0.0", 4, 4, 1, 0.0);
        insert_convnext(&mut tensors, device, "quantizer.upsample.0.1", 4);

        insert_conv(&mut tensors, device, "decoder.model.0", 4, 4, 7, 0.0);
        insert_snake(&mut tensors, device, "decoder.model.1.block.0", 4);
        insert_trans_conv(
            &mut tensors,
            device,
            "decoder.model.1.block.1",
            4,
            2,
            2,
            0.0,
        );
        for idx in 2..=4 {
            insert_residual_unit(
                &mut tensors,
                device,
                &format!("decoder.model.1.block.{idx}"),
                2,
            );
        }
        insert_snake(&mut tensors, device, "decoder.model.2", 2);
        insert_conv(&mut tensors, device, "decoder.model.3", 1, 2, 7, 0.0);

        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        FishS2DacDecoder::load(config, vb).unwrap()
    }

    #[test]
    fn dac_decode_returns_finite_waveform_shape() {
        let device = Device::Cpu;
        let decoder = tiny_decoder(&device);
        let codes = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let audio = decoder.decode_codebooks(&codes).unwrap();
        assert_eq!(audio.dims(), &[1, 1, 3]);
        let samples = audio.to_vec3::<f32>().unwrap();
        assert!(samples[0][0].iter().all(|value| value.is_finite()));
    }

    #[test]
    fn dac_decode_rejects_mismatched_codebook_lengths() {
        let device = Device::Cpu;
        let decoder = tiny_decoder(&device);
        let err = decoder
            .decode_codebooks(&[vec![0, 1], vec![2]])
            .unwrap_err();
        assert!(err.to_string().contains("same frame count"));
    }

    #[test]
    fn current_config_matches_contract_frame_size() {
        let config = FishS2DacConfig::current();
        assert_eq!(
            config.num_codebooks(),
            FishS2DacContract::CURRENT.total_codebooks()
        );
        assert_eq!(
            config.samples_per_frame().unwrap(),
            FishS2DacContract::CURRENT.frame_length().unwrap()
        );
        assert_eq!(config.sample_rate, FishS2DacContract::CURRENT.sample_rate);
    }
}
