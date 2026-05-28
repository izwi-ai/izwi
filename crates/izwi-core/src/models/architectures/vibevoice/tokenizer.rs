//! VibeVoice continuous speech tokenizer encoder/decoder.

use candle_core::{DType, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, LayerNorm, Linear, Module,
    RmsNorm, VarBuilder,
};

use crate::error::{Error, Result};
use crate::models::architectures::vibevoice::config::VibeVoiceTokenizerConfig;
use crate::models::shared::weights::mlx;

pub struct VibeVoiceTokenizerEncoderOutput {
    pub mean: Tensor,
    pub std: Option<f32>,
}

impl VibeVoiceTokenizerEncoderOutput {
    pub fn mode(&self) -> Tensor {
        self.mean.clone()
    }
}

#[derive(Default)]
pub struct VibeVoiceTokenizerStreamingCache {
    encoder: Option<TokenizerEncoderStreamingCache>,
    decoder: Option<TokenizerDecoderStreamingCache>,
}

impl VibeVoiceTokenizerStreamingCache {
    pub fn new() -> Self {
        Self::default()
    }

    fn encoder_mut(&mut self, encoder: &TokenizerEncoder) -> &mut TokenizerEncoderStreamingCache {
        self.encoder
            .get_or_insert_with(|| encoder.streaming_cache())
    }

    fn decoder_mut(&mut self, decoder: &TokenizerDecoder) -> &mut TokenizerDecoderStreamingCache {
        self.decoder
            .get_or_insert_with(|| decoder.streaming_cache())
    }
}

pub struct VibeVoiceAcousticTokenizer {
    encoder: TokenizerEncoder,
    decoder: TokenizerDecoder,
    fix_std: f32,
    std_dist_type: String,
    vae_dim: usize,
}

impl VibeVoiceAcousticTokenizer {
    pub fn load(config: &VibeVoiceTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let encoder_depths = config.encoder_depths_vec()?;
        let decoder_depths = config.decoder_depths_vec()?;
        let encoder = TokenizerEncoder::load(
            TokenizerStackConfig {
                dimension: config.vae_dim,
                channels: config.channels,
                n_filters: config.encoder_n_filters,
                ratios: config.encoder_ratios.clone(),
                depths: encoder_depths,
                causal: config.causal,
                kernel_size: config.kernel_size,
                last_kernel_size: config.last_kernel_size,
                layernorm: config.layernorm.clone(),
                layernorm_eps: config.layernorm_eps,
                disable_last_norm: config.disable_last_norm,
                mixer_layer: config.mixer_layer.clone(),
                layer_scale_init_value: config.layer_scale_init_value,
                conv_bias: config.conv_bias,
                pad_mode: config.pad_mode.clone(),
                trim_right_ratio: config.trim_right_ratio,
            },
            vb.pp("encoder"),
        )?;
        let decoder = TokenizerDecoder::load(
            TokenizerStackConfig {
                dimension: config.vae_dim,
                channels: config.channels,
                n_filters: config.decoder_n_filters,
                ratios: config.decoder_ratios_vec(),
                depths: decoder_depths,
                causal: config.causal,
                kernel_size: config.kernel_size,
                last_kernel_size: config.last_kernel_size,
                layernorm: config.layernorm.clone(),
                layernorm_eps: config.layernorm_eps,
                disable_last_norm: config.disable_last_norm,
                mixer_layer: config.mixer_layer.clone(),
                layer_scale_init_value: config.layer_scale_init_value,
                conv_bias: config.conv_bias,
                pad_mode: config.pad_mode.clone(),
                trim_right_ratio: config.trim_right_ratio,
            },
            vb.pp("decoder"),
        )?;
        Ok(Self {
            encoder,
            decoder,
            fix_std: config.fix_std,
            std_dist_type: config.std_dist_type.clone(),
            vae_dim: config.vae_dim,
        })
    }

    pub fn encode(&self, audio: &Tensor) -> Result<VibeVoiceTokenizerEncoderOutput> {
        let latents = self.encoder.forward(audio)?;
        Ok(VibeVoiceTokenizerEncoderOutput {
            mean: latents.transpose(1, 2)?,
            std: Some(self.fix_std),
        })
    }

    pub fn sample(&self, output: &VibeVoiceTokenizerEncoderOutput) -> Result<Tensor> {
        if self.std_dist_type == "none" || self.fix_std == 0.0 {
            return Ok(output.mean.clone());
        }
        let noise = Tensor::randn(0f32, 1f32, output.mean.shape(), output.mean.device())?
            .to_dtype(output.mean.dtype())?;
        let scale =
            Tensor::new(self.fix_std, output.mean.device())?.to_dtype(output.mean.dtype())?;
        output
            .mean
            .broadcast_add(&noise.broadcast_mul(&scale)?)
            .map_err(Error::from)
    }

    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let latents = if latents.dim(1)? == self.vae_dim {
            latents.clone()
        } else {
            latents.transpose(1, 2)?
        };
        self.decoder.forward(&latents)
    }

    pub fn decode_streaming(
        &self,
        latents: &Tensor,
        cache: &mut VibeVoiceTokenizerStreamingCache,
    ) -> Result<Tensor> {
        let latents = if latents.dim(1)? == self.vae_dim {
            latents.clone()
        } else {
            latents.transpose(1, 2)?
        };
        self.decoder
            .forward_streaming(&latents, cache.decoder_mut(&self.decoder))
    }
}

pub struct VibeVoiceSemanticTokenizer {
    encoder: TokenizerEncoder,
}

impl VibeVoiceSemanticTokenizer {
    pub fn load(config: &VibeVoiceTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = TokenizerEncoder::load(
            TokenizerStackConfig {
                dimension: config.vae_dim,
                channels: config.channels,
                n_filters: config.encoder_n_filters,
                ratios: config.encoder_ratios.clone(),
                depths: config.encoder_depths_vec()?,
                causal: config.causal,
                kernel_size: config.kernel_size,
                last_kernel_size: config.last_kernel_size,
                layernorm: config.layernorm.clone(),
                layernorm_eps: config.layernorm_eps,
                disable_last_norm: config.disable_last_norm,
                mixer_layer: config.mixer_layer.clone(),
                layer_scale_init_value: config.layer_scale_init_value,
                conv_bias: config.conv_bias,
                pad_mode: config.pad_mode.clone(),
                trim_right_ratio: config.trim_right_ratio,
            },
            vb.pp("encoder"),
        )?;
        Ok(Self { encoder })
    }

    pub fn encode(&self, audio: &Tensor) -> Result<VibeVoiceTokenizerEncoderOutput> {
        let latents = self.encoder.forward(audio)?;
        Ok(VibeVoiceTokenizerEncoderOutput {
            mean: latents.transpose(1, 2)?,
            std: None,
        })
    }

    pub fn encode_streaming(
        &self,
        audio: &Tensor,
        cache: &mut VibeVoiceTokenizerStreamingCache,
    ) -> Result<VibeVoiceTokenizerEncoderOutput> {
        let latents = self
            .encoder
            .forward_streaming(audio, cache.encoder_mut(&self.encoder))?;
        Ok(VibeVoiceTokenizerEncoderOutput {
            mean: latents.transpose(1, 2)?,
            std: None,
        })
    }
}

#[derive(Clone)]
struct TokenizerStackConfig {
    dimension: usize,
    channels: usize,
    n_filters: usize,
    ratios: Vec<usize>,
    depths: Vec<usize>,
    causal: bool,
    kernel_size: usize,
    last_kernel_size: usize,
    layernorm: String,
    layernorm_eps: f64,
    disable_last_norm: bool,
    mixer_layer: String,
    layer_scale_init_value: f32,
    conv_bias: bool,
    pad_mode: String,
    trim_right_ratio: f32,
}

struct TokenizerEncoder {
    downsample_layers: Vec<SConv1d>,
    stages: Vec<Vec<Block1D>>,
    norm: ConvNorm,
    head: SConv1d,
}

impl TokenizerEncoder {
    fn load(config: TokenizerStackConfig, vb: VarBuilder) -> Result<Self> {
        validate_constant_padding(&config.pad_mode)?;
        let mut ratios = config.ratios.clone();
        ratios.reverse();
        let mut downsample_layers = Vec::with_capacity(ratios.len() + 1);
        downsample_layers.push(SConv1d::load(
            config.channels,
            config.n_filters,
            config.kernel_size,
            1,
            1,
            1,
            config.conv_bias,
            config.causal,
            vb.pp("downsample_layers.0.0"),
        )?);
        for (idx, ratio) in ratios.iter().copied().enumerate() {
            let in_ch = config.n_filters * (1usize << idx);
            let out_ch = config.n_filters * (1usize << (idx + 1));
            downsample_layers.push(SConv1d::load(
                in_ch,
                out_ch,
                ratio * 2,
                ratio,
                1,
                1,
                config.conv_bias,
                config.causal,
                vb.pp(format!("downsample_layers.{}.0", idx + 1)),
            )?);
        }

        let mut stages = Vec::with_capacity(config.depths.len());
        for (stage_idx, depth) in config.depths.iter().copied().enumerate() {
            let dim = config.n_filters * (1usize << stage_idx);
            let mut blocks = Vec::with_capacity(depth);
            for block_idx in 0..depth {
                blocks.push(Block1D::load(
                    dim,
                    &config,
                    vb.pp(format!("stages.{stage_idx}.{block_idx}")),
                )?);
            }
            stages.push(blocks);
        }
        let last_dim = config.n_filters * (1usize << config.depths.len().saturating_sub(1));
        let norm = if config.disable_last_norm {
            ConvNorm::Identity
        } else {
            ConvNorm::load(
                last_dim,
                &config.layernorm,
                config.layernorm_eps,
                vb.pp("norm"),
            )?
        };
        let head = SConv1d::load(
            last_dim,
            config.dimension,
            config.last_kernel_size,
            1,
            1,
            1,
            config.conv_bias,
            config.causal,
            vb.pp("head"),
        )?;
        Ok(Self {
            downsample_layers,
            stages,
            norm,
            head,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for idx in 0..self.stages.len() {
            x = self.downsample_layers[idx].forward(&x)?;
            for block in &self.stages[idx] {
                x = block.forward(&x)?;
            }
        }
        let x = self.norm.forward(&x)?;
        self.head.forward(&x)
    }

    fn forward_streaming(
        &self,
        x: &Tensor,
        cache: &mut TokenizerEncoderStreamingCache,
    ) -> Result<Tensor> {
        validate_cache_len(
            "VibeVoice tokenizer encoder downsample",
            cache.downsample_layers.len(),
            self.downsample_layers.len(),
        )?;
        validate_cache_len(
            "VibeVoice tokenizer encoder stages",
            cache.stages.len(),
            self.stages.len(),
        )?;
        let mut x = x.clone();
        for idx in 0..self.stages.len() {
            x = self.downsample_layers[idx]
                .forward_streaming(&x, &mut cache.downsample_layers[idx])?;
            validate_cache_len(
                "VibeVoice tokenizer encoder stage blocks",
                cache.stages[idx].len(),
                self.stages[idx].len(),
            )?;
            for (block, block_cache) in self.stages[idx].iter().zip(cache.stages[idx].iter_mut()) {
                x = block.forward_streaming(&x, block_cache)?;
            }
        }
        let x = self.norm.forward(&x)?;
        self.head.forward_streaming(&x, &mut cache.head)
    }

    fn streaming_cache(&self) -> TokenizerEncoderStreamingCache {
        TokenizerEncoderStreamingCache {
            downsample_layers: self
                .downsample_layers
                .iter()
                .map(SConv1d::streaming_cache)
                .collect(),
            stages: self
                .stages
                .iter()
                .map(|stage| stage.iter().map(Block1D::streaming_cache).collect())
                .collect(),
            head: self.head.streaming_cache(),
        }
    }
}

struct TokenizerDecoder {
    upsample_layers: Vec<UpsampleLayer>,
    stages: Vec<Vec<Block1D>>,
    norm: ConvNorm,
    head: SConv1d,
}

impl TokenizerDecoder {
    fn load(config: TokenizerStackConfig, vb: VarBuilder) -> Result<Self> {
        validate_constant_padding(&config.pad_mode)?;
        let mut upsample_layers = Vec::with_capacity(config.ratios.len() + 1);
        let top_dim = config.n_filters * (1usize << config.depths.len().saturating_sub(1));
        upsample_layers.push(UpsampleLayer::Conv(SConv1d::load(
            config.dimension,
            top_dim,
            config.kernel_size,
            1,
            1,
            1,
            config.conv_bias,
            config.causal,
            vb.pp("upsample_layers.0.0"),
        )?));
        for (idx, ratio) in config.ratios.iter().copied().enumerate() {
            let in_ch =
                config.n_filters * (1usize << (config.depths.len().saturating_sub(1 + idx)));
            let out_ch =
                config.n_filters * (1usize << (config.depths.len().saturating_sub(2 + idx)));
            upsample_layers.push(UpsampleLayer::Transposed(SConvTranspose1d::load(
                in_ch,
                out_ch,
                ratio * 2,
                ratio,
                config.causal,
                config.trim_right_ratio,
                vb.pp(format!("upsample_layers.{}.0", idx + 1)),
            )?));
        }

        let mut stages = Vec::with_capacity(config.depths.len());
        for (stage_idx, depth) in config.depths.iter().copied().enumerate() {
            let dim =
                config.n_filters * (1usize << (config.depths.len().saturating_sub(1 + stage_idx)));
            let mut blocks = Vec::with_capacity(depth);
            for block_idx in 0..depth {
                blocks.push(Block1D::load(
                    dim,
                    &config,
                    vb.pp(format!("stages.{stage_idx}.{block_idx}")),
                )?);
            }
            stages.push(blocks);
        }
        let last_dim = config.n_filters;
        let norm = if config.disable_last_norm {
            ConvNorm::Identity
        } else {
            ConvNorm::load(
                last_dim,
                &config.layernorm,
                config.layernorm_eps,
                vb.pp("norm"),
            )?
        };
        let head = SConv1d::load(
            last_dim,
            config.channels,
            config.last_kernel_size,
            1,
            1,
            1,
            config.conv_bias,
            config.causal,
            vb.pp("head"),
        )?;
        Ok(Self {
            upsample_layers,
            stages,
            norm,
            head,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for idx in 0..self.stages.len() {
            x = self.upsample_layers[idx].forward(&x)?;
            for block in &self.stages[idx] {
                x = block.forward(&x)?;
            }
        }
        let x = self.norm.forward(&x)?;
        self.head.forward(&x)
    }

    fn forward_streaming(
        &self,
        x: &Tensor,
        cache: &mut TokenizerDecoderStreamingCache,
    ) -> Result<Tensor> {
        validate_cache_len(
            "VibeVoice tokenizer decoder upsample",
            cache.upsample_layers.len(),
            self.upsample_layers.len(),
        )?;
        validate_cache_len(
            "VibeVoice tokenizer decoder stages",
            cache.stages.len(),
            self.stages.len(),
        )?;
        let mut x = x.clone();
        for idx in 0..self.stages.len() {
            x = self.upsample_layers[idx].forward_streaming(&x, &mut cache.upsample_layers[idx])?;
            validate_cache_len(
                "VibeVoice tokenizer decoder stage blocks",
                cache.stages[idx].len(),
                self.stages[idx].len(),
            )?;
            for (block, block_cache) in self.stages[idx].iter().zip(cache.stages[idx].iter_mut()) {
                x = block.forward_streaming(&x, block_cache)?;
            }
        }
        let x = self.norm.forward(&x)?;
        self.head.forward_streaming(&x, &mut cache.head)
    }

    fn streaming_cache(&self) -> TokenizerDecoderStreamingCache {
        TokenizerDecoderStreamingCache {
            upsample_layers: self
                .upsample_layers
                .iter()
                .map(UpsampleLayer::streaming_cache)
                .collect(),
            stages: self
                .stages
                .iter()
                .map(|stage| stage.iter().map(Block1D::streaming_cache).collect())
                .collect(),
            head: self.head.streaming_cache(),
        }
    }
}

enum UpsampleLayer {
    Conv(SConv1d),
    Transposed(SConvTranspose1d),
}

impl UpsampleLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Conv(layer) => layer.forward(x),
            Self::Transposed(layer) => layer.forward(x),
        }
    }

    fn forward_streaming(
        &self,
        x: &Tensor,
        cache: &mut UpsampleLayerStreamingCache,
    ) -> Result<Tensor> {
        match (self, cache) {
            (Self::Conv(layer), UpsampleLayerStreamingCache::Conv(cache)) => {
                layer.forward_streaming(x, cache)
            }
            (Self::Transposed(layer), UpsampleLayerStreamingCache::Transposed(cache)) => {
                layer.forward_streaming(x, cache)
            }
            _ => Err(Error::InferenceError(
                "VibeVoice tokenizer streaming cache layer type mismatch".to_string(),
            )),
        }
    }

    fn streaming_cache(&self) -> UpsampleLayerStreamingCache {
        match self {
            Self::Conv(layer) => UpsampleLayerStreamingCache::Conv(layer.streaming_cache()),
            Self::Transposed(layer) => {
                UpsampleLayerStreamingCache::Transposed(layer.streaming_cache())
            }
        }
    }
}

struct Block1D {
    norm: ConvNorm,
    mixer: SConv1d,
    ffn_norm: ConvNorm,
    linear1: Linear,
    linear2: Linear,
    gamma: Option<Tensor>,
    ffn_gamma: Option<Tensor>,
}

impl Block1D {
    fn load(dim: usize, config: &TokenizerStackConfig, vb: VarBuilder) -> Result<Self> {
        let norm = ConvNorm::load(dim, &config.layernorm, config.layernorm_eps, vb.pp("norm"))?;
        let ffn_norm = ConvNorm::load(
            dim,
            &config.layernorm,
            config.layernorm_eps,
            vb.pp("ffn_norm"),
        )?;
        let groups = if config.mixer_layer == "depthwise_conv" {
            dim
        } else {
            1
        };
        let mixer = SConv1d::load(
            dim,
            dim,
            config.kernel_size,
            1,
            1,
            groups,
            config.conv_bias,
            config.causal,
            vb.pp("mixer.conv"),
        )?;
        let ffn_dim = 4 * dim;
        let linear1 = mlx::load_linear_no_bias(dim, ffn_dim, vb.pp("ffn.linear1"))?;
        let linear2 = mlx::load_linear_no_bias(ffn_dim, dim, vb.pp("ffn.linear2"))?;
        let gamma = (config.layer_scale_init_value > 0.0)
            .then(|| vb.get((dim,), "gamma"))
            .transpose()?;
        let ffn_gamma = (config.layer_scale_init_value > 0.0)
            .then(|| vb.get((dim,), "ffn_gamma"))
            .transpose()?;
        Ok(Self {
            norm,
            mixer,
            ffn_norm,
            linear1,
            linear2,
            gamma,
            ffn_gamma,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let mut mixed = self.mixer.forward(&self.norm.forward(x)?)?;
        if let Some(gamma) = &self.gamma {
            mixed = mixed.broadcast_mul(&gamma.reshape((1, gamma.dim(0)?, 1))?)?;
        }
        let x = residual.broadcast_add(&mixed)?;

        let residual = &x;
        let ffn_in = self.ffn_norm.forward(&x)?.transpose(1, 2)?;
        let hidden = self.linear1.forward(&ffn_in)?.gelu()?;
        let mut hidden = self.linear2.forward(&hidden)?.transpose(1, 2)?;
        if let Some(gamma) = &self.ffn_gamma {
            hidden = hidden.broadcast_mul(&gamma.reshape((1, gamma.dim(0)?, 1))?)?;
        }
        residual.broadcast_add(&hidden).map_err(Error::from)
    }

    fn forward_streaming(&self, x: &Tensor, cache: &mut Block1DStreamingCache) -> Result<Tensor> {
        let residual = x;
        let normed = self.norm.forward(x)?;
        let mut mixed = self.mixer.forward_streaming(&normed, &mut cache.mixer)?;
        if let Some(gamma) = &self.gamma {
            mixed = mixed.broadcast_mul(&gamma.reshape((1, gamma.dim(0)?, 1))?)?;
        }
        let x = residual.broadcast_add(&mixed)?;

        let residual = &x;
        let ffn_in = self.ffn_norm.forward(&x)?.transpose(1, 2)?;
        let hidden = self.linear1.forward(&ffn_in)?.gelu()?;
        let mut hidden = self.linear2.forward(&hidden)?.transpose(1, 2)?;
        if let Some(gamma) = &self.ffn_gamma {
            hidden = hidden.broadcast_mul(&gamma.reshape((1, gamma.dim(0)?, 1))?)?;
        }
        residual.broadcast_add(&hidden).map_err(Error::from)
    }

    fn streaming_cache(&self) -> Block1DStreamingCache {
        Block1DStreamingCache {
            mixer: self.mixer.streaming_cache(),
        }
    }
}

struct SConv1d {
    conv: Conv1d,
    causal_padding: usize,
    right_padding: usize,
}

impl SConv1d {
    #[allow(clippy::too_many_arguments)]
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        with_bias: bool,
        causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups,
            ..Default::default()
        };
        let conv_vb = vb.pp("conv.conv");
        let conv = if with_bias {
            candle_nn::conv1d(in_channels, out_channels, kernel_size, cfg, conv_vb)?
        } else {
            candle_nn::conv1d_no_bias(in_channels, out_channels, kernel_size, cfg, conv_vb)?
        };
        let padding_total = dilation * (kernel_size - 1) - stride.saturating_sub(1);
        let (causal_padding, right_padding) = if causal {
            (padding_total, 0)
        } else {
            let right = padding_total / 2;
            (padding_total - right, right)
        };
        Ok(Self {
            conv,
            causal_padding,
            right_padding,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = if self.causal_padding > 0 || self.right_padding > 0 {
            x.pad_with_zeros(2, self.causal_padding, self.right_padding)?
        } else {
            x.clone()
        };
        self.conv.forward(&x).map_err(Error::from)
    }

    fn forward_streaming(&self, x: &Tensor, cache: &mut SConv1dStreamingCache) -> Result<Tensor> {
        if self.right_padding > 0 {
            return Err(Error::InferenceError(
                "VibeVoice tokenizer streaming cache requires causal SConv1d padding".to_string(),
            ));
        }
        let stride = self.conv.config().stride.max(1);
        if stride > 1 && x.dim(2)? % stride != 0 {
            return Err(Error::InferenceError(format!(
                "VibeVoice tokenizer streaming SConv1d chunk length {} is not aligned to stride {stride}",
                x.dim(2)?
            )));
        }
        let padded = if self.causal_padding > 0 {
            let prefix = cache.prefix_or_zeros(x, self.causal_padding)?;
            Tensor::cat(&[prefix, x.clone()], 2)?
        } else {
            x.clone()
        };
        let output = self.conv.forward(&padded)?;
        cache.update_from_padded_input(&padded, self.causal_padding)?;
        Ok(output)
    }

    fn streaming_cache(&self) -> SConv1dStreamingCache {
        SConv1dStreamingCache::default()
    }
}

struct SConvTranspose1d {
    conv: ConvTranspose1d,
    left_trim: usize,
    right_trim: usize,
}

impl SConvTranspose1d {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        causal: bool,
        trim_right_ratio: f32,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride,
            dilation: 1,
            groups: 1,
        };
        let conv = candle_nn::conv_transpose1d(
            in_channels,
            out_channels,
            kernel_size,
            cfg,
            vb.pp("convtr.convtr"),
        )?;
        let padding_total = kernel_size.saturating_sub(stride);
        let (left_trim, right_trim) = if causal {
            let right = ((padding_total as f32) * trim_right_ratio).ceil() as usize;
            (padding_total.saturating_sub(right), right)
        } else {
            let right = padding_total / 2;
            (padding_total - right, right)
        };
        Ok(Self {
            conv,
            left_trim,
            right_trim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.conv.forward(x)?;
        let len = y.dim(2)?;
        let keep = len.saturating_sub(self.left_trim + self.right_trim);
        y.narrow(2, self.left_trim.min(len), keep)
            .map_err(Error::from)
    }

    fn forward_streaming(
        &self,
        x: &Tensor,
        cache: &mut SConvTranspose1dStreamingCache,
    ) -> Result<Tensor> {
        if self.left_trim > 0 {
            return Err(Error::InferenceError(
                "VibeVoice tokenizer streaming cache requires causal transposed-conv trim"
                    .to_string(),
            ));
        }
        let mut y = self.conv.forward(x)?;
        if let Some(tail) = cache.tail.take() {
            y = add_transposed_overlap(y, tail, self.conv.bias())?;
        }
        let len = y.dim(2)?;
        let emit_len = len.saturating_sub(self.right_trim);
        let emitted = y.narrow(2, 0, emit_len)?;
        cache.tail = if self.right_trim > 0 {
            Some(y.narrow(2, emit_len, len - emit_len)?)
        } else {
            None
        };
        Ok(emitted)
    }

    fn streaming_cache(&self) -> SConvTranspose1dStreamingCache {
        SConvTranspose1dStreamingCache::default()
    }
}

#[derive(Default)]
struct TokenizerEncoderStreamingCache {
    downsample_layers: Vec<SConv1dStreamingCache>,
    stages: Vec<Vec<Block1DStreamingCache>>,
    head: SConv1dStreamingCache,
}

#[derive(Default)]
struct TokenizerDecoderStreamingCache {
    upsample_layers: Vec<UpsampleLayerStreamingCache>,
    stages: Vec<Vec<Block1DStreamingCache>>,
    head: SConv1dStreamingCache,
}

enum UpsampleLayerStreamingCache {
    Conv(SConv1dStreamingCache),
    Transposed(SConvTranspose1dStreamingCache),
}

struct Block1DStreamingCache {
    mixer: SConv1dStreamingCache,
}

#[derive(Default)]
struct SConv1dStreamingCache {
    prefix: Option<Tensor>,
}

impl SConv1dStreamingCache {
    fn prefix_or_zeros(&mut self, x: &Tensor, len: usize) -> Result<Tensor> {
        if let Some(prefix) = &self.prefix {
            return Ok(prefix.clone());
        }
        Tensor::zeros((x.dim(0)?, x.dim(1)?, len), x.dtype(), x.device()).map_err(Error::from)
    }

    fn update_from_padded_input(&mut self, x: &Tensor, len: usize) -> Result<()> {
        if len == 0 {
            self.prefix = None;
            return Ok(());
        }
        let input_len = x.dim(2)?;
        let start = input_len.saturating_sub(len);
        self.prefix = Some(x.narrow(2, start, input_len - start)?);
        Ok(())
    }
}

#[derive(Default)]
struct SConvTranspose1dStreamingCache {
    tail: Option<Tensor>,
}

fn add_transposed_overlap(y: Tensor, tail: Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let y_len = y.dim(2)?;
    let tail_len = tail.dim(2)?;
    if tail_len == 0 {
        return Ok(y);
    }
    if y_len < tail_len {
        return Err(Error::InferenceError(format!(
            "VibeVoice transposed-conv streaming overlap length {tail_len} exceeds chunk length {y_len}"
        )));
    }
    let mut overlap = y.narrow(2, 0, tail_len)?.broadcast_add(&tail)?;
    if let Some(bias) = bias {
        overlap = overlap.broadcast_sub(&bias.reshape((1, bias.dim(0)?, 1))?)?;
    }
    let mut parts = vec![overlap];
    if tail_len < y_len {
        parts.push(y.narrow(2, tail_len, y_len - tail_len)?);
    }
    Tensor::cat(&parts, 2).map_err(Error::from)
}

fn validate_cache_len(context: &str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        return Ok(());
    }
    Err(Error::InferenceError(format!(
        "{context} streaming cache has {actual} entries, expected {expected}"
    )))
}

enum ConvNorm {
    Rms(RmsNorm),
    Layer(LayerNorm),
    Identity,
}

impl ConvNorm {
    fn load(dim: usize, kind: &str, eps: f64, vb: VarBuilder) -> Result<Self> {
        match kind {
            "LN" => Ok(Self::Layer(candle_nn::layer_norm(dim, eps, vb)?)),
            "RMSNorm" => Ok(Self::Rms(candle_nn::rms_norm(dim, eps, vb)?)),
            other => Err(Error::ModelLoadError(format!(
                "Unsupported VibeVoice tokenizer norm type: {other}"
            ))),
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Rms(norm) => {
                let x = x.transpose(1, 2)?;
                let y = norm.forward(&x)?;
                y.transpose(1, 2).map_err(Error::from)
            }
            Self::Layer(norm) => {
                let x = x.transpose(1, 2)?;
                let y = norm.forward(&x)?;
                y.transpose(1, 2).map_err(Error::from)
            }
            Self::Identity => Ok(x.clone()),
        }
    }
}

fn validate_constant_padding(pad_mode: &str) -> Result<()> {
    match pad_mode {
        "constant" | "zeros" => Ok(()),
        other => Err(Error::ModelLoadError(format!(
            "VibeVoice tokenizer pad mode '{other}' is not implemented in the Candle loader"
        ))),
    }
}

#[allow(dead_code)]
fn _dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        _ => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use candle_core::Device;
    use candle_nn::VarBuilder;

    #[test]
    fn semantic_encode_shape_matches_continuous_latent_contract() {
        let output = VibeVoiceTokenizerEncoderOutput {
            mean: Tensor::zeros((1, 3, 128), DType::F32, &candle_core::Device::Cpu).unwrap(),
            std: None,
        };
        assert_eq!(output.mode().dims(), &[1, 3, 128]);
    }

    #[test]
    fn causal_sconv1d_streaming_matches_full_strided_forward() {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "conv.conv.weight".to_string(),
            Tensor::from_vec(vec![0.25f32, -0.5, 0.75, 0.125], (1, 1, 4), &device).unwrap(),
        );
        tensors.insert(
            "conv.conv.bias".to_string(),
            Tensor::from_vec(vec![0.1f32], (1,), &device).unwrap(),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let layer = SConv1d::load(1, 1, 4, 2, 1, 1, true, true, vb).unwrap();
        let x = Tensor::from_vec(
            (0..12).map(|value| value as f32 / 10.0).collect::<Vec<_>>(),
            (1, 1, 12),
            &device,
        )
        .unwrap();

        let full = layer.forward(&x).unwrap();
        let mut cache = layer.streaming_cache();
        let mut chunks = Vec::new();
        for offset in [0usize, 4, 8] {
            let chunk = x.narrow(2, offset, 4).unwrap();
            chunks.push(layer.forward_streaming(&chunk, &mut cache).unwrap());
        }
        let streamed = Tensor::cat(&chunks, 2).unwrap();

        assert_tensor_close(&streamed, &full, 1e-6);
    }

    #[test]
    fn causal_sconvtranspose_streaming_matches_full_overlap_forward() {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "convtr.convtr.weight".to_string(),
            Tensor::from_vec(vec![0.2f32, -0.4, 0.6, 0.8], (1, 1, 4), &device).unwrap(),
        );
        tensors.insert(
            "convtr.convtr.bias".to_string(),
            Tensor::from_vec(vec![0.3f32], (1,), &device).unwrap(),
        );
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let layer = SConvTranspose1d::load(1, 1, 4, 2, true, 1.0, vb).unwrap();
        let x = Tensor::from_vec(vec![0.5f32, -1.0, 0.25, 0.75, -0.5], (1, 1, 5), &device).unwrap();

        let full = layer.forward(&x).unwrap();
        let mut cache = layer.streaming_cache();
        let chunks = [
            x.narrow(2, 0, 1).unwrap(),
            x.narrow(2, 1, 2).unwrap(),
            x.narrow(2, 3, 2).unwrap(),
        ];
        let streamed_chunks = chunks
            .iter()
            .map(|chunk| layer.forward_streaming(chunk, &mut cache).unwrap())
            .collect::<Vec<_>>();
        let streamed = Tensor::cat(&streamed_chunks, 2).unwrap();

        assert_tensor_close(&streamed, &full, 1e-6);
    }

    fn assert_tensor_close(actual: &Tensor, expected: &Tensor, epsilon: f32) {
        assert_eq!(actual.dims(), expected.dims());
        let actual = actual.to_vec3::<f32>().unwrap();
        let expected = expected.to_vec3::<f32>().unwrap();
        for (actual_batch, expected_batch) in actual.iter().zip(expected.iter()) {
            for (actual_channel, expected_channel) in actual_batch.iter().zip(expected_batch.iter())
            {
                for (actual, expected) in actual_channel.iter().zip(expected_channel.iter()) {
                    assert!(
                        (*actual - *expected).abs() <= epsilon,
                        "expected {actual} to be within {epsilon} of {expected}"
                    );
                }
            }
        }
    }
}
