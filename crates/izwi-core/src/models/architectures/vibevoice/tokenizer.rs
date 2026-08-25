//! VibeVoice continuous speech tokenizer encoder/decoder.

use candle_core::{DType, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, LayerNorm, Linear, Module,
    RmsNorm, VarBuilder,
};

use crate::backends::state::{
    InvocationTensorComponentSlice, InvocationTensorComponentValue, InvocationTensorUpdateV2,
    PhysicalStateTransactionId, StateComponentValue, StateDomainSnapshot, TensorStateArena,
};
use crate::engine::InvocationTensorLease;
use crate::error::{Error, Result};
use crate::kv::v2::{
    ComponentShapeInstantiation, DomainStepIntent, ShapeAxis, ShapeDimensionValue,
    StateComponentId, StateDomainId, StateUpdateKind,
};
use crate::models::architectures::vibevoice::config::VibeVoiceTokenizerConfig;
use crate::models::shared::weights::mlx;

pub struct VibeVoiceTokenizerEncoderOutput {
    pub mean: Tensor,
    pub std: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct VibeVoiceTokenizerStateComponentGeometry {
    pub(crate) channels: usize,
    pub(crate) frames: usize,
}

impl VibeVoiceTokenizerEncoderOutput {
    pub fn mode(&self) -> Tensor {
        self.mean.clone()
    }
}

#[derive(Default)]
struct TokenizerStreamingState {
    encoder: Option<TokenizerEncoderStreamingCache>,
    decoder: Option<TokenizerDecoderStreamingCache>,
}

impl TokenizerStreamingState {
    fn new() -> Self {
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

    pub(crate) fn encoder_state_geometry(&self) -> Vec<VibeVoiceTokenizerStateComponentGeometry> {
        self.encoder.streaming_state_geometry()
    }

    pub(crate) fn decoder_state_geometry(&self) -> Vec<VibeVoiceTokenizerStateComponentGeometry> {
        self.decoder.streaming_state_geometry()
    }

    fn encode_streaming(
        &self,
        audio: &Tensor,
        cache: &mut TokenizerStreamingState,
    ) -> Result<VibeVoiceTokenizerEncoderOutput> {
        let latents = self
            .encoder
            .forward_streaming(audio, cache.encoder_mut(&self.encoder))?;
        Ok(VibeVoiceTokenizerEncoderOutput {
            mean: latents.transpose(1, 2)?,
            std: Some(self.fix_std),
        })
    }

    pub(crate) fn encode_streaming_physical(
        &self,
        audio: &Tensor,
        domain: StateDomainId,
        advance_samples: u64,
        lease: &mut InvocationTensorLease,
    ) -> Result<VibeVoiceTokenizerEncoderOutput> {
        let mut cache = TokenizerStreamingState::new();
        let expected_cursor = hydrate_encoder_cache(lease, &self.encoder, &mut cache)?;
        let output = self.encode_streaming(audio, &mut cache)?;
        commit_encoder_cache(
            lease,
            domain,
            expected_cursor,
            advance_samples,
            &self.encoder,
            cache.encoder.as_ref().ok_or_else(|| {
                Error::InferenceError(
                    "VibeVoice acoustic encoder did not initialize physical state".into(),
                )
            })?,
        )?;
        Ok(output)
    }

    /// Execute one exact scheduler-selected audio span against retained tensor
    /// state. The transaction remains private to the caller; the tokenizer
    /// only stages its domain and never seals or publishes the consistency
    /// group on its own.
    pub(crate) fn encode_streaming_retained(
        &self,
        audio: &Tensor,
        domain: StateDomainId,
        expected_cursor: u64,
        target_cursor: u64,
        transaction: PhysicalStateTransactionId,
        arena: &TensorStateArena,
    ) -> Result<VibeVoiceTokenizerEncoderOutput> {
        let mut output = encode_encoder_span_retained(
            &self.encoder,
            audio,
            domain,
            expected_cursor,
            target_cursor,
            transaction,
            arena,
            "acoustic",
        )?;
        output.std = Some(self.fix_std);
        Ok(output)
    }

    pub(crate) fn requires_sampling_noise(&self) -> bool {
        self.std_dist_type != "none" && self.fix_std != 0.0
    }

    /// Sample with request-owned noise so retries and native batch membership
    /// cannot change a row's stochastic acoustic latent.
    pub(crate) fn sample_with_supplied_noise(
        &self,
        output: &VibeVoiceTokenizerEncoderOutput,
        noise: Option<&Tensor>,
    ) -> Result<Tensor> {
        if !self.requires_sampling_noise() {
            if noise.is_some() {
                return Err(Error::InvalidInput(
                    "VibeVoice deterministic acoustic sampling received unexpected noise".into(),
                ));
            }
            return Ok(output.mean.clone());
        }
        if output.std != Some(self.fix_std) {
            return Err(Error::InferenceError(
                "VibeVoice acoustic sampling output has stale standard deviation".into(),
            ));
        }
        let noise = noise.ok_or_else(|| {
            Error::InvalidInput(
                "VibeVoice stochastic acoustic sampling requires supplied noise".into(),
            )
        })?;
        if noise.dims() != output.mean.dims()
            || noise.dtype() != output.mean.dtype()
            || !noise.device().same_device(output.mean.device())
        {
            return Err(Error::InvalidInput(format!(
                "VibeVoice acoustic sampling noise {:?}/{:?}/{:?} does not match mean {:?}/{:?}/{:?}",
                noise.dims(),
                noise.dtype(),
                noise.device().location(),
                output.mean.dims(),
                output.mean.dtype(),
                output.mean.device().location()
            )));
        }
        let scale =
            Tensor::new(self.fix_std, output.mean.device())?.to_dtype(output.mean.dtype())?;
        output
            .mean
            .broadcast_add(&noise.broadcast_mul(&scale)?)
            .map_err(Error::from)
    }

    pub fn sample(&self, output: &VibeVoiceTokenizerEncoderOutput) -> Result<Tensor> {
        let noise = self
            .requires_sampling_noise()
            .then(|| {
                Tensor::randn(0f32, 1f32, output.mean.shape(), output.mean.device())?
                    .to_dtype(output.mean.dtype())
                    .map_err(Error::from)
            })
            .transpose()?;
        self.sample_with_supplied_noise(output, noise.as_ref())
    }

    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let latents = if latents.dim(1)? == self.vae_dim {
            latents.clone()
        } else {
            latents.transpose(1, 2)?
        };
        self.decoder.forward(&latents)
    }

    fn decode_streaming(
        &self,
        latents: &Tensor,
        cache: &mut TokenizerStreamingState,
    ) -> Result<Tensor> {
        let latents = if latents.dim(1)? == self.vae_dim {
            latents.clone()
        } else {
            latents.transpose(1, 2)?
        };
        self.decoder
            .forward_streaming(&latents, cache.decoder_mut(&self.decoder))
    }

    pub(crate) fn decode_streaming_physical(
        &self,
        latents: &Tensor,
        domain: StateDomainId,
        advance_frames: u64,
        lease: &mut InvocationTensorLease,
    ) -> Result<Tensor> {
        let mut cache = TokenizerStreamingState::new();
        let expected_cursor = hydrate_decoder_cache(lease, &self.decoder, &mut cache)?;
        let output = self.decode_streaming(latents, &mut cache)?;
        commit_decoder_cache(
            lease,
            domain,
            expected_cursor,
            advance_frames,
            &self.decoder,
            cache.decoder.as_ref().ok_or_else(|| {
                Error::InferenceError(
                    "VibeVoice acoustic decoder did not initialize physical state".into(),
                )
            })?,
        )?;
        Ok(output)
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

    pub(crate) fn encoder_state_geometry(&self) -> Vec<VibeVoiceTokenizerStateComponentGeometry> {
        self.encoder.streaming_state_geometry()
    }

    fn encode_streaming(
        &self,
        audio: &Tensor,
        cache: &mut TokenizerStreamingState,
    ) -> Result<VibeVoiceTokenizerEncoderOutput> {
        let latents = self
            .encoder
            .forward_streaming(audio, cache.encoder_mut(&self.encoder))?;
        Ok(VibeVoiceTokenizerEncoderOutput {
            mean: latents.transpose(1, 2)?,
            std: None,
        })
    }

    pub(crate) fn encode_streaming_physical(
        &self,
        audio: &Tensor,
        domain: StateDomainId,
        advance: u64,
        lease: &mut InvocationTensorLease,
    ) -> Result<VibeVoiceTokenizerEncoderOutput> {
        let mut cache = TokenizerStreamingState::new();
        let expected_cursor = hydrate_encoder_cache(lease, &self.encoder, &mut cache)?;
        let output = self.encode_streaming(audio, &mut cache)?;
        commit_encoder_cache(
            lease,
            domain,
            expected_cursor,
            advance,
            &self.encoder,
            cache.encoder.as_ref().ok_or_else(|| {
                Error::InferenceError(
                    "VibeVoice semantic encoder did not initialize physical state".into(),
                )
            })?,
        )?;
        Ok(output)
    }

    pub(crate) fn encode_streaming_retained(
        &self,
        audio: &Tensor,
        domain: StateDomainId,
        expected_cursor: u64,
        target_cursor: u64,
        transaction: PhysicalStateTransactionId,
        arena: &TensorStateArena,
    ) -> Result<VibeVoiceTokenizerEncoderOutput> {
        encode_encoder_span_retained(
            &self.encoder,
            audio,
            domain,
            expected_cursor,
            target_cursor,
            transaction,
            arena,
            "semantic",
        )
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

impl TokenizerStackConfig {
    fn validate_topology(&self, stack: &str) -> Result<()> {
        let expected_depths =
            self.ratios.len().checked_add(1).ok_or_else(|| {
                Error::ModelLoadError("VibeVoice tokenizer topology overflow".into())
            })?;
        if self.depths.len() != expected_depths {
            return Err(Error::ModelLoadError(format!(
                "VibeVoice tokenizer {stack} requires depths.len() == ratios.len() + 1, got {} depths and {} ratios",
                self.depths.len(),
                self.ratios.len()
            )));
        }
        Ok(())
    }
}

struct TokenizerEncoder {
    downsample_layers: Vec<SConv1d>,
    stages: Vec<Vec<Block1D>>,
    norm: ConvNorm,
    head: SConv1d,
}

impl TokenizerEncoder {
    fn load(config: TokenizerStackConfig, vb: VarBuilder) -> Result<Self> {
        config.validate_topology("encoder")?;
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

    fn streaming_state_geometry(&self) -> Vec<VibeVoiceTokenizerStateComponentGeometry> {
        let mut geometry = Vec::new();
        for (downsample, stage) in self.downsample_layers.iter().zip(&self.stages) {
            downsample.push_streaming_state_geometry(&mut geometry);
            for block in stage {
                block.push_streaming_state_geometry(&mut geometry);
            }
        }
        self.head.push_streaming_state_geometry(&mut geometry);
        geometry
    }

    fn hydrate_streaming_state(
        &self,
        cache: &mut TokenizerEncoderStreamingCache,
        components: &[TokenizerStateComponentSlice],
    ) -> Result<()> {
        let mut cursor = PhysicalComponentCursor::new(components);
        for ((downsample, downsample_cache), (stage, stage_cache)) in self
            .downsample_layers
            .iter()
            .zip(&mut cache.downsample_layers)
            .zip(self.stages.iter().zip(&mut cache.stages))
        {
            downsample.hydrate_streaming_state(downsample_cache, &mut cursor)?;
            for (block, block_cache) in stage.iter().zip(stage_cache) {
                block.hydrate_streaming_state(block_cache, &mut cursor)?;
            }
        }
        self.head
            .hydrate_streaming_state(&mut cache.head, &mut cursor)?;
        cursor.finish()
    }

    fn collect_streaming_state(
        &self,
        cache: &TokenizerEncoderStreamingCache,
    ) -> Result<Vec<Tensor>> {
        let mut components = Vec::new();
        for ((downsample, downsample_cache), (stage, stage_cache)) in self
            .downsample_layers
            .iter()
            .zip(&cache.downsample_layers)
            .zip(self.stages.iter().zip(&cache.stages))
        {
            downsample.collect_streaming_state(downsample_cache, &mut components)?;
            for (block, block_cache) in stage.iter().zip(stage_cache) {
                block.collect_streaming_state(block_cache, &mut components)?;
            }
        }
        self.head
            .collect_streaming_state(&cache.head, &mut components)?;
        Ok(components)
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
        config.validate_topology("decoder")?;
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

    fn streaming_state_geometry(&self) -> Vec<VibeVoiceTokenizerStateComponentGeometry> {
        let mut geometry = Vec::new();
        for (upsample, stage) in self.upsample_layers.iter().zip(&self.stages) {
            upsample.push_streaming_state_geometry(&mut geometry);
            for block in stage {
                block.push_streaming_state_geometry(&mut geometry);
            }
        }
        self.head.push_streaming_state_geometry(&mut geometry);
        geometry
    }

    fn hydrate_streaming_state(
        &self,
        cache: &mut TokenizerDecoderStreamingCache,
        components: &[TokenizerStateComponentSlice],
    ) -> Result<()> {
        let mut cursor = PhysicalComponentCursor::new(components);
        for ((upsample, upsample_cache), (stage, stage_cache)) in self
            .upsample_layers
            .iter()
            .zip(&mut cache.upsample_layers)
            .zip(self.stages.iter().zip(&mut cache.stages))
        {
            upsample.hydrate_streaming_state(upsample_cache, &mut cursor)?;
            for (block, block_cache) in stage.iter().zip(stage_cache) {
                block.hydrate_streaming_state(block_cache, &mut cursor)?;
            }
        }
        self.head
            .hydrate_streaming_state(&mut cache.head, &mut cursor)?;
        cursor.finish()
    }

    fn collect_streaming_state(
        &self,
        cache: &TokenizerDecoderStreamingCache,
    ) -> Result<Vec<Tensor>> {
        let mut components = Vec::new();
        for ((upsample, upsample_cache), (stage, stage_cache)) in self
            .upsample_layers
            .iter()
            .zip(&cache.upsample_layers)
            .zip(self.stages.iter().zip(&cache.stages))
        {
            upsample.collect_streaming_state(upsample_cache, &mut components)?;
            for (block, block_cache) in stage.iter().zip(stage_cache) {
                block.collect_streaming_state(block_cache, &mut components)?;
            }
        }
        self.head
            .collect_streaming_state(&cache.head, &mut components)?;
        Ok(components)
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

    fn push_streaming_state_geometry(
        &self,
        geometry: &mut Vec<VibeVoiceTokenizerStateComponentGeometry>,
    ) {
        match self {
            Self::Conv(layer) => layer.push_streaming_state_geometry(geometry),
            Self::Transposed(layer) => layer.push_streaming_state_geometry(geometry),
        }
    }

    fn hydrate_streaming_state(
        &self,
        cache: &mut UpsampleLayerStreamingCache,
        cursor: &mut PhysicalComponentCursor<'_>,
    ) -> Result<()> {
        match (self, cache) {
            (Self::Conv(layer), UpsampleLayerStreamingCache::Conv(cache)) => {
                layer.hydrate_streaming_state(cache, cursor)
            }
            (Self::Transposed(layer), UpsampleLayerStreamingCache::Transposed(cache)) => {
                layer.hydrate_streaming_state(cache, cursor)
            }
            _ => Err(Error::InferenceError(
                "VibeVoice tokenizer physical decoder state layer type mismatch".into(),
            )),
        }
    }

    fn collect_streaming_state(
        &self,
        cache: &UpsampleLayerStreamingCache,
        components: &mut Vec<Tensor>,
    ) -> Result<()> {
        match (self, cache) {
            (Self::Conv(layer), UpsampleLayerStreamingCache::Conv(cache)) => {
                layer.collect_streaming_state(cache, components)
            }
            (Self::Transposed(layer), UpsampleLayerStreamingCache::Transposed(cache)) => {
                layer.collect_streaming_state(cache, components)
            }
            _ => Err(Error::InferenceError(
                "VibeVoice tokenizer physical decoder state layer type mismatch".into(),
            )),
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

    fn push_streaming_state_geometry(
        &self,
        geometry: &mut Vec<VibeVoiceTokenizerStateComponentGeometry>,
    ) {
        self.mixer.push_streaming_state_geometry(geometry);
    }

    fn hydrate_streaming_state(
        &self,
        cache: &mut Block1DStreamingCache,
        cursor: &mut PhysicalComponentCursor<'_>,
    ) -> Result<()> {
        self.mixer.hydrate_streaming_state(&mut cache.mixer, cursor)
    }

    fn collect_streaming_state(
        &self,
        cache: &Block1DStreamingCache,
        components: &mut Vec<Tensor>,
    ) -> Result<()> {
        self.mixer.collect_streaming_state(&cache.mixer, components)
    }
}

struct SConv1d {
    conv: Conv1d,
    input_channels: usize,
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
            input_channels: in_channels,
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

    fn push_streaming_state_geometry(
        &self,
        geometry: &mut Vec<VibeVoiceTokenizerStateComponentGeometry>,
    ) {
        if self.causal_padding > 0 {
            geometry.push(VibeVoiceTokenizerStateComponentGeometry {
                channels: self.input_channels,
                frames: self.causal_padding,
            });
        }
    }

    fn hydrate_streaming_state(
        &self,
        cache: &mut SConv1dStreamingCache,
        cursor: &mut PhysicalComponentCursor<'_>,
    ) -> Result<()> {
        cache.prefix = if self.causal_padding == 0 {
            None
        } else {
            Some(cursor.next(VibeVoiceTokenizerStateComponentGeometry {
                channels: self.input_channels,
                frames: self.causal_padding,
            })?)
        };
        Ok(())
    }

    fn collect_streaming_state(
        &self,
        cache: &SConv1dStreamingCache,
        components: &mut Vec<Tensor>,
    ) -> Result<()> {
        if self.causal_padding > 0 {
            components.push(
                cache
                    .prefix
                    .as_ref()
                    .ok_or_else(|| {
                        Error::InferenceError(
                            "VibeVoice tokenizer physical prefix was not initialized".into(),
                        )
                    })?
                    .contiguous()?,
            );
        }
        Ok(())
    }
}

struct SConvTranspose1d {
    conv: ConvTranspose1d,
    output_channels: usize,
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
            output_channels: out_channels,
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

    fn push_streaming_state_geometry(
        &self,
        geometry: &mut Vec<VibeVoiceTokenizerStateComponentGeometry>,
    ) {
        if self.right_trim > 0 {
            geometry.push(VibeVoiceTokenizerStateComponentGeometry {
                channels: self.output_channels,
                frames: self.right_trim,
            });
        }
    }

    fn hydrate_streaming_state(
        &self,
        cache: &mut SConvTranspose1dStreamingCache,
        cursor: &mut PhysicalComponentCursor<'_>,
    ) -> Result<()> {
        cache.tail = if self.right_trim == 0 {
            None
        } else {
            Some(cursor.next(VibeVoiceTokenizerStateComponentGeometry {
                channels: self.output_channels,
                frames: self.right_trim,
            })?)
        };
        Ok(())
    }

    fn collect_streaming_state(
        &self,
        cache: &SConvTranspose1dStreamingCache,
        components: &mut Vec<Tensor>,
    ) -> Result<()> {
        if self.right_trim > 0 {
            components.push(
                cache
                    .tail
                    .as_ref()
                    .ok_or_else(|| {
                        Error::InferenceError(
                            "VibeVoice tokenizer physical overlap tail was not initialized".into(),
                        )
                    })?
                    .contiguous()?,
            );
        }
        Ok(())
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

struct PhysicalComponentCursor<'a> {
    components: &'a [TokenizerStateComponentSlice],
    index: usize,
}

#[derive(Clone)]
struct TokenizerStateComponentSlice {
    component: StateComponentId,
    tensor: Tensor,
}

impl<'a> PhysicalComponentCursor<'a> {
    fn new(components: &'a [TokenizerStateComponentSlice]) -> Self {
        Self {
            components,
            index: 0,
        }
    }

    fn next(&mut self, geometry: VibeVoiceTokenizerStateComponentGeometry) -> Result<Tensor> {
        let expected_id = StateComponentId::new(u32::try_from(self.index + 1).map_err(|_| {
            Error::InferenceError("VibeVoice tokenizer physical component count exceeds u32".into())
        })?);
        let component = self.components.get(self.index).ok_or_else(|| {
            Error::InferenceError(
                "VibeVoice tokenizer physical snapshot is missing a component".into(),
            )
        })?;
        if component.component != expected_id
            || component.tensor.dims() != [1, geometry.channels, geometry.frames]
        {
            return Err(Error::InferenceError(format!(
                "VibeVoice tokenizer physical component {} has shape {:?}, expected [1, {}, {}]",
                component.component.get(),
                component.tensor.dims(),
                geometry.channels,
                geometry.frames
            )));
        }
        self.index += 1;
        Ok(component.tensor.clone())
    }

    fn finish(self) -> Result<()> {
        if self.index != self.components.len() {
            return Err(Error::InferenceError(format!(
                "VibeVoice tokenizer physical snapshot has {} unused components",
                self.components.len() - self.index
            )));
        }
        Ok(())
    }
}

fn invocation_component_slices(
    components: &[InvocationTensorComponentSlice],
) -> Vec<TokenizerStateComponentSlice> {
    components
        .iter()
        .map(|component| TokenizerStateComponentSlice {
            component: component.component,
            tensor: component.tensor.clone(),
        })
        .collect()
}

fn retained_component_slices(
    snapshot: &StateDomainSnapshot,
    context: &str,
) -> Result<Vec<TokenizerStateComponentSlice>> {
    snapshot
        .components
        .iter()
        .map(|component| {
            Ok(TokenizerStateComponentSlice {
                component: component.component,
                tensor: component.tensor.clone().ok_or_else(|| {
                    Error::InferenceError(format!(
                        "VibeVoice {context} retained state contains an absent component"
                    ))
                })?,
            })
        })
        .collect()
}

fn hydrate_encoder_from_slices(
    encoder: &TokenizerEncoder,
    cache: &mut TokenizerStreamingState,
    components: &[TokenizerStateComponentSlice],
) -> Result<()> {
    encoder.hydrate_streaming_state(cache.encoder_mut(encoder), components)
}

fn hydrate_decoder_from_slices(
    decoder: &TokenizerDecoder,
    cache: &mut TokenizerStreamingState,
    components: &[TokenizerStateComponentSlice],
) -> Result<()> {
    decoder.hydrate_streaming_state(cache.decoder_mut(decoder), components)
}

fn hydrate_encoder_cache(
    lease: &InvocationTensorLease,
    encoder: &TokenizerEncoder,
    cache: &mut TokenizerStreamingState,
) -> Result<u64> {
    let expected_cursor = lease.arena()?.absolute_cursor();
    if expected_cursor == 0 {
        return Ok(0);
    }
    let snapshot = lease.read_snapshot()?;
    if snapshot.absolute_cursor != expected_cursor || snapshot.valid_length != 1 {
        return Err(Error::InferenceError(
            "VibeVoice tokenizer encoder snapshot has stale cursor metadata".into(),
        ));
    }
    let components = invocation_component_slices(&snapshot.components);
    hydrate_encoder_from_slices(encoder, cache, &components)?;
    Ok(expected_cursor)
}

fn hydrate_decoder_cache(
    lease: &InvocationTensorLease,
    decoder: &TokenizerDecoder,
    cache: &mut TokenizerStreamingState,
) -> Result<u64> {
    let expected_cursor = lease.arena()?.absolute_cursor();
    if expected_cursor == 0 {
        return Ok(0);
    }
    let snapshot = lease.read_snapshot()?;
    if snapshot.absolute_cursor != expected_cursor || snapshot.valid_length != 1 {
        return Err(Error::InferenceError(
            "VibeVoice tokenizer decoder snapshot has stale cursor metadata".into(),
        ));
    }
    let components = invocation_component_slices(&snapshot.components);
    hydrate_decoder_from_slices(decoder, cache, &components)?;
    Ok(expected_cursor)
}

fn encode_encoder_span_retained(
    encoder: &TokenizerEncoder,
    audio: &Tensor,
    domain: StateDomainId,
    expected_cursor: u64,
    target_cursor: u64,
    transaction: PhysicalStateTransactionId,
    arena: &TensorStateArena,
    context: &str,
) -> Result<VibeVoiceTokenizerEncoderOutput> {
    let advance = target_cursor.checked_sub(expected_cursor).ok_or_else(|| {
        Error::InvalidInput(format!("VibeVoice {context} retained span moves backwards"))
    })?;
    let audio_samples = u64::try_from(audio.dim(2)?).map_err(|_| {
        Error::InvalidInput(format!(
            "VibeVoice {context} retained audio extent exceeds u64"
        ))
    })?;
    if advance == 0 || advance != audio_samples {
        return Err(Error::InvalidInput(format!(
            "VibeVoice {context} retained span [{expected_cursor}, {target_cursor}) does not match {audio_samples} input samples"
        )));
    }

    let mut cache = TokenizerStreamingState::new();
    match arena.read_transaction_base(transaction, domain)? {
        None if expected_cursor == 0 => {}
        None => {
            return Err(Error::InferenceError(format!(
                "VibeVoice {context} retained state is absent at cursor {expected_cursor}"
            )))
        }
        Some(snapshot) if snapshot.cursor == expected_cursor && expected_cursor > 0 => {
            let components = retained_component_slices(&snapshot, context)?;
            hydrate_encoder_from_slices(encoder, &mut cache, &components)?;
        }
        Some(snapshot) => {
            return Err(Error::InferenceError(format!(
                "VibeVoice {context} retained state cursor {} does not match expected {expected_cursor}",
                snapshot.cursor
            )))
        }
    }

    let latents = encoder.forward_streaming(audio, cache.encoder_mut(encoder))?;
    let values = retained_state_values(encoder.collect_streaming_state(
        cache.encoder.as_ref().ok_or_else(|| {
            Error::InferenceError(format!(
                "VibeVoice {context} retained encoder did not initialize state"
            ))
        })?,
    )?)?;
    arena.stage_replace(transaction, domain, expected_cursor, target_cursor, values)?;
    Ok(VibeVoiceTokenizerEncoderOutput {
        mean: latents.transpose(1, 2)?,
        std: None,
    })
}

fn retained_state_values(components: Vec<Tensor>) -> Result<Vec<StateComponentValue>> {
    if components.is_empty() {
        return Err(Error::InferenceError(
            "VibeVoice retained tokenizer produced no state components".into(),
        ));
    }
    components
        .into_iter()
        .enumerate()
        .map(|(index, tensor)| {
            Ok(StateComponentValue {
                component: StateComponentId::new(u32::try_from(index + 1).map_err(|_| {
                    Error::InferenceError(
                        "VibeVoice retained tokenizer component count exceeds u32".into(),
                    )
                })?),
                tensor: Some(tensor.contiguous()?),
            })
        })
        .collect()
}

fn commit_encoder_cache(
    lease: &mut InvocationTensorLease,
    domain: StateDomainId,
    expected_cursor: u64,
    advance: u64,
    encoder: &TokenizerEncoder,
    cache: &TokenizerEncoderStreamingCache,
) -> Result<()> {
    commit_streaming_cache(
        lease,
        domain,
        expected_cursor,
        advance,
        encoder.collect_streaming_state(cache),
    )
}

fn commit_decoder_cache(
    lease: &mut InvocationTensorLease,
    domain: StateDomainId,
    expected_cursor: u64,
    advance: u64,
    decoder: &TokenizerDecoder,
    cache: &TokenizerDecoderStreamingCache,
) -> Result<()> {
    commit_streaming_cache(
        lease,
        domain,
        expected_cursor,
        advance,
        decoder.collect_streaming_state(cache),
    )
}

fn commit_streaming_cache(
    lease: &mut InvocationTensorLease,
    domain: StateDomainId,
    expected_cursor: u64,
    advance: u64,
    components: Result<Vec<Tensor>>,
) -> Result<()> {
    let components = components?;
    if advance == 0 || components.is_empty() {
        return Err(Error::InferenceError(
            "VibeVoice tokenizer physical update requires state and a non-zero advance".into(),
        ));
    }
    let target_cursor = expected_cursor
        .checked_add(advance)
        .ok_or_else(|| Error::InferenceError("VibeVoice tokenizer cursor overflow".into()))?;
    let mut declared = Vec::with_capacity(components.len());
    let mut values = Vec::with_capacity(components.len());
    for (index, tensor) in components.into_iter().enumerate() {
        let [batch, channels, frames] = tensor.dims() else {
            return Err(Error::InferenceError(format!(
                "VibeVoice tokenizer physical component has non-convolution shape {:?}",
                tensor.dims()
            )));
        };
        let component = StateComponentId::new(u32::try_from(index + 1).map_err(|_| {
            Error::InferenceError("VibeVoice tokenizer component count exceeds u32".into())
        })?);
        declared.push(ComponentShapeInstantiation {
            component,
            dimensions: vec![
                ShapeDimensionValue {
                    axis: ShapeAxis::Batch,
                    units: u64::try_from(*batch).map_err(|_| {
                        Error::InferenceError("VibeVoice tokenizer batch exceeds u64".into())
                    })?,
                },
                ShapeDimensionValue {
                    axis: ShapeAxis::Channels,
                    units: u64::try_from(*channels).map_err(|_| {
                        Error::InferenceError("VibeVoice tokenizer channels exceed u64".into())
                    })?,
                },
                ShapeDimensionValue {
                    axis: ShapeAxis::Frames,
                    units: u64::try_from(*frames).map_err(|_| {
                        Error::InferenceError("VibeVoice tokenizer frames exceed u64".into())
                    })?,
                },
            ],
        });
        values.push(InvocationTensorComponentValue { component, tensor });
    }
    lease.apply_intent(
        &DomainStepIntent {
            domain,
            expected_cursor,
            target_cursor,
            update: StateUpdateKind::TensorReplace {
                components: declared,
            },
        },
        InvocationTensorUpdateV2::TensorReplace { components: values },
    )
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
    use std::sync::Arc;

    use candle_core::Device;
    use candle_nn::VarBuilder;

    use crate::backends::state::{
        negotiate_state_plan, PhysicalStateSequenceId, StateBackendPlanRequest,
        TensorStateCapacity, TensorStateSelection,
    };
    use crate::backends::BackendKind;
    use crate::engine::{InvocationTensorPoolOwner, ModelInstanceId};
    use crate::kv::v2::{
        BoundedShape, CheckpointPolicy, InferenceStateContract, InvocationStateCapacity,
        InvocationWorkspaceDomain, PlacementPolicy, PrefixPolicy, ShapeDimension, ShapeExtent,
        StateClock, StateDType, StateDomainHeader, StateDomainSpec, StateGroupId, StateGroupSpec,
        StateScope, TensorComponentSpec, TensorRole, TensorStateDomainSpec, WorkspaceFormula,
        CURRENT_INFERENCE_STATE_ABI,
    };

    fn test_stack_config(ratios: Vec<usize>, depths: Vec<usize>) -> TokenizerStackConfig {
        TokenizerStackConfig {
            dimension: 1,
            channels: 1,
            n_filters: 1,
            ratios,
            depths,
            causal: true,
            kernel_size: 3,
            last_kernel_size: 3,
            layernorm: "RMSNorm".to_string(),
            layernorm_eps: 1e-5,
            disable_last_norm: false,
            mixer_layer: "depthwise_conv".to_string(),
            layer_scale_init_value: 0.0,
            conv_bias: true,
            pad_mode: "constant".to_string(),
            trim_right_ratio: 1.0,
        }
    }

    #[test]
    fn semantic_encode_shape_matches_continuous_latent_contract() {
        let output = VibeVoiceTokenizerEncoderOutput {
            mean: Tensor::zeros((1, 3, 128), DType::F32, &candle_core::Device::Cpu).unwrap(),
            std: None,
        };
        assert_eq!(output.mode().dims(), &[1, 3, 128]);
    }

    #[test]
    fn supplied_acoustic_noise_is_retry_exact_and_checked() {
        let tokenizer = VibeVoiceAcousticTokenizer {
            encoder: tiny_encoder(),
            decoder: tiny_decoder(),
            fix_std: 0.5,
            std_dist_type: "normal".into(),
            vae_dim: 1,
        };
        let output = VibeVoiceTokenizerEncoderOutput {
            mean: Tensor::from_vec(vec![1.0f32, -2.0], (1, 2, 1), &Device::Cpu).unwrap(),
            std: Some(0.5),
        };
        let noise = Tensor::from_vec(vec![0.25f32, -0.75], (1, 2, 1), &Device::Cpu).unwrap();
        let first = tokenizer
            .sample_with_supplied_noise(&output, Some(&noise))
            .unwrap();
        let retry = tokenizer
            .sample_with_supplied_noise(&output, Some(&noise))
            .unwrap();
        assert_eq!(
            first.to_vec3::<f32>().unwrap(),
            retry.to_vec3::<f32>().unwrap()
        );
        assert_eq!(
            first.to_vec3::<f32>().unwrap(),
            vec![vec![vec![1.125], vec![-2.375]]]
        );
        assert!(tokenizer.sample_with_supplied_noise(&output, None).is_err());
        let wrong_shape = Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu).unwrap();
        assert!(tokenizer
            .sample_with_supplied_noise(&output, Some(&wrong_shape))
            .is_err());

        let deterministic = VibeVoiceAcousticTokenizer {
            encoder: tiny_encoder(),
            decoder: tiny_decoder(),
            fix_std: 0.0,
            std_dist_type: "none".into(),
            vae_dim: 1,
        };
        assert!(deterministic
            .sample_with_supplied_noise(&output, Some(&noise))
            .is_err());
        assert_tensor_close(
            &deterministic
                .sample_with_supplied_noise(&output, None)
                .unwrap()
                .transpose(1, 2)
                .unwrap(),
            &output.mean.transpose(1, 2).unwrap(),
            0.0,
        );
    }

    #[test]
    fn tokenizer_topology_requires_one_more_depth_stage_than_ratio() {
        assert!(test_stack_config(vec![2, 4], vec![1, 1, 1])
            .validate_topology("test")
            .is_ok());
        let err = test_stack_config(vec![2, 4], vec![1, 1])
            .validate_topology("test")
            .unwrap_err();
        assert!(err.to_string().contains("depths.len() == ratios.len() + 1"));
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
        let mut geometry = Vec::new();
        layer.push_streaming_state_geometry(&mut geometry);
        assert_eq!(
            geometry,
            vec![VibeVoiceTokenizerStateComponentGeometry {
                channels: 1,
                frames: 2,
            }]
        );
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
        let mut geometry = Vec::new();
        layer.push_streaming_state_geometry(&mut geometry);
        assert_eq!(
            geometry,
            vec![VibeVoiceTokenizerStateComponentGeometry {
                channels: 1,
                frames: 2,
            }]
        );
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

    #[test]
    fn physical_encoder_and_decoder_state_round_trip_across_steps() {
        let encoder = tiny_encoder();
        let encoder_domain = StateDomainId::new(1);
        let encoder_owner = tensor_owner(
            encoder_domain,
            &encoder.streaming_state_geometry(),
            StateClock::AudioSamples,
        );
        let mut encoder_lease = encoder_owner.lease().unwrap();
        let encoder_input = Tensor::from_vec(
            (0..8).map(|value| value as f32 / 10.0).collect::<Vec<_>>(),
            (1, 1, 8),
            &Device::Cpu,
        )
        .unwrap();
        let mut local_encoder = encoder.streaming_cache();
        for (step, offset) in [0usize, 4].into_iter().enumerate() {
            let chunk = encoder_input.narrow(2, offset, 4).unwrap();
            let expected = encoder
                .forward_streaming(&chunk, &mut local_encoder)
                .unwrap();
            let mut physical = TokenizerStreamingState::new();
            let cursor = hydrate_encoder_cache(&encoder_lease, &encoder, &mut physical).unwrap();
            assert_eq!(cursor, (step * 4) as u64);
            let actual = encoder
                .forward_streaming(&chunk, physical.encoder_mut(&encoder))
                .unwrap();
            commit_encoder_cache(
                &mut encoder_lease,
                encoder_domain,
                cursor,
                4,
                &encoder,
                physical.encoder.as_ref().unwrap(),
            )
            .unwrap();
            assert_tensor_close(&actual, &expected, 1e-6);
        }
        assert_eq!(encoder_lease.arena().unwrap().absolute_cursor(), 8);
        let mut hydrated_encoder = TokenizerStreamingState::new();
        hydrate_encoder_cache(&encoder_lease, &encoder, &mut hydrated_encoder).unwrap();
        assert!(commit_encoder_cache(
            &mut encoder_lease,
            StateDomainId::new(9),
            8,
            4,
            &encoder,
            hydrated_encoder.encoder.as_ref().unwrap(),
        )
        .is_err());
        assert!(commit_encoder_cache(
            &mut encoder_lease,
            encoder_domain,
            7,
            4,
            &encoder,
            hydrated_encoder.encoder.as_ref().unwrap(),
        )
        .is_err());
        assert_eq!(encoder_lease.arena().unwrap().absolute_cursor(), 8);

        let decoder = tiny_decoder();
        let decoder_domain = StateDomainId::new(2);
        let decoder_owner = tensor_owner(
            decoder_domain,
            &decoder.streaming_state_geometry(),
            StateClock::CodecFrames,
        );
        let mut decoder_lease = decoder_owner.lease().unwrap();
        let decoder_input = Tensor::from_vec(vec![0.5f32, -1.0], (1, 1, 2), &Device::Cpu).unwrap();
        let mut local_decoder = decoder.streaming_cache();
        for offset in [0usize, 1] {
            let frame = decoder_input.narrow(2, offset, 1).unwrap();
            let expected = decoder
                .forward_streaming(&frame, &mut local_decoder)
                .unwrap();
            let mut physical = TokenizerStreamingState::new();
            let cursor = hydrate_decoder_cache(&decoder_lease, &decoder, &mut physical).unwrap();
            let actual = decoder
                .forward_streaming(&frame, physical.decoder_mut(&decoder))
                .unwrap();
            commit_decoder_cache(
                &mut decoder_lease,
                decoder_domain,
                cursor,
                1,
                &decoder,
                physical.decoder.as_ref().unwrap(),
            )
            .unwrap();
            assert_tensor_close(&actual, &expected, 1e-6);
        }
        assert_eq!(decoder_lease.arena().unwrap().absolute_cursor(), 2);
    }

    #[test]
    fn retained_encoder_continuation_is_transactional_and_rollback_safe() {
        let tokenizer = VibeVoiceSemanticTokenizer {
            encoder: tiny_encoder(),
        };
        let domain = StateDomainId::new(7);
        let group = StateGroupId::new(3);
        let arena =
            retained_tensor_arena(domain, group, &tokenizer.encoder.streaming_state_geometry());
        let sequence = PhysicalStateSequenceId::new(11).unwrap();
        arena.register(sequence).unwrap();
        let input = Tensor::from_vec(
            (0..8).map(|value| value as f32 / 10.0).collect::<Vec<_>>(),
            (1, 1, 8),
            &Device::Cpu,
        )
        .unwrap();
        let first_chunk = input.narrow(2, 0, 4).unwrap();
        let second_chunk = input.narrow(2, 4, 4).unwrap();

        let rollback = PhysicalStateTransactionId::new(21).unwrap();
        arena
            .begin_selected(
                rollback,
                sequence,
                &[TensorStateSelection {
                    group,
                    clock: StateClock::AudioSamples,
                    expected_cursor: 0,
                    target_cursor: 4,
                }],
            )
            .unwrap();
        tokenizer
            .encode_streaming_retained(&first_chunk, domain, 0, 4, rollback, &arena)
            .unwrap();
        assert!(arena.read(sequence, domain).unwrap().is_none());
        arena.abort(rollback).unwrap();
        assert!(arena.read(sequence, domain).unwrap().is_none());

        let mut local_cache = tokenizer.encoder.streaming_cache();
        let expected_first = tokenizer
            .encoder
            .forward_streaming(&first_chunk, &mut local_cache)
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let first = PhysicalStateTransactionId::new(22).unwrap();
        arena
            .begin_selected(
                first,
                sequence,
                &[TensorStateSelection {
                    group,
                    clock: StateClock::AudioSamples,
                    expected_cursor: 0,
                    target_cursor: 4,
                }],
            )
            .unwrap();
        let actual_first = tokenizer
            .encode_streaming_retained(&first_chunk, domain, 0, 4, first, &arena)
            .unwrap();
        assert_tensor_close(&actual_first.mean, &expected_first, 1e-6);
        let completion = arena.seal_selected_completion(first).unwrap();
        arena.commit_selected(first, &completion).unwrap();
        assert_eq!(arena.read(sequence, domain).unwrap().unwrap().cursor, 4);

        let expected_second = tokenizer
            .encoder
            .forward_streaming(&second_chunk, &mut local_cache)
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let second = PhysicalStateTransactionId::new(23).unwrap();
        arena
            .begin_selected(
                second,
                sequence,
                &[TensorStateSelection {
                    group,
                    clock: StateClock::AudioSamples,
                    expected_cursor: 4,
                    target_cursor: 8,
                }],
            )
            .unwrap();
        assert!(tokenizer
            .encode_streaming_retained(&second_chunk, domain, 3, 8, second, &arena)
            .is_err());
        let actual_second = tokenizer
            .encode_streaming_retained(&second_chunk, domain, 4, 8, second, &arena)
            .unwrap();
        assert_tensor_close(&actual_second.mean, &expected_second, 1e-6);
        let completion = arena.seal_selected_completion(second).unwrap();
        arena.commit_selected(second, &completion).unwrap();
        assert_eq!(arena.read(sequence, domain).unwrap().unwrap().cursor, 8);
    }

    #[test]
    fn physical_component_order_and_shapes_fail_closed() {
        let geometry = VibeVoiceTokenizerStateComponentGeometry {
            channels: 1,
            frames: 2,
        };
        let tensor = Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu).unwrap();
        let exact = TokenizerStateComponentSlice {
            component: StateComponentId::new(1),
            tensor: tensor.clone(),
        };

        let mut missing = PhysicalComponentCursor::new(&[]);
        assert!(missing.next(geometry).is_err());

        let wrong_id = [TokenizerStateComponentSlice {
            component: StateComponentId::new(2),
            tensor: tensor.clone(),
        }];
        assert!(PhysicalComponentCursor::new(&wrong_id)
            .next(geometry)
            .is_err());

        let wrong_shape = [TokenizerStateComponentSlice {
            component: StateComponentId::new(1),
            tensor: Tensor::zeros((1, 2, 1), DType::F32, &Device::Cpu).unwrap(),
        }];
        assert!(PhysicalComponentCursor::new(&wrong_shape)
            .next(geometry)
            .is_err());

        let extra = [
            exact.clone(),
            TokenizerStateComponentSlice {
                component: StateComponentId::new(2),
                tensor,
            },
        ];
        let mut cursor = PhysicalComponentCursor::new(&extra);
        cursor.next(geometry).unwrap();
        assert!(cursor.finish().is_err());
    }

    fn tiny_encoder() -> TokenizerEncoder {
        TokenizerEncoder {
            downsample_layers: vec![test_sconv1d(&[0.25, -0.5, 0.75], 0.1)],
            stages: vec![vec![]],
            norm: ConvNorm::Identity,
            head: test_sconv1d(&[-0.2, 0.4, 0.6], -0.05),
        }
    }

    fn tiny_decoder() -> TokenizerDecoder {
        TokenizerDecoder {
            upsample_layers: vec![
                UpsampleLayer::Conv(test_sconv1d(&[0.3, -0.1, 0.8], 0.2)),
                UpsampleLayer::Transposed(test_sconvtranspose(&[0.2, -0.4, 0.6, 0.8], 0.3)),
            ],
            stages: vec![vec![], vec![]],
            norm: ConvNorm::Identity,
            head: test_sconv1d(&[0.5, 0.25, -0.4], -0.1),
        }
    }

    fn test_sconv1d(weight: &[f32], bias: f32) -> SConv1d {
        let mut tensors = HashMap::new();
        tensors.insert(
            "conv.conv.weight".to_string(),
            Tensor::from_slice(weight, (1, 1, weight.len()), &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "conv.conv.bias".to_string(),
            Tensor::from_slice(&[bias], 1, &Device::Cpu).unwrap(),
        );
        SConv1d::load(
            1,
            1,
            weight.len(),
            1,
            1,
            1,
            true,
            true,
            VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu),
        )
        .unwrap()
    }

    fn test_sconvtranspose(weight: &[f32], bias: f32) -> SConvTranspose1d {
        let mut tensors = HashMap::new();
        tensors.insert(
            "convtr.convtr.weight".to_string(),
            Tensor::from_slice(weight, (1, 1, weight.len()), &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "convtr.convtr.bias".to_string(),
            Tensor::from_slice(&[bias], 1, &Device::Cpu).unwrap(),
        );
        SConvTranspose1d::load(
            1,
            1,
            weight.len(),
            2,
            true,
            1.0,
            VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu),
        )
        .unwrap()
    }

    fn tensor_owner(
        domain: StateDomainId,
        geometry: &[VibeVoiceTokenizerStateComponentGeometry],
        clock: StateClock,
    ) -> InvocationTensorPoolOwner {
        let components = geometry
            .iter()
            .enumerate()
            .map(|(index, geometry)| TensorComponentSpec {
                id: StateComponentId::new(u32::try_from(index + 1).unwrap()),
                role: TensorRole::ConvolutionState,
                shape: BoundedShape {
                    dimensions: vec![
                        ShapeDimension {
                            axis: ShapeAxis::Batch,
                            extent: ShapeExtent::Fixed { value: 1 },
                        },
                        ShapeDimension {
                            axis: ShapeAxis::Channels,
                            extent: ShapeExtent::Fixed {
                                value: geometry.channels as u64,
                            },
                        },
                        ShapeDimension {
                            axis: ShapeAxis::Frames,
                            extent: ShapeExtent::Fixed {
                                value: geometry.frames as u64,
                            },
                        },
                    ],
                },
                accepted_dtypes: vec![StateDType::F32],
            })
            .collect::<Vec<_>>();
        let state = StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: StateDomainHeader {
                id: domain,
                scope: StateScope::Invocation,
                clock,
                placement: PlacementPolicy::BackendLocal,
                prefix: PrefixPolicy::Disabled,
                checkpoint: CheckpointPolicy::None,
            },
            components,
        });
        let contract = InferenceStateContract {
            abi: CURRENT_INFERENCE_STATE_ABI,
            domains: vec![state.clone()],
            groups: vec![StateGroupSpec {
                id: StateGroupId::new(domain.get()),
                domains: vec![domain],
                prefix_shareable: false,
            }],
        };
        let plan = Arc::new(
            negotiate_state_plan(
                &contract,
                &StateBackendPlanRequest {
                    backend: BackendKind::Cpu,
                    device_ordinal: None,
                    page_tokens_hint: None,
                    storage_dtype_hint: None,
                },
            )
            .unwrap(),
        );
        let fixed_bytes = plan
            .non_paged
            .iter()
            .find(|resolved| resolved.domain() == domain)
            .unwrap()
            .maximum_bytes();
        InvocationTensorPoolOwner::new(
            &contract,
            plan,
            InvocationWorkspaceDomain::State {
                state,
                capacity: InvocationStateCapacity::SemanticBounded,
                placement: PlacementPolicy::BackendLocal,
                formula: WorkspaceFormula {
                    fixed_bytes,
                    dimensions: vec![],
                    terms: vec![],
                },
            },
            Device::Cpu,
            ModelInstanceId::new(99),
            1,
            domain.get(),
        )
        .unwrap()
    }

    fn retained_tensor_arena(
        domain: StateDomainId,
        group: StateGroupId,
        geometry: &[VibeVoiceTokenizerStateComponentGeometry],
    ) -> TensorStateArena {
        let components = geometry
            .iter()
            .enumerate()
            .map(|(index, geometry)| TensorComponentSpec {
                id: StateComponentId::new(u32::try_from(index + 1).unwrap()),
                role: TensorRole::ConvolutionState,
                shape: BoundedShape {
                    dimensions: vec![
                        ShapeDimension {
                            axis: ShapeAxis::Batch,
                            extent: ShapeExtent::Fixed { value: 1 },
                        },
                        ShapeDimension {
                            axis: ShapeAxis::Channels,
                            extent: ShapeExtent::Fixed {
                                value: geometry.channels as u64,
                            },
                        },
                        ShapeDimension {
                            axis: ShapeAxis::Frames,
                            extent: ShapeExtent::Fixed {
                                value: geometry.frames as u64,
                            },
                        },
                    ],
                },
                accepted_dtypes: vec![StateDType::F32],
            })
            .collect::<Vec<_>>();
        let contract = InferenceStateContract {
            abi: CURRENT_INFERENCE_STATE_ABI,
            domains: vec![StateDomainSpec::Tensor(TensorStateDomainSpec {
                header: StateDomainHeader {
                    id: domain,
                    scope: StateScope::Retained,
                    clock: StateClock::AudioSamples,
                    placement: PlacementPolicy::BackendLocal,
                    prefix: PrefixPolicy::Disabled,
                    checkpoint: CheckpointPolicy::Transactional,
                },
                components,
            })],
            groups: vec![StateGroupSpec {
                id: group,
                domains: vec![domain],
                prefix_shareable: false,
            }],
        };
        let plan = negotiate_state_plan(
            &contract,
            &StateBackendPlanRequest {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                page_tokens_hint: None,
                storage_dtype_hint: None,
            },
        )
        .unwrap();
        let capacity = TensorStateCapacity::for_plan(&plan, 1, 1).unwrap();
        TensorStateArena::new_with_contract(Arc::new(plan), &contract, capacity, Device::Cpu)
            .unwrap()
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
