//! Fish S2 modified DAC codec.

use candle_core::{DType, IndexOp, Tensor, D};
use candle_nn::{
    ops, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Embedding, LayerNorm,
    Linear, Module, RmsNorm, VarBuilder,
};

use crate::audio::{resample_mono_high_quality, target_sample_count};
use crate::error::{Error, Result};
use crate::models::architectures::fish_s2::codec::fuse_weight_norm_dim0;
use crate::models::architectures::fish_s2::contracts::FishS2DacContract;
use crate::models::architectures::fish_s2::rotary::FishS2RotaryCache;
use crate::models::architectures::fish_s2::tokenizer::FishS2VqCodes;
use crate::models::architectures::qwen3::core::repeat_kv;

pub(crate) const ATTENTION_QUERY_BLOCK: usize = 64;

type CancelCheck<'a> = &'a dyn Fn() -> Result<()>;

#[derive(Debug, Clone, PartialEq)]
pub struct FishS2DacConfig {
    pub sample_rate: u32,
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub encoder_transformer_layers: Vec<usize>,
    pub encoder_window_size: usize,
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
    encoder: FishS2DacAudioEncoder,
    quantizer: FishS2DownsampleResidualVectorQuantizer,
    decoder: FishS2DacAudioDecoder,
}

struct FishS2DownsampleResidualVectorQuantizer {
    semantic_quantizer: FishS2ResidualVectorQuantizer,
    residual_quantizer: FishS2ResidualVectorQuantizer,
    downsample: Vec<FishS2DownsampleBlock>,
    pre_module: FishS2WindowLimitedTransformer,
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

struct FishS2DownsampleBlock {
    conv: FishS2CausalConv1d,
    convnext: FishS2ConvNeXtBlock,
}

struct FishS2UpsampleBlock {
    transposed: FishS2CausalConvTranspose1d,
    convnext: FishS2ConvNeXtBlock,
}

struct FishS2DacAudioEncoder {
    first: FishS2CausalConv1d,
    blocks: Vec<FishS2EncoderBlock>,
    final_snake: FishS2Snake1d,
    final_conv: FishS2CausalConv1d,
}

struct FishS2EncoderBlock {
    residuals: Vec<FishS2ResidualUnit>,
    snake: FishS2Snake1d,
    conv: FishS2CausalConv1d,
    transformer: Option<FishS2WindowLimitedTransformer>,
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
    rotary: FishS2RotaryCache,
    window_size: usize,
}

struct FishS2DacFeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

#[derive(Debug, Clone, PartialEq)]
struct FishS2DacTransformerParams {
    dim: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    intermediate: usize,
    max_seq_len: usize,
    window_size: usize,
    rope_theta: f64,
    rms_norm_eps: f64,
}

impl FishS2DacConfig {
    /// Position capacities saved in the public S2 Pro codec checkpoint.
    pub const MAX_ENCODER_FRAMES: usize = 16_384;
    pub const MAX_QUANTIZER_FRAMES: usize = 4_096;
    /// Reference resampling is bounded to supported audio sample rates.
    pub const MAX_REFERENCE_SAMPLE_RATE: u32 = 384_000;

    pub fn current() -> Self {
        let contract = FishS2DacContract::CURRENT;
        Self {
            sample_rate: contract.sample_rate,
            encoder_dim: 64,
            encoder_rates: contract.encoder_rates.to_vec(),
            encoder_transformer_layers: vec![0, 0, 0, 4],
            encoder_window_size: 512,
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

    pub fn maximum_reference_samples(&self) -> Result<usize> {
        let encoder_stride = self.encoder_rates.iter().try_fold(1usize, |stride, rate| {
            stride
                .checked_mul(*rate)
                .ok_or_else(|| Error::ConfigError("Fish S2 DAC encoder stride overflowed".into()))
        })?;
        let encoder_limit = encoder_stride
            .checked_mul(Self::MAX_ENCODER_FRAMES)
            .ok_or_else(|| Error::ConfigError("Fish S2 DAC reference limit overflowed".into()))?;
        let quantizer_limit = self
            .samples_per_frame()?
            .checked_mul(Self::MAX_QUANTIZER_FRAMES)
            .ok_or_else(|| Error::ConfigError("Fish S2 DAC reference limit overflowed".into()))?;
        Ok(encoder_limit.min(quantizer_limit))
    }

    /// Validates the resampled length before allocating a resampling or encoder buffer.
    pub fn reference_frame_count(&self, input_samples: usize, sample_rate: u32) -> Result<usize> {
        if sample_rate == 0 || sample_rate > Self::MAX_REFERENCE_SAMPLE_RATE || input_samples == 0 {
            return Err(Error::InvalidInput(format!(
                "Fish S2 reference audio requires samples and a sample rate between 1 and {} Hz",
                Self::MAX_REFERENCE_SAMPLE_RATE
            )));
        }
        let samples_per_frame = self.samples_per_frame()?.max(1);
        let resampled_samples = (input_samples as u128 * u128::from(self.sample_rate)
            + u128::from(sample_rate / 2))
            / u128::from(sample_rate);
        let maximum_samples = self.maximum_reference_samples()?;
        if resampled_samples > maximum_samples as u128 {
            return Err(Error::InvalidInput(format!(
                "Fish S2 reference audio exceeds the codec capacity of {maximum_samples} samples at {} Hz",
                self.sample_rate
            )));
        }
        let frames = target_sample_count(input_samples, sample_rate, self.sample_rate)
            .div_ceil(samples_per_frame)
            .max(1);
        let prepared_samples = frames.checked_mul(samples_per_frame).ok_or_else(|| {
            Error::InvalidInput("Fish S2 reference audio length overflowed".into())
        })?;
        if prepared_samples > maximum_samples {
            return Err(Error::InvalidInput(format!(
                "Fish S2 reference audio exceeds the codec capacity of {} samples at {} Hz",
                maximum_samples, self.sample_rate
            )));
        }
        Ok(frames)
    }

    /// One shared pair of F32 tables per transformer, after BF16 frequency rounding.
    pub fn rotary_cache_bytes(&self) -> Result<u64> {
        let encoder_count = self
            .encoder_transformer_layers
            .iter()
            .filter(|layers| **layers > 0)
            .count();
        let quantizer_count: usize = if self.transformer_layers > 0 { 2 } else { 0 };
        encoder_count
            .checked_mul(Self::MAX_ENCODER_FRAMES)
            .and_then(|positions| {
                quantizer_count
                    .checked_mul(Self::MAX_QUANTIZER_FRAMES)
                    .and_then(|quantizer_positions| positions.checked_add(quantizer_positions))
            })
            .and_then(|positions| positions.checked_mul(self.transformer_head_dim))
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| Error::ConfigError("Fish S2 DAC rotary cache size overflowed".into()))
    }
}

impl FishS2DacTransformerParams {
    fn quantizer_post(config: &FishS2DacConfig) -> Self {
        Self {
            dim: config.transformer_heads * config.transformer_head_dim,
            layers: config.transformer_layers,
            heads: config.transformer_heads,
            kv_heads: config.transformer_kv_heads,
            head_dim: config.transformer_head_dim,
            intermediate: config.transformer_intermediate,
            max_seq_len: FishS2DacConfig::MAX_QUANTIZER_FRAMES,
            window_size: config.transformer_window_size,
            rope_theta: config.rope_theta,
            rms_norm_eps: config.rms_norm_eps,
        }
    }

    fn encoder_stage(
        dim: usize,
        layers: usize,
        window_size: usize,
        config: &FishS2DacConfig,
    ) -> Self {
        let heads = (dim / config.transformer_head_dim).max(1);
        Self {
            dim,
            layers,
            heads,
            kv_heads: heads,
            head_dim: config.transformer_head_dim,
            intermediate: dim * 3,
            max_seq_len: FishS2DacConfig::MAX_ENCODER_FRAMES,
            window_size,
            rope_theta: config.rope_theta,
            rms_norm_eps: config.rms_norm_eps,
        }
    }
}

impl FishS2DacDecoder {
    pub fn load(config: FishS2DacConfig, vb: VarBuilder) -> Result<Self> {
        let encoder = FishS2DacAudioEncoder::load(&config, vb.pp("encoder"))?;
        let quantizer = FishS2DownsampleResidualVectorQuantizer::load(&config, vb.pp("quantizer"))?;
        let decoder = FishS2DacAudioDecoder::load(&config, vb.pp("decoder"))?;
        Ok(Self {
            config,
            encoder,
            quantizer,
            decoder,
        })
    }

    pub fn config(&self) -> &FishS2DacConfig {
        &self.config
    }

    pub fn decode_vq_codes(&self, codes: &FishS2VqCodes) -> Result<Vec<f32>> {
        self.decode_vq_codes_with_cancel(codes, &|| Ok(()))
    }

    pub fn decode_vq_codes_with_cancel(
        &self,
        codes: &FishS2VqCodes,
        check_cancelled: CancelCheck<'_>,
    ) -> Result<Vec<f32>> {
        let audio = self.decode_codebooks_with_cancel(&codes.codebooks, check_cancelled)?;
        check_cancelled()?;
        let samples = audio
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        check_cancelled()?;
        Ok(samples)
    }

    pub fn decode_codebooks(&self, codebooks: &[Vec<u32>]) -> Result<Tensor> {
        self.decode_codebooks_with_cancel(codebooks, &|| Ok(()))
    }

    fn decode_codebooks_with_cancel(
        &self,
        codebooks: &[Vec<u32>],
        check_cancelled: CancelCheck<'_>,
    ) -> Result<Tensor> {
        check_cancelled()?;
        let codes = codebooks_to_tensor(codebooks, &self.config, self.decoder_device()?)?;
        let latents = self
            .quantizer
            .decode(&codes, &self.config, check_cancelled)?;
        check_cancelled()?;
        let audio = self.decoder.forward(&latents, check_cancelled)?.tanh()?;
        check_cancelled()?;
        Ok(audio)
    }

    pub fn encode_reference_audio(
        &self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<FishS2VqCodes> {
        self.encode_reference_audio_with_cancel(samples, sample_rate, &|| Ok(()))
    }

    pub fn encode_reference_audio_with_cancel(
        &self,
        samples: &[f32],
        sample_rate: u32,
        check_cancelled: CancelCheck<'_>,
    ) -> Result<FishS2VqCodes> {
        check_cancelled()?;
        let prepared = prepare_reference_samples(samples, sample_rate, &self.config)?;
        check_cancelled()?;
        let device = self.encoder_device()?;
        let frames = prepared.len();
        let audio = Tensor::from_vec(prepared, (1, 1, frames), device)?;
        let codes = self.encode_tensor_with_cancel(&audio, check_cancelled)?;
        check_cancelled()?;
        let codes = tensor_to_vq_codes(&codes, &self.config)?;
        check_cancelled()?;
        Ok(codes)
    }

    pub fn encode_tensor(&self, audio: &Tensor) -> Result<Tensor> {
        self.encode_tensor_with_cancel(audio, &|| Ok(()))
    }

    fn encode_tensor_with_cancel(
        &self,
        audio: &Tensor,
        check_cancelled: CancelCheck<'_>,
    ) -> Result<Tensor> {
        check_cancelled()?;
        if audio.rank() != 3 || audio.dim(1)? != 1 {
            return Err(Error::AudioError(format!(
                "Fish S2 DAC encoder expects audio shape [B, 1, T], got {:?}",
                audio.dims()
            )));
        }
        if audio.dim(2)? > self.config.maximum_reference_samples()? {
            return Err(Error::AudioError(
                "Fish S2 DAC encoder input exceeds the codec position capacity".into(),
            ));
        }
        let z = self.encoder.forward(audio, check_cancelled)?;
        self.quantizer.encode(&z, &self.config, check_cancelled)
    }

    fn decoder_device(&self) -> Result<&candle_core::Device> {
        Ok(self.decoder.final_conv.conv.weight().device())
    }

    fn encoder_device(&self) -> Result<&candle_core::Device> {
        Ok(self.encoder.first.conv.weight().device())
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
        let dims = std::iter::once(config.latent_dim)
            .chain(std::iter::repeat_n(
                config.latent_dim,
                config.downsample_factors.len(),
            ))
            .collect::<Vec<_>>();
        let mut downsample = Vec::with_capacity(config.downsample_factors.len());
        for (idx, factor) in config.downsample_factors.iter().copied().enumerate() {
            downsample.push(FishS2DownsampleBlock::load(
                dims[idx],
                dims[idx + 1],
                factor,
                vb.pp(format!("downsample.{idx}")),
            )?);
        }
        let post_params = FishS2DacTransformerParams::quantizer_post(config);
        let pre_module = FishS2WindowLimitedTransformer::load(
            config.latent_dim,
            config.latent_dim,
            &post_params,
            true,
            vb.pp("pre_module"),
        )?;
        let post_module = FishS2WindowLimitedTransformer::load(
            config.latent_dim,
            config.latent_dim,
            &post_params,
            true,
            vb.pp("post_module"),
        )?;

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
            downsample,
            pre_module,
            post_module,
            upsample,
        })
    }

    fn encode(
        &self,
        z: &Tensor,
        config: &FishS2DacConfig,
        check_cancelled: CancelCheck<'_>,
    ) -> Result<Tensor> {
        let mut hidden = z.clone();
        for block in &self.downsample {
            check_cancelled()?;
            hidden = block.forward(&hidden)?;
        }
        hidden = self.pre_module.forward(&hidden, check_cancelled)?;
        let (semantic_z, semantic_codes) =
            self.semantic_quantizer.encode(&hidden, check_cancelled)?;
        let residual_input = hidden.broadcast_sub(&semantic_z)?;
        let (_residual_z, residual_codes) = self
            .residual_quantizer
            .encode(&residual_input, check_cancelled)?;
        check_cancelled()?;
        if semantic_codes.len() != 1 || residual_codes.len() != config.residual_codebooks {
            return Err(Error::InferenceError(
                "Fish S2 DAC quantizer produced an unexpected codebook count".to_string(),
            ));
        }
        let mut code_tensors = Vec::with_capacity(config.num_codebooks());
        code_tensors.push(semantic_codes[0].unsqueeze(1)?);
        for code in residual_codes {
            code_tensors.push(code.unsqueeze(1)?);
        }
        Tensor::cat(&code_tensors, 1).map_err(Error::from)
    }

    fn decode(
        &self,
        codes: &Tensor,
        config: &FishS2DacConfig,
        check_cancelled: CancelCheck<'_>,
    ) -> Result<Tensor> {
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
        let semantic = self
            .semantic_quantizer
            .decode_codes(&semantic_codes, check_cancelled)?;
        let residual = self
            .residual_quantizer
            .decode_codes(&residual_codes, check_cancelled)?;
        let mut z = semantic.broadcast_add(&residual)?;
        z = self.post_module.forward(&z, check_cancelled)?;
        for block in &self.upsample {
            check_cancelled()?;
            z = block.forward(&z)?;
        }
        check_cancelled()?;
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

    fn decode_codes(&self, codes: &Tensor, check_cancelled: CancelCheck<'_>) -> Result<Tensor> {
        let mut sum: Option<Tensor> = None;
        for (idx, quantizer) in self.quantizers.iter().enumerate() {
            check_cancelled()?;
            let z_p = quantizer.decode_code(&codes.i((.., idx, ..))?)?;
            let z_q = quantizer.out_proj.forward(&z_p)?;
            sum = Some(match sum {
                Some(current) => current.broadcast_add(&z_q)?,
                None => z_q,
            });
        }
        sum.ok_or_else(|| Error::AudioError("Fish S2 DAC has no quantizers".to_string()))
    }

    fn encode(
        &self,
        z: &Tensor,
        check_cancelled: CancelCheck<'_>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut residual = z.clone();
        let mut sum: Option<Tensor> = None;
        let mut codes = Vec::with_capacity(self.quantizers.len());
        for quantizer in &self.quantizers {
            check_cancelled()?;
            let (z_q_i, indices_i) = quantizer.encode(&residual)?;
            sum = Some(match sum {
                Some(current) => current.broadcast_add(&z_q_i)?,
                None => z_q_i.clone(),
            });
            residual = residual.broadcast_sub(&z_q_i)?;
            codes.push(indices_i);
        }
        let sum = sum.ok_or_else(|| {
            Error::AudioError("Fish S2 DAC cannot encode with no quantizers".to_string())
        })?;
        Ok((sum, codes))
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
        self.codebook
            .forward(codes)?
            .transpose(1, 2)
            .map_err(Error::from)
    }

    fn encode(&self, z: &Tensor) -> Result<(Tensor, Tensor)> {
        let z_e = self.in_proj.forward(z)?;
        let (z_p, indices) = self.decode_latents(&z_e)?;
        let z_q = self.out_proj.forward(&z_p)?;
        Ok((z_q, indices))
    }

    fn decode_latents(&self, latents: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, dim, frames) = latents.dims3()?;
        let encodings = latents.transpose(1, 2)?.reshape((batch * frames, dim))?;
        let encodings = l2_normalize_last_dim(&encodings)?;
        let codebook = l2_normalize_last_dim(self.codebook.embeddings())?;
        let mut dist = encodings.sqr()?.sum_keepdim(1)?;
        dist = dist.broadcast_sub(&(encodings.matmul(&codebook.t()?)? * 2.0)?)?;
        dist = dist.broadcast_add(&codebook.sqr()?.sum_keepdim(1)?.t()?)?;
        let indices = dist.argmin(1)?.reshape((batch, frames))?;
        let z_q = self.decode_code(&indices)?;
        Ok((z_q, indices))
    }
}

impl FishS2DownsampleBlock {
    fn load(in_dim: usize, out_dim: usize, factor: usize, vb: VarBuilder) -> Result<Self> {
        let conv = FishS2CausalConv1d::load(in_dim, out_dim, factor, factor, 1, vb.pp("0"))?;
        let convnext = FishS2ConvNeXtBlock::load(out_dim, vb.pp("1"))?;
        Ok(Self { conv, convnext })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.convnext.forward(&self.conv.forward(x)?)
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
        hidden = self.pwconv1.forward(&hidden)?.gelu_erf()?;
        hidden = self.pwconv2.forward(&hidden)?;
        if let Some(gamma) = &self.gamma {
            hidden = hidden.broadcast_mul(&gamma.reshape((1, 1, gamma.dim(0)?))?)?;
        }
        hidden = hidden.transpose(1, 2)?;
        residual.broadcast_add(&hidden).map_err(Error::from)
    }
}

impl FishS2DacAudioEncoder {
    fn load(config: &FishS2DacConfig, vb: VarBuilder) -> Result<Self> {
        if config.encoder_rates.len() != config.encoder_transformer_layers.len() {
            return Err(Error::ConfigError(format!(
                "Fish S2 DAC encoder rate count {} does not match transformer layer count {}",
                config.encoder_rates.len(),
                config.encoder_transformer_layers.len()
            )));
        }
        let vb = vb.pp("block");
        let first = FishS2CausalConv1d::load(1, config.encoder_dim, 7, 1, 1, vb.pp("0"))?;
        let mut blocks = Vec::with_capacity(config.encoder_rates.len());
        let mut dim = config.encoder_dim;
        for (idx, (stride, transformer_layers)) in config
            .encoder_rates
            .iter()
            .copied()
            .zip(config.encoder_transformer_layers.iter().copied())
            .enumerate()
        {
            dim = dim.checked_mul(2).ok_or_else(|| {
                Error::ConfigError("Fish S2 DAC encoder channel count overflowed".to_string())
            })?;
            blocks.push(FishS2EncoderBlock::load(
                dim,
                stride,
                transformer_layers,
                config,
                vb.pp(idx + 1),
            )?);
        }
        let final_snake = FishS2Snake1d::load(dim, vb.pp(config.encoder_rates.len() + 1))?;
        let final_conv = FishS2CausalConv1d::load(
            dim,
            config.latent_dim,
            3,
            1,
            1,
            vb.pp(config.encoder_rates.len() + 2),
        )?;
        Ok(Self {
            first,
            blocks,
            final_snake,
            final_conv,
        })
    }

    fn forward(&self, x: &Tensor, check_cancelled: CancelCheck<'_>) -> Result<Tensor> {
        check_cancelled()?;
        let mut hidden = self.first.forward(x)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, check_cancelled)?;
        }
        check_cancelled()?;
        self.final_conv.forward(&self.final_snake.forward(&hidden)?)
    }
}

impl FishS2EncoderBlock {
    fn load(
        dim: usize,
        stride: usize,
        transformer_layers: usize,
        config: &FishS2DacConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb = vb.pp("block");
        let residual_dim = dim / 2;
        let residuals = [1usize, 3, 9]
            .into_iter()
            .enumerate()
            .map(|(idx, dilation)| {
                FishS2ResidualUnit::load(residual_dim, dilation, true, vb.pp(idx))
            })
            .collect::<Result<Vec<_>>>()?;
        let snake = FishS2Snake1d::load(residual_dim, vb.pp("3"))?;
        let conv = FishS2CausalConv1d::load(residual_dim, dim, 2 * stride, stride, 1, vb.pp("4"))?;
        let transformer = if transformer_layers == 0 {
            None
        } else {
            let params = FishS2DacTransformerParams::encoder_stage(
                dim,
                transformer_layers,
                config.encoder_window_size,
                config,
            );
            Some(FishS2WindowLimitedTransformer::load(
                dim,
                dim,
                &params,
                true,
                vb.pp("5"),
            )?)
        };
        Ok(Self {
            residuals,
            snake,
            conv,
            transformer,
        })
    }

    fn forward(&self, x: &Tensor, check_cancelled: CancelCheck<'_>) -> Result<Tensor> {
        let mut hidden = x.clone();
        for residual in &self.residuals {
            check_cancelled()?;
            hidden = residual.forward(&hidden)?;
        }
        check_cancelled()?;
        hidden = self.conv.forward(&self.snake.forward(&hidden)?)?;
        if let Some(transformer) = &self.transformer {
            hidden = transformer.forward(&hidden, check_cancelled)?;
        }
        Ok(hidden)
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

    fn forward(&self, x: &Tensor, check_cancelled: CancelCheck<'_>) -> Result<Tensor> {
        check_cancelled()?;
        let mut hidden = self.first.forward(x)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, check_cancelled)?;
        }
        check_cancelled()?;
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

    fn forward(&self, x: &Tensor, check_cancelled: CancelCheck<'_>) -> Result<Tensor> {
        check_cancelled()?;
        let mut hidden = self.transposed.forward(&self.snake.forward(x)?)?;
        for residual in &self.residuals {
            check_cancelled()?;
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
            // Keep cuDNN scratch bounded by the portable workspace estimate. Profile
            // faster algorithms against an explicit workspace budget before changing it.
            #[cfg(feature = "cudnn")]
            cudnn_fwd_algo: Some(candle_core::conv::CudnnFwdAlgo::ImplicitGemm),
            ..Default::default()
        };
        let conv_vb = conv_weight_vb(&vb);
        let conv =
            load_conv1d_weight(in_channels, out_channels, kernel_size, groups, cfg, conv_vb)?;
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
        let conv_vb = conv_weight_vb(&vb);
        let conv =
            load_conv_transpose1d_weight(in_channels, out_channels, kernel_size, cfg, conv_vb)?;
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
        params: &FishS2DacTransformerParams,
        channels_first: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_proj = if input_dim != dim || vb.contains_tensor("input_proj.weight") {
            Some(candle_nn::linear(input_dim, dim, vb.pp("input_proj"))?)
        } else {
            None
        };
        let rotary = FishS2RotaryCache::new(
            params.max_seq_len,
            params.head_dim,
            params.rope_theta,
            DType::BF16,
            vb.device(),
        )?;
        let mut layers = Vec::with_capacity(params.layers);
        for idx in 0..params.layers {
            layers.push(FishS2DacTransformerBlock::load(
                params,
                rotary.clone(),
                vb.pp(format!("layers.{idx}")),
            )?);
        }
        let norm = candle_nn::rms_norm(dim, params.rms_norm_eps, vb.pp("norm"))?;
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

    fn forward(&self, x: &Tensor, check_cancelled: CancelCheck<'_>) -> Result<Tensor> {
        check_cancelled()?;
        let mut hidden = if self.channels_first {
            x.transpose(1, 2)?
        } else {
            x.clone()
        };
        if let Some(input_proj) = &self.input_proj {
            hidden = input_proj.forward(&hidden)?;
        }
        for layer in &self.layers {
            hidden = layer.forward(&hidden, check_cancelled)?;
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
    fn load(
        params: &FishS2DacTransformerParams,
        rotary: FishS2RotaryCache,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dim = params.dim;
        Ok(Self {
            attention: FishS2DacAttention::load(params, rotary, vb.pp("attention"))?,
            feed_forward: FishS2DacFeedForward::load(params, vb.pp("feed_forward"))?,
            attention_norm: candle_nn::rms_norm(dim, params.rms_norm_eps, vb.pp("attention_norm"))?,
            ffn_norm: candle_nn::rms_norm(dim, params.rms_norm_eps, vb.pp("ffn_norm"))?,
            attention_scale: vb.get((dim,), "attention_layer_scale.gamma")?,
            ffn_scale: vb.get((dim,), "ffn_layer_scale.gamma")?,
        })
    }

    fn forward(&self, x: &Tensor, check_cancelled: CancelCheck<'_>) -> Result<Tensor> {
        check_cancelled()?;
        let attn = self
            .attention
            .forward(&self.attention_norm.forward(x)?, check_cancelled)?
            .broadcast_mul(&self.attention_scale.reshape((
                1,
                1,
                self.attention_scale.dim(0)?,
            ))?)?;
        let hidden = x.broadcast_add(&attn)?;
        check_cancelled()?;
        let ff = self
            .feed_forward
            .forward(&self.ffn_norm.forward(&hidden)?)?
            .broadcast_mul(&self.ffn_scale.reshape((1, 1, self.ffn_scale.dim(0)?))?)?;
        hidden.broadcast_add(&ff).map_err(Error::from)
    }
}

impl FishS2DacAttention {
    fn load(
        params: &FishS2DacTransformerParams,
        rotary: FishS2RotaryCache,
        vb: VarBuilder,
    ) -> Result<Self> {
        let q_dim = params.heads * params.head_dim;
        let kv_dim = params.kv_heads * params.head_dim;
        let total = q_dim + 2 * kv_dim;
        Ok(Self {
            wqkv: candle_nn::linear_no_bias(q_dim, total, vb.pp("wqkv"))?,
            wo: candle_nn::linear_no_bias(q_dim, q_dim, vb.pp("wo"))?,
            num_heads: params.heads,
            num_kv_heads: params.kv_heads,
            head_dim: params.head_dim,
            rotary,
            window_size: params.window_size.max(1),
        })
    }

    fn forward(&self, x: &Tensor, check_cancelled: CancelCheck<'_>) -> Result<Tensor> {
        check_cancelled()?;
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
        let q = self.rotary.apply(&q, 0)?.transpose(1, 2)?;
        let k = self.rotary.apply(&k, 0)?.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let k = repeat_kv(&k, self.num_heads, self.num_kv_heads)?;
        let v = repeat_kv(&v, self.num_heads, self.num_kv_heads)?;
        let q = q.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;
        let k = k.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;
        let v = v.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;

        let out = windowed_attention(&q, &k, &v, self.window_size, check_cancelled)?;
        let out = out
            .reshape((bsz, self.num_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((bsz, seq_len, q_dim))?;
        self.wo.forward(&out).map_err(Error::from)
    }
}

impl FishS2DacFeedForward {
    fn load(params: &FishS2DacTransformerParams, vb: VarBuilder) -> Result<Self> {
        let dim = params.dim;
        Ok(Self {
            w1: candle_nn::linear_no_bias(dim, params.intermediate, vb.pp("w1"))?,
            w2: candle_nn::linear_no_bias(params.intermediate, dim, vb.pp("w2"))?,
            w3: candle_nn::linear_no_bias(dim, params.intermediate, vb.pp("w3"))?,
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
    if frames > FishS2DacConfig::MAX_QUANTIZER_FRAMES {
        return Err(Error::AudioError(format!(
            "Fish S2 DAC codebooks exceed the codec capacity of {} frames",
            FishS2DacConfig::MAX_QUANTIZER_FRAMES
        )));
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

fn tensor_to_vq_codes(codes: &Tensor, config: &FishS2DacConfig) -> Result<FishS2VqCodes> {
    if codes.rank() != 3 || codes.dim(0)? != 1 || codes.dim(1)? != config.num_codebooks() {
        return Err(Error::AudioError(format!(
            "Fish S2 DAC produced invalid code tensor shape {:?}",
            codes.dims()
        )));
    }
    let frames = codes.dim(2)?;
    let values = codes.to_dtype(DType::U32)?.to_vec3::<u32>()?;
    let mut codebooks = Vec::with_capacity(config.num_codebooks());
    for idx in 0..config.num_codebooks() {
        let row = values
            .first()
            .and_then(|batch| batch.get(idx))
            .ok_or_else(|| Error::AudioError("Fish S2 DAC code tensor missing row".to_string()))?;
        if row.len() != frames {
            return Err(Error::AudioError(
                "Fish S2 DAC code tensor row length mismatch".to_string(),
            ));
        }
        codebooks.push(row.clone());
    }
    Ok(FishS2VqCodes { codebooks })
}

fn prepare_reference_samples(
    samples: &[f32],
    sample_rate: u32,
    config: &FishS2DacConfig,
) -> Result<Vec<f32>> {
    if sample_rate == 0 {
        return Err(Error::InvalidInput(
            "Fish S2 reference audio sample rate must be greater than zero".to_string(),
        ));
    }
    if samples.is_empty() {
        return Err(Error::InvalidInput(
            "Fish S2 reference audio cannot be empty".to_string(),
        ));
    }
    let frames = config.reference_frame_count(samples.len(), sample_rate)?;
    let mut prepared = resample_mono_high_quality(samples, sample_rate, config.sample_rate)?;
    for sample in &mut prepared {
        if !sample.is_finite() {
            *sample = 0.0;
        }
    }
    let frame = config.samples_per_frame()?.max(1);
    let target_len = frames * frame;
    prepared.resize(target_len.max(frame), 0.0);
    Ok(prepared)
}

fn l2_normalize_last_dim(x: &Tensor) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    x.broadcast_div(&norm.maximum(1e-12f32)?)
        .map_err(Error::from)
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

fn conv_weight_vb<'a>(vb: &VarBuilder<'a>) -> VarBuilder<'a> {
    let nested = vb.pp("conv");
    if has_conv_weight(&nested) {
        nested
    } else {
        vb.clone()
    }
}

fn has_conv_weight(vb: &VarBuilder) -> bool {
    vb.contains_tensor("weight")
        || (vb.contains_tensor("weight_g") && vb.contains_tensor("weight_v"))
        || (vb.contains_tensor("parametrizations.weight.original0")
            && vb.contains_tensor("parametrizations.weight.original1"))
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

fn windowed_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    window_size: usize,
    check_cancelled: CancelCheck<'_>,
) -> Result<Tensor> {
    let (_, seq_len, head_dim) = q.dims3()?;
    if seq_len == 0 || window_size == 0 || q.dims() != k.dims() || k.dims() != v.dims() {
        return Err(Error::InferenceError(
            "Fish S2 codec attention requires matching nonempty Q/K/V and a positive window".into(),
        ));
    }
    let scale = (head_dim as f64).sqrt().recip();
    let mut blocks = Vec::with_capacity(seq_len.div_ceil(ATTENTION_QUERY_BLOCK));
    for query_start in (0..seq_len).step_by(ATTENTION_QUERY_BLOCK) {
        check_cancelled()?;
        let query_len = ATTENTION_QUERY_BLOCK.min(seq_len - query_start);
        let key_start = query_start.saturating_sub(window_size - 1);
        let key_len = query_start + query_len - key_start;
        let query = q.narrow(1, query_start, query_len)?.contiguous()?;
        let key = k.narrow(1, key_start, key_len)?.contiguous()?;
        let value = v.narrow(1, key_start, key_len)?.contiguous()?;
        let att = (query.matmul(&key.transpose(1, 2)?)? * scale)?;
        #[cfg(test)]
        ATTENTION_SCORE_ALLOCATIONS.with(|allocations| {
            allocations.borrow_mut().push(att.elem_count());
        });
        let mut mask = Vec::with_capacity(query_len * key_len);
        for row in query_start..query_start + query_len {
            let first = (row + 1).saturating_sub(window_size);
            for col in key_start..key_start + key_len {
                mask.push(if col <= row && col >= first {
                    0.0f32
                } else {
                    f32::NEG_INFINITY
                });
            }
        }
        let mask =
            Tensor::from_vec(mask, (1, query_len, key_len), q.device())?.to_dtype(q.dtype())?;
        let att = ops::softmax_last_dim(&att.broadcast_add(&mask)?)?;
        blocks.push(att.matmul(&value)?);
        check_cancelled()?;
    }
    Tensor::cat(&blocks, 1).map_err(Error::from)
}

#[cfg(test)]
thread_local! {
    static ATTENTION_SCORE_ALLOCATIONS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Shape};
    use std::collections::HashMap;

    fn tiny_config() -> FishS2DacConfig {
        FishS2DacConfig {
            sample_rate: 100,
            encoder_dim: 2,
            encoder_rates: vec![1],
            encoder_transformer_layers: vec![0],
            encoder_window_size: 4,
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

    fn insert_transformer(
        tensors: &mut HashMap<String, Tensor>,
        device: &Device,
        prefix: &str,
        dim: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        intermediate: usize,
    ) {
        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;
        tensors.insert(
            format!("{prefix}.layers.0.attention.wqkv.weight"),
            tensor(device, (q_dim + 2 * kv_dim, dim), 0.0),
        );
        tensors.insert(
            format!("{prefix}.layers.0.attention.wo.weight"),
            tensor(device, (dim, q_dim), 0.0),
        );
        tensors.insert(
            format!("{prefix}.layers.0.feed_forward.w1.weight"),
            tensor(device, (intermediate, dim), 0.0),
        );
        tensors.insert(
            format!("{prefix}.layers.0.feed_forward.w2.weight"),
            tensor(device, (dim, intermediate), 0.0),
        );
        tensors.insert(
            format!("{prefix}.layers.0.feed_forward.w3.weight"),
            tensor(device, (intermediate, dim), 0.0),
        );
        for name in [
            "layers.0.attention_norm.weight",
            "layers.0.ffn_norm.weight",
            "layers.0.attention_layer_scale.gamma",
            "layers.0.ffn_layer_scale.gamma",
            "norm.weight",
        ] {
            tensors.insert(format!("{prefix}.{name}"), tensor(device, (dim,), 1.0));
        }
    }

    fn tiny_decoder(device: &Device) -> FishS2DacDecoder {
        let config = tiny_config();
        let mut tensors = HashMap::new();
        insert_conv(&mut tensors, device, "encoder.block.0", 2, 1, 7, 0.0);
        for idx in 0..=2 {
            insert_residual_unit(
                &mut tensors,
                device,
                &format!("encoder.block.1.block.{idx}"),
                2,
            );
        }
        insert_snake(&mut tensors, device, "encoder.block.1.block.3", 2);
        insert_conv(
            &mut tensors,
            device,
            "encoder.block.1.block.4",
            4,
            2,
            2,
            0.01,
        );
        insert_snake(&mut tensors, device, "encoder.block.2", 4);
        insert_conv(&mut tensors, device, "encoder.block.3", 4, 4, 3, 0.01);

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

        insert_conv(
            &mut tensors,
            device,
            "quantizer.downsample.0.0",
            4,
            4,
            1,
            0.0,
        );
        insert_convnext(&mut tensors, device, "quantizer.downsample.0.1", 4);
        insert_transformer(&mut tensors, device, "quantizer.pre_module", 4, 2, 2, 2, 8);
        insert_transformer(&mut tensors, device, "quantizer.post_module", 4, 2, 2, 2, 8);

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
    fn causal_conv_loads_direct_weight_norm_without_conv_child() {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert("weight_g".to_string(), tensor(&device, (4, 1, 1), 1.0));
        tensors.insert("weight_v".to_string(), tensor(&device, (4, 2, 1), 0.5));
        tensors.insert("bias".to_string(), tensor(&device, (4,), 0.0));
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let conv = FishS2CausalConv1d::load(2, 4, 1, 1, 1, vb).unwrap();
        let input = Tensor::zeros((1, 2, 3), DType::F32, &device).unwrap();
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 4, 3]);
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
    fn dac_encode_reference_audio_returns_codebook_major_codes() {
        let device = Device::Cpu;
        let codec = tiny_decoder(&device);
        let codes = codec
            .encode_reference_audio(&[0.0, 0.25, -0.25], 100)
            .unwrap();
        assert_eq!(codes.codebooks.len(), 2);
        assert_eq!(codes.codebooks[0].len(), 3);
        assert!(codes.codebooks.iter().flatten().all(|code| *code < 8));
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

    #[test]
    fn dac_attention_matches_adjacent_pair_reference() {
        let device = Device::Cpu;
        let params = FishS2DacTransformerParams {
            dim: 4,
            layers: 1,
            heads: 1,
            kv_heads: 1,
            head_dim: 4,
            intermediate: 8,
            max_seq_len: 3,
            window_size: 3,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
        };
        let mut qkv = vec![0.0f32; 12 * 4];
        for column in 0..4 {
            qkv[column * 4 + column] = 1.0;
            qkv[(column + 4) * 4 + column] = [0.5, 1.1, -0.7, 0.9][column];
            qkv[(column + 8) * 4 + column] = 1.0;
        }
        let mut tensors = HashMap::new();
        tensors.insert(
            "wqkv.weight".into(),
            Tensor::from_vec(qkv, (12, 4), &device).unwrap(),
        );
        tensors.insert(
            "wo.weight".into(),
            Tensor::eye(4, DType::F32, &device).unwrap(),
        );
        let rotary = FishS2RotaryCache::new(3, 4, 10_000.0, DType::BF16, &device).unwrap();
        let attention = FishS2DacAttention::load(
            &params,
            rotary,
            VarBuilder::from_tensors(tensors, DType::F32, &device),
        )
        .unwrap();
        let input = Tensor::from_vec(
            vec![
                0.4f32, -0.7, 1.2, 0.3, -0.2, 0.9, -0.5, 1.1, 1.3, 0.1, 0.2, -0.8,
            ],
            (1, 3, 4),
            &device,
        )
        .unwrap();
        // Independent scalar oracle: published DAC adjacent-pair rotation with
        // BF16-rounded cos/sin, q=x, k=x*[.5,1.1,-.7,.9], v=x, causal softmax.
        let expected = [
            0.4f32,
            -0.7,
            1.2,
            0.3,
            -0.0083854375,
            0.38902783,
            0.042907927,
            0.8445139,
            0.6966825,
            0.25058666,
            0.1083576,
            -0.04114733,
        ];
        let actual = attention
            .forward(&input, &|| Ok(()))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "element {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn convnext_uses_exact_gelu() {
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        insert_convnext(&mut tensors, &device, "block", 2);
        tensors.insert(
            "block.pwconv1.bias".into(),
            Tensor::from_vec(
                vec![-3.0f32, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                (8,),
                &device,
            )
            .unwrap(),
        );
        let mut projection = vec![0.0f32; 16];
        projection[0] = 1.0;
        projection[9] = 1.0;
        tensors.insert(
            "block.pwconv2.weight".into(),
            Tensor::from_vec(projection, (2, 8), &device).unwrap(),
        );
        let block = FishS2ConvNeXtBlock::load(
            2,
            VarBuilder::from_tensors(tensors, DType::F32, &device).pp("block"),
        )
        .unwrap();
        let output = block
            .forward(&Tensor::zeros((1, 2, 1), DType::F32, &device).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // x/2 * (1 + erf(x/sqrt(2))); the tanh approximation differs by 4.1e-4.
        assert!((output[0] - -0.004049694f32).abs() < 1e-6);
        assert!((output[1] - 2.9959503f32).abs() < 1e-6);
    }

    #[test]
    fn quantizer_normalization_clamps_small_norms() {
        let input = Tensor::from_vec(
            vec![3.0f32, 4.0, 3e-13, 4e-13, 0.0, 0.0],
            (3, 2),
            &Device::Cpu,
        )
        .unwrap();
        let actual = l2_normalize_last_dim(&input)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for (row, expected) in actual.iter().zip([[0.6f32, 0.8], [0.3, 0.4], [0.0, 0.0]]) {
            for (actual, expected) in row.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn codec_position_limits_are_checked_before_allocation() {
        let config = FishS2DacConfig::current();
        assert_eq!(config.maximum_reference_samples().unwrap(), 8_388_608);
        assert_eq!(config.rotary_cache_bytes().unwrap(), 6_291_456);
        assert_eq!(config.reference_frame_count(16_000, 16_000).unwrap(), 22);
        assert_eq!(
            config.reference_frame_count(8_388_608, 44_100).unwrap(),
            4_096
        );
        assert!(config.reference_frame_count(8_388_609, 44_100).is_err());
        assert!(config.reference_frame_count(usize::MAX, 1).is_err());
        let rows = vec![vec![0; 4_097]; config.num_codebooks()];
        assert!(codebooks_to_tensor(&rows, &config, &Device::Cpu)
            .unwrap_err()
            .to_string()
            .contains("capacity"));
    }

    fn dense_attention_oracle(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        heads: usize,
        frames: usize,
        dim: usize,
        window: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0f32; q.len()];
        for head in 0..heads {
            for row in 0..frames {
                let first = (row + 1).saturating_sub(window);
                let mut logits = Vec::new();
                for col in first..=row {
                    let score = (0..dim)
                        .map(|channel| {
                            f64::from(q[(head * frames + row) * dim + channel])
                                * f64::from(k[(head * frames + col) * dim + channel])
                        })
                        .sum::<f64>()
                        / (dim as f64).sqrt();
                    logits.push(score);
                }
                let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let sum = logits.iter().map(|score| (score - max).exp()).sum::<f64>();
                for channel in 0..dim {
                    output[(head * frames + row) * dim + channel] = logits
                        .iter()
                        .enumerate()
                        .map(|(offset, score)| {
                            (score - max).exp() / sum
                                * f64::from(v[(head * frames + first + offset) * dim + channel])
                        })
                        .sum::<f64>()
                        as f32;
                }
            }
        }
        output
    }

    #[test]
    fn blocked_attention_matches_dense_oracle_across_windows_and_block_edges() {
        let device = Device::Cpu;
        for frames in [1usize, 63, 64, 65, 129] {
            let (heads, dim) = (2, 4);
            let q = (0..heads * frames * dim)
                .map(|i| (i as f32 * 0.13).sin())
                .collect::<Vec<_>>();
            let k = (0..q.len())
                .map(|i| (i as f32 * 0.17).cos())
                .collect::<Vec<_>>();
            let v = (0..q.len())
                .map(|i| (i as f32 * 0.19).sin())
                .collect::<Vec<_>>();
            let query = Tensor::from_vec(q.clone(), (heads, frames, dim), &device).unwrap();
            let key = Tensor::from_vec(k.clone(), (heads, frames, dim), &device).unwrap();
            let value = Tensor::from_vec(v.clone(), (heads, frames, dim), &device).unwrap();
            for window in [1usize, 7, 64, 128] {
                let expected = dense_attention_oracle(&q, &k, &v, heads, frames, dim, window);
                let actual = windowed_attention(&query, &key, &value, window, &|| Ok(()))
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    assert!(
                        (actual - expected).abs() < 2e-6,
                        "frames={frames}, window={window}, element={index}: {actual} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn actual_attention_score_allocations_stay_bounded_when_sequence_doubles() {
        let (heads, window) = (2usize, 128usize);
        let mut maxima = Vec::new();
        for frames in [1024usize, 2048] {
            ATTENTION_SCORE_ALLOCATIONS.with(|allocations| allocations.borrow_mut().clear());
            let x = Tensor::zeros((heads, frames, 2), DType::F32, &Device::Cpu).unwrap();
            let output = windowed_attention(&x, &x, &x, window, &|| Ok(())).unwrap();
            assert_eq!(output.dims(), &[heads, frames, 2]);
            let allocations =
                ATTENTION_SCORE_ALLOCATIONS.with(|allocations| allocations.borrow().clone());
            assert_eq!(allocations.len(), frames.div_ceil(ATTENTION_QUERY_BLOCK));
            let max = allocations.into_iter().max().unwrap();
            assert!(max <= heads * ATTENTION_QUERY_BLOCK * (window + ATTENTION_QUERY_BLOCK - 1));
            assert!(max < heads * frames * frames / 16);
            maxima.push(max);
        }
        assert_eq!(maxima[0], maxima[1]);
    }

    #[test]
    fn attention_cancellation_stops_before_the_next_score_allocation() {
        let calls = std::cell::Cell::new(0usize);
        let cancel = || {
            calls.set(calls.get() + 1);
            if calls.get() == 3 {
                Err(Error::Cancelled("test".into()))
            } else {
                Ok(())
            }
        };
        ATTENTION_SCORE_ALLOCATIONS.with(|allocations| allocations.borrow_mut().clear());
        let x = Tensor::zeros((1, 129, 4), DType::F32, &Device::Cpu).unwrap();
        assert!(matches!(
            windowed_attention(&x, &x, &x, 7, &cancel),
            Err(Error::Cancelled(_))
        ));
        assert_eq!(calls.get(), 3);
        ATTENTION_SCORE_ALLOCATIONS.with(|allocations| assert_eq!(allocations.borrow().len(), 1));
    }

    #[test]
    fn codec_cancellation_propagates_and_leaves_the_next_request_usable() {
        let codec = tiny_decoder(&Device::Cpu);
        let cancel_now = || Err(Error::Cancelled("test".into()));
        assert!(matches!(
            codec.encode_reference_audio_with_cancel(&[], 0, &cancel_now),
            Err(Error::Cancelled(_))
        ));
        let calls = std::cell::Cell::new(0usize);
        let cancel_during_encode = || {
            calls.set(calls.get() + 1);
            if calls.get() == 8 {
                Err(Error::Cancelled("test".into()))
            } else {
                Ok(())
            }
        };
        assert!(matches!(
            codec.encode_reference_audio_with_cancel(&[0.0, 0.2, -0.2], 100, &cancel_during_encode),
            Err(Error::Cancelled(_))
        ));
        let codes = codec
            .encode_reference_audio(&[0.0, 0.2, -0.2], 100)
            .unwrap();
        calls.set(0);
        let cancel_during_decode = || {
            calls.set(calls.get() + 1);
            if calls.get() == 4 {
                Err(Error::Cancelled("test".into()))
            } else {
                Ok(())
            }
        };
        assert!(matches!(
            codec.decode_vq_codes_with_cancel(&codes, &cancel_during_decode),
            Err(Error::Cancelled(_))
        ));
        let audio = codec.decode_vq_codes(&codes).unwrap();
        assert_eq!(audio.len(), 3);
        assert!(audio.iter().all(|sample| sample.is_finite()));
    }
}
