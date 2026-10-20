use candle_core::quantized::{GgmlDType, QMatMul};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::ops;
use candle_nn::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, LayerNorm, Linear, Module};

use crate::error::{Error, Result};
use crate::kernels::depthwise_conv::{depthwise_conv1d, DepthwiseConv1dSpec};
use crate::models::architectures::parakeet::asr::metal_kernels as parakeet_metal_kernels;
use crate::models::shared::weights::gguf::GgufLoader;

use super::config::Lfm25AudioEncoderConfig;

pub struct Lfm25AudioEncoder {
    pre_encode: ConvSubsamplingDw,
    layers: Vec<ConformerLayer>,
    adapter: AudioAdapter,
    cfg: Lfm25AudioEncoderConfig,
}

impl Lfm25AudioEncoder {
    pub fn load(
        loader: &GgufLoader,
        cfg: Lfm25AudioEncoderConfig,
        device: &Device,
    ) -> Result<Self> {
        let pre_encode = ConvSubsamplingDw::load(loader, &cfg, device)?;

        let mut layers = Vec::with_capacity(cfg.block_count);
        for idx in 0..cfg.block_count {
            layers.push(ConformerLayer::load(loader, &cfg, device, idx)?);
        }

        let adapter = AudioAdapter::load(loader, &cfg, device)?;

        Ok(Self {
            pre_encode,
            layers,
            adapter,
            cfg,
        })
    }

    pub fn encode(&self, features: &Tensor, feature_frames: usize) -> Result<Tensor> {
        let (mut x, encoded_len) = self.pre_encode.forward(features, feature_frames)?;
        x = x.narrow(1, 0, encoded_len)?;

        let pos_emb = self.rel_positional_embedding(encoded_len, x.device())?;
        for layer in &self.layers {
            x = layer.forward(&x, &pos_emb)?;
        }

        self.adapter.forward(&x)
    }

    fn rel_positional_embedding(&self, encoded_len: usize, device: &Device) -> Result<Tensor> {
        build_rel_positional_embedding(encoded_len, self.cfg.embedding_length, device)
    }
}

struct ConvSubsamplingDw {
    conv0: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv5: Conv2d,
    conv6: Conv2d,
    out: ConformerLinear,
}

impl ConvSubsamplingDw {
    fn load(loader: &GgufLoader, cfg: &Lfm25AudioEncoderConfig, device: &Device) -> Result<Self> {
        let stride_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let point_cfg = Conv2dConfig {
            stride: 1,
            padding: 0,
            ..Default::default()
        };

        let conv0 = load_conv2d_any(
            loader,
            device,
            1,
            cfg.subsampling_channels,
            3,
            stride_cfg,
            &[
                "a.conv1d.0.weight".to_string(),
                "audio_encoder.pre_encode.conv.0.weight".to_string(),
            ],
            &[
                "a.conv1d.0.bias".to_string(),
                "audio_encoder.pre_encode.conv.0.bias".to_string(),
            ],
        )?;

        let mut dw_stride_cfg = stride_cfg;
        dw_stride_cfg.groups = cfg.subsampling_channels;
        let conv2 = load_conv2d_any(
            loader,
            device,
            cfg.subsampling_channels,
            cfg.subsampling_channels,
            3,
            dw_stride_cfg,
            &[
                "a.conv1d.2.weight".to_string(),
                "audio_encoder.pre_encode.conv.2.weight".to_string(),
            ],
            &[
                "a.conv1d.2.bias".to_string(),
                "audio_encoder.pre_encode.conv.2.bias".to_string(),
            ],
        )?;
        let conv3 = load_conv2d_any(
            loader,
            device,
            cfg.subsampling_channels,
            cfg.subsampling_channels,
            1,
            point_cfg,
            &[
                "a.conv1d.3.weight".to_string(),
                "audio_encoder.pre_encode.conv.3.weight".to_string(),
            ],
            &[
                "a.conv1d.3.bias".to_string(),
                "audio_encoder.pre_encode.conv.3.bias".to_string(),
            ],
        )?;
        let conv5 = load_conv2d_any(
            loader,
            device,
            cfg.subsampling_channels,
            cfg.subsampling_channels,
            3,
            dw_stride_cfg,
            &[
                "a.conv1d.5.weight".to_string(),
                "audio_encoder.pre_encode.conv.5.weight".to_string(),
            ],
            &[
                "a.conv1d.5.bias".to_string(),
                "audio_encoder.pre_encode.conv.5.bias".to_string(),
            ],
        )?;
        let conv6 = load_conv2d_any(
            loader,
            device,
            cfg.subsampling_channels,
            cfg.subsampling_channels,
            1,
            point_cfg,
            &[
                "a.conv1d.6.weight".to_string(),
                "audio_encoder.pre_encode.conv.6.weight".to_string(),
            ],
            &[
                "a.conv1d.6.bias".to_string(),
                "audio_encoder.pre_encode.conv.6.bias".to_string(),
            ],
        )?;

        let out = load_linear_any(
            loader,
            device,
            cfg.subsampling_channels * 16,
            cfg.embedding_length,
            &[
                "a.pre_encode.out.weight".to_string(),
                "audio_encoder.pre_encode.out.weight".to_string(),
            ],
            &[
                "a.pre_encode.out.bias".to_string(),
                "audio_encoder.pre_encode.out.bias".to_string(),
            ],
        )?;

        Ok(Self {
            conv0,
            conv2,
            conv3,
            conv5,
            conv6,
            out,
        })
    }

    fn forward(&self, features: &Tensor, feature_frames: usize) -> Result<(Tensor, usize)> {
        let mut x = features.transpose(1, 2)?.unsqueeze(1)?; // [B, 1, T, M]

        x = self.conv0.forward(&x)?;
        x = x.relu()?;

        x = forward_depthwise_conv2d(&self.conv2, &x)?;
        x = self.conv3.forward(&x)?;
        x = x.relu()?;

        x = forward_depthwise_conv2d(&self.conv5, &x)?;
        x = self.conv6.forward(&x)?;
        x = x.relu()?;

        let (batch, channels, time, freq) = x.dims4()?;
        let x = x
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, time, channels * freq))?;
        let x = self.out.forward(&x)?;

        Ok((x, subsampled_len_3x(feature_frames).min(time)))
    }
}

struct ConformerLayer {
    ff1_norm: LayerNorm,
    ff1: FeedForward,
    attn_norm: LayerNorm,
    self_attn: RelPosSelfAttention,
    conv_norm: LayerNorm,
    conv: ConformerConv,
    ff2_norm: LayerNorm,
    ff2: FeedForward,
    final_norm: LayerNorm,
}

impl ConformerLayer {
    fn load(
        loader: &GgufLoader,
        cfg: &Lfm25AudioEncoderConfig,
        device: &Device,
        idx: usize,
    ) -> Result<Self> {
        let ff1_norm = load_layer_norm_any(
            loader,
            device,
            cfg.embedding_length,
            cfg.attention_layer_norm_epsilon,
            &[
                format!("a.blk.{idx}.ffn_norm.weight"),
                format!("audio_encoder.layers.{idx}.ff1_norm.weight"),
            ],
            &[
                format!("a.blk.{idx}.ffn_norm.bias"),
                format!("audio_encoder.layers.{idx}.ff1_norm.bias"),
            ],
        )?;
        let ff1 = FeedForward::load(loader, cfg, device, idx, false)?;

        let attn_norm = load_layer_norm_any(
            loader,
            device,
            cfg.embedding_length,
            cfg.attention_layer_norm_epsilon,
            &[
                format!("a.blk.{idx}.ln1.weight"),
                format!("a.blk.{idx}.attn_norm.weight"),
                format!("a.blk.{idx}.attention_norm.weight"),
                format!("audio_encoder.layers.{idx}.attn_norm.weight"),
            ],
            &[
                format!("a.blk.{idx}.ln1.bias"),
                format!("a.blk.{idx}.attn_norm.bias"),
                format!("a.blk.{idx}.attention_norm.bias"),
                format!("audio_encoder.layers.{idx}.attn_norm.bias"),
            ],
        )?;
        let self_attn = RelPosSelfAttention::load(loader, cfg, device, idx)?;

        let conv_norm = load_layer_norm_any(
            loader,
            device,
            cfg.embedding_length,
            cfg.attention_layer_norm_epsilon,
            &[
                format!("a.blk.{idx}.norm_conv.weight"),
                format!("audio_encoder.layers.{idx}.conv_norm.weight"),
            ],
            &[
                format!("a.blk.{idx}.norm_conv.bias"),
                format!("audio_encoder.layers.{idx}.conv_norm.bias"),
            ],
        )?;
        let conv = ConformerConv::load(loader, cfg, device, idx)?;

        let ff2_norm = load_layer_norm_any(
            loader,
            device,
            cfg.embedding_length,
            cfg.attention_layer_norm_epsilon,
            &[
                format!("a.blk.{idx}.ffn_norm_1.weight"),
                format!("audio_encoder.layers.{idx}.ff2_norm.weight"),
            ],
            &[
                format!("a.blk.{idx}.ffn_norm_1.bias"),
                format!("audio_encoder.layers.{idx}.ff2_norm.bias"),
            ],
        )?;
        let ff2 = FeedForward::load(loader, cfg, device, idx, true)?;

        let final_norm = load_layer_norm_any(
            loader,
            device,
            cfg.embedding_length,
            cfg.attention_layer_norm_epsilon,
            &[
                format!("a.blk.{idx}.ln2.weight"),
                format!("a.blk.{idx}.final_norm.weight"),
                format!("audio_encoder.layers.{idx}.final_norm.weight"),
            ],
            &[
                format!("a.blk.{idx}.ln2.bias"),
                format!("a.blk.{idx}.final_norm.bias"),
                format!("audio_encoder.layers.{idx}.final_norm.bias"),
            ],
        )?;

        Ok(Self {
            ff1_norm,
            ff1,
            attn_norm,
            self_attn,
            conv_norm,
            conv,
            ff2_norm,
            ff2,
            final_norm,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let mut residual = x.clone();

        let ff1 = self.ff1.forward(&self.ff1_norm.forward(&residual)?)?;
        residual = residual.broadcast_add(&ff1.affine(0.5, 0.0)?)?;

        let attn = self
            .self_attn
            .forward(&self.attn_norm.forward(&residual)?, pos_emb)?;
        residual = residual.broadcast_add(&attn)?;

        let conv = self.conv.forward(&self.conv_norm.forward(&residual)?)?;
        residual = residual.broadcast_add(&conv)?;

        let ff2 = self.ff2.forward(&self.ff2_norm.forward(&residual)?)?;
        residual = residual.broadcast_add(&ff2.affine(0.5, 0.0)?)?;

        self.final_norm.forward(&residual).map_err(Error::from)
    }
}

struct FeedForward {
    linear1: ConformerLinear,
    linear2: ConformerLinear,
}

impl FeedForward {
    fn load(
        loader: &GgufLoader,
        cfg: &Lfm25AudioEncoderConfig,
        device: &Device,
        idx: usize,
        second: bool,
    ) -> Result<Self> {
        let suffix = if second { "_1" } else { "" };
        let hf_prefix = if second { "ff2" } else { "ff1" };
        let linear1 = load_linear_any(
            loader,
            device,
            cfg.embedding_length,
            cfg.feed_forward_length,
            &[
                format!("a.blk.{idx}.ffn_up{suffix}.weight"),
                format!("audio_encoder.layers.{idx}.{hf_prefix}.linear1.weight"),
            ],
            &[
                format!("a.blk.{idx}.ffn_up{suffix}.bias"),
                format!("audio_encoder.layers.{idx}.{hf_prefix}.linear1.bias"),
            ],
        )?;
        let linear2 = load_linear_any(
            loader,
            device,
            cfg.feed_forward_length,
            cfg.embedding_length,
            &[
                format!("a.blk.{idx}.ffn_down{suffix}.weight"),
                format!("audio_encoder.layers.{idx}.{hf_prefix}.linear2.weight"),
            ],
            &[
                format!("a.blk.{idx}.ffn_down{suffix}.bias"),
                format!("audio_encoder.layers.{idx}.{hf_prefix}.linear2.bias"),
            ],
        )?;
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = swish(&x)?;
        self.linear2.forward(&x).map_err(Error::from)
    }
}

struct ConformerConv {
    pointwise_conv1: Conv1d,
    depthwise_conv: Conv1d,
    affine_norm: AffineNorm1d,
    folded_depthwise_weight: Tensor,
    folded_depthwise_bias: Tensor,
    pointwise_conv2: Conv1d,
}

impl ConformerConv {
    fn load(
        loader: &GgufLoader,
        cfg: &Lfm25AudioEncoderConfig,
        device: &Device,
        idx: usize,
    ) -> Result<Self> {
        let pointwise_conv1 = load_conv1d_any(
            loader,
            device,
            cfg.embedding_length,
            cfg.embedding_length * 2,
            1,
            Conv1dConfig::default(),
            &[
                format!("a.blk.{idx}.conv_pw1.weight"),
                format!("audio_encoder.layers.{idx}.conv.pointwise_conv1.weight"),
            ],
            &[
                format!("a.blk.{idx}.conv_pw1.bias"),
                format!("audio_encoder.layers.{idx}.conv.pointwise_conv1.bias"),
            ],
        )?;
        let depthwise_conv = load_conv1d_any(
            loader,
            device,
            cfg.embedding_length,
            cfg.embedding_length,
            9,
            Conv1dConfig {
                padding: 4,
                groups: cfg.embedding_length,
                ..Default::default()
            },
            &[
                format!("a.blk.{idx}.conv_dw.weight"),
                format!("audio_encoder.layers.{idx}.conv.depthwise_conv.weight"),
            ],
            &[
                format!("a.blk.{idx}.conv_dw.bias"),
                format!("audio_encoder.layers.{idx}.conv.depthwise_conv.bias"),
            ],
        )?;
        let affine_norm = AffineNorm1d::load(
            loader,
            device,
            cfg.embedding_length,
            &[
                format!("a.blk.{idx}.conv_norm.weight"),
                format!("audio_encoder.layers.{idx}.conv.norm.weight"),
            ],
            &[
                format!("a.blk.{idx}.conv_norm.bias"),
                format!("audio_encoder.layers.{idx}.conv.norm.bias"),
            ],
        )?;
        let pointwise_conv2 = load_conv1d_any(
            loader,
            device,
            cfg.embedding_length,
            cfg.embedding_length,
            1,
            Conv1dConfig::default(),
            &[
                format!("a.blk.{idx}.conv_pw2.weight"),
                format!("audio_encoder.layers.{idx}.conv.pointwise_conv2.weight"),
            ],
            &[
                format!("a.blk.{idx}.conv_pw2.bias"),
                format!("audio_encoder.layers.{idx}.conv.pointwise_conv2.bias"),
            ],
        )?;
        let (folded_depthwise_weight, folded_depthwise_bias) =
            fold_depthwise_conv_affine(&depthwise_conv, &affine_norm)?;

        Ok(Self {
            pointwise_conv1,
            depthwise_conv,
            affine_norm,
            folded_depthwise_weight,
            folded_depthwise_bias,
            pointwise_conv2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = x.dim(2)?;
        let mut x = x.transpose(1, 2)?;

        x = self.pointwise_conv1.forward(&x)?;
        let x_a = x.i((.., ..hidden, ..))?;
        let x_b = x.i((.., hidden.., ..))?;
        x = x_a.broadcast_mul(&ops::sigmoid(&x_b)?)?;

        x = self.depthwise_affine_swish(&x)?;
        x = self.pointwise_conv2.forward(&x)?;

        x.transpose(1, 2).map_err(Error::from)
    }

    fn depthwise_affine_swish(&self, x: &Tensor) -> Result<Tensor> {
        if use_cuda_depthwise_candidate(&self.depthwise_conv, x) {
            // Fold the inference affine transform before the activation, as on Metal.
            return candle_depthwise_conv1d(
                x,
                &self.folded_depthwise_weight,
                &self.folded_depthwise_bias,
                CUDA_DEPTHWISE_PRODUCT_ELEMENTS,
            )?
            .silu()
            .map_err(Error::from);
        }
        if let Some(fused) = try_forward_depthwise_conv1d_affine_swish(
            &self.depthwise_conv,
            &self.folded_depthwise_weight,
            &self.folded_depthwise_bias,
            x,
        ) {
            return Ok(fused);
        }
        if let Some(y) = try_metal_depthwise_conv1d(&self.depthwise_conv, x)? {
            return swish(&self.affine_norm.forward(&y)?);
        }
        // CPU (and CUDA without the opt-in): the shared shift-and-accumulate op
        // with the folded affine, instead of Candle's per-channel grouped conv.
        let cfg = self.depthwise_conv.config();
        let y = depthwise_conv1d(
            x,
            &self.folded_depthwise_weight,
            Some(&self.folded_depthwise_bias),
            DepthwiseConv1dSpec::symmetric(cfg.padding).with_dilation(cfg.dilation),
        )?;
        swish(&y)
    }
}

fn fold_depthwise_conv_affine(
    conv: &Conv1d,
    affine_norm: &AffineNorm1d,
) -> Result<(Tensor, Tensor)> {
    let weight = conv.weight();
    let channels = affine_norm.weight.dim(0)?;
    let scale = affine_norm.weight.reshape((channels, 1, 1))?;
    let folded_weight = weight.broadcast_mul(&scale)?.contiguous()?;
    let conv_bias = match conv.bias() {
        Some(bias) => bias.clone(),
        None => Tensor::zeros(
            channels,
            affine_norm.weight.dtype(),
            affine_norm.weight.device(),
        )?,
    };
    let folded_bias = conv_bias
        .broadcast_mul(&affine_norm.weight)?
        .broadcast_add(&affine_norm.bias)?
        .contiguous()?;
    Ok((folded_weight, folded_bias))
}

fn try_forward_depthwise_conv1d_affine_swish(
    conv: &Conv1d,
    folded_weight: &Tensor,
    folded_bias: &Tensor,
    x: &Tensor,
) -> Option<Tensor> {
    let cfg = conv.config();
    parakeet_metal_kernels::try_depthwise_conv1d_bias_swish(
        x,
        folded_weight,
        folded_bias,
        cfg.padding,
        cfg.stride,
        cfg.dilation,
    )
}

// Experimental CUDA path: at most 1 MiB of F32 products per chunk. This is
// additional to the padded input, reduced chunks and concatenated output. Keep
// opt-in until GPU profiling establishes a benefit and validates memory usage.
const CUDA_DEPTHWISE_PRODUCT_ELEMENTS: usize = 1024 * 1024 / 4;

fn use_cuda_depthwise_candidate(conv: &Conv1d, x: &Tensor) -> bool {
    let cfg = conv.config();
    x.device().is_cuda()
        && x.dtype() == DType::F32
        && cfg.padding == 4
        && cfg.stride == 1
        && cfg.dilation == 1
        && x.dims().len() == 3
        && x.dims()[0] > 0
        && x.dims()[1] > 0
        && x.dims()[2] > 0
        && cfg.groups == x.dims()[1]
        && conv.weight().dims() == [x.dims()[1], 1, 9]
        && x.dims()[0].saturating_mul(x.dims()[1]).saturating_mul(9)
            <= CUDA_DEPTHWISE_PRODUCT_ELEMENTS
        && cuda_depthwise_conv_opted_in()
}

fn cuda_depthwise_conv_opted_in() -> bool {
    static ENABLED: crate::env::ReadOnce<bool> =
        crate::env::ReadOnce::new(|| crate::env::flag_or("IZWI_LFM_CUDA_DEPTHWISE_CONV", false));
    ENABLED.get()
}

// Candle's grouped Conv1d dispatches once per channel. Unfold instead keeps all
// channels in each multiply/reduce, while chunks bound the expanded product.
fn candle_depthwise_conv1d(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    max_product_elements: usize,
) -> Result<Tensor> {
    let (batch, channels, len) = x.dims3()?;
    let chunk_len = (max_product_elements / (batch * channels * 9)).max(1);
    let windows = x.pad_with_zeros(2, 4, 4)?.unfold(2, 9, 1)?;
    let weight = weight.reshape((1, channels, 1, 9))?;
    let bias = bias.reshape((1, channels, 1))?;
    let mut chunks = Vec::with_capacity(len.div_ceil(chunk_len));
    for offset in (0..len).step_by(chunk_len) {
        let width = chunk_len.min(len - offset);
        let y = windows
            .narrow(2, offset, width)?
            .broadcast_mul(&weight)?
            .sum(3)?
            .broadcast_add(&bias)?;
        chunks.push(y);
    }
    Tensor::cat(&chunks, 2).map_err(Error::from)
}

fn try_metal_depthwise_conv1d(conv: &Conv1d, x: &Tensor) -> Result<Option<Tensor>> {
    let cfg = conv.config();
    let Some(mut y) = parakeet_metal_kernels::try_depthwise_conv1d(
        x,
        conv.weight(),
        cfg.padding,
        cfg.stride,
        cfg.dilation,
    ) else {
        return Ok(None);
    };

    if let Some(bias) = conv.bias() {
        let channels = bias.dims1()?;
        y = y.broadcast_add(&bias.reshape((1, channels, 1))?)?;
    }

    Ok(Some(y))
}

fn forward_depthwise_conv2d(conv: &Conv2d, x: &Tensor) -> Result<Tensor> {
    let cfg = conv.config();
    let Some(mut y) = parakeet_metal_kernels::try_depthwise_conv2d(
        x,
        conv.weight(),
        cfg.padding,
        cfg.stride,
        cfg.dilation,
    ) else {
        return conv.forward(x).map_err(Error::from);
    };

    if let Some(bias) = conv.bias() {
        let channels = bias.dims1()?;
        y = y.broadcast_add(&bias.reshape((1, channels, 1, 1))?)?;
    }

    Ok(y)
}

struct RelPosSelfAttention {
    q_proj: ConformerLinear,
    k_proj: ConformerLinear,
    v_proj: ConformerLinear,
    out_proj: ConformerLinear,
    pos_proj: ConformerLinear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
    n_heads: usize,
    head_dim: usize,
}

impl RelPosSelfAttention {
    fn load(
        loader: &GgufLoader,
        cfg: &Lfm25AudioEncoderConfig,
        device: &Device,
        idx: usize,
    ) -> Result<Self> {
        let q_proj = load_linear_any(
            loader,
            device,
            cfg.embedding_length,
            cfg.embedding_length,
            &[
                format!("a.blk.{idx}.attn_q.weight"),
                format!("a.blk.{idx}.q_proj.weight"),
                format!("audio_encoder.layers.{idx}.attn.q_proj.weight"),
            ],
            &[
                format!("a.blk.{idx}.attn_q.bias"),
                format!("a.blk.{idx}.q_proj.bias"),
                format!("audio_encoder.layers.{idx}.attn.q_proj.bias"),
            ],
        )?;
        let k_proj = load_linear_any(
            loader,
            device,
            cfg.embedding_length,
            cfg.embedding_length,
            &[
                format!("a.blk.{idx}.attn_k.weight"),
                format!("a.blk.{idx}.k_proj.weight"),
                format!("audio_encoder.layers.{idx}.attn.k_proj.weight"),
            ],
            &[
                format!("a.blk.{idx}.attn_k.bias"),
                format!("a.blk.{idx}.k_proj.bias"),
                format!("audio_encoder.layers.{idx}.attn.k_proj.bias"),
            ],
        )?;
        let v_proj = load_linear_any(
            loader,
            device,
            cfg.embedding_length,
            cfg.embedding_length,
            &[
                format!("a.blk.{idx}.attn_v.weight"),
                format!("a.blk.{idx}.v_proj.weight"),
                format!("audio_encoder.layers.{idx}.attn.v_proj.weight"),
            ],
            &[
                format!("a.blk.{idx}.attn_v.bias"),
                format!("a.blk.{idx}.v_proj.bias"),
                format!("audio_encoder.layers.{idx}.attn.v_proj.bias"),
            ],
        )?;
        let out_proj = load_linear_any(
            loader,
            device,
            cfg.embedding_length,
            cfg.embedding_length,
            &[
                format!("a.blk.{idx}.attn_out.weight"),
                format!("a.blk.{idx}.attn_output.weight"),
                format!("a.blk.{idx}.attn_o.weight"),
                format!("audio_encoder.layers.{idx}.attn.out_proj.weight"),
            ],
            &[
                format!("a.blk.{idx}.attn_out.bias"),
                format!("a.blk.{idx}.attn_output.bias"),
                format!("a.blk.{idx}.attn_o.bias"),
                format!("audio_encoder.layers.{idx}.attn.out_proj.bias"),
            ],
        )?;
        let pos_proj = load_linear_any(
            loader,
            device,
            cfg.embedding_length,
            cfg.embedding_length,
            &[
                format!("a.blk.{idx}.linear_pos.weight"),
                format!("audio_encoder.layers.{idx}.attn.pos_proj.weight"),
            ],
            &[
                format!("a.blk.{idx}.linear_pos.bias"),
                format!("audio_encoder.layers.{idx}.attn.pos_proj.bias"),
            ],
        )?;

        let pos_bias_u = load_tensor_any(
            loader,
            device,
            &[
                format!("a.blk.{idx}.pos_bias_u"),
                format!("audio_encoder.layers.{idx}.attn.pos_bias_u"),
            ],
            DType::F32,
        )?;
        let pos_bias_v = load_tensor_any(
            loader,
            device,
            &[
                format!("a.blk.{idx}.pos_bias_v"),
                format!("audio_encoder.layers.{idx}.attn.pos_bias_v"),
            ],
            DType::F32,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            pos_proj,
            pos_bias_u,
            pos_bias_v,
            n_heads: cfg.attention_head_count,
            head_dim: cfg.embedding_length / cfg.attention_head_count,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        let q = self
            .q_proj
            .forward(x)?
            .reshape((batch, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((batch, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((batch, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let p = self.projected_pos_embedding(pos_emb, seq_len)?;

        let pos_bias_u = self
            .pos_bias_u
            .reshape((1, self.n_heads, 1, self.head_dim))?;
        let pos_bias_v = self
            .pos_bias_v
            .reshape((1, self.n_heads, 1, self.head_dim))?;

        let q_u = q.broadcast_add(&pos_bias_u)?;
        let q_v = q.broadcast_add(&pos_bias_v)?;
        let matrix_ac = q_u.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let matrix_bd = rel_shift(&q_v.matmul(&p.transpose(2, 3)?.contiguous()?)?)?;
        let matrix_bd = matrix_bd.narrow(3, 0, seq_len)?;

        let scores = matrix_ac
            .broadcast_add(&matrix_bd)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let attn = if scores.device().is_cuda() {
            ops::softmax_last_dim(&scores.contiguous()?)?
        } else {
            ops::softmax(&scores, 3)?
        };
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((
            batch,
            seq_len,
            self.n_heads * self.head_dim,
        ))?;

        self.out_proj.forward(&out).map_err(Error::from)
    }

    fn projected_pos_embedding(&self, pos_emb: &Tensor, seq_len: usize) -> Result<Tensor> {
        self.pos_proj
            .forward(pos_emb)?
            .reshape((1, 2 * seq_len - 1, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
            .map_err(Error::from)
    }
}

struct AudioAdapter {
    norm: LayerNorm,
    linear1: ConformerLinear,
    linear2: ConformerLinear,
}

impl AudioAdapter {
    fn load(loader: &GgufLoader, cfg: &Lfm25AudioEncoderConfig, device: &Device) -> Result<Self> {
        let norm = load_layer_norm_any(
            loader,
            device,
            cfg.embedding_length,
            cfg.attention_layer_norm_epsilon,
            &[
                "mm.a.mlp.0.weight".to_string(),
                "audio_adapter.layers.0.weight".to_string(),
            ],
            &[
                "mm.a.mlp.0.bias".to_string(),
                "audio_adapter.layers.0.bias".to_string(),
            ],
        )?;
        let linear1 = load_linear_any(
            loader,
            device,
            cfg.embedding_length,
            cfg.projection_dim,
            &[
                "mm.a.mlp.1.weight".to_string(),
                "audio_adapter.layers.1.weight".to_string(),
            ],
            &[
                "mm.a.mlp.1.bias".to_string(),
                "audio_adapter.layers.1.bias".to_string(),
            ],
        )?;
        let linear2 = load_linear_any(
            loader,
            device,
            cfg.projection_dim,
            cfg.projection_dim,
            &[
                "mm.a.mlp.3.weight".to_string(),
                "audio_adapter.layers.3.weight".to_string(),
            ],
            &[
                "mm.a.mlp.3.bias".to_string(),
                "audio_adapter.layers.3.bias".to_string(),
            ],
        )?;
        Ok(Self {
            norm,
            linear1,
            linear2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward(x)?;
        let x = self.linear1.forward(&x)?;
        let x = gelu(&x)?;
        self.linear2.forward(&x).map_err(Error::from)
    }
}

struct AffineNorm1d {
    weight: Tensor,
    bias: Tensor,
}

impl AffineNorm1d {
    fn load(
        loader: &GgufLoader,
        device: &Device,
        channels: usize,
        weight_names: &[String],
        bias_names: &[String],
    ) -> Result<Self> {
        let weight = load_vector_any(loader, device, weight_names, channels)?;
        let bias = load_optional_vector_any(loader, device, bias_names, channels)?
            .unwrap_or(Tensor::zeros(channels, DType::F32, device)?);
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let channels = self.weight.dim(0)?;
        let weight = self.weight.reshape((1, channels, 1))?;
        let bias = self.bias.reshape((1, channels, 1))?;
        x.broadcast_mul(&weight)?
            .broadcast_add(&bias)
            .map_err(Error::from)
    }
}

fn load_layer_norm_any(
    loader: &GgufLoader,
    device: &Device,
    dim: usize,
    eps: f64,
    weight_names: &[String],
    bias_names: &[String],
) -> Result<LayerNorm> {
    let weight = load_vector_any(loader, device, weight_names, dim)?;
    let bias = load_optional_vector_any(loader, device, bias_names, dim)?.unwrap_or(Tensor::zeros(
        dim,
        DType::F32,
        device,
    )?);
    Ok(LayerNorm::new(weight, bias, eps))
}

/// A projection of the encoder or adapter: on Metal a quantized checkpoint
/// weight stays quantized (`QMatMul`, whose ggml kernels take the F32
/// activations as they are); elsewhere, and for a float weight, a dense F32
/// `Linear`. CPU and CUDA quantize the activations of a quantized matmul, so
/// they keep the dequantized weight.
#[derive(Debug, Clone)]
enum ConformerLinear {
    Dense(Linear),
    Quantized {
        weight: QMatMul,
        bias: Option<Tensor>,
    },
}

impl From<Linear> for ConformerLinear {
    fn from(linear: Linear) -> Self {
        Self::Dense(linear)
    }
}

impl Module for ConformerLinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(x),
            Self::Quantized { weight, bias } => {
                let y = weight.forward(&x.contiguous()?)?;
                match bias {
                    Some(bias) => y.broadcast_add(bias),
                    None => Ok(y),
                }
            }
        }
    }
}

/// Whether a stored weight of `dtype` loads as a quantized projection on
/// `device` ([`ConformerLinear`]).
fn conformer_projection_stays_quantized(dtype: GgmlDType, device: &Device) -> bool {
    device.is_metal() && !matches!(dtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16)
}

/// Whether `name` is a projection weight the encoder and adapter load
/// through [`load_linear_any`] (and so may keep quantized).
pub(crate) fn lfm25_audio_conformer_projection(name: &str) -> bool {
    const LAYER: [&str; 9] = [
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_out",
        "linear_pos",
        "ffn_up",
        "ffn_down",
        "ffn_up_1",
        "ffn_down_1",
    ];
    if matches!(
        name,
        "a.pre_encode.out.weight" | "mm.a.mlp.1.weight" | "mm.a.mlp.3.weight"
    ) {
        return true;
    }
    name.strip_prefix("a.blk.")
        .and_then(|rest| rest.split_once('.'))
        .is_some_and(|(layer, rest)| {
            layer.parse::<usize>().is_ok()
                && rest
                    .strip_suffix(".weight")
                    .is_some_and(|projection| LAYER.contains(&projection))
        })
}

fn load_linear_any(
    loader: &GgufLoader,
    device: &Device,
    in_dim: usize,
    out_dim: usize,
    weight_names: &[String],
    bias_names: &[String],
) -> Result<ConformerLinear> {
    let mut selected = None;
    let mut mismatches = Vec::new();
    for name in weight_names {
        let Some(shape) = loader.tensor_shape(name) else {
            continue;
        };
        match shape.as_slice() {
            [rows, cols] if (*rows, *cols) == (out_dim, in_dim) => {
                selected = Some((name.clone(), false));
                break;
            }
            [rows, cols] if (*rows, *cols) == (in_dim, out_dim) => {
                selected = Some((name.clone(), true));
                break;
            }
            [rows, cols] => mismatches.push(format!("{name}: got ({rows}, {cols})")),
            _ => mismatches.push(format!("{name}: expected rank 2, found shape {shape:?}")),
        }
    }

    let Some((selected_name, transpose)) = selected else {
        if mismatches.is_empty() {
            return Err(Error::ModelLoadError(format!(
                "Missing GGUF tensor; tried {}",
                weight_names.join(" | ")
            )));
        }
        return Err(Error::ModelLoadError(format!(
            "Linear weight shape mismatch for {}: expected ({out_dim}, {in_dim}), candidates {}",
            weight_names.join(" | "),
            mismatches.join("; ")
        )));
    };

    let bias = load_optional_vector_any(loader, device, bias_names, out_dim)?;
    let stored = loader.load_qtensor(&selected_name, device)?;
    if !transpose && conformer_projection_stays_quantized(stored.dtype(), device) {
        return Ok(ConformerLinear::Quantized {
            weight: QMatMul::from_qtensor(stored)?,
            bias,
        });
    }
    let mut weight = stored.dequantize(device)?;
    if transpose {
        weight = weight.transpose(0, 1)?.contiguous()?;
    }
    Ok(Linear::new(weight, bias).into())
}

fn load_conv2d_any(
    loader: &GgufLoader,
    device: &Device,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv2dConfig,
    weight_names: &[String],
    bias_names: &[String],
) -> Result<Conv2d> {
    let in_per_group = in_channels / cfg.groups.max(1);
    let expected_oihw = (out_channels, in_per_group, kernel_size, kernel_size);
    let expected_ohwi = (out_channels, kernel_size, kernel_size, in_per_group);

    let mut selected = None;
    let mut mismatches = Vec::new();
    for name in weight_names {
        let Some(shape) = loader.tensor_shape(name) else {
            continue;
        };
        match shape.as_slice() {
            [o, i, kh, kw] if (*o, *i, *kh, *kw) == expected_oihw => {
                selected = Some((name.clone(), false));
                break;
            }
            [o, kh, kw, i] if (*o, *kh, *kw, *i) == expected_ohwi => {
                selected = Some((name.clone(), true));
                break;
            }
            _ => mismatches.push(format!("{name}: got {shape:?}")),
        }
    }

    let Some((selected_name, permute_ohwi)) = selected else {
        if mismatches.is_empty() {
            return Err(Error::ModelLoadError(format!(
                "Missing GGUF tensor; tried {}",
                weight_names.join(" | ")
            )));
        }
        return Err(Error::ModelLoadError(format!(
            "Conv2d weight shape mismatch for {}: expected OIHW={expected_oihw:?} or OHWI={expected_ohwi:?}, candidates {}",
            weight_names.join(" | "),
            mismatches.join("; ")
        )));
    };

    let mut weight = loader.load_tensor(&selected_name, DType::F32, device)?;
    if permute_ohwi {
        weight = weight.permute((0, 3, 1, 2))?.contiguous()?;
    }

    let bias = load_optional_vector_any(loader, device, bias_names, out_channels)?;
    Ok(Conv2d::new(weight, bias, cfg))
}

fn load_conv1d_any(
    loader: &GgufLoader,
    device: &Device,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    cfg: Conv1dConfig,
    weight_names: &[String],
    bias_names: &[String],
) -> Result<Conv1d> {
    let in_per_group = in_channels / cfg.groups.max(1);
    let expected_oik = (out_channels, in_per_group, kernel_size);
    let expected_oki = (out_channels, kernel_size, in_per_group);

    enum Conv1dLayout {
        Oik,
        Oki,
        Matrix,
        DepthwiseMatrix,
    }

    let mut selected = None;
    let mut mismatches = Vec::new();
    for name in weight_names {
        let Some(shape) = loader.tensor_shape(name) else {
            continue;
        };
        match shape.as_slice() {
            [o, i, k] if (*o, *i, *k) == expected_oik => {
                selected = Some((name.clone(), Conv1dLayout::Oik));
                break;
            }
            [o, k, i] if (*o, *k, *i) == expected_oki => {
                selected = Some((name.clone(), Conv1dLayout::Oki));
                break;
            }
            [o, i] if kernel_size == 1 && (*o, *i) == (out_channels, in_per_group) => {
                selected = Some((name.clone(), Conv1dLayout::Matrix));
                break;
            }
            [o, k] if in_per_group == 1 && (*o, *k) == (out_channels, kernel_size) => {
                selected = Some((name.clone(), Conv1dLayout::DepthwiseMatrix));
                break;
            }
            [..] => mismatches.push(format!("{name}: got {shape:?}")),
        }
    }

    let Some((selected_name, layout)) = selected else {
        if mismatches.is_empty() {
            return Err(Error::ModelLoadError(format!(
                "Missing GGUF tensor; tried {}",
                weight_names.join(" | ")
            )));
        }
        return Err(Error::ModelLoadError(format!(
            "Conv1d weight shape mismatch for {}: expected OIK={expected_oik:?}, OKI={expected_oki:?}, matrix=({}, {}), or depthwise-matrix=({}, {}), candidates {}",
            weight_names.join(" | "),
            out_channels,
            in_per_group,
            out_channels,
            kernel_size,
            mismatches.join("; ")
        )));
    };

    let mut weight = loader.load_tensor(&selected_name, DType::F32, device)?;
    match layout {
        Conv1dLayout::Oik => {}
        Conv1dLayout::Oki => {
            weight = weight.permute((0, 2, 1))?.contiguous()?;
        }
        Conv1dLayout::Matrix => {
            weight = weight.unsqueeze(2)?;
        }
        Conv1dLayout::DepthwiseMatrix => {
            weight = weight.unsqueeze(1)?;
        }
    }

    let bias = load_optional_vector_any(loader, device, bias_names, out_channels)?;
    Ok(Conv1d::new(weight, bias, cfg))
}

fn load_vector_any(
    loader: &GgufLoader,
    device: &Device,
    names: &[String],
    expected_len: usize,
) -> Result<Tensor> {
    let mut selected = None;
    let mut mismatches = Vec::new();
    for name in names {
        let Some(shape) = loader.tensor_shape(name) else {
            continue;
        };
        let len = shape_elem_count(&shape);
        if len == expected_len {
            selected = Some(name.clone());
            break;
        }
        mismatches.push(format!("{name}: found {len} values from shape {shape:?}"));
    }

    let Some(selected_name) = selected else {
        if mismatches.is_empty() {
            return Err(Error::ModelLoadError(format!(
                "Missing GGUF tensor; tried {}",
                names.join(" | ")
            )));
        }
        return Err(Error::ModelLoadError(format!(
            "Vector shape mismatch for {}: expected {expected_len} values, candidates {}",
            names.join(" | "),
            mismatches.join("; ")
        )));
    };

    let tensor = loader.load_tensor(&selected_name, DType::F32, device)?;
    if tensor.rank() == 1 {
        Ok(tensor)
    } else {
        tensor.flatten_all().map_err(Error::from)
    }
}

fn load_optional_vector_any(
    loader: &GgufLoader,
    device: &Device,
    names: &[String],
    expected_len: usize,
) -> Result<Option<Tensor>> {
    let mut mismatches = Vec::new();
    for name in names {
        let Some(shape) = loader.tensor_shape(name) else {
            continue;
        };
        let len = shape_elem_count(&shape);
        if len == expected_len {
            return load_vector_any(loader, device, std::slice::from_ref(name), expected_len)
                .map(Some);
        }
        mismatches.push(format!("{name}: found {len} values from shape {shape:?}"));
    }
    if !mismatches.is_empty() {
        return Err(Error::ModelLoadError(format!(
            "Vector shape mismatch for {}: expected {expected_len} values, candidates {}",
            names.join(" | "),
            mismatches.join("; ")
        )));
    }
    Ok(None)
}

fn load_tensor_any(
    loader: &GgufLoader,
    device: &Device,
    names: &[String],
    dtype: DType,
) -> Result<Tensor> {
    for name in names {
        if loader.has_tensor(name) {
            return loader.load_tensor(name, dtype, device);
        }
    }
    Err(Error::ModelLoadError(format!(
        "Missing GGUF tensor; tried {}",
        names.join(" | ")
    )))
}

fn shape_elem_count(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

fn build_rel_positional_embedding(len: usize, d_model: usize, device: &Device) -> Result<Tensor> {
    if len == 0 {
        return Err(Error::InvalidInput(
            "Cannot build positional embedding for empty sequence".to_string(),
        ));
    }

    let pos_len = 2 * len - 1;
    let mut emb = vec![0f32; pos_len * d_model];
    let denom = (10_000f32).ln() / d_model as f32;

    for pos in 0..pos_len {
        let relative = (len as isize - pos as isize - 1) as f32;
        for idx in (0..d_model).step_by(2) {
            let inv_freq = (-denom * idx as f32).exp();
            let angle = relative * inv_freq;
            emb[pos * d_model + idx] = angle.sin();
            if idx + 1 < d_model {
                emb[pos * d_model + idx + 1] = angle.cos();
            }
        }
    }

    Tensor::from_vec(emb, (1, pos_len, d_model), device).map_err(Error::from)
}

fn rel_shift(x: &Tensor) -> Result<Tensor> {
    let (batch, heads, q_len, pos_len) = x.dims4()?;
    let x = x.pad_with_zeros(3, 1, 0)?;
    let x = x.reshape((batch, heads, pos_len + 1, q_len))?;
    let x = x.narrow(2, 1, pos_len)?;
    x.reshape((batch, heads, q_len, pos_len))
        .map_err(Error::from)
}

fn swish(x: &Tensor) -> Result<Tensor> {
    if x.device().is_cuda() {
        return x.silu().map_err(Error::from);
    }
    x.broadcast_mul(&ops::sigmoid(x)?).map_err(Error::from)
}

/// Exact (erf) GELU: liquid-audio's adapter `MLP` uses `nn.GELU()`
/// (`approximate="none"`), not the tanh approximation.
fn gelu(x: &Tensor) -> Result<Tensor> {
    x.gelu_erf().map_err(Error::from)
}

pub fn subsampled_len_3x(mut len: usize) -> usize {
    for _ in 0..3 {
        len = len.div_ceil(2);
    }
    len
}

#[cfg(test)]
mod tests {
    use std::fs::{self, File};
    use std::path::{Path, PathBuf};

    use candle_core::quantized::gguf_file::{write as write_gguf, Value as GgufValue};
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device, Module, Tensor};
    use candle_nn::{Conv1d, Conv1dConfig, Linear};
    use uuid::Uuid;

    use super::{
        build_rel_positional_embedding, fold_depthwise_conv_affine, load_linear_any, rel_shift,
        subsampled_len_3x, AffineNorm1d, Lfm25AudioEncoder, RelPosSelfAttention,
    };
    use crate::models::architectures::lfm25_audio::config::parse_audio_encoder_config;
    use crate::models::shared::weights::gguf::GgufLoader;

    fn make_temp_dir() -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("izwi-lfm25-audio-conformer-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    fn write_test_gguf(
        path: &Path,
        metadata: Vec<(&'static str, GgufValue)>,
        tensors: Vec<(&'static str, Tensor)>,
    ) {
        let quantized = tensors
            .into_iter()
            .map(|(name, tensor)| {
                Ok::<_, candle_core::Error>((name, QTensor::quantize(&tensor, GgmlDType::F32)?))
            })
            .collect::<std::result::Result<Vec<_>, _>>()
            .expect("quantize tensors");
        let metadata_refs = metadata
            .iter()
            .map(|(name, value)| (*name, value))
            .collect::<Vec<_>>();
        let tensor_refs = quantized
            .iter()
            .map(|(name, tensor)| (*name, tensor))
            .collect::<Vec<_>>();
        let mut file = File::create(path).expect("create gguf");
        write_gguf(&mut file, &metadata_refs, &tensor_refs).expect("write gguf");
    }

    fn local_model_dir(name: &str) -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        PathBuf::from(home)
            .join("Library/Application Support/izwi/models")
            .join(name)
    }

    fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32) {
        assert_eq!(actual.dims(), expected.dims());
        let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (index, (a, b)) in actual.iter().zip(&expected).enumerate() {
            assert!((a - b).abs() <= tolerance, "index {index}: {a} != {b}");
        }
    }

    #[test]
    fn native_candle_conformer_operations_match_portable_formulations() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(
            (0..257)
                .map(|i| -8.0f32 + i as f32 / 16.0)
                .collect::<Vec<_>>(),
            (1, 257),
            &device,
        )
        .unwrap();
        assert_close(&x.silu().unwrap(), &super::swish(&x).unwrap(), 1e-6);
        assert_close(&x.gelu_erf().unwrap(), &super::gelu(&x).unwrap(), 1e-6);
        // torch.nn.functional.gelu(torch.tensor([1., -1., 3.])), approximate="none".
        let probe = Tensor::new(&[1.0f32, -1.0, 3.0], &device).unwrap();
        let values = super::gelu(&probe).unwrap().to_vec1::<f32>().unwrap();
        let expected = [0.841_344_7f32, -0.158_655_26, 2.995_950_2];
        for (got, want) in values.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "gelu {got} != {want}");
        }
        // Non-contiguous scores exercise the layout normalization required by
        // Candle's dedicated last-dimension softmax kernel.
        let scores = Tensor::from_vec(
            (0..3 * 257)
                .map(|i| (i as f32 * 0.17).sin() * 8.0)
                .collect::<Vec<_>>(),
            (3, 257),
            &device,
        )
        .unwrap()
        .transpose(0, 1)
        .unwrap();
        assert_close(
            &candle_nn::ops::softmax_last_dim(&scores.contiguous().unwrap()).unwrap(),
            &candle_nn::ops::softmax(&scores, 1).unwrap(),
            1e-6,
        );
    }

    fn check_chunked_depthwise(device: Device) {
        for (batch, channels, len) in [(1, 1, 1), (1, 3, 7), (2, 3, 13), (2, 4, 16), (3, 5, 31)] {
            let input = Tensor::from_vec(
                (0..batch * channels * len)
                    .map(|i| (i as f32 * 0.13).sin())
                    .collect::<Vec<_>>(),
                (batch, channels, len),
                &device,
            )
            .unwrap();
            let weight = Tensor::from_vec(
                (0..channels * 9)
                    .map(|i| (i as f32 * 0.21).cos())
                    .collect::<Vec<_>>(),
                (channels, 1, 9),
                &device,
            )
            .unwrap();
            let bias = Tensor::from_vec(
                (0..channels).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
                channels,
                &device,
            )
            .unwrap();
            let conv = Conv1d::new(
                weight,
                Some(bias),
                Conv1dConfig {
                    padding: 4,
                    groups: channels,
                    ..Default::default()
                },
            );
            let affine = AffineNorm1d {
                weight: Tensor::from_vec(
                    (0..channels)
                        .map(|i| if i % 2 == 0 { 1.25f32 } else { -0.75 })
                        .collect::<Vec<_>>(),
                    channels,
                    &device,
                )
                .unwrap(),
                bias: Tensor::from_vec(vec![0.2f32; channels], channels, &device).unwrap(),
            };
            let expected =
                super::swish(&affine.forward(&conv.forward(&input).unwrap()).unwrap()).unwrap();
            let (weight, bias) = fold_depthwise_conv_affine(&conv, &affine).unwrap();
            for chunk in [1, 5, len] {
                let actual = super::candle_depthwise_conv1d(
                    &input,
                    &weight,
                    &bias,
                    batch * channels * 9 * chunk,
                )
                .unwrap()
                .silu()
                .unwrap();
                assert_close(&actual, &expected, 1e-5);
            }
            if !device.is_cuda() {
                assert!(!super::use_cuda_depthwise_candidate(&conv, &input));
            }
        }
    }

    /// The CPU conformer conv path (shared shift-and-accumulate op with the
    /// folded affine) matches Candle's grouped conv, the affine and swish.
    #[test]
    fn conformer_depthwise_shared_op_matches_candle_grouped_conv_and_affine() {
        let device = Device::Cpu;
        let mut state = 0xBB67_AE85_84CA_A73Bu64;
        let mut tensor = |shape: &[usize], scale: f32, offset: f32| {
            let n = shape.iter().product::<usize>();
            let values = (0..n)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    offset + scale * ((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
                })
                .collect::<Vec<_>>();
            Tensor::from_vec(values, shape, &device).unwrap()
        };
        let channels = 48;
        let depthwise_conv = Conv1d::new(
            tensor(&[channels, 1, 9], 0.4, 0.0),
            Some(tensor(&[channels], 0.1, 0.0)),
            Conv1dConfig {
                padding: 4,
                groups: channels,
                ..Default::default()
            },
        );
        let affine_norm = AffineNorm1d {
            weight: tensor(&[channels], 0.5, 1.0),
            bias: tensor(&[channels], 0.2, 0.0),
        };
        let (folded_depthwise_weight, folded_depthwise_bias) =
            fold_depthwise_conv_affine(&depthwise_conv, &affine_norm).unwrap();
        let pointwise = Conv1d::new(
            tensor(&[channels, channels, 1], 0.1, 0.0),
            None,
            Conv1dConfig::default(),
        );
        let conv = super::ConformerConv {
            pointwise_conv1: pointwise.clone(),
            depthwise_conv,
            affine_norm,
            folded_depthwise_weight,
            folded_depthwise_bias,
            pointwise_conv2: pointwise,
        };
        let x = tensor(&[2, channels, 37], 1.0, 0.0);
        let calls = crate::kernels::depthwise_conv::depthwise_conv1d_calls();
        let actual = conv.depthwise_affine_swish(&x).unwrap();
        assert!(crate::kernels::depthwise_conv::depthwise_conv1d_calls() > calls);
        let expected = super::swish(
            &conv
                .affine_norm
                .forward(&conv.depthwise_conv.forward(&x).unwrap())
                .unwrap(),
        )
        .unwrap();
        assert_close(&actual, &expected, 1e-5);
    }

    #[test]
    fn chunked_candle_depthwise_matches_grouped_convolution_and_affine() {
        check_chunked_depthwise(Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an NVIDIA GPU; run explicitly with --ignored"]
    fn cuda_candle_conformer_operations_and_depthwise_match_reference() {
        let device = Device::new_cuda(0).expect("CUDA device required");
        check_chunked_depthwise(device.clone());
        let reference = Tensor::from_vec(
            (0..257)
                .map(|i| -8.0f32 + i as f32 / 16.0)
                .collect::<Vec<_>>(),
            (1, 257),
            &Device::Cpu,
        )
        .unwrap();
        let input = reference.to_device(&device).unwrap();
        assert_close(
            &super::swish(&input).unwrap(),
            &super::swish(&reference).unwrap(),
            2e-6,
        );
        assert_close(
            &super::gelu(&input).unwrap(),
            &super::gelu(&reference).unwrap(),
            2e-6,
        );
        assert_close(
            &candle_nn::ops::softmax_last_dim(&input).unwrap(),
            &candle_nn::ops::softmax(&reference, 1).unwrap(),
            1e-6,
        );
    }

    #[test]
    fn subsampled_len_matches_three_stride_two_layers() {
        assert_eq!(subsampled_len_3x(1), 1);
        assert_eq!(subsampled_len_3x(2), 1);
        assert_eq!(subsampled_len_3x(3), 1);
        assert_eq!(subsampled_len_3x(8), 1);
        assert_eq!(subsampled_len_3x(9), 2);
        assert_eq!(subsampled_len_3x(100), 13);
    }

    #[test]
    fn relative_position_embedding_has_expected_shape() {
        let device = Device::Cpu;
        let emb = build_rel_positional_embedding(5, 8, &device).expect("build embedding");
        assert_eq!(emb.dims3().unwrap(), (1, 9, 8));
    }

    #[test]
    fn folded_depthwise_conv_affine_matches_explicit_path() {
        let device = Device::Cpu;
        let cfg = Conv1dConfig {
            padding: 1,
            groups: 2,
            ..Default::default()
        };
        let weight = Tensor::from_vec(
            vec![
                1.0f32, 2.0, 3.0, //
                -1.0, 0.5, 0.25,
            ],
            (2, 1, 3),
            &device,
        )
        .expect("conv weight");
        let bias = Tensor::from_vec(vec![0.5f32, -0.25], 2, &device).expect("conv bias");
        let conv = Conv1d::new(weight, Some(bias), cfg);
        let affine_norm = AffineNorm1d {
            weight: Tensor::from_vec(vec![2.0f32, -3.0], 2, &device).expect("affine weight"),
            bias: Tensor::from_vec(vec![0.1f32, 0.2], 2, &device).expect("affine bias"),
        };
        let input = Tensor::from_vec(
            vec![
                0.1f32, 0.2, 0.3, 0.4, //
                -0.5, -0.4, -0.3, -0.2,
            ],
            (1, 2, 4),
            &device,
        )
        .expect("input");

        let (folded_weight, folded_bias) =
            fold_depthwise_conv_affine(&conv, &affine_norm).expect("fold affine");
        let folded_conv = Conv1d::new(folded_weight, Some(folded_bias), cfg);
        let expected = affine_norm
            .forward(&conv.forward(&input).expect("conv forward"))
            .expect("affine forward");
        let actual = folded_conv.forward(&input).expect("folded conv forward");

        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "folded conv mismatch at {idx}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn rel_shift_preserves_relative_attention_layout() {
        let device = Device::Cpu;
        let x = Tensor::arange(0u32, 15u32, &device)
            .unwrap()
            .reshape((1, 1, 3, 5))
            .unwrap()
            .to_dtype(candle_core::DType::F32)
            .unwrap();
        let shifted = rel_shift(&x).expect("rel shift");
        assert_eq!(shifted.dims4().unwrap(), (1, 1, 3, 5));
    }

    #[test]
    fn projected_relative_position_embedding_is_request_local() {
        let device = Device::Cpu;
        let identity = Tensor::eye(4, candle_core::DType::F32, &device).expect("identity");
        let zero_bias = Tensor::zeros(4, candle_core::DType::F32, &device).expect("bias");
        let linear =
            || super::ConformerLinear::from(Linear::new(identity.clone(), Some(zero_bias.clone())));
        let attn = RelPosSelfAttention {
            q_proj: linear(),
            k_proj: linear(),
            v_proj: linear(),
            out_proj: linear(),
            pos_proj: linear(),
            pos_bias_u: Tensor::zeros((2, 2), candle_core::DType::F32, &device).expect("bias u"),
            pos_bias_v: Tensor::zeros((2, 2), candle_core::DType::F32, &device).expect("bias v"),
            n_heads: 2,
            head_dim: 2,
        };
        let pos_emb =
            build_rel_positional_embedding(5, 4, &device).expect("relative position embedding");

        let first = attn
            .projected_pos_embedding(&pos_emb, 5)
            .expect("first projection");
        let second = attn
            .projected_pos_embedding(&pos_emb, 5)
            .expect("cached projection");

        assert_eq!(first.dims4().unwrap(), (1, 2, 9, 2));
        assert_eq!(second.dims4().unwrap(), (1, 2, 9, 2));
        assert_eq!(
            first.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            second.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn load_linear_any_skips_incompatible_alias_shapes() {
        let dir = make_temp_dir();
        let path = dir.join("linear.gguf");
        write_test_gguf(
            &path,
            vec![],
            vec![
                (
                    "wrong.weight",
                    Tensor::zeros((32, 32), candle_core::DType::F32, &Device::Cpu)
                        .expect("wrong weight"),
                ),
                (
                    "right.weight",
                    Tensor::zeros((64, 32), candle_core::DType::F32, &Device::Cpu)
                        .expect("right weight"),
                ),
                (
                    "right.bias",
                    Tensor::zeros(64, candle_core::DType::F32, &Device::Cpu).expect("right bias"),
                ),
            ],
        );

        let loader = GgufLoader::from_path(&path).expect("load gguf");
        let linear = load_linear_any(
            &loader,
            &Device::Cpu,
            32,
            64,
            &["wrong.weight".to_string(), "right.weight".to_string()],
            &["wrong.bias".to_string(), "right.bias".to_string()],
        )
        .expect("load linear");

        let input =
            Tensor::zeros((1, 32), candle_core::DType::F32, &Device::Cpu).expect("input tensor");
        let output = linear.forward(&input).expect("forward");
        assert_eq!(output.dims2().unwrap(), (1, 64));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_local_lfm25_audio_encoder_smoke_if_available() {
        let model_dir = local_model_dir("LFM2.5-Audio-1.5B-GGUF");
        let path = model_dir.join("mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf");
        if !path.exists() {
            return;
        }

        let loader = GgufLoader::from_path(&path).expect("load mmproj");
        let config = parse_audio_encoder_config(&loader).expect("parse encoder config");
        assert_eq!(config.embedding_length, 512);
        assert_eq!(config.feed_forward_length, 2048);

        let _encoder =
            Lfm25AudioEncoder::load(&loader, config, &Device::Cpu).expect("load audio encoder");
    }

    /// On Metal a Q4_0 projection stays quantized and takes F32 activations
    /// as they are: its output matches the dequantized weight's to F32
    /// rounding. Elsewhere it loads dense.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_quantized_projection_matches_the_dequantized_weight() {
        use super::{conformer_projection_stays_quantized, ConformerLinear};
        assert!(!conformer_projection_stays_quantized(
            GgmlDType::Q4_0,
            &Device::Cpu
        ));
        let Some((metal, _lease)) = crate::backends::metal_test_device_or_skip(
            "metal_quantized_projection_matches_the_dequantized_weight",
        ) else {
            return;
        };
        assert!(conformer_projection_stays_quantized(
            GgmlDType::Q4_0,
            &metal
        ));
        assert!(!conformer_projection_stays_quantized(
            GgmlDType::F32,
            &metal
        ));
        let weight =
            crate::kernels::numeric_check::xorshift_tensor(&[96, 128], 0xC0F0, 0.2, &Device::Cpu)
                .unwrap();
        let stored = QTensor::quantize(&weight, GgmlDType::Q4_0).unwrap();
        let dense = stored.dequantize(&metal).unwrap();
        let bias = Tensor::arange(0f32, 96.0, &metal).unwrap();
        let input =
            crate::kernels::numeric_check::xorshift_tensor(&[2, 7, 128], 0xFEED, 1.0, &metal)
                .unwrap();
        let quantized = ConformerLinear::Quantized {
            weight: candle_core::quantized::QMatMul::from_qtensor(
                QTensor::quantize(&weight.to_device(&metal).unwrap(), GgmlDType::Q4_0).unwrap(),
            )
            .unwrap(),
            bias: Some(bias.clone()),
        };
        let expected = Linear::new(dense, Some(bias)).forward(&input).unwrap();
        let actual = quantized.forward(&input).unwrap();
        assert_eq!(actual.dims(), &[2, 7, 96]);
        let values = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let difference =
            crate::kernels::numeric_check::max_abs_diff(&values(&actual), &values(&expected));
        assert!(difference < 1e-3, "{difference}");
    }
}
