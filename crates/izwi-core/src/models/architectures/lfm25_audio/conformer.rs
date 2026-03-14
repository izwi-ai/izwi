use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::ops;
use candle_nn::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, LayerNorm, Linear, Module};

use crate::error::{Error, Result};
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

        let pos_emb =
            build_rel_positional_embedding(encoded_len, self.cfg.embedding_length, x.device())?;
        for layer in &self.layers {
            x = layer.forward(&x, &pos_emb)?;
        }

        self.adapter.forward(&x)
    }
}

struct ConvSubsamplingDw {
    conv0: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv5: Conv2d,
    conv6: Conv2d,
    out: Linear,
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
        dw_stride_cfg.groups = cfg.feed_forward_length / 2;
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

        x = self.conv2.forward(&x)?;
        x = self.conv3.forward(&x)?;
        x = x.relu()?;

        x = self.conv5.forward(&x)?;
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
                format!("a.blk.{idx}.attn_norm.weight"),
                format!("a.blk.{idx}.attention_norm.weight"),
                format!("audio_encoder.layers.{idx}.attn_norm.weight"),
            ],
            &[
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
                format!("a.blk.{idx}.final_norm.weight"),
                format!("audio_encoder.layers.{idx}.final_norm.weight"),
            ],
            &[
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
    linear1: Linear,
    linear2: Linear,
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

        Ok(Self {
            pointwise_conv1,
            depthwise_conv,
            affine_norm,
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

        x = self.depthwise_conv.forward(&x)?;
        x = self.affine_norm.forward(&x)?;
        x = swish(&x)?;
        x = self.pointwise_conv2.forward(&x)?;

        x.transpose(1, 2).map_err(Error::from)
    }
}

struct RelPosSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    pos_proj: Linear,
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
                format!("a.blk.{idx}.attn_output.weight"),
                format!("a.blk.{idx}.attn_o.weight"),
                format!("audio_encoder.layers.{idx}.attn.out_proj.weight"),
            ],
            &[
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
        let p = self
            .pos_proj
            .forward(pos_emb)?
            .reshape((1, 2 * seq_len - 1, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let pos_bias_u =
            self.pos_bias_u
                .reshape((1, self.n_heads, 1, self.head_dim))?;
        let pos_bias_v =
            self.pos_bias_v
                .reshape((1, self.n_heads, 1, self.head_dim))?;

        let q_u = q.broadcast_add(&pos_bias_u)?;
        let q_v = q.broadcast_add(&pos_bias_v)?;
        let matrix_ac = q_u.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let matrix_bd = rel_shift(&q_v.matmul(&p.transpose(2, 3)?.contiguous()?)?)?;
        let matrix_bd = matrix_bd.narrow(3, 0, seq_len)?;

        let scores = matrix_ac
            .broadcast_add(&matrix_bd)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let attn = ops::softmax(&scores, 3)?;
        let out = attn.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, self.n_heads * self.head_dim))?;

        self.out_proj.forward(&out).map_err(Error::from)
    }
}

struct AudioAdapter {
    norm: LayerNorm,
    linear1: Linear,
    linear2: Linear,
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
        x.broadcast_mul(&weight)?.broadcast_add(&bias).map_err(Error::from)
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
    let bias = load_optional_vector_any(loader, device, bias_names, dim)?
        .unwrap_or(Tensor::zeros(dim, DType::F32, device)?);
    Ok(LayerNorm::new(weight, bias, eps))
}

fn load_linear_any(
    loader: &GgufLoader,
    device: &Device,
    in_dim: usize,
    out_dim: usize,
    weight_names: &[String],
    bias_names: &[String],
) -> Result<Linear> {
    let mut weight = load_tensor_any(loader, device, weight_names, DType::F32)?;
    let dims = weight.dims2()?;
    if dims == (out_dim, in_dim) {
        // already in Candle layout.
    } else if dims == (in_dim, out_dim) {
        weight = weight.transpose(0, 1)?.contiguous()?;
    } else {
        return Err(Error::ModelLoadError(format!(
            "Linear weight shape mismatch for {}: got {dims:?}, expected ({out_dim}, {in_dim})",
            weight_names.join(" | ")
        )));
    }

    let bias = load_optional_vector_any(loader, device, bias_names, out_dim)?;
    Ok(Linear::new(weight, bias))
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
    let mut weight = load_tensor_any(loader, device, weight_names, DType::F32)?;
    let in_per_group = in_channels / cfg.groups.max(1);
    let expected_oihw = (out_channels, in_per_group, kernel_size, kernel_size);
    let expected_ohwi = (out_channels, kernel_size, kernel_size, in_per_group);

    let dims = weight.dims4()?;
    if dims == expected_ohwi {
        weight = weight.permute((0, 3, 1, 2))?.contiguous()?;
    } else if dims != expected_oihw {
        return Err(Error::ModelLoadError(format!(
            "Conv2d weight shape mismatch for {}: got {dims:?}, expected OIHW={expected_oihw:?} or OHWI={expected_ohwi:?}",
            weight_names.join(" | ")
        )));
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

    let mut weight = load_tensor_any(loader, device, weight_names, DType::F32)?;
    match weight.rank() {
        3 => {
            let dims = weight.dims3()?;
            if dims == expected_oki {
                weight = weight.permute((0, 2, 1))?.contiguous()?;
            } else if dims != expected_oik {
                return Err(Error::ModelLoadError(format!(
                    "Conv1d weight shape mismatch for {}: got {dims:?}, expected OIK={expected_oik:?} or OKI={expected_oki:?}",
                    weight_names.join(" | ")
                )));
            }
        }
        2 => {
            let dims = weight.dims2()?;
            if kernel_size != 1 || dims != (out_channels, in_per_group) {
                return Err(Error::ModelLoadError(format!(
                    "Conv1d 2D weight shape mismatch for {}: got {dims:?}, expected ({out_channels}, {in_per_group})",
                    weight_names.join(" | ")
                )));
            }
            weight = weight.unsqueeze(2)?;
        }
        rank => {
            return Err(Error::ModelLoadError(format!(
                "Conv1d weight rank mismatch for {}: expected 2 or 3, got {rank}",
                weight_names.join(" | ")
            )));
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
    let tensor = load_tensor_any(loader, device, names, DType::F32)?;
    let len = tensor.elem_count();
    if len != expected_len {
        return Err(Error::ModelLoadError(format!(
            "Vector shape mismatch for {}: expected {expected_len} values, found {len}",
            names.join(" | ")
        )));
    }
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
    for name in names {
        if loader.has_tensor(name) {
            return load_vector_any(loader, device, std::slice::from_ref(name), expected_len)
                .map(Some);
        }
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
    x.reshape((batch, heads, q_len, pos_len)).map_err(Error::from)
}

fn swish(x: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&ops::sigmoid(x)?).map_err(Error::from)
}

fn gelu(x: &Tensor) -> Result<Tensor> {
    let coeff = 0.044715f32;
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let dtype = x.dtype();
    let x3 = x.powf(3.0)?;
    let coeff_t = Tensor::new(&[coeff], x.device())?.to_dtype(dtype)?;
    let sqrt_t = Tensor::new(&[sqrt_2_over_pi], x.device())?.to_dtype(dtype)?;
    let one = Tensor::new(&[1f32], x.device())?.to_dtype(dtype)?;
    let half = Tensor::new(&[0.5f32], x.device())?.to_dtype(dtype)?;
    let inner = (x + x3.broadcast_mul(&coeff_t)?)?.broadcast_mul(&sqrt_t)?;
    let out = x.broadcast_mul(&one.broadcast_add(&inner.tanh()?)?)?;
    out.broadcast_mul(&half).map_err(Error::from)
}

pub fn subsampled_len_3x(mut len: usize) -> usize {
    for _ in 0..3 {
        len = (len + 1) / 2;
    }
    len
}

#[cfg(test)]
mod tests {
    use super::{build_rel_positional_embedding, rel_shift, subsampled_len_3x};
    use candle_core::{Device, Tensor};

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
}
