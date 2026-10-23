//! Granite Speech 5.0's conformer encoder: 16 blocks of macaron feed-forward,
//! block-local self-attention with Shaw relative positions over 4 s
//! (200-frame) blocks, a depthwise convolution module, and self-conditioning
//! from a character CTC head at the midpoint layer.

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{
    batch_norm, conv1d, conv1d_no_bias, embedding, layer_norm, linear, linear_no_bias, ops,
    Conv1dConfig, Embedding, LayerNorm, Module, VarBuilder,
};

use crate::error::{Error, Result};
use crate::kernels::depthwise_conv::{
    batch_norm_affine, depthwise_conv1d, fold_channel_affine, DepthwiseConv1dSpec,
};
use crate::kernels::f16_weight_linear::{
    projection_var_builder, PointwiseConv1d, ProjectionLinear, F16_PROJECTION_MAX_RELATIVE_L2,
};
use crate::kernels::numeric_check::{
    run_numeric_self_check, xorshift_tensor, NumericSelfCheckReport, F16_SAFE_MAX_ABS,
};

use super::config::GraniteSpeech5EncoderConfig;

pub(super) struct ConformerEncoder {
    input_linear: ProjectionLinear,
    layers: Vec<ConformerBlock>,
    /// Character CTC head at the midpoint, and its projection back into the
    /// residual stream (self-conditioning).
    out: ProjectionLinear,
    out_mid: ProjectionLinear,
}

/// A linear layer read from `vb`, held dense or, with `f16` set, as
/// weight-only F16 on that device (read and converted on the host).
fn encoder_projection(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: &VarBuilder,
    f16: Option<&Device>,
) -> Result<ProjectionLinear> {
    let vb = projection_var_builder(vb, f16);
    let linear = if bias {
        linear(in_dim, out_dim, vb)?
    } else {
        linear_no_bias(in_dim, out_dim, vb)?
    };
    Ok(ProjectionLinear::new(linear, f16)?)
}

impl ConformerEncoder {
    /// With `f16` set, every encoder linear and pointwise convolution loads
    /// as weight-only F16 on that device (F32 activations); norms, the
    /// depthwise convolution with its folded BatchNorm, biases and the
    /// relative-position table stay in the builder's dtype.
    pub(super) fn load(
        config: &GraniteSpeech5EncoderConfig,
        vb: VarBuilder,
        f16: Option<&Device>,
    ) -> Result<Self> {
        let input_linear = encoder_projection(
            config.input_dim,
            config.hidden_dim,
            true,
            &vb.pp("input_linear"),
            f16,
        )?;
        let layers = (0..config.num_layers)
            .map(|idx| ConformerBlock::load(config, vb.pp(format!("layers.{idx}")), f16))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            input_linear,
            layers,
            out: encoder_projection(
                config.hidden_dim,
                config.output_dim,
                true,
                &vb.pp("out"),
                f16,
            )?,
            out_mid: encoder_projection(
                config.output_dim,
                config.hidden_dim,
                true,
                &vb.pp("out_mid"),
                f16,
            )?,
        })
    }

    pub(super) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_observed(input, &mut |_, _| Ok(()))
    }

    /// [`Self::forward`], passing each conformer block's output to
    /// `observe` (the load-time self-check reads the residual stream).
    fn forward_observed(
        &self,
        input: &Tensor,
        observe: &mut dyn FnMut(usize, &Tensor) -> Result<()>,
    ) -> Result<Tensor> {
        let mut x = self.input_linear.forward(input)?;
        let midpoint = self.layers.len() / 2;
        for (idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            observe(idx, &x)?;
            if idx + 1 == midpoint {
                let mid = ops::softmax_last_dim(&self.out.forward(&x)?)?;
                x = x.broadcast_add(&self.out_mid.forward(&mid)?)?;
            }
        }
        Ok(x)
    }

    pub(super) fn context_size(&self) -> usize {
        self.layers
            .first()
            .map(|layer| layer.attn.context_size)
            .unwrap_or(0)
    }

    pub(super) fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Checks the first FFN projection's F16 conversion against its F32
    /// weight, then runs the encoder on xorshift features spanning two full
    /// attention blocks and a padded third, and requires every block output
    /// and the encoder output to be finite and inside the F16-safe range.
    pub(super) fn weight_only_f16_self_check(
        &self,
        config: &GraniteSpeech5EncoderConfig,
        vb: &VarBuilder,
    ) -> NumericSelfCheckReport {
        run_numeric_self_check(
            "Granite Speech 5.0 encoder weight-only F16",
            F16_SAFE_MAX_ABS,
            || {
                let first = self.layers.first().ok_or_else(|| {
                    Error::ModelLoadError("Granite Speech 5.0 encoder has no layers".into())
                })?;
                if let Some(converted) = first.ff1.up_proj.as_f16() {
                    let original = linear(
                        config.hidden_dim,
                        config.hidden_dim * config.feedforward_mult,
                        vb.clone()
                            .set_device(Device::Cpu)
                            .set_dtype(DType::F32)
                            .pp("layers.0.ff1.up_proj"),
                    )?;
                    let relative =
                        converted.check_against(original.weight(), original.bias(), 8)?;
                    if !relative.is_finite() || relative > F16_PROJECTION_MAX_RELATIVE_L2 {
                        return Err(Error::ModelLoadError(format!(
                            "layers.0.ff1.up_proj in F16 differs from its F32 weights by relative L2 {relative:.3e}"
                        )));
                    }
                }
                let frames = 2 * config.context_size + 3;
                let features =
                    xorshift_tensor(&[1, frames, config.input_dim], 0x6A5E, 2.0, vb.device())?;
                let mut outputs = Vec::new();
                let encoded = self.forward_observed(&features, &mut |index, x| {
                    outputs.push((format!("layers.{index}"), x.clone()));
                    Ok(())
                })?;
                outputs.push(("output".to_string(), encoded));
                Ok::<_, Error>(outputs)
            },
        )
    }
}

struct ConformerBlock {
    ff1: ConformerFeedForward,
    attn: ConformerAttention,
    conv: ConformerConv,
    ff2: ConformerFeedForward,
    post_norm: LayerNorm,
}

impl ConformerBlock {
    fn load(
        config: &GraniteSpeech5EncoderConfig,
        vb: VarBuilder,
        f16: Option<&Device>,
    ) -> Result<Self> {
        Ok(Self {
            ff1: ConformerFeedForward::load(config, vb.pp("ff1"), f16)?,
            attn: ConformerAttention::load(config, vb.pp("attn"), f16)?,
            conv: ConformerConv::load(config, vb.pp("conv"), f16)?,
            ff2: ConformerFeedForward::load(config, vb.pp("ff2"), f16)?,
            post_norm: layer_norm(config.hidden_dim, 1e-5, vb.pp("post_norm"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let ff1 = self.ff1.forward(x)?;
        let mut out = x.broadcast_add(&(ff1 * 0.5)?)?;
        out = out.broadcast_add(&self.attn.forward(&out)?)?;
        out = out.broadcast_add(&self.conv.forward(&out)?)?;
        let ff2 = self.ff2.forward(&out)?;
        out = out.broadcast_add(&(ff2 * 0.5)?)?;
        self.post_norm.forward(&out).map_err(Error::from)
    }
}

struct ConformerFeedForward {
    pre_norm: LayerNorm,
    up_proj: ProjectionLinear,
    down_proj: ProjectionLinear,
}

impl ConformerFeedForward {
    fn load(
        config: &GraniteSpeech5EncoderConfig,
        vb: VarBuilder,
        f16: Option<&Device>,
    ) -> Result<Self> {
        let hidden = config.hidden_dim;
        let intermediate = hidden * config.feedforward_mult;
        Ok(Self {
            pre_norm: layer_norm(hidden, 1e-5, vb.pp("pre_norm"))?,
            up_proj: encoder_projection(hidden, intermediate, true, &vb.pp("up_proj"), f16)?,
            down_proj: encoder_projection(intermediate, hidden, true, &vb.pp("down_proj"), f16)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.pre_norm.forward(x)?;
        let x = self.up_proj.forward(&x)?.silu()?;
        self.down_proj.forward(&x).map_err(Error::from)
    }
}

struct ConformerAttention {
    pre_norm: LayerNorm,
    to_q: ProjectionLinear,
    to_kv: ProjectionLinear,
    to_out: ProjectionLinear,
    rel_pos_emb: Embedding,
    /// `(context, context)` indices into `rel_pos_emb`: the query-key
    /// distance clamped to the block, offset by `max_pos_emb`.
    attention_dists: Tensor,
    context_size: usize,
    num_heads: usize,
    dim_head: usize,
}

impl ConformerAttention {
    fn load(
        config: &GraniteSpeech5EncoderConfig,
        vb: VarBuilder,
        f16: Option<&Device>,
    ) -> Result<Self> {
        let inner = config.dim_head * config.num_heads;
        let hidden = config.hidden_dim;
        Ok(Self {
            pre_norm: layer_norm(hidden, 1e-5, vb.pp("pre_norm"))?,
            to_q: encoder_projection(hidden, inner, false, &vb.pp("to_q"), f16)?,
            to_kv: encoder_projection(hidden, inner * 2, false, &vb.pp("to_kv"), f16)?,
            to_out: encoder_projection(inner, hidden, true, &vb.pp("to_out"), f16)?,
            rel_pos_emb: embedding(
                2 * config.max_pos_emb + 1,
                config.dim_head,
                vb.pp("rel_pos_emb"),
            )?,
            attention_dists: attention_dists(config.context_size, config.max_pos_emb, vb.device())?,
            context_size: config.context_size,
            num_heads: config.num_heads,
            dim_head: config.dim_head,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.pre_norm.forward(x)?;
        let (batch, seq_len, _) = x.dims3()?;
        let blocks = seq_len.div_ceil(self.context_size);
        let padded_len = blocks * self.context_size;
        let pad = padded_len - seq_len;
        let padded = if pad > 0 {
            x.pad_with_zeros(1, 0, pad)?
        } else {
            x
        };
        let inner = self.num_heads * self.dim_head;
        let q = self.to_q.forward(&padded)?;
        let kv = self.to_kv.forward(&padded)?;
        let heads = |x: &Tensor| {
            attention_heads(
                x,
                batch,
                blocks,
                self.context_size,
                self.num_heads,
                self.dim_head,
            )
        };
        let q = heads(&q)?;
        let k = heads(&kv.narrow(2, 0, inner)?)?;
        let v = heads(&kv.narrow(2, inner, inner)?)?;

        let rel = self
            .rel_pos_emb
            .forward(&self.attention_dists)?
            .to_dtype(q.dtype())?;
        let pos_bias = relative_position_bias(&q, &rel)?;
        let scale = 1.0 / (self.dim_head as f64).sqrt();
        let mut scores = (attention_scores(&q, &k)? * scale)?.broadcast_add(&pos_bias)?;
        if pad > 0 {
            let mask = padding_mask(
                self.context_size,
                seq_len % self.context_size,
                blocks,
                scores.device(),
                scores.dtype(),
            )?;
            scores = scores.broadcast_add(&mask)?;
        }
        let probs = ops::softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?.to_dtype(q.dtype())?;
        let out = attention_context(&probs, &v)?
            .transpose(2, 3)?
            .reshape((batch, padded_len, inner))?
            .narrow(1, 0, seq_len)?;
        self.to_out.forward(&out).map_err(Error::from)
    }
}

/// Shaw-style relative-position scores for every block-local query/key pair,
/// `bias[.., i, j] = q[.., i, :] · rel[i, j, :] / sqrt(dim)`, as one batched
/// matmul over the query position. `q` is `(batch, blocks, heads, context,
/// dim)`; `rel` is the gathered `(context, context, dim)` table.
fn relative_position_bias(q: &Tensor, rel: &Tensor) -> Result<Tensor> {
    let out_dtype = q.dtype();
    let (batch, blocks, heads, context, dim) = q.dims5()?;
    let rows = batch * blocks * heads;
    // (context, rows, dim) @ (context, dim, context) -> (context, rows, context).
    let q = q
        .reshape((rows, context, dim))?
        .to_dtype(DType::F32)?
        .transpose(0, 1)?
        .contiguous()?;
    let rel = rel.to_dtype(DType::F32)?.transpose(1, 2)?.contiguous()?;
    let scores = q.matmul(&rel)?.transpose(0, 1)?.contiguous()?;
    (scores.reshape((batch, blocks, heads, context, context))? / (dim as f64).sqrt())?
        .to_dtype(out_dtype)
        .map_err(Error::from)
}

fn attention_heads(
    x: &Tensor,
    batch: usize,
    blocks: usize,
    context: usize,
    num_heads: usize,
    dim_head: usize,
) -> Result<Tensor> {
    x.reshape((batch, blocks, context, num_heads, dim_head))?
        .transpose(2, 3)?
        .contiguous()
        .map_err(Error::from)
}

/// Candle's CPU matmul takes at most 4-D operands, so the 5-D block heads
/// fold into 3-D there.
fn attention_scores(q: &Tensor, k: &Tensor) -> Result<Tensor> {
    if q.device().is_cpu() {
        let (batch, blocks, heads, context, dim_head) = q.dims5()?;
        let flat = batch * blocks * heads;
        return q
            .reshape((flat, context, dim_head))?
            .matmul(&k.reshape((flat, context, dim_head))?.t()?)?
            .reshape((batch, blocks, heads, context, context))
            .map_err(Error::from);
    }
    q.matmul(&k.t()?).map_err(Error::from)
}

fn attention_context(probs: &Tensor, v: &Tensor) -> Result<Tensor> {
    if probs.device().is_cpu() {
        let (batch, blocks, heads, context, _) = probs.dims5()?;
        let dim_head = v.dim(D::Minus1)?;
        let flat = batch * blocks * heads;
        return probs
            .reshape((flat, context, context))?
            .matmul(&v.reshape((flat, context, dim_head))?)?
            .reshape((batch, blocks, heads, context, dim_head))
            .map_err(Error::from);
    }
    probs.matmul(v).map_err(Error::from)
}

fn attention_dists(context: usize, max_pos_emb: usize, device: &Device) -> Result<Tensor> {
    let context_span = context as isize;
    let values = (0..context)
        .flat_map(|i| {
            (0..context).map(move |j| {
                let dist = (i as isize - j as isize).clamp(-context_span, context_span);
                (dist + max_pos_emb as isize) as u32
            })
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (context, context), device).map_err(Error::from)
}

/// Masks the zero-padded tail of the final block: `-1e4` wherever a query
/// or key lies past the `remainder` real frames. Built as F32 on the host
/// (Candle's Metal backend cannot cast F64).
fn padding_mask(
    context: usize,
    remainder: usize,
    blocks: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let valid = remainder.max(1);
    let mut data = vec![0f32; blocks * context * context];
    if blocks > 0 {
        let final_block = (blocks - 1) * context * context;
        for q in 0..context {
            for k in 0..context {
                if q >= valid || k >= valid {
                    data[final_block + q * context + k] = -1e4;
                }
            }
        }
    }
    Tensor::from_vec(data, (1, blocks, 1, context, context), device)?
        .to_dtype(dtype)
        .map_err(Error::from)
}

struct ConformerConv {
    norm: LayerNorm,
    up_conv: PointwiseConv1d,
    /// Depthwise kernel `(inner, 1, K)` with the inference BatchNorm folded in.
    depth_weight: Tensor,
    depth_bias: Tensor,
    depth_spec: DepthwiseConv1dSpec,
    down_conv: PointwiseConv1d,
}

impl ConformerConv {
    fn load(
        config: &GraniteSpeech5EncoderConfig,
        vb: VarBuilder,
        f16: Option<&Device>,
    ) -> Result<Self> {
        let hidden = config.hidden_dim;
        let inner = hidden * config.conv_expansion_factor;
        let depth_conv = conv1d_no_bias(
            inner,
            inner,
            config.conv_kernel_size,
            Conv1dConfig {
                groups: inner,
                ..Default::default()
            },
            vb.pp("depth_conv.conv"),
        )?;
        let batch_norm = batch_norm(inner, 1e-5, vb.pp("batch_norm"))?;
        // Candle's grouped conv runs one convolution per channel; the shared
        // shift-and-accumulate op runs 2K + 2 element-wise ops.
        let (scale, shift) = batch_norm_affine(
            batch_norm.running_mean(),
            batch_norm.running_var(),
            batch_norm.eps(),
            batch_norm.weight_and_bias(),
        )?;
        let (depth_weight, depth_bias) =
            fold_channel_affine(depth_conv.weight(), None, &scale, &shift)?;
        // Same-length output: K / 2 on the left, one fewer on the right for
        // an even kernel.
        let width = config.conv_kernel_size;
        let pad = width / 2;
        let right = pad.saturating_sub((width + 1) % 2);
        let pointwise = |in_dim, out_dim, name: &str| -> Result<PointwiseConv1d> {
            Ok(PointwiseConv1d::new(
                conv1d(
                    in_dim,
                    out_dim,
                    1,
                    Conv1dConfig::default(),
                    projection_var_builder(&vb.pp(name), f16),
                )?,
                f16,
            )?)
        };
        Ok(Self {
            norm: layer_norm(hidden, 1e-5, vb.pp("norm"))?,
            up_conv: pointwise(hidden, inner * 2, "up_conv")?,
            depth_weight,
            depth_bias,
            depth_spec: DepthwiseConv1dSpec::padded(pad, right),
            down_conv: pointwise(inner, hidden, "down_conv")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward(x)?.transpose(1, 2)?.contiguous()?;
        let x = glu_channels(&self.up_conv.forward(&x)?)?;
        let x = depthwise_conv1d(
            &x,
            &self.depth_weight,
            Some(&self.depth_bias),
            self.depth_spec,
        )?
        .silu()?;
        self.down_conv
            .forward(&x)?
            .transpose(1, 2)?
            .contiguous()
            .map_err(Error::from)
    }
}

/// GLU over channels: the first half gated by the sigmoid of the second.
fn glu_channels(x: &Tensor) -> Result<Tensor> {
    let half = x.dim(1)? / 2;
    x.narrow(1, 0, half)?
        .broadcast_mul(&ops::sigmoid(&x.narrow(1, half, half)?)?)
        .map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{Conv1d, VarMap};

    fn assert_close(left: &Tensor, right: &Tensor, tolerance: f32) {
        let left = left.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let right = right.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(left.len(), right.len());
        for (a, b) in left.iter().zip(&right) {
            let diff = (a - b).abs();
            // NaN-safe: a non-finite difference fails.
            assert!(diff.is_finite() && diff <= tolerance, "{a} vs {b}");
        }
    }

    #[test]
    fn batched_relative_position_bias_matches_per_row_matmuls() {
        let device = Device::Cpu;
        let (batch, blocks, heads, context, dim) = (1, 2, 2, 3, 4);
        let q = xorshift_tensor(&[batch, blocks, heads, context, dim], 11, 1.0, &device).unwrap();
        let rel = xorshift_tensor(&[context, context, dim], 12, 1.0, &device).unwrap();
        let bias = relative_position_bias(&q, &rel).unwrap();
        let q_values = q.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let rel_values = rel.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut expected = Vec::new();
        for row in 0..batch * blocks * heads {
            for i in 0..context {
                for j in 0..context {
                    let dot = (0..dim)
                        .map(|d| {
                            q_values[(row * context + i) * dim + d]
                                * rel_values[(i * context + j) * dim + d]
                        })
                        .sum::<f32>();
                    expected.push(dot / (dim as f32).sqrt());
                }
            }
        }
        let expected = Tensor::from_vec(expected, bias.shape(), &device).unwrap();
        assert_close(&bias, &expected, 1e-5);
    }

    #[test]
    fn attention_dists_clamp_to_the_block_and_offset_by_max_pos() {
        let dists = attention_dists(3, 5, &Device::Cpu).unwrap();
        assert_eq!(
            dists.to_vec2::<u32>().unwrap(),
            vec![vec![5, 4, 3], vec![6, 5, 4], vec![7, 6, 5]]
        );
    }

    #[test]
    fn padding_mask_covers_only_the_final_blocks_padding() {
        let mask = padding_mask(2, 1, 2, &Device::Cpu, DType::F32).unwrap();
        let values = mask.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(&values[..4], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&values[4..], &[0.0, -1e4, -1e4, -1e4]);
    }

    #[test]
    fn conformer_conv_shared_depthwise_matches_candle_grouped_conv_and_batch_norm() {
        use candle_nn::ModuleT;

        let device = Device::Cpu;
        let mut state = 0x6A09_E667_F3BC_C908u64;
        let mut values = |n: usize, scale: f32, offset: f32| {
            (0..n)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    offset + scale * ((state >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
                })
                .collect::<Vec<_>>()
        };
        for kernel in [15usize, 4] {
            let mut config = tiny_encoder_config();
            config.hidden_dim = 16;
            config.conv_expansion_factor = 2;
            config.conv_kernel_size = kernel;
            let (hidden, inner) = (16usize, 32usize);
            let mut tensors = std::collections::HashMap::new();
            let mut put = |name: &str, shape: &[usize], scale: f32, offset: f32| {
                let n = shape.iter().product();
                let t = Tensor::from_vec(values(n, scale, offset), shape, &device).unwrap();
                tensors.insert(name.to_string(), t.clone());
                t
            };
            let norm_w = put("norm.weight", &[hidden], 0.1, 1.0);
            let norm_b = put("norm.bias", &[hidden], 0.1, 0.0);
            let up_w = put("up_conv.weight", &[2 * inner, hidden, 1], 0.25, 0.0);
            let up_b = put("up_conv.bias", &[2 * inner], 0.1, 0.0);
            let dw = put("depth_conv.conv.weight", &[inner, 1, kernel], 0.3, 0.0);
            let gamma = put("batch_norm.weight", &[inner], 0.2, 1.0);
            let beta = put("batch_norm.bias", &[inner], 0.1, 0.0);
            let mean = put("batch_norm.running_mean", &[inner], 0.2, 0.0);
            let var = put("batch_norm.running_var", &[inner], 0.25, 1.0);
            let down_w = put("down_conv.weight", &[hidden, inner, 1], 0.2, 0.0);
            let down_b = put("down_conv.bias", &[hidden], 0.1, 0.0);
            let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
            let conv = ConformerConv::load(&config, vb, None).unwrap();

            let x = Tensor::from_vec(values(2 * 29 * hidden, 1.0, 0.0), (2, 29, hidden), &device)
                .unwrap();
            let reference = {
                let x = candle_nn::LayerNorm::new(norm_w, norm_b, 1e-5)
                    .forward(&x)
                    .unwrap()
                    .transpose(1, 2)
                    .unwrap()
                    .contiguous()
                    .unwrap();
                let x = Conv1d::new(up_w, Some(up_b), Conv1dConfig::default())
                    .forward(&x)
                    .unwrap();
                let x = glu_channels(&x).unwrap();
                let pad = kernel / 2;
                let right = pad.saturating_sub((kernel + 1) % 2);
                let x = x.pad_with_zeros(2, pad, right).unwrap();
                let depth_cfg = Conv1dConfig {
                    groups: inner,
                    ..Default::default()
                };
                let x = Conv1d::new(dw, None, depth_cfg).forward(&x).unwrap();
                let bn = candle_nn::BatchNorm::new(inner, mean, var, gamma, beta, 1e-5).unwrap();
                let x = bn.forward_t(&x, false).unwrap().silu().unwrap();
                Conv1d::new(down_w, Some(down_b), Conv1dConfig::default())
                    .forward(&x)
                    .unwrap()
                    .transpose(1, 2)
                    .unwrap()
                    .contiguous()
                    .unwrap()
            };
            let calls = crate::kernels::depthwise_conv::depthwise_conv1d_calls();
            assert_close(&conv.forward(&x).unwrap(), &reference, 1e-5);
            assert!(crate::kernels::depthwise_conv::depthwise_conv1d_calls() > calls);
        }
    }

    #[test]
    fn encoder_runs_padded_blocks_and_self_conditions_at_the_midpoint() {
        let device = Device::Cpu;
        let config = tiny_encoder_config();
        let vars = VarMap::new();
        let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);
        let encoder = ConformerEncoder::load(&config, vb.clone(), None).unwrap();
        assert_eq!(encoder.layer_count(), 2);
        assert_eq!(encoder.context_size(), 2);
        // Three frames: one full block and a padded second block.
        let input = xorshift_tensor(&[1, 3, config.input_dim], 5, 1.0, &device).unwrap();
        let output = encoder.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 3, config.hidden_dim]);
        assert!(output
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|value| value.is_finite()));
        let report = encoder.weight_only_f16_self_check(&config, &vb);
        assert!(report.passed, "{report:?}");
    }

    /// The encoder with weight-only F16 projections on `device` against the
    /// same weights in F32 on the CPU.
    fn assert_f16_encoder_tracks_cpu_f32(device: &Device) {
        let config = GraniteSpeech5EncoderConfig {
            hidden_dim: 16,
            input_dim: 12,
            output_dim: 7,
            dim_head: 4,
            num_heads: 4,
            context_size: 4,
            max_pos_emb: 8,
            conv_kernel_size: 5,
            ..tiny_encoder_config()
        };
        let vars = VarMap::new();
        let cpu = VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu);
        ConformerEncoder::load(&config, cpu.clone(), None).unwrap();
        let mut seed = 30;
        let tensors = vars
            .data()
            .lock()
            .unwrap()
            .iter()
            .map(|(name, tensor)| {
                seed += 1;
                let scale = if name.ends_with("running_var") || name.ends_with("norm.weight") {
                    0.1
                } else {
                    0.3
                };
                let mut value = xorshift_tensor(tensor.dims(), seed, scale, &Device::Cpu).unwrap();
                if name.ends_with("running_var") || name.ends_with("norm.weight") {
                    value = (value.abs().unwrap() + 1.0).unwrap();
                }
                (name.clone(), value)
            })
            .collect::<std::collections::HashMap<_, _>>();
        let reference = ConformerEncoder::load(
            &config,
            VarBuilder::from_tensors(tensors.clone(), DType::F32, &Device::Cpu),
            None,
        )
        .unwrap();
        let on_device = ConformerEncoder::load(
            &config,
            VarBuilder::from_tensors(tensors, DType::F32, device),
            Some(device),
        )
        .unwrap();
        let input = xorshift_tensor(&[1, 9, config.input_dim], 77, 1.0, &Device::Cpu).unwrap();
        let expected = reference.forward(&input).unwrap();
        let actual = on_device
            .forward(&input.to_device(device).unwrap())
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap();
        assert_close(&actual, &expected, 2e-2);
    }

    #[test]
    fn weight_only_f16_encoder_tracks_f32_on_cpu() {
        assert_f16_encoder_tracks_cpu_f32(&Device::Cpu);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_weight_only_f16_encoder_tracks_cpu_f32() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "metal_weight_only_f16_encoder_tracks_cpu_f32",
        ) else {
            return;
        };
        assert_f16_encoder_tracks_cpu_f32(&device);
    }

    fn tiny_encoder_config() -> GraniteSpeech5EncoderConfig {
        GraniteSpeech5EncoderConfig {
            cat_hidden_layers: vec![],
            context_size: 2,
            conv_expansion_factor: 2,
            conv_kernel_size: 3,
            dim_head: 2,
            dropout: 0.0,
            feedforward_mult: 2,
            hidden_dim: 4,
            input_dim: 6,
            max_pos_emb: 4,
            model_type: None,
            num_heads: 2,
            num_layers: 2,
            output_dim: 5,
        }
    }
}
