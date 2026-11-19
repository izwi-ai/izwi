//! Ming's flow path: the two small bidirectional transformers around the
//! language model, and the sampler that turns a hidden state into an audio
//! latent (`fm/dit.py`, `fm/modules.py`, `fm/CFM.py`).
//!
//! * [`MingAggregator`] (`linear_proj_audio`): a latent → the embedding the
//!   language model reads next. Two positions: a learned first token, then
//!   the latent; the first position's output is the embedding.
//! * [`MingFlowHead`] (`flowloss.cfm.model`): the velocity field. Three
//!   positions: time + condition, the previous latent, the noisy latent;
//!   the last position's output is the velocity.
//! * [`MingFlowHead::sample`]: ten Euler steps from noise on the
//!   reference's time grid, both guidance branches in one batch of two.
//!
//! Both transformers are pre-norm RMSNorm blocks with x_transformers'
//! rotary embedding (base 10,000, pairs interleaved), which is not the
//! half-split form the Qwen2 stacks use.

use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;

use super::config::{MingFlowConfig, FLOW_GRID_32, FLOW_STEPS, LATENT_DIM};
use super::stack::rms_norm;
use crate::error::{Error, Result};

const NORM_EPS: f64 = 1e-6;
/// Width of the sinusoidal time embedding.
const TIME_EMBED_DIM: usize = 256;
/// The reference scales time by 1,000 before the sinusoids.
const TIME_SCALE: f64 = 1_000.0;
/// x_transformers' `RotaryEmbedding` base.
const ROTARY_BASE: f64 = 10_000.0;

/// A linear layer with its weight as stored, [out, in]: transposed as a
/// view at each call, so no transposed copy is written on the device.
struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    fn load(vb: VarBuilder, input: usize, output: usize, bias: bool) -> Result<Self> {
        Ok(Self {
            weight: vb.get((output, input), "weight")?,
            bias: if bias {
                Some(vb.get(output, "bias")?)
            } else {
                None
            },
        })
    }

    /// `x` [..., in] → [..., out].
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let input = *dims.last().expect("a linear input has a last dim");
        let rows = dims[..dims.len() - 1].iter().product::<usize>();
        let mut y = x.reshape((rows, input))?.matmul(&self.weight.t()?)?;
        if let Some(bias) = &self.bias {
            y = y.broadcast_add(bias)?;
        }
        let mut shape = dims;
        *shape.last_mut().expect("a last dim") = y.dim(1)?;
        Ok(y.reshape(shape)?)
    }
}

struct FlowBlock {
    norm1: Tensor,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm2: Tensor,
    mlp_in: Linear,
    mlp_out: Linear,
}

impl FlowBlock {
    fn load(cfg: &MingFlowConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let attn = vb.pp("attn");
        let ff = vb.pp("mlp").pp("ff");
        Ok(Self {
            norm1: vb.pp("norm1").get(hidden, "weight")?,
            to_q: Linear::load(attn.pp("to_q"), hidden, hidden, true)?,
            to_k: Linear::load(attn.pp("to_k"), hidden, hidden, true)?,
            to_v: Linear::load(attn.pp("to_v"), hidden, hidden, true)?,
            to_out: Linear::load(attn.pp("to_out").pp("0"), hidden, hidden, true)?,
            norm2: vb.pp("norm2").get(hidden, "weight")?,
            mlp_in: Linear::load(ff.pp("0").pp("0"), hidden, cfg.mlp_size, true)?,
            mlp_out: Linear::load(ff.pp("2"), cfg.mlp_size, hidden, true)?,
        })
    }

    /// `x` [batch, positions, hidden]; `cos`/`sin` [positions, head_dim / 2].
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cfg: &MingFlowConfig,
    ) -> Result<Tensor> {
        let (batch, positions, hidden) = x.dims3()?;
        let dtype = x.dtype();
        let normed = rms_norm(x, &self.norm1, NORM_EPS)?;
        let heads = |t: Tensor| -> Result<Tensor> {
            Ok(t.reshape((batch, positions, cfg.num_heads, cfg.head_dim))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let q = rotate_pairs(&heads(self.to_q.forward(&normed)?)?, cos, sin)?;
        let k = rotate_pairs(&heads(self.to_k.forward(&normed)?)?, cos, sin)?;
        let v = heads(self.to_v.forward(&normed)?)?;
        let scale = 1.0 / (cfg.head_dim as f64).sqrt();
        let scores =
            (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?.to_dtype(DType::F32)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(dtype)?;
        let attn = weights
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, positions, hidden))?;
        let x = x.add(&self.to_out.forward(&attn)?)?;
        let normed = rms_norm(&x, &self.norm2, NORM_EPS)?;
        let mlp = self
            .mlp_out
            .forward(&self.mlp_in.forward(&normed)?.gelu()?)?;
        Ok(x.add(&mlp)?)
    }
}

/// x_transformers' rotation of `x` [batch, heads, positions, dim]: pairs
/// (2i, 2i + 1) turned by `cos`/`sin` [positions, dim / 2].
fn rotate_pairs(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (batch, heads, positions, dim) = x.dims4()?;
    let pairs = x.reshape((batch, heads, positions, dim / 2, 2))?;
    let even = pairs.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
    let odd = pairs.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;
    let first = even.broadcast_mul(cos)?.sub(&odd.broadcast_mul(sin)?)?;
    let second = odd.broadcast_mul(cos)?.add(&even.broadcast_mul(sin)?)?;
    Ok(Tensor::stack(&[first, second], D::Minus1)?
        .reshape((batch, heads, positions, dim))?
        .contiguous()?)
}

/// cos/sin of `positions` positions, [positions, head_dim / 2], built on
/// the host in F64.
fn rotary_table(
    positions: usize,
    head_dim: usize,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let pairs = head_dim / 2;
    let mut cos = Vec::with_capacity(positions * pairs);
    let mut sin = Vec::with_capacity(positions * pairs);
    for position in 0..positions {
        for pair in 0..pairs {
            let inv_freq = ROTARY_BASE.powf(-((2 * pair) as f64) / head_dim as f64);
            let angle = position as f64 * inv_freq;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    let table = |values: Vec<f32>| -> Result<Tensor> {
        Ok(Tensor::from_vec(values, (positions, pairs), &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(device)?)
    };
    Ok((table(cos)?, table(sin)?))
}

struct FlowTransformer {
    cfg: MingFlowConfig,
    blocks: Vec<FlowBlock>,
    final_norm: Tensor,
    final_linear: Linear,
    cos: Tensor,
    sin: Tensor,
}

impl FlowTransformer {
    fn load(cfg: MingFlowConfig, vb: &VarBuilder, positions: usize, output: usize) -> Result<Self> {
        let blocks = (0..cfg.depth)
            .map(|index| FlowBlock::load(&cfg, vb.pp(format!("blocks.{index}"))))
            .collect::<Result<Vec<_>>>()?;
        let last = vb.pp("final_layer");
        let (cos, sin) = rotary_table(positions, cfg.head_dim, vb.dtype(), vb.device())?;
        Ok(Self {
            final_norm: last.pp("norm_final").get(cfg.hidden_size, "weight")?,
            final_linear: Linear::load(last.pp("linear"), cfg.hidden_size, output, true)?,
            blocks,
            cos,
            sin,
            cfg,
        })
    }

    /// `x` [batch, positions, hidden] → [batch, positions, output].
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x, &self.cos, &self.sin, &self.cfg)?;
        }
        self.final_linear
            .forward(&rms_norm(&x, &self.final_norm, NORM_EPS)?)
    }
}

/// `linear_proj_audio`: an audio latent to a language-model embedding.
pub struct MingAggregator {
    body: FlowTransformer,
    /// The learned first token, [1, 1, hidden].
    first: Tensor,
    x_embedder: Linear,
    dtype: DType,
}

impl MingAggregator {
    pub fn load(cfg: MingFlowConfig, vb: VarBuilder, llm_hidden: usize) -> Result<Self> {
        let hidden = cfg.hidden_size;
        Ok(Self {
            first: vb
                .pp("word_embedder")
                .get((1, hidden), "weight")?
                .reshape((1, 1, hidden))?,
            x_embedder: Linear::load(vb.pp("x_embedder"), LATENT_DIM, hidden, true)?,
            dtype: vb.dtype(),
            body: FlowTransformer::load(cfg, &vb, 2, llm_hidden)?,
        })
    }

    /// `latents` [rows, 64] (any float dtype) → [rows, llm_hidden] in the
    /// model dtype. Each row is its own two-position sequence.
    pub fn forward(&self, latents: &Tensor) -> Result<Tensor> {
        let (rows, width) = latents.dims2()?;
        if rows == 0 || width != LATENT_DIM {
            return Err(Error::InvalidInput(
                "Ming aggregator input must be [rows, 64]".into(),
            ));
        }
        let x = self
            .x_embedder
            .forward(&latents.to_dtype(self.dtype)?.unsqueeze(1)?)?;
        let hidden = self.first.dim(2)?;
        let x = Tensor::cat(&[&self.first.broadcast_as((rows, 1, hidden))?, &x], 1)?;
        Ok(self.body.forward(&x)?.narrow(1, 0, 1)?.squeeze(1)?)
    }
}

/// `flowloss.cfm.model`: the velocity field and its sampler.
pub struct MingFlowHead {
    body: FlowTransformer,
    time_in: Linear,
    time_out: Linear,
    cond: Linear,
    x_embedder: Linear,
    /// The time embedding's sinusoids for each sampler step, [steps, 256].
    time_features: Tensor,
    /// Each step's `dt` on the sway-warped grid.
    dts: [f64; FLOW_STEPS],
    dtype: DType,
    device: Device,
}

/// The sampler's times: the pruned grid, then sway sampling with
/// coefficient -1 (`t + (-1)(cos(pi t / 2) - 1 + t)`).
pub(crate) fn flow_times() -> [f64; FLOW_STEPS + 1] {
    FLOW_GRID_32.map(|step| {
        let t = f64::from(step) / 32.0;
        t - ((std::f64::consts::FRAC_PI_2 * t).cos() - 1.0 + t)
    })
}

impl MingFlowHead {
    pub fn load(cfg: MingFlowConfig, vb: VarBuilder, llm_hidden: usize) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let time = vb.pp("t_embedder").pp("time_mlp");
        let times = flow_times();
        let half = TIME_EMBED_DIM / 2;
        let rate = ROTARY_BASE.ln() / (half - 1) as f64;
        let mut features = Vec::with_capacity(FLOW_STEPS * TIME_EMBED_DIM);
        for &t in &times[..FLOW_STEPS] {
            let angle = |index: usize| TIME_SCALE * t * (-(index as f64) * rate).exp();
            features.extend((0..half).map(|index| angle(index).sin() as f32));
            features.extend((0..half).map(|index| angle(index).cos() as f32));
        }
        let mut dts = [0.0; FLOW_STEPS];
        for (step, dt) in dts.iter_mut().enumerate() {
            *dt = times[step + 1] - times[step];
        }
        Ok(Self {
            time_in: Linear::load(time.pp("0"), TIME_EMBED_DIM, hidden, true)?,
            time_out: Linear::load(time.pp("2"), hidden, hidden, true)?,
            cond: Linear::load(
                vb.pp("c_embedder").pp("cond_embedder"),
                llm_hidden,
                hidden,
                true,
            )?,
            x_embedder: Linear::load(vb.pp("x_embedder"), LATENT_DIM, hidden, true)?,
            time_features: Tensor::from_vec(features, (FLOW_STEPS, TIME_EMBED_DIM), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(&device)?,
            dts,
            dtype,
            device,
            body: FlowTransformer::load(cfg, &vb, 3, LATENT_DIM)?,
        })
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The velocity at sampler step `step`: `x`, `history` [batch, 64] and
    /// `cond` [batch, llm_hidden], all in the model dtype → [batch, 64].
    fn velocity(&self, step: usize, x: &Tensor, history: &Tensor, cond: &Tensor) -> Result<Tensor> {
        let batch = x.dim(0)?;
        let time = self.time_out.forward(&candle_nn::ops::silu(
            &self
                .time_in
                .forward(&self.time_features.narrow(0, step, 1)?)?,
        )?)?;
        let first = self.cond.forward(cond)?.broadcast_add(&time)?;
        let sequence = Tensor::stack(
            &[
                first,
                self.x_embedder.forward(history)?,
                self.x_embedder.forward(x)?,
            ],
            1,
        )?;
        debug_assert_eq!(sequence.dim(0)?, batch);
        Ok(self.body.forward(&sequence)?.narrow(1, 2, 1)?.squeeze(1)?)
    }

    /// One latent for one request: `hidden` [896] is the language model's
    /// state, `history` [64] the previous latent, `noise` [64] the start.
    /// Returns the latent, [64] F32 on the host.
    pub fn sample(
        &self,
        hidden: &[f32],
        history: &[f32],
        noise: &[f32],
        guidance: f32,
    ) -> Result<Vec<f32>> {
        if history.len() != LATENT_DIM || noise.len() != LATENT_DIM {
            return Err(Error::InvalidInput(
                "Ming flow sampling needs 64-wide history and noise".into(),
            ));
        }
        let host = |values: &[f32], width: usize| -> Result<Tensor> {
            Ok(Tensor::from_vec(values.to_vec(), (1, width), &self.device)?)
        };
        // Conditional row, then the unconditional one (a zero condition).
        let hidden = host(hidden, hidden.len())?;
        let cond = Tensor::cat(&[&hidden, &hidden.zeros_like()?], 0)?.to_dtype(self.dtype)?;
        let history = host(history, LATENT_DIM)?
            .broadcast_as((2, LATENT_DIM))?
            .to_dtype(self.dtype)?
            .contiguous()?;
        let mut y = host(noise, LATENT_DIM)?;
        let guidance = f64::from(guidance);
        for (step, &dt) in self.dts.iter().enumerate() {
            let x = y
                .broadcast_as((2, LATENT_DIM))?
                .to_dtype(self.dtype)?
                .contiguous()?;
            let both = self
                .velocity(step, &x, &history, &cond)?
                .to_dtype(DType::F32)?;
            let pred = both.narrow(0, 0, 1)?;
            let null = both.narrow(0, 1, 1)?;
            // pred + (pred - null) * guidance.
            let velocity = (&pred + ((&pred - &null)? * guidance)?)?;
            y = (y + (velocity * dt)?)?;
        }
        Ok(y.flatten_all()?.to_vec1::<f32>()?)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::models::architectures::ming_tts::stack::tests::{max_diff, random};
    use crate::models::shared::noise::SplitMix64;

    pub(crate) fn tiny_flow_config() -> MingFlowConfig {
        MingFlowConfig {
            hidden_size: 32,
            depth: 2,
            num_heads: 4,
            head_dim: 8,
            mlp_size: 64,
        }
    }

    fn put_linear(
        tensors: &mut HashMap<String, Tensor>,
        rng: &mut SplitMix64,
        name: &str,
        input: usize,
        output: usize,
    ) {
        tensors.insert(format!("{name}.weight"), random(rng, &[output, input], 0.2));
        tensors.insert(format!("{name}.bias"), random(rng, &[output], 0.1));
    }

    /// Random weights of a flow transformer under `prefix`.
    pub(crate) fn tiny_flow_weights(
        cfg: &MingFlowConfig,
        prefix: &str,
        output: usize,
        rng: &mut SplitMix64,
        tensors: &mut HashMap<String, Tensor>,
    ) {
        let hidden = cfg.hidden_size;
        let norm = |rng: &mut SplitMix64| (random(rng, &[hidden], 0.05) + 1.0).unwrap();
        for block in 0..cfg.depth {
            let name = |suffix: &str| format!("{prefix}blocks.{block}.{suffix}");
            tensors.insert(name("norm1.weight"), norm(rng));
            tensors.insert(name("norm2.weight"), norm(rng));
            for part in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"] {
                put_linear(tensors, rng, &name(part), hidden, hidden);
            }
            put_linear(tensors, rng, &name("mlp.ff.0.0"), hidden, cfg.mlp_size);
            put_linear(tensors, rng, &name("mlp.ff.2"), cfg.mlp_size, hidden);
        }
        tensors.insert(format!("{prefix}final_layer.norm_final.weight"), norm(rng));
        put_linear(
            tensors,
            rng,
            &format!("{prefix}final_layer.linear"),
            hidden,
            output,
        );
        put_linear(
            tensors,
            rng,
            &format!("{prefix}x_embedder"),
            LATENT_DIM,
            hidden,
        );
    }

    pub(crate) fn tiny_aggregator_weights(
        cfg: &MingFlowConfig,
        prefix: &str,
        llm_hidden: usize,
        rng: &mut SplitMix64,
        tensors: &mut HashMap<String, Tensor>,
    ) {
        tiny_flow_weights(cfg, prefix, llm_hidden, rng, tensors);
        tensors.insert(
            format!("{prefix}word_embedder.weight"),
            random(rng, &[1, cfg.hidden_size], 0.5),
        );
    }

    pub(crate) fn tiny_head_weights(
        cfg: &MingFlowConfig,
        prefix: &str,
        llm_hidden: usize,
        rng: &mut SplitMix64,
        tensors: &mut HashMap<String, Tensor>,
    ) {
        tiny_flow_weights(cfg, prefix, LATENT_DIM, rng, tensors);
        let hidden = cfg.hidden_size;
        put_linear(
            tensors,
            rng,
            &format!("{prefix}t_embedder.time_mlp.0"),
            TIME_EMBED_DIM,
            hidden,
        );
        put_linear(
            tensors,
            rng,
            &format!("{prefix}t_embedder.time_mlp.2"),
            hidden,
            hidden,
        );
        put_linear(
            tensors,
            rng,
            &format!("{prefix}c_embedder.cond_embedder"),
            llm_hidden,
            hidden,
        );
    }

    #[test]
    fn sway_grid_runs_from_noise_to_data() {
        let times = flow_times();
        assert_eq!(times[0], 0.0);
        assert!((times[FLOW_STEPS] - 1.0).abs() < 1e-12);
        assert!(times.windows(2).all(|pair| pair[1] > pair[0]));
        // 1 - cos(pi t / 2) at t = 1/16.
        let expected = 1.0 - (std::f64::consts::FRAC_PI_2 / 16.0).cos();
        assert!((times[1] - expected).abs() < 1e-12);
    }

    #[test]
    fn pair_rotation_matches_complex_multiplication() {
        let x = random(&mut SplitMix64::new(5), &[1, 2, 3, 8], 1.0);
        let (cos, sin) = rotary_table(3, 8, DType::F32, &Device::Cpu).unwrap();
        let rotated = rotate_pairs(&x, &cos, &sin).unwrap();
        let x = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let rotated = rotated.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for head in 0..2 {
            for position in 0..3 {
                for pair in 0..4 {
                    let at = (head * 3 + position) * 8 + 2 * pair;
                    let angle = position as f64 * ROTARY_BASE.powf(-((2 * pair) as f64) / 8.0);
                    let (re, im) = (f64::from(x[at]), f64::from(x[at + 1]));
                    let want_re = re * angle.cos() - im * angle.sin();
                    let want_im = im * angle.cos() + re * angle.sin();
                    assert!((f64::from(rotated[at]) - want_re).abs() < 1e-5);
                    assert!((f64::from(rotated[at + 1]) - want_im).abs() < 1e-5);
                }
            }
        }
    }

    #[test]
    fn aggregator_rows_are_independent_and_position_zero_is_read() {
        let cfg = tiny_flow_config();
        let mut rng = SplitMix64::new(7);
        let mut tensors = HashMap::new();
        tiny_aggregator_weights(&cfg, "", 24, &mut rng, &mut tensors);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let aggregator = MingAggregator::load(cfg, vb, 24).unwrap();
        let latents = random(&mut rng, &[3, LATENT_DIM], 1.0);
        let together = aggregator.forward(&latents).unwrap();
        assert_eq!(together.dims(), &[3, 24]);
        for row in 0..3 {
            let alone = aggregator
                .forward(&latents.narrow(0, row, 1).unwrap())
                .unwrap();
            let a = alone.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let b = together
                .narrow(0, row, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert!(max_diff(&a, &b) < 1e-5);
        }
    }

    #[test]
    fn sampling_is_deterministic_and_guidance_changes_it() {
        let cfg = tiny_flow_config();
        let mut rng = SplitMix64::new(9);
        let mut tensors = HashMap::new();
        tiny_head_weights(&cfg, "", 24, &mut rng, &mut tensors);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let head = MingFlowHead::load(cfg, vb, 24).unwrap();
        let hidden = rng.standard_normals(24);
        let history = rng.standard_normals(LATENT_DIM);
        let noise = rng.standard_normals(LATENT_DIM);
        let a = head.sample(&hidden, &history, &noise, 2.0).unwrap();
        let b = head.sample(&hidden, &history, &noise, 2.0).unwrap();
        assert_eq!(a, b);
        assert!(a.iter().all(|value| value.is_finite()));
        let c = head.sample(&hidden, &history, &noise, 1.0).unwrap();
        assert!(max_diff(&a, &c) > 1e-4);
        // With a zero condition both branches agree, so guidance cancels.
        let zero = vec![0.0; 24];
        let d = head.sample(&zero, &history, &noise, 2.0).unwrap();
        let e = head.sample(&zero, &history, &noise, 5.0).unwrap();
        assert!(max_diff(&d, &e) < 1e-4);
    }
}
