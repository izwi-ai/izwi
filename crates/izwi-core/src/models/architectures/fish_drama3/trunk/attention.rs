//! Gated full attention (10 of the 40 trunk layers) and the family's
//! partial RoPE.
//!
//! ```text
//! [q_h | gate_h] = q_proj(x) per head      # interleaved per head, not [Q | Gate]
//! q, k = (1 + w)-RMSNorm per head          # zero-centred q/k norms
//! q, k = RoPE(q, k) on dims 0..rotary      # NeoX half-split, F64 angles, θ 1e7
//! o    = causal GQA softmax(q kᵀ / sqrt(head_dim)) v
//! out  = o_proj(o * sigmoid(gate))
//! ```
//!
//! Keys enter the paged cache after RoPE (`KeyEncoding::Raw`), through one
//! prepared append shared by every attention layer of the forward.

use candle_core::{DType, Device, Tensor};

use super::super::checkpoint::{trunk_layer, FishDrama3Checkpoint};
use super::super::config::{
    FishDrama3AttentionConfig, FishDrama3RopeLayout, FishDrama3TrunkConfig,
};
use super::super::linear::{FishDrama3KernelPhase, FishDrama3Linear};
use super::super::residency::{FishDrama3ClassForms, FishDrama3ProjectionClass};
use super::{load_linear, norm_gain, rms_norm_rows, sigmoid, NormGain};
use crate::error::{Error, Result};
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};

/// Partial rotary position embedding over the first `rotary_dims` of each
/// head. Angles are `position * θ^(-2i / rotary_dims)` computed in F64 and
/// rounded to F32 once, so long positions keep their phase (an F32 product
/// at position 262,143 is off by up to 0.016 rad).
#[derive(Debug, Clone)]
pub struct FishDrama3Rope {
    inv_freq: Vec<f64>,
    rotary_dims: usize,
    head_dim: usize,
    layout: FishDrama3RopeLayout,
}

impl FishDrama3Rope {
    pub fn new(attention: &FishDrama3AttentionConfig) -> Self {
        let rotary = attention.rotary_dims;
        Self {
            inv_freq: (0..rotary / 2)
                .map(|i| attention.rope_theta.powf(-((2 * i) as f64) / rotary as f64))
                .collect(),
            rotary_dims: rotary,
            head_dim: attention.head_dim,
            layout: attention.rope_layout,
        }
    }

    pub fn layout(&self) -> FishDrama3RopeLayout {
        self.layout
    }

    /// `(cos, sin)` of each rotated pair at `position`.
    pub fn cos_sin(&self, position: usize) -> (Vec<f32>, Vec<f32>) {
        self.inv_freq
            .iter()
            .map(|freq| {
                let angle = position as f64 * freq;
                (angle.cos() as f32, angle.sin() as f32)
            })
            .unzip()
    }

    /// Rotates every head of one row (`[heads, head_dim]`) in place.
    pub fn apply(&self, row: &mut [f32], cos: &[f32], sin: &[f32]) {
        let half = self.rotary_dims / 2;
        for head in row.chunks_exact_mut(self.head_dim) {
            for i in 0..half {
                let (a, b) = match self.layout {
                    FishDrama3RopeLayout::HalfSplit => (i, i + half),
                    FishDrama3RopeLayout::InterleavedPairs => (2 * i, 2 * i + 1),
                };
                let (x1, x2) = (head[a], head[b]);
                head[a] = x1 * cos[i] - x2 * sin[i];
                head[b] = x2 * cos[i] + x1 * sin[i];
            }
        }
    }
}

pub(crate) struct FishDrama3Attention {
    q_proj: FishDrama3Linear,
    k_proj: FishDrama3Linear,
    v_proj: FishDrama3Linear,
    o_proj: FishDrama3Linear,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    config: FishDrama3AttentionConfig,
    rope: FishDrama3Rope,
    eps: f32,
    device: Device,
}

impl FishDrama3Attention {
    pub(crate) fn load(
        checkpoint: &FishDrama3Checkpoint,
        layer: usize,
        trunk: &FishDrama3TrunkConfig,
        forms: &FishDrama3ClassForms,
        device: &Device,
    ) -> Result<Self> {
        let a = trunk.attention;
        let h = trunk.hidden;
        let x = format!("{}.self_attn", trunk_layer(layer));
        let input = forms.form(FishDrama3ProjectionClass::AttentionIn);
        let output = forms.form(FishDrama3ProjectionClass::AttentionOut);
        let gain = |name: &str| {
            norm_gain(
                checkpoint,
                &format!("{x}.{name}.weight"),
                a.head_dim,
                NormGain::ZeroCentred,
            )
        };
        Ok(Self {
            q_proj: load_linear(
                checkpoint,
                &format!("{x}.q_proj.weight"),
                2 * a.q_width(),
                h,
                input,
                device,
            )?,
            k_proj: load_linear(
                checkpoint,
                &format!("{x}.k_proj.weight"),
                a.kv_width(),
                h,
                input,
                device,
            )?,
            v_proj: load_linear(
                checkpoint,
                &format!("{x}.v_proj.weight"),
                a.kv_width(),
                h,
                input,
                device,
            )?,
            o_proj: load_linear(
                checkpoint,
                &format!("{x}.o_proj.weight"),
                h,
                a.q_width(),
                output,
                device,
            )?,
            q_norm: gain("q_norm")?,
            k_norm: gain("k_norm")?,
            config: a,
            rope: FishDrama3Rope::new(&a),
            eps: trunk.rms_norm_eps as f32,
            device: device.clone(),
        })
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        [&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj]
            .iter()
            .map(|l| l.resident_bytes())
            .sum::<usize>()
            + (self.q_norm.len() + self.k_norm.len()) * 4
    }

    pub(crate) fn projections(&self) -> [&FishDrama3Linear; 4] {
        [&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj]
    }

    /// The mixer output `[n, hidden]` of the normed rows `h` at positions
    /// `start..start + n`; their K/V are written into cache layer `slot`
    /// through `prepared`.
    pub(crate) fn forward(
        &self,
        h: &[f32],
        n: usize,
        start: usize,
        cache: &PhysicalPagedKvCache,
        prepared: &mut PreparedPhysicalPagedStep,
        slot: usize,
        phase: FishDrama3KernelPhase,
    ) -> Result<Vec<f32>> {
        let a = &self.config;
        let (heads, kv_heads, d) = (a.heads, a.kv_heads, a.head_dim);
        let packed = self.q_proj.forward_host(h, n, phase)?;
        let mut keys = self.k_proj.forward_host(h, n, phase)?;
        let values = self.v_proj.forward_host(h, n, phase)?;
        // Each head's 2·d outputs are its query, then its gate.
        let mut queries = Vec::with_capacity(n * heads * d);
        let mut gate = Vec::with_capacity(n * heads * d);
        for head in packed.chunks_exact(2 * d) {
            queries.extend_from_slice(&head[..d]);
            gate.extend_from_slice(&head[d..]);
        }
        let mut queries = rms_norm_rows(&queries, d, &self.q_norm, self.eps);
        keys = rms_norm_rows(&keys, d, &self.k_norm, self.eps);
        for t in 0..n {
            let (cos, sin) = self.rope.cos_sin(start + t);
            self.rope
                .apply(&mut queries[t * heads * d..(t + 1) * heads * d], &cos, &sin);
            self.rope.apply(
                &mut keys[t * kv_heads * d..(t + 1) * kv_heads * d],
                &cos,
                &sin,
            );
        }
        let dtype = cache.arena().config().dtype;
        let tensor = |values: Vec<f32>, heads: usize| -> Result<Tensor> {
            Ok(Tensor::from_vec(values, (n, heads, d), &Device::Cpu)?
                .to_dtype(dtype)?
                .to_device(&self.device)?)
        };
        let scale = (1.0 / (d as f64).sqrt()) as f32;
        let attended = cache.write_and_attend(
            slot,
            prepared,
            &tensor(queries, heads)?,
            &tensor(keys, kv_heads)?,
            &tensor(values, kv_heads)?,
            scale,
        )?;
        let mut attended = attended
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        if attended.len() != gate.len() {
            return Err(Error::InferenceError(format!(
                "attention returned {} values for {} gates",
                attended.len(),
                gate.len()
            )));
        }
        for (o, g) in attended.iter_mut().zip(&gate) {
            *o *= sigmoid(*g);
        }
        self.o_proj.forward_host(&attended, n, phase)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::numeric_check::{relative_l2, xorshift_values};

    /// The F64 rotation of `x` at `position` in `layout`.
    fn reference(x: &[f32], position: usize, config: &FishDrama3AttentionConfig) -> Vec<f64> {
        let rotary = config.rotary_dims;
        let half = rotary / 2;
        let mut out = x.iter().map(|v| f64::from(*v)).collect::<Vec<_>>();
        for head in out.chunks_exact_mut(config.head_dim) {
            let original = head.to_vec();
            for i in 0..half {
                let angle =
                    position as f64 / config.rope_theta.powf((2 * i) as f64 / rotary as f64);
                let (a, b) = match config.rope_layout {
                    FishDrama3RopeLayout::HalfSplit => (i, i + half),
                    FishDrama3RopeLayout::InterleavedPairs => (2 * i, 2 * i + 1),
                };
                head[a] = original[a] * angle.cos() - original[b] * angle.sin();
                head[b] = original[b] * angle.cos() + original[a] * angle.sin();
            }
        }
        out
    }

    /// NeoX half-split RoPE at the published geometry (head 256, rotary 64,
    /// θ 1e7) against F64, at the first, a typical last-row and the last
    /// representable positions; the pass-through dims are untouched.
    #[test]
    fn neox_rope_matches_f64_at_the_published_geometry() {
        let config = FishDrama3AttentionConfig {
            heads: 2,
            kv_heads: 1,
            head_dim: 256,
            rotary_dims: 64,
            rope_theta: 1e7,
            rope_layout: FishDrama3RopeLayout::HalfSplit,
        };
        let rope = FishDrama3Rope::new(&config);
        let x = xorshift_values(2 * 256, 41, 1.0);
        for position in [0usize, 1, 12_287, 262_143] {
            let (cos, sin) = rope.cos_sin(position);
            let mut ours = x.clone();
            rope.apply(&mut ours, &cos, &sin);
            let expected = reference(&x, position, &config);
            let worst = ours
                .iter()
                .zip(&expected)
                .map(|(a, b)| (f64::from(*a) - b).abs())
                .fold(0f64, f64::max);
            assert!(worst < 4e-7, "position {position}: {worst}");
            for head in 0..2 {
                let at = head * 256;
                assert_eq!(ours[at + 64..at + 256], x[at + 64..at + 256]);
            }
            if position == 0 {
                assert_eq!(ours, x);
            }
            // RED: the interleaved convention is far off past position 0.
            if position > 0 {
                let interleaved = reference(
                    &x,
                    position,
                    &FishDrama3AttentionConfig {
                        rope_layout: FishDrama3RopeLayout::InterleavedPairs,
                        ..config
                    },
                );
                let red = interleaved.iter().map(|v| *v as f32).collect::<Vec<_>>();
                assert!(relative_l2(&ours, &red) > 0.05, "{position}");
            }
        }
        // The pair frequencies the table rotates by.
        let (cos, sin) = rope.cos_sin(1);
        assert_eq!((cos[0], sin[0]), (1f64.cos() as f32, 1f64.sin() as f32));
        let last = 1e7f64.powf(-62.0 / 64.0);
        assert_eq!(sin[31], last.sin() as f32);
    }
}
