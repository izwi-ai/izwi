//! Trunk parity on the synthetic checkpoint against an independent F64
//! reference, written from upstream's `fish_omni` modules and reading the
//! weights straight from the checkpoint (BF16 widened, E4M3 × scale).
//!
//! The reference takes its conventions as switches. With upstream's it must
//! match the trunk within the tolerances below; with any one switch flipped
//! (a RED variant) it must miss by far more, which proves each convention is
//! both followed and visible at this geometry. It can also hold the Q8_0
//! weights a CPU load computes with (Candle's quantizer, round-tripped), so
//! the Q8_0 wiring is checked as tightly as the F32 one.
//!
//! Tolerances, relative L2 over every output value (measured on aarch64 in
//! brackets):
//! * against the weights the load holds, F32 or Q8_0, on spans and split
//!   spans (and on F32 steps): 1e-5 per layer [≤ 6.6e-7] and 2e-5 for the
//!   whole trunk [≤ 3.2e-6]. Only F32 rounding differs: another summation
//!   order, the chunked against the recurrent DeltaNet form, RoPE's cos and
//!   sin rounded to F32.
//! * Q8_0 steps, whose row-wise kernel quantizes the activations to Q8_0
//!   as well: 3e-2 per layer [≤ 1.2e-2], 0.3 for the trunk [0.115];
//! * Q8_0 spans against the BF16 weights (the cost of Q8_0 weights): 3e-2
//!   per layer [≤ 1.0e-2], 0.25 for the trunk [0.095]. The synthetic trunk
//!   is untrained and deep for its width, so a 1% perturbation grows to
//!   5-20% per row by the last layer; a trained one is far less sensitive.
//! * RED variants against the F32 trunk: at least 0.1 [≥ 0.13], 5,000
//!   times its tolerance.

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};

use super::super::checkpoint::{
    expert, trunk_layer, FishDrama3Checkpoint, CODEBOOK_EMBEDDINGS, TRUNK_PREFIX,
};
use super::super::config::{FishDrama3Config, FishDrama3LayerKind, FishDrama3RopeLayout};
use super::super::embed::FishDrama3Embedding;
use super::super::residency::{
    FishDrama3ClassForms, FishDrama3CpuPrecision, FishDrama3ProjectionClass, FishDrama3WeightForm,
};
use super::super::synthetic::SyntheticCheckpoint;
use super::*;
use crate::kernels::cuda::fp8::decode_e4m3fn;
use crate::kernels::numeric_check::{relative_l2, xorshift_values};
use crate::models::shared::attention::physical::test_physical_caches_for_model_layers_on;

const F32_LAYER_TOLERANCE: f64 = 1e-5;
const F32_TRUNK_TOLERANCE: f64 = 2e-5;
const Q8_0_LAYER_TOLERANCE: f64 = 3e-2;
const Q8_0_STEP_TRUNK_TOLERANCE: f64 = 0.3;
const Q8_0_TRUNK_TOLERANCE: f64 = 0.25;
const RED_MARGIN: f64 = 0.1;

/// The conventions the reference follows.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Conventions {
    /// `1 + w` on the zero-centred trunk norms (input, post-attention,
    /// final, q/k); `false` uses `w`.
    zero_centred: bool,
    /// A plain `w` on the DeltaNet output norm; `false` uses `1 + w`.
    gdn_plain: bool,
    /// Value head `j` reads key head `j / groups`; `false` tiles (`j % K`).
    grouped: bool,
    rope: FishDrama3RopeLayout,
    /// The router's logits as BF16 running sums.
    bf16_router: bool,
    /// Semantic rows scaled by `1 / sqrt(2)`.
    cb1_scaled: bool,
}

const UPSTREAM: Conventions = Conventions {
    zero_centred: true,
    gdn_plain: true,
    grouped: true,
    rope: FishDrama3RopeLayout::HalfSplit,
    bf16_router: false,
    cb1_scaled: true,
};

fn sigmoid64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn silu64(x: f64) -> f64 {
    x * sigmoid64(x)
}

fn softplus64(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// `x @ w.T`, `w` `[rows, cols]`.
fn linear64(x: &[f64], w: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    x.chunks_exact(cols)
        .flat_map(|x| {
            w.chunks_exact(cols)
                .take(rows)
                .map(|w| x.iter().zip(w).map(|(a, b)| a * b).sum::<f64>())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn rms64(x: &[f64], gain: &[f64], eps: f64) -> Vec<f64> {
    let width = gain.len();
    x.chunks_exact(width)
        .flat_map(|row| {
            let r = 1.0 / (row.iter().map(|v| v * v).sum::<f64>() / width as f64 + eps).sqrt();
            row.iter()
                .zip(gain)
                .map(move |(v, g)| v * r * g)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn f32s(x: &[f64]) -> Vec<f32> {
    x.iter().map(|v| *v as f32).collect()
}

fn f64s(x: &[f32]) -> Vec<f64> {
    x.iter().map(|v| f64::from(*v)).collect()
}

struct Reference<'a> {
    ck: &'a FishDrama3Checkpoint,
    config: &'a FishDrama3Config,
    conv: Conventions,
    /// The projections a Q8_0 load quantizes, round-tripped through
    /// Candle's Q8_0 quantizer: the weights that load computes with.
    q8: bool,
}

impl<'a> Reference<'a> {
    fn new(ck: &'a FishDrama3Checkpoint, config: &'a FishDrama3Config) -> Self {
        Self {
            ck,
            config,
            conv: UPSTREAM,
            q8: false,
        }
    }

    fn with(self, conv: Conventions) -> Self {
        Self { conv, ..self }
    }

    /// Q8_0 weights when `q8` is set.
    fn holding(self, precision: FishDrama3CpuPrecision) -> Self {
        Self {
            q8: precision == FishDrama3CpuPrecision::Q8_0,
            ..self
        }
    }

    fn tensor(&self, name: &str, shape: &[usize]) -> Vec<f64> {
        f64s(&self.ck.bf16_to_f32(name, shape).unwrap())
    }

    fn quantized(&self, values: Vec<f32>, rows: usize, cols: usize) -> Vec<f64> {
        if !self.q8 {
            return f64s(&values);
        }
        let tensor = Tensor::from_vec(values, (rows, cols), &Device::Cpu).unwrap();
        let q = QTensor::quantize(&tensor, GgmlDType::Q8_0).unwrap();
        f64s(&host(&q.dequantize(&Device::Cpu).unwrap()))
    }

    /// A BF16 projection a Q8_0 load quantizes (not the router or the
    /// DeltaNet gates, which stay F32).
    fn projection(&self, name: &str, rows: usize, cols: usize) -> Vec<f64> {
        self.quantized(
            self.ck.bf16_to_f32(name, &[rows, cols]).unwrap(),
            rows,
            cols,
        )
    }

    /// An FP8 expert projection: `e4m3(code) * scale[r / 128][c / 128]`.
    fn fp8(&self, module: &str, rows: usize, cols: usize) -> Vec<f64> {
        let block = self.config.fp8_block;
        let w = self.ck.fp8_block_weight(module, rows, cols, block).unwrap();
        let grid = cols.div_ceil(block);
        let exact = w
            .codes
            .iter()
            .enumerate()
            .map(|(i, &code)| {
                let (r, c) = (i / cols, i % cols);
                f64::from(decode_e4m3fn(code)) * f64::from(w.scales[(r / block) * grid + c / block])
            })
            .collect::<Vec<_>>();
        if self.q8 {
            self.quantized(f32s(&exact), rows, cols)
        } else {
            exact
        }
    }

    fn zero_centred(&self, name: &str, width: usize) -> Vec<f64> {
        let w = self.tensor(name, &[width]);
        if self.conv.zero_centred {
            w.into_iter().map(|w| 1.0 + w).collect()
        } else {
            w
        }
    }

    fn eps(&self) -> f64 {
        self.config.trunk.rms_norm_eps
    }

    fn embed(&self, rows: &[u32]) -> Vec<f64> {
        let t = &self.config.trunk;
        let v = self.config.tower.codebook_vocab;
        let e = self.tensor(
            &format!("{TRUNK_PREFIX}.embed_tokens.weight"),
            &[t.vocab, t.hidden],
        );
        let cb = self.tensor(CODEBOOK_EMBEDDINGS, &[10 * v, t.hidden]);
        let h = t.hidden;
        rows.as_chunks::<11>()
            .0
            .iter()
            .flat_map(|row| {
                let main = row[0] as usize;
                let er = &e[main * h..(main + 1) * h];
                if self.config.ids.is_semantic(row[0]) {
                    let code = (row[2] as usize).min(v - 1);
                    let c = &cb[(v + code) * h..(v + code + 1) * h];
                    let scale = if self.conv.cb1_scaled {
                        2f64.sqrt()
                    } else {
                        1.0
                    };
                    er.iter()
                        .zip(c)
                        .map(|(a, b)| (a + b) / scale)
                        .collect::<Vec<_>>()
                } else {
                    er.to_vec()
                }
            })
            .collect()
    }

    /// Gated attention over rows `0..n` (causal, positions from 0).
    fn attention(&self, layer: usize, h: &[f64], n: usize) -> Vec<f64> {
        let t = &self.config.trunk;
        let a = &t.attention;
        let (heads, kv, d, hidden) = (a.heads, a.kv_heads, a.head_dim, t.hidden);
        let x = format!("{}.self_attn", trunk_layer(layer));
        let packed = linear64(
            h,
            &self.projection(&format!("{x}.q_proj.weight"), 2 * heads * d, hidden),
            2 * heads * d,
            hidden,
        );
        let k = linear64(
            h,
            &self.projection(&format!("{x}.k_proj.weight"), kv * d, hidden),
            kv * d,
            hidden,
        );
        let v = linear64(
            h,
            &self.projection(&format!("{x}.v_proj.weight"), kv * d, hidden),
            kv * d,
            hidden,
        );
        let (mut q, mut gate) = (Vec::new(), Vec::new());
        for head in packed.chunks_exact(2 * d) {
            q.extend_from_slice(&head[..d]);
            gate.extend_from_slice(&head[d..]);
        }
        let mut q = rms64(
            &q,
            &self.zero_centred(&format!("{x}.q_norm.weight"), d),
            self.eps(),
        );
        let mut k = rms64(
            &k,
            &self.zero_centred(&format!("{x}.k_norm.weight"), d),
            self.eps(),
        );
        let rotary = a.rotary_dims;
        let rotate = |x: &mut [f64], position: usize| {
            for head in x.chunks_exact_mut(d) {
                let original = head.to_vec();
                for i in 0..rotary / 2 {
                    let angle =
                        position as f64 * a.rope_theta.powf(-((2 * i) as f64) / rotary as f64);
                    let (p, r) = match self.conv.rope {
                        FishDrama3RopeLayout::HalfSplit => (i, i + rotary / 2),
                        FishDrama3RopeLayout::InterleavedPairs => (2 * i, 2 * i + 1),
                    };
                    head[p] = original[p] * angle.cos() - original[r] * angle.sin();
                    head[r] = original[r] * angle.cos() + original[p] * angle.sin();
                }
            }
        };
        for t in 0..n {
            rotate(&mut q[t * heads * d..(t + 1) * heads * d], t);
            rotate(&mut k[t * kv * d..(t + 1) * kv * d], t);
        }
        let scale = 1.0 / (d as f64).sqrt();
        let mut out = vec![0f64; n * heads * d];
        for t in 0..n {
            for head in 0..heads {
                let kvh = head / (heads / kv);
                let qh = &q[(t * heads + head) * d..][..d];
                let scores = (0..=t)
                    .map(|s| {
                        qh.iter()
                            .zip(&k[(s * kv + kvh) * d..][..d])
                            .map(|(a, b)| a * b)
                            .sum::<f64>()
                            * scale
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let total = scores.iter().map(|s| (s - max).exp()).sum::<f64>();
                let o = &mut out[(t * heads + head) * d..][..d];
                for (s, score) in scores.iter().enumerate() {
                    let p = (score - max).exp() / total;
                    for (o, v) in o.iter_mut().zip(&v[(s * kv + kvh) * d..][..d]) {
                        *o += p * v;
                    }
                }
            }
        }
        for (o, g) in out.iter_mut().zip(&gate) {
            *o *= sigmoid64(*g);
        }
        linear64(
            &out,
            &self.projection(&format!("{x}.o_proj.weight"), hidden, heads * d),
            hidden,
            heads * d,
        )
    }

    /// Gated DeltaNet over rows `0..n` from the zero state, token by token.
    fn deltanet(&self, layer: usize, h: &[f64], n: usize) -> Vec<f64> {
        let t = &self.config.trunk;
        let la = &t.linear;
        let hidden = t.hidden;
        let (hk, hv, dk, dv) = (
            la.key_heads,
            la.value_heads,
            la.key_head_dim,
            la.value_head_dim,
        );
        let channels = la.conv_channels();
        let taps = la.conv_kernel;
        let x = format!("{}.linear_attn", trunk_layer(layer));
        let mixed = linear64(
            h,
            &self.projection(&format!("{x}.in_proj_qkv.weight"), channels, hidden),
            channels,
            hidden,
        );
        let z = linear64(
            h,
            &self.projection(&format!("{x}.in_proj_z.weight"), hv * dv, hidden),
            hv * dv,
            hidden,
        );
        let b = linear64(
            h,
            &self.tensor(&format!("{x}.in_proj_b.weight"), &[hv, hidden]),
            hv,
            hidden,
        );
        let a = linear64(
            h,
            &self.tensor(&format!("{x}.in_proj_a.weight"), &[hv, hidden]),
            hv,
            hidden,
        );
        let w = self.tensor(&format!("{x}.conv1d.weight"), &[channels, 1, taps]);
        let a_log = self.tensor(&format!("{x}.A_log"), &[hv]);
        let dt = self.tensor(&format!("{x}.dt_bias"), &[hv]);
        let norm = self.tensor(&format!("{x}.norm.weight"), &[dv]);
        let norm = if self.conv.gdn_plain {
            norm
        } else {
            norm.into_iter().map(|w| 1.0 + w).collect()
        };
        // Causal conv over zero history.
        let at = |s: isize, c: usize| {
            if s < 0 {
                0.0
            } else {
                mixed[s as usize * channels + c]
            }
        };
        let mut conv = vec![0f64; n * channels];
        for t in 0..n {
            for c in 0..channels {
                let sum = (0..taps)
                    .map(|j| w[c * taps + j] * at(t as isize + j as isize - (taps as isize - 1), c))
                    .sum::<f64>();
                conv[t * channels + c] = silu64(sum);
            }
        }
        let l2 = |x: &[f64]| {
            let r = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() + 1e-6).sqrt();
            x.iter().map(|v| v * r).collect::<Vec<_>>()
        };
        let mut state = vec![0f64; hv * dk * dv];
        let mut core = vec![0f64; n * hv * dv];
        for t in 0..n {
            let row = &conv[t * channels..(t + 1) * channels];
            for head in 0..hv {
                let kh = if self.conv.grouped {
                    head / (hv / hk)
                } else {
                    head % hk
                };
                let q = l2(&row[kh * dk..(kh + 1) * dk])
                    .into_iter()
                    .map(|v| v / (dk as f64).sqrt())
                    .collect::<Vec<_>>();
                let k = l2(&row[hk * dk + kh * dk..hk * dk + (kh + 1) * dk]);
                let v = &row[2 * hk * dk + head * dv..2 * hk * dk + (head + 1) * dv];
                let beta = sigmoid64(b[t * hv + head]);
                let g = -a_log[head].exp() * softplus64(a[t * hv + head] + dt[head]);
                let s = &mut state[head * dk * dv..(head + 1) * dk * dv];
                s.iter_mut().for_each(|x| *x *= g.exp());
                for j in 0..dv {
                    let mem = (0..dk).map(|i| s[i * dv + j] * k[i]).sum::<f64>();
                    let delta = (v[j] - mem) * beta;
                    for i in 0..dk {
                        s[i * dv + j] += k[i] * delta;
                    }
                }
                for j in 0..dv {
                    core[(t * hv + head) * dv + j] = (0..dk).map(|i| s[i * dv + j] * q[i]).sum();
                }
            }
        }
        let mut gated = rms64(&core, &norm, self.eps());
        for (g, z) in gated.iter_mut().zip(&z) {
            *g *= silu64(*z);
        }
        linear64(
            &gated,
            &self.projection(&format!("{x}.out_proj.weight"), hidden, hv * dv),
            hidden,
            hv * dv,
        )
    }

    /// The MoE block, upstream's routing with BF16-rounded weights.
    fn moe(&self, layer: usize, h: &[f64], n: usize) -> Vec<f64> {
        let t = &self.config.trunk;
        let (hidden, e, top_k, mi, si) = (
            t.hidden,
            t.moe.experts,
            t.moe.top_k,
            t.moe.intermediate,
            t.moe.shared_intermediate,
        );
        let m = format!("{}.mlp", trunk_layer(layer));
        let router = self.tensor(&format!("{m}.gate.weight"), &[e, hidden]);
        let swiglu = |x: &[f64], gate: &[f64], up: &[f64], down: &[f64], inter: usize| {
            let g = linear64(x, gate, inter, hidden);
            let u = linear64(x, up, inter, hidden);
            let act = g
                .iter()
                .zip(&u)
                .map(|(g, u)| silu64(*g) * u)
                .collect::<Vec<_>>();
            linear64(&act, down, hidden, inter)
        };
        let experts = (0..e)
            .map(|index| {
                let x = expert(layer, index);
                (
                    self.fp8(&format!("{x}.gate_proj"), mi, hidden),
                    self.fp8(&format!("{x}.up_proj"), mi, hidden),
                    self.fp8(&format!("{x}.down_proj"), hidden, mi),
                )
            })
            .collect::<Vec<_>>();
        let s = format!("{m}.shared_expert");
        let shared = (
            self.projection(&format!("{s}.gate_proj.weight"), si, hidden),
            self.projection(&format!("{s}.up_proj.weight"), si, hidden),
            self.projection(&format!("{s}.down_proj.weight"), hidden, si),
        );
        let shared_gate = self.tensor(&format!("{m}.shared_expert_gate.weight"), &[1, hidden]);
        let mut out = Vec::with_capacity(n * hidden);
        for x in h.chunks_exact(hidden).take(n) {
            let logits = router
                .chunks_exact(hidden)
                .map(|w| {
                    if self.conv.bf16_router {
                        w.iter()
                            .zip(x)
                            .fold(half::bf16::ZERO, |acc, (a, b)| {
                                half::bf16::from_f64(
                                    acc.to_f64() + half::bf16::from_f64(a * b).to_f64(),
                                )
                            })
                            .to_f64()
                    } else {
                        w.iter().zip(x).map(|(a, b)| a * b).sum::<f64>()
                    }
                })
                .collect::<Vec<_>>();
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total = logits.iter().map(|v| (v - max).exp()).sum::<f64>();
            let p = logits
                .iter()
                .map(|v| (v - max).exp() / total)
                .collect::<Vec<_>>();
            let mut order = (0..e).collect::<Vec<_>>();
            order.sort_by(|a, b| p[*b].total_cmp(&p[*a]));
            order.truncate(top_k);
            let sum = order.iter().map(|i| p[*i]).sum::<f64>();
            let mut y = vec![0f64; hidden];
            for &index in &order {
                let weight = f64::from(half::bf16::from_f64(p[index] / sum).to_f32());
                let (g, u, d) = &experts[index];
                for (y, v) in y.iter_mut().zip(swiglu(x, g, u, d, mi)) {
                    *y += weight * v;
                }
            }
            let gate = sigmoid64(x.iter().zip(&shared_gate).map(|(a, b)| a * b).sum());
            for (y, v) in y
                .iter_mut()
                .zip(swiglu(x, &shared.0, &shared.1, &shared.2, si))
            {
                *y += gate * v;
            }
            out.extend(y);
        }
        out
    }

    /// The whole trunk over rows `0..n`: the final-norm hidden.
    fn trunk(&self, mut x: Vec<f64>, n: usize) -> Vec<f64> {
        let t = &self.config.trunk;
        let hidden = t.hidden;
        for (layer, kind) in t.layers.iter().enumerate() {
            let l = trunk_layer(layer);
            let normed = rms64(
                &x,
                &self.zero_centred(&format!("{l}.input_layernorm.weight"), hidden),
                self.eps(),
            );
            let mixed = match kind {
                FishDrama3LayerKind::FullAttention => self.attention(layer, &normed, n),
                FishDrama3LayerKind::LinearAttention => self.deltanet(layer, &normed, n),
            };
            x.iter_mut().zip(&mixed).for_each(|(x, m)| *x += m);
            let normed = rms64(
                &x,
                &self.zero_centred(&format!("{l}.post_attention_layernorm.weight"), hidden),
                self.eps(),
            );
            let moe = self.moe(layer, &normed, n);
            x.iter_mut().zip(&moe).for_each(|(x, m)| *x += m);
        }
        rms64(
            &x,
            &self.zero_centred(&format!("{TRUNK_PREFIX}.norm.weight"), hidden),
            self.eps(),
        )
    }
}

/// 20 prompt-like rows: text, a run of semantic frames (one with an
/// out-of-range code 1, clamped), then text.
fn rows(config: &FishDrama3Config) -> Vec<u32> {
    let ids = config.ids;
    let draws = xorshift_values(20 * 11, 77, 1.0);
    let pick = |i: usize, limit: u32| ((draws[i].abs() * limit as f32) as u32).min(limit - 1);
    let mut out = Vec::new();
    for t in 0..20 {
        let mut row = vec![0u32; 11];
        if (8..14).contains(&t) {
            let c0 = pick(t * 11, 64);
            row[0] = ids.semantic_start + c0;
            row[1] = c0;
            for k in 2..11 {
                row[k] = pick(t * 11 + k, 64);
            }
            if t == 10 {
                row[2] = 900;
            }
        } else {
            row[0] = pick(t * 11, ids.eos);
        }
        out.extend(row);
    }
    out
}

fn caches(
    trunk_layers: &[u32],
    config: &FishDrama3Config,
    rows: usize,
) -> Vec<PhysicalPagedKvCache> {
    let a = &config.trunk.attention;
    test_physical_caches_for_model_layers_on(
        &Device::Cpu,
        DType::F32,
        trunk_layers,
        a.kv_heads,
        a.head_dim,
        64,
        rows,
    )
}

fn host(tensor: &Tensor) -> Vec<f32> {
    tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn forms(precision: FishDrama3CpuPrecision) -> FishDrama3ClassForms {
    match precision {
        FishDrama3CpuPrecision::F32 => FishDrama3ClassForms::uniform(FishDrama3WeightForm::F32),
        FishDrama3CpuPrecision::Q8_0 => FishDrama3ClassForms::uniform(FishDrama3WeightForm::Q8_0),
    }
}

fn load(synthetic: &SyntheticCheckpoint, precision: FishDrama3CpuPrecision) -> FishDrama3Trunk {
    FishDrama3Trunk::load(
        &synthetic.open(),
        &synthetic.config,
        &Device::Cpu,
        FishDrama3TrunkOptions {
            precision,
            ..Default::default()
        },
    )
    .unwrap()
}

/// Measured tolerances against `reference`: `(name, got, tolerance)`.
fn check(results: &[(&str, &[f32], f64)], reference: &[f32], context: &str) {
    for (name, got, tolerance) in results {
        let error = relative_l2(got, reference);
        assert!(
            error < *tolerance,
            "{context} {name}: {error:.3e} >= {tolerance:.1e}"
        );
    }
}

/// The step tolerance against the weights the load holds: tight on F32,
/// Q8_0-activation noise on Q8_0.
fn step_tolerance(precision: FishDrama3CpuPrecision, f32_tolerance: f64, q8_tolerance: f64) -> f64 {
    match precision {
        FishDrama3CpuPrecision::F32 => f32_tolerance,
        FishDrama3CpuPrecision::Q8_0 => q8_tolerance,
    }
}

/// One attention layer (layer 3): spans match F64 on the weights the load
/// holds tightly on both forms, steps tightly on F32; interleaved RoPE is
/// far off.
#[test]
fn an_attention_layer_matches_f64() {
    let synthetic = SyntheticCheckpoint::write(11);
    let config = &synthetic.config;
    let ck = synthetic.open();
    let (n, hidden) = (20, config.trunk.hidden);
    let h = xorshift_values(n * hidden, 5, 1.0);
    let upstream = f32s(&Reference::new(&ck, config).attention(3, &f64s(&h), n));
    for precision in [FishDrama3CpuPrecision::F32, FishDrama3CpuPrecision::Q8_0] {
        let held = f32s(
            &Reference::new(&ck, config)
                .holding(precision)
                .attention(3, &f64s(&h), n),
        );
        let attention =
            FishDrama3Attention::load(&ck, 3, &config.trunk, &forms(precision), &Device::Cpu)
                .unwrap();
        let mut cache = caches(&[3], config, 1).remove(0);
        let mut prepared = cache.prepare_append(0, n).unwrap();
        let span = attention
            .forward(
                &h,
                n,
                0,
                &cache,
                &mut prepared,
                0,
                FishDrama3KernelPhase::Span,
            )
            .unwrap();
        cache.commit_prepared(prepared).unwrap();
        let mut cache = caches(&[3], config, 1).remove(0);
        let mut steps = Vec::new();
        for t in 0..n {
            let mut prepared = cache.prepare_append(t, 1).unwrap();
            steps.extend(
                attention
                    .forward(
                        &h[t * hidden..(t + 1) * hidden],
                        1,
                        t,
                        &cache,
                        &mut prepared,
                        0,
                        FishDrama3KernelPhase::Step,
                    )
                    .unwrap(),
            );
            cache.commit_prepared(prepared).unwrap();
        }
        let context = format!("attention {precision:?}");
        check(
            &[
                ("span", &span, F32_LAYER_TOLERANCE),
                (
                    "step",
                    &steps,
                    step_tolerance(precision, F32_LAYER_TOLERANCE, Q8_0_LAYER_TOLERANCE),
                ),
            ],
            &held,
            &context,
        );
        check(
            &[("span vs BF16 weights", &span, Q8_0_LAYER_TOLERANCE)],
            &upstream,
            &context,
        );
    }
    let red = f32s(
        &Reference::new(&ck, config)
            .with(Conventions {
                rope: FishDrama3RopeLayout::InterleavedPairs,
                ..UPSTREAM
            })
            .attention(3, &f64s(&h), n),
    );
    let error = relative_l2(&red, &upstream);
    assert!(error > RED_MARGIN, "interleaved RoPE: {error}");
}

/// One DeltaNet layer (layer 0): the chunked span, split spans and the
/// recurrent steps against the F64 recurrence; tiled value heads and a
/// `1 + w` output norm are far off.
#[test]
fn a_deltanet_layer_matches_f64() {
    let synthetic = SyntheticCheckpoint::write(12);
    let config = &synthetic.config;
    let ck = synthetic.open();
    let (n, hidden) = (20, config.trunk.hidden);
    let h = xorshift_values(n * hidden, 6, 1.0);
    let upstream = f32s(&Reference::new(&ck, config).deltanet(0, &f64s(&h), n));
    for precision in [FishDrama3CpuPrecision::F32, FishDrama3CpuPrecision::Q8_0] {
        let held = f32s(
            &Reference::new(&ck, config)
                .holding(precision)
                .deltanet(0, &f64s(&h), n),
        );
        let net = FishDrama3DeltaNet::load(&ck, 0, &config.trunk, &forms(precision), &Device::Cpu)
            .unwrap();
        let fresh = || {
            FishDrama3TrunkState::new(&config.trunk, &Device::Cpu)
                .unwrap()
                .layers()
                .unwrap()[0]
                .clone()
        };
        let mut state = fresh();
        let span = net
            .forward(&h, n, &mut state, FishDrama3KernelPhase::Span)
            .unwrap();
        let mut split = fresh();
        let mut pieces = Vec::new();
        for (a, b) in [(0, 7), (7, 8), (8, 20)] {
            pieces.extend(
                net.forward(
                    &h[a * hidden..b * hidden],
                    b - a,
                    &mut split,
                    FishDrama3KernelPhase::Span,
                )
                .unwrap(),
            );
        }
        let mut stepped = fresh();
        let mut steps = Vec::new();
        for t in 0..n {
            steps.extend(
                net.forward(
                    &h[t * hidden..(t + 1) * hidden],
                    1,
                    &mut stepped,
                    FishDrama3KernelPhase::Step,
                )
                .unwrap(),
            );
        }
        let step = step_tolerance(precision, F32_LAYER_TOLERANCE, Q8_0_LAYER_TOLERANCE);
        let context = format!("deltanet {precision:?}");
        check(
            &[
                ("span", &span, F32_LAYER_TOLERANCE),
                ("split", &pieces, F32_LAYER_TOLERANCE),
                ("step", &steps, step),
            ],
            &held,
            &context,
        );
        check(
            &[("span vs BF16 weights", &span, Q8_0_LAYER_TOLERANCE)],
            &upstream,
            &context,
        );
        // The split spans and the steps leave the span's state.
        let span_state = [
            state.conv.ordered_host().unwrap(),
            state.recurrent_host().unwrap(),
        ]
        .concat();
        let split_state = [
            split.conv.ordered_host().unwrap(),
            split.recurrent_host().unwrap(),
        ]
        .concat();
        let step_state = [
            stepped.conv.ordered_host().unwrap(),
            stepped.recurrent_host().unwrap(),
        ]
        .concat();
        check(
            &[
                ("split state", &split_state, F32_LAYER_TOLERANCE),
                ("step state", &step_state, step),
            ],
            &span_state,
            &context,
        );
    }
    for (name, conv) in [
        (
            "tiled value heads",
            Conventions {
                grouped: false,
                ..UPSTREAM
            },
        ),
        (
            "1 + w on the DeltaNet norm",
            Conventions {
                gdn_plain: false,
                ..UPSTREAM
            },
        ),
    ] {
        let red = f32s(
            &Reference::new(&ck, config)
                .with(conv)
                .deltanet(0, &f64s(&h), n),
        );
        let error = relative_l2(&red, &upstream);
        assert!(error > RED_MARGIN, "{name}: {error}");
    }
}

/// One MoE layer (layer 1) on both forms against F64, and the loaded F32
/// expert weights are exactly `e4m3 * scale`.
#[test]
fn a_moe_layer_matches_f64() {
    let synthetic = SyntheticCheckpoint::write(13);
    let config = &synthetic.config;
    let ck = synthetic.open();
    let (n, hidden) = (20, config.trunk.hidden);
    let h = xorshift_values(n * hidden, 7, 1.0);
    let upstream = f32s(&Reference::new(&ck, config).moe(1, &f64s(&h), n));
    for precision in [FishDrama3CpuPrecision::F32, FishDrama3CpuPrecision::Q8_0] {
        let held = f32s(
            &Reference::new(&ck, config)
                .holding(precision)
                .moe(1, &f64s(&h), n),
        );
        let moe = FishDrama3Moe::load(
            &ck,
            1,
            &config.trunk,
            &forms(precision),
            config.fp8_block,
            &Device::Cpu,
        )
        .unwrap();
        let span = moe.forward(&h, n, FishDrama3KernelPhase::Span).unwrap();
        let steps = (0..n)
            .flat_map(|t| {
                moe.forward(
                    &h[t * hidden..(t + 1) * hidden],
                    1,
                    FishDrama3KernelPhase::Step,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let context = format!("moe {precision:?}");
        check(
            &[
                ("span", &span, F32_LAYER_TOLERANCE),
                (
                    "step",
                    &steps,
                    step_tolerance(precision, F32_LAYER_TOLERANCE, Q8_0_LAYER_TOLERANCE),
                ),
            ],
            &held,
            &context,
        );
        check(
            &[("span vs BF16 weights", &span, Q8_0_LAYER_TOLERANCE)],
            &upstream,
            &context,
        );
        if precision == FishDrama3CpuPrecision::F32 {
            // Column 37 of an expert's gate projection, read through a basis
            // vector, is the exact dequantization.
            let gate = moe.expert(6).projections()[0];
            let mut basis = vec![0f32; hidden];
            basis[37] = 1.0;
            let column = gate
                .forward_host(&basis, 1, FishDrama3KernelPhase::Span)
                .unwrap();
            let exact = Reference::new(&ck, config).fp8(
                &format!("{}.gate_proj", expert(1, 6)),
                config.trunk.moe.intermediate,
                hidden,
            );
            for (row, value) in column.iter().enumerate() {
                assert_eq!(*value, exact[row * hidden + 37] as f32);
            }
        }
    }
}

/// The whole trunk over 20 rows on every path (one span, split spans, row
/// by row steps) against F64, on F32 and Q8_0 weights, and every RED
/// variant far off.
#[test]
fn the_trunk_matches_f64_on_every_path() {
    let synthetic = SyntheticCheckpoint::write(14);
    let config = &synthetic.config;
    let ck = synthetic.open();
    let embedding = FishDrama3Embedding::load(&ck, config).unwrap();
    let rows = rows(config);
    let n = rows.len() / 11;
    let hidden = config.trunk.hidden;
    // The embedding is exact (embed.rs), so the reference may start from it.
    let embeds = embedding.embed(&rows).unwrap();
    let independent = f32s(&Reference::new(&ck, config).embed(&rows));
    assert!(relative_l2(&embeds, &independent) < 1e-7);
    let upstream = f32s(&Reference::new(&ck, config).trunk(f64s(&embeds), n));
    let embeds_tensor = Tensor::from_vec(embeds.clone(), (n, hidden), &Device::Cpu).unwrap();
    let mut f32_span = Vec::new();
    for precision in [FishDrama3CpuPrecision::F32, FishDrama3CpuPrecision::Q8_0] {
        let held = f32s(
            &Reference::new(&ck, config)
                .holding(precision)
                .trunk(f64s(&embeds), n),
        );
        let trunk = load(&synthetic, precision);
        let layers = trunk.attention_layers().to_vec();
        assert_eq!(layers, vec![3, 7]);
        // One span.
        let mut cache = caches(&layers, config, 1).remove(0);
        let mut state = trunk.new_state().unwrap();
        let span = host(
            &trunk
                .forward_span(&embeds_tensor, 0, &mut state, &mut cache)
                .unwrap(),
        );
        assert_eq!((cache.context_len(), state.position()), (n, n));
        // Tracing records every layer and changes no bit.
        let mut cache = caches(&layers, config, 1).remove(0);
        let mut traced_state = trunk.new_state().unwrap();
        let (traced, trace) = trunk
            .forward_span_traced(&embeds_tensor, 0, &mut traced_state, &mut cache)
            .unwrap();
        assert_eq!(host(&traced), span);
        let depth = config.trunk.layers.len();
        for part in [&trace.outputs, &trace.mixers, &trace.moes] {
            assert_eq!(part.len(), depth);
            assert!(part.iter().all(|layer| layer.len() == n * hidden));
        }
        assert_eq!(trace.router_logits.len(), depth);
        assert!(trace
            .router_logits
            .iter()
            .all(|layer| layer.len() == n * config.trunk.moe.experts));
        // Each layer's output is its input plus its two parts.
        let mut x = embeds.clone();
        for layer in 0..depth {
            for (i, value) in x.iter_mut().enumerate() {
                *value += trace.mixers[layer][i];
                *value += trace.moes[layer][i];
            }
            assert_eq!(x, trace.outputs[layer], "layer {layer}");
        }
        // Split spans.
        let mut cache = caches(&layers, config, 1).remove(0);
        let mut split_state = trunk.new_state().unwrap();
        let mut split = Vec::new();
        for (a, b) in [(0, 7), (7, 8), (8, 20)] {
            split.extend(host(
                &trunk
                    .forward_span(
                        &embeds_tensor.narrow(0, a, b - a).unwrap(),
                        a,
                        &mut split_state,
                        &mut cache,
                    )
                    .unwrap(),
            ));
        }
        // Row by row.
        let mut cache = caches(&layers, config, 1).remove(0);
        let mut step_state = trunk.new_state().unwrap();
        let mut steps = Vec::new();
        for t in 0..n {
            let mut row = [FishDrama3StepRow {
                state: &mut step_state,
                cache: &mut cache,
            }];
            steps.extend(host(
                &trunk
                    .forward_step(&embeds_tensor.narrow(0, t, 1).unwrap(), &mut row)
                    .unwrap(),
            ));
        }
        let step = step_tolerance(precision, F32_TRUNK_TOLERANCE, Q8_0_STEP_TRUNK_TOLERANCE);
        let context = format!("trunk {precision:?}");
        check(
            &[
                ("span", &span, F32_TRUNK_TOLERANCE),
                ("split", &split, F32_TRUNK_TOLERANCE),
                ("step", &steps, step),
            ],
            &held,
            &context,
        );
        check(
            &[
                ("split vs span", &split, F32_TRUNK_TOLERANCE),
                ("step vs span", &steps, step),
            ],
            &span,
            &context,
        );
        check(
            &[("span vs BF16 weights", &span, Q8_0_TRUNK_TOLERANCE)],
            &upstream,
            &context,
        );
        if precision == FishDrama3CpuPrecision::F32 {
            f32_span = span;
        }
    }

    // RED: each convention flipped in the reference misses the F32 trunk by
    // far more than its tolerance.
    for (name, conv) in [
        (
            "no 1 + w on the trunk norms",
            Conventions {
                zero_centred: false,
                ..UPSTREAM
            },
        ),
        (
            "1 + w on the DeltaNet norm",
            Conventions {
                gdn_plain: false,
                ..UPSTREAM
            },
        ),
        (
            "tiled value heads",
            Conventions {
                grouped: false,
                ..UPSTREAM
            },
        ),
        (
            "interleaved RoPE",
            Conventions {
                rope: FishDrama3RopeLayout::InterleavedPairs,
                ..UPSTREAM
            },
        ),
        (
            "BF16 router accumulation",
            Conventions {
                bf16_router: true,
                ..UPSTREAM
            },
        ),
        (
            "CB1 unscaled",
            Conventions {
                cb1_scaled: false,
                ..UPSTREAM
            },
        ),
    ] {
        let red = Reference::new(&ck, config).with(conv);
        let red = f32s(&red.trunk(red.embed(&rows), n));
        let error = relative_l2(&f32_span, &red);
        assert!(error > RED_MARGIN, "{name}: {error}");
    }
}

/// A step over several requests is a loop over them: each row's bits are
/// the bits it gets alone, whatever the order, and every row is one trunk
/// pass.
#[test]
fn a_step_batch_is_a_per_row_loop() {
    let synthetic = SyntheticCheckpoint::write(15);
    let config = &synthetic.config;
    let trunk = load(&synthetic, FishDrama3CpuPrecision::Q8_0);
    let embedding = FishDrama3Embedding::load(&synthetic.open(), config).unwrap();
    let rows = rows(config);
    let hidden = config.trunk.hidden;
    let embeds = embedding.embed_tensor(&rows, &Device::Cpu).unwrap();
    let layers = trunk.attention_layers().to_vec();
    // Three requests prefilled to different lengths.
    let lengths = [3usize, 9, 5];
    let prefill = || {
        let mut caches = caches(&layers, config, 3);
        let mut states = (0..3)
            .map(|_| trunk.new_state().unwrap())
            .collect::<Vec<_>>();
        for (r, length) in lengths.iter().enumerate() {
            trunk
                .forward_span(
                    &embeds.narrow(0, r, *length).unwrap(),
                    0,
                    &mut states[r],
                    &mut caches[r],
                )
                .unwrap();
        }
        (caches, states)
    };
    let next = Tensor::cat(
        &[
            embeds.narrow(0, 15, 1).unwrap(),
            embeds.narrow(0, 10, 1).unwrap(),
            embeds.narrow(0, 2, 1).unwrap(),
        ],
        0,
    )
    .unwrap();
    // Alone.
    let mut alone = Vec::new();
    for r in 0..3 {
        let (mut caches, mut states) = prefill();
        let mut row = [FishDrama3StepRow {
            state: &mut states[r],
            cache: &mut caches[r],
        }];
        alone.push(host(
            &trunk
                .forward_step(&next.narrow(0, r, 1).unwrap(), &mut row)
                .unwrap(),
        ));
        assert_eq!(states[r].position(), lengths[r] + 1);
    }
    // Together, in two orders.
    for order in [[0usize, 1, 2], [2, 0, 1]] {
        let (mut caches, mut states) = prefill();
        let before = trunk.step_rows();
        let batch = Tensor::cat(
            &order
                .iter()
                .map(|r| next.narrow(0, *r, 1).unwrap())
                .collect::<Vec<_>>(),
            0,
        )
        .unwrap();
        let mut refs = states
            .iter_mut()
            .zip(caches.iter_mut())
            .map(|(state, cache)| Some((state, cache)))
            .collect::<Vec<_>>();
        let mut batch_rows = order
            .iter()
            .map(|r| {
                let (state, cache) = refs[*r].take().unwrap();
                FishDrama3StepRow { state, cache }
            })
            .collect::<Vec<_>>();
        let out = host(&trunk.forward_step(&batch, &mut batch_rows).unwrap());
        assert_eq!(trunk.step_rows() - before, 3);
        for (slot, r) in order.iter().enumerate() {
            assert_eq!(
                out[slot * hidden..(slot + 1) * hidden],
                alone[*r][..],
                "request {r} at {slot}"
            );
        }
        drop(batch_rows);
        for (r, length) in lengths.iter().enumerate() {
            assert_eq!(
                (states[r].position(), caches[r].context_len()),
                (length + 1, length + 1)
            );
        }
    }
    // A failing row leaves every row where it was.
    let (mut caches, mut states) = prefill();
    let snapshot = states.iter().map(|s| s.values()).collect::<Vec<_>>();
    let (first, rest) = states.split_at_mut(1);
    let (cache_first, cache_rest) = caches.split_at_mut(1);
    // The second row claims a position its cache does not hold.
    let mut wrong = rest[0].clone();
    wrong.install(rest[0].layers().unwrap().to_vec(), 1);
    let mut batch_rows = [
        FishDrama3StepRow {
            state: &mut first[0],
            cache: &mut cache_first[0],
        },
        FishDrama3StepRow {
            state: &mut wrong,
            cache: &mut cache_rest[0],
        },
    ];
    assert!(trunk
        .forward_step(&next.narrow(0, 0, 2).unwrap(), &mut batch_rows)
        .is_err());
    assert_eq!((states[0].position(), caches[0].context_len()), (3, 3));
    assert_eq!(states[0].values(), snapshot[0]);
    assert_eq!(caches[1].context_len(), 9);
}

/// The paged KV and the state move together: a span appends its rows, a
/// refused forward and an aborted quantum leave both exactly as they were,
/// and the state survives the tensor arena bit for bit.
#[test]
fn the_kv_and_the_state_move_together_and_round_trip() {
    use crate::backends::state::{
        negotiate_state_plan, PhysicalStateSequenceId, PhysicalStateTransactionId,
        StateBackendPlanRequest, TensorStateArena, TensorStateCapacity,
    };
    use crate::backends::BackendKind;
    use crate::kv::v2::{
        InferenceStateContract, StateGroupId, StateGroupSpec, CURRENT_INFERENCE_STATE_ABI,
    };
    use std::sync::Arc;

    let synthetic = SyntheticCheckpoint::write(16);
    let config = &synthetic.config;
    let trunk = load(&synthetic, FishDrama3CpuPrecision::Q8_0);
    let embedding = FishDrama3Embedding::load(&synthetic.open(), config).unwrap();
    let embeds = embedding.embed_tensor(&rows(config), &Device::Cpu).unwrap();
    let mut cache = caches(trunk.attention_layers(), config, 1).remove(0);
    let mut state = trunk.new_state().unwrap();
    assert_eq!(state.bytes(), 6 * (4 * 32 * 32 + 3 * 256) * 4);
    trunk
        .forward_span(&embeds.narrow(0, 0, 6).unwrap(), 0, &mut state, &mut cache)
        .unwrap();
    assert_eq!((cache.context_len(), state.position()), (6, 6));
    let committed = state.values();

    // Refused before anything moves: a wrong start, the wrong width.
    assert!(trunk
        .forward_span(&embeds.narrow(0, 6, 2).unwrap(), 5, &mut state, &mut cache)
        .is_err());
    assert!(trunk
        .forward_span(&embeds.narrow(1, 0, 64).unwrap(), 6, &mut state, &mut cache)
        .is_err());
    assert_eq!((cache.context_len(), state.position()), (6, 6));
    assert_eq!(state.values(), committed);

    // An aborted quantum rewinds a span and a step.
    let checkpoint = state.begin_quantum(&cache).unwrap();
    trunk
        .forward_span(&embeds.narrow(0, 6, 4).unwrap(), 6, &mut state, &mut cache)
        .unwrap();
    let mut row = [FishDrama3StepRow {
        state: &mut state,
        cache: &mut cache,
    }];
    let hidden_first = host(
        &trunk
            .forward_step(&embeds.narrow(0, 10, 1).unwrap(), &mut row)
            .unwrap(),
    );
    assert_eq!((cache.context_len(), state.position()), (11, 11));
    state.abort_quantum(&mut cache, checkpoint).unwrap();
    assert_eq!((cache.context_len(), state.position()), (6, 6));
    assert_eq!(state.values(), committed);
    // Replaying the quantum gives the same bits, and commits.
    let checkpoint = state.begin_quantum(&cache).unwrap();
    trunk
        .forward_span(&embeds.narrow(0, 6, 4).unwrap(), 6, &mut state, &mut cache)
        .unwrap();
    let mut row = [FishDrama3StepRow {
        state: &mut state,
        cache: &mut cache,
    }];
    let hidden_again = host(
        &trunk
            .forward_step(&embeds.narrow(0, 10, 1).unwrap(), &mut row)
            .unwrap(),
    );
    assert_eq!(hidden_again, hidden_first);
    state.commit_quantum(&cache, &checkpoint).unwrap();

    // The tensor arena: stage, commit, restore into a fresh state.
    let [recurrent, convolution] = state::tensor_state_domains(&config.trunk).unwrap();
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: vec![recurrent, convolution],
        groups: vec![StateGroupSpec {
            id: StateGroupId::new(1),
            domains: vec![
                state::RECURRENT_STATE_DOMAIN,
                state::CONVOLUTION_STATE_DOMAIN,
            ],
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
    let capacity = TensorStateCapacity::for_plan(&plan, 2, 2).unwrap();
    assert_eq!(capacity.per_sequence_bytes(), state.bytes() as u64);
    let arena =
        TensorStateArena::new_with_contract(Arc::new(plan), &contract, capacity, Device::Cpu)
            .unwrap();
    let sequence = PhysicalStateSequenceId::new(7).unwrap();
    arena.register(sequence).unwrap();
    state.bind_tensor_sequence(7).unwrap();
    let saved = state.values();
    let position = state.position();
    let transaction = PhysicalStateTransactionId::new(1).unwrap();
    arena.begin(transaction, sequence).unwrap();
    state.stage_tensor_state(&arena, transaction.get()).unwrap();
    assert!(state.is_drained());
    // A drained state refuses to run.
    assert!(trunk
        .forward_span(
            &embeds.narrow(0, 11, 1).unwrap(),
            position,
            &mut state,
            &mut cache
        )
        .is_err());
    arena.commit(transaction, position as u64).unwrap();
    state.restore_tensor_state(&arena).unwrap();
    assert_eq!(state.values(), saved);
    assert_eq!(state.position(), position);
    let mut fresh = trunk.new_state().unwrap();
    fresh.bind_tensor_sequence(7).unwrap();
    fresh.restore_tensor_state(&arena).unwrap();
    assert_eq!(fresh.values(), saved);
    assert_eq!(fresh.position(), position);
    assert!(fresh.bind_tensor_sequence(8).is_err());
    // The restored state continues exactly as the original would.
    let mut copy = cache_copy(&trunk, config, &embeds, position);
    let a = host(
        &trunk
            .forward_span(
                &embeds.narrow(0, position, 3).unwrap(),
                position,
                &mut fresh,
                &mut copy,
            )
            .unwrap(),
    );
    let b = host(
        &trunk
            .forward_span(
                &embeds.narrow(0, position, 3).unwrap(),
                position,
                &mut state,
                &mut cache,
            )
            .unwrap(),
    );
    assert_eq!(a, b);
}

/// A cache holding rows `0..length` exactly as the round-trip test built
/// them (the same spans and step).
fn cache_copy(
    trunk: &FishDrama3Trunk,
    config: &FishDrama3Config,
    embeds: &Tensor,
    length: usize,
) -> PhysicalPagedKvCache {
    let mut cache = caches(trunk.attention_layers(), config, 1).remove(0);
    let mut state = trunk.new_state().unwrap();
    trunk
        .forward_span(&embeds.narrow(0, 0, 6).unwrap(), 0, &mut state, &mut cache)
        .unwrap();
    trunk
        .forward_span(&embeds.narrow(0, 6, 4).unwrap(), 6, &mut state, &mut cache)
        .unwrap();
    let mut row = [FishDrama3StepRow {
        state: &mut state,
        cache: &mut cache,
    }];
    trunk
        .forward_step(&embeds.narrow(0, 10, 1).unwrap(), &mut row)
        .unwrap();
    assert_eq!(cache.context_len(), length);
    cache
}

/// The self-check keeps a class F32 when its Q8_0 form fails (here the
/// shared experts' block scales underflow F16) and counts it; healthy
/// classes load Q8_0.
#[test]
fn a_class_failing_its_q8_0_check_stays_f32_and_is_counted() {
    let healthy = SyntheticCheckpoint::write(17);
    let trunk = load(&healthy, FishDrama3CpuPrecision::Q8_0);
    let q8_0_bytes = trunk.resident_bytes();
    let report = trunk.precision_report();
    assert_eq!(report.q8_fallback_classes, 0);
    assert!(report
        .forms
        .iter()
        .all(|(_, form)| *form == FishDrama3WeightForm::Q8_0));
    assert_eq!(report.checks.len(), 3 + 1 + 2 + 1 + 3 + 6);
    assert!(
        report
            .checks
            .iter()
            .all(|c| c.relative_l2 < Q8_0_TOLERANCE / 2.0),
        "{:?}",
        report.checks
    );
    let summary = trunk.diagnostics_summary();
    assert_eq!(summary["precision"]["q8_0_fallback_classes"], 0);

    let tiny = SyntheticCheckpoint::write_with(17, |tensors| {
        for (name, _, _, bytes) in tensors.iter_mut() {
            if name.contains(".shared_expert.") {
                for pair in bytes.as_chunks_mut::<2>().0 {
                    let v = half::bf16::from_le_bytes(*pair).to_f32() * 1e-5;
                    pair.copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
                }
            }
        }
    });
    let trunk = load(&tiny, FishDrama3CpuPrecision::Q8_0);
    let report = trunk.precision_report();
    assert_eq!(report.q8_fallback_classes, 1);
    for (class, form) in &report.forms {
        let expected = if *class == FishDrama3ProjectionClass::SharedExpert {
            FishDrama3WeightForm::F32
        } else {
            FishDrama3WeightForm::Q8_0
        };
        assert_eq!(*form, expected, "{class:?}");
    }
    assert_eq!(
        trunk.host_layers()[0].moe.shared_expert().projections()[0].form(),
        FishDrama3WeightForm::F32
    );
    assert_eq!(
        trunk.diagnostics_summary()["precision"]["q8_0_fallback_classes"],
        1
    );
    // Everything F32 when asked, with no checks.
    let exact = load(&healthy, FishDrama3CpuPrecision::F32);
    assert!(exact.precision_report().checks.is_empty());
    assert!(exact.resident_bytes() > 3 * q8_0_bytes);
}
