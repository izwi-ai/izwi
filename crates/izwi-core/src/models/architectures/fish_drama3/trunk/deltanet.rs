//! Gated DeltaNet (30 of the 40 trunk layers), the F32 CPU reference.
//!
//! ```text
//! qkv   = silu(causal depthwise conv_k4(history ++ in_proj_qkv(x)))   # 3-column history
//! q, k  = l2norm(q), l2norm(k) per key head (eps 1e-6); q *= key_dim^-1/2
//! beta  = sigmoid(in_proj_b(x));  g = -exp(A_log) * softplus(in_proj_a(x) + dt_bias)
//! S_v  *= exp(g);  delta = (v - Sᵀk) * beta;  S += k ⊗ delta;  o = Sᵀq   # per value head
//! out   = out_proj(w * rmsnorm(o) * silu(z))                           # plain w, not 1 + w
//! ```
//!
//! Value head `j` reads key head `j / (value_heads / key_heads)` (grouped,
//! upstream's `repeat_interleave`), never `j % key_heads`. The state is F32
//! `[value_heads, key_dim, value_dim]`. A step runs the recurrence; a span
//! runs upstream's chunked form (chunk 64), the same maths regrouped.

use candle_core::{Device, Tensor};
use rayon::prelude::*;

use super::super::checkpoint::{trunk_layer, FishDrama3Checkpoint};
use super::super::config::{FishDrama3LinearAttentionConfig, FishDrama3TrunkConfig};
use super::super::linear::{FishDrama3KernelPhase, FishDrama3Linear};
use super::super::residency::{
    FishDrama3ClassForms, FishDrama3ProjectionClass, FishDrama3WeightForm,
};
use super::state::FishDrama3LinearState;
use super::{load_linear, norm_gain, rms_norm_rows, sigmoid, silu, softplus, NormGain};
use crate::error::{Error, Result};

/// The chunk of upstream's chunked prefill form.
pub const GDN_CHUNK: usize = 64;
/// The q/k L2 norm's epsilon (upstream's `l2norm` default, not the config's).
pub const L2_NORM_EPS: f32 = 1e-6;

/// The head layout of one delta rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeltaGeometry {
    pub(crate) key_heads: usize,
    pub(crate) value_heads: usize,
    pub(crate) key_dim: usize,
    pub(crate) value_dim: usize,
}

impl DeltaGeometry {
    pub(crate) fn of(config: &FishDrama3LinearAttentionConfig) -> Self {
        Self {
            key_heads: config.key_heads,
            value_heads: config.value_heads,
            key_dim: config.key_head_dim,
            value_dim: config.value_head_dim,
        }
    }

    /// The key head value head `head` reads: grouped.
    pub(crate) fn key_head(&self, head: usize) -> usize {
        head / (self.value_heads / self.key_heads)
    }

    pub(crate) fn state_len(&self) -> usize {
        self.value_heads * self.key_dim * self.value_dim
    }
}

/// One value head's inputs over `n` rows.
struct HeadInputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    g: Vec<f32>,
    beta: Vec<f32>,
}

fn head_inputs(
    geo: &DeltaGeometry,
    head: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    n: usize,
) -> HeadInputs {
    let (dk, dv) = (geo.key_dim, geo.value_dim);
    let kh = geo.key_head(head);
    let rows = |x: &[f32], heads: usize, h: usize, d: usize| {
        (0..n)
            .flat_map(|t| {
                x[(t * heads + h) * d..(t * heads + h + 1) * d]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>()
    };
    HeadInputs {
        q: rows(q, geo.key_heads, kh, dk),
        k: rows(k, geo.key_heads, kh, dk),
        v: rows(v, geo.value_heads, head, dv),
        g: (0..n).map(|t| g[t * geo.value_heads + head]).collect(),
        beta: (0..n).map(|t| beta[t * geo.value_heads + head]).collect(),
    }
}

/// One value head, token by token; `s` is its `[key_dim, value_dim]` state.
fn head_recurrent(x: &HeadInputs, n: usize, dk: usize, dv: usize, s: &mut [f32]) -> Vec<f32> {
    let mut out = vec![0f32; n * dv];
    let mut mem = vec![0f32; dv];
    for t in 0..n {
        let (q, k, v) = (
            &x.q[t * dk..(t + 1) * dk],
            &x.k[t * dk..(t + 1) * dk],
            &x.v[t * dv..(t + 1) * dv],
        );
        let decay = x.g[t].exp();
        s.iter_mut().for_each(|value| *value *= decay);
        mem.fill(0.0);
        for (ki, row) in k.iter().zip(s.chunks_exact(dv)) {
            for (m, value) in mem.iter_mut().zip(row) {
                *m += value * ki;
            }
        }
        for (m, value) in mem.iter_mut().zip(v) {
            *m = (value - *m) * x.beta[t];
        }
        for (ki, row) in k.iter().zip(s.chunks_exact_mut(dv)) {
            for (value, delta) in row.iter_mut().zip(&mem) {
                *value += ki * delta;
            }
        }
        let o = &mut out[t * dv..(t + 1) * dv];
        for (qi, row) in q.iter().zip(s.chunks_exact(dv)) {
            for (value, state) in o.iter_mut().zip(row) {
                *value += state * qi;
            }
        }
    }
    out
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// One value head in chunks of `chunk` rows (upstream's
/// `chunk_gated_delta_rule`): within a chunk the delta rule is a unit
/// lower-triangular system `(I + A) U = beta v`, solved by forward
/// substitution, and the state crosses chunk boundaries once.
fn head_chunked(
    x: &HeadInputs,
    n: usize,
    dk: usize,
    dv: usize,
    chunk: usize,
    s: &mut [f32],
) -> Vec<f32> {
    let mut out = vec![0f32; n * dv];
    for start in (0..n).step_by(chunk) {
        let len = chunk.min(n - start);
        let rows = start..start + len;
        let q = &x.q[rows.start * dk..rows.end * dk];
        let k = &x.k[rows.start * dk..rows.end * dk];
        let v = &x.v[rows.start * dv..rows.end * dv];
        let beta = &x.beta[rows.clone()];
        // Cumulative log decay within the chunk.
        let mut c = Vec::with_capacity(len);
        let mut acc = 0f32;
        for g in &x.g[rows] {
            acc += g;
            c.push(acc);
        }
        let kb = k
            .chunks_exact(dk)
            .zip(beta)
            .flat_map(|(k, b)| k.iter().map(move |v| v * b))
            .collect::<Vec<_>>();
        // A[i][j] = (beta_i k_i · k_j) exp(c_i - c_j), strictly below the
        // diagonal.
        let mut a = vec![0f32; len * len];
        for i in 0..len {
            for j in 0..i {
                a[i * len + j] =
                    dot(&kb[i * dk..(i + 1) * dk], &k[j * dk..(j + 1) * dk]) * (c[i] - c[j]).exp();
            }
        }
        // U = (I + A)^-1 (beta v), W = (I + A)^-1 (beta k exp(c)).
        let mut u = vec![0f32; len * dv];
        let mut w = vec![0f32; len * dk];
        for i in 0..len {
            let decay = c[i].exp();
            for d in 0..dv {
                u[i * dv + d] = v[i * dv + d] * beta[i];
            }
            for d in 0..dk {
                w[i * dk + d] = kb[i * dk + d] * decay;
            }
            for j in 0..i {
                let aij = a[i * len + j];
                for d in 0..dv {
                    u[i * dv + d] -= aij * u[j * dv + d];
                }
                for d in 0..dk {
                    w[i * dk + d] -= aij * w[j * dk + d];
                }
            }
        }
        // corrected = U - W S.
        let mut corrected = u;
        for i in 0..len {
            for (d, wi) in w[i * dk..(i + 1) * dk].iter().enumerate() {
                for (value, state) in corrected[i * dv..(i + 1) * dv]
                    .iter_mut()
                    .zip(&s[d * dv..(d + 1) * dv])
                {
                    *value -= wi * state;
                }
            }
        }
        // o_i = (q_i exp(c_i)) S + sum_{j <= i} (q_i · k_j) exp(c_i - c_j) corrected_j.
        for i in 0..len {
            let o = &mut out[(start + i) * dv..(start + i + 1) * dv];
            let decay = c[i].exp();
            for (d, qi) in q[i * dk..(i + 1) * dk].iter().enumerate() {
                let scaled = qi * decay;
                for (value, state) in o.iter_mut().zip(&s[d * dv..(d + 1) * dv]) {
                    *value += scaled * state;
                }
            }
            for j in 0..=i {
                let intra =
                    dot(&q[i * dk..(i + 1) * dk], &k[j * dk..(j + 1) * dk]) * (c[i] - c[j]).exp();
                for (value, corr) in o.iter_mut().zip(&corrected[j * dv..(j + 1) * dv]) {
                    *value += intra * corr;
                }
            }
        }
        // S = S exp(c_last) + sum_j (k_j exp(c_last - c_j))ᵀ corrected_j.
        let last = c[len - 1];
        let chunk_decay = last.exp();
        s.iter_mut().for_each(|value| *value *= chunk_decay);
        for j in 0..len {
            let decay = (last - c[j]).exp();
            for (d, kj) in k[j * dk..(j + 1) * dk].iter().enumerate() {
                let scaled = kj * decay;
                for (state, corr) in s[d * dv..(d + 1) * dv]
                    .iter_mut()
                    .zip(&corrected[j * dv..(j + 1) * dv])
                {
                    *state += scaled * corr;
                }
            }
        }
    }
    out
}

/// The delta rule over `n` rows: `q`, `k` `[n, key_heads, key_dim]` (L2
/// normed, q scaled), `v` `[n, value_heads, value_dim]`, `g` and `beta`
/// `[n, value_heads]`. Updates `state` and returns `[n, value_heads,
/// value_dim]`. `chunk` selects the chunked form; `None` the recurrence.
pub(crate) fn delta_rule(
    geo: &DeltaGeometry,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    n: usize,
    state: &mut [f32],
    chunk: Option<usize>,
) -> Vec<f32> {
    let (dk, dv, hv) = (geo.key_dim, geo.value_dim, geo.value_heads);
    let heads = state
        .par_chunks_mut(dk * dv)
        .enumerate()
        .map(|(head, s)| {
            let inputs = head_inputs(geo, head, q, k, v, g, beta, n);
            match chunk {
                Some(chunk) => head_chunked(&inputs, n, dk, dv, chunk.max(1), s),
                None => head_recurrent(&inputs, n, dk, dv, s),
            }
        })
        .collect::<Vec<_>>();
    let mut out = vec![0f32; n * hv * dv];
    for (head, values) in heads.iter().enumerate() {
        for t in 0..n {
            out[(t * hv + head) * dv..(t * hv + head + 1) * dv]
                .copy_from_slice(&values[t * dv..(t + 1) * dv]);
        }
    }
    out
}

/// `x / sqrt(sum(x²) + eps)` per row of `width`, in place.
pub(crate) fn l2_normalize(x: &mut [f32], width: usize, eps: f32) {
    for row in x.chunks_exact_mut(width) {
        let inverse = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
        row.iter_mut().for_each(|v| *v *= inverse);
    }
}

pub(crate) struct FishDrama3DeltaNet {
    in_proj_qkv: FishDrama3Linear,
    in_proj_z: FishDrama3Linear,
    in_proj_a: FishDrama3Linear,
    in_proj_b: FishDrama3Linear,
    /// `[channels, kernel]`: tap 0 multiplies the oldest column.
    conv: Vec<f32>,
    /// `-exp(A_log)`.
    neg_a: Vec<f32>,
    dt_bias: Vec<f32>,
    /// The output norm's plain gain.
    norm: Vec<f32>,
    out_proj: FishDrama3Linear,
    config: FishDrama3LinearAttentionConfig,
    eps: f32,
    device: Device,
}

impl FishDrama3DeltaNet {
    pub(crate) fn load(
        checkpoint: &FishDrama3Checkpoint,
        layer: usize,
        trunk: &FishDrama3TrunkConfig,
        forms: &FishDrama3ClassForms,
        device: &Device,
    ) -> Result<Self> {
        let la = trunk.linear;
        let h = trunk.hidden;
        let x = format!("{}.linear_attn", trunk_layer(layer));
        let input = forms.form(FishDrama3ProjectionClass::DeltaNetIn);
        let output = forms.form(FishDrama3ProjectionClass::DeltaNetOut);
        let channels = la.conv_channels();
        let vector = |name: &str| checkpoint.bf16_to_f32(&format!("{x}.{name}"), &[la.value_heads]);
        let neg_a = vector("A_log")?.into_iter().map(|v| -v.exp()).collect();
        Ok(Self {
            in_proj_qkv: load_linear(
                checkpoint,
                &format!("{x}.in_proj_qkv.weight"),
                channels,
                h,
                input,
                device,
            )?,
            in_proj_z: load_linear(
                checkpoint,
                &format!("{x}.in_proj_z.weight"),
                la.value_width(),
                h,
                input,
                device,
            )?,
            // The gate projections stay F32 beside the router (residency.rs).
            in_proj_a: load_linear(
                checkpoint,
                &format!("{x}.in_proj_a.weight"),
                la.value_heads,
                h,
                FishDrama3WeightForm::F32,
                device,
            )?,
            in_proj_b: load_linear(
                checkpoint,
                &format!("{x}.in_proj_b.weight"),
                la.value_heads,
                h,
                FishDrama3WeightForm::F32,
                device,
            )?,
            conv: checkpoint.bf16_to_f32(
                &format!("{x}.conv1d.weight"),
                &[channels, 1, la.conv_kernel],
            )?,
            neg_a,
            dt_bias: vector("dt_bias")?,
            norm: norm_gain(
                checkpoint,
                &format!("{x}.norm.weight"),
                la.value_head_dim,
                NormGain::Plain,
            )?,
            out_proj: load_linear(
                checkpoint,
                &format!("{x}.out_proj.weight"),
                h,
                la.value_width(),
                output,
                device,
            )?,
            config: la,
            eps: trunk.rms_norm_eps as f32,
            device: device.clone(),
        })
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.projections()
            .iter()
            .map(|l| l.resident_bytes())
            .sum::<usize>()
            + (self.conv.len() + self.neg_a.len() + self.dt_bias.len() + self.norm.len()) * 4
    }

    pub(crate) fn projections(&self) -> [&FishDrama3Linear; 5] {
        [
            &self.in_proj_qkv,
            &self.in_proj_z,
            &self.in_proj_a,
            &self.in_proj_b,
            &self.out_proj,
        ]
    }

    /// The mixer output `[n, hidden]` of the normed rows `h`, continuing
    /// `state` (updated in place). A step advances the conv ring by one
    /// column and runs the recurrence; a span replaces the history and runs
    /// the chunked form.
    pub(crate) fn forward(
        &self,
        h: &[f32],
        n: usize,
        state: &mut FishDrama3LinearState,
        phase: FishDrama3KernelPhase,
    ) -> Result<Vec<f32>> {
        let la = &self.config;
        let geo = DeltaGeometry::of(la);
        let channels = la.conv_channels();
        let taps = la.conv_kernel;
        let history = taps - 1;
        if phase == FishDrama3KernelPhase::Step && n != 1 {
            return Err(Error::InferenceError(format!(
                "a DeltaNet step takes one row, not {n}"
            )));
        }
        let mixed = self.in_proj_qkv.forward_host(h, n, phase)?;
        let z = self.in_proj_z.forward_host(h, n, phase)?;
        let b = self.in_proj_b.forward_host(h, n, phase)?;
        let a = self.in_proj_a.forward_host(h, n, phase)?;

        // Causal depthwise conv over the history and the new columns.
        let mut padded = state.conv.ordered_host()?;
        if padded.len() != history * channels {
            return Err(Error::InferenceError(
                "the DeltaNet conv history does not match the layer".into(),
            ));
        }
        padded.extend_from_slice(&mixed);
        let mut conv = vec![0f32; n * channels];
        for t in 0..n {
            let out = &mut conv[t * channels..(t + 1) * channels];
            for (c, value) in out.iter_mut().enumerate() {
                let w = &self.conv[c * taps..(c + 1) * taps];
                let mut sum = 0f32;
                for (j, wj) in w.iter().enumerate() {
                    sum += wj * padded[(t + j) * channels + c];
                }
                *value = silu(sum);
            }
        }
        match phase {
            FishDrama3KernelPhase::Step => state.conv.push(
                Tensor::from_slice(&mixed, channels, &self.device)?,
                channels,
            )?,
            FishDrama3KernelPhase::Span => {
                state
                    .conv
                    .replace(&padded[n * channels..], channels, &self.device)?
            }
        }

        let (kw, vw) = (la.key_width(), la.value_width());
        let mut q = Vec::with_capacity(n * kw);
        let mut k = Vec::with_capacity(n * kw);
        let mut v = Vec::with_capacity(n * vw);
        for row in conv.chunks_exact(channels) {
            q.extend_from_slice(&row[..kw]);
            k.extend_from_slice(&row[kw..2 * kw]);
            v.extend_from_slice(&row[2 * kw..]);
        }
        l2_normalize(&mut q, la.key_head_dim, L2_NORM_EPS);
        l2_normalize(&mut k, la.key_head_dim, L2_NORM_EPS);
        let scale = (1.0 / (la.key_head_dim as f64).sqrt()) as f32;
        q.iter_mut().for_each(|value| *value *= scale);
        let beta = b.iter().map(|v| sigmoid(*v)).collect::<Vec<_>>();
        let g = a
            .chunks_exact(la.value_heads)
            .flat_map(|row| {
                row.iter()
                    .zip(&self.neg_a)
                    .zip(&self.dt_bias)
                    .map(|((a, neg_a), bias)| neg_a * softplus(a + bias))
            })
            .collect::<Vec<_>>();

        let mut s = state.recurrent_host()?;
        if s.len() != geo.state_len() {
            return Err(Error::InferenceError(
                "the DeltaNet recurrent state does not match the layer".into(),
            ));
        }
        let chunk = (phase == FishDrama3KernelPhase::Span).then_some(GDN_CHUNK);
        let core = delta_rule(&geo, &q, &k, &v, &g, &beta, n, &mut s, chunk);
        state.set_recurrent(s, &geo, &self.device)?;

        // Gated RMSNorm: w * rmsnorm(core) * silu(z), per value head.
        let mut gated = rms_norm_rows(&core, la.value_head_dim, &self.norm, self.eps);
        for (value, z) in gated.iter_mut().zip(&z) {
            *value *= silu(*z);
        }
        self.out_proj.forward_host(&gated, n, phase)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::numeric_check::{relative_l2, xorshift_values};

    /// Delta-rule inputs as the layer feeds them: q and k L2 normed (q
    /// scaled), beta in (0, 1), log decays in [-0.6, 0).
    fn inputs(
        geo: &DeltaGeometry,
        n: usize,
        seed: u64,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let kw = geo.key_heads * geo.key_dim;
        let mut q = xorshift_values(n * kw, seed, 1.0);
        let mut k = xorshift_values(n * kw, seed + 1, 1.0);
        l2_normalize(&mut q, geo.key_dim, L2_NORM_EPS);
        l2_normalize(&mut k, geo.key_dim, L2_NORM_EPS);
        let scale = 1.0 / (geo.key_dim as f32).sqrt();
        q.iter_mut().for_each(|v| *v *= scale);
        let v = xorshift_values(n * geo.value_heads * geo.value_dim, seed + 2, 1.0);
        let g = xorshift_values(n * geo.value_heads, seed + 3, 0.3)
            .into_iter()
            .map(|x| x - 0.3)
            .collect();
        let beta = xorshift_values(n * geo.value_heads, seed + 4, 3.0)
            .into_iter()
            .map(sigmoid)
            .collect();
        (q, k, v, g, beta)
    }

    /// The F64 recurrence with an explicit key head per value head.
    fn reference(
        geo: &DeltaGeometry,
        key_head: impl Fn(usize) -> usize,
        (q, k, v, g, beta): &(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>),
        n: usize,
        state: &mut [f64],
    ) -> Vec<f64> {
        let (hk, hv, dk, dv) = (geo.key_heads, geo.value_heads, geo.key_dim, geo.value_dim);
        let mut out = vec![0f64; n * hv * dv];
        for t in 0..n {
            for h in 0..hv {
                let kh = key_head(h);
                let qs = &q[(t * hk + kh) * dk..][..dk];
                let ks = &k[(t * hk + kh) * dk..][..dk];
                let vs = &v[(t * hv + h) * dv..][..dv];
                let s = &mut state[h * dk * dv..(h + 1) * dk * dv];
                let decay = f64::from(g[t * hv + h]).exp();
                s.iter_mut().for_each(|x| *x *= decay);
                for j in 0..dv {
                    let mem = (0..dk)
                        .map(|i| s[i * dv + j] * f64::from(ks[i]))
                        .sum::<f64>();
                    let delta = (f64::from(vs[j]) - mem) * f64::from(beta[t * hv + h]);
                    for i in 0..dk {
                        s[i * dv + j] += f64::from(ks[i]) * delta;
                    }
                }
                for j in 0..dv {
                    out[(t * hv + h) * dv + j] =
                        (0..dk).map(|i| s[i * dv + j] * f64::from(qs[i])).sum();
                }
            }
        }
        out
    }

    fn f32s(x: &[f64]) -> Vec<f32> {
        x.iter().map(|v| *v as f32).collect()
    }

    /// The recurrence and the chunked form agree at the chunk's edges and
    /// across several chunks, from a non-zero state, and both match F64.
    #[test]
    fn the_recurrence_and_the_chunked_form_are_the_same_maths() {
        let geo = DeltaGeometry {
            key_heads: 2,
            value_heads: 4,
            key_dim: 32,
            value_dim: 32,
        };
        for n in [1usize, 63, 64, 65, 300] {
            let x = inputs(&geo, n, 7 + n as u64);
            let initial = xorshift_values(geo.state_len(), 99, 0.5);
            let mut s_rec = initial.clone();
            let mut s_chunk = initial.clone();
            let rec = delta_rule(&geo, &x.0, &x.1, &x.2, &x.3, &x.4, n, &mut s_rec, None);
            let chunk = delta_rule(
                &geo,
                &x.0,
                &x.1,
                &x.2,
                &x.3,
                &x.4,
                n,
                &mut s_chunk,
                Some(GDN_CHUNK),
            );
            let mut s_ref = initial.iter().map(|v| f64::from(*v)).collect::<Vec<_>>();
            let expected = reference(&geo, |h| geo.key_head(h), &x, n, &mut s_ref);
            for (name, out, state) in [("recurrent", &rec, &s_rec), ("chunked", &chunk, &s_chunk)] {
                let out_error = relative_l2(out, &f32s(&expected));
                let state_error = relative_l2(state, &f32s(&s_ref));
                assert!(out_error < 2e-5, "{name} n={n}: output {out_error}");
                assert!(state_error < 2e-5, "{name} n={n}: state {state_error}");
            }
            let between = relative_l2(&chunk, &rec);
            assert!(between < 2e-5, "n={n}: {between}");
        }
    }

    /// Value head `j` reads key head `j / 2`; the tiled pairing (`j % 2`)
    /// is far off.
    #[test]
    fn value_heads_are_grouped_over_key_heads() {
        let geo = DeltaGeometry {
            key_heads: 2,
            value_heads: 4,
            key_dim: 32,
            value_dim: 32,
        };
        assert_eq!(
            (0..4).map(|h| geo.key_head(h)).collect::<Vec<_>>(),
            [0, 0, 1, 1]
        );
        let n = 20;
        let x = inputs(&geo, n, 3);
        let mut state = vec![0f32; geo.state_len()];
        let ours = delta_rule(&geo, &x.0, &x.1, &x.2, &x.3, &x.4, n, &mut state, None);
        let mut s = vec![0f64; geo.state_len()];
        let grouped = reference(&geo, |h| h / 2, &x, n, &mut s);
        let mut s = vec![0f64; geo.state_len()];
        let tiled = reference(&geo, |h| h % 2, &x, n, &mut s);
        let error = relative_l2(&ours, &f32s(&grouped));
        let red = relative_l2(&ours, &f32s(&tiled));
        assert!(error < 1e-5, "{error}");
        assert!(red > 0.2, "{red}");
    }
}
