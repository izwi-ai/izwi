//! The device column (§6): one implementation of the trunk, the tower and
//! resident sampling for Metal and CUDA, in each backend's forms.
//!
//! | Part | Metal | CUDA (SM80+) |
//! |---|---|---|
//! | Residual stream | F32 | BF16 (upstream's) |
//! | Dense projections (trunk, shared expert, tower, heads) | F16 weights and activations | BF16 weights and activations |
//! | Packing at load | trunk `q_proj` / `k_proj` / `v_proj` and `in_proj_qkv` / `in_proj_z` as one matrix each; the tower's contiguous `wqkv` repacked per head like the trunk's | the same |
//! | Step projections | one GEMV per row | the row-invariant multi-row GEMV (`kernels::cuda::row_gemv`), once its device check passed; else one GEMV per row |
//! | Span projections | one tile GEMM | one tile GEMM (cuBLAS) |
//! | Routed experts | raw E4M3 + F32 128×128 scales stacked per layer; the fused per-(token, slot) kernels in a step, the grouped kernels in a span of 32 rows or more | the same; the grouped kernels' tensor-core tiles once their device check passed |
//! | Router, shared-expert gate, DeltaNet `a` / `b`, norm gains, conv, `A_log`, `dt_bias` | F32 | F32 |
//! | RMSNorm, q/k norm, attention gate | the Qwen3.6 F16 kernels with F32 gains | their BF16 forms |
//! | RoPE | host F64 tables rounded to F32, rotated in F32 (NeoX for the trunk, interleaved pairs for the tower) | the same |
//! | DeltaNet | `gdn_conv` + `gdn_decode` row by row in steps; a span's rows through their span forms (two launches, the state on chip from token to token, bit for bit the row-by-row result), F32 state | row by row in steps and short spans; a span of 32 rows or more runs the causal conv over the span and the chunked prefill (FLA's chunk rule), once its device check passed |
//! | Paged KV (both domains) | F16, F32 after a breached load-time probe | BF16 |
//! | Embedding and code tables | F16 device tables, gathered to F32 | BF16 device tables |
//! | Logits | F16 projections widened to F32 | BF16 projections into F32 |
//! | Sampling | on the device, one readback a frame on the fused path, two on the composed one (`sampling.rs`) | the same |
//!
//! Each backend keeps its own load-time checks: `metal::self_check` (the
//! F16 range probe, FP8, RoPE) and `cuda::self_check` (FP8, RoPE, and the
//! device checks of the row-invariant GEMV, the chunked DeltaNet and the
//! grouped MoE tiles). The table checks both run are in `checks.rs`.
//!
//! **Kernel phases (§7.4).** A step's rows run together layer by layer, but
//! no reduction ever spans two rows: each projection runs one GEMV per row
//! or the row-invariant multi-row GEMV (one fixed reduction order per row,
//! whatever the row count), attention reads each request's own paged view
//! one request at a time, the DeltaNet kernels each row's own state, and
//! the fused MoE and norm kernels compute every token on its own. So a
//! row's bits never depend on its batch. A span (one request's prefill)
//! runs one GEMM per projection and, from 32 rows, the grouped expert
//! kernels (and on CUDA the chunked DeltaNet).

pub(crate) mod checks;
pub(crate) mod sampling;
#[cfg(all(test, any(feature = "metal", feature = "cuda")))]
pub(crate) mod test_support;
pub(crate) mod tower;
pub(crate) mod trunk;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::{DType, Device, Tensor};

use super::config::{FishDrama3AttentionConfig, FishDrama3Config, FishDrama3RopeLayout};
use super::linear::{FishDrama3KernelPhase, FishDrama3Linear};
use super::residency::{FishDrama3Residency, FishDrama3WeightForm};
use super::trunk::attention::FishDrama3Rope;
use crate::error::{Error, Result};
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};

/// Kill switch for the fused device forward (`0`, `off`, `false`): the
/// trunk and tower then run the eager per-layer chain and the draws the
/// composed race. Read at load.
pub const FUSED_DECODE_ENV: &str = "IZWI_FISH_DRAMA3_FUSED_DECODE";

/// Kill switch for the CUDA graphs of a fused trunk step's stretches (`0`,
/// `off`, `false`): every stretch runs eagerly. The global
/// `IZWI_CUDA_DECODE_GRAPHS=off` turns them off too. Read at load.
pub const SEGMENT_GRAPHS_ENV: &str = "IZWI_FISH_DRAMA3_SEGMENT_GRAPHS";

/// Largest relative L2 deviation of a device's fused kernels from their CPU
/// forms that keeps the fused forward on.
pub(crate) const FUSED_CHECK_LIMIT: f64 = 1e-3;

/// One RoPE's cos/sin tables on a device, `[positions, rotary / 2]` F32 each:
/// the family's F64 angles rounded once, exactly the rows the eager chain
/// uploads per call.
pub(crate) struct RopeTables {
    pub(crate) cos: Tensor,
    pub(crate) sin: Tensor,
}

/// The tables of `rope` for positions `0..positions` on `device`, shared by
/// every layer of a stack (one upload per load, not four per layer a step).
pub(crate) fn rope_tables(
    rope: &FishDrama3Rope,
    positions: usize,
    device: &Device,
) -> Result<Arc<RopeTables>> {
    type Key = (String, Vec<u32>, usize);
    static CACHE: OnceLock<Mutex<Vec<(Key, std::sync::Weak<RopeTables>)>>> = OnceLock::new();
    let half = rope.rotary_dims() / 2;
    let (c0, s0) = rope.cos_sin(1);
    let mut signature = c0
        .iter()
        .chain(&s0)
        .map(|v| v.to_bits())
        .collect::<Vec<_>>();
    signature.push(rope.rotary_dims() as u32);
    signature.push(u32::from(
        rope.layout() == FishDrama3RopeLayout::InterleavedPairs,
    ));
    let key = (format!("{:?}", device.location()), signature, positions);
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut entries = cache
        .lock()
        .map_err(|_| Error::InferenceError("the RoPE table cache is poisoned".into()))?;
    entries.retain(|(_, weak)| weak.strong_count() > 0);
    if let Some(tables) = entries
        .iter()
        .find(|(k, _)| *k == key)
        .and_then(|(_, weak)| weak.upgrade())
    {
        return Ok(tables);
    }
    let mut cos = Vec::with_capacity(positions * half);
    let mut sin = Vec::with_capacity(positions * half);
    for position in 0..positions {
        let (c, s) = rope.cos_sin(position);
        cos.extend(c);
        sin.extend(s);
    }
    let tables = Arc::new(RopeTables {
        cos: Tensor::from_vec(cos, (positions, half.max(1)), device)?,
        sin: Tensor::from_vec(sin, (positions, half.max(1)), device)?,
    });
    entries.push((key, Arc::downgrade(&tables)));
    Ok(tables)
}

/// Each row position of `rows`, `[n]` U32 on `device`: the one upload a
/// fused stack forward makes for its RoPE.
pub(crate) fn positions_tensor(rows: &[DeviceRow<'_>], device: &Device) -> Result<Tensor> {
    let at = positions(rows)
        .into_iter()
        .map(|p| p as u32)
        .collect::<Vec<_>>();
    let n = at.len();
    count_uploads(1);
    Ok(Tensor::from_vec(at, n, device)?)
}

/// How a device load's trunk and tower run, and why.
#[derive(Debug, Clone, PartialEq)]
pub enum FishDrama3DecodePath {
    /// Fused residual norms, q/k norm + RoPE and SwiGLU, and one-launch
    /// races, with the load-time kernel check's worst relative error.
    Fused { check: f64 },
    /// The eager per-layer chain: switched off, or the check failed.
    Eager { reason: String },
}

impl FishDrama3DecodePath {
    /// The path a load on `device` takes: fused unless [`FUSED_DECODE_ENV`]
    /// is off or the kernels miss their CPU forms on this device.
    pub(crate) fn select(device: &Device, config: &FishDrama3Config) -> Self {
        let path = if !crate::env::flag_or(FUSED_DECODE_ENV, true) {
            Self::Eager {
                reason: format!("{FUSED_DECODE_ENV} is off"),
            }
        } else {
            match fused_kernels_check(device, config) {
                Ok(check) if check <= FUSED_CHECK_LIMIT => Self::Fused { check },
                Ok(check) => Self::Eager {
                    reason: format!("the fused kernels deviate by {check:.3e}"),
                },
                Err(error) => Self::Eager {
                    reason: format!("the fused kernel check failed: {error}"),
                },
            }
        };
        if let Self::Eager { reason } = &path {
            tracing::warn!("Fish Audio Drama 3 runs its device forwards eagerly: {reason}");
        }
        path
    }

    pub fn is_fused(&self) -> bool {
        matches!(self, Self::Fused { .. })
    }

    pub fn diagnostics(&self) -> serde_json::Value {
        match self {
            Self::Fused { check } => serde_json::json!({ "path": "fused", "check": check }),
            Self::Eager { reason } => serde_json::json!({ "path": "eager", "reason": reason }),
        }
    }
}

/// The worst relative L2 deviation of the fused kernels on `device` from
/// their CPU forms, at the model's widths and dtype mixes: the residual
/// norms of both stacks, both attentions' q/k norm + RoPE at the first and
/// last table rows and both SwiGLUs; a device race that picks otherwise than
/// its CPU form is an error.
fn fused_kernels_check(device: &Device, config: &FishDrama3Config) -> Result<f64> {
    use crate::kernels::numeric_check::{host_f32, relative_l2, xorshift_tensor};
    use crate::kernels::race_sampler::{race_sample, RacePolicy};
    use crate::kernels::{qk_norm_rope::qk_norm_rope, residual_norm, swiglu::swiglu};
    let forms = DeviceForms::of(device)?;
    let cpu = Device::Cpu;
    let on = |t: &Tensor| t.to_device(device);
    let mut worst = 0f64;
    let mut track = |a: &Tensor, b: &Tensor| -> Result<()> {
        worst = worst.max(relative_l2(&host_f32(a)?, &host_f32(b)?));
        Ok(())
    };
    let rows = 2;
    let (t, w) = (&config.trunk, &config.tower);
    for (width, eps) in [(t.hidden, t.rms_norm_eps), (w.dim, w.norm_eps)] {
        let eps = eps as f32;
        let x = xorshift_tensor(&[rows, width], 1, 4.0, &cpu)?.to_dtype(forms.residual)?;
        let gain = (xorshift_tensor(&[width], 2, 0.3, &cpu)? + 1.0)?;
        // A mixer's delta in the activation dtype, the experts' in F32.
        for dtype in [forms.half, DType::F32] {
            let delta = xorshift_tensor(&[rows, width], 3, 2.0, &cpu)?.to_dtype(dtype)?;
            let (sum, normed) = residual_norm::add_rms_norm(&x, &delta, &gain, eps, forms.half)?;
            let (dsum, dnormed) =
                residual_norm::add_rms_norm(&on(&x)?, &on(&delta)?, &on(&gain)?, eps, forms.half)?;
            track(&dsum, &sum)?;
            track(&dnormed, &normed)?;
        }
        // The residual, the tower's F32 injections and its 16-bit hidden.
        for dtype in [forms.residual, DType::F32, forms.half] {
            let x = x.to_dtype(dtype)?;
            let normed = residual_norm::rms_norm(&x, &gain, eps, forms.half)?;
            track(
                &residual_norm::rms_norm(&on(&x)?, &on(&gain)?, eps, forms.half)?,
                &normed,
            )?;
        }
    }
    let positions = t.max_positions.min(super::model::MAX_REQUEST_POSITIONS);
    for (attention, eps) in [(&t.attention, t.rms_norm_eps), (&w.attention, w.norm_eps)] {
        let rope = FishDrama3Rope::new(attention);
        let spec = qk_norm_rope_spec(attention, &rope, eps as f32);
        let width = 2 * attention.q_width() + 2 * attention.kv_width();
        let packed = xorshift_tensor(&[rows, width], 4, 2.0, &cpu)?.to_dtype(forms.half)?;
        let d = attention.head_dim;
        let q_gain = (xorshift_tensor(&[d], 5, 0.3, &cpu)? + 1.0)?;
        let k_gain = (xorshift_tensor(&[d], 6, 0.3, &cpu)? + 1.0)?;
        let at = Tensor::new(&[0u32, positions as u32 - 1], &cpu)?;
        let host = rope_tables(&rope, positions, &cpu)?;
        let tables = rope_tables(&rope, positions, device)?;
        let want = qk_norm_rope(&packed, &q_gain, &k_gain, &host.cos, &host.sin, &at, spec)?;
        let got = qk_norm_rope(
            &on(&packed)?,
            &on(&q_gain)?,
            &on(&k_gain)?,
            &tables.cos,
            &tables.sin,
            &on(&at)?,
            spec,
        )?;
        track(&got.0, &want.0)?;
        track(&got.1, &want.1)?;
        track(&got.2, &want.2)?;
    }
    for width in [t.moe.shared_intermediate, w.intermediate] {
        let packed = xorshift_tensor(&[rows, 2 * width], 7, 6.0, &cpu)?.to_dtype(forms.half)?;
        track(
            &swiglu(&on(&packed)?, forms.half)?,
            &swiglu(&packed, forms.half)?,
        )?;
    }
    // The race picks what its CPU form picks, at both vocabularies, capped
    // and uncapped.
    let policies = [
        sampling::race_policy(&super::sampling::FishDrama3DrawPolicy::RAS),
        RacePolicy {
            temperature: 0.7,
            top_p: 0.8,
            top_k: None,
        },
    ];
    for vocab in [config.ids.compact_head_rows().len(), w.codebook_vocab] {
        let logits = xorshift_tensor(&[rows, vocab], 8, 6.0, &cpu)?;
        let noise = (xorshift_tensor(&[rows, vocab], 9, 2.0, &cpu)?.abs()? + 1e-3)?;
        for policy in policies {
            let want = race_sample(&logits, &noise, 1, policy)?;
            let got = race_sample(&on(&logits)?, &on(&noise)?, 1, policy)?;
            let (want, got) = (
                want.flatten_all()?.to_vec1::<u32>()?,
                got.flatten_all()?.to_device(&cpu)?.to_vec1::<u32>()?,
            );
            if got != want {
                return Err(Error::InferenceError(format!(
                    "the device race drew {got:?} where its CPU form drew {want:?}"
                )));
            }
        }
    }
    Ok(worst)
}

/// Where q, k and v sit in a packed `[q_h | gate_h]... | k | v` projection
/// row, and how `rope` rotates them.
fn qk_norm_rope_spec(
    config: &FishDrama3AttentionConfig,
    rope: &FishDrama3Rope,
    eps: f32,
) -> crate::kernels::qk_norm_rope::QkNormRopeSpec {
    use crate::kernels::qk_norm_rope::{QkNormRopeSpec, RopeLayout};
    let (qw, kvw, d) = (config.q_width(), config.kv_width(), config.head_dim);
    QkNormRopeSpec {
        heads: config.heads,
        kv_heads: config.kv_heads,
        head_dim: d,
        rotary_dims: rope.rotary_dims(),
        layout: match rope.layout() {
            FishDrama3RopeLayout::HalfSplit => RopeLayout::HalfSplit,
            FishDrama3RopeLayout::InterleavedPairs => RopeLayout::Interleaved,
        },
        q_offset: 0,
        q_stride: 2 * d,
        k_offset: 2 * qw,
        v_offset: 2 * qw + kvw,
        eps,
    }
}

/// What a device load holds and runs in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeviceForms {
    pub(crate) residency: FishDrama3Residency,
    /// Dense weights, tables and a healthy layer's activations: F16 on
    /// Metal, BF16 on CUDA.
    pub(crate) half: DType,
    /// The residual stream: F32 on Metal, BF16 on CUDA.
    pub(crate) residual: DType,
}

impl DeviceForms {
    /// The forms of `device`; a CUDA device below SM80 is refused.
    pub(crate) fn of(device: &Device) -> Result<Self> {
        match FishDrama3Residency::of(device) {
            FishDrama3Residency::Metal => Ok(Self {
                residency: FishDrama3Residency::Metal,
                half: DType::F16,
                residual: DType::F32,
            }),
            FishDrama3Residency::Cuda => {
                super::cuda::require_sm80(device)?;
                Ok(Self {
                    residency: FishDrama3Residency::Cuda,
                    half: DType::BF16,
                    residual: DType::BF16,
                })
            }
            FishDrama3Residency::Cpu => Err(Error::ModelLoadError(
                "the Fish Audio Drama 3 device forms run on Metal or CUDA".into(),
            )),
        }
    }

    /// The dense weights' form.
    pub(crate) fn weight_form(self) -> FishDrama3WeightForm {
        if self.half == DType::BF16 {
            FishDrama3WeightForm::Bf16
        } else {
            FishDrama3WeightForm::F16
        }
    }

    /// The activations of the host-routed expert loop: F32 on Metal; CUDA's
    /// FP8 GEMV takes 16-bit activations only.
    pub(crate) fn loop_activations(self) -> DType {
        match self.residency {
            FishDrama3Residency::Cuda => self.half,
            _ => DType::F32,
        }
    }
}

static STEP_ROW_GEMV_CALLS: AtomicU64 = AtomicU64::new(0);
static STEP_ROW_LOOP_CALLS: AtomicU64 = AtomicU64::new(0);
static CHUNKED_DELTANET_SPANS: AtomicU64 = AtomicU64::new(0);
static RECURRENT_DELTANET_SPANS: AtomicU64 = AtomicU64::new(0);
static GROUPED_MOE_SPANS: AtomicU64 = AtomicU64::new(0);

/// Process-wide count of CUDA step projections the row-invariant multi-row
/// GEMV ran (several rows, weights read once).
pub fn fish_drama3_step_row_gemv_calls() -> u64 {
    STEP_ROW_GEMV_CALLS.load(Ordering::Relaxed)
}

/// Process-wide count of CUDA step projections of several rows that ran
/// one GEMV per row instead (the multi-row GEMV unverified or unsupported).
pub fn fish_drama3_step_row_loop_calls() -> u64 {
    STEP_ROW_LOOP_CALLS.load(Ordering::Relaxed)
}

/// Process-wide count of DeltaNet span layers through the chunked prefill.
pub fn fish_drama3_chunked_deltanet_spans() -> u64 {
    CHUNKED_DELTANET_SPANS.load(Ordering::Relaxed)
}

/// Process-wide count of DeltaNet span layers (one request's rows) through
/// the span kernels that walk the tokens with the state on chip (Metal).
pub fn fish_drama3_recurrent_deltanet_spans() -> u64 {
    RECURRENT_DELTANET_SPANS.load(Ordering::Relaxed)
}

/// Process-wide count of MoE span layers through the grouped kernels.
pub fn fish_drama3_grouped_moe_spans() -> u64 {
    GROUPED_MOE_SPANS.load(Ordering::Relaxed)
}

static UPLOADS: AtomicU64 = AtomicU64::new(0);
static STEP_MAIN_DRAW_US: AtomicU64 = AtomicU64::new(0);
static STEP_TOWER_US: AtomicU64 = AtomicU64::new(0);
static STEP_TRUNK_US: AtomicU64 = AtomicU64::new(0);

/// The three parts of a decode step, timed on the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepSection {
    /// The main head, its ranks and races, and on the two-readback path
    /// readback 1 (on a GPU it also waits for the previous step's trunk);
    /// on the one-readback path RAS too, enqueued.
    MainDraw,
    /// The 13 tower layers and the 9 head draws, with readback 2 on the
    /// two-readback path.
    Tower,
    /// The frame embedding and the 40 trunk layers, enqueued; on the
    /// one-readback path also the frame's readback and commits.
    Trunk,
}

/// Process-wide count of host-to-device uploads on the decode step's hot
/// paths: RoPE tables, sampling scalars and noise, frame codes and labels,
/// embedding inputs.
pub fn fish_drama3_step_uploads() -> u64 {
    UPLOADS.load(Ordering::Relaxed)
}

/// The device column's process-wide fast-path counters, for diagnostics.
pub fn fast_path_counters() -> serde_json::Value {
    let ms = |counter: &AtomicU64| counter.load(Ordering::Relaxed) as f64 / 1e3;
    serde_json::json!({
        "step_row_gemv_calls_total": fish_drama3_step_row_gemv_calls(),
        "step_row_loop_calls_total": fish_drama3_step_row_loop_calls(),
        "chunked_deltanet_spans_total": fish_drama3_chunked_deltanet_spans(),
        "recurrent_deltanet_spans_total": fish_drama3_recurrent_deltanet_spans(),
        "grouped_moe_spans_total": fish_drama3_grouped_moe_spans(),
        "step_uploads_total": fish_drama3_step_uploads(),
        "step_main_draw_ms_total": ms(&STEP_MAIN_DRAW_US),
        "step_tower_ms_total": ms(&STEP_TOWER_US),
        "step_trunk_ms_total": ms(&STEP_TRUNK_US),
    })
}

pub(crate) fn count_uploads(count: usize) {
    UPLOADS.fetch_add(count as u64, Ordering::Relaxed);
}

pub(crate) fn record_step_section(section: StepSection, elapsed: std::time::Duration) {
    let counter = match section {
        StepSection::MainDraw => &STEP_MAIN_DRAW_US,
        StepSection::Tower => &STEP_TOWER_US,
        StepSection::Trunk => &STEP_TRUNK_US,
    };
    counter.fetch_add(elapsed.as_micros() as u64, Ordering::Relaxed);
}

pub(crate) fn count_step_row_gemv() {
    STEP_ROW_GEMV_CALLS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn count_step_row_loop() {
    STEP_ROW_LOOP_CALLS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn count_chunked_deltanet_span() {
    CHUNKED_DELTANET_SPANS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn count_recurrent_deltanet_span() {
    RECURRENT_DELTANET_SPANS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn count_grouped_moe_span() {
    GROUPED_MOE_SPANS.fetch_add(1, Ordering::Relaxed);
}

/// The tensors a probed forward records, by name (`metal::self_check`).
pub(crate) type Probe<'a> = Option<&'a mut Vec<(String, Tensor)>>;

/// Records `tensor` under `name()` when the forward is probed.
pub(crate) fn record(probe: &mut Probe<'_>, name: impl FnOnce() -> String, tensor: &Tensor) {
    if let Some(probe) = probe.as_deref_mut() {
        probe.push((name(), tensor.clone()));
    }
}

/// One request's rows in a device forward: rows `rows` of the forward's,
/// in request order, at positions `start..start + rows`, with its paged
/// view of the stack's domain and the append prepared for them.
pub(crate) struct DeviceRow<'a> {
    pub(crate) start: usize,
    pub(crate) rows: usize,
    pub(crate) cache: &'a PhysicalPagedKvCache,
    pub(crate) prepared: &'a mut PreparedPhysicalPagedStep,
}

/// The positions of every row of `rows`, in order.
pub(crate) fn positions(rows: &[DeviceRow<'_>]) -> Vec<usize> {
    rows.iter()
        .flat_map(|row| row.start..row.start + row.rows)
        .collect()
}

/// An F32 vector on `device`.
pub(crate) fn vector(values: &[f32], device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_slice(values, values.len(), device)?)
}

/// `x * rsqrt(mean(x²) + eps) * gain` over the last dim: the Qwen3.6 kernel
/// with its F32 gain for a 16-bit layer (it reads the residual through the
/// 16-bit dtype), Candle's F32 RMSNorm for an F32 one. Returns `act`.
pub(crate) fn rms_norm(x: &Tensor, gain: &Tensor, eps: f32, act: DType) -> Result<Tensor> {
    if act != DType::F32 {
        let x = x.to_dtype(act)?.contiguous()?;
        return Ok(crate::kernels::cuda::norm::rms_norm(&x, gain, eps)?);
    }
    let x = x.to_dtype(DType::F32)?.contiguous()?;
    Ok(candle_nn::ops::rms_norm(&x, gain, eps)?)
}

/// `silu(gate) * up` in F32, returned in `act`.
pub(crate) fn swiglu(gate: &Tensor, up: &Tensor, act: DType) -> Result<Tensor> {
    let gate = gate.to_dtype(DType::F32)?;
    let up = up.to_dtype(DType::F32)?;
    Ok((candle_nn::ops::silu(&gate)? * up)?.to_dtype(act)?)
}

/// `residual + delta`, summed in F32 and rounded once to the residual's
/// dtype (an F32 residual on Metal adds exactly as before; CUDA's BF16
/// residual rounds the sum, not each operand).
pub(crate) fn add_residual(residual: &Tensor, delta: &Tensor) -> Result<Tensor> {
    let sum = (residual.to_dtype(DType::F32)? + delta.to_dtype(DType::F32)?)?;
    Ok(sum.to_dtype(residual.dtype())?)
}

/// Rotates the first `rotary_dims` of every head of `x` (`[n, heads,
/// head_dim]`, row `i` at `positions[i]`) in `rope`'s layout, in F32 from
/// the family's host tables (F64 angles rounded once to F32): never device
/// trigonometry, whose fast-math error grows with the position. Returns
/// `x`'s dtype.
pub(crate) fn rope(x: &Tensor, rope: &FishDrama3Rope, positions: &[usize]) -> Result<Tensor> {
    let (n, _, d) = x.dims3()?;
    let rotary = rope.rotary_dims();
    if positions.len() != n || rotary == 0 || rotary > d || !rotary.is_multiple_of(2) {
        return Err(Error::InferenceError(format!(
            "Fish Audio Drama 3 RoPE over {:?} at {} positions, {rotary} rotary dims",
            x.dims(),
            positions.len()
        )));
    }
    let half = rotary / 2;
    let mut cos = Vec::with_capacity(n * half);
    let mut sin = Vec::with_capacity(n * half);
    for &position in positions {
        let (c, s) = rope.cos_sin(position);
        cos.extend(c);
        sin.extend(s);
    }
    let device = x.device();
    let cos = Tensor::from_vec(cos, (n, half), device)?;
    let sin = Tensor::from_vec(sin, (n, half), device)?;
    count_uploads(2);
    let xf = x.to_dtype(DType::F32)?;
    let part = xf.narrow(2, 0, rotary)?;
    let rotated = match rope.layout() {
        // `[1, t, h, d]` against `[t, d / 2]`.
        FishDrama3RopeLayout::HalfSplit => {
            candle_nn::rotary_emb::rope_thd(&part.unsqueeze(0)?.contiguous()?, &cos, &sin)?
                .squeeze(0)?
        }
        // `[1, h, t, d]` against `[t, d / 2]`.
        FishDrama3RopeLayout::InterleavedPairs => candle_nn::rotary_emb::rope_i(
            &part.transpose(0, 1)?.unsqueeze(0)?.contiguous()?,
            &cos,
            &sin,
        )?
        .squeeze(0)?
        .transpose(0, 1)?,
    };
    let out = if rotary < d {
        Tensor::cat(&[&rotated, &xf.narrow(2, rotary, d - rotary)?], 2)?
    } else {
        rotated
    };
    Ok(out.to_dtype(x.dtype())?)
}

/// Gated attention on the device, the trunk's and the tower's alike:
///
/// ```text
/// [q_h | gate_h], K, V = qkv(x)              # one 16-bit matrix, q and gate interleaved per head
/// q, k = (1 + w)-RMSNorm per head; RoPE      # 16-bit kernel, then F32 host-table rotation
/// o    = paged GQA(q, k, v) per request      # K/V stored in the domain's dtype
/// out  = o_proj(o * sigmoid(gate))           # the 16-bit gate kernel
/// ```
pub(crate) struct DeviceAttention {
    /// `[2 q_width + 2 kv_width, hidden]`: per head its query rows then its
    /// gate rows, then K, then V.
    qkv: FishDrama3Linear,
    o_proj: FishDrama3Linear,
    q_gain: Tensor,
    k_gain: Tensor,
    config: FishDrama3AttentionConfig,
    rope: FishDrama3Rope,
    eps: f32,
    /// The stack's shared cos/sin tables, for the fused q/k norm + RoPE.
    tables: Arc<RopeTables>,
}

impl DeviceAttention {
    /// From exact F32 projection values: `query_gate` `[2 q_width, hidden]`
    /// interleaved per head (`[q_h | gate_h]`, the trunk's layout), `keys`
    /// and `values` `[kv_width, hidden]`, `out` `[hidden, q_width]`, and the
    /// q/k norms' `(1 + w)` gains, in `forms`' dense form.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        query_gate: Vec<f32>,
        keys: &[f32],
        values: &[f32],
        out: Vec<f32>,
        q_gain: &[f32],
        k_gain: &[f32],
        config: FishDrama3AttentionConfig,
        eps: f32,
        positions: usize,
        forms: DeviceForms,
        device: &Device,
    ) -> Result<Self> {
        let (qw, kvw) = (config.q_width(), config.kv_width());
        if query_gate.is_empty()
            || !query_gate.len().is_multiple_of(2 * qw)
            || keys.len() * 2 * qw != query_gate.len() * kvw
            || values.len() != keys.len()
            || q_gain.len() != config.head_dim
            || k_gain.len() != config.head_dim
        {
            return Err(Error::ModelLoadError(
                "Fish Audio Drama 3 device attention weights do not match the geometry".into(),
            ));
        }
        let hidden = query_gate.len() / (2 * qw);
        let mut packed = query_gate;
        packed.extend_from_slice(keys);
        packed.extend_from_slice(values);
        let rows = 2 * qw + 2 * kvw;
        Ok(Self {
            qkv: FishDrama3Linear::from_values(packed, rows, hidden, forms.weight_form(), device)?,
            o_proj: FishDrama3Linear::from_values(out, hidden, qw, forms.weight_form(), device)?,
            q_gain: vector(q_gain, device)?,
            k_gain: vector(k_gain, device)?,
            tables: rope_tables(&FishDrama3Rope::new(&config), positions, device)?,
            rope: FishDrama3Rope::new(&config),
            config,
            eps,
        })
    }

    /// `query`, `gate` `[q_width, hidden]` contiguous blocks (the tower's
    /// `wqkv` layout) interleaved per head into the trunk's `[q_h | gate_h]`.
    pub(crate) fn interleave(
        query: &[f32],
        gate: &[f32],
        head_dim: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let block = head_dim * hidden;
        query
            .chunks_exact(block)
            .zip(gate.chunks_exact(block))
            .flat_map(|(q, g)| q.iter().chain(g).copied())
            .collect()
    }

    pub(crate) fn resident_bytes(&self) -> usize {
        self.qkv.resident_bytes() + self.o_proj.resident_bytes() + 2 * self.config.head_dim * 4
    }

    /// The mixer output `[N, hidden]` in `act` of the normed rows `x`
    /// (`[N, hidden]`, `rows` in order); each request's K/V go into layer
    /// `slot` of its view through its prepared append.
    pub(crate) fn forward(
        &self,
        x: &Tensor,
        rows: &mut [DeviceRow<'_>],
        slot: usize,
        phase: FishDrama3KernelPhase,
        act: DType,
        probe: &mut Probe<'_>,
    ) -> Result<Tensor> {
        let a = &self.config;
        let (heads, kv_heads, d) = (a.heads, a.kv_heads, a.head_dim);
        let (qw, kvw) = (a.q_width(), a.kv_width());
        let n = x.dim(0)?;
        let packed = self.qkv.forward_device(x, phase, act)?;
        record(probe, || "qkv".into(), &packed);
        let query_gate = packed
            .narrow(1, 0, 2 * qw)?
            .contiguous()?
            .reshape((n, heads, 2 * d))?;
        let queries = query_gate.narrow(2, 0, d)?.contiguous()?;
        let keys = packed.narrow(1, 2 * qw, kvw)?.contiguous()?;
        let values = packed
            .narrow(1, 2 * qw + kvw, kvw)?
            .contiguous()?
            .reshape((n, kv_heads, d))?;
        let queries = rms_norm(
            &queries.reshape((n * heads, d))?,
            &self.q_gain,
            self.eps,
            act,
        )?
        .reshape((n, heads, d))?;
        let keys = rms_norm(
            &keys.reshape((n * kv_heads, d))?,
            &self.k_gain,
            self.eps,
            act,
        )?
        .reshape((n, kv_heads, d))?;
        let at = positions(rows);
        let queries = rope(&queries, &self.rope, &at)?;
        let keys = rope(&keys, &self.rope, &at)?;
        record(probe, || "keys".into(), &keys);
        record(probe, || "values".into(), &values);
        let scale = (1.0 / (d as f64).sqrt()) as f32;
        let mut attended = Vec::with_capacity(rows.len());
        let mut offset = 0;
        for row in rows.iter_mut() {
            let kv = row.cache.arena().config().dtype;
            let take = |t: &Tensor| -> Result<Tensor> {
                Ok(t.narrow(0, offset, row.rows)?.to_dtype(kv)?.contiguous()?)
            };
            // One request per call: a decode row's partitions come from its
            // own length only.
            let out = row.cache.write_and_attend_row_invariant(
                slot,
                row.prepared,
                &take(&queries)?,
                &take(&keys)?,
                &take(&values)?,
                scale,
            )?;
            attended.push(out.to_dtype(act)?.reshape((row.rows, qw))?);
            offset += row.rows;
        }
        if offset != n {
            return Err(Error::InferenceError(format!(
                "Fish Audio Drama 3 attention got {n} rows for requests of {offset}"
            )));
        }
        let attended = Tensor::cat(&attended, 0)?;
        let gated = crate::kernels::cuda::gate::attn_gate(&attended, &query_gate, heads, d)?;
        record(probe, || "gated".into(), &gated);
        let out = self.o_proj.forward_device(&gated, phase, act)?;
        record(probe, || "attention_out".into(), &out);
        Ok(out)
    }

    /// [`Self::forward`] with q, k and v read, normalized and rotated straight
    /// from the packed projection in one launch (`kernels::qk_norm_rope`) from
    /// the stack's tables at `positions` (`[N]` U32, one upload per stack
    /// forward). Matches it to rounding.
    pub(crate) fn forward_fused(
        &self,
        x: &Tensor,
        rows: &mut [DeviceRow<'_>],
        slot: usize,
        phase: FishDrama3KernelPhase,
        act: DType,
        positions: &Tensor,
    ) -> Result<Tensor> {
        let a = &self.config;
        let (heads, d, qw) = (a.heads, a.head_dim, a.q_width());
        let n = x.dim(0)?;
        let limit = self.tables.cos.dim(0)?;
        if let Some(row) = rows.iter().find(|row| row.start + row.rows > limit) {
            return Err(Error::InferenceError(format!(
                "Fish Audio Drama 3 attention position {} is past its {limit} RoPE rows",
                row.start + row.rows - 1
            )));
        }
        let packed = self.qkv.forward_device(x, phase, act)?;
        let spec = qk_norm_rope_spec(a, &self.rope, self.eps);
        let (queries, keys, values) = crate::kernels::qk_norm_rope::qk_norm_rope(
            &packed,
            &self.q_gain,
            &self.k_gain,
            &self.tables.cos,
            &self.tables.sin,
            positions,
            spec,
        )?;
        let scale = (1.0 / (d as f64).sqrt()) as f32;
        let mut attended = Vec::with_capacity(rows.len());
        let mut offset = 0;
        for row in rows.iter_mut() {
            let kv = row.cache.arena().config().dtype;
            let take = |t: &Tensor| -> Result<Tensor> {
                Ok(t.narrow(0, offset, row.rows)?.to_dtype(kv)?.contiguous()?)
            };
            let out = row.cache.write_and_attend_row_invariant(
                slot,
                row.prepared,
                &take(&queries)?,
                &take(&keys)?,
                &take(&values)?,
                scale,
            )?;
            attended.push(out.to_dtype(act)?.reshape((row.rows, qw))?);
            offset += row.rows;
        }
        if offset != n {
            return Err(Error::InferenceError(format!(
                "Fish Audio Drama 3 attention got {n} rows for requests of {offset}"
            )));
        }
        let attended = Tensor::cat(&attended, 0)?;
        let query_gate = packed
            .narrow(1, 0, 2 * qw)?
            .contiguous()?
            .reshape((n, heads, 2 * d))?;
        let gated = crate::kernels::cuda::gate::attn_gate(&attended, &query_gate, heads, d)?;
        self.o_proj.forward_device(&gated, phase, act)
    }
}
