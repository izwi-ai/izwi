//! A Metal load's checks (§6 "Self-checks"), each with its counter in
//! [`FishDrama3MetalChecks`]:
//!
//! * **Host RoPE tables.** Both conventions (the trunk's NeoX half-split,
//!   the tower's interleaved pairs) rotate xorshift heads at positions 0,
//!   12,287 and 262,143 on the device against F64. A breach refuses the
//!   load: a wrong rotation has no slower path that fixes it.
//! * **FP8 GEMV vs F32 dequant.** The first and last routed experts of the
//!   first and last MoE layers, every projection kind, through the FP8
//!   GEMV over the stacked resident bytes, against the F64 dequantization
//!   of the original checkpoint tensors on 64 sampled rows. A breach with
//!   F16 activations that F32 activations pass moves every layer to the
//!   host expert loop (F32 activations); a breach in both refuses the load.
//! * **F16 range probe** (`kernels::numeric_check`, Fish S2's pattern at
//!   per-layer grain): xorshift text ids embedded and run through every
//!   trunk layer, the final norm, the compact head, every tower layer and
//!   the codebook heads, one at a time. A component whose activations are
//!   non-finite or reach [`F16_SAFE_MAX_ABS`] re-runs in F32 and keeps F32
//!   activations; a breached trunk layer also moves its experts to the
//!   host loop (the fused kernels are F16 only). A K or V past the limit
//!   puts both paged domains in F32. The probe's KV are F32 scratch caches,
//!   so one overflowing write cannot hide the layers after it.

use candle_core::{DType, Device, Tensor};

use super::super::checkpoint::{expert, FishDrama3Checkpoint};
use super::super::config::{FishDrama3AttentionConfig, FishDrama3Config, FishDrama3RopeLayout};
use super::super::embed::FishDrama3Embedding;
use super::super::head::FishDrama3CompactHead;
use super::super::linear::FishDrama3KernelPhase;
use super::super::precision::{
    FishDrama3MetalChecks, FishDrama3ProbedComponent, FP8_GEMV_TOLERANCE, ROPE_TOLERANCE,
};
use super::super::tower::FishDrama3Tower;
use super::super::trunk::attention::FishDrama3Rope;
use super::super::trunk::FishDrama3Trunk;
use super::trunk::FishDrama3MoeBackend;
use super::{rope, MetalRow, HALF};
use crate::error::{Error, Result};
use crate::kernels::numeric_check::{relative_l2, xorshift_values, ProbedTensor, F16_SAFE_MAX_ABS};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

/// Rows the range probe runs.
pub(crate) const PROBE_ROWS: usize = 16;
/// The positions the RoPE check rotates at: the first, the last of a
/// worst admitted request, the last a table may be asked for.
pub(crate) const ROPE_POSITIONS: [usize; 3] = [0, 12_287, 262_143];
/// Rows of each FP8 projection the check compares.
const FP8_ROWS: usize = 64;

fn load_error(message: impl Into<String>) -> Error {
    Error::ModelLoadError(format!(
        "Fish Audio Drama 3 Metal self-check: {}",
        message.into()
    ))
}

/// The F64 rotation of one head-major row `x` at `position`.
pub(crate) fn rope_reference(
    x: &[f32],
    position: usize,
    config: &FishDrama3AttentionConfig,
) -> Vec<f64> {
    let rotary = config.rotary_dims;
    let half = rotary / 2;
    let mut out = x.iter().map(|v| f64::from(*v)).collect::<Vec<_>>();
    for head in out.chunks_exact_mut(config.head_dim) {
        let original = head.to_vec();
        for i in 0..half {
            let angle = position as f64 / config.rope_theta.powf((2 * i) as f64 / rotary as f64);
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

/// Rotates xorshift heads on `device` through the family's host tables at
/// [`ROPE_POSITIONS`] for each of `configs`.
pub(crate) fn check_rope(
    configs: &[(&str, FishDrama3AttentionConfig)],
    device: &Device,
    checks: &mut FishDrama3MetalChecks,
) -> Result<()> {
    for (name, config) in configs {
        let table = FishDrama3Rope::new(config);
        let heads = config.heads;
        let x = xorshift_values(heads * config.head_dim, 0x0D3A, 1.0);
        for position in ROPE_POSITIONS {
            let input = Tensor::from_slice(&x, (1, heads, config.head_dim), device)?;
            let rotated = rope(&input, &table, &[position])?
                .flatten_all()?
                .to_device(&Device::Cpu)?
                .to_vec1::<f32>()?;
            let expected = rope_reference(&x, position, config);
            let error = rotated
                .iter()
                .zip(&expected)
                .map(|(a, b)| (f64::from(*a) - b).abs())
                .fold(0f64, |worst, e| {
                    if e.is_finite() {
                        worst.max(e)
                    } else {
                        f64::INFINITY
                    }
                });
            checks
                .rope_checks
                .push((format!("{name} {:?}", config.rope_layout), position, error));
            if error > ROPE_TOLERANCE {
                checks.rope_table_breaches += 1;
            }
        }
    }
    if checks.rope_table_breaches > 0 {
        return Err(load_error(format!(
            "{} RoPE rotations deviate from F64 by more than {ROPE_TOLERANCE:.0e}: {:?}",
            checks.rope_table_breaches, checks.rope_checks
        )));
    }
    Ok(())
}

/// The FP8 GEMV of `x` (`[2, cols]` F32, rounded to `dtype` for the
/// kernel) over `[rows, cols]` resident bytes `weights` and `scales`,
/// against the original tensor's F64 dequantization on [`FP8_ROWS`]
/// sampled rows.
pub(crate) fn fp8_error(
    weights: &Tensor,
    scales: &Tensor,
    original: &super::super::checkpoint::Fp8BlockWeight<'_>,
    dtype: DType,
) -> Result<f64> {
    let (rows, cols) = (original.rows, original.cols);
    let device = weights.device();
    let x = xorshift_values(2 * cols, 0xF8, 1.0);
    let input = Tensor::from_slice(&x, (2, cols), device)?.to_dtype(dtype)?;
    let read = input
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .to_vec2::<f32>()?;
    let y = crate::kernels::cuda::fp8::block_fp8_projection(
        &input,
        &weights.contiguous()?,
        &scales.contiguous()?,
    )?
    .to_dtype(DType::F32)?
    .to_device(&Device::Cpu)?
    .to_vec2::<f32>()?;
    let count = rows.min(FP8_ROWS);
    let sampled = (0..count).map(|i| i * rows / count).collect::<Vec<_>>();
    let mut got = Vec::with_capacity(2 * count);
    let mut expected = Vec::with_capacity(2 * count);
    for (m, input) in read.iter().enumerate() {
        for &row in &sampled {
            let w = original
                .dequantized_row(row)
                .ok_or_else(|| load_error(format!("no row {row}")))?;
            expected.push(
                w.iter()
                    .zip(input)
                    .map(|(w, x)| f64::from(*w) * f64::from(*x))
                    .sum::<f64>() as f32,
            );
            got.push(y[m][row]);
        }
    }
    Ok(relative_l2(&got, &expected))
}

/// Checks the stacked experts against the checkpoint's originals and
/// returns whether the fused (F16) path is sound; refuses the load when
/// the F32 GEMV is not either.
pub(crate) fn check_fp8(
    checkpoint: &FishDrama3Checkpoint,
    config: &FishDrama3Config,
    trunk: &FishDrama3Trunk,
    checks: &mut FishDrama3MetalChecks,
) -> Result<bool> {
    let metal = trunk
        .metal()
        .ok_or_else(|| load_error("the trunk holds no Metal weights"))?;
    let t = &config.trunk;
    let (inter, block) = (t.moe.intermediate, config.fp8_block);
    let layers = [0, t.layers.len() - 1];
    let experts = [0, t.moe.experts - 1];
    let mut fused_sound = true;
    for &layer in layers
        .iter()
        .take(if layers[0] == layers[1] { 1 } else { 2 })
    {
        let moe = &metal.layers[layer].moe;
        for &e in experts
            .iter()
            .take(if experts[0] == experts[1] { 1 } else { 2 })
        {
            let (w13, s13) = (moe.w13.get(e)?, moe.s13.get(e)?);
            let (w2, s2) = (moe.w2.get(e)?, moe.s2.get(e)?);
            let kinds = [
                (
                    "gate_proj",
                    w13.narrow(0, 0, inter)?,
                    s13.narrow(0, 0, inter / block)?,
                ),
                (
                    "up_proj",
                    w13.narrow(0, inter, inter)?,
                    s13.narrow(0, inter / block, inter / block)?,
                ),
                ("down_proj", w2, s2),
            ];
            for (kind, weights, scales) in kinds {
                let module = format!("{}.{kind}", expert(layer, e));
                let (rows, cols) = weights.dims2()?;
                let original = checkpoint.fp8_block_weight(&module, rows, cols, block)?;
                let half = fp8_error(&weights, &scales, &original, HALF)?;
                checks.fp8_checks.push((format!("{module} f16"), half));
                if half <= FP8_GEMV_TOLERANCE {
                    continue;
                }
                checks.fp8_gemv_breaches += 1;
                fused_sound = false;
                let full = fp8_error(&weights, &scales, &original, DType::F32)?;
                checks.fp8_checks.push((format!("{module} f32"), full));
                if full > FP8_GEMV_TOLERANCE {
                    return Err(load_error(format!(
                        "the FP8 GEMV over {module} deviates by {full:.3e} with F32 activations \
                         (tolerance {FP8_GEMV_TOLERANCE:.0e})"
                    )));
                }
            }
        }
    }
    Ok(fused_sound)
}

/// One probed component's verdict.
fn verdict(name: String, tensors: &[(String, Tensor)]) -> Result<FishDrama3ProbedComponent> {
    let mut max_abs = 0f32;
    let mut non_finite = 0;
    for (_, tensor) in tensors {
        let values = crate::kernels::numeric_check::host_f32(tensor)?;
        let probed = ProbedTensor::from_values("", &values);
        max_abs = max_abs.max(probed.max_abs);
        non_finite += probed.non_finite;
    }
    Ok(FishDrama3ProbedComponent {
        name,
        max_abs,
        non_finite,
        breached: non_finite > 0 || max_abs >= F16_SAFE_MAX_ABS,
    })
}

/// Whether the K/V a probed forward recorded leave F16's safe range.
fn kv_breached(tensors: &[(String, Tensor)]) -> Result<bool> {
    for (name, tensor) in tensors {
        if name == "keys" || name == "values" {
            let values = crate::kernels::numeric_check::host_f32(tensor)?;
            let probed = ProbedTensor::from_values(name.as_str(), &values);
            if probed.non_finite > 0 || probed.max_abs >= F16_SAFE_MAX_ABS {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// An F32 scratch view of `layers` layers for the probe's rows.
fn scratch(
    device: &Device,
    layers: usize,
    attention: &FishDrama3AttentionConfig,
) -> Result<PhysicalPagedKvCache> {
    PhysicalPagedKvCache::scratch_for_probe(
        device,
        DType::F32,
        layers,
        attention.kv_heads,
        attention.head_dim,
        PROBE_ROWS,
    )
}

/// Runs the F16 range probe and moves every breached component to F32.
/// Returns the paged KV dtype.
pub(crate) fn probe_ranges(
    config: &FishDrama3Config,
    embedding: &FishDrama3Embedding,
    trunk: &mut FishDrama3Trunk,
    tower: &mut FishDrama3Tower,
    head: &mut FishDrama3CompactHead,
    checks: &mut FishDrama3MetalChecks,
) -> Result<DType> {
    let device = trunk.device().clone();
    let width = config.tower.frame_width();
    let draws = xorshift_values(PROBE_ROWS * width, 0xF15A, 1.0);
    let pick = |draw: f32, limit: u32| ((draw.abs() * limit as f32) as u32).min(limit - 1);
    let text = config
        .ids
        .semantic_start
        .min(config.trunk.vocab as u32)
        .max(1);
    let codes = config.tower.codebook_vocab as u32;
    // Text rows through the trunk; their frames (semantic mains with
    // xorshift codes) label the tower.
    let mut rows = Vec::with_capacity(PROBE_ROWS * width);
    let mut frames = Vec::with_capacity(PROBE_ROWS * width);
    for row in draws.chunks_exact(width) {
        rows.push(pick(row[0], text));
        rows.extend(std::iter::repeat_n(0, width - 1));
        let c0 = pick(row[1], config.ids.semantic_codes() as u32);
        frames.push(config.ids.semantic_start + c0);
        frames.push(c0);
        frames.extend(row[2..].iter().map(|draw| pick(*draw, codes)));
    }
    let mut x = embedding.embed_tensor(&rows, &device)?;
    let mut kv_f32 = false;

    // The trunk, layer by layer. A layer's attention reads only its own
    // slot's rows of this span, so each run takes a fresh scratch view (a
    // re-run may not write its slot twice through one append).
    let attention_layers = trunk.attention_layers().len();
    let mut layers = trunk.new_state()?.layers()?.to_vec();
    let depth = trunk.config().layers.len();
    for index in 0..depth {
        let metal = trunk
            .metal()
            .ok_or_else(|| load_error("the trunk holds no Metal weights"))?;
        let run = |layers: &mut Vec<_>, act, recorded: &mut Vec<(String, Tensor)>| {
            let mut cache = scratch(&device, attention_layers, &config.trunk.attention)?;
            let mut prepared = cache.prepare_append(0, PROBE_ROWS)?;
            let out = metal.layer(
                index,
                &x,
                &mut [MetalRow {
                    start: 0,
                    rows: PROBE_ROWS,
                    cache: &cache,
                    prepared: &mut prepared,
                }],
                &mut [layers.as_mut_slice()],
                FishDrama3KernelPhase::Span,
                act,
                &mut Some(recorded),
                None,
            );
            let _ = cache.abort_prepared(prepared);
            out
        };
        let before = layers.clone();
        let mut recorded = Vec::new();
        let out = run(&mut layers, None, &mut recorded)?;
        let component = verdict(format!("trunk.{index}"), &recorded)?;
        let breached = component.breached && metal.layers[index].act == HALF;
        if breached {
            layers = before;
            recorded.clear();
            x = run(&mut layers, Some(DType::F32), &mut recorded)?;
        } else {
            x = out;
        }
        kv_f32 |= kv_breached(&recorded)?;
        checks.components.push(component);
        if breached {
            checks.f16_range_breaches += 1;
            let layer = &mut trunk.metal_mut().expect("checked above").layers[index];
            layer.act = DType::F32;
            layer.moe.backend = FishDrama3MoeBackend::Loop;
        }
    }

    // The final norm, then the compact head.
    let metal = trunk.metal().expect("checked above");
    let final_half = metal.final_norm(&x, Some(HALF))?;
    let component = verdict(
        "trunk.final_norm".into(),
        &[("final".into(), final_half.clone())],
    )?;
    let hidden = if component.breached {
        checks.f16_range_breaches += 1;
        trunk.metal_mut().expect("checked above").final_act = DType::F32;
        trunk.metal().expect("checked above").final_norm(&x, None)?
    } else {
        final_half
    };
    checks.components.push(component);
    let logits = head.logits(&hidden)?;
    let component = verdict("head".into(), &[("logits".into(), logits)])?;
    if component.breached && head.act() == HALF {
        checks.f16_range_breaches += 1;
        head.set_act(DType::F32);
    }
    checks.components.push(component);

    // The tower, teacher-forced on the probe frames.
    let metal_tower = tower
        .metal()
        .ok_or_else(|| load_error("the tower holds no Metal weights"))?;
    let shared = config.tower.shared_layers;
    let tower_layers = metal_tower.layers.len();
    let extra = embedding.extra_device(&frames)?;
    let labelled = Tensor::ones((PROBE_ROWS, 1), DType::U8, &device)?;
    let mut h = hidden;
    let mut head_tensors = Vec::new();
    for slot in 0..tower_layers {
        let metal_tower = tower.metal().expect("checked above");
        let column = 1 + slot.saturating_sub(shared);
        let codes = Tensor::from_vec(
            frames
                .chunks_exact(width)
                .map(|frame| frame[column])
                .collect::<Vec<_>>(),
            PROBE_ROWS,
            &device,
        )?;
        let e = metal_tower.inject(slot, &codes, &labelled, (slot == 0).then_some(&extra))?;
        let run = |act, recorded: &mut Vec<(String, Tensor)>| {
            let mut cache = scratch(&device, tower_layers, &config.tower.attention)?;
            let mut prepared = cache.prepare_append(0, PROBE_ROWS)?;
            let out = metal_tower.layer(
                slot,
                &e,
                &h,
                &mut [MetalRow {
                    start: 0,
                    rows: PROBE_ROWS,
                    cache: &cache,
                    prepared: &mut prepared,
                }],
                FishDrama3KernelPhase::Span,
                act,
                &mut Some(recorded),
            );
            let _ = cache.abort_prepared(prepared);
            out
        };
        let mut recorded = Vec::new();
        let out = run(None, &mut recorded)?;
        let mut component = verdict(format!("tower.{slot}"), &recorded)?;
        let breached = component.breached && metal_tower.layers[slot].act == HALF;
        if breached {
            recorded.clear();
            h = run(Some(DType::F32), &mut recorded)?;
        } else {
            h = out;
            // The layer's output feeds the next one in F16.
            let output = verdict(String::new(), &[("output".into(), h.clone())])?;
            component.max_abs = component.max_abs.max(output.max_abs);
        }
        kv_f32 |= kv_breached(&recorded)?;
        checks.components.push(component);
        if breached {
            checks.f16_range_breaches += 1;
            tower.metal_mut().expect("checked above").layers[slot].act = DType::F32;
        }
        if slot >= shared {
            let metal_tower = tower.metal().expect("checked above");
            head_tensors.push((
                format!("head {}", slot - shared),
                metal_tower.head_logits(
                    slot - shared,
                    &h,
                    FishDrama3KernelPhase::Span,
                    Some(HALF),
                )?,
            ));
        }
    }
    let component = verdict("tower.heads".into(), &head_tensors)?;
    if component.breached {
        checks.f16_range_breaches += 1;
        tower.metal_mut().expect("checked above").head_act = DType::F32;
    }
    checks.components.push(component);
    Ok(if kv_f32 { DType::F32 } else { HALF })
}

/// Every check of a Metal load at `activations` (F16 unless
/// `IZWI_FISH_DRAMA3_DTYPE=f32`), moving what breached; returns the report
/// and the paged KV dtype.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    checkpoint: &FishDrama3Checkpoint,
    config: &FishDrama3Config,
    embedding: &FishDrama3Embedding,
    trunk: &mut FishDrama3Trunk,
    tower: &mut FishDrama3Tower,
    head: &mut FishDrama3CompactHead,
    activations: DType,
) -> Result<(FishDrama3MetalChecks, DType)> {
    let device = trunk.device().clone();
    let mut checks = FishDrama3MetalChecks {
        activations: format!("{activations:?}"),
        max_abs_limit: F16_SAFE_MAX_ABS,
        ..Default::default()
    };
    check_rope(
        &[
            ("trunk", config.trunk.attention),
            ("tower", config.tower.attention),
        ],
        &device,
        &mut checks,
    )?;
    let fused_sound = check_fp8(checkpoint, config, trunk, &mut checks)?;
    let kv = if activations == DType::F32 {
        // Every component in F32: nothing to probe.
        let metal = trunk.metal_mut().expect("a Metal trunk");
        for layer in &mut metal.layers {
            layer.act = DType::F32;
            layer.moe.backend = FishDrama3MoeBackend::Loop;
        }
        metal.final_act = DType::F32;
        head.set_act(DType::F32);
        let metal = tower.metal_mut().expect("a Metal tower");
        for layer in &mut metal.layers {
            layer.act = DType::F32;
        }
        metal.head_act = DType::F32;
        DType::F32
    } else {
        if !fused_sound {
            for layer in &mut trunk.metal_mut().expect("a Metal trunk").layers {
                layer.moe.backend = FishDrama3MoeBackend::Loop;
            }
        }
        probe_ranges(config, embedding, trunk, tower, head, &mut checks)?
    };
    let metal = trunk.metal().expect("a Metal trunk");
    checks.moe_loop_layers = metal
        .layers
        .iter()
        .enumerate()
        .filter(|(_, layer)| layer.moe.backend == FishDrama3MoeBackend::Loop)
        .map(|(index, _)| index)
        .collect();
    checks.kv_dtype = format!("{kv:?}");
    if checks.f16_range_breaches > 0 || kv == DType::F32 && activations == HALF {
        tracing::warn!(
            "Fish Audio Drama 3 on Metal: {} components left F16's range on the load probe and \
             run in F32; paged KV {kv:?}",
            checks.f16_range_breaches
        );
    }
    Ok((checks, kv))
}
