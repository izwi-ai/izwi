//! What the device column's tests share (`metal/tests.rs`,
//! `cuda/tests.rs`): the synthetic loads, prompt rows, caches and the
//! decode-together harness of the batch-invariance tests.

use std::sync::Arc;

use candle_core::{DType, Device, Tensor};

use super::super::config::FishDrama3ConfigPolicy;
use super::super::linear::with_split_k_steps;
use super::super::model::tests::write_synthetic_model_in;
use super::super::model::{FishDrama3GenerationParams, FishDrama3ModelOptions};
use super::super::prompt::FishDrama3Prompt;
use super::super::residency::FishDrama3CpuPrecision;
use super::super::sampling::FishDrama3SamplingParams;
use super::super::synthetic::{SyntheticCheckpoint, SyntheticGeometry};
use super::super::test_support::StubCodec;
use super::super::tower::FishDrama3TowerOptions;
use super::super::trunk::FishDrama3TrunkOptions;
use super::super::FishDrama3Model;
use crate::kernels::numeric_check::{relative_l2, xorshift_values};
use crate::models::shared::attention::physical::{
    test_physical_caches_for_model_layers_on, PhysicalPagedKvCache,
};

pub(crate) fn options(precision: FishDrama3CpuPrecision) -> FishDrama3ModelOptions {
    FishDrama3ModelOptions {
        trunk: FishDrama3TrunkOptions {
            precision,
            ..Default::default()
        },
        tower: FishDrama3TowerOptions {
            precision,
            ..Default::default()
        },
        config_policy: FishDrama3ConfigPolicy::Synthetic,
        ..Default::default()
    }
}

pub(crate) fn load(
    synthetic: &SyntheticCheckpoint,
    device: &Device,
    options: FishDrama3ModelOptions,
) -> FishDrama3Model {
    FishDrama3Model::load_with_codec(synthetic.path(), device, options, Arc::new(StubCodec))
        .unwrap()
}

/// The CPU reference with exact F32 weights and the device load of one
/// checkpoint.
pub(crate) fn pair(
    geometry: SyntheticGeometry,
    seed: u64,
    device: &Device,
) -> (SyntheticCheckpoint, FishDrama3Model, FishDrama3Model) {
    let synthetic = write_synthetic_model_in(seed, geometry, |_| {});
    let cpu = load(
        &synthetic,
        &Device::Cpu,
        options(FishDrama3CpuPrecision::F32),
    );
    let on_device = load(&synthetic, device, options(FishDrama3CpuPrecision::Q8_0));
    (synthetic, cpu, on_device)
}

/// `n` prompt-like rows: text rows around a run of semantic rows with
/// xorshift codes.
pub(crate) fn rows(model: &FishDrama3Model, n: usize, seed: u64) -> Vec<u32> {
    let ids = model.config().ids;
    let width = model.config().tower.frame_width();
    let draws = xorshift_values(n * width, seed, 1.0);
    let pick = |i: usize, limit: u32| ((draws[i].abs() * limit as f32) as u32).min(limit - 1);
    let mut out = Vec::with_capacity(n * width);
    for t in 0..n {
        let mut row = vec![0u32; width];
        if t % 3 != 0 {
            let c0 = pick(t * width, 64);
            row[0] = ids.semantic_start + c0;
            row[1] = c0;
            for (k, value) in row.iter_mut().enumerate().skip(2) {
                *value = pick(t * width + k, 64);
            }
        } else {
            row[0] = pick(t * width, ids.eos);
        }
        out.extend(row);
    }
    out
}

pub(crate) fn host(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

pub(crate) fn trunk_caches(
    model: &FishDrama3Model,
    device: &Device,
    dtype: DType,
    count: usize,
) -> Vec<PhysicalPagedKvCache> {
    let a = &model.config().trunk.attention;
    test_physical_caches_for_model_layers_on(
        device,
        dtype,
        model.trunk().attention_layers(),
        a.kv_heads,
        a.head_dim,
        64,
        count,
    )
}

pub(crate) fn ok() -> crate::error::Result<()> {
    Ok(())
}

/// The prompt of `rows` (row-major frames).
pub(crate) fn prompt_of(model: &FishDrama3Model, rows: &[u32]) -> FishDrama3Prompt {
    let width = model.config().tower.frame_width();
    let n = rows.len() / width;
    FishDrama3Prompt {
        values: (0..width)
            .map(|k| (0..n).map(|t| rows[t * width + k]).collect())
            .collect(),
        rows: n,
    }
}

pub(crate) fn params(seed: u64, frames: usize) -> FishDrama3GenerationParams {
    FishDrama3GenerationParams::new(
        FishDrama3SamplingParams::from_request(None, None, None, seed).unwrap(),
        frames,
    )
}

/// Each request prefilled, then `rounds` decode rounds of all of them
/// together (under the split-K RED hook when `red`): every request's codes
/// and `h_t`.
pub(crate) fn decode_together(
    model: &FishDrama3Model,
    requests: &[(&FishDrama3Prompt, u64)],
    rounds: usize,
    red: bool,
) -> Vec<(Vec<Vec<u32>>, Vec<f32>)> {
    let mut states = requests
        .iter()
        .map(|(prompt, seed)| {
            model
                .new_private_request((*prompt).clone(), params(*seed, 16))
                .unwrap()
        })
        .collect::<Vec<_>>();
    for state in &mut states {
        model.run_prefill(state, &ok).unwrap();
    }
    for _ in 0..rounds {
        let mut going = states
            .iter_mut()
            .filter(|state| !state.is_finished())
            .collect::<Vec<_>>();
        if going.is_empty() {
            break;
        }
        if red {
            with_split_k_steps(|| model.decode_round(&mut going)).unwrap();
        } else {
            model.decode_round(&mut going).unwrap();
        }
    }
    states
        .iter()
        .map(|state| (state.codes().to_vec(), state.last_hidden_values()))
        .collect()
}

/// The trunk's span hidden on the device against the CPU's over `n`
/// prompt rows.
pub(crate) fn trunk_span_error(
    cpu: &FishDrama3Model,
    on_device: &FishDrama3Model,
    device: &Device,
    n: usize,
) -> f64 {
    let prompt = rows(cpu, n, 9);
    let run = |model: &FishDrama3Model, device: &Device, dtype| {
        let mut state = model.trunk().new_state().unwrap();
        let mut cache = trunk_caches(model, device, dtype, 1).remove(0);
        let x = model.embedding().embed_tensor(&prompt, device).unwrap();
        host(
            &model
                .trunk()
                .forward_span(&x, 0, &mut state, &mut cache)
                .unwrap(),
        )
    };
    let a = run(cpu, &Device::Cpu, DType::F32);
    let b = run(on_device, device, on_device.kv_dtype());
    relative_l2(&b, &a)
}

/// One load of `seed`'s checkpoint on `device`, through the fused path and
/// then the eager one: relative L2 of the trunk's final hidden over a span
/// and three steps, and of the tower's layer outputs and head logits over a
/// span and a forced step.
pub(crate) fn fused_against_eager(device: &Device, seed: u64) -> (f64, f64) {
    use super::super::tower::FishDrama3TowerForcedRow;
    use super::super::trunk::FishDrama3StepRow;
    let synthetic = write_synthetic_model_in(seed, SyntheticGeometry::KernelHeads, |_| {});
    let mut model = load(&synthetic, device, options(FishDrama3CpuPrecision::Q8_0));
    assert!(
        model.decode_path().is_some_and(|path| path.is_fused()),
        "{:?}",
        model.decode_path()
    );
    let config = model.config().clone();
    let (width, n) = (config.tower.frame_width(), 12);
    let prompt = rows(&model, n + 3, seed);
    let frames = rows(&model, n + 2, seed + 1);
    let d = config.tower.dim;
    let hidden = xorshift_values((n + 1) * d, seed + 2, 1.7);
    let run = |model: &FishDrama3Model| -> (Vec<f32>, Vec<f32>) {
        let mut trunk_out = Vec::new();
        let mut state = model.trunk().new_state().unwrap();
        let mut cache = trunk_caches(model, device, model.kv_dtype(), 1).remove(0);
        let (span, steps) = prompt.split_at(n * width);
        let x = model.embedding().embed_tensor(span, device).unwrap();
        let out = model
            .trunk()
            .forward_span(&x, 0, &mut state, &mut cache)
            .unwrap();
        trunk_out.extend(host(&out));
        for row in steps.chunks_exact(width) {
            let x = model.embedding().embed_tensor(row, device).unwrap();
            let out = model
                .trunk()
                .forward_step(
                    &x,
                    &mut [FishDrama3StepRow {
                        state: &mut state,
                        cache: &mut cache,
                    }],
                )
                .unwrap();
            trunk_out.extend(host(&out));
        }
        let a = &config.tower.attention;
        let mut cache = test_physical_caches_for_model_layers_on(
            device,
            model.kv_dtype(),
            model.tower().attention_layers(),
            a.kv_heads,
            a.head_dim,
            64,
            1,
        )
        .remove(0);
        let span = Tensor::from_slice(&hidden[..n * d], (n, d), device).unwrap();
        let mut tower_out = model
            .tower()
            .forward_span_traced(
                model.embedding(),
                &span,
                &frames[..(n + 1) * width],
                0,
                &mut cache,
            )
            .unwrap()
            .concat();
        let step = Tensor::from_slice(&hidden[n * d..], (1, d), device).unwrap();
        tower_out.extend(
            model
                .tower()
                .forward_step_forced(
                    model.embedding(),
                    &step,
                    &mut [FishDrama3TowerForcedRow {
                        cache: &mut cache,
                        input: &frames[n * width..(n + 1) * width],
                        label: &frames[(n + 1) * width..],
                    }],
                )
                .unwrap(),
        );
        (trunk_out, tower_out)
    };
    let (trunk_fused, tower_fused) = run(&model);
    model.force_fused(false);
    let (trunk_eager, tower_eager) = run(&model);
    (
        relative_l2(&trunk_fused, &trunk_eager),
        relative_l2(&tower_fused, &tower_eager),
    )
}
