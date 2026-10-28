//! Decode-step timing for LFM2.5-1.2B on the real GGUF (ignored; needs the
//! checkpoint). Separates model time from engine overhead: the forward pass
//! alone, the forward plus a blocking argmax readback, and the full
//! `decode_step_batch` (mask, finite check, selection) for one row.
//!
//! Run: `cargo test -p izwi-core --features metal --release --lib lfm2::chat::bench_tests -- --ignored --nocapture`
//! (`IZWI_BENCH_LFM_MODEL=LFM3-1.7B-GGUF` times LFM3-1.7B in the decode-step
//! bench instead).
use super::reference_tests::{models_dir_for_bench, Sessions};
use super::*;
use crate::backends::DeviceProfile;
use candle_core::DType;
use std::time::Instant;

fn steps() -> usize {
    std::env::var("IZWI_BENCH_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(128)
        .min(200)
}

/// Extra rounds of the serving loop, for attaching a sampling profiler.
fn profile_rounds() -> usize {
    std::env::var("IZWI_BENCH_PROFILE_ROUNDS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

/// The checkpoint the decode-step bench loads: LFM2.5-1.2B-Instruct, or the
/// catalog GGUF `IZWI_BENCH_LFM_MODEL` names.
fn bench_variant() -> ModelVariant {
    std::env::var("IZWI_BENCH_LFM_MODEL")
        .ok()
        .map(|name| {
            crate::catalog::parse_model_variant(&name)
                .expect("IZWI_BENCH_LFM_MODEL names a catalog model")
        })
        .unwrap_or(ModelVariant::Lfm2512BInstructGguf)
}

fn bench(profile: DeviceProfile) {
    let variant = bench_variant();
    let dir = models_dir_for_bench().join(variant.dir_name());
    if !dir.exists() {
        eprintln!("skipping: {} is missing", dir.display());
        return;
    }
    let model = Lfm2ChatModel::load(&dir, variant, profile).unwrap();
    let backend = format!("{:?}", model.device_kind());
    let sessions = Sessions::new(&model);
    let device = model.device.device.clone();
    let prompt: Vec<u32> = (0..32).map(|i| 1000 + i).collect();
    let config = ChatGenerationConfig::default();

    // Warm up (Metal pipeline compilation, pools) before any timing.
    let mut warm = model
        .begin_resumable_prefill_state_managed(&prompt, 64, &config, sessions.session(3))
        .unwrap();
    model
        .continue_resumable_prefill_managed(&mut warm, &prompt, 0, prompt.len())
        .unwrap();
    model.decode_step(&mut warm).unwrap();
    for _ in 0..16 {
        model.decode_step_batch(&mut [&mut warm]).unwrap();
    }
    drop(warm);

    // Forward only (synchronized once at the end).
    let mut state = model
        .begin_resumable_prefill_state_managed(&prompt, 1024, &config, sessions.session(0))
        .unwrap();
    model
        .continue_resumable_prefill_managed(&mut state, &prompt, 0, prompt.len())
        .unwrap();
    let started = Instant::now();
    let mut last = None;
    for position in prompt.len()..prompt.len() + steps() {
        let input = Tensor::from_slice(&[1234u32], (1, 1), &device).unwrap();
        last = model
            .text_model
            .forward_tokens_retained(
                &input,
                position,
                &mut state.physical_kv,
                &mut state.shortconv,
                true,
            )
            .unwrap();
    }
    let _ = last.unwrap().to_vec2::<f32>().unwrap();
    let forward_ms = started.elapsed().as_secs_f64() * 1000.0 / steps() as f64;

    // Forward plus a device synchronization per token, and the host time
    // spent inside the forward call (encoding) alone.
    let mut state = model
        .begin_resumable_prefill_state_managed(&prompt, 1024, &config, sessions.session(3))
        .unwrap();
    model
        .continue_resumable_prefill_managed(&mut state, &prompt, 0, prompt.len())
        .unwrap();
    let mut encode = std::time::Duration::ZERO;
    let started = Instant::now();
    for position in prompt.len()..prompt.len() + steps() {
        let input = Tensor::from_slice(&[1234u32], (1, 1), &device).unwrap();
        let call = Instant::now();
        let logits = model
            .text_model
            .forward_tokens_retained(
                &input,
                position,
                &mut state.physical_kv,
                &mut state.shortconv,
                true,
            )
            .unwrap()
            .unwrap();
        encode += call.elapsed();
        device.synchronize().unwrap();
        drop(logits);
    }
    let sync_ms = started.elapsed().as_secs_f64() * 1000.0 / steps() as f64;
    let encode_ms = encode.as_secs_f64() * 1000.0 / steps() as f64;
    eprintln!(
        "{backend}: forward+sync {sync_ms:.2} ms/token, host time inside forward {encode_ms:.2} ms"
    );

    // Forward plus a blocking argmax readback per token.
    let mut state = model
        .begin_resumable_prefill_state_managed(&prompt, 1024, &config, sessions.session(1))
        .unwrap();
    model
        .continue_resumable_prefill_managed(&mut state, &prompt, 0, prompt.len())
        .unwrap();
    let started = Instant::now();
    for position in prompt.len()..prompt.len() + steps() {
        let input = Tensor::from_slice(&[1234u32], (1, 1), &device).unwrap();
        let logits = model
            .text_model
            .forward_tokens_retained(
                &input,
                position,
                &mut state.physical_kv,
                &mut state.shortconv,
                true,
            )
            .unwrap()
            .unwrap();
        let finite = diagnostics::finite_flag(&logits).unwrap();
        let mut sampler = ChatSampler::new(config.clone(), &prompt);
        let vocab = logits.dim(D::Minus1).unwrap();
        let _ =
            sample_chat_rows_guarded(&mut [&mut sampler], &logits, vocab, Some(&finite)).unwrap();
    }
    let readback_ms = started.elapsed().as_secs_f64() * 1000.0 / steps() as f64;

    // The serving step: decode_step_batch with one row.
    let mut state = model
        .begin_resumable_prefill_state_managed(&prompt, 4096, &config, sessions.session(2))
        .unwrap();
    model
        .continue_resumable_prefill_managed(&mut state, &prompt, 0, prompt.len())
        .unwrap();
    model.decode_step(&mut state).unwrap();
    let started = Instant::now();
    let mut served = 0;
    for _ in 0..steps() {
        if state.finished {
            break;
        }
        model.decode_step_batch(&mut [&mut state]).unwrap();
        served += 1;
    }
    let serving_ms = started.elapsed().as_secs_f64() * 1000.0 / served.max(1) as f64;
    for _ in 0..profile_rounds() {
        let mut state = model
            .begin_resumable_prefill_state_managed(&prompt, 200, &config, sessions.session(2))
            .unwrap();
        model
            .continue_resumable_prefill_managed(&mut state, &prompt, 0, prompt.len())
            .unwrap();
        model.decode_step(&mut state).unwrap();
        while !state.finished {
            model.decode_step_batch(&mut [&mut state]).unwrap();
        }
    }
    eprintln!(
        "{backend}: forward {forward_ms:.2} ms/token ({:.0} tok/s), +argmax readback {readback_ms:.2} ms, decode_step_batch {serving_ms:.2} ms ({:.0} tok/s, {served} steps)",
        1000.0 / forward_ms,
        1000.0 / serving_ms
    );
}

#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF"]
fn lfm25_decode_step_timing_cpu() {
    bench(DeviceProfile::cpu());
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF"]
fn lfm25_decode_step_timing_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let _lease = crate::backends::metal_test_device_or_skip("lfm25_decode_step_timing_metal");
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if !profile.device.is_metal() {
        return;
    }
    bench(profile);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF and an NVIDIA GPU"]
fn lfm25_decode_step_timing_cuda() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Cuda).unwrap();
    if !profile.device.is_cuda() {
        return;
    }
    bench(profile);
}

/// GPU time of the decode-shaped (one row) quantized projections alone:
/// `iterations` back-to-back forwards, one synchronization at the end.
fn projection_bandwidth(device: &candle_core::Device) {
    use candle_core::quantized::QMatMul;
    use candle_core::Module;
    let path = models_dir_for_bench()
        .join("LFM2.5-1.2B-Instruct-GGUF")
        .join("LFM2.5-1.2B-Instruct-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }
    let loader = GgufLoader::from_path(&path).unwrap();
    for name in [
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
        "blk.2.ffn_down.weight",
        "blk.0.shortconv.in_proj.weight",
        "blk.0.shortconv.out_proj.weight",
        "blk.2.attn_k.weight",
        "token_embd.weight",
    ] {
        let qtensor = loader.load_qtensor(name, device).unwrap();
        let dtype = qtensor.dtype();
        let bytes = qtensor.storage_size_in_bytes();
        let (rows, cols) = qtensor.shape().dims2().unwrap();
        let matmul = QMatMul::from_qtensor(qtensor).unwrap();
        let input = Tensor::randn(0f32, 1f32, (1, cols), device).unwrap();
        for _ in 0..8 {
            let _ = matmul.forward(&input).unwrap();
        }
        device.synchronize().unwrap();
        let iterations = 200;
        let started = Instant::now();
        let mut last = None;
        for _ in 0..iterations {
            last = Some(matmul.forward(&input).unwrap());
        }
        device.synchronize().unwrap();
        drop(last);
        let micros = started.elapsed().as_secs_f64() * 1e6 / iterations as f64;
        eprintln!(
            "{name:34} {dtype:?} [{rows}x{cols}] {:6.2} MB {micros:8.1} us/call {:6.1} GB/s",
            bytes as f64 / 1e6,
            bytes as f64 / micros / 1e3
        );
    }
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF"]
fn lfm25_projection_bandwidth_metal() {
    let Some((device, _lease)) =
        crate::backends::metal_test_device_or_skip("lfm25_projection_bandwidth_metal")
    else {
        return;
    };
    projection_bandwidth(&device);
}

#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF"]
fn lfm25_projection_bandwidth_cpu() {
    projection_bandwidth(&candle_core::Device::Cpu);
}

/// GPU time of the per-step selection ops over one 65,536-wide logits row.
#[cfg(feature = "metal")]
#[test]
#[ignore = "timing"]
fn lfm25_selection_ops_metal() {
    let Some((device, _lease)) =
        crate::backends::metal_test_device_or_skip("lfm25_selection_ops_metal")
    else {
        return;
    };
    let logits = Tensor::randn(0f32, 1f32, (1, 65_536), &device).unwrap();
    let mask = Tensor::zeros(65_536, DType::F32, &device).unwrap();
    let time = |label: &str, op: &dyn Fn() -> Tensor| {
        for _ in 0..8 {
            let _ = op();
        }
        device.synchronize().unwrap();
        let started = Instant::now();
        let mut last = None;
        for _ in 0..200 {
            last = Some(op());
        }
        device.synchronize().unwrap();
        drop(last);
        eprintln!(
            "{label:24} {:8.1} us",
            started.elapsed().as_secs_f64() * 1e6 / 200.0
        );
    };
    time("mask broadcast_add", &|| {
        logits.broadcast_add(&mask).unwrap()
    });
    time("argmax", &|| logits.argmax(D::Minus1).unwrap());
    time("finite flag", &|| {
        diagnostics::finite_flag(&logits).unwrap()
    });
    time("max_all", &|| logits.max_all().unwrap());
}

/// Scalar vs one-row batched decode forward, interleaved step by step so
/// both see the same machine state; each step is synchronized.
fn scalar_vs_batched(profile: DeviceProfile) {
    let dir = models_dir_for_bench().join("LFM2.5-1.2B-Instruct-GGUF");
    if !dir.exists() {
        return;
    }
    let model = Lfm2ChatModel::load(&dir, ModelVariant::Lfm2512BInstructGguf, profile).unwrap();
    let sessions = Sessions::new(&model);
    let device = model.device.device.clone();
    let prompt: Vec<u32> = (0..32).map(|i| 1000 + i).collect();
    let config = ChatGenerationConfig::default();
    let new_state = |index| {
        let mut state = model
            .begin_resumable_prefill_state_managed(&prompt, 1024, &config, sessions.session(index))
            .unwrap();
        model
            .continue_resumable_prefill_managed(&mut state, &prompt, 0, prompt.len())
            .unwrap();
        state.unconsumed_output = None;
        state
    };
    let mut scalar = new_state(0);
    let mut batched = new_state(1);
    let (mut scalar_time, mut batched_time) =
        (std::time::Duration::ZERO, std::time::Duration::ZERO);
    let steps = steps();
    for step in 0..steps + 8 {
        let position = prompt.len() + step;
        let started = Instant::now();
        let input = Tensor::from_slice(&[1234u32], (1, 1), &device).unwrap();
        let logits = model
            .text_model
            .forward_tokens_retained(
                &input,
                position,
                &mut scalar.physical_kv,
                &mut scalar.shortconv,
                true,
            )
            .unwrap()
            .unwrap();
        let _ = logits.to_vec2::<f32>().unwrap();
        if step >= 8 {
            scalar_time += started.elapsed();
        }
        let started = Instant::now();
        let logits = model
            .text_model
            .forward_token_ids_retained_batch(
                &[1234],
                &[position],
                &mut [&mut batched.shortconv],
                &mut [&mut batched.physical_kv],
            )
            .unwrap();
        let _ = logits.to_vec2::<f32>().unwrap();
        if step >= 8 {
            batched_time += started.elapsed();
        }
    }
    let per = |time: std::time::Duration| time.as_secs_f64() * 1000.0 / steps as f64;
    eprintln!(
        "{:?}: scalar forward {:.2} ms/token, one-row batched forward {:.2} ms/token",
        model.device_kind(),
        per(scalar_time),
        per(batched_time)
    );
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF"]
fn lfm25_scalar_vs_batched_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let _lease = crate::backends::metal_test_device_or_skip("lfm25_scalar_vs_batched_metal");
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if profile.device.is_metal() {
        scalar_vs_batched(profile);
    }
}

#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF"]
fn lfm25_scalar_vs_batched_cpu() {
    scalar_vs_batched(DeviceProfile::cpu());
}

/// One-sync decode step (deferred KV commit, packed greedy readback) vs the
/// previous three-sync sequence (sealed forward, finite readback, argmax
/// readback), interleaved step by step on one row each.
fn one_sync_vs_three(profile: DeviceProfile) {
    let dir = models_dir_for_bench().join("LFM2.5-1.2B-Instruct-GGUF");
    if !dir.exists() {
        return;
    }
    let model = Lfm2ChatModel::load(&dir, ModelVariant::Lfm2512BInstructGguf, profile).unwrap();
    let sessions = Sessions::new(&model);
    let prompt: Vec<u32> = (0..32).map(|i| 1000 + i).collect();
    let config = ChatGenerationConfig::default();
    let new_state = |index| {
        let mut state = model
            .begin_resumable_prefill_state_managed(&prompt, 1024, &config, sessions.session(index))
            .unwrap();
        model
            .continue_resumable_prefill_managed(&mut state, &prompt, 0, prompt.len())
            .unwrap();
        state
    };
    let mut old = new_state(0);
    let mut new = new_state(1);
    let (mut old_time, mut new_time) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
    let steps = steps();
    for step in 0..steps + 8 {
        let position = prompt.len() + step;
        let started = Instant::now();
        let logits = model
            .text_model
            .forward_token_ids_retained_batch(
                &[1234],
                &[position],
                &mut [&mut old.shortconv],
                &mut [&mut old.physical_kv],
            )
            .unwrap();
        model.tokenizer.validate_logits(&logits).unwrap();
        let masked = model.mask_non_generatable(&logits).unwrap();
        let _ = masked.argmax(D::Minus1).unwrap().to_vec1::<u32>().unwrap();
        if step >= 8 {
            old_time += started.elapsed();
        }
        let started = Instant::now();
        let (logits, pending) = model
            .text_model
            .forward_token_ids_retained_batch_deferred(
                &[1234],
                &[position],
                &[&new.shortconv],
                &[&new.physical_kv],
            )
            .unwrap();
        let finite = diagnostics::finite_flag(&logits).unwrap();
        let masked = model.mask_non_generatable(&logits).unwrap();
        let mut sampler = ChatSampler::new(config.clone(), &prompt);
        let _ = sample_chat_rows_guarded(
            &mut [&mut sampler],
            &masked,
            model.tokenizer.vocab_size,
            Some(&finite),
        )
        .unwrap();
        pending
            .commit(&mut [&mut new.shortconv], &mut [&mut new.physical_kv])
            .unwrap();
        if step >= 8 {
            new_time += started.elapsed();
        }
    }
    let per = |time: std::time::Duration| time.as_secs_f64() * 1000.0 / steps as f64;
    eprintln!(
        "{:?}: three-sync step {:.2} ms/token, one-sync step {:.2} ms/token",
        model.device_kind(),
        per(old_time),
        per(new_time)
    );
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF"]
fn lfm25_one_sync_vs_three_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let _lease = crate::backends::metal_test_device_or_skip("lfm25_one_sync_vs_three_metal");
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if profile.device.is_metal() {
        one_sync_vs_three(profile);
    }
}

/// Fused ShortConv decode step vs the op-by-op path, interleaved step by
/// step on two rows of the same model (everything else identical).
fn fused_vs_unfused_shortconv(profile: DeviceProfile) {
    use super::super::backbone::FORCE_UNFUSED_SHORTCONV;
    let dir = models_dir_for_bench().join("LFM2.5-1.2B-Instruct-GGUF");
    if !dir.exists() {
        return;
    }
    let model = Lfm2ChatModel::load(&dir, ModelVariant::Lfm2512BInstructGguf, profile).unwrap();
    let sessions = Sessions::new(&model);
    let prompt: Vec<u32> = (0..32).map(|i| 1000 + i).collect();
    let config = ChatGenerationConfig::default();
    let new_state = |index| {
        let mut state = model
            .begin_resumable_prefill_state_managed(&prompt, 1024, &config, sessions.session(index))
            .unwrap();
        model
            .continue_resumable_prefill_managed(&mut state, &prompt, 0, prompt.len())
            .unwrap();
        model.decode_step(&mut state).unwrap();
        state
    };
    let mut fused = new_state(0);
    let mut unfused = new_state(1);
    let (mut fused_time, mut unfused_time) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
    let steps = steps();
    for step in 0..steps + 8 {
        FORCE_UNFUSED_SHORTCONV.with(|flag| flag.set(true));
        let started = Instant::now();
        model.decode_step_batch(&mut [&mut unfused]).unwrap();
        if step >= 8 {
            unfused_time += started.elapsed();
        }
        FORCE_UNFUSED_SHORTCONV.with(|flag| flag.set(false));
        let started = Instant::now();
        model.decode_step_batch(&mut [&mut fused]).unwrap();
        if step >= 8 {
            fused_time += started.elapsed();
        }
    }
    let per = |time: std::time::Duration| time.as_secs_f64() * 1000.0 / steps as f64;
    eprintln!(
        "{:?}: op-by-op ShortConv {:.2} ms/token, fused ShortConv {:.2} ms/token",
        model.device_kind(),
        per(unfused_time),
        per(fused_time)
    );
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF"]
fn lfm25_fused_vs_unfused_shortconv_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let _lease =
        crate::backends::metal_test_device_or_skip("lfm25_fused_vs_unfused_shortconv_metal");
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if profile.device.is_metal() {
        fused_vs_unfused_shortconv(profile);
    }
}

#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF"]
fn lfm25_fused_vs_unfused_shortconv_cpu() {
    fused_vs_unfused_shortconv(DeviceProfile::cpu());
}

/// B decode rows through one quantized projection, shaped `(B, 1, k)` (Candle
/// mat-vec, one dispatch per row) against `(1, B, k)` (Candle mat-mat, one
/// dispatch), with the largest difference between the two results.
fn batched_projection_shapes(device: &candle_core::Device) {
    use candle_core::quantized::QMatMul;
    use candle_core::Module;
    let path = models_dir_for_bench()
        .join("LFM2.5-1.2B-Instruct-GGUF")
        .join("LFM2.5-1.2B-Instruct-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }
    let loader = GgufLoader::from_path(&path).unwrap();
    for name in [
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
        "blk.2.ffn_down.weight",
        "blk.0.shortconv.out_proj.weight",
        "blk.2.attn_k.weight",
        "token_embd.weight",
    ] {
        let qtensor = loader.load_qtensor(name, device).unwrap();
        let dtype = qtensor.dtype();
        let (_, cols) = qtensor.shape().dims2().unwrap();
        let matmul = QMatMul::from_qtensor(qtensor).unwrap();
        for batch in [1usize, 2, 4, 8] {
            let rows = Tensor::randn(0f32, 1f32, (batch, 1, cols), device).unwrap();
            let seq = rows.reshape((1, batch, cols)).unwrap();
            let time = |input: &Tensor| {
                for _ in 0..4 {
                    let _ = matmul.forward(input).unwrap();
                }
                device.synchronize().unwrap();
                let started = Instant::now();
                for _ in 0..100 {
                    let _ = matmul.forward(input).unwrap();
                }
                device.synchronize().unwrap();
                started.elapsed().as_secs_f64() * 1e6 / 100.0
            };
            let (mv, mm) = (time(&rows), time(&seq));
            let a = matmul.forward(&rows).unwrap().flatten_all().unwrap();
            let b = matmul.forward(&seq).unwrap().flatten_all().unwrap();
            let diff = (a - b)
                .unwrap()
                .abs()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            eprintln!(
                "{name:32} {dtype:?} B={batch}: rows {mv:8.1} us, sequence {mm:8.1} us, max diff {diff:.2e}"
            );
        }
    }
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF"]
fn lfm25_batched_projection_shapes_metal() {
    if let Some((device, _lease)) =
        crate::backends::metal_test_device_or_skip("lfm25_batched_projection_shapes_metal")
    {
        batched_projection_shapes(&device);
    }
}

#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF"]
fn lfm25_batched_projection_shapes_cpu() {
    batched_projection_shapes(&candle_core::Device::Cpu);
}

/// izwi's Metal k-quant mat-vec against Candle's per-row mat-vec on the
/// checkpoint's own Q4_K and Q6_K weights, for 1..=8 decode rows. Synthetic
/// quantized weights missed a half-precision scale bug that these catch.
#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF"]
fn lfm25_kquant_mat_vec_matches_candle_on_real_weights() {
    use candle_core::quantized::{QMatMul, QStorage, QTensor};
    use candle_core::Module;
    let path = models_dir_for_bench()
        .join("LFM2.5-1.2B-Instruct-GGUF")
        .join("LFM2.5-1.2B-Instruct-Q4_K_M.gguf");
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "lfm25_kquant_mat_vec_matches_candle_on_real_weights",
    ) else {
        return;
    };
    if !path.exists() {
        return;
    }
    let loader = GgufLoader::from_path(&path).unwrap();
    for name in [
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
        "blk.2.ffn_down.weight",
        "blk.2.attn_v.weight",
        "token_embd.weight",
    ] {
        let host = loader
            .load_qtensor(name, &candle_core::Device::Cpu)
            .unwrap();
        let dtype = host.dtype();
        let (rows, cols) = host.shape().dims2().unwrap();
        let storage = QStorage::from_data(host.data().unwrap(), &device, dtype).unwrap();
        let QStorage::Metal(metal) = &storage else {
            unreachable!("Metal storage")
        };
        let kernel = crate::kernels::metal_kquant_mv::MetalKQuantMatVec::new(
            metal.buffer().clone(),
            dtype,
            rows,
            cols,
        )
        .unwrap();
        let matmul = QMatMul::from_qtensor(QTensor::new(storage, (rows, cols)).unwrap()).unwrap();
        for batch in [2usize, 5, 8] {
            let x = Tensor::randn(0f32, 1f32, (batch, 1, cols), &device).unwrap();
            let expected = matmul.forward(&x).unwrap().flatten_all().unwrap();
            let actual = kernel.forward(&x).unwrap().flatten_all().unwrap();
            let expected = expected.to_vec1::<f32>().unwrap();
            let actual = actual.to_vec1::<f32>().unwrap();
            let scale = expected.iter().fold(0f32, |max, e| max.max(e.abs()));
            let worst = actual
                .iter()
                .zip(&expected)
                .map(|(a, e)| (a - e).abs())
                .fold(0f32, f32::max);
            eprintln!("{name} {dtype:?} batch {batch}: worst {worst:.3e} of scale {scale:.3e}");
            assert!(
                worst <= 1e-5 * scale,
                "{name} batch {batch}: {worst} vs scale {scale}"
            );
        }
    }
}

/// Aggregate decode throughput by batch width: B rows decode together in one
/// batched step, for B = 1, 2, 4 and 8, interleaved step by step.
fn batch_width_scaling(profile: DeviceProfile) {
    let dir = models_dir_for_bench().join("LFM2.5-1.2B-Instruct-GGUF");
    if !dir.exists() {
        return;
    }
    let model = Lfm2ChatModel::load(&dir, ModelVariant::Lfm2512BInstructGguf, profile).unwrap();
    let widths = [1usize, 2, 4, 8];
    let sessions: Vec<Sessions> = widths.iter().map(|_| Sessions::new(&model)).collect();
    let config = ChatGenerationConfig::default();
    let mut groups: Vec<Vec<ChatDecodeState>> = widths
        .iter()
        .zip(&sessions)
        .map(|(&width, sessions)| {
            (0..width)
                .map(|row| {
                    // A real request per row, long enough that no row ends
                    // inside the timed steps.
                    let prompt = model
                        .prompt_token_ids(&[ChatMessage {
                            role: ChatRole::User,
                            content: format!(
                                "Write a detailed 500-word essay on topic number {row}: \
                                 the history of bridges."
                            ),
                        }])
                        .unwrap();
                    let mut state = model
                        .begin_resumable_prefill_state_managed(
                            &prompt,
                            1024,
                            &config,
                            sessions.session(row as u32),
                        )
                        .unwrap();
                    model
                        .continue_resumable_prefill_managed(&mut state, &prompt, 0, prompt.len())
                        .unwrap();
                    model.decode_step(&mut state).unwrap();
                    state
                })
                .collect()
        })
        .collect();
    // Each width runs with the small-batch mat-vec (where the backend has
    // it) and with Candle's per-row mat-vec, interleaved.
    use super::super::backbone::FORCE_CANDLE_MATVEC;
    let mut times = vec![[std::time::Duration::ZERO; 2]; widths.len()];
    let steps = steps();
    for step in 0..steps + 8 {
        for (group, time) in groups.iter_mut().zip(&mut times) {
            for (candle, slot) in [(false, 0), (true, 1)] {
                FORCE_CANDLE_MATVEC.with(|flag| flag.set(candle));
                let mut rows: Vec<&mut ChatDecodeState> = group.iter_mut().collect();
                let started = Instant::now();
                model.decode_step_batch(&mut rows).unwrap();
                if step >= 8 {
                    time[slot] += started.elapsed();
                }
            }
        }
    }
    FORCE_CANDLE_MATVEC.with(|flag| flag.set(false));
    for (&width, time) in widths.iter().zip(&times) {
        let per = |time: std::time::Duration| time.as_secs_f64() * 1000.0 / steps as f64;
        let (small, candle) = (per(time[0]), per(time[1]));
        eprintln!(
            "{:?}: batch {width}: small-batch {small:.2} ms/step ({:.1} tok/s), Candle per-row {candle:.2} ms/step ({:.1} tok/s)",
            model.device_kind(),
            width as f64 * 1000.0 / small,
            width as f64 * 1000.0 / candle
        );
    }
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF"]
fn lfm25_batch_width_scaling_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let _lease = crate::backends::metal_test_device_or_skip("lfm25_batch_width_scaling_metal");
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if profile.device.is_metal() {
        batch_width_scaling(profile);
    }
}

#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF"]
fn lfm25_batch_width_scaling_cpu() {
    batch_width_scaling(DeviceProfile::cpu());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF and an NVIDIA GPU"]
fn lfm25_batch_width_scaling_cuda() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Cuda).unwrap();
    if profile.device.is_cuda() {
        batch_width_scaling(profile);
    }
}

/// Decode time with F32 against F16 KV storage, interleaved token by token,
/// at a short and a `IZWI_BENCH_CONTEXT`-token (default 4,096) context.
fn kv_dtype_decode(profile: DeviceProfile) {
    let dir = models_dir_for_bench().join("LFM2.5-1.2B-Instruct-GGUF");
    if !dir.exists() {
        return;
    }
    let model = Lfm2ChatModel::load(&dir, ModelVariant::Lfm2512BInstructGguf, profile).unwrap();
    let long = std::env::var("IZWI_BENCH_CONTEXT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4096);
    let pages = (long + steps() + 64).div_ceil(16) as u32;
    let f32_sessions = Sessions::sized(&model, DType::F32, pages);
    let f16_sessions = Sessions::sized(&model, DType::F16, pages);
    let config = ChatGenerationConfig::default();
    for context in [32, long] {
        let prompt: Vec<u32> = (0..context)
            .map(|i| 1000 + (i as u32 * 7) % 20_000)
            .collect();
        let new_state = |sessions: &Sessions| {
            let mut state = model
                .begin_resumable_prefill_state_managed(&prompt, 1024, &config, sessions.session(0))
                .unwrap();
            model
                .continue_resumable_prefill_managed(&mut state, &prompt, 0, prompt.len())
                .unwrap();
            model.decode_step(&mut state).unwrap();
            state
        };
        let mut f32_state = new_state(&f32_sessions);
        let mut f16_state = new_state(&f16_sessions);
        let (mut f32_time, mut f16_time) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
        let steps = steps();
        for step in 0..steps + 8 {
            let started = Instant::now();
            model.decode_step_batch(&mut [&mut f32_state]).unwrap();
            if step >= 8 {
                f32_time += started.elapsed();
            }
            let started = Instant::now();
            model.decode_step_batch(&mut [&mut f16_state]).unwrap();
            if step >= 8 {
                f16_time += started.elapsed();
            }
        }
        let per = |time: std::time::Duration| time.as_secs_f64() * 1000.0 / steps as f64;
        eprintln!(
            "{:?} context {context}: F32 KV {:.2} ms/token, F16 KV {:.2} ms/token",
            model.device_kind(),
            per(f32_time),
            per(f16_time)
        );
    }
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF"]
fn lfm25_kv_dtype_decode_metal() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let _lease = crate::backends::metal_test_device_or_skip("lfm25_kv_dtype_decode_metal");
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
    if profile.device.is_metal() {
        kv_dtype_decode(profile);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF and an NVIDIA GPU"]
fn lfm25_kv_dtype_decode_cuda() {
    use crate::backends::{BackendPreference, DeviceSelector};
    let profile = DeviceSelector::detect_for_preference(BackendPreference::Cuda).unwrap();
    if profile.device.is_cuda() {
        kv_dtype_decode(profile);
    }
}

/// Prefill time for a long prompt, one span, so CPU prefill can be split
/// between matmuls and the ShortConv ring path.
fn prefill_timing(profile: DeviceProfile, tokens: usize) {
    let dir = models_dir_for_bench().join("LFM2.5-1.2B-Instruct-GGUF");
    if !dir.exists() {
        return;
    }
    let model = Lfm2ChatModel::load(&dir, ModelVariant::Lfm2512BInstructGguf, profile).unwrap();
    let backend = format!("{:?}", model.device_kind());
    let device = model.device.device.clone();
    let layout = super::super::physical::Lfm2StateLayout::from_config(&model.config).unwrap();
    let bindings: Vec<crate::kv::KvLayerBinding> = layout
        .attention_model_layers
        .iter()
        .enumerate()
        .map(|(physical, model_layer)| crate::kv::KvLayerBinding {
            model_layer: *model_layer,
            physical_layer: physical as u32,
        })
        .collect();
    let _ = (&bindings, &device);
    let sessions = Sessions::new(&model);
    let prompt: Vec<u32> = (0..tokens as u32)
        .map(|i| 1000 + (i * 7919) % 50_000)
        .collect();
    let config = ChatGenerationConfig::default();
    for round in 0..2 + profile_rounds() {
        let mut state = model
            .begin_resumable_prefill_state_managed(
                &prompt,
                16,
                &config,
                sessions.session(round as u32 % 2),
            )
            .unwrap();
        let started = Instant::now();
        model
            .continue_resumable_prefill_managed(&mut state, &prompt, 0, prompt.len())
            .unwrap();
        let _ = state
            .unconsumed_output
            .take()
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let seconds = started.elapsed().as_secs_f64();
        eprintln!(
            "{backend}: prefill {tokens} tokens in {:.0} ms ({:.0} tok/s), round {round}",
            seconds * 1000.0,
            tokens as f64 / seconds
        );
    }
}

#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF"]
fn lfm25_prefill_timing_cpu() {
    prefill_timing(DeviceProfile::cpu(), 200);
}

/// CPU prefill matmul: Candle's quantized `QMatMul` vs expanding the weight
/// to F32 tile by tile and using the F32 GEMM, at prefill row counts.
#[test]
#[ignore = "needs the LFM2.5-1.2B GGUF"]
fn lfm25_cpu_prefill_matmul_strategies() {
    use candle_core::quantized::{ggml_file::qtensor_from_ggml, QMatMul};
    use candle_core::Module;
    let path = models_dir_for_bench()
        .join("LFM2.5-1.2B-Instruct-GGUF")
        .join("LFM2.5-1.2B-Instruct-Q4_K_M.gguf");
    if !path.exists() {
        return;
    }
    let device = candle_core::Device::Cpu;
    let loader = GgufLoader::from_path(&path).unwrap();
    for name in ["blk.2.ffn_gate.weight", "blk.0.ffn_down.weight"] {
        let qtensor = loader.load_qtensor(name, &device).unwrap();
        let dtype = qtensor.dtype();
        let (n, k) = qtensor.shape().dims2().unwrap();
        let data = qtensor.data().unwrap().to_vec();
        let matmul = QMatMul::from_qtensor(qtensor).unwrap();
        for m in [16usize, 64, 256, 512] {
            let x = Tensor::randn(0f32, 1f32, (m, k), &device).unwrap();
            let started = Instant::now();
            let reference = matmul.forward(&x).unwrap();
            let quantized_ms = started.elapsed().as_secs_f64() * 1000.0;
            let started = Instant::now();
            let tile_rows = 512;
            let row_bytes = data.len() / n;
            let mut tiles = Vec::new();
            for start in (0..n).step_by(tile_rows) {
                let rows = tile_rows.min(n - start);
                let tile = qtensor_from_ggml(
                    dtype,
                    &data[start * row_bytes..(start + rows) * row_bytes],
                    vec![rows, k],
                    &device,
                )
                .unwrap()
                .dequantize(&device)
                .unwrap();
                tiles.push(x.matmul(&tile.t().unwrap()).unwrap());
            }
            let tiled = Tensor::cat(&tiles, 1).unwrap();
            let tiled_ms = started.elapsed().as_secs_f64() * 1000.0;
            let err = (&tiled - &reference)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            eprintln!(
                "{name} {dtype:?} m={m:4}: QMatMul {quantized_ms:7.1} ms, F32 tiles {tiled_ms:7.1} ms, max diff {err:.4}"
            );
        }
    }
}
