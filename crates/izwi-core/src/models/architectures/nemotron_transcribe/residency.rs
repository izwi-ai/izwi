//! What a load holds on each backend, priced from the config's census
//! before any weight is read, so load admission can refuse a machine that
//! cannot hold the model with a clear message instead of failing mid-load.
//!
//! | Weight | CPU | Metal | CUDA |
//! |---|---|---|---|
//! | FP8 linears | Q8_0 (34 B per 32) | E4M3 bytes + scale grid | E4M3 bytes padded to 64×128 + scale grid |
//! | Routed experts (`w8a16`) | Q8_0 | E4M3 bytes stacked per layer, one scale | the same as Metal |
//! | BF16 linears and tables | F32 | F32 | BF16 |
//! | Router gates, norms, Mamba parameters | F32 | F32 | F32 |
//! | Language-model head | Q8_0 | F16 | BF16 |
//! | Embedding table | BF16, host | BF16, host | BF16, host |
//!
//! The MTP head is never loaded and the CTC head is priced only when its
//! file is present. In the `vllm-fp8` parity mode the routed experts stay
//! per-expert FP8 linears, padded on CUDA like the others.

use super::checkpoint::{expected_census, CensusDtype, CtcGeometry};
use super::config::NemotronTranscribeConfig;
use super::precision::Fp8Mode;

/// A load's resident bytes, by memory domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NemotronResidency {
    /// Weights the device holds (all of them on the CPU).
    pub device_bytes: u64,
    /// Weights kept in host memory (the embedding table).
    pub host_bytes: u64,
    /// The largest single tensor a load converts at once (host staging).
    pub largest_tensor_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyBackend {
    Cpu,
    Metal,
    Cuda,
}

/// Whether a load stacks each layer's routed experts for the device-routed
/// kernels (`kernels::moe_relu2`): on Metal and CUDA in `w8a16`.
pub fn stacks_routed_experts(backend: ResidencyBackend, mode: Fp8Mode) -> bool {
    backend != ResidencyBackend::Cpu && mode == Fp8Mode::W8A16
}

/// The resident weights of `config` on `backend` in `mode`.
pub fn residency(
    config: &NemotronTranscribeConfig,
    ctc: Option<&CtcGeometry>,
    backend: ResidencyBackend,
    mode: Fp8Mode,
) -> NemotronResidency {
    let mut out = NemotronResidency::default();
    let stacked = stacks_routed_experts(backend, mode);
    let census = expected_census(config, ctc);
    let dims = |name: &str| -> Option<(u64, u64)> {
        census
            .iter()
            .find(|e| e.name == name)
            .and_then(|e| (e.shape.len() == 2).then(|| (e.shape[0] as u64, e.shape[1] as u64)))
    };
    for entry in &census {
        let elements = entry.shape.iter().product::<usize>().max(1) as u64;
        let name = entry.name.as_str();
        let bytes = if name.ends_with(".input_scale")
            || name.ends_with(".k_scale")
            || name.ends_with(".v_scale")
            || name.ends_with(".weight_scale")
            || name.ends_with(".featurizer.fb")
            || name.ends_with(".featurizer.window")
            || name.ends_with("model_proto")
            || name.ends_with("upsample_factor")
            || name.ends_with("hidden_to_spks.weight") && !name.contains("single")
            || name.ends_with("hidden_to_spks.bias") && !name.contains("single")
        {
            // Validated and dropped, read on the host, or unused.
            0
        } else if name == "backbone.embeddings.weight" {
            out.host_bytes += elements * 2;
            0
        } else if name == "lm_head.weight" {
            match backend {
                ResidencyBackend::Cpu => elements / 32 * 34,
                ResidencyBackend::Metal | ResidencyBackend::Cuda => elements * 2,
            }
        } else if entry.dtype == CensusDtype::F8E4M3 && stacked && name.contains(".mixer.experts.")
        {
            // One stacked projection per layer is staged on the host whole.
            let staged = elements * config.decoder.moe.experts as u64;
            out.largest_tensor_bytes = out.largest_tensor_bytes.max(staged);
            elements + 4
        } else if entry.dtype == CensusDtype::F8E4M3 {
            let (rows, cols) = dims(name).unwrap_or((elements, 1));
            match backend {
                ResidencyBackend::Cpu => rows * cols / 32 * 34,
                ResidencyBackend::Metal => {
                    rows * cols + 4 * rows.div_ceil(128) * cols.div_ceil(128)
                }
                ResidencyBackend::Cuda => {
                    let (r, c) = (rows.next_multiple_of(64), cols.next_multiple_of(128));
                    r * c + 4 * r.div_ceil(128) * c.div_ceil(128)
                }
            }
        } else if entry.shape.len() == 2
            && !name.contains(".gate.weight")
            && backend == ResidencyBackend::Cuda
        {
            // Dense matrices stay BF16 on CUDA; vectors and the router F32.
            elements * 2
        } else {
            elements * 4
        };
        out.device_bytes += bytes;
        out.largest_tensor_bytes = out.largest_tensor_bytes.max(elements * 4);
    }
    if backend == ResidencyBackend::Cpu {
        // The CPU's host is its device.
        out.device_bytes += out.host_bytes;
        out.host_bytes = 0;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_published_model_needs_about_32_gib_on_every_backend() {
        let config = super::super::config::tests::published();
        let gib = |bytes: u64| bytes as f64 / (1u64 << 30) as f64;
        let mode = Fp8Mode::W8A16;
        let cpu = residency(&config, None, ResidencyBackend::Cpu, mode);
        let metal = residency(&config, None, ResidencyBackend::Metal, mode);
        let cuda = residency(&config, None, ResidencyBackend::Cuda, mode);
        // About 32.6 GiB as CPU Q8_0; as raw FP8, 30.4 GiB on Metal and
        // 30.2 GiB on CUDA plus the 0.66 GiB host embedding table (without
        // the MTP head, the CTC head priced separately).
        assert!(
            (31.0..35.0).contains(&gib(cpu.device_bytes)),
            "{}",
            gib(cpu.device_bytes)
        );
        assert!(
            (29.0..33.0).contains(&gib(metal.device_bytes)),
            "{}",
            gib(metal.device_bytes)
        );
        assert!(
            (29.0..33.0).contains(&gib(cuda.device_bytes)),
            "{}",
            gib(cuda.device_bytes)
        );
        // The stacked experts are unpadded on both devices, so CUDA's BF16
        // dense matrices (F32 on Metal) outweigh its padded other linears.
        assert!(cuda.device_bytes < metal.device_bytes);
        // The parity mode keeps per-expert linears, their down projections
        // padded on CUDA (1856 -> 1920 columns): 128 experts x 23 layers x
        // 2688 x 64 bytes, about 0.47 GiB more.
        let cuda_parity = residency(&config, None, ResidencyBackend::Cuda, Fp8Mode::VllmFp8);
        let extra = gib(cuda_parity.device_bytes) - gib(cuda.device_bytes);
        assert!((0.45..0.5).contains(&extra), "{extra}");
        // A stacked projection (128 experts) is staged on the host at load.
        assert!(cuda.largest_tensor_bytes >= 128 * 1856 * 2688);
        assert!((gib(metal.host_bytes) - 0.656).abs() < 0.01);
        assert_eq!(cpu.host_bytes, 0);
    }
}
