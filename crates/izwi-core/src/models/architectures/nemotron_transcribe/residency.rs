//! What a load holds on each backend, priced from the config's census
//! before any weight is read, so load admission can refuse a machine that
//! cannot hold the model with a clear message instead of failing mid-load.
//!
//! | Weight | CPU | Metal | CUDA |
//! |---|---|---|---|
//! | FP8 linears | Q8_0 (34 B per 32) | E4M3 bytes + scale grid | E4M3 bytes padded to 64×128 + scale grid |
//! | BF16 linears and tables | F32 | F32 | BF16 |
//! | Router gates, norms, Mamba parameters | F32 | F32 | F32 |
//! | Language-model head | Q8_0 | F16 | BF16 |
//! | Embedding table | BF16, host | BF16, host | BF16, host |
//!
//! The MTP head is never loaded and the CTC head is priced only when its
//! file is present.

use super::checkpoint::{expected_census, CensusDtype, CtcGeometry};
use super::config::NemotronTranscribeConfig;

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

/// The resident weights of `config` on `backend`.
pub fn residency(
    config: &NemotronTranscribeConfig,
    ctc: Option<&CtcGeometry>,
    backend: ResidencyBackend,
) -> NemotronResidency {
    let mut out = NemotronResidency::default();
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
        let cpu = residency(&config, None, ResidencyBackend::Cpu);
        let metal = residency(&config, None, ResidencyBackend::Metal);
        let cuda = residency(&config, None, ResidencyBackend::Cuda);
        // Plan §4.1: about 33.6 GiB as CPU Q8_0 and 31.6 GiB as raw FP8
        // (without the MTP head, the CTC head priced separately).
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
        // CUDA pads the experts' down projections (1856 -> 1920 columns).
        assert!(cuda.device_bytes > metal.device_bytes);
        assert!((gib(metal.host_bytes) - 0.656).abs() < 0.01);
        assert_eq!(cpu.host_bytes, 0);
    }
}
