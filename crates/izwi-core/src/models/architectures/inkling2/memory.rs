//! What a loaded Inkling 2 model holds, read from the checkpoint header.
//!
//! * The **experts** stay in their stored form (NVFP4 codes with FP8 block
//!   scales, MXFP8 values with scale bytes) and are read from the mapped
//!   file on every backend; they are host (or unified) memory, never copied.
//! * Every **dense** tensor, the text trunk's and the vision tower's, is
//!   held on the device the model is loaded on, the unembedding without its
//!   padded rows. The large matrices (`precision::is_projection`) are held
//!   in the backend's projection dtype: F32 on the CPU, F16 on Metal (bar a
//!   matrix whose input width is not a multiple of 16; the published model
//!   has none), F32 on CUDA or BF16 when asked for. Everything else is F32. The routers' weights and biases are
//!   host F32.
//! * The MTP head is not loaded.
//!
//! The footprint is a function of the config, the checkpoint header and the
//! projection dtype, so it is known before a weight is read. A projection
//! that F16 cannot hold stays F32 on Metal (the load warns); that is not
//! known from the header and such a load holds more than is priced here.

use std::path::Path;

use super::checkpoint::{expected_census, Inkling2Checkpoint, Stored, TEXT_PREFIX, VISION_PREFIX};
use candle_core::DType;

use super::config::Inkling2Config;
use super::precision::{held_element_bytes, projection_dtype};
use super::quant::ExpertFormat;
use crate::backends::BackendKind;
use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inkling2Footprint {
    /// Stored bytes of every expert tensor.
    pub expert_bytes: u64,
    /// Bytes of the dense tensors on the device as they are held (the
    /// projections in the projection dtype, the rest in F32), the vision
    /// tower's among them.
    pub dense_bytes: u64,
    /// The vision tower's share of `dense_bytes`.
    pub vision_bytes: u64,
    /// F32 bytes of the routers on the host.
    pub router_bytes: u64,
    /// Bytes of the largest dense tensor as it is held (the embedding).
    pub largest_dense_bytes: u64,
}

impl Inkling2Footprint {
    /// The footprint of `checkpoint` with its projections held in
    /// `projection`.
    pub fn of(config: &Inkling2Config, checkpoint: &Inkling2Checkpoint, projection: DType) -> Self {
        Self::of_census(config, checkpoint.expert_formats(), projection)
    }

    /// [`Self::of`] from the expert format of every layer.
    pub fn of_census(config: &Inkling2Config, formats: &[ExpertFormat], projection: DType) -> Self {
        let mut footprint = Self {
            expert_bytes: 0,
            dense_bytes: 0,
            vision_bytes: 0,
            router_bytes: 0,
            largest_dense_bytes: 0,
        };
        let unembedding = format!("{TEXT_PREFIX}.unembed.weight");
        for entry in expected_census(config, formats) {
            let vision = entry.name.starts_with(VISION_PREFIX);
            if !vision && !entry.name.starts_with(TEXT_PREFIX) {
                continue;
            }
            let elements = entry.shape.iter().product::<usize>() as u64;
            match entry.stored {
                Stored::Dense | Stored::DenseF32 => {
                    let elements = if entry.name == unembedding {
                        (config.unpadded_vocab * config.hidden) as u64
                    } else {
                        elements
                    };
                    if entry.name.contains(".mlp.gate.") {
                        footprint.router_bytes += elements * 4;
                    } else {
                        let bytes =
                            elements * held_element_bytes(&entry.name, &entry.shape, projection);
                        footprint.dense_bytes += bytes;
                        footprint.largest_dense_bytes = footprint.largest_dense_bytes.max(bytes);
                        if vision {
                            footprint.vision_bytes += bytes;
                        }
                    }
                }
                // Expert codes, scales and per-expert factors.
                Stored::U8 | Stored::Fp8 => footprint.expert_bytes += elements,
                Stored::F32 => footprint.expert_bytes += elements * 4,
            }
        }
        footprint
    }

    /// The footprint of the checkpoint in `model_dir` as a load on `backend`
    /// holds it, read from the header alone.
    pub fn load(model_dir: &Path, backend: BackendKind) -> Result<Self> {
        let config = Inkling2Config::load(model_dir)?;
        let checkpoint = Inkling2Checkpoint::open(model_dir, &config)?;
        let mut footprint = Self::of(&config, &checkpoint, projection_dtype(backend));
        // The drafter, when the load will serve it, is dense weights in F32
        // beside the trunk's. Its keys and values are paged state, which the
        // engine prices per token.
        let drafter = model_dir.join(super::checkpoint::MTP_FILE);
        if super::chat::drafter_asked() && drafter.is_file() {
            footprint.dense_bytes += super::drafter::held_bytes(&drafter)?;
        }
        Ok(footprint)
    }

    /// Bytes on the host that stay for the model's lifetime: the routers,
    /// and the mapped experts unless they run on the device.
    pub fn host_resident_bytes(&self, experts_on_device: bool) -> u64 {
        self.router_bytes
            + if experts_on_device {
                0
            } else {
                self.expert_bytes
            }
    }

    /// Bytes in the load domain (host memory on the CPU, unified memory on
    /// Metal, device memory on CUDA) after the load. `experts_on_device`:
    /// the packed experts and a copy of the routers are uploaded.
    pub fn resident_bytes(&self, backend: BackendKind, experts_on_device: bool) -> u64 {
        let device_experts = if experts_on_device {
            self.expert_bytes + self.router_bytes
        } else {
            0
        };
        match backend {
            // One memory: the mapped experts count beside a device copy.
            // (With the experts on the device the mapping is no longer read
            // and the system may drop its pages, but the load holds both.)
            BackendKind::Cpu | BackendKind::Metal => {
                self.dense_bytes + self.expert_bytes + self.router_bytes + device_experts
            }
            BackendKind::Cuda => self.dense_bytes + device_experts,
        }
    }

    /// The load's high point in the load domain. Beside what stays, one
    /// dense tensor at a time is in flight, and its flight never takes more
    /// than twice its held size beyond its final copy where host and device
    /// share memory, or its held size on each side where they do not:
    ///
    /// * an F32 tensor passes through a host F32 buffer and, for a
    ///   projection, a transposed device copy;
    /// * an F16 projection (Metal) is converted on the host, the F32 values
    ///   and the F16 copy together three times its held size before its
    ///   device copy exists, then uploaded;
    /// * a BF16 projection (CUDA, on request) uploads the stored bytes through a host
    ///   copy of the same size and is transposed on the device.
    pub fn load_peak_bytes(&self, backend: BackendKind, experts_on_device: bool) -> u64 {
        let transient = match backend {
            BackendKind::Cpu | BackendKind::Metal => 2 * self.largest_dense_bytes,
            // The host buffer is not device memory.
            BackendKind::Cuda => self.largest_dense_bytes,
        };
        self.resident_bytes(backend, experts_on_device) + transient
    }

    /// CUDA: host bytes at the load's high point. Uploaded experts are read
    /// through the mapping, which the host can drop again.
    pub fn cuda_host_load_peak_bytes(&self, experts_on_device: bool) -> u64 {
        self.host_resident_bytes(experts_on_device) + self.largest_dense_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::super::checkpoint::ExpertProjection;
    use super::super::config::tests::{fixture, published, tiny};
    use super::super::quant::ExpertFormat;
    use super::*;

    #[test]
    fn the_tiny_checkpoint_is_priced_from_its_header() {
        let dir = fixture("inkling2_golden");
        let config = tiny();
        let checkpoint = Inkling2Checkpoint::open(&dir, &config).unwrap();
        let footprint = Inkling2Footprint::load(&dir, BackendKind::Cpu).unwrap();
        assert_eq!(
            footprint,
            Inkling2Footprint::of(&config, &checkpoint, DType::F32)
        );
        // The experts are what the packed stacks report holding.
        let packed = (0..config.layers)
            .map(|layer| {
                checkpoint
                    .experts(layer, ExpertProjection::GateUp)
                    .unwrap()
                    .resident_bytes()
                    + checkpoint
                        .experts(layer, ExpertProjection::Down)
                        .unwrap()
                        .resident_bytes()
            })
            .sum::<u64>();
        // NVFP4 layers also store one denominator per projection.
        let nvfp4_layers = checkpoint
            .expert_formats()
            .iter()
            .filter(|format| **format == ExpertFormat::Nvfp4)
            .count() as u64;
        assert_eq!(footprint.expert_bytes, packed + nvfp4_layers * 2 * 4);
        assert_eq!(
            footprint.router_bytes,
            (config.layers * (config.experts * config.hidden + config.experts) * 4) as u64
        );
        assert_eq!(
            footprint.largest_dense_bytes,
            (config.vocab * config.hidden * 4) as u64
        );
        assert!(footprint.dense_bytes > footprint.largest_dense_bytes);
        // The vision tower is loaded with the model and priced with it.
        assert_eq!(
            footprint.vision_bytes,
            checkpoint.stored_bytes_under(VISION_PREFIX) * 2
        );
        assert!(footprint.vision_bytes > 0 && footprint.vision_bytes < footprint.dense_bytes);
        // On the device, CUDA holds the experts and the host only the routers.
        assert_eq!(
            footprint.resident_bytes(BackendKind::Cuda, true),
            footprint.dense_bytes + footprint.expert_bytes + footprint.router_bytes
        );
        assert_eq!(footprint.host_resident_bytes(true), footprint.router_bytes);
        assert!(
            footprint.load_peak_bytes(BackendKind::Cpu, false)
                > footprint.resident_bytes(BackendKind::Cpu, false)
        );
        assert!(
            footprint.resident_bytes(BackendKind::Cuda, false)
                < footprint.resident_bytes(BackendKind::Metal, false)
        );
    }

    /// With 16-bit projections the large matrices cost half and nothing
    /// else moves.
    #[test]
    fn projections_are_priced_at_the_dtype_they_are_held_in() {
        let dir = fixture("inkling2_golden");
        let config = tiny();
        let checkpoint = Inkling2Checkpoint::open(&dir, &config).unwrap();
        let wide = Inkling2Footprint::of(&config, &checkpoint, DType::F32);
        let (mut text, mut vision, mut unreadable) = (0u64, 0u64, 0u64);
        for entry in expected_census(&config, checkpoint.expert_formats()) {
            let held = |dtype| held_element_bytes(&entry.name, &entry.shape, dtype);
            // Every projection halves in BF16; in F16 all but the tiny
            // patch embedding, whose input width the Metal kernel cannot
            // read.
            if held(DType::BF16) != 2 {
                continue;
            }
            if held(DType::F16) != 2 {
                assert!(entry.name.ends_with(".patch_embedding.weight"));
                unreadable += (entry.shape[0] * entry.shape[1]) as u64;
            }
            let rows = if entry.name.ends_with(".unembed.weight") {
                config.unpadded_vocab
            } else {
                entry.shape[0]
            };
            let elements = (rows * entry.shape[1]) as u64;
            if entry.name.starts_with(VISION_PREFIX) {
                vision += elements;
            } else {
                text += elements;
            }
        }
        assert!(text > 0 && vision > 0);
        for (dtype, wide_vision) in [(DType::F16, unreadable), (DType::BF16, 0)] {
            let narrow = Inkling2Footprint::of(&config, &checkpoint, dtype);
            let vision = vision - wide_vision;
            assert_eq!(narrow.dense_bytes, wide.dense_bytes - 2 * (text + vision));
            assert_eq!(narrow.vision_bytes, wide.vision_bytes - 2 * vision);
            assert_eq!(
                narrow.largest_dense_bytes,
                (config.vocab * config.hidden * 2) as u64
            );
            assert_eq!(narrow.expert_bytes, wide.expert_bytes);
            assert_eq!(narrow.router_bytes, wide.router_bytes);
        }
        // What a load on each backend prices is what its loader holds.
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            assert_eq!(
                Inkling2Footprint::load(&dir, backend).unwrap(),
                Inkling2Footprint::of(&config, &checkpoint, projection_dtype(backend))
            );
        }
    }

    /// The published model, priced from its config and the published expert
    /// formats: 14.6 GiB of packed experts, 6.0 GiB of F32 dense text
    /// tensors and 1.2 GiB of F32 vision tensors.
    #[test]
    fn the_published_checkpoint_fits_the_documented_sizes() {
        let config = published();
        let formats = (0..config.layers)
            .map(|layer| {
                if layer < 20 {
                    ExpertFormat::Nvfp4
                } else {
                    ExpertFormat::Mxfp8
                }
            })
            .collect::<Vec<_>>();
        let mut expert_bytes = 0u64;
        let mut dense_bytes = 0u64;
        let mut vision_bytes = 0u64;
        for entry in expected_census(&config, &formats) {
            let elements = entry.shape.iter().product::<usize>() as u64;
            if entry.name.starts_with(VISION_PREFIX) {
                vision_bytes += elements * 4;
                continue;
            }
            if !entry.name.starts_with(TEXT_PREFIX) {
                continue;
            }
            match entry.stored {
                Stored::Dense | Stored::DenseF32 => dense_bytes += elements * 4,
                Stored::U8 | Stored::Fp8 => expert_bytes += elements,
                Stored::F32 => expert_bytes += elements * 4,
            }
        }
        let gib = (1u64 << 30) as f64;
        // As held: 7.1 GiB of dense tensors in F32 (the unembedding without
        // its padded rows), 3.6 GiB with the projections in 16 bits, of
        // which the vision tower is 1.2 and 0.6 GiB.
        let wide = Inkling2Footprint::of_census(&config, &formats, DType::F32);
        let narrow = Inkling2Footprint::of_census(&config, &formats, DType::BF16);
        // Every published projection has a width the Metal kernel reads.
        assert_eq!(
            narrow,
            Inkling2Footprint::of_census(&config, &formats, DType::F16)
        );
        assert_eq!(wide.expert_bytes, expert_bytes);
        assert_eq!(narrow.expert_bytes, expert_bytes);
        for (footprint, dense, vision, largest) in
            [(wide, 7.15, 1.19, 1.73), (narrow, 3.63, 0.60, 0.86)]
        {
            let held = [
                (footprint.dense_bytes, dense),
                (footprint.vision_bytes, vision),
                (footprint.largest_dense_bytes, largest),
            ];
            for (bytes, want) in held {
                assert!(
                    (bytes as f64 / gib - want).abs() < 0.02,
                    "{} GiB, not {want}",
                    bytes as f64 / gib
                );
            }
        }
        assert!(
            (expert_bytes as f64 / gib - 14.59).abs() < 0.05,
            "{}",
            expert_bytes as f64 / gib
        );
        assert!(
            (dense_bytes as f64 / gib - 5.99).abs() < 0.05,
            "{}",
            dense_bytes as f64 / gib
        );
        assert!(
            (vision_bytes as f64 / gib - 1.2).abs() < 0.1,
            "{}",
            vision_bytes as f64 / gib
        );
    }
}
