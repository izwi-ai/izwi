//! Load admission for Fish Audio Drama 3 (§4.1, §7.3), priced from the
//! pinned census before a weight is read.
//!
//! The census (`fish_drama3::checkpoint::expected_census`) is derived from
//! `config.json` under the published policy, so every dimension is the
//! pinned checkpoint's, and the family's census test ties it to the 15
//! published shard headers. Nothing past `config.json` (5.5 kB, which the
//! load's identity check hashes) is opened: not a shard, not the index, not
//! `codec.pth`, whose metadata is the pinned codec's (it is byte-identical
//! to Fish S2 Pro's).
//!
//! Each tensor is priced in the form its backend holds it (§6,
//! `fish_drama3::residency`):
//!
//! | Part | CPU | Metal | CUDA |
//! |---|---|---|---|
//! | Routed experts | Q8_0 (34 B per 32) | raw E4M3 + F32 block scales | raw E4M3 + F32 block scales |
//! | Dense trunk projections, shared expert | Q8_0 | F16 | BF16 |
//! | Router, shared-expert gate, DeltaNet a/b gates, norms, conv, `A_log`, `dt_bias` | F32 | F32 | F32 |
//! | Embedding | BF16 host table | F16 | BF16 |
//! | Compact head (4,097 rows) | F32 | F16 | BF16 |
//! | Tower projections and heads | Q8_0 | F16 | BF16 |
//! | Tower code tables, frame codebooks 1..=9 | BF16 | F16 | BF16 |
//! | Codec | F32 | F32 | F32 |
//!
//! Load transients: the codec loads first and alone (its raw read, its
//! fused weight-norm pairs and per-tensor conversions); then, beside the
//! resident weights, the conversion scratch (1 GiB, 256 MiB on CUDA), the
//! per-tensor instantiation slack of the 62,245 tensors, and on Metal and
//! CUDA the fused MoE's expert stacking (4/3 of one layer's experts). CUDA
//! also authorizes its host staging window. The reference cache is a
//! deferred host charge. Every backend loads in these forms: a Metal load
//! holds exactly what this prices (`a_metal_load_holds_what_its_census_prices`),
//! and a CUDA load, which runs Metal's device column in BF16, holds the
//! same bytes (`a_cuda_load_holds_what_its_census_prices`, on a GPU).
//!
//! A machine whose memory can never hold the weights and one request is
//! refused here, with the numbers, before anything is evicted or read
//! ([`refuse_undersized_device`]).

use std::path::Path;

use super::{model_resource_plan, ModelMemoryEstimate, ModelResourcePlan};
use crate::backends::BackendKind;
use crate::engine::{ResourceAmount, ResourceVector};
use crate::error::{Error, Result};
use crate::kv::v2::InferenceStateContract;
use crate::models::architectures::fish_drama3::checkpoint::{
    expected_census, CensusDtype, CensusEntry, CODEBOOK_EMBEDDINGS, FP8_SCALE_SUFFIX, LM_HEAD,
    TOWER_HEADS_PREFIX, TOWER_RESIDUAL_PREFIX, TOWER_SHARED_PREFIX, TRUNK_PREFIX,
};
use crate::models::architectures::fish_drama3::codec::{
    FishDrama3CodecMemory, FishDrama3DacConfig,
};
use crate::models::architectures::fish_drama3::config::FishDrama3Config;
use crate::models::architectures::fish_drama3::physical::state_contract;
use crate::models::architectures::fish_drama3::reference_cache::FISH_DRAMA3_REFERENCE_CACHE_BYTES;
use crate::models::architectures::fish_drama3::residency::{
    FishDrama3Residency, FishDrama3WeightForm,
};
use crate::runtime::fish_drama3_capacity::{FishDrama3RowCeilings, ResolvedFishDrama3Capacity};

const GIB: u64 = 1024 * 1024 * 1024;
const PORTABLE_CONVERSION_SCRATCH_BYTES: u64 = GIB;
const CUDA_DEVICE_CONVERSION_SCRATCH_BYTES: u64 = 256 * 1024 * 1024;
/// The CUDA host staging window: qwen36moe's 8 GiB, kept until a GPU run
/// measures the family's own uploads (a load holds at most two thirds of one
/// layer's experts on the host while it stacks them, about 0.5 GiB, beside
/// one projection's F32 conversion).
const CUDA_HOST_STAGING_BYTES: u64 = 8 * GIB;

/// `codec.pth`'s metadata as `fish_drama3_codec_memory` prices it. The file
/// is pinned (`artifacts.rs`) and byte-identical to Fish S2 Pro's;
/// `the_pinned_codec_memory_is_the_published_codec_s` checks these numbers
/// against the real file.
const PINNED_CODEC_MEMORY: FishDrama3CodecMemory = FishDrama3CodecMemory {
    resident_parameter_bytes: 1_565_593_540,
    // The F32 weights; the positional buffers are never read.
    raw_load_bytes: 1_565_722_120,
    fused_load_bytes: 306_057_600,
    largest_source_tensor_bytes: 75_497_472,
    largest_target_tensor_bytes: 75_497_472,
};

/// Words for a backend's load domain.
fn domain_name(backend: BackendKind) -> &'static str {
    match backend {
        BackendKind::Cpu => "system",
        BackendKind::Metal => "unified",
        BackendKind::Cuda => "GPU",
    }
}

fn overflow() -> Error {
    Error::ModelLoadError("Fish Audio Drama 3 memory estimate overflows".into())
}

fn sum(values: &[u64]) -> Result<u64> {
    values
        .iter()
        .try_fold(0u64, |total, value| total.checked_add(*value))
        .ok_or_else(overflow)
}

fn to_u64(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| overflow())
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / GIB as f64
}

fn residency(backend: BackendKind) -> FishDrama3Residency {
    match backend {
        BackendKind::Cpu => FishDrama3Residency::Cpu,
        BackendKind::Metal => FishDrama3Residency::Metal,
        BackendKind::Cuda => FishDrama3Residency::Cuda,
    }
}

/// Resident bytes by part, the columns of §4.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct FishDrama3ResidentParts {
    pub(super) experts: u64,
    pub(super) dense: u64,
    pub(super) small: u64,
    pub(super) embedding: u64,
    pub(super) head: u64,
    pub(super) tower: u64,
    pub(super) codec: u64,
}

impl FishDrama3ResidentParts {
    pub(super) fn total(&self) -> Result<u64> {
        sum(&[
            self.experts,
            self.dense,
            self.small,
            self.embedding,
            self.head,
            self.tower,
            self.codec,
        ])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct FishDrama3MemoryEstimate {
    pub(super) parts: FishDrama3ResidentParts,
    pub(super) resident_bytes: u64,
    /// The load's peak in its domain: host memory on the CPU, unified on
    /// Metal, device memory on CUDA.
    pub(super) load_peak_bytes: u64,
    /// CUDA: host memory the load stages beside the device.
    pub(super) cuda_host_load_peak_bytes: u64,
}

/// The trunk matrices that stay F32 beside the router (`residency.rs`).
const F32_TRUNK_MATRICES: [&str; 4] = [
    "mlp.gate.weight",
    "mlp.shared_expert_gate.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight",
];

fn is_tower(name: &str) -> bool {
    [
        TOWER_SHARED_PREFIX,
        TOWER_RESIDUAL_PREFIX,
        TOWER_HEADS_PREFIX,
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
}

/// The bytes `backend` holds `entry` in, and the part they count in.
fn price(
    entry: &CensusEntry,
    config: &FishDrama3Config,
    backend: BackendKind,
    parts: &mut FishDrama3ResidentParts,
) -> Result<()> {
    let residency = residency(backend);
    let elements = entry
        .shape
        .iter()
        .try_fold(1usize, |total, dim| total.checked_mul(*dim))
        .ok_or_else(overflow)?;
    let matrix = |form: FishDrama3WeightForm| -> Result<u64> {
        let &[rows, cols] = entry.shape.as_slice() else {
            return Err(Error::ModelLoadError(format!(
                "Fish Audio Drama 3 census entry {} is not a matrix",
                entry.name
            )));
        };
        to_u64(form.bytes(rows, cols))
    };
    // A 16-bit device table on Metal and CUDA, BF16 on the host on the CPU.
    let half = |elements: usize| to_u64(elements.checked_mul(2).ok_or_else(overflow)?);
    let f32 = |elements: usize| to_u64(elements.checked_mul(4).ok_or_else(overflow)?);
    let name = entry.name.as_str();
    if entry.dtype == CensusDtype::F8E4M3 {
        parts.experts += match residency.expert_form() {
            FishDrama3WeightForm::RawFp8 => to_u64(elements)?,
            form => matrix(form)?,
        };
    } else if name.ends_with(FP8_SCALE_SUFFIX) {
        // Folded into Q8_0 on the CPU; kept beside the raw codes otherwise.
        if residency.expert_form() == FishDrama3WeightForm::RawFp8 {
            parts.experts += f32(elements)?;
        }
    } else if name == LM_HEAD {
        // Only the compact head's rows: eos and the semantic band.
        let rows = config.ids.semantic_codes() + 1;
        let row = rows.checked_mul(config.trunk.hidden).ok_or_else(overflow)?;
        parts.head += match backend {
            BackendKind::Cpu => f32(row)?,
            BackendKind::Metal | BackendKind::Cuda => half(row)?,
        };
    } else if name.strip_prefix(TRUNK_PREFIX) == Some(".embed_tokens.weight") {
        parts.embedding += half(elements)?;
    } else if name == CODEBOOK_EMBEDDINGS {
        // Codebook 0's rows are never read: a frame embeds its semantic
        // code through the trunk's table.
        let tower = &config.tower;
        let rows = tower.residual_layers() * tower.codebook_vocab;
        parts.tower += half(rows.checked_mul(tower.dim).ok_or_else(overflow)?)?;
    } else if is_tower(name) {
        parts.tower += if name.ends_with("embedding.weight") {
            half(elements)?
        } else if entry.shape.len() == 2 {
            matrix(residency.dense_form())?
        } else {
            f32(elements)?
        };
    } else if entry.shape.len() == 2
        && !F32_TRUNK_MATRICES
            .iter()
            .any(|suffix| name.ends_with(suffix))
    {
        parts.dense += matrix(residency.dense_form())?;
    } else {
        parts.small += f32(elements)?;
    }
    Ok(())
}

/// Load transient of the fused MoE's expert stacking on Metal and CUDA:
/// while a layer stacks, its gate and up stacks and their concatenation
/// coexist with the originals, 4/3 of one layer's experts and scales.
fn expert_stacking_bytes(experts: u64, layers: usize) -> Result<u64> {
    experts
        .div_ceil(to_u64(layers.max(1))?)
        .div_ceil(3)
        .checked_mul(4)
        .ok_or_else(overflow)
}

/// The resident bytes and load peaks of `config`'s census on `backend`.
pub(super) fn census_estimate(
    config: &FishDrama3Config,
    backend: BackendKind,
) -> Result<FishDrama3MemoryEstimate> {
    let census = expected_census(config);
    let mut parts = FishDrama3ResidentParts::default();
    for entry in &census {
        price(entry, config, backend, &mut parts)?;
    }
    let codec = PINNED_CODEC_MEMORY;
    parts.codec = sum(&[
        codec.resident_parameter_bytes,
        FishDrama3DacConfig::current().rotary_cache_bytes()?,
    ])?;
    let resident_bytes = parts.total()?;
    let slack = to_u64(census.len())?
        .checked_mul(super::PER_TENSOR_INSTANTIATION_SLACK_BYTES)
        .ok_or_else(overflow)?;
    let transient = match backend {
        BackendKind::Cpu => PORTABLE_CONVERSION_SCRATCH_BYTES,
        BackendKind::Metal => sum(&[
            PORTABLE_CONVERSION_SCRATCH_BYTES,
            expert_stacking_bytes(parts.experts, config.trunk.layers.len())?,
        ])?,
        BackendKind::Cuda => sum(&[
            CUDA_DEVICE_CONVERSION_SCRATCH_BYTES,
            expert_stacking_bytes(parts.experts, config.trunk.layers.len())?,
        ])?,
    };
    // The codec loads first, alone: its raw read and fused weights coexist
    // until the DAC is built, beside one tensor's conversions.
    let codec_peak = sum(&[
        codec.raw_load_bytes,
        codec.fused_load_bytes,
        codec.largest_source_tensor_bytes,
        codec
            .largest_target_tensor_bytes
            .checked_mul(3)
            .ok_or_else(overflow)?,
    ])?;
    let load_peak_bytes = sum(&[resident_bytes, transient, slack])?.max(codec_peak);
    Ok(FishDrama3MemoryEstimate {
        parts,
        resident_bytes,
        load_peak_bytes,
        cuda_host_load_peak_bytes: CUDA_HOST_STAGING_BYTES.max(codec_peak),
    })
}

/// `config.json` under the published policy (a test build's synthetic
/// checkpoint marked for the lifecycle, under the synthetic one): the only
/// file pricing reads.
fn published_config(model_path: &Path) -> Result<FishDrama3Config> {
    FishDrama3Config::load_with_policy(
        model_path,
        crate::models::architectures::fish_drama3::config::lifecycle_policy(model_path),
    )
}

/// The plan load admission prices for the checkpoint at `model_path`.
pub(super) fn resource_plan(model_path: &Path, backend: BackendKind) -> Result<ModelResourcePlan> {
    let config = published_config(model_path)?;
    let estimate = census_estimate(&config, backend)?;
    tracing::info!(
        backend = ?backend,
        resident_bytes = estimate.resident_bytes,
        load_peak_bytes = estimate.load_peak_bytes,
        "Priced Fish Audio Drama 3 from its pinned census"
    );
    let mut plan = model_resource_plan(
        backend,
        ModelMemoryEstimate {
            load_peak_bytes: estimate.load_peak_bytes,
            resident_bytes: estimate.resident_bytes,
        },
    );
    if backend == BackendKind::Cuda {
        // The host holds the staging window and the codec's host-side
        // normalization, not the weights.
        plan.load_authorization.host_bytes =
            ResourceAmount::Known(estimate.cuda_host_load_peak_bytes);
    }
    // Encoded references are host data owned by the model instance, filled
    // lazily: charged from load to unload, pending until filled, so state
    // fitting cannot take the room first.
    let reference_cache = ResourceVector {
        host_bytes: ResourceAmount::Known(FISH_DRAMA3_REFERENCE_CACHE_BYTES),
        ..ResourceVector::zero()
    };
    plan.load_authorization = plan.load_authorization.checked_add(reference_cache)?;
    plan.resident_authorization = plan.resident_authorization.checked_add(reference_cache)?;
    plan.deferred_resident_authorization = reference_cache;
    Ok(plan)
}

/// The retained contract a load at `model_path` allocates on `backend`, for
/// the pre-load state floor.
pub(super) fn retained_state_contract(
    model_path: &Path,
    backend: BackendKind,
) -> Result<InferenceStateContract> {
    state_contract(
        &published_config(model_path)?,
        crate::runtime::fish_drama3_capacity::planned_kv_dtype(backend)?,
        crate::models::shared::attention::paged::default_kv_page_size(),
    )
}

/// One worst-case request's row on `backend` (`fish_drama3_capacity`).
fn one_row_bytes(config: &FishDrama3Config, backend: BackendKind) -> Result<u64> {
    Ok(ResolvedFishDrama3Capacity::resolve(
        config,
        backend,
        None,
        FishDrama3RowCeilings {
            retained: 1,
            staged: 1,
            batch: Some(1),
        },
        None,
        0,
        false,
    )?
    .row_bytes)
}

/// Refuses a machine whose `capacity` (the planning ceiling of the load's
/// memory domain: the configured budget, else the physical memory) can
/// never hold the load's peak, or its weights beside one worst-case
/// request, before any weight is read or any resident model evicted. An
/// unknown capacity leaves the decision to live admission.
pub(super) fn refuse_undersized_device(
    model_path: &Path,
    backend: BackendKind,
    capacity: ResourceAmount,
) -> Result<()> {
    let ResourceAmount::Known(capacity) = capacity else {
        return Ok(());
    };
    let config = published_config(model_path)?;
    let estimate = census_estimate(&config, backend)?;
    check_capacity(
        &estimate,
        one_row_bytes(&config, backend)?,
        backend,
        capacity,
    )
}

fn check_capacity(
    estimate: &FishDrama3MemoryEstimate,
    row_bytes: u64,
    backend: BackendKind,
    capacity: u64,
) -> Result<()> {
    let required = sum(&[estimate.resident_bytes, row_bytes])?
        .max(estimate.load_peak_bytes)
        .checked_add(FISH_DRAMA3_REFERENCE_CACHE_BYTES)
        .ok_or_else(overflow)?;
    if required > capacity {
        return Err(Error::ModelLoadError(format!(
            "Fish Audio Drama 3 needs about {:.1} GiB of {} memory ({:.1} GiB of weights, a \
             {:.1} GiB load peak, {:.1} GiB for one request); this device has {:.1} GiB",
            gib(required),
            domain_name(backend),
            gib(estimate.resident_bytes),
            gib(estimate.load_peak_bytes),
            gib(row_bytes),
            gib(capacity),
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::architectures::fish_drama3::config::FishDrama3ConfigPolicy;

    const PUBLISHED_CONFIG: &str = include_str!("../../../tests/fixtures/fish_drama3/config.json");

    fn published() -> FishDrama3Config {
        FishDrama3Config::from_json(PUBLISHED_CONFIG, FishDrama3ConfigPolicy::Published).unwrap()
    }

    fn estimate(backend: BackendKind) -> FishDrama3MemoryEstimate {
        census_estimate(&published(), backend).unwrap()
    }

    /// `value` (bytes) is within `tolerance` GiB of the plan's `gib`.
    fn near(value: u64, gib_expected: f64, tolerance: f64) -> bool {
        (gib(value) - gib_expected).abs() <= tolerance
    }

    /// The §4.1 table, part by part, to its two decimals.
    #[test]
    fn the_published_census_prices_each_backend_as_the_plan_does() {
        let cpu = estimate(BackendKind::Cpu).parts;
        let metal = estimate(BackendKind::Metal).parts;
        let cuda = estimate(BackendKind::Cuda).parts;
        assert_eq!(
            metal, cuda,
            "Metal and CUDA hold the same bytes (F16 vs BF16)"
        );

        // Experts: 40 × 256 × 3 × 512 × 2048 values; Q8_0 is 34 B per 32,
        // raw E4M3 one byte plus 7.5 MiB of F32 block scales.
        let expert_values: u64 = 40 * 256 * 3 * 512 * 2048;
        assert_eq!(cpu.experts, expert_values / 32 * 34);
        assert_eq!(cuda.experts, expert_values + 40 * 256 * 3 * 64 * 4);
        assert!(near(cpu.experts, 31.88, 0.005));
        assert!(near(cuda.experts, 30.01, 0.005));
        // Dense trunk projections: Q8_0 1.39, 16-bit 2.62.
        assert!(near(cpu.dense, 1.39, 0.005), "{}", gib(cpu.dense));
        assert!(near(cuda.dense, 2.62, 0.005), "{}", gib(cuda.dense));
        // Router, gates and small F32 tensors: the plan's 0.08, plus the
        // DeltaNet a/b gates (15 MiB), conv and norms the loader keeps F32.
        assert!(near(cpu.small, 0.10, 0.005), "{}", gib(cpu.small));
        assert_eq!(cpu.small, cuda.small);
        // Embedding 0.96 everywhere.
        assert_eq!(cpu.embedding, 252_288 * 2048 * 2);
        assert_eq!(cuda.embedding, cpu.embedding);
        // Compact head: 4,097 rows, F32 on the CPU (0.03), 16-bit (0.02).
        assert_eq!(cpu.head, 4_097 * 2048 * 4);
        assert_eq!(cuda.head, 4_097 * 2048 * 2);
        // Tower: Q8_0 projections and BF16 tables 1.36, 16-bit 2.26 (the
        // plan's 1.38 and 2.28 also hold codebook 0's never-read rows and
        // BF16 norms).
        assert!(near(cpu.tower, 1.36, 0.005), "{}", gib(cpu.tower));
        assert!(near(cuda.tower, 2.26, 0.005), "{}", gib(cuda.tower));
        // The F32 codec, 1.46.
        assert!(near(cpu.codec, 1.46, 0.01), "{}", gib(cpu.codec));
        assert_eq!(cpu.codec, cuda.codec);

        // Totals: CPU ≈ 37.2, Metal and CUDA ≈ 37.4.
        let cpu_total = estimate(BackendKind::Cpu).resident_bytes;
        let cuda_total = estimate(BackendKind::Cuda).resident_bytes;
        assert!(near(cpu_total, 37.2, 0.05), "{}", gib(cpu_total));
        assert!(near(cuda_total, 37.4, 0.05), "{}", gib(cuda_total));
        assert_eq!(estimate(BackendKind::Metal).resident_bytes, cuda_total);
    }

    #[test]
    fn the_load_peak_adds_scratch_slack_and_the_stacking_transient() {
        let slack = 62_245 * super::super::PER_TENSOR_INSTANTIATION_SLACK_BYTES;
        assert!(near(slack, 1.9, 0.01));
        let cpu = estimate(BackendKind::Cpu);
        assert_eq!(cpu.load_peak_bytes, cpu.resident_bytes + GIB + slack);
        let metal = estimate(BackendKind::Metal);
        let stacking = expert_stacking_bytes(metal.parts.experts, 40).unwrap();
        // 4/3 of one layer's 768 MiB of experts and scales: about 1 GiB.
        assert!(near(stacking, 1.0, 0.01), "{}", gib(stacking));
        assert_eq!(
            metal.load_peak_bytes,
            metal.resident_bytes + GIB + stacking + slack
        );
        let cuda = estimate(BackendKind::Cuda);
        assert_eq!(
            cuda.load_peak_bytes,
            cuda.resident_bytes + 256 * 1024 * 1024 + stacking + slack
        );
        assert_eq!(cuda.cuda_host_load_peak_bytes, CUDA_HOST_STAGING_BYTES);
        // About 40 GiB at the peak on every backend.
        for peak in [
            cpu.load_peak_bytes,
            metal.load_peak_bytes,
            cuda.load_peak_bytes,
        ] {
            assert!(gib(peak) > 39.0 && gib(peak) < 41.5, "{}", gib(peak));
        }
    }

    /// A model directory holding only `config.json`: pricing needs nothing
    /// else, so the refusal comes before a weight file could be opened (none
    /// exists to open).
    fn config_only_dir() -> (std::path::PathBuf, impl Drop) {
        struct Remove(std::path::PathBuf);
        impl Drop for Remove {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let dir =
            std::env::temp_dir().join(format!("izwi-fish-drama3-memory-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), PUBLISHED_CONFIG).unwrap();
        (dir.clone(), Remove(dir))
    }

    #[test]
    fn the_plan_reads_only_the_config_and_defers_the_reference_cache() {
        let (dir, _remove) = config_only_dir();
        let entries = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(entries, 1);
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let plan = resource_plan(&dir, backend).unwrap();
            let estimate = estimate(backend);
            let cache = FISH_DRAMA3_REFERENCE_CACHE_BYTES;
            assert_eq!(
                plan.deferred_resident_authorization.host_bytes,
                ResourceAmount::Known(cache)
            );
            match backend {
                BackendKind::Cpu => assert_eq!(
                    plan.resident_authorization.host_bytes,
                    ResourceAmount::Known(estimate.resident_bytes + cache)
                ),
                BackendKind::Metal => assert_eq!(
                    plan.load_authorization.unified_bytes,
                    ResourceAmount::Known(estimate.load_peak_bytes)
                ),
                BackendKind::Cuda => {
                    assert_eq!(
                        plan.load_authorization.device_bytes,
                        ResourceAmount::Known(estimate.load_peak_bytes)
                    );
                    assert_eq!(
                        plan.load_authorization.host_bytes,
                        ResourceAmount::Known(CUDA_HOST_STAGING_BYTES + cache)
                    );
                }
            }
        }
        // The retained contract prices from the same file.
        let contract = retained_state_contract(&dir, BackendKind::Cpu).unwrap();
        assert_eq!(contract.domains.len(), 4);
        // A directory without even the config fails closed.
        std::fs::remove_file(dir.join("config.json")).unwrap();
        assert!(resource_plan(&dir, BackendKind::Cpu).is_err());
    }

    /// First admission and the Metal OOM retry price Drama 3 through this
    /// module, not the catalog's figure or a tensor inventory.
    #[test]
    fn the_checkpoint_plan_and_the_oom_retry_take_this_plan() {
        use crate::catalog::ModelVariant;
        let (dir, _remove) = config_only_dir();
        let variant = ModelVariant::FishAudioDrama3;
        let device = crate::backends::DeviceProfile::cpu();
        let granite = crate::config::GraniteSpeechOptions::default();
        let plan = super::super::checkpoint_resource_plan(variant, &dir, &device, &granite)
            .unwrap()
            .expect("a Drama 3 plan");
        assert_eq!(plan, resource_plan(&dir, BackendKind::Cpu).unwrap());
        let peak = super::super::retry_load_peak_bytes(
            variant,
            BackendKind::Cpu,
            &crate::performance::PerformanceConfig::default(),
            Some((&dir, &device, &granite)),
        );
        assert_eq!(
            peak,
            estimate(BackendKind::Cpu).load_peak_bytes + FISH_DRAMA3_REFERENCE_CACHE_BYTES
        );
    }

    #[test]
    fn an_undersized_device_is_refused_before_any_weight_is_read() {
        let (dir, _remove) = config_only_dir();
        let error =
            refuse_undersized_device(&dir, BackendKind::Cpu, ResourceAmount::Known(16 * GIB))
                .unwrap_err()
                .to_string();
        assert!(
            error.contains("Fish Audio Drama 3 needs about 4")
                && error.contains("GiB of system memory")
                && error.contains("this device has 16.0 GiB"),
            "{error}"
        );
        // A 48 GB Mac's working set (about 36 GiB) cannot hold it either.
        let error =
            refuse_undersized_device(&dir, BackendKind::Metal, ResourceAmount::Known(36 * GIB))
                .unwrap_err()
                .to_string();
        assert!(error.contains("of unified memory"), "{error}");
        let error =
            refuse_undersized_device(&dir, BackendKind::Cuda, ResourceAmount::Known(40 * GIB))
                .unwrap_err()
                .to_string();
        assert!(error.contains("of GPU memory"), "{error}");
        // A 64 GB box, a 64 GB Mac's 48 GiB working set and a 48 GB card
        // pass; unknown memory is live admission's to decide.
        refuse_undersized_device(&dir, BackendKind::Cpu, ResourceAmount::Known(59 * GIB)).unwrap();
        refuse_undersized_device(&dir, BackendKind::Metal, ResourceAmount::Known(48 * GIB))
            .unwrap();
        refuse_undersized_device(&dir, BackendKind::Cuda, ResourceAmount::Known(44 * GIB)).unwrap();
        refuse_undersized_device(&dir, BackendKind::Cpu, ResourceAmount::Unknown).unwrap();
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 1);
    }

    #[test]
    fn the_refusal_holds_the_weights_beside_one_request() {
        let cpu = estimate(BackendKind::Cpu);
        let row = one_row_bytes(&published(), BackendKind::Cpu).unwrap();
        // One worst-case CPU row: 62.8 MiB of state and 12,289 positions of
        // F32 KV, plus its workspaces.
        assert!(row > 65_863_680 + 94_208 * 12_289);
        let needed =
            (cpu.resident_bytes + row).max(cpu.load_peak_bytes) + FISH_DRAMA3_REFERENCE_CACHE_BYTES;
        check_capacity(&cpu, row, BackendKind::Cpu, needed).unwrap();
        assert!(check_capacity(&cpu, row, BackendKind::Cpu, needed - 1).is_err());
        // A row too big for the room past the weights is refused even when
        // the load peak fits.
        let huge = cpu.load_peak_bytes;
        assert!(check_capacity(&cpu, huge, BackendKind::Cpu, cpu.load_peak_bytes + GIB).is_err());
    }

    /// The pinned codec numbers are the real `codec.pth`'s
    /// (`IZWI_FISH_DRAMA3_CODEC_PATH`; Fish S2 Pro's file is the same).
    #[test]
    #[ignore = "needs the published codec.pth (IZWI_FISH_DRAMA3_CODEC_PATH)"]
    fn the_pinned_codec_memory_is_the_published_codec_s() {
        let path = std::env::var("IZWI_FISH_DRAMA3_CODEC_PATH")
            .expect("set IZWI_FISH_DRAMA3_CODEC_PATH to the published codec.pth");
        let archive = crate::models::shared::weights::pytorch::PthTensorMap::open_first_non_empty(
            Path::new(&path),
            &[None, Some("state_dict"), Some("generator")],
        )
        .unwrap();
        let measured = crate::models::architectures::fish_drama3::codec::fish_drama3_codec_memory(
            &archive.specs(),
        )
        .unwrap();
        assert_eq!(measured, PINNED_CODEC_MEMORY);
    }

    /// What a Metal load holds is what its census prices, part for part
    /// but the codec (the synthetic checkpoint's census is priced as the
    /// published one is): stacked raw FP8 experts and F32 scales, packed
    /// F16 projections, F32 small tensors, F16 tables and head.
    #[cfg(feature = "metal")]
    #[test]
    fn a_metal_load_holds_what_its_census_prices() {
        use crate::models::architectures::fish_drama3::model::tests::{
            write_synthetic_model_in, StubCodec,
        };
        use crate::models::architectures::fish_drama3::synthetic::SyntheticGeometry;
        use crate::models::architectures::fish_drama3::{FishDrama3Model, FishDrama3ModelOptions};
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("a_metal_load_holds_what_its_census_prices")
        else {
            return;
        };
        let synthetic = write_synthetic_model_in(5, SyntheticGeometry::KernelHeads, |_| {});
        let model = FishDrama3Model::load_with_codec(
            synthetic.path(),
            &device,
            FishDrama3ModelOptions {
                config_policy: FishDrama3ConfigPolicy::Synthetic,
                ..Default::default()
            },
            std::sync::Arc::new(StubCodec),
        )
        .unwrap();
        let estimate = census_estimate(&synthetic.config, BackendKind::Metal).unwrap();
        let held = model.trunk().resident_bytes()
            + model.tower().resident_bytes()
            + model.head().resident_bytes()
            + model.embedding().device_bytes();
        assert_eq!(held as u64, estimate.resident_bytes - estimate.parts.codec);
        assert_eq!(model.head().resident_bytes() as u64, estimate.parts.head);
        assert_eq!(model.embedding().host_bytes(), 0);
    }

    /// What a CUDA load holds is what its census prices, part for part but
    /// the codec: stacked raw FP8 experts and F32 scales, packed BF16
    /// projections, F32 small tensors, BF16 tables and head. Runs on a GPU.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_cuda_load_holds_what_its_census_prices() {
        use crate::models::architectures::fish_drama3::model::tests::{
            write_synthetic_model_in, StubCodec,
        };
        use crate::models::architectures::fish_drama3::synthetic::SyntheticGeometry;
        use crate::models::architectures::fish_drama3::{FishDrama3Model, FishDrama3ModelOptions};
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let synthetic = write_synthetic_model_in(5, SyntheticGeometry::KernelHeads, |_| {});
        let model = FishDrama3Model::load_with_codec(
            synthetic.path(),
            &device,
            FishDrama3ModelOptions {
                config_policy: FishDrama3ConfigPolicy::Synthetic,
                ..Default::default()
            },
            std::sync::Arc::new(StubCodec),
        )
        .unwrap();
        let estimate = census_estimate(&synthetic.config, BackendKind::Cuda).unwrap();
        let held = model.trunk().resident_bytes()
            + model.tower().resident_bytes()
            + model.head().resident_bytes()
            + model.embedding().device_bytes();
        assert_eq!(held as u64, estimate.resident_bytes - estimate.parts.codec);
        assert_eq!(model.head().resident_bytes() as u64, estimate.parts.head);
        assert_eq!(model.embedding().host_bytes(), 0);
        assert_eq!(model.kv_dtype(), candle_core::DType::BF16);
    }
}
