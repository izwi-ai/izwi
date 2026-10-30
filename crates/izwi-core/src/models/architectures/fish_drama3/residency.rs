//! Per-backend weight forms (§6, D4–D6).
//!
//! | Weight | CPU | Metal | CUDA |
//! |---|---|---|---|
//! | Routed experts | Q8_0 (FP8 → F32 → Q8_0 at load) | raw E4M3 + F32 block scales | raw E4M3 + F32 block scales |
//! | Dense trunk projections, shared expert | Q8_0 | F16 | BF16 |
//! | Norm gains, router, shared-expert gate, DeltaNet gates, `A_log`, `dt_bias`, conv | F32 | F32 | F32 |
//! | Embedding (and the frame codebooks) | BF16 host table, gathered to F32 | — | — |
//! | Compact main head | F32 | — | — |
//! | Tower projections and codebook heads (BF16) | Q8_0 | F16 | BF16 |
//! | Tower code embeddings | BF16 host tables, gathered to F32 | — | — |
//!
//! The DeltaNet gate projections `in_proj_a` / `in_proj_b` (32 rows each)
//! stay F32 beside the router: they feed `exp` and `softplus` decays, and
//! they are 0.4% of the dense bytes.
//!
//! A CPU load checks each Q8_0 projection class against F32 first
//! (`precision.rs`); a class over tolerance stays F32. Only the CPU column
//! loads today: Metal and CUDA are Phase 3 and refused until then.

use std::collections::BTreeMap;

use candle_core::Device;

use crate::error::{Error, Result};

/// The backend a load targets (`Device` alone does not carry the policy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FishDrama3Residency {
    Cpu,
    Metal,
    Cuda,
}

/// How one weight is held.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FishDrama3WeightForm {
    /// ggml Q8_0 blocks (34 B per 32 values), CPU `QMatMul`.
    Q8_0,
    F32,
    /// Metal dense (Phase 3).
    F16,
    /// CUDA dense (Phase 3).
    Bf16,
    /// E4M3 codes plus their F32 block scales, unchanged (Phase 3).
    RawFp8,
}

impl FishDrama3WeightForm {
    /// Resident bytes of a `[rows, cols]` weight (RawFp8 with 128×128
    /// blocks).
    pub fn bytes(self, rows: usize, cols: usize) -> usize {
        let elements = rows * cols;
        match self {
            Self::Q8_0 => elements / 32 * 34,
            Self::F32 => elements * 4,
            Self::F16 | Self::Bf16 => elements * 2,
            Self::RawFp8 => elements + 4 * rows.div_ceil(128) * cols.div_ceil(128),
        }
    }
}

/// The projections a load checks and stores as one class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FishDrama3ProjectionClass {
    /// Full-attention `q_proj` (query and gate), `k_proj`, `v_proj`.
    AttentionIn,
    /// Full-attention `o_proj`.
    AttentionOut,
    /// DeltaNet `in_proj_qkv` and `in_proj_z`.
    DeltaNetIn,
    /// DeltaNet `out_proj`.
    DeltaNetOut,
    /// The dense shared expert's gate, up and down projections.
    SharedExpert,
    /// The FP8 routed experts' gate, up and down projections.
    RoutedExpert,
    /// The tower layers' `eh_proj` merges.
    TowerMerge,
    /// The tower layers' `wqkv` and `wo`.
    TowerAttention,
    /// The tower layers' `w1`, `w3` and `w2`.
    TowerFeedForward,
    /// The tower's codebook heads.
    TowerHead,
}

impl FishDrama3ProjectionClass {
    pub const ALL: [Self; 10] = [
        Self::AttentionIn,
        Self::AttentionOut,
        Self::DeltaNetIn,
        Self::DeltaNetOut,
        Self::SharedExpert,
        Self::RoutedExpert,
        Self::TowerMerge,
        Self::TowerAttention,
        Self::TowerFeedForward,
        Self::TowerHead,
    ];

    /// The classes a trunk load checks.
    pub const TRUNK: [Self; 6] = [
        Self::AttentionIn,
        Self::AttentionOut,
        Self::DeltaNetIn,
        Self::DeltaNetOut,
        Self::SharedExpert,
        Self::RoutedExpert,
    ];

    /// The classes a tower load checks.
    pub const TOWER: [Self; 4] = [
        Self::TowerMerge,
        Self::TowerAttention,
        Self::TowerFeedForward,
        Self::TowerHead,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::AttentionIn => "attention_in",
            Self::AttentionOut => "attention_out",
            Self::DeltaNetIn => "deltanet_in",
            Self::DeltaNetOut => "deltanet_out",
            Self::SharedExpert => "shared_expert",
            Self::RoutedExpert => "routed_expert",
            Self::TowerMerge => "tower_merge",
            Self::TowerAttention => "tower_attention",
            Self::TowerFeedForward => "tower_feed_forward",
            Self::TowerHead => "tower_head",
        }
    }

    /// Whether the checkpoint holds the class as FP8 (else BF16).
    pub fn is_fp8(self) -> bool {
        self == Self::RoutedExpert
    }
}

/// How a CPU load stores its projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FishDrama3CpuPrecision {
    /// D4: Q8_0, except a class whose self-check fails.
    #[default]
    Q8_0,
    /// Every projection F32 (parity runs; the routed experts alone then take
    /// about 120 GiB on the published model).
    F32,
}

impl FishDrama3Residency {
    pub fn of(device: &Device) -> Self {
        match device {
            Device::Cpu => Self::Cpu,
            Device::Metal(_) => Self::Metal,
            Device::Cuda(_) => Self::Cuda,
        }
    }

    /// Refuses the backends whose forms are not implemented yet.
    pub fn require_implemented(self) -> Result<()> {
        match self {
            Self::Cpu => Ok(()),
            Self::Metal | Self::Cuda => Err(Error::ModelLoadError(format!(
                "Fish Audio Drama 3 has no {self:?} weight residency yet; load it on the CPU"
            ))),
        }
    }

    /// The form of the dense BF16 projections.
    pub fn dense_form(self) -> FishDrama3WeightForm {
        match self {
            Self::Cpu => FishDrama3WeightForm::Q8_0,
            Self::Metal => FishDrama3WeightForm::F16,
            Self::Cuda => FishDrama3WeightForm::Bf16,
        }
    }

    /// The form of the routed FP8 experts.
    pub fn expert_form(self) -> FishDrama3WeightForm {
        match self {
            Self::Cpu => FishDrama3WeightForm::Q8_0,
            Self::Metal | Self::Cuda => FishDrama3WeightForm::RawFp8,
        }
    }

    /// The form a class takes before any self-check.
    pub fn class_form(self, class: FishDrama3ProjectionClass) -> FishDrama3WeightForm {
        if class.is_fp8() {
            self.expert_form()
        } else {
            self.dense_form()
        }
    }
}

/// The form each projection class loads in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FishDrama3ClassForms {
    forms: BTreeMap<FishDrama3ProjectionClass, FishDrama3WeightForm>,
}

impl FishDrama3ClassForms {
    /// Every class in `form`.
    pub fn uniform(form: FishDrama3WeightForm) -> Self {
        Self {
            forms: FishDrama3ProjectionClass::ALL
                .into_iter()
                .map(|class| (class, form))
                .collect(),
        }
    }

    /// Each class in its backend form.
    pub fn of(residency: FishDrama3Residency) -> Self {
        Self {
            forms: FishDrama3ProjectionClass::ALL
                .into_iter()
                .map(|class| (class, residency.class_form(class)))
                .collect(),
        }
    }

    pub fn form(&self, class: FishDrama3ProjectionClass) -> FishDrama3WeightForm {
        self.forms[&class]
    }

    pub fn set(&mut self, class: FishDrama3ProjectionClass, form: FishDrama3WeightForm) {
        self.forms.insert(class, form);
    }

    pub fn iter(
        &self,
    ) -> impl Iterator<Item = (FishDrama3ProjectionClass, FishDrama3WeightForm)> + '_ {
        self.forms.iter().map(|(class, form)| (*class, *form))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_backend_names_its_forms_and_only_the_cpu_loads() {
        use FishDrama3WeightForm::*;
        let cpu = FishDrama3ClassForms::of(FishDrama3Residency::Cpu);
        assert!(cpu.iter().all(|(_, form)| form == Q8_0));
        let metal = FishDrama3ClassForms::of(FishDrama3Residency::Metal);
        assert_eq!(metal.form(FishDrama3ProjectionClass::RoutedExpert), RawFp8);
        assert_eq!(metal.form(FishDrama3ProjectionClass::AttentionIn), F16);
        let cuda = FishDrama3ClassForms::of(FishDrama3Residency::Cuda);
        assert_eq!(cuda.form(FishDrama3ProjectionClass::SharedExpert), Bf16);
        assert_eq!(cuda.form(FishDrama3ProjectionClass::RoutedExpert), RawFp8);
        assert!(FishDrama3Residency::Cpu.require_implemented().is_ok());
        assert!(FishDrama3Residency::Metal.require_implemented().is_err());
        assert!(FishDrama3Residency::Cuda.require_implemented().is_err());
        // A published routed expert projection, [512, 2048].
        assert_eq!(Q8_0.bytes(512, 2048), 1_114_112);
        assert_eq!(RawFp8.bytes(512, 2048), 1_048_576 + 4 * 4 * 16);
    }
}
