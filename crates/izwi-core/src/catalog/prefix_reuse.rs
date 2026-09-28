//! Cross-request committed prefix-reuse capability per model family and
//! backend (DS1.6).
//!
//! A prefix-reuse flip that passes on one backend lane but not another is
//! enabled only on the lanes with evidence. This table is the record of that
//! evidence: default-on (Auto) serving engages reuse only for cells whose
//! evidence level proves committed-page parity on the active backend. An
//! explicit operator enablement bypasses the table — the operator asked for
//! it and owns the risk — while [`PrefixReuseMode::Disabled`] keeps reuse off
//! regardless of evidence.

use serde::{Deserialize, Serialize};

use super::{ModelFamily, ModelVariant};
use crate::backends::BackendKind;

/// How the serving stack decides whether committed prefix reuse engages for
/// a loaded model. The registry default is fail-closed; product surfaces opt
/// into [`PrefixReuseMode::CatalogAuto`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrefixReuseMode {
    /// Reuse stays off for every model regardless of evidence.
    #[default]
    Disabled,
    /// The operator explicitly enabled prefix caching; engages for enabled
    /// variants on every backend.
    Explicit,
    /// Engage reuse only where the catalog cell has lane evidence.
    CatalogAuto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixReuseSupportLevel {
    /// Auto mode engages reuse for this family on this backend.
    Supported,
    /// Auto mode keeps reuse off for this cell.
    NotEnabled,
}

impl PrefixReuseSupportLevel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Supported => "supported",
            Self::NotEnabled => "not_enabled",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrefixReuseEvidenceLevel {
    /// Committed-prefix attach produced bit-identical outputs across worker
    /// processes on this backend lane (`backend_parity` prefix leg).
    ProcessParity,
    /// The DS1.2 fixture correctness suite (concurrent shared-prefix hits,
    /// counter evidence, eviction, salt isolation) passed on this lane.
    FixtureSuite,
    /// The model contract declares shareable domains, but no lane evidence
    /// exists yet.
    ContractDeclared,
    /// No evidence was collected on this lane.
    NotRun,
    /// The family is excluded by design (hybrid reuse unproven, or the task
    /// gate keeps managed prefix reuse chat-only).
    Excluded,
}

impl PrefixReuseEvidenceLevel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ProcessParity => "process_parity",
            Self::FixtureSuite => "fixture_suite",
            Self::ContractDeclared => "contract_declared",
            Self::NotRun => "not_run",
            Self::Excluded => "excluded",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PrefixReuseSupport {
    pub level: PrefixReuseSupportLevel,
    pub evidence: PrefixReuseEvidenceLevel,
    pub reason: &'static str,
}

impl PrefixReuseSupport {
    pub const fn engages(self) -> bool {
        matches!(self.level, PrefixReuseSupportLevel::Supported)
    }
}

const fn supported(evidence: PrefixReuseEvidenceLevel, reason: &'static str) -> PrefixReuseSupport {
    PrefixReuseSupport {
        level: PrefixReuseSupportLevel::Supported,
        evidence,
        reason,
    }
}

const fn not_enabled(
    evidence: PrefixReuseEvidenceLevel,
    reason: &'static str,
) -> PrefixReuseSupport {
    PrefixReuseSupport {
        level: PrefixReuseSupportLevel::NotEnabled,
        evidence,
        reason,
    }
}

const fn excluded(reason: &'static str) -> PrefixReuseSupport {
    PrefixReuseSupport {
        level: PrefixReuseSupportLevel::NotEnabled,
        evidence: PrefixReuseEvidenceLevel::Excluded,
        reason,
    }
}

impl ModelFamily {
    /// Catalog cell for committed cross-request prefix reuse on `backend`.
    /// One cell per family per backend; never a blanket flip.
    pub fn prefix_reuse_support(self, backend: BackendKind) -> PrefixReuseSupport {
        match self {
            // DS1.5: hybrid qwen3.8 publishes committed pages on attention
            // domains and transactional tensor snapshots on recurrent/conv
            // domains. The parity legs ran on the CPU and Metal lanes.
            Self::Qwen38Chat => match backend {
                BackendKind::Cpu => supported(
                    PrefixReuseEvidenceLevel::ProcessParity,
                    "backend_parity qwen38 hybrid prefix leg: bit-identical attached prefill across worker processes",
                ),
                BackendKind::Metal => supported(
                    PrefixReuseEvidenceLevel::ProcessParity,
                    "metal-feature parity leg bit-identical to the CPU lane (DS1.5)",
                ),
                BackendKind::Cuda => not_enabled(
                    PrefixReuseEvidenceLevel::NotRun,
                    "no CUDA lane parity evidence; DS1.6 requires lane parity before default-on",
                ),
            },
            // DS1.2 dense enablement: contracts publish committed pages and
            // the fixture correctness suite (concurrency, counters, eviction,
            // salt isolation) ran on CPU. No dense process-parity leg exists
            // on Metal or CUDA yet.
            Self::Qwen3Chat | Self::Gemma3Chat => match backend {
                BackendKind::Cpu => supported(
                    PrefixReuseEvidenceLevel::FixtureSuite,
                    "DS1.2 dense fixture correctness suite (shared-prefix hits, eviction, salt isolation)",
                ),
                BackendKind::Metal => not_enabled(
                    PrefixReuseEvidenceLevel::ContractDeclared,
                    "contract declares committed pages; no Metal lane parity evidence yet",
                ),
                BackendKind::Cuda => not_enabled(
                    PrefixReuseEvidenceLevel::NotRun,
                    "no CUDA lane evidence for dense prefix reuse",
                ),
            },
            // DS10 groundwork (ADR 0008): the sparse-MoE family shares the
            // dense qwen3 committed-pages machinery, but per-family evidence
            // is not collected yet — the cell stays fail-closed until the MoE
            // fixture correctness suite runs.
            Self::Qwen3MoeChat => match backend {
                BackendKind::Cpu | BackendKind::Metal => not_enabled(
                    PrefixReuseEvidenceLevel::ContractDeclared,
                    "contract declares committed pages; MoE fixture correctness suite pending",
                ),
                BackendKind::Cuda => not_enabled(
                    PrefixReuseEvidenceLevel::NotRun,
                    "no CUDA lane evidence for sparse-MoE prefix reuse",
                ),
            },
            Self::Voxtral => not_enabled(
                PrefixReuseEvidenceLevel::ContractDeclared,
                "LM contract declares committed pages; no lane parity evidence yet",
            ),
            Self::Qwen35Chat => excluded(
                "hybrid linear-attention/conv reuse is not proven (DS1.1 scope); contract keeps hybrid domains Disabled",
            ),
            Self::Lfm2Chat => excluded(
                "hybrid state domains keep per-request KV; reuse is not proven for this family",
            ),
            Self::Lfm25Audio => not_enabled(
                PrefixReuseEvidenceLevel::NotRun,
                "audio-chat reuse evidence not collected",
            ),
            Self::Qwen3Tts
            | Self::KokoroTts
            | Self::VoxtralTts
            | Self::VibeVoiceTts
            | Self::FishS2Tts
            | Self::ParakeetAsr
            | Self::WhisperAsr
            | Self::Qwen3Asr
            | Self::VibeVoiceAsr
            | Self::NemotronAsr
            | Self::GraniteSpeechAsr
            | Self::SortformerDiarization
            | Self::Qwen3ForcedAligner => excluded(
                "managed prefix reuse is gated to chat task requests",
            ),
            Self::Tokenizer => excluded("tokenizer artifacts carry no inference state"),
        }
    }
}

impl ModelVariant {
    /// Catalog cell for committed cross-request prefix reuse on `backend`.
    /// Disabled variants fail closed regardless of family evidence.
    pub fn prefix_reuse_support(&self, backend: BackendKind) -> PrefixReuseSupport {
        if !self.is_enabled() {
            return excluded("variant is disabled in the application catalog");
        }
        self.family().prefix_reuse_support(backend)
    }
}

/// Resolve whether committed prefix reuse engages for `variant` on `backend`
/// under `mode`. Explicit operator enablement bypasses the evidence table.
pub fn prefix_reuse_engages(
    variant: ModelVariant,
    backend: BackendKind,
    mode: PrefixReuseMode,
) -> bool {
    match mode {
        PrefixReuseMode::Disabled => false,
        PrefixReuseMode::Explicit => variant.is_enabled(),
        PrefixReuseMode::CatalogAuto => variant.prefix_reuse_support(backend).engages(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_BACKENDS: [BackendKind; 3] =
        [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda];

    #[test]
    fn prefix_reuse_inventory_covers_every_variant_and_backend() {
        for variant in ModelVariant::all() {
            for backend in ALL_BACKENDS {
                let cell = variant.prefix_reuse_support(backend);
                assert!(
                    !cell.reason.is_empty(),
                    "{variant:?} x {backend:?} has an empty reason"
                );
                if !variant.is_enabled() {
                    assert_eq!(
                        cell.evidence,
                        PrefixReuseEvidenceLevel::Excluded,
                        "disabled variant {variant:?} must fail closed"
                    );
                    assert!(!cell.engages());
                }
            }
        }
    }

    #[test]
    fn qwen38_hybrid_prefix_reuse_engages_only_on_parity_lanes() {
        let variant = ModelVariant::Qwen3827BFp8;
        assert!(variant.prefix_reuse_support(BackendKind::Cpu).engages());
        assert_eq!(
            variant.prefix_reuse_support(BackendKind::Cpu).evidence,
            PrefixReuseEvidenceLevel::ProcessParity
        );
        assert!(variant.prefix_reuse_support(BackendKind::Metal).engages());
        let cuda = variant.prefix_reuse_support(BackendKind::Cuda);
        assert!(!cuda.engages());
        assert_eq!(cuda.evidence, PrefixReuseEvidenceLevel::NotRun);
    }

    #[test]
    fn dense_fixture_suite_cells_engage_on_cpu_only() {
        for variant in [ModelVariant::Qwen34BGguf, ModelVariant::Gemma31BIt] {
            let cpu = variant.prefix_reuse_support(BackendKind::Cpu);
            assert!(cpu.engages(), "{variant:?} cpu cell must engage");
            assert_eq!(cpu.evidence, PrefixReuseEvidenceLevel::FixtureSuite);
            for backend in [BackendKind::Metal, BackendKind::Cuda] {
                assert!(
                    !variant.prefix_reuse_support(backend).engages(),
                    "{variant:?} must stay off on {backend:?}"
                );
            }
        }
    }

    #[test]
    fn hybrid_unproven_families_stay_excluded() {
        for variant in [
            ModelVariant::Qwen3508BGguf,
            ModelVariant::Lfm2512BInstructGguf,
        ] {
            for backend in ALL_BACKENDS {
                let cell = variant.prefix_reuse_support(backend);
                assert!(!cell.engages(), "{variant:?} must stay off on {backend:?}");
                assert_eq!(cell.evidence, PrefixReuseEvidenceLevel::Excluded);
            }
        }
    }

    #[test]
    fn explicit_mode_bypasses_the_catalog_and_disabled_mode_blocks() {
        let variant = ModelVariant::Qwen3827BFp8;
        assert!(prefix_reuse_engages(
            variant,
            BackendKind::Cuda,
            PrefixReuseMode::Explicit
        ));
        assert!(!prefix_reuse_engages(
            variant,
            BackendKind::Cuda,
            PrefixReuseMode::CatalogAuto
        ));
        assert!(prefix_reuse_engages(
            variant,
            BackendKind::Cpu,
            PrefixReuseMode::CatalogAuto
        ));
        assert!(!prefix_reuse_engages(
            variant,
            BackendKind::Cpu,
            PrefixReuseMode::Disabled
        ));
    }
}
