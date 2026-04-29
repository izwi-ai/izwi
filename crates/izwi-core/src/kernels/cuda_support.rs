//! CUDA kernel support inventory.
//!
//! This module is intentionally declarative. It records where CUDA kernel work
//! is expected to live without changing runtime dispatch or fallback behavior.

use crate::catalog::{ModelFamily, ModelVariant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaKernelOwnership {
    /// Izwi owns enough of the Rust model path to add CUDA-only kernel hooks.
    NativeIzwiOwned,
    /// The hot model path is primarily owned by Candle model implementations.
    CandleOwned,
    /// The current runtime path is CPU-only until routing is made explicit.
    CpuOnlyUntilRouted,
    /// The model variant is implemented or known, but not enabled for users.
    Disabled,
    /// The variant has no standalone model kernels but may use shared helpers.
    SharedOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaKernelPrimitive {
    Attention,
    PagedKv,
    Rope,
    MRope,
    Norm,
    Activation,
    KvQuantization,
    QuantizedMatmul,
    Conformer,
    ShortConv,
    DeltaNet,
    ArgmaxSampling,
    AudioPostprocess,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaKernelSupport {
    pub variant: ModelVariant,
    pub family: ModelFamily,
    pub ownership: CudaKernelOwnership,
    pub planned_primitives: &'static [CudaKernelPrimitive],
}

pub fn support_for_variant(variant: ModelVariant) -> CudaKernelSupport {
    let family = variant.family();
    let ownership = ownership_for_variant(variant, family);
    let planned_primitives = planned_primitives_for(variant, family, ownership);

    CudaKernelSupport {
        variant,
        family,
        ownership,
        planned_primitives,
    }
}

pub fn all_support_records() -> impl Iterator<Item = CudaKernelSupport> {
    ModelVariant::all().iter().copied().map(support_for_variant)
}

fn ownership_for_variant(variant: ModelVariant, family: ModelFamily) -> CudaKernelOwnership {
    if !variant.is_enabled() {
        return CudaKernelOwnership::Disabled;
    }

    match family {
        ModelFamily::Tokenizer => CudaKernelOwnership::SharedOnly,
        ModelFamily::Gemma3Chat | ModelFamily::Lfm2Chat => CudaKernelOwnership::CandleOwned,
        ModelFamily::Qwen3Chat if variant.is_qwen_chat_gguf() => CudaKernelOwnership::CandleOwned,
        ModelFamily::SortformerDiarization => CudaKernelOwnership::CpuOnlyUntilRouted,
        ModelFamily::Qwen3Tts
        | ModelFamily::KokoroTts
        | ModelFamily::ParakeetAsr
        | ModelFamily::WhisperAsr
        | ModelFamily::Qwen3Asr
        | ModelFamily::Qwen35Chat
        | ModelFamily::Lfm25Audio
        | ModelFamily::Qwen3ForcedAligner
        | ModelFamily::Voxtral
        | ModelFamily::Qwen3Chat => CudaKernelOwnership::NativeIzwiOwned,
    }
}

fn planned_primitives_for(
    _variant: ModelVariant,
    family: ModelFamily,
    ownership: CudaKernelOwnership,
) -> &'static [CudaKernelPrimitive] {
    use CudaKernelOwnership::*;
    use CudaKernelPrimitive::*;
    use ModelFamily::*;

    if ownership == Disabled {
        return &[];
    }

    match family {
        Qwen3Tts => &[
            Attention,
            PagedKv,
            Rope,
            MRope,
            Norm,
            Activation,
            KvQuantization,
            QuantizedMatmul,
            ArgmaxSampling,
            AudioPostprocess,
        ],
        KokoroTts => &[Attention, Norm, Activation, AudioPostprocess],
        ParakeetAsr => &[Attention, Conformer, ArgmaxSampling],
        WhisperAsr => &[Attention, ArgmaxSampling],
        Qwen3Asr => &[
            Attention,
            PagedKv,
            Rope,
            MRope,
            Norm,
            Activation,
            KvQuantization,
            QuantizedMatmul,
            ArgmaxSampling,
        ],
        SortformerDiarization => &[Attention, Conformer, AudioPostprocess],
        Qwen3Chat => &[
            Attention,
            PagedKv,
            Rope,
            Norm,
            Activation,
            KvQuantization,
            QuantizedMatmul,
            ArgmaxSampling,
        ],
        Qwen35Chat => &[
            Attention,
            PagedKv,
            Rope,
            MRope,
            Norm,
            Activation,
            KvQuantization,
            QuantizedMatmul,
            DeltaNet,
            ShortConv,
            ArgmaxSampling,
        ],
        Lfm2Chat => &[
            Attention,
            Rope,
            Norm,
            Activation,
            QuantizedMatmul,
            ShortConv,
        ],
        Lfm25Audio => &[
            Attention,
            Rope,
            Norm,
            Activation,
            QuantizedMatmul,
            ShortConv,
            Conformer,
            ArgmaxSampling,
            AudioPostprocess,
        ],
        Gemma3Chat => &[Attention, Norm, Activation, QuantizedMatmul, ArgmaxSampling],
        Qwen3ForcedAligner => &[
            Attention,
            PagedKv,
            Rope,
            MRope,
            Norm,
            Activation,
            KvQuantization,
            QuantizedMatmul,
        ],
        Voxtral => &[Attention, PagedKv, Rope, Norm, Activation, ArgmaxSampling],
        Tokenizer => &[],
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{
        all_support_records, support_for_variant, CudaKernelOwnership, CudaKernelPrimitive,
    };
    use crate::catalog::{ModelFamily, ModelVariant};

    #[test]
    fn cuda_support_inventory_covers_every_variant_once() {
        let records = all_support_records().collect::<Vec<_>>();
        assert_eq!(records.len(), ModelVariant::all().len());

        let unique = records
            .iter()
            .map(|record| record.variant)
            .collect::<HashSet<_>>();
        assert_eq!(unique.len(), ModelVariant::all().len());
    }

    #[test]
    fn enabled_variants_are_not_marked_disabled() {
        for record in all_support_records() {
            if record.variant.is_enabled() {
                assert_ne!(
                    record.ownership,
                    CudaKernelOwnership::Disabled,
                    "{} is enabled but marked disabled",
                    record.variant
                );
            }
        }
    }

    #[test]
    fn disabled_variants_are_explicitly_marked_disabled() {
        for record in all_support_records() {
            if !record.variant.is_enabled() {
                assert_eq!(
                    record.ownership,
                    CudaKernelOwnership::Disabled,
                    "{} is disabled but classified as {:?}",
                    record.variant,
                    record.ownership
                );
            }
        }
    }

    #[test]
    fn non_tokenizer_enabled_variants_have_planned_primitives() {
        for record in all_support_records() {
            if record.variant.is_enabled() && record.family != ModelFamily::Tokenizer {
                assert!(
                    !record.planned_primitives.is_empty(),
                    "{} has no planned CUDA primitives",
                    record.variant
                );
            }
        }
    }

    #[test]
    fn candle_owned_family_boundaries_are_explicit() {
        let gemma = support_for_variant(ModelVariant::Gemma31BIt);
        assert_eq!(gemma.ownership, CudaKernelOwnership::CandleOwned);
        assert!(gemma
            .planned_primitives
            .contains(&CudaKernelPrimitive::Attention));

        let lfm = support_for_variant(ModelVariant::Lfm2512BInstructGguf);
        assert_eq!(lfm.ownership, CudaKernelOwnership::CandleOwned);

        let qwen_gguf = support_for_variant(ModelVariant::Qwen38BGguf);
        assert_eq!(qwen_gguf.ownership, CudaKernelOwnership::CandleOwned);
    }

    #[test]
    fn cpu_only_until_routed_family_is_explicit() {
        let sortformer = support_for_variant(ModelVariant::DiarStreamingSortformer4SpkV21);
        assert_eq!(
            sortformer.ownership,
            CudaKernelOwnership::CpuOnlyUntilRouted
        );
        assert!(sortformer
            .planned_primitives
            .contains(&CudaKernelPrimitive::Conformer));
    }

    #[test]
    fn izwi_owned_qwen35_records_delta_net_work() {
        let qwen35 = support_for_variant(ModelVariant::Qwen354BGguf);
        assert_eq!(qwen35.ownership, CudaKernelOwnership::NativeIzwiOwned);
        assert!(qwen35
            .planned_primitives
            .contains(&CudaKernelPrimitive::DeltaNet));
    }
}
