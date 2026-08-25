//! Catalog-to-runtime inference-state certification.
//!
//! This fixture deliberately uses the production adapter inventory. A newly
//! advertised variant, capability, or backend therefore has to receive an
//! explicit topology classification before this test can pass.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CertifiedTopology {
    Stateless,
    RetainedPaged,
    RetainedPagedRing,
    RetainedPagedTensorAndInvocationPaged,
    RetainedPagedStaticAndInvocationPagedStatic,
    RetainedPagedTensorTensor,
    RetainedAppendRingTensor,
    InvocationPaged,
    InvocationPagedPaged,
    InvocationPagedRing,
    InvocationPagedTensor,
    InvocationPagedStaticAttention,
    InvocationTensor,
    InvocationTensorStatic,
}

impl CertifiedTopology {
    const fn lifetime(self) -> InferenceStateRequirement {
        match self {
            Self::Stateless => InferenceStateRequirement::Stateless,
            Self::RetainedPaged | Self::RetainedPagedRing | Self::RetainedPagedTensorTensor => {
                InferenceStateRequirement::Retained
            }
            Self::RetainedPagedTensorAndInvocationPaged
            | Self::RetainedPagedStaticAndInvocationPagedStatic
            | Self::RetainedAppendRingTensor => InferenceStateRequirement::RetainedAndInvocation,
            Self::InvocationPaged
            | Self::InvocationPagedPaged
            | Self::InvocationPagedRing
            | Self::InvocationPagedTensor
            | Self::InvocationPagedStaticAttention
            | Self::InvocationTensor
            | Self::InvocationTensorStatic => InferenceStateRequirement::Invocation,
        }
    }
}

fn certified_topology(
    variant: ModelVariant,
    capability: CapabilityKind,
) -> Option<CertifiedTopology> {
    use CapabilityKind::*;
    use CertifiedTopology::*;
    use ModelFamily::*;

    Some(match (variant.family(), capability) {
        // Autoregressive text routes retain scheduler-owned paged state.
        (Qwen3Chat, Chat) => RetainedPaged,
        // Qwen3.5 commits full-attention pages, recurrent state, and
        // convolution state as one composite transaction.
        (Qwen35Chat, Chat) => RetainedPagedTensorTensor,
        // Qwen3.8 owns an independent implementation of the same three-state
        // topology so either family can optimize its representation separately.
        (Qwen38Chat, Chat) => RetainedPagedTensorTensor,
        // Gemma alternates full and sliding-window pages; every layer carries
        // the loaded attention softcap in the paged semantic contract.
        (Gemma3Chat, Chat) => RetainedPaged,
        // LFM2 commits sparse attention pages and its ShortConv ring under one
        // retained decoder-token transaction.
        (Lfm2Chat, Chat) => RetainedPagedRing,

        // Qwen TTS retains talker pages plus tensor continuation state and
        // leases a separate predictor page domain per invocation.
        (Qwen3Tts, Tts | StreamingTts) => RetainedPagedTensorAndInvocationPaged,
        (KokoroTts, Tts | StreamingTts) => Stateless,
        (VoxtralTts, Tts) => InvocationPaged,
        (VibeVoiceTts, Tts) => InvocationPagedTensor,
        (FishS2Tts, Tts) => InvocationPagedPaged,

        // Qwen ASR commits decoder pages and immutable prepared inputs under
        // one retained transaction; long-form leases invocation pages only.
        (Qwen3Asr, Asr) => RetainedPagedTensorAndInvocationPaged,
        (Voxtral, Asr) => InvocationPaged,
        (VibeVoiceAsr, Asr) => InvocationPagedTensor,
        (WhisperAsr, Asr) => RetainedPagedStaticAndInvocationPagedStatic,
        (ParakeetAsr, Asr) => InvocationTensor,
        (GraniteSpeechAsr, Asr | SpeakerAttributedAsr) => InvocationPaged,
        (NemotronAsr, Asr) => InvocationTensorStatic,
        // Realtime Nemotron retains append/ring/tensor streaming state and
        // uses invocation-scoped workspace for the exact graph.
        (NemotronAsr, RealtimeAsr) => RetainedAppendRingTensor,

        // LFM2.5 Audio shares paged+ShortConv state across all advertised
        // audio capabilities; generation additionally adds Depthformer pages.
        (Lfm25Audio, Asr | Tts | AudioChat | SpeechToSpeech) => InvocationPagedRing,

        (SortformerDiarization, Diarization) => InvocationTensor,
        (Qwen3ForcedAligner, ForcedAlignment) => Stateless,
        (ModelFamily::Tokenizer, CapabilityKind::Tokenizer) => Stateless,
        _ => return None,
    })
}

#[test]
fn every_enabled_catalog_backend_route_has_an_explicit_state_topology() {
    let registry = RuntimeAdapterRegistry::built_in();
    let backends = [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda];
    let mut certified_cells = 0usize;

    for variant in ModelVariant::all().iter().copied() {
        if !variant.is_enabled() {
            continue;
        }
        let capabilities = registry.capabilities_for(variant);
        assert!(
            !capabilities.is_empty(),
            "enabled catalog variant {variant} has no explicit runtime route"
        );

        for metadata in capabilities {
            let topology = certified_topology(variant, metadata.capability).unwrap_or_else(|| {
                panic!(
                    "catalog route {variant}/{:?} has no state-topology certification cell",
                    metadata.capability
                )
            });
            assert_eq!(
                metadata.state_requirement,
                topology.lifetime(),
                "catalog route {variant}/{:?} disagrees with its physical-state lifetime",
                metadata.capability
            );

            for backend in backends {
                // This uses the same factory selection used at load time. A
                // backend cell cannot silently pass through an absent adapter.
                registry
                    .loaded_adapter_factory(metadata, backend)
                    .unwrap_or_else(|error| {
                        panic!(
                            "catalog route {variant}/{:?}/{backend:?} has no loadable adapter: {error}",
                            metadata.capability
                        )
                    });
                certified_cells += 1;
            }
        }
    }

    assert!(certified_cells > 0, "certification matrix was empty");
}

#[test]
fn disabled_catalog_variants_are_explicitly_outside_the_route_matrix() {
    let disabled = ModelVariant::all()
        .iter()
        .copied()
        .filter(|variant| !variant.is_enabled())
        .collect::<Vec<_>>();
    assert!(
        !disabled.is_empty(),
        "fixture must exercise rejected variants"
    );
    for variant in disabled {
        assert_eq!(
            variant.cuda_support_level(),
            crate::catalog::CudaSupportLevel::Disabled,
            "disabled catalog variant {variant} must be rejected explicitly"
        );
    }
}
