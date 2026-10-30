//! Catalog-to-runtime inference-state certification.
//!
//! This fixture deliberately uses the production adapter inventory. A newly
//! advertised variant, capability, or backend therefore has to receive an
//! explicit topology classification before this test can pass.

use super::*;
use crate::catalog::ModelFamily;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CertifiedTopology {
    Stateless,
    RetainedPaged,
    RetainedPagedRing,
    RetainedPagedRotatingAndInvocationScratch,
    RetainedPagedAndInvocationScratch,
    RetainedPagedPagedAndInvocationScratch,
    RetainedPagedPagedTensorTensorAndInvocationScratch,
    RetainedPagedTensorAndInvocationPaged,
    RetainedPagedPagedTensorAndInvocationPagedPagedTensor,
    RetainedTensorAndInvocationTensor,
    RetainedPagedAndInvocationPagedTensor,
    RetainedPagedAndInvocationPaged,
    RetainedPagedRingAndInvocationPagedRing,
    RetainedPagedRingAndInvocationScratch,
    RetainedPagedStaticAndInvocationPagedStatic,
    RetainedPagedTensorTensor,
    RetainedPagedTensorTensorAndInvocationScratch,
    RetainedAppendRingTensor,
    InvocationPaged,
    InvocationPagedRing,
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
            | Self::RetainedPagedAndInvocationPaged
            | Self::RetainedPagedRingAndInvocationPagedRing
            | Self::RetainedPagedRingAndInvocationScratch
            | Self::RetainedPagedAndInvocationPagedTensor
            | Self::RetainedPagedStaticAndInvocationPagedStatic
            | Self::RetainedPagedRotatingAndInvocationScratch
            | Self::RetainedPagedAndInvocationScratch
            | Self::RetainedPagedPagedAndInvocationScratch
            | Self::RetainedPagedPagedTensorTensorAndInvocationScratch
            | Self::RetainedPagedTensorTensorAndInvocationScratch
            | Self::RetainedAppendRingTensor => InferenceStateRequirement::RetainedAndInvocation,
            Self::RetainedPagedPagedTensorAndInvocationPagedPagedTensor => {
                InferenceStateRequirement::RetainedAndInvocation
            }
            Self::RetainedTensorAndInvocationTensor => {
                InferenceStateRequirement::RetainedAndInvocation
            }
            Self::InvocationPaged
            | Self::InvocationPagedRing
            | Self::InvocationTensor
            | Self::InvocationTensorStatic => InferenceStateRequirement::Invocation,
        }
    }

    /// Whether the route retains paged K/V between requests.
    const fn retains_pages(self) -> bool {
        match self {
            Self::RetainedPaged
            | Self::RetainedPagedRing
            | Self::RetainedPagedRotatingAndInvocationScratch
            | Self::RetainedPagedAndInvocationScratch
            | Self::RetainedPagedPagedAndInvocationScratch
            | Self::RetainedPagedPagedTensorTensorAndInvocationScratch
            | Self::RetainedPagedTensorAndInvocationPaged
            | Self::RetainedPagedPagedTensorAndInvocationPagedPagedTensor
            | Self::RetainedPagedAndInvocationPagedTensor
            | Self::RetainedPagedAndInvocationPaged
            | Self::RetainedPagedRingAndInvocationPagedRing
            | Self::RetainedPagedRingAndInvocationScratch
            | Self::RetainedPagedStaticAndInvocationPagedStatic
            | Self::RetainedPagedTensorTensor
            | Self::RetainedPagedTensorTensorAndInvocationScratch => true,
            Self::Stateless
            | Self::RetainedTensorAndInvocationTensor
            | Self::RetainedAppendRingTensor
            | Self::InvocationPaged
            | Self::InvocationPagedRing
            | Self::InvocationTensor
            | Self::InvocationTensorStatic => false,
        }
    }
}

/// The kind of one state domain, as a topology names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DomainKind {
    Paged,
    StaticAttention,
    Tensor,
    Append,
    Ring,
    StaticTensor,
    Scratch,
}

fn domain_kind(domain: &crate::kv::v2::StateDomainSpec) -> DomainKind {
    use crate::kv::v2::StateDomainSpec;
    match domain {
        StateDomainSpec::PagedAttention(_) => DomainKind::Paged,
        StateDomainSpec::StaticAttention(_) => DomainKind::StaticAttention,
        StateDomainSpec::Tensor(_) => DomainKind::Tensor,
        StateDomainSpec::Append(_) => DomainKind::Append,
        StateDomainSpec::Ring(_) => DomainKind::Ring,
        StateDomainSpec::StaticTensor(_) => DomainKind::StaticTensor,
    }
}

/// What a descriptor holds: its retained domains' kinds in contract order,
/// whether they form one transactional consistency group, and the kinds of
/// every invocation domain of every stage.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DescriptorShape {
    retained: Vec<DomainKind>,
    one_transactional_group: bool,
    invocation: Vec<DomainKind>,
}

fn descriptor_shape(descriptor: &crate::kv::v2::CapabilityStateDescriptorV2) -> DescriptorShape {
    use crate::kv::v2::{
        CheckpointPolicy, InvocationWorkspaceDomain, InvocationWorkspaceSet,
        RetainedStateCapability, StateScope,
    };
    let (retained, one_transactional_group) = match &descriptor.retained {
        RetainedStateCapability::Stateless => (Vec::new(), false),
        RetainedStateCapability::Managed { contract } => (
            contract.domains.iter().map(domain_kind).collect(),
            contract.groups.len() == 1
                && contract.groups[0].domains.len() == contract.domains.len()
                && contract.domains.iter().all(|domain| {
                    domain.scope() == StateScope::Retained
                        && domain.header().checkpoint == CheckpointPolicy::Transactional
                }),
        ),
    };
    let invocation = match &descriptor.invocation {
        InvocationWorkspaceSet::None { .. } => Vec::new(),
        InvocationWorkspaceSet::Bounded { profiles } => profiles
            .iter()
            .flat_map(|profile| &profile.stages)
            .flat_map(|stage| &stage.domains)
            .map(|domain| match domain {
                InvocationWorkspaceDomain::Scratch { .. } => DomainKind::Scratch,
                InvocationWorkspaceDomain::State { state, .. } => domain_kind(state),
            })
            .collect(),
    };
    DescriptorShape {
        retained,
        one_transactional_group,
        invocation,
    }
}

impl CertifiedTopology {
    /// Whether `descriptor` has exactly this topology's shape, for the
    /// topologies certified by shape here (`None` for the others, which the
    /// route matrix certifies by lifetime alone).
    fn admits(self, descriptor: &crate::kv::v2::CapabilityStateDescriptorV2) -> Option<bool> {
        use DomainKind::*;
        let shape = descriptor_shape(descriptor);
        match self {
            Self::RetainedPagedPagedTensorTensorAndInvocationScratch => Some(
                shape.retained == [Paged, Paged, Tensor, Tensor]
                    && shape.one_transactional_group
                    && !shape.invocation.is_empty()
                    && shape.invocation.iter().all(|kind| *kind == Scratch),
            ),
            _ => None,
        }
    }
}

/// Retained-page routes whose lifecycle allocator loads managed state without
/// the rollout provider-eligibility check.
const UNCHECKED_RETAINED_PAGE_ROUTES: &[(ModelFamily, CapabilityKind)] = &[
    (ModelFamily::WhisperAsr, CapabilityKind::Asr),
    (ModelFamily::Voxtral, CapabilityKind::RealtimeAsr),
];

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
        // The Qwen3.5-MoE family shares the dense family's three-domain
        // composite contract (paged KV + F32 recurrent + conv) through the
        // same hybrid decode machinery.
        (Qwen35MoeChat, Chat) => RetainedPagedTensorTensor,
        // Qwen3.8 owns an independent implementation of the same three-state
        // topology so either family can optimize its representation separately.
        (Qwen38Chat, Chat) => RetainedPagedTensorTensor,
        // Gemma alternates full and sliding-window pages; every layer carries
        // the loaded attention softcap in the paged semantic contract.
        (Gemma3Chat, Chat) => RetainedPaged,
        // LFM2 commits sparse attention pages and its ShortConv ring under one
        // retained decoder-token transaction.
        (Lfm2Chat, Chat) => RetainedPagedRing,
        (Lfm3Chat, Chat) => RetainedPagedRing,

        // Qwen TTS retains talker pages plus tensor continuation state and
        // leases a separate predictor page domain per invocation.
        (Qwen3Tts, Tts | StreamingTts) => RetainedPagedTensorAndInvocationPaged,
        (KokoroTts, Tts | StreamingTts) => Stateless,
        // Voxtral TTS retains its LM pages for one request on the engine
        // route; its stages lease scratch only.
        (VoxtralTts, Tts) => RetainedPagedAndInvocationScratch,
        (VibeVoiceTts, Tts) => RetainedPagedPagedTensorAndInvocationPagedPagedTensor,
        (FishS2Tts, Tts | StreamingTts) => RetainedPagedAndInvocationPaged,
        // Fish Audio Drama 3 retains the trunk's and the tower's pages (the
        // tower one row behind) with the DeltaNet's recurrent and conv
        // tensors in one transaction; its stages lease scratch only.
        (FishDrama3Tts, Tts | StreamingTts) => RetainedPagedPagedTensorTensorAndInvocationScratch,
        // Chatterbox retains one page domain per guidance branch (each its
        // own group and arena); the acoustic stage, whole or windowed,
        // leases scratch only.
        (ChatterboxTts, Tts) => RetainedPagedPagedAndInvocationScratch,
        // Maya1 retains its decoder pages; the SNAC window stage leases
        // scratch only.
        (Maya1Tts, Tts) => RetainedPagedAndInvocationScratch,
        // Maya 2 Native retains its backbone pages; the depth decoder and the
        // Mimi window stage lease scratch only.
        (Maya2Tts, Tts) => RetainedPagedAndInvocationScratch,

        // Qwen ASR commits decoder pages and immutable prepared inputs under
        // one retained transaction; long-form leases invocation pages only.
        (Qwen3Asr, Asr) => RetainedPagedTensorAndInvocationPaged,
        (Voxtral, Asr) => InvocationPaged,
        // Realtime Voxtral retains transactional external pages while its
        // absolute-position sliding window rotates the visible cache view.
        (Voxtral, RealtimeAsr) => RetainedPagedRotatingAndInvocationScratch,
        // Normal VibeVoice ASR retains decoder pages after freezing its
        // stochastic tokenizer output; the long/chunk compatibility graph
        // still leases decoder pages plus acoustic/semantic tensor state.
        (VibeVoiceAsr, Asr) => RetainedPagedAndInvocationPagedTensor,
        (WhisperAsr, Asr) => RetainedPagedStaticAndInvocationPagedStatic,
        (ParakeetAsr, Asr) => RetainedTensorAndInvocationTensor,
        (GraniteSpeechAsr, Asr) => RetainedPagedAndInvocationPaged,
        (GraniteSpeechAsr, SpeakerAttributedAsr) => InvocationPaged,
        // Granite Speech 5.0 runs one atomic invocation on leased pages.
        (GraniteSpeech5Asr, Asr) => InvocationPaged,
        // Nemotron 3.5 Transcribe rows commit attention pages, the Mamba SSM
        // state and the convolution history as one transaction; each stage
        // leases its activations as per-row scratch.
        (NemotronTranscribeAsr, Asr) => RetainedPagedTensorTensorAndInvocationScratch,
        (NemotronAsr, Asr) => InvocationTensorStatic,
        // Realtime Nemotron retains append/ring/tensor streaming state and
        // uses invocation-scoped workspace for the exact graph.
        (NemotronAsr, RealtimeAsr) => RetainedAppendRingTensor,

        // LFM2.5 Audio shares paged+ShortConv state across all advertised
        // audio capabilities; generation additionally adds Depthformer pages.
        (Lfm25Audio, Asr) => RetainedPagedRingAndInvocationPagedRing,
        (Lfm25Audio, Tts) => RetainedPagedRingAndInvocationPagedRing,
        (Lfm25Audio, AudioChat | SpeechToSpeech) => InvocationPagedRing,
        // LFM3-Audio speech, voice-chat replies and LiveASR sessions retain
        // the language model's pages (and, but for LiveASR, its ring) between
        // quanta and lease scratch per stage.
        (Lfm3Audio, Tts | AudioChat | SpeechToSpeech | RealtimeAsr) => {
            RetainedPagedRingAndInvocationScratch
        }
        // Transcription retains the same pages and ring; its long form leases
        // the language model's pages and ring for one atomic invocation.
        (Lfm3Audio, Asr) => RetainedPagedRingAndInvocationPagedRing,

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
fn every_checked_retained_page_route_passes_managed_provider_eligibility() {
    // A route the rollout allowlist omits fails every load with "managed KV
    // provider route is not validated", so a new retained-page family must
    // either join the allowlist or be listed as unchecked above.
    let plan = crate::backends::state::negotiate_state_plan(
        &crate::kv::v2::test_contract(),
        &crate::backends::state::StateBackendPlanRequest {
            backend: BackendKind::Cpu,
            device_ordinal: None,
            page_tokens_hint: Some(16),
            storage_dtype_hint: None,
        },
    )
    .unwrap();
    let registry = RuntimeAdapterRegistry::built_in();
    let mut checked_routes = 0usize;

    for variant in ModelVariant::all().iter().copied() {
        if !variant.is_enabled() {
            continue;
        }
        for metadata in registry.capabilities_for(variant) {
            let retains_pages = certified_topology(variant, metadata.capability)
                .is_some_and(CertifiedTopology::retains_pages);
            if !retains_pages
                || UNCHECKED_RETAINED_PAGE_ROUTES.contains(&(variant.family(), metadata.capability))
            {
                continue;
            }
            crate::runtime::rollout::validate_managed_state_plan_eligibility(
                variant,
                metadata.capability,
                &plan,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "retained-page route {variant}/{:?} cannot load: {error}",
                    metadata.capability
                )
            });
            checked_routes += 1;
        }
    }

    assert!(checked_routes > 0, "no retained-page route was checked");
}

/// Drama 3's descriptor has exactly its topology's shape, and every
/// mutation away from it is rejected.
#[test]
fn fish_drama3_certifies_two_paged_and_two_tensor_domains_with_scratch() {
    use crate::kv::v2::{
        CapabilityStateDescriptorV2, CheckpointPolicy, InferenceStateContract,
        InvocationStateCapacity, InvocationWorkspaceDomain, InvocationWorkspaceSet,
        RetainedStateCapability, StateDomainSpec, StateGroupId, StateGroupSpec, StateScope,
        WorkspaceFormula,
    };
    use crate::models::architectures::fish_drama3::physical::{
        state_contract, state_descriptor, test_stage_graph, FishDrama3Workspace,
    };
    let topology = certified_topology(ModelVariant::FishAudioDrama3, CapabilityKind::Tts).unwrap();
    assert_eq!(
        topology,
        CertifiedTopology::RetainedPagedPagedTensorTensorAndInvocationScratch
    );
    assert_eq!(
        certified_topology(ModelVariant::FishAudioDrama3, CapabilityKind::StreamingTts),
        Some(topology)
    );
    assert_eq!(
        topology.lifetime(),
        InferenceStateRequirement::RetainedAndInvocation
    );
    assert!(topology.retains_pages());

    let config = crate::models::architectures::fish_drama3::config::FishDrama3Config::from_json(
        include_str!("../../../tests/fixtures/fish_drama3/config.json"),
        crate::models::architectures::fish_drama3::FishDrama3ConfigPolicy::Published,
    )
    .unwrap();
    let contract = state_contract(&config, candle_core::DType::F32, 16).unwrap();
    let workspace = FishDrama3Workspace::of(&config, 256).unwrap();
    let stages = test_stage_graph(&workspace);
    let descriptor = state_descriptor(&contract, &workspace, &[&stages]).unwrap();
    assert_eq!(topology.admits(&descriptor), Some(true));
    // Shape certification is this topology's alone so far.
    assert_eq!(
        CertifiedTopology::RetainedPagedTensorTensorAndInvocationScratch.admits(&descriptor),
        None
    );

    fn managed(descriptor: &mut CapabilityStateDescriptorV2) -> &mut InferenceStateContract {
        match &mut descriptor.retained {
            RetainedStateCapability::Managed { contract } => contract,
            RetainedStateCapability::Stateless => unreachable!("a managed descriptor"),
        }
    }
    let mutated = |edit: &dyn Fn(&mut CapabilityStateDescriptorV2)| {
        let mut descriptor = descriptor.clone();
        edit(&mut descriptor);
        topology.admits(&descriptor).unwrap()
    };
    // Without the tower's paged domain.
    assert!(!mutated(&|descriptor| {
        let contract = managed(descriptor);
        contract.domains.remove(1);
        contract.groups[0].domains.remove(1);
    }));
    // The tower's pages leased per invocation instead of retained (rev 1's
    // reading of the tower).
    assert!(!mutated(&|descriptor| {
        let contract = managed(descriptor);
        let mut tower = contract.domains.remove(1);
        contract.groups[0].domains.remove(1);
        if let StateDomainSpec::PagedAttention(spec) = &mut tower {
            spec.header.scope = StateScope::Invocation;
            spec.header.checkpoint = CheckpointPolicy::None;
        }
        let InvocationWorkspaceSet::Bounded { profiles } = &mut descriptor.invocation else {
            unreachable!("bounded invocation workspace")
        };
        profiles[0].stages[2]
            .domains
            .push(InvocationWorkspaceDomain::State {
                placement: tower.header().placement,
                state: tower,
                capacity: InvocationStateCapacity::PagedTokens { max_tokens: 16 },
                formula: WorkspaceFormula {
                    fixed_bytes: 1,
                    dimensions: vec![],
                    terms: vec![],
                },
            });
    }));
    // The tensor domains in a group of their own.
    assert!(!mutated(&|descriptor| {
        let contract = managed(descriptor);
        let tensors = contract.groups[0].domains.split_off(2);
        contract.groups.push(StateGroupSpec {
            id: StateGroupId::new(2),
            domains: tensors,
            prefix_shareable: false,
        });
    }));
    // A domain that is not transactional.
    assert!(!mutated(&|descriptor| {
        if let StateDomainSpec::Tensor(spec) = &mut managed(descriptor).domains[3] {
            spec.header.checkpoint = CheckpointPolicy::CopyOnWrite;
        }
    }));
    // A third tensor domain.
    assert!(!mutated(&|descriptor| {
        let contract = managed(descriptor);
        let extra = contract.domains[3].clone();
        contract.domains.push(extra);
    }));
    // No invocation workspace at all.
    assert!(!mutated(&|descriptor| {
        descriptor.invocation = InvocationWorkspaceSet::None {
            stage_graph_fingerprints: vec![],
        };
    }));
    // Stateless.
    assert!(!mutated(&|descriptor| {
        descriptor.retained = RetainedStateCapability::Stateless;
    }));
}

#[test]
fn voxtral_realtime_certifies_transactional_rotating_pages_and_invocation_scratch() {
    assert_eq!(
        certified_topology(
            ModelVariant::VoxtralMini4BRealtime2602,
            CapabilityKind::RealtimeAsr,
        ),
        Some(CertifiedTopology::RetainedPagedRotatingAndInvocationScratch)
    );
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
