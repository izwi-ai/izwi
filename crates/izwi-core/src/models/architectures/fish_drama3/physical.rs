//! The family's engine state contract (§7.2, R1) and its invocation
//! workspace.
//!
//! One consistency group holds four retained domains:
//!
//! | # | Domain | Shape |
//! |---|---|---|
//! | 1 | paged attention, the trunk | the full-attention layers of `layer_types` (published 3, 7, …, 39), 2 KV heads × 256 |
//! | 2 | paged attention, the tower | the 13 tower layers, numbered after the trunk's (published 40..=52), 2 KV heads × 256 |
//! | 3 | tensor, recurrent hidden | one `[value_heads, key_dim, value_dim]` F32 per DeltaNet layer (30 × [32, 128, 128]) |
//! | 4 | tensor, convolution state | one `[kernel - 1, channels]` F32 history per DeltaNet layer, oldest first, the layout `trunk/state.rs` stages (30 × [3, 8192]) |
//!
//! Every domain is retained, `BackendLocalWithHostOffload`, transactional,
//! clocked by decoder rows, and never shares prefixes (prefix reuse with
//! recurrent state is a 62.8 MiB snapshot per prefix, out of v1). Keys are
//! stored after RoPE (`KeyEncoding::Raw`): the trunk rotates NeoX
//! half-split and the tower interleaved pairs, both before they write, so
//! no reader may rotate them again.
//!
//! **Why two paged domains.** A paged view holds one length and prepares
//! one append for all of its layers, and the tower writes one row behind
//! the trunk: after prefill `tower_len == trunk_len - 1`, which every
//! quantum commit asserts. The tower's KV covers the whole sequence, text
//! rows included, and is retained like the trunk's; it is not
//! invocation-scoped.
//!
//! **Invocation workspace.** Scratch only, leased per row: a prefill span's
//! activations (the trunk over the span, then the teacher-forced tower) and
//! a decode row's (a trunk step, the 13 tower layers, the head logits and
//! the sampler's noise). [`FishDrama3Workspace`] prices both from the
//! config.

use candle_core::DType;

use super::config::FishDrama3Config;
use super::model::{FishDrama3PagedDomain, FishDrama3StateGeometry, MAX_REQUEST_POSITIONS};
use super::trunk::deltanet::GDN_CHUNK;
use super::trunk::state::tensor_state_domains;
pub(crate) use super::trunk::state::{CONVOLUTION_STATE_DOMAIN, RECURRENT_STATE_DOMAIN};
use crate::backends::BackendKind;
use crate::engine::{StageDescriptor, StageWorkSelector};
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, AttentionMask, AttentionPattern, CapabilityStateDescriptorV2,
    CheckpointPolicy, InferenceStateContract, InvocationLeaseScope, InvocationStageWorkspace,
    InvocationWorkspaceDomain, InvocationWorkspaceProfile, InvocationWorkspaceSet, KeyEncoding,
    PageSizeConstraint, PagedAttentionDomainSpec, PagedAttentionLayerSpec, PlacementPolicy,
    PrefixPolicy, RetainedStateCapability, StateClock, StateDType, StateDomainHeader,
    StateDomainId, StateDomainSpec, StateGroupId, StateGroupSpec, StateScope, WorkspaceFormula,
    CURRENT_INFERENCE_STATE_ABI,
};

/// The trunk's full-attention KV.
pub(crate) const TRUNK_KV_DOMAIN: StateDomainId = StateDomainId::new(1);
/// The tower's KV, one row behind the trunk's.
pub(crate) const TOWER_KV_DOMAIN: StateDomainId = StateDomainId::new(2);
/// The one consistency group of all four domains.
pub(crate) const STATE_GROUP: StateGroupId = StateGroupId::new(1);
/// Positions one retained row is sized for: the longest prompt and the
/// largest frame budget (§7.5).
pub const STATE_CONTEXT_TOKENS: usize = MAX_REQUEST_POSITIONS;

/// The route's five stages (§7.1), in stage-id order.
pub const PREPARATION_STAGE: &str = "tts.prepare.fish_drama3";
pub const PREFILL_STAGE: &str = "tts.prefill.fish_drama3";
pub const DECODE_STAGE: &str = "tts.decode.fish_drama3";
pub const AUDIO_DECODE_STAGE: &str = "tts.audio_decode.fish_drama3";
pub const FINALIZE_STAGE: &str = "tts.codec.fish_drama3.scalar";

fn contract_error(message: impl Into<String>) -> Error {
    Error::ModelLoadError(format!(
        "Fish Audio Drama 3 state contract: {}",
        message.into()
    ))
}

fn as_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| contract_error(format!("{what} exceeds u32")))
}

/// Opt-in F16 paged KV on the CPU (`1`, `on`, `true`; read at load): half
/// the KV bytes (94 to 47 KiB per position). K, V and the queries round to
/// F16 when written; attention still accumulates in F32. Off by default
/// until a quality run clears it.
pub const CPU_F16_KV_ENV: &str = "IZWI_FISH_DRAMA3_CPU_F16_KV";

/// The paged KV dtype a load on `backend` plans (§6): F32 on the CPU (F16
/// under [`CPU_F16_KV_ENV`]), F16 on Metal, BF16 on CUDA (priced, not loaded
/// yet). A Metal load whose self-check breached stores F32 instead
/// (`FishDrama3Model::kv_dtype`).
pub fn kv_dtype(backend: BackendKind) -> DType {
    match backend {
        BackendKind::Cpu if crate::env::flag_or(CPU_F16_KV_ENV, false) => DType::F16,
        BackendKind::Cpu => DType::F32,
        BackendKind::Metal => DType::F16,
        BackendKind::Cuda => DType::BF16,
    }
}

fn state_dtype(dtype: DType) -> Result<StateDType> {
    match dtype {
        DType::F32 => Ok(StateDType::F32),
        DType::F16 => Ok(StateDType::F16),
        DType::BF16 => Ok(StateDType::Bf16),
        other => Err(contract_error(format!(
            "{other:?} KV storage is not supported"
        ))),
    }
}

/// The retained contract with both paged domains stored as `kv_dtype` in
/// pages of `page_tokens` (preferred) positions.
pub(crate) fn state_contract(
    config: &FishDrama3Config,
    kv_dtype: DType,
    page_tokens: usize,
) -> Result<InferenceStateContract> {
    let geometry = FishDrama3StateGeometry::of(config);
    let dtype = state_dtype(kv_dtype)?;
    let preferred = as_u32(page_tokens.max(1), "page size")?;
    let page_size = PageSizeConstraint {
        min_tokens: 1,
        preferred_tokens: preferred,
        max_tokens: preferred.max(256),
        multiple_of: 1,
    };
    let paged = |id: StateDomainId,
                 domain: &FishDrama3PagedDomain,
                 query_heads: usize|
     -> Result<StateDomainSpec> {
        if domain.layers.is_empty() {
            return Err(contract_error(format!(
                "paged domain {} has no layers",
                id.get()
            )));
        }
        let layers = domain
            .layers
            .iter()
            .map(|&model_layer| {
                Ok(PagedAttentionLayerSpec {
                    model_layer,
                    query_heads: as_u32(query_heads, "query heads")?,
                    kv_heads: as_u32(domain.kv_heads, "KV heads")?,
                    key_head_dim: as_u32(domain.head_dim, "head dimension")?,
                    value_head_dim: as_u32(domain.head_dim, "head dimension")?,
                    pattern: AttentionPattern::Full,
                    mask: AttentionMask::Causal,
                    // Both stacks write keys already rotated.
                    key_encoding: KeyEncoding::Raw,
                    attention_logit_softcap: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(StateDomainSpec::PagedAttention(PagedAttentionDomainSpec {
            header: StateDomainHeader {
                id,
                scope: StateScope::Retained,
                clock: StateClock::DecoderTokens,
                placement: PlacementPolicy::BackendLocalWithHostOffload,
                prefix: PrefixPolicy::Disabled,
                checkpoint: CheckpointPolicy::Transactional,
            },
            layers,
            page_size,
            accepted_dtypes: vec![dtype],
        }))
    };
    let [recurrent, convolution] = tensor_state_domains(&config.trunk)?;
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: vec![
            paged(
                TRUNK_KV_DOMAIN,
                &geometry.trunk,
                config.trunk.attention.heads,
            )?,
            paged(
                TOWER_KV_DOMAIN,
                &geometry.tower,
                config.tower.attention.heads,
            )?,
            recurrent,
            convolution,
        ],
        groups: vec![StateGroupSpec {
            id: STATE_GROUP,
            domains: vec![
                TRUNK_KV_DOMAIN,
                TOWER_KV_DOMAIN,
                RECURRENT_STATE_DOMAIN,
                CONVOLUTION_STATE_DOMAIN,
            ],
            prefix_shareable: false,
        }],
    };
    contract.validate()?;
    Ok(contract)
}

/// The activation workspace of the two stages that run the model, F32 on
/// every backend (the CPU's form; Metal's F16 activations only shrink it).
///
/// A row is priced at its widest point: the residual and its norm, the
/// larger of the two mixers' live buffers, and the MoE's (router logits,
/// the top-k experts' gate, up, product and down rows, the shared expert);
/// a tower layer's merge, `wqkv`, gated attention and SwiGLU. The DeltaNet's
/// chunked prefill adds one chunk's buffers per span; a decode row adds the
/// compact head's and the 9 codebook heads' logits and as many sampling
/// noise draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FishDrama3Workspace {
    /// Rows per prefill span (the load's grid cell).
    pub span_rows: usize,
    /// One span: the trunk over its rows, then the teacher-forced tower.
    pub span_bytes: u64,
    /// One decode row: a trunk step, the tower's 13 layers, the head logits
    /// and the sampler's noise.
    pub decode_row_bytes: u64,
}

impl FishDrama3Workspace {
    pub fn of(config: &FishDrama3Config, span_rows: usize) -> Result<Self> {
        if span_rows == 0 {
            return Err(contract_error("a prefill span needs at least one row"));
        }
        let t = &config.trunk;
        let (h, a, la, moe) = (t.hidden, &t.attention, &t.linear, &t.moe);
        // Packed query and gate, the split queries, gates and attended
        // values, K and V, and the output projection.
        let attention = 2 * a.q_width() + 3 * a.q_width() + 2 * a.kv_width() + h;
        // `in_proj_qkv` and its conv, the normalised q, k and v, `z`, the
        // a and b gates, the delta rule's output and its gated norm, and
        // the output projection.
        let deltanet = 3 * la.conv_channels()
            + la.value_width()
            + 2 * la.value_heads
            + 2 * la.value_width()
            + h;
        let moe_row =
            moe.experts + moe.top_k * (3 * moe.intermediate + h) + 3 * moe.shared_intermediate + h;
        let trunk_row = 2 * h + attention.max(deltanet) + moe_row;
        let w = &config.tower;
        let d = w.dim;
        // The code embedding, its norm, the hidden norm and the merge input;
        // `wqkv`; the attended and gated values; `wo`; w1, w3 and their
        // product; w2.
        let tower_row =
            4 * d + w.wqkv_rows() + 2 * w.attention.q_width() + d + 3 * w.intermediate + d;
        // The compact head (eos and the band) and the residual codebooks'
        // heads.
        let logits = config.ids.semantic_codes() + 1 + w.residual_layers() * w.codebook_vocab;
        let chunk = la.value_heads
            * (GDN_CHUNK * (2 * la.key_head_dim + 2 * la.value_head_dim)
                + 2 * GDN_CHUNK * GDN_CHUNK);
        let row = trunk_row + tower_row + h;
        let bytes = |elements: Option<usize>| -> Result<u64> {
            elements
                .and_then(|elements| elements.checked_mul(4))
                .and_then(|bytes| u64::try_from(bytes).ok())
                .ok_or_else(|| contract_error("workspace size overflows"))
        };
        Ok(Self {
            span_rows,
            span_bytes: bytes(
                span_rows
                    .checked_mul(row)
                    .and_then(|elements| elements.checked_add(chunk)),
            )?,
            decode_row_bytes: bytes(row.checked_add(2 * logits))?,
        })
    }

    /// Span bytes per prompt row, rounded up.
    pub fn span_bytes_per_row(&self) -> u64 {
        self.span_bytes.div_ceil(self.span_rows as u64)
    }
}

/// The capability descriptor of `contract` for `stage_graphs`: the managed
/// retained contract, and per stage the scratch it declares, leased per
/// row. The span stage must declare at least `workspace.span_bytes` a row
/// and the decode stage at least `workspace.decode_row_bytes` (the adapter
/// takes both from the sealed capacity); an under-priced stage, a workspace
/// that is not a whole number of rows, a graph without its span and decode
/// stages, and an atomic graph are refused: the family runs only its
/// retained sequence route.
pub(crate) fn state_descriptor(
    contract: &InferenceStateContract,
    workspace: &FishDrama3Workspace,
    stage_graphs: &[&[StageDescriptor]],
) -> Result<CapabilityStateDescriptorV2> {
    if stage_graphs.is_empty() {
        return Err(contract_error("no execution graph"));
    }
    let max_domain = contract
        .domains
        .iter()
        .map(|domain| domain.id().get())
        .max()
        .ok_or_else(|| contract_error("the retained contract is empty"))?;
    let mut profiles = Vec::with_capacity(stage_graphs.len());
    for stages in stage_graphs {
        let has = |selector| stages.iter().any(|stage| stage.selector == selector);
        if !has(StageWorkSelector::SequencePrefill) || !has(StageWorkSelector::SequenceDecode) {
            return Err(contract_error(
                "the execution graph lacks its prefill span or decode stage",
            ));
        }
        let mut ordered = stages.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|stage| stage.id);
        let mut workspaces = Vec::with_capacity(ordered.len());
        for (index, stage) in ordered.into_iter().enumerate() {
            let required = match stage.selector {
                StageWorkSelector::Atomic => {
                    return Err(contract_error(
                        "the family runs its retained sequence graph, never an atomic one",
                    ))
                }
                StageWorkSelector::SequencePrefill => workspace.span_bytes,
                StageWorkSelector::SequenceDecode => workspace.decode_row_bytes,
                _ => 0,
            };
            let rows = u64::try_from(stage.max_batch_size.max(1))
                .map_err(|_| contract_error("stage rows exceed u64"))?;
            let bytes = if stage.workspace_per_row_bytes == 0 {
                stage.max_workspace_bytes / rows
            } else {
                stage.workspace_per_row_bytes
            };
            if bytes.checked_mul(rows) != Some(stage.max_workspace_bytes) {
                return Err(contract_error(format!(
                    "stage {} declares {} workspace bytes, not a whole number of {rows} rows",
                    stage.name, stage.max_workspace_bytes
                )));
            }
            if bytes < required {
                return Err(contract_error(format!(
                    "stage {} declares {bytes} workspace bytes a row; the family needs {required}",
                    stage.name
                )));
            }
            let mut domains = Vec::new();
            if bytes > 0 {
                let id = max_domain
                    .checked_add(1)
                    .and_then(|id| id.checked_add(as_u32(index, "stage index").ok()?))
                    .ok_or_else(|| contract_error("scratch domain id overflows"))?;
                domains.push(InvocationWorkspaceDomain::Scratch {
                    id: StateDomainId::new(id),
                    placement: PlacementPolicy::BackendLocal,
                    alignment_bytes: 64,
                    zero_on_release: false,
                    formula: WorkspaceFormula {
                        fixed_bytes: bytes,
                        dimensions: vec![],
                        terms: vec![],
                    },
                });
            }
            workspaces.push(InvocationStageWorkspace {
                stage: stage.id,
                lease_scope: InvocationLeaseScope::PerRow,
                groups: Vec::new(),
                domains,
            });
        }
        profiles.push(InvocationWorkspaceProfile {
            stage_graph_fingerprint: stage_graph_fingerprint(stages)?,
            stages: workspaces,
        });
    }
    profiles.sort_unstable_by_key(|profile| profile.stage_graph_fingerprint);
    profiles.dedup();
    let descriptor = CapabilityStateDescriptorV2 {
        abi: CURRENT_INFERENCE_STATE_ABI,
        retained: RetainedStateCapability::Managed {
            contract: contract.clone(),
        },
        invocation: InvocationWorkspaceSet::Bounded { profiles },
    };
    for stages in stage_graphs {
        descriptor.validate_against_stages(stages)?;
    }
    Ok(descriptor)
}

/// The five stages of the family's route (§7.1) as the execution step's
/// adapter will declare them, one row each, the span and decode stages with
/// `workspace`'s bytes: for tests of the descriptor and of the topology
/// certification.
#[cfg(test)]
pub(crate) fn test_stage_graph(workspace: &FishDrama3Workspace) -> Vec<StageDescriptor> {
    use std::time::Duration;

    use crate::engine::{
        ClockedStateSelection, ConcurrencyClass, ExecutionDomain, MembershipSafePoint,
        NativeBatchMode, OutputVisibility, PhysicalLaunchPolicy, StageId, StageProgressKind,
        StageShapePolicy,
    };
    let stage = |id: u32, name: &str, selector, retained: bool| StageDescriptor {
        id: StageId::new(id),
        name: name.into(),
        selector,
        domain: ExecutionDomain::ExecutionGroup,
        progress: StageProgressKind::Iterative,
        concurrency: ConcurrencyClass::Exclusive,
        physical_launch_policy: PhysicalLaunchPolicy::ExecutionGroupExclusive,
        batch_mode: NativeBatchMode::None,
        max_batch_size: 1,
        max_work_units: 1,
        workspace_base_bytes: 0,
        workspace_per_row_bytes: 0,
        workspace_per_work_unit_bytes: 0,
        max_workspace_bytes: 0,
        max_padding_basis_points: 0,
        max_formation_delay: Duration::ZERO,
        shape_policy: StageShapePolicy::Exact,
        membership_safe_point: MembershipSafePoint::OperationBoundary,
        output_visibility: OutputVisibility::AfterQuantumCommit,
        retained_state_selections: retained.then(|| {
            vec![ClockedStateSelection::new(STATE_GROUP, StateClock::DecoderTokens).unwrap()]
        }),
    };
    let mut prefill = stage(1, PREFILL_STAGE, StageWorkSelector::SequencePrefill, true);
    prefill.max_workspace_bytes = workspace.span_bytes;
    let mut decode = stage(2, DECODE_STAGE, StageWorkSelector::SequenceDecode, true);
    decode.workspace_per_row_bytes = workspace.decode_row_bytes;
    decode.max_workspace_bytes = workspace.decode_row_bytes;
    vec![
        stage(
            0,
            PREPARATION_STAGE,
            StageWorkSelector::PreSequencePreparation,
            false,
        ),
        prefill,
        decode,
        stage(
            3,
            AUDIO_DECODE_STAGE,
            StageWorkSelector::SequenceAudioDecode,
            false,
        ),
        stage(
            4,
            FINALIZE_STAGE,
            StageWorkSelector::SequenceFinalize,
            false,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::super::config::tests::{published, synthetic};
    use super::*;
    use crate::kv::v2::{TensorRole, TensorStateDomainSpec};

    fn paged(contract: &InferenceStateContract, index: usize) -> &PagedAttentionDomainSpec {
        match &contract.domains[index] {
            StateDomainSpec::PagedAttention(spec) => spec,
            other => panic!("domain {index} is not paged: {other:?}"),
        }
    }

    fn tensor(contract: &InferenceStateContract, index: usize) -> &TensorStateDomainSpec {
        match &contract.domains[index] {
            StateDomainSpec::Tensor(spec) => spec,
            other => panic!("domain {index} is not a tensor domain: {other:?}"),
        }
    }

    fn model_layers(spec: &PagedAttentionDomainSpec) -> Vec<u32> {
        spec.layers.iter().map(|layer| layer.model_layer).collect()
    }

    /// The contract with its geometry removed: what must not differ between
    /// the published and a synthetic config.
    fn shape(contract: &InferenceStateContract) -> serde_json::Value {
        let mut value = serde_json::to_value(contract).unwrap();
        for domain in value["domains"].as_array_mut().unwrap() {
            if let Some(layers) = domain.get_mut("layers") {
                let first = layers[0].clone();
                for layer in layers.as_array().unwrap() {
                    for key in ["pattern", "mask", "key_encoding", "attention_logit_softcap"] {
                        assert_eq!(layer[key], first[key], "{key}");
                    }
                }
                *layers = serde_json::json!({
                    "pattern": first["pattern"],
                    "mask": first["mask"],
                    "key_encoding": first["key_encoding"],
                    "attention_logit_softcap": first["attention_logit_softcap"],
                });
            }
            if let Some(components) = domain.get_mut("components") {
                let first = components[0].clone();
                *components = serde_json::json!({
                    "role": first["role"],
                    "accepted_dtypes": first["accepted_dtypes"],
                });
            }
        }
        value
    }

    #[test]
    fn the_published_contract_holds_two_paged_and_two_tensor_domains_in_one_group() {
        let config = published();
        let page = crate::models::shared::attention::paged::DEFAULT_KV_PAGE_SIZE;
        let contract = state_contract(&config, kv_dtype(BackendKind::Cpu), page).unwrap();
        assert_eq!(
            contract
                .domains
                .iter()
                .map(|domain| domain.id().get())
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(contract.groups.len(), 1);
        assert_eq!(contract.groups[0].id, STATE_GROUP);
        assert_eq!(
            contract.groups[0].domains,
            vec![
                TRUNK_KV_DOMAIN,
                TOWER_KV_DOMAIN,
                RECURRENT_STATE_DOMAIN,
                CONVOLUTION_STATE_DOMAIN
            ]
        );
        assert!(!contract.groups[0].prefix_shareable);
        for domain in &contract.domains {
            let header = domain.header();
            assert_eq!(header.scope, StateScope::Retained);
            assert_eq!(
                header.placement,
                PlacementPolicy::BackendLocalWithHostOffload
            );
            assert_eq!(header.checkpoint, CheckpointPolicy::Transactional);
            assert_eq!(header.clock, StateClock::DecoderTokens);
            assert_eq!(header.prefix, PrefixPolicy::Disabled);
        }

        let trunk = paged(&contract, 0);
        assert_eq!(
            model_layers(trunk),
            vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
        );
        let tower = paged(&contract, 1);
        assert_eq!(model_layers(tower), (40..=52).collect::<Vec<_>>());
        for spec in [trunk, tower] {
            assert_eq!(spec.accepted_dtypes, vec![StateDType::F32]);
            assert_eq!(spec.page_size.preferred_tokens, 64);
            for layer in &spec.layers {
                assert_eq!(
                    (layer.query_heads, layer.kv_heads, layer.key_head_dim),
                    (16, 2, 256)
                );
                assert_eq!(layer.value_head_dim, 256);
                assert_eq!(layer.key_encoding, KeyEncoding::Raw);
                assert_eq!(layer.pattern, AttentionPattern::Full);
                assert_eq!(layer.mask, AttentionMask::Causal);
            }
        }

        let recurrent = tensor(&contract, 2);
        let convolution = tensor(&contract, 3);
        assert_eq!(recurrent.components.len(), 30);
        assert_eq!(convolution.components.len(), 30);
        let elements = |spec: &TensorStateDomainSpec| {
            spec.components
                .iter()
                .map(|component| {
                    assert_eq!(component.accepted_dtypes, vec![StateDType::F32]);
                    serde_json::to_value(&component.shape).unwrap()["dimensions"][0]["extent"]
                        ["value"]
                        .as_u64()
                        .unwrap()
                })
                .collect::<std::collections::BTreeSet<_>>()
        };
        assert_eq!(elements(recurrent), [32 * 128 * 128].into());
        assert_eq!(elements(convolution), [3 * 8192].into());
        assert!(recurrent
            .components
            .iter()
            .all(|component| component.role == TensorRole::RecurrentHidden));
        assert!(convolution
            .components
            .iter()
            .all(|component| component.role == TensorRole::ConvolutionState));

        // The CPU contract's identity at the default page size: a change to
        // any domain, layer, dtype, page size or policy changes it.
        assert_eq!(
            hex::encode(contract.fingerprint().unwrap()),
            "a8270998e7fdbd27df53e1e64b1eb29c10276b17d374483654e44ae1e4b1e08a"
        );
        let metal = state_contract(&config, kv_dtype(BackendKind::Metal), page).unwrap();
        assert_ne!(
            metal.fingerprint().unwrap(),
            contract.fingerprint().unwrap()
        );
        assert_eq!(paged(&metal, 0).accepted_dtypes, vec![StateDType::F16]);
        let cuda = state_contract(&config, kv_dtype(BackendKind::Cuda), page).unwrap();
        assert_eq!(paged(&cuda, 1).accepted_dtypes, vec![StateDType::Bf16]);
    }

    #[test]
    fn a_synthetic_config_has_the_published_contract_but_its_geometry() {
        let published = state_contract(&published(), DType::F32, 16).unwrap();
        let config = synthetic();
        let contract = state_contract(&config, DType::F32, 16).unwrap();
        assert_eq!(shape(&contract), shape(&published));
        // Its layers come from its own layer_types and tower.
        assert_eq!(model_layers(paged(&contract, 0)), vec![3]);
        assert_eq!(model_layers(paged(&contract, 1)), vec![4, 5, 6]);
        assert_eq!(tensor(&contract, 2).components.len(), 3);
        assert_ne!(
            contract.fingerprint().unwrap(),
            published.fingerprint().unwrap()
        );
    }

    #[test]
    fn kv_bytes_per_position_double_in_f32() {
        let geometry = FishDrama3StateGeometry::of(&published());
        let per_position = |dtype| {
            geometry.trunk.bytes_per_position(dtype) + geometry.tower.bytes_per_position(dtype)
        };
        // 23 layers × K/V × 2 heads × 256: 46 KiB in 16 bits.
        assert_eq!(per_position(kv_dtype(BackendKind::Metal)), 46 * 1024);
        assert_eq!(per_position(kv_dtype(BackendKind::Cuda)), 46 * 1024);
        assert_eq!(per_position(kv_dtype(BackendKind::Cpu)), 92 * 1024);
        assert!(state_contract(&published(), DType::U8, 16).is_err());
    }

    #[test]
    fn the_published_workspace_is_priced_per_span_and_per_decode_row() {
        let workspace = FishDrama3Workspace::of(&published(), 256).unwrap();
        // Trunk row: 2 × 2048 + the DeltaNet's 38,976 (wider than
        // attention's 23,552) + the MoE's 32,512 = 75,584; tower row 48,128;
        // the embedding 2,048. The chunk: 32 × (64 × 512 + 2 × 64 × 64).
        let row = 75_584 + 48_128 + 2_048;
        let chunk = 32 * (64 * 512 + 2 * 64 * 64);
        assert_eq!(workspace.span_bytes, 4 * (256 * row + chunk) as u64);
        // The compact head's 4,097 and the 9 heads' 4,096 logits, and as
        // many noise draws.
        assert_eq!(
            workspace.decode_row_bytes,
            4 * (row + 2 * (4_097 + 9 * 4_096)) as u64
        );
        assert_eq!(
            workspace.span_bytes_per_row(),
            workspace.span_bytes.div_ceil(256)
        );
        assert!(FishDrama3Workspace::of(&published(), 0).is_err());
    }

    #[test]
    fn the_descriptor_leases_span_and_decode_scratch_per_row() {
        let config = published();
        let contract = state_contract(&config, DType::F32, 16).unwrap();
        let workspace = FishDrama3Workspace::of(&config, 256).unwrap();
        let mut stages = test_stage_graph(&workspace);
        // The codec stage's own per-row workspace, two rows of it.
        stages[3].max_batch_size = 2;
        stages[3].workspace_per_row_bytes = 4096;
        stages[3].max_workspace_bytes = 8192;
        let descriptor = state_descriptor(&contract, &workspace, &[&stages]).unwrap();
        let RetainedStateCapability::Managed { contract: retained } = &descriptor.retained else {
            panic!("managed retained state");
        };
        assert_eq!(retained, &contract);
        let InvocationWorkspaceSet::Bounded { profiles } = &descriptor.invocation else {
            panic!("bounded invocation workspace");
        };
        assert_eq!(profiles.len(), 1);
        let scratch = |stage: usize| {
            let stage = &profiles[0].stages[stage];
            assert_eq!(stage.lease_scope, InvocationLeaseScope::PerRow);
            assert!(stage.groups.is_empty());
            match stage.domains.as_slice() {
                [] => None,
                [InvocationWorkspaceDomain::Scratch { id, formula, .. }] => {
                    assert!(id.get() > 4);
                    Some(formula.fixed_bytes)
                }
                other => panic!("scratch only: {other:?}"),
            }
        };
        assert_eq!(scratch(0), None);
        assert_eq!(scratch(1), Some(workspace.span_bytes));
        assert_eq!(scratch(2), Some(workspace.decode_row_bytes));
        assert_eq!(scratch(3), Some(4096));
        assert_eq!(scratch(4), None);

        // An under-priced span or decode stage, or a workspace that is not a
        // whole number of rows.
        let mut cheap = test_stage_graph(&workspace);
        cheap[1].max_workspace_bytes -= 1;
        assert!(state_descriptor(&contract, &workspace, &[&cheap]).is_err());
        let mut cheap = test_stage_graph(&workspace);
        cheap[2].workspace_per_row_bytes -= 1;
        cheap[2].max_workspace_bytes -= 1;
        assert!(state_descriptor(&contract, &workspace, &[&cheap]).is_err());
        let mut ragged = test_stage_graph(&workspace);
        ragged[3].max_batch_size = 2;
        ragged[3].max_workspace_bytes = 4097;
        assert!(state_descriptor(&contract, &workspace, &[&ragged]).is_err());

        // Only its own route.
        let mut atomic = test_stage_graph(&workspace);
        atomic[0].selector = StageWorkSelector::Atomic;
        assert!(state_descriptor(&contract, &workspace, &[&atomic]).is_err());
        let without_decode = test_stage_graph(&workspace)
            .into_iter()
            .filter(|stage| stage.selector != StageWorkSelector::SequenceDecode)
            .collect::<Vec<_>>();
        assert!(state_descriptor(&contract, &workspace, &[&without_decode]).is_err());
        assert!(state_descriptor(&contract, &workspace, &[]).is_err());
    }
}
