//! Physical state contracts shared by the LFM2 chat and audio capabilities.

use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::kv::v2::{
    stage_graph_fingerprint, AttentionMask, AttentionPattern, BoundedShape,
    CapabilityStateDescriptorV2, CheckpointPolicy, ComponentShapeInstantiation, DomainStepIntent,
    InferenceStateContract, InvocationLeaseScope, InvocationStageWorkspace,
    InvocationStateCapacity, InvocationWorkspaceDomain, InvocationWorkspaceProfile,
    InvocationWorkspaceSet, KeyEncoding, PageSizeConstraint, PagedAttentionDomainSpec,
    PagedAttentionLayerSpec, PlacementPolicy, PrefixPolicy, RetainedStateCapability,
    RingStateDomainSpec, ShapeAxis, ShapeDimension, ShapeDimensionValue, ShapeExtent, StateClock,
    StateComponentId, StateDType, StateDomainHeader, StateDomainId, StateDomainSpec, StateGroupId,
    StateGroupSpec, StateScope, StateUpdateKind, TensorComponentSpec, TensorRole, WorkspaceFormula,
    CURRENT_INFERENCE_STATE_ABI,
};
use crate::models::shared::attention::paged::default_kv_page_size;

use super::config::Lfm2BackboneConfig;

pub(crate) const LFM2_ATTENTION_STATE_DOMAIN: StateDomainId = StateDomainId::new(1);
pub(crate) const LFM2_SHORTCONV_STATE_DOMAIN: StateDomainId = StateDomainId::new(2);
const LFM2_MAIN_STATE_GROUP: StateGroupId = StateGroupId::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Lfm2StateIds {
    pub(crate) attention: StateDomainId,
    pub(crate) shortconv: StateDomainId,
    pub(crate) main_group: StateGroupId,
}

impl Lfm2StateIds {
    const CANONICAL: Self = Self {
        attention: LFM2_ATTENTION_STATE_DOMAIN,
        shortconv: LFM2_SHORTCONV_STATE_DOMAIN,
        main_group: LFM2_MAIN_STATE_GROUP,
    };
}

#[derive(Debug, Clone)]
pub(crate) struct Lfm2PhysicalStateSpec {
    pub(crate) descriptor: CapabilityStateDescriptorV2,
    pub(crate) invocation: InferenceStateContract,
}

/// Canonical mapping between loaded LFM layers and physical state ordinals.
///
/// Attention remains sparse in model-layer space while the backend arena uses
/// dense physical ordinals. ShortConv components are dense and ordered by
/// model-layer occurrence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Lfm2StateLayout {
    pub(crate) attention_model_layers: Vec<u32>,
    pub(crate) shortconv_components: Vec<(usize, StateComponentId)>,
}

impl Lfm2StateLayout {
    pub(crate) fn from_config(config: &Lfm2BackboneConfig) -> Result<Self> {
        validate_config(config)?;
        let mut attention_model_layers = Vec::new();
        let mut shortconv_components = Vec::new();
        for (model_layer, kv_heads) in config.attention_head_count_kv.iter().copied().enumerate() {
            if kv_heads > 0 {
                attention_model_layers.push(u32::try_from(model_layer).map_err(|_| {
                    Error::ModelLoadError("LFM2 model-layer index exceeds u32".into())
                })?);
            } else {
                let component = u32::try_from(shortconv_components.len() + 1).map_err(|_| {
                    Error::ModelLoadError("LFM2 ShortConv component count exceeds u32".into())
                })?;
                shortconv_components.push((model_layer, StateComponentId::new(component)));
            }
        }
        if attention_model_layers.is_empty() || shortconv_components.is_empty() {
            return Err(Error::ModelLoadError(
                "LFM2 physical state requires both attention and ShortConv layers".into(),
            ));
        }
        Ok(Self {
            attention_model_layers,
            shortconv_components,
        })
    }

    pub(crate) fn shortconv_component(&self, model_layer: usize) -> Result<StateComponentId> {
        self.shortconv_components
            .iter()
            .find(|(candidate, _)| *candidate == model_layer)
            .map(|(_, component)| *component)
            .ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "LFM2 model layer {model_layer} is not a ShortConv component"
                ))
            })
    }

    pub(crate) fn ring_step_intent(
        &self,
        domain: StateDomainId,
        expected_cursor: usize,
        batch: usize,
        hidden: usize,
        steps: usize,
    ) -> Result<DomainStepIntent> {
        let expected_cursor = u64::try_from(expected_cursor)
            .map_err(|_| Error::InvalidInput("LFM2 ring cursor exceeds u64".into()))?;
        let steps = u64::try_from(steps)
            .map_err(|_| Error::InvalidInput("LFM2 ring step count exceeds u64".into()))?;
        let target_cursor = expected_cursor
            .checked_add(steps)
            .ok_or_else(|| Error::InvalidInput("LFM2 ring cursor overflow".into()))?;
        let batch = u64::try_from(batch)
            .map_err(|_| Error::InvalidInput("LFM2 ring batch exceeds u64".into()))?;
        let hidden = u64::try_from(hidden)
            .map_err(|_| Error::InvalidInput("LFM2 ring hidden width exceeds u64".into()))?;
        Ok(DomainStepIntent {
            domain,
            expected_cursor,
            target_cursor,
            update: StateUpdateKind::RingAdvance {
                steps,
                components_per_step: self
                    .shortconv_components
                    .iter()
                    .map(|(_, component)| ComponentShapeInstantiation {
                        component: *component,
                        dimensions: vec![
                            ShapeDimensionValue {
                                axis: ShapeAxis::Batch,
                                units: batch,
                            },
                            ShapeDimensionValue {
                                axis: ShapeAxis::Hidden,
                                units: hidden,
                            },
                        ],
                    })
                    .collect(),
            },
        })
    }
}

pub(crate) fn lfm2_physical_state_spec(
    config: &Lfm2BackboneConfig,
    stage_graphs: &[&[StageDescriptor]],
) -> Result<Lfm2PhysicalStateSpec> {
    if stage_graphs.is_empty() {
        return Err(Error::ModelLoadError(
            "LFM2 physical state has no execution graph".into(),
        ));
    }
    let layout = Lfm2StateLayout::from_config(config)?;
    let invocation = lfm2_main_invocation_contract(config, &layout, Lfm2StateIds::CANONICAL)?;
    let max_tokens = u64::try_from(config.context_length)
        .map_err(|_| Error::ModelLoadError("LFM2 context exceeds u64".into()))?;
    let max_domain_id = invocation
        .domains
        .iter()
        .map(|domain| domain.id().get())
        .max()
        .ok_or_else(|| Error::ModelLoadError("LFM2 invocation contract is empty".into()))?;

    let mut profiles = Vec::with_capacity(stage_graphs.len());
    for stages in stage_graphs {
        let mut ordered = stages.iter().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|stage| stage.id);
        let mut invocation_stages = Vec::with_capacity(ordered.len());
        for (index, stage) in ordered.into_iter().enumerate() {
            let mut domains = invocation
                .domains
                .iter()
                .cloned()
                .map(|state| {
                    let (fixed_bytes, capacity) = match &state {
                        StateDomainSpec::PagedAttention(_) => (
                            paged_f32_invocation_bytes(&state, max_tokens)?,
                            InvocationStateCapacity::decoder_context(max_tokens)?,
                        ),
                        StateDomainSpec::Ring(_) => (
                            ring_f32_invocation_bytes(&state)?,
                            InvocationStateCapacity::SemanticBounded,
                        ),
                        _ => {
                            return Err(Error::ModelLoadError(
                                "LFM2 invocation contract contains an unsupported state kind"
                                    .into(),
                            ));
                        }
                    };
                    Ok(InvocationWorkspaceDomain::State {
                        placement: state.header().placement,
                        formula: WorkspaceFormula {
                            fixed_bytes,
                            dimensions: vec![],
                            terms: vec![],
                        },
                        state,
                        capacity,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if stage.max_workspace_bytes > 0 {
                let scratch_id = max_domain_id
                    .checked_add(u32::try_from(index + 1).map_err(|_| {
                        Error::ModelLoadError("LFM2 execution stage count exceeds u32".into())
                    })?)
                    .ok_or_else(|| {
                        Error::ModelLoadError("LFM2 scratch domain id overflow".into())
                    })?;
                domains.push(InvocationWorkspaceDomain::Scratch {
                    id: StateDomainId::new(scratch_id),
                    placement: PlacementPolicy::BackendLocal,
                    alignment_bytes: 64,
                    zero_on_release: false,
                    formula: WorkspaceFormula {
                        fixed_bytes: stage.max_workspace_bytes,
                        dimensions: vec![],
                        terms: vec![],
                    },
                });
            }
            invocation_stages.push(InvocationStageWorkspace {
                stage: stage.id,
                lease_scope: InvocationLeaseScope::PerRow,
                groups: invocation.groups.clone(),
                domains,
            });
        }
        profiles.push(InvocationWorkspaceProfile {
            stage_graph_fingerprint: stage_graph_fingerprint(stages)?,
            stages: invocation_stages,
        });
    }
    profiles.sort_unstable_by_key(|profile| profile.stage_graph_fingerprint);
    profiles.dedup();
    let descriptor = CapabilityStateDescriptorV2 {
        abi: CURRENT_INFERENCE_STATE_ABI,
        retained: RetainedStateCapability::Stateless,
        invocation: InvocationWorkspaceSet::Bounded { profiles },
    };
    for stages in stage_graphs {
        descriptor.validate_against_stages(stages)?;
    }
    Ok(Lfm2PhysicalStateSpec {
        descriptor,
        invocation,
    })
}

pub(crate) fn lfm2_main_invocation_contract(
    config: &Lfm2BackboneConfig,
    layout: &Lfm2StateLayout,
    ids: Lfm2StateIds,
) -> Result<InferenceStateContract> {
    lfm2_main_contract(config, layout, ids, invocation_header)
}

/// Scheduler-owned cache contract for incremental LFM2 chat.
///
/// The semantic domains intentionally remain identical to the invocation
/// contract shared with LFM2.5 Audio. Only their lifetime, placement, and
/// checkpoint policy change so paged attention and ShortConv advance under one
/// retained row transaction.
pub(crate) fn lfm2_managed_cache_contract(
    config: &Lfm2BackboneConfig,
) -> Result<InferenceStateContract> {
    let layout = Lfm2StateLayout::from_config(config)?;
    lfm2_main_contract(config, &layout, Lfm2StateIds::CANONICAL, retained_header)
}

fn lfm2_main_contract(
    config: &Lfm2BackboneConfig,
    layout: &Lfm2StateLayout,
    ids: Lfm2StateIds,
    header: fn(StateDomainId, StateClock) -> StateDomainHeader,
) -> Result<InferenceStateContract> {
    let head_dim = config.embedding_length / config.attention_head_count;
    let query_heads = u32::try_from(config.attention_head_count)
        .map_err(|_| Error::ModelLoadError("LFM2 query-head count exceeds u32".into()))?;
    let head_dim = u32::try_from(head_dim)
        .map_err(|_| Error::ModelLoadError("LFM2 attention head dimension exceeds u32".into()))?;
    let pattern = match config.attention_sliding_window {
        Some(window_tokens) => AttentionPattern::SlidingWindow {
            window_tokens: u32::try_from(window_tokens)
                .map_err(|_| Error::ModelLoadError("LFM2 sliding window exceeds u32".into()))?,
        },
        None => AttentionPattern::Full,
    };
    let layers = layout
        .attention_model_layers
        .iter()
        .map(|model_layer| {
            let kv_heads = config.attention_head_count_kv[*model_layer as usize];
            Ok(PagedAttentionLayerSpec {
                model_layer: *model_layer,
                query_heads,
                kv_heads: u32::try_from(kv_heads)
                    .map_err(|_| Error::ModelLoadError("LFM2 KV-head count exceeds u32".into()))?,
                key_head_dim: head_dim,
                value_head_dim: head_dim,
                pattern,
                mask: AttentionMask::Causal,
                key_encoding: KeyEncoding::Rotary {
                    rotary_dim: head_dim,
                },
                attention_logit_softcap: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let preferred_tokens = u32::try_from(default_kv_page_size())
        .map_err(|_| Error::ModelLoadError("LFM2 page size exceeds u32".into()))?;
    let attention = StateDomainSpec::PagedAttention(PagedAttentionDomainSpec {
        header: header(ids.attention, StateClock::DecoderTokens),
        layers,
        page_size: PageSizeConstraint {
            min_tokens: 1,
            preferred_tokens,
            max_tokens: preferred_tokens.max(256),
            multiple_of: 1,
        },
        accepted_dtypes: vec![StateDType::F32],
    });
    let hidden = u64::try_from(config.embedding_length)
        .map_err(|_| Error::ModelLoadError("LFM2 hidden width exceeds u64".into()))?;
    let shortconv = StateDomainSpec::Ring(RingStateDomainSpec {
        header: header(ids.shortconv, StateClock::DecoderTokens),
        components_per_step: layout
            .shortconv_components
            .iter()
            .map(|(_, component)| TensorComponentSpec {
                id: *component,
                role: TensorRole::ConvolutionState,
                shape: BoundedShape {
                    dimensions: vec![
                        ShapeDimension {
                            axis: ShapeAxis::Batch,
                            extent: ShapeExtent::Fixed { value: 1 },
                        },
                        ShapeDimension {
                            axis: ShapeAxis::Hidden,
                            extent: ShapeExtent::Fixed { value: hidden },
                        },
                    ],
                },
                accepted_dtypes: vec![StateDType::F32],
            })
            .collect(),
        capacity_steps: u64::try_from(config.shortconv_l_cache)
            .map_err(|_| Error::ModelLoadError("LFM2 ShortConv capacity exceeds u64".into()))?,
    });
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: vec![attention, shortconv],
        groups: vec![StateGroupSpec {
            id: ids.main_group,
            domains: vec![ids.attention, ids.shortconv],
            prefix_shareable: false,
        }],
    };
    contract.validate()?;
    Ok(contract)
}

pub(crate) fn invocation_header(id: StateDomainId, clock: StateClock) -> StateDomainHeader {
    StateDomainHeader {
        id,
        scope: StateScope::Invocation,
        clock,
        placement: PlacementPolicy::BackendLocal,
        prefix: PrefixPolicy::Disabled,
        checkpoint: CheckpointPolicy::None,
    }
}

fn retained_header(id: StateDomainId, clock: StateClock) -> StateDomainHeader {
    StateDomainHeader {
        id,
        scope: StateScope::Retained,
        clock,
        placement: PlacementPolicy::BackendLocalWithHostOffload,
        prefix: PrefixPolicy::Disabled,
        checkpoint: CheckpointPolicy::Transactional,
    }
}

fn validate_config(config: &Lfm2BackboneConfig) -> Result<()> {
    if config.block_count == 0
        || config.context_length == 0
        || config.embedding_length == 0
        || config.attention_head_count == 0
        || !config
            .embedding_length
            .is_multiple_of(config.attention_head_count)
        || config.attention_head_count_kv.len() != config.block_count
        || config.shortconv_l_cache == 0
        || config.attention_sliding_window == Some(0)
        || config
            .attention_head_count_kv
            .iter()
            .any(|kv_heads| *kv_heads > 0 && !config.attention_head_count.is_multiple_of(*kv_heads))
    {
        return Err(Error::ModelLoadError(
            "LFM2 physical state received invalid loaded backbone geometry".into(),
        ));
    }
    Ok(())
}

pub(crate) fn paged_f32_invocation_bytes(state: &StateDomainSpec, max_tokens: u64) -> Result<u64> {
    let StateDomainSpec::PagedAttention(spec) = state else {
        return Err(Error::ModelLoadError(
            "LFM2 paged byte bound received non-paged state".into(),
        ));
    };
    let mut rounded_tokens = 0_u64;
    for candidate in spec.page_size.min_tokens..=spec.page_size.max_tokens {
        if !spec.page_size.accepts(candidate) {
            continue;
        }
        let page_tokens = u64::from(candidate);
        let candidate_tokens = max_tokens
            .checked_add(page_tokens.saturating_sub(1))
            .and_then(|tokens| tokens.checked_div(page_tokens))
            .and_then(|pages| pages.checked_mul(page_tokens))
            .ok_or_else(|| Error::ModelLoadError("LFM2 page capacity overflow".into()))?;
        rounded_tokens = rounded_tokens.max(candidate_tokens);
    }
    if rounded_tokens == 0 {
        return Err(Error::ModelLoadError(
            "LFM2 page constraint has no admissible size".into(),
        ));
    }
    spec.layers
        .iter()
        .try_fold(0_u64, |total, layer| {
            let per_token = u64::from(layer.kv_heads)
                .checked_mul(u64::from(layer.key_head_dim) + u64::from(layer.value_head_dim))
                .and_then(|elements| elements.checked_mul(4))
                .ok_or_else(|| Error::ModelLoadError("LFM2 KV geometry overflow".into()))?;
            total
                .checked_add(per_token)
                .ok_or_else(|| Error::ModelLoadError("LFM2 KV geometry overflow".into()))
        })?
        .checked_mul(rounded_tokens)
        .ok_or_else(|| Error::ModelLoadError("LFM2 paged byte bound overflow".into()))
}

pub(crate) fn ring_f32_invocation_bytes(state: &StateDomainSpec) -> Result<u64> {
    let StateDomainSpec::Ring(spec) = state else {
        return Err(Error::ModelLoadError(
            "LFM2 ring byte bound received non-ring state".into(),
        ));
    };
    spec.components_per_step
        .iter()
        .try_fold(0_u64, |total, component| {
            total
                .checked_add(
                    component
                        .shape
                        .maximum_elements()?
                        .checked_mul(4)
                        .ok_or_else(|| {
                            Error::ModelLoadError("LFM2 ShortConv component bytes overflow".into())
                        })?,
                )
                .ok_or_else(|| Error::ModelLoadError("LFM2 ShortConv bytes overflow".into()))
        })?
        .checked_mul(spec.capacity_steps)
        .ok_or_else(|| Error::ModelLoadError("LFM2 ShortConv byte bound overflow".into()))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::engine::{
        ConcurrencyClass, ExecutionDomain, MembershipSafePoint, NativeBatchMode, OutputVisibility,
        StageDescriptor, StageId, StageProgressKind, StageShapePolicy, StageWorkSelector,
    };

    fn config() -> Lfm2BackboneConfig {
        Lfm2BackboneConfig {
            architecture: "lfm2".into(),
            block_count: 5,
            context_length: 17,
            embedding_length: 16,
            embedding_length_out: None,
            feed_forward_length: Some(32),
            attention_head_count: 4,
            attention_head_count_kv: vec![0, 2, 0, 2, 0],
            attention_layer_norm_rms_epsilon: 1e-5,
            attention_sliding_window: Some(8),
            rope_freq_base: 1_000_000.0,
            shortconv_l_cache: 3,
        }
    }

    #[test]
    fn layout_uses_sparse_attention_layers_and_dense_shortconv_components() {
        let layout = Lfm2StateLayout::from_config(&config()).unwrap();
        assert_eq!(layout.attention_model_layers, vec![1, 3]);
        assert_eq!(
            layout.shortconv_components,
            vec![
                (0, StateComponentId::new(1)),
                (2, StateComponentId::new(2)),
                (4, StateComponentId::new(3)),
            ]
        );
        let ring_domain = StateDomainId::new(9);
        let intent = layout.ring_step_intent(ring_domain, 7, 1, 16, 2).unwrap();
        assert_eq!(intent.domain, ring_domain);
        assert_eq!(intent.expected_cursor, 7);
        assert_eq!(intent.target_cursor, 9);
        let StateUpdateKind::RingAdvance {
            steps,
            components_per_step,
        } = intent.update
        else {
            panic!("LFM2 ShortConv intent must advance a ring");
        };
        assert_eq!(steps, 2);
        assert_eq!(components_per_step.len(), 3);
        assert!(components_per_step.iter().all(|component| {
            component.dimensions
                == vec![
                    ShapeDimensionValue {
                        axis: ShapeAxis::Batch,
                        units: 1,
                    },
                    ShapeDimensionValue {
                        axis: ShapeAxis::Hidden,
                        units: 16,
                    },
                ]
        }));
    }

    #[test]
    fn managed_contract_declares_one_transactional_retained_paged_ring_group() {
        let contract = lfm2_managed_cache_contract(&config()).unwrap();
        assert_eq!(contract.domains.len(), 2);
        assert_eq!(contract.groups.len(), 1);
        assert_eq!(
            contract.groups[0].domains,
            vec![LFM2_ATTENTION_STATE_DOMAIN, LFM2_SHORTCONV_STATE_DOMAIN]
        );
        assert!(!contract.groups[0].prefix_shareable);
        for domain in &contract.domains {
            assert_eq!(domain.header().scope, StateScope::Retained);
            assert_eq!(domain.header().clock, StateClock::DecoderTokens);
            assert_eq!(
                domain.header().placement,
                PlacementPolicy::BackendLocalWithHostOffload
            );
            assert_eq!(domain.header().prefix, PrefixPolicy::Disabled);
            assert_eq!(domain.header().checkpoint, CheckpointPolicy::Transactional);
        }

        let StateDomainSpec::PagedAttention(attention) = &contract.domains[0] else {
            panic!("first domain must be paged attention");
        };
        assert_eq!(
            attention
                .layers
                .iter()
                .map(|layer| layer.model_layer)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert!(attention.layers.iter().all(|layer| {
            layer.pattern == AttentionPattern::SlidingWindow { window_tokens: 8 }
        }));

        let StateDomainSpec::Ring(shortconv) = &contract.domains[1] else {
            panic!("second domain must be ShortConv ring");
        };
        assert_eq!(shortconv.capacity_steps, 3);
        assert_eq!(shortconv.components_per_step.len(), 3);
        assert!(shortconv.components_per_step.iter().all(|component| {
            component.shape.dimensions
                == vec![
                    ShapeDimension {
                        axis: ShapeAxis::Batch,
                        extent: ShapeExtent::Fixed { value: 1 },
                    },
                    ShapeDimension {
                        axis: ShapeAxis::Hidden,
                        extent: ShapeExtent::Fixed { value: 16 },
                    },
                ]
                && component.accepted_dtypes == vec![StateDType::F32]
        }));
    }

    #[test]
    fn physical_spec_declares_one_atomic_paged_and_ring_group() {
        let stage = StageDescriptor {
            id: StageId::new(1),
            name: "lfm2_decode".into(),
            selector: StageWorkSelector::Atomic,
            domain: ExecutionDomain::ExecutionGroup,
            progress: StageProgressKind::Atomic,
            concurrency: ConcurrencyClass::Exclusive,
            physical_launch_policy: crate::engine::PhysicalLaunchPolicy::ExecutionGroupExclusive,
            batch_mode: NativeBatchMode::None,
            max_batch_size: 1,
            max_work_units: 1,
            workspace_base_bytes: 256,
            workspace_per_row_bytes: 0,
            workspace_per_work_unit_bytes: 0,
            max_workspace_bytes: 256,
            max_padding_basis_points: 0,
            max_formation_delay: Duration::ZERO,
            shape_policy: StageShapePolicy::Exact,
            membership_safe_point: MembershipSafePoint::OperationBoundary,
            output_visibility: OutputVisibility::AfterQuantumCommit,
            retained_state_selections: None,
        };
        let spec = lfm2_physical_state_spec(&config(), &[&[stage]]).unwrap();
        assert_eq!(spec.invocation.domains.len(), 2);
        assert_eq!(
            spec.invocation.groups[0].domains,
            vec![LFM2_ATTENTION_STATE_DOMAIN, LFM2_SHORTCONV_STATE_DOMAIN]
        );
        assert!(spec.invocation.domains.iter().all(|domain| {
            domain.header().scope == StateScope::Invocation
                && domain.header().placement == PlacementPolicy::BackendLocal
                && domain.header().checkpoint == CheckpointPolicy::None
        }));
        let StateDomainSpec::PagedAttention(attention) = &spec.invocation.domains[0] else {
            panic!("first domain must be paged attention");
        };
        assert_eq!(
            attention
                .layers
                .iter()
                .map(|layer| layer.model_layer)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert!(attention.layers.iter().all(|layer| {
            layer.pattern == AttentionPattern::SlidingWindow { window_tokens: 8 }
        }));
        assert_eq!(
            paged_f32_invocation_bytes(&spec.invocation.domains[0], 17).unwrap(),
            32_768
        );
        let StateDomainSpec::Ring(shortconv) = &spec.invocation.domains[1] else {
            panic!("second domain must be ShortConv ring");
        };
        assert_eq!(shortconv.capacity_steps, 3);
        assert_eq!(shortconv.components_per_step.len(), 3);
    }

    #[test]
    fn production_context_cost_exposes_full_slot_multiplication_without_cuda() {
        let config = Lfm2BackboneConfig {
            architecture: "lfm2".into(),
            block_count: 16,
            context_length: 128_000,
            embedding_length: 2_048,
            embedding_length_out: None,
            feed_forward_length: Some(10_496),
            attention_head_count: 32,
            attention_head_count_kv: vec![8, 0, 0, 8, 0, 8, 0, 0, 8, 0, 8, 0, 0, 8, 0, 0],
            attention_layer_norm_rms_epsilon: 1e-5,
            attention_sliding_window: None,
            rope_freq_base: 1_000_000.0,
            shortconv_l_cache: 3,
        };
        let layout = Lfm2StateLayout::from_config(&config).unwrap();
        let contract =
            lfm2_main_invocation_contract(&config, &layout, Lfm2StateIds::CANONICAL).unwrap();
        let per_slot = paged_f32_invocation_bytes(&contract.domains[0], 128_000).unwrap();

        const L40S_BYTES: u64 = 48 * 1024 * 1024 * 1024;
        const CONSERVATIVE_MODEL_BYTES: u64 = 1_200_000_000;

        assert_eq!(per_slot, 3_151_503_360);
        assert_eq!(per_slot * 16, 50_424_053_760);
        assert!(per_slot + CONSERVATIVE_MODEL_BYTES < L40S_BYTES);
        assert!(per_slot * 16 + CONSERVATIVE_MODEL_BYTES > L40S_BYTES);
    }
}
