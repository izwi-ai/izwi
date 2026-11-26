//! The state a served Inkling 2 sequence keeps in engine-owned storage.
//!
//! Two domains advance together, one step per token:
//!
//! * **attention**: paged keys and values, one physical layer per layer
//!   application (the loop's passes each have their own). Keys are stored
//!   as the model normalised them; the model composes its own attention over
//!   the gathered context, because its scores add a per-query position bias
//!   and a sink logit that the paged kernels do not implement. Windowed
//!   applications share the page table and narrow their own view.
//! * **convolution**: a ring of the last `kernel - 1` inputs of each of the
//!   four short convolutions of every layer application.

use candle_core::{DType, Tensor};

use super::config::{Inkling2Config, CONV_STATES_PER_APPLICATION};
use super::trunk::{Inkling2Context, Inkling2ConvHistory, Inkling2ConvState};
use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, StateComponentValue, TensorStateArena,
};
use crate::backends::BackendKind;
use crate::error::{Error, Result};
use crate::kv::v2::{
    AttentionMask, AttentionPattern, BoundedShape, CheckpointPolicy, InferenceStateContract,
    KeyEncoding, PageSizeConstraint, PagedAttentionDomainSpec, PagedAttentionLayerSpec,
    PlacementPolicy, PrefixPolicy, RingStateDomainSpec, ShapeAxis, ShapeDimension, ShapeExtent,
    StateClock, StateComponentId, StateDType, StateDomainHeader, StateDomainId, StateDomainSpec,
    StateGroupId, StateGroupSpec, StateScope, TensorComponentSpec, TensorRole,
    CURRENT_INFERENCE_STATE_ABI,
};
use crate::models::shared::attention::paged::default_kv_page_size;
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PreparedPhysicalPagedStep};

pub(crate) const INKLING2_ATTENTION_STATE_DOMAIN: StateDomainId = StateDomainId::new(1);
pub(crate) const INKLING2_CONV_STATE_DOMAIN: StateDomainId = StateDomainId::new(2);
const INKLING2_MAIN_STATE_GROUP: StateGroupId = StateGroupId::new(1);

fn state_error(message: impl std::fmt::Display) -> Error {
    Error::InferenceError(format!("Inkling 2 state: {message}"))
}

/// Paged KV storage per backend. Activations stay F32; keys and values are
/// cast on write and on gather. Metal and CUDA halve the bytes.
pub(crate) fn kv_dtype(backend: BackendKind) -> StateDType {
    match backend {
        BackendKind::Cpu => StateDType::F32,
        BackendKind::Metal => StateDType::F16,
        BackendKind::Cuda => StateDType::Bf16,
    }
}

/// The ring component of convolution `slot` (key, value, attention, mlp) of
/// layer application `application`.
pub(crate) fn conv_component(application: usize, slot: usize) -> Result<StateComponentId> {
    u32::try_from(application * CONV_STATES_PER_APPLICATION + slot + 1)
        .map(StateComponentId::new)
        .map_err(|_| state_error("convolution component index exceeds u32"))
}

/// Channels of each convolution of an application, in component order.
fn conv_widths(config: &Inkling2Config, layer: usize) -> [usize; CONV_STATES_PER_APPLICATION] {
    let kv = config.heads(layer).kv_width();
    [kv, kv, config.hidden, config.hidden]
}

fn retained_header(id: StateDomainId) -> StateDomainHeader {
    StateDomainHeader {
        id,
        scope: StateScope::Retained,
        clock: StateClock::DecoderTokens,
        placement: PlacementPolicy::BackendLocalWithHostOffload,
        prefix: PrefixPolicy::Disabled,
        checkpoint: CheckpointPolicy::Transactional,
    }
}

fn to_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::ModelLoadError(format!("Inkling 2 {what} exceeds u32")))
}

/// The scheduler-owned state contract of a chat sequence.
pub(crate) fn managed_contract(
    config: &Inkling2Config,
    kv_dtype: StateDType,
) -> Result<InferenceStateContract> {
    let applications = config.applications();
    let layers = applications
        .iter()
        .enumerate()
        .map(|(index, application)| {
            let heads = config.heads(application.layer);
            Ok(PagedAttentionLayerSpec {
                // One physical layer per application: the loop's passes do
                // not share keys.
                model_layer: to_u32(index, "layer application")?,
                query_heads: to_u32(heads.heads, "query-head count")?,
                kv_heads: to_u32(heads.kv_heads, "KV-head count")?,
                key_head_dim: to_u32(heads.head_dim, "head dimension")?,
                value_head_dim: to_u32(heads.head_dim, "head dimension")?,
                pattern: match heads.window {
                    Some(window) => AttentionPattern::SlidingWindow {
                        window_tokens: to_u32(window, "sliding window")?,
                    },
                    None => AttentionPattern::Full,
                },
                mask: AttentionMask::Causal,
                key_encoding: KeyEncoding::Raw,
                attention_logit_softcap: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let preferred_tokens = to_u32(default_kv_page_size(), "page size")?;
    let attention = StateDomainSpec::PagedAttention(PagedAttentionDomainSpec {
        header: retained_header(INKLING2_ATTENTION_STATE_DOMAIN),
        layers,
        page_size: PageSizeConstraint {
            min_tokens: 1,
            preferred_tokens,
            max_tokens: preferred_tokens.max(256),
            multiple_of: 1,
        },
        accepted_dtypes: vec![kv_dtype],
    });
    let mut components = Vec::with_capacity(applications.len() * CONV_STATES_PER_APPLICATION);
    for (index, application) in applications.iter().enumerate() {
        for (slot, width) in conv_widths(config, application.layer)
            .into_iter()
            .enumerate()
        {
            components.push(TensorComponentSpec {
                id: conv_component(index, slot)?,
                role: TensorRole::ConvolutionState,
                shape: BoundedShape {
                    dimensions: vec![
                        ShapeDimension {
                            axis: ShapeAxis::Batch,
                            extent: ShapeExtent::Fixed { value: 1 },
                        },
                        ShapeDimension {
                            axis: ShapeAxis::Hidden,
                            extent: ShapeExtent::Fixed {
                                value: width as u64,
                            },
                        },
                    ],
                },
                accepted_dtypes: vec![StateDType::F32],
            });
        }
    }
    let convolution = StateDomainSpec::Ring(RingStateDomainSpec {
        header: retained_header(INKLING2_CONV_STATE_DOMAIN),
        components_per_step: components,
        capacity_steps: (config.conv_kernel - 1) as u64,
    });
    let contract = InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: vec![attention, convolution],
        groups: vec![StateGroupSpec {
            id: INKLING2_MAIN_STATE_GROUP,
            domains: vec![INKLING2_ATTENTION_STATE_DOMAIN, INKLING2_CONV_STATE_DOMAIN],
            prefix_shareable: false,
        }],
    };
    contract.validate()?;
    Ok(contract)
}

/// `(model_layer, kv_heads, key_head_dim, value_head_dim)` of every paged
/// layer, for the cache's geometry check.
pub(crate) fn paged_geometry(config: &Inkling2Config) -> Vec<(u32, usize, usize, usize)> {
    config
        .applications()
        .iter()
        .enumerate()
        .map(|(index, application)| {
            let heads = config.heads(application.layer);
            (index as u32, heads.kv_heads, heads.head_dim, heads.head_dim)
        })
        .collect()
}

/// Keys and values in the engine's paged cache, for one prepared append.
pub(crate) struct Inkling2PagedContext<'a> {
    pub(crate) cache: &'a PhysicalPagedKvCache,
    pub(crate) prepared: &'a mut PreparedPhysicalPagedStep,
}

impl Inkling2Context for Inkling2PagedContext<'_> {
    fn append(
        &mut self,
        application: usize,
        keys: &Tensor,
        values: &Tensor,
        window: Option<usize>,
        first: usize,
    ) -> Result<(Tensor, Tensor, usize)> {
        if first != self.cache.context_len() {
            return Err(state_error(format!(
                "a span at {first} does not continue a cache of {} tokens",
                self.cache.context_len()
            )));
        }
        let (keys, values, oldest) =
            self.cache
                .write_and_gather(application, self.prepared, keys, values, window)?;
        Ok((
            keys.to_dtype(DType::F32)?,
            values.to_dtype(DType::F32)?,
            oldest,
        ))
    }
}

/// The convolution histories of a served sequence, mirrored into the
/// engine's tensor-state arena at each committed step.
#[derive(Debug, Clone)]
pub(crate) struct Inkling2ConvRuntimeState {
    cursor: u64,
    /// `None` between a stage and the next restore.
    conv: Option<Inkling2ConvState>,
    /// The state before the first token.
    initial: Inkling2ConvState,
}

impl Inkling2ConvRuntimeState {
    pub(crate) fn new(initial: Inkling2ConvState) -> Self {
        Self {
            cursor: 0,
            conv: Some(initial.clone()),
            initial,
        }
    }

    pub(crate) fn cursor(&self) -> u64 {
        self.cursor
    }

    /// The live histories.
    pub(crate) fn conv(&self) -> Result<&Inkling2ConvState> {
        self.conv
            .as_ref()
            .ok_or_else(|| state_error("the convolution state was staged and not restored"))
    }

    /// Replaces the histories after a span of `steps` tokens.
    pub(crate) fn advance(&mut self, conv: Inkling2ConvState, steps: usize) {
        self.conv = Some(conv);
        self.cursor += steps as u64;
    }

    /// Reads the committed histories back from the arena.
    pub(crate) fn restore(
        &mut self,
        arena: &TensorStateArena,
        sequence: PhysicalStateSequenceId,
    ) -> Result<()> {
        let Some(snapshot) = arena.read(sequence, INKLING2_CONV_STATE_DOMAIN)? else {
            if self.cursor == 0 {
                self.conv = Some(self.initial.clone());
                return Ok(());
            }
            return Err(state_error("the retained convolution state disappeared"));
        };
        let expected = self.initial.histories.len() * CONV_STATES_PER_APPLICATION;
        if snapshot.components.len() != expected {
            return Err(state_error(
                "the retained convolution component count changed",
            ));
        }
        if snapshot.cursor == 0 {
            self.conv = Some(self.initial.clone());
            self.cursor = 0;
            return Ok(());
        }
        let mut histories = Vec::with_capacity(self.initial.histories.len());
        for (application, initial) in self.initial.histories.iter().enumerate() {
            let mut tensors = Vec::with_capacity(CONV_STATES_PER_APPLICATION);
            for (slot, shape) in initial.tensors().into_iter().enumerate() {
                let value = &snapshot.components[application * CONV_STATES_PER_APPLICATION + slot];
                if value.component != conv_component(application, slot)? {
                    return Err(state_error(
                        "a retained convolution component changed identity",
                    ));
                }
                let (steps, width) = shape.dims2()?;
                match value.tensor.as_ref() {
                    Some(tensor)
                        if tensor.dims() == [steps, 1, width] && tensor.dtype() == DType::F32 =>
                    {
                        tensors.push(tensor.reshape((steps, width))?);
                    }
                    _ => {
                        return Err(state_error(
                            "a retained convolution tensor changed geometry",
                        ))
                    }
                }
            }
            let tensors: [Tensor; CONV_STATES_PER_APPLICATION] = tensors
                .try_into()
                .map_err(|_| state_error("a convolution history set is incomplete"))?;
            histories.push(Inkling2ConvHistory::from_tensors(tensors));
        }
        self.conv = Some(Inkling2ConvState { histories });
        self.cursor = snapshot.cursor;
        Ok(())
    }

    /// Stages the histories for commit at `target_cursor` (the KV cursor).
    pub(crate) fn stage(
        &mut self,
        arena: &TensorStateArena,
        transaction: PhysicalStateTransactionId,
        target_cursor: u64,
    ) -> Result<()> {
        if self.cursor != target_cursor {
            return Err(state_error(
                "the convolution cursor does not match the target KV cursor",
            ));
        }
        let expected_cursor = arena
            .read_transaction_base(transaction, INKLING2_CONV_STATE_DOMAIN)?
            .map(|snapshot| snapshot.cursor)
            .unwrap_or(0);
        let conv = self.conv()?;
        let mut values = Vec::with_capacity(conv.histories.len() * CONV_STATES_PER_APPLICATION);
        for (application, history) in conv.histories.iter().enumerate() {
            for (slot, tensor) in history.tensors().into_iter().enumerate() {
                let (steps, width) = tensor.dims2()?;
                values.push(StateComponentValue {
                    component: conv_component(application, slot)?,
                    tensor: Some(tensor.reshape((steps, 1, width))?),
                });
            }
        }
        arena.stage_replace(
            transaction,
            INKLING2_CONV_STATE_DOMAIN,
            expected_cursor,
            target_cursor,
            values,
        )?;
        self.conv = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::tests::{published, tiny};
    use super::*;

    #[test]
    fn the_contract_has_one_paged_layer_and_four_rings_per_application() {
        for config in [tiny(), published()] {
            let contract = managed_contract(&config, StateDType::F32).unwrap();
            let applications = config.application_count();
            let StateDomainSpec::PagedAttention(attention) = &contract.domains[0] else {
                panic!("first domain is not paged attention");
            };
            assert_eq!(attention.layers.len(), applications);
            let windowed = attention
                .layers
                .iter()
                .filter(|layer| matches!(layer.pattern, AttentionPattern::SlidingWindow { .. }))
                .count();
            let StateDomainSpec::Ring(ring) = &contract.domains[1] else {
                panic!("second domain is not a ring");
            };
            assert_eq!(ring.components_per_step.len(), 4 * applications);
            assert_eq!(ring.capacity_steps, 3);
            if config.hidden == 2304 {
                // 48 applications: 40 windowed, 8 full.
                assert_eq!((applications, windowed), (48, 40));
            }
            assert_eq!(paged_geometry(&config).len(), applications);
        }
    }

    #[test]
    fn conv_components_are_distinct_and_one_based() {
        let mut seen = std::collections::BTreeSet::new();
        for application in 0..48 {
            for slot in 0..CONV_STATES_PER_APPLICATION {
                assert!(seen.insert(conv_component(application, slot).unwrap()));
            }
        }
        assert_eq!(seen.len(), 192);
        assert_eq!(conv_component(0, 0).unwrap(), StateComponentId::new(1));
    }
}
