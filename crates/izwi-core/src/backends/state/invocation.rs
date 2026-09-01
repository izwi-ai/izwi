//! Fixed physical backing for invocation-scoped non-paged state.
//!
//! Every component is allocated at its semantic maximum on the selected
//! Candle device. Logical shapes and cursors are metadata over those stable
//! buffers; updates copy into the existing storage and never replace it.

use std::sync::Arc;

use candle_core::{DType, Device, DeviceLocation, Tensor, Var};
use candle_nn::{Conv1d, Conv1dConfig, Module};

use crate::backends::backend_kind_for_device;
use crate::error::{Error, Result};
use crate::kernels::try_lfm_shortconv_ring_sequence;
use crate::kv::v2::{
    ComponentShapeInstantiation, DomainStepIntent, InferenceStateContract, InvocationStateCapacity,
    InvocationWorkspaceDomain, ResolvedNonPagedDomainPlan, ResolvedStatePlan, StateComponentId,
    StateDType, StateDomainId, StateDomainSpec, StateGroupId, StateScope, StateStorageFormat,
    StateUpdateKind, TensorComponentSpec,
};

use super::StateBackendRegistry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InvocationTensorDomainKind {
    StaticTensor,
    Tensor,
    Append,
    Ring,
}

#[derive(Debug, Clone)]
pub(crate) struct InvocationTensorComponentValue {
    pub(crate) component: StateComponentId,
    pub(crate) tensor: Tensor,
}

#[derive(Debug, Clone)]
pub(crate) struct InvocationTensorComponentSlice {
    pub(crate) component: StateComponentId,
    pub(crate) tensor: Tensor,
}

#[derive(Debug, Clone)]
pub(crate) struct InvocationTensorSnapshot {
    pub(crate) source_identity: Option<[u8; 32]>,
    pub(crate) absolute_cursor: u64,
    pub(crate) valid_length: u64,
    pub(crate) components: Vec<InvocationTensorComponentSlice>,
}

#[derive(Debug, Clone)]
pub(crate) struct InvocationTensorStepValues {
    pub(crate) components: Vec<InvocationTensorComponentValue>,
}

/// One component of a uniform multi-step sequence update.
///
/// The leading tensor dimension is the authenticated step count. Remaining
/// dimensions are the component's semantic per-step shape. Ring arenas retain
/// only the physical tail that fits their fixed capacity.
#[derive(Debug, Clone)]
pub(crate) struct InvocationTensorBulkComponentValue {
    pub(crate) component: StateComponentId,
    pub(crate) tensor: Tensor,
}

#[derive(Debug, Clone)]
pub(crate) enum InvocationTensorUpdateV2 {
    StaticInitialize {
        source_identity: [u8; 32],
        components: Vec<InvocationTensorComponentValue>,
    },
    TensorReplace {
        components: Vec<InvocationTensorComponentValue>,
    },
    Append {
        steps: Vec<InvocationTensorStepValues>,
    },
    RingAdvance {
        steps: Vec<InvocationTensorStepValues>,
    },
    RingAdvanceBulk {
        components: Vec<InvocationTensorBulkComponentValue>,
    },
    Reset,
    NoOp,
}

#[derive(Debug, Clone)]
pub(crate) struct InvocationTensorChronologicalSegment {
    pub(crate) absolute_start: u64,
    pub(crate) length: u64,
    pub(crate) components: Vec<InvocationTensorComponentSlice>,
}

#[derive(Debug)]
struct ComponentBacking {
    semantic: TensorComponentSpec,
    storage: Var,
    maximum_shape: Vec<usize>,
    logical_shapes: Vec<Option<Vec<usize>>>,
}

#[derive(Debug)]
struct ChronologicalRun {
    absolute_start: u64,
    physical_start: usize,
    length: usize,
    logical_shapes: Vec<Vec<usize>>,
}

/// One invocation-exclusive, load-time allocated physical state domain.
///
/// Pooling and lease generations live above this type. The arena itself owns
/// one stable set of device buffers and can be reset before reuse.
#[derive(Debug)]
pub(crate) struct InvocationTensorArena {
    plan: Arc<ResolvedStatePlan>,
    workspace_domain: InvocationWorkspaceDomain,
    domain: StateDomainId,
    group: StateGroupId,
    kind: InvocationTensorDomainKind,
    device: Device,
    components: Vec<ComponentBacking>,
    maximum_bytes: u64,
    capacity_steps: usize,
    source_identity: Option<[u8; 32]>,
    absolute_cursor: u64,
    valid_length: usize,
    initialized: bool,
    dirty: bool,
}

/// A sealed all-component ShortConv step over one physical ring domain.
///
/// Model code can consume the convolution result but never receives a tensor
/// alias to arena backing. Every component is staged from model activations
/// and the physical ring advances exactly once, after all layers succeed.
pub(crate) struct InvocationRingDepthwiseConvTransaction<'a> {
    arena: &'a mut InvocationTensorArena,
    intent: DomainStepIntent,
    declared_steps: u64,
    declared: Vec<ComponentShapeInstantiation>,
    updates: Vec<Option<InvocationTensorBulkComponentValue>>,
}

impl InvocationTensorArena {
    pub(crate) fn new(
        contract: &InferenceStateContract,
        plan: Arc<ResolvedStatePlan>,
        workspace_domain: InvocationWorkspaceDomain,
        device: Device,
    ) -> Result<Self> {
        let backend = backend_kind_for_device(&device);
        let ordinal = device_ordinal(&device)?;
        if plan.backend != backend || plan.device_ordinal != ordinal {
            return Err(invalid(
                "invocation tensor arena plan does not match its Candle device",
            ));
        }
        let InvocationWorkspaceDomain::State {
            state,
            capacity: InvocationStateCapacity::SemanticBounded,
            placement,
            formula,
        } = &workspace_domain
        else {
            return Err(invalid(
                "invocation tensor arena requires semantic-bounded typed state",
            ));
        };
        let canonical_state = contract
            .domains
            .iter()
            .find(|domain| domain.id() == state.id())
            .ok_or_else(|| {
                invalid("invocation tensor workspace domain is absent from its contract")
            })?;
        if canonical_state != state {
            return Err(invalid(
                "invocation tensor workspace state is not the canonical contract member",
            ));
        }
        if state.scope() != StateScope::Invocation || state.header().placement != *placement {
            return Err(invalid(
                "invocation tensor arena requires invocation-scoped matching placement",
            ));
        }

        let matches = plan
            .non_paged
            .iter()
            .filter(|resolved| resolved.domain() == state.id())
            .collect::<Vec<_>>();
        let [resolved] = matches.as_slice() else {
            return Err(invalid(
                "invocation tensor arena requires one exact resolved non-paged domain",
            ));
        };
        if matches!(resolved, ResolvedNonPagedDomainPlan::StaticAttention(_))
            || matches!(state, StateDomainSpec::StaticAttention(_))
        {
            return Err(invalid(
                "static attention requires a direct install-and-attend arena",
            ));
        }
        let registry = StateBackendRegistry::new(backend, ordinal)?;
        plan.validate_against(contract, &registry)?;
        if formula.maximum_bytes()? < resolved.maximum_bytes() {
            return Err(invalid(
                "invocation tensor workspace formula is smaller than its resolved backing",
            ));
        }

        let (kind, semantic_components, resolved_components, capacity_steps) =
            domain_components(state, resolved)?;
        let mut components = Vec::with_capacity(semantic_components.len());
        let mut allocated_bytes = 0_u64;
        for (semantic, resolved) in semantic_components.iter().zip(resolved_components) {
            if semantic.id != resolved.component {
                return Err(invalid(
                    "invocation tensor component identity changed during resolution",
                ));
            }
            let maximum_shape = maximum_shape(&semantic.shape.dimensions)?;
            let mut physical_shape =
                Vec::with_capacity(maximum_shape.len() + usize::from(capacity_steps > 1));
            if matches!(
                kind,
                InvocationTensorDomainKind::Append | InvocationTensorDomainKind::Ring
            ) {
                physical_shape.push(capacity_steps);
            }
            physical_shape.extend_from_slice(&maximum_shape);
            let dtype = candle_dtype(resolved.storage)?;
            let storage = Var::zeros(physical_shape.as_slice(), dtype, &device)?;
            let bytes = tensor_bytes(storage.as_tensor())?;
            let expected = resolved.maximum_bytes.checked_mul(
                if matches!(
                    kind,
                    InvocationTensorDomainKind::Append | InvocationTensorDomainKind::Ring
                ) {
                    u64::try_from(capacity_steps)
                        .map_err(|_| invalid("invocation tensor capacity exceeds u64"))?
                } else {
                    1
                },
            );
            if Some(bytes) != expected {
                return Err(invalid(
                    "invocation tensor allocation does not match its resolved byte bound",
                ));
            }
            allocated_bytes = allocated_bytes
                .checked_add(bytes)
                .ok_or_else(|| invalid("invocation tensor allocation byte bound overflow"))?;
            components.push(ComponentBacking {
                semantic: semantic.clone(),
                storage,
                maximum_shape,
                logical_shapes: vec![None; capacity_steps],
            });
        }
        if allocated_bytes != resolved.maximum_bytes() {
            return Err(invalid(
                "invocation tensor arena bytes do not equal the resolved domain capacity",
            ));
        }
        device.synchronize()?;

        let domain = state.id();
        let group = resolved.group();
        Ok(Self {
            plan,
            workspace_domain,
            domain,
            group,
            kind,
            device,
            components,
            maximum_bytes: allocated_bytes,
            capacity_steps,
            source_identity: None,
            absolute_cursor: 0,
            valid_length: 0,
            initialized: false,
            dirty: false,
        })
    }

    pub(crate) fn plan(&self) -> &ResolvedStatePlan {
        &self.plan
    }

    pub(crate) fn workspace_domain(&self) -> &InvocationWorkspaceDomain {
        &self.workspace_domain
    }

    pub(crate) const fn domain(&self) -> StateDomainId {
        self.domain
    }

    pub(crate) const fn group(&self) -> StateGroupId {
        self.group
    }

    pub(crate) const fn kind(&self) -> InvocationTensorDomainKind {
        self.kind
    }

    pub(crate) const fn absolute_cursor(&self) -> u64 {
        self.absolute_cursor
    }

    pub(crate) const fn valid_length(&self) -> usize {
        self.valid_length
    }

    pub(crate) const fn capacity_steps(&self) -> usize {
        self.capacity_steps
    }

    pub(crate) const fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub(crate) const fn maximum_bytes(&self) -> u64 {
        self.maximum_bytes
    }

    pub(crate) fn apply_intent(
        &mut self,
        intent: &DomainStepIntent,
        update: InvocationTensorUpdateV2,
    ) -> Result<()> {
        self.require_clean()?;
        if intent.domain != self.domain || intent.expected_cursor != self.absolute_cursor {
            return Err(invalid(
                "invocation tensor intent has a foreign domain or stale cursor",
            ));
        }
        match (&intent.update, update) {
            (
                StateUpdateKind::StaticInitialize {
                    source_identity,
                    components: declared,
                },
                InvocationTensorUpdateV2::StaticInitialize {
                    source_identity: actual_source,
                    components,
                },
            ) if self.kind == InvocationTensorDomainKind::StaticTensor => {
                if *source_identity != actual_source {
                    return Err(invalid(
                        "invocation static-tensor source identity does not match its intent",
                    ));
                }
                self.validate_declared_values(declared, &components)?;
                self.install(actual_source, intent.target_cursor, &components)
            }
            (
                StateUpdateKind::TensorReplace {
                    components: declared,
                },
                InvocationTensorUpdateV2::TensorReplace { components },
            ) if self.kind == InvocationTensorDomainKind::Tensor => {
                self.validate_declared_values(declared, &components)?;
                self.replace(intent.expected_cursor, intent.target_cursor, &components)
            }
            (
                StateUpdateKind::Append {
                    steps: declared_steps,
                    components_per_step: declared,
                },
                InvocationTensorUpdateV2::Append { steps },
            ) if self.kind == InvocationTensorDomainKind::Append => {
                self.validate_declared_steps(*declared_steps, declared, &steps)?;
                self.append(intent.expected_cursor, intent.target_cursor, &steps)
            }
            (
                StateUpdateKind::RingAdvance {
                    steps: declared_steps,
                    components_per_step: declared,
                },
                InvocationTensorUpdateV2::RingAdvance { steps },
            ) if self.kind == InvocationTensorDomainKind::Ring => {
                self.validate_declared_steps(*declared_steps, declared, &steps)?;
                self.ring_advance(intent.expected_cursor, intent.target_cursor, &steps)
            }
            (
                StateUpdateKind::RingAdvance {
                    steps: declared_steps,
                    components_per_step: declared,
                },
                InvocationTensorUpdateV2::RingAdvanceBulk { components },
            ) if self.kind == InvocationTensorDomainKind::Ring => {
                self.validate_declared_bulk(*declared_steps, declared, &components)?;
                self.ring_advance_bulk(
                    intent.expected_cursor,
                    intent.target_cursor,
                    *declared_steps,
                    &components,
                )
            }
            (StateUpdateKind::Reset, InvocationTensorUpdateV2::Reset)
                if matches!(
                    self.kind,
                    InvocationTensorDomainKind::Tensor
                        | InvocationTensorDomainKind::Append
                        | InvocationTensorDomainKind::Ring
                ) && intent.target_cursor == 0 =>
            {
                self.reset()
            }
            (StateUpdateKind::NoOp, InvocationTensorUpdateV2::NoOp)
                if intent.target_cursor == intent.expected_cursor =>
            {
                Ok(())
            }
            _ => Err(invalid(
                "invocation tensor update does not exactly match its authenticated intent",
            )),
        }
    }

    pub(crate) fn begin_ring_depthwise_conv(
        &mut self,
        intent: &DomainStepIntent,
    ) -> Result<InvocationRingDepthwiseConvTransaction<'_>> {
        self.require_clean()?;
        if self.kind != InvocationTensorDomainKind::Ring {
            return Err(invalid(
                "physical depthwise convolution requires ring state",
            ));
        }
        if intent.domain != self.domain || intent.expected_cursor != self.absolute_cursor {
            return Err(invalid(
                "physical depthwise convolution has a foreign domain or stale cursor",
            ));
        }
        let StateUpdateKind::RingAdvance {
            steps,
            components_per_step,
        } = &intent.update
        else {
            return Err(invalid(
                "physical depthwise convolution requires a ring-advance intent",
            ));
        };
        self.validate_cursor_advance_u64(intent.expected_cursor, intent.target_cursor, *steps)?;
        if components_per_step.len() != self.components.len() {
            return Err(invalid(
                "physical depthwise convolution must declare every ring component",
            ));
        }
        for (declared, component) in components_per_step.iter().zip(&self.components) {
            if declared.component != component.semantic.id
                || declared.dimensions.len() != component.semantic.shape.dimensions.len()
            {
                return Err(invalid(
                    "physical depthwise convolution component identity or rank mismatch",
                ));
            }
            for (declared_dimension, semantic_dimension) in declared
                .dimensions
                .iter()
                .zip(&component.semantic.shape.dimensions)
            {
                if declared_dimension.axis != semantic_dimension.axis
                    || !semantic_dimension.extent.accepts(declared_dimension.units)
                {
                    return Err(invalid(
                        "physical depthwise convolution component shape is outside its contract",
                    ));
                }
            }
        }
        Ok(InvocationRingDepthwiseConvTransaction {
            arena: self,
            intent: intent.clone(),
            declared_steps: *steps,
            declared: components_per_step.clone(),
            updates: vec![None; components_per_step.len()],
        })
    }

    pub(crate) fn read_snapshot(&self) -> Result<InvocationTensorSnapshot> {
        self.snapshot()
    }

    pub(crate) fn read_chronological_segments(
        &self,
    ) -> Result<Vec<InvocationTensorChronologicalSegment>> {
        self.chronological_segments()
    }

    pub(crate) fn reset_for_reuse(&mut self) -> Result<()> {
        self.reset()
    }

    /// Fence every write issued through this exact arena before the pool
    /// publishes an authenticated completion receipt.
    pub(crate) fn prepare_completion(&mut self) -> Result<()> {
        self.require_clean()?;
        if let Err(error) = self.device.synchronize() {
            self.dirty = true;
            return Err(error.into());
        }
        Ok(())
    }

    fn backing(&self, component: StateComponentId) -> Result<Tensor> {
        self.require_clean()?;
        Ok(self.component(component)?.storage.as_tensor().clone())
    }

    fn install(
        &mut self,
        source_identity: [u8; 32],
        target_cursor: u64,
        values: &[InvocationTensorComponentValue],
    ) -> Result<()> {
        self.require_clean()?;
        if self.kind != InvocationTensorDomainKind::StaticTensor {
            return Err(invalid(
                "install is only available for static-tensor invocation domains",
            ));
        }
        if self.initialized {
            return Err(invalid("invocation tensor domain is already initialized"));
        }
        if target_cursor == 0 || source_identity.iter().all(|byte| *byte == 0) {
            return Err(invalid(
                "invocation static-tensor installation requires non-zero source and target identities",
            ));
        }
        self.validate_values(values)?;
        self.dirty = true;
        self.write_tensor_values(values)?;
        self.device.synchronize()?;
        self.publish_tensor_shapes(values);
        self.source_identity = Some(source_identity);
        self.absolute_cursor = target_cursor;
        self.valid_length = 1;
        self.initialized = true;
        self.dirty = false;
        Ok(())
    }

    fn replace(
        &mut self,
        expected_cursor: u64,
        target_cursor: u64,
        values: &[InvocationTensorComponentValue],
    ) -> Result<()> {
        self.require_clean()?;
        if self.kind != InvocationTensorDomainKind::Tensor {
            return Err(invalid(
                "replace is only available for mutable tensor invocation domains",
            ));
        }
        if expected_cursor != self.absolute_cursor || target_cursor <= expected_cursor {
            return Err(invalid(
                "invocation tensor replacement has a stale or non-increasing cursor",
            ));
        }
        self.validate_values(values)?;
        self.dirty = true;
        self.write_tensor_values(values)?;
        self.device.synchronize()?;
        self.publish_tensor_shapes(values);
        self.source_identity = None;
        self.absolute_cursor = target_cursor;
        self.valid_length = 1;
        self.initialized = true;
        self.dirty = false;
        Ok(())
    }

    fn write_tensor_values(&self, values: &[InvocationTensorComponentValue]) -> Result<()> {
        for (component, value) in self.components.iter().zip(values) {
            write_logical_value(component, &value.tensor, None)?;
        }
        Ok(())
    }

    fn snapshot(&self) -> Result<InvocationTensorSnapshot> {
        self.require_clean()?;
        if !self.initialized {
            return Err(invalid("invocation tensor domain is not initialized"));
        }
        match self.kind {
            InvocationTensorDomainKind::StaticTensor | InvocationTensorDomainKind::Tensor => {
                Ok(InvocationTensorSnapshot {
                    source_identity: self.source_identity,
                    absolute_cursor: self.absolute_cursor,
                    valid_length: 1,
                    components: self
                        .components
                        .iter()
                        .map(|component| {
                            Ok(InvocationTensorComponentSlice {
                                component: component.semantic.id,
                                tensor: logical_view(
                                    component.storage.as_tensor(),
                                    component.logical_shapes[0]
                                        .as_deref()
                                        .expect("initialized tensor has a logical shape"),
                                    0,
                                )?
                                .copy()?,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?,
                })
            }
            InvocationTensorDomainKind::Append | InvocationTensorDomainKind::Ring => Err(invalid(
                "sequence state requires chronological segments instead of one snapshot",
            )),
        }
    }

    fn append(
        &mut self,
        expected_cursor: u64,
        target_cursor: u64,
        steps: &[InvocationTensorStepValues],
    ) -> Result<()> {
        self.require_clean()?;
        if self.kind != InvocationTensorDomainKind::Append {
            return Err(invalid("append requires an append invocation domain"));
        }
        self.validate_cursor_advance(expected_cursor, target_cursor, steps.len())?;
        let target = usize::try_from(target_cursor)
            .map_err(|_| invalid("append invocation target cursor exceeds usize"))?;
        if target > self.capacity_steps {
            return Err(invalid("append invocation capacity is exhausted"));
        }
        let start = usize::try_from(expected_cursor)
            .map_err(|_| invalid("append invocation cursor exceeds usize"))?;
        start
            .checked_add(steps.len())
            .ok_or_else(|| invalid("append physical index overflow"))?;
        self.validate_steps(steps)?;
        self.dirty = true;
        for (offset, step) in steps.iter().enumerate() {
            self.write_step_values(start + offset, step)?;
        }
        self.device.synchronize()?;
        for (offset, step) in steps.iter().enumerate() {
            self.publish_step_shapes(start + offset, step);
        }
        self.absolute_cursor = target_cursor;
        self.valid_length = target;
        self.initialized = true;
        self.dirty = false;
        Ok(())
    }

    fn ring_advance(
        &mut self,
        expected_cursor: u64,
        target_cursor: u64,
        steps: &[InvocationTensorStepValues],
    ) -> Result<()> {
        self.require_clean()?;
        if self.kind != InvocationTensorDomainKind::Ring {
            return Err(invalid("ring advance requires a ring invocation domain"));
        }
        self.validate_cursor_advance(expected_cursor, target_cursor, steps.len())?;
        let capacity = u64::try_from(self.capacity_steps)
            .map_err(|_| invalid("ring invocation capacity exceeds u64"))?;
        let retained = steps.len().min(self.capacity_steps);
        let first_retained_step = steps.len() - retained;
        let retained =
            u64::try_from(retained).map_err(|_| invalid("ring retained step count exceeds u64"))?;
        let first_retained_cursor = target_cursor
            .checked_sub(retained)
            .ok_or_else(|| invalid("ring retained cursor underflow"))?;
        let new_valid_length = self
            .valid_length
            .checked_add(steps.len())
            .unwrap_or(self.capacity_steps)
            .min(self.capacity_steps);
        self.validate_steps(steps)?;
        self.dirty = true;
        for (offset, step) in steps[first_retained_step..].iter().enumerate() {
            let absolute = first_retained_cursor
                + u64::try_from(offset)
                    .expect("retained ring offset was already bounded by capacity");
            let physical = usize::try_from(absolute % capacity)
                .expect("ring modulo capacity always fits usize");
            self.write_step_values(physical, step)?;
        }
        self.device.synchronize()?;
        for (offset, step) in steps[first_retained_step..].iter().enumerate() {
            let absolute = first_retained_cursor
                + u64::try_from(offset)
                    .expect("retained ring offset was already bounded by capacity");
            let physical = usize::try_from(absolute % capacity)
                .expect("ring modulo capacity always fits usize");
            self.publish_step_shapes(physical, step);
        }
        self.absolute_cursor = target_cursor;
        self.valid_length = new_valid_length;
        self.initialized = true;
        self.dirty = false;
        Ok(())
    }

    fn ring_advance_bulk(
        &mut self,
        expected_cursor: u64,
        target_cursor: u64,
        declared_steps: u64,
        values: &[InvocationTensorBulkComponentValue],
    ) -> Result<()> {
        self.require_clean()?;
        if self.kind != InvocationTensorDomainKind::Ring {
            return Err(invalid(
                "bulk ring advance requires a ring invocation domain",
            ));
        }
        self.validate_cursor_advance_u64(expected_cursor, target_cursor, declared_steps)?;
        let capacity = u64::try_from(self.capacity_steps)
            .map_err(|_| invalid("ring invocation capacity exceeds u64"))?;
        let retained = declared_steps.min(capacity);
        let first_retained_source = declared_steps
            .checked_sub(retained)
            .ok_or_else(|| invalid("bulk ring retained source underflow"))?;
        let first_retained_cursor = target_cursor
            .checked_sub(retained)
            .ok_or_else(|| invalid("bulk ring retained cursor underflow"))?;
        let retained_usize = usize::try_from(retained)
            .map_err(|_| invalid("bulk ring retained length exceeds usize"))?;
        let new_valid_length = self
            .valid_length
            .checked_add(usize::try_from(declared_steps).unwrap_or(self.capacity_steps))
            .unwrap_or(self.capacity_steps)
            .min(self.capacity_steps);

        self.dirty = true;
        for offset in 0..retained_usize {
            let offset_u64 = u64::try_from(offset).expect("retained bulk ring offset fits u64");
            let source_step = usize::try_from(first_retained_source + offset_u64)
                .map_err(|_| invalid("bulk ring source index exceeds usize"))?;
            let absolute = first_retained_cursor + offset_u64;
            let physical = usize::try_from(absolute % capacity)
                .expect("ring modulo capacity always fits usize");
            for (component, value) in self.components.iter().zip(values) {
                let source = value.tensor.get(source_step)?;
                write_logical_value(component, &source, Some(physical))?;
            }
        }
        self.device.synchronize()?;
        let logical_shapes = values
            .iter()
            .map(|value| value.tensor.dims()[1..].to_vec())
            .collect::<Vec<_>>();
        for offset in 0..retained_usize {
            let offset_u64 = u64::try_from(offset).expect("retained bulk ring offset fits u64");
            let absolute = first_retained_cursor + offset_u64;
            let physical = usize::try_from(absolute % capacity)
                .expect("ring modulo capacity always fits usize");
            for (component, logical_shape) in self.components.iter_mut().zip(&logical_shapes) {
                component.logical_shapes[physical] = Some(logical_shape.clone());
            }
        }
        self.absolute_cursor = target_cursor;
        self.valid_length = new_valid_length;
        self.initialized = true;
        self.dirty = false;
        Ok(())
    }

    fn chronological_segments(&self) -> Result<Vec<InvocationTensorChronologicalSegment>> {
        self.require_clean()?;
        if !matches!(
            self.kind,
            InvocationTensorDomainKind::Append | InvocationTensorDomainKind::Ring
        ) {
            return Err(invalid(
                "chronological segments require an append or ring invocation domain",
            ));
        }
        if !self.initialized {
            return Err(invalid("sequence invocation domain is not initialized"));
        }
        let valid = u64::try_from(self.valid_length)
            .map_err(|_| invalid("sequence invocation valid length exceeds u64"))?;
        let oldest = self
            .absolute_cursor
            .checked_sub(valid)
            .ok_or_else(|| invalid("sequence cursor is smaller than its valid length"))?;
        self.chronological_runs(oldest)?
            .into_iter()
            .map(|run| {
                Ok(InvocationTensorChronologicalSegment {
                    absolute_start: run.absolute_start,
                    length: u64::try_from(run.length)
                        .map_err(|_| invalid("chronological segment length exceeds u64"))?,
                    components: self
                        .components
                        .iter()
                        .zip(&run.logical_shapes)
                        .map(|(component, logical_shape)| {
                            Ok(InvocationTensorComponentSlice {
                                component: component.semantic.id,
                                tensor: logical_step_view(
                                    component,
                                    run.physical_start,
                                    run.length,
                                    logical_shape,
                                )?
                                .copy()?,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?,
                })
            })
            .collect()
    }

    fn ring_component_step_view(
        &self,
        component_id: StateComponentId,
        absolute: u64,
        logical_shape: &[usize],
    ) -> Result<Tensor> {
        self.require_clean()?;
        if self.kind != InvocationTensorDomainKind::Ring {
            return Err(invalid(
                "physical depthwise convolution view requires ring state",
            ));
        }
        if !self.initialized || self.valid_length == 0 {
            return Err(invalid(
                "physical depthwise convolution ring is not initialized",
            ));
        }
        let component_index = self
            .components
            .iter()
            .position(|component| component.semantic.id == component_id)
            .ok_or_else(|| invalid("physical depthwise convolution component is absent"))?;
        let valid = u64::try_from(self.valid_length)
            .map_err(|_| invalid("physical depthwise convolution history exceeds u64"))?;
        let oldest = self
            .absolute_cursor
            .checked_sub(valid)
            .ok_or_else(|| invalid("physical depthwise convolution history underflow"))?;
        if absolute < oldest || absolute >= self.absolute_cursor {
            return Err(invalid(
                "physical depthwise convolution view is outside retained history",
            ));
        }
        let capacity = u64::try_from(self.capacity_steps)
            .map_err(|_| invalid("physical depthwise convolution capacity exceeds u64"))?;
        let physical = usize::try_from(absolute % capacity)
            .map_err(|_| invalid("physical depthwise convolution slot exceeds usize"))?;
        let stored_shape = self.components[component_index].logical_shapes[physical]
            .as_deref()
            .ok_or_else(|| invalid("physical depthwise convolution slot shape is absent"))?;
        if stored_shape != logical_shape {
            return Err(invalid(
                "physical depthwise convolution history shape changed across steps",
            ));
        }
        logical_step_view(&self.components[component_index], physical, 1, stored_shape)?
            .get(0)
            .map_err(Error::from)
    }

    fn reset(&mut self) -> Result<()> {
        for component in &mut self.components {
            if let Err(error) = component.storage.zero_set() {
                self.dirty = true;
                return Err(error.into());
            }
        }
        if let Err(error) = self.device.synchronize() {
            self.dirty = true;
            return Err(error.into());
        }
        for component in &mut self.components {
            component.logical_shapes.fill(None);
        }
        self.source_identity = None;
        self.absolute_cursor = 0;
        self.valid_length = 0;
        self.initialized = false;
        self.dirty = false;
        Ok(())
    }

    fn require_clean(&self) -> Result<()> {
        if self.dirty {
            return Err(invalid(
                "invocation tensor arena is dirty and must be reset before reuse",
            ));
        }
        Ok(())
    }

    fn validate_cursor_advance(
        &self,
        expected_cursor: u64,
        target_cursor: u64,
        step_count: usize,
    ) -> Result<()> {
        let step_count = u64::try_from(step_count)
            .map_err(|_| invalid("sequence invocation step count exceeds u64"))?;
        self.validate_cursor_advance_u64(expected_cursor, target_cursor, step_count)
    }

    fn validate_cursor_advance_u64(
        &self,
        expected_cursor: u64,
        target_cursor: u64,
        step_count: u64,
    ) -> Result<()> {
        if expected_cursor != self.absolute_cursor {
            return Err(invalid("sequence invocation cursor is stale"));
        }
        if step_count == 0 || target_cursor <= expected_cursor {
            return Err(invalid(
                "sequence invocation update requires a non-empty cursor advance",
            ));
        }
        let required_target = expected_cursor
            .checked_add(step_count)
            .ok_or_else(|| invalid("sequence invocation cursor overflow"))?;
        if target_cursor != required_target {
            return Err(invalid(
                "sequence invocation target cursor must equal expected cursor plus steps",
            ));
        }
        Ok(())
    }

    fn chronological_runs(&self, oldest: u64) -> Result<Vec<ChronologicalRun>> {
        let capacity = u64::try_from(self.capacity_steps)
            .map_err(|_| invalid("sequence invocation capacity exceeds u64"))?;
        let mut runs = Vec::<ChronologicalRun>::new();
        for offset in 0..self.valid_length {
            let absolute = oldest
                .checked_add(
                    u64::try_from(offset)
                        .map_err(|_| invalid("chronological offset exceeds u64"))?,
                )
                .ok_or_else(|| invalid("chronological cursor overflow"))?;
            let physical = match self.kind {
                InvocationTensorDomainKind::Append => usize::try_from(absolute)
                    .map_err(|_| invalid("append chronological index exceeds usize"))?,
                InvocationTensorDomainKind::Ring => usize::try_from(absolute % capacity)
                    .map_err(|_| invalid("ring chronological index exceeds usize"))?,
                _ => unreachable!("chronological runs require a sequence domain"),
            };
            let logical_shapes = self
                .components
                .iter()
                .map(|component| {
                    component.logical_shapes[physical]
                        .clone()
                        .ok_or_else(|| invalid("chronological physical slot has no logical shape"))
                })
                .collect::<Result<Vec<_>>>()?;
            if let Some(run) = runs.last_mut() {
                if run.physical_start + run.length == physical
                    && run.logical_shapes == logical_shapes
                {
                    run.length += 1;
                    continue;
                }
            }
            runs.push(ChronologicalRun {
                absolute_start: absolute,
                physical_start: physical,
                length: 1,
                logical_shapes,
            });
        }
        Ok(runs)
    }

    fn component(&self, id: StateComponentId) -> Result<&ComponentBacking> {
        self.components
            .iter()
            .find(|component| component.semantic.id == id)
            .ok_or_else(|| invalid("invocation tensor arena has no such component"))
    }

    fn validate_values(&self, values: &[InvocationTensorComponentValue]) -> Result<()> {
        if values.len() != self.components.len() {
            return Err(invalid(
                "invocation tensor update must cover every component",
            ));
        }
        for (component, value) in self.components.iter().zip(values) {
            if value.component != component.semantic.id {
                return Err(invalid(
                    "invocation tensor components are not in canonical identity order",
                ));
            }
            validate_input_tensor(&value.tensor, component, &self.device)?;
            if !value.tensor.is_contiguous() {
                return Err(invalid(
                    "invocation tensor updates require contiguous source tensors",
                ));
            }
            if self
                .components
                .iter()
                .any(|backing| shares_storage(&value.tensor, backing.storage.as_tensor()))
            {
                return Err(invalid(
                    "invocation tensor update source aliases arena storage",
                ));
            }
        }
        Ok(())
    }

    fn validate_steps(&self, steps: &[InvocationTensorStepValues]) -> Result<()> {
        for step in steps {
            self.validate_values(&step.components)?;
        }
        Ok(())
    }

    fn validate_declared_values(
        &self,
        declared: &[ComponentShapeInstantiation],
        values: &[InvocationTensorComponentValue],
    ) -> Result<()> {
        self.validate_values(values)?;
        if declared.len() != self.components.len() {
            return Err(invalid(
                "invocation tensor intent must declare every component exactly once",
            ));
        }
        for ((declared, component), value) in declared.iter().zip(&self.components).zip(values) {
            if declared.component != component.semantic.id
                || value.component != component.semantic.id
                || declared.dimensions.len() != component.semantic.shape.dimensions.len()
                || value.tensor.rank() != declared.dimensions.len()
            {
                return Err(invalid(
                    "invocation tensor intent component identity or rank mismatch",
                ));
            }
            for ((declared, semantic), actual) in declared
                .dimensions
                .iter()
                .zip(&component.semantic.shape.dimensions)
                .zip(value.tensor.dims())
            {
                let actual = u64::try_from(*actual)
                    .map_err(|_| invalid("invocation tensor dimension exceeds u64"))?;
                if declared.axis != semantic.axis
                    || declared.units != actual
                    || !semantic.extent.accepts(declared.units)
                {
                    return Err(invalid(
                        "invocation tensor intent axis or extent does not match its value",
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_declared_steps(
        &self,
        declared_steps: u64,
        declared: &[ComponentShapeInstantiation],
        steps: &[InvocationTensorStepValues],
    ) -> Result<()> {
        if u64::try_from(steps.len())
            .map_err(|_| invalid("invocation tensor update step count exceeds u64"))?
            != declared_steps
        {
            return Err(invalid(
                "invocation tensor update step count does not match its intent",
            ));
        }
        for step in steps {
            self.validate_declared_values(declared, &step.components)?;
        }
        Ok(())
    }

    fn validate_declared_bulk(
        &self,
        declared_steps: u64,
        declared: &[ComponentShapeInstantiation],
        values: &[InvocationTensorBulkComponentValue],
    ) -> Result<()> {
        if declared_steps == 0
            || declared.len() != self.components.len()
            || values.len() != self.components.len()
        {
            return Err(invalid(
                "bulk ring update must declare every component and at least one step",
            ));
        }
        for ((declared, component), value) in declared.iter().zip(&self.components).zip(values) {
            if declared.component != component.semantic.id
                || value.component != component.semantic.id
                || declared.dimensions.len() != component.semantic.shape.dimensions.len()
                || value.tensor.rank() != declared.dimensions.len() + 1
                || u64::try_from(value.tensor.dims()[0])
                    .map_err(|_| invalid("bulk ring step dimension exceeds u64"))?
                    != declared_steps
                || value.tensor.device().location() != self.device.location()
                || value.tensor.dtype() != component.storage.dtype()
                || !value.tensor.is_contiguous()
                || self
                    .components
                    .iter()
                    .any(|backing| shares_storage(&value.tensor, backing.storage.as_tensor()))
            {
                return Err(invalid(
                    "bulk ring component has incompatible identity, steps, storage, or layout",
                ));
            }
            for ((declared, semantic), actual) in declared
                .dimensions
                .iter()
                .zip(&component.semantic.shape.dimensions)
                .zip(&value.tensor.dims()[1..])
            {
                let actual = u64::try_from(*actual)
                    .map_err(|_| invalid("bulk ring dimension exceeds u64"))?;
                if declared.axis != semantic.axis
                    || declared.units != actual
                    || !semantic.extent.accepts(actual)
                {
                    return Err(invalid(
                        "bulk ring component shape does not match its authenticated intent",
                    ));
                }
            }
        }
        Ok(())
    }

    fn write_step_values(&self, physical: usize, step: &InvocationTensorStepValues) -> Result<()> {
        for (component, value) in self.components.iter().zip(&step.components) {
            write_logical_value(component, &value.tensor, Some(physical))?;
        }
        Ok(())
    }

    fn publish_tensor_shapes(&mut self, values: &[InvocationTensorComponentValue]) {
        for (component, value) in self.components.iter_mut().zip(values) {
            component.logical_shapes[0] = Some(value.tensor.dims().to_vec());
        }
    }

    fn publish_step_shapes(&mut self, physical: usize, step: &InvocationTensorStepValues) {
        for (component, value) in self.components.iter_mut().zip(&step.components) {
            component.logical_shapes[physical] = Some(value.tensor.dims().to_vec());
        }
    }
}

impl InvocationRingDepthwiseConvTransaction<'_> {
    /// Apply one causal depthwise convolution component.
    ///
    /// `input` is `[batch, hidden, steps]`, `weight` is
    /// `[hidden, ring_capacity]`, and the returned tensor preserves the input
    /// shape. The model input is retained only as the pending ring update;
    /// retained history is consumed through backing-storage views while the
    /// transaction holds the arena lock, and no arena-backed tensor is returned.
    pub(crate) fn apply(
        &mut self,
        component_id: StateComponentId,
        input: &Tensor,
        weight: &Tensor,
    ) -> Result<Tensor> {
        let component_index = self
            .arena
            .components
            .iter()
            .position(|component| component.semantic.id == component_id)
            .ok_or_else(|| invalid("physical depthwise convolution component is absent"))?;
        if self.updates[component_index].is_some() {
            return Err(invalid(
                "physical depthwise convolution component was applied more than once",
            ));
        }
        let declared = &self.declared[component_index];
        if declared.dimensions.len() != 2
            || declared.dimensions[0].axis != crate::kv::v2::ShapeAxis::Batch
            || declared.dimensions[1].axis != crate::kv::v2::ShapeAxis::Hidden
        {
            return Err(invalid(
                "physical depthwise convolution requires [batch, hidden] ring components",
            ));
        }
        let (batch, hidden, steps) = input.dims3()?;
        let (weight_hidden, kernel) = weight.dims2()?;
        if u64::try_from(steps)
            .map_err(|_| invalid("physical depthwise convolution steps exceed u64"))?
            != self.declared_steps
            || u64::try_from(batch)
                .map_err(|_| invalid("physical depthwise convolution batch exceeds u64"))?
                != declared.dimensions[0].units
            || u64::try_from(hidden)
                .map_err(|_| invalid("physical depthwise convolution width exceeds u64"))?
                != declared.dimensions[1].units
            || weight_hidden != hidden
            || kernel != self.arena.capacity_steps
            || kernel == 0
        {
            return Err(invalid(
                "physical depthwise convolution input, weight, or ring geometry mismatch",
            ));
        }
        let component = self.arena.component(component_id)?;
        if !self.arena.device.same_device(input.device())
            || !self.arena.device.same_device(weight.device())
            || input.dtype() != component.storage.dtype()
            || weight.dtype() != component.storage.dtype()
        {
            return Err(invalid(
                "physical depthwise convolution tensors do not match arena storage",
            ));
        }

        let input = input.contiguous()?;
        let physical_ring = component.storage.as_tensor().clone();
        let output = try_lfm_shortconv_ring_sequence(
            &physical_ring,
            &input,
            weight,
            self.intent.expected_cursor,
            u64::try_from(self.arena.valid_length)
                .map_err(|_| invalid("physical depthwise convolution history exceeds u64"))?,
        );
        if physical_ring_kernel_required(&self.arena.device) && output.is_none() {
            return Err(Error::InferenceError(format!(
                "the {:?} physical ShortConv ring kernel is unavailable; accelerator inference cannot fall back to a second state path",
                backend_kind_for_device(&self.arena.device)
            )));
        }
        let output = if output.is_some() {
            output
        } else if self.intent.expected_cursor > 0 {
            Some(self.direct_ring_convolution(
                component_id,
                &input,
                weight,
                batch,
                hidden,
                steps,
                kernel,
            )?)
        } else {
            None
        };
        let output = match output {
            Some(output) => output,
            None => {
                let conv = Conv1d::new(
                    weight.reshape((hidden, 1, kernel))?.contiguous()?,
                    None,
                    Conv1dConfig {
                        padding: kernel.saturating_sub(1),
                        groups: hidden,
                        ..Default::default()
                    },
                );
                conv.forward(&input)?.narrow(2, 0, steps)?
            }
        }
        .contiguous()?;

        let update = input.permute((2, 0, 1))?.contiguous()?;
        self.updates[component_index] = Some(InvocationTensorBulkComponentValue {
            component: component_id,
            tensor: update,
        });
        Ok(output)
    }

    fn direct_ring_convolution(
        &self,
        component_id: StateComponentId,
        input: &Tensor,
        weight: &Tensor,
        batch: usize,
        hidden: usize,
        steps: usize,
        kernel: usize,
    ) -> Result<Tensor> {
        let expected = self.intent.expected_cursor;
        let valid = u64::try_from(self.arena.valid_length)
            .map_err(|_| invalid("physical depthwise convolution history exceeds u64"))?;
        let oldest = expected
            .checked_sub(valid)
            .ok_or_else(|| invalid("physical depthwise convolution history underflow"))?;
        let kernel_i128 = i128::try_from(kernel)
            .map_err(|_| invalid("physical depthwise convolution kernel exceeds i128"))?;
        let expected_i128 = i128::from(expected);
        let mut outputs = Vec::with_capacity(steps);
        for step in 0..steps {
            let step_i128 = i128::try_from(step)
                .map_err(|_| invalid("physical depthwise convolution step exceeds i128"))?;
            let window_start = expected_i128 + step_i128 + 1 - kernel_i128;
            let mut output = None::<Tensor>;
            for tap in 0..kernel {
                let tap_i128 = i128::try_from(tap)
                    .map_err(|_| invalid("physical depthwise convolution tap exceeds i128"))?;
                let source_absolute = window_start + tap_i128;
                if source_absolute < 0 {
                    continue;
                }
                let source_absolute = u64::try_from(source_absolute)
                    .map_err(|_| invalid("physical depthwise convolution cursor exceeds u64"))?;
                let source = if source_absolute < expected {
                    if source_absolute < oldest {
                        continue;
                    }
                    self.arena
                        .ring_component_step_view(component_id, source_absolute, &[batch, hidden])?
                        .unsqueeze(2)?
                } else {
                    let input_step = usize::try_from(source_absolute - expected)
                        .map_err(|_| invalid("physical depthwise input step exceeds usize"))?;
                    if input_step > step {
                        return Err(invalid(
                            "physical depthwise convolution attempted to read a future step",
                        ));
                    }
                    input.narrow(2, input_step, 1)?
                };
                let tap_weight = weight.narrow(1, tap, 1)?.reshape((1, hidden, 1))?;
                let contribution = source.broadcast_mul(&tap_weight)?;
                output = Some(match output {
                    Some(current) => (&current + &contribution)?,
                    None => contribution,
                });
            }
            outputs.push(output.ok_or_else(|| {
                invalid("physical depthwise convolution produced no current-step contribution")
            })?);
        }
        match outputs.as_slice() {
            [single] => Ok(single.clone()),
            _ => Tensor::cat(&outputs, 2).map_err(Error::from),
        }
    }

    /// Publish all layer inputs into the physical ring in one authenticated
    /// cursor advance. An incomplete transaction is discarded without
    /// changing physical state.
    pub(crate) fn commit(mut self) -> Result<()> {
        let updates = self
            .updates
            .drain(..)
            .map(|update| {
                update.ok_or_else(|| {
                    invalid("physical depthwise convolution transaction omitted a ring component")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.arena.apply_intent(
            &self.intent,
            InvocationTensorUpdateV2::RingAdvanceBulk {
                components: updates,
            },
        )
    }
}

fn physical_ring_kernel_required(device: &Device) -> bool {
    device.is_metal() || device.is_cuda()
}

fn domain_components<'a>(
    semantic: &'a StateDomainSpec,
    resolved: &'a ResolvedNonPagedDomainPlan,
) -> Result<(
    InvocationTensorDomainKind,
    &'a [TensorComponentSpec],
    &'a [crate::kv::v2::ResolvedTensorComponent],
    usize,
)> {
    match (semantic, resolved) {
        (
            StateDomainSpec::StaticTensor(semantic),
            ResolvedNonPagedDomainPlan::StaticTensor(resolved),
        ) => Ok((
            InvocationTensorDomainKind::StaticTensor,
            &semantic.components,
            &resolved.components,
            1,
        )),
        (StateDomainSpec::Tensor(semantic), ResolvedNonPagedDomainPlan::Tensor(resolved)) => Ok((
            InvocationTensorDomainKind::Tensor,
            &semantic.components,
            &resolved.components,
            1,
        )),
        (StateDomainSpec::Append(semantic), ResolvedNonPagedDomainPlan::Append(resolved)) => Ok((
            InvocationTensorDomainKind::Append,
            &semantic.components_per_step,
            &resolved.components_per_step,
            usize::try_from(semantic.max_steps)
                .map_err(|_| invalid("append capacity exceeds usize"))?,
        )),
        (StateDomainSpec::Ring(semantic), ResolvedNonPagedDomainPlan::Ring(resolved)) => Ok((
            InvocationTensorDomainKind::Ring,
            &semantic.components_per_step,
            &resolved.components_per_step,
            usize::try_from(semantic.capacity_steps)
                .map_err(|_| invalid("ring capacity exceeds usize"))?,
        )),
        (StateDomainSpec::StaticAttention(_), _)
        | (_, ResolvedNonPagedDomainPlan::StaticAttention(_)) => Err(invalid(
            "static attention requires a direct install-and-attend arena",
        )),
        _ => Err(invalid(
            "invocation tensor semantic and resolved domain kinds disagree",
        )),
    }
}

fn maximum_shape(dimensions: &[crate::kv::v2::ShapeDimension]) -> Result<Vec<usize>> {
    dimensions
        .iter()
        .map(|dimension| {
            usize::try_from(dimension.extent.max())
                .map_err(|_| invalid("invocation tensor dimension exceeds usize"))
        })
        .collect()
}

fn validate_input_tensor(
    tensor: &Tensor,
    component: &ComponentBacking,
    device: &Device,
) -> Result<()> {
    if tensor.device().location() != device.location()
        || tensor.dtype() != component.storage.dtype()
        || tensor.rank() != component.semantic.shape.dimensions.len()
    {
        return Err(invalid(
            "invocation tensor input has incompatible device, dtype, or rank",
        ));
    }
    for (actual, dimension) in tensor
        .dims()
        .iter()
        .zip(&component.semantic.shape.dimensions)
    {
        let actual = u64::try_from(*actual)
            .map_err(|_| invalid("invocation tensor dimension exceeds u64"))?;
        if !dimension.extent.accepts(actual) {
            return Err(invalid(
                "invocation tensor input violates its logical shape bound",
            ));
        }
    }
    Ok(())
}

fn shares_storage(left: &Tensor, right: &Tensor) -> bool {
    fn address(tensor: &Tensor) -> *const () {
        let (storage, _) = tensor.storage_and_layout();
        std::ptr::from_ref(&*storage).cast()
    }

    address(left) == address(right)
}

fn write_logical_value(
    component: &ComponentBacking,
    source: &Tensor,
    physical_step: Option<usize>,
) -> Result<()> {
    let maximum_elements =
        component
            .maximum_shape
            .iter()
            .try_fold(1_usize, |elements, extent| {
                elements
                    .checked_mul(*extent)
                    .ok_or_else(|| invalid("invocation tensor physical element count overflow"))
            })?;
    let backing = component.storage.as_tensor();
    let flat_backing = backing.flatten_all()?;
    let destination_base = physical_step
        .unwrap_or(0)
        .checked_mul(maximum_elements)
        .ok_or_else(|| invalid("invocation tensor destination offset overflow"))?;
    if physical_step.is_some() {
        flat_backing
            .narrow(0, destination_base, maximum_elements)?
            .zero_set()?;
    } else {
        backing.zero_set()?;
    }

    let flat_source = source.flatten_all()?;
    let logical_shape = source.dims();
    let (row_count, row_length) = match logical_shape.split_last() {
        Some((row_length, prefix)) => (
            prefix.iter().try_fold(1_usize, |rows, extent| {
                rows.checked_mul(*extent)
                    .ok_or_else(|| invalid("invocation tensor logical row count overflow"))
            })?,
            *row_length,
        ),
        None => (1, 1),
    };
    let prefix_rank = logical_shape.len().saturating_sub(1);
    for row in 0..row_count {
        let source_offset = row
            .checked_mul(row_length)
            .ok_or_else(|| invalid("invocation tensor source row offset overflow"))?;
        let source_row = flat_source.narrow(0, source_offset, row_length)?;
        let mut remaining = row;
        let mut destination_offset = destination_base;
        for dimension in (0..prefix_rank).rev() {
            let coordinate = remaining % logical_shape[dimension];
            remaining /= logical_shape[dimension];
            let stride = component.maximum_shape[dimension + 1..].iter().try_fold(
                1_usize,
                |stride, extent| {
                    stride
                        .checked_mul(*extent)
                        .ok_or_else(|| invalid("invocation tensor destination stride overflow"))
                },
            )?;
            destination_offset =
                destination_offset
                    .checked_add(coordinate.checked_mul(stride).ok_or_else(|| {
                        invalid("invocation tensor destination coordinate overflow")
                    })?)
                    .ok_or_else(|| invalid("invocation tensor destination row offset overflow"))?;
        }
        flat_backing.slice_set(&source_row, 0, destination_offset)?;
    }
    Ok(())
}

fn logical_view(
    tensor: &Tensor,
    logical_shape: &[usize],
    prefix_dimensions: usize,
) -> Result<Tensor> {
    let mut view = tensor.clone();
    for (index, extent) in logical_shape.iter().copied().enumerate() {
        let dimension = prefix_dimensions + index;
        if extent < view.dims()[dimension] {
            view = view.narrow(dimension, 0, extent)?;
        }
    }
    Ok(view)
}

fn logical_step_view(
    component: &ComponentBacking,
    start: usize,
    length: usize,
    logical_shape: &[usize],
) -> Result<Tensor> {
    let view = component.storage.narrow(0, start, length)?;
    logical_view(&view, logical_shape, 1)
}

fn tensor_bytes(tensor: &Tensor) -> Result<u64> {
    let bytes = tensor
        .elem_count()
        .checked_mul(tensor.dtype().size_in_bytes())
        .ok_or_else(|| invalid("invocation tensor byte count overflow"))?;
    u64::try_from(bytes).map_err(|_| invalid("invocation tensor byte count exceeds u64"))
}

fn candle_dtype(storage: StateStorageFormat) -> Result<DType> {
    match storage.dtype() {
        StateDType::F32 => Ok(DType::F32),
        StateDType::F16 => Ok(DType::F16),
        StateDType::Bf16 => Ok(DType::BF16),
        StateDType::I64 => Ok(DType::I64),
        StateDType::I8 | StateDType::Q4 => Err(invalid(
            "quantized invocation tensor state requires an explicit packing ABI",
        )),
    }
}

fn device_ordinal(device: &Device) -> Result<Option<u32>> {
    match device.location() {
        DeviceLocation::Cpu => Ok(None),
        DeviceLocation::Cuda { gpu_id } => u32::try_from(gpu_id)
            .map(Some)
            .map_err(|_| invalid("CUDA device identity exceeds u32")),
        DeviceLocation::Metal { gpu_id } => {
            let id = gpu_id as u64;
            Ok(Some((id ^ (id >> 32)) as u32))
        }
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::state::{negotiate_state_plan, StateBackendPlanRequest};
    use crate::backends::BackendKind;
    use crate::kv::v2::{
        test_contract, AppendStateDomainSpec, BoundedShape, CheckpointPolicy,
        InferenceStateContract, PlacementPolicy, PrefixPolicy, RingStateDomainSpec, ShapeAxis,
        ShapeDimension, ShapeDimensionValue, ShapeExtent, StateClock, StateDomainHeader,
        StateGroupSpec, StaticTensorDomainSpec, TensorRole, TensorStateDomainSpec,
        WorkspaceFormula, CURRENT_INFERENCE_STATE_ABI,
    };

    fn component(max: u64) -> TensorComponentSpec {
        component_with_extent(ShapeExtent::RuntimeBounded { min: 1, max })
    }

    fn fixed_component(value: u64) -> TensorComponentSpec {
        component_with_extent(ShapeExtent::Fixed { value })
    }

    fn component_with_extent(extent: ShapeExtent) -> TensorComponentSpec {
        TensorComponentSpec {
            id: StateComponentId::new(1),
            role: TensorRole::Control,
            shape: BoundedShape {
                dimensions: vec![ShapeDimension {
                    axis: ShapeAxis::Hidden,
                    extent,
                }],
            },
            accepted_dtypes: vec![StateDType::F32],
        }
    }

    fn bounded_matrix_component(rows: u64, columns: u64) -> TensorComponentSpec {
        TensorComponentSpec {
            id: StateComponentId::new(1),
            role: TensorRole::Control,
            shape: BoundedShape {
                dimensions: vec![
                    ShapeDimension {
                        axis: ShapeAxis::Sequence,
                        extent: ShapeExtent::RuntimeBounded { min: 1, max: rows },
                    },
                    ShapeDimension {
                        axis: ShapeAxis::Hidden,
                        extent: ShapeExtent::RuntimeBounded {
                            min: 1,
                            max: columns,
                        },
                    },
                ],
            },
            accepted_dtypes: vec![StateDType::F32],
        }
    }

    fn header(id: u32) -> StateDomainHeader {
        StateDomainHeader {
            id: StateDomainId::new(id),
            scope: StateScope::Invocation,
            clock: StateClock::DecoderTokens,
            placement: PlacementPolicy::BackendLocal,
            prefix: PrefixPolicy::Disabled,
            checkpoint: CheckpointPolicy::None,
        }
    }

    fn arena_from_contract(
        contract: &InferenceStateContract,
        state: StateDomainSpec,
    ) -> InvocationTensorArena {
        let plan = Arc::new(
            negotiate_state_plan(
                contract,
                &StateBackendPlanRequest {
                    backend: BackendKind::Cpu,
                    device_ordinal: None,
                    page_tokens_hint: None,
                    storage_dtype_hint: None,
                },
            )
            .unwrap(),
        );
        let fixed_bytes = plan
            .non_paged
            .iter()
            .find(|resolved| resolved.domain() == state.id())
            .unwrap()
            .maximum_bytes();
        InvocationTensorArena::new(
            contract,
            plan,
            InvocationWorkspaceDomain::State {
                state,
                capacity: InvocationStateCapacity::SemanticBounded,
                placement: PlacementPolicy::BackendLocal,
                formula: WorkspaceFormula {
                    fixed_bytes,
                    dimensions: vec![],
                    terms: vec![],
                },
            },
            Device::Cpu,
        )
        .unwrap()
    }

    fn arena(state: StateDomainSpec) -> InvocationTensorArena {
        let domain = state.id();
        let contract = InferenceStateContract {
            abi: CURRENT_INFERENCE_STATE_ABI,
            domains: vec![state.clone()],
            groups: vec![StateGroupSpec {
                id: StateGroupId::new(1),
                domains: vec![domain],
                prefix_shareable: false,
            }],
        };
        arena_from_contract(&contract, state)
    }

    fn value(values: &[f32]) -> InvocationTensorComponentValue {
        InvocationTensorComponentValue {
            component: StateComponentId::new(1),
            tensor: Tensor::from_slice(values, values.len(), &Device::Cpu).unwrap(),
        }
    }

    fn matrix_value(values: &[f32], rows: usize, columns: usize) -> InvocationTensorComponentValue {
        InvocationTensorComponentValue {
            component: StateComponentId::new(1),
            tensor: Tensor::from_slice(values, (rows, columns), &Device::Cpu).unwrap(),
        }
    }

    fn step(values: &[f32]) -> InvocationTensorStepValues {
        InvocationTensorStepValues {
            components: vec![value(values)],
        }
    }

    fn source_identity() -> [u8; 32] {
        [7; 32]
    }

    fn declared_component(units: u64) -> ComponentShapeInstantiation {
        ComponentShapeInstantiation {
            component: StateComponentId::new(1),
            dimensions: vec![ShapeDimensionValue {
                axis: ShapeAxis::Hidden,
                units,
            }],
        }
    }

    fn shortconv_component(id: u32, hidden: u64) -> TensorComponentSpec {
        TensorComponentSpec {
            id: StateComponentId::new(id),
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
        }
    }

    fn shortconv_declared(id: u32, hidden: u64) -> ComponentShapeInstantiation {
        ComponentShapeInstantiation {
            component: StateComponentId::new(id),
            dimensions: vec![
                ShapeDimensionValue {
                    axis: ShapeAxis::Batch,
                    units: 1,
                },
                ShapeDimensionValue {
                    axis: ShapeAxis::Hidden,
                    units: hidden,
                },
            ],
        }
    }

    #[test]
    fn fixed_shape_replace_keeps_the_same_physical_backing() {
        let mut arena = arena(StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: header(1),
            components: vec![fixed_component(4)],
        }));
        let before = arena.backing(StateComponentId::new(1)).unwrap().id();
        assert!(arena
            .install(source_identity(), 0, &[value(&[1.0, 2.0, 3.0, 4.0])])
            .is_err());
        arena
            .replace(0, 3, &[value(&[1.0, 2.0, 3.0, 4.0])])
            .unwrap();
        let snapshot = arena.snapshot().unwrap();
        assert_eq!(snapshot.source_identity, None);
        assert_eq!(snapshot.absolute_cursor, 3);
        assert_eq!(
            snapshot.components[0].tensor.to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        arena
            .replace(3, 5, &[value(&[5.0, 6.0, 7.0, 8.0])])
            .unwrap();
        assert_eq!(arena.absolute_cursor(), 5);
        let after = arena.backing(StateComponentId::new(1)).unwrap().id();
        assert_eq!(before, after);
        assert_eq!(
            arena
                .backing(StateComponentId::new(1))
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![5.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(
            snapshot.components[0].tensor.to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        arena.reset().unwrap();
        assert_eq!(
            snapshot.components[0].tensor.to_vec1::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn bounded_matrix_rows_write_in_place_and_clear_all_stale_tails() {
        let mut arena = arena(StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: header(1),
            components: vec![bounded_matrix_component(2, 3)],
        }));
        let non_contiguous = InvocationTensorComponentValue {
            component: StateComponentId::new(1),
            tensor: Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu)
                .unwrap()
                .transpose(0, 1)
                .unwrap(),
        };
        assert!(arena.replace(0, 1, &[non_contiguous]).is_err());
        assert!(!arena.is_dirty());
        assert_eq!(arena.absolute_cursor(), 0);
        arena
            .replace(0, 1, &[matrix_value(&[1.0, 2.0, 3.0, 4.0], 2, 2)])
            .unwrap();
        let held = arena.snapshot().unwrap();
        assert_eq!(held.components[0].tensor.dims(), &[2, 2]);
        assert_eq!(
            held.components[0]
                .tensor
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            arena
                .backing(StateComponentId::new(1))
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1.0, 2.0, 0.0, 3.0, 4.0, 0.0]
        );

        arena.replace(1, 2, &[matrix_value(&[9.0], 1, 1)]).unwrap();
        assert_eq!(
            arena
                .backing(StateComponentId::new(1))
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![9.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            held.components[0]
                .tensor
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn alias_preflight_is_clean_and_dirty_state_requires_reset() {
        let mut arena = arena(StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: header(1),
            components: vec![fixed_component(4)],
        }));
        arena
            .replace(0, 1, &[value(&[1.0, 2.0, 3.0, 4.0])])
            .unwrap();
        arena.prepare_completion().unwrap();
        let aliased_value = InvocationTensorComponentValue {
            component: StateComponentId::new(1),
            tensor: arena.backing(StateComponentId::new(1)).unwrap(),
        };
        assert!(arena.replace(1, 2, &[aliased_value]).is_err());
        assert!(!arena.is_dirty());
        assert_eq!(arena.absolute_cursor(), 1);
        assert_eq!(
            arena.snapshot().unwrap().components[0]
                .tensor
                .to_vec1::<f32>()
                .unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );

        // CPU writes do not expose a deterministic device-failure injection.
        // Model the state left by any failed write/sync and verify that only a
        // successful reset makes the arena observable and reusable.
        arena.dirty = true;
        assert!(arena.prepare_completion().is_err());
        assert!(arena.snapshot().is_err());
        assert!(arena.backing(StateComponentId::new(1)).is_err());
        assert!(arena
            .replace(1, 2, &[value(&[5.0, 6.0, 7.0, 8.0])])
            .is_err());

        arena.reset().unwrap();
        arena.prepare_completion().unwrap();
        assert!(!arena.is_dirty());
        assert_eq!(
            arena
                .backing(StateComponentId::new(1))
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![0.0; 4]
        );
        arena
            .replace(0, 1, &[value(&[9.0, 8.0, 7.0, 6.0])])
            .unwrap();
    }

    #[test]
    fn static_tensor_install_preserves_the_exact_source_identity() {
        let mut arena = arena(StateDomainSpec::StaticTensor(StaticTensorDomainSpec {
            header: header(1),
            components: vec![fixed_component(4)],
        }));
        assert!(arena
            .install([0; 32], 1, &[value(&[1.0, 2.0, 3.0, 4.0])])
            .is_err());
        arena
            .install(source_identity(), 4, &[value(&[1.0, 2.0, 3.0, 4.0])])
            .unwrap();
        let snapshot = arena.snapshot().unwrap();
        assert_eq!(snapshot.source_identity, Some(source_identity()));
        assert_eq!(snapshot.absolute_cursor, 4);
        assert!(arena
            .replace(4, 5, &[value(&[5.0, 6.0, 7.0, 8.0])])
            .is_err());
    }

    #[test]
    fn append_validates_a_multi_step_update_before_any_write() {
        let mut arena = arena(StateDomainSpec::Append(AppendStateDomainSpec {
            header: header(1),
            components_per_step: vec![component(2)],
            max_steps: 4,
        }));
        assert!(arena
            .append(0, 2, &[step(&[1.0]), step(&[2.0, 3.0, 4.0])])
            .is_err());
        assert_eq!(arena.absolute_cursor(), 0);
        assert!(!arena.is_dirty());
        assert_eq!(
            arena
                .backing(StateComponentId::new(1))
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![0.0; 8]
        );

        assert!(arena.append(0, 1, &[step(&[1.0]), step(&[2.0])]).is_err());
        arena.append(0, 2, &[step(&[1.0]), step(&[2.0])]).unwrap();
        assert!(arena.append(0, 3, &[step(&[3.0])]).is_err());
        assert!(arena.append(2, 2, &[]).is_err());
        assert_eq!(arena.absolute_cursor(), 2);
        assert!(!arena.is_dirty());

        let segments = arena.chronological_segments().unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].absolute_start, 0);
        assert_eq!(segments[0].length, 2);
        assert_eq!(
            segments[0].components[0]
                .tensor
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![1.0, 2.0]
        );
    }

    #[test]
    fn ring_segments_preserve_per_slot_extents_across_wrap() {
        let mut arena = arena(StateDomainSpec::Ring(RingStateDomainSpec {
            header: header(1),
            components_per_step: vec![component(2)],
            capacity_steps: 3,
        }));
        arena
            .ring_advance(0, 2, &[step(&[1.0, 2.0]), step(&[3.0, 4.0])])
            .unwrap();
        arena
            .ring_advance(2, 4, &[step(&[5.0, 6.0]), step(&[7.0])])
            .unwrap();
        assert_eq!(arena.absolute_cursor(), 4);
        assert_eq!(arena.valid_length(), 3);
        let segments = arena.chronological_segments().unwrap();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].absolute_start, 1);
        assert_eq!(segments[0].length, 2);
        assert_eq!(
            segments[0].components[0]
                .tensor
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(segments[0].components[0].tensor.dims(), &[2, 2]);
        assert_eq!(segments[1].absolute_start, 3);
        assert_eq!(segments[1].length, 1);
        assert_eq!(
            segments[1].components[0]
                .tensor
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![7.0]
        );
        assert_eq!(segments[1].components[0].tensor.dims(), &[1, 1]);
        assert_eq!(
            arena
                .backing(StateComponentId::new(1))
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![7.0, 0.0, 3.0, 4.0, 5.0, 6.0]
        );
        arena.reset().unwrap();
        assert_eq!(
            segments[0].components[0]
                .tensor
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(
            segments[1].components[0]
                .tensor
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![7.0]
        );
    }

    #[test]
    fn one_ring_advance_larger_than_capacity_retains_only_its_final_steps() {
        let mut arena = arena(StateDomainSpec::Ring(RingStateDomainSpec {
            header: header(1),
            components_per_step: vec![component(2)],
            capacity_steps: 3,
        }));
        assert!(arena
            .ring_advance(
                0,
                5,
                &[
                    step(&[0.0, 1.0, 2.0]),
                    step(&[3.0]),
                    step(&[4.0, 5.0]),
                    step(&[6.0]),
                    step(&[7.0, 8.0]),
                ],
            )
            .is_err());
        assert_eq!(arena.absolute_cursor(), 0);
        assert!(!arena.is_dirty());

        arena
            .ring_advance(
                0,
                5,
                &[
                    step(&[1.0, 2.0]),
                    step(&[3.0]),
                    step(&[4.0, 5.0]),
                    step(&[6.0]),
                    step(&[7.0, 8.0]),
                ],
            )
            .unwrap();
        assert_eq!(arena.absolute_cursor(), 5);
        assert_eq!(arena.valid_length(), 3);
        assert_eq!(
            arena
                .backing(StateComponentId::new(1))
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![6.0, 0.0, 7.0, 8.0, 4.0, 5.0]
        );
        let segments = arena.chronological_segments().unwrap();
        assert_eq!(
            segments
                .iter()
                .map(|segment| (segment.absolute_start, segment.length))
                .collect::<Vec<_>>(),
            vec![(2, 1), (3, 1), (4, 1)]
        );
        assert_eq!(
            segments[0].components[0]
                .tensor
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![4.0, 5.0]
        );
        assert_eq!(
            segments[1].components[0]
                .tensor
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![6.0]
        );
        assert_eq!(
            segments[2].components[0]
                .tensor
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![7.0, 8.0]
        );
    }

    #[test]
    fn authenticated_bulk_ring_advance_writes_only_the_physical_tail() {
        let mut arena = arena(StateDomainSpec::Ring(RingStateDomainSpec {
            header: header(1),
            components_per_step: vec![fixed_component(2)],
            capacity_steps: 3,
        }));
        let intent = DomainStepIntent {
            domain: StateDomainId::new(1),
            expected_cursor: 0,
            target_cursor: 5,
            update: StateUpdateKind::RingAdvance {
                steps: 5,
                components_per_step: vec![declared_component(2)],
            },
        };
        let update = InvocationTensorUpdateV2::RingAdvanceBulk {
            components: vec![InvocationTensorBulkComponentValue {
                component: StateComponentId::new(1),
                tensor: Tensor::from_slice(
                    &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                    (5, 2),
                    &Device::Cpu,
                )
                .unwrap(),
            }],
        };

        arena.apply_intent(&intent, update).unwrap();

        assert_eq!(arena.absolute_cursor(), 5);
        assert_eq!(arena.valid_length(), 3);
        assert_eq!(
            arena
                .backing(StateComponentId::new(1))
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![5.0, 6.0]]
        );
        let segments = arena.read_chronological_segments().unwrap();
        assert_eq!(
            segments
                .iter()
                .flat_map(|segment| segment.components[0].tensor.to_vec2::<f32>().unwrap())
                .collect::<Vec<_>>(),
            vec![vec![5.0, 6.0], vec![7.0, 8.0], vec![9.0, 10.0]]
        );
    }

    #[test]
    fn sealed_depthwise_conv_transaction_matches_causal_reference_and_commits_atomically() {
        let mut arena = arena(StateDomainSpec::Ring(RingStateDomainSpec {
            header: header(1),
            components_per_step: vec![shortconv_component(1, 2), shortconv_component(2, 2)],
            capacity_steps: 3,
        }));
        let prefill_intent = DomainStepIntent {
            domain: StateDomainId::new(1),
            expected_cursor: 0,
            target_cursor: 4,
            update: StateUpdateKind::RingAdvance {
                steps: 4,
                components_per_step: vec![shortconv_declared(1, 2), shortconv_declared(2, 2)],
            },
        };
        let input_one = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 0.5, 1.0, 1.5, 2.0],
            (1, 2, 4),
            &Device::Cpu,
        )
        .unwrap();
        let input_two = Tensor::from_slice(
            &[2.0f32, 1.0, 0.0, -1.0, 1.0, 1.5, 2.0, 2.5],
            (1, 2, 4),
            &Device::Cpu,
        )
        .unwrap();
        let weight =
            Tensor::from_slice(&[0.1f32, 0.2, 0.3, -0.5, 0.25, 0.75], (2, 3), &Device::Cpu)
                .unwrap();

        {
            let mut incomplete = arena.begin_ring_depthwise_conv(&prefill_intent).unwrap();
            incomplete
                .apply(StateComponentId::new(1), &input_one, &weight)
                .unwrap();
            assert!(incomplete.commit().is_err());
        }
        assert_eq!(arena.absolute_cursor(), 0);
        assert_eq!(arena.valid_length(), 0);
        assert!(!arena.is_dirty());

        let (prefill_one, prefill_two) = {
            let mut transaction = arena.begin_ring_depthwise_conv(&prefill_intent).unwrap();
            let first = transaction
                .apply(StateComponentId::new(1), &input_one, &weight)
                .unwrap();
            assert!(transaction
                .apply(StateComponentId::new(1), &input_one, &weight)
                .is_err());
            let second = transaction
                .apply(StateComponentId::new(2), &input_two, &weight)
                .unwrap();
            transaction.commit().unwrap();
            (first, second)
        };
        assert_nested_f32_close(
            prefill_one.to_vec3::<f32>().unwrap(),
            causal_depthwise_reference(&input_one, &weight, None),
        );
        assert_nested_f32_close(
            prefill_two.to_vec3::<f32>().unwrap(),
            causal_depthwise_reference(&input_two, &weight, None),
        );
        assert_eq!(arena.absolute_cursor(), 4);
        assert_eq!(arena.valid_length(), 3);
        assert_ne!(
            prefill_one.id(),
            arena.backing(StateComponentId::new(1)).unwrap().id()
        );

        let decode_input = Tensor::from_slice(&[5.0f32, 2.5], (1, 2, 1), &Device::Cpu).unwrap();
        let history = input_one.narrow(2, 1, 3).unwrap();
        let decode_intent = DomainStepIntent {
            domain: StateDomainId::new(1),
            expected_cursor: 4,
            target_cursor: 5,
            update: StateUpdateKind::RingAdvance {
                steps: 1,
                components_per_step: vec![shortconv_declared(1, 2), shortconv_declared(2, 2)],
            },
        };
        let decode_output = {
            let mut transaction = arena.begin_ring_depthwise_conv(&decode_intent).unwrap();
            let output = transaction
                .apply(StateComponentId::new(1), &decode_input, &weight)
                .unwrap();
            transaction
                .apply(StateComponentId::new(2), &decode_input, &weight)
                .unwrap();
            transaction.commit().unwrap();
            output
        };
        assert_nested_f32_close(
            decode_output.to_vec3::<f32>().unwrap(),
            causal_depthwise_reference(&decode_input, &weight, Some(&history)),
        );
        assert_eq!(arena.absolute_cursor(), 5);
        assert_eq!(arena.valid_length(), 3);

        let continuation =
            Tensor::from_slice(&[6.0f32, 7.0, 3.0, 3.5], (1, 2, 2), &Device::Cpu).unwrap();
        let continuation_history =
            Tensor::from_slice(&[3.0f32, 4.0, 5.0, 1.5, 2.0, 2.5], (1, 2, 3), &Device::Cpu)
                .unwrap();
        let continuation_intent = DomainStepIntent {
            domain: StateDomainId::new(1),
            expected_cursor: 5,
            target_cursor: 7,
            update: StateUpdateKind::RingAdvance {
                steps: 2,
                components_per_step: vec![shortconv_declared(1, 2), shortconv_declared(2, 2)],
            },
        };
        let continuation_output = {
            let mut transaction = arena
                .begin_ring_depthwise_conv(&continuation_intent)
                .unwrap();
            let output = transaction
                .apply(StateComponentId::new(1), &continuation, &weight)
                .unwrap();
            transaction
                .apply(StateComponentId::new(2), &continuation, &weight)
                .unwrap();
            transaction.commit().unwrap();
            output
        };
        assert_nested_f32_close(
            continuation_output.to_vec3::<f32>().unwrap(),
            causal_depthwise_reference(&continuation, &weight, Some(&continuation_history)),
        );
        assert_eq!(arena.absolute_cursor(), 7);
        assert_eq!(arena.valid_length(), 3);
    }

    fn causal_depthwise_reference(
        input: &Tensor,
        weight: &Tensor,
        history: Option<&Tensor>,
    ) -> Vec<Vec<Vec<f32>>> {
        let input = input.to_vec3::<f32>().unwrap();
        let weight = weight.to_vec2::<f32>().unwrap();
        let batch = input.len();
        let hidden = input[0].len();
        let steps = input[0][0].len();
        let kernel = weight[0].len();
        let prior = history
            .map(|history| history.to_vec3::<f32>().unwrap())
            .unwrap_or_else(|| vec![vec![vec![0.0; kernel]; hidden]; batch]);
        let mut output = vec![vec![vec![0.0; steps]; hidden]; batch];
        for batch_index in 0..batch {
            for hidden_index in 0..hidden {
                let mut state = prior[batch_index][hidden_index].clone();
                if state.len() < kernel {
                    let mut padded = vec![0.0; kernel - state.len()];
                    padded.extend(state);
                    state = padded;
                }
                for step in 0..steps {
                    state.remove(0);
                    state.push(input[batch_index][hidden_index][step]);
                    output[batch_index][hidden_index][step] = state
                        .iter()
                        .zip(&weight[hidden_index])
                        .map(|(state, weight)| state * weight)
                        .sum();
                }
            }
        }
        output
    }

    fn assert_nested_f32_close(actual: Vec<Vec<Vec<f32>>>, expected: Vec<Vec<Vec<f32>>>) {
        assert_eq!(actual.len(), expected.len(), "batch size mismatch");
        for (batch, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.len(),
                expected.len(),
                "hidden size mismatch at batch {batch}"
            );
            for (hidden, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                assert_eq!(
                    actual.len(),
                    expected.len(),
                    "step count mismatch at batch {batch}, hidden {hidden}"
                );
                for (step, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                    assert!(
                        actual.is_finite()
                            && expected.is_finite()
                            && (actual - expected).abs() <= 1e-6,
                        "value mismatch at batch {batch}, hidden {hidden}, step {step}: {actual} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn sealed_depthwise_conv_handles_partial_history_and_non_fused_kernels() {
        let mut arena = arena(StateDomainSpec::Ring(RingStateDomainSpec {
            header: header(1),
            components_per_step: vec![shortconv_component(1, 2)],
            capacity_steps: 4,
        }));
        let weight = Tensor::from_slice(
            &[0.1f32, 0.2, 0.3, 0.4, -0.5, 0.25, 0.75, 0.1],
            (2, 4),
            &Device::Cpu,
        )
        .unwrap();
        let first = Tensor::from_slice(&[1.0f32, 2.0], (1, 2, 1), &Device::Cpu).unwrap();
        let first_intent = DomainStepIntent {
            domain: StateDomainId::new(1),
            expected_cursor: 0,
            target_cursor: 1,
            update: StateUpdateKind::RingAdvance {
                steps: 1,
                components_per_step: vec![shortconv_declared(1, 2)],
            },
        };
        let first_output = {
            let mut transaction = arena.begin_ring_depthwise_conv(&first_intent).unwrap();
            let output = transaction
                .apply(StateComponentId::new(1), &first, &weight)
                .unwrap();
            transaction.commit().unwrap();
            output
        };
        assert_eq!(
            first_output.to_vec3::<f32>().unwrap(),
            causal_depthwise_reference(&first, &weight, None)
        );

        let second = Tensor::from_slice(&[2.0f32, 3.0], (1, 2, 1), &Device::Cpu).unwrap();
        let second_intent = DomainStepIntent {
            domain: StateDomainId::new(1),
            expected_cursor: 1,
            target_cursor: 2,
            update: StateUpdateKind::RingAdvance {
                steps: 1,
                components_per_step: vec![shortconv_declared(1, 2)],
            },
        };
        let second_output = {
            let mut transaction = arena.begin_ring_depthwise_conv(&second_intent).unwrap();
            assert!(transaction
                .apply(
                    StateComponentId::new(1),
                    &second,
                    &weight.to_dtype(DType::F64).unwrap(),
                )
                .is_err());
            let output = transaction
                .apply(StateComponentId::new(1), &second, &weight)
                .unwrap();
            transaction.commit().unwrap();
            output
        };
        assert_eq!(
            second_output.to_vec3::<f32>().unwrap(),
            causal_depthwise_reference(&second, &weight, Some(&first))
        );
        assert_eq!(arena.absolute_cursor(), 2);
        assert_eq!(arena.valid_length(), 2);
    }

    #[test]
    fn bulk_ring_mismatch_fails_before_mutating_the_arena() {
        let mut arena = arena(StateDomainSpec::Ring(RingStateDomainSpec {
            header: header(1),
            components_per_step: vec![fixed_component(2)],
            capacity_steps: 3,
        }));
        let intent = DomainStepIntent {
            domain: StateDomainId::new(1),
            expected_cursor: 0,
            target_cursor: 4,
            update: StateUpdateKind::RingAdvance {
                steps: 4,
                components_per_step: vec![declared_component(2)],
            },
        };
        let update = InvocationTensorUpdateV2::RingAdvanceBulk {
            components: vec![InvocationTensorBulkComponentValue {
                component: StateComponentId::new(1),
                tensor: Tensor::zeros((3, 2), DType::F32, &Device::Cpu).unwrap(),
            }],
        };

        assert!(arena.apply_intent(&intent, update).is_err());
        assert_eq!(arena.absolute_cursor(), 0);
        assert_eq!(arena.valid_length(), 0);
        assert!(!arena.is_dirty());
        assert_eq!(
            arena
                .backing(StateComponentId::new(1))
                .unwrap()
                .to_vec2::<f32>()
                .unwrap(),
            vec![vec![0.0; 2]; 3]
        );
    }

    #[test]
    fn bulk_ring_rejects_truncated_semantic_rank_before_mutation() {
        let mut arena = arena(StateDomainSpec::Ring(RingStateDomainSpec {
            header: header(1),
            components_per_step: vec![bounded_matrix_component(2, 3)],
            capacity_steps: 3,
        }));
        let intent = DomainStepIntent {
            domain: StateDomainId::new(1),
            expected_cursor: 0,
            target_cursor: 4,
            update: StateUpdateKind::RingAdvance {
                steps: 4,
                components_per_step: vec![ComponentShapeInstantiation {
                    component: StateComponentId::new(1),
                    dimensions: vec![ShapeDimensionValue {
                        axis: ShapeAxis::Sequence,
                        units: 2,
                    }],
                }],
            },
        };
        let update = InvocationTensorUpdateV2::RingAdvanceBulk {
            components: vec![InvocationTensorBulkComponentValue {
                component: StateComponentId::new(1),
                tensor: Tensor::zeros((4, 2), DType::F32, &Device::Cpu).unwrap(),
            }],
        };

        assert!(arena.apply_intent(&intent, update).is_err());
        assert_eq!(arena.absolute_cursor(), 0);
        assert_eq!(arena.valid_length(), 0);
        assert!(!arena.is_dirty());
        assert_eq!(
            arena
                .backing(StateComponentId::new(1))
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![0.0; 18]
        );
    }

    #[test]
    fn arena_selects_one_non_paged_domain_from_a_full_mixed_plan() {
        let state = StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: header(2),
            components: vec![fixed_component(4)],
        });
        let other_state = StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: header(3),
            components: vec![fixed_component(2)],
        });
        let mut contract = test_contract();
        contract.domains.push(state.clone());
        contract.domains.push(other_state);
        contract.groups.push(StateGroupSpec {
            id: StateGroupId::new(2),
            domains: vec![StateDomainId::new(2)],
            prefix_shareable: false,
        });
        contract.groups.push(StateGroupSpec {
            id: StateGroupId::new(3),
            domains: vec![StateDomainId::new(3)],
            prefix_shareable: false,
        });

        let mut arena = arena_from_contract(&contract, state);
        assert_eq!(arena.plan().paged_attention.len(), 1);
        assert_eq!(arena.plan().non_paged.len(), 2);
        assert_eq!(arena.domain(), StateDomainId::new(2));
        assert_eq!(arena.group(), StateGroupId::new(2));
        arena
            .replace(0, 1, &[value(&[1.0, 2.0, 3.0, 4.0])])
            .unwrap();
    }

    #[test]
    fn arena_rejects_same_geometry_with_a_noncanonical_clock() {
        let state = StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: header(1),
            components: vec![fixed_component(4)],
        });
        let contract = InferenceStateContract {
            abi: CURRENT_INFERENCE_STATE_ABI,
            domains: vec![state.clone()],
            groups: vec![StateGroupSpec {
                id: StateGroupId::new(1),
                domains: vec![StateDomainId::new(1)],
                prefix_shareable: false,
            }],
        };
        let plan = Arc::new(
            negotiate_state_plan(
                &contract,
                &StateBackendPlanRequest {
                    backend: BackendKind::Cpu,
                    device_ordinal: None,
                    page_tokens_hint: None,
                    storage_dtype_hint: None,
                },
            )
            .unwrap(),
        );
        let fixed_bytes = plan.non_paged[0].maximum_bytes();
        let mut wrong_clock = state;
        let StateDomainSpec::Tensor(wrong_clock) = &mut wrong_clock else {
            unreachable!()
        };
        wrong_clock.header.clock = StateClock::EncoderTokens;

        assert!(InvocationTensorArena::new(
            &contract,
            plan,
            InvocationWorkspaceDomain::State {
                state: StateDomainSpec::Tensor(wrong_clock.clone()),
                capacity: InvocationStateCapacity::SemanticBounded,
                placement: PlacementPolicy::BackendLocal,
                formula: WorkspaceFormula {
                    fixed_bytes,
                    dimensions: vec![],
                    terms: vec![],
                },
            },
            Device::Cpu,
        )
        .is_err());
    }

    #[test]
    fn authenticated_intent_rejects_every_mismatch_before_mutation() {
        let mut tensor = arena(StateDomainSpec::Tensor(TensorStateDomainSpec {
            header: header(1),
            components: vec![component(4)],
        }));
        let exact = DomainStepIntent {
            domain: StateDomainId::new(1),
            expected_cursor: 0,
            target_cursor: 1,
            update: StateUpdateKind::TensorReplace {
                components: vec![declared_component(2)],
            },
        };
        let update = || InvocationTensorUpdateV2::TensorReplace {
            components: vec![value(&[1.0, 2.0])],
        };
        let mut wrong_domain = exact.clone();
        wrong_domain.domain = StateDomainId::new(2);
        assert!(tensor.apply_intent(&wrong_domain, update()).is_err());
        let mut wrong_cursor = exact.clone();
        wrong_cursor.expected_cursor = 1;
        assert!(tensor.apply_intent(&wrong_cursor, update()).is_err());
        let mut wrong_shape = exact.clone();
        let StateUpdateKind::TensorReplace { components } = &mut wrong_shape.update else {
            unreachable!()
        };
        components[0].dimensions[0].units = 3;
        assert!(tensor.apply_intent(&wrong_shape, update()).is_err());
        assert!(tensor
            .apply_intent(&exact, InvocationTensorUpdateV2::NoOp)
            .is_err());
        assert_eq!(tensor.absolute_cursor(), 0);
        assert_eq!(
            tensor
                .backing(StateComponentId::new(1))
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![0.0; 4]
        );
        tensor.apply_intent(&exact, update()).unwrap();

        let no_op = DomainStepIntent {
            domain: StateDomainId::new(1),
            expected_cursor: 1,
            target_cursor: 1,
            update: StateUpdateKind::NoOp,
        };
        tensor
            .apply_intent(&no_op, InvocationTensorUpdateV2::NoOp)
            .unwrap();
        let reset = DomainStepIntent {
            domain: StateDomainId::new(1),
            expected_cursor: 1,
            target_cursor: 0,
            update: StateUpdateKind::Reset,
        };
        tensor
            .apply_intent(&reset, InvocationTensorUpdateV2::Reset)
            .unwrap();

        let mut static_tensor = arena(StateDomainSpec::StaticTensor(StaticTensorDomainSpec {
            header: header(1),
            components: vec![fixed_component(2)],
        }));
        let static_intent = DomainStepIntent {
            domain: StateDomainId::new(1),
            expected_cursor: 0,
            target_cursor: 2,
            update: StateUpdateKind::StaticInitialize {
                source_identity: source_identity(),
                components: vec![declared_component(2)],
            },
        };
        assert!(static_tensor
            .apply_intent(
                &static_intent,
                InvocationTensorUpdateV2::StaticInitialize {
                    source_identity: [8; 32],
                    components: vec![value(&[1.0, 2.0])],
                },
            )
            .is_err());
        assert_eq!(static_tensor.absolute_cursor(), 0);
        static_tensor
            .apply_intent(
                &static_intent,
                InvocationTensorUpdateV2::StaticInitialize {
                    source_identity: source_identity(),
                    components: vec![value(&[1.0, 2.0])],
                },
            )
            .unwrap();
    }

    #[test]
    fn authenticated_multi_step_validation_is_atomic() {
        let mut append = arena(StateDomainSpec::Append(AppendStateDomainSpec {
            header: header(1),
            components_per_step: vec![component(2)],
            max_steps: 4,
        }));
        let intent = DomainStepIntent {
            domain: StateDomainId::new(1),
            expected_cursor: 0,
            target_cursor: 2,
            update: StateUpdateKind::Append {
                steps: 2,
                components_per_step: vec![declared_component(1)],
            },
        };
        assert!(append
            .apply_intent(
                &intent,
                InvocationTensorUpdateV2::Append {
                    steps: vec![step(&[1.0]), step(&[2.0, 3.0])],
                },
            )
            .is_err());
        assert_eq!(append.absolute_cursor(), 0);
        assert!(!append.is_dirty());
        assert_eq!(
            append
                .backing(StateComponentId::new(1))
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![0.0; 8]
        );
        append
            .apply_intent(
                &intent,
                InvocationTensorUpdateV2::Append {
                    steps: vec![step(&[1.0]), step(&[2.0])],
                },
            )
            .unwrap();

        let mut ring = arena(StateDomainSpec::Ring(RingStateDomainSpec {
            header: header(1),
            components_per_step: vec![component(1)],
            capacity_steps: 2,
        }));
        let ring_intent = DomainStepIntent {
            domain: StateDomainId::new(1),
            expected_cursor: 0,
            target_cursor: 3,
            update: StateUpdateKind::RingAdvance {
                steps: 3,
                components_per_step: vec![declared_component(1)],
            },
        };
        ring.apply_intent(
            &ring_intent,
            InvocationTensorUpdateV2::RingAdvance {
                steps: vec![step(&[1.0]), step(&[2.0]), step(&[3.0])],
            },
        )
        .unwrap();
        assert_eq!(ring.absolute_cursor(), 3);
        assert_eq!(
            ring.read_chronological_segments()
                .unwrap()
                .iter()
                .flat_map(|segment| segment.components[0]
                    .tensor
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap())
                .collect::<Vec<_>>(),
            vec![2.0, 3.0]
        );
    }
}
