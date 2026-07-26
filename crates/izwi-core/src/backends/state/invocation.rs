//! Fixed physical backing for invocation-scoped non-paged state.
//!
//! Every component is allocated at its semantic maximum on the selected
//! Candle device. Logical shapes and cursors are metadata over those stable
//! buffers; updates copy into the existing storage and never replace it.

use std::sync::Arc;

use candle_core::{DType, Device, DeviceLocation, Tensor, Var};

use crate::backends::backend_kind_for_device;
use crate::error::{Error, Result};
use crate::kv::v2::{
    InferenceStateContract, InvocationStateCapacity, InvocationWorkspaceDomain,
    ResolvedNonPagedDomainPlan, ResolvedStatePlan, StateComponentId, StateDType, StateDomainId,
    StateDomainSpec, StateGroupId, StateScope, StateStorageFormat, TensorComponentSpec,
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

    pub(super) fn backing(&self, component: StateComponentId) -> Result<Tensor> {
        self.require_clean()?;
        Ok(self.component(component)?.storage.as_tensor().clone())
    }

    pub(super) fn install(
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

    pub(super) fn replace(
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

    pub(super) fn snapshot(&self) -> Result<InvocationTensorSnapshot> {
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

    pub(super) fn append(
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

    pub(super) fn ring_advance(
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

    pub(super) fn chronological_segments(
        &self,
    ) -> Result<Vec<InvocationTensorChronologicalSegment>> {
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

    pub(super) fn reset(&mut self) -> Result<()> {
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
        if expected_cursor != self.absolute_cursor {
            return Err(invalid("sequence invocation cursor is stale"));
        }
        if step_count == 0 || target_cursor <= expected_cursor {
            return Err(invalid(
                "sequence invocation update requires a non-empty cursor advance",
            ));
        }
        let step_count = u64::try_from(step_count)
            .map_err(|_| invalid("sequence invocation step count exceeds u64"))?;
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
        ShapeDimension, ShapeExtent, StateClock, StateDomainHeader, StateGroupSpec,
        StaticTensorDomainSpec, TensorRole, TensorStateDomainSpec, WorkspaceFormula,
        CURRENT_INFERENCE_STATE_ABI,
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
        assert!(arena.snapshot().is_err());
        assert!(arena.backing(StateComponentId::new(1)).is_err());
        assert!(arena
            .replace(1, 2, &[value(&[5.0, 6.0, 7.0, 8.0])])
            .is_err());

        arena.reset().unwrap();
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
}
