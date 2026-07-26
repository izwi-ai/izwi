//! Fixed physical backing for invocation-scoped static attention state.
//!
//! Static attention differs from paged decoder state: every layer installs one
//! immutable, bounded K/V memory and noncausal attention may revisit it many
//! times during the invocation. The arena therefore allocates stable
//! layer-major K/V buffers once, installs every layer as one atomic publication,
//! and keeps source/cursor metadata separate from the physical tensors.

use std::sync::Arc;

use candle_core::{CpuStorage, DType, Device, DeviceLocation, Layout, Storage, Tensor, Var};

use crate::backends::{backend_kind_for_device, BackendKind};
use crate::error::{Error, Result};
use crate::kv::v2::{
    DomainStepIntent, InferenceStateContract, InvocationStateCapacity, InvocationWorkspaceDomain,
    ResolvedNonPagedDomainPlan, ResolvedPlacement, ResolvedStatePlan, ResolvedStaticAttentionPlan,
    StateDType, StateDomainId, StateDomainSpec, StateGroupId, StateScope, StateStorageFormat,
    StateUpdateKind, StaticAttentionDomainSpec, StaticAttentionLayerSpec, TensorPhysicalLayout,
};

use super::StateBackendRegistry;

#[derive(Debug, Clone)]
pub(crate) struct StaticAttentionLayerValue {
    pub(crate) model_layer: u32,
    /// `[memory_tokens, kv_heads, key_head_dim]`.
    pub(crate) keys: Tensor,
    /// `[memory_tokens, kv_heads, value_head_dim]`.
    pub(crate) values: Tensor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StaticAttentionRaggedRow {
    pub(crate) query_start: u32,
    pub(crate) query_len: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StaticAttentionMetadata {
    pub(crate) source_identity: [u8; 32],
    /// The StaticInitialize cursor is the exact installed memory length.
    pub(crate) absolute_cursor: u64,
}

#[derive(Debug)]
struct StaticAttentionLayerBacking {
    semantic: StaticAttentionLayerSpec,
    keys: Var,
    values: Var,
}

#[derive(Debug)]
struct PendingStaticAttentionInstall {
    source_identity: [u8; 32],
    target_cursor: u64,
    memory_tokens: usize,
    written_layers: Vec<bool>,
    failed: bool,
}

/// One invocation-exclusive fixed arena for one StaticAttention workspace slot.
///
/// Pool ownership and lease generations live above this type. Mutation takes
/// `&mut self`, so an installed memory cannot race attention within one lease.
#[derive(Debug)]
pub(crate) struct InvocationStaticAttentionArena {
    plan: Arc<ResolvedStatePlan>,
    workspace_domain: InvocationWorkspaceDomain,
    domain: StateDomainId,
    group: StateGroupId,
    backend: BackendKind,
    device: Device,
    layers: Vec<StaticAttentionLayerBacking>,
    max_memory_tokens: usize,
    maximum_bytes: u64,
    source_identity: Option<[u8; 32]>,
    absolute_cursor: u64,
    initialized: bool,
    dirty: bool,
    pending_install: Option<PendingStaticAttentionInstall>,
}

impl InvocationStaticAttentionArena {
    pub(crate) fn new(
        contract: &InferenceStateContract,
        plan: Arc<ResolvedStatePlan>,
        workspace_domain: InvocationWorkspaceDomain,
        device: Device,
    ) -> Result<Self> {
        let backend = backend_kind_for_device(&device);
        let ordinal = device_ordinal(&device)?;
        let registry = StateBackendRegistry::new(backend, ordinal)?;
        Self::new_with_operation_registry(contract, plan, workspace_domain, device, &registry)
    }

    fn new_with_operation_registry(
        contract: &InferenceStateContract,
        plan: Arc<ResolvedStatePlan>,
        workspace_domain: InvocationWorkspaceDomain,
        device: Device,
        registry: &StateBackendRegistry,
    ) -> Result<Self> {
        let backend = backend_kind_for_device(&device);
        let ordinal = device_ordinal(&device)?;
        if plan.backend != backend || plan.device_ordinal != ordinal {
            return Err(invalid(
                "static-attention arena plan does not match its Candle device",
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
                "static-attention arena requires semantic-bounded typed state",
            ));
        };
        let canonical_state = contract
            .domains
            .iter()
            .find(|domain| domain.id() == state.id())
            .ok_or_else(|| {
                invalid("static-attention workspace domain is absent from its contract")
            })?;
        if canonical_state != state {
            return Err(invalid(
                "static-attention workspace state is not the canonical contract member",
            ));
        }
        let StateDomainSpec::StaticAttention(semantic) = state else {
            return Err(invalid(
                "static-attention arena requires a StaticAttention domain",
            ));
        };
        if semantic.header.scope != StateScope::Invocation
            || semantic.header.placement != *placement
        {
            return Err(invalid(
                "static-attention arena requires invocation-scoped matching placement",
            ));
        }

        let matches = plan
            .non_paged
            .iter()
            .filter(|resolved| resolved.domain() == semantic.header.id)
            .collect::<Vec<_>>();
        let [ResolvedNonPagedDomainPlan::StaticAttention(resolved)] = matches.as_slice() else {
            return Err(invalid(
                "static-attention arena requires one exact resolved StaticAttention domain",
            ));
        };

        plan.validate_against(contract, registry)?;
        if formula.maximum_bytes()? != resolved.maximum_bytes {
            return Err(invalid(
                "static-attention workspace formula does not exactly equal resolved backing",
            ));
        }
        validate_direct_plan(semantic, resolved, backend)?;
        let domain = semantic.header.id;
        let group = resolved.group;
        let resolved_maximum_bytes = resolved.maximum_bytes;

        let dtype = candle_dtype(resolved.storage)?;
        let max_memory_tokens = usize::try_from(semantic.max_memory_tokens)
            .map_err(|_| invalid("static-attention memory capacity exceeds usize"))?;
        let mut layers = Vec::with_capacity(resolved.layers.len());
        let mut allocated_bytes = 0_u64;
        for (physical_index, binding) in resolved.layers.iter().enumerate() {
            if usize::try_from(binding.physical_layer).ok() != Some(physical_index) {
                return Err(invalid(
                    "static-attention physical layers are not contiguous and layer-major",
                ));
            }
            let layer = semantic
                .layers
                .iter()
                .find(|layer| layer.model_layer == binding.model_layer)
                .ok_or_else(|| invalid("static-attention plan references an unknown model layer"))?
                .clone();
            let kv_heads = usize::try_from(layer.kv_heads)
                .map_err(|_| invalid("static-attention KV head count exceeds usize"))?;
            let key_head_dim = usize::try_from(layer.key_head_dim)
                .map_err(|_| invalid("static-attention key head dimension exceeds usize"))?;
            let value_head_dim = usize::try_from(layer.value_head_dim)
                .map_err(|_| invalid("static-attention value head dimension exceeds usize"))?;
            let keys = Var::zeros((max_memory_tokens, kv_heads, key_head_dim), dtype, &device)?;
            let values = Var::zeros(
                (max_memory_tokens, kv_heads, value_head_dim),
                dtype,
                &device,
            )?;
            let layer_bytes = tensor_bytes(keys.as_tensor())?
                .checked_add(tensor_bytes(values.as_tensor())?)
                .ok_or_else(|| invalid("static-attention layer byte count overflow"))?;
            allocated_bytes = allocated_bytes
                .checked_add(layer_bytes)
                .ok_or_else(|| invalid("static-attention allocation byte count overflow"))?;
            layers.push(StaticAttentionLayerBacking {
                semantic: layer,
                keys,
                values,
            });
        }
        if allocated_bytes != resolved_maximum_bytes {
            return Err(invalid(
                "static-attention allocation does not equal its resolved byte bound",
            ));
        }
        device.synchronize()?;

        Ok(Self {
            plan,
            workspace_domain,
            domain,
            group,
            backend,
            device,
            layers,
            max_memory_tokens,
            maximum_bytes: allocated_bytes,
            source_identity: None,
            absolute_cursor: 0,
            initialized: false,
            dirty: false,
            pending_install: None,
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

    pub(crate) const fn maximum_bytes(&self) -> u64 {
        self.maximum_bytes
    }

    pub(crate) const fn max_memory_tokens(&self) -> usize {
        self.max_memory_tokens
    }

    pub(crate) const fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub(crate) fn metadata(&self) -> Result<Option<StaticAttentionMetadata>> {
        self.require_clean()?;
        Ok(self
            .source_identity
            .map(|source_identity| StaticAttentionMetadata {
                source_identity,
                absolute_cursor: self.absolute_cursor,
            }))
    }

    /// Atomically publish a complete immutable K/V memory for every layer.
    ///
    /// All identity, shape, device, dtype, contiguity, alias, capacity, and
    /// common-length checks run before the first write. Once copying begins,
    /// any copy/synchronization failure leaves the arena dirty and therefore
    /// unobservable until a successful reset.
    pub(crate) fn install(
        &mut self,
        source_identity: [u8; 32],
        layers: Vec<StaticAttentionLayerValue>,
    ) -> Result<()> {
        self.install_with_copy_barrier(source_identity, layers, |_| Ok(()))
    }

    pub(crate) fn install_from_intent(
        &mut self,
        intent: &DomainStepIntent,
        layers: Vec<StaticAttentionLayerValue>,
    ) -> Result<()> {
        let pending = self.validate_begin_install(intent)?;
        let memory_tokens = self.validate_install_values(&layers)?;
        if pending.memory_tokens != memory_tokens {
            return Err(invalid(
                "static-attention installation does not match its authenticated intent",
            ));
        }
        self.begin_install(intent)?;
        for layer in layers {
            self.install_layer(layer)?;
        }
        self.commit_install()
    }

    pub(crate) fn begin_install(&mut self, intent: &DomainStepIntent) -> Result<()> {
        let pending = self.validate_begin_install(intent)?;
        self.dirty = true;
        self.pending_install = Some(pending);
        Ok(())
    }

    pub(crate) fn install_layer(&mut self, value: StaticAttentionLayerValue) -> Result<()> {
        self.install_layer_with_sync(value, |device| device.synchronize().map_err(Error::from))
    }

    fn install_layer_with_sync<F>(
        &mut self,
        value: StaticAttentionLayerValue,
        mut synchronize: F,
    ) -> Result<()>
    where
        F: FnMut(&Device) -> Result<()>,
    {
        let result = (|| {
            let pending = self.pending_install.as_ref().ok_or_else(|| {
                invalid("static-attention layer write has no authenticated pending install")
            })?;
            if pending.failed {
                return Err(invalid(
                    "static-attention pending install failed and must be reset",
                ));
            }
            let index = self
                .layers
                .binary_search_by_key(&value.model_layer, |layer| layer.semantic.model_layer)
                .map_err(|_| {
                    invalid("static-attention layer write references an unknown model layer")
                })?;
            if pending.written_layers[index] {
                return Err(invalid(
                    "static-attention pending install repeats one model layer",
                ));
            }
            let backing = &self.layers[index];
            let key_tokens = validate_install_tensor(
                &value.keys,
                &self.device,
                backing.keys.dtype(),
                backing.semantic.kv_heads,
                backing.semantic.key_head_dim,
                "key",
            )?;
            let value_tokens = validate_install_tensor(
                &value.values,
                &self.device,
                backing.values.dtype(),
                backing.semantic.kv_heads,
                backing.semantic.value_head_dim,
                "value",
            )?;
            if key_tokens != pending.memory_tokens || value_tokens != pending.memory_tokens {
                return Err(invalid(
                    "static-attention layer write does not match the authenticated memory length",
                ));
            }
            if self.layers.iter().any(|candidate| {
                shares_storage(&value.keys, candidate.keys.as_tensor())
                    || shares_storage(&value.keys, candidate.values.as_tensor())
                    || shares_storage(&value.values, candidate.keys.as_tensor())
                    || shares_storage(&value.values, candidate.values.as_tensor())
            }) {
                return Err(invalid(
                    "static-attention layer write source aliases arena storage",
                ));
            }
            Ok(index)
        })();
        let index = match result {
            Ok(index) => index,
            Err(error) => {
                self.mark_pending_failed();
                return Err(error);
            }
        };
        let copy = (|| -> Result<()> {
            let backing = &mut self.layers[index];
            backing.keys.zero_set().map_err(Error::from)?;
            backing.values.zero_set().map_err(Error::from)?;
            backing
                .keys
                .slice_set(&value.keys, 0, 0)
                .map_err(Error::from)?;
            backing
                .values
                .slice_set(&value.values, 0, 0)
                .map_err(Error::from)?;
            Ok(())
        })();
        if let Err(error) = copy {
            self.mark_pending_failed();
            return Err(error);
        }
        if let Err(error) = synchronize(&self.device) {
            self.mark_pending_failed();
            return Err(error);
        }
        self.pending_install
            .as_mut()
            .expect("pending install was validated before its layer copy")
            .written_layers[index] = true;
        Ok(())
    }

    pub(crate) fn commit_install(&mut self) -> Result<()> {
        let pending = self.pending_install.as_ref().ok_or_else(|| {
            invalid("static-attention commit has no authenticated pending install")
        })?;
        if pending.failed || pending.written_layers.iter().any(|written| !written) {
            return Err(invalid(
                "static-attention commit requires every layer exactly once",
            ));
        }
        if let Err(error) = self.device.synchronize() {
            self.mark_pending_failed();
            return Err(error.into());
        }
        let pending = self
            .pending_install
            .take()
            .expect("pending install was validated before its commit");
        self.source_identity = Some(pending.source_identity);
        self.absolute_cursor = pending.target_cursor;
        self.initialized = true;
        self.dirty = false;
        Ok(())
    }

    /// Attend noncausally over the installed memory without expanding GQA KV
    /// heads. Queries are flattened as `[total_queries, query_heads, key_dim]`;
    /// rows must form a canonical, non-empty partition of that first axis.
    pub(crate) fn attend(
        &self,
        model_layer: u32,
        queries: &Tensor,
        rows: &[StaticAttentionRaggedRow],
        softmax_scale: f32,
    ) -> Result<Tensor> {
        self.require_clean()?;
        if !self.initialized {
            return Err(invalid("static-attention memory is not initialized"));
        }
        let layer = self
            .layers
            .binary_search_by_key(&model_layer, |layer| layer.semantic.model_layer)
            .ok()
            .map(|index| &self.layers[index])
            .ok_or_else(|| invalid("static-attention request references an unknown model layer"))?;
        validate_queries(
            queries,
            rows,
            layer,
            &self.device,
            self.absolute_cursor,
            softmax_scale,
        )?;
        match self.backend {
            BackendKind::Cpu => self.attend_cpu(layer, queries, softmax_scale),
            BackendKind::Metal => self.attend_metal(layer, queries, softmax_scale),
            BackendKind::Cuda => self.attend_cuda(layer, queries, rows, softmax_scale),
        }
    }

    pub(crate) fn reset_for_reuse(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            if let Err(error) = layer.keys.zero_set() {
                self.dirty = true;
                return Err(error.into());
            }
            if let Err(error) = layer.values.zero_set() {
                self.dirty = true;
                return Err(error.into());
            }
        }
        if let Err(error) = self.device.synchronize() {
            self.dirty = true;
            return Err(error.into());
        }
        self.source_identity = None;
        self.absolute_cursor = 0;
        self.initialized = false;
        self.pending_install = None;
        self.dirty = false;
        Ok(())
    }

    pub(crate) fn prepare_completion(&mut self) -> Result<()> {
        if self.pending_install.is_some() {
            return Err(invalid(
                "static-attention pending install must commit or reset before completion",
            ));
        }
        self.require_clean()?;
        if let Err(error) = self.device.synchronize() {
            self.dirty = true;
            return Err(error.into());
        }
        Ok(())
    }

    fn install_with_copy_barrier<F>(
        &mut self,
        source_identity: [u8; 32],
        layers: Vec<StaticAttentionLayerValue>,
        mut after_layer_copy: F,
    ) -> Result<()>
    where
        F: FnMut(usize) -> Result<()>,
    {
        self.require_clean()?;
        if self.initialized || self.pending_install.is_some() {
            return Err(invalid(
                "static-attention memory is already initialized or pending; reset before reinstall",
            ));
        }
        if source_identity.iter().all(|byte| *byte == 0) {
            return Err(invalid(
                "static-attention installation requires a non-zero source identity",
            ));
        }
        let memory_tokens = self.validate_install_values(&layers)?;
        let intent = DomainStepIntent {
            domain: self.domain,
            expected_cursor: 0,
            target_cursor: u64::try_from(memory_tokens)
                .map_err(|_| invalid("static-attention memory length exceeds u64"))?,
            update: StateUpdateKind::StaticInitialize {
                source_identity,
                components: vec![],
            },
        };
        self.begin_install(&intent)?;
        for (index, layer) in layers.into_iter().enumerate() {
            self.install_layer(layer)?;
            if let Err(error) = after_layer_copy(index) {
                self.mark_pending_failed();
                return Err(error);
            }
        }
        self.commit_install()
    }

    fn validate_begin_install(
        &self,
        intent: &DomainStepIntent,
    ) -> Result<PendingStaticAttentionInstall> {
        self.require_clean()?;
        if self.initialized || self.pending_install.is_some() {
            return Err(invalid(
                "static-attention memory is already initialized or pending; reset before reinstall",
            ));
        }
        let StateUpdateKind::StaticInitialize {
            source_identity,
            components,
        } = &intent.update
        else {
            return Err(invalid(
                "static-attention installation requires a StaticInitialize intent",
            ));
        };
        let memory_tokens = usize::try_from(intent.target_cursor)
            .map_err(|_| invalid("static-attention target cursor exceeds usize"))?;
        if intent.domain != self.domain
            || intent.expected_cursor != 0
            || memory_tokens == 0
            || memory_tokens > self.max_memory_tokens
            || source_identity.iter().all(|byte| *byte == 0)
            || !components.is_empty()
        {
            return Err(invalid(
                "static-attention installation does not match its authenticated intent",
            ));
        }
        Ok(PendingStaticAttentionInstall {
            source_identity: *source_identity,
            target_cursor: intent.target_cursor,
            memory_tokens,
            written_layers: vec![false; self.layers.len()],
            failed: false,
        })
    }

    fn mark_pending_failed(&mut self) {
        if let Some(pending) = self.pending_install.as_mut() {
            self.dirty = true;
            pending.failed = true;
        }
    }

    fn validate_install_values(&self, values: &[StaticAttentionLayerValue]) -> Result<usize> {
        if values.len() != self.layers.len() {
            return Err(invalid(
                "static-attention installation must cover every layer",
            ));
        }
        let mut common_memory_tokens = None;
        for (backing, value) in self.layers.iter().zip(values) {
            if value.model_layer != backing.semantic.model_layer {
                return Err(invalid(
                    "static-attention layers are not in canonical model-layer order",
                ));
            }
            let memory_tokens = validate_install_tensor(
                &value.keys,
                &self.device,
                backing.keys.dtype(),
                backing.semantic.kv_heads,
                backing.semantic.key_head_dim,
                "key",
            )?;
            let value_tokens = validate_install_tensor(
                &value.values,
                &self.device,
                backing.values.dtype(),
                backing.semantic.kv_heads,
                backing.semantic.value_head_dim,
                "value",
            )?;
            if memory_tokens != value_tokens
                || common_memory_tokens.is_some_and(|common| common != memory_tokens)
            {
                return Err(invalid(
                    "static-attention K/V layers must share one exact sequence length",
                ));
            }
            if memory_tokens == 0 || memory_tokens > self.max_memory_tokens {
                return Err(invalid(
                    "static-attention memory length is zero or exceeds capacity",
                ));
            }
            if self.layers.iter().any(|candidate| {
                shares_storage(&value.keys, candidate.keys.as_tensor())
                    || shares_storage(&value.keys, candidate.values.as_tensor())
                    || shares_storage(&value.values, candidate.keys.as_tensor())
                    || shares_storage(&value.values, candidate.values.as_tensor())
            }) {
                return Err(invalid(
                    "static-attention installation source aliases arena storage",
                ));
            }
            common_memory_tokens = Some(memory_tokens);
        }
        common_memory_tokens
            .ok_or_else(|| invalid("static-attention installation has no layer values"))
    }

    fn attend_cpu(
        &self,
        layer: &StaticAttentionLayerBacking,
        queries: &Tensor,
        softmax_scale: f32,
    ) -> Result<Tensor> {
        let memory_tokens = usize::try_from(self.absolute_cursor)
            .map_err(|_| invalid("static-attention cursor exceeds usize"))?;
        let query_heads = queries.dims()[1];
        let key_dim = usize::try_from(layer.semantic.key_head_dim)
            .map_err(|_| invalid("static-attention key dimension exceeds usize"))?;
        let value_dim = usize::try_from(layer.semantic.value_head_dim)
            .map_err(|_| invalid("static-attention value dimension exceeds usize"))?;
        let kv_heads = usize::try_from(layer.semantic.kv_heads)
            .map_err(|_| invalid("static-attention KV head count exceeds usize"))?;
        let (key_storage, key_layout) = layer.keys.as_tensor().storage_and_layout();
        let (value_storage, value_layout) = layer.values.as_tensor().storage_and_layout();
        let (query_storage, query_layout) = queries.storage_and_layout();
        let key_start = contiguous_range(key_layout, "static-attention CPU keys")?.0;
        let value_start = contiguous_range(value_layout, "static-attention CPU values")?.0;
        let query_start = contiguous_range(query_layout, "static-attention CPU queries")?.0;

        macro_rules! attend_typed {
            ($keys:expr, $values:expr, $queries:expr) => {
                online_static_attention(
                    $keys,
                    $values,
                    $queries,
                    key_start,
                    value_start,
                    query_start,
                    memory_tokens,
                    kv_heads,
                    key_dim,
                    value_dim,
                    query_heads,
                    queries.dims()[0],
                    softmax_scale,
                )
            };
        }

        let output = match (&*key_storage, &*value_storage, &*query_storage) {
            (
                Storage::Cpu(CpuStorage::F32(keys)),
                Storage::Cpu(CpuStorage::F32(values)),
                Storage::Cpu(CpuStorage::F32(queries)),
            ) => attend_typed!(keys, values, queries),
            (
                Storage::Cpu(CpuStorage::F16(keys)),
                Storage::Cpu(CpuStorage::F16(values)),
                Storage::Cpu(CpuStorage::F16(queries)),
            ) => attend_typed!(keys, values, queries),
            (
                Storage::Cpu(CpuStorage::BF16(keys)),
                Storage::Cpu(CpuStorage::BF16(values)),
                Storage::Cpu(CpuStorage::BF16(queries)),
            ) => attend_typed!(keys, values, queries),
            _ => Err(invalid(
                "static-attention CPU storage dtype does not match queries",
            )),
        }?;

        Tensor::from_vec(
            output,
            (queries.dims()[0], query_heads, value_dim),
            &Device::Cpu,
        )?
        .to_dtype(queries.dtype())
        .map_err(Error::from)
    }

    #[cfg(feature = "metal")]
    fn attend_metal(
        &self,
        layer: &StaticAttentionLayerBacking,
        queries: &Tensor,
        softmax_scale: f32,
    ) -> Result<Tensor> {
        let batch_size = queries.dims()[0];
        let page_tokens = self.max_memory_tokens;
        let memory_tokens = u32::try_from(self.absolute_cursor)
            .map_err(|_| invalid("static-attention Metal cursor exceeds u32"))?;
        let mut metadata = Vec::with_capacity(batch_size * 3);
        metadata.extend(std::iter::repeat_n(memory_tokens, batch_size));
        metadata.extend(std::iter::repeat_n(0_u32, batch_size));
        metadata.extend(std::iter::repeat_n(0_u32, batch_size));
        let keys = layer.keys.as_tensor().reshape((
            1,
            page_tokens,
            usize::try_from(layer.semantic.kv_heads)
                .map_err(|_| invalid("static-attention KV heads exceed usize"))?,
            usize::try_from(layer.semantic.key_head_dim)
                .map_err(|_| invalid("static-attention key dimension exceeds usize"))?,
        ))?;
        let values = layer.values.as_tensor().reshape((
            1,
            page_tokens,
            usize::try_from(layer.semantic.kv_heads)
                .map_err(|_| invalid("static-attention KV heads exceed usize"))?,
            usize::try_from(layer.semantic.value_head_dim)
                .map_err(|_| invalid("static-attention value dimension exceeds usize"))?,
        ))?;
        crate::kernels::metal::paged_decode_attention(
            queries,
            &keys,
            &values,
            metadata,
            batch_size,
            usize::try_from(layer.semantic.query_heads)
                .map_err(|_| invalid("static-attention query heads exceed usize"))?,
            usize::try_from(layer.semantic.kv_heads)
                .map_err(|_| invalid("static-attention KV heads exceed usize"))?,
            page_tokens,
            1,
            usize::try_from(layer.semantic.key_head_dim)
                .map_err(|_| invalid("static-attention key dimension exceeds usize"))?,
            usize::try_from(layer.semantic.value_head_dim)
                .map_err(|_| invalid("static-attention value dimension exceeds usize"))?,
            softmax_scale,
        )
        .map_err(Error::from)
    }

    #[cfg(not(feature = "metal"))]
    fn attend_metal(
        &self,
        _layer: &StaticAttentionLayerBacking,
        _queries: &Tensor,
        _softmax_scale: f32,
    ) -> Result<Tensor> {
        Err(invalid("static-attention Metal operation is not compiled"))
    }

    #[cfg(feature = "flash-attn")]
    fn attend_cuda(
        &self,
        layer: &StaticAttentionLayerBacking,
        queries: &Tensor,
        rows: &[StaticAttentionRaggedRow],
        softmax_scale: f32,
    ) -> Result<Tensor> {
        let batch_size = rows.len();
        let mut seqlens_q = Vec::with_capacity(batch_size + 1);
        let mut seqlens_k = Vec::with_capacity(batch_size + 1);
        seqlens_q.push(0_u32);
        seqlens_k.push(0_u32);
        let memory_tokens = u32::try_from(self.absolute_cursor)
            .map_err(|_| invalid("static-attention CUDA cursor exceeds u32"))?;
        let mut cumulative_q = 0_u32;
        let mut cumulative_k = 0_u32;
        let mut max_query_len = 0_usize;
        for row in rows {
            cumulative_q = cumulative_q
                .checked_add(row.query_len)
                .ok_or_else(|| invalid("static-attention CUDA query cursor overflow"))?;
            cumulative_k = cumulative_k
                .checked_add(memory_tokens)
                .ok_or_else(|| invalid("static-attention CUDA memory cursor overflow"))?;
            seqlens_q.push(cumulative_q);
            seqlens_k.push(cumulative_k);
            max_query_len = max_query_len.max(
                usize::try_from(row.query_len)
                    .map_err(|_| invalid("static-attention query length exceeds usize"))?,
            );
        }
        let seqlens_q = Tensor::from_vec(seqlens_q, batch_size + 1, &self.device)?;
        let seqlens_k = Tensor::from_vec(seqlens_k, batch_size + 1, &self.device)?;
        let block_table = Tensor::zeros((batch_size, 1), DType::U32, &self.device)?;
        let kv_heads = usize::try_from(layer.semantic.kv_heads)
            .map_err(|_| invalid("static-attention KV heads exceed usize"))?;
        let head_dim = usize::try_from(layer.semantic.key_head_dim)
            .map_err(|_| invalid("static-attention head dimension exceeds usize"))?;
        let keys =
            layer
                .keys
                .as_tensor()
                .reshape((1, self.max_memory_tokens, kv_heads, head_dim))?;
        let values =
            layer
                .values
                .as_tensor()
                .reshape((1, self.max_memory_tokens, kv_heads, head_dim))?;
        candle_flash_attn::flash_attn_varlen_paged_windowed(
            queries,
            &keys,
            &values,
            &seqlens_q,
            &seqlens_k,
            &block_table,
            None,
            max_query_len,
            usize::try_from(memory_tokens)
                .map_err(|_| invalid("static-attention CUDA memory length exceeds usize"))?,
            softmax_scale,
            None,
            None,
            self.max_memory_tokens,
            None,
        )
        .map_err(Error::from)
    }

    #[cfg(not(feature = "flash-attn"))]
    fn attend_cuda(
        &self,
        _layer: &StaticAttentionLayerBacking,
        _queries: &Tensor,
        _rows: &[StaticAttentionRaggedRow],
        _softmax_scale: f32,
    ) -> Result<Tensor> {
        Err(invalid(
            "static-attention CUDA paged flash-attention is not compiled",
        ))
    }

    #[cfg(test)]
    fn layer_backing_ids(
        &self,
        model_layer: u32,
    ) -> Result<(candle_core::TensorId, candle_core::TensorId)> {
        let layer = self
            .layers
            .iter()
            .find(|layer| layer.semantic.model_layer == model_layer)
            .ok_or_else(|| invalid("static-attention test requested an unknown layer"))?;
        Ok((layer.keys.as_tensor().id(), layer.values.as_tensor().id()))
    }

    #[cfg(test)]
    fn layer_backing_values(&self, model_layer: u32) -> Result<(Vec<f32>, Vec<f32>)> {
        self.require_clean()?;
        let layer = self
            .layers
            .iter()
            .find(|layer| layer.semantic.model_layer == model_layer)
            .ok_or_else(|| invalid("static-attention test requested an unknown layer"))?;
        Ok((
            layer.keys.as_tensor().flatten_all()?.to_vec1::<f32>()?,
            layer.values.as_tensor().flatten_all()?.to_vec1::<f32>()?,
        ))
    }

    #[cfg(test)]
    fn layer_key_backing(&self, model_layer: u32) -> Result<Tensor> {
        Ok(self
            .layers
            .iter()
            .find(|layer| layer.semantic.model_layer == model_layer)
            .ok_or_else(|| invalid("static-attention test requested an unknown layer"))?
            .keys
            .as_tensor()
            .clone())
    }

    fn require_clean(&self) -> Result<()> {
        if self.dirty {
            return Err(invalid(
                "static-attention arena is dirty and must be reset before reuse",
            ));
        }
        Ok(())
    }
}

pub(super) fn static_plan_is_supported(
    resolved: &ResolvedStaticAttentionPlan,
    semantic: &StaticAttentionDomainSpec,
    backend: BackendKind,
) -> bool {
    if resolved.layout != TensorPhysicalLayout::ContiguousRowMajor
        || resolved.alignment_bytes != 1
        || resolved.operations != super::static_attention_operations()
        || resolved.layers.len() != semantic.layers.len()
    {
        return false;
    }
    match backend {
        BackendKind::Cpu => {
            matches!(
                resolved.storage.dtype(),
                StateDType::F32 | StateDType::F16 | StateDType::Bf16
            ) && matches!(
                resolved.placement,
                ResolvedPlacement::BackendLocal | ResolvedPlacement::Host
            )
        }
        BackendKind::Metal => {
            cfg!(feature = "metal")
                && resolved.placement == ResolvedPlacement::BackendLocal
                && matches!(resolved.storage.dtype(), StateDType::F32 | StateDType::F16)
                && u32::try_from(semantic.max_memory_tokens).is_ok()
                && semantic
                    .layers
                    .iter()
                    .all(|layer| layer.key_head_dim <= 512 && layer.value_head_dim <= 512)
        }
        BackendKind::Cuda => {
            cfg!(feature = "flash-attn")
                && resolved.placement == ResolvedPlacement::BackendLocal
                && matches!(resolved.storage.dtype(), StateDType::F16 | StateDType::Bf16)
                && u32::try_from(semantic.max_memory_tokens).is_ok()
                && semantic.max_memory_tokens % 32 == 0
                && semantic.layers.iter().all(|layer| {
                    layer.key_head_dim == layer.value_head_dim
                        && layer.key_head_dim % 8 == 0
                        && layer.key_head_dim <= 512
                })
        }
    }
}

fn validate_direct_plan(
    semantic: &StaticAttentionDomainSpec,
    resolved: &ResolvedStaticAttentionPlan,
    backend: BackendKind,
) -> Result<()> {
    if !static_plan_is_supported(resolved, semantic, backend) {
        return Err(invalid(
            "resolved static-attention plan has no exact direct backend implementation",
        ));
    }
    Ok(())
}

fn validate_install_tensor(
    tensor: &Tensor,
    device: &Device,
    dtype: DType,
    kv_heads: u32,
    head_dim: u32,
    label: &str,
) -> Result<usize> {
    let expected_suffix = [
        usize::try_from(kv_heads)
            .map_err(|_| invalid("static-attention KV head count exceeds usize"))?,
        usize::try_from(head_dim)
            .map_err(|_| invalid("static-attention head dimension exceeds usize"))?,
    ];
    if !device.same_device(tensor.device())
        || tensor.dtype() != dtype
        || tensor.rank() != 3
        || tensor.dims()[1..] != expected_suffix
    {
        return Err(invalid(format!(
            "static-attention {label} source has incompatible device, dtype, or shape"
        )));
    }
    if !tensor.is_contiguous() {
        return Err(invalid(format!(
            "static-attention {label} source must be contiguous"
        )));
    }
    Ok(tensor.dims()[0])
}

fn validate_queries(
    queries: &Tensor,
    rows: &[StaticAttentionRaggedRow],
    layer: &StaticAttentionLayerBacking,
    device: &Device,
    memory_tokens: u64,
    softmax_scale: f32,
) -> Result<()> {
    let expected = [
        usize::try_from(layer.semantic.query_heads)
            .map_err(|_| invalid("static-attention query head count exceeds usize"))?,
        usize::try_from(layer.semantic.key_head_dim)
            .map_err(|_| invalid("static-attention query head dimension exceeds usize"))?,
    ];
    if !device.same_device(queries.device())
        || queries.dtype() != layer.keys.dtype()
        || queries.rank() != 3
        || queries.dims()[1..] != expected
        || !queries.is_contiguous()
    {
        return Err(invalid(
            "static-attention queries have incompatible device, dtype, shape, or layout",
        ));
    }
    if memory_tokens == 0 || !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(invalid(
            "static-attention requires installed memory and a finite positive scale",
        ));
    }
    let mut next_query = 0_u32;
    for (index, row) in rows.iter().enumerate() {
        if row.query_start != next_query || row.query_len == 0 {
            return Err(invalid(format!(
                "static-attention ragged row {index} is not a canonical non-empty range"
            )));
        }
        next_query = next_query
            .checked_add(row.query_len)
            .ok_or_else(|| invalid("static-attention ragged query range overflow"))?;
    }
    if rows.is_empty()
        || usize::try_from(next_query)
            .map_err(|_| invalid("static-attention query range exceeds usize"))?
            != queries.dims()[0]
    {
        return Err(invalid(
            "static-attention ragged rows must cover every query exactly once",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn online_static_attention<T>(
    keys: &[T],
    values: &[T],
    queries: &[T],
    key_start: usize,
    value_start: usize,
    query_start: usize,
    memory_tokens: usize,
    kv_heads: usize,
    key_dim: usize,
    value_dim: usize,
    query_heads: usize,
    total_queries: usize,
    softmax_scale: f32,
) -> Result<Vec<f32>>
where
    T: Copy,
    f32: From<T>,
{
    let required_query_elements = total_queries
        .checked_mul(query_heads)
        .and_then(|elements| elements.checked_mul(key_dim))
        .and_then(|elements| elements.checked_add(query_start))
        .ok_or_else(|| invalid("static-attention query element count overflow"))?;
    if required_query_elements > queries.len() {
        return Err(invalid(
            "static-attention query view exceeds its physical storage",
        ));
    }
    let queries_per_kv_head = query_heads / kv_heads;
    let mut output = vec![0.0_f32; total_queries * query_heads * value_dim];

    for query in 0..total_queries {
        for query_head in 0..query_heads {
            let kv_head = query_head / queries_per_kv_head;
            let query_offset = query_start + (query * query_heads + query_head) * key_dim;
            let mut running_max = f32::NEG_INFINITY;
            let mut running_sum = 0.0_f32;
            let mut accumulator = vec![0.0_f32; value_dim];
            for token in 0..memory_tokens {
                let key_offset = key_start + (token * kv_heads + kv_head) * key_dim;
                let value_offset = value_start + (token * kv_heads + kv_head) * value_dim;
                let mut score = 0.0_f32;
                for dim in 0..key_dim {
                    score +=
                        f32::from(queries[query_offset + dim]) * f32::from(keys[key_offset + dim]);
                }
                score *= softmax_scale;
                let next_max = running_max.max(score);
                let previous_weight = (running_max - next_max).exp();
                let token_weight = (score - next_max).exp();
                running_sum = running_sum * previous_weight + token_weight;
                for dim in 0..value_dim {
                    accumulator[dim] = accumulator[dim] * previous_weight
                        + f32::from(values[value_offset + dim]) * token_weight;
                }
                running_max = next_max;
            }
            let output_offset = (query * query_heads + query_head) * value_dim;
            for dim in 0..value_dim {
                output[output_offset + dim] = accumulator[dim] / running_sum;
            }
        }
    }
    Ok(output)
}

fn tensor_bytes(tensor: &Tensor) -> Result<u64> {
    let bytes = tensor
        .elem_count()
        .checked_mul(tensor.dtype().size_in_bytes())
        .ok_or_else(|| invalid("static-attention tensor byte count overflow"))?;
    u64::try_from(bytes).map_err(|_| invalid("static-attention tensor bytes exceed u64"))
}

fn candle_dtype(storage: StateStorageFormat) -> Result<DType> {
    match storage.dtype() {
        StateDType::F32 => Ok(DType::F32),
        StateDType::F16 => Ok(DType::F16),
        StateDType::Bf16 => Ok(DType::BF16),
        StateDType::I8 | StateDType::Q4 => Err(invalid(
            "quantized static-attention state requires an explicit packing ABI",
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

fn shares_storage(left: &Tensor, right: &Tensor) -> bool {
    fn address(tensor: &Tensor) -> *const () {
        let (storage, _) = tensor.storage_and_layout();
        std::ptr::from_ref(&*storage).cast()
    }

    address(left) == address(right)
}

fn contiguous_range(layout: &Layout, operation: &str) -> Result<(usize, usize)> {
    layout
        .contiguous_offsets()
        .ok_or_else(|| invalid(format!("{operation} requires contiguous storage")))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::v2::{
        CheckpointPolicy, KeyEncoding, PlacementPolicy, PrefixPolicy, ResolvedStaticAttentionPlan,
        StateClock, StateDomainHeader, StateLayerBinding, StateScope, WorkspaceFormula,
        CURRENT_INFERENCE_STATE_ABI,
    };

    fn header() -> StateDomainHeader {
        StateDomainHeader {
            id: StateDomainId::new(1),
            scope: StateScope::Invocation,
            clock: StateClock::EncoderTokens,
            placement: PlacementPolicy::BackendLocal,
            prefix: PrefixPolicy::Disabled,
            checkpoint: CheckpointPolicy::None,
        }
    }

    fn layer(model_layer: u32, query_heads: u32, kv_heads: u32) -> StaticAttentionLayerSpec {
        StaticAttentionLayerSpec {
            model_layer,
            query_heads,
            kv_heads,
            key_head_dim: 2,
            value_head_dim: 2,
            key_encoding: KeyEncoding::Raw,
        }
    }

    fn arena(
        semantic_layers: Vec<StaticAttentionLayerSpec>,
        max_memory_tokens: u64,
    ) -> InvocationStaticAttentionArena {
        let domain = StateDomainSpec::StaticAttention(StaticAttentionDomainSpec {
            header: header(),
            layers: semantic_layers.clone(),
            max_memory_tokens,
            accepted_dtypes: vec![StateDType::F32],
        });
        let contract = InferenceStateContract {
            abi: CURRENT_INFERENCE_STATE_ABI,
            domains: vec![domain.clone()],
            groups: vec![crate::kv::v2::StateGroupSpec {
                id: StateGroupId::new(1),
                domains: vec![domain.id()],
                prefix_shareable: false,
            }],
        };
        let elements = semantic_layers
            .iter()
            .map(|layer| {
                u64::from(layer.kv_heads)
                    * u64::from(layer.key_head_dim + layer.value_head_dim)
                    * max_memory_tokens
            })
            .sum::<u64>();
        let resolved = ResolvedNonPagedDomainPlan::StaticAttention(ResolvedStaticAttentionPlan {
            group: StateGroupId::new(1),
            domain: domain.id(),
            placement: ResolvedPlacement::BackendLocal,
            layers: semantic_layers
                .iter()
                .enumerate()
                .map(|(physical_layer, layer)| StateLayerBinding {
                    model_layer: layer.model_layer,
                    physical_layer: u32::try_from(physical_layer).unwrap(),
                })
                .collect(),
            storage: StateStorageFormat::Dense {
                dtype: StateDType::F32,
            },
            layout: TensorPhysicalLayout::ContiguousRowMajor,
            alignment_bytes: 1,
            maximum_bytes: elements * 4,
            operations: super::super::static_attention_operations(),
        });
        let registry = StateBackendRegistry::new(BackendKind::Cpu, None).unwrap();
        let plan = Arc::new(
            ResolvedStatePlan::build(
                BackendKind::Cpu,
                None,
                &contract,
                vec![],
                vec![resolved],
                &registry,
            )
            .unwrap(),
        );
        InvocationStaticAttentionArena::new(
            &contract,
            plan,
            InvocationWorkspaceDomain::State {
                state: domain,
                capacity: InvocationStateCapacity::SemanticBounded,
                placement: PlacementPolicy::BackendLocal,
                formula: WorkspaceFormula {
                    fixed_bytes: elements * 4,
                    dimensions: vec![],
                    terms: vec![],
                },
            },
            Device::Cpu,
        )
        .unwrap()
    }

    fn values(
        model_layer: u32,
        keys: &[f32],
        values: &[f32],
        memory_tokens: usize,
        kv_heads: usize,
    ) -> StaticAttentionLayerValue {
        StaticAttentionLayerValue {
            model_layer,
            keys: Tensor::from_slice(keys, (memory_tokens, kv_heads, 2), &Device::Cpu).unwrap(),
            values: Tensor::from_slice(values, (memory_tokens, kv_heads, 2), &Device::Cpu).unwrap(),
        }
    }

    fn install_intent(target_cursor: u64) -> DomainStepIntent {
        DomainStepIntent {
            domain: StateDomainId::new(1),
            expected_cursor: 0,
            target_cursor,
            update: StateUpdateKind::StaticInitialize {
                source_identity: [8; 32],
                components: vec![],
            },
        }
    }

    fn dense_reference(
        queries: &[f32],
        keys: &[f32],
        values: &[f32],
        total_queries: usize,
        query_heads: usize,
        kv_heads: usize,
        memory_tokens: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0; total_queries * query_heads * 2];
        let group = query_heads / kv_heads;
        for query in 0..total_queries {
            for query_head in 0..query_heads {
                let kv_head = query_head / group;
                let mut scores = Vec::with_capacity(memory_tokens);
                for token in 0..memory_tokens {
                    let q = (query * query_heads + query_head) * 2;
                    let k = (token * kv_heads + kv_head) * 2;
                    scores.push((queries[q] * keys[k] + queries[q + 1] * keys[k + 1]) * scale);
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights = scores
                    .iter()
                    .map(|score| (*score - max).exp())
                    .collect::<Vec<_>>();
                let denominator = weights.iter().sum::<f32>();
                for token in 0..memory_tokens {
                    let v = (token * kv_heads + kv_head) * 2;
                    let out = (query * query_heads + query_head) * 2;
                    output[out] += weights[token] / denominator * values[v];
                    output[out + 1] += weights[token] / denominator * values[v + 1];
                }
            }
        }
        output
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "value {index}: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn staged_install_writes_one_layer_at_a_time_and_publishes_only_on_commit() {
        let mut arena = arena(vec![layer(0, 4, 2), layer(3, 4, 2)], 3);
        arena.begin_install(&install_intent(2)).unwrap();
        assert!(arena.metadata().is_err());
        arena
            .install_layer(values(
                0,
                &[1.0, 0.0, 0.0, 1.0, 0.5, 0.5, -0.5, 0.5],
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                2,
                2,
            ))
            .unwrap();
        arena
            .install_layer(values(
                3,
                &[0.0, 1.0, 1.0, 0.0, 0.25, 0.75, 0.75, 0.25],
                &[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                2,
                2,
            ))
            .unwrap();
        assert!(arena.metadata().is_err());
        arena.commit_install().unwrap();
        assert_eq!(
            arena.metadata().unwrap(),
            Some(StaticAttentionMetadata {
                source_identity: [8; 32],
                absolute_cursor: 2,
            })
        );
        let queries = Tensor::from_slice(
            &[1.0_f32, 0.0, 0.5, 0.5, 0.0, 1.0, -0.5, 0.5],
            (1, 4, 2),
            &Device::Cpu,
        )
        .unwrap();
        assert_eq!(
            arena
                .attend(
                    0,
                    &queries,
                    &[StaticAttentionRaggedRow {
                        query_start: 0,
                        query_len: 1,
                    }],
                    1.0,
                )
                .unwrap()
                .dims(),
            &[1, 4, 2]
        );
    }

    #[test]
    fn staged_install_failures_block_completion_until_reset() {
        let mut arena = arena(vec![layer(0, 2, 2), layer(3, 2, 2)], 3);
        let first = values(0, &[1.0; 8], &[2.0; 8], 2, 2);
        arena.begin_install(&install_intent(2)).unwrap();
        arena.install_layer(first.clone()).unwrap();
        assert!(arena.install_layer(first).is_err());
        assert!(arena.commit_install().is_err());
        assert!(arena.prepare_completion().is_err());
        arena.reset_for_reuse().unwrap();
        assert_eq!(arena.metadata().unwrap(), None);

        arena.begin_install(&install_intent(2)).unwrap();
        arena
            .install_layer(values(0, &[1.0; 8], &[2.0; 8], 2, 2))
            .unwrap();
        assert!(arena.commit_install().is_err());
        assert!(arena.prepare_completion().is_err());
        arena.reset_for_reuse().unwrap();

        arena.begin_install(&install_intent(2)).unwrap();
        assert!(arena
            .install_layer(values(0, &[1.0; 4], &[2.0; 4], 1, 2))
            .is_err());
        assert!(arena.prepare_completion().is_err());
        arena.reset_for_reuse().unwrap();

        arena.begin_install(&install_intent(2)).unwrap();
        let error = arena
            .install_layer_with_sync(values(0, &[1.0; 8], &[2.0; 8], 2, 2), |_| {
                Err(invalid("injected per-layer synchronization failure"))
            })
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("injected per-layer synchronization failure"));
        assert!(arena.is_dirty());
        let pending = arena.pending_install.as_ref().unwrap();
        assert!(pending.failed);
        assert!(!pending.written_layers[0]);
        assert!(arena.commit_install().is_err());
        assert!(arena.prepare_completion().is_err());
        arena.reset_for_reuse().unwrap();
        assert!(arena.pending_install.is_none());

        arena.begin_install(&install_intent(2)).unwrap();
        assert!(arena
            .install_layer(values(99, &[1.0; 8], &[2.0; 8], 2, 2))
            .is_err());
        assert!(arena.prepare_completion().is_err());
        arena.reset_for_reuse().unwrap();
        arena
            .install(
                [9; 32],
                vec![
                    values(0, &[1.0; 8], &[2.0; 8], 2, 2),
                    values(3, &[3.0; 8], &[4.0; 8], 2, 2),
                ],
            )
            .unwrap();
        assert_eq!(arena.metadata().unwrap().unwrap().absolute_cursor, 2);
    }

    #[test]
    fn cpu_attention_matches_dense_noncausal_reference() {
        let mut arena = arena(vec![layer(4, 2, 2)], 4);
        let keys = [1.0, 0.0, 0.0, 1.0, 0.5, 0.5, -0.5, 0.5, 0.0, 1.0, 1.0, 0.0];
        let memory_values = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        arena
            .install([9; 32], vec![values(4, &keys, &memory_values, 3, 2)])
            .unwrap();
        let queries = [
            1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 1.0, -1.0, -0.25, 0.75, 0.5, 0.25,
        ];
        let query_tensor = Tensor::from_slice(&queries, (3, 2, 2), &Device::Cpu).unwrap();
        let scale = 1.0 / 2.0_f32.sqrt();
        let output = arena
            .attend(
                4,
                &query_tensor,
                &[
                    StaticAttentionRaggedRow {
                        query_start: 0,
                        query_len: 2,
                    },
                    StaticAttentionRaggedRow {
                        query_start: 2,
                        query_len: 1,
                    },
                ],
                scale,
            )
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected = dense_reference(&queries, &keys, &memory_values, 3, 2, 2, 3, scale);
        assert_close(&output, &expected);
    }

    #[test]
    fn cpu_attention_maps_grouped_query_heads_without_repeat_kv() {
        let mut arena = arena(vec![layer(0, 4, 2)], 3);
        let keys = [1.0, 0.0, 0.0, 1.0, 0.25, 0.75, 0.75, 0.25];
        let memory_values = [1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0];
        arena
            .install([1; 32], vec![values(0, &keys, &memory_values, 2, 2)])
            .unwrap();
        let queries = [1.0, 0.0, 0.5, 0.5, 0.0, 1.0, -0.5, 0.5];
        let query_tensor = Tensor::from_slice(&queries, (1, 4, 2), &Device::Cpu).unwrap();
        let output = arena
            .attend(
                0,
                &query_tensor,
                &[StaticAttentionRaggedRow {
                    query_start: 0,
                    query_len: 1,
                }],
                1.0,
            )
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let expected = dense_reference(&queries, &keys, &memory_values, 1, 4, 2, 2, 1.0);
        assert_close(&output, &expected);
    }

    #[test]
    fn ragged_rows_must_cover_queries_canonically() {
        let mut arena = arena(vec![layer(0, 2, 2)], 2);
        arena
            .install(
                [2; 32],
                vec![values(
                    0,
                    &[1.0, 0.0, 0.0, 1.0],
                    &[1.0, 2.0, 3.0, 4.0],
                    1,
                    2,
                )],
            )
            .unwrap();
        let queries = Tensor::zeros((3, 2, 2), DType::F32, &Device::Cpu).unwrap();
        assert!(arena
            .attend(
                0,
                &queries,
                &[StaticAttentionRaggedRow {
                    query_start: 1,
                    query_len: 3,
                }],
                1.0,
            )
            .is_err());
        assert!(arena
            .attend(
                0,
                &queries,
                &[
                    StaticAttentionRaggedRow {
                        query_start: 0,
                        query_len: 1,
                    },
                    StaticAttentionRaggedRow {
                        query_start: 1,
                        query_len: 1,
                    },
                ],
                1.0,
            )
            .is_err());
    }

    #[test]
    fn install_rejects_oversize_and_cross_layer_length_mismatch_before_copy() {
        let mut arena = arena(vec![layer(0, 2, 2), layer(3, 2, 2)], 2);
        let first = values(0, &[1.0; 8], &[2.0; 8], 2, 2);
        let second = values(3, &[3.0; 8], &[4.0; 8], 2, 2);
        assert!(arena.install([0; 32], vec![first, second]).is_err());
        assert!(!arena.is_dirty());
        assert_eq!(arena.metadata().unwrap(), None);

        let oversize = values(0, &[0.0; 12], &[0.0; 12], 3, 2);
        let second = values(3, &[0.0; 8], &[0.0; 8], 2, 2);
        assert!(arena.install([3; 32], vec![oversize, second]).is_err());
        assert!(!arena.is_dirty());
        assert_eq!(arena.metadata().unwrap(), None);

        let first = values(0, &[1.0; 8], &[2.0; 8], 2, 2);
        let short_second = values(3, &[3.0; 4], &[4.0; 4], 1, 2);
        assert!(arena.install([3; 32], vec![first, short_second]).is_err());
        assert!(!arena.is_dirty());
        assert_eq!(arena.metadata().unwrap(), None);
        let (keys, values) = arena.layer_backing_values(0).unwrap();
        assert_eq!(keys, vec![0.0; 8]);
        assert_eq!(values, vec![0.0; 8]);
    }

    #[test]
    fn copy_failure_is_unobservable_until_reset_recovers_the_slot() {
        let mut arena = arena(vec![layer(0, 2, 2), layer(3, 2, 2)], 2);
        arena.prepare_completion().unwrap();
        let first = values(0, &[1.0; 8], &[2.0; 8], 2, 2);
        let second = values(3, &[3.0; 8], &[4.0; 8], 2, 2);
        let error = arena
            .install_with_copy_barrier([4; 32], vec![first, second], |index| {
                if index == 0 {
                    Err(invalid("injected copy failure"))
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
        assert!(error.to_string().contains("injected copy failure"));
        assert!(arena.is_dirty());
        assert!(arena.prepare_completion().is_err());
        assert!(arena.metadata().is_err());
        let queries = Tensor::zeros((1, 2, 2), DType::F32, &Device::Cpu).unwrap();
        assert!(arena
            .attend(
                0,
                &queries,
                &[StaticAttentionRaggedRow {
                    query_start: 0,
                    query_len: 1,
                }],
                1.0,
            )
            .is_err());
        arena.reset_for_reuse().unwrap();
        arena.prepare_completion().unwrap();
        assert!(!arena.is_dirty());
        assert_eq!(arena.metadata().unwrap(), None);
        let (keys, values) = arena.layer_backing_values(0).unwrap();
        assert_eq!(keys, vec![0.0; 8]);
        assert_eq!(values, vec![0.0; 8]);
    }

    #[test]
    fn install_rejects_cross_layer_same_kind_alias_before_copy() {
        let mut arena = arena(vec![layer(0, 2, 2), layer(3, 2, 2)], 2);
        let aliased_keys = arena.layer_key_backing(3).unwrap();
        let first = StaticAttentionLayerValue {
            model_layer: 0,
            keys: aliased_keys,
            values: Tensor::from_slice(&[2.0; 8], (2, 2, 2), &Device::Cpu).unwrap(),
        };
        let second = values(3, &[3.0; 8], &[4.0; 8], 2, 2);
        assert!(arena.install([5; 32], vec![first, second]).is_err());
        assert!(!arena.is_dirty());
        assert_eq!(arena.metadata().unwrap(), None);
        let (keys, values) = arena.layer_backing_values(0).unwrap();
        assert_eq!(keys, vec![0.0; 8]);
        assert_eq!(values, vec![0.0; 8]);
    }

    #[test]
    fn reset_preserves_backing_identity_and_clears_source_cursor_metadata() {
        let mut arena = arena(vec![layer(7, 2, 2)], 2);
        let ids_before = arena.layer_backing_ids(7).unwrap();
        arena
            .install([7; 32], vec![values(7, &[1.0; 8], &[2.0; 8], 2, 2)])
            .unwrap();
        assert_eq!(
            arena.metadata().unwrap(),
            Some(StaticAttentionMetadata {
                source_identity: [7; 32],
                absolute_cursor: 2,
            })
        );
        assert!(arena
            .install([8; 32], vec![values(7, &[3.0; 8], &[4.0; 8], 2, 2)])
            .is_err());
        arena.reset_for_reuse().unwrap();
        assert_eq!(arena.metadata().unwrap(), None);
        assert_eq!(arena.layer_backing_ids(7).unwrap(), ids_before);
        assert_eq!(arena.maximum_bytes(), 64);
        arena
            .install([8; 32], vec![values(7, &[3.0; 8], &[4.0; 8], 2, 2)])
            .unwrap();
    }
}
