use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

const CONTRACT_FINGERPRINT_DOMAIN: &[u8] = b"izwi.inference-state.contract.v2\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct InferenceStateAbi(u32);

impl InferenceStateAbi {
    pub(crate) const fn new(value: u32) -> Self {
        Self(value)
    }

    pub(crate) const fn get(self) -> u32 {
        self.0
    }
}

pub(crate) const CURRENT_INFERENCE_STATE_ABI: InferenceStateAbi = InferenceStateAbi::new(2);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct StateDomainId(u32);

impl StateDomainId {
    pub(crate) const fn new(value: u32) -> Self {
        Self(value)
    }

    pub(crate) const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct StateGroupId(u32);

impl StateGroupId {
    pub(crate) const fn new(value: u32) -> Self {
        Self(value)
    }

    pub(crate) const fn get(self) -> u32 {
        self.0
    }
}

/// Model-derived semantic requirements. It contains no backend, capacity,
/// allocation, model-instance, device pointer, or scheduler identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct InferenceStateContract {
    pub(crate) abi: InferenceStateAbi,
    pub(crate) domains: Vec<StateDomainSpec>,
    pub(crate) groups: Vec<StateGroupSpec>,
}

impl InferenceStateContract {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.abi != CURRENT_INFERENCE_STATE_ABI {
            return Err(invalid(format!(
                "unsupported inference-state ABI {}; runtime supports {}",
                self.abi.get(),
                CURRENT_INFERENCE_STATE_ABI.get()
            )));
        }
        if self.domains.is_empty() {
            return Err(invalid("inference-state contract has no domains"));
        }
        if self.groups.is_empty() {
            return Err(invalid(
                "inference-state contract has no consistency groups",
            ));
        }

        let mut domains = HashMap::with_capacity(self.domains.len());
        let mut previous_domain = None;
        for domain in &self.domains {
            let id = domain.id();
            if id.get() == 0 {
                return Err(invalid("state domain id must be non-zero"));
            }
            if previous_domain.is_some_and(|previous| id <= previous) {
                return Err(invalid(
                    "state domains must be in strictly increasing id order",
                ));
            }
            previous_domain = Some(id);
            if domains.insert(id, domain).is_some() {
                return Err(invalid(format!("duplicate state domain {}", id.get())));
            }
            domain.validate()?;
        }

        let mut assigned_domains = HashSet::with_capacity(self.domains.len());
        let mut previous_group = None;
        for group in &self.groups {
            if group.id.get() == 0 {
                return Err(invalid("state group id must be non-zero"));
            }
            if previous_group.is_some_and(|previous| group.id <= previous) {
                return Err(invalid(
                    "state groups must be in strictly increasing id order",
                ));
            }
            previous_group = Some(group.id);
            group.validate(&domains, &mut assigned_domains)?;
        }

        if assigned_domains.len() != domains.len() {
            let missing = domains
                .keys()
                .filter(|id| !assigned_domains.contains(id))
                .map(|id| id.get().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(invalid(format!(
                "state domains are not assigned to a consistency group: {missing}"
            )));
        }
        Ok(())
    }

    pub(crate) fn fingerprint(&self) -> Result<[u8; 32]> {
        self.validate()?;
        let encoded = serde_json::to_vec(self).map_err(|error| {
            invalid(format!(
                "failed to encode inference-state contract: {error}"
            ))
        })?;
        let mut hasher = Sha256::new();
        hasher.update(CONTRACT_FINGERPRINT_DOMAIN);
        hasher.update(encoded);
        Ok(hasher.finalize().into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct StateGroupSpec {
    pub(crate) id: StateGroupId,
    /// Domains committed and checkpointed as one consistency closure.
    pub(crate) domains: Vec<StateDomainId>,
    /// Prefix reuse is safe only when every domain in this group reaches the
    /// same declared semantic boundary.
    pub(crate) prefix_shareable: bool,
}

impl StateGroupSpec {
    fn validate(
        &self,
        domains: &HashMap<StateDomainId, &StateDomainSpec>,
        assigned_domains: &mut HashSet<StateDomainId>,
    ) -> Result<()> {
        if self.domains.is_empty() {
            return Err(invalid(format!(
                "state group {} has no domains",
                self.id.get()
            )));
        }
        let mut previous = None;
        for domain_id in &self.domains {
            if previous.is_some_and(|value| *domain_id <= value) {
                return Err(invalid(format!(
                    "state group {} domains must be in strictly increasing id order",
                    self.id.get()
                )));
            }
            previous = Some(*domain_id);
            let domain = domains.get(domain_id).ok_or_else(|| {
                invalid(format!(
                    "state group {} references unknown domain {}",
                    self.id.get(),
                    domain_id.get()
                ))
            })?;
            if !assigned_domains.insert(*domain_id) {
                return Err(invalid(format!(
                    "state domain {} belongs to more than one consistency group",
                    domain_id.get()
                )));
            }
            if self.prefix_shareable && !domain.prefix_policy().is_shareable() {
                return Err(invalid(format!(
                    "state group {} is prefix-shareable but domain {} is not",
                    self.id.get(),
                    domain_id.get()
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum StateDomainSpec {
    PagedAttention(PagedAttentionDomainSpec),
    StaticAttention(StaticAttentionDomainSpec),
    Tensor(TensorStateDomainSpec),
    Append(AppendStateDomainSpec),
    Ring(RingStateDomainSpec),
    StaticTensor(StaticTensorDomainSpec),
}

impl StateDomainSpec {
    pub(crate) const fn id(&self) -> StateDomainId {
        self.header().id
    }

    pub(crate) const fn scope(&self) -> StateScope {
        self.header().scope
    }

    pub(crate) const fn prefix_policy(&self) -> &PrefixPolicy {
        &self.header().prefix
    }

    pub(crate) const fn header(&self) -> &StateDomainHeader {
        match self {
            Self::PagedAttention(spec) => &spec.header,
            Self::StaticAttention(spec) => &spec.header,
            Self::Tensor(spec) => &spec.header,
            Self::Append(spec) => &spec.header,
            Self::Ring(spec) => &spec.header,
            Self::StaticTensor(spec) => &spec.header,
        }
    }

    fn validate(&self) -> Result<()> {
        match self {
            Self::PagedAttention(spec) => spec.validate(),
            Self::StaticAttention(spec) => spec.validate(),
            Self::Tensor(spec) => spec.validate(),
            Self::Append(spec) => spec.validate(),
            Self::Ring(spec) => spec.validate(),
            Self::StaticTensor(spec) => spec.validate(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct StateDomainHeader {
    pub(crate) id: StateDomainId,
    pub(crate) scope: StateScope,
    pub(crate) clock: StateClock,
    pub(crate) placement: PlacementPolicy,
    pub(crate) prefix: PrefixPolicy,
    pub(crate) checkpoint: CheckpointPolicy,
}

impl StateDomainHeader {
    fn validate(&self) -> Result<()> {
        self.clock.validate()?;
        self.placement.validate()?;
        self.prefix.validate(self.scope)?;
        self.checkpoint.validate(self.scope)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StateScope {
    Retained,
    Invocation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PlacementPolicy {
    BackendLocal,
    Host,
    BackendLocalWithHostOffload,
}

impl PlacementPolicy {
    fn validate(self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum CheckpointPolicy {
    None,
    Transactional,
    CopyOnWrite,
    Replay {
        interval_steps: u32,
        max_replay_steps: u32,
    },
}

impl CheckpointPolicy {
    fn validate(self, scope: StateScope) -> Result<()> {
        if scope == StateScope::Invocation && !matches!(self, Self::None) {
            return Err(invalid(
                "invocation-scoped state cannot publish retained checkpoints",
            ));
        }
        if let Self::Replay {
            interval_steps,
            max_replay_steps,
        } = self
        {
            if interval_steps == 0 || max_replay_steps == 0 || interval_steps > max_replay_steps {
                return Err(invalid(
                    "replay checkpoint policy requires 0 < interval <= max replay steps",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "name", rename_all = "snake_case")]
pub(crate) enum StateClock {
    DecoderTokens,
    EncoderTokens,
    AudioSamples,
    AudioFrames,
    CodecFrames,
    CodebookSteps,
    Custom(String),
}

impl StateClock {
    fn validate(&self) -> Result<()> {
        if let Self::Custom(name) = self {
            require_non_empty(name, "custom state clock")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StateDType {
    F32,
    F16,
    Bf16,
    I8,
    Q4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct PageSizeConstraint {
    pub(crate) min_tokens: u32,
    pub(crate) preferred_tokens: u32,
    pub(crate) max_tokens: u32,
    pub(crate) multiple_of: u32,
}

impl PageSizeConstraint {
    pub(crate) fn accepts(self, page_tokens: u32) -> bool {
        page_tokens >= self.min_tokens
            && page_tokens <= self.max_tokens
            && self.multiple_of != 0
            && page_tokens % self.multiple_of == 0
    }

    fn validate(self) -> Result<()> {
        if self.min_tokens == 0
            || self.preferred_tokens == 0
            || self.max_tokens == 0
            || self.multiple_of == 0
        {
            return Err(invalid("page-size constraints must be non-zero"));
        }
        if self.min_tokens > self.preferred_tokens
            || self.preferred_tokens > self.max_tokens
            || !self.accepts(self.preferred_tokens)
        {
            return Err(invalid(
                "page-size constraints must satisfy min <= preferred <= max and the required multiple",
            ));
        }
        let remainder = self.min_tokens % self.multiple_of;
        let first_accepted = if remainder == 0 {
            self.min_tokens
        } else {
            self.min_tokens
                .checked_add(self.multiple_of - remainder)
                .ok_or_else(|| invalid("page-size constraint calculation overflow"))?
        };
        if first_accepted > self.max_tokens {
            return Err(invalid(
                "page-size constraints contain no value satisfying the required multiple",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PagedAttentionDomainSpec {
    pub(crate) header: StateDomainHeader,
    pub(crate) layers: Vec<PagedAttentionLayerSpec>,
    pub(crate) page_size: PageSizeConstraint,
    /// Ordered storage preference. Reordering this list is a semantic change
    /// and therefore intentionally changes the contract fingerprint.
    pub(crate) accepted_dtypes: Vec<StateDType>,
}

impl PagedAttentionDomainSpec {
    fn validate(&self) -> Result<()> {
        self.header.validate()?;
        self.page_size.validate()?;
        if self.layers.is_empty() {
            return Err(invalid(format!(
                "paged-attention domain {} has no layers",
                self.header.id.get()
            )));
        }
        let mut layer_ids = HashSet::with_capacity(self.layers.len());
        let mut previous_layer = None;
        for layer in &self.layers {
            if previous_layer.is_some_and(|previous| layer.model_layer <= previous) {
                return Err(invalid(format!(
                    "paged-attention domain {} layers must be in increasing model-layer order",
                    self.header.id.get()
                )));
            }
            previous_layer = Some(layer.model_layer);
            if !layer_ids.insert(layer.model_layer) {
                return Err(invalid(format!(
                    "paged-attention domain {} repeats model layer {}",
                    self.header.id.get(),
                    layer.model_layer
                )));
            }
            layer.validate()?;
        }
        if self.accepted_dtypes.is_empty() {
            return Err(invalid(format!(
                "paged-attention domain {} has no accepted storage dtype",
                self.header.id.get()
            )));
        }
        let unique = self.accepted_dtypes.iter().copied().collect::<HashSet<_>>();
        if unique.len() != self.accepted_dtypes.len() {
            return Err(invalid(format!(
                "paged-attention domain {} repeats a storage dtype",
                self.header.id.get()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct StateComponentId(u32);

impl StateComponentId {
    pub(crate) const fn new(value: u32) -> Self {
        Self(value)
    }

    pub(crate) const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "name", rename_all = "snake_case")]
pub(crate) enum ShapeAxis {
    Batch,
    Sequence,
    Heads,
    HeadDim,
    Channels,
    Hidden,
    Layers,
    Samples,
    Frames,
    Codebooks,
    Custom(String),
}

impl ShapeAxis {
    fn validate(&self) -> Result<()> {
        if let Self::Custom(name) = self {
            require_non_empty(name, "custom shape axis")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum ShapeExtent {
    Fixed { value: u64 },
    RuntimeBounded { min: u64, max: u64 },
}

impl ShapeExtent {
    fn validate(self) -> Result<()> {
        match self {
            Self::Fixed { value } if value > 0 => Ok(()),
            Self::RuntimeBounded { min, max } if min > 0 && min <= max => Ok(()),
            _ => Err(invalid(
                "shape extent must be non-zero and satisfy min <= max",
            )),
        }
    }

    const fn max(self) -> u64 {
        match self {
            Self::Fixed { value } => value,
            Self::RuntimeBounded { max, .. } => max,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct ShapeDimension {
    pub(crate) axis: ShapeAxis,
    pub(crate) extent: ShapeExtent,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct BoundedShape {
    pub(crate) dimensions: Vec<ShapeDimension>,
}

impl BoundedShape {
    fn validate(&self) -> Result<()> {
        if self.dimensions.is_empty() {
            return Err(invalid("bounded tensor shape has no dimensions"));
        }
        let mut axes = HashSet::with_capacity(self.dimensions.len());
        let mut max_elements = 1_u64;
        for dimension in &self.dimensions {
            dimension.axis.validate()?;
            dimension.extent.validate()?;
            if !axes.insert(&dimension.axis) {
                return Err(invalid("bounded tensor shape repeats an axis"));
            }
            max_elements = max_elements
                .checked_mul(dimension.extent.max())
                .ok_or_else(|| invalid("bounded tensor shape element count overflow"))?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "name", rename_all = "snake_case")]
pub(crate) enum TensorRole {
    RecurrentHidden,
    RecurrentCell,
    ConvolutionState,
    RetainedEmbedding,
    RetainedLogits,
    AudioHistory,
    EncoderMemory,
    Control,
    Custom(String),
}

impl TensorRole {
    fn validate(&self) -> Result<()> {
        if let Self::Custom(name) = self {
            require_non_empty(name, "custom tensor role")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct TensorComponentSpec {
    pub(crate) id: StateComponentId,
    pub(crate) role: TensorRole,
    pub(crate) shape: BoundedShape,
    /// Ordered storage preference, as for paged attention.
    pub(crate) accepted_dtypes: Vec<StateDType>,
}

fn validate_components(domain_id: StateDomainId, components: &[TensorComponentSpec]) -> Result<()> {
    if components.is_empty() {
        return Err(invalid(format!(
            "state domain {} has no tensor components",
            domain_id.get()
        )));
    }
    let mut previous = None;
    for component in components {
        if component.id.get() == 0 || previous.is_some_and(|previous| component.id <= previous) {
            return Err(invalid(format!(
                "state domain {} components require non-zero increasing ids",
                domain_id.get()
            )));
        }
        previous = Some(component.id);
        component.role.validate()?;
        component.shape.validate()?;
        validate_dtype_preferences(domain_id, &component.accepted_dtypes)?;
    }
    Ok(())
}

fn validate_dtype_preferences(domain_id: StateDomainId, dtypes: &[StateDType]) -> Result<()> {
    if dtypes.is_empty() {
        return Err(invalid(format!(
            "state domain {} has no accepted storage dtype",
            domain_id.get()
        )));
    }
    if dtypes.iter().copied().collect::<HashSet<_>>().len() != dtypes.len() {
        return Err(invalid(format!(
            "state domain {} repeats a storage dtype",
            domain_id.get()
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct TensorStateDomainSpec {
    pub(crate) header: StateDomainHeader,
    pub(crate) components: Vec<TensorComponentSpec>,
}

impl TensorStateDomainSpec {
    fn validate(&self) -> Result<()> {
        self.header.validate()?;
        validate_components(self.header.id, &self.components)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct StaticTensorDomainSpec {
    pub(crate) header: StateDomainHeader,
    pub(crate) components: Vec<TensorComponentSpec>,
}

impl StaticTensorDomainSpec {
    fn validate(&self) -> Result<()> {
        self.header.validate()?;
        validate_components(self.header.id, &self.components)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct AppendStateDomainSpec {
    pub(crate) header: StateDomainHeader,
    /// Shape of one appended clock step for each component.
    pub(crate) components_per_step: Vec<TensorComponentSpec>,
    pub(crate) max_steps: u64,
}

impl AppendStateDomainSpec {
    fn validate(&self) -> Result<()> {
        self.header.validate()?;
        if self.max_steps == 0 {
            return Err(invalid(format!(
                "append state domain {} has zero maximum steps",
                self.header.id.get()
            )));
        }
        validate_components(self.header.id, &self.components_per_step)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RingStateDomainSpec {
    pub(crate) header: StateDomainHeader,
    /// Shape of one ring position for each component.
    pub(crate) components_per_step: Vec<TensorComponentSpec>,
    pub(crate) capacity_steps: u64,
}

impl RingStateDomainSpec {
    fn validate(&self) -> Result<()> {
        self.header.validate()?;
        if self.capacity_steps == 0 {
            return Err(invalid(format!(
                "ring state domain {} has zero capacity",
                self.header.id.get()
            )));
        }
        validate_components(self.header.id, &self.components_per_step)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct StaticAttentionDomainSpec {
    pub(crate) header: StateDomainHeader,
    pub(crate) layers: Vec<StaticAttentionLayerSpec>,
    pub(crate) max_memory_tokens: u64,
    pub(crate) accepted_dtypes: Vec<StateDType>,
}

impl StaticAttentionDomainSpec {
    fn validate(&self) -> Result<()> {
        self.header.validate()?;
        if self.max_memory_tokens == 0 || self.layers.is_empty() {
            return Err(invalid(format!(
                "static-attention domain {} requires layers and bounded memory tokens",
                self.header.id.get()
            )));
        }
        let mut previous = None;
        for layer in &self.layers {
            if previous.is_some_and(|previous| layer.model_layer <= previous) {
                return Err(invalid(format!(
                    "static-attention domain {} layers must be in increasing model-layer order",
                    self.header.id.get()
                )));
            }
            previous = Some(layer.model_layer);
            layer.validate()?;
        }
        validate_dtype_preferences(self.header.id, &self.accepted_dtypes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct StaticAttentionLayerSpec {
    pub(crate) model_layer: u32,
    pub(crate) query_heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) key_head_dim: u32,
    pub(crate) value_head_dim: u32,
    pub(crate) key_encoding: KeyEncoding,
}

impl StaticAttentionLayerSpec {
    fn validate(&self) -> Result<()> {
        if self.query_heads == 0
            || self.kv_heads == 0
            || self.key_head_dim == 0
            || self.value_head_dim == 0
            || self.query_heads % self.kv_heads != 0
        {
            return Err(invalid(format!(
                "static-attention layer {} has invalid geometry",
                self.model_layer
            )));
        }
        self.key_encoding.validate(self.key_head_dim)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PagedAttentionLayerSpec {
    pub(crate) model_layer: u32,
    pub(crate) query_heads: u32,
    pub(crate) kv_heads: u32,
    pub(crate) key_head_dim: u32,
    pub(crate) value_head_dim: u32,
    pub(crate) pattern: AttentionPattern,
    pub(crate) mask: AttentionMask,
    pub(crate) key_encoding: KeyEncoding,
}

impl PagedAttentionLayerSpec {
    fn validate(&self) -> Result<()> {
        if self.query_heads == 0
            || self.kv_heads == 0
            || self.key_head_dim == 0
            || self.value_head_dim == 0
        {
            return Err(invalid(format!(
                "paged-attention layer {} has zero geometry",
                self.model_layer
            )));
        }
        if self.query_heads % self.kv_heads != 0 {
            return Err(invalid(format!(
                "paged-attention layer {} query heads are not divisible by KV heads",
                self.model_layer
            )));
        }
        self.pattern.validate()?;
        self.mask.validate()?;
        self.key_encoding.validate(self.key_head_dim)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AttentionMask {
    Causal,
    Bidirectional,
}

impl AttentionMask {
    fn validate(self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "name", rename_all = "snake_case")]
pub(crate) enum KeyEncoding {
    Raw,
    Rotary { rotary_dim: u32 },
    ModelDefined(String),
}

impl KeyEncoding {
    fn validate(&self, key_head_dim: u32) -> Result<()> {
        match self {
            Self::Raw => Ok(()),
            Self::Rotary { rotary_dim }
                if *rotary_dim > 0
                    && *rotary_dim <= key_head_dim
                    && *rotary_dim % 2 == 0 =>
            {
                Ok(())
            }
            Self::Rotary { rotary_dim } => Err(invalid(format!(
                "rotary dimension {rotary_dim} must be even, non-zero, and no greater than key head dimension {key_head_dim}"
            ))),
            Self::ModelDefined(name) => require_non_empty(name, "model-defined key encoding"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum AttentionPattern {
    Full,
    SlidingWindow { window_tokens: u32 },
}

impl AttentionPattern {
    fn validate(self) -> Result<()> {
        if matches!(self, Self::SlidingWindow { window_tokens: 0 }) {
            return Err(invalid("sliding-attention window must be non-zero"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum PrefixPolicy {
    Disabled,
    CommittedPages { positions: PositionSemantics },
}

impl PrefixPolicy {
    const fn is_shareable(&self) -> bool {
        matches!(self, Self::CommittedPages { .. })
    }

    fn validate(&self, scope: StateScope) -> Result<()> {
        if self.is_shareable() && scope != StateScope::Retained {
            return Err(invalid(
                "invocation-scoped state cannot publish shared prefixes",
            ));
        }
        if let Self::CommittedPages { positions } = self {
            positions.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "name", rename_all = "snake_case")]
pub(crate) enum PositionSemantics {
    Absolute,
    WindowRelative,
    ModelDefined(String),
}

impl PositionSemantics {
    fn validate(&self) -> Result<()> {
        if let Self::ModelDefined(name) = self {
            require_non_empty(name, "model-defined position semantics")?;
        }
        Ok(())
    }
}

fn require_non_empty(value: &str, name: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(invalid(format!("{name} must not be empty")));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

pub(super) fn hex_fingerprint(bytes: &[u8; 32], formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    for byte in bytes {
        write!(formatter, "{byte:02x}")?;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn test_contract() -> InferenceStateContract {
    InferenceStateContract {
        abi: CURRENT_INFERENCE_STATE_ABI,
        domains: vec![StateDomainSpec::PagedAttention(PagedAttentionDomainSpec {
            header: StateDomainHeader {
                id: StateDomainId::new(1),
                scope: StateScope::Retained,
                clock: StateClock::DecoderTokens,
                placement: PlacementPolicy::BackendLocalWithHostOffload,
                prefix: PrefixPolicy::CommittedPages {
                    positions: PositionSemantics::Absolute,
                },
                checkpoint: CheckpointPolicy::CopyOnWrite,
            },
            layers: vec![PagedAttentionLayerSpec {
                model_layer: 0,
                query_heads: 16,
                kv_heads: 4,
                key_head_dim: 64,
                value_head_dim: 64,
                pattern: AttentionPattern::Full,
                mask: AttentionMask::Causal,
                key_encoding: KeyEncoding::Rotary { rotary_dim: 64 },
            }],
            page_size: PageSizeConstraint {
                min_tokens: 8,
                preferred_tokens: 16,
                max_tokens: 64,
                multiple_of: 8,
            },
            accepted_dtypes: vec![StateDType::F16, StateDType::Bf16],
        })],
        groups: vec![StateGroupSpec {
            id: StateGroupId::new(1),
            domains: vec![StateDomainId::new(1)],
            prefix_shareable: true,
        }],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(
        id: u32,
        scope: StateScope,
        clock: StateClock,
        checkpoint: CheckpointPolicy,
    ) -> StateDomainHeader {
        StateDomainHeader {
            id: StateDomainId::new(id),
            scope,
            clock,
            placement: PlacementPolicy::BackendLocal,
            prefix: PrefixPolicy::Disabled,
            checkpoint,
        }
    }

    fn component(id: u32, role: TensorRole) -> TensorComponentSpec {
        TensorComponentSpec {
            id: StateComponentId::new(id),
            role,
            shape: BoundedShape {
                dimensions: vec![ShapeDimension {
                    axis: ShapeAxis::Hidden,
                    extent: ShapeExtent::Fixed { value: 64 },
                }],
            },
            accepted_dtypes: vec![StateDType::F16, StateDType::Bf16],
        }
    }

    #[test]
    fn canonical_contract_fingerprint_is_stable_and_semantic() {
        let first = test_contract();
        let second = test_contract();
        assert_eq!(first.fingerprint().unwrap(), second.fingerprint().unwrap());

        let mut changed = second;
        let StateDomainSpec::PagedAttention(domain) = &mut changed.domains[0] else {
            unreachable!()
        };
        domain.layers[0].kv_heads = 8;
        assert_ne!(first.fingerprint().unwrap(), changed.fingerprint().unwrap());
    }

    #[test]
    fn contract_rejects_invalid_geometry_and_noncanonical_ids() {
        let mut invalid_geometry = test_contract();
        let StateDomainSpec::PagedAttention(domain) = &mut invalid_geometry.domains[0] else {
            unreachable!()
        };
        domain.layers[0].query_heads = 10;
        assert!(invalid_geometry.validate().is_err());

        let mut zero_id = test_contract();
        let StateDomainSpec::PagedAttention(domain) = &mut zero_id.domains[0] else {
            unreachable!()
        };
        domain.header.id = StateDomainId::new(0);
        assert!(zero_id.validate().is_err());

        let mut duplicate = test_contract();
        duplicate.domains.push(duplicate.domains[0].clone());
        assert!(duplicate.validate().is_err());

        let mut noncanonical_layers = test_contract();
        let StateDomainSpec::PagedAttention(domain) = &mut noncanonical_layers.domains[0] else {
            unreachable!()
        };
        let mut second_layer = domain.layers[0].clone();
        second_layer.model_layer = 1;
        domain.layers.push(second_layer);
        domain.layers.swap(0, 1);
        assert!(noncanonical_layers.validate().is_err());
    }

    #[test]
    fn key_mask_position_and_dtype_preference_are_semantic() {
        let contract = test_contract();

        let mut changed_mask = contract.clone();
        let StateDomainSpec::PagedAttention(domain) = &mut changed_mask.domains[0] else {
            unreachable!()
        };
        domain.layers[0].mask = AttentionMask::Bidirectional;
        assert_ne!(
            contract.fingerprint().unwrap(),
            changed_mask.fingerprint().unwrap()
        );

        let mut changed_preference = contract.clone();
        let StateDomainSpec::PagedAttention(domain) = &mut changed_preference.domains[0] else {
            unreachable!()
        };
        domain.accepted_dtypes.reverse();
        assert_ne!(
            contract.fingerprint().unwrap(),
            changed_preference.fingerprint().unwrap()
        );
    }

    #[test]
    fn prefix_sharing_requires_retained_shareable_domains() {
        let mut contract = test_contract();
        let StateDomainSpec::PagedAttention(domain) = &mut contract.domains[0] else {
            unreachable!()
        };
        domain.header.scope = StateScope::Invocation;
        assert!(contract.validate().is_err());

        let mut contract = test_contract();
        let StateDomainSpec::PagedAttention(domain) = &mut contract.domains[0] else {
            unreachable!()
        };
        domain.header.prefix = PrefixPolicy::Disabled;
        assert!(contract.validate().is_err());
    }

    #[test]
    fn composite_contract_covers_static_tensor_append_ring_and_cross_attention_state() {
        let mut contract = test_contract();
        contract.domains.push(StateDomainSpec::StaticAttention(
            StaticAttentionDomainSpec {
                header: header(
                    2,
                    StateScope::Retained,
                    StateClock::EncoderTokens,
                    CheckpointPolicy::Transactional,
                ),
                layers: vec![StaticAttentionLayerSpec {
                    model_layer: 0,
                    query_heads: 16,
                    kv_heads: 4,
                    key_head_dim: 64,
                    value_head_dim: 64,
                    key_encoding: KeyEncoding::Raw,
                }],
                max_memory_tokens: 2048,
                accepted_dtypes: vec![StateDType::F16],
            },
        ));
        contract
            .domains
            .push(StateDomainSpec::Tensor(TensorStateDomainSpec {
                header: header(
                    3,
                    StateScope::Retained,
                    StateClock::DecoderTokens,
                    CheckpointPolicy::Transactional,
                ),
                components: vec![component(1, TensorRole::RecurrentHidden)],
            }));
        contract
            .domains
            .push(StateDomainSpec::Append(AppendStateDomainSpec {
                header: header(
                    4,
                    StateScope::Retained,
                    StateClock::AudioSamples,
                    CheckpointPolicy::Replay {
                        interval_steps: 64,
                        max_replay_steps: 256,
                    },
                ),
                components_per_step: vec![component(1, TensorRole::AudioHistory)],
                max_steps: 16_000,
            }));
        contract
            .domains
            .push(StateDomainSpec::Ring(RingStateDomainSpec {
                header: header(
                    5,
                    StateScope::Retained,
                    StateClock::AudioFrames,
                    CheckpointPolicy::CopyOnWrite,
                ),
                components_per_step: vec![component(1, TensorRole::ConvolutionState)],
                capacity_steps: 32,
            }));
        contract
            .domains
            .push(StateDomainSpec::StaticTensor(StaticTensorDomainSpec {
                header: header(
                    6,
                    StateScope::Invocation,
                    StateClock::CodecFrames,
                    CheckpointPolicy::None,
                ),
                components: vec![component(1, TensorRole::EncoderMemory)],
            }));
        contract.groups = vec![StateGroupSpec {
            id: StateGroupId::new(1),
            domains: (1..=6).map(StateDomainId::new).collect(),
            prefix_shareable: false,
        }];

        contract.validate().unwrap();
        assert_ne!(
            contract.fingerprint().unwrap(),
            test_contract().fingerprint().unwrap()
        );

        let StateDomainSpec::Append(append) = &mut contract.domains[3] else {
            unreachable!()
        };
        append.max_steps = 0;
        assert!(contract.validate().is_err());
    }
}
