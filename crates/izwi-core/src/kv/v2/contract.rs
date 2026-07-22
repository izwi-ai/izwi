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
}

impl StateDomainSpec {
    pub(crate) const fn id(&self) -> StateDomainId {
        match self {
            Self::PagedAttention(spec) => spec.id,
        }
    }

    pub(crate) const fn scope(&self) -> StateScope {
        match self {
            Self::PagedAttention(spec) => spec.scope,
        }
    }

    pub(crate) const fn prefix_policy(&self) -> &PrefixPolicy {
        match self {
            Self::PagedAttention(spec) => &spec.prefix,
        }
    }

    fn validate(&self) -> Result<()> {
        match self {
            Self::PagedAttention(spec) => spec.validate(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StateScope {
    Retained,
    Invocation,
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
    pub(crate) id: StateDomainId,
    pub(crate) scope: StateScope,
    pub(crate) clock: StateClock,
    pub(crate) layers: Vec<PagedAttentionLayerSpec>,
    pub(crate) page_size: PageSizeConstraint,
    /// Ordered storage preference. Reordering this list is a semantic change
    /// and therefore intentionally changes the contract fingerprint.
    pub(crate) accepted_dtypes: Vec<StateDType>,
    pub(crate) prefix: PrefixPolicy,
}

impl PagedAttentionDomainSpec {
    fn validate(&self) -> Result<()> {
        self.clock.validate()?;
        self.page_size.validate()?;
        self.prefix.validate(self.scope)?;
        if self.layers.is_empty() {
            return Err(invalid(format!(
                "paged-attention domain {} has no layers",
                self.id.get()
            )));
        }
        let mut layer_ids = HashSet::with_capacity(self.layers.len());
        let mut previous_layer = None;
        for layer in &self.layers {
            if previous_layer.is_some_and(|previous| layer.model_layer <= previous) {
                return Err(invalid(format!(
                    "paged-attention domain {} layers must be in increasing model-layer order",
                    self.id.get()
                )));
            }
            previous_layer = Some(layer.model_layer);
            if !layer_ids.insert(layer.model_layer) {
                return Err(invalid(format!(
                    "paged-attention domain {} repeats model layer {}",
                    self.id.get(),
                    layer.model_layer
                )));
            }
            layer.validate()?;
        }
        if self.accepted_dtypes.is_empty() {
            return Err(invalid(format!(
                "paged-attention domain {} has no accepted storage dtype",
                self.id.get()
            )));
        }
        let unique = self.accepted_dtypes.iter().copied().collect::<HashSet<_>>();
        if unique.len() != self.accepted_dtypes.len() {
            return Err(invalid(format!(
                "paged-attention domain {} repeats a storage dtype",
                self.id.get()
            )));
        }
        Ok(())
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
            id: StateDomainId::new(1),
            scope: StateScope::Retained,
            clock: StateClock::DecoderTokens,
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
            prefix: PrefixPolicy::CommittedPages {
                positions: PositionSemantics::Absolute,
            },
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

    #[test]
    fn canonical_contract_fingerprint_is_stable_and_semantic() {
        let first = test_contract();
        let second = test_contract();
        assert_eq!(first.fingerprint().unwrap(), second.fingerprint().unwrap());

        let mut changed = second;
        let StateDomainSpec::PagedAttention(domain) = &mut changed.domains[0];
        domain.layers[0].kv_heads = 8;
        assert_ne!(first.fingerprint().unwrap(), changed.fingerprint().unwrap());
    }

    #[test]
    fn contract_rejects_invalid_geometry_and_noncanonical_ids() {
        let mut invalid_geometry = test_contract();
        let StateDomainSpec::PagedAttention(domain) = &mut invalid_geometry.domains[0];
        domain.layers[0].query_heads = 10;
        assert!(invalid_geometry.validate().is_err());

        let mut zero_id = test_contract();
        let StateDomainSpec::PagedAttention(domain) = &mut zero_id.domains[0];
        domain.id = StateDomainId::new(0);
        assert!(zero_id.validate().is_err());

        let mut duplicate = test_contract();
        duplicate.domains.push(duplicate.domains[0].clone());
        assert!(duplicate.validate().is_err());

        let mut noncanonical_layers = test_contract();
        let StateDomainSpec::PagedAttention(domain) = &mut noncanonical_layers.domains[0];
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
        let StateDomainSpec::PagedAttention(domain) = &mut changed_mask.domains[0];
        domain.layers[0].mask = AttentionMask::Bidirectional;
        assert_ne!(
            contract.fingerprint().unwrap(),
            changed_mask.fingerprint().unwrap()
        );

        let mut changed_preference = contract.clone();
        let StateDomainSpec::PagedAttention(domain) = &mut changed_preference.domains[0];
        domain.accepted_dtypes.reverse();
        assert_ne!(
            contract.fingerprint().unwrap(),
            changed_preference.fingerprint().unwrap()
        );
    }

    #[test]
    fn prefix_sharing_requires_retained_shareable_domains() {
        let mut contract = test_contract();
        let StateDomainSpec::PagedAttention(domain) = &mut contract.domains[0];
        domain.scope = StateScope::Invocation;
        assert!(contract.validate().is_err());

        let mut contract = test_contract();
        let StateDomainSpec::PagedAttention(domain) = &mut contract.domains[0];
        domain.prefix = PrefixPolicy::Disabled;
        assert!(contract.validate().is_err());
    }
}
