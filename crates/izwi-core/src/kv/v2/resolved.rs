use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::backends::BackendKind;
use crate::error::{Error, Result};

use super::contract::{
    hex_fingerprint, InferenceStateContract, PagedAttentionDomainSpec, PlacementPolicy, StateDType,
    StateDomainId, StateDomainSpec, StateGroupId,
};
#[cfg(test)]
use super::contract::{AttentionMask, KeyEncoding, PrefixPolicy, StateClock};
#[cfg(test)]
use super::resolved_domains::NonPagedStateOperationQuery;
use super::resolved_domains::{NonPagedStateOperationRegistry, ResolvedNonPagedDomainPlan};

const PLAN_FINGERPRINT_DOMAIN: &[u8] = b"izwi.inference-state.resolved-plan.v2\0";

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct StatePlanFingerprint([u8; 32]);

impl StatePlanFingerprint {
    pub(crate) const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub(crate) const fn bytes(self) -> [u8; 32] {
        self.0
    }
}

impl fmt::Debug for StatePlanFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("StatePlanFingerprint(")?;
        hex_fingerprint(&self.0, formatter)?;
        formatter.write_str(")")
    }
}

impl fmt::Display for StatePlanFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        hex_fingerprint(&self.0, formatter)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct StatePlanId(StatePlanFingerprint);

impl StatePlanId {
    pub(crate) const fn new(fingerprint: StatePlanFingerprint) -> Self {
        Self(fingerprint)
    }

    pub(crate) const fn fingerprint(self) -> StatePlanFingerprint {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct OperationAbi(u32);

impl OperationAbi {
    pub(crate) const fn new(value: u32) -> Self {
        Self(value)
    }

    pub(crate) const fn get(self) -> u32 {
        self.0
    }
}

/// Stable registry identity for one backend operation. Names describe an ABI,
/// never a dynamically selected implementation or a model-family branch.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct RegisteredOperationId {
    pub(crate) name: String,
    pub(crate) abi: OperationAbi,
}

impl RegisteredOperationId {
    pub(crate) fn new(name: impl Into<String>, abi: OperationAbi) -> Self {
        Self {
            name: name.into(),
            abi,
        }
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.name.is_empty()
            || self.name.len() > 96
            || !self
                .name
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
        {
            return Err(invalid(format!(
                "invalid registered operation name `{}`",
                self.name
            )));
        }
        if self.abi.get() == 0 {
            return Err(invalid(format!(
                "registered operation {} has zero ABI revision",
                self.name
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct StateOperationSet {
    pub(crate) write: RegisteredOperationId,
    pub(crate) prefill: RegisteredOperationId,
    pub(crate) decode: RegisteredOperationId,
    /// Resolved dispatch class for each stable operation ABI. This is kept
    /// separate from the registry identity so an optimized implementation can
    /// replace a portable one without inventing a new operation name.
    #[serde(default)]
    pub(crate) implementations: PagedOperationImplementationSet,
}

impl StateOperationSet {
    fn validate(&self) -> Result<()> {
        self.write.validate()?;
        self.prefill.validate()?;
        self.decode.validate()
    }
}

/// Implementation class selected for one paged-attention operation.
///
/// `Portable` remains a direct backend implementation with the same ABI and
/// semantics; `Optimized` records that the selected backend has a specialized
/// implementation for the resolved plan. Runtime fallback is not encoded by
/// changing operation names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub(crate) enum PagedOperationImplementation {
    #[default]
    Portable,
    Optimized,
}

/// Fingerprinted implementation plan for the three paged-attention ABIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct PagedOperationImplementationSet {
    pub(crate) write: PagedOperationImplementation,
    pub(crate) prefill: PagedOperationImplementation,
    pub(crate) decode: PagedOperationImplementation,
}

impl PagedOperationImplementationSet {
    pub(crate) const fn portable() -> Self {
        Self {
            write: PagedOperationImplementation::Portable,
            prefill: PagedOperationImplementation::Portable,
            decode: PagedOperationImplementation::Portable,
        }
    }
}

impl Default for PagedOperationImplementationSet {
    fn default() -> Self {
        Self::portable()
    }
}

pub(crate) struct PagedAttentionOperationQuery<'a> {
    pub(crate) backend: BackendKind,
    pub(crate) device_ordinal: Option<u32>,
    pub(crate) page_tokens: u32,
    pub(crate) layout: StatePhysicalLayout,
    pub(crate) storage: StateStorageFormat,
    pub(crate) placement: ResolvedPlacement,
    pub(crate) semantic: &'a PagedAttentionDomainSpec,
    pub(crate) layers: &'a [StateLayerBinding],
    pub(crate) operations: &'a StateOperationSet,
}

/// Backend-owned proof used when a plan is built or revalidated. Operation
/// names alone never make a layout executable on CPU, Metal, or CUDA.
pub(crate) trait StateOperationRegistry: NonPagedStateOperationRegistry {
    fn supports_paged_attention(&self, query: &PagedAttentionOperationQuery<'_>) -> bool;
}

/// Immutable result of matching a semantic contract to one backend/device.
/// Capacity, allocation generations, model-instance identity, and resource
/// receipts are deliberately not part of this plan or its fingerprint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ResolvedStatePlan {
    pub(crate) id: StatePlanId,
    pub(crate) backend: BackendKind,
    pub(crate) device_ordinal: Option<u32>,
    pub(crate) contract_fingerprint: [u8; 32],
    pub(crate) paged_attention: Vec<ResolvedPagedAttentionGroup>,
    pub(crate) non_paged: Vec<ResolvedNonPagedDomainPlan>,
}

impl ResolvedStatePlan {
    pub(crate) fn build(
        backend: BackendKind,
        device_ordinal: Option<u32>,
        contract: &InferenceStateContract,
        paged_attention: Vec<ResolvedPagedAttentionGroup>,
        non_paged: Vec<ResolvedNonPagedDomainPlan>,
        operations: &dyn StateOperationRegistry,
    ) -> Result<Self> {
        let mut plan = Self {
            id: StatePlanId::new(StatePlanFingerprint::new([0; 32])),
            backend,
            device_ordinal,
            contract_fingerprint: contract.fingerprint()?,
            paged_attention,
            non_paged,
        };
        plan.id = StatePlanId::new(plan.compute_fingerprint()?);
        plan.validate_against(contract, operations)?;
        Ok(plan)
    }

    pub(crate) fn fingerprint(&self) -> StatePlanFingerprint {
        self.id.fingerprint()
    }

    pub(crate) fn validate_against(
        &self,
        contract: &InferenceStateContract,
        operations: &dyn StateOperationRegistry,
    ) -> Result<()> {
        contract.validate()?;
        if self.contract_fingerprint != contract.fingerprint()? {
            return Err(invalid(
                "resolved state plan belongs to a different semantic contract",
            ));
        }
        if self.id.fingerprint() != self.compute_fingerprint()? {
            return Err(invalid("resolved state plan fingerprint is stale"));
        }

        let domains = contract
            .domains
            .iter()
            .map(|domain| (domain.id(), domain))
            .collect::<HashMap<_, _>>();
        let expected_groups = contract
            .groups
            .iter()
            .map(|group| (group.id, group))
            .collect::<HashMap<_, _>>();
        let mut resolved_domains = HashSet::with_capacity(domains.len());
        let mut resolved_pairs = HashSet::with_capacity(self.paged_attention.len());
        let mut previous_pair = None;

        for group in &self.paged_attention {
            let pair = (group.group, group.domain);
            if previous_pair.is_some_and(|previous| pair <= previous) {
                return Err(invalid(
                    "resolved paged-attention groups must be in canonical group/domain order",
                ));
            }
            previous_pair = Some(pair);
            if !resolved_pairs.insert((group.group, group.domain)) {
                return Err(invalid(format!(
                    "resolved state plan repeats group {} / domain {}",
                    group.group.get(),
                    group.domain.get()
                )));
            }
            if !resolved_domains.insert(group.domain) {
                return Err(invalid(format!(
                    "state domain {} resolves more than once",
                    group.domain.get()
                )));
            }
            let semantic_group = expected_groups.get(&group.group).ok_or_else(|| {
                invalid(format!(
                    "resolved state plan references unknown group {}",
                    group.group.get()
                ))
            })?;
            if !semantic_group.domains.contains(&group.domain) {
                return Err(invalid(format!(
                    "domain {} does not belong to consistency group {}",
                    group.domain.get(),
                    group.group.get()
                )));
            }
            let domain = domains.get(&group.domain).ok_or_else(|| {
                invalid(format!(
                    "resolved state plan references unknown domain {}",
                    group.domain.get()
                ))
            })?;
            group.validate(domain, self.backend, self.device_ordinal, operations)?;
        }

        let mut previous_non_paged = None;
        for plan in &self.non_paged {
            let pair = (plan.group(), plan.domain());
            if previous_non_paged.is_some_and(|previous| pair <= previous) {
                return Err(invalid(
                    "resolved non-paged domains must be in canonical group/domain order",
                ));
            }
            previous_non_paged = Some(pair);
            if !resolved_pairs.insert(pair) || !resolved_domains.insert(plan.domain()) {
                return Err(invalid("resolved state domain appears more than once"));
            }
            let semantic_group = expected_groups.get(&plan.group()).ok_or_else(|| {
                invalid(format!(
                    "resolved state plan references unknown group {}",
                    plan.group().get()
                ))
            })?;
            if !semantic_group.domains.contains(&plan.domain()) {
                return Err(invalid(format!(
                    "domain {} does not belong to consistency group {}",
                    plan.domain().get(),
                    plan.group().get()
                )));
            }
            let domain = domains.get(&plan.domain()).ok_or_else(|| {
                invalid(format!(
                    "resolved state plan references unknown domain {}",
                    plan.domain().get()
                ))
            })?;
            plan.validate_against(domain, self.backend, self.device_ordinal, operations)?;
        }

        if resolved_domains.len() != domains.len() {
            return Err(invalid(
                "resolved state plan does not cover every semantic domain",
            ));
        }
        Ok(())
    }

    fn compute_fingerprint(&self) -> Result<StatePlanFingerprint> {
        #[derive(Serialize)]
        struct FingerprintPayload<'a> {
            backend: BackendKind,
            device_ordinal: Option<u32>,
            contract_fingerprint: &'a [u8; 32],
            paged_attention: &'a [ResolvedPagedAttentionGroup],
            non_paged: &'a [ResolvedNonPagedDomainPlan],
        }

        let payload = FingerprintPayload {
            backend: self.backend,
            device_ordinal: self.device_ordinal,
            contract_fingerprint: &self.contract_fingerprint,
            paged_attention: &self.paged_attention,
            non_paged: &self.non_paged,
        };
        let encoded = serde_json::to_vec(&payload)
            .map_err(|error| invalid(format!("failed to encode resolved state plan: {error}")))?;
        let mut hasher = Sha256::new();
        hasher.update(PLAN_FINGERPRINT_DOMAIN);
        hasher.update(encoded);
        Ok(StatePlanFingerprint::new(hasher.finalize().into()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ResolvedPagedAttentionGroup {
    pub(crate) group: StateGroupId,
    pub(crate) domain: StateDomainId,
    pub(crate) page_tokens: u32,
    pub(crate) bytes_per_page: u64,
    pub(crate) layout: StatePhysicalLayout,
    pub(crate) storage: StateStorageFormat,
    pub(crate) placement: ResolvedPlacement,
    pub(crate) layers: Vec<StateLayerBinding>,
    pub(crate) operations: StateOperationSet,
}

impl ResolvedPagedAttentionGroup {
    fn validate(
        &self,
        domain: &StateDomainSpec,
        backend: BackendKind,
        device_ordinal: Option<u32>,
        operation_registry: &dyn StateOperationRegistry,
    ) -> Result<()> {
        let StateDomainSpec::PagedAttention(spec) = domain else {
            return Err(invalid(format!(
                "resolved paged-attention group references non-paged domain {}",
                self.domain.get()
            )));
        };
        if !spec.page_size.accepts(self.page_tokens) {
            return Err(invalid(format!(
                "resolved page size {} violates domain {} constraints",
                self.page_tokens,
                self.domain.get()
            )));
        }
        if self.bytes_per_page == 0 {
            return Err(invalid(format!(
                "resolved domain {} has zero bytes per page",
                self.domain.get()
            )));
        }
        if !spec.accepted_dtypes.contains(&self.storage.dtype()) {
            return Err(invalid(format!(
                "resolved dtype {:?} was not accepted by domain {}",
                self.storage.dtype(),
                self.domain.get()
            )));
        }
        self.placement.validate_against(spec.header.placement)?;
        self.storage.validate()?;
        self.operations.validate()?;
        if self.layers.len() != spec.layers.len() {
            return Err(invalid(format!(
                "resolved domain {} has incomplete layer coverage",
                self.domain.get()
            )));
        }
        let semantic_layers = spec
            .layers
            .iter()
            .map(|layer| (layer.model_layer, layer))
            .collect::<HashMap<_, _>>();
        let mut physical_layers = HashSet::with_capacity(self.layers.len());
        let mut model_layers = HashSet::with_capacity(self.layers.len());
        let mut previous_model_layer = None;
        let mut expected_elements = 0_u64;
        for binding in &self.layers {
            if previous_model_layer.is_some_and(|previous| binding.model_layer <= previous) {
                return Err(invalid(format!(
                    "resolved domain {} layers must be in increasing model-layer order",
                    self.domain.get()
                )));
            }
            previous_model_layer = Some(binding.model_layer);
            if !model_layers.insert(binding.model_layer) {
                return Err(invalid(format!(
                    "resolved domain {} repeats model layer {}",
                    self.domain.get(),
                    binding.model_layer
                )));
            }
            if !physical_layers.insert(binding.physical_layer) {
                return Err(invalid(format!(
                    "resolved domain {} repeats physical layer {}",
                    self.domain.get(),
                    binding.physical_layer
                )));
            }
            let layer = semantic_layers.get(&binding.model_layer).ok_or_else(|| {
                invalid(format!(
                    "resolved domain {} references unknown model layer {}",
                    self.domain.get(),
                    binding.model_layer
                ))
            })?;
            let head_width = u64::from(layer.key_head_dim)
                .checked_add(u64::from(layer.value_head_dim))
                .ok_or_else(|| invalid("state page element calculation overflow"))?;
            expected_elements = expected_elements
                .checked_add(
                    u64::from(self.page_tokens)
                        .checked_mul(u64::from(layer.kv_heads))
                        .and_then(|value| value.checked_mul(head_width))
                        .ok_or_else(|| invalid("state page element calculation overflow"))?,
                )
                .ok_or_else(|| invalid("state page element calculation overflow"))?;
        }
        if model_layers.len() != semantic_layers.len() {
            return Err(invalid(format!(
                "resolved domain {} does not cover every semantic model layer",
                self.domain.get()
            )));
        }
        let expected_bytes = self.storage.bytes_for_elements(expected_elements)?;
        if expected_bytes != self.bytes_per_page {
            return Err(invalid(format!(
                "resolved domain {} declares {} bytes per page, expected {}",
                self.domain.get(),
                self.bytes_per_page,
                expected_bytes
            )));
        }

        let query = PagedAttentionOperationQuery {
            backend,
            device_ordinal,
            page_tokens: self.page_tokens,
            layout: self.layout,
            storage: self.storage,
            placement: self.placement,
            semantic: spec,
            layers: &self.layers,
            operations: &self.operations,
        };
        if !operation_registry.supports_paged_attention(&query) {
            return Err(invalid(format!(
                "backend {backend:?} device {device_ordinal:?} does not support resolved domain {}",
                self.domain.get()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) struct StateLayerBinding {
    pub(crate) model_layer: u32,
    pub(crate) physical_layer: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StatePhysicalLayout {
    PageTokenHeadDim,
    PageHeadTokenDim,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ResolvedPlacement {
    BackendLocal,
    Host,
}

impl ResolvedPlacement {
    pub(crate) fn validate_against(self, policy: PlacementPolicy) -> Result<()> {
        match (policy, self) {
            (PlacementPolicy::BackendLocal, Self::BackendLocal)
            | (PlacementPolicy::Host, Self::Host)
            | (PlacementPolicy::BackendLocalWithHostOffload, Self::BackendLocal | Self::Host) => {
                Ok(())
            }
            _ => Err(invalid(
                "resolved state placement violates the semantic placement policy",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
/// v2 initially resolves only byte-exact dense layouts. Quantized storage is
/// added only with an explicit packing ABI (grouping boundaries, alignment,
/// scale/zero-point representation, and per-segment padding).
pub(crate) enum StateStorageFormat {
    Dense { dtype: StateDType },
}

impl StateStorageFormat {
    pub(crate) const fn dtype(self) -> StateDType {
        match self {
            Self::Dense { dtype } => dtype,
        }
    }

    pub(crate) fn bytes_for_elements(self, elements: u64) -> Result<u64> {
        match self {
            Self::Dense {
                dtype: StateDType::F32,
            } => elements
                .checked_mul(4)
                .ok_or_else(|| invalid("state page byte calculation overflow")),
            Self::Dense {
                dtype: StateDType::F16 | StateDType::Bf16,
            } => elements
                .checked_mul(2)
                .ok_or_else(|| invalid("state page byte calculation overflow")),
            Self::Dense {
                dtype: StateDType::I8,
            } => Ok(elements),
            Self::Dense {
                dtype: StateDType::Q4,
            } => Err(invalid("Q4 state storage requires a quantized layout")),
        }
    }

    pub(crate) fn validate(self) -> Result<()> {
        match self {
            Self::Dense {
                dtype: StateDType::Q4,
            } => Err(invalid("Q4 state storage requires a quantized layout")),
            Self::Dense { .. } => Ok(()),
        }
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
pub(crate) struct TestOperationRegistry;

#[cfg(test)]
impl StateOperationRegistry for TestOperationRegistry {
    fn supports_paged_attention(&self, query: &PagedAttentionOperationQuery<'_>) -> bool {
        query.backend == BackendKind::Cpu
            && query.device_ordinal.is_none()
            && query.page_tokens == 16
            && !query.layers.is_empty()
            && query.semantic.header.clock == StateClock::DecoderTokens
            && matches!(
                query.semantic.header.prefix,
                PrefixPolicy::CommittedPages { .. }
            )
            && query.semantic.layers.iter().all(|layer| {
                layer.mask == AttentionMask::Causal
                    && matches!(layer.key_encoding, KeyEncoding::Rotary { .. })
            })
            && matches!(
                query.layout,
                StatePhysicalLayout::PageTokenHeadDim | StatePhysicalLayout::PageHeadTokenDim
            )
            && matches!(
                query.storage,
                StateStorageFormat::Dense {
                    dtype: StateDType::F16
                }
            )
            && query.placement == ResolvedPlacement::BackendLocal
            && query.operations.write.name == "paged_kv_write"
            && query.operations.prefill.name == "paged_attention_prefill"
            && query.operations.decode.name == "paged_attention_decode"
    }
}

#[cfg(test)]
impl NonPagedStateOperationRegistry for TestOperationRegistry {
    fn supports_non_paged(&self, _query: &NonPagedStateOperationQuery<'_>) -> bool {
        true
    }
}

#[cfg(test)]
pub(crate) fn test_plan(contract: &InferenceStateContract) -> ResolvedStatePlan {
    ResolvedStatePlan::build(
        BackendKind::Cpu,
        None,
        contract,
        vec![ResolvedPagedAttentionGroup {
            group: StateGroupId::new(1),
            domain: StateDomainId::new(1),
            page_tokens: 16,
            bytes_per_page: 16 * 4 * (64 + 64) * 2,
            layout: StatePhysicalLayout::PageTokenHeadDim,
            storage: StateStorageFormat::Dense {
                dtype: StateDType::F16,
            },
            placement: ResolvedPlacement::BackendLocal,
            layers: vec![StateLayerBinding {
                model_layer: 0,
                physical_layer: 0,
            }],
            operations: StateOperationSet {
                write: RegisteredOperationId::new("paged_kv_write", OperationAbi::new(1)),
                prefill: RegisteredOperationId::new(
                    "paged_attention_prefill",
                    OperationAbi::new(1),
                ),
                decode: RegisteredOperationId::new("paged_attention_decode", OperationAbi::new(1)),
                implementations: PagedOperationImplementationSet::portable(),
            },
        }],
        vec![],
        &TestOperationRegistry,
    )
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::v2::test_contract;

    struct RejectAllOperations;

    impl StateOperationRegistry for RejectAllOperations {
        fn supports_paged_attention(&self, _query: &PagedAttentionOperationQuery<'_>) -> bool {
            false
        }
    }

    impl NonPagedStateOperationRegistry for RejectAllOperations {
        fn supports_non_paged(&self, _query: &NonPagedStateOperationQuery<'_>) -> bool {
            false
        }
    }

    #[test]
    fn plan_fingerprint_tracks_layout_but_not_runtime_capacity() {
        let contract = test_contract();
        let first = test_plan(&contract);
        let second = test_plan(&contract);
        assert_eq!(first.fingerprint(), second.fingerprint());

        let changed = ResolvedStatePlan::build(
            BackendKind::Cpu,
            None,
            &contract,
            vec![ResolvedPagedAttentionGroup {
                layout: StatePhysicalLayout::PageHeadTokenDim,
                ..first.paged_attention[0].clone()
            }],
            vec![],
            &TestOperationRegistry,
        )
        .unwrap();
        assert_ne!(first.fingerprint(), changed.fingerprint());
    }

    #[test]
    fn plan_fingerprint_tracks_paged_operation_implementation_class() {
        let contract = test_contract();
        let first = test_plan(&contract);
        let mut changed_group = first.paged_attention[0].clone();
        changed_group.operations.implementations.decode = PagedOperationImplementation::Optimized;

        let changed = ResolvedStatePlan::build(
            BackendKind::Cpu,
            None,
            &contract,
            vec![changed_group],
            vec![],
            &TestOperationRegistry,
        )
        .unwrap();

        assert_ne!(first.fingerprint(), changed.fingerprint());
    }

    #[test]
    fn legacy_operation_set_defaults_to_portable_implementations() {
        let operations: StateOperationSet = serde_json::from_value(serde_json::json!({
            "write": { "name": "paged_kv_write", "abi": 1 },
            "prefill": { "name": "paged_attention_prefill", "abi": 1 },
            "decode": { "name": "paged_attention_decode", "abi": 1 }
        }))
        .unwrap();

        assert_eq!(
            operations.implementations,
            PagedOperationImplementationSet::portable()
        );
        assert_eq!(
            serde_json::to_value(&operations).unwrap()["implementations"]["decode"],
            "portable"
        );
    }

    #[test]
    fn plan_rejects_invalid_operation_identity_and_page_bytes() {
        let contract = test_contract();
        let valid = test_plan(&contract);

        let mut invalid_operation = valid.paged_attention[0].clone();
        invalid_operation.operations.decode =
            RegisteredOperationId::new("Paged Decode", OperationAbi::new(1));
        assert!(ResolvedStatePlan::build(
            BackendKind::Cpu,
            None,
            &contract,
            vec![invalid_operation],
            vec![],
            &TestOperationRegistry,
        )
        .is_err());

        let mut unsupported_semantics = contract.clone();
        let StateDomainSpec::PagedAttention(domain) = &mut unsupported_semantics.domains[0] else {
            unreachable!()
        };
        domain.layers[0].mask = AttentionMask::Bidirectional;
        assert!(ResolvedStatePlan::build(
            BackendKind::Cpu,
            None,
            &unsupported_semantics,
            valid.paged_attention.clone(),
            vec![],
            &TestOperationRegistry,
        )
        .is_err());

        assert!(ResolvedStatePlan::build(
            BackendKind::Cpu,
            None,
            &contract,
            valid.paged_attention.clone(),
            vec![],
            &RejectAllOperations,
        )
        .is_err());

        let mut invalid_bytes = valid.paged_attention[0].clone();
        invalid_bytes.bytes_per_page += 2;
        assert!(ResolvedStatePlan::build(
            BackendKind::Cpu,
            None,
            &contract,
            vec![invalid_bytes],
            vec![],
            &TestOperationRegistry
        )
        .is_err());

        let mut invalid_storage = valid.paged_attention[0].clone();
        invalid_storage.storage = StateStorageFormat::Dense {
            dtype: StateDType::Q4,
        };
        assert!(ResolvedStatePlan::build(
            BackendKind::Cpu,
            None,
            &contract,
            vec![invalid_storage],
            vec![],
            &TestOperationRegistry
        )
        .is_err());
    }

    #[test]
    fn plan_rejects_duplicate_semantic_layer_bindings() {
        let mut contract = test_contract();
        let StateDomainSpec::PagedAttention(domain) = &mut contract.domains[0] else {
            unreachable!()
        };
        let mut second = domain.layers[0].clone();
        second.model_layer = 1;
        domain.layers.push(second);

        let valid = test_plan(&test_contract());
        let mut group = valid.paged_attention[0].clone();
        group.bytes_per_page *= 2;
        group.layers.push(StateLayerBinding {
            model_layer: 0,
            physical_layer: 1,
        });
        assert!(ResolvedStatePlan::build(
            BackendKind::Cpu,
            None,
            &contract,
            vec![group],
            vec![],
            &TestOperationRegistry,
        )
        .is_err());
    }
}
