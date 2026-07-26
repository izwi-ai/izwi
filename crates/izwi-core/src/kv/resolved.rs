use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::backends::BackendKind;
use crate::engine::ModelInstanceId;
use crate::error::{Error, Result};

use super::contract::{hex_fingerprint, KvDomainSpec};
use super::{CacheDomainId, KvCacheContract, KvStorageDType};

const PLAN_FINGERPRINT_DOMAIN: &[u8] = b"izwi.kv.resolved-plan.v1\0";

#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct KvPlanFingerprint([u8; 32]);

impl KvPlanFingerprint {
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn bytes(self) -> [u8; 32] {
        self.0
    }
}

impl fmt::Debug for KvPlanFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("KvPlanFingerprint(")?;
        hex_fingerprint(&self.0, formatter)?;
        formatter.write_str(")")
    }
}

impl fmt::Display for KvPlanFingerprint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        hex_fingerprint(&self.0, formatter)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct KvPlanId(KvPlanFingerprint);

impl KvPlanId {
    pub const fn new(fingerprint: KvPlanFingerprint) -> Self {
        Self(fingerprint)
    }

    pub const fn fingerprint(self) -> KvPlanFingerprint {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct KvGroupId(u32);

impl KvGroupId {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvArenaId {
    pub model_instance: ModelInstanceId,
    pub backend: BackendKind,
    pub device_ordinal: Option<u32>,
    pub generation: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedKvPlan {
    pub id: KvPlanId,
    pub model_instance: ModelInstanceId,
    pub backend: BackendKind,
    pub device_ordinal: Option<u32>,
    pub contract_fingerprint: [u8; 32],
    pub groups: Vec<ResolvedKvGroup>,
}

impl ResolvedKvPlan {
    pub fn build(
        model_instance: ModelInstanceId,
        backend: BackendKind,
        device_ordinal: Option<u32>,
        contract: &KvCacheContract,
        groups: Vec<ResolvedKvGroup>,
    ) -> Result<Self> {
        let mut plan = Self {
            id: KvPlanId::new(KvPlanFingerprint::new([0; 32])),
            model_instance,
            backend,
            device_ordinal,
            contract_fingerprint: contract.fingerprint()?,
            groups,
        };
        let fingerprint = plan.compute_fingerprint()?;
        plan.id = KvPlanId::new(fingerprint);
        plan.validate_against(contract)?;
        Ok(plan)
    }

    pub fn fingerprint(&self) -> KvPlanFingerprint {
        self.id.fingerprint()
    }

    pub fn validate_against(&self, contract: &KvCacheContract) -> Result<()> {
        contract.validate()?;
        if self.contract_fingerprint != contract.fingerprint()? {
            return Err(invalid(
                "resolved KV plan was built for a different semantic contract",
            ));
        }
        if self.groups.is_empty() {
            return Err(invalid("resolved KV plan has no physical groups"));
        }
        if self.id.fingerprint() != self.compute_fingerprint()? {
            return Err(invalid("resolved KV plan fingerprint is stale or invalid"));
        }

        let domains: HashMap<_, _> = contract
            .domains
            .iter()
            .map(|domain| (domain.id(), domain))
            .collect();
        let mut group_ids = HashSet::with_capacity(self.groups.len());
        let mut resolved_layers = HashSet::new();

        for group in &self.groups {
            if !group_ids.insert(group.id) {
                return Err(invalid(format!(
                    "resolved KV plan repeats group {}",
                    group.id.get()
                )));
            }
            if group.arena.model_instance != self.model_instance
                || group.arena.backend != self.backend
                || group.arena.device_ordinal != self.device_ordinal
            {
                return Err(invalid(format!(
                    "resolved KV group {} points at an incompatible arena",
                    group.id.get()
                )));
            }
            let domain = domains.get(&group.domain).ok_or_else(|| {
                invalid(format!(
                    "resolved KV group {} references unknown domain {}",
                    group.id.get(),
                    group.domain.get()
                ))
            })?;
            group.validate(domain)?;
            for model_layer in group.model_layers() {
                if !resolved_layers.insert((group.domain, model_layer)) {
                    return Err(invalid(format!(
                        "cache domain {} resolves model layer {} more than once",
                        group.domain.get(),
                        model_layer
                    )));
                }
            }
        }

        for domain in &contract.domains {
            for model_layer in domain.model_layers() {
                if !resolved_layers.contains(&(domain.id(), model_layer)) {
                    return Err(invalid(format!(
                        "cache domain {} does not resolve model layer {}",
                        domain.id().get(),
                        model_layer
                    )));
                }
            }
        }
        Ok(())
    }

    fn compute_fingerprint(&self) -> Result<KvPlanFingerprint> {
        #[derive(Serialize)]
        struct FingerprintPayload<'a> {
            model_instance: ModelInstanceId,
            backend: BackendKind,
            device_ordinal: Option<u32>,
            contract_fingerprint: &'a [u8; 32],
            groups: &'a [ResolvedKvGroup],
        }

        let payload = FingerprintPayload {
            model_instance: self.model_instance,
            backend: self.backend,
            device_ordinal: self.device_ordinal,
            contract_fingerprint: &self.contract_fingerprint,
            groups: &self.groups,
        };
        let encoded = serde_json::to_vec(&payload)
            .map_err(|error| invalid(format!("failed to encode resolved KV plan: {error}")))?;
        let mut hasher = Sha256::new();
        hasher.update(PLAN_FINGERPRINT_DOMAIN);
        hasher.update(encoded);
        Ok(KvPlanFingerprint::new(hasher.finalize().into()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedKvGroup {
    pub id: KvGroupId,
    pub arena: KvArenaId,
    pub domain: CacheDomainId,
    pub page_tokens: u32,
    pub capacity_pages: u32,
    pub bytes_per_page: u64,
    pub layout: KvPhysicalLayout,
    pub storage: KvStorageFormat,
    pub kernel: PagedAttentionKernel,
    pub kind: ResolvedKvGroupKind,
}

impl ResolvedKvGroup {
    fn validate(&self, domain: &KvDomainSpec) -> Result<()> {
        if self.arena.generation == 0 {
            return Err(invalid(format!(
                "resolved KV group {} has zero arena generation",
                self.id.get()
            )));
        }
        if self.page_tokens == 0 || self.capacity_pages == 0 || self.bytes_per_page == 0 {
            return Err(invalid(format!(
                "resolved KV group {} has zero page geometry or capacity",
                self.id.get()
            )));
        }
        if !self.kernel.supports_backend(self.arena.backend) {
            return Err(invalid(format!(
                "kernel {:?} cannot execute on {:?}",
                self.kernel, self.arena.backend
            )));
        }
        self.storage.validate()?;

        match (domain, &self.kind) {
            (
                KvDomainSpec::PagedAttention(spec),
                ResolvedKvGroupKind::PagedAttention { layers },
            ) => {
                if !spec.page_tokens.accepts(self.page_tokens) {
                    return Err(invalid(format!(
                        "resolved page size {} violates domain {} constraints",
                        self.page_tokens,
                        spec.id.get()
                    )));
                }
                if !spec.storage.dtypes.contains(&self.storage.dtype()) {
                    return Err(invalid(format!(
                        "resolved dtype {:?} was not requested by domain {}",
                        self.storage.dtype(),
                        spec.id.get()
                    )));
                }
                if self.storage.is_quantized() && !spec.storage.allow_quantized {
                    return Err(invalid(format!(
                        "resolved quantized storage was not permitted by domain {}",
                        spec.id.get()
                    )));
                }
                if layers.is_empty() {
                    return Err(invalid(format!(
                        "resolved paged-attention group {} has no layers",
                        self.id.get()
                    )));
                }
                let expected = spec
                    .layers
                    .iter()
                    .map(|layer| (layer.model_layer, layer))
                    .collect::<HashMap<_, _>>();
                let mut physical_layers = HashSet::with_capacity(layers.len());
                let mut expected_bytes = 0_u64;
                for binding in layers {
                    if !physical_layers.insert(binding.physical_layer) {
                        return Err(invalid(format!(
                            "resolved group {} repeats physical layer {}",
                            self.id.get(),
                            binding.physical_layer
                        )));
                    }
                    let layer = expected.get(&binding.model_layer).ok_or_else(|| {
                        invalid(format!(
                            "resolved group {} references unknown model layer {}",
                            self.id.get(),
                            binding.model_layer
                        ))
                    })?;
                    if let Some(dtype_bytes) = self.storage.exact_element_bytes() {
                        let head_elements = u64::from(layer.key_head_dim)
                            .checked_add(u64::from(layer.value_head_dim))
                            .ok_or_else(|| invalid("resolved KV page byte calculation overflow"))?;
                        let elements = u64::from(self.page_tokens)
                            .checked_mul(u64::from(layer.num_kv_heads))
                            .and_then(|elements| elements.checked_mul(head_elements))
                            .ok_or_else(|| invalid("resolved KV page byte calculation overflow"))?;
                        expected_bytes = expected_bytes
                            .checked_add(elements.checked_mul(dtype_bytes).ok_or_else(|| {
                                invalid("resolved KV page byte calculation overflow")
                            })?)
                            .ok_or_else(|| invalid("resolved KV page byte calculation overflow"))?;
                    }
                }
                if self
                    .storage
                    .exact_element_bytes()
                    .is_some_and(|_| expected_bytes != self.bytes_per_page)
                {
                    return Err(invalid(format!(
                        "resolved group {} declares {} bytes per page, expected {}",
                        self.id.get(),
                        self.bytes_per_page,
                        expected_bytes
                    )));
                }
            }
            (KvDomainSpec::ModelState(spec), ResolvedKvGroupKind::ModelState { layers }) => {
                if !spec.storage.dtypes.contains(&self.storage.dtype()) {
                    return Err(invalid(format!(
                        "resolved dtype {:?} was not requested by domain {}",
                        self.storage.dtype(),
                        spec.id.get()
                    )));
                }
                if layers.is_empty() {
                    return Err(invalid(format!(
                        "resolved model-state group {} has no layers",
                        self.id.get()
                    )));
                }
                let expected = spec
                    .layers
                    .iter()
                    .map(|layer| layer.model_layer)
                    .collect::<HashSet<_>>();
                for binding in layers {
                    if !expected.contains(&binding.model_layer) {
                        return Err(invalid(format!(
                            "resolved group {} references unknown model-state layer {}",
                            self.id.get(),
                            binding.model_layer
                        )));
                    }
                }
            }
            (KvDomainSpec::PagedAttention(_), ResolvedKvGroupKind::ModelState { .. })
            | (KvDomainSpec::ModelState(_), ResolvedKvGroupKind::PagedAttention { .. }) => {
                return Err(invalid(format!(
                    "resolved group {} kind does not match cache domain {}",
                    self.id.get(),
                    self.domain.get()
                )));
            }
        }
        Ok(())
    }

    fn model_layers(&self) -> impl Iterator<Item = u32> + '_ {
        match &self.kind {
            ResolvedKvGroupKind::PagedAttention { layers }
            | ResolvedKvGroupKind::ModelState { layers } => {
                layers.iter().map(|layer| layer.model_layer)
            }
        }
    }
}

trait DomainLayers {
    fn model_layers(&self) -> Box<dyn Iterator<Item = u32> + '_>;
}

impl DomainLayers for KvDomainSpec {
    fn model_layers(&self) -> Box<dyn Iterator<Item = u32> + '_> {
        match self {
            Self::PagedAttention(spec) => {
                Box::new(spec.layers.iter().map(|layer| layer.model_layer))
            }
            Self::ModelState(spec) => Box::new(spec.layers.iter().map(|layer| layer.model_layer)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResolvedKvGroupKind {
    PagedAttention { layers: Vec<KvLayerBinding> },
    ModelState { layers: Vec<KvLayerBinding> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KvLayerBinding {
    pub model_layer: u32,
    pub physical_layer: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvPhysicalLayout {
    PageTokenHeadDim,
    PageHeadTokenDim,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KvStorageFormat {
    Dense {
        dtype: KvStorageDType,
    },
    Quantized {
        dtype: KvStorageDType,
        scale_dtype: KvStorageDType,
        group_size: u32,
    },
}

impl KvStorageFormat {
    pub const fn dtype(self) -> KvStorageDType {
        match self {
            Self::Dense { dtype } | Self::Quantized { dtype, .. } => dtype,
        }
    }

    pub const fn is_quantized(self) -> bool {
        matches!(self, Self::Quantized { .. })
    }

    const fn exact_element_bytes(self) -> Option<u64> {
        match self {
            Self::Dense { dtype } => dtype.dense_bytes(),
            Self::Quantized { .. } => None,
        }
    }

    fn validate(self) -> Result<()> {
        match self {
            Self::Dense { dtype } if matches!(dtype, KvStorageDType::Q4) => {
                Err(invalid("Q4 KV storage requires quantization metadata"))
            }
            Self::Dense { .. } => Ok(()),
            Self::Quantized {
                dtype,
                scale_dtype,
                group_size,
            } => {
                if !matches!(dtype, KvStorageDType::I8 | KvStorageDType::Q4) {
                    return Err(invalid(
                        "quantized KV storage requires an I8 or Q4 data dtype",
                    ));
                }
                if !matches!(
                    scale_dtype,
                    KvStorageDType::F32 | KvStorageDType::F16 | KvStorageDType::Bf16
                ) {
                    return Err(invalid("quantized KV scale dtype must be floating point"));
                }
                if group_size == 0 {
                    return Err(invalid("quantized KV group size must be non-zero"));
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PagedAttentionKernel {
    PortableReference,
    CudaPaged,
    MetalPaged,
}

impl PagedAttentionKernel {
    const fn supports_backend(self, backend: BackendKind) -> bool {
        match self {
            Self::PortableReference => true,
            Self::CudaPaged => matches!(backend, BackendKind::Cuda),
            Self::MetalPaged => matches!(backend, BackendKind::Metal),
        }
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::contract::test_contract;

    fn resolved_group(bytes_per_page: u64) -> ResolvedKvGroup {
        ResolvedKvGroup {
            id: KvGroupId::new(7),
            arena: KvArenaId {
                model_instance: ModelInstanceId::new(42),
                backend: BackendKind::Cpu,
                device_ordinal: None,
                generation: 1,
            },
            domain: CacheDomainId::new(1),
            page_tokens: 32,
            capacity_pages: 128,
            bytes_per_page,
            layout: KvPhysicalLayout::PageTokenHeadDim,
            storage: KvStorageFormat::Dense {
                dtype: KvStorageDType::F16,
            },
            kernel: PagedAttentionKernel::PortableReference,
            kind: ResolvedKvGroupKind::PagedAttention {
                layers: vec![KvLayerBinding {
                    model_layer: 0,
                    physical_layer: 0,
                }],
            },
        }
    }

    #[test]
    fn resolved_plan_validates_exact_dense_page_bytes() {
        // 32 tokens * 4 KV heads * (64 key + 64 value) * 2 bytes.
        let group = resolved_group(32 * 4 * 128 * 2);
        let plan = ResolvedKvPlan::build(
            ModelInstanceId::new(42),
            BackendKind::Cpu,
            None,
            &test_contract(),
            vec![group],
        )
        .unwrap();
        plan.validate_against(&test_contract()).unwrap();
    }

    #[test]
    fn resolved_plan_rejects_wrong_page_bytes() {
        let error = ResolvedKvPlan::build(
            ModelInstanceId::new(42),
            BackendKind::Cpu,
            None,
            &test_contract(),
            vec![resolved_group(1)],
        )
        .unwrap_err();
        assert!(error.to_string().contains("bytes per page"));
    }

    #[test]
    fn resolved_plan_fingerprint_covers_physical_layout() {
        let contract = test_contract();
        let group = resolved_group(32 * 4 * 128 * 2);
        let first = ResolvedKvPlan::build(
            ModelInstanceId::new(42),
            BackendKind::Cpu,
            None,
            &contract,
            vec![group.clone()],
        )
        .unwrap();
        let mut changed = group;
        changed.capacity_pages += 1;
        let second = ResolvedKvPlan::build(
            ModelInstanceId::new(42),
            BackendKind::Cpu,
            None,
            &contract,
            vec![changed],
        )
        .unwrap();
        assert_ne!(first.fingerprint(), second.fingerprint());
    }
}
