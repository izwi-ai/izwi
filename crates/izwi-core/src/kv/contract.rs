use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

const CONTRACT_FINGERPRINT_DOMAIN: &[u8] = b"izwi.kv.contract.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct KvContractAbi(u32);

impl KvContractAbi {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

pub const CURRENT_KV_CONTRACT_ABI: KvContractAbi = KvContractAbi::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CacheDomainId(u32);

impl CacheDomainId {
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Cache behavior published by an exact loaded-model implementation.
///
/// `OpaqueModelOwned` is a supported compatibility mode, not an implicit
/// managed-cache contract. The engine must never infer physical paging from
/// model family or catalog metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", content = "contract", rename_all = "snake_case")]
pub enum CacheCapability {
    None,
    OpaqueModelOwned,
    Managed(KvCacheContract),
}

impl CacheCapability {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::None | Self::OpaqueModelOwned => Ok(()),
            Self::Managed(contract) => contract.validate(),
        }
    }

    pub fn managed_contract(&self) -> Option<&KvCacheContract> {
        match self {
            Self::Managed(contract) => Some(contract),
            Self::None | Self::OpaqueModelOwned => None,
        }
    }
}

/// Implemented by the loaded adapter/model boundary, never by catalog entries.
pub trait KvCacheContractProvider {
    fn kv_cache_contract(&self) -> Result<CacheCapability>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheContract {
    pub abi: KvContractAbi,
    pub domains: Vec<KvDomainSpec>,
}

impl KvCacheContract {
    pub fn validate(&self) -> Result<()> {
        if self.abi != CURRENT_KV_CONTRACT_ABI {
            return Err(invalid(format!(
                "unsupported KV contract ABI {}; runtime supports {}",
                self.abi.get(),
                CURRENT_KV_CONTRACT_ABI.get()
            )));
        }
        if self.domains.is_empty() {
            return Err(invalid("managed KV contract has no cache domains"));
        }

        let mut domain_ids = HashSet::with_capacity(self.domains.len());
        for domain in &self.domains {
            if !domain_ids.insert(domain.id()) {
                return Err(invalid(format!(
                    "managed KV contract repeats cache domain {}",
                    domain.id().get()
                )));
            }
            domain.validate()?;
        }
        Ok(())
    }

    /// Stable SHA-256 identity of the semantic cache ABI.
    ///
    /// The domain separator makes this unsuitable for accidental reuse as a
    /// resolved-layout fingerprint. Struct fields and enum tags are serialized
    /// in declaration order, so changing the public contract schema requires a
    /// new ABI revision.
    pub fn fingerprint(&self) -> Result<[u8; 32]> {
        self.validate()?;
        let encoded = serde_json::to_vec(self)
            .map_err(|error| invalid(format!("failed to encode KV contract: {error}")))?;
        let mut hasher = Sha256::new();
        hasher.update(CONTRACT_FINGERPRINT_DOMAIN);
        hasher.update(encoded);
        Ok(hasher.finalize().into())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KvDomainSpec {
    PagedAttention(PagedAttentionDomainSpec),
    ModelState(ModelStateDomainSpec),
}

impl KvDomainSpec {
    pub const fn id(&self) -> CacheDomainId {
        match self {
            Self::PagedAttention(spec) => spec.id,
            Self::ModelState(spec) => spec.id,
        }
    }

    fn validate(&self) -> Result<()> {
        match self {
            Self::PagedAttention(spec) => spec.validate(),
            Self::ModelState(spec) => spec.validate(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "name", rename_all = "snake_case")]
pub enum CacheTokenAxis {
    DecoderTokens,
    EncoderTokens,
    CrossAttentionMemory,
    Custom(String),
}

impl CacheTokenAxis {
    fn validate(&self) -> Result<()> {
        if let Self::Custom(name) = self {
            require_non_empty(name, "custom cache token axis")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvStorageDType {
    F32,
    F16,
    Bf16,
    I8,
    Q4,
}

impl KvStorageDType {
    pub const fn dense_bytes(self) -> Option<u64> {
        match self {
            Self::F32 => Some(4),
            Self::F16 | Self::Bf16 => Some(2),
            Self::I8 => Some(1),
            Self::Q4 => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvStorageRequest {
    /// Ordered backend preference; the backend may select only a listed type.
    pub dtypes: Vec<KvStorageDType>,
    pub allow_quantized: bool,
}

impl KvStorageRequest {
    fn validate(&self) -> Result<()> {
        if self.dtypes.is_empty() {
            return Err(invalid("KV storage request has no accepted dtypes"));
        }
        let mut seen = HashSet::with_capacity(self.dtypes.len());
        for dtype in &self.dtypes {
            if !seen.insert(*dtype) {
                return Err(invalid(format!(
                    "KV storage request repeats dtype {dtype:?}"
                )));
            }
            if matches!(dtype, KvStorageDType::I8 | KvStorageDType::Q4) && !self.allow_quantized {
                return Err(invalid(format!(
                    "quantized KV dtype {dtype:?} requires allow_quantized"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageTokenConstraint {
    pub min: u32,
    pub preferred: u32,
    pub max: u32,
    pub multiple_of: u32,
}

impl PageTokenConstraint {
    pub fn accepts(self, page_tokens: u32) -> bool {
        page_tokens >= self.min
            && page_tokens <= self.max
            && self.multiple_of != 0
            && page_tokens % self.multiple_of == 0
    }

    fn validate(self) -> Result<()> {
        if self.min == 0 || self.preferred == 0 || self.max == 0 || self.multiple_of == 0 {
            return Err(invalid("page token constraints must be non-zero"));
        }
        if self.min > self.preferred || self.preferred > self.max {
            return Err(invalid(
                "page token constraint must satisfy min <= preferred <= max",
            ));
        }
        if !self.accepts(self.preferred) {
            return Err(invalid(
                "preferred page token count does not satisfy its multiple",
            ));
        }
        if self.min.div_ceil(self.multiple_of) * self.multiple_of > self.max {
            return Err(invalid(
                "page token constraint contains no value satisfying its multiple",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PagedAttentionDomainSpec {
    pub id: CacheDomainId,
    pub token_axis: CacheTokenAxis,
    pub layers: Vec<PagedAttentionLayerSpec>,
    pub page_tokens: PageTokenConstraint,
    pub storage: KvStorageRequest,
    pub prefix_semantics: KvPrefixSemantics,
}

impl PagedAttentionDomainSpec {
    fn validate(&self) -> Result<()> {
        self.token_axis.validate()?;
        self.page_tokens.validate()?;
        self.storage.validate()?;
        self.prefix_semantics.validate()?;
        if self.layers.is_empty() {
            return Err(invalid(format!(
                "paged-attention domain {} has no layers",
                self.id.get()
            )));
        }

        let mut layers = HashSet::with_capacity(self.layers.len());
        for layer in &self.layers {
            if !layers.insert(layer.model_layer) {
                return Err(invalid(format!(
                    "paged-attention domain {} repeats model layer {}",
                    self.id.get(),
                    layer.model_layer
                )));
            }
            layer.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PagedAttentionLayerSpec {
    pub model_layer: u32,
    pub num_query_heads: u32,
    pub num_kv_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub attention: AttentionSemantics,
    pub key_encoding: KeyEncoding,
}

impl PagedAttentionLayerSpec {
    fn validate(&self) -> Result<()> {
        if self.num_query_heads == 0
            || self.num_kv_heads == 0
            || self.key_head_dim == 0
            || self.value_head_dim == 0
        {
            return Err(invalid(format!(
                "KV geometry for model layer {} must be non-zero",
                self.model_layer
            )));
        }
        if self.num_query_heads % self.num_kv_heads != 0 {
            return Err(invalid(format!(
                "model layer {} has {} query heads not divisible by {} KV heads",
                self.model_layer, self.num_query_heads, self.num_kv_heads
            )));
        }
        self.attention.validate()?;
        self.key_encoding.validate(self.key_head_dim)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AttentionSemantics {
    Full,
    SlidingWindow { window_tokens: u32 },
}

impl AttentionSemantics {
    fn validate(self) -> Result<()> {
        if matches!(self, Self::SlidingWindow { window_tokens: 0 }) {
            return Err(invalid("sliding attention window must be non-zero"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KeyEncoding {
    Raw,
    Rotary { rotary_dim: u32 },
}

impl KeyEncoding {
    fn validate(self, key_head_dim: u32) -> Result<()> {
        if let Self::Rotary { rotary_dim } = self {
            if rotary_dim == 0 || rotary_dim > key_head_dim || rotary_dim % 2 != 0 {
                return Err(invalid(format!(
                    "rotary dimension {rotary_dim} must be even, non-zero, and no greater than key head dimension {key_head_dim}"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "name", rename_all = "snake_case")]
pub enum PositionSemantics {
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

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KvPrefixSemantics {
    Disabled,
    CommittedFullPages { positions: PositionSemantics },
}

impl KvPrefixSemantics {
    fn validate(&self) -> Result<()> {
        if let Self::CommittedFullPages { positions } = self {
            positions.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStateDomainSpec {
    pub id: CacheDomainId,
    pub token_axis: CacheTokenAxis,
    pub layers: Vec<ModelStateLayerSpec>,
    pub storage: KvStorageRequest,
    pub prefix_semantics: KvPrefixSemantics,
}

impl ModelStateDomainSpec {
    fn validate(&self) -> Result<()> {
        self.token_axis.validate()?;
        self.storage.validate()?;
        self.prefix_semantics.validate()?;
        if self.layers.is_empty() {
            return Err(invalid(format!(
                "model-state domain {} has no layers",
                self.id.get()
            )));
        }

        let mut layers = HashSet::with_capacity(self.layers.len());
        for layer in &self.layers {
            if !layers.insert(layer.model_layer) {
                return Err(invalid(format!(
                    "model-state domain {} repeats model layer {}",
                    self.id.get(),
                    layer.model_layer
                )));
            }
            layer.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStateLayerSpec {
    pub model_layer: u32,
    pub kind: ModelStateKind,
    pub elements_per_sequence: u64,
}

impl ModelStateLayerSpec {
    fn validate(&self) -> Result<()> {
        self.kind.validate()?;
        if self.elements_per_sequence == 0 {
            return Err(invalid(format!(
                "model-state layer {} has zero elements per sequence",
                self.model_layer
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "name", rename_all = "snake_case")]
pub enum ModelStateKind {
    Recurrent,
    Convolution,
    CrossAttention,
    Custom(String),
}

impl ModelStateKind {
    fn validate(&self) -> Result<()> {
        if let Self::Custom(name) = self {
            require_non_empty(name, "custom model state kind")?;
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

pub(crate) fn hex_fingerprint(bytes: &[u8; 32], formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    for byte in bytes {
        write!(formatter, "{byte:02x}")?;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn test_contract() -> KvCacheContract {
    KvCacheContract {
        abi: CURRENT_KV_CONTRACT_ABI,
        domains: vec![KvDomainSpec::PagedAttention(PagedAttentionDomainSpec {
            id: CacheDomainId::new(1),
            token_axis: CacheTokenAxis::DecoderTokens,
            layers: vec![PagedAttentionLayerSpec {
                model_layer: 0,
                num_query_heads: 16,
                num_kv_heads: 4,
                key_head_dim: 64,
                value_head_dim: 64,
                attention: AttentionSemantics::Full,
                key_encoding: KeyEncoding::Rotary { rotary_dim: 64 },
            }],
            page_tokens: PageTokenConstraint {
                min: 16,
                preferred: 32,
                max: 64,
                multiple_of: 16,
            },
            storage: KvStorageRequest {
                dtypes: vec![KvStorageDType::F16, KvStorageDType::Bf16],
                allow_quantized: false,
            },
            prefix_semantics: KvPrefixSemantics::CommittedFullPages {
                positions: PositionSemantics::Absolute,
            },
        })],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_contract_has_stable_semantic_fingerprint() {
        let first = test_contract();
        let second = test_contract();
        assert_eq!(first.fingerprint().unwrap(), second.fingerprint().unwrap());

        let mut changed = second;
        let KvDomainSpec::PagedAttention(domain) = &mut changed.domains[0] else {
            unreachable!()
        };
        domain.layers[0].num_kv_heads = 8;
        assert_ne!(first.fingerprint().unwrap(), changed.fingerprint().unwrap());
    }

    #[test]
    fn contract_rejects_duplicate_domains_and_invalid_geometry() {
        let mut duplicate = test_contract();
        duplicate.domains.push(duplicate.domains[0].clone());
        assert!(duplicate.validate().is_err());

        let mut invalid_geometry = test_contract();
        let KvDomainSpec::PagedAttention(domain) = &mut invalid_geometry.domains[0] else {
            unreachable!()
        };
        domain.layers[0].num_query_heads = 10;
        assert!(invalid_geometry.validate().is_err());
    }

    #[test]
    fn contract_rejects_impossible_page_constraint() {
        let mut contract = test_contract();
        let KvDomainSpec::PagedAttention(domain) = &mut contract.domains[0] else {
            unreachable!()
        };
        domain.page_tokens = PageTokenConstraint {
            min: 17,
            preferred: 18,
            max: 19,
            multiple_of: 4,
        };
        assert!(contract.validate().is_err());
    }

    #[test]
    fn opaque_capability_has_no_managed_contract() {
        let capability = CacheCapability::OpaqueModelOwned;
        capability.validate().unwrap();
        assert!(capability.managed_contract().is_none());
    }
}
