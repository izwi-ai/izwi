//! Committed tensor-snapshot sharing for hybrid managed-cache contracts.
//!
//! A contract whose paged-attention domain declares `CommittedPages` and whose
//! tensor domains declare `CommittedSnapshots { interval_steps }` may publish
//! the committed tensor state of one consistency group at interval-aligned
//! cursors, keyed by the digest of the paged prefix page whose end equals that
//! cursor. Because the key IS the verified page-chain digest, a snapshot can
//! only be found through the same exact token chain the paged-attention lookup
//! just authenticated, so tenant salt and model generation binding (DINV-02)
//! are inherited from `KvPrefixNamespace` instead of being re-derived here.
//!
//! Component tensors are shared by handle once committed. Tensor updates
//! replace whole component sets and never mutate committed storage, so the
//! index and any forked sequence can hold the same handles safely.

use std::collections::HashMap;
use std::sync::Arc;

use super::telemetry::ManagedKvTelemetry;
use crate::backends::state::StateComponentValue;
use crate::error::{Error, Result};
use crate::kv::v2::{
    CapabilityStateDescriptorV2, InferenceStateContract, PrefixPolicy, RetainedStateCapability,
    StateDomainId, StateDomainSpec, StateGroupId,
};

/// The strict contract shape that may share committed tensor snapshots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TensorSnapshotPrefixPolicy {
    /// The contract's single paged-attention domain. Its published page chain
    /// authenticates every snapshot boundary.
    pub(crate) paged_domain: StateDomainId,
    /// Every tensor domain of the contract; all share one snapshot cursor.
    pub(crate) tensor_domains: Arc<[StateDomainId]>,
    /// Snapshot cadence in decoder tokens. Boundaries must additionally land
    /// on the resolved paged page size, which the manager checks at bind time.
    pub(crate) interval_steps: u64,
    pub(crate) group: StateGroupId,
}

/// Declared sharing policy for a contract, or `None` when the contract must
/// keep its snapshot declarations admission-inert.
///
/// The shape is deliberately stricter than the contract validator requires:
/// exactly one paged-attention domain (so exactly one fork cursor), every
/// tensor domain of the contract participating with one shared interval (a
/// legacy tensor transaction stages and commits all domains at one cursor),
/// one consistency group, and no append/ring domains whose clocks could
/// diverge from the shared decoder cursor.
pub(crate) fn tensor_snapshot_prefix_policy(
    contract: &InferenceStateContract,
) -> Option<TensorSnapshotPrefixPolicy> {
    let mut paged_domains = Vec::new();
    let mut tensor_domains = Vec::new();
    let mut interval = None;
    for domain in &contract.domains {
        match domain {
            StateDomainSpec::PagedAttention(_) => paged_domains.push(domain.id()),
            StateDomainSpec::Tensor(_) => match domain.prefix_policy() {
                PrefixPolicy::CommittedSnapshots { interval_steps } => {
                    if interval.is_some_and(|existing| existing != *interval_steps) {
                        return None;
                    }
                    interval = Some(*interval_steps);
                    tensor_domains.push(domain.id());
                }
                _ => return None,
            },
            // Append, ring, static-attention, and static-tensor state has no
            // snapshot fork path; keep those contracts admission-inert.
            _ => return None,
        }
    }
    if paged_domains.len() != 1 || tensor_domains.is_empty() {
        return None;
    }
    let paged = contract
        .domains
        .iter()
        .find(|domain| domain.id() == paged_domains[0])?;
    if !matches!(paged.prefix_policy(), PrefixPolicy::CommittedPages { .. }) {
        return None;
    }
    let interval_steps = interval?;
    if interval_steps == 0 {
        return None;
    }
    // One shared consistency group holding exactly the paged domain and every
    // tensor domain; an extra group would commit a divergent clock.
    let group = contract
        .groups
        .iter()
        .find(|group| {
            group.domains.len() == 1 + tensor_domains.len()
                && group.domains.contains(&paged_domains[0])
                && tensor_domains
                    .iter()
                    .all(|domain| group.domains.contains(domain))
        })
        .map(|group| group.id)?;
    if contract.groups.len() != 1 {
        return None;
    }
    Some(TensorSnapshotPrefixPolicy {
        paged_domain: paged_domains[0],
        tensor_domains: tensor_domains.into(),
        interval_steps,
        group,
    })
}

/// Scheduler-side projection of the sharing gate: the declared snapshot
/// interval of a sealed managed contract, when the contract passes the same
/// strict shape check the manager applies. Only Incremental prefill chunks can
/// honor the alignment, so `None` never changes scheduling behavior.
pub(crate) fn declared_snapshot_prefill_interval(
    descriptor: &CapabilityStateDescriptorV2,
) -> Option<u32> {
    match &descriptor.retained {
        RetainedStateCapability::Managed { contract } => tensor_snapshot_prefix_policy(contract)
            .and_then(|policy| u32::try_from(policy.interval_steps).ok())
            .filter(|interval| *interval > 1),
        _ => None,
    }
}

/// Byte size of one committed component set; absent components count as zero.
pub(crate) fn tensor_snapshot_byte_size(components: &[StateComponentValue]) -> u64 {
    components
        .iter()
        .map(|value| {
            value
                .tensor
                .as_ref()
                .map(|tensor| (tensor.elem_count() * tensor.dtype().size_in_bytes()) as u64)
                .unwrap_or(0)
        })
        .sum()
}

/// One committed snapshot: the tensor domains of the shared group at a cursor
/// that ends a published paged page.
#[derive(Debug, Clone)]
pub(crate) struct TensorSnapshotEntry {
    pub(crate) cursor: u64,
    pub(crate) domains: Arc<[(StateDomainId, Arc<[StateComponentValue]>)]>,
    byte_size: u64,
    last_access: u64,
}

/// Bounded LRU index of committed tensor snapshots for one paged arena.
///
/// Bounds are derived from the tensor arena's own authorization envelope: the
/// snapshot cache never holds more sequence rows than the arena's sequence
/// capacity nor more bytes than its authorized committed envelope.
#[derive(Debug, Clone)]
pub(crate) struct TensorStateSnapshotIndex {
    max_entries: usize,
    max_bytes: u64,
    clock: u64,
    total_bytes: u64,
    entries: HashMap<[u8; 32], TensorSnapshotEntry>,
    telemetry: Arc<ManagedKvTelemetry>,
}

impl TensorStateSnapshotIndex {
    pub(crate) fn new(max_entries: usize, max_bytes: u64) -> Self {
        Self::with_telemetry(
            max_entries,
            max_bytes,
            Arc::new(ManagedKvTelemetry::default()),
        )
    }

    pub(crate) fn with_telemetry(
        max_entries: usize,
        max_bytes: u64,
        telemetry: Arc<ManagedKvTelemetry>,
    ) -> Self {
        Self {
            max_entries,
            max_bytes,
            clock: 0,
            total_bytes: 0,
            entries: HashMap::new(),
            telemetry,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    fn tick(&mut self) -> Result<u64> {
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or_else(|| Error::InferenceError("tensor snapshot clock overflow".into()))?;
        Ok(self.clock)
    }

    /// Publish the committed snapshot at `cursor`, keyed by the page-chain
    /// digest whose page ends exactly there. Republishing an identical
    /// boundary is idempotent; a different cursor under a live chain digest is
    /// a programming error and is rejected.
    pub(crate) fn publish(
        &mut self,
        chain_digest: [u8; 32],
        cursor: u64,
        domains: Arc<[(StateDomainId, Arc<[StateComponentValue]>)]>,
        byte_size: u64,
    ) -> Result<bool> {
        if cursor == 0 {
            return Err(Error::InferenceError(
                "tensor snapshot cursor must be positive".into(),
            ));
        }
        let access = self.tick()?;
        if let Some(existing) = self.entries.get_mut(&chain_digest) {
            if existing.cursor != cursor {
                return Err(Error::InferenceError(
                    "tensor snapshot chain digest is bound to a different cursor".into(),
                ));
            }
            existing.last_access = access;
            return Ok(false);
        }
        let evicted_bytes = self.evict_for_insert(byte_size, access)?;
        self.total_bytes = self
            .total_bytes
            .checked_add(byte_size)
            .and_then(|total| total.checked_sub(evicted_bytes))
            .ok_or_else(|| {
                Error::InferenceError("tensor snapshot byte accounting underflowed".into())
            })?;
        self.entries.insert(
            chain_digest,
            TensorSnapshotEntry {
                cursor,
                domains,
                byte_size,
                last_access: access,
            },
        );
        Ok(true)
    }

    /// Return the snapshot published at `expected_cursor` under `chain_digest`,
    /// refreshing its recency. Lookup correctness does not depend on digest
    /// uniqueness because callers pass digests taken from an exact paged page
    /// match rather than from tokens they hashed themselves.
    pub(crate) fn lookup(
        &mut self,
        chain_digest: [u8; 32],
        expected_cursor: u64,
    ) -> Result<Option<Arc<[(StateDomainId, Arc<[StateComponentValue]>)]>>> {
        let access = self.tick()?;
        let Some(entry) = self.entries.get_mut(&chain_digest) else {
            return Ok(None);
        };
        if entry.cursor != expected_cursor {
            return Ok(None);
        }
        entry.last_access = access;
        Ok(Some(entry.domains.clone()))
    }

    /// Evict least-recently-used entries until inserting an entry of
    /// `byte_size` keeps both bounds. Returns the bytes reclaimed. The bounds
    /// are always satisfiable because each entry is smaller than the whole
    /// envelope unless the operator sized the arena below one sequence.
    fn evict_for_insert(&mut self, byte_size: u64, access: u64) -> Result<u64> {
        let mut evicted_bytes = 0_u64;
        let mut evicted_entries = 0_usize;
        while self.entries.len() + 1 > self.max_entries
            || self
                .total_bytes
                .saturating_sub(evicted_bytes)
                .checked_add(byte_size)
                .is_none_or(|projected| projected > self.max_bytes)
        {
            let Some(victim) = self
                .entries
                .iter()
                .filter(|(_, entry)| entry.last_access != access)
                .min_by_key(|(_, entry)| entry.last_access)
                .map(|(digest, _)| *digest)
            else {
                break;
            };
            if let Some(removed) = self.entries.remove(&victim) {
                evicted_bytes += removed.byte_size;
                evicted_entries += 1;
            }
        }
        self.telemetry
            .record_tensor_snapshot_eviction(evicted_entries);
        Ok(evicted_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv::v2::StateComponentId;
    use candle_core::{Device, Tensor};

    fn component(id: u32, values: &[f32]) -> StateComponentValue {
        StateComponentValue {
            component: StateComponentId::new(id),
            tensor: Some(Tensor::from_slice(values, values.len(), &Device::Cpu).unwrap()),
        }
    }

    fn snapshot_domains(values: &[f32]) -> Arc<[(StateDomainId, Arc<[StateComponentValue]>)]> {
        Arc::from([(
            StateDomainId::new(2),
            Arc::from([component(1, values)]) as Arc<[StateComponentValue]>,
        )])
    }

    fn byte_size_of(domains: &Arc<[(StateDomainId, Arc<[StateComponentValue]>)]>) -> u64 {
        domains
            .iter()
            .flat_map(|(_, components)| components.iter())
            .map(|value| {
                value
                    .tensor
                    .as_ref()
                    .map(|tensor| (tensor.elem_count() * tensor.dtype().size_in_bytes()) as u64)
                    .unwrap_or(0)
            })
            .sum()
    }

    fn publish(
        index: &mut TensorStateSnapshotIndex,
        digest: [u8; 32],
        cursor: u64,
        values: &[f32],
    ) -> bool {
        let domains = snapshot_domains(values);
        let bytes = byte_size_of(&domains);
        index.publish(digest, cursor, domains, bytes).unwrap()
    }

    #[test]
    fn policy_accepts_the_strict_hybrid_shape_only() {
        use crate::backends::state::negotiate_state_plan;
        use crate::backends::BackendKind;
        use crate::kv::v2::{
            AttentionMask, AttentionPattern, BoundedShape, CheckpointPolicy, KeyEncoding,
            PageSizeConstraint, PagedAttentionDomainSpec, PagedAttentionLayerSpec, PlacementPolicy,
            PositionSemantics, ShapeAxis, ShapeDimension, ShapeExtent, StateClock,
            StateComponentId, StateDType, StateDomainHeader, StateGroupSpec, StateScope,
            TensorComponentSpec, TensorRole, TensorStateDomainSpec, CURRENT_INFERENCE_STATE_ABI,
        };

        let hybrid = |paged_prefix: PrefixPolicy, tensor_prefix: PrefixPolicy| {
            let header = |id, prefix| StateDomainHeader {
                id,
                scope: StateScope::Retained,
                clock: StateClock::DecoderTokens,
                placement: PlacementPolicy::BackendLocalWithHostOffload,
                prefix,
                checkpoint: CheckpointPolicy::Transactional,
            };
            InferenceStateContract {
                abi: CURRENT_INFERENCE_STATE_ABI,
                domains: vec![
                    StateDomainSpec::PagedAttention(PagedAttentionDomainSpec {
                        header: header(StateDomainId::new(1), paged_prefix),
                        layers: vec![PagedAttentionLayerSpec {
                            model_layer: 1,
                            query_heads: 2,
                            kv_heads: 1,
                            key_head_dim: 8,
                            value_head_dim: 8,
                            pattern: AttentionPattern::Full,
                            mask: AttentionMask::Causal,
                            key_encoding: KeyEncoding::Rotary { rotary_dim: 4 },
                            attention_logit_softcap: None,
                        }],
                        page_size: PageSizeConstraint {
                            min_tokens: 1,
                            preferred_tokens: 4,
                            max_tokens: 4,
                            multiple_of: 1,
                        },
                        accepted_dtypes: vec![StateDType::F32],
                    }),
                    StateDomainSpec::Tensor(TensorStateDomainSpec {
                        header: header(StateDomainId::new(2), tensor_prefix.clone()),
                        components: vec![TensorComponentSpec {
                            id: StateComponentId::new(1),
                            role: TensorRole::RecurrentHidden,
                            shape: BoundedShape {
                                dimensions: vec![ShapeDimension {
                                    axis: ShapeAxis::Hidden,
                                    extent: ShapeExtent::Fixed { value: 4 },
                                }],
                            },
                            accepted_dtypes: vec![StateDType::F32],
                        }],
                    }),
                ],
                groups: vec![StateGroupSpec {
                    id: StateGroupId::new(1),
                    domains: vec![StateDomainId::new(1), StateDomainId::new(2)],
                    prefix_shareable: false,
                }],
            }
        };
        let shared = hybrid(
            PrefixPolicy::CommittedPages {
                positions: PositionSemantics::Absolute,
            },
            PrefixPolicy::CommittedSnapshots { interval_steps: 4 },
        );
        let policy = tensor_snapshot_prefix_policy(&shared).expect("strict shape passes");
        assert_eq!(policy.paged_domain, StateDomainId::new(1));
        assert_eq!(policy.tensor_domains.as_ref(), &[StateDomainId::new(2)]);
        assert_eq!(policy.interval_steps, 4);

        // A disabled tensor domain keeps the whole contract admission-inert.
        let disabled_tensor = hybrid(
            PrefixPolicy::CommittedPages {
                positions: PositionSemantics::Absolute,
            },
            PrefixPolicy::Disabled,
        );
        assert!(tensor_snapshot_prefix_policy(&disabled_tensor).is_none());
        // A disabled paged domain (the MTP-enabled shape) stays inert.
        let disabled_paged = hybrid(
            PrefixPolicy::Disabled,
            PrefixPolicy::CommittedSnapshots { interval_steps: 4 },
        );
        assert!(tensor_snapshot_prefix_policy(&disabled_paged).is_none());

        // The resolved plan for the shared contract keeps a tensor arena.
        let plan = negotiate_state_plan(
            &shared,
            &crate::backends::state::StateBackendPlanRequest {
                backend: BackendKind::Cpu,
                device_ordinal: None,
                page_tokens_hint: None,
                storage_dtype_hint: None,
            },
        )
        .unwrap();
        assert_eq!(plan.non_paged.len(), 1);
    }

    #[test]
    fn publish_is_idempotent_and_rejects_cursor_conflicts() {
        let mut index = TensorStateSnapshotIndex::new(4, 4096);
        assert!(publish(&mut index, [1; 32], 32, &[1.0]));
        assert!(!publish(&mut index, [1; 32], 32, &[1.0]));
        assert!(matches!(
            index.publish(
                [1; 32],
                64,
                snapshot_domains(&[1.0]),
                byte_size_of(&snapshot_domains(&[1.0]))
            ),
            Err(Error::InferenceError(message)) if message.contains("different cursor")
        ));
        assert_eq!(index.len(), 1);
        assert!(index.lookup([1; 32], 32).unwrap().is_some());
        assert!(index.lookup([1; 32], 64).unwrap().is_none());
        assert_eq!(index.total_bytes(), 4);
    }

    #[test]
    fn lru_eviction_holds_entry_and_byte_bounds() {
        // Two entries of 4 bytes fit the 6-byte envelope only one at a time.
        let mut index = TensorStateSnapshotIndex::new(2, 6);
        assert!(publish(&mut index, [1; 32], 32, &[1.0]));
        assert!(publish(&mut index, [2; 32], 64, &[1.0]));
        assert_eq!(index.len(), 1, "byte bound evicted the older entry");
        assert!(index.lookup([1; 32], 32).unwrap().is_none());
        assert!(index.lookup([2; 32], 64).unwrap().is_some());

        // Entry-count bound: three chains, capacity two.
        let mut index = TensorStateSnapshotIndex::new(2, 4096);
        assert!(publish(&mut index, [1; 32], 32, &[1.0]));
        assert!(publish(&mut index, [2; 32], 64, &[1.0]));
        assert!(publish(&mut index, [3; 32], 96, &[1.0]));
        assert_eq!(index.len(), 2);
        assert!(index.lookup([1; 32], 32).unwrap().is_none());
        assert!(index.lookup([3; 32], 96).unwrap().is_some());

        // A lookup refreshes recency and protects the entry from eviction.
        let mut index = TensorStateSnapshotIndex::new(2, 4096);
        assert!(publish(&mut index, [1; 32], 32, &[1.0]));
        assert!(publish(&mut index, [2; 32], 64, &[1.0]));
        assert!(index.lookup([1; 32], 32).unwrap().is_some());
        assert!(publish(&mut index, [3; 32], 96, &[1.0]));
        assert!(index.lookup([1; 32], 32).unwrap().is_some());
        assert!(index.lookup([2; 32], 64).unwrap().is_none());
    }
}
