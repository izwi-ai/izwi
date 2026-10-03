//! Approved deployment pools for gateway-visible deployments.
//!
//! The table is assembled from the effective approval view (operator CLI
//! approvals plus the shared approvals file) and the exact deployment
//! contracts validated against live workers. Pool membership and generation
//! structure are view-authoritative; per-generation replica contracts are
//! learned from validated worker expectations only.
//!
//! The rollout window (DS6) is the only state in which one pool holds two
//! generations of one deployment, and DINV-07 holds by construction: at
//! most one generation is ever `Eligible`. The window view makes the
//! current generation `Eligible` and the successor `PendingCutover`; the
//! successor's first Ready observation completes the cutover in a single
//! table mutation (successor `Eligible`, current `DrainingPrevious`), so
//! there is never an instant with two eligible generations or with no
//! eligible generation while a Ready worker exists. Reverting the view
//! (abort) or advancing it (commit) restores single-generation eligibility
//! without touching any worker. Actual backend and precision remain worker
//! properties so compatible CPU, Metal, and CUDA replicas may share a
//! generation; registry policy still filters them.

use std::collections::BTreeMap;

use izwi_serving_protocol::{
    ArtifactRevision, Capability, DeploymentId, ModelAlias, ModelGeneration, TaskKind,
};

pub use izwi_serving_protocol::{GatewayWorkerApproval, GatewayWorkerApprovalIdentity};

use crate::worker_registry::ApprovedDeployment;

pub const MAX_GATEWAY_DEPLOYMENT_POOLS: usize = 64;
pub const MAX_GATEWAY_REPLICAS_PER_POOL: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationEligibility {
    /// The generation new admissions are routed to.
    Eligible,
    /// A rollout successor approved by the window view; it becomes
    /// `Eligible` at its first Ready observation, in the same table
    /// mutation that marks the current generation `DrainingPrevious`.
    PendingCutover,
    /// The rollout predecessor after cutover: not selectable for new
    /// admissions, still observed while its in-flight work drains.
    DrainingPrevious,
}

/// The contract fields every replica of one generation must share. Backend
/// and precision remain worker properties.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationContract {
    pub artifact_revision: ArtifactRevision,
    pub execution_representation: String,
    pub tokenizer_revision: Option<ArtifactRevision>,
    pub capability: Capability,
}

impl GenerationContract {
    fn of(candidate: &ApprovedDeployment) -> Self {
        Self {
            artifact_revision: candidate.artifact_revision.clone(),
            execution_representation: candidate.execution_representation.clone(),
            tokenizer_revision: candidate.tokenizer_revision.clone(),
            capability: candidate.capability.clone(),
        }
    }

    fn matches(&self, candidate: &ApprovedDeployment) -> bool {
        self.artifact_revision == candidate.artifact_revision
            && self.execution_representation == candidate.execution_representation
            && self.tokenizer_revision == candidate.tokenizer_revision
            && self.capability == candidate.capability
    }
}

/// One approved generation within a pool: its learned replica contract and
/// its rollout eligibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationRecord {
    contract: Option<GenerationContract>,
    replicas: usize,
    pub eligibility: GenerationEligibility,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayDeploymentPool {
    deployment_id: DeploymentId,
    public_model: ModelAlias,
    task: TaskKind,
    generations: BTreeMap<ModelGeneration, GenerationRecord>,
}

impl GatewayDeploymentPool {
    pub fn deployment_id(&self) -> &DeploymentId {
        &self.deployment_id
    }

    /// The one generation new admissions are routed to. Pools hold at most
    /// one `Eligible` generation by construction (DINV-07).
    pub fn eligible_generation(&self) -> Option<ModelGeneration> {
        self.generations
            .iter()
            .find(|(_, record)| record.eligibility == GenerationEligibility::Eligible)
            .map(|(generation, _)| *generation)
    }

    fn derive_eligibility(&mut self) {
        let generations = &mut self.generations;
        if generations.len() <= 1 {
            for record in generations.values_mut() {
                record.eligibility = GenerationEligibility::Eligible;
            }
            return;
        }
        // Rollout window structure: the lower generation is the current one,
        // the higher is the pending successor.
        for (index, (_, record)) in generations.iter_mut().enumerate() {
            record.eligibility = if index == 0 {
                GenerationEligibility::Eligible
            } else {
                GenerationEligibility::PendingCutover
            };
        }
    }

    fn replicas(&self) -> usize {
        self.generations
            .values()
            .map(|record| record.replicas)
            .sum()
    }

    #[cfg(test)]
    pub fn generations(&self) -> &BTreeMap<ModelGeneration, GenerationRecord> {
        &self.generations
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GatewayDeploymentTable {
    pools: Vec<GatewayDeploymentPool>,
}

/// One pool generation identified by its full routing key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationKey {
    pub deployment_id: DeploymentId,
    pub task: TaskKind,
    pub public_model: ModelAlias,
    pub model_generation: ModelGeneration,
}

/// What adopting an approval view changed in the table.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ApprovalViewOutcome {
    pub added: Vec<GenerationKey>,
    pub removed: Vec<GenerationKey>,
}

impl GatewayDeploymentTable {
    /// Approves one validated replica. The replica's generation must already
    /// exist in its pool's view-derived structure (a bare second generation
    /// is rejected — that is the DINV-07 guard against two admission-eligible
    /// generations outside an explicit rollout window); the call fills the
    /// generation's learned contract on first sight and otherwise requires
    /// exact contract equality.
    pub fn approve_replica(
        &mut self,
        candidate: ApprovedDeployment,
    ) -> Result<(), GatewayDeploymentTableError> {
        if candidate.capability.task != candidate.task {
            return Err(GatewayDeploymentTableError::CapabilityTaskMismatch);
        }

        if let Some(pool) = self
            .pools
            .iter_mut()
            .find(|pool| pool.task == candidate.task && pool.public_model == candidate.public_model)
        {
            if pool.deployment_id != candidate.deployment_id {
                return Err(GatewayDeploymentTableError::ConflictingDeploymentId);
            }
            let pool_replicas = pool.replicas();
            let Some(record) = pool.generations.get_mut(&candidate.model_generation) else {
                return Err(GatewayDeploymentTableError::ConflictingModelGeneration);
            };
            match &record.contract {
                Some(contract) => {
                    if contract.capability != candidate.capability {
                        return Err(GatewayDeploymentTableError::ConflictingCapability);
                    }
                    if !contract.matches(&candidate) {
                        return Err(GatewayDeploymentTableError::ConflictingDeploymentContract);
                    }
                }
                None => record.contract = Some(GenerationContract::of(&candidate)),
            }
            if pool_replicas >= MAX_GATEWAY_REPLICAS_PER_POOL {
                return Err(GatewayDeploymentTableError::ReplicaLimitReached);
            }
            record.replicas += 1;
            return Ok(());
        }

        if self
            .pools
            .iter()
            .any(|pool| pool.deployment_id == candidate.deployment_id)
        {
            return Err(GatewayDeploymentTableError::ConflictingDeploymentKey);
        }
        if self.pools.len() >= MAX_GATEWAY_DEPLOYMENT_POOLS {
            return Err(GatewayDeploymentTableError::PoolLimitReached);
        }
        let mut generations = BTreeMap::new();
        generations.insert(
            candidate.model_generation,
            GenerationRecord {
                contract: Some(GenerationContract::of(&candidate)),
                replicas: 1,
                eligibility: GenerationEligibility::Eligible,
            },
        );
        self.pools.push(GatewayDeploymentPool {
            deployment_id: candidate.deployment_id,
            public_model: candidate.public_model,
            task: candidate.task,
            generations,
        });
        Ok(())
    }

    /// Adopts an effective approval view: the pool/generation structure is
    /// recomputed from the view while learned contracts and an already
    /// completed cutover are preserved for surviving generations. Pools or
    /// generations absent from the view are removed (abort/commit), a second
    /// generation opens the rollout window per the DINV-07 rule, and a third
    /// generation of one deployment is rejected. On error the table is
    /// unchanged so the previous view keeps driving admission.
    pub fn apply_view(
        &mut self,
        approvals: &[GatewayWorkerApproval],
    ) -> Result<ApprovalViewOutcome, GatewayDeploymentTableError> {
        // Group the view by pool key, enforcing the deployment-key identity
        // rules the replica path enforces.
        let mut groups: Vec<(
            TaskKind,
            ModelAlias,
            DeploymentId,
            BTreeMap<ModelGeneration, ()>,
        )> = Vec::new();
        for approval in approvals {
            if let Some((_, _, deployment_id, generations)) =
                groups.iter_mut().find(|(task, public_model, _, _)| {
                    *task == approval.task && *public_model == approval.public_model
                })
            {
                if *deployment_id != approval.deployment_id {
                    return Err(GatewayDeploymentTableError::ConflictingDeploymentId);
                }
                generations.insert(approval.model_generation, ());
                continue;
            }
            if groups
                .iter()
                .any(|(_, _, deployment_id, _)| *deployment_id == approval.deployment_id)
            {
                return Err(GatewayDeploymentTableError::ConflictingDeploymentKey);
            }
            if groups.len() >= MAX_GATEWAY_DEPLOYMENT_POOLS {
                return Err(GatewayDeploymentTableError::PoolLimitReached);
            }
            let mut generations = BTreeMap::new();
            generations.insert(approval.model_generation, ());
            groups.push((
                approval.task,
                approval.public_model.clone(),
                approval.deployment_id.clone(),
                generations,
            ));
        }

        let mut next_pools: Vec<GatewayDeploymentPool> = Vec::with_capacity(groups.len());
        for (task, public_model, deployment_id, view_generations) in groups {
            if view_generations.len() > 2 {
                return Err(GatewayDeploymentTableError::TooManyGenerations);
            }
            let existing = self
                .pools
                .iter()
                .find(|pool| pool.task == task && pool.public_model == public_model)
                .cloned();
            let (kept_generations, structure_changed) = match existing {
                Some(mut pool) => {
                    if pool.deployment_id != deployment_id {
                        return Err(GatewayDeploymentTableError::ConflictingDeploymentId);
                    }
                    let same_set = pool
                        .generations
                        .keys()
                        .copied()
                        .collect::<std::collections::BTreeSet<_>>()
                        == view_generations
                            .keys()
                            .copied()
                            .collect::<std::collections::BTreeSet<_>>();
                    pool.generations
                        .retain(|generation, _| view_generations.contains_key(generation));
                    (pool.generations, !same_set)
                }
                None => (BTreeMap::new(), true),
            };
            let mut generations = kept_generations;
            for generation in view_generations.keys() {
                generations
                    .entry(*generation)
                    .or_insert_with(|| GenerationRecord {
                        contract: None,
                        replicas: 0,
                        eligibility: GenerationEligibility::Eligible,
                    });
            }
            let mut pool = GatewayDeploymentPool {
                deployment_id,
                public_model,
                task,
                generations,
            };
            if structure_changed {
                pool.derive_eligibility();
            }
            next_pools.push(pool);
        }

        let outcome = diff_generations(&self.pools, &next_pools);
        self.pools = next_pools;
        Ok(outcome)
    }

    /// Completes a pending rollout cutover when a worker of the successor
    /// generation is observed Ready: the successor becomes the one eligible
    /// generation and the current generation becomes `DrainingPrevious` in
    /// the same mutation. Returns whether this call completed a cutover.
    pub fn observe_generation_ready(
        &mut self,
        task: TaskKind,
        public_model: &ModelAlias,
        deployment_id: &DeploymentId,
        generation: ModelGeneration,
    ) -> bool {
        let Some(pool) = self
            .pools
            .iter_mut()
            .find(|pool| pool.task == task && pool.public_model == *public_model)
        else {
            return false;
        };
        if pool.deployment_id != *deployment_id {
            return false;
        }
        let Some(successor) = pool.generations.get_mut(&generation) else {
            return false;
        };
        if successor.eligibility != GenerationEligibility::PendingCutover {
            return false;
        }
        successor.eligibility = GenerationEligibility::Eligible;
        for (other, record) in pool.generations.iter_mut() {
            if *other != generation {
                record.eligibility = GenerationEligibility::DrainingPrevious;
            }
        }
        true
    }

    pub fn select(
        &self,
        task: TaskKind,
        public_model: &ModelAlias,
    ) -> Option<&GatewayDeploymentPool> {
        self.pools
            .iter()
            .find(|pool| pool.task == task && pool.public_model == *public_model)
    }

    /// The one generation of a pool that new admissions may be routed to.
    pub fn eligible_generation(
        &self,
        task: TaskKind,
        public_model: &ModelAlias,
    ) -> Option<ModelGeneration> {
        self.select(task, public_model)
            .and_then(GatewayDeploymentPool::eligible_generation)
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.pools.len()
    }
}

fn diff_generations(
    previous: &[GatewayDeploymentPool],
    next: &[GatewayDeploymentPool],
) -> ApprovalViewOutcome {
    let key = |pool: &GatewayDeploymentPool, generation: ModelGeneration| GenerationKey {
        deployment_id: pool.deployment_id.clone(),
        task: pool.task,
        public_model: pool.public_model.clone(),
        model_generation: generation,
    };
    let mut outcome = ApprovalViewOutcome::default();
    for pool in previous {
        for generation in pool.generations.keys() {
            let still_present = next
                .iter()
                .find(|next_pool| {
                    next_pool.task == pool.task && next_pool.public_model == pool.public_model
                })
                .is_some_and(|next_pool| next_pool.generations.contains_key(generation));
            if !still_present {
                outcome.removed.push(key(pool, *generation));
            }
        }
    }
    for pool in next {
        for generation in pool.generations.keys() {
            let was_present = previous
                .iter()
                .find(|previous_pool| {
                    previous_pool.task == pool.task
                        && previous_pool.public_model == pool.public_model
                })
                .is_some_and(|previous_pool| previous_pool.generations.contains_key(generation));
            if !was_present {
                outcome.added.push(key(pool, *generation));
            }
        }
    }
    outcome
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GatewayDeploymentTableError {
    #[error("gateway deployment capability task differs from its deployment task")]
    CapabilityTaskMismatch,
    #[error("a gateway deployment key resolves to conflicting deployment identifiers")]
    ConflictingDeploymentId,
    #[error("a gateway deployment key resolves to conflicting model generations")]
    ConflictingModelGeneration,
    #[error("a gateway deployment key resolves to conflicting capabilities")]
    ConflictingCapability,
    #[error("a gateway deployment key resolves to conflicting immutable contracts")]
    ConflictingDeploymentContract,
    #[error("a deployment identifier is assigned to more than one task/model key")]
    ConflictingDeploymentKey,
    #[error(
        "a gateway deployment rollout window may hold at most two generations of one deployment"
    )]
    TooManyGenerations,
    #[error("gateway deployment table has reached its pool limit")]
    PoolLimitReached,
    #[error("gateway deployment pool has reached its replica limit")]
    ReplicaLimitReached,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use izwi_serving_protocol::{
        BackendKind, CancellationBehavior, Capability, InputFormat, OutputFormat,
    };

    use super::*;

    fn approval(deployment_id: &str, public_model: &str, task: TaskKind) -> ApprovedDeployment {
        ApprovedDeployment {
            deployment_id: DeploymentId::new(deployment_id).unwrap(),
            public_model: ModelAlias::new(public_model).unwrap(),
            artifact_revision: ArtifactRevision::new("artifact-v1").unwrap(),
            model_generation: ModelGeneration::new(1).unwrap(),
            task,
            backend: BackendKind::Cpu,
            precision: "f32".into(),
            execution_representation: "test".into(),
            tokenizer_revision: None,
            capability: Capability {
                task,
                streaming: task == TaskKind::Chat,
                realtime: false,
                cancellation: CancellationBehavior::Cooperative,
                accepted_input_formats: BTreeSet::from([match task {
                    TaskKind::Chat => InputFormat::ChatMessages,
                    TaskKind::TextToSpeech => InputFormat::ChatMessages,
                    TaskKind::SpeechToText => InputFormat::EncodedAudio,
                }]),
                output_formats: BTreeSet::from([match task {
                    TaskKind::Chat | TaskKind::SpeechToText => OutputFormat::Text,
                    TaskKind::TextToSpeech => OutputFormat::EncodedAudio,
                }]),
                max_input_bytes: 4096,
                max_context_tokens: (task == TaskKind::Chat).then_some(4096),
                max_output_tokens: (task == TaskKind::Chat).then_some(128),
            },
        }
    }

    fn view_line(
        endpoint: &str,
        task: &str,
        public_model: &str,
        deployment: &str,
        generation: u64,
    ) -> GatewayWorkerApproval {
        format!("{endpoint}|{task}|{public_model}|{deployment}|{generation}")
            .parse()
            .unwrap()
    }

    #[test]
    fn replicas_require_one_exact_deployment_contract() {
        let chat = approval("chat-prod", "chat-model", TaskKind::Chat);
        let mut table = GatewayDeploymentTable::default();
        table.approve_replica(chat.clone()).unwrap();
        table.approve_replica(chat.clone()).unwrap();
        assert_eq!(table.len(), 1);
        assert_eq!(
            table
                .select(TaskKind::Chat, &chat.public_model)
                .unwrap()
                .replicas(),
            2
        );

        let mut metal_replica = chat.clone();
        metal_replica.backend = BackendKind::Metal;
        metal_replica.precision = "f16".into();
        table.approve_replica(metal_replica).unwrap();
        assert_eq!(
            table
                .select(TaskKind::Chat, &chat.public_model)
                .unwrap()
                .replicas(),
            3
        );

        let mut wrong_id = chat.clone();
        wrong_id.deployment_id = DeploymentId::new("chat-other").unwrap();
        assert_eq!(
            table.approve_replica(wrong_id).unwrap_err(),
            GatewayDeploymentTableError::ConflictingDeploymentId
        );

        let mut wrong_generation = chat.clone();
        wrong_generation.model_generation = ModelGeneration::new(2).unwrap();
        assert_eq!(
            table.approve_replica(wrong_generation).unwrap_err(),
            GatewayDeploymentTableError::ConflictingModelGeneration
        );

        let mut wrong_capability = chat.clone();
        wrong_capability.capability.max_output_tokens = Some(64);
        assert_eq!(
            table.approve_replica(wrong_capability).unwrap_err(),
            GatewayDeploymentTableError::ConflictingCapability
        );
    }

    #[test]
    fn task_and_deployment_identity_conflicts_fail_closed() {
        let chat = approval("shared-id", "shared-model", TaskKind::Chat);
        let mut table = GatewayDeploymentTable::default();
        table.approve_replica(chat.clone()).unwrap();

        let mut mismatched_capability = chat.clone();
        mismatched_capability.capability.task = TaskKind::TextToSpeech;
        assert_eq!(
            table.approve_replica(mismatched_capability).unwrap_err(),
            GatewayDeploymentTableError::CapabilityTaskMismatch
        );

        let tts = approval("shared-id", "shared-model", TaskKind::TextToSpeech);
        assert_eq!(
            table.approve_replica(tts).unwrap_err(),
            GatewayDeploymentTableError::ConflictingDeploymentKey
        );
    }

    #[test]
    fn disjoint_task_pools_are_selected_only_by_task_and_alias() {
        let chat = approval("chat-prod", "shared-model", TaskKind::Chat);
        let tts = approval("tts-prod", "shared-model", TaskKind::TextToSpeech);
        let asr = approval("asr-prod", "speech-model", TaskKind::SpeechToText);
        let mut table = GatewayDeploymentTable::default();
        table.approve_replica(chat.clone()).unwrap();
        table.approve_replica(tts.clone()).unwrap();
        table.approve_replica(asr.clone()).unwrap();

        assert_eq!(table.len(), 3);
        assert_eq!(
            table
                .select(TaskKind::Chat, &chat.public_model)
                .unwrap()
                .deployment_id()
                .as_str(),
            chat.deployment_id.as_str()
        );
        assert_eq!(
            table
                .select(TaskKind::TextToSpeech, &tts.public_model)
                .unwrap()
                .deployment_id()
                .as_str(),
            tts.deployment_id.as_str()
        );
        assert!(table
            .select(TaskKind::SpeechToText, &chat.public_model)
            .is_none());
    }

    #[test]
    fn window_view_holds_two_generations_with_exactly_one_eligible() {
        let mut table = GatewayDeploymentTable::default();
        table
            .apply_view(&[view_line(
                "http://w1:9",
                "chat",
                "chat-model",
                "chat-prod",
                1,
            )])
            .unwrap();
        assert_eq!(
            table.eligible_generation(TaskKind::Chat, &ModelAlias::new("chat-model").unwrap()),
            Some(ModelGeneration::new(1).unwrap())
        );

        // The window view approves the successor generation alongside the
        // current one: exactly one eligible generation (DINV-07).
        let outcome = table
            .apply_view(&[
                view_line("http://w1:9", "chat", "chat-model", "chat-prod", 1),
                view_line("http://w2:9", "chat", "chat-model", "chat-prod", 2),
            ])
            .unwrap();
        assert_eq!(outcome.added.len(), 1);
        assert_eq!(
            outcome.added[0].model_generation,
            ModelGeneration::new(2).unwrap()
        );
        assert_eq!(
            table.eligible_generation(TaskKind::Chat, &ModelAlias::new("chat-model").unwrap()),
            Some(ModelGeneration::new(1).unwrap())
        );

        // Replicas of both generations fill their learned contracts.
        let mut old = approval("chat-prod", "chat-model", TaskKind::Chat);
        old.model_generation = ModelGeneration::new(1).unwrap();
        let mut new = approval("chat-prod", "chat-model", TaskKind::Chat);
        new.model_generation = ModelGeneration::new(2).unwrap();
        new.artifact_revision = ArtifactRevision::new("artifact-v2").unwrap();
        table.approve_replica(old).unwrap();
        table.approve_replica(new).unwrap();

        // The successor's first Ready observation completes the cutover in
        // one mutation: never two eligible generations, never none.
        assert!(table.observe_generation_ready(
            TaskKind::Chat,
            &ModelAlias::new("chat-model").unwrap(),
            &DeploymentId::new("chat-prod").unwrap(),
            ModelGeneration::new(2).unwrap(),
        ));
        assert_eq!(
            table.eligible_generation(TaskKind::Chat, &ModelAlias::new("chat-model").unwrap()),
            Some(ModelGeneration::new(2).unwrap())
        );
        let pool = table
            .select(TaskKind::Chat, &ModelAlias::new("chat-model").unwrap())
            .unwrap();
        assert_eq!(
            pool.generations()[&ModelGeneration::new(1).unwrap()].eligibility,
            GenerationEligibility::DrainingPrevious
        );
        assert_eq!(pool.replicas(), 2);
    }

    #[test]
    fn abort_view_restores_previous_eligibility_and_commit_advances_it() {
        let mut table = GatewayDeploymentTable::default();
        table
            .apply_view(&[
                view_line("http://w1:9", "chat", "chat-model", "chat-prod", 1),
                view_line("http://w2:9", "chat", "chat-model", "chat-prod", 2),
            ])
            .unwrap();
        let mut new = approval("chat-prod", "chat-model", TaskKind::Chat);
        new.model_generation = ModelGeneration::new(2).unwrap();
        table.approve_replica(new).unwrap();
        assert!(table.observe_generation_ready(
            TaskKind::Chat,
            &ModelAlias::new("chat-model").unwrap(),
            &DeploymentId::new("chat-prod").unwrap(),
            ModelGeneration::new(2).unwrap(),
        ));

        // Abort: the restored single-generation view makes the untouched
        // current generation eligible again.
        table
            .apply_view(&[view_line(
                "http://w1:9",
                "chat",
                "chat-model",
                "chat-prod",
                1,
            )])
            .unwrap();
        assert_eq!(
            table.eligible_generation(TaskKind::Chat, &ModelAlias::new("chat-model").unwrap()),
            Some(ModelGeneration::new(1).unwrap())
        );

        // Commit: the successor-only view removes the drained predecessor.
        let outcome = table
            .apply_view(&[view_line(
                "http://w2:9",
                "chat",
                "chat-model",
                "chat-prod",
                2,
            )])
            .unwrap();
        assert_eq!(outcome.removed.len(), 1);
        assert_eq!(
            outcome.removed[0].model_generation,
            ModelGeneration::new(1).unwrap()
        );
        assert_eq!(
            table.eligible_generation(TaskKind::Chat, &ModelAlias::new("chat-model").unwrap()),
            Some(ModelGeneration::new(2).unwrap())
        );
    }

    #[test]
    fn same_structure_view_preserves_completed_cutover_and_contracts() {
        let mut table = GatewayDeploymentTable::default();
        table
            .apply_view(&[
                view_line("http://w1:9", "chat", "chat-model", "chat-prod", 1),
                view_line("http://w2:9", "chat", "chat-model", "chat-prod", 2),
            ])
            .unwrap();
        let mut new = approval("chat-prod", "chat-model", TaskKind::Chat);
        new.model_generation = ModelGeneration::new(2).unwrap();
        table.approve_replica(new).unwrap();
        assert!(table.observe_generation_ready(
            TaskKind::Chat,
            &ModelAlias::new("chat-model").unwrap(),
            &DeploymentId::new("chat-prod").unwrap(),
            ModelGeneration::new(2).unwrap(),
        ));

        // Re-adopting the identical window structure (e.g. an unrelated pool
        // changed elsewhere in the file) must not reset the cutover.
        table
            .apply_view(&[
                view_line("http://w1:9", "chat", "chat-model", "chat-prod", 1),
                view_line("http://w2:9", "chat", "chat-model", "chat-prod", 2),
            ])
            .unwrap();
        let pool = table
            .select(TaskKind::Chat, &ModelAlias::new("chat-model").unwrap())
            .unwrap();
        assert_eq!(
            pool.eligible_generation(),
            Some(ModelGeneration::new(2).unwrap())
        );
        assert_eq!(
            pool.generations()[&ModelGeneration::new(1).unwrap()].eligibility,
            GenerationEligibility::DrainingPrevious
        );
    }

    #[test]
    fn invalid_views_fail_closed_and_leave_the_table_unchanged() {
        let mut table = GatewayDeploymentTable::default();
        table
            .apply_view(&[view_line(
                "http://w1:9",
                "chat",
                "chat-model",
                "chat-prod",
                1,
            )])
            .unwrap();
        let before = table.clone();

        // Three generations of one deployment never fit the window rule.
        assert_eq!(
            table
                .apply_view(&[
                    view_line("http://w1:9", "chat", "chat-model", "chat-prod", 1),
                    view_line("http://w2:9", "chat", "chat-model", "chat-prod", 2),
                    view_line("http://w3:9", "chat", "chat-model", "chat-prod", 3),
                ])
                .unwrap_err(),
            GatewayDeploymentTableError::TooManyGenerations
        );
        // One deployment key may not resolve to two task/model pools.
        assert_eq!(
            table
                .apply_view(&[
                    view_line("http://w1:9", "chat", "chat-model", "chat-prod", 1),
                    view_line(
                        "http://w4:9",
                        "text_to_speech",
                        "chat-model",
                        "chat-prod",
                        4
                    ),
                ])
                .unwrap_err(),
            GatewayDeploymentTableError::ConflictingDeploymentKey
        );
        // A pool's deployment identity may not change under the same key.
        assert_eq!(
            table
                .apply_view(&[view_line(
                    "http://w1:9",
                    "chat",
                    "chat-model",
                    "chat-other",
                    1
                )])
                .unwrap_err(),
            GatewayDeploymentTableError::ConflictingDeploymentId
        );
        assert_eq!(table, before);
    }

    #[test]
    fn pools_absent_from_the_view_are_removed() {
        let mut table = GatewayDeploymentTable::default();
        table
            .apply_view(&[
                view_line("http://w1:9", "chat", "chat-model", "chat-prod", 1),
                view_line("http://w2:9", "speech_to_text", "asr-model", "asr-prod", 1),
            ])
            .unwrap();
        assert_eq!(table.len(), 2);
        table
            .apply_view(&[view_line(
                "http://w1:9",
                "chat",
                "chat-model",
                "chat-prod",
                1,
            )])
            .unwrap();
        assert_eq!(table.len(), 1);
        assert!(table
            .select(
                TaskKind::SpeechToText,
                &ModelAlias::new("asr-model").unwrap()
            )
            .is_none());
    }
}
