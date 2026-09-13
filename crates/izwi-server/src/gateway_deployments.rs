//! Bounded startup approvals for gateway-visible deployment pools.
//!
//! The table is assembled only from operator-configured endpoint bindings and
//! the exact deployment contracts validated during gateway startup. Runtime
//! status refreshes cannot add a deployment or change a pool contract. Actual
//! backend and precision remain worker properties so compatible CPU, Metal,
//! and CUDA replicas may share a pool; registry policy still filters them.

use std::str::FromStr;

use izwi_serving_protocol::{DeploymentId, ModelAlias, ModelGeneration, TaskKind};

use crate::worker_registry::ApprovedDeployment;

pub const MAX_GATEWAY_DEPLOYMENT_POOLS: usize = 64;
pub const MAX_GATEWAY_REPLICAS_PER_POOL: usize = 256;
const MAX_GATEWAY_WORKER_APPROVAL_BYTES: usize = 4 * 1024;
const MAX_GATEWAY_ENDPOINT_BYTES: usize = 2 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayWorkerApproval {
    pub endpoint: String,
    pub task: TaskKind,
    pub public_model: ModelAlias,
    pub deployment_id: DeploymentId,
    pub model_generation: ModelGeneration,
}

impl FromStr for GatewayWorkerApproval {
    type Err = GatewayDeploymentTableError;

    /// Parses `URL|TASK|PUBLIC_MODEL|DEPLOYMENT_ID|MODEL_GENERATION`.
    ///
    /// The endpoint is subsequently validated by `WorkerClient`; this parser
    /// only applies retention bounds and parses the statically pinned route.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.len() > MAX_GATEWAY_WORKER_APPROVAL_BYTES {
            return Err(GatewayDeploymentTableError::ApprovalTooLong);
        }
        let fields = value.split('|').map(str::trim).collect::<Vec<_>>();
        if fields.len() != 5 || fields.iter().any(|field| field.is_empty()) {
            return Err(GatewayDeploymentTableError::InvalidApprovalSyntax);
        }
        if fields[0].len() > MAX_GATEWAY_ENDPOINT_BYTES {
            return Err(GatewayDeploymentTableError::EndpointTooLong);
        }
        let task = match fields[1] {
            "chat" => TaskKind::Chat,
            "text_to_speech" => TaskKind::TextToSpeech,
            "speech_to_text" => TaskKind::SpeechToText,
            _ => return Err(GatewayDeploymentTableError::UnknownTask),
        };
        let generation = fields[4]
            .parse::<u64>()
            .map_err(|_| GatewayDeploymentTableError::InvalidGeneration)
            .and_then(|generation| {
                ModelGeneration::new(generation)
                    .map_err(|_| GatewayDeploymentTableError::InvalidGeneration)
            })?;
        Ok(Self {
            endpoint: fields[0].to_string(),
            task,
            public_model: ModelAlias::new(fields[2])
                .map_err(|_| GatewayDeploymentTableError::InvalidPublicModel)?,
            deployment_id: DeploymentId::new(fields[3])
                .map_err(|_| GatewayDeploymentTableError::InvalidDeploymentId)?,
            model_generation: generation,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayDeploymentPool {
    deployment_id: DeploymentId,
    public_model: ModelAlias,
    task: TaskKind,
    model_generation: ModelGeneration,
    artifact_revision: izwi_serving_protocol::ArtifactRevision,
    execution_representation: String,
    tokenizer_revision: Option<izwi_serving_protocol::ArtifactRevision>,
    capability: izwi_serving_protocol::Capability,
    replicas: usize,
}

impl GatewayDeploymentPool {
    pub fn deployment_id(&self) -> &DeploymentId {
        &self.deployment_id
    }

    #[cfg(test)]
    pub const fn replicas(&self) -> usize {
        self.replicas
    }
}

#[derive(Debug, Clone, Default)]
pub struct GatewayDeploymentTable {
    pools: Vec<GatewayDeploymentPool>,
}

impl GatewayDeploymentTable {
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
            if pool.model_generation != candidate.model_generation {
                return Err(GatewayDeploymentTableError::ConflictingModelGeneration);
            }
            if pool.capability != candidate.capability {
                return Err(GatewayDeploymentTableError::ConflictingCapability);
            }
            if pool.artifact_revision != candidate.artifact_revision
                || pool.execution_representation != candidate.execution_representation
                || pool.tokenizer_revision != candidate.tokenizer_revision
            {
                return Err(GatewayDeploymentTableError::ConflictingDeploymentContract);
            }
            if pool.replicas >= MAX_GATEWAY_REPLICAS_PER_POOL {
                return Err(GatewayDeploymentTableError::ReplicaLimitReached);
            }
            pool.replicas += 1;
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
        self.pools.push(GatewayDeploymentPool {
            deployment_id: candidate.deployment_id,
            public_model: candidate.public_model,
            task: candidate.task,
            model_generation: candidate.model_generation,
            artifact_revision: candidate.artifact_revision,
            execution_representation: candidate.execution_representation,
            tokenizer_revision: candidate.tokenizer_revision,
            capability: candidate.capability,
            replicas: 1,
        });
        Ok(())
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

    #[cfg(test)]
    fn len(&self) -> usize {
        self.pools.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum GatewayDeploymentTableError {
    #[error("gateway worker approval exceeds its encoded size limit")]
    ApprovalTooLong,
    #[error(
        "gateway worker approval must be URL|TASK|PUBLIC_MODEL|DEPLOYMENT_ID|MODEL_GENERATION"
    )]
    InvalidApprovalSyntax,
    #[error("gateway worker approval endpoint exceeds its size limit")]
    EndpointTooLong,
    #[error("gateway worker approval task must be chat, text_to_speech, or speech_to_text")]
    UnknownTask,
    #[error("gateway worker approval has an invalid public model alias")]
    InvalidPublicModel,
    #[error("gateway worker approval has an invalid deployment identifier")]
    InvalidDeploymentId,
    #[error("gateway worker approval model generation must be a non-zero integer")]
    InvalidGeneration,
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
    #[error("gateway deployment table has reached its pool limit")]
    PoolLimitReached,
    #[error("gateway deployment pool has reached its replica limit")]
    ReplicaLimitReached,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use izwi_serving_protocol::{
        ArtifactRevision, BackendKind, CancellationBehavior, Capability, InputFormat, OutputFormat,
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
    fn worker_approval_parser_is_bounded_and_typed() {
        let parsed: GatewayWorkerApproval = "http://127.0.0.1:19091|chat|chat-model|chat-prod|7"
            .parse()
            .unwrap();
        assert_eq!(parsed.task, TaskKind::Chat);
        assert_eq!(parsed.public_model.as_str(), "chat-model");
        assert_eq!(parsed.model_generation.get(), 7);
        assert_eq!(
            format!(
                "http://127.0.0.1:19091|chat|chat-model|chat-prod|1{}",
                "x".repeat(MAX_GATEWAY_WORKER_APPROVAL_BYTES)
            )
            .parse::<GatewayWorkerApproval>()
            .unwrap_err(),
            GatewayDeploymentTableError::ApprovalTooLong
        );
    }
}
