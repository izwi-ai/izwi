use serde::{Deserialize, Serialize};

pub const STAGE_RESOURCE_HINTS_VERSION: u16 = 1;
pub const WORKER_REGISTRATION_VERSION: u16 = 1;
pub const WORKER_HEARTBEAT_DETAILS_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum QueueClass {
    Batch,
    InteractiveAsr,
    BatchAsr,
    LongFormAsr,
    BatchTts,
    StreamingTts,
    Diarization,
    Export,
    Evaluation,
}

impl QueueClass {
    pub const fn as_db_value(self) -> &'static str {
        match self {
            Self::Batch => "batch",
            Self::InteractiveAsr => "interactive_asr",
            Self::BatchAsr => "batch_asr",
            Self::LongFormAsr => "long_form_asr",
            Self::BatchTts => "batch_tts",
            Self::StreamingTts => "streaming_tts",
            Self::Diarization => "diarization",
            Self::Export => "export",
            Self::Evaluation => "evaluation",
        }
    }

    pub fn from_db_value(value: &str) -> Option<Self> {
        match value {
            "batch" => Some(Self::Batch),
            "interactive_asr" => Some(Self::InteractiveAsr),
            "batch_asr" => Some(Self::BatchAsr),
            "long_form_asr" => Some(Self::LongFormAsr),
            "batch_tts" => Some(Self::BatchTts),
            "streaming_tts" => Some(Self::StreamingTts),
            "diarization" => Some(Self::Diarization),
            "export" => Some(Self::Export),
            "evaluation" => Some(Self::Evaluation),
            _ => None,
        }
    }
}

impl Default for QueueClass {
    fn default() -> Self {
        Self::Batch
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ResourceTarget {
    Any,
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeBackendClass {
    Cpu,
    Metal,
    Cuda,
}

impl RuntimeBackendClass {
    pub const fn as_db_value(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum DeviceClass {
    Cpu,
    AppleGpu,
    NvidiaGpu,
}

impl DeviceClass {
    pub const fn as_db_value(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::AppleGpu => "apple_gpu",
            Self::NvidiaGpu => "nvidia_gpu",
        }
    }
}

impl ResourceTarget {
    pub const fn as_db_value(self) -> &'static str {
        match self {
            Self::Any => "any",
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }

    pub fn from_db_value(value: &str) -> Option<Self> {
        match value {
            "any" => Some(Self::Any),
            "cpu" => Some(Self::Cpu),
            "gpu" => Some(Self::Gpu),
            _ => None,
        }
    }
}

impl Default for ResourceTarget {
    fn default() -> Self {
        Self::Any
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct StageResourceHints {
    pub version: u16,
    pub target: ResourceTarget,
    pub backend: Option<RuntimeBackendClass>,
    pub device_class: Option<DeviceClass>,
    pub min_memory_bytes: Option<u64>,
    pub estimated_memory_bytes: Option<u64>,
    pub estimated_duration_ms: Option<u64>,
    pub concurrency_weight: u32,
}

impl Default for StageResourceHints {
    fn default() -> Self {
        Self {
            version: STAGE_RESOURCE_HINTS_VERSION,
            target: ResourceTarget::Any,
            backend: None,
            device_class: None,
            min_memory_bytes: None,
            estimated_memory_bytes: None,
            estimated_duration_ms: None,
            concurrency_weight: 1,
        }
    }
}

impl StageResourceHints {
    pub fn normalized(mut self) -> Self {
        self.version = STAGE_RESOURCE_HINTS_VERSION;
        self.concurrency_weight = self.concurrency_weight.max(1);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct WorkerResourceCapacity {
    pub version: u16,
    pub targets: Vec<ResourceTarget>,
    pub backends: Vec<RuntimeBackendClass>,
    pub device_classes: Vec<DeviceClass>,
    pub memory_bytes: Option<u64>,
    pub concurrency_slots: u32,
}

impl Default for WorkerResourceCapacity {
    fn default() -> Self {
        Self {
            version: WORKER_REGISTRATION_VERSION,
            targets: vec![ResourceTarget::Any],
            backends: Vec::new(),
            device_classes: Vec::new(),
            memory_bytes: None,
            concurrency_slots: 1,
        }
    }
}

impl WorkerResourceCapacity {
    pub fn supports(&self, hints: &StageResourceHints) -> bool {
        if hints.version != STAGE_RESOURCE_HINTS_VERSION {
            return false;
        }
        let target_supported =
            hints.target == ResourceTarget::Any || self.targets.contains(&hints.target);
        let memory_supported = match (hints.min_memory_bytes, self.memory_bytes) {
            (None, _) => true,
            (Some(required), Some(available)) => required <= available,
            (Some(_), None) => false,
        };
        let backend_supported = hints
            .backend
            .is_none_or(|backend| self.backends.contains(&backend));
        let device_supported = hints
            .device_class
            .is_none_or(|device| self.device_classes.contains(&device));
        target_supported
            && backend_supported
            && device_supported
            && memory_supported
            && self.concurrency_slots >= hints.concurrency_weight
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuntimeWorkerRegistration {
    pub version: u16,
    pub worker_id: String,
    pub instance_id: String,
    pub queue_classes: Vec<QueueClass>,
    pub capabilities: Vec<String>,
    pub model_ids: Vec<String>,
    pub stage_kinds: Vec<String>,
    pub resources: WorkerResourceCapacity,
    pub software_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuntimeWorkerHeartbeatDetails {
    pub version: u16,
    pub available_slots: u32,
    pub active_lease_ids: Vec<String>,
    pub last_error: Option<String>,
    pub health_json: serde_json::Value,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeJobStatus {
    Created,
    Queued,
    Running,
    Paused,
    Retrying,
    Postprocessing,
    Completed,
    Failed,
    Cancelled,
    Expired,
}

impl RuntimeJobStatus {
    pub const fn as_db_value(self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Paused => "paused",
            Self::Retrying => "retrying",
            Self::Postprocessing => "postprocessing",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Expired => "expired",
        }
    }

    pub fn from_db_value(value: &str) -> Option<Self> {
        match value {
            "created" => Some(Self::Created),
            "queued" => Some(Self::Queued),
            "running" => Some(Self::Running),
            "paused" => Some(Self::Paused),
            "retrying" => Some(Self::Retrying),
            "postprocessing" => Some(Self::Postprocessing),
            "completed" => Some(Self::Completed),
            "failed" => Some(Self::Failed),
            "cancelled" => Some(Self::Cancelled),
            "expired" => Some(Self::Expired),
            _ => None,
        }
    }
}

impl Default for RuntimeJobStatus {
    fn default() -> Self {
        Self::Created
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeStageStatus {
    Created,
    Queued,
    Running,
    Paused,
    Retrying,
    Postprocessing,
    Completed,
    Failed,
    Cancelled,
    Expired,
    Skipped,
}

impl RuntimeStageStatus {
    pub const fn as_db_value(self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Paused => "paused",
            Self::Retrying => "retrying",
            Self::Postprocessing => "postprocessing",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Expired => "expired",
            Self::Skipped => "skipped",
        }
    }

    pub fn from_db_value(value: &str) -> Option<Self> {
        match value {
            "created" => Some(Self::Created),
            "queued" => Some(Self::Queued),
            "running" => Some(Self::Running),
            "paused" => Some(Self::Paused),
            "retrying" => Some(Self::Retrying),
            "postprocessing" => Some(Self::Postprocessing),
            "completed" => Some(Self::Completed),
            "failed" => Some(Self::Failed),
            "cancelled" => Some(Self::Cancelled),
            "expired" => Some(Self::Expired),
            "skipped" => Some(Self::Skipped),
            _ => None,
        }
    }
}

impl Default for RuntimeStageStatus {
    fn default() -> Self {
        Self::Created
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeJobKind {
    AsrTranscription,
    TtsSpeech,
}

impl RuntimeJobKind {
    pub const fn as_db_value(self) -> &'static str {
        match self {
            Self::AsrTranscription => "asr_transcription",
            Self::TtsSpeech => "tts_speech",
        }
    }

    pub fn from_db_value(value: &str) -> Option<Self> {
        match value {
            "asr_transcription" => Some(Self::AsrTranscription),
            "tts_speech" => Some(Self::TtsSpeech),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeArtifactKind {
    Media,
    Text,
    Transcript,
    Audio,
    Metadata,
}

impl RuntimeArtifactKind {
    pub const fn as_db_value(self) -> &'static str {
        match self {
            Self::Media => "media",
            Self::Text => "text",
            Self::Transcript => "transcript",
            Self::Audio => "audio",
            Self::Metadata => "metadata",
        }
    }

    pub fn from_db_value(value: &str) -> Option<Self> {
        match value {
            "media" => Some(Self::Media),
            "text" => Some(Self::Text),
            "transcript" => Some(Self::Transcript),
            "audio" => Some(Self::Audio),
            "metadata" => Some(Self::Metadata),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeArtifactRole {
    InputOriginal,
    InputCanonical,
    OutputPrimary,
    OutputIntermediate,
    Debug,
}

impl RuntimeArtifactRole {
    pub const fn as_db_value(self) -> &'static str {
        match self {
            Self::InputOriginal => "input_original",
            Self::InputCanonical => "input_canonical",
            Self::OutputPrimary => "output_primary",
            Self::OutputIntermediate => "output_intermediate",
            Self::Debug => "debug",
        }
    }

    pub fn from_db_value(value: &str) -> Option<Self> {
        match value {
            "input_original" => Some(Self::InputOriginal),
            "input_canonical" => Some(Self::InputCanonical),
            "output_primary" => Some(Self::OutputPrimary),
            "output_intermediate" => Some(Self::OutputIntermediate),
            "debug" => Some(Self::Debug),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MediaAsset {
    pub id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub asset_kind: String,
    pub storage_namespace: String,
    pub storage_key: String,
    pub content_type: String,
    pub filename: Option<String>,
    pub size_bytes: u64,
    pub sha256: Option<String>,
    pub duration_secs: Option<f64>,
    pub sample_rate_hz: Option<u32>,
    pub channel_count: Option<u16>,
    pub peak_amplitude: Option<f32>,
    pub rms_amplitude: Option<f32>,
    pub scan_status: String,
    pub retention_policy: String,
    pub deleted_at: Option<u64>,
    pub metadata_json: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextAsset {
    pub id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub raw_text: String,
    pub normalized_text: String,
    pub language_hint: Option<String>,
    pub character_count: u64,
    pub sha256: Option<String>,
    pub safety_status: String,
    pub retention_policy: String,
    pub structure_json: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuntimeJob {
    pub id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub queued_at: Option<u64>,
    pub started_at: Option<u64>,
    pub finished_at: Option<u64>,
    pub job_kind: RuntimeJobKind,
    pub status: RuntimeJobStatus,
    pub priority: i32,
    pub model_id: Option<String>,
    pub capability: Option<String>,
    pub route_record_kind: Option<String>,
    pub route_record_id: Option<String>,
    pub input_media_asset_id: Option<String>,
    pub input_text_asset_id: Option<String>,
    pub request_json: serde_json::Value,
    pub model_snapshot_json: serde_json::Value,
    pub progress_json: Option<serde_json::Value>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
    pub attempt_count: u32,
    pub max_attempts: u32,
    pub retry_policy_json: serde_json::Value,
    pub idempotency_key: Option<String>,
    pub correlation_id: Option<String>,
    pub cancellation_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JobStage {
    pub id: String,
    pub job_id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub sequence: u32,
    pub stage_kind: String,
    pub queue_class: QueueClass,
    pub resource_hints: StageResourceHints,
    pub status: RuntimeStageStatus,
    pub capability: Option<String>,
    pub model_id: Option<String>,
    pub worker_id: Option<String>,
    pub lease_expires_at: Option<u64>,
    /// Earliest wall-clock time at which a queued/retrying stage may be claimed.
    pub available_at: Option<u64>,
    /// Opaque identity for the current or most recently completed execution attempt.
    pub attempt_token: Option<String>,
    pub attempt_count: u32,
    pub max_attempts: u32,
    pub input_artifact_ids: Vec<String>,
    pub output_artifact_ids: Vec<String>,
    pub progress_json: Option<serde_json::Value>,
    pub started_at: Option<u64>,
    pub finished_at: Option<u64>,
    pub error_code: Option<String>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuntimeArtifact {
    pub id: String,
    pub job_id: String,
    pub stage_id: Option<String>,
    pub created_at: u64,
    pub artifact_kind: RuntimeArtifactKind,
    pub artifact_role: RuntimeArtifactRole,
    pub media_asset_id: Option<String>,
    pub text_asset_id: Option<String>,
    pub storage_key: Option<String>,
    pub content_type: Option<String>,
    pub filename: Option<String>,
    pub size_bytes: Option<u64>,
    pub sha256: Option<String>,
    pub metadata_json: serde_json::Value,
    pub retention_policy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IdempotencyRecord {
    pub operation: String,
    pub idempotency_key: String,
    pub created_at: u64,
    pub expires_at: Option<u64>,
    pub request_hash: String,
    pub response_json: Option<serde_json::Value>,
    pub runtime_job_id: Option<String>,
    pub conflict_message: Option<String>,
    pub metadata_json: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RuntimeWorkerHeartbeat {
    pub worker_id: String,
    pub started_at: u64,
    pub last_heartbeat_at: u64,
    pub status: String,
    pub queue_names: Vec<String>,
    pub instance_id: String,
    pub registration: RuntimeWorkerRegistration,
    pub details: RuntimeWorkerHeartbeatDetails,
    pub current_job_id: Option<String>,
    pub current_stage_id: Option<String>,
    pub diagnostic_json: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClaimedStage {
    pub job: RuntimeJob,
    pub stage: JobStage,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StageLease {
    pub stage_id: String,
    pub worker_id: String,
    pub attempt_count: u32,
    pub attempt_token: Option<String>,
}

impl ClaimedStage {
    pub fn lease(&self) -> Option<StageLease> {
        Some(StageLease {
            stage_id: self.stage.id.clone(),
            worker_id: self.stage.worker_id.clone()?,
            attempt_count: self.stage.attempt_count,
            attempt_token: self.stage.attempt_token.clone(),
        })
    }
}
