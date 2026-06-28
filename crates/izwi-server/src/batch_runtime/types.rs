use serde::{Deserialize, Serialize};

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
    pub status: RuntimeStageStatus,
    pub capability: Option<String>,
    pub model_id: Option<String>,
    pub worker_id: Option<String>,
    pub lease_expires_at: Option<u64>,
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
    pub current_job_id: Option<String>,
    pub current_stage_id: Option<String>,
    pub diagnostic_json: serde_json::Value,
}
