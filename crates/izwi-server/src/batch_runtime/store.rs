use super::types::{
    ClaimedStage, IdempotencyRecord, JobStage, MediaAsset, QueueClass, RuntimeArtifact,
    RuntimeArtifactKind, RuntimeArtifactRole, RuntimeCancellationState, RuntimeJob, RuntimeJobKind,
    RuntimeJobStatus, RuntimeStageStatus, RuntimeWorkerHeartbeat, RuntimeWorkerHeartbeatDetails,
    RuntimeWorkerRegistration, StageLease, StageResourceHints, TextAsset, WorkerResourceCapacity,
    WORKER_HEARTBEAT_DETAILS_VERSION, WORKER_REGISTRATION_VERSION,
};
#[cfg(test)]
use super::types::{DeviceClass, ResourceTarget, RuntimeBackendClass};
use crate::{
    db::{raw, StoreDatabase},
    ids::new_uuid,
    speech_history_store::{
        sanitize_audio_mime_type, sanitize_optional_text, NewSpeechHistoryRecord,
        SpeechHistoryProcessingStatus, SpeechHistoryRecord, SpeechRouteKind,
    },
};
use anyhow::{anyhow, bail, Context};
use izwi_hooks::{HookMetadata, MediaNamespace, MediaWriteRequest};
use sea_orm::{
    ConnectionTrait, DatabaseConnection, DbBackend, QueryResult, SqliteTransactionMode,
    TransactionOptions, TransactionTrait, Value,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    io::{self, Write},
    time::{SystemTime, UNIX_EPOCH},
};

#[cfg(test)]
use std::sync::{atomic::AtomicI64, atomic::Ordering, Arc};

#[derive(Debug, Clone)]
pub struct BatchRuntimeStore {
    db: StoreDatabase,
    #[cfg(test)]
    test_clock: Option<Arc<AtomicI64>>,
    #[cfg(test)]
    test_tts_admission_limits: Option<(usize, usize)>,
    #[cfg(test)]
    test_artifact_cleanup_capacity: Option<u64>,
    #[cfg(test)]
    test_provider_write_capacity: Option<u64>,
    #[cfg(test)]
    test_durable_tts_acceptance_failpoint: Option<DurableTtsAcceptanceFailpoint>,
}

#[derive(Debug, Clone)]
pub struct NewMediaAsset {
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
    pub source_asset_id: Option<String>,
    pub canonical_profile_version: Option<String>,
    pub scan_status: String,
    pub retention_policy: String,
    pub metadata_json: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArtifactCleanupReason {
    ArtifactDeleted,
}

impl ArtifactCleanupReason {
    fn as_db_value(self) -> &'static str {
        match self {
            Self::ArtifactDeleted => "artifact_deleted",
        }
    }

    fn from_db_value(value: &str) -> anyhow::Result<Self> {
        match value {
            "artifact_deleted" => Ok(Self::ArtifactDeleted),
            _ => bail!("Unknown artifact cleanup reason"),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ArtifactCleanupIntent {
    pub id: String,
    pub created_at: u64,
    pub updated_at: u64,
    pub available_at: u64,
    pub storage_key: String,
    pub tenant_scope: String,
    pub reason: ArtifactCleanupReason,
    pub attempt_count: u32,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct NewProviderWriteReservation {
    pub write_id: String,
    pub tenant_scope: String,
    pub storage_namespace: String,
    pub content_type: String,
    pub filename: Option<String>,
    pub expected_size_bytes: u64,
    pub expected_sha256: String,
    pub lifetime_ms: u64,
    pub provider_request: MediaWriteRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProviderWriteReservation {
    pub write_id: String,
    pub reservation_token: String,
    pub created_at: u64,
    pub expires_at: u64,
    pub tenant_scope: String,
    pub storage_namespace: String,
    pub content_type: String,
    pub filename: Option<String>,
    pub expected_size_bytes: u64,
    pub expected_sha256: String,
    pub provider_request: MediaWriteRequest,
    pub storage_key: Option<String>,
    pub cleanup_claim_token: Option<String>,
    pub cleanup_attempt_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderWriteRequestEnvelope {
    version: u16,
    request: ProviderWriteRequestV1,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderWriteRequestV1 {
    namespace: MediaNamespace,
    record_id: String,
    preferred_filename: Option<String>,
    content_type: String,
    metadata: HookMetadata,
}

impl From<&MediaWriteRequest> for ProviderWriteRequestV1 {
    fn from(request: &MediaWriteRequest) -> Self {
        Self {
            namespace: request.namespace.clone(),
            record_id: request.record_id.clone(),
            preferred_filename: request.preferred_filename.clone(),
            content_type: request.content_type.clone(),
            metadata: request.metadata.clone(),
        }
    }
}

impl From<ProviderWriteRequestV1> for MediaWriteRequest {
    fn from(request: ProviderWriteRequestV1) -> Self {
        Self {
            namespace: request.namespace,
            record_id: request.record_id,
            preferred_filename: request.preferred_filename,
            content_type: request.content_type,
            metadata: request.metadata,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NewTextAsset {
    pub raw_text: String,
    pub normalized_text: Option<String>,
    pub language_hint: Option<String>,
    pub sha256: Option<String>,
    pub safety_status: String,
    pub retention_policy: String,
    pub structure_json: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct NewRuntimeJob {
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
    pub retry_policy_json: serde_json::Value,
    pub max_attempts: u32,
    pub idempotency_key: Option<String>,
    pub correlation_id: Option<String>,
}

/// The bounded, single-stage durable graph used by the first text-only TTS
/// acceptance path. Reference media is deliberately excluded until object
/// publication has its own crash-safe ownership ledger.
#[derive(Debug, Clone)]
pub struct NewDurableTextTtsAcceptance {
    pub projection: NewSpeechHistoryRecord,
    pub request_json: serde_json::Value,
    pub model_snapshot_json: serde_json::Value,
    pub retry_policy_json: serde_json::Value,
    pub priority: i32,
    pub max_attempts: u32,
    pub correlation_id: Option<String>,
    pub stage_kind: String,
    pub queue_class: QueueClass,
    pub resource_hints: StageResourceHints,
    pub reservation: Option<DurableIdempotencyReservation>,
    pub idempotency_retention_ms: u64,
}

#[derive(Debug, Clone)]
pub struct DurableTextTtsAcceptance {
    pub record: SpeechHistoryRecord,
    pub job: RuntimeJob,
    pub stage: JobStage,
    pub input_artifact: RuntimeArtifact,
}

#[derive(Debug, Clone)]
pub enum DurableTextTtsAcceptanceOutcome {
    Committed(DurableTextTtsAcceptance),
    ReservationLost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DurableTtsAcceptanceFailpoint {
    Projection,
    TextAsset,
    Job,
    Artifact,
    Stage,
    Idempotency,
}

#[derive(Debug, Clone)]
pub struct NewJobStage {
    pub job_id: String,
    pub sequence: u32,
    pub stage_kind: String,
    pub status: RuntimeStageStatus,
    pub capability: Option<String>,
    pub model_id: Option<String>,
    pub max_attempts: u32,
    pub input_artifact_ids: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct NewJobStageDispatch {
    pub stage: NewJobStage,
    pub queue_class: QueueClass,
    pub resource_hints: StageResourceHints,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageLeaseState {
    Active,
    CancellationRequested,
    ExecutionStopping,
}

const DEFAULT_STAGE_CLAIM_CANDIDATE_LIMIT: usize = 64;
const MAX_STAGE_CLAIM_CANDIDATE_LIMIT: usize = 512;
pub(crate) const DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT: usize = 64;
const MAX_RUNTIME_MAINTENANCE_BATCH_LIMIT: usize = 512;
pub(crate) const MAX_STAGE_OUTPUT_ARTIFACTS: usize = 64;
pub(crate) const MAX_STAGE_OUTPUT_ARTIFACT_ID_BYTES: usize = 64;
const MAX_ARTIFACT_CLEANUP_INTENTS: u64 = 65_536;
const MAX_ARTIFACT_CLEANUP_BATCH: usize = 64;
const MAX_ARTIFACT_CLEANUP_STORAGE_KEY_BYTES: usize = 2 * 1024;
const MAX_ARTIFACT_CLEANUP_TENANT_BYTES: usize = 128;
const MAX_ARTIFACT_CLEANUP_ERROR_BYTES: usize = 512;
const MAX_ARTIFACT_CLEANUP_BACKOFF_MS: u64 = 60 * 60 * 1000;
const MAX_PROVIDER_WRITE_RESERVATIONS: u64 = 65_536;
const MAX_PROVIDER_WRITE_CLEANUP_BATCH: usize = 64;
const MAX_PROVIDER_WRITE_NAMESPACE_BYTES: usize = 128;
const MAX_PROVIDER_WRITE_CONTENT_TYPE_BYTES: usize = 256;
const MAX_PROVIDER_WRITE_FILENAME_BYTES: usize = 1024;
const MAX_PROVIDER_WRITE_REQUEST_ENVELOPE_BYTES: usize = 8 * 1024;
const MAX_PROVIDER_WRITE_RECORD_ID_BYTES: usize = 64;
const MAX_PROVIDER_WRITE_METADATA_ENTRIES: usize = 16;
const MAX_PROVIDER_WRITE_METADATA_KEY_BYTES: usize = 128;
const MAX_PROVIDER_WRITE_METADATA_VALUE_BYTES: usize = 1024;
const MAX_PROVIDER_WRITE_METADATA_BYTES: usize = 4 * 1024;
const MAX_PROVIDER_WRITE_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_PROVIDER_WRITE_LIFETIME_MS: u64 = 10 * 60 * 1000;
const PROVIDER_WRITE_CLEANUP_CLAIM_MS: u64 = 30 * 1000;
const PROVIDER_WRITE_REQUEST_ENVELOPE_VERSION: u16 = 1;

fn bounded_maintenance_batch_limit(limit: usize) -> usize {
    limit.clamp(1, MAX_RUNTIME_MAINTENANCE_BATCH_LIMIT)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageClaimFilter {
    pub queue_names: Vec<String>,
    pub capabilities: Vec<String>,
    pub model_ids: Vec<String>,
    pub stage_kinds: Vec<String>,
    pub resources: WorkerResourceCapacity,
    pub max_candidates: usize,
}

impl Default for StageClaimFilter {
    fn default() -> Self {
        Self {
            queue_names: vec!["batch".to_string()],
            capabilities: Vec::new(),
            model_ids: Vec::new(),
            stage_kinds: Vec::new(),
            resources: WorkerResourceCapacity::default(),
            max_candidates: DEFAULT_STAGE_CLAIM_CANDIDATE_LIMIT,
        }
    }
}

impl StageClaimFilter {
    pub fn for_worker_queues(queue_names: &[String]) -> Self {
        let mut queue_names = normalize_filter_values(queue_names);
        if queue_names.is_empty() {
            queue_names.push("batch".to_string());
        }
        Self {
            queue_names,
            ..Default::default()
        }
    }

    fn normalized(&self) -> Self {
        Self {
            queue_names: normalize_filter_values(&self.queue_names),
            capabilities: normalize_filter_values(&self.capabilities),
            model_ids: normalize_filter_values(&self.model_ids),
            stage_kinds: normalize_filter_values(&self.stage_kinds),
            resources: self.resources.clone(),
            max_candidates: self.max_candidates,
        }
    }

    pub fn matches(&self, candidate: &StageClaimCandidate) -> bool {
        self.queue_matches(candidate)
            && optional_filter_matches(&self.capabilities, candidate.capability.as_deref())
            && optional_filter_matches(&self.model_ids, candidate.model_id.as_deref())
            && optional_filter_matches(&self.stage_kinds, Some(candidate.stage_kind.as_str()))
            && self.resources.supports(&candidate.resource_hints)
    }

    fn queue_matches(&self, candidate: &StageClaimCandidate) -> bool {
        if self.queue_names.is_empty() {
            return true;
        }

        self.queue_names.iter().any(|queue| {
            queue == QueueClass::Batch.as_db_value() || queue == candidate.queue_class.as_db_value()
        })
    }

    fn candidate_limit(&self) -> usize {
        self.max_candidates
            .clamp(1, MAX_STAGE_CLAIM_CANDIDATE_LIMIT)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageClaimCandidate {
    pub stage_id: String,
    pub stage_kind: String,
    pub job_kind: RuntimeJobKind,
    pub queue_class: QueueClass,
    pub resource_hints: StageResourceHints,
    pub capability: Option<String>,
    pub model_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct NewRuntimeArtifact {
    pub job_id: String,
    pub stage_id: Option<String>,
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

#[derive(Debug, Clone)]
pub struct NewStageOutputArtifact {
    pub publication_key: String,
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

#[derive(Debug, Clone)]
pub struct NewIdempotencyRecord {
    pub operation: String,
    pub idempotency_key: String,
    pub expires_at: Option<u64>,
    pub request_hash: String,
    pub response_json: Option<serde_json::Value>,
    pub runtime_job_id: Option<String>,
    pub conflict_message: Option<String>,
    pub metadata_json: serde_json::Value,
}

/// Versioned, bounded identity for one durable create-operation request.
///
/// This contract is intentionally separate from worker-attempt duplicate
/// suppression and does not imply replay for synchronous streaming requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableIdempotencyRequest {
    pub tenant_scope: String,
    pub operation: String,
    pub idempotency_key: String,
    pub digest_version: u16,
    pub request_digest: String,
    pub reservation_ttl_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurableIdempotencyReservation {
    pub tenant_scope: String,
    pub operation: String,
    pub idempotency_key: String,
    pub digest_version: u16,
    pub request_digest: String,
    pub reservation_token: String,
    pub expires_at: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DurableIdempotencyReplay {
    pub runtime_job_id: String,
    pub response_json: serde_json::Value,
    pub expires_at: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DurableIdempotencyBegin {
    Acquired(DurableIdempotencyReservation),
    Replay(DurableIdempotencyReplay),
    Conflict,
    InProgress { expires_at: u64 },
    CapacityExceeded,
}

pub const DURABLE_IDEMPOTENCY_DIGEST_VERSION: u16 = 1;
pub const MAX_DURABLE_IDEMPOTENCY_TENANT_BYTES: usize = 256;
pub const MAX_DURABLE_IDEMPOTENCY_OPERATION_BYTES: usize = 128;
pub const MAX_DURABLE_IDEMPOTENCY_KEY_BYTES: usize = 256;
pub const MAX_DURABLE_IDEMPOTENCY_CANONICAL_REQUEST_BYTES: usize = 1024 * 1024;
pub const MAX_DURABLE_IDEMPOTENCY_RESULT_BYTES: usize = 64 * 1024;
pub const MAX_DURABLE_IDEMPOTENCY_RESERVATION_TTL_MS: u64 = 10 * 60 * 1_000;
pub const MAX_DURABLE_IDEMPOTENCY_RETENTION_MS: u64 = 7 * 24 * 60 * 60 * 1_000;
pub const MAX_DURABLE_TTS_TEXT_BYTES: usize = 1024 * 1024;
const MAX_DURABLE_TTS_REQUEST_JSON_BYTES: usize = 2 * 1024 * 1024;
const MAX_DURABLE_TTS_METADATA_JSON_BYTES: usize = 64 * 1024;
const MAX_DURABLE_TTS_OPTIONAL_FIELD_BYTES: usize = 64 * 1024;
const MAX_DURABLE_TTS_CORRELATION_BYTES: usize = 256;
const MAX_DURABLE_IDEMPOTENCY_RECORDS: u64 = 65_536;
const DEFAULT_DURABLE_IDEMPOTENCY_PRUNE_LIMIT: usize = 64;
const MAX_DURABLE_IDEMPOTENCY_PRUNE_LIMIT: usize = 512;
const MAX_DURABLE_IDEMPOTENCY_JSON_DEPTH: usize = 128;
const MAX_DURABLE_IDEMPOTENCY_JSON_OBJECT_KEYS: usize = 16_384;

/// Hash a semantically complete request envelope using the version-one
/// canonical JSON encoding. Object keys are sorted recursively by Rust string
/// order (Unicode scalar value order); array order, scalar types, and values
/// remain significant. Uploads must be represented in the envelope by their
/// verified content digest rather than a temporary path.
pub fn canonical_request_digest(request: &serde_json::Value) -> anyhow::Result<String> {
    let mut canonical = BoundedJsonBytes::new(MAX_DURABLE_IDEMPOTENCY_CANONICAL_REQUEST_BYTES);
    write_canonical_json(request, &mut canonical, 0)?;
    Ok(sha256_hex(&canonical.bytes))
}

#[derive(Debug, Clone)]
pub struct WorkerHeartbeatUpdate {
    pub worker_id: String,
    pub status: String,
    pub queue_names: Vec<String>,
    pub current_job_id: Option<String>,
    pub current_stage_id: Option<String>,
    pub diagnostic_json: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct RegisteredWorkerHeartbeatUpdate {
    pub registration: RuntimeWorkerRegistration,
    pub status: String,
    pub current_job_id: Option<String>,
    pub current_stage_id: Option<String>,
    pub details: RuntimeWorkerHeartbeatDetails,
    pub diagnostic_json: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RuntimeJobStatusCount {
    pub status: RuntimeJobStatus,
    pub count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RuntimeStageStatusCount {
    pub status: RuntimeStageStatus,
    pub count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RuntimeQueueDepth {
    pub queue_class: QueueClass,
    pub count: u64,
    pub oldest_age_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct RuntimeQueueHealthSnapshot {
    pub heartbeat_stale_after_ms: u64,
    pub active_workers: u64,
    pub healthy_workers: u64,
    pub stale_workers: u64,
    pub queues: Vec<RuntimeQueueDepth>,
    pub uncovered_queue_classes: Vec<QueueClass>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub struct RuntimeReconciliationReport {
    pub jobs_repaired: u64,
    pub stages_repaired: u64,
    pub route_projections_repaired: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct StoredRetryPolicy {
    max_attempts: Option<u32>,
    #[serde(alias = "backoff_ms", alias = "base_delay_ms")]
    initial_backoff_ms: u64,
    backoff_multiplier: f64,
    max_backoff_ms: u64,
    jitter_ratio: f64,
}

impl Default for StoredRetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: None,
            // Preserve the current immediate-retry behavior unless a producer opts in.
            initial_backoff_ms: 0,
            backoff_multiplier: 2.0,
            max_backoff_ms: 60_000,
            jitter_ratio: 0.2,
        }
    }
}

impl StoredRetryPolicy {
    fn from_job(job: &RuntimeJob) -> Self {
        serde_json::from_value(job.retry_policy_json.clone()).unwrap_or_default()
    }

    fn effective_max_attempts(&self, job: &RuntimeJob, stage: &JobStage) -> u32 {
        self.max_attempts
            .unwrap_or(job.max_attempts)
            .min(job.max_attempts)
            .min(stage.max_attempts)
    }

    fn backoff_ms(&self, stage: &JobStage) -> u64 {
        if self.initial_backoff_ms == 0 {
            return 0;
        }

        const MAX_RETRY_BACKOFF_MS: u64 = 24 * 60 * 60 * 1_000;
        let exponent = stage.attempt_count.saturating_sub(1).min(31) as i32;
        let multiplier = self.backoff_multiplier.clamp(1.0, 10.0);
        let delay = (self.initial_backoff_ms as f64 * multiplier.powi(exponent)) as u64;
        let configured_max = self
            .max_backoff_ms
            .max(self.initial_backoff_ms)
            .min(MAX_RETRY_BACKOFF_MS);
        let delay = delay.min(configured_max);
        let jitter_span = (delay as f64 * self.jitter_ratio.clamp(0.0, 1.0)) as u64;
        if jitter_span == 0 {
            return delay;
        }

        // Stable per stage attempt so recovery workers converge on the same eligibility time.
        let mut hasher = DefaultHasher::new();
        stage.id.hash(&mut hasher);
        stage.attempt_count.hash(&mut hasher);
        let width = jitter_span.saturating_mul(2).saturating_add(1);
        delay
            .saturating_sub(jitter_span)
            .saturating_add(hasher.finish() % width)
            .min(configured_max)
    }
}

#[derive(Debug, Clone, Copy)]
enum LeaseValidity {
    Expired,
    Any,
}

impl LeaseValidity {
    fn sql_predicate(self) -> &'static str {
        match self {
            Self::Expired => "lease_expires_at <= ?7",
            // Owner-fenced relinquish must not depend on wall-clock expiry:
            // attempt identity still protects against a reclaimed lease.
            Self::Any => "1 = 1",
        }
    }
}

impl BatchRuntimeStore {
    pub fn initialize_with_database(db: StoreDatabase) -> Self {
        Self {
            db,
            #[cfg(test)]
            test_clock: None,
            #[cfg(test)]
            test_tts_admission_limits: None,
            #[cfg(test)]
            test_artifact_cleanup_capacity: None,
            #[cfg(test)]
            test_provider_write_capacity: None,
            #[cfg(test)]
            test_durable_tts_acceptance_failpoint: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn set_test_clock(&mut self, clock: Arc<AtomicI64>) {
        self.test_clock = Some(clock);
    }

    #[cfg(test)]
    pub(crate) fn set_artifact_cleanup_capacity_for_test(&mut self, capacity: u64) {
        self.test_artifact_cleanup_capacity = Some(capacity);
    }

    #[cfg(test)]
    pub(crate) fn set_provider_write_capacity_for_test(&mut self, capacity: u64) {
        self.test_provider_write_capacity = Some(capacity);
    }

    #[cfg(test)]
    fn set_durable_tts_acceptance_failpoint(
        &mut self,
        failpoint: Option<DurableTtsAcceptanceFailpoint>,
    ) {
        self.test_durable_tts_acceptance_failpoint = failpoint;
    }

    fn now_millis(&self) -> i64 {
        #[cfg(test)]
        if let Some(clock) = &self.test_clock {
            return clock.load(Ordering::SeqCst);
        }
        current_timestamp_millis()
    }

    fn inject_durable_tts_acceptance_failure(
        &self,
        failpoint: DurableTtsAcceptanceFailpoint,
    ) -> anyhow::Result<()> {
        #[cfg(test)]
        if self.test_durable_tts_acceptance_failpoint == Some(failpoint) {
            bail!("Injected durable text TTS acceptance failure after {failpoint:?}");
        }
        #[cfg(not(test))]
        let _ = failpoint;
        Ok(())
    }

    pub async fn connection(&self) -> anyhow::Result<&DatabaseConnection> {
        self.db.connection().await
    }

    pub async fn create_media_asset(&self, input: NewMediaAsset) -> anyhow::Result<MediaAsset> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let id = new_uuid();
        let metadata_json = json_to_db_string(&input.metadata_json, "{}")?;

        db.execute_raw(raw::statement(
            db,
            r#"
            INSERT INTO media_assets (
                id,
                created_at,
                updated_at,
                asset_kind,
                storage_namespace,
                storage_key,
                content_type,
                filename,
                size_bytes,
                sha256,
                duration_secs,
                sample_rate_hz,
                channel_count,
                peak_amplitude,
                rms_amplitude,
                source_asset_id,
                canonical_profile_version,
                scan_status,
                retention_policy,
                deleted_at,
                metadata_json
            )
            VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, NULL, ?19)
            "#,
            vec![
                id.clone().into(),
                now.into(),
                input.asset_kind.into(),
                input.storage_namespace.into(),
                input.storage_key.into(),
                input.content_type.into(),
                opt_string(input.filename),
                u64_to_i64_value(input.size_bytes)?,
                opt_string(input.sha256),
                opt_f64(input.duration_secs),
                opt_u32(input.sample_rate_hz),
                opt_u16(input.channel_count),
                opt_f32(input.peak_amplitude),
                opt_f32(input.rms_amplitude),
                opt_string(input.source_asset_id),
                opt_string(input.canonical_profile_version),
                input.scan_status.into(),
                input.retention_policy.into(),
                metadata_json.into(),
            ],
        )?)
        .await
        .context("Failed to create media asset")?;

        self.get_media_asset(&id)
            .await?
            .ok_or_else(|| anyhow!("Created media asset was not found"))
    }

    pub async fn get_media_asset(&self, id: &str) -> anyhow::Result<Option<MediaAsset>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                MEDIA_ASSET_COLUMNS_SQL,
                vec![id.into()],
            )?)
            .await
            .context("Failed to load media asset")?;

        row.as_ref().map(map_media_asset).transpose()
    }

    /// Logically delete an artifact-backed media row before physical object cleanup.
    ///
    /// The tombstone is intentionally retained so a failed provider deletion cannot
    /// make the object visible again. Provider-specific garbage collection may retry
    /// physical deletion without changing the public artifact identity.
    pub async fn tombstone_media_asset(&self, id: &str) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE media_assets
                SET updated_at = ?1, deleted_at = ?1
                WHERE id = ?2 AND deleted_at IS NULL
                "#,
                vec![now.into(), id.into()],
            )?)
            .await
            .context("Failed to tombstone media asset")?;

        Ok(result.rows_affected() == 1)
    }

    /// Atomically hide one opaque artifact and retain its provider key for
    /// idempotent physical deletion. Capacity is checked under a durable lock
    /// before the tombstone changes visibility.
    pub(crate) async fn tombstone_media_asset_with_cleanup(
        &self,
        id: &str,
        tenant_scope: &str,
        reason: ArtifactCleanupReason,
    ) -> anyhow::Result<bool> {
        validate_artifact_cleanup_tenant(tenant_scope)?;
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(runtime_write_transaction_options())
            .await
            .context("Failed to start artifact cleanup transaction")?;
        lock_artifact_cleanup_capacity(&tx).await?;

        let Some(asset) = tx
            .query_one_raw(raw::statement(
                &tx,
                r#"
                SELECT storage_key, deleted_at
                FROM media_assets
                WHERE id = ?1
                  AND asset_kind = 'opaque_artifact'
                  AND storage_namespace = 'artifact_store_v1'
                "#,
                vec![id.into()],
            )?)
            .await
            .context("Failed to load opaque artifact for deletion")?
        else {
            tx.rollback().await?;
            return Ok(false);
        };
        let storage_key: String = asset.try_get_by_index(0)?;
        validate_artifact_cleanup_storage_key(&storage_key)?;
        let already_deleted = asset.try_get_by_index::<Option<i64>>(1)?.is_some();

        let existing_tenant = tx
            .query_one_raw(raw::statement(
                &tx,
                "SELECT tenant_scope FROM artifact_cleanup_intents WHERE storage_key = ?1",
                vec![storage_key.clone().into()],
            )?)
            .await
            .context("Failed to inspect existing artifact cleanup intent")?
            .map(|row| row.try_get_by_index::<String>(0))
            .transpose()?;
        if let Some(existing_tenant) = existing_tenant {
            anyhow::ensure!(
                existing_tenant == tenant_scope,
                "Artifact cleanup key is already owned by another tenant"
            );
        } else {
            let count = tx
                .query_one_raw(raw::statement(
                    &tx,
                    "SELECT COUNT(*) FROM artifact_cleanup_intents",
                    vec![],
                )?)
                .await
                .context("Failed to count artifact cleanup intents")?
                .ok_or_else(|| anyhow!("Artifact cleanup count returned no row"))?
                .try_get_by_index::<i64>(0)?;
            let capacity = self.artifact_cleanup_capacity();
            anyhow::ensure!(
                u64::try_from(count)? < capacity,
                "Artifact cleanup capacity exhausted ({capacity} pending intents)"
            );
            let now = self.now_millis();
            tx.execute_raw(raw::statement(
                &tx,
                r#"
                INSERT INTO artifact_cleanup_intents (
                    id, created_at, updated_at, available_at, storage_key,
                    tenant_scope, reason, attempt_count, last_error
                )
                VALUES (?1, ?2, ?2, ?2, ?3, ?4, ?5, 0, NULL)
                "#,
                vec![
                    new_uuid().into(),
                    now.into(),
                    storage_key.into(),
                    tenant_scope.into(),
                    reason.as_db_value().into(),
                ],
            )?)
            .await
            .context("Failed to persist artifact cleanup intent")?;
        }

        let newly_deleted = if already_deleted {
            false
        } else {
            let now = self.now_millis();
            tx.execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE media_assets
                SET updated_at = ?1, deleted_at = ?1
                WHERE id = ?2 AND deleted_at IS NULL
                "#,
                vec![now.into(), id.into()],
            )?)
            .await
            .context("Failed to tombstone media asset")?
            .rows_affected()
                == 1
        };
        tx.commit()
            .await
            .context("Failed to commit artifact cleanup intent")?;
        Ok(newly_deleted)
    }

    pub(crate) async fn due_artifact_cleanup_intents(
        &self,
        limit: usize,
    ) -> anyhow::Result<Vec<ArtifactCleanupIntent>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(
                db,
                r#"
                SELECT c.id, c.created_at, c.updated_at, c.available_at,
                       c.storage_key, c.tenant_scope, c.reason,
                       c.attempt_count, c.last_error
                FROM artifact_cleanup_intents c
                JOIN media_assets m ON m.storage_key = c.storage_key
                WHERE c.available_at <= ?1
                  AND c.reason = 'artifact_deleted'
                  AND m.asset_kind = 'opaque_artifact'
                  AND m.storage_namespace = 'artifact_store_v1'
                  AND m.deleted_at IS NOT NULL
                ORDER BY c.available_at ASC, c.created_at ASC, c.id ASC
                LIMIT ?2
                "#,
                vec![
                    self.now_millis().into(),
                    i64::try_from(limit.min(MAX_ARTIFACT_CLEANUP_BATCH))?.into(),
                ],
            )?)
            .await
            .context("Failed to list due artifact cleanup intents")?;
        rows.iter().map(map_artifact_cleanup_intent).collect()
    }

    pub(crate) async fn artifact_cleanup_intent_for_storage_key(
        &self,
        storage_key: &str,
    ) -> anyhow::Result<Option<ArtifactCleanupIntent>> {
        validate_artifact_cleanup_storage_key(storage_key)?;
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                r#"
                SELECT id, created_at, updated_at, available_at, storage_key,
                       tenant_scope, reason, attempt_count, last_error
                FROM artifact_cleanup_intents
                WHERE storage_key = ?1
                "#,
                vec![storage_key.into()],
            )?)
            .await
            .context("Failed to load artifact cleanup intent")?;
        row.as_ref().map(map_artifact_cleanup_intent).transpose()
    }

    pub(crate) async fn complete_artifact_cleanup(
        &self,
        id: &str,
        storage_key: &str,
    ) -> anyhow::Result<bool> {
        validate_artifact_cleanup_storage_key(storage_key)?;
        let db = self.db.connection().await?;
        let result = db
            .execute_raw(raw::statement(
                db,
                "DELETE FROM artifact_cleanup_intents WHERE id = ?1 AND storage_key = ?2",
                vec![id.into(), storage_key.into()],
            )?)
            .await
            .context("Failed to complete artifact cleanup intent")?;
        Ok(result.rows_affected() == 1)
    }

    pub(crate) async fn defer_artifact_cleanup(
        &self,
        intent: &ArtifactCleanupIntent,
        error: &str,
    ) -> anyhow::Result<bool> {
        validate_artifact_cleanup_storage_key(&intent.storage_key)?;
        let attempt_count = intent.attempt_count.saturating_add(1);
        let shift = intent.attempt_count.min(12);
        let backoff_ms = 1_000_u64
            .checked_shl(shift)
            .unwrap_or(MAX_ARTIFACT_CLEANUP_BACKOFF_MS)
            .min(MAX_ARTIFACT_CLEANUP_BACKOFF_MS);
        let now = self.now_millis();
        let available_at = now.saturating_add(i64::try_from(backoff_ms)?);
        let last_error = truncate_utf8_bytes(error, MAX_ARTIFACT_CLEANUP_ERROR_BYTES);
        let db = self.db.connection().await?;
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE artifact_cleanup_intents
                SET updated_at = ?1, available_at = ?2, attempt_count = ?3,
                    last_error = ?4
                WHERE id = ?5 AND storage_key = ?6
                "#,
                vec![
                    now.into(),
                    available_at.into(),
                    i64::from(attempt_count).into(),
                    last_error.into(),
                    intent.id.clone().into(),
                    intent.storage_key.clone().into(),
                ],
            )?)
            .await
            .context("Failed to defer artifact cleanup intent")?;
        Ok(result.rows_affected() == 1)
    }

    fn artifact_cleanup_capacity(&self) -> u64 {
        #[cfg(test)]
        if let Some(capacity) = self.test_artifact_cleanup_capacity {
            return capacity;
        }
        MAX_ARTIFACT_CLEANUP_INTENTS
    }

    pub(crate) async fn reserve_provider_write(
        &self,
        input: NewProviderWriteReservation,
    ) -> anyhow::Result<ProviderWriteReservation> {
        validate_provider_write_input(&input)?;
        let request_envelope_json = serialize_provider_write_request(&input.provider_request)?;
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(runtime_write_transaction_options())
            .await
            .context("Failed to start provider write reservation transaction")?;
        lock_provider_write_capacity(&tx).await?;
        let count = tx
            .query_one_raw(raw::statement(
                &tx,
                "SELECT COUNT(*) FROM provider_write_reservations",
                vec![],
            )?)
            .await?
            .context("Provider write reservation count returned no row")?
            .try_get_by_index::<i64>(0)?;
        let capacity = self.provider_write_capacity();
        anyhow::ensure!(
            u64::try_from(count)? < capacity,
            "Provider write reservation capacity exhausted ({capacity} active reservations)"
        );
        let now = self.now_millis();
        let expires_at = now.saturating_add(i64::try_from(input.lifetime_ms)?);
        let write_id = input.write_id.clone();
        let reservation_token = new_uuid();
        tx.execute_raw(raw::statement(
            &tx,
            r#"
            INSERT INTO provider_write_reservations (
                write_id, reservation_token, created_at, updated_at, expires_at,
                available_at, state, tenant_scope, storage_namespace,
                content_type, filename, expected_size_bytes, expected_sha256,
                provider_request_json,
                storage_key, cleanup_claim_token, cleanup_claim_expires_at,
                cleanup_attempt_count, last_error
            ) VALUES (?1, ?2, ?3, ?3, ?4, ?4, 'reserved', ?5, ?6, ?7, ?8,
                      ?9, ?10, ?11, NULL, NULL, NULL, 0, NULL)
            "#,
            vec![
                write_id.clone().into(),
                reservation_token.clone().into(),
                now.into(),
                expires_at.into(),
                input.tenant_scope.into(),
                input.storage_namespace.into(),
                input.content_type.into(),
                opt_string(input.filename),
                u64_to_i64_value(input.expected_size_bytes)?,
                input.expected_sha256.into(),
                request_envelope_json.into(),
            ],
        )?)
        .await
        .context("Failed to reserve provider write")?;
        let reservation = get_provider_write_with(&tx, &write_id)
            .await?
            .context("Reserved provider write was not found")?;
        tx.commit().await?;
        Ok(reservation)
    }

    pub(crate) async fn record_provider_write_stored(
        &self,
        reservation: &ProviderWriteReservation,
        storage_key: &str,
    ) -> anyhow::Result<Option<ProviderWriteReservation>> {
        validate_artifact_cleanup_storage_key(storage_key)?;
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE provider_write_reservations
                SET state = 'stored', storage_key = ?1, updated_at = ?2
                WHERE write_id = ?3 AND reservation_token = ?4
                  AND state = 'reserved' AND expires_at > ?2
                "#,
                vec![
                    storage_key.into(),
                    now.into(),
                    reservation.write_id.clone().into(),
                    reservation.reservation_token.clone().into(),
                ],
            )?)
            .await?;
        if result.rows_affected() == 0 {
            return Ok(None);
        }
        get_provider_write_with(db, &reservation.write_id).await
    }

    pub(crate) async fn abandon_provider_write(
        &self,
        reservation: &ProviderWriteReservation,
        error: &str,
    ) -> anyhow::Result<bool> {
        let now = self.now_millis();
        let available_at = i64::try_from(reservation.expires_at)?.max(now);
        let db = self.db.connection().await?;
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE provider_write_reservations
                SET state = 'cleanup_pending', updated_at = ?1, available_at = ?2,
                    last_error = ?3, cleanup_claim_token = NULL,
                    cleanup_claim_expires_at = NULL
                WHERE write_id = ?4 AND reservation_token = ?5
                  AND state IN ('reserved', 'stored')
                "#,
                vec![
                    now.into(),
                    available_at.into(),
                    truncate_utf8_bytes(error, MAX_ARTIFACT_CLEANUP_ERROR_BYTES).into(),
                    reservation.write_id.clone().into(),
                    reservation.reservation_token.clone().into(),
                ],
            )?)
            .await?;
        Ok(result.rows_affected() == 1)
    }

    pub(crate) async fn publish_reserved_opaque_artifact(
        &self,
        reservation: &ProviderWriteReservation,
        input: NewMediaAsset,
    ) -> anyhow::Result<Option<MediaAsset>> {
        anyhow::ensure!(
            input.storage_key.as_str() == reservation.storage_key.as_deref().unwrap_or_default()
                && input.content_type == reservation.content_type
                && input.filename == reservation.filename
                && input.size_bytes == reservation.expected_size_bytes
                && input.sha256.as_deref() == Some(reservation.expected_sha256.as_str()),
            "Provider write publication did not match its reservation"
        );
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(runtime_write_transaction_options())
            .await?;
        let now = self.now_millis();
        let row = get_provider_write_with(&tx, &reservation.write_id).await?;
        if row.as_ref().is_none_or(|row| {
            row.reservation_token != reservation.reservation_token
                || row.storage_key != reservation.storage_key
        }) {
            tx.rollback().await?;
            return Ok(None);
        }
        let id = new_uuid();
        let metadata_json = json_to_db_string(&input.metadata_json, "{}")?;
        tx.execute_raw(raw::statement(
            &tx,
            r#"
            INSERT INTO media_assets (
                id, created_at, updated_at, asset_kind, storage_namespace,
                storage_key, content_type, filename, size_bytes, sha256,
                duration_secs, sample_rate_hz, channel_count, peak_amplitude,
                rms_amplitude, source_asset_id, canonical_profile_version,
                scan_status, retention_policy, deleted_at, metadata_json
            ) VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11,
                      ?12, ?13, ?14, ?15, ?16, ?17, ?18, NULL, ?19)
        "#,
            vec![
                id.clone().into(),
                now.into(),
                input.asset_kind.into(),
                input.storage_namespace.into(),
                input.storage_key.into(),
                input.content_type.into(),
                opt_string(input.filename),
                u64_to_i64_value(input.size_bytes)?,
                opt_string(input.sha256),
                opt_f64(input.duration_secs),
                opt_u32(input.sample_rate_hz),
                opt_u16(input.channel_count),
                opt_f32(input.peak_amplitude),
                opt_f32(input.rms_amplitude),
                opt_string(input.source_asset_id),
                opt_string(input.canonical_profile_version),
                input.scan_status.into(),
                input.retention_policy.into(),
                metadata_json.into(),
            ],
        )?)
        .await?;
        let deleted = tx.execute_raw(raw::statement(&tx,
            "DELETE FROM provider_write_reservations WHERE write_id = ?1 AND reservation_token = ?2 AND state = 'stored'",
            vec![reservation.write_id.clone().into(), reservation.reservation_token.clone().into()],
        )?).await?;
        if deleted.rows_affected() != 1 {
            tx.rollback().await?;
            return Ok(None);
        }
        let asset = get_media_asset_with(&tx, &id).await?;
        tx.commit().await?;
        Ok(asset)
    }

    pub(crate) async fn claim_due_provider_write_cleanup(
        &self,
        limit: usize,
    ) -> anyhow::Result<Vec<ProviderWriteReservation>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(runtime_write_transaction_options())
            .await?;
        let now = self.now_millis();
        let request_length = provider_write_request_length_sql(tx.get_database_backend())?;
        let selection_sql = format!(
            r#"
            SELECT write_id FROM provider_write_reservations
            WHERE (
                (state IN ('reserved', 'stored', 'cleanup_pending') AND available_at <= ?1)
                OR (state = 'cleanup_claimed' AND cleanup_claim_expires_at <= ?1)
            )
              AND (provider_request_json IS NOT NULL OR storage_namespace = 'artifact-store')
              AND (provider_request_json IS NULL OR {request_length} <= ?2)
            ORDER BY available_at ASC, created_at ASC, write_id ASC LIMIT ?3
            "#
        );
        let rows = tx
            .query_all_raw(raw::statement(
                &tx,
                selection_sql,
                vec![
                    now.into(),
                    i64::try_from(MAX_PROVIDER_WRITE_REQUEST_ENVELOPE_BYTES)?.into(),
                    i64::try_from(limit.min(MAX_PROVIDER_WRITE_CLEANUP_BATCH))?.into(),
                ],
            )?)
            .await?;
        let mut claimed = Vec::with_capacity(rows.len());
        for row in rows {
            let write_id: String = row.try_get_by_index(0)?;
            let claim_token = new_uuid();
            let claim_expires_at =
                now.saturating_add(i64::try_from(PROVIDER_WRITE_CLEANUP_CLAIM_MS)?);
            let result = tx
                .execute_raw(raw::statement(
                    &tx,
                    r#"
                UPDATE provider_write_reservations
                SET state = 'cleanup_claimed', updated_at = ?1,
                    cleanup_claim_token = ?2, cleanup_claim_expires_at = ?3
                WHERE write_id = ?4 AND (
                    (state IN ('reserved', 'stored', 'cleanup_pending') AND available_at <= ?1)
                    OR (state = 'cleanup_claimed' AND cleanup_claim_expires_at <= ?1)
                )
            "#,
                    vec![
                        now.into(),
                        claim_token.into(),
                        claim_expires_at.into(),
                        write_id.clone().into(),
                    ],
                )?)
                .await?;
            if result.rows_affected() == 1 {
                match get_provider_write_with(&tx, &write_id).await {
                    Ok(Some(reservation)) => claimed.push(reservation),
                    Ok(None) => {
                        quarantine_invalid_provider_write(
                            &tx,
                            &write_id,
                            now,
                            "Provider write request envelope could not be loaded safely",
                        )
                        .await?;
                    }
                    Err(error) => {
                        quarantine_invalid_provider_write(&tx, &write_id, now, &error.to_string())
                            .await?;
                    }
                }
            }
        }
        tx.commit().await?;
        Ok(claimed)
    }

    pub(crate) async fn complete_provider_write_cleanup(
        &self,
        reservation: &ProviderWriteReservation,
    ) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let result = db.execute_raw(raw::statement(db,
            "DELETE FROM provider_write_reservations WHERE write_id = ?1 AND state = 'cleanup_claimed' AND cleanup_claim_token = ?2",
            vec![reservation.write_id.clone().into(), opt_string(reservation.cleanup_claim_token.clone())],
        )?).await?;
        Ok(result.rows_affected() == 1)
    }

    pub(crate) async fn defer_provider_write_cleanup(
        &self,
        reservation: &ProviderWriteReservation,
        error: &str,
    ) -> anyhow::Result<bool> {
        let attempt = reservation.cleanup_attempt_count.saturating_add(1);
        let backoff = 1_000_u64
            .checked_shl(reservation.cleanup_attempt_count.min(12))
            .unwrap_or(MAX_ARTIFACT_CLEANUP_BACKOFF_MS)
            .min(MAX_ARTIFACT_CLEANUP_BACKOFF_MS);
        let now = self.now_millis();
        let db = self.db.connection().await?;
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
            UPDATE provider_write_reservations
            SET state = 'cleanup_pending', updated_at = ?1, available_at = ?2,
                cleanup_attempt_count = ?3, last_error = ?4,
                cleanup_claim_token = NULL, cleanup_claim_expires_at = NULL
            WHERE write_id = ?5 AND state = 'cleanup_claimed' AND cleanup_claim_token = ?6
        "#,
                vec![
                    now.into(),
                    now.saturating_add(i64::try_from(backoff)?).into(),
                    i64::from(attempt).into(),
                    truncate_utf8_bytes(error, MAX_ARTIFACT_CLEANUP_ERROR_BYTES).into(),
                    reservation.write_id.clone().into(),
                    opt_string(reservation.cleanup_claim_token.clone()),
                ],
            )?)
            .await?;
        Ok(result.rows_affected() == 1)
    }

    fn provider_write_capacity(&self) -> u64 {
        #[cfg(test)]
        if let Some(capacity) = self.test_provider_write_capacity {
            return capacity;
        }
        MAX_PROVIDER_WRITE_RESERVATIONS
    }

    pub async fn get_media_asset_by_storage_key(
        &self,
        storage_key: &str,
    ) -> anyhow::Result<Option<MediaAsset>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                MEDIA_ASSET_BY_STORAGE_KEY_SQL,
                vec![storage_key.into()],
            )?)
            .await
            .context("Failed to load media asset by storage key")?;

        row.as_ref().map(map_media_asset).transpose()
    }

    pub async fn get_canonical_media_asset(
        &self,
        source_asset_id: &str,
        canonical_profile_version: &str,
    ) -> anyhow::Result<Option<MediaAsset>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                MEDIA_ASSET_BY_SOURCE_PROFILE_SQL,
                vec![source_asset_id.into(), canonical_profile_version.into()],
            )?)
            .await
            .context("Failed to load canonical media asset by source and profile")?;

        row.as_ref().map(map_media_asset).transpose()
    }

    pub async fn create_text_asset(&self, input: NewTextAsset) -> anyhow::Result<TextAsset> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let id = new_uuid();
        let normalized_text = input
            .normalized_text
            .clone()
            .unwrap_or_else(|| input.raw_text.clone());
        let character_count = normalized_text.chars().count() as u64;
        let structure_json = json_to_db_string(&input.structure_json, "{}")?;

        db.execute_raw(raw::statement(
            db,
            r#"
            INSERT INTO text_assets (
                id,
                created_at,
                updated_at,
                raw_text,
                normalized_text,
                language_hint,
                character_count,
                sha256,
                safety_status,
                retention_policy,
                structure_json
            )
            VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            "#,
            vec![
                id.clone().into(),
                now.into(),
                input.raw_text.into(),
                normalized_text.into(),
                opt_string(input.language_hint),
                u64_to_i64_value(character_count)?,
                opt_string(input.sha256),
                input.safety_status.into(),
                input.retention_policy.into(),
                structure_json.into(),
            ],
        )?)
        .await
        .context("Failed to create text asset")?;

        self.get_text_asset(&id)
            .await?
            .ok_or_else(|| anyhow!("Created text asset was not found"))
    }

    pub async fn get_text_asset(&self, id: &str) -> anyhow::Result<Option<TextAsset>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(db, TEXT_ASSET_COLUMNS_SQL, vec![id.into()])?)
            .await
            .context("Failed to load text asset")?;

        row.as_ref().map(map_text_asset).transpose()
    }

    fn tts_admission_limits(&self) -> (usize, usize) {
        #[cfg(test)]
        if let Some(limits) = self.test_tts_admission_limits {
            return limits;
        }
        let read = |key, default| {
            std::env::var(key)
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(default)
        };
        (
            read("IZWI_TTS_MAX_ACTIVE_JOBS", 256),
            read("IZWI_TTS_MAX_ACTIVE_JOBS_PER_TENANT", 32),
        )
    }

    async fn check_tts_admission(
        &self,
        tx: &sea_orm::DatabaseTransaction,
        tenant: &str,
    ) -> anyhow::Result<()> {
        // The unique insert and row update serialize admission across processes and
        // PostgreSQL connections; a read/count alone would allow concurrent overflow.
        tx.execute_raw(raw::statement(tx,
            "INSERT INTO runtime_admission_locks (id, lock_value) VALUES ('tts', 1) ON CONFLICT (id) DO NOTHING", vec![])?)
            .await?;
        tx.execute_raw(raw::statement(
            tx,
            "UPDATE runtime_admission_locks SET lock_value = 1 WHERE id = 'tts'",
            vec![],
        )?)
        .await?;
        let row = tx.query_one_raw(raw::statement(tx,
            "SELECT COUNT(*), COALESCE(SUM(CASE WHEN COALESCE(admission_tenant, 'anonymous') = ?1 THEN 1 ELSE 0 END), 0) FROM runtime_jobs WHERE job_kind = 'tts_speech' AND status IN ('created','queued','running','paused','retrying','postprocessing')",
            vec![tenant.into()],
        )?).await?.ok_or_else(|| anyhow!("Speech admission count returned no row"))?;
        let active = u64::try_from(row.try_get_by_index::<i64>(0)?)?;
        let owned = u64::try_from(row.try_get_by_index::<i64>(1)?)?;
        let (global_limit, tenant_limit) = self.tts_admission_limits();
        if active >= global_limit as u64 || owned >= tenant_limit as u64 {
            bail!("Speech job admission capacity exhausted: active={active}/{global_limit}, tenant={owned}/{tenant_limit}");
        }
        Ok(())
    }

    /// Avoid expensive input ingestion during overload. The create transaction
    /// still performs the authoritative check because preflight reserves no slot.
    pub async fn preflight_speech_admission(
        &self,
        tenant_key: Option<[u8; 32]>,
    ) -> anyhow::Result<()> {
        let tenant = speech_admission_tenant(&json!({"tenant_key": tenant_key}))?;
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(runtime_write_transaction_options())
            .await?;
        self.check_tts_admission(&tx, &tenant).await?;
        tx.rollback().await?;
        Ok(())
    }

    pub async fn remove_unreferenced_text_asset(&self, id: &str) -> anyhow::Result<()> {
        let db = self.db.connection().await?;
        db.execute_raw(raw::statement(db,
            "DELETE FROM text_assets WHERE id = ?1 AND NOT EXISTS (SELECT 1 FROM runtime_jobs WHERE input_text_asset_id = ?1) AND NOT EXISTS (SELECT 1 FROM runtime_artifacts WHERE text_asset_id = ?1)",
            vec![id.into()],
        )?).await?;
        Ok(())
    }

    pub async fn create_job(&self, input: NewRuntimeJob) -> anyhow::Result<RuntimeJob> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(runtime_write_transaction_options())
            .await?;
        let admission_tenant = if input.job_kind == RuntimeJobKind::TtsSpeech {
            let tenant = speech_admission_tenant(&input.request_json)?;
            if !is_terminal_job_status(input.status) {
                self.check_tts_admission(&tx, &tenant).await?;
            }
            Some(tenant)
        } else {
            None
        };
        let now = self.now_millis();
        let id = new_uuid();
        let request_json = json_to_db_string(&input.request_json, "{}")?;
        let model_snapshot_json = json_to_db_string(&input.model_snapshot_json, "{}")?;
        let retry_policy_json = json_to_db_string(&input.retry_policy_json, "{}")?;
        let queued_at = matches!(input.status, RuntimeJobStatus::Queued).then_some(now);
        let started_at = matches!(input.status, RuntimeJobStatus::Running).then_some(now);
        let finished_at = is_terminal_job_status(input.status).then_some(now);

        tx.execute_raw(raw::statement(
            &tx,
            r#"
            INSERT INTO runtime_jobs (
                id,
                created_at,
                updated_at,
                queued_at,
                started_at,
                finished_at,
                job_kind,
                status,
                priority,
                model_id,
                capability,
                route_record_kind,
                route_record_id,
                input_media_asset_id,
                input_text_asset_id,
                request_json,
                model_snapshot_json,
                progress_json,
                error_code,
                error_message,
                attempt_count,
                max_attempts,
                retry_policy_json,
                idempotency_key,
                correlation_id,
                cancellation_reason,
                admission_tenant
            )
            VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, NULL, NULL, NULL, 0, ?17, ?18, ?19, ?20, NULL, ?21)
            "#,
            vec![
                id.clone().into(),
                now.into(),
                opt_i64(queued_at),
                opt_i64(started_at),
                opt_i64(finished_at),
                input.job_kind.as_db_value().into(),
                input.status.as_db_value().into(),
                input.priority.into(),
                opt_string(input.model_id),
                opt_string(input.capability),
                opt_string(input.route_record_kind),
                opt_string(input.route_record_id),
                opt_string(input.input_media_asset_id),
                opt_string(input.input_text_asset_id),
                request_json.into(),
                model_snapshot_json.into(),
                u32_to_i64_value(input.max_attempts).into(),
                retry_policy_json.into(),
                opt_string(input.idempotency_key),
                opt_string(input.correlation_id),
                opt_string(admission_tenant),
            ],
        )?)
        .await
        .context("Failed to create runtime job")?;

        tx.commit().await?;
        self.get_job(&id)
            .await?
            .ok_or_else(|| anyhow!("Created runtime job was not found"))
    }

    /// Atomically publish one text-only TTS job graph and, when supplied, its
    /// durable idempotency result. No worker can observe the queued stage
    /// before the projection, input, job, and replay record are all committed.
    pub async fn accept_durable_text_tts(
        &self,
        input: NewDurableTextTtsAcceptance,
    ) -> anyhow::Result<DurableTextTtsAcceptanceOutcome> {
        anyhow::ensure!(
            input.projection.route_kind == SpeechRouteKind::TextToSpeech,
            "Durable text TTS acceptance requires a text-to-speech projection"
        );
        anyhow::ensure!(
            input.projection.processing_status == SpeechHistoryProcessingStatus::Pending
                && input.projection.processing_error.is_none(),
            "Durable text TTS acceptance requires a clean pending projection"
        );
        anyhow::ensure!(
            input.projection.audio_bytes.is_empty()
                && input.projection.reference_text.is_none()
                && input.projection.saved_voice_id.is_none(),
            "Durable text TTS acceptance does not accept reference or output media"
        );
        anyhow::ensure!(
            input.projection.generation_time_ms == 0.0
                && input.projection.audio_duration_secs.is_none()
                && input.projection.rtf.is_none()
                && input.projection.tokens_generated.is_none(),
            "Durable text TTS acceptance requires an unexecuted projection"
        );
        anyhow::ensure!(
            input.max_attempts > 0,
            "Durable text TTS acceptance requires at least one attempt"
        );
        anyhow::ensure!(
            input.queue_class == QueueClass::BatchTts,
            "Durable text TTS acceptance requires the batch TTS queue"
        );
        validate_bounded_field("stage kind", &input.stage_kind, 128)?;

        anyhow::ensure!(
            !input.projection.input_text.is_empty()
                && input.projection.input_text.len() <= MAX_DURABLE_TTS_TEXT_BYTES,
            "Durable text TTS input must be between 1 and {MAX_DURABLE_TTS_TEXT_BYTES} bytes"
        );
        for (name, value) in [
            ("model ID", input.projection.model_id.as_deref()),
            ("speaker", input.projection.speaker.as_deref()),
            ("language", input.projection.language.as_deref()),
            (
                "voice description",
                input.projection.voice_description.as_deref(),
            ),
            ("audio filename", input.projection.audio_filename.as_deref()),
        ] {
            validate_optional_field_bytes(name, value, MAX_DURABLE_TTS_OPTIONAL_FIELD_BYTES)?;
        }
        validate_optional_field_bytes(
            "correlation ID",
            input.correlation_id.as_deref(),
            MAX_DURABLE_TTS_CORRELATION_BYTES,
        )?;
        anyhow::ensure!(
            input.projection.audio_mime_type.len() <= MAX_DURABLE_TTS_OPTIONAL_FIELD_BYTES,
            "Durable text TTS audio MIME type exceeds {MAX_DURABLE_TTS_OPTIONAL_FIELD_BYTES} bytes"
        );

        let model_id = sanitize_optional_text(input.projection.model_id.as_deref(), 160)
            .ok_or_else(|| anyhow!("Durable text TTS acceptance requires a model ID"))?;
        let speaker = sanitize_optional_text(input.projection.speaker.as_deref(), 120);
        let language = sanitize_optional_text(input.projection.language.as_deref(), 80);
        let voice_description =
            sanitize_optional_text(input.projection.voice_description.as_deref(), 2_000);
        let input_text = input.projection.input_text.trim().to_string();
        anyhow::ensure!(
            !input_text.is_empty(),
            "Durable text TTS acceptance requires non-empty input text"
        );
        let speed = input
            .projection
            .speed
            .filter(|value| value.is_finite() && *value > 0.0);
        let audio_mime_type = sanitize_audio_mime_type(&input.projection.audio_mime_type);
        let audio_filename =
            sanitize_optional_text(input.projection.audio_filename.as_deref(), 260);
        let request_json_string = bounded_json_string(
            &input.request_json,
            MAX_DURABLE_TTS_REQUEST_JSON_BYTES,
            "Durable text TTS request",
        )?;
        let model_snapshot_json_string = bounded_json_string(
            &input.model_snapshot_json,
            MAX_DURABLE_TTS_METADATA_JSON_BYTES,
            "Durable text TTS model snapshot",
        )?;
        let retry_policy_json_string = bounded_json_string(
            &input.retry_policy_json,
            MAX_DURABLE_TTS_METADATA_JSON_BYTES,
            "Durable text TTS retry policy",
        )?;
        let resource_hints = input.resource_hints.normalized();
        let resource_hints_json = bounded_json_string(
            &json!(resource_hints.clone()),
            MAX_DURABLE_TTS_METADATA_JSON_BYTES,
            "Durable text TTS resource hints",
        )?;
        let admission_tenant = speech_admission_tenant(&input.request_json)?;

        if let Some(reservation) = input.reservation.as_ref() {
            validate_durable_idempotency_identity(
                &reservation.tenant_scope,
                &reservation.operation,
                &reservation.idempotency_key,
                reservation.digest_version,
                &reservation.request_digest,
            )?;
            validate_bounded_field("reservation token", &reservation.reservation_token, 64)?;
            anyhow::ensure!(
                (1..=MAX_DURABLE_IDEMPOTENCY_RETENTION_MS)
                    .contains(&input.idempotency_retention_ms),
                "Durable idempotency retention must be between 1 and {MAX_DURABLE_IDEMPOTENCY_RETENTION_MS} milliseconds"
            );
        }

        let now = self.now_millis();
        let now_u64 = nonnegative_timestamp(now)?;
        let record_id = new_uuid();
        let text_asset_id = new_uuid();
        let job_id = new_uuid();
        let artifact_id = new_uuid();
        let stage_id = new_uuid();
        let text_sha256 = sha256_hex(input_text.as_bytes());
        let record = SpeechHistoryRecord {
            id: record_id.clone(),
            created_at: now_u64,
            route_kind: SpeechRouteKind::TextToSpeech,
            processing_status: SpeechHistoryProcessingStatus::Pending,
            processing_error: None,
            model_id: Some(model_id.clone()),
            speaker: speaker.clone(),
            language: language.clone(),
            saved_voice_id: None,
            speed,
            input_text: input_text.clone(),
            voice_description: voice_description.clone(),
            reference_text: None,
            generation_time_ms: 0.0,
            audio_duration_secs: None,
            rtf: None,
            tokens_generated: None,
            audio_mime_type: audio_mime_type.clone(),
            audio_filename: audio_filename.clone(),
        };
        // Persist only a bounded lookup key. Replays load the route projection
        // by ID, avoiding a second copy of potentially 1 MiB input text.
        let response_json = json!({"record_id": record_id.clone()});
        let response_json_string = input
            .reservation
            .as_ref()
            .map(|_| {
                bounded_json_string(
                    &response_json,
                    MAX_DURABLE_IDEMPOTENCY_RESULT_BYTES,
                    "Durable idempotency result",
                )
            })
            .transpose()?;
        let structure_json = json_to_db_string(
            &json!({"source": "tts_input", "route_record_id": record_id.clone()}),
            "{}",
        )?;
        let artifact_metadata_json = json_to_db_string(
            &json!({"route_record_id": record_id.clone(), "input_kind": "text"}),
            "{}",
        )?;
        let input_artifact_ids_json = json_to_db_string(&json!([artifact_id.clone()]), "[]")?;

        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(runtime_write_transaction_options())
            .await
            .context("Failed to start durable text TTS acceptance transaction")?;

        if let Some(reservation) = input.reservation.as_ref() {
            lock_durable_idempotency(&tx).await?;
            let Some(stored) = load_durable_idempotency_with(
                &tx,
                &reservation.tenant_scope,
                &reservation.operation,
                &reservation.idempotency_key,
            )
            .await?
            else {
                tx.rollback().await?;
                return Ok(DurableTextTtsAcceptanceOutcome::ReservationLost);
            };
            if stored.state != "reserved"
                || stored.expires_at <= now_u64
                || stored.digest_version != reservation.digest_version
                || stored.request_digest != reservation.request_digest
                || stored.reservation_token != reservation.reservation_token
            {
                tx.rollback().await?;
                return Ok(DurableTextTtsAcceptanceOutcome::ReservationLost);
            }
        }

        self.check_tts_admission(&tx, &admission_tenant).await?;
        tx.execute_raw(raw::statement(
            &tx,
            r#"
            INSERT INTO speech_history_records (
                id, created_at, route_kind, processing_status, processing_error,
                runtime_stage_id, runtime_attempt_token, model_id, speaker,
                language, saved_voice_id, speed, input_text, voice_description,
                reference_text, generation_time_ms, audio_duration_secs, rtf,
                tokens_generated, audio_mime_type, audio_filename, audio_storage_path
            )
            VALUES (?1, ?2, 'text_to_speech', 'pending', NULL, NULL, NULL, ?3,
                    ?4, ?5, NULL, ?6, ?7, ?8, NULL, 0.0, NULL, NULL, NULL,
                    ?9, ?10, '')
            "#,
            vec![
                record_id.clone().into(),
                now.into(),
                model_id.clone().into(),
                opt_string(speaker),
                opt_string(language.clone()),
                opt_f64(speed),
                input_text.clone().into(),
                opt_string(voice_description),
                audio_mime_type.clone().into(),
                opt_string(audio_filename.clone()),
            ],
        )?)
        .await
        .context("Failed to create durable text TTS projection")?;
        self.inject_durable_tts_acceptance_failure(DurableTtsAcceptanceFailpoint::Projection)?;

        tx.execute_raw(raw::statement(
            &tx,
            r#"
            INSERT INTO text_assets (
                id, created_at, updated_at, raw_text, normalized_text,
                language_hint, character_count, sha256, safety_status,
                retention_policy, structure_json
            )
            VALUES (?1, ?2, ?2, ?3, ?3, ?4, ?5, ?6, 'unchecked', 'default', ?7)
            "#,
            vec![
                text_asset_id.clone().into(),
                now.into(),
                input_text.clone().into(),
                opt_string(language),
                u64_to_i64_value(input_text.chars().count() as u64)?,
                text_sha256.clone().into(),
                structure_json.into(),
            ],
        )?)
        .await
        .context("Failed to create durable text TTS input")?;
        self.inject_durable_tts_acceptance_failure(DurableTtsAcceptanceFailpoint::TextAsset)?;

        tx.execute_raw(raw::statement(
            &tx,
            r#"
            INSERT INTO runtime_jobs (
                id, created_at, updated_at, queued_at, started_at, finished_at,
                job_kind, status, priority, model_id, capability,
                route_record_kind, route_record_id, input_media_asset_id,
                input_text_asset_id, request_json, model_snapshot_json,
                progress_json, error_code, error_message, attempt_count,
                max_attempts, retry_policy_json, idempotency_key,
                correlation_id, cancellation_reason, admission_tenant
            )
            VALUES (?1, ?2, ?2, ?2, NULL, NULL, 'tts_speech', 'queued', ?3,
                    ?4, 'tts', 'text_to_speech', ?5, NULL, ?6, ?7, ?8,
                    NULL, NULL, NULL, 0, ?9, ?10, ?11, ?12, NULL, ?13)
            "#,
            vec![
                job_id.clone().into(),
                now.into(),
                input.priority.into(),
                model_id.clone().into(),
                record_id.clone().into(),
                text_asset_id.clone().into(),
                request_json_string.into(),
                model_snapshot_json_string.into(),
                u32_to_i64_value(input.max_attempts).into(),
                retry_policy_json_string.into(),
                opt_string(
                    input
                        .reservation
                        .as_ref()
                        .map(|reservation| reservation.idempotency_key.clone()),
                ),
                opt_string(input.correlation_id),
                admission_tenant.into(),
            ],
        )?)
        .await
        .context("Failed to create durable text TTS job")?;
        self.inject_durable_tts_acceptance_failure(DurableTtsAcceptanceFailpoint::Job)?;

        tx.execute_raw(raw::statement(
            &tx,
            r#"
            INSERT INTO runtime_artifacts (
                id, job_id, stage_id, created_at, artifact_kind,
                artifact_role, media_asset_id, text_asset_id, storage_key,
                content_type, filename, size_bytes, sha256, metadata_json,
                retention_policy
            )
            VALUES (?1, ?2, NULL, ?3, 'text', 'input_original', NULL, ?4,
                    NULL, 'text/plain', ?5, ?6, ?7, ?8, 'default')
            "#,
            vec![
                artifact_id.clone().into(),
                job_id.clone().into(),
                now.into(),
                text_asset_id.clone().into(),
                format!("{record_id}.input.txt").into(),
                u64_to_i64_value(input_text.len() as u64)?,
                text_sha256.into(),
                artifact_metadata_json.into(),
            ],
        )?)
        .await
        .context("Failed to create durable text TTS input artifact")?;
        self.inject_durable_tts_acceptance_failure(DurableTtsAcceptanceFailpoint::Artifact)?;

        tx.execute_raw(raw::statement(
            &tx,
            r#"
            INSERT INTO job_stages (
                id, job_id, created_at, updated_at, sequence, stage_kind,
                queue_class, resource_hints_json, resource_target,
                required_backend, required_device_class,
                min_resource_memory_bytes, resource_concurrency_weight,
                status, capability, model_id, worker_id, lease_expires_at,
                available_at, attempt_token, attempt_count, max_attempts,
                input_artifact_ids_json, output_artifact_ids_json,
                progress_json, started_at, finished_at, error_code,
                error_message, cancellation_state
            )
            VALUES (?1, ?2, ?3, ?3, 0, ?4, ?5, ?6, ?7, ?8, ?9, ?10,
                    ?11, 'queued', 'tts', ?12, NULL, NULL, ?3, NULL, 0, ?13,
                    ?14, '[]', NULL, NULL, NULL, NULL, NULL, NULL)
            "#,
            vec![
                stage_id.clone().into(),
                job_id.clone().into(),
                now.into(),
                input.stage_kind.into(),
                input.queue_class.as_db_value().into(),
                resource_hints_json.into(),
                resource_hints.target.as_db_value().into(),
                opt_string(
                    resource_hints
                        .backend
                        .map(|backend| backend.as_db_value().to_string()),
                ),
                opt_string(
                    resource_hints
                        .device_class
                        .map(|device| device.as_db_value().to_string()),
                ),
                opt_u64(resource_hints.min_memory_bytes),
                u32_to_i64_value(resource_hints.concurrency_weight).into(),
                model_id.into(),
                u32_to_i64_value(input.max_attempts).into(),
                input_artifact_ids_json.into(),
            ],
        )?)
        .await
        .context("Failed to create durable text TTS stage")?;
        self.inject_durable_tts_acceptance_failure(DurableTtsAcceptanceFailpoint::Stage)?;

        if let Some(reservation) = input.reservation.as_ref() {
            let expires_at = now_u64
                .checked_add(input.idempotency_retention_ms)
                .context("Durable idempotency retention expiry overflow")?;
            let result = tx
                .execute_raw(raw::statement(
                    &tx,
                    r#"
                    UPDATE durable_idempotency_keys_v2
                    SET state = 'committed', updated_at = ?1, expires_at = ?2,
                        runtime_job_id = ?3, response_json = ?4
                    WHERE tenant_scope = ?5 AND operation = ?6
                      AND idempotency_key = ?7 AND state = 'reserved'
                      AND reservation_token = ?8 AND digest_version = ?9
                      AND request_digest = ?10 AND expires_at > ?1
                    "#,
                    vec![
                        now.into(),
                        i64::try_from(expires_at)?.into(),
                        job_id.clone().into(),
                        response_json_string
                            .clone()
                            .expect("keyed response was bounded")
                            .into(),
                        reservation.tenant_scope.clone().into(),
                        reservation.operation.clone().into(),
                        reservation.idempotency_key.clone().into(),
                        reservation.reservation_token.clone().into(),
                        i64::from(reservation.digest_version).into(),
                        reservation.request_digest.clone().into(),
                    ],
                )?)
                .await
                .context("Failed to commit durable text TTS idempotency result")?;
            if result.rows_affected() != 1 {
                tx.rollback().await?;
                return Ok(DurableTextTtsAcceptanceOutcome::ReservationLost);
            }
            self.inject_durable_tts_acceptance_failure(DurableTtsAcceptanceFailpoint::Idempotency)?;
        }

        let job = get_job_with(&tx, &job_id)
            .await?
            .ok_or_else(|| anyhow!("Created durable text TTS job was not found"))?;
        let stage = get_stage_with(&tx, &stage_id)
            .await?
            .ok_or_else(|| anyhow!("Created durable text TTS stage was not found"))?;
        let input_artifact = get_artifact_with(&tx, &artifact_id)
            .await?
            .ok_or_else(|| anyhow!("Created durable text TTS artifact was not found"))?;
        tx.commit()
            .await
            .context("Failed to commit durable text TTS acceptance transaction")?;

        Ok(DurableTextTtsAcceptanceOutcome::Committed(
            DurableTextTtsAcceptance {
                record,
                job,
                stage,
                input_artifact,
            },
        ))
    }

    pub async fn get_job(&self, id: &str) -> anyhow::Result<Option<RuntimeJob>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                RUNTIME_JOB_COLUMNS_SQL,
                vec![id.into()],
            )?)
            .await
            .context("Failed to load runtime job")?;

        row.as_ref().map(map_runtime_job).transpose()
    }

    pub async fn list_active_jobs_by_kind(
        &self,
        job_kind: RuntimeJobKind,
    ) -> anyhow::Result<Vec<RuntimeJob>> {
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(
                db,
                r#"
                SELECT id, created_at, updated_at, queued_at, started_at, finished_at,
                       job_kind, status, priority, model_id, capability,
                       route_record_kind, route_record_id, input_media_asset_id,
                       input_text_asset_id, request_json, model_snapshot_json,
                       progress_json, error_code, error_message, attempt_count,
                       max_attempts, retry_policy_json, idempotency_key,
                       correlation_id, cancellation_reason, cancellation_state
                FROM runtime_jobs
                WHERE job_kind = ?1
                  AND status IN ('created', 'queued', 'running', 'paused', 'retrying', 'postprocessing')
                ORDER BY created_at ASC, id ASC
                "#,
                vec![job_kind.as_db_value().into()],
            )?)
            .await
            .context("Failed to list active runtime jobs by kind")?;

        rows.iter().map(map_runtime_job).collect()
    }

    pub async fn get_active_job_for_route_record(
        &self,
        job_kind: RuntimeJobKind,
        route_record_kind: &str,
        route_record_id: &str,
    ) -> anyhow::Result<Option<RuntimeJob>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                r#"
                SELECT id, created_at, updated_at, queued_at, started_at, finished_at,
                       job_kind, status, priority, model_id, capability,
                       route_record_kind, route_record_id, input_media_asset_id,
                       input_text_asset_id, request_json, model_snapshot_json,
                       progress_json, error_code, error_message, attempt_count,
                       max_attempts, retry_policy_json, idempotency_key,
                       correlation_id, cancellation_reason, cancellation_state
                FROM runtime_jobs
                WHERE job_kind = ?1
                  AND route_record_kind = ?2
                  AND route_record_id = ?3
                  AND status IN ('created', 'queued', 'running', 'paused', 'retrying', 'postprocessing')
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                "#,
                vec![
                    job_kind.as_db_value().into(),
                    route_record_kind.into(),
                    route_record_id.into(),
                ],
            )?)
            .await
            .context("Failed to load active runtime job for route record")?;

        row.as_ref().map(map_runtime_job).transpose()
    }

    pub async fn get_latest_job_for_route_record(
        &self,
        job_kind: RuntimeJobKind,
        route_record_kind: &str,
        route_record_id: &str,
    ) -> anyhow::Result<Option<RuntimeJob>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                r#"
                SELECT id, created_at, updated_at, queued_at, started_at, finished_at,
                       job_kind, status, priority, model_id, capability,
                       route_record_kind, route_record_id, input_media_asset_id,
                       input_text_asset_id, request_json, model_snapshot_json,
                       progress_json, error_code, error_message, attempt_count,
                       max_attempts, retry_policy_json, idempotency_key,
                       correlation_id, cancellation_reason, cancellation_state
                FROM runtime_jobs
                WHERE job_kind = ?1
                  AND route_record_kind = ?2
                  AND route_record_id = ?3
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                "#,
                vec![
                    job_kind.as_db_value().into(),
                    route_record_kind.into(),
                    route_record_id.into(),
                ],
            )?)
            .await
            .context("Failed to load active runtime job for route record")?;

        row.as_ref().map(map_runtime_job).transpose()
    }

    pub async fn job_status_counts(&self) -> anyhow::Result<Vec<RuntimeJobStatusCount>> {
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(
                db,
                "SELECT status, COUNT(*) FROM runtime_jobs GROUP BY status ORDER BY status",
                vec![],
            )?)
            .await
            .context("Failed to count runtime jobs by status")?;

        rows.iter()
            .map(|row| {
                let status_raw: String = row.try_get_by_index(0)?;
                let status = RuntimeJobStatus::from_db_value(status_raw.as_str())
                    .ok_or_else(|| anyhow!("Unknown runtime job status: {status_raw}"))?;
                let count = i64_to_u64(row.try_get_by_index(1)?)?;
                Ok(RuntimeJobStatusCount { status, count })
            })
            .collect()
    }

    pub async fn transition_job_status(
        &self,
        job_id: &str,
        expected_statuses: &[RuntimeJobStatus],
        next_status: RuntimeJobStatus,
        error_code: Option<String>,
        error_message: Option<String>,
        cancellation_reason: Option<String>,
    ) -> anyhow::Result<Option<RuntimeJob>> {
        if expected_statuses.is_empty() {
            bail!("At least one expected status is required for runtime job transitions");
        }

        let db = self.db.connection().await?;
        let now = self.now_millis();
        let expected_placeholders = (0..expected_statuses.len())
            .map(|index| format!("?{}", index + 7))
            .collect::<Vec<_>>()
            .join(", ");

        let sql = format!(
            r#"
            UPDATE runtime_jobs
            SET
                status = ?1,
                updated_at = ?2,
                queued_at = CASE WHEN ?1 = 'queued' THEN COALESCE(queued_at, ?2) ELSE queued_at END,
                started_at = CASE WHEN ?1 = 'running' THEN COALESCE(started_at, ?2) ELSE started_at END,
                finished_at = CASE WHEN ?1 IN ('completed', 'failed', 'cancelled', 'expired') THEN COALESCE(finished_at, ?2) ELSE finished_at END,
                error_code = ?3,
                error_message = ?4,
                cancellation_reason = CASE WHEN ?1 = 'cancelled' THEN ?5 ELSE cancellation_reason END,
                cancellation_state = CASE
                    WHEN ?1 IN ('completed', 'failed', 'cancelled', 'expired') THEN NULL
                    ELSE cancellation_state
                END
            WHERE id = ?6
              AND status IN ({expected_placeholders})
              AND cancellation_state IS NULL
            "#
        );
        let mut values = vec![
            next_status.as_db_value().into(),
            now.into(),
            opt_string(error_code),
            opt_string(error_message),
            opt_string(cancellation_reason),
            job_id.into(),
        ];
        values.extend(
            expected_statuses
                .iter()
                .map(|status| status.as_db_value().into()),
        );

        let result = db
            .execute_raw(raw::statement(db, sql, values)?)
            .await
            .context("Failed to transition runtime job status")?;
        if result.rows_affected() == 0 {
            return Ok(None);
        }

        self.get_job(job_id).await
    }

    pub async fn retry_job(&self, job_id: &str) -> anyhow::Result<Option<RuntimeJob>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(runtime_write_transaction_options())
            .await
            .context("Failed to start runtime job retry transaction")?;
        let Some(job) = get_job_with(&tx, job_id).await? else {
            tx.rollback().await?;
            return Ok(None);
        };
        if !matches!(
            job.status,
            RuntimeJobStatus::Failed | RuntimeJobStatus::Cancelled | RuntimeJobStatus::Expired
        ) {
            tx.rollback().await?;
            return Ok(None);
        }
        if job.error_code.as_deref() == Some("speech_replay_expired")
            || job.attempt_count >= job.max_attempts
        {
            tx.rollback().await?;
            return Ok(None);
        }

        let retryable_stage_counts = tx
            .query_one_raw(raw::statement(
                &tx,
                r#"
                SELECT
                    COUNT(*),
                    SUM(CASE WHEN attempt_count < max_attempts THEN 1 ELSE 0 END)
                FROM job_stages
                WHERE job_id = ?1
                  AND status IN ('failed', 'cancelled', 'expired')
                "#,
                vec![job_id.into()],
            )?)
            .await
            .context("Failed to count retryable runtime job stages")?
            .ok_or_else(|| anyhow!("Runtime retryable stage count returned no row"))?;
        let retryable_stage_count = retryable_stage_counts.try_get_by_index::<i64>(0)?;
        let eligible_stage_count = retryable_stage_counts
            .try_get_by_index::<Option<i64>>(1)?
            .unwrap_or(0);
        if retryable_stage_count == 0 || eligible_stage_count != retryable_stage_count {
            tx.rollback().await?;
            return Ok(None);
        }

        let admission_tenant = if job.job_kind == RuntimeJobKind::TtsSpeech {
            let tenant = speech_admission_tenant(&job.request_json)?;
            self.check_tts_admission(&tx, &tenant).await?;
            Some(tenant)
        } else {
            None
        };
        let now = self.now_millis();
        let result = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE runtime_jobs
                SET
                    status = 'queued',
                    updated_at = ?1,
                    queued_at = ?1,
                    started_at = NULL,
                    finished_at = NULL,
                    error_code = NULL,
                    error_message = NULL,
                    attempt_count = attempt_count + 1,
                    cancellation_reason = NULL,
                    cancellation_state = NULL,
                    admission_tenant = ?3
                WHERE id = ?2
                  AND status IN ('failed', 'cancelled', 'expired')
                  AND (error_code IS NULL OR error_code <> 'speech_replay_expired')
                  AND attempt_count < max_attempts
                "#,
                vec![now.into(), job_id.into(), opt_string(admission_tenant)],
            )?)
            .await
            .context("Failed to retry runtime job")?;
        if result.rows_affected() == 0 {
            tx.rollback().await?;
            return Ok(None);
        }

        let stages = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
            UPDATE job_stages
            SET
                status = 'retrying',
                updated_at = ?1,
                finished_at = NULL,
                lease_expires_at = NULL,
                worker_id = NULL,
                available_at = ?1,
                attempt_token = NULL,
                cancellation_state = NULL,
                output_artifact_ids_json = '[]',
                error_code = NULL,
                error_message = NULL
            WHERE job_id = ?2
              AND status IN ('failed', 'cancelled', 'expired')
              AND attempt_count < max_attempts
            "#,
                vec![now.into(), job_id.into()],
            )?)
            .await
            .context("Failed to retry runtime job stages")?;
        if stages.rows_affected() != u64::try_from(retryable_stage_count)? {
            bail!("Runtime job retry changed an unexpected number of stages");
        }

        tx.commit()
            .await
            .context("Failed to commit runtime job retry transaction")?;
        self.get_job(job_id).await
    }

    pub async fn claim_next_stage(
        &self,
        worker_id: &str,
        lease_duration_ms: u64,
    ) -> anyhow::Result<Option<ClaimedStage>> {
        self.claim_next_stage_with_filter(
            worker_id,
            lease_duration_ms,
            &StageClaimFilter::default(),
        )
        .await
    }

    pub async fn claim_next_stage_with_filter(
        &self,
        worker_id: &str,
        lease_duration_ms: u64,
        filter: &StageClaimFilter,
    ) -> anyhow::Result<Option<ClaimedStage>> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let lease_expires_at = now.saturating_add(i64::try_from(lease_duration_ms)?);
        let filter = filter.normalized();
        let mut params: Vec<Value> = vec![now.into()];
        let mut claim_filter_sql = String::new();
        push_claim_queue_clause(&mut claim_filter_sql, &mut params, &filter.queue_names);
        push_claim_resource_clause(&mut claim_filter_sql, &mut params, &filter.resources);
        push_claim_string_filter_clause(
            &mut claim_filter_sql,
            &mut params,
            "COALESCE(s.capability, j.capability)",
            &filter.capabilities,
        );
        push_claim_string_filter_clause(
            &mut claim_filter_sql,
            &mut params,
            "COALESCE(s.model_id, j.model_id)",
            &filter.model_ids,
        );
        push_claim_string_filter_clause(
            &mut claim_filter_sql,
            &mut params,
            "s.stage_kind",
            &filter.stage_kinds,
        );
        let limit_placeholder = params.len() + 1;
        params.push(i64::try_from(filter.candidate_limit())?.into());

        let rows = db
            .query_all_raw(raw::statement(
                db,
                format!(
                    r#"
                SELECT
                    s.id,
                    s.stage_kind,
                    j.job_kind,
                    COALESCE(s.capability, j.capability),
                    COALESCE(s.model_id, j.model_id),
                    s.queue_class,
                    s.resource_hints_json
                FROM job_stages s
                INNER JOIN runtime_jobs j ON j.id = s.job_id
                WHERE s.status IN ('queued', 'retrying')
                  AND (s.available_at IS NULL OR s.available_at <= ?1)
                  AND (s.lease_expires_at IS NULL OR s.lease_expires_at <= ?1)
                  AND j.status IN ('created', 'queued', 'running', 'retrying', 'postprocessing')
                  AND NOT EXISTS (
                      SELECT 1
                      FROM job_stages predecessor
                      WHERE predecessor.job_id = s.job_id
                        AND predecessor.sequence < s.sequence
                        AND predecessor.status NOT IN ('completed', 'skipped')
                  )
                  {claim_filter_sql}
                ORDER BY j.priority DESC, s.sequence ASC, COALESCE(s.available_at, s.created_at) ASC,
                         CASE WHEN s.started_at IS NULL THEN 0 ELSE 1 END ASC, s.id ASC
                LIMIT ?{limit_placeholder}
                "#,
                ),
                params,
            )?)
            .await
            .context("Failed to select next runtime job stage")?;
        let candidates = rows
            .iter()
            .map(map_stage_claim_candidate)
            .collect::<anyhow::Result<Vec<_>>>()?;
        for candidate in candidates
            .into_iter()
            .filter(|candidate| filter.matches(candidate))
        {
            if let Some(claimed) = self
                .try_claim_stage_candidate(db, candidate, worker_id, now, lease_expires_at)
                .await?
            {
                return Ok(Some(claimed));
            }
        }

        Ok(None)
    }

    async fn try_claim_stage_candidate(
        &self,
        db: &DatabaseConnection,
        candidate: StageClaimCandidate,
        worker_id: &str,
        now: i64,
        lease_expires_at: i64,
    ) -> anyhow::Result<Option<ClaimedStage>> {
        let tx = db
            .begin_with_options(runtime_write_transaction_options())
            .await
            .context("Failed to start runtime stage claim transaction")?;
        let attempt_token = new_uuid();
        let result = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE job_stages
                SET
                    status = 'running',
                    worker_id = ?1,
                    lease_expires_at = ?2,
                    available_at = NULL,
                    attempt_token = ?5,
                    attempt_count = attempt_count + 1,
                    started_at = COALESCE(started_at, ?3),
                    updated_at = ?3,
                    error_code = NULL,
                    error_message = NULL
                WHERE id = ?4
                  AND status IN ('queued', 'retrying')
                  AND (available_at IS NULL OR available_at <= ?3)
                  AND (lease_expires_at IS NULL OR lease_expires_at <= ?3)
                  AND NOT EXISTS (
                      SELECT 1
                      FROM job_stages predecessor
                      WHERE predecessor.job_id = job_stages.job_id
                        AND predecessor.sequence < job_stages.sequence
                        AND predecessor.status NOT IN ('completed', 'skipped')
                  )
                  AND EXISTS (
                      SELECT 1
                      FROM runtime_jobs
                      WHERE runtime_jobs.id = job_stages.job_id
                        AND runtime_jobs.status IN (
                            'created',
                            'queued',
                            'running',
                            'retrying',
                            'postprocessing'
                        )
                  )
                "#,
                vec![
                    worker_id.to_string().into(),
                    lease_expires_at.into(),
                    now.into(),
                    candidate.stage_id.clone().into(),
                    attempt_token.clone().into(),
                ],
            )?)
            .await
            .context("Failed to claim runtime job stage")?;
        if result.rows_affected() == 0 {
            tx.rollback().await?;
            return Ok(None);
        }

        tx.execute_raw(raw::statement(
            &tx,
            r#"
            UPDATE runtime_jobs
            SET
                status = 'running',
                updated_at = ?1,
                started_at = COALESCE(started_at, ?1),
                error_code = NULL,
                error_message = NULL
            WHERE id = (SELECT job_id FROM job_stages WHERE id = ?2)
              AND status IN ('created', 'queued', 'running', 'retrying', 'postprocessing')
            "#,
            vec![now.into(), candidate.stage_id.clone().into()],
        )?)
        .await
        .context("Failed to mark claimed runtime job running")?;

        let stage = get_stage_with(&tx, &candidate.stage_id)
            .await?
            .ok_or_else(|| anyhow!("Claimed runtime job stage was not found"))?;
        if stage.status != RuntimeStageStatus::Running
            || stage.worker_id.as_deref() != Some(worker_id)
            || stage.lease_expires_at != Some(u64::try_from(lease_expires_at)?)
            || stage.attempt_token.as_deref() != Some(attempt_token.as_str())
        {
            tx.rollback().await?;
            return Ok(None);
        }

        let job = get_job_with(&tx, stage.job_id.as_str())
            .await?
            .ok_or_else(|| anyhow!("Claimed runtime job was not found"))?;
        if !is_claimable_job_status(job.status) {
            tx.rollback().await?;
            return Ok(None);
        }

        tx.commit()
            .await
            .context("Failed to commit runtime stage claim transaction")?;
        Ok(Some(ClaimedStage { job, stage }))
    }

    pub async fn complete_stage(
        &self,
        lease: &StageLease,
        output_artifact_ids: Vec<String>,
    ) -> anyhow::Result<Option<JobStage>> {
        validate_stage_output_artifact_retention_bounds(&output_artifact_ids)?;
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(runtime_write_transaction_options())
            .await
            .context("Failed to start runtime stage completion transaction")?;
        let now = self.now_millis();
        let output_json = json_to_db_string(&json!(output_artifact_ids), "[]")?;
        let result = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE job_stages
                SET
                    status = 'completed',
                    cancellation_state = NULL,
                    updated_at = ?1,
                    finished_at = COALESCE(finished_at, ?1),
                    lease_expires_at = NULL,
                    worker_id = NULL,
                    output_artifact_ids_json = ?2,
                    error_code = NULL,
                    error_message = NULL
                WHERE id = ?3
                  AND status IN ('running', 'postprocessing')
                  AND cancellation_state IS NULL
                  AND worker_id = ?4
                  AND attempt_count = ?5
                  AND (attempt_token = ?6 OR (attempt_token IS NULL AND ?6 IS NULL))
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at > ?1
                  AND EXISTS (
                      SELECT 1
                      FROM runtime_jobs
                      WHERE runtime_jobs.id = job_stages.job_id
                        AND runtime_jobs.status IN ('created', 'queued', 'running', 'retrying', 'postprocessing')
                        AND runtime_jobs.cancellation_state IS NULL
                  )
                "#,
                vec![
                    now.into(),
                    output_json.into(),
                    lease.stage_id.clone().into(),
                    lease.worker_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    opt_string(lease.attempt_token.clone()),
                ],
            )?)
            .await
            .context("Failed to complete runtime job stage")?;
        if result.rows_affected() == 0 {
            tx.rollback().await?;
            return Ok(None);
        }

        let stage = get_stage_with(&tx, &lease.stage_id)
            .await?
            .ok_or_else(|| anyhow!("Completed runtime job stage was not found"))?;
        if let Err(error) = validate_stage_output_artifact_ids(&output_artifact_ids) {
            tx.rollback().await?;
            return Err(error);
        }
        if !output_artifact_ids.is_empty() {
            let Some(attempt_token) = lease.attempt_token.as_ref() else {
                tx.rollback().await?;
                bail!("Stage output artifacts require an exact attempt token");
            };
            let row_lock = match tx.get_database_backend() {
                DbBackend::Sqlite => "",
                DbBackend::Postgres | DbBackend::MySql => " FOR UPDATE",
                backend => bail!("Unsupported runtime artifact database backend: {backend:?}"),
            };
            for artifact_id in &output_artifact_ids {
                let ownership_sql = format!(
                    r#"
                    SELECT 1
                    FROM runtime_artifacts
                    WHERE id = ?1
                      AND job_id = ?2
                      AND stage_id = ?3
                      AND producer_attempt_count = ?4
                      AND producer_attempt_token = ?5
                      AND artifact_role IN ('output_primary', 'output_intermediate', 'debug')
                    LIMIT 1{row_lock}
                    "#
                );
                let owned = tx
                    .query_one_raw(raw::statement(
                        &tx,
                        ownership_sql,
                        vec![
                            artifact_id.clone().into(),
                            stage.job_id.clone().into(),
                            lease.stage_id.clone().into(),
                            u32_to_i64_value(lease.attempt_count).into(),
                            attempt_token.clone().into(),
                        ],
                    )?)
                    .await
                    .context("Failed to validate runtime stage output ownership")?;
                if owned.is_none() {
                    tx.rollback().await?;
                    bail!(
                        "Stage output artifact is not owned by the exact active job, stage, and attempt"
                    );
                }
            }
        }
        complete_job_if_all_stages_finished_with(&tx, stage.job_id.as_str(), now).await?;
        tx.commit()
            .await
            .context("Failed to commit runtime stage completion transaction")?;
        Ok(Some(stage))
    }

    pub async fn renew_stage_lease(
        &self,
        lease: &StageLease,
        lease_duration_ms: u64,
    ) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let lease_expires_at = now.saturating_add(i64::try_from(lease_duration_ms.max(1))?);
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE job_stages
                SET lease_expires_at = ?1, updated_at = ?2
                WHERE id = ?3
                  AND status IN ('running', 'postprocessing')
                  AND worker_id = ?4
                  AND attempt_count = ?5
                  AND (attempt_token = ?6 OR (attempt_token IS NULL AND ?6 IS NULL))
                  AND lease_expires_at IS NOT NULL
                  AND (
                      lease_expires_at > ?2
                      OR cancellation_state IN ('requested', 'execution_stopping')
                  )
                "#,
                vec![
                    lease_expires_at.into(),
                    now.into(),
                    lease.stage_id.clone().into(),
                    lease.worker_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    opt_string(lease.attempt_token.clone()),
                ],
            )?)
            .await
            .context("Failed to renew runtime job stage lease")?;
        Ok(result.rows_affected() == 1)
    }

    pub async fn stage_lease_state(
        &self,
        lease: &StageLease,
    ) -> anyhow::Result<Option<StageLeaseState>> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let row = db
            .query_one_raw(raw::statement(
                db,
                r#"
                SELECT s.cancellation_state
                FROM job_stages s
                JOIN runtime_jobs j ON j.id = s.job_id
                WHERE s.id = ?1
                  AND s.status IN ('running', 'postprocessing')
                  AND s.worker_id = ?2
                  AND s.attempt_count = ?3
                  AND (s.attempt_token = ?4 OR (s.attempt_token IS NULL AND ?4 IS NULL))
                  AND s.lease_expires_at IS NOT NULL
                  AND (
                      s.lease_expires_at > ?5
                      OR s.cancellation_state IN ('requested', 'execution_stopping')
                  )
                  AND j.status IN ('created', 'queued', 'running', 'retrying', 'postprocessing')
                LIMIT 1
                "#,
                vec![
                    lease.stage_id.clone().into(),
                    lease.worker_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    opt_string(lease.attempt_token.clone()),
                    now.into(),
                ],
            )?)
            .await
            .context("Failed to verify runtime stage lease ownership")?;
        let Some(row) = row else {
            return Ok(None);
        };
        let state: Option<String> = row.try_get_by_index(0)?;
        match parse_cancellation_state(state)? {
            None => Ok(Some(StageLeaseState::Active)),
            Some(RuntimeCancellationState::Requested) => {
                Ok(Some(StageLeaseState::CancellationRequested))
            }
            Some(RuntimeCancellationState::ExecutionStopping) => {
                Ok(Some(StageLeaseState::ExecutionStopping))
            }
        }
    }

    pub async fn stage_lease_is_active(&self, lease: &StageLease) -> anyhow::Result<bool> {
        Ok(self.stage_lease_state(lease).await?.is_some())
    }

    pub async fn update_stage_progress(
        &self,
        lease: &StageLease,
        progress: serde_json::Value,
    ) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let progress_json = json_to_db_string(&progress, "{}")?;
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE job_stages
                SET progress_json = ?1, updated_at = ?2
                WHERE id = ?3
                  AND status IN ('running', 'postprocessing')
                  AND cancellation_state IS NULL
                  AND worker_id = ?4
                  AND attempt_count = ?5
                  AND (attempt_token = ?6 OR (attempt_token IS NULL AND ?6 IS NULL))
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at > ?2
                  AND EXISTS (
                      SELECT 1 FROM runtime_jobs
                      WHERE runtime_jobs.id = job_stages.job_id
                        AND runtime_jobs.status IN ('created', 'queued', 'running', 'retrying', 'postprocessing')
                        AND runtime_jobs.cancellation_state IS NULL
                  )
                "#,
                vec![
                    progress_json.into(),
                    now.into(),
                    lease.stage_id.clone().into(),
                    lease.worker_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    opt_string(lease.attempt_token.clone()),
                ],
            )?)
            .await
            .context("Failed to update runtime stage progress")?;
        Ok(result.rows_affected() == 1)
    }

    /// Cooperative continuation consumes no retry attempt and rejoins the queue tail.
    /// Progress/artifacts remain durable. The next claim mints a fresh attempt token,
    /// so reusing the retry count cannot authorize any write from the old worker.
    pub async fn yield_stage(&self, lease: &StageLease) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
            UPDATE job_stages SET status = 'queued', worker_id = NULL,
                lease_expires_at = NULL, attempt_token = NULL,
                attempt_count = attempt_count - 1, available_at = ?1, updated_at = ?1
            WHERE id = ?2 AND worker_id = ?3 AND attempt_count = ?4 AND attempt_token = ?5
                AND attempt_count > 0 AND status IN ('running','postprocessing')
                AND cancellation_state IS NULL AND lease_expires_at > ?1
                AND job_id IN (
                    SELECT id FROM runtime_jobs
                    WHERE status IN ('running','queued','retrying','postprocessing')
                      AND cancellation_state IS NULL
                )
        "#,
                vec![
                    now.into(),
                    lease.stage_id.clone().into(),
                    lease.worker_id.clone().into(),
                    i64::from(lease.attempt_count).into(),
                    opt_string(lease.attempt_token.clone()),
                ],
            )?)
            .await?;
        Ok(result.rows_affected() == 1)
    }

    pub async fn relinquish_stage_lease(
        &self,
        lease: &StageLease,
        error_code: impl Into<String>,
        reason: impl Into<String>,
    ) -> anyhow::Result<Option<JobStage>> {
        self.fail_stage(lease, true, Some(error_code.into()), Some(reason.into()))
            .await
    }

    pub async fn fail_stage(
        &self,
        lease: &StageLease,
        retryable: bool,
        error_code: Option<String>,
        error_message: Option<String>,
    ) -> anyhow::Result<Option<JobStage>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(runtime_write_transaction_options())
            .await
            .context("Failed to start runtime stage failure transaction")?;
        let Some(stage) = get_stage_with(&tx, &lease.stage_id).await? else {
            tx.rollback().await?;
            return Ok(None);
        };
        if !matches!(
            stage.status,
            RuntimeStageStatus::Running | RuntimeStageStatus::Postprocessing
        ) || stage.cancellation_state.is_some()
        {
            tx.rollback().await?;
            return Ok(None);
        }
        let job = get_job_with(&tx, &stage.job_id)
            .await?
            .ok_or_else(|| anyhow!("Runtime stage parent job was not found"))?;
        if !is_claimable_job_status(job.status) || job.cancellation_state.is_some() {
            tx.rollback().await?;
            return Ok(None);
        }
        let policy = StoredRetryPolicy::from_job(&job);
        let should_retry =
            retryable && stage.attempt_count < policy.effective_max_attempts(&job, &stage);
        let now = self.now_millis();

        let result = if should_retry {
            let available_at = now.saturating_add(i64::try_from(policy.backoff_ms(&stage))?);
            self.retry_stage(
                &tx,
                &stage,
                lease,
                LeaseValidity::Any,
                now,
                available_at,
                error_code,
                error_message,
            )
            .await
        } else {
            self.mark_stage_failed(
                &tx,
                &stage,
                lease,
                LeaseValidity::Any,
                now,
                error_code,
                error_message,
            )
            .await
        }?;
        if result.is_none() {
            tx.rollback().await?;
            return Ok(None);
        }

        tx.commit()
            .await
            .context("Failed to commit runtime stage failure transaction")?;
        Ok(result)
    }

    pub async fn cancel_job(
        &self,
        job_id: &str,
        reason: Option<String>,
    ) -> anyhow::Result<Option<RuntimeJob>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(runtime_write_transaction_options())
            .await
            .context("Failed to start runtime job cancellation transaction")?;
        let now = self.now_millis();

        let result = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE runtime_jobs
                SET
                    status = CASE
                        WHEN EXISTS (
                            SELECT 1 FROM job_stages
                            WHERE job_stages.job_id = runtime_jobs.id
                              AND job_stages.status IN ('running', 'postprocessing')
                              AND job_stages.worker_id IS NOT NULL
                              AND job_stages.lease_expires_at IS NOT NULL
                        ) THEN status
                        ELSE 'cancelled'
                    END,
                    updated_at = ?1,
                    finished_at = CASE
                        WHEN EXISTS (
                            SELECT 1 FROM job_stages
                            WHERE job_stages.job_id = runtime_jobs.id
                              AND job_stages.status IN ('running', 'postprocessing')
                              AND job_stages.worker_id IS NOT NULL
                              AND job_stages.lease_expires_at IS NOT NULL
                        ) THEN NULL
                        ELSE COALESCE(finished_at, ?1)
                    END,
                    cancellation_reason = ?2,
                    cancellation_state = CASE
                        WHEN EXISTS (
                            SELECT 1 FROM job_stages
                            WHERE job_stages.job_id = runtime_jobs.id
                              AND job_stages.status IN ('running', 'postprocessing')
                              AND job_stages.worker_id IS NOT NULL
                              AND job_stages.lease_expires_at IS NOT NULL
                        ) THEN COALESCE(cancellation_state, 'requested')
                        ELSE NULL
                    END
                WHERE id = ?3
                  AND status IN ('created', 'queued', 'running', 'paused', 'retrying', 'postprocessing')
                "#,
                vec![now.into(), opt_string(reason), job_id.into()],
            )?)
            .await
            .context("Failed to cancel runtime job")?;
        if result.rows_affected() == 0 {
            tx.rollback().await?;
            return Ok(None);
        }

        tx.execute_raw(raw::statement(
            &tx,
            r#"
            UPDATE job_stages
            SET
                status = 'cancelled',
                cancellation_state = NULL,
                updated_at = ?1,
                finished_at = COALESCE(finished_at, ?1),
                lease_expires_at = NULL,
                worker_id = NULL,
                available_at = NULL
            WHERE job_id = ?2
              AND status IN ('created', 'queued', 'running', 'paused', 'retrying', 'postprocessing')
              AND (
                  status NOT IN ('running', 'postprocessing')
                  OR worker_id IS NULL
                  OR lease_expires_at IS NULL
              )
            "#,
            vec![now.into(), job_id.into()],
        )?)
        .await
        .context("Failed to cancel non-running runtime job stages")?;

        tx.execute_raw(raw::statement(
            &tx,
            r#"
            UPDATE job_stages
            SET
                cancellation_state = CASE
                    WHEN cancellation_state = 'execution_stopping'
                        THEN cancellation_state
                    ELSE 'requested'
                END,
                updated_at = ?1,
                available_at = NULL
            WHERE job_id = ?2
              AND status IN ('running', 'postprocessing')
              AND worker_id IS NOT NULL
              AND lease_expires_at IS NOT NULL
            "#,
            vec![now.into(), job_id.into()],
        )?)
        .await
        .context("Failed to request cancellation of running runtime job stages")?;

        self.finalize_cancelled_route_projection_with(&tx, job_id)
            .await?;

        tx.commit()
            .await
            .context("Failed to commit runtime job cancellation transaction")?;
        self.get_job(job_id).await
    }

    pub async fn mark_stage_execution_stopping(&self, lease: &StageLease) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(runtime_write_transaction_options())
            .await
            .context("Failed to start execution-stopping transaction")?;
        let now = self.now_millis();
        let result = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE job_stages
                SET cancellation_state = 'execution_stopping', updated_at = ?1
                WHERE id = ?2
                  AND status IN ('running', 'postprocessing')
                  AND cancellation_state = 'requested'
                  AND worker_id = ?3
                  AND attempt_count = ?4
                  AND (attempt_token = ?5 OR (attempt_token IS NULL AND ?5 IS NULL))
                  AND lease_expires_at IS NOT NULL
                "#,
                vec![
                    now.into(),
                    lease.stage_id.clone().into(),
                    lease.worker_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    opt_string(lease.attempt_token.clone()),
                ],
            )?)
            .await
            .context("Failed to mark runtime stage execution stopping")?;
        if result.rows_affected() == 0 {
            tx.rollback().await?;
            return Ok(false);
        }
        tx.execute_raw(raw::statement(
            &tx,
            r#"
            UPDATE runtime_jobs
            SET cancellation_state = 'execution_stopping', updated_at = ?1
            WHERE id = (SELECT job_id FROM job_stages WHERE id = ?2)
              AND cancellation_state IS NOT NULL
              AND status IN ('created', 'queued', 'running', 'retrying', 'postprocessing')
            "#,
            vec![now.into(), lease.stage_id.clone().into()],
        )?)
        .await
        .context("Failed to mark runtime job execution stopping")?;
        tx.commit()
            .await
            .context("Failed to commit execution-stopping transaction")?;
        Ok(true)
    }

    /// Finalize only after the caller knows this exact attempt's executor has
    /// resolved. Attempt identity, rather than wall-clock lease freshness,
    /// fences late settlement from stale workers.
    pub async fn finalize_stage_cancellation(
        &self,
        lease: &StageLease,
    ) -> anyhow::Result<Option<JobStage>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(runtime_write_transaction_options())
            .await
            .context("Failed to start stage cancellation finalization transaction")?;
        let now = self.now_millis();
        let finalized = self
            .finalize_stage_cancellation_with(&tx, lease, now)
            .await?;
        if finalized.is_none() {
            tx.rollback().await?;
            return Ok(None);
        }
        tx.commit()
            .await
            .context("Failed to commit stage cancellation finalization transaction")?;
        Ok(finalized)
    }

    pub async fn recover_expired_stage_leases(&self, limit: usize) -> anyhow::Result<u64> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let limit = bounded_maintenance_batch_limit(limit);
        // A wall-clock expiry does not prove that requested cancellation tore
        // down execution. Only ordinary leases are eligible for retry here;
        // cancelling attempts stay owner-fenced until exact-attempt settlement
        // or a future supervisor-confirmed process teardown transition.
        let rows = db
            .query_all_raw(raw::statement(
                db,
                r#"
                SELECT id, worker_id, attempt_count, attempt_token
                FROM job_stages
                WHERE status IN ('running', 'postprocessing')
                  AND cancellation_state IS NULL
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at <= ?1
                ORDER BY lease_expires_at ASC, id ASC
                LIMIT ?2
                "#,
                vec![now.into(), i64::try_from(limit)?.into()],
            )?)
            .await
            .context("Failed to list expired runtime stage leases")?;

        let mut recovered = 0_u64;
        for row in rows {
            let stage_id: String = row.try_get_by_index(0)?;
            let worker_id: Option<String> = row.try_get_by_index(1)?;
            let Some(worker_id) = worker_id else {
                continue;
            };
            let attempt_count = i64_to_u32(row.try_get_by_index(2)?)?;
            let lease = StageLease {
                stage_id,
                worker_id,
                attempt_count,
                attempt_token: row.try_get_by_index(3)?,
            };
            let tx = db
                .begin_with_options(runtime_write_transaction_options())
                .await
                .context("Failed to start expired lease recovery transaction")?;
            let Some(stage) = get_stage_with(&tx, &lease.stage_id).await? else {
                tx.rollback().await?;
                continue;
            };
            let Some(job) = get_job_with(&tx, &stage.job_id).await? else {
                tx.rollback().await?;
                continue;
            };
            let policy = StoredRetryPolicy::from_job(&job);
            let result = if stage.attempt_count < policy.effective_max_attempts(&job, &stage) {
                let available_at = now.saturating_add(i64::try_from(policy.backoff_ms(&stage))?);
                self.retry_stage(
                    &tx,
                    &stage,
                    &lease,
                    LeaseValidity::Expired,
                    now,
                    available_at,
                    Some("lease_expired".to_string()),
                    Some("Worker lease expired before completion".to_string()),
                )
                .await?
            } else {
                self.mark_stage_failed(
                    &tx,
                    &stage,
                    &lease,
                    LeaseValidity::Expired,
                    now,
                    Some("lease_expired".to_string()),
                    Some("Worker lease expired before completion".to_string()),
                )
                .await?
            };
            if result.is_some() {
                tx.commit()
                    .await
                    .context("Failed to commit expired lease recovery transaction")?;
                recovered = recovered.saturating_add(1);
            } else {
                tx.rollback().await?;
            }
        }

        Ok(recovered)
    }

    pub async fn queued_stage_count(&self) -> anyhow::Result<u64> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                "SELECT COUNT(*) FROM job_stages WHERE status IN ('queued', 'retrying')",
                vec![],
            )?)
            .await
            .context("Failed to count queued runtime stages")?;
        let count = row
            .ok_or_else(|| anyhow!("Queued runtime stage count returned no row"))?
            .try_get_by_index::<i64>(0)?;
        i64_to_u64(count)
    }

    pub async fn runtime_queue_health(
        &self,
        heartbeat_stale_after_ms: u64,
    ) -> anyhow::Result<RuntimeQueueHealthSnapshot> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let queue_rows = db
            .query_all_raw(raw::statement(
                db,
                r#"
                SELECT queue_class, COUNT(*), MIN(created_at)
                FROM job_stages
                WHERE status IN ('queued', 'retrying')
                GROUP BY queue_class
                ORDER BY queue_class
                "#,
                vec![],
            )?)
            .await
            .context("Failed to load runtime queue depth and age")?;
        let queues = queue_rows
            .iter()
            .map(|row| {
                let queue_raw: String = row.try_get_by_index(0)?;
                let queue_class = QueueClass::from_db_value(&queue_raw)
                    .ok_or_else(|| anyhow!("Unknown runtime queue class: {queue_raw}"))?;
                let count = i64_to_u64(row.try_get_by_index(1)?)?;
                let oldest_created_at = row.try_get_by_index::<Option<i64>>(2)?.unwrap_or(now);
                Ok(RuntimeQueueDepth {
                    queue_class,
                    count,
                    oldest_age_ms: i64_to_u64(now.saturating_sub(oldest_created_at).max(0))?,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let heartbeats = self.list_worker_heartbeats().await?;
        let stale_cutoff = now.saturating_sub(i64::try_from(heartbeat_stale_after_ms)?);
        let active_workers = heartbeats
            .iter()
            .filter(|heartbeat| worker_heartbeat_accepts_claims(heartbeat))
            .count() as u64;
        let healthy = heartbeats
            .iter()
            .filter(|heartbeat| {
                worker_heartbeat_accepts_claims(heartbeat)
                    && i64::try_from(heartbeat.last_heartbeat_at)
                        .is_ok_and(|last| last >= stale_cutoff)
            })
            .collect::<Vec<_>>();
        let stale_workers = active_workers.saturating_sub(healthy.len() as u64);
        let uncovered_queue_classes = queues
            .iter()
            .filter(|queue| {
                !healthy.iter().any(|heartbeat| {
                    heartbeat
                        .registration
                        .queue_classes
                        .iter()
                        .any(|worker_queue| {
                            *worker_queue == QueueClass::Batch || *worker_queue == queue.queue_class
                        })
                })
            })
            .map(|queue| queue.queue_class)
            .collect();

        Ok(RuntimeQueueHealthSnapshot {
            heartbeat_stale_after_ms,
            active_workers,
            healthy_workers: healthy.len() as u64,
            stale_workers,
            queues,
            uncovered_queue_classes,
        })
    }

    pub async fn stage_status_counts(&self) -> anyhow::Result<Vec<RuntimeStageStatusCount>> {
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(
                db,
                "SELECT status, COUNT(*) FROM job_stages GROUP BY status ORDER BY status",
                vec![],
            )?)
            .await
            .context("Failed to count runtime stages by status")?;

        rows.iter()
            .map(|row| {
                let status_raw: String = row.try_get_by_index(0)?;
                let status = RuntimeStageStatus::from_db_value(status_raw.as_str())
                    .ok_or_else(|| anyhow!("Unknown runtime stage status: {status_raw}"))?;
                let count = i64_to_u64(row.try_get_by_index(1)?)?;
                Ok(RuntimeStageStatusCount { status, count })
            })
            .collect()
    }

    pub async fn create_stage(&self, input: NewJobStage) -> anyhow::Result<JobStage> {
        let queue_class = queue_class_for_stage_kind(&input.stage_kind);
        self.create_stage_with_dispatch(NewJobStageDispatch {
            stage: input,
            queue_class,
            resource_hints: StageResourceHints::default(),
        })
        .await
    }

    pub async fn create_stage_with_dispatch(
        &self,
        input: NewJobStageDispatch,
    ) -> anyhow::Result<JobStage> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let id = new_uuid();
        let stage = input.stage;
        let resource_hints = input.resource_hints.normalized();
        let resource_hints_json = json_to_db_string(&json!(resource_hints), "{}")?;
        let input_artifact_ids_json = json_to_db_string(&json!(stage.input_artifact_ids), "[]")?;

        db.execute_raw(raw::statement(
            db,
            r#"
            INSERT INTO job_stages (
                id,
                job_id,
                created_at,
                updated_at,
                sequence,
                stage_kind,
                queue_class,
                resource_hints_json,
                resource_target,
                required_backend,
                required_device_class,
                min_resource_memory_bytes,
                resource_concurrency_weight,
                status,
                capability,
                model_id,
                worker_id,
                lease_expires_at,
                available_at,
                attempt_token,
                attempt_count,
                max_attempts,
                input_artifact_ids_json,
                output_artifact_ids_json,
                progress_json,
                started_at,
                finished_at,
                error_code,
                error_message
            )
            VALUES (?1, ?2, ?3, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, NULL, NULL, ?3, NULL, 0, ?16, ?17, '[]', NULL, NULL, NULL, NULL, NULL)
            "#,
            vec![
                id.clone().into(),
                stage.job_id.into(),
                now.into(),
                u32_to_i64_value(stage.sequence).into(),
                stage.stage_kind.into(),
                input.queue_class.as_db_value().into(),
                resource_hints_json.into(),
                resource_hints.target.as_db_value().into(),
                opt_string(
                    resource_hints
                        .backend
                        .map(|backend| backend.as_db_value().to_string()),
                ),
                opt_string(
                    resource_hints
                        .device_class
                        .map(|device| device.as_db_value().to_string()),
                ),
                opt_u64(resource_hints.min_memory_bytes),
                u32_to_i64_value(resource_hints.concurrency_weight).into(),
                stage.status.as_db_value().into(),
                opt_string(stage.capability),
                opt_string(stage.model_id),
                u32_to_i64_value(stage.max_attempts).into(),
                input_artifact_ids_json.into(),
            ],
        )?)
        .await
        .context("Failed to create runtime job stage")?;

        self.get_stage(&id)
            .await?
            .ok_or_else(|| anyhow!("Created runtime job stage was not found"))
    }

    pub async fn get_stage(&self, id: &str) -> anyhow::Result<Option<JobStage>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(db, JOB_STAGE_COLUMNS_SQL, vec![id.into()])?)
            .await
            .context("Failed to load runtime job stage")?;

        row.as_ref().map(map_job_stage).transpose()
    }

    pub async fn list_stages_for_job(&self, job_id: &str) -> anyhow::Result<Vec<JobStage>> {
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(
                db,
                JOB_STAGE_LIST_FOR_JOB_SQL,
                vec![job_id.into()],
            )?)
            .await
            .context("Failed to list runtime job stages")?;

        rows.iter().map(map_job_stage).collect()
    }

    pub async fn create_artifact(
        &self,
        input: NewRuntimeArtifact,
    ) -> anyhow::Result<RuntimeArtifact> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let id = new_uuid();
        let metadata_json = json_to_db_string(&input.metadata_json, "{}")?;

        db.execute_raw(raw::statement(
            db,
            r#"
            INSERT INTO runtime_artifacts (
                id,
                job_id,
                stage_id,
                created_at,
                artifact_kind,
                artifact_role,
                media_asset_id,
                text_asset_id,
                storage_key,
                content_type,
                filename,
                size_bytes,
                sha256,
                metadata_json,
                retention_policy
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)
            "#,
            vec![
                id.clone().into(),
                input.job_id.into(),
                opt_string(input.stage_id),
                now.into(),
                input.artifact_kind.as_db_value().into(),
                input.artifact_role.as_db_value().into(),
                opt_string(input.media_asset_id),
                opt_string(input.text_asset_id),
                opt_string(input.storage_key),
                opt_string(input.content_type),
                opt_string(input.filename),
                opt_u64(input.size_bytes),
                opt_string(input.sha256),
                metadata_json.into(),
                input.retention_policy.into(),
            ],
        )?)
        .await
        .context("Failed to create runtime artifact")?;

        self.get_artifact(&id)
            .await?
            .ok_or_else(|| anyhow!("Created runtime artifact was not found"))
    }

    pub async fn publish_stage_output_artifact(
        &self,
        lease: &StageLease,
        input: NewStageOutputArtifact,
    ) -> anyhow::Result<Option<RuntimeArtifact>> {
        if !matches!(
            input.artifact_role,
            RuntimeArtifactRole::OutputPrimary
                | RuntimeArtifactRole::OutputIntermediate
                | RuntimeArtifactRole::Debug
        ) {
            bail!("Attempt-owned artifact publication requires an output or debug role");
        }
        let publication_key = input.publication_key.trim().to_string();
        if publication_key.is_empty() {
            bail!("Attempt-owned artifact publication requires a publication key");
        }
        let Some(attempt_token) = lease.attempt_token.as_ref() else {
            return Ok(None);
        };

        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(runtime_write_transaction_options())
            .await
            .context("Failed to start runtime artifact publication transaction")?;
        let now = self.now_millis();
        let id = new_uuid();
        let metadata_json = json_to_db_string(&input.metadata_json, "{}")?;
        let conflict_clause = match tx.get_database_backend() {
            DbBackend::Sqlite | DbBackend::Postgres => {
                "ON CONFLICT(stage_id, producer_attempt_token, publication_key) DO NOTHING"
            }
            DbBackend::MySql => "ON DUPLICATE KEY UPDATE id = id",
            backend => bail!("Unsupported runtime artifact database backend: {backend:?}"),
        };
        let insert_sql = format!(
            r#"
            INSERT INTO runtime_artifacts (
                id,
                job_id,
                stage_id,
                producer_attempt_count,
                producer_attempt_token,
                publication_key,
                created_at,
                artifact_kind,
                artifact_role,
                media_asset_id,
                text_asset_id,
                storage_key,
                content_type,
                filename,
                size_bytes,
                sha256,
                metadata_json,
                retention_policy
            )
            SELECT
                ?1,
                s.job_id,
                s.id,
                ?2,
                ?3,
                ?4,
                ?5,
                ?6,
                ?7,
                ?8,
                ?9,
                ?10,
                ?11,
                ?12,
                ?13,
                ?14,
                ?15,
                ?16
            FROM job_stages s
            JOIN runtime_jobs j ON j.id = s.job_id
            WHERE s.id = ?17
              AND s.status IN ('running', 'postprocessing')
              AND s.cancellation_state IS NULL
              AND s.worker_id = ?18
              AND s.attempt_count = ?2
              AND s.attempt_token = ?3
              AND s.lease_expires_at IS NOT NULL
              AND s.lease_expires_at > ?5
              AND j.status IN ('created', 'queued', 'running', 'retrying', 'postprocessing')
              AND j.cancellation_state IS NULL
            {conflict_clause}
            "#
        );
        tx.execute_raw(raw::statement(
            &tx,
            insert_sql,
            vec![
                id.into(),
                u32_to_i64_value(lease.attempt_count).into(),
                attempt_token.clone().into(),
                publication_key.clone().into(),
                now.into(),
                input.artifact_kind.as_db_value().into(),
                input.artifact_role.as_db_value().into(),
                opt_string(input.media_asset_id),
                opt_string(input.text_asset_id),
                opt_string(input.storage_key),
                opt_string(input.content_type),
                opt_string(input.filename),
                opt_u64(input.size_bytes),
                opt_string(input.sha256),
                metadata_json.into(),
                input.retention_policy.into(),
                lease.stage_id.clone().into(),
                lease.worker_id.clone().into(),
            ],
        )?)
        .await
        .context("Failed to publish attempt-owned runtime artifact")?;

        let row = tx
            .query_one_raw(raw::statement(
                &tx,
                r#"
                SELECT
                    a.id,
                    a.job_id,
                    a.stage_id,
                    a.producer_attempt_count,
                    a.producer_attempt_token,
                    a.publication_key,
                    a.created_at,
                    a.artifact_kind,
                    a.artifact_role,
                    a.media_asset_id,
                    a.text_asset_id,
                    a.storage_key,
                    a.content_type,
                    a.filename,
                    a.size_bytes,
                    a.sha256,
                    a.metadata_json,
                    a.retention_policy
                FROM runtime_artifacts a
                JOIN job_stages s ON s.id = a.stage_id
                JOIN runtime_jobs j ON j.id = s.job_id
                WHERE a.stage_id = ?1
                  AND a.producer_attempt_count = ?2
                  AND a.producer_attempt_token = ?3
                  AND a.publication_key = ?4
                  AND s.status IN ('running', 'postprocessing')
                  AND s.cancellation_state IS NULL
                  AND s.worker_id = ?5
                  AND s.attempt_count = ?2
                  AND s.attempt_token = ?3
                  AND s.lease_expires_at IS NOT NULL
                  AND s.lease_expires_at > ?6
                  AND j.status IN ('created', 'queued', 'running', 'retrying', 'postprocessing')
                  AND j.cancellation_state IS NULL
                LIMIT 1
                "#,
                vec![
                    lease.stage_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    attempt_token.clone().into(),
                    publication_key.into(),
                    lease.worker_id.clone().into(),
                    now.into(),
                ],
            )?)
            .await
            .context("Failed to load attempt-owned runtime artifact")?;
        let artifact = row.as_ref().map(map_runtime_artifact).transpose()?;
        tx.commit()
            .await
            .context("Failed to commit runtime artifact publication transaction")?;
        Ok(artifact)
    }

    pub async fn get_artifact(&self, id: &str) -> anyhow::Result<Option<RuntimeArtifact>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                RUNTIME_ARTIFACT_COLUMNS_SQL,
                vec![id.into()],
            )?)
            .await
            .context("Failed to load runtime artifact")?;

        row.as_ref().map(map_runtime_artifact).transpose()
    }

    pub async fn list_artifacts_for_job(
        &self,
        job_id: &str,
    ) -> anyhow::Result<Vec<RuntimeArtifact>> {
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(
                db,
                RUNTIME_ARTIFACT_LIST_FOR_JOB_SQL,
                vec![job_id.into()],
            )?)
            .await
            .context("Failed to list runtime job artifacts")?;

        rows.iter().map(map_runtime_artifact).collect()
    }

    /// Claim expired replay for deletion before exposing storage keys to GC.
    /// A terminal job loses retry eligibility atomically with this claim. Retrying
    /// and GC both conditionally update the job row, fencing their race on every DB.
    pub async fn expired_speech_pcm(&self, before: u64) -> anyhow::Result<Vec<RuntimeArtifact>> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(runtime_write_transaction_options())
            .await?;
        let cutoff = i64::try_from(before)?;
        let sql = RUNTIME_ARTIFACT_LIST_FOR_JOB_SQL.replace(
            "WHERE job_id = ?1 ORDER BY created_at ASC, id ASC",
            "WHERE publication_key LIKE 'speech-pcm/%' AND job_id IN (SELECT id FROM runtime_jobs WHERE status IN ('completed','failed','cancelled','expired') AND updated_at < ?1) ORDER BY created_at ASC, id ASC LIMIT 64",
        );
        let rows = tx
            .query_all_raw(raw::statement(&tx, sql, vec![cutoff.into()])?)
            .await?;
        let artifacts = rows
            .iter()
            .map(map_runtime_artifact)
            .collect::<anyhow::Result<Vec<_>>>()?;
        let jobs = artifacts
            .iter()
            .map(|artifact| artifact.job_id.clone())
            .collect::<std::collections::BTreeSet<_>>();
        let mut claimed = std::collections::HashSet::new();
        for job in jobs {
            let result = tx.execute_raw(raw::statement(&tx,
                "UPDATE runtime_jobs SET error_code = 'speech_replay_expired' WHERE id = ?1 AND status IN ('completed','failed','cancelled','expired') AND updated_at < ?2",
                vec![job.clone().into(), cutoff.into()],
            )?).await?;
            if result.rows_affected() == 1 {
                claimed.insert(job);
            }
        }
        tx.commit().await?;
        Ok(artifacts
            .into_iter()
            .filter(|artifact| claimed.contains(&artifact.job_id))
            .collect())
    }

    /// Permanently fence manual replay deletion against concurrent job retry.
    /// The retry transition tests this marker in its conditional UPDATE too.
    pub async fn fence_speech_replay_deletion(&self, job_id: &str) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let result = db.execute_raw(raw::statement(db,
            "UPDATE runtime_jobs SET error_code = 'speech_replay_expired' WHERE id = ?1 AND status IN ('completed','failed','cancelled','expired')",
            vec![job_id.into()],
        )?).await?;
        Ok(result.rows_affected() == 1)
    }

    pub async fn remove_speech_pcm_artifact(&self, id: &str) -> anyhow::Result<()> {
        let db = self.db.connection().await?;
        db.execute_raw(raw::statement(
            db,
            "DELETE FROM runtime_artifacts WHERE id = ?1 AND publication_key LIKE 'speech-pcm/%'",
            vec![id.into()],
        )?)
        .await?;
        Ok(())
    }

    pub async fn stage_output_for_key(
        &self,
        lease: &StageLease,
        key: &str,
    ) -> anyhow::Result<Option<RuntimeArtifact>> {
        let db = self.db.connection().await?;
        let sql = RUNTIME_ARTIFACT_COLUMNS_SQL.replace(
            "WHERE id = ?1",
            "WHERE stage_id = ?1 AND producer_attempt_token = ?2 AND publication_key = ?3",
        );
        let row = db
            .query_one_raw(raw::statement(
                db,
                sql,
                vec![
                    lease.stage_id.clone().into(),
                    opt_string(lease.attempt_token.clone()),
                    key.into(),
                ],
            )?)
            .await?;
        row.as_ref().map(map_runtime_artifact).transpose()
    }

    /// Read a bounded page of committed speech PCM without loading the whole journal.
    pub async fn speech_pcm_after(
        &self,
        job_id: &str,
        after_sequence: Option<u64>,
        limit: u32,
    ) -> anyhow::Result<Vec<RuntimeArtifact>> {
        let db = self.db.connection().await?;
        let after_key = after_sequence
            .map(super::speech_progress::pcm_publication_key)
            .unwrap_or_else(|| "speech-pcm/".to_string());
        let sql = RUNTIME_ARTIFACT_LIST_FOR_JOB_SQL.replace(
            "ORDER BY created_at ASC, id ASC",
            "AND publication_key > ?2 AND publication_key < 'speech-pcm0' ORDER BY publication_key ASC LIMIT ?3",
        );
        let rows = db
            .query_all_raw(raw::statement(
                db,
                sql,
                vec![
                    job_id.into(),
                    after_key.into(),
                    i64::from(limit.clamp(1, 64)).into(),
                ],
            )?)
            .await
            .context("Failed to read speech PCM replay journal")?;
        rows.iter().map(map_runtime_artifact).collect()
    }

    /// Reserve one tenant-scoped durable create operation.
    ///
    /// Callers must not acknowledge work until `commit_durable_idempotency`
    /// succeeds. An expired reservation is eligible for a new owner; an active
    /// reservation never permits a second caller to create work concurrently.
    pub async fn reserve_durable_idempotency(
        &self,
        request: DurableIdempotencyRequest,
    ) -> anyhow::Result<DurableIdempotencyBegin> {
        self.reserve_durable_idempotency_with_capacity(request, MAX_DURABLE_IDEMPOTENCY_RECORDS)
            .await
    }

    async fn reserve_durable_idempotency_with_capacity(
        &self,
        request: DurableIdempotencyRequest,
        max_records: u64,
    ) -> anyhow::Result<DurableIdempotencyBegin> {
        validate_durable_idempotency_request(&request)?;
        anyhow::ensure!(
            max_records > 0,
            "Durable idempotency capacity must be positive"
        );
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(runtime_write_transaction_options())
            .await
            .context("Failed to start durable idempotency reservation transaction")?;
        lock_durable_idempotency(&tx).await?;
        let now = nonnegative_timestamp(self.now_millis())?;
        prune_expired_durable_idempotency_with(&tx, now, DEFAULT_DURABLE_IDEMPOTENCY_PRUNE_LIMIT)
            .await?;

        if let Some(existing) = load_durable_idempotency_with(
            &tx,
            &request.tenant_scope,
            &request.operation,
            &request.idempotency_key,
        )
        .await?
        {
            if existing.expires_at <= now {
                delete_durable_idempotency_with(
                    &tx,
                    &request.tenant_scope,
                    &request.operation,
                    &request.idempotency_key,
                    now,
                )
                .await?;
            } else {
                let outcome = existing.begin_outcome(&request)?;
                tx.commit().await?;
                return Ok(outcome);
            }
        }

        let count = tx
            .query_one_raw(raw::statement(
                &tx,
                "SELECT COUNT(*) FROM durable_idempotency_keys_v2",
                vec![],
            )?)
            .await?
            .ok_or_else(|| anyhow!("Durable idempotency count returned no row"))?
            .try_get_by_index::<i64>(0)?;
        if u64::try_from(count)? >= max_records {
            tx.commit().await?;
            return Ok(DurableIdempotencyBegin::CapacityExceeded);
        }

        let reservation_token = new_uuid();
        let expires_at = now
            .checked_add(request.reservation_ttl_ms)
            .context("Durable idempotency reservation expiry overflow")?;
        let insert_sql = match tx.get_database_backend() {
            DbBackend::Sqlite | DbBackend::Postgres => {
                r#"
                INSERT INTO durable_idempotency_keys_v2 (
                    tenant_scope, operation, idempotency_key, created_at,
                    updated_at, expires_at, digest_version, request_digest,
                    state, reservation_token, runtime_job_id, response_json
                )
                VALUES (?1, ?2, ?3, ?4, ?4, ?5, ?6, ?7, 'reserved', ?8, NULL, NULL)
                ON CONFLICT(tenant_scope, operation, idempotency_key) DO NOTHING
            "#
            }
            DbBackend::MySql => {
                r#"
                INSERT IGNORE INTO durable_idempotency_keys_v2 (
                    tenant_scope, operation, idempotency_key, created_at,
                    updated_at, expires_at, digest_version, request_digest,
                    state, reservation_token, runtime_job_id, response_json
                )
                VALUES (?1, ?2, ?3, ?4, ?4, ?5, ?6, ?7, 'reserved', ?8, NULL, NULL)
            "#
            }
            backend => bail!("Unsupported durable idempotency database backend: {backend:?}"),
        };
        let inserted = tx
            .execute_raw(raw::statement(
                &tx,
                insert_sql,
                vec![
                    request.tenant_scope.clone().into(),
                    request.operation.clone().into(),
                    request.idempotency_key.clone().into(),
                    i64::try_from(now)?.into(),
                    i64::try_from(expires_at)?.into(),
                    i64::from(request.digest_version).into(),
                    request.request_digest.clone().into(),
                    reservation_token.clone().into(),
                ],
            )?)
            .await
            .context("Failed to reserve durable idempotency key")?;

        if inserted.rows_affected() == 0 {
            let existing = load_durable_idempotency_with(
                &tx,
                &request.tenant_scope,
                &request.operation,
                &request.idempotency_key,
            )
            .await?
            .ok_or_else(|| anyhow!("Conflicting durable idempotency key disappeared"))?;
            let outcome = existing.begin_outcome(&request)?;
            tx.commit().await?;
            return Ok(outcome);
        }

        tx.commit()
            .await
            .context("Failed to commit durable idempotency reservation")?;
        Ok(DurableIdempotencyBegin::Acquired(
            DurableIdempotencyReservation {
                tenant_scope: request.tenant_scope,
                operation: request.operation,
                idempotency_key: request.idempotency_key,
                digest_version: request.digest_version,
                request_digest: request.request_digest,
                reservation_token,
                expires_at,
            },
        ))
    }

    /// Commit the replayable response for an acknowledged durable job.
    ///
    /// The conditional update verifies reservation ownership, request digest,
    /// unexpired state, and existence of the referenced job in one transaction.
    pub async fn commit_durable_idempotency(
        &self,
        reservation: &DurableIdempotencyReservation,
        runtime_job_id: &str,
        response_json: serde_json::Value,
        retention_ms: u64,
    ) -> anyhow::Result<Option<DurableIdempotencyReplay>> {
        validate_durable_idempotency_identity(
            &reservation.tenant_scope,
            &reservation.operation,
            &reservation.idempotency_key,
            reservation.digest_version,
            &reservation.request_digest,
        )?;
        validate_bounded_field("reservation token", &reservation.reservation_token, 64)?;
        validate_bounded_field("runtime job ID", runtime_job_id, 128)?;
        anyhow::ensure!(
            (1..=MAX_DURABLE_IDEMPOTENCY_RETENTION_MS).contains(&retention_ms),
            "Durable idempotency retention must be between 1 and {MAX_DURABLE_IDEMPOTENCY_RETENTION_MS} milliseconds"
        );
        let response_json_string = bounded_json_string(
            &response_json,
            MAX_DURABLE_IDEMPOTENCY_RESULT_BYTES,
            "Durable idempotency result",
        )?;

        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(runtime_write_transaction_options())
            .await
            .context("Failed to start durable idempotency commit transaction")?;
        let now = nonnegative_timestamp(self.now_millis())?;
        let expires_at = now
            .checked_add(retention_ms)
            .context("Durable idempotency retention expiry overflow")?;
        let result = tx
            .execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE durable_idempotency_keys_v2
                SET state = 'committed', updated_at = ?1, expires_at = ?2,
                    runtime_job_id = ?3, response_json = ?4
                WHERE tenant_scope = ?5
                  AND operation = ?6
                  AND idempotency_key = ?7
                  AND state = 'reserved'
                  AND reservation_token = ?8
                  AND digest_version = ?9
                  AND request_digest = ?10
                  AND expires_at > ?1
                  AND EXISTS (SELECT 1 FROM runtime_jobs WHERE id = ?3)
                "#,
                vec![
                    i64::try_from(now)?.into(),
                    i64::try_from(expires_at)?.into(),
                    runtime_job_id.into(),
                    response_json_string.into(),
                    reservation.tenant_scope.clone().into(),
                    reservation.operation.clone().into(),
                    reservation.idempotency_key.clone().into(),
                    reservation.reservation_token.clone().into(),
                    i64::from(reservation.digest_version).into(),
                    reservation.request_digest.clone().into(),
                ],
            )?)
            .await
            .context("Failed to commit durable idempotency result")?;
        if result.rows_affected() == 0 {
            let existing = load_durable_idempotency_with(
                &tx,
                &reservation.tenant_scope,
                &reservation.operation,
                &reservation.idempotency_key,
            )
            .await?;
            let replay = existing.and_then(|existing| {
                (existing.state == "committed"
                    && existing.expires_at > now
                    && existing.digest_version == reservation.digest_version
                    && existing.request_digest == reservation.request_digest
                    && existing.reservation_token == reservation.reservation_token
                    && existing.runtime_job_id.as_deref() == Some(runtime_job_id)
                    && existing.response_json.as_ref() == Some(&response_json))
                .then(|| DurableIdempotencyReplay {
                    runtime_job_id: runtime_job_id.to_string(),
                    response_json: response_json.clone(),
                    expires_at: existing.expires_at,
                })
            });
            if replay.is_some() {
                tx.commit().await?;
            } else {
                tx.rollback().await?;
            }
            return Ok(replay);
        }
        tx.commit()
            .await
            .context("Failed to commit durable idempotency result transaction")?;
        Ok(Some(DurableIdempotencyReplay {
            runtime_job_id: runtime_job_id.to_string(),
            response_json,
            expires_at,
        }))
    }

    /// Release an uncommitted reservation after a request is rejected locally.
    /// The opaque token prevents an old owner from deleting a replacement.
    pub async fn release_durable_idempotency(
        &self,
        reservation: &DurableIdempotencyReservation,
    ) -> anyhow::Result<bool> {
        validate_durable_idempotency_identity(
            &reservation.tenant_scope,
            &reservation.operation,
            &reservation.idempotency_key,
            reservation.digest_version,
            &reservation.request_digest,
        )?;
        validate_bounded_field("reservation token", &reservation.reservation_token, 64)?;
        let db = self.db.connection().await?;
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                DELETE FROM durable_idempotency_keys_v2
                WHERE tenant_scope = ?1 AND operation = ?2 AND idempotency_key = ?3
                  AND state = 'reserved' AND reservation_token = ?4
                  AND digest_version = ?5 AND request_digest = ?6
                "#,
                vec![
                    reservation.tenant_scope.clone().into(),
                    reservation.operation.clone().into(),
                    reservation.idempotency_key.clone().into(),
                    reservation.reservation_token.clone().into(),
                    i64::from(reservation.digest_version).into(),
                    reservation.request_digest.clone().into(),
                ],
            )?)
            .await
            .context("Failed to release durable idempotency reservation")?;
        Ok(result.rows_affected() == 1)
    }

    /// Delete a deterministic bounded page of expired reservations/results.
    pub async fn prune_expired_durable_idempotency(&self, limit: usize) -> anyhow::Result<u64> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(runtime_write_transaction_options())
            .await
            .context("Failed to start durable idempotency prune transaction")?;
        lock_durable_idempotency(&tx).await?;
        let removed = prune_expired_durable_idempotency_with(
            &tx,
            nonnegative_timestamp(self.now_millis())?,
            limit.clamp(1, MAX_DURABLE_IDEMPOTENCY_PRUNE_LIMIT),
        )
        .await?;
        tx.commit()
            .await
            .context("Failed to commit durable idempotency pruning")?;
        Ok(removed)
    }

    pub async fn record_idempotency(
        &self,
        input: NewIdempotencyRecord,
    ) -> anyhow::Result<IdempotencyRecord> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let response_json = input
            .response_json
            .as_ref()
            .map(|value| json_to_db_string(value, "{}"))
            .transpose()?;
        let metadata_json = json_to_db_string(&input.metadata_json, "{}")?;

        db.execute_raw(raw::statement(
            db,
            r#"
            INSERT INTO idempotency_keys (
                operation,
                idempotency_key,
                created_at,
                expires_at,
                request_hash,
                response_json,
                runtime_job_id,
                conflict_message,
                metadata_json
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            "#,
            vec![
                input.operation.clone().into(),
                input.idempotency_key.clone().into(),
                now.into(),
                opt_u64(input.expires_at),
                input.request_hash.into(),
                opt_string(response_json),
                opt_string(input.runtime_job_id),
                opt_string(input.conflict_message),
                metadata_json.into(),
            ],
        )?)
        .await
        .context("Failed to record idempotency key")?;

        self.get_idempotency_record(&input.operation, &input.idempotency_key)
            .await?
            .ok_or_else(|| anyhow!("Created idempotency record was not found"))
    }

    pub async fn get_idempotency_record(
        &self,
        operation: &str,
        idempotency_key: &str,
    ) -> anyhow::Result<Option<IdempotencyRecord>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                IDEMPOTENCY_RECORD_COLUMNS_SQL,
                vec![operation.into(), idempotency_key.into()],
            )?)
            .await
            .context("Failed to load idempotency record")?;

        row.as_ref().map(map_idempotency_record).transpose()
    }

    pub async fn upsert_worker_heartbeat(
        &self,
        update: WorkerHeartbeatUpdate,
    ) -> anyhow::Result<RuntimeWorkerHeartbeat> {
        let queue_classes = update
            .queue_names
            .iter()
            .filter_map(|queue| QueueClass::from_db_value(queue))
            .collect::<Vec<_>>();
        let registration = RuntimeWorkerRegistration {
            version: WORKER_REGISTRATION_VERSION,
            worker_id: update.worker_id.clone(),
            instance_id: update.worker_id.clone(),
            queue_classes: if queue_classes.is_empty() {
                vec![QueueClass::Batch]
            } else {
                queue_classes
            },
            capabilities: Vec::new(),
            model_ids: Vec::new(),
            stage_kinds: Vec::new(),
            resources: WorkerResourceCapacity::default(),
            software_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        let diagnostic_json = update.diagnostic_json;
        let details = RuntimeWorkerHeartbeatDetails {
            version: WORKER_HEARTBEAT_DETAILS_VERSION,
            available_slots: if update.current_stage_id.is_none() {
                1
            } else {
                0
            },
            active_lease_ids: update.current_stage_id.clone().into_iter().collect(),
            last_error: None,
            health_json: diagnostic_json.clone(),
        };
        self.upsert_registered_worker_heartbeat(RegisteredWorkerHeartbeatUpdate {
            registration,
            status: update.status,
            current_job_id: update.current_job_id,
            current_stage_id: update.current_stage_id,
            details,
            diagnostic_json,
        })
        .await
    }

    pub async fn upsert_registered_worker_heartbeat(
        &self,
        update: RegisteredWorkerHeartbeatUpdate,
    ) -> anyhow::Result<RuntimeWorkerHeartbeat> {
        let db = self.db.connection().await?;
        let now = self.now_millis();
        let queue_names = update
            .registration
            .queue_classes
            .iter()
            .map(|queue| queue.as_db_value())
            .collect::<Vec<_>>();
        let queue_names_json = json_to_db_string(&json!(queue_names), "[]")?;
        let registration_json = json_to_db_string(&json!(update.registration), "{}")?;
        let heartbeat_details_json = json_to_db_string(&json!(update.details), "{}")?;
        let diagnostic_json = json_to_db_string(&update.diagnostic_json, "{}")?;

        db.execute_raw(worker_heartbeat_upsert_statement(
            db,
            now,
            &update,
            queue_names_json,
            registration_json,
            heartbeat_details_json,
            diagnostic_json,
        )?)
        .await
        .context("Failed to upsert runtime worker heartbeat")?;

        self.get_worker_heartbeat(&update.registration.worker_id)
            .await?
            .ok_or_else(|| anyhow!("Runtime worker heartbeat was not found after upsert"))
    }

    pub async fn get_worker_heartbeat(
        &self,
        worker_id: &str,
    ) -> anyhow::Result<Option<RuntimeWorkerHeartbeat>> {
        let db = self.db.connection().await?;
        let row = db
            .query_one_raw(raw::statement(
                db,
                WORKER_HEARTBEAT_COLUMNS_SQL,
                vec![worker_id.into()],
            )?)
            .await
            .context("Failed to load runtime worker heartbeat")?;

        row.as_ref().map(map_worker_heartbeat).transpose()
    }

    pub async fn list_worker_heartbeats(&self) -> anyhow::Result<Vec<RuntimeWorkerHeartbeat>> {
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement(db, RUNTIME_WORKER_HEARTBEATS_SQL, vec![])?)
            .await
            .context("Failed to list runtime worker heartbeats")?;
        rows.iter().map(map_worker_heartbeat).collect()
    }

    async fn retry_stage<C: ConnectionTrait>(
        &self,
        db: &C,
        stage: &JobStage,
        lease: &StageLease,
        lease_validity: LeaseValidity,
        now: i64,
        available_at: i64,
        error_code: Option<String>,
        error_message: Option<String>,
    ) -> anyhow::Result<Option<JobStage>> {
        let sql = format!(
            r#"
                UPDATE job_stages
                SET
                    status = 'retrying',
                    updated_at = ?1,
                    lease_expires_at = NULL,
                    worker_id = NULL,
                    available_at = ?8,
                    attempt_token = NULL,
                    error_code = ?2,
                    error_message = ?3
                WHERE id = ?4
                  AND status IN ('running', 'postprocessing')
                  AND worker_id = ?5
                  AND attempt_count = ?6
                  AND (attempt_token = ?9 OR (attempt_token IS NULL AND ?9 IS NULL))
                  AND lease_expires_at IS NOT NULL
                  AND {}
                "#,
            lease_validity.sql_predicate()
        );
        let result = db
            .execute_raw(raw::statement(
                db,
                sql,
                vec![
                    now.into(),
                    opt_string(error_code.clone()),
                    opt_string(error_message.clone()),
                    stage.id.clone().into(),
                    lease.worker_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    now.into(),
                    available_at.into(),
                    opt_string(lease.attempt_token.clone()),
                ],
            )?)
            .await
            .context("Failed to mark runtime stage retrying")?;
        if result.rows_affected() == 0 {
            return Ok(None);
        }

        db.execute_raw(raw::statement(
            db,
            r#"
            UPDATE runtime_jobs
            SET status = 'retrying', updated_at = ?1, error_code = ?2, error_message = ?3
            WHERE id = ?4 AND status IN ('running', 'retrying')
            "#,
            vec![
                now.into(),
                opt_string(error_code),
                opt_string(error_message),
                stage.job_id.clone().into(),
            ],
        )?)
        .await
        .context("Failed to mark runtime job retrying")?;
        get_stage_with(db, stage.id.as_str()).await
    }

    async fn finalize_stage_cancellation_with<C: ConnectionTrait>(
        &self,
        db: &C,
        lease: &StageLease,
        now: i64,
    ) -> anyhow::Result<Option<JobStage>> {
        let result = db
            .execute_raw(raw::statement(
                db,
                r#"
                    UPDATE job_stages
                    SET
                        status = 'cancelled',
                        cancellation_state = NULL,
                        updated_at = ?1,
                        finished_at = COALESCE(finished_at, ?1),
                        lease_expires_at = NULL,
                        worker_id = NULL,
                        available_at = NULL,
                        error_code = NULL,
                        error_message = NULL
                    WHERE id = ?2
                      AND status IN ('running', 'postprocessing')
                      AND cancellation_state IN ('requested', 'execution_stopping')
                      AND worker_id = ?3
                      AND attempt_count = ?4
                      AND (attempt_token = ?5 OR (attempt_token IS NULL AND ?5 IS NULL))
                      AND lease_expires_at IS NOT NULL
                    "#,
                vec![
                    now.into(),
                    lease.stage_id.clone().into(),
                    lease.worker_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    opt_string(lease.attempt_token.clone()),
                ],
            )?)
            .await
            .context("Failed to finalize runtime stage cancellation")?;
        if result.rows_affected() == 0 {
            return Ok(None);
        }

        db.execute_raw(raw::statement(
            db,
            r#"
            UPDATE runtime_jobs
            SET
                status = 'cancelled',
                cancellation_state = NULL,
                updated_at = ?1,
                finished_at = COALESCE(finished_at, ?1)
            WHERE id = (SELECT job_id FROM job_stages WHERE id = ?2)
              AND cancellation_state IS NOT NULL
              AND status IN ('created', 'queued', 'running', 'retrying', 'postprocessing')
              AND NOT EXISTS (
                  SELECT 1 FROM job_stages
                  WHERE job_stages.job_id = runtime_jobs.id
                    AND job_stages.status IN ('running', 'postprocessing')
                    AND job_stages.cancellation_state IS NOT NULL
              )
            "#,
            vec![now.into(), lease.stage_id.clone().into()],
        )?)
        .await
        .context("Failed to finalize runtime job cancellation")?;

        let finalized = get_stage_with(db, &lease.stage_id).await?;
        if let Some(stage) = &finalized {
            self.finalize_cancelled_route_projection_with(db, &stage.job_id)
                .await?;
        }
        Ok(finalized)
    }

    async fn finalize_cancelled_route_projection_with<C: ConnectionTrait>(
        &self,
        db: &C,
        job_id: &str,
    ) -> anyhow::Result<u64> {
        let transcription = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE transcription_records
                SET
                    processing_status = 'failed',
                    processing_error = COALESCE(
                        NULLIF((SELECT cancellation_reason FROM runtime_jobs WHERE id = ?1), ''),
                        'Runtime job cancelled'
                    ),
                    processing_progress_json = NULL,
                    runtime_stage_id = NULL,
                    runtime_attempt_token = NULL
                WHERE id = (SELECT route_record_id FROM runtime_jobs WHERE id = ?1)
                  AND processing_status IN ('pending', 'processing')
                  AND EXISTS (
                      SELECT 1 FROM runtime_jobs
                      WHERE id = ?1
                        AND job_kind = 'asr_transcription'
                        AND status = 'cancelled'
                        AND route_record_kind IN ('transcription', 'speaker_attributed_asr')
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM runtime_jobs AS active
                      JOIN runtime_jobs AS cancelled ON cancelled.id = ?1
                      WHERE active.id <> cancelled.id
                        AND active.job_kind = cancelled.job_kind
                        AND active.route_record_kind = cancelled.route_record_kind
                        AND active.route_record_id = cancelled.route_record_id
                        AND active.status IN ('created', 'queued', 'running', 'paused', 'retrying', 'postprocessing')
                  )
                "#,
                vec![job_id.into()],
            )?)
            .await
            .context("Failed to finalize cancelled transcription projection")?
            .rows_affected();
        let speech = db
            .execute_raw(raw::statement(
                db,
                r#"
                UPDATE speech_history_records
                SET
                    processing_status = 'failed',
                    processing_error = COALESCE(
                        NULLIF((SELECT cancellation_reason FROM runtime_jobs WHERE id = ?1), ''),
                        'Runtime job cancelled'
                    ),
                    runtime_stage_id = NULL,
                    runtime_attempt_token = NULL
                WHERE id = (SELECT route_record_id FROM runtime_jobs WHERE id = ?1)
                  AND route_kind = (SELECT route_record_kind FROM runtime_jobs WHERE id = ?1)
                  AND processing_status IN ('pending', 'processing')
                  AND EXISTS (
                      SELECT 1 FROM runtime_jobs
                      WHERE id = ?1
                        AND job_kind = 'tts_speech'
                        AND status = 'cancelled'
                        AND route_record_kind IN ('text_to_speech', 'voice_design', 'voice_cloning')
                  )
                  AND NOT EXISTS (
                      SELECT 1
                      FROM runtime_jobs AS active
                      JOIN runtime_jobs AS cancelled ON cancelled.id = ?1
                      WHERE active.id <> cancelled.id
                        AND active.job_kind = cancelled.job_kind
                        AND active.route_record_kind = cancelled.route_record_kind
                        AND active.route_record_id = cancelled.route_record_id
                        AND active.status IN ('created', 'queued', 'running', 'paused', 'retrying', 'postprocessing')
                  )
                "#,
                vec![job_id.into()],
            )?)
            .await
            .context("Failed to finalize cancelled speech projection")?
            .rows_affected();

        Ok(transcription.saturating_add(speech))
    }

    async fn mark_stage_failed<C: ConnectionTrait>(
        &self,
        db: &C,
        stage: &JobStage,
        lease: &StageLease,
        lease_validity: LeaseValidity,
        now: i64,
        error_code: Option<String>,
        error_message: Option<String>,
    ) -> anyhow::Result<Option<JobStage>> {
        let sql = format!(
            r#"
                UPDATE job_stages
                SET
                    status = 'failed',
                    updated_at = ?1,
                    finished_at = COALESCE(finished_at, ?1),
                    lease_expires_at = NULL,
                    worker_id = NULL,
                    available_at = NULL,
                    error_code = ?2,
                    error_message = ?3
                WHERE id = ?4
                  AND status IN ('running', 'postprocessing')
                  AND worker_id = ?5
                  AND attempt_count = ?6
                  AND (attempt_token = ?8 OR (attempt_token IS NULL AND ?8 IS NULL))
                  AND lease_expires_at IS NOT NULL
                  AND {}
                "#,
            lease_validity.sql_predicate()
        );
        let result = db
            .execute_raw(raw::statement(
                db,
                sql,
                vec![
                    now.into(),
                    opt_string(error_code.clone()),
                    opt_string(error_message.clone()),
                    stage.id.clone().into(),
                    lease.worker_id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    now.into(),
                    opt_string(lease.attempt_token.clone()),
                ],
            )?)
            .await
            .context("Failed to mark runtime stage failed")?;
        if result.rows_affected() == 0 {
            return Ok(None);
        }

        db.execute_raw(raw::statement(
            db,
            r#"
            UPDATE runtime_jobs
            SET
                status = 'failed',
                updated_at = ?1,
                finished_at = COALESCE(finished_at, ?1),
                error_code = ?2,
                error_message = ?3
            WHERE id = ?4 AND status IN ('running', 'retrying', 'postprocessing')
            "#,
            vec![
                now.into(),
                opt_string(error_code),
                opt_string(error_message),
                stage.job_id.clone().into(),
            ],
        )?)
        .await
        .context("Failed to mark runtime job failed")?;
        get_stage_with(db, stage.id.as_str()).await
    }

    pub async fn reconcile_inconsistent_states(
        &self,
        limit: usize,
    ) -> anyhow::Result<RuntimeReconciliationReport> {
        let db = self.db.connection().await?;
        let tx = db
            .begin_with_options(runtime_write_transaction_options())
            .await
            .context("Failed to start runtime reconciliation transaction")?;
        let now = self.now_millis();
        let mut report = RuntimeReconciliationReport::default();
        let mut remaining = bounded_maintenance_batch_limit(limit);

        for (status, stage_status, excluded_statuses) in [
            ("failed", "failed", "'failed'"),
            ("expired", "expired", "'failed', 'expired'"),
        ] {
            if remaining == 0 {
                break;
            }
            let result = tx
                .execute_raw(raw::statement(
                    &tx,
                    format!(
                        r#"
                        UPDATE runtime_jobs
                        SET
                            status = '{status}',
                            updated_at = ?1,
                            finished_at = COALESCE(finished_at, ?1),
                            error_code = COALESCE(error_code, 'stage_{status}'),
                            error_message = COALESCE(error_message, 'Runtime stage became {status}')
                        WHERE id IN (
                            SELECT candidate.id
                            FROM runtime_jobs AS candidate
                            WHERE candidate.status IN ('created', 'queued', 'running', 'paused', 'retrying', 'postprocessing')
                              AND candidate.cancellation_state IS NULL
                              AND EXISTS (
                                  SELECT 1 FROM job_stages
                                  WHERE job_stages.job_id = candidate.id
                                    AND job_stages.status = '{stage_status}'
                              )
                              AND NOT EXISTS (
                                  SELECT 1 FROM job_stages
                                  WHERE job_stages.job_id = candidate.id
                                    AND job_stages.status IN ({excluded_statuses})
                                    AND job_stages.status <> '{stage_status}'
                              )
                            ORDER BY candidate.updated_at ASC, candidate.id ASC
                            LIMIT ?2
                          )
                        "#
                    ),
                    vec![now.into(), i64::try_from(remaining)?.into()],
                )?)
                .await
                .with_context(|| format!("Failed to reconcile {status} runtime jobs"))?;
            report.jobs_repaired = report.jobs_repaired.saturating_add(result.rows_affected());
            remaining = remaining.saturating_sub(result.rows_affected() as usize);
        }

        let cancelled = if remaining == 0 {
            0
        } else {
            tx.execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE runtime_jobs
                SET
                    status = 'cancelled',
                    cancellation_state = NULL,
                    updated_at = ?1,
                    finished_at = COALESCE(finished_at, ?1),
                    cancellation_reason = COALESCE(cancellation_reason, 'All remaining stages were cancelled')
                WHERE id IN (
                    SELECT candidate.id
                    FROM runtime_jobs AS candidate
                    WHERE candidate.status IN ('created', 'queued', 'running', 'paused', 'retrying', 'postprocessing')
                      AND EXISTS (
                          SELECT 1 FROM job_stages
                          WHERE job_stages.job_id = candidate.id AND status = 'cancelled'
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM job_stages
                          WHERE job_stages.job_id = candidate.id
                            AND status NOT IN ('completed', 'skipped', 'cancelled')
                      )
                    ORDER BY candidate.updated_at ASC, candidate.id ASC
                    LIMIT ?2
                  )
                "#,
                vec![now.into(), i64::try_from(remaining)?.into()],
            )?)
            .await
            .context("Failed to reconcile cancelled runtime jobs")?
            .rows_affected()
        };
        report.jobs_repaired = report.jobs_repaired.saturating_add(cancelled);
        remaining = remaining.saturating_sub(cancelled as usize);

        let completed = if remaining == 0 {
            0
        } else {
            tx.execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE runtime_jobs
                SET
                    status = 'completed',
                    updated_at = ?1,
                    finished_at = COALESCE(finished_at, ?1),
                    error_code = NULL,
                    error_message = NULL
                WHERE id IN (
                    SELECT candidate.id
                    FROM runtime_jobs AS candidate
                    WHERE candidate.status IN ('created', 'queued', 'running', 'retrying', 'postprocessing')
                      AND candidate.cancellation_state IS NULL
                      AND EXISTS (SELECT 1 FROM job_stages WHERE job_stages.job_id = candidate.id)
                      AND NOT EXISTS (
                          SELECT 1 FROM job_stages
                          WHERE job_stages.job_id = candidate.id
                            AND status NOT IN ('completed', 'skipped')
                      )
                    ORDER BY candidate.updated_at ASC, candidate.id ASC
                    LIMIT ?2
                  )
                "#,
                vec![now.into(), i64::try_from(remaining)?.into()],
            )?)
            .await
            .context("Failed to reconcile completed runtime jobs")?
            .rows_affected()
        };
        report.jobs_repaired = report.jobs_repaired.saturating_add(completed);
        remaining = remaining.saturating_sub(completed as usize);

        let retrying = if remaining == 0 {
            0
        } else {
            tx.execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE runtime_jobs
                SET status = 'retrying', updated_at = ?1
                WHERE id IN (
                    SELECT candidate.id
                    FROM runtime_jobs AS candidate
                    WHERE candidate.status IN ('created', 'queued', 'running', 'postprocessing')
                      AND candidate.cancellation_state IS NULL
                      AND EXISTS (
                          SELECT 1 FROM job_stages
                          WHERE job_stages.job_id = candidate.id AND status = 'retrying'
                      )
                    ORDER BY candidate.updated_at ASC, candidate.id ASC
                    LIMIT ?2
                  )
                "#,
                vec![now.into(), i64::try_from(remaining)?.into()],
            )?)
            .await
            .context("Failed to reconcile retrying runtime jobs")?
            .rows_affected()
        };
        report.jobs_repaired = report.jobs_repaired.saturating_add(retrying);
        remaining = remaining.saturating_sub(retrying as usize);

        let stages = if remaining == 0 {
            0
        } else {
            tx.execute_raw(raw::statement(
                &tx,
                r#"
                UPDATE job_stages
                SET
                    status = CASE
                        WHEN (SELECT status FROM runtime_jobs WHERE id = job_stages.job_id) = 'expired' THEN 'expired'
                        ELSE 'cancelled'
                    END,
                    updated_at = ?1,
                    finished_at = COALESCE(finished_at, ?1),
                    worker_id = NULL,
                    lease_expires_at = NULL,
                    available_at = NULL,
                    error_code = COALESCE(error_code, 'parent_terminal'),
                    error_message = COALESCE(error_message, 'Parent runtime job is terminal')
                WHERE id IN (
                    SELECT candidate.id
                    FROM job_stages AS candidate
                    WHERE candidate.status IN ('created', 'queued', 'running', 'paused', 'retrying', 'postprocessing')
                      AND EXISTS (
                          SELECT 1 FROM runtime_jobs
                          WHERE runtime_jobs.id = candidate.job_id
                            AND runtime_jobs.status IN ('failed', 'cancelled', 'expired')
                      )
                    ORDER BY candidate.updated_at ASC, candidate.id ASC
                    LIMIT ?2
                  )
                "#,
                vec![now.into(), i64::try_from(remaining)?.into()],
            )?)
            .await
            .context("Failed to reconcile stages owned by terminal runtime jobs")?
            .rows_affected()
        };
        report.stages_repaired = stages;
        remaining = remaining.saturating_sub(stages as usize);

        if remaining > 0 {
            let cancelled_route_jobs = tx
                .query_all_raw(raw::statement(
                    &tx,
                    r#"
                    SELECT candidate.id
                    FROM runtime_jobs AS candidate
                    WHERE candidate.status = 'cancelled'
                      AND NOT EXISTS (
                          SELECT 1 FROM runtime_jobs AS active
                          WHERE active.id <> candidate.id
                            AND active.job_kind = candidate.job_kind
                            AND active.route_record_kind = candidate.route_record_kind
                            AND active.route_record_id = candidate.route_record_id
                            AND active.status IN ('created', 'queued', 'running', 'paused', 'retrying', 'postprocessing')
                      )
                      AND (
                          (
                              candidate.job_kind = 'asr_transcription'
                              AND candidate.route_record_kind IN ('transcription', 'speaker_attributed_asr')
                              AND EXISTS (
                                  SELECT 1 FROM transcription_records
                                  WHERE transcription_records.id = candidate.route_record_id
                                    AND processing_status IN ('pending', 'processing')
                              )
                          )
                          OR (
                              candidate.job_kind = 'tts_speech'
                              AND candidate.route_record_kind IN ('text_to_speech', 'voice_design', 'voice_cloning')
                              AND EXISTS (
                                  SELECT 1 FROM speech_history_records
                                  WHERE speech_history_records.id = candidate.route_record_id
                                    AND speech_history_records.route_kind = candidate.route_record_kind
                                    AND processing_status IN ('pending', 'processing')
                              )
                          )
                      )
                    ORDER BY candidate.updated_at ASC, candidate.id ASC
                    LIMIT ?1
                    "#,
                    vec![i64::try_from(remaining)?.into()],
                )?)
                .await
                .context("Failed to select cancelled route projections for reconciliation")?;
            for row in cancelled_route_jobs {
                let job_id: String = row.try_get_by_index(0)?;
                report.route_projections_repaired =
                    report.route_projections_repaired.saturating_add(
                        self.finalize_cancelled_route_projection_with(&tx, &job_id)
                            .await?,
                    );
            }
        }

        tx.commit()
            .await
            .context("Failed to commit runtime reconciliation transaction")?;

        Ok(report)
    }
}

#[derive(Debug)]
struct StoredDurableIdempotency {
    expires_at: u64,
    digest_version: u16,
    request_digest: String,
    state: String,
    reservation_token: String,
    runtime_job_id: Option<String>,
    response_json: Option<serde_json::Value>,
}

impl StoredDurableIdempotency {
    fn begin_outcome(
        self,
        request: &DurableIdempotencyRequest,
    ) -> anyhow::Result<DurableIdempotencyBegin> {
        if self.digest_version != request.digest_version
            || self.request_digest != request.request_digest
        {
            return Ok(DurableIdempotencyBegin::Conflict);
        }
        match self.state.as_str() {
            "reserved" => Ok(DurableIdempotencyBegin::InProgress {
                expires_at: self.expires_at,
            }),
            "committed" => Ok(DurableIdempotencyBegin::Replay(DurableIdempotencyReplay {
                runtime_job_id: self.runtime_job_id.ok_or_else(|| {
                    anyhow!("Committed durable idempotency record is missing its runtime job")
                })?,
                response_json: self.response_json.ok_or_else(|| {
                    anyhow!("Committed durable idempotency record is missing its response")
                })?,
                expires_at: self.expires_at,
            })),
            state => bail!("Unknown durable idempotency state: {state}"),
        }
    }
}

fn validate_durable_idempotency_request(request: &DurableIdempotencyRequest) -> anyhow::Result<()> {
    validate_durable_idempotency_identity(
        &request.tenant_scope,
        &request.operation,
        &request.idempotency_key,
        request.digest_version,
        &request.request_digest,
    )?;
    anyhow::ensure!(
        (1..=MAX_DURABLE_IDEMPOTENCY_RESERVATION_TTL_MS)
            .contains(&request.reservation_ttl_ms),
        "Durable idempotency reservation TTL must be between 1 and {MAX_DURABLE_IDEMPOTENCY_RESERVATION_TTL_MS} milliseconds"
    );
    Ok(())
}

fn validate_durable_idempotency_identity(
    tenant_scope: &str,
    operation: &str,
    idempotency_key: &str,
    digest_version: u16,
    request_digest: &str,
) -> anyhow::Result<()> {
    validate_bounded_field(
        "tenant scope",
        tenant_scope,
        MAX_DURABLE_IDEMPOTENCY_TENANT_BYTES,
    )?;
    validate_bounded_field(
        "operation",
        operation,
        MAX_DURABLE_IDEMPOTENCY_OPERATION_BYTES,
    )?;
    validate_bounded_field(
        "idempotency key",
        idempotency_key,
        MAX_DURABLE_IDEMPOTENCY_KEY_BYTES,
    )?;
    validate_request_digest(digest_version, request_digest)
}

fn validate_request_digest(digest_version: u16, request_digest: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        digest_version == DURABLE_IDEMPOTENCY_DIGEST_VERSION,
        "Unsupported durable idempotency digest version: {digest_version}"
    );
    anyhow::ensure!(
        request_digest.len() == 64
            && request_digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "Durable idempotency request digest must be lowercase SHA-256 hex"
    );
    Ok(())
}

fn validate_bounded_field(name: &str, value: &str, max_bytes: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        !value.is_empty(),
        "Durable idempotency {name} cannot be empty"
    );
    anyhow::ensure!(
        value.len() <= max_bytes,
        "Durable idempotency {name} exceeds {max_bytes} bytes"
    );
    anyhow::ensure!(
        !value.chars().any(char::is_control),
        "Durable idempotency {name} contains control characters"
    );
    Ok(())
}

fn validate_optional_field_bytes(
    name: &str,
    value: Option<&str>,
    max_bytes: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        value.is_none_or(|value| value.len() <= max_bytes),
        "{name} exceeds {max_bytes} bytes"
    );
    Ok(())
}

fn nonnegative_timestamp(now: i64) -> anyhow::Result<u64> {
    u64::try_from(now).context("Durable idempotency clock preceded the Unix epoch")
}

struct BoundedJsonBytes {
    bytes: Vec<u8>,
    limit: usize,
}

impl BoundedJsonBytes {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(limit.min(8192)),
            limit,
        }
    }
}

impl Write for BoundedJsonBytes {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self.bytes.len().saturating_add(buffer.len()) > self.limit {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bounded JSON encoding exceeded its byte limit",
            ));
        }
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn bounded_json_string(
    value: &serde_json::Value,
    limit: usize,
    label: &str,
) -> anyhow::Result<String> {
    let mut encoded = BoundedJsonBytes::new(limit);
    write_canonical_json(value, &mut encoded, 0)
        .with_context(|| format!("{label} exceeds {limit} bytes or could not be encoded"))?;
    String::from_utf8(encoded.bytes).context("JSON serialization produced invalid UTF-8")
}

fn write_canonical_json<W: Write>(
    value: &serde_json::Value,
    output: &mut W,
    depth: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        depth <= MAX_DURABLE_IDEMPOTENCY_JSON_DEPTH,
        "Durable idempotency JSON exceeds maximum nesting depth of {MAX_DURABLE_IDEMPOTENCY_JSON_DEPTH}"
    );
    match value {
        serde_json::Value::Null => output.write_all(b"null")?,
        serde_json::Value::Bool(true) => output.write_all(b"true")?,
        serde_json::Value::Bool(false) => output.write_all(b"false")?,
        serde_json::Value::Number(value) => serde_json::to_writer(&mut *output, value)?,
        serde_json::Value::String(value) => serde_json::to_writer(&mut *output, value)?,
        serde_json::Value::Array(values) => {
            output.write_all(b"[")?;
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.write_all(b",")?;
                }
                write_canonical_json(value, output, depth + 1)?;
            }
            output.write_all(b"]")?;
        }
        serde_json::Value::Object(values) => {
            anyhow::ensure!(
                values.len() <= MAX_DURABLE_IDEMPOTENCY_JSON_OBJECT_KEYS,
                "Durable idempotency JSON object exceeds {MAX_DURABLE_IDEMPOTENCY_JSON_OBJECT_KEYS} keys"
            );
            output.write_all(b"{")?;
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    output.write_all(b",")?;
                }
                serde_json::to_writer(&mut *output, key)?;
                output.write_all(b":")?;
                write_canonical_json(&values[key], output, depth + 1)?;
            }
            output.write_all(b"}")?;
        }
    }
    Ok(())
}

async fn lock_durable_idempotency<C: ConnectionTrait>(db: &C) -> anyhow::Result<()> {
    let insert_sql = match db.get_database_backend() {
        DbBackend::Sqlite | DbBackend::Postgres => {
            "INSERT INTO runtime_admission_locks (id, lock_value) VALUES ('durable_idempotency_v2', 1) ON CONFLICT (id) DO NOTHING"
        }
        DbBackend::MySql => {
            "INSERT IGNORE INTO runtime_admission_locks (id, lock_value) VALUES ('durable_idempotency_v2', 1)"
        }
        backend => bail!("Unsupported durable idempotency database backend: {backend:?}"),
    };
    db.execute_raw(raw::statement(db, insert_sql, vec![])?)
        .await
        .context("Failed to initialize durable idempotency lock")?;
    db.execute_raw(raw::statement(
        db,
        "UPDATE runtime_admission_locks SET lock_value = lock_value WHERE id = 'durable_idempotency_v2'",
        vec![],
    )?)
    .await
    .context("Failed to lock durable idempotency capacity")?;
    Ok(())
}

async fn lock_artifact_cleanup_capacity<C: ConnectionTrait>(db: &C) -> anyhow::Result<()> {
    let insert_sql = match db.get_database_backend() {
        DbBackend::Sqlite | DbBackend::Postgres => {
            "INSERT INTO runtime_admission_locks (id, lock_value) VALUES ('artifact_cleanup', 1) ON CONFLICT (id) DO NOTHING"
        }
        DbBackend::MySql => {
            "INSERT IGNORE INTO runtime_admission_locks (id, lock_value) VALUES ('artifact_cleanup', 1)"
        }
        backend => bail!("Unsupported artifact cleanup database backend: {backend:?}"),
    };
    db.execute_raw(raw::statement(db, insert_sql, vec![])?)
        .await
        .context("Failed to initialize artifact cleanup lock")?;
    db.execute_raw(raw::statement(
        db,
        "UPDATE runtime_admission_locks SET lock_value = lock_value WHERE id = 'artifact_cleanup'",
        vec![],
    )?)
    .await
    .context("Failed to lock artifact cleanup capacity")?;
    Ok(())
}

async fn lock_provider_write_capacity<C: ConnectionTrait>(db: &C) -> anyhow::Result<()> {
    let insert_sql = match db.get_database_backend() {
        DbBackend::Sqlite | DbBackend::Postgres => {
            "INSERT INTO runtime_admission_locks (id, lock_value) VALUES ('provider_writes', 1) ON CONFLICT (id) DO NOTHING"
        }
        DbBackend::MySql => {
            "INSERT IGNORE INTO runtime_admission_locks (id, lock_value) VALUES ('provider_writes', 1)"
        }
        backend => bail!("Unsupported provider write database backend: {backend:?}"),
    };
    db.execute_raw(raw::statement(db, insert_sql, vec![])?)
        .await?;
    db.execute_raw(raw::statement(
        db,
        "UPDATE runtime_admission_locks SET lock_value = lock_value WHERE id = 'provider_writes'",
        vec![],
    )?)
    .await?;
    Ok(())
}

async fn quarantine_invalid_provider_write<C: ConnectionTrait>(
    db: &C,
    write_id: &str,
    now: i64,
    error: &str,
) -> anyhow::Result<()> {
    let available_at = now.saturating_add(i64::try_from(MAX_ARTIFACT_CLEANUP_BACKOFF_MS)?);
    let result = db
        .execute_raw(raw::statement(
            db,
            r#"
            UPDATE provider_write_reservations
            SET state = 'cleanup_pending', updated_at = ?1, available_at = ?2,
                cleanup_attempt_count = CASE
                    WHEN cleanup_attempt_count < 4294967295
                    THEN cleanup_attempt_count + 1
                    ELSE cleanup_attempt_count
                END,
                last_error = ?3, cleanup_claim_token = NULL,
                cleanup_claim_expires_at = NULL
            WHERE write_id = ?4 AND state = 'cleanup_claimed'
            "#,
            vec![
                now.into(),
                available_at.into(),
                truncate_utf8_bytes(error, MAX_ARTIFACT_CLEANUP_ERROR_BYTES).into(),
                write_id.into(),
            ],
        )?)
        .await
        .context("Failed to quarantine invalid provider write request")?;
    anyhow::ensure!(
        result.rows_affected() == 1,
        "Invalid provider write request lost its cleanup claim"
    );
    Ok(())
}

fn validate_provider_write_input(input: &NewProviderWriteReservation) -> anyhow::Result<()> {
    let write_id = uuid::Uuid::parse_str(&input.write_id)
        .map_err(|_| anyhow!("Invalid provider write identity"))?;
    anyhow::ensure!(
        write_id.hyphenated().to_string() == input.write_id,
        "Invalid provider write identity"
    );
    validate_artifact_cleanup_tenant(&input.tenant_scope)?;
    anyhow::ensure!(
        !input.storage_namespace.is_empty()
            && input.storage_namespace.len() <= MAX_PROVIDER_WRITE_NAMESPACE_BYTES
            && !input.storage_namespace.chars().any(char::is_control),
        "Invalid provider write namespace"
    );
    anyhow::ensure!(
        !input.content_type.is_empty()
            && input.content_type.len() <= MAX_PROVIDER_WRITE_CONTENT_TYPE_BYTES
            && !input.content_type.chars().any(char::is_control),
        "Invalid provider write content type"
    );
    anyhow::ensure!(
        input.filename.as_ref().is_none_or(|filename| {
            !filename.is_empty()
                && filename.len() <= MAX_PROVIDER_WRITE_FILENAME_BYTES
                && !filename.chars().any(char::is_control)
        }),
        "Invalid provider write filename"
    );
    anyhow::ensure!(
        input.expected_size_bytes > 0 && input.expected_size_bytes <= MAX_PROVIDER_WRITE_BYTES,
        "Invalid provider write size"
    );
    anyhow::ensure!(
        input.expected_sha256.len() == 64
            && input
                .expected_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit()),
        "Invalid provider write digest"
    );
    anyhow::ensure!(
        input.lifetime_ms > 0 && input.lifetime_ms <= MAX_PROVIDER_WRITE_LIFETIME_MS,
        "Invalid provider write lifetime"
    );
    validate_provider_write_request(
        &input.provider_request,
        &input.write_id,
        &input.tenant_scope,
        &input.storage_namespace,
        &input.content_type,
        input.filename.as_deref(),
    )?;
    Ok(())
}

fn validate_provider_write_request(
    request: &MediaWriteRequest,
    write_id: &str,
    tenant_scope: &str,
    storage_namespace: &str,
    content_type: &str,
    filename: Option<&str>,
) -> anyhow::Result<()> {
    match &request.namespace {
        MediaNamespace::Other(namespace) => anyhow::ensure!(
            !namespace.is_empty()
                && namespace.len() <= MAX_PROVIDER_WRITE_NAMESPACE_BYTES
                && !namespace.chars().any(char::is_control),
            "Invalid provider request namespace"
        ),
        MediaNamespace::TranscriptionUpload
        | MediaNamespace::DiarizationUpload
        | MediaNamespace::GeneratedSpeech
        | MediaNamespace::SavedVoice
        | MediaNamespace::ChatMedia
        | MediaNamespace::Export => {}
    }
    anyhow::ensure!(
        provider_namespace_storage_value(&request.namespace) == storage_namespace,
        "Provider request namespace does not match its reservation"
    );
    anyhow::ensure!(
        request.record_id == write_id
            && !request.record_id.is_empty()
            && request.record_id.len() <= MAX_PROVIDER_WRITE_RECORD_ID_BYTES
            && !request.record_id.chars().any(char::is_control),
        "Invalid provider request record identity"
    );
    anyhow::ensure!(
        request.content_type == content_type && request.preferred_filename.as_deref() == filename,
        "Provider request does not match its reservation"
    );
    anyhow::ensure!(
        request.metadata.get("tenant_id").map(String::as_str) == Some(tenant_scope),
        "Provider request tenant does not match its reservation"
    );
    anyhow::ensure!(
        request.metadata.len() <= MAX_PROVIDER_WRITE_METADATA_ENTRIES,
        "Provider request metadata entry count exceeds the limit"
    );
    let mut metadata_bytes = 0usize;
    for (key, value) in &request.metadata {
        anyhow::ensure!(
            !key.is_empty()
                && key.len() <= MAX_PROVIDER_WRITE_METADATA_KEY_BYTES
                && !key.chars().any(char::is_control),
            "Invalid provider request metadata key"
        );
        anyhow::ensure!(
            value.len() <= MAX_PROVIDER_WRITE_METADATA_VALUE_BYTES
                && !value.chars().any(char::is_control),
            "Invalid provider request metadata value"
        );
        metadata_bytes = metadata_bytes
            .checked_add(key.len())
            .and_then(|bytes| bytes.checked_add(value.len()))
            .context("Provider request metadata size overflowed")?;
    }
    anyhow::ensure!(
        metadata_bytes <= MAX_PROVIDER_WRITE_METADATA_BYTES,
        "Provider request metadata exceeds the aggregate byte limit"
    );
    Ok(())
}

fn serialize_provider_write_request(request: &MediaWriteRequest) -> anyhow::Result<String> {
    let mut output = BoundedJsonBuffer::new(MAX_PROVIDER_WRITE_REQUEST_ENVELOPE_BYTES);
    let result = serde_json::to_writer(
        &mut output,
        &ProviderWriteRequestEnvelope {
            version: PROVIDER_WRITE_REQUEST_ENVELOPE_VERSION,
            request: request.into(),
        },
    );
    if output.overflowed {
        bail!("Provider write request envelope exceeds the byte limit");
    }
    result.context("Failed to serialize provider write request envelope")?;
    String::from_utf8(output.bytes).context("Provider write request envelope was not UTF-8")
}

fn deserialize_provider_write_request(raw: &str) -> anyhow::Result<MediaWriteRequest> {
    anyhow::ensure!(
        raw.len() <= MAX_PROVIDER_WRITE_REQUEST_ENVELOPE_BYTES,
        "Stored provider write request envelope exceeds the byte limit"
    );
    let envelope: ProviderWriteRequestEnvelope =
        serde_json::from_str(raw).context("Stored provider write request envelope is invalid")?;
    anyhow::ensure!(
        envelope.version == PROVIDER_WRITE_REQUEST_ENVELOPE_VERSION,
        "Stored provider write request envelope version is unsupported"
    );
    Ok(envelope.request.into())
}

fn legacy_provider_write_request(
    write_id: &str,
    tenant_scope: &str,
    storage_namespace: &str,
    content_type: &str,
    filename: Option<&str>,
) -> anyhow::Result<MediaWriteRequest> {
    anyhow::ensure!(
        storage_namespace == "artifact-store",
        "Legacy provider write request cannot be reconstructed safely"
    );
    let mut metadata = HookMetadata::new();
    metadata.insert("tenant_id".to_string(), tenant_scope.to_string());
    Ok(MediaWriteRequest {
        namespace: MediaNamespace::Other(storage_namespace.to_string()),
        record_id: write_id.to_string(),
        preferred_filename: filename.map(str::to_string),
        content_type: content_type.to_string(),
        metadata,
    })
}

fn provider_namespace_storage_value(namespace: &MediaNamespace) -> &str {
    match namespace {
        MediaNamespace::TranscriptionUpload => "transcription_upload",
        MediaNamespace::DiarizationUpload => "diarization_upload",
        MediaNamespace::GeneratedSpeech => "generated_speech",
        MediaNamespace::SavedVoice => "saved_voice",
        MediaNamespace::ChatMedia => "chat_media",
        MediaNamespace::Export => "export",
        MediaNamespace::Other(namespace) => namespace,
    }
}

fn provider_write_request_length_sql(backend: DbBackend) -> anyhow::Result<&'static str> {
    match backend {
        DbBackend::Sqlite => Ok("LENGTH(CAST(provider_request_json AS BLOB))"),
        DbBackend::Postgres | DbBackend::MySql => Ok("OCTET_LENGTH(provider_request_json)"),
        backend => bail!("Unsupported provider write database backend: {backend:?}"),
    }
}

struct BoundedJsonBuffer {
    bytes: Vec<u8>,
    max_bytes: usize,
    overflowed: bool,
}

impl BoundedJsonBuffer {
    fn new(max_bytes: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(max_bytes.min(1024)),
            max_bytes,
            overflowed: false,
        }
    }
}

impl Write for BoundedJsonBuffer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let Some(new_len) = self.bytes.len().checked_add(bytes.len()) else {
            self.overflowed = true;
            return Err(io::Error::other("bounded JSON buffer overflow"));
        };
        if new_len > self.max_bytes {
            self.overflowed = true;
            return Err(io::Error::other("bounded JSON buffer limit exceeded"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(crate) fn validate_artifact_cleanup_storage_key(storage_key: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !storage_key.is_empty()
            && storage_key.len() <= MAX_ARTIFACT_CLEANUP_STORAGE_KEY_BYTES
            && !storage_key.chars().any(char::is_control),
        "Invalid artifact cleanup storage key"
    );
    Ok(())
}

fn validate_artifact_cleanup_tenant(tenant_scope: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !tenant_scope.is_empty()
            && tenant_scope.len() <= MAX_ARTIFACT_CLEANUP_TENANT_BYTES
            && tenant_scope
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"_-.:@".contains(&byte)),
        "Invalid artifact cleanup tenant scope"
    );
    Ok(())
}

fn truncate_utf8_bytes(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_string();
    }
    let mut boundary = max_bytes;
    while !value.is_char_boundary(boundary) {
        boundary -= 1;
    }
    value[..boundary].to_string()
}

async fn load_durable_idempotency_with<C: ConnectionTrait>(
    db: &C,
    tenant_scope: &str,
    operation: &str,
    idempotency_key: &str,
) -> anyhow::Result<Option<StoredDurableIdempotency>> {
    let row = db
        .query_one_raw(raw::statement(
            db,
            r#"
            SELECT expires_at, digest_version, request_digest, state,
                   reservation_token, runtime_job_id, response_json
            FROM durable_idempotency_keys_v2
            WHERE tenant_scope = ?1 AND operation = ?2 AND idempotency_key = ?3
            "#,
            vec![
                tenant_scope.into(),
                operation.into(),
                idempotency_key.into(),
            ],
        )?)
        .await
        .context("Failed to load durable idempotency key")?;
    row.map(|row| -> anyhow::Result<StoredDurableIdempotency> {
        let digest_version = u16::try_from(row.try_get_by_index::<i64>(1)?)?;
        let request_digest: String = row.try_get_by_index(2)?;
        validate_request_digest(digest_version, &request_digest)?;
        let state: String = row.try_get_by_index(3)?;
        anyhow::ensure!(
            matches!(state.as_str(), "reserved" | "committed"),
            "Unknown durable idempotency state: {state}"
        );
        let reservation_token: String = row.try_get_by_index(4)?;
        validate_bounded_field("reservation token", &reservation_token, 64)?;
        let runtime_job_id: Option<String> = row.try_get_by_index(5)?;
        if let Some(runtime_job_id) = runtime_job_id.as_deref() {
            validate_bounded_field("runtime job ID", runtime_job_id, 128)?;
        }
        let response_json = match row.try_get_by_index::<Option<String>>(6)? {
            Some(raw) => {
                anyhow::ensure!(
                    raw.len() <= MAX_DURABLE_IDEMPOTENCY_RESULT_BYTES,
                    "Stored durable idempotency result exceeds {MAX_DURABLE_IDEMPOTENCY_RESULT_BYTES} bytes"
                );
                Some(
                    serde_json::from_str(&raw)
                        .context("Failed to parse durable idempotency response")?,
                )
            }
            None => None,
        };
        Ok(StoredDurableIdempotency {
            expires_at: u64::try_from(row.try_get_by_index::<i64>(0)?)?,
            digest_version,
            request_digest,
            state,
            reservation_token,
            runtime_job_id,
            response_json,
        })
    })
    .transpose()
}

async fn delete_durable_idempotency_with<C: ConnectionTrait>(
    db: &C,
    tenant_scope: &str,
    operation: &str,
    idempotency_key: &str,
    now: u64,
) -> anyhow::Result<bool> {
    let result = db
        .execute_raw(raw::statement(
            db,
            r#"
            DELETE FROM durable_idempotency_keys_v2
            WHERE tenant_scope = ?1 AND operation = ?2 AND idempotency_key = ?3
              AND expires_at <= ?4
            "#,
            vec![
                tenant_scope.into(),
                operation.into(),
                idempotency_key.into(),
                i64::try_from(now)?.into(),
            ],
        )?)
        .await
        .context("Failed to delete expired durable idempotency key")?;
    Ok(result.rows_affected() == 1)
}

async fn prune_expired_durable_idempotency_with<C: ConnectionTrait>(
    db: &C,
    now: u64,
    limit: usize,
) -> anyhow::Result<u64> {
    let rows = db
        .query_all_raw(raw::statement(
            db,
            r#"
            SELECT tenant_scope, operation, idempotency_key
            FROM durable_idempotency_keys_v2
            WHERE expires_at <= ?1
            ORDER BY expires_at ASC, created_at ASC, tenant_scope ASC,
                     operation ASC, idempotency_key ASC
            LIMIT ?2
            "#,
            vec![i64::try_from(now)?.into(), i64::try_from(limit)?.into()],
        )?)
        .await
        .context("Failed to select expired durable idempotency keys")?;
    let mut removed = 0_u64;
    for row in rows {
        let tenant_scope: String = row.try_get_by_index(0)?;
        let operation: String = row.try_get_by_index(1)?;
        let idempotency_key: String = row.try_get_by_index(2)?;
        if delete_durable_idempotency_with(db, &tenant_scope, &operation, &idempotency_key, now)
            .await?
        {
            removed = removed.saturating_add(1);
        }
    }
    Ok(removed)
}

async fn get_job_with<C: ConnectionTrait>(db: &C, id: &str) -> anyhow::Result<Option<RuntimeJob>> {
    let row = db
        .query_one_raw(raw::statement(
            db,
            RUNTIME_JOB_COLUMNS_SQL,
            vec![id.into()],
        )?)
        .await
        .context("Failed to load runtime job")?;
    row.as_ref().map(map_runtime_job).transpose()
}

async fn get_stage_with<C: ConnectionTrait>(db: &C, id: &str) -> anyhow::Result<Option<JobStage>> {
    let row = db
        .query_one_raw(raw::statement(db, JOB_STAGE_COLUMNS_SQL, vec![id.into()])?)
        .await
        .context("Failed to load runtime job stage")?;
    row.as_ref().map(map_job_stage).transpose()
}

async fn get_artifact_with<C: ConnectionTrait>(
    db: &C,
    id: &str,
) -> anyhow::Result<Option<RuntimeArtifact>> {
    let row = db
        .query_one_raw(raw::statement(
            db,
            RUNTIME_ARTIFACT_COLUMNS_SQL,
            vec![id.into()],
        )?)
        .await
        .context("Failed to load runtime artifact")?;
    row.as_ref().map(map_runtime_artifact).transpose()
}

async fn get_media_asset_with<C: ConnectionTrait>(
    db: &C,
    id: &str,
) -> anyhow::Result<Option<MediaAsset>> {
    let row = db
        .query_one_raw(raw::statement(
            db,
            MEDIA_ASSET_COLUMNS_SQL,
            vec![id.into()],
        )?)
        .await?;
    row.as_ref().map(map_media_asset).transpose()
}

async fn get_provider_write_with<C: ConnectionTrait>(
    db: &C,
    write_id: &str,
) -> anyhow::Result<Option<ProviderWriteReservation>> {
    let request_length = provider_write_request_length_sql(db.get_database_backend())?;
    let sql = format!(
        "{PROVIDER_WRITE_COLUMNS_SQL} AND (provider_request_json IS NULL OR {request_length} <= ?2)"
    );
    let row = db
        .query_one_raw(raw::statement(
            db,
            sql,
            vec![
                write_id.into(),
                i64::try_from(MAX_PROVIDER_WRITE_REQUEST_ENVELOPE_BYTES)?.into(),
            ],
        )?)
        .await?;
    row.as_ref().map(map_provider_write).transpose()
}

async fn complete_job_if_all_stages_finished_with<C: ConnectionTrait>(
    db: &C,
    job_id: &str,
    now: i64,
) -> anyhow::Result<()> {
    db.execute_raw(raw::statement(
        db,
        r#"
        UPDATE runtime_jobs
        SET
            status = 'completed',
            updated_at = ?1,
            finished_at = COALESCE(finished_at, ?1),
            error_code = NULL,
            error_message = NULL
        WHERE id = ?2
          AND status IN ('running', 'retrying', 'postprocessing', 'queued')
          AND cancellation_state IS NULL
          AND EXISTS (SELECT 1 FROM job_stages WHERE job_id = ?2)
          AND NOT EXISTS (
              SELECT 1 FROM job_stages
              WHERE job_id = ?2 AND status NOT IN ('completed', 'skipped')
          )
        "#,
        vec![now.into(), job_id.into()],
    )?)
    .await
    .context("Failed to complete runtime job after its stages finished")?;
    Ok(())
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn current_timestamp_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

const MEDIA_ASSET_COLUMNS_SQL: &str =
    "SELECT id, created_at, updated_at, asset_kind, storage_namespace, storage_key, content_type, filename, size_bytes, sha256, duration_secs, sample_rate_hz, channel_count, peak_amplitude, rms_amplitude, source_asset_id, canonical_profile_version, scan_status, retention_policy, deleted_at, metadata_json FROM media_assets WHERE id = ?1";
const PROVIDER_WRITE_COLUMNS_SQL: &str = "SELECT write_id, reservation_token, created_at, expires_at, tenant_scope, storage_namespace, content_type, filename, expected_size_bytes, expected_sha256, provider_request_json, storage_key, cleanup_claim_token, cleanup_attempt_count FROM provider_write_reservations WHERE write_id = ?1";
const MEDIA_ASSET_BY_STORAGE_KEY_SQL: &str =
    "SELECT id, created_at, updated_at, asset_kind, storage_namespace, storage_key, content_type, filename, size_bytes, sha256, duration_secs, sample_rate_hz, channel_count, peak_amplitude, rms_amplitude, source_asset_id, canonical_profile_version, scan_status, retention_policy, deleted_at, metadata_json FROM media_assets WHERE storage_key = ?1 AND deleted_at IS NULL";
const MEDIA_ASSET_BY_SOURCE_PROFILE_SQL: &str =
    "SELECT id, created_at, updated_at, asset_kind, storage_namespace, storage_key, content_type, filename, size_bytes, sha256, duration_secs, sample_rate_hz, channel_count, peak_amplitude, rms_amplitude, source_asset_id, canonical_profile_version, scan_status, retention_policy, deleted_at, metadata_json FROM media_assets WHERE source_asset_id = ?1 AND canonical_profile_version = ?2 AND deleted_at IS NULL";
const TEXT_ASSET_COLUMNS_SQL: &str =
    "SELECT id, created_at, updated_at, raw_text, normalized_text, language_hint, character_count, sha256, safety_status, retention_policy, structure_json FROM text_assets WHERE id = ?1";
const RUNTIME_JOB_COLUMNS_SQL: &str =
    "SELECT id, created_at, updated_at, queued_at, started_at, finished_at, job_kind, status, priority, model_id, capability, route_record_kind, route_record_id, input_media_asset_id, input_text_asset_id, request_json, model_snapshot_json, progress_json, error_code, error_message, attempt_count, max_attempts, retry_policy_json, idempotency_key, correlation_id, cancellation_reason, cancellation_state FROM runtime_jobs WHERE id = ?1";
const JOB_STAGE_COLUMNS_SQL: &str =
    "SELECT id, job_id, created_at, updated_at, sequence, stage_kind, queue_class, resource_hints_json, status, capability, model_id, worker_id, lease_expires_at, available_at, attempt_token, attempt_count, max_attempts, input_artifact_ids_json, output_artifact_ids_json, progress_json, started_at, finished_at, error_code, error_message, cancellation_state FROM job_stages WHERE id = ?1";
const JOB_STAGE_LIST_FOR_JOB_SQL: &str =
    "SELECT id, job_id, created_at, updated_at, sequence, stage_kind, queue_class, resource_hints_json, status, capability, model_id, worker_id, lease_expires_at, available_at, attempt_token, attempt_count, max_attempts, input_artifact_ids_json, output_artifact_ids_json, progress_json, started_at, finished_at, error_code, error_message, cancellation_state FROM job_stages WHERE job_id = ?1 ORDER BY sequence ASC, created_at ASC, id ASC";
const RUNTIME_ARTIFACT_COLUMNS_SQL: &str =
    "SELECT id, job_id, stage_id, producer_attempt_count, producer_attempt_token, publication_key, created_at, artifact_kind, artifact_role, media_asset_id, text_asset_id, storage_key, content_type, filename, size_bytes, sha256, metadata_json, retention_policy FROM runtime_artifacts WHERE id = ?1";
const RUNTIME_ARTIFACT_LIST_FOR_JOB_SQL: &str =
    "SELECT id, job_id, stage_id, producer_attempt_count, producer_attempt_token, publication_key, created_at, artifact_kind, artifact_role, media_asset_id, text_asset_id, storage_key, content_type, filename, size_bytes, sha256, metadata_json, retention_policy FROM runtime_artifacts WHERE job_id = ?1 ORDER BY created_at ASC, id ASC";
const IDEMPOTENCY_RECORD_COLUMNS_SQL: &str =
    "SELECT operation, idempotency_key, created_at, expires_at, request_hash, response_json, runtime_job_id, conflict_message, metadata_json FROM idempotency_keys WHERE operation = ?1 AND idempotency_key = ?2";
const WORKER_HEARTBEAT_COLUMNS_SQL: &str =
    "SELECT worker_id, started_at, last_heartbeat_at, status, queue_names_json, instance_id, registration_version, registration_json, heartbeat_version, available_slots, heartbeat_details_json, current_job_id, current_stage_id, diagnostic_json FROM runtime_worker_heartbeats WHERE worker_id = ?1";
const RUNTIME_WORKER_HEARTBEATS_SQL: &str =
    "SELECT worker_id, started_at, last_heartbeat_at, status, queue_names_json, instance_id, registration_version, registration_json, heartbeat_version, available_slots, heartbeat_details_json, current_job_id, current_stage_id, diagnostic_json FROM runtime_worker_heartbeats ORDER BY worker_id";

fn worker_heartbeat_upsert_statement(
    db: &DatabaseConnection,
    now: i64,
    update: &RegisteredWorkerHeartbeatUpdate,
    queue_names_json: String,
    registration_json: String,
    heartbeat_details_json: String,
    diagnostic_json: String,
) -> anyhow::Result<sea_orm::Statement> {
    let values = vec![
        update.registration.worker_id.clone().into(),
        now.into(),
        update.status.clone().into(),
        queue_names_json.into(),
        update.registration.instance_id.clone().into(),
        i64::from(update.registration.version).into(),
        registration_json.into(),
        i64::from(update.details.version).into(),
        i64::from(update.details.available_slots).into(),
        heartbeat_details_json.into(),
        opt_string(update.current_job_id.clone()),
        opt_string(update.current_stage_id.clone()),
        diagnostic_json.into(),
    ];

    match db.get_database_backend() {
        DbBackend::Sqlite | DbBackend::Postgres => raw::statement(
            db,
            r#"
            INSERT INTO runtime_worker_heartbeats (
                worker_id,
                started_at,
                last_heartbeat_at,
                status,
                queue_names_json,
                instance_id,
                registration_version,
                registration_json,
                heartbeat_version,
                available_slots,
                heartbeat_details_json,
                current_job_id,
                current_stage_id,
                diagnostic_json
            )
            VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
            ON CONFLICT(worker_id) DO UPDATE SET
                started_at = CASE
                    WHEN runtime_worker_heartbeats.instance_id <> excluded.instance_id THEN excluded.started_at
                    ELSE runtime_worker_heartbeats.started_at
                END,
                last_heartbeat_at = excluded.last_heartbeat_at,
                status = excluded.status,
                queue_names_json = excluded.queue_names_json,
                instance_id = excluded.instance_id,
                registration_version = excluded.registration_version,
                registration_json = excluded.registration_json,
                heartbeat_version = excluded.heartbeat_version,
                available_slots = excluded.available_slots,
                heartbeat_details_json = excluded.heartbeat_details_json,
                current_job_id = excluded.current_job_id,
                current_stage_id = excluded.current_stage_id,
                diagnostic_json = excluded.diagnostic_json
            "#,
            values,
        ),
        DbBackend::MySql => raw::statement(
            db,
            r#"
            INSERT INTO runtime_worker_heartbeats (
                worker_id,
                started_at,
                last_heartbeat_at,
                status,
                queue_names_json,
                instance_id,
                registration_version,
                registration_json,
                heartbeat_version,
                available_slots,
                heartbeat_details_json,
                current_job_id,
                current_stage_id,
                diagnostic_json
            )
            VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
            ON DUPLICATE KEY UPDATE
                started_at = IF(instance_id <> VALUES(instance_id), VALUES(started_at), started_at),
                last_heartbeat_at = VALUES(last_heartbeat_at),
                status = VALUES(status),
                queue_names_json = VALUES(queue_names_json),
                instance_id = VALUES(instance_id),
                registration_version = VALUES(registration_version),
                registration_json = VALUES(registration_json),
                heartbeat_version = VALUES(heartbeat_version),
                available_slots = VALUES(available_slots),
                heartbeat_details_json = VALUES(heartbeat_details_json),
                current_job_id = VALUES(current_job_id),
                current_stage_id = VALUES(current_stage_id),
                diagnostic_json = VALUES(diagnostic_json)
            "#,
            values,
        ),
        backend => bail!("Unsupported SeaORM database backend: {backend:?}"),
    }
}

fn map_media_asset(row: &QueryResult) -> anyhow::Result<MediaAsset> {
    Ok(MediaAsset {
        id: row.try_get_by_index(0)?,
        created_at: i64_to_u64(row.try_get_by_index(1)?)?,
        updated_at: i64_to_u64(row.try_get_by_index(2)?)?,
        asset_kind: row.try_get_by_index(3)?,
        storage_namespace: row.try_get_by_index(4)?,
        storage_key: row.try_get_by_index(5)?,
        content_type: row.try_get_by_index(6)?,
        filename: row.try_get_by_index(7)?,
        size_bytes: i64_to_u64(row.try_get_by_index(8)?)?,
        sha256: row.try_get_by_index(9)?,
        duration_secs: row.try_get_by_index(10)?,
        sample_rate_hz: opt_i64_to_u32(row.try_get_by_index(11)?)?,
        channel_count: opt_i64_to_u16(row.try_get_by_index(12)?)?,
        peak_amplitude: row
            .try_get_by_index::<Option<f64>>(13)?
            .map(|value| value as f32),
        rms_amplitude: row
            .try_get_by_index::<Option<f64>>(14)?
            .map(|value| value as f32),
        source_asset_id: row.try_get_by_index(15)?,
        canonical_profile_version: row.try_get_by_index(16)?,
        scan_status: row.try_get_by_index(17)?,
        retention_policy: row.try_get_by_index(18)?,
        deleted_at: opt_i64_to_u64(row.try_get_by_index(19)?)?,
        metadata_json: parse_json_value(row.try_get_by_index::<String>(20)?, json!({})),
    })
}

fn map_artifact_cleanup_intent(row: &QueryResult) -> anyhow::Result<ArtifactCleanupIntent> {
    let storage_key: String = row.try_get_by_index(4)?;
    let tenant_scope: String = row.try_get_by_index(5)?;
    validate_artifact_cleanup_storage_key(&storage_key)?;
    validate_artifact_cleanup_tenant(&tenant_scope)?;
    let last_error: Option<String> = row.try_get_by_index(8)?;
    anyhow::ensure!(
        last_error
            .as_ref()
            .is_none_or(|error| error.len() <= MAX_ARTIFACT_CLEANUP_ERROR_BYTES),
        "Stored artifact cleanup error exceeds its bound"
    );
    Ok(ArtifactCleanupIntent {
        id: row.try_get_by_index(0)?,
        created_at: i64_to_u64(row.try_get_by_index(1)?)?,
        updated_at: i64_to_u64(row.try_get_by_index(2)?)?,
        available_at: i64_to_u64(row.try_get_by_index(3)?)?,
        storage_key,
        tenant_scope,
        reason: ArtifactCleanupReason::from_db_value(&row.try_get_by_index::<String>(6)?)?,
        attempt_count: i64_to_u32(row.try_get_by_index(7)?)?,
        last_error,
    })
}

fn map_provider_write(row: &QueryResult) -> anyhow::Result<ProviderWriteReservation> {
    let write_id: String = row.try_get_by_index(0)?;
    let tenant_scope: String = row.try_get_by_index(4)?;
    let storage_namespace: String = row.try_get_by_index(5)?;
    let content_type: String = row.try_get_by_index(6)?;
    let filename: Option<String> = row.try_get_by_index(7)?;
    let provider_request = match row.try_get_by_index::<Option<String>>(10)? {
        Some(raw) => deserialize_provider_write_request(&raw)?,
        None => legacy_provider_write_request(
            &write_id,
            &tenant_scope,
            &storage_namespace,
            &content_type,
            filename.as_deref(),
        )?,
    };
    let reservation = ProviderWriteReservation {
        write_id,
        reservation_token: row.try_get_by_index(1)?,
        created_at: i64_to_u64(row.try_get_by_index(2)?)?,
        expires_at: i64_to_u64(row.try_get_by_index(3)?)?,
        tenant_scope,
        storage_namespace,
        content_type,
        filename,
        expected_size_bytes: i64_to_u64(row.try_get_by_index(8)?)?,
        expected_sha256: row.try_get_by_index(9)?,
        provider_request,
        storage_key: row.try_get_by_index(11)?,
        cleanup_claim_token: row.try_get_by_index(12)?,
        cleanup_attempt_count: i64_to_u32(row.try_get_by_index(13)?)?,
    };
    anyhow::ensure!(
        uuid::Uuid::parse_str(&reservation.write_id).is_ok()
            && uuid::Uuid::parse_str(&reservation.reservation_token).is_ok()
            && reservation
                .cleanup_claim_token
                .as_deref()
                .is_none_or(|token| { uuid::Uuid::parse_str(token).is_ok() }),
        "Stored provider write identity is invalid"
    );
    validate_provider_write_input(&NewProviderWriteReservation {
        write_id: reservation.write_id.clone(),
        tenant_scope: reservation.tenant_scope.clone(),
        storage_namespace: reservation.storage_namespace.clone(),
        content_type: reservation.content_type.clone(),
        filename: reservation.filename.clone(),
        expected_size_bytes: reservation.expected_size_bytes,
        expected_sha256: reservation.expected_sha256.clone(),
        lifetime_ms: reservation
            .expires_at
            .saturating_sub(reservation.created_at),
        provider_request: reservation.provider_request.clone(),
    })?;
    if let Some(key) = reservation.storage_key.as_deref() {
        validate_artifact_cleanup_storage_key(key)?;
    }
    Ok(reservation)
}

fn map_text_asset(row: &QueryResult) -> anyhow::Result<TextAsset> {
    Ok(TextAsset {
        id: row.try_get_by_index(0)?,
        created_at: i64_to_u64(row.try_get_by_index(1)?)?,
        updated_at: i64_to_u64(row.try_get_by_index(2)?)?,
        raw_text: row.try_get_by_index(3)?,
        normalized_text: row.try_get_by_index(4)?,
        language_hint: row.try_get_by_index(5)?,
        character_count: i64_to_u64(row.try_get_by_index(6)?)?,
        sha256: row.try_get_by_index(7)?,
        safety_status: row.try_get_by_index(8)?,
        retention_policy: row.try_get_by_index(9)?,
        structure_json: parse_json_value(row.try_get_by_index::<String>(10)?, json!({})),
    })
}

fn map_runtime_job(row: &QueryResult) -> anyhow::Result<RuntimeJob> {
    let kind_raw: String = row.try_get_by_index(6)?;
    let status_raw: String = row.try_get_by_index(7)?;

    Ok(RuntimeJob {
        id: row.try_get_by_index(0)?,
        created_at: i64_to_u64(row.try_get_by_index(1)?)?,
        updated_at: i64_to_u64(row.try_get_by_index(2)?)?,
        queued_at: opt_i64_to_u64(row.try_get_by_index(3)?)?,
        started_at: opt_i64_to_u64(row.try_get_by_index(4)?)?,
        finished_at: opt_i64_to_u64(row.try_get_by_index(5)?)?,
        job_kind: RuntimeJobKind::from_db_value(kind_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime job kind: {kind_raw}"))?,
        status: RuntimeJobStatus::from_db_value(status_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime job status: {status_raw}"))?,
        priority: i64_to_i32(row.try_get_by_index(8)?)?,
        model_id: row.try_get_by_index(9)?,
        capability: row.try_get_by_index(10)?,
        route_record_kind: row.try_get_by_index(11)?,
        route_record_id: row.try_get_by_index(12)?,
        input_media_asset_id: row.try_get_by_index(13)?,
        input_text_asset_id: row.try_get_by_index(14)?,
        request_json: parse_json_value(row.try_get_by_index::<String>(15)?, json!({})),
        model_snapshot_json: parse_json_value(row.try_get_by_index::<String>(16)?, json!({})),
        progress_json: row
            .try_get_by_index::<Option<String>>(17)?
            .map(|raw| parse_json_value(raw, json!({}))),
        error_code: row.try_get_by_index(18)?,
        error_message: row.try_get_by_index(19)?,
        attempt_count: i64_to_u32(row.try_get_by_index(20)?)?,
        max_attempts: i64_to_u32(row.try_get_by_index(21)?)?,
        retry_policy_json: parse_json_value(row.try_get_by_index::<String>(22)?, json!({})),
        idempotency_key: row.try_get_by_index(23)?,
        correlation_id: row.try_get_by_index(24)?,
        cancellation_reason: row.try_get_by_index(25)?,
        cancellation_state: parse_cancellation_state(row.try_get_by_index(26)?)?,
    })
}

fn map_job_stage(row: &QueryResult) -> anyhow::Result<JobStage> {
    let queue_class_raw: String = row.try_get_by_index(6)?;
    let status_raw: String = row.try_get_by_index(8)?;

    Ok(JobStage {
        id: row.try_get_by_index(0)?,
        job_id: row.try_get_by_index(1)?,
        created_at: i64_to_u64(row.try_get_by_index(2)?)?,
        updated_at: i64_to_u64(row.try_get_by_index(3)?)?,
        sequence: i64_to_u32(row.try_get_by_index(4)?)?,
        stage_kind: row.try_get_by_index(5)?,
        queue_class: QueueClass::from_db_value(&queue_class_raw)
            .ok_or_else(|| anyhow!("Unknown runtime queue class: {queue_class_raw}"))?,
        resource_hints: parse_resource_hints(row.try_get_by_index(7)?),
        status: RuntimeStageStatus::from_db_value(status_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime stage status: {status_raw}"))?,
        capability: row.try_get_by_index(9)?,
        model_id: row.try_get_by_index(10)?,
        worker_id: row.try_get_by_index(11)?,
        lease_expires_at: opt_i64_to_u64(row.try_get_by_index(12)?)?,
        available_at: opt_i64_to_u64(row.try_get_by_index(13)?)?,
        attempt_token: row.try_get_by_index(14)?,
        attempt_count: i64_to_u32(row.try_get_by_index(15)?)?,
        max_attempts: i64_to_u32(row.try_get_by_index(16)?)?,
        input_artifact_ids: parse_string_array(row.try_get_by_index::<String>(17)?),
        output_artifact_ids: parse_string_array(row.try_get_by_index::<String>(18)?),
        progress_json: row
            .try_get_by_index::<Option<String>>(19)?
            .map(|raw| parse_json_value(raw, json!({}))),
        started_at: opt_i64_to_u64(row.try_get_by_index(20)?)?,
        finished_at: opt_i64_to_u64(row.try_get_by_index(21)?)?,
        error_code: row.try_get_by_index(22)?,
        error_message: row.try_get_by_index(23)?,
        cancellation_state: parse_cancellation_state(row.try_get_by_index(24)?)?,
    })
}

fn parse_cancellation_state(
    value: Option<String>,
) -> anyhow::Result<Option<RuntimeCancellationState>> {
    value
        .map(|value| {
            RuntimeCancellationState::from_db_value(&value)
                .ok_or_else(|| anyhow!("Unknown runtime cancellation state: {value}"))
        })
        .transpose()
}

fn map_stage_claim_candidate(row: &QueryResult) -> anyhow::Result<StageClaimCandidate> {
    let job_kind_raw: String = row.try_get_by_index(2)?;
    let queue_class_raw: String = row.try_get_by_index(5)?;

    Ok(StageClaimCandidate {
        stage_id: row.try_get_by_index(0)?,
        stage_kind: row.try_get_by_index(1)?,
        job_kind: RuntimeJobKind::from_db_value(job_kind_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime job kind: {job_kind_raw}"))?,
        queue_class: QueueClass::from_db_value(&queue_class_raw)
            .ok_or_else(|| anyhow!("Unknown runtime queue class: {queue_class_raw}"))?,
        resource_hints: parse_resource_hints(row.try_get_by_index(6)?),
        capability: row.try_get_by_index(3)?,
        model_id: row.try_get_by_index(4)?,
    })
}

fn map_runtime_artifact(row: &QueryResult) -> anyhow::Result<RuntimeArtifact> {
    let kind_raw: String = row.try_get_by_index(7)?;
    let role_raw: String = row.try_get_by_index(8)?;

    Ok(RuntimeArtifact {
        id: row.try_get_by_index(0)?,
        job_id: row.try_get_by_index(1)?,
        stage_id: row.try_get_by_index(2)?,
        producer_attempt_count: opt_i64_to_u32(row.try_get_by_index(3)?)?,
        producer_attempt_token: row.try_get_by_index(4)?,
        publication_key: row.try_get_by_index(5)?,
        created_at: i64_to_u64(row.try_get_by_index(6)?)?,
        artifact_kind: RuntimeArtifactKind::from_db_value(kind_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime artifact kind: {kind_raw}"))?,
        artifact_role: RuntimeArtifactRole::from_db_value(role_raw.as_str())
            .ok_or_else(|| anyhow!("Unknown runtime artifact role: {role_raw}"))?,
        media_asset_id: row.try_get_by_index(9)?,
        text_asset_id: row.try_get_by_index(10)?,
        storage_key: row.try_get_by_index(11)?,
        content_type: row.try_get_by_index(12)?,
        filename: row.try_get_by_index(13)?,
        size_bytes: opt_i64_to_u64(row.try_get_by_index(14)?)?,
        sha256: row.try_get_by_index(15)?,
        metadata_json: parse_json_value(row.try_get_by_index::<String>(16)?, json!({})),
        retention_policy: row.try_get_by_index(17)?,
    })
}

fn map_idempotency_record(row: &QueryResult) -> anyhow::Result<IdempotencyRecord> {
    Ok(IdempotencyRecord {
        operation: row.try_get_by_index(0)?,
        idempotency_key: row.try_get_by_index(1)?,
        created_at: i64_to_u64(row.try_get_by_index(2)?)?,
        expires_at: opt_i64_to_u64(row.try_get_by_index(3)?)?,
        request_hash: row.try_get_by_index(4)?,
        response_json: row
            .try_get_by_index::<Option<String>>(5)?
            .map(|raw| parse_json_value(raw, json!({}))),
        runtime_job_id: row.try_get_by_index(6)?,
        conflict_message: row.try_get_by_index(7)?,
        metadata_json: parse_json_value(row.try_get_by_index::<String>(8)?, json!({})),
    })
}

fn map_worker_heartbeat(row: &QueryResult) -> anyhow::Result<RuntimeWorkerHeartbeat> {
    let worker_id: String = row.try_get_by_index(0)?;
    let queue_names = parse_string_array(row.try_get_by_index::<String>(4)?);
    let stored_instance_id: String = row.try_get_by_index(5)?;
    let instance_id = if stored_instance_id.is_empty() {
        worker_id.clone()
    } else {
        stored_instance_id
    };
    let registration_version = i64_to_u32(row.try_get_by_index(6)?)? as u16;
    let registration =
        serde_json::from_str::<RuntimeWorkerRegistration>(&row.try_get_by_index::<String>(7)?)
            .unwrap_or_else(|_| RuntimeWorkerRegistration {
                version: registration_version,
                worker_id: worker_id.clone(),
                instance_id: instance_id.clone(),
                queue_classes: queue_names
                    .iter()
                    .filter_map(|queue| QueueClass::from_db_value(queue))
                    .collect(),
                capabilities: Vec::new(),
                model_ids: Vec::new(),
                stage_kinds: Vec::new(),
                resources: WorkerResourceCapacity::default(),
                software_version: "legacy".to_string(),
            });
    let heartbeat_version = i64_to_u32(row.try_get_by_index(8)?)? as u16;
    let available_slots = i64_to_u32(row.try_get_by_index(9)?)?;
    let details =
        serde_json::from_str::<RuntimeWorkerHeartbeatDetails>(&row.try_get_by_index::<String>(10)?)
            .unwrap_or_else(|_| RuntimeWorkerHeartbeatDetails {
                version: heartbeat_version,
                available_slots,
                active_lease_ids: Vec::new(),
                last_error: None,
                health_json: json!({}),
            });

    Ok(RuntimeWorkerHeartbeat {
        worker_id,
        started_at: i64_to_u64(row.try_get_by_index(1)?)?,
        last_heartbeat_at: i64_to_u64(row.try_get_by_index(2)?)?,
        status: row.try_get_by_index(3)?,
        queue_names,
        instance_id,
        registration,
        details,
        current_job_id: row.try_get_by_index(11)?,
        current_stage_id: row.try_get_by_index(12)?,
        diagnostic_json: parse_json_value(row.try_get_by_index::<String>(13)?, json!({})),
    })
}

fn worker_heartbeat_accepts_claims(heartbeat: &RuntimeWorkerHeartbeat) -> bool {
    matches!(heartbeat.status.as_str(), "polling" | "idle" | "running")
}

fn json_to_db_string(value: &serde_json::Value, fallback: &str) -> anyhow::Result<String> {
    serde_json::to_string(value)
        .or_else(|_| Ok::<String, serde_json::Error>(fallback.to_string()))
        .context("Failed to serialize runtime JSON payload")
}

pub(crate) fn validate_stage_output_artifact_retention_bounds(
    output_artifact_ids: &[String],
) -> anyhow::Result<()> {
    if output_artifact_ids.len() > MAX_STAGE_OUTPUT_ARTIFACTS {
        bail!("Stage output artifact count exceeds the limit of {MAX_STAGE_OUTPUT_ARTIFACTS}");
    }
    if output_artifact_ids
        .iter()
        .any(|artifact_id| artifact_id.len() > MAX_STAGE_OUTPUT_ARTIFACT_ID_BYTES)
    {
        bail!(
            "Stage output artifact identifier exceeds the {MAX_STAGE_OUTPUT_ARTIFACT_ID_BYTES}-byte limit"
        );
    }
    Ok(())
}

pub(crate) fn validate_stage_output_artifact_ids(
    output_artifact_ids: &[String],
) -> anyhow::Result<()> {
    validate_stage_output_artifact_retention_bounds(output_artifact_ids)?;
    let mut unique = std::collections::HashSet::with_capacity(output_artifact_ids.len());
    for artifact_id in output_artifact_ids {
        let parsed = uuid::Uuid::parse_str(artifact_id)
            .map_err(|_| anyhow!("Stage output artifact identifiers must be canonical UUIDs"))?;
        if parsed.hyphenated().to_string() != *artifact_id {
            bail!("Stage output artifact identifiers must be canonical UUIDs");
        }
        if !unique.insert(parsed) {
            bail!("Stage output artifact identifiers must be unique");
        }
    }
    Ok(())
}

fn parse_json_value(raw: String, fallback: serde_json::Value) -> serde_json::Value {
    serde_json::from_str(raw.as_str()).unwrap_or(fallback)
}

fn parse_string_array(raw: String) -> Vec<String> {
    serde_json::from_str::<Vec<String>>(raw.as_str()).unwrap_or_default()
}

fn parse_resource_hints(raw: String) -> StageResourceHints {
    serde_json::from_str::<StageResourceHints>(&raw).unwrap_or_default()
}

fn queue_class_for_stage_kind(stage_kind: &str) -> QueueClass {
    match stage_kind {
        "asr_transcribe" | "asr_infer" => QueueClass::BatchAsr,
        "tts_synthesize" | "tts_generate" => QueueClass::BatchTts,
        "diarization" | "diarization_segment" => QueueClass::Diarization,
        "export" | "encode" | "notify" => QueueClass::Export,
        "evaluation" | "evaluate" => QueueClass::Evaluation,
        _ => QueueClass::Batch,
    }
}

fn normalize_filter_values(values: &[String]) -> Vec<String> {
    values
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect()
}

fn push_claim_queue_clause(sql: &mut String, params: &mut Vec<Value>, queue_names: &[String]) {
    if queue_names.is_empty()
        || queue_names
            .iter()
            .any(|queue| queue == QueueClass::Batch.as_db_value())
    {
        return;
    }

    sql.push_str(" AND ");
    push_claim_in_expression(sql, params, "s.queue_class", queue_names);
}

fn push_claim_resource_clause(
    sql: &mut String,
    params: &mut Vec<Value>,
    resources: &WorkerResourceCapacity,
) {
    let targets = resources
        .targets
        .iter()
        .map(|target| target.as_db_value().to_string())
        .collect::<Vec<_>>();
    sql.push_str(" AND (s.resource_target = 'any'");
    if !targets.is_empty() {
        sql.push_str(" OR ");
        push_claim_in_expression(sql, params, "s.resource_target", &targets);
    }
    sql.push(')');

    let backends = resources
        .backends
        .iter()
        .map(|backend| backend.as_db_value().to_string())
        .collect::<Vec<_>>();
    push_optional_resource_requirement(sql, params, "s.required_backend", &backends);
    let device_classes = resources
        .device_classes
        .iter()
        .map(|device| device.as_db_value().to_string())
        .collect::<Vec<_>>();
    push_optional_resource_requirement(sql, params, "s.required_device_class", &device_classes);

    match resources.memory_bytes {
        Some(memory_bytes) => {
            let placeholder = params.len() + 1;
            sql.push_str(
                " AND (s.min_resource_memory_bytes IS NULL OR s.min_resource_memory_bytes <= ?",
            );
            sql.push_str(&placeholder.to_string());
            sql.push(')');
            params.push(u64_to_i64_value(memory_bytes).unwrap_or(Value::BigInt(Some(i64::MAX))));
        }
        None => sql.push_str(" AND s.min_resource_memory_bytes IS NULL"),
    }

    let placeholder = params.len() + 1;
    sql.push_str(" AND s.resource_concurrency_weight <= ?");
    sql.push_str(&placeholder.to_string());
    params.push(u32_to_i64_value(resources.concurrency_slots).into());
}

fn push_optional_resource_requirement(
    sql: &mut String,
    params: &mut Vec<Value>,
    expression: &str,
    values: &[String],
) {
    sql.push_str(" AND (");
    sql.push_str(expression);
    sql.push_str(" IS NULL");
    if !values.is_empty() {
        sql.push_str(" OR ");
        push_claim_in_expression(sql, params, expression, values);
    }
    sql.push(')');
}

fn push_claim_string_filter_clause(
    sql: &mut String,
    params: &mut Vec<Value>,
    expression: &str,
    values: &[String],
) {
    if values.is_empty() {
        return;
    }

    sql.push_str(" AND ");
    push_claim_in_expression(sql, params, expression, values);
}

fn push_claim_in_expression(
    sql: &mut String,
    params: &mut Vec<Value>,
    expression: &str,
    values: &[String],
) {
    sql.push_str(expression);
    sql.push_str(" IN (");
    for (idx, value) in values.iter().enumerate() {
        if idx > 0 {
            sql.push_str(", ");
        }
        let placeholder = params.len() + 1;
        sql.push('?');
        sql.push_str(placeholder.to_string().as_str());
        params.push(value.clone().into());
    }
    sql.push(')');
}

fn optional_filter_matches(filter: &[String], value: Option<&str>) -> bool {
    filter.is_empty()
        || value.is_some_and(|value| filter.iter().any(|entry| entry.as_str() == value))
}

fn opt_string(value: Option<String>) -> Value {
    Value::String(value)
}

fn opt_u64(value: Option<u64>) -> Value {
    Value::BigInt(value.and_then(|value| i64::try_from(value).ok()))
}

fn opt_i64(value: Option<i64>) -> Value {
    Value::BigInt(value)
}

fn opt_u32(value: Option<u32>) -> Value {
    Value::BigInt(value.map(i64::from))
}

fn opt_u16(value: Option<u16>) -> Value {
    Value::BigInt(value.map(i64::from))
}

fn opt_f64(value: Option<f64>) -> Value {
    Value::Double(value)
}

fn opt_f32(value: Option<f32>) -> Value {
    Value::Double(value.map(f64::from))
}

fn u64_to_i64_value(value: u64) -> anyhow::Result<Value> {
    Ok(Value::BigInt(Some(i64::try_from(value)?)))
}

fn u32_to_i64_value(value: u32) -> i64 {
    i64::from(value)
}

fn i64_to_u64(value: i64) -> anyhow::Result<u64> {
    u64::try_from(value).map_err(Into::into)
}

fn opt_i64_to_u64(value: Option<i64>) -> anyhow::Result<Option<u64>> {
    value.map(i64_to_u64).transpose()
}

fn i64_to_u32(value: i64) -> anyhow::Result<u32> {
    u32::try_from(value).map_err(Into::into)
}

fn opt_i64_to_u32(value: Option<i64>) -> anyhow::Result<Option<u32>> {
    value.map(i64_to_u32).transpose()
}

fn opt_i64_to_u16(value: Option<i64>) -> anyhow::Result<Option<u16>> {
    value
        .map(|value| u16::try_from(value).map_err(Into::into))
        .transpose()
}

fn i64_to_i32(value: i64) -> anyhow::Result<i32> {
    i32::try_from(value).map_err(Into::into)
}

fn is_terminal_job_status(status: RuntimeJobStatus) -> bool {
    matches!(
        status,
        RuntimeJobStatus::Completed
            | RuntimeJobStatus::Failed
            | RuntimeJobStatus::Cancelled
            | RuntimeJobStatus::Expired
    )
}

fn is_claimable_job_status(status: RuntimeJobStatus) -> bool {
    matches!(
        status,
        RuntimeJobStatus::Created
            | RuntimeJobStatus::Queued
            | RuntimeJobStatus::Running
            | RuntimeJobStatus::Retrying
            | RuntimeJobStatus::Postprocessing
    )
}

/// All transactions in this store write durable state. SQLite must acquire
/// its write reservation before reading a snapshot: DEFERRED promotion can
/// fail immediately with SQLITE_BUSY_SNAPSHOT when concurrent workers renew,
/// finish, or relinquish leases. Other backends ignore the SQLite option.
fn speech_admission_tenant(request: &serde_json::Value) -> anyhow::Result<String> {
    match request.get("tenant_key").filter(|value| !value.is_null()) {
        Some(value) => {
            let key: [u8; 32] = serde_json::from_value(value.clone())
                .context("Invalid server-authored speech tenant identity")?;
            Ok(key.iter().map(|byte| format!("{byte:02x}")).collect())
        }
        None => Ok("anonymous".into()),
    }
}

fn runtime_write_transaction_options() -> TransactionOptions {
    TransactionOptions {
        sqlite_transaction_mode: Some(SqliteTransactionMode::Immediate),
        ..TransactionOptions::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::StoreDatabase;
    use tempfile::TempDir;

    fn build_store() -> (BatchRuntimeStore, TempDir) {
        let root = tempfile::tempdir().expect("temp dir");
        let db_path = root.path().join("runtime.sqlite");
        (
            BatchRuntimeStore::initialize_with_database(StoreDatabase::new(db_path)),
            root,
        )
    }

    fn test_provider_write_input(
        namespace: MediaNamespace,
        storage_namespace: &str,
        lifetime_ms: u64,
    ) -> NewProviderWriteReservation {
        let write_id = new_uuid();
        let mut metadata = HookMetadata::new();
        metadata.insert("tenant_id".to_string(), "tenant-a".to_string());
        NewProviderWriteReservation {
            write_id: write_id.clone(),
            tenant_scope: "tenant-a".to_string(),
            storage_namespace: storage_namespace.to_string(),
            content_type: "application/octet-stream".to_string(),
            filename: Some("artifact.bin".to_string()),
            expected_size_bytes: 4,
            expected_sha256: sha256_hex(b"data"),
            lifetime_ms,
            provider_request: MediaWriteRequest {
                namespace,
                record_id: write_id,
                preferred_filename: Some("artifact.bin".to_string()),
                content_type: "application/octet-stream".to_string(),
                metadata,
            },
        }
    }

    #[test]
    fn provider_write_request_envelope_is_typed_and_bounded() {
        let mut input = test_provider_write_input(
            MediaNamespace::Other("artifact-store".to_string()),
            "artifact-store",
            1_000,
        );
        input
            .provider_request
            .metadata
            .insert("workflow_stage".to_string(), "finalize".to_string());
        validate_provider_write_input(&input).expect("valid exact provider request");
        let serialized = serialize_provider_write_request(&input.provider_request).unwrap();
        assert_eq!(
            deserialize_provider_write_request(&serialized).unwrap(),
            input.provider_request
        );
        assert!(deserialize_provider_write_request("{}").is_err());
        assert!(deserialize_provider_write_request(
            &"x".repeat(MAX_PROVIDER_WRITE_REQUEST_ENVELOPE_BYTES + 1)
        )
        .unwrap_err()
        .to_string()
        .contains("exceeds"));
        let unsupported = serialized.replacen("\"version\":1", "\"version\":2", 1);
        assert!(deserialize_provider_write_request(&unsupported)
            .unwrap_err()
            .to_string()
            .contains("version is unsupported"));
        let mut nested_unknown: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        nested_unknown["request"]["future_field"] = json!("must-not-be-dropped");
        assert!(
            deserialize_provider_write_request(&nested_unknown.to_string())
                .unwrap_err()
                .to_string()
                .contains("invalid")
        );

        let mut invalid = input.clone();
        invalid.write_id = invalid.write_id.to_ascii_uppercase();
        invalid.provider_request.record_id = invalid.write_id.clone();
        assert!(validate_provider_write_input(&invalid)
            .unwrap_err()
            .to_string()
            .contains("identity"));

        let mut invalid = input.clone();
        invalid.provider_request.record_id = new_uuid();
        assert!(validate_provider_write_input(&invalid)
            .unwrap_err()
            .to_string()
            .contains("record identity"));

        let mut invalid = input.clone();
        invalid.provider_request.namespace = MediaNamespace::Other("other".to_string());
        assert!(validate_provider_write_input(&invalid)
            .unwrap_err()
            .to_string()
            .contains("namespace does not match"));

        let mut invalid = input.clone();
        invalid.provider_request.namespace = MediaNamespace::Other(String::new());
        assert!(validate_provider_write_input(&invalid)
            .unwrap_err()
            .to_string()
            .contains("request namespace"));

        let mut invalid = input.clone();
        invalid.provider_request.content_type = "text/plain".to_string();
        assert!(validate_provider_write_input(&invalid)
            .unwrap_err()
            .to_string()
            .contains("does not match"));

        let mut invalid = input.clone();
        invalid.provider_request.metadata = (0..=MAX_PROVIDER_WRITE_METADATA_ENTRIES)
            .map(|index| (format!("key-{index}"), "value".to_string()))
            .collect();
        invalid
            .provider_request
            .metadata
            .insert("tenant_id".to_string(), "tenant-a".to_string());
        assert!(validate_provider_write_input(&invalid)
            .unwrap_err()
            .to_string()
            .contains("entry count"));

        let mut invalid = input.clone();
        invalid.provider_request.metadata.insert(
            "k".repeat(MAX_PROVIDER_WRITE_METADATA_KEY_BYTES + 1),
            "value".to_string(),
        );
        assert!(validate_provider_write_input(&invalid)
            .unwrap_err()
            .to_string()
            .contains("metadata key"));

        let mut invalid = input.clone();
        invalid.provider_request.metadata.insert(
            "large".to_string(),
            "v".repeat(MAX_PROVIDER_WRITE_METADATA_VALUE_BYTES + 1),
        );
        assert!(validate_provider_write_input(&invalid)
            .unwrap_err()
            .to_string()
            .contains("metadata value"));

        let mut invalid = input.clone();
        for index in 0..5 {
            invalid
                .provider_request
                .metadata
                .insert(format!("aggregate-{index}"), "v".repeat(900));
        }
        assert!(validate_provider_write_input(&invalid)
            .unwrap_err()
            .to_string()
            .contains("aggregate byte limit"));

        let mut invalid = input.clone();
        invalid.provider_request.namespace =
            MediaNamespace::Other("n".repeat(MAX_PROVIDER_WRITE_NAMESPACE_BYTES + 1));
        assert!(validate_provider_write_input(&invalid)
            .unwrap_err()
            .to_string()
            .contains("request namespace"));

        let mut invalid = input;
        invalid.provider_request.metadata.clear();
        invalid
            .provider_request
            .metadata
            .insert("tenant_id".to_string(), "tenant-a".to_string());
        for index in 0..4 {
            invalid
                .provider_request
                .metadata
                .insert(format!("escaped-{index}"), "\\\"".repeat(500));
        }
        assert!(serialize_provider_write_request(&invalid.provider_request)
            .unwrap_err()
            .to_string()
            .contains("envelope exceeds"));
    }

    #[tokio::test]
    async fn provider_write_request_roundtrips_and_legacy_recovery_fails_closed() {
        let (mut store, _root) = build_store();
        let clock = Arc::new(AtomicI64::new(1_000));
        store.set_test_clock(clock.clone());

        let mut exact =
            test_provider_write_input(MediaNamespace::GeneratedSpeech, "generated_speech", 10);
        exact
            .provider_request
            .metadata
            .insert("workflow_stage".to_string(), "finalize".to_string());
        let exact = store.reserve_provider_write(exact).await.unwrap();
        assert_eq!(
            exact.provider_request.namespace,
            MediaNamespace::GeneratedSpeech
        );
        assert_eq!(
            exact
                .provider_request
                .metadata
                .get("workflow_stage")
                .map(String::as_str),
            Some("finalize")
        );

        let legacy = store
            .reserve_provider_write(test_provider_write_input(
                MediaNamespace::Other("artifact-store".to_string()),
                "artifact-store",
                10,
            ))
            .await
            .unwrap();
        let db = store.connection().await.unwrap();
        db.execute_raw(
            raw::statement(
                db,
                "UPDATE provider_write_reservations SET provider_request_json = NULL WHERE write_id IN (?1, ?2)",
                vec![exact.write_id.clone().into(), legacy.write_id.clone().into()],
            )
            .unwrap(),
        )
        .await
        .unwrap();

        clock.store(1_010, Ordering::SeqCst);
        let claimed = store.claim_due_provider_write_cleanup(64).await.unwrap();
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].write_id, legacy.write_id);
        assert_eq!(
            claimed[0].provider_request,
            legacy_provider_write_request(
                &legacy.write_id,
                &legacy.tenant_scope,
                &legacy.storage_namespace,
                &legacy.content_type,
                legacy.filename.as_deref(),
            )
            .unwrap()
        );
        let retained = db
            .query_one_raw(
                raw::statement(
                    db,
                    "SELECT COUNT(*) FROM provider_write_reservations WHERE write_id = ?1 AND provider_request_json IS NULL",
                    vec![exact.write_id.into()],
                )
                .unwrap(),
            )
            .await
            .unwrap()
            .unwrap()
            .try_get_by_index::<i64>(0)
            .unwrap();
        assert_eq!(retained, 1);
    }

    #[tokio::test]
    async fn invalid_provider_write_rows_remain_fenced_without_starving_cleanup() {
        let (mut store, _root) = build_store();
        let clock = Arc::new(AtomicI64::new(1_000));
        store.set_test_clock(clock.clone());
        let poison = store
            .reserve_provider_write(test_provider_write_input(
                MediaNamespace::Other("artifact-store".to_string()),
                "artifact-store",
                10,
            ))
            .await
            .unwrap();
        let oversized = store
            .reserve_provider_write(test_provider_write_input(
                MediaNamespace::GeneratedSpeech,
                "generated_speech",
                10,
            ))
            .await
            .unwrap();
        let valid = store
            .reserve_provider_write(test_provider_write_input(
                MediaNamespace::Other("artifact-store".to_string()),
                "artifact-store",
                10,
            ))
            .await
            .unwrap();
        let db = store.connection().await.unwrap();
        db.execute_raw(
            raw::statement(
                db,
                "UPDATE provider_write_reservations SET provider_request_json = ?1, created_at = 0 WHERE write_id = ?2",
                vec![
                    "{\"version\":1,\"request\":{\"future_field\":true}}"
                        .into(),
                    poison.write_id.clone().into(),
                ],
            )
            .unwrap(),
        )
        .await
        .unwrap();
        db.execute_raw(
            raw::statement(
                db,
                "UPDATE provider_write_reservations SET provider_request_json = ?1, created_at = -1 WHERE write_id = ?2",
                vec![
                    "x"
                        .repeat(MAX_PROVIDER_WRITE_REQUEST_ENVELOPE_BYTES + 1)
                        .into(),
                    oversized.write_id.clone().into(),
                ],
            )
            .unwrap(),
        )
        .await
        .unwrap();

        clock.store(1_010, Ordering::SeqCst);
        let claimed = store.claim_due_provider_write_cleanup(64).await.unwrap();
        assert_eq!(claimed.len(), 1);
        assert_eq!(claimed[0].write_id, valid.write_id);
        assert!(get_provider_write_with(db, &oversized.write_id)
            .await
            .unwrap()
            .is_none());

        let quarantined = db
            .query_one_raw(
                raw::statement(
                    db,
                    "SELECT state, available_at, cleanup_attempt_count, last_error FROM provider_write_reservations WHERE write_id = ?1",
                    vec![poison.write_id.into()],
                )
                .unwrap(),
            )
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            quarantined.try_get_by_index::<String>(0).unwrap(),
            "cleanup_pending"
        );
        assert!(quarantined.try_get_by_index::<i64>(1).unwrap() > 1_010);
        assert_eq!(quarantined.try_get_by_index::<i64>(2).unwrap(), 1);
        assert!(quarantined
            .try_get_by_index::<Option<String>>(3)
            .unwrap()
            .is_some());

        let oversized_retained = db
            .query_one_raw(
                raw::statement(
                    db,
                    "SELECT COUNT(*) FROM provider_write_reservations WHERE write_id = ?1 AND state = 'reserved'",
                    vec![oversized.write_id.into()],
                )
                .unwrap(),
            )
            .await
            .unwrap()
            .unwrap()
            .try_get_by_index::<i64>(0)
            .unwrap();
        assert_eq!(oversized_retained, 1);
    }

    fn durable_idempotency_request(
        tenant_scope: &str,
        operation: &str,
        idempotency_key: &str,
        payload: &[u8],
        reservation_ttl_ms: u64,
    ) -> DurableIdempotencyRequest {
        DurableIdempotencyRequest {
            tenant_scope: tenant_scope.to_string(),
            operation: operation.to_string(),
            idempotency_key: idempotency_key.to_string(),
            digest_version: DURABLE_IDEMPOTENCY_DIGEST_VERSION,
            request_digest: sha256_hex(payload),
            reservation_ttl_ms,
        }
    }

    fn durable_text_tts_acceptance(
        reservation: Option<DurableIdempotencyReservation>,
    ) -> NewDurableTextTtsAcceptance {
        NewDurableTextTtsAcceptance {
            projection: NewSpeechHistoryRecord {
                route_kind: SpeechRouteKind::TextToSpeech,
                processing_status: SpeechHistoryProcessingStatus::Pending,
                processing_error: None,
                model_id: Some("Kokoro-82M".to_string()),
                speaker: Some("af_heart".to_string()),
                language: Some("en".to_string()),
                saved_voice_id: None,
                speed: Some(1.0),
                input_text: "hello durable world".to_string(),
                voice_description: None,
                reference_text: None,
                generation_time_ms: 0.0,
                audio_duration_secs: None,
                rtf: None,
                tokens_generated: None,
                audio_mime_type: "audio/wav".to_string(),
                audio_filename: Some("speech.wav".to_string()),
                audio_bytes: Vec::new(),
            },
            request_json: json!({
                "tenant_key": null,
                "route_kind": "text_to_speech",
                "model_id": "Kokoro-82M",
                "input_text": "hello durable world",
                "request": {"speaker": "af_heart", "speed": 1.0}
            }),
            model_snapshot_json: json!({"version": 1, "model_id": "Kokoro-82M"}),
            retry_policy_json: json!({"max_attempts": 2}),
            priority: 0,
            max_attempts: 2,
            correlation_id: Some("durable-test-request".to_string()),
            stage_kind: "tts_synthesize".to_string(),
            queue_class: QueueClass::BatchTts,
            resource_hints: StageResourceHints::default(),
            reservation,
            idempotency_retention_ms: 60_000,
        }
    }

    async fn durable_text_tts_row_counts(store: &BatchRuntimeStore) -> [u64; 6] {
        let db = store.connection().await.expect("database");
        let mut counts = [0_u64; 6];
        for (index, table) in [
            "speech_history_records",
            "text_assets",
            "runtime_jobs",
            "runtime_artifacts",
            "job_stages",
            "durable_idempotency_keys_v2",
        ]
        .into_iter()
        .enumerate()
        {
            let row = db
                .query_one_raw(
                    raw::statement(db, format!("SELECT COUNT(*) FROM {table}"), vec![])
                        .expect("count statement"),
                )
                .await
                .expect("count query")
                .expect("count row");
            counts[index] = u64::try_from(
                row.try_get_by_index::<i64>(0)
                    .expect("nonnegative row count"),
            )
            .expect("nonnegative row count");
        }
        counts
    }

    #[test]
    fn durable_idempotency_digest_uses_canonical_json_object_order() {
        let first: serde_json::Value = serde_json::from_str(
            r#"{"text":"hello","options":{"voice":"a","speed":1},"parts":[1,2]}"#,
        )
        .expect("first request");
        let reordered: serde_json::Value = serde_json::from_str(
            r#"{"parts":[1,2],"options":{"speed":1,"voice":"a"},"text":"hello"}"#,
        )
        .expect("reordered request");
        let changed: serde_json::Value = serde_json::from_str(
            r#"{"parts":[2,1],"options":{"speed":1,"voice":"a"},"text":"hello"}"#,
        )
        .expect("changed request");

        assert_eq!(
            canonical_request_digest(&first).expect("first digest"),
            canonical_request_digest(&reordered).expect("reordered digest")
        );
        assert_ne!(
            canonical_request_digest(&first).expect("first digest"),
            canonical_request_digest(&changed).expect("changed digest")
        );
        assert!(canonical_request_digest(&json!(
            "x".repeat(MAX_DURABLE_IDEMPOTENCY_CANONICAL_REQUEST_BYTES + 1)
        ))
        .is_err());

        let mut too_deep = serde_json::Value::Null;
        for _ in 0..=MAX_DURABLE_IDEMPOTENCY_JSON_DEPTH {
            too_deep = serde_json::Value::Array(vec![too_deep]);
        }
        assert!(canonical_request_digest(&too_deep)
            .unwrap_err()
            .to_string()
            .contains("maximum nesting depth"));
    }

    async fn create_test_job_and_stage(
        store: &BatchRuntimeStore,
        priority: i32,
        stage_kind: &str,
        max_attempts: u32,
    ) -> (RuntimeJob, JobStage) {
        let job = create_test_job(store, priority, max_attempts).await;
        let stage = store
            .create_stage(NewJobStage {
                job_id: job.id.clone(),
                sequence: 0,
                stage_kind: stage_kind.to_string(),
                status: RuntimeStageStatus::Queued,
                capability: Some("test".to_string()),
                model_id: None,
                max_attempts,
                input_artifact_ids: vec![],
            })
            .await
            .expect("test stage");
        (job, stage)
    }

    async fn create_test_job(
        store: &BatchRuntimeStore,
        priority: i32,
        max_attempts: u32,
    ) -> RuntimeJob {
        let job = store
            .create_job(NewRuntimeJob {
                job_kind: RuntimeJobKind::TtsSpeech,
                status: RuntimeJobStatus::Queued,
                priority,
                model_id: None,
                capability: Some("test".to_string()),
                route_record_kind: Some("test".to_string()),
                route_record_id: None,
                input_media_asset_id: None,
                input_text_asset_id: None,
                request_json: json!({}),
                model_snapshot_json: json!({}),
                retry_policy_json: json!({"max_attempts": max_attempts}),
                max_attempts,
                idempotency_key: None,
                correlation_id: None,
            })
            .await
            .expect("test job");
        job
    }

    #[tokio::test]
    async fn cooperative_yield_frees_worker_for_short_job_without_consuming_retries() {
        let (mut store, _root) = build_store();
        let clock = Arc::new(AtomicI64::new(1_000));
        store.set_test_clock(clock);
        let (_long_job, long_stage) = create_test_job_and_stage(&store, 0, "speech", 1).await;
        let first = store
            .claim_next_stage("worker", 60_000)
            .await
            .unwrap()
            .unwrap();
        let old = first.lease().unwrap();
        let checkpoint = json!({"completed_segments": 1, "completed_text_bytes": 480});
        assert!(store
            .update_stage_progress(&old, checkpoint.clone())
            .await
            .unwrap());
        let (_short_job, short_stage) = create_test_job_and_stage(&store, 0, "speech", 1).await;
        assert!(store.yield_stage(&old).await.unwrap());
        assert!(!store.yield_stage(&old).await.unwrap());
        let next = store
            .claim_next_stage("worker", 60_000)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            next.stage.id, short_stage.id,
            "same-millisecond arrivals must outrank the yielded job"
        );
        store
            .complete_stage(&next.lease().unwrap(), vec![])
            .await
            .unwrap()
            .unwrap();
        let mut current = store
            .claim_next_stage("worker", 60_000)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(current.stage.id, long_stage.id);
        assert_eq!(current.stage.progress_json, Some(checkpoint));
        assert_eq!(current.stage.attempt_count, 1);
        assert_ne!(current.stage.attempt_token, old.attempt_token);
        assert!(!store
            .update_stage_progress(&old, json!({"stale": true}))
            .await
            .unwrap());
        assert!(store.complete_stage(&old, vec![]).await.unwrap().is_none());
        assert!(store
            .publish_stage_output_artifact(&old, test_stage_output("stale"))
            .await
            .unwrap()
            .is_none());
        // A one-attempt job may cooperatively continue many times and still finish.
        for _ in 0..8 {
            let lease = current.lease().unwrap();
            assert!(store.yield_stage(&lease).await.unwrap());
            current = store
                .claim_next_stage("worker", 60_000)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(current.stage.attempt_count, 1);
            assert_ne!(current.stage.attempt_token, lease.attempt_token);
        }
        store
            .complete_stage(&current.lease().unwrap(), vec![])
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn cooperative_yield_rejects_foreign_expired_and_cancelled_attempts() {
        let (mut store, _root) = build_store();
        let clock = Arc::new(AtomicI64::new(1_000));
        store.set_test_clock(clock.clone());
        let (job, _stage) = create_test_job_and_stage(&store, 0, "speech", 2).await;
        let claim = store
            .claim_next_stage("worker", 100)
            .await
            .unwrap()
            .unwrap();
        let lease = claim.lease().unwrap();
        let foreign = StageLease {
            attempt_token: Some("foreign".into()),
            ..lease.clone()
        };
        assert!(!store.yield_stage(&foreign).await.unwrap());
        clock.store(1_101, Ordering::SeqCst);
        assert!(!store.yield_stage(&lease).await.unwrap());
        clock.store(1_001, Ordering::SeqCst);
        store
            .cancel_job(&job.id, Some("user stop".into()))
            .await
            .unwrap();
        assert!(!store.yield_stage(&lease).await.unwrap());
    }

    #[tokio::test]
    async fn speech_replay_gc_fences_retry_before_storage_deletion() {
        for retry_first in [false, true] {
            let (mut store, _root) = build_store();
            let clock = Arc::new(AtomicI64::new(1_000));
            store.set_test_clock(clock.clone());
            let (job, _) = create_test_job_and_stage(&store, 0, "speech", 3).await;
            let claim = store
                .claim_next_stage("worker", 60_000)
                .await
                .unwrap()
                .unwrap();
            let lease = claim.lease().unwrap();
            let artifact = store
                .publish_stage_output_artifact(
                    &lease,
                    test_stage_output("speech-pcm/00000000000000000000"),
                )
                .await
                .unwrap()
                .unwrap();
            store
                .fail_stage(
                    &lease,
                    false,
                    Some("injected".into()),
                    Some("original failure".into()),
                )
                .await
                .unwrap()
                .unwrap();
            clock.store(10_000, Ordering::SeqCst);
            if retry_first {
                assert!(store.retry_job(&job.id).await.unwrap().is_some());
                assert!(store.expired_speech_pcm(5_000).await.unwrap().is_empty());
            } else {
                let expired = store.expired_speech_pcm(5_000).await.unwrap();
                assert_eq!(expired.len(), 1);
                assert_eq!(expired[0].id, artifact.id);
                assert!(store.retry_job(&job.id).await.unwrap().is_none());
                let job = store.get_job(&job.id).await.unwrap().unwrap();
                assert_eq!(job.error_code.as_deref(), Some("speech_replay_expired"));
                assert_eq!(job.error_message.as_deref(), Some("original failure"));
                // Interrupted deletion remains discoverable and cannot re-enable retry.
                assert_eq!(store.expired_speech_pcm(5_000).await.unwrap().len(), 1);
                store
                    .remove_speech_pcm_artifact(&artifact.id)
                    .await
                    .unwrap();
                assert!(store.expired_speech_pcm(5_000).await.unwrap().is_empty());
            }
        }
    }

    #[tokio::test]
    async fn manual_speech_replay_deletion_fences_concurrent_retry() {
        for retry_first in [false, true] {
            let (store, _root) = build_store();
            let (job, _) = create_test_job_and_stage(&store, 0, "speech", 3).await;
            let claim = store
                .claim_next_stage("worker", 60_000)
                .await
                .unwrap()
                .unwrap();
            let artifact = store
                .publish_stage_output_artifact(
                    &claim.lease().unwrap(),
                    test_stage_output("speech-pcm/00000000000000000000"),
                )
                .await
                .unwrap()
                .unwrap();
            let lease = claim.lease().unwrap();
            store
                .cancel_job(&job.id, Some("delete recording".into()))
                .await
                .unwrap()
                .unwrap();
            store
                .finalize_stage_cancellation(&lease)
                .await
                .unwrap()
                .unwrap();
            if retry_first {
                assert!(store.retry_job(&job.id).await.unwrap().is_some());
                assert!(!store.fence_speech_replay_deletion(&job.id).await.unwrap());
                assert_eq!(
                    store.speech_pcm_after(&job.id, None, 1).await.unwrap()[0].id,
                    artifact.id
                );
            } else {
                assert!(store.fence_speech_replay_deletion(&job.id).await.unwrap());
                assert!(store.retry_job(&job.id).await.unwrap().is_none());
                // Retrying an interrupted deletion keeps the fence in force.
                assert!(store.fence_speech_replay_deletion(&job.id).await.unwrap());
                store
                    .remove_speech_pcm_artifact(&artifact.id)
                    .await
                    .unwrap();
                assert!(store
                    .speech_pcm_after(&job.id, None, 1)
                    .await
                    .unwrap()
                    .is_empty());
            }
        }
    }

    fn admission_job(tenant: u8) -> NewRuntimeJob {
        NewRuntimeJob {
            job_kind: RuntimeJobKind::TtsSpeech,
            status: RuntimeJobStatus::Queued,
            priority: 0,
            model_id: Some("FishAudio-S2-Pro".into()),
            capability: Some("tts".into()),
            route_record_kind: Some("text_to_speech".into()),
            route_record_id: None,
            input_media_asset_id: None,
            input_text_asset_id: None,
            request_json: json!({"tenant_key": vec![tenant; 32]}),
            model_snapshot_json: json!({}),
            retry_policy_json: json!({"max_attempts": 2}),
            max_attempts: 2,
            idempotency_key: None,
            correlation_id: None,
        }
    }

    #[tokio::test]
    async fn speech_admission_preflight_has_no_reservation_and_rejects_overload() {
        let (mut store, _root) = build_store();
        store.test_tts_admission_limits = Some((1, 1));
        store
            .preflight_speech_admission(Some([1; 32]))
            .await
            .unwrap();
        store
            .preflight_speech_admission(Some([1; 32]))
            .await
            .unwrap();
        store.create_job(admission_job(1)).await.unwrap();
        assert!(store
            .preflight_speech_admission(Some([1; 32]))
            .await
            .unwrap_err()
            .to_string()
            .contains("Speech job admission capacity exhausted"));
    }

    #[tokio::test]
    async fn speech_admission_serializes_concurrent_enqueue_and_terminal_releases_capacity() {
        let (mut store, _root) = build_store();
        store.test_tts_admission_limits = Some((2, 2));
        store.connection().await.unwrap();
        let mut tasks = Vec::new();
        for tenant in 0..8 {
            let store = store.clone();
            tasks.push(tokio::spawn(async move {
                store.create_job(admission_job(tenant)).await
            }));
        }
        let mut admitted = Vec::new();
        let mut rejected = 0;
        for task in tasks {
            match task.await.unwrap() {
                Ok(job) => admitted.push(job),
                Err(error) => {
                    assert!(
                        error
                            .to_string()
                            .contains("Speech job admission capacity exhausted"),
                        "{error}"
                    );
                    rejected += 1;
                }
            }
        }
        assert_eq!(admitted.len(), 2);
        assert_eq!(rejected, 6);
        store
            .cancel_job(&admitted[0].id, Some("release slot".into()))
            .await
            .unwrap()
            .unwrap();
        store.create_job(admission_job(99)).await.unwrap();
        assert!(store.create_job(admission_job(100)).await.is_err());
    }

    #[tokio::test]
    async fn speech_admission_separates_tenants_and_retry_obeys_the_same_capacity() {
        let (mut store, _root) = build_store();
        store.test_tts_admission_limits = Some((3, 1));
        let first = store.create_job(admission_job(1)).await.unwrap();
        assert!(store.create_job(admission_job(1)).await.is_err());
        let other = store.create_job(admission_job(2)).await.unwrap();
        store.cancel_job(&other.id, None).await.unwrap();
        store
            .create_stage(NewJobStage {
                job_id: first.id.clone(),
                sequence: 0,
                stage_kind: "speech".into(),
                status: RuntimeStageStatus::Queued,
                capability: Some("tts".into()),
                model_id: None,
                max_attempts: 2,
                input_artifact_ids: vec![],
            })
            .await
            .unwrap();
        let claim = store
            .claim_next_stage("worker", 60_000)
            .await
            .unwrap()
            .unwrap();
        store
            .fail_stage(
                &claim.lease().unwrap(),
                false,
                Some("injected".into()),
                None,
            )
            .await
            .unwrap()
            .unwrap();
        let replacement = store.create_job(admission_job(1)).await.unwrap();
        let error = store.retry_job(&first.id).await.unwrap_err();
        assert!(error
            .to_string()
            .contains("Speech job admission capacity exhausted"));
        assert_eq!(
            store.get_job(&first.id).await.unwrap().unwrap().status,
            RuntimeJobStatus::Failed
        );
        store.cancel_job(&replacement.id, None).await.unwrap();
        assert!(store.retry_job(&first.id).await.unwrap().is_some());
    }

    fn test_stage_output(publication_key: &str) -> NewStageOutputArtifact {
        NewStageOutputArtifact {
            publication_key: publication_key.to_string(),
            artifact_kind: RuntimeArtifactKind::Metadata,
            artifact_role: RuntimeArtifactRole::OutputPrimary,
            media_asset_id: None,
            text_asset_id: None,
            storage_key: Some(format!("outputs/{publication_key}.json")),
            content_type: Some("application/json".to_string()),
            filename: Some(format!("{publication_key}.json")),
            size_bytes: Some(2),
            sha256: Some(sha256_hex(b"{}")),
            metadata_json: json!({"publication_key": publication_key}),
            retention_policy: "default".to_string(),
        }
    }

    #[tokio::test]
    async fn creates_runtime_foundation_records() {
        let (store, _root) = build_store();

        let media = store
            .create_media_asset(NewMediaAsset {
                asset_kind: "audio_original".to_string(),
                storage_namespace: "uploads".to_string(),
                storage_key: "uploads/transcription/test.wav".to_string(),
                content_type: "audio/wav".to_string(),
                filename: Some("test.wav".to_string()),
                size_bytes: 4,
                sha256: Some(sha256_hex(&[1, 2, 3, 4])),
                duration_secs: Some(1.25),
                sample_rate_hz: Some(16_000),
                channel_count: Some(1),
                peak_amplitude: Some(0.5),
                rms_amplitude: Some(0.1),
                source_asset_id: None,
                canonical_profile_version: None,
                scan_status: "passed".to_string(),
                retention_policy: "default".to_string(),
                metadata_json: json!({"source": "test"}),
            })
            .await
            .expect("media asset");

        let text = store
            .create_text_asset(NewTextAsset {
                raw_text: "Hello world".to_string(),
                normalized_text: None,
                language_hint: Some("en".to_string()),
                sha256: Some(sha256_hex(b"Hello world")),
                safety_status: "allowed".to_string(),
                retention_policy: "default".to_string(),
                structure_json: json!({"kind": "plain"}),
            })
            .await
            .expect("text asset");

        let job = store
            .create_job(NewRuntimeJob {
                job_kind: RuntimeJobKind::AsrTranscription,
                status: RuntimeJobStatus::Queued,
                priority: 5,
                model_id: Some("Granite-Speech-4.1-2B".to_string()),
                capability: Some("asr".to_string()),
                route_record_kind: Some("transcription".to_string()),
                route_record_id: Some("route-1".to_string()),
                input_media_asset_id: Some(media.id.clone()),
                input_text_asset_id: Some(text.id.clone()),
                request_json: json!({"language": "en"}),
                model_snapshot_json: json!({"license": "apache-2.0"}),
                retry_policy_json: json!({"max_attempts": 2}),
                max_attempts: 2,
                idempotency_key: Some("idem-1".to_string()),
                correlation_id: Some("corr-1".to_string()),
            })
            .await
            .expect("job");

        assert_eq!(job.status, RuntimeJobStatus::Queued);
        assert_eq!(job.queued_at, Some(job.created_at));
        assert_eq!(job.input_media_asset_id.as_deref(), Some(media.id.as_str()));

        let stage = store
            .create_stage(NewJobStage {
                job_id: job.id.clone(),
                sequence: 10,
                stage_kind: "asr_infer".to_string(),
                status: RuntimeStageStatus::Queued,
                capability: Some("asr".to_string()),
                model_id: job.model_id.clone(),
                max_attempts: 2,
                input_artifact_ids: vec![],
            })
            .await
            .expect("stage");

        let artifact = store
            .create_artifact(NewRuntimeArtifact {
                job_id: job.id.clone(),
                stage_id: Some(stage.id.clone()),
                artifact_kind: RuntimeArtifactKind::Transcript,
                artifact_role: RuntimeArtifactRole::OutputPrimary,
                media_asset_id: None,
                text_asset_id: Some(text.id.clone()),
                storage_key: None,
                content_type: Some("application/json".to_string()),
                filename: Some("transcript.json".to_string()),
                size_bytes: Some(128),
                sha256: None,
                metadata_json: json!({"format": "segments"}),
                retention_policy: "default".to_string(),
            })
            .await
            .expect("artifact");

        let idempotency = store
            .record_idempotency(NewIdempotencyRecord {
                operation: "job.create".to_string(),
                idempotency_key: "idem-1".to_string(),
                expires_at: None,
                request_hash: sha256_hex(br#"{"language":"en"}"#),
                response_json: Some(json!({"job_id": job.id})),
                runtime_job_id: Some(job.id.clone()),
                conflict_message: None,
                metadata_json: json!({}),
            })
            .await
            .expect("idempotency");

        let heartbeat = store
            .upsert_worker_heartbeat(WorkerHeartbeatUpdate {
                worker_id: "worker-1".to_string(),
                status: "idle".to_string(),
                queue_names: vec!["batch".to_string()],
                current_job_id: None,
                current_stage_id: None,
                diagnostic_json: json!({"pid": 123}),
            })
            .await
            .expect("heartbeat");

        assert_eq!(artifact.stage_id.as_deref(), Some(stage.id.as_str()));
        assert_eq!(artifact.producer_attempt_count, None);
        assert_eq!(artifact.producer_attempt_token, None);
        assert_eq!(artifact.publication_key, None);
        assert_eq!(idempotency.runtime_job_id.as_deref(), Some(job.id.as_str()));
        assert_eq!(heartbeat.queue_names, vec!["batch"]);
        assert_eq!(heartbeat.instance_id, "worker-1");
        assert_eq!(
            heartbeat.registration.queue_classes,
            vec![QueueClass::Batch]
        );
    }

    #[tokio::test]
    async fn durable_idempotency_is_tenant_and_operation_scoped_with_replay() {
        let (mut store, _root) = build_store();
        let clock = Arc::new(AtomicI64::new(1_000));
        store.set_test_clock(clock);
        let job = create_test_job(&store, 0, 1).await;
        let request = durable_idempotency_request("tenant-a", "job.create", "key-1", b"one", 500);
        let reservation = match store
            .reserve_durable_idempotency(request.clone())
            .await
            .expect("reserve")
        {
            DurableIdempotencyBegin::Acquired(reservation) => reservation,
            outcome => panic!("unexpected reservation outcome: {outcome:?}"),
        };
        let committed = store
            .commit_durable_idempotency(&reservation, &job.id, json!({"job_id": job.id}), 2_000)
            .await
            .expect("commit")
            .expect("owned reservation");
        assert_eq!(committed.runtime_job_id, job.id);
        assert_eq!(
            store
                .commit_durable_idempotency(
                    &reservation,
                    &job.id,
                    json!({"job_id": job.id}),
                    2_000,
                )
                .await
                .expect("idempotent commit"),
            Some(committed)
        );

        assert!(matches!(
            store
                .reserve_durable_idempotency(request.clone())
                .await
                .expect("replay"),
            DurableIdempotencyBegin::Replay(replay)
                if replay.runtime_job_id == job.id
                    && replay.response_json == json!({"job_id": job.id})
        ));
        assert!(matches!(
            store
                .reserve_durable_idempotency(DurableIdempotencyRequest {
                    request_digest: sha256_hex(b"different"),
                    ..request.clone()
                })
                .await
                .expect("conflict"),
            DurableIdempotencyBegin::Conflict
        ));
        assert!(matches!(
            store
                .reserve_durable_idempotency(DurableIdempotencyRequest {
                    tenant_scope: "tenant-b".to_string(),
                    ..request.clone()
                })
                .await
                .expect("other tenant"),
            DurableIdempotencyBegin::Acquired(_)
        ));
        assert!(matches!(
            store
                .reserve_durable_idempotency(DurableIdempotencyRequest {
                    operation: "job.export".to_string(),
                    ..request
                })
                .await
                .expect("other operation"),
            DurableIdempotencyBegin::Acquired(_)
        ));
    }

    #[tokio::test]
    async fn durable_idempotency_reservations_expire_and_fence_stale_commits() {
        let (mut store, _root) = build_store();
        let clock = Arc::new(AtomicI64::new(1_000));
        store.set_test_clock(clock.clone());
        let job = create_test_job(&store, 0, 1).await;
        let request = durable_idempotency_request("tenant-a", "job.create", "key-1", b"one", 100);
        let first = match store
            .reserve_durable_idempotency(request.clone())
            .await
            .expect("first reserve")
        {
            DurableIdempotencyBegin::Acquired(reservation) => reservation,
            outcome => panic!("unexpected reservation outcome: {outcome:?}"),
        };
        assert!(matches!(
            store
                .reserve_durable_idempotency(request.clone())
                .await
                .expect("in progress"),
            DurableIdempotencyBegin::InProgress { expires_at: 1_100 }
        ));

        clock.store(1_100, Ordering::SeqCst);
        let second = match store
            .reserve_durable_idempotency(request.clone())
            .await
            .expect("reserve after expiry")
        {
            DurableIdempotencyBegin::Acquired(reservation) => reservation,
            outcome => panic!("unexpected reservation outcome: {outcome:?}"),
        };
        assert_ne!(second.reservation_token, first.reservation_token);
        assert!(store
            .commit_durable_idempotency(&first, &job.id, json!({"job_id": job.id}), 100)
            .await
            .expect("stale commit")
            .is_none());
        store
            .commit_durable_idempotency(&second, &job.id, json!({"job_id": job.id}), 100)
            .await
            .expect("current commit")
            .expect("current owner");

        clock.store(1_200, Ordering::SeqCst);
        assert!(matches!(
            store
                .reserve_durable_idempotency(request)
                .await
                .expect("reserve after result expiry"),
            DurableIdempotencyBegin::Acquired(_)
        ));
    }

    #[tokio::test]
    async fn durable_idempotency_concurrent_reservation_has_one_owner_and_fenced_release() {
        let (mut store, _root) = build_store();
        store.set_test_clock(Arc::new(AtomicI64::new(1_000)));
        let request =
            durable_idempotency_request("tenant-a", "job.create", "concurrent-key", b"one", 500);
        let (first, second) = tokio::join!(
            store.reserve_durable_idempotency(request.clone()),
            store.reserve_durable_idempotency(request.clone()),
        );
        let outcomes = [
            first.expect("first reserve"),
            second.expect("second reserve"),
        ];
        assert_eq!(
            outcomes
                .iter()
                .filter(|outcome| matches!(outcome, DurableIdempotencyBegin::Acquired(_)))
                .count(),
            1
        );
        assert_eq!(
            outcomes
                .iter()
                .filter(|outcome| matches!(outcome, DurableIdempotencyBegin::InProgress { .. }))
                .count(),
            1
        );
        let reservation = outcomes
            .into_iter()
            .find_map(|outcome| match outcome {
                DurableIdempotencyBegin::Acquired(reservation) => Some(reservation),
                _ => None,
            })
            .expect("reservation owner");
        let mut forged = reservation.clone();
        forged.reservation_token = new_uuid();
        assert!(!store
            .release_durable_idempotency(&forged)
            .await
            .expect("forged release"));
        assert!(store
            .release_durable_idempotency(&reservation)
            .await
            .expect("owner release"));
        assert!(matches!(
            store
                .reserve_durable_idempotency(request)
                .await
                .expect("reserve after release"),
            DurableIdempotencyBegin::Acquired(_)
        ));
    }

    #[tokio::test]
    async fn durable_idempotency_pruning_and_capacity_are_bounded_and_deterministic() {
        let (mut store, _root) = build_store();
        let clock = Arc::new(AtomicI64::new(1_000));
        store.set_test_clock(clock.clone());
        for (offset, key) in [(0, "a"), (1, "b"), (2, "c")] {
            clock.store(1_000 + offset, Ordering::SeqCst);
            assert!(matches!(
                store
                    .reserve_durable_idempotency(durable_idempotency_request(
                        "tenant-a",
                        "job.create",
                        key,
                        key.as_bytes(),
                        100,
                    ))
                    .await
                    .expect("reserve expiring key"),
                DurableIdempotencyBegin::Acquired(_)
            ));
        }
        clock.store(1_200, Ordering::SeqCst);
        assert_eq!(
            store
                .prune_expired_durable_idempotency(2)
                .await
                .expect("bounded prune"),
            2
        );
        let db = store.connection().await.expect("database");
        let rows = db
            .query_all_raw(
                raw::statement(
                    db,
                    "SELECT idempotency_key FROM durable_idempotency_keys_v2 ORDER BY idempotency_key",
                    vec![],
                )
                .expect("remaining-key query"),
            )
            .await
            .expect("remaining keys");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].try_get_by_index::<String>(0).unwrap(), "c");

        store
            .prune_expired_durable_idempotency(1)
            .await
            .expect("remove final expired key");
        clock.store(2_000, Ordering::SeqCst);
        for key in ["one", "two"] {
            assert!(matches!(
                store
                    .reserve_durable_idempotency_with_capacity(
                        durable_idempotency_request(
                            "tenant-a",
                            "job.create",
                            key,
                            key.as_bytes(),
                            100,
                        ),
                        2,
                    )
                    .await
                    .expect("within capacity"),
                DurableIdempotencyBegin::Acquired(_)
            ));
        }
        assert!(matches!(
            store
                .reserve_durable_idempotency_with_capacity(
                    durable_idempotency_request("tenant-a", "job.create", "three", b"three", 100,),
                    2,
                )
                .await
                .expect("capacity outcome"),
            DurableIdempotencyBegin::CapacityExceeded
        ));
    }

    #[tokio::test]
    async fn durable_idempotency_rejects_unbounded_or_unattached_records() {
        let (mut store, _root) = build_store();
        let clock = Arc::new(AtomicI64::new(1_000));
        store.set_test_clock(clock);
        let oversized_key = "x".repeat(MAX_DURABLE_IDEMPOTENCY_KEY_BYTES + 1);
        assert!(store
            .reserve_durable_idempotency(durable_idempotency_request(
                "tenant-a",
                "job.create",
                &oversized_key,
                b"one",
                100,
            ))
            .await
            .unwrap_err()
            .to_string()
            .contains("idempotency key exceeds"));
        assert!(store
            .reserve_durable_idempotency(DurableIdempotencyRequest {
                request_digest: "A".repeat(64),
                ..durable_idempotency_request("tenant-a", "job.create", "key-1", b"one", 100,)
            })
            .await
            .unwrap_err()
            .to_string()
            .contains("lowercase SHA-256"));

        let reservation = match store
            .reserve_durable_idempotency(durable_idempotency_request(
                "tenant-a",
                "job.create",
                "key-2",
                b"two",
                100,
            ))
            .await
            .expect("reserve")
        {
            DurableIdempotencyBegin::Acquired(reservation) => reservation,
            outcome => panic!("unexpected reservation outcome: {outcome:?}"),
        };
        assert!(store
            .commit_durable_idempotency(
                &reservation,
                "missing-job",
                json!({"job_id": "missing-job"}),
                100,
            )
            .await
            .expect("unattached commit")
            .is_none());
        let job = create_test_job(&store, 0, 1).await;
        assert!(store
            .commit_durable_idempotency(
                &reservation,
                &job.id,
                json!({"result": "x".repeat(MAX_DURABLE_IDEMPOTENCY_RESULT_BYTES)}),
                100,
            )
            .await
            .unwrap_err()
            .to_string()
            .contains("result exceeds"));
    }

    #[tokio::test]
    async fn durable_text_tts_acceptance_rolls_back_every_partial_graph_failpoint() {
        for failpoint in [
            DurableTtsAcceptanceFailpoint::Projection,
            DurableTtsAcceptanceFailpoint::TextAsset,
            DurableTtsAcceptanceFailpoint::Job,
            DurableTtsAcceptanceFailpoint::Artifact,
            DurableTtsAcceptanceFailpoint::Stage,
            DurableTtsAcceptanceFailpoint::Idempotency,
        ] {
            let (mut store, _root) = build_store();
            let request = durable_idempotency_request(
                "tenant-a",
                "speech.text_to_speech.create.v1",
                "rollback-key",
                b"hello durable world",
                60_000,
            );
            let reservation = match store
                .reserve_durable_idempotency(request.clone())
                .await
                .expect("reserve acceptance")
            {
                DurableIdempotencyBegin::Acquired(reservation) => reservation,
                outcome => panic!("unexpected reservation outcome: {outcome:?}"),
            };
            store.set_durable_tts_acceptance_failpoint(Some(failpoint));
            let error = store
                .accept_durable_text_tts(durable_text_tts_acceptance(Some(reservation)))
                .await
                .expect_err("failpoint must roll back acceptance");
            assert!(error.to_string().contains("Injected durable text TTS"));
            assert_eq!(
                durable_text_tts_row_counts(&store).await,
                [0, 0, 0, 0, 0, 1],
                "partial graph survived {failpoint:?}"
            );
            assert!(store
                .claim_next_stage("worker-after-rollback", 60_000)
                .await
                .expect("claim after rollback")
                .is_none());
            assert!(matches!(
                store
                    .reserve_durable_idempotency(request)
                    .await
                    .expect("reservation remains fenced"),
                DurableIdempotencyBegin::InProgress { .. }
            ));
        }
    }

    #[tokio::test]
    async fn durable_text_tts_acceptance_has_one_graph_and_replays_after_reopen() {
        let (store, root) = build_store();
        let request = durable_idempotency_request(
            "tenant-a",
            "speech.text_to_speech.create.v1",
            "reopen-key",
            b"hello durable world",
            60_000,
        );
        let (first, second) = tokio::join!(
            store.reserve_durable_idempotency(request.clone()),
            store.reserve_durable_idempotency(request.clone()),
        );
        let outcomes = [
            first.expect("first reservation"),
            second.expect("second reservation"),
        ];
        let reservation = outcomes
            .iter()
            .find_map(|outcome| match outcome {
                DurableIdempotencyBegin::Acquired(reservation) => Some(reservation.clone()),
                _ => None,
            })
            .expect("one reservation owner");
        assert_eq!(
            outcomes
                .iter()
                .filter(|outcome| matches!(outcome, DurableIdempotencyBegin::Acquired(_)))
                .count(),
            1
        );
        assert_eq!(
            outcomes
                .iter()
                .filter(|outcome| matches!(outcome, DurableIdempotencyBegin::InProgress { .. }))
                .count(),
            1
        );

        let accepted = match store
            .accept_durable_text_tts(durable_text_tts_acceptance(Some(reservation)))
            .await
            .expect("atomic acceptance")
        {
            DurableTextTtsAcceptanceOutcome::Committed(accepted) => accepted,
            DurableTextTtsAcceptanceOutcome::ReservationLost => {
                panic!("reservation unexpectedly lost")
            }
        };
        assert_eq!(
            durable_text_tts_row_counts(&store).await,
            [1, 1, 1, 1, 1, 1]
        );
        assert_eq!(accepted.job.status, RuntimeJobStatus::Queued);
        assert_eq!(accepted.stage.status, RuntimeStageStatus::Queued);
        assert_eq!(
            accepted.stage.input_artifact_ids,
            vec![accepted.input_artifact.id.clone()]
        );
        assert_eq!(accepted.input_artifact.job_id, accepted.job.id);

        drop(store);
        let reopened = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(
            root.path().join("runtime.sqlite"),
        ));
        let replay = match reopened
            .reserve_durable_idempotency(request)
            .await
            .expect("replay after reopen")
        {
            DurableIdempotencyBegin::Replay(replay) => replay,
            outcome => panic!("unexpected reopened outcome: {outcome:?}"),
        };
        assert_eq!(replay.runtime_job_id, accepted.job.id);
        assert_eq!(
            replay.response_json["record_id"].as_str(),
            Some(accepted.record.id.as_str())
        );
        assert_eq!(
            durable_text_tts_row_counts(&reopened).await,
            [1, 1, 1, 1, 1, 1]
        );
    }

    #[tokio::test]
    async fn durable_text_tts_acceptance_fences_expiry_and_bounds_replay_response() {
        let (mut store, _root) = build_store();
        let clock = Arc::new(AtomicI64::new(1_000));
        store.set_test_clock(clock.clone());
        let reservation = match store
            .reserve_durable_idempotency(durable_idempotency_request(
                "tenant-a",
                "speech.text_to_speech.create.v1",
                "expired-key",
                b"expired",
                10,
            ))
            .await
            .expect("expired reservation")
        {
            DurableIdempotencyBegin::Acquired(reservation) => reservation,
            outcome => panic!("unexpected reservation outcome: {outcome:?}"),
        };
        clock.store(1_010, Ordering::SeqCst);
        assert!(matches!(
            store
                .accept_durable_text_tts(durable_text_tts_acceptance(Some(reservation)))
                .await
                .expect("expired acceptance outcome"),
            DurableTextTtsAcceptanceOutcome::ReservationLost
        ));
        assert_eq!(
            durable_text_tts_row_counts(&store).await,
            [0, 0, 0, 0, 0, 1]
        );

        clock.store(2_000, Ordering::SeqCst);
        let reservation = match store
            .reserve_durable_idempotency(durable_idempotency_request(
                "tenant-a",
                "speech.text_to_speech.create.v1",
                "bounded-key",
                b"bounded",
                60_000,
            ))
            .await
            .expect("bounded reservation")
        {
            DurableIdempotencyBegin::Acquired(reservation) => reservation,
            outcome => panic!("unexpected reservation outcome: {outcome:?}"),
        };
        let mut acceptance = durable_text_tts_acceptance(Some(reservation));
        acceptance.projection.input_text = "x".repeat(MAX_DURABLE_TTS_TEXT_BYTES);
        let accepted = match store
            .accept_durable_text_tts(acceptance)
            .await
            .expect("maximum bounded input should be accepted")
        {
            DurableTextTtsAcceptanceOutcome::Committed(accepted) => accepted,
            DurableTextTtsAcceptanceOutcome::ReservationLost => {
                panic!("bounded reservation unexpectedly lost")
            }
        };
        let replay_payload = json!({"record_id": accepted.record.id});
        assert!(
            serde_json::to_vec(&replay_payload).unwrap().len()
                <= MAX_DURABLE_IDEMPOTENCY_RESULT_BYTES
        );
        assert_eq!(
            durable_text_tts_row_counts(&store).await,
            [1, 1, 1, 1, 1, 1]
        );

        let mut oversized_text = durable_text_tts_acceptance(None);
        oversized_text.projection.input_text = "x".repeat(MAX_DURABLE_TTS_TEXT_BYTES + 1);
        assert!(store
            .accept_durable_text_tts(oversized_text)
            .await
            .unwrap_err()
            .to_string()
            .contains("input must be between"));

        let mut oversized_metadata = durable_text_tts_acceptance(None);
        oversized_metadata.model_snapshot_json =
            json!({"metadata": "x".repeat(MAX_DURABLE_TTS_METADATA_JSON_BYTES)});
        assert!(store
            .accept_durable_text_tts(oversized_metadata)
            .await
            .unwrap_err()
            .to_string()
            .contains("model snapshot exceeds"));
        assert_eq!(
            durable_text_tts_row_counts(&store).await,
            [1, 1, 1, 1, 1, 1]
        );
    }

    #[tokio::test]
    async fn unkeyed_text_tts_acceptance_is_atomic_and_only_claimable_after_commit() {
        let (mut store, _root) = build_store();
        let mut acceptance = durable_text_tts_acceptance(None);
        acceptance.idempotency_retention_ms = 0;
        store.set_durable_tts_acceptance_failpoint(Some(DurableTtsAcceptanceFailpoint::Stage));
        assert!(store
            .accept_durable_text_tts(acceptance.clone())
            .await
            .is_err());
        assert_eq!(
            durable_text_tts_row_counts(&store).await,
            [0, 0, 0, 0, 0, 0]
        );
        assert!(store
            .claim_next_stage("worker-before-commit", 60_000)
            .await
            .expect("claim rolled-back graph")
            .is_none());

        store.set_durable_tts_acceptance_failpoint(None);
        let accepted = match store
            .accept_durable_text_tts(acceptance)
            .await
            .expect("unkeyed acceptance")
        {
            DurableTextTtsAcceptanceOutcome::Committed(accepted) => accepted,
            DurableTextTtsAcceptanceOutcome::ReservationLost => {
                panic!("unkeyed acceptance cannot lose a reservation")
            }
        };
        assert_eq!(
            durable_text_tts_row_counts(&store).await,
            [1, 1, 1, 1, 1, 0]
        );
        let claimed = store
            .claim_next_stage("worker-after-commit", 60_000)
            .await
            .expect("claim committed graph")
            .expect("queued stage");
        assert_eq!(claimed.job.id, accepted.job.id);
        assert_eq!(claimed.stage.id, accepted.stage.id);
        assert!(store
            .claim_next_stage("other-worker", 60_000)
            .await
            .expect("second claim")
            .is_none());
    }

    #[tokio::test]
    async fn active_job_indexes_exclude_terminal_jobs() {
        let (store, _root) = build_store();
        let job = store
            .create_job(NewRuntimeJob {
                job_kind: RuntimeJobKind::TtsSpeech,
                status: RuntimeJobStatus::Queued,
                priority: 0,
                model_id: Some("Qwen3-TTS-12Hz-0.6B-CustomVoice".to_string()),
                capability: Some("tts".to_string()),
                route_record_kind: Some("text_to_speech".to_string()),
                route_record_id: Some("speech-1".to_string()),
                input_media_asset_id: None,
                input_text_asset_id: None,
                request_json: json!({}),
                model_snapshot_json: json!({}),
                retry_policy_json: json!({"max_attempts": 1}),
                max_attempts: 1,
                idempotency_key: None,
                correlation_id: None,
            })
            .await
            .expect("active job");
        let stage = store
            .create_stage(NewJobStage {
                job_id: job.id.clone(),
                sequence: 0,
                stage_kind: "speech".to_string(),
                status: RuntimeStageStatus::Queued,
                capability: Some("tts".to_string()),
                model_id: job.model_id.clone(),
                max_attempts: 1,
                input_artifact_ids: Vec::new(),
            })
            .await
            .expect("queued stage");

        assert_eq!(
            store
                .get_active_job_for_route_record(
                    RuntimeJobKind::TtsSpeech,
                    "text_to_speech",
                    "speech-1",
                )
                .await
                .expect("route lookup")
                .expect("active route job")
                .id,
            job.id
        );
        assert_eq!(
            store
                .list_active_jobs_by_kind(RuntimeJobKind::TtsSpeech)
                .await
                .expect("kind lookup")
                .len(),
            1
        );

        let cancelled = store
            .cancel_job(&job.id, Some("test cancellation".to_string()))
            .await
            .expect("cancel")
            .expect("cancelled job");
        assert_eq!(cancelled.status, RuntimeJobStatus::Cancelled);
        assert_eq!(cancelled.cancellation_state, None);
        assert!(cancelled.finished_at.is_some());
        let cancelled_stage = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(cancelled_stage.status, RuntimeStageStatus::Cancelled);
        assert_eq!(cancelled_stage.cancellation_state, None);
        assert_eq!(cancelled_stage.worker_id, None);
        assert!(store
            .get_active_job_for_route_record(
                RuntimeJobKind::TtsSpeech,
                "text_to_speech",
                "speech-1",
            )
            .await
            .expect("terminal route lookup")
            .is_none());
        assert!(store
            .list_active_jobs_by_kind(RuntimeJobKind::TtsSpeech)
            .await
            .expect("terminal kind lookup")
            .is_empty());
    }

    #[tokio::test]
    async fn registered_worker_heartbeat_tracks_instance_resources_and_capacity() {
        let (store, _root) = build_store();
        let registration = RuntimeWorkerRegistration {
            version: WORKER_REGISTRATION_VERSION,
            worker_id: "worker-logical".to_string(),
            instance_id: "instance-a".to_string(),
            queue_classes: vec![QueueClass::BatchAsr, QueueClass::Evaluation],
            capabilities: vec!["asr".to_string()],
            model_ids: vec!["model-a".to_string()],
            stage_kinds: vec!["asr_transcribe".to_string()],
            resources: WorkerResourceCapacity {
                targets: vec![ResourceTarget::Gpu],
                memory_bytes: Some(24 * 1024 * 1024 * 1024),
                concurrency_slots: 2,
                ..WorkerResourceCapacity::default()
            },
            software_version: "test-version".to_string(),
        };
        let details = RuntimeWorkerHeartbeatDetails {
            version: WORKER_HEARTBEAT_DETAILS_VERSION,
            available_slots: 1,
            active_lease_ids: vec!["stage-a".to_string()],
            last_error: None,
            health_json: json!({"temperature_c": 60}),
        };

        let heartbeat = store
            .upsert_registered_worker_heartbeat(RegisteredWorkerHeartbeatUpdate {
                registration: registration.clone(),
                status: "running".to_string(),
                current_job_id: None,
                current_stage_id: None,
                details: details.clone(),
                diagnostic_json: json!({"source": "test"}),
            })
            .await
            .expect("registered heartbeat");
        assert_eq!(heartbeat.instance_id, "instance-a");
        assert_eq!(heartbeat.registration, registration);
        assert_eq!(heartbeat.details, details);
        assert_eq!(
            heartbeat.queue_names,
            vec!["batch_asr".to_string(), "evaluation".to_string()]
        );

        let replacement = RuntimeWorkerRegistration {
            instance_id: "instance-b".to_string(),
            ..heartbeat.registration
        };
        let replaced = store
            .upsert_registered_worker_heartbeat(RegisteredWorkerHeartbeatUpdate {
                registration: replacement,
                status: "idle".to_string(),
                current_job_id: None,
                current_stage_id: None,
                details: RuntimeWorkerHeartbeatDetails {
                    available_slots: 2,
                    active_lease_ids: vec![],
                    ..heartbeat.details
                },
                diagnostic_json: json!({"source": "replacement"}),
            })
            .await
            .expect("replacement heartbeat");
        assert_eq!(replaced.instance_id, "instance-b");
        assert_eq!(replaced.status, "idle");
        assert_eq!(replaced.details.available_slots, 2);
    }

    #[tokio::test]
    async fn queue_health_requires_fresh_queue_coverage() {
        let (store, _root) = build_store();
        let (_job, stage) = create_test_job_and_stage(&store, 0, "asr_infer", 1).await;
        assert_eq!(stage.queue_class, QueueClass::BatchAsr);

        let uncovered = store.runtime_queue_health(5_000).await.expect("health");
        assert_eq!(uncovered.queues.len(), 1);
        assert_eq!(
            uncovered.uncovered_queue_classes,
            vec![QueueClass::BatchAsr]
        );

        store
            .upsert_registered_worker_heartbeat(RegisteredWorkerHeartbeatUpdate {
                registration: RuntimeWorkerRegistration {
                    version: WORKER_REGISTRATION_VERSION,
                    worker_id: "asr-worker".to_string(),
                    instance_id: "asr-worker-instance".to_string(),
                    queue_classes: vec![QueueClass::BatchAsr],
                    capabilities: vec!["asr".to_string()],
                    model_ids: vec![],
                    stage_kinds: vec!["asr_infer".to_string()],
                    resources: WorkerResourceCapacity::default(),
                    software_version: "test".to_string(),
                },
                status: "idle".to_string(),
                current_job_id: None,
                current_stage_id: None,
                details: RuntimeWorkerHeartbeatDetails {
                    version: WORKER_HEARTBEAT_DETAILS_VERSION,
                    available_slots: 1,
                    active_lease_ids: vec![],
                    last_error: None,
                    health_json: json!({}),
                },
                diagnostic_json: json!({}),
            })
            .await
            .expect("worker heartbeat");

        let covered = store.runtime_queue_health(5_000).await.expect("health");
        assert_eq!(covered.healthy_workers, 1);
        assert!(covered.uncovered_queue_classes.is_empty());

        let db = store.connection().await.expect("database");
        db.execute_raw(
            crate::db::raw::statement(
                db,
                "UPDATE runtime_worker_heartbeats SET last_heartbeat_at = ?1 WHERE worker_id = ?2",
                vec![
                    current_timestamp_millis().saturating_sub(10_000).into(),
                    "asr-worker".into(),
                ],
            )
            .expect("statement"),
        )
        .await
        .expect("stale heartbeat update");

        let stale = store.runtime_queue_health(1_000).await.expect("health");
        assert_eq!(stale.stale_workers, 1);
        assert_eq!(stale.uncovered_queue_classes, vec![QueueClass::BatchAsr]);
    }

    #[tokio::test]
    async fn job_transitions_are_status_conditional() {
        let (store, _root) = build_store();
        let job = store
            .create_job(NewRuntimeJob {
                job_kind: RuntimeJobKind::TtsSpeech,
                status: RuntimeJobStatus::Queued,
                priority: 0,
                model_id: Some("Qwen3-TTS-0.6B".to_string()),
                capability: Some("tts".to_string()),
                route_record_kind: Some("speech_history".to_string()),
                route_record_id: Some("speech-1".to_string()),
                input_media_asset_id: None,
                input_text_asset_id: None,
                request_json: json!({"text": "hello"}),
                model_snapshot_json: json!({}),
                retry_policy_json: json!({}),
                max_attempts: 1,
                idempotency_key: None,
                correlation_id: None,
            })
            .await
            .expect("job");

        let cancelled = store
            .transition_job_status(
                &job.id,
                &[RuntimeJobStatus::Queued],
                RuntimeJobStatus::Cancelled,
                None,
                None,
                Some("user requested".to_string()),
            )
            .await
            .expect("cancel transition")
            .expect("job should transition");

        assert_eq!(cancelled.status, RuntimeJobStatus::Cancelled);
        assert_eq!(
            cancelled.cancellation_reason.as_deref(),
            Some("user requested")
        );

        let late_completion = store
            .transition_job_status(
                &job.id,
                &[RuntimeJobStatus::Running],
                RuntimeJobStatus::Completed,
                None,
                None,
                None,
            )
            .await
            .expect("late transition should not error");

        assert!(late_completion.is_none());
        let fetched = store
            .get_job(&job.id)
            .await
            .expect("fetch")
            .expect("job still exists");
        assert_eq!(fetched.status, RuntimeJobStatus::Cancelled);
    }

    #[tokio::test]
    async fn queue_claim_recovery_and_cancel_are_durable() {
        let (store, _root) = build_store();
        let job = store
            .create_job(NewRuntimeJob {
                job_kind: RuntimeJobKind::AsrTranscription,
                status: RuntimeJobStatus::Queued,
                priority: 10,
                model_id: None,
                capability: Some("asr".to_string()),
                route_record_kind: Some("transcription".to_string()),
                route_record_id: Some("route-1".to_string()),
                input_media_asset_id: None,
                input_text_asset_id: None,
                request_json: json!({}),
                model_snapshot_json: json!({}),
                retry_policy_json: json!({"max_attempts": 2}),
                max_attempts: 2,
                idempotency_key: None,
                correlation_id: None,
            })
            .await
            .expect("job");
        let stage = store
            .create_stage(NewJobStage {
                job_id: job.id.clone(),
                sequence: 0,
                stage_kind: "asr_infer".to_string(),
                status: RuntimeStageStatus::Queued,
                capability: Some("asr".to_string()),
                model_id: None,
                max_attempts: 2,
                input_artifact_ids: vec![],
            })
            .await
            .expect("stage");

        let claimed = store
            .claim_next_stage("worker-1", 0)
            .await
            .expect("claim")
            .expect("stage should be claimed");
        assert_eq!(claimed.stage.id, stage.id);
        assert_eq!(claimed.stage.status, RuntimeStageStatus::Running);
        assert_eq!(claimed.stage.attempt_count, 1);

        let recovered = store
            .recover_expired_stage_leases(DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT)
            .await
            .expect("recover");
        assert_eq!(recovered, 1);

        let retried = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(retried.status, RuntimeStageStatus::Retrying);
        assert_eq!(store.queued_stage_count().await.expect("count"), 1);

        let cancelled = store
            .cancel_job(&job.id, Some("test cleanup".to_string()))
            .await
            .expect("cancel")
            .expect("job should cancel");
        assert_eq!(cancelled.status, RuntimeJobStatus::Cancelled);

        let cancelled_stage = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(cancelled_stage.status, RuntimeStageStatus::Cancelled);
    }

    #[tokio::test]
    async fn expired_lease_recovery_runs_in_stable_bounded_batches() {
        let (mut store, _root) = build_store();
        let clock = Arc::new(AtomicI64::new(1_000));
        store.set_test_clock(clock.clone());
        let mut stage_ids = Vec::new();

        for index in 0..3 {
            clock.store(1_000 + index * 100, Ordering::SeqCst);
            let (_job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 3).await;
            let worker_id = format!("worker-{index}");
            let claimed = store
                .claim_next_stage(&worker_id, 10)
                .await
                .expect("claim")
                .expect("stage should be claimed");
            assert_eq!(claimed.stage.id, stage.id);
            stage_ids.push(stage.id);
        }

        clock.store(10_000, Ordering::SeqCst);
        assert_eq!(
            store
                .recover_expired_stage_leases(2)
                .await
                .expect("first recovery batch"),
            2
        );
        for stage_id in &stage_ids[..2] {
            assert_eq!(
                store
                    .get_stage(stage_id)
                    .await
                    .expect("stage")
                    .expect("stage exists")
                    .status,
                RuntimeStageStatus::Retrying
            );
        }
        assert_eq!(
            store
                .get_stage(&stage_ids[2])
                .await
                .expect("stage")
                .expect("stage exists")
                .status,
            RuntimeStageStatus::Running
        );

        assert_eq!(
            store
                .recover_expired_stage_leases(2)
                .await
                .expect("second recovery batch"),
            1
        );
        assert_eq!(
            store
                .get_stage(&stage_ids[2])
                .await
                .expect("stage")
                .expect("stage exists")
                .status,
            RuntimeStageStatus::Retrying
        );
        assert_eq!(
            store
                .recover_expired_stage_leases(2)
                .await
                .expect("empty recovery batch"),
            0
        );
    }

    #[tokio::test]
    async fn acknowledged_job_survives_store_reconstruction_and_completes() {
        let root = tempfile::tempdir().expect("temp dir");
        let db_path = root.path().join("runtime.sqlite");
        let first_store =
            BatchRuntimeStore::initialize_with_database(StoreDatabase::new(db_path.clone()));
        let input = first_store
            .create_text_asset(NewTextAsset {
                raw_text: "durable input".to_string(),
                normalized_text: None,
                language_hint: Some("en".to_string()),
                sha256: Some(sha256_hex(b"durable input")),
                safety_status: "accepted".to_string(),
                retention_policy: "default".to_string(),
                structure_json: json!({}),
            })
            .await
            .expect("durable input");
        let job = first_store
            .create_job(NewRuntimeJob {
                job_kind: RuntimeJobKind::TtsSpeech,
                status: RuntimeJobStatus::Queued,
                priority: 0,
                model_id: Some("test-model".to_string()),
                capability: Some("tts".to_string()),
                route_record_kind: Some("durability_test".to_string()),
                route_record_id: Some("acknowledged-job".to_string()),
                input_media_asset_id: None,
                input_text_asset_id: Some(input.id.clone()),
                request_json: json!({"text": "durable input"}),
                model_snapshot_json: json!({"model_id": "test-model"}),
                retry_policy_json: json!({"max_attempts": 1}),
                max_attempts: 1,
                idempotency_key: Some("durability-test".to_string()),
                correlation_id: Some("durability-test".to_string()),
            })
            .await
            .expect("acknowledged job");
        let stage = first_store
            .create_stage(NewJobStage {
                job_id: job.id.clone(),
                sequence: 0,
                stage_kind: "tts_generate".to_string(),
                status: RuntimeStageStatus::Queued,
                capability: Some("tts".to_string()),
                model_id: job.model_id.clone(),
                max_attempts: 1,
                input_artifact_ids: vec![],
            })
            .await
            .expect("acknowledged stage");

        drop(first_store);

        let restarted_store =
            BatchRuntimeStore::initialize_with_database(StoreDatabase::new(db_path.clone()));
        let rediscovered_input = restarted_store
            .get_text_asset(&input.id)
            .await
            .expect("rediscover input")
            .expect("durable input exists");
        let rediscovered_job = restarted_store
            .get_job(&job.id)
            .await
            .expect("rediscover job")
            .expect("acknowledged job exists");
        let rediscovered_stage = restarted_store
            .get_stage(&stage.id)
            .await
            .expect("rediscover stage")
            .expect("acknowledged stage exists");
        assert_eq!(rediscovered_input.raw_text, "durable input");
        assert_eq!(
            rediscovered_job.input_text_asset_id.as_deref(),
            Some(input.id.as_str())
        );
        assert_eq!(rediscovered_job.status, RuntimeJobStatus::Queued);
        assert_eq!(rediscovered_stage.status, RuntimeStageStatus::Queued);

        let claimed = restarted_store
            .claim_next_stage("worker-after-restart", 60_000)
            .await
            .expect("claim after restart")
            .expect("rediscovered stage is claimable");
        assert_eq!(claimed.stage.id, stage.id);
        assert_eq!(
            claimed.stage.worker_id.as_deref(),
            Some("worker-after-restart")
        );
        let lease = claimed.lease().expect("restart worker lease");
        let artifact = restarted_store
            .publish_stage_output_artifact(&lease, test_stage_output("restart-result"))
            .await
            .expect("publish after restart")
            .expect("active attempt owns publication");
        restarted_store
            .complete_stage(&lease, vec![artifact.id.clone()])
            .await
            .expect("complete after restart")
            .expect("active attempt completes stage");

        drop(restarted_store);

        let final_store = BatchRuntimeStore::initialize_with_database(StoreDatabase::new(db_path));
        let final_job = final_store
            .get_job(&job.id)
            .await
            .expect("load final job")
            .expect("final job exists");
        let final_stage = final_store
            .get_stage(&stage.id)
            .await
            .expect("load final stage")
            .expect("final stage exists");
        let final_artifact = final_store
            .get_artifact(&artifact.id)
            .await
            .expect("load final artifact")
            .expect("final artifact exists");
        assert_eq!(final_job.status, RuntimeJobStatus::Completed);
        assert_eq!(final_stage.status, RuntimeStageStatus::Completed);
        assert_eq!(final_stage.output_artifact_ids, vec![artifact.id.clone()]);
        assert_eq!(
            final_artifact.producer_attempt_token,
            artifact.producer_attempt_token
        );
        assert_eq!(
            final_artifact.publication_key.as_deref(),
            Some("restart-result")
        );
    }

    #[tokio::test]
    async fn filtered_stage_claim_skips_incompatible_higher_priority_stage() {
        let (store, _root) = build_store();
        let tts_job = store
            .create_job(NewRuntimeJob {
                job_kind: RuntimeJobKind::TtsSpeech,
                status: RuntimeJobStatus::Queued,
                priority: 50,
                model_id: Some("Qwen3-TTS-0.6B".to_string()),
                capability: Some("tts".to_string()),
                route_record_kind: Some("speech_history".to_string()),
                route_record_id: Some("speech-1".to_string()),
                input_media_asset_id: None,
                input_text_asset_id: None,
                request_json: json!({}),
                model_snapshot_json: json!({}),
                retry_policy_json: json!({}),
                max_attempts: 1,
                idempotency_key: None,
                correlation_id: None,
            })
            .await
            .expect("tts job");
        let tts_stage = store
            .create_stage(NewJobStage {
                job_id: tts_job.id.clone(),
                sequence: 0,
                stage_kind: "tts_generate".to_string(),
                status: RuntimeStageStatus::Queued,
                capability: Some("tts".to_string()),
                model_id: tts_job.model_id.clone(),
                max_attempts: 1,
                input_artifact_ids: vec![],
            })
            .await
            .expect("tts stage");

        let asr_job = store
            .create_job(NewRuntimeJob {
                job_kind: RuntimeJobKind::AsrTranscription,
                status: RuntimeJobStatus::Queued,
                priority: 10,
                model_id: Some("Parakeet-TDT-0.6B-v3".to_string()),
                capability: Some("asr".to_string()),
                route_record_kind: Some("transcription".to_string()),
                route_record_id: Some("transcription-1".to_string()),
                input_media_asset_id: None,
                input_text_asset_id: None,
                request_json: json!({}),
                model_snapshot_json: json!({}),
                retry_policy_json: json!({}),
                max_attempts: 1,
                idempotency_key: None,
                correlation_id: None,
            })
            .await
            .expect("asr job");
        let asr_stage = store
            .create_stage(NewJobStage {
                job_id: asr_job.id.clone(),
                sequence: 0,
                stage_kind: "asr_infer".to_string(),
                status: RuntimeStageStatus::Queued,
                capability: Some("asr".to_string()),
                model_id: asr_job.model_id.clone(),
                max_attempts: 1,
                input_artifact_ids: vec![],
            })
            .await
            .expect("asr stage");

        let mut filter = StageClaimFilter::for_worker_queues(&["batch_asr".to_string()]);
        filter.capabilities = vec!["asr".to_string()];

        let claimed = store
            .claim_next_stage_with_filter("asr-worker", 60_000, &filter)
            .await
            .expect("claim")
            .expect("asr stage should be claimed");

        assert_eq!(claimed.stage.id, asr_stage.id);
        assert_eq!(claimed.stage.queue_class, QueueClass::BatchAsr);
        assert_eq!(claimed.stage.capability.as_deref(), Some("asr"));
        assert_eq!(claimed.stage.worker_id.as_deref(), Some("asr-worker"));

        let tts_stage = store
            .get_stage(&tts_stage.id)
            .await
            .expect("fetch tts stage")
            .expect("tts stage exists");
        assert_eq!(tts_stage.status, RuntimeStageStatus::Queued);
        assert_eq!(tts_stage.worker_id, None);

        let wildcard_claim = store
            .claim_next_stage("general-batch-worker", 60_000)
            .await
            .expect("wildcard claim")
            .expect("legacy batch wildcard should claim remaining TTS stage");
        assert_eq!(wildcard_claim.stage.id, tts_stage.id);
        assert_eq!(wildcard_claim.stage.queue_class, QueueClass::BatchTts);
    }

    #[tokio::test]
    async fn resource_aware_claim_rejects_backend_device_and_capacity_mismatch() {
        let (store, _root) = build_store();
        let gpu_job = create_test_job(&store, 50, 1).await;
        let gpu_stage = store
            .create_stage_with_dispatch(NewJobStageDispatch {
                stage: NewJobStage {
                    job_id: gpu_job.id,
                    sequence: 0,
                    stage_kind: "gpu_evaluation".to_string(),
                    status: RuntimeStageStatus::Queued,
                    capability: Some("test".to_string()),
                    model_id: None,
                    max_attempts: 1,
                    input_artifact_ids: vec![],
                },
                queue_class: QueueClass::Evaluation,
                resource_hints: StageResourceHints {
                    target: ResourceTarget::Gpu,
                    backend: Some(RuntimeBackendClass::Metal),
                    device_class: Some(DeviceClass::AppleGpu),
                    min_memory_bytes: Some(16 * 1024 * 1024 * 1024),
                    concurrency_weight: 2,
                    ..StageResourceHints::default()
                },
            })
            .await
            .expect("GPU stage");
        let cpu_job = create_test_job(&store, 10, 1).await;
        let cpu_stage = store
            .create_stage_with_dispatch(NewJobStageDispatch {
                stage: NewJobStage {
                    job_id: cpu_job.id,
                    sequence: 0,
                    stage_kind: "cpu_evaluation".to_string(),
                    status: RuntimeStageStatus::Queued,
                    capability: Some("test".to_string()),
                    model_id: None,
                    max_attempts: 1,
                    input_artifact_ids: vec![],
                },
                queue_class: QueueClass::Evaluation,
                resource_hints: StageResourceHints {
                    target: ResourceTarget::Gpu,
                    backend: Some(RuntimeBackendClass::Cuda),
                    device_class: Some(DeviceClass::NvidiaGpu),
                    min_memory_bytes: Some(2 * 1024 * 1024 * 1024),
                    ..StageResourceHints::default()
                },
            })
            .await
            .expect("CPU stage");

        let mut filter = StageClaimFilter::for_worker_queues(&["evaluation".to_string()]);
        filter.resources = WorkerResourceCapacity {
            targets: vec![ResourceTarget::Gpu],
            backends: vec![RuntimeBackendClass::Cuda],
            device_classes: vec![DeviceClass::NvidiaGpu],
            memory_bytes: Some(8 * 1024 * 1024 * 1024),
            concurrency_slots: 1,
            ..WorkerResourceCapacity::default()
        };
        let claimed = store
            .claim_next_stage_with_filter("cpu-worker", 60_000, &filter)
            .await
            .expect("resource-aware claim")
            .expect("compatible CUDA stage");
        assert_eq!(claimed.stage.id, cpu_stage.id);
        assert_eq!(claimed.stage.queue_class, QueueClass::Evaluation);
        assert_eq!(
            claimed.stage.resource_hints.backend,
            Some(RuntimeBackendClass::Cuda)
        );
        assert_eq!(
            claimed.stage.resource_hints.device_class,
            Some(DeviceClass::NvidiaGpu)
        );

        let gpu_stage = store
            .get_stage(&gpu_stage.id)
            .await
            .expect("GPU stage fetch")
            .expect("GPU stage exists");
        assert_eq!(gpu_stage.status, RuntimeStageStatus::Queued);
        assert_eq!(gpu_stage.worker_id, None);
    }

    #[tokio::test]
    async fn concurrent_claimers_take_distinct_candidates() {
        let (store, _root) = build_store();
        create_test_job_and_stage(&store, 20, "fake_stage", 1).await;
        create_test_job_and_stage(&store, 10, "fake_stage", 1).await;

        let first_store = store.clone();
        let second_store = store.clone();
        let (first, second) = tokio::join!(
            first_store.claim_next_stage("worker-1", 60_000),
            second_store.claim_next_stage("worker-2", 60_000),
        );
        let first = first.expect("first claim").expect("first candidate");
        let second = second.expect("second claim").expect("second candidate");

        assert_ne!(first.stage.id, second.stage.id);
        assert_ne!(first.stage.worker_id, second.stage.worker_id);
    }

    #[tokio::test]
    async fn claim_waits_for_predecessor_completion() {
        let (store, _root) = build_store();
        let (job, first_stage) = create_test_job_and_stage(&store, 0, "first_stage", 1).await;
        let second_stage = store
            .create_stage(NewJobStage {
                job_id: job.id,
                sequence: 1,
                stage_kind: "second_stage".to_string(),
                status: RuntimeStageStatus::Queued,
                capability: Some("test".to_string()),
                model_id: None,
                max_attempts: 1,
                input_artifact_ids: vec![],
            })
            .await
            .expect("second stage");

        let first = store
            .claim_next_stage("worker-1", 60_000)
            .await
            .expect("first claim")
            .expect("first stage should be claimable");
        assert_eq!(first.stage.id, first_stage.id);
        assert!(store
            .claim_next_stage("worker-2", 60_000)
            .await
            .expect("blocked claim")
            .is_none());

        store
            .complete_stage(&first.lease().expect("first lease"), vec![])
            .await
            .expect("first completion")
            .expect("owned first completion");
        let second = store
            .claim_next_stage("worker-2", 60_000)
            .await
            .expect("second claim")
            .expect("second stage should become claimable");
        assert_eq!(second.stage.id, second_stage.id);
    }

    #[tokio::test]
    async fn claim_cas_rechecks_parent_job_eligibility() {
        let (store, _root) = build_store();
        let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
        store
            .transition_job_status(
                &job.id,
                &[RuntimeJobStatus::Queued],
                RuntimeJobStatus::Cancelled,
                None,
                None,
                Some("cancel before claim CAS".to_string()),
            )
            .await
            .expect("cancel job")
            .expect("job transition");

        let now = current_timestamp_millis();
        let claimed = store
            .try_claim_stage_candidate(
                store.connection().await.expect("database"),
                StageClaimCandidate {
                    stage_id: stage.id.clone(),
                    stage_kind: stage.stage_kind.clone(),
                    job_kind: job.job_kind,
                    queue_class: stage.queue_class,
                    resource_hints: stage.resource_hints.clone(),
                    capability: stage.capability.clone(),
                    model_id: stage.model_id.clone(),
                },
                "worker-1",
                now,
                now + 60_000,
            )
            .await
            .expect("claim CAS");

        assert!(claimed.is_none());
        let stage = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(stage.status, RuntimeStageStatus::Queued);
        assert_eq!(stage.worker_id, None);
    }

    #[tokio::test]
    async fn terminal_parent_fences_late_stage_completion() {
        let (store, _root) = build_store();
        let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
        let claimed = store
            .claim_next_stage("worker-1", 60_000)
            .await
            .expect("claim")
            .expect("stage should be claimed");
        store
            .transition_job_status(
                &job.id,
                &[RuntimeJobStatus::Running],
                RuntimeJobStatus::Cancelled,
                None,
                None,
                Some("cancelled outside stage transaction".to_string()),
            )
            .await
            .expect("cancel parent")
            .expect("parent transition");

        assert!(store
            .complete_stage(&claimed.lease().expect("lease"), vec![])
            .await
            .expect("late completion")
            .is_none());
        let running_stage = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(running_stage.status, RuntimeStageStatus::Running);
        assert!(running_stage.output_artifact_ids.is_empty());
    }

    #[tokio::test]
    async fn stale_owner_cannot_finish_a_reclaimed_attempt() {
        let (store, _root) = build_store();
        let (_job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 3).await;
        let first = store
            .claim_next_stage("worker-1", 0)
            .await
            .expect("first claim")
            .expect("first attempt");
        let first_lease = first.lease().expect("first lease");
        assert_eq!(
            store
                .recover_expired_stage_leases(DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT)
                .await
                .expect("recover"),
            1
        );

        let second = store
            .claim_next_stage("worker-2", 60_000)
            .await
            .expect("second claim")
            .expect("second attempt");
        let second_lease = second.lease().expect("second lease");
        assert_eq!(second_lease.attempt_count, first_lease.attempt_count + 1);

        assert!(store
            .complete_stage(&first_lease, vec![])
            .await
            .expect("stale completion")
            .is_none());
        assert!(store
            .fail_stage(
                &first_lease,
                false,
                Some("stale".to_string()),
                Some("stale owner".to_string()),
            )
            .await
            .expect("stale failure")
            .is_none());

        let running = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(running.status, RuntimeStageStatus::Running);
        assert_eq!(running.worker_id.as_deref(), Some("worker-2"));
        assert_eq!(running.attempt_count, second_lease.attempt_count);

        let artifact = store
            .publish_stage_output_artifact(&second_lease, test_stage_output("current-output"))
            .await
            .expect("publish current output")
            .expect("current attempt owns output");
        let completed = store
            .complete_stage(&second_lease, vec![artifact.id.clone()])
            .await
            .expect("current completion")
            .expect("current owner completes");
        assert_eq!(completed.output_artifact_ids, vec![artifact.id]);
    }

    #[tokio::test]
    async fn cancellation_fences_results_until_active_stage_teardown() {
        let (store, _root) = build_store();
        let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
        let claimed = store
            .claim_next_stage("worker-1", 60_000)
            .await
            .expect("claim")
            .expect("active attempt");
        let lease = claimed.lease().expect("lease");

        store
            .cancel_job(&job.id, Some("user cancelled".to_string()))
            .await
            .expect("cancel")
            .expect("cancelled job");

        assert!(store
            .complete_stage(&lease, vec![])
            .await
            .expect("late completion")
            .is_none());
        assert!(store
            .fail_stage(
                &lease,
                false,
                Some("late".to_string()),
                Some("late failure".to_string()),
            )
            .await
            .expect("late failure")
            .is_none());
        let requested = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(requested.status, RuntimeStageStatus::Running);
        assert_eq!(
            requested.cancellation_state,
            Some(RuntimeCancellationState::Requested)
        );
        assert_eq!(requested.worker_id.as_deref(), Some("worker-1"));
        assert!(requested.lease_expires_at.is_some());
        assert!(store
            .stage_lease_is_active(&lease)
            .await
            .expect("cancelling lease remains owned"));
        let stale = StageLease {
            attempt_token: Some("stale-attempt".to_string()),
            ..lease.clone()
        };
        assert!(!store
            .mark_stage_execution_stopping(&stale)
            .await
            .expect("stale stopping fence"));
        assert!(store
            .finalize_stage_cancellation(&stale)
            .await
            .expect("stale finalization fence")
            .is_none());

        assert!(store
            .mark_stage_execution_stopping(&lease)
            .await
            .expect("mark execution stopping"));
        assert!(store
            .finalize_stage_cancellation(&lease)
            .await
            .expect("finalize cancellation")
            .is_some());
        assert!(store
            .finalize_stage_cancellation(&lease)
            .await
            .expect("duplicate finalization")
            .is_none());

        let cancelled = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(cancelled.status, RuntimeStageStatus::Cancelled);
        assert_eq!(cancelled.cancellation_state, None);
        assert_eq!(cancelled.worker_id, None);
        assert_eq!(cancelled.lease_expires_at, None);
    }

    #[tokio::test]
    async fn active_tts_cancellation_updates_projection_only_after_teardown() {
        let (store, _root) = build_store();
        let accepted = match store
            .accept_durable_text_tts(durable_text_tts_acceptance(None))
            .await
            .expect("durable TTS acceptance")
        {
            DurableTextTtsAcceptanceOutcome::Committed(accepted) => accepted,
            DurableTextTtsAcceptanceOutcome::ReservationLost => {
                panic!("unkeyed acceptance cannot lose a reservation")
            }
        };
        let claimed = store
            .claim_next_stage("tts-worker", 60_000)
            .await
            .expect("claim")
            .expect("active TTS attempt");
        let lease = claimed.lease().expect("lease");
        let attempt_token = lease
            .attempt_token
            .clone()
            .expect("claimed stage attempt token");
        let db = store.connection().await.expect("database");
        db.execute_raw(
            raw::statement(
                db,
                r#"
                UPDATE speech_history_records
                SET processing_status = 'processing', runtime_stage_id = ?1, runtime_attempt_token = ?2
                WHERE id = ?3
                "#,
                vec![
                    lease.stage_id.clone().into(),
                    attempt_token.into(),
                    accepted.record.id.clone().into(),
                ],
            )
            .expect("projection update statement"),
        )
        .await
        .expect("mark projection processing");

        let requested = store
            .cancel_job(&accepted.job.id, Some("user cancelled speech".to_string()))
            .await
            .expect("request cancellation")
            .expect("active job");
        assert_eq!(requested.status, RuntimeJobStatus::Running);
        let processing = db
            .query_one_raw(
                raw::statement(
                    db,
                    "SELECT processing_status, processing_error FROM speech_history_records WHERE id = ?1",
                    vec![accepted.record.id.clone().into()],
                )
                .expect("processing projection statement"),
            )
            .await
            .expect("processing projection query")
            .expect("processing projection");
        assert_eq!(
            processing
                .try_get_by_index::<String>(0)
                .expect("processing status"),
            "processing"
        );
        assert_eq!(
            processing
                .try_get_by_index::<Option<String>>(1)
                .expect("processing error"),
            None
        );

        assert!(store
            .mark_stage_execution_stopping(&lease)
            .await
            .expect("mark stopping"));
        store
            .finalize_stage_cancellation(&lease)
            .await
            .expect("finalize cancellation")
            .expect("exact attempt finalizes");
        let terminal = db
            .query_one_raw(
                raw::statement(
                    db,
                    "SELECT processing_status, processing_error, runtime_stage_id, runtime_attempt_token FROM speech_history_records WHERE id = ?1",
                    vec![accepted.record.id.into()],
                )
                .expect("terminal projection statement"),
            )
            .await
            .expect("terminal projection query")
            .expect("terminal projection");
        assert_eq!(
            terminal.try_get_by_index::<String>(0).expect("status"),
            "failed"
        );
        assert_eq!(
            terminal
                .try_get_by_index::<Option<String>>(1)
                .expect("error")
                .as_deref(),
            Some("user cancelled speech")
        );
        assert_eq!(
            terminal
                .try_get_by_index::<Option<String>>(2)
                .expect("stage binding"),
            None
        );
        assert_eq!(
            terminal
                .try_get_by_index::<Option<String>>(3)
                .expect("attempt binding"),
            None
        );
    }

    #[tokio::test]
    async fn bounded_reconciliation_repairs_cancelled_transcription_projection() {
        let (store, _root) = build_store();
        let record_id = "cancelled-transcription";
        let db = store.connection().await.expect("database");
        db.execute_raw(
            raw::statement(
                db,
                r#"
                INSERT INTO transcription_records (
                    id, created_at, processing_status, processing_error,
                    processing_progress_json, runtime_stage_id, runtime_attempt_token,
                    processing_time_ms, audio_mime_type, audio_storage_path, transcription
                ) VALUES (?1, ?2, 'processing', NULL, '{}', 'old-stage', 'old-attempt', 0, 'audio/wav', '', '')
                "#,
                vec![record_id.into(), current_timestamp_millis().into()],
            )
            .expect("transcription projection statement"),
        )
        .await
        .expect("transcription projection");
        let job = store
            .create_job(NewRuntimeJob {
                job_kind: RuntimeJobKind::AsrTranscription,
                status: RuntimeJobStatus::Queued,
                priority: 0,
                model_id: None,
                capability: Some("asr".to_string()),
                route_record_kind: Some("transcription".to_string()),
                route_record_id: Some(record_id.to_string()),
                input_media_asset_id: None,
                input_text_asset_id: None,
                request_json: json!({}),
                model_snapshot_json: json!({}),
                retry_policy_json: json!({}),
                max_attempts: 1,
                idempotency_key: None,
                correlation_id: None,
            })
            .await
            .expect("runtime job");
        db.execute_raw(
            raw::statement(
                db,
                r#"
                UPDATE runtime_jobs
                SET status = 'cancelled', cancellation_reason = 'cancelled while offline',
                    finished_at = ?1, updated_at = ?1
                WHERE id = ?2
                "#,
                vec![current_timestamp_millis().into(), job.id.into()],
            )
            .expect("simulate terminal job statement"),
        )
        .await
        .expect("simulate terminal job");

        let report = store
            .reconcile_inconsistent_states(1)
            .await
            .expect("bounded reconciliation");
        assert_eq!(report.jobs_repaired, 0);
        assert_eq!(report.stages_repaired, 0);
        assert_eq!(report.route_projections_repaired, 1);
        let projection = db
            .query_one_raw(
                raw::statement(
                    db,
                    "SELECT processing_status, processing_error, processing_progress_json, runtime_stage_id, runtime_attempt_token FROM transcription_records WHERE id = ?1",
                    vec![record_id.into()],
                )
                .expect("reconciled projection statement"),
            )
            .await
            .expect("reconciled projection query")
            .expect("reconciled projection");
        assert_eq!(
            projection.try_get_by_index::<String>(0).expect("status"),
            "failed"
        );
        assert_eq!(
            projection
                .try_get_by_index::<Option<String>>(1)
                .expect("error")
                .as_deref(),
            Some("cancelled while offline")
        );
        for index in 2..=4 {
            assert_eq!(
                projection
                    .try_get_by_index::<Option<String>>(index)
                    .expect("cleared runtime projection field"),
                None
            );
        }
    }

    #[tokio::test]
    async fn cancelled_job_cannot_overwrite_an_active_replacement_projection() {
        let (store, _root) = build_store();
        let record_id = "replacement-owned-transcription";
        let db = store.connection().await.expect("database");
        db.execute_raw(
            raw::statement(
                db,
                r#"
                INSERT INTO transcription_records (
                    id, created_at, processing_status, processing_error,
                    processing_progress_json, runtime_stage_id, runtime_attempt_token,
                    processing_time_ms, audio_mime_type, audio_storage_path, transcription
                ) VALUES (?1, ?2, 'processing', NULL, '{}', 'replacement-stage',
                    'replacement-attempt', 0, 'audio/wav', '', '')
                "#,
                vec![record_id.into(), current_timestamp_millis().into()],
            )
            .expect("transcription projection statement"),
        )
        .await
        .expect("transcription projection");
        let job = |correlation_id: &str| NewRuntimeJob {
            job_kind: RuntimeJobKind::AsrTranscription,
            status: RuntimeJobStatus::Queued,
            priority: 0,
            model_id: None,
            capability: Some("asr".to_string()),
            route_record_kind: Some("transcription".to_string()),
            route_record_id: Some(record_id.to_string()),
            input_media_asset_id: None,
            input_text_asset_id: None,
            request_json: json!({}),
            model_snapshot_json: json!({}),
            retry_policy_json: json!({}),
            max_attempts: 1,
            idempotency_key: None,
            correlation_id: Some(correlation_id.to_string()),
        };
        let cancelled = store.create_job(job("old")).await.expect("old job");
        let replacement = store
            .create_job(job("replacement"))
            .await
            .expect("replacement job");

        let cancelled = store
            .cancel_job(&cancelled.id, Some("old job cancelled".to_string()))
            .await
            .expect("cancel old job")
            .expect("old job is cancellable");
        assert_eq!(cancelled.status, RuntimeJobStatus::Cancelled);
        assert_eq!(
            store
                .get_job(&replacement.id)
                .await
                .expect("replacement lookup")
                .expect("replacement job")
                .status,
            RuntimeJobStatus::Queued
        );

        let report = store
            .reconcile_inconsistent_states(10)
            .await
            .expect("bounded reconciliation");
        assert_eq!(report.route_projections_repaired, 0);
        let projection = db
            .query_one_raw(
                raw::statement(
                    db,
                    "SELECT processing_status, processing_error, processing_progress_json, runtime_stage_id, runtime_attempt_token FROM transcription_records WHERE id = ?1",
                    vec![record_id.into()],
                )
                .expect("replacement projection statement"),
            )
            .await
            .expect("replacement projection query")
            .expect("replacement projection");
        assert_eq!(
            projection.try_get_by_index::<String>(0).expect("status"),
            "processing"
        );
        assert_eq!(
            projection
                .try_get_by_index::<Option<String>>(1)
                .expect("error"),
            None
        );
        assert_eq!(
            projection
                .try_get_by_index::<Option<String>>(2)
                .expect("progress")
                .as_deref(),
            Some("{}")
        );
        assert_eq!(
            projection
                .try_get_by_index::<Option<String>>(3)
                .expect("stage binding")
                .as_deref(),
            Some("replacement-stage")
        );
        assert_eq!(
            projection
                .try_get_by_index::<Option<String>>(4)
                .expect("attempt binding")
                .as_deref(),
            Some("replacement-attempt")
        );
    }

    #[tokio::test]
    async fn requested_cancellation_survives_lease_expiry_until_exact_owner_teardown() {
        let (mut store, _root) = build_store();
        let clock = Arc::new(AtomicI64::new(1_000));
        store.set_test_clock(clock.clone());
        let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
        let claimed = store
            .claim_next_stage("worker-1", 10)
            .await
            .expect("claim")
            .expect("active attempt");
        let lease = claimed.lease().expect("lease");
        store
            .cancel_job(&job.id, Some("cancel".to_string()))
            .await
            .expect("cancel")
            .expect("cancellation requested");

        clock.store(2_000, Ordering::SeqCst);
        assert_eq!(
            store
                .recover_expired_stage_leases(DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT)
                .await
                .expect("recovery must skip uncertain execution"),
            0
        );
        assert_eq!(
            store.stage_lease_state(&lease).await.expect("lease state"),
            Some(StageLeaseState::CancellationRequested)
        );
        assert!(store
            .mark_stage_execution_stopping(&lease)
            .await
            .expect("exact owner marks stopping after expiry"));
        assert!(store
            .renew_stage_lease(&lease, 100)
            .await
            .expect("exact owner renews cancellation lease"));
        let stopping = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(
            stopping.cancellation_state,
            Some(RuntimeCancellationState::ExecutionStopping)
        );
        assert_eq!(stopping.lease_expires_at, Some(2_100));
        assert_eq!(stopping.worker_id.as_deref(), Some("worker-1"));
    }

    #[tokio::test]
    async fn completion_and_cancellation_race_has_one_authoritative_winner() {
        let (store, _root) = build_store();
        let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
        let claimed = store
            .claim_next_stage("worker-1", 60_000)
            .await
            .expect("claim")
            .expect("active attempt");
        let lease = claimed.lease().expect("lease");
        let barrier = Arc::new(tokio::sync::Barrier::new(3));

        let completion_store = store.clone();
        let completion_lease = lease.clone();
        let completion_barrier = barrier.clone();
        let completion = tokio::spawn(async move {
            completion_barrier.wait().await;
            completion_store
                .complete_stage(&completion_lease, vec![])
                .await
        });
        let cancellation_store = store.clone();
        let cancellation_job_id = job.id.clone();
        let cancellation_barrier = barrier.clone();
        let cancellation = tokio::spawn(async move {
            cancellation_barrier.wait().await;
            cancellation_store
                .cancel_job(&cancellation_job_id, Some("race".to_string()))
                .await
        });
        barrier.wait().await;

        let completed = completion
            .await
            .expect("completion join")
            .expect("completion");
        let cancelled = cancellation
            .await
            .expect("cancellation join")
            .expect("cancellation");
        assert_ne!(completed.is_some(), cancelled.is_some());

        if cancelled.is_some() {
            assert!(store
                .finalize_stage_cancellation(&lease)
                .await
                .expect("finalize cancellation")
                .is_some());
        }
        let terminal_job = store
            .get_job(&job.id)
            .await
            .expect("job")
            .expect("job exists");
        assert!(matches!(
            terminal_job.status,
            RuntimeJobStatus::Completed | RuntimeJobStatus::Cancelled
        ));
        assert_eq!(terminal_job.cancellation_state, None);
        let terminal_stage = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert!(matches!(
            terminal_stage.status,
            RuntimeStageStatus::Completed | RuntimeStageStatus::Cancelled
        ));
        assert_eq!(terminal_stage.worker_id, None);
        assert_eq!(terminal_stage.lease_expires_at, None);
    }

    #[tokio::test]
    async fn manual_retry_requeues_failed_job_and_stage() {
        let (store, _root) = build_store();
        let job = store
            .create_job(NewRuntimeJob {
                job_kind: RuntimeJobKind::TtsSpeech,
                status: RuntimeJobStatus::Queued,
                priority: 0,
                model_id: Some("Qwen3-TTS-0.6B".to_string()),
                capability: Some("tts".to_string()),
                route_record_kind: Some("text_to_speech".to_string()),
                route_record_id: Some("speech-1".to_string()),
                input_media_asset_id: None,
                input_text_asset_id: None,
                request_json: json!({"text": "hello"}),
                model_snapshot_json: json!({}),
                retry_policy_json: json!({"max_attempts": 1}),
                max_attempts: 1,
                idempotency_key: None,
                correlation_id: None,
            })
            .await
            .expect("job");
        let stage = store
            .create_stage(NewJobStage {
                job_id: job.id.clone(),
                sequence: 0,
                stage_kind: "tts_synthesize".to_string(),
                status: RuntimeStageStatus::Queued,
                capability: Some("tts".to_string()),
                model_id: job.model_id.clone(),
                max_attempts: 2,
                input_artifact_ids: vec![],
            })
            .await
            .expect("stage");

        let claimed = store
            .claim_next_stage("worker-1", 60_000)
            .await
            .expect("claim")
            .expect("stage should be claimed");
        assert_eq!(claimed.stage.id, stage.id);
        let lease = claimed.lease().expect("lease");

        let failed_stage = store
            .fail_stage(
                &lease,
                false,
                Some("boom".to_string()),
                Some("first attempt failed".to_string()),
            )
            .await
            .expect("fail")
            .expect("stage should fail");
        assert_eq!(failed_stage.status, RuntimeStageStatus::Failed);

        let failed_job = store
            .get_job(&job.id)
            .await
            .expect("job")
            .expect("job exists");
        assert_eq!(failed_job.status, RuntimeJobStatus::Failed);

        let retried_job = store
            .retry_job(&job.id)
            .await
            .expect("retry")
            .expect("job should retry");
        assert_eq!(retried_job.status, RuntimeJobStatus::Queued);
        assert_eq!(retried_job.attempt_count, 1);
        assert!(retried_job.error_code.is_none());
        assert!(retried_job.finished_at.is_none());

        let retried_stage = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(retried_stage.status, RuntimeStageStatus::Retrying);
        assert!(retried_stage.error_code.is_none());
        assert!(retried_stage.finished_at.is_none());

        let stages = store
            .list_stages_for_job(&job.id)
            .await
            .expect("list stages");
        assert_eq!(stages.len(), 1);
        let stage_counts = store.stage_status_counts().await.expect("stage counts");
        assert_eq!(
            stage_counts,
            vec![RuntimeStageStatusCount {
                status: RuntimeStageStatus::Retrying,
                count: 1,
            }]
        );

        let second_claim = store
            .claim_next_stage("worker-2", 60_000)
            .await
            .expect("second claim")
            .expect("retried stage should be claimable");
        store
            .fail_stage(
                &second_claim.lease().expect("second lease"),
                false,
                Some("boom-again".to_string()),
                Some("second attempt failed".to_string()),
            )
            .await
            .expect("second failure")
            .expect("stage should fail again");
        assert!(store
            .retry_job(&job.id)
            .await
            .expect("retry budget check")
            .is_none());
    }

    #[tokio::test]
    async fn retry_policy_delays_claims_and_enforces_attempt_budget() {
        let (store, _root) = build_store();
        let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 3).await;
        let db = store.connection().await.expect("database");
        db.execute_raw(
            raw::statement(
                db,
                "UPDATE runtime_jobs SET retry_policy_json = ?1 WHERE id = ?2",
                vec![
                    json!({
                        "max_attempts": 2,
                        "initial_backoff_ms": 1_000,
                        "backoff_multiplier": 2.0,
                        "max_backoff_ms": 10_000
                    })
                    .to_string()
                    .into(),
                    job.id.clone().into(),
                ],
            )
            .expect("retry policy statement"),
        )
        .await
        .expect("retry policy update");

        let first = store
            .claim_next_stage("worker-1", 60_000)
            .await
            .expect("first claim")
            .expect("first attempt");
        let first_token = first.stage.attempt_token.clone();
        let retrying = store
            .fail_stage(
                &first.lease().expect("first lease"),
                true,
                Some("transient".to_string()),
                Some("try again".to_string()),
            )
            .await
            .expect("retry transition")
            .expect("stage should retry");
        assert_eq!(retrying.status, RuntimeStageStatus::Retrying);
        assert!(retrying.available_at.expect("retry eligibility") > retrying.updated_at);
        assert!(store
            .claim_next_stage("worker-2", 60_000)
            .await
            .expect("delayed claim")
            .is_none());

        db.execute_raw(
            raw::statement(
                db,
                "UPDATE job_stages SET available_at = ?1 WHERE id = ?2",
                vec![
                    current_timestamp_millis().saturating_sub(1).into(),
                    stage.id.clone().into(),
                ],
            )
            .expect("release retry statement"),
        )
        .await
        .expect("release retry");
        let second = store
            .claim_next_stage("worker-2", 60_000)
            .await
            .expect("second claim")
            .expect("second attempt");
        assert_ne!(second.stage.attempt_token, first_token);

        let failed = store
            .fail_stage(
                &second.lease().expect("second lease"),
                true,
                Some("transient".to_string()),
                Some("budget exhausted".to_string()),
            )
            .await
            .expect("terminal failure")
            .expect("stage should fail");
        assert_eq!(failed.status, RuntimeStageStatus::Failed);
        assert_eq!(failed.attempt_count, 2);
        assert_eq!(
            store
                .get_job(&job.id)
                .await
                .expect("job")
                .expect("job exists")
                .status,
            RuntimeJobStatus::Failed
        );
    }

    #[tokio::test]
    async fn attempt_token_fences_same_worker_and_attempt_publication() {
        let (store, _root) = build_store();
        let (_job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
        let claimed = store
            .claim_next_stage("worker-1", 60_000)
            .await
            .expect("claim")
            .expect("attempt");
        let lease = claimed.lease().expect("lease");
        assert!(lease.attempt_token.is_some());

        let forged = StageLease {
            attempt_token: Some("wrong-attempt-token".to_string()),
            ..lease.clone()
        };
        assert!(store
            .complete_stage(&forged, vec![])
            .await
            .expect("forged completion")
            .is_none());
        let running = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(running.status, RuntimeStageStatus::Running);
        assert!(running.output_artifact_ids.is_empty());

        let artifact = store
            .publish_stage_output_artifact(&lease, test_stage_output("current-output"))
            .await
            .expect("publish current output")
            .expect("current attempt owns output");
        assert!(store
            .complete_stage(&lease, vec![artifact.id])
            .await
            .expect("owned completion")
            .is_some());
    }

    #[tokio::test]
    async fn attempt_owned_artifact_publication_is_idempotent() {
        let (store, _root) = build_store();
        let (job, _stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
        let claimed = store
            .claim_next_stage("worker-1", 60_000)
            .await
            .expect("claim")
            .expect("attempt");
        let lease = claimed.lease().expect("lease");
        assert!(store
            .stage_lease_is_active(&lease)
            .await
            .expect("active lease"));

        let first = store
            .publish_stage_output_artifact(&lease, test_stage_output("primary-result"))
            .await
            .expect("first publication")
            .expect("active publication");
        let duplicate = store
            .publish_stage_output_artifact(&lease, test_stage_output("primary-result"))
            .await
            .expect("duplicate publication")
            .expect("idempotent publication");

        assert_eq!(duplicate.id, first.id);
        assert_eq!(first.job_id, job.id);
        assert_eq!(first.stage_id.as_deref(), Some(lease.stage_id.as_str()));
        assert_eq!(first.producer_attempt_count, Some(lease.attempt_count));
        assert_eq!(
            first.producer_attempt_token.as_deref(),
            lease.attempt_token.as_deref()
        );
        assert_eq!(first.publication_key.as_deref(), Some("primary-result"));
        let artifacts = store
            .list_artifacts_for_job(&job.id)
            .await
            .expect("artifacts");
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].id, first.id);
    }

    #[tokio::test]
    async fn stage_completion_rejects_unbounded_or_unowned_output_references_atomically() {
        let (store, _root) = build_store();
        let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
        let claimed = store
            .claim_next_stage("worker-a", 60_000)
            .await
            .expect("claim")
            .expect("attempt");
        let lease = claimed.lease().expect("lease");
        let owned = store
            .publish_stage_output_artifact(&lease, test_stage_output("owned-output"))
            .await
            .expect("publish owned output")
            .expect("attempt owns output");

        let too_many = (0..=MAX_STAGE_OUTPUT_ARTIFACTS)
            .map(|_| new_uuid())
            .collect::<Vec<_>>();
        assert!(store
            .complete_stage(&lease, too_many)
            .await
            .unwrap_err()
            .to_string()
            .contains("count exceeds"));
        assert!(store
            .complete_stage(
                &lease,
                vec!["x".repeat(MAX_STAGE_OUTPUT_ARTIFACT_ID_BYTES + 1)],
            )
            .await
            .unwrap_err()
            .to_string()
            .contains("byte limit"));
        assert!(store
            .complete_stage(&lease, vec!["not-a-uuid".to_string()])
            .await
            .unwrap_err()
            .to_string()
            .contains("canonical UUIDs"));
        assert!(store
            .complete_stage(&lease, vec![owned.id.clone(), owned.id.clone()])
            .await
            .unwrap_err()
            .to_string()
            .contains("must be unique"));
        assert!(store
            .complete_stage(&lease, vec![new_uuid()])
            .await
            .unwrap_err()
            .to_string()
            .contains("not owned"));

        let (foreign_job, foreign_stage) =
            create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
        let foreign_claim = store
            .claim_next_stage("worker-b", 60_000)
            .await
            .expect("foreign claim")
            .expect("foreign attempt");
        assert_eq!(foreign_claim.job.id, foreign_job.id);
        let foreign_lease = foreign_claim.lease().expect("foreign lease");
        let foreign = store
            .publish_stage_output_artifact(&foreign_lease, test_stage_output("foreign-output"))
            .await
            .expect("publish foreign output")
            .expect("foreign attempt owns output");
        assert!(store
            .complete_stage(&lease, vec![foreign.id.clone()])
            .await
            .unwrap_err()
            .to_string()
            .contains("not owned"));

        let db = store.db.connection().await.expect("database");
        db.execute_raw(
            raw::statement(
                db,
                r#"
                UPDATE runtime_artifacts
                SET job_id = ?1,
                    producer_attempt_count = ?2,
                    producer_attempt_token = ?3
                WHERE id = ?4
                "#,
                vec![
                    job.id.clone().into(),
                    u32_to_i64_value(lease.attempt_count).into(),
                    lease.attempt_token.clone().unwrap().into(),
                    foreign.id.clone().into(),
                ],
            )
            .expect("cross-stage corruption statement"),
        )
        .await
        .expect("inject cross-stage artifact row");
        assert_ne!(foreign_stage.id, stage.id);
        assert!(store
            .complete_stage(&lease, vec![foreign.id])
            .await
            .unwrap_err()
            .to_string()
            .contains("not owned"));

        let still_running = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(still_running.status, RuntimeStageStatus::Running);
        assert!(still_running.output_artifact_ids.is_empty());
        let completed = store
            .complete_stage(&lease, vec![owned.id.clone()])
            .await
            .expect("valid completion")
            .expect("exact attempt completes");
        assert_eq!(completed.output_artifact_ids, vec![owned.id]);
    }

    #[tokio::test]
    async fn reclaimed_attempt_cannot_adopt_a_previous_attempts_output() {
        let clock = Arc::new(AtomicI64::new(current_timestamp_millis()));
        let (mut store, _root) = build_store();
        store.set_test_clock(clock.clone());
        let (_job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 2).await;
        let first = store
            .claim_next_stage("worker-a", 100)
            .await
            .expect("first claim")
            .expect("first attempt");
        let first_lease = first.lease().expect("first lease");
        let old_output = store
            .publish_stage_output_artifact(&first_lease, test_stage_output("old-output"))
            .await
            .expect("publish old output")
            .expect("first attempt owns output");

        clock.fetch_add(101, Ordering::SeqCst);
        assert_eq!(
            store
                .recover_expired_stage_leases(DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT)
                .await
                .expect("recover first attempt"),
            1
        );
        let second = store
            .claim_next_stage("worker-b", 60_000)
            .await
            .expect("second claim")
            .expect("second attempt");
        let second_lease = second.lease().expect("second lease");
        assert!(store
            .complete_stage(&second_lease, vec![old_output.id])
            .await
            .unwrap_err()
            .to_string()
            .contains("not owned"));
        let running = store
            .get_stage(&stage.id)
            .await
            .expect("stage")
            .expect("stage exists");
        assert_eq!(running.status, RuntimeStageStatus::Running);
        assert!(running.output_artifact_ids.is_empty());

        let current_output = store
            .publish_stage_output_artifact(&second_lease, test_stage_output("current-output"))
            .await
            .expect("publish current output")
            .expect("second attempt owns output");
        store
            .complete_stage(&second_lease, vec![current_output.id])
            .await
            .expect("complete current attempt")
            .expect("current attempt completes");
    }

    #[tokio::test]
    async fn speech_replay_is_cursor_ordered_bounded_and_attempt_fenced() {
        let (store, _root) = build_store();
        let (job, _) = create_test_job_and_stage(&store, 0, "fake_stage", 2).await;
        let claimed = store
            .claim_next_stage("worker-1", 60_000)
            .await
            .unwrap()
            .unwrap();
        let lease = claimed.lease().unwrap();
        // Publish out of insertion order to prove cursor ordering, not timestamps.
        for sequence in [10, 0, 2, 1] {
            let key = super::super::speech_progress::pcm_publication_key(sequence);
            let mut artifact = test_stage_output(&key);
            artifact.artifact_role = RuntimeArtifactRole::OutputIntermediate;
            store
                .publish_stage_output_artifact(&lease, artifact)
                .await
                .unwrap()
                .unwrap();
        }
        store
            .publish_stage_output_artifact(&lease, test_stage_output("primary-result"))
            .await
            .unwrap()
            .unwrap();
        let first = store.speech_pcm_after(&job.id, None, 2).await.unwrap();
        assert_eq!(first.len(), 2);
        assert_eq!(
            first[0].publication_key.as_deref(),
            Some("speech-pcm/00000000000000000000")
        );
        assert_eq!(
            first[1].publication_key.as_deref(),
            Some("speech-pcm/00000000000000000001")
        );
        let remaining = store.speech_pcm_after(&job.id, Some(1), 64).await.unwrap();
        assert_eq!(remaining.len(), 2);
        assert_eq!(
            remaining[0].publication_key.as_deref(),
            Some("speech-pcm/00000000000000000002")
        );
        assert!(store
            .speech_pcm_after("other-job", None, 64)
            .await
            .unwrap()
            .is_empty());
        let forged = StageLease {
            attempt_token: Some("stale".to_string()),
            ..lease.clone()
        };
        assert!(!store
            .update_stage_progress(&forged, json!({"completed_segments": 999}))
            .await
            .unwrap());
        assert!(store
            .publish_stage_output_artifact(
                &forged,
                test_stage_output("speech-pcm/00000000000000000011")
            )
            .await
            .unwrap()
            .is_none());
        store
            .cancel_job(&job.id, Some("stop".to_string()))
            .await
            .unwrap();
        assert!(!store
            .update_stage_progress(&lease, json!({"completed_segments": 999}))
            .await
            .unwrap());
        assert!(store
            .publish_stage_output_artifact(
                &lease,
                test_stage_output("speech-pcm/00000000000000000011")
            )
            .await
            .unwrap()
            .is_none());
        assert_eq!(
            store
                .speech_pcm_after(&job.id, None, 64)
                .await
                .unwrap()
                .len(),
            4
        );
    }

    #[tokio::test]
    async fn stale_and_cancelled_attempts_cannot_publish_artifacts() {
        let (store, _root) = build_store();
        let (job, _stage) = create_test_job_and_stage(&store, 0, "fake_stage", 2).await;
        let first = store
            .claim_next_stage("worker-1", 60_000)
            .await
            .expect("first claim")
            .expect("first attempt");
        let first_lease = first.lease().expect("first lease");
        assert!(store
            .stage_lease_is_active(&first_lease)
            .await
            .expect("first lease active"));

        store
            .fail_stage(
                &first_lease,
                true,
                Some("retry".to_string()),
                Some("replace attempt".to_string()),
            )
            .await
            .expect("retry first attempt")
            .expect("retrying stage");
        let second = store
            .claim_next_stage("worker-2", 60_000)
            .await
            .expect("second claim")
            .expect("replacement attempt");
        let second_lease = second.lease().expect("second lease");

        assert!(!store
            .stage_lease_is_active(&first_lease)
            .await
            .expect("stale lease check"));
        assert!(store
            .stage_lease_is_active(&second_lease)
            .await
            .expect("replacement lease check"));
        assert!(store
            .publish_stage_output_artifact(&first_lease, test_stage_output("stale-result"))
            .await
            .expect("stale publication")
            .is_none());

        store
            .cancel_job(&job.id, Some("cancel active attempt".to_string()))
            .await
            .expect("cancel job")
            .expect("cancelled job");
        assert!(store
            .stage_lease_is_active(&second_lease)
            .await
            .expect("cancelling lease check"));
        assert!(store
            .publish_stage_output_artifact(&second_lease, test_stage_output("cancelled-result"))
            .await
            .expect("cancelled publication")
            .is_none());
        assert!(store
            .mark_stage_execution_stopping(&second_lease)
            .await
            .expect("mark execution stopping"));
        assert!(store
            .finalize_stage_cancellation(&second_lease)
            .await
            .expect("finalize cancellation")
            .is_some());
        assert!(!store
            .stage_lease_is_active(&second_lease)
            .await
            .expect("terminal lease check"));
        assert!(store
            .list_artifacts_for_job(&job.id)
            .await
            .expect("artifacts")
            .is_empty());
    }

    #[tokio::test]
    async fn reconciliation_repairs_crash_interrupted_terminal_transitions() {
        let (store, _root) = build_store();
        let (completed_job, completed_stage) =
            create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
        let claimed = store
            .claim_next_stage("worker-complete", 60_000)
            .await
            .expect("claim")
            .expect("attempt");
        let db = store.connection().await.expect("database");
        db.execute_raw(
            raw::statement(
                db,
                "UPDATE job_stages SET status = 'completed', worker_id = NULL, lease_expires_at = NULL, finished_at = ?1 WHERE id = ?2",
                vec![current_timestamp_millis().into(), claimed.stage.id.into()],
            )
            .expect("crash completion statement"),
        )
        .await
        .expect("crash completion");

        let (cancelled_job, cancelled_stage) =
            create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
        db.execute_raw(
            raw::statement(
                db,
                "UPDATE runtime_jobs SET status = 'cancelled', finished_at = ?1 WHERE id = ?2",
                vec![
                    current_timestamp_millis().into(),
                    cancelled_job.id.clone().into(),
                ],
            )
            .expect("crash cancellation statement"),
        )
        .await
        .expect("crash cancellation");

        let report = store
            .reconcile_inconsistent_states(DEFAULT_RUNTIME_MAINTENANCE_BATCH_LIMIT)
            .await
            .expect("reconciliation");
        assert!(report.jobs_repaired >= 1);
        assert!(report.stages_repaired >= 1);
        assert_eq!(
            store
                .get_job(&completed_job.id)
                .await
                .expect("completed job")
                .expect("completed job exists")
                .status,
            RuntimeJobStatus::Completed
        );
        assert_eq!(
            store
                .get_stage(&completed_stage.id)
                .await
                .expect("completed stage")
                .expect("completed stage exists")
                .status,
            RuntimeStageStatus::Completed
        );
        let cancelled = store
            .get_stage(&cancelled_stage.id)
            .await
            .expect("cancelled stage")
            .expect("cancelled stage exists");
        assert_eq!(cancelled.status, RuntimeStageStatus::Cancelled);
        assert!(cancelled.worker_id.is_none());
        assert!(cancelled.lease_expires_at.is_none());
    }

    #[tokio::test]
    async fn reconciliation_consumes_one_shared_bounded_budget() {
        let (store, _root) = build_store();
        let mut job_ids = Vec::new();
        let db = store.connection().await.expect("database");

        for _ in 0..3 {
            let (job, stage) = create_test_job_and_stage(&store, 0, "fake_stage", 1).await;
            db.execute_raw(
                raw::statement(
                    db,
                    "UPDATE job_stages SET status = 'completed', worker_id = NULL, lease_expires_at = NULL, finished_at = ?1 WHERE id = ?2",
                    vec![current_timestamp_millis().into(), stage.id.into()],
                )
                .expect("crash completion statement"),
            )
            .await
            .expect("crash completion");
            job_ids.push(job.id);
        }

        let first = store
            .reconcile_inconsistent_states(2)
            .await
            .expect("first reconciliation batch");
        assert_eq!(first.jobs_repaired, 2);
        assert_eq!(first.stages_repaired, 0);
        let mut completed = 0;
        for job_id in &job_ids {
            if store
                .get_job(job_id)
                .await
                .expect("job")
                .expect("job exists")
                .status
                == RuntimeJobStatus::Completed
            {
                completed += 1;
            }
        }
        assert_eq!(completed, 2);

        let second = store
            .reconcile_inconsistent_states(2)
            .await
            .expect("second reconciliation batch");
        assert_eq!(second.jobs_repaired, 1);
        assert_eq!(second.stages_repaired, 0);
    }
}
