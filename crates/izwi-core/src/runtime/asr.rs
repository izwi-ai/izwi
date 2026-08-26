//! ASR runtime methods routed through the unified core engine.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};

use crate::backends::state::{PhysicalStateSequenceId, PhysicalStateTransactionId};
use crate::backends::BackendKind;
use crate::catalog::{parse_model_variant, resolve_asr_model_variant, ModelFamily};
use crate::engine::{
    AsrProgress, AsrProgressPhase, EngineCoreRequest, RealtimeAsrSessionHandle, ResourceAmount,
    ResourceVector, RetainedTensorStateRuntimeV2, StreamingOutput, TaskType, WorkCost, WorkUnit,
};
use crate::error::{Error, Result};
use crate::model::{ModelResidencyLease, ModelVariant};
use crate::models::architectures::granite_speech::asr::{
    parse_granite_speech_output, GraniteSpeechTask,
};
use crate::models::architectures::voxtral::realtime::{
    VoxtralRealtimePreparationBatchGeometry, VoxtralRealtimePreparationGeometry,
    VoxtralRealtimePreparationMode, VoxtralRealtimePreparationStageSeal,
    VoxtralRealtimePreparedResourceUsage,
};
use crate::models::registry::{
    NativeAsrGenerationOptions, NativeAsrModel, NativeAsrRealtimeEvent,
    NativeAsrRealtimeResourceReservation, NativeAsrRealtimeState, NativeAsrTranscription,
    VoxtralModelLease,
};
use crate::runtime::adapters::{
    CapabilityKind, ExecutionTargetKind, LoadedCapabilityBinding, LoadedExecutionContract,
};
use crate::runtime::audio_io::{
    base64_decode, decode_audio_bytes, validate_base64_audio_retained_size,
    validate_base64_audio_source_size, MAX_AUDIO_SOURCE_BYTES,
};
use crate::runtime::coordinator::{InferenceCoordinator, JobLease, JobResourceObservation};
use crate::runtime::request::AsrRuntimeRequest;
#[cfg(test)]
use crate::runtime::service::retained_engine_request_input_bytes;
use crate::runtime::service::{
    copy_optional_preparation_string, copy_preparation_bytes, copy_preparation_string,
    AdmittedEngineRequest, RuntimeService,
};
use crate::runtime::types::{
    AsrTranscription, RuntimeRequestContext, SpeakerAttributedAsrResult,
    SpeakerAttributedAsrStatus, SpeakerAttributedAsrTurn,
};
use crate::runtime::CoordinatorLane;
use izwi_asr_toolkit::{plan_audio_chunks, AsrLongFormConfig, AudioChunk};

#[derive(Clone, Copy)]
enum AsrAudioInput<'a> {
    Base64(&'a str),
    Bytes(&'a [u8]),
}

enum OwnedAsrAudioInput {
    Base64(String),
    Bytes(Vec<u8>),
}

impl AsrAudioInput<'_> {
    fn input_bytes(self) -> usize {
        match self {
            Self::Base64(audio) => audio.len(),
            Self::Bytes(audio) => audio.len(),
        }
    }

    fn validate_retained_size(self) -> Result<()> {
        match self {
            Self::Base64(audio) => {
                validate_base64_audio_retained_size(audio.len(), MAX_AUDIO_SOURCE_BYTES)
            }
            Self::Bytes([]) => Err(Error::InvalidInput(
                "ASR request missing audio bytes".to_string(),
            )),
            Self::Bytes(audio) if audio.len() > MAX_AUDIO_SOURCE_BYTES => {
                Err(Error::InvalidInput(format!(
                    "ASR encoded audio is {} bytes, exceeding the {MAX_AUDIO_SOURCE_BYTES}-byte source limit",
                    audio.len()
                )))
            }
            Self::Bytes(_) => Ok(()),
        }
    }

    async fn into_owned_for_job(self, job: &JobLease) -> Result<OwnedAsrAudioInput> {
        match self {
            Self::Base64(audio) => Ok(OwnedAsrAudioInput::Base64(
                copy_preparation_string(job, audio, "ASR base64 audio").await?,
            )),
            Self::Bytes(audio) => Ok(OwnedAsrAudioInput::Bytes(
                copy_preparation_bytes(job, audio, "ASR encoded audio").await?,
            )),
        }
    }
}

impl OwnedAsrAudioInput {
    fn retained_bytes(&self) -> usize {
        match self {
            Self::Base64(audio) => audio.capacity(),
            Self::Bytes(audio) => audio.capacity(),
        }
    }

    fn decode(&self) -> Result<(Vec<f32>, u32)> {
        match self {
            Self::Base64(audio) => {
                let audio = base64_decode(audio)?;
                decode_audio_bytes(&audio)
            }
            Self::Bytes(audio) => decode_audio_bytes(audio),
        }
    }

    fn validate_source_size(&self) -> Result<()> {
        match self {
            Self::Base64(audio) => {
                validate_base64_audio_source_size(audio, MAX_AUDIO_SOURCE_BYTES)
            }
            Self::Bytes(audio) if audio.len() > MAX_AUDIO_SOURCE_BYTES => {
                Err(Error::InvalidInput(format!(
                    "ASR encoded audio is {} bytes, exceeding the {MAX_AUDIO_SOURCE_BYTES}-byte source limit",
                    audio.len()
                )))
            }
            Self::Bytes(_) => Ok(()),
        }
    }
}

const GRANITE_ASR_AUTO_MIN_TOKENS: usize = 76;
const GRANITE_ASR_AUTO_MAX_TOKENS: usize = 2048;
const GRANITE_ASR_AUTO_BASE_SECONDS: f32 = 28.0;
const GRANITE_ASR_AUTO_TOKENS_PER_SECOND: f32 = 4.0;
const GRANITE_SAA_MIN_NEW_TOKENS: usize = 800;
const GRANITE_SAA_MAX_NEW_TOKENS: usize = 2_500;
const GRANITE_SAA_NEW_TOKENS_PER_SECOND: f32 = 8.0;
const GRANITE_SAA_NEW_TOKEN_RESERVE: usize = 256;
const GRANITE_SAA_TARGET_CHUNK_SECS: f32 = 240.0;
const GRANITE_SAA_HARD_MAX_CHUNK_SECS: f32 = 510.0;
const GRANITE_SAA_OVERLAP_SECS: f32 = 12.0;
const GRANITE_SAA_MIN_CHUNK_SECS: f32 = 30.0;
const GRANITE_SAA_SILENCE_SEARCH_SECS: f32 = 12.0;
const GRANITE_SAA_PREFIX_MAX_TURNS: usize = 12;
const GRANITE_SAA_PREFIX_MAX_CHARS: usize = 6_000;
const GRANITE_SAA_MIN_OVERLAP_WORDS: usize = 4;
const GRANITE_SAA_MAX_OVERLAP_WORDS: usize = 80;
const UNKNOWN_SAA_SPEAKER: &str = "UNKNOWN";
const ASR_REALTIME_MAX_SESSIONS_ENV: &str = "IZWI_ASR_REALTIME_MAX_SESSIONS";
const ASR_REALTIME_MAX_LIFETIME_SECS_ENV: &str = "IZWI_ASR_REALTIME_MAX_LIFETIME_SECS";
const ASR_REALTIME_IDLE_TIMEOUT_SECS_ENV: &str = "IZWI_ASR_REALTIME_IDLE_TIMEOUT_SECS";
const DEFAULT_ASR_REALTIME_MAX_SESSIONS: usize = 16;
const DEFAULT_ASR_REALTIME_MAX_LIFETIME_SECS: u64 = 10 * 60;
const DEFAULT_ASR_REALTIME_IDLE_TIMEOUT_SECS: u64 = 30;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraniteSaaPrefixMode {
    None,
    FullTranscript,
}

impl GraniteSaaPrefixMode {
    fn from_env() -> Self {
        match std::env::var("IZWI_GRANITE_SAA_PREFIX_MODE")
            .ok()
            .map(|raw| raw.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("full") | Some("full_transcript") | Some("legacy") => Self::FullTranscript,
            _ => Self::None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::FullTranscript => "full_transcript",
        }
    }
}

fn resolve_asr_realtime_stream_variant(model_id: Option<&str>) -> Option<ModelVariant> {
    let variant = resolve_asr_model_variant(model_id);
    matches!(
        variant.family(),
        ModelFamily::NemotronAsr | ModelFamily::Voxtral
    )
    .then_some(variant)
}

pub(crate) fn granite_auto_asr_max_tokens_for_duration(audio_seconds: f32) -> usize {
    let duration_budget =
        if audio_seconds.is_finite() && audio_seconds > GRANITE_ASR_AUTO_BASE_SECONDS {
            ((audio_seconds - GRANITE_ASR_AUTO_BASE_SECONDS) * GRANITE_ASR_AUTO_TOKENS_PER_SECOND)
                .ceil() as usize
        } else {
            0
        };
    GRANITE_ASR_AUTO_MIN_TOKENS
        .saturating_add(duration_budget)
        .clamp(GRANITE_ASR_AUTO_MIN_TOKENS, GRANITE_ASR_AUTO_MAX_TOKENS)
}

fn granite_auto_asr_token_ceiling(variant: ModelVariant) -> Option<usize> {
    // The executor already derives the exact per-chunk budget after its
    // admitted blocking audio decode. Reserving the production ceiling here
    // avoids decoding compressed public input synchronously on a Tokio worker
    // merely to construct the engine request.
    (variant.family() == ModelFamily::GraniteSpeechAsr).then_some(GRANITE_ASR_AUTO_MAX_TOKENS)
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RealtimeAsrSessionLimits {
    max_sessions: usize,
    max_lifetime: Duration,
    idle_timeout: Duration,
}

impl RealtimeAsrSessionLimits {
    fn from_env() -> Result<Self> {
        Ok(Self {
            max_sessions: positive_usize_env_or_default(
                ASR_REALTIME_MAX_SESSIONS_ENV,
                DEFAULT_ASR_REALTIME_MAX_SESSIONS,
            )?,
            max_lifetime: Duration::from_secs(positive_u64_env_or_default(
                ASR_REALTIME_MAX_LIFETIME_SECS_ENV,
                DEFAULT_ASR_REALTIME_MAX_LIFETIME_SECS,
            )?),
            idle_timeout: Duration::from_secs(positive_u64_env_or_default(
                ASR_REALTIME_IDLE_TIMEOUT_SECS_ENV,
                DEFAULT_ASR_REALTIME_IDLE_TIMEOUT_SECS,
            )?),
        })
    }
}

#[derive(Clone)]
pub(super) struct RealtimeAsrSessionPolicy {
    limits: RealtimeAsrSessionLimits,
    permits: Arc<Semaphore>,
}

impl RealtimeAsrSessionPolicy {
    pub(super) fn from_env() -> Result<Self> {
        Self::new(RealtimeAsrSessionLimits::from_env()?)
    }

    fn new(limits: RealtimeAsrSessionLimits) -> Result<Self> {
        if limits.max_sessions > Semaphore::MAX_PERMITS {
            return Err(Error::ConfigError(format!(
                "{ASR_REALTIME_MAX_SESSIONS_ENV} cannot exceed {}",
                Semaphore::MAX_PERMITS
            )));
        }
        if Instant::now().checked_add(limits.max_lifetime).is_none() {
            return Err(Error::ConfigError(format!(
                "{ASR_REALTIME_MAX_LIFETIME_SECS_ENV} exceeds the platform clock range"
            )));
        }
        Ok(Self {
            permits: Arc::new(Semaphore::new(limits.max_sessions)),
            limits,
        })
    }

    pub(super) fn retained_sequence_capacity(&self) -> Result<u32> {
        u32::try_from(self.limits.max_sessions).map_err(|_| {
            Error::ConfigError(
                "realtime ASR session quota exceeds physical sequence capacity".into(),
            )
        })
    }

    fn try_acquire(&self) -> Result<Arc<RealtimeAsrSessionLease>> {
        let permit = self
            .permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error::Overloaded("realtime ASR session quota is full".to_string()))?;
        Ok(Arc::new(RealtimeAsrSessionLease { _permit: permit }))
    }
}

struct RealtimeAsrSessionLease {
    _permit: OwnedSemaphorePermit,
}

fn positive_usize_env_or_default(name: &str, default: usize) -> Result<usize> {
    match std::env::var(name) {
        Ok(raw) => raw
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| Error::ConfigError(format!("{name} must be a positive integer"))),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(std::env::VarError::NotUnicode(_)) => Err(Error::ConfigError(format!(
            "{name} must contain valid UTF-8"
        ))),
    }
}

fn positive_u64_env_or_default(name: &str, default: u64) -> Result<u64> {
    match std::env::var(name) {
        Ok(raw) => raw
            .trim()
            .parse::<u64>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| Error::ConfigError(format!("{name} must be a positive integer"))),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(std::env::VarError::NotUnicode(_)) => Err(Error::ConfigError(format!(
            "{name} must contain valid UTF-8"
        ))),
    }
}

struct RealtimeAsrPhysicalSession {
    runtime: Arc<RetainedTensorStateRuntimeV2>,
    sequence: PhysicalStateSequenceId,
    active_transaction: Option<PhysicalStateTransactionId>,
    revision: u64,
}

impl RealtimeAsrPhysicalSession {
    fn new(runtime: Arc<RetainedTensorStateRuntimeV2>) -> Result<Self> {
        let sequence = runtime.register_sequence()?;
        Ok(Self {
            runtime,
            sequence,
            active_transaction: None,
            revision: 0,
        })
    }

    fn begin(&mut self) -> Result<PhysicalStateTransactionId> {
        if self.active_transaction.is_some() {
            return Err(Error::InferenceError(
                "realtime ASR physical transaction is already active".into(),
            ));
        }
        let transaction = self.runtime.begin_transaction(self.sequence)?;
        self.active_transaction = Some(transaction);
        Ok(transaction)
    }

    fn commit(&mut self, transaction: PhysicalStateTransactionId, revision: u64) -> Result<()> {
        if self.active_transaction != Some(transaction) {
            return Err(Error::InferenceError(
                "realtime ASR physical transaction identity changed".into(),
            ));
        }
        self.runtime.commit_transaction(transaction, revision)?;
        self.active_transaction = None;
        self.revision = revision;
        Ok(())
    }

    fn abort(&mut self) {
        if let Some(transaction) = self.active_transaction.take() {
            let _ = self.runtime.abort_transaction(transaction);
        }
    }
}

impl Drop for RealtimeAsrPhysicalSession {
    fn drop(&mut self) {
        self.abort();
        if let Err(error) = self.runtime.release_sequence(self.sequence) {
            tracing::error!(
                sequence = self.sequence.get(),
                error = %error,
                "failed to release realtime ASR physical sequence"
            );
        }
    }
}

struct RuntimeAsrRealtimeState {
    native: NativeAsrRealtimeState,
    physical: RealtimeAsrPhysicalSession,
}

impl RuntimeAsrRealtimeState {
    fn new(
        model: &NativeAsrModel,
        native: NativeAsrRealtimeState,
        runtime: Arc<RetainedTensorStateRuntimeV2>,
    ) -> Result<Self> {
        let mut owner = Self {
            native,
            physical: RealtimeAsrPhysicalSession::new(runtime)?,
        };
        let transaction = owner.physical.begin()?;
        let revision = 1;
        let result = model.stage_realtime_physical_state(
            &mut owner.native,
            &owner.physical.runtime,
            transaction,
            revision,
        );
        if let Err(error) = result {
            owner.physical.abort();
            return Err(error);
        }
        if let Err(error) = owner.physical.commit(transaction, revision) {
            owner.physical.abort();
            return Err(error);
        }
        Ok(owner)
    }

    fn transact<T>(
        &mut self,
        model: &NativeAsrModel,
        operation: impl FnOnce(&mut NativeAsrRealtimeState) -> Result<T>,
    ) -> Result<T> {
        let transaction = self.physical.begin()?;
        let result = (|| {
            model.hydrate_realtime_physical_state(
                &mut self.native,
                &self.physical.runtime,
                transaction,
            )?;
            let output = operation(&mut self.native)?;
            let revision = self.physical.revision.checked_add(1).ok_or_else(|| {
                Error::InferenceError("realtime ASR physical revision overflowed".into())
            })?;
            model.stage_realtime_physical_state(
                &mut self.native,
                &self.physical.runtime,
                transaction,
                revision,
            )?;
            self.physical.commit(transaction, revision)?;
            Ok(output)
        })();
        if result.is_err() {
            if let Err(error) = model.clear_realtime_tensor_handles(&mut self.native) {
                tracing::error!(
                    error = %error,
                    "failed to drain Nemotron tensor handles after aborted operation"
                );
            }
            self.physical.abort();
        }
        result
    }
}

struct RuntimeAsrRealtimeResources {
    model: Option<Arc<NativeAsrModel>>,
    state: Option<Arc<StdMutex<RuntimeAsrRealtimeState>>>,
    execution_contract: Option<LoadedExecutionContract>,
    residency_lease: Option<ModelResidencyLease>,
    job: Option<JobLease>,
    session_lease: Option<Arc<RealtimeAsrSessionLease>>,
    engine_session: Option<RuntimeVoxtralRealtimeSession>,
    voxtral_model: Option<VoxtralModelLease>,
    absolute_deadline: Instant,
    idle_timeout: Duration,
    last_activity: Instant,
    active_operations: usize,
    closed: bool,
    timeout_reason: Option<&'static str>,
}

impl RuntimeAsrRealtimeResources {
    fn idle_deadline(&self) -> Instant {
        self.last_activity
            .checked_add(self.idle_timeout)
            .unwrap_or(self.absolute_deadline)
            .min(self.absolute_deadline)
    }

    fn next_watchdog_deadline(&self) -> Option<Instant> {
        if self.closed {
            None
        } else if self.active_operations > 0 {
            Some(self.absolute_deadline)
        } else {
            Some(self.idle_deadline())
        }
    }

    fn expiration(&self, now: Instant) -> Option<&'static str> {
        realtime_asr_session_expiration(
            now,
            self.absolute_deadline,
            self.idle_deadline(),
            self.active_operations,
        )
    }

    fn detach(&mut self) -> Option<RealtimeAsrDetachedResources> {
        if self.closed {
            return None;
        }
        self.closed = true;
        Some(RealtimeAsrDetachedResources {
            state: self.state.take(),
            model: self.model.take(),
            execution_contract: self.execution_contract.take(),
            residency_lease: self.residency_lease.take(),
            job: self.job.take(),
            session_lease: self.session_lease.take(),
            engine_session: self.engine_session.take(),
            voxtral_model: self.voxtral_model.take(),
        })
    }

    fn detach_expired(&mut self, reason: &'static str) -> Option<RealtimeAsrDetachedResources> {
        self.timeout_reason = Some(reason);
        self.detach()
    }

    fn closed_error(&self) -> Error {
        match self.timeout_reason {
            Some(reason) => Error::Timeout(reason.to_string()),
            None => Error::InvalidInput("realtime ASR stream is closed".to_string()),
        }
    }
}

struct RealtimeAsrDetachedResources {
    state: Option<Arc<StdMutex<RuntimeAsrRealtimeState>>>,
    model: Option<Arc<NativeAsrModel>>,
    execution_contract: Option<LoadedExecutionContract>,
    residency_lease: Option<ModelResidencyLease>,
    job: Option<JobLease>,
    session_lease: Option<Arc<RealtimeAsrSessionLease>>,
    engine_session: Option<RuntimeVoxtralRealtimeSession>,
    voxtral_model: Option<VoxtralModelLease>,
}

impl RealtimeAsrDetachedResources {
    fn release(mut self) {
        // Physical allocations must disappear before their immutable
        // authorization and quota are released for another session.
        self.state.take();
        self.model.take();
        self.execution_contract.take();
        self.residency_lease.take();
        self.job.take();
        self.session_lease.take();
        self.engine_session.take();
        self.voxtral_model.take();
    }
}

/// Fail-closed owner for detached Runtime and Engine authorities. If the
/// asynchronous cleanup future is cancelled or its runtime is torn down,
/// dropping this guard deliberately retains every authority. Only an exact
/// Engine cleanup confirmation may disarm it.
struct RealtimeAsrCleanupGuard {
    cleanup: Option<RealtimeAsrDetachedResources>,
    engine_session: Option<RuntimeVoxtralRealtimeSession>,
    disarmed: bool,
}

impl RealtimeAsrCleanupGuard {
    fn new(
        cleanup: RealtimeAsrDetachedResources,
        engine_session: RuntimeVoxtralRealtimeSession,
    ) -> Self {
        Self {
            cleanup: Some(cleanup),
            engine_session: Some(engine_session),
            disarmed: false,
        }
    }

    fn engine_session(&self) -> &RuntimeVoxtralRealtimeSession {
        self.engine_session
            .as_ref()
            .expect("armed realtime cleanup guard must retain its Engine session")
    }

    #[cfg(test)]
    fn for_test(cleanup: RealtimeAsrDetachedResources) -> Self {
        Self {
            cleanup: Some(cleanup),
            engine_session: None,
            disarmed: false,
        }
    }

    fn release_confirmed(mut self) {
        self.disarmed = true;
        self.engine_session.take();
        if let Some(cleanup) = self.cleanup.take() {
            cleanup.release();
        }
    }
}

impl Drop for RealtimeAsrCleanupGuard {
    fn drop(&mut self) {
        if self.disarmed {
            return;
        }
        if let Some(engine_session) = self.engine_session.take() {
            std::mem::forget(engine_session);
        }
        if let Some(cleanup) = self.cleanup.take() {
            std::mem::forget(cleanup);
        }
    }
}

fn schedule_realtime_asr_cleanup(cleanup: Option<RealtimeAsrDetachedResources>) {
    let Some(mut cleanup) = cleanup else {
        return;
    };
    if let Some(engine_session) = cleanup.engine_session.take() {
        let guard = RealtimeAsrCleanupGuard::new(cleanup, engine_session);
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let cleanup_receipt = match guard
                    .engine_session()
                    .engine
                    .cleanup_realtime_asr_session(&guard.engine_session().handle)
                    .await
                {
                    Ok(receipt) => receipt,
                    Err(error) => {
                        tracing::error!(
                            request_id = %guard.engine_session().handle.request_id(),
                            error = %error,
                            "Engine realtime ASR cleanup receipt could not be created; retaining authorities"
                        );
                        return;
                    }
                };
                if let Err(error) = cleanup_receipt.confirmed().await {
                    tracing::error!(
                        request_id = %guard.engine_session().handle.request_id(),
                        error = %error,
                        "Engine realtime ASR cleanup was not confirmed; retaining authorities"
                    );
                    return;
                }
                if let Ok(runtime) = tokio::runtime::Handle::try_current() {
                    runtime.spawn_blocking(move || guard.release_confirmed());
                } else {
                    guard.release_confirmed();
                }
            });
            return;
        }
        tracing::error!(
            request_id = %guard.engine_session().handle.request_id(),
            "cannot asynchronously abort Engine realtime ASR session without a Tokio runtime"
        );
        drop(guard);
        return;
    }
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        handle.spawn_blocking(move || cleanup.release());
    } else {
        cleanup.release();
    }
}

struct RealtimeAsrOperationHandles {
    model: Arc<NativeAsrModel>,
    state: Arc<StdMutex<RuntimeAsrRealtimeState>>,
    execution_contract: LoadedExecutionContract,
    job: JobLease,
    session_lease: Arc<RealtimeAsrSessionLease>,
    _guard: RealtimeAsrOperationGuard,
}

#[derive(Clone)]
struct RuntimeVoxtralRealtimeSession {
    engine: Arc<crate::engine::Engine>,
    handle: RealtimeAsrSessionHandle,
    model: VoxtralModelLease,
    retained_metadata_bytes: u64,
    max_output_steps: usize,
    max_cache_append: usize,
}

struct RealtimeAsrEngineOperationHandles {
    session: RuntimeVoxtralRealtimeSession,
    job: JobLease,
    session_lease: Arc<RealtimeAsrSessionLease>,
    _guard: RealtimeAsrOperationGuard,
}

struct RealtimeAsrOperationGuard {
    resources: Arc<StdMutex<RuntimeAsrRealtimeResources>>,
    activity: Arc<Notify>,
}

impl Drop for RealtimeAsrOperationGuard {
    fn drop(&mut self) {
        if let Ok(mut resources) = self.resources.lock() {
            resources.active_operations = resources.active_operations.saturating_sub(1);
            if !resources.closed {
                resources.last_activity = Instant::now();
            }
        }
        self.activity.notify_one();
    }
}

pub struct RuntimeAsrRealtimeStream {
    variant: ModelVariant,
    resources: Arc<StdMutex<RuntimeAsrRealtimeResources>>,
    activity: Arc<Notify>,
    operation_gate: Arc<tokio::sync::Mutex<()>>,
    max_samples: usize,
    committed_samples: usize,
    engine_sample_rate: Option<u32>,
    engine_text: String,
    engine_chunk_index: usize,
}

impl RuntimeAsrRealtimeStream {
    fn begin_engine_operation(
        &mut self,
        refresh_activity: bool,
    ) -> Result<RealtimeAsrEngineOperationHandles> {
        let now = Instant::now();
        let mut resources = self.resources.lock().map_err(|_| {
            Error::InferenceError("realtime ASR resource mutex poisoned".to_string())
        })?;
        if resources.closed {
            return Err(resources.closed_error());
        }
        if let Some(reason) = resources.expiration(now) {
            let cleanup = resources.detach_expired(reason);
            drop(resources);
            schedule_realtime_asr_cleanup(cleanup);
            return Err(Error::Timeout(reason.to_string()));
        }
        let session = resources.engine_session.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("Engine realtime ASR session is unavailable".into())
        })?;
        let job = resources.job.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("realtime ASR stream job is unavailable".into())
        })?;
        let session_lease = resources.session_lease.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("realtime ASR session lease is unavailable".into())
        })?;
        resources.active_operations = resources
            .active_operations
            .checked_add(1)
            .ok_or_else(|| Error::Overloaded("realtime ASR operation count overflowed".into()))?;
        if refresh_activity {
            resources.last_activity = now;
        }
        drop(resources);
        if refresh_activity {
            self.activity.notify_one();
        }
        Ok(RealtimeAsrEngineOperationHandles {
            session,
            job,
            session_lease,
            _guard: RealtimeAsrOperationGuard {
                resources: self.resources.clone(),
                activity: self.activity.clone(),
            },
        })
    }

    fn map_engine_outputs(
        &mut self,
        outputs: Vec<StreamingOutput>,
    ) -> Vec<RuntimeAsrRealtimeEvent> {
        outputs
            .into_iter()
            .map(|output| {
                let delta = output.text.unwrap_or_default();
                self.engine_text.push_str(&delta);
                let chunk_index = self.engine_chunk_index;
                self.engine_chunk_index = self.engine_chunk_index.saturating_add(1);
                RuntimeAsrRealtimeEvent {
                    delta,
                    text: self.engine_text.clone(),
                    is_final: output.is_final,
                    chunk_index,
                }
            })
            .collect()
    }
    fn begin_operation(&mut self, refresh_activity: bool) -> Result<RealtimeAsrOperationHandles> {
        let now = Instant::now();
        let mut resources = self.resources.lock().map_err(|_| {
            Error::InferenceError("realtime ASR resource mutex poisoned".to_string())
        })?;
        if resources.closed {
            return Err(resources.closed_error());
        }
        if let Some(reason) = resources.expiration(now) {
            let cleanup = resources.detach_expired(reason);
            drop(resources);
            schedule_realtime_asr_cleanup(cleanup);
            return Err(Error::Timeout(reason.to_string()));
        }
        let model = resources.model.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("realtime ASR stream model is unavailable".to_string())
        })?;
        let state = resources.state.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("realtime ASR stream state is unavailable".to_string())
        })?;
        let execution_contract =
            resources
                .execution_contract
                .as_ref()
                .cloned()
                .ok_or_else(|| {
                    Error::InferenceError(
                        "realtime ASR stream execution contract is unavailable".to_string(),
                    )
                })?;
        let job = resources.job.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("realtime ASR stream job is unavailable".to_string())
        })?;
        let session_lease = resources.session_lease.as_ref().cloned().ok_or_else(|| {
            Error::InferenceError("realtime ASR session lease is unavailable".to_string())
        })?;
        let active_operations = resources.active_operations.checked_add(1).ok_or_else(|| {
            Error::Overloaded("realtime ASR operation count overflowed".to_string())
        })?;
        if refresh_activity {
            resources.last_activity = now;
        }
        resources.active_operations = active_operations;
        drop(resources);
        if refresh_activity {
            self.activity.notify_one();
        }
        Ok(RealtimeAsrOperationHandles {
            model: model.clone(),
            state,
            execution_contract,
            job,
            session_lease,
            _guard: RealtimeAsrOperationGuard {
                resources: self.resources.clone(),
                activity: self.activity.clone(),
            },
        })
    }

    fn ensure_open(&mut self) -> Result<()> {
        let now = Instant::now();
        let mut resources = self.resources.lock().map_err(|_| {
            Error::InferenceError("realtime ASR resource mutex poisoned".to_string())
        })?;
        if resources.closed {
            return Err(resources.closed_error());
        }
        if let Some(reason) = resources.expiration(now) {
            let cleanup = resources.detach_expired(reason);
            drop(resources);
            schedule_realtime_asr_cleanup(cleanup);
            return Err(Error::Timeout(reason.to_string()));
        }
        Ok(())
    }

    fn close(&self) {
        let cleanup = self
            .resources
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .detach();
        schedule_realtime_asr_cleanup(cleanup);
        self.activity.notify_one();
    }

    fn close_due_to_timeout(&self) {
        let mut resources = self
            .resources
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let reason = resources
            .expiration(Instant::now())
            .unwrap_or("realtime ASR operation timed out");
        let cleanup = resources.detach_expired(reason);
        drop(resources);
        schedule_realtime_asr_cleanup(cleanup);
        self.activity.notify_one();
    }
}

impl Drop for RuntimeAsrRealtimeStream {
    fn drop(&mut self) {
        self.close();
    }
}

fn realtime_asr_session_expiration(
    now: Instant,
    absolute_deadline: Instant,
    idle_deadline: Instant,
    active_operations: usize,
) -> Option<&'static str> {
    if now >= absolute_deadline {
        Some("realtime ASR stream exceeded its absolute lifetime")
    } else if active_operations == 0 && now >= idle_deadline {
        Some("realtime ASR stream exceeded its idle timeout")
    } else {
        None
    }
}

fn spawn_realtime_asr_watchdog(
    resources: &Arc<StdMutex<RuntimeAsrRealtimeResources>>,
    activity: Arc<Notify>,
) {
    let weak_resources = Arc::downgrade(resources);
    tokio::spawn(async move {
        loop {
            // Register before reading the deadline so activity between the
            // state snapshot and select cannot be lost.
            let notified = activity.notified();
            let Some(resources) = weak_resources.upgrade() else {
                return;
            };
            let deadline = match resources.lock() {
                Ok(resources) => resources.next_watchdog_deadline(),
                Err(_) => return,
            };
            drop(resources);
            let Some(deadline) = deadline else {
                return;
            };

            tokio::select! {
                _ = tokio::time::sleep_until(deadline.into()) => {
                    let Some(resources) = weak_resources.upgrade() else {
                        return;
                    };
                    let mut resources = match resources.lock() {
                        Ok(resources) => resources,
                        Err(_) => return,
                    };
                    if let Some(reason) = resources.expiration(Instant::now()) {
                        let cleanup = resources.detach_expired(reason);
                        drop(resources);
                        schedule_realtime_asr_cleanup(cleanup);
                        return;
                    }
                }
                _ = notified => {}
            }
        }
    });
}

#[derive(Debug, Clone)]
pub struct RuntimeAsrRealtimeEvent {
    pub delta: String,
    pub text: String,
    pub is_final: bool,
    pub chunk_index: usize,
}

fn map_native_realtime_events(events: Vec<NativeAsrRealtimeEvent>) -> Vec<RuntimeAsrRealtimeEvent> {
    events
        .into_iter()
        .map(|event| RuntimeAsrRealtimeEvent {
            delta: event.delta,
            text: event.text,
            is_final: event.is_final,
            chunk_index: event.chunk_index,
        })
        .collect()
}

fn realtime_state_observation(state: &NativeAsrRealtimeState) -> Result<JobResourceObservation> {
    let (host_bytes, tensor_bytes) = state.resource_usage().ok_or_else(|| {
        Error::InferenceError("realtime ASR state resource usage is unavailable".to_string())
    })?;
    Ok(JobResourceObservation::new(host_bytes, tensor_bytes))
}

fn realtime_state_with_input_observation(
    state: &NativeAsrRealtimeState,
    input_sample_capacity: usize,
) -> Result<JobResourceObservation> {
    let mut observation = realtime_state_observation(state)?;
    let input_bytes = input_sample_capacity
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| Error::Overloaded("realtime ASR input storage overflowed".to_string()))?;
    observation.host_bytes = observation
        .host_bytes
        .checked_add(input_bytes)
        .ok_or_else(|| Error::Overloaded("realtime ASR host usage overflowed".to_string()))?;
    Ok(observation)
}

fn host_input_observation(input_bytes: usize) -> Result<JobResourceObservation> {
    Ok(JobResourceObservation::host(
        u64::try_from(input_bytes)
            .map_err(|_| Error::InvalidInput("runtime input size exceeds u64".to_string()))?,
    ))
}

fn retained_host_observation(parts: &[usize]) -> Result<JobResourceObservation> {
    let bytes = parts.iter().try_fold(0usize, |total, bytes| {
        total
            .checked_add(*bytes)
            .ok_or_else(|| Error::Overloaded("retained ASR storage overflowed".to_string()))
    })?;
    host_input_observation(bytes)
}

fn add_retained_host_bytes(
    mut observation: JobResourceObservation,
    retained_bytes: usize,
) -> Result<JobResourceObservation> {
    let retained_bytes = u64::try_from(retained_bytes)
        .map_err(|_| Error::Overloaded("retained ASR storage exceeds u64".to_string()))?;
    observation.host_bytes = observation
        .host_bytes
        .checked_add(retained_bytes)
        .ok_or_else(|| Error::Overloaded("retained ASR storage overflowed".to_string()))?;
    Ok(observation)
}

fn decoded_audio_observation(
    input_bytes: usize,
    sample_capacity: usize,
) -> Result<JobResourceObservation> {
    let sample_bytes = sample_capacity
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|bytes| bytes.checked_add(input_bytes))
        .ok_or_else(|| Error::Overloaded("decoded audio storage overflowed".to_string()))?;
    host_input_observation(sample_bytes)
}

fn decoded_audio_with_scratch_observation(
    input_bytes: usize,
    sample_count: usize,
    scratch_sample_count: usize,
) -> Result<JobResourceObservation> {
    let samples = sample_count
        .checked_add(scratch_sample_count)
        .ok_or_else(|| Error::Overloaded("ASR sample storage overflowed".to_string()))?;
    decoded_audio_observation(input_bytes, samples)
}

fn validate_realtime_input_copy(samples: usize, max_samples: usize) -> Result<()> {
    if samples > max_samples {
        return Err(Error::InvalidInput(format!(
            "Realtime ASR input chunk of {samples} samples exceeds the stream reservation of {max_samples} samples"
        )));
    }
    Ok(())
}

async fn run_realtime_blocking_operation<T, B, F>(
    coordinator: &Arc<InferenceCoordinator>,
    job: &JobLease,
    execution_contract: LoadedExecutionContract,
    operation_gate: Arc<tokio::sync::Mutex<()>>,
    operation_kind: &'static str,
    build_operation: B,
) -> Result<T>
where
    T: Send + 'static,
    B: FnOnce() -> Result<F> + Send,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    let operation_gate = coordinator
        .acquire_job_ordering(job, operation_gate)
        .await?;
    let operation = build_operation()?;
    coordinator
        .run_loaded_blocking_stage(
            job,
            execution_contract,
            WorkUnit::AtomicJob {
                kind: operation_kind.to_string(),
            },
            move || {
                let _operation_gate = operation_gate;
                operation()
            },
        )
        .await
}

fn realtime_stream_resource_vector(
    backend: BackendKind,
    host_bytes: u64,
    tensor_bytes: u64,
) -> Result<ResourceVector> {
    let mut resources = ResourceVector::zero();
    match backend {
        BackendKind::Cpu => {
            resources.host_bytes =
                ResourceAmount::Known(host_bytes.checked_add(tensor_bytes).ok_or_else(|| {
                    Error::Overloaded("realtime ASR resource reservation overflowed".to_string())
                })?);
        }
        BackendKind::Metal => {
            resources.unified_bytes =
                ResourceAmount::Known(host_bytes.checked_add(tensor_bytes).ok_or_else(|| {
                    Error::Overloaded("realtime ASR resource reservation overflowed".to_string())
                })?);
        }
        BackendKind::Cuda => {
            resources.host_bytes = ResourceAmount::Known(host_bytes);
            resources.device_bytes = ResourceAmount::Known(tensor_bytes);
        }
    }
    Ok(resources)
}

fn voxtral_realtime_preparation_cost(
    model: &VoxtralModelLease,
    geometry: VoxtralRealtimePreparationGeometry,
) -> Result<WorkCost> {
    let batch = model.realtime_preparation_batch_geometry(&[geometry])?;
    let seal = model.realtime_preparation_stage_seal()?;
    voxtral_realtime_preparation_cost_from_geometry(geometry, batch, seal)
}

fn voxtral_realtime_preparation_cost_from_geometry(
    geometry: VoxtralRealtimePreparationGeometry,
    batch: VoxtralRealtimePreparationBatchGeometry,
    seal: VoxtralRealtimePreparationStageSeal,
) -> Result<WorkCost> {
    if batch.materialized_tensor_elements_per_row > seal.max_materialized_tensor_elements_per_row
        || batch.workspace_per_row_bytes > seal.max_workspace_bytes
    {
        return Err(Error::InferenceError(
            "Voxtral preparation geometry exceeds its loaded stage seal".into(),
        ));
    }
    Ok(WorkCost::new(
        u64::try_from(geometry.source_samples)
            .map_err(|_| Error::Overloaded("Voxtral preparation work overflowed".into()))?
            .max(1),
        batch.materialized_tensor_elements_per_row,
        batch.workspace_per_row_bytes,
    ))
}

fn voxtral_realtime_committed_observation(
    retained_metadata_bytes: u64,
    usage: VoxtralRealtimePreparedResourceUsage,
) -> Result<JobResourceObservation> {
    Ok(JobResourceObservation::new(
        retained_metadata_bytes
            .checked_mul(2)
            .ok_or_else(|| Error::Overloaded("Voxtral retained metadata overflowed".into()))?
            .checked_add(usage.host_bytes)
            .ok_or_else(|| Error::Overloaded("Voxtral retained host usage overflowed".into()))?,
        usage.tensor_bytes,
    ))
}

fn add_realtime_stream_reservation(
    base: ResourceVector,
    backend: BackendKind,
    reservation: NativeAsrRealtimeResourceReservation,
) -> Result<ResourceVector> {
    base.checked_add(realtime_stream_resource_vector(
        backend,
        reservation.host_bytes(),
        crate::models::architectures::nemotron::asr::NEMOTRON_REALTIME_STAGE_WORKSPACE_BYTES,
    )?)
}

impl RuntimeService {
    async fn start_voxtral_realtime_stream(
        &self,
        variant: ModelVariant,
        language: Option<&str>,
        prompt: Option<&str>,
        context: RuntimeRequestContext,
        metadata_bytes: usize,
        session_lease: Arc<RealtimeAsrSessionLease>,
        absolute_deadline: Instant,
        idle_timeout: Duration,
    ) -> Result<RuntimeAsrRealtimeStream> {
        if prompt.is_some_and(|prompt| !prompt.trim().is_empty()) {
            return Err(Error::InvalidInput(
                "Voxtral realtime ASR does not support an initial text prompt".into(),
            ));
        }
        let request_id = uuid::Uuid::new_v4().to_string();
        let initial_spec = self.coordinator_job_for_input(
            request_id.clone(),
            CoordinatorLane::Realtime,
            context,
            metadata_bytes,
        );
        let initial_job = self
            .coordinator
            .admit_observed(initial_spec, host_input_observation(metadata_bytes)?)
            .await?;
        let (residency_lease, execution_contract, state_binding) = self
            .load_capability_with_state_for_job(
                &initial_job,
                variant,
                CapabilityKind::RealtimeAsr,
                true,
                ExecutionTargetKind::RealtimeRunner,
            )
            .await?;
        let model = self
            .model_registry
            .get_voxtral_lease(variant)
            .await
            .ok_or_else(|| {
                Error::ModelNotFound(format!("Voxtral model {variant} is not loaded"))
            })?;
        let peak = model.realtime_stream_peak_reservation()?;
        let max_output_steps = model.realtime_max_output_steps()?;
        let max_cache_append = state_binding
            .state
            .managed_kv_runtime()
            .map(|runtime| runtime.logical_token_reach())
            .filter(|tokens| *tokens > 0)
            .and_then(|tokens| usize::try_from(tokens).ok())
            .ok_or_else(|| {
                Error::ModelLoadError(
                    "Voxtral realtime state has no finite physical KV-token ceiling".into(),
                )
            })?;
        let stream_resources = realtime_stream_resource_vector(
            self.backend_context().backend_kind,
            peak.max_host_bytes
                .checked_add(u64::try_from(metadata_bytes).map_err(|_| {
                    Error::Overloaded("Voxtral retained metadata usage exceeds u64".into())
                })?)
                .ok_or_else(|| {
                    Error::Overloaded("Voxtral transactional host reservation overflowed".into())
                })?,
            peak.max_tensor_bytes,
        )?;
        let mut execution_spec = initial_job.spec.clone();
        let bridge = self.coordinator.bridge_preparation_admission(initial_job)?;
        execution_spec.resources = execution_spec.resources.checked_add(stream_resources)?;
        let job = match self
            .coordinator
            .admit_observed_from_preparation(
                bridge,
                execution_spec,
                host_input_observation(metadata_bytes)?,
            )
            .await
        {
            Ok(job) => job,
            Err(failure) => {
                let error = failure.error;
                drop(failure.bridge);
                return Err(error);
            }
        };

        let mut request = EngineCoreRequest::asr_bytes(Vec::new()).with_model_variant(variant);
        if let Some(language) = language {
            request = request.with_language(language);
        }
        request.id = request_id;
        request.workload_class = context.workload_class;
        request.priority = context.priority;
        request.admission_ms = context.admission_ms;
        request.deadline = Some(absolute_deadline);
        request.params.max_tokens = max_cache_append;
        request.bind_execution_adapter(state_binding.execution.clone())?;
        request.bind_v2_state_runtime(
            state_binding.state.clone(),
            state_binding.state.state_fingerprint,
            self.backend_context().backend_kind,
        )?;
        let admission = self.core_engine.start_realtime_asr_session(request);
        let handle = tokio::time::timeout_at(absolute_deadline.into(), admission)
            .await
            .map_err(|_| Error::Timeout("Voxtral realtime Engine admission".into()))??;
        let engine_session = RuntimeVoxtralRealtimeSession {
            engine: self.core_engine.clone(),
            handle,
            model: model.clone(),
            retained_metadata_bytes: u64::try_from(metadata_bytes).map_err(|_| {
                Error::Overloaded("Voxtral retained metadata usage exceeds u64".into())
            })?,
            max_output_steps,
            max_cache_append,
        };
        let resources = Arc::new(StdMutex::new(RuntimeAsrRealtimeResources {
            model: None,
            state: None,
            execution_contract: Some(execution_contract),
            residency_lease: Some(residency_lease),
            job: Some(job),
            session_lease: Some(session_lease),
            engine_session: Some(engine_session),
            voxtral_model: Some(model),
            absolute_deadline,
            idle_timeout,
            last_activity: Instant::now(),
            active_operations: 0,
            closed: false,
            timeout_reason: None,
        }));
        let activity = Arc::new(Notify::new());
        spawn_realtime_asr_watchdog(&resources, activity.clone());
        Ok(RuntimeAsrRealtimeStream {
            variant,
            resources,
            activity,
            operation_gate: Arc::new(tokio::sync::Mutex::new(())),
            max_samples: peak.max_source_samples,
            committed_samples: 0,
            engine_sample_rate: None,
            engine_text: String::new(),
            engine_chunk_index: 0,
        })
    }

    pub async fn try_start_asr_realtime_stream(
        &self,
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<Option<RuntimeAsrRealtimeStream>> {
        let Some(variant) = resolve_asr_realtime_stream_variant(model_id) else {
            return Ok(None);
        };

        self.observe_broker_capability_request(CapabilityKind::RealtimeAsr, Some(variant), true)?;
        let started_at = Instant::now();
        let limits = self.asr_realtime_sessions.limits;
        let session_lease = self.asr_realtime_sessions.try_acquire()?;
        let absolute_deadline = started_at.checked_add(limits.max_lifetime).ok_or_else(|| {
            Error::ConfigError("realtime ASR absolute lifetime overflowed".to_string())
        })?;
        let context = RuntimeRequestContext::new(crate::engine::WorkloadClass::Realtime)
            .with_deadline(absolute_deadline);
        let metadata_bytes =
            language.map(str::len).unwrap_or_default() + prompt.map(str::len).unwrap_or_default();
        if variant.family() == ModelFamily::Voxtral {
            return self
                .start_voxtral_realtime_stream(
                    variant,
                    language,
                    prompt,
                    context,
                    metadata_bytes,
                    session_lease,
                    absolute_deadline,
                    limits.idle_timeout,
                )
                .await
                .map(Some);
        }
        let reservation = NativeAsrModel::conservative_realtime_stream_resource_reservation(
            variant, language, prompt, None,
        )?;
        let mut job_spec = self.coordinator_job_for_input(
            uuid::Uuid::new_v4().to_string(),
            CoordinatorLane::Realtime,
            context,
            metadata_bytes,
        );
        job_spec.resources = add_realtime_stream_reservation(
            job_spec.resources,
            self.backend_context().backend_kind,
            reservation,
        )?;
        let job = self
            .coordinator
            .admit_observed(job_spec, host_input_observation(metadata_bytes)?)
            .await?;
        let (lease, execution_contract, state_binding) = self
            .load_capability_with_state_for_job(
                &job,
                variant,
                CapabilityKind::RealtimeAsr,
                true,
                ExecutionTargetKind::RealtimeRunner,
            )
            .await?;
        let physical_runtime = state_binding
            .state
            .retained_tensor_state_runtime()
            .ok_or_else(|| {
                Error::ModelLoadError(
                    "realtime ASR did not bind its retained tensor runtime".into(),
                )
            })?;
        let model =
            self.model_registry.get_asr(variant).await.ok_or_else(|| {
                Error::ModelNotFound(format!("ASR model {variant} is not loaded"))
            })?;
        if !model.supports_realtime_stream_decode() {
            return Ok(None);
        }
        let operation_lease = self
            .model_lifecycle
            .try_acquire_ready_lease(variant)
            .ok_or_else(|| Error::ModelNotFound(format!("ASR model {variant} is not resident")))?;
        let task_model = model.clone();
        let language = language.map(ToOwned::to_owned);
        let prompt = prompt.map(ToOwned::to_owned);
        let retained_metadata_bytes = metadata_bytes
            .checked_add(language.as_ref().map_or(0, String::capacity))
            .and_then(|bytes| bytes.checked_add(prompt.as_ref().map_or(0, String::capacity)))
            .ok_or_else(|| Error::Overloaded("realtime ASR metadata overflowed".to_string()))?;
        job.record_materialized_usage(retained_host_observation(&[
            metadata_bytes,
            language.as_ref().map_or(0, String::capacity),
            prompt.as_ref().map_or(0, String::capacity),
        ])?)?;
        let task_session_lease = session_lease.clone();
        let (state, language, prompt) = self
            .coordinator
            .run_loaded_blocking_stage(
                &job,
                execution_contract.clone(),
                WorkUnit::AtomicJob {
                    kind: "asr.realtime.start".to_string(),
                },
                move || {
                    let _operation_lease = operation_lease;
                    let _session_lease = task_session_lease;
                    let native = task_model.start_realtime_stream_state_with_reservation(
                        language.as_deref(),
                        prompt.as_deref(),
                        None,
                        reservation,
                    )?;
                    let state = RuntimeAsrRealtimeState::new(
                        task_model.as_ref(),
                        native,
                        physical_runtime,
                    )?;
                    Ok((state, language, prompt))
                },
            )
            .await?;
        let steady_usage = realtime_state_observation(&state.native)?;
        job.record_materialized_usage(add_retained_host_bytes(
            steady_usage,
            retained_metadata_bytes,
        )?)?;
        job.prepare_materialized_release(steady_usage)?;
        drop((language, prompt));

        let resources = Arc::new(StdMutex::new(RuntimeAsrRealtimeResources {
            model: Some(model),
            state: Some(Arc::new(StdMutex::new(state))),
            execution_contract: Some(execution_contract),
            residency_lease: Some(lease),
            job: Some(job),
            session_lease: Some(session_lease),
            engine_session: None,
            voxtral_model: None,
            absolute_deadline,
            idle_timeout: limits.idle_timeout,
            last_activity: Instant::now(),
            active_operations: 0,
            closed: false,
            timeout_reason: None,
        }));
        let activity = Arc::new(Notify::new());
        spawn_realtime_asr_watchdog(&resources, activity.clone());

        Ok(Some(RuntimeAsrRealtimeStream {
            variant,
            resources,
            activity,
            operation_gate: Arc::new(tokio::sync::Mutex::new(())),
            max_samples: reservation.max_samples(),
            committed_samples: 0,
            engine_sample_rate: None,
            engine_text: String::new(),
            engine_chunk_index: 0,
        }))
    }

    pub async fn push_asr_realtime_samples(
        &self,
        stream: &mut RuntimeAsrRealtimeStream,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<RuntimeAsrRealtimeEvent>> {
        // Reject an impossible single packet before acquiring residency or
        // allocating the owned copy needed by the blocking worker.
        validate_realtime_input_copy(samples.len(), stream.max_samples)?;
        if samples.is_empty() {
            stream.ensure_open()?;
            return Ok(Vec::new());
        }
        if stream.variant.family() == ModelFamily::Voxtral {
            let total_samples = stream
                .committed_samples
                .checked_add(samples.len())
                .ok_or_else(|| {
                    Error::Overloaded("Voxtral realtime sample clock overflowed".into())
                })?;
            validate_realtime_input_copy(total_samples, stream.max_samples)?;
            let handles = stream.begin_engine_operation(true)?;
            let operation_lease = self
                .model_lifecycle
                .try_acquire_ready_lease(stream.variant)
                .ok_or_else(|| {
                    Error::ModelNotFound(format!("ASR model {} is not resident", stream.variant))
                });
            let operation_lease = match operation_lease {
                Ok(lease) => lease,
                Err(error) => {
                    stream.close();
                    return Err(error);
                }
            };
            let RealtimeAsrEngineOperationHandles {
                session,
                job,
                session_lease,
                _guard: operation_guard,
            } = handles;
            if stream
                .engine_sample_rate
                .is_some_and(|established| established != sample_rate)
            {
                drop((operation_lease, session_lease, operation_guard));
                return Err(Error::InvalidInput(
                    "Voxtral realtime sample rate changed within one stream".into(),
                ));
            }
            let geometry = session
                .model
                .realtime_preparation_geometry_for_source_samples(
                    total_samples,
                    sample_rate,
                    VoxtralRealtimePreparationMode::Push,
                )?;
            let preparation_cost = voxtral_realtime_preparation_cost(&session.model, geometry)?;
            let committed_observation = voxtral_realtime_committed_observation(
                session.retained_metadata_bytes,
                session.model.realtime_prepared_resource_usage(geometry)?,
            )?;
            let ordering = match self
                .coordinator
                .acquire_job_ordering(&job, stream.operation_gate.clone())
                .await
            {
                Ok(ordering) => ordering,
                Err(error) => {
                    drop((operation_lease, session_lease, operation_guard));
                    if matches!(error, Error::Timeout(_)) {
                        stream.close_due_to_timeout();
                    } else {
                        stream.close();
                    }
                    return Err(error);
                }
            };
            let operation = session
                .engine
                .push_realtime_asr_samples_with_outputs_and_cost(
                    &session.handle,
                    samples.to_vec(),
                    sample_rate,
                    session.max_output_steps,
                    session.max_cache_append,
                    preparation_cost,
                );
            let result = match job.spec.deadline {
                Some(deadline) => match tokio::time::timeout_at(deadline.into(), operation).await {
                    Ok(result) => result,
                    Err(_) => Err(Error::Timeout("Voxtral realtime push".into())),
                },
                None => operation.await,
            };
            drop((ordering, operation_lease, session_lease, operation_guard));
            let (ack, outputs) = match result {
                Ok(result) => result,
                Err(error) => {
                    if matches!(error, Error::Timeout(_)) {
                        stream.close_due_to_timeout();
                    } else {
                        stream.close();
                    }
                    return Err(error);
                }
            };
            if ack.accepted_samples() != samples.len() {
                stream.close();
                return Err(Error::InferenceError(
                    "Voxtral Engine acknowledgement reported the wrong sample span".into(),
                ));
            }
            if let Err(error) = job.record_materialized_usage(committed_observation) {
                stream.close();
                return Err(error);
            }
            stream.committed_samples = total_samples;
            stream.engine_sample_rate = Some(sample_rate);
            return Ok(stream.map_engine_outputs(outputs));
        }
        let handles = stream.begin_operation(true)?;
        let operation_lease = self
            .model_lifecycle
            .try_acquire_ready_lease(stream.variant)
            .ok_or_else(|| {
                Error::ModelNotFound(format!("ASR model {} is not resident", stream.variant))
            });
        let operation_lease = match operation_lease {
            Ok(lease) => lease,
            Err(err) => {
                stream.close();
                return Err(err);
            }
        };
        let operation_gate = stream.operation_gate.clone();
        let RealtimeAsrOperationHandles {
            model,
            state,
            execution_contract,
            job,
            session_lease,
            _guard: operation_guard,
        } = handles;
        let observation_job = job.clone();
        let result = run_realtime_blocking_operation(
            &self.coordinator,
            &job,
            execution_contract,
            operation_gate,
            "asr.realtime.push",
            || {
                // The ordering guard is held before this allocation, bounding
                // each stream to one owned packet even after caller timeout.
                let samples = samples.to_vec();
                Ok(move || {
                    let _operation_lease = operation_lease;
                    let _session_lease = session_lease;
                    let _operation_guard = operation_guard;
                    let mut state = state.lock().map_err(|_| {
                        Error::InferenceError("realtime ASR state mutex poisoned".to_string())
                    })?;
                    // Streaming internals replace and release model-owned
                    // tensors while decoding. Restore the full pending claim
                    // before entering that transition so any physical frees
                    // cannot be observed as unclaimed headroom.
                    observation_job
                        .prepare_materialized_release(JobResourceObservation::default())?;
                    let events = state.transact(model.as_ref(), |native| {
                        model.push_realtime_stream_samples(native, &samples, sample_rate)
                    });
                    observation_job.record_materialized_usage(
                        realtime_state_with_input_observation(&state.native, samples.capacity())?,
                    )?;
                    let steady_usage = realtime_state_observation(&state.native)?;
                    observation_job.prepare_materialized_release(steady_usage)?;
                    drop(samples);
                    events
                })
            },
        )
        .await;
        if let Err(err) = &result {
            if matches!(err, Error::Timeout(_)) {
                stream.close_due_to_timeout();
            } else {
                stream.close();
            }
        }
        let events = result?;
        Ok(map_native_realtime_events(events))
    }

    pub async fn finish_asr_realtime_stream(
        &self,
        stream: &mut RuntimeAsrRealtimeStream,
    ) -> Result<Vec<RuntimeAsrRealtimeEvent>> {
        if stream.variant.family() == ModelFamily::Voxtral {
            let handles = stream.begin_engine_operation(false)?;
            let operation_lease = self
                .model_lifecycle
                .try_acquire_ready_lease(stream.variant)
                .ok_or_else(|| {
                    Error::ModelNotFound(format!("ASR model {} is not resident", stream.variant))
                });
            let operation_lease = match operation_lease {
                Ok(lease) => lease,
                Err(error) => {
                    stream.close();
                    return Err(error);
                }
            };
            let RealtimeAsrEngineOperationHandles {
                session,
                job,
                session_lease,
                _guard: operation_guard,
            } = handles;
            let sample_rate = stream.engine_sample_rate.ok_or_else(|| {
                Error::InvalidInput("Voxtral realtime finish requires committed audio".into())
            })?;
            let geometry = session
                .model
                .realtime_preparation_geometry_for_source_samples(
                    stream.committed_samples,
                    sample_rate,
                    VoxtralRealtimePreparationMode::Finish,
                )?;
            let preparation_cost = voxtral_realtime_preparation_cost(&session.model, geometry)?;
            let committed_observation = voxtral_realtime_committed_observation(
                session.retained_metadata_bytes,
                session.model.realtime_prepared_resource_usage(geometry)?,
            )?;
            let ordering = match self
                .coordinator
                .acquire_job_ordering(&job, stream.operation_gate.clone())
                .await
            {
                Ok(ordering) => ordering,
                Err(error) => {
                    drop((operation_lease, session_lease, operation_guard));
                    if matches!(error, Error::Timeout(_)) {
                        stream.close_due_to_timeout();
                    } else {
                        stream.close();
                    }
                    return Err(error);
                }
            };
            let operation = session
                .engine
                .finish_realtime_asr_session_with_outputs_and_cost(
                    &session.handle,
                    session.max_output_steps,
                    session.max_cache_append,
                    preparation_cost,
                );
            let result = match job.spec.deadline {
                Some(deadline) => match tokio::time::timeout_at(deadline.into(), operation).await {
                    Ok(result) => result,
                    Err(_) => Err(Error::Timeout("Voxtral realtime finish".into())),
                },
                None => operation.await,
            };
            drop((ordering, operation_lease, session_lease, operation_guard));
            let (_ack, outputs) = match result {
                Ok(result) => result,
                Err(error) => {
                    if matches!(error, Error::Timeout(_)) {
                        stream.close_due_to_timeout();
                    } else {
                        stream.close();
                    }
                    return Err(error);
                }
            };
            if let Err(error) = job.record_materialized_usage(committed_observation) {
                stream.close();
                return Err(error);
            }
            stream.close();
            return Ok(stream.map_engine_outputs(outputs));
        }
        let handles = stream.begin_operation(false)?;
        let operation_lease = self
            .model_lifecycle
            .try_acquire_ready_lease(stream.variant)
            .ok_or_else(|| {
                Error::ModelNotFound(format!("ASR model {} is not resident", stream.variant))
            });
        let operation_lease = match operation_lease {
            Ok(lease) => lease,
            Err(err) => {
                stream.close();
                return Err(err);
            }
        };
        let operation_gate = stream.operation_gate.clone();
        let RealtimeAsrOperationHandles {
            model,
            state,
            execution_contract,
            job,
            session_lease,
            _guard: operation_guard,
        } = handles;
        let observation_job = job.clone();
        let result = run_realtime_blocking_operation(
            &self.coordinator,
            &job,
            execution_contract,
            operation_gate,
            "asr.realtime.finish",
            || {
                Ok(move || {
                    let _operation_lease = operation_lease;
                    let _session_lease = session_lease;
                    let _operation_guard = operation_guard;
                    let mut state = state.lock().map_err(|_| {
                        Error::InferenceError("realtime ASR state mutex poisoned".to_string())
                    })?;
                    observation_job
                        .prepare_materialized_release(JobResourceObservation::default())?;
                    state.transact(model.as_ref(), |native| {
                        model.finish_realtime_stream(native)
                    })
                })
            },
        )
        .await;
        if matches!(&result, Err(Error::Timeout(_))) {
            stream.close_due_to_timeout();
        } else {
            stream.close();
        }
        let events = result?;
        Ok(map_native_realtime_events(events))
    }

    pub fn asr_realtime_stream_variant(&self, stream: &RuntimeAsrRealtimeStream) -> ModelVariant {
        stream.variant
    }

    async fn asr_transcribe_audio_chat<F>(
        &self,
        variant: ModelVariant,
        audio_input: AsrAudioInput<'_>,
        max_tokens: Option<usize>,
        streaming_required: bool,
        on_delta: F,
        request_id: String,
        runtime_context: RuntimeRequestContext,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let input_bytes = audio_input.input_bytes();
        if input_bytes == 0 {
            return Err(Error::InvalidInput(
                "ASR request missing audio input".to_string(),
            ));
        }
        audio_input.validate_retained_size()?;
        let spec = self.coordinator_job_for_audio_input(
            request_id,
            CoordinatorLane::Atomic,
            runtime_context,
            input_bytes,
        )?;
        let job = self
            .coordinator
            .admit_observed(spec, host_input_observation(input_bytes)?)
            .await?;

        // Establish an owned packet only after admission. The caller-owned
        // allocation remains live while this future runs, so both allocations
        // are reconciled until the blocking decoder consumes the owned copy.
        let audio_input = audio_input.into_owned_for_job(&job).await?;
        job.record_materialized_usage(retained_host_observation(&[
            input_bytes,
            audio_input.retained_bytes(),
        ])?)?;

        let (residency_lease, execution_contract, state_binding) = self
            .load_capability_with_state_for_job(
                &job,
                variant,
                CapabilityKind::Asr,
                streaming_required,
                ExecutionTargetKind::DirectModel,
            )
            .await?;
        let model = self
            .model_registry
            .get_audio_chat(variant)
            .await
            .ok_or_else(|| {
                Error::ModelNotFound(format!("Audio-chat model {variant} is not loaded"))
            })?;
        let observation_job = job.clone();
        self.coordinator
            .run_loaded_blocking_stage_with_invocation_workspace(
                &job,
                execution_contract,
                state_binding,
                WorkUnit::AtomicJob {
                    kind: "asr.audio_chat".to_string(),
                },
                move |leases| {
                    let _residency_lease = residency_lease;
                    let retained_audio_bytes = audio_input.retained_bytes();
                    let (samples, sample_rate) = audio_input.decode()?;
                    let steady_usage = decoded_audio_observation(input_bytes, samples.capacity())?;
                    observation_job.record_materialized_usage(add_retained_host_bytes(
                        steady_usage,
                        retained_audio_bytes,
                    )?)?;
                    observation_job.prepare_materialized_release(steady_usage)?;
                    drop(audio_input);
                    let duration_secs = if sample_rate > 0 {
                        samples.len() as f32 / sample_rate as f32
                    } else {
                        0.0
                    };
                    let mut on_delta = on_delta;
                    let mut delta_sink = |delta: &str| {
                        if !delta.is_empty() {
                            on_delta(delta.to_string());
                        }
                    };
                    let output = model
                        .transcribe_with_callback_and_max_tokens_from_invocation_workspace(
                            &samples,
                            sample_rate,
                            max_tokens,
                            leases,
                            &mut delta_sink,
                        )?;

                    Ok(AsrTranscription {
                        text: output.text,
                        language: output.language,
                        duration_secs,
                        asr_diagnostics: output.diagnostics,
                    })
                },
            )
            .await
    }

    async fn build_asr_request(
        &self,
        variant: ModelVariant,
        audio_input: AsrAudioInput<'_>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        streaming: bool,
    ) -> Result<AdmittedEngineRequest> {
        let audio_bytes = match audio_input {
            AsrAudioInput::Base64("") => {
                return Err(Error::InvalidInput(
                    "ASR request missing audio input".to_string(),
                ));
            }
            AsrAudioInput::Bytes([]) => {
                return Err(Error::InvalidInput(
                    "ASR request missing audio bytes".to_string(),
                ));
            }
            AsrAudioInput::Base64(audio) => audio.len(),
            AsrAudioInput::Bytes(audio) => audio.len(),
        };
        audio_input.validate_retained_size()?;
        let input_bytes = audio_bytes
            .checked_add(language.map(str::len).unwrap_or_default())
            .and_then(|bytes| bytes.checked_add(prompt.map(str::len).unwrap_or_default()))
            .and_then(|bytes| bytes.checked_add(correlation_id.map(str::len).unwrap_or_default()))
            .ok_or_else(|| Error::Overloaded("ASR preparation input overflowed".to_string()))?;
        self.prepare_engine_request_blocking_with_input(
            variant,
            TaskType::ASR,
            streaming,
            runtime_context,
            input_bytes,
            ResourceVector::zero(),
            move |job| async move {
                let audio_input = audio_input.into_owned_for_job(&job).await?;
                let language =
                    copy_optional_preparation_string(&job, language, "ASR language metadata")
                        .await?;
                let prompt =
                    copy_optional_preparation_string(&job, prompt, "ASR prompt metadata").await?;
                let correlation_id = copy_optional_preparation_string(
                    &job,
                    correlation_id,
                    "ASR correlation metadata",
                )
                .await?;
                let retained_metadata_bytes = language
                    .as_ref()
                    .map_or(0, String::capacity)
                    .checked_add(prompt.as_ref().map_or(0, String::capacity))
                    .and_then(|bytes| {
                        bytes.checked_add(correlation_id.as_ref().map_or(0, String::capacity))
                    })
                    .ok_or_else(|| {
                        Error::Overloaded("ASR retained metadata overflowed".to_string())
                    })?;
                job.record_materialized_usage(retained_host_observation(&[
                    input_bytes,
                    audio_input.retained_bytes(),
                    retained_metadata_bytes,
                ])?)?;
                Ok((audio_input, language, prompt, correlation_id))
            },
            move |_registry, (audio_input, language, prompt, correlation_id)| {
                audio_input.validate_source_size()?;
                let runtime_request = match audio_input {
                    OwnedAsrAudioInput::Base64(audio_base64) => AsrRuntimeRequest::from_base64(
                        audio_base64,
                        variant,
                        language,
                        correlation_id,
                        runtime_context,
                    )?,
                    OwnedAsrAudioInput::Bytes(audio_bytes) => AsrRuntimeRequest::from_bytes(
                        audio_bytes,
                        variant,
                        language,
                        correlation_id,
                        runtime_context,
                    )?,
                }
                .with_prompt(prompt);
                let mut request = runtime_request.into_engine_request();
                if let Some(max_tokens) = max_tokens {
                    request.params.max_tokens = max_tokens;
                } else if let Some(auto_max_tokens) = granite_auto_asr_token_ceiling(variant) {
                    request.params.max_tokens = auto_max_tokens;
                    request.asr_auto_max_tokens = true;
                }
                Ok(request)
            },
        )
        .await
    }

    pub(crate) async fn asr_transcribe_with_variant(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_with_variant_and_prompt(
            variant,
            audio_base64,
            language,
            None,
            correlation_id,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_with_variant_and_prompt(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_with_variant_and_prompt_options(
            variant,
            audio_base64,
            language,
            prompt,
            None,
            correlation_id,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_with_variant_and_prompt_options(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        if variant.is_audio_chat() && variant.family() != ModelFamily::Lfm25Audio {
            self.observe_broker_capability_request(CapabilityKind::Asr, Some(variant), false)?;
            let context = RuntimeRequestContext::default();
            return self
                .asr_transcribe_audio_chat(
                    variant,
                    AsrAudioInput::Base64(audio_base64),
                    max_tokens,
                    false,
                    |_delta| {},
                    correlation_id
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    context,
                )
                .await;
        }

        let admitted = self
            .build_asr_request(
                variant,
                AsrAudioInput::Base64(audio_base64),
                language,
                prompt,
                max_tokens,
                correlation_id,
                RuntimeRequestContext::default(),
                false,
            )
            .await?;
        let output = self.run_admitted_request(admitted).await?;
        let text = output.text.unwrap_or_default();

        Ok(AsrTranscription {
            text,
            language: language.map(|s| s.to_string()),
            duration_secs: output.audio.duration_secs,
            asr_diagnostics: output.asr_diagnostics,
        })
    }

    pub(crate) async fn asr_transcribe_with_variant_streaming<F>(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_with_variant_streaming_and_prompt(
            variant,
            audio_base64,
            language,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_with_variant_streaming_and_prompt<F>(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_with_variant_streaming_and_prompt_options(
            variant,
            audio_base64,
            language,
            prompt,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_with_variant_streaming_and_prompt_options<F>(
        &self,
        variant: ModelVariant,
        audio_base64: &str,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        mut on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        if variant.is_audio_chat() && variant.family() != ModelFamily::Lfm25Audio {
            self.observe_broker_capability_request(CapabilityKind::Asr, Some(variant), true)?;
            let context = RuntimeRequestContext::new(crate::engine::WorkloadClass::Streaming);
            return self
                .asr_transcribe_audio_chat(
                    variant,
                    AsrAudioInput::Base64(audio_base64),
                    max_tokens,
                    true,
                    on_delta,
                    correlation_id
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    context,
                )
                .await;
        }

        let admitted = self
            .build_asr_request(
                variant,
                AsrAudioInput::Base64(audio_base64),
                language,
                prompt,
                max_tokens,
                correlation_id,
                RuntimeRequestContext::default(),
                true,
            )
            .await?;
        let mut streamed_text = String::new();
        let output = self
            .run_admitted_streaming_request(admitted, |chunk| {
                if let Some(delta) = chunk.text {
                    if !delta.is_empty() {
                        streamed_text.push_str(&delta);
                        on_delta(delta);
                    }
                }
                std::future::ready(Ok(()))
            })
            .await?;
        let text = output.text.unwrap_or(streamed_text);

        Ok(AsrTranscription {
            text,
            language: language.map(|s| s.to_string()),
            duration_secs: output.audio.duration_secs,
            asr_diagnostics: output.asr_diagnostics,
        })
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_bytes_with_variant_and_prompt(
            variant,
            audio_bytes,
            language,
            None,
            correlation_id,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant_and_prompt(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_bytes_with_variant_and_prompt_options(
            variant,
            audio_bytes,
            language,
            prompt,
            None,
            correlation_id,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant_and_prompt_options(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_bytes_with_variant_and_prompt_options_and_runtime_context(
            variant,
            audio_bytes,
            language,
            prompt,
            max_tokens,
            correlation_id,
            RuntimeRequestContext::default(),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn asr_transcribe_bytes_with_variant_and_prompt_options_and_runtime_context(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
    ) -> Result<AsrTranscription> {
        if variant.is_audio_chat() && variant.family() != ModelFamily::Lfm25Audio {
            self.observe_broker_capability_request(CapabilityKind::Asr, Some(variant), false)?;
            return self
                .asr_transcribe_audio_chat(
                    variant,
                    AsrAudioInput::Bytes(audio_bytes),
                    max_tokens,
                    false,
                    |_delta| {},
                    correlation_id
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    runtime_context,
                )
                .await;
        }

        let admitted = self
            .build_asr_request(
                variant,
                AsrAudioInput::Bytes(audio_bytes),
                language,
                prompt,
                max_tokens,
                correlation_id,
                runtime_context,
                false,
            )
            .await?;
        let output = self.run_admitted_request(admitted).await?;
        let text = output.text.unwrap_or_default();

        Ok(AsrTranscription {
            text,
            language: language.map(|s| s.to_string()),
            duration_secs: output.audio.duration_secs,
            asr_diagnostics: output.asr_diagnostics,
        })
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant_streaming<F>(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt(
            variant,
            audio_bytes,
            language,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant_streaming_and_prompt<F>(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt_options(
            variant,
            audio_bytes,
            language,
            prompt,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub(crate) async fn asr_transcribe_bytes_with_variant_streaming_and_prompt_options<F>(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            prompt,
            max_tokens,
            correlation_id,
            on_delta,
            |_progress| {},
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn asr_transcribe_bytes_with_variant_streaming_and_prompt_options_with_progress<
        F,
        P,
    >(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        on_delta: F,
        on_progress: P,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        self.asr_transcribe_bytes_with_variant_callback_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            prompt,
            max_tokens,
            correlation_id,
            on_delta,
            on_progress,
            true,
            RuntimeRequestContext::default(),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn asr_transcribe_bytes_with_variant_callback_and_prompt_options_with_progress<F, P>(
        &self,
        variant: ModelVariant,
        audio_bytes: &[u8],
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        mut on_delta: F,
        mut on_progress: P,
        broker_streaming_required: bool,
        runtime_context: RuntimeRequestContext,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        if variant.is_audio_chat() && variant.family() != ModelFamily::Lfm25Audio {
            self.observe_broker_capability_request(
                CapabilityKind::Asr,
                Some(variant),
                broker_streaming_required,
            )?;
            return self
                .asr_transcribe_audio_chat(
                    variant,
                    AsrAudioInput::Bytes(audio_bytes),
                    max_tokens,
                    broker_streaming_required,
                    on_delta,
                    correlation_id
                        .map(ToOwned::to_owned)
                        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    runtime_context,
                )
                .await;
        }

        let admitted = self
            .build_asr_request(
                variant,
                AsrAudioInput::Bytes(audio_bytes),
                language,
                prompt,
                max_tokens,
                correlation_id,
                runtime_context,
                true,
            )
            .await?;
        let mut streamed_text = String::new();
        let handle_chunk = |chunk: crate::engine::StreamingOutput| {
            if let Some(progress) = chunk.asr_progress {
                on_progress(progress);
            }
            if let Some(delta) = chunk.text {
                if !delta.is_empty() {
                    streamed_text.push_str(&delta);
                    on_delta(delta);
                }
            }
            std::future::ready(Ok(()))
        };
        let output = if broker_streaming_required {
            self.run_admitted_streaming_request(admitted, handle_chunk)
                .await?
        } else {
            self.run_admitted_transport_streaming_request(admitted, handle_chunk)
                .await?
        };
        let text = output.text.unwrap_or(streamed_text);

        Ok(AsrTranscription {
            text,
            language: language.map(|s| s.to_string()),
            duration_secs: output.audio.duration_secs,
            asr_diagnostics: output.asr_diagnostics,
        })
    }

    /// Transcribe audio with Voxtral through the offline transcription path.
    pub async fn voxtral_transcribe(
        &self,
        audio_base64: &str,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_with_variant(
            ModelVariant::VoxtralMini4BRealtime2602,
            audio_base64,
            language,
            None,
        )
        .await
    }

    /// Transcribe audio with Voxtral and emit incremental deltas from offline decode.
    pub async fn voxtral_transcribe_streaming<F>(
        &self,
        audio_base64: &str,
        language: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_with_variant_streaming(
            ModelVariant::VoxtralMini4BRealtime2602,
            audio_base64,
            language,
            None,
            on_delta,
        )
        .await
    }

    /// Transcribe audio with native ASR models.
    pub async fn asr_transcribe(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_with_correlation(audio_base64, model_id, language, None)
            .await
    }

    /// Transcribe audio with request correlation metadata.
    pub async fn asr_transcribe_with_correlation(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_with_prompt_and_correlation(
            audio_base64,
            model_id,
            language,
            None,
            correlation_id,
        )
        .await
    }

    /// Transcribe audio with optional ASR initial prompt/context metadata.
    pub async fn asr_transcribe_with_prompt_and_correlation(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_with_variant_and_prompt(
            variant,
            audio_base64,
            language,
            prompt,
            correlation_id,
        )
        .await
    }

    /// Transcribe audio with optional ASR prompt and max-token decode budget.
    pub async fn asr_transcribe_with_prompt_max_tokens_and_correlation(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_with_variant_and_prompt_options(
            variant,
            audio_base64,
            language,
            prompt,
            max_tokens,
            correlation_id,
        )
        .await
    }

    pub async fn asr_transcribe_bytes(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
    ) -> Result<AsrTranscription> {
        self.asr_transcribe_bytes_with_prompt(audio_bytes, model_id, language, None)
            .await
    }

    pub async fn asr_transcribe_bytes_with_prompt(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_and_prompt(
            variant,
            audio_bytes,
            language,
            prompt,
            None,
        )
        .await
    }

    pub async fn asr_transcribe_bytes_with_prompt_max_tokens_and_correlation(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_and_prompt_options(
            variant,
            audio_bytes,
            language,
            prompt,
            max_tokens,
            correlation_id,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_bytes_with_runtime_context(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
    ) -> Result<AsrTranscription> {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_and_prompt_options_and_runtime_context(
            variant,
            audio_bytes,
            language,
            prompt,
            max_tokens,
            correlation_id,
            runtime_context,
        )
        .await
    }

    /// Transcribe audio and emit deltas.
    pub async fn asr_transcribe_streaming<F>(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_streaming_with_correlation(
            audio_base64,
            model_id,
            language,
            None,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_bytes<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_streaming_bytes_with_correlation(
            audio_bytes,
            model_id,
            language,
            None,
            on_delta,
        )
        .await
    }

    /// Transcribe audio and emit deltas with request correlation metadata.
    pub async fn asr_transcribe_streaming_with_correlation<F>(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_streaming_with_prompt_and_correlation(
            audio_base64,
            model_id,
            language,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    /// Transcribe audio with optional ASR initial prompt/context metadata and deltas.
    pub async fn asr_transcribe_streaming_with_prompt_and_correlation<F>(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_with_variant_streaming_and_prompt(
            variant,
            audio_base64,
            language,
            prompt,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_with_prompt_max_tokens_and_correlation<F>(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_with_variant_streaming_and_prompt_options(
            variant,
            audio_base64,
            language,
            prompt,
            max_tokens,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_bytes_with_correlation<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        self.asr_transcribe_streaming_bytes_with_prompt_and_correlation(
            audio_bytes,
            model_id,
            language,
            None,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_bytes_with_prompt_and_correlation<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt(
            variant,
            audio_bytes,
            language,
            prompt,
            correlation_id,
            on_delta,
        )
        .await
    }

    pub async fn asr_transcribe_streaming_bytes_with_prompt_max_tokens_and_correlation<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt_options(
            variant,
            audio_bytes,
            language,
            prompt,
            max_tokens,
            correlation_id,
            on_delta,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_streaming_bytes_with_runtime_context<F>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        prompt: Option<&str>,
        max_tokens: Option<usize>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        on_delta: F,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_callback_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            prompt,
            max_tokens,
            correlation_id,
            on_delta,
            |_progress| {},
            true,
            runtime_context,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_streaming_bytes_with_progress_and_correlation<F, P>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
        on_progress: P,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_streaming_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            None,
            None,
            correlation_id,
            on_delta,
            on_progress,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_streaming_bytes_with_progress_and_runtime_context<F, P>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        on_delta: F,
        on_progress: P,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_callback_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            None,
            None,
            correlation_id,
            on_delta,
            on_progress,
            true,
            runtime_context,
        )
        .await
    }

    pub async fn asr_transcribe_bytes_with_progress_and_correlation<F, P>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        on_delta: F,
        on_progress: P,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_callback_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            None,
            None,
            correlation_id,
            on_delta,
            on_progress,
            false,
            RuntimeRequestContext::default(),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn asr_transcribe_bytes_with_progress_and_runtime_context<F, P>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        correlation_id: Option<&str>,
        runtime_context: RuntimeRequestContext,
        on_delta: F,
        on_progress: P,
    ) -> Result<AsrTranscription>
    where
        F: FnMut(String) + Send + 'static,
        P: FnMut(AsrProgress) + Send + 'static,
    {
        let variant = resolve_asr_model_variant(model_id);
        self.asr_transcribe_bytes_with_variant_callback_and_prompt_options_with_progress(
            variant,
            audio_bytes,
            language,
            None,
            None,
            correlation_id,
            on_delta,
            on_progress,
            false,
            runtime_context,
        )
        .await
    }

    pub async fn speaker_attributed_asr(
        &self,
        audio_base64: &str,
        model_id: Option<&str>,
        language: Option<&str>,
        min_speakers: Option<usize>,
        max_speakers: Option<usize>,
    ) -> Result<SpeakerAttributedAsrResult> {
        self.speaker_attributed_asr_input_with_progress(
            AsrAudioInput::Base64(audio_base64),
            model_id,
            language,
            min_speakers,
            max_speakers,
            |_| {},
        )
        .await
    }

    pub async fn speaker_attributed_asr_bytes(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        min_speakers: Option<usize>,
        max_speakers: Option<usize>,
    ) -> Result<SpeakerAttributedAsrResult> {
        self.speaker_attributed_asr_bytes_with_progress(
            audio_bytes,
            model_id,
            language,
            min_speakers,
            max_speakers,
            |_| {},
        )
        .await
    }

    pub async fn speaker_attributed_asr_bytes_with_progress<P>(
        &self,
        audio_bytes: &[u8],
        model_id: Option<&str>,
        language: Option<&str>,
        min_speakers: Option<usize>,
        max_speakers: Option<usize>,
        on_progress: P,
    ) -> Result<SpeakerAttributedAsrResult>
    where
        P: FnMut(AsrProgress) + Send + 'static,
    {
        self.speaker_attributed_asr_input_with_progress(
            AsrAudioInput::Bytes(audio_bytes),
            model_id,
            language,
            min_speakers,
            max_speakers,
            on_progress,
        )
        .await
    }

    async fn speaker_attributed_asr_input_with_progress<P>(
        &self,
        audio_input: AsrAudioInput<'_>,
        model_id: Option<&str>,
        language: Option<&str>,
        min_speakers: Option<usize>,
        max_speakers: Option<usize>,
        mut on_progress: P,
    ) -> Result<SpeakerAttributedAsrResult>
    where
        P: FnMut(AsrProgress) + Send + 'static,
    {
        let variant = resolve_speaker_attributed_asr_variant(model_id)?;
        let audio_bytes = audio_input.input_bytes();
        if audio_bytes == 0 {
            return Err(Error::InvalidInput(
                "ASR request missing audio input".to_string(),
            ));
        }
        audio_input.validate_retained_size()?;
        let input_bytes = audio_bytes
            .checked_add(language.map(str::len).unwrap_or_default())
            .ok_or_else(|| {
                Error::Overloaded("speaker-attributed ASR input overflowed".to_string())
            })?;
        self.observe_broker_capability_request(
            CapabilityKind::SpeakerAttributedAsr,
            Some(variant),
            false,
        )?;
        let context = RuntimeRequestContext::new(crate::engine::WorkloadClass::Background);
        let spec = self.coordinator_job_for_audio_input(
            uuid::Uuid::new_v4().to_string(),
            CoordinatorLane::Pipeline,
            context,
            input_bytes,
        )?;
        let job = self
            .coordinator
            .admit_observed(spec, host_input_observation(input_bytes)?)
            .await?;
        let owned_audio = audio_input.into_owned_for_job(&job).await?;
        let language_owned =
            copy_optional_preparation_string(&job, language, "speaker-attributed ASR language")
                .await?;
        let language_bytes = language_owned.as_ref().map_or(0, String::capacity);
        job.record_materialized_usage(retained_host_observation(&[
            input_bytes,
            owned_audio.retained_bytes(),
            language_bytes,
        ])?)?;
        let (residency_lease, execution_contract, state_binding) = self
            .load_capability_with_state_for_job(
                &job,
                variant,
                CapabilityKind::SpeakerAttributedAsr,
                false,
                ExecutionTargetKind::PipelineRunner,
            )
            .await?;
        let model = self
            .model_registry
            .get_asr(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;

        let observation_job = job.clone();
        let (residency_lease, samples, sample_rate) = self
            .coordinator
            .run_host_blocking_stage(&job, move || {
                let retained_audio_bytes = owned_audio.retained_bytes();
                let (samples, sample_rate) = owned_audio.decode()?;
                let samples = Arc::<[f32]>::from(samples);
                let steady_usage = decoded_audio_observation(
                    input_bytes.checked_add(language_bytes).ok_or_else(|| {
                        Error::Overloaded(
                            "speaker-attributed ASR retained input overflowed".to_string(),
                        )
                    })?,
                    samples.len(),
                )?;
                observation_job.record_materialized_usage(add_retained_host_bytes(
                    steady_usage,
                    retained_audio_bytes,
                )?)?;
                observation_job.prepare_materialized_release(steady_usage)?;
                drop(owned_audio);
                Ok((residency_lease, samples, sample_rate))
            })
            .await?;
        let duration_secs = if sample_rate > 0 {
            samples.len() as f32 / sample_rate as f32
        } else {
            0.0
        };
        let retained_input_bytes = input_bytes.checked_add(language_bytes).ok_or_else(|| {
            Error::Overloaded("speaker-attributed ASR retained input overflowed".to_string())
        })?;
        let model_limit_secs = model.max_audio_seconds_hint();
        if granite_saa_should_use_single_pass(duration_secs, model_limit_secs) {
            let task_language = language_owned.clone();
            let task_model = model.clone();
            let max_new_tokens = granite_saa_max_new_tokens(&samples, sample_rate);
            let observation_job = job.clone();
            let transcription =
                self.coordinator
                    .run_loaded_blocking_stage_with_invocation_paged(
                        &job,
                        execution_contract,
                        state_binding,
                        WorkUnit::PipelineStage {
                            name: "asr.speaker_attributed.decode".to_string(),
                            ordinal: 0,
                        },
                        move |leases| {
                            let _residency_lease = residency_lease;
                            observation_job.record_materialized_usage(
                                decoded_audio_observation(retained_input_bytes, samples.len())?,
                            )?;
                            granite_saa_transcribe_chunk(
                                &task_model,
                                &samples,
                                sample_rate,
                                task_language.as_deref(),
                                None,
                                max_new_tokens,
                                granite_saa_single_invocation_cache(leases)?,
                            )
                        },
                    )
                    .await?;

            return Ok(speaker_attributed_asr_result_from_text_with_warnings(
                transcription.text.as_str(),
                transcription.language.or(language_owned),
                duration_secs,
                min_speakers,
                max_speakers,
                Vec::new(),
            ));
        }

        let long_form = granite_saa_long_form_transcribe(
            &self.coordinator,
            &job,
            model,
            execution_contract,
            state_binding,
            samples,
            sample_rate,
            language_owned.as_deref(),
            model_limit_secs,
            residency_lease,
            retained_input_bytes,
            &mut on_progress,
        )
        .await?;

        Ok(speaker_attributed_asr_result_from_text_with_warnings(
            long_form.text.as_str(),
            long_form.language.or(language_owned),
            duration_secs,
            min_speakers,
            max_speakers,
            long_form.warnings,
        ))
    }

    /// Force alignment remains a specialized operation not expressed by the
    /// generic engine output type.
    pub async fn force_align(
        &self,
        audio_base64: &str,
        reference_text: &str,
    ) -> Result<Vec<(String, u32, u32)>> {
        self.force_align_with_model_and_language(audio_base64, reference_text, None, None)
            .await
    }

    pub async fn force_align_with_model(
        &self,
        audio_base64: &str,
        reference_text: &str,
        model_id: Option<&str>,
    ) -> Result<Vec<(String, u32, u32)>> {
        self.force_align_with_model_and_language(audio_base64, reference_text, None, model_id)
            .await
    }

    pub async fn force_align_with_model_and_language(
        &self,
        audio_base64: &str,
        reference_text: &str,
        language: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<Vec<(String, u32, u32)>> {
        self.force_align_input(
            AsrAudioInput::Base64(audio_base64),
            reference_text,
            language,
            model_id,
        )
        .await
    }

    pub async fn force_align_bytes_with_model_and_language(
        &self,
        audio_bytes: &[u8],
        reference_text: &str,
        language: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<Vec<(String, u32, u32)>> {
        self.force_align_input(
            AsrAudioInput::Bytes(audio_bytes),
            reference_text,
            language,
            model_id,
        )
        .await
    }

    async fn force_align_input(
        &self,
        audio_input: AsrAudioInput<'_>,
        reference_text: &str,
        language: Option<&str>,
        model_id: Option<&str>,
    ) -> Result<Vec<(String, u32, u32)>> {
        let variant = resolve_forced_aligner_variant(model_id)?;
        if audio_input.input_bytes() == 0 {
            return Err(Error::InvalidInput(
                "forced alignment request missing audio input".to_string(),
            ));
        }
        audio_input.validate_retained_size()?;
        if reference_text.trim().is_empty() {
            return Err(Error::InvalidInput(
                "Forced alignment request missing transcript".to_string(),
            ));
        }
        self.observe_broker_capability_request(
            CapabilityKind::ForcedAlignment,
            Some(variant),
            false,
        )?;
        let input_bytes = audio_input
            .input_bytes()
            .checked_add(reference_text.len())
            .and_then(|bytes| bytes.checked_add(language.map(str::len).unwrap_or_default()))
            .ok_or_else(|| Error::Overloaded("forced alignment input overflowed".to_string()))?;
        let context = RuntimeRequestContext::default();
        let spec = self.coordinator_job_for_audio_input(
            uuid::Uuid::new_v4().to_string(),
            CoordinatorLane::Atomic,
            context,
            input_bytes,
        )?;
        let job = self
            .coordinator
            .admit_observed(spec, host_input_observation(input_bytes)?)
            .await?;
        let audio_input = audio_input.into_owned_for_job(&job).await?;
        let reference_text =
            copy_preparation_string(&job, reference_text, "forced alignment transcript").await?;
        let language =
            copy_optional_preparation_string(&job, language, "forced alignment language").await?;
        let owned_metadata_bytes = reference_text
            .capacity()
            .checked_add(language.as_ref().map(String::capacity).unwrap_or_default())
            .ok_or_else(|| {
                Error::Overloaded("forced alignment metadata storage overflowed".to_string())
            })?;
        job.record_materialized_usage(retained_host_observation(&[
            input_bytes,
            audio_input.retained_bytes(),
            owned_metadata_bytes,
        ])?)?;
        let (residency_lease, execution_contract) = self
            .load_capability_for_job(
                &job,
                variant,
                CapabilityKind::ForcedAlignment,
                false,
                ExecutionTargetKind::BatchRunner,
            )
            .await?;
        let model = self
            .model_registry
            .get_asr(variant)
            .await
            .ok_or_else(|| Error::ModelNotFound(variant.to_string()))?;
        let observation_job = job.clone();
        self.coordinator
            .run_loaded_blocking_stage(
                &job,
                execution_contract,
                WorkUnit::AtomicJob {
                    kind: "asr.forced_alignment".to_string(),
                },
                move || {
                    let _residency_lease = residency_lease;
                    let retained_audio_bytes = audio_input.retained_bytes();
                    let (samples, sample_rate) = audio_input.decode()?;
                    let retained_bytes =
                        input_bytes
                            .checked_add(owned_metadata_bytes)
                            .ok_or_else(|| {
                                Error::Overloaded(
                                    "forced alignment retained storage overflowed".to_string(),
                                )
                            })?;
                    let steady_usage =
                        decoded_audio_observation(retained_bytes, samples.capacity())?;
                    observation_job.record_materialized_usage(add_retained_host_bytes(
                        steady_usage,
                        retained_audio_bytes,
                    )?)?;
                    observation_job.prepare_materialized_release(steady_usage)?;
                    drop(audio_input);
                    model.force_align(&samples, sample_rate, &reference_text, language.as_deref())
                },
            )
            .await
    }
}

fn resolve_speaker_attributed_asr_variant(model_id: Option<&str>) -> Result<ModelVariant> {
    let variant = match model_id {
        Some(raw) => {
            parse_model_variant(raw).map_err(|err| Error::InvalidInput(err.to_string()))?
        }
        None => ModelVariant::GraniteSpeech412BPlus,
    };

    if variant != ModelVariant::GraniteSpeech412BPlus {
        return Err(Error::InvalidInput(format!(
            "Speaker attributed ASR currently requires Granite-Speech-4.1-2B-Plus, got {variant}"
        )));
    }

    Ok(variant)
}

#[derive(Debug, Clone)]
struct GraniteSaaLongFormOutput {
    text: String,
    language: Option<String>,
    warnings: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct GraniteSaaChunkDecodeDiagnostics {
    generated_tokens: Option<usize>,
    max_new_tokens: Option<usize>,
    stop_reason: Option<String>,
    prompt_tokens: Option<usize>,
    prompt_prefix_tokens: Option<usize>,
}

#[derive(Debug, Default)]
struct GraniteSaaTranscriptAssembler {
    turns: Vec<SpeakerAttributedAsrTurn>,
}

impl GraniteSaaTranscriptAssembler {
    fn push_chunk_text(&mut self, chunk_text: &str, chunk_index: usize) -> Vec<String> {
        let mut warnings = Vec::new();
        let incoming = speaker_attributed_asr_turns_from_text(chunk_text);
        if incoming.is_empty() {
            if !chunk_text.trim().is_empty() {
                warnings.push(format!(
                    "Granite SAA chunk {} returned text without speaker labels.",
                    chunk_index + 1
                ));
            }
            return warnings;
        }

        let mut speaker_aliases = HashMap::<String, String>::new();
        if let (Some(last), Some(first)) = (self.turns.last(), incoming.first()) {
            let overlap = overlap_prefix_word_count(
                last.text.as_str(),
                first.text.as_str(),
                GRANITE_SAA_MIN_OVERLAP_WORDS,
                GRANITE_SAA_MAX_OVERLAP_WORDS,
            );
            if overlap > 0
                && first.speaker != UNKNOWN_SAA_SPEAKER
                && last.speaker != UNKNOWN_SAA_SPEAKER
                && first.speaker != last.speaker
            {
                speaker_aliases.insert(first.speaker.clone(), last.speaker.clone());
                warnings.push(format!(
                    "Granite SAA chunk {} reused speaker label {} across an overlap; mapped it to {}.",
                    chunk_index + 1,
                    first.speaker,
                    last.speaker
                ));
            }
        }
        if !speaker_aliases.is_empty() {
            let mut known_global_speakers = Vec::<String>::new();
            for turn in &self.turns {
                if turn.speaker != UNKNOWN_SAA_SPEAKER
                    && !known_global_speakers.contains(&turn.speaker)
                {
                    known_global_speakers.push(turn.speaker.clone());
                }
            }

            let mut used_global_speakers = speaker_aliases
                .values()
                .cloned()
                .collect::<HashSet<String>>();
            let mut local_speakers = Vec::<String>::new();
            for turn in &incoming {
                if turn.speaker != UNKNOWN_SAA_SPEAKER && !local_speakers.contains(&turn.speaker) {
                    local_speakers.push(turn.speaker.clone());
                }
            }

            for local_speaker in local_speakers {
                if speaker_aliases.contains_key(local_speaker.as_str())
                    || !used_global_speakers.contains(local_speaker.as_str())
                {
                    continue;
                }
                if let Some(global_speaker) = known_global_speakers
                    .iter()
                    .find(|speaker| !used_global_speakers.contains(speaker.as_str()))
                    .cloned()
                {
                    speaker_aliases.insert(local_speaker.clone(), global_speaker.clone());
                    used_global_speakers.insert(global_speaker.clone());
                    warnings.push(format!(
                        "Granite SAA chunk {} remapped local speaker label {} to {} after detecting a label reset.",
                        chunk_index + 1,
                        local_speaker,
                        global_speaker
                    ));
                }
            }
        }

        for mut turn in incoming {
            if let Some(mapped) = speaker_aliases.get(turn.speaker.as_str()) {
                turn.speaker = mapped.clone();
            }
            self.push_turn(turn);
        }

        warnings
    }

    fn push_turn(&mut self, mut turn: SpeakerAttributedAsrTurn) {
        turn.text = turn.text.trim().to_string();
        if turn.text.is_empty() {
            return;
        }

        let Some(last) = self.turns.last_mut() else {
            self.turns.push(turn);
            return;
        };

        let overlap = overlap_prefix_word_count(
            last.text.as_str(),
            turn.text.as_str(),
            GRANITE_SAA_MIN_OVERLAP_WORDS,
            GRANITE_SAA_MAX_OVERLAP_WORDS,
        );
        if overlap > 0 {
            turn.text = drop_prefix_words(turn.text.as_str(), overlap);
            if turn.text.trim().is_empty() {
                return;
            }
        }

        if last.speaker == turn.speaker {
            append_with_spacing(&mut last.text, turn.text.trim());
        } else {
            self.turns.push(turn);
        }
    }

    fn text(&self) -> String {
        format_saa_turns(&self.turns)
    }

    fn prefix_text(&self) -> String {
        let mut selected = Vec::<String>::new();
        let mut selected_chars = 0usize;

        for turn in self.turns.iter().rev().take(GRANITE_SAA_PREFIX_MAX_TURNS) {
            let formatted = format_saa_turn(turn);
            let separator_chars = usize::from(!selected.is_empty());
            let formatted_chars = formatted.chars().count();
            if selected_chars + separator_chars + formatted_chars <= GRANITE_SAA_PREFIX_MAX_CHARS {
                selected.push(formatted);
                selected_chars += separator_chars + formatted_chars;
                continue;
            }

            if selected.is_empty() {
                let label_overhead = turn.speaker.chars().count() + "[]: ".chars().count();
                let text_budget = GRANITE_SAA_PREFIX_MAX_CHARS.saturating_sub(label_overhead);
                let suffix = recent_word_suffix(turn.text.as_str(), text_budget);
                if !suffix.is_empty() {
                    selected.push(format!("[{}]: {}", turn.speaker, suffix));
                }
            }
            break;
        }

        selected.reverse();
        selected.join(" ")
    }
}

fn speaker_attributed_asr_turns_from_text(raw_text: &str) -> Vec<SpeakerAttributedAsrTurn> {
    let parsed = parse_granite_speech_output(raw_text);
    if parsed.segments.is_empty() {
        let text = parsed.text.trim();
        if text.is_empty() {
            return Vec::new();
        }
        return vec![SpeakerAttributedAsrTurn {
            speaker: UNKNOWN_SAA_SPEAKER.to_string(),
            text: text.to_string(),
            start_secs: None,
            end_secs: None,
        }];
    }

    parsed
        .segments
        .into_iter()
        .map(|segment| SpeakerAttributedAsrTurn {
            speaker: segment
                .speaker
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or(UNKNOWN_SAA_SPEAKER)
                .to_string(),
            text: segment.text.trim().to_string(),
            start_secs: None,
            end_secs: None,
        })
        .filter(|turn| !turn.text.is_empty())
        .fold(Vec::<SpeakerAttributedAsrTurn>::new(), |mut turns, turn| {
            if let Some(last) = turns.last_mut() {
                if last.speaker == turn.speaker {
                    append_with_spacing(&mut last.text, turn.text.as_str());
                    return turns;
                }
            }
            turns.push(turn);
            turns
        })
}

fn granite_saa_should_use_single_pass(duration_secs: f32, model_limit_secs: Option<f32>) -> bool {
    let limit = model_limit_secs
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(GRANITE_SAA_HARD_MAX_CHUNK_SECS);
    duration_secs.is_finite() && duration_secs <= limit
}

fn granite_saa_long_form_config(model_limit_secs: Option<f32>) -> AsrLongFormConfig {
    let mut cfg = AsrLongFormConfig {
        target_chunk_secs: GRANITE_SAA_TARGET_CHUNK_SECS,
        hard_max_chunk_secs: GRANITE_SAA_HARD_MAX_CHUNK_SECS,
        overlap_secs: GRANITE_SAA_OVERLAP_SECS,
        min_chunk_secs: GRANITE_SAA_MIN_CHUNK_SECS,
        silence_search_secs: GRANITE_SAA_SILENCE_SEARCH_SECS,
        min_word_overlap: GRANITE_SAA_MIN_OVERLAP_WORDS,
        max_word_overlap: GRANITE_SAA_MAX_OVERLAP_WORDS,
        min_context_replay_words: GRANITE_SAA_MIN_OVERLAP_WORDS,
        max_context_replay_words: GRANITE_SAA_MAX_OVERLAP_WORDS,
        ..Default::default()
    };

    if let Some(limit) = model_limit_secs.filter(|value| value.is_finite() && *value > 0.0) {
        cfg.hard_max_chunk_secs = cfg.hard_max_chunk_secs.min(limit * 0.95);
    }
    if let Some(value) = env_positive_f32("IZWI_GRANITE_SAA_CHUNK_TARGET_SECS") {
        cfg.target_chunk_secs = value;
    }
    if let Some(value) = env_positive_f32("IZWI_GRANITE_SAA_CHUNK_MAX_SECS") {
        cfg.hard_max_chunk_secs = value;
    }
    if let Some(value) = env_positive_f32("IZWI_GRANITE_SAA_CHUNK_OVERLAP_SECS") {
        cfg.overlap_secs = value;
    }
    if let Some(value) = env_positive_f32("IZWI_GRANITE_SAA_CHUNK_MIN_SECS") {
        cfg.min_chunk_secs = value;
    }
    if let Some(value) = env_positive_f32("IZWI_GRANITE_SAA_CHUNK_SILENCE_SEARCH_SECS") {
        cfg.silence_search_secs = value;
    }

    if let Some(limit) = model_limit_secs.filter(|value| value.is_finite() && *value > 0.0) {
        cfg.hard_max_chunk_secs = cfg.hard_max_chunk_secs.min(limit * 0.95);
    }
    cfg.hard_max_chunk_secs = cfg
        .hard_max_chunk_secs
        .max(cfg.min_chunk_secs.max(1.0))
        .min(GRANITE_SAA_HARD_MAX_CHUNK_SECS);
    cfg.target_chunk_secs = cfg
        .target_chunk_secs
        .max(cfg.min_chunk_secs.max(1.0))
        .min(cfg.hard_max_chunk_secs);
    cfg.overlap_secs = cfg.overlap_secs.clamp(0.0, cfg.target_chunk_secs * 0.45);
    cfg
}

fn granite_saa_chunk_plan(
    samples: &[f32],
    sample_rate: u32,
    model_limit_secs: Option<f32>,
) -> Vec<AudioChunk> {
    let cfg = granite_saa_long_form_config(model_limit_secs);
    plan_audio_chunks(samples, sample_rate, &cfg, Some(cfg.hard_max_chunk_secs))
}

fn granite_saa_max_new_tokens_for_duration(duration_secs: f32) -> usize {
    if let Some(override_tokens) = env_positive_usize("IZWI_GRANITE_SAA_MAX_NEW_TOKENS") {
        return override_tokens;
    }

    let duration_budget = if duration_secs.is_finite() && duration_secs > 0.0 {
        (duration_secs * GRANITE_SAA_NEW_TOKENS_PER_SECOND).ceil() as usize
    } else {
        0
    };
    GRANITE_SAA_NEW_TOKEN_RESERVE
        .saturating_add(duration_budget)
        .clamp(GRANITE_SAA_MIN_NEW_TOKENS, GRANITE_SAA_MAX_NEW_TOKENS)
}

fn granite_saa_max_new_tokens(audio: &[f32], sample_rate: u32) -> usize {
    let duration_secs = if sample_rate > 0 {
        audio.len() as f32 / sample_rate as f32
    } else {
        0.0
    };
    granite_saa_max_new_tokens_for_duration(duration_secs)
}

fn granite_saa_chunk_prefix_text(
    assembler: &GraniteSaaTranscriptAssembler,
    mode: GraniteSaaPrefixMode,
) -> Option<String> {
    match mode {
        GraniteSaaPrefixMode::None => None,
        GraniteSaaPrefixMode::FullTranscript => {
            let prefix_text = assembler.prefix_text();
            (!prefix_text.trim().is_empty()).then_some(prefix_text)
        }
    }
}

fn granite_saa_decode_diagnostics(
    diagnostics: Option<&serde_json::Value>,
) -> GraniteSaaChunkDecodeDiagnostics {
    let Some(diagnostics) = diagnostics else {
        return GraniteSaaChunkDecodeDiagnostics::default();
    };
    let decode = diagnostics.get("decode").unwrap_or(diagnostics);
    let prompt = diagnostics.get("prompt");

    GraniteSaaChunkDecodeDiagnostics {
        generated_tokens: decode
            .get("generated_tokens")
            .or_else(|| diagnostics.get("generated_tokens"))
            .and_then(json_usize),
        max_new_tokens: decode.get("max_new_tokens").and_then(json_usize),
        stop_reason: decode
            .get("stop_reason")
            .or_else(|| diagnostics.get("stop_reason"))
            .and_then(|value| value.as_str())
            .map(ToOwned::to_owned),
        prompt_tokens: prompt
            .and_then(|value| value.get("prompt_tokens"))
            .or_else(|| diagnostics.get("prompt_tokens"))
            .and_then(json_usize),
        prompt_prefix_tokens: prompt
            .and_then(|value| value.get("prefix_tokens"))
            .or_else(|| diagnostics.get("prompt_prefix_tokens"))
            .and_then(json_usize),
    }
}

fn granite_saa_max_token_warning(
    chunk_index: usize,
    diagnostics: &GraniteSaaChunkDecodeDiagnostics,
) -> Option<String> {
    if diagnostics.stop_reason.as_deref() != Some("max_tokens") {
        return None;
    }

    let generated = diagnostics
        .generated_tokens
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let max_new_tokens = diagnostics
        .max_new_tokens
        .map(|value| value.to_string())
        .unwrap_or_else(|| "configured".to_string());
    Some(format!(
        "Granite SAA chunk {} reached the decode token limit ({generated}/{max_new_tokens}); transcript may be incomplete.",
        chunk_index + 1
    ))
}

fn json_usize(value: &serde_json::Value) -> Option<usize> {
    value.as_u64().and_then(|value| usize::try_from(value).ok())
}

async fn granite_saa_long_form_transcribe<P>(
    coordinator: &Arc<InferenceCoordinator>,
    job: &JobLease,
    model: Arc<NativeAsrModel>,
    execution_contract: LoadedExecutionContract,
    state_binding: LoadedCapabilityBinding,
    samples: Arc<[f32]>,
    sample_rate: u32,
    language: Option<&str>,
    model_limit_secs: Option<f32>,
    residency_lease: ModelResidencyLease,
    retained_input_bytes: usize,
    on_progress: &mut P,
) -> Result<GraniteSaaLongFormOutput>
where
    P: FnMut(AsrProgress) + Send + 'static,
{
    let planning_samples = samples.clone();
    let planning_job = job.clone();
    let (mut residency_lease, chunks) = coordinator
        .run_host_blocking_stage(job, move || {
            planning_job.record_materialized_usage(decoded_audio_observation(
                retained_input_bytes,
                planning_samples.len(),
            )?)?;
            Ok((
                residency_lease,
                granite_saa_chunk_plan(&planning_samples, sample_rate, model_limit_secs),
            ))
        })
        .await?;
    if chunks.is_empty() {
        return Err(Error::InvalidInput(
            "Granite SAA chunk planner produced no chunks".to_string(),
        ));
    }

    on_progress(granite_saa_processing_progress(&chunks, sample_rate));

    let mut assembler = GraniteSaaTranscriptAssembler::default();
    let mut language_out = language.map(ToOwned::to_owned);
    let mut warnings = vec![format!(
        "Granite SAA processed long audio in {} chunks; speaker label continuity across chunks is best-effort.",
        chunks.len()
    )];

    for (idx, chunk) in chunks.iter().enumerate() {
        if chunk.end_sample <= chunk.start_sample || chunk.end_sample > samples.len() {
            warnings.push(format!(
                "Granite SAA skipped invalid chunk {} with sample range {}..{}.",
                idx + 1,
                chunk.start_sample,
                chunk.end_sample
            ));
            continue;
        }

        on_progress(granite_saa_chunk_progress(
            AsrProgressPhase::ChunkStarted,
            idx,
            chunk,
            &chunks,
            sample_rate,
        ));

        let chunk_start = chunk.start_sample;
        let chunk_end = chunk.end_sample;
        let task_model = model.clone();
        let task_samples = samples.clone();
        let observation_job = job.clone();
        let task_state_binding = state_binding.clone();
        let language_owned = language.map(ToOwned::to_owned);
        let prefix_mode = GraniteSaaPrefixMode::from_env();
        let prefix_text = granite_saa_chunk_prefix_text(&assembler, prefix_mode);
        let prefix_chars = prefix_text
            .as_deref()
            .map(|text| text.chars().count())
            .unwrap_or(0);
        let chunk_sample_count = chunk_end - chunk_start;
        let max_new_tokens =
            granite_saa_max_new_tokens(&samples[chunk_start..chunk_end], sample_rate);
        let chunk_duration_secs = samples_to_seconds_f64(chunk_sample_count, sample_rate);
        tracing::info!(
            chunk_index = idx + 1,
            total_chunks = chunks.len(),
            chunk_start_secs = samples_to_seconds_f64(chunk.start_sample, sample_rate),
            chunk_end_secs = samples_to_seconds_f64(chunk.end_sample, sample_rate),
            chunk_duration_secs,
            max_new_tokens,
            prefix_mode = prefix_mode.as_str(),
            prefix_chars,
            "starting Granite SAA chunk decode"
        );
        let chunk_started = Instant::now();
        let (returned_lease, transcription) = coordinator
            .run_loaded_blocking_stage_with_invocation_paged(
                job,
                execution_contract.clone(),
                task_state_binding,
                WorkUnit::PipelineStage {
                    name: "asr.speaker_attributed.chunk".to_string(),
                    ordinal: idx,
                },
                move |leases| {
                    let chunk_audio = task_samples[chunk_start..chunk_end].to_vec();
                    observation_job.record_materialized_usage(
                        decoded_audio_with_scratch_observation(
                            retained_input_bytes,
                            task_samples.len(),
                            chunk_audio.capacity(),
                        )?,
                    )?;
                    let transcription = granite_saa_transcribe_chunk(
                        &task_model,
                        &chunk_audio,
                        sample_rate,
                        language_owned.as_deref(),
                        prefix_text.as_deref(),
                        max_new_tokens,
                        granite_saa_single_invocation_cache(leases)?,
                    );
                    let steady_usage =
                        decoded_audio_observation(retained_input_bytes, task_samples.len())?;
                    observation_job.prepare_materialized_release(steady_usage)?;
                    drop(chunk_audio);
                    Ok((residency_lease, transcription?))
                },
            )
            .await?;
        residency_lease = returned_lease;
        let decode_diagnostics = granite_saa_decode_diagnostics(transcription.diagnostics.as_ref());
        tracing::info!(
            chunk_index = idx + 1,
            total_chunks = chunks.len(),
            elapsed_ms = chunk_started.elapsed().as_secs_f64() * 1000.0,
            generated_tokens = ?decode_diagnostics.generated_tokens,
            max_new_tokens = decode_diagnostics.max_new_tokens.unwrap_or(max_new_tokens),
            stop_reason = decode_diagnostics
                .stop_reason
                .as_deref()
                .unwrap_or("unknown"),
            prompt_tokens = ?decode_diagnostics.prompt_tokens,
            prompt_prefix_tokens = ?decode_diagnostics.prompt_prefix_tokens,
            "finished Granite SAA chunk decode"
        );

        if language_out.is_none() {
            language_out = transcription.language.clone();
        }
        if transcription.text.trim().is_empty() {
            warnings.push(format!(
                "Granite SAA chunk {} returned empty text.",
                idx + 1
            ));
        }
        if let Some(warning) = granite_saa_max_token_warning(idx, &decode_diagnostics) {
            tracing::warn!(
                chunk_index = idx + 1,
                warning = warning.as_str(),
                "Granite SAA chunk reached decode token limit"
            );
            warnings.push(warning);
        }
        warnings.extend(assembler.push_chunk_text(transcription.text.as_str(), idx));

        on_progress(granite_saa_chunk_progress(
            AsrProgressPhase::ChunkFinished,
            idx,
            chunk,
            &chunks,
            sample_rate,
        ));
    }

    on_progress(granite_saa_complete_progress(&chunks, sample_rate));

    Ok(GraniteSaaLongFormOutput {
        text: assembler.text(),
        language: language_out,
        warnings,
    })
}

fn granite_saa_transcribe_chunk(
    model: &NativeAsrModel,
    audio: &[f32],
    sample_rate: u32,
    language: Option<&str>,
    prefix_text: Option<&str>,
    max_new_tokens: usize,
    cache: &mut crate::models::shared::attention::physical::PhysicalPagedKvCache,
) -> Result<NativeAsrTranscription> {
    model.transcribe_granite_speech_task_and_options_physical(
        audio,
        sample_rate,
        language,
        GraniteSpeechTask::SpeakerAttributed,
        prefix_text,
        NativeAsrGenerationOptions {
            max_new_tokens,
            ..NativeAsrGenerationOptions::default()
        },
        cache,
    )
}

fn granite_saa_single_invocation_cache(
    leases: &mut crate::kv::v2::InvocationPagedLeaseSetV2,
) -> Result<&mut crate::models::shared::attention::physical::PhysicalPagedKvCache> {
    let domains = leases.domains().collect::<Vec<_>>();
    let [domain] = domains.as_slice() else {
        return Err(Error::InferenceError(format!(
            "Granite speaker-attributed ASR requires one invocation KV domain, found {}",
            domains.len()
        )));
    };
    leases.cache_mut(*domain)
}

fn granite_saa_processing_progress(chunks: &[AudioChunk], sample_rate: u32) -> AsrProgress {
    let total_audio_secs = chunks
        .last()
        .map(|chunk| samples_to_seconds_f64(chunk.end_sample, sample_rate));
    AsrProgress {
        phase: AsrProgressPhase::Processing,
        current_chunk: None,
        total_chunks: Some(chunks.len()),
        processed_audio_secs: Some(0.0),
        total_audio_secs,
        percent: Some(0.0),
    }
}

fn granite_saa_complete_progress(chunks: &[AudioChunk], sample_rate: u32) -> AsrProgress {
    let total_audio_secs = chunks
        .last()
        .map(|chunk| samples_to_seconds_f64(chunk.end_sample, sample_rate));
    AsrProgress {
        phase: AsrProgressPhase::Complete,
        current_chunk: Some(chunks.len()),
        total_chunks: Some(chunks.len()),
        processed_audio_secs: total_audio_secs,
        total_audio_secs,
        percent: Some(100.0),
    }
}

fn granite_saa_chunk_progress(
    phase: AsrProgressPhase,
    index: usize,
    chunk: &AudioChunk,
    chunks: &[AudioChunk],
    sample_rate: u32,
) -> AsrProgress {
    let total_audio_secs = chunks
        .last()
        .map(|last| samples_to_seconds_f64(last.end_sample, sample_rate));
    let processed_audio_secs = match phase {
        AsrProgressPhase::ChunkStarted => samples_to_seconds_f64(chunk.start_sample, sample_rate),
        AsrProgressPhase::ChunkFinished => samples_to_seconds_f64(chunk.end_sample, sample_rate),
        AsrProgressPhase::Processing => 0.0,
        AsrProgressPhase::Aligning | AsrProgressPhase::Complete => {
            total_audio_secs.unwrap_or_default()
        }
    };
    let percent = total_audio_secs
        .filter(|total| *total > 0.0)
        .map(|total| ((processed_audio_secs / total) * 100.0).clamp(0.0, 100.0));

    AsrProgress {
        phase,
        current_chunk: Some(index + 1),
        total_chunks: Some(chunks.len()),
        processed_audio_secs: Some(processed_audio_secs),
        total_audio_secs,
        percent,
    }
}

fn samples_to_seconds_f64(samples: usize, sample_rate: u32) -> f64 {
    if sample_rate == 0 {
        0.0
    } else {
        samples as f64 / sample_rate as f64
    }
}

fn env_positive_f32(key: &str) -> Option<f32> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn env_positive_usize(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn speaker_attributed_asr_result_from_text(
    raw_text: &str,
    language: Option<String>,
    duration_secs: f32,
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
) -> SpeakerAttributedAsrResult {
    speaker_attributed_asr_result_from_text_with_warnings(
        raw_text,
        language,
        duration_secs,
        min_speakers,
        max_speakers,
        Vec::new(),
    )
}

fn speaker_attributed_asr_result_from_text_with_warnings(
    raw_text: &str,
    language: Option<String>,
    duration_secs: f32,
    min_speakers: Option<usize>,
    max_speakers: Option<usize>,
    mut warnings: Vec<String>,
) -> SpeakerAttributedAsrResult {
    let parsed = parse_granite_speech_output(raw_text);
    let mut speakers = HashSet::<String>::new();
    let mut turns = speaker_attributed_asr_turns_from_text(raw_text);

    for turn in &turns {
        let speaker = turn.speaker.trim();
        if speaker != UNKNOWN_SAA_SPEAKER {
            speakers.insert(speaker.to_string());
        }
    }
    if turns.is_empty() && !parsed.text.trim().is_empty() {
        turns.push(SpeakerAttributedAsrTurn {
            speaker: UNKNOWN_SAA_SPEAKER.to_string(),
            text: parsed.text.trim().to_string(),
            start_secs: None,
            end_secs: None,
        });
    }

    let speaker_count = speakers.len();
    if let Some(min_speakers) = min_speakers {
        if speaker_count < min_speakers {
            warnings.push(format!(
                "Granite SAA emitted {speaker_count} speaker label(s), below requested minimum {min_speakers}."
            ));
        }
    }
    if let Some(max_speakers) = max_speakers {
        if speaker_count > max_speakers {
            warnings.push(format!(
                "Granite SAA emitted {speaker_count} speaker label(s), above requested maximum {max_speakers}."
            ));
        }
    }
    if parsed.text.trim().is_empty() {
        warnings.push("Granite SAA returned an empty transcript.".to_string());
    }

    SpeakerAttributedAsrResult {
        text: parsed.text,
        language,
        duration_secs,
        speaker_turns: turns,
        speaker_count,
        status: if warnings.is_empty() {
            SpeakerAttributedAsrStatus::Ready
        } else {
            SpeakerAttributedAsrStatus::Warning
        },
        warnings,
    }
}

fn format_saa_turns(turns: &[SpeakerAttributedAsrTurn]) -> String {
    turns
        .iter()
        .map(format_saa_turn)
        .collect::<Vec<_>>()
        .join(" ")
}

fn format_saa_turn(turn: &SpeakerAttributedAsrTurn) -> String {
    format!("[{}]: {}", turn.speaker, turn.text.trim())
}

fn append_with_spacing(target: &mut String, text: &str) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    if !target.trim().is_empty() && !target.ends_with(char::is_whitespace) {
        target.push(' ');
    }
    target.push_str(text);
}

fn overlap_prefix_word_count(
    existing: &str,
    incoming: &str,
    min_words: usize,
    max_words: usize,
) -> usize {
    let existing_words = normalized_overlap_words(existing);
    let incoming_words = normalized_overlap_words(incoming);
    let max_words = max_words
        .min(existing_words.len())
        .min(incoming_words.len());
    if max_words < min_words {
        return 0;
    }

    for count in (min_words..=max_words).rev() {
        if existing_words[existing_words.len() - count..] == incoming_words[..count] {
            return count;
        }
    }
    0
}

fn normalized_overlap_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter_map(normalize_overlap_word)
        .collect()
}

fn normalize_overlap_word(word: &str) -> Option<String> {
    let normalized = word
        .chars()
        .filter(|ch| ch.is_alphanumeric() || *ch == '\'')
        .flat_map(char::to_lowercase)
        .collect::<String>();
    (!normalized.is_empty()).then_some(normalized)
}

fn drop_prefix_words(text: &str, words_to_drop: usize) -> String {
    text.split_whitespace()
        .skip(words_to_drop)
        .collect::<Vec<_>>()
        .join(" ")
}

fn recent_word_suffix(text: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }

    let mut selected = Vec::<&str>::new();
    let mut selected_chars = 0usize;
    for word in text.split_whitespace().rev() {
        let separator_chars = usize::from(!selected.is_empty());
        let word_chars = word.chars().count();
        if selected_chars + separator_chars + word_chars > max_chars {
            break;
        }
        selected.push(word);
        selected_chars += separator_chars + word_chars;
    }
    selected.reverse();
    selected.join(" ")
}

pub(crate) fn resolve_forced_aligner_variant(model_id: Option<&str>) -> Result<ModelVariant> {
    let variant = match model_id {
        Some(raw) => {
            parse_model_variant(raw).map_err(|err| Error::InvalidInput(err.to_string()))?
        }
        None => ModelVariant::Qwen3ForcedAligner06B,
    };

    if !variant.is_forced_aligner() {
        return Err(Error::InvalidInput(format!(
            "Model {} is not a forced aligner model",
            variant.dir_name()
        )));
    }

    Ok(variant)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::backends::BackendPreference;
    use crate::config::EngineConfig;
    use crate::engine::{EngineCoreRequest, ModelInstanceId, Priority, WorkloadClass};
    use crate::runtime::adapters::{LoadedModelBundleDraft, RuntimeAdapterRegistry};
    use crate::runtime::coordinator::JobSpec;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Condvar, Mutex, OnceLock};
    use uuid::Uuid;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn realtime_stream_variant_resolves_engine_voxtral_and_direct_nemotron() {
        assert_eq!(
            resolve_asr_realtime_stream_variant(Some("nvidia/nemotron-3.5-asr-streaming-0.6b",)),
            Some(ModelVariant::Nemotron35AsrStreaming06B)
        );
        assert_eq!(
            resolve_asr_realtime_stream_variant(Some("Nemotron 3.5 ASR Streaming 0.6B")),
            Some(ModelVariant::Nemotron35AsrStreaming06B)
        );
        assert_eq!(
            resolve_asr_realtime_stream_variant(Some("Voxtral-Mini-4B-Realtime-2602")),
            Some(ModelVariant::VoxtralMini4BRealtime2602)
        );

        assert_eq!(resolve_asr_realtime_stream_variant(None), None);
        assert_eq!(
            resolve_asr_realtime_stream_variant(Some("Qwen3-ASR-1.7B")),
            None
        );
        assert_eq!(
            resolve_asr_realtime_stream_variant(Some("Whisper-Large-v3-Turbo")),
            None
        );
        assert_eq!(
            resolve_asr_realtime_stream_variant(Some("Parakeet-TDT-0.6B-v3")),
            None
        );
        assert_eq!(
            resolve_asr_realtime_stream_variant(Some("not-a-real-model")),
            None
        );
    }

    #[test]
    fn realtime_stream_reservation_routes_to_backend_memory_domains() {
        let cpu = realtime_stream_resource_vector(BackendKind::Cpu, 11, 29).unwrap();
        let metal = realtime_stream_resource_vector(BackendKind::Metal, 11, 29).unwrap();
        let cuda = realtime_stream_resource_vector(BackendKind::Cuda, 11, 29).unwrap();

        assert_eq!(cpu.host_bytes, ResourceAmount::Known(40));
        assert_eq!(cpu.device_bytes, ResourceAmount::Known(0));
        assert_eq!(cpu.unified_bytes, ResourceAmount::Known(0));
        assert_eq!(metal.host_bytes, ResourceAmount::Known(0));
        assert_eq!(metal.device_bytes, ResourceAmount::Known(0));
        assert_eq!(metal.unified_bytes, ResourceAmount::Known(40));
        assert_eq!(cuda.host_bytes, ResourceAmount::Known(11));
        assert_eq!(cuda.device_bytes, ResourceAmount::Known(29));
        assert_eq!(cuda.unified_bytes, ResourceAmount::Known(0));
    }

    #[test]
    fn voxtral_operation_cost_uses_preparation_geometry_not_raw_samples() {
        let geometry = VoxtralRealtimePreparationGeometry {
            source_samples: 160,
            resampled_samples: 160,
            padded_samples: 320,
            mel_frames: 8,
            conv1_frames: 4,
            conv2_frames: 2,
            pooled_frames: 1,
            stable_frames: 1,
            embedding_elements: 32,
        };
        let batch = VoxtralRealtimePreparationBatchGeometry {
            width: 1,
            padded_mel_frames: 8,
            padded_conv_frames: 2,
            materialized_tensor_elements_per_row: 704,
            workspace_per_row_bytes: 4096,
        };
        let seal = VoxtralRealtimePreparationStageSeal {
            max_source_samples: 1_600,
            max_work_units: 1_600,
            max_materialized_tensor_elements_per_row: 1_024,
            max_workspace_bytes: 8192,
        };

        let cost = voxtral_realtime_preparation_cost_from_geometry(geometry, batch, seal).unwrap();

        assert_eq!(cost.logical_units, 160);
        assert_eq!(cost.tensor_elements, 704);
        assert_eq!(cost.workspace.temporary_bytes, ResourceAmount::Known(4096));
        assert_ne!(cost.tensor_elements, geometry.source_samples as u64);

        let undersized_seal = VoxtralRealtimePreparationStageSeal {
            max_materialized_tensor_elements_per_row: 703,
            ..seal
        };
        assert!(
            voxtral_realtime_preparation_cost_from_geometry(geometry, batch, undersized_seal)
                .is_err()
        );
    }

    #[test]
    fn voxtral_committed_observation_excludes_transient_workspace() {
        let observed = voxtral_realtime_committed_observation(
            11,
            VoxtralRealtimePreparedResourceUsage {
                host_bytes: 40,
                tensor_bytes: 96,
            },
        )
        .unwrap();

        assert_eq!(observed, JobResourceObservation::new(62, 96));
    }

    #[test]
    fn realtime_session_quota_is_held_until_all_in_flight_clones_exit() {
        let policy = RealtimeAsrSessionPolicy::new(RealtimeAsrSessionLimits {
            max_sessions: 1,
            max_lifetime: Duration::from_secs(60),
            idle_timeout: Duration::from_secs(10),
        })
        .expect("session policy");
        let stream_lease = policy.try_acquire().expect("first session");
        let in_flight_lease = stream_lease.clone();

        drop(stream_lease);
        assert!(matches!(policy.try_acquire(), Err(Error::Overloaded(_))));

        drop(in_flight_lease);
        assert!(policy.try_acquire().is_ok());
    }

    #[test]
    fn realtime_session_policy_rejects_quota_that_would_panic_semaphore() {
        let result = RealtimeAsrSessionPolicy::new(RealtimeAsrSessionLimits {
            max_sessions: Semaphore::MAX_PERMITS + 1,
            max_lifetime: Duration::from_secs(60),
            idle_timeout: Duration::from_secs(10),
        });

        assert!(matches!(result, Err(Error::ConfigError(_))));
    }

    #[test]
    fn realtime_session_expiration_distinguishes_idle_and_absolute_limits() {
        let started_at = Instant::now();
        let idle_deadline = started_at + Duration::from_secs(2);
        let absolute_deadline = started_at + Duration::from_secs(10);

        assert_eq!(
            realtime_asr_session_expiration(idle_deadline, absolute_deadline, idle_deadline, 0,),
            Some("realtime ASR stream exceeded its idle timeout")
        );
        assert_eq!(
            realtime_asr_session_expiration(idle_deadline, absolute_deadline, idle_deadline, 1,),
            None
        );
        assert_eq!(
            realtime_asr_session_expiration(absolute_deadline, absolute_deadline, idle_deadline, 1,),
            Some("realtime ASR stream exceeded its absolute lifetime")
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn realtime_watchdog_releases_idle_session_quota() {
        let policy = RealtimeAsrSessionPolicy::new(RealtimeAsrSessionLimits {
            max_sessions: 1,
            max_lifetime: Duration::from_secs(60),
            idle_timeout: Duration::from_millis(1),
        })
        .expect("session policy");
        let session_lease = policy.try_acquire().expect("first session");
        let now = Instant::now();
        let resources = Arc::new(StdMutex::new(RuntimeAsrRealtimeResources {
            model: None,
            state: None,
            execution_contract: None,
            residency_lease: None,
            job: None,
            session_lease: Some(session_lease),
            engine_session: None,
            voxtral_model: None,
            absolute_deadline: now + Duration::from_secs(60),
            idle_timeout: Duration::from_millis(1),
            last_activity: now - Duration::from_secs(1),
            active_operations: 0,
            closed: false,
            timeout_reason: None,
        }));
        let activity = Arc::new(Notify::new());
        spawn_realtime_asr_watchdog(&resources, activity);

        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let closed = resources.lock().expect("resource lock").closed;
                if closed && policy.permits.available_permits() == 1 {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("watchdog cleanup");

        let resources = resources.lock().expect("resource lock");
        assert_eq!(
            resources.timeout_reason,
            Some("realtime ASR stream exceeded its idle timeout")
        );
        assert!(resources.session_lease.is_none());
        drop(resources);
        assert!(policy.try_acquire().is_ok());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn realtime_stream_drop_releases_detached_session_quota() {
        let policy = RealtimeAsrSessionPolicy::new(RealtimeAsrSessionLimits {
            max_sessions: 1,
            max_lifetime: Duration::from_secs(60),
            idle_timeout: Duration::from_secs(60),
        })
        .expect("session policy");
        let session_lease = policy.try_acquire().expect("first session");
        let now = Instant::now();
        let resources = Arc::new(StdMutex::new(RuntimeAsrRealtimeResources {
            model: None,
            state: None,
            execution_contract: None,
            residency_lease: None,
            job: None,
            session_lease: Some(session_lease),
            engine_session: None,
            voxtral_model: None,
            absolute_deadline: now + Duration::from_secs(60),
            idle_timeout: Duration::from_secs(60),
            last_activity: now,
            active_operations: 0,
            closed: false,
            timeout_reason: None,
        }));
        let stream = RuntimeAsrRealtimeStream {
            variant: ModelVariant::VoxtralMini4BRealtime2602,
            resources,
            activity: Arc::new(Notify::new()),
            operation_gate: Arc::new(tokio::sync::Mutex::new(())),
            max_samples: 1,
            committed_samples: 0,
            engine_sample_rate: None,
            engine_text: String::new(),
            engine_chunk_index: 0,
        };

        drop(stream);
        tokio::time::timeout(Duration::from_secs(1), async {
            while policy.permits.available_permits() != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("drop cleanup");
        assert!(policy.try_acquire().is_ok());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn aborted_cleanup_task_retains_detached_authorities_fail_closed() {
        let policy = RealtimeAsrSessionPolicy::new(RealtimeAsrSessionLimits {
            max_sessions: 1,
            max_lifetime: Duration::from_secs(60),
            idle_timeout: Duration::from_secs(60),
        })
        .expect("session policy");
        let session_lease = policy.try_acquire().expect("first session");
        let cleanup = RealtimeAsrDetachedResources {
            state: None,
            model: None,
            execution_contract: None,
            residency_lease: None,
            job: None,
            session_lease: Some(session_lease),
            engine_session: None,
            voxtral_model: None,
        };
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            let _guard = RealtimeAsrCleanupGuard::for_test(cleanup);
            let _ = entered_tx.send(());
            std::future::pending::<()>().await;
        });
        entered_rx.await.expect("cleanup task entered");

        task.abort();
        let _ = task.await;
        assert_eq!(policy.permits.available_permits(), 0);
        assert!(policy.try_acquire().is_err());
    }

    fn realtime_test_job(id: &str, backend: BackendKind, deadline: Option<Instant>) -> JobSpec {
        let mut resources = ResourceVector::zero();
        match backend {
            BackendKind::Cpu => resources.host_bytes = ResourceAmount::Known(1024 * 1024),
            BackendKind::Metal => resources.unified_bytes = ResourceAmount::Known(1024 * 1024),
            BackendKind::Cuda => {
                resources.host_bytes = ResourceAmount::Known(64 * 1024);
                resources.device_bytes = ResourceAmount::Known(1024 * 1024);
            }
        }
        JobSpec {
            request_id: id.to_string(),
            lane: CoordinatorLane::Realtime,
            priority: Priority::Normal,
            workload_class: WorkloadClass::Realtime,
            deadline,
            resources,
        }
    }

    fn realtime_test_contract(
        coordinator: &InferenceCoordinator,
        backend: BackendKind,
    ) -> LoadedExecutionContract {
        let draft = LoadedModelBundleDraft::build(
            &RuntimeAdapterRegistry::built_in(),
            coordinator.execution_group_id(),
            ModelInstanceId::new(1),
            ModelVariant::Nemotron35AsrStreaming06B,
            backend,
        )
        .expect("realtime test bundle draft");
        draft
            .execution_contracts(CapabilityKind::RealtimeAsr)
            .expect("realtime test contracts")
            .into_iter()
            .next()
            .expect("realtime adapter publishes a contract")
    }

    #[test]
    fn oversized_realtime_packet_is_rejected_before_copy_or_model_work() {
        let side_effects = AtomicUsize::new(0);
        let result = validate_realtime_input_copy(4_801, 4_800).map(|()| {
            side_effects.fetch_add(1, Ordering::Relaxed);
            let _owned = vec![0.0_f32; 4_801];
            side_effects.fetch_add(1, Ordering::Relaxed);
        });

        assert!(matches!(result, Err(Error::InvalidInput(_))));
        assert_eq!(side_effects.load(Ordering::Relaxed), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn realtime_blocking_work_respects_deadline_without_blocking_tokio() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
        let job = coordinator
            .admit(realtime_test_job(
                "realtime-blocking-deadline",
                BackendKind::Cpu,
                Some(Instant::now() + std::time::Duration::from_millis(200)),
            ))
            .await
            .unwrap();
        let runner_job = job.clone();
        drop(job);
        let gate = Arc::new(tokio::sync::Mutex::new(()));
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let task_release = release.clone();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let task_coordinator = coordinator.clone();
        let execution_contract = realtime_test_contract(&coordinator, BackendKind::Cpu);
        let runner = tokio::spawn(async move {
            run_realtime_blocking_operation(
                &task_coordinator,
                &runner_job,
                execution_contract,
                gate,
                "asr.realtime.test",
                move || {
                    Ok(move || {
                        let _ = started_tx.send(());
                        let (lock, wake) = &*task_release;
                        let mut released = lock.lock().unwrap_or_else(|poison| poison.into_inner());
                        while !*released {
                            released = wake
                                .wait(released)
                                .unwrap_or_else(|poison| poison.into_inner());
                        }
                        Ok(())
                    })
                },
            )
            .await
        });

        started_rx.await.unwrap();
        let heartbeat = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            tokio::time::sleep(std::time::Duration::from_millis(5)),
        )
        .await;
        let result = tokio::time::timeout(std::time::Duration::from_secs(1), runner).await;
        let while_blocked = coordinator.snapshot();
        {
            let (lock, wake) = &*release;
            *lock.lock().unwrap_or_else(|poison| poison.into_inner()) = true;
            wake.notify_all();
        }
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while coordinator.snapshot().active_executions != 0
                || coordinator.snapshot().active_jobs != 0
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        assert!(heartbeat.is_ok());
        assert!(matches!(
            result,
            Ok(Ok(Err(Error::Timeout(id)))) if id == "realtime-blocking-deadline"
        ));
        assert_eq!(while_blocked.active_executions, 1);
        assert_eq!(while_blocked.active_jobs, 1);
        assert_eq!(coordinator.snapshot().expired_total, 1);
    }

    #[tokio::test]
    async fn cancelled_realtime_call_retains_stream_operation_order() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cuda, 2, 4));
        let job = coordinator
            .admit(realtime_test_job(
                "realtime-cancellation-order",
                BackendKind::Cuda,
                None,
            ))
            .await
            .unwrap();
        let gate = Arc::new(tokio::sync::Mutex::new(()));
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let stage = Arc::new(AtomicUsize::new(0));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();

        let first_coordinator = coordinator.clone();
        let first_job = job.clone();
        let first_gate = gate.clone();
        let first_release = release.clone();
        let first_stage = stage.clone();
        let first_contract = realtime_test_contract(&coordinator, BackendKind::Cuda);
        let first = tokio::spawn(async move {
            run_realtime_blocking_operation(
                &first_coordinator,
                &first_job,
                first_contract,
                first_gate,
                "asr.realtime.test",
                move || {
                    Ok(move || {
                        assert_eq!(first_stage.fetch_add(1, Ordering::SeqCst), 0);
                        let _ = started_tx.send(());
                        let (lock, wake) = &*first_release;
                        let mut released = lock.lock().unwrap_or_else(|poison| poison.into_inner());
                        while !*released {
                            released = wake
                                .wait(released)
                                .unwrap_or_else(|poison| poison.into_inner());
                        }
                        assert_eq!(first_stage.fetch_add(1, Ordering::SeqCst), 1);
                        Ok(())
                    })
                },
            )
            .await
        });
        started_rx.await.unwrap();
        first.abort();
        assert!(first.await.unwrap_err().is_cancelled());

        let second_coordinator = coordinator.clone();
        let second_job = job.clone();
        let second_gate = gate.clone();
        let second_stage = stage.clone();
        let second_contract = realtime_test_contract(&coordinator, BackendKind::Cuda);
        let second = tokio::spawn(async move {
            run_realtime_blocking_operation(
                &second_coordinator,
                &second_job,
                second_contract,
                second_gate,
                "asr.realtime.test",
                move || {
                    Ok(move || {
                        assert_eq!(second_stage.fetch_add(1, Ordering::SeqCst), 2);
                        Ok(())
                    })
                },
            )
            .await
        });

        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert_eq!(stage.load(Ordering::SeqCst), 1);
        {
            let (lock, wake) = &*release;
            *lock.lock().unwrap_or_else(|poison| poison.into_inner()) = true;
            wake.notify_all();
        }
        tokio::time::timeout(std::time::Duration::from_secs(1), second)
            .await
            .unwrap()
            .unwrap()
            .unwrap();

        assert_eq!(stage.load(Ordering::SeqCst), 3);
        assert_eq!(coordinator.snapshot().active_executions, 0);
        drop(job);
        assert_eq!(coordinator.snapshot().active_jobs, 0);
    }

    #[test]
    fn granite_auto_asr_budget_scales_with_audio_duration() {
        assert_eq!(granite_auto_asr_max_tokens_for_duration(0.0), 76);
        assert_eq!(granite_auto_asr_max_tokens_for_duration(3.6), 76);
        assert_eq!(granite_auto_asr_max_tokens_for_duration(27.303175), 76);
        assert_eq!(granite_auto_asr_max_tokens_for_duration(28.0), 76);
        assert_eq!(granite_auto_asr_max_tokens_for_duration(60.0), 204);
        assert_eq!(granite_auto_asr_max_tokens_for_duration(600.0), 2048);
        assert_eq!(granite_auto_asr_max_tokens_for_duration(1200.0), 2048);
    }

    #[test]
    fn granite_auto_asr_request_uses_executor_adjusted_ceiling() {
        assert_eq!(
            granite_auto_asr_token_ceiling(ModelVariant::GraniteSpeech412BPlus),
            Some(GRANITE_ASR_AUTO_MAX_TOKENS)
        );
        assert_eq!(
            granite_auto_asr_token_ceiling(ModelVariant::WhisperLargeV3Turbo),
            None
        );
    }

    #[test]
    fn speaker_attributed_asr_result_preserves_granite_turns() {
        let result = speaker_attributed_asr_result_from_text(
            "[Speaker 1]: hello there [Speaker 2]: hi back",
            Some("English".to_string()),
            4.0,
            Some(2),
            None,
        );

        assert_eq!(result.status, SpeakerAttributedAsrStatus::Ready);
        assert_eq!(result.language.as_deref(), Some("English"));
        assert_eq!(result.speaker_count, 2);
        assert!(result.warnings.is_empty());
        assert_eq!(
            result.speaker_turns,
            vec![
                SpeakerAttributedAsrTurn {
                    speaker: "Speaker 1".to_string(),
                    text: "hello there".to_string(),
                    start_secs: None,
                    end_secs: None,
                },
                SpeakerAttributedAsrTurn {
                    speaker: "Speaker 2".to_string(),
                    text: "hi back".to_string(),
                    start_secs: None,
                    end_secs: None,
                },
            ]
        );
    }

    #[test]
    fn speaker_attributed_asr_warns_when_expected_speakers_are_missing() {
        let result = speaker_attributed_asr_result_from_text(
            "[Speaker 1]: one long turn",
            None,
            2.0,
            Some(2),
            None,
        );

        assert_eq!(result.status, SpeakerAttributedAsrStatus::Warning);
        assert_eq!(result.speaker_count, 1);
        assert_eq!(result.speaker_turns.len(), 1);
        assert!(result.warnings[0].contains("below requested minimum 2"));
    }

    #[test]
    fn granite_saa_chunk_plan_keeps_long_audio_under_model_limit() {
        let sample_rate = 10;
        let samples = vec![0.0f32; 15_565];
        let chunks = granite_saa_chunk_plan(&samples, sample_rate, Some(540.0));

        assert!(chunks.len() > 1);
        assert_eq!(chunks.first().unwrap().start_sample, 0);
        assert_eq!(chunks.last().unwrap().end_sample, samples.len());
        assert!(chunks.iter().all(|chunk| chunk.len_samples()
            <= (GRANITE_SAA_HARD_MAX_CHUNK_SECS * sample_rate as f32) as usize));
    }

    #[test]
    fn granite_saa_decode_budget_uses_saa_sized_limit() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let previous = std::env::var("IZWI_GRANITE_SAA_MAX_NEW_TOKENS").ok();
        std::env::remove_var("IZWI_GRANITE_SAA_MAX_NEW_TOKENS");

        assert_eq!(
            granite_saa_max_new_tokens_for_duration(27.0),
            GRANITE_SAA_MIN_NEW_TOKENS
        );
        assert_eq!(granite_saa_max_new_tokens_for_duration(240.0), 2176);
        assert_eq!(
            granite_saa_max_new_tokens_for_duration(510.0),
            GRANITE_SAA_MAX_NEW_TOKENS
        );
        assert!(granite_saa_max_new_tokens_for_duration(510.0) < 10_000);

        if let Some(previous) = previous {
            std::env::set_var("IZWI_GRANITE_SAA_MAX_NEW_TOKENS", previous);
        } else {
            std::env::remove_var("IZWI_GRANITE_SAA_MAX_NEW_TOKENS");
        }
    }

    #[test]
    fn granite_saa_decode_budget_allows_explicit_override() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let previous = std::env::var("IZWI_GRANITE_SAA_MAX_NEW_TOKENS").ok();
        std::env::set_var("IZWI_GRANITE_SAA_MAX_NEW_TOKENS", "1234");

        assert_eq!(granite_saa_max_new_tokens_for_duration(240.0), 1234);

        if let Some(previous) = previous {
            std::env::set_var("IZWI_GRANITE_SAA_MAX_NEW_TOKENS", previous);
        } else {
            std::env::remove_var("IZWI_GRANITE_SAA_MAX_NEW_TOKENS");
        }
    }

    #[test]
    fn granite_saa_assembler_dedupes_overlap_and_preserves_turns() {
        let mut assembler = GraniteSaaTranscriptAssembler::default();
        assert!(assembler
            .push_chunk_text(
                "[Speaker 1]: hello there [Speaker 2]: hi back from me now",
                0,
            )
            .is_empty());
        assert!(assembler
            .push_chunk_text(
                "[Speaker 2]: hi back from me now and more [Speaker 1]: ok",
                1,
            )
            .is_empty());

        let text = assembler.text();
        assert_eq!(text.matches("hi back from me now").count(), 1);
        assert_eq!(
            text,
            "[Speaker 1]: hello there [Speaker 2]: hi back from me now and more [Speaker 1]: ok"
        );
    }

    #[test]
    fn granite_saa_assembler_maps_reset_label_when_overlap_proves_continuity() {
        let mut assembler = GraniteSaaTranscriptAssembler::default();
        assembler.push_chunk_text(
            "[Speaker 1]: first person [Speaker 2]: boundary overlap words now",
            0,
        );
        let warnings =
            assembler.push_chunk_text("[Speaker 1]: boundary overlap words now continuing", 1);

        assert!(warnings.iter().any(|warning| warning.contains("mapped it")));
        assert_eq!(
            assembler.text(),
            "[Speaker 1]: first person [Speaker 2]: boundary overlap words now continuing"
        );
    }

    #[test]
    fn granite_saa_prefix_text_is_bounded_on_turn_boundaries() {
        let mut assembler = GraniteSaaTranscriptAssembler::default();
        for idx in 0..32 {
            let speaker = if idx % 2 == 0 { 1 } else { 2 };
            assembler.push_chunk_text(format!("[Speaker {speaker}]: turn {idx}").as_str(), idx);
        }

        let prefix = assembler.prefix_text();
        assert!(prefix.chars().count() <= GRANITE_SAA_PREFIX_MAX_CHARS);
        assert!(prefix.contains("[Speaker 1]:"));
        assert!(!prefix.contains("turn 0"));
    }

    #[test]
    fn granite_saa_chunk_prefix_defaults_to_none() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let previous = std::env::var("IZWI_GRANITE_SAA_PREFIX_MODE").ok();
        std::env::remove_var("IZWI_GRANITE_SAA_PREFIX_MODE");
        let mut assembler = GraniteSaaTranscriptAssembler::default();
        assembler.push_chunk_text("[Speaker 1]: prior audio text", 0);

        assert_eq!(GraniteSaaPrefixMode::from_env(), GraniteSaaPrefixMode::None);
        assert_eq!(
            granite_saa_chunk_prefix_text(&assembler, GraniteSaaPrefixMode::from_env()),
            None
        );

        if let Some(previous) = previous {
            std::env::set_var("IZWI_GRANITE_SAA_PREFIX_MODE", previous);
        } else {
            std::env::remove_var("IZWI_GRANITE_SAA_PREFIX_MODE");
        }
    }

    #[test]
    fn granite_saa_chunk_prefix_full_mode_is_explicit_and_bounded() {
        let _guard = env_lock().lock().expect("env lock poisoned");
        let previous = std::env::var("IZWI_GRANITE_SAA_PREFIX_MODE").ok();
        std::env::set_var("IZWI_GRANITE_SAA_PREFIX_MODE", "full");
        let mut assembler = GraniteSaaTranscriptAssembler::default();
        for idx in 0..32 {
            assembler.push_chunk_text(format!("[Speaker 1]: turn {idx}").as_str(), idx);
        }

        let prefix = granite_saa_chunk_prefix_text(&assembler, GraniteSaaPrefixMode::from_env())
            .expect("full prefix");
        assert_eq!(
            GraniteSaaPrefixMode::from_env(),
            GraniteSaaPrefixMode::FullTranscript
        );
        assert!(prefix.chars().count() <= GRANITE_SAA_PREFIX_MAX_CHARS);
        assert!(prefix.contains("[Speaker 1]:"));

        if let Some(previous) = previous {
            std::env::set_var("IZWI_GRANITE_SAA_PREFIX_MODE", previous);
        } else {
            std::env::remove_var("IZWI_GRANITE_SAA_PREFIX_MODE");
        }
    }

    #[test]
    fn granite_saa_decode_diagnostics_warn_when_chunk_hits_token_limit() {
        let diagnostics = serde_json::json!({
            "decode": {
                "generated_tokens": 2500,
                "max_new_tokens": 2500,
                "stop_reason": "max_tokens"
            },
            "prompt": {
                "prompt_tokens": 512,
                "prefix_tokens": 0
            }
        });

        let decoded = granite_saa_decode_diagnostics(Some(&diagnostics));
        assert_eq!(
            decoded,
            GraniteSaaChunkDecodeDiagnostics {
                generated_tokens: Some(2500),
                max_new_tokens: Some(2500),
                stop_reason: Some("max_tokens".to_string()),
                prompt_tokens: Some(512),
                prompt_prefix_tokens: Some(0),
            }
        );

        let warning = granite_saa_max_token_warning(2, &decoded).expect("max token limit warning");
        assert!(warning.contains("chunk 3"));
        assert!(warning.contains("2500/2500"));
    }

    #[test]
    fn granite_saa_progress_events_report_chunks_and_percent() {
        let chunks = vec![
            AudioChunk {
                start_sample: 0,
                end_sample: 100,
            },
            AudioChunk {
                start_sample: 90,
                end_sample: 200,
            },
        ];

        let processing = granite_saa_processing_progress(&chunks, 100);
        assert_eq!(processing.phase, AsrProgressPhase::Processing);
        assert_eq!(processing.total_chunks, Some(2));
        assert_eq!(processing.percent, Some(0.0));

        let finished = granite_saa_chunk_progress(
            AsrProgressPhase::ChunkFinished,
            0,
            &chunks[0],
            &chunks,
            100,
        );
        assert_eq!(finished.current_chunk, Some(1));
        assert_eq!(finished.total_chunks, Some(2));
        assert_eq!(finished.processed_audio_secs, Some(1.0));
        assert_eq!(finished.total_audio_secs, Some(2.0));
        assert_eq!(finished.percent, Some(50.0));
    }

    #[test]
    fn speaker_attributed_asr_result_includes_long_form_warnings() {
        let result = speaker_attributed_asr_result_from_text_with_warnings(
            "[Speaker 1]: hello [Speaker 2]: hi",
            None,
            2.0,
            None,
            None,
            vec!["Granite SAA processed long audio in 2 chunks.".to_string()],
        );

        assert_eq!(result.status, SpeakerAttributedAsrStatus::Warning);
        assert_eq!(result.speaker_count, 2);
        assert_eq!(result.warnings.len(), 1);
        assert!(result.warnings[0].contains("2 chunks"));
    }

    #[test]
    fn speaker_attributed_asr_rejects_non_granite_models() {
        let err = resolve_speaker_attributed_asr_variant(Some("Whisper-Large-v3-Turbo"))
            .expect_err("SAA should be Granite-only");
        assert!(err.to_string().contains("Granite-Speech-4.1-2B-Plus"));
    }

    #[test]
    fn offline_asr_observations_include_owned_and_decoded_storage() {
        let input = [1_u8, 2, 3, 4];
        let owned = OwnedAsrAudioInput::Bytes(input.to_vec());
        let owned_bytes = owned.retained_bytes();
        assert!(owned_bytes >= input.len());

        let copied = retained_host_observation(&[input.len(), owned_bytes]).unwrap();
        assert_eq!(copied.host_bytes, (input.len() + owned_bytes) as u64);

        let decoded = decoded_audio_with_scratch_observation(input.len(), 8, 4).unwrap();
        assert_eq!(
            decoded.host_bytes,
            (input.len() + 12 * std::mem::size_of::<f32>()) as u64
        );
    }

    #[test]
    fn qwen_asr_build_route_accounts_retained_prepared_audio_once() {
        let variant = ModelVariant::Qwen3Asr06BGguf;
        let mut request =
            EngineCoreRequest::asr_bytes(vec![1, 2, 3, 4]).with_model_variant(variant);
        let source_only = retained_engine_request_input_bytes(&request).unwrap();
        request
            .install_prepared_asr_audio(variant, vec![0.0; 513], 16_000)
            .unwrap();
        request
            .install_prepared_sequence_input_tokens(48, 4096)
            .unwrap();
        let retained = retained_engine_request_input_bytes(&request).unwrap();
        assert_eq!(retained - source_only, 513 * std::mem::size_of::<f32>());
        assert_eq!(
            retained_host_observation(&[retained]).unwrap().host_bytes,
            retained as u64
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancelled_offline_asr_stage_retains_job_and_execution_until_physical_exit() {
        let coordinator = Arc::new(InferenceCoordinator::new(BackendKind::Cpu, 1, 4));
        let job = coordinator
            .admit(realtime_test_job(
                "offline-asr-cancel",
                BackendKind::Cpu,
                None,
            ))
            .await
            .unwrap();
        let release = Arc::new((Mutex::new(false), Condvar::new()));
        let task_release = release.clone();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let task_coordinator = coordinator.clone();
        let task_job = job.clone();
        drop(job);
        let runner = tokio::spawn(async move {
            task_coordinator
                .run_blocking_stage(&task_job, move || {
                    let _ = started_tx.send(());
                    let (lock, wake) = &*task_release;
                    let mut released = lock.lock().unwrap_or_else(|poison| poison.into_inner());
                    while !*released {
                        released = wake
                            .wait(released)
                            .unwrap_or_else(|poison| poison.into_inner());
                    }
                    Ok(())
                })
                .await
        });

        started_rx.await.unwrap();
        runner.abort();
        assert!(runner.await.unwrap_err().is_cancelled());
        assert_eq!(coordinator.snapshot().active_jobs, 1);
        assert_eq!(coordinator.snapshot().active_executions, 1);

        {
            let (lock, wake) = &*release;
            *lock.lock().unwrap_or_else(|poison| poison.into_inner()) = true;
            wake.notify_all();
        }
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while coordinator.snapshot().active_jobs != 0
                || coordinator.snapshot().active_executions != 0
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    // The guard intentionally serializes process-global environment access for
    // the complete asynchronous load attempt in this test.
    #[allow(clippy::await_holding_lock)]
    async fn parakeet_load_rejects_invalid_nemo_archive() {
        let _guard = env_lock().lock().expect("env lock poisoned");

        let root = std::env::temp_dir().join(format!("izwi-parakeet-runtime-{}", Uuid::new_v4()));
        let model_dir = root.join("Parakeet-TDT-0.6B-v3");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("parakeet-tdt-0.6b-v3.nemo"), b"mock-nemo").unwrap();

        let config = EngineConfig {
            models_dir: root.clone(),
            backend: BackendPreference::Cpu,
            ..Default::default()
        };

        let engine = RuntimeService::new(config).unwrap();
        let err = engine
            .load_model(ModelVariant::ParakeetTdt06BV3)
            .await
            .expect_err("invalid .nemo archive should fail to load");
        let msg = err.to_string();
        assert!(
            msg.contains(".nemo")
                || msg.contains("archive")
                || msg.contains("Failed to load")
                || msg.contains("invalid")
        );

        let _ = std::fs::remove_dir_all(root);
    }
}
