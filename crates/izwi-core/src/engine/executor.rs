//! Model executor - handles forward pass execution.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info};

#[path = "executor/audio.rs"]
mod audio;
#[path = "executor/dispatch.rs"]
mod dispatch;
#[path = "executor/handler_asr.rs"]
mod handler_asr;
#[path = "executor/handler_audio_chat.rs"]
mod handler_audio_chat;
#[path = "executor/handler_chat.rs"]
mod handler_chat;
#[path = "executor/handler_tts.rs"]
mod handler_tts;
#[path = "executor/state.rs"]
mod state;
#[path = "executor/streaming.rs"]
mod streaming;

pub(crate) use streaming::{
    deliver_committed_streams, CommittedStreamDelivery, IncrementalStreamDeliveryWorkers,
    StreamDeliveryFailure, StreamDeliveryFailureKind,
};

pub(crate) fn decode_request_audio_with_rate(
    request: &EngineCoreRequest,
) -> Result<(Vec<f32>, u32)> {
    audio::decode_request_audio_with_rate(request)
}

pub(crate) fn qwen3_asr_requires_long_form(
    samples: &[f32],
    sample_rate: u32,
    model_max_chunk_secs: Option<f32>,
) -> bool {
    audio::qwen3_asr_requires_long_form(samples, sample_rate, model_max_chunk_secs)
}

use super::config::EngineCoreConfig;
use super::execution::{
    BatchDispatch, BatchId, BatchLaneKey, CacheMode, CancellationGranularity, ConcurrencyClass,
    DispatchState, ExecutionCapabilities, ExecutionDisposition, ExecutionDomain, ExecutionFailure,
    ExecutionMode, ExecutionProfile, FailureKind, FailureOrigin, FailureScope, FinishReason,
    HealthImpact, ManagedSessionGeneration, NativeBatchMode, OutcomeProvenance, PhysicalBatch,
    PhysicalLaunchPolicy, PlanId, PrefillMode, RetryDisposition, SequencePhase, SessionKey,
    StageId, StageProgressKind, StageWorkSelector, WorkUnit, YieldReason,
};
use super::metrics::{
    begin_engine_physical_dispatch, record_engine_physical_defer, record_engine_physical_fallback,
    EnginePhysicalDeferReason, EnginePhysicalFallbackReason,
};
use super::output::StreamingOutput;
use super::request::EngineCoreRequest;
use super::resources::{BatchWorkspaceLease, ResourceAuthority, ResourceVector};
use super::scheduler::ScheduledRequest;
use super::types::{AudioOutput, TaskType};
use crate::backends::{
    can_parallelize_requests, BackendContext, BackendKind, BackendPreference, BackendRouter,
    BackendSelectionSource,
};
use crate::error::{Error, Result};
use crate::kv::{CacheDomainId, KvArenaId, KvGroupId, KvStorageDType, KvStorageFormat};
use crate::model::ModelVariant;
use crate::models::architectures::qwen3::tts::Qwen3TtsModel;
use crate::models::registry::{AsrModelLease, NativeAsrModel, NativeChatModel, QwenTtsModelLease};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::ModelRegistry;
use crate::runtime::{PhysicalExecutionAdmission, PhysicalExecutionLease};
use state::{
    ActiveAsrDecode, ActiveChatDecode, ActiveQwenTtsDecode, ActiveVoxtralRealtime,
    PendingVoxtralRealtimeQuantum, PreparedVoxtralRealtimeQuantum,
};

const QWEN38_TARGET_ATTENTION_DOMAIN: CacheDomainId = CacheDomainId::new(1);
const QWEN38_MTP_ATTENTION_DOMAIN: CacheDomainId = CacheDomainId::new(4);
// Cancellation signals are AtomicBools without a notification edge. Polling
// at 40 Hz bounds cancelled FIFO residency without turning admission into a
// hot loop.
const PHYSICAL_ADMISSION_CANCELLATION_POLL: Duration = Duration::from_millis(25);

/// Exact executor-private route for one load-sealed native model call.
///
/// The public stage contract already supplies the durable proof surface: the
/// adapter ABI, opaque stage identity, selector, batch mode, shape policy, and
/// exact batch lane. This projection prevents the executor from authorizing a
/// native call from a capability string alone. Audio routes are represented
/// now so later family adapters can add model calls without weakening the
/// shared validation boundary; no current audio adapter publishes a native
/// batch mode, so those variants remain unreachable until an exact opt-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NativeBatchRoute {
    ChatContinuousDecode {
        stage_id: StageId,
    },
    Audio {
        task: TaskType,
        stage: NativeAudioStage,
        mode: NativeBatchMode,
        stage_id: StageId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NativeAudioStage {
    PreSequencePreparation,
    SequencePrefill,
    SequenceDecode,
    RealtimePush,
    RealtimeFinish,
    RealtimePreparation,
    RealtimePromptPrefill,
    RealtimeDecodeContinuation,
    RealtimeCompletion,
    Atomic,
    Pipeline { ordinal: usize },
}

impl NativeBatchRoute {
    fn capability_matches_task(task: TaskType, capability: &str) -> bool {
        match task {
            TaskType::Chat => capability == "chat",
            TaskType::ASR => matches!(
                capability,
                "asr" | "realtime_asr" | "speaker_attributed_asr"
            ),
            TaskType::TTS => matches!(capability, "tts" | "streaming_tts"),
            TaskType::SpeechToSpeech => {
                matches!(capability, "audio_chat" | "speech_to_speech")
            }
        }
    }

    fn audio_stage(work: &WorkUnit) -> NativeAudioStage {
        match work {
            WorkUnit::PreSequencePreparation { .. } => NativeAudioStage::PreSequencePreparation,
            WorkUnit::SequenceStep {
                phase: SequencePhase::Prefill,
                ..
            } => NativeAudioStage::SequencePrefill,
            WorkUnit::SequenceStep {
                phase: SequencePhase::Decode,
                ..
            } => NativeAudioStage::SequenceDecode,
            WorkUnit::RealtimePush { .. } => NativeAudioStage::RealtimePush,
            WorkUnit::RealtimeFinish { .. } => NativeAudioStage::RealtimeFinish,
            WorkUnit::RealtimePreparation { .. } => NativeAudioStage::RealtimePreparation,
            WorkUnit::RealtimePromptPrefill { .. } => NativeAudioStage::RealtimePromptPrefill,
            WorkUnit::RealtimeDecodeContinuation { .. } => {
                NativeAudioStage::RealtimeDecodeContinuation
            }
            WorkUnit::RealtimeCompletion { .. } => NativeAudioStage::RealtimeCompletion,
            WorkUnit::AtomicJob { .. } => NativeAudioStage::Atomic,
            WorkUnit::PipelineStage { ordinal, .. } => {
                NativeAudioStage::Pipeline { ordinal: *ordinal }
            }
        }
    }

    fn resolve(execution: &PhysicalBatchExecution<'_>) -> Result<Self> {
        if execution.batch.mode == NativeBatchMode::None {
            return Err(Error::InvalidInput(
                "scalar physical work has no native tensor route".to_string(),
            ));
        }
        let first_scheduled = execution.scheduled.first().ok_or_else(|| {
            Error::InvalidInput("native physical batch has no scheduled rows".to_string())
        })?;
        let first_request = execution
            .requests
            .iter()
            .copied()
            .find(|request| request.id == first_scheduled.request_id)
            .ok_or_else(|| {
                Error::InferenceError(
                    "native physical batch has no request for its first row".to_string(),
                )
            })?;
        let task = first_request.task_type;
        let role = Self::audio_stage(&first_scheduled.work);

        for scheduled in execution.scheduled {
            let request = execution
                .requests
                .iter()
                .copied()
                .find(|request| request.id == scheduled.request_id)
                .ok_or_else(|| {
                    Error::InferenceError(format!(
                        "native physical row {} has no request snapshot",
                        scheduled.request_id
                    ))
                })?;
            if request.task_type != task || Self::audio_stage(&scheduled.work) != role {
                return Err(Error::InvalidInput(
                    "native physical batch mixed task or stage roles".to_string(),
                ));
            }
            let binding = request.execution_adapter_binding().ok_or_else(|| {
                Error::InferenceError(
                    "native physical row has no loaded adapter binding".to_string(),
                )
            })?;
            if binding.execution_group_id != execution.batch.lane.execution_group
                || binding.model_instance_id != execution.batch.lane.model_instance
                || binding.adapter_instance_id != execution.batch.lane.adapter_instance
                || binding.adapter_abi_revision != execution.batch.lane.adapter_abi
                || binding.capability_id != execution.batch.lane.capability_id
                || request.model_variant != Some(binding.model_variant)
                || request.model_instance_id() != Some(binding.model_instance_id)
                || !Self::capability_matches_task(task, &binding.capability_id)
            {
                return Err(Error::InvalidInput(
                    "native physical row crossed its loaded adapter identity".to_string(),
                ));
            }
            let stage = binding.stage_for_work(&scheduled.work)?;
            if stage.selector == StageWorkSelector::Any
                || stage.domain != ExecutionDomain::ExecutionGroup
                || stage.concurrency != ConcurrencyClass::Batchable
                || stage.batch_mode != execution.batch.mode
                || stage.id != execution.batch.lane.stage_id
                || stage.name != execution.batch.lane.kernel_mode
            {
                return Err(Error::InvalidInput(
                    "native physical row has no exact load-sealed model-call stage".to_string(),
                ));
            }
        }

        match (task, role, execution.batch.mode) {
            (TaskType::Chat, NativeAudioStage::SequenceDecode, NativeBatchMode::Continuous) => {
                Ok(Self::ChatContinuousDecode {
                    stage_id: execution.batch.lane.stage_id,
                })
            }
            (TaskType::ASR | TaskType::TTS, stage, mode) => Ok(Self::Audio {
                task,
                stage,
                mode,
                stage_id: execution.batch.lane.stage_id,
            }),
            _ => Err(Error::InvalidInput(
                "loaded stage has no compatible native executor route".to_string(),
            )),
        }
    }
}

struct Qwen38ManagedCaches {
    target: PhysicalPagedKvCache,
    mtp: Option<PhysicalPagedKvCache>,
}

fn exact_managed_group_for_domain(
    groups: &[(CacheDomainId, KvGroupId, KvArenaId)],
    reservation: &super::ManagedCacheReservation,
    domain_id: CacheDomainId,
    required: bool,
) -> Result<Option<KvGroupId>> {
    let planned = groups
        .iter()
        .filter(|(domain, _, _)| *domain == domain_id)
        .collect::<Vec<_>>();
    let reserved = reservation
        .domains
        .iter()
        .filter(|domain| domain.domain == domain_id)
        .collect::<Vec<_>>();

    if planned.is_empty() && reserved.is_empty() {
        if required {
            return Err(Error::InferenceError(format!(
                "managed Qwen3.8 reservation omitted required domain {}",
                domain_id.get()
            )));
        }
        return Ok(None);
    }
    if planned.len() != 1 || reserved.len() != 1 {
        return Err(Error::InferenceError(format!(
            "managed Qwen3.8 domain {} must resolve exactly once in both the plan and reservation",
            domain_id.get()
        )));
    }
    let (_, group_id, arena) = *planned[0];
    if reserved[0].arena != arena {
        return Err(Error::InferenceError(format!(
            "managed Qwen3.8 domain {} crossed its planned arena",
            domain_id.get()
        )));
    }
    Ok(Some(group_id))
}

fn qwen38_managed_group_ids(
    groups: &[(CacheDomainId, KvGroupId, KvArenaId)],
    reservation: &super::ManagedCacheReservation,
) -> Result<(KvGroupId, Option<KvGroupId>)> {
    let target =
        exact_managed_group_for_domain(groups, reservation, QWEN38_TARGET_ATTENTION_DOMAIN, true)?
            .ok_or_else(|| {
                Error::InferenceError("managed Qwen3.8 target domain did not resolve".into())
            })?;
    let mtp =
        exact_managed_group_for_domain(groups, reservation, QWEN38_MTP_ATTENTION_DOMAIN, false)?;
    Ok((target, mtp))
}

fn qwen38_managed_caches_for_row(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
    reservation: &super::ManagedCacheReservation,
) -> Result<Qwen38ManagedCaches> {
    let runtime = request.managed_cache_runtime().ok_or_else(|| {
        Error::InferenceError("managed Qwen3.8 row has no model runtime".to_string())
    })?;
    let groups = runtime
        .plan()
        .groups
        .iter()
        .map(|group| (group.domain, group.id, group.arena))
        .collect::<Vec<_>>();
    let (target_group, mtp_group) = qwen38_managed_group_ids(&groups, reservation)?;
    let target = physical_paged_cache_for_row(
        request,
        scheduled,
        reservation,
        QWEN38_TARGET_ATTENTION_DOMAIN,
        target_group,
    )?;
    let mtp = mtp_group
        .map(|group| {
            physical_paged_cache_for_row(
                request,
                scheduled,
                reservation,
                QWEN38_MTP_ATTENTION_DOMAIN,
                group,
            )
        })
        .transpose()?;
    Ok(Qwen38ManagedCaches { target, mtp })
}

fn qwen3_managed_cache_for_row(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
    reservation: &super::ManagedCacheReservation,
) -> Result<PhysicalPagedKvCache> {
    let runtime = request.managed_cache_runtime().ok_or_else(|| {
        Error::InferenceError("managed Qwen3 row has no model runtime".to_string())
    })?;
    let mut groups = runtime.plan().groups.iter().filter(|group| {
        reservation
            .domains
            .iter()
            .any(|domain| domain.domain == group.domain && domain.arena == group.arena)
    });
    let group = groups.next().ok_or_else(|| {
        Error::InvalidInput("native Qwen3 reservation has no resolved paged-attention group".into())
    })?;
    if groups.next().is_some() {
        return Err(Error::InvalidInput(
            "native Qwen3 reservation resolves more than one paged-attention group".into(),
        ));
    }
    physical_paged_cache_for_row(request, scheduled, reservation, group.domain, group.id)
}

struct RetainedPagedRowState {
    domain: CacheDomainId,
    group: KvGroupId,
    cache: PhysicalPagedKvCache,
}

/// Complete scheduler-owned retained state projection for one native row.
///
/// Unlike the former chat-specific dense/hybrid enum, this preserves every
/// paged domain/group identity plus the transactional tensor reservation.
/// Audio adapters can therefore select their exact authored domains without
/// teaching the shared engine about target, predictor, codec, or transducer
/// conventions.
pub(super) struct RetainedRowManagedState {
    paged: Vec<RetainedPagedRowState>,
    pub(super) tensor_state: Option<super::ManagedTensorStateReservation>,
    session_generation: ManagedSessionGeneration,
}

impl RetainedRowManagedState {
    pub(super) fn session_generation(&self) -> ManagedSessionGeneration {
        self.session_generation
    }

    pub(super) fn take_paged_domain(
        &mut self,
        domain: CacheDomainId,
        required: bool,
    ) -> Result<Option<PhysicalPagedKvCache>> {
        let matches = self
            .paged
            .iter()
            .enumerate()
            .filter_map(|(index, row)| (row.domain == domain).then_some(index))
            .collect::<Vec<_>>();
        if matches.is_empty() {
            if required {
                return Err(Error::InferenceError(format!(
                    "retained row omitted required paged domain {}",
                    domain.get()
                )));
            }
            return Ok(None);
        }
        if matches.len() != 1 {
            return Err(Error::InvalidInput(format!(
                "retained row domain {} resolves more than one physical group",
                domain.get()
            )));
        }
        Ok(Some(self.paged.swap_remove(matches[0]).cache))
    }

    pub(super) fn take_only_paged(&mut self) -> Result<PhysicalPagedKvCache> {
        if self.paged.len() != 1 {
            return Err(Error::InvalidInput(format!(
                "retained row expected one paged group, found {}",
                self.paged.len()
            )));
        }
        Ok(self.paged.pop().expect("length checked").cache)
    }

    pub(super) fn ensure_all_paged_consumed(&self) -> Result<()> {
        if self.paged.is_empty() {
            Ok(())
        } else {
            let identities = self
                .paged
                .iter()
                .map(|row| format!("{}:{}", row.domain.get(), row.group.get()))
                .collect::<Vec<_>>()
                .join(",");
            Err(Error::InvalidInput(format!(
                "retained row left unexpected paged domains/groups: {identities}"
            )))
        }
    }
}

fn retained_row_managed_state_for_row(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
    reservation: &super::ManagedCacheReservation,
) -> Result<RetainedRowManagedState> {
    if reservation.txn_id != scheduled.plan_id || reservation.session != scheduled.session_key() {
        return Err(Error::InferenceError(
            "retained row reservation crossed its scheduled row fence".to_string(),
        ));
    }
    let runtime = request.managed_cache_runtime().ok_or_else(|| {
        Error::InferenceError("retained native row has no physical model runtime".to_string())
    })?;
    if request.model_instance_id() != Some(runtime.plan().model_instance) {
        return Err(Error::InferenceError(
            "retained native row crossed its loaded model instance".to_string(),
        ));
    }

    let mut paged = Vec::new();
    let mut seen = HashSet::new();
    for domain in &reservation.domains {
        for table in &domain.provisional_groups {
            let mut groups = runtime.plan().groups.iter().filter(|group| {
                group.domain == domain.domain
                    && group.id == table.group
                    && group.arena == domain.arena
            });
            let group = groups.next().ok_or_else(|| {
                Error::InferenceError(
                    "retained row reservation references an unresolved paged group".to_string(),
                )
            })?;
            if groups.next().is_some() || !seen.insert((group.domain, group.id, group.arena)) {
                return Err(Error::InvalidInput(
                    "retained row reservation repeats a paged domain/group".to_string(),
                ));
            }
            paged.push(RetainedPagedRowState {
                domain: group.domain,
                group: group.id,
                cache: physical_paged_cache_for_row(
                    request,
                    scheduled,
                    reservation,
                    group.domain,
                    group.id,
                )?,
            });
        }
    }
    if paged.is_empty() && reservation.clocked_state.is_none() {
        return Err(Error::InvalidInput(
            "retained native row has neither paged nor tensor state".to_string(),
        ));
    }
    Ok(RetainedRowManagedState {
        paged,
        tensor_state: reservation.clocked_state.clone(),
        session_generation: reservation.session_generation,
    })
}

/// Resolve one exact scheduler-owned paged-attention view without assuming
/// that the row reservation contains only one state domain or physical group.
fn physical_paged_cache_for_row(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
    reservation: &super::ManagedCacheReservation,
    domain_id: CacheDomainId,
    group_id: KvGroupId,
) -> Result<PhysicalPagedKvCache> {
    if reservation.txn_id != scheduled.plan_id || reservation.session != scheduled.session_key() {
        return Err(Error::InferenceError(
            "managed-cache reservation crossed its scheduled row fence".to_string(),
        ));
    }
    let runtime = request.managed_cache_runtime().ok_or_else(|| {
        Error::InferenceError("managed-cache row has no physical model runtime".to_string())
    })?;
    let plan = runtime.plan();
    if request.model_instance_id() != Some(plan.model_instance) {
        return Err(Error::InferenceError(
            "managed-cache runtime does not match the row's loaded model instance".into(),
        ));
    }

    let mut groups = plan
        .groups
        .iter()
        .filter(|group| group.domain == domain_id && group.id == group_id);
    let group = groups.next().ok_or_else(|| {
        Error::InferenceError("managed-cache row references an unresolved domain/group pair".into())
    })?;
    if groups.next().is_some() {
        return Err(Error::InferenceError(
            "managed-cache plan repeats a domain/group pair".into(),
        ));
    }
    let layers = &group.layers;
    if layers.is_empty() {
        return Err(Error::InferenceError(
            "managed-cache paged-attention group has no layer bindings".into(),
        ));
    }
    if group.arena.model_instance != plan.model_instance
        || group.arena.backend != plan.backend
        || group.arena.device_ordinal != plan.device_ordinal
    {
        return Err(Error::InferenceError(
            "managed-cache group crossed its resolved runtime identity".into(),
        ));
    }

    let mut domains = reservation
        .domains
        .iter()
        .filter(|domain| domain.domain == domain_id && domain.arena == group.arena);
    let domain = domains.next().ok_or_else(|| {
        Error::InferenceError(
            "managed-cache reservation omitted the selected domain/group arena".into(),
        )
    })?;
    if domains.next().is_some() {
        return Err(Error::InvalidInput(
            "managed-cache reservation repeats the selected domain/group arena".into(),
        ));
    }
    if domain.execution_start_tokens < domain.expected_committed_tokens
        || domain.target_committed_tokens < domain.execution_start_tokens
        || domain.target_window_start > domain.execution_start_tokens
    {
        return Err(Error::InvalidInput(
            "managed-cache domain has an invalid execution/window range".into(),
        ));
    }
    if group.page_tokens == 0
        || domain.first_page_offset >= group.page_tokens
        || domain.first_page_offset != domain.target_window_start % group.page_tokens
    {
        return Err(Error::InvalidInput(
            "managed-cache first-page offset does not match its logical window".to_string(),
        ));
    }

    let mut tables = domain
        .provisional_groups
        .iter()
        .filter(|table| table.group == group_id);
    let table = tables.next().ok_or_else(|| {
        Error::InferenceError("managed-cache reservation omitted its selected block table".into())
    })?;
    if tables.next().is_some() {
        return Err(Error::InvalidInput(
            "managed-cache reservation repeats its selected block table".into(),
        ));
    }

    let arena = runtime.arena(group.arena).ok_or_else(|| {
        Error::InferenceError("managed-cache physical arena is no longer live".to_string())
    })?;
    let config = arena.config();
    if arena.id() != group.arena
        || arena.backend_kind() != plan.backend
        || config.id != group.arena
        || config.group != group_id
        || config.page_tokens != group.page_tokens
        || config.capacity_pages != group.capacity_pages
    {
        return Err(Error::InferenceError(
            "managed-cache arena geometry does not match its resolved group".into(),
        ));
    }
    let storage_matches = matches!(
        (group.storage, config.dtype),
        (
            KvStorageFormat::Dense {
                dtype: KvStorageDType::F32
            },
            candle_core::DType::F32
        ) | (
            KvStorageFormat::Dense {
                dtype: KvStorageDType::F16
            },
            candle_core::DType::F16
        ) | (
            KvStorageFormat::Dense {
                dtype: KvStorageDType::Bf16
            },
            candle_core::DType::BF16
        )
    );
    if !storage_matches
        || config.layers.len() != layers.len()
        || config
            .layers
            .iter()
            .zip(layers)
            .any(|(configured, resolved)| {
                configured.binding != *resolved
                    || configured.num_kv_heads == 0
                    || configured.key_head_dim == 0
                    || configured.value_head_dim == 0
            })
    {
        return Err(Error::InferenceError(
            "managed-cache arena layer or storage geometry is stale".into(),
        ));
    }
    let element_bytes = match config.dtype {
        candle_core::DType::F32 => 4_u64,
        candle_core::DType::F16 | candle_core::DType::BF16 => 2_u64,
        _ => {
            return Err(Error::InferenceError(
                "managed-cache arena uses unsupported paged storage".into(),
            ));
        }
    };
    let bytes_per_page = config.layers.iter().try_fold(0_u64, |total, layer| {
        let per_token = u64::from(layer.num_kv_heads)
            .checked_mul(u64::from(layer.key_head_dim) + u64::from(layer.value_head_dim))
            .ok_or_else(|| Error::InferenceError("managed-cache layer geometry overflow".into()))?;
        let bytes = u64::from(config.page_tokens)
            .checked_mul(per_token)
            .and_then(|elements| elements.checked_mul(element_bytes))
            .ok_or_else(|| Error::InferenceError("managed-cache page geometry overflow".into()))?;
        total
            .checked_add(bytes)
            .ok_or_else(|| Error::InferenceError("managed-cache page geometry overflow".into()))
    })?;
    if bytes_per_page != group.bytes_per_page {
        return Err(Error::InferenceError(
            "managed-cache arena byte geometry does not match its resolved group".into(),
        ));
    }

    let visible_target = domain
        .target_committed_tokens
        .checked_sub(domain.target_window_start)
        .ok_or_else(|| Error::InvalidInput("managed-cache window exceeds its target".into()))?;
    let physical_target = visible_target
        .checked_add(domain.first_page_offset)
        .ok_or_else(|| Error::InvalidInput("managed-cache window geometry overflow".into()))?;
    let required_pages = usize::try_from(physical_target.div_ceil(group.page_tokens))
        .map_err(|_| Error::InvalidInput("managed-cache page count exceeds usize".into()))?;
    if required_pages == 0 || table.blocks.len() != required_pages {
        return Err(Error::InvalidInput(format!(
            "managed-cache block table has {} pages, expected {required_pages}",
            table.blocks.len()
        )));
    }
    let mut unique_blocks = HashSet::with_capacity(table.blocks.len());
    if table.blocks.iter().any(|block| {
        block.arena != group.arena
            || block.group != group_id
            || block.index >= group.capacity_pages
            || block.slot_generation == 0
            || !unique_blocks.insert(*block)
    }) {
        return Err(Error::InvalidInput(
            "managed-cache block table contains a foreign, stale, duplicate, or out-of-range page"
                .into(),
        ));
    }

    PhysicalPagedKvCache::new_windowed(
        arena.clone(),
        layers.clone(),
        table.blocks.clone(),
        usize::try_from(domain.target_window_start)
            .map_err(|_| Error::InvalidInput("managed-cache window exceeds usize".into()))?,
        usize::try_from(domain.execution_start_tokens)
            .map_err(|_| Error::InvalidInput("managed-cache context exceeds usize".into()))?,
    )
}

fn invocation_paged_stage_and_domains<'a>(
    request: &'a EngineCoreRequest,
    scheduled: &ScheduledRequest,
) -> Result<(
    &'a super::StageDescriptor,
    Vec<crate::kv::v2::StateDomainId>,
)> {
    let binding = request.execution_adapter_binding().ok_or_else(|| {
        Error::InferenceError("physical invocation row has no loaded adapter binding".to_string())
    })?;
    let stage = binding.stage_for_work(&scheduled.work)?;
    let graph = crate::kv::v2::stage_graph_fingerprint(&binding.stages)?;
    let descriptor = request.v2_state_descriptor().ok_or_else(|| {
        Error::InferenceError("physical invocation row has no state descriptor".to_string())
    })?;
    let crate::kv::v2::InvocationWorkspaceSet::Bounded { profiles } = &descriptor.invocation else {
        return Err(Error::InferenceError(
            "physical invocation row has no bounded workspace profile".to_string(),
        ));
    };
    let profile = profiles
        .iter()
        .find(|profile| profile.stage_graph_fingerprint == graph)
        .ok_or_else(|| {
            Error::InferenceError(
                "physical invocation row has no workspace for its adapter graph".to_string(),
            )
        })?;
    let workspace = profile
        .stages
        .iter()
        .find(|workspace| workspace.stage == stage.id)
        .ok_or_else(|| {
            Error::InferenceError(
                "physical invocation row has no workspace for its scheduled stage".to_string(),
            )
        })?;
    let paged = workspace
        .domains
        .iter()
        .filter_map(|domain| match domain {
            crate::kv::v2::InvocationWorkspaceDomain::State {
                state: crate::kv::v2::StateDomainSpec::PagedAttention(state),
                capacity,
                ..
            } if capacity.paged_max_tokens().is_some() => Some(state.header.id),
            _ => None,
        })
        .collect::<Vec<_>>();
    if paged.is_empty() {
        return Err(Error::InferenceError(
            "physical invocation stage has no paged workspace domain".to_string(),
        ));
    }
    Ok((stage, paged))
}

/// Lease the one paged invocation domain authored for this exact scheduled
/// stage. Models receive only the physical cache view and cannot select a pool
/// by convention or by model-family-specific IDs.
fn invocation_paged_lease_for_row(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
) -> Result<super::InvocationPagedKvLease> {
    let (stage, domains) = invocation_paged_stage_and_domains(request, scheduled)?;
    if domains.len() != 1 {
        return Err(Error::InferenceError(
            "physical invocation stage has multiple paged workspace domains".to_string(),
        ));
    }
    request
        .v2_state_runtime()
        .ok_or_else(|| {
            Error::InferenceError("physical invocation row has no sealed runtime".to_string())
        })?
        .lease_invocation_paged(stage.id, domains[0])
}

fn validate_atomic_scalar_invocation_stage(
    stage: &super::StageDescriptor,
    work: &WorkUnit,
) -> Result<()> {
    if !matches!(work, WorkUnit::AtomicJob { .. }) {
        return Err(Error::InvalidInput(
            "scalar invocation workspace requires an atomic scheduled row".to_string(),
        ));
    }
    if stage.progress != StageProgressKind::Atomic || stage.batch_mode != NativeBatchMode::None {
        return Err(Error::InvalidInput(
            "atomic invocation workspace requires a scalar atomic execution stage".to_string(),
        ));
    }
    Ok(())
}

/// Acquire one atomic scalar row's complete authored typed workspace in
/// canonical domain order. This is the model-neutral path for mixed paged,
/// recurrent, append, ring, and static state.
pub(super) fn invocation_workspace_leases_for_atomic_scalar_row(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
) -> Result<crate::kv::v2::InvocationWorkspaceLeaseSetV2> {
    let binding = request.execution_adapter_binding().ok_or_else(|| {
        Error::InferenceError("physical invocation row has no loaded adapter binding".to_string())
    })?;
    let stage = binding.stage_for_work(&scheduled.work)?;
    validate_atomic_scalar_invocation_stage(stage, &scheduled.work)?;
    request
        .v2_state_runtime()
        .ok_or_else(|| {
            Error::InferenceError("physical invocation row has no sealed runtime".to_string())
        })?
        .lease_complete_invocation_workspace_set(stage.id)
}

/// Acquire one atomic scalar row's complete authored paged-domain set in
/// canonical identity order. Callers cannot omit a required domain. The
/// returned set releases every already-acquired lease if a later domain fails,
/// and explicit completion returns only authenticated writes.
pub(super) fn invocation_paged_leases_for_atomic_scalar_row(
    request: &EngineCoreRequest,
    scheduled: &ScheduledRequest,
) -> Result<crate::kv::v2::InvocationPagedLeaseSetV2> {
    let (stage, _) = invocation_paged_stage_and_domains(request, scheduled)?;
    validate_atomic_scalar_invocation_stage(stage, &scheduled.work)?;
    request
        .v2_state_runtime()
        .ok_or_else(|| {
            Error::InferenceError("physical invocation row has no sealed runtime".to_string())
        })?
        .lease_complete_invocation_paged_set(stage.id)
}

fn panic_payload_to_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(msg) = payload.downcast_ref::<&str>() {
        return (*msg).to_string();
    }
    if let Some(msg) = payload.downcast_ref::<String>() {
        return msg.clone();
    }
    "unknown panic payload".to_string()
}

/// Hard upper bound for scoped CPU row workers in one physical dispatch.
const MAX_CPU_REQUEST_PARALLELISM: usize = 4;

/// Configuration for the model executor.
#[derive(Clone)]
pub struct WorkerConfig {
    /// Path to models directory
    pub models_dir: PathBuf,
    /// Backend to use (cpu, metal, cuda)
    pub backend: BackendKind,
    /// Resolved backend/device context for this worker.
    pub backend_context: BackendContext,
    /// Data type (float32, float16, bfloat16)
    pub dtype: String,
    /// KV cache storage dtype hint (e.g. float16, int8).
    pub kv_cache_dtype: String,
    /// Number of threads
    pub num_threads: usize,
    /// Maximum number of requests to execute in parallel.
    pub request_parallelism: usize,
    /// Decode-time KV cache page size.
    pub kv_page_size: usize,
    /// Optional shared model registry for loaded runtime models.
    pub model_registry: Option<Arc<ModelRegistry>>,
    /// Shared physical resource authority used for bounded executor workspaces.
    pub resource_authority: Option<Arc<ResourceAuthority>>,
    /// Shared physical-launch admission. Runtime services replace the local
    /// fail-closed gate with their coordinator-owned handle.
    pub(crate) physical_execution_admission: Option<PhysicalExecutionAdmission>,
    /// Maximum width of a model-native tensor batch on this backend.
    pub max_tensor_batch_size: usize,
    /// Exact model variants enabled for static tensor execution on this worker.
    pub static_tensor_batch_variants: Arc<HashSet<ModelVariant>>,
    /// Opt-in scheduler-level chunked prefill for resumable-prefill models.
    pub enable_chunked_prefill: bool,
}

impl std::fmt::Debug for WorkerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerConfig")
            .field("models_dir", &self.models_dir)
            .field("backend", &self.backend)
            .field("backend_context", &self.backend_context)
            .field("dtype", &self.dtype)
            .field("kv_cache_dtype", &self.kv_cache_dtype)
            .field("num_threads", &self.num_threads)
            .field("request_parallelism", &self.request_parallelism)
            .field("kv_page_size", &self.kv_page_size)
            .field(
                "model_registry",
                &self.model_registry.as_ref().map(|_| "<shared>"),
            )
            .field(
                "resource_authority",
                &self.resource_authority.as_ref().map(|_| "<shared>"),
            )
            .field(
                "physical_execution_admission",
                &self
                    .physical_execution_admission
                    .as_ref()
                    .map(PhysicalExecutionAdmission::capacity),
            )
            .field("max_tensor_batch_size", &self.max_tensor_batch_size)
            .field(
                "static_tensor_batch_variants",
                &self.static_tensor_batch_variants.len(),
            )
            .finish()
    }
}

impl Default for WorkerConfig {
    fn default() -> Self {
        let backend_context = BackendRouter::resolve_context(
            BackendPreference::Auto,
            BackendSelectionSource::Default,
        );
        let backend_kind = backend_context.backend_kind;
        let num_threads = 4;
        Self {
            models_dir: dirs::data_local_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("izwi")
                .join("models"),
            backend: backend_kind,
            backend_context,
            dtype: "float32".to_string(),
            kv_cache_dtype: "float16".to_string(),
            num_threads,
            request_parallelism: Self::request_parallelism_for(backend_kind, num_threads),
            kv_page_size: 64,
            model_registry: None,
            resource_authority: None,
            physical_execution_admission: Some(PhysicalExecutionAdmission::standalone(1)),
            max_tensor_batch_size: 1,
            static_tensor_batch_variants: Arc::new(HashSet::new()),
            enable_chunked_prefill: false,
        }
    }
}

impl From<&EngineCoreConfig> for WorkerConfig {
    fn from(config: &EngineCoreConfig) -> Self {
        let backend_context =
            BackendRouter::resolve_context_for_kind(config.backend, BackendSelectionSource::Config);
        let backend_kind = backend_context.backend_kind;
        let num_threads = config.num_threads.max(1);
        let max_tensor_batch_size = config
            .max_tensor_batch_size
            .resolve(backend_kind)
            .min(Self::tensor_batch_cap(backend_kind))
            .max(1);
        let request_parallelism = Self::request_parallelism_for(backend_kind, num_threads);
        let physical_execution_capacity = Self::physical_execution_capacity(
            config.physical_execution_mode,
            backend_kind,
            request_parallelism,
            config
                .resolved_physical_execution_capacity()
                .physical_launch_limit
                .get(),
        );
        Self {
            models_dir: config.models_dir.clone(),
            backend: backend_kind,
            backend_context,
            dtype: "float32".to_string(),
            kv_cache_dtype: config.kv_cache_dtype.clone(),
            num_threads,
            request_parallelism,
            kv_page_size: config.block_size.max(1),
            model_registry: None,
            resource_authority: None,
            physical_execution_admission: Some(PhysicalExecutionAdmission::standalone(
                physical_execution_capacity,
            )),
            max_tensor_batch_size,
            static_tensor_batch_variants: Arc::new(HashSet::new()),
            enable_chunked_prefill: config.enable_chunked_prefill,
        }
    }
}

impl WorkerConfig {
    fn physical_execution_capacity(
        mode: crate::config::PhysicalExecutionMode,
        backend: BackendKind,
        request_parallelism: usize,
        configured_launch_limit: usize,
    ) -> usize {
        if mode == crate::config::PhysicalExecutionMode::Concurrent && backend == BackendKind::Cpu {
            request_parallelism.max(configured_launch_limit).max(1)
        } else {
            1
        }
    }

    fn tensor_batch_cap(backend: BackendKind) -> usize {
        match backend {
            BackendKind::Cpu | BackendKind::Metal => 2,
            // Runtime CUDA defaults remain VRAM-tiered and resource-admitted;
            // this is only the hard kernel/metadata width ceiling.
            BackendKind::Cuda => 32,
        }
    }

    fn request_parallelism_override() -> Option<usize> {
        std::env::var("IZWI_REQUEST_PARALLELISM")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
    }

    fn available_cpu_capacity() -> usize {
        std::thread::available_parallelism()
            .map(|capacity| capacity.get())
            .unwrap_or(1)
            .max(1)
    }

    fn resolve_request_parallelism(
        backend: BackendKind,
        configured_intra_op_threads: usize,
        available_cpu_capacity: usize,
        override_value: Option<usize>,
    ) -> usize {
        // Candle's Metal path is intentionally serialized in dispatch. Do not
        // let an environment override inflate coordinator capacity beyond what
        // the executor can actually run concurrently.
        if backend == BackendKind::Metal {
            return 1;
        }

        if backend == BackendKind::Cpu {
            let cpu_capacity = available_cpu_capacity.max(1);
            let hard_cap = cpu_capacity.min(MAX_CPU_REQUEST_PARALLELISM);
            let automatic = (cpu_capacity / configured_intra_op_threads.max(1)).max(1);
            // Preserve the operator override as the selected value, then keep
            // both automatic and explicit widths within the process-visible
            // CPU allocation and the executor's conservative worker ceiling.
            return override_value.unwrap_or(automatic).clamp(1, hard_cap);
        }

        // CUDA does not gain automatic scalar concurrency here. Preserve the
        // existing explicit override contract without coupling it to host CPU
        // capacity; accelerator launch policy is sealed independently.
        override_value.unwrap_or(1).max(1)
    }

    fn request_parallelism_for(backend: BackendKind, configured_intra_op_threads: usize) -> usize {
        Self::resolve_request_parallelism(
            backend,
            configured_intra_op_threads,
            Self::available_cpu_capacity(),
            Self::request_parallelism_override(),
        )
    }
}

/// Output from the executor after a forward pass.
pub const REQUEST_DEADLINE_EXCEEDED: &str = "request deadline exceeded";

#[derive(Debug, Clone)]
pub struct ExecutorOutput {
    /// Request ID
    pub request_id: String,
    /// Generated audio samples
    pub audio: Option<AudioOutput>,
    /// Generated text (for ASR/chat)
    pub text: Option<String>,
    /// Optional input transcription for speech-to-speech requests.
    pub input_transcription: Option<String>,
    /// Number of tokens processed
    pub tokens_processed: usize,
    /// Number of tokens generated
    pub tokens_generated: usize,
    /// Whether generation is complete
    pub finished: bool,
    /// Optional per-request phase timing override from model-specific execution paths.
    pub phase_timing_override: Option<ExecutorPhaseTiming>,
    /// Optional ASR diagnostics payload surfaced by model-specific paths.
    pub asr_diagnostics: Option<serde_json::Value>,
    /// Error if any
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ExecutorPhaseTiming {
    /// Audio/media decode duration in milliseconds.
    pub media_decode_ms: Option<f64>,
    /// Input normalization duration in milliseconds.
    pub normalization_ms: Option<f64>,
    /// Prefill phase duration in milliseconds.
    pub prefill_ms: Option<f64>,
    /// Decode phase duration in milliseconds.
    pub decode_ms: Option<f64>,
    /// Sampling duration in milliseconds.
    pub sampling_ms: Option<f64>,
    /// Codec encode/decode duration in milliseconds.
    pub codec_ms: Option<f64>,
    /// Postprocess duration in milliseconds.
    pub postprocess_ms: Option<f64>,
    /// Time to first user-visible output in milliseconds since model execution start.
    pub first_output_ms_since_start: Option<f64>,
    /// Number of prefill steps attributed to this request.
    pub prefill_steps: Option<u32>,
    /// Number of decode steps attributed to this request.
    pub decode_steps: Option<u32>,
}

impl ExecutorPhaseTiming {
    pub fn with_media_decode_ms(media_decode_ms: f64) -> Self {
        Self {
            media_decode_ms: Some(media_decode_ms.max(0.0)),
            ..Self::default()
        }
    }
}

impl ExecutorOutput {
    pub fn error(request_id: String, error: impl Into<String>) -> Self {
        Self {
            request_id,
            audio: None,
            text: None,
            input_transcription: None,
            tokens_processed: 0,
            tokens_generated: 0,
            finished: true,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: Some(error.into()),
        }
    }

    pub fn cancelled(request_id: String) -> Self {
        Self::terminal(request_id)
    }

    /// Construct a terminal payload whose precise outcome is carried by the
    /// authoritative execution disposition.
    pub fn terminal(request_id: String) -> Self {
        Self {
            request_id,
            audio: None,
            text: None,
            input_transcription: None,
            tokens_processed: 0,
            tokens_generated: 0,
            finished: true,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        }
    }
}

/// Backend-neutral result produced by one model-owned session safe point.
/// Native handlers must choose sequence, yield, or atomic semantics explicitly.
#[derive(Debug, Clone)]
pub struct ModelSessionResult {
    pub output: ExecutorOutput,
    pub disposition: ExecutionDisposition,
    pub safe_point: bool,
    pub provenance: OutcomeProvenance,
    pub staged_stream_outputs: Vec<StreamingOutput>,
    /// Backend-sealed physical write batches produced by this safe point.
    /// The executor reconciles these against the exact row reservation before
    /// it can construct a managed-cache receipt.
    pub(crate) managed_cache_completions: Vec<Arc<crate::backends::kv::KvWriteBatchCompletion>>,
    /// Exact decoder-KV cursor advance, distinct from source samples consumed
    /// and user-visible output events for realtime audio work.
    pub(crate) managed_cache_append: Option<usize>,
    /// The result is not authoritative unless Core resolves the exact
    /// executor-owned post-model transaction registered for its plan/session.
    pub(crate) pending_quantum_required: bool,
    pub(crate) clocked_state_completion: Option<crate::backends::state::TensorStateBatchCompletion>,
}

impl ModelSessionResult {
    fn executor_failure(message: String) -> ExecutionDisposition {
        ExecutionDisposition::Failed(ExecutionFailure {
            kind: FailureKind::Executor,
            scope: FailureScope::Row,
            retry: RetryDisposition::Never,
            health: HealthImpact::None,
            message,
        })
    }

    pub fn sequence(output: ExecutorOutput) -> Self {
        let disposition = if let Some(message) = output.error.as_ref() {
            Self::executor_failure(message.clone())
        } else if output.finished {
            ExecutionDisposition::Finished(FinishReason::Completed)
        } else {
            ExecutionDisposition::Yielded(YieldReason::QuantumExhausted)
        };
        let provenance = if matches!(disposition, ExecutionDisposition::Failed(_)) {
            OutcomeProvenance::failure(FailureOrigin::Model, DispatchState::Started)
        } else {
            OutcomeProvenance::produced_output()
        };
        Self {
            output,
            disposition,
            safe_point: true,
            provenance,
            staged_stream_outputs: Vec::new(),
            managed_cache_completions: Vec::new(),
            managed_cache_append: None,
            pending_quantum_required: false,
            clocked_state_completion: None,
        }
    }

    pub fn yielded(output: ExecutorOutput, reason: YieldReason) -> Self {
        Self {
            output,
            disposition: ExecutionDisposition::Yielded(reason),
            safe_point: true,
            provenance: OutcomeProvenance::produced_output(),
            staged_stream_outputs: Vec::new(),
            managed_cache_completions: Vec::new(),
            managed_cache_append: None,
            pending_quantum_required: false,
            clocked_state_completion: None,
        }
    }

    /// Request a model-neutral restart of this exact retained sequence. The
    /// model commits its semantic restart checkpoint (including pending retry
    /// intent) but publishes no physical write completion; Core aborts the
    /// scheduler reservation and advances the managed subgeneration.
    pub fn restart_sequence(request_id: String, reason: super::SequenceRestartReason) -> Self {
        Self {
            output: ExecutorOutput {
                request_id,
                audio: None,
                text: None,
                input_transcription: None,
                tokens_processed: 0,
                tokens_generated: 0,
                finished: false,
                phase_timing_override: None,
                asr_diagnostics: None,
                error: None,
            },
            disposition: ExecutionDisposition::RestartSequence(reason),
            safe_point: true,
            provenance: OutcomeProvenance::started(),
            staged_stream_outputs: Vec::new(),
            managed_cache_completions: Vec::new(),
            managed_cache_append: None,
            pending_quantum_required: false,
            clocked_state_completion: None,
        }
    }

    pub fn cancelled(mut output: ExecutorOutput) -> Self {
        output.finished = true;
        Self {
            output,
            disposition: ExecutionDisposition::Finished(FinishReason::Cancelled),
            safe_point: true,
            provenance: OutcomeProvenance::started(),
            staged_stream_outputs: Vec::new(),
            managed_cache_completions: Vec::new(),
            managed_cache_append: None,
            pending_quantum_required: false,
            clocked_state_completion: None,
        }
    }

    pub fn cancelled_before_dispatch(mut output: ExecutorOutput) -> Self {
        output.finished = true;
        Self {
            output,
            disposition: ExecutionDisposition::Finished(FinishReason::Cancelled),
            safe_point: true,
            provenance: OutcomeProvenance::not_started(),
            staged_stream_outputs: Vec::new(),
            managed_cache_completions: Vec::new(),
            managed_cache_append: None,
            pending_quantum_required: false,
            clocked_state_completion: None,
        }
    }

    pub fn atomic(mut output: ExecutorOutput) -> Self {
        let disposition = if let Some(message) = output.error.as_ref() {
            Self::executor_failure(message.clone())
        } else if output.finished {
            ExecutionDisposition::Finished(FinishReason::Completed)
        } else {
            let message = "atomic model session returned before reaching a terminal state";
            output.error = Some(message.to_string());
            output.finished = true;
            ExecutionDisposition::Failed(ExecutionFailure::invalid_output(message))
        };
        let provenance = if matches!(disposition, ExecutionDisposition::Failed(_)) {
            OutcomeProvenance::failure(FailureOrigin::Model, DispatchState::Started)
        } else {
            OutcomeProvenance::produced_output()
        };
        Self {
            output,
            disposition,
            safe_point: true,
            provenance,
            staged_stream_outputs: Vec::new(),
            managed_cache_completions: Vec::new(),
            managed_cache_append: None,
            pending_quantum_required: false,
            clocked_state_completion: None,
        }
    }

    fn with_staged_stream_outputs(mut self, outputs: Vec<StreamingOutput>) -> Self {
        self.staged_stream_outputs = outputs;
        self
    }

    pub(crate) fn with_managed_cache_completions(
        mut self,
        completions: Vec<Arc<crate::backends::kv::KvWriteBatchCompletion>>,
    ) -> Self {
        self.managed_cache_completions = completions;
        self
    }

    pub(crate) fn with_managed_cache_append(mut self, appended: usize) -> Self {
        self.managed_cache_append = Some(appended);
        self
    }

    pub(crate) fn requiring_pending_quantum(mut self) -> Self {
        self.pending_quantum_required = true;
        self
    }

    pub(crate) fn with_clocked_state_completion(
        mut self,
        completion: crate::backends::state::TensorStateBatchCompletion,
    ) -> Self {
        self.clocked_state_completion = Some(completion);
        self
    }
}

/// Executor payload fenced to the exact scheduler transaction that produced it.
#[derive(Debug, Clone)]
pub struct ExecutorStepResult {
    pub plan_id: PlanId,
    pub session: SessionKey,
    pub disposition: ExecutionDisposition,
    pub safe_point: bool,
    pub dispatch: BatchDispatch,
    pub provenance: OutcomeProvenance,
    /// Executor-owned resources retained after this safe point. Persistent
    /// inference state is reported by its lifecycle-owned physical manager.
    pub observed_resources: ResourceVector,
    pub output: ExecutorOutput,
    pub staged_stream_outputs: Vec<StreamingOutput>,
    /// Optional physical KV write acknowledgement for this exact row.
    pub managed_cache: Option<super::ManagedCacheReceipt>,
    pub(crate) managed_cache_completions: Vec<Arc<crate::backends::kv::KvWriteBatchCompletion>>,
    pub(crate) managed_cache_append: Option<usize>,
    pub(crate) pending_quantum_required: bool,
    pub(crate) clocked_state_completion: Option<crate::backends::state::TensorStateBatchCompletion>,
}

impl ExecutorStepResult {
    pub fn new(scheduled: &ScheduledRequest, output: ExecutorOutput) -> Self {
        let session_result = if output.finished || output.error.is_some() {
            ModelSessionResult::atomic(output)
        } else {
            // Compatibility for third-party/test executors. Native production
            // handlers use `from_session` with an explicit session result.
            ModelSessionResult::sequence(output)
        };
        Self::from_session(scheduled, session_result)
    }

    pub fn from_session(scheduled: &ScheduledRequest, session_result: ModelSessionResult) -> Self {
        Self {
            plan_id: scheduled.plan_id,
            session: scheduled.session_key(),
            disposition: session_result.disposition,
            safe_point: session_result.safe_point,
            dispatch: BatchDispatch::serial(),
            provenance: session_result.provenance,
            observed_resources: ResourceVector::zero(),
            output: session_result.output,
            staged_stream_outputs: session_result.staged_stream_outputs,
            managed_cache: None,
            managed_cache_completions: session_result.managed_cache_completions,
            managed_cache_append: session_result.managed_cache_append,
            pending_quantum_required: session_result.pending_quantum_required,
            clocked_state_completion: session_result.clocked_state_completion,
        }
    }

    pub fn with_dispatch(mut self, dispatch: BatchDispatch) -> Self {
        self.dispatch = dispatch;
        self
    }

    pub fn with_provenance(mut self, provenance: OutcomeProvenance) -> Self {
        self.provenance = provenance;
        self
    }

    pub fn with_observed_resources(mut self, resources: ResourceVector) -> Self {
        self.observed_resources = resources;
        self
    }

    pub fn with_managed_cache_receipt(mut self, receipt: super::ManagedCacheReceipt) -> Self {
        self.managed_cache = Some(receipt);
        self
    }
}

/// Model executor trait - abstracts the model inference backend.
pub struct PhysicalBatchExecution<'a> {
    pub batch: &'a PhysicalBatch,
    pub requests: &'a [&'a EngineCoreRequest],
    pub scheduled: &'a [ScheduledRequest],
}

#[derive(Debug)]
pub struct PhysicalDispatchError {
    pub error: Error,
    pub dispatch: BatchDispatch,
    pub provenance: OutcomeProvenance,
}

impl PhysicalDispatchError {
    pub(crate) fn not_started(error: Error, width: usize, origin: FailureOrigin) -> Self {
        Self {
            error,
            dispatch: BatchDispatch::not_dispatched(width),
            provenance: OutcomeProvenance::failure(origin, DispatchState::NotStarted),
        }
    }

    pub(crate) fn started(error: Error, dispatch: BatchDispatch, origin: FailureOrigin) -> Self {
        Self {
            error,
            dispatch,
            provenance: OutcomeProvenance::failure(origin, DispatchState::Started),
        }
    }
}

pub type PhysicalDispatchResult =
    std::result::Result<Vec<ExecutorStepResult>, PhysicalDispatchError>;

impl PhysicalBatchExecution<'_> {
    pub fn expected_dispatch(&self) -> BatchDispatch {
        self.batch.expected_dispatch()
    }

    pub fn validate(&self) -> Result<()> {
        self.batch.validate()?;
        if self.batch.rows.len() != self.scheduled.len()
            || self.scheduled.len() != self.requests.len()
        {
            return Err(Error::InferenceError(
                "physical executor inputs do not match the batch width".to_string(),
            ));
        }

        let expected = self
            .batch
            .rows
            .iter()
            .map(|row| ((row.plan_id, row.session.clone()), &row.work))
            .collect::<HashMap<_, _>>();
        let mut scheduled_ids = HashSet::with_capacity(self.scheduled.len());
        for scheduled in self.scheduled {
            let key = (scheduled.plan_id, scheduled.session_key());
            let work = expected.get(&key).ok_or_else(|| {
                Error::InferenceError(
                    "scheduled work is not present in the physical batch envelope".to_string(),
                )
            })?;
            if **work != scheduled.work {
                return Err(Error::InferenceError(
                    "scheduled work differs from the physical batch quantum".to_string(),
                ));
            }
            if !scheduled_ids.insert(scheduled.request_id.as_str()) {
                return Err(Error::InferenceError(
                    "physical executor inputs contain a duplicate request".to_string(),
                ));
            }
        }

        let request_ids = self
            .requests
            .iter()
            .map(|request| request.id.as_str())
            .collect::<HashSet<_>>();
        if request_ids.len() != self.requests.len() || request_ids != scheduled_ids {
            return Err(Error::InferenceError(
                "physical executor request snapshots do not match scheduled rows".to_string(),
            ));
        }

        let is_prefill = self.scheduled[0].is_prefill;
        if self
            .scheduled
            .iter()
            .any(|scheduled| scheduled.is_prefill != is_prefill)
        {
            return Err(Error::InferenceError(
                "one physical batch cannot mix prefill and decode dispatch".to_string(),
            ));
        }
        Ok(())
    }

    pub fn is_prefill(&self) -> bool {
        self.scheduled
            .first()
            .is_some_and(|scheduled| scheduled.is_prefill)
    }
}

pub trait ModelExecutor: Send + Sync {
    /// Effective loaded-model/request/backend execution profile. Executors
    /// that cannot prove their behavior return `None` and therefore remain on
    /// the conservative compatibility path.
    fn execution_profile(&self, _request: &EngineCoreRequest) -> Option<ExecutionProfile> {
        None
    }

    /// Effective capabilities. The default is deliberately conservative so an
    /// executor must opt in before the scheduler relies on incremental or batch behavior.
    fn execution_capabilities(&self, request: &EngineCoreRequest) -> ExecutionCapabilities {
        self.execution_profile(request)
            .map(|profile| profile.capabilities())
            .unwrap_or_default()
    }

    /// Execute one already-validated physical batch transaction. Native
    /// tensor adapters override this boundary; compatibility executors retain
    /// their existing phase methods at width one.
    fn execute_physical_batch(
        &self,
        execution: PhysicalBatchExecution<'_>,
    ) -> PhysicalDispatchResult {
        let width = execution.scheduled.len().max(1);
        execution.validate().map_err(|error| {
            PhysicalDispatchError::not_started(error, width, FailureOrigin::ExecutorValidation)
        })?;
        let dispatch = execution.expected_dispatch();
        let result = if execution.is_prefill() {
            self.execute_prefill(execution.requests, execution.scheduled)
        } else {
            self.execute_decode(execution.requests, execution.scheduled)
        };
        result
            .map(|mut outputs| {
                let actual_dispatch = if !outputs.is_empty()
                    && outputs
                        .iter()
                        .all(|output| output.provenance.dispatch_state == DispatchState::NotStarted)
                {
                    BatchDispatch::not_dispatched(width)
                } else {
                    dispatch
                };
                for output in &mut outputs {
                    output.dispatch = actual_dispatch;
                }
                outputs
            })
            .map_err(|error| PhysicalDispatchError::started(error, dispatch, FailureOrigin::Model))
    }

    /// Execute prefill pass for newly admitted or in-progress prefill requests.
    fn execute_prefill(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>>;

    /// Execute decode pass for running requests.
    fn execute_decode(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>>;

    /// Check if the executor is ready.
    fn is_ready(&self) -> bool;

    /// Initialize the executor (load models, etc.)
    fn initialize(&mut self) -> Result<()>;

    /// Shutdown the executor.
    fn shutdown(&mut self) -> Result<()>;

    /// Cleanup transient per-request state held by the executor backend.
    fn cleanup_request(&self, _request_id: &str) -> CacheReleaseReport {
        CacheReleaseReport::unconfirmed()
    }

    /// Cleanup state for one exact request incarnation. Legacy executors may
    /// conservatively clear all state for the public request ID.
    fn cleanup_session(&self, session: &SessionKey) -> CacheReleaseReport {
        self.cleanup_request(&session.request_id)
    }

    /// Purge model-owned reusable cache state before one model is unloaded.
    fn purge_model_cache(&self, _variant: ModelVariant) -> CacheReleaseReport {
        CacheReleaseReport::unconfirmed()
    }
}

/// Proof returned after an executor cache cleanup request. Preemption may only
/// recompute when the executor confirms that the exact session no longer owns
/// tensor cache state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheReleaseReport {
    pub outcome: CacheReleaseOutcome,
    pub confirmed: bool,
    pub released_sessions: usize,
    pub busy_sessions: usize,
}

/// Typed executor cleanup result. `BusyInFlight` is deliberately distinct from
/// a generic unconfirmed cleanup: the exact session is still owned by a model
/// forward and must be retried after that forward reaches its RAII boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheReleaseOutcome {
    Confirmed,
    BusyInFlight,
    Unconfirmed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PendingQuantumDecision {
    Commit,
    Abort,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PendingQuantumFinalizeStatus {
    NotFound,
    Finalized,
}

pub(crate) trait PendingQuantumFinalizer: Send + Sync {
    fn contains(&self, plan_id: PlanId, session: &SessionKey) -> bool;

    fn prepare(
        &self,
        plan_id: PlanId,
        session: &SessionKey,
        decision: PendingQuantumDecision,
    ) -> Result<PendingQuantumFinalizeStatus>;

    fn publish(
        &self,
        plan_id: PlanId,
        session: &SessionKey,
    ) -> Result<PendingQuantumFinalizeStatus>;

    fn discard(&self, plan_id: PlanId, session: &SessionKey);
}

impl CacheReleaseReport {
    pub const fn confirmed(released_sessions: usize) -> Self {
        Self {
            outcome: CacheReleaseOutcome::Confirmed,
            confirmed: true,
            released_sessions,
            busy_sessions: 0,
        }
    }

    pub const fn busy_in_flight(released_sessions: usize, busy_sessions: usize) -> Self {
        Self {
            outcome: CacheReleaseOutcome::BusyInFlight,
            confirmed: false,
            released_sessions,
            busy_sessions,
        }
    }

    pub const fn unconfirmed() -> Self {
        Self {
            outcome: CacheReleaseOutcome::Unconfirmed,
            confirmed: false,
            released_sessions: 0,
            busy_sessions: 0,
        }
    }
}

enum ExecutorStateSlot<T> {
    Ready(T),
    InFlight,
    Poisoned,
}

type ExecutorStateStore<T> = Mutex<HashMap<SessionKey, ExecutorStateSlot<T>>>;

/// Exclusive ownership of one executor session state while a physical forward
/// is running. The `InFlight` marker stays visible in the map for cleanup for
/// the complete lifetime of this lease.
///
/// Before model state is mutated, dropping the lease restores a previously
/// ready state. Once `mark_dirty` is called, an uncommitted unwind drops the
/// possibly-mutated state and leaves a `Poisoned` marker so cleanup cannot
/// mistake the temporary absence for a successful release.
struct ExecutorStateLease<'a, T> {
    store: &'a ExecutorStateStore<T>,
    session: SessionKey,
    state: Option<T>,
    label: &'static str,
    dirty: bool,
    armed: bool,
}

impl<'a, T> ExecutorStateLease<'a, T> {
    fn checkout(
        store: &'a ExecutorStateStore<T>,
        session: SessionKey,
        label: &'static str,
    ) -> Result<Self> {
        use std::collections::hash_map::Entry;

        let state = {
            let mut states = store
                .lock()
                .map_err(|_| Error::InferenceError(format!("{label} state mutex poisoned")))?;
            match states.entry(session.clone()) {
                Entry::Vacant(entry) => {
                    entry.insert(ExecutorStateSlot::InFlight);
                    None
                }
                Entry::Occupied(mut entry) => match entry.get() {
                    ExecutorStateSlot::Ready(_) => {
                        let previous = entry.insert(ExecutorStateSlot::InFlight);
                        let ExecutorStateSlot::Ready(state) = previous else {
                            unreachable!("ready executor state changed under one mutex guard")
                        };
                        Some(state)
                    }
                    ExecutorStateSlot::InFlight => {
                        return Err(Error::InferenceError(format!(
                            "{label} session {}:{} is already in flight",
                            session.request_id, session.epoch
                        )))
                    }
                    ExecutorStateSlot::Poisoned => {
                        return Err(Error::InferenceError(format!(
                            "{label} session {}:{} is poisoned and requires cleanup",
                            session.request_id, session.epoch
                        )))
                    }
                },
            }
        };

        Ok(Self {
            store,
            session,
            state,
            label,
            dirty: false,
            armed: true,
        })
    }

    fn state(&self) -> Option<&T> {
        self.state.as_ref()
    }

    fn state_mut(&mut self) -> Option<&mut T> {
        self.state.as_mut()
    }

    fn require_state_mut(&mut self) -> Result<&mut T> {
        self.state.as_mut().ok_or_else(|| {
            Error::InferenceError(format!(
                "{} session {}:{} has no checked-out state",
                self.label, self.session.request_id, self.session.epoch
            ))
        })
    }

    fn discard_state(&mut self) {
        self.state.take();
        self.dirty = false;
    }

    fn install_state(&mut self, state: T) -> Result<()> {
        if self.state.is_some() {
            return Err(Error::InferenceError(format!(
                "{} session {}:{} replaced an owned state without releasing it",
                self.label, self.session.request_id, self.session.epoch
            )));
        }
        self.state = Some(state);
        Ok(())
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    fn mark_clean(&mut self) {
        self.dirty = false;
    }

    fn restore(mut self) -> Result<()> {
        let state = self.state.take().ok_or_else(|| {
            Error::InferenceError(format!(
                "{} session {}:{} cannot restore an empty state",
                self.label, self.session.request_id, self.session.epoch
            ))
        })?;
        let result = self.replace_in_flight(ExecutorStateSlot::Ready(state));
        if result.is_ok() {
            self.armed = false;
        }
        result
    }

    fn release(mut self) -> Result<()> {
        let mut states = self
            .store
            .lock()
            .map_err(|_| Error::InferenceError(format!("{} state mutex poisoned", self.label)))?;
        match states.get(&self.session) {
            Some(ExecutorStateSlot::InFlight) => {
                states.remove(&self.session);
                self.armed = false;
                Ok(())
            }
            _ => Err(self.transition_collision()),
        }
    }

    /// Transfer the checked-out value to a post-execution transaction while
    /// deliberately retaining the visible `InFlight` ownership marker.
    fn defer(mut self) -> Result<T> {
        {
            let states = self.store.lock().map_err(|_| {
                Error::InferenceError(format!("{} state mutex poisoned", self.label))
            })?;
            if !matches!(states.get(&self.session), Some(ExecutorStateSlot::InFlight)) {
                return Err(self.transition_collision());
            }
        }
        let state = self.state.take().ok_or_else(|| {
            Error::InferenceError(format!(
                "{} session {}:{} cannot defer an empty state",
                self.label, self.session.request_id, self.session.epoch
            ))
        })?;
        self.armed = false;
        Ok(state)
    }

    fn replace_in_flight(&mut self, replacement: ExecutorStateSlot<T>) -> Result<()> {
        let mut states = self
            .store
            .lock()
            .map_err(|_| Error::InferenceError(format!("{} state mutex poisoned", self.label)))?;
        match states.get(&self.session) {
            Some(ExecutorStateSlot::InFlight) => {
                states.insert(self.session.clone(), replacement);
                Ok(())
            }
            _ => Err(self.transition_collision()),
        }
    }

    fn transition_collision(&self) -> Error {
        Error::InferenceError(format!(
            "{} session {}:{} lost its in-flight ownership marker",
            self.label, self.session.request_id, self.session.epoch
        ))
    }
}

struct VoxtralRealtimeStateCoordinator {
    states: ExecutorStateStore<ActiveVoxtralRealtime>,
    pending: Mutex<HashMap<PlanId, PendingVoxtralRealtimeQuantum>>,
    prepared: Mutex<HashMap<PlanId, PreparedVoxtralRealtimeQuantum>>,
}

impl VoxtralRealtimeStateCoordinator {
    fn new() -> Self {
        Self {
            states: Mutex::new(HashMap::new()),
            pending: Mutex::new(HashMap::new()),
            prepared: Mutex::new(HashMap::new()),
        }
    }

    fn register(
        &self,
        plan_id: PlanId,
        pending: PendingVoxtralRealtimeQuantum,
    ) -> std::result::Result<(), (Error, PendingVoxtralRealtimeQuantum)> {
        let mut quanta = match self.pending.lock() {
            Ok(quanta) => quanta,
            Err(_) => {
                return Err((
                    Error::InferenceError("Voxtral pending-quantum mutex poisoned".to_string()),
                    pending,
                ));
            }
        };
        if quanta.contains_key(&plan_id) {
            return Err((
                Error::InferenceError(format!(
                    "Voxtral pending quantum already exists for plan {plan_id}"
                )),
                pending,
            ));
        }
        quanta.insert(plan_id, pending);
        Ok(())
    }

    fn replace_in_flight(
        &self,
        session: &SessionKey,
        replacement: Option<ActiveVoxtralRealtime>,
    ) -> Result<()> {
        // Publication follows a successful model prepare and managed-KV
        // finalize. Recover the guard so mutex poisoning cannot strand a
        // successfully committed authoritative transaction.
        let mut states = self
            .states
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !matches!(states.get(session), Some(ExecutorStateSlot::InFlight)) {
            states.insert(session.clone(), ExecutorStateSlot::Poisoned);
            return Err(Error::InferenceError(format!(
                "Voxtral realtime session {}:{} lost its pending ownership marker",
                session.request_id, session.epoch
            )));
        }
        match replacement {
            Some(active) => {
                states.insert(session.clone(), ExecutorStateSlot::Ready(active));
            }
            None => {
                states.remove(session);
            }
        }
        Ok(())
    }

    fn poison_in_flight(&self, session: &SessionKey) {
        let mut states = self
            .states
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        states.insert(session.clone(), ExecutorStateSlot::Poisoned);
    }

    fn resolve_pending(
        &self,
        mut pending: PendingVoxtralRealtimeQuantum,
        decision: PendingQuantumDecision,
    ) -> Result<PreparedVoxtralRealtimeQuantum> {
        let operation = match decision {
            PendingQuantumDecision::Commit => pending.active.model.commit_realtime_quantum(
                &mut pending.active.state,
                &pending.cache,
                &mut pending.checkpoint,
            ),
            PendingQuantumDecision::Abort => {
                pending.active.last_tokens_generated = pending.prior_last_tokens_generated;
                pending.active.stream_sequence = pending.prior_stream_sequence;
                pending.active.input_sample_rate = pending.prior_input_sample_rate;
                pending.active.model.rollback_realtime_quantum(
                    &mut pending.active.state,
                    &mut pending.cache,
                    &mut pending.checkpoint,
                )
            }
        };
        if let Err(error) = operation {
            self.poison_in_flight(&pending.session);
            return Err(error);
        }
        if decision == PendingQuantumDecision::Commit {
            pending.active.last_tokens_generated = pending.active.state.tokens_generated();
        }
        let replacement = if decision == PendingQuantumDecision::Commit && pending.finished {
            None
        } else {
            Some(pending.active)
        };
        Ok(PreparedVoxtralRealtimeQuantum {
            session: pending.session,
            replacement,
        })
    }

    fn abort_matching(
        &self,
        predicate: impl Fn(&PendingVoxtralRealtimeQuantum) -> bool,
    ) -> Result<usize> {
        let pending = {
            let mut quanta = self
                .pending
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let ids = quanta
                .iter()
                .filter_map(|(id, pending)| predicate(pending).then_some(*id))
                .collect::<Vec<_>>();
            ids.into_iter()
                .filter_map(|id| quanta.remove(&id))
                .collect::<Vec<_>>()
        };
        let count = pending.len();
        let mut failure = None;
        for quantum in pending {
            let session = quantum.session.clone();
            let result = self
                .resolve_pending(quantum, PendingQuantumDecision::Abort)
                .and_then(|prepared| self.replace_in_flight(&session, prepared.replacement));
            if let Err(error) = result {
                error!(error = %error, "Failed to abort a pending Voxtral realtime quantum");
                failure.get_or_insert(error);
            }
        }
        match failure {
            Some(error) => Err(error),
            None => Ok(count),
        }
    }

    fn has_prepared(&self) -> bool {
        !self
            .prepared
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_empty()
    }
}

impl PendingQuantumFinalizer for VoxtralRealtimeStateCoordinator {
    fn contains(&self, plan_id: PlanId, session: &SessionKey) -> bool {
        self.pending
            .lock()
            .ok()
            .and_then(|pending| pending.get(&plan_id).map(|row| &row.session == session))
            .unwrap_or(false)
    }

    fn prepare(
        &self,
        plan_id: PlanId,
        session: &SessionKey,
        decision: PendingQuantumDecision,
    ) -> Result<PendingQuantumFinalizeStatus> {
        let pending = {
            let mut quanta = self.pending.lock().map_err(|_| {
                Error::InferenceError("Voxtral pending-quantum mutex poisoned".to_string())
            })?;
            let Some(pending) = quanta.get(&plan_id) else {
                return Ok(PendingQuantumFinalizeStatus::NotFound);
            };
            if &pending.session != session {
                return Err(Error::InferenceError(format!(
                    "pending quantum plan {plan_id} belongs to a different session"
                )));
            }
            quanta
                .remove(&plan_id)
                .expect("pending quantum was present")
        };
        let prepared = self.resolve_pending(pending, decision)?;
        let mut rows = self.prepared.lock().map_err(|_| {
            self.poison_in_flight(session);
            Error::InferenceError("Voxtral prepared-quantum mutex poisoned".to_string())
        })?;
        if rows.insert(plan_id, prepared).is_some() {
            self.poison_in_flight(session);
            return Err(Error::InferenceError(format!(
                "Voxtral prepared quantum already exists for plan {plan_id}"
            )));
        }
        Ok(PendingQuantumFinalizeStatus::Finalized)
    }

    fn publish(
        &self,
        plan_id: PlanId,
        session: &SessionKey,
    ) -> Result<PendingQuantumFinalizeStatus> {
        let prepared = {
            let mut rows = self
                .prepared
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let Some(prepared) = rows.remove(&plan_id) else {
                return Ok(PendingQuantumFinalizeStatus::NotFound);
            };
            prepared
        };
        if &prepared.session != session {
            self.poison_in_flight(&prepared.session);
            return Err(Error::InferenceError(format!(
                "prepared quantum plan {plan_id} belongs to a different session"
            )));
        }
        self.replace_in_flight(session, prepared.replacement)?;
        Ok(PendingQuantumFinalizeStatus::Finalized)
    }

    fn discard(&self, plan_id: PlanId, session: &SessionKey) {
        let prepared = self
            .prepared
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(&plan_id);
        if let Some(prepared) = prepared {
            self.poison_in_flight(&prepared.session);
        } else {
            self.poison_in_flight(session);
        }
    }
}

impl<T> Drop for ExecutorStateLease<'_, T> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }

        use std::collections::hash_map::Entry;
        let mut states = self
            .store
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match states.entry(self.session.clone()) {
            Entry::Occupied(mut entry) if matches!(entry.get(), ExecutorStateSlot::InFlight) => {
                if self.dirty {
                    entry.insert(ExecutorStateSlot::Poisoned);
                } else if let Some(state) = self.state.take() {
                    entry.insert(ExecutorStateSlot::Ready(state));
                } else {
                    entry.remove();
                }
            }
            Entry::Vacant(entry) => {
                // A missing marker means another path observed ownership that
                // was not actually released. Fence the session until cleanup.
                entry.insert(ExecutorStateSlot::Poisoned);
                tracing::error!(
                    request_id = %self.session.request_id,
                    epoch = self.session.epoch,
                    state = self.label,
                    "executor state lease lost its in-flight marker"
                );
            }
            Entry::Occupied(mut entry) => {
                entry.insert(ExecutorStateSlot::Poisoned);
                tracing::error!(
                    request_id = %self.session.request_id,
                    epoch = self.session.epoch,
                    state = self.label,
                    "executor state lease collided with another visible state; session fenced until cleanup"
                );
            }
        }
    }
}

pub struct NativeExecutor {
    config: WorkerConfig,
    initialized: bool,
    loaded_tts_model: Option<Arc<Qwen3TtsModel>>,
    chat_decode_states: ExecutorStateStore<ActiveChatDecode>,
    asr_decode_states: ExecutorStateStore<ActiveAsrDecode>,
    voxtral_realtime: Arc<VoxtralRealtimeStateCoordinator>,
    qwen_tts_decode_states: ExecutorStateStore<ActiveQwenTtsDecode>,
}

impl NativeExecutor {
    /// Create a new native executor.
    pub fn new(config: WorkerConfig) -> Self {
        Self::with_voxtral_realtime(config, Arc::new(VoxtralRealtimeStateCoordinator::new()))
    }

    fn with_voxtral_realtime(
        config: WorkerConfig,
        voxtral_realtime: Arc<VoxtralRealtimeStateCoordinator>,
    ) -> Self {
        Self {
            config,
            initialized: false,
            loaded_tts_model: None,
            chat_decode_states: Mutex::new(HashMap::new()),
            asr_decode_states: Mutex::new(HashMap::new()),
            voxtral_realtime,
            qwen_tts_decode_states: Mutex::new(HashMap::new()),
        }
    }

    fn qwen_model_for_request(
        &self,
        request: &EngineCoreRequest,
    ) -> Result<(Arc<Qwen3TtsModel>, Option<QwenTtsModelLease>)> {
        if let Some(lease) = request.prepared_qwen_tts_model_lease_for_executor()? {
            return Ok((lease.model_arc(), Some(lease)));
        }
        if let Some(registry) = &self.config.model_registry {
            let variant = request.model_variant.ok_or_else(|| {
                Error::InferenceError("Qwen TTS request is missing model variant".to_string())
            })?;
            let lease = registry.try_get_qwen_tts_lease(variant).ok_or_else(|| {
                Error::ModelNotFound(format!("Qwen TTS model {variant} is not loaded"))
            })?;
            return Ok((lease.model_arc(), Some(lease)));
        }
        self.loaded_tts_model
            .clone()
            .map(|model| (model, None))
            .ok_or_else(|| Error::InferenceError("Executor model not initialized".to_string()))
    }

    fn asr_model_for_request(
        &self,
        request: &EngineCoreRequest,
        variant: ModelVariant,
    ) -> Result<(Arc<NativeAsrModel>, AsrModelLease)> {
        if let Some(lease) = request.prepared_asr_model_lease_for_executor()? {
            return Ok((lease.model_arc(), lease));
        }
        self.with_registry(|registry| {
            let lease = registry.try_get_asr_lease(variant).ok_or_else(|| {
                Error::ModelNotFound(format!("ASR model {variant} is not loaded"))
            })?;
            Ok((lease.model_arc(), lease))
        })
    }

    fn with_registry<T>(&self, f: impl FnOnce(&ModelRegistry) -> Result<T>) -> Result<T> {
        let registry =
            self.config.model_registry.as_ref().ok_or_else(|| {
                Error::InferenceError("Model registry is not configured".to_string())
            })?;
        f(registry)
    }

    fn run_blocking<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
        let run_catching_panic = || {
            let unwind_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
            match unwind_result {
                Ok(result) => result,
                Err(payload) => {
                    let message = panic_payload_to_string(payload.as_ref());
                    error!("Model execution panicked: {message}");
                    std::panic::resume_unwind(payload)
                }
            }
        };

        match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
                // Long-running CPU inference should not monopolize Tokio workers; this allows
                // async tasks (including SSE stream forwarding) to continue making progress.
                tokio::task::block_in_place(run_catching_panic)
            }
            _ => run_catching_panic(),
        }
    }
}

fn is_isolated_continuous_model_quantum(scheduled: &[ScheduledRequest]) -> bool {
    scheduled.len() == 1 && !scheduled[0].is_prefill && scheduled[0].num_tokens > 1
}

fn has_native_vibevoice_tokenizer_batch(scheduled: &[ScheduledRequest]) -> bool {
    scheduled.len() > 1
        && scheduled.iter().all(|scheduled| {
            matches!(
                &scheduled.work,
                WorkUnit::SequenceStep {
                    auxiliary_state: Some(spans),
                    ..
                } if !spans.is_empty()
            )
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NativeBatchSupport {
    prefill: NativeBatchMode,
    decode: NativeBatchMode,
}

impl NativeBatchSupport {
    const NONE: Self = Self {
        prefill: NativeBatchMode::None,
        decode: NativeBatchMode::None,
    };

    fn is_native(self) -> bool {
        self.prefill != NativeBatchMode::None || self.decode != NativeBatchMode::None
    }
}

/// Intersect a load-sealed stage declaration with an implemented model call.
/// Merely publishing `NativeBatchMode` in catalog or request data is not
/// sufficient. Each audio family remains `NONE` until its exact loaded model
/// and adapter both expose a real multi-row call at this boundary.
fn loaded_native_batch_support(request: &EngineCoreRequest) -> NativeBatchSupport {
    let (prefill_supported, decode_supported) = match request.task_type {
        TaskType::Chat => (
            false,
            request
                .prepared_chat_model_for_executor()
                .is_ok_and(|model| model.supports_continuous_decode_batch()),
        ),
        TaskType::ASR => request
            .prepared_asr_model_for_executor()
            .ok()
            .flatten()
            .map(|model| {
                (
                    model.supports_static_prefill_batch(),
                    model.supports_continuous_decode_batch(),
                )
            })
            .unwrap_or((false, false)),
        TaskType::TTS => (
            false,
            request
                .prepared_qwen_tts_model_for_executor()
                .is_ok_and(|model| {
                    model.is_some_and(|model| model.supports_continuous_decode_batch())
                }),
        ),
        TaskType::SpeechToSpeech => (false, false),
    };
    if !prefill_supported && !decode_supported {
        return NativeBatchSupport::NONE;
    }
    let Some(binding) = request.execution_adapter_binding() else {
        return NativeBatchSupport::NONE;
    };
    let decode_work = WorkUnit::SequenceStep {
        phase: SequencePhase::Decode,
        input: super::InputRange { start: 0, end: 1 },
        max_output_steps: 1,
        auxiliary_state: None,
    };
    let decode = decode_supported
        && binding.stage_for_work(&decode_work).is_ok_and(|stage| {
            stage.selector == StageWorkSelector::SequenceDecode
                && stage.batch_mode == NativeBatchMode::Continuous
                && stage.concurrency == ConcurrencyClass::Batchable
        });
    let prefill_work = WorkUnit::SequenceStep {
        phase: SequencePhase::Prefill,
        input: super::InputRange { start: 0, end: 1 },
        max_output_steps: 1,
        auxiliary_state: None,
    };
    let prefill = prefill_supported
        && binding.stage_for_work(&prefill_work).is_ok_and(|stage| {
            stage.selector == StageWorkSelector::SequencePrefill
                && stage.batch_mode == NativeBatchMode::Static
                && stage.concurrency == ConcurrencyClass::Batchable
        });
    NativeBatchSupport {
        prefill: if prefill {
            NativeBatchMode::Static
        } else {
            NativeBatchMode::None
        },
        decode: if decode {
            NativeBatchMode::Continuous
        } else {
            NativeBatchMode::None
        },
    }
}

fn resolved_resumable_prefill_mode(
    chunking_enabled: bool,
    exact_model_proof: Option<bool>,
) -> PrefillMode {
    if chunking_enabled && exact_model_proof == Some(true) {
        PrefillMode::Incremental
    } else {
        PrefillMode::Full
    }
}

impl ModelExecutor for NativeExecutor {
    fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
        let variant = request.model_variant?;
        let mut profile = ExecutionProfile::fail_closed(
            self.config.backend,
            Some(variant),
            ExecutionMode::Atomic,
        );
        profile.compute_dtype = self.config.dtype.clone();
        profile.kv_dtype = self.config.kv_cache_dtype.clone();
        profile.cache_namespace = Some(format!(
            "{}:{}:{}:{}",
            variant,
            self.config.backend.as_str(),
            self.config.dtype,
            self.config.kv_cache_dtype
        ));

        let loaded_incremental = match request.task_type {
            super::types::TaskType::Chat => {
                request
                    .prepared_chat_model_for_executor()
                    .ok()
                    .map(|model| match model.as_ref() {
                        NativeChatModel::Qwen3(model) => model.supports_incremental_decode(),
                        NativeChatModel::Qwen35(model) => model.supports_incremental_decode(),
                        NativeChatModel::Qwen38(model) => model.supports_incremental_decode(),
                        NativeChatModel::Gemma3(model) => model.supports_incremental_decode(),
                        NativeChatModel::Lfm2(_) => false,
                    })
            }
            super::types::TaskType::ASR => request
                .prepared_asr_model_for_executor()
                .ok()
                .flatten()
                .or_else(|| {
                    self.config
                        .model_registry
                        .as_ref()
                        .and_then(|registry| registry.try_get_asr(variant))
                })
                .map(|model| model.supports_incremental_decode()),
            super::types::TaskType::TTS => {
                let loaded = request
                    .prepared_qwen_tts_model_for_executor()
                    .ok()
                    .flatten()
                    .or_else(|| {
                        self.config
                            .model_registry
                            .as_ref()
                            .and_then(|registry| registry.try_get_qwen_tts(variant))
                    })
                    .is_some()
                    || (self.config.model_registry.is_none() && self.loaded_tts_model.is_some());
                loaded.then_some(variant.family() == crate::catalog::ModelFamily::Qwen3Tts)
            }
            super::types::TaskType::SpeechToSpeech => self
                .config
                .model_registry
                .as_ref()
                .and_then(|registry| registry.try_get_audio_chat(variant))
                .map(|_| false),
        };
        let native_batch_support = loaded_native_batch_support(request);
        profile.resolved_from_loaded_model = loaded_incremental.is_some();
        let asr_long_form =
            request.task_type == super::types::TaskType::ASR && request.uses_asr_long_form_atomic();
        let implementation_incremental = !asr_long_form
            && loaded_incremental.unwrap_or_else(|| match request.task_type {
                super::types::TaskType::Chat => {
                    matches!(
                        variant.family(),
                        crate::catalog::ModelFamily::Qwen35Chat
                            | crate::catalog::ModelFamily::Qwen38Chat
                    ) || matches!(
                        variant,
                        ModelVariant::Qwen306B
                            | ModelVariant::Qwen306B4Bit
                            | ModelVariant::Qwen317B
                            | ModelVariant::Qwen317B4Bit
                    )
                }
                super::types::TaskType::ASR => {
                    variant.family() == crate::catalog::ModelFamily::Qwen3Asr
                }
                super::types::TaskType::TTS => {
                    variant.family() == crate::catalog::ModelFamily::Qwen3Tts
                }
                super::types::TaskType::SpeechToSpeech => false,
            });
        let resumable_prefill_proof = match request.task_type {
            super::types::TaskType::Chat => request
                .prepared_chat_model_for_executor()
                .ok()
                .map(|model| model.supports_resumable_prefill()),
            super::types::TaskType::ASR => request
                .prepared_asr_model_for_executor()
                .ok()
                .flatten()
                .map(|model| model.supports_resumable_prefill()),
            super::types::TaskType::TTS => request
                .prepared_qwen_tts_model_for_executor()
                .ok()
                .flatten()
                .map(|model| model.supports_resumable_prefill()),
            super::types::TaskType::SpeechToSpeech => None,
        };

        if implementation_incremental {
            profile.mode = ExecutionMode::Sequence;
            // Scheduler-level spans require a stronger capability than
            // incremental decode: the exact loaded family must publish a
            // resumable prefill safe point. Unsupported families remain full.
            profile.prefill = resolved_resumable_prefill_mode(
                self.config.enable_chunked_prefill,
                resumable_prefill_proof,
            );
            profile.incremental_decode = true;
            profile.recompute_safe = profile.resolved_from_loaded_model;
            profile.cache_release_safe = profile.resolved_from_loaded_model;
        }
        if matches!(request.task_type, super::types::TaskType::ASR) && !implementation_incremental {
            // Long audio can switch to a full chunk-plan operation after media
            // decode, so cancellation is conservatively operation-boundary.
            profile.cancellation = CancellationGranularity::OperationBoundary;
        }
        if asr_long_form {
            profile.mode = ExecutionMode::Atomic;
            profile.prefill = PrefillMode::None;
            profile.incremental_decode = false;
            profile.prefill_batch = NativeBatchMode::None;
            profile.decode_batch = NativeBatchMode::None;
            profile.cache_mode = CacheMode::None;
            profile.cache_namespace = None;
            profile.kv_dtype = "none".to_string();
            profile.concurrency = ConcurrencyClass::Exclusive;
            profile.max_batch_size = 1;
            profile.recompute_safe = false;
            profile.cache_release_safe = false;
            profile.prefix_reuse_safe = false;
        }

        if native_batch_support.is_native() {
            profile.prefill_batch = native_batch_support.prefill;
            profile.decode_batch = native_batch_support.decode;
            profile.concurrency = ConcurrencyClass::Batchable;
            profile.max_batch_size = self.config.max_tensor_batch_size.max(1);
        } else {
            let request_parallel_width = if can_parallelize_requests(self.config.backend) {
                self.config.request_parallelism.max(1)
            } else {
                1
            };
            profile.prefill_batch = NativeBatchMode::None;
            profile.decode_batch = NativeBatchMode::None;
            profile.concurrency = if request_parallel_width > 1 {
                ConcurrencyClass::Batchable
            } else {
                ConcurrencyClass::Exclusive
            };
            profile.max_batch_size = request_parallel_width;
        }
        if request.managed_cache_runtime().is_some() {
            profile.cache_mode = CacheMode::ExternalPaged;
        }
        if asr_long_form {
            profile.prefill_batch = NativeBatchMode::None;
            profile.decode_batch = NativeBatchMode::None;
            profile.concurrency = ConcurrencyClass::Exclusive;
            profile.max_batch_size = 1;
            profile.cache_mode = CacheMode::None;
            profile.cache_namespace = None;
            profile.kv_dtype = "none".to_string();
        }
        if matches!(request.task_type, super::types::TaskType::Chat) {
            profile.preferred_decode_tokens = request
                .prepared_chat_model_for_executor()
                .ok()
                .and_then(|model| match model.as_ref() {
                    NativeChatModel::Qwen38(model) => Some(model.preferred_decode_tokens()),
                    _ => None,
                })
                .unwrap_or(1);
        }
        Some(profile)
    }

    fn execute_physical_batch(
        &self,
        execution: PhysicalBatchExecution<'_>,
    ) -> PhysicalDispatchResult {
        let width = execution.scheduled.len().max(1);
        let expected_dispatch = execution.expected_dispatch();
        execution.validate().map_err(|error| {
            PhysicalDispatchError::not_started(error, width, FailureOrigin::ExecutorValidation)
        })?;
        if !self.initialized {
            return Err(PhysicalDispatchError::not_started(
                Error::InferenceError("Executor not initialized".into()),
                width,
                FailureOrigin::ExecutorValidation,
            ));
        }
        if execution.batch.mode != NativeBatchMode::None {
            let route = NativeBatchRoute::resolve(&execution).map_err(|error| {
                PhysicalDispatchError::not_started(error, width, FailureOrigin::ExecutorValidation)
            })?;
            if execution.scheduled.len() > self.config.max_tensor_batch_size.max(1) {
                return Err(PhysicalDispatchError::not_started(
                    Error::Overloaded(
                        "native tensor batch exceeds the backend width cap".to_string(),
                    ),
                    width,
                    FailureOrigin::ExecutorValidation,
                ));
            }
            if is_isolated_continuous_model_quantum(execution.scheduled) {
                // An isolated model-preferred quantum is still planned through
                // the continuous stage so it can yield back to shared
                // membership afterwards, but its model work is scalar/MTP and
                // must use the existing transactional scalar handler. This
                // keeps tensor-batch telemetry truthful and preserves the
                // shared handler's one-token-per-row invariant.
                let result = self.execute_requests_with_rows(
                    execution.requests,
                    execution.scheduled,
                    Some(&execution.batch.rows),
                );
                if result
                    .as_ref()
                    .is_ok_and(|outputs| outputs.iter().all(|output| output.output.error.is_none()))
                {
                    crate::engine::metrics::record_engine_model_call(
                        crate::engine::metrics::EngineModelCall::ScalarRows {
                            envelope: NativeBatchMode::Continuous,
                            rows: 1,
                        },
                    );
                }
                return result.map_err(|error| {
                    PhysicalDispatchError::started(
                        error,
                        BatchDispatch::serial(),
                        FailureOrigin::Model,
                    )
                });
            }
            let result = match route {
                NativeBatchRoute::ChatContinuousDecode { .. } => self
                    .execute_continuous_chat_requests_with_rows(
                        execution.requests,
                        execution.scheduled,
                        Some(&execution.batch.rows),
                    ),
                NativeBatchRoute::Audio {
                    task: TaskType::ASR,
                    stage: NativeAudioStage::SequencePrefill,
                    mode: NativeBatchMode::Static,
                    ..
                } if execution.batch.lane.kernel_mode
                    == crate::models::architectures::vibevoice::VIBEVOICE_ASR_PREFILL_STAGE
                    && has_native_vibevoice_tokenizer_batch(execution.scheduled)
                    && execution.requests.iter().all(|request| {
                        request.model_variant.is_some_and(|variant| {
                            variant.family() == crate::catalog::ModelFamily::VibeVoiceAsr
                        })
                    }) =>
                {
                    self.execute_static_vibevoice_prefill_requests_with_rows(
                        execution.requests,
                        execution.scheduled,
                        Some(&execution.batch.rows),
                    )
                }
                NativeBatchRoute::Audio {
                    task: TaskType::ASR,
                    stage: NativeAudioStage::SequencePrefill,
                    mode: NativeBatchMode::Static,
                    ..
                } if execution.batch.lane.kernel_mode
                    == crate::models::architectures::vibevoice::VIBEVOICE_ASR_PREFILL_STAGE
                    && execution.requests.iter().all(|request| {
                        request.model_variant.is_some_and(|variant| {
                            variant.family() == crate::catalog::ModelFamily::VibeVoiceAsr
                        })
                    }) =>
                {
                    let result = self.execute_requests_with_rows(
                        execution.requests,
                        execution.scheduled,
                        Some(&execution.batch.rows),
                    );
                    if result.as_ref().is_ok_and(|outputs| {
                        outputs.iter().all(|output| output.output.error.is_none())
                    }) {
                        crate::engine::metrics::record_engine_model_call(
                            crate::engine::metrics::EngineModelCall::ScalarRows {
                                envelope: NativeBatchMode::Static,
                                rows: execution.scheduled.len(),
                            },
                        );
                    }
                    result
                }
                NativeBatchRoute::Audio {
                    task: TaskType::ASR,
                    stage: NativeAudioStage::SequenceDecode,
                    mode: NativeBatchMode::Continuous,
                    ..
                } => self.execute_continuous_asr_requests_with_rows(
                    execution.requests,
                    execution.scheduled,
                    Some(&execution.batch.rows),
                ),
                NativeBatchRoute::Audio {
                    task: TaskType::TTS,
                    stage: NativeAudioStage::SequenceDecode,
                    mode: NativeBatchMode::Continuous,
                    ..
                } => self.execute_continuous_tts_requests_with_rows(
                    execution.requests,
                    execution.scheduled,
                    Some(&execution.batch.rows),
                ),
                NativeBatchRoute::Audio { .. } => {
                    // An exact audio stage without a registered family call
                    // fails before model entry; catalog identity alone never
                    // fabricates native tensor execution.
                    return Err(PhysicalDispatchError::not_started(
                        Error::InferenceError(
                            "loaded audio stage has no registered native model call".to_string(),
                        ),
                        width,
                        FailureOrigin::ExecutorValidation,
                    ));
                }
            };
            return result.map_err(|error| {
                PhysicalDispatchError::started(error, expected_dispatch, FailureOrigin::Model)
            });
        }
        let result = self.execute_requests_with_rows(
            execution.requests,
            execution.scheduled,
            Some(&execution.batch.rows),
        );
        result.map_err(|error| {
            PhysicalDispatchError::started(error, expected_dispatch, FailureOrigin::Model)
        })
    }

    fn execute_prefill(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        if !self.initialized {
            return Err(Error::InferenceError("Executor not initialized".into()));
        }
        self.execute_requests(requests, scheduled)
    }

    fn execute_decode(
        &self,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> Result<Vec<ExecutorStepResult>> {
        if !self.initialized {
            return Err(Error::InferenceError("Executor not initialized".into()));
        }
        self.execute_requests(requests, scheduled)
    }

    fn is_ready(&self) -> bool {
        self.initialized
    }

    fn initialize(&mut self) -> Result<()> {
        info!("Initializing native executor");
        if self.config.model_registry.is_none() {
            let device = self.config.backend_context.device.clone();
            let model = Qwen3TtsModel::load(
                &self.config.models_dir,
                device,
                self.config.kv_page_size.max(1),
                &self.config.kv_cache_dtype,
            )?;
            self.loaded_tts_model = Some(Arc::new(model));
            debug!(
                "Native executor loaded TTS model from {:?}",
                self.config.models_dir
            );
        } else {
            debug!("Native executor will use shared model registry");
        }
        self.initialized = true;
        Ok(())
    }

    fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down native executor");
        let mut chat = self
            .chat_decode_states
            .lock()
            .map_err(|_| Error::InferenceError("chat decode state mutex poisoned".to_string()))?;
        let mut asr = self
            .asr_decode_states
            .lock()
            .map_err(|_| Error::InferenceError("ASR decode state mutex poisoned".to_string()))?;
        if self.voxtral_realtime.abort_matching(|_| true).is_err() {
            return Err(Error::InferenceError(
                "failed to abort pending Voxtral realtime state during shutdown".to_string(),
            ));
        }
        if self.voxtral_realtime.has_prepared() {
            return Err(Error::InferenceError(
                "cannot shut down with a prepared Voxtral realtime quantum".to_string(),
            ));
        }
        let mut voxtral = self.voxtral_realtime.states.lock().map_err(|_| {
            Error::InferenceError("Voxtral realtime state mutex poisoned".to_string())
        })?;
        let mut tts = self.qwen_tts_decode_states.lock().map_err(|_| {
            Error::InferenceError("Qwen TTS decode state mutex poisoned".to_string())
        })?;
        chat.clear();
        asr.clear();
        voxtral.clear();
        tts.clear();
        drop((chat, asr, voxtral, tts));
        self.initialized = false;
        self.loaded_tts_model = None;
        Ok(())
    }

    fn cleanup_request(&self, request_id: &str) -> CacheReleaseReport {
        if self
            .voxtral_realtime
            .abort_matching(|pending| pending.session.request_id == request_id)
            .is_err()
        {
            return CacheReleaseReport::unconfirmed();
        }
        let (Ok(mut chat), Ok(mut asr), Ok(mut voxtral), Ok(mut tts)) = (
            self.chat_decode_states.lock(),
            self.asr_decode_states.lock(),
            self.voxtral_realtime.states.lock(),
            self.qwen_tts_decode_states.lock(),
        ) else {
            return CacheReleaseReport::unconfirmed();
        };

        let chat = cleanup_request_states_locked(&mut chat, request_id);
        let asr = cleanup_request_states_locked(&mut asr, request_id);
        let voxtral = cleanup_request_states_locked(&mut voxtral, request_id);
        let tts = cleanup_request_states_locked(&mut tts, request_id);
        cleanup_report(chat.combine(asr).combine(voxtral).combine(tts))
    }

    fn cleanup_session(&self, session: &SessionKey) -> CacheReleaseReport {
        if self
            .voxtral_realtime
            .abort_matching(|pending| &pending.session == session)
            .is_err()
        {
            return CacheReleaseReport::unconfirmed();
        }
        let (Ok(mut chat), Ok(mut asr), Ok(mut voxtral), Ok(mut tts)) = (
            self.chat_decode_states.lock(),
            self.asr_decode_states.lock(),
            self.voxtral_realtime.states.lock(),
            self.qwen_tts_decode_states.lock(),
        ) else {
            return CacheReleaseReport::unconfirmed();
        };

        let chat = cleanup_session_state_locked(&mut chat, session);
        let asr = cleanup_session_state_locked(&mut asr, session);
        let voxtral = cleanup_session_state_locked(&mut voxtral, session);
        let tts = cleanup_session_state_locked(&mut tts, session);
        cleanup_report(chat.combine(asr).combine(voxtral).combine(tts))
    }

    fn purge_model_cache(&self, variant: ModelVariant) -> CacheReleaseReport {
        if self
            .voxtral_realtime
            .abort_matching(|pending| pending.active.variant == variant)
            .is_err()
        {
            return CacheReleaseReport::unconfirmed();
        }
        let Ok(mut states) = self.voxtral_realtime.states.lock() else {
            return CacheReleaseReport::unconfirmed();
        };
        cleanup_report(cleanup_model_states_locked(
            &mut states,
            variant,
            |active| active.variant,
        ))
    }
}

/// Unified executor that wraps a model executor implementation.
#[derive(Clone)]
struct BatchWorkspaceContext {
    backend: BackendKind,
    authority: Arc<ResourceAuthority>,
}

#[derive(Clone)]
pub struct UnifiedExecutor {
    inner: Arc<RwLock<Box<dyn ModelExecutor>>>,
    batch_workspace: Option<BatchWorkspaceContext>,
    physical_execution_admission: Option<PhysicalExecutionAdmission>,
    pending_quantum_finalizer: Option<Arc<dyn PendingQuantumFinalizer>>,
}

/// Opaque proof that one exact physical envelope owns execution capacity.
/// Consuming this token is the only runner path into the native executor.
pub(super) struct AdmittedPhysicalExecution {
    batch_id: BatchId,
    lane: BatchLaneKey,
    width: usize,
    _lease: Option<PhysicalExecutionLease>,
}

pub(super) enum PhysicalExecutionAdmissionOutcome {
    Admitted(AdmittedPhysicalExecution),
    Cancelled,
}

impl UnifiedExecutor {
    /// Create a new unified executor with native backend.
    pub fn new_native(config: WorkerConfig) -> Self {
        let physical_execution_admission = config.physical_execution_admission.clone();
        let batch_workspace =
            config
                .resource_authority
                .as_ref()
                .map(|authority| BatchWorkspaceContext {
                    backend: config.backend,
                    authority: authority.clone(),
                });
        let voxtral_realtime = Arc::new(VoxtralRealtimeStateCoordinator::new());
        Self {
            inner: Arc::new(RwLock::new(Box::new(
                NativeExecutor::with_voxtral_realtime(config, voxtral_realtime.clone()),
            ))),
            batch_workspace,
            physical_execution_admission,
            pending_quantum_finalizer: Some(voxtral_realtime),
        }
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(executor: Box<dyn ModelExecutor>) -> Self {
        Self {
            inner: Arc::new(RwLock::new(executor)),
            batch_workspace: None,
            physical_execution_admission: None,
            pending_quantum_finalizer: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_pending_quantum_finalizer_for_test(
        mut self,
        finalizer: Arc<dyn PendingQuantumFinalizer>,
    ) -> Self {
        self.pending_quantum_finalizer = Some(finalizer);
        self
    }

    #[cfg(test)]
    pub(crate) fn new_for_test_with_physical_context(
        executor: Box<dyn ModelExecutor>,
        backend: BackendKind,
        authority: Arc<ResourceAuthority>,
        admission: PhysicalExecutionAdmission,
    ) -> Self {
        Self {
            inner: Arc::new(RwLock::new(executor)),
            batch_workspace: Some(BatchWorkspaceContext { backend, authority }),
            physical_execution_admission: Some(admission),
            pending_quantum_finalizer: None,
        }
    }

    fn explicit_physical_launch_policy(
        batch: &PhysicalBatch,
        requests: &[&EngineCoreRequest],
    ) -> PhysicalLaunchPolicy {
        if requests.len() != batch.rows.len() {
            record_engine_physical_fallback(EnginePhysicalFallbackReason::BatchIncompatible);
            return PhysicalLaunchPolicy::ExecutionGroupExclusive;
        }
        let mut policy = None;
        for (request, row) in requests.iter().zip(&batch.rows) {
            let Some(binding) = request.execution_adapter_binding() else {
                record_engine_physical_fallback(EnginePhysicalFallbackReason::UncertifiedProfile);
                return PhysicalLaunchPolicy::ExecutionGroupExclusive;
            };
            if binding.execution_group_id != batch.lane.execution_group
                || binding.model_instance_id != batch.lane.model_instance
                || binding.adapter_instance_id != batch.lane.adapter_instance
                || binding.adapter_abi_revision != batch.lane.adapter_abi
                || binding.capability_id != batch.lane.capability_id
            {
                record_engine_physical_fallback(EnginePhysicalFallbackReason::AdapterUnsupported);
                return PhysicalLaunchPolicy::ExecutionGroupExclusive;
            }
            let Ok(stage) = binding.stage_for_work(&row.work) else {
                record_engine_physical_fallback(EnginePhysicalFallbackReason::AdapterUnsupported);
                return PhysicalLaunchPolicy::ExecutionGroupExclusive;
            };
            if stage.id != batch.lane.stage_id || stage.batch_mode != batch.mode {
                record_engine_physical_fallback(EnginePhysicalFallbackReason::BatchIncompatible);
                return PhysicalLaunchPolicy::ExecutionGroupExclusive;
            }
            match policy {
                None => policy = Some(stage.physical_launch_policy),
                Some(active) if active == stage.physical_launch_policy => {}
                Some(_) => {
                    record_engine_physical_fallback(
                        EnginePhysicalFallbackReason::BatchIncompatible,
                    );
                    return PhysicalLaunchPolicy::ExecutionGroupExclusive;
                }
            }
        }
        policy.unwrap_or_else(|| {
            record_engine_physical_fallback(EnginePhysicalFallbackReason::UncertifiedProfile);
            PhysicalLaunchPolicy::ExecutionGroupExclusive
        })
    }

    pub(super) fn reserve_batch_workspace(
        &self,
        batch: &PhysicalBatch,
    ) -> Result<Option<BatchWorkspaceLease>> {
        let workspace_bytes = match batch.workspace.workspace_bytes() {
            Ok(workspace_bytes) => workspace_bytes,
            Err(error) => {
                record_engine_physical_fallback(EnginePhysicalFallbackReason::BatchIncompatible);
                return Err(error);
            }
        };
        if workspace_bytes == 0 {
            return Ok(None);
        }
        let Some(context) = self.batch_workspace.as_ref() else {
            record_engine_physical_defer(EnginePhysicalDeferReason::WorkspaceCapacity);
            return Err(Error::Overloaded(
                "physical batch requires workspace but no resource authority is installed"
                    .to_string(),
            ));
        };
        if batch.lane.backend != context.backend {
            record_engine_physical_fallback(EnginePhysicalFallbackReason::BatchIncompatible);
            return Err(Error::InvalidInput(
                "physical batch workspace backend does not match its executor".to_string(),
            ));
        }
        let reservation = context
            .authority
            .reserve_batch_workspace(batch.lane.execution_group, batch.batch_id, batch.workspace)
            .map(Some);
        if let Err(error) = &reservation {
            match error {
                Error::Overloaded(_) => {
                    record_engine_physical_defer(EnginePhysicalDeferReason::WorkspaceCapacity)
                }
                _ => {
                    record_engine_physical_fallback(EnginePhysicalFallbackReason::BatchIncompatible)
                }
            }
        }
        reservation
    }

    /// Permanently fail physical admission after a panic crossed native/model
    /// entry. Per-model recovery is not available, so both execution capacity
    /// and the shared resource authority fail closed until runtime recreation.
    pub(super) fn poison_physical_runtime(&self, reason: impl Into<String>) {
        let reason = reason.into();
        if let Some(admission) = &self.physical_execution_admission {
            admission.poison(reason.clone());
        }
        if let Some(context) = &self.batch_workspace {
            context.authority.poison(reason);
        }
    }

    /// Acquire physical execution capacity for one validated batch envelope.
    /// Workspace and stream state must not be acquired until this returns.
    pub(super) async fn acquire_physical_execution(
        &self,
        batch: &PhysicalBatch,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> std::result::Result<PhysicalExecutionAdmissionOutcome, PhysicalDispatchError> {
        let width = batch.rows.len().max(1);
        PhysicalBatchExecution {
            batch,
            requests,
            scheduled,
        }
        .validate()
        .map_err(|error| {
            record_engine_physical_fallback(EnginePhysicalFallbackReason::BatchIncompatible);
            PhysicalDispatchError::not_started(error, width, FailureOrigin::ExecutorValidation)
        })?;
        if !requests.is_empty() && requests.iter().all(|request| request.is_cancelled()) {
            return Ok(PhysicalExecutionAdmissionOutcome::Cancelled);
        }
        let lease = if let Some(admission) = &self.physical_execution_admission {
            let launch_policy = Self::explicit_physical_launch_policy(batch, requests);
            let deadline = requests.iter().filter_map(|request| request.deadline).min();
            let acquire = admission.acquire_dispatch(
                batch.lane.execution_group,
                batch.lane.model_instance,
                launch_policy,
                batch.mode,
                width,
                deadline,
            );
            tokio::pin!(acquire);
            let acquired = loop {
                tokio::select! {
                    biased;
                    _ = tokio::time::sleep(PHYSICAL_ADMISSION_CANCELLATION_POLL) => {
                        if !requests.is_empty()
                            && requests.iter().all(|request| request.is_cancelled())
                        {
                            break None;
                        }
                    }
                    acquired = &mut acquire => break Some(acquired),
                }
            };
            let Some(acquired) = acquired else {
                return Ok(PhysicalExecutionAdmissionOutcome::Cancelled);
            };
            match acquired {
                Ok(lease) => Some(lease),
                Err(error) => {
                    return Err(PhysicalDispatchError::not_started(
                        error,
                        width,
                        FailureOrigin::DispatchCoordination,
                    ));
                }
            }
        } else {
            None
        };
        Ok(PhysicalExecutionAdmissionOutcome::Admitted(
            AdmittedPhysicalExecution {
                batch_id: batch.batch_id,
                lane: batch.lane.clone(),
                width,
                _lease: lease,
            },
        ))
    }

    /// Execute a batch whose exact physical capacity has already been admitted.
    pub(super) async fn execute_admitted_physical_batch(
        &self,
        admitted: AdmittedPhysicalExecution,
        batch: &PhysicalBatch,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> PhysicalDispatchResult {
        if admitted.batch_id != batch.batch_id
            || admitted.lane != batch.lane
            || admitted.width != batch.rows.len().max(1)
        {
            return Err(PhysicalDispatchError::not_started(
                Error::InvalidInput(
                    "physical execution admission does not match its batch envelope".to_string(),
                ),
                batch.rows.len().max(1),
                FailureOrigin::ExecutorValidation,
            ));
        }
        if let Some(deadline) = requests.iter().filter_map(|request| request.deadline).min() {
            if deadline <= std::time::Instant::now() {
                return Err(PhysicalDispatchError::not_started(
                    Error::Timeout("physical batch device entry".to_string()),
                    admitted.width,
                    FailureOrigin::DispatchCoordination,
                ));
            }
        }
        if !requests.is_empty() && requests.iter().all(|request| request.is_cancelled()) {
            let dispatch = BatchDispatch::not_dispatched(admitted.width);
            return Ok(scheduled
                .iter()
                .map(|scheduled| {
                    ExecutorStepResult::from_session(
                        scheduled,
                        ModelSessionResult::cancelled_before_dispatch(ExecutorOutput::cancelled(
                            scheduled.request_id.clone(),
                        )),
                    )
                    .with_dispatch(dispatch)
                })
                .collect());
        }
        if let Some(admission) = &self.physical_execution_admission {
            if let Err(error) = admission.ensure_healthy() {
                return Err(PhysicalDispatchError::not_started(
                    error,
                    admitted.width,
                    FailureOrigin::DispatchCoordination,
                ));
            }
        }
        let _physical_dispatch = begin_engine_physical_dispatch();
        let executor = self.inner.read().await;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            executor.execute_physical_batch(PhysicalBatchExecution {
                batch,
                requests,
                scheduled,
            })
        }));
        match result {
            Ok(result) => {
                drop(executor);
                drop(admitted);
                result
            }
            Err(payload) => {
                let message = panic_payload_to_string(payload.as_ref());
                error!(
                    batch_id = batch.batch_id.get(),
                    "Physical model execution panicked after entry: {message}"
                );
                record_engine_physical_fallback(EnginePhysicalFallbackReason::DispatchFailure);
                self.poison_physical_runtime("physical model execution panicked after entry");
                drop(executor);
                drop(admitted);
                Err(PhysicalDispatchError::started(
                    Error::InferenceError(
                        "physical model execution panicked; runtime must be recreated".to_string(),
                    ),
                    batch.expected_dispatch(),
                    FailureOrigin::Panic,
                ))
            }
        }
    }

    /// Compatibility boundary for callers that do not need to interpose
    /// workspace admission between physical admission and device entry.
    pub async fn execute_physical_batch(
        &self,
        batch: &PhysicalBatch,
        requests: &[&EngineCoreRequest],
        scheduled: &[ScheduledRequest],
    ) -> PhysicalDispatchResult {
        let admitted = self
            .acquire_physical_execution(batch, requests, scheduled)
            .await?;
        match admitted {
            PhysicalExecutionAdmissionOutcome::Admitted(admitted) => {
                self.execute_admitted_physical_batch(admitted, batch, requests, scheduled)
                    .await
            }
            PhysicalExecutionAdmissionOutcome::Cancelled => {
                let dispatch = BatchDispatch::not_dispatched(batch.rows.len().max(1));
                Ok(scheduled
                    .iter()
                    .map(|scheduled| {
                        ExecutorStepResult::from_session(
                            scheduled,
                            ModelSessionResult::cancelled_before_dispatch(
                                ExecutorOutput::cancelled(scheduled.request_id.clone()),
                            ),
                        )
                        .with_dispatch(dispatch)
                    })
                    .collect())
            }
        }
    }

    pub async fn execution_profile(&self, request: &EngineCoreRequest) -> Option<ExecutionProfile> {
        let executor = self.inner.read().await;
        executor.execution_profile(request)
    }

    pub(crate) fn has_pending_quantum(&self, plan_id: PlanId, session: &SessionKey) -> bool {
        self.pending_quantum_finalizer
            .as_ref()
            .is_some_and(|finalizer| finalizer.contains(plan_id, session))
    }

    pub(crate) fn finalize_pending_quantum(
        &self,
        plan_id: PlanId,
        session: &SessionKey,
        decision: PendingQuantumDecision,
    ) -> Result<PendingQuantumFinalizeStatus> {
        let Some(finalizer) = self.pending_quantum_finalizer.as_ref() else {
            return Ok(PendingQuantumFinalizeStatus::NotFound);
        };
        let prepared = finalizer.prepare(plan_id, session, decision)?;
        if prepared == PendingQuantumFinalizeStatus::NotFound {
            return Ok(prepared);
        }
        finalizer.publish(plan_id, session)
    }

    pub(crate) fn prepare_pending_quantum(
        &self,
        plan_id: PlanId,
        session: &SessionKey,
        decision: PendingQuantumDecision,
    ) -> Result<PendingQuantumFinalizeStatus> {
        let Some(finalizer) = self.pending_quantum_finalizer.as_ref() else {
            return Ok(PendingQuantumFinalizeStatus::NotFound);
        };
        finalizer.prepare(plan_id, session, decision)
    }

    pub(crate) fn publish_pending_quantum(
        &self,
        plan_id: PlanId,
        session: &SessionKey,
    ) -> Result<PendingQuantumFinalizeStatus> {
        let Some(finalizer) = self.pending_quantum_finalizer.as_ref() else {
            return Ok(PendingQuantumFinalizeStatus::NotFound);
        };
        finalizer.publish(plan_id, session)
    }

    pub(crate) fn discard_prepared_quantum(&self, plan_id: PlanId, session: &SessionKey) {
        if let Some(finalizer) = self.pending_quantum_finalizer.as_ref() {
            finalizer.discard(plan_id, session);
        }
    }

    /// Check if ready.
    pub async fn is_ready(&self) -> bool {
        if self
            .physical_execution_admission
            .as_ref()
            .is_some_and(|admission| admission.ensure_healthy().is_err())
        {
            return false;
        }
        let executor = self.inner.read().await;
        executor.is_ready()
    }

    /// Initialize.
    pub async fn initialize(&self) -> Result<()> {
        let mut executor = self.inner.write().await;
        executor.initialize()
    }

    /// Shutdown.
    pub async fn shutdown(&self) -> Result<()> {
        let mut executor = self.inner.write().await;
        executor.shutdown()
    }

    /// Cleanup transient backend state for a completed/aborted request.
    pub async fn cleanup_request(&self, request_id: &str) -> CacheReleaseReport {
        let executor = self.inner.read().await;
        executor.cleanup_request(request_id)
    }

    pub async fn cleanup_session(&self, session: &SessionKey) -> CacheReleaseReport {
        let executor = self.inner.read().await;
        executor.cleanup_session(session)
    }

    pub async fn purge_model_cache(&self, variant: ModelVariant) -> CacheReleaseReport {
        let executor = self.inner.read().await;
        executor.purge_model_cache(variant)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct StateCleanupSummary {
    released: usize,
    busy: usize,
    unknown: usize,
}

impl StateCleanupSummary {
    fn combine(self, other: Self) -> Self {
        Self {
            released: self.released.saturating_add(other.released),
            busy: self.busy.saturating_add(other.busy),
            unknown: self.unknown.saturating_add(other.unknown),
        }
    }
}

fn cleanup_report(summary: StateCleanupSummary) -> CacheReleaseReport {
    if summary.unknown > 0 {
        CacheReleaseReport::unconfirmed()
    } else if summary.busy > 0 {
        CacheReleaseReport::busy_in_flight(summary.released, summary.busy)
    } else {
        CacheReleaseReport::confirmed(summary.released)
    }
}

fn cleanup_model_states_locked<T>(
    states: &mut HashMap<SessionKey, ExecutorStateSlot<T>>,
    variant: ModelVariant,
    model_variant: impl Fn(&T) -> ModelVariant,
) -> StateCleanupSummary {
    let busy = states
        .values()
        .filter(|state| matches!(state, ExecutorStateSlot::InFlight))
        .count();
    let unknown = states
        .values()
        .filter(|state| matches!(state, ExecutorStateSlot::Poisoned))
        .count();
    let sessions = states
        .iter()
        .filter_map(|(session, state)| match state {
            ExecutorStateSlot::Ready(active) if model_variant(active) == variant => {
                Some(session.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let released = sessions.len();
    for session in sessions {
        states.remove(&session);
    }
    StateCleanupSummary {
        released,
        busy,
        unknown,
    }
}

fn cleanup_request_states_locked<T>(
    states: &mut HashMap<SessionKey, ExecutorStateSlot<T>>,
    request_id: &str,
) -> StateCleanupSummary {
    let mut summary = StateCleanupSummary::default();
    states.retain(|session, slot| {
        if session.request_id != request_id {
            return true;
        }
        match slot {
            ExecutorStateSlot::InFlight => {
                summary.busy = summary.busy.saturating_add(1);
                true
            }
            ExecutorStateSlot::Ready(_) | ExecutorStateSlot::Poisoned => {
                summary.released = summary.released.saturating_add(1);
                false
            }
        }
    });
    summary
}

fn cleanup_session_state_locked<T>(
    states: &mut HashMap<SessionKey, ExecutorStateSlot<T>>,
    session: &SessionKey,
) -> StateCleanupSummary {
    match states.get(session) {
        Some(ExecutorStateSlot::InFlight) => StateCleanupSummary {
            released: 0,
            busy: 1,
            unknown: 0,
        },
        Some(ExecutorStateSlot::Ready(_) | ExecutorStateSlot::Poisoned) => {
            states.remove(session);
            StateCleanupSummary {
                released: 1,
                busy: 0,
                unknown: 0,
            }
        }
        None => StateCleanupSummary::default(),
    }
}

/// Decode base64-encoded audio to samples.
pub fn decode_audio_base64(audio_b64: &str, _sample_rate: u32) -> Result<Vec<f32>> {
    let (samples, _) = decode_audio_base64_with_rate(audio_b64)?;
    Ok(samples)
}

fn decode_audio_base64_with_rate(audio_b64: &str) -> Result<(Vec<f32>, u32)> {
    audio::decode_audio_base64_with_rate(audio_b64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::request::StreamStagingBuffer;
    use crate::engine::{
        CapacitySource, ClockedStateSpan, ManagedCacheDomainReservation, ManagedCacheReservation,
        PhysicalCapacityProvider, PhysicalCapacitySnapshot, ResourceAmount, StageShapePolicy,
    };
    use crate::kv::v2::{StateClock, StateGroupId};
    use crate::model::ModelVariant;
    use base64::Engine;

    struct RecordingPendingFinalizer {
        session: SessionKey,
        calls: Mutex<Vec<(PlanId, PendingQuantumDecision)>>,
        prepared: Mutex<bool>,
    }

    impl PendingQuantumFinalizer for RecordingPendingFinalizer {
        fn contains(&self, plan_id: PlanId, session: &SessionKey) -> bool {
            plan_id == 41 && session == &self.session
        }

        fn prepare(
            &self,
            plan_id: PlanId,
            session: &SessionKey,
            decision: PendingQuantumDecision,
        ) -> Result<PendingQuantumFinalizeStatus> {
            if !self.contains(plan_id, session) {
                return Ok(PendingQuantumFinalizeStatus::NotFound);
            }
            self.calls.lock().unwrap().push((plan_id, decision));
            *self.prepared.lock().unwrap() = true;
            Ok(PendingQuantumFinalizeStatus::Finalized)
        }

        fn publish(
            &self,
            plan_id: PlanId,
            session: &SessionKey,
        ) -> Result<PendingQuantumFinalizeStatus> {
            if plan_id != 41 || session != &self.session || !*self.prepared.lock().unwrap() {
                return Ok(PendingQuantumFinalizeStatus::NotFound);
            }
            *self.prepared.lock().unwrap() = false;
            Ok(PendingQuantumFinalizeStatus::Finalized)
        }

        fn discard(&self, _plan_id: PlanId, _session: &SessionKey) {
            *self.prepared.lock().unwrap() = false;
        }
    }

    #[derive(Debug)]
    struct FixedCapacityProvider {
        capacity: ResourceVector,
    }

    impl PhysicalCapacityProvider for FixedCapacityProvider {
        fn snapshot(&self) -> PhysicalCapacitySnapshot {
            PhysicalCapacitySnapshot {
                capacity: self.capacity,
                available: self.capacity,
                source: CapacitySource::Test,
            }
        }
    }

    struct PanickingPhysicalExecutor;

    impl ModelExecutor for PanickingPhysicalExecutor {
        fn execute_prefill(
            &self,
            _requests: &[&EngineCoreRequest],
            _scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            panic!("physical executor panic sentinel");
        }

        fn execute_decode(
            &self,
            _requests: &[&EngineCoreRequest],
            _scheduled: &[ScheduledRequest],
        ) -> Result<Vec<ExecutorStepResult>> {
            panic!("physical executor panic sentinel");
        }

        fn is_ready(&self) -> bool {
            true
        }

        fn initialize(&mut self) -> Result<()> {
            Ok(())
        }

        fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn unified_pending_quantum_finalizer_is_exact_and_model_neutral() {
        let session = SessionKey::new("pending-finalizer".to_string(), 9);
        let finalizer = Arc::new(RecordingPendingFinalizer {
            session: session.clone(),
            calls: Mutex::new(Vec::new()),
            prepared: Mutex::new(false),
        });
        let executor = UnifiedExecutor::new_for_test(Box::new(PanickingPhysicalExecutor))
            .with_pending_quantum_finalizer_for_test(finalizer.clone());

        assert!(executor.has_pending_quantum(41, &session));
        assert!(!executor.has_pending_quantum(41, &SessionKey::new(session.request_id.clone(), 10)));
        assert_eq!(
            executor
                .finalize_pending_quantum(41, &session, PendingQuantumDecision::Abort)
                .unwrap(),
            PendingQuantumFinalizeStatus::Finalized
        );
        assert_eq!(
            finalizer.calls.lock().unwrap().as_slice(),
            &[(41, PendingQuantumDecision::Abort)]
        );
    }

    #[test]
    fn executor_state_lease_keeps_in_flight_ownership_visible_to_cleanup() {
        let session = SessionKey::new("visible-in-flight".to_string(), 7);
        let store = Mutex::new(HashMap::from([(
            session.clone(),
            ExecutorStateSlot::Ready("ready".to_string()),
        )]));

        let lease = ExecutorStateLease::checkout(&store, session.clone(), "test state").unwrap();
        assert_eq!(lease.state().map(String::as_str), Some("ready"));

        let summary = {
            let mut states = store.lock().unwrap();
            cleanup_session_state_locked(&mut states, &session)
        };
        assert_eq!(
            summary,
            StateCleanupSummary {
                released: 0,
                busy: 1,
                unknown: 0,
            }
        );
        drop(lease);

        let states = store.lock().unwrap();
        assert!(matches!(
            states.get(&session),
            Some(ExecutorStateSlot::Ready(state)) if state == "ready"
        ));
    }

    #[test]
    fn dirty_executor_state_unwind_is_poisoned_until_cleanup() {
        let session = SessionKey::new("poison-on-unwind".to_string(), 3);
        let store = Mutex::new(HashMap::from([(
            session.clone(),
            ExecutorStateSlot::Ready(41usize),
        )]));

        let mut lease =
            ExecutorStateLease::checkout(&store, session.clone(), "test state").unwrap();
        lease.mark_dirty();
        *lease.require_state_mut().unwrap() = 42;
        drop(lease);

        assert!(matches!(
            store.lock().unwrap().get(&session),
            Some(ExecutorStateSlot::Poisoned)
        ));
        assert!(ExecutorStateLease::checkout(&store, session.clone(), "test state").is_err());

        let summary = {
            let mut states = store.lock().unwrap();
            cleanup_session_state_locked(&mut states, &session)
        };
        assert_eq!(
            summary,
            StateCleanupSummary {
                released: 1,
                busy: 0,
                unknown: 0,
            }
        );
        assert!(!store.lock().unwrap().contains_key(&session));
    }

    #[test]
    fn model_purge_never_confirms_unknown_poisoned_ownership() {
        let session = SessionKey::new("poisoned-model-purge".to_string(), 1);
        let mut states = HashMap::<SessionKey, ExecutorStateSlot<ModelVariant>>::from([(
            session,
            ExecutorStateSlot::Poisoned,
        )]);

        let summary = cleanup_model_states_locked(
            &mut states,
            ModelVariant::VoxtralMini4BRealtime2602,
            |variant| *variant,
        );
        assert_eq!(summary.unknown, 1);
        assert_eq!(
            cleanup_report(summary).outcome,
            CacheReleaseOutcome::Unconfirmed
        );
    }

    #[test]
    fn executor_state_lease_explicitly_restores_or_releases() {
        let session = SessionKey::new("explicit-transition".to_string(), 2);
        let store = Mutex::new(HashMap::new());

        let mut lease =
            ExecutorStateLease::checkout(&store, session.clone(), "test state").unwrap();
        lease.install_state(9usize).unwrap();
        lease.mark_dirty();
        lease.restore().unwrap();
        assert!(matches!(
            store.lock().unwrap().get(&session),
            Some(ExecutorStateSlot::Ready(9))
        ));

        let lease = ExecutorStateLease::checkout(&store, session.clone(), "test state").unwrap();
        lease.release().unwrap();
        assert!(!store.lock().unwrap().contains_key(&session));
    }

    #[test]
    fn native_cleanup_reports_busy_chat_tts_and_asr_sessions() {
        let executor = NativeExecutor::new(WorkerConfig::default());
        let session = SessionKey::new("all-modalities-in-flight".to_string(), 11);
        executor
            .chat_decode_states
            .lock()
            .unwrap()
            .insert(session.clone(), ExecutorStateSlot::InFlight);
        executor
            .qwen_tts_decode_states
            .lock()
            .unwrap()
            .insert(session.clone(), ExecutorStateSlot::InFlight);
        executor
            .asr_decode_states
            .lock()
            .unwrap()
            .insert(session.clone(), ExecutorStateSlot::InFlight);

        let report = executor.cleanup_session(&session);
        assert_eq!(report.outcome, CacheReleaseOutcome::BusyInFlight);
        assert!(!report.confirmed);
        assert_eq!(report.released_sessions, 0);
        assert_eq!(report.busy_sessions, 3);
    }

    fn qwen38_test_arena(generation: u32) -> KvArenaId {
        KvArenaId {
            model_instance: super::super::ModelInstanceId::new(38),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            generation,
        }
    }

    #[test]
    fn only_multi_token_solo_decode_uses_the_scalar_continuous_route() {
        let scheduled = |num_tokens: usize, is_prefill: bool| ScheduledRequest {
            plan_id: 1,
            request_id: "route".to_string(),
            sequence_id: 1,
            num_tokens,
            is_prefill,
            num_computed_tokens: 0,
            work: WorkUnit::SequenceStep {
                phase: if is_prefill {
                    crate::engine::SequencePhase::Prefill
                } else {
                    crate::engine::SequencePhase::Decode
                },
                input: crate::engine::InputRange {
                    start: 0,
                    end: num_tokens,
                },
                max_output_steps: num_tokens,
                auxiliary_state: None,
            },
        };

        assert!(is_isolated_continuous_model_quantum(&[scheduled(4, false)]));
        assert!(!is_isolated_continuous_model_quantum(&[scheduled(
            1, false
        )]));
        assert!(!is_isolated_continuous_model_quantum(&[scheduled(4, true)]));
        assert!(!is_isolated_continuous_model_quantum(&[
            scheduled(1, false),
            scheduled(1, false),
        ]));
    }

    fn native_route_fixture(
        task: TaskType,
        capability: &str,
        variant: ModelVariant,
        selector: StageWorkSelector,
        stage_name: &str,
        mode: NativeBatchMode,
        phase: SequencePhase,
    ) -> (EngineCoreRequest, ScheduledRequest, PhysicalBatch) {
        let mut profile =
            ExecutionProfile::fail_closed(BackendKind::Cpu, Some(variant), ExecutionMode::Sequence);
        profile.max_batch_size = 4;
        let mut stage = super::super::StageDescriptor::from_execution_profile(
            StageId::new(7),
            stage_name,
            &profile,
            mode,
        );
        stage.selector = selector;
        stage.max_work_units = 4;
        stage.validate().unwrap();
        let binding = super::super::ExecutionAdapterBinding {
            execution_group_id: super::super::ExecutionGroupId::new(1),
            model_instance_id: super::super::ModelInstanceId::new(2),
            adapter_instance_id: super::super::AdapterInstanceId::new(3),
            adapter_abi_revision: super::super::AdapterAbiRevision::new(4),
            model_variant: variant,
            capability_id: capability.to_string(),
            stages: Arc::from([stage.clone()]),
        };
        binding.validate().unwrap();

        let mut request = match task {
            TaskType::TTS => EngineCoreRequest::tts("native route"),
            TaskType::ASR => EngineCoreRequest::asr(""),
            TaskType::Chat => EngineCoreRequest::chat(Vec::new()),
            TaskType::SpeechToSpeech => EngineCoreRequest::speech_to_speech(""),
        };
        request.id = "native-route".to_string();
        request.model_variant = Some(variant);
        request.bind_execution_adapter(binding).unwrap();
        let scheduled = ScheduledRequest {
            plan_id: 11,
            request_id: request.id.clone(),
            sequence_id: 11,
            num_tokens: 1,
            is_prefill: phase == SequencePhase::Prefill,
            num_computed_tokens: if phase == SequencePhase::Decode { 1 } else { 0 },
            work: WorkUnit::SequenceStep {
                phase,
                input: if phase == SequencePhase::Prefill {
                    super::super::InputRange { start: 0, end: 1 }
                } else {
                    super::super::InputRange { start: 1, end: 2 }
                },
                max_output_steps: 1,
                auxiliary_state: None,
            },
        };
        let lane = BatchLaneKey {
            execution_group: super::super::ExecutionGroupId::new(1),
            model_instance: super::super::ModelInstanceId::new(2),
            adapter_instance: super::super::AdapterInstanceId::new(3),
            adapter_abi: super::super::AdapterAbiRevision::new(4),
            capability_id: capability.to_string(),
            stage_id: StageId::new(7),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            compute_dtype: "f32".to_string(),
            state_dtype: "f32".to_string(),
            tensor_layout: format!("{:?}", stage.shape_policy).to_ascii_lowercase(),
            quantization: "none".to_string(),
            state_schema: "test".to_string(),
            kernel_mode: stage_name.to_string(),
            semantic_mode: format!("{phase:?}").to_ascii_lowercase(),
            shape_bucket: if stage.shape_policy == StageShapePolicy::Padded {
                "padded".to_string()
            } else {
                "ragged".to_string()
            },
        };
        let batch = PhysicalBatch {
            batch_id: BatchId::new(12),
            lane: lane.clone(),
            mode,
            budget: super::super::BatchBudget {
                max_rows: 4,
                max_logical_units: 4,
                max_tensor_elements: 4,
                max_workspace_bytes: 0,
                max_padding_basis_points: stage.max_padding_basis_points,
                max_formation_delay: Duration::ZERO,
            },
            rows: vec![super::super::ReadyQuantum {
                plan_id: scheduled.plan_id,
                session: scheduled.session_key(),
                lane,
                work: scheduled.work.clone(),
                cost: super::super::WorkCost::new(1, 1, 0),
                managed_cache: None,
            }],
            materialized_tensor_elements: 1,
            workspace: ResourceVector::zero(),
        };
        batch.validate().unwrap();
        (request, scheduled, batch)
    }

    #[test]
    fn native_route_uses_exact_loaded_audio_stage_identity() {
        let (request, scheduled, batch) = native_route_fixture(
            TaskType::TTS,
            "tts",
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            StageWorkSelector::SequenceDecode,
            "tts.decode.tensor_continuous",
            NativeBatchMode::Continuous,
            SequencePhase::Decode,
        );
        let requests = [&request];
        let scheduled = [scheduled];
        let execution = PhysicalBatchExecution {
            batch: &batch,
            requests: &requests,
            scheduled: &scheduled,
        };
        execution.validate().unwrap();
        assert_eq!(
            NativeBatchRoute::resolve(&execution).unwrap(),
            NativeBatchRoute::Audio {
                task: TaskType::TTS,
                stage: NativeAudioStage::SequenceDecode,
                mode: NativeBatchMode::Continuous,
                stage_id: StageId::new(7),
            }
        );
    }

    #[test]
    fn native_audio_stage_authenticates_realtime_push_and_finish_roles() {
        assert_eq!(
            NativeBatchRoute::audio_stage(&WorkUnit::RealtimePush {
                operation_id: super::super::RealtimeOperationId::new(1),
                input: super::super::InputRange::new(0, 160).unwrap(),
                max_output_steps: 2,
                max_cache_append: 4,
            }),
            NativeAudioStage::RealtimePush
        );
        assert_eq!(
            NativeBatchRoute::audio_stage(&WorkUnit::RealtimeFinish {
                operation_id: super::super::RealtimeOperationId::new(2),
                max_output_steps: 4,
                max_cache_append: 4,
            }),
            NativeAudioStage::RealtimeFinish
        );
    }

    #[test]
    fn native_route_authenticates_vibevoice_static_prefill() {
        let (request, scheduled, batch) = native_route_fixture(
            TaskType::ASR,
            "asr",
            ModelVariant::VibeVoiceAsr,
            StageWorkSelector::SequencePrefill,
            crate::models::architectures::vibevoice::VIBEVOICE_ASR_PREFILL_STAGE,
            NativeBatchMode::Static,
            SequencePhase::Prefill,
        );
        let requests = [&request];
        let scheduled = [scheduled];
        assert_eq!(
            NativeBatchRoute::resolve(&PhysicalBatchExecution {
                batch: &batch,
                requests: &requests,
                scheduled: &scheduled,
            })
            .unwrap(),
            NativeBatchRoute::Audio {
                task: TaskType::ASR,
                stage: NativeAudioStage::SequencePrefill,
                mode: NativeBatchMode::Static,
                stage_id: StageId::new(7),
            }
        );
    }

    #[test]
    fn vibevoice_static_prefill_requires_two_selected_audio_rows() {
        let (_, mut scheduled, _) = native_route_fixture(
            TaskType::ASR,
            "asr",
            ModelVariant::VibeVoiceAsr,
            StageWorkSelector::SequencePrefill,
            crate::models::architectures::vibevoice::VIBEVOICE_ASR_PREFILL_STAGE,
            NativeBatchMode::Static,
            SequencePhase::Prefill,
        );
        assert!(!has_native_vibevoice_tokenizer_batch(&[
            scheduled.clone(),
            scheduled.clone(),
        ]));
        let WorkUnit::SequenceStep {
            auxiliary_state, ..
        } = &mut scheduled.work
        else {
            unreachable!()
        };
        *auxiliary_state = Some(Arc::from([ClockedStateSpan::new(
            StateGroupId::new(2),
            StateClock::AudioSamples,
            super::super::InputRange::new(0, 3_200).unwrap(),
        )
        .unwrap()]));
        assert!(!has_native_vibevoice_tokenizer_batch(&[scheduled.clone()]));
        assert!(has_native_vibevoice_tokenizer_batch(&[
            scheduled.clone(),
            scheduled,
        ]));
    }

    #[test]
    fn native_route_preserves_chat_decode_and_rejects_fallback_stage_proof() {
        let (request, scheduled, batch) = native_route_fixture(
            TaskType::Chat,
            "chat",
            ModelVariant::Qwen306B,
            StageWorkSelector::SequenceDecode,
            "chat.decode.tensor_continuous",
            NativeBatchMode::Continuous,
            SequencePhase::Decode,
        );
        let requests = [&request];
        let scheduled = [scheduled];
        let execution = PhysicalBatchExecution {
            batch: &batch,
            requests: &requests,
            scheduled: &scheduled,
        };
        assert_eq!(
            NativeBatchRoute::resolve(&execution).unwrap(),
            NativeBatchRoute::ChatContinuousDecode {
                stage_id: StageId::new(7)
            }
        );

        let (request, scheduled, batch) = native_route_fixture(
            TaskType::TTS,
            "tts",
            ModelVariant::Qwen3Tts12Hz06BCustomVoice,
            StageWorkSelector::Any,
            "tts.compatibility",
            NativeBatchMode::Continuous,
            SequencePhase::Decode,
        );
        let requests = [&request];
        let scheduled = [scheduled];
        assert!(NativeBatchRoute::resolve(&PhysicalBatchExecution {
            batch: &batch,
            requests: &requests,
            scheduled: &scheduled,
        })
        .is_err());
    }

    fn qwen38_test_reservation(domains: &[(CacheDomainId, KvArenaId)]) -> ManagedCacheReservation {
        ManagedCacheReservation {
            txn_id: 1,
            session: SessionKey::new("qwen38-domain-selection".into(), 1),
            session_generation: crate::engine::ManagedSessionGeneration::INITIAL,
            domains: domains
                .iter()
                .map(|(domain, arena)| ManagedCacheDomainReservation {
                    arena: *arena,
                    domain: *domain,
                    expected_version: 0,
                    expected_committed_tokens: 0,
                    execution_start_tokens: 0,
                    target_committed_tokens: 1,
                    target_window_start: 0,
                    first_page_offset: 0,
                    provisional_groups: Vec::new(),
                    writable_blocks: Vec::new(),
                })
                .collect(),
            clocked_state: None,
            allow_unchanged_prefix: false,
        }
    }

    #[test]
    fn qwen38_managed_group_selection_resolves_exact_target_and_optional_mtp() {
        let target_arena = qwen38_test_arena(1);
        let mtp_arena = qwen38_test_arena(2);
        let target_group = KvGroupId::new(1);
        let mtp_group = KvGroupId::new(1);
        let dual_groups = [
            (QWEN38_MTP_ATTENTION_DOMAIN, mtp_group, mtp_arena),
            (QWEN38_TARGET_ATTENTION_DOMAIN, target_group, target_arena),
        ];
        let dual_reservation = qwen38_test_reservation(&[
            (QWEN38_TARGET_ATTENTION_DOMAIN, target_arena),
            (QWEN38_MTP_ATTENTION_DOMAIN, mtp_arena),
        ]);
        assert_eq!(
            qwen38_managed_group_ids(&dual_groups, &dual_reservation).unwrap(),
            (target_group, Some(mtp_group))
        );

        let target_groups = [(QWEN38_TARGET_ATTENTION_DOMAIN, target_group, target_arena)];
        let target_reservation =
            qwen38_test_reservation(&[(QWEN38_TARGET_ATTENTION_DOMAIN, target_arena)]);
        assert_eq!(
            qwen38_managed_group_ids(&target_groups, &target_reservation).unwrap(),
            (target_group, None)
        );
    }

    #[test]
    fn qwen38_managed_group_selection_rejects_half_resolved_mtp_domain() {
        let target_arena = qwen38_test_arena(1);
        let mtp_arena = qwen38_test_arena(2);
        let groups = [
            (
                QWEN38_TARGET_ATTENTION_DOMAIN,
                KvGroupId::new(1),
                target_arena,
            ),
            (QWEN38_MTP_ATTENTION_DOMAIN, KvGroupId::new(1), mtp_arena),
        ];
        let target_only =
            qwen38_test_reservation(&[(QWEN38_TARGET_ATTENTION_DOMAIN, target_arena)]);
        assert!(qwen38_managed_group_ids(&groups, &target_only).is_err());

        let target_group_only = &groups[..1];
        let reservation_with_mtp = qwen38_test_reservation(&[
            (QWEN38_TARGET_ATTENTION_DOMAIN, target_arena),
            (QWEN38_MTP_ATTENTION_DOMAIN, mtp_arena),
        ]);
        assert!(qwen38_managed_group_ids(target_group_only, &reservation_with_mtp).is_err());
    }

    #[test]
    fn qwen38_managed_group_selection_rejects_missing_duplicate_or_foreign_target() {
        let target_arena = qwen38_test_arena(1);
        let foreign_arena = qwen38_test_arena(2);
        let empty_reservation = qwen38_test_reservation(&[]);
        assert!(qwen38_managed_group_ids(&[], &empty_reservation).is_err());

        let target_reservation =
            qwen38_test_reservation(&[(QWEN38_TARGET_ATTENTION_DOMAIN, target_arena)]);
        assert!(qwen38_managed_group_ids(&[], &target_reservation).is_err());

        let duplicate_groups = [
            (
                QWEN38_TARGET_ATTENTION_DOMAIN,
                KvGroupId::new(1),
                target_arena,
            ),
            (
                QWEN38_TARGET_ATTENTION_DOMAIN,
                KvGroupId::new(2),
                target_arena,
            ),
        ];
        assert!(qwen38_managed_group_ids(&duplicate_groups, &target_reservation).is_err());

        let foreign_groups = [(
            QWEN38_TARGET_ATTENTION_DOMAIN,
            KvGroupId::new(1),
            foreign_arena,
        )];
        assert!(qwen38_managed_group_ids(&foreign_groups, &target_reservation).is_err());
    }

    #[test]
    fn test_worker_config_default() {
        let config = WorkerConfig::default();
        assert_eq!(config.backend, config.backend_context.backend_kind);
        assert_eq!(
            config.request_parallelism,
            WorkerConfig::resolve_request_parallelism(
                config.backend,
                config.num_threads,
                WorkerConfig::available_cpu_capacity(),
                WorkerConfig::request_parallelism_override(),
            )
        );
    }

    #[test]
    fn atomic_invocation_leases_reject_non_atomic_or_tensor_stages() {
        let profile = ExecutionProfile::fail_closed(BackendKind::Cpu, None, ExecutionMode::Atomic);
        let scalar = super::super::StageDescriptor::from_execution_profile(
            super::super::StageId::new(1),
            "atomic.scalar",
            &profile,
            NativeBatchMode::None,
        );
        let atomic = WorkUnit::AtomicJob {
            kind: "test".to_string(),
        };
        validate_atomic_scalar_invocation_stage(&scalar, &atomic).unwrap();

        let sequence = WorkUnit::SequenceStep {
            phase: super::super::SequencePhase::Decode,
            input: super::super::InputRange { start: 0, end: 1 },
            max_output_steps: 1,
            auxiliary_state: None,
        };
        assert!(validate_atomic_scalar_invocation_stage(&scalar, &sequence).is_err());

        let tensor = super::super::StageDescriptor::from_execution_profile(
            super::super::StageId::new(1),
            "atomic.tensor",
            &profile,
            NativeBatchMode::Static,
        );
        assert!(validate_atomic_scalar_invocation_stage(&tensor, &atomic).is_err());
    }

    #[test]
    fn test_worker_config_from_engine_config_uses_backend_context() {
        let engine = EngineCoreConfig {
            backend: BackendKind::Cpu,
            ..Default::default()
        };

        let config = WorkerConfig::from(&engine);
        assert_eq!(config.backend, config.backend_context.backend_kind);
        assert_eq!(
            config.request_parallelism,
            WorkerConfig::resolve_request_parallelism(
                BackendKind::Cpu,
                engine.num_threads,
                WorkerConfig::available_cpu_capacity(),
                WorkerConfig::request_parallelism_override(),
            )
        );
        assert_eq!(
            config.backend_context.source,
            BackendSelectionSource::Config
        );
    }

    #[test]
    fn standalone_execution_capacity_obeys_physical_rollout_mode() {
        let mut engine = EngineCoreConfig {
            backend: BackendKind::Cpu,
            max_physical_in_flight: crate::config::PhysicalInFlightLimit::new(4).unwrap(),
            ..Default::default()
        };

        for mode in [
            crate::config::PhysicalExecutionMode::Serial,
            crate::config::PhysicalExecutionMode::Shadow,
        ] {
            engine.physical_execution_mode = mode;
            let worker = WorkerConfig::from(&engine);
            assert_eq!(
                worker
                    .physical_execution_admission
                    .as_ref()
                    .unwrap()
                    .capacity(),
                1
            );
        }

        engine.physical_execution_mode = crate::config::PhysicalExecutionMode::Concurrent;
        let worker = WorkerConfig::from(&engine);
        assert_eq!(
            worker
                .physical_execution_admission
                .as_ref()
                .unwrap()
                .capacity(),
            worker.request_parallelism.max(4)
        );

        assert_eq!(
            WorkerConfig::physical_execution_capacity(
                crate::config::PhysicalExecutionMode::Concurrent,
                BackendKind::Cuda,
                4,
                4,
            ),
            1,
            "CUDA remains device-serialized until a backend certificate exists"
        );
    }

    #[test]
    fn cpu_request_parallelism_uses_available_capacity_and_intra_op_threads() {
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Cpu, 8, 8, None),
            1
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Cpu, 8, 16, None),
            2
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Cpu, 8, 32, None),
            MAX_CPU_REQUEST_PARALLELISM
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Cpu, 1, 128, None),
            MAX_CPU_REQUEST_PARALLELISM
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Cpu, 64, 8, None),
            1
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Cpu, 0, 0, None),
            1
        );
    }

    #[test]
    fn request_parallelism_override_precedes_auto_with_backend_clamps() {
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Cpu, 8, 32, Some(3)),
            3
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Cpu, 8, 32, Some(usize::MAX)),
            MAX_CPU_REQUEST_PARALLELISM
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Cpu, 1, 2, Some(3)),
            2
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Metal, 1, 32, Some(3)),
            1
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Cuda, 1, 32, None),
            1
        );
        assert_eq!(
            WorkerConfig::resolve_request_parallelism(BackendKind::Cuda, 1, 32, Some(4)),
            4
        );
    }

    #[test]
    fn automatic_tensor_width_follows_resolved_backend_and_not_scheduler_rows() {
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let engine = EngineCoreConfig {
                backend,
                max_batch_size: 19,
                ..EngineCoreConfig::default()
            };
            let worker = WorkerConfig::from(&engine);
            assert_eq!(
                worker.max_tensor_batch_size,
                engine.max_tensor_batch_size.resolve(worker.backend)
            );
            assert_eq!(
                worker.request_parallelism,
                WorkerConfig::resolve_request_parallelism(
                    backend,
                    engine.num_threads,
                    WorkerConfig::available_cpu_capacity(),
                    WorkerConfig::request_parallelism_override(),
                )
            );
            assert_eq!(engine.max_batch_size, 19);
        }
    }

    #[test]
    fn tensor_batch_caps_are_backend_conservative() {
        assert_eq!(WorkerConfig::tensor_batch_cap(BackendKind::Cpu), 2);
        assert_eq!(WorkerConfig::tensor_batch_cap(BackendKind::Metal), 2);
        assert_eq!(WorkerConfig::tensor_batch_cap(BackendKind::Cuda), 32);
    }

    #[test]
    fn physical_batch_workspace_uses_the_backend_resource_domain_and_releases() {
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let mut capacity = ResourceVector::zero();
            match backend {
                BackendKind::Cpu => capacity.host_bytes = ResourceAmount::Known(64),
                BackendKind::Metal => capacity.unified_bytes = ResourceAmount::Known(64),
                BackendKind::Cuda => {
                    capacity.host_bytes = ResourceAmount::Known(64);
                    capacity.device_bytes = ResourceAmount::Known(64);
                }
            }
            let authority = Arc::new(ResourceAuthority::new(Arc::new(FixedCapacityProvider {
                capacity,
            })));
            let mut executor = UnifiedExecutor::new_for_test(Box::new(NativeExecutor::new(
                WorkerConfig::default(),
            )));
            executor.batch_workspace = Some(BatchWorkspaceContext {
                backend,
                authority: authority.clone(),
            });
            let lane = super::super::BatchLaneKey {
                execution_group: super::super::ExecutionGroupId::new(7),
                model_instance: super::super::ModelInstanceId::new(8),
                adapter_instance: super::super::AdapterInstanceId::new(9),
                adapter_abi: super::super::AdapterAbiRevision::new(1),
                capability_id: "test".to_string(),
                stage_id: super::super::StageId::new(1),
                backend,
                device_ordinal: None,
                compute_dtype: "f32".to_string(),
                state_dtype: "f32".to_string(),
                tensor_layout: "exact".to_string(),
                quantization: "none".to_string(),
                state_schema: "none".to_string(),
                kernel_mode: "test".to_string(),
                semantic_mode: "test".to_string(),
                shape_bucket: "exact.1".to_string(),
            };
            let expected_workspace = match backend {
                BackendKind::Cpu => ResourceVector {
                    host_bytes: ResourceAmount::Known(8),
                    ..ResourceVector::zero()
                },
                BackendKind::Metal => ResourceVector {
                    unified_bytes: ResourceAmount::Known(8),
                    ..ResourceVector::zero()
                },
                BackendKind::Cuda => ResourceVector {
                    host_bytes: ResourceAmount::Known(3),
                    device_bytes: ResourceAmount::Known(8),
                    ..ResourceVector::zero()
                },
            };
            let batch = PhysicalBatch {
                batch_id: super::super::BatchId::new(10),
                lane: lane.clone(),
                mode: NativeBatchMode::None,
                budget: super::super::BatchBudget::width_one(),
                rows: vec![super::super::ReadyQuantum {
                    plan_id: 1,
                    session: SessionKey::new("workspace".to_string(), 1),
                    lane,
                    work: super::super::WorkUnit::AtomicJob {
                        kind: "test".to_string(),
                    },
                    cost: super::super::WorkCost::new(1, 1, 8),
                    managed_cache: None,
                }],
                materialized_tensor_elements: 1,
                workspace: expected_workspace,
            };

            let workspace = executor
                .reserve_batch_workspace(&batch)
                .unwrap()
                .expect("workspace lease");
            assert_eq!(workspace.resources(), expected_workspace);
            assert_eq!(authority.snapshot().reservations, 1);
            drop(workspace);
            assert_eq!(authority.snapshot().reservations, 0);
        }
    }

    #[test]
    fn uncertified_physical_policy_fallback_is_observed_in_production_derivation() {
        let before = super::super::metrics::engine_physical_execution_metrics_snapshot();
        let request = EngineCoreRequest::tts("uncertified physical policy");
        let lane = super::super::BatchLaneKey {
            execution_group: super::super::ExecutionGroupId::new(17),
            model_instance: super::super::ModelInstanceId::new(18),
            adapter_instance: super::super::AdapterInstanceId::new(19),
            adapter_abi: super::super::AdapterAbiRevision::new(1),
            capability_id: "test".to_string(),
            stage_id: super::super::StageId::new(1),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            compute_dtype: "f32".to_string(),
            state_dtype: "f32".to_string(),
            tensor_layout: "exact".to_string(),
            quantization: "none".to_string(),
            state_schema: "none".to_string(),
            kernel_mode: "test".to_string(),
            semantic_mode: "test".to_string(),
            shape_bucket: "exact.1".to_string(),
        };
        let batch = PhysicalBatch {
            batch_id: super::super::BatchId::new(20),
            lane: lane.clone(),
            mode: NativeBatchMode::None,
            budget: super::super::BatchBudget::width_one(),
            rows: vec![super::super::ReadyQuantum {
                plan_id: 1,
                session: SessionKey::new(request.id.clone(), 1),
                lane,
                work: super::super::WorkUnit::AtomicJob {
                    kind: "test".to_string(),
                },
                cost: super::super::WorkCost::new(1, 1, 0),
                managed_cache: None,
            }],
            materialized_tensor_elements: 1,
            workspace: ResourceVector::zero(),
        };

        assert_eq!(
            UnifiedExecutor::explicit_physical_launch_policy(&batch, &[&request]),
            PhysicalLaunchPolicy::ExecutionGroupExclusive
        );
        let after = super::super::metrics::engine_physical_execution_metrics_snapshot();
        assert!(
            after.fallbacks.uncertified_profile
                >= before.fallbacks.uncertified_profile.saturating_add(1)
        );
    }

    #[test]
    fn test_run_blocking_propagates_panic_to_physical_boundary() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            NativeExecutor::run_blocking(|| -> Result<()> {
                panic!("executor panic sentinel");
            })
        }));

        let payload = result.expect_err("model panic must cross the native boundary");
        assert!(panic_payload_to_string(payload.as_ref()).contains("executor panic sentinel"));
    }

    #[tokio::test]
    async fn post_entry_panic_poisons_physical_runtime_and_releases_admission() {
        let authority = Arc::new(ResourceAuthority::new(Arc::new(FixedCapacityProvider {
            capacity: ResourceVector {
                host_bytes: ResourceAmount::Known(64),
                ..ResourceVector::zero()
            },
        })));
        let admission = PhysicalExecutionAdmission::standalone(1);
        let executor = UnifiedExecutor::new_for_test_with_physical_context(
            Box::new(PanickingPhysicalExecutor),
            BackendKind::Cpu,
            authority.clone(),
            admission.clone(),
        );
        let mut request = EngineCoreRequest::tts("panic containment");
        request.id = "panic-containment".to_string();
        let lane = super::super::BatchLaneKey {
            execution_group: super::super::ExecutionGroupId::new(27),
            model_instance: super::super::ModelInstanceId::new(28),
            adapter_instance: super::super::AdapterInstanceId::new(29),
            adapter_abi: super::super::AdapterAbiRevision::new(1),
            capability_id: "panic-test".to_string(),
            stage_id: super::super::StageId::new(1),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            compute_dtype: "f32".to_string(),
            state_dtype: "f32".to_string(),
            tensor_layout: "exact".to_string(),
            quantization: "none".to_string(),
            state_schema: "none".to_string(),
            kernel_mode: "test".to_string(),
            semantic_mode: "test".to_string(),
            shape_bucket: "exact.1".to_string(),
        };
        let work = super::super::WorkUnit::AtomicJob {
            kind: "panic-test".to_string(),
        };
        let batch = PhysicalBatch {
            batch_id: super::super::BatchId::new(30),
            lane: lane.clone(),
            mode: NativeBatchMode::None,
            budget: super::super::BatchBudget::width_one(),
            rows: vec![super::super::ReadyQuantum {
                plan_id: 1,
                session: SessionKey::new(request.id.clone(), 1),
                lane: lane.clone(),
                work: work.clone(),
                cost: super::super::WorkCost::new(1, 1, 0),
                managed_cache: None,
            }],
            materialized_tensor_elements: 1,
            workspace: ResourceVector::zero(),
        };
        let scheduled = ScheduledRequest {
            plan_id: 1,
            request_id: request.id.clone(),
            sequence_id: 1,
            num_tokens: 1,
            is_prefill: true,
            num_computed_tokens: 0,
            work,
        };

        let failure = executor
            .execute_physical_batch(&batch, &[&request], &[scheduled])
            .await
            .expect_err("physical panic must become a typed dispatch failure");
        assert_eq!(failure.provenance.dispatch_state, DispatchState::Started);
        assert_eq!(
            failure.provenance.failure_origin,
            Some(FailureOrigin::Panic)
        );
        assert_eq!(admission.active(), 0);
        assert_eq!(
            admission.poison_reason().as_deref(),
            Some("physical model execution panicked after entry")
        );
        assert_eq!(
            authority.poison_reason().as_deref(),
            Some("physical model execution panicked after entry")
        );
        assert!(!executor.is_ready().await);
        assert!(matches!(
            admission
                .acquire_dispatch(
                    lane.execution_group,
                    lane.model_instance,
                    PhysicalLaunchPolicy::ExecutionGroupExclusive,
                    NativeBatchMode::None,
                    1,
                    None,
                )
                .await,
            Err(Error::Overloaded(message)) if message.contains("poisoned")
        ));
    }

    #[test]
    fn test_run_blocking_is_safe_inside_current_thread_runtime() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build runtime");

        let result =
            runtime.block_on(async { NativeExecutor::run_blocking(|| Ok::<_, Error>(())) });
        assert!(result.is_ok());
    }

    #[test]
    fn test_stream_audio_stages_inside_current_thread_runtime() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to build runtime");

        let result = runtime.block_on(async {
            let tx = StreamStagingBuffer::default();
            let mut sequence = 0usize;
            NativeExecutor::stream_audio(
                &tx,
                "req-1",
                &mut sequence,
                vec![0.1, -0.1],
                24_000,
                false,
            )?;
            let chunk = tx
                .take()?
                .into_iter()
                .next()
                .ok_or_else(|| Error::InferenceError("missing staged chunk".to_string()))?;
            if chunk.request_id != "req-1" || chunk.sequence != 0 || chunk.samples.len() != 2 {
                return Err(Error::InferenceError(
                    "unexpected streamed chunk payload".to_string(),
                ));
            }
            Ok::<(), Error>(())
        });
        assert!(result.is_ok());
    }

    #[test]
    fn test_to_tts_params_uses_model_native_auto_limit() {
        let mut request = EngineCoreRequest::tts("Long-form synthesis");
        request.model_variant = Some(ModelVariant::Qwen3Tts12Hz17BVoiceDesign);
        request.params.max_tokens = 0;

        let params = NativeExecutor::to_tts_params(&request);
        assert_eq!(params.max_frames, ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES);
    }

    #[test]
    fn test_to_tts_params_clamps_to_model_native_limit() {
        let mut request = EngineCoreRequest::tts("Long-form synthesis");
        request.model_variant = Some(ModelVariant::Qwen3Tts12Hz06BCustomVoice);
        request.params.max_tokens = 50_000;

        let params = NativeExecutor::to_tts_params(&request);
        assert_eq!(params.max_frames, ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES);
    }

    #[test]
    fn resumable_prefill_classification_is_opt_in_and_fail_closed() {
        let mut request = EngineCoreRequest::chat(vec![crate::models::shared::chat::ChatMessage {
            role: crate::models::shared::chat::ChatRole::User,
            content: "chunk me".to_string(),
        }]);
        request.model_variant = Some(ModelVariant::Qwen3827BFp8);
        request.streaming = true;

        let default_executor = NativeExecutor::new(WorkerConfig::default());
        assert_eq!(
            default_executor
                .execution_profile(&request)
                .unwrap()
                .prefill,
            PrefillMode::Full
        );

        let chunking_executor = NativeExecutor::new(WorkerConfig {
            enable_chunked_prefill: true,
            ..Default::default()
        });
        assert_eq!(
            chunking_executor
                .execution_profile(&request)
                .unwrap()
                .prefill,
            PrefillMode::Full
        );

        // Sibling hybrid without a resumable prefill path stays full.
        request.model_variant = Some(ModelVariant::Qwen3508BGguf);
        assert_eq!(
            chunking_executor
                .execution_profile(&request)
                .unwrap()
                .prefill,
            PrefillMode::Full
        );
    }

    #[test]
    fn resumable_prefill_mode_requires_exact_positive_model_proof() {
        assert_eq!(
            resolved_resumable_prefill_mode(true, Some(true)),
            PrefillMode::Incremental
        );
        assert_eq!(
            resolved_resumable_prefill_mode(false, Some(true)),
            PrefillMode::Full
        );
        assert_eq!(
            resolved_resumable_prefill_mode(true, Some(false)),
            PrefillMode::Full
        );
        assert_eq!(
            resolved_resumable_prefill_mode(true, None),
            PrefillMode::Full
        );
    }

    #[test]
    fn qwen_asr_raw_profile_uses_the_exact_prepared_execution_route() {
        let executor = NativeExecutor::new(WorkerConfig {
            backend: BackendKind::Cpu,
            request_parallelism: 4,
            enable_chunked_prefill: true,
            ..Default::default()
        });
        let mut normal = EngineCoreRequest::asr_bytes(vec![1, 2, 3]);
        normal.model_variant = Some(ModelVariant::Qwen3Asr06BGguf);
        normal
            .install_prepared_asr_audio(ModelVariant::Qwen3Asr06BGguf, vec![0.0; 160], 16_000)
            .unwrap();
        normal
            .install_prepared_sequence_input_tokens(32, 4096)
            .unwrap();
        let normal_profile = executor.execution_profile(&normal).unwrap();
        assert_eq!(normal_profile.mode, ExecutionMode::Sequence);
        assert!(normal_profile.incremental_decode);

        let mut long = EngineCoreRequest::asr_bytes(vec![1, 2, 3]);
        long.model_variant = Some(ModelVariant::Qwen3Asr06BGguf);
        long.install_prepared_asr_audio(ModelVariant::Qwen3Asr06BGguf, vec![0.0; 160], 16_000)
            .unwrap();
        long.install_prepared_asr_long_form_atomic().unwrap();
        let long_profile = executor.execution_profile(&long).unwrap();
        assert_eq!(long_profile.mode, ExecutionMode::Atomic);
        assert_eq!(long_profile.prefill, PrefillMode::None);
        assert!(!long_profile.incremental_decode);
        assert_eq!(long_profile.cache_mode, CacheMode::None);
        assert_eq!(long_profile.decode_batch, NativeBatchMode::None);
        assert_eq!(long_profile.concurrency, ConcurrencyClass::Exclusive);
        assert_eq!(long_profile.max_batch_size, 1);
    }

    #[test]
    fn unloaded_models_cannot_claim_native_batch_capability() {
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            let config = WorkerConfig {
                backend,
                request_parallelism: 4,
                ..Default::default()
            };
            let executor = NativeExecutor::new(config);
            let mut request = EngineCoreRequest::tts("batch me");
            request.model_variant = Some(ModelVariant::Qwen3Tts12Hz06BCustomVoice);

            let profile = executor.execution_profile(&request).unwrap();
            assert_eq!(profile.backend, backend);
            assert_eq!(profile.mode, ExecutionMode::Sequence);
            assert_eq!(profile.prefill, PrefillMode::Full);
            assert!(!profile.capabilities().native_batch);
            assert_eq!(profile.decode_batch, NativeBatchMode::None);
            let expected_parallelism = if backend == BackendKind::Metal { 1 } else { 4 };
            assert_eq!(profile.max_batch_size, expected_parallelism);
            assert_eq!(
                profile.concurrency,
                if expected_parallelism > 1 {
                    ConcurrencyClass::Batchable
                } else {
                    ConcurrencyClass::Exclusive
                }
            );
            request.streaming = true;
            assert!(!executor.execution_capabilities(&request).native_batch);
            request.streaming = false;
            request.reference_audio = Some("reference".to_string());
            assert!(!executor.execution_capabilities(&request).native_batch);
        }
    }

    #[test]
    fn model_session_results_declare_safe_points_and_terminal_semantics() {
        let sequence = ModelSessionResult::sequence(ExecutorOutput {
            request_id: "sequence".to_string(),
            audio: None,
            text: None,
            input_transcription: None,
            tokens_processed: 1,
            tokens_generated: 1,
            finished: false,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        });
        assert_eq!(
            sequence.disposition,
            ExecutionDisposition::Yielded(YieldReason::QuantumExhausted)
        );
        assert!(sequence.safe_point);

        let atomic = ModelSessionResult::atomic(ExecutorOutput {
            request_id: "atomic".to_string(),
            audio: None,
            text: None,
            input_transcription: None,
            tokens_processed: 0,
            tokens_generated: 0,
            finished: false,
            phase_timing_override: None,
            asr_diagnostics: None,
            error: None,
        });
        assert!(matches!(
            atomic.disposition,
            ExecutionDisposition::Failed(ExecutionFailure {
                kind: FailureKind::InvalidOutput,
                ..
            })
        ));
        assert!(atomic.output.finished);

        let cancelled =
            ModelSessionResult::cancelled(ExecutorOutput::cancelled("cancelled".to_string()));
        assert_eq!(
            cancelled.disposition,
            ExecutionDisposition::Finished(FinishReason::Cancelled)
        );
        assert!(cancelled.output.error.is_none());
    }

    #[test]
    fn decode_audio_base64_with_rate_downmixes_stereo_wav() {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut wav_bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut wav_bytes);
            let mut writer = hound::WavWriter::new(cursor, spec).expect("writer");
            // 2 stereo frames: [L,R]=[0.25,0.75] then [0.5,-0.5]
            writer.write_sample((0.25f32 * 32767.0) as i16).unwrap();
            writer.write_sample((0.75f32 * 32767.0) as i16).unwrap();
            writer.write_sample((0.5f32 * 32767.0) as i16).unwrap();
            writer.write_sample((-0.5f32 * 32767.0) as i16).unwrap();
            writer.finalize().unwrap();
        }

        let b64 = base64::engine::general_purpose::STANDARD.encode(&wav_bytes);
        let (samples, sample_rate) =
            decode_audio_base64_with_rate(&b64).expect("decode should succeed");

        assert_eq!(sample_rate, 16_000);
        assert_eq!(samples.len(), 2);
        // After downmixing, expected mono values are averages: 0.5 and 0.0.
        assert!(
            (samples[0] - 0.5).abs() < 0.02,
            "first sample was {}",
            samples[0]
        );
        assert!(samples[1].abs() < 0.02, "second sample was {}", samples[1]);
    }

    #[test]
    fn decode_request_audio_with_rate_accepts_raw_audio_bytes() {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };

        let mut wav_bytes = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut wav_bytes);
            let mut writer = hound::WavWriter::new(cursor, spec).expect("writer");
            writer.write_sample((0.25f32 * 32767.0) as i16).unwrap();
            writer.write_sample((-0.25f32 * 32767.0) as i16).unwrap();
            writer.finalize().unwrap();
        }

        let request = EngineCoreRequest::asr_bytes(wav_bytes);
        let (samples, sample_rate) =
            audio::decode_request_audio_with_rate(&request).expect("decode should succeed");

        assert_eq!(sample_rate, 16_000);
        assert_eq!(samples.len(), 2);
        assert!(
            (samples[0] - 0.25).abs() < 0.02,
            "first sample was {}",
            samples[0]
        );
        assert!(
            (samples[1] + 0.25).abs() < 0.02,
            "second sample was {}",
            samples[1]
        );
    }
}
