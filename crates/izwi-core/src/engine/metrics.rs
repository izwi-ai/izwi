//! Metrics and benchmarking infrastructure for the inference engine.
//!
//! Provides detailed performance tracking including:
//! - Request latency histograms
//! - Throughput measurements
//! - Real-time factor (RTF) tracking
//! - KV cache utilization
//! - Queue depth monitoring

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use super::{
    BatchDispatch, BatchDispatchKind, DeadlinePhase, DispatchState, FailureOrigin,
    OutcomeProvenance, PhysicalBatch, ResourceAmount,
};

/// Stable metric names for scheduler and KV-cache observability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct EngineMetricDescriptor {
    pub name: &'static str,
    pub description: &'static str,
}

pub const ENGINE_SCHEDULER_QUEUE_DEPTH: &str = "engine.scheduler.queue_depth";
pub const ENGINE_SCHEDULER_RUNNING_REQUESTS: &str = "engine.scheduler.running_requests";
pub const ENGINE_SCHEDULER_STEP_TOKENS_TOTAL: &str = "engine.scheduler.step_tokens_total";
pub const ENGINE_KV_CACHE_HITS_TOTAL: &str = "engine.kv_cache.hits_total";
pub const ENGINE_KV_CACHE_MISSES_TOTAL: &str = "engine.kv_cache.misses_total";
pub const ENGINE_KV_CACHE_EVICTIONS_TOTAL: &str = "engine.kv_cache.evictions_total";
pub const ENGINE_KV_CACHE_ALLOCATED_BLOCKS: &str = "engine.kv_cache.allocated_blocks";
pub const ENGINE_KV_CACHE_FREE_BLOCKS: &str = "engine.kv_cache.free_blocks";
pub const ENGINE_KV_CACHE_UTILIZATION_RATIO: &str = "engine.kv_cache.utilization_ratio";
pub const ENGINE_KV_CACHE_MEMORY_USED_BYTES: &str = "engine.kv_cache.memory_used_bytes";
pub const ENGINE_KV_CACHE_MEMORY_CAPACITY_BYTES: &str = "engine.kv_cache.memory_capacity_bytes";
pub const ENGINE_KV_CACHE_GPU_RESIDENT_BLOCKS: &str = "engine.kv_cache.gpu_resident_blocks";
pub const ENGINE_STREAM_BACKPRESSURE_TOTAL: &str = "engine.stream.backpressure_total";
pub const ENGINE_STREAM_CHECKPOINTS_COMMITTED_TOTAL: &str =
    "engine.stream.checkpoints_committed_total";
pub const ENGINE_STREAM_CHECKPOINT_REJECTIONS_TOTAL: &str =
    "engine.stream.checkpoint_rejections_total";
pub const ENGINE_STREAM_DELIVERY_FAILURES_TOTAL: &str = "engine.stream.delivery_failures_total";
pub const ENGINE_EXECUTOR_TENSOR_BATCHES_TOTAL: &str = "engine.executor.tensor_batches_total";
pub const ENGINE_EXECUTOR_REQUEST_PARALLEL_BATCHES_TOTAL: &str =
    "engine.executor.request_parallel_batches_total";
pub const ENGINE_EXECUTOR_TENSOR_BATCH_MAX_WIDTH: &str = "engine.executor.tensor_batch_max_width";
pub const ENGINE_EXECUTOR_TENSOR_STATIC_BATCHES_TOTAL: &str =
    "engine.executor.tensor_static_batches_total";
pub const ENGINE_EXECUTOR_TENSOR_CONTINUOUS_BATCHES_TOTAL: &str =
    "engine.executor.tensor_continuous_batches_total";
pub const ENGINE_EXECUTOR_TENSOR_CONTINUOUS_MULTIROW_BATCHES_TOTAL: &str =
    "engine.executor.tensor_continuous_multirow_batches_total";
pub const ENGINE_EXECUTOR_PHYSICAL_BATCH_REJECTIONS_TOTAL: &str =
    "engine.executor.physical_batch_rejections_total";
pub const ENGINE_EXECUTOR_TENSOR_BATCH_ROWS_TOTAL: &str = "engine.executor.tensor_batch_rows_total";
pub const ENGINE_EXECUTOR_TENSOR_BATCH_CAPACITY_ROWS_TOTAL: &str =
    "engine.executor.tensor_batch_capacity_rows_total";
pub const ENGINE_EXECUTOR_TENSOR_BATCH_USEFUL_ELEMENTS_TOTAL: &str =
    "engine.executor.tensor_batch_useful_elements_total";
pub const ENGINE_EXECUTOR_TENSOR_BATCH_MATERIALIZED_ELEMENTS_TOTAL: &str =
    "engine.executor.tensor_batch_materialized_elements_total";
pub const ENGINE_EXECUTOR_BATCH_WORKSPACE_BYTES_TOTAL: &str =
    "engine.executor.batch_workspace_bytes_total";
pub const ENGINE_EXECUTOR_DISPATCH_STATE_ROWS_TOTAL: &str =
    "engine.executor.dispatch_state_rows_total";
pub const ENGINE_EXECUTOR_FAILURE_ORIGIN_ROWS_TOTAL: &str =
    "engine.executor.failure_origin_rows_total";
pub const ENGINE_EXECUTOR_DEADLINE_PHASE_ROWS_TOTAL: &str =
    "engine.executor.deadline_phase_rows_total";
pub const ENGINE_EXECUTOR_BATCH_WORKSPACE_DOMAIN_BYTES_TOTAL: &str =
    "engine.executor.batch_workspace_domain_bytes_total";
pub const ENGINE_EXECUTOR_TENSOR_BATCH_FILL_RATIO: &str = "engine.executor.tensor_batch_fill_ratio";
pub const ENGINE_EXECUTOR_TENSOR_BATCH_PADDING_RATIO: &str =
    "engine.executor.tensor_batch_padding_ratio";
pub const ENGINE_EXECUTOR_MODEL_TENSOR_BATCHES_TOTAL: &str =
    "engine.executor.model_tensor_batches_total";
pub const ENGINE_EXECUTOR_MODEL_TENSOR_BATCH_ROWS_TOTAL: &str =
    "engine.executor.model_tensor_batch_rows_total";
pub const ENGINE_EXECUTOR_MODEL_TENSOR_BATCH_MAX_WIDTH: &str =
    "engine.executor.model_tensor_batch_max_width";
pub const ENGINE_EXECUTOR_MODEL_SCALAR_ROW_DISPATCHES_TOTAL: &str =
    "engine.executor.model_scalar_row_dispatches_total";
pub const ENGINE_EXECUTOR_MODEL_DECODE_CALLS_TOTAL: &str =
    "engine.executor.model_decode_calls_total";
pub const ENGINE_EXECUTOR_MODEL_TENSOR_MULTIROW_CALLS_TOTAL: &str =
    "engine.executor.model_tensor_multirow_calls_total";
pub const ENGINE_EXECUTOR_CONTINUOUS_ENVELOPE_SCALAR_FALLBACKS_TOTAL: &str =
    "engine.executor.continuous_envelope_scalar_fallbacks_total";

pub const ENGINE_METRIC_CATALOG: &[EngineMetricDescriptor] = &[
    EngineMetricDescriptor {
        name: ENGINE_SCHEDULER_QUEUE_DEPTH,
        description: "Requests waiting in the scheduler queue.",
    },
    EngineMetricDescriptor {
        name: ENGINE_SCHEDULER_RUNNING_REQUESTS,
        description: "Requests currently running in the scheduler.",
    },
    EngineMetricDescriptor {
        name: ENGINE_SCHEDULER_STEP_TOKENS_TOTAL,
        description: "Tokens admitted into scheduler execution steps.",
    },
    EngineMetricDescriptor {
        name: ENGINE_KV_CACHE_HITS_TOTAL,
        description: "Managed prefix-cache lookups that reused at least one physical KV page.",
    },
    EngineMetricDescriptor {
        name: ENGINE_KV_CACHE_MISSES_TOTAL,
        description: "Managed prefix-cache lookups that reused no physical KV pages.",
    },
    EngineMetricDescriptor {
        name: ENGINE_KV_CACHE_EVICTIONS_TOTAL,
        description: "KV-cache evictions labeled by reason when emitted.",
    },
    EngineMetricDescriptor {
        name: ENGINE_KV_CACHE_ALLOCATED_BLOCKS,
        description: "Currently allocated physical KV-cache pages (legacy metric name retained).",
    },
    EngineMetricDescriptor {
        name: ENGINE_KV_CACHE_FREE_BLOCKS,
        description: "Currently free physical KV-cache pages (legacy metric name retained).",
    },
    EngineMetricDescriptor {
        name: ENGINE_KV_CACHE_UTILIZATION_RATIO,
        description: "Physical KV-cache page utilization ratio.",
    },
    EngineMetricDescriptor {
        name: ENGINE_KV_CACHE_MEMORY_USED_BYTES,
        description: "Physical bytes owned by currently allocated managed KV pages.",
    },
    EngineMetricDescriptor {
        name: ENGINE_KV_CACHE_MEMORY_CAPACITY_BYTES,
        description: "Resident managed KV pages plus authorized retained tensor-state bytes.",
    },
    EngineMetricDescriptor {
        name: ENGINE_KV_CACHE_GPU_RESIDENT_BLOCKS,
        description:
            "Allocated physical KV pages in Metal or CUDA arenas (legacy metric name retained).",
    },
    EngineMetricDescriptor {
        name: ENGINE_STREAM_BACKPRESSURE_TOTAL,
        description: "Engine stream backpressure events.",
    },
    EngineMetricDescriptor {
        name: ENGINE_STREAM_CHECKPOINTS_COMMITTED_TOTAL,
        description: "Incremental stream checkpoints accepted by exact engine transaction fences.",
    },
    EngineMetricDescriptor {
        name: ENGINE_STREAM_CHECKPOINT_REJECTIONS_TOTAL,
        description: "Incremental stream checkpoints rejected by lifecycle or protocol validation.",
    },
    EngineMetricDescriptor {
        name: ENGINE_STREAM_DELIVERY_FAILURES_TOTAL,
        description: "Committed stream outboxes that could not be delivered to their consumer.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_TENSOR_BATCHES_TOTAL,
        description: "Legacy physical batch envelopes declared native-batch by their adapter; use model_tensor_batches_total for proven tensor forwards.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_REQUEST_PARALLEL_BATCHES_TOTAL,
        description: "Observed thread-parallel request groups; these are not tensor batches.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_TENSOR_BATCH_MAX_WIDTH,
        description: "Largest legacy native-batch envelope width; use model_tensor_batch_max_width for proven tensor forwards.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_TENSOR_STATIC_BATCHES_TOTAL,
        description: "Observed static model-native tensor batch dispatches.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_TENSOR_CONTINUOUS_BATCHES_TOTAL,
        description: "Observed continuous model-native tensor batch dispatches.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_TENSOR_CONTINUOUS_MULTIROW_BATCHES_TOTAL,
        description: "Observed continuous tensor batches containing at least two rows.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_PHYSICAL_BATCH_REJECTIONS_TOTAL,
        description: "Physical batches rejected before entering model code.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_TENSOR_BATCH_ROWS_TOTAL,
        description: "Rows dispatched through model-native tensor batches.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_TENSOR_BATCH_CAPACITY_ROWS_TOTAL,
        description: "Configured row capacity of dispatched model-native tensor batches.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_TENSOR_BATCH_USEFUL_ELEMENTS_TOTAL,
        description: "Useful tensor elements dispatched through model-native batches.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_TENSOR_BATCH_MATERIALIZED_ELEMENTS_TOTAL,
        description: "Materialized tensor elements, including padding, in model-native batches.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_BATCH_WORKSPACE_BYTES_TOTAL,
        description: "Transient workspace bytes admitted for dispatched physical batches.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_DISPATCH_STATE_ROWS_TOTAL,
        description: "Execution rows by bounded dispatch-state label.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_FAILURE_ORIGIN_ROWS_TOTAL,
        description: "Failed execution rows by bounded failure-origin label.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_DEADLINE_PHASE_ROWS_TOTAL,
        description: "Timed-out execution rows by bounded deadline-phase label.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_BATCH_WORKSPACE_DOMAIN_BYTES_TOTAL,
        description: "Transient physical-batch workspace bytes by bounded memory-domain label.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_TENSOR_BATCH_FILL_RATIO,
        description: "Cumulative tensor-batch row utilization against configured capacity.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_TENSOR_BATCH_PADDING_RATIO,
        description: "Cumulative padded tensor elements as a fraction of materialized elements.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_MODEL_TENSOR_BATCHES_TOTAL,
        description: "Model calls proven to execute their live rows in one tensor-batched forward path.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_MODEL_TENSOR_BATCH_ROWS_TOTAL,
        description: "Rows executed by proven tensor-batched model forward paths.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_MODEL_TENSOR_BATCH_MAX_WIDTH,
        description: "Largest live row width observed in a proven tensor-batched model forward path.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_MODEL_SCALAR_ROW_DISPATCHES_TOTAL,
        description: "Rows executed through scalar continuous-decode model paths.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_MODEL_DECODE_CALLS_TOTAL,
        description: "Successful shape-valid continuous decode model call-path invocations.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_MODEL_TENSOR_MULTIROW_CALLS_TOTAL,
        description: "Proven tensor-batched model call paths that executed at least two live rows.",
    },
    EngineMetricDescriptor {
        name: ENGINE_EXECUTOR_CONTINUOUS_ENVELOPE_SCALAR_FALLBACKS_TOTAL,
        description: "Continuous physical envelopes executed through scalar model call paths.",
    },
];

static ENGINE_STREAM_BACKPRESSURE_EVENTS: AtomicU64 = AtomicU64::new(0);
static ENGINE_STREAM_CHECKPOINTS_COMMITTED: AtomicU64 = AtomicU64::new(0);
static ENGINE_STREAM_CHECKPOINT_REJECTIONS: AtomicU64 = AtomicU64::new(0);
static ENGINE_STREAM_DELIVERY_FAILURES: AtomicU64 = AtomicU64::new(0);
static ENGINE_TENSOR_BATCHES: AtomicU64 = AtomicU64::new(0);
static ENGINE_REQUEST_PARALLEL_BATCHES: AtomicU64 = AtomicU64::new(0);
static ENGINE_TENSOR_BATCH_MAX_WIDTH: AtomicU64 = AtomicU64::new(0);
static ENGINE_TENSOR_STATIC_BATCHES: AtomicU64 = AtomicU64::new(0);
static ENGINE_TENSOR_CONTINUOUS_BATCHES: AtomicU64 = AtomicU64::new(0);
static ENGINE_TENSOR_CONTINUOUS_MULTIROW_BATCHES: AtomicU64 = AtomicU64::new(0);
static ENGINE_PHYSICAL_BATCH_REJECTIONS: AtomicU64 = AtomicU64::new(0);
static ENGINE_TENSOR_BATCH_ROWS: AtomicU64 = AtomicU64::new(0);
static ENGINE_TENSOR_BATCH_CAPACITY_ROWS: AtomicU64 = AtomicU64::new(0);
static ENGINE_TENSOR_BATCH_USEFUL_ELEMENTS: AtomicU64 = AtomicU64::new(0);
static ENGINE_TENSOR_BATCH_MATERIALIZED_ELEMENTS: AtomicU64 = AtomicU64::new(0);
static ENGINE_BATCH_WORKSPACE_BYTES: AtomicU64 = AtomicU64::new(0);
static ENGINE_MODEL_TENSOR_BATCHES: AtomicU64 = AtomicU64::new(0);
static ENGINE_MODEL_TENSOR_BATCH_ROWS: AtomicU64 = AtomicU64::new(0);
static ENGINE_MODEL_TENSOR_BATCH_MAX_WIDTH: AtomicU64 = AtomicU64::new(0);
static ENGINE_MODEL_SCALAR_ROW_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static ENGINE_MODEL_DECODE_CALLS: AtomicU64 = AtomicU64::new(0);
static ENGINE_MODEL_TENSOR_MULTIROW_CALLS: AtomicU64 = AtomicU64::new(0);
static ENGINE_CONTINUOUS_ENVELOPE_SCALAR_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static ENGINE_DISPATCH_STATE_ROWS: [AtomicU64; 3] =
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static ENGINE_FAILURE_ORIGIN_ROWS: [AtomicU64; 9] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static ENGINE_DEADLINE_PHASE_ROWS: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];
static ENGINE_WORKSPACE_DOMAIN_BYTES: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct EngineDispatchStateMetricsSnapshot {
    pub not_started: u64,
    pub started: u64,
    pub produced_output: u64,
}

impl EngineDispatchStateMetricsSnapshot {
    pub fn labeled_values(self) -> [(&'static str, u64); 3] {
        [
            ("not_started", self.not_started),
            ("started", self.started),
            ("produced_output", self.produced_output),
        ]
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct EngineFailureOriginMetricsSnapshot {
    pub adapter_planning: u64,
    pub dispatch_coordination: u64,
    pub workspace_admission: u64,
    pub executor_validation: u64,
    pub model: u64,
    pub stream_delivery: u64,
    pub state_commit: u64,
    pub cleanup: u64,
    pub panic: u64,
}

impl EngineFailureOriginMetricsSnapshot {
    pub fn labeled_values(self) -> [(&'static str, u64); 9] {
        [
            ("adapter_planning", self.adapter_planning),
            ("dispatch_coordination", self.dispatch_coordination),
            ("workspace_admission", self.workspace_admission),
            ("executor_validation", self.executor_validation),
            ("model", self.model),
            ("stream_delivery", self.stream_delivery),
            ("state_commit", self.state_commit),
            ("cleanup", self.cleanup),
            ("panic", self.panic),
        ]
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct EngineDeadlinePhaseMetricsSnapshot {
    pub scheduler_queue: u64,
    pub dispatch_wait: u64,
    pub model_execution: u64,
    pub stream_delivery: u64,
    pub terminal_delivery: u64,
}

impl EngineDeadlinePhaseMetricsSnapshot {
    pub fn labeled_values(self) -> [(&'static str, u64); 5] {
        [
            ("scheduler_queue", self.scheduler_queue),
            ("dispatch_wait", self.dispatch_wait),
            ("model_execution", self.model_execution),
            ("stream_delivery", self.stream_delivery),
            ("terminal_delivery", self.terminal_delivery),
        ]
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct EngineWorkspaceDomainMetricsSnapshot {
    pub host: u64,
    pub device: u64,
    pub unified: u64,
    pub temporary: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct EngineStreamMetricsSnapshot {
    pub backpressure_total: u64,
    pub checkpoints_committed_total: u64,
    pub checkpoint_rejections_total: u64,
    pub delivery_failures_total: u64,
}

impl EngineWorkspaceDomainMetricsSnapshot {
    pub fn labeled_values(self) -> [(&'static str, u64); 4] {
        [
            ("host", self.host),
            ("device", self.device),
            ("unified", self.unified),
            ("temporary", self.temporary),
        ]
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EngineBatchMetricsSnapshot {
    pub tensor_batches_total: u64,
    pub tensor_static_batches_total: u64,
    pub tensor_continuous_batches_total: u64,
    pub tensor_continuous_multirow_batches_total: u64,
    pub request_parallel_batches_total: u64,
    pub physical_batch_rejections_total: u64,
    pub tensor_batch_max_width: u64,
    pub tensor_batch_rows_total: u64,
    pub tensor_batch_capacity_rows_total: u64,
    pub tensor_batch_useful_elements_total: u64,
    pub tensor_batch_materialized_elements_total: u64,
    pub batch_workspace_bytes_total: u64,
    pub dispatch_states: EngineDispatchStateMetricsSnapshot,
    pub failure_origins: EngineFailureOriginMetricsSnapshot,
    pub deadline_phases: EngineDeadlinePhaseMetricsSnapshot,
    pub workspace_domains: EngineWorkspaceDomainMetricsSnapshot,
    pub tensor_batch_fill_ratio: f64,
    pub tensor_batch_padding_ratio: f64,
    pub model_tensor_batches_total: u64,
    pub model_tensor_batch_rows_total: u64,
    pub model_tensor_batch_max_width: u64,
    pub model_scalar_row_dispatches_total: u64,
    pub model_decode_calls_total: u64,
    pub model_tensor_multirow_calls_total: u64,
    pub continuous_envelope_scalar_fallbacks_total: u64,
}

pub fn engine_metric_catalog() -> &'static [EngineMetricDescriptor] {
    ENGINE_METRIC_CATALOG
}

pub(crate) fn record_engine_stream_backpressure() {
    ENGINE_STREAM_BACKPRESSURE_EVENTS.fetch_add(1, Ordering::Relaxed);
}

pub fn engine_stream_backpressure_total() -> u64 {
    ENGINE_STREAM_BACKPRESSURE_EVENTS.load(Ordering::Relaxed)
}

pub(crate) fn record_engine_stream_checkpoint_committed() {
    ENGINE_STREAM_CHECKPOINTS_COMMITTED.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_engine_stream_checkpoint_rejection() {
    ENGINE_STREAM_CHECKPOINT_REJECTIONS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_engine_stream_delivery_failure() {
    ENGINE_STREAM_DELIVERY_FAILURES.fetch_add(1, Ordering::Relaxed);
}

pub fn engine_stream_metrics_snapshot() -> EngineStreamMetricsSnapshot {
    EngineStreamMetricsSnapshot {
        backpressure_total: engine_stream_backpressure_total(),
        checkpoints_committed_total: ENGINE_STREAM_CHECKPOINTS_COMMITTED.load(Ordering::Relaxed),
        checkpoint_rejections_total: ENGINE_STREAM_CHECKPOINT_REJECTIONS.load(Ordering::Relaxed),
        delivery_failures_total: ENGINE_STREAM_DELIVERY_FAILURES.load(Ordering::Relaxed),
    }
}

pub(crate) fn record_engine_execution_outcome(provenance: OutcomeProvenance) {
    let dispatch_index = match provenance.dispatch_state {
        DispatchState::NotStarted => 0,
        DispatchState::Started => 1,
        DispatchState::ProducedOutput => 2,
    };
    ENGINE_DISPATCH_STATE_ROWS[dispatch_index].fetch_add(1, Ordering::Relaxed);

    if let Some(origin) = provenance.failure_origin {
        let origin_index = match origin {
            FailureOrigin::AdapterPlanning => 0,
            FailureOrigin::DispatchCoordination => 1,
            FailureOrigin::WorkspaceAdmission => 2,
            FailureOrigin::ExecutorValidation => 3,
            FailureOrigin::Model => 4,
            FailureOrigin::StreamDelivery => 5,
            FailureOrigin::StateCommit => 6,
            FailureOrigin::Cleanup => 7,
            FailureOrigin::Panic => 8,
        };
        ENGINE_FAILURE_ORIGIN_ROWS[origin_index].fetch_add(1, Ordering::Relaxed);
    }

    if let Some(phase) = provenance.deadline_phase {
        let phase_index = match phase {
            DeadlinePhase::SchedulerQueue => 0,
            DeadlinePhase::DispatchWait => 1,
            DeadlinePhase::ModelExecution => 2,
            DeadlinePhase::StreamDelivery => 3,
            DeadlinePhase::TerminalDelivery => 4,
        };
        ENGINE_DEADLINE_PHASE_ROWS[phase_index].fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) fn record_engine_batch_dispatch(dispatch: BatchDispatch) {
    match dispatch.kind {
        BatchDispatchKind::TensorStatic => {
            ENGINE_TENSOR_BATCHES.fetch_add(1, Ordering::Relaxed);
            ENGINE_TENSOR_STATIC_BATCHES.fetch_add(1, Ordering::Relaxed);
            ENGINE_TENSOR_BATCH_MAX_WIDTH.fetch_max(dispatch.width as u64, Ordering::Relaxed);
        }
        BatchDispatchKind::TensorContinuous => {
            ENGINE_TENSOR_BATCHES.fetch_add(1, Ordering::Relaxed);
            ENGINE_TENSOR_CONTINUOUS_BATCHES.fetch_add(1, Ordering::Relaxed);
            ENGINE_TENSOR_BATCH_MAX_WIDTH.fetch_max(dispatch.width as u64, Ordering::Relaxed);
        }
        BatchDispatchKind::RequestParallel => {
            ENGINE_REQUEST_PARALLEL_BATCHES.fetch_add(1, Ordering::Relaxed);
        }
        BatchDispatchKind::Serial | BatchDispatchKind::NotDispatched => {}
    }
}

pub(crate) fn record_engine_chat_model_dispatch(tensor_batched: bool, live_rows: usize) {
    let live_rows = live_rows.max(1) as u64;
    ENGINE_MODEL_DECODE_CALLS.fetch_add(1, Ordering::Relaxed);
    if tensor_batched {
        ENGINE_MODEL_TENSOR_BATCHES.fetch_add(1, Ordering::Relaxed);
        ENGINE_MODEL_TENSOR_BATCH_ROWS.fetch_add(live_rows, Ordering::Relaxed);
        ENGINE_MODEL_TENSOR_BATCH_MAX_WIDTH.fetch_max(live_rows, Ordering::Relaxed);
        if live_rows >= 2 {
            ENGINE_MODEL_TENSOR_MULTIROW_CALLS.fetch_add(1, Ordering::Relaxed);
        }
    } else {
        ENGINE_MODEL_SCALAR_ROW_DISPATCHES.fetch_add(live_rows, Ordering::Relaxed);
        ENGINE_CONTINUOUS_ENVELOPE_SCALAR_FALLBACKS.fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) fn record_engine_physical_batch(batch: &PhysicalBatch, dispatch: BatchDispatch) {
    if dispatch.kind == BatchDispatchKind::NotDispatched {
        ENGINE_PHYSICAL_BATCH_REJECTIONS.fetch_add(1, Ordering::Relaxed);
        return;
    }

    let workspace_bytes = batch.workspace.workspace_bytes().unwrap_or(0);
    ENGINE_BATCH_WORKSPACE_BYTES.fetch_add(workspace_bytes, Ordering::Relaxed);
    for (index, amount) in [
        batch.workspace.host_bytes,
        batch.workspace.device_bytes,
        batch.workspace.unified_bytes,
        batch.workspace.temporary_bytes,
    ]
    .into_iter()
    .enumerate()
    {
        if let ResourceAmount::Known(bytes) = amount {
            ENGINE_WORKSPACE_DOMAIN_BYTES[index].fetch_add(bytes, Ordering::Relaxed);
        }
    }
    if !matches!(
        dispatch.kind,
        BatchDispatchKind::TensorStatic | BatchDispatchKind::TensorContinuous
    ) {
        record_engine_batch_dispatch(dispatch);
        return;
    }

    record_engine_batch_dispatch(dispatch);
    if dispatch.kind == BatchDispatchKind::TensorContinuous && batch.rows.len() >= 2 {
        ENGINE_TENSOR_CONTINUOUS_MULTIROW_BATCHES.fetch_add(1, Ordering::Relaxed);
    }
    ENGINE_TENSOR_BATCH_ROWS.fetch_add(batch.rows.len() as u64, Ordering::Relaxed);
    ENGINE_TENSOR_BATCH_CAPACITY_ROWS.fetch_add(batch.budget.max_rows as u64, Ordering::Relaxed);
    let useful_elements = batch.rows.iter().fold(0u64, |total, row| {
        total.saturating_add(row.cost.tensor_elements)
    });
    ENGINE_TENSOR_BATCH_USEFUL_ELEMENTS.fetch_add(useful_elements, Ordering::Relaxed);
    ENGINE_TENSOR_BATCH_MATERIALIZED_ELEMENTS
        .fetch_add(batch.materialized_tensor_elements, Ordering::Relaxed);
}

pub fn engine_tensor_batches_total() -> u64 {
    ENGINE_TENSOR_BATCHES.load(Ordering::Relaxed)
}

pub fn engine_request_parallel_batches_total() -> u64 {
    ENGINE_REQUEST_PARALLEL_BATCHES.load(Ordering::Relaxed)
}

pub fn engine_tensor_batch_max_width() -> u64 {
    ENGINE_TENSOR_BATCH_MAX_WIDTH.load(Ordering::Relaxed)
}

pub fn engine_batch_metrics_snapshot() -> EngineBatchMetricsSnapshot {
    let rows = ENGINE_TENSOR_BATCH_ROWS.load(Ordering::Relaxed);
    let capacity_rows = ENGINE_TENSOR_BATCH_CAPACITY_ROWS.load(Ordering::Relaxed);
    let useful_elements = ENGINE_TENSOR_BATCH_USEFUL_ELEMENTS.load(Ordering::Relaxed);
    let materialized_elements = ENGINE_TENSOR_BATCH_MATERIALIZED_ELEMENTS.load(Ordering::Relaxed);
    EngineBatchMetricsSnapshot {
        tensor_batches_total: engine_tensor_batches_total(),
        tensor_static_batches_total: ENGINE_TENSOR_STATIC_BATCHES.load(Ordering::Relaxed),
        tensor_continuous_batches_total: ENGINE_TENSOR_CONTINUOUS_BATCHES.load(Ordering::Relaxed),
        tensor_continuous_multirow_batches_total: ENGINE_TENSOR_CONTINUOUS_MULTIROW_BATCHES
            .load(Ordering::Relaxed),
        request_parallel_batches_total: engine_request_parallel_batches_total(),
        physical_batch_rejections_total: ENGINE_PHYSICAL_BATCH_REJECTIONS.load(Ordering::Relaxed),
        tensor_batch_max_width: engine_tensor_batch_max_width(),
        tensor_batch_rows_total: rows,
        tensor_batch_capacity_rows_total: capacity_rows,
        tensor_batch_useful_elements_total: useful_elements,
        tensor_batch_materialized_elements_total: materialized_elements,
        batch_workspace_bytes_total: ENGINE_BATCH_WORKSPACE_BYTES.load(Ordering::Relaxed),
        dispatch_states: EngineDispatchStateMetricsSnapshot {
            not_started: ENGINE_DISPATCH_STATE_ROWS[0].load(Ordering::Relaxed),
            started: ENGINE_DISPATCH_STATE_ROWS[1].load(Ordering::Relaxed),
            produced_output: ENGINE_DISPATCH_STATE_ROWS[2].load(Ordering::Relaxed),
        },
        failure_origins: EngineFailureOriginMetricsSnapshot {
            adapter_planning: ENGINE_FAILURE_ORIGIN_ROWS[0].load(Ordering::Relaxed),
            dispatch_coordination: ENGINE_FAILURE_ORIGIN_ROWS[1].load(Ordering::Relaxed),
            workspace_admission: ENGINE_FAILURE_ORIGIN_ROWS[2].load(Ordering::Relaxed),
            executor_validation: ENGINE_FAILURE_ORIGIN_ROWS[3].load(Ordering::Relaxed),
            model: ENGINE_FAILURE_ORIGIN_ROWS[4].load(Ordering::Relaxed),
            stream_delivery: ENGINE_FAILURE_ORIGIN_ROWS[5].load(Ordering::Relaxed),
            state_commit: ENGINE_FAILURE_ORIGIN_ROWS[6].load(Ordering::Relaxed),
            cleanup: ENGINE_FAILURE_ORIGIN_ROWS[7].load(Ordering::Relaxed),
            panic: ENGINE_FAILURE_ORIGIN_ROWS[8].load(Ordering::Relaxed),
        },
        deadline_phases: EngineDeadlinePhaseMetricsSnapshot {
            scheduler_queue: ENGINE_DEADLINE_PHASE_ROWS[0].load(Ordering::Relaxed),
            dispatch_wait: ENGINE_DEADLINE_PHASE_ROWS[1].load(Ordering::Relaxed),
            model_execution: ENGINE_DEADLINE_PHASE_ROWS[2].load(Ordering::Relaxed),
            stream_delivery: ENGINE_DEADLINE_PHASE_ROWS[3].load(Ordering::Relaxed),
            terminal_delivery: ENGINE_DEADLINE_PHASE_ROWS[4].load(Ordering::Relaxed),
        },
        workspace_domains: EngineWorkspaceDomainMetricsSnapshot {
            host: ENGINE_WORKSPACE_DOMAIN_BYTES[0].load(Ordering::Relaxed),
            device: ENGINE_WORKSPACE_DOMAIN_BYTES[1].load(Ordering::Relaxed),
            unified: ENGINE_WORKSPACE_DOMAIN_BYTES[2].load(Ordering::Relaxed),
            temporary: ENGINE_WORKSPACE_DOMAIN_BYTES[3].load(Ordering::Relaxed),
        },
        tensor_batch_fill_ratio: ratio(rows, capacity_rows),
        tensor_batch_padding_ratio: ratio(
            materialized_elements.saturating_sub(useful_elements),
            materialized_elements,
        ),
        model_tensor_batches_total: ENGINE_MODEL_TENSOR_BATCHES.load(Ordering::Relaxed),
        model_tensor_batch_rows_total: ENGINE_MODEL_TENSOR_BATCH_ROWS.load(Ordering::Relaxed),
        model_tensor_batch_max_width: ENGINE_MODEL_TENSOR_BATCH_MAX_WIDTH.load(Ordering::Relaxed),
        model_scalar_row_dispatches_total: ENGINE_MODEL_SCALAR_ROW_DISPATCHES
            .load(Ordering::Relaxed),
        model_decode_calls_total: ENGINE_MODEL_DECODE_CALLS.load(Ordering::Relaxed),
        model_tensor_multirow_calls_total: ENGINE_MODEL_TENSOR_MULTIROW_CALLS
            .load(Ordering::Relaxed),
        continuous_envelope_scalar_fallbacks_total: ENGINE_CONTINUOUS_ENVELOPE_SCALAR_FALLBACKS
            .load(Ordering::Relaxed),
    }
}

fn ratio(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

pub fn prometheus_engine_metric_name(name: &str) -> String {
    format!("izwi_{}", name.replace('.', "_"))
}

pub fn prometheus_engine_metric_type(name: &str) -> &'static str {
    if name.ends_with("_total") {
        "counter"
    } else {
        "gauge"
    }
}

/// Global metrics collector for the engine.
#[derive(Debug)]
pub struct MetricsCollector {
    /// Request latency samples (for histogram)
    latency_samples: RwLock<VecDeque<f64>>,
    /// RTF samples
    rtf_samples: RwLock<VecDeque<f64>>,
    /// Throughput samples (tokens/sec)
    throughput_samples: RwLock<VecDeque<f64>>,
    /// Total requests processed
    total_requests: AtomicU64,
    /// Total tokens generated
    total_tokens: AtomicU64,
    /// Total audio duration generated (microseconds)
    total_audio_duration_us: AtomicU64,
    /// Total processing time (microseconds)
    total_processing_time_us: AtomicU64,
    /// Start time for uptime tracking
    start_time: Instant,
    /// Maximum samples to keep
    max_samples: usize,
}

impl MetricsCollector {
    /// Create a new metrics collector.
    pub fn new() -> Self {
        Self {
            latency_samples: RwLock::new(VecDeque::with_capacity(1000)),
            rtf_samples: RwLock::new(VecDeque::with_capacity(1000)),
            throughput_samples: RwLock::new(VecDeque::with_capacity(1000)),
            total_requests: AtomicU64::new(0),
            total_tokens: AtomicU64::new(0),
            total_audio_duration_us: AtomicU64::new(0),
            total_processing_time_us: AtomicU64::new(0),
            start_time: Instant::now(),
            max_samples: 1000,
        }
    }

    /// Record a completed request.
    pub async fn record_request(
        &self,
        latency: Duration,
        tokens_generated: u64,
        audio_duration: Duration,
    ) {
        let latency_ms = latency.as_secs_f64() * 1000.0;
        let audio_secs = audio_duration.as_secs_f64();
        let rtf = if audio_secs > 0.0 {
            latency.as_secs_f64() / audio_secs
        } else {
            0.0
        };
        let tokens_per_sec = if latency.as_secs_f64() > 0.0 {
            tokens_generated as f64 / latency.as_secs_f64()
        } else {
            0.0
        };

        // Update counters
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_tokens
            .fetch_add(tokens_generated, Ordering::Relaxed);
        self.total_audio_duration_us
            .fetch_add(audio_duration.as_micros() as u64, Ordering::Relaxed);
        self.total_processing_time_us
            .fetch_add(latency.as_micros() as u64, Ordering::Relaxed);

        // Add samples
        {
            let mut samples = self.latency_samples.write().await;
            if samples.len() >= self.max_samples {
                samples.pop_front();
            }
            samples.push_back(latency_ms);
        }

        {
            let mut samples = self.rtf_samples.write().await;
            if samples.len() >= self.max_samples {
                samples.pop_front();
            }
            samples.push_back(rtf);
        }

        {
            let mut samples = self.throughput_samples.write().await;
            if samples.len() >= self.max_samples {
                samples.pop_front();
            }
            samples.push_back(tokens_per_sec);
        }
    }

    /// Get current metrics snapshot.
    pub async fn snapshot(&self) -> MetricsSnapshot {
        let latency_samples = self.latency_samples.read().await;
        let rtf_samples = self.rtf_samples.read().await;
        let throughput_samples = self.throughput_samples.read().await;

        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let total_tokens = self.total_tokens.load(Ordering::Relaxed);
        let total_audio_us = self.total_audio_duration_us.load(Ordering::Relaxed);
        let total_processing_us = self.total_processing_time_us.load(Ordering::Relaxed);

        MetricsSnapshot {
            uptime_secs: self.start_time.elapsed().as_secs_f64(),
            total_requests,
            total_tokens,
            total_audio_duration_secs: total_audio_us as f64 / 1_000_000.0,
            total_processing_time_secs: total_processing_us as f64 / 1_000_000.0,
            avg_latency_ms: compute_mean(&latency_samples),
            p50_latency_ms: compute_percentile(&latency_samples, 0.50),
            p90_latency_ms: compute_percentile(&latency_samples, 0.90),
            p99_latency_ms: compute_percentile(&latency_samples, 0.99),
            avg_rtf: compute_mean(&rtf_samples),
            avg_tokens_per_sec: compute_mean(&throughput_samples),
            requests_per_sec: if self.start_time.elapsed().as_secs_f64() > 0.0 {
                total_requests as f64 / self.start_time.elapsed().as_secs_f64()
            } else {
                0.0
            },
        }
    }

    /// Reset all metrics.
    pub async fn reset(&self) {
        self.total_requests.store(0, Ordering::Relaxed);
        self.total_tokens.store(0, Ordering::Relaxed);
        self.total_audio_duration_us.store(0, Ordering::Relaxed);
        self.total_processing_time_us.store(0, Ordering::Relaxed);

        self.latency_samples.write().await.clear();
        self.rtf_samples.write().await.clear();
        self.throughput_samples.write().await.clear();
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// A snapshot of current metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// Engine uptime in seconds
    pub uptime_secs: f64,
    /// Total requests processed
    pub total_requests: u64,
    /// Total tokens generated
    pub total_tokens: u64,
    /// Total audio duration generated (seconds)
    pub total_audio_duration_secs: f64,
    /// Total processing time (seconds)
    pub total_processing_time_secs: f64,
    /// Average latency (milliseconds)
    pub avg_latency_ms: f64,
    /// 50th percentile latency (milliseconds)
    pub p50_latency_ms: f64,
    /// 90th percentile latency (milliseconds)
    pub p90_latency_ms: f64,
    /// 99th percentile latency (milliseconds)
    pub p99_latency_ms: f64,
    /// Average real-time factor
    pub avg_rtf: f64,
    /// Average tokens per second
    pub avg_tokens_per_sec: f64,
    /// Requests per second
    pub requests_per_sec: f64,
}

impl MetricsSnapshot {
    /// Create an empty snapshot.
    pub fn empty() -> Self {
        Self {
            uptime_secs: 0.0,
            total_requests: 0,
            total_tokens: 0,
            total_audio_duration_secs: 0.0,
            total_processing_time_secs: 0.0,
            avg_latency_ms: 0.0,
            p50_latency_ms: 0.0,
            p90_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            avg_rtf: 0.0,
            avg_tokens_per_sec: 0.0,
            requests_per_sec: 0.0,
        }
    }
}

/// Timer for tracking request latency.
pub struct RequestTimer {
    start: Instant,
    metrics: Arc<MetricsCollector>,
}

impl RequestTimer {
    /// Start a new request timer.
    pub fn start(metrics: Arc<MetricsCollector>) -> Self {
        Self {
            start: Instant::now(),
            metrics,
        }
    }

    /// Stop the timer and record metrics.
    pub async fn stop(self, tokens_generated: u64, audio_duration: Duration) {
        let latency = self.start.elapsed();
        self.metrics
            .record_request(latency, tokens_generated, audio_duration)
            .await;
    }

    /// Get elapsed time without stopping.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

/// Compute mean of samples.
fn compute_mean(samples: &VecDeque<f64>) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().sum::<f64>() / samples.len() as f64
}

/// Compute percentile of samples.
fn compute_percentile(samples: &VecDeque<f64>, percentile: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }

    let mut sorted: Vec<f64> = samples.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let index = ((percentile * sorted.len() as f64) as usize).min(sorted.len() - 1);
    sorted[index]
}

/// Benchmark results for a test run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Test name/description
    pub name: String,
    /// Number of requests in the benchmark
    pub num_requests: u64,
    /// Total duration of the benchmark
    pub total_duration_secs: f64,
    /// Metrics snapshot at end of benchmark
    pub metrics: MetricsSnapshot,
    /// Throughput in requests per second
    pub throughput_rps: f64,
    /// Average time to first token (TTFT) in milliseconds
    pub avg_ttft_ms: f64,
    /// Average time per output token (TPOT) in milliseconds  
    pub avg_tpot_ms: f64,
}

impl BenchmarkResult {
    /// Create a new benchmark result.
    pub fn new(
        name: impl Into<String>,
        num_requests: u64,
        total_duration: Duration,
        metrics: MetricsSnapshot,
    ) -> Self {
        let total_secs = total_duration.as_secs_f64();

        Self {
            name: name.into(),
            num_requests,
            total_duration_secs: total_secs,
            metrics: metrics.clone(),
            throughput_rps: if total_secs > 0.0 {
                num_requests as f64 / total_secs
            } else {
                0.0
            },
            avg_ttft_ms: metrics.p50_latency_ms * 0.3, // Estimate TTFT as ~30% of total latency
            avg_tpot_ms: if metrics.avg_tokens_per_sec > 0.0 {
                1000.0 / metrics.avg_tokens_per_sec
            } else {
                0.0
            },
        }
    }

    /// Format as a summary string.
    pub fn summary(&self) -> String {
        format!(
            "Benchmark: {}\n\
             Requests: {}, Duration: {:.2}s\n\
             Throughput: {:.2} req/s\n\
             Latency: avg={:.1}ms, p50={:.1}ms, p90={:.1}ms, p99={:.1}ms\n\
             RTF: {:.3} (< 1.0 = faster than real-time)\n\
             Tokens/sec: {:.1}",
            self.name,
            self.num_requests,
            self.total_duration_secs,
            self.throughput_rps,
            self.metrics.avg_latency_ms,
            self.metrics.p50_latency_ms,
            self.metrics.p90_latency_ms,
            self.metrics.p99_latency_ms,
            self.metrics.avg_rtf,
            self.metrics.avg_tokens_per_sec,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::BackendKind;
    use crate::engine::{
        AdapterAbiRevision, AdapterInstanceId, BatchBudget, BatchId, BatchLaneKey,
        ExecutionGroupId, InputRange, ModelInstanceId, NativeBatchMode, PlanId, ReadyQuantum,
        ResourceVector, SequencePhase, SessionKey, StageId, WorkCost, WorkUnit,
    };

    #[tokio::test]
    async fn test_metrics_collector() {
        let collector = MetricsCollector::new();

        // Record some requests
        collector
            .record_request(Duration::from_millis(100), 50, Duration::from_secs(1))
            .await;

        collector
            .record_request(Duration::from_millis(200), 100, Duration::from_secs(2))
            .await;

        let snapshot = collector.snapshot().await;
        assert_eq!(snapshot.total_requests, 2);
        assert_eq!(snapshot.total_tokens, 150);
    }

    #[test]
    fn test_percentile() {
        let mut samples = VecDeque::new();
        for i in 1..=100 {
            samples.push_back(i as f64);
        }

        assert!((compute_percentile(&samples, 0.50) - 50.0).abs() < 2.0);
        assert!((compute_percentile(&samples, 0.90) - 90.0).abs() < 2.0);
    }

    #[test]
    fn engine_metric_catalog_exposes_scheduler_and_cache_contract() {
        let names = engine_metric_catalog()
            .iter()
            .map(|descriptor| descriptor.name)
            .collect::<std::collections::HashSet<_>>();

        assert!(names.contains(ENGINE_SCHEDULER_QUEUE_DEPTH));
        assert!(names.contains(ENGINE_KV_CACHE_HITS_TOTAL));
        assert!(names.contains(ENGINE_KV_CACHE_EVICTIONS_TOTAL));
        assert!(names.contains(ENGINE_STREAM_BACKPRESSURE_TOTAL));
        assert!(names.contains(ENGINE_STREAM_CHECKPOINTS_COMMITTED_TOTAL));
        assert!(names.contains(ENGINE_STREAM_CHECKPOINT_REJECTIONS_TOTAL));
        assert!(names.contains(ENGINE_STREAM_DELIVERY_FAILURES_TOTAL));
        assert!(names.contains(ENGINE_EXECUTOR_TENSOR_BATCHES_TOTAL));
        assert!(names.contains(ENGINE_EXECUTOR_REQUEST_PARALLEL_BATCHES_TOTAL));
        assert!(names.contains(ENGINE_EXECUTOR_TENSOR_BATCH_MAX_WIDTH));
        assert!(names.contains(ENGINE_EXECUTOR_TENSOR_STATIC_BATCHES_TOTAL));
        assert!(names.contains(ENGINE_EXECUTOR_TENSOR_CONTINUOUS_BATCHES_TOTAL));
        assert!(names.contains(ENGINE_EXECUTOR_PHYSICAL_BATCH_REJECTIONS_TOTAL));
        assert!(names.contains(ENGINE_EXECUTOR_DISPATCH_STATE_ROWS_TOTAL));
        assert!(names.contains(ENGINE_EXECUTOR_FAILURE_ORIGIN_ROWS_TOTAL));
        assert!(names.contains(ENGINE_EXECUTOR_DEADLINE_PHASE_ROWS_TOTAL));
        assert!(names.contains(ENGINE_EXECUTOR_BATCH_WORKSPACE_DOMAIN_BYTES_TOTAL));
        assert!(names.contains(ENGINE_EXECUTOR_TENSOR_BATCH_FILL_RATIO));
        assert!(names.contains(ENGINE_EXECUTOR_TENSOR_BATCH_PADDING_RATIO));
        assert_eq!(names.len(), ENGINE_METRIC_CATALOG.len());
    }

    #[test]
    fn engine_metric_prometheus_helpers_preserve_counter_suffix() {
        assert_eq!(
            prometheus_engine_metric_name(ENGINE_KV_CACHE_HITS_TOTAL),
            "izwi_engine_kv_cache_hits_total"
        );
        assert_eq!(
            prometheus_engine_metric_type(ENGINE_KV_CACHE_HITS_TOTAL),
            "counter"
        );
        assert_eq!(
            prometheus_engine_metric_type(ENGINE_KV_CACHE_ALLOCATED_BLOCKS),
            "gauge"
        );
    }

    #[test]
    fn engine_stream_counters_are_observable() {
        let before = engine_stream_metrics_snapshot();
        record_engine_stream_backpressure();
        record_engine_stream_checkpoint_committed();
        record_engine_stream_checkpoint_rejection();
        record_engine_stream_delivery_failure();
        let after = engine_stream_metrics_snapshot();
        assert_eq!(after.backpressure_total, before.backpressure_total + 1);
        assert_eq!(
            after.checkpoints_committed_total,
            before.checkpoints_committed_total + 1
        );
        assert_eq!(
            after.checkpoint_rejections_total,
            before.checkpoint_rejections_total + 1
        );
        assert_eq!(
            after.delivery_failures_total,
            before.delivery_failures_total + 1
        );
    }

    #[test]
    fn batch_dispatch_metrics_distinguish_tensor_and_request_parallel_work() {
        let tensor_before = engine_tensor_batches_total();
        let parallel_before = engine_request_parallel_batches_total();
        record_engine_batch_dispatch(BatchDispatch::new(BatchDispatchKind::TensorStatic, 3));
        record_engine_batch_dispatch(BatchDispatch::new(BatchDispatchKind::RequestParallel, 4));
        assert!(engine_tensor_batches_total() > tensor_before);
        assert!(engine_request_parallel_batches_total() > parallel_before);
        assert!(engine_tensor_batch_max_width() >= 3);
    }

    #[test]
    fn model_dispatch_metrics_distinguish_true_tensor_batches_from_scalar_rows() {
        let before = engine_batch_metrics_snapshot();
        record_engine_chat_model_dispatch(true, 3);
        record_engine_chat_model_dispatch(false, 2);
        let after = engine_batch_metrics_snapshot();

        assert!(after.model_tensor_batches_total > before.model_tensor_batches_total);
        assert!(after.model_tensor_batch_rows_total >= before.model_tensor_batch_rows_total + 3);
        assert!(after.model_tensor_batch_max_width >= 3);
        assert!(
            after.model_scalar_row_dispatches_total >= before.model_scalar_row_dispatches_total + 2
        );
        assert!(after.model_decode_calls_total >= before.model_decode_calls_total + 2);
        assert!(after.model_tensor_multirow_calls_total > before.model_tensor_multirow_calls_total);
        assert!(
            after.continuous_envelope_scalar_fallbacks_total
                > before.continuous_envelope_scalar_fallbacks_total
        );
    }

    #[test]
    fn execution_outcome_metrics_use_only_bounded_provenance_dimensions() {
        let before = engine_batch_metrics_snapshot();
        record_engine_execution_outcome(OutcomeProvenance::failure(
            FailureOrigin::WorkspaceAdmission,
            DispatchState::NotStarted,
        ));
        record_engine_execution_outcome(OutcomeProvenance::deadline(
            DeadlinePhase::ModelExecution,
            DispatchState::Started,
        ));
        record_engine_execution_outcome(OutcomeProvenance::produced_output());
        let after = engine_batch_metrics_snapshot();

        assert!(after.dispatch_states.not_started > before.dispatch_states.not_started);
        assert!(after.dispatch_states.started > before.dispatch_states.started);
        assert!(after.dispatch_states.produced_output > before.dispatch_states.produced_output);
        assert!(
            after.failure_origins.workspace_admission > before.failure_origins.workspace_admission
        );
        assert!(after.deadline_phases.model_execution > before.deadline_phases.model_execution);
        assert_eq!(after.failure_origins.labeled_values().len(), 9);
        assert_eq!(after.deadline_phases.labeled_values().len(), 5);
    }

    #[test]
    fn physical_batch_metrics_measure_fill_padding_workspace_and_rejection() {
        let lane = BatchLaneKey {
            execution_group: ExecutionGroupId::new(1),
            model_instance: ModelInstanceId::new(2),
            adapter_instance: AdapterInstanceId::new(3),
            adapter_abi: AdapterAbiRevision::new(1),
            capability_id: "chat".to_string(),
            stage_id: StageId::new(4),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            compute_dtype: "f32".to_string(),
            state_dtype: "f32".to_string(),
            tensor_layout: "ragged".to_string(),
            quantization: "none".to_string(),
            state_schema: "test.v1".to_string(),
            kernel_mode: "test".to_string(),
            semantic_mode: "greedy".to_string(),
            shape_bucket: "token.1".to_string(),
        };
        let row = |plan: PlanId, request: &str| ReadyQuantum {
            plan_id: plan,
            session: SessionKey::new(request.to_string(), plan),
            lane: lane.clone(),
            work: WorkUnit::SequenceStep {
                phase: SequencePhase::Decode,
                input: InputRange { start: 0, end: 1 },
                max_output_steps: 1,
            },
            cost: WorkCost::new(1, 10, 0),
            managed_cache: None,
        };
        let batch = PhysicalBatch {
            batch_id: BatchId::new(5),
            lane: lane.clone(),
            mode: NativeBatchMode::Static,
            budget: BatchBudget {
                max_rows: 4,
                max_logical_units: 4,
                max_tensor_elements: 40,
                max_workspace_bytes: 8,
                max_padding_basis_points: 5_000,
                max_formation_delay: Duration::ZERO,
            },
            rows: vec![row(1, "a"), row(2, "b")],
            materialized_tensor_elements: 30,
            workspace: ResourceVector {
                host_bytes: ResourceAmount::Known(1),
                device_bytes: ResourceAmount::Known(2),
                unified_bytes: ResourceAmount::Known(3),
                temporary_bytes: ResourceAmount::Known(2),
                ..ResourceVector::zero()
            },
        };
        batch.validate().unwrap();

        let before = engine_batch_metrics_snapshot();
        record_engine_physical_batch(
            &batch,
            BatchDispatch::new(BatchDispatchKind::TensorStatic, 2),
        );
        let dispatched = engine_batch_metrics_snapshot();
        assert!(dispatched.tensor_static_batches_total > before.tensor_static_batches_total);
        assert!(dispatched.tensor_batch_rows_total >= before.tensor_batch_rows_total + 2);
        assert!(
            dispatched.tensor_batch_capacity_rows_total
                >= before.tensor_batch_capacity_rows_total + 4
        );
        assert!(
            dispatched.tensor_batch_useful_elements_total
                >= before.tensor_batch_useful_elements_total + 20
        );
        assert!(
            dispatched.tensor_batch_materialized_elements_total
                >= before.tensor_batch_materialized_elements_total + 30
        );
        assert!(dispatched.batch_workspace_bytes_total >= before.batch_workspace_bytes_total + 8);
        assert!(dispatched.workspace_domains.host > before.workspace_domains.host);
        assert!(dispatched.workspace_domains.device >= before.workspace_domains.device + 2);
        assert!(dispatched.workspace_domains.unified >= before.workspace_domains.unified + 3);
        assert!(dispatched.workspace_domains.temporary >= before.workspace_domains.temporary + 2);

        record_engine_physical_batch(
            &batch,
            BatchDispatch::new(BatchDispatchKind::TensorContinuous, 2),
        );
        let continuous = engine_batch_metrics_snapshot();
        assert!(
            continuous.tensor_continuous_multirow_batches_total
                > dispatched.tensor_continuous_multirow_batches_total
        );

        record_engine_physical_batch(&batch, BatchDispatch::not_dispatched(2));
        let rejected = engine_batch_metrics_snapshot();
        assert!(
            rejected.physical_batch_rejections_total > continuous.physical_batch_rejections_total
        );
        assert!(rejected.batch_workspace_bytes_total >= continuous.batch_workspace_bytes_total);
    }
}
