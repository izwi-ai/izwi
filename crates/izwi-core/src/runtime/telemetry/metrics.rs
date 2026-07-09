//! Runtime metrics, snapshots, and Prometheus formatting.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex as StdMutex;
use std::time::Instant;

use serde::Serialize;
use tokio::sync::Mutex;

use crate::engine::{
    engine_metric_catalog, prometheus_engine_metric_name, prometheus_engine_metric_type,
    EngineMetricDescriptor, EngineOutput,
};
use crate::models::shared::telemetry::{
    prometheus as kernel_path_prometheus, snapshot as kernel_path_telemetry_snapshot,
};
use crate::runtime::pipeline::{
    PipelineExecutionSummary, PipelineExecutor, PipelineGraph, PipelineKind,
};
use crate::runtime::voice_metrics::{
    prometheus_voice_metric_name, voice_metric_catalog, voice_metric_prometheus_contract,
    VoiceMetricDescriptor, VOICE_BARGE_IN_TOTAL, VOICE_SESSION_CLOSED_TOTAL,
    VOICE_SESSION_INTERRUPTED_TOTAL, VOICE_SESSION_STARTED_TOTAL, VOICE_STREAM_BACKPRESSURE_TOTAL,
};
use crate::KernelPathTelemetrySnapshot;

#[derive(Debug, Clone, Serialize)]
pub struct VoiceRuntimeTelemetrySnapshot {
    pub sessions_started: u64,
    pub sessions_closed: u64,
    pub interruptions: u64,
    pub barge_ins: u64,
    pub stream_backpressure_total: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct InferenceBrokerRuntimeTelemetrySnapshot {
    pub shadow_requests: u64,
    pub execution_requests: u64,
    pub route_decisions: u64,
    pub validation_failures: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct EngineRuntimeTelemetrySnapshot {
    pub scheduler_queue_depth: u64,
    pub scheduler_running_requests: u64,
    pub kv_cache_hits_total: u64,
    pub kv_cache_misses_total: u64,
    pub kv_cache_evictions_total: u64,
    pub kv_cache_allocated_blocks: u64,
    pub kv_cache_prefix_reuse_blocks_total: u64,
    pub stream_backpressure_total: u64,
    pub kv_cache: EngineKvCacheRuntimeSnapshot,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct EngineKvCacheRuntimeSnapshot {
    pub total_blocks: u64,
    pub soft_max_blocks: u64,
    pub allocated_blocks: u64,
    pub free_blocks: u64,
    pub block_size: u64,
    pub dtype_bytes: u64,
    pub block_memory_bytes: u64,
    pub memory_used_bytes: u64,
    pub memory_capacity_bytes: u64,
    pub utilization_ratio: f64,
    pub gpu_resident_blocks: u64,
    pub pinned_blocks: u64,
    pub shared_prefixes: u64,
    pub total_allocations: u64,
    pub total_frees: u64,
    pub shared_prefix_hits: u64,
    pub copy_on_write_splits: u64,
    pub last_churn_ratio: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct PipelineRuntimeTelemetrySnapshot {
    pub modular_voice_turns: u64,
    pub unified_voice_turns: u64,
    pub diarization_transcripts: u64,
    pub batch_asr_transcriptions: u64,
    pub batch_tts_speech: u64,
    pub stages_recorded: u64,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct RuntimeObservationContext {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capability: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_variant: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend_kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_target: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub streaming_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workload_class: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline_kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline_stage: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_job_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_stage_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route_record_id: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeStageOutcome {
    Created,
    Claimed,
    Started,
    Completed,
    Failed,
    Retried,
    Skipped,
    Cancelled,
    Observed,
}

impl RuntimeStageOutcome {
    fn is_failure(self) -> bool {
        matches!(self, Self::Failed)
    }
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct RuntimeStageTiming {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_wait_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub admission_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_decode_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub normalization_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampling_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codec_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postprocess_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_write_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_ms: Option<f64>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct RuntimeStageOutputCounters {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_frames: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_samples: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcript_chars: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcript_segments: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_artifacts: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RuntimeStageObservation {
    pub context: RuntimeObservationContext,
    pub outcome: RuntimeStageOutcome,
    #[serde(default)]
    pub timing: RuntimeStageTiming,
    #[serde(default)]
    pub outputs: RuntimeStageOutputCounters,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub quality_flags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<String>,
}

impl RuntimeStageObservation {
    pub fn new(context: RuntimeObservationContext, outcome: RuntimeStageOutcome) -> Self {
        Self {
            context,
            outcome,
            timing: RuntimeStageTiming::default(),
            outputs: RuntimeStageOutputCounters::default(),
            quality_flags: Vec::new(),
            error_kind: None,
        }
    }

    pub fn with_total_ms(mut self, total_ms: f64) -> Self {
        self.timing.total_ms = Some(total_ms.max(0.0));
        self
    }

    pub fn with_error_kind(mut self, error_kind: impl Into<String>) -> Self {
        self.error_kind = Some(error_kind.into());
        self
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct RuntimeObservabilityTelemetrySnapshot {
    pub stage_observations_total: u64,
    pub stage_failures_total: u64,
    pub stage_duration_ms_avg: f64,
    pub stage_duration_ms_p50: f64,
    pub stage_duration_ms_p95: f64,
    pub workload_classes: Vec<RuntimeWorkloadClassTelemetrySnapshot>,
    pub recent_stage_samples: Vec<RuntimeStageObservation>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct RuntimeLatencyStats {
    pub count: usize,
    pub avg: f64,
    pub p50: f64,
    pub p95: f64,
}

impl RuntimeLatencyStats {
    fn from_slice(values: &[f64]) -> Self {
        Self {
            count: values.len(),
            avg: mean_slice(values),
            p50: percentile_slice(values, 0.50),
            p95: percentile_slice(values, 0.95),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct RuntimeWorkloadClassTelemetrySnapshot {
    pub workload_class: String,
    pub observations: u64,
    pub failures: u64,
    pub queue_wait_ms: RuntimeLatencyStats,
    pub prefill_ms: RuntimeLatencyStats,
    pub decode_ms: RuntimeLatencyStats,
    pub ttft_ms: RuntimeLatencyStats,
    pub stage_duration_ms: RuntimeLatencyStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeTelemetrySnapshot {
    pub uptime_secs: f64,
    pub requests_queued: u64,
    pub requests_completed: u64,
    pub requests_failed: u64,
    pub requests_active: u64,
    pub worker_restarts: u64,
    pub worker_panics: u64,
    pub queue_wait_ms_avg: f64,
    pub queue_wait_ms_p50: f64,
    pub queue_wait_ms_p95: f64,
    pub prefill_ms_avg: f64,
    pub prefill_ms_p50: f64,
    pub prefill_ms_p95: f64,
    pub decode_ms_avg: f64,
    pub decode_ms_p50: f64,
    pub decode_ms_p95: f64,
    pub ttft_ms_avg: f64,
    pub ttft_ms_p50: f64,
    pub ttft_ms_p95: f64,
    pub end_to_end_ms_avg: f64,
    pub end_to_end_ms_p50: f64,
    pub end_to_end_ms_p95: f64,
    pub kernel_path: KernelPathTelemetrySnapshot,
    pub engine: EngineRuntimeTelemetrySnapshot,
    pub voice: VoiceRuntimeTelemetrySnapshot,
    pub broker: InferenceBrokerRuntimeTelemetrySnapshot,
    pub pipelines: PipelineRuntimeTelemetrySnapshot,
    pub observability: RuntimeObservabilityTelemetrySnapshot,
    pub engine_metrics: &'static [EngineMetricDescriptor],
    pub voice_metrics: &'static [VoiceMetricDescriptor],
}

#[derive(Debug)]
pub(crate) struct RuntimeTelemetryCollector {
    start_time: Instant,
    max_samples: usize,
    requests_queued: AtomicU64,
    requests_completed: AtomicU64,
    requests_failed: AtomicU64,
    requests_active: AtomicU64,
    worker_restarts: AtomicU64,
    worker_panics: AtomicU64,
    voice_sessions_started: AtomicU64,
    voice_sessions_closed: AtomicU64,
    voice_interruptions: AtomicU64,
    voice_barge_ins: AtomicU64,
    voice_stream_backpressure: AtomicU64,
    broker_shadow_requests: AtomicU64,
    broker_execution_requests: AtomicU64,
    broker_route_decisions: AtomicU64,
    broker_validation_failures: AtomicU64,
    pipeline_modular_voice_turns: AtomicU64,
    pipeline_unified_voice_turns: AtomicU64,
    pipeline_diarization_transcripts: AtomicU64,
    pipeline_batch_asr_transcriptions: AtomicU64,
    pipeline_batch_tts_speech: AtomicU64,
    pipeline_stages_recorded: AtomicU64,
    stage_observations_total: AtomicU64,
    stage_failures_total: AtomicU64,
    stage_duration_ms_samples: StdMutex<VecDeque<f64>>,
    stage_observation_samples: StdMutex<VecDeque<RuntimeStageObservation>>,
    queue_wait_ms_samples: Mutex<VecDeque<f64>>,
    prefill_ms_samples: Mutex<VecDeque<f64>>,
    decode_ms_samples: Mutex<VecDeque<f64>>,
    ttft_ms_samples: Mutex<VecDeque<f64>>,
    end_to_end_ms_samples: Mutex<VecDeque<f64>>,
}

impl RuntimeTelemetryCollector {
    pub(crate) fn new(max_samples: usize) -> Self {
        Self {
            start_time: Instant::now(),
            max_samples: max_samples.max(64),
            requests_queued: AtomicU64::new(0),
            requests_completed: AtomicU64::new(0),
            requests_failed: AtomicU64::new(0),
            requests_active: AtomicU64::new(0),
            worker_restarts: AtomicU64::new(0),
            worker_panics: AtomicU64::new(0),
            voice_sessions_started: AtomicU64::new(0),
            voice_sessions_closed: AtomicU64::new(0),
            voice_interruptions: AtomicU64::new(0),
            voice_barge_ins: AtomicU64::new(0),
            voice_stream_backpressure: AtomicU64::new(0),
            broker_shadow_requests: AtomicU64::new(0),
            broker_execution_requests: AtomicU64::new(0),
            broker_route_decisions: AtomicU64::new(0),
            broker_validation_failures: AtomicU64::new(0),
            pipeline_modular_voice_turns: AtomicU64::new(0),
            pipeline_unified_voice_turns: AtomicU64::new(0),
            pipeline_diarization_transcripts: AtomicU64::new(0),
            pipeline_batch_asr_transcriptions: AtomicU64::new(0),
            pipeline_batch_tts_speech: AtomicU64::new(0),
            pipeline_stages_recorded: AtomicU64::new(0),
            stage_observations_total: AtomicU64::new(0),
            stage_failures_total: AtomicU64::new(0),
            stage_duration_ms_samples: StdMutex::new(VecDeque::with_capacity(max_samples.max(64))),
            stage_observation_samples: StdMutex::new(VecDeque::with_capacity(max_samples.max(64))),
            queue_wait_ms_samples: Mutex::new(VecDeque::with_capacity(max_samples.max(64))),
            prefill_ms_samples: Mutex::new(VecDeque::with_capacity(max_samples.max(64))),
            decode_ms_samples: Mutex::new(VecDeque::with_capacity(max_samples.max(64))),
            ttft_ms_samples: Mutex::new(VecDeque::with_capacity(max_samples.max(64))),
            end_to_end_ms_samples: Mutex::new(VecDeque::with_capacity(max_samples.max(64))),
        }
    }

    pub(crate) async fn record_request_queued(&self) {
        self.requests_queued.fetch_add(1, Ordering::Relaxed);
        self.requests_active.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) async fn record_request_finished(&self, output: &EngineOutput) {
        self.requests_completed.fetch_add(1, Ordering::Relaxed);
        if output.error.is_some() {
            self.requests_failed.fetch_add(1, Ordering::Relaxed);
        }
        self.requests_active.fetch_sub(1, Ordering::Relaxed);

        if let Some(latency) = output.latency_breakdown.as_ref() {
            Self::push_sample(
                &self.queue_wait_ms_samples,
                self.max_samples,
                latency.queue_wait_ms,
            )
            .await;
            Self::push_sample(
                &self.prefill_ms_samples,
                self.max_samples,
                latency.prefill_ms,
            )
            .await;
            Self::push_sample(&self.decode_ms_samples, self.max_samples, latency.decode_ms).await;
            if let Some(ttft_ms) = latency.ttft_ms {
                Self::push_sample(&self.ttft_ms_samples, self.max_samples, ttft_ms).await;
            }
            Self::push_sample(
                &self.end_to_end_ms_samples,
                self.max_samples,
                latency.total_ms,
            )
            .await;
        } else {
            Self::push_sample(
                &self.end_to_end_ms_samples,
                self.max_samples,
                output.generation_time.as_secs_f64() * 1000.0,
            )
            .await;
        }
    }

    pub(crate) fn record_forced_failures(&self, count: usize) {
        if count == 0 {
            return;
        }
        let count_u64 = count as u64;
        self.requests_completed
            .fetch_add(count_u64, Ordering::Relaxed);
        self.requests_failed.fetch_add(count_u64, Ordering::Relaxed);
        let _ = self
            .requests_active
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
                Some(v.saturating_sub(count_u64))
            });
    }

    pub(crate) fn record_worker_restart(&self) {
        self.worker_restarts.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_worker_panic(&self) {
        self.worker_panics.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_voice_session_started(&self) {
        self.voice_sessions_started.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_voice_session_closed(&self) {
        self.voice_sessions_closed.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_voice_interruption(&self) {
        self.voice_interruptions.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_voice_barge_in(&self) {
        self.voice_barge_ins.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_voice_stream_backpressure(&self) {
        self.voice_stream_backpressure
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_broker_shadow_request(&self) {
        self.broker_shadow_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_broker_execution_request(&self) {
        self.broker_execution_requests
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_broker_route_decision(&self) {
        self.broker_route_decisions.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_broker_validation_failure(&self) {
        self.broker_validation_failures
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_pipeline_graph(&self, graph: &PipelineGraph) {
        let summary = PipelineExecutor.execute_contract(graph);
        self.record_pipeline_execution(&summary);
    }

    pub(crate) fn record_pipeline_execution(&self, summary: &PipelineExecutionSummary) {
        match summary.kind() {
            PipelineKind::ModularVoiceTurn => {
                self.pipeline_modular_voice_turns
                    .fetch_add(1, Ordering::Relaxed);
            }
            PipelineKind::UnifiedVoiceTurn => {
                self.pipeline_unified_voice_turns
                    .fetch_add(1, Ordering::Relaxed);
            }
            PipelineKind::DiarizationTranscript => {
                self.pipeline_diarization_transcripts
                    .fetch_add(1, Ordering::Relaxed);
            }
            PipelineKind::BatchAsrTranscription => {
                self.pipeline_batch_asr_transcriptions
                    .fetch_add(1, Ordering::Relaxed);
            }
            PipelineKind::BatchTtsSpeech => {
                self.pipeline_batch_tts_speech
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        self.pipeline_stages_recorded
            .fetch_add(summary.stages().len() as u64, Ordering::Relaxed);
    }

    pub(crate) fn record_stage_observation(&self, observation: RuntimeStageObservation) {
        self.stage_observations_total
            .fetch_add(1, Ordering::Relaxed);
        if observation.outcome.is_failure() {
            self.stage_failures_total.fetch_add(1, Ordering::Relaxed);
        }

        if let Some(total_ms) = observation.timing.total_ms {
            Self::push_sample_sync(&self.stage_duration_ms_samples, self.max_samples, total_ms);
        }
        Self::push_observation_sample_sync(
            &self.stage_observation_samples,
            self.max_samples,
            observation,
        );
    }

    pub(crate) async fn snapshot(&self) -> RuntimeTelemetrySnapshot {
        let queue = self.queue_wait_ms_samples.lock().await.clone();
        let prefill = self.prefill_ms_samples.lock().await.clone();
        let decode = self.decode_ms_samples.lock().await.clone();
        let ttft = self.ttft_ms_samples.lock().await.clone();
        let end_to_end = self.end_to_end_ms_samples.lock().await.clone();
        let stage_duration = self
            .stage_duration_ms_samples
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone();
        let recent_stage_samples = self
            .stage_observation_samples
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let workload_classes = workload_class_latency_snapshots(&recent_stage_samples);

        RuntimeTelemetrySnapshot {
            uptime_secs: self.start_time.elapsed().as_secs_f64(),
            requests_queued: self.requests_queued.load(Ordering::Relaxed),
            requests_completed: self.requests_completed.load(Ordering::Relaxed),
            requests_failed: self.requests_failed.load(Ordering::Relaxed),
            requests_active: self.requests_active.load(Ordering::Relaxed),
            worker_restarts: self.worker_restarts.load(Ordering::Relaxed),
            worker_panics: self.worker_panics.load(Ordering::Relaxed),
            queue_wait_ms_avg: mean(&queue),
            queue_wait_ms_p50: percentile(&queue, 0.50),
            queue_wait_ms_p95: percentile(&queue, 0.95),
            prefill_ms_avg: mean(&prefill),
            prefill_ms_p50: percentile(&prefill, 0.50),
            prefill_ms_p95: percentile(&prefill, 0.95),
            decode_ms_avg: mean(&decode),
            decode_ms_p50: percentile(&decode, 0.50),
            decode_ms_p95: percentile(&decode, 0.95),
            ttft_ms_avg: mean(&ttft),
            ttft_ms_p50: percentile(&ttft, 0.50),
            ttft_ms_p95: percentile(&ttft, 0.95),
            end_to_end_ms_avg: mean(&end_to_end),
            end_to_end_ms_p50: percentile(&end_to_end, 0.50),
            end_to_end_ms_p95: percentile(&end_to_end, 0.95),
            kernel_path: kernel_path_telemetry_snapshot(),
            engine: EngineRuntimeTelemetrySnapshot::default(),
            voice: VoiceRuntimeTelemetrySnapshot {
                sessions_started: self.voice_sessions_started.load(Ordering::Relaxed),
                sessions_closed: self.voice_sessions_closed.load(Ordering::Relaxed),
                interruptions: self.voice_interruptions.load(Ordering::Relaxed),
                barge_ins: self.voice_barge_ins.load(Ordering::Relaxed),
                stream_backpressure_total: self.voice_stream_backpressure.load(Ordering::Relaxed),
            },
            broker: InferenceBrokerRuntimeTelemetrySnapshot {
                shadow_requests: self.broker_shadow_requests.load(Ordering::Relaxed),
                execution_requests: self.broker_execution_requests.load(Ordering::Relaxed),
                route_decisions: self.broker_route_decisions.load(Ordering::Relaxed),
                validation_failures: self.broker_validation_failures.load(Ordering::Relaxed),
            },
            pipelines: PipelineRuntimeTelemetrySnapshot {
                modular_voice_turns: self.pipeline_modular_voice_turns.load(Ordering::Relaxed),
                unified_voice_turns: self.pipeline_unified_voice_turns.load(Ordering::Relaxed),
                diarization_transcripts: self
                    .pipeline_diarization_transcripts
                    .load(Ordering::Relaxed),
                batch_asr_transcriptions: self
                    .pipeline_batch_asr_transcriptions
                    .load(Ordering::Relaxed),
                batch_tts_speech: self.pipeline_batch_tts_speech.load(Ordering::Relaxed),
                stages_recorded: self.pipeline_stages_recorded.load(Ordering::Relaxed),
            },
            observability: RuntimeObservabilityTelemetrySnapshot {
                stage_observations_total: self.stage_observations_total.load(Ordering::Relaxed),
                stage_failures_total: self.stage_failures_total.load(Ordering::Relaxed),
                stage_duration_ms_avg: mean(&stage_duration),
                stage_duration_ms_p50: percentile(&stage_duration, 0.50),
                stage_duration_ms_p95: percentile(&stage_duration, 0.95),
                workload_classes,
                recent_stage_samples,
            },
            engine_metrics: engine_metric_catalog(),
            voice_metrics: voice_metric_catalog(),
        }
    }

    pub(crate) async fn prometheus(&self) -> String {
        let snapshot = self.snapshot().await;
        let mut payload = format!(
            "# TYPE izwi_requests_queued_total counter\nizwi_requests_queued_total {}\n\
# TYPE izwi_requests_completed_total counter\nizwi_requests_completed_total {}\n\
# TYPE izwi_requests_failed_total counter\nizwi_requests_failed_total {}\n\
# TYPE izwi_requests_active gauge\nizwi_requests_active {}\n\
# TYPE izwi_worker_restarts_total counter\nizwi_worker_restarts_total {}\n\
# TYPE izwi_worker_panics_total counter\nizwi_worker_panics_total {}\n\
# TYPE izwi_latency_queue_wait_ms gauge\nizwi_latency_queue_wait_ms{{quantile=\"avg\"}} {:.6}\nizwi_latency_queue_wait_ms{{quantile=\"p50\"}} {:.6}\nizwi_latency_queue_wait_ms{{quantile=\"p95\"}} {:.6}\n\
# TYPE izwi_latency_prefill_ms gauge\nizwi_latency_prefill_ms{{quantile=\"avg\"}} {:.6}\nizwi_latency_prefill_ms{{quantile=\"p50\"}} {:.6}\nizwi_latency_prefill_ms{{quantile=\"p95\"}} {:.6}\n\
# TYPE izwi_latency_decode_ms gauge\nizwi_latency_decode_ms{{quantile=\"avg\"}} {:.6}\nizwi_latency_decode_ms{{quantile=\"p50\"}} {:.6}\nizwi_latency_decode_ms{{quantile=\"p95\"}} {:.6}\n\
# TYPE izwi_latency_ttft_ms gauge\nizwi_latency_ttft_ms{{quantile=\"avg\"}} {:.6}\nizwi_latency_ttft_ms{{quantile=\"p50\"}} {:.6}\nizwi_latency_ttft_ms{{quantile=\"p95\"}} {:.6}\n\
# TYPE izwi_latency_end_to_end_ms gauge\nizwi_latency_end_to_end_ms{{quantile=\"avg\"}} {:.6}\nizwi_latency_end_to_end_ms{{quantile=\"p50\"}} {:.6}\nizwi_latency_end_to_end_ms{{quantile=\"p95\"}} {:.6}\n",
            snapshot.requests_queued,
            snapshot.requests_completed,
            snapshot.requests_failed,
            snapshot.requests_active,
            snapshot.worker_restarts,
            snapshot.worker_panics,
            snapshot.queue_wait_ms_avg,
            snapshot.queue_wait_ms_p50,
            snapshot.queue_wait_ms_p95,
            snapshot.prefill_ms_avg,
            snapshot.prefill_ms_p50,
            snapshot.prefill_ms_p95,
            snapshot.decode_ms_avg,
            snapshot.decode_ms_p50,
            snapshot.decode_ms_p95,
            snapshot.ttft_ms_avg,
            snapshot.ttft_ms_p50,
            snapshot.ttft_ms_p95,
            snapshot.end_to_end_ms_avg,
            snapshot.end_to_end_ms_p50,
            snapshot.end_to_end_ms_p95,
        );
        payload.push_str(&kernel_path_prometheus());
        push_voice_counter(
            &mut payload,
            VOICE_SESSION_STARTED_TOTAL,
            "Voice sessions started.",
            snapshot.voice.sessions_started,
        );
        push_voice_counter(
            &mut payload,
            VOICE_SESSION_CLOSED_TOTAL,
            "Voice sessions closed.",
            snapshot.voice.sessions_closed,
        );
        push_voice_counter(
            &mut payload,
            VOICE_SESSION_INTERRUPTED_TOTAL,
            "Voice turns interrupted before completion.",
            snapshot.voice.interruptions,
        );
        push_voice_counter(
            &mut payload,
            VOICE_BARGE_IN_TOTAL,
            "Voice barge-in interruptions.",
            snapshot.voice.barge_ins,
        );
        push_voice_counter(
            &mut payload,
            VOICE_STREAM_BACKPRESSURE_TOTAL,
            "Runtime stream backpressure events.",
            snapshot.voice.stream_backpressure_total,
        );
        payload.push_str(&format!(
            "# TYPE izwi_inference_broker_shadow_requests_total counter\nizwi_inference_broker_shadow_requests_total {}\n\
# TYPE izwi_inference_broker_execution_requests_total counter\nizwi_inference_broker_execution_requests_total {}\n\
# TYPE izwi_inference_broker_route_decisions_total counter\nizwi_inference_broker_route_decisions_total {}\n\
# TYPE izwi_inference_broker_validation_failures_total counter\nizwi_inference_broker_validation_failures_total {}\n",
            snapshot.broker.shadow_requests,
            snapshot.broker.execution_requests,
            snapshot.broker.route_decisions,
            snapshot.broker.validation_failures
        ));
        payload.push_str(&format!(
            "# TYPE izwi_inference_pipeline_modular_voice_turns_total counter\nizwi_inference_pipeline_modular_voice_turns_total {}\n\
# TYPE izwi_inference_pipeline_unified_voice_turns_total counter\nizwi_inference_pipeline_unified_voice_turns_total {}\n\
# TYPE izwi_inference_pipeline_diarization_transcripts_total counter\nizwi_inference_pipeline_diarization_transcripts_total {}\n\
# TYPE izwi_inference_pipeline_batch_asr_transcriptions_total counter\nizwi_inference_pipeline_batch_asr_transcriptions_total {}\n\
# TYPE izwi_inference_pipeline_batch_tts_speech_total counter\nizwi_inference_pipeline_batch_tts_speech_total {}\n\
# TYPE izwi_inference_pipeline_stages_recorded_total counter\nizwi_inference_pipeline_stages_recorded_total {}\n",
            snapshot.pipelines.modular_voice_turns,
            snapshot.pipelines.unified_voice_turns,
            snapshot.pipelines.diarization_transcripts,
            snapshot.pipelines.batch_asr_transcriptions,
            snapshot.pipelines.batch_tts_speech,
            snapshot.pipelines.stages_recorded
        ));
        payload.push_str(&format!(
            "# TYPE izwi_runtime_stage_observations_total counter\nizwi_runtime_stage_observations_total {}\n\
# TYPE izwi_runtime_stage_failures_total counter\nizwi_runtime_stage_failures_total {}\n\
# TYPE izwi_runtime_stage_duration_ms gauge\nizwi_runtime_stage_duration_ms{{quantile=\"avg\"}} {:.6}\nizwi_runtime_stage_duration_ms{{quantile=\"p50\"}} {:.6}\nizwi_runtime_stage_duration_ms{{quantile=\"p95\"}} {:.6}\n",
            snapshot.observability.stage_observations_total,
            snapshot.observability.stage_failures_total,
            snapshot.observability.stage_duration_ms_avg,
            snapshot.observability.stage_duration_ms_p50,
            snapshot.observability.stage_duration_ms_p95
        ));
        push_workload_class_prometheus(&mut payload, &snapshot.observability.workload_classes);
        payload.push_str(&voice_metric_prometheus_contract());
        payload
    }

    async fn push_sample(buffer: &Mutex<VecDeque<f64>>, max_samples: usize, value: f64) {
        let mut guard = buffer.lock().await;
        if guard.len() >= max_samples {
            guard.pop_front();
        }
        guard.push_back(value.max(0.0));
    }

    fn push_sample_sync(buffer: &StdMutex<VecDeque<f64>>, max_samples: usize, value: f64) {
        let mut guard = buffer.lock().unwrap_or_else(|poison| poison.into_inner());
        if guard.len() >= max_samples {
            guard.pop_front();
        }
        guard.push_back(value.max(0.0));
    }

    fn push_observation_sample_sync(
        buffer: &StdMutex<VecDeque<RuntimeStageObservation>>,
        max_samples: usize,
        value: RuntimeStageObservation,
    ) {
        let mut guard = buffer.lock().unwrap_or_else(|poison| poison.into_inner());
        if guard.len() >= max_samples {
            guard.pop_front();
        }
        guard.push_back(value);
    }
}

#[derive(Default)]
struct WorkloadClassLatencyAccumulator {
    observations: u64,
    failures: u64,
    queue_wait_ms: Vec<f64>,
    prefill_ms: Vec<f64>,
    decode_ms: Vec<f64>,
    ttft_ms: Vec<f64>,
    stage_duration_ms: Vec<f64>,
}

impl WorkloadClassLatencyAccumulator {
    fn record(&mut self, observation: &RuntimeStageObservation) {
        self.observations = self.observations.saturating_add(1);
        if observation.outcome.is_failure() {
            self.failures = self.failures.saturating_add(1);
        }
        push_optional_sample(&mut self.queue_wait_ms, observation.timing.queue_wait_ms);
        push_optional_sample(&mut self.prefill_ms, observation.timing.prefill_ms);
        push_optional_sample(&mut self.decode_ms, observation.timing.decode_ms);
        push_optional_sample(&mut self.ttft_ms, observation.timing.ttft_ms);
        push_optional_sample(&mut self.stage_duration_ms, observation.timing.total_ms);
    }

    fn into_snapshot(self, workload_class: String) -> RuntimeWorkloadClassTelemetrySnapshot {
        RuntimeWorkloadClassTelemetrySnapshot {
            workload_class,
            observations: self.observations,
            failures: self.failures,
            queue_wait_ms: RuntimeLatencyStats::from_slice(&self.queue_wait_ms),
            prefill_ms: RuntimeLatencyStats::from_slice(&self.prefill_ms),
            decode_ms: RuntimeLatencyStats::from_slice(&self.decode_ms),
            ttft_ms: RuntimeLatencyStats::from_slice(&self.ttft_ms),
            stage_duration_ms: RuntimeLatencyStats::from_slice(&self.stage_duration_ms),
        }
    }
}

fn workload_class_latency_snapshots(
    observations: &[RuntimeStageObservation],
) -> Vec<RuntimeWorkloadClassTelemetrySnapshot> {
    let mut by_class = BTreeMap::<String, WorkloadClassLatencyAccumulator>::new();
    for observation in observations {
        let Some(workload_class) = observation
            .context
            .workload_class
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        by_class
            .entry(workload_class.to_string())
            .or_default()
            .record(observation);
    }
    by_class
        .into_iter()
        .map(|(workload_class, accumulator)| accumulator.into_snapshot(workload_class))
        .collect()
}

fn push_optional_sample(samples: &mut Vec<f64>, value: Option<f64>) {
    if let Some(value) = value {
        samples.push(value.max(0.0));
    }
}

fn push_workload_class_prometheus(
    payload: &mut String,
    classes: &[RuntimeWorkloadClassTelemetrySnapshot],
) {
    if classes.is_empty() {
        return;
    }
    payload.push_str("# TYPE izwi_runtime_workload_stage_observations gauge\n");
    for class in classes {
        let label = prometheus_label_value(&class.workload_class);
        payload.push_str(&format!(
            "izwi_runtime_workload_stage_observations{{workload_class=\"{label}\"}} {}\n",
            class.observations
        ));
    }
    payload.push_str("# TYPE izwi_runtime_workload_stage_failures gauge\n");
    for class in classes {
        let label = prometheus_label_value(&class.workload_class);
        payload.push_str(&format!(
            "izwi_runtime_workload_stage_failures{{workload_class=\"{label}\"}} {}\n",
            class.failures
        ));
    }
    payload.push_str("# TYPE izwi_runtime_workload_queue_wait_ms gauge\n");
    for class in classes {
        push_workload_class_stats(
            payload,
            "izwi_runtime_workload_queue_wait_ms",
            class,
            &class.queue_wait_ms,
        );
    }
    payload.push_str("# TYPE izwi_runtime_workload_prefill_ms gauge\n");
    for class in classes {
        push_workload_class_stats(
            payload,
            "izwi_runtime_workload_prefill_ms",
            class,
            &class.prefill_ms,
        );
    }
    payload.push_str("# TYPE izwi_runtime_workload_decode_ms gauge\n");
    for class in classes {
        push_workload_class_stats(
            payload,
            "izwi_runtime_workload_decode_ms",
            class,
            &class.decode_ms,
        );
    }
    payload.push_str("# TYPE izwi_runtime_workload_ttft_ms gauge\n");
    for class in classes {
        push_workload_class_stats(
            payload,
            "izwi_runtime_workload_ttft_ms",
            class,
            &class.ttft_ms,
        );
    }
    payload.push_str("# TYPE izwi_runtime_workload_stage_duration_ms gauge\n");
    for class in classes {
        push_workload_class_stats(
            payload,
            "izwi_runtime_workload_stage_duration_ms",
            class,
            &class.stage_duration_ms,
        );
    }
}

fn push_workload_class_stats(
    payload: &mut String,
    metric_name: &str,
    class: &RuntimeWorkloadClassTelemetrySnapshot,
    stats: &RuntimeLatencyStats,
) {
    let workload_class = prometheus_label_value(&class.workload_class);
    for (quantile, value) in [("avg", stats.avg), ("p50", stats.p50), ("p95", stats.p95)] {
        payload.push_str(&format!(
            "{metric_name}{{workload_class=\"{workload_class}\",quantile=\"{quantile}\"}} {value:.6}\n"
        ));
    }
}

fn push_voice_counter(payload: &mut String, name: &str, help: &str, value: u64) {
    let prometheus_name = prometheus_voice_metric_name(name);
    payload.push_str(&format!(
        "# HELP {prometheus_name} {help}\n# TYPE {prometheus_name} counter\n{prometheus_name} {value}\n"
    ));
}

pub(crate) fn push_engine_metric(payload: &mut String, name: &str, value: u64) {
    let prometheus_name = prometheus_engine_metric_name(name);
    let metric_type = prometheus_engine_metric_type(name);
    payload.push_str(&format!(
        "# TYPE {prometheus_name} {metric_type}\n{prometheus_name} {value}\n"
    ));
}

pub(crate) fn push_engine_metric_f64(payload: &mut String, name: &str, value: f64) {
    let prometheus_name = prometheus_engine_metric_name(name);
    let metric_type = prometheus_engine_metric_type(name);
    payload.push_str(&format!(
        "# TYPE {prometheus_name} {metric_type}\n{prometheus_name} {value:.6}\n"
    ));
}

fn mean(values: &VecDeque<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn mean_slice(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn percentile(values: &VecDeque<f64>, q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = values.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len().saturating_sub(1)) as f64 * q.clamp(0.0, 1.0)) as usize;
    sorted[idx]
}

fn percentile_slice(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((sorted.len().saturating_sub(1)) as f64 * q.clamp(0.0, 1.0)) as usize;
    sorted[idx]
}

fn prometheus_label_value(value: &str) -> String {
    value
        .replace('\\', r"\\")
        .replace('"', r#"\""#)
        .replace('\n', r"\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ENGINE_SCHEDULER_QUEUE_DEPTH;

    #[tokio::test]
    async fn voice_telemetry_snapshot_and_prometheus_include_recorded_counters() {
        let telemetry = RuntimeTelemetryCollector::new(64);

        telemetry.record_voice_session_started();
        telemetry.record_voice_session_closed();
        telemetry.record_voice_interruption();
        telemetry.record_voice_barge_in();
        telemetry.record_voice_stream_backpressure();

        let snapshot = telemetry.snapshot().await;
        assert_eq!(snapshot.voice.sessions_started, 1);
        assert_eq!(snapshot.voice.sessions_closed, 1);
        assert_eq!(snapshot.voice.interruptions, 1);
        assert_eq!(snapshot.voice.barge_ins, 1);
        assert_eq!(snapshot.voice.stream_backpressure_total, 1);
        assert!(snapshot
            .voice_metrics
            .iter()
            .any(|metric| metric.name == VOICE_SESSION_STARTED_TOTAL));

        let payload = telemetry.prometheus().await;
        assert!(payload.contains("izwi_voice_session_started_total 1"));
        assert!(payload.contains("izwi_voice_session_closed_total 1"));
        assert!(payload.contains("izwi_voice_stream_backpressure_total 1"));
        assert!(payload.contains("izwi_voice_session_interruptions_total 1"));
        assert!(payload.contains("izwi_voice_barge_in_events_total 1"));
        assert!(payload.contains("izwi_voice_metric_contract_info"));
    }

    #[tokio::test]
    async fn broker_telemetry_snapshot_and_prometheus_include_recorded_counters() {
        let telemetry = RuntimeTelemetryCollector::new(64);

        telemetry.record_broker_shadow_request();
        telemetry.record_broker_execution_request();
        telemetry.record_broker_route_decision();
        telemetry.record_broker_validation_failure();

        let snapshot = telemetry.snapshot().await;
        assert_eq!(snapshot.broker.shadow_requests, 1);
        assert_eq!(snapshot.broker.execution_requests, 1);
        assert_eq!(snapshot.broker.route_decisions, 1);
        assert_eq!(snapshot.broker.validation_failures, 1);

        let payload = telemetry.prometheus().await;
        assert!(payload.contains("izwi_inference_broker_shadow_requests_total 1"));
        assert!(payload.contains("izwi_inference_broker_execution_requests_total 1"));
        assert!(payload.contains("izwi_inference_broker_route_decisions_total 1"));
        assert!(payload.contains("izwi_inference_broker_validation_failures_total 1"));
    }

    #[tokio::test]
    async fn pipeline_telemetry_snapshot_and_prometheus_include_recorded_counters() {
        let telemetry = RuntimeTelemetryCollector::new(64);

        telemetry.record_pipeline_graph(&PipelineGraph::modular_voice_turn());
        telemetry.record_pipeline_graph(&PipelineGraph::unified_voice_turn());
        telemetry.record_pipeline_graph(&PipelineGraph::diarization_transcript(true));
        telemetry.record_pipeline_graph(&PipelineGraph::batch_asr_transcription());
        telemetry.record_pipeline_graph(&PipelineGraph::batch_tts_speech());

        let snapshot = telemetry.snapshot().await;
        assert_eq!(snapshot.pipelines.modular_voice_turns, 1);
        assert_eq!(snapshot.pipelines.unified_voice_turns, 1);
        assert_eq!(snapshot.pipelines.diarization_transcripts, 1);
        assert_eq!(snapshot.pipelines.batch_asr_transcriptions, 1);
        assert_eq!(snapshot.pipelines.batch_tts_speech, 1);
        assert_eq!(snapshot.pipelines.stages_recorded, 22);

        let payload = telemetry.prometheus().await;
        assert!(payload.contains("izwi_inference_pipeline_modular_voice_turns_total 1"));
        assert!(payload.contains("izwi_inference_pipeline_unified_voice_turns_total 1"));
        assert!(payload.contains("izwi_inference_pipeline_diarization_transcripts_total 1"));
        assert!(payload.contains("izwi_inference_pipeline_batch_asr_transcriptions_total 1"));
        assert!(payload.contains("izwi_inference_pipeline_batch_tts_speech_total 1"));
        assert!(payload.contains("izwi_inference_pipeline_stages_recorded_total 22"));
    }

    #[tokio::test]
    async fn stage_observation_snapshot_and_prometheus_include_safe_aggregates() {
        let telemetry = RuntimeTelemetryCollector::new(64);
        let context = RuntimeObservationContext {
            route_source: Some("openai_audio_speech".to_string()),
            capability: Some("tts".to_string()),
            model_variant: Some("Kokoro-82M".to_string()),
            backend_kind: Some("cpu".to_string()),
            workload_class: Some("interactive".to_string()),
            pipeline_stage: Some("tts_synthesize".to_string()),
            request_id: Some("req-1".to_string()),
            correlation_id: Some("corr-1".to_string()),
            runtime_job_id: Some("job-1".to_string()),
            job_stage_id: Some("stage-1".to_string()),
            ..RuntimeObservationContext::default()
        };

        let mut completed = RuntimeStageObservation::new(context, RuntimeStageOutcome::Completed)
            .with_total_ms(42.0);
        completed.timing.queue_wait_ms = Some(5.0);
        completed.timing.prefill_ms = Some(7.0);
        completed.timing.decode_ms = Some(11.0);
        completed.timing.ttft_ms = Some(13.0);
        telemetry.record_stage_observation(completed);
        telemetry.record_stage_observation(
            RuntimeStageObservation::new(
                RuntimeObservationContext {
                    workload_class: Some("batch".to_string()),
                    pipeline_stage: Some("tts_synthesize".to_string()),
                    ..RuntimeObservationContext::default()
                },
                RuntimeStageOutcome::Failed,
            )
            .with_total_ms(100.0)
            .with_error_kind("executor_failed"),
        );

        let snapshot = telemetry.snapshot().await;
        assert_eq!(snapshot.observability.stage_observations_total, 2);
        assert_eq!(snapshot.observability.stage_failures_total, 1);
        assert_eq!(snapshot.observability.stage_duration_ms_avg, 71.0);
        assert_eq!(snapshot.observability.stage_duration_ms_p50, 42.0);
        assert_eq!(snapshot.observability.recent_stage_samples.len(), 2);
        let interactive = snapshot
            .observability
            .workload_classes
            .iter()
            .find(|class| class.workload_class == "interactive")
            .expect("interactive class aggregate");
        assert_eq!(interactive.observations, 1);
        assert_eq!(interactive.failures, 0);
        assert_eq!(interactive.queue_wait_ms.avg, 5.0);
        assert_eq!(interactive.prefill_ms.avg, 7.0);
        assert_eq!(interactive.decode_ms.avg, 11.0);
        assert_eq!(interactive.ttft_ms.avg, 13.0);
        assert_eq!(
            snapshot.observability.recent_stage_samples[0]
                .context
                .request_id
                .as_deref(),
            Some("req-1")
        );

        let payload = telemetry.prometheus().await;
        assert!(payload.contains("izwi_runtime_stage_observations_total 2"));
        assert!(payload.contains("izwi_runtime_stage_failures_total 1"));
        assert!(payload.contains("izwi_runtime_stage_duration_ms{quantile=\"avg\"} 71.000000"));
        assert!(payload.contains(
            "izwi_runtime_workload_stage_observations{workload_class=\"interactive\"} 1"
        ));
        assert!(payload.contains(
            "izwi_runtime_workload_queue_wait_ms{workload_class=\"interactive\",quantile=\"avg\"} 5.000000"
        ));
        assert!(
            payload.contains("izwi_runtime_workload_stage_failures{workload_class=\"batch\"} 1")
        );
        assert!(!payload.contains("req-1"));
        assert!(!payload.contains("job-1"));
    }

    #[tokio::test]
    async fn stage_observation_samples_are_bounded() {
        let telemetry = RuntimeTelemetryCollector::new(64);

        for idx in 0..70 {
            telemetry.record_stage_observation(RuntimeStageObservation::new(
                RuntimeObservationContext {
                    request_id: Some(format!("req-{idx}")),
                    ..RuntimeObservationContext::default()
                },
                RuntimeStageOutcome::Observed,
            ));
        }

        let snapshot = telemetry.snapshot().await;
        assert_eq!(snapshot.observability.stage_observations_total, 70);
        assert_eq!(snapshot.observability.recent_stage_samples.len(), 64);
        assert_eq!(
            snapshot.observability.recent_stage_samples[0]
                .context
                .request_id
                .as_deref(),
            Some("req-6")
        );
    }

    #[test]
    fn stage_observation_contract_is_metadata_only() {
        let observation = RuntimeStageObservation::new(
            RuntimeObservationContext {
                route_source: Some("openai_audio_transcriptions".to_string()),
                capability: Some("asr".to_string()),
                request_id: Some("req-redacted".to_string()),
                ..RuntimeObservationContext::default()
            },
            RuntimeStageOutcome::Completed,
        );

        let payload = serde_json::to_string(&observation).expect("serialize observation");
        assert!(payload.contains("openai_audio_transcriptions"));
        assert!(!payload.contains("prompt"));
        assert!(!payload.contains("transcript_text"));
        assert!(!payload.contains("audio_samples"));
        assert!(!payload.contains("reference_audio"));
    }

    #[test]
    fn engine_metric_prometheus_helper_uses_catalog_name() {
        let mut payload = String::new();
        push_engine_metric(&mut payload, ENGINE_SCHEDULER_QUEUE_DEPTH, 7);

        assert!(payload.contains("izwi_engine_scheduler_queue_depth 7"));
    }
}
