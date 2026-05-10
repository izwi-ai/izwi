//! Runtime metrics, snapshots, and Prometheus formatting.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
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
use crate::runtime::pipeline::{PipelineExecutionSummary, PipelineExecutor, PipelineGraph, PipelineKind};
use crate::runtime::voice_metrics::{
    prometheus_voice_metric_name, voice_metric_catalog, voice_metric_prometheus_contract,
    VoiceMetricDescriptor, VOICE_BARGE_IN_TOTAL, VOICE_SESSION_CLOSED_TOTAL,
    VOICE_SESSION_INTERRUPTED_TOTAL, VOICE_SESSION_STARTED_TOTAL,
};
use crate::KernelPathTelemetrySnapshot;

#[derive(Debug, Clone, Serialize)]
pub struct VoiceRuntimeTelemetrySnapshot {
    pub sessions_started: u64,
    pub sessions_closed: u64,
    pub interruptions: u64,
    pub barge_ins: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct InferenceBrokerRuntimeTelemetrySnapshot {
    pub shadow_requests: u64,
    pub execution_requests: u64,
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
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct PipelineRuntimeTelemetrySnapshot {
    pub modular_voice_turns: u64,
    pub unified_voice_turns: u64,
    pub diarization_transcripts: u64,
    pub stages_recorded: u64,
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
    broker_shadow_requests: AtomicU64,
    broker_execution_requests: AtomicU64,
    broker_validation_failures: AtomicU64,
    pipeline_modular_voice_turns: AtomicU64,
    pipeline_unified_voice_turns: AtomicU64,
    pipeline_diarization_transcripts: AtomicU64,
    pipeline_stages_recorded: AtomicU64,
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
            broker_shadow_requests: AtomicU64::new(0),
            broker_execution_requests: AtomicU64::new(0),
            broker_validation_failures: AtomicU64::new(0),
            pipeline_modular_voice_turns: AtomicU64::new(0),
            pipeline_unified_voice_turns: AtomicU64::new(0),
            pipeline_diarization_transcripts: AtomicU64::new(0),
            pipeline_stages_recorded: AtomicU64::new(0),
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

    pub(crate) fn record_broker_shadow_request(&self) {
        self.broker_shadow_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_broker_execution_request(&self) {
        self.broker_execution_requests.fetch_add(1, Ordering::Relaxed);
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
        }
        self.pipeline_stages_recorded
            .fetch_add(summary.stages().len() as u64, Ordering::Relaxed);
    }

    pub(crate) async fn snapshot(&self) -> RuntimeTelemetrySnapshot {
        let queue = self.queue_wait_ms_samples.lock().await.clone();
        let prefill = self.prefill_ms_samples.lock().await.clone();
        let decode = self.decode_ms_samples.lock().await.clone();
        let ttft = self.ttft_ms_samples.lock().await.clone();
        let end_to_end = self.end_to_end_ms_samples.lock().await.clone();

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
            },
            broker: InferenceBrokerRuntimeTelemetrySnapshot {
                shadow_requests: self.broker_shadow_requests.load(Ordering::Relaxed),
                execution_requests: self.broker_execution_requests.load(Ordering::Relaxed),
                validation_failures: self.broker_validation_failures.load(Ordering::Relaxed),
            },
            pipelines: PipelineRuntimeTelemetrySnapshot {
                modular_voice_turns: self.pipeline_modular_voice_turns.load(Ordering::Relaxed),
                unified_voice_turns: self.pipeline_unified_voice_turns.load(Ordering::Relaxed),
                diarization_transcripts: self
                    .pipeline_diarization_transcripts
                    .load(Ordering::Relaxed),
                stages_recorded: self.pipeline_stages_recorded.load(Ordering::Relaxed),
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
        payload.push_str(&format!(
            "# TYPE izwi_inference_broker_shadow_requests_total counter\nizwi_inference_broker_shadow_requests_total {}\n\
# TYPE izwi_inference_broker_execution_requests_total counter\nizwi_inference_broker_execution_requests_total {}\n\
# TYPE izwi_inference_broker_validation_failures_total counter\nizwi_inference_broker_validation_failures_total {}\n",
            snapshot.broker.shadow_requests,
            snapshot.broker.execution_requests,
            snapshot.broker.validation_failures
        ));
        payload.push_str(&format!(
            "# TYPE izwi_inference_pipeline_modular_voice_turns_total counter\nizwi_inference_pipeline_modular_voice_turns_total {}\n\
# TYPE izwi_inference_pipeline_unified_voice_turns_total counter\nizwi_inference_pipeline_unified_voice_turns_total {}\n\
# TYPE izwi_inference_pipeline_diarization_transcripts_total counter\nizwi_inference_pipeline_diarization_transcripts_total {}\n\
# TYPE izwi_inference_pipeline_stages_recorded_total counter\nizwi_inference_pipeline_stages_recorded_total {}\n",
            snapshot.pipelines.modular_voice_turns,
            snapshot.pipelines.unified_voice_turns,
            snapshot.pipelines.diarization_transcripts,
            snapshot.pipelines.stages_recorded
        ));
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

fn mean(values: &VecDeque<f64>) -> f64 {
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

        let snapshot = telemetry.snapshot().await;
        assert_eq!(snapshot.voice.sessions_started, 1);
        assert_eq!(snapshot.voice.sessions_closed, 1);
        assert_eq!(snapshot.voice.interruptions, 1);
        assert_eq!(snapshot.voice.barge_ins, 1);
        assert!(snapshot
            .voice_metrics
            .iter()
            .any(|metric| metric.name == VOICE_SESSION_STARTED_TOTAL));

        let payload = telemetry.prometheus().await;
        assert!(payload.contains("izwi_voice_session_started_total 1"));
        assert!(payload.contains("izwi_voice_session_closed_total 1"));
        assert!(payload.contains("izwi_voice_session_interruptions_total 1"));
        assert!(payload.contains("izwi_voice_barge_in_events_total 1"));
        assert!(payload.contains("izwi_voice_metric_contract_info"));
    }

    #[tokio::test]
    async fn broker_telemetry_snapshot_and_prometheus_include_recorded_counters() {
        let telemetry = RuntimeTelemetryCollector::new(64);

        telemetry.record_broker_shadow_request();
        telemetry.record_broker_execution_request();
        telemetry.record_broker_validation_failure();

        let snapshot = telemetry.snapshot().await;
        assert_eq!(snapshot.broker.shadow_requests, 1);
        assert_eq!(snapshot.broker.execution_requests, 1);
        assert_eq!(snapshot.broker.validation_failures, 1);

        let payload = telemetry.prometheus().await;
        assert!(payload.contains("izwi_inference_broker_shadow_requests_total 1"));
        assert!(payload.contains("izwi_inference_broker_execution_requests_total 1"));
        assert!(payload.contains("izwi_inference_broker_validation_failures_total 1"));
    }

    #[tokio::test]
    async fn pipeline_telemetry_snapshot_and_prometheus_include_recorded_counters() {
        let telemetry = RuntimeTelemetryCollector::new(64);

        telemetry.record_pipeline_graph(&PipelineGraph::modular_voice_turn());
        telemetry.record_pipeline_graph(&PipelineGraph::unified_voice_turn());
        telemetry.record_pipeline_graph(&PipelineGraph::diarization_transcript(true));

        let snapshot = telemetry.snapshot().await;
        assert_eq!(snapshot.pipelines.modular_voice_turns, 1);
        assert_eq!(snapshot.pipelines.unified_voice_turns, 1);
        assert_eq!(snapshot.pipelines.diarization_transcripts, 1);
        assert_eq!(snapshot.pipelines.stages_recorded, 14);

        let payload = telemetry.prometheus().await;
        assert!(payload.contains("izwi_inference_pipeline_modular_voice_turns_total 1"));
        assert!(payload.contains("izwi_inference_pipeline_unified_voice_turns_total 1"));
        assert!(payload.contains("izwi_inference_pipeline_diarization_transcripts_total 1"));
        assert!(payload.contains("izwi_inference_pipeline_stages_recorded_total 14"));
    }

    #[test]
    fn engine_metric_prometheus_helper_uses_catalog_name() {
        let mut payload = String::new();
        push_engine_metric(&mut payload, ENGINE_SCHEDULER_QUEUE_DEPTH, 7);

        assert!(payload.contains("izwi_engine_scheduler_queue_depth 7"));
    }
}
