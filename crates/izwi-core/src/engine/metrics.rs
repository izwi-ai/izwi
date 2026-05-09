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

/// Stable metric names for scheduler and KV-cache observability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct EngineMetricDescriptor {
    pub name: &'static str,
    pub description: &'static str,
}

pub const ENGINE_SCHEDULER_QUEUE_DEPTH: &str = "engine.scheduler.queue_depth";
pub const ENGINE_SCHEDULER_RUNNING_REQUESTS: &str = "engine.scheduler.running_requests";
pub const ENGINE_SCHEDULER_PREEMPTIONS_TOTAL: &str = "engine.scheduler.preemptions_total";
pub const ENGINE_SCHEDULER_STEP_TOKENS_TOTAL: &str = "engine.scheduler.step_tokens_total";
pub const ENGINE_KV_CACHE_HITS_TOTAL: &str = "engine.kv_cache.hits_total";
pub const ENGINE_KV_CACHE_MISSES_TOTAL: &str = "engine.kv_cache.misses_total";
pub const ENGINE_KV_CACHE_EVICTIONS_TOTAL: &str = "engine.kv_cache.evictions_total";
pub const ENGINE_KV_CACHE_ALLOCATED_BLOCKS: &str = "engine.kv_cache.allocated_blocks";
pub const ENGINE_KV_CACHE_PREFIX_REUSE_BLOCKS_TOTAL: &str =
    "engine.kv_cache.prefix_reuse_blocks_total";
pub const ENGINE_STREAM_BACKPRESSURE_TOTAL: &str = "engine.stream.backpressure_total";

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
        name: ENGINE_SCHEDULER_PREEMPTIONS_TOTAL,
        description: "Scheduler preemptions caused by priority or KV pressure.",
    },
    EngineMetricDescriptor {
        name: ENGINE_SCHEDULER_STEP_TOKENS_TOTAL,
        description: "Tokens admitted into scheduler execution steps.",
    },
    EngineMetricDescriptor {
        name: ENGINE_KV_CACHE_HITS_TOTAL,
        description: "KV-cache or prefix-cache hits.",
    },
    EngineMetricDescriptor {
        name: ENGINE_KV_CACHE_MISSES_TOTAL,
        description: "KV-cache or prefix-cache misses.",
    },
    EngineMetricDescriptor {
        name: ENGINE_KV_CACHE_EVICTIONS_TOTAL,
        description: "KV-cache evictions labeled by reason when emitted.",
    },
    EngineMetricDescriptor {
        name: ENGINE_KV_CACHE_ALLOCATED_BLOCKS,
        description: "Currently allocated KV-cache blocks.",
    },
    EngineMetricDescriptor {
        name: ENGINE_KV_CACHE_PREFIX_REUSE_BLOCKS_TOTAL,
        description: "Prompt blocks reused from prefix cache.",
    },
    EngineMetricDescriptor {
        name: ENGINE_STREAM_BACKPRESSURE_TOTAL,
        description: "Engine stream backpressure events.",
    },
];

static ENGINE_STREAM_BACKPRESSURE_EVENTS: AtomicU64 = AtomicU64::new(0);

pub fn engine_metric_catalog() -> &'static [EngineMetricDescriptor] {
    ENGINE_METRIC_CATALOG
}

pub(crate) fn record_engine_stream_backpressure() {
    ENGINE_STREAM_BACKPRESSURE_EVENTS.fetch_add(1, Ordering::Relaxed);
}

pub fn engine_stream_backpressure_total() -> u64 {
    ENGINE_STREAM_BACKPRESSURE_EVENTS.load(Ordering::Relaxed)
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
        assert!(names.contains(ENGINE_SCHEDULER_PREEMPTIONS_TOTAL));
        assert!(names.contains(ENGINE_KV_CACHE_HITS_TOTAL));
        assert!(names.contains(ENGINE_KV_CACHE_EVICTIONS_TOTAL));
        assert!(names.contains(ENGINE_STREAM_BACKPRESSURE_TOTAL));
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
    fn engine_stream_backpressure_counter_is_observable() {
        let before = engine_stream_backpressure_total();
        record_engine_stream_backpressure();
        assert_eq!(engine_stream_backpressure_total(), before + 1);
    }
}
