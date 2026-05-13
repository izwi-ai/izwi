use crate::error::{CliError, Result};
use crate::http;
use crate::style::Theme;
use crate::{BenchCommands, OutputFormat};
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use chrono::{DateTime, Utc};
use futures::{stream, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeTelemetryContext {
    Default,
    AsrWhisper,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct RuntimeTelemetrySnapshot {
    requests_queued: u64,
    requests_completed: u64,
    requests_failed: u64,
    requests_active: u64,
    worker_restarts: u64,
    worker_panics: u64,
    queue_wait_ms_avg: f64,
    queue_wait_ms_p50: f64,
    queue_wait_ms_p95: f64,
    prefill_ms_avg: f64,
    prefill_ms_p50: f64,
    prefill_ms_p95: f64,
    decode_ms_avg: f64,
    decode_ms_p50: f64,
    decode_ms_p95: f64,
    ttft_ms_avg: f64,
    ttft_ms_p50: f64,
    ttft_ms_p95: f64,
    end_to_end_ms_avg: f64,
    end_to_end_ms_p50: f64,
    end_to_end_ms_p95: f64,
    #[serde(default)]
    kernel_path: KernelPathTelemetrySnapshot,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
struct KernelPathTelemetrySnapshot {
    prefill_token_mode_steps_total: u64,
    prefill_sequence_spans_total: u64,
    prefill_sequence_tokens_total: u64,
    decode_attention_dense_total: u64,
    decode_attention_paged_total: u64,
    #[serde(default)]
    chunk_attention_sequence_calls_total: u64,
    #[serde(default)]
    chunk_attention_spans_total: u64,
    #[serde(default)]
    chunk_attention_tokens_total: u64,
    #[serde(default)]
    chunk_attention_fused_spans_total: u64,
    #[serde(default)]
    chunk_attention_unfused_spans_total: u64,
    #[serde(default)]
    chunk_attention_mask_fallback_total: u64,
    rope_kernel_total: u64,
    rope_manual_total: u64,
    fused_attention_attempts_total: u64,
    fused_attention_success_total: u64,
    fused_attention_fallback_total: u64,
    #[serde(default)]
    fused_attention_masked_attempts_total: u64,
    #[serde(default)]
    fused_attention_masked_success_total: u64,
    #[serde(default)]
    fused_attention_masked_fallback_total: u64,
    #[serde(default)]
    fused_attention_fallback_metal_sdpa_mask_policy_disabled_total: u64,
    #[serde(default)]
    fused_attention_fallback_metal_sdpa_mask_shape_unsupported_total: u64,
    #[serde(default)]
    fused_attention_fallback_metal_sdpa_mask_dtype_unsupported_total: u64,
}

#[derive(Debug, Clone)]
struct ChatBenchSample {
    ttft_ms: f64,
    total_ms: f64,
    prompt_tokens: usize,
    completion_tokens: usize,
    generation_time_ms: Option<f64>,
}

#[derive(Debug, Clone)]
struct TtsBenchSample {
    total_ms: f64,
    generation_time_ms: Option<f64>,
    audio_duration_secs: Option<f64>,
    rtf: Option<f64>,
    tokens_generated: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct AsrBenchResponse {
    #[serde(default)]
    duration: Option<f64>,
    #[serde(default)]
    processing_time_ms: Option<f64>,
    #[serde(default)]
    rtf: Option<f64>,
    #[serde(default)]
    izwi_asr_diagnostics: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
struct AsrBenchSample {
    total_ms: f64,
    response: AsrBenchResponse,
}

#[derive(Debug, Clone, Copy)]
struct WhisperStageTimings {
    audio_decode: Option<f64>,
    mel_prepare: Option<f64>,
    encoder_forward: Option<f64>,
    language_detect: Option<f64>,
    decode: Option<f64>,
    model_total: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ChatStreamChunk {
    choices: Vec<ChatStreamChoice>,
    usage: Option<ChatStreamUsage>,
    izwi_generation_time_ms: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ChatStreamChoice {
    delta: ChatStreamDelta,
}

#[derive(Debug, Default, Deserialize)]
struct ChatStreamDelta {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChatStreamUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
}

#[derive(Clone)]
struct BenchOptions {
    output_format: OutputFormat,
    quiet: bool,
}

impl BenchOptions {
    fn human_output(&self) -> bool {
        !matches!(self.output_format, OutputFormat::Json)
    }

    fn interactive(&self) -> bool {
        self.human_output() && !self.quiet
    }
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    schema_version: u32,
    command: &'static str,
    server: String,
    started_at: DateTime<Utc>,
    ended_at: DateTime<Utc>,
    duration_ms: f64,
    config: BenchmarkRunConfig,
    summary: BenchmarkSummary,
    samples: Vec<BenchmarkSample>,
    telemetry: RuntimeTelemetryReport,
}

#[derive(Debug, Serialize)]
struct BenchmarkRunConfig {
    model: Option<String>,
    iterations: Option<u32>,
    concurrent: Option<u32>,
    warmup: bool,
    prompt: Option<String>,
    system: Option<String>,
    max_tokens: Option<usize>,
    text: Option<String>,
    file: Option<String>,
    language: Option<String>,
    duration_secs: Option<u64>,
}

#[derive(Debug, Serialize)]
struct BenchmarkSummary {
    latency_ms: Option<Stats>,
    ttft_ms: Option<Stats>,
    end_to_end_ms: Option<Stats>,
    completion_tps: Option<Stats>,
    tokens_per_second: Option<Stats>,
    server_generation_ms: Option<Stats>,
    server_processing_ms: Option<Stats>,
    audio_duration_secs: Option<Stats>,
    rtf: Option<Stats>,
    prompt_tokens_avg: Option<f64>,
    completion_tokens_avg: Option<f64>,
    throughput_rps: Option<f64>,
    successful: Option<u64>,
    failed: Option<u64>,
    total: Option<u64>,
}

#[derive(Debug, Serialize)]
struct Stats {
    count: usize,
    avg: f64,
    min: f64,
    max: f64,
    p50: f64,
    p95: f64,
    p99: f64,
}

#[derive(Debug, Serialize)]
struct BenchmarkSample {
    index: usize,
    latency_ms: Option<f64>,
    ttft_ms: Option<f64>,
    end_to_end_ms: Option<f64>,
    completion_tps: Option<f64>,
    tokens_per_second: Option<f64>,
    prompt_tokens: Option<usize>,
    completion_tokens: Option<usize>,
    server_generation_ms: Option<f64>,
    server_processing_ms: Option<f64>,
    audio_duration_secs: Option<f64>,
    rtf: Option<f64>,
    tokens_generated: Option<u64>,
}

#[derive(Debug, Serialize)]
struct RuntimeTelemetryReport {
    delta_available: bool,
    before: Option<RuntimeTelemetrySnapshot>,
    after: Option<RuntimeTelemetrySnapshot>,
}

#[derive(Debug, Deserialize)]
struct BenchmarkManifest {
    server: Option<String>,
    benchmarks: Vec<BenchmarkManifestCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct BenchmarkManifestCase {
    name: Option<String>,
    command: String,
    model: Option<String>,
    iterations: Option<u32>,
    concurrent: Option<u32>,
    warmup: Option<bool>,
    prompt: Option<String>,
    system: Option<String>,
    max_tokens: Option<usize>,
    text: Option<String>,
    file: Option<String>,
    language: Option<String>,
    duration_secs: Option<u64>,
    matrix: Option<BenchmarkManifestMatrix>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct BenchmarkManifestMatrix {
    model: Option<Vec<String>>,
    iterations: Option<Vec<u32>>,
    concurrent: Option<Vec<u32>>,
    warmup: Option<Vec<bool>>,
    prompt: Option<Vec<String>>,
    system: Option<Vec<String>>,
    max_tokens: Option<Vec<usize>>,
    text: Option<Vec<String>>,
    file: Option<Vec<String>>,
    language: Option<Vec<String>>,
    duration_secs: Option<Vec<u64>>,
}

#[derive(Debug, Clone)]
struct MatrixDimension {
    key: &'static str,
    values: Vec<MatrixValue>,
}

#[derive(Debug, Clone)]
enum MatrixValue {
    Model(String),
    Iterations(u32),
    Concurrent(u32),
    Warmup(bool),
    Prompt(String),
    System(String),
    MaxTokens(usize),
    Text(String),
    File(String),
    Language(String),
    DurationSecs(u64),
}

#[derive(Debug, Serialize)]
struct BenchmarkSuiteReport {
    schema_version: u32,
    manifest: String,
    server: String,
    started_at: DateTime<Utc>,
    ended_at: DateTime<Utc>,
    reports: Vec<BenchmarkSuiteCaseReport>,
}

#[derive(Debug, Serialize)]
struct BenchmarkSuiteCaseReport {
    name: Option<String>,
    report: BenchmarkReport,
}

#[derive(Debug, Serialize)]
struct BenchmarkArtifactMetadata {
    schema_version: u32,
    cli_version: &'static str,
    server: String,
    manifest: String,
    git_sha: Option<String>,
    os: &'static str,
    arch: &'static str,
    started_at: DateTime<Utc>,
    ended_at: DateTime<Utc>,
}

#[derive(Debug, Serialize)]
struct BenchmarkObservabilityBundle {
    before: ObservabilitySnapshot,
    after: ObservabilitySnapshot,
}

#[derive(Debug, Serialize)]
struct ObservabilitySnapshot {
    captured_at: DateTime<Utc>,
    health: Option<serde_json::Value>,
    metrics: Option<serde_json::Value>,
    prometheus: Option<String>,
}

#[derive(Debug, Serialize)]
struct BenchmarkCompareReport {
    schema_version: u32,
    current: String,
    baseline: String,
    tolerance_percent: f64,
    regressions: usize,
    checks: Vec<BenchmarkComparison>,
}

#[derive(Debug, Serialize)]
struct BenchmarkComparison {
    case: String,
    metric: String,
    baseline: f64,
    current: f64,
    change_percent: f64,
    lower_is_better: bool,
    status: &'static str,
}

#[derive(Debug, Clone)]
struct ReportEntry {
    name: String,
    summary: serde_json::Value,
}

pub async fn execute(
    command: BenchCommands,
    server: &str,
    output_format: OutputFormat,
    quiet: bool,
    theme: &Theme,
) -> Result<()> {
    let options = BenchOptions {
        output_format,
        quiet,
    };
    match command {
        BenchCommands::Chat {
            model,
            iterations,
            prompt,
            system,
            max_tokens,
            concurrent,
            warmup,
        } => bench_chat(
            server,
            &model,
            iterations,
            &prompt,
            system.as_deref(),
            max_tokens,
            concurrent,
            warmup,
            &options,
            theme,
        )
        .await
        .and_then(|report| emit_report(&options, &report)),
        BenchCommands::Tts {
            model,
            iterations,
            text,
            concurrent,
            warmup,
        } => bench_tts(
            server, &model, iterations, &text, concurrent, warmup, &options, theme,
        )
        .await
        .and_then(|report| emit_report(&options, &report)),
        BenchCommands::Asr {
            model,
            iterations,
            file,
            language,
            concurrent,
            warmup,
        } => bench_asr(
            server,
            &model,
            iterations,
            file,
            language.as_deref(),
            concurrent,
            warmup,
            &options,
            theme,
        )
        .await
        .and_then(|report| emit_report(&options, &report)),
        BenchCommands::Throughput {
            duration,
            concurrent,
        } => bench_throughput(server, duration, concurrent, &options, theme)
            .await
            .and_then(|report| emit_report(&options, &report)),
        BenchCommands::Run {
            manifest,
            artifact_dir,
        } => bench_manifest(server, &manifest, artifact_dir.as_deref(), &options, theme).await,
        BenchCommands::Compare {
            current,
            baseline,
            tolerance_percent,
        } => bench_compare(&current, &baseline, tolerance_percent, &options).await,
    }
}

async fn bench_compare(
    current_path: &Path,
    baseline_path: &Path,
    tolerance_percent: f64,
    options: &BenchOptions,
) -> Result<()> {
    if !tolerance_percent.is_finite() || tolerance_percent < 0.0 {
        return Err(CliError::InvalidInput(
            "Comparison tolerance must be a non-negative percentage".to_string(),
        ));
    }

    let current_value = read_json_report(current_path).await?;
    let baseline_value = read_json_report(baseline_path).await?;
    let current_reports = report_entry_map(report_entries(&current_value)?, "Current")?;
    let baseline_reports = report_entry_map(report_entries(&baseline_value)?, "Baseline")?;
    if current_reports.len() != baseline_reports.len() {
        return Err(CliError::InvalidInput(format!(
            "Report shape mismatch: current has {} case(s), baseline has {} case(s)",
            current_reports.len(),
            baseline_reports.len()
        )));
    }
    let current_names: BTreeSet<_> = current_reports.keys().cloned().collect();
    let baseline_names: BTreeSet<_> = baseline_reports.keys().cloned().collect();
    if current_names != baseline_names {
        let only_current: Vec<_> = current_names.difference(&baseline_names).cloned().collect();
        let only_baseline: Vec<_> = baseline_names.difference(&current_names).cloned().collect();
        return Err(CliError::InvalidInput(format!(
            "Report case mismatch: only in current: {}; only in baseline: {}",
            format_case_list(&only_current),
            format_case_list(&only_baseline)
        )));
    }

    let mut checks = Vec::new();
    for (case, current) in &current_reports {
        let baseline = baseline_reports
            .get(case)
            .expect("case set equality should guarantee baseline entry");
        collect_metric_comparisons(
            case,
            &current.summary,
            &baseline.summary,
            tolerance_percent,
            &mut checks,
        );
    }

    if checks.is_empty() {
        return Err(CliError::InvalidInput(
            "No comparable benchmark summary metrics found".to_string(),
        ));
    }

    let regressions = checks
        .iter()
        .filter(|check| check.status == "regression")
        .count();
    let report = BenchmarkCompareReport {
        schema_version: 1,
        current: current_path.display().to_string(),
        baseline: baseline_path.display().to_string(),
        tolerance_percent,
        regressions,
        checks,
    };

    emit_compare_report(options, &report)?;
    if regressions > 0 {
        return Err(CliError::Other(format!(
            "Benchmark comparison failed: {regressions} regression(s) exceeded {tolerance_percent:.2}% tolerance"
        )));
    }

    Ok(())
}

fn report_entry_map(
    entries: Vec<ReportEntry>,
    label: &str,
) -> Result<BTreeMap<String, ReportEntry>> {
    let mut map = BTreeMap::new();
    for entry in entries {
        let name = entry.name.clone();
        if map.insert(name.clone(), entry).is_some() {
            return Err(CliError::InvalidInput(format!(
                "{label} report has duplicate benchmark case name `{name}`"
            )));
        }
    }
    Ok(map)
}

fn format_case_list(cases: &[String]) -> String {
    if cases.is_empty() {
        "none".to_string()
    } else {
        cases.join(", ")
    }
}

impl MatrixValue {
    fn apply(&self, case: &mut BenchmarkManifestCase) {
        match self {
            MatrixValue::Model(value) => case.model = Some(value.clone()),
            MatrixValue::Iterations(value) => case.iterations = Some(*value),
            MatrixValue::Concurrent(value) => case.concurrent = Some(*value),
            MatrixValue::Warmup(value) => case.warmup = Some(*value),
            MatrixValue::Prompt(value) => case.prompt = Some(value.clone()),
            MatrixValue::System(value) => case.system = Some(value.clone()),
            MatrixValue::MaxTokens(value) => case.max_tokens = Some(*value),
            MatrixValue::Text(value) => case.text = Some(value.clone()),
            MatrixValue::File(value) => case.file = Some(value.clone()),
            MatrixValue::Language(value) => case.language = Some(value.clone()),
            MatrixValue::DurationSecs(value) => case.duration_secs = Some(*value),
        }
    }

    fn label_value(&self) -> String {
        match self {
            MatrixValue::Model(value)
            | MatrixValue::Prompt(value)
            | MatrixValue::System(value)
            | MatrixValue::Text(value)
            | MatrixValue::File(value)
            | MatrixValue::Language(value) => matrix_label_string(value),
            MatrixValue::Iterations(value) => value.to_string(),
            MatrixValue::Concurrent(value) => value.to_string(),
            MatrixValue::MaxTokens(value) => value.to_string(),
            MatrixValue::DurationSecs(value) => value.to_string(),
            MatrixValue::Warmup(value) => value.to_string(),
        }
    }
}

impl BenchmarkManifestMatrix {
    fn dimensions(&self) -> Result<Vec<MatrixDimension>> {
        let mut dimensions = Vec::new();
        add_matrix_dimension(&mut dimensions, "model", &self.model, MatrixValue::Model)?;
        add_matrix_dimension(
            &mut dimensions,
            "iterations",
            &self.iterations,
            MatrixValue::Iterations,
        )?;
        add_matrix_dimension(
            &mut dimensions,
            "concurrent",
            &self.concurrent,
            MatrixValue::Concurrent,
        )?;
        add_matrix_dimension(&mut dimensions, "warmup", &self.warmup, MatrixValue::Warmup)?;
        add_matrix_dimension(&mut dimensions, "prompt", &self.prompt, MatrixValue::Prompt)?;
        add_matrix_dimension(&mut dimensions, "system", &self.system, MatrixValue::System)?;
        add_matrix_dimension(
            &mut dimensions,
            "max_tokens",
            &self.max_tokens,
            MatrixValue::MaxTokens,
        )?;
        add_matrix_dimension(&mut dimensions, "text", &self.text, MatrixValue::Text)?;
        add_matrix_dimension(&mut dimensions, "file", &self.file, MatrixValue::File)?;
        add_matrix_dimension(
            &mut dimensions,
            "language",
            &self.language,
            MatrixValue::Language,
        )?;
        add_matrix_dimension(
            &mut dimensions,
            "duration_secs",
            &self.duration_secs,
            MatrixValue::DurationSecs,
        )?;
        if dimensions.is_empty() {
            return Err(CliError::InvalidInput(
                "Benchmark matrix must include at least one non-empty field".to_string(),
            ));
        }
        Ok(dimensions)
    }
}

fn add_matrix_dimension<T, F>(
    dimensions: &mut Vec<MatrixDimension>,
    key: &'static str,
    values: &Option<Vec<T>>,
    map: F,
) -> Result<()>
where
    T: Clone,
    F: Fn(T) -> MatrixValue,
{
    let Some(values) = values else {
        return Ok(());
    };
    if values.is_empty() {
        return Err(CliError::InvalidInput(format!(
            "Benchmark matrix field `{key}` must contain at least one value"
        )));
    }
    dimensions.push(MatrixDimension {
        key,
        values: values.iter().cloned().map(map).collect(),
    });
    Ok(())
}

fn matrix_label_string(value: &str) -> String {
    let normalized = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= 32 {
        normalized
    } else {
        format!(
            "{}~{:016x}",
            normalized.chars().take(32).collect::<String>(),
            stable_label_hash(&normalized)
        )
    }
}

fn stable_label_hash(value: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn expand_manifest_cases(manifest: &BenchmarkManifest) -> Result<Vec<BenchmarkManifestCase>> {
    let mut expanded = Vec::new();
    for case in &manifest.benchmarks {
        expanded.extend(expand_manifest_case(case)?);
    }
    reject_duplicate_manifest_case_names(&expanded)?;
    Ok(expanded)
}

fn expand_manifest_case(case: &BenchmarkManifestCase) -> Result<Vec<BenchmarkManifestCase>> {
    let Some(matrix) = case.matrix.as_ref() else {
        return Ok(vec![case.clone()]);
    };
    let dimensions = matrix.dimensions()?;
    let mut expanded = vec![(case.clone(), Vec::<String>::new())];

    for dimension in dimensions {
        let mut next = Vec::new();
        for (base, labels) in expanded {
            for value in &dimension.values {
                let mut case = base.clone();
                value.apply(&mut case);
                let mut labels = labels.clone();
                labels.push(format!("{}={}", dimension.key, value.label_value()));
                next.push((case, labels));
            }
        }
        expanded = next;
    }

    Ok(expanded
        .into_iter()
        .map(|(mut case, labels)| {
            case.matrix = None;
            case.name = Some(match case.name.as_deref() {
                Some(name) => format!("{name}[{}]", labels.join(",")),
                None => format!(
                    "{}[{}]",
                    case.command.to_ascii_lowercase(),
                    labels.join(",")
                ),
            });
            case
        })
        .collect())
}

fn reject_duplicate_manifest_case_names(cases: &[BenchmarkManifestCase]) -> Result<()> {
    let mut names = BTreeSet::new();
    for (index, case) in cases.iter().enumerate() {
        let name = case
            .name
            .clone()
            .unwrap_or_else(|| format!("case-{}", index + 1));
        if !names.insert(name.clone()) {
            return Err(CliError::InvalidInput(format!(
                "Benchmark manifest expands to duplicate case name `{name}`"
            )));
        }
    }
    Ok(())
}

async fn read_json_report(path: &Path) -> Result<serde_json::Value> {
    let text = tokio::fs::read_to_string(path)
        .await
        .map_err(CliError::Io)?;
    serde_json::from_str(&text)
        .map_err(|e| CliError::InvalidInput(format!("Invalid benchmark JSON report: {e}")))
}

fn report_entries(value: &serde_json::Value) -> Result<Vec<ReportEntry>> {
    if let Some(reports) = value.get("reports").and_then(|value| value.as_array()) {
        return reports
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                let report = entry.get("report").ok_or_else(|| {
                    CliError::InvalidInput("Suite report entry missing `report`".to_string())
                })?;
                let summary = report.get("summary").cloned().ok_or_else(|| {
                    CliError::InvalidInput("Benchmark report missing `summary`".to_string())
                })?;
                let name = entry
                    .get("name")
                    .and_then(|value| value.as_str())
                    .or_else(|| report.get("command").and_then(|value| value.as_str()))
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("case-{}", index + 1));
                Ok(ReportEntry { name, summary })
            })
            .collect();
    }

    let summary = value
        .get("summary")
        .cloned()
        .ok_or_else(|| CliError::InvalidInput("Benchmark report missing `summary`".to_string()))?;
    let name = value
        .get("command")
        .and_then(|value| value.as_str())
        .unwrap_or("benchmark")
        .to_string();
    Ok(vec![ReportEntry { name, summary }])
}

fn collect_metric_comparisons(
    case: &str,
    current: &serde_json::Value,
    baseline: &serde_json::Value,
    tolerance_percent: f64,
    checks: &mut Vec<BenchmarkComparison>,
) {
    for (metric, path, lower_is_better) in [
        ("latency_ms.p95", &["latency_ms", "p95"][..], true),
        ("ttft_ms.p95", &["ttft_ms", "p95"][..], true),
        ("end_to_end_ms.p95", &["end_to_end_ms", "p95"][..], true),
        ("rtf.avg", &["rtf", "avg"][..], true),
        ("throughput_rps", &["throughput_rps"][..], false),
        ("completion_tps.p50", &["completion_tps", "p50"][..], false),
        (
            "tokens_per_second.p50",
            &["tokens_per_second", "p50"][..],
            false,
        ),
    ] {
        let Some(current_value) = summary_metric(current, path) else {
            continue;
        };
        let Some(baseline_value) = summary_metric(baseline, path) else {
            continue;
        };
        let change_percent = percent_change(current_value, baseline_value);
        let regression = if lower_is_better {
            current_value > baseline_value * (1.0 + tolerance_percent / 100.0)
        } else {
            current_value < baseline_value * (1.0 - tolerance_percent / 100.0)
        };
        checks.push(BenchmarkComparison {
            case: case.to_string(),
            metric: metric.to_string(),
            baseline: baseline_value,
            current: current_value,
            change_percent,
            lower_is_better,
            status: if regression { "regression" } else { "ok" },
        });
    }
}

fn summary_metric(summary: &serde_json::Value, path: &[&str]) -> Option<f64> {
    let mut value = summary;
    for key in path {
        value = value.get(*key)?;
    }
    value.as_f64().filter(|value| value.is_finite())
}

fn percent_change(current: f64, baseline: f64) -> f64 {
    if baseline == 0.0 {
        if current == 0.0 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        (current - baseline) * 100.0 / baseline
    }
}

async fn bench_manifest(
    server: &str,
    manifest_path: &Path,
    artifact_dir: Option<&Path>,
    options: &BenchOptions,
    theme: &Theme,
) -> Result<()> {
    let manifest_text = tokio::fs::read_to_string(manifest_path)
        .await
        .map_err(CliError::Io)?;
    let manifest: BenchmarkManifest = toml::from_str(&manifest_text)
        .map_err(|e| CliError::InvalidInput(format!("Invalid benchmark manifest: {e}")))?;
    if manifest.benchmarks.is_empty() {
        return Err(CliError::InvalidInput(
            "Benchmark manifest must include at least one [[benchmarks]] entry".to_string(),
        ));
    }
    let benchmark_cases = expand_manifest_cases(&manifest)?;

    let suite_server = manifest.server.as_deref().unwrap_or(server).to_string();
    let started_at = Utc::now();
    let observability_before = capture_observability(&suite_server).await;
    let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let mut reports = Vec::new();

    if options.interactive() {
        theme.step(
            1,
            benchmark_cases.len(),
            &format!("Running benchmark manifest {}", manifest_path.display()),
        );
    }

    for (index, case) in benchmark_cases.iter().enumerate() {
        if options.interactive() {
            let label = case.name.as_deref().unwrap_or(case.command.as_str());
            theme.info(&format!(
                "Case {}/{}: {}",
                index + 1,
                benchmark_cases.len(),
                label
            ));
        }

        let report = match case.command.to_ascii_lowercase().as_str() {
            "chat" => {
                let prompt = case.prompt.as_deref().unwrap_or(
                    "Summarize the main trade-offs between chunked prefill and continuous batching in two concise paragraphs.",
                );
                let model = case.model.as_deref().unwrap_or("Qwen3.5-4B");
                bench_chat(
                    &suite_server,
                    model,
                    case.iterations.unwrap_or(10),
                    prompt,
                    case.system.as_deref(),
                    case.max_tokens.unwrap_or(128),
                    case.concurrent.unwrap_or(1),
                    case.warmup.unwrap_or(false),
                    options,
                    theme,
                )
                .await?
            }
            "tts" => {
                let text = case
                    .text
                    .as_deref()
                    .unwrap_or("Hello, this is a benchmark test for text to speech synthesis.");
                let model = case.model.as_deref().unwrap_or("qwen3-tts-0.6b-base");
                bench_tts(
                    &suite_server,
                    model,
                    case.iterations.unwrap_or(10),
                    text,
                    case.concurrent.unwrap_or(1),
                    case.warmup.unwrap_or(false),
                    options,
                    theme,
                )
                .await?
            }
            "asr" => {
                let file = case.file.as_ref().map(|file| {
                    let path = Path::new(file);
                    if path.is_absolute() {
                        path.to_path_buf()
                    } else {
                        manifest_dir.join(path)
                    }
                });
                let model = case.model.as_deref().unwrap_or("parakeet-tdt-0.6b-v3");
                bench_asr(
                    &suite_server,
                    model,
                    case.iterations.unwrap_or(10),
                    file,
                    case.language.as_deref(),
                    case.concurrent.unwrap_or(1),
                    case.warmup.unwrap_or(false),
                    options,
                    theme,
                )
                .await?
            }
            "throughput" => {
                bench_throughput(
                    &suite_server,
                    case.duration_secs.unwrap_or(30),
                    case.concurrent.unwrap_or(1),
                    options,
                    theme,
                )
                .await?
            }
            other => {
                return Err(CliError::InvalidInput(format!(
                    "Unsupported benchmark manifest command: {other}"
                )));
            }
        };

        reports.push(BenchmarkSuiteCaseReport {
            name: case.name.clone(),
            report,
        });
    }

    let ended_at = Utc::now();
    let observability_after = capture_observability(&suite_server).await;
    let suite = BenchmarkSuiteReport {
        schema_version: 1,
        manifest: manifest_path.display().to_string(),
        server: suite_server.clone(),
        started_at,
        ended_at,
        reports,
    };
    if let Some(artifact_dir) = artifact_dir {
        write_artifact_bundle(
            artifact_dir,
            &suite,
            &manifest_text,
            BenchmarkArtifactMetadata {
                schema_version: 1,
                cli_version: env!("CARGO_PKG_VERSION"),
                server: suite_server,
                manifest: manifest_path.display().to_string(),
                git_sha: current_git_sha(),
                os: std::env::consts::OS,
                arch: std::env::consts::ARCH,
                started_at,
                ended_at,
            },
            BenchmarkObservabilityBundle {
                before: observability_before,
                after: observability_after,
            },
        )
        .await?;
    }
    emit_suite_report(options, &suite)?;

    if options.human_output() {
        println!(
            "\n{} completed {} benchmark case(s)",
            console::style("Manifest").bold(),
            suite.reports.len()
        );
        if let Some(artifact_dir) = artifact_dir {
            println!("  Artifacts: {}", artifact_dir.display());
        }
    }

    Ok(())
}

async fn capture_observability(server: &str) -> ObservabilitySnapshot {
    let metrics = match fetch_json(server, "/internal/metrics").await {
        Some(value) => Some(value),
        None => fetch_json(server, "/v1/metrics").await,
    };
    let prometheus = match fetch_text(server, "/internal/metrics/prometheus").await {
        Some(value) => Some(value),
        None => fetch_text(server, "/v1/metrics/prometheus").await,
    };
    ObservabilitySnapshot {
        captured_at: Utc::now(),
        health: fetch_json(server, "/v1/health").await,
        metrics,
        prometheus,
    }
}

async fn fetch_json(server: &str, path: &str) -> Option<serde_json::Value> {
    let text = fetch_text(server, path).await?;
    serde_json::from_str(&text).ok()
}

async fn fetch_text(server: &str, path: &str) -> Option<String> {
    let client = http::client(Some(std::time::Duration::from_secs(10))).ok()?;
    let base = server.trim_end_matches('/');
    let response = client.get(format!("{base}{path}")).send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    response.text().await.ok()
}

async fn write_artifact_bundle(
    artifact_dir: &Path,
    suite: &BenchmarkSuiteReport,
    manifest_text: &str,
    metadata: BenchmarkArtifactMetadata,
    observability: BenchmarkObservabilityBundle,
) -> Result<()> {
    tokio::fs::create_dir_all(artifact_dir)
        .await
        .map_err(CliError::Io)?;
    write_json_artifact(artifact_dir.join("report.json"), suite).await?;
    write_json_artifact(artifact_dir.join("metadata.json"), &metadata).await?;
    write_json_artifact(artifact_dir.join("observability.json"), &observability).await?;
    tokio::fs::write(artifact_dir.join("manifest.toml"), manifest_text)
        .await
        .map_err(CliError::Io)?;
    Ok(())
}

async fn write_json_artifact<T: Serialize>(path: std::path::PathBuf, value: &T) -> Result<()> {
    let payload = serde_json::to_vec_pretty(value)
        .map_err(|e| CliError::Other(format!("Failed to serialize artifact: {e}")))?;
    tokio::fs::write(path, payload).await.map_err(CliError::Io)
}

fn current_git_sha() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

async fn bench_chat(
    server: &str,
    model: &str,
    iterations: u32,
    prompt: &str,
    system: Option<&str>,
    max_tokens: usize,
    concurrent: u32,
    warmup: bool,
    options: &BenchOptions,
    theme: &Theme,
) -> Result<BenchmarkReport> {
    if iterations == 0 {
        return Err(CliError::InvalidInput(
            "Iterations must be greater than 0".to_string(),
        ));
    }
    if concurrent == 0 {
        return Err(CliError::InvalidInput(
            "Concurrent requests must be greater than 0".to_string(),
        ));
    }
    if max_tokens == 0 {
        return Err(CliError::InvalidInput(
            "Max tokens must be greater than 0".to_string(),
        ));
    }

    if options.interactive() {
        theme.step(1, 3, &format!("Benchmarking chat with '{}'", model));
    }
    let started_at = Utc::now();
    let run_start = Instant::now();
    let metrics_before = fetch_runtime_metrics(server).await;

    if warmup {
        if options.interactive() {
            theme.info("Running warmup iteration...");
        }
        let _ = run_chat_request(server, model, prompt, system, max_tokens).await?;
    }

    let pb = progress_bar(options.interactive(), iterations as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} chat requests",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    let progress = Arc::new(pb);
    let prompt = Arc::new(prompt.to_string());
    let system = system.map(|value| Arc::new(value.to_string()));
    let samples: Vec<ChatBenchSample> = stream::iter(0..iterations)
        .map(|_| {
            let progress = Arc::clone(&progress);
            let prompt = Arc::clone(&prompt);
            let system = system.clone();
            let model = model.to_string();
            let server = server.to_string();
            async move {
                let result = run_chat_request(
                    &server,
                    &model,
                    prompt.as_str(),
                    system.as_deref().map(|s| s.as_str()),
                    max_tokens,
                )
                .await;
                progress.inc(1);
                result
            }
        })
        .buffer_unordered(concurrent as usize)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

    progress.finish_with_message("Benchmark complete");

    let ttft_ms: Vec<f64> = samples.iter().map(|sample| sample.ttft_ms).collect();
    let total_ms: Vec<f64> = samples.iter().map(|sample| sample.total_ms).collect();
    let server_generation_ms: Vec<f64> = samples
        .iter()
        .filter_map(|sample| sample.generation_time_ms)
        .collect();
    let completion_tps: Vec<f64> = samples
        .iter()
        .map(|sample| {
            if sample.total_ms > 0.0 {
                sample.completion_tokens as f64 * 1000.0 / sample.total_ms
            } else {
                0.0
            }
        })
        .collect();
    let prompt_tokens_avg = samples
        .iter()
        .map(|sample| sample.prompt_tokens as f64)
        .sum::<f64>()
        / samples.len() as f64;
    let completion_tokens_avg = samples
        .iter()
        .map(|sample| sample.completion_tokens as f64)
        .sum::<f64>()
        / samples.len() as f64;

    if options.human_output() {
        println!("\n{}", console::style("Results:").bold().underlined());
        println!("  Iterations: {}", iterations);
        println!("  Concurrent: {}", concurrent);
        println!("  Prompt tokens (avg):      {:.2}", prompt_tokens_avg);
        println!("  Completion tokens (avg):  {:.2}", completion_tokens_avg);
        println!(
            "  TTFT (avg/p50/p95):       {:.2} / {:.2} / {:.2} ms",
            ttft_ms.iter().sum::<f64>() / ttft_ms.len() as f64,
            percentile(&ttft_ms, 0.5),
            percentile(&ttft_ms, 0.95)
        );
        println!(
            "  End-to-end (avg/p50/p95): {:.2} / {:.2} / {:.2} ms",
            total_ms.iter().sum::<f64>() / total_ms.len() as f64,
            percentile(&total_ms, 0.5),
            percentile(&total_ms, 0.95)
        );
        println!(
            "  Completion TPS (avg/p50/p95): {:.2} / {:.2} / {:.2} tok/s",
            completion_tps.iter().sum::<f64>() / completion_tps.len() as f64,
            percentile(&completion_tps, 0.5),
            percentile(&completion_tps, 0.95)
        );
        if !server_generation_ms.is_empty() {
            println!(
                "  Server generation (avg/p50/p95): {:.2} / {:.2} / {:.2} ms",
                server_generation_ms.iter().sum::<f64>() / server_generation_ms.len() as f64,
                percentile(&server_generation_ms, 0.5),
                percentile(&server_generation_ms, 0.95)
            );
        }
    }
    let metrics_after = fetch_runtime_metrics(server).await;
    if options.human_output() {
        print_runtime_delta(
            metrics_before.clone(),
            metrics_after.clone(),
            RuntimeTelemetryContext::Default,
        );
    }
    let ended_at = Utc::now();
    Ok(BenchmarkReport {
        schema_version: 1,
        command: "chat",
        server: server.to_string(),
        started_at,
        ended_at,
        duration_ms: run_start.elapsed().as_secs_f64() * 1000.0,
        config: BenchmarkRunConfig {
            model: Some(model.to_string()),
            iterations: Some(iterations),
            concurrent: Some(concurrent),
            warmup,
            prompt: Some(prompt.to_string()),
            system: system.as_deref().map(|value| value.to_string()),
            max_tokens: Some(max_tokens),
            text: None,
            file: None,
            language: None,
            duration_secs: None,
        },
        summary: BenchmarkSummary {
            latency_ms: None,
            ttft_ms: stats(&ttft_ms),
            end_to_end_ms: stats(&total_ms),
            completion_tps: stats(&completion_tps),
            tokens_per_second: None,
            server_generation_ms: stats(&server_generation_ms),
            server_processing_ms: None,
            audio_duration_secs: None,
            rtf: None,
            prompt_tokens_avg: Some(prompt_tokens_avg),
            completion_tokens_avg: Some(completion_tokens_avg),
            throughput_rps: Some(iterations as f64 / run_start.elapsed().as_secs_f64()),
            successful: Some(iterations as u64),
            failed: Some(0),
            total: Some(iterations as u64),
        },
        samples: samples
            .iter()
            .enumerate()
            .map(|(index, sample)| BenchmarkSample {
                index: index + 1,
                latency_ms: Some(sample.total_ms),
                ttft_ms: Some(sample.ttft_ms),
                end_to_end_ms: Some(sample.total_ms),
                completion_tps: Some(if sample.total_ms > 0.0 {
                    sample.completion_tokens as f64 * 1000.0 / sample.total_ms
                } else {
                    0.0
                }),
                tokens_per_second: None,
                prompt_tokens: Some(sample.prompt_tokens),
                completion_tokens: Some(sample.completion_tokens),
                server_generation_ms: sample.generation_time_ms,
                server_processing_ms: None,
                audio_duration_secs: None,
                rtf: None,
                tokens_generated: None,
            })
            .collect(),
        telemetry: RuntimeTelemetryReport {
            delta_available: metrics_before.is_some() && metrics_after.is_some(),
            before: metrics_before,
            after: metrics_after,
        },
    })
}

async fn bench_tts(
    server: &str,
    model: &str,
    iterations: u32,
    text: &str,
    concurrent: u32,
    warmup: bool,
    options: &BenchOptions,
    theme: &Theme,
) -> Result<BenchmarkReport> {
    if iterations == 0 {
        return Err(CliError::InvalidInput(
            "Iterations must be greater than 0".to_string(),
        ));
    }
    if concurrent == 0 {
        return Err(CliError::InvalidInput(
            "Concurrent requests must be greater than 0".to_string(),
        ));
    }

    if options.interactive() {
        theme.step(1, 3, &format!("Benchmarking TTS with '{}'", model));
    }
    let started_at = Utc::now();
    let run_start = Instant::now();
    let metrics_before = fetch_runtime_metrics(server).await;

    if warmup {
        if options.interactive() {
            theme.info("Running warmup iteration...");
        }
        let _ = run_tts_request(server, model, text).await?;
    }

    let pb = progress_bar(options.interactive(), iterations as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} iterations",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    let progress = Arc::new(pb);
    let text = Arc::new(text.to_string());
    let wall_start = Instant::now();
    let samples: Vec<TtsBenchSample> = stream::iter(0..iterations)
        .map(|_| {
            let progress = Arc::clone(&progress);
            let text = Arc::clone(&text);
            let model = model.to_string();
            let server = server.to_string();
            async move {
                let start = Instant::now();
                let result =
                    run_tts_request(&server, &model, text.as_str())
                        .await
                        .map(|mut sample| {
                            sample.total_ms = start.elapsed().as_secs_f64() * 1000.0;
                            sample
                        });
                progress.inc(1);
                result
            }
        })
        .buffer_unordered(concurrent as usize)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let wall_elapsed = wall_start.elapsed();

    progress.finish_with_message("Benchmark complete");

    // Calculate statistics
    let times: Vec<f64> = samples.iter().map(|sample| sample.total_ms).collect();
    let generation_ms: Vec<f64> = samples
        .iter()
        .filter_map(|sample| sample.generation_time_ms)
        .collect();
    let audio_duration_secs: Vec<f64> = samples
        .iter()
        .filter_map(|sample| sample.audio_duration_secs)
        .collect();
    let rtf: Vec<f64> = samples.iter().filter_map(|sample| sample.rtf).collect();
    let tokens_per_second: Vec<f64> = samples
        .iter()
        .filter_map(
            |sample| match (sample.tokens_generated, sample.generation_time_ms) {
                (Some(tokens), Some(ms)) if ms > 0.0 => Some(tokens as f64 * 1000.0 / ms),
                _ => None,
            },
        )
        .collect();
    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(0.0, f64::max);
    let p50 = percentile(&times, 0.5);
    let p95 = percentile(&times, 0.95);
    let p99 = percentile(&times, 0.99);

    if options.human_output() {
        println!("\n{}", console::style("Results:").bold().underlined());
        println!("  Iterations: {}", iterations);
        println!("  Concurrent: {}", concurrent);
        println!("  Average:    {:.2} ms", avg);
        println!("  Min:        {:.2} ms", min);
        println!("  Max:        {:.2} ms", max);
        println!("  P50:        {:.2} ms", p50);
        println!("  P95:        {:.2} ms", p95);
        println!("  P99:        {:.2} ms", p99);
        println!(
            "  Throughput: {:.2} req/s",
            iterations as f64 / wall_elapsed.as_secs_f64()
        );
        if !generation_ms.is_empty() {
            println!(
                "  Server generation (avg/p50/p95): {:.2} / {:.2} / {:.2} ms",
                generation_ms.iter().sum::<f64>() / generation_ms.len() as f64,
                percentile(&generation_ms, 0.5),
                percentile(&generation_ms, 0.95)
            );
        }
        if !audio_duration_secs.is_empty() {
            println!(
                "  Audio duration (avg/p50/p95):    {:.2} / {:.2} / {:.2} s",
                audio_duration_secs.iter().sum::<f64>() / audio_duration_secs.len() as f64,
                percentile(&audio_duration_secs, 0.5),
                percentile(&audio_duration_secs, 0.95)
            );
        }
        if !rtf.is_empty() {
            println!(
                "  RTF (avg/p50/p95):               {:.3} / {:.3} / {:.3}",
                rtf.iter().sum::<f64>() / rtf.len() as f64,
                percentile(&rtf, 0.5),
                percentile(&rtf, 0.95)
            );
        }
        if !tokens_per_second.is_empty() {
            println!(
                "  Tokens/sec (avg/p50/p95):        {:.2} / {:.2} / {:.2}",
                tokens_per_second.iter().sum::<f64>() / tokens_per_second.len() as f64,
                percentile(&tokens_per_second, 0.5),
                percentile(&tokens_per_second, 0.95)
            );
        }
    }
    let metrics_after = fetch_runtime_metrics(server).await;
    if options.human_output() {
        print_runtime_delta(
            metrics_before.clone(),
            metrics_after.clone(),
            RuntimeTelemetryContext::Default,
        );
    }
    let ended_at = Utc::now();
    Ok(BenchmarkReport {
        schema_version: 1,
        command: "tts",
        server: server.to_string(),
        started_at,
        ended_at,
        duration_ms: run_start.elapsed().as_secs_f64() * 1000.0,
        config: BenchmarkRunConfig {
            model: Some(model.to_string()),
            iterations: Some(iterations),
            concurrent: Some(concurrent),
            warmup,
            prompt: None,
            system: None,
            max_tokens: None,
            text: Some(text.as_ref().clone()),
            file: None,
            language: None,
            duration_secs: None,
        },
        summary: BenchmarkSummary {
            latency_ms: stats(&times),
            ttft_ms: None,
            end_to_end_ms: stats(&times),
            completion_tps: None,
            tokens_per_second: stats(&tokens_per_second),
            server_generation_ms: stats(&generation_ms),
            server_processing_ms: None,
            audio_duration_secs: stats(&audio_duration_secs),
            rtf: stats(&rtf),
            prompt_tokens_avg: None,
            completion_tokens_avg: None,
            throughput_rps: Some(iterations as f64 / wall_elapsed.as_secs_f64()),
            successful: Some(iterations as u64),
            failed: Some(0),
            total: Some(iterations as u64),
        },
        samples: samples
            .iter()
            .enumerate()
            .map(|(index, sample)| BenchmarkSample {
                index: index + 1,
                latency_ms: Some(sample.total_ms),
                ttft_ms: None,
                end_to_end_ms: Some(sample.total_ms),
                completion_tps: None,
                tokens_per_second: match (sample.tokens_generated, sample.generation_time_ms) {
                    (Some(tokens), Some(ms)) if ms > 0.0 => Some(tokens as f64 * 1000.0 / ms),
                    _ => None,
                },
                prompt_tokens: None,
                completion_tokens: None,
                server_generation_ms: sample.generation_time_ms,
                server_processing_ms: None,
                audio_duration_secs: sample.audio_duration_secs,
                rtf: sample.rtf,
                tokens_generated: sample.tokens_generated,
            })
            .collect(),
        telemetry: RuntimeTelemetryReport {
            delta_available: metrics_before.is_some() && metrics_after.is_some(),
            before: metrics_before,
            after: metrics_after,
        },
    })
}

async fn bench_asr(
    server: &str,
    model: &str,
    iterations: u32,
    file: Option<std::path::PathBuf>,
    language: Option<&str>,
    concurrent: u32,
    warmup: bool,
    options: &BenchOptions,
    theme: &Theme,
) -> Result<BenchmarkReport> {
    if iterations == 0 {
        return Err(CliError::InvalidInput(
            "Iterations must be greater than 0".to_string(),
        ));
    }
    if concurrent == 0 {
        return Err(CliError::InvalidInput(
            "Concurrent requests must be greater than 0".to_string(),
        ));
    }

    if options.interactive() {
        theme.step(1, 3, &format!("Benchmarking ASR with '{}'", model));
    }
    let started_at = Utc::now();
    let run_start = Instant::now();
    let metrics_before = fetch_runtime_metrics(server).await;

    // Use sample audio if no file provided
    let audio_file = file.unwrap_or_else(|| std::path::PathBuf::from("data/test.wav"));

    if !audio_file.exists() {
        return Err(CliError::InvalidInput(format!(
            "Audio file not found: {}",
            audio_file.display()
        )));
    }

    let audio_data = tokio::fs::read(&audio_file).await.map_err(CliError::Io)?;
    let audio_base64 = STANDARD.encode(&audio_data);
    if let Some(language) = language {
        if options.interactive() {
            theme.info(&format!("Using language hint: {}", language));
        }
    }

    if warmup {
        if options.interactive() {
            theme.info("Running warmup iteration...");
        }
        let _ = run_asr_request(server, model, &audio_base64, language).await?;
    }

    let pb = progress_bar(options.interactive(), iterations as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} iterations",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    let progress = Arc::new(pb);
    let audio_base64 = Arc::new(audio_base64);
    let language = language.map(|value| Arc::new(value.to_string()));
    let wall_start = Instant::now();
    let samples: Vec<AsrBenchSample> = stream::iter(0..iterations)
        .map(|_| {
            let progress = Arc::clone(&progress);
            let audio_base64 = Arc::clone(&audio_base64);
            let language = language.clone();
            let model = model.to_string();
            let server = server.to_string();
            async move {
                let start = Instant::now();
                let result = run_asr_request(
                    &server,
                    &model,
                    audio_base64.as_str(),
                    language.as_deref().map(|value| value.as_str()),
                )
                .await
                .map(|response| AsrBenchSample {
                    total_ms: start.elapsed().as_secs_f64() * 1000.0,
                    response,
                });
                progress.inc(1);
                result
            }
        })
        .buffer_unordered(concurrent as usize)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let wall_elapsed = wall_start.elapsed();

    progress.finish_with_message("Benchmark complete");

    let times: Vec<f64> = samples.iter().map(|sample| sample.total_ms).collect();
    let audio_duration_secs: Vec<f64> = samples
        .iter()
        .filter_map(|sample| sample.response.duration)
        .collect();
    let processing_ms: Vec<f64> = samples
        .iter()
        .filter_map(|sample| sample.response.processing_time_ms)
        .collect();
    let rtf: Vec<f64> = samples
        .iter()
        .filter_map(|sample| sample.response.rtf)
        .collect();
    let mut stage_samples = Vec::new();
    let mut saw_whisper_diagnostics = false;
    for sample in &samples {
        if let Some(diagnostics) = sample.response.izwi_asr_diagnostics.as_ref() {
            if diagnostics_contains_whisper_model(diagnostics) {
                saw_whisper_diagnostics = true;
            }
            stage_samples.extend(collect_whisper_stage_timings_from_diagnostics(diagnostics));
        }
    }

    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().copied().fold(f64::INFINITY, f64::min);
    let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let p50 = percentile(&times, 0.5);
    let p95 = percentile(&times, 0.95);
    let p99 = percentile(&times, 0.99);
    if options.human_output() {
        println!("\n{}", console::style("Results:").bold().underlined());
        println!("  Iterations: {}", iterations);
        println!("  Concurrent: {}", concurrent);
        println!("  Average:    {:.2} ms", avg);
        println!("  Min:        {:.2} ms", min);
        println!("  Max:        {:.2} ms", max);
        println!("  P50:        {:.2} ms", p50);
        println!("  P95:        {:.2} ms", p95);
        println!("  P99:        {:.2} ms", p99);
        println!(
            "  Throughput: {:.2} req/s",
            iterations as f64 / wall_elapsed.as_secs_f64()
        );
        if !audio_duration_secs.is_empty() {
            println!(
                "  Audio duration (avg/p50/p95): {:.2} / {:.2} / {:.2} s",
                audio_duration_secs.iter().sum::<f64>() / audio_duration_secs.len() as f64,
                percentile(&audio_duration_secs, 0.5),
                percentile(&audio_duration_secs, 0.95)
            );
        }
        if !rtf.is_empty() {
            println!(
                "  RTF (avg/p50/p95):            {:.3} / {:.3} / {:.3}",
                rtf.iter().sum::<f64>() / rtf.len() as f64,
                percentile(&rtf, 0.5),
                percentile(&rtf, 0.95)
            );
        }
        print_whisper_stage_timing_summary(&stage_samples);
    }

    let model_lower = model.to_ascii_lowercase();
    let telemetry_context = if saw_whisper_diagnostics || model_lower.contains("whisper") {
        RuntimeTelemetryContext::AsrWhisper
    } else {
        RuntimeTelemetryContext::Default
    };
    let metrics_after = fetch_runtime_metrics(server).await;
    if options.human_output() {
        print_runtime_delta(
            metrics_before.clone(),
            metrics_after.clone(),
            telemetry_context,
        );
    }
    let ended_at = Utc::now();
    Ok(BenchmarkReport {
        schema_version: 1,
        command: "asr",
        server: server.to_string(),
        started_at,
        ended_at,
        duration_ms: run_start.elapsed().as_secs_f64() * 1000.0,
        config: BenchmarkRunConfig {
            model: Some(model.to_string()),
            iterations: Some(iterations),
            concurrent: Some(concurrent),
            warmup,
            prompt: None,
            system: None,
            max_tokens: None,
            text: None,
            file: Some(audio_file.display().to_string()),
            language: language.as_deref().map(|value| value.to_string()),
            duration_secs: None,
        },
        summary: BenchmarkSummary {
            latency_ms: stats(&times),
            ttft_ms: None,
            end_to_end_ms: stats(&times),
            completion_tps: None,
            tokens_per_second: None,
            server_generation_ms: None,
            server_processing_ms: stats(&processing_ms),
            audio_duration_secs: stats(&audio_duration_secs),
            rtf: stats(&rtf),
            prompt_tokens_avg: None,
            completion_tokens_avg: None,
            throughput_rps: Some(iterations as f64 / wall_elapsed.as_secs_f64()),
            successful: Some(iterations as u64),
            failed: Some(0),
            total: Some(iterations as u64),
        },
        samples: samples
            .iter()
            .enumerate()
            .map(|(index, sample)| BenchmarkSample {
                index: index + 1,
                latency_ms: Some(sample.total_ms),
                ttft_ms: None,
                end_to_end_ms: Some(sample.total_ms),
                completion_tps: None,
                tokens_per_second: None,
                prompt_tokens: None,
                completion_tokens: None,
                server_generation_ms: None,
                server_processing_ms: sample.response.processing_time_ms,
                audio_duration_secs: sample.response.duration,
                rtf: sample.response.rtf,
                tokens_generated: None,
            })
            .collect(),
        telemetry: RuntimeTelemetryReport {
            delta_available: metrics_before.is_some() && metrics_after.is_some(),
            before: metrics_before,
            after: metrics_after,
        },
    })
}

async fn bench_throughput(
    server: &str,
    duration: u64,
    concurrent: u32,
    options: &BenchOptions,
    theme: &Theme,
) -> Result<BenchmarkReport> {
    if duration == 0 {
        return Err(CliError::InvalidInput(
            "Duration must be greater than 0 seconds".to_string(),
        ));
    }
    if concurrent == 0 {
        return Err(CliError::InvalidInput(
            "Concurrent requests must be greater than 0".to_string(),
        ));
    }

    if options.interactive() {
        theme.step(
            1,
            1,
            &format!("Throughput test: {}s, {} concurrent", duration, concurrent),
        );
    }
    if options.human_output() {
        println!("Running throughput benchmark against /livez...");
    }
    let client = http::client(Some(Duration::from_secs(5)))?;
    let started_at = Utc::now();
    let run_start = Instant::now();
    let deadline = run_start + Duration::from_secs(duration);

    let mut workers = Vec::new();
    for _ in 0..concurrent {
        let client = client.clone();
        let server = server.to_string();
        workers.push(tokio::spawn(async move {
            let mut success = 0u64;
            let mut failed = 0u64;
            while std::time::Instant::now() < deadline {
                match client.get(format!("{}/livez", server)).send().await {
                    Ok(resp) if resp.status().is_success() => success += 1,
                    _ => failed += 1,
                }
            }
            (success, failed)
        }));
    }

    let mut success = 0u64;
    let mut failed = 0u64;
    for worker in workers {
        let (ok, err) = worker
            .await
            .map_err(|e| CliError::Other(format!("Benchmark worker failed: {}", e)))?;
        success += ok;
        failed += err;
    }

    let total = success + failed;
    let measured_elapsed = run_start.elapsed();
    let rps = throughput_rps(total, measured_elapsed);
    if options.human_output() {
        println!("\n{}", console::style("Results:").bold().underlined());
        println!("  Successful: {:.0}", success);
        println!("  Failed:     {:.0}", failed);
        println!("  Total:      {:.0}", total);
        println!("  Throughput: {:.2} req/s", rps);
    }
    let ended_at = Utc::now();
    Ok(BenchmarkReport {
        schema_version: 1,
        command: "throughput",
        server: server.to_string(),
        started_at,
        ended_at,
        duration_ms: measured_elapsed.as_secs_f64() * 1000.0,
        config: BenchmarkRunConfig {
            model: None,
            iterations: None,
            concurrent: Some(concurrent),
            warmup: false,
            prompt: None,
            system: None,
            max_tokens: None,
            text: None,
            file: None,
            language: None,
            duration_secs: Some(duration),
        },
        summary: BenchmarkSummary {
            latency_ms: None,
            ttft_ms: None,
            end_to_end_ms: None,
            completion_tps: None,
            tokens_per_second: None,
            server_generation_ms: None,
            server_processing_ms: None,
            audio_duration_secs: None,
            rtf: None,
            prompt_tokens_avg: None,
            completion_tokens_avg: None,
            throughput_rps: Some(rps),
            successful: Some(success),
            failed: Some(failed),
            total: Some(total),
        },
        samples: Vec::new(),
        telemetry: RuntimeTelemetryReport {
            delta_available: false,
            before: None,
            after: None,
        },
    })
}

fn header_f64(response: &reqwest::Response, name: &'static str) -> Option<f64> {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<f64>().ok())
}

fn header_u64(response: &reqwest::Response, name: &'static str) -> Option<u64> {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
}

async fn run_tts_request(server: &str, model: &str, text: &str) -> Result<TtsBenchSample> {
    let client = http::client(Some(std::time::Duration::from_secs(300)))?;
    let request_body = serde_json::json!({
        "model": model,
        "input": text,
        "voice": "default",
        "response_format": "wav",
    });

    let response = client
        .post(format!("{}/v1/audio/speech", server))
        .json(&request_body)
        .send()
        .await
        .map_err(|e| CliError::ConnectionError(e.to_string()))?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        return Err(CliError::ApiError {
            status,
            message: text,
        });
    }

    Ok(TtsBenchSample {
        total_ms: 0.0,
        generation_time_ms: header_f64(&response, "x-generation-time-ms"),
        audio_duration_secs: header_f64(&response, "x-audio-duration-secs"),
        rtf: header_f64(&response, "x-rtf"),
        tokens_generated: header_u64(&response, "x-tokens-generated"),
    })
}

async fn run_asr_request(
    server: &str,
    model: &str,
    audio_base64: &str,
    language: Option<&str>,
) -> Result<AsrBenchResponse> {
    let client = http::client(Some(std::time::Duration::from_secs(300)))?;
    let mut request_body = serde_json::json!({
        "model": model,
        "audio_base64": audio_base64,
        "response_format": "verbose_json",
    });
    if let Some(language) = language {
        request_body["language"] = serde_json::Value::String(language.to_string());
    }

    let response = client
        .post(format!("{}/v1/audio/transcriptions", server))
        .json(&request_body)
        .send()
        .await
        .map_err(|e| CliError::ConnectionError(e.to_string()))?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        return Err(CliError::ApiError {
            status,
            message: text,
        });
    }

    response
        .json::<AsrBenchResponse>()
        .await
        .map_err(|e| CliError::Other(format!("Failed to parse ASR benchmark response: {e}")))
}

async fn run_chat_request(
    server: &str,
    model: &str,
    prompt: &str,
    system: Option<&str>,
    max_tokens: usize,
) -> Result<ChatBenchSample> {
    let client = http::client(Some(std::time::Duration::from_secs(300)))?;
    let mut messages = Vec::new();
    if let Some(system) = system {
        messages.push(serde_json::json!({
            "role": "system",
            "content": system,
        }));
    }
    messages.push(serde_json::json!({
        "role": "user",
        "content": prompt,
    }));

    let request_body = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "stream_options": {
            "include_usage": true,
        },
        "max_completion_tokens": max_tokens,
    });

    let response = client
        .post(format!("{}/v1/chat/completions", server))
        .json(&request_body)
        .send()
        .await
        .map_err(|e| CliError::ConnectionError(e.to_string()))?;

    if !response.status().is_success() {
        let status = response.status().as_u16();
        let text = response.text().await.unwrap_or_default();
        return Err(CliError::ApiError {
            status,
            message: text,
        });
    }

    let started = Instant::now();
    let mut first_delta_at: Option<f64> = None;
    let mut prompt_tokens = 0usize;
    let mut completion_tokens = 0usize;
    let mut generation_time_ms = None;
    let mut buffer = String::new();
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| CliError::ConnectionError(e.to_string()))?;
        let chunk_text = std::str::from_utf8(&chunk)
            .map_err(|e| CliError::Other(format!("Invalid UTF-8 in chat stream: {e}")))?;
        buffer.push_str(chunk_text);

        while let Some(boundary) = buffer.find("\n\n") {
            let event = buffer[..boundary].to_string();
            buffer.drain(..boundary + 2);
            if let Some(sample) = handle_chat_stream_event(
                &event,
                started,
                &mut first_delta_at,
                &mut prompt_tokens,
                &mut completion_tokens,
                &mut generation_time_ms,
            )? {
                return Ok(sample);
            }
        }
    }

    Err(CliError::Other(
        "Chat benchmark stream ended before a terminal event".to_string(),
    ))
}

fn handle_chat_stream_event(
    event: &str,
    started: Instant,
    first_delta_at: &mut Option<f64>,
    prompt_tokens: &mut usize,
    completion_tokens: &mut usize,
    generation_time_ms: &mut Option<f64>,
) -> Result<Option<ChatBenchSample>> {
    for line in event.lines() {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        let payload = payload.trim();
        if payload.is_empty() {
            continue;
        }
        if payload == "[DONE]" {
            continue;
        }

        if let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) {
            if let Some(message) = value
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(|message| message.as_str())
            {
                return Err(CliError::Other(format!(
                    "Chat benchmark request failed: {message}"
                )));
            }
        }

        let chunk: ChatStreamChunk = serde_json::from_str(payload)
            .map_err(|e| CliError::Other(format!("Invalid chat stream payload: {e}")))?;

        let has_delta = chunk.choices.iter().any(|choice| {
            choice
                .delta
                .content
                .as_ref()
                .is_some_and(|content| !content.is_empty())
        });
        if has_delta && first_delta_at.is_none() {
            *first_delta_at = Some(started.elapsed().as_secs_f64() * 1000.0);
        }

        if let Some(usage) = chunk.usage {
            *prompt_tokens = usage.prompt_tokens;
            *completion_tokens = usage.completion_tokens;
            *generation_time_ms = chunk.izwi_generation_time_ms;
            let total_ms = started.elapsed().as_secs_f64() * 1000.0;
            return Ok(Some(ChatBenchSample {
                ttft_ms: first_delta_at.unwrap_or(total_ms),
                total_ms,
                prompt_tokens: *prompt_tokens,
                completion_tokens: *completion_tokens,
                generation_time_ms: *generation_time_ms,
            }));
        }
    }

    Ok(None)
}

fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let index = (p * (sorted.len() - 1) as f64) as usize;
    sorted[index]
}

fn stats(data: &[f64]) -> Option<Stats> {
    if data.is_empty() {
        return None;
    }

    Some(Stats {
        count: data.len(),
        avg: data.iter().sum::<f64>() / data.len() as f64,
        min: data.iter().copied().fold(f64::INFINITY, f64::min),
        max: data.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        p50: percentile(data, 0.5),
        p95: percentile(data, 0.95),
        p99: percentile(data, 0.99),
    })
}

fn throughput_rps(total: u64, elapsed: Duration) -> f64 {
    let elapsed_secs = elapsed.as_secs_f64();
    if elapsed_secs > 0.0 {
        total as f64 / elapsed_secs
    } else {
        0.0
    }
}

fn progress_bar(visible: bool, len: u64) -> ProgressBar {
    if visible {
        ProgressBar::new(len)
    } else {
        ProgressBar::hidden()
    }
}

fn emit_report(options: &BenchOptions, report: &BenchmarkReport) -> Result<()> {
    if matches!(options.output_format, OutputFormat::Json) {
        let payload = serde_json::to_string_pretty(&report)
            .map_err(|e| CliError::Other(format!("Failed to serialize benchmark report: {e}")))?;
        println!("{payload}");
    }
    Ok(())
}

fn emit_suite_report(options: &BenchOptions, report: &BenchmarkSuiteReport) -> Result<()> {
    if matches!(options.output_format, OutputFormat::Json) {
        let payload = serde_json::to_string_pretty(&report).map_err(|e| {
            CliError::Other(format!("Failed to serialize benchmark suite report: {e}"))
        })?;
        println!("{payload}");
    }
    Ok(())
}

fn emit_compare_report(options: &BenchOptions, report: &BenchmarkCompareReport) -> Result<()> {
    if matches!(options.output_format, OutputFormat::Json) {
        let payload = serde_json::to_string_pretty(&report).map_err(|e| {
            CliError::Other(format!("Failed to serialize benchmark comparison: {e}"))
        })?;
        println!("{payload}");
        return Ok(());
    }

    println!(
        "\n{}",
        console::style("Benchmark Comparison:").bold().underlined()
    );
    println!("  Current:   {}", report.current);
    println!("  Baseline:  {}", report.baseline);
    println!("  Tolerance: {:.2}%", report.tolerance_percent);
    for check in &report.checks {
        let marker = if check.status == "regression" {
            console::style("REGRESSION").red().to_string()
        } else {
            console::style("ok").green().to_string()
        };
        println!(
            "  [{}] {} {}: current={:.4}, baseline={:.4}, change={:.2}%",
            marker, check.case, check.metric, check.current, check.baseline, check.change_percent
        );
    }
    println!("  Regressions: {}", report.regressions);
    Ok(())
}

async fn fetch_runtime_metrics(server: &str) -> Option<RuntimeTelemetrySnapshot> {
    let client = http::client(Some(std::time::Duration::from_secs(15))).ok()?;
    let base = server.trim_end_matches('/');

    for attempt in 0..3 {
        for path in ["/internal/metrics", "/v1/metrics"] {
            let response = match client.get(format!("{base}{path}")).send().await {
                Ok(response) => response,
                Err(_) => continue,
            };
            if !response.status().is_success() {
                continue;
            }
            if let Ok(metrics) = response.json::<RuntimeTelemetrySnapshot>().await {
                return Some(metrics);
            }
        }

        if attempt < 2 {
            tokio::time::sleep(std::time::Duration::from_millis(250 * (attempt + 1))).await;
        }
    }

    None
}

fn whisper_stage_timings_from_diagnostics(
    diagnostics: &serde_json::Value,
) -> Option<WhisperStageTimings> {
    let timings = diagnostics.get("timings_ms")?.as_object()?;
    Some(WhisperStageTimings {
        audio_decode: timings.get("audio_decode").and_then(|value| value.as_f64()),
        mel_prepare: timings.get("mel_prepare").and_then(|value| value.as_f64()),
        encoder_forward: timings
            .get("encoder_forward")
            .and_then(|value| value.as_f64()),
        language_detect: timings
            .get("language_detect")
            .and_then(|value| value.as_f64()),
        decode: timings.get("decode").and_then(|value| value.as_f64()),
        model_total: timings.get("model_total").and_then(|value| value.as_f64()),
    })
}

fn diagnostics_contains_whisper_model(diagnostics: &serde_json::Value) -> bool {
    if diagnostics
        .get("model_family")
        .and_then(|value| value.as_str())
        == Some("whisper_asr")
    {
        return true;
    }

    diagnostics
        .get("chunking")
        .and_then(|value| value.get("chunk_transcriptions"))
        .and_then(|value| value.as_array())
        .map(|chunks| {
            chunks.iter().any(|chunk| {
                chunk
                    .get("model_diagnostics")
                    .and_then(|value| value.get("model_family"))
                    .and_then(|value| value.as_str())
                    == Some("whisper_asr")
            })
        })
        .unwrap_or(false)
}

fn collect_whisper_stage_timings_from_diagnostics(
    diagnostics: &serde_json::Value,
) -> Vec<WhisperStageTimings> {
    let mut samples = Vec::new();
    if let Some(stage_sample) = whisper_stage_timings_from_diagnostics(diagnostics) {
        samples.push(stage_sample);
    }

    if let Some(chunks) = diagnostics
        .get("chunking")
        .and_then(|value| value.get("chunk_transcriptions"))
        .and_then(|value| value.as_array())
    {
        for chunk in chunks {
            let Some(model_diagnostics) = chunk.get("model_diagnostics") else {
                continue;
            };
            if model_diagnostics
                .get("model_family")
                .and_then(|value| value.as_str())
                != Some("whisper_asr")
            {
                continue;
            }
            if let Some(stage_sample) = whisper_stage_timings_from_diagnostics(model_diagnostics) {
                samples.push(stage_sample);
            }
        }
    }

    samples
}

fn summarize_stage(stage: &str, values: &[f64]) {
    if values.is_empty() {
        return;
    }
    let avg = values.iter().sum::<f64>() / values.len() as f64;
    let p50 = percentile(values, 0.5);
    let p95 = percentile(values, 0.95);
    println!(
        "  {:<14} avg/p50/p95: {:.2} / {:.2} / {:.2} ms",
        stage, avg, p50, p95
    );
}

fn print_whisper_stage_timing_summary(samples: &[WhisperStageTimings]) {
    if samples.is_empty() {
        return;
    }

    let mut audio_decode = Vec::new();
    let mut mel_prepare = Vec::new();
    let mut encoder_forward = Vec::new();
    let mut language_detect = Vec::new();
    let mut decode = Vec::new();
    let mut model_total = Vec::new();

    for sample in samples {
        if let Some(value) = sample.audio_decode {
            audio_decode.push(value);
        }
        if let Some(value) = sample.mel_prepare {
            mel_prepare.push(value);
        }
        if let Some(value) = sample.encoder_forward {
            encoder_forward.push(value);
        }
        if let Some(value) = sample.language_detect {
            language_detect.push(value);
        }
        if let Some(value) = sample.decode {
            decode.push(value);
        }
        if let Some(value) = sample.model_total {
            model_total.push(value);
        }
    }

    if audio_decode.is_empty()
        && mel_prepare.is_empty()
        && encoder_forward.is_empty()
        && language_detect.is_empty()
        && decode.is_empty()
        && model_total.is_empty()
    {
        return;
    }

    println!(
        "\n{}",
        console::style("ASR Stage Timings (run-local):")
            .bold()
            .underlined()
    );
    summarize_stage("audio_decode", &audio_decode);
    summarize_stage("mel_prepare", &mel_prepare);
    summarize_stage("encoder_fwd", &encoder_forward);
    summarize_stage("lang_detect", &language_detect);
    summarize_stage("decode", &decode);
    summarize_stage("model_total", &model_total);
}

fn print_runtime_delta(
    before: Option<RuntimeTelemetrySnapshot>,
    after: Option<RuntimeTelemetrySnapshot>,
    context: RuntimeTelemetryContext,
) {
    let Some(after) = after else {
        println!(
            "\nRuntime telemetry delta: unavailable (/internal/metrics or /v1/metrics not reachable)"
        );
        return;
    };
    let delta_available = before.is_some();
    let before = before.unwrap_or_default();

    let completed_delta = after
        .requests_completed
        .saturating_sub(before.requests_completed);
    let failed_delta = after.requests_failed.saturating_sub(before.requests_failed);
    let queued_delta = after.requests_queued.saturating_sub(before.requests_queued);
    let restart_delta = after.worker_restarts.saturating_sub(before.worker_restarts);
    let panic_delta = after.worker_panics.saturating_sub(before.worker_panics);

    println!(
        "\n{}",
        console::style("Runtime Telemetry Snapshot (post-run):")
            .bold()
            .underlined()
    );
    println!("  Note: rolling latency quantiles are runtime-wide, not run-local.");
    if !delta_available {
        println!("  Note: pre-run snapshot was unavailable; counters below are post-run totals.");
    }
    let counter_suffix = if delta_available { "" } else { " (total)" };
    println!("  Queued{}:             {}", counter_suffix, queued_delta);
    println!(
        "  Completed{}:          {}",
        counter_suffix, completed_delta
    );
    println!("  Failed{}:             {}", counter_suffix, failed_delta);
    println!("  Active (current):     {}", after.requests_active);
    println!("  Worker restarts{}:    {}", counter_suffix, restart_delta);
    println!("  Worker panics{}:      {}", counter_suffix, panic_delta);
    println!(
        "  Queue wait rolling(avg/p50/p95): {:.2} / {:.2} / {:.2} ms",
        after.queue_wait_ms_avg, after.queue_wait_ms_p50, after.queue_wait_ms_p95
    );
    println!(
        "  Prefill rolling(avg/p50/p95):    {:.2} / {:.2} / {:.2} ms",
        after.prefill_ms_avg, after.prefill_ms_p50, after.prefill_ms_p95
    );
    println!(
        "  Decode rolling(avg/p50/p95):     {:.2} / {:.2} / {:.2} ms",
        after.decode_ms_avg, after.decode_ms_p50, after.decode_ms_p95
    );
    if matches!(context, RuntimeTelemetryContext::AsrWhisper) {
        println!("  Decode rolling note: single-pass Whisper ASR may report near-zero engine decode phase.");
    }
    println!(
        "  TTFT rolling(avg/p50/p95):       {:.2} / {:.2} / {:.2} ms",
        after.ttft_ms_avg, after.ttft_ms_p50, after.ttft_ms_p95
    );
    println!(
        "  End-to-end rolling(avg/p50/p95): {:.2} / {:.2} / {:.2} ms",
        after.end_to_end_ms_avg, after.end_to_end_ms_p50, after.end_to_end_ms_p95
    );
    let kernel_before = &before.kernel_path;
    let kernel_after = &after.kernel_path;
    let prefill_token_mode_delta = kernel_after
        .prefill_token_mode_steps_total
        .saturating_sub(kernel_before.prefill_token_mode_steps_total);
    let prefill_sequence_spans_delta = kernel_after
        .prefill_sequence_spans_total
        .saturating_sub(kernel_before.prefill_sequence_spans_total);
    let prefill_sequence_tokens_delta = kernel_after
        .prefill_sequence_tokens_total
        .saturating_sub(kernel_before.prefill_sequence_tokens_total);
    let dense_decode_delta = kernel_after
        .decode_attention_dense_total
        .saturating_sub(kernel_before.decode_attention_dense_total);
    let paged_decode_delta = kernel_after
        .decode_attention_paged_total
        .saturating_sub(kernel_before.decode_attention_paged_total);
    let chunk_sequence_delta = kernel_after
        .chunk_attention_sequence_calls_total
        .saturating_sub(kernel_before.chunk_attention_sequence_calls_total);
    let chunk_spans_delta = kernel_after
        .chunk_attention_spans_total
        .saturating_sub(kernel_before.chunk_attention_spans_total);
    let chunk_tokens_delta = kernel_after
        .chunk_attention_tokens_total
        .saturating_sub(kernel_before.chunk_attention_tokens_total);
    let chunk_fused_delta = kernel_after
        .chunk_attention_fused_spans_total
        .saturating_sub(kernel_before.chunk_attention_fused_spans_total);
    let chunk_unfused_delta = kernel_after
        .chunk_attention_unfused_spans_total
        .saturating_sub(kernel_before.chunk_attention_unfused_spans_total);
    let chunk_fallback_delta = kernel_after
        .chunk_attention_mask_fallback_total
        .saturating_sub(kernel_before.chunk_attention_mask_fallback_total);
    let rope_kernel_delta = kernel_after
        .rope_kernel_total
        .saturating_sub(kernel_before.rope_kernel_total);
    let rope_manual_delta = kernel_after
        .rope_manual_total
        .saturating_sub(kernel_before.rope_manual_total);
    let fused_attempts_delta = kernel_after
        .fused_attention_attempts_total
        .saturating_sub(kernel_before.fused_attention_attempts_total);
    let fused_success_delta = kernel_after
        .fused_attention_success_total
        .saturating_sub(kernel_before.fused_attention_success_total);
    let fused_fallback_delta = kernel_after
        .fused_attention_fallback_total
        .saturating_sub(kernel_before.fused_attention_fallback_total);
    let fused_masked_attempts_delta = kernel_after
        .fused_attention_masked_attempts_total
        .saturating_sub(kernel_before.fused_attention_masked_attempts_total);
    let fused_masked_success_delta = kernel_after
        .fused_attention_masked_success_total
        .saturating_sub(kernel_before.fused_attention_masked_success_total);
    let fused_masked_fallback_delta = kernel_after
        .fused_attention_masked_fallback_total
        .saturating_sub(kernel_before.fused_attention_masked_fallback_total);
    let fused_mask_policy_disabled_delta = kernel_after
        .fused_attention_fallback_metal_sdpa_mask_policy_disabled_total
        .saturating_sub(
            kernel_before.fused_attention_fallback_metal_sdpa_mask_policy_disabled_total,
        );
    let fused_mask_shape_unsupported_delta = kernel_after
        .fused_attention_fallback_metal_sdpa_mask_shape_unsupported_total
        .saturating_sub(
            kernel_before.fused_attention_fallback_metal_sdpa_mask_shape_unsupported_total,
        );
    let fused_mask_dtype_unsupported_delta = kernel_after
        .fused_attention_fallback_metal_sdpa_mask_dtype_unsupported_total
        .saturating_sub(
            kernel_before.fused_attention_fallback_metal_sdpa_mask_dtype_unsupported_total,
        );
    let decode_total = dense_decode_delta + paged_decode_delta;
    println!(
        "  Prefill path counts (token-mode/sequence-spans/sequence-tokens): {} / {} / {}",
        prefill_token_mode_delta, prefill_sequence_spans_delta, prefill_sequence_tokens_delta
    );
    println!(
        "  Decode path counts (dense/paged): {} / {}",
        dense_decode_delta, paged_decode_delta
    );
    println!(
        "  Chunk attention (calls/spans/tokens): {} / {} / {}",
        chunk_sequence_delta, chunk_spans_delta, chunk_tokens_delta
    );
    println!(
        "  Chunk attention (fused/unfused/fallback): {} / {} / {}",
        chunk_fused_delta, chunk_unfused_delta, chunk_fallback_delta
    );
    if decode_total > 0 {
        println!(
            "  Decode path share (dense/paged): {:.2}% / {:.2}%",
            100.0 * dense_decode_delta as f64 / decode_total as f64,
            100.0 * paged_decode_delta as f64 / decode_total as f64
        );
    }
    println!(
        "  RoPE path counts (kernel/manual): {} / {}",
        rope_kernel_delta, rope_manual_delta
    );
    println!(
        "  Fused attention (attempt/success/fallback): {} / {} / {}",
        fused_attempts_delta, fused_success_delta, fused_fallback_delta
    );
    println!(
        "  Fused masked attention (attempt/success/fallback): {} / {} / {}",
        fused_masked_attempts_delta, fused_masked_success_delta, fused_masked_fallback_delta
    );
    println!(
        "  Masked fused fallback reasons (policy/shape/dtype): {} / {} / {}",
        fused_mask_policy_disabled_delta,
        fused_mask_shape_unsupported_delta,
        fused_mask_dtype_unsupported_delta
    );
    if matches!(context, RuntimeTelemetryContext::AsrWhisper) {
        println!(
            "  Kernel-path note: fused-attention/RoPE counters track shared LLM/TTS paths and are not Whisper decoder proxies."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn chat_stream_event_records_first_delta_and_terminal_usage() {
        let started = Instant::now();
        let mut first_delta_at = None;
        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;
        let mut generation_time_ms = None;

        let delta = r#"data: {"choices":[{"delta":{"content":"hello"}}],"usage":null,"izwi_generation_time_ms":null}"#;
        let terminal = r#"data: {"choices":[{"delta":{"content":null}}],"usage":{"prompt_tokens":42,"completion_tokens":7},"izwi_generation_time_ms":123.0}"#;

        let sample = handle_chat_stream_event(
            delta,
            started,
            &mut first_delta_at,
            &mut prompt_tokens,
            &mut completion_tokens,
            &mut generation_time_ms,
        )
        .expect("delta should parse");
        assert!(sample.is_none());
        assert!(first_delta_at.is_some());

        let sample = handle_chat_stream_event(
            terminal,
            started,
            &mut first_delta_at,
            &mut prompt_tokens,
            &mut completion_tokens,
            &mut generation_time_ms,
        )
        .expect("terminal should parse")
        .expect("terminal event should finish sample");
        assert_eq!(sample.prompt_tokens, 42);
        assert_eq!(sample.completion_tokens, 7);
        assert_eq!(sample.generation_time_ms, Some(123.0));
        assert!(sample.ttft_ms >= 0.0);
    }

    #[test]
    fn whisper_stage_timing_collection_includes_chunk_model_diagnostics() {
        let diagnostics = serde_json::json!({
            "chunking": {
                "chunk_transcriptions": [
                    {
                        "model_diagnostics": {
                            "model_family": "whisper_asr",
                            "timings_ms": {
                                "decode": 10.0,
                                "model_total": 20.0
                            }
                        }
                    }
                ]
            }
        });

        assert!(diagnostics_contains_whisper_model(&diagnostics));
        let samples = collect_whisper_stage_timings_from_diagnostics(&diagnostics);

        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].decode, Some(10.0));
        assert_eq!(samples[0].model_total, Some(20.0));
    }

    #[tokio::test]
    async fn compare_matches_suite_cases_by_name_not_order() {
        let dir = tempdir().expect("temp dir should be created");
        let baseline = dir.path().join("baseline.json");
        let current = dir.path().join("current.json");

        let baseline_report = serde_json::json!({
            "reports": [
                {
                    "name": "fast-case",
                    "report": {
                        "summary": {
                            "latency_ms": { "p95": 100.0 }
                        }
                    }
                },
                {
                    "name": "slow-case",
                    "report": {
                        "summary": {
                            "latency_ms": { "p95": 1000.0 }
                        }
                    }
                }
            ]
        });
        let current_report = serde_json::json!({
            "reports": [
                {
                    "name": "slow-case",
                    "report": {
                        "summary": {
                            "latency_ms": { "p95": 1000.0 }
                        }
                    }
                },
                {
                    "name": "fast-case",
                    "report": {
                        "summary": {
                            "latency_ms": { "p95": 100.0 }
                        }
                    }
                }
            ]
        });
        std::fs::write(
            &baseline,
            serde_json::to_vec(&baseline_report).expect("baseline should serialize"),
        )
        .expect("baseline should be written");
        std::fs::write(
            &current,
            serde_json::to_vec(&current_report).expect("current should serialize"),
        )
        .expect("current should be written");

        bench_compare(
            &current,
            &baseline,
            5.0,
            &BenchOptions {
                output_format: OutputFormat::Json,
                quiet: true,
            },
        )
        .await
        .expect("reordered same-name suite cases should compare successfully");
    }

    #[test]
    fn duplicate_suite_case_names_are_rejected() {
        let report = serde_json::json!({
            "reports": [
                {
                    "name": "duplicate",
                    "report": {
                        "summary": {
                            "latency_ms": { "p95": 100.0 }
                        }
                    }
                },
                {
                    "name": "duplicate",
                    "report": {
                        "summary": {
                            "latency_ms": { "p95": 101.0 }
                        }
                    }
                }
            ]
        });

        let entries = report_entries(&report).expect("suite entries should parse");
        let err = report_entry_map(entries, "Current").expect_err("duplicates should fail");
        assert!(format!("{err}").contains("duplicate benchmark case name `duplicate`"));
    }

    #[test]
    fn manifest_matrix_expands_cartesian_cases() {
        let manifest: BenchmarkManifest = toml::from_str(
            r#"
[[benchmarks]]
name = "chat-short"
command = "chat"
prompt = "hello"
iterations = 1

[benchmarks.matrix]
model = ["m1", "m2"]
concurrent = [1, 2]
"#,
        )
        .expect("manifest should parse");

        let cases = expand_manifest_cases(&manifest).expect("matrix should expand");
        let names: Vec<_> = cases
            .iter()
            .map(|case| case.name.as_deref().expect("expanded cases are named"))
            .collect();
        assert_eq!(
            names,
            vec![
                "chat-short[model=m1,concurrent=1]",
                "chat-short[model=m1,concurrent=2]",
                "chat-short[model=m2,concurrent=1]",
                "chat-short[model=m2,concurrent=2]",
            ]
        );
        assert_eq!(cases[0].model.as_deref(), Some("m1"));
        assert_eq!(cases[0].concurrent, Some(1));
        assert_eq!(cases[3].model.as_deref(), Some("m2"));
        assert_eq!(cases[3].concurrent, Some(2));
    }

    #[test]
    fn manifest_matrix_rejects_duplicate_expanded_names() {
        let manifest: BenchmarkManifest = toml::from_str(
            r#"
[[benchmarks]]
name = "chat-short"
command = "chat"

[benchmarks.matrix]
concurrent = [1, 1]
"#,
        )
        .expect("manifest should parse");

        let err = expand_manifest_cases(&manifest).expect_err("duplicate matrix names should fail");
        assert!(format!("{err}").contains("duplicate case name `chat-short[concurrent=1]`"));
    }

    #[test]
    fn throughput_rps_uses_measured_elapsed() {
        assert_eq!(throughput_rps(10, Duration::from_millis(2500)), 4.0);
        assert_eq!(throughput_rps(10, Duration::ZERO), 0.0);
    }
}
