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
use std::path::{Path, PathBuf};
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
    fused_attention_fallback_flash_not_requested_total: u64,
    #[serde(default)]
    fused_attention_fallback_flash_not_compiled_total: u64,
    #[serde(default)]
    fused_attention_fallback_flash_mask_unsupported_total: u64,
    #[serde(default)]
    fused_attention_fallback_flash_dtype_unsupported_total: u64,
    #[serde(default)]
    fused_attention_fallback_flash_dtype_mismatch_total: u64,
    #[serde(default)]
    fused_attention_fallback_flash_runtime_error_total: u64,
    #[serde(default)]
    fused_attention_fallback_metal_sdpa_runtime_error_total: u64,
    #[serde(default)]
    fused_attention_fallback_metal_sdpa_mask_policy_disabled_total: u64,
    #[serde(default)]
    fused_attention_fallback_metal_sdpa_mask_shape_unsupported_total: u64,
    #[serde(default)]
    fused_attention_fallback_metal_sdpa_mask_dtype_unsupported_total: u64,
    #[serde(default)]
    fused_attention_fallback_unsupported_backend_total: u64,
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
    diagnostics: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default)]
struct TtsBenchReference {
    speaker: Option<String>,
    saved_voice_id: Option<String>,
    reference_audio_base64: Option<String>,
    reference_audio_path: Option<String>,
    reference_text: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct AsrBenchResponse {
    #[serde(default)]
    text: Option<String>,
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

#[derive(Debug, Clone)]
struct TtsStageTimings {
    prompt_build: Option<f64>,
    prompt_embed: Option<f64>,
    prefill: Option<f64>,
    text_sampling: Option<f64>,
    tokenizer_decode: Option<f64>,
    text_forward: Option<f64>,
    audio_head: Option<f64>,
    audio_head_depth_linear: Option<f64>,
    audio_head_depth_reshape: Option<f64>,
    audio_head_cache_setup: Option<f64>,
    audio_head_codebook_input: Option<f64>,
    audio_head_depthformer: Option<f64>,
    audio_head_sample: Option<f64>,
    audio_head_embed_step: Option<f64>,
    audio_head_materialize: Option<f64>,
    audio_embed: Option<f64>,
    audio_forward: Option<f64>,
    main_backbone: Option<f64>,
    detokenizer: Option<f64>,
    detokenizer_embedding: Option<f64>,
    detokenizer_upsample: Option<f64>,
    detokenizer_backbone: Option<f64>,
    detokenizer_projection: Option<f64>,
    detokenizer_waveform_prepare: Option<f64>,
    detokenizer_readback: Option<f64>,
    detokenizer_istft: Option<f64>,
    model_total: Option<f64>,
    prompt_tokens: Option<u64>,
    generated_tokens: Option<u64>,
    audio_frames: Option<u64>,
    audio_head_calls: Option<u64>,
    audio_head_codebook_steps: Option<u64>,
    text_sample_calls: Option<u64>,
}

#[derive(Debug, Clone)]
struct AsrStageTimings {
    audio_decode: Option<f64>,
    mel_prepare: Option<f64>,
    feature_extract: Option<f64>,
    feature_upload: Option<f64>,
    subsample: Option<f64>,
    encoder_forward: Option<f64>,
    encoder_ffn: Option<f64>,
    encoder_attention: Option<f64>,
    encoder_conv: Option<f64>,
    encoder_norm: Option<f64>,
    prompt_kernel: Option<f64>,
    language_detect: Option<f64>,
    resample: Option<f64>,
    mel: Option<f64>,
    mel_flatten_upload: Option<f64>,
    audio_encode: Option<f64>,
    prompt_embed: Option<f64>,
    prompt_concat: Option<f64>,
    prefill: Option<f64>,
    decode: Option<f64>,
    decode_argmax: Option<f64>,
    decode_token_tensor: Option<f64>,
    decode_forward: Option<f64>,
    tokenizer_decode: Option<f64>,
    main_backbone: Option<f64>,
    model_total: Option<f64>,
    prompt_tokens: Option<u64>,
    audio_tokens: Option<u64>,
    generated_tokens: Option<u64>,
    max_new_tokens: Option<u64>,
    rnnt_joint_steps: Option<u64>,
    token_select_reads: Option<u64>,
    host_argmax_reads: Option<u64>,
    device_argmax_reads: Option<u64>,
    execution: Option<AsrExecutionDiagnostics>,
    qwen_profile: Option<AsrQwenProfileDiagnostics>,
}

#[derive(Debug, Clone, Copy)]
struct AsrDecodeProfileTimings {
    steps: Option<u64>,
    step_total_avg_ms: Option<f64>,
    step_total_p95_ms: Option<f64>,
    loop_argmax_ms: Option<f64>,
    loop_scalar_read_ms: Option<f64>,
    loop_model_forward_ms: Option<f64>,
    forward_token_embedding_ms: Option<f64>,
    forward_rope_build_ms: Option<f64>,
    forward_layers_total_ms: Option<f64>,
    forward_final_norm_ms: Option<f64>,
    forward_lm_head_ms: Option<f64>,
    decoder_total_ms: Option<f64>,
    attention_qkv_ms: Option<f64>,
    attention_rope_ms: Option<f64>,
    attention_cache_ms: Option<f64>,
    attention_kernel_ms: Option<f64>,
    attention_output_ms: Option<f64>,
    mlp_gate_up_ms: Option<f64>,
    mlp_activation_ms: Option<f64>,
    mlp_down_ms: Option<f64>,
    residual_ms: Option<f64>,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct AsrExecutionDiagnostics {
    flash_attention_requested: Option<bool>,
    flash_attention_compiled: Option<bool>,
    kv_page_size: Option<u64>,
    cuda_dense_decode_cache: Option<bool>,
    dense_head_decode_enabled: Option<bool>,
    qkv_projection_fused: Option<bool>,
    gate_up_projection_fused: Option<bool>,
    rope_cache_precomputed: Option<bool>,
    dense_decode_max_tokens: Option<u64>,
    gguf_qmatmul_text_enabled: Option<bool>,
    text_projection_quantized: Option<bool>,
    qmatmul_projection_count: Option<u64>,
    dense_projection_count: Option<u64>,
    dense_bias_projection_count: Option<u64>,
    audio_embedding_cache_hit: Option<bool>,
    cuda_device_argmax: Option<bool>,
    residual_branches_prescaled: Option<bool>,
    dense_decode_preallocated: Option<bool>,
    dense_decode_initial_capacity: Option<u64>,
    deferred_stop_check: Option<bool>,
    chunked_stop_check: Option<bool>,
    stop_check_interval: Option<u64>,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct AsrQwenProfileDiagnostics {
    qwen3_profile_enabled: Option<bool>,
    qmatmul_calls: Option<u64>,
    qmatmul_ms: Option<f64>,
    qmatmul_input_casts: Option<u64>,
    qmatmul_input_cast_ms: Option<f64>,
    qmatmul_output_casts: Option<u64>,
    qmatmul_output_cast_ms: Option<f64>,
    lm_head_calls: Option<u64>,
    lm_head_ms: Option<f64>,
    silu_mul_fused_calls: Option<u64>,
    silu_mul_fallback_calls: Option<u64>,
    argmax_calls: Option<u64>,
    argmax_ms: Option<f64>,
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
    speaker: Option<String>,
    file: Option<String>,
    saved_voice_id: Option<String>,
    reference_audio: Option<String>,
    reference_text: Option<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    tts_diagnostics: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    asr_execution: Option<AsrExecutionDiagnostics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    asr_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    asr_diagnostics: Option<serde_json::Value>,
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
    speaker: Option<String>,
    file: Option<String>,
    saved_voice_id: Option<String>,
    reference_audio: Option<String>,
    reference_text: Option<String>,
    reference_text_file: Option<String>,
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
    speaker: Option<Vec<String>>,
    file: Option<Vec<String>>,
    saved_voice_id: Option<Vec<String>>,
    reference_audio: Option<Vec<String>>,
    reference_text: Option<Vec<String>>,
    reference_text_file: Option<Vec<String>>,
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
    Speaker(String),
    File(String),
    SavedVoiceId(String),
    ReferenceAudio(String),
    ReferenceText(String),
    ReferenceTextFile(String),
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
            speaker,
            saved_voice_id,
            reference_audio,
            reference_text,
            reference_text_file,
            concurrent,
            warmup,
        } => bench_tts(
            server,
            &model,
            iterations,
            &text,
            speaker.as_deref(),
            saved_voice_id.as_deref(),
            reference_audio.as_deref(),
            reference_text.as_deref(),
            reference_text_file.as_deref(),
            concurrent,
            warmup,
            &options,
            theme,
        )
        .await
        .and_then(|report| emit_report(&options, &report)),
        BenchCommands::Asr {
            model,
            iterations,
            file,
            language,
            max_tokens,
            concurrent,
            warmup,
        } => bench_asr(
            server,
            &model,
            iterations,
            file,
            language.as_deref(),
            max_tokens,
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
            MatrixValue::Speaker(value) => case.speaker = Some(value.clone()),
            MatrixValue::File(value) => case.file = Some(value.clone()),
            MatrixValue::SavedVoiceId(value) => case.saved_voice_id = Some(value.clone()),
            MatrixValue::ReferenceAudio(value) => case.reference_audio = Some(value.clone()),
            MatrixValue::ReferenceText(value) => case.reference_text = Some(value.clone()),
            MatrixValue::ReferenceTextFile(value) => case.reference_text_file = Some(value.clone()),
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
            | MatrixValue::Speaker(value)
            | MatrixValue::File(value)
            | MatrixValue::SavedVoiceId(value)
            | MatrixValue::ReferenceAudio(value)
            | MatrixValue::ReferenceText(value)
            | MatrixValue::ReferenceTextFile(value)
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
        add_matrix_dimension(
            &mut dimensions,
            "speaker",
            &self.speaker,
            MatrixValue::Speaker,
        )?;
        add_matrix_dimension(&mut dimensions, "file", &self.file, MatrixValue::File)?;
        add_matrix_dimension(
            &mut dimensions,
            "saved_voice_id",
            &self.saved_voice_id,
            MatrixValue::SavedVoiceId,
        )?;
        add_matrix_dimension(
            &mut dimensions,
            "reference_audio",
            &self.reference_audio,
            MatrixValue::ReferenceAudio,
        )?;
        add_matrix_dimension(
            &mut dimensions,
            "reference_text",
            &self.reference_text,
            MatrixValue::ReferenceText,
        )?;
        add_matrix_dimension(
            &mut dimensions,
            "reference_text_file",
            &self.reference_text_file,
            MatrixValue::ReferenceTextFile,
        )?;
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

fn resolve_manifest_path(raw: Option<&str>, manifest_dir: &Path) -> Option<PathBuf> {
    raw.map(|file| {
        let path = Path::new(file);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            manifest_dir.join(path)
        }
    })
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
                let reference_audio =
                    resolve_manifest_path(case.reference_audio.as_deref(), manifest_dir);
                let reference_text_file =
                    resolve_manifest_path(case.reference_text_file.as_deref(), manifest_dir);
                bench_tts(
                    &suite_server,
                    model,
                    case.iterations.unwrap_or(10),
                    text,
                    case.speaker.as_deref(),
                    case.saved_voice_id.as_deref(),
                    reference_audio.as_deref(),
                    case.reference_text.as_deref(),
                    reference_text_file.as_deref(),
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
                    case.max_tokens,
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
            speaker: None,
            file: None,
            saved_voice_id: None,
            reference_audio: None,
            reference_text: None,
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
                tts_diagnostics: None,
                asr_execution: None,
                asr_text: None,
                asr_diagnostics: None,
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
    speaker: Option<&str>,
    saved_voice_id: Option<&str>,
    reference_audio: Option<&Path>,
    reference_text: Option<&str>,
    reference_text_file: Option<&Path>,
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
    let reference = resolve_tts_bench_reference(
        speaker,
        saved_voice_id,
        reference_audio,
        reference_text,
        reference_text_file,
    )
    .await?;

    if warmup {
        if options.interactive() {
            theme.info("Running warmup iteration...");
        }
        let _ = run_tts_request(server, model, text, &reference).await?;
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
    let reference = Arc::new(reference);
    let wall_start = Instant::now();
    let samples: Vec<TtsBenchSample> = stream::iter(0..iterations)
        .map(|_| {
            let progress = Arc::clone(&progress);
            let text = Arc::clone(&text);
            let reference = Arc::clone(&reference);
            let model = model.to_string();
            let server = server.to_string();
            async move {
                let start = Instant::now();
                let result = run_tts_request(&server, &model, text.as_str(), reference.as_ref())
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
    let tts_stage_samples: Vec<TtsStageTimings> = samples
        .iter()
        .filter_map(|sample| sample.diagnostics.as_ref())
        .filter_map(tts_stage_timings_from_diagnostics)
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
        print_tts_stage_timing_summary(&tts_stage_samples);
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
            speaker: reference.speaker.clone(),
            file: None,
            saved_voice_id: reference.saved_voice_id.clone(),
            reference_audio: reference.reference_audio_path.clone(),
            reference_text: reference.reference_text.clone(),
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
                tts_diagnostics: sample.diagnostics.clone(),
                asr_execution: None,
                asr_text: None,
                asr_diagnostics: None,
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
    max_tokens: Option<usize>,
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
    if let Some(max_tokens) = max_tokens {
        if max_tokens == 0 {
            return Err(CliError::InvalidInput(
                "ASR max tokens must be greater than 0".to_string(),
            ));
        }
        if options.interactive() {
            theme.info(&format!("Using ASR max tokens: {}", max_tokens));
        }
    }

    if warmup {
        if options.interactive() {
            theme.info("Running warmup iteration...");
        }
        let _ = run_asr_request(server, model, &audio_base64, language, max_tokens).await?;
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
                    max_tokens,
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
    let mut decode_profile_samples = Vec::new();
    let mut saw_whisper_diagnostics = false;
    for sample in &samples {
        if let Some(diagnostics) = sample.response.izwi_asr_diagnostics.as_ref() {
            if diagnostics_contains_whisper_model(diagnostics) {
                saw_whisper_diagnostics = true;
            }
            stage_samples.extend(collect_asr_stage_timings_from_diagnostics(diagnostics));
            decode_profile_samples
                .extend(collect_asr_decode_profiles_from_diagnostics(diagnostics));
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
        print_asr_stage_timing_summary(&stage_samples);
        print_asr_decode_profile_summary(&decode_profile_samples);
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
            max_tokens,
            text: None,
            speaker: None,
            file: Some(audio_file.display().to_string()),
            saved_voice_id: None,
            reference_audio: None,
            reference_text: None,
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
                tts_diagnostics: None,
                asr_execution: sample
                    .response
                    .izwi_asr_diagnostics
                    .as_ref()
                    .and_then(asr_execution_from_diagnostics),
                asr_text: sample.response.text.clone(),
                asr_diagnostics: sample.response.izwi_asr_diagnostics.clone(),
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
            speaker: None,
            file: None,
            saved_voice_id: None,
            reference_audio: None,
            reference_text: None,
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

fn header_json(response: &reqwest::Response, name: &'static str) -> Option<serde_json::Value> {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| serde_json::from_str(value).ok())
}

async fn resolve_tts_bench_reference(
    speaker: Option<&str>,
    saved_voice_id: Option<&str>,
    reference_audio: Option<&Path>,
    reference_text: Option<&str>,
    reference_text_file: Option<&Path>,
) -> Result<TtsBenchReference> {
    let speaker = normalize_optional_bench_text(speaker);
    let saved_voice_id = normalize_optional_bench_text(saved_voice_id);
    let reference_text = normalize_optional_bench_text(reference_text);
    if reference_text.is_some() && reference_text_file.is_some() {
        return Err(CliError::InvalidInput(
            "Use either --reference-text or --reference-text-file, not both.".to_string(),
        ));
    }
    let reference_text = match (reference_text, reference_text_file) {
        (Some(text), None) => Some(text),
        (None, Some(path)) => normalize_optional_bench_string(
            tokio::fs::read_to_string(path)
                .await
                .map_err(CliError::Io)?,
        ),
        (None, None) => None,
        (Some(_), Some(_)) => unreachable!("checked above"),
    };

    if saved_voice_id.is_some() && (reference_audio.is_some() || reference_text.is_some()) {
        return Err(CliError::InvalidInput(
            "Use either --saved-voice-id or --reference-audio/--reference-text, not both."
                .to_string(),
        ));
    }
    if reference_audio.is_some() != reference_text.is_some() {
        return Err(CliError::InvalidInput(
            "Provide --reference-audio and --reference-text together.".to_string(),
        ));
    }

    let (reference_audio_base64, reference_audio_path) = match reference_audio {
        Some(path) => {
            let audio = tokio::fs::read(path).await.map_err(CliError::Io)?;
            (
                Some(STANDARD.encode(audio)),
                Some(path.display().to_string()),
            )
        }
        None => (None, None),
    };

    Ok(TtsBenchReference {
        speaker,
        saved_voice_id,
        reference_audio_base64,
        reference_audio_path,
        reference_text,
    })
}

fn normalize_optional_bench_text(value: Option<&str>) -> Option<String> {
    value.and_then(|value| normalize_optional_bench_string(value.to_string()))
}

fn normalize_optional_bench_string(value: String) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

async fn run_tts_request(
    server: &str,
    model: &str,
    text: &str,
    reference: &TtsBenchReference,
) -> Result<TtsBenchSample> {
    let client = http::client(Some(std::time::Duration::from_secs(300)))?;
    let request_body = build_tts_bench_request_body(model, text, reference);

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
        diagnostics: header_json(&response, "x-izwi-tts-diagnostics"),
    })
}

fn build_tts_bench_request_body(
    model: &str,
    text: &str,
    reference: &TtsBenchReference,
) -> serde_json::Value {
    let mut request_body = serde_json::json!({
        "model": model,
        "input": text,
        "response_format": "wav",
    });
    if let Some(speaker) = reference.speaker.as_deref() {
        request_body["voice"] = serde_json::Value::String(speaker.to_string());
    }
    if let Some(saved_voice_id) = reference.saved_voice_id.as_deref() {
        request_body["saved_voice_id"] = serde_json::Value::String(saved_voice_id.to_string());
    }
    if let Some(reference_audio) = reference.reference_audio_base64.as_deref() {
        request_body["reference_audio"] = serde_json::Value::String(reference_audio.to_string());
    }
    if let Some(reference_text) = reference.reference_text.as_deref() {
        request_body["reference_text"] = serde_json::Value::String(reference_text.to_string());
    }
    request_body
}

async fn run_asr_request(
    server: &str,
    model: &str,
    audio_base64: &str,
    language: Option<&str>,
    max_tokens: Option<usize>,
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
    if let Some(max_tokens) = max_tokens {
        request_body["max_tokens"] = serde_json::json!(max_tokens);
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

fn tts_stage_timings_from_diagnostics(diagnostics: &serde_json::Value) -> Option<TtsStageTimings> {
    let timings = diagnostics.get("timings_ms")?.as_object()?;
    let prompt = diagnostics.get("prompt");
    let decode = diagnostics.get("decode");
    let audio = diagnostics.get("audio");
    Some(TtsStageTimings {
        prompt_build: timings.get("prompt_build").and_then(|value| value.as_f64()),
        prompt_embed: timings.get("prompt_embed").and_then(|value| value.as_f64()),
        prefill: timings
            .get("prefill")
            .or_else(|| timings.get("main_prefill"))
            .and_then(|value| value.as_f64()),
        text_sampling: timings
            .get("text_sampling")
            .and_then(|value| value.as_f64()),
        tokenizer_decode: timings
            .get("tokenizer_decode")
            .and_then(|value| value.as_f64()),
        text_forward: timings.get("text_forward").and_then(|value| value.as_f64()),
        audio_head: timings.get("audio_head").and_then(|value| value.as_f64()),
        audio_head_depth_linear: timings
            .get("audio_head_depth_linear")
            .and_then(|value| value.as_f64()),
        audio_head_depth_reshape: timings
            .get("audio_head_depth_reshape")
            .and_then(|value| value.as_f64()),
        audio_head_cache_setup: timings
            .get("audio_head_cache_setup")
            .and_then(|value| value.as_f64()),
        audio_head_codebook_input: timings
            .get("audio_head_codebook_input")
            .and_then(|value| value.as_f64()),
        audio_head_depthformer: timings
            .get("audio_head_depthformer")
            .and_then(|value| value.as_f64()),
        audio_head_sample: timings
            .get("audio_head_sample")
            .and_then(|value| value.as_f64()),
        audio_head_embed_step: timings
            .get("audio_head_embed_step")
            .and_then(|value| value.as_f64()),
        audio_head_materialize: timings
            .get("audio_head_materialize")
            .and_then(|value| value.as_f64()),
        audio_embed: timings.get("audio_embed").and_then(|value| value.as_f64()),
        audio_forward: timings
            .get("audio_forward")
            .and_then(|value| value.as_f64()),
        main_backbone: timings
            .get("main_backbone")
            .and_then(|value| value.as_f64()),
        detokenizer: timings.get("detokenizer").and_then(|value| value.as_f64()),
        detokenizer_embedding: timings
            .get("detokenizer_embedding")
            .and_then(|value| value.as_f64()),
        detokenizer_upsample: timings
            .get("detokenizer_upsample")
            .and_then(|value| value.as_f64()),
        detokenizer_backbone: timings
            .get("detokenizer_backbone")
            .and_then(|value| value.as_f64()),
        detokenizer_projection: timings
            .get("detokenizer_projection")
            .and_then(|value| value.as_f64()),
        detokenizer_waveform_prepare: timings
            .get("detokenizer_waveform_prepare")
            .and_then(|value| value.as_f64()),
        detokenizer_readback: timings
            .get("detokenizer_readback")
            .and_then(|value| value.as_f64()),
        detokenizer_istft: timings
            .get("detokenizer_istft")
            .and_then(|value| value.as_f64()),
        model_total: timings.get("model_total").and_then(|value| value.as_f64()),
        prompt_tokens: diagnostic_u64(prompt, &["prompt_tokens", "tokens"]),
        generated_tokens: diagnostic_u64(decode, &["generated_tokens"]),
        audio_frames: diagnostic_u64(audio, &["audio_frames", "acoustic_frames"]),
        audio_head_calls: diagnostic_u64(decode, &["audio_head_calls"]),
        audio_head_codebook_steps: diagnostic_u64(decode, &["audio_head_codebook_steps"]),
        text_sample_calls: diagnostic_u64(decode, &["text_sample_calls"]),
    })
}

fn asr_stage_timings_from_diagnostics(diagnostics: &serde_json::Value) -> Option<AsrStageTimings> {
    let timings = diagnostics.get("timings_ms")?.as_object()?;
    let prompt = diagnostics.get("prompt");
    let audio = diagnostics.get("audio");
    let decode = diagnostics.get("decode");
    let execution = asr_execution_from_single_diagnostics(diagnostics);
    let qwen_profile = asr_qwen_profile_from_single_diagnostics(diagnostics);
    Some(AsrStageTimings {
        audio_decode: timings.get("audio_decode").and_then(|value| value.as_f64()),
        mel_prepare: timings.get("mel_prepare").and_then(|value| value.as_f64()),
        feature_extract: timings
            .get("feature_extract")
            .and_then(|value| value.as_f64()),
        feature_upload: timings
            .get("feature_upload")
            .and_then(|value| value.as_f64()),
        subsample: timings.get("subsample").and_then(|value| value.as_f64()),
        encoder_forward: timings
            .get("encoder_forward")
            .and_then(|value| value.as_f64()),
        encoder_ffn: timings.get("encoder_ffn").and_then(|value| value.as_f64()),
        encoder_attention: timings
            .get("encoder_attention")
            .and_then(|value| value.as_f64()),
        encoder_conv: timings.get("encoder_conv").and_then(|value| value.as_f64()),
        encoder_norm: timings.get("encoder_norm").and_then(|value| value.as_f64()),
        prompt_kernel: timings
            .get("prompt_kernel")
            .and_then(|value| value.as_f64()),
        language_detect: timings
            .get("language_detect")
            .and_then(|value| value.as_f64()),
        resample: timings.get("resample").and_then(|value| value.as_f64()),
        mel: timings.get("mel").and_then(|value| value.as_f64()),
        mel_flatten_upload: timings
            .get("mel_flatten_upload")
            .and_then(|value| value.as_f64()),
        audio_encode: timings.get("audio_encode").and_then(|value| value.as_f64()),
        prompt_embed: timings.get("prompt_embed").and_then(|value| value.as_f64()),
        prompt_concat: timings
            .get("prompt_concat")
            .and_then(|value| value.as_f64()),
        prefill: timings.get("prefill").and_then(|value| value.as_f64()),
        decode: timings.get("decode").and_then(|value| value.as_f64()),
        decode_argmax: timings
            .get("decode_argmax")
            .and_then(|value| value.as_f64()),
        decode_token_tensor: timings
            .get("decode_token_tensor")
            .and_then(|value| value.as_f64()),
        decode_forward: timings
            .get("decode_forward")
            .and_then(|value| value.as_f64()),
        tokenizer_decode: timings
            .get("tokenizer_decode")
            .and_then(|value| value.as_f64()),
        main_backbone: timings
            .get("main_backbone")
            .and_then(|value| value.as_f64()),
        model_total: timings.get("model_total").and_then(|value| value.as_f64()),
        prompt_tokens: diagnostic_u64(prompt, &["prompt_tokens", "tokens"]),
        audio_tokens: diagnostic_u64(audio, &["audio_tokens", "acoustic_frames"]),
        generated_tokens: decode
            .and_then(|value| {
                value
                    .get("generated_tokens")
                    .or_else(|| value.get("generated_token_count"))
            })
            .and_then(|value| value.as_u64()),
        max_new_tokens: decode
            .and_then(|value| {
                value
                    .get("max_new_tokens")
                    .or_else(|| value.get("max_steps"))
            })
            .and_then(|value| value.as_u64()),
        rnnt_joint_steps: diagnostic_u64(decode, &["joint_steps", "rnnt_joint_steps"]),
        token_select_reads: diagnostic_u64(decode, &["token_select_reads"]),
        host_argmax_reads: diagnostic_u64(decode, &["host_argmax_reads"]),
        device_argmax_reads: diagnostic_u64(decode, &["device_argmax_reads"]),
        execution,
        qwen_profile,
    })
}

fn diagnostic_u64(value: Option<&serde_json::Value>, keys: &[&str]) -> Option<u64> {
    let value = value?;
    keys.iter()
        .find_map(|key| value.get(*key).and_then(|value| value.as_u64()))
}

fn asr_execution_from_diagnostics(
    diagnostics: &serde_json::Value,
) -> Option<AsrExecutionDiagnostics> {
    collect_asr_stage_timings_from_diagnostics(diagnostics)
        .into_iter()
        .find_map(|sample| sample.execution)
}

fn asr_execution_from_single_diagnostics(
    diagnostics: &serde_json::Value,
) -> Option<AsrExecutionDiagnostics> {
    let execution = diagnostics.get("execution")?;
    let sample = AsrExecutionDiagnostics {
        flash_attention_requested: execution
            .get("flash_attention_requested")
            .and_then(|value| value.as_bool()),
        flash_attention_compiled: execution
            .get("flash_attention_compiled")
            .and_then(|value| value.as_bool()),
        kv_page_size: execution
            .get("kv_page_size")
            .and_then(|value| value.as_u64()),
        cuda_dense_decode_cache: execution
            .get("cuda_dense_decode_cache")
            .and_then(|value| value.as_bool()),
        dense_head_decode_enabled: execution
            .get("dense_head_decode_enabled")
            .or_else(|| execution.get("dense_decode_enabled"))
            .and_then(|value| value.as_bool()),
        qkv_projection_fused: execution
            .get("qkv_projection_fused")
            .and_then(|value| value.as_bool()),
        gate_up_projection_fused: execution
            .get("gate_up_projection_fused")
            .and_then(|value| value.as_bool()),
        rope_cache_precomputed: execution
            .get("rope_cache_precomputed")
            .and_then(|value| value.as_bool()),
        dense_decode_max_tokens: execution
            .get("dense_decode_max_tokens")
            .and_then(|value| value.as_u64()),
        gguf_qmatmul_text_enabled: execution
            .get("gguf_qmatmul_text_enabled")
            .and_then(|value| value.as_bool()),
        text_projection_quantized: execution
            .get("text_projection_quantized")
            .and_then(|value| value.as_bool()),
        qmatmul_projection_count: execution
            .get("qmatmul_projection_count")
            .and_then(|value| value.as_u64()),
        dense_projection_count: execution
            .get("dense_projection_count")
            .and_then(|value| value.as_u64()),
        dense_bias_projection_count: execution
            .get("dense_bias_projection_count")
            .and_then(|value| value.as_u64()),
        audio_embedding_cache_hit: execution
            .get("audio_embedding_cache_hit")
            .and_then(|value| value.as_bool()),
        cuda_device_argmax: execution
            .get("cuda_device_argmax")
            .and_then(|value| value.as_bool()),
        residual_branches_prescaled: execution
            .get("residual_branches_prescaled")
            .and_then(|value| value.as_bool()),
        dense_decode_preallocated: execution
            .get("dense_decode_preallocated")
            .and_then(|value| value.as_bool()),
        dense_decode_initial_capacity: execution
            .get("dense_decode_initial_capacity")
            .and_then(|value| value.as_u64()),
        deferred_stop_check: execution
            .get("deferred_stop_check")
            .and_then(|value| value.as_bool()),
        chunked_stop_check: execution
            .get("chunked_stop_check")
            .and_then(|value| value.as_bool()),
        stop_check_interval: execution
            .get("stop_check_interval")
            .and_then(|value| value.as_u64()),
    };
    (sample.flash_attention_requested.is_some()
        || sample.flash_attention_compiled.is_some()
        || sample.kv_page_size.is_some()
        || sample.cuda_dense_decode_cache.is_some()
        || sample.dense_head_decode_enabled.is_some()
        || sample.qkv_projection_fused.is_some()
        || sample.gate_up_projection_fused.is_some()
        || sample.rope_cache_precomputed.is_some()
        || sample.dense_decode_max_tokens.is_some()
        || sample.gguf_qmatmul_text_enabled.is_some()
        || sample.text_projection_quantized.is_some()
        || sample.qmatmul_projection_count.is_some()
        || sample.dense_projection_count.is_some()
        || sample.dense_bias_projection_count.is_some()
        || sample.audio_embedding_cache_hit.is_some()
        || sample.cuda_device_argmax.is_some()
        || sample.residual_branches_prescaled.is_some()
        || sample.dense_decode_preallocated.is_some()
        || sample.dense_decode_initial_capacity.is_some()
        || sample.deferred_stop_check.is_some()
        || sample.chunked_stop_check.is_some()
        || sample.stop_check_interval.is_some())
    .then_some(sample)
}

fn asr_qwen_profile_from_single_diagnostics(
    diagnostics: &serde_json::Value,
) -> Option<AsrQwenProfileDiagnostics> {
    let profile = diagnostics.get("profile")?;
    let sample = AsrQwenProfileDiagnostics {
        qwen3_profile_enabled: profile
            .get("qwen3_profile_enabled")
            .and_then(|value| value.as_bool()),
        qmatmul_calls: profile
            .get("qmatmul_calls")
            .and_then(|value| value.as_u64()),
        qmatmul_ms: profile.get("qmatmul_ms").and_then(|value| value.as_f64()),
        qmatmul_input_casts: profile
            .get("qmatmul_input_casts")
            .and_then(|value| value.as_u64()),
        qmatmul_input_cast_ms: profile
            .get("qmatmul_input_cast_ms")
            .and_then(|value| value.as_f64()),
        qmatmul_output_casts: profile
            .get("qmatmul_output_casts")
            .and_then(|value| value.as_u64()),
        qmatmul_output_cast_ms: profile
            .get("qmatmul_output_cast_ms")
            .and_then(|value| value.as_f64()),
        lm_head_calls: profile
            .get("lm_head_calls")
            .and_then(|value| value.as_u64()),
        lm_head_ms: profile.get("lm_head_ms").and_then(|value| value.as_f64()),
        silu_mul_fused_calls: profile
            .get("silu_mul_fused_calls")
            .and_then(|value| value.as_u64()),
        silu_mul_fallback_calls: profile
            .get("silu_mul_fallback_calls")
            .and_then(|value| value.as_u64()),
        argmax_calls: profile.get("argmax_calls").and_then(|value| value.as_u64()),
        argmax_ms: profile.get("argmax_ms").and_then(|value| value.as_f64()),
    };
    (sample.qwen3_profile_enabled.unwrap_or(false)
        || sample.qmatmul_calls.unwrap_or(0) > 0
        || sample.argmax_calls.unwrap_or(0) > 0
        || sample.silu_mul_fused_calls.unwrap_or(0) > 0
        || sample.silu_mul_fallback_calls.unwrap_or(0) > 0)
        .then_some(sample)
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

fn collect_asr_stage_timings_from_diagnostics(
    diagnostics: &serde_json::Value,
) -> Vec<AsrStageTimings> {
    let mut samples = Vec::new();
    if let Some(stage_sample) = asr_stage_timings_from_diagnostics(diagnostics) {
        samples.push(stage_sample);
    }
    if let Some(model_diagnostics) = diagnostics.get("model_diagnostics") {
        if let Some(stage_sample) = asr_stage_timings_from_diagnostics(model_diagnostics) {
            samples.push(stage_sample);
        }
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
            if let Some(stage_sample) = asr_stage_timings_from_diagnostics(model_diagnostics) {
                samples.push(stage_sample);
            }
        }
    }

    samples
}

fn asr_decode_profile_from_diagnostics(
    diagnostics: &serde_json::Value,
) -> Option<AsrDecodeProfileTimings> {
    if let Some(profile) = diagnostics.get("decode_profile") {
        if profile.get("enabled").and_then(|value| value.as_bool()) != Some(true) {
            return None;
        }

        return Some(AsrDecodeProfileTimings {
            steps: profile.get("steps").and_then(|value| value.as_u64()),
            step_total_avg_ms: summary_metric(profile, &["step_total_ms", "avg"]),
            step_total_p95_ms: summary_metric(profile, &["step_total_ms", "p95"]),
            loop_argmax_ms: summary_metric(profile, &["loop_totals_ms", "argmax"]),
            loop_scalar_read_ms: summary_metric(profile, &["loop_totals_ms", "scalar_read"]),
            loop_model_forward_ms: summary_metric(profile, &["loop_totals_ms", "model_forward"]),
            forward_token_embedding_ms: summary_metric(
                profile,
                &["forward_totals_ms", "token_embedding"],
            ),
            forward_rope_build_ms: summary_metric(profile, &["forward_totals_ms", "rope_build"]),
            forward_layers_total_ms: summary_metric(
                profile,
                &["forward_totals_ms", "layers_total"],
            ),
            forward_final_norm_ms: summary_metric(profile, &["forward_totals_ms", "final_norm"]),
            forward_lm_head_ms: summary_metric(profile, &["forward_totals_ms", "lm_head"]),
            decoder_total_ms: summary_metric(profile, &["decoder_totals_ms", "total"]),
            attention_qkv_ms: summary_metric(profile, &["decoder_totals_ms", "attention", "qkv"]),
            attention_rope_ms: summary_metric(profile, &["decoder_totals_ms", "attention", "rope"]),
            attention_cache_ms: summary_metric(
                profile,
                &["decoder_totals_ms", "attention", "cache"],
            ),
            attention_kernel_ms: summary_metric(
                profile,
                &["decoder_totals_ms", "attention", "kernel"],
            ),
            attention_output_ms: summary_metric(
                profile,
                &["decoder_totals_ms", "attention", "output"],
            ),
            mlp_gate_up_ms: summary_metric(profile, &["decoder_totals_ms", "mlp", "gate_up"]),
            mlp_activation_ms: summary_metric(profile, &["decoder_totals_ms", "mlp", "activation"]),
            mlp_down_ms: summary_metric(profile, &["decoder_totals_ms", "mlp", "down"]),
            residual_ms: summary_metric(profile, &["decoder_totals_ms", "residual"]),
        });
    }

    let decode = diagnostics.get("decode")?;
    let profile = decode.get("profile")?;
    if profile.get("enabled").and_then(|value| value.as_bool()) != Some(true) {
        return None;
    }

    Some(AsrDecodeProfileTimings {
        steps: diagnostic_u64(Some(decode), &["generated_tokens", "generated_token_count"]),
        step_total_avg_ms: profile
            .get("step_total_ms")
            .and_then(|value| value.as_f64()),
        step_total_p95_ms: None,
        loop_argmax_ms: profile.get("argmax_ms").and_then(|value| value.as_f64()),
        loop_scalar_read_ms: profile
            .get("host_read_ms")
            .or_else(|| profile.get("sampling_ms"))
            .and_then(|value| value.as_f64()),
        loop_model_forward_ms: profile
            .get("decoder_forward_ms")
            .and_then(|value| value.as_f64()),
        forward_token_embedding_ms: profile
            .get("token_tensor_ms")
            .and_then(|value| value.as_f64()),
        forward_rope_build_ms: None,
        forward_layers_total_ms: None,
        forward_final_norm_ms: None,
        forward_lm_head_ms: profile
            .get("final_linear_ms")
            .and_then(|value| value.as_f64()),
        decoder_total_ms: profile
            .get("step_total_ms")
            .and_then(|value| value.as_f64()),
        attention_qkv_ms: None,
        attention_rope_ms: None,
        attention_cache_ms: None,
        attention_kernel_ms: None,
        attention_output_ms: None,
        mlp_gate_up_ms: None,
        mlp_activation_ms: None,
        mlp_down_ms: None,
        residual_ms: profile
            .get("unattributed_ms")
            .and_then(|value| value.as_f64()),
    })
}

fn collect_asr_decode_profiles_from_diagnostics(
    diagnostics: &serde_json::Value,
) -> Vec<AsrDecodeProfileTimings> {
    let mut samples = Vec::new();
    if let Some(profile) = asr_decode_profile_from_diagnostics(diagnostics) {
        samples.push(profile);
    }
    if let Some(model_diagnostics) = diagnostics.get("model_diagnostics") {
        if let Some(profile) = asr_decode_profile_from_diagnostics(model_diagnostics) {
            samples.push(profile);
        }
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
            if let Some(profile) = asr_decode_profile_from_diagnostics(model_diagnostics) {
                samples.push(profile);
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

fn summarize_count(stage: &str, values: &[u64]) {
    if values.is_empty() {
        return;
    }
    let values_f64 = values.iter().map(|value| *value as f64).collect::<Vec<_>>();
    let avg = values_f64.iter().sum::<f64>() / values_f64.len() as f64;
    let p50 = percentile(&values_f64, 0.5);
    let p95 = percentile(&values_f64, 0.95);
    println!(
        "  {:<14} avg/p50/p95: {:.1} / {:.1} / {:.1}",
        stage, avg, p50, p95
    );
}

fn summarize_bool_count(stage: &str, values: &[bool]) {
    if values.is_empty() {
        return;
    }
    let enabled = values.iter().filter(|value| **value).count();
    println!("  {:<14} enabled: {}/{}", stage, enabled, values.len());
}

fn print_tts_stage_timing_summary(samples: &[TtsStageTimings]) {
    if samples.is_empty() {
        return;
    }

    let mut prompt_build = Vec::new();
    let mut prompt_embed = Vec::new();
    let mut prefill = Vec::new();
    let mut text_sampling = Vec::new();
    let mut tokenizer_decode = Vec::new();
    let mut text_forward = Vec::new();
    let mut audio_head = Vec::new();
    let mut audio_head_depth_linear = Vec::new();
    let mut audio_head_depth_reshape = Vec::new();
    let mut audio_head_cache_setup = Vec::new();
    let mut audio_head_codebook_input = Vec::new();
    let mut audio_head_depthformer = Vec::new();
    let mut audio_head_sample = Vec::new();
    let mut audio_head_embed_step = Vec::new();
    let mut audio_head_materialize = Vec::new();
    let mut audio_embed = Vec::new();
    let mut audio_forward = Vec::new();
    let mut main_backbone = Vec::new();
    let mut detokenizer = Vec::new();
    let mut detokenizer_embedding = Vec::new();
    let mut detokenizer_upsample = Vec::new();
    let mut detokenizer_backbone = Vec::new();
    let mut detokenizer_projection = Vec::new();
    let mut detokenizer_waveform_prepare = Vec::new();
    let mut detokenizer_readback = Vec::new();
    let mut detokenizer_istft = Vec::new();
    let mut model_total = Vec::new();
    let mut prompt_tokens = Vec::new();
    let mut generated_tokens = Vec::new();
    let mut audio_frames = Vec::new();
    let mut audio_head_calls = Vec::new();
    let mut audio_head_codebook_steps = Vec::new();
    let mut text_sample_calls = Vec::new();

    for sample in samples {
        if let Some(value) = sample.prompt_build {
            prompt_build.push(value);
        }
        if let Some(value) = sample.prompt_embed {
            prompt_embed.push(value);
        }
        if let Some(value) = sample.prefill {
            prefill.push(value);
        }
        if let Some(value) = sample.text_sampling {
            text_sampling.push(value);
        }
        if let Some(value) = sample.tokenizer_decode {
            tokenizer_decode.push(value);
        }
        if let Some(value) = sample.text_forward {
            text_forward.push(value);
        }
        if let Some(value) = sample.audio_head {
            audio_head.push(value);
        }
        if let Some(value) = sample.audio_head_depth_linear {
            audio_head_depth_linear.push(value);
        }
        if let Some(value) = sample.audio_head_depth_reshape {
            audio_head_depth_reshape.push(value);
        }
        if let Some(value) = sample.audio_head_cache_setup {
            audio_head_cache_setup.push(value);
        }
        if let Some(value) = sample.audio_head_codebook_input {
            audio_head_codebook_input.push(value);
        }
        if let Some(value) = sample.audio_head_depthformer {
            audio_head_depthformer.push(value);
        }
        if let Some(value) = sample.audio_head_sample {
            audio_head_sample.push(value);
        }
        if let Some(value) = sample.audio_head_embed_step {
            audio_head_embed_step.push(value);
        }
        if let Some(value) = sample.audio_head_materialize {
            audio_head_materialize.push(value);
        }
        if let Some(value) = sample.audio_embed {
            audio_embed.push(value);
        }
        if let Some(value) = sample.audio_forward {
            audio_forward.push(value);
        }
        if let Some(value) = sample.main_backbone {
            main_backbone.push(value);
        }
        if let Some(value) = sample.detokenizer {
            detokenizer.push(value);
        }
        if let Some(value) = sample.detokenizer_embedding {
            detokenizer_embedding.push(value);
        }
        if let Some(value) = sample.detokenizer_upsample {
            detokenizer_upsample.push(value);
        }
        if let Some(value) = sample.detokenizer_backbone {
            detokenizer_backbone.push(value);
        }
        if let Some(value) = sample.detokenizer_projection {
            detokenizer_projection.push(value);
        }
        if let Some(value) = sample.detokenizer_waveform_prepare {
            detokenizer_waveform_prepare.push(value);
        }
        if let Some(value) = sample.detokenizer_readback {
            detokenizer_readback.push(value);
        }
        if let Some(value) = sample.detokenizer_istft {
            detokenizer_istft.push(value);
        }
        if let Some(value) = sample.model_total {
            model_total.push(value);
        }
        if let Some(value) = sample.prompt_tokens {
            prompt_tokens.push(value);
        }
        if let Some(value) = sample.generated_tokens {
            generated_tokens.push(value);
        }
        if let Some(value) = sample.audio_frames {
            audio_frames.push(value);
        }
        if let Some(value) = sample.audio_head_calls {
            audio_head_calls.push(value);
        }
        if let Some(value) = sample.audio_head_codebook_steps {
            audio_head_codebook_steps.push(value);
        }
        if let Some(value) = sample.text_sample_calls {
            text_sample_calls.push(value);
        }
    }

    println!(
        "\n{}",
        console::style("TTS Stage Timings (run-local):")
            .bold()
            .underlined()
    );
    summarize_stage("prompt_build", &prompt_build);
    summarize_stage("prompt_embed", &prompt_embed);
    summarize_stage("prefill", &prefill);
    summarize_stage("text_sample", &text_sampling);
    summarize_stage("tokenizer", &tokenizer_decode);
    summarize_stage("text_forward", &text_forward);
    summarize_stage("audio_head", &audio_head);
    summarize_stage("ah_depth_lin", &audio_head_depth_linear);
    summarize_stage("ah_reshape", &audio_head_depth_reshape);
    summarize_stage("ah_cache", &audio_head_cache_setup);
    summarize_stage("ah_input", &audio_head_codebook_input);
    summarize_stage("ah_depthform", &audio_head_depthformer);
    summarize_stage("ah_sample", &audio_head_sample);
    summarize_stage("ah_embed", &audio_head_embed_step);
    summarize_stage("ah_material", &audio_head_materialize);
    summarize_stage("audio_embed", &audio_embed);
    summarize_stage("audio_forward", &audio_forward);
    summarize_stage("main_backbone", &main_backbone);
    summarize_stage("detokenizer", &detokenizer);
    summarize_stage("detok_embed", &detokenizer_embedding);
    summarize_stage("detok_up", &detokenizer_upsample);
    summarize_stage("detok_backbn", &detokenizer_backbone);
    summarize_stage("detok_proj", &detokenizer_projection);
    summarize_stage("detok_prep", &detokenizer_waveform_prepare);
    summarize_stage("detok_read", &detokenizer_readback);
    summarize_stage("detok_istft", &detokenizer_istft);
    summarize_stage("model_total", &model_total);
    summarize_count("prompt_tokens", &prompt_tokens);
    summarize_count("gen_tokens", &generated_tokens);
    summarize_count("audio_frames", &audio_frames);
    summarize_count("audio_head", &audio_head_calls);
    summarize_count("ah_codebooks", &audio_head_codebook_steps);
    summarize_count("text_sample", &text_sample_calls);
}

fn print_asr_stage_timing_summary(samples: &[AsrStageTimings]) {
    if samples.is_empty() {
        return;
    }

    let mut audio_decode = Vec::new();
    let mut mel_prepare = Vec::new();
    let mut feature_extract = Vec::new();
    let mut feature_upload = Vec::new();
    let mut subsample = Vec::new();
    let mut encoder_forward = Vec::new();
    let mut encoder_ffn = Vec::new();
    let mut encoder_attention = Vec::new();
    let mut encoder_conv = Vec::new();
    let mut encoder_norm = Vec::new();
    let mut prompt_kernel = Vec::new();
    let mut language_detect = Vec::new();
    let mut resample = Vec::new();
    let mut mel = Vec::new();
    let mut mel_flatten_upload = Vec::new();
    let mut audio_encode = Vec::new();
    let mut prompt_embed = Vec::new();
    let mut prompt_concat = Vec::new();
    let mut prefill = Vec::new();
    let mut decode = Vec::new();
    let mut decode_argmax = Vec::new();
    let mut decode_token_tensor = Vec::new();
    let mut decode_forward = Vec::new();
    let mut tokenizer_decode = Vec::new();
    let mut main_backbone = Vec::new();
    let mut model_total = Vec::new();
    let mut prompt_tokens = Vec::new();
    let mut audio_tokens = Vec::new();
    let mut generated_tokens = Vec::new();
    let mut max_new_tokens = Vec::new();
    let mut rnnt_joint_steps = Vec::new();
    let mut token_select_reads = Vec::new();
    let mut host_argmax_reads = Vec::new();
    let mut device_argmax_reads = Vec::new();
    let mut flash_attention_requested = Vec::new();
    let mut flash_attention_compiled = Vec::new();
    let mut kv_page_size = Vec::new();
    let mut cuda_dense_decode_cache = Vec::new();
    let mut dense_head_decode_enabled = Vec::new();
    let mut qkv_projection_fused = Vec::new();
    let mut gate_up_projection_fused = Vec::new();
    let mut rope_cache_precomputed = Vec::new();
    let mut dense_decode_max_tokens = Vec::new();
    let mut gguf_qmatmul_text_enabled = Vec::new();
    let mut text_projection_quantized = Vec::new();
    let mut qmatmul_projection_count = Vec::new();
    let mut dense_projection_count = Vec::new();
    let mut dense_bias_projection_count = Vec::new();
    let mut audio_embedding_cache_hit = Vec::new();
    let mut cuda_device_argmax = Vec::new();
    let mut residual_branches_prescaled = Vec::new();
    let mut dense_decode_preallocated = Vec::new();
    let mut dense_decode_initial_capacity = Vec::new();
    let mut deferred_stop_check = Vec::new();
    let mut chunked_stop_check = Vec::new();
    let mut stop_check_interval = Vec::new();
    let mut qwen_profile_enabled = Vec::new();
    let mut qwen_qmatmul_calls = Vec::new();
    let mut qwen_qmatmul_ms = Vec::new();
    let mut qwen_qmatmul_input_casts = Vec::new();
    let mut qwen_qmatmul_input_cast_ms = Vec::new();
    let mut qwen_qmatmul_output_casts = Vec::new();
    let mut qwen_qmatmul_output_cast_ms = Vec::new();
    let mut qwen_lm_head_calls = Vec::new();
    let mut qwen_lm_head_ms = Vec::new();
    let mut qwen_silu_mul_fused_calls = Vec::new();
    let mut qwen_silu_mul_fallback_calls = Vec::new();
    let mut qwen_argmax_calls = Vec::new();
    let mut qwen_argmax_ms = Vec::new();

    for sample in samples {
        if let Some(value) = sample.audio_decode {
            audio_decode.push(value);
        }
        if let Some(value) = sample.mel_prepare {
            mel_prepare.push(value);
        }
        if let Some(value) = sample.feature_extract {
            feature_extract.push(value);
        }
        if let Some(value) = sample.feature_upload {
            feature_upload.push(value);
        }
        if let Some(value) = sample.subsample {
            subsample.push(value);
        }
        if let Some(value) = sample.encoder_forward {
            encoder_forward.push(value);
        }
        if let Some(value) = sample.encoder_ffn {
            encoder_ffn.push(value);
        }
        if let Some(value) = sample.encoder_attention {
            encoder_attention.push(value);
        }
        if let Some(value) = sample.encoder_conv {
            encoder_conv.push(value);
        }
        if let Some(value) = sample.encoder_norm {
            encoder_norm.push(value);
        }
        if let Some(value) = sample.prompt_kernel {
            prompt_kernel.push(value);
        }
        if let Some(value) = sample.language_detect {
            language_detect.push(value);
        }
        if let Some(value) = sample.resample {
            resample.push(value);
        }
        if let Some(value) = sample.mel {
            mel.push(value);
        }
        if let Some(value) = sample.mel_flatten_upload {
            mel_flatten_upload.push(value);
        }
        if let Some(value) = sample.audio_encode {
            audio_encode.push(value);
        }
        if let Some(value) = sample.prompt_embed {
            prompt_embed.push(value);
        }
        if let Some(value) = sample.prompt_concat {
            prompt_concat.push(value);
        }
        if let Some(value) = sample.prefill {
            prefill.push(value);
        }
        if let Some(value) = sample.decode {
            decode.push(value);
        }
        if let Some(value) = sample.decode_argmax {
            decode_argmax.push(value);
        }
        if let Some(value) = sample.decode_token_tensor {
            decode_token_tensor.push(value);
        }
        if let Some(value) = sample.decode_forward {
            decode_forward.push(value);
        }
        if let Some(value) = sample.tokenizer_decode {
            tokenizer_decode.push(value);
        }
        if let Some(value) = sample.main_backbone {
            main_backbone.push(value);
        }
        if let Some(value) = sample.model_total {
            model_total.push(value);
        }
        if let Some(value) = sample.prompt_tokens {
            prompt_tokens.push(value);
        }
        if let Some(value) = sample.audio_tokens {
            audio_tokens.push(value);
        }
        if let Some(value) = sample.generated_tokens {
            generated_tokens.push(value);
        }
        if let Some(value) = sample.max_new_tokens {
            max_new_tokens.push(value);
        }
        if let Some(value) = sample.rnnt_joint_steps {
            rnnt_joint_steps.push(value);
        }
        if let Some(value) = sample.token_select_reads {
            token_select_reads.push(value);
        }
        if let Some(value) = sample.host_argmax_reads {
            host_argmax_reads.push(value);
        }
        if let Some(value) = sample.device_argmax_reads {
            device_argmax_reads.push(value);
        }
        if let Some(execution) = sample.execution.as_ref() {
            if let Some(value) = execution.flash_attention_requested {
                flash_attention_requested.push(value);
            }
            if let Some(value) = execution.flash_attention_compiled {
                flash_attention_compiled.push(value);
            }
            if let Some(value) = execution.kv_page_size {
                kv_page_size.push(value);
            }
            if let Some(value) = execution.cuda_dense_decode_cache {
                cuda_dense_decode_cache.push(value);
            }
            if let Some(value) = execution.dense_head_decode_enabled {
                dense_head_decode_enabled.push(value);
            }
            if let Some(value) = execution.qkv_projection_fused {
                qkv_projection_fused.push(value);
            }
            if let Some(value) = execution.gate_up_projection_fused {
                gate_up_projection_fused.push(value);
            }
            if let Some(value) = execution.rope_cache_precomputed {
                rope_cache_precomputed.push(value);
            }
            if let Some(value) = execution.dense_decode_max_tokens {
                dense_decode_max_tokens.push(value);
            }
            if let Some(value) = execution.gguf_qmatmul_text_enabled {
                gguf_qmatmul_text_enabled.push(value);
            }
            if let Some(value) = execution.text_projection_quantized {
                text_projection_quantized.push(value);
            }
            if let Some(value) = execution.qmatmul_projection_count {
                qmatmul_projection_count.push(value);
            }
            if let Some(value) = execution.dense_projection_count {
                dense_projection_count.push(value);
            }
            if let Some(value) = execution.dense_bias_projection_count {
                dense_bias_projection_count.push(value);
            }
            if let Some(value) = execution.audio_embedding_cache_hit {
                audio_embedding_cache_hit.push(value);
            }
            if let Some(value) = execution.cuda_device_argmax {
                cuda_device_argmax.push(value);
            }
            if let Some(value) = execution.residual_branches_prescaled {
                residual_branches_prescaled.push(value);
            }
            if let Some(value) = execution.dense_decode_preallocated {
                dense_decode_preallocated.push(value);
            }
            if let Some(value) = execution.dense_decode_initial_capacity {
                dense_decode_initial_capacity.push(value);
            }
            if let Some(value) = execution.deferred_stop_check {
                deferred_stop_check.push(value);
            }
            if let Some(value) = execution.chunked_stop_check {
                chunked_stop_check.push(value);
            }
            if let Some(value) = execution.stop_check_interval {
                stop_check_interval.push(value);
            }
        }
        if let Some(profile) = sample.qwen_profile {
            if let Some(value) = profile.qwen3_profile_enabled {
                qwen_profile_enabled.push(value);
            }
            if let Some(value) = profile.qmatmul_calls {
                qwen_qmatmul_calls.push(value);
            }
            if let Some(value) = profile.qmatmul_ms {
                qwen_qmatmul_ms.push(value);
            }
            if let Some(value) = profile.qmatmul_input_casts {
                qwen_qmatmul_input_casts.push(value);
            }
            if let Some(value) = profile.qmatmul_input_cast_ms {
                qwen_qmatmul_input_cast_ms.push(value);
            }
            if let Some(value) = profile.qmatmul_output_casts {
                qwen_qmatmul_output_casts.push(value);
            }
            if let Some(value) = profile.qmatmul_output_cast_ms {
                qwen_qmatmul_output_cast_ms.push(value);
            }
            if let Some(value) = profile.lm_head_calls {
                qwen_lm_head_calls.push(value);
            }
            if let Some(value) = profile.lm_head_ms {
                qwen_lm_head_ms.push(value);
            }
            if let Some(value) = profile.silu_mul_fused_calls {
                qwen_silu_mul_fused_calls.push(value);
            }
            if let Some(value) = profile.silu_mul_fallback_calls {
                qwen_silu_mul_fallback_calls.push(value);
            }
            if let Some(value) = profile.argmax_calls {
                qwen_argmax_calls.push(value);
            }
            if let Some(value) = profile.argmax_ms {
                qwen_argmax_ms.push(value);
            }
        }
    }

    if audio_decode.is_empty()
        && mel_prepare.is_empty()
        && feature_extract.is_empty()
        && feature_upload.is_empty()
        && subsample.is_empty()
        && encoder_forward.is_empty()
        && encoder_ffn.is_empty()
        && encoder_attention.is_empty()
        && encoder_conv.is_empty()
        && encoder_norm.is_empty()
        && prompt_kernel.is_empty()
        && language_detect.is_empty()
        && resample.is_empty()
        && mel.is_empty()
        && mel_flatten_upload.is_empty()
        && audio_encode.is_empty()
        && prompt_embed.is_empty()
        && prompt_concat.is_empty()
        && prefill.is_empty()
        && decode.is_empty()
        && decode_argmax.is_empty()
        && decode_token_tensor.is_empty()
        && decode_forward.is_empty()
        && tokenizer_decode.is_empty()
        && main_backbone.is_empty()
        && model_total.is_empty()
        && prompt_tokens.is_empty()
        && audio_tokens.is_empty()
        && generated_tokens.is_empty()
        && max_new_tokens.is_empty()
        && rnnt_joint_steps.is_empty()
        && token_select_reads.is_empty()
        && host_argmax_reads.is_empty()
        && device_argmax_reads.is_empty()
        && flash_attention_requested.is_empty()
        && flash_attention_compiled.is_empty()
        && kv_page_size.is_empty()
        && cuda_dense_decode_cache.is_empty()
        && dense_head_decode_enabled.is_empty()
        && qkv_projection_fused.is_empty()
        && gate_up_projection_fused.is_empty()
        && rope_cache_precomputed.is_empty()
        && dense_decode_max_tokens.is_empty()
        && gguf_qmatmul_text_enabled.is_empty()
        && text_projection_quantized.is_empty()
        && qmatmul_projection_count.is_empty()
        && dense_projection_count.is_empty()
        && dense_bias_projection_count.is_empty()
        && audio_embedding_cache_hit.is_empty()
        && cuda_device_argmax.is_empty()
        && residual_branches_prescaled.is_empty()
        && dense_decode_preallocated.is_empty()
        && dense_decode_initial_capacity.is_empty()
        && deferred_stop_check.is_empty()
        && chunked_stop_check.is_empty()
        && stop_check_interval.is_empty()
        && qwen_profile_enabled.is_empty()
        && qwen_qmatmul_calls.is_empty()
        && qwen_qmatmul_ms.is_empty()
        && qwen_qmatmul_input_casts.is_empty()
        && qwen_qmatmul_input_cast_ms.is_empty()
        && qwen_qmatmul_output_casts.is_empty()
        && qwen_qmatmul_output_cast_ms.is_empty()
        && qwen_lm_head_calls.is_empty()
        && qwen_lm_head_ms.is_empty()
        && qwen_silu_mul_fused_calls.is_empty()
        && qwen_silu_mul_fallback_calls.is_empty()
        && qwen_argmax_calls.is_empty()
        && qwen_argmax_ms.is_empty()
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
    summarize_stage("feature_ext", &feature_extract);
    summarize_stage("feature_up", &feature_upload);
    summarize_stage("subsample", &subsample);
    summarize_stage("encoder_fwd", &encoder_forward);
    summarize_stage("encoder_ffn", &encoder_ffn);
    summarize_stage("encoder_attn", &encoder_attention);
    summarize_stage("encoder_conv", &encoder_conv);
    summarize_stage("encoder_norm", &encoder_norm);
    summarize_stage("prompt", &prompt_kernel);
    summarize_stage("lang_detect", &language_detect);
    summarize_stage("resample", &resample);
    summarize_stage("mel", &mel);
    summarize_stage("mel_flat_upload", &mel_flatten_upload);
    summarize_stage("audio_encode", &audio_encode);
    summarize_stage("prompt_embed", &prompt_embed);
    summarize_stage("prompt_concat", &prompt_concat);
    summarize_stage("prefill", &prefill);
    summarize_stage("decode", &decode);
    summarize_stage("decode_argmax", &decode_argmax);
    summarize_stage("token_tensor", &decode_token_tensor);
    summarize_stage("decode_fwd", &decode_forward);
    summarize_stage("tokenizer", &tokenizer_decode);
    summarize_stage("main_backbone", &main_backbone);
    summarize_stage("model_total", &model_total);
    summarize_count("prompt_tokens", &prompt_tokens);
    summarize_count("audio_tokens", &audio_tokens);
    summarize_count("gen_tokens", &generated_tokens);
    summarize_count("max_new_tokens", &max_new_tokens);
    summarize_count("rnnt_steps", &rnnt_joint_steps);
    summarize_count("token_select", &token_select_reads);
    summarize_count("host_argmax", &host_argmax_reads);
    summarize_count("dev_argmax", &device_argmax_reads);
    summarize_bool_count("flash_req", &flash_attention_requested);
    summarize_bool_count("flash_compiled", &flash_attention_compiled);
    summarize_count("kv_page", &kv_page_size);
    summarize_bool_count("cuda_dense", &cuda_dense_decode_cache);
    summarize_bool_count("dense_head", &dense_head_decode_enabled);
    summarize_bool_count("qkv_fused", &qkv_projection_fused);
    summarize_bool_count("gate_up_fused", &gate_up_projection_fused);
    summarize_bool_count("rope_cache", &rope_cache_precomputed);
    summarize_count("dense_max", &dense_decode_max_tokens);
    summarize_bool_count("gguf_qmatmul", &gguf_qmatmul_text_enabled);
    summarize_bool_count("qproj_quant", &text_projection_quantized);
    summarize_count("qmatmul_proj", &qmatmul_projection_count);
    summarize_count("dense_proj", &dense_projection_count);
    summarize_count("dense_bias_proj", &dense_bias_projection_count);
    summarize_bool_count("audio_cache", &audio_embedding_cache_hit);
    summarize_bool_count("cuda_argmax", &cuda_device_argmax);
    summarize_bool_count("resid_prescale", &residual_branches_prescaled);
    summarize_bool_count("dense_prealloc", &dense_decode_preallocated);
    summarize_count("dense_init", &dense_decode_initial_capacity);
    summarize_bool_count("defer_stop", &deferred_stop_check);
    summarize_bool_count("chunk_stop", &chunked_stop_check);
    summarize_count("stop_interval", &stop_check_interval);
    summarize_bool_count("qwen_profile", &qwen_profile_enabled);
    summarize_count("qmatmul_calls", &qwen_qmatmul_calls);
    summarize_stage("qmatmul", &qwen_qmatmul_ms);
    summarize_count("qmatmul_in_casts", &qwen_qmatmul_input_casts);
    summarize_stage("qmatmul_in_cast", &qwen_qmatmul_input_cast_ms);
    summarize_count("qmatmul_out_casts", &qwen_qmatmul_output_casts);
    summarize_stage("qmatmul_out_cast", &qwen_qmatmul_output_cast_ms);
    summarize_count("lm_head_calls", &qwen_lm_head_calls);
    summarize_stage("lm_head", &qwen_lm_head_ms);
    summarize_count("silu_fused", &qwen_silu_mul_fused_calls);
    summarize_count("silu_fallback", &qwen_silu_mul_fallback_calls);
    summarize_count("argmax_calls", &qwen_argmax_calls);
    summarize_stage("argmax", &qwen_argmax_ms);
}

fn print_asr_decode_profile_summary(samples: &[AsrDecodeProfileTimings]) {
    if samples.is_empty() {
        return;
    }

    let mut steps = Vec::new();
    let mut step_total_avg = Vec::new();
    let mut step_total_p95 = Vec::new();
    let mut loop_argmax = Vec::new();
    let mut loop_scalar_read = Vec::new();
    let mut loop_model_forward = Vec::new();
    let mut forward_token_embedding = Vec::new();
    let mut forward_rope_build = Vec::new();
    let mut forward_layers_total = Vec::new();
    let mut forward_final_norm = Vec::new();
    let mut forward_lm_head = Vec::new();
    let mut decoder_total = Vec::new();
    let mut attention_qkv = Vec::new();
    let mut attention_rope = Vec::new();
    let mut attention_cache = Vec::new();
    let mut attention_kernel = Vec::new();
    let mut attention_output = Vec::new();
    let mut mlp_gate_up = Vec::new();
    let mut mlp_activation = Vec::new();
    let mut mlp_down = Vec::new();
    let mut residual = Vec::new();

    for sample in samples {
        if let Some(value) = sample.steps {
            steps.push(value);
        }
        if let Some(value) = sample.step_total_avg_ms {
            step_total_avg.push(value);
        }
        if let Some(value) = sample.step_total_p95_ms {
            step_total_p95.push(value);
        }
        if let Some(value) = sample.loop_argmax_ms {
            loop_argmax.push(value);
        }
        if let Some(value) = sample.loop_scalar_read_ms {
            loop_scalar_read.push(value);
        }
        if let Some(value) = sample.loop_model_forward_ms {
            loop_model_forward.push(value);
        }
        if let Some(value) = sample.forward_token_embedding_ms {
            forward_token_embedding.push(value);
        }
        if let Some(value) = sample.forward_rope_build_ms {
            forward_rope_build.push(value);
        }
        if let Some(value) = sample.forward_layers_total_ms {
            forward_layers_total.push(value);
        }
        if let Some(value) = sample.forward_final_norm_ms {
            forward_final_norm.push(value);
        }
        if let Some(value) = sample.forward_lm_head_ms {
            forward_lm_head.push(value);
        }
        if let Some(value) = sample.decoder_total_ms {
            decoder_total.push(value);
        }
        if let Some(value) = sample.attention_qkv_ms {
            attention_qkv.push(value);
        }
        if let Some(value) = sample.attention_rope_ms {
            attention_rope.push(value);
        }
        if let Some(value) = sample.attention_cache_ms {
            attention_cache.push(value);
        }
        if let Some(value) = sample.attention_kernel_ms {
            attention_kernel.push(value);
        }
        if let Some(value) = sample.attention_output_ms {
            attention_output.push(value);
        }
        if let Some(value) = sample.mlp_gate_up_ms {
            mlp_gate_up.push(value);
        }
        if let Some(value) = sample.mlp_activation_ms {
            mlp_activation.push(value);
        }
        if let Some(value) = sample.mlp_down_ms {
            mlp_down.push(value);
        }
        if let Some(value) = sample.residual_ms {
            residual.push(value);
        }
    }

    println!(
        "\n{}",
        console::style("ASR Decode Profile (run-local):")
            .bold()
            .underlined()
    );
    summarize_count("steps", &steps);
    summarize_stage("step_avg", &step_total_avg);
    summarize_stage("step_p95", &step_total_p95);
    summarize_stage("loop_fwd", &loop_model_forward);
    summarize_stage("argmax", &loop_argmax);
    summarize_stage("scalar_read", &loop_scalar_read);
    summarize_stage("embed", &forward_token_embedding);
    summarize_stage("rope_build", &forward_rope_build);
    summarize_stage("layers_total", &forward_layers_total);
    summarize_stage("final_norm", &forward_final_norm);
    summarize_stage("lm_head", &forward_lm_head);
    summarize_stage("dec_total", &decoder_total);
    summarize_stage("attn_qkv", &attention_qkv);
    summarize_stage("attn_rope", &attention_rope);
    summarize_stage("attn_cache", &attention_cache);
    summarize_stage("attn_kernel", &attention_kernel);
    summarize_stage("attn_out", &attention_output);
    summarize_stage("mlp_gate_up", &mlp_gate_up);
    summarize_stage("mlp_act", &mlp_activation);
    summarize_stage("mlp_down", &mlp_down);
    summarize_stage("residual", &residual);
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
        println!(
            "  Decode rolling note: single-pass Whisper ASR may report near-zero engine decode phase."
        );
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
    let fused_flash_not_requested_delta = kernel_after
        .fused_attention_fallback_flash_not_requested_total
        .saturating_sub(kernel_before.fused_attention_fallback_flash_not_requested_total);
    let fused_flash_not_compiled_delta = kernel_after
        .fused_attention_fallback_flash_not_compiled_total
        .saturating_sub(kernel_before.fused_attention_fallback_flash_not_compiled_total);
    let fused_flash_mask_unsupported_delta = kernel_after
        .fused_attention_fallback_flash_mask_unsupported_total
        .saturating_sub(kernel_before.fused_attention_fallback_flash_mask_unsupported_total);
    let fused_flash_dtype_unsupported_delta = kernel_after
        .fused_attention_fallback_flash_dtype_unsupported_total
        .saturating_sub(kernel_before.fused_attention_fallback_flash_dtype_unsupported_total);
    let fused_flash_dtype_mismatch_delta = kernel_after
        .fused_attention_fallback_flash_dtype_mismatch_total
        .saturating_sub(kernel_before.fused_attention_fallback_flash_dtype_mismatch_total);
    let fused_flash_runtime_error_delta = kernel_after
        .fused_attention_fallback_flash_runtime_error_total
        .saturating_sub(kernel_before.fused_attention_fallback_flash_runtime_error_total);
    let fused_metal_runtime_error_delta = kernel_after
        .fused_attention_fallback_metal_sdpa_runtime_error_total
        .saturating_sub(kernel_before.fused_attention_fallback_metal_sdpa_runtime_error_total);
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
    let fused_unsupported_backend_delta = kernel_after
        .fused_attention_fallback_unsupported_backend_total
        .saturating_sub(kernel_before.fused_attention_fallback_unsupported_backend_total);
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
        "  CUDA FlashAttention fallback reasons (not-requested/not-compiled/mask/dtype/mismatch/runtime): {} / {} / {} / {} / {} / {}",
        fused_flash_not_requested_delta,
        fused_flash_not_compiled_delta,
        fused_flash_mask_unsupported_delta,
        fused_flash_dtype_unsupported_delta,
        fused_flash_dtype_mismatch_delta,
        fused_flash_runtime_error_delta
    );
    println!(
        "  Masked fused fallback reasons (policy/shape/dtype): {} / {} / {}",
        fused_mask_policy_disabled_delta,
        fused_mask_shape_unsupported_delta,
        fused_mask_dtype_unsupported_delta
    );
    println!(
        "  Fused backend fallback reasons (metal-runtime/unsupported): {} / {}",
        fused_metal_runtime_error_delta, fused_unsupported_backend_delta
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
    fn lfm25_tts_stage_timing_collection_reads_diagnostics_header_shape() {
        let diagnostics = serde_json::json!({
            "model": "lfm25_audio",
            "task": "tts",
            "timings_ms": {
                "prompt_build": 1.0,
                "prompt_embed": 2.0,
                "prefill": 3.0,
                "text_sampling": 4.0,
                "audio_head": 5.0,
                "audio_head_depth_linear": 5.1,
                "audio_head_depthformer": 5.2,
                "audio_head_sample": 5.3,
                "audio_head_materialize": 5.4,
                "detokenizer": 6.0,
                "detokenizer_backbone": 6.1,
                "detokenizer_readback": 6.2,
                "detokenizer_istft": 6.3,
                "model_total": 21.0
            },
            "prompt": {
                "prompt_tokens": 11
            },
            "decode": {
                "generated_tokens": 42,
                "audio_head_calls": 7,
                "audio_head_codebook_steps": 56,
                "text_sample_calls": 3
            },
            "audio": {
                "audio_frames": 6
            }
        });

        let timings = tts_stage_timings_from_diagnostics(&diagnostics)
            .expect("LFM2.5 TTS diagnostics should parse");

        assert_eq!(timings.prompt_build, Some(1.0));
        assert_eq!(timings.prefill, Some(3.0));
        assert_eq!(timings.audio_head, Some(5.0));
        assert_eq!(timings.audio_head_depth_linear, Some(5.1));
        assert_eq!(timings.audio_head_depthformer, Some(5.2));
        assert_eq!(timings.audio_head_sample, Some(5.3));
        assert_eq!(timings.audio_head_materialize, Some(5.4));
        assert_eq!(timings.detokenizer, Some(6.0));
        assert_eq!(timings.detokenizer_backbone, Some(6.1));
        assert_eq!(timings.detokenizer_readback, Some(6.2));
        assert_eq!(timings.detokenizer_istft, Some(6.3));
        assert_eq!(timings.model_total, Some(21.0));
        assert_eq!(timings.prompt_tokens, Some(11));
        assert_eq!(timings.generated_tokens, Some(42));
        assert_eq!(timings.audio_frames, Some(6));
        assert_eq!(timings.audio_head_calls, Some(7));
        assert_eq!(timings.audio_head_codebook_steps, Some(56));
        assert_eq!(timings.text_sample_calls, Some(3));
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
        let samples = collect_asr_stage_timings_from_diagnostics(&diagnostics);

        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].decode, Some(10.0));
        assert_eq!(samples[0].model_total, Some(20.0));
    }

    #[test]
    fn asr_stage_timing_collection_includes_qwen_diagnostics() {
        let diagnostics = serde_json::json!({
            "model_family": "qwen3_asr",
            "audio": {
                "audio_tokens": 355
            },
            "prompt": {
                "prompt_tokens": 482
            },
            "decode": {
                "generated_tokens": 93,
                "max_new_tokens": 512
            },
            "execution": {
                "flash_attention_requested": true,
                "flash_attention_compiled": true,
                "kv_page_size": 64,
                "dense_decode_enabled": true,
                "dense_decode_max_tokens": 882,
                "gguf_qmatmul_text_enabled": true,
                "text_projection_quantized": true,
                "qmatmul_projection_count": 197,
                "dense_projection_count": 0,
                "dense_bias_projection_count": 0
            },
            "profile": {
                "qwen3_profile_enabled": true,
                "qmatmul_calls": 128,
                "qmatmul_ms": 40.5,
                "qmatmul_input_casts": 96,
                "qmatmul_input_cast_ms": 3.5,
                "qmatmul_output_casts": 96,
                "qmatmul_output_cast_ms": 2.5,
                "lm_head_calls": 4,
                "lm_head_ms": 8.5,
                "silu_mul_fused_calls": 28,
                "silu_mul_fallback_calls": 0,
                "argmax_calls": 3,
                "argmax_ms": 1.25
            },
            "timings_ms": {
                "resample": 0.0,
                "mel": 12.5,
                "mel_flatten_upload": 0.75,
                "audio_encode": 820.0,
                "prompt_embed": 4.0,
                "prompt_concat": 5.0,
                "prefill": 1080.0,
                "decode": 21230.0,
                "decode_argmax": 77.0,
                "decode_token_tensor": 2.0,
                "decode_forward": 700.0,
                "tokenizer_decode": 9.0,
                "main_backbone": 1800.0,
                "model_total": 22360.0
            }
        });

        let samples = collect_asr_stage_timings_from_diagnostics(&diagnostics);

        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].mel, Some(12.5));
        assert_eq!(samples[0].mel_flatten_upload, Some(0.75));
        assert_eq!(samples[0].audio_encode, Some(820.0));
        assert_eq!(samples[0].prompt_embed, Some(4.0));
        assert_eq!(samples[0].prompt_concat, Some(5.0));
        assert_eq!(samples[0].prefill, Some(1080.0));
        assert_eq!(samples[0].decode, Some(21230.0));
        assert_eq!(samples[0].decode_argmax, Some(77.0));
        assert_eq!(samples[0].decode_token_tensor, Some(2.0));
        assert_eq!(samples[0].decode_forward, Some(700.0));
        assert_eq!(samples[0].tokenizer_decode, Some(9.0));
        assert_eq!(samples[0].main_backbone, Some(1800.0));
        assert_eq!(samples[0].prompt_tokens, Some(482));
        assert_eq!(samples[0].audio_tokens, Some(355));
        assert_eq!(samples[0].generated_tokens, Some(93));
        assert_eq!(samples[0].max_new_tokens, Some(512));
        let execution = samples[0].execution.as_ref().expect("execution");
        assert_eq!(execution.flash_attention_requested, Some(true));
        assert_eq!(execution.kv_page_size, Some(64));
        assert_eq!(execution.dense_head_decode_enabled, Some(true));
        assert_eq!(execution.dense_decode_max_tokens, Some(882));
        assert_eq!(execution.gguf_qmatmul_text_enabled, Some(true));
        assert_eq!(execution.qmatmul_projection_count, Some(197));
        let profile = samples[0].qwen_profile.expect("qwen profile");
        assert_eq!(profile.qwen3_profile_enabled, Some(true));
        assert_eq!(profile.qmatmul_calls, Some(128));
        assert_eq!(profile.qmatmul_ms, Some(40.5));
        assert_eq!(profile.qmatmul_input_casts, Some(96));
        assert_eq!(profile.qmatmul_output_casts, Some(96));
        assert_eq!(profile.lm_head_calls, Some(4));
        assert_eq!(profile.silu_mul_fused_calls, Some(28));
        assert_eq!(profile.argmax_calls, Some(3));
    }

    #[test]
    fn asr_stage_timing_collection_accepts_whisper_generated_token_count() {
        let diagnostics = serde_json::json!({
            "model_family": "whisper_asr",
            "decode": {
                "generated_token_count": 41,
                "max_steps": 448
            },
            "timings_ms": {
                "decode": 1234.0,
                "model_total": 1300.0
            }
        });

        let samples = collect_asr_stage_timings_from_diagnostics(&diagnostics);

        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].decode, Some(1234.0));
        assert_eq!(samples[0].generated_tokens, Some(41));
        assert_eq!(samples[0].max_new_tokens, Some(448));
    }

    #[test]
    fn asr_stage_timing_collection_includes_vibevoice_diagnostics() {
        let diagnostics = serde_json::json!({
            "model_family": "vibevoice_asr",
            "audio": {
                "acoustic_frames": 186
            },
            "prompt": {
                "tokens": 221
            },
            "decode": {
                "generated_tokens": 37,
                "max_new_tokens": 512
            },
            "execution": {
                "cuda_dense_decode_cache": true,
                "dense_head_decode_enabled": true,
                "qkv_projection_fused": true,
                "gate_up_projection_fused": true,
                "rope_cache_precomputed": true,
                "dense_decode_max_tokens": 384,
                "audio_embedding_cache_hit": true,
                "cuda_device_argmax": true,
                "residual_branches_prescaled": true,
                "dense_decode_preallocated": true,
                "dense_decode_initial_capacity": 512,
                "deferred_stop_check": true,
                "chunked_stop_check": true,
                "stop_check_interval": 8
            },
            "timings_ms": {
                "audio_encode": 420.0,
                "prefill": 530.0,
                "decode": 910.0,
                "model_total": 1860.0
            }
        });

        let samples = collect_asr_stage_timings_from_diagnostics(&diagnostics);

        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].audio_encode, Some(420.0));
        assert_eq!(samples[0].prefill, Some(530.0));
        assert_eq!(samples[0].decode, Some(910.0));
        assert_eq!(samples[0].prompt_tokens, Some(221));
        assert_eq!(samples[0].audio_tokens, Some(186));
        assert_eq!(samples[0].generated_tokens, Some(37));
        assert_eq!(samples[0].max_new_tokens, Some(512));
        let execution = samples[0]
            .execution
            .expect("execution diagnostics should be extracted");
        assert_eq!(execution.cuda_dense_decode_cache, Some(true));
        assert_eq!(execution.dense_head_decode_enabled, Some(true));
        assert_eq!(execution.qkv_projection_fused, Some(true));
        assert_eq!(execution.gate_up_projection_fused, Some(true));
        assert_eq!(execution.rope_cache_precomputed, Some(true));
        assert_eq!(execution.dense_decode_max_tokens, Some(384));
        assert_eq!(execution.audio_embedding_cache_hit, Some(true));
        assert_eq!(execution.cuda_device_argmax, Some(true));
        assert_eq!(execution.residual_branches_prescaled, Some(true));
        assert_eq!(execution.dense_decode_preallocated, Some(true));
        assert_eq!(execution.dense_decode_initial_capacity, Some(512));
        assert_eq!(execution.deferred_stop_check, Some(true));
        assert_eq!(execution.chunked_stop_check, Some(true));
        assert_eq!(execution.stop_check_interval, Some(8));
        assert_eq!(
            asr_execution_from_diagnostics(&diagnostics)
                .expect("execution diagnostics")
                .dense_decode_max_tokens,
            Some(384)
        );
    }

    #[test]
    fn asr_decode_profile_collection_includes_nested_granite_diagnostics() {
        let diagnostics = serde_json::json!({
            "model_diagnostics": {
                "model_family": "granite_speech_asr",
                "decode_profile": {
                    "enabled": true,
                    "steps": 75,
                    "step_total_ms": {
                        "avg": 76.0,
                        "p95": 80.0
                    },
                    "loop_totals_ms": {
                        "argmax": 12.0,
                        "scalar_read": 15.0,
                        "model_forward": 5500.0
                    },
                    "forward_totals_ms": {
                        "token_embedding": 3.0,
                        "rope_build": 4.0,
                        "layers_total": 5300.0,
                        "final_norm": 5.0,
                        "lm_head": 120.0
                    },
                    "decoder_totals_ms": {
                        "total": 5300.0,
                        "attention": {
                            "qkv": 900.0,
                            "rope": 180.0,
                            "cache": 75.0,
                            "kernel": 1100.0,
                            "output": 650.0
                        },
                        "mlp": {
                            "gate_up": 800.0,
                            "activation": 250.0,
                            "down": 700.0
                        },
                        "residual": 120.0
                    }
                }
            }
        });

        let samples = collect_asr_decode_profiles_from_diagnostics(&diagnostics);

        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].steps, Some(75));
        assert_eq!(samples[0].step_total_avg_ms, Some(76.0));
        assert_eq!(samples[0].loop_model_forward_ms, Some(5500.0));
        assert_eq!(samples[0].forward_lm_head_ms, Some(120.0));
        assert_eq!(samples[0].attention_kernel_ms, Some(1100.0));
        assert_eq!(samples[0].mlp_activation_ms, Some(250.0));
        assert_eq!(samples[0].residual_ms, Some(120.0));
    }

    #[test]
    fn asr_decode_profile_collection_includes_whisper_decode_profile() {
        let diagnostics = serde_json::json!({
            "model_family": "whisper_asr",
            "decode": {
                "generated_token_count": 95,
                "profile": {
                    "enabled": true,
                    "token_tensor_ms": 2.0,
                    "decoder_forward_ms": 700.0,
                    "final_linear_ms": 80.0,
                    "sampling_ms": 30.0,
                    "step_total_ms": 830.0,
                    "unattributed_ms": 18.0
                }
            }
        });

        let samples = collect_asr_decode_profiles_from_diagnostics(&diagnostics);

        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].steps, Some(95));
        assert_eq!(samples[0].loop_model_forward_ms, Some(700.0));
        assert_eq!(samples[0].forward_lm_head_ms, Some(80.0));
        assert_eq!(samples[0].loop_scalar_read_ms, Some(30.0));
        assert_eq!(samples[0].decoder_total_ms, Some(830.0));
        assert_eq!(samples[0].residual_ms, Some(18.0));
    }

    #[test]
    fn asr_stage_timing_collection_includes_nested_model_diagnostics() {
        let diagnostics = serde_json::json!({
            "timings_ms": {
                "audio_decode": 23.0
            },
            "model_diagnostics": {
                "model_family": "qwen3_asr",
                "audio": {
                    "audio_tokens": 355
                },
                "prompt": {
                    "prompt_tokens": 482
                },
                "decode": {
                    "generated_tokens": 93
                },
                "timings_ms": {
                    "audio_encode": 820.0,
                    "prefill": 1080.0,
                    "decode": 21230.0,
                    "model_total": 22360.0
                }
            }
        });

        let samples = collect_asr_stage_timings_from_diagnostics(&diagnostics);

        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].audio_decode, Some(23.0));
        assert_eq!(samples[1].audio_encode, Some(820.0));
        assert_eq!(samples[1].prefill, Some(1080.0));
        assert_eq!(samples[1].generated_tokens, Some(93));
    }

    #[test]
    fn benchmark_sample_serializes_raw_asr_diagnostics_when_present() {
        let diagnostics = serde_json::json!({
            "model_family": "granite_speech_asr",
            "execution": {
                "device_kind": "Metal",
                "dense_head_decode_enabled": true
            }
        });
        let sample = BenchmarkSample {
            index: 1,
            latency_ms: Some(123.0),
            ttft_ms: None,
            end_to_end_ms: Some(123.0),
            completion_tps: None,
            tokens_per_second: None,
            prompt_tokens: None,
            completion_tokens: None,
            server_generation_ms: None,
            server_processing_ms: Some(120.0),
            audio_duration_secs: Some(10.0),
            rtf: Some(0.012),
            tokens_generated: None,
            tts_diagnostics: None,
            asr_execution: None,
            asr_text: Some("hello granite".to_string()),
            asr_diagnostics: Some(diagnostics.clone()),
        };

        let serialized = serde_json::to_value(sample).expect("sample should serialize");

        assert_eq!(serialized["asr_text"], "hello granite");
        assert_eq!(serialized["asr_diagnostics"], diagnostics);
    }

    #[tokio::test]
    async fn tts_bench_reference_resolves_audio_and_text_file_once() {
        let dir = tempdir().expect("temp dir should be created");
        let audio = dir.path().join("reference.wav");
        let text = dir.path().join("reference.txt");
        tokio::fs::write(&audio, b"RIFF")
            .await
            .expect("write audio");
        tokio::fs::write(&text, " reference words \n")
            .await
            .expect("write text");

        let reference = resolve_tts_bench_reference(
            None,
            None,
            Some(audio.as_path()),
            None,
            Some(text.as_path()),
        )
        .await
        .expect("reference should resolve");

        assert_eq!(
            reference.reference_audio_base64.as_deref(),
            Some("UklGRg==")
        );
        let expected_path = audio.display().to_string();
        assert_eq!(
            reference.reference_audio_path.as_deref(),
            Some(expected_path.as_str())
        );
        assert_eq!(reference.reference_text.as_deref(), Some("reference words"));
        assert!(reference.speaker.is_none());
        assert!(reference.saved_voice_id.is_none());
    }

    #[tokio::test]
    async fn tts_bench_reference_rejects_mixed_saved_and_direct_reference() {
        let err = resolve_tts_bench_reference(
            None,
            Some("voice-1"),
            Some(Path::new("reference.wav")),
            Some("reference words"),
            None,
        )
        .await
        .expect_err("saved and direct references should conflict");

        assert!(err
            .to_string()
            .contains("Use either --saved-voice-id or --reference-audio/--reference-text"));
    }

    #[tokio::test]
    async fn tts_bench_reference_trims_speaker() {
        let reference = resolve_tts_bench_reference(Some(" af_bella "), None, None, None, None)
            .await
            .expect("speaker should resolve");

        assert_eq!(reference.speaker.as_deref(), Some("af_bella"));
    }

    #[test]
    fn tts_bench_request_omits_voice_without_speaker() {
        let body =
            build_tts_bench_request_body("Kokoro-82M", "hello", &TtsBenchReference::default());

        assert!(body.get("voice").is_none());
    }

    #[test]
    fn tts_bench_request_includes_speaker_when_provided() {
        let reference = TtsBenchReference {
            speaker: Some("af_bella".to_string()),
            ..TtsBenchReference::default()
        };
        let body = build_tts_bench_request_body("Kokoro-82M", "hello", &reference);

        assert_eq!(body["voice"], "af_bella");
        assert_eq!(body["model"], "Kokoro-82M");
        assert_eq!(body["input"], "hello");
    }

    #[test]
    fn kernel_path_telemetry_deserializes_flash_fallback_reasons() {
        let kernel_path: KernelPathTelemetrySnapshot = serde_json::from_value(serde_json::json!({
            "prefill_token_mode_steps_total": 0,
            "prefill_sequence_spans_total": 0,
            "prefill_sequence_tokens_total": 0,
            "decode_attention_dense_total": 0,
            "decode_attention_paged_total": 0,
            "rope_kernel_total": 0,
            "rope_manual_total": 0,
            "fused_attention_attempts_total": 12,
            "fused_attention_success_total": 7,
            "fused_attention_fallback_total": 5,
            "fused_attention_fallback_flash_not_requested_total": 1,
            "fused_attention_fallback_flash_not_compiled_total": 2,
            "fused_attention_fallback_flash_dtype_mismatch_total": 3,
            "fused_attention_fallback_unsupported_backend_total": 4
        }))
        .expect("kernel path telemetry should deserialize");

        assert_eq!(
            kernel_path.fused_attention_fallback_flash_not_requested_total,
            1
        );
        assert_eq!(
            kernel_path.fused_attention_fallback_flash_not_compiled_total,
            2
        );
        assert_eq!(
            kernel_path.fused_attention_fallback_flash_dtype_mismatch_total,
            3
        );
        assert_eq!(
            kernel_path.fused_attention_fallback_unsupported_backend_total,
            4
        );
        assert_eq!(
            kernel_path.fused_attention_fallback_flash_runtime_error_total,
            0
        );
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
    fn manifest_matrix_expands_tts_speaker_cases() {
        let manifest: BenchmarkManifest = toml::from_str(
            r#"
[[benchmarks]]
name = "kokoro"
command = "tts"
model = "Kokoro-82M"

[benchmarks.matrix]
speaker = ["af_bella", "am_adam"]
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
            vec!["kokoro[speaker=af_bella]", "kokoro[speaker=am_adam]"]
        );
        assert_eq!(cases[0].speaker.as_deref(), Some("af_bella"));
        assert_eq!(cases[1].speaker.as_deref(), Some("am_adam"));
    }

    #[test]
    fn throughput_rps_uses_measured_elapsed() {
        assert_eq!(throughput_rps(10, Duration::from_millis(2500)), 4.0);
        assert_eq!(throughput_rps(10, Duration::ZERO), 0.0);
    }
}
