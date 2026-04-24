use crate::error::{CliError, Result};
use crate::http;
use crate::style::Theme;
use crate::BenchCommands;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use futures::{stream, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeTelemetryContext {
    Default,
    AsrWhisper,
}

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Default, Deserialize)]
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

#[derive(Debug, Clone, Deserialize, Default)]
struct AsrBenchResponse {
    #[serde(default)]
    izwi_asr_diagnostics: Option<serde_json::Value>,
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

pub async fn execute(command: BenchCommands, server: &str, theme: &Theme) -> Result<()> {
    match command {
        BenchCommands::Chat {
            model,
            iterations,
            prompt,
            system,
            max_tokens,
            concurrent,
            warmup,
        } => {
            bench_chat(
                server,
                &model,
                iterations,
                &prompt,
                system.as_deref(),
                max_tokens,
                concurrent,
                warmup,
                theme,
            )
            .await
        }
        BenchCommands::Tts {
            model,
            iterations,
            text,
            warmup,
        } => bench_tts(server, &model, iterations, &text, warmup, theme).await,
        BenchCommands::Asr {
            model,
            iterations,
            file,
            language,
            warmup,
        } => bench_asr(
            server,
            &model,
            iterations,
            file,
            language.as_deref(),
            warmup,
            theme,
        )
        .await,
        BenchCommands::Throughput {
            duration,
            concurrent,
        } => bench_throughput(server, duration, concurrent, theme).await,
    }
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
    theme: &Theme,
) -> Result<()> {
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

    theme.step(1, 3, &format!("Benchmarking chat with '{}'", model));
    let metrics_before = fetch_runtime_metrics(server).await;

    if warmup {
        theme.info("Running warmup iteration...");
        let _ = run_chat_request(server, model, prompt, system, max_tokens).await?;
    }

    let pb = ProgressBar::new(iterations as u64);
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
    print_runtime_delta(
        metrics_before,
        fetch_runtime_metrics(server).await,
        RuntimeTelemetryContext::Default,
    );

    Ok(())
}

async fn bench_tts(
    server: &str,
    model: &str,
    iterations: u32,
    text: &str,
    warmup: bool,
    theme: &Theme,
) -> Result<()> {
    if iterations == 0 {
        return Err(CliError::InvalidInput(
            "Iterations must be greater than 0".to_string(),
        ));
    }

    theme.step(1, 3, &format!("Benchmarking TTS with '{}'", model));
    let metrics_before = fetch_runtime_metrics(server).await;

    if warmup {
        theme.info("Running warmup iteration...");
        let _ = run_tts_request(server, model, text).await?;
    }

    let pb = ProgressBar::new(iterations as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} iterations",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut times = Vec::new();

    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let _ = run_tts_request(server, model, text).await?;
        let elapsed = start.elapsed().as_millis() as f64;
        times.push(elapsed);
        pb.inc(1);
    }

    pb.finish_with_message("Benchmark complete");

    // Calculate statistics
    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(0.0, f64::max);
    let p50 = percentile(&times, 0.5);
    let p95 = percentile(&times, 0.95);
    let p99 = percentile(&times, 0.99);

    println!("\n{}", console::style("Results:").bold().underlined());
    println!("  Iterations: {}", iterations);
    println!("  Average:    {:.2} ms", avg);
    println!("  Min:        {:.2} ms", min);
    println!("  Max:        {:.2} ms", max);
    println!("  P50:        {:.2} ms", p50);
    println!("  P95:        {:.2} ms", p95);
    println!("  P99:        {:.2} ms", p99);
    println!("  Throughput: {:.2} req/s", 1000.0 / avg);
    print_runtime_delta(
        metrics_before,
        fetch_runtime_metrics(server).await,
        RuntimeTelemetryContext::Default,
    );

    Ok(())
}

async fn bench_asr(
    server: &str,
    model: &str,
    iterations: u32,
    file: Option<std::path::PathBuf>,
    language: Option<&str>,
    warmup: bool,
    theme: &Theme,
) -> Result<()> {
    if iterations == 0 {
        return Err(CliError::InvalidInput(
            "Iterations must be greater than 0".to_string(),
        ));
    }

    theme.step(1, 3, &format!("Benchmarking ASR with '{}'", model));
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
        theme.info(&format!("Using language hint: {}", language));
    }

    if warmup {
        theme.info("Running warmup iteration...");
        let _ = run_asr_request(server, model, &audio_base64, language).await?;
    }

    let pb = ProgressBar::new(iterations as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} iterations",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut times = Vec::new();
    let mut stage_samples = Vec::new();
    let mut saw_whisper_diagnostics = false;

    for _ in 0..iterations {
        let start = std::time::Instant::now();
        let response = run_asr_request(server, model, &audio_base64, language).await?;
        if let Some(diagnostics) = response.izwi_asr_diagnostics.as_ref() {
            if diagnostics
                .get("model_family")
                .and_then(|value| value.as_str())
                == Some("whisper_asr")
            {
                saw_whisper_diagnostics = true;
            }
            if let Some(sample) = whisper_stage_timings_from_diagnostics(diagnostics) {
                stage_samples.push(sample);
            }
        }
        let elapsed = start.elapsed().as_millis() as f64;
        times.push(elapsed);
        pb.inc(1);
    }

    pb.finish_with_message("Benchmark complete");

    let avg = times.iter().sum::<f64>() / times.len() as f64;
    let min = times.iter().copied().fold(f64::INFINITY, f64::min);
    let max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let p50 = percentile(&times, 0.5);
    let p95 = percentile(&times, 0.95);
    let p99 = percentile(&times, 0.99);
    println!("\n{}", console::style("Results:").bold().underlined());
    println!("  Iterations: {}", iterations);
    println!("  Average:    {:.2} ms", avg);
    println!("  Min:        {:.2} ms", min);
    println!("  Max:        {:.2} ms", max);
    println!("  P50:        {:.2} ms", p50);
    println!("  P95:        {:.2} ms", p95);
    println!("  P99:        {:.2} ms", p99);
    println!("  Throughput: {:.2} req/s", 1000.0 / avg);
    print_whisper_stage_timing_summary(&stage_samples);

    let model_lower = model.to_ascii_lowercase();
    let telemetry_context = if saw_whisper_diagnostics || model_lower.contains("whisper") {
        RuntimeTelemetryContext::AsrWhisper
    } else {
        RuntimeTelemetryContext::Default
    };
    print_runtime_delta(
        metrics_before,
        fetch_runtime_metrics(server).await,
        telemetry_context,
    );

    Ok(())
}

async fn bench_throughput(
    server: &str,
    duration: u64,
    concurrent: u32,
    theme: &Theme,
) -> Result<()> {
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

    theme.step(
        1,
        1,
        &format!("Throughput test: {}s, {} concurrent", duration, concurrent),
    );

    println!("Running throughput benchmark against /livez...");
    let client = http::client(Some(std::time::Duration::from_secs(5)))?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(duration);

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
    let rps = total as f64 / duration as f64;
    println!("\n{}", console::style("Results:").bold().underlined());
    println!("  Successful: {:.0}", success);
    println!("  Failed:     {:.0}", failed);
    println!("  Total:      {:.0}", total);
    println!("  Throughput: {:.2} req/s", rps);

    Ok(())
}

async fn run_tts_request(server: &str, model: &str, text: &str) -> Result<()> {
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

    Ok(())
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

async fn fetch_runtime_metrics(server: &str) -> Option<RuntimeTelemetrySnapshot> {
    let client = http::client(Some(std::time::Duration::from_secs(3))).ok()?;
    let base = server.trim_end_matches('/');
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
    None
}

fn whisper_stage_timings_from_diagnostics(diagnostics: &serde_json::Value) -> Option<WhisperStageTimings> {
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
        console::style("Whisper Stage Timings (run-local):")
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
    let (Some(before), Some(after)) = (before, after) else {
        println!(
            "\nRuntime telemetry delta: unavailable (/internal/metrics or /v1/metrics not reachable)"
        );
        return;
    };

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
    println!("  Queued:             {}", queued_delta);
    println!("  Completed:          {}", completed_delta);
    println!("  Failed:             {}", failed_delta);
    println!("  Active (current):   {}", after.requests_active);
    println!("  Worker restarts:    {}", restart_delta);
    println!("  Worker panics:      {}", panic_delta);
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
        .saturating_sub(kernel_before.fused_attention_fallback_metal_sdpa_mask_policy_disabled_total);
    let fused_mask_shape_unsupported_delta = kernel_after
        .fused_attention_fallback_metal_sdpa_mask_shape_unsupported_total
        .saturating_sub(kernel_before.fused_attention_fallback_metal_sdpa_mask_shape_unsupported_total);
    let fused_mask_dtype_unsupported_delta = kernel_after
        .fused_attention_fallback_metal_sdpa_mask_dtype_unsupported_total
        .saturating_sub(kernel_before.fused_attention_fallback_metal_sdpa_mask_dtype_unsupported_total);
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
}
