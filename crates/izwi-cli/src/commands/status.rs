use crate::error::Result;
use crate::http;
use crate::style::Theme;
use crate::utils;
use comfy_table::{Cell, CellAlignment, Color, Table};
use console::style;
use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
struct StatusHealthResponse {
    status: Option<String>,
    version: Option<String>,
    runtime: Option<RuntimeBackendStatus>,
}

#[derive(Debug, Default, Deserialize)]
struct RuntimeBackendStatus {
    requested_backend: Option<String>,
    requested_backend_available: Option<bool>,
    selected_backend: Option<String>,
    selection_source: Option<String>,
    selection_reason: Option<String>,
    compiled_backends: Option<CompiledBackendsStatus>,
    detected_device: Option<DetectedDeviceStatus>,
    dtype_policy: Option<DTypePolicyStatus>,
    fused_attention: Option<FusedAttentionStatus>,
    cuda_runtime: Option<CudaRuntimeStatus>,
    loaded_tts_model: Option<serde_json::Value>,
}

#[derive(Debug, Default, Deserialize)]
struct CompiledBackendsStatus {
    cpu: bool,
    metal: bool,
    cuda: bool,
}

#[derive(Debug, Default, Deserialize)]
struct DetectedDeviceStatus {
    kind: Option<String>,
    supports_bf16: Option<bool>,
    supports_f16: Option<bool>,
    supports_int8_tensor_cores: Option<bool>,
    has_unified_memory: Option<bool>,
    recommended_batch_size: Option<usize>,
    available_memory_bytes: Option<usize>,
    cuda_compute_capability: Option<String>,
    cuda_device_name: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct DTypePolicyStatus {
    selected_dtype: Option<String>,
    reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct FusedAttentionStatus {
    cuda_flash_attention_compiled: Option<bool>,
    requested: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct CudaRuntimeStatus {
    current_binary_cuda_compiled: Option<bool>,
    private_runtime_active: Option<bool>,
    private_runtime_packaged: Option<bool>,
    runtime_libraries_available: Option<bool>,
    missing_runtime_libraries: Option<Vec<String>>,
    driver_available: Option<bool>,
    device_usable: Option<bool>,
}

pub async fn execute(
    detailed: bool,
    watch: Option<u64>,
    server: &str,
    theme: &Theme,
) -> Result<()> {
    if let Some(interval) = watch {
        // Watch mode
        theme.info(&format!(
            "Watching status every {} seconds (Ctrl+C to stop)...",
            interval
        ));

        loop {
            print!("\x1B[2J\x1B[1;1H"); // Clear screen
            show_status(server, detailed).await?;
            tokio::time::sleep(tokio::time::Duration::from_secs(interval)).await;
        }
    } else {
        show_status(server, detailed).await
    }
}

async fn show_status(server: &str, detailed: bool) -> Result<()> {
    let client = http::client(Some(std::time::Duration::from_secs(10)))?;

    // Health check
    let health_resp = client.get(format!("{}/v1/health", server)).send().await;

    let health_details = match health_resp {
        Ok(resp) => {
            let status = resp.status();
            let health_details = resp.json::<StatusHealthResponse>().await.ok();

            if status.is_success() {
                println!(
                    "{}  Server: {}",
                    style("●").green(),
                    style("Healthy").green()
                );
            } else {
                println!(
                    "{}  Server: {} (Status: {})",
                    style("●").red(),
                    style("Unhealthy").red(),
                    status
                );
            }

            health_details
        }
        Err(e) => {
            println!(
                "{}  Server: {} - {}",
                style("●").red(),
                style("Unreachable").red(),
                e
            );
            return Ok(());
        }
    };

    // Get models status
    let models_resp = client
        .get(format!("{}/v1/admin/models", server))
        .send()
        .await;

    if let Ok(resp) = models_resp {
        if let Ok(data) = resp.json::<serde_json::Value>().await {
            let models = data.get("models").and_then(|d| d.as_array());

            if let Some(models) = models {
                println!("\n{}", style("Models:").bold());

                let mut table = Table::new();
                if detailed {
                    table.set_header(vec!["Model", "Status", "Size", "CUDA", "Quant"]);
                } else {
                    table.set_header(vec!["Model", "Status", "Size"]);
                }

                for model in models {
                    let id = model
                        .get("variant")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown");
                    let mut status = model
                        .get("status")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let mut size = model
                        .get("size_bytes")
                        .and_then(|v| v.as_u64())
                        .map(|s| humansize::format_size(s, humansize::BINARY))
                        .unwrap_or_else(|| "-".to_string());

                    if status == "not_downloaded" {
                        if let Some(path) = utils::model_dir_if_present(id) {
                            status = "downloaded".to_string();
                            if size == "-" {
                                if let Some(bytes) = utils::directory_size_bytes(&path) {
                                    size = humansize::format_size(bytes, humansize::BINARY);
                                }
                            }
                        }
                    }

                    let status_cell = match status.as_str() {
                        "ready" | "downloaded" => Cell::new(&status).fg(Color::Green),
                        "downloading" => Cell::new(&status).fg(Color::Yellow),
                        "loading" => Cell::new(&status).fg(Color::Blue),
                        "error" => Cell::new(&status).fg(Color::Red),
                        _ => Cell::new(&status).fg(Color::DarkGrey),
                    };

                    let mut row = vec![
                        Cell::new(id).fg(Color::Cyan),
                        status_cell,
                        Cell::new(size).set_alignment(CellAlignment::Right),
                    ];
                    if detailed {
                        row.push(Cell::new(model_cuda_level(model)).fg(Color::DarkGrey));
                        row.push(Cell::new(model_cuda_quant_level(model)).fg(Color::DarkGrey));
                    }
                    table.add_row(row);
                }

                println!("{}", table);
            }
        }
    }

    if detailed {
        println!("\n{}", style("Server Info:").bold());
        println!("  Endpoint: {}", server);
        println!(
            "  Version:  {}",
            health_details
                .as_ref()
                .and_then(|health| health.version.as_deref())
                .unwrap_or(env!("CARGO_PKG_VERSION"))
        );
        println!(
            "  Platform: {}-{}",
            std::env::consts::OS,
            std::env::consts::ARCH
        );

        if let Some(runtime) = health_details.and_then(|health| health.runtime) {
            println!("\n{}", style("Runtime Backend:").bold());
            if let Some(requested) = runtime.requested_backend.as_deref() {
                let availability = runtime
                    .requested_backend_available
                    .map(|available| if available { "yes" } else { "no" })
                    .unwrap_or("unknown");
                println!("  Requested: {}", requested);
                println!("  Available: {}", availability);
            }
            if let Some(selected) = runtime.selected_backend.as_deref() {
                println!("  Selected:  {}", selected);
            }
            if let Some(source) = runtime.selection_source.as_deref() {
                println!("  Source:    {}", source);
            }
            if let Some(compiled) = runtime.compiled_backends.as_ref() {
                println!("  Compiled:  {}", format_compiled_backends(compiled));
            }
            if let Some(device) = runtime.detected_device.as_ref() {
                if let Some(kind) = device.kind.as_deref() {
                    println!("  Device:    {}", kind);
                }
                if let Some(name) = device.cuda_device_name.as_deref() {
                    println!("  CUDA name: {}", name);
                }
                if let Some(capability) = device.cuda_compute_capability.as_deref() {
                    println!("  CUDA cc:   {}", capability);
                }
                if let Some(supports_bf16) = device.supports_bf16 {
                    println!("  BF16:      {}", if supports_bf16 { "yes" } else { "no" });
                }
                if let Some(supports_f16) = device.supports_f16 {
                    println!("  F16:       {}", if supports_f16 { "yes" } else { "no" });
                }
                if let Some(supports_int8) = device.supports_int8_tensor_cores {
                    println!("  INT8 TC:   {}", if supports_int8 { "yes" } else { "no" });
                }
                if let Some(has_unified_memory) = device.has_unified_memory {
                    println!(
                        "  Unified:   {}",
                        if has_unified_memory { "yes" } else { "no" }
                    );
                }
                if let Some(batch_size) = device.recommended_batch_size {
                    println!("  Batch:     {}", batch_size);
                }
                if let Some(memory_bytes) = device.available_memory_bytes {
                    println!(
                        "  Memory:    {}",
                        humansize::format_size(memory_bytes, humansize::BINARY)
                    );
                }
            }
            if let Some(reason) = runtime.selection_reason.as_deref() {
                println!("  Reason:    {}", reason);
            }
            if let Some(dtype) = runtime.dtype_policy.as_ref() {
                if let Some(selected) = dtype.selected_dtype.as_deref() {
                    println!("  DType:     {}", selected);
                }
                if let Some(reason) = dtype.reason.as_deref() {
                    println!("  DType why: {}", reason);
                }
            }
            if let Some(model) = runtime.loaded_tts_model.as_ref() {
                println!("  TTS model: {}", format_loaded_tts_model(model));
            }
            if let Some(fused) = runtime.fused_attention.as_ref() {
                println!(
                    "  FlashAttn: compiled={} requested={}",
                    fused
                        .cuda_flash_attention_compiled
                        .map(yes_no)
                        .unwrap_or("unknown"),
                    fused.requested.map(yes_no).unwrap_or("unknown")
                );
            }
            if let Some(cuda_runtime) = runtime.cuda_runtime.as_ref() {
                println!("\n{}", style("CUDA Runtime:").bold());
                print_cuda_runtime_status(cuda_runtime);
            }
        }
    }

    Ok(())
}

fn print_cuda_runtime_status(runtime: &CudaRuntimeStatus) {
    if let Some(compiled) = runtime.current_binary_cuda_compiled {
        println!(
            "  Current binary: {}",
            if compiled {
                "cuda-compiled"
            } else {
                "cpu-safe"
            }
        );
    }
    if let Some(active) = runtime.private_runtime_active {
        println!("  Private active: {}", yes_no(active));
    }
    if let Some(packaged) = runtime.private_runtime_packaged {
        println!("  Private packaged: {}", yes_no(packaged));
    }
    if let Some(available) = runtime.runtime_libraries_available {
        println!("  Runtime libs: {}", yes_no(available));
    }
    if let Some(driver) = runtime.driver_available {
        println!("  NVIDIA driver: {}", yes_no(driver));
    }
    if let Some(device) = runtime.device_usable {
        println!("  CUDA device: {}", yes_no(device));
    } else {
        println!("  CUDA device: unknown");
    }
    if let Some(missing) = runtime.missing_runtime_libraries.as_ref() {
        if !missing.is_empty() {
            println!("  Missing libs: {}", missing.join(", "));
        }
    }
}

fn model_cuda_level(model: &serde_json::Value) -> &str {
    model
        .get("cuda_support")
        .and_then(|value| value.get("level"))
        .and_then(|value| value.as_str())
        .unwrap_or("unknown")
}

fn model_cuda_quant_level(model: &serde_json::Value) -> &str {
    model
        .get("cuda_quantization")
        .and_then(|value| value.get("level"))
        .and_then(|value| value.as_str())
        .unwrap_or("unknown")
}

fn format_loaded_tts_model(model: &serde_json::Value) -> String {
    let family = model
        .get("model_family")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown");
    let device = model
        .get("device_kind")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown");
    let dtype = model
        .get("dtype")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown");
    format!("{family} on {device} ({dtype})")
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "yes"
    } else {
        "no"
    }
}

fn format_compiled_backends(backends: &CompiledBackendsStatus) -> String {
    let mut labels = Vec::new();
    if backends.cpu {
        labels.push("cpu");
    }
    if backends.metal {
        labels.push("metal");
    }
    if backends.cuda {
        labels.push("cuda");
    }

    if labels.is_empty() {
        "none".to_string()
    } else {
        labels.join(", ")
    }
}
