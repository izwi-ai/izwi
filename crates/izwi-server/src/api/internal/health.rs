//! Health check endpoint

use axum::{extract::State, Json};
use izwi_core::backends::{CudaRuntimeDiagnostics, DTypeSelectionRequest};
use izwi_core::runtime_models::shared::attention::flash::{
    flash_attention_compiled, flash_attention_requested,
};
use serde::Serialize;

use crate::state::AppState;

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
    pub runtime: RuntimeBackendResponse,
}

#[derive(Serialize)]
pub struct RuntimeBackendResponse {
    pub requested_backend: String,
    pub requested_backend_available: bool,
    pub selected_backend: String,
    pub selection_source: String,
    pub selection_reason: String,
    pub compiled_backends: CompiledBackendsResponse,
    pub detected_device: DetectedDeviceResponse,
    pub dtype_policy: DTypePolicyResponse,
    pub fused_attention: FusedAttentionResponse,
    pub cuda_runtime: CudaRuntimeResponse,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loaded_tts_model: Option<serde_json::Value>,
}

#[derive(Serialize)]
pub struct CompiledBackendsResponse {
    pub cpu: bool,
    pub metal: bool,
    pub cuda: bool,
}

#[derive(Serialize)]
pub struct DetectedDeviceResponse {
    pub kind: String,
    pub supports_bf16: bool,
    pub supports_f16: bool,
    pub supports_int8_tensor_cores: bool,
    pub has_unified_memory: bool,
    pub recommended_batch_size: usize,
    pub available_memory_bytes: Option<usize>,
    pub cuda_compute_capability: Option<String>,
    pub cuda_device_name: Option<String>,
}

#[derive(Serialize)]
pub struct DTypePolicyResponse {
    pub selected_dtype: String,
    pub reason: String,
}

#[derive(Serialize)]
pub struct FusedAttentionResponse {
    pub cuda_flash_attention_compiled: bool,
    pub requested: bool,
}

#[derive(Serialize)]
pub struct CudaRuntimeResponse {
    pub current_binary_cuda_compiled: bool,
    pub private_runtime_active: bool,
    pub private_runtime_packaged: bool,
    pub runtime_libraries_available: bool,
    pub missing_runtime_libraries: Vec<String>,
    pub driver_available: bool,
    pub device_usable: Option<bool>,
    pub notes: Vec<String>,
}

pub async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    let context = state.runtime.backend_context();
    let capabilities = context.capabilities;
    let requested_backend_available = context.matches_preference();
    let device = context.device.clone();
    let dtype_selection = device.resolve_dtype(DTypeSelectionRequest::new(None));
    let cuda_runtime = CudaRuntimeDiagnostics::detect(&current_server_binary_name());
    let loaded_tts_model = state.runtime.loaded_tts_model_diagnostics().await;

    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        runtime: RuntimeBackendResponse {
            requested_backend: context.preference.as_str().to_string(),
            requested_backend_available,
            selected_backend: context.backend_kind.as_str().to_string(),
            selection_source: context.source.as_str().to_string(),
            selection_reason: context.reason,
            compiled_backends: CompiledBackendsResponse {
                cpu: capabilities.cpu_compiled,
                metal: capabilities.metal_compiled,
                cuda: capabilities.cuda_compiled,
            },
            detected_device: DetectedDeviceResponse {
                kind: context.backend_kind.as_str().to_string(),
                supports_bf16: device.capabilities.supports_bf16,
                supports_f16: device.capabilities.supports_f16,
                supports_int8_tensor_cores: device.capabilities.supports_int8_tensor_cores,
                has_unified_memory: device.capabilities.has_unified_memory,
                recommended_batch_size: device.capabilities.recommended_batch_size,
                available_memory_bytes: device.capabilities.available_memory_bytes,
                cuda_compute_capability: device
                    .capabilities
                    .cuda_compute_capability
                    .map(|(major, minor)| format!("{major}.{minor}")),
                cuda_device_name: device.capabilities.cuda_device_name.clone(),
            },
            dtype_policy: DTypePolicyResponse {
                selected_dtype: format!("{:?}", dtype_selection.dtype).to_ascii_lowercase(),
                reason: dtype_selection.reason.into_owned(),
            },
            fused_attention: FusedAttentionResponse {
                cuda_flash_attention_compiled: flash_attention_compiled(),
                requested: flash_attention_requested(),
            },
            cuda_runtime: CudaRuntimeResponse::from(cuda_runtime),
            loaded_tts_model,
        },
    })
}

impl From<CudaRuntimeDiagnostics> for CudaRuntimeResponse {
    fn from(value: CudaRuntimeDiagnostics) -> Self {
        Self {
            current_binary_cuda_compiled: value.current_binary_cuda_compiled,
            private_runtime_active: value.private_runtime_active,
            private_runtime_packaged: value.private_runtime_packaged,
            runtime_libraries_available: value.runtime_libraries_available,
            missing_runtime_libraries: value.missing_runtime_libraries,
            driver_available: value.driver_available,
            device_usable: value.device_usable,
            notes: value.notes,
        }
    }
}

fn current_server_binary_name() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|path| {
            path.file_name()
                .map(|name| name.to_string_lossy().to_string())
        })
        .unwrap_or_else(|| {
            if cfg!(windows) {
                "izwi-server.exe".to_string()
            } else {
                "izwi-server".to_string()
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_runtime_health_response_does_not_expose_local_paths() {
        let response = CudaRuntimeResponse {
            current_binary_cuda_compiled: false,
            private_runtime_active: false,
            private_runtime_packaged: true,
            runtime_libraries_available: true,
            missing_runtime_libraries: Vec::new(),
            driver_available: false,
            device_usable: None,
            notes: vec!["private CUDA runtime binary is packaged".to_string()],
        };

        let value = serde_json::to_value(response).expect("serialize CUDA runtime health response");

        assert!(value.get("private_runtime_path").is_none());
        assert!(value.get("search_paths").is_none());
        assert_eq!(
            value
                .get("private_runtime_packaged")
                .and_then(|entry| entry.as_bool()),
            Some(true)
        );
    }

    #[test]
    fn runtime_backend_health_serializes_loaded_tts_model_diagnostics() {
        let response = RuntimeBackendResponse {
            requested_backend: "cuda".to_string(),
            requested_backend_available: true,
            selected_backend: "cuda".to_string(),
            selection_source: "cli".to_string(),
            selection_reason: "requested".to_string(),
            compiled_backends: CompiledBackendsResponse {
                cpu: true,
                metal: false,
                cuda: true,
            },
            detected_device: DetectedDeviceResponse {
                kind: "cuda".to_string(),
                supports_bf16: true,
                supports_f16: true,
                supports_int8_tensor_cores: false,
                has_unified_memory: false,
                recommended_batch_size: 1,
                available_memory_bytes: None,
                cuda_compute_capability: Some("8.9".to_string()),
                cuda_device_name: Some("CUDA Device".to_string()),
            },
            dtype_policy: DTypePolicyResponse {
                selected_dtype: "bf16".to_string(),
                reason: "policy".to_string(),
            },
            fused_attention: FusedAttentionResponse {
                cuda_flash_attention_compiled: true,
                requested: true,
            },
            cuda_runtime: CudaRuntimeResponse {
                current_binary_cuda_compiled: true,
                private_runtime_active: false,
                private_runtime_packaged: false,
                runtime_libraries_available: true,
                missing_runtime_libraries: Vec::new(),
                driver_available: true,
                device_usable: Some(true),
                notes: Vec::new(),
            },
            loaded_tts_model: Some(serde_json::json!({
                "model_family": "vibevoice_tts",
                "device_kind": "Cuda",
                "dtype": "BF16"
            })),
        };

        let value =
            serde_json::to_value(response).expect("serialize runtime backend health response");
        assert_eq!(
            value["loaded_tts_model"]["model_family"],
            serde_json::json!("vibevoice_tts")
        );
        assert_eq!(value["loaded_tts_model"]["device_kind"], "Cuda");
        assert_eq!(value["loaded_tts_model"]["dtype"], "BF16");
    }
}
