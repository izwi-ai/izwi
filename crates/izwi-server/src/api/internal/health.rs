//! Health check endpoint

use axum::{extract::State, Json};
use izwi_core::backends::CudaRuntimeDiagnostics;
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
    pub cuda_runtime: CudaRuntimeResponse,
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
    pub has_unified_memory: bool,
    pub recommended_batch_size: usize,
    pub available_memory_bytes: Option<usize>,
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
    let cuda_runtime = CudaRuntimeDiagnostics::detect(&current_server_binary_name());

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
                has_unified_memory: device.capabilities.has_unified_memory,
                recommended_batch_size: device.capabilities.recommended_batch_size,
                available_memory_bytes: device.capabilities.available_memory_bytes,
            },
            cuda_runtime: CudaRuntimeResponse::from(cuda_runtime),
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
}
