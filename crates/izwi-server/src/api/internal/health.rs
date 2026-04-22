//! Health check endpoint

use axum::{extract::State, Json};
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

pub async fn health_check(State(state): State<AppState>) -> Json<HealthResponse> {
    let context = state.runtime.backend_context();
    let capabilities = context.capabilities;
    let requested_backend_available = context.matches_preference();
    let device = context.device.clone();

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
        },
    })
}
