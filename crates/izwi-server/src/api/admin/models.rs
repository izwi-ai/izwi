//! Model management API endpoints

use axum::{
    extract::{Path, Query, State},
    response::Sse,
    Json,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::convert::Infallible;
use std::pin::Pin;
use std::time::{Duration, Instant};
use tokio::sync::broadcast::error::RecvError;
use tracing::{info, warn};
use utoipa::ToSchema;

use crate::error::{core_api_error, ApiError, ErrorClass};
use crate::state::AppState;
use izwi_core::model::download::{download_disk_required_bytes, DownloadState};
use izwi_core::{
    parse_model_variant, removed_model_variant, ChatModelCapabilities, ChatReasoningEffort,
    ModelInfo, ModelStatus, ModelVariant, SpeechModelCapabilities,
};

/// Response for model list
#[derive(Serialize, ToSchema)]
pub struct AdminModelsResponse {
    pub models: Vec<AdminModelInfo>,
    /// Downloads of variants whose support was removed, still on disk. They
    /// are never deleted automatically; `DELETE /v1/admin/models/{variant}`
    /// with the listed `variant` reclaims the space.
    #[serde(default)]
    pub removed_models: Vec<AdminRemovedModelInfo>,
    /// Residency snapshot for the `ollama ps`-style observability surface.
    #[serde(default)]
    pub residency: AdminResidencySummary,
}

/// A leftover download of a removed variant.
#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct AdminRemovedModelInfo {
    pub variant: String,
    pub display_name: String,
    /// Always `removed`.
    pub status: String,
    pub local_path: String,
    pub size_bytes: u64,
    /// The supported variant to use instead.
    pub replacement: String,
    pub reason: String,
}

impl From<izwi_core::model::RemovedModelDownload> for AdminRemovedModelInfo {
    fn from(download: izwi_core::model::RemovedModelDownload) -> Self {
        Self {
            variant: download.removed.id.to_string(),
            display_name: download.removed.display_name.to_string(),
            status: "removed".to_string(),
            local_path: download.local_path.to_string_lossy().into_owned(),
            size_bytes: download.size_bytes,
            replacement: download.removed.replacement.dir_name().to_string(),
            reason: download.removed.reason.to_string(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, ToSchema)]
pub struct AdminResidencySummary {
    /// Models currently tracked as resident (loading or ready).
    pub resident_count: usize,
    /// Residency budget (`None` = unbounded).
    pub max_loaded_models: Option<usize>,
    /// Idle keep-alive for transient residents in seconds (0 = reaper off).
    pub model_keep_alive_secs: u64,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct AdminModelInfo {
    pub variant: String,
    /// Human-readable model name from the catalog.
    pub display_name: String,
    /// Primary task: tts, asr, diarization, chat, forced_align, audio_chat
    /// or tokenizer.
    pub task: String,
    /// Expected download size from the catalog, for models not yet on disk.
    pub estimated_size_bytes: u64,
    /// Catalog estimate of memory needed to run the model, in GB.
    pub memory_required_gb: f32,
    /// Memory the model needs to load, in bytes: the catalog's estimate,
    /// which is all there is before the model is downloaded.
    pub estimated_memory_bytes: u64,
    /// Whether the model can load on this device at all: `fits`,
    /// `too_large` (it needs more memory than loads here are planned
    /// against) or `unknown`. Other loaded models may still have to make
    /// room for one that fits.
    pub device_fit: String,
    /// Free disk a download of the model needs: its files, plus the
    /// checkpoint a `.nemo` archive unpacks on first load. 0 once
    /// downloaded.
    pub disk_required_bytes: u64,
    pub enabled: bool,
    pub license: Option<String>,
    pub status: String,
    /// Explicitly loaded in this session: survives budget and memory-pressure
    /// eviction until explicitly unloaded.
    #[serde(default)]
    pub pinned: bool,
    pub local_path: Option<String>,
    pub size_bytes: Option<u64>,
    pub download_progress: Option<f32>,
    pub error_message: Option<String>,
    pub deployment_state: String,
    pub modalities: Vec<String>,
    pub route_capabilities: AdminModelRouteCapabilities,
    pub batch_capabilities: AdminModelBatchCapabilities,
    pub chat_capabilities: Option<AdminChatModelCapabilities>,
    pub speech_capabilities: Option<AdminSpeechModelCapabilities>,
    pub cuda_support: serde_json::Value,
    pub cuda_quantization: serde_json::Value,
    pub cuda_operators: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime_diagnostics: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct AdminSpeechModelCapabilities {
    pub supports_builtin_voices: bool,
    pub built_in_voice_count: Option<usize>,
    pub supports_reference_voice: bool,
    pub supports_voice_description: bool,
    pub supports_streaming: bool,
    /// What a streaming request delivers on this server (TPIPE-3):
    /// `chunked` (audio as the model commits it) or `unsupported`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub streaming_mode: Option<&'static str>,
    pub supports_speed_control: bool,
    pub supports_auto_long_form: bool,
    /// Model-specific controls a speech request may set, each with its
    /// range and the model's default (Chatterbox: `exaggeration`,
    /// `cfg_weight`, `min_p`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub controls: Vec<AdminSpeechControlRange>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct AdminSpeechControlRange {
    pub control: &'static str,
    pub min: f32,
    pub max: f32,
    pub default: f32,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct AdminChatModelCapabilities {
    pub supports_thinking: bool,
    /// False when the model always reasons: a thinking switch would do nothing.
    pub thinking_toggleable: bool,
    pub default_thinking_enabled: bool,
    pub reasoning_efforts: Vec<String>,
    pub default_reasoning_effort: Option<String>,
    pub supports_preserve_thinking: bool,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct AdminModelRouteCapabilities {
    pub openai_chat_completions: bool,
    pub openai_responses: bool,
    pub openai_audio_speech: bool,
    pub openai_audio_transcriptions: bool,
    pub speech_to_text_jobs: bool,
    pub speech_to_text_realtime: bool,
    /// Transcription requests may ask for a translation (`target_language`).
    pub speech_translation: bool,
    /// Transcription requests may bias recognition with `keywords`.
    pub keyword_biasing: bool,
    /// Speaker-attributed ASR jobs: the model writes speaker turns itself.
    pub speaker_attributed_asr: bool,
    pub diarization_records: bool,
    pub text_to_speech_records: bool,
    pub voice_design_records: bool,
    pub voice_clone_records: bool,
    pub saved_voice_reuse: bool,
    pub studio_projects: bool,
    pub voice_realtime_text_model: bool,
    pub voice_realtime_modular_asr: bool,
    pub voice_realtime_modular_tts: bool,
    pub voice_realtime_unified: bool,
    pub forced_alignment: bool,
    pub tokenizer: bool,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct AdminModelBatchCapabilities {
    pub batch_asr: bool,
    pub batch_tts: bool,
    pub durable_jobs: bool,
    pub artifacts: bool,
    pub cancellation: bool,
    pub retry: bool,
    pub requires_voice_permission: bool,
}

#[derive(Serialize, ToSchema)]
pub struct AdminModelActionResponse {
    pub status: &'static str,
    pub message: String,
    /// Why the action may be slow, when the runtime noted something (a Metal
    /// load larger than the host memory macOS reports available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
}

#[derive(Serialize, Clone, ToSchema)]
pub struct AdminModelDownloadProgressEvent {
    pub variant: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub current_file: String,
    pub current_file_downloaded: u64,
    pub current_file_total: u64,
    pub files_completed: usize,
    pub files_total: usize,
    pub percent: f32,
    pub status: String,
    /// Why the download failed, on an `error` event.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl From<ModelInfo> for AdminModelInfo {
    fn from(info: ModelInfo) -> Self {
        Self::from_model_info(info, None, false)
    }
}

impl AdminModelInfo {
    fn from_model_info(
        info: ModelInfo,
        runtime_diagnostics: Option<serde_json::Value>,
        pinned: bool,
    ) -> Self {
        let variant = info.variant;
        Self {
            variant: variant.dir_name().to_string(),
            display_name: variant.display_name().to_string(),
            task: variant.primary_task().as_str().to_string(),
            estimated_size_bytes: variant.estimated_size(),
            memory_required_gb: variant.memory_required_gb(),
            estimated_memory_bytes: 0,
            device_fit: DEVICE_FIT_UNKNOWN.to_string(),
            disk_required_bytes: match info.status {
                ModelStatus::NotDownloaded | ModelStatus::Downloading | ModelStatus::Error => {
                    download_disk_required_bytes(
                        variant,
                        info.size_bytes.unwrap_or_else(|| variant.estimated_size()),
                    )
                }
                ModelStatus::Downloaded | ModelStatus::Loading | ModelStatus::Ready => 0,
            },
            enabled: info.enabled,
            license: variant.license_label().map(str::to_string),
            status: model_status_as_str(info.status).to_string(),
            pinned,
            local_path: info
                .local_path
                .map(|path| path.to_string_lossy().into_owned()),
            size_bytes: info.size_bytes,
            download_progress: info.download_progress,
            error_message: info.error_message,
            deployment_state: model_deployment_state_as_str(info.status).to_string(),
            modalities: model_modalities(variant),
            route_capabilities: AdminModelRouteCapabilities::from_variant(variant),
            batch_capabilities: AdminModelBatchCapabilities::from_variant(variant),
            chat_capabilities: info.chat_capabilities.map(AdminChatModelCapabilities::from),
            speech_capabilities: info.speech_capabilities.map(|capabilities| {
                AdminSpeechModelCapabilities::from(capabilities).with_controls(variant)
            }),
            cuda_support: serde_json::to_value(info.cuda_support)
                .unwrap_or(serde_json::Value::Null),
            cuda_quantization: serde_json::to_value(info.cuda_quantization)
                .unwrap_or(serde_json::Value::Null),
            cuda_operators: serde_json::to_value(variant.cuda_operator_capabilities())
                .unwrap_or(serde_json::Value::Null),
            runtime_diagnostics,
        }
    }
}

const DEVICE_FIT_UNKNOWN: &str = "unknown";

impl AdminModelInfo {
    /// The memory the model needs against what loads on this device are
    /// planned against.
    fn with_device_fit(
        mut self,
        runtime: &izwi_core::RuntimeService,
        variant: ModelVariant,
    ) -> Self {
        let needed = runtime.estimated_model_load_bytes(variant);
        self.estimated_memory_bytes = needed;
        self.device_fit = match runtime.model_capacity_bytes() {
            Some(capacity) if needed > capacity => "too_large",
            Some(_) => "fits",
            None => DEVICE_FIT_UNKNOWN,
        }
        .to_string();
        self
    }
}

impl From<ChatModelCapabilities> for AdminChatModelCapabilities {
    fn from(capabilities: ChatModelCapabilities) -> Self {
        Self {
            supports_thinking: capabilities.supports_thinking,
            thinking_toggleable: capabilities.thinking_toggleable,
            default_thinking_enabled: capabilities.default_thinking_enabled,
            reasoning_efforts: capabilities
                .reasoning_efforts
                .into_iter()
                .map(reasoning_effort_as_str)
                .map(str::to_string)
                .collect(),
            default_reasoning_effort: capabilities
                .default_reasoning_effort
                .map(reasoning_effort_as_str)
                .map(str::to_string),
            supports_preserve_thinking: capabilities.supports_preserve_thinking,
        }
    }
}

const fn reasoning_effort_as_str(effort: ChatReasoningEffort) -> &'static str {
    match effort {
        ChatReasoningEffort::Xhigh => "xhigh",
        ChatReasoningEffort::Medium => "medium",
        ChatReasoningEffort::Low => "low",
    }
}

impl From<SpeechModelCapabilities> for AdminSpeechModelCapabilities {
    fn from(capabilities: SpeechModelCapabilities) -> Self {
        Self {
            supports_builtin_voices: capabilities.supports_builtin_voices,
            built_in_voice_count: capabilities.built_in_voice_count,
            supports_reference_voice: capabilities.supports_reference_voice,
            supports_voice_description: capabilities.supports_voice_description,
            supports_streaming: capabilities.supports_streaming,
            streaming_mode: None,
            supports_speed_control: capabilities.supports_speed_control,
            supports_auto_long_form: capabilities.supports_auto_long_form,
            controls: Vec::new(),
        }
    }
}

impl AdminSpeechModelCapabilities {
    fn with_controls(mut self, variant: ModelVariant) -> Self {
        self.controls = variant
            .speech_controls()
            .iter()
            .map(|range| AdminSpeechControlRange {
                control: range.control.as_str(),
                min: range.min,
                max: range.max,
                default: range.default,
            })
            .collect();
        self
    }
}

impl AdminModelRouteCapabilities {
    fn from_variant(variant: ModelVariant) -> Self {
        let speech_capabilities = variant.speech_capabilities();
        let supports_tts = speech_capabilities.is_some();
        let supports_reference_voice = speech_capabilities
            .as_ref()
            .is_some_and(|capabilities| capabilities.supports_reference_voice);
        let supports_voice_description = speech_capabilities
            .as_ref()
            .is_some_and(|capabilities| capabilities.supports_voice_description);

        Self {
            openai_chat_completions: variant.is_chat(),
            openai_responses: variant.is_chat(),
            openai_audio_speech: supports_tts,
            openai_audio_transcriptions: variant.is_asr()
                || variant.is_voxtral()
                || variant.is_audio_chat(),
            speech_to_text_jobs: variant.is_asr()
                || variant.is_voxtral()
                || variant.supports_diarization_records(),
            speech_to_text_realtime: variant.is_asr() || variant.is_audio_chat(),
            speech_translation: variant.supports_asr_translation(),
            keyword_biasing: variant.supports_asr_keyword_biasing(),
            speaker_attributed_asr: variant.supports_speaker_attributed_asr(),
            diarization_records: variant.supports_diarization_records(),
            text_to_speech_records: supports_tts,
            voice_design_records: supports_voice_description,
            voice_clone_records: supports_reference_voice,
            saved_voice_reuse: supports_reference_voice,
            studio_projects: supports_tts,
            voice_realtime_text_model: variant.is_chat(),
            voice_realtime_modular_asr: variant.is_asr(),
            voice_realtime_modular_tts: crate::app::voice_realtime::validate_realtime_tts_model(
                variant,
            )
            .is_ok(),
            voice_realtime_unified: variant.is_audio_chat(),
            forced_alignment: variant.is_forced_aligner(),
            tokenizer: variant.is_tokenizer(),
        }
    }
}

impl AdminModelBatchCapabilities {
    fn from_variant(variant: ModelVariant) -> Self {
        let speech_capabilities = variant.speech_capabilities();
        let supports_reference_voice = speech_capabilities
            .as_ref()
            .is_some_and(|capabilities| capabilities.supports_reference_voice);
        let supports_voice_description = speech_capabilities
            .as_ref()
            .is_some_and(|capabilities| capabilities.supports_voice_description);
        let batch_asr = variant.is_asr() || variant.is_voxtral() || variant.is_audio_chat();
        let batch_tts = speech_capabilities.is_some();
        let durable_jobs = batch_asr || batch_tts;

        Self {
            batch_asr,
            batch_tts,
            durable_jobs,
            artifacts: durable_jobs,
            cancellation: durable_jobs,
            retry: durable_jobs,
            requires_voice_permission: supports_reference_voice || supports_voice_description,
        }
    }
}

fn model_status_as_str(status: ModelStatus) -> &'static str {
    match status {
        ModelStatus::NotDownloaded => "not_downloaded",
        ModelStatus::Downloading => "downloading",
        ModelStatus::Downloaded => "downloaded",
        ModelStatus::Loading => "loading",
        ModelStatus::Ready => "ready",
        ModelStatus::Error => "error",
    }
}

fn model_deployment_state_as_str(status: ModelStatus) -> &'static str {
    match status {
        ModelStatus::NotDownloaded => "not_deployed",
        ModelStatus::Downloading => "installing",
        ModelStatus::Downloaded => "available",
        ModelStatus::Loading => "starting",
        ModelStatus::Ready => "ready",
        ModelStatus::Error => "failed",
    }
}

fn model_modalities(variant: ModelVariant) -> Vec<String> {
    let mut modalities = Vec::new();
    let mut push = |modality: &'static str| {
        if !modalities.iter().any(|existing| existing == modality) {
            modalities.push(modality.to_string());
        }
    };

    if variant.is_chat() {
        push("text_input");
        push("text_output");
    }
    if crate::app::chat_content::variant_accepts_image_input(variant) {
        push("image_input");
    }
    if variant.speech_capabilities().is_some() {
        push("text_input");
        push("audio_output");
    }
    if variant.is_asr() || variant.is_voxtral() || variant.is_audio_chat() {
        push("audio_input");
        push("text_output");
    }
    if variant.is_audio_chat() {
        push("text_input");
        push("audio_output");
    }
    if variant.supports_diarization_records() {
        push("audio_input");
        push("speaker_labels");
        push("timestamps");
    }
    if variant.supports_speaker_attributed_asr() {
        push("speaker_labels");
    }
    if variant.is_forced_aligner() {
        push("audio_input");
        push("text_input");
        push("timestamps");
    }
    if variant.is_tokenizer() {
        push("audio_tokens");
    }

    modalities
}

/// List all available models
pub async fn list_models(
    State(state): State<AppState>,
) -> Result<Json<AdminModelsResponse>, ApiError> {
    let mut models: Vec<ModelInfo> = state
        .runtime
        .list_models()
        .await
        .into_iter()
        .filter(|model| model.enabled)
        .collect();
    models.sort_by_key(model_sort_key);
    let runtime_diagnostics = loaded_model_diagnostics_by_variant(&state).await;
    let pinned_variants = state.runtime.pinned_model_variants().await;
    let residency = AdminResidencySummary {
        resident_count: state.runtime.resident_model_variants().await.len(),
        max_loaded_models: state.runtime.config().max_loaded_models,
        model_keep_alive_secs: state.runtime.config().model_keep_alive_secs,
    };
    let removed_models = state
        .runtime
        .model_manager()
        .removed_model_downloads()
        .await
        .into_iter()
        .map(AdminRemovedModelInfo::from)
        .collect();
    Ok(Json(AdminModelsResponse {
        removed_models,
        residency,
        models: models
            .into_iter()
            .map(|info| {
                let diagnostics = runtime_diagnostics.get(info.variant.dir_name()).cloned();
                let pinned = pinned_variants.contains(&info.variant);
                let variant = info.variant;
                let mut model = AdminModelInfo::from_model_info(info, diagnostics, pinned)
                    .with_device_fit(&state.runtime, variant);
                if let Some(speech) = model.speech_capabilities.as_mut() {
                    speech.streaming_mode =
                        Some(state.runtime.tts_streaming_mode(variant).as_str());
                }
                model
            })
            .collect(),
    }))
}

/// Get info for a specific model
pub async fn get_model_info(
    State(state): State<AppState>,
    Path(variant): Path<String>,
) -> Result<Json<AdminModelInfo>, ApiError> {
    let variant = parse_variant(&variant)?;
    if !variant.is_enabled() {
        return Err(ApiError::not_found("Model not found"));
    }

    let info = state
        .runtime
        .model_manager()
        .get_model_info(variant)
        .await
        .ok_or_else(|| ApiError::not_found("Model not found"))?;

    let runtime_diagnostics = loaded_model_diagnostics_by_variant(&state).await;
    let pinned = state.runtime.pinned_model_variants().await.contains(&variant);
    let mut model = AdminModelInfo::from_model_info(
        info,
        runtime_diagnostics.get(variant.dir_name()).cloned(),
        pinned,
    )
    .with_device_fit(&state.runtime, variant);
    if let Some(speech) = model.speech_capabilities.as_mut() {
        speech.streaming_mode = Some(state.runtime.tts_streaming_mode(variant).as_str());
    }
    Ok(Json(model))
}

async fn loaded_model_diagnostics_by_variant(
    state: &AppState,
) -> HashMap<String, serde_json::Value> {
    state
        .runtime
        .loaded_model_diagnostics()
        .await
        .into_iter()
        .filter_map(|diagnostics| {
            serde_json::to_value(&diagnostics)
                .ok()
                .map(|value| (diagnostics.variant_id, value))
        })
        .collect()
}

/// SSE progress event
type ProgressEvent = AdminModelDownloadProgressEvent;

/// Download progress response
pub type DownloadResponse = AdminModelActionResponse;

/// Download a model from HuggingFace (non-blocking)
pub async fn download_model(
    State(state): State<AppState>,
    Path(variant): Path<String>,
) -> Result<Json<DownloadResponse>, ApiError> {
    let variant = parse_variant(&variant)?;
    if !variant.is_enabled() {
        return Err(ApiError::not_found(format!(
            "Model {} is not available for download",
            variant.dir_name()
        )));
    }
    info!("Starting non-blocking download for model: {}", variant);

    // Check if already downloading
    if state.runtime.is_download_active(variant).await {
        return Ok(Json(DownloadResponse {
            status: "downloading",
            message: format!("Model {} is already being downloaded", variant),
            warning: None,
        }));
    }

    // Spawn non-blocking download
    state
        .runtime
        .spawn_download(variant)
        .await
        .map_err(core_api_error)?;

    Ok(Json(DownloadResponse {
        status: "started",
        message: format!("Download of {} started in background", variant),
        warning: None,
    }))
}

#[derive(Debug, Default, Deserialize)]
pub struct LoadModelQuery {
    /// Keep the model loaded until it is unloaded (the default). With
    /// `false` it stays loaded while idle, but a load that needs its room
    /// evicts it.
    #[serde(default)]
    pub pin: Option<bool>,
}

/// Load a model into memory
pub async fn load_model(
    State(state): State<AppState>,
    Path(variant): Path<String>,
    Query(query): Query<LoadModelQuery>,
) -> Result<Json<DownloadResponse>, ApiError> {
    let variant = parse_variant(&variant)?;
    info!("Loading model: {}", variant);

    let loaded = if query.pin.unwrap_or(true) {
        state.runtime.load_model(variant).await
    } else {
        state.runtime.load_model_unpinned(variant).await
    };
    loaded.map_err(core_api_error)?;

    Ok(Json(DownloadResponse {
        status: "loaded",
        message: format!("Model {} loaded successfully", variant),
        warning: state.runtime.take_load_warning(variant),
    }))
}

/// Keep a loaded model loaded until it is unloaded: other loads cannot
/// evict it.
pub async fn pin_model(
    State(state): State<AppState>,
    Path(variant): Path<String>,
) -> Result<Json<DownloadResponse>, ApiError> {
    let variant = parse_variant(&variant)?;
    if !state.runtime.pin_resident_model(variant).await {
        return Err(ApiError::of(
            ErrorClass::Conflict,
            format!("{} is not loaded: load it first", variant.display_name()),
        ));
    }
    Ok(Json(DownloadResponse {
        status: "pinned",
        message: format!(
            "{} stays loaded until you unload it",
            variant.display_name()
        ),
        warning: None,
    }))
}

/// Let other loads evict a loaded model when they need its room.
pub async fn unpin_model(
    State(state): State<AppState>,
    Path(variant): Path<String>,
) -> Result<Json<DownloadResponse>, ApiError> {
    let variant = parse_variant(&variant)?;
    if !state.runtime.unpin_resident_model(variant).await {
        return Err(ApiError::of(
            ErrorClass::Conflict,
            format!("{} is not loaded", variant.display_name()),
        ));
    }
    Ok(Json(DownloadResponse {
        status: "unpinned",
        message: format!(
            "{} stays loaded until another model needs its room",
            variant.display_name()
        ),
        warning: None,
    }))
}

/// Where models are stored and how much room is left there.
#[derive(Debug, Serialize, ToSchema)]
pub struct AdminStorageResponse {
    pub models_dir: String,
    /// Free bytes on the volume holding the models; `None` when it cannot
    /// be read.
    pub free_bytes: Option<u64>,
    pub total_bytes: Option<u64>,
    /// Disk the downloaded models use.
    pub models_bytes: u64,
}

pub async fn storage(
    State(state): State<AppState>,
) -> Result<Json<AdminStorageResponse>, ApiError> {
    let models_dir = state.runtime.config().models_dir.clone();
    let models_bytes = state
        .runtime
        .list_models()
        .await
        .into_iter()
        .filter(|model| {
            matches!(
                model.status,
                ModelStatus::Downloaded | ModelStatus::Loading | ModelStatus::Ready
            )
        })
        .filter_map(|model| model.size_bytes)
        .fold(0u64, u64::saturating_add);
    let probe_dir = models_dir.clone();
    let (free_bytes, total_bytes) = tokio::task::spawn_blocking(move || {
        // A fresh install has no models directory yet: ask its volume.
        let volume = probe_dir
            .ancestors()
            .find(|path| path.exists())
            .map(std::path::Path::to_path_buf)
            .unwrap_or(probe_dir);
        (
            fs4::available_space(&volume).ok(),
            fs4::total_space(&volume).ok(),
        )
    })
    .await
    .unwrap_or((None, None));
    Ok(Json(AdminStorageResponse {
        models_dir: models_dir.to_string_lossy().into_owned(),
        free_bytes,
        total_bytes,
        models_bytes,
    }))
}

/// Unload a model from memory
pub async fn unload_model(
    State(state): State<AppState>,
    Path(variant): Path<String>,
) -> Result<Json<DownloadResponse>, ApiError> {
    let variant = parse_variant(&variant)?;
    info!("Unloading model: {}", variant);

    let cancelled_jobs =
        crate::api::jobs::cancel_active_audio_jobs_for_model(&state, variant).await?;
    unload_after_audio_job_cancellation(&state, variant, cancelled_jobs).await?;

    Ok(Json(DownloadResponse {
        status: "unloaded",
        message: if cancelled_jobs == 0 {
            format!("Model {} unloaded successfully", variant)
        } else {
            format!(
                "Model {} unloaded successfully after cancelling {} audio job(s)",
                variant, cancelled_jobs
            )
        },
        warning: None,
    }))
}

/// Delete a downloaded model from disk
pub async fn delete_model(
    State(state): State<AppState>,
    Path(variant): Path<String>,
) -> Result<Json<DownloadResponse>, ApiError> {
    // A removed variant cannot load or run, so its leftover download is
    // deleted directly (the only operation it still supports).
    if let Some(removed) = removed_model_variant(&variant) {
        info!("Deleting removed model download: {}", removed.id);
        if !state
            .runtime
            .model_manager()
            .delete_removed_model(removed)
            .await
            .map_err(core_api_error)?
        {
            return Err(ApiError::not_found(format!(
                "Removed model {} has no download to delete",
                removed.id
            )));
        }
        return Ok(Json(DownloadResponse {
            status: "deleted",
            message: format!("Removed model {} deleted successfully", removed.id),
            warning: None,
        }));
    }
    let variant = parse_variant(&variant)?;
    info!("Deleting model: {}", variant);

    // First cancel durable speech work, then unload through the runtime path so
    // registry/executor/cache cleanup cannot race a batch retry that reloads it.
    let cancelled_jobs =
        crate::api::jobs::cancel_active_audio_jobs_for_model(&state, variant).await?;
    unload_after_audio_job_cancellation(&state, variant, cancelled_jobs).await?;

    // Delete the model files
    state
        .runtime
        .model_manager()
        .delete_model(variant)
        .await
        .map_err(core_api_error)?;

    Ok(Json(DownloadResponse {
        status: "deleted",
        message: format!("Model {} deleted successfully", variant),
        warning: None,
    }))
}

async fn unload_after_audio_job_cancellation(
    state: &AppState,
    variant: ModelVariant,
    cancelled_jobs: usize,
) -> Result<(), ApiError> {
    const CANCEL_SETTLE_TIMEOUT: Duration = Duration::from_secs(5);
    const CANCEL_SETTLE_POLL: Duration = Duration::from_millis(50);

    let deadline = Instant::now() + CANCEL_SETTLE_TIMEOUT;
    loop {
        match state.runtime.unload_model(variant).await {
            Ok(()) => return Ok(()),
            Err(err)
                if cancelled_jobs > 0
                    && err.kind() == izwi_core::ErrorKind::ModelBusy
                    && Instant::now() < deadline =>
            {
                tokio::time::sleep(CANCEL_SETTLE_POLL).await;
            }
            Err(err) => return Err(core_api_error(err)),
        }
    }
}

/// Stream download progress as SSE
pub async fn download_progress_stream(
    State(state): State<AppState>,
    Path(variant): Path<String>,
) -> Result<
    Sse<impl Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>>,
    ApiError,
> {
    use axum::response::sse::Event;

    let variant = parse_variant(&variant)?;
    info!("Starting SSE progress stream for: {}", variant);

    // Get progress receiver from engine.
    // If no active stream exists (normal race after completion), emit a terminal
    // one-shot event instead of returning 500 so EventSource clients don't log errors.
    let mut progress_rx = match state
        .runtime
        .model_manager()
        .subscribe_progress(variant)
        .await
    {
        Ok(rx) => rx,
        Err(err) => {
            warn!(
                "No active progress stream for {} ({}); returning terminal progress snapshot",
                variant, err
            );
            let fallback_event = terminal_progress_event(&state, variant).await;
            let stream: Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>> =
                Box::pin(async_stream::stream! {
                    let json = serde_json::to_string(&fallback_event).unwrap_or_default();
                    yield Ok::<Event, Infallible>(Event::default().data(json));
                });
            return Ok(Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default()));
        }
    };

    let stream: Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>> = Box::pin(
        async_stream::stream! {
            let mut last_event: Option<ProgressEvent> = None;
            loop {
                match progress_rx.recv().await {
                    Ok(progress) => {
                        let is_completed = progress.files_total > 0
                            && progress.files_completed >= progress.files_total
                            && (progress.total_bytes == 0
                                || progress.downloaded_bytes >= progress.total_bytes);

                        let event = ProgressEvent {
                            variant: progress.variant.dir_name().to_string(),
                            downloaded_bytes: progress.downloaded_bytes,
                            total_bytes: progress.total_bytes,
                            current_file: progress.current_file.clone(),
                            current_file_downloaded: progress.current_file_downloaded,
                            current_file_total: progress.current_file_total,
                            files_completed: progress.files_completed,
                            files_total: progress.files_total,
                            percent: progress.total_percent(),
                            status: if is_completed {
                                "completed".to_string()
                            } else {
                                "downloading".to_string()
                            },
                            error: None,
                        };

                        let json = serde_json::to_string(&event).unwrap_or_default();
                        yield Ok(Event::default().data(json));
                        last_event = Some(event.clone());

                        // Stop if download is complete
                        if event.status == "completed" {
                            break;
                        }
                    }
                    Err(RecvError::Lagged(skipped)) => {
                        warn!(
                            "Download progress stream lagged for {} (skipped {} updates); continuing",
                            variant, skipped
                        );
                        continue;
                    }
                    Err(RecvError::Closed) => {
                        // Channel closed: the download task has exited. Emit one final state
                        // so clients do not see a silent stream cutoff.
                        let final_state = state
                            .runtime
                            .model_manager()
                            .get_download_state(variant)
                            .await;

                        match final_state {
                            DownloadState::Downloaded => {
                                if let Some(mut event) = last_event.clone() {
                                    event.status = "completed".to_string();
                                    if event.total_bytes > 0 {
                                        event.downloaded_bytes = event.total_bytes;
                                        event.percent = 100.0;
                                    }
                                    event.files_completed = event.files_total.max(event.files_completed);
                                    event.current_file = String::new();
                                    event.current_file_downloaded = 0;
                                    event.current_file_total = 0;
                                    let json = serde_json::to_string(&event).unwrap_or_default();
                                    yield Ok(Event::default().data(json));
                                } else {
                                    let event = ProgressEvent {
                                        variant: variant.dir_name().to_string(),
                                        downloaded_bytes: 0,
                                        total_bytes: 0,
                                        current_file: String::new(),
                                        current_file_downloaded: 0,
                                        current_file_total: 0,
                                        files_completed: 0,
                                        files_total: 0,
                                        percent: 100.0,
                                        status: "completed".to_string(),
                                        error: None,
                                    };
                                    let json = serde_json::to_string(&event).unwrap_or_default();
                                    yield Ok(Event::default().data(json));
                                }
                            }
                            DownloadState::Error => {
                                let mut event = last_event.clone().unwrap_or(ProgressEvent {
                                    variant: variant.dir_name().to_string(),
                                    downloaded_bytes: 0,
                                    total_bytes: 0,
                                    current_file: String::new(),
                                    current_file_downloaded: 0,
                                    current_file_total: 0,
                                    files_completed: 0,
                                    files_total: 0,
                                    percent: 0.0,
                                    status: "error".to_string(),
                                    error: None,
                                });
                                event.status = "error".to_string();
                                event.error = state
                                    .runtime
                                    .model_manager()
                                    .download_error(variant)
                                    .await;
                                let json = serde_json::to_string(&event).unwrap_or_default();
                                yield Ok(Event::default().data(json));
                            }
                            DownloadState::NotDownloaded => {
                                let mut event = last_event.clone().unwrap_or(ProgressEvent {
                                    variant: variant.dir_name().to_string(),
                                    downloaded_bytes: 0,
                                    total_bytes: 0,
                                    current_file: String::new(),
                                    current_file_downloaded: 0,
                                    current_file_total: 0,
                                    files_completed: 0,
                                    files_total: 0,
                                    percent: 0.0,
                                    status: "cancelled".to_string(),
                                    error: None,
                                });
                                event.status = "cancelled".to_string();
                                let json = serde_json::to_string(&event).unwrap_or_default();
                                yield Ok(Event::default().data(json));
                            }
                            DownloadState::Downloading => {}
                        }
                        break;
                    }
                }
            }
        },
    );

    Ok(Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default()))
}

async fn terminal_progress_event(state: &AppState, variant: ModelVariant) -> ProgressEvent {
    let download_state = state
        .runtime
        .model_manager()
        .get_download_state(variant)
        .await;
    let known_size = state
        .runtime
        .model_manager()
        .get_model_info(variant)
        .await
        .and_then(|info| info.size_bytes)
        .unwrap_or(0);

    match download_state {
        DownloadState::Downloaded => ProgressEvent {
            variant: variant.dir_name().to_string(),
            downloaded_bytes: known_size,
            total_bytes: known_size,
            current_file: String::new(),
            current_file_downloaded: 0,
            current_file_total: 0,
            files_completed: 1,
            files_total: 1,
            percent: 100.0,
            status: "completed".to_string(),
            error: None,
        },
        DownloadState::Error => ProgressEvent {
            variant: variant.dir_name().to_string(),
            downloaded_bytes: 0,
            total_bytes: known_size,
            current_file: String::new(),
            current_file_downloaded: 0,
            current_file_total: 0,
            files_completed: 0,
            files_total: 0,
            percent: 0.0,
            status: "error".to_string(),
            error: state.runtime.model_manager().download_error(variant).await,
        },
        DownloadState::NotDownloaded => ProgressEvent {
            variant: variant.dir_name().to_string(),
            downloaded_bytes: 0,
            total_bytes: known_size,
            current_file: String::new(),
            current_file_downloaded: 0,
            current_file_total: 0,
            files_completed: 0,
            files_total: 0,
            percent: 0.0,
            status: "cancelled".to_string(),
            error: None,
        },
        DownloadState::Downloading => ProgressEvent {
            variant: variant.dir_name().to_string(),
            downloaded_bytes: 0,
            total_bytes: known_size,
            current_file: String::new(),
            current_file_downloaded: 0,
            current_file_total: 0,
            files_completed: 0,
            files_total: 0,
            percent: 0.0,
            status: "error".to_string(),
            error: None,
        },
    }
}

/// Cancel an active download
pub async fn cancel_download(
    State(state): State<AppState>,
    Path(variant): Path<String>,
) -> Result<Json<DownloadResponse>, ApiError> {
    let variant = parse_variant(&variant)?;
    info!("Cancelling download for: {}", variant);

    // Check if download is active
    if !state.runtime.is_download_active(variant).await {
        return Ok(Json(DownloadResponse {
            status: "not_active",
            message: format!("No active download for {}", variant),
            warning: None,
        }));
    }

    // Cancel the download
    match state.runtime.model_manager().cancel_download(variant).await {
        Ok(_) => Ok(Json(DownloadResponse {
            status: "cancelled",
            message: format!("Download of {} cancelled", variant),
            warning: None,
        })),
        Err(e) => {
            warn!("Failed to cancel download: {}", e);
            Err(ApiError::internal(format!(
                "Failed to cancel download: {}",
                e
            )))
        }
    }
}

/// Parse model variant from string
fn parse_variant(s: &str) -> Result<ModelVariant, ApiError> {
    parse_model_variant(s).map_err(|err| ApiError::bad_request(err.to_string()))
}

fn model_sort_key(model: &ModelInfo) -> (u8, u8, u8, String) {
    let variant = model.variant;
    let dir_name = variant.dir_name();
    (
        model_type_rank(variant),
        model_size_rank(dir_name),
        model_precision_rank(dir_name),
        dir_name.to_string(),
    )
}

fn model_type_rank(variant: ModelVariant) -> u8 {
    if variant.is_diarization() {
        0
    } else if variant.is_asr() {
        1
    } else if variant.is_tts() {
        2
    } else if variant.is_chat() {
        3
    } else if variant.is_forced_aligner() {
        4
    } else if variant.is_tokenizer() {
        5
    } else if variant.is_voxtral() {
        6
    } else {
        7
    }
}

fn model_size_rank(dir_name: &str) -> u8 {
    let normalized = dir_name.to_ascii_lowercase();
    if normalized.contains("0.6b") {
        0
    } else if normalized.contains("1b") {
        1
    } else if normalized.contains("1.7b") {
        2
    } else if normalized.contains("4b") {
        3
    } else {
        4
    }
}

fn model_precision_rank(dir_name: &str) -> u8 {
    if dir_name.contains("-4bit") {
        2
    } else if dir_name.contains("-8bit") {
        3
    } else if dir_name.contains("-bf16") {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_core::LoadedModelDiagnostics;

    /// The realtime TTS flag matches what `/v1/voice/realtime/ws` accepts:
    /// a speech model without built-in voices or streaming is not offered.
    #[test]
    fn realtime_tts_capability_matches_the_voice_socket() {
        let realtime_tts =
            |variant| AdminModelRouteCapabilities::from_variant(variant).voice_realtime_modular_tts;
        assert!(realtime_tts(ModelVariant::Kokoro82M));
        assert!(!realtime_tts(ModelVariant::Qwen3Tts12Hz06BBase));
        assert!(
            AdminModelRouteCapabilities::from_variant(ModelVariant::Qwen3Tts12Hz06BBase)
                .openai_audio_speech
        );
    }

    /// P1.0: naming a removed variant (CLI `pull` reads it through this
    /// route) is a 400 that names the dense replacement.
    #[test]
    fn removed_model_ids_are_bad_requests_naming_the_replacement() {
        let error = parse_variant("Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit").unwrap_err();
        assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(
            error.message.contains("Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
            "{}",
            error.message
        );
    }

    #[test]
    fn admin_model_runtime_diagnostics_preserve_qwen38_family_and_policy_values() {
        let diagnostics = LoadedModelDiagnostics {
            variant_id: ModelVariant::Qwen3827BFp8.dir_name().to_string(),
            variant: ModelVariant::Qwen3827BFp8.to_string(),
            family: "qwen38_chat",
            task: "chat",
            handle_kind: "native_chat",
            loaded_model_kind: "qwen38_chat",
            backend_kind: "cuda".to_string(),
            device_kind: "Cuda".to_string(),
            actual_device_kind: Some("cuda".to_string()),
            actual_compute_dtype: Some("f16".to_string()),
            default_compute_dtype: "bf16".to_string(),
            default_dtype_reason: "CUDA policy prefers BF16".to_string(),
            effective_context_tokens: None,
            weight_residency: None,
            kv_storage: None,
            supports_incremental_decode: Some(true),
            supports_realtime_stream_decode: None,
            family_diagnostics: Some(serde_json::json!({
                "checkpoint_format": "safetensors_block_fp8",
                "resident_representation": "q8_0_requantized_projections_with_dense_bf16",
                "fp8_execution_mode": "q8_0_compressed_fallback",
                "fallback_reason": "native FP8 execution is not runtime-certified",
            })),
        };
        let model = AdminModelInfo::from_model_info(
            ModelInfo::new(ModelVariant::Qwen3827BFp8),
            Some(serde_json::to_value(diagnostics).expect("serialize runtime diagnostics")),
            true,
        );

        let value = serde_json::to_value(model).expect("serialize admin model");
        assert_eq!(value["variant"], "Qwen3.8-27B-FP8");
        assert_eq!(value["pinned"], true);
        assert_eq!(value["chat_capabilities"]["supports_thinking"], true);
        assert_eq!(value["chat_capabilities"]["thinking_toggleable"], true);
        assert_eq!(
            value["chat_capabilities"]["default_reasoning_effort"],
            "xhigh"
        );
        assert_eq!(
            value["chat_capabilities"]["reasoning_efforts"],
            serde_json::json!(["xhigh", "medium", "low"])
        );
        assert_eq!(
            value["chat_capabilities"]["supports_preserve_thinking"],
            true
        );
        assert_eq!(value["runtime_diagnostics"]["family"], "qwen38_chat");
        assert_eq!(
            value["runtime_diagnostics"]["loaded_model_kind"],
            "qwen38_chat"
        );
        assert_eq!(value["runtime_diagnostics"]["actual_compute_dtype"], "f16");
        assert_eq!(
            value["runtime_diagnostics"]["default_compute_dtype"],
            "bf16"
        );
        assert_eq!(
            value["runtime_diagnostics"]["family_diagnostics"]["resident_representation"],
            "q8_0_requantized_projections_with_dense_bf16"
        );
        assert_eq!(
            value["runtime_diagnostics"]["family_diagnostics"]["fp8_execution_mode"],
            "q8_0_compressed_fallback"
        );
        assert!(
            value["runtime_diagnostics"]["family_diagnostics"]["fallback_reason"]
                .as_str()
                .is_some_and(|reason| reason.contains("not runtime-certified"))
        );
    }
}
