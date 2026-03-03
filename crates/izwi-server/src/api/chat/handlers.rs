use std::convert::Infallible;
use std::path::Path as FsPath;
use std::time::Duration;

use axum::{
    body::Body,
    extract::{Extension, Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::Engine;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::api::request_context::RequestContext;
use crate::chat_store::{ChatThreadMessage, ChatThreadSummary};
use crate::error::ApiError;
use crate::state::AppState;
use crate::storage_layout::{self, ChatMediaKind};
use izwi_core::ModelVariant;
use izwi_core::{
    parse_chat_model_variant, qwen35_multimodal_control_content, qwen35_thinking_control_content,
    ChatMessage, ChatRole, Qwen35MultimodalInput, Qwen35MultimodalKind,
};

const QWEN_VISION_IMAGE_TOKEN: &str = "<|vision_start|><|image_pad|><|vision_end|>";
const QWEN_VISION_VIDEO_TOKEN: &str = "<|vision_start|><|video_pad|><|vision_end|>";

#[derive(Debug, Serialize)]
pub struct ChatThreadListResponse {
    pub threads: Vec<ChatThreadSummary>,
}

#[derive(Debug, Deserialize)]
pub struct CreateChatThreadRequest {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatThreadDetailResponse {
    pub thread: ChatThreadSummary,
    pub messages: Vec<ChatThreadMessage>,
}

#[derive(Debug, Serialize)]
pub struct DeleteChatThreadResponse {
    pub id: String,
    pub deleted: bool,
}

#[derive(Debug, Deserialize)]
pub struct UpdateChatThreadRequest {
    pub title: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct CreateThreadMessageRequest {
    pub model: String,
    pub content: String,
    #[serde(default)]
    pub content_parts: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub enable_thinking: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatGenerationStats {
    pub tokens_generated: usize,
    pub generation_time_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CreateThreadMessageResponse {
    pub thread_id: String,
    pub model_id: String,
    pub user_message: ChatThreadMessage,
    pub assistant_message: ChatThreadMessage,
    pub stats: ChatGenerationStats,
}

#[derive(Debug, Serialize)]
struct ThreadStreamStartEvent {
    event: &'static str,
    thread_id: String,
    model_id: String,
    user_message: ChatThreadMessage,
}

#[derive(Debug, Serialize)]
struct ThreadStreamDeltaEvent {
    event: &'static str,
    delta: String,
}

#[derive(Debug, Serialize)]
struct ThreadStreamDoneEvent {
    event: &'static str,
    thread_id: String,
    model_id: String,
    assistant_message: ChatThreadMessage,
    stats: ChatGenerationStats,
}

#[derive(Debug, Serialize)]
struct ThreadStreamErrorEvent {
    event: &'static str,
    error: String,
}

pub async fn list_threads(
    State(state): State<AppState>,
) -> Result<Json<ChatThreadListResponse>, ApiError> {
    let threads = state
        .chat_store
        .list_threads()
        .await
        .map_err(map_store_error)?;

    Ok(Json(ChatThreadListResponse { threads }))
}

pub async fn create_thread(
    State(state): State<AppState>,
    Json(req): Json<CreateChatThreadRequest>,
) -> Result<Json<ChatThreadSummary>, ApiError> {
    let thread = state
        .chat_store
        .create_thread(req.title, req.model_id)
        .await
        .map_err(map_store_error)?;

    Ok(Json(thread))
}

pub async fn get_thread(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
) -> Result<Json<ChatThreadDetailResponse>, ApiError> {
    let thread = get_thread_or_not_found(&state, &thread_id).await?;
    let messages = state
        .chat_store
        .list_messages(thread_id)
        .await
        .map_err(map_store_error)?;

    Ok(Json(ChatThreadDetailResponse { thread, messages }))
}

pub async fn list_thread_messages(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
) -> Result<Json<Vec<ChatThreadMessage>>, ApiError> {
    get_thread_or_not_found(&state, &thread_id).await?;
    let messages = state
        .chat_store
        .list_messages(thread_id)
        .await
        .map_err(map_store_error)?;

    Ok(Json(messages))
}

pub async fn delete_thread(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
) -> Result<Json<DeleteChatThreadResponse>, ApiError> {
    let deleted = state
        .chat_store
        .delete_thread(thread_id.clone())
        .await
        .map_err(map_store_error)?;

    if !deleted {
        return Err(ApiError::not_found("Thread not found"));
    }

    Ok(Json(DeleteChatThreadResponse {
        id: thread_id,
        deleted: true,
    }))
}

pub async fn update_thread(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
    Json(req): Json<UpdateChatThreadRequest>,
) -> Result<Json<ChatThreadSummary>, ApiError> {
    if req.title.trim().is_empty() {
        return Err(ApiError::bad_request("Thread title cannot be empty"));
    }

    let updated = state
        .chat_store
        .update_thread_title(thread_id, req.title)
        .await
        .map_err(map_store_error)?;

    let thread = updated.ok_or_else(|| ApiError::not_found("Thread not found"))?;
    Ok(Json(thread))
}

pub async fn create_thread_message(
    State(state): State<AppState>,
    Path(thread_id): Path<String>,
    Extension(ctx): Extension<RequestContext>,
    Json(req): Json<CreateThreadMessageRequest>,
) -> Result<Response, ApiError> {
    let prepared_content_parts = persist_chat_media_parts(req.content_parts.clone())?;
    let flattened_content = flatten_thread_content(&req.content, prepared_content_parts.as_deref())
        .map_err(|err| {
            ApiError::bad_request(format!("Invalid chat message content payload: {err}"))
        })?;
    if flattened_content.runtime.trim().is_empty() {
        return Err(ApiError::bad_request("Message content cannot be empty"));
    }

    get_thread_or_not_found(&state, &thread_id).await?;
    let existing_messages = state
        .chat_store
        .list_messages(thread_id.clone())
        .await
        .map_err(map_store_error)?;

    let model_variant = parse_chat_model(&req.model)?;
    let model_id = model_variant.dir_name().to_string();
    if !flattened_content.multimodal.is_empty() && !is_qwen35_chat_variant(model_variant) {
        return Err(ApiError::bad_request(
            "Image/video inputs in threaded chat are currently supported only for Qwen3.5 models",
        ));
    }

    let user_message = state
        .chat_store
        .append_message(
            thread_id.clone(),
            "user".to_string(),
            flattened_content.display.clone(),
            prepared_content_parts.clone(),
            Some(model_id.clone()),
            None,
            None,
        )
        .await
        .map_err(map_store_or_not_found)?;

    let runtime_messages = build_runtime_messages(
        model_variant,
        &existing_messages,
        &flattened_content.runtime,
        req.system_prompt.as_deref(),
        req.enable_thinking,
        &flattened_content.multimodal,
    )?;

    if req.stream.unwrap_or(false) {
        return create_streaming_thread_message(
            state,
            req,
            model_variant,
            model_id,
            thread_id,
            user_message,
            runtime_messages,
            ctx.correlation_id,
        )
        .await;
    }

    let _permit = state.acquire_permit().await;
    let generation = state
        .runtime
        .chat_generate_with_correlation(
            model_variant,
            runtime_messages,
            max_new_tokens(model_variant, req.max_completion_tokens, req.max_tokens),
            Some(&ctx.correlation_id),
        )
        .await?;

    let assistant_message = state
        .chat_store
        .append_message(
            thread_id.clone(),
            "assistant".to_string(),
            generation.text.clone(),
            None,
            Some(model_id.clone()),
            Some(generation.tokens_generated),
            Some(generation.generation_time_ms),
        )
        .await
        .map_err(map_store_or_not_found)?;

    let response = CreateThreadMessageResponse {
        thread_id,
        model_id,
        user_message,
        assistant_message,
        stats: ChatGenerationStats {
            tokens_generated: generation.tokens_generated,
            generation_time_ms: generation.generation_time_ms,
        },
    };

    Ok(Json(response).into_response())
}

async fn create_streaming_thread_message(
    state: AppState,
    req: CreateThreadMessageRequest,
    model_variant: ModelVariant,
    model_id: String,
    thread_id: String,
    user_message: ChatThreadMessage,
    runtime_messages: Vec<ChatMessage>,
    correlation_id: String,
) -> Result<Response, ApiError> {
    let timeout = Duration::from_secs(state.request_timeout_secs);
    let max_tokens = max_new_tokens(model_variant, req.max_completion_tokens, req.max_tokens);
    let semaphore = state.request_semaphore.clone();
    let runtime = state.runtime.clone();
    let chat_store = state.chat_store.clone();

    let thread_id_for_task = thread_id.clone();
    let model_id_for_task = model_id.clone();
    let user_message_for_start = user_message.clone();

    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<String>();

    tokio::spawn(async move {
        let _permit = match semaphore.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                let _ = event_tx.send(
                    serde_json::to_string(&ThreadStreamErrorEvent {
                        event: "error",
                        error: "Server is shutting down".to_string(),
                    })
                    .unwrap_or_default(),
                );
                let _ = event_tx.send("[DONE]".to_string());
                return;
            }
        };

        let _ = event_tx.send(
            serde_json::to_string(&ThreadStreamStartEvent {
                event: "start",
                thread_id: thread_id_for_task.clone(),
                model_id: model_id_for_task.clone(),
                user_message: user_message_for_start,
            })
            .unwrap_or_default(),
        );

        let full_text = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let full_text_for_delta = full_text.clone();
        let delta_tx = event_tx.clone();

        let generation_result = tokio::time::timeout(timeout, async {
            runtime
                .chat_generate_streaming_with_correlation(
                    model_variant,
                    runtime_messages,
                    max_tokens,
                    Some(correlation_id.as_str()),
                    move |delta| {
                        if let Ok(mut output_text) = full_text_for_delta.lock() {
                            output_text.push_str(&delta);
                        }
                        let payload = ThreadStreamDeltaEvent {
                            event: "delta",
                            delta,
                        };
                        let _ = delta_tx.send(serde_json::to_string(&payload).unwrap_or_default());
                    },
                )
                .await
        })
        .await;

        match generation_result {
            Ok(Ok(generation)) => {
                let assistant_text = full_text.lock().map(|s| s.clone()).unwrap_or_default();
                match chat_store
                    .append_message(
                        thread_id_for_task.clone(),
                        "assistant".to_string(),
                        assistant_text,
                        None,
                        Some(model_id_for_task.clone()),
                        Some(generation.tokens_generated),
                        Some(generation.generation_time_ms),
                    )
                    .await
                {
                    Ok(assistant_message) => {
                        let done_event = ThreadStreamDoneEvent {
                            event: "done",
                            thread_id: thread_id_for_task,
                            model_id: model_id_for_task,
                            assistant_message,
                            stats: ChatGenerationStats {
                                tokens_generated: generation.tokens_generated,
                                generation_time_ms: generation.generation_time_ms,
                            },
                        };
                        let _ =
                            event_tx.send(serde_json::to_string(&done_event).unwrap_or_default());
                    }
                    Err(err) => {
                        let _ = event_tx.send(
                            serde_json::to_string(&ThreadStreamErrorEvent {
                                event: "error",
                                error: format!("Failed to persist assistant message: {err}"),
                            })
                            .unwrap_or_default(),
                        );
                    }
                }
            }
            Ok(Err(err)) => {
                let _ = event_tx.send(
                    serde_json::to_string(&ThreadStreamErrorEvent {
                        event: "error",
                        error: err.to_string(),
                    })
                    .unwrap_or_default(),
                );
            }
            Err(_) => {
                let _ = event_tx.send(
                    serde_json::to_string(&ThreadStreamErrorEvent {
                        event: "error",
                        error: "Chat request timed out".to_string(),
                    })
                    .unwrap_or_default(),
                );
            }
        }

        let _ = event_tx.send("[DONE]".to_string());
    });

    let stream = async_stream::stream! {
        while let Some(payload) = event_rx.recv().await {
            yield Ok::<_, Infallible>(format!("data: {payload}\n\n"));
            if payload == "[DONE]" {
                break;
            }
        }
    };

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(stream))
        .unwrap())
}

fn max_new_tokens(
    variant: ModelVariant,
    max_completion_tokens: Option<usize>,
    max_tokens: Option<usize>,
) -> usize {
    let requested = max_completion_tokens.or(max_tokens);

    let default = match variant {
        ModelVariant::Gemma34BIt => 4096,
        ModelVariant::Gemma31BIt => 4096,
        _ => 1536,
    };

    requested.unwrap_or(default).clamp(1, 4096)
}

fn parse_chat_model(model_id: &str) -> Result<ModelVariant, ApiError> {
    parse_chat_model_variant(Some(model_id)).map_err(|err| ApiError::bad_request(err.to_string()))
}

fn build_runtime_messages(
    model_variant: ModelVariant,
    existing: &[ChatThreadMessage],
    new_user_content: &str,
    system_prompt: Option<&str>,
    enable_thinking: Option<bool>,
    new_user_multimodal: &[Qwen35MultimodalInput],
) -> Result<Vec<ChatMessage>, ApiError> {
    let mut messages = Vec::new();

    if is_qwen35_chat_variant(model_variant) {
        if let Some(enable_thinking) = enable_thinking {
            messages.push(ChatMessage {
                role: ChatRole::System,
                content: qwen35_thinking_control_content(enable_thinking),
            });
        }
    }

    if let Some(prompt) = system_prompt
        .map(str::trim)
        .filter(|prompt| !prompt.is_empty())
    {
        messages.push(ChatMessage {
            role: ChatRole::System,
            content: prompt.to_string(),
        });
    }

    if !new_user_multimodal.is_empty() {
        if is_qwen35_chat_variant(model_variant) {
            if let Some(control) = qwen35_multimodal_control_content(new_user_multimodal) {
                messages.push(ChatMessage {
                    role: ChatRole::System,
                    content: control,
                });
            }
        } else {
            return Err(ApiError::bad_request(
                "Image/video inputs are currently supported only for Qwen3.5 chat models",
            ));
        }
    }

    for message in existing {
        let role = parse_stored_role(&message.role)?;
        messages.push(ChatMessage {
            role,
            content: message.content.clone(),
        });
    }

    messages.push(ChatMessage {
        role: ChatRole::User,
        content: new_user_content.to_string(),
    });

    Ok(messages)
}

fn is_qwen35_chat_variant(variant: ModelVariant) -> bool {
    matches!(
        variant,
        ModelVariant::Qwen3508B
            | ModelVariant::Qwen352B
            | ModelVariant::Qwen354B
            | ModelVariant::Qwen359B
    )
}

fn persist_chat_media_parts(
    content_parts: Option<Vec<serde_json::Value>>,
) -> Result<Option<Vec<serde_json::Value>>, ApiError> {
    let Some(mut parts) = content_parts else {
        return Ok(None);
    };
    if parts.is_empty() {
        return Ok(Some(parts));
    }

    let media_root = storage_layout::resolve_media_root();
    let db_path = storage_layout::resolve_db_path();
    storage_layout::ensure_storage_dirs(&db_path, &media_root).map_err(|err| {
        ApiError::internal(format!("Failed preparing media storage directories: {err}"))
    })?;

    for part in &mut parts {
        if content_part_is_image(part) {
            persist_part_media_source(part, Qwen35MultimodalKind::Image, &media_root)?;
            continue;
        }
        if content_part_is_video(part) {
            persist_part_media_source(part, Qwen35MultimodalKind::Video, &media_root)?;
        }
    }

    Ok(Some(parts))
}

fn persist_part_media_source(
    part: &mut serde_json::Value,
    kind: Qwen35MultimodalKind,
    media_root: &FsPath,
) -> Result<(), ApiError> {
    let keys: &[&str] = match kind {
        Qwen35MultimodalKind::Image => &["input_image", "image_url", "image"],
        Qwen35MultimodalKind::Video => &["input_video", "video_url", "video"],
    };

    if let Some(map) = part.as_object_mut() {
        for key in keys {
            if let Some(value) = map.get_mut(*key) {
                persist_media_value(value, kind, media_root)?;
                return Ok(());
            }
        }
    }

    persist_media_value(part, kind, media_root)
}

fn persist_media_value(
    value: &mut serde_json::Value,
    kind: Qwen35MultimodalKind,
    media_root: &FsPath,
) -> Result<(), ApiError> {
    let mut source = None;
    let mut preferred_name = None;
    let mut preferred_mime = None;

    match value {
        serde_json::Value::String(raw) => {
            source = Some(raw.trim().to_string());
        }
        serde_json::Value::Object(map) => {
            for key in ["url", "uri", "source", "data"] {
                if let Some(candidate) = map.get(key).and_then(|entry| entry.as_str()) {
                    let candidate = candidate.trim();
                    if !candidate.is_empty() {
                        source = Some(candidate.to_string());
                        break;
                    }
                }
            }

            if source.is_none() {
                if let Some(b64_json) = map.get("b64_json").and_then(|entry| entry.as_str()) {
                    let b64 = b64_json.trim();
                    if !b64.is_empty() {
                        let mime = map
                            .get("mime_type")
                            .or_else(|| map.get("media_type"))
                            .or_else(|| map.get("content_type"))
                            .and_then(|entry| entry.as_str())
                            .map(str::trim)
                            .filter(|entry| !entry.is_empty())
                            .unwrap_or("application/octet-stream");
                        source = Some(format!("data:{mime};base64,{b64}"));
                    }
                }
            }

            preferred_name = map
                .get("name")
                .or_else(|| map.get("file_name"))
                .or_else(|| map.get("filename"))
                .and_then(|entry| entry.as_str())
                .map(str::trim)
                .filter(|entry| !entry.is_empty())
                .map(ToString::to_string);

            preferred_mime = map
                .get("media_type")
                .or_else(|| map.get("mime_type"))
                .or_else(|| map.get("content_type"))
                .and_then(|entry| entry.as_str())
                .map(str::trim)
                .filter(|entry| !entry.is_empty())
                .map(ToString::to_string);
        }
        _ => {}
    }

    let Some(source) = source else {
        return Ok(());
    };
    if !source.to_ascii_lowercase().starts_with("data:") {
        return Ok(());
    }

    let (mime_type, bytes) = decode_data_url(&source)?;
    let effective_mime = preferred_mime.unwrap_or(mime_type);
    let relative_path = storage_layout::persist_chat_media_file(
        media_root,
        media_kind_for_storage(kind),
        preferred_name.as_deref(),
        Some(effective_mime.as_str()),
        &bytes,
    )
    .map_err(|err| ApiError::internal(format!("Failed persisting chat attachment: {err}")))?;

    let public_url = format!("/v1/media/{relative_path}");
    let absolute_path = media_root
        .join(&relative_path)
        .to_string_lossy()
        .to_string();

    match value {
        serde_json::Value::String(_) => {
            *value = serde_json::json!({
                "url": public_url,
                "path": absolute_path,
                "media_type": effective_mime,
            });
        }
        serde_json::Value::Object(map) => {
            map.insert("url".to_string(), serde_json::Value::String(public_url));
            map.insert("path".to_string(), serde_json::Value::String(absolute_path));
            map.insert(
                "media_type".to_string(),
                serde_json::Value::String(effective_mime),
            );
            map.remove("b64_json");
            if map
                .get("encoding")
                .and_then(|entry| entry.as_str())
                .is_some_and(|encoding| encoding.eq_ignore_ascii_case("base64"))
            {
                map.remove("data");
            }
        }
        _ => {}
    }

    Ok(())
}

fn media_kind_for_storage(kind: Qwen35MultimodalKind) -> ChatMediaKind {
    match kind {
        Qwen35MultimodalKind::Image => ChatMediaKind::Image,
        Qwen35MultimodalKind::Video => ChatMediaKind::Video,
    }
}

fn decode_data_url(data_url: &str) -> Result<(String, Vec<u8>), ApiError> {
    let Some(without_prefix) = data_url.strip_prefix("data:") else {
        return Err(ApiError::bad_request("Unsupported media URL format"));
    };
    let Some((metadata, payload)) = without_prefix.split_once(',') else {
        return Err(ApiError::bad_request("Malformed data URL payload"));
    };

    let mut mime_type = "application/octet-stream";
    let mut base64_payload = false;
    for (index, segment) in metadata.split(';').enumerate() {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        if index == 0 && segment.contains('/') {
            mime_type = segment;
            continue;
        }
        if segment.eq_ignore_ascii_case("base64") {
            base64_payload = true;
        }
    }

    if !base64_payload {
        return Err(ApiError::bad_request(
            "Only base64-encoded data URLs are supported for chat media",
        ));
    }

    let payload = payload.trim();
    if payload.is_empty() {
        return Err(ApiError::bad_request("Data URL payload is empty"));
    }

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(payload))
        .map_err(|_| ApiError::bad_request("Invalid base64 data URL payload"))?;

    Ok((mime_type.to_string(), bytes))
}

#[derive(Debug, Default)]
struct FlattenedThreadContent {
    display: String,
    runtime: String,
    multimodal: Vec<Qwen35MultimodalInput>,
}

fn flatten_thread_content(
    raw_content: &str,
    content_parts: Option<&[serde_json::Value]>,
) -> Result<FlattenedThreadContent, String> {
    let raw_trimmed = raw_content.trim().to_string();
    let Some(parts) = content_parts else {
        return Ok(FlattenedThreadContent {
            display: raw_trimmed.clone(),
            runtime: raw_trimmed,
            multimodal: Vec::new(),
        });
    };

    if parts.is_empty() {
        return Ok(FlattenedThreadContent {
            display: raw_trimmed.clone(),
            runtime: raw_trimmed,
            multimodal: Vec::new(),
        });
    }

    let mut out = FlattenedThreadContent::default();
    for part in parts {
        if content_part_is_image(part) {
            let media =
                media_from_part_value(part, Qwen35MultimodalKind::Image).ok_or_else(|| {
                    "Image content part is missing a usable source URL/data".to_string()
                })?;
            out.runtime.push_str(QWEN_VISION_IMAGE_TOKEN);
            if !out.display.is_empty() {
                out.display.push('\n');
            }
            out.display
                .push_str(&display_media_label(part, Qwen35MultimodalKind::Image));
            out.multimodal.push(media);
            continue;
        }
        if content_part_is_video(part) {
            let media =
                media_from_part_value(part, Qwen35MultimodalKind::Video).ok_or_else(|| {
                    "Video content part is missing a usable source URL/data".to_string()
                })?;
            out.runtime.push_str(QWEN_VISION_VIDEO_TOKEN);
            if !out.display.is_empty() {
                out.display.push('\n');
            }
            out.display
                .push_str(&display_media_label(part, Qwen35MultimodalKind::Video));
            out.multimodal.push(media);
            continue;
        }
        if let Some(text) = resolve_text_part(part) {
            out.runtime.push_str(&text);
            out.display.push_str(&text);
        }
    }

    if out.runtime.trim().is_empty() && !raw_trimmed.is_empty() {
        out.runtime = raw_trimmed.clone();
        out.display = raw_trimmed;
    }

    Ok(out)
}

fn display_media_label(part: &serde_json::Value, kind: Qwen35MultimodalKind) -> String {
    let prefix = match kind {
        Qwen35MultimodalKind::Image => "[image]",
        Qwen35MultimodalKind::Video => "[video]",
    };
    let name = media_display_name(part, kind);
    if let Some(name) = name {
        return format!("{prefix} {name}");
    }
    prefix.to_string()
}

fn media_display_name(part: &serde_json::Value, kind: Qwen35MultimodalKind) -> Option<String> {
    let map = part.as_object()?;
    let mut values = Vec::new();
    match kind {
        Qwen35MultimodalKind::Image => {
            values.push(map.get("input_image"));
            values.push(map.get("image_url"));
            values.push(map.get("image"));
        }
        Qwen35MultimodalKind::Video => {
            values.push(map.get("input_video"));
            values.push(map.get("video_url"));
            values.push(map.get("video"));
        }
    }
    values.push(map.get("name"));
    values.push(map.get("file_name"));
    values.push(map.get("filename"));

    for value in values.into_iter().flatten() {
        if let Some(name) = resolve_display_name(value) {
            return Some(name);
        }
    }
    None
}

fn resolve_display_name(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                return None;
            }
            let candidate = trimmed
                .split('/')
                .next_back()
                .unwrap_or(trimmed)
                .split('?')
                .next()
                .unwrap_or(trimmed)
                .split('#')
                .next()
                .unwrap_or(trimmed)
                .trim();
            if candidate.is_empty() {
                None
            } else {
                Some(candidate.to_string())
            }
        }
        serde_json::Value::Object(map) => {
            for key in ["name", "file_name", "filename"] {
                if let Some(name) = map.get(key).and_then(|entry| entry.as_str()) {
                    let name = name.trim();
                    if !name.is_empty() {
                        return Some(name.to_string());
                    }
                }
            }
            map.get("url").and_then(resolve_display_name)
        }
        _ => None,
    }
}

fn resolve_text_part(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(raw) => {
            let trimmed = raw.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        serde_json::Value::Object(map) => map
            .get("text")
            .or_else(|| map.get("input_text"))
            .and_then(|v| v.as_str())
            .map(str::to_string),
        _ => None,
    }
}

fn content_part_is_image(part: &serde_json::Value) -> bool {
    let Some(map) = part.as_object() else {
        return false;
    };
    if matches!(
        map.get("type")
            .or_else(|| map.get("kind"))
            .and_then(|v| v.as_str()),
        Some("image") | Some("image_url") | Some("input_image")
    ) {
        return true;
    }
    map.contains_key("image") || map.contains_key("image_url") || map.contains_key("input_image")
}

fn content_part_is_video(part: &serde_json::Value) -> bool {
    let Some(map) = part.as_object() else {
        return false;
    };
    if matches!(
        map.get("type")
            .or_else(|| map.get("kind"))
            .and_then(|v| v.as_str()),
        Some("video") | Some("video_url") | Some("input_video")
    ) {
        return true;
    }
    map.contains_key("video") || map.contains_key("video_url") || map.contains_key("input_video")
}

fn media_from_part_value(
    value: &serde_json::Value,
    kind: Qwen35MultimodalKind,
) -> Option<Qwen35MultimodalInput> {
    let map = value.as_object();
    let source = match kind {
        Qwen35MultimodalKind::Image => map.and_then(|entry| {
            ["image_url", "input_image", "image"]
                .into_iter()
                .find_map(|key| entry.get(key).and_then(|v| resolve_media_source(v, 3)))
        }),
        Qwen35MultimodalKind::Video => map.and_then(|entry| {
            ["video", "video_url", "input_video"]
                .into_iter()
                .find_map(|key| entry.get(key).and_then(|v| resolve_media_source(v, 3)))
        }),
    }
    .or_else(|| resolve_media_source(value, 3))?;

    Some(Qwen35MultimodalInput { kind, source })
}

fn resolve_media_source(value: &serde_json::Value, max_depth: usize) -> Option<String> {
    if max_depth == 0 {
        return None;
    }

    match value {
        serde_json::Value::String(raw) => {
            let source = raw.trim();
            if source.is_empty() {
                None
            } else {
                Some(source.to_string())
            }
        }
        serde_json::Value::Object(map) => {
            for key in [
                "path",
                "file",
                "file_path",
                "src",
                "uri",
                "url",
                "image_url",
                "video_url",
                "input_image",
                "input_video",
            ] {
                if let Some(source) = map
                    .get(key)
                    .and_then(|v| resolve_media_source(v, max_depth - 1))
                {
                    return Some(source);
                }
            }

            if let Some(data_url) = map
                .get("b64_json")
                .and_then(|v| v.as_str())
                .and_then(|b64| data_url_from_base64_field(b64, map))
            {
                return Some(data_url);
            }

            if let Some(data) = map.get("data").and_then(|v| v.as_str()) {
                let data = data.trim();
                if data.starts_with("data:")
                    || data.starts_with("http://")
                    || data.starts_with("https://")
                    || data.starts_with("file://")
                {
                    return Some(data.to_string());
                }

                let is_base64 = map
                    .get("encoding")
                    .and_then(|v| v.as_str())
                    .is_some_and(|encoding| encoding.eq_ignore_ascii_case("base64"));
                if is_base64 {
                    return data_url_from_base64_field(data, map);
                }
            }

            None
        }
        _ => None,
    }
}

fn data_url_from_base64_field(
    b64: &str,
    map: &serde_json::Map<String, serde_json::Value>,
) -> Option<String> {
    let payload = b64.trim();
    if payload.is_empty() {
        return None;
    }
    let mime = map
        .get("mime_type")
        .or_else(|| map.get("media_type"))
        .or_else(|| map.get("content_type"))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("application/octet-stream");
    Some(format!("data:{mime};base64,{payload}"))
}

fn parse_stored_role(role: &str) -> Result<ChatRole, ApiError> {
    match role {
        "system" => Ok(ChatRole::System),
        "user" => Ok(ChatRole::User),
        "assistant" => Ok(ChatRole::Assistant),
        other => Err(ApiError::internal(format!(
            "Invalid stored chat role: {other}"
        ))),
    }
}

async fn get_thread_or_not_found(
    state: &AppState,
    thread_id: &str,
) -> Result<ChatThreadSummary, ApiError> {
    let thread = state
        .chat_store
        .get_thread(thread_id.to_string())
        .await
        .map_err(map_store_error)?;

    thread.ok_or_else(|| ApiError::not_found("Thread not found"))
}

fn map_store_error(err: anyhow::Error) -> ApiError {
    ApiError::internal(format!("Chat storage error: {err}"))
}

fn map_store_or_not_found(err: anyhow::Error) -> ApiError {
    let error_text = err.to_string();
    if error_text.contains("Thread not found") {
        ApiError::not_found("Thread not found")
    } else {
        map_store_error(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_core::parse_qwen35_multimodal_control_content;
    use serde_json::json;

    #[test]
    fn flattens_thread_multimodal_parts() {
        let flattened = flatten_thread_content(
            "",
            Some(&[
                json!({"type":"text","text":"Look "}),
                json!({"type":"input_image","input_image":{"url":"https://example.com/cat.png"}}),
                json!({"type":"input_video","input_video":{"url":"https://example.com/clip.mp4"}}),
                json!({"type":"text","text":" now"}),
            ]),
        )
        .expect("flatten thread content");

        assert_eq!(
            flattened.runtime,
            format!("Look {QWEN_VISION_IMAGE_TOKEN}{QWEN_VISION_VIDEO_TOKEN} now")
        );
        assert_eq!(flattened.multimodal.len(), 2);
        assert_eq!(flattened.multimodal[0].kind, Qwen35MultimodalKind::Image);
        assert_eq!(flattened.multimodal[1].kind, Qwen35MultimodalKind::Video);
    }

    #[test]
    fn flatten_thread_content_rejects_missing_media_source() {
        let err = flatten_thread_content("", Some(&[json!({"type":"image_url","image_url":{}})]))
            .expect_err("missing source should fail");
        assert!(err.contains("missing a usable source"));
    }

    #[test]
    fn build_runtime_messages_injects_multimodal_control_for_qwen35() {
        let multimodal = vec![Qwen35MultimodalInput {
            kind: Qwen35MultimodalKind::Image,
            source: "https://example.com/cat.png".to_string(),
        }];
        let messages = build_runtime_messages(
            ModelVariant::Qwen352B,
            &[],
            &format!("Describe {QWEN_VISION_IMAGE_TOKEN}"),
            Some("You are helpful."),
            Some(true),
            &multimodal,
        )
        .expect("build runtime messages");

        let control = messages
            .iter()
            .find(|m| parse_qwen35_multimodal_control_content(&m.content).is_some())
            .expect("multimodal control message");
        assert!(matches!(control.role, ChatRole::System));
    }

    #[test]
    fn build_runtime_messages_rejects_multimodal_for_non_qwen35() {
        let multimodal = vec![Qwen35MultimodalInput {
            kind: Qwen35MultimodalKind::Image,
            source: "https://example.com/cat.png".to_string(),
        }];
        let err = build_runtime_messages(
            ModelVariant::Qwen306B4Bit,
            &[],
            &format!("Describe {QWEN_VISION_IMAGE_TOKEN}"),
            None,
            None,
            &multimodal,
        )
        .expect_err("non-qwen35 multimodal should fail");
        assert!(err.message.contains("supported only for Qwen3.5"));
    }
}
