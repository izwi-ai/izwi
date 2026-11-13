use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Extension, Path, State},
    Json,
};
use izwi_agent::{
    AgentDefinition, AgentEngine, AgentEvent, AgentSession, AgentSet, AgentTurnOptions,
    BackgroundResults, TurnCancel, TurnContext, TurnEvents, TurnInput,
};
use izwi_core::{parse_chat_model_variant, WorkloadClass};
use serde::{Deserialize, Serialize};

use crate::api::request_context::RequestContext;
use crate::app::agent_runtime::{ChatAgentBackend, RoundAdmission, ThreadMemory};
use crate::error::{ApiError, ApiJson};
use crate::ids::new_uuid;
use crate::state::{AppState, StoredAgentSessionRecord};
use crate::voice_defaults::{
    DEFAULT_VOICE_AGENT_ID, DEFAULT_VOICE_AGENT_NAME, DEFAULT_VOICE_AGENT_SYSTEM_PROMPT,
};
const DEFAULT_CHAT_MODEL: &str = "Qwen3.5-4B";

#[derive(Debug, Deserialize)]
pub struct CreateAgentSessionRequest {
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Ignored: turns no longer plan ahead of the model.
    #[serde(default)]
    pub planning_mode: Option<serde_json::Value>,
    #[serde(default)]
    pub title: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AgentSessionResponse {
    pub id: String,
    pub agent_id: String,
    pub thread_id: String,
    pub model_id: String,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Deserialize)]
pub struct CreateAgentTurnRequest {
    pub input: String,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct AgentTurnResponse {
    pub session_id: String,
    pub thread_id: String,
    pub model_id: String,
    pub assistant_text: String,
    pub tool_calls: Vec<izwi_agent::ToolCallRecord>,
    pub events: Vec<AgentEvent>,
}

pub async fn create_session(
    State(state): State<AppState>,
    ApiJson(req): ApiJson<CreateAgentSessionRequest>,
) -> Result<Json<AgentSessionResponse>, ApiError> {
    let agent_id = req
        .agent_id
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_VOICE_AGENT_ID.to_string());

    let model_id = resolve_chat_model_id(req.model_id.as_deref())?;
    let system_prompt = req
        .system_prompt
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_VOICE_AGENT_SYSTEM_PROMPT)
        .to_string();
    let _ = req.planning_mode;

    let thread = state
        .chat_store
        .create_thread(req.title, Some(model_id.clone()))
        .await
        .map_err(map_store_error)?;

    let now = now_unix_millis();
    let session_id = new_uuid();
    let record = StoredAgentSessionRecord {
        id: session_id.clone(),
        agent_id: agent_id.clone(),
        thread_id: thread.id.clone(),
        model_id: model_id.clone(),
        system_prompt,
        created_at: now,
        updated_at: now,
    };

    state.store_agent_session_record(record.clone()).await;

    Ok(Json(session_response(record)))
}

pub async fn get_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> Result<Json<AgentSessionResponse>, ApiError> {
    let store = state.agent_session_store.read().await;
    let record = store
        .get(&session_id)
        .cloned()
        .ok_or_else(|| ApiError::not_found("Agent session not found"))?;
    Ok(Json(session_response(record)))
}

pub async fn create_turn(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    Extension(ctx): Extension<RequestContext>,
    ApiJson(req): ApiJson<CreateAgentTurnRequest>,
) -> Result<Json<AgentTurnResponse>, ApiError> {
    let session_record = {
        let store = state.agent_session_store.read().await;
        store
            .get(&session_id)
            .cloned()
            .ok_or_else(|| ApiError::not_found("Agent session not found"))?
    };

    let requested_model_id = match req.model_id.as_deref() {
        Some(model_id) => Some(resolve_chat_model_id(Some(model_id))?),
        None => None,
    };
    let resolved_model_id = requested_model_id
        .clone()
        .unwrap_or_else(|| session_record.model_id.clone());

    let agents = AgentSet::single(AgentDefinition::new(
        session_record.agent_id.clone(),
        DEFAULT_VOICE_AGENT_NAME,
        session_record.system_prompt.clone(),
        session_record.model_id.clone(),
    ));
    let session = AgentSession {
        id: session_record.id.clone(),
        agent_id: session_record.agent_id.clone(),
        thread_id: session_record.thread_id.clone(),
        created_at: session_record.created_at,
        updated_at: session_record.updated_at,
    };

    let permit = state.admit_request(WorkloadClass::Interactive).await?;
    let memory = ThreadMemory::new(state.chat_store.clone(), None);
    let backend = ChatAgentBackend {
        runtime: state.runtime.clone(),
        correlation_id: ctx.correlation_id,
        admission: RoundAdmission::Held(permit.runtime_context()),
    };
    let tools = crate::app::voice_tools::builtin_tool_registry();
    let turn = TurnContext {
        agents: &agents,
        memory: &memory,
        model: &backend,
        tools: &tools,
        events: TurnEvents::none(),
        cancel: TurnCancel::new(),
        background: BackgroundResults::default(),
    };
    let result = AgentEngine
        .run_turn(
            &turn,
            &session,
            TurnInput { text: req.input },
            Some(resolved_model_id.clone()),
            AgentTurnOptions {
                max_output_tokens: resolve_max_output_tokens_for_model(
                    &resolved_model_id,
                    req.max_output_tokens,
                )?,
                ..AgentTurnOptions::default()
            },
        )
        .await
        .map_err(map_agent_error)?;

    state
        .touch_agent_session_record(&session_id, now_unix_millis(), resolved_model_id.clone())
        .await;

    Ok(Json(AgentTurnResponse {
        session_id,
        thread_id: session_record.thread_id,
        model_id: result.model_id,
        assistant_text: result.assistant_text,
        tool_calls: result.tool_calls,
        events: result.events,
    }))
}

fn session_response(record: StoredAgentSessionRecord) -> AgentSessionResponse {
    AgentSessionResponse {
        id: record.id,
        agent_id: record.agent_id,
        thread_id: record.thread_id,
        model_id: record.model_id,
        created_at: record.created_at,
        updated_at: record.updated_at,
    }
}

fn resolve_chat_model_id(raw: Option<&str>) -> Result<String, ApiError> {
    let requested = raw
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_CHAT_MODEL);
    let variant = parse_chat_model_variant(Some(requested))
        .map_err(|err| ApiError::bad_request(err.to_string()))?;
    Ok(variant.dir_name().to_string())
}

fn resolve_max_output_tokens_for_model(
    model_id: &str,
    requested: Option<usize>,
) -> Result<usize, ApiError> {
    let variant = parse_chat_model_variant(Some(model_id))
        .map_err(|err| ApiError::bad_request(err.to_string()))?;
    let default = match variant {
        izwi_core::ModelVariant::Gemma34BIt => 4096,
        izwi_core::ModelVariant::Lfm2512BInstructGguf => 4096,
        izwi_core::ModelVariant::Lfm2512BThinkingGguf => 4096,
        // LFM3 reasons before it answers: room for its thinking and the
        // answer after it.
        izwi_core::ModelVariant::Lfm317BGguf | izwi_core::ModelVariant::Lfm317BQ8Gguf => 8192,
        _ => 1536,
    };
    Ok(requested
        .unwrap_or(default)
        .clamp(1, max_output_token_ceiling(variant)))
}

/// The most tokens one agent turn may generate: 8,192 for LFM3, which
/// thinks before it answers by default, 4,096 for every other model.
fn max_output_token_ceiling(variant: izwi_core::ModelVariant) -> usize {
    match variant {
        izwi_core::ModelVariant::Lfm317BGguf | izwi_core::ModelVariant::Lfm317BQ8Gguf => 8192,
        _ => 4096,
    }
}

fn now_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn map_store_error(err: anyhow::Error) -> ApiError {
    tracing::error!(error = %format!("{err:#}"), "agent chat storage error");
    ApiError::internal(format!("Chat storage error: {err:#}"))
}

fn map_agent_error(err: izwi_agent::AgentError) -> ApiError {
    match err {
        izwi_agent::AgentError::InvalidInput(msg) => ApiError::bad_request(msg),
        other => ApiError::internal(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lfm3_agent_turns_have_room_to_reason() {
        assert_eq!(
            resolve_max_output_tokens_for_model("LFM3-1.7B-GGUF", None).unwrap(),
            8192
        );
        assert_eq!(
            resolve_max_output_tokens_for_model("LFM3-1.7B-Q8_0-GGUF", Some(20_000)).unwrap(),
            8192
        );
        assert_eq!(
            resolve_max_output_tokens_for_model("LFM2.5-1.2B-Thinking-GGUF", Some(20_000)).unwrap(),
            4096
        );
        assert_eq!(
            resolve_max_output_tokens_for_model("Qwen3.5-4B", None).unwrap(),
            1536
        );
    }
}
