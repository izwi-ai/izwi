use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Extension, Path, State},
    Json,
};
use izwi_agent::{
    planner::{PlanningMode, SimplePlanner},
    AgentDefinition, AgentEngine, AgentEvent, AgentSession, AgentTurnOptions, MemoryMessage,
    MemoryMessageMeta, MemoryMessageRole, MemoryStore, ModelBackend, ModelOutput, ModelRequest,
    NoopTool, TimeTool, ToolRegistry, TurnInput,
};
use izwi_core::{parse_chat_model_variant, ChatMessage, ChatRole};
use serde::{Deserialize, Serialize};

use crate::api::request_context::RequestContext;
use crate::chat_store::ChatStore;
use crate::error::ApiError;
use crate::state::{AppState, StoredAgentSessionRecord};
use crate::voice_defaults::{
    DEFAULT_VOICE_AGENT_ID, DEFAULT_VOICE_AGENT_NAME, DEFAULT_VOICE_AGENT_SYSTEM_PROMPT,
};
const DEFAULT_CHAT_MODEL: &str = "Qwen3-1.7B-GGUF";

#[derive(Debug, Deserialize)]
pub struct CreateAgentSessionRequest {
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub planning_mode: Option<PlanningMode>,
    #[serde(default)]
    pub title: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AgentSessionResponse {
    pub id: String,
    pub agent_id: String,
    pub thread_id: String,
    pub model_id: String,
    pub planning_mode: PlanningMode,
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
    pub plan: Option<izwi_agent::PlanSummary>,
    pub tool_calls: Vec<izwi_agent::ToolCallRecord>,
    pub events: Vec<AgentEvent>,
}

pub async fn create_session(
    State(state): State<AppState>,
    Json(req): Json<CreateAgentSessionRequest>,
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
    let planning_mode = req.planning_mode.unwrap_or(PlanningMode::Auto);

    let thread = state
        .chat_store
        .create_thread(req.title, Some(model_id.clone()))
        .await
        .map_err(map_store_error)?;

    let now = now_unix_millis();
    let session_id = format!("agent_sess_{}", uuid::Uuid::new_v4().simple());
    let record = StoredAgentSessionRecord {
        id: session_id.clone(),
        agent_id: agent_id.clone(),
        thread_id: thread.id.clone(),
        model_id: model_id.clone(),
        system_prompt,
        planning_mode,
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
    Json(req): Json<CreateAgentTurnRequest>,
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

    let agent = AgentDefinition {
        id: session_record.agent_id.clone(),
        name: DEFAULT_VOICE_AGENT_NAME.to_string(),
        system_prompt: session_record.system_prompt.clone(),
        default_model: session_record.model_id.clone(),
        capabilities: Default::default(),
        planning_mode: session_record.planning_mode,
    };
    let session = AgentSession {
        id: session_record.id.clone(),
        agent_id: session_record.agent_id.clone(),
        thread_id: session_record.thread_id.clone(),
        created_at: session_record.created_at,
        updated_at: session_record.updated_at,
    };

    let memory = ChatStoreMemory::new(state.chat_store.clone());
    let backend = IzwiRuntimeBackend {
        runtime: state.runtime.clone(),
        correlation_id: ctx.correlation_id,
    };
    let planner = SimplePlanner;
    let mut tools = ToolRegistry::new();
    tools.register(NoopTool);
    tools.register(TimeTool);

    let engine = AgentEngine;
    let result = engine
        .run_turn(
            &agent,
            &session,
            TurnInput { text: req.input },
            Some(resolved_model_id.clone()),
            &memory,
            &backend,
            &planner,
            &tools,
            AgentTurnOptions {
                max_output_tokens: resolve_max_output_tokens_for_model(
                    &resolved_model_id,
                    req.max_output_tokens,
                )?,
                max_tool_calls: 1,
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
        plan: result.plan,
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
        planning_mode: record.planning_mode,
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
        izwi_core::ModelVariant::Gemma31BIt => 4096,
        izwi_core::ModelVariant::Lfm2512BInstructGguf => 4096,
        izwi_core::ModelVariant::Lfm2512BThinkingGguf => 4096,
        _ => 1536,
    };
    Ok(requested.unwrap_or(default).clamp(1, 4096))
}

fn now_unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn map_store_error(err: anyhow::Error) -> ApiError {
    ApiError::internal(format!("Chat storage error: {err}"))
}

fn map_agent_error(err: izwi_agent::AgentError) -> ApiError {
    match err {
        izwi_agent::AgentError::InvalidInput(msg) => ApiError::bad_request(msg),
        other => ApiError::internal(other.to_string()),
    }
}

struct ChatStoreMemory {
    chat_store: Arc<ChatStore>,
}

impl ChatStoreMemory {
    fn new(chat_store: Arc<ChatStore>) -> Self {
        Self { chat_store }
    }
}

#[async_trait::async_trait]
impl MemoryStore for ChatStoreMemory {
    async fn load_messages(&self, thread_id: &str) -> izwi_agent::Result<Vec<MemoryMessage>> {
        let records = self
            .chat_store
            .list_messages(thread_id.to_string())
            .await
            .map_err(|err| izwi_agent::AgentError::Memory(err.to_string()))?;

        let mut out = Vec::with_capacity(records.len());
        for record in records {
            let role = match record.role.as_str() {
                "system" => MemoryMessageRole::System,
                "user" => MemoryMessageRole::User,
                "assistant" => MemoryMessageRole::Assistant,
                other => {
                    return Err(izwi_agent::AgentError::Memory(format!(
                        "Invalid stored chat role: {other}"
                    )))
                }
            };
            out.push(MemoryMessage {
                role,
                content: record.content,
            });
        }

        Ok(out)
    }

    async fn append_message(
        &self,
        thread_id: &str,
        role: MemoryMessageRole,
        content: String,
        meta: MemoryMessageMeta,
    ) -> izwi_agent::Result<()> {
        self.chat_store
            .append_message(
                thread_id.to_string(),
                role.as_str().to_string(),
                content,
                None,
                meta.model_id,
                meta.tokens_generated,
                meta.generation_time_ms,
            )
            .await
            .map_err(|err| izwi_agent::AgentError::Memory(err.to_string()))?;
        Ok(())
    }
}

struct IzwiRuntimeBackend {
    runtime: Arc<izwi_core::RuntimeService>,
    correlation_id: String,
}

#[async_trait::async_trait]
impl ModelBackend for IzwiRuntimeBackend {
    async fn generate(&self, request: ModelRequest) -> izwi_agent::Result<ModelOutput> {
        let variant = parse_chat_model_variant(Some(&request.model_id))
            .map_err(|err| izwi_agent::AgentError::Model(err.to_string()))?;

        let mut runtime_messages = Vec::with_capacity(request.messages.len());
        for message in request.messages {
            let role = match message.role {
                MemoryMessageRole::System => ChatRole::System,
                MemoryMessageRole::User => ChatRole::User,
                MemoryMessageRole::Assistant => ChatRole::Assistant,
            };
            runtime_messages.push(ChatMessage {
                role,
                content: message.content,
            });
        }

        let generation = self
            .runtime
            .chat_generate_with_correlation(
                variant,
                runtime_messages,
                request.max_output_tokens.clamp(1, 4096),
                Some(&self.correlation_id),
            )
            .await
            .map_err(|err| izwi_agent::AgentError::Model(err.to_string()))?;

        Ok(ModelOutput {
            text: generation.text,
            tokens_generated: generation.tokens_generated,
            generation_time_ms: generation.generation_time_ms,
        })
    }
}
