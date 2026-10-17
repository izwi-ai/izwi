//! OpenAI-compatible chat completions on this process's runtime. Request
//! validation and answer rendering are izwi-gateway's, which the gateway's
//! route shares.

use axum::{
    extract::{Extension, State},
    response::Response,
};

pub use izwi_gateway::api::openai::chat::completions::*;

use crate::api::openai::chat::dto::ChatCompletionRequest;
use crate::api::openai::compat::compatibility_profile;
use crate::api::request_context::RequestContext;
use crate::app::chat::LocalChat;
use crate::error::{ApiError, ApiJson};
use crate::state::AppState;

pub async fn completions(
    State(state): State<AppState>,
    Extension(ctx): Extension<RequestContext>,
    ApiJson(req): ApiJson<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let profile = compatibility_profile();
    validate_chat_request_compatibility(&req, profile)?;
    let (execution_request, format) = prepare_execution_request(&req, &ctx, profile)?;
    complete_chat(
        LocalChat(state),
        req,
        execution_request,
        format,
        ctx.stream_outcome.clone(),
    )
    .await
}
