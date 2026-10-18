//! Chat on this process's own runtime. The request type, request
//! validation, the stream plumbing and the remote (worker) path live in
//! izwi-gateway, which both roles share.

use futures::FutureExt;
use izwi_core::{ChatGeneration, ChatGenerationControls, GenerationParams, WorkloadClass};
use tokio::sync::mpsc;

pub use izwi_gateway::app::chat::*;

use crate::error::ApiError;
use crate::state::AppState;

/// The engine generation parameters of a chat request: its resolved
/// sampling, output budget and controls.
pub(crate) fn chat_generation_params(request: &ChatExecutionRequest) -> GenerationParams {
    GenerationParams::for_chat(
        request.resolved_sampling(),
        request.resolved_max_new_tokens(),
        ChatGenerationControls {
            seed: request.seed,
            stop: request.stop.clone(),
            logprobs: request.requested_logprobs(),
            constrain_json_object: request.response_format_json_object,
            keep_special_tokens: request.skip_special_tokens == Some(false),
        },
    )
}

/// Chat on this process's own runtime.
pub(crate) struct LocalChat(pub(crate) AppState);

impl ChatBackend for LocalChat {
    type StreamGuard = ();

    async fn generate(self, request: ChatExecutionRequest) -> Result<ChatGeneration, ApiError> {
        generate_chat(&self.0, request).await
    }

    async fn stream(
        self,
        request: ChatExecutionRequest,
    ) -> Result<(mpsc::Receiver<ChatStreamEvent>, ()), ApiError> {
        Ok((start_chat_stream(self.0, request).await?, ()))
    }
}

pub async fn generate_chat(
    state: &AppState,
    request: ChatExecutionRequest,
) -> Result<ChatGeneration, ApiError> {
    let params = chat_generation_params(&request);
    let chat_config = request.resolved_chat_config();
    let variant = request.variant;
    let messages = request.messages;
    let correlation_id = request.correlation_id;
    let permit = state
        .acquire_workload_permit(WorkloadClass::Interactive)
        .await;

    state
        .runtime
        .chat_generate_with_runtime_context(
            variant,
            messages,
            params,
            chat_config,
            correlation_id.as_deref(),
            permit.runtime_context(),
        )
        .await
        .map_err(chat_error)
}

/// A local chat failure as an API error. An overflow is about the request's
/// `messages`, as the gateway reports it.
fn chat_error(err: izwi_core::Error) -> ApiError {
    let overflow = err.kind() == izwi_core::ErrorKind::ContextLengthExceeded;
    let error = crate::error::core_api_error(err);
    if overflow {
        error.with_param("messages")
    } else {
        error
    }
}

/// Start a chat stream on this process's runtime. The request is admitted
/// first, so a request that cannot run (its prompt does not fit the context,
/// the server is shutting down) fails here, before a streamed response
/// commits to `200`.
pub async fn start_chat_stream(
    state: AppState,
    request: ChatExecutionRequest,
) -> Result<mpsc::Receiver<ChatStreamEvent>, ApiError> {
    start_chat_stream_inner(state, request, (), None).await
}

/// [`start_chat_stream`] retaining `keepalive` until inference has fully
/// unwound and the response consumer has handled the terminal event. This is
/// used by persisted thread chats to keep their per-thread turn lock through
/// cancellation and atomic terminal persistence.
pub async fn start_chat_stream_with_keepalive<K>(
    state: AppState,
    request: ChatExecutionRequest,
    keepalive: K,
) -> Result<(mpsc::Receiver<ChatStreamEvent>, ChatStreamCompletion), ApiError>
where
    K: Send + 'static,
{
    let (completion, completion_rx) = ChatStreamCompletion::channel();
    let events = start_chat_stream_inner(state, request, keepalive, Some(completion_rx)).await?;
    Ok((events, completion))
}

async fn start_chat_stream_inner<K>(
    state: AppState,
    request: ChatExecutionRequest,
    keepalive: K,
    completion_rx: Option<tokio::sync::oneshot::Receiver<()>>,
) -> Result<mpsc::Receiver<ChatStreamEvent>, ApiError>
where
    K: Send + 'static,
{
    let runtime = state.runtime.clone();
    let shutdown = state.stream_shutdown.subscribe();
    let params = chat_generation_params(&request);
    let chat_config = request.resolved_chat_config();
    let permit = state
        .acquire_owned_workload_permit(WorkloadClass::Streaming)
        .await
        .map_err(|_| ApiError::service_unavailable("Server is shutting down"))?;
    let admitted = runtime
        .admit_streaming_chat(
            request.variant,
            request.messages,
            params,
            chat_config,
            request.correlation_id.as_deref(),
            permit.runtime_context(),
        )
        .await
        .map_err(chat_error)?;

    let (event_tx, event_rx) = mpsc::channel(CHAT_STREAM_CAPACITY);
    let sink = ChatDeltaSink::new(event_tx.clone(), izwi_core::stream_slow_consumer_timeout());
    tokio::spawn(async move {
        async move {
            if event_tx.send(ChatStreamEvent::Started).await.is_err() {
                return;
            }

            let generation = runtime
                .run_admitted_streaming_chat(admitted, {
                    let sink = sink.clone();
                    move |delta: String, logprobs: Vec<izwi_core::engine::TokenLogprob>| {
                        sink.offer(delta, logprobs);
                    }
                })
                .map(|result| result.map_err(chat_error));
            let terminal = resolve_chat_terminal(&sink, generation, Some(shutdown)).await;
            drop(permit);
            if let Some(terminal) = terminal {
                sink.finish(terminal).await;
            }
        }
        .await;

        if let Some(completion_rx) = completion_rx {
            let _ = completion_rx.await;
        }
        drop(keepalive);
    });

    Ok(event_rx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_core::{ChatMessage, ChatRequestConfig, ChatRole, ModelVariant};
    use izwi_serving_client::WorkerClientError;
    use izwi_serving_protocol::{AttemptId, InvocationRejection, RejectionCode, RequestId};

    #[test]
    fn explicit_overrides_win_over_default_generation_params() {
        let request = ChatExecutionRequest {
            variant: ModelVariant::Gemma34BIt,
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "hello".to_string(),
            }],
            max_completion_tokens: None,
            max_tokens: Some(32),
            temperature: Some(0.42),
            top_p: Some(0.73),
            top_k: None,
            repetition_penalty: None,
            presence_penalty: Some(0.25),
            seed: Some(7),
            stop: vec!["END".into()],
            logprobs: None,
            top_logprobs: None,
            response_format_json_object: false,
            skip_special_tokens: None,
            chat_config: ChatRequestConfig::default(),
            correlation_id: None,
        };

        let params = chat_generation_params(&request);
        assert_eq!(params.temperature, 0.42);
        assert_eq!(params.top_p, 0.73);
        assert_eq!(params.top_k, 0);
        assert_eq!(params.presence_penalty, 0.25);
        assert_eq!(params.max_tokens, 32);
        assert_eq!(params.seed, Some(7));
        assert_eq!(params.stop_sequences, ["END"]);
        assert!(params.skip_special_tokens);

        let keep_special = ChatExecutionRequest {
            skip_special_tokens: Some(false),
            ..request
        };
        assert!(!chat_generation_params(&keep_special).skip_special_tokens);
    }

    /// D8 over the wire: a worker's context overflow is the same OpenAI
    /// `context_length_exceeded` 400 the local server returns.
    #[test]
    fn worker_context_overflow_maps_like_the_local_server() {
        let message = "This model's maximum context length is 32 tokens.";
        let remote = map_worker_client_error(WorkerClientError::Rejected {
            rejection: InvocationRejection::new(
                RequestId::new("request-1").unwrap(),
                AttemptId::new("attempt-1").unwrap(),
                RejectionCode::ContextLengthExceeded,
                message,
            ),
        });
        let local = chat_error(izwi_core::Error::ContextLengthExceeded(message.into()));
        assert_eq!(remote.status, local.status);
        assert_eq!(remote.message, local.message);
        assert_eq!(remote.detail, local.detail);
    }
}
