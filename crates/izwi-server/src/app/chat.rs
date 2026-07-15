use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::Notify;

use crate::error::ApiError;
use crate::state::AppState;
use izwi_core::{
    parse_chat_model_variant, ChatGeneration, ChatMessage, ChatRequestConfig, GenerationParams,
    ModelVariant, WorkloadClass,
};

#[derive(Debug, Clone)]
pub struct ChatExecutionRequest {
    pub variant: ModelVariant,
    pub messages: Vec<ChatMessage>,
    pub max_completion_tokens: Option<usize>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub chat_config: ChatRequestConfig,
    pub correlation_id: Option<String>,
}

impl ChatExecutionRequest {
    fn resolved_max_new_tokens(&self) -> usize {
        max_new_tokens(self.variant, self.max_completion_tokens, self.max_tokens)
    }

    fn resolved_generation_params(&self) -> GenerationParams {
        let max_new_tokens = self.resolved_max_new_tokens();
        let mut params = GenerationParams::default();
        params.max_tokens = max_new_tokens;

        if let Some(temperature) = self.temperature {
            params.temperature = temperature;
        }
        if let Some(top_p) = self.top_p {
            params.top_p = top_p;
        }
        if let Some(presence_penalty) = self.presence_penalty {
            params.presence_penalty = presence_penalty;
        }
        params
    }

    fn resolved_chat_config(&self) -> ChatRequestConfig {
        self.chat_config.clone()
    }
}

#[derive(Debug, Clone)]
pub enum ChatStreamEvent {
    Started,
    Delta(String),
    Completed(ChatGeneration),
    Failed(String),
    ShuttingDown,
}

const CHAT_STREAM_CAPACITY: usize = 64;
const CHAT_STREAM_BACKPRESSURE_ERROR: &str =
    "Chat stream consumer is too slow; generation was cancelled";
#[cfg(not(test))]
const CHAT_TERMINAL_SEND_TIMEOUT: Duration = Duration::from_secs(5);
#[cfg(test)]
const CHAT_TERMINAL_SEND_TIMEOUT: Duration = Duration::from_millis(100);

#[derive(Debug, Default)]
struct ChatStreamBackpressure {
    full: AtomicBool,
    notify: Notify,
}

impl ChatStreamBackpressure {
    fn trip(&self) {
        self.full.store(true, Ordering::Release);
        self.notify.notify_one();
    }

    fn is_tripped(&self) -> bool {
        self.full.load(Ordering::Acquire)
    }

    async fn notified(&self) {
        if self.is_tripped() {
            return;
        }
        self.notify.notified().await;
    }
}

fn try_send_chat_delta(
    event_tx: &mpsc::Sender<ChatStreamEvent>,
    backpressure: &ChatStreamBackpressure,
    delta: String,
) {
    match event_tx.try_send(ChatStreamEvent::Delta(delta)) {
        Ok(()) => {}
        Err(TrySendError::Full(_)) => backpressure.trip(),
        // The receiver has gone away, so there is nobody to notify with a
        // terminal event. `event_tx.closed()` cancels the generation task.
        Err(TrySendError::Closed(_)) => {}
    }
}

async fn resolve_chat_terminal<F>(
    event_tx: &mpsc::Sender<ChatStreamEvent>,
    backpressure: Arc<ChatStreamBackpressure>,
    generation: F,
) -> Option<ChatStreamEvent>
where
    F: Future<Output = izwi_core::Result<ChatGeneration>>,
{
    tokio::pin!(generation);
    let result = tokio::select! {
        result = &mut generation => Some(result),
        _ = event_tx.closed() => return None,
        _ = backpressure.notified() => None,
    };

    // A generation can produce the overflowing delta and complete in the same
    // poll. Check the latch even when the generation branch wins the select so
    // a dropped delta can never be reported as a successful completion.
    if result.is_none() || backpressure.is_tripped() {
        return Some(ChatStreamEvent::Failed(
            CHAT_STREAM_BACKPRESSURE_ERROR.to_string(),
        ));
    }

    match result.expect("completed generation result must be present") {
        Ok(generation) => Some(ChatStreamEvent::Completed(generation)),
        Err(err) => Some(ChatStreamEvent::Failed(err.to_string())),
    }
}

async fn send_chat_terminal(event_tx: mpsc::Sender<ChatStreamEvent>, event: ChatStreamEvent) {
    // A connected receiver may stop polling forever. Terminal delivery remains
    // best-effort for that transport, but it must never retain inference or
    // workload capacity indefinitely.
    let _ = tokio::time::timeout(CHAT_TERMINAL_SEND_TIMEOUT, event_tx.send(event)).await;
}

pub fn max_new_tokens(
    _variant: ModelVariant,
    max_completion_tokens: Option<usize>,
    max_tokens: Option<usize>,
) -> usize {
    let requested = max_completion_tokens.or(max_tokens);

    requested.unwrap_or(4096).clamp(1, 4096)
}

pub fn parse_chat_model(model_id: &str) -> Result<ModelVariant, ApiError> {
    parse_chat_model_variant(Some(model_id)).map_err(|err| ApiError::bad_request(err.to_string()))
}

pub async fn generate_chat(
    state: &AppState,
    request: ChatExecutionRequest,
) -> Result<ChatGeneration, ApiError> {
    let params = request.resolved_generation_params();
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
        .map_err(ApiError::from)
}

pub fn spawn_chat_stream(
    state: AppState,
    request: ChatExecutionRequest,
) -> mpsc::Receiver<ChatStreamEvent> {
    let runtime = state.runtime.clone();
    let params = request.resolved_generation_params();
    let chat_config = request.resolved_chat_config();
    let variant = request.variant;
    let messages = request.messages;
    let correlation_id = request.correlation_id;

    let (event_tx, event_rx) = mpsc::channel(CHAT_STREAM_CAPACITY);
    let backpressure = Arc::new(ChatStreamBackpressure::default());
    tokio::spawn(async move {
        let permit = match state
            .acquire_owned_workload_permit(WorkloadClass::Streaming)
            .await
        {
            Ok(permit) => permit,
            Err(_) => {
                send_chat_terminal(event_tx, ChatStreamEvent::ShuttingDown).await;
                return;
            }
        };

        if event_tx.send(ChatStreamEvent::Started).await.is_err() {
            return;
        }

        let generation = runtime.chat_generate_streaming_with_runtime_context(
            variant,
            messages,
            params,
            chat_config,
            correlation_id.as_deref(),
            permit.runtime_context(),
            {
                let event_tx = event_tx.clone();
                let backpressure = backpressure.clone();
                move |delta| {
                    try_send_chat_delta(&event_tx, &backpressure, delta);
                }
            },
        );
        let terminal = resolve_chat_terminal(&event_tx, backpressure, generation).await;
        drop(permit);
        if let Some(terminal) = terminal {
            send_chat_terminal(event_tx, terminal).await;
        }
    });

    event_rx
}

#[cfg(test)]
fn spawn_chat_stream_with_task<G, Fut>(
    semaphore: Arc<tokio::sync::Semaphore>,
    capacity: usize,
    generation_task: G,
) -> mpsc::Receiver<ChatStreamEvent>
where
    G: FnOnce(mpsc::Sender<ChatStreamEvent>, Arc<ChatStreamBackpressure>) -> Fut + Send + 'static,
    Fut: Future<Output = izwi_core::Result<ChatGeneration>> + Send + 'static,
{
    let (event_tx, event_rx) = mpsc::channel(capacity);
    let backpressure = Arc::new(ChatStreamBackpressure::default());

    tokio::spawn(async move {
        let permit = match semaphore.acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                send_chat_terminal(event_tx, ChatStreamEvent::ShuttingDown).await;
                return;
            }
        };

        if event_tx.send(ChatStreamEvent::Started).await.is_err() {
            return;
        }

        let generation = generation_task(event_tx.clone(), backpressure.clone());
        let terminal = resolve_chat_terminal(&event_tx, backpressure, generation).await;
        drop(permit);
        if let Some(terminal) = terminal {
            send_chat_terminal(event_tx, terminal).await;
        }
    });

    event_rx
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_core::ChatRole;
    use std::time::Duration;
    use tokio::sync::Semaphore;

    #[test]
    fn explicit_overrides_win_over_default_generation_params() {
        let request = ChatExecutionRequest {
            variant: ModelVariant::Qwen34BGguf,
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: "hello".to_string(),
            }],
            max_completion_tokens: None,
            max_tokens: Some(32),
            temperature: Some(0.42),
            top_p: Some(0.73),
            presence_penalty: Some(0.25),
            chat_config: ChatRequestConfig::default(),
            correlation_id: None,
        };

        let params = request.resolved_generation_params();
        assert_eq!(params.temperature, 0.42);
        assert_eq!(params.top_p, 0.73);
        assert_eq!(params.top_k, 0);
        assert_eq!(params.presence_penalty, 0.25);
        assert_eq!(params.max_tokens, 32);
    }

    #[test]
    fn chat_models_default_to_4096_max_tokens_when_request_omits_limits() {
        for variant in [
            ModelVariant::Gemma34BIt,
            ModelVariant::Lfm2512BInstructGguf,
            ModelVariant::Qwen306BGguf,
            ModelVariant::Qwen317BGguf,
            ModelVariant::Qwen34BGguf,
            ModelVariant::Qwen38BGguf,
            ModelVariant::Qwen314BGguf,
            ModelVariant::Qwen352BGguf,
        ] {
            assert_eq!(max_new_tokens(variant, None, None), 4096);
        }
    }

    #[tokio::test]
    async fn streaming_chat_allows_long_running_generations_to_complete() {
        let semaphore = Arc::new(Semaphore::new(1));
        let mut event_rx =
            spawn_chat_stream_with_task(semaphore, 4, |event_tx, backpressure| async move {
                try_send_chat_delta(&event_tx, &backpressure, "Hello".to_string());
                tokio::time::sleep(Duration::from_millis(25)).await;
                try_send_chat_delta(&event_tx, &backpressure, " world".to_string());
                Ok(ChatGeneration {
                    text: "Hello world".to_string(),
                    prompt_tokens: 12,
                    tokens_generated: 2,
                    generation_time_ms: 25.0,
                })
            });

        match event_rx.recv().await {
            Some(ChatStreamEvent::Started) => {}
            other => panic!("expected stream start event, got {other:?}"),
        }

        match event_rx.recv().await {
            Some(ChatStreamEvent::Delta(delta)) => assert_eq!(delta, "Hello"),
            other => panic!("expected first delta event, got {other:?}"),
        }

        match event_rx.recv().await {
            Some(ChatStreamEvent::Delta(delta)) => assert_eq!(delta, " world"),
            other => panic!("expected second delta event, got {other:?}"),
        }

        match event_rx.recv().await {
            Some(ChatStreamEvent::Completed(generation)) => {
                assert_eq!(generation.text, "Hello world");
                assert_eq!(generation.tokens_generated, 2);
            }
            other => panic!("expected completed event, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn saturated_chat_stream_emits_explicit_terminal_failure() {
        let semaphore = Arc::new(Semaphore::new(1));
        let mut event_rx =
            spawn_chat_stream_with_task(semaphore, 2, |event_tx, backpressure| async move {
                try_send_chat_delta(&event_tx, &backpressure, "first".to_string());
                try_send_chat_delta(&event_tx, &backpressure, "overflow".to_string());
                // Completion in the same poll as the overflow must not win the
                // race and turn a truncated stream into apparent success.
                Ok(ChatGeneration {
                    text: "firstoverflow".to_string(),
                    prompt_tokens: 1,
                    tokens_generated: 2,
                    generation_time_ms: 1.0,
                })
            });

        assert!(matches!(
            event_rx.recv().await,
            Some(ChatStreamEvent::Started)
        ));
        assert!(matches!(
            event_rx.recv().await,
            Some(ChatStreamEvent::Delta(delta)) if delta == "first"
        ));
        assert!(matches!(
            event_rx.recv().await,
            Some(ChatStreamEvent::Failed(error)) if error == CHAT_STREAM_BACKPRESSURE_ERROR
        ));
        assert!(event_rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn non_draining_chat_consumer_cannot_retain_workload_capacity() {
        let semaphore = Arc::new(Semaphore::new(1));
        let generation_started = Arc::new(Notify::new());
        let release_generation = Arc::new(Notify::new());
        let started = generation_started.clone();
        let release = release_generation.clone();
        let mut event_rx = spawn_chat_stream_with_task(
            semaphore.clone(),
            1,
            move |_event_tx, _backpressure| async move {
                started.notify_one();
                release.notified().await;
                Ok(ChatGeneration {
                    text: "done".to_string(),
                    prompt_tokens: 1,
                    tokens_generated: 1,
                    generation_time_ms: 1.0,
                })
            },
        );

        generation_started.notified().await;
        assert_eq!(semaphore.available_permits(), 0);
        assert_eq!(
            event_rx.len(),
            1,
            "the unread Started event fills the queue"
        );
        release_generation.notify_one();
        tokio::time::timeout(Duration::from_millis(50), async {
            while semaphore.available_permits() == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("terminal transport wait must not retain the workload permit");

        tokio::time::sleep(CHAT_TERMINAL_SEND_TIMEOUT + Duration::from_millis(25)).await;
        assert!(matches!(
            event_rx.recv().await,
            Some(ChatStreamEvent::Started)
        ));
        assert!(event_rx.recv().await.is_none());
    }
}
