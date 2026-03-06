//! Application state management with high-concurrency optimizations

use crate::chat_store::ChatStore;
use crate::diarization_store::DiarizationStore;
use crate::saved_voice_store::SavedVoiceStore;
use crate::speech_history_store::SpeechHistoryStore;
use crate::transcription_store::TranscriptionStore;
use izwi_agent::planner::PlanningMode;
use izwi_core::RuntimeService;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredResponseInputItem {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredResponseRecord {
    pub id: String,
    pub created_at: u64,
    pub status: String,
    pub model: String,
    pub input_items: Vec<StoredResponseInputItem>,
    pub output_text: Option<String>,
    #[serde(default)]
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub error: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredAgentSessionRecord {
    pub id: String,
    pub agent_id: String,
    pub thread_id: String,
    pub model_id: String,
    pub system_prompt: String,
    pub planning_mode: PlanningMode,
    pub created_at: u64,
    pub updated_at: u64,
}

/// Shared application state with fine-grained locking and backpressure
#[derive(Clone)]
pub struct AppState {
    /// Runtime service reference - using Arc for cheap clones
    pub runtime: Arc<RuntimeService>,
    /// Concurrency limiter to prevent resource exhaustion
    pub request_semaphore: Arc<Semaphore>,
    /// Request timeout configuration (seconds)
    pub request_timeout_secs: u64,
    /// In-memory store for OpenAI-compatible `/v1/responses` objects.
    pub response_store: Arc<RwLock<HashMap<String, StoredResponseRecord>>>,
    /// In-memory store for minimal agent sessions.
    pub agent_session_store: Arc<RwLock<HashMap<String, StoredAgentSessionRecord>>>,
    /// SQLite-backed chat thread/message store.
    pub chat_store: Arc<ChatStore>,
    /// SQLite-backed transcription history store.
    pub transcription_store: Arc<TranscriptionStore>,
    /// SQLite-backed diarization history store.
    pub diarization_store: Arc<DiarizationStore>,
    /// SQLite-backed speech generation history store.
    pub speech_history_store: Arc<SpeechHistoryStore>,
    /// SQLite-backed saved voice store.
    pub saved_voice_store: Arc<SavedVoiceStore>,
}

impl AppState {
    pub fn new(runtime: RuntimeService) -> anyhow::Result<Self> {
        // Limit concurrent requests to prevent overwhelming the system
        // Default: 100 concurrent requests (tunable based on hardware)
        let max_concurrent = std::env::var("MAX_CONCURRENT_REQUESTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(100);

        let timeout = std::env::var("REQUEST_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300); // 5 minutes default

        let chat_store = Arc::new(ChatStore::initialize()?);
        let transcription_store = Arc::new(TranscriptionStore::initialize()?);
        let diarization_store = Arc::new(DiarizationStore::initialize()?);
        let speech_history_store = Arc::new(SpeechHistoryStore::initialize()?);
        let saved_voice_store = Arc::new(SavedVoiceStore::initialize()?);

        Ok(Self {
            runtime: Arc::new(runtime),
            request_semaphore: Arc::new(Semaphore::new(max_concurrent)),
            request_timeout_secs: timeout,
            response_store: Arc::new(RwLock::new(HashMap::new())),
            agent_session_store: Arc::new(RwLock::new(HashMap::new())),
            chat_store,
            transcription_store,
            diarization_store,
            speech_history_store,
            saved_voice_store,
        })
    }

    /// Acquire a permit for concurrent request processing
    pub async fn acquire_permit(&self) -> tokio::sync::SemaphorePermit<'_> {
        self.request_semaphore
            .acquire()
            .await
            .expect("Semaphore should never be closed")
    }
}
