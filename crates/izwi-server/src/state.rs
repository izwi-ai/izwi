//! Application state management with high-concurrency optimizations

use crate::chat_store::ChatStore;
use crate::diarization_store::DiarizationStore;
use crate::onboarding_store::OnboardingStore;
use crate::saved_voice_store::SavedVoiceStore;
use crate::speech_history_store::SpeechHistoryStore;
use crate::studio_store::TtsProjectStore;
use crate::transcription_store::TranscriptionStore;
use crate::voice_observation_store::VoiceObservationStore;
use crate::voice_store::VoiceStore;
use izwi_agent::planner::PlanningMode;
use izwi_core::{RuntimeService, ServeRuntimeConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};

const DEFAULT_RESPONSE_STORE_LIMIT: usize = 512;
const DEFAULT_AGENT_SESSION_STORE_LIMIT: usize = 512;

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
    /// Max retained OpenAI-compatible response objects in memory.
    response_store_limit: usize,
    /// Max retained agent session records in memory.
    agent_session_store_limit: usize,
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
    /// SQLite-backed onboarding completion store.
    pub onboarding_store: Arc<OnboardingStore>,
    /// SQLite-backed TTS project store.
    pub studio_store: Arc<TtsProjectStore>,
    /// SQLite-backed voice profile/session store.
    pub voice_store: Arc<VoiceStore>,
    /// SQLite-backed voice observation store.
    pub voice_observation_store: Arc<VoiceObservationStore>,
}

impl AppState {
    pub fn new(runtime: RuntimeService, serve_config: &ServeRuntimeConfig) -> anyhow::Result<Self> {
        let (max_concurrent_requests, request_timeout_secs) = request_limits(serve_config);
        let response_store_limit = store_limit_from_env(
            "IZWI_MAX_RESPONSE_STORE_ENTRIES",
            DEFAULT_RESPONSE_STORE_LIMIT,
        );
        let agent_session_store_limit = store_limit_from_env(
            "IZWI_MAX_AGENT_SESSION_STORE_ENTRIES",
            DEFAULT_AGENT_SESSION_STORE_LIMIT,
        );

        let chat_store = Arc::new(ChatStore::initialize()?);
        let transcription_store = Arc::new(TranscriptionStore::initialize()?);
        let diarization_store = Arc::new(DiarizationStore::initialize()?);
        let speech_history_store = Arc::new(SpeechHistoryStore::initialize()?);
        let saved_voice_store = Arc::new(SavedVoiceStore::initialize()?);
        let studio_store = Arc::new(TtsProjectStore::initialize()?);
        let voice_store = Arc::new(VoiceStore::initialize()?);
        let voice_observation_store = Arc::new(VoiceObservationStore::initialize()?);
        let onboarding_store = Arc::new(OnboardingStore::initialize()?);

        Ok(Self {
            runtime: Arc::new(runtime),
            request_semaphore: Arc::new(Semaphore::new(max_concurrent_requests)),
            request_timeout_secs,
            response_store_limit,
            agent_session_store_limit,
            response_store: Arc::new(RwLock::new(HashMap::new())),
            agent_session_store: Arc::new(RwLock::new(HashMap::new())),
            chat_store,
            transcription_store,
            diarization_store,
            speech_history_store,
            saved_voice_store,
            onboarding_store,
            studio_store,
            voice_store,
            voice_observation_store,
        })
    }

    /// Acquire a permit for concurrent request processing
    pub async fn acquire_permit(&self) -> tokio::sync::SemaphorePermit<'_> {
        self.request_semaphore
            .acquire()
            .await
            .expect("Semaphore should never be closed")
    }

    pub async fn store_response_record(&self, record: StoredResponseRecord) {
        let mut store = self.response_store.write().await;
        store.insert(record.id.clone(), record);
        trim_store_by(&mut store, self.response_store_limit, |record| {
            record.created_at
        });
    }

    pub async fn store_agent_session_record(&self, record: StoredAgentSessionRecord) {
        let mut store = self.agent_session_store.write().await;
        store.insert(record.id.clone(), record);
        trim_store_by(&mut store, self.agent_session_store_limit, |record| {
            record.updated_at
        });
    }

    pub async fn touch_agent_session_record(
        &self,
        session_id: &str,
        updated_at: u64,
        model_id: String,
    ) {
        let mut store = self.agent_session_store.write().await;
        if let Some(record) = store.get_mut(session_id) {
            record.updated_at = updated_at;
            record.model_id = model_id;
        }
    }
}

fn request_limits(serve_config: &ServeRuntimeConfig) -> (usize, u64) {
    (
        serve_config.max_concurrent_requests.max(1),
        serve_config.request_timeout_secs.max(1),
    )
}

fn store_limit_from_env(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn trim_store_by<T>(
    store: &mut HashMap<String, T>,
    max_entries: usize,
    timestamp_of: impl Fn(&T) -> u64,
) {
    while store.len() > max_entries {
        let Some(oldest_key) = store
            .iter()
            .min_by(|(left_key, left), (right_key, right)| {
                timestamp_of(left)
                    .cmp(&timestamp_of(right))
                    .then_with(|| left_key.cmp(right_key))
            })
            .map(|(key, _)| key.clone())
        else {
            break;
        };
        store.remove(&oldest_key);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        request_limits, trim_store_by, StoredAgentSessionRecord, StoredResponseInputItem,
        StoredResponseRecord,
    };
    use izwi_agent::planner::PlanningMode;
    use izwi_core::ServeRuntimeConfig;
    use std::collections::HashMap;

    #[test]
    fn trim_store_by_evicts_oldest_response_record() {
        let mut store = HashMap::new();
        store.insert(
            "resp-old".to_string(),
            StoredResponseRecord {
                id: "resp-old".to_string(),
                created_at: 10,
                status: "completed".to_string(),
                model: "test".to_string(),
                input_items: vec![StoredResponseInputItem {
                    role: "user".to_string(),
                    content: "old".to_string(),
                }],
                output_text: Some("old".to_string()),
                input_tokens: 1,
                output_tokens: 1,
                error: None,
                metadata: None,
            },
        );
        store.insert(
            "resp-new".to_string(),
            StoredResponseRecord {
                id: "resp-new".to_string(),
                created_at: 20,
                status: "completed".to_string(),
                model: "test".to_string(),
                input_items: Vec::new(),
                output_text: Some("new".to_string()),
                input_tokens: 1,
                output_tokens: 1,
                error: None,
                metadata: None,
            },
        );

        trim_store_by(&mut store, 1, |record| record.created_at);

        assert!(store.contains_key("resp-new"));
        assert!(!store.contains_key("resp-old"));
    }

    #[test]
    fn trim_store_by_uses_updated_at_for_agent_sessions() {
        let mut store = HashMap::new();
        store.insert(
            "sess-old".to_string(),
            StoredAgentSessionRecord {
                id: "sess-old".to_string(),
                agent_id: "agent".to_string(),
                thread_id: "thread-old".to_string(),
                model_id: "model".to_string(),
                system_prompt: "prompt".to_string(),
                planning_mode: PlanningMode::Auto,
                created_at: 1,
                updated_at: 5,
            },
        );
        store.insert(
            "sess-new".to_string(),
            StoredAgentSessionRecord {
                id: "sess-new".to_string(),
                agent_id: "agent".to_string(),
                thread_id: "thread-new".to_string(),
                model_id: "model".to_string(),
                system_prompt: "prompt".to_string(),
                planning_mode: PlanningMode::Auto,
                created_at: 2,
                updated_at: 15,
            },
        );

        trim_store_by(&mut store, 1, |record| record.updated_at);

        assert!(store.contains_key("sess-new"));
        assert!(!store.contains_key("sess-old"));
    }

    #[test]
    fn request_limits_use_serve_runtime_limits() {
        let serve_config = ServeRuntimeConfig {
            max_concurrent_requests: 7,
            request_timeout_secs: 91,
            ..ServeRuntimeConfig::default()
        };

        let (max_concurrent_requests, request_timeout_secs) = request_limits(&serve_config);

        assert_eq!(request_timeout_secs, 91);
        assert_eq!(max_concurrent_requests, 7);
    }
}
