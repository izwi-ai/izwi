//! Optional voice history for OpenAI Realtime sessions (D5). Off by default:
//! OpenAI's sessions are ephemeral, and a client that keeps its own history
//! (LiveKit, Pipecat) should not leave a second copy behind. With
//! `IZWI_REALTIME_PERSIST=1` each session and its default-conversation turns
//! are recorded in the voice store, where the desktop's voice history shows
//! them. Observational memory is never extracted from them.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tracing::warn;

use super::profiles::{ProfileMode, RealtimeProfile};
use crate::state::AppState;
use crate::voice_store::{CreateVoiceSessionRequest, CreateVoiceTurnRequest};

/// Whether `IZWI_REALTIME_PERSIST` turns the history on.
pub(crate) fn enabled() -> bool {
    std::env::var("IZWI_REALTIME_PERSIST").is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

/// One session's record in the voice store, created with its first turn.
pub(crate) struct History {
    state: AppState,
    profile: RealtimeProfile,
    system_prompt: String,
    session_id: tokio::sync::Mutex<Option<String>>,
    turns: AtomicU64,
}

impl History {
    pub(crate) fn new(
        state: AppState,
        profile: RealtimeProfile,
        system_prompt: String,
    ) -> Arc<Self> {
        Arc::new(Self {
            state,
            profile,
            system_prompt,
            session_id: tokio::sync::Mutex::new(None),
            turns: AtomicU64::new(0),
        })
    }

    fn mode(&self) -> &'static str {
        if self.profile.is_unified() {
            "unified"
        } else {
            "modular"
        }
    }

    async fn session(&self) -> anyhow::Result<String> {
        let mut session_id = self.session_id.lock().await;
        if let Some(id) = session_id.as_ref() {
            return Ok(id.clone());
        }
        let profile = self.state.voice_store.get_default_profile().await?;
        let session = self
            .state
            .voice_store
            .create_session(CreateVoiceSessionRequest {
                profile_id: profile.id,
                mode: self.mode().to_string(),
                system_prompt: self.system_prompt.clone(),
            })
            .await?;
        *session_id = Some(session.id.clone());
        Ok(session.id)
    }

    /// Record one turn in the background: what the user said and the reply.
    /// `status` is the response's (`completed`, `cancelled`, `failed`,
    /// `incomplete`).
    pub(crate) fn record_turn(
        self: &Arc<Self>,
        user_text: Option<String>,
        assistant_text: Option<String>,
        status: &'static str,
        speaker: Option<String>,
    ) {
        let history = self.clone();
        tokio::spawn(async move {
            if let Err(err) = history
                .write_turn(user_text, assistant_text, status, speaker)
                .await
            {
                warn!("realtime turn not recorded: {err:#}");
            }
        });
    }

    async fn write_turn(
        &self,
        user_text: Option<String>,
        assistant_text: Option<String>,
        status: &'static str,
        speaker: Option<String>,
    ) -> anyhow::Result<()> {
        let session_id = self.session().await?;
        let seq = self.turns.fetch_add(1, Ordering::Relaxed) + 1;
        let (asr, text, tts, s2s) = match &self.profile.mode {
            ProfileMode::Modular {
                asr_model_id,
                text_model_id,
                tts_model_id,
                ..
            } => (
                Some(asr_model_id.clone()),
                Some(text_model_id.clone()),
                Some(tts_model_id.clone()),
                None,
            ),
            ProfileMode::Unified { s2s_model_id, .. } => {
                (None, None, None, Some(s2s_model_id.clone()))
            }
        };
        let turn = self
            .state
            .voice_store
            .create_turn(CreateVoiceTurnRequest {
                turn_id: None,
                session_id,
                utterance_id: format!("openai-{seq}"),
                utterance_seq: seq,
                mode: self.mode().to_string(),
                vad_end_reason: None,
                asr_model_id: asr,
                text_model_id: text,
                tts_model_id: tts,
                s2s_model_id: s2s,
                speaker,
            })
            .await?;
        if let Some(user_text) = user_text {
            self.state
                .voice_store
                .update_turn_transcript(turn.id.clone(), user_text, None, None)
                .await?;
        }
        if assistant_text.is_some() {
            self.state
                .voice_store
                .update_turn_assistant(turn.id.clone(), assistant_text, None)
                .await?;
        }
        let (status, error) = match status {
            "completed" | "incomplete" => ("ok", None),
            "cancelled" => ("interrupted", Some("cancelled".to_string())),
            _ => ("error", Some("the response failed".to_string())),
        };
        self.state
            .voice_store
            .complete_turn(turn.id, status, error)
            .await?;
        Ok(())
    }

    /// End the session's record, if it has one.
    pub(crate) async fn close(&self) {
        if let Some(session_id) = self.session_id.lock().await.take() {
            if let Err(err) = self.state.voice_store.end_session(session_id).await {
                warn!("realtime voice session not closed in history: {err:#}");
            }
        }
    }
}
