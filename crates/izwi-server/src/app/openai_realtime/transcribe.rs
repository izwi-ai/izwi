//! Transcribing a committed user audio item. The modular stack's chat model
//! reads the transcript, and a unified model reads earlier turns as text, so
//! every user audio item is transcribed; the
//! `conversation.item.input_audio_transcription.*` events go out only when
//! the session turned transcription on. Transcription runs beside the
//! response, as OpenAI's does.

use std::sync::{Arc, Mutex};

use serde_json::json;
use tokio::sync::watch;

use super::items::ItemStore;
use super::respond::{lock_items, Outbound};
use crate::app::voice_realtime::{model_load_error, with_stage_permit};
use crate::state::AppState;
use izwi_core::SessionModelLeases;

/// One transcription to run.
pub(crate) struct Transcription {
    pub(crate) state: AppState,
    pub(crate) correlation_id: String,
    pub(crate) out: Outbound,
    pub(crate) items: Arc<Mutex<ItemStore>>,
    pub(crate) model_leases: SessionModelLeases,
    pub(crate) item_id: String,
    pub(crate) wav: Vec<u8>,
    pub(crate) model_id: String,
    pub(crate) language: Option<String>,
    pub(crate) prompt: Option<String>,
    /// Whether the client asked for transcription events.
    pub(crate) emit: bool,
    pub(crate) ready: watch::Sender<bool>,
    /// The session's LFM-Audio lane, when the transcribing model is the one
    /// that also replies.
    pub(crate) lane: Option<Arc<tokio::sync::Mutex<()>>>,
}

impl Transcription {
    pub(crate) fn spawn(self) {
        tokio::spawn(self.run());
    }

    async fn run(self) {
        let Self {
            state,
            correlation_id,
            out,
            items,
            model_leases,
            item_id,
            wav,
            model_id,
            language,
            prompt,
            emit,
            ready,
            lane,
        } = self;
        let _lane = match lane.as_ref() {
            Some(lane) => Some(lane.lock().await),
            None => None,
        };
        let variant = izwi_core::resolve_asr_model_variant(Some(&model_id));
        let delta_out = out.clone();
        let delta_item = item_id.clone();
        let result = with_stage_permit(&state, async |runtime_context| {
            model_leases
                .hold(&state.runtime, variant)
                .await
                .map_err(|err| model_load_error("ASR", err))?;
            state
                .runtime
                .asr_transcribe_streaming_bytes_with_runtime_context(
                    &wav,
                    Some(&model_id),
                    language.as_deref(),
                    prompt.as_deref(),
                    izwi_core::AsrTaskOptions::default(),
                    None,
                    Some(&correlation_id),
                    runtime_context,
                    move |delta| {
                        if emit && !delta.is_empty() {
                            delta_out.event(
                                "conversation.item.input_audio_transcription.delta",
                                json!({"item_id": delta_item, "content_index": 0, "delta": delta}),
                            );
                        }
                    },
                )
                .await
                .map_err(|err| format!("ASR failed: {err}"))
        })
        .await;
        match result {
            Ok(transcription) => {
                let text = transcription.text.trim().to_string();
                if let Some(item) = lock_items(&items).get_mut(&item_id) {
                    item.set_transcript(&text);
                }
                if emit {
                    out.event(
                        "conversation.item.input_audio_transcription.completed",
                        json!({
                            "item_id": item_id,
                            "content_index": 0,
                            "transcript": text,
                            "usage": {"type": "duration", "seconds": transcription.duration_secs},
                        }),
                    );
                }
            }
            Err(err) => {
                if let Some(item) = lock_items(&items).get_mut(&item_id) {
                    item.set_transcript("");
                }
                if emit {
                    out.event(
                        "conversation.item.input_audio_transcription.failed",
                        json!({
                            "item_id": item_id,
                            "content_index": 0,
                            "error": {"type": "transcription_error", "code": "transcription_failed",
                                      "message": err, "param": null},
                        }),
                    );
                }
            }
        }
        let _ = ready.send(true);
    }
}
