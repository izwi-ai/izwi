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
    /// Whether this pass sends the transcript's deltas. A turn whose live
    /// transcript already sent some sends only the outcome.
    pub(crate) emit_deltas: bool,
    pub(crate) ready: watch::Sender<bool>,
    /// The session's LFM-Audio lane, when the transcribing model is the one
    /// that also replies.
    pub(crate) lane: Option<Arc<tokio::sync::Mutex<()>>>,
}

impl Transcription {
    pub(crate) fn spawn(self) {
        tokio::spawn(self.run());
    }

    pub(crate) async fn run(self) {
        let lane = self.lane.clone();
        let _lane = match lane.as_ref() {
            Some(lane) => Some(lane.lock().await),
            None => None,
        };
        let variant = izwi_core::resolve_asr_model_variant(Some(&self.model_id));
        let delta_out = self.out.clone();
        let delta_item = self.item_id.clone();
        let emit_deltas = self.emit && self.emit_deltas;
        let result = with_stage_permit(&self.state, async |runtime_context| {
            self.model_leases
                .hold(&self.state.runtime, variant)
                .await
                .map_err(|err| model_load_error("ASR", err))?;
            self.state
                .runtime
                .asr_transcribe_streaming_bytes_with_runtime_context(
                    &self.wav,
                    Some(&self.model_id),
                    self.language.as_deref(),
                    self.prompt.as_deref(),
                    izwi_core::AsrTaskOptions::default(),
                    None,
                    Some(&self.correlation_id),
                    runtime_context,
                    move |delta| {
                        if emit_deltas && !delta.is_empty() {
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
            Ok(transcription) => self.complete(&transcription.text, transcription.duration_secs),
            Err(err) => {
                if let Some(item) = lock_items(&self.items).get_mut(&self.item_id) {
                    item.set_transcript("");
                }
                if self.emit {
                    self.out.event(
                        "conversation.item.input_audio_transcription.failed",
                        json!({
                            "item_id": self.item_id,
                            "content_index": 0,
                            "error": {"type": "transcription_error", "code": "transcription_failed",
                                      "message": err, "param": null},
                        }),
                    );
                }
                let _ = self.ready.send(true);
            }
        }
    }

    /// The item's transcript is `text`: it is stored, announced when the
    /// client asked for transcription events, and whoever waits for it
    /// (the reply) goes on.
    pub(crate) fn complete(self, text: &str, duration_secs: f32) {
        let text = text.trim();
        if let Some(item) = lock_items(&self.items).get_mut(&self.item_id) {
            item.set_transcript(text);
        }
        if self.emit {
            self.out.event(
                "conversation.item.input_audio_transcription.completed",
                json!({
                    "item_id": self.item_id,
                    "content_index": 0,
                    "transcript": text,
                    "usage": {"type": "duration", "seconds": duration_secs},
                }),
            );
        }
        let _ = self.ready.send(true);
    }
}
