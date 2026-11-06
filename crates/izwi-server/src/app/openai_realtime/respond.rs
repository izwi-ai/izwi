//! One response: a model turn over the conversation (or an out-of-band
//! input), streamed as OpenAI Realtime events.
//!
//! Event order per response follows what LiveKit and Pipecat rely on:
//! `response.created` (sent by the session before this runs), then for the
//! assistant message `response.output_item.added` → `conversation.item.added`
//! → `response.content_part.added` → transcript and audio deltas →
//! `.done` events → `response.output_item.done` → `conversation.item.done`;
//! then any function calls (`output_item.added`, `conversation.item.added`,
//! `function_call_arguments.delta/done`, `output_item.done`); then
//! `response.done`. Cancellation still closes every open part.
//!
//! The modular stack reuses the voice socket's `speak_reply` (sentence
//! segments, one segment of TTS lookahead, the spoken ledger) through
//! [`OpenAiReplySink`]; the unified stack streams an LFM-Audio round. A
//! response that calls functions ends there: the client answers with
//! `function_call_output` items and a new `response.create`.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::ws::Message;
use izwi_agent::ReasoningFilter;
use izwi_core::{
    ChatFinishReason, ChatMessage, ChatRequestConfig, ChatRole, ChatTruncation, GenerationParams,
    ModelFamily, RuntimeService, SessionModelLeases, CHAT_AUDIO_PLACEHOLDER,
};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tracing::warn;

use super::audio::{base64, wav, OutputEncoder};
use super::events::{server_event, Ids};
use super::items::{ContentPart, Item, ItemKind, ItemStatus, ItemStore, Position, Role};
use super::profiles::{ProfileMode, RealtimeProfile};
use super::session_config::{chat_tool, AudioFormat, Modality};
use crate::api::openai::convert::InboundContent;
use crate::api::openai::responses::dto::{ResponseInput, ResponseInputItem};
use crate::api::openai::responses::handlers::build_input_messages;
use crate::api::openai::tool_calls::{parse_answer, ToolCall, ToolCallFormat, ToolCallParser};
use crate::app::realtime_transport::{
    OutboundClass, OutboundEnqueueError, OutboundItem, OutboundMailbox,
};
use crate::app::voice_realtime::{
    acquire_stage_permit, model_load_error, speak_reply, with_stage_permit, ReplySink,
    SegmentSynthesis, TtsVoice, TurnControl,
};
use crate::state::AppState;

/// A reply's token budget when the client leaves it at `"inf"`.
const DEFAULT_MODULAR_MAX_TOKENS: usize = 1_024;
const DEFAULT_UNIFIED_MAX_TOKENS: usize = 1_536;
/// How long a response waits for a user turn's transcript.
const TRANSCRIPT_WAIT: Duration = Duration::from_secs(60);
/// The silence a unified model hears when a response has no spoken input
/// (a text turn, a greeting): LFM-Audio always listens to a clip.
const SILENT_INPUT_MS: u32 = 200;

/// The session's outbound queue: JSON events, audio with credit backpressure.
#[derive(Clone)]
pub(crate) struct Outbound {
    pub(crate) mailbox: Arc<OutboundMailbox>,
    pub(crate) ids: Arc<Ids>,
}

impl Outbound {
    /// Queue a server event; `false` once the session is closing.
    pub(crate) fn event(&self, event_type: &str, fields: Value) -> bool {
        self.value(server_event(&self.ids, event_type, fields))
    }

    pub(crate) fn value(&self, value: Value) -> bool {
        self.mailbox
            .enqueue(OutboundItem {
                message: Message::Text(value.to_string().into()).into(),
                class: OutboundClass::Critical,
                turn_generation: None,
            })
            .is_ok()
    }

    /// Queue an audio event once the client has taken enough of the audio
    /// already queued. `generation` lets a cancelled response's queued audio
    /// be dropped at once.
    pub(crate) async fn audio(
        &self,
        generation: Option<u64>,
        value: Value,
        bytes: usize,
        duration_ms: u64,
    ) -> Result<(), OutboundEnqueueError> {
        self.mailbox
            .enqueue_audio(OutboundItem {
                message: Message::Text(value.to_string().into()).into(),
                class: OutboundClass::Audio { bytes, duration_ms },
                turn_generation: generation,
            })
            .await
    }
}

/// Token counts for `response.done.usage`. Text tokens are the chat model's
/// own (a speech-to-speech model's output text counts its streamed text
/// steps, one token each); audio tokens are estimated at OpenAI's documented
/// rates (user audio 1 token per 100 ms, assistant audio 1 per 50 ms), since
/// the modular stack has no audio tokens (D7).
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct Usage {
    pub(crate) input_text_tokens: u64,
    pub(crate) cached_tokens: u64,
    pub(crate) output_text_tokens: u64,
    pub(crate) input_audio_ms: u64,
    pub(crate) output_audio_ms: u64,
}

impl Usage {
    pub(crate) fn to_json(&self) -> Value {
        let input_audio = self.input_audio_ms / 100;
        let output_audio = self.output_audio_ms / 50;
        let input = self.input_text_tokens + input_audio;
        let output = self.output_text_tokens + output_audio;
        json!({
            "total_tokens": input + output,
            "input_tokens": input,
            "output_tokens": output,
            "input_token_details": {
                "text_tokens": self.input_text_tokens,
                "audio_tokens": input_audio,
                "image_tokens": 0,
                "cached_tokens": self.cached_tokens,
                "cached_tokens_details": {"text_tokens": self.cached_tokens, "audio_tokens": 0, "image_tokens": 0},
            },
            "output_token_details": {"text_tokens": self.output_text_tokens, "audio_tokens": output_audio},
        })
    }
}

/// What a finished response tells its session.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ResponseSummary {
    pub(crate) spoke: bool,
    /// `completed`, `cancelled`, `failed` or `incomplete`.
    pub(crate) status: &'static str,
    /// The reply's text (its audio transcript), if it said anything.
    pub(crate) text: Option<String>,
}

/// How a response ended.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Outcome {
    Completed,
    /// Out of output tokens.
    Incomplete,
    /// `turn_detected` (barge-in) or `client_cancelled`.
    Cancelled(String),
    Failed(String),
}

/// What a response is asked to do.
#[derive(Debug, Clone)]
pub(crate) struct ResponsePlan {
    pub(crate) id: String,
    pub(crate) conversation_id: String,
    /// Whether it writes to the default conversation (`false`:
    /// `conversation: "none"`).
    pub(crate) default_conversation: bool,
    /// Its input when not the conversation (`input` items, references
    /// resolved).
    pub(crate) input: Option<Vec<Item>>,
    pub(crate) instructions: String,
    pub(crate) modality: Modality,
    pub(crate) tools: Vec<Value>,
    pub(crate) tool_choice: Value,
    pub(crate) max_output_tokens: Option<usize>,
    pub(crate) voice: TtsVoice,
    /// The voice and format echoed in the response object.
    pub(crate) voice_echo: Value,
    pub(crate) output_format: AudioFormat,
    pub(crate) metadata: Value,
    /// The cut-off generation of its audio (default-conversation responses).
    pub(crate) audio_generation: Option<u64>,
}

impl ResponsePlan {
    /// The response object of `response.created` and `response.done`.
    pub(crate) fn response_json(
        &self,
        status: &str,
        status_details: Value,
        output: Vec<Value>,
        usage: Value,
    ) -> Value {
        json!({
            "id": self.id,
            "object": "realtime.response",
            "status": status,
            "status_details": status_details,
            "output": output,
            "conversation_id": if self.default_conversation { json!(self.conversation_id) } else { Value::Null },
            "output_modalities": self.modality.to_json(),
            "max_output_tokens": self.max_output_tokens.map_or_else(|| json!("inf"), |tokens| json!(tokens)),
            "audio": {"output": {"format": self.output_format.to_json(), "voice": self.voice_echo}},
            "usage": usage,
            "metadata": self.metadata,
        })
    }
}

/// What a response works with.
#[derive(Clone)]
pub(crate) struct ResponseContext {
    pub(crate) state: AppState,
    pub(crate) correlation_id: String,
    pub(crate) out: Outbound,
    pub(crate) items: Arc<Mutex<ItemStore>>,
    pub(crate) profile: RealtimeProfile,
    pub(crate) model_leases: SessionModelLeases,
    /// User turns whose transcript the next unified response writes.
    pub(crate) pending_transcripts: PendingTranscripts,
    /// One LFM-Audio request at a time per session: a separate LFM3
    /// transcription beside a reply exceeds what Metal admits.
    pub(crate) lane: Arc<tokio::sync::Mutex<()>>,
}

/// A user turn's transcription left to the response that hears it: LFM3-Audio
/// transcribes the turn inside its reply round, as the voice socket does,
/// rather than as a second request beside it.
pub(crate) struct PendingTranscript {
    pub(crate) ready: tokio::sync::watch::Sender<bool>,
    /// Whether the client asked for transcription events.
    pub(crate) emit: bool,
    pub(crate) audio_secs: f32,
}

pub(crate) type PendingTranscripts =
    Arc<Mutex<std::collections::HashMap<String, PendingTranscript>>>;

/// A response's claim on one pending transcript; dropped unfinished (the
/// response failed or was cancelled), it goes back for the session to run.
struct TranscriptClaim {
    pending: PendingTranscripts,
    entry: Option<(String, PendingTranscript)>,
}

impl TranscriptClaim {
    fn take(pending: &PendingTranscripts, item_id: &str) -> Option<Self> {
        let entry = pending
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .remove(item_id)?;
        Some(Self {
            pending: pending.clone(),
            entry: Some((item_id.to_string(), entry)),
        })
    }

    fn finish(mut self) -> Option<(String, PendingTranscript)> {
        self.entry.take()
    }
}

impl Drop for TranscriptClaim {
    fn drop(&mut self) {
        if let Some((item_id, entry)) = self.entry.take() {
            self.pending
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .insert(item_id, entry);
        }
    }
}

pub(crate) fn lock_items(items: &Mutex<ItemStore>) -> std::sync::MutexGuard<'_, ItemStore> {
    items.lock().unwrap_or_else(|poison| poison.into_inner())
}

struct MessageState {
    item_id: String,
    output_index: usize,
    transcript: String,
    audio_ms: u64,
}

#[derive(Default)]
struct EmitterState {
    message: Option<MessageState>,
    next_output_index: usize,
    outputs: Vec<(usize, Value)>,
    usage: Usage,
}

/// Emits one response's output events and records its items.
#[derive(Clone)]
pub(crate) struct Emitter {
    out: Outbound,
    items: Arc<Mutex<ItemStore>>,
    plan: Arc<ResponsePlan>,
    control: TurnControl,
    runtime: Arc<RuntimeService>,
    state: Arc<Mutex<EmitterState>>,
}

impl Emitter {
    pub(crate) fn new(
        ctx: &ResponseContext,
        plan: Arc<ResponsePlan>,
        control: TurnControl,
    ) -> Self {
        Self {
            out: ctx.out.clone(),
            items: ctx.items.clone(),
            plan,
            control,
            runtime: ctx.state.runtime.clone(),
            state: Arc::default(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, EmitterState> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn audio_out(&self) -> bool {
        self.plan.modality == Modality::Audio
    }

    fn add_usage(&self, add: impl FnOnce(&mut Usage)) {
        add(&mut self.lock().usage);
    }

    /// Insert `item` at the end of the conversation and announce it.
    fn add_to_conversation(&self, item: Item) {
        let json = item.to_json();
        let previous = match lock_items(&self.items).insert(item, &Position::End) {
            Ok(previous) => previous,
            Err(err) => {
                warn!("realtime response item not stored: {}", err.message);
                return;
            }
        };
        self.out.event(
            "conversation.item.added",
            json!({"previous_item_id": previous, "item": json}),
        );
    }

    /// Store `item`'s final state and announce it done.
    fn finish_in_conversation(&self, item: Item) {
        let json = item.to_json();
        let previous = {
            let mut items = lock_items(&self.items);
            let ids: Vec<String> = items
                .items()
                .iter()
                .map(|stored| stored.id.clone())
                .collect();
            let previous = ids
                .iter()
                .position(|id| *id == item.id)
                .and_then(|index| index.checked_sub(1))
                .map(|index| ids[index].clone());
            if let Some(stored) = items.get_mut(&item.id) {
                *stored = item;
            }
            previous
        };
        self.out.event(
            "conversation.item.done",
            json!({"previous_item_id": previous, "item": json}),
        );
    }

    /// Open the assistant message, once: its output item, conversation item
    /// and content part.
    fn open_message(&self) -> (String, usize) {
        let mut state = self.lock();
        if let Some(message) = state.message.as_ref() {
            return (message.item_id.clone(), message.output_index);
        }
        let item_id = self.out.ids.next("item");
        let output_index = state.next_output_index;
        state.next_output_index += 1;
        state.message = Some(MessageState {
            item_id: item_id.clone(),
            output_index,
            transcript: String::new(),
            audio_ms: 0,
        });
        let mut item = Item::new(
            item_id.clone(),
            ItemKind::Message {
                role: Role::Assistant,
                content: Vec::new(),
            },
            ItemStatus::InProgress,
        );
        item.reply = Some(self.control.clone());
        self.out.event(
            "response.output_item.added",
            json!({"response_id": self.plan.id, "output_index": output_index, "item": item.to_json()}),
        );
        if self.plan.default_conversation {
            self.add_to_conversation(item);
        }
        let part = if self.audio_out() {
            json!({"type": "audio", "transcript": ""})
        } else {
            json!({"type": "text", "text": ""})
        };
        self.out.event(
            "response.content_part.added",
            json!({"response_id": self.plan.id, "item_id": item_id, "output_index": output_index,
                   "content_index": 0, "part": part}),
        );
        (item_id, output_index)
    }

    /// More of the reply's text: its audio transcript, or its text output.
    pub(crate) fn text_delta(&self, delta: &str) {
        if delta.is_empty() {
            return;
        }
        let (item_id, output_index) = self.open_message();
        if let Some(message) = self.lock().message.as_mut() {
            message.transcript.push_str(delta);
        }
        let event_type = if self.audio_out() {
            "response.output_audio_transcript.delta"
        } else {
            "response.output_text.delta"
        };
        self.out.event(
            event_type,
            json!({"response_id": self.plan.id, "item_id": item_id, "output_index": output_index,
                   "content_index": 0, "delta": delta}),
        );
    }

    /// One encoded audio delta. `account` records it on the response's
    /// playback clock (the unified path; `speak_reply` does it itself).
    pub(crate) async fn audio_delta(
        &self,
        bytes: Vec<u8>,
        duration_ms: u64,
        account: bool,
    ) -> Result<(), String> {
        let (item_id, output_index) = self.open_message();
        let event = server_event(
            &self.out.ids,
            "response.output_audio.delta",
            json!({"response_id": self.plan.id, "item_id": item_id, "output_index": output_index,
                   "content_index": 0, "delta": base64(&bytes)}),
        );
        self.out
            .audio(self.plan.audio_generation, event, bytes.len(), duration_ms)
            .await
            .map_err(|err| format!("Outbound audio delivery failed: {err:?}"))?;
        {
            let mut state = self.lock();
            if let Some(message) = state.message.as_mut() {
                message.audio_ms += duration_ms;
            }
            state.usage.output_audio_ms += duration_ms;
        }
        if account {
            self.control
                .audio_sent(&self.runtime, Duration::from_millis(duration_ms));
        }
        Ok(())
    }

    /// Close the assistant message, if open: its `.done` events and final
    /// item. A message the client truncated keeps the heard transcript.
    fn close_message(&self, status: ItemStatus) {
        let Some(message) = self.lock().message.take() else {
            return;
        };
        let truncated = self.control.reply().heard_ms.is_some();
        let transcript = if truncated {
            lock_items(&self.items)
                .get(&message.item_id)
                .and_then(Item::text)
                .unwrap_or_default()
        } else {
            message.transcript.trim().to_string()
        };
        let ids = json!({"response_id": self.plan.id, "item_id": message.item_id,
                         "output_index": message.output_index, "content_index": 0});
        let with = |extra: Value| {
            let mut fields = ids.clone();
            if let (Value::Object(fields), Value::Object(extra)) = (&mut fields, extra) {
                fields.extend(extra);
            }
            fields
        };
        let (part, content) = if self.audio_out() {
            self.out.event("response.output_audio.done", ids.clone());
            self.out.event(
                "response.output_audio_transcript.done",
                with(json!({"transcript": transcript})),
            );
            (
                json!({"type": "audio", "transcript": transcript}),
                ContentPart::OutputAudio {
                    transcript: transcript.clone(),
                },
            )
        } else {
            self.out.event(
                "response.output_text.done",
                with(json!({"text": transcript})),
            );
            (
                json!({"type": "text", "text": transcript}),
                ContentPart::OutputText(transcript.clone()),
            )
        };
        self.out
            .event("response.content_part.done", with(json!({"part": part})));
        let mut item = Item::new(
            message.item_id.clone(),
            ItemKind::Message {
                role: Role::Assistant,
                content: vec![content],
            },
            status,
        );
        item.reply = Some(self.control.clone());
        item.audio_ms = message.audio_ms;
        let json = item.to_json();
        self.out.event(
            "response.output_item.done",
            json!({"response_id": self.plan.id, "output_index": message.output_index, "item": json}),
        );
        if self.plan.default_conversation {
            self.finish_in_conversation(item);
        }
        self.lock().outputs.push((message.output_index, json));
    }

    /// The function calls a reply made, after its message.
    pub(crate) fn function_calls(&self, calls: &[ToolCall]) {
        if calls.is_empty() {
            return;
        }
        self.close_message(ItemStatus::Completed);
        for call in calls {
            let item_id = self.out.ids.next("item");
            let call_id = if call.id.trim().is_empty() {
                self.out.ids.next("call")
            } else {
                call.id.clone()
            };
            let output_index = {
                let mut state = self.lock();
                state.next_output_index += 1;
                state.next_output_index - 1
            };
            let item = |arguments: &str, status: ItemStatus| {
                Item::new(
                    item_id.clone(),
                    ItemKind::FunctionCall {
                        call_id: call_id.clone(),
                        name: call.function.name.clone(),
                        arguments: arguments.to_string(),
                    },
                    status,
                )
            };
            let started = item("", ItemStatus::InProgress);
            self.out.event(
                "response.output_item.added",
                json!({"response_id": self.plan.id, "output_index": output_index, "item": started.to_json()}),
            );
            if self.plan.default_conversation {
                self.add_to_conversation(started);
            }
            let ids = json!({"response_id": self.plan.id, "item_id": item_id,
                             "output_index": output_index, "call_id": call_id});
            let mut delta = ids.clone();
            delta["delta"] = json!(call.function.arguments);
            self.out
                .event("response.function_call_arguments.delta", delta);
            let mut done = ids;
            done["name"] = json!(call.function.name);
            done["arguments"] = json!(call.function.arguments);
            self.out
                .event("response.function_call_arguments.done", done);
            let finished = item(&call.function.arguments, ItemStatus::Completed);
            let json = finished.to_json();
            self.out.event(
                "response.output_item.done",
                json!({"response_id": self.plan.id, "output_index": output_index, "item": json}),
            );
            if self.plan.default_conversation {
                self.finish_in_conversation(finished);
            }
            self.lock().outputs.push((output_index, json));
        }
    }

    /// Close what is open and send `response.done`.
    pub(crate) fn finish(&self, outcome: &Outcome) -> ResponseSummary {
        let item_status = match outcome {
            Outcome::Completed => ItemStatus::Completed,
            _ => ItemStatus::Incomplete,
        };
        self.close_message(item_status);
        let (status, details) = match outcome {
            Outcome::Completed => ("completed", Value::Null),
            Outcome::Incomplete => (
                "incomplete",
                json!({"type": "incomplete", "reason": "max_output_tokens"}),
            ),
            Outcome::Cancelled(reason) => {
                ("cancelled", json!({"type": "cancelled", "reason": reason}))
            }
            Outcome::Failed(message) => (
                "failed",
                json!({"type": "failed", "error": {"type": "server_error", "code": "response_failed", "message": message}}),
            ),
        };
        let (output, usage, spoke) = {
            let mut state = self.lock();
            state.outputs.sort_by_key(|(index, _)| *index);
            (
                state
                    .outputs
                    .iter()
                    .map(|(_, item)| item.clone())
                    .collect::<Vec<_>>(),
                state.usage.clone(),
                state.usage.output_audio_ms > 0,
            )
        };
        let text = output
            .iter()
            .filter(|item| item["type"] == "message")
            .filter_map(|item| {
                let part = &item["content"][0];
                part.get("transcript")
                    .or_else(|| part.get("text"))
                    .and_then(Value::as_str)
            })
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        self.out.event(
            "response.done",
            json!({"response": self.plan.response_json(status, details, output, usage.to_json())}),
        );
        ResponseSummary {
            spoke,
            status,
            text: (!text.is_empty()).then_some(text),
        }
    }
}

/// `speak_reply`'s sink for OpenAI Realtime: speech as base64 deltas in the
/// session's output format, the visible reply text as transcript deltas.
pub(crate) struct OpenAiReplySink {
    emitter: Emitter,
    format: AudioFormat,
    encoder: Option<OutputEncoder>,
    sent_text: String,
}

impl OpenAiReplySink {
    pub(crate) fn new(emitter: Emitter, format: AudioFormat) -> Self {
        Self {
            emitter,
            format,
            encoder: None,
            sent_text: String::new(),
        }
    }
}

impl ReplySink for OpenAiReplySink {
    fn speech_started(&mut self, model_rate: u32) -> Result<(), String> {
        self.encoder = Some(OutputEncoder::new(self.format, model_rate)?);
        Ok(())
    }

    async fn speech(
        &mut self,
        samples: &[f32],
        _model_rate: u32,
        is_final: bool,
    ) -> Result<bool, String> {
        let Some(encoder) = self.encoder.as_mut() else {
            return Ok(false);
        };
        let deltas: Vec<(Vec<u8>, u64)> = encoder
            .push(samples, is_final)?
            .into_iter()
            .map(|delta| {
                let ms = encoder.duration_ms(delta.len());
                (delta, ms)
            })
            .collect();
        for (delta, ms) in deltas {
            self.emitter.audio_delta(delta, ms, false).await?;
        }
        Ok(!samples.is_empty())
    }

    fn text(&mut self, visible: &str) {
        if let Some(rest) = visible.strip_prefix(self.sent_text.as_str()) {
            if !rest.is_empty() {
                self.emitter.text_delta(rest);
                self.sent_text = visible.to_string();
            }
        }
    }

    fn speech_done(&mut self) {}
}

/// Wait until every user audio item in `items` has its transcript, then
/// take the items again with their transcripts.
async fn with_transcripts(
    ctx: &ResponseContext,
    items: Vec<Item>,
    skip: Option<&str>,
) -> Vec<Item> {
    for item in &items {
        if Some(item.id.as_str()) == skip || !item.awaits_transcript() {
            continue;
        }
        if let Some(mut ready) = item.transcript_ready.clone() {
            if tokio::time::timeout(TRANSCRIPT_WAIT, ready.wait_for(|ready| *ready))
                .await
                .is_err()
            {
                warn!(item_id = %item.id, "realtime transcript did not arrive in time");
            }
        }
    }
    let store = lock_items(&ctx.items);
    items
        .into_iter()
        .map(|item| store.get(&item.id).cloned().unwrap_or(item))
        .collect()
}

/// The items a response reads: its `input`, or the conversation so far.
fn prompt_items(ctx: &ResponseContext, plan: &ResponsePlan) -> Vec<Item> {
    match &plan.input {
        Some(items) => items.clone(),
        None => lock_items(&ctx.items).items().to_vec(),
    }
}

/// The system prompt: the instructions, then the conversation's system
/// messages (chat templates take one system turn, at the start), then a
/// hint for a required or named tool, which local models cannot be forced
/// to call (D16).
fn system_prompt(plan: &ResponsePlan, items: &[Item], tools_offered: bool) -> String {
    let mut parts: Vec<String> = Vec::new();
    if !plan.instructions.trim().is_empty() {
        parts.push(plan.instructions.trim().to_string());
    }
    parts.extend(
        items
            .iter()
            .filter(|item| item.role() == Some(Role::System))
            .filter_map(Item::text),
    );
    if tools_offered {
        match &plan.tool_choice {
            Value::String(choice) if choice == "required" => {
                parts.push("Answer by calling one of the available functions.".to_string())
            }
            Value::Object(choice) => {
                if let Some(name) = choice
                    .get("name")
                    .or_else(|| {
                        choice
                            .get("function")
                            .and_then(|function| function.get("name"))
                    })
                    .and_then(Value::as_str)
                {
                    parts.push(format!("Answer by calling the function `{name}`."));
                }
            }
            _ => {}
        }
    }
    parts.join("\n\n")
}

fn tools_for(plan: &ResponsePlan) -> Vec<Value> {
    if plan.tool_choice.as_str() == Some("none") {
        return Vec::new();
    }
    plan.tools.iter().map(chat_tool).collect()
}

fn user_audio_ms(items: &[Item]) -> u64 {
    items
        .iter()
        .filter(|item| item.role() == Some(Role::User))
        .filter_map(|item| item.audio.as_ref())
        .map(|audio| audio.samples.len() as u64 * 1_000 / u64::from(audio.sample_rate.max(1)))
        .sum()
}

/// The conversation as chat messages, through the Responses API's item
/// conversion (tool calls and results rendered as the model's template
/// expects).
fn chat_messages(
    variant: izwi_core::ModelVariant,
    system: &str,
    items: &[Item],
    tools: &[Value],
) -> Result<Vec<ChatMessage>, String> {
    let input: Vec<ResponseInputItem> = items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Message {
                role: Role::System, ..
            } => None,
            ItemKind::Message { role, .. } => item.text().map(|text| ResponseInputItem {
                role: Some(role.as_str().to_string()),
                content: Some(InboundContent::Text(text)),
                ..ResponseInputItem::default()
            }),
            ItemKind::FunctionCall {
                call_id,
                name,
                arguments,
            } => Some(ResponseInputItem {
                item_type: Some("function_call".to_string()),
                call_id: Some(call_id.clone()),
                name: Some(name.clone()),
                arguments: Some(Value::String(arguments.clone())),
                ..ResponseInputItem::default()
            }),
            ItemKind::FunctionCallOutput { call_id, output } => Some(ResponseInputItem {
                item_type: Some("function_call_output".to_string()),
                call_id: Some(call_id.clone()),
                output: Some(Value::String(output.clone())),
                ..ResponseInputItem::default()
            }),
        })
        .collect();
    let mut input = input;
    // A response with no user turn yet (a greeting asked for with
    // `response.create` instructions) still needs one: chat templates
    // refuse a prompt without a user query.
    if !input
        .iter()
        .any(|item| item.role.as_deref() == Some("user") && item.item_type.is_none())
    {
        input.insert(
            0,
            ResponseInputItem {
                role: Some("user".to_string()),
                content: Some(InboundContent::Text(OPENING_TURN.to_string())),
                ..ResponseInputItem::default()
            },
        );
    }
    let system = Some(system).filter(|system| !system.trim().is_empty());
    build_input_messages(variant, system, Some(ResponseInput::Many(input)), tools)
        .map(|(messages, _, _)| messages)
        .map_err(|err| err.message)
}

/// The user turn of a conversation that has none yet.
const OPENING_TURN: &str = "(The user has joined the conversation.)";

/// Run one response to its end and send `response.done`.
pub(crate) async fn run_response(
    ctx: ResponseContext,
    plan: Arc<ResponsePlan>,
    control: TurnControl,
) -> ResponseSummary {
    let emitter = Emitter::new(&ctx, plan.clone(), control.clone());
    let timeout = Duration::from_secs(ctx.state.request_timeout_secs.max(1));
    let work = async {
        match &ctx.profile.mode {
            ProfileMode::Modular { .. } => run_modular(&ctx, &plan, &emitter, &control).await,
            ProfileMode::Unified { .. } => run_unified(&ctx, &plan, &emitter, &control).await,
        }
    };
    let outcome = tokio::select! {
        biased;
        _ = control.cancelled() => Outcome::Cancelled(control.reason().unwrap_or_else(|| "client_cancelled".to_string())),
        result = tokio::time::timeout(timeout, work) => match result {
            Ok(Ok(outcome)) => outcome,
            Ok(Err(err)) => Outcome::Failed(err),
            Err(_) => Outcome::Failed(format!("The response timed out after {} seconds", timeout.as_secs())),
        },
    };
    emitter.finish(&outcome)
}

/// A modular response: the chat model writes, `speak_reply` speaks.
async fn run_modular(
    ctx: &ResponseContext,
    plan: &ResponsePlan,
    emitter: &Emitter,
    control: &TurnControl,
) -> Result<Outcome, String> {
    let ProfileMode::Modular { text, tts, .. } = &ctx.profile.mode else {
        return Err("Not a modular profile".to_string());
    };
    let (text, tts) = (*text, *tts);
    let items = with_transcripts(ctx, prompt_items(ctx, plan), None).await;
    let format = ToolCallFormat::of(text);
    let tools = if format.is_some() {
        tools_for(plan)
    } else {
        Vec::new()
    };
    if format.is_none() && !plan.tools.is_empty() {
        warn!(model = %text.dir_name(), "realtime tools ignored: the chat model cannot call tools");
    }
    let system = system_prompt(plan, &items, !tools.is_empty());
    let messages = chat_messages(text, &system, &items, &tools)?;
    emitter.add_usage(|usage| usage.input_audio_ms += user_audio_ms(&items));

    let audio_out = plan.modality == Modality::Audio;
    let parser = Arc::new(Mutex::new(
        format
            .filter(|_| !tools.is_empty())
            .map(ToolCallParser::new),
    ));
    let reasoning = Arc::new(Mutex::new(ReasoningFilter::default()));
    let (deltas, delta_rx) = mpsc::unbounded_channel::<String>();
    let state = &ctx.state;
    let max_tokens = plan
        .max_output_tokens
        .unwrap_or(DEFAULT_MODULAR_MAX_TOKENS)
        .clamp(1, 8_192);

    let chat = async {
        let parser_sink = parser.clone();
        let reasoning_sink = reasoning.clone();
        let emitter_sink = emitter.clone();
        let deltas_sink = deltas.clone();
        let generation = with_stage_permit(state, async |runtime_context| {
            ctx.model_leases
                .hold(&state.runtime, text)
                .await
                .map_err(|err| model_load_error("chat", err))?;
            state
                .runtime
                .chat_generate_streaming_with_runtime_context(
                    text,
                    messages,
                    GenerationParams {
                        max_tokens,
                        ..Default::default()
                    },
                    ChatRequestConfig {
                        enable_thinking: Some(false),
                        tools: tools.clone(),
                        truncation: ChatTruncation::Auto,
                        ..ChatRequestConfig::default()
                    },
                    Some(&ctx.correlation_id),
                    runtime_context,
                    move |delta| {
                        let visible = match parser_sink
                            .lock()
                            .unwrap_or_else(|poison| poison.into_inner())
                            .as_mut()
                        {
                            Some(parser) => parser.push(&delta).content,
                            None => delta,
                        };
                        if visible.is_empty() {
                            return;
                        }
                        if audio_out {
                            let _ = deltas_sink.send(visible);
                        } else {
                            let text = reasoning_sink
                                .lock()
                                .unwrap_or_else(|poison| poison.into_inner())
                                .push(&visible);
                            emitter_sink.text_delta(&text);
                        }
                    },
                )
                .await
                .map_err(|err| format!("Chat failed: {err}"))
        })
        .await?;
        // The parser holds back text that could still open a call.
        let tail = parser
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take()
            .map(|mut parser| parser.finish().content)
            .unwrap_or_default();
        if audio_out {
            if !tail.is_empty() {
                let _ = deltas.send(tail);
            }
        } else {
            let mut reasoning = reasoning
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            let mut text = reasoning.push(&tail);
            text.push_str(&reasoning.finish());
            drop(reasoning);
            emitter.text_delta(&text);
        }
        drop(deltas);
        Ok::<_, String>(generation)
    };
    let speech = async {
        if !audio_out {
            return Ok(());
        }
        let mut sink = OpenAiReplySink::new(emitter.clone(), plan.output_format);
        speak_reply(
            state,
            &mut sink,
            control,
            delta_rx,
            Instant::now(),
            |text| {
                SegmentSynthesis::start(
                    state,
                    &ctx.correlation_id,
                    tts,
                    plan.voice.clone(),
                    &ctx.model_leases,
                    text,
                )
            },
        )
        .await
    };
    let (generation, ()) = tokio::try_join!(chat, speech)?;
    emitter.add_usage(|usage| {
        usage.input_text_tokens += generation.prompt_tokens as u64;
        usage.cached_tokens += generation.cached_prompt_tokens.unwrap_or(0);
        usage.output_text_tokens += generation.tokens_generated as u64;
    });
    if let Some(format) = format.filter(|_| !tools.is_empty()) {
        let (_, calls) = parse_answer(format, &generation.text);
        emitter.function_calls(&calls);
    }
    Ok(
        if generation.finish_reason == Some(ChatFinishReason::MaxTokens) {
            Outcome::Incomplete
        } else {
            Outcome::Completed
        },
    )
}

/// A function call as LFM3-Audio's template writes one.
fn qwen_xml_call(name: &str, arguments: &str) -> String {
    let mut call = format!("<tool_call>\n<function={name}>\n");
    if let Ok(Value::Object(arguments)) = serde_json::from_str::<Value>(arguments) {
        for (key, value) in arguments {
            let value = match value {
                Value::String(text) => text,
                other => other.to_string(),
            };
            call.push_str(&format!("<parameter={key}>\n{value}\n</parameter>\n"));
        }
    }
    call.push_str("</function>\n</tool_call>");
    call
}

/// A unified response: one LFM-Audio round hears the latest user speech in
/// the context of the conversation and speaks its answer.
async fn run_unified(
    ctx: &ResponseContext,
    plan: &ResponsePlan,
    emitter: &Emitter,
    control: &TurnControl,
) -> Result<Outcome, String> {
    let ProfileMode::Unified { s2s, .. } = &ctx.profile.mode else {
        return Err("Not a unified profile".to_string());
    };
    let variant = *s2s;
    let lfm3 = variant.family() == ModelFamily::Lfm3Audio;
    let items = prompt_items(ctx, plan);
    // The latest user speech is heard as audio; earlier turns as text.
    let target = items
        .iter()
        .rposition(|item| item.role() == Some(Role::User) && item.audio.is_some());
    let skip = target.map(|index| items[index].id.clone());
    let claim = skip
        .as_deref()
        .filter(|_| lfm3)
        .and_then(|item_id| TranscriptClaim::take(&ctx.pending_transcripts, item_id));
    let items = with_transcripts(ctx, items, skip.as_deref()).await;
    let tools = if lfm3 { tools_for(plan) } else { Vec::new() };
    if !lfm3 && !plan.tools.is_empty() {
        warn!(model = %variant.dir_name(), "realtime tools ignored: only LFM3-Audio calls tools");
    }

    let mut messages = Vec::new();
    for (index, item) in items.iter().enumerate() {
        if Some(index) == target {
            if !lfm3 {
                // LFM2.5-Audio hears its clip last.
                break;
            }
            messages.push(ChatMessage {
                role: ChatRole::User,
                content: CHAT_AUDIO_PLACEHOLDER.to_string(),
            });
            continue;
        }
        let message = match &item.kind {
            ItemKind::Message {
                role: Role::System, ..
            } => None,
            ItemKind::Message { role, .. } => item.text().map(|text| ChatMessage {
                role: if *role == Role::User {
                    ChatRole::User
                } else {
                    ChatRole::Assistant
                },
                content: text,
            }),
            ItemKind::FunctionCall {
                name, arguments, ..
            } => Some(ChatMessage {
                role: ChatRole::Assistant,
                content: qwen_xml_call(name, arguments),
            }),
            ItemKind::FunctionCallOutput { output, .. } => Some(ChatMessage {
                role: ChatRole::Tool,
                content: output.clone(),
            }),
        };
        messages.extend(message);
    }
    let (wav_bytes, heard_ms) = match target.and_then(|index| items[index].audio.as_ref()) {
        Some(audio) => (
            wav(&audio.samples, audio.sample_rate)?,
            audio.samples.len() as u64 * 1_000 / u64::from(audio.sample_rate.max(1)),
        ),
        None => (
            wav(&vec![0i16; (24 * SILENT_INPUT_MS) as usize], 24_000)?,
            0,
        ),
    };
    emitter.add_usage(|usage| usage.input_audio_ms += heard_ms);
    let system = system_prompt(plan, &items, !tools.is_empty());

    ctx.model_leases
        .hold(&ctx.state.runtime, variant)
        .await
        .map_err(|err| model_load_error("unified audio", err))?;
    let parser = Arc::new(Mutex::new(
        (!tools.is_empty()).then(|| ToolCallParser::new(ToolCallFormat::QwenXml)),
    ));
    let encoder: Arc<Mutex<Option<OutputEncoder>>> = Arc::default();
    let audio_out = plan.modality == Modality::Audio;
    let output_format = plan.output_format;
    let sink_parser = parser.clone();
    let sink_encoder = encoder.clone();
    let sink_emitter = emitter.clone();
    let sink_control = control.clone();
    let on_chunk = move |chunk: izwi_core::StreamingOutput| -> futures::future::Either<
        std::future::Ready<izwi_core::Result<()>>,
        std::pin::Pin<Box<dyn std::future::Future<Output = izwi_core::Result<()>> + Send>>,
    > {
        let cancelled = || izwi_core::Error::Cancelled("realtime response cancelled".to_string());
        if sink_control.is_cancelled() {
            return futures::future::Either::Left(std::future::ready(Err(cancelled())));
        }
        if let Some(delta) = chunk.text.as_deref().filter(|delta| !delta.is_empty()) {
            // A step decodes at most one text token.
            sink_emitter.add_usage(|usage| usage.output_text_tokens += 1);
            let visible = match sink_parser
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .as_mut()
            {
                Some(parser) => parser.push(delta).content,
                None => delta.to_string(),
            };
            sink_emitter.text_delta(&visible);
        }
        // A text-only response on a speech model: the speech it generates
        // anyway is not sent (D13, emulated).
        if !audio_out || chunk.samples.is_empty() {
            return futures::future::Either::Left(std::future::ready(Ok(())));
        }
        let deltas = {
            let mut guard = sink_encoder
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if guard.is_none() {
                match OutputEncoder::new(output_format, chunk.sample_rate.max(1)) {
                    Ok(created) => *guard = Some(created),
                    Err(err) => {
                        warn!("realtime unified audio encoder failed: {err}");
                        return futures::future::Either::Left(std::future::ready(Err(cancelled())));
                    }
                }
            }
            let Some(encoder) = guard.as_mut() else {
                return futures::future::Either::Left(std::future::ready(Err(cancelled())));
            };
            match encoder.push(&chunk.samples, false) {
                Ok(deltas) => deltas
                    .into_iter()
                    .map(|delta| {
                        let ms = encoder.duration_ms(delta.len());
                        (delta, ms)
                    })
                    .collect::<Vec<_>>(),
                Err(err) => {
                    warn!("realtime unified audio encoding failed: {err}");
                    return futures::future::Either::Left(std::future::ready(Err(cancelled())));
                }
            }
        };
        let emitter = sink_emitter.clone();
        futures::future::Either::Right(Box::pin(async move {
            for (delta, ms) in deltas {
                emitter
                    .audio_delta(delta, ms, true)
                    .await
                    .map_err(|_| cancelled())?;
            }
            Ok(())
        }))
    };
    let max_tokens = plan
        .max_output_tokens
        .unwrap_or(DEFAULT_UNIFIED_MAX_TOKENS)
        .clamp(1, 4_096);
    let _lane = ctx.lane.lock().await;
    let permit = acquire_stage_permit(&ctx.state).await?;
    let generation = ctx
        .state
        .runtime
        .speech_to_speech_round_streaming_bytes(
            variant,
            &wav_bytes,
            messages,
            tools.clone(),
            claim.is_some(),
            GenerationParams {
                max_tokens,
                speaker: plan.voice.speaker.clone(),
                voice: plan.voice.speaker.clone(),
                ..Default::default()
            },
            Some(&system),
            Some(&ctx.correlation_id),
            permit.runtime_context(),
            on_chunk,
        )
        .await
        .map_err(|err| format!("Speech-to-speech failed: {err}"))?;
    drop(permit);
    if let Some((item_id, pending)) = claim.and_then(TranscriptClaim::finish) {
        let text = generation
            .input_transcription
            .as_deref()
            .unwrap_or_default()
            .trim()
            .to_string();
        if let Some(item) = lock_items(&ctx.items).get_mut(&item_id) {
            item.set_transcript(&text);
        }
        if pending.emit {
            ctx.out.event(
                "conversation.item.input_audio_transcription.completed",
                json!({"item_id": item_id, "content_index": 0, "transcript": text,
                       "usage": {"type": "duration", "seconds": pending.audio_secs}}),
            );
        }
        let _ = pending.ready.send(true);
    }

    let tail = parser
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .take()
        .map(|mut parser| parser.finish().content)
        .unwrap_or_default();
    emitter.text_delta(&tail);
    let remaining = {
        let mut encoder = encoder.lock().unwrap_or_else(|poison| poison.into_inner());
        match encoder.as_mut() {
            Some(encoder) => encoder
                .push(&[], true)?
                .into_iter()
                .map(|delta| {
                    let ms = encoder.duration_ms(delta.len());
                    (delta, ms)
                })
                .collect::<Vec<_>>(),
            None => Vec::new(),
        }
    };
    for (delta, ms) in remaining {
        emitter.audio_delta(delta, ms, true).await?;
    }
    if !tools.is_empty() {
        let (_, calls) = parse_answer(ToolCallFormat::QwenXml, &generation.text);
        emitter.function_calls(&calls);
    }
    Ok(Outcome::Completed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_counts_audio_at_openais_rates() {
        let usage = Usage {
            input_text_tokens: 100,
            cached_tokens: 40,
            output_text_tokens: 20,
            input_audio_ms: 3_000,
            output_audio_ms: 2_000,
        }
        .to_json();
        assert_eq!(usage["input_token_details"]["audio_tokens"], 30);
        assert_eq!(usage["output_token_details"]["audio_tokens"], 40);
        assert_eq!(usage["input_tokens"], 130);
        assert_eq!(usage["output_tokens"], 60);
        assert_eq!(usage["total_tokens"], 190);
        assert_eq!(usage["input_token_details"]["cached_tokens"], 40);
    }

    #[test]
    fn lfm3_tool_calls_render_in_its_xml() {
        assert_eq!(
            qwen_xml_call("get_weather", r#"{"city":"Harare","days":2}"#),
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nHarare\n</parameter>\n\
             <parameter=days>\n2\n</parameter>\n</function>\n</tool_call>"
        );
    }
}
