//! Telephony media streams at `/v1/realtime/media-stream`: the WebSocket
//! framing call-centre platforms and carrier APIs hand a phone call's audio
//! to an application in, mapped onto a `/v1/realtime` session.
//!
//! The framing is JSON text frames, one per event. The carrier sends
//! `connected`, `start` (the call's `streamSid`, its media format and custom
//! parameters), `media` (base64 8 kHz μ-law, usually 20 ms a frame), `mark`,
//! `dtmf` and `stop`; the application sends `media`, `mark` and `clear`.
//!
//! This is a translation at the socket, not a second session: once `start`
//! has arrived the call runs the session actor of `openai_realtime` with
//! G.711 μ-law in and out and server VAD. Inbound `media` becomes
//! `input_audio_buffer.append`, and the session's events become outbound
//! frames: speech as `media`, a `mark` at the end of each reply, `clear`
//! when the caller talks over a reply that is still playing, and (Izwi's
//! extensions, which carriers ignore) `transcript` and `error`.

use std::time::{Duration, Instant};

use axum::extract::ws::{CloseFrame, Message, WebSocket};
use futures::channel::mpsc;
use futures::{future, stream, SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::debug;

use crate::app::openai_realtime::{run_session, SessionKind, SessionSpec};
use crate::state::AppState;

/// The largest carrier frame: `media` frames are a few hundred bytes, and a
/// `start` carries the call's custom parameters.
pub(crate) const MAX_MESSAGE_BYTES: usize = 256 * 1024;
/// How long a socket may stay open before its `start` arrives.
const START_TIMEOUT: Duration = Duration::from_secs(10);
/// Bytes of 8 kHz G.711 in a millisecond.
const BYTES_PER_MS: u64 = 8;

/// What a call runs. The query string of the stream URL and the `start`
/// frame's `customParameters` both carry these (some carriers drop the query
/// string); a custom parameter wins.
#[derive(Debug, Default, Clone, Deserialize)]
pub(crate) struct CallParams {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    intent: Option<String>,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    transcription_model: Option<String>,
    /// `false` keeps the `transcript` frames off the stream.
    #[serde(default)]
    transcripts: Option<String>,
}

impl CallParams {
    /// These parameters, with `base`'s where one is not given.
    fn over(self, base: Self) -> Self {
        Self {
            model: self.model.or(base.model),
            intent: self.intent.or(base.intent),
            voice: self.voice.or(base.voice),
            instructions: self.instructions.or(base.instructions),
            language: self.language.or(base.language),
            transcription_model: self.transcription_model.or(base.transcription_model),
            transcripts: self.transcripts.or(base.transcripts),
        }
    }

    fn sends_transcripts(&self) -> bool {
        !self.transcripts.as_deref().is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "false" | "0" | "off" | "no"
            )
        })
    }

    /// The session the call runs: `bound` (an ephemeral key's session
    /// config, if the call presented one) with the call's audio formats and
    /// parameters laid over it.
    pub(crate) fn session_spec(&self, bound: Option<Value>) -> Result<SessionSpec, String> {
        let spec = SessionSpec::resolve(self.model.as_deref(), self.intent.as_deref(), bound)?;
        let g711 = json!({"type": "audio/pcmu"});
        let mut call = json!({"audio": {"input": {"format": g711}, "output": {"format": g711}}});
        if let Some(model) = self.transcription_model.as_deref() {
            call["audio"]["input"]["transcription"]["model"] = json!(model);
        }
        if let Some(language) = self.language.as_deref() {
            call["audio"]["input"]["transcription"]["language"] = json!(language);
        }
        if let Some(voice) = self.voice.as_deref() {
            call["audio"]["output"]["voice"] = json!(voice);
        }
        if let Some(instructions) = self.instructions.as_deref() {
            call["instructions"] = json!(instructions);
        }
        let mut session = spec
            .bound_session
            .unwrap_or_else(|| json!({"type": "realtime"}));
        merge(&mut session, call);
        // The formats are the carrier's whole, not laid over a key's.
        session["audio"]["input"]["format"] = g711.clone();
        session["audio"]["output"]["format"] = g711;
        // The carrier commits nothing itself: the server cuts the turns.
        let turn_detection = &mut session["audio"]["input"]["turn_detection"];
        if !turn_detection.is_object() {
            *turn_detection = json!({"type": "server_vad"});
        }
        Ok(SessionSpec {
            bound_session: Some(session),
            ..spec
        })
    }
}

/// Lay `over` on `base`, object by object.
fn merge(base: &mut Value, over: Value) {
    match (base, over) {
        (Value::Object(base), Value::Object(over)) => {
            for (key, value) in over {
                merge(base.entry(key).or_insert(Value::Null), value);
            }
        }
        (base, over) => *base = over,
    }
}

/// A call's `start` frame.
#[derive(Debug)]
struct Start {
    stream_sid: String,
    call_sid: Option<String>,
    params: CallParams,
}

fn parse_start(frame: &Value) -> Result<Start, String> {
    let start = &frame["start"];
    let stream_sid = [&frame["streamSid"], &start["streamSid"]]
        .into_iter()
        .find_map(Value::as_str)
        .filter(|sid| !sid.is_empty())
        .ok_or("The `start` frame carries no `streamSid`.")?;
    let format = &start["mediaFormat"];
    let unsupported = format["encoding"]
        .as_str()
        .is_some_and(|encoding| encoding != "audio/x-mulaw")
        || format["sampleRate"]
            .as_u64()
            .is_some_and(|rate| rate != 8_000)
        || format["channels"]
            .as_u64()
            .is_some_and(|channels| channels != 1);
    if unsupported {
        return Err(format!(
            "Unsupported media format {format}; a media stream carries mono 8 kHz audio/x-mulaw."
        ));
    }
    Ok(Start {
        stream_sid: stream_sid.to_string(),
        call_sid: start["callSid"].as_str().map(str::to_string),
        params: serde_json::from_value(start["customParameters"].clone()).unwrap_or_default(),
    })
}

/// Read frames up to the call's `start`; `None` when the carrier leaves first.
async fn read_start(socket: &mut WebSocket) -> Option<Result<Start, String>> {
    loop {
        let Message::Text(text) = socket.recv().await?.ok()? else {
            continue;
        };
        let Ok(frame) = serde_json::from_str::<Value>(text.as_str()) else {
            continue;
        };
        match frame["event"].as_str() {
            Some("start") => return Some(parse_start(&frame)),
            Some("stop") => return None,
            _ => {}
        }
    }
}

/// The session event a carrier frame becomes, if any. A frame that is not
/// part of the framing is dropped, never an error: carriers add events.
fn inbound<E>(message: Result<Message, E>) -> Option<Result<Message, E>> {
    let text = match message {
        Ok(Message::Text(text)) => text,
        Ok(Message::Binary(_)) => return None,
        other => return Some(other),
    };
    let frame: Value = serde_json::from_str(text.as_str()).ok()?;
    match frame["event"].as_str()? {
        "media" => {
            let media = &frame["media"];
            // With both tracks streamed, only the caller's is the input.
            if media["track"]
                .as_str()
                .is_some_and(|track| track != "inbound")
            {
                return None;
            }
            let event =
                json!({"type": "input_audio_buffer.append", "audio": media["payload"].as_str()?});
            Some(Ok(Message::Text(event.to_string().into())))
        }
        "stop" => Some(Ok(Message::Close(None))),
        // `connected`, a repeated `start`, `mark` echoes and `dtmf`.
        _ => None,
    }
}

/// The reply the carrier is playing.
struct Playing {
    item_id: String,
    started: Instant,
    sent_ms: u64,
}

impl Playing {
    fn ends(&self) -> Instant {
        self.started + Duration::from_millis(self.sent_ms)
    }
}

/// Turns the session's events into the call's outbound frames.
struct Outbound {
    stream_sid: String,
    transcripts: bool,
    playing: Option<Playing>,
    /// Events the adapter sends the session itself, as a client would.
    session: mpsc::UnboundedSender<String>,
}

impl Outbound {
    fn frame(&self, event: &str, fields: Value) -> Message {
        let mut frame = json!({"event": event, "streamSid": self.stream_sid});
        merge(&mut frame, fields);
        Message::Text(frame.to_string().into())
    }

    /// The frames one outbound message becomes at `now`.
    fn frames(&mut self, message: Message, now: Instant) -> Vec<Message> {
        let Message::Text(text) = &message else {
            return vec![message];
        };
        let Ok(event) = serde_json::from_str::<Value>(text.as_str()) else {
            return Vec::new();
        };
        let item_id = event["item_id"].as_str().unwrap_or_default();
        match event["type"].as_str().unwrap_or_default() {
            "response.output_audio.delta" => {
                let Some(payload) = event["delta"].as_str() else {
                    return Vec::new();
                };
                let padding = payload
                    .bytes()
                    .rev()
                    .take_while(|byte| *byte == b'=')
                    .count();
                let bytes = (payload.len() / 4 * 3).saturating_sub(padding) as u64;
                // The carrier plays what it is sent in order, in real time.
                let queued = self.playing.take();
                let mut playing = match queued {
                    Some(playing) if playing.item_id == item_id => playing,
                    queued => Playing {
                        item_id: item_id.to_string(),
                        started: queued.map_or(now, |playing| playing.ends().max(now)),
                        sent_ms: 0,
                    },
                };
                playing.sent_ms += bytes / BYTES_PER_MS;
                self.playing = Some(playing);
                vec![self.frame("media", json!({"media": {"payload": payload}}))]
            }
            // The carrier echoes the mark when the reply has played out.
            "response.output_audio.done" => {
                vec![self.frame("mark", json!({"mark": {"name": item_id}}))]
            }
            // The caller talks over a reply the carrier still holds: flush
            // it, and tell the session how much of it was heard.
            "input_audio_buffer.speech_started" => {
                let Some(playing) = self.playing.take().filter(|playing| playing.ends() > now)
                else {
                    return Vec::new();
                };
                let heard_ms = now.saturating_duration_since(playing.started).as_millis() as u64;
                let truncate = json!({"type": "conversation.item.truncate", "item_id": playing.item_id,
                    "content_index": 0, "audio_end_ms": heard_ms});
                let _ = self.session.unbounded_send(truncate.to_string());
                vec![self.frame("clear", json!({}))]
            }
            "conversation.item.input_audio_transcription.delta" if self.transcripts => {
                vec![self.frame(
                    "transcript",
                    json!({"item_id": item_id, "type": "delta", "text": event["delta"]}),
                )]
            }
            "conversation.item.input_audio_transcription.completed" if self.transcripts => {
                vec![self.frame(
                    "transcript",
                    json!({"item_id": item_id, "type": "completed", "text": event["transcript"]}),
                )]
            }
            "error" | "conversation.item.input_audio_transcription.failed" => {
                vec![self.frame("error", json!({"error": event["error"]}))]
            }
            "response.done" if event["response"]["status"] == "failed" => {
                let error = &event["response"]["status_details"]["error"];
                vec![self.frame("error", json!({"error": error}))]
            }
            _ => Vec::new(),
        }
    }
}

/// Refuse a call: an `error` frame, then a policy close.
async fn refuse(socket: &mut WebSocket, stream_sid: Option<&str>, message: &str) {
    let frame = json!({"event": "error", "streamSid": stream_sid,
        "error": {"type": "invalid_request_error", "code": "invalid_call", "message": message}});
    let _ = socket.send(Message::Text(frame.to_string().into())).await;
    let _ = socket
        .send(Message::Close(Some(CloseFrame {
            code: 1008,
            reason: "invalid_call".into(),
        })))
        .await;
}

/// Run one media-stream socket, which carries one call, to its end.
pub(crate) async fn handle_socket(
    mut socket: WebSocket,
    state: AppState,
    correlation_id: String,
    query: CallParams,
    bound_session: Option<Value>,
) {
    let start = match tokio::time::timeout(START_TIMEOUT, read_start(&mut socket)).await {
        Ok(Some(Ok(start))) => start,
        Ok(Some(Err(message))) => return refuse(&mut socket, None, &message).await,
        Ok(None) => return,
        Err(_) => return refuse(&mut socket, None, "No `start` frame arrived in time.").await,
    };
    let params = start.params.over(query);
    let spec = match params.session_spec(bound_session) {
        Ok(spec) => spec,
        Err(message) => return refuse(&mut socket, Some(&start.stream_sid), &message).await,
    };
    debug!(
        stream_sid = %start.stream_sid,
        call_sid = start.call_sid.as_deref().unwrap_or_default(),
        model = %spec.requested_model,
        transcription = spec.kind == SessionKind::Transcription,
        "media-stream call started"
    );

    let (sink, frames) = socket.split();
    let (session, own_events) = mpsc::unbounded();
    let mut outbound = Outbound {
        stream_sid: start.stream_sid,
        transcripts: params.sends_transcripts(),
        playing: None,
        session,
    };
    let sink = sink.with_flat_map(move |message| {
        stream::iter(outbound.frames(message, Instant::now()).into_iter().map(Ok))
    });
    // `stop`, or the carrier hanging up, ends the session.
    let frames = frames
        .filter_map(|message| future::ready(inbound(message)))
        .chain(stream::once(future::ready(Ok(Message::Close(None)))));
    let events = stream::select(
        frames,
        own_events.map(|event: String| Ok(Message::Text(event.into()))),
    );
    run_session(sink, events, state, correlation_id, spec).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outbound() -> (Outbound, mpsc::UnboundedReceiver<String>) {
        let (session, own_events) = mpsc::unbounded();
        let outbound = Outbound {
            stream_sid: "MZ1".to_string(),
            transcripts: true,
            playing: None,
            session,
        };
        (outbound, own_events)
    }

    fn event(event: Value) -> Message {
        Message::Text(event.to_string().into())
    }

    fn json_frames(frames: Vec<Message>) -> Vec<Value> {
        frames
            .into_iter()
            .map(|frame| match frame {
                Message::Text(text) => serde_json::from_str(text.as_str()).expect("json frame"),
                other => panic!("not a text frame: {other:?}"),
            })
            .collect()
    }

    fn params(value: Value) -> CallParams {
        serde_json::from_value(value).expect("params")
    }

    #[test]
    fn a_call_is_a_g711_server_vad_session_with_its_parameters() {
        let spec = params(json!({"voice": "cedar", "instructions": "Be brief.", "language": "en"}))
            .session_spec(None)
            .expect("spec");
        assert_eq!(spec.kind, SessionKind::Realtime);
        let session = spec.bound_session.expect("session");
        assert_eq!(session["type"], "realtime");
        assert_eq!(session["instructions"], "Be brief.");
        let audio = &session["audio"];
        assert_eq!(audio["input"]["format"], json!({"type": "audio/pcmu"}));
        assert_eq!(audio["output"]["format"], json!({"type": "audio/pcmu"}));
        assert_eq!(audio["input"]["turn_detection"]["type"], "server_vad");
        assert_eq!(audio["input"]["transcription"], json!({"language": "en"}));
        assert_eq!(audio["output"]["voice"], "cedar");

        // An ephemeral key's config stays, under the call's formats; a
        // session the client would have to commit gets server VAD.
        let bound = json!({"type": "transcription", "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "transcription": {"model": "Voxtral-Mini-4B-Realtime-2602"},
            "turn_detection": null}}});
        let spec = CallParams::default()
            .session_spec(Some(bound))
            .expect("spec");
        assert_eq!(spec.kind, SessionKind::Transcription);
        let input = &spec.bound_session.expect("session")["audio"]["input"];
        assert_eq!(input["format"], json!({"type": "audio/pcmu"}));
        assert_eq!(input["turn_detection"], json!({"type": "server_vad"}));
        assert_eq!(
            input["transcription"]["model"],
            "Voxtral-Mini-4B-Realtime-2602"
        );
        let tuned = json!({"audio": {"input": {"turn_detection":
            {"type": "server_vad", "silence_duration_ms": 900}}}});
        let spec = CallParams::default()
            .session_spec(Some(tuned))
            .expect("spec");
        assert_eq!(
            spec.bound_session.expect("session")["audio"]["input"]["turn_detection"]
                ["silence_duration_ms"],
            900
        );

        assert!(params(json!({"intent": "fax"})).session_spec(None).is_err());
        assert!(params(json!({"model": "no-such-model"}))
            .session_spec(None)
            .is_err());
    }

    #[test]
    fn custom_parameters_win_over_the_query_string() {
        let query =
            params(json!({"model": "izwi-voice", "voice": "marin", "transcripts": "false"}));
        let call = params(json!({"voice": "cedar", "intent": "transcription"})).over(query);
        assert_eq!(call.model.as_deref(), Some("izwi-voice"));
        assert_eq!(call.voice.as_deref(), Some("cedar"));
        assert_eq!(call.intent.as_deref(), Some("transcription"));
        assert!(!call.sends_transcripts());
        assert!(CallParams::default().sends_transcripts());
    }

    #[test]
    fn start_frames_name_the_stream_and_its_format() {
        let start = parse_start(&json!({"event": "start", "streamSid": "MZ1", "start": {
            "streamSid": "MZ1", "callSid": "CA1", "tracks": ["inbound"],
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
            "customParameters": {"voice": "cedar", "customer": "42"}}}))
        .expect("start");
        assert_eq!(start.stream_sid, "MZ1");
        assert_eq!(start.call_sid.as_deref(), Some("CA1"));
        assert_eq!(start.params.voice.as_deref(), Some("cedar"));
        // A start with the sid in its object only, and no format, is taken.
        let bare = parse_start(&json!({"event": "start", "start": {"streamSid": "MZ2"}}));
        assert_eq!(bare.expect("start").stream_sid, "MZ2");

        assert!(parse_start(&json!({"event": "start", "start": {}})).is_err());
        let wideband = json!({"event": "start", "streamSid": "MZ1", "start": {
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 16000, "channels": 1}}});
        assert!(parse_start(&wideband).is_err());
        let alaw = json!({"event": "start", "streamSid": "MZ1", "start": {
            "mediaFormat": {"encoding": "audio/x-alaw", "sampleRate": 8000, "channels": 1}}});
        assert!(parse_start(&alaw).is_err());
    }

    #[test]
    fn carrier_frames_become_session_events() {
        let frame = |frame: Value| inbound::<()>(Ok(Message::Text(frame.to_string().into())));
        let media = frame(json!({"event": "media", "streamSid": "MZ1",
            "media": {"track": "inbound", "chunk": "1", "timestamp": "20", "payload": "/38="}}));
        let Some(Ok(Message::Text(text))) = media else {
            panic!("media is an append");
        };
        assert_eq!(
            serde_json::from_str::<Value>(text.as_str()).expect("json"),
            json!({"type": "input_audio_buffer.append", "audio": "/38="})
        );
        assert!(matches!(
            frame(json!({"event": "stop", "streamSid": "MZ1"})),
            Some(Ok(Message::Close(None)))
        ));
        // The agent's own track, bookkeeping and unknown events are dropped.
        for dropped in [
            json!({"event": "media", "media": {"track": "outbound", "payload": "/38="}}),
            json!({"event": "media", "media": {}}),
            json!({"event": "connected", "protocol": "Call", "version": "1.0.0"}),
            json!({"event": "mark", "mark": {"name": "item_1"}}),
            json!({"event": "dtmf", "dtmf": {"digit": "5"}}),
            json!({"event": "something-new"}),
            json!({"no": "event"}),
        ] {
            assert!(frame(dropped.clone()).is_none(), "{dropped}");
        }
        assert!(inbound::<()>(Ok(Message::Text("not json".into()))).is_none());
        assert!(inbound::<()>(Ok(Message::Binary(vec![1, 2].into()))).is_none());
        assert!(matches!(
            inbound::<()>(Ok(Message::Ping(vec![1].into()))),
            Some(Ok(Message::Ping(_)))
        ));
    }

    #[test]
    fn speech_goes_out_as_media_with_a_mark_at_its_end() {
        let (mut outbound, _own_events) = outbound();
        let now = Instant::now();
        // 800 bytes of μ-law: 100 ms.
        let payload = crate::app::openai_realtime::audio::base64(&[0xFF; 800]);
        let frames = json_frames(outbound.frames(
            event(
                json!({"type": "response.output_audio.delta", "item_id": "item_1",
                "delta": payload}),
            ),
            now,
        ));
        assert_eq!(
            frames,
            [json!({"event": "media", "streamSid": "MZ1", "media": {"payload": payload}})]
        );
        assert_eq!(outbound.playing.as_ref().expect("playing").sent_ms, 100);
        let frames = json_frames(outbound.frames(
            event(json!({"type": "response.output_audio.done", "item_id": "item_1"})),
            now,
        ));
        assert_eq!(
            frames,
            [json!({"event": "mark", "streamSid": "MZ1", "mark": {"name": "item_1"}})]
        );
        // The session's own bookkeeping stays off the stream; control
        // frames pass through.
        for kind in ["session.created", "response.created", "response.done"] {
            assert!(outbound
                .frames(event(json!({"type": kind})), now)
                .is_empty());
        }
        assert!(matches!(
            outbound.frames(Message::Close(None), now)[..],
            [Message::Close(None)]
        ));
    }

    #[test]
    fn talking_over_a_reply_clears_it_and_tells_the_session_what_was_heard() {
        let (mut outbound, mut own_events) = outbound();
        let now = Instant::now();
        let payload = crate::app::openai_realtime::audio::base64(&[0xFF; 800]);
        for _ in 0..10 {
            outbound.frames(
                event(
                    json!({"type": "response.output_audio.delta", "item_id": "item_1",
                    "delta": payload}),
                ),
                now,
            );
        }
        // One second was sent at once; the caller speaks 300 ms into it.
        let started = event(json!({"type": "input_audio_buffer.speech_started",
            "item_id": "item_2", "audio_start_ms": 0}));
        let frames =
            json_frames(outbound.frames(started.clone(), now + Duration::from_millis(300)));
        assert_eq!(frames, [json!({"event": "clear", "streamSid": "MZ1"})]);
        let truncate: Value =
            serde_json::from_str(&own_events.try_recv().expect("event")).expect("json");
        assert_eq!(
            truncate,
            json!({"type": "conversation.item.truncate", "item_id": "item_1",
                "content_index": 0, "audio_end_ms": 300})
        );

        // Nothing is playing now, and a reply that has played out is left.
        assert!(outbound
            .frames(started.clone(), now + Duration::from_millis(400))
            .is_empty());
        outbound.frames(
            event(
                json!({"type": "response.output_audio.delta", "item_id": "item_3",
                "delta": payload}),
            ),
            now + Duration::from_secs(1),
        );
        assert!(outbound
            .frames(started, now + Duration::from_secs(2))
            .is_empty());
        assert!(own_events.try_recv().is_err(), "no second truncate");
    }

    #[test]
    fn transcripts_and_errors_are_izwi_frames() {
        let (mut outbound, _own_events) = outbound();
        let now = Instant::now();
        let delta = event(
            json!({"type": "conversation.item.input_audio_transcription.delta",
            "item_id": "item_1", "content_index": 0, "delta": "hello"}),
        );
        assert_eq!(
            json_frames(outbound.frames(delta.clone(), now)),
            [
                json!({"event": "transcript", "streamSid": "MZ1", "item_id": "item_1",
                "type": "delta", "text": "hello"})
            ]
        );
        let completed = event(
            json!({"type": "conversation.item.input_audio_transcription.completed",
            "item_id": "item_1", "content_index": 0, "transcript": "hello there"}),
        );
        assert_eq!(
            json_frames(outbound.frames(completed, now)),
            [
                json!({"event": "transcript", "streamSid": "MZ1", "item_id": "item_1",
                "type": "completed", "text": "hello there"})
            ]
        );
        let failed = event(
            json!({"type": "response.done", "response": {"status": "failed",
            "status_details": {"type": "failed", "error": {"code": "x", "message": "no model"}}}}),
        );
        assert_eq!(
            json_frames(outbound.frames(failed, now)),
            [json!({"event": "error", "streamSid": "MZ1",
                "error": {"code": "x", "message": "no model"}})]
        );
        outbound.transcripts = false;
        assert!(outbound.frames(delta, now).is_empty());
    }
}
