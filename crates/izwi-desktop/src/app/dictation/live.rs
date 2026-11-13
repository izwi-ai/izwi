//! Live dictation for speech models that stream: the audio goes to the
//! server's OpenAI Realtime endpoint while it is spoken, the words so far
//! are shown in the indicator, and at the end of the recording the turn's
//! transcript is already nearly done.
//!
//! Anything that goes wrong here is silent: the recording is always kept in
//! full, and the caller transcribes it the ordinary way when this session
//! has no transcript to give.

use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use futures::{SinkExt, StreamExt};
use izwi_audio_codec::StreamResampler;
use izwi_dictation::audio;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use url::Url;

/// The Realtime endpoint takes PCM at this rate.
const REALTIME_RATE: u32 = 24_000;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);

enum Outgoing {
    Audio(Vec<f32>),
    /// End the turn and wait for its transcript.
    Commit,
}

pub type PartialSink = Arc<dyn Fn(String) + Send + Sync>;

/// One recording's live session. Dropping it closes the connection.
pub struct LiveSession {
    outgoing: mpsc::UnboundedSender<Outgoing>,
    transcript: std_mpsc::Receiver<Option<String>>,
}

/// The WebSocket address of the Realtime endpoint, for servers reached over
/// plain HTTP (the desktop's own server). `None` otherwise: a TLS connection
/// would need a stack this client does not carry.
fn realtime_url(server: &Url) -> Option<String> {
    if server.scheme() != "http" {
        return None;
    }
    let host = server.host_str()?;
    let port = server.port_or_known_default()?;
    Some(format!(
        "ws://{host}:{port}/v1/realtime?intent=transcription"
    ))
}

impl LiveSession {
    /// Open a session for one recording. `None` when the server cannot be
    /// reached this way.
    pub fn start(
        server: &Url,
        model: &str,
        language: Option<&str>,
        prompt: Option<String>,
        on_partial: PartialSink,
    ) -> Option<Self> {
        let url = realtime_url(server)?;
        let (outgoing, inbox) = mpsc::unbounded_channel();
        let (result, transcript) = std_mpsc::channel();
        let mut transcription = json!({ "model": model });
        if let Some(language) = language {
            transcription["language"] = json!(language);
        }
        if let Some(prompt) = prompt {
            transcription["prompt"] = json!(prompt);
        }
        tauri::async_runtime::spawn(async move {
            let text = run(url, transcription, inbox, on_partial).await;
            let _ = result.send(text);
        });
        Some(Self {
            outgoing,
            transcript,
        })
    }

    /// Send newly recorded 16 kHz mono audio.
    pub fn push(&self, samples: Vec<f32>) {
        let _ = self.outgoing.send(Outgoing::Audio(samples));
    }

    /// End the turn and return its transcript, if it arrives within `wait`.
    pub fn finish(self, wait: Duration) -> Option<String> {
        self.outgoing.send(Outgoing::Commit).ok()?;
        self.transcript
            .recv_timeout(wait)
            .ok()
            .flatten()
            .filter(|text| !text.trim().is_empty())
    }
}

fn append_event(samples: &[f32]) -> Message {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        bytes.extend_from_slice(&audio::pcm16(*sample).to_le_bytes());
    }
    let audio = base64::engine::general_purpose::STANDARD.encode(bytes);
    Message::text(json!({ "type": "input_audio_buffer.append", "audio": audio }).to_string())
}

/// The session's life: configure, stream, commit, read the transcript.
async fn run(
    url: String,
    transcription: Value,
    mut inbox: mpsc::UnboundedReceiver<Outgoing>,
    on_partial: PartialSink,
) -> Option<String> {
    let (mut socket, _) =
        tokio::time::timeout(CONNECT_TIMEOUT, tokio_tungstenite::connect_async(url))
            .await
            .ok()?
            .ok()?;
    // No turn detection: the hotkey decides where the turn ends.
    let configure = json!({
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": { "input": { "transcription": transcription, "turn_detection": null } },
        },
    });
    socket
        .send(Message::text(configure.to_string()))
        .await
        .ok()?;

    let mut resampler = StreamResampler::new(audio::SAMPLE_RATE, REALTIME_RATE).ok()?;
    let mut partial = String::new();
    let mut committed = false;
    loop {
        tokio::select! {
            outgoing = inbox.recv(), if !committed => match outgoing {
                Some(Outgoing::Audio(samples)) => {
                    let resampled = resampler.push(&samples).ok()?;
                    if !resampled.is_empty() {
                        socket.send(append_event(&resampled)).await.ok()?;
                    }
                }
                Some(Outgoing::Commit) => {
                    let rest = resampler.finish().ok()?;
                    if !rest.is_empty() {
                        socket.send(append_event(&rest)).await.ok()?;
                    }
                    let commit = json!({ "type": "input_audio_buffer.commit" });
                    socket.send(Message::text(commit.to_string())).await.ok()?;
                    committed = true;
                }
                // The recording was discarded.
                None => return None,
            },
            incoming = socket.next() => {
                let Message::Text(text) = incoming?.ok()? else {
                    continue;
                };
                let event: Value = serde_json::from_str(text.as_str()).ok()?;
                match event["type"].as_str()? {
                    "conversation.item.input_audio_transcription.delta" => {
                        if let Some(delta) = event["delta"].as_str() {
                            partial.push_str(delta);
                            on_partial(partial.clone());
                        }
                    }
                    "conversation.item.input_audio_transcription.completed" => {
                        return event["transcript"].as_str().map(str::to_string);
                    }
                    "conversation.item.input_audio_transcription.failed" | "error" => return None,
                    _ => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_plain_http_servers_get_a_live_session() {
        let url = |server: &str| realtime_url(&Url::parse(server).unwrap());
        assert_eq!(
            url("http://localhost:8080").as_deref(),
            Some("ws://localhost:8080/v1/realtime?intent=transcription")
        );
        assert_eq!(
            url("http://127.0.0.1:8081/").as_deref(),
            Some("ws://127.0.0.1:8081/v1/realtime?intent=transcription")
        );
        assert_eq!(url("https://speech.example.com"), None);
    }

    #[test]
    fn audio_is_sent_as_base64_pcm16() {
        let Message::Text(text) = append_event(&[0.0, 1.0]) else {
            panic!("append events are text frames");
        };
        let event: Value = serde_json::from_str(text.as_str()).unwrap();
        assert_eq!(event["type"], "input_audio_buffer.append");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(event["audio"].as_str().unwrap())
            .unwrap();
        assert_eq!(bytes, [0, 0, 0xff, 0x7f]);
    }
}
