//! Live transcripts: an ASR model with a native streaming session (Nemotron
//! streaming ASR, Voxtral Realtime, LFM3-Audio LiveASR) hears a turn as it
//! is spoken, so `conversation.item.input_audio_transcription.delta` events
//! go out while the user still speaks. Other models transcribe the turn once
//! it is committed (`transcribe.rs`).
//!
//! One native session serves one turn. The runtime's contract is start,
//! push, finish, and finish closes the session, so a session opens when the
//! turn's speech is confirmed and is finished at the commit. Between turns a
//! connection holds no slot of the session quota, and a session never sits
//! idle long enough to expire.
//!
//! The session's final text is the turn's transcript: the turn is not
//! transcribed a second time. When the turn has no session to finish (the
//! quota is full, or the session failed to start, to take audio or to
//! finish), the committed audio is transcribed the turn-based way instead,
//! with no error event; deltas already sent are not sent again.

use std::time::{Duration, Instant};

use izwi_core::{RuntimeAsrRealtimeEvent, RuntimeAsrRealtimeStream, VoiceLatency};
use serde_json::json;
use tokio::sync::mpsc;
use tracing::debug;

use super::profiles::RealtimeProfile;
use super::transcribe::Transcription;
use crate::app::native_asr_stream;
use crate::state::AppState;

/// Whether a session on `profile` transcribes `model_id` live. A unified
/// conversation keeps the turn-based path: its model replies on the
/// session's one lane, and LFM3-Audio's reply transcribes the turn itself.
pub(crate) fn transcribes_live(
    profile: &RealtimeProfile,
    conversation: bool,
    model_id: &str,
) -> bool {
    !(conversation && profile.is_unified())
        && izwi_core::asr_streams_live(izwi_core::resolve_asr_model_variant(Some(model_id)))
}

/// `transcription_mode` as `/v1/realtime/capabilities` reports it.
pub(crate) fn transcription_mode(live: bool) -> &'static str {
    if live {
        "live"
    } else {
        "turn"
    }
}

enum Command {
    Audio(Vec<i16>),
    Finish(Box<Transcription>),
}

/// The session's handle on a turn being transcribed live. Dropping it
/// abandons the turn and closes its native session.
pub(crate) struct LiveTurn {
    /// The user item the turn becomes at its commit.
    pub(crate) item_id: String,
    commands: mpsc::UnboundedSender<Command>,
}

impl LiveTurn {
    /// The next audio of the turn.
    pub(crate) fn push(&self, samples: Vec<i16>) {
        let _ = self.commands.send(Command::Audio(samples));
    }

    /// The turn is committed: its transcript completes `transcription`,
    /// from the native session or, failing that, the turn-based way.
    pub(crate) fn finish(self, transcription: Transcription) {
        if let Err(mpsc::error::SendError(Command::Finish(transcription))) =
            self.commands.send(Command::Finish(Box::new(transcription)))
        {
            transcription.spawn();
        }
    }
}

/// A turn to transcribe live.
pub(crate) struct LiveTranscription {
    pub(crate) state: AppState,
    pub(crate) item_id: String,
    pub(crate) model_id: String,
    pub(crate) language: Option<String>,
    pub(crate) prompt: Option<String>,
    /// Whether the client asked for transcription events.
    pub(crate) emit: bool,
    pub(crate) out: super::respond::Outbound,
    pub(crate) model_leases: izwi_core::SessionModelLeases,
    pub(crate) sample_rate: u32,
}

impl LiveTranscription {
    /// Start transcribing a turn whose audio so far is `heard`.
    pub(crate) fn spawn(self, heard: Vec<i16>) -> LiveTurn {
        // The first partial's latency runs from the turn's first audio.
        let heard_for = Duration::from_secs_f64(heard.len() as f64 / f64::from(self.sample_rate));
        let now = Instant::now();
        let speech_started = now.checked_sub(heard_for).unwrap_or(now);
        let (commands, inbox) = mpsc::unbounded_channel();
        let turn = LiveTurn {
            item_id: self.item_id.clone(),
            commands,
        };
        turn.push(heard);
        tokio::spawn(self.run(inbox, speech_started));
        turn
    }

    async fn run(self, mut inbox: mpsc::UnboundedReceiver<Command>, speech_started: Instant) {
        let mut stream = self.open().await;
        let mut text = String::new();
        let mut partials = false;
        let mut samples_heard = 0usize;
        let mut next = None;
        loop {
            let command = match next.take() {
                Some(command) => command,
                None => match inbox.recv().await {
                    Some(command) => command,
                    // The turn was abandoned: cleared, rejected, or the
                    // session closed.
                    None => return,
                },
            };
            match command {
                Command::Audio(mut samples) => {
                    // Audio that queued behind a push joins the next one.
                    loop {
                        match inbox.try_recv() {
                            Ok(Command::Audio(more)) => samples.extend(more),
                            Ok(finish) => {
                                next = Some(finish);
                                break;
                            }
                            Err(_) => break,
                        }
                    }
                    samples_heard += samples.len();
                    let Some(open) = stream.as_mut() else {
                        continue;
                    };
                    match open.push(&self.state, &samples, self.sample_rate).await {
                        Ok(events) => {
                            if self.take_events(events, &mut text) && !partials {
                                partials = true;
                                self.state.runtime.record_voice_latency(
                                    VoiceLatency::AsrLiveFirstPartial,
                                    speech_started.elapsed(),
                                );
                            }
                        }
                        Err(err) => {
                            debug!("live transcript falls back to the committed turn: {err}");
                            stream = None;
                        }
                    }
                }
                Command::Finish(mut transcription) => {
                    let finished = match stream.take() {
                        Some(mut open) => open
                            .finish(&self.state)
                            .await
                            .map_err(|err| {
                                debug!("live transcript falls back to the committed turn: {err}")
                            })
                            .ok(),
                        None => None,
                    };
                    match finished {
                        Some(events) => {
                            self.take_events(events, &mut text);
                            let seconds = samples_heard as f32 / self.sample_rate as f32;
                            transcription.complete(&text, seconds);
                        }
                        None => {
                            transcription.emit_deltas = !partials;
                            transcription.run().await;
                        }
                    }
                    return;
                }
            }
        }
    }

    /// The turn's native session, or `None` when it has none to use.
    async fn open(&self) -> Option<Stream> {
        #[cfg(test)]
        if let Some(script) = scripted::installed(&self.state) {
            return script.start().map(Stream::Scripted);
        }
        let variant = izwi_core::resolve_asr_model_variant(Some(&self.model_id));
        let started = match self.model_leases.hold(&self.state.runtime, variant).await {
            Ok(()) => {
                native_asr_stream::start(
                    &self.state,
                    Some(&self.model_id),
                    self.language.as_deref(),
                    self.prompt.as_deref(),
                )
                .await
            }
            Err(err) => Err(err),
        };
        match started {
            Ok(stream) => stream.map(|stream| Stream::Native(Box::new(stream))),
            Err(err) => {
                debug!("live transcript not started; the committed turn is transcribed: {err}");
                None
            }
        }
    }

    /// Send the deltas in `events` and keep the transcript so far in
    /// `text`. Returns whether any text arrived.
    fn take_events(&self, events: Vec<RuntimeAsrRealtimeEvent>, text: &mut String) -> bool {
        let mut any = false;
        for event in events {
            if event.delta.is_empty() && !event.is_final {
                continue;
            }
            *text = event.text;
            if event.delta.is_empty() {
                continue;
            }
            any = true;
            if self.emit {
                self.out.event(
                    "conversation.item.input_audio_transcription.delta",
                    json!({"item_id": self.item_id, "content_index": 0, "delta": event.delta}),
                );
            }
        }
        any
    }
}

/// A turn's native session.
enum Stream {
    Native(Box<RuntimeAsrRealtimeStream>),
    #[cfg(test)]
    Scripted(scripted::ScriptedStream),
}

impl Stream {
    async fn push(
        &mut self,
        state: &AppState,
        samples: &[i16],
        sample_rate: u32,
    ) -> izwi_core::Result<Vec<RuntimeAsrRealtimeEvent>> {
        match self {
            Self::Native(stream) => {
                native_asr_stream::push(state, stream, samples, sample_rate).await
            }
            #[cfg(test)]
            Self::Scripted(stream) => stream.push(),
        }
    }

    async fn finish(
        &mut self,
        state: &AppState,
    ) -> izwi_core::Result<Vec<RuntimeAsrRealtimeEvent>> {
        match self {
            Self::Native(stream) => native_asr_stream::finish(state, stream).await,
            #[cfg(test)]
            Self::Scripted(stream) => stream.finish(),
        }
    }
}

/// A scripted native session for the socket tests, which run without model
/// weights: it words each push, and counts the sessions open.
#[cfg(test)]
pub(crate) mod scripted {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use izwi_core::RuntimeAsrRealtimeEvent;

    use crate::state::AppState;

    /// What the sessions of one test server do.
    #[derive(Default)]
    pub(crate) struct Script {
        /// No session can be had, as when the quota is full.
        pub(crate) unavailable: AtomicBool,
        /// Every push fails.
        pub(crate) fail_pushes: AtomicBool,
        /// Sessions started, and those still open.
        pub(crate) started: AtomicUsize,
        pub(crate) open: AtomicUsize,
    }

    static SCRIPTS: Mutex<Vec<(usize, Arc<Script>)>> = Mutex::new(Vec::new());

    fn key(state: &AppState) -> usize {
        Arc::as_ptr(&state.runtime) as usize
    }

    /// A server's script, in place until dropped.
    pub(crate) struct Installed {
        owner: usize,
        script: Arc<Script>,
    }

    impl std::ops::Deref for Installed {
        type Target = Script;

        fn deref(&self) -> &Script {
            &self.script
        }
    }

    impl Drop for Installed {
        fn drop(&mut self) {
            SCRIPTS
                .lock()
                .unwrap_or_else(|poison| poison.into_inner())
                .retain(|(owner, _)| *owner != self.owner);
        }
    }

    /// Script the native sessions of `state`'s server.
    pub(crate) fn install(state: &AppState) -> Installed {
        let script = Arc::new(Script::default());
        SCRIPTS
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .push((key(state), script.clone()));
        Installed {
            owner: key(state),
            script,
        }
    }

    pub(super) fn installed(state: &AppState) -> Option<Arc<Script>> {
        SCRIPTS
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .iter()
            .find(|(owner, _)| *owner == key(state))
            .map(|(_, script)| script.clone())
    }

    impl Script {
        pub(super) fn start(self: Arc<Self>) -> Option<ScriptedStream> {
            if self.unavailable.load(Ordering::SeqCst) {
                return None;
            }
            self.started.fetch_add(1, Ordering::SeqCst);
            self.open.fetch_add(1, Ordering::SeqCst);
            Some(ScriptedStream {
                script: self,
                text: String::new(),
                pushes: 0,
            })
        }
    }

    pub(crate) struct ScriptedStream {
        script: Arc<Script>,
        text: String,
        pushes: usize,
    }

    impl ScriptedStream {
        fn event(&mut self, delta: String, is_final: bool) -> RuntimeAsrRealtimeEvent {
            self.text.push_str(&delta);
            RuntimeAsrRealtimeEvent {
                delta,
                text: self.text.clone(),
                is_final,
                chunk_index: self.pushes,
            }
        }

        pub(super) fn push(&mut self) -> izwi_core::Result<Vec<RuntimeAsrRealtimeEvent>> {
            if self.script.fail_pushes.load(Ordering::SeqCst) {
                return Err(izwi_core::Error::InferenceError(
                    "scripted push failure".into(),
                ));
            }
            self.pushes += 1;
            let delta = format!(
                "{}word{}",
                if self.pushes > 1 { " " } else { "" },
                self.pushes
            );
            Ok(vec![self.event(delta, false)])
        }

        pub(super) fn finish(&mut self) -> izwi_core::Result<Vec<RuntimeAsrRealtimeEvent>> {
            Ok(vec![self.event(" end.".to_string(), true)])
        }
    }

    impl Drop for ScriptedStream {
        fn drop(&mut self) {
            self.script.open.fetch_sub(1, Ordering::SeqCst);
        }
    }
}
