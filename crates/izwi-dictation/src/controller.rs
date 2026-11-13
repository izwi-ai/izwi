//! The dictation loop: hotkey inputs go in, and for each recording the audio
//! is captured, cut to the speech, transcribed, put through the text rules
//! and inserted. Everything that touches the outside world is behind
//! [`Host`], so the loop is tested with a fake.

use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::audio;
use crate::insert::InsertOutcome;
use crate::session::{Action, DiscardReason, Input, Session};
use crate::settings::DictationSettings;
use crate::text;

/// Why a recording produced no text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Discarded {
    /// The hotkey was tapped, not held.
    TooShort,
    Cancelled,
    /// Nobody spoke, or the model heard nothing.
    NoSpeech,
}

/// How long each step of a dictation took, in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
pub struct Timings {
    /// Length of the audio sent to the model.
    pub audio_ms: u64,
    /// Hotkey released to recording in hand (includes the release tail).
    pub capture_ms: u64,
    pub transcribe_ms: u64,
    /// Optional rewrite by a chat model; 0 when off.
    pub cleanup_ms: u64,
    pub insert_ms: u64,
}

/// What the indicator, the tray and the Dictation page show.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Status {
    Idle,
    /// The hotkey is down and the microphone is opening.
    Starting,
    /// Recording. `locked` means hands-free: the hotkey is not held.
    Listening {
        locked: bool,
    },
    Transcribing,
    Inserted {
        text: String,
        outcome: InsertOutcome,
        timings: Timings,
    },
    Discarded {
        reason: Discarded,
    },
    Failed {
        message: String,
    },
}

/// The outside world, as the dictation loop sees it.
pub trait Host: Send + 'static {
    /// Open the microphone and begin recording.
    fn start_capture(&mut self, settings: &DictationSettings) -> Result<(), String>;
    /// End the recording and return it as 16 kHz mono samples.
    fn finish_capture(&mut self) -> Result<Vec<f32>, String>;
    fn discard_capture(&mut self);
    /// Transcribe a clip of 16 kHz mono speech.
    fn transcribe(&mut self, clip: &[f32], settings: &DictationSettings) -> Result<String, String>;
    /// Rewrite the transcript with a chat model. `None` keeps it as it is:
    /// clean-up is off, failed or missed its deadline.
    fn clean_up(&mut self, _transcript: &str, _settings: &DictationSettings) -> Option<String> {
        None
    }
    fn insert(&mut self, text: &str, settings: &DictationSettings) -> InsertOutcome;
    fn status(&mut self, status: Status);
}

/// The loop's state, driven by [`Engine::handle`] with explicit times.
pub struct Engine<H: Host> {
    host: H,
    session: Session,
    settings: DictationSettings,
}

impl<H: Host> Engine<H> {
    pub fn new(host: H, settings: DictationSettings) -> Self {
        Self {
            host,
            session: Session::new(settings.session_config()),
            settings,
        }
    }

    pub fn host(&self) -> &H {
        &self.host
    }

    pub fn deadline(&self) -> Option<u64> {
        self.session.deadline()
    }

    pub fn is_recording(&self) -> bool {
        self.session.is_recording()
    }

    pub fn set_settings(&mut self, settings: DictationSettings) {
        self.session.set_config(settings.session_config());
        // Turning dictation off ends a recording in progress.
        if !settings.enabled && self.session.is_recording() {
            self.session.handle(Input::Cancel, 0);
            self.host.discard_capture();
            self.host.status(Status::Idle);
        }
        self.settings = settings;
    }

    /// The microphone reported its first audio: the user can speak.
    pub fn listening(&mut self) {
        if self.session.is_recording() {
            self.host.status(Status::Listening {
                locked: self.session.is_locked(),
            });
        }
    }

    pub fn handle(&mut self, input: Input, now: u64) {
        if !self.settings.enabled {
            return;
        }
        let was_locked = self.session.is_locked();
        match self.session.handle(input, now) {
            Some(Action::StartCapture) => {
                self.host.status(Status::Starting);
                if let Err(message) = self.host.start_capture(&self.settings) {
                    self.session.handle(Input::Cancel, now);
                    self.host.status(Status::Failed { message });
                    self.host.status(Status::Idle);
                }
            }
            Some(Action::DiscardCapture(reason)) => {
                self.host.discard_capture();
                self.host.status(Status::Discarded {
                    reason: match reason {
                        DiscardReason::TooShort => Discarded::TooShort,
                        DiscardReason::Cancelled => Discarded::Cancelled,
                    },
                });
                self.host.status(Status::Idle);
            }
            Some(Action::FinishCapture) => {
                let status = self.finish();
                self.host.status(status);
                self.host.status(Status::Idle);
                self.session.handle(Input::Finished, now);
            }
            None => {
                // A double tap turned the hold into a hands-free recording.
                if self.session.is_locked() && !was_locked {
                    self.host.status(Status::Listening { locked: true });
                }
            }
        }
    }

    /// Recording to inserted text. Blocks while the model runs.
    fn finish(&mut self) -> Status {
        self.host.status(Status::Transcribing);
        let mut timings = Timings::default();

        let started = Instant::now();
        let samples = match self.host.finish_capture() {
            Ok(samples) => samples,
            Err(message) => return Status::Failed { message },
        };
        timings.capture_ms = elapsed_ms(started);
        let Some(clip) = audio::speech_clip(&samples) else {
            return Status::Discarded {
                reason: Discarded::NoSpeech,
            };
        };
        timings.audio_ms = (clip.len() as u64 * 1000) / u64::from(audio::SAMPLE_RATE);

        let started = Instant::now();
        let transcript = match self.host.transcribe(&clip, &self.settings) {
            Ok(transcript) => transcript,
            Err(message) => return Status::Failed { message },
        };
        timings.transcribe_ms = elapsed_ms(started);
        if transcript.trim().is_empty() {
            return Status::Discarded {
                reason: Discarded::NoSpeech,
            };
        }

        let started = Instant::now();
        let transcript = match self.host.clean_up(&transcript, &self.settings) {
            Some(cleaned) if !cleaned.trim().is_empty() => {
                timings.cleanup_ms = elapsed_ms(started);
                cleaned
            }
            _ => transcript,
        };

        // The user's replacements run last, so they also fix a rewrite.
        let text = text::apply(&transcript, &self.settings.text_rules());
        if text.is_empty() {
            return Status::Discarded {
                reason: Discarded::NoSpeech,
            };
        }
        let started = Instant::now();
        let outcome = self.host.insert(&text, &self.settings);
        timings.insert_ms = elapsed_ms(started);
        Status::Inserted {
            text,
            outcome,
            timings,
        }
    }
}

fn elapsed_ms(since: Instant) -> u64 {
    since.elapsed().as_millis() as u64
}

enum Message {
    Input(Input),
    Listening,
    Settings(Box<DictationSettings>),
}

/// Handle to the dictation thread. Dropping every clone stops it.
#[derive(Clone)]
pub struct Controller {
    messages: Sender<Message>,
}

impl Controller {
    /// Start the dictation thread with `host` and the saved settings.
    pub fn spawn<H: Host>(host: H, settings: DictationSettings) -> Self {
        let (messages, inbox) = mpsc::channel();
        let _ = std::thread::Builder::new()
            .name("izwi-dictation".to_string())
            .spawn(move || run(Engine::new(host, settings), inbox));
        Self { messages }
    }

    pub fn send(&self, input: Input) {
        let _ = self.messages.send(Message::Input(input));
    }

    /// The microphone delivered its first audio.
    pub fn listening(&self) {
        let _ = self.messages.send(Message::Listening);
    }

    pub fn set_settings(&self, settings: DictationSettings) {
        let _ = self.messages.send(Message::Settings(Box::new(settings)));
    }
}

fn run<H: Host>(mut engine: Engine<H>, inbox: Receiver<Message>) {
    let epoch = Instant::now();
    let now = || epoch.elapsed().as_millis() as u64;
    loop {
        let message = match engine.deadline() {
            Some(deadline) => {
                let wait = Duration::from_millis(deadline.saturating_sub(now()));
                match inbox.recv_timeout(wait) {
                    Ok(message) => Some(message),
                    Err(RecvTimeoutError::Timeout) => None,
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            }
            None => match inbox.recv() {
                Ok(message) => Some(message),
                Err(_) => return,
            },
        };
        match message {
            Some(Message::Input(input)) => {
                let was_recording = engine.is_recording();
                engine.handle(input, now());
                if was_recording && !engine.is_recording() {
                    // Hotkey presses made while the text was on its way are
                    // stale; settings changes are kept.
                    let mut pending = Vec::new();
                    while let Ok(message) = inbox.try_recv() {
                        if let Message::Settings(settings) = message {
                            pending.push(settings);
                        }
                    }
                    for settings in pending {
                        engine.set_settings(*settings);
                    }
                }
            }
            Some(Message::Listening) => engine.listening(),
            Some(Message::Settings(settings)) => engine.set_settings(*settings),
            None => engine.handle(Input::Tick, now()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::insert::ClipboardReason;
    use crate::text::Replacement;

    #[derive(Default)]
    struct FakeHost {
        samples: Vec<f32>,
        transcript: String,
        cleaned: Option<String>,
        start_error: Option<String>,
        statuses: Vec<Status>,
        inserted: Vec<String>,
        capturing: bool,
        discards: usize,
    }

    impl Host for FakeHost {
        fn start_capture(&mut self, _: &DictationSettings) -> Result<(), String> {
            match self.start_error.clone() {
                Some(error) => Err(error),
                None => {
                    self.capturing = true;
                    Ok(())
                }
            }
        }

        fn finish_capture(&mut self) -> Result<Vec<f32>, String> {
            self.capturing = false;
            Ok(self.samples.clone())
        }

        fn discard_capture(&mut self) {
            self.capturing = false;
            self.discards += 1;
        }

        fn transcribe(&mut self, _: &[f32], _: &DictationSettings) -> Result<String, String> {
            Ok(self.transcript.clone())
        }

        fn clean_up(&mut self, _: &str, _: &DictationSettings) -> Option<String> {
            self.cleaned.clone()
        }

        fn insert(&mut self, text: &str, _: &DictationSettings) -> InsertOutcome {
            self.inserted.push(text.to_string());
            InsertOutcome::Pasted
        }

        fn status(&mut self, status: Status) {
            self.statuses.push(status);
        }
    }

    fn speech() -> Vec<f32> {
        (0..audio::SAMPLE_RATE as usize)
            .map(|index| 0.5 * (index as f32 * 0.09).sin())
            .collect()
    }

    fn enabled() -> DictationSettings {
        DictationSettings {
            enabled: true,
            ..DictationSettings::default()
        }
    }

    fn engine(host: FakeHost, settings: DictationSettings) -> Engine<FakeHost> {
        Engine::new(host, settings)
    }

    /// The key goes up at `at` and stays up. Where releases are debounced
    /// (X11), the release only counts at the tick that follows the debounce.
    fn release(engine: &mut Engine<FakeHost>, at: u64) {
        engine.handle(Input::Release, at);
        let debounce = engine.session.config().release_debounce_ms;
        engine.handle(Input::Tick, at + debounce);
    }

    #[test]
    fn a_hold_ends_in_inserted_text() {
        let host = FakeHost {
            samples: speech(),
            transcript: " Hello world. ".to_string(),
            ..FakeHost::default()
        };
        let mut engine = engine(host, enabled());
        engine.handle(Input::Press, 0);
        engine.listening();
        release(&mut engine, 2_000);

        let host = engine.host();
        assert_eq!(host.inserted, vec!["Hello world. "]);
        assert!(matches!(host.statuses[0], Status::Starting));
        assert_eq!(host.statuses[1], Status::Listening { locked: false });
        assert_eq!(host.statuses[2], Status::Transcribing);
        assert!(matches!(
            &host.statuses[3],
            Status::Inserted { text, outcome: InsertOutcome::Pasted, timings }
                if text == "Hello world. " && timings.audio_ms > 0
        ));
        assert_eq!(host.statuses[4], Status::Idle);
        // Ready for the next one.
        engine.handle(Input::Press, 3_000);
        assert!(engine.is_recording());
    }

    #[test]
    fn nothing_happens_while_dictation_is_off() {
        let mut engine = engine(FakeHost::default(), DictationSettings::default());
        engine.handle(Input::Press, 0);
        assert!(!engine.is_recording());
        assert!(engine.host().statuses.is_empty());
    }

    #[test]
    fn silence_and_empty_transcripts_insert_nothing() {
        let silent = FakeHost {
            samples: vec![0.0; 16_000],
            transcript: "ignored".to_string(),
            ..FakeHost::default()
        };
        let empty = FakeHost {
            samples: speech(),
            transcript: "  ".to_string(),
            ..FakeHost::default()
        };
        for host in [silent, empty] {
            let mut engine = engine(host, enabled());
            engine.handle(Input::Press, 0);
            release(&mut engine, 1_000);
            assert!(engine.host().inserted.is_empty());
            assert!(engine.host().statuses.contains(&Status::Discarded {
                reason: Discarded::NoSpeech
            }));
        }
    }

    #[test]
    fn a_tap_and_a_cancel_discard_the_recording() {
        let mut engine = engine(FakeHost::default(), enabled());
        engine.handle(Input::Press, 0);
        engine.handle(Input::Release, 100);
        engine.handle(Input::Tick, 400);
        engine.handle(Input::Press, 1_000);
        engine.handle(Input::Cancel, 1_500);
        let host = engine.host();
        assert_eq!(host.discards, 2);
        assert!(host.statuses.contains(&Status::Discarded {
            reason: Discarded::TooShort
        }));
        assert!(host.statuses.contains(&Status::Discarded {
            reason: Discarded::Cancelled
        }));
    }

    #[test]
    fn a_microphone_failure_is_reported_and_leaves_the_engine_ready() {
        let host = FakeHost {
            start_error: Some("No microphone was found.".to_string()),
            ..FakeHost::default()
        };
        let mut engine = engine(host, enabled());
        engine.handle(Input::Press, 0);
        assert!(!engine.is_recording());
        assert!(engine.host().statuses.contains(&Status::Failed {
            message: "No microphone was found.".to_string()
        }));
        assert_eq!(engine.host().statuses.last(), Some(&Status::Idle));
    }

    #[test]
    fn replacements_apply_after_the_clean_up() {
        let host = FakeHost {
            samples: speech(),
            transcript: "um open is we".to_string(),
            cleaned: Some("Open is we.".to_string()),
            ..FakeHost::default()
        };
        let settings = DictationSettings {
            replacements: vec![Replacement {
                from: "is we".to_string(),
                to: "Izwi".to_string(),
            }],
            trailing_space: false,
            ..enabled()
        };
        let mut engine = engine(host, settings);
        engine.handle(Input::Toggle, 0);
        engine.handle(Input::Toggle, 2_000);
        assert_eq!(engine.host().inserted, vec!["Open Izwi."]);
    }

    #[test]
    fn a_double_tap_reports_hands_free_listening() {
        let mut engine = engine(FakeHost::default(), enabled());
        engine.handle(Input::Press, 0);
        release(&mut engine, 100);
        engine.handle(Input::Press, 200);
        assert_eq!(
            engine.host().statuses.last(),
            Some(&Status::Listening { locked: true })
        );
    }

    #[test]
    fn turning_dictation_off_ends_a_recording() {
        let mut engine = engine(FakeHost::default(), enabled());
        engine.handle(Input::Press, 0);
        engine.set_settings(DictationSettings::default());
        assert!(!engine.is_recording());
        assert_eq!(engine.host().discards, 1);
    }

    #[test]
    fn statuses_serialize_for_the_page() {
        let status = Status::Inserted {
            text: "Hi ".to_string(),
            outcome: InsertOutcome::Clipboard {
                reason: ClipboardReason::NotPermitted,
                detail: None,
            },
            timings: Timings::default(),
        };
        let json = serde_json::to_value(&status).unwrap();
        assert_eq!(json["state"], "inserted");
        assert_eq!(json["outcome"]["kind"], "clipboard");
        assert_eq!(json["outcome"]["reason"], "not_permitted");
        assert_eq!(
            serde_json::to_value(Status::Listening { locked: true }).unwrap()["locked"],
            true
        );
    }
}
