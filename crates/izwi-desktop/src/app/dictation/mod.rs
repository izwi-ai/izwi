//! System-wide dictation: hold a hotkey in any app, speak, and the text is
//! inserted at the cursor. The session rules, microphone capture and text
//! insertion live in `izwi-dictation`; this module connects them to the
//! app: the local server for transcription, the saved settings, the hotkey
//! registration, the indicator window, the tray and the Dictation page.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use izwi_dictation::capture::{microphones, CaptureConfig, CaptureEvent, Recorder};
use izwi_dictation::insert::{self, FocusedApp, InsertOutcome};
use izwi_dictation::sound::{self, Tone};
use izwi_dictation::system_audio::{self, Muted};
use izwi_dictation::{audio, Controller, DictationSettings, Host, Input, Shortcut, Status};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};
use url::Url;

use super::{desktop_log, settings as desktop_settings, tray};

pub mod api;
pub mod history;
mod hotkeys;
mod indicator;
mod live;

use api::{model_id, ModelStatus, ServerApi};
use history::HistoryEntry;
use hotkeys::Hotkeys;

const SETTINGS_KEY: &str = "dictation";
const STATUS_EVENT: &str = "dictation://status";
const LEVEL_EVENT: &str = "dictation://level";
const SNAPSHOT_EVENT: &str = "dictation://snapshot";
const PARTIAL_EVENT: &str = "dictation://partial";
/// How long the end of a recording waits for the live transcript before the
/// recording is transcribed the ordinary way.
const LIVE_FINISH_WAIT: Duration = Duration::from_millis(1_500);
/// How long a notice (copied to the clipboard, a failure) stays on screen.
const NOTICE_TIME: Duration = Duration::from_millis(2_600);
/// The start tone plays before the output is muted.
const MUTE_DELAY: Duration = Duration::from_millis(180);

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ModelView {
    id: String,
    /// `ready`, `loading`, `not_downloaded`, `downloading`, `error`, `unknown`.
    state: &'static str,
    message: Option<String>,
    #[serde(skip)]
    keyword_biasing: bool,
    /// The model transcribes while the audio arrives.
    #[serde(skip)]
    live: bool,
}

impl ModelView {
    fn unknown(id: &str) -> Self {
        Self {
            id: id.to_string(),
            state: "unknown",
            message: None,
            keyword_biasing: false,
            live: false,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct PlatformView {
    os: &'static str,
    wayland: bool,
    modifier_only_hotkeys: bool,
    fn_key: bool,
    default_hotkey: &'static str,
    needs_keyboard_permission: bool,
}

#[derive(Debug, Clone, Serialize)]
struct PermissionsView {
    keyboard: bool,
}

#[derive(Debug, Clone, Serialize)]
struct MicrophoneView {
    name: String,
    is_default: bool,
}

/// Everything the Dictation page shows.
#[derive(Debug, Clone, Serialize)]
pub struct Snapshot {
    settings: DictationSettings,
    status: Status,
    platform: PlatformView,
    permissions: PermissionsView,
    hotkey_error: Option<String>,
    paste_last_hotkey_error: Option<String>,
    model: ModelView,
    microphones: Vec<MicrophoneView>,
    server_is_remote: bool,
}

struct Shared {
    app: AppHandle,
    api: ServerApi,
    settings: Mutex<DictationSettings>,
    status: Mutex<Status>,
    model: Mutex<ModelView>,
    model_check_running: AtomicBool,
    hotkeys: Mutex<Hotkeys>,
    hotkey_error: Mutex<Option<String>>,
    paste_last_error: Mutex<Option<String>>,
    controller: OnceLock<Controller>,
    recorder: OnceLock<Recorder>,
    muted: Mutex<Option<Muted>>,
    recording: AtomicBool,
    last_text: Mutex<Option<String>>,
    target_app: Mutex<Option<FocusedApp>>,
    microphones: Mutex<Vec<MicrophoneView>>,
    /// The chat model clean-up uses when the settings name none.
    cleanup_model: Mutex<Option<String>>,
    /// The live session of the recording in progress, when the model streams.
    live: Mutex<Option<live::LiveSession>>,
    /// The last transcript came from the live session (for the log).
    used_live: AtomicBool,
}

/// Managed app state.
pub struct Dictation(Arc<Shared>);

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn is_wayland() -> bool {
    cfg!(target_os = "linux")
        && std::env::var_os("WAYLAND_DISPLAY").is_some_and(|value| !value.is_empty())
}

fn load_settings(app: &AppHandle) -> DictationSettings {
    desktop_settings::read(app, SETTINGS_KEY)
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default()
}

fn capture_config(settings: &DictationSettings) -> CaptureConfig {
    CaptureConfig {
        microphone: settings.microphone.clone().filter(|name| !name.is_empty()),
        // An open microphone is only kept while dictation is on.
        keep_ready: settings.enabled && settings.keep_microphone_ready,
    }
}

/// Start dictation support. Called once the server's address is known.
pub fn init(app: &AppHandle, server_url: &Url) {
    let settings = load_settings(app);
    let shared = Arc::new(Shared {
        app: app.clone(),
        api: ServerApi::new(server_url.clone()),
        model: Mutex::new(ModelView::unknown(model_id(&settings))),
        settings: Mutex::new(settings.clone()),
        status: Mutex::new(Status::Idle),
        model_check_running: AtomicBool::new(false),
        hotkeys: Mutex::new(Hotkeys::default()),
        hotkey_error: Mutex::new(None),
        paste_last_error: Mutex::new(None),
        controller: OnceLock::new(),
        recorder: OnceLock::new(),
        muted: Mutex::new(None),
        recording: AtomicBool::new(false),
        last_text: Mutex::new(None),
        target_app: Mutex::new(None),
        microphones: Mutex::new(Vec::new()),
        cleanup_model: Mutex::new(None),
        live: Mutex::new(None),
        used_live: AtomicBool::new(false),
    });

    let sink_shared = Arc::clone(&shared);
    let recorder = Recorder::new(
        capture_config(&settings),
        Arc::new(move |event| match event {
            CaptureEvent::Listening => {
                if let Some(controller) = sink_shared.controller.get() {
                    controller.listening();
                }
            }
            CaptureEvent::Level(level) => {
                let _ = sink_shared.app.emit(LEVEL_EVENT, level);
            }
            CaptureEvent::Audio(samples) => {
                // The test file stands in for the microphone (see below).
                #[cfg(debug_assertions)]
                if std::env::var_os("IZWI_DICTATION_TEST_AUDIO").is_some() {
                    return;
                }
                if let Some(session) = lock(&sink_shared.live).as_ref() {
                    session.push(samples);
                }
            }
        }),
    );
    let _ = shared.recorder.set(recorder);
    let controller = Controller::spawn(
        DesktopHost {
            shared: Arc::clone(&shared),
        },
        settings.clone(),
    );
    let _ = shared.controller.set(controller);
    app.manage(Dictation(Arc::clone(&shared)));

    if let Ok(state_dir) = app.path().app_data_dir() {
        izwi_dictation::hotkey::init_desktop_portal(state_dir, &app.config().identifier);
    }

    // Off the main thread: registering hotkeys waits on it.
    std::thread::spawn(move || {
        if let Err(error) = register_hotkeys(&shared, &settings) {
            desktop_log::warn(format!("dictation hotkey: {error}"));
            *lock(&shared.hotkey_error) = Some(error);
        }
        apply_indicator(&shared, &settings);
        check_model(&shared);
        refresh_microphones(&shared);
    });
}

fn controller(shared: &Shared) -> Option<&Controller> {
    shared.controller.get()
}

/// Register the hotkeys for `settings`. An error is about the dictation
/// hotkey; a failed paste-last hotkey is stored for the page.
fn register_hotkeys(shared: &Arc<Shared>, settings: &DictationSettings) -> Result<(), String> {
    let Some(controller) = controller(shared).cloned() else {
        return Ok(());
    };
    let mut hotkeys = lock(&shared.hotkeys);
    hotkeys.clear(&shared.app);
    *lock(&shared.paste_last_error) = None;
    *lock(&shared.hotkey_error) = None;
    if !settings.enabled {
        // Dropping the listener of a modifier-only hotkey stops it.
        *hotkeys = Hotkeys::default();
        return Ok(());
    }
    let dictate = hotkeys::validate(&settings.hotkey)?;
    let warning = hotkeys.set_dictate(&shared.app, &dictate, controller)?;
    *lock(&shared.hotkey_error) = warning;

    let paste_last = settings
        .paste_last_hotkey
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let paste_shared = Arc::clone(shared);
    let registered = match paste_last.map(hotkeys::validate).transpose() {
        Ok(shortcut) => hotkeys.set_paste_last(&shared.app, shortcut.as_ref(), move || {
            paste_last_text(&paste_shared)
        }),
        Err(error) => Err(error),
    };
    if let Err(error) = registered {
        *lock(&shared.paste_last_error) = Some(error);
    }
    Ok(())
}

fn apply_indicator(shared: &Shared, settings: &DictationSettings) {
    if settings.enabled && settings.show_indicator {
        indicator::ensure(&shared.app);
    } else {
        indicator::destroy(&shared.app);
    }
}

fn refresh_microphones(shared: &Arc<Shared>) {
    let found = microphones()
        .into_iter()
        .map(|microphone| MicrophoneView {
            name: microphone.name,
            is_default: microphone.is_default,
        })
        .collect();
    *lock(&shared.microphones) = found;
}

fn set_model(shared: &Shared, view: ModelView) {
    let changed = {
        let mut model = lock(&shared.model);
        let changed = *model != view;
        *model = view;
        changed
    };
    if changed {
        emit_snapshot(shared);
    }
}

/// Make sure the speech model is loaded and pinned, so the first dictation
/// is as fast as the rest and the model is never unloaded while idle.
fn check_model(shared: &Arc<Shared>) {
    let settings = lock(&shared.settings).clone();
    if !settings.enabled {
        return;
    }
    if shared.model_check_running.swap(true, Ordering::AcqRel) {
        return;
    }
    let shared = Arc::clone(shared);
    std::thread::spawn(move || {
        let id = model_id(&settings).to_string();
        let live = shared.api.transcribes_live(&id);
        let view =
            |state: &'static str, message: Option<String>, keyword_biasing: bool| ModelView {
                id: id.clone(),
                state,
                message,
                keyword_biasing,
                live,
            };
        let result = match shared.api.model_info(&id) {
            Err(message) => view("error", Some(message), false),
            Ok(info) => match info.status {
                ModelStatus::NotDownloaded => view("not_downloaded", None, info.keyword_biasing),
                ModelStatus::Downloading => view("downloading", None, info.keyword_biasing),
                ModelStatus::Ready if info.pinned => view("ready", None, info.keyword_biasing),
                ModelStatus::Error
                | ModelStatus::Downloaded
                | ModelStatus::Loading
                | ModelStatus::Ready => {
                    if info.status != ModelStatus::Ready {
                        set_model(&shared, view("loading", None, info.keyword_biasing));
                    }
                    match shared.api.load_pinned(&id) {
                        Ok(()) => view("ready", None, info.keyword_biasing),
                        Err(message) => view("error", Some(message), info.keyword_biasing),
                    }
                }
            },
        };
        if result.state == "error" {
            desktop_log::warn(format!(
                "dictation model {id}: {}",
                result.message.as_deref().unwrap_or("unknown error")
            ));
        }
        let loaded_now = result.state == "ready" && lock(&shared.model).state != "ready";
        set_model(&shared, result);
        if loaded_now {
            warm_model(&shared);
        }
        shared.model_check_running.store(false, Ordering::Release);
    });
}

/// Run a short clip through the speech model so the real one finds it
/// awake. The first transcription after loading takes seconds (measured: 6 s
/// against 0.25 s for a 10 s clip on Metal) and one after an idle spell takes
/// up to 1.5 s longer, so this runs after loading and when the hotkey goes
/// down: the model is warm again by the time the key is released.
fn warm_model(shared: &Arc<Shared>) {
    let model = lock(&shared.model).clone();
    if model.state != "ready" {
        return;
    }
    let shared = Arc::clone(shared);
    std::thread::spawn(move || {
        // A quiet tone, not silence: silence may be answered without the model.
        let clip: Vec<f32> = (0..audio::SAMPLE_RATE as usize / 2)
            .map(|index| 0.05 * (index as f32 * 0.0864).sin())
            .collect();
        let settings = DictationSettings {
            model: Some(model.id),
            ..DictationSettings::default()
        };
        let _ = shared
            .api
            .transcribe(audio::wav_bytes(&clip), &settings, false);
    });
}

fn snapshot(shared: &Shared) -> Snapshot {
    Snapshot {
        settings: lock(&shared.settings).clone(),
        status: lock(&shared.status).clone(),
        platform: PlatformView {
            os: if cfg!(target_os = "macos") {
                "macos"
            } else if cfg!(target_os = "windows") {
                "windows"
            } else {
                "linux"
            },
            wayland: is_wayland(),
            modifier_only_hotkeys: izwi_dictation::hotkey::modifier_only_supported(),
            fn_key: cfg!(target_os = "macos"),
            default_hotkey: izwi_dictation::settings::default_hotkey(),
            needs_keyboard_permission: cfg!(target_os = "macos"),
        },
        permissions: PermissionsView {
            keyboard: insert::can_control_keyboard(),
        },
        hotkey_error: lock(&shared.hotkey_error).clone(),
        paste_last_hotkey_error: lock(&shared.paste_last_error).clone(),
        model: lock(&shared.model).clone(),
        microphones: lock(&shared.microphones).clone(),
        server_is_remote: shared.api.is_remote(),
    }
}

fn emit_snapshot(shared: &Shared) {
    let _ = shared.app.emit(SNAPSHOT_EVENT, snapshot(shared));
}

/// Insert the last transcript again.
fn paste_last_text(shared: &Arc<Shared>) {
    let text = lock(&shared.last_text).clone().or_else(|| {
        history::load(&shared.app)
            .into_iter()
            .next()
            .map(|entry| entry.text)
    });
    let Some(text) = text else {
        return;
    };
    let method = lock(&shared.settings).insert_method;
    std::thread::spawn(move || {
        insert::insert(&text, method);
    });
}

/// Open a live session for this recording when the speech model streams
/// and the user wants to see the words as they speak.
fn start_live(shared: &Arc<Shared>, settings: &DictationSettings) -> Option<live::LiveSession> {
    let model = lock(&shared.model).clone();
    if !settings.live_preview || !model.live || model.state != "ready" {
        return None;
    }
    let language = settings
        .language
        .as_deref()
        .map(str::trim)
        .filter(|language| !language.is_empty() && !language.eq_ignore_ascii_case("auto"));
    let terms = settings.vocabulary_terms();
    let prompt = (!terms.is_empty()).then(|| terms.join(", "));
    let app = shared.app.clone();
    live::LiveSession::start(
        shared.api.base(),
        &model.id,
        language,
        prompt,
        Arc::new(move |partial| {
            let _ = app.emit(PARTIAL_EVENT, partial);
        }),
    )
}

fn unmute(shared: &Shared) {
    lock(&shared.muted).take();
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

fn outcome_name(outcome: &InsertOutcome) -> &'static str {
    match outcome {
        InsertOutcome::Pasted => "pasted",
        InsertOutcome::Typed => "typed",
        InsertOutcome::Clipboard { .. } => "clipboard",
        InsertOutcome::Failed { .. } => "failed",
    }
}

/// The app side of the dictation loop. Runs on the dictation thread.
struct DesktopHost {
    shared: Arc<Shared>,
}

impl Host for DesktopHost {
    fn start_capture(&mut self, settings: &DictationSettings) -> Result<(), String> {
        let shared = &self.shared;
        *lock(&shared.target_app) = insert::focused_app();
        #[cfg(target_os = "macos")]
        {
            // The key that types V depends on the keyboard layout, and the
            // layout may only be read on the main thread.
            let _ = shared.app.run_on_main_thread(|| {
                insert::set_paste_key_code(insert::paste_key_code());
            });
        }
        let recorder = shared
            .recorder
            .get()
            .ok_or("The microphone is not set up.")?;
        // Before the microphone opens, so the session misses no audio.
        *lock(&shared.live) = start_live(shared, settings);
        #[cfg(debug_assertions)]
        if let (Some(session), Some(samples)) = (lock(&shared.live).as_ref(), test_audio()) {
            for chunk in samples.chunks(audio::SAMPLE_RATE as usize / 10) {
                session.push(chunk.to_vec());
            }
        }
        if let Err(error) = recorder.start() {
            lock(&shared.live).take();
            return Err(error.to_string());
        }
        shared.recording.store(true, Ordering::Release);
        warm_model(shared);
        if settings.sounds {
            sound::play(Tone::Start);
        }
        if settings.mute_while_recording {
            let shared = Arc::clone(shared);
            std::thread::spawn(move || {
                std::thread::sleep(MUTE_DELAY);
                if shared.recording.load(Ordering::Acquire) {
                    let muted = system_audio::mute();
                    let mut slot = lock(&shared.muted);
                    // The recording may have ended while muting.
                    if shared.recording.load(Ordering::Acquire) {
                        *slot = muted;
                    }
                }
            });
        }
        Ok(())
    }

    fn finish_capture(&mut self) -> Result<Vec<f32>, String> {
        let shared = &self.shared;
        shared.recording.store(false, Ordering::Release);
        let recorder = shared
            .recorder
            .get()
            .ok_or("The microphone is not set up.")?;
        let samples = recorder.stop().map_err(|error| error.to_string());
        #[cfg(debug_assertions)]
        let samples = test_audio().map(Ok).unwrap_or(samples);
        unmute(shared);
        if lock(&shared.settings).sounds {
            sound::play(Tone::Stop);
        }
        samples
    }

    fn discard_capture(&mut self) {
        let shared = &self.shared;
        shared.recording.store(false, Ordering::Release);
        if let Some(recorder) = shared.recorder.get() {
            recorder.discard();
        }
        lock(&shared.live).take();
        unmute(shared);
    }

    fn transcribe(&mut self, clip: &[f32], settings: &DictationSettings) -> Result<String, String> {
        // A live session has heard the whole recording already.
        let live = lock(&self.shared.live).take();
        if let Some(session) = live {
            match session.finish(LIVE_FINISH_WAIT) {
                Some(text) => {
                    self.shared.used_live.store(true, Ordering::Release);
                    return Ok(text);
                }
                None => desktop_log::info(
                    "dictation: no live transcript in time; transcribing the recording",
                ),
            }
        }
        let keyword_biasing = lock(&self.shared.model).keyword_biasing;
        self.shared
            .api
            .transcribe(audio::wav_bytes(clip), settings, keyword_biasing)
    }

    fn clean_up(&mut self, transcript: &str, settings: &DictationSettings) -> Option<String> {
        if !settings.cleanup.enabled {
            return None;
        }
        let shared = &self.shared;
        let chosen = settings
            .cleanup
            .model
            .clone()
            .filter(|model| !model.trim().is_empty());
        let model = match chosen {
            Some(model) => model,
            None => {
                let mut cached = lock(&shared.cleanup_model);
                if cached.is_none() {
                    *cached = shared.api.default_cleanup_model();
                }
                cached.clone()?
            }
        };
        // The request runs to its end even when the deadline passes: a model
        // that had to be loaded is then ready for the next dictation.
        let (sender, receiver) = std::sync::mpsc::channel();
        let api = shared.api.clone();
        let text = transcript.to_string();
        std::thread::spawn(move || {
            let _ = sender.send(api.clean_up(&text, &model));
        });
        match receiver.recv_timeout(Duration::from_millis(settings.cleanup.deadline_ms)) {
            Ok(Ok(cleaned)) => Some(cleaned),
            Ok(Err(error)) => {
                desktop_log::warn(format!("dictation clean-up skipped: {error}"));
                None
            }
            Err(_) => {
                desktop_log::info("dictation clean-up missed its deadline; raw text inserted");
                None
            }
        }
    }

    fn insert(&mut self, text: &str, settings: &DictationSettings) -> InsertOutcome {
        *lock(&self.shared.last_text) = Some(text.to_string());
        insert::insert(text, settings.insert_method)
    }

    fn status(&mut self, status: Status) {
        let shared = &self.shared;
        let settings = lock(&shared.settings).clone();
        let recording = matches!(status, Status::Starting | Status::Listening { .. });

        // Escape cancels a recording, and only while one runs.
        {
            let mut hotkeys = lock(&shared.hotkeys);
            if recording {
                if let Some(controller) = controller(shared).cloned() {
                    let dictate: Option<Shortcut> = settings.hotkey.parse().ok();
                    hotkeys.grab_cancel(&shared.app, dictate.as_ref(), controller);
                }
            } else {
                hotkeys.release_cancel(&shared.app);
            }
        }
        tray::set_dictation_recording(&shared.app, recording);

        match &status {
            Status::Starting | Status::Listening { .. } | Status::Transcribing => {
                if settings.show_indicator {
                    indicator::show(&shared.app);
                }
            }
            Status::Inserted {
                text,
                outcome,
                timings,
            } => {
                let total_ms = timings.capture_ms
                    + timings.transcribe_ms
                    + timings.cleanup_ms
                    + timings.insert_ms;
                desktop_log::info(format!(
                    "dictation: {} ms of speech, {} ms to text ({} capture, {} transcribe{}, {} clean-up, {} insert), {}",
                    timings.audio_ms,
                    total_ms,
                    timings.capture_ms,
                    timings.transcribe_ms,
                    if shared.used_live.swap(false, Ordering::AcqRel) {
                        " live"
                    } else {
                        ""
                    },
                    timings.cleanup_ms,
                    timings.insert_ms,
                    outcome_name(outcome)
                ));
                if settings.keep_history {
                    let at_ms = now_ms();
                    let entry = HistoryEntry {
                        id: format!("{at_ms:x}"),
                        at_ms,
                        text: text.clone(),
                        app: lock(&shared.target_app)
                            .as_ref()
                            .map(|app| app.name.clone())
                            .filter(|name| !name.is_empty()),
                        outcome: outcome_name(outcome).to_string(),
                        audio_ms: timings.audio_ms,
                        total_ms,
                    };
                    if let Err(error) = history::append(&shared.app, &entry) {
                        desktop_log::warn(format!("dictation history not saved: {error}"));
                    }
                }
                // A paste needs no notice; anything else says what happened.
                let delay = match outcome {
                    InsertOutcome::Pasted | InsertOutcome::Typed => Duration::ZERO,
                    _ => NOTICE_TIME,
                };
                indicator::hide_after(&shared.app, delay);
                // The server may have restarted and lost the pin.
                check_model(shared);
            }
            Status::Discarded { reason } => {
                desktop_log::info(format!("dictation: nothing inserted ({reason:?})"));
                let delay = match reason {
                    izwi_dictation::Discarded::NoSpeech => NOTICE_TIME,
                    _ => Duration::ZERO,
                };
                indicator::hide_after(&shared.app, delay);
            }
            Status::Failed { message } => {
                desktop_log::warn(format!("dictation failed: {message}"));
                if settings.sounds {
                    sound::play(Tone::Error);
                }
                indicator::hide_after(&shared.app, NOTICE_TIME);
                check_model(shared);
            }
            // The result before it set the indicator's hide time. A live
            // session of a recording that had no speech ends here.
            Status::Idle => {
                lock(&shared.live).take();
            }
        }

        *lock(&shared.status) = status.clone();
        let _ = shared.app.emit(STATUS_EVENT, status);
    }
}

/// Development builds can dictate a WAV file (16 kHz, 16-bit, mono) named
/// by `IZWI_DICTATION_TEST_AUDIO` instead of the microphone, to test the
/// path from recording to inserted text on a machine nobody is speaking to.
#[cfg(debug_assertions)]
fn test_audio() -> Option<Vec<f32>> {
    let path = std::env::var_os("IZWI_DICTATION_TEST_AUDIO")?;
    audio::samples_from_wav(&std::fs::read(path).ok()?)
}

/// What `--dictation <action>` asks a running Izwi to do.
pub fn cli_input(arguments: &[String]) -> Option<Input> {
    let position = arguments
        .iter()
        .position(|argument| argument == "--dictation" || argument.starts_with("--dictation="))?;
    let action = match arguments[position].split_once('=') {
        Some((_, action)) => action,
        None => arguments.get(position + 1)?.as_str(),
    };
    match action {
        "start" => Some(Input::Start),
        "stop" => Some(Input::Stop),
        "toggle" => Some(Input::Toggle),
        "cancel" => Some(Input::Cancel),
        _ => None,
    }
}

/// Send `input` to the dictation loop: from the command line or the tray.
pub fn send(app: &AppHandle, input: Input) {
    if let Some(state) = app.try_state::<Dictation>() {
        if let Some(controller) = controller(&state.0) {
            controller.send(input);
        }
    }
}

pub fn is_enabled(app: &AppHandle) -> bool {
    app.try_state::<Dictation>()
        .is_some_and(|state| lock(&state.0.settings).enabled)
}

pub fn paste_last(app: &AppHandle) {
    if let Some(state) = app.try_state::<Dictation>() {
        paste_last_text(&state.0);
    }
}

// The commands run off the main thread (`async`): registering a hotkey
// waits for the main thread, which would deadlock from on it.

#[tauri::command(async)]
pub fn dictation_snapshot(state: State<'_, Dictation>) -> Snapshot {
    let shared = &state.0;
    if lock(&shared.model).state != "ready" {
        check_model(shared);
    }
    let refresh = Arc::clone(shared);
    std::thread::spawn(move || refresh_microphones(&refresh));
    snapshot(shared)
}

#[tauri::command(async)]
pub fn dictation_update_settings(
    state: State<'_, Dictation>,
    settings: DictationSettings,
) -> Result<Snapshot, String> {
    let shared = &state.0;
    hotkeys::validate(&settings.hotkey)?;
    let paste_last = settings
        .paste_last_hotkey
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if let Some(paste_last) = paste_last {
        let shortcut = hotkeys::validate(paste_last)?;
        if shortcut.is_modifier_only() {
            return Err("The paste shortcut needs a key, such as V.".to_string());
        }
        if Some(shortcut) == settings.hotkey.parse().ok() {
            return Err("The paste shortcut must differ from the dictation shortcut.".to_string());
        }
    }

    let previous = lock(&shared.settings).clone();
    let hotkeys_changed = previous.enabled != settings.enabled
        || previous.hotkey != settings.hotkey
        || previous.paste_last_hotkey != settings.paste_last_hotkey;
    if hotkeys_changed {
        if let Err(error) = register_hotkeys(shared, &settings) {
            // Keep the hotkey that worked.
            let _ = register_hotkeys(shared, &previous);
            return Err(error);
        }
    }

    *lock(&shared.settings) = settings.clone();
    let saved = serde_json::to_value(&settings)
        .map_err(|error| error.to_string())
        .and_then(|value| {
            desktop_settings::write(&shared.app, SETTINGS_KEY, value)
                .map_err(|error| error.to_string())
        });
    if let Err(error) = saved {
        desktop_log::warn(format!("dictation settings not saved: {error}"));
    }

    if let Some(recorder) = shared.recorder.get() {
        recorder.configure(capture_config(&settings));
    }
    if let Some(controller) = controller(shared) {
        controller.set_settings(settings.clone());
    }
    apply_indicator(shared, &settings);

    let old_model = model_id(&previous).to_string();
    let new_model = model_id(&settings).to_string();
    let model_changed = old_model != new_model;
    // The previous model was pinned for dictation; let the server have its
    // room back when dictation no longer uses it.
    if previous.enabled && (!settings.enabled || model_changed) {
        let api = shared.api.clone();
        std::thread::spawn(move || api.unpin(&old_model));
    }
    if model_changed || previous.enabled != settings.enabled {
        *lock(&shared.model) = ModelView::unknown(&new_model);
    }
    check_model(shared);
    Ok(snapshot(shared))
}

/// Suspend the hotkeys while the page records a new one, so pressing the
/// current hotkey does not start a dictation.
#[tauri::command(async)]
pub fn dictation_pause_hotkeys(state: State<'_, Dictation>, paused: bool) {
    let shared = &state.0;
    if paused {
        let mut hotkeys = lock(&shared.hotkeys);
        hotkeys.clear(&shared.app);
        *hotkeys = Hotkeys::default();
        return;
    }
    let settings = lock(&shared.settings).clone();
    if let Err(error) = register_hotkeys(shared, &settings) {
        *lock(&shared.hotkey_error) = Some(error);
    }
    emit_snapshot(shared);
}

#[tauri::command(async)]
pub fn dictation_request_keyboard_permission() -> bool {
    insert::request_keyboard_control()
}

#[tauri::command(async)]
pub fn dictation_open_system_settings(pane: String) -> Result<(), String> {
    open_system_settings(&pane)
}

#[cfg(target_os = "macos")]
fn open_system_settings(pane: &str) -> Result<(), String> {
    let anchor = match pane {
        "accessibility" => "Privacy_Accessibility",
        "microphone" => "Privacy_Microphone",
        _ => return Err(format!("Unknown settings pane: {pane}")),
    };
    std::process::Command::new("open")
        .arg(format!(
            "x-apple.systempreferences:com.apple.preference.security?{anchor}"
        ))
        .status()
        .map(|_| ())
        .map_err(|error| error.to_string())
}

#[cfg(target_os = "windows")]
fn open_system_settings(pane: &str) -> Result<(), String> {
    let page = match pane {
        "microphone" => "ms-settings:privacy-microphone",
        "accessibility" => return Ok(()),
        _ => return Err(format!("Unknown settings pane: {pane}")),
    };
    std::process::Command::new("explorer")
        .arg(page)
        .status()
        .map(|_| ())
        .map_err(|error| error.to_string())
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn open_system_settings(_pane: &str) -> Result<(), String> {
    Ok(())
}

#[tauri::command(async)]
pub fn dictation_toggle(state: State<'_, Dictation>) -> Result<(), String> {
    let shared = &state.0;
    if !lock(&shared.settings).enabled {
        return Err("Turn dictation on first.".to_string());
    }
    if let Some(controller) = controller(shared) {
        controller.send(Input::Toggle);
    }
    Ok(())
}

#[tauri::command(async)]
pub fn dictation_cancel(state: State<'_, Dictation>) {
    if let Some(controller) = controller(&state.0) {
        controller.send(Input::Cancel);
    }
}

#[tauri::command(async)]
pub fn dictation_history(app: AppHandle) -> Vec<HistoryEntry> {
    history::load(&app)
}

#[tauri::command(async)]
pub fn dictation_clear_history(app: AppHandle, state: State<'_, Dictation>) -> Result<(), String> {
    *lock(&state.0.last_text) = None;
    history::clear(&app).map_err(|error| error.to_string())
}

#[tauri::command(async)]
pub fn dictation_copy_text(text: String) -> Result<(), String> {
    insert::copy(&text)
}

#[tauri::command(async)]
pub fn dictation_paste_last(state: State<'_, Dictation>) {
    paste_last_text(&state.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn command_line_actions_are_read_in_both_forms() {
        assert_eq!(
            cli_input(&arguments(&["izwi-desktop", "--dictation", "toggle"])),
            Some(Input::Toggle)
        );
        assert_eq!(
            cli_input(&arguments(&["izwi-desktop", "--dictation=start"])),
            Some(Input::Start)
        );
        assert_eq!(
            cli_input(&arguments(&["izwi-desktop", "--dictation", "stop"])),
            Some(Input::Stop)
        );
        assert_eq!(
            cli_input(&arguments(&["izwi-desktop", "--dictation", "cancel"])),
            Some(Input::Cancel)
        );
        assert_eq!(cli_input(&arguments(&["izwi-desktop"])), None);
        assert_eq!(
            cli_input(&arguments(&["izwi-desktop", "--dictation"])),
            None
        );
        assert_eq!(
            cli_input(&arguments(&["izwi-desktop", "--dictation", "dance"])),
            None
        );
    }

    #[test]
    fn an_open_microphone_is_kept_only_while_dictation_is_on() {
        let mut settings = DictationSettings {
            keep_microphone_ready: true,
            microphone: Some(String::new()),
            ..DictationSettings::default()
        };
        assert!(!capture_config(&settings).keep_ready);
        assert_eq!(capture_config(&settings).microphone, None);
        settings.enabled = true;
        assert!(capture_config(&settings).keep_ready);
    }
}
