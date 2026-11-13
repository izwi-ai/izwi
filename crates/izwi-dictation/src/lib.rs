//! System-wide dictation for the Izwi desktop app: hold a hotkey in any app,
//! speak, and the transcript is inserted at the cursor.
//!
//! This crate holds everything that does not need Tauri: the session rules,
//! the text rules, the settings, microphone capture and the per-OS code that
//! listens for keys and inserts text. Transcription itself is a trait the
//! desktop app implements against its local server.

pub mod audio;
#[cfg(feature = "capture")]
pub mod capture;
pub mod controller;
pub mod hotkey;
pub mod insert;
#[cfg(target_os = "linux")]
pub mod portal;
pub mod session;
pub mod settings;
pub mod shortcut;
#[cfg(feature = "capture")]
pub mod sound;
pub mod system_audio;
pub mod text;

pub use controller::{Controller, Discarded, Engine, Host, Status, Timings};
pub use session::{Action, DiscardReason, Input, Mode, Session, SessionConfig};
pub use settings::{CleanupSettings, DictationSettings, InsertMethod};
pub use shortcut::{Shortcut, ShortcutError};
pub use text::{Replacement, TextRules};
