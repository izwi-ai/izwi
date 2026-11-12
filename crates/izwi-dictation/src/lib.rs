//! System-wide dictation for the Izwi desktop app: hold a hotkey in any app,
//! speak, and the transcript is inserted at the cursor.
//!
//! This crate holds everything that does not need Tauri: the session rules,
//! the text rules, the settings, microphone capture and the per-OS code that
//! listens for keys and inserts text. Transcription itself is a trait the
//! desktop app implements against its local server.

pub mod session;
pub mod settings;
pub mod shortcut;
pub mod text;

pub use session::{Action, DiscardReason, Input, Mode, Session, SessionConfig};
pub use settings::{CleanupSettings, DictationSettings, InsertMethod};
pub use shortcut::{Shortcut, ShortcutError};
pub use text::{Replacement, TextRules};
