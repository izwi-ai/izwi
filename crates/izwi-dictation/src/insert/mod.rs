//! Putting the finished text at the cursor of whatever app has focus.
//!
//! The default is the one every dictation app converges on: put the text on
//! the clipboard, send the paste shortcut, then give the clipboard back.
//! Typing the text as key events is the alternative for apps that block
//! pasting. When neither can work (no permission, no way to send keys on
//! this desktop) the text stays on the clipboard and the caller says so.

use serde::Serialize;

use crate::settings::InsertMethod;

#[cfg(target_os = "macos")]
#[path = "macos.rs"]
mod platform;
#[cfg(target_os = "windows")]
#[path = "windows.rs"]
mod platform;
#[cfg(target_os = "linux")]
#[path = "linux.rs"]
mod platform;

#[cfg(target_os = "macos")]
pub use platform::{paste_key_code, set_paste_key_code};

/// How long after pasting the previous clipboard is put back. Long enough
/// for a busy app to read the paste; the restore is skipped if anything
/// else wrote to the clipboard meanwhile.
pub(crate) const RESTORE_AFTER_MS: u64 = 600;
/// How long to wait for the hotkey's modifiers to be let go before sending
/// the paste shortcut: a held Alt would turn Ctrl+V into Ctrl+Alt+V.
pub(crate) const MODIFIER_WAIT_MS: u64 = 1_000;

// Each OS reports only some of these.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum InjectError {
    /// The OS will not let this process send keys (macOS Accessibility).
    #[error("Izwi is not allowed to control the keyboard.")]
    NotPermitted,
    /// The focused window belongs to an administrator process.
    #[error("The focused window runs as administrator.")]
    Elevated,
    /// This desktop offers no way to send keys.
    #[error("{0}")]
    Unsupported(String),
    #[error("{0}")]
    Failed(String),
}

/// Why the text was left on the clipboard instead of inserted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ClipboardReason {
    /// The user chose clipboard-only insertion.
    Requested,
    NotPermitted,
    Elevated,
    Unsupported,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InsertOutcome {
    Pasted,
    Typed,
    /// The text is on the clipboard; the user pastes it.
    Clipboard {
        reason: ClipboardReason,
        /// Detail for the log and the page, when there is any.
        detail: Option<String>,
    },
    /// Not even the clipboard could be written.
    Failed {
        message: String,
    },
}

/// Insert `text` into the focused app.
pub fn insert(text: &str, method: InsertMethod) -> InsertOutcome {
    let injected = match method {
        InsertMethod::Clipboard => return copy_only(text, ClipboardReason::Requested, None),
        InsertMethod::Paste => platform::paste(text).map(|()| InsertOutcome::Pasted),
        InsertMethod::Type => platform::type_text(text).map(|()| InsertOutcome::Typed),
    };
    match injected {
        Ok(outcome) => outcome,
        Err(error) => {
            let reason = match &error {
                InjectError::NotPermitted => ClipboardReason::NotPermitted,
                InjectError::Elevated => ClipboardReason::Elevated,
                InjectError::Unsupported(_) => ClipboardReason::Unsupported,
                InjectError::Failed(_) => ClipboardReason::Failed,
            };
            copy_only(text, reason, Some(error.to_string()))
        }
    }
}

fn copy_only(text: &str, reason: ClipboardReason, detail: Option<String>) -> InsertOutcome {
    match platform::copy(text) {
        Ok(()) => InsertOutcome::Clipboard { reason, detail },
        Err(message) => InsertOutcome::Failed { message },
    }
}

/// Put `text` on the clipboard and leave it there.
pub fn copy(text: &str) -> Result<(), String> {
    platform::copy(text)
}

/// Whether the OS lets Izwi send keys to other apps. Always true where no
/// permission exists (Windows, X11).
pub fn can_control_keyboard() -> bool {
    platform::can_control_keyboard()
}

/// Ask the OS to show its permission prompt (macOS Accessibility). Returns
/// whether the permission is already granted.
pub fn request_keyboard_control() -> bool {
    platform::request_keyboard_control()
}

/// The app that has keyboard focus.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize)]
pub struct FocusedApp {
    pub name: String,
    /// Bundle id on macOS, executable name on Windows, window class on Linux.
    pub id: String,
}

pub fn focused_app() -> Option<FocusedApp> {
    platform::focused_app()
}
