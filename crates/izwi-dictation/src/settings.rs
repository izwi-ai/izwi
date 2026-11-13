//! The user's dictation preferences, as saved by the desktop app and edited
//! on the Dictation page. Every field has a default, so settings written by
//! an older version still load.

use serde::{Deserialize, Serialize};

use crate::session::{Mode, SessionConfig};
use crate::text::{Replacement, TextRules};

/// How the finished text reaches the focused app.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InsertMethod {
    /// Put the text on the clipboard, paste it, then restore the clipboard.
    #[default]
    Paste,
    /// Type the text as key events. Slower; for apps that block pasting.
    Type,
    /// Leave the text on the clipboard for the user to paste.
    Clipboard,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CleanupSettings {
    /// Rewrite the transcript with a local chat model before inserting it.
    pub enabled: bool,
    /// Chat model to use; the smallest installed one when unset.
    pub model: Option<String>,
    /// The raw transcript is inserted if the rewrite takes longer.
    pub deadline_ms: u64,
}

impl Default for CleanupSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            model: None,
            deadline_ms: 1_500,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DictationSettings {
    pub enabled: bool,
    /// See [`crate::shortcut::Shortcut`].
    pub hotkey: String,
    pub mode: Mode,
    pub double_tap_lock: bool,
    /// Pastes the last transcript again. Unset by default.
    pub paste_last_hotkey: Option<String>,
    /// Speech model; the server's default when unset.
    pub model: Option<String>,
    /// Spoken language code; the model detects it when unset.
    pub language: Option<String>,
    /// Input device name; the system default when unset.
    pub microphone: Option<String>,
    /// Keep the microphone open between dictations so the first word is
    /// never clipped. The OS shows its microphone indicator all the time.
    pub keep_microphone_ready: bool,
    pub show_indicator: bool,
    pub sounds: bool,
    /// Mute system audio output while recording.
    pub mute_while_recording: bool,
    pub insert_method: InsertMethod,
    pub trailing_space: bool,
    pub remove_fillers: bool,
    pub replacements: Vec<Replacement>,
    /// Names and terms the speech model should expect.
    pub vocabulary: Vec<String>,
    /// Show the words in the indicator while speaking (models that stream).
    pub live_preview: bool,
    pub cleanup: CleanupSettings,
    /// Keep the text of past dictations on this computer.
    pub keep_history: bool,
}

impl Default for DictationSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            hotkey: default_hotkey().to_string(),
            mode: Mode::Hold,
            double_tap_lock: true,
            paste_last_hotkey: None,
            model: None,
            language: None,
            microphone: None,
            keep_microphone_ready: false,
            // A Linux indicator window can take focus from the app being
            // dictated into; the tray icon and sounds carry the state there.
            show_indicator: !cfg!(target_os = "linux"),
            sounds: true,
            mute_while_recording: false,
            insert_method: InsertMethod::Paste,
            trailing_space: true,
            remove_fillers: false,
            replacements: Vec::new(),
            vocabulary: Vec::new(),
            live_preview: true,
            cleanup: CleanupSettings::default(),
            keep_history: true,
        }
    }
}

pub fn default_hotkey() -> &'static str {
    if cfg!(target_os = "macos") {
        "Alt+Space"
    } else {
        "Ctrl+Shift+Space"
    }
}

impl DictationSettings {
    pub fn session_config(&self) -> SessionConfig {
        SessionConfig {
            mode: self.mode,
            double_tap_lock: self.double_tap_lock,
            // X11 repeats a held key as release and press pairs.
            release_debounce_ms: if cfg!(target_os = "linux") { 40 } else { 0 },
            ..SessionConfig::default()
        }
    }

    pub fn text_rules(&self) -> TextRules {
        TextRules {
            remove_fillers: self.remove_fillers,
            replacements: self.replacements.clone(),
            trailing_space: self.trailing_space,
        }
    }

    /// The vocabulary without blanks and repeats, in the user's order.
    pub fn vocabulary_terms(&self) -> Vec<String> {
        let mut terms: Vec<String> = Vec::new();
        for term in self.vocabulary.iter().map(|term| term.trim()) {
            if !term.is_empty() && !terms.iter().any(|seen| seen.eq_ignore_ascii_case(term)) {
                terms.push(term.to_string());
            }
        }
        terms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn older_settings_files_load_with_defaults() {
        let settings: DictationSettings =
            serde_json::from_str(r#"{"enabled": true, "mode": "toggle"}"#).unwrap();
        assert!(settings.enabled);
        assert_eq!(settings.mode, Mode::Toggle);
        assert_eq!(settings.hotkey, default_hotkey());
        assert!(settings.trailing_space && settings.keep_history);
        assert_eq!(settings.cleanup, CleanupSettings::default());
    }

    #[test]
    fn settings_round_trip_through_json() {
        let settings = DictationSettings {
            enabled: true,
            vocabulary: vec!["Izwi".into()],
            replacements: vec![Replacement {
                from: "is we".into(),
                to: "Izwi".into(),
            }],
            insert_method: InsertMethod::Type,
            ..DictationSettings::default()
        };
        let json = serde_json::to_string(&settings).unwrap();
        assert_eq!(
            serde_json::from_str::<DictationSettings>(&json).unwrap(),
            settings
        );
    }

    #[test]
    fn vocabulary_terms_are_trimmed_and_unique() {
        let settings = DictationSettings {
            vocabulary: vec![" Izwi ".into(), "".into(), "izwi".into(), "Parakeet".into()],
            ..DictationSettings::default()
        };
        assert_eq!(settings.vocabulary_terms(), vec!["Izwi", "Parakeet"]);
    }
}
