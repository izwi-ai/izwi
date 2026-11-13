//! A hotkey as the user sets it: modifiers plus at most one key, written
//! `Ctrl+Shift+Space`. A shortcut with no key (`Alt`, `Fn`) is modifier-only
//! and needs a low-level key listener; the rest register with the OS.

use std::fmt;
use std::str::FromStr;

/// Keys are named as in the W3C `KeyboardEvent.code` list (`Space`, `KeyD`,
/// `F5`), which is what the settings page captures.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Shortcut {
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
    /// Command on macOS, the Windows key, Super on Linux.
    pub meta: bool,
    /// The Fn / Globe key (macOS only).
    pub function: bool,
    pub key: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ShortcutError {
    #[error("The shortcut is empty.")]
    Empty,
    #[error("A shortcut can have only one key besides its modifiers.")]
    TwoKeys,
    #[error("\"{0}\" is not a key Izwi can use in a shortcut.")]
    UnknownKey(String),
    #[error("Add a modifier such as Ctrl or Alt: \"{0}\" alone would stop that key from typing.")]
    NeedsModifier(String),
}

impl Shortcut {
    pub fn is_modifier_only(&self) -> bool {
        self.key.is_none()
    }

    fn modifier_count(&self) -> usize {
        [self.ctrl, self.alt, self.shift, self.meta, self.function]
            .into_iter()
            .filter(|set| *set)
            .count()
    }

    /// The form the OS hotkey registration parses (`ctrl+shift+Space`).
    /// `None` for shortcuts it cannot express: modifier-only or with Fn.
    pub fn accelerator(&self) -> Option<String> {
        if self.function {
            return None;
        }
        let key = self.key.as_deref()?;
        let mut parts: Vec<&str> = Vec::new();
        if self.ctrl {
            parts.push("ctrl");
        }
        if self.alt {
            parts.push("alt");
        }
        if self.shift {
            parts.push("shift");
        }
        if self.meta {
            parts.push("super");
        }
        parts.push(key);
        Some(parts.join("+"))
    }
}

impl fmt::Display for Shortcut {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<&str> = Vec::new();
        if self.function {
            parts.push("Fn");
        }
        if self.ctrl {
            parts.push("Ctrl");
        }
        if self.alt {
            parts.push("Alt");
        }
        if self.shift {
            parts.push("Shift");
        }
        if self.meta {
            parts.push("Meta");
        }
        if let Some(key) = &self.key {
            parts.push(key);
        }
        f.write_str(&parts.join("+"))
    }
}

impl FromStr for Shortcut {
    type Err = ShortcutError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let mut shortcut = Shortcut::default();
        for part in value
            .split('+')
            .map(str::trim)
            .filter(|part| !part.is_empty())
        {
            match part.to_ascii_lowercase().as_str() {
                "ctrl" | "control" => shortcut.ctrl = true,
                "alt" | "option" | "opt" => shortcut.alt = true,
                "shift" => shortcut.shift = true,
                "meta" | "super" | "cmd" | "command" | "win" => shortcut.meta = true,
                "fn" | "globe" => shortcut.function = true,
                _ => {
                    if shortcut.key.is_some() {
                        return Err(ShortcutError::TwoKeys);
                    }
                    shortcut.key = Some(key_code(part)?);
                }
            }
        }
        if shortcut.modifier_count() == 0 {
            match &shortcut.key {
                None => return Err(ShortcutError::Empty),
                Some(key) if !is_standalone_key(key) => {
                    return Err(ShortcutError::NeedsModifier(key.clone()))
                }
                Some(_) => {}
            }
        }
        Ok(shortcut)
    }
}

/// Keys that type nothing, so they may be a shortcut without a modifier.
fn is_standalone_key(code: &str) -> bool {
    is_function_key(code) || matches!(code, "Pause" | "ScrollLock" | "Insert" | "PrintScreen")
}

fn is_function_key(code: &str) -> bool {
    code.strip_prefix('F')
        .and_then(|number| number.parse::<u8>().ok())
        .is_some_and(|number| (1..=24).contains(&number))
}

const NAMED_KEYS: &[&str] = &[
    "Space",
    "Enter",
    "Tab",
    "Escape",
    "Backspace",
    "Delete",
    "Insert",
    "Home",
    "End",
    "PageUp",
    "PageDown",
    "ArrowUp",
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
    "Backquote",
    "Minus",
    "Equal",
    "BracketLeft",
    "BracketRight",
    "Backslash",
    "Semicolon",
    "Quote",
    "Comma",
    "Period",
    "Slash",
    "CapsLock",
    "Pause",
    "ScrollLock",
    "PrintScreen",
];

/// The `KeyboardEvent.code` name for `name`, which may be that name already
/// or a bare letter or digit.
fn key_code(name: &str) -> Result<String, ShortcutError> {
    let mut chars = name.chars();
    if let (Some(ch), None) = (chars.next(), chars.next()) {
        if ch.is_ascii_alphabetic() {
            return Ok(format!("Key{}", ch.to_ascii_uppercase()));
        }
        if ch.is_ascii_digit() {
            return Ok(format!("Digit{ch}"));
        }
    }
    if let Some(known) = NAMED_KEYS
        .iter()
        .find(|known| known.eq_ignore_ascii_case(name))
    {
        return Ok((*known).to_string());
    }
    let upper = name.to_ascii_uppercase();
    if is_function_key(&upper) {
        return Ok(upper);
    }
    let letter = name
        .strip_prefix("Key")
        .filter(|rest| rest.len() == 1 && rest.chars().all(|ch| ch.is_ascii_uppercase()));
    let digit = name
        .strip_prefix("Digit")
        .or_else(|| name.strip_prefix("Numpad"))
        .filter(|rest| rest.len() == 1 && rest.chars().all(|ch| ch.is_ascii_digit()));
    if letter.is_some() || digit.is_some() {
        return Ok(name.to_string());
    }
    Err(ShortcutError::UnknownKey(name.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(value: &str) -> Result<Shortcut, ShortcutError> {
        value.parse()
    }

    #[test]
    fn parses_and_prints_keyed_shortcuts() {
        let shortcut = parse("ctrl + Shift+space").unwrap();
        assert!(shortcut.ctrl && shortcut.shift && !shortcut.alt);
        assert_eq!(shortcut.key.as_deref(), Some("Space"));
        assert_eq!(shortcut.to_string(), "Ctrl+Shift+Space");
        assert_eq!(shortcut.accelerator().as_deref(), Some("ctrl+shift+Space"));
        assert_eq!(parse("Option+d").unwrap().to_string(), "Alt+KeyD");
        assert_eq!(parse("Cmd+Digit1").unwrap().to_string(), "Meta+Digit1");
        assert_eq!(parse("f5").unwrap().to_string(), "F5");
    }

    #[test]
    fn modifier_only_shortcuts_have_no_accelerator() {
        let shortcut = parse("Alt").unwrap();
        assert!(shortcut.is_modifier_only());
        assert_eq!(shortcut.accelerator(), None);
        assert_eq!(parse("Fn").unwrap().to_string(), "Fn");
        assert_eq!(parse("Fn+Space").unwrap().accelerator(), None);
    }

    #[test]
    fn rejects_shortcuts_that_cannot_work() {
        assert_eq!(parse(""), Err(ShortcutError::Empty));
        assert_eq!(parse("Ctrl+A+B"), Err(ShortcutError::TwoKeys));
        assert_eq!(
            parse("Space"),
            Err(ShortcutError::NeedsModifier("Space".to_string()))
        );
        assert_eq!(
            parse("Ctrl+Banana"),
            Err(ShortcutError::UnknownKey("Banana".to_string()))
        );
        assert_eq!(
            parse("Ctrl+F99"),
            Err(ShortcutError::UnknownKey("F99".to_string()))
        );
    }

    #[test]
    fn printing_round_trips() {
        for value in [
            "Ctrl+Alt+Space",
            "Fn",
            "Alt",
            "Shift+Meta+KeyV",
            "F13",
            "Ctrl+Slash",
        ] {
            let shortcut = parse(value).unwrap();
            assert_eq!(shortcut.to_string(), value);
            assert_eq!(parse(&shortcut.to_string()).unwrap(), shortcut);
        }
    }
}
