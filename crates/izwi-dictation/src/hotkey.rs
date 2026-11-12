//! Hotkeys the OS registration cannot express: a modifier held alone
//! (`Alt`, `Ctrl+Shift`) or the Fn key. They need a low-level key listener,
//! available on macOS (an event tap, which needs the Accessibility
//! permission) and Windows (a keyboard hook).
//!
//! Shortcuts with a key go through the OS registration in the desktop app.

use crate::shortcut::Shortcut;

/// What a modifier-only hotkey did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyEdge {
    Pressed,
    Released,
    /// Another key was pressed while the modifiers were held: the user is
    /// typing a shortcut (Alt+Tab), not dictating.
    Chord,
}

/// Whether this OS can listen for modifier-only hotkeys at all.
pub fn modifier_only_supported() -> bool {
    cfg!(any(target_os = "macos", target_os = "windows"))
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
mod listener {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{mpsc, Arc};
    use std::time::Duration;

    use handy_keys::{KeyboardListener, Modifiers};

    use super::KeyEdge;
    use crate::shortcut::Shortcut;

    /// Listens until dropped.
    pub struct ModifierListener {
        running: Arc<AtomicBool>,
    }

    impl ModifierListener {
        pub fn start(
            shortcut: &Shortcut,
            on_edge: impl Fn(KeyEdge) + Send + 'static,
        ) -> Result<Self, String> {
            if !shortcut.is_modifier_only() {
                return Err("This shortcut has a key and is registered with the system.".into());
            }
            let wanted = modifiers(shortcut);
            let running = Arc::new(AtomicBool::new(true));
            let thread_running = Arc::clone(&running);
            let (ready, started) = mpsc::channel();
            std::thread::Builder::new()
                .name("izwi-dictation-keys".to_string())
                .spawn(move || {
                    let listener = match KeyboardListener::new() {
                        Ok(listener) => {
                            let _ = ready.send(Ok(()));
                            listener
                        }
                        Err(error) => {
                            let _ = ready.send(Err(error.to_string()));
                            return;
                        }
                    };
                    let mut held = false;
                    while thread_running.load(Ordering::Acquire) {
                        let Ok(event) = listener.recv_timeout(Duration::from_millis(50)) else {
                            continue;
                        };
                        if event.key.is_some() {
                            if held && event.is_key_down {
                                held = false;
                                on_edge(KeyEdge::Chord);
                            }
                            continue;
                        }
                        let matches = wanted.matches(event.modifiers);
                        if matches && !held {
                            held = true;
                            on_edge(KeyEdge::Pressed);
                        } else if !matches && held {
                            held = false;
                            on_edge(KeyEdge::Released);
                        }
                    }
                })
                .map_err(|error| error.to_string())?;
            started
                .recv()
                .map_err(|_| "The key listener stopped.".to_string())??;
            Ok(Self { running })
        }
    }

    impl Drop for ModifierListener {
        fn drop(&mut self) {
            self.running.store(false, Ordering::Release);
        }
    }

    fn modifiers(shortcut: &Shortcut) -> Modifiers {
        let mut modifiers = Modifiers::empty();
        if shortcut.ctrl {
            modifiers |= Modifiers::CTRL;
        }
        if shortcut.alt {
            modifiers |= Modifiers::OPT;
        }
        if shortcut.shift {
            modifiers |= Modifiers::SHIFT;
        }
        if shortcut.meta {
            modifiers |= Modifiers::CMD;
        }
        if shortcut.function {
            modifiers |= Modifiers::FN;
        }
        modifiers
    }
}

#[cfg(any(target_os = "macos", target_os = "windows"))]
pub use listener::ModifierListener;

/// Linux has no listener: modifier-only hotkeys are not offered there.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
pub struct ModifierListener;

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
impl ModifierListener {
    pub fn start(
        _shortcut: &Shortcut,
        _on_edge: impl Fn(KeyEdge) + Send + 'static,
    ) -> Result<Self, String> {
        Err("Shortcuts made of modifier keys alone are not available on Linux. Add a key, such as Space.".to_string())
    }
}

/// True in a Wayland session, where hotkeys go through the desktop's
/// GlobalShortcuts portal instead of an OS registration.
pub fn uses_desktop_portal() -> bool {
    cfg!(target_os = "linux")
        && std::env::var_os("WAYLAND_DISPLAY").is_some_and(|value| !value.is_empty())
}

/// Register `shortcut` with the Wayland desktop. `None` outside Wayland,
/// where the caller registers it with the OS.
pub fn bind_desktop_shortcut(
    shortcut: &Shortcut,
    on_edge: impl Fn(KeyEdge) + Send + Sync + 'static,
) -> Option<Result<(), String>> {
    if !uses_desktop_portal() {
        return None;
    }
    #[cfg(target_os = "linux")]
    return Some(crate::portal::bind_shortcut(shortcut, on_edge));
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (shortcut, on_edge);
        None
    }
}

pub fn unbind_desktop_shortcut() {
    #[cfg(target_os = "linux")]
    if uses_desktop_portal() {
        crate::portal::unbind_shortcut();
    }
}

/// One-time setup for the desktop portals: where their state is kept and
/// which app this is. Does nothing outside a Wayland session.
pub fn init_desktop_portal(state_dir: std::path::PathBuf, app_id: &str) {
    #[cfg(target_os = "linux")]
    if uses_desktop_portal() {
        crate::portal::set_state_dir(state_dir);
        crate::portal::register_app(app_id);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (state_dir, app_id);
}

/// Why `shortcut` cannot be used on this OS, if it cannot.
pub fn unsupported_reason(shortcut: &Shortcut) -> Option<String> {
    if shortcut.function && !cfg!(target_os = "macos") {
        return Some("The Fn key can only be used in shortcuts on macOS.".to_string());
    }
    if shortcut.function && !shortcut.is_modifier_only() {
        return Some("Fn can be combined with other modifiers, but not with a key.".to_string());
    }
    if shortcut.is_modifier_only() && !modifier_only_supported() {
        return Some(
            "Shortcuts made of modifier keys alone are not available on Linux. Add a key, such as Space."
                .to_string(),
        );
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fn_with_a_key_is_refused_everywhere() {
        let shortcut: Shortcut = "Fn+Space".parse().unwrap();
        assert!(unsupported_reason(&shortcut).is_some());
    }

    #[test]
    fn keyed_shortcuts_are_supported_everywhere() {
        let shortcut: Shortcut = "Ctrl+Shift+Space".parse().unwrap();
        assert_eq!(unsupported_reason(&shortcut), None);
    }

    #[test]
    fn modifier_only_follows_the_platform() {
        let shortcut: Shortcut = "Alt".parse().unwrap();
        assert_eq!(
            unsupported_reason(&shortcut).is_none(),
            modifier_only_supported()
        );
    }
}
