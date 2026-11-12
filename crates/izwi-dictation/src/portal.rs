//! Wayland support through the desktop portals (xdg-desktop-portal).
//!
//! A Wayland session lets no program watch the keyboard or send keys by
//! itself. Two portals stand in:
//!
//! - **GlobalShortcuts** registers the dictation hotkey with the desktop,
//!   which reports when it goes down and up (GNOME 48+, KDE Plasma,
//!   Hyprland, niri).
//! - **RemoteDesktop** lets Izwi send the paste shortcut after the user
//!   agrees once; the grant is remembered with a restore token (GNOME, KDE).
//!
//! All portal calls run on one thread with its own small async runtime.

use std::path::PathBuf;
use std::sync::mpsc as std_mpsc;
use std::sync::{Mutex, OnceLock};

use ashpd::desktop::global_shortcuts::{GlobalShortcuts, NewShortcut};
use ashpd::desktop::remote_desktop::{DeviceType, KeyState, RemoteDesktop};
use ashpd::desktop::{PersistMode, Session};
use futures_util::StreamExt;
use tokio::sync::mpsc;

use crate::hotkey::KeyEdge;
use crate::shortcut::Shortcut;

const SHORTCUT_ID: &str = "dictate";
const TOKEN_FILE: &str = "remote-desktop-token";
/// Evdev key codes.
const KEY_LEFT_CTRL: i32 = 29;
const KEY_LEFT_SHIFT: i32 = 42;
const KEY_V: i32 = 47;

type EdgeSink = Box<dyn Fn(KeyEdge) + Send + Sync>;
type Reply = std_mpsc::Sender<Result<(), String>>;

enum Command {
    Register(String),
    Bind {
        trigger: Option<String>,
        on_edge: EdgeSink,
        reply: Reply,
    },
    Unbind,
    Paste {
        with_shift: bool,
        reply: Reply,
    },
}

static COMMANDS: OnceLock<mpsc::UnboundedSender<Command>> = OnceLock::new();
static STATE_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Where the RemoteDesktop restore token is kept between runs.
pub fn set_state_dir(dir: PathBuf) {
    *STATE_DIR
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(dir);
}

fn token_path() -> Option<PathBuf> {
    STATE_DIR
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .as_ref()
        .map(|dir| dir.join(TOKEN_FILE))
}

fn commands() -> &'static mpsc::UnboundedSender<Command> {
    COMMANDS.get_or_init(|| {
        let (sender, inbox) = mpsc::unbounded_channel();
        let _ = std::thread::Builder::new()
            .name("izwi-dictation-portal".to_string())
            .spawn(move || {
                let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                else {
                    return;
                };
                runtime.block_on(run(inbox));
            });
        sender
    })
}

fn call(build: impl FnOnce(Reply) -> Command) -> Result<(), String> {
    let (reply, result) = std_mpsc::channel();
    commands()
        .send(build(reply))
        .map_err(|_| "The portal thread stopped.".to_string())?;
    result
        .recv()
        .map_err(|_| "The portal thread stopped.".to_string())?
}

/// Tell the portals which app this is. A program outside Flatpak has no app
/// id of its own, and without one the desktop cannot remember its shortcuts
/// or grants (an AppImage has no installed desktop file to go by).
pub fn register_app(app_id: &str) {
    let _ = commands().send(Command::Register(app_id.to_string()));
}

/// Register the dictation hotkey with the desktop. The desktop may show a
/// dialog, and the user may pick a different key there.
pub fn bind_shortcut(
    shortcut: &Shortcut,
    on_edge: impl Fn(KeyEdge) + Send + Sync + 'static,
) -> Result<(), String> {
    let trigger = portal_trigger(shortcut);
    call(|reply| Command::Bind {
        trigger,
        on_edge: Box::new(on_edge),
        reply,
    })
}

pub fn unbind_shortcut() {
    let _ = commands().send(Command::Unbind);
}

/// Send Ctrl+V (Ctrl+Shift+V with `with_shift`) to the focused window. The
/// first call asks the user for permission.
pub fn send_paste(with_shift: bool) -> Result<(), String> {
    call(|reply| Command::Paste { with_shift, reply })
}

/// The shortcut in the portal's notation (`CTRL+SHIFT+space`), or `None`
/// when a key has no name there: the user then sets it in the desktop's
/// dialog.
fn portal_trigger(shortcut: &Shortcut) -> Option<String> {
    let key = shortcut.key.as_deref()?;
    let name = if let Some(letter) = key.strip_prefix("Key") {
        letter.to_ascii_lowercase()
    } else if let Some(digit) = key.strip_prefix("Digit") {
        digit.to_string()
    } else {
        match key {
            "Space" => "space",
            "Enter" => "Return",
            "Tab" => "Tab",
            "Backquote" => "grave",
            "Minus" => "minus",
            "Equal" => "equal",
            "Comma" => "comma",
            "Period" => "period",
            "Slash" => "slash",
            "Semicolon" => "semicolon",
            "Pause" => "Pause",
            "Insert" => "Insert",
            other if other.starts_with('F') => other,
            _ => return None,
        }
        .to_string()
    };
    let mut parts: Vec<&str> = Vec::new();
    if shortcut.ctrl {
        parts.push("CTRL");
    }
    if shortcut.alt {
        parts.push("ALT");
    }
    if shortcut.shift {
        parts.push("SHIFT");
    }
    if shortcut.meta {
        parts.push("LOGO");
    }
    parts.push(&name);
    Some(parts.join("+"))
}

struct Shortcuts {
    session: Session<GlobalShortcuts>,
    listener: tokio::task::JoinHandle<()>,
}

struct Keyboard {
    portal: RemoteDesktop,
    session: Session<RemoteDesktop>,
}

async fn run(mut inbox: mpsc::UnboundedReceiver<Command>) {
    let mut shortcuts: Option<Shortcuts> = None;
    let mut keyboard: Option<Keyboard> = None;
    while let Some(command) = inbox.recv().await {
        match command {
            Command::Register(app_id) => {
                if let Ok(app_id) = ashpd::AppID::try_from(app_id.as_str()) {
                    // Older portals have no registry; they fall back to the
                    // desktop file, which installed packages have.
                    let _ = ashpd::register_host_app(app_id).await;
                }
            }
            Command::Bind {
                trigger,
                on_edge,
                reply,
            } => {
                if let Some(previous) = shortcuts.take() {
                    previous.listener.abort();
                    let _ = previous.session.close().await;
                }
                let bound = bind(trigger, on_edge).await;
                let _ = reply.send(match bound {
                    Ok(bound) => {
                        shortcuts = Some(bound);
                        Ok(())
                    }
                    Err(error) => Err(error),
                });
            }
            Command::Unbind => {
                if let Some(previous) = shortcuts.take() {
                    previous.listener.abort();
                    let _ = previous.session.close().await;
                }
            }
            Command::Paste { with_shift, reply } => {
                if keyboard.is_none() {
                    match open_keyboard().await {
                        Ok(opened) => keyboard = Some(opened),
                        Err(error) => {
                            let _ = reply.send(Err(error));
                            continue;
                        }
                    }
                }
                let result = match keyboard.as_ref() {
                    Some(opened) => paste(opened, with_shift).await,
                    None => Err("The keyboard portal is not open.".to_string()),
                };
                if result.is_err() {
                    // The desktop may have ended the session; open a new one
                    // next time.
                    keyboard = None;
                }
                let _ = reply.send(result);
            }
        }
    }
}

fn portal_error(what: &str, error: impl std::fmt::Display) -> String {
    format!("{what}: {error}")
}

async fn bind(trigger: Option<String>, on_edge: EdgeSink) -> Result<Shortcuts, String> {
    let unavailable = |error| portal_error("This desktop has no global shortcuts portal", error);
    let portal = GlobalShortcuts::new().await.map_err(unavailable)?;
    let session = portal
        .create_session(Default::default())
        .await
        .map_err(unavailable)?;
    let shortcut = NewShortcut::new(SHORTCUT_ID, "Dictate with Izwi (hold or press)")
        .preferred_trigger(trigger.as_deref());
    portal
        .bind_shortcuts(&session, &[shortcut], None, Default::default())
        .await
        .and_then(|request| request.response())
        .map_err(|error| portal_error("The desktop did not accept the shortcut", error))?;

    let mut pressed = portal.receive_activated().await.map_err(unavailable)?;
    let mut released = portal.receive_deactivated().await.map_err(unavailable)?;
    let listener = tokio::spawn(async move {
        loop {
            tokio::select! {
                Some(event) = pressed.next() => {
                    if event.shortcut_id() == SHORTCUT_ID {
                        on_edge(KeyEdge::Pressed);
                    }
                }
                Some(event) = released.next() => {
                    if event.shortcut_id() == SHORTCUT_ID {
                        on_edge(KeyEdge::Released);
                    }
                }
                else => break,
            }
        }
    });
    Ok(Shortcuts { session, listener })
}

async fn open_keyboard() -> Result<Keyboard, String> {
    let unavailable = |error| portal_error("This desktop has no remote desktop portal", error);
    let portal = RemoteDesktop::new().await.map_err(unavailable)?;
    let session = portal
        .create_session(Default::default())
        .await
        .map_err(unavailable)?;
    let saved = token_path().and_then(|path| std::fs::read_to_string(path).ok());
    let options = ashpd::desktop::remote_desktop::SelectDevicesOptions::default()
        .set_devices(DeviceType::KEYBOARD)
        .set_persist_mode(PersistMode::ExplicitlyRevoked)
        .set_restore_token(
            saved
                .as_deref()
                .map(str::trim)
                .filter(|token| !token.is_empty()),
        );
    portal
        .select_devices(&session, options)
        .await
        .and_then(|request| request.response())
        .map_err(unavailable)?;
    let started = portal
        .start(&session, None, Default::default())
        .await
        .and_then(|request| request.response())
        .map_err(|error| portal_error("Permission to send keys was not given", error))?;
    if let (Some(token), Some(path)) = (started.restore_token(), token_path()) {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(path, token);
    }
    Ok(Keyboard { portal, session })
}

async fn paste(keyboard: &Keyboard, with_shift: bool) -> Result<(), String> {
    let mut keys = vec![KEY_LEFT_CTRL];
    if with_shift {
        keys.push(KEY_LEFT_SHIFT);
    }
    keys.push(KEY_V);
    let presses = keys.iter().map(|key| (*key, KeyState::Pressed));
    let releases = keys.iter().rev().map(|key| (*key, KeyState::Released));
    for (key, state) in presses.chain(releases) {
        keyboard
            .portal
            .notify_keyboard_keycode(&keyboard.session, key, state, Default::default())
            .await
            .map_err(|error| portal_error("The desktop refused the key", error))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trigger(value: &str) -> Option<String> {
        portal_trigger(&value.parse().unwrap())
    }

    #[test]
    fn shortcuts_are_written_in_the_portal_notation() {
        assert_eq!(
            trigger("Ctrl+Shift+Space").as_deref(),
            Some("CTRL+SHIFT+space")
        );
        assert_eq!(trigger("Alt+KeyD").as_deref(), Some("ALT+d"));
        assert_eq!(trigger("Meta+Digit1").as_deref(), Some("LOGO+1"));
        assert_eq!(trigger("F9").as_deref(), Some("F9"));
        // No portal name: the user picks the key in the desktop's dialog.
        assert_eq!(trigger("Ctrl+BracketLeft"), None);
    }
}
