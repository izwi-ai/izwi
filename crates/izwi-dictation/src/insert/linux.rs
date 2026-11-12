//! Linux. On X11 the paste shortcut is sent with the XTest extension. A
//! Wayland session has no one way to send keys, so the tools a desktop may
//! offer are tried in turn (`wtype`, `ydotool`, `dotool`); when none works
//! the text stays on the clipboard.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use x11rb::connection::Connection;
use x11rb::protocol::xproto::{
    AtomEnum, ConnectionExt as _, KeyButMask, KEY_PRESS_EVENT, KEY_RELEASE_EVENT,
};
use x11rb::protocol::xtest::ConnectionExt as _;
use x11rb::rust_connection::RustConnection;

use super::{FocusedApp, InjectError, MODIFIER_WAIT_MS, RESTORE_AFTER_MS};

const XK_V: u32 = 0x0076;
const XK_SHIFT_L: u32 = 0xffe1;
const XK_CONTROL_L: u32 = 0xffe3;

/// Window classes of terminals, which paste with Ctrl+Shift+V.
const TERMINALS: &[&str] = &[
    "alacritty",
    "foot",
    "gnome-terminal",
    "gnome-terminal-server",
    "kgx",
    "kitty",
    "konsole",
    "ptyxis",
    "st",
    "terminator",
    "tilix",
    "urxvt",
    "wezterm",
    "xfce4-terminal",
    "xterm",
    "ghostty",
];

/// The clipboard must stay alive to serve the text it holds: on X11 and
/// Wayland the data lives in the program that copied it.
static CLIPBOARD: Mutex<Option<arboard::Clipboard>> = Mutex::new(None);

fn with_clipboard<T>(
    action: impl FnOnce(&mut arboard::Clipboard) -> Result<T, arboard::Error>,
) -> Result<T, String> {
    let mut slot = CLIPBOARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if slot.is_none() {
        *slot = Some(arboard::Clipboard::new().map_err(|error| error.to_string())?);
    }
    let clipboard = slot.as_mut().ok_or("The clipboard is unavailable.")?;
    action(clipboard).map_err(|error| error.to_string())
}

fn is_wayland() -> bool {
    std::env::var_os("WAYLAND_DISPLAY").is_some_and(|value| !value.is_empty())
}

fn has_program(name: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|directory| directory.join(name).is_file())
    })
}

/// Run `program`, writing `input` to its stdin when given.
fn run(program: &str, arguments: &[&str], input: Option<&str>) -> Result<(), String> {
    let mut child = Command::new(program)
        .args(arguments)
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("{program}: {error}"))?;
    if let (Some(input), Some(mut stdin)) = (input, child.stdin.take()) {
        let _ = stdin.write_all(input.as_bytes());
    }
    let output = child
        .wait_with_output()
        .map_err(|error| format!("{program}: {error}"))?;
    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("{program}: {}", stderr.trim()))
    }
}

pub(super) fn can_control_keyboard() -> bool {
    true
}

pub(super) fn request_keyboard_control() -> bool {
    true
}

pub(super) fn copy(text: &str) -> Result<(), String> {
    match with_clipboard(|clipboard| clipboard.set_text(text.to_string())) {
        Ok(()) => Ok(()),
        // GNOME has no data-control protocol for background programs.
        Err(error) if is_wayland() && has_program("wl-copy") => {
            run("wl-copy", &[], Some(text)).map_err(|fallback| format!("{error}; {fallback}"))
        }
        Err(error) => Err(error),
    }
}

fn clipboard_text() -> Option<String> {
    with_clipboard(|clipboard| clipboard.get_text()).ok()
}

pub(super) fn paste(text: &str) -> Result<(), InjectError> {
    let saved = clipboard_text();
    copy(text).map_err(InjectError::Failed)?;
    if is_wayland() {
        wayland_paste()?;
    } else {
        x11_paste()?;
    }
    if let Some(saved) = saved {
        let written = text.to_string();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(RESTORE_AFTER_MS));
            // There is no change counter: restore only while the clipboard
            // still holds the dictated text.
            if clipboard_text().as_deref() == Some(written.as_str()) {
                let _ = copy(&saved);
            }
        });
    }
    Ok(())
}

pub(super) fn type_text(text: &str) -> Result<(), InjectError> {
    let typed = if is_wayland() {
        if has_program("wtype") {
            run("wtype", &["--", text], None)
        } else if has_program("ydotool") {
            run("ydotool", &["type", "--", text], None)
        } else if has_program("dotool") {
            run(
                "dotool",
                &[],
                Some(&format!("type {}\n", text.replace('\n', " "))),
            )
        } else {
            return Err(InjectError::Unsupported(
                "Typing on Wayland needs wtype, ydotool or dotool.".to_string(),
            ));
        }
    } else if has_program("xdotool") {
        run(
            "xdotool",
            &["type", "--clearmodifiers", "--delay", "2", "--", text],
            None,
        )
    } else {
        return Err(InjectError::Unsupported(
            "Typing on X11 needs xdotool.".to_string(),
        ));
    };
    typed.map_err(InjectError::Failed)
}

/// Send the paste shortcut on Wayland with whichever tool is installed.
fn wayland_paste() -> Result<(), InjectError> {
    // Keys still held from the hotkey would merge into the shortcut, and
    // Wayland offers no way to ask whether they are.
    std::thread::sleep(Duration::from_millis(250));
    let mut failures: Vec<String> = Vec::new();
    let attempts: [(&str, &[&str], Option<&str>); 3] = [
        ("wtype", &["-M", "ctrl", "-k", "v", "-m", "ctrl"], None),
        // Evdev key codes: 29 is Left Ctrl, 47 is V.
        ("ydotool", &["key", "29:1", "47:1", "47:0", "29:0"], None),
        ("dotool", &[], Some("key ctrl+v\n")),
    ];
    for (program, arguments, input) in attempts {
        if !has_program(program) {
            continue;
        }
        match run(program, arguments, input) {
            Ok(()) => return Ok(()),
            Err(error) => failures.push(error),
        }
    }
    if failures.is_empty() {
        Err(InjectError::Unsupported(
            "Pasting on Wayland needs wtype, ydotool or dotool.".to_string(),
        ))
    } else {
        Err(InjectError::Failed(failures.join("; ")))
    }
}

fn x11_error(error: impl std::fmt::Display) -> InjectError {
    InjectError::Failed(format!("X11: {error}"))
}

fn x11_paste() -> Result<(), InjectError> {
    let (connection, screen) = x11rb::connect(None).map_err(x11_error)?;
    let root = connection.setup().roots[screen].root;
    let in_terminal = active_window_class(&connection, root)
        .is_some_and(|class| TERMINALS.contains(&class.to_ascii_lowercase().as_str()));

    let control = key_code(&connection, XK_CONTROL_L)?;
    let shift = key_code(&connection, XK_SHIFT_L)?;
    let v = key_code(&connection, XK_V)?;
    wait_for_modifiers_released(&connection, root);

    let mut down = vec![control];
    if in_terminal {
        down.push(shift);
    }
    down.push(v);
    for key in &down {
        connection
            .xtest_fake_input(KEY_PRESS_EVENT, *key, 0, root, 0, 0, 0)
            .map_err(x11_error)?;
    }
    for key in down.iter().rev() {
        connection
            .xtest_fake_input(KEY_RELEASE_EVENT, *key, 0, root, 0, 0, 0)
            .map_err(x11_error)?;
    }
    connection.flush().map_err(x11_error)?;
    // Make sure the server has processed the events before disconnecting.
    connection
        .get_input_focus()
        .map_err(x11_error)?
        .reply()
        .map_err(x11_error)?;
    Ok(())
}

/// The key code whose unshifted symbol is `keysym`.
fn key_code(connection: &RustConnection, keysym: u32) -> Result<u8, InjectError> {
    let setup = connection.setup();
    let first = setup.min_keycode;
    let count = setup.max_keycode - first + 1;
    let mapping = connection
        .get_keyboard_mapping(first, count)
        .map_err(x11_error)?
        .reply()
        .map_err(x11_error)?;
    let per_code = usize::from(mapping.keysyms_per_keycode).max(1);
    mapping
        .keysyms
        .chunks(per_code)
        .position(|symbols| symbols.first() == Some(&keysym))
        .map(|index| first + index as u8)
        .ok_or_else(|| x11_error("the keyboard layout has no key for the paste shortcut"))
}

fn wait_for_modifiers_released(connection: &RustConnection, root: u32) {
    let held = KeyButMask::SHIFT | KeyButMask::CONTROL | KeyButMask::MOD1 | KeyButMask::MOD4;
    let until = Instant::now() + Duration::from_millis(MODIFIER_WAIT_MS);
    while Instant::now() < until {
        let mask = connection
            .query_pointer(root)
            .ok()
            .and_then(|cookie| cookie.reply().ok())
            .map(|reply| reply.mask);
        match mask {
            Some(mask) if mask.intersects(held) => std::thread::sleep(Duration::from_millis(10)),
            _ => return,
        }
    }
}

/// The class of the window with keyboard focus (`WM_CLASS`).
fn active_window_class(connection: &RustConnection, root: u32) -> Option<String> {
    let atom = connection
        .intern_atom(true, b"_NET_ACTIVE_WINDOW")
        .ok()?
        .reply()
        .ok()?
        .atom;
    let window = connection
        .get_property(false, root, atom, AtomEnum::WINDOW, 0, 1)
        .ok()?
        .reply()
        .ok()?
        .value32()?
        .next()?;
    let class = connection
        .get_property(false, window, AtomEnum::WM_CLASS, AtomEnum::STRING, 0, 256)
        .ok()?
        .reply()
        .ok()?
        .value;
    // "instance\0class\0": the class is the stable name.
    let mut parts = class
        .split(|byte| *byte == 0)
        .filter(|part| !part.is_empty());
    let instance = parts.next()?;
    let name = parts.next().unwrap_or(instance);
    Some(String::from_utf8_lossy(name).into_owned())
}

pub(super) fn focused_app() -> Option<FocusedApp> {
    if is_wayland() {
        // Wayland does not tell programs which window has focus.
        return None;
    }
    let (connection, screen) = x11rb::connect(None).ok()?;
    let root = connection.setup().roots[screen].root;
    let class = active_window_class(&connection, root)?;
    Some(FocusedApp {
        name: class.clone(),
        id: class,
    })
}
