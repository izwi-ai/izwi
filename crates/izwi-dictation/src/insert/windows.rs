//! Windows: the Win32 clipboard and `SendInput`. No permission is needed,
//! but a normal process cannot send keys to an administrator's window.

use std::time::{Duration, Instant};

use windows::core::{w, PCWSTR, PWSTR};
use windows::Win32::Foundation::{CloseHandle, HANDLE, HGLOBAL};
use windows::Win32::Security::{GetTokenInformation, TokenElevation, TOKEN_ELEVATION, TOKEN_QUERY};
use windows::Win32::System::DataExchange::{
    CloseClipboard, EmptyClipboard, GetClipboardData, GetClipboardSequenceNumber, OpenClipboard,
    RegisterClipboardFormatW, SetClipboardData,
};
use windows::Win32::System::Memory::{
    GlobalAlloc, GlobalLock, GlobalSize, GlobalUnlock, GMEM_MOVEABLE,
};
use windows::Win32::System::Threading::{
    GetCurrentProcess, OpenProcess, OpenProcessToken, QueryFullProcessImageNameW,
    PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYBD_EVENT_FLAGS,
    KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, VIRTUAL_KEY, VK_CONTROL, VK_LWIN, VK_MENU, VK_RETURN,
    VK_RWIN, VK_SHIFT, VK_V,
};
use windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};

use super::{FocusedApp, InjectError, MODIFIER_WAIT_MS, RESTORE_AFTER_MS};

const CF_UNICODETEXT: u32 = 13;

pub(super) fn can_control_keyboard() -> bool {
    true
}

pub(super) fn request_keyboard_control() -> bool {
    true
}

/// Holds the clipboard open; closes it when dropped.
struct OpenedClipboard;

impl OpenedClipboard {
    fn open() -> Result<Self, String> {
        // Another program may hold the clipboard for a moment.
        for _ in 0..20 {
            // SAFETY: no owner window; paired with CloseClipboard in Drop.
            if unsafe { OpenClipboard(None) }.is_ok() {
                return Ok(Self);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        Err("The clipboard is in use by another program.".to_string())
    }
}

impl Drop for OpenedClipboard {
    fn drop(&mut self) {
        // SAFETY: this thread opened the clipboard.
        unsafe {
            let _ = CloseClipboard();
        }
    }
}

/// The clipboard's text, if it holds any.
fn read_text(_clipboard: &OpenedClipboard) -> Option<Vec<u16>> {
    // SAFETY: the clipboard is open; the handle stays valid while it is, and
    // the copy stops at the terminator or the end of the allocation.
    unsafe {
        let handle = GetClipboardData(CF_UNICODETEXT).ok()?;
        let global = HGLOBAL(handle.0);
        let pointer = GlobalLock(global).cast::<u16>();
        if pointer.is_null() {
            return None;
        }
        let capacity = GlobalSize(global) / 2;
        let mut text = Vec::new();
        for index in 0..capacity {
            let unit = *pointer.add(index);
            if unit == 0 {
                break;
            }
            text.push(unit);
        }
        let _ = GlobalUnlock(global);
        Some(text)
    }
}

/// Put `bytes` on the open clipboard under `format`.
fn set_data(format: u32, bytes: &[u8]) -> Result<(), String> {
    // SAFETY: the allocation is `bytes.len()` long and is filled before the
    // clipboard takes ownership of it; on failure it is leaked rather than
    // freed twice.
    unsafe {
        let global =
            GlobalAlloc(GMEM_MOVEABLE, bytes.len().max(1)).map_err(|error| error.to_string())?;
        let pointer = GlobalLock(global).cast::<u8>();
        if pointer.is_null() {
            return Err("The clipboard memory could not be locked.".to_string());
        }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer, bytes.len());
        let _ = GlobalUnlock(global);
        SetClipboardData(format, Some(HANDLE(global.0)))
            .map(|_| ())
            .map_err(|error| error.to_string())
    }
}

fn register_format(name: PCWSTR) -> u32 {
    // SAFETY: `name` is a NUL-terminated wide string literal.
    unsafe { RegisterClipboardFormatW(name) }
}

/// Replace the clipboard with `text`. With `transient`, tell Windows to keep
/// it out of clipboard history and cloud sync.
fn write_text(text: &[u16], transient: bool) -> Result<u32, String> {
    let clipboard = OpenedClipboard::open()?;
    // SAFETY: the clipboard is open on this thread.
    unsafe { EmptyClipboard() }.map_err(|error| error.to_string())?;
    let mut bytes: Vec<u8> = text.iter().flat_map(|unit| unit.to_le_bytes()).collect();
    bytes.extend_from_slice(&[0, 0]);
    set_data(CF_UNICODETEXT, &bytes)?;
    if transient {
        let zero = 0u32.to_le_bytes();
        for name in [
            w!("ExcludeClipboardContentFromMonitorProcessing"),
            w!("CanIncludeInClipboardHistory"),
            w!("CanUploadToCloudClipboard"),
        ] {
            let format = register_format(name);
            if format != 0 {
                let _ = set_data(format, &zero);
            }
        }
    }
    drop(clipboard);
    // SAFETY: no preconditions.
    Ok(unsafe { GetClipboardSequenceNumber() })
}

pub(super) fn copy(text: &str) -> Result<(), String> {
    let units: Vec<u16> = text.encode_utf16().collect();
    write_text(&units, false).map(|_| ())
}

pub(super) fn paste(text: &str) -> Result<(), InjectError> {
    if foreground_is_elevated() {
        return Err(InjectError::Elevated);
    }
    let saved = OpenedClipboard::open()
        .ok()
        .and_then(|clipboard| read_text(&clipboard));
    let units: Vec<u16> = text.encode_utf16().collect();
    let written = write_text(&units, true).map_err(InjectError::Failed)?;

    wait_for_modifiers_released();
    send(&[
        key(VK_CONTROL, false),
        key(VK_V, false),
        key(VK_V, true),
        key(VK_CONTROL, true),
    ])?;

    // Only text is saved: a copied image or file list is not put back.
    if let Some(saved) = saved {
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(RESTORE_AFTER_MS));
            // SAFETY: no preconditions.
            if unsafe { GetClipboardSequenceNumber() } == written {
                let _ = write_text(&saved, false);
            }
        });
    }
    Ok(())
}

pub(super) fn type_text(text: &str) -> Result<(), InjectError> {
    if foreground_is_elevated() {
        return Err(InjectError::Elevated);
    }
    wait_for_modifiers_released();
    for line in text.split_inclusive('\n') {
        let mut inputs = Vec::new();
        for unit in line.trim_end_matches('\n').encode_utf16() {
            inputs.push(unicode(unit, false));
            inputs.push(unicode(unit, true));
        }
        if line.ends_with('\n') {
            inputs.push(key(VK_RETURN, false));
            inputs.push(key(VK_RETURN, true));
        }
        for chunk in inputs.chunks(64) {
            send(chunk)?;
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    Ok(())
}

fn keyboard_input(virtual_key: VIRTUAL_KEY, scan: u16, flags: KEYBD_EVENT_FLAGS) -> INPUT {
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: virtual_key,
                wScan: scan,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

fn key(virtual_key: VIRTUAL_KEY, up: bool) -> INPUT {
    let flags = if up {
        KEYEVENTF_KEYUP
    } else {
        KEYBD_EVENT_FLAGS(0)
    };
    keyboard_input(virtual_key, 0, flags)
}

fn unicode(unit: u16, up: bool) -> INPUT {
    let flags = if up {
        KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
    } else {
        KEYEVENTF_UNICODE
    };
    keyboard_input(VIRTUAL_KEY(0), unit, flags)
}

fn send(inputs: &[INPUT]) -> Result<(), InjectError> {
    // SAFETY: `inputs` is a valid slice of INPUT of the given size.
    let sent = unsafe { SendInput(inputs, std::mem::size_of::<INPUT>() as i32) };
    if sent as usize == inputs.len() {
        Ok(())
    } else {
        Err(InjectError::Failed(
            "Windows blocked the key events.".to_string(),
        ))
    }
}

fn wait_for_modifiers_released() {
    let until = Instant::now() + Duration::from_millis(MODIFIER_WAIT_MS);
    while Instant::now() < until {
        let held = [VK_CONTROL, VK_SHIFT, VK_MENU, VK_LWIN, VK_RWIN]
            .into_iter()
            // SAFETY: no preconditions; the high bit means the key is down.
            .any(|virtual_key| unsafe { GetAsyncKeyState(i32::from(virtual_key.0)) } < 0);
        if !held {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

fn foreground_process_id() -> Option<u32> {
    // SAFETY: the window handle is only passed back to the system.
    unsafe {
        let window = GetForegroundWindow();
        if window.0.is_null() {
            return None;
        }
        let mut process_id = 0u32;
        GetWindowThreadProcessId(window, Some(&mut process_id));
        (process_id != 0).then_some(process_id)
    }
}

/// Whether `process` runs with an elevated (administrator) token.
fn is_elevated(process: HANDLE) -> Option<bool> {
    // SAFETY: the token handle is closed before returning; the buffer is a
    // TOKEN_ELEVATION of the size passed.
    unsafe {
        let mut token = HANDLE::default();
        OpenProcessToken(process, TOKEN_QUERY, &mut token).ok()?;
        let mut elevation = TOKEN_ELEVATION::default();
        let mut length = 0u32;
        let result = GetTokenInformation(
            token,
            TokenElevation,
            Some((&raw mut elevation).cast()),
            std::mem::size_of::<TOKEN_ELEVATION>() as u32,
            &mut length,
        );
        let _ = CloseHandle(token);
        result.ok()?;
        Some(elevation.TokenIsElevated != 0)
    }
}

/// True when the focused window belongs to an administrator process and
/// Izwi does not run as one: Windows drops the key events (UIPI).
fn foreground_is_elevated() -> bool {
    // SAFETY: the pseudo handle needs no closing; the opened one is closed.
    unsafe {
        if is_elevated(GetCurrentProcess()).unwrap_or(false) {
            return false;
        }
        let Some(process_id) = foreground_process_id() else {
            return false;
        };
        match OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, process_id) {
            Ok(process) => {
                let elevated = is_elevated(process).unwrap_or(false);
                let _ = CloseHandle(process);
                elevated
            }
            // Not even its name may be read: a protected or system process.
            Err(_) => true,
        }
    }
}

pub(super) fn focused_app() -> Option<FocusedApp> {
    let process_id = foreground_process_id()?;
    // SAFETY: the handle is closed; the buffer length is passed in and the
    // written length is returned in `length`.
    unsafe {
        let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, process_id).ok()?;
        let mut buffer = [0u16; 1024];
        let mut length = buffer.len() as u32;
        let result = QueryFullProcessImageNameW(
            process,
            PROCESS_NAME_WIN32,
            PWSTR(buffer.as_mut_ptr()),
            &mut length,
        );
        let _ = CloseHandle(process);
        result.ok()?;
        let path = String::from_utf16_lossy(&buffer[..length as usize]);
        let file = path.rsplit(['\\', '/']).next().unwrap_or(&path).to_string();
        let name = file.strip_suffix(".exe").unwrap_or(&file).to_string();
        Some(FocusedApp { name, id: file })
    }
}
