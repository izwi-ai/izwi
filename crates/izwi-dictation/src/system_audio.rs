//! Muting the computer's sound output while a recording runs, so music or a
//! video does not end up in the transcript.

/// Sound output muted by [`mute`]; unmutes when dropped.
#[derive(Debug)]
pub struct Muted(());

/// Mute the default output. `None` when it was already muted (it must stay
/// muted afterwards) or cannot be muted on this system.
pub fn mute() -> Option<Muted> {
    if platform::is_muted()? {
        return None;
    }
    platform::set_muted(true).then_some(Muted(()))
}

impl Drop for Muted {
    fn drop(&mut self) {
        platform::set_muted(false);
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use std::process::Command;

    fn osascript(script: &str) -> Option<String> {
        let output = Command::new("osascript").args(["-e", script]).output().ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    pub fn is_muted() -> Option<bool> {
        osascript("output muted of (get volume settings)").map(|value| value == "true")
    }

    pub fn set_muted(muted: bool) -> bool {
        osascript(&format!("set volume output muted {muted}")).is_some()
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use std::process::Command;

    fn run(program: &str, arguments: &[&str]) -> Option<String> {
        let output = Command::new(program).args(arguments).output().ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).to_string())
    }

    pub fn is_muted() -> Option<bool> {
        // PipeWire first, then PulseAudio.
        if let Some(volume) = run("wpctl", &["get-volume", "@DEFAULT_AUDIO_SINK@"]) {
            return Some(volume.contains("MUTED"));
        }
        run("pactl", &["get-sink-mute", "@DEFAULT_SINK@"]).map(|mute| mute.contains("yes"))
    }

    pub fn set_muted(muted: bool) -> bool {
        let flag = if muted { "1" } else { "0" };
        run("wpctl", &["set-mute", "@DEFAULT_AUDIO_SINK@", flag]).is_some()
            || run("pactl", &["set-sink-mute", "@DEFAULT_SINK@", flag]).is_some()
    }
}

#[cfg(target_os = "windows")]
mod platform {
    use windows::Win32::Media::Audio::Endpoints::IAudioEndpointVolume;
    use windows::Win32::Media::Audio::{eConsole, eRender, IMMDeviceEnumerator, MMDeviceEnumerator};
    use windows::Win32::System::Com::{
        CoCreateInstance, CoInitializeEx, CLSCTX_ALL, COINIT_MULTITHREADED,
    };

    fn endpoint() -> Option<IAudioEndpointVolume> {
        // SAFETY: COM is initialised for this thread (a repeat call is
        // harmless); the interfaces are reference counted by the wrapper.
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            let enumerator: IMMDeviceEnumerator =
                CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL).ok()?;
            let device = enumerator.GetDefaultAudioEndpoint(eRender, eConsole).ok()?;
            device.Activate(CLSCTX_ALL, None).ok()
        }
    }

    pub fn is_muted() -> Option<bool> {
        // SAFETY: `endpoint` returned a live interface.
        unsafe { endpoint()?.GetMute().ok().map(|muted| muted.as_bool()) }
    }

    pub fn set_muted(muted: bool) -> bool {
        // SAFETY: a null event context is allowed.
        endpoint().is_some_and(|endpoint| unsafe {
            endpoint.SetMute(muted, std::ptr::null()).is_ok()
        })
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
mod platform {
    pub fn is_muted() -> Option<bool> {
        None
    }

    pub fn set_muted(_muted: bool) -> bool {
        false
    }
}
