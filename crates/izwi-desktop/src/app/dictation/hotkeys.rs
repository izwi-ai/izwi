//! Registering the dictation hotkeys. A shortcut with a key goes to the OS
//! through the global-shortcut plugin (no permission needed, press and
//! release reported). A modifier-only shortcut uses the key listener in
//! izwi-dictation.

use izwi_dictation::hotkey::{self, KeyEdge, ModifierListener};
use izwi_dictation::{Controller, Input, Shortcut};
use tauri::{AppHandle, Runtime};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

enum Registration {
    /// The accelerator string registered with the OS.
    System(String),
    /// Listening until dropped.
    Listener(#[allow(dead_code)] ModifierListener),
    /// Bound through the Wayland desktop's shortcuts portal.
    Desktop,
}

#[derive(Default)]
pub struct Hotkeys {
    dictate: Option<Registration>,
    paste_last: Option<String>,
    /// Escape and its variants, registered only while recording.
    cancel: Vec<String>,
}

/// Why `value` cannot be a hotkey, if it cannot.
pub fn validate(value: &str) -> Result<Shortcut, String> {
    let shortcut: Shortcut = value.parse().map_err(|error| format!("{error}"))?;
    match hotkey::unsupported_reason(&shortcut) {
        Some(reason) => Err(reason),
        None => Ok(shortcut),
    }
}

fn registration_error(accelerator: &str, error: impl std::fmt::Display) -> String {
    format!("\"{accelerator}\" could not be registered ({error}). Another app may be using it.")
}

impl Hotkeys {
    pub fn clear<R: Runtime>(&mut self, app: &AppHandle<R>) {
        self.release_cancel(app);
        self.release_dictate(app);
        if let Some(accelerator) = self.paste_last.take() {
            let _ = app.global_shortcut().unregister(accelerator.as_str());
        }
    }

    fn release_dictate<R: Runtime>(&mut self, app: &AppHandle<R>) {
        match self.dictate.take() {
            Some(Registration::System(accelerator)) => {
                let _ = app.global_shortcut().unregister(accelerator.as_str());
            }
            Some(Registration::Desktop) => hotkey::unbind_desktop_shortcut(),
            Some(Registration::Listener(_)) | None => {}
        }
    }

    /// Register the dictation hotkey, replacing the previous one.
    ///
    /// On Wayland the desktop owns shortcuts. When it refuses (no portal on
    /// this desktop), dictation still works from the command line, so that
    /// is a warning (`Ok(Some(..))`), not a failure.
    pub fn set_dictate<R: Runtime>(
        &mut self,
        app: &AppHandle<R>,
        shortcut: &Shortcut,
        controller: Controller,
    ) -> Result<Option<String>, String> {
        self.release_dictate(app);
        let portal_controller = controller.clone();
        let bound = hotkey::bind_desktop_shortcut(shortcut, move |edge| {
            portal_controller.send(match edge {
                KeyEdge::Pressed => Input::Press,
                KeyEdge::Released => Input::Release,
                KeyEdge::Chord => Input::Cancel,
            });
        });
        match bound {
            Some(Ok(())) => {
                self.dictate = Some(Registration::Desktop);
                return Ok(None);
            }
            Some(Err(error)) => {
                return Ok(Some(format!(
                    "{error}. Bind a key to `izwi-desktop --dictation toggle` in your desktop's keyboard settings instead."
                )));
            }
            None => {}
        }
        let registration = match shortcut.accelerator() {
            Some(accelerator) => {
                app.global_shortcut()
                    .on_shortcut(accelerator.as_str(), move |_app, _shortcut, event| {
                        controller.send(match event.state() {
                            ShortcutState::Pressed => Input::Press,
                            ShortcutState::Released => Input::Release,
                        });
                    })
                    .map_err(|error| registration_error(&accelerator, error))?;
                Registration::System(accelerator)
            }
            None => Registration::Listener(ModifierListener::start(shortcut, move |edge| {
                controller.send(match edge {
                    KeyEdge::Pressed => Input::Press,
                    KeyEdge::Released => Input::Release,
                    KeyEdge::Chord => Input::Cancel,
                });
            })?),
        };
        self.dictate = Some(registration);
        Ok(None)
    }

    /// Register the "paste last transcript" hotkey, or remove it.
    pub fn set_paste_last<R: Runtime>(
        &mut self,
        app: &AppHandle<R>,
        shortcut: Option<&Shortcut>,
        on_press: impl Fn() + Send + Sync + 'static,
    ) -> Result<(), String> {
        if let Some(accelerator) = self.paste_last.take() {
            let _ = app.global_shortcut().unregister(accelerator.as_str());
        }
        let Some(shortcut) = shortcut else {
            return Ok(());
        };
        let accelerator = shortcut
            .accelerator()
            .ok_or("This shortcut needs a key, such as V.")?;
        app.global_shortcut()
            .on_shortcut(accelerator.as_str(), move |_app, _shortcut, event| {
                if event.state() == ShortcutState::Pressed {
                    on_press();
                }
            })
            .map_err(|error| registration_error(&accelerator, error))?;
        self.paste_last = Some(accelerator);
        Ok(())
    }

    /// While recording, Escape cancels. It is registered alone and with the
    /// hotkey's modifiers, which are still down during a hold.
    pub fn grab_cancel<R: Runtime>(
        &mut self,
        app: &AppHandle<R>,
        dictate: Option<&Shortcut>,
        controller: Controller,
    ) {
        if !self.cancel.is_empty() {
            return;
        }
        let mut accelerators = vec!["Escape".to_string()];
        let held = dictate.and_then(|shortcut| {
            Shortcut {
                key: Some("Escape".to_string()),
                function: false,
                ..shortcut.clone()
            }
            .accelerator()
        });
        if let Some(held) = held.filter(|held| held != "Escape") {
            accelerators.push(held);
        }
        for accelerator in accelerators {
            let controller = controller.clone();
            let registered = app.global_shortcut().on_shortcut(
                accelerator.as_str(),
                move |_app, _shortcut, event| {
                    if event.state() == ShortcutState::Pressed {
                        controller.send(Input::Cancel);
                    }
                },
            );
            if registered.is_ok() {
                self.cancel.push(accelerator);
            }
        }
    }

    pub fn release_cancel<R: Runtime>(&mut self, app: &AppHandle<R>) {
        for accelerator in self.cancel.drain(..) {
            let _ = app.global_shortcut().unregister(accelerator.as_str());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::validate;

    #[test]
    fn validation_explains_what_is_wrong() {
        assert!(validate("Ctrl+Shift+Space").is_ok());
        assert!(validate("").unwrap_err().contains("empty"));
        assert!(validate("Space").unwrap_err().contains("modifier"));
        assert!(validate("Fn+Space").is_err());
    }
}
