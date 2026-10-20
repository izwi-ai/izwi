//! The main window's size and position, saved when it closes and restored at
//! the next launch.

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, PhysicalPosition, PhysicalSize, Runtime, WebviewWindow};

use super::settings;

const WINDOW_STATE_KEY: &str = "main_window";

/// Outer position and inner size in physical pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowGeometry {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub maximized: bool,
}

/// A monitor's rectangle in physical pixels.
#[derive(Debug, Clone, Copy)]
struct ScreenRect {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
}

pub fn load<R: Runtime>(app: &AppHandle<R>) -> Option<WindowGeometry> {
    settings::read(app, WINDOW_STATE_KEY).and_then(|value| serde_json::from_value(value).ok())
}

/// Save the window's geometry. A maximized window keeps the size it had
/// before it was maximized, so un-maximizing after a restore works.
pub fn save<R: Runtime>(window: &WebviewWindow<R>) {
    if window.is_minimized().unwrap_or(false) || window.is_fullscreen().unwrap_or(false) {
        return;
    }
    let maximized = window.is_maximized().unwrap_or(false);
    let previous = load(window.app_handle());
    let geometry = match previous {
        Some(previous) if maximized => WindowGeometry {
            maximized: true,
            ..previous
        },
        _ => {
            let (Ok(position), Ok(size)) = (window.outer_position(), window.inner_size()) else {
                return;
            };
            WindowGeometry {
                x: position.x,
                y: position.y,
                width: size.width,
                height: size.height,
                maximized,
            }
        }
    };
    if previous == Some(geometry) {
        return;
    }
    let value = serde_json::to_value(geometry).unwrap_or_default();
    if let Err(error) = settings::write(window.app_handle(), WINDOW_STATE_KEY, value) {
        super::desktop_log::warn(format!("could not save the window size: {error}"));
    }
}

/// Apply saved geometry to a window built hidden. A position on a monitor
/// that is gone is dropped, with the size, and the window is centered.
pub fn restore<R: Runtime>(window: &WebviewWindow<R>, geometry: WindowGeometry) {
    let screens: Vec<ScreenRect> = window
        .available_monitors()
        .unwrap_or_default()
        .iter()
        .map(|monitor| ScreenRect {
            x: monitor.position().x,
            y: monitor.position().y,
            width: monitor.size().width,
            height: monitor.size().height,
        })
        .collect();
    if is_on_screen(&geometry, &screens) {
        let _ = window.set_size(PhysicalSize::new(geometry.width, geometry.height));
        let _ = window.set_position(PhysicalPosition::new(geometry.x, geometry.y));
    } else {
        let _ = window.center();
    }
    if geometry.maximized {
        let _ = window.maximize();
    }
}

/// Whether the top of the window, where it is dragged from, is on a screen.
fn is_on_screen(geometry: &WindowGeometry, screens: &[ScreenRect]) -> bool {
    let grab_x = i64::from(geometry.x) + i64::from(geometry.width.min(200) / 2);
    let grab_y = i64::from(geometry.y) + 16;
    geometry.width > 0
        && geometry.height > 0
        && screens.iter().any(|screen| {
            let left = i64::from(screen.x);
            let top = i64::from(screen.y);
            (left..left + i64::from(screen.width)).contains(&grab_x)
                && (top..top + i64::from(screen.height)).contains(&grab_y)
        })
}

#[cfg(test)]
mod tests {
    use super::{is_on_screen, ScreenRect, WindowGeometry};

    const SCREENS: [ScreenRect; 2] = [
        ScreenRect {
            x: 0,
            y: 0,
            width: 2560,
            height: 1600,
        },
        ScreenRect {
            x: 2560,
            y: -200,
            width: 1920,
            height: 1080,
        },
    ];

    fn geometry(x: i32, y: i32) -> WindowGeometry {
        WindowGeometry {
            x,
            y,
            width: 1360,
            height: 900,
            maximized: false,
        }
    }

    #[test]
    fn keeps_windows_on_any_connected_screen() {
        assert!(is_on_screen(&geometry(100, 80), &SCREENS));
        assert!(is_on_screen(&geometry(3000, -100), &SCREENS));
        // Hanging off the left edge, but the title bar is still reachable.
        assert!(is_on_screen(&geometry(-50, 80), &SCREENS));
    }

    #[test]
    fn drops_positions_on_screens_that_are_gone() {
        assert!(!is_on_screen(&geometry(5000, 100), &SCREENS));
        assert!(!is_on_screen(&geometry(100, 2000), &SCREENS));
        assert!(!is_on_screen(&geometry(100, 80), &[]));
    }

    #[test]
    fn saved_geometry_round_trips_and_defaults_maximized() {
        let saved = serde_json::to_value(geometry(10, 20)).unwrap();
        assert_eq!(
            serde_json::from_value::<WindowGeometry>(saved).unwrap(),
            geometry(10, 20)
        );
        let old = serde_json::json!({"x": 1, "y": 2, "width": 3, "height": 4});
        assert!(
            !serde_json::from_value::<WindowGeometry>(old)
                .unwrap()
                .maximized
        );
    }
}
