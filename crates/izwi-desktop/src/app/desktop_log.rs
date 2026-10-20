//! The desktop app's own log, `izwi-desktop.log` in the app log directory
//! next to the managed server's `izwi-server.log`, so a user can find out why
//! the app failed. Lines also go to stderr for development runs; lines logged
//! before the directory is known are written once it is.

use std::fmt::Display;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

pub const DESKTOP_LOG_FILE: &str = "izwi-desktop.log";
/// A larger log is moved to `izwi-desktop.old.log` at launch.
const ROTATE_AT_BYTES: u64 = 2 * 1024 * 1024;
const MAX_PENDING_LINES: usize = 200;

struct Sink {
    file: Option<File>,
    pending: Vec<String>,
}

static SINK: Mutex<Sink> = Mutex::new(Sink {
    file: None,
    pending: Vec::new(),
});

/// Open the log in `dir` and write the lines logged so far.
pub fn init(dir: &Path) -> std::io::Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let path = dir.join(DESKTOP_LOG_FILE);
    if fs::metadata(&path).is_ok_and(|metadata| metadata.len() > ROTATE_AT_BYTES) {
        let _ = fs::rename(&path, path.with_extension("old.log"));
    }
    let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
    let mut sink = lock();
    for line in sink.pending.drain(..) {
        let _ = file.write_all(line.as_bytes());
    }
    sink.file = Some(file);
    Ok(path)
}

pub fn info(message: impl Display) {
    write("INFO", message);
}

pub fn warn(message: impl Display) {
    write("WARN", message);
}

pub fn error(message: impl Display) {
    write("ERROR", message);
}

/// Record panics in the log too; the default hook still prints them.
pub fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // try_lock: the panic may have happened while the log was locked.
        if let Ok(mut sink) = SINK.try_lock() {
            if let Some(file) = sink.file.as_mut() {
                let line = format_line(SystemTime::now(), "PANIC", info);
                let _ = file.write_all(line.as_bytes());
            }
        }
        default_hook(info);
    }));
}

fn write(level: &str, message: impl Display) {
    let line = format_line(SystemTime::now(), level, message);
    eprint!("{line}");
    let mut sink = lock();
    if let Some(file) = sink.file.as_mut() {
        let _ = file.write_all(line.as_bytes());
    } else if sink.pending.len() < MAX_PENDING_LINES {
        sink.pending.push(line);
    }
}

fn lock() -> MutexGuard<'static, Sink> {
    SINK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn format_line(at: SystemTime, level: &str, message: impl Display) -> String {
    format!("{} {level} {message}\n", utc_timestamp(at))
}

/// `2026-10-20T18:15:03Z`, without pulling in a date crate.
fn utc_timestamp(at: SystemTime) -> String {
    let secs = at
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    let (year, month, day) = civil_from_days((secs / 86_400) as i64);
    let time = secs % 86_400;
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        time / 3_600,
        time % 3_600 / 60,
        time % 60
    )
}

/// Days since 1970-01-01 to a proleptic Gregorian date (Howard Hinnant's
/// `civil_from_days`).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (year, month as u32, day as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn timestamps_are_utc_dates() {
        let at = |secs| UNIX_EPOCH + Duration::from_secs(secs);
        assert_eq!(utc_timestamp(at(0)), "1970-01-01T00:00:00Z");
        assert_eq!(utc_timestamp(at(951_782_400)), "2000-02-29T00:00:00Z");
        assert_eq!(utc_timestamp(at(1_792_520_103)), "2026-10-20T18:15:03Z");
    }

    #[test]
    fn lines_carry_time_level_and_message() {
        let line = format_line(UNIX_EPOCH, "WARN", "port 8080 is in use");
        assert_eq!(line, "1970-01-01T00:00:00Z WARN port 8080 is in use\n");
    }
}
