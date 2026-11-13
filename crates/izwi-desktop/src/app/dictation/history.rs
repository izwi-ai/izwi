//! The text of past dictations, kept on this computer in
//! `dictation-history.jsonl` in the app's data directory: one JSON object
//! per line, oldest first. No audio is stored.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, Runtime};

const HISTORY_FILE: &str = "dictation-history.jsonl";
/// The file is cut back to `KEEP` entries once it holds `KEEP + SLACK`.
const KEEP: usize = 200;
const SLACK: usize = 50;

static WRITE_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub id: String,
    /// Milliseconds since the Unix epoch.
    pub at_ms: u64,
    pub text: String,
    /// The app that had focus when the recording started.
    pub app: Option<String>,
    /// `pasted`, `typed`, `clipboard` or `failed`.
    pub outcome: String,
    pub audio_ms: u64,
    /// Hotkey released to text inserted.
    pub total_ms: u64,
}

fn path<R: Runtime>(app: &AppHandle<R>) -> Option<PathBuf> {
    app.path()
        .app_data_dir()
        .ok()
        .map(|dir| dir.join(HISTORY_FILE))
}

fn read(path: &Path) -> Vec<HistoryEntry> {
    std::fs::read_to_string(path)
        .map(|contents| parse(&contents))
        .unwrap_or_default()
}

/// Entries oldest first; lines that do not parse are skipped.
fn parse(contents: &str) -> Vec<HistoryEntry> {
    contents
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

fn serialize(entries: &[HistoryEntry]) -> String {
    entries
        .iter()
        .filter_map(|entry| serde_json::to_string(entry).ok())
        .map(|line| line + "\n")
        .collect()
}

/// Past dictations, newest first.
pub fn load<R: Runtime>(app: &AppHandle<R>) -> Vec<HistoryEntry> {
    let mut entries = path(app).map(|path| read(&path)).unwrap_or_default();
    entries.reverse();
    entries
}

pub fn append<R: Runtime>(app: &AppHandle<R>, entry: &HistoryEntry) -> std::io::Result<()> {
    let Some(path) = path(app) else {
        return Ok(());
    };
    let _guard = WRITE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let entries = read(&path);
    if entries.len() >= KEEP + SLACK {
        let mut kept = entries[entries.len() - (KEEP - 1)..].to_vec();
        kept.push(entry.clone());
        let temp = path.with_extension("jsonl.tmp");
        std::fs::write(&temp, serialize(&kept))?;
        return std::fs::rename(temp, path);
    }
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(serialize(std::slice::from_ref(entry)).as_bytes())
}

pub fn clear<R: Runtime>(app: &AppHandle<R>) -> std::io::Result<()> {
    let Some(path) = path(app) else {
        return Ok(());
    };
    let _guard = WRITE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match std::fs::remove_file(path) {
        Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str) -> HistoryEntry {
        HistoryEntry {
            id: id.to_string(),
            at_ms: 1,
            text: "Hello\nthere".to_string(),
            app: Some("Notes".to_string()),
            outcome: "pasted".to_string(),
            audio_ms: 1_200,
            total_ms: 340,
        }
    }

    #[test]
    fn entries_round_trip_one_per_line() {
        let entries = vec![entry("a"), entry("b")];
        let contents = serialize(&entries);
        assert_eq!(contents.lines().count(), 2);
        assert_eq!(parse(&contents), entries);
    }

    #[test]
    fn damaged_lines_are_skipped() {
        let contents = format!(
            "{}not json\n{}",
            serialize(&[entry("a")]),
            serialize(&[entry("b")])
        );
        let ids: Vec<String> = parse(&contents).into_iter().map(|entry| entry.id).collect();
        assert_eq!(ids, vec!["a", "b"]);
    }
}
