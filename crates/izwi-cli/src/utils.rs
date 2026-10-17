use std::path::{Path, PathBuf};

pub fn models_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("IZWI_MODELS_DIR") {
        return PathBuf::from(dir);
    }

    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("izwi")
        .join("models")
}

pub fn model_dir_if_present(variant: &str) -> Option<PathBuf> {
    let path = models_dir().join(variant);
    if path.is_dir() && directory_has_files(&path) {
        Some(path)
    } else {
        None
    }
}

pub fn directory_size_bytes(path: &Path) -> Option<u64> {
    if !path.is_dir() {
        return None;
    }

    let mut total = 0u64;
    let mut stack = vec![path.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let entries = std::fs::read_dir(&dir).ok()?;
        for entry in entries {
            let entry = entry.ok()?;
            let entry_path = entry.path();
            let meta = entry.metadata().ok()?;
            if meta.is_dir() {
                stack.push(entry_path);
            } else {
                total = total.saturating_add(meta.len());
            }
        }
    }

    Some(total)
}

fn directory_has_files(path: &Path) -> bool {
    std::fs::read_dir(path)
        .ok()
        .map(|mut entries| entries.next().is_some())
        .unwrap_or(false)
}
