use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

use memmap2::{Mmap, MmapOptions};
use tracing::{debug, warn};

use super::types::BackendKind;
use crate::error::Result;

/// GGUF file loading mode for memory mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GgufMmapMode {
    /// Backend-aware default policy.
    #[default]
    Auto,
    /// Force mmap.
    On,
    /// Force buffered I/O.
    Off,
}

impl GgufMmapMode {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "" => None,
            "auto" => Some(Self::Auto),
            "on" | "true" | "1" | "yes" => Some(Self::On),
            "off" | "false" | "0" | "no" => Some(Self::Off),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }
}

pub fn backend_kind_for_device(device: &Device) -> BackendKind {
    if device.is_metal() {
        BackendKind::Metal
    } else if device.is_cuda() {
        BackendKind::Cuda
    } else {
        BackendKind::Cpu
    }
}

pub fn gguf_mmap_mode_from_env() -> Option<GgufMmapMode> {
    std::env::var("IZWI_GGUF_MMAP")
        .ok()
        .as_deref()
        .and_then(GgufMmapMode::parse)
}

pub fn resolve_gguf_mmap_mode(configured: Option<GgufMmapMode>) -> GgufMmapMode {
    configured
        .or_else(gguf_mmap_mode_from_env)
        .unwrap_or(GgufMmapMode::Auto)
}

/// Default mmap policy used for `auto` mode.
pub fn auto_gguf_mmap_for_backend(kind: BackendKind) -> bool {
    match kind {
        // Keep mmap enabled by default across current targets.
        BackendKind::Cpu | BackendKind::Metal | BackendKind::Cuda => true,
    }
}

pub fn gguf_mmap_enabled(mode: GgufMmapMode, kind: BackendKind) -> bool {
    match mode {
        GgufMmapMode::Auto => auto_gguf_mmap_for_backend(kind),
        GgufMmapMode::On => true,
        GgufMmapMode::Off => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufReaderKind {
    Buffered,
    Mapped,
}

pub enum GgufReader {
    Buffered(BufReader<fs::File>),
    Mapped(Cursor<Mmap>),
}

impl GgufReader {
    pub fn kind(&self) -> GgufReaderKind {
        match self {
            Self::Buffered(_) => GgufReaderKind::Buffered,
            Self::Mapped(_) => GgufReaderKind::Mapped,
        }
    }
}

impl Read for GgufReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::Buffered(reader) => reader.read(buf),
            Self::Mapped(reader) => reader.read(buf),
        }
    }
}

impl Seek for GgufReader {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match self {
            Self::Buffered(reader) => reader.seek(pos),
            Self::Mapped(reader) => reader.seek(pos),
        }
    }
}

pub fn open_gguf_reader(path: &Path, backend: BackendKind) -> Result<GgufReader> {
    open_gguf_reader_with_mode(path, None, backend)
}

pub fn open_gguf_reader_with_mode(
    path: &Path,
    configured_mode: Option<GgufMmapMode>,
    backend: BackendKind,
) -> Result<GgufReader> {
    let mode = resolve_gguf_mmap_mode(configured_mode);
    let mmap_enabled = gguf_mmap_enabled(mode, backend);
    let file = fs::File::open(path)?;

    debug!(
        "Opening GGUF file {} (backend={}, mmap_mode={}, mmap_enabled={})",
        path.display(),
        backend.as_str(),
        mode.as_str(),
        mmap_enabled
    );

    if mmap_enabled {
        // SAFETY: the mapping is read-only and the underlying file descriptor
        // remains valid for the lifetime of the mapping object.
        match unsafe { MmapOptions::new().map(&file) } {
            Ok(mmap) => return Ok(GgufReader::Mapped(Cursor::new(mmap))),
            Err(err) => {
                warn!(
                    "Failed to memory-map GGUF file {}; falling back to buffered I/O: {}",
                    path.display(),
                    err
                );
            }
        }
    }

    Ok(GgufReader::Buffered(BufReader::new(file)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    use crate::env_test_lock;

    fn temp_test_file(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!(
            "izwi-gguf-reader-{name}-{}.bin",
            uuid::Uuid::new_v4()
        ));
        let mut file = File::create(&path).expect("create temp file");
        file.write_all(bytes).expect("write temp file");
        path
    }

    #[test]
    fn parse_mode_supports_aliases() {
        assert_eq!(GgufMmapMode::parse("auto"), Some(GgufMmapMode::Auto));
        assert_eq!(GgufMmapMode::parse("on"), Some(GgufMmapMode::On));
        assert_eq!(GgufMmapMode::parse("true"), Some(GgufMmapMode::On));
        assert_eq!(GgufMmapMode::parse("1"), Some(GgufMmapMode::On));
        assert_eq!(GgufMmapMode::parse("yes"), Some(GgufMmapMode::On));
        assert_eq!(GgufMmapMode::parse("off"), Some(GgufMmapMode::Off));
        assert_eq!(GgufMmapMode::parse("false"), Some(GgufMmapMode::Off));
        assert_eq!(GgufMmapMode::parse("0"), Some(GgufMmapMode::Off));
        assert_eq!(GgufMmapMode::parse("no"), Some(GgufMmapMode::Off));
    }

    #[test]
    fn parse_mode_rejects_unknown_values() {
        assert_eq!(GgufMmapMode::parse(""), None);
        assert_eq!(GgufMmapMode::parse("invalid"), None);
    }

    #[test]
    fn resolve_mode_prefers_explicit_value_then_env_then_auto_default() {
        let _guard = env_test_lock().lock().expect("env lock poisoned");

        std::env::remove_var("IZWI_GGUF_MMAP");
        assert_eq!(resolve_gguf_mmap_mode(None), GgufMmapMode::Auto);

        std::env::set_var("IZWI_GGUF_MMAP", "off");
        assert_eq!(resolve_gguf_mmap_mode(None), GgufMmapMode::Off);

        assert_eq!(
            resolve_gguf_mmap_mode(Some(GgufMmapMode::On)),
            GgufMmapMode::On
        );

        std::env::remove_var("IZWI_GGUF_MMAP");
    }

    #[test]
    fn backend_kind_for_device_maps_cpu() {
        assert_eq!(backend_kind_for_device(&Device::Cpu), BackendKind::Cpu);
    }

    #[test]
    fn auto_policy_currently_enables_mmap_for_all_backends() {
        for backend in [BackendKind::Cpu, BackendKind::Metal, BackendKind::Cuda] {
            assert!(gguf_mmap_enabled(GgufMmapMode::Auto, backend));
        }
    }

    #[test]
    fn open_reader_uses_buffered_mode_when_explicitly_disabled() {
        let _guard = env_test_lock().lock().expect("env lock poisoned");
        std::env::remove_var("IZWI_GGUF_MMAP");

        let path = temp_test_file("buffered", b"GGUF-test-payload");
        let reader = open_gguf_reader_with_mode(&path, Some(GgufMmapMode::Off), BackendKind::Cpu)
            .expect("open reader");
        assert_eq!(reader.kind(), GgufReaderKind::Buffered);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn open_reader_supports_read_and_seek() {
        let _guard = env_test_lock().lock().expect("env lock poisoned");
        std::env::remove_var("IZWI_GGUF_MMAP");

        let path = temp_test_file("seek", b"0123456789");
        let mut reader =
            open_gguf_reader_with_mode(&path, Some(GgufMmapMode::Off), BackendKind::Cpu)
                .expect("open reader");

        let mut head = [0u8; 4];
        reader.read_exact(&mut head).expect("read head");
        assert_eq!(&head, b"0123");

        reader.seek(SeekFrom::Start(6)).expect("seek");
        let mut tail = [0u8; 2];
        reader.read_exact(&mut tail).expect("read tail");
        assert_eq!(&tail, b"67");

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn open_reader_on_mode_returns_mapped_or_safe_fallback() {
        let _guard = env_test_lock().lock().expect("env lock poisoned");
        std::env::remove_var("IZWI_GGUF_MMAP");

        let path = temp_test_file("mapped", b"GGUF");
        let reader = open_gguf_reader_with_mode(&path, Some(GgufMmapMode::On), BackendKind::Cpu)
            .expect("open reader");
        assert!(matches!(
            reader.kind(),
            GgufReaderKind::Mapped | GgufReaderKind::Buffered
        ));

        let _ = std::fs::remove_file(path);
    }
}
