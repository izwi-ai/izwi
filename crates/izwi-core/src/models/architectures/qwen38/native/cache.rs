//! Disposable, versioned pre-initialization Q8 tiles. No source file is changed.
//! A global OS lock serializes eviction/publication and is released on crash.
//! Every entry authenticates source, converter ABI, shape, packing and payload.
use super::*;
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::sync::Mutex;

const MAGIC: &[u8; 8] = b"IZQ8v001";
const HEADER: usize = 80; // magic, u64 payload size, source key, payload sha256
pub(super) const ABI: &[u8] =
    b"izwi-qwen38-q8_0-v1;le;f32-scale-multiply;round-away;preinit;row-concat;candle-0.11";

#[derive(Debug)]
pub(super) struct DerivedCache {
    dir: PathBuf,
    max: u64,
    writer: Mutex<Option<Writer>>,
}
#[derive(Debug)]
struct Writer {
    _lock: File,
    used: u64,
    sizes: BTreeMap<PathBuf, u64>,
    oldest: VecDeque<PathBuf>,
    directory: PathBuf,
}
impl Drop for Writer {
    fn drop(&mut self) {
        // Cache is disposable: a crash may lose the latest rename, never expose
        // partial bytes. Amortize directory durability across the load session.
        #[cfg(unix)]
        if let Ok(dir) = File::open(&self.directory) {
            let _ = dir.sync_all();
        }
    }
}
impl DerivedCache {
    pub fn new(options: &LoadingPerformanceConfig) -> Option<Self> {
        if !options.enabled()
            || !options.derived_weight_cache.enabled()
            || options.cache_max_bytes <= HEADER as u64
        {
            return None;
        }
        let root = options
            .cache_dir
            .clone()
            .or_else(|| dirs::cache_dir().map(|p| p.join("izwi/derived-weights")))?;
        let cache = Self {
            dir: root.join("qwen38-q8-v1"),
            max: options.cache_max_bytes,
            writer: Mutex::new(None),
        };
        if let Ok(mut writer) = cache.writer.lock() {
            let _ = cache.prepare_writer(&mut writer);
        }
        Some(cache)
    }
    fn path(&self, key: &[u8; 32]) -> PathBuf {
        self.dir.join(format!("{}.q8", hex(key)))
    }
    pub fn read(&self, key: &[u8; 32], words: usize) -> Option<Vec<u16>> {
        self.read_checked(key, words).ok()
    }
    fn read_checked(&self, key: &[u8; 32], words: usize) -> std::io::Result<Vec<u16>> {
        let bytes = words.checked_mul(2).ok_or_else(invalid)?;
        let length = bytes.checked_add(HEADER).ok_or_else(invalid)?;
        if length as u64 > self.max {
            return Err(invalid());
        }
        let path = self.path(key);
        if !fs::symlink_metadata(&path)?.is_file() {
            return Err(invalid());
        }
        let mut f = File::open(path)?;
        if f.metadata()?.len() != length as u64 {
            return Err(invalid());
        }
        let mut header = [0u8; HEADER];
        f.read_exact(&mut header)?;
        if &header[..8] != MAGIC
            || u64::from_le_bytes(header[8..16].try_into().unwrap()) != bytes as u64
            || &header[16..48] != key
        {
            return Err(invalid());
        }
        // u16 staging explicitly satisfies Candle BlockQ8_0's two-byte alignment.
        let mut output = vec![0u16; words];
        let raw = words_bytes_mut(&mut output);
        f.read_exact(raw)?;
        if Sha256::digest(&*raw).as_slice() != &header[48..80] {
            return Err(invalid());
        }
        // Reject invalid block scales even if a manually generated entry has a
        // matching checksum. Never reach Candle rawdata with malformed layout.
        validate_q8(&output).map_err(|_| invalid())?;
        Ok(output)
    }
    pub fn publish(&self, key: &[u8; 32], words: &[u16]) {
        if let Err(e) = self.publish_checked(key, words) {
            tracing::debug!(error=%e,"Derived Q8 cache write skipped; source conversion remains usable");
        }
    }
    fn prepare_writer(&self, session: &mut Option<Writer>) -> std::io::Result<()> {
        if session.is_none() {
            fs::create_dir_all(&self.dir)?;
            let lock = fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .open(self.dir.join("writer.lock"))?;
            // The writer is retained for this entire load; competing loaders
            // proceed with conversion/read hits instead of waiting.
            if lock.try_lock().is_err() {
                return Ok(());
            }
            let mut files = Vec::new();
            let mut used = 0u64;
            for entry in fs::read_dir(&self.dir)? {
                let path = entry?.path();
                let m = fs::symlink_metadata(&path)?;
                if !m.is_file() {
                    continue;
                }
                match path.extension().and_then(|e| e.to_str()) {
                    Some("tmp") => {
                        fs::remove_file(path)?;
                    }
                    Some("q8") => {
                        if files.len() >= 65536 {
                            fs::remove_file(path)?;
                            continue;
                        }
                        used = used.checked_add(m.len()).ok_or_else(invalid)?;
                        files.push((m.modified().ok(), path, m.len()));
                    }
                    _ => {}
                }
            }
            files.sort_by(|a, b| a.0.cmp(&b.0));
            let sizes = files.iter().map(|(_, p, n)| (p.clone(), *n)).collect();
            let oldest = files.into_iter().map(|(_, p, _)| p).collect();
            *session = Some(Writer {
                _lock: lock,
                used,
                sizes,
                oldest,
                directory: self.dir.clone(),
            });
        }
        if let Some(writer) = session.as_mut() {
            while writer.used > self.max {
                let Some(path) = writer.oldest.pop_front() else {
                    break;
                };
                if let Some(len) = writer.sizes.get(&path).copied() {
                    fs::remove_file(&path)?;
                    writer.sizes.remove(&path);
                    writer.used = writer.used.saturating_sub(len);
                }
            }
        }
        Ok(())
    }
    fn publish_checked(&self, key: &[u8; 32], words: &[u16]) -> std::io::Result<()> {
        validate_q8(words).map_err(|_| invalid())?;
        let payload = words_bytes(words);
        let size = payload.len().checked_add(HEADER).ok_or_else(invalid)? as u64;
        if size > self.max {
            return Ok(());
        }
        let mut session = self.writer.lock().map_err(|_| invalid())?;
        self.prepare_writer(&mut session)?;
        let Some(writer) = session.as_mut() else {
            return Ok(());
        };
        while writer.used > self.max.saturating_sub(size) || writer.sizes.len() >= 65536 {
            let Some(path) = writer.oldest.pop_front() else {
                return Ok(());
            };
            if let Some(len) = writer.sizes.get(&path).copied() {
                // Retain accounting if removal fails (e.g. read-only filesystem).
                fs::remove_file(&path)?;
                writer.sizes.remove(&path);
                writer.used = writer.used.saturating_sub(len);
            }
        }
        let temp = Temp(self.dir.join(format!("{}.tmp", uuid::Uuid::new_v4())));
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp.0)?;
        file.write_all(MAGIC)?;
        file.write_all(&(payload.len() as u64).to_le_bytes())?;
        file.write_all(key)?;
        file.write_all(&Sha256::digest(payload))?;
        file.write_all(payload)?;
        file.sync_all()?;
        drop(file);
        let destination = self.path(key);
        fs::rename(&temp.0, &destination)?;
        let replaced = writer.sizes.insert(destination.clone(), size).unwrap_or(0);
        writer.used = writer.used.saturating_sub(replaced) + size;
        // Keep one eviction queue entry per path, including corruption rebuilds.
        if replaced > 0 {
            writer.oldest.retain(|p| p != &destination);
        }
        writer.oldest.push_back(destination);
        Ok(())
    }
}
struct Temp(PathBuf);
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
fn invalid() -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid Q8 cache entry")
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub(super) fn words_bytes(words: &[u16]) -> &[u8] {
    // SAFETY: a byte slice may inspect any initialized u16 representation.
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), std::mem::size_of_val(words)) }
}
fn words_bytes_mut(words: &mut [u16]) -> &mut [u8] {
    // SAFETY: all u16 bit patterns are valid, exclusive borrow covers full slice.
    unsafe {
        std::slice::from_raw_parts_mut(words.as_mut_ptr().cast(), std::mem::size_of_val(words))
    }
}
pub(super) fn validate_q8(words: &[u16]) -> Result<()> {
    if !cfg!(target_endian = "little")
        || words.is_empty()
        || !words.len().is_multiple_of(17)
        || !(words.as_ptr() as usize).is_multiple_of(2)
    {
        return Err(Error::ModelLoadError(
            "Invalid aligned little-endian Q8_0 payload".into(),
        ));
    }
    for block in words.chunks_exact(17) {
        let d = f16::from_bits(block[0]).to_f32();
        if !d.is_finite() || d < 0. {
            return Err(Error::ModelLoadError("Invalid Q8_0 scale".into()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::TestDir;
    use super::*;
    fn cache(dir: &Path, max: u64) -> DerivedCache {
        DerivedCache {
            dir: dir.to_owned(),
            max,
            writer: Mutex::new(None),
        }
    }
    #[test]
    fn cache_hits_corruption_and_source_identity() {
        let d = TestDir::new("derived-cache");
        let c = cache(d.path(), 10000);
        let key = [3; 32];
        let words = vec![0u16; 34];
        c.publish(&key, &words);
        assert_eq!(c.read(&key, 34), Some(words.clone()));
        assert!(c.read(&[4; 32], 34).is_none());
        let p = c.path(&key);
        let mut b = fs::read(&p).unwrap();
        b[HEADER + 2] ^= 1;
        fs::write(&p, &b).unwrap();
        assert!(c.read(&key, 34).is_none());
        c.publish(&key, &words);
        assert_eq!(c.read(&key, 34), Some(words));
        fs::write(&p, &b[..HEADER + 1]).unwrap();
        assert!(c.read(&key, 34).is_none());
        fs::write(&p, [0u8; 4]).unwrap();
        assert!(c.read(&key, usize::MAX).is_none());
    }
    #[test]
    fn disk_budget_and_lock_are_recoverable() {
        let d = TestDir::new("cache-budget");
        let c = cache(d.path(), 150);
        let words = vec![0u16; 17];
        for i in 0..8 {
            c.publish(&[i; 32], &words);
        }
        let used: u64 = fs::read_dir(d.path())
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|s| s == "q8"))
            .map(|e| e.metadata().unwrap().len())
            .sum();
        assert!(used <= 150);
        drop(c);
        let c = cache(d.path(), 150);
        let lock = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(d.path().join("writer.lock"))
            .unwrap();
        lock.lock().unwrap();
        c.publish(&[99; 32], &words);
        assert!(c.read(&[99; 32], 17).is_none());
        drop(lock);
        c.publish(&[99; 32], &words);
        assert!(c.read(&[99; 32], 17).is_some());
        let tiny = cache(d.path(), 10);
        tiny.publish(&[88; 32], &words);
        assert!(tiny.read(&[88; 32], 17).is_none());
        let bad = d.path().join("not-directory");
        fs::write(&bad, b"x").unwrap();
        cache(&bad, 1000).publish(&[0; 32], &words);
    }
}
