//! Prepared voice-clone references (TPIPE-23). A saved voice, or the same
//! reference sent again, used to be encoded by the speech-tokenizer encoder
//! and the ECAPA speaker encoder on every request and every long-form
//! segment. The codes and x-vector a reference prepares to are kept here,
//! keyed by its audio (rate and samples); the transcript does not change
//! them.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use candle_core::Tensor;
use serde::Serialize;
use sha2::{Digest, Sha256};

/// Prepared references kept, least recently used first out. A prepared
/// reference is its codes (at most 16 groups of a few hundred frames) and
/// one x-vector row, a few tens of kilobytes.
const REFERENCE_CACHE_ENTRIES: usize = 16;

/// What one reference audio prepares to.
#[derive(Clone)]
pub(crate) struct PreparedReference {
    pub(crate) codes: Vec<Vec<u32>>,
    pub(crate) speaker_embedding: Option<Tensor>,
}

/// Hit and miss counts since load.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ReferenceCacheStats {
    pub hits: u64,
    pub misses: u64,
}

#[derive(Default)]
pub(crate) struct ReferenceCache {
    entries: Mutex<VecDeque<([u8; 32], PreparedReference)>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl ReferenceCache {
    /// The key of a reference audio: its rate and samples.
    pub(crate) fn key(samples: &[f32], sample_rate: u32) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(sample_rate.to_le_bytes());
        for sample in samples {
            hasher.update(sample.to_le_bytes());
        }
        hasher.finalize().into()
    }

    /// The prepared reference for `key`, or `prepare` it and keep it.
    pub(crate) fn get_or_prepare(
        &self,
        key: [u8; 32],
        prepare: impl FnOnce() -> crate::error::Result<PreparedReference>,
    ) -> crate::error::Result<PreparedReference> {
        {
            let mut entries = self
                .entries
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if let Some(position) = entries.iter().position(|(entry, _)| *entry == key) {
                let entry = entries.remove(position).expect("found entry");
                let prepared = entry.1.clone();
                entries.push_back(entry);
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Ok(prepared);
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        let prepared = prepare()?;
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if !entries.iter().any(|(entry, _)| *entry == key) {
            if entries.len() >= REFERENCE_CACHE_ENTRIES {
                entries.pop_front();
            }
            entries.push_back((key, prepared.clone()));
        }
        Ok(prepared)
    }

    pub(crate) fn stats(&self) -> ReferenceCacheStats {
        ReferenceCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prepared(code: u32) -> PreparedReference {
        PreparedReference {
            codes: vec![vec![code]],
            speaker_embedding: None,
        }
    }

    #[test]
    fn a_reference_prepares_once_and_the_oldest_leaves_first() {
        let cache = ReferenceCache::default();
        let key = ReferenceCache::key(&[0.1, 0.2], 24_000);
        assert_ne!(key, ReferenceCache::key(&[0.1, 0.2], 16_000));
        let mut prepares = 0;
        for _ in 0..3 {
            let reference = cache
                .get_or_prepare(key, || {
                    prepares += 1;
                    Ok(prepared(7))
                })
                .unwrap();
            assert_eq!(reference.codes, vec![vec![7]]);
        }
        assert_eq!(prepares, 1);
        assert_eq!(cache.stats(), ReferenceCacheStats { hits: 2, misses: 1 });

        for index in 0..REFERENCE_CACHE_ENTRIES as u32 {
            let key = ReferenceCache::key(&[index as f32], 24_000);
            cache.get_or_prepare(key, || Ok(prepared(index))).unwrap();
        }
        // The first reference was the least recently used, so it left.
        cache
            .get_or_prepare(key, || {
                prepares += 1;
                Ok(prepared(7))
            })
            .unwrap();
        assert_eq!(prepares, 2);
    }
}
