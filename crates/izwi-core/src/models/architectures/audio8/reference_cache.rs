//! Model-instance-local reference codes. No target text or transcript is cached.
//!
//! The full host budget is reserved by model residency, independently of request
//! preparation. Returned codes are request-owned copies; eviction never leaves
//! unaccounted shared borrows alive. Per-key builders coalesce concurrent misses without blocking unrelated hits.

use std::mem::size_of;
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use sha2::{Digest, Sha256};

use super::Audio8VqCodes;
use crate::error::{Error, Result};

pub(crate) const AUDIO8_REFERENCE_CACHE_BYTES: u64 = 4 * 1024 * 1024;
const MAX_ENTRIES: usize = 32;

struct Entry {
    key: [u8; 32],
    used: u64,
    rows: usize,
    frames: usize,
    codes: Box<[u32]>,
}

struct State {
    entries: [Option<Entry>; MAX_ENTRIES],
    clock: u64,
    payload_bytes: usize,
    building: [Option<[u8; 32]>; MAX_ENTRIES],
}

pub(super) struct ReferenceCodeCache {
    state: Mutex<State>,
    budget: usize,
    changed: Condvar,
}

impl Default for ReferenceCodeCache {
    fn default() -> Self {
        Self::with_budget(AUDIO8_REFERENCE_CACHE_BYTES as usize)
    }
}

impl ReferenceCodeCache {
    fn with_budget(budget: usize) -> Self {
        Self {
            state: Mutex::new(State {
                entries: std::array::from_fn(|_| None),
                clock: 0,
                payload_bytes: 0,
                building: [None; MAX_ENTRIES],
            }),
            budget,
            changed: Condvar::new(),
        }
    }

    pub(super) fn get_or_encode(
        &self,
        samples: &[f32],
        sample_rate: u32,
        check: &dyn Fn() -> Result<()>,
        encode: impl FnOnce() -> Result<Audio8VqCodes>,
    ) -> Result<(Audio8VqCodes, bool)> {
        check()?;
        let key = reference_key(samples, sample_rate, check)?;
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::InferenceError("Audio8 reference cache poisoned".into()))?;
        let builder_slot = loop {
            check()?;
            state.clock = state.clock.saturating_add(1);
            let clock = state.clock;
            if let Some(entry) = state
                .entries
                .iter_mut()
                .flatten()
                .find(|entry| entry.key == key)
            {
                entry.used = clock;
                let codebooks = if entry.frames == 0 {
                    vec![Vec::new(); entry.rows]
                } else {
                    entry
                        .codes
                        .chunks_exact(entry.frames)
                        .map(<[u32]>::to_vec)
                        .collect()
                };
                return Ok((Audio8VqCodes { codebooks }, true));
            }
            if !state.building.contains(&Some(key)) {
                if let Some(slot) = state.building.iter().position(Option::is_none) {
                    state.building[slot] = Some(key);
                    break slot;
                }
            }
            // Waits are bounded so cancellation remains observable. The fixed
            // metadata table also bounds distinct concurrent cache builders;
            // their tensors remain covered by request preparation admission.
            state = self
                .changed
                .wait_timeout(state, Duration::from_millis(5))
                .map_err(|_| Error::InferenceError("Audio8 reference cache poisoned".into()))?
                .0;
        };
        drop(state);
        let _builder = BuilderGuard {
            cache: self,
            slot: builder_slot,
        };
        let codes = encode()?;
        // Failed/cancelled builders never publish a partial cache entry.
        check()?;
        let rows = codes.codebooks.len();
        let frames = codes.codebooks.first().map_or(0, Vec::len);
        if codes.codebooks.iter().any(|row| row.len() != frames) {
            return Err(Error::InferenceError(
                "Audio8 reference encoder returned ragged codes".into(),
            ));
        }
        let bytes = rows
            .checked_mul(frames)
            .and_then(|count| count.checked_mul(size_of::<u32>()))
            .ok_or_else(|| Error::InferenceError("Audio8 reference cache size overflow".into()))?;
        let capacity = self.budget.saturating_sub(size_of::<Self>());
        if bytes == 0 || bytes > capacity {
            return Ok((codes, false));
        }
        let mut state = self
            .state
            .lock()
            .map_err(|_| Error::InferenceError("Audio8 reference cache poisoned".into()))?;
        state.clock = state.clock.saturating_add(1);
        let clock = state.clock;
        // Evict before allocating the cache-owned copy. The encoder result and
        // any caller-owned hit copy stay within that request's preparation lease.
        while state.payload_bytes > capacity - bytes || state.entries.iter().all(Option::is_some) {
            let oldest = state
                .entries
                .iter()
                .enumerate()
                .filter_map(|(index, entry)| entry.as_ref().map(|entry| (index, entry.used)))
                .min_by_key(|(_, used)| *used)
                .map(|(index, _)| index)
                .expect("nonempty cache when eviction is required");
            let old = state.entries[oldest].take().expect("selected cache entry");
            state.payload_bytes -= old.codes.len() * size_of::<u32>();
            drop(old);
        }
        let mut flat = vec![0; rows * frames].into_boxed_slice();
        for (target, source) in flat.chunks_exact_mut(frames).zip(&codes.codebooks) {
            target.copy_from_slice(source);
        }
        check()?;
        let slot = state
            .entries
            .iter()
            .position(Option::is_none)
            .expect("evicted cache slot");
        state.entries[slot] = Some(Entry {
            key,
            used: clock,
            rows,
            frames,
            codes: flat,
        });
        state.payload_bytes += bytes;
        Ok((codes, false))
    }
}

// Clear the in-flight key on every return, cancellation, encoder failure and
// unwind. Never publish partial values or hold metadata while running a codec.
struct BuilderGuard<'a> {
    cache: &'a ReferenceCodeCache,
    slot: usize,
}

impl Drop for BuilderGuard<'_> {
    fn drop(&mut self) {
        let mut state = self.cache.state.lock().unwrap_or_else(|e| e.into_inner());
        state.building[self.slot] = None;
        self.cache.changed.notify_all();
    }
}

fn reference_key(
    samples: &[f32],
    sample_rate: u32,
    check: &dyn Fn() -> Result<()>,
) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    // This version binds normalization/codec input semantics. Codec weights and
    // revision are implicit in cache lifetime: every model load owns a fresh cache.
    hash.update(b"audio8-reference-input-v1");
    hash.update(sample_rate.to_le_bytes());
    hash.update((samples.len() as u64).to_le_bytes());
    for chunk in samples.chunks(4096) {
        check()?;
        for sample in chunk {
            hash.update(sample.to_bits().to_le_bytes());
        }
    }
    Ok(hash.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn codes(value: u32) -> Audio8VqCodes {
        Audio8VqCodes {
            codebooks: vec![vec![value; 4]; 2],
        }
    }

    #[test]
    fn hit_preserves_codes_and_keys_audio_rate_and_model_lifetime() {
        let cache = ReferenceCodeCache::default();
        let encoded = cache
            .get_or_encode(&[0.1], 44_100, &|| Ok(()), || Ok(codes(7)))
            .unwrap();
        assert!(!encoded.1);
        let hit = cache
            .get_or_encode(&[0.1], 44_100, &|| Ok(()), || panic!("cache miss"))
            .unwrap();
        assert_eq!(hit.0, encoded.0);
        assert!(hit.1);
        for (audio, rate) in [(0.2, 44_100), (0.1, 16_000)] {
            assert!(
                !cache
                    .get_or_encode(&[audio], rate, &|| Ok(()), || Ok(codes(9)))
                    .unwrap()
                    .1
            );
        }
        assert!(
            !ReferenceCodeCache::default()
                .get_or_encode(&[0.1], 44_100, &|| Ok(()), || Ok(codes(8)))
                .unwrap()
                .1
        );
    }

    #[test]
    fn eviction_budget_and_caller_copy_are_independent() {
        let cache = ReferenceCodeCache::with_budget(size_of::<ReferenceCodeCache>() + 32);
        let owned = cache
            .get_or_encode(&[1.], 1, &|| Ok(()), || Ok(codes(1)))
            .unwrap()
            .0;
        cache
            .get_or_encode(&[2.], 1, &|| Ok(()), || Ok(codes(2)))
            .unwrap();
        assert_eq!(owned, codes(1));
        assert_eq!(cache.state.lock().unwrap().payload_bytes, 32);
        assert!(
            !cache
                .get_or_encode(&[1.], 1, &|| Ok(()), || Ok(codes(1)))
                .unwrap()
                .1
        );
        let tiny = ReferenceCodeCache::with_budget(size_of::<ReferenceCodeCache>());
        tiny.get_or_encode(&[1.], 1, &|| Ok(()), || Ok(codes(1)))
            .unwrap();
        assert_eq!(tiny.state.lock().unwrap().payload_bytes, 0);
    }

    #[test]
    fn failed_builder_is_not_published_and_can_retry() {
        let cache = ReferenceCodeCache::default();
        assert!(cache
            .get_or_encode(&[1.], 1, &|| Ok(()), || Err(Error::Cancelled(
                "cancelled".into()
            )))
            .is_err());
        assert_eq!(cache.state.lock().unwrap().payload_bytes, 0);
        assert!(
            !cache
                .get_or_encode(&[1.], 1, &|| Ok(()), || Ok(codes(1)))
                .unwrap()
                .1
        );
    }

    #[test]
    fn cancelled_completed_encoder_does_not_publish() {
        use std::sync::atomic::AtomicBool;
        let cache = ReferenceCodeCache::default();
        let cancelled = AtomicBool::new(false);
        let check = || {
            if cancelled.load(Ordering::SeqCst) {
                Err(Error::Cancelled("cancelled".into()))
            } else {
                Ok(())
            }
        };
        assert!(cache
            .get_or_encode(&[1.], 1, &check, || {
                cancelled.store(true, Ordering::SeqCst);
                Ok(codes(1))
            })
            .is_err());
        assert_eq!(cache.state.lock().unwrap().payload_bytes, 0);
        cancelled.store(false, Ordering::SeqCst);
        assert!(
            !cache
                .get_or_encode(&[1.], 1, &check, || Ok(codes(1)))
                .unwrap()
                .1
        );
    }

    #[test]
    fn concurrent_identical_misses_encode_once() {
        let cache = ReferenceCodeCache::default();
        let count = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    cache
                        .get_or_encode(&[1.], 1, &|| Ok(()), || {
                            count.fetch_add(1, Ordering::SeqCst);
                            std::thread::sleep(Duration::from_millis(10));
                            Ok(codes(1))
                        })
                        .unwrap();
                });
            }
        });
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn warm_hit_and_distinct_miss_progress_while_builder_is_blocked() {
        let cache = ReferenceCodeCache::default();
        cache
            .get_or_encode(&[1.], 1, &|| Ok(()), || Ok(codes(1)))
            .unwrap();
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let cache_ref = &cache;
            scope.spawn(move || {
                cache_ref
                    .get_or_encode(&[2.], 1, &|| Ok(()), || {
                        entered_tx.send(()).unwrap();
                        release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                        Ok(codes(2))
                    })
                    .unwrap()
            });
            entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            assert!(
                cache
                    .get_or_encode(&[1.], 1, &|| Ok(()), || panic!("warm hit blocked"))
                    .unwrap()
                    .1
            );
            assert!(
                !cache
                    .get_or_encode(&[3.], 1, &|| Ok(()), || Ok(codes(3)))
                    .unwrap()
                    .1
            );
            release_tx.send(()).unwrap();
        });
    }
}
