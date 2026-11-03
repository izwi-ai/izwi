//! Reference clips to codes, cached per model load (§3.8, §7.1 stage 0).
//!
//! A clip is checked (at most 60 s, whole frames of its channels), then on
//! a miss mixed to mono by the mean of its channels, resampled to the
//! codec's 44.1 kHz, clamped to [-1, 1] and handed to the codec, which pads
//! it to whole 2,048-sample frames and encodes it. The cache is keyed on
//! the clip as given (samples, channels and rate under the
//! `fish-drama3-reference-input-v1` domain), so a hit skips the resampler
//! and the encoder. Each load owns a fresh cache, which binds the codec
//! weights implicitly. No transcript or target text is cached.
//!
//! The rest is Fish S2's cache (`fish_s2/reference_cache.rs`), as an own
//! copy: a fixed host budget reserved with the model, request-owned copies
//! of the codes, and per-key builders that coalesce concurrent misses
//! without blocking hits.

use std::mem::size_of;
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use sha2::{Digest, Sha256};

use super::prompt::FishDrama3Codes;
use crate::audio::{resample_mono_high_quality, target_sample_count};
use crate::error::{Error, Result};

pub const FISH_DRAMA3_REFERENCE_CACHE_BYTES: u64 = 4 * 1024 * 1024;
/// The longest inline reference clip (saved voices hold 30 s).
pub const MAX_REFERENCE_SECONDS: u32 = 60;
/// References one prompt may hold.
pub const MAX_REFERENCES: usize = 8;
const KEY_DOMAIN: &[u8] = b"fish-drama3-reference-input-v1";
const MAX_ENTRIES: usize = 32;

/// A reference clip as the caller has it.
#[derive(Debug, Clone, Copy)]
pub struct FishDrama3ReferenceAudio<'a> {
    /// Interleaved samples, `channels` per frame.
    pub samples: &'a [f32],
    pub channels: usize,
    pub sample_rate: u32,
}

impl FishDrama3ReferenceAudio<'_> {
    /// Refuses an empty or ragged clip, a zero rate and anything over
    /// [`MAX_REFERENCE_SECONDS`].
    pub fn validate(&self) -> Result<()> {
        if self.channels == 0
            || self.sample_rate == 0
            || self.samples.is_empty()
            || !self.samples.len().is_multiple_of(self.channels)
        {
            return Err(Error::InvalidInput(format!(
                "Fish Audio Drama 3 reference audio needs whole frames of {} channels at a \
                 non-zero rate; got {} samples at {} Hz",
                self.channels,
                self.samples.len(),
                self.sample_rate
            )));
        }
        let frames = (self.samples.len() / self.channels) as u64;
        if frames > u64::from(MAX_REFERENCE_SECONDS) * u64::from(self.sample_rate) {
            return Err(Error::InvalidInput(format!(
                "Fish Audio Drama 3 reference audio is {:.1} s; at most {MAX_REFERENCE_SECONDS} s",
                frames as f64 / f64::from(self.sample_rate)
            )));
        }
        Ok(())
    }

    /// Mono by the channels' mean, at `rate`, clamped to [-1, 1]
    /// (non-finite samples become 0).
    pub fn prepared(&self, rate: u32) -> Result<Vec<f32>> {
        self.validate()?;
        let mono = if self.channels == 1 {
            self.samples.to_vec()
        } else {
            self.samples
                .chunks_exact(self.channels)
                .map(|frame| frame.iter().sum::<f32>() / self.channels as f32)
                .collect()
        };
        let mut samples = resample_mono_high_quality(&mono, self.sample_rate, rate)?;
        for sample in &mut samples {
            *sample = if sample.is_finite() {
                sample.clamp(-1.0, 1.0)
            } else {
                0.0
            };
        }
        Ok(samples)
    }

    /// The length of [`Self::prepared`] at `rate`, without resampling.
    pub fn prepared_len(&self, rate: u32) -> Result<usize> {
        self.validate()?;
        Ok(target_sample_count(
            self.samples.len() / self.channels,
            self.sample_rate,
            rate,
        ))
    }
}

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

pub(crate) struct FishDrama3ReferenceCache {
    state: Mutex<State>,
    budget: usize,
    changed: Condvar,
}

impl Default for FishDrama3ReferenceCache {
    fn default() -> Self {
        Self::with_budget(FISH_DRAMA3_REFERENCE_CACHE_BYTES as usize)
    }
}

fn poisoned() -> Error {
    Error::InferenceError("Fish Audio Drama 3 reference cache poisoned".into())
}

impl FishDrama3ReferenceCache {
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

    /// The codes of `audio`, from the cache or from `encode` over its
    /// samples prepared at `rate`; and whether the cache held them.
    pub(crate) fn get_or_encode(
        &self,
        audio: &FishDrama3ReferenceAudio<'_>,
        rate: u32,
        check: &dyn Fn() -> Result<()>,
        encode: impl FnOnce(&[f32]) -> Result<FishDrama3Codes>,
    ) -> Result<(FishDrama3Codes, bool)> {
        check()?;
        audio.validate()?;
        let key = reference_key(audio, check)?;
        let mut state = self.state.lock().map_err(|_| poisoned())?;
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
                let codebooks = entry
                    .codes
                    .chunks_exact(entry.frames)
                    .map(<[u32]>::to_vec)
                    .collect();
                return Ok((FishDrama3Codes::new(codebooks), true));
            }
            if !state.building.contains(&Some(key)) {
                if let Some(slot) = state.building.iter().position(Option::is_none) {
                    state.building[slot] = Some(key);
                    break slot;
                }
            }
            // Bounded waits keep cancellation observable; the fixed table
            // bounds concurrent builders.
            state = self
                .changed
                .wait_timeout(state, Duration::from_millis(5))
                .map_err(|_| poisoned())?
                .0;
        };
        drop(state);
        let _builder = BuilderGuard {
            cache: self,
            slot: builder_slot,
        };
        let prepared = audio.prepared(rate)?;
        check()?;
        let codes = encode(&prepared)?;
        // A failed or cancelled build never publishes a partial entry.
        check()?;
        let rows = codes.codebooks.len();
        let frames = codes.frames();
        if codes.codebooks.iter().any(|row| row.len() != frames) {
            return Err(Error::InferenceError(
                "Fish Audio Drama 3 reference encoder returned ragged codes".into(),
            ));
        }
        let bytes = rows
            .checked_mul(frames)
            .and_then(|count| count.checked_mul(size_of::<u32>()))
            .ok_or_else(|| {
                Error::InferenceError("Fish Audio Drama 3 reference cache size overflow".into())
            })?;
        let capacity = self.budget.saturating_sub(size_of::<Self>());
        if bytes == 0 || bytes > capacity {
            return Ok((codes, false));
        }
        let mut state = self.state.lock().map_err(|_| poisoned())?;
        state.clock = state.clock.saturating_add(1);
        let clock = state.clock;
        // Evict before allocating the cache's own copy.
        while state.payload_bytes > capacity - bytes || state.entries.iter().all(Option::is_some) {
            let oldest = state
                .entries
                .iter()
                .enumerate()
                .filter_map(|(index, entry)| entry.as_ref().map(|entry| (index, entry.used)))
                .min_by_key(|(_, used)| *used)
                .map(|(index, _)| index)
                .expect("a non-empty cache when eviction is required");
            let old = state.entries[oldest].take().expect("the selected entry");
            state.payload_bytes -= old.codes.len() * size_of::<u32>();
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
            .expect("an evicted slot");
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

    #[cfg(test)]
    fn payload_bytes(&self) -> usize {
        self.state.lock().unwrap().payload_bytes
    }

    #[cfg(test)]
    fn entry_rows(&self) -> Vec<usize> {
        let state = self.state.lock().unwrap();
        state
            .entries
            .iter()
            .flatten()
            .map(|entry| entry.rows)
            .collect()
    }
}

/// Clears the in-flight key on every return, cancellation, encoder failure
/// and unwind.
struct BuilderGuard<'a> {
    cache: &'a FishDrama3ReferenceCache,
    slot: usize,
}

impl Drop for BuilderGuard<'_> {
    fn drop(&mut self) {
        let mut state = self.cache.state.lock().unwrap_or_else(|e| e.into_inner());
        state.building[self.slot] = None;
        self.cache.changed.notify_all();
    }
}

/// The clip as given, under the key domain: the domain versions what the
/// preparation and the codec input mean.
fn reference_key(
    audio: &FishDrama3ReferenceAudio<'_>,
    check: &dyn Fn() -> Result<()>,
) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    hash.update(KEY_DOMAIN);
    hash.update(audio.sample_rate.to_le_bytes());
    hash.update((audio.channels as u64).to_le_bytes());
    hash.update((audio.samples.len() as u64).to_le_bytes());
    // One update per 4,096 samples: the same bytes in the same order as a
    // 4-byte update per sample, so the same digest, without SHA-256's
    // per-call buffering for every sample of a long clip.
    let mut bytes = Vec::with_capacity(4096 * 4);
    for chunk in audio.samples.chunks(4096) {
        check()?;
        bytes.clear();
        for sample in chunk {
            bytes.extend_from_slice(&sample.to_bits().to_le_bytes());
        }
        hash.update(&bytes);
    }
    Ok(hash.finalize().into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    const RATE: u32 = 44_100;

    fn codes(value: u32) -> FishDrama3Codes {
        FishDrama3Codes::new(vec![vec![value; 4]; 10])
    }

    fn mono(samples: &[f32]) -> FishDrama3ReferenceAudio<'_> {
        FishDrama3ReferenceAudio {
            samples,
            channels: 1,
            sample_rate: RATE,
        }
    }

    fn ok() -> Result<()> {
        Ok(())
    }

    /// Keys stay what they were: the bulk updates hash the bytes the
    /// per-sample updates did, so a key never changes with this code.
    #[test]
    fn the_key_is_the_per_sample_digest() {
        let samples = (0..10_000)
            .map(|i| ((i as f32) * 0.37).sin())
            .collect::<Vec<_>>();
        let audio = mono(&samples);
        let mut hash = Sha256::new();
        hash.update(KEY_DOMAIN);
        hash.update(audio.sample_rate.to_le_bytes());
        hash.update((audio.channels as u64).to_le_bytes());
        hash.update((audio.samples.len() as u64).to_le_bytes());
        for sample in audio.samples {
            hash.update(sample.to_bits().to_le_bytes());
        }
        let expected: [u8; 32] = hash.finalize().into();
        assert_eq!(reference_key(&audio, &ok).unwrap(), expected);
    }

    /// The prepared length is known without resampling: mono frames at
    /// the target rate, as `prepared` makes them.
    #[test]
    fn the_prepared_length_matches_the_prepared_samples() {
        let samples = (0..4_802)
            .map(|i| (i as f32 * 0.01).sin())
            .collect::<Vec<_>>();
        for (channels, sample_rate) in [(1, 8_000), (2, 16_000), (1, RATE), (2, 48_000)] {
            let audio = FishDrama3ReferenceAudio {
                samples: &samples,
                channels,
                sample_rate,
            };
            assert_eq!(
                audio.prepared_len(RATE).unwrap(),
                audio.prepared(RATE).unwrap().len(),
                "{channels} channels at {sample_rate} Hz"
            );
        }
    }

    #[test]
    fn a_hit_returns_the_codes_and_the_key_covers_samples_channels_and_rate() {
        let cache = FishDrama3ReferenceCache::default();
        let samples = [0.1f32, -0.2, 0.3, 0.4];
        let (encoded, hit) = cache
            .get_or_encode(&mono(&samples), RATE, &ok, |_| Ok(codes(7)))
            .unwrap();
        assert!(!hit);
        let (again, hit) = cache
            .get_or_encode(&mono(&samples), RATE, &ok, |_| panic!("a cache miss"))
            .unwrap();
        assert!(hit);
        assert_eq!(again, encoded);
        // Another sample, another rate, the same samples read as stereo:
        // each is its own key.
        let stereo = FishDrama3ReferenceAudio {
            channels: 2,
            ..mono(&samples)
        };
        let other_rate = FishDrama3ReferenceAudio {
            sample_rate: 16_000,
            ..mono(&samples)
        };
        let other_samples = [0.1f32, -0.2, 0.3, 0.5];
        for audio in [stereo, other_rate, mono(&other_samples)] {
            let (_, hit) = cache
                .get_or_encode(&audio, RATE, &ok, |_| Ok(codes(9)))
                .unwrap();
            assert!(!hit, "{audio:?}");
        }
        // Every load owns a fresh cache.
        assert!(
            !FishDrama3ReferenceCache::default()
                .get_or_encode(&mono(&samples), RATE, &ok, |_| Ok(codes(8)))
                .unwrap()
                .1
        );
        assert_eq!(cache.entry_rows(), vec![10; 4]);
    }

    #[test]
    fn the_encoder_gets_mono_means_resampled_and_clamped() {
        let cache = FishDrama3ReferenceCache::default();
        // Stereo at the codec rate: the mean of each frame, clamped.
        let samples = [0.5f32, 0.25, 3.0, 1.0, -4.0, -2.0, f32::NAN, 0.0];
        let audio = FishDrama3ReferenceAudio {
            samples: &samples,
            channels: 2,
            sample_rate: RATE,
        };
        cache
            .get_or_encode(&audio, RATE, &ok, |prepared| {
                assert_eq!(prepared, [0.375, 1.0, -1.0, 0.0]);
                Ok(codes(1))
            })
            .unwrap();
        // Another rate is resampled to the codec's.
        let samples = vec![0.25f32; 22_050];
        cache
            .get_or_encode(
                &FishDrama3ReferenceAudio {
                    samples: &samples,
                    channels: 1,
                    sample_rate: 22_050,
                },
                RATE,
                &ok,
                |prepared| {
                    assert_eq!(prepared.len(), 44_100);
                    assert!(prepared.iter().all(|s| (-1.0..=1.0).contains(s)));
                    Ok(codes(2))
                },
            )
            .unwrap();
    }

    #[test]
    fn a_clip_over_60_s_or_ragged_is_refused_before_encoding() {
        let cache = FishDrama3ReferenceCache::default();
        let rate = 100;
        let exact = vec![0.0f32; 60 * rate as usize];
        fn at(samples: &[f32], channels: usize) -> FishDrama3ReferenceAudio<'_> {
            FishDrama3ReferenceAudio {
                samples,
                channels,
                sample_rate: 100,
            }
        }
        // The codec rate is the clip's here, so nothing is resampled.
        let (_, hit) = cache
            .get_or_encode(&at(&exact, 1), rate, &ok, |_| Ok(codes(1)))
            .unwrap();
        assert!(!hit);
        let over = vec![0.0f32; 60 * rate as usize + 1];
        // Two channels make the same samples 30 s long.
        cache
            .get_or_encode(&at(&over[1..], 2), rate, &ok, |_| Ok(codes(1)))
            .unwrap();
        for audio in [at(&over, 1), at(&over, 2), at(&[], 1), at(&exact, 0)] {
            let err = cache
                .get_or_encode(&audio, rate, &ok, |_| panic!("encoded a refused clip"))
                .unwrap_err();
            assert!(matches!(err, Error::InvalidInput(_)), "{err}");
        }
        let zero_rate = FishDrama3ReferenceAudio {
            sample_rate: 0,
            ..at(&exact, 1)
        };
        assert!(zero_rate.validate().is_err());
    }

    #[test]
    fn eviction_and_failed_or_cancelled_builds_publish_nothing() {
        let entry = 10 * 4 * size_of::<u32>();
        let cache =
            FishDrama3ReferenceCache::with_budget(size_of::<FishDrama3ReferenceCache>() + entry);
        let owned = cache
            .get_or_encode(&mono(&[1.0]), RATE, &ok, |_| Ok(codes(1)))
            .unwrap()
            .0;
        cache
            .get_or_encode(&mono(&[0.5]), RATE, &ok, |_| Ok(codes(2)))
            .unwrap();
        assert_eq!(owned, codes(1));
        assert_eq!(cache.payload_bytes(), entry);
        // The first was evicted.
        assert!(
            !cache
                .get_or_encode(&mono(&[1.0]), RATE, &ok, |_| Ok(codes(1)))
                .unwrap()
                .1
        );

        let cache = FishDrama3ReferenceCache::default();
        assert!(cache
            .get_or_encode(&mono(&[1.0]), RATE, &ok, |_| Err(Error::Cancelled(
                "cancelled".into()
            )))
            .is_err());
        assert_eq!(cache.payload_bytes(), 0);
        let cancelled = AtomicBool::new(false);
        let check = || {
            if cancelled.load(Ordering::SeqCst) {
                Err(Error::Cancelled("cancelled".into()))
            } else {
                Ok(())
            }
        };
        assert!(cache
            .get_or_encode(&mono(&[1.0]), RATE, &check, |_| {
                cancelled.store(true, Ordering::SeqCst);
                Ok(codes(1))
            })
            .is_err());
        assert_eq!(cache.payload_bytes(), 0);
        cancelled.store(false, Ordering::SeqCst);
        assert!(
            !cache
                .get_or_encode(&mono(&[1.0]), RATE, &check, |_| Ok(codes(1)))
                .unwrap()
                .1
        );
    }

    #[test]
    fn concurrent_identical_misses_encode_once() {
        let cache = FishDrama3ReferenceCache::default();
        let count = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    cache
                        .get_or_encode(&mono(&[1.0]), RATE, &ok, |_| {
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
}
