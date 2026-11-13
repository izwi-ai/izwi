//! What happens to a recording between the microphone and the speech model:
//! silence is cut from both ends, an empty recording is dropped, and the
//! rest is packed as a WAV file.

use izwi_vad::{detect_speech_regions_f32, VadRegionConfig};

/// Recordings are 16 kHz mono: what the voice detector and the speech
/// models take.
pub const SAMPLE_RATE: u32 = izwi_vad::VAD_SAMPLE_RATE;

/// Kept on each side of the detected speech, on top of the detector's own
/// padding, so soft first and last consonants survive.
const EDGE_MARGIN_MS: usize = 120;

/// A recording the detector calls silent is still sent when it is at least
/// this loud: quiet or whispered speech can score below the threshold.
const LOUD_ENOUGH_DBFS: f32 = -38.0;

/// The part of `samples` to transcribe, or `None` when nobody spoke.
pub fn speech_clip(samples: &[f32]) -> Option<Vec<f32>> {
    if samples.is_empty() {
        return None;
    }
    let regions = detect_speech_regions_f32(samples, SAMPLE_RATE, &VadRegionConfig::default())
        .unwrap_or_default();
    let (Some(first), Some(last)) = (regions.first(), regions.last()) else {
        return (level_dbfs(samples) >= LOUD_ENOUGH_DBFS).then(|| samples.to_vec());
    };
    let margin = EDGE_MARGIN_MS * SAMPLE_RATE as usize / 1000;
    let start = first.start_sample.saturating_sub(margin);
    let end = last.end_sample.saturating_add(margin).min(samples.len());
    Some(samples[start..end].to_vec())
}

/// Root-mean-square level in dB relative to full scale.
pub fn level_dbfs(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return izwi_vad::MIN_ENERGY_DBFS;
    }
    let mean_square = samples
        .iter()
        .map(|sample| f64::from(*sample) * f64::from(*sample))
        .sum::<f64>()
        / samples.len() as f64;
    (10.0 * mean_square.max(1e-12).log10()) as f32
}

/// A 0..1 meter value for the indicator: -60 dBFS and below is 0, -10 dBFS
/// and above is 1.
pub fn meter(samples: &[f32]) -> f32 {
    ((level_dbfs(samples) + 60.0) / 50.0).clamp(0.0, 1.0)
}

/// True when every sample is exactly zero: what a muted input or a denied
/// microphone permission delivers.
pub fn is_digital_silence(samples: &[f32]) -> bool {
    samples.iter().all(|sample| *sample == 0.0)
}

/// `samples` as a 16-bit mono WAV file.
pub fn wav_bytes(samples: &[f32]) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    out.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes()); // bytes per second
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        out.extend_from_slice(&pcm16(*sample).to_le_bytes());
    }
    out
}

/// The samples of a 16-bit mono WAV file at [`SAMPLE_RATE`], as written by
/// [`wav_bytes`]. For tests that stand in a file for the microphone.
pub fn samples_from_wav(wav: &[u8]) -> Option<Vec<f32>> {
    if wav.len() < 12 || &wav[..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return None;
    }
    let mut offset = 12;
    let mut format_ok = false;
    while offset + 8 <= wav.len() {
        let id = &wav[offset..offset + 4];
        let size = u32::from_le_bytes(wav[offset + 4..offset + 8].try_into().ok()?) as usize;
        let body = wav.get(offset + 8..(offset + 8 + size).min(wav.len()))?;
        match id {
            b"fmt " if body.len() >= 16 => {
                let pcm = u16::from_le_bytes([body[0], body[1]]) == 1;
                let mono = u16::from_le_bytes([body[2], body[3]]) == 1;
                let rate = u32::from_le_bytes(body[4..8].try_into().ok()?);
                let bits = u16::from_le_bytes([body[14], body[15]]);
                format_ok = pcm && mono && rate == SAMPLE_RATE && bits == 16;
            }
            b"data" => {
                return format_ok.then(|| {
                    body.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|pair| f32::from(i16::from_le_bytes(*pair)) / f32::from(i16::MAX))
                        .collect()
                });
            }
            _ => {}
        }
        // Chunks are padded to an even length.
        offset += 8 + size + (size & 1);
    }
    None
}

pub fn pcm16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(ms: usize, amplitude: f32) -> Vec<f32> {
        (0..ms * SAMPLE_RATE as usize / 1000)
            .map(|index| {
                let t = index as f32 / SAMPLE_RATE as f32;
                amplitude * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn silence_is_not_a_clip() {
        assert_eq!(speech_clip(&[]), None);
        assert_eq!(speech_clip(&vec![0.0; SAMPLE_RATE as usize * 2]), None);
        // Room noise far below speech level.
        assert_eq!(speech_clip(&tone(1_500, 0.001)), None);
    }

    #[test]
    fn loud_audio_is_kept_even_when_the_detector_is_unsure() {
        let clip = speech_clip(&tone(1_000, 0.5));
        assert!(clip.is_some_and(|clip| !clip.is_empty()));
    }

    #[test]
    fn levels_and_meter() {
        assert!(level_dbfs(&vec![0.0; 100]) <= -100.0);
        let full = level_dbfs(&[1.0, -1.0, 1.0, -1.0]);
        assert!(full.abs() < 0.01, "{full}");
        assert_eq!(meter(&vec![0.0; 100]), 0.0);
        assert_eq!(meter(&[1.0, -1.0]), 1.0);
        assert!(is_digital_silence(&[0.0, 0.0]));
        assert!(!is_digital_silence(&[0.0, 1e-6]));
    }

    #[test]
    fn wav_has_a_valid_header_and_clamped_samples() {
        let wav = wav_bytes(&[0.0, 1.0, -2.0]);
        assert_eq!(wav.len(), 44 + 6);
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()), 36 + 6);
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 16_000);
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 6);
        assert_eq!(i16::from_le_bytes([wav[46], wav[47]]), i16::MAX);
        assert_eq!(i16::from_le_bytes([wav[48], wav[49]]), -i16::MAX);
    }

    #[test]
    fn wav_files_read_back() {
        let samples = tone(50, 0.25);
        let read = samples_from_wav(&wav_bytes(&samples)).unwrap();
        assert_eq!(read.len(), samples.len());
        assert!(read
            .iter()
            .zip(&samples)
            .all(|(read, wrote)| (read - wrote).abs() < 1e-3));
        assert_eq!(samples_from_wav(b"not a wav file"), None);
        // Another rate is refused rather than played at the wrong speed.
        let mut other_rate = wav_bytes(&samples);
        other_rate[24..28].copy_from_slice(&44_100u32.to_le_bytes());
        assert_eq!(samples_from_wav(&other_rate), None);
    }
}
