//! The embedded Nemotron-3-Diarization Sortformer, run as NeMo's streaming
//! speaker encoder (`forward_streaming_step`) with the serving overrides:
//! chunk 500, FIFO 40, cache-update period 300, cache 188 frames, one
//! silence frame per speaker. Ported into this family, not imported.
//!
//! Per window:
//! 1. stack and project the raw-mel window (`pre_encode`);
//! 2. encode `[speaker cache | FIFO | window]` with positions from 0 over
//!    the concatenation and padded keys masked;
//! 3. `encoder_proj`, zero the padded frames, `subpixel_upsample` (Conv1d
//!    k3, pad 1; output channel `u * 192 + h` is sub-frame `u`), then
//!    `relu → first_hidden_to_hidden → relu → single_hidden_to_spks →
//!    sigmoid`, masked: 10 ms predictions;
//! 4. the window core's 10 ms predictions, averaged ×8 to 80 ms;
//! 5. unless it is the last window: the FIFO and speaker-cache update and
//!    compression (speaker-major log-odds scores, strong and weak boosts,
//!    the learnable silence embedding for disabled slots).
//!
//! The compression ranks BF16 scores upstream, so near-ties can order
//! differently here; tests compare probabilities before the first
//! compression and binarized activity after it.

use candle_core::{Device, Tensor};

use super::checkpoint::NemotronCheckpoint;
use super::config::{NemotronTranscribeConfig, SortformerConfig};
use super::linear::{NtLinear, Pass};
use super::speech_transformer::{host_values, SpeechTransformer};
use crate::error::{Error, Result};

const PREFIX: &str = "perception.encoder.diarization_model";

/// The Sortformer head (`sortformer_modules`).
pub struct SortformerHead {
    config: SortformerConfig,
    encoder_proj: NtLinear,
    /// `[tf * upsample, tf, 3]` as host F32 (applied as three shifted
    /// linears).
    subpixel_taps: [NtLinear; 3],
    subpixel_bias: Tensor,
    first_hidden_to_hidden: NtLinear,
    single_hidden_to_spks: NtLinear,
    silence: Vec<f32>,
    device: Device,
}

/// The streaming state carried from window to window: pre-`embed_norm`
/// embeddings and their 80 ms predictions.
#[derive(Debug, Clone, Default)]
pub struct SortformerState {
    cache: Vec<Vec<f32>>,
    /// `None` until the first compression: before it the cache's
    /// predictions are re-read from each step.
    cache_preds: Option<Vec<Vec<f32>>>,
    fifo: Vec<Vec<f32>>,
    fifo_preds: Vec<Vec<f32>>,
    pub(crate) compressions: u64,
}

impl SortformerState {
    pub fn cache_len(&self) -> usize {
        self.cache.len()
    }

    pub fn fifo_len(&self) -> usize {
        self.fifo.len()
    }
}

/// One window's diarizer output.
#[derive(Debug, Clone)]
pub struct SortformerWindowOutput {
    /// The core's 80 ms probabilities `[core][speakers]`.
    pub preds_80ms: Vec<Vec<f32>>,
    /// The core's native 10 ms probabilities `[core * 8][speakers]`.
    pub preds_10ms: Vec<Vec<f32>>,
}

impl SortformerHead {
    pub fn load(
        checkpoint: &NemotronCheckpoint,
        config: &NemotronTranscribeConfig,
        device: &Device,
    ) -> Result<Self> {
        let sf = config.sortformer;
        let m = format!("{PREFIX}.sortformer_modules");
        let dense = |name: &str, rows, cols| {
            NtLinear::load_dense(checkpoint, &format!("{m}.{name}"), rows, cols, true, device)
        };
        let up = sf.upsample;
        let tf = sf.tf_d_model;
        let taps =
            checkpoint.f32_values(&format!("{m}.subpixel_upsample.weight"), &[tf * up, tf, 3])?;
        let tap = |k: usize| {
            let values = (0..tf * up)
                .flat_map(|o| (0..tf).map(move |i| (o, i)))
                .map(|(o, i)| taps[(o * tf + i) * 3 + k])
                .collect::<Vec<_>>();
            NtLinear::dense_from_values(values, tf * up, tf, device, candle_core::DType::F32)
        };
        Ok(Self {
            encoder_proj: dense("encoder_proj", tf, sf.fc_d_model)?,
            subpixel_taps: [tap(0)?, tap(1)?, tap(2)?],
            subpixel_bias: Tensor::from_vec(
                checkpoint.f32_values(&format!("{m}.subpixel_upsample.bias"), &[tf * up])?,
                tf * up,
                device,
            )?,
            first_hidden_to_hidden: dense("first_hidden_to_hidden", tf, tf)?,
            single_hidden_to_spks: dense("single_hidden_to_spks", sf.speakers, tf)?,
            silence: checkpoint.f32_values(&format!("{m}.learnable_sil_emb"), &[sf.fc_d_model])?,
            config: sf,
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &SortformerConfig {
        &self.config
    }

    /// 10 ms probabilities `[T * upsample, speakers]` of encoded frames
    /// `[T, fc]`, the first `valid` of them real.
    fn forward_probabilities(&self, encoded: &Tensor, valid: usize) -> Result<Tensor> {
        let (frames, _) = encoded.dims2()?;
        let up = self.config.upsample;
        let tf = self.config.tf_d_model;
        let projected = self.encoder_proj.forward(encoded, Pass::Span)?;
        let projected = if valid < frames {
            let mask = Tensor::from_vec(
                (0..frames)
                    .map(|t| f32::from(t < valid))
                    .collect::<Vec<_>>(),
                (frames, 1),
                &self.device,
            )?;
            projected.broadcast_mul(&mask)?
        } else {
            projected
        };
        // Conv1d(k3, pad 1) over time: tap k reads frame t + k - 1.
        let padded = projected.pad_with_zeros(0, 1, 1)?;
        let mut conv = self
            .subpixel_bias
            .reshape((1, tf * up))?
            .broadcast_as((frames, tf * up))?;
        for (k, tap) in self.subpixel_taps.iter().enumerate() {
            conv =
                (conv + tap.forward(&padded.narrow(0, k, frames)?.contiguous()?, Pass::Span)?)?;
        }
        let hidden = conv.reshape((frames * up, tf))?.relu()?;
        let hidden = self
            .first_hidden_to_hidden
            .forward(&hidden, Pass::Span)?
            .relu()?;
        let probs =
            candle_nn::ops::sigmoid(&self.single_hidden_to_spks.forward(&hidden, Pass::Span)?)?;
        if valid < frames {
            let mask = Tensor::from_vec(
                (0..frames * up)
                    .map(|t| f32::from(t < valid * up))
                    .collect::<Vec<_>>(),
                (frames * up, 1),
                &self.device,
            )?;
            Ok(probs.broadcast_mul(&mask)?)
        } else {
            Ok(probs)
        }
    }

    /// One streaming step over a raw-mel window (time-major host F32,
    /// `valid_mel` frames real), `left_offset` / `right_offset` mel frames
    /// of context. Updates `state` unless `update_state` is false (the last
    /// window).
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        encoder: &SpeechTransformer,
        state: &mut SortformerState,
        mel: &[f32],
        valid_mel: usize,
        left_offset: usize,
        right_offset: usize,
        update_state: bool,
    ) -> Result<SortformerWindowOutput> {
        let sf = &self.config;
        let factor = encoder.config().stacking;
        let chunk = encoder.pre_encode(mel)?;
        let chunk_frames = chunk.dim(0)?;
        let chunk_valid = valid_mel.div_ceil(factor);
        let lc = ((left_offset as f64) / factor as f64).round() as usize;
        let rc = right_offset.div_ceil(factor);
        let core = chunk_frames.checked_sub(lc + rc).ok_or_else(|| {
            Error::InferenceError("Sortformer window is shorter than its context".into())
        })?;
        let prefix = state.cache.len() + state.fifo.len();
        let width = sf.fc_d_model;
        let combined = if prefix == 0 {
            chunk.clone()
        } else {
            let rows = state
                .cache
                .iter()
                .chain(&state.fifo)
                .flat_map(|row| row.iter().copied())
                .collect::<Vec<_>>();
            Tensor::cat(
                &[
                    &Tensor::from_vec(rows, (prefix, width), &self.device)?,
                    &chunk,
                ],
                0,
            )?
        };
        let valid = prefix + chunk_valid;
        let encoded = encoder.forward_pre_encoded(&combined, valid)?;
        let probs = self.forward_probabilities(&encoded, valid)?;
        let total = prefix + chunk_frames;
        let up = sf.upsample;
        let speakers = sf.speakers;
        let high = host_values(&probs)?;
        let row = |t: usize| high[t * speakers..(t + 1) * speakers].to_vec();
        // 80 ms state predictions: the mean of 8 sub-frames (an in-order sum,
        // then a multiply), zero past the valid length.
        let pooled = (0..total)
            .map(|t| {
                if t >= valid {
                    return vec![0f32; speakers];
                }
                let mut sum = vec![0f32; speakers];
                for u in 0..up {
                    for (acc, v) in sum.iter_mut().zip(&high[(t * up + u) * speakers..]) {
                        *acc += v;
                    }
                }
                let scale = (1.0 / up as f64) as f32;
                sum.into_iter().map(|v| v * scale).collect()
            })
            .collect::<Vec<_>>();
        let start = prefix + lc;
        let preds_10ms = ((start * up)..((start + core) * up))
            .map(row)
            .collect::<Vec<_>>();
        let preds_80ms = pooled[start..start + core].to_vec();
        if update_state {
            let chunk_rows = host_values(&chunk.narrow(0, lc, core)?)?
                .chunks_exact(width)
                .map(<[f32]>::to_vec)
                .collect::<Vec<_>>();
            self.update(state, chunk_rows, &pooled, lc)?;
        }
        Ok(SortformerWindowOutput {
            preds_80ms,
            preds_10ms,
        })
    }

    /// NeMo's `streaming_update`: the window core joins the FIFO; an
    /// overflowing FIFO pops into the speaker cache, which compresses past
    /// its length.
    fn update(
        &self,
        state: &mut SortformerState,
        chunk: Vec<Vec<f32>>,
        preds: &[Vec<f32>],
        lc: usize,
    ) -> Result<()> {
        let sf = &self.config;
        let cache_len = state.cache.len();
        let fifo_len = state.fifo.len();
        state.fifo_preds = preds[cache_len..cache_len + fifo_len].to_vec();
        let chunk_preds =
            preds[cache_len + fifo_len + lc..cache_len + fifo_len + lc + chunk.len()].to_vec();
        let chunk_len = chunk.len();
        state.fifo.extend(chunk);
        state.fifo_preds.extend(chunk_preds);
        if fifo_len + chunk_len > sf.fifo_len {
            let overflow = fifo_len + chunk_len - sf.fifo_len;
            let pop = sf.update_period.max(overflow).min(fifo_len + chunk_len);
            let popped = state.fifo.drain(..pop).collect::<Vec<_>>();
            let popped_preds = state.fifo_preds.drain(..pop).collect::<Vec<_>>();
            let previous = state.cache.len();
            state.cache.extend(popped);
            if let Some(cache_preds) = state.cache_preds.as_mut() {
                cache_preds.extend(popped_preds);
            } else if state.cache.len() > sf.cache_len {
                let mut seeded = preds[..previous].to_vec();
                seeded.extend(popped_preds);
                state.cache_preds = Some(seeded);
            }
            if state.cache.len() > sf.cache_len {
                let cache_preds = state.cache_preds.take().ok_or_else(|| {
                    Error::InferenceError("Sortformer cache lost its predictions".into())
                })?;
                let (cache, cache_preds) = compress(sf, &state.cache, &cache_preds, &self.silence);
                state.cache = cache;
                state.cache_preds = Some(cache_preds);
                state.compressions += 1;
            }
        }
        Ok(())
    }
}

const LN2: f32 = std::f32::consts::LN_2;

/// NeMo's `_compress_spkcache` (eval): keep `cache_len` frames, chosen per
/// speaker by boosted log-odds scores, with disabled slots and the silence
/// frames taking the learnable silence embedding and zero predictions.
pub(crate) fn compress(
    sf: &SortformerConfig,
    embeddings: &[Vec<f32>],
    preds: &[Vec<f32>],
    silence: &[f32],
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let speakers = sf.speakers;
    let per_speaker = sf.cache_len / speakers - sf.sil_frames_per_speaker;
    let strong = (per_speaker as f32 * sf.strong_boost_rate).floor() as usize;
    let weak = (per_speaker as f32 * sf.weak_boost_rate).floor() as usize;
    let min_positive = (per_speaker as f32 * sf.min_pos_scores_rate).floor() as usize;
    let threshold = sf.pred_score_threshold;
    let mut scores = preds
        .iter()
        .map(|p| {
            let log_not = (0..speakers)
                .map(|s| (1.0 - p[s]).clamp(threshold, 1.0).ln())
                .collect::<Vec<_>>();
            let sum_not = log_not.iter().sum::<f32>();
            (0..speakers)
                .map(|s| p[s].clamp(threshold, 1.0).ln() - log_not[s] + sum_not - 0.5f32.ln())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    // Disable non-speech, then (for speakers with enough positive frames)
    // non-positive scores.
    let mut positive = vec![0usize; speakers];
    for (p, row) in preds.iter().zip(scores.iter_mut()) {
        for s in 0..speakers {
            if p[s] <= 0.5 {
                row[s] = f32::NEG_INFINITY;
            } else if row[s] > 0.0 {
                positive[s] += 1;
            }
        }
    }
    for (p, row) in preds.iter().zip(scores.iter_mut()) {
        for s in 0..speakers {
            if p[s] > 0.5 && row[s].is_finite() && row[s] <= 0.0 && positive[s] >= min_positive {
                row[s] = f32::NEG_INFINITY;
            }
        }
    }
    if sf.scores_boost_latest > 0.0 && embeddings.len() > sf.cache_len {
        for row in scores.iter_mut().skip(sf.cache_len) {
            for score in row.iter_mut().filter(|s| s.is_finite()) {
                *score += sf.scores_boost_latest;
            }
        }
    }
    for (count, factor) in [(strong, 2.0f32), (weak, 1.0)] {
        if count == 0 {
            continue;
        }
        for s in 0..speakers {
            let mut ranked = scores
                .iter()
                .enumerate()
                .filter_map(|(t, row)| row[s].is_finite().then_some((row[s], t)))
                .collect::<Vec<_>>();
            ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            for (_, t) in ranked.into_iter().take(count) {
                scores[t][s] += factor * LN2;
            }
        }
    }
    // Speaker-major candidates, each speaker's silence frames last at +inf.
    let span = embeddings.len() + sf.sil_frames_per_speaker;
    let mut candidates = Vec::with_capacity(span * speakers);
    for s in 0..speakers {
        for (t, row) in scores.iter().enumerate() {
            candidates.push((s * span + t, Some(t), row[s]));
        }
        for i in 0..sf.sil_frames_per_speaker {
            candidates.push((s * span + embeddings.len() + i, None, f32::INFINITY));
        }
    }
    candidates.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
    let mut selected = candidates
        .into_iter()
        .take(sf.cache_len)
        .collect::<Vec<_>>();
    // Disabled (-inf) picks become the placeholder index, so they sort last.
    selected.sort_by_key(|c| (c.2 == f32::NEG_INFINITY, c.0));
    let mut cache = Vec::with_capacity(sf.cache_len);
    let mut cache_preds = Vec::with_capacity(sf.cache_len);
    for (_, frame, score) in selected {
        match frame {
            Some(t) if score.is_finite() => {
                cache.push(embeddings[t].clone());
                cache_preds.push(preds[t].clone());
            }
            _ => {
                cache.push(silence.to_vec());
                cache_preds.push(vec![0.0; speakers]);
            }
        }
    }
    (cache, cache_preds)
}

/// `_align_diar_frames`: repeat the last frame, or truncate, to `frames`.
pub fn align_frames(preds: &[Vec<f32>], frames: usize, speakers: usize) -> Vec<Vec<f32>> {
    let mut out = preds.iter().take(frames).cloned().collect::<Vec<_>>();
    let last = preds.last().cloned().unwrap_or_else(|| vec![0.0; speakers]);
    while out.len() < frames {
        out.push(last.clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SortformerConfig {
        super::super::config::tests::published().sortformer
    }

    #[test]
    fn compression_keeps_the_cache_length_with_silence_frames_last() {
        let sf = config();
        let frames = 230;
        let speakers = sf.speakers;
        // Speaker 0 talks in the first half, speaker 1 in the second.
        let preds = (0..frames)
            .map(|t| {
                let mut p = vec![0.01f32; speakers];
                p[usize::from(t >= frames / 2)] = 0.9 + 0.0001 * (t % 7) as f32;
                p
            })
            .collect::<Vec<_>>();
        let embeddings = (0..frames).map(|t| vec![t as f32; 4]).collect::<Vec<_>>();
        let silence = vec![-1.0f32; 4];
        let (cache, cache_preds) = compress(&sf, &embeddings, &preds, &silence);
        assert_eq!(cache.len(), sf.cache_len);
        assert_eq!(cache_preds.len(), sf.cache_len);
        // Each speaker's silence frame sits at the end of its block (flat
        // speaker-major order), with zero predictions; every kept frame is
        // speech. The top-188 is global, so two busy speakers hold every
        // slot but the eight silence frames.
        let kept = cache.iter().filter(|row| row != &&silence).count();
        assert_eq!(kept, sf.cache_len - speakers * sf.sil_frames_per_speaker);
        for (row, p) in cache.iter().zip(&cache_preds) {
            if row == &silence {
                assert!(p.iter().all(|v| *v == 0.0));
            } else {
                assert!(p.iter().any(|v| *v > 0.5));
            }
        }
        // Kept frames are in speaker-major order and in time order within a
        // speaker.
        let times = cache
            .iter()
            .filter(|row| row != &&silence)
            .map(|row| row[0] as usize)
            .collect::<Vec<_>>();
        let speaker0 = times.iter().filter(|t| **t < frames / 2).count();
        assert!(times[..speaker0].windows(2).all(|w| w[0] < w[1]));
        assert!(times[speaker0..].windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn aligned_frames_repeat_the_last_or_truncate() {
        let preds = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        assert_eq!(align_frames(&preds, 1, 2), vec![vec![0.1, 0.2]]);
        assert_eq!(
            align_frames(&preds, 3, 2),
            vec![vec![0.1, 0.2], vec![0.3, 0.4], vec![0.3, 0.4]]
        );
        assert_eq!(align_frames(&[], 2, 2), vec![vec![0.0, 0.0]; 2]);
    }
}
