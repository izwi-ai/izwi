use izwi_asr_toolkit::{plan_audio_chunks, AsrLongFormConfig, AudioChunk, TranscriptAssembler};
use tokio::sync::mpsc;
use tracing::debug;

use crate::error::{Error, Result};
use crate::runtime::audio_io::{base64_decode, decode_audio_bytes};

use super::super::output::StreamingOutput;
use super::NativeExecutor;

impl NativeExecutor {
    // LFM2/LFM2.5 detokenizers rely on overlap-add reconstruction; the newest
    // tail can shift when subsequent frames arrive. Hold back one overlap
    // region to avoid streaming seam artifacts.
    pub(super) const LFM2_STREAM_TAIL_HOLDBACK_SAMPLES: usize = 960;

    pub(super) fn env_f32(key: &str) -> Option<f32> {
        std::env::var(key)
            .ok()
            .and_then(|raw| raw.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
    }

    pub(super) fn asr_long_form_config() -> AsrLongFormConfig {
        let mut cfg = AsrLongFormConfig::default();
        if let Some(v) = Self::env_f32("IZWI_ASR_CHUNK_TARGET_SECS") {
            cfg.target_chunk_secs = v;
        }
        if let Some(v) = Self::env_f32("IZWI_ASR_CHUNK_MAX_SECS") {
            cfg.hard_max_chunk_secs = v;
        }
        if let Some(v) = Self::env_f32("IZWI_ASR_CHUNK_OVERLAP_SECS") {
            cfg.overlap_secs = v;
        }
        cfg
    }

    pub(super) fn asr_chunk_plan(
        samples: &[f32],
        sample_rate: u32,
        model_max_chunk_secs: Option<f32>,
    ) -> (AsrLongFormConfig, Vec<AudioChunk>) {
        let cfg = Self::asr_long_form_config();
        let tuned_limit = model_max_chunk_secs
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|v| (v * 0.95).max(cfg.min_chunk_secs.max(1.0)));
        let chunks = plan_audio_chunks(samples, sample_rate, &cfg, tuned_limit);
        (cfg, chunks)
    }

    pub(super) fn transcribe_with_chunk_plan<F>(
        request_id: &str,
        stream_tx: Option<&mpsc::UnboundedSender<StreamingOutput>>,
        sequence: &mut usize,
        samples: &[f32],
        sample_rate: u32,
        chunk_plan: &[AudioChunk],
        chunk_cfg: &AsrLongFormConfig,
        mut transcribe_chunk: F,
    ) -> Result<String>
    where
        F: FnMut(&[f32], u32) -> Result<String>,
    {
        if chunk_plan.is_empty() {
            return Err(Error::InvalidInput(
                "ASR chunk planner produced no chunks".to_string(),
            ));
        }

        debug!(
            "ASR long-form chunking enabled for request {}: {} chunks (~{:.1}s audio)",
            request_id,
            chunk_plan.len(),
            samples.len() as f32 / sample_rate.max(1) as f32
        );

        let mut assembler = TranscriptAssembler::new(chunk_cfg.clone());
        for chunk in chunk_plan {
            if chunk.end_sample <= chunk.start_sample || chunk.end_sample > samples.len() {
                continue;
            }
            let chunk_audio = &samples[chunk.start_sample..chunk.end_sample];
            let chunk_text = transcribe_chunk(chunk_audio, sample_rate)?;
            let delta = assembler.push_chunk_text(&chunk_text);
            if !delta.is_empty() {
                if let Some(tx) = stream_tx {
                    Self::stream_text(tx, request_id, sequence, delta)?;
                }
            }
        }

        if let Some(tx) = stream_tx {
            Self::stream_final_marker(tx, request_id, sequence)?;
        }

        Ok(assembler.finish())
    }

    pub(super) fn next_audio_delta(all_samples: &[f32], emitted_samples: &mut usize) -> Vec<f32> {
        let start = (*emitted_samples).min(all_samples.len());
        let delta = all_samples[start..].to_vec();
        *emitted_samples = all_samples.len();
        delta
    }

    pub(super) fn next_audio_delta_stable(
        all_samples: &[f32],
        emitted_samples: &mut usize,
        holdback_samples: usize,
        is_final: bool,
    ) -> Vec<f32> {
        let stable_end = if is_final {
            all_samples.len()
        } else {
            all_samples.len().saturating_sub(holdback_samples)
        };
        let start = (*emitted_samples).min(stable_end);
        let delta = all_samples[start..stable_end].to_vec();
        *emitted_samples = stable_end;
        delta
    }
}

pub(super) fn decode_audio_base64_with_rate(audio_b64: &str) -> Result<(Vec<f32>, u32)> {
    let audio_bytes = base64_decode(audio_b64)?;
    decode_audio_bytes(&audio_bytes)
}

#[cfg(test)]
mod tests {
    use super::NativeExecutor;

    #[test]
    fn next_audio_delta_emits_only_new_tail_samples() {
        let mut emitted = 0usize;
        let all1 = vec![0.1f32, 0.2, 0.3];
        let delta1 = NativeExecutor::next_audio_delta(&all1, &mut emitted);
        assert_eq!(delta1, all1);
        assert_eq!(emitted, 3);

        let all2 = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let delta2 = NativeExecutor::next_audio_delta(&all2, &mut emitted);
        assert_eq!(delta2, vec![0.4, 0.5]);
        assert_eq!(emitted, 5);
    }

    #[test]
    fn next_audio_delta_handles_shorter_redecode_safely() {
        let mut emitted = 5usize;
        let all = vec![1.0f32, 2.0];
        let delta = NativeExecutor::next_audio_delta(&all, &mut emitted);
        assert!(delta.is_empty());
        assert_eq!(emitted, 2);
    }

    #[test]
    fn next_audio_delta_stable_holds_back_tail_until_final() {
        let mut emitted = 0usize;
        let all = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let delta = NativeExecutor::next_audio_delta_stable(&all, &mut emitted, 2, false);
        assert_eq!(delta, vec![0.1, 0.2, 0.3]);
        assert_eq!(emitted, 3);

        let delta_final = NativeExecutor::next_audio_delta_stable(&all, &mut emitted, 2, true);
        assert_eq!(delta_final, vec![0.4, 0.5]);
        assert_eq!(emitted, 5);
    }

    #[test]
    fn next_audio_delta_stable_emits_nothing_when_window_is_unstable() {
        let mut emitted = 0usize;
        let all = vec![0.1f32, 0.2, 0.3];
        let delta = NativeExecutor::next_audio_delta_stable(&all, &mut emitted, 8, false);
        assert!(delta.is_empty());
        assert_eq!(emitted, 0);
    }
}
