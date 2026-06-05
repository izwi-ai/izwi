use izwi_asr_toolkit::{
    plan_audio_chunks, plan_speech_audio_chunks, AsrLongFormConfig, AsrSpeechChunkConfig,
    AudioChunk, SpeechChunkPlan, TranscriptAssembler,
};
use serde_json::json;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::debug;

use crate::engine::EngineCoreRequest;
use crate::error::{Error, Result};
use crate::runtime::audio_io::{base64_decode, decode_audio_bytes};

use super::super::output::StreamingOutput;
use super::super::request::EngineStreamPolicy;
use super::NativeExecutor;

const DEFAULT_STREAM_SHORT_TARGET_CHUNK_SECS: f32 = 2.4;
const DEFAULT_STREAM_SHORT_MAX_CHUNK_SECS: f32 = 3.2;
const DEFAULT_STREAM_SHORT_OVERLAP_SECS: f32 = 0.45;
const DEFAULT_STREAM_SHORT_MIN_CHUNK_SECS: f32 = 1.0;
const DEFAULT_STREAM_SHORT_SILENCE_SEARCH_SECS: f32 = 0.75;
const DEFAULT_STREAM_LONG_TARGET_CHUNK_SECS: f32 = 6.0;
const DEFAULT_STREAM_LONG_MAX_CHUNK_SECS: f32 = 8.0;
const DEFAULT_STREAM_LONG_OVERLAP_SECS: f32 = 1.0;
const DEFAULT_STREAM_LONG_MIN_CHUNK_SECS: f32 = 2.0;
const DEFAULT_STREAM_LONG_SILENCE_SEARCH_SECS: f32 = 1.5;
const STREAM_SHORT_AUDIO_SECS_THRESHOLD: f32 = 12.0;
const STREAM_SINGLE_CHUNK_SECS_THRESHOLD: f32 = 4.5;
const MAX_STREAMING_LOW_LATENCY_CHUNKS: usize = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AsrChunkPlannerKind {
    Duration,
    Speech,
}

#[derive(Debug, Clone)]
pub(crate) struct PlannedAsrChunks {
    pub(crate) config: AsrLongFormConfig,
    pub(crate) chunks: Vec<AudioChunk>,
    pub(crate) planner: AsrChunkPlannerKind,
    pub(crate) speech_plan: Option<SpeechChunkPlan>,
    pub(crate) input_samples: usize,
    pub(crate) sample_rate: u32,
    pub(crate) planning_ms: f64,
}

#[derive(Debug, Clone)]
pub(super) struct AsrChunkTranscription {
    pub(super) text: String,
    pub(super) diagnostics: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub(super) struct ChunkedAsrTranscription {
    pub(super) text: String,
    pub(super) chunk_diagnostics: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct AsrChunkStreamOptions {
    pub(super) stable_text_holdback_chars: usize,
}

impl PlannedAsrChunks {
    pub(crate) fn requires_chunk_path(&self) -> bool {
        if self
            .speech_plan
            .as_ref()
            .map(|plan| plan.no_speech)
            .unwrap_or(false)
        {
            return true;
        }
        if self.chunks.len() > 1 {
            return true;
        }
        if self.planner != AsrChunkPlannerKind::Speech {
            return false;
        }
        let Some(chunk) = self.chunks.first() else {
            return false;
        };
        chunk.start_sample > 0 || chunk.end_sample < self.input_samples
    }

    pub(crate) fn diagnostics(&self) -> serde_json::Value {
        let audio_secs = if self.sample_rate > 0 {
            self.input_samples as f32 / self.sample_rate as f32
        } else {
            0.0
        };
        let planner = match self.planner {
            AsrChunkPlannerKind::Duration => "duration",
            AsrChunkPlannerKind::Speech => "speech",
        };
        let chunks = self
            .chunks
            .iter()
            .map(|chunk| {
                let start_secs = samples_to_seconds(chunk.start_sample, self.sample_rate);
                let end_secs = samples_to_seconds(chunk.end_sample, self.sample_rate);
                json!({
                    "start_sample": chunk.start_sample,
                    "end_sample": chunk.end_sample,
                    "start_seconds": start_secs,
                    "end_seconds": end_secs,
                    "duration_seconds": (end_secs - start_secs).max(0.0),
                })
            })
            .collect::<Vec<_>>();
        let speech = self.speech_plan.as_ref().map(|plan| {
            let speech_regions = plan
                .speech_regions
                .iter()
                .map(|region| {
                    let start_secs = samples_to_seconds(region.start_sample, self.sample_rate);
                    let end_secs = samples_to_seconds(region.end_sample, self.sample_rate);
                    json!({
                        "start_sample": region.start_sample,
                        "end_sample": region.end_sample,
                        "start_seconds": start_secs,
                        "end_seconds": end_secs,
                        "duration_seconds": (end_secs - start_secs).max(0.0),
                    })
                })
                .collect::<Vec<_>>();
            json!({
                "no_speech": plan.no_speech,
                "speech_region_count": plan.speech_regions.len(),
                "speech_samples": plan.speech_samples,
                "speech_seconds": samples_to_seconds(plan.speech_samples, self.sample_rate),
                "included_samples": plan.included_samples,
                "included_seconds": samples_to_seconds(plan.included_samples, self.sample_rate),
                "skipped_samples": plan.skipped_samples,
                "skipped_seconds": samples_to_seconds(plan.skipped_samples, self.sample_rate),
                "speech_regions": speech_regions,
            })
        });
        json!({
            "chunking": {
                "planner": planner,
                "chunk_count": self.chunks.len(),
                "input_samples": self.input_samples,
                "sample_rate": self.sample_rate,
                "audio_seconds": audio_secs,
                "planning_ms": self.planning_ms,
                "config": {
                    "target_chunk_seconds": self.config.target_chunk_secs,
                    "hard_max_chunk_seconds": self.config.hard_max_chunk_secs,
                    "overlap_seconds": self.config.overlap_secs,
                    "silence_search_seconds": self.config.silence_search_secs,
                    "min_chunk_seconds": self.config.min_chunk_secs,
                    "analysis_frame_ms": self.config.analysis_frame_ms,
                    "min_word_overlap": self.config.min_word_overlap,
                    "max_word_overlap": self.config.max_word_overlap,
                    "min_context_replay_words": self.config.min_context_replay_words,
                    "max_context_replay_words": self.config.max_context_replay_words,
                },
                "chunks": chunks,
                "speech": speech,
            }
        })
    }

    pub(crate) fn diagnostics_with_chunk_transcriptions(
        &self,
        chunk_transcriptions: Vec<serde_json::Value>,
    ) -> serde_json::Value {
        let mut diagnostics = self.diagnostics();
        if let Some(chunking) = diagnostics
            .get_mut("chunking")
            .and_then(|value| value.as_object_mut())
        {
            chunking.insert(
                "chunk_transcriptions".to_string(),
                serde_json::Value::Array(chunk_transcriptions),
            );
        }
        diagnostics
    }
}

fn samples_to_seconds(samples: usize, sample_rate: u32) -> f32 {
    if sample_rate > 0 {
        samples as f32 / sample_rate as f32
    } else {
        0.0
    }
}

impl NativeExecutor {
    pub(super) fn env_f32(key: &str) -> Option<f32> {
        std::env::var(key)
            .ok()
            .and_then(|raw| raw.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
    }

    pub(super) fn env_bool(key: &str) -> Option<bool> {
        std::env::var(key).ok().and_then(|raw| {
            let normalized = raw.trim().to_ascii_lowercase();
            match normalized.as_str() {
                "1" | "true" | "yes" | "on" => Some(true),
                "0" | "false" | "no" | "off" => Some(false),
                _ => None,
            }
        })
    }

    pub(super) fn env_usize(key: &str) -> Option<usize> {
        std::env::var(key)
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
    }

    pub(super) fn qwen_asr_chunk_stream_options() -> AsrChunkStreamOptions {
        let rollback_tokens = Self::env_usize("IZWI_QWEN_ASR_STREAM_ROLLBACK_TOKENS")
            .unwrap_or(5)
            .min(32);
        let chars_per_token = Self::env_usize("IZWI_QWEN_ASR_STREAM_CHARS_PER_TOKEN")
            .unwrap_or(4)
            .clamp(1, 12);
        let stable_text_holdback_chars = Self::env_usize("IZWI_QWEN_ASR_STREAM_HOLDBACK_CHARS")
            .unwrap_or_else(|| rollback_tokens.saturating_mul(chars_per_token));
        AsrChunkStreamOptions {
            stable_text_holdback_chars,
        }
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

    pub(super) fn asr_streaming_low_latency_config(audio_secs: f32) -> AsrLongFormConfig {
        let mut cfg = Self::asr_long_form_config();
        let stream_chunk_overrides_present = Self::stream_chunk_overrides_present();
        if audio_secs <= STREAM_SHORT_AUDIO_SECS_THRESHOLD {
            cfg.target_chunk_secs = DEFAULT_STREAM_SHORT_TARGET_CHUNK_SECS;
            cfg.hard_max_chunk_secs = DEFAULT_STREAM_SHORT_MAX_CHUNK_SECS;
            cfg.overlap_secs = DEFAULT_STREAM_SHORT_OVERLAP_SECS;
            cfg.min_chunk_secs = DEFAULT_STREAM_SHORT_MIN_CHUNK_SECS;
            cfg.silence_search_secs = DEFAULT_STREAM_SHORT_SILENCE_SEARCH_SECS;
        } else {
            cfg.target_chunk_secs = cfg
                .target_chunk_secs
                .min(DEFAULT_STREAM_LONG_TARGET_CHUNK_SECS);
            cfg.hard_max_chunk_secs = cfg
                .hard_max_chunk_secs
                .min(DEFAULT_STREAM_LONG_MAX_CHUNK_SECS);
            cfg.overlap_secs = cfg.overlap_secs.min(DEFAULT_STREAM_LONG_OVERLAP_SECS);
            cfg.min_chunk_secs = cfg.min_chunk_secs.min(DEFAULT_STREAM_LONG_MIN_CHUNK_SECS);
            cfg.silence_search_secs = cfg
                .silence_search_secs
                .min(DEFAULT_STREAM_LONG_SILENCE_SEARCH_SECS);
        }

        if let Some(v) = Self::env_f32("IZWI_ASR_STREAM_CHUNK_TARGET_SECS") {
            cfg.target_chunk_secs = v;
        }
        if let Some(v) = Self::env_f32("IZWI_ASR_STREAM_CHUNK_MAX_SECS") {
            cfg.hard_max_chunk_secs = v;
        }
        if let Some(v) = Self::env_f32("IZWI_ASR_STREAM_CHUNK_OVERLAP_SECS") {
            cfg.overlap_secs = v;
        }
        if let Some(v) = Self::env_f32("IZWI_ASR_STREAM_CHUNK_MIN_SECS") {
            cfg.min_chunk_secs = v;
        }
        if let Some(v) = Self::env_f32("IZWI_ASR_STREAM_CHUNK_SILENCE_SEARCH_SECS") {
            cfg.silence_search_secs = v;
        }

        if cfg.hard_max_chunk_secs < cfg.min_chunk_secs {
            cfg.hard_max_chunk_secs = cfg.min_chunk_secs;
        }
        if cfg.target_chunk_secs < cfg.min_chunk_secs {
            cfg.target_chunk_secs = cfg.min_chunk_secs;
        }
        if cfg.target_chunk_secs > cfg.hard_max_chunk_secs {
            cfg.target_chunk_secs = cfg.hard_max_chunk_secs;
        }
        if cfg.overlap_secs > cfg.target_chunk_secs * 0.45 {
            cfg.overlap_secs = cfg.target_chunk_secs * 0.45;
        }
        if cfg.silence_search_secs > cfg.target_chunk_secs * 0.5 {
            cfg.silence_search_secs = cfg.target_chunk_secs * 0.5;
        }
        if audio_secs <= STREAM_SHORT_AUDIO_SECS_THRESHOLD {
            let desired_overlap_floor = (cfg.target_chunk_secs * 0.2).min(0.5);
            if cfg.overlap_secs < desired_overlap_floor {
                cfg.overlap_secs = desired_overlap_floor;
            }

            if !stream_chunk_overrides_present && audio_secs <= STREAM_SINGLE_CHUNK_SECS_THRESHOLD {
                let single_chunk_secs = (audio_secs + 0.05).max(cfg.min_chunk_secs.max(0.5));
                cfg.target_chunk_secs = single_chunk_secs;
                cfg.hard_max_chunk_secs = single_chunk_secs;
                cfg.overlap_secs = cfg.overlap_secs.min(single_chunk_secs * 0.1);
                cfg.silence_search_secs = cfg.silence_search_secs.min(single_chunk_secs * 0.25);
            }
        }

        cfg
    }

    fn stream_chunk_overrides_present() -> bool {
        [
            "IZWI_ASR_STREAM_CHUNK_TARGET_SECS",
            "IZWI_ASR_STREAM_CHUNK_MAX_SECS",
            "IZWI_ASR_STREAM_CHUNK_OVERLAP_SECS",
            "IZWI_ASR_STREAM_CHUNK_MIN_SECS",
            "IZWI_ASR_STREAM_CHUNK_SILENCE_SEARCH_SECS",
        ]
        .iter()
        .any(|key| std::env::var(key).is_ok())
    }

    pub(super) fn asr_chunk_plan(
        samples: &[f32],
        sample_rate: u32,
        model_max_chunk_secs: Option<f32>,
        streaming_low_latency: bool,
        allow_speech_planner: bool,
    ) -> PlannedAsrChunks {
        Self::asr_chunk_plan_with_options(
            samples,
            sample_rate,
            model_max_chunk_secs,
            streaming_low_latency,
            allow_speech_planner,
        )
    }

    fn asr_chunk_plan_with_options(
        samples: &[f32],
        sample_rate: u32,
        model_max_chunk_secs: Option<f32>,
        mut streaming_low_latency: bool,
        allow_speech_planner: bool,
    ) -> PlannedAsrChunks {
        let planning_started = Instant::now();
        if allow_speech_planner {
            streaming_low_latency = false;
        }
        let audio_secs = if sample_rate > 0 {
            samples.len() as f32 / sample_rate as f32
        } else {
            0.0
        };
        let model_fit_limit_secs = model_max_chunk_secs
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|v| v * 0.95);
        if streaming_low_latency
            && !Self::stream_chunk_overrides_present()
            && model_fit_limit_secs
                .map(|limit| audio_secs <= limit)
                .unwrap_or(false)
        {
            let mut single_chunk_cfg = Self::asr_long_form_config();
            let single_chunk_secs = audio_secs.max(single_chunk_cfg.min_chunk_secs.max(0.5));
            single_chunk_cfg.target_chunk_secs = single_chunk_secs;
            single_chunk_cfg.hard_max_chunk_secs = single_chunk_secs;
            single_chunk_cfg.overlap_secs =
                single_chunk_cfg.overlap_secs.min(single_chunk_secs * 0.1);
            single_chunk_cfg.silence_search_secs = single_chunk_cfg
                .silence_search_secs
                .min(single_chunk_secs * 0.25);
            let chunks = plan_audio_chunks(
                samples,
                sample_rate,
                &single_chunk_cfg,
                model_fit_limit_secs.map(|v| v.max(single_chunk_cfg.min_chunk_secs.max(1.0))),
            );
            return PlannedAsrChunks {
                config: single_chunk_cfg,
                chunks,
                planner: AsrChunkPlannerKind::Duration,
                speech_plan: None,
                input_samples: samples.len(),
                sample_rate,
                planning_ms: planning_started.elapsed().as_secs_f64() * 1000.0,
            };
        }

        let cfg = if streaming_low_latency {
            Self::asr_streaming_low_latency_config(audio_secs)
        } else {
            Self::asr_long_form_config()
        };
        let tuned_limit = model_max_chunk_secs
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|v| (v * 0.95).max(cfg.min_chunk_secs.max(1.0)));

        if allow_speech_planner {
            let speech_cfg = AsrSpeechChunkConfig::default();
            let speech_plan =
                plan_speech_audio_chunks(samples, sample_rate, &cfg, &speech_cfg, tuned_limit);
            return PlannedAsrChunks {
                chunks: speech_plan.chunks.clone(),
                config: cfg,
                planner: AsrChunkPlannerKind::Speech,
                speech_plan: Some(speech_plan),
                input_samples: samples.len(),
                sample_rate,
                planning_ms: planning_started.elapsed().as_secs_f64() * 1000.0,
            };
        }

        let mut chunks = plan_audio_chunks(samples, sample_rate, &cfg, tuned_limit);

        if streaming_low_latency && chunks.len() > MAX_STREAMING_LOW_LATENCY_CHUNKS {
            debug!(
                "ASR streaming chunk plan too fragmented ({} chunks), relaxing to long-form defaults",
                chunks.len()
            );
            let fallback_cfg = Self::asr_long_form_config();
            let fallback_limit = model_max_chunk_secs
                .filter(|v| v.is_finite() && *v > 0.0)
                .map(|v| (v * 0.95).max(fallback_cfg.min_chunk_secs.max(1.0)));
            chunks = plan_audio_chunks(samples, sample_rate, &fallback_cfg, fallback_limit);
            return PlannedAsrChunks {
                config: fallback_cfg,
                chunks,
                planner: AsrChunkPlannerKind::Duration,
                speech_plan: None,
                input_samples: samples.len(),
                sample_rate,
                planning_ms: planning_started.elapsed().as_secs_f64() * 1000.0,
            };
        }

        PlannedAsrChunks {
            config: cfg,
            chunks,
            planner: AsrChunkPlannerKind::Duration,
            speech_plan: None,
            input_samples: samples.len(),
            sample_rate,
            planning_ms: planning_started.elapsed().as_secs_f64() * 1000.0,
        }
    }

    pub(super) fn transcribe_with_chunk_plan<F>(
        request_id: &str,
        stream_tx: Option<&mpsc::Sender<StreamingOutput>>,
        stream_policy: EngineStreamPolicy,
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
        Self::transcribe_with_chunk_plan_with_details(
            request_id,
            stream_tx,
            stream_policy,
            sequence,
            samples,
            sample_rate,
            chunk_plan,
            chunk_cfg,
            |chunk_audio, sr| {
                transcribe_chunk(chunk_audio, sr).map(|text| AsrChunkTranscription {
                    text,
                    diagnostics: None,
                })
            },
        )
        .map(|result| result.text)
    }

    pub(super) fn transcribe_with_chunk_plan_with_details<F>(
        request_id: &str,
        stream_tx: Option<&mpsc::Sender<StreamingOutput>>,
        stream_policy: EngineStreamPolicy,
        sequence: &mut usize,
        samples: &[f32],
        sample_rate: u32,
        chunk_plan: &[AudioChunk],
        chunk_cfg: &AsrLongFormConfig,
        transcribe_chunk: F,
    ) -> Result<ChunkedAsrTranscription>
    where
        F: FnMut(&[f32], u32) -> Result<AsrChunkTranscription>,
    {
        Self::transcribe_with_chunk_plan_with_details_and_options(
            request_id,
            stream_tx,
            stream_policy,
            sequence,
            samples,
            sample_rate,
            chunk_plan,
            chunk_cfg,
            AsrChunkStreamOptions::default(),
            transcribe_chunk,
        )
    }

    pub(super) fn transcribe_with_chunk_plan_with_details_and_options<F>(
        request_id: &str,
        stream_tx: Option<&mpsc::Sender<StreamingOutput>>,
        stream_policy: EngineStreamPolicy,
        sequence: &mut usize,
        samples: &[f32],
        sample_rate: u32,
        chunk_plan: &[AudioChunk],
        chunk_cfg: &AsrLongFormConfig,
        stream_options: AsrChunkStreamOptions,
        mut transcribe_chunk: F,
    ) -> Result<ChunkedAsrTranscription>
    where
        F: FnMut(&[f32], u32) -> Result<AsrChunkTranscription>,
    {
        if chunk_plan.is_empty() {
            if let Some(tx) = stream_tx {
                Self::stream_final_marker_with_policy(tx, stream_policy, request_id, sequence)?;
            }
            return Ok(ChunkedAsrTranscription {
                text: String::new(),
                chunk_diagnostics: Vec::new(),
            });
        }

        debug!(
            "ASR long-form chunking enabled for request {}: {} chunks (~{:.1}s audio)",
            request_id,
            chunk_plan.len(),
            samples.len() as f32 / sample_rate.max(1) as f32
        );

        let mut assembler = TranscriptAssembler::new(chunk_cfg.clone());
        let mut deferred_boundary_delta = String::new();
        let mut emitted_stream_chars = 0usize;
        let stable_text_streaming =
            stream_tx.is_some() && stream_options.stable_text_holdback_chars > 0;
        let mut chunk_diagnostics = Vec::new();
        for (idx, chunk) in chunk_plan.iter().enumerate() {
            if chunk.end_sample <= chunk.start_sample || chunk.end_sample > samples.len() {
                chunk_diagnostics.push(Self::chunk_transcription_diagnostics(
                    idx,
                    chunk,
                    sample_rate,
                    0.0,
                    "",
                    Some(json!({
                        "skipped": true,
                        "skip_reason": "invalid_bounds",
                    })),
                ));
                continue;
            }
            let chunk_audio = &samples[chunk.start_sample..chunk.end_sample];
            let chunk_started = Instant::now();
            let chunk_result = transcribe_chunk(chunk_audio, sample_rate)?;
            let chunk_text = chunk_result.text.clone();
            let transcribe_ms = chunk_started.elapsed().as_secs_f64() * 1000.0;
            chunk_diagnostics.push(Self::chunk_transcription_diagnostics(
                idx,
                chunk,
                sample_rate,
                transcribe_ms,
                &chunk_text,
                chunk_result.diagnostics,
            ));
            let mut delta = assembler.push_chunk_text(&chunk_text);
            let is_last_chunk = idx + 1 == chunk_plan.len();

            if stable_text_streaming {
                if let Some(tx) = stream_tx {
                    let stable_delta = Self::next_text_delta_stable(
                        assembler.text(),
                        &mut emitted_stream_chars,
                        stream_options.stable_text_holdback_chars,
                        is_last_chunk,
                    );
                    if !stable_delta.is_empty() {
                        Self::stream_text_with_policy(
                            tx,
                            stream_policy,
                            request_id,
                            sequence,
                            stable_delta,
                        )?;
                    }
                }
                continue;
            }

            if !deferred_boundary_delta.is_empty() {
                deferred_boundary_delta.push_str(&delta);
                delta = std::mem::take(&mut deferred_boundary_delta);
            }

            if !is_last_chunk && is_boundary_noise_delta(&delta) {
                deferred_boundary_delta.push_str(&delta);
                continue;
            }

            if !delta.is_empty() {
                if let Some(tx) = stream_tx {
                    Self::stream_text_with_policy(tx, stream_policy, request_id, sequence, delta)?;
                }
            }
        }

        if stable_text_streaming {
            if let Some(tx) = stream_tx {
                let final_delta = Self::next_text_delta_stable(
                    assembler.text(),
                    &mut emitted_stream_chars,
                    0,
                    true,
                );
                if !final_delta.is_empty() {
                    Self::stream_text_with_policy(
                        tx,
                        stream_policy,
                        request_id,
                        sequence,
                        final_delta,
                    )?;
                }
            }
        }

        if !deferred_boundary_delta.is_empty() {
            if let Some(tx) = stream_tx {
                Self::stream_text_with_policy(
                    tx,
                    stream_policy,
                    request_id,
                    sequence,
                    deferred_boundary_delta,
                )?;
            }
        }

        if let Some(tx) = stream_tx {
            Self::stream_final_marker_with_policy(tx, stream_policy, request_id, sequence)?;
        }

        Ok(ChunkedAsrTranscription {
            text: assembler.finish(),
            chunk_diagnostics,
        })
    }

    fn chunk_transcription_diagnostics(
        index: usize,
        chunk: &AudioChunk,
        sample_rate: u32,
        transcribe_ms: f64,
        text: &str,
        model_diagnostics: Option<serde_json::Value>,
    ) -> serde_json::Value {
        let start_seconds = samples_to_seconds(chunk.start_sample, sample_rate);
        let end_seconds = samples_to_seconds(chunk.end_sample, sample_rate);
        let mut payload = json!({
            "index": index,
            "start_sample": chunk.start_sample,
            "end_sample": chunk.end_sample,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "duration_seconds": (end_seconds - start_seconds).max(0.0),
            "transcribe_ms": transcribe_ms,
            "text_chars": text.chars().count(),
            "text_empty": text.trim().is_empty(),
        });
        if let Some(model_diagnostics) = model_diagnostics {
            if let Some(payload) = payload.as_object_mut() {
                payload.insert("model_diagnostics".to_string(), model_diagnostics);
            }
        }
        payload
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

    pub(super) fn next_text_delta_stable(
        all_text: &str,
        emitted_chars: &mut usize,
        holdback_chars: usize,
        is_final: bool,
    ) -> String {
        let total_chars = all_text.chars().count();
        let stable_end = if is_final {
            total_chars
        } else {
            total_chars.saturating_sub(holdback_chars)
        };
        let start = (*emitted_chars).min(stable_end);
        if start >= stable_end {
            return String::new();
        }

        let start_byte = char_to_byte_index(all_text, start);
        let end_byte = char_to_byte_index(all_text, stable_end);
        *emitted_chars = stable_end;
        all_text[start_byte..end_byte].to_string()
    }
}

fn char_to_byte_index(text: &str, char_idx: usize) -> usize {
    if char_idx == 0 {
        return 0;
    }
    text.char_indices()
        .nth(char_idx)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len())
}

pub(super) fn decode_audio_base64_with_rate(audio_b64: &str) -> Result<(Vec<f32>, u32)> {
    let audio_bytes = base64_decode(audio_b64)?;
    decode_audio_bytes(&audio_bytes)
}

pub(super) fn decode_request_audio_with_rate(
    request: &EngineCoreRequest,
) -> Result<(Vec<f32>, u32)> {
    if let Some(audio_bytes) = request.audio_bytes.as_deref() {
        return decode_audio_bytes(audio_bytes);
    }

    let audio_b64 = request
        .audio_input
        .as_deref()
        .ok_or_else(|| Error::InvalidInput("Request missing audio input".to_string()))?;
    decode_audio_base64_with_rate(audio_b64)
}

fn is_boundary_noise_delta(delta: &str) -> bool {
    let trimmed = delta.trim();
    if trimmed.is_empty() {
        return true;
    }
    let char_count = trimmed.chars().count();
    char_count <= 4 && trimmed.chars().all(|ch| !ch.is_alphanumeric())
}

#[cfg(test)]
mod tests {
    use izwi_asr_toolkit::AudioChunk;
    use tokio::sync::mpsc;

    use crate::engine::EngineStreamPolicy;

    use super::{AsrChunkPlannerKind, AsrChunkTranscription, NativeExecutor};

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

    #[test]
    fn next_text_delta_stable_holds_back_unstable_tail() {
        let mut emitted = 0usize;

        let first = NativeExecutor::next_text_delta_stable("hello world", &mut emitted, 5, false);
        assert_eq!(first, "hello ");
        assert_eq!(emitted, 6);

        let second =
            NativeExecutor::next_text_delta_stable("hello world again", &mut emitted, 5, false);
        assert_eq!(second, "world ");
        assert_eq!(emitted, 12);

        let final_delta =
            NativeExecutor::next_text_delta_stable("hello world again", &mut emitted, 5, true);
        assert_eq!(final_delta, "again");
        assert_eq!(emitted, 17);
    }

    #[test]
    fn next_text_delta_stable_respects_utf8_boundaries() {
        let mut emitted = 0usize;
        let first = NativeExecutor::next_text_delta_stable("héllo 世界", &mut emitted, 2, false);
        assert_eq!(first, "héllo ");
        assert_eq!(emitted, 6);

        let final_delta =
            NativeExecutor::next_text_delta_stable("héllo 世界", &mut emitted, 2, true);
        assert_eq!(final_delta, "世界");
        assert_eq!(emitted, 8);
    }

    #[test]
    fn streaming_low_latency_chunk_plan_keeps_very_short_audio_single_chunk() {
        let sr = 16_000u32;
        let samples = vec![0.0f32; (sr as usize) * 4];
        let plan = NativeExecutor::asr_chunk_plan(&samples, sr, Some(30.0), true, false);
        assert_eq!(
            plan.chunks.len(),
            1,
            "expected a single chunk, got {}",
            plan.chunks.len()
        );
    }

    #[test]
    fn streaming_low_latency_chunk_plan_splits_medium_audio() {
        let sr = 16_000u32;
        let samples = vec![0.0f32; (sr as usize) * 8];
        let plan = NativeExecutor::asr_chunk_plan(&samples, sr, None, true, false);
        assert!(
            plan.chunks.len() > 1,
            "expected multiple chunks, got {}",
            plan.chunks.len()
        );
    }

    #[test]
    fn streaming_chunk_plan_keeps_model_fit_audio_single_chunk() {
        let sr = 16_000u32;
        let samples = vec![0.0f32; (sr as usize) * 20];
        let plan = NativeExecutor::asr_chunk_plan(&samples, sr, Some(30.0), true, false);
        assert_eq!(
            plan.chunks.len(),
            1,
            "expected a single chunk for model-fit audio, got {}",
            plan.chunks.len()
        );
    }

    #[test]
    fn streaming_chunk_plan_splits_audio_when_model_hint_is_exceeded() {
        let sr = 16_000u32;
        let samples = vec![0.0f32; (sr as usize) * 40];
        let plan = NativeExecutor::asr_chunk_plan(&samples, sr, Some(30.0), true, false);
        assert!(
            plan.chunks.len() > 1,
            "expected chunking beyond model hint, got {}",
            plan.chunks.len()
        );
    }

    #[test]
    fn streaming_chunk_plan_keeps_long_audio_single_chunk_when_model_hint_fits() {
        let sr = 10u32;
        let audio_secs = 50 * 60;
        let samples = vec![0.0f32; sr as usize * audio_secs];

        let plan = NativeExecutor::asr_chunk_plan(&samples, sr, Some(60.0 * 60.0), true, false);

        assert_eq!(
            plan.chunks.len(),
            1,
            "expected model-fit 50-minute audio to stay single chunk"
        );
        assert!(
            !plan.requires_chunk_path(),
            "model-fit single chunk should bypass chunked transcription"
        );
    }

    #[test]
    fn streaming_chunk_plan_forces_long_audio_chunk_path_before_model_prefill() {
        let sr = 10u32;
        let audio_secs = 50 * 60;
        let model_hint_secs = 120.0f32;
        let model_fit_limit_secs = model_hint_secs * 0.95;
        let samples = vec![0.0f32; sr as usize * audio_secs];

        let plan = NativeExecutor::asr_chunk_plan(&samples, sr, Some(model_hint_secs), true, false);

        assert!(
            plan.requires_chunk_path(),
            "expected 50-minute model-limited audio to use chunked transcription"
        );
        assert!(
            plan.chunks.len() > 1,
            "expected multiple chunks, got {}",
            plan.chunks.len()
        );
        let max_chunk_secs = plan
            .chunks
            .iter()
            .map(|chunk| chunk.len_samples() as f32 / sr as f32)
            .fold(0.0f32, f32::max);
        assert!(
            max_chunk_secs <= model_fit_limit_secs + 1.0 / sr as f32,
            "expected chunk duration <= {model_fit_limit_secs}s, got {max_chunk_secs}s"
        );
    }

    #[test]
    fn whisper_streaming_chunk_plan_keeps_standard_long_form_chunks() {
        let sr = 16_000u32;
        let samples = vec![0.0f32; (sr as usize) * 40];
        let streaming = NativeExecutor::asr_chunk_plan(&samples, sr, Some(30.0), true, true);
        let standard = NativeExecutor::asr_chunk_plan(&samples, sr, Some(30.0), false, true);

        assert_eq!(
            streaming.config.target_chunk_secs,
            standard.config.target_chunk_secs
        );
        assert_eq!(
            streaming.config.hard_max_chunk_secs,
            standard.config.hard_max_chunk_secs
        );
        assert_eq!(streaming.config.overlap_secs, standard.config.overlap_secs);
        assert_eq!(streaming.chunks, standard.chunks);
    }

    #[test]
    fn standard_chunk_plan_keeps_short_audio_single_chunk() {
        let sr = 16_000u32;
        let samples = vec![0.0f32; (sr as usize) * 4];
        let plan = NativeExecutor::asr_chunk_plan(&samples, sr, Some(30.0), false, false);
        assert_eq!(plan.chunks.len(), 1);
    }

    #[test]
    fn streaming_chunk_plan_relaxes_when_chunk_count_is_too_high() {
        let sr = 16_000u32;
        let samples = vec![0.0f32; (sr as usize) * 180];
        let plan = NativeExecutor::asr_chunk_plan(&samples, sr, Some(30.0), true, false);
        assert!(
            plan.chunks.len() <= 24,
            "expected guarded chunk count, got {}",
            plan.chunks.len()
        );
    }

    #[test]
    fn speech_chunk_plan_is_default_for_allowed_models() {
        let sr = 16_000u32;
        let samples = vec![0.0f32; (sr as usize) * 4];
        let plan = NativeExecutor::asr_chunk_plan(&samples, sr, Some(30.0), false, true);
        assert_eq!(plan.planner, AsrChunkPlannerKind::Speech);
        assert!(plan.requires_chunk_path());
        assert!(plan.chunks.is_empty());
        assert!(plan.speech_plan.as_ref().expect("speech plan").no_speech);
    }

    #[test]
    fn vad_chunk_plan_can_return_no_speech() {
        let sr = 16_000u32;
        let samples = vec![0.0f32; (sr as usize) * 4];
        let plan =
            NativeExecutor::asr_chunk_plan_with_options(&samples, sr, Some(30.0), false, true);
        assert_eq!(plan.planner, AsrChunkPlannerKind::Speech);
        assert!(plan.requires_chunk_path());
        assert!(plan.chunks.is_empty());
        assert!(plan.speech_plan.as_ref().expect("speech plan").no_speech);
    }

    #[test]
    fn chunk_plan_diagnostics_include_timing_and_seconds() {
        let sr = 16_000u32;
        let samples = vec![0.0f32; (sr as usize) * 4];
        let plan =
            NativeExecutor::asr_chunk_plan_with_options(&samples, sr, Some(30.0), false, true);

        let diagnostics = plan.diagnostics();
        let chunking = diagnostics
            .get("chunking")
            .and_then(|value| value.as_object())
            .expect("chunking diagnostics");

        assert!(chunking
            .get("planning_ms")
            .and_then(|v| v.as_f64())
            .is_some());
        let config = chunking
            .get("config")
            .and_then(|value| value.as_object())
            .expect("chunk config diagnostics");
        assert_eq!(
            config
                .get("min_context_replay_words")
                .and_then(|value| value.as_u64()),
            Some(8)
        );
        let speech = chunking
            .get("speech")
            .and_then(|value| value.as_object())
            .expect("speech diagnostics");
        assert_eq!(
            speech.get("skipped_seconds").and_then(|v| v.as_f64()),
            Some(4.0)
        );
    }

    #[test]
    fn empty_chunk_plan_streams_final_marker() {
        let sr = 16_000u32;
        let samples = Vec::<f32>::new();
        let chunk_plan = Vec::<AudioChunk>::new();
        let (tx, mut rx) = mpsc::channel(8);
        let mut sequence = 0usize;
        let merged = NativeExecutor::transcribe_with_chunk_plan(
            "req-empty",
            Some(&tx),
            EngineStreamPolicy::FailOnFull,
            &mut sequence,
            &samples,
            sr,
            &chunk_plan,
            &NativeExecutor::asr_long_form_config(),
            |_chunk_audio, _sr| Ok("unreachable".to_string()),
        )
        .expect("empty no-speech plan should complete");

        assert!(merged.is_empty());
        let event = rx.try_recv().expect("final marker");
        assert!(event.is_final);
    }

    #[test]
    fn chunk_plan_details_collects_per_chunk_diagnostics() {
        let sr = 10u32;
        let samples = vec![0.0f32; sr as usize * 2];
        let chunk_plan = vec![
            AudioChunk {
                start_sample: 0,
                end_sample: sr as usize,
            },
            AudioChunk {
                start_sample: sr as usize,
                end_sample: sr as usize * 2,
            },
        ];

        let mut chunk_idx = 0usize;
        let mut sequence = 0usize;
        let merged = NativeExecutor::transcribe_with_chunk_plan_with_details(
            "req-details",
            None,
            EngineStreamPolicy::FailOnFull,
            &mut sequence,
            &samples,
            sr,
            &chunk_plan,
            &NativeExecutor::asr_long_form_config(),
            |_chunk_audio, _sr| {
                let idx = chunk_idx;
                chunk_idx = chunk_idx.saturating_add(1);
                Ok(AsrChunkTranscription {
                    text: format!("chunk-{idx}"),
                    diagnostics: Some(serde_json::json!({ "decoder_chunk": idx })),
                })
            },
        )
        .expect("chunk plan should complete");

        assert!(merged.text.contains("chunk-0"));
        assert!(merged.text.contains("chunk-1"));
        assert_eq!(merged.chunk_diagnostics.len(), 2);
        assert_eq!(
            merged.chunk_diagnostics[0]
                .get("index")
                .and_then(|value| value.as_u64()),
            Some(0)
        );
        assert_eq!(
            merged.chunk_diagnostics[0]
                .get("start_seconds")
                .and_then(|value| value.as_f64()),
            Some(0.0)
        );
        assert_eq!(
            merged.chunk_diagnostics[0]
                .get("end_seconds")
                .and_then(|value| value.as_f64()),
            Some(1.0)
        );
        assert_eq!(
            merged.chunk_diagnostics[0]
                .get("model_diagnostics")
                .and_then(|value| value.get("decoder_chunk"))
                .and_then(|value| value.as_u64()),
            Some(0)
        );
        assert!(merged.chunk_diagnostics[0]
            .get("transcribe_ms")
            .and_then(|value| value.as_f64())
            .is_some());
    }

    #[test]
    fn chunk_plan_diagnostics_can_embed_transcription_details() {
        let sr = 10u32;
        let samples = vec![0.0f32; sr as usize * 2];
        let plan = NativeExecutor::asr_chunk_plan(&samples, sr, Some(1.0), false, false);

        let diagnostics = plan.diagnostics_with_chunk_transcriptions(vec![
            serde_json::json!({ "index": 0, "text_chars": 5 }),
        ]);
        let chunk_transcriptions = diagnostics
            .get("chunking")
            .and_then(|value| value.get("chunk_transcriptions"))
            .and_then(|value| value.as_array())
            .expect("chunk transcription diagnostics");

        assert_eq!(chunk_transcriptions.len(), 1);
        assert_eq!(
            chunk_transcriptions[0]
                .get("text_chars")
                .and_then(|value| value.as_u64()),
            Some(5)
        );
    }

    #[test]
    fn chunk_streaming_defers_boundary_noise_until_next_delta() {
        let sr = 16_000u32;
        let samples = vec![0.0f32; sr as usize * 3];
        let chunk_plan = vec![
            AudioChunk {
                start_sample: 0,
                end_sample: sr as usize,
            },
            AudioChunk {
                start_sample: sr as usize,
                end_sample: sr as usize * 2,
            },
            AudioChunk {
                start_sample: sr as usize * 2,
                end_sample: sr as usize * 3,
            },
        ];

        let (tx, mut rx) = mpsc::channel(128);
        let mut sequence = 0usize;
        let mut chunk_idx = 0usize;
        let merged = NativeExecutor::transcribe_with_chunk_plan(
            "req-1",
            Some(&tx),
            EngineStreamPolicy::FailOnFull,
            &mut sequence,
            &samples,
            sr,
            &chunk_plan,
            &NativeExecutor::asr_long_form_config(),
            |_chunk_audio, _sr| {
                let out = match chunk_idx {
                    0 => "...".to_string(),
                    1 => "hello".to_string(),
                    _ => "world".to_string(),
                };
                chunk_idx = chunk_idx.saturating_add(1);
                Ok(out)
            },
        )
        .expect("chunk plan should succeed");

        assert!(merged.contains("hello"));
        assert!(merged.contains("world"));

        let mut streamed = String::new();
        let mut saw_final = false;
        while let Ok(event) = rx.try_recv() {
            if event.is_final {
                saw_final = true;
                continue;
            }
            if let Some(delta) = event.text {
                streamed.push_str(&delta);
            }
        }
        assert!(saw_final, "expected final marker");
        assert!(streamed.contains("hello"));
        assert!(streamed.contains("world"));
    }

    #[test]
    fn chunk_streaming_batches_large_text_delta_to_avoid_queue_pressure() {
        let sr = 16_000u32;
        let samples = vec![0.0f32; sr as usize];
        let chunk_plan = vec![AudioChunk {
            start_sample: 0,
            end_sample: samples.len(),
        }];
        let long_delta = "word ".repeat(1024);
        let (tx, mut rx) = mpsc::channel(2);
        let mut sequence = 0usize;

        let merged = NativeExecutor::transcribe_with_chunk_plan(
            "req-large-delta",
            Some(&tx),
            EngineStreamPolicy::FailOnFull,
            &mut sequence,
            &samples,
            sr,
            &chunk_plan,
            &NativeExecutor::asr_long_form_config(),
            |_chunk_audio, _sr| Ok(long_delta.clone()),
        )
        .expect("large chunk delta should fit as one streaming event");

        assert_eq!(merged, long_delta.trim());
        assert_eq!(sequence, 2);

        let text_event = rx.try_recv().expect("text delta event");
        assert_eq!(text_event.sequence, 0);
        assert_eq!(text_event.text.as_deref(), Some(long_delta.trim()));

        let final_event = rx.try_recv().expect("final marker");
        assert_eq!(final_event.sequence, 1);
        assert!(final_event.is_final);
    }
}
