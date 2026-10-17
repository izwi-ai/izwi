use izwi_asr_toolkit::{
    chunk_seam_secs, plan_audio_chunks, plan_speech_audio_chunks, AsrLongFormConfig,
    AsrSpeechChunkConfig, AudioChunk, SettledChunkOverlap, SpeechChunkPlan, TimedWord,
    TranscriptAssembler,
};
use serde_json::json;
use std::time::Instant;
use tracing::debug;

use crate::catalog::ModelFamily;
use crate::engine::EngineCoreRequest;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::registry::NativeAsrModel;
use crate::runtime::audio_io::{base64_decode, decode_audio_bytes_canonical};

use super::super::output::{AsrProgress, AsrProgressPhase};
use super::super::request::{EngineStreamPolicy, StreamStagingBuffer};
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

#[cfg(test)]
impl PlannedAsrChunks {
    /// A long-form plan for routing tests that never execute it.
    pub(crate) fn unexecuted_for_tests() -> Self {
        Self {
            config: AsrLongFormConfig::default(),
            chunks: Vec::new(),
            planner: AsrChunkPlannerKind::Duration,
            speech_plan: None,
            input_samples: 0,
            sample_rate: 0,
            planning_ms: 0.0,
        }
    }
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
    /// The transcript's word timings, when every chunk was merged by time.
    pub(super) timestamp_words: Option<Vec<TimedWord>>,
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

/// What the ASR route planner knows about a loaded model. The route plan is a
/// function of these, the audio and whether the request streams.
#[derive(Debug, Clone)]
pub(crate) struct AsrRouteCapabilities {
    /// Longest audio one invocation may cover: the model window composed with
    /// the effective context and decode budget. `None` means unbounded.
    pub(crate) invocation_limit_secs: Option<f32>,
    /// One pass covers the whole model window rather than one long-form
    /// chunk.
    pub(crate) single_pass_spans_model_window: bool,
    /// Every chunk costs the encoder's full fixed window.
    pub(crate) fixed_encoder_window: bool,
    /// Long audio is chunked on detected speech islands.
    pub(crate) prefers_vad_planner: bool,
    /// The chunk route streams this model's tokens while a chunk decodes.
    pub(crate) streams_within_chunk: bool,
    /// Chunking for audio that needs more than one pass.
    pub(crate) chunk_config: AsrLongFormConfig,
}

impl AsrRouteCapabilities {
    pub(crate) fn for_family(family: ModelFamily, invocation_limit_secs: Option<f32>) -> Self {
        let fixed_encoder_window = family.asr_fixed_encoder_window();
        Self {
            invocation_limit_secs,
            single_pass_spans_model_window: family.asr_single_pass_spans_model_window(),
            fixed_encoder_window,
            // A fixed-window encoder pads a mostly silent chunk with a full
            // window of silence, which a decoder that hallucinates on
            // non-speech turns into text. Speech islands avoid such chunks.
            prefers_vad_planner: fixed_encoder_window && family.hallucinates_on_non_speech(),
            streams_within_chunk: family.asr_streams_within_chunk(),
            chunk_config: NativeExecutor::asr_long_form_config(),
        }
    }

    /// Capabilities of a loaded `NativeAsrModel` for one request: its
    /// invocation window fitted to the effective context with the decode
    /// budget the model plans for that much audio.
    pub(crate) fn for_native_model(
        model: &NativeAsrModel,
        variant: ModelVariant,
        request: &EngineCoreRequest,
        effective_context: Option<usize>,
    ) -> Result<Self> {
        let window = model.asr_invocation_window(variant, request.asr_native_word_timestamps);
        let limit = crate::runtime::compose_invocation_audio_limit(
            window.max_audio_seconds,
            model.audio_token_rate(),
            effective_context,
            |secs| super::handler_asr::asr_planned_decode_tokens(request, variant.family(), secs),
        )?;
        let limit = match window.max_chunk_seconds {
            Some(cap) => Some(limit.map_or(cap, |limit| limit.min(cap))),
            None => limit,
        };
        Ok(Self::for_family(variant.family(), limit))
    }

    pub(crate) fn with_chunk_config(mut self, chunk_config: AsrLongFormConfig) -> Self {
        self.chunk_config = chunk_config;
        self
    }

    /// Audio one invocation fits, with a 5% margin.
    fn fit_limit_secs(&self) -> Option<f32> {
        self.invocation_limit_secs
            .filter(|v| v.is_finite() && *v > 0.0)
            .map(|v| v * 0.95)
    }

    /// Cap the chunk planner applies on top of `config`'s own chunk length.
    fn tuned_chunk_limit_secs(&self, config: &AsrLongFormConfig) -> Option<f32> {
        self.fit_limit_secs()
            .map(|v| v.max(config.min_chunk_secs.max(1.0)))
    }

    /// Longest audio this model transcribes in one pass, streaming or not;
    /// `None` when any length runs in one pass.
    pub(crate) fn single_pass_limit_secs(&self) -> Option<f32> {
        if self.single_pass_spans_model_window {
            return self.fit_limit_secs();
        }
        // The chunk planner's own single-chunk bound.
        let floor = self.chunk_config.min_chunk_secs.max(0.1);
        let chunk = self.chunk_config.hard_max_chunk_secs.max(floor);
        Some(match self.tuned_chunk_limit_secs(&self.chunk_config) {
            Some(limit) => chunk.min(limit.max(floor)),
            None => chunk,
        })
    }

    pub(crate) fn supports_single_pass_up_to(&self, audio_secs: f32) -> bool {
        self.single_pass_limit_secs()
            .is_none_or(|limit| audio_secs <= limit)
    }

    /// Short latency chunks replace the accuracy plan for a streaming request
    /// only when the model would otherwise emit text a whole chunk at a time,
    /// and only when a short chunk is actually cheaper to decode.
    fn latency_chunks_when_streaming(&self) -> bool {
        !self.streams_within_chunk && !self.fixed_encoder_window
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

    pub(super) fn env_usize(key: &str) -> Option<usize> {
        std::env::var(key)
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
    }

    /// Holdback options for streaming ASR chunk transcription. The env keys
    /// keep their historical QWEN prefix for compatibility but apply to every
    /// family's streaming chunk path.
    pub(super) fn asr_chunk_stream_options() -> AsrChunkStreamOptions {
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

    /// The ASR route plan: how a request's audio is segmented, decided once
    /// from the model's capabilities, the audio and whether the request
    /// streams. Preparation stores it on the request and the executor runs it
    /// as stored.
    ///
    /// - Audio that fits one pass runs single pass in both modes. One pass is
    ///   the whole model window for models built to cover it, otherwise one
    ///   long-form chunk, and never more than 95% of the invocation limit.
    /// - Longer audio is chunked the same way whether or not the request
    ///   streams: the chunk route streams each chunk's tokens as they decode.
    ///   The one exception is a streaming request for a model that cannot
    ///   stream within a chunk and has no fixed encoder window; it gets short
    ///   latency chunks, since its text otherwise arrives a full chunk at a
    ///   time.
    /// - A model that prefers the VAD planner chunks long audio on speech
    ///   islands, in both modes.
    pub(crate) fn asr_route_plan(
        samples: &[f32],
        sample_rate: u32,
        capabilities: &AsrRouteCapabilities,
        streaming: bool,
    ) -> PlannedAsrChunks {
        let planning_started = Instant::now();
        let audio_secs = samples_to_seconds(samples.len(), sample_rate);
        let finish = |config: AsrLongFormConfig,
                      chunks: Vec<AudioChunk>,
                      planner: AsrChunkPlannerKind,
                      speech_plan: Option<SpeechChunkPlan>| PlannedAsrChunks {
            config,
            chunks,
            planner,
            speech_plan,
            input_samples: samples.len(),
            sample_rate,
            planning_ms: planning_started.elapsed().as_secs_f64() * 1000.0,
        };
        let cfg = capabilities.chunk_config.clone();
        if capabilities.supports_single_pass_up_to(audio_secs) {
            let chunks = if samples.is_empty() || sample_rate == 0 {
                Vec::new()
            } else {
                vec![AudioChunk {
                    start_sample: 0,
                    end_sample: samples.len(),
                }]
            };
            return finish(cfg, chunks, AsrChunkPlannerKind::Duration, None);
        }

        if streaming && capabilities.latency_chunks_when_streaming() {
            let latency_cfg = Self::asr_streaming_low_latency_config(audio_secs);
            let chunks = plan_audio_chunks(
                samples,
                sample_rate,
                &latency_cfg,
                capabilities.tuned_chunk_limit_secs(&latency_cfg),
            );
            if chunks.len() <= MAX_STREAMING_LOW_LATENCY_CHUNKS {
                return finish(latency_cfg, chunks, AsrChunkPlannerKind::Duration, None);
            }
            debug!(
                "ASR streaming chunk plan too fragmented ({} chunks), relaxing to long-form defaults",
                chunks.len()
            );
        }

        let tuned_limit = capabilities.tuned_chunk_limit_secs(&cfg);
        if capabilities.prefers_vad_planner {
            let speech_cfg = AsrSpeechChunkConfig::default();
            let speech_plan =
                plan_speech_audio_chunks(samples, sample_rate, &cfg, &speech_cfg, tuned_limit);
            let chunks = speech_plan.chunks.clone();
            return finish(cfg, chunks, AsrChunkPlannerKind::Speech, Some(speech_plan));
        }
        let chunks = plan_audio_chunks(samples, sample_rate, &cfg, tuned_limit);
        finish(cfg, chunks, AsrChunkPlannerKind::Duration, None)
    }

    #[cfg(test)]
    pub(super) fn transcribe_with_chunk_plan<F>(
        request_id: &str,
        stream_tx: Option<&StreamStagingBuffer>,
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
            &|| false,
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

    #[cfg(test)]
    pub(super) fn transcribe_with_chunk_plan_with_details<F>(
        request_id: &str,
        is_cancelled: &dyn Fn() -> bool,
        stream_tx: Option<&StreamStagingBuffer>,
        stream_policy: EngineStreamPolicy,
        sequence: &mut usize,
        samples: &[f32],
        sample_rate: u32,
        chunk_plan: &[AudioChunk],
        chunk_cfg: &AsrLongFormConfig,
        mut transcribe_chunk: F,
    ) -> Result<ChunkedAsrTranscription>
    where
        F: FnMut(&[f32], u32) -> Result<AsrChunkTranscription>,
    {
        Self::transcribe_with_chunk_plan_with_details_and_options(
            request_id,
            is_cancelled,
            stream_tx,
            stream_policy,
            sequence,
            samples,
            sample_rate,
            chunk_plan,
            chunk_cfg,
            AsrChunkStreamOptions::default(),
            |chunk_audio, sr, _prefix_text| transcribe_chunk(chunk_audio, sr),
        )
    }

    #[cfg(test)]
    pub(super) fn transcribe_with_chunk_plan_with_context_and_details<F>(
        request_id: &str,
        stream_tx: Option<&StreamStagingBuffer>,
        stream_policy: EngineStreamPolicy,
        sequence: &mut usize,
        samples: &[f32],
        sample_rate: u32,
        chunk_plan: &[AudioChunk],
        chunk_cfg: &AsrLongFormConfig,
        transcribe_chunk: F,
    ) -> Result<ChunkedAsrTranscription>
    where
        F: FnMut(&[f32], u32, &str) -> Result<AsrChunkTranscription>,
    {
        Self::transcribe_with_chunk_plan_with_details_and_options(
            request_id,
            &|| false,
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

    #[cfg(test)]
    pub(super) fn transcribe_with_chunk_plan_with_details_and_options<F>(
        request_id: &str,
        is_cancelled: &dyn Fn() -> bool,
        stream_tx: Option<&StreamStagingBuffer>,
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
        F: FnMut(&[f32], u32, &str) -> Result<AsrChunkTranscription>,
    {
        Self::transcribe_with_chunk_plan_with_streaming_details_and_options(
            request_id,
            is_cancelled,
            stream_tx,
            stream_policy,
            sequence,
            samples,
            sample_rate,
            chunk_plan,
            chunk_cfg,
            stream_options,
            |chunk_audio, sr, prefix_text, _partial| transcribe_chunk(chunk_audio, sr, prefix_text),
        )
    }

    /// Chunked transcription with intra-chunk streaming support: every chunk
    /// of the plan in one call (see [`ChunkedAsrRun`] for the streaming and
    /// cancellation rules of each chunk).
    pub(super) fn transcribe_with_chunk_plan_with_streaming_details_and_options<F>(
        request_id: &str,
        is_cancelled: &dyn Fn() -> bool,
        stream_tx: Option<&StreamStagingBuffer>,
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
        F: FnMut(&[f32], u32, &str, Option<&mut dyn FnMut(&str)>) -> Result<AsrChunkTranscription>,
    {
        if chunk_plan.is_empty() {
            if let Some(tx) = stream_tx {
                Self::stream_final_marker_with_policy(tx, stream_policy, request_id, sequence)?;
            }
            return Ok(ChunkedAsrTranscription {
                text: String::new(),
                chunk_diagnostics: Vec::new(),
                timestamp_words: None,
            });
        }
        let stream = ChunkStream {
            request_id,
            tx: stream_tx,
            policy: stream_policy,
        };
        let mut run = ChunkedAsrRun::start(
            &stream,
            sequence,
            samples.len(),
            sample_rate,
            chunk_plan,
            chunk_cfg,
        )?;
        while !run.is_complete(chunk_plan) {
            run.run_chunk(
                &stream,
                sequence,
                is_cancelled,
                samples,
                sample_rate,
                chunk_plan,
                stream_options,
                &mut transcribe_chunk,
            )?;
        }
        run.finish(&stream, sequence, chunk_plan, sample_rate)
    }

    fn chunk_transcription_diagnostics(
        index: usize,
        chunk: &AudioChunk,
        sample_rate: u32,
        transcribe_ms: f64,
        prefix_text: &str,
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
            "context_prefix_chars": prefix_text.chars().count(),
            "context_prefix_words": prefix_text.split_whitespace().count(),
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

    fn asr_processing_progress(chunk_plan: &[AudioChunk], sample_rate: u32) -> AsrProgress {
        let total_audio_secs = chunk_plan
            .last()
            .map(|chunk| f64::from(samples_to_seconds(chunk.end_sample, sample_rate)));
        AsrProgress {
            phase: AsrProgressPhase::Processing,
            current_chunk: None,
            total_chunks: Some(chunk_plan.len()),
            processed_audio_secs: Some(0.0),
            total_audio_secs,
            percent: Some(0.0),
        }
    }

    fn asr_complete_progress(chunk_plan: &[AudioChunk], sample_rate: u32) -> AsrProgress {
        let total_audio_secs = chunk_plan
            .last()
            .map(|chunk| f64::from(samples_to_seconds(chunk.end_sample, sample_rate)));
        AsrProgress {
            phase: AsrProgressPhase::Complete,
            current_chunk: Some(chunk_plan.len()),
            total_chunks: Some(chunk_plan.len()),
            processed_audio_secs: total_audio_secs,
            total_audio_secs,
            percent: Some(100.0),
        }
    }

    fn asr_chunk_progress(
        phase: AsrProgressPhase,
        index: usize,
        chunk: &AudioChunk,
        chunk_plan: &[AudioChunk],
        sample_rate: u32,
    ) -> AsrProgress {
        let total_audio_secs = chunk_plan
            .last()
            .map(|last| f64::from(samples_to_seconds(last.end_sample, sample_rate)));
        let processed_audio_secs = match phase {
            AsrProgressPhase::ChunkStarted => {
                f64::from(samples_to_seconds(chunk.start_sample, sample_rate))
            }
            AsrProgressPhase::ChunkFinished => {
                f64::from(samples_to_seconds(chunk.end_sample, sample_rate))
            }
            AsrProgressPhase::Processing => 0.0,
            AsrProgressPhase::Aligning => total_audio_secs.unwrap_or_default(),
            AsrProgressPhase::Complete => total_audio_secs.unwrap_or_default(),
        };
        let percent = total_audio_secs
            .filter(|total| *total > 0.0)
            .map(|total| ((processed_audio_secs / total) * 100.0).clamp(0.0, 100.0));

        AsrProgress {
            phase,
            current_chunk: Some(index + 1),
            total_chunks: Some(chunk_plan.len()),
            processed_audio_secs: Some(processed_audio_secs),
            total_audio_secs,
            percent,
        }
    }

    #[cfg(test)]
    pub(super) fn next_audio_delta(all_samples: &[f32], emitted_samples: &mut usize) -> Vec<f32> {
        let start = (*emitted_samples).min(all_samples.len());
        let delta = all_samples[start..].to_vec();
        *emitted_samples = all_samples.len();
        delta
    }

    #[cfg(test)]
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

/// Where a chunk run streams: the request and its staged stream, if any.
pub(super) struct ChunkStream<'a> {
    pub(super) request_id: &'a str,
    pub(super) tx: Option<&'a StreamStagingBuffer>,
    pub(super) policy: EngineStreamPolicy,
}

/// A long-form chunk plan transcribed one chunk at a time. Between chunks it
/// is host state only (the transcript assembled so far, the stream cursor,
/// per-chunk diagnostics), so the chunks of one file can run as separate
/// dispatches with other work on the device in between.
///
/// `is_cancelled` is polled before every chunk, inside the partial emitter
/// and before a finished chunk commits; once it reports true the run stops
/// streaming and returns `Error::Cancelled`. A model call already running is
/// not preempted, so an abandoned long-form request releases its slot within
/// one chunk instead of after the whole file.
#[derive(Debug, Clone)]
pub(super) struct ChunkedAsrRun {
    assembler: TranscriptAssembler,
    cursor: TranscriptStreamCursor,
    previous_chunk_end: Option<usize>,
    chunk_diagnostics: Vec<serde_json::Value>,
    /// Word timings of the transcript so far, while every committed chunk
    /// was merged by time.
    timestamp_words: Option<Vec<TimedWord>>,
    timed_chunks: usize,
    next_chunk: usize,
}

impl ChunkedAsrRun {
    /// Start a run of a non-empty `chunk_plan`, streaming its processing
    /// progress.
    pub(super) fn start(
        stream: &ChunkStream<'_>,
        sequence: &mut usize,
        input_samples: usize,
        sample_rate: u32,
        chunk_plan: &[AudioChunk],
        chunk_cfg: &AsrLongFormConfig,
    ) -> Result<Self> {
        debug!(
            "ASR long-form chunking enabled for request {}: {} chunks (~{:.1}s audio)",
            stream.request_id,
            chunk_plan.len(),
            input_samples as f32 / sample_rate.max(1) as f32
        );
        if let Some(tx) = stream.tx {
            NativeExecutor::stream_asr_progress_with_policy(
                tx,
                stream.policy,
                stream.request_id,
                sequence,
                NativeExecutor::asr_processing_progress(chunk_plan, sample_rate),
            )?;
        }
        Ok(Self {
            assembler: TranscriptAssembler::new(chunk_cfg.clone()),
            cursor: TranscriptStreamCursor::default(),
            previous_chunk_end: None,
            chunk_diagnostics: Vec::new(),
            timestamp_words: Some(Vec::new()),
            timed_chunks: 0,
            next_chunk: 0,
        })
    }

    pub(super) fn is_complete(&self, chunk_plan: &[AudioChunk]) -> bool {
        self.next_chunk >= chunk_plan.len()
    }

    /// Transcribe the next chunk. When a chunk transcription wants to stream
    /// provisional text before it completes, it pushes deltas through the
    /// `partial` emitter; the run dedups them against the committed
    /// transcript (previewing the assembler's boundary rules) and reconciles
    /// the accounting once the chunk text commits. Families that ignore the
    /// emitter keep the plain per-chunk delta behavior.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_chunk<F>(
        &mut self,
        stream: &ChunkStream<'_>,
        sequence: &mut usize,
        is_cancelled: &dyn Fn() -> bool,
        samples: &[f32],
        sample_rate: u32,
        chunk_plan: &[AudioChunk],
        stream_options: AsrChunkStreamOptions,
        transcribe_chunk: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&[f32], u32, &str, Option<&mut dyn FnMut(&str)>) -> Result<AsrChunkTranscription>,
    {
        let (request_id, stream_tx, stream_policy) = (stream.request_id, stream.tx, stream.policy);
        let holdback_chars = stream_options.stable_text_holdback_chars;
        let stable_text_streaming = stream_tx.is_some() && holdback_chars > 0;
        let cancelled = || Error::Cancelled(request_id.to_string());
        let idx = self.next_chunk;
        let chunk = &chunk_plan[idx];
        self.next_chunk += 1;
        if is_cancelled() {
            return Err(cancelled());
        }
        if chunk.end_sample <= chunk.start_sample || chunk.end_sample > samples.len() {
            self.chunk_diagnostics
                .push(NativeExecutor::chunk_transcription_diagnostics(
                    idx,
                    chunk,
                    sample_rate,
                    0.0,
                    "",
                    "",
                    Some(json!({
                        "skipped": true,
                        "skip_reason": "invalid_bounds",
                    })),
                ));
            return Ok(());
        }
        let chunk_audio = &samples[chunk.start_sample..chunk.end_sample];
        // Audio this chunk shares with the previous one. Chunks that do not
        // overlap (speech-island plans) are merged without boundary dedup.
        let overlap_secs = self.previous_chunk_end.map_or(0.0, |end| {
            samples_to_seconds(end.saturating_sub(chunk.start_sample), sample_rate)
        });
        // Seams with the neighbouring chunks, for chunks merged by time.
        let chunk_start_secs = samples_to_seconds(chunk.start_sample, sample_rate);
        let chunk_end_secs = samples_to_seconds(chunk.end_sample, sample_rate);
        let seam_after = self
            .previous_chunk_end
            .map(|end| chunk_seam_secs(samples_to_seconds(end, sample_rate), chunk_start_secs));
        let seam_until = chunk_plan.get(idx + 1).map(|next| {
            chunk_seam_secs(
                chunk_end_secs,
                samples_to_seconds(next.start_sample, sample_rate),
            )
        });
        self.previous_chunk_end = Some(chunk.end_sample);
        let prefix_text = self.assembler.text().to_string();
        if let Some(tx) = stream_tx {
            NativeExecutor::stream_asr_progress_with_policy(
                tx,
                stream_policy,
                request_id,
                sequence,
                NativeExecutor::asr_chunk_progress(
                    AsrProgressPhase::ChunkStarted,
                    idx,
                    chunk,
                    chunk_plan,
                    sample_rate,
                ),
            )?;
        }
        let chunk_started = Instant::now();
        let mut partial_text = String::new();
        let mut settled: Option<SettledChunkOverlap> = None;
        let mut partial_error: Option<Error> = None;
        let mut partial_streamed = false;
        let chunk_result = {
            let assembler = &self.assembler;
            let cursor = &mut self.cursor;
            let mut emit_partial = |delta: &str| {
                partial_streamed = true;
                if partial_error.is_some() || delta.is_empty() {
                    return;
                }
                if is_cancelled() {
                    partial_error = Some(cancelled());
                    return;
                }
                partial_text.push_str(delta);
                // Stream nothing from this chunk until its text covers the
                // overlap with the previous chunk; after that its deltas
                // only append, and the commit reuses the same decision.
                if settled.is_none() {
                    settled = assembler.settle_chunk_overlap(&partial_text, overlap_secs);
                }
                let Some(overlap) = settled.as_ref() else {
                    return;
                };
                let provisional = assembler.provisional_chunk_text(&partial_text, overlap);
                let stable_delta =
                    cursor.advance(assembler.text(), &provisional, holdback_chars, false);
                if stable_delta.is_empty() {
                    return;
                }
                if let Some(tx) = stream_tx {
                    if let Err(err) = NativeExecutor::stream_text_with_policy(
                        tx,
                        stream_policy,
                        request_id,
                        sequence,
                        stable_delta,
                    ) {
                        partial_error = Some(err);
                    }
                }
            };
            let partial: Option<&mut dyn FnMut(&str)> = if stream_tx.is_some() {
                Some(&mut emit_partial as &mut dyn FnMut(&str))
            } else {
                None
            };
            transcribe_chunk(chunk_audio, sample_rate, prefix_text.as_str(), partial)?
        };
        if let Some(err) = partial_error {
            return Err(err);
        }
        if is_cancelled() {
            return Err(cancelled());
        }
        let chunk_text = chunk_result.text.clone();
        let chunk_words = chunk_result
            .diagnostics
            .as_ref()
            .and_then(|diagnostics| chunk_timed_words(diagnostics, &chunk_text, chunk_start_secs));
        let transcribe_ms = chunk_started.elapsed().as_secs_f64() * 1000.0;
        self.chunk_diagnostics
            .push(NativeExecutor::chunk_transcription_diagnostics(
                idx,
                chunk,
                sample_rate,
                transcribe_ms,
                &prefix_text,
                &chunk_text,
                chunk_result.diagnostics,
            ));
        if let Some(tx) = stream_tx {
            NativeExecutor::stream_asr_progress_with_policy(
                tx,
                stream_policy,
                request_id,
                sequence,
                NativeExecutor::asr_chunk_progress(
                    AsrProgressPhase::ChunkFinished,
                    idx,
                    chunk,
                    chunk_plan,
                    sample_rate,
                ),
            )?;
        }
        let committed_len = self.assembler.text().len();
        let provisional = settled.as_ref().map(|overlap| {
            self.assembler
                .provisional_chunk_text(&partial_text, overlap)
        });
        // A chunk with word timings is merged by time while every chunk
        // before it was; otherwise by text, which ends timing for the rest
        // of the transcript.
        match (chunk_words, self.timestamp_words.as_mut()) {
            (Some(words), Some(merged_words)) => {
                merged_words.extend(
                    self.assembler
                        .push_timed_chunk_words(&words, seam_after, seam_until),
                );
                self.timed_chunks += 1;
            }
            _ => {
                self.timestamp_words = None;
                match settled.as_ref() {
                    Some(overlap) => {
                        self.assembler.push_settled_chunk_text(&chunk_text, overlap);
                    }
                    None => {
                        self.assembler
                            .push_chunk_text_with_overlap(&chunk_text, overlap_secs);
                    }
                }
            }
        }
        if provisional.is_some_and(|provisional| {
            !self.assembler.text()[committed_len..].starts_with(&provisional)
        }) {
            debug!(
                "ASR chunk {idx} of request {request_id} committed text that differs from its streamed partials"
            );
        }
        // What this chunk added after overlap dedup. The chunk texts
        // concatenate to the transcript, so with the chunk's time range
        // they form the response's default segments.
        if let Some(entry) = self
            .chunk_diagnostics
            .last_mut()
            .and_then(|value| value.as_object_mut())
        {
            entry.insert(
                "committed_text".to_string(),
                json!(self.assembler.text()[committed_len..].trim_start()),
            );
        }
        self.cursor.align(self.assembler.text(), "");

        let Some(tx) = stream_tx else {
            return Ok(());
        };
        let is_last_chunk = idx + 1 == chunk_plan.len();
        // Without partials or a holdback, a delta that is only boundary
        // punctuation waits to be streamed with the next chunk's text.
        if !partial_streamed
            && !stable_text_streaming
            && !is_last_chunk
            && is_boundary_noise_delta(self.cursor.pending(self.assembler.text()))
        {
            return Ok(());
        }
        let stable_delta =
            self.cursor
                .advance(self.assembler.text(), "", holdback_chars, is_last_chunk);
        if !stable_delta.is_empty() {
            NativeExecutor::stream_text_with_policy(
                tx,
                stream_policy,
                request_id,
                sequence,
                stable_delta,
            )?;
        }
        Ok(())
    }

    /// Stream the rest of the transcript and the final marker, and return
    /// the transcript of a complete run.
    pub(super) fn finish(
        mut self,
        stream: &ChunkStream<'_>,
        sequence: &mut usize,
        chunk_plan: &[AudioChunk],
        sample_rate: u32,
    ) -> Result<ChunkedAsrTranscription> {
        if let Some(tx) = stream.tx {
            let final_delta = self.cursor.advance(self.assembler.text(), "", 0, true);
            if !final_delta.is_empty() {
                NativeExecutor::stream_text_with_policy(
                    tx,
                    stream.policy,
                    stream.request_id,
                    sequence,
                    final_delta,
                )?;
            }
            NativeExecutor::stream_asr_progress_with_policy(
                tx,
                stream.policy,
                stream.request_id,
                sequence,
                NativeExecutor::asr_complete_progress(chunk_plan, sample_rate),
            )?;
            NativeExecutor::stream_final_marker_with_policy(
                tx,
                stream.policy,
                stream.request_id,
                sequence,
            )?;
        }
        let timed_chunks = self.timed_chunks;
        Ok(ChunkedAsrTranscription {
            text: self.assembler.finish(),
            chunk_diagnostics: self.chunk_diagnostics,
            timestamp_words: self.timestamp_words.filter(|_| timed_chunks > 0),
        })
    }
}

/// Append-only cursor over streamed transcript text: the committed transcript
/// followed by the current chunk's provisional text. `emitted` is a byte
/// offset into that text. Committed text extends the provisional text it
/// replaces, so the offset stays valid across commits.
#[derive(Debug, Clone, Default)]
struct TranscriptStreamCursor {
    emitted: usize,
}

impl TranscriptStreamCursor {
    /// Text past the cursor, minus a tail of `holdback_chars` characters until
    /// `is_final`. Work is bounded by the delta and the holdback, not by the
    /// transcript length.
    fn advance(
        &mut self,
        committed: &str,
        provisional: &str,
        holdback_chars: usize,
        is_final: bool,
    ) -> String {
        self.align(committed, provisional);
        let end = if is_final {
            committed.len() + provisional.len()
        } else {
            holdback_start(committed, provisional, holdback_chars)
        };
        if end <= self.emitted {
            return String::new();
        }
        let split = committed.len();
        let mut delta = String::with_capacity(end - self.emitted);
        if self.emitted < split {
            delta.push_str(&committed[self.emitted..end.min(split)]);
        }
        if end > split {
            delta.push_str(&provisional[self.emitted.max(split) - split..end - split]);
        }
        self.emitted = end;
        delta
    }

    /// Committed text not yet streamed.
    fn pending<'a>(&self, committed: &'a str) -> &'a str {
        committed.get(self.emitted..).unwrap_or("")
    }

    /// Keep the cursor within `committed` followed by `provisional` and on a
    /// character boundary. It only moves when model text contradicted text
    /// that was already streamed.
    fn align(&mut self, committed: &str, provisional: &str) {
        let split = committed.len();
        let mut emitted = self.emitted.min(split + provisional.len());
        while !(if emitted <= split {
            committed.is_char_boundary(emitted)
        } else {
            provisional.is_char_boundary(emitted - split)
        }) {
            emitted += 1;
        }
        self.emitted = emitted;
    }
}

/// Byte offset, in `committed` followed by `provisional`, where the last
/// `holdback_chars` characters begin.
fn holdback_start(committed: &str, provisional: &str, holdback_chars: usize) -> usize {
    let total = committed.len() + provisional.len();
    if holdback_chars == 0 {
        return total;
    }
    let mut remaining = holdback_chars;
    for (idx, _) in provisional.char_indices().rev() {
        remaining -= 1;
        if remaining == 0 {
            return committed.len() + idx;
        }
    }
    for (idx, _) in committed.char_indices().rev() {
        remaining -= 1;
        if remaining == 0 {
            return idx;
        }
    }
    0
}

/// A chunk's word timings from its model diagnostics (`timestamp_words`,
/// timed from the chunk's start), placed on the input timeline. `None` when
/// the model reported no word timings for the chunk's text: a model may
/// report an empty list for a task that does not time words.
fn chunk_timed_words(
    diagnostics: &serde_json::Value,
    chunk_text: &str,
    chunk_start_secs: f32,
) -> Option<Vec<TimedWord>> {
    let words = diagnostics.get("timestamp_words")?.as_array()?;
    if words.is_empty() && !chunk_text.trim().is_empty() {
        return None;
    }
    words
        .iter()
        .map(|word| {
            Some(TimedWord {
                text: word.get("word")?.as_str()?.to_string(),
                start_secs: chunk_start_secs + word.get("start_time_seconds")?.as_f64()? as f32,
                end_secs: chunk_start_secs + word.get("end_time_seconds")?.as_f64()? as f32,
            })
        })
        .collect()
}

/// `timestamp_words` diagnostics for a transcript's word timings.
pub(super) fn timestamp_words_json(words: &[TimedWord]) -> serde_json::Value {
    serde_json::Value::Array(
        words
            .iter()
            .map(|word| {
                json!({
                    "word": word.text,
                    "start_time_seconds": word.start_secs,
                    "end_time_seconds": word.end_secs,
                })
            })
            .collect(),
    )
}

pub(super) fn decode_audio_base64_with_rate(audio_b64: &str) -> Result<(Vec<f32>, u32)> {
    let audio_bytes = base64_decode(audio_b64)?;
    decode_audio_bytes_canonical(&audio_bytes)
}

pub(super) fn decode_request_audio_with_rate(
    request: &EngineCoreRequest,
) -> Result<(Vec<f32>, u32)> {
    if let Some(audio_bytes) = request.audio_bytes_for_execution() {
        return decode_audio_bytes_canonical(audio_bytes);
    }

    let audio_b64 = request
        .audio_base64_for_execution()
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

    use crate::engine::output::AsrProgressPhase;
    use crate::engine::request::StreamStagingBuffer;
    use crate::engine::{EngineStreamPolicy, StreamingOutput};

    use super::{
        AsrChunkPlannerKind, AsrChunkStreamOptions, AsrChunkTranscription, AsrRouteCapabilities,
        ChunkStream, ChunkedAsrRun, NativeExecutor, PlannedAsrChunks, TranscriptStreamCursor,
    };
    use crate::catalog::ModelFamily;

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
    fn stream_cursor_holds_back_unstable_tail() {
        let mut cursor = TranscriptStreamCursor::default();

        assert_eq!(cursor.advance("hello world", "", 5, false), "hello ");
        assert_eq!(cursor.advance("hello world again", "", 5, false), "world ");
        assert_eq!(cursor.advance("hello world again", "", 5, true), "again");
        assert_eq!(cursor.advance("hello world again", "", 5, true), "");
    }

    #[test]
    fn stream_cursor_respects_utf8_boundaries() {
        let mut cursor = TranscriptStreamCursor::default();

        assert_eq!(cursor.advance("héllo 世界", "", 2, false), "héllo ");
        assert_eq!(cursor.advance("héllo 世界", "", 2, true), "世界");
    }

    #[test]
    fn stream_cursor_spans_committed_and_provisional_text() {
        let mut cursor = TranscriptStreamCursor::default();

        assert_eq!(
            cursor.advance("alpha", " beta gam", 3, false),
            "alpha beta "
        );
        assert_eq!(
            cursor.advance("alpha", " beta gamma delta", 3, false),
            "gamma de"
        );
        // The commit extends the provisional text, so the cursor carries over.
        assert_eq!(
            cursor.advance("alpha beta gamma delta epsilon", "", 0, true),
            "lta epsilon"
        );
    }

    fn caps(family: ModelFamily, limit: Option<f32>) -> AsrRouteCapabilities {
        AsrRouteCapabilities::for_family(family, limit)
    }

    fn plan(
        family: ModelFamily,
        limit: Option<f32>,
        samples: &[f32],
        sr: u32,
        streaming: bool,
    ) -> PlannedAsrChunks {
        NativeExecutor::asr_route_plan(samples, sr, &caps(family, limit), streaming)
    }

    /// Representative invocation limits: Qwen3 off CUDA (1500 encoder
    /// positions), Whisper's and Nemotron's 30 s window, Granite's 9 min ASR
    /// task, VibeVoice's hour; Parakeet, Voxtral and LFM2.5 set none.
    const ROUTE_FAMILIES: [(ModelFamily, Option<f32>); 8] = [
        (ModelFamily::Qwen3Asr, Some(120.0)),
        (ModelFamily::WhisperAsr, Some(30.0)),
        (ModelFamily::GraniteSpeechAsr, Some(540.0)),
        (ModelFamily::VibeVoiceAsr, Some(3600.0)),
        (ModelFamily::NemotronAsr, Some(30.0)),
        (ModelFamily::ParakeetAsr, None),
        (ModelFamily::Voxtral, None),
        (ModelFamily::Lfm25Audio, None),
    ];

    #[derive(Debug, PartialEq)]
    enum Route {
        SinglePass,
        /// Long-form chunks on the given planner, identical in both modes.
        Chunked(AsrChunkPlannerKind),
        /// Streaming only: short chunks for latency.
        LatencyChunks,
    }

    fn route_of(plan: &PlannedAsrChunks, accuracy: &PlannedAsrChunks) -> Route {
        if !plan.requires_chunk_path() {
            return Route::SinglePass;
        }
        if plan.chunks == accuracy.chunks
            && plan.config.target_chunk_secs == accuracy.config.target_chunk_secs
            && plan.planner == accuracy.planner
        {
            return Route::Chunked(plan.planner);
        }
        assert!(plan.config.target_chunk_secs <= super::DEFAULT_STREAM_LONG_TARGET_CHUNK_SECS);
        Route::LatencyChunks
    }

    /// The route decision table: family x duration x mode.
    #[test]
    fn route_plan_decision_table() {
        use AsrChunkPlannerKind::{Duration, Speech};
        let sr = 16_000u32;
        let expected = |family: ModelFamily, secs: usize, streaming: bool| -> Route {
            if secs <= 20 {
                return Route::SinglePass;
            }
            match family {
                ModelFamily::VibeVoiceAsr | ModelFamily::ParakeetAsr => Route::SinglePass,
                ModelFamily::WhisperAsr => Route::Chunked(Speech),
                // Granite cannot stream within a chunk; 600 s of latency
                // chunks is too fragmented and relaxes to the accuracy plan.
                ModelFamily::GraniteSpeechAsr if streaming && secs < 600 => Route::LatencyChunks,
                _ => Route::Chunked(Duration),
            }
        };
        for secs in [20usize, 60, 116, 600] {
            let samples = vec![0.0f32; sr as usize * secs];
            for (family, limit) in ROUTE_FAMILIES {
                let accuracy = plan(family, limit, &samples, sr, false);
                for streaming in [false, true] {
                    let planned = plan(family, limit, &samples, sr, streaming);
                    assert_eq!(
                        route_of(&planned, &accuracy),
                        expected(family, secs, streaming),
                        "{family:?} {secs}s streaming={streaming}"
                    );
                    if planned.requires_chunk_path() {
                        let fit = limit.map_or(f32::INFINITY, |limit| limit * 0.95);
                        for chunk in &planned.chunks {
                            let chunk_secs = chunk.len_samples() as f32 / sr as f32;
                            assert!(chunk_secs <= fit.min(30.0) + 1e-3, "{family:?} {secs}s");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn single_pass_limit_agrees_with_the_chunk_planner() {
        let sr = 1_000u32;
        for (family, limit) in ROUTE_FAMILIES {
            let capabilities = caps(family, limit);
            let Some(limit_secs) = capabilities.single_pass_limit_secs() else {
                assert!(capabilities.supports_single_pass_up_to(1e6));
                continue;
            };
            let at_limit = vec![0.0f32; (limit_secs * sr as f32).floor() as usize];
            let past_limit = vec![0.0f32; at_limit.len() + 2];
            assert!(
                !NativeExecutor::asr_route_plan(&at_limit, sr, &capabilities, false)
                    .requires_chunk_path(),
                "{family:?}"
            );
            assert!(
                NativeExecutor::asr_route_plan(&past_limit, sr, &capabilities, false)
                    .requires_chunk_path(),
                "{family:?}"
            );
            if !capabilities.single_pass_spans_model_window {
                let chunks = izwi_asr_toolkit::plan_audio_chunks(
                    &at_limit,
                    sr,
                    &capabilities.chunk_config,
                    capabilities.tuned_chunk_limit_secs(&capabilities.chunk_config),
                );
                assert_eq!(chunks.len(), 1, "{family:?}");
            }
        }
    }

    #[test]
    fn capability_flags_come_from_the_family() {
        let whisper = caps(ModelFamily::WhisperAsr, Some(30.0));
        assert!(whisper.fixed_encoder_window && whisper.prefers_vad_planner);
        let qwen = caps(ModelFamily::Qwen3Asr, Some(120.0));
        assert!(!qwen.fixed_encoder_window && !qwen.prefers_vad_planner);
        assert!(qwen.streams_within_chunk);
        assert!(!caps(ModelFamily::GraniteSpeechAsr, None).streams_within_chunk);
        assert!(caps(ModelFamily::VibeVoiceAsr, None).single_pass_spans_model_window);
    }

    #[test]
    fn whole_window_model_chunks_past_its_window() {
        let sr = 10u32;
        let samples = vec![0.0f32; sr as usize * 50 * 60];
        let fits = plan(ModelFamily::VibeVoiceAsr, Some(3600.0), &samples, sr, false);
        assert!(!fits.requires_chunk_path());
        let limited = plan(ModelFamily::VibeVoiceAsr, Some(120.0), &samples, sr, true);
        assert!(limited.requires_chunk_path());
        let max_chunk_secs = limited
            .chunks
            .iter()
            .map(|chunk| chunk.len_samples() as f32 / sr as f32)
            .fold(0.0f32, f32::max);
        assert!(max_chunk_secs <= 30.0 + 1.0 / sr as f32);
    }

    #[test]
    fn latency_plan_relaxes_when_chunk_count_is_too_high() {
        let sr = 16_000u32;
        let samples = vec![0.0f32; (sr as usize) * 180];
        let planned = plan(
            ModelFamily::GraniteSpeechAsr,
            Some(540.0),
            &samples,
            sr,
            true,
        );
        assert!(planned.chunks.len() <= 24, "got {}", planned.chunks.len());
    }

    #[test]
    fn vad_route_plan_can_return_no_speech() {
        let sr = 16_000u32;
        let samples = vec![0.0f32; (sr as usize) * 60];
        for streaming in [false, true] {
            let planned = plan(ModelFamily::WhisperAsr, Some(30.0), &samples, sr, streaming);
            assert_eq!(planned.planner, AsrChunkPlannerKind::Speech);
            assert!(planned.requires_chunk_path());
            assert!(planned.chunks.is_empty());
            assert!(planned.speech_plan.as_ref().expect("speech plan").no_speech);
        }
    }

    #[test]
    fn chunk_plan_diagnostics_include_timing_and_seconds() {
        let sr = 16_000u32;
        let samples = vec![0.0f32; (sr as usize) * 60];
        let planned = plan(ModelFamily::WhisperAsr, Some(30.0), &samples, sr, false);

        let diagnostics = planned.diagnostics();
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
            Some(60.0)
        );
    }

    #[test]
    fn empty_chunk_plan_streams_final_marker() {
        let sr = 16_000u32;
        let samples = Vec::<f32>::new();
        let chunk_plan = Vec::<AudioChunk>::new();
        let tx = StreamStagingBuffer::default();
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
        let event = tx
            .take()
            .expect("staged events")
            .into_iter()
            .next()
            .expect("final marker");
        assert!(event.is_final);
    }

    #[test]
    fn chunk_plan_streaming_emits_intra_chunk_partial_deltas() {
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
        let tx = StreamStagingBuffer::default();
        let mut sequence = 0usize;

        let mut chunk_idx = 0usize;
        let chunked = NativeExecutor::transcribe_with_chunk_plan_with_streaming_details_and_options(
            "req-partial",
            &|| false,
            Some(&tx),
            EngineStreamPolicy::FailOnFull,
            &mut sequence,
            &samples,
            sr,
            &chunk_plan,
            &NativeExecutor::asr_long_form_config(),
            AsrChunkStreamOptions {
                stable_text_holdback_chars: 0,
            },
            |_chunk_audio, _sr, _prefix, partial| {
                let idx = chunk_idx;
                chunk_idx += 1;
                if idx == 0 {
                    if let Some(emit) = partial {
                        emit("alpha ");
                        emit("beta");
                    }
                    return Ok(AsrChunkTranscription {
                        text: "alpha beta".to_string(),
                        diagnostics: None,
                    });
                }
                if let Some(emit) = partial {
                    emit("gamma ");
                }
                Ok(AsrChunkTranscription {
                    text: "gamma delta".to_string(),
                    diagnostics: None,
                })
            },
        )
        .expect("streaming chunk plan should complete");

        assert_eq!(chunked.text, "alpha beta gamma delta");
        let events = tx.take().expect("staged events");
        let text_deltas: Vec<&str> = events
            .iter()
            .filter_map(|event| event.text.as_deref())
            .filter(|text| !text.is_empty())
            .collect();
        assert!(
            text_deltas.len() > 2,
            "expected intra-chunk deltas beyond one per chunk, got {text_deltas:?}"
        );
        let streamed: String = text_deltas.concat();
        assert_eq!(streamed, "alpha beta gamma delta");
        assert!(
            events.last().map(|event| event.is_final).unwrap_or(false),
            "stream should end with the final marker"
        );
    }

    fn three_one_second_chunks(sr: u32) -> (Vec<f32>, Vec<AudioChunk>) {
        let samples = vec![0.0f32; sr as usize * 3];
        let chunk_plan = (0..3)
            .map(|idx| AudioChunk {
                start_sample: idx * sr as usize,
                end_sample: (idx + 1) * sr as usize,
            })
            .collect();
        (samples, chunk_plan)
    }

    #[test]
    fn chunk_diagnostics_record_the_text_each_chunk_committed() {
        let sr = 10u32;
        let samples = vec![0.0f32; sr as usize * 5];
        // The second chunk overlaps the first by one second; the third
        // follows a gap.
        let chunk_plan = vec![
            AudioChunk {
                start_sample: 0,
                end_sample: 30,
            },
            AudioChunk {
                start_sample: 20,
                end_sample: 40,
            },
            AudioChunk {
                start_sample: 42,
                end_sample: 50,
            },
        ];
        let texts = ["one two three", "three four", "five"];
        let mut sequence = 0usize;
        let mut idx = 0usize;
        let chunked = NativeExecutor::transcribe_with_chunk_plan_with_details(
            "req-committed",
            &|| false,
            None,
            EngineStreamPolicy::FailOnFull,
            &mut sequence,
            &samples,
            sr,
            &chunk_plan,
            &NativeExecutor::asr_long_form_config(),
            |_chunk_audio, _sr| {
                idx += 1;
                Ok(AsrChunkTranscription {
                    text: texts[idx - 1].to_string(),
                    diagnostics: None,
                })
            },
        )
        .expect("chunk plan should complete");

        let committed: Vec<&str> = chunked
            .chunk_diagnostics
            .iter()
            .map(|entry| entry["committed_text"].as_str().expect("committed text"))
            .collect();
        assert_eq!(chunked.text, "one two three four five");
        assert_eq!(committed, ["one two three", "four", "five"]);
    }

    #[test]
    fn chunk_plan_stops_at_the_next_chunk_boundary_after_cancellation() {
        let sr = 10u32;
        let (samples, chunk_plan) = three_one_second_chunks(sr);
        let tx = StreamStagingBuffer::default();
        let mut sequence = 0usize;
        let cancelled = std::cell::Cell::new(false);
        let mut calls = 0usize;

        let result = NativeExecutor::transcribe_with_chunk_plan_with_details(
            "req-cancel",
            &|| cancelled.get(),
            Some(&tx),
            EngineStreamPolicy::FailOnFull,
            &mut sequence,
            &samples,
            sr,
            &chunk_plan,
            &NativeExecutor::asr_long_form_config(),
            |_chunk_audio, _sr| {
                calls += 1;
                // The client goes away while the first chunk is running.
                cancelled.set(true);
                Ok(AsrChunkTranscription {
                    text: format!("chunk {calls}"),
                    diagnostics: None,
                })
            },
        );

        assert!(
            matches!(result, Err(crate::error::Error::Cancelled(ref id)) if id == "req-cancel"),
            "{result:?}"
        );
        assert_eq!(calls, 1, "no chunk may start after cancellation");
        let events = tx.take().expect("staged events");
        assert!(events.iter().all(|event| event.text.is_none()));
        assert!(events.iter().all(|event| !event.is_final));
    }

    #[test]
    fn chunk_partial_emitter_stops_streaming_once_cancelled() {
        let sr = 10u32;
        let (samples, chunk_plan) = three_one_second_chunks(sr);
        let tx = StreamStagingBuffer::default();
        let mut sequence = 0usize;
        let cancelled = std::cell::Cell::new(false);
        let mut calls = 0usize;

        let result = NativeExecutor::transcribe_with_chunk_plan_with_streaming_details_and_options(
            "req-cancel-partial",
            &|| cancelled.get(),
            Some(&tx),
            EngineStreamPolicy::FailOnFull,
            &mut sequence,
            &samples,
            sr,
            &chunk_plan,
            &NativeExecutor::asr_long_form_config(),
            AsrChunkStreamOptions {
                stable_text_holdback_chars: 0,
            },
            |_chunk_audio, _sr, _prefix, partial| {
                calls += 1;
                let emit = partial.expect("streaming chunk plans pass an emitter");
                emit("alpha ");
                cancelled.set(true);
                emit("beta ");
                emit("gamma");
                Ok(AsrChunkTranscription {
                    text: "alpha beta gamma".to_string(),
                    diagnostics: None,
                })
            },
        );

        assert!(
            matches!(result, Err(crate::error::Error::Cancelled(_))),
            "{result:?}"
        );
        assert_eq!(calls, 1);
        let streamed: String = tx
            .take()
            .expect("staged events")
            .iter()
            .filter_map(|event| event.text.clone())
            .collect();
        assert_eq!(streamed, "alpha");
    }

    #[test]
    fn chunk_plan_streaming_without_partials_keeps_chunk_level_deltas() {
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
        let tx = StreamStagingBuffer::default();
        let mut sequence = 0usize;

        let mut chunk_idx = 0usize;
        let chunked = NativeExecutor::transcribe_with_chunk_plan_with_streaming_details_and_options(
            "req-no-partial",
            &|| false,
            Some(&tx),
            EngineStreamPolicy::FailOnFull,
            &mut sequence,
            &samples,
            sr,
            &chunk_plan,
            &NativeExecutor::asr_long_form_config(),
            AsrChunkStreamOptions {
                stable_text_holdback_chars: 0,
            },
            |_chunk_audio, _sr, _prefix, _partial| {
                let idx = chunk_idx;
                chunk_idx += 1;
                Ok(AsrChunkTranscription {
                    text: format!("chunk-{idx}"),
                    diagnostics: None,
                })
            },
        )
        .expect("streaming chunk plan should complete");

        assert_eq!(chunked.text, "chunk-0 chunk-1");
        let events = tx.take().expect("staged events");
        let text_deltas: Vec<&str> = events
            .iter()
            .filter_map(|event| event.text.as_deref())
            .filter(|text| !text.is_empty())
            .collect();
        assert_eq!(
            text_deltas.len(),
            2,
            "families that ignore the emitter keep one delta per chunk, got {text_deltas:?}"
        );
        assert_eq!(text_deltas.concat(), "chunk-0 chunk-1");
    }

    /// A run resumed chunk by chunk (one dispatch each, its state cloned
    /// between them, a fresh staging buffer per dispatch) streams and
    /// assembles exactly what one call over the whole plan does.
    #[test]
    fn a_chunk_run_resumed_per_dispatch_matches_one_call() {
        let sr = 10u32;
        let samples = vec![0.0f32; sr as usize * 3];
        let chunk_plan = (0..3)
            .map(|chunk| AudioChunk {
                start_sample: (chunk * sr as usize).saturating_sub(sr as usize / 2),
                end_sample: (chunk + 1) * sr as usize,
            })
            .collect::<Vec<_>>();
        let texts = [
            "the quick brown fox",
            "brown fox jumps over",
            "jumps over the lazy dog",
        ];
        let transcribe = |idx: &mut usize| {
            let text = texts[*idx];
            *idx += 1;
            move |_: &[f32], _: u32, _: &str, partial: Option<&mut dyn FnMut(&str)>| {
                if let Some(emit) = partial {
                    emit(text);
                }
                Ok(AsrChunkTranscription {
                    text: text.to_string(),
                    diagnostics: None,
                })
            }
        };
        let options = AsrChunkStreamOptions {
            stable_text_holdback_chars: 4,
        };
        let config = NativeExecutor::asr_long_form_config();
        let streamed = |events: Vec<StreamingOutput>| {
            events
                .into_iter()
                .map(|event| (event.sequence, event.text, event.is_final))
                .collect::<Vec<_>>()
        };

        let tx = StreamStagingBuffer::default();
        let (mut sequence, mut idx) = (0usize, 0usize);
        let whole = NativeExecutor::transcribe_with_chunk_plan_with_streaming_details_and_options(
            "req-resume",
            &|| false,
            Some(&tx),
            EngineStreamPolicy::FailOnFull,
            &mut sequence,
            &samples,
            sr,
            &chunk_plan,
            &config,
            options,
            |audio, sr, prefix, partial| transcribe(&mut idx)(audio, sr, prefix, partial),
        )
        .expect("one call");
        let whole_stream = streamed(tx.take().expect("staged"));

        let mut resumed_stream = Vec::new();
        let (mut sequence, mut idx) = (0usize, 0usize);
        let mut committed: Option<ChunkedAsrRun> = None;
        let resumed = loop {
            let tx = StreamStagingBuffer::default();
            let stream = ChunkStream {
                request_id: "req-resume",
                tx: Some(&tx),
                policy: EngineStreamPolicy::FailOnFull,
            };
            let mut run = match committed.clone() {
                Some(run) => run,
                None => ChunkedAsrRun::start(
                    &stream,
                    &mut sequence,
                    samples.len(),
                    sr,
                    &chunk_plan,
                    &config,
                )
                .expect("start"),
            };
            let mut chunk = transcribe(&mut idx);
            run.run_chunk(
                &stream,
                &mut sequence,
                &|| false,
                &samples,
                sr,
                &chunk_plan,
                options,
                &mut chunk,
            )
            .expect("chunk");
            let done = run.is_complete(&chunk_plan);
            let result = done.then(|| {
                run.clone()
                    .finish(&stream, &mut sequence, &chunk_plan, sr)
                    .expect("finish")
            });
            resumed_stream.extend(streamed(tx.take().expect("staged")));
            if let Some(result) = result {
                break result;
            }
            committed = Some(run);
        };

        assert_eq!(resumed.text, whole.text);
        assert_eq!(resumed.text, "the quick brown fox jumps over the lazy dog");
        assert_eq!(
            resumed.chunk_diagnostics.len(),
            whole.chunk_diagnostics.len()
        );
        assert_eq!(resumed_stream, whole_stream);
    }

    #[test]
    fn chunk_plan_streaming_partial_deltas_dedup_overlap_words() {
        let sr = 10u32;
        let samples = vec![0.0f32; sr as usize * 2];
        // The chunks share one second of audio.
        let chunk_plan = vec![
            AudioChunk {
                start_sample: 0,
                end_sample: sr as usize * 3 / 2,
            },
            AudioChunk {
                start_sample: sr as usize / 2,
                end_sample: sr as usize * 2,
            },
        ];
        let tx = StreamStagingBuffer::default();
        let mut sequence = 0usize;

        let mut chunk_idx = 0usize;
        let chunked = NativeExecutor::transcribe_with_chunk_plan_with_streaming_details_and_options(
            "req-overlap",
            &|| false,
            Some(&tx),
            EngineStreamPolicy::FailOnFull,
            &mut sequence,
            &samples,
            sr,
            &chunk_plan,
            &NativeExecutor::asr_long_form_config(),
            AsrChunkStreamOptions {
                stable_text_holdback_chars: 0,
            },
            |_chunk_audio, _sr, _prefix, partial| {
                let idx = chunk_idx;
                chunk_idx += 1;
                if idx == 0 {
                    return Ok(AsrChunkTranscription {
                        text: "the quick brown fox jumps over".to_string(),
                        diagnostics: None,
                    });
                }
                if let Some(emit) = partial {
                    emit("fox jumps over the");
                }
                Ok(AsrChunkTranscription {
                    text: "fox jumps over the lazy dog".to_string(),
                    diagnostics: None,
                })
            },
        )
        .expect("streaming chunk plan should complete");

        assert_eq!(
            chunked.text,
            "the quick brown fox jumps over the lazy dog"
        );
        let events = tx.take().expect("staged events");
        let text_deltas: Vec<&str> = events
            .iter()
            .filter_map(|event| event.text.as_deref())
            .filter(|text| !text.is_empty())
            .collect();
        let streamed: String = text_deltas.concat();
        assert_eq!(
            streamed,
            "the quick brown fox jumps over the lazy dog",
            "the overlap words must not appear twice in the stream"
        );
    }

    #[test]
    fn chunk_plan_streaming_holdback_defers_unstable_tail_until_commit() {
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
        let tx = StreamStagingBuffer::default();
        let mut sequence = 0usize;
        let chunk_zero_text = "one two three four five six seven";

        let mut chunk_idx = 0usize;
        let chunked = NativeExecutor::transcribe_with_chunk_plan_with_streaming_details_and_options(
            "req-holdback",
            &|| false,
            Some(&tx),
            EngineStreamPolicy::FailOnFull,
            &mut sequence,
            &samples,
            sr,
            &chunk_plan,
            &NativeExecutor::asr_long_form_config(),
            AsrChunkStreamOptions {
                stable_text_holdback_chars: 20,
            },
            |_chunk_audio, _sr, _prefix, partial| {
                let idx = chunk_idx;
                chunk_idx += 1;
                if idx == 0 {
                    if let Some(emit) = partial {
                        emit("one two three ");
                        emit("four five six seven");
                    }
                    return Ok(AsrChunkTranscription {
                        text: chunk_zero_text.to_string(),
                        diagnostics: None,
                    });
                }
                Ok(AsrChunkTranscription {
                    text: "eight".to_string(),
                    diagnostics: None,
                })
            },
        )
        .expect("streaming chunk plan should complete");

        assert_eq!(
            chunked.text,
            "one two three four five six seven eight"
        );
        let events = tx.take().expect("staged events");
        let text_deltas: Vec<&str> = events
            .iter()
            .filter_map(|event| event.text.as_deref())
            .filter(|text| !text.is_empty())
            .collect();
        assert_eq!(
            text_deltas.first().copied(),
            Some("one two three"),
            "only the stable head beyond the holdback may stream before commit"
        );
        let streamed: String = text_deltas.concat();
        assert_eq!(streamed, "one two three four five six seven eight");
    }

    /// Stream `chunk_texts` through the chunk executor (as word partials when
    /// `emit_partials`) and return the final transcript and the concatenated
    /// stream deltas.
    fn stream_chunks(
        chunk_plan: &[AudioChunk],
        sr: u32,
        chunk_texts: &[&str],
        holdback_chars: usize,
        emit_partials: bool,
    ) -> (String, String) {
        let samples = vec![0.0f32; chunk_plan.last().expect("chunks").end_sample];
        let tx = StreamStagingBuffer::default();
        let mut sequence = 0usize;
        let mut chunk_idx = 0usize;
        let chunked =
            NativeExecutor::transcribe_with_chunk_plan_with_streaming_details_and_options(
                "req-stream",
                &|| false,
                Some(&tx),
                EngineStreamPolicy::FailOnFull,
                &mut sequence,
                &samples,
                sr,
                chunk_plan,
                &NativeExecutor::asr_long_form_config(),
                AsrChunkStreamOptions {
                    stable_text_holdback_chars: holdback_chars,
                },
                |_chunk_audio, _sr, _prefix, partial| {
                    let text = chunk_texts[chunk_idx];
                    chunk_idx += 1;
                    if let (true, Some(emit)) = (emit_partials, partial) {
                        for word in text.split_inclusive(' ') {
                            emit(word);
                        }
                    }
                    Ok(AsrChunkTranscription {
                        text: text.to_string(),
                        diagnostics: None,
                    })
                },
            )
            .expect("streaming chunk plan should complete");
        let streamed = tx
            .take()
            .expect("staged events")
            .into_iter()
            .filter_map(|event| event.text)
            .collect::<String>();
        (chunked.text, streamed)
    }

    fn assert_stream_matches_final(
        chunk_plan: &[AudioChunk],
        sr: u32,
        chunk_texts: &[&str],
        expected: &str,
    ) {
        for holdback_chars in [0, 20] {
            for emit_partials in [true, false] {
                let (text, streamed) =
                    stream_chunks(chunk_plan, sr, chunk_texts, holdback_chars, emit_partials);
                assert_eq!(
                    text, expected,
                    "holdback {holdback_chars}, partials {emit_partials}"
                );
                assert_eq!(
                    streamed, text,
                    "holdback {holdback_chars}, partials {emit_partials}"
                );
            }
        }
    }

    #[test]
    fn chunk_stream_matches_the_final_transcript_with_one_second_overlaps() {
        // Six-second chunks at 10 Hz sharing one second of audio each.
        let chunk_plan = [(0, 60), (50, 110), (100, 160)].map(|(start, end)| AudioChunk {
            start_sample: start,
            end_sample: end,
        });
        assert_stream_matches_final(
            &chunk_plan,
            10,
            &[
                "one two three four five six seven",
                "seven eight nine ten eleven twelve thirteen",
                "thirteen fourteen fifteen sixteen seventeen",
            ],
            "one two three four five six seven eight nine ten eleven twelve thirteen \
             fourteen fifteen sixteen seventeen",
        );
    }

    #[test]
    fn chunk_stream_matches_the_final_transcript_with_three_second_overlaps() {
        // The partials of each later chunk start inside a three-second overlap
        // and never stream duplicated words that the commit then drops.
        let chunk_plan = [(0, 240), (210, 480), (450, 600)].map(|(start, end)| AudioChunk {
            start_sample: start,
            end_sample: end,
        });
        assert_stream_matches_final(
            &chunk_plan,
            10,
            &[
                "we walked along the river past the old mill",
                "past the old mill and on toward the harbor where boats",
                "harbor where boats rocked gently in the evening light",
            ],
            "we walked along the river past the old mill and on toward the harbor where \
             boats rocked gently in the evening light",
        );
    }

    #[test]
    fn chunk_stream_keeps_genuine_repeats_across_chunks_that_do_not_overlap() {
        // Speech-island plans leave a gap between chunks.
        let chunk_plan = [(0, 20), (30, 50)].map(|(start, end)| AudioChunk {
            start_sample: start,
            end_sample: end,
        });
        assert_stream_matches_final(
            &chunk_plan,
            10,
            &["I don't know.", "I don't know why."],
            "I don't know. I don't know why.",
        );
    }

    /// Run `chunks` (text and `timestamp_words` diagnostics, timed from each
    /// chunk's start) through the chunk executor, streaming.
    fn run_timed_chunks(
        chunk_plan: &[AudioChunk],
        sr: u32,
        chunks: &[(&str, Option<serde_json::Value>)],
    ) -> (super::ChunkedAsrTranscription, String) {
        let samples = vec![0.0f32; chunk_plan.last().expect("chunks").end_sample];
        let tx = StreamStagingBuffer::default();
        let mut sequence = 0usize;
        let mut chunk_idx = 0usize;
        let chunked =
            NativeExecutor::transcribe_with_chunk_plan_with_streaming_details_and_options(
                "req-timed",
                &|| false,
                Some(&tx),
                EngineStreamPolicy::FailOnFull,
                &mut sequence,
                &samples,
                sr,
                chunk_plan,
                &NativeExecutor::asr_long_form_config(),
                AsrChunkStreamOptions::default(),
                |_chunk_audio, _sr, _prefix, _partial| {
                    let (text, words) = &chunks[chunk_idx];
                    chunk_idx += 1;
                    Ok(AsrChunkTranscription {
                        text: text.to_string(),
                        diagnostics: words
                            .clone()
                            .map(|words| serde_json::json!({ "timestamp_words": words })),
                    })
                },
            )
            .expect("timed chunk plan should complete");
        let streamed = tx
            .take()
            .expect("staged events")
            .into_iter()
            .filter_map(|event| event.text)
            .collect::<String>();
        (chunked, streamed)
    }

    fn words_json(words: &[(&str, f32, f32)]) -> serde_json::Value {
        serde_json::Value::Array(
            words
                .iter()
                .map(|(word, start, end)| {
                    serde_json::json!({
                        "word": word,
                        "start_time_seconds": start,
                        "end_time_seconds": end,
                    })
                })
                .collect(),
        )
    }

    /// Chunks that time their words merge by time: the one-word overlap
    /// ("fox.") that text dedup cannot see is kept once, the transcript's
    /// word timings come back on the input timeline, and the stream matches
    /// the final text.
    #[test]
    fn timed_chunks_merge_by_word_time() {
        // 10 Hz: chunk 0 covers 0-8 s, chunk 1 7-15 s; they hand over at 7.5 s.
        let chunk_plan = [(0, 80), (70, 150)].map(|(start, end)| AudioChunk {
            start_sample: start,
            end_sample: end,
        });
        let (chunked, streamed) = run_timed_chunks(
            &chunk_plan,
            10,
            &[
                (
                    "the quick fox.",
                    Some(words_json(&[
                        ("the", 5.0, 5.4),
                        ("quick", 5.5, 6.5),
                        ("fox.", 7.0, 7.6),
                    ])),
                ),
                (
                    "fox. It ran.",
                    Some(words_json(&[
                        ("fox.", 0.0, 0.6),
                        ("It", 1.0, 1.3),
                        ("ran.", 1.4, 2.0),
                    ])),
                ),
            ],
        );
        assert_eq!(chunked.text, "the quick fox. It ran.");
        assert_eq!(streamed, chunked.text);
        let words = chunked.timestamp_words.expect("timed transcript");
        let texts = words
            .iter()
            .map(|word| word.text.as_str())
            .collect::<Vec<_>>();
        assert_eq!(texts, ["the", "quick", "fox.", "It", "ran."]);
        assert_eq!((words[3].start_secs, words[3].end_secs), (8.0, 8.3));
    }

    /// A phrase repeated just after the overlap looks like overlap text to the
    /// text merge, which drops it; by time it lies past the seam and stays.
    #[test]
    fn timed_chunks_keep_a_repeat_the_text_merge_would_drop() {
        let chunk_plan = [(0, 80), (70, 150)].map(|(start, end)| AudioChunk {
            start_sample: start,
            end_sample: end,
        });
        let first = (
            "you say I know",
            Some(words_json(&[
                ("you", 4.0, 4.3),
                ("say", 4.4, 4.8),
                ("I", 6.0, 6.2),
                ("know", 6.2, 6.6),
            ])),
        );
        let second = (
            "I know that",
            Some(words_json(&[
                ("I", 0.6, 0.8),
                ("know", 0.8, 1.2),
                ("that", 1.3, 1.6),
            ])),
        );
        let (timed, _) = run_timed_chunks(&chunk_plan, 10, &[first.clone(), second.clone()]);
        assert_eq!(timed.text, "you say I know I know that");
        let (by_text, _) = run_timed_chunks(&chunk_plan, 10, &[(first.0, None), (second.0, None)]);
        assert_eq!(by_text.text, "you say I know that");
    }

    /// A chunk without word timings for its text (no list, or an empty list
    /// beside non-empty text) is merged by text, and the transcript then
    /// reports no word timings.
    #[test]
    fn untimed_chunk_falls_back_to_the_text_merge() {
        let chunk_plan = [(0, 80), (70, 150)].map(|(start, end)| AudioChunk {
            start_sample: start,
            end_sample: end,
        });
        for untimed in [None, Some(serde_json::json!([]))] {
            let (chunked, streamed) = run_timed_chunks(
                &chunk_plan,
                10,
                &[
                    ("one two three four", untimed.clone()),
                    ("two three four five", untimed),
                ],
            );
            assert_eq!(chunked.text, "one two three four five");
            assert_eq!(streamed, chunked.text);
            assert!(chunked.timestamp_words.is_none());
        }
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
            &|| false,
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
    fn chunk_plan_streams_asr_progress_at_chunk_boundaries() {
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
        let tx = StreamStagingBuffer::default();
        let mut chunk_idx = 0usize;
        let mut sequence = 0usize;

        let merged = NativeExecutor::transcribe_with_chunk_plan_with_details(
            "req-progress",
            &|| false,
            Some(&tx),
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
                    diagnostics: None,
                })
            },
        )
        .expect("chunk plan should complete");

        assert!(merged.text.contains("chunk-0"));
        assert!(sequence > 0);

        let mut progress_events = Vec::new();
        let mut final_seen = false;
        for event in tx.take().expect("staged events") {
            if let Some(progress) = event.asr_progress {
                progress_events.push(progress);
            }
            if event.is_final {
                final_seen = true;
            }
        }

        assert!(final_seen);
        assert_eq!(progress_events.len(), 6);
        assert_eq!(progress_events[0].phase, AsrProgressPhase::Processing);
        assert_eq!(progress_events[0].total_chunks, Some(2));
        assert_eq!(progress_events[1].phase, AsrProgressPhase::ChunkStarted);
        assert_eq!(progress_events[1].current_chunk, Some(1));
        assert_eq!(progress_events[2].phase, AsrProgressPhase::ChunkFinished);
        assert_eq!(progress_events[2].processed_audio_secs, Some(1.0));
        assert_eq!(progress_events[3].phase, AsrProgressPhase::ChunkStarted);
        assert_eq!(progress_events[3].current_chunk, Some(2));
        assert_eq!(progress_events[4].phase, AsrProgressPhase::ChunkFinished);
        assert_eq!(progress_events[4].percent, Some(100.0));
        assert_eq!(progress_events[5].phase, AsrProgressPhase::Complete);

        let percents: Vec<f64> = progress_events
            .iter()
            .filter_map(|progress| progress.percent)
            .collect();
        assert!(
            percents.windows(2).all(|pair| pair[0] <= pair[1]),
            "progress percent should be monotonic: {percents:?}"
        );
    }

    #[test]
    fn chunk_plan_context_callback_receives_assembled_prefix_text() {
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

        let mut prefixes = Vec::new();
        let mut chunk_idx = 0usize;
        let mut sequence = 0usize;
        let merged = NativeExecutor::transcribe_with_chunk_plan_with_context_and_details(
            "req-prefix",
            None,
            EngineStreamPolicy::FailOnFull,
            &mut sequence,
            &samples,
            sr,
            &chunk_plan,
            &NativeExecutor::asr_long_form_config(),
            |_chunk_audio, _sr, prefix_text| {
                prefixes.push(prefix_text.to_string());
                let idx = chunk_idx;
                chunk_idx = chunk_idx.saturating_add(1);
                Ok(AsrChunkTranscription {
                    text: format!("chunk-{idx}"),
                    diagnostics: None,
                })
            },
        )
        .expect("chunk plan should complete");

        assert_eq!(prefixes.len(), 2);
        assert!(prefixes[0].is_empty());
        assert!(prefixes[1].contains("chunk-0"));
        assert_eq!(merged.chunk_diagnostics[0]["context_prefix_words"], 0);
        assert_eq!(merged.chunk_diagnostics[1]["context_prefix_words"], 1);
        assert!(merged.text.contains("chunk-1"));
    }

    #[test]
    fn chunk_plan_diagnostics_can_embed_transcription_details() {
        let sr = 10u32;
        let samples = vec![0.0f32; sr as usize * 2];
        let plan = NativeExecutor::asr_route_plan(
            &samples,
            sr,
            &caps(ModelFamily::Qwen3Asr, Some(1.0)),
            false,
        );

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

        let tx = StreamStagingBuffer::default();
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
        for event in tx.take().expect("staged events") {
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
        let tx = StreamStagingBuffer::default();
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
        assert_eq!(sequence, 6);

        let events = tx.take().expect("staged events");

        assert_eq!(events.len(), 6);
        assert_eq!(
            events
                .iter()
                .filter_map(|event| event.asr_progress.as_ref().map(|progress| progress.phase))
                .collect::<Vec<_>>(),
            vec![
                AsrProgressPhase::Processing,
                AsrProgressPhase::ChunkStarted,
                AsrProgressPhase::ChunkFinished,
                AsrProgressPhase::Complete,
            ]
        );

        let text_events = events
            .iter()
            .filter(|event| event.text.is_some())
            .collect::<Vec<_>>();
        assert_eq!(text_events.len(), 1);
        let text_event = text_events[0];
        assert_eq!(text_event.sequence, 3);
        assert_eq!(text_event.text.as_deref(), Some(long_delta.trim()));

        let final_event = events
            .iter()
            .find(|event| event.is_final)
            .expect("final marker");
        assert_eq!(final_event.sequence, 5);
        assert!(final_event.is_final);
    }
}
