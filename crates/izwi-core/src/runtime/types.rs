//! Runtime request/response types.

use crate::catalog::ModelVariant;
use crate::engine::{GenerationParams, Priority, WorkloadClass};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use uuid::Uuid;

pub use izwi_model_catalog::chat::{ChatGeneration, FinishReason as ChatFinishReason};

/// Server-side scheduling and admission metadata carried with an inference request.
///
/// This is deliberately skipped by serde on public request payloads. API layers
/// populate it after admission so clients cannot choose their own scheduler class.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RuntimeRequestContext {
    pub workload_class: WorkloadClass,
    /// Opaque hash of a trusted server tenant/principal namespace; never deserialized from public requests.
    pub tenant_key: Option<[u8; 32]>,
    pub admission_ms: Option<f64>,
    pub priority: Priority,
    pub deadline: Option<Instant>,
}

impl RuntimeRequestContext {
    pub fn new(workload_class: WorkloadClass) -> Self {
        Self {
            workload_class,
            tenant_key: None,
            admission_ms: None,
            priority: Priority::Normal,
            deadline: None,
        }
    }

    pub fn with_tenant_key(mut self, tenant_key: [u8; 32]) -> Self {
        self.tenant_key = Some(tenant_key);
        self
    }

    /// Derive the opaque tenant key from a trusted tenant identity (for
    /// example a gateway-attested tenant id). The key scopes fair scheduling
    /// and the managed prefix-cache namespace, so tenants never share KV.
    pub fn with_tenant_id(self, tenant_id: &str) -> Self {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"izwi.runtime.tenant-key.v1\0");
        hasher.update(tenant_id.as_bytes());
        self.with_tenant_key(hasher.finalize().into())
    }

    pub fn with_admission_ms(mut self, admission_ms: f64) -> Self {
        self.admission_ms = Some(admission_ms.max(0.0));
        self
    }

    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }
}

impl Default for RuntimeRequestContext {
    fn default() -> Self {
        Self::new(WorkloadClass::Online)
    }
}

#[derive(Debug, Clone)]
pub struct AsrTranscription {
    pub text: String,
    pub language: Option<String>,
    pub duration_secs: f32,
    pub asr_diagnostics: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerAttributedAsrStatus {
    Ready,
    Warning,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpeakerAttributedAsrTurn {
    pub speaker: String,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_secs: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_secs: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerAttributedAsrResult {
    pub text: String,
    pub language: Option<String>,
    pub duration_secs: f32,
    pub speaker_turns: Vec<SpeakerAttributedAsrTurn>,
    pub speaker_count: usize,
    pub status: SpeakerAttributedAsrStatus,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationSegment {
    pub speaker: String,
    pub start_secs: f32,
    pub end_secs: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct DiarizationResult {
    pub segments: Vec<DiarizationSegment>,
    pub duration_secs: f32,
    pub speaker_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationWord {
    pub word: String,
    pub speaker: String,
    pub start_secs: f32,
    pub end_secs: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speaker_confidence: Option<f32>,
    #[serde(default)]
    pub overlaps_segment: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationUtterance {
    pub speaker: String,
    pub start_secs: f32,
    pub end_secs: f32,
    pub text: String,
    #[serde(default)]
    pub word_start: usize,
    #[serde(default)]
    pub word_end: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiarizationTranscriptResult {
    pub segments: Vec<DiarizationSegment>,
    pub words: Vec<DiarizationWord>,
    pub utterances: Vec<DiarizationUtterance>,
    pub asr_text: String,
    pub raw_transcript: String,
    pub transcript: String,
    pub duration_secs: f32,
    pub speaker_count: usize,
    pub alignment_coverage: f32,
    pub unattributed_words: usize,
    /// Degraded steps the transcript went through (aligner windows with
    /// interval timings, a missing aligner, a truncated ASR decode).
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiarizationConfig {
    #[serde(default)]
    pub min_speakers: Option<usize>,
    #[serde(default)]
    pub max_speakers: Option<usize>,
    #[serde(default)]
    pub min_speech_duration_ms: Option<f32>,
    #[serde(default)]
    pub min_silence_duration_ms: Option<f32>,
    /// Clip Sortformer segments to padded external VAD speech regions.
    /// `None` defers to `IZWI_SORTFORMER_VAD_GATE` (default off): NeMo has
    /// no such gate, and it only helps on noisy input.
    #[serde(default)]
    pub vad_gate: Option<bool>,
}

/// Per-job controls of the diarization transcript pipeline.
#[derive(Debug, Clone, Default)]
pub struct DiarizationPipelineOptions {
    /// Language hint for the pipeline ASR; `None` lets it detect the
    /// language of each chunk.
    pub language: Option<String>,
    /// Stops the pipeline at its next stage (between ASR chunks and aligner
    /// windows) once cancelled.
    pub cancellation: crate::runtime::coordinator::JobCancellation,
}

#[derive(Debug, Clone)]
pub struct SpeechToSpeechGeneration {
    pub text: String,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub input_transcription: Option<String>,
    pub generation_time_ms: f64,
}

/// Canonical runtime inference options (shared with the engine layer).
pub type InferenceOptions = GenerationParams;

/// Compatibility wrapper for legacy runtime request payloads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Canonical inference options used end-to-end by engine/runtime.
    #[serde(flatten)]
    pub options: InferenceOptions,

    /// Enable streaming output for compatibility with older payloads.
    #[serde(default = "default_streaming")]
    pub streaming: bool,
}

fn default_streaming() -> bool {
    true
}

impl Default for GenerationConfig {
    fn default() -> Self {
        // Preserve runtime API behavior where max_tokens=0 means "auto".
        let options = InferenceOptions {
            max_tokens: 0,
            ..Default::default()
        };

        Self {
            options,
            streaming: default_streaming(),
        }
    }
}

/// Request for TTS generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    /// Unique request ID
    #[serde(default = "generate_request_id")]
    pub id: String,

    /// Resolved TTS model variant for this request.
    ///
    /// Public API layers should set this after resolving request defaults so
    /// concurrent TTS calls do not depend on process-global active model state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_variant: Option<ModelVariant>,

    /// Optional request correlation ID.
    #[serde(default)]
    pub correlation_id: Option<String>,

    /// Internal scheduling context populated by the server after admission.
    #[serde(skip)]
    pub runtime_context: RuntimeRequestContext,

    /// Text to synthesize
    pub text: String,

    /// Generation configuration
    #[serde(default)]
    pub config: GenerationConfig,

    /// Optional language hint for multilingual TTS models.
    #[serde(default)]
    pub language: Option<String>,

    /// Reference audio for voice cloning (base64 encoded)
    #[serde(default)]
    pub reference_audio: Option<String>,

    /// Reference text/transcript for voice cloning
    #[serde(default)]
    pub reference_text: Option<String>,

    /// Voice description for voice design models
    #[serde(default)]
    pub voice_description: Option<String>,
}

fn generate_request_id() -> String {
    Uuid::new_v4().to_string()
}

impl GenerationRequest {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            id: generate_request_id(),
            model_variant: None,
            correlation_id: None,
            runtime_context: RuntimeRequestContext::default(),
            text: text.into(),
            config: GenerationConfig::default(),
            language: None,
            reference_audio: None,
            reference_text: None,
            voice_description: None,
        }
    }

    pub fn with_config(mut self, config: GenerationConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_model_variant(mut self, model_variant: ModelVariant) -> Self {
        self.model_variant = Some(model_variant);
        self
    }

    pub fn with_runtime_context(mut self, runtime_context: RuntimeRequestContext) -> Self {
        self.runtime_context = runtime_context;
        self
    }

    pub fn with_speaker(mut self, speaker: impl Into<String>) -> Self {
        let speaker = speaker.into();
        self.config.options.speaker = Some(speaker.clone());
        self.config.options.voice = Some(speaker);
        self
    }
}

/// A chunk of generated audio
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Request ID this chunk belongs to
    pub request_id: String,

    /// Chunk sequence number
    pub sequence: usize,

    /// Audio samples (f32, mono)
    pub samples: Vec<f32>,

    /// Audio sample rate in Hz. A value of 0 means the producer did not attach
    /// rate metadata and callers should fall back to their negotiated rate.
    pub sample_rate: u32,

    /// Whether this is the final chunk
    pub is_final: bool,

    /// Generation statistics. Final chunks report cumulative request totals,
    /// including terminal markers with no PCM. Non-final statistics are deltas.
    pub stats: Option<ChunkStats>,
}

impl AudioChunk {
    pub fn new(request_id: String, sequence: usize, samples: Vec<f32>) -> Self {
        Self {
            request_id,
            sequence,
            samples,
            sample_rate: 0,
            is_final: false,
            stats: None,
        }
    }

    pub fn final_chunk(request_id: String, sequence: usize, samples: Vec<f32>) -> Self {
        Self {
            request_id,
            sequence,
            samples,
            sample_rate: 0,
            is_final: true,
            stats: None,
        }
    }

    pub fn with_sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    pub fn sample_rate_or(&self, fallback_sample_rate: u32) -> u32 {
        if self.sample_rate > 0 {
            self.sample_rate
        } else {
            fallback_sample_rate
        }
    }

    /// Duration in seconds
    pub fn duration_secs(&self, sample_rate: u32) -> f32 {
        let sample_rate = self.sample_rate_or(sample_rate);
        if sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f32 / sample_rate as f32
    }
}

/// How a TTS model's streaming route delivers audio (TPIPE-3): progressive
/// chunks as the model commits audio, or no streaming at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TtsStreamingMode {
    Chunked,
    Unsupported,
}

impl TtsStreamingMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Chunked => "chunked",
            Self::Unsupported => "unsupported",
        }
    }
}

/// Why speech generation ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TtsStopReason {
    /// The model ended speech itself.
    #[serde(rename = "stop")]
    EndOfSpeech,
    /// The output-frame budget ended it: the audio may be cut short.
    #[serde(rename = "length")]
    OutputBudget,
    /// The loaded context ended it before the budget: the audio may be cut
    /// short.
    #[serde(rename = "context_length")]
    ContextLimit,
}

impl TtsStopReason {
    /// The stop reason of a generation that committed `frames` audio frames
    /// against a budget of `budget_frames` (`None` without a budget, `0`).
    ///
    /// `decode_steps`, when the family reports it, tells end of speech from
    /// a context stop: a model that ends on its own runs one more decode step
    /// (the end-of-speech step) than it commits frames, while a request cut
    /// by the context the request was fitted to stops with every step
    /// committed. Without it, an early stop is taken as end of speech.
    pub fn classify(
        budget_frames: usize,
        frames: usize,
        decode_steps: Option<usize>,
    ) -> Option<Self> {
        if budget_frames == 0 {
            return None;
        }
        Some(if frames >= budget_frames {
            Self::OutputBudget
        } else {
            match decode_steps {
                Some(steps) if steps <= frames => Self::ContextLimit,
                _ => Self::EndOfSpeech,
            }
        })
    }

    /// Whether the audio may be cut short.
    pub fn truncated(self) -> bool {
        !matches!(self, Self::EndOfSpeech)
    }

    /// The combined reason of several segments: a budget stop in any
    /// segment cuts the whole, and an unknown segment makes it unknown.
    pub fn combine(reasons: impl IntoIterator<Item = Option<Self>>) -> Option<Self> {
        let mut combined = Some(Self::EndOfSpeech);
        for reason in reasons {
            match reason {
                Some(Self::EndOfSpeech) => {}
                Some(cut) => return Some(cut),
                None => combined = None,
            }
        }
        combined
    }

    /// The label: `stop`, `length` (budget) or `context_length`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EndOfSpeech => "stop",
            Self::OutputBudget => "length",
            Self::ContextLimit => "context_length",
        }
    }
}

/// Statistics for a generated chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkStats {
    /// Execution time (ms): request total on final chunks, otherwise chunk delta.
    pub generation_time_ms: f32,
    /// Committed model tokens (semantic frames for Fish), with the same total/delta convention.
    pub tokens_generated: usize,
    /// Real-time factor (< 1.0 means faster than real-time)
    pub rtf: f32,
    /// On a final chunk: why generation ended, when a frame budget applied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<TtsStopReason>,
}

/// Complete generation result
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub request_id: String,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub total_tokens: usize,
    pub total_time_ms: f32,
    pub diagnostics: Option<serde_json::Value>,
    /// Why generation ended, when a frame budget applied.
    pub stop_reason: Option<TtsStopReason>,
}

impl GenerationResult {
    /// Duration in seconds
    pub fn duration_secs(&self) -> f32 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f32 / self.sample_rate as f32
    }

    /// Real-time factor
    pub fn rtf(&self) -> f32 {
        let duration = self.duration_secs();
        if duration > 0.0 {
            (self.total_time_ms / 1000.0) / duration
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A budget stop is `length` (the audio may be cut), an earlier end is
    /// `stop`, and a combined long-form result is cut if any segment was.
    #[test]
    fn tts_stop_reason_reports_budget_stops_and_combines_segments() {
        assert_eq!(TtsStopReason::classify(0, 400, None), None);
        // Live, CPU Qwen: short 54 steps / 53 frames and long 983 / 982
        // ended on EOS; medium 313 / 313 ran into its 313-frame budget.
        assert_eq!(
            TtsStopReason::classify(82, 53, Some(54)),
            Some(TtsStopReason::EndOfSpeech)
        );
        assert_eq!(
            TtsStopReason::classify(1_500, 982, Some(983)),
            Some(TtsStopReason::EndOfSpeech)
        );
        assert_eq!(
            TtsStopReason::classify(313, 313, Some(313)),
            Some(TtsStopReason::OutputBudget)
        );
        // Stopped before the budget with every step committed: the context.
        assert_eq!(
            TtsStopReason::classify(313, 200, Some(200)),
            Some(TtsStopReason::ContextLimit)
        );
        assert_eq!(
            TtsStopReason::classify(400, 252, None),
            Some(TtsStopReason::EndOfSpeech)
        );
        assert!(TtsStopReason::ContextLimit.truncated());
        assert!(!TtsStopReason::EndOfSpeech.truncated());
        let stop = Some(TtsStopReason::EndOfSpeech);
        let cut = Some(TtsStopReason::OutputBudget);
        assert_eq!(TtsStopReason::combine([stop, stop]), stop);
        assert_eq!(TtsStopReason::combine([stop, cut, None]), cut);
        assert_eq!(TtsStopReason::combine([stop, None]), None);
        assert_eq!(TtsStopReason::OutputBudget.as_str(), "length");
        assert_eq!(
            serde_json::to_string(&TtsStopReason::EndOfSpeech).unwrap(),
            "\"stop\""
        );
    }

    #[test]
    fn generation_request_keeps_explicit_model_variant() {
        let request = GenerationRequest::new("hello").with_model_variant(ModelVariant::Kokoro82M);

        assert_eq!(request.model_variant, Some(ModelVariant::Kokoro82M));

        let serialized = serde_json::to_value(&request).expect("serialize generation request");
        assert_eq!(serialized["model_variant"], "Kokoro-82M");
    }

    #[test]
    fn generation_request_defaults_to_no_model_variant() {
        let request: GenerationRequest =
            serde_json::from_value(serde_json::json!({ "text": "hello" }))
                .expect("deserialize generation request");

        assert_eq!(request.model_variant, None);
    }

    #[test]
    fn audio_chunk_duration_prefers_attached_sample_rate() {
        let chunk =
            AudioChunk::new("req".to_string(), 0, vec![0.0; 48_000]).with_sample_rate(48_000);

        assert_eq!(chunk.duration_secs(24_000), 1.0);
        assert_eq!(chunk.sample_rate_or(24_000), 48_000);
    }
}
