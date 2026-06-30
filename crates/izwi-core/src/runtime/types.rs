//! Runtime request/response types.

use crate::catalog::ModelVariant;
use crate::engine::GenerationParams;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
    pub llm_refined: bool,
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
}

#[derive(Debug, Clone)]
pub struct ChatGeneration {
    pub text: String,
    pub prompt_tokens: usize,
    pub tokens_generated: usize,
    pub generation_time_ms: f64,
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
        let mut options = InferenceOptions::default();
        // Preserve runtime API behavior where max_tokens=0 means "auto".
        options.max_tokens = 0;

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

    /// Generation statistics
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

/// Statistics for a generated chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkStats {
    /// Time to generate this chunk (ms)
    pub generation_time_ms: f32,
    /// Tokens generated for this chunk
    pub tokens_generated: usize,
    /// Real-time factor (< 1.0 means faster than real-time)
    pub rtf: f32,
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
}

impl GenerationResult {
    /// Duration in seconds
    pub fn duration_secs(&self) -> f32 {
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
