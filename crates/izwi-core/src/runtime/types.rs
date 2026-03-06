//! Runtime request/response types.

use crate::engine::GenerationParams;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct AsrTranscription {
    pub text: String,
    pub language: Option<String>,
    pub duration_secs: f32,
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
            is_final: false,
            stats: None,
        }
    }

    pub fn final_chunk(request_id: String, sequence: usize, samples: Vec<f32>) -> Self {
        Self {
            request_id,
            sequence,
            samples,
            is_final: true,
            stats: None,
        }
    }

    /// Duration in seconds
    pub fn duration_secs(&self, sample_rate: u32) -> f32 {
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
