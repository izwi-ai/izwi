//! Core types for the inference engine.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Unique identifier for a request.
pub type RequestId = String;

/// Unique identifier for a sequence within a request.
pub type SequenceId = u64;

/// Token ID type.
pub type TokenId = u32;

/// Block ID for KV cache.
pub type BlockId = usize;

/// Generation parameters for audio synthesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParams {
    /// Temperature for sampling (0.0 = deterministic, 1.0 = more random)
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p (nucleus) sampling threshold
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Top-k sampling (0 = disabled)
    #[serde(default)]
    pub top_k: usize,

    /// Repetition penalty to avoid loops
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,

    /// Presence penalty applied once to previously generated tokens.
    #[serde(default)]
    pub presence_penalty: f32,

    /// Maximum number of tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Speaker/voice identifier
    #[serde(default)]
    pub speaker: Option<String>,

    /// Voice identifier alias used by some TTS models.
    #[serde(default)]
    pub voice: Option<String>,

    /// Audio temperature for audio token sampling
    #[serde(default)]
    pub audio_temperature: Option<f32>,

    /// Audio top-k for audio token sampling
    #[serde(default)]
    pub audio_top_k: Option<usize>,

    /// Speed factor (1.0 = normal)
    #[serde(default = "default_speed")]
    pub speed: f32,

    /// Stop sequences (generation stops when any of these are produced)
    #[serde(default)]
    pub stop_sequences: Vec<String>,

    /// Stop token IDs
    #[serde(default)]
    pub stop_token_ids: Vec<TokenId>,
}

fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    0.9
}
fn default_repetition_penalty() -> f32 {
    1.1
}
fn default_max_tokens() -> usize {
    2048
}
fn default_speed() -> f32 {
    1.0
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            temperature: default_temperature(),
            top_p: default_top_p(),
            top_k: 0,
            repetition_penalty: default_repetition_penalty(),
            presence_penalty: 0.0,
            max_tokens: default_max_tokens(),
            speaker: None,
            voice: None,
            audio_temperature: None,
            audio_top_k: None,
            speed: default_speed(),
            stop_sequences: Vec::new(),
            stop_token_ids: Vec::new(),
        }
    }
}

/// Audio output from generation.
#[derive(Debug, Clone)]
pub struct AudioOutput {
    /// Raw audio samples (f32, mono)
    pub samples: Vec<f32>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Duration in seconds
    pub duration_secs: f32,
}

impl AudioOutput {
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        let duration_secs = samples.len() as f32 / sample_rate as f32;
        Self {
            samples,
            sample_rate,
            duration_secs,
        }
    }

    /// Create empty audio output
    pub fn empty(sample_rate: u32) -> Self {
        Self {
            samples: Vec::new(),
            sample_rate,
            duration_secs: 0.0,
        }
    }

    /// Append samples from another output
    pub fn append(&mut self, other: &AudioOutput) {
        self.samples.extend_from_slice(&other.samples);
        self.duration_secs = self.samples.len() as f32 / self.sample_rate as f32;
    }
}

/// Complete engine output for a request.
#[derive(Debug, Clone)]
pub struct EngineOutput {
    /// Request ID
    pub request_id: RequestId,
    /// Sequence ID
    pub sequence_id: SequenceId,
    /// Generated audio
    pub audio: AudioOutput,
    /// Generated text (for ASR/chat)
    pub text: Option<String>,
    /// Optional input transcription for speech-to-speech requests.
    pub input_transcription: Option<String>,
    /// Number of tokens generated
    pub num_tokens: usize,
    /// Generation time
    pub generation_time: Duration,
    /// Whether generation is finished
    pub is_finished: bool,
    /// Finish reason
    pub finish_reason: Option<FinishReason>,
    /// Token statistics
    pub token_stats: TokenStats,
    /// Latency breakdown by request phase.
    pub latency_breakdown: Option<LatencyBreakdown>,
    /// Backend execution error when generation failed.
    pub error: Option<String>,
}

impl EngineOutput {
    /// Calculate real-time factor (RTF)
    /// RTF < 1.0 means faster than real-time
    pub fn rtf(&self) -> f32 {
        if self.audio.duration_secs > 0.0 {
            self.generation_time.as_secs_f32() / self.audio.duration_secs
        } else {
            0.0
        }
    }
}

/// Reason for finishing generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    /// Reached maximum token limit
    MaxTokens,
    /// Generated stop token (EOS)
    StopToken,
    /// Generated stop sequence
    StopSequence,
    /// Request was aborted
    Aborted,
    /// Error during generation
    Error,
}

/// Token generation statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenStats {
    /// Number of prompt tokens (prefill)
    pub prompt_tokens: usize,
    /// Number of generated tokens (decode)
    pub generated_tokens: usize,
    /// Prefill time in milliseconds
    pub prefill_time_ms: f32,
    /// Decode time in milliseconds
    pub decode_time_ms: f32,
    /// Tokens per second during decode
    pub tokens_per_second: f32,
}

/// Request latency phases captured by the scheduler/engine loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyBreakdown {
    /// Time spent waiting in queue before first scheduling.
    pub queue_wait_ms: f64,
    /// Total scheduler prefill phase wall-clock time attributed to this request.
    pub prefill_ms: f64,
    /// Total scheduler decode phase wall-clock time attributed to this request.
    pub decode_ms: f64,
    /// Time to first user-visible output in milliseconds, when observed.
    pub ttft_ms: Option<f64>,
    /// End-to-end request time in milliseconds.
    pub total_ms: f64,
    /// Number of prefill steps.
    pub prefill_steps: u32,
    /// Number of decode steps.
    pub decode_steps: u32,
}

impl TokenStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// Update decode statistics
    pub fn update_decode(&mut self, tokens: usize, time_ms: f32) {
        self.generated_tokens += tokens;
        self.decode_time_ms += time_ms;
        if self.decode_time_ms > 0.0 {
            self.tokens_per_second = (self.generated_tokens as f32 * 1000.0) / self.decode_time_ms;
        }
    }
}

/// Engine-level metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngineMetrics {
    /// Total number of engine steps executed
    pub total_steps: u64,
    /// Total requests processed
    pub requests_processed: u64,
    /// Total tokens generated
    pub tokens_generated: u64,
    /// Total audio seconds generated
    pub audio_seconds_generated: f64,
    /// Average tokens per second
    pub avg_tokens_per_second: f32,
    /// Average real-time factor
    pub avg_rtf: f32,
    /// Current KV cache memory usage in bytes
    pub kv_cache_memory_bytes: usize,
    /// Number of KV cache blocks allocated
    pub kv_cache_blocks_allocated: usize,
    /// Number of KV cache blocks free
    pub kv_cache_blocks_free: usize,
    /// Timestamp of last update
    #[serde(skip)]
    pub last_updated: Option<Instant>,
}

impl EngineMetrics {
    pub fn new() -> Self {
        Self {
            last_updated: Some(Instant::now()),
            ..Default::default()
        }
    }

    /// Update metrics with a completed request
    pub fn record_completion(&mut self, output: &EngineOutput) {
        self.tokens_generated += output.num_tokens as u64;
        self.audio_seconds_generated += output.audio.duration_secs as f64;

        // Update running averages
        let n = self.requests_processed as f32;
        if n > 0.0 {
            self.avg_tokens_per_second =
                (self.avg_tokens_per_second * (n - 1.0) + output.token_stats.tokens_per_second) / n;
            self.avg_rtf = (self.avg_rtf * (n - 1.0) + output.rtf()) / n;
        } else {
            self.avg_tokens_per_second = output.token_stats.tokens_per_second;
            self.avg_rtf = output.rtf();
        }

        self.last_updated = Some(Instant::now());
    }
}

/// Priority level for requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Priority {
    /// Low priority (background tasks)
    Low = 0,
    /// Normal priority (default)
    Normal = 1,
    /// High priority (user-facing)
    High = 2,
    /// Critical priority (system/admin)
    Critical = 3,
}

impl Default for Priority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Task type for the request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskType {
    /// Text-to-speech
    TTS,
    /// Automatic speech recognition
    ASR,
    /// Text chat generation.
    Chat,
    /// Speech-to-speech generation.
    SpeechToSpeech,
}

impl Default for TaskType {
    fn default() -> Self {
        Self::TTS
    }
}
