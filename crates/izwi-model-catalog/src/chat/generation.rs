//! What a chat generation returns: its text, token counts, finish reason,
//! latency phases and per-token logprobs.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct ChatGeneration {
    pub latency_breakdown: Option<LatencyBreakdown>,
    pub finish_reason: Option<FinishReason>,
    pub text: String,
    pub prompt_tokens: usize,
    pub tokens_generated: usize,
    pub generation_time_ms: f64,
    /// DS9.1: prompt tokens served from the managed prefix cache. Always a
    /// subset of `prompt_tokens`; `None` when the executing runtime did not
    /// measure prefix reuse (unavailable, not a zero measurement).
    pub cached_prompt_tokens: Option<u64>,
    /// DS9.3: per-token logprob entries for the generated text. Empty
    /// unless the request asked for logprobs.
    pub logprobs: Vec<TokenLogprob>,
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

/// DS9.3: one top alternative in a token's logprob entry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopTokenLogprob {
    /// Decoded surface text of this token (single-token decode).
    pub token: String,
    /// log_softmax of the raw logit, before penalties and temperature.
    pub logprob: f32,
    /// UTF-8 bytes of `token`.
    pub bytes: Vec<u8>,
}

/// DS9.3: per-token logprob entry for one sampled output token.
///
/// Logprobs are computed from the raw model distribution (log_softmax of the
/// raw logits) so they stay independent of sampling parameters, matching the
/// raw-logprob semantics of serving engines. Token strings use single-token
/// decoding: byte-level pieces that split a multi-byte UTF-8 sequence decode
/// lossily, and `bytes` carries those lossy bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<TopTokenLogprob>,
}

/// Request latency phases captured by the scheduler/engine loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyBreakdown {
    /// Time between runtime coordinator admission and engine insertion:
    /// input preparation, model residency and lane waits. Every other phase
    /// is measured from engine insertion, so a request's time since runtime
    /// admission is this plus that phase. Absent for direct engine callers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pre_engine_ms: Option<f64>,
    /// Time spent waiting in the engine scheduler queue before first scheduling.
    pub queue_wait_ms: f64,
    /// Time spent decoding input media before model execution, when measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_decode_ms: Option<f64>,
    /// Time spent normalizing/preparing inputs, when measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalization_ms: Option<f64>,
    /// Total scheduler prefill phase wall-clock time attributed to this request.
    pub prefill_ms: f64,
    /// Sum of physical decode batch service durations attributed to this request.
    /// Shared batches are attributed in full; this is not decode wall time.
    pub decode_ms: f64,
    /// Monotonic interval from registration of the first entered decode dispatch
    /// through the last successful decode token commit. Includes dispatch waiting,
    /// inter-quantum scheduling, retries and commit overhead; excludes initial
    /// admission queue and prefill. Absent when no decode tokens were committed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_wall_ms: Option<f64>,
    /// Tokens durably committed by decode quanta, never speculative proposals.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_tokens: Option<usize>,
    /// First committed token group through last committed token group. Groups may
    /// contain multiple speculative tokens; this is not SSE inter-token latency.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_first_token_ms: Option<f64>,
    /// Time spent sampling model outputs, when measured separately.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_ms: Option<f64>,
    /// Time spent encoding or decoding codec/audio representations, when measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codec_ms: Option<f64>,
    /// Time spent on final postprocessing and artifact preparation, when measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub postprocess_ms: Option<f64>,
    /// Time to first user-visible output in milliseconds, when observed.
    pub ttft_ms: Option<f64>,
    /// End-to-end request time in milliseconds.
    pub total_ms: f64,
    /// Number of prefill steps.
    pub prefill_steps: u32,
    /// Number of decode steps.
    pub decode_steps: u32,
}
