//! Output processing for the inference engine.
//!
//! Handles conversion of raw model outputs to user-facing results,
//! including streaming chunk and progress types.

use std::time::Duration;

use super::execution::{
    ExecutionDisposition, ExecutionFailure, FinishReason as ExecutionFinishReason,
    RetryDisposition, YieldReason,
};
use super::executor::{ExecutorOutput, REQUEST_DEADLINE_EXCEEDED};
use super::types::{AudioOutput, EngineOutput, FinishReason, RequestId, SequenceId, TokenStats};

use serde::{Deserialize, Serialize};

/// Coarse ASR processing progress for long-form and streaming transcription.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AsrProgress {
    /// Current processing phase.
    pub phase: AsrProgressPhase,
    /// 1-based chunk index when chunk progress is known.
    pub current_chunk: Option<usize>,
    /// Total chunk count when chunk progress is known.
    pub total_chunks: Option<usize>,
    /// Amount of source audio processed so far.
    pub processed_audio_secs: Option<f64>,
    /// Total source audio duration.
    pub total_audio_secs: Option<f64>,
    /// Monotonic percentage in the range 0-100 when known.
    pub percent: Option<f64>,
}

/// ASR progress phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AsrProgressPhase {
    Processing,
    ChunkStarted,
    ChunkFinished,
    Aligning,
    Complete,
}

/// Streaming output chunk.
#[derive(Debug, Clone)]
pub struct StreamingOutput {
    /// Request ID
    pub request_id: RequestId,
    /// Sequence number of this chunk
    pub sequence: usize,
    /// Audio samples in this chunk
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
    /// Whether this is the final chunk
    pub is_final: bool,
    /// Text output (for ASR/chat)
    pub text: Option<String>,
    /// Cumulative statistics
    pub stats: Option<StreamingStats>,
    /// Optional ASR progress metadata.
    pub asr_progress: Option<AsrProgress>,
    /// DS9.3: logprob entries for the tokens this chunk carries. Empty
    /// unless the request asked for logprobs.
    pub logprobs: Vec<crate::engine::types::TokenLogprob>,
}

impl StreamingOutput {
    /// Create a new streaming output chunk.
    pub fn new(
        request_id: RequestId,
        sequence: usize,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> Self {
        Self {
            request_id,
            sequence,
            samples,
            sample_rate,
            is_final: false,
            text: None,
            stats: None,
            asr_progress: None,
            logprobs: Vec::new(),
        }
    }

    /// Create a final chunk.
    pub fn final_chunk(
        request_id: RequestId,
        sequence: usize,
        samples: Vec<f32>,
        sample_rate: u32,
    ) -> Self {
        Self {
            request_id,
            sequence,
            samples,
            sample_rate,
            is_final: true,
            text: None,
            stats: None,
            asr_progress: None,
            logprobs: Vec::new(),
        }
    }

    /// Duration of this chunk in seconds.
    pub fn duration_secs(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }
}

/// Statistics for streaming output.
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    /// Total samples generated so far
    pub total_samples: usize,
    /// Total audio duration so far (seconds)
    pub total_duration_secs: f32,
    /// Number of chunks sent
    pub chunks_sent: usize,
    /// Time since first chunk
    pub elapsed_secs: f32,
    /// Real-time factor
    pub rtf: f32,
}

/// Output processor - converts raw outputs to user-facing results.
pub struct OutputProcessor {
    /// Sample rate for audio output
    sample_rate: u32,
}

impl OutputProcessor {
    /// Create a new output processor.
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Process executor output into engine output.
    pub fn process(
        &mut self,
        executor_output: ExecutorOutput,
        sequence_id: SequenceId,
        generation_time: Duration,
    ) -> EngineOutput {
        let disposition = if let Some(message) = executor_output.error.as_ref() {
            ExecutionDisposition::Failed(ExecutionFailure::invalid_output(message))
        } else if executor_output.finished {
            ExecutionDisposition::Finished(ExecutionFinishReason::Completed)
        } else {
            ExecutionDisposition::Yielded(YieldReason::QuantumExhausted)
        };
        self.process_execution(executor_output, &disposition, sequence_id, generation_time)
    }

    /// Process a payload whose lifecycle outcome has already been validated
    /// against an execution plan. The disposition is authoritative; raw
    /// `finished`/`error` flags are canonicalized at this public boundary.
    pub(crate) fn process_execution(
        &mut self,
        mut executor_output: ExecutorOutput,
        disposition: &ExecutionDisposition,
        sequence_id: SequenceId,
        generation_time: Duration,
    ) -> EngineOutput {
        let finish_reason = match disposition {
            ExecutionDisposition::Progress
            | ExecutionDisposition::Yielded(_)
            | ExecutionDisposition::RestartSequence(_) => {
                executor_output.finished = false;
                executor_output.error = None;
                None
            }
            ExecutionDisposition::Finished(ExecutionFinishReason::Completed) => {
                executor_output.finished = true;
                executor_output.error = None;
                Some(FinishReason::StopToken)
            }
            ExecutionDisposition::Finished(ExecutionFinishReason::Cancelled) => {
                executor_output.finished = true;
                executor_output.error = Some("request cancelled".to_string());
                Some(FinishReason::Aborted)
            }
            ExecutionDisposition::Finished(ExecutionFinishReason::TimedOut) => {
                executor_output.finished = true;
                executor_output.error = Some(REQUEST_DEADLINE_EXCEEDED.to_string());
                Some(FinishReason::Error)
            }
            ExecutionDisposition::Finished(ExecutionFinishReason::Rejected) => {
                executor_output.finished = true;
                executor_output.error = Some("request rejected".to_string());
                Some(FinishReason::Error)
            }
            ExecutionDisposition::Failed(failure) if failure.retry == RetryDisposition::Never => {
                executor_output.finished = true;
                executor_output.error = Some(failure.message.clone());
                Some(FinishReason::Error)
            }
            ExecutionDisposition::Failed(_) => {
                executor_output.finished = false;
                executor_output.error = None;
                None
            }
        };

        let audio = executor_output
            .audio
            .unwrap_or_else(|| AudioOutput::empty(self.sample_rate));
        // The executor's own count in its family's unit (frames, codec
        // steps, text tokens). Audio length never implies a count: one
        // sample-based estimate (samples / 256) reported 94 "tokens" a second
        // for 12.5 Qwen frames (TPIPE-9).
        let num_tokens = executor_output.tokens_generated;

        let token_stats = TokenStats {
            prompt_tokens: executor_output.tokens_processed,
            generated_tokens: num_tokens,
            // Stamped by the engine core from the request's phase timing; an
            // output that never reached scheduling has no prefill.
            prefill_time_ms: 0.0,
            decode_time_ms: generation_time.as_secs_f32() * 1000.0,
            tokens_per_second: if generation_time.as_secs_f32() > 0.0 {
                num_tokens as f32 / generation_time.as_secs_f32()
            } else {
                0.0
            },
            // Stamped by the engine core from the request's admission-time
            // managed prefix cursor; the processor never sees the request.
            cached_prefix_tokens: None,
        };

        EngineOutput {
            request_id: executor_output.request_id,
            sequence_id,
            audio,
            text: executor_output.text,
            input_transcription: executor_output.input_transcription,
            num_tokens,
            generation_time,
            is_finished: executor_output.finished,
            finish_reason,
            token_stats,
            latency_breakdown: None,
            asr_diagnostics: executor_output.asr_diagnostics,
            // Stamped by the engine core from the request's prepared audio.
            input_audio_duration_secs: None,
            logprobs: executor_output.logprobs,
            error: executor_output.error,
            provenance: super::OutcomeProvenance::produced_output(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_processor() {
        let processor = OutputProcessor::new(24000);
        assert_eq!(processor.sample_rate, 24000);
    }

    #[test]
    fn streamed_completion_preserves_rtf_without_counting_asr_input_as_output() {
        let mut processor = OutputProcessor::new(24_000);
        let mut streaming = ExecutorOutput::terminal("streamed".into());
        streaming.audio = Some(AudioOutput::streamed(48_000, 24_000));
        let output = processor.process(streaming, 1, Duration::from_secs(1));
        assert_eq!(output.audio.streamed_samples, Some(48_000));
        assert_eq!(output.rtf(), 0.5);
        // No count reported means none: audio length implies no tokens.
        assert_eq!(output.num_tokens, 0);
        assert!(output.audio.samples.is_empty());

        let mut asr = ExecutorOutput::terminal("input-duration".into());
        let mut input = AudioOutput::empty(24_000);
        input.duration_secs = 100.0;
        asr.audio = Some(input);
        asr.tokens_generated = 7;
        assert_eq!(
            processor.process(asr, 2, Duration::from_secs(1)).num_tokens,
            7
        );
    }

    /// TPIPE-9: 120 Qwen frames over 10 s of 24 kHz audio report 120, not
    /// the 937 that samples / 256 gave.
    #[test]
    fn generated_tokens_are_the_executors_count_not_an_audio_estimate() {
        let mut processor = OutputProcessor::new(24_000);
        let mut output = ExecutorOutput::terminal("frames".into());
        output.audio = Some(AudioOutput::new(vec![0.0; 240_000], 24_000));
        output.tokens_generated = 120;
        let output = processor.process(output, 1, Duration::from_secs(2));
        assert_eq!(output.num_tokens, 120);
        assert_eq!(output.rtf(), 0.2);
    }

    #[test]
    fn executor_error_does_not_invent_generated_tokens() {
        let mut processor = OutputProcessor::new(24_000);

        let output = processor.process(
            ExecutorOutput::error("failed".to_string(), "boom"),
            7,
            Duration::from_millis(1),
        );

        assert!(output.is_finished);
        assert_eq!(output.finish_reason, Some(FinishReason::Error));
        assert_eq!(output.num_tokens, 0);
        assert_eq!(output.token_stats.generated_tokens, 0);
    }

    #[test]
    fn cancelled_disposition_surfaces_as_aborted_error() {
        let mut processor = OutputProcessor::new(24_000);
        let output = processor.process_execution(
            ExecutorOutput::cancelled("cancelled".to_string()),
            &ExecutionDisposition::Finished(ExecutionFinishReason::Cancelled),
            9,
            Duration::from_millis(1),
        );

        assert!(output.is_finished);
        assert_eq!(output.finish_reason, Some(FinishReason::Aborted));
        assert_eq!(output.error.as_deref(), Some("request cancelled"));
        assert_eq!(output.num_tokens, 0);
    }

    #[test]
    fn test_streaming_output() {
        let chunk = StreamingOutput::new("test-req".to_string(), 0, vec![0.0; 4800], 24000);

        assert_eq!(chunk.duration_secs(), 0.2); // 4800 samples at 24kHz = 200ms
    }
}
