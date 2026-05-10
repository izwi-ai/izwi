//! Output processing for the inference engine.
//!
//! Handles conversion of raw model outputs to user-facing results,
//! including streaming chunked output and stop condition detection.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::debug;

use super::executor::ExecutorOutput;
use super::types::{AudioOutput, EngineOutput, FinishReason, RequestId, SequenceId, TokenStats};

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
    /// Chunk size for streaming (samples)
    streaming_chunk_size: usize,
    /// Active streaming sessions
    streaming_sessions: HashMap<RequestId, StreamingSession>,
}

/// State for an active streaming session.
struct StreamingSession {
    request_id: RequestId,
    sequence_id: SequenceId,
    start_time: Instant,
    samples_buffer: Vec<f32>,
    chunks_sent: usize,
    total_samples_sent: usize,
    tx: mpsc::Sender<StreamingOutput>,
}

impl OutputProcessor {
    /// Create a new output processor.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            streaming_chunk_size: 4800, // 200ms at 24kHz
            streaming_sessions: HashMap::new(),
        }
    }

    /// Set streaming chunk size.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.streaming_chunk_size = size;
        self
    }

    /// Process executor output into engine output.
    pub fn process(
        &mut self,
        executor_output: ExecutorOutput,
        sequence_id: SequenceId,
        generation_time: Duration,
    ) -> EngineOutput {
        let finish_reason = if executor_output.error.is_some() {
            Some(FinishReason::Error)
        } else if executor_output.finished {
            Some(FinishReason::StopToken)
        } else {
            None
        };

        let audio = executor_output
            .audio
            .unwrap_or_else(|| AudioOutput::empty(self.sample_rate));
        let num_tokens = executor_output.tokens_generated.max(
            // Estimate tokens from audio length if not provided
            (audio.samples.len() / 256).max(1),
        );

        let token_stats = TokenStats {
            prompt_tokens: executor_output.tokens_processed,
            generated_tokens: num_tokens,
            prefill_time_ms: 0.0, // Not tracked at this level
            decode_time_ms: generation_time.as_secs_f32() * 1000.0,
            tokens_per_second: if generation_time.as_secs_f32() > 0.0 {
                num_tokens as f32 / generation_time.as_secs_f32()
            } else {
                0.0
            },
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
            error: executor_output.error,
        }
    }

    /// Start a streaming session.
    pub fn start_streaming(
        &mut self,
        request_id: RequestId,
        sequence_id: SequenceId,
        tx: mpsc::Sender<StreamingOutput>,
    ) {
        let session = StreamingSession {
            request_id: request_id.clone(),
            sequence_id,
            start_time: Instant::now(),
            samples_buffer: Vec::new(),
            chunks_sent: 0,
            total_samples_sent: 0,
            tx,
        };
        self.streaming_sessions.insert(request_id, session);
    }

    /// Add samples to a streaming session.
    pub async fn add_streaming_samples(
        &mut self,
        request_id: &RequestId,
        samples: Vec<f32>,
    ) -> bool {
        let session = match self.streaming_sessions.get_mut(request_id) {
            Some(s) => s,
            None => return false,
        };

        session.samples_buffer.extend(samples);

        // Send chunks when buffer is large enough
        while session.samples_buffer.len() >= self.streaming_chunk_size {
            let chunk_samples: Vec<f32> = session
                .samples_buffer
                .drain(..self.streaming_chunk_size)
                .collect();

            let stats = StreamingStats {
                total_samples: session.total_samples_sent + chunk_samples.len(),
                total_duration_secs: (session.total_samples_sent + chunk_samples.len()) as f32
                    / self.sample_rate as f32,
                chunks_sent: session.chunks_sent + 1,
                elapsed_secs: session.start_time.elapsed().as_secs_f32(),
                rtf: session.start_time.elapsed().as_secs_f32()
                    / ((session.total_samples_sent + chunk_samples.len()) as f32
                        / self.sample_rate as f32),
            };

            let output = StreamingOutput {
                request_id: session.request_id.clone(),
                sequence: session.chunks_sent,
                samples: chunk_samples.clone(),
                sample_rate: self.sample_rate,
                is_final: false,
                text: None,
                stats: Some(stats),
            };

            session.total_samples_sent += chunk_samples.len();
            session.chunks_sent += 1;

            if session.tx.send(output).await.is_err() {
                debug!("Streaming channel closed for {}", request_id);
                return false;
            }
        }

        true
    }

    /// Finish a streaming session.
    pub async fn finish_streaming(
        &mut self,
        request_id: &RequestId,
        text: Option<String>,
    ) -> Option<StreamingStats> {
        let session = self.streaming_sessions.remove(request_id)?;

        // Send remaining samples as final chunk
        let remaining_samples = session.samples_buffer;
        let total_samples = session.total_samples_sent + remaining_samples.len();

        let stats = StreamingStats {
            total_samples,
            total_duration_secs: total_samples as f32 / self.sample_rate as f32,
            chunks_sent: session.chunks_sent + 1,
            elapsed_secs: session.start_time.elapsed().as_secs_f32(),
            rtf: if total_samples > 0 {
                session.start_time.elapsed().as_secs_f32()
                    / (total_samples as f32 / self.sample_rate as f32)
            } else {
                0.0
            },
        };

        let output = StreamingOutput {
            request_id: session.request_id,
            sequence: session.chunks_sent,
            samples: remaining_samples,
            sample_rate: self.sample_rate,
            is_final: true,
            text,
            stats: Some(stats.clone()),
        };

        let _ = session.tx.send(output).await;

        Some(stats)
    }

    /// Cancel a streaming session.
    pub fn cancel_streaming(&mut self, request_id: &RequestId) {
        self.streaming_sessions.remove(request_id);
    }

    /// Check if a streaming session is active.
    pub fn is_streaming(&self, request_id: &RequestId) -> bool {
        self.streaming_sessions.contains_key(request_id)
    }

    /// Get number of active streaming sessions.
    pub fn active_streams(&self) -> usize {
        self.streaming_sessions.len()
    }
}

/// Stop condition checker.
pub struct StopChecker {
    /// Stop token IDs
    stop_token_ids: Vec<u32>,
    /// Maximum tokens
    max_tokens: usize,
    /// Maximum sequence length
    max_seq_len: usize,
}

impl StopChecker {
    /// Create a new stop checker.
    pub fn new(stop_token_ids: Vec<u32>, max_tokens: usize, max_seq_len: usize) -> Self {
        Self {
            stop_token_ids,
            max_tokens,
            max_seq_len,
        }
    }

    /// Check if generation should stop.
    pub fn should_stop(
        &self,
        generated_tokens: usize,
        total_tokens: usize,
        last_token: Option<u32>,
    ) -> Option<FinishReason> {
        // Check max tokens
        if generated_tokens >= self.max_tokens {
            return Some(FinishReason::MaxTokens);
        }

        // Check max sequence length
        if total_tokens >= self.max_seq_len {
            return Some(FinishReason::MaxTokens);
        }

        // Check stop tokens
        if let Some(token) = last_token {
            if self.stop_token_ids.contains(&token) {
                return Some(FinishReason::StopToken);
            }
        }

        None
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
    fn test_stop_checker() {
        let checker = StopChecker::new(vec![151673], 100, 1000);

        // Should not stop
        assert!(checker.should_stop(50, 100, Some(12345)).is_none());

        // Should stop - max tokens
        assert_eq!(
            checker.should_stop(100, 150, None),
            Some(FinishReason::MaxTokens)
        );

        // Should stop - stop token
        assert_eq!(
            checker.should_stop(50, 100, Some(151673)),
            Some(FinishReason::StopToken)
        );
    }

    #[test]
    fn test_streaming_output() {
        let chunk = StreamingOutput::new("test-req".to_string(), 0, vec![0.0; 4800], 24000);

        assert_eq!(chunk.duration_secs(), 0.2); // 4800 samples at 24kHz = 200ms
    }
}
