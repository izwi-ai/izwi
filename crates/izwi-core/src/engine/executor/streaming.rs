use tokio::sync::mpsc;

use crate::error::{Error, Result};

use super::super::metrics::record_engine_stream_backpressure;
use super::super::output::{AsrProgress, StreamingOutput};
use super::super::request::{EngineCoreRequest, EngineStreamPolicy};
use super::NativeExecutor;

pub(super) type StreamBackpressurePolicy = EngineStreamPolicy;

pub(super) struct StreamSink<'a> {
    tx: &'a mpsc::Sender<StreamingOutput>,
    policy: StreamBackpressurePolicy,
}

impl<'a> StreamSink<'a> {
    fn fail_on_full(tx: &'a mpsc::Sender<StreamingOutput>) -> Self {
        Self::with_policy(tx, StreamBackpressurePolicy::FailOnFull)
    }

    fn with_policy(
        tx: &'a mpsc::Sender<StreamingOutput>,
        policy: StreamBackpressurePolicy,
    ) -> Self {
        Self { tx, policy }
    }

    fn policy(&self) -> StreamBackpressurePolicy {
        self.policy
    }

    fn send(&self, output: StreamingOutput) -> Result<()> {
        match self.policy {
            StreamBackpressurePolicy::FailOnFull => {
                self.tx.try_send(output).map_err(stream_send_error)
            }
            StreamBackpressurePolicy::BlockWithDeadline => {
                self.tx.try_send(output).map_err(stream_send_error)
            }
            StreamBackpressurePolicy::DropOldest
            | StreamBackpressurePolicy::Coalesce
            | StreamBackpressurePolicy::Sample => match self.tx.try_send(output) {
                Ok(()) => Ok(()),
                Err(mpsc::error::TrySendError::Closed(output)) => {
                    Err(stream_send_error(mpsc::error::TrySendError::Closed(output)))
                }
                Err(mpsc::error::TrySendError::Full(_)) => {
                    record_engine_stream_backpressure();
                    Ok(())
                }
            },
        }
    }
}

impl NativeExecutor {
    pub(super) fn stream_sender(
        request: &EngineCoreRequest,
    ) -> Option<mpsc::Sender<StreamingOutput>> {
        if request.streaming {
            request.streaming_tx.clone()
        } else {
            None
        }
    }

    pub(super) fn stream_text(
        tx: &mpsc::Sender<StreamingOutput>,
        request_id: &str,
        sequence: &mut usize,
        text: String,
    ) -> Result<()> {
        Self::stream_text_with_policy(
            tx,
            StreamBackpressurePolicy::FailOnFull,
            request_id,
            sequence,
            text,
        )
    }

    pub(super) fn stream_text_with_policy(
        tx: &mpsc::Sender<StreamingOutput>,
        policy: StreamBackpressurePolicy,
        request_id: &str,
        sequence: &mut usize,
        text: String,
    ) -> Result<()> {
        StreamSink::with_policy(tx, policy).send(StreamingOutput {
            request_id: request_id.to_string(),
            sequence: *sequence,
            samples: Vec::new(),
            sample_rate: 0,
            is_final: false,
            text: Some(text),
            stats: None,
            asr_progress: None,
        })?;
        *sequence += 1;
        Ok(())
    }

    pub(super) fn stream_text_per_character(
        tx: &mpsc::Sender<StreamingOutput>,
        request_id: &str,
        sequence: &mut usize,
        text: &str,
    ) -> Result<()> {
        Self::stream_text_per_character_with_policy(
            tx,
            StreamBackpressurePolicy::FailOnFull,
            request_id,
            sequence,
            text,
        )
    }

    pub(super) fn stream_text_per_character_with_policy(
        tx: &mpsc::Sender<StreamingOutput>,
        policy: StreamBackpressurePolicy,
        request_id: &str,
        sequence: &mut usize,
        text: &str,
    ) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }

        for ch in text.chars() {
            Self::stream_text_with_policy(tx, policy, request_id, sequence, ch.to_string())?;
        }
        Ok(())
    }

    pub(super) fn stream_audio(
        tx: &mpsc::Sender<StreamingOutput>,
        request_id: &str,
        sequence: &mut usize,
        samples: Vec<f32>,
        sample_rate: u32,
        is_final: bool,
    ) -> Result<()> {
        Self::stream_audio_with_policy(
            tx,
            StreamBackpressurePolicy::FailOnFull,
            request_id,
            sequence,
            samples,
            sample_rate,
            is_final,
        )
    }

    pub(super) fn stream_audio_with_policy(
        tx: &mpsc::Sender<StreamingOutput>,
        policy: StreamBackpressurePolicy,
        request_id: &str,
        sequence: &mut usize,
        samples: Vec<f32>,
        sample_rate: u32,
        is_final: bool,
    ) -> Result<()> {
        StreamSink::with_policy(tx, policy).send(StreamingOutput {
            request_id: request_id.to_string(),
            sequence: *sequence,
            samples,
            sample_rate,
            is_final,
            text: None,
            stats: None,
            asr_progress: None,
        })?;
        *sequence += 1;
        Ok(())
    }

    pub(super) fn stream_asr_progress_with_policy(
        tx: &mpsc::Sender<StreamingOutput>,
        policy: StreamBackpressurePolicy,
        request_id: &str,
        sequence: &mut usize,
        progress: AsrProgress,
    ) -> Result<()> {
        StreamSink::with_policy(tx, policy).send(StreamingOutput {
            request_id: request_id.to_string(),
            sequence: *sequence,
            samples: Vec::new(),
            sample_rate: 0,
            is_final: false,
            text: None,
            stats: None,
            asr_progress: Some(progress),
        })?;
        *sequence += 1;
        Ok(())
    }

    pub(super) fn stream_final_marker(
        tx: &mpsc::Sender<StreamingOutput>,
        request_id: &str,
        sequence: &mut usize,
    ) -> Result<()> {
        Self::stream_final_marker_with_policy(
            tx,
            StreamBackpressurePolicy::FailOnFull,
            request_id,
            sequence,
        )
    }

    pub(super) fn stream_final_marker_with_policy(
        tx: &mpsc::Sender<StreamingOutput>,
        policy: StreamBackpressurePolicy,
        request_id: &str,
        sequence: &mut usize,
    ) -> Result<()> {
        Self::stream_audio_with_policy(tx, policy, request_id, sequence, Vec::new(), 0, true)
    }
}

fn stream_send_error(err: mpsc::error::TrySendError<StreamingOutput>) -> Error {
    match err {
        mpsc::error::TrySendError::Closed(_) => {
            Error::InferenceError("Streaming output channel closed".to_string())
        }
        mpsc::error::TrySendError::Full(_) => {
            record_engine_stream_backpressure();
            Error::InferenceError(
                "Streaming output backpressure exceeded queue capacity".to_string(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use tokio::sync::mpsc;

    use crate::engine::executor::NativeExecutor;
    use crate::engine::output::StreamingOutput;
    use crate::error::Error;

    use super::{StreamBackpressurePolicy, StreamSink};

    #[test]
    fn stream_text_per_character_emits_one_delta_per_character() {
        let (tx, mut rx) = mpsc::channel(8);
        let mut sequence = 0usize;

        NativeExecutor::stream_text_per_character(&tx, "req-1", &mut sequence, "abé")
            .expect("stream should succeed");

        assert_eq!(sequence, 3);

        let first = rx.try_recv().expect("missing first chunk");
        assert_eq!(first.sequence, 0);
        assert_eq!(first.text.as_deref(), Some("a"));

        let second = rx.try_recv().expect("missing second chunk");
        assert_eq!(second.sequence, 1);
        assert_eq!(second.text.as_deref(), Some("b"));

        let third = rx.try_recv().expect("missing third chunk");
        assert_eq!(third.sequence, 2);
        assert_eq!(third.text.as_deref(), Some("é"));

        assert!(rx.try_recv().is_err());
    }

    #[test]
    fn stream_sink_fail_on_full_preserves_current_backpressure_behavior() {
        let (tx, _rx) = mpsc::channel(1);
        let sink = StreamSink::fail_on_full(&tx);
        assert_eq!(sink.policy(), StreamBackpressurePolicy::FailOnFull);

        sink.send(StreamingOutput {
            request_id: "req-1".to_string(),
            sequence: 0,
            samples: vec![0.0],
            sample_rate: 24_000,
            is_final: false,
            text: None,
            stats: None,
            asr_progress: None,
        })
        .expect("first chunk should fit");

        let err = sink
            .send(StreamingOutput {
                request_id: "req-1".to_string(),
                sequence: 1,
                samples: vec![0.0],
                sample_rate: 24_000,
                is_final: false,
                text: None,
                stats: None,
                asr_progress: None,
            })
            .expect_err("full queue should fail with default policy");

        let Error::InferenceError(message) = err else {
            panic!("expected inference error for stream backpressure");
        };
        assert!(message.contains("backpressure"));
    }

    #[test]
    fn stream_sink_lossy_policies_drop_when_queue_is_full() {
        let (tx, _rx) = mpsc::channel(1);
        let sink = StreamSink::with_policy(&tx, StreamBackpressurePolicy::Coalesce);

        sink.send(StreamingOutput {
            request_id: "req-1".to_string(),
            sequence: 0,
            samples: vec![0.0],
            sample_rate: 24_000,
            is_final: false,
            text: None,
            stats: None,
            asr_progress: None,
        })
        .expect("first chunk should fit");

        sink.send(StreamingOutput {
            request_id: "req-1".to_string(),
            sequence: 1,
            samples: vec![0.0],
            sample_rate: 24_000,
            is_final: false,
            text: None,
            stats: None,
            asr_progress: None,
        })
        .expect("lossy policy should drop full-queue chunk");
    }
}
