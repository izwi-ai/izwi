use tokio::sync::mpsc;

use crate::error::{Error, Result};

use super::super::output::StreamingOutput;
use super::super::request::EngineCoreRequest;
use super::NativeExecutor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum StreamBackpressurePolicy {
    FailOnFull,
    BlockWithDeadline,
    DropOldest,
    Coalesce,
    Sample,
}

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
        Self {
            tx,
            policy,
        }
    }

    fn policy(&self) -> StreamBackpressurePolicy {
        self.policy
    }

    fn send(&self, output: StreamingOutput) -> Result<()> {
        match self.policy {
            StreamBackpressurePolicy::FailOnFull => {
                self.tx.try_send(output).map_err(stream_send_error)
            }
            StreamBackpressurePolicy::BlockWithDeadline
            | StreamBackpressurePolicy::DropOldest
            | StreamBackpressurePolicy::Coalesce
            | StreamBackpressurePolicy::Sample => Err(Error::InferenceError(format!(
                "Streaming backpressure policy {:?} is not enabled for engine execution",
                self.policy
            ))),
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
        StreamSink::fail_on_full(tx).send(StreamingOutput {
            request_id: request_id.to_string(),
            sequence: *sequence,
            samples: Vec::new(),
            sample_rate: 0,
            is_final: false,
            text: Some(text),
            stats: None,
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
        if text.is_empty() {
            return Ok(());
        }

        for ch in text.chars() {
            Self::stream_text(tx, request_id, sequence, ch.to_string())?;
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
        StreamSink::fail_on_full(tx).send(StreamingOutput {
            request_id: request_id.to_string(),
            sequence: *sequence,
            samples,
            sample_rate,
            is_final,
            text: None,
            stats: None,
        })?;
        *sequence += 1;
        Ok(())
    }

    pub(super) fn stream_final_marker(
        tx: &mpsc::Sender<StreamingOutput>,
        request_id: &str,
        sequence: &mut usize,
    ) -> Result<()> {
        Self::stream_audio(tx, request_id, sequence, Vec::new(), 0, true)
    }
}

fn stream_send_error(err: mpsc::error::TrySendError<StreamingOutput>) -> Error {
    match err {
        mpsc::error::TrySendError::Closed(_) => {
            Error::InferenceError("Streaming output channel closed".to_string())
        }
        mpsc::error::TrySendError::Full(_) => Error::InferenceError(
            "Streaming output backpressure exceeded queue capacity".to_string(),
        ),
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
            })
            .expect_err("full queue should fail with default policy");

        let Error::InferenceError(message) = err else {
            panic!("expected inference error for stream backpressure");
        };
        assert!(message.contains("backpressure"));
    }

    #[test]
    fn stream_sink_reserved_policies_fail_closed_until_rollout() {
        let (tx, _rx) = mpsc::channel(1);
        let sink = StreamSink::with_policy(&tx, StreamBackpressurePolicy::Coalesce);

        let err = sink
            .send(StreamingOutput {
                request_id: "req-1".to_string(),
                sequence: 0,
                samples: vec![0.0],
                sample_rate: 24_000,
                is_final: false,
                text: None,
                stats: None,
            })
            .expect_err("reserved policy should not silently change stream behavior");

        let Error::InferenceError(message) = err else {
            panic!("expected inference error for reserved policy");
        };
        assert!(message.contains("not enabled"));
    }
}
