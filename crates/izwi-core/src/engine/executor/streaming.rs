use std::time::Duration;
use tokio::sync::mpsc;

use crate::error::{Error, Result};

use super::super::metrics::record_engine_stream_backpressure;
use super::super::output::{AsrProgress, StreamingOutput};
use super::super::request::{EngineCoreRequest, EngineStreamPolicy, StreamStagingBuffer};
use super::super::SessionKey;
use super::NativeExecutor;

pub(super) type StreamBackpressurePolicy = EngineStreamPolicy;

/// Stream events whose producing model state has already committed. Delivery
/// owns a clone of the exact request channel so terminal request cleanup cannot
/// close the channel before the outbox is flushed.
#[derive(Debug)]
pub(crate) struct CommittedStreamDelivery {
    pub(crate) session: SessionKey,
    tx: mpsc::Sender<StreamingOutput>,
    policy: StreamBackpressurePolicy,
    outputs: Vec<StreamingOutput>,
}

impl CommittedStreamDelivery {
    pub(crate) fn new(
        session: SessionKey,
        tx: mpsc::Sender<StreamingOutput>,
        policy: StreamBackpressurePolicy,
        outputs: Vec<StreamingOutput>,
    ) -> Self {
        Self {
            session,
            tx,
            policy,
            outputs,
        }
    }

    async fn deliver(self) -> Result<()> {
        for output in self.outputs {
            if output.request_id != self.session.request_id {
                return Err(Error::InferenceError(
                    "committed stream output does not match its exact session".to_string(),
                ));
            }
            send_committed_output(&self.tx, self.policy, output).await?;
        }
        Ok(())
    }
}

/// Deliver every committed row independently. A failed consumer is returned
/// for exact-session cancellation after all peer rows have had a chance to
/// flush their own outboxes.
pub(crate) async fn deliver_committed_streams(
    deliveries: Vec<CommittedStreamDelivery>,
) -> Vec<SessionKey> {
    let mut failed = Vec::new();
    for delivery in deliveries {
        let session = delivery.session.clone();
        if delivery.deliver().await.is_err() {
            failed.push(session);
        }
    }
    failed
}

async fn send_committed_output(
    tx: &mpsc::Sender<StreamingOutput>,
    policy: StreamBackpressurePolicy,
    output: StreamingOutput,
) -> Result<()> {
    match policy {
        StreamBackpressurePolicy::FailOnFull => tx.try_send(output).map_err(stream_send_error),
        StreamBackpressurePolicy::BlockWithDeadline { timeout_ms } => {
            match tokio::time::timeout(Duration::from_millis(timeout_ms.max(1)), tx.send(output))
                .await
            {
                Ok(Ok(())) => Ok(()),
                Ok(Err(_)) => Err(Error::InferenceError(
                    "Streaming output channel closed".to_string(),
                )),
                Err(_) => {
                    record_engine_stream_backpressure();
                    Err(Error::InferenceError(
                        "Streaming output backpressure deadline elapsed".to_string(),
                    ))
                }
            }
        }
        StreamBackpressurePolicy::DropNewest => match tx.try_send(output) {
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

impl NativeExecutor {
    pub(super) fn stream_sender(request: &EngineCoreRequest) -> Option<StreamStagingBuffer> {
        if request.streaming {
            Some(request.stream_staging_buffer())
        } else {
            None
        }
    }

    pub(super) fn stream_text(
        tx: &StreamStagingBuffer,
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
        tx: &StreamStagingBuffer,
        policy: StreamBackpressurePolicy,
        request_id: &str,
        sequence: &mut usize,
        text: String,
    ) -> Result<()> {
        let _ = policy;
        tx.push(StreamingOutput {
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
        tx: &StreamStagingBuffer,
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
        tx: &StreamStagingBuffer,
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
        tx: &StreamStagingBuffer,
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
        tx: &StreamStagingBuffer,
        policy: StreamBackpressurePolicy,
        request_id: &str,
        sequence: &mut usize,
        samples: Vec<f32>,
        sample_rate: u32,
        is_final: bool,
    ) -> Result<()> {
        let _ = policy;
        tx.push(StreamingOutput {
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
        tx: &StreamStagingBuffer,
        policy: StreamBackpressurePolicy,
        request_id: &str,
        sequence: &mut usize,
        progress: AsrProgress,
    ) -> Result<()> {
        let _ = policy;
        tx.push(StreamingOutput {
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
        tx: &StreamStagingBuffer,
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
        tx: &StreamStagingBuffer,
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
    use crate::engine::request::StreamStagingBuffer;
    use crate::engine::SessionKey;

    use super::{deliver_committed_streams, CommittedStreamDelivery, StreamBackpressurePolicy};

    fn output(request_id: &str, sequence: usize) -> StreamingOutput {
        StreamingOutput {
            request_id: request_id.to_string(),
            sequence,
            samples: vec![0.0],
            sample_rate: 24_000,
            is_final: false,
            text: None,
            stats: None,
            asr_progress: None,
        }
    }

    #[tokio::test]
    async fn staged_text_is_invisible_until_committed_delivery() {
        let staging = StreamStagingBuffer::default();
        let (tx, mut rx) = mpsc::channel(8);
        let mut sequence = 0usize;

        NativeExecutor::stream_text_per_character(&staging, "req-1", &mut sequence, "abé")
            .expect("stream should succeed");

        assert_eq!(sequence, 3);
        assert!(
            rx.try_recv().is_err(),
            "staged output escaped before commit"
        );

        let session = SessionKey::new("req-1".to_string(), 7);
        let failed = deliver_committed_streams(vec![CommittedStreamDelivery::new(
            session,
            tx,
            StreamBackpressurePolicy::FailOnFull,
            staging.take().expect("staged events"),
        )])
        .await;
        assert!(failed.is_empty());

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

    #[tokio::test]
    async fn failed_row_backpressure_does_not_skip_peer_delivery() {
        let first = SessionKey::new("full".to_string(), 1);
        let second = SessionKey::new("open".to_string(), 2);
        let (full_tx, _full_rx) = mpsc::channel(1);
        full_tx.try_send(output("full", 0)).expect("prefill queue");
        let (open_tx, mut open_rx) = mpsc::channel(1);

        let failed = deliver_committed_streams(vec![
            CommittedStreamDelivery::new(
                first.clone(),
                full_tx,
                StreamBackpressurePolicy::FailOnFull,
                vec![output("full", 1)],
            ),
            CommittedStreamDelivery::new(
                second,
                open_tx,
                StreamBackpressurePolicy::FailOnFull,
                vec![output("open", 0)],
            ),
        ])
        .await;

        assert_eq!(failed, vec![first]);
        assert_eq!(open_rx.try_recv().expect("peer output").request_id, "open");
    }

    #[tokio::test]
    async fn committed_blocking_delivery_uses_async_deadline() {
        let session = SessionKey::new("req-1".to_string(), 1);
        let (tx, _rx) = mpsc::channel(1);
        tx.try_send(output("req-1", 0)).expect("prefill queue");
        let started = std::time::Instant::now();
        let failed = deliver_committed_streams(vec![CommittedStreamDelivery::new(
            session.clone(),
            tx,
            StreamBackpressurePolicy::BlockWithDeadline { timeout_ms: 5 },
            vec![output("req-1", 1)],
        )])
        .await;

        assert!(started.elapsed() >= std::time::Duration::from_millis(5));
        assert_eq!(failed, vec![session]);
    }
}
