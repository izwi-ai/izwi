use std::time::Duration;
use tokio::sync::mpsc;

use crate::error::{Error, Result};

use super::super::metrics::record_engine_stream_backpressure;
use super::super::output::{AsrProgress, StreamingOutput};
use super::super::request::{
    EngineCoreRequest, EngineStreamPolicy, FencedStreamProgress, StreamProgressPermit,
    StreamPushOutcome, StreamStagingBuffer,
};
use super::super::SessionKey;
use super::NativeExecutor;

pub(super) type StreamBackpressurePolicy = EngineStreamPolicy;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamDeliveryFailureKind {
    Delivery,
    Deadline,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StreamDeliveryFailure {
    pub(crate) session: SessionKey,
    pub(crate) kind: StreamDeliveryFailureKind,
}

/// Stream events whose producing model state has already committed. Delivery
/// owns a clone of the exact request channel so terminal request cleanup cannot
/// close the channel before the outbox is flushed.
#[derive(Debug)]
pub(crate) struct CommittedStreamDelivery {
    pub(crate) session: SessionKey,
    tx: mpsc::Sender<StreamingOutput>,
    policy: StreamBackpressurePolicy,
    outputs: Vec<CommittedStreamOutput>,
}

#[derive(Debug)]
struct CommittedStreamOutput {
    output: StreamingOutput,
    _progress_permit: Option<StreamProgressPermit>,
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
            outputs: outputs
                .into_iter()
                .map(|output| CommittedStreamOutput {
                    output,
                    _progress_permit: None,
                })
                .collect(),
        }
    }

    pub(crate) fn from_progress(
        session: SessionKey,
        tx: mpsc::Sender<StreamingOutput>,
        policy: StreamBackpressurePolicy,
        progress: FencedStreamProgress,
    ) -> Self {
        Self {
            session,
            tx,
            policy,
            outputs: vec![CommittedStreamOutput {
                output: progress.output,
                _progress_permit: Some(progress.budget_permit),
            }],
        }
    }

    async fn deliver(self) -> std::result::Result<(), StreamDeliveryFailureKind> {
        for committed in self.outputs {
            let output = committed.output;
            if output.request_id != self.session.request_id {
                return Err(StreamDeliveryFailureKind::Delivery);
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
) -> Vec<StreamDeliveryFailure> {
    let mut failed = Vec::new();
    for delivery in deliveries {
        let session = delivery.session.clone();
        if let Err(kind) = delivery.deliver().await {
            failed.push(StreamDeliveryFailure { session, kind });
        }
    }
    failed
}

async fn send_committed_output(
    tx: &mpsc::Sender<StreamingOutput>,
    policy: StreamBackpressurePolicy,
    output: StreamingOutput,
) -> std::result::Result<(), StreamDeliveryFailureKind> {
    match policy {
        StreamBackpressurePolicy::FailOnFull => match tx.try_send(output) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Closed(_)) => Err(StreamDeliveryFailureKind::Delivery),
            Err(mpsc::error::TrySendError::Full(_)) => {
                record_engine_stream_backpressure();
                Err(StreamDeliveryFailureKind::Delivery)
            }
        },
        StreamBackpressurePolicy::BlockWithDeadline { timeout_ms } => {
            match tokio::time::timeout(Duration::from_millis(timeout_ms.max(1)), tx.send(output))
                .await
            {
                Ok(Ok(())) => Ok(()),
                Ok(Err(_)) => Err(StreamDeliveryFailureKind::Delivery),
                Err(_) => {
                    record_engine_stream_backpressure();
                    Err(StreamDeliveryFailureKind::Deadline)
                }
            }
        }
        StreamBackpressurePolicy::DropNewest => match tx.try_send(output) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Closed(_)) => Err(StreamDeliveryFailureKind::Delivery),
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
        let outcome = tx.push_with_policy(
            StreamingOutput {
                request_id: request_id.to_string(),
                sequence: *sequence,
                samples: Vec::new(),
                sample_rate: 0,
                is_final: false,
                text: Some(text),
                stats: None,
                asr_progress: None,
            },
            policy,
        )?;
        if outcome == StreamPushOutcome::Accepted {
            *sequence += 1;
        }
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
        let outcome = tx.push_with_policy(
            StreamingOutput {
                request_id: request_id.to_string(),
                sequence: *sequence,
                samples,
                sample_rate,
                is_final,
                text: None,
                stats: None,
                asr_progress: None,
            },
            policy,
        )?;
        if outcome == StreamPushOutcome::Accepted {
            *sequence += 1;
        }
        Ok(())
    }

    pub(super) fn stream_asr_progress_with_policy(
        tx: &StreamStagingBuffer,
        policy: StreamBackpressurePolicy,
        request_id: &str,
        sequence: &mut usize,
        progress: AsrProgress,
    ) -> Result<()> {
        let outcome = tx.push_with_policy(
            StreamingOutput {
                request_id: request_id.to_string(),
                sequence: *sequence,
                samples: Vec::new(),
                sample_rate: 0,
                is_final: false,
                text: None,
                stats: None,
                asr_progress: Some(progress),
            },
            policy,
        )?;
        if outcome == StreamPushOutcome::Accepted {
            *sequence += 1;
        }
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

    use crate::backends::BackendKind;
    use crate::engine::executor::NativeExecutor;
    use crate::engine::output::StreamingOutput;
    use crate::engine::request::{
        StreamProgressBudget, StreamStagingBuffer, STREAM_PROGRESS_MAX_BUFFERED_BYTES,
    };
    use crate::engine::{
        AdapterAbiRevision, AdapterInstanceId, BatchId, BatchLaneKey, ExecutionGroupId,
        ModelInstanceId, OutputVisibility, SessionKey, StageId,
    };

    use super::{
        deliver_committed_streams, CommittedStreamDelivery, StreamBackpressurePolicy,
        StreamDeliveryFailure, StreamDeliveryFailureKind,
    };

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

    fn lane() -> BatchLaneKey {
        BatchLaneKey {
            execution_group: ExecutionGroupId::new(1),
            model_instance: ModelInstanceId::new(2),
            adapter_instance: AdapterInstanceId::new(3),
            adapter_abi: AdapterAbiRevision::new(7),
            capability_id: "chat".to_string(),
            stage_id: StageId::new(4),
            backend: BackendKind::Cpu,
            device_ordinal: None,
            compute_dtype: "f32".to_string(),
            state_dtype: "none".to_string(),
            tensor_layout: "exact".to_string(),
            quantization: "none".to_string(),
            state_schema: "none".to_string(),
            kernel_mode: "compatibility".to_string(),
            semantic_mode: "chat".to_string(),
            shape_bucket: "exact.1".to_string(),
        }
    }

    #[test]
    fn incremental_binding_routes_nonterminal_output_and_stages_final() {
        let staging = StreamStagingBuffer::default();
        let (progress_tx, mut progress_rx) = mpsc::channel(4);
        let session = SessionKey::new("req-1".to_string(), 7);
        let guard = staging
            .bind_quantum(
                BatchId::new(9),
                lane(),
                11,
                session.clone(),
                OutputVisibility::IncrementalCommitted,
                progress_tx,
                StreamProgressBudget::new(STREAM_PROGRESS_MAX_BUFFERED_BYTES),
            )
            .expect("incremental binding");
        let mut sequence = 0usize;

        NativeExecutor::stream_text(&staging, "req-1", &mut sequence, "hello".to_string())
            .expect("progress output");
        NativeExecutor::stream_final_marker(&staging, "req-1", &mut sequence)
            .expect("final marker");

        let progress = progress_rx.try_recv().expect("incremental progress");
        assert_eq!(progress.batch_id, BatchId::new(9));
        assert_eq!(progress.plan_id, 11);
        assert_eq!(progress.session, session);
        assert_eq!(progress.output.sequence, 0);
        assert_eq!(progress.output.text.as_deref(), Some("hello"));
        assert!(!progress.output.is_final);
        assert!(progress_rx.try_recv().is_err());

        let staged = staging.take().expect("staged final");
        assert_eq!(staged.len(), 1);
        assert_eq!(staged[0].sequence, 1);
        assert!(staged[0].is_final);

        drop(guard);
        NativeExecutor::stream_text(&staging, "req-1", &mut sequence, "later".to_string())
            .expect("post-binding staged output");
        assert_eq!(
            staging.take().expect("post-binding staging")[0]
                .text
                .as_deref(),
            Some("later")
        );
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

        assert_eq!(
            failed,
            vec![StreamDeliveryFailure {
                session: first,
                kind: StreamDeliveryFailureKind::Delivery,
            }]
        );
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
        assert_eq!(
            failed,
            vec![StreamDeliveryFailure {
                session,
                kind: StreamDeliveryFailureKind::Deadline,
            }]
        );
    }
}
