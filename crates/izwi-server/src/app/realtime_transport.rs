//! The websocket transport shared by the realtime endpoints (`/voice` and
//! `/transcription`, SPK-F16): one bounded outbound mailbox with delivery
//! classes, the writer task that drains it, the client PCM16 frame layout
//! and the PCM helpers.
//!
//! Each endpoint keeps its own thin sender over the mailbox (event encoding,
//! metrics, its binary magic and close code), so the wire formats are
//! unchanged.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::ws::{CloseFrame, Message, WebSocket};
use futures::stream::SplitSink;
use futures::SinkExt;
use izwi_core::audio::{AudioEncoder, AudioFormat};
use tokio::sync::Notify;

pub(crate) use izwi_gateway::app::realtime_wire::{
    now_unix_millis, parse_client_pcm16_frame, ClientPcm16Frame, WS_BIN_VERSION,
};

/// How long the writer waits for the socket to take one message before it
/// treats the peer as gone.
pub(crate) const WS_WRITER_SEND_TIMEOUT: Duration = Duration::from_secs(5);

/// How the mailbox treats one outbound message when the socket is slow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum OutboundClass {
    /// Must arrive; a full queue closes the session.
    Critical,
    /// A replaceable snapshot: a newer one with the same key (and turn)
    /// replaces the queued one.
    Coalescible(String),
    /// Ordered audio, bounded by queued bytes and duration.
    Audio { bytes: usize, duration_ms: u64 },
    /// Dropped when the queue is full.
    Droppable,
}

pub(crate) struct OutboundItem {
    pub(crate) message: Message,
    pub(crate) class: OutboundClass,
    /// The turn the message belongs to; [`OutboundMailbox::cutoff_turn`]
    /// drops a cancelled turn's queued and later messages.
    pub(crate) turn_generation: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OutboundLimits {
    pub(crate) max_items: usize,
    pub(crate) max_audio_bytes: usize,
    pub(crate) max_audio_ms: u64,
}

impl OutboundLimits {
    /// Limits for a mailbox that carries no audio.
    pub(crate) fn items(max_items: usize) -> Self {
        Self {
            max_items,
            max_audio_bytes: usize::MAX,
            max_audio_ms: u64::MAX,
        }
    }
}

pub(crate) enum MailboxOutput {
    Message(Message),
    Close(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutboundEnqueueError {
    Cutoff,
    AudioSaturated,
    QueueSaturated,
    Closed,
    DroppedDiagnostic,
}

pub(crate) struct OutboundMailbox {
    limits: OutboundLimits,
    state: Mutex<OutboundMailboxState>,
    notify: Notify,
    /// Wakes audio producers waiting for credit: the writer took a message,
    /// a turn was cut off or the mailbox closed.
    credit: Notify,
    minimum_turn_generation: AtomicU64,
}

#[derive(Default)]
struct OutboundMailboxState {
    queue: VecDeque<OutboundItem>,
    audio_bytes: usize,
    audio_ms: u64,
    close_reason: Option<String>,
    closed: bool,
}

impl OutboundMailbox {
    pub(crate) fn new(limits: OutboundLimits) -> Self {
        Self {
            limits: OutboundLimits {
                max_items: limits.max_items.max(1),
                ..limits
            },
            state: Mutex::new(OutboundMailboxState::default()),
            notify: Notify::new(),
            credit: Notify::new(),
            minimum_turn_generation: AtomicU64::new(0),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, OutboundMailboxState> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn fail_locked(&self, state: &mut OutboundMailboxState, reason: &str) {
        state.close_reason = Some(reason.to_string());
        self.notify.notify_one();
    }

    pub(crate) fn enqueue(&self, item: OutboundItem) -> Result<(), OutboundEnqueueError> {
        let mut state = self.lock();
        if item.turn_generation.is_some_and(|generation| {
            generation < self.minimum_turn_generation.load(Ordering::Acquire)
        }) {
            return Err(OutboundEnqueueError::Cutoff);
        }
        if state.closed {
            return Err(OutboundEnqueueError::Closed);
        }

        if let OutboundClass::Coalescible(key) = &item.class {
            if let Some(existing) = state.queue.iter_mut().find(|queued| {
                queued.turn_generation == item.turn_generation
                    && matches!(&queued.class, OutboundClass::Coalescible(existing_key) if existing_key == key)
            }) {
                existing.message = item.message;
                return Ok(());
            }
        }

        if let OutboundClass::Audio { bytes, duration_ms } = &item.class {
            if state.audio_bytes.saturating_add(*bytes) > self.limits.max_audio_bytes
                || state.audio_ms.saturating_add(*duration_ms) > self.limits.max_audio_ms
            {
                self.fail_locked(&mut state, "outbound_audio_saturated");
                return Err(OutboundEnqueueError::AudioSaturated);
            }
        }

        if state.queue.len() >= self.limits.max_items {
            return match &item.class {
                OutboundClass::Critical => {
                    self.fail_locked(&mut state, "critical_outbound_queue_saturated");
                    Err(OutboundEnqueueError::QueueSaturated)
                }
                OutboundClass::Audio { .. } => {
                    self.fail_locked(&mut state, "outbound_audio_saturated");
                    Err(OutboundEnqueueError::AudioSaturated)
                }
                OutboundClass::Droppable => Err(OutboundEnqueueError::DroppedDiagnostic),
                OutboundClass::Coalescible(_) => Err(OutboundEnqueueError::QueueSaturated),
            };
        }

        if let OutboundClass::Audio { bytes, duration_ms } = &item.class {
            state.audio_bytes = state.audio_bytes.saturating_add(*bytes);
            state.audio_ms = state.audio_ms.saturating_add(*duration_ms);
        }
        state.queue.push_back(item);
        drop(state);
        self.notify.notify_one();
        Ok(())
    }

    /// Queue one audio message once the queue has credit for it: queued
    /// audio plus this message within the byte and duration limits (a
    /// message alone always fits) and a free slot. A slow client therefore
    /// slows the audio producer down instead of closing the session (SPK-F11).
    /// Fails when the turn is cut off or the mailbox closes while waiting.
    pub(crate) async fn enqueue_audio(
        &self,
        item: OutboundItem,
    ) -> Result<(), OutboundEnqueueError> {
        let OutboundClass::Audio { bytes, duration_ms } = item.class else {
            return self.enqueue(item);
        };
        loop {
            let credit = self.credit.notified();
            {
                let mut state = self.lock();
                if item.turn_generation.is_some_and(|generation| {
                    generation < self.minimum_turn_generation.load(Ordering::Acquire)
                }) {
                    return Err(OutboundEnqueueError::Cutoff);
                }
                if state.closed || state.close_reason.is_some() {
                    return Err(OutboundEnqueueError::Closed);
                }
                let audio_queued = state.audio_bytes > 0 || state.audio_ms > 0;
                let fits = !audio_queued
                    || (state.audio_bytes.saturating_add(bytes) <= self.limits.max_audio_bytes
                        && state.audio_ms.saturating_add(duration_ms) <= self.limits.max_audio_ms);
                if fits && state.queue.len() < self.limits.max_items {
                    state.audio_bytes = state.audio_bytes.saturating_add(bytes);
                    state.audio_ms = state.audio_ms.saturating_add(duration_ms);
                    state.queue.push_back(item);
                    drop(state);
                    self.notify.notify_one();
                    return Ok(());
                }
            }
            credit.await;
        }
    }

    /// Drop turn `generation`'s queued messages and refuse its later ones.
    pub(crate) fn cutoff_turn(&self, generation: u64) {
        let mut state = self.lock();
        self.minimum_turn_generation
            .fetch_max(generation.saturating_add(1), Ordering::AcqRel);
        let mut retained = VecDeque::with_capacity(state.queue.len());
        while let Some(item) = state.queue.pop_front() {
            if item.turn_generation == Some(generation) {
                subtract_audio_depth(&mut state, &item.class);
            } else {
                retained.push_back(item);
            }
        }
        state.queue = retained;
        drop(state);
        self.credit.notify_waiters();
    }

    pub(crate) fn accepts_turn_generation(&self, generation: u64) -> bool {
        generation >= self.minimum_turn_generation.load(Ordering::Acquire)
    }

    /// The next message for the writer, a close the mailbox decided on, or
    /// `None` once it is closed and drained.
    pub(crate) async fn next(&self) -> Option<MailboxOutput> {
        loop {
            let notified = self.notify.notified();
            {
                let mut state = self.lock();
                if let Some(reason) = state.close_reason.take() {
                    state.queue.clear();
                    state.audio_bytes = 0;
                    state.audio_ms = 0;
                    state.closed = true;
                    return Some(MailboxOutput::Close(reason));
                }
                if let Some(item) = state.queue.pop_front() {
                    subtract_audio_depth(&mut state, &item.class);
                    drop(state);
                    self.credit.notify_waiters();
                    return Some(MailboxOutput::Message(item.message));
                }
                if state.closed {
                    return None;
                }
            }
            notified.await;
        }
    }

    /// Stop taking messages; the writer drains what is queued.
    pub(crate) fn close(&self) {
        let mut state = self.lock();
        state.closed = true;
        drop(state);
        self.notify.notify_waiters();
        self.credit.notify_waiters();
    }

    /// Close the session with `reason`, dropping what is queued.
    pub(crate) fn fail(&self, reason: &str) {
        let mut state = self.lock();
        if !state.closed {
            state.close_reason = Some(reason.to_string());
        }
        drop(state);
        self.notify.notify_one();
        self.credit.notify_waiters();
    }
}

fn subtract_audio_depth(state: &mut OutboundMailboxState, class: &OutboundClass) {
    if let OutboundClass::Audio { bytes, duration_ms } = class {
        state.audio_bytes = state.audio_bytes.saturating_sub(*bytes);
        state.audio_ms = state.audio_ms.saturating_sub(*duration_ms);
    }
}

/// Drain `mailbox` into the socket until it closes, the peer stops taking
/// messages for [`WS_WRITER_SEND_TIMEOUT`], or the mailbox closes the session
/// (sent as a close frame with `close_code`). The mailbox is closed when the
/// writer stops, so producers waiting for audio credit give up.
pub(crate) fn spawn_writer(
    mut sink: SplitSink<WebSocket, Message>,
    mailbox: Arc<OutboundMailbox>,
    close_code: u16,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(output) = mailbox.next().await {
            let message = match output {
                MailboxOutput::Message(message) => message,
                MailboxOutput::Close(reason) => Message::Close(Some(CloseFrame {
                    code: close_code,
                    reason: reason.into(),
                })),
            };
            let terminal = matches!(message, Message::Close(_));
            let sent = matches!(
                tokio::time::timeout(WS_WRITER_SEND_TIMEOUT, sink.send(message)).await,
                Ok(Ok(()))
            );
            if !sent || terminal {
                break;
            }
        }
        mailbox.close();
    })
}

pub(crate) fn pcm16_bytes_to_i16(bytes: &[u8]) -> Vec<i16> {
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.as_chunks::<2>().0 {
        out.push(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    out
}

pub(crate) fn pcm16_i16_to_f32(samples_i16: &[i16]) -> Vec<f32> {
    samples_i16
        .iter()
        .map(|sample| *sample as f32 / 32768.0)
        .collect()
}

pub(crate) fn wav_bytes_from_pcm16_mono(
    samples_i16: &[i16],
    sample_rate: u32,
) -> Result<Vec<u8>, String> {
    if sample_rate == 0 {
        return Err("Invalid sample rate 0".to_string());
    }
    AudioEncoder::new(sample_rate, 1)
        .encode(&pcm16_i16_to_f32(samples_i16), AudioFormat::Wav)
        .map_err(|err| format!("Failed to encode streamed WAV: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_gateway::app::realtime_wire::WS_BIN_KIND_CLIENT_PCM16;

    fn audio(byte: u8, bytes: usize, duration_ms: u64) -> OutboundItem {
        OutboundItem {
            message: Message::Binary(vec![byte].into()),
            class: OutboundClass::Audio { bytes, duration_ms },
            turn_generation: Some(1),
        }
    }

    #[tokio::test]
    async fn mailbox_coalesces_snapshots_per_key_and_turn() {
        let mailbox = OutboundMailbox::new(OutboundLimits::items(4));
        for text in ["first", "latest"] {
            mailbox
                .enqueue(OutboundItem {
                    message: Message::Text(text.into()),
                    class: OutboundClass::Coalescible("transcript-1".to_string()),
                    turn_generation: Some(1),
                })
                .expect("snapshot should enqueue");
        }
        let Some(MailboxOutput::Message(Message::Text(text))) = mailbox.next().await else {
            panic!("latest text snapshot should be delivered");
        };
        assert_eq!(text.as_str(), "latest");
    }

    #[tokio::test]
    async fn mailbox_preserves_audio_order_and_closes_on_saturation() {
        let ordered = OutboundMailbox::new(OutboundLimits {
            max_items: 4,
            max_audio_bytes: 8,
            max_audio_ms: 100,
        });
        for byte in [1_u8, 2_u8] {
            ordered.enqueue(audio(byte, 1, 10)).expect("audio");
        }
        for expected in [1_u8, 2_u8] {
            let Some(MailboxOutput::Message(Message::Binary(bytes))) = ordered.next().await else {
                panic!("audio should be delivered in order");
            };
            assert_eq!(bytes.as_ref(), &[expected]);
        }

        let saturated = OutboundMailbox::new(OutboundLimits {
            max_items: 4,
            max_audio_bytes: 1,
            max_audio_ms: 10,
        });
        saturated.enqueue(audio(1, 1, 10)).expect("first audio");
        assert_eq!(
            saturated.enqueue(audio(2, 1, 1)),
            Err(OutboundEnqueueError::AudioSaturated)
        );
        assert!(matches!(
            saturated.next().await,
            Some(MailboxOutput::Close(reason)) if reason == "outbound_audio_saturated"
        ));
    }

    /// SPK-F11: audio past the queue's limits waits for the writer to take
    /// queued audio instead of closing the session, and arrives in order.
    #[tokio::test]
    async fn audio_waits_for_credit_instead_of_closing() {
        let mailbox = Arc::new(OutboundMailbox::new(OutboundLimits {
            max_items: 8,
            max_audio_bytes: 1_000,
            max_audio_ms: 100,
        }));
        mailbox
            .enqueue_audio(audio(1, 10, 80))
            .await
            .expect("first audio");
        let waiting = tokio::spawn({
            let mailbox = mailbox.clone();
            async move { mailbox.enqueue_audio(audio(2, 10, 40)).await }
        });
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(
            !waiting.is_finished(),
            "audio past the limit waits for credit"
        );

        let Some(MailboxOutput::Message(Message::Binary(first))) = mailbox.next().await else {
            panic!("the first audio is delivered, not a close");
        };
        assert_eq!(first.as_ref(), &[1]);
        tokio::time::timeout(Duration::from_secs(1), waiting)
            .await
            .expect("credit wakes the producer")
            .expect("join")
            .expect("second audio queued");
        let Some(MailboxOutput::Message(Message::Binary(second))) = mailbox.next().await else {
            panic!("the second audio follows in order");
        };
        assert_eq!(second.as_ref(), &[2]);

        // A single message larger than the limits still goes out alone.
        mailbox
            .enqueue_audio(audio(3, 5_000, 500))
            .await
            .expect("an oversized message alone fits");
    }

    /// SPK-F11: cutting off the turn, or the writer stopping, releases a
    /// producer waiting for credit.
    #[tokio::test]
    async fn waiting_audio_is_released_by_cutoff_and_close() {
        for release in ["cutoff", "close"] {
            let mailbox = Arc::new(OutboundMailbox::new(OutboundLimits {
                max_items: 8,
                max_audio_bytes: 1_000,
                max_audio_ms: 100,
            }));
            mailbox
                .enqueue_audio(audio(1, 10, 100))
                .await
                .expect("audio");
            let waiting = tokio::spawn({
                let mailbox = mailbox.clone();
                async move { mailbox.enqueue_audio(audio(2, 10, 10)).await }
            });
            tokio::time::sleep(Duration::from_millis(20)).await;
            let expected = if release == "cutoff" {
                mailbox.cutoff_turn(1);
                OutboundEnqueueError::Cutoff
            } else {
                mailbox.close();
                OutboundEnqueueError::Closed
            };
            let result = tokio::time::timeout(Duration::from_secs(1), waiting)
                .await
                .expect("the waiting producer is released")
                .expect("join");
            assert_eq!(result, Err(expected), "{release}");
        }
    }

    #[tokio::test]
    async fn mailbox_drops_diagnostics_and_cuts_off_cancelled_turns() {
        let mailbox = OutboundMailbox::new(OutboundLimits::items(2));
        let critical = |text: &'static str, generation| OutboundItem {
            message: Message::Text(text.into()),
            class: OutboundClass::Critical,
            turn_generation: Some(generation),
        };
        mailbox.enqueue(critical("cancelled", 7)).expect("enqueue");
        mailbox.enqueue(critical("next", 8)).expect("enqueue");
        assert_eq!(
            mailbox.enqueue(OutboundItem {
                message: Message::Text("diagnostic".into()),
                class: OutboundClass::Droppable,
                turn_generation: None,
            }),
            Err(OutboundEnqueueError::DroppedDiagnostic)
        );

        mailbox.cutoff_turn(7);
        assert_eq!(
            mailbox.enqueue(critical("late", 7)),
            Err(OutboundEnqueueError::Cutoff)
        );
        let Some(MailboxOutput::Message(Message::Text(text))) = mailbox.next().await else {
            panic!("next turn output should remain queued");
        };
        assert_eq!(text.as_str(), "next");
    }

    #[tokio::test]
    async fn mailbox_closes_when_a_critical_message_cannot_queue() {
        let mailbox = OutboundMailbox::new(OutboundLimits::items(1));
        let critical = |text: &'static str| OutboundItem {
            message: Message::Text(text.into()),
            class: OutboundClass::Critical,
            turn_generation: None,
        };
        mailbox.enqueue(critical("one")).expect("enqueue");
        assert_eq!(
            mailbox.enqueue(critical("two")),
            Err(OutboundEnqueueError::QueueSaturated)
        );
        assert!(matches!(
            mailbox.next().await,
            Some(MailboxOutput::Close(reason)) if reason == "critical_outbound_queue_saturated"
        ));
        assert_eq!(
            mailbox.enqueue(critical("three")),
            Err(OutboundEnqueueError::Closed)
        );
    }

    #[test]
    fn client_frames_parse_with_their_endpoint_magic() {
        let mut frame = Vec::from(*b"IVWS");
        frame.extend_from_slice(&[WS_BIN_VERSION, WS_BIN_KIND_CLIENT_PCM16, 0, 0]);
        frame.extend_from_slice(&16_000_u32.to_le_bytes());
        frame.extend_from_slice(&7_u32.to_le_bytes());
        frame.extend_from_slice(&[1, 0, 2, 0]);
        let parsed = parse_client_pcm16_frame(&frame, b"IVWS", "voice").expect("frame");
        assert_eq!(
            (parsed.frame_seq, parsed.sample_rate, parsed.payload),
            (7, 16_000, vec![1, 0, 2, 0])
        );
        assert!(parse_client_pcm16_frame(&frame, b"ITRW", "transcription")
            .unwrap_err()
            .contains("transcription frame header"));
        frame[5] = 9;
        assert!(parse_client_pcm16_frame(&frame, b"IVWS", "voice")
            .unwrap_err()
            .contains("kind 9"));
    }
}
