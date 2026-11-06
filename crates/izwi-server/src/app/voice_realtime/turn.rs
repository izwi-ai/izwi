//! A turn's control (cancellation, playback clock, the reply heard so far)
//! and the connection's spoken conversation held until stored.

use super::*;

/// The connection's spoken conversation: the latest assistant replies,
/// held until what the user heard of them is known, then stored in order
/// (SPK-F13). Replies are stored when the next turn needs the history, or
/// when the stream or socket ends; until then the client may truncate them.
#[derive(Default)]
pub(crate) struct VoiceConversation {
    pub(crate) pending: Mutex<VecDeque<(u64, TurnControl)>>,
}

impl VoiceConversation {
    pub(crate) fn lock(&self) -> std::sync::MutexGuard<'_, VecDeque<(u64, TurnControl)>> {
        self.pending
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// Hold `utterance_seq`'s reply until it is stored.
    pub(crate) fn hold(&self, utterance_seq: u64, control: TurnControl) {
        let mut pending = self.lock();
        pending.retain(|(seq, _)| *seq != utterance_seq);
        pending.push_back((utterance_seq, control));
    }

    /// The held replies, oldest first.
    pub(crate) fn take(&self) -> Vec<TurnControl> {
        self.lock().drain(..).map(|(_, control)| control).collect()
    }

    /// The held reply of `utterance_seq`.
    pub(crate) fn reply_of(&self, utterance_seq: u64) -> Option<TurnControl> {
        self.lock()
            .iter()
            .find(|(seq, _)| *seq == utterance_seq)
            .map(|(_, control)| control.clone())
    }

    /// The user interrupted playback: the latest held reply was heard up to
    /// where the client's playback had got.
    pub(crate) fn interrupt_playback(&self) {
        if let Some((_, control)) = self.lock().back() {
            control.stop_playback();
        }
    }

    /// Store the held replies in order.
    pub(crate) async fn store(&self, state: &AppState) {
        for control in self.take() {
            store_reply(state, &control).await;
        }
    }

    /// Store the held replies in the background.
    pub(crate) fn store_in_background(self: &Arc<Self>, state: &AppState) {
        let conversation = self.clone();
        let state = state.clone();
        tokio::spawn(async move { conversation.store(&state).await });
    }
}

#[derive(Clone)]
pub(crate) struct TurnControl {
    pub(crate) inner: Arc<TurnControlInner>,
}

impl std::fmt::Debug for TurnControl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TurnControl")
            .field("cancelled", &self.is_cancelled())
            .field("speaking", &self.is_speaking())
            .finish()
    }
}

pub(crate) struct TurnControlInner {
    pub(crate) cancelled: AtomicBool,
    pub(crate) reason: Mutex<Option<String>>,
    pub(crate) notify: Notify,
    pub(crate) endpointed_at: Instant,
    /// Set once the turn has sent assistant audio.
    pub(crate) speaking: AtomicBool,
    pub(crate) playback: Mutex<PlaybackClock>,
    pub(crate) reply: Mutex<ReplyRecord>,
}

/// What the turn's assistant reply said, and how much of it was heard.
#[derive(Default)]
pub(crate) struct ReplyRecord {
    /// The agent session (thread) the reply belongs to.
    pub(crate) session: Option<AgentSession>,
    pub(crate) turn_record_id: Option<String>,
    /// The whole reply as generated, and without its reasoning.
    pub(crate) raw_text: Option<String>,
    pub(crate) text: Option<String>,
    pub(crate) meta: MemoryMessageMeta,
    pub(crate) ledger: SpokenLedger,
    /// How much of the reply's audio the user heard, when not all of it.
    pub(crate) heard_ms: Option<f64>,
}

impl ReplyRecord {
    /// The text to remember: all of it, or what was heard.
    pub(crate) fn remembered_text(&self) -> Option<String> {
        match self.heard_ms {
            Some(heard_ms) => Some(self.ledger.heard_text(heard_ms)),
            None => self.text.clone(),
        }
    }
}

impl TurnControl {
    pub(crate) fn new(endpointed_at: Instant) -> Self {
        Self {
            inner: Arc::new(TurnControlInner {
                cancelled: AtomicBool::new(false),
                reason: Mutex::new(None),
                notify: Notify::new(),
                endpointed_at,
                speaking: AtomicBool::new(false),
                playback: Mutex::new(PlaybackClock::default()),
                reply: Mutex::new(ReplyRecord::default()),
            }),
        }
    }

    pub(crate) fn reply(&self) -> std::sync::MutexGuard<'_, ReplyRecord> {
        self.inner
            .reply
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    /// The client stopped playing the reply now: it was heard as far as its
    /// playback had got, unless the client already said how far.
    pub(crate) fn stop_playback(&self) {
        let played = self
            .inner
            .playback
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .played(Instant::now());
        let mut reply = self.reply();
        if reply.heard_ms.is_none() && played.as_secs_f64() * 1000.0 < reply.ledger.audio_ms() {
            reply.heard_ms = Some(played.as_secs_f64() * 1000.0);
        }
    }

    /// The client says it played `audio_end_ms` of the reply; returns the
    /// text that covers.
    pub(crate) fn truncate(&self, audio_end_ms: u64) -> String {
        let mut reply = self.reply();
        reply.heard_ms = Some(audio_end_ms as f64);
        reply.ledger.heard_text(audio_end_ms as f64)
    }

    pub(crate) fn endpointed_at(&self) -> Instant {
        self.inner.endpointed_at
    }

    pub(crate) fn is_speaking(&self) -> bool {
        self.inner.speaking.load(Ordering::Acquire)
    }

    /// Account for `duration` of assistant audio just sent: the first audio
    /// is the turn's response latency, and a chunk that found the client's
    /// playback already drained is an underrun.
    pub(crate) fn audio_sent(&self, runtime: &RuntimeService, duration: Duration) {
        let now = Instant::now();
        if !self.inner.speaking.swap(true, Ordering::AcqRel) {
            runtime.record_voice_latency(
                VoiceLatency::TurnFirstAudio,
                now.saturating_duration_since(self.inner.endpointed_at),
            );
        }
        let underrun = self
            .inner
            .playback
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .sent(now, duration);
        if underrun {
            runtime.record_voice_egress_underrun();
        }
    }

    pub(crate) fn cancel(&self, reason: impl Into<String>) {
        let mut stored_reason = self
            .inner
            .reason
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if self.inner.cancelled.load(Ordering::Acquire) {
            return;
        }
        *stored_reason = Some(reason.into());
        self.inner.cancelled.store(true, Ordering::Release);
        drop(stored_reason);
        self.inner.notify.notify_waiters();
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Acquire)
    }

    pub(crate) fn reason(&self) -> Option<String> {
        self.inner
            .reason
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .clone()
    }

    pub(crate) async fn cancelled(&self) {
        loop {
            let notified = self.inner.notify.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

/// The server's view of the client's playback: the audio sent so far finishes
/// playing at `playing_until`, assuming the client plays it as it arrives.
#[derive(Debug, Default)]
pub(crate) struct PlaybackClock {
    pub(crate) playing_until: Option<Instant>,
    pub(crate) sent: Duration,
}

impl PlaybackClock {
    /// Account for `duration` of audio sent at `now`; returns whether the
    /// audio sent before it had already finished playing (an underrun).
    pub(crate) fn sent(&mut self, now: Instant, duration: Duration) -> bool {
        let underrun = self
            .playing_until
            .is_some_and(|until| now > until + PLAYBACK_UNDERRUN_TOLERANCE);
        let starts = self
            .playing_until
            .filter(|until| *until > now)
            .unwrap_or(now);
        self.playing_until = Some(starts + duration);
        self.sent += duration;
        underrun
    }

    /// How much of the audio sent has played by `now`.
    pub(crate) fn played(&self, now: Instant) -> Duration {
        let queued = self
            .playing_until
            .map_or(Duration::ZERO, |until| until.saturating_duration_since(now));
        self.sent.saturating_sub(queued)
    }
}
