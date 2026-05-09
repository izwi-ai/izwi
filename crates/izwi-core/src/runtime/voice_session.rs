//! Reusable voice-session state machine primitives.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VoiceSessionPhase {
    Idle,
    Listening,
    Processing,
    Speaking,
    Interrupted,
    Closed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoiceSession {
    session_id: Option<String>,
    phase: VoiceSessionPhase,
    active_turn_id: Option<String>,
    last_interrupt_reason: Option<String>,
}

impl Default for VoiceSession {
    fn default() -> Self {
        Self {
            session_id: None,
            phase: VoiceSessionPhase::Idle,
            active_turn_id: None,
            last_interrupt_reason: None,
        }
    }
}

impl VoiceSession {
    pub fn start(&mut self, session_id: impl Into<String>) {
        self.session_id = Some(session_id.into());
        self.phase = VoiceSessionPhase::Idle;
        self.active_turn_id = None;
        self.last_interrupt_reason = None;
    }

    pub fn begin_listening(&mut self) {
        if self.phase != VoiceSessionPhase::Closed {
            self.phase = VoiceSessionPhase::Listening;
            self.active_turn_id = None;
        }
    }

    pub fn begin_processing(&mut self, turn_id: impl Into<String>) {
        if self.phase != VoiceSessionPhase::Closed {
            self.phase = VoiceSessionPhase::Processing;
            self.active_turn_id = Some(turn_id.into());
            self.last_interrupt_reason = None;
        }
    }

    pub fn begin_speaking(&mut self, turn_id: impl Into<String>) {
        if self.phase != VoiceSessionPhase::Closed {
            self.phase = VoiceSessionPhase::Speaking;
            self.active_turn_id = Some(turn_id.into());
            self.last_interrupt_reason = None;
        }
    }

    pub fn interrupt(&mut self, reason: impl Into<String>) {
        if self.phase != VoiceSessionPhase::Closed {
            self.phase = VoiceSessionPhase::Interrupted;
            self.active_turn_id = None;
            self.last_interrupt_reason = Some(reason.into());
        }
    }

    pub fn close(&mut self) {
        self.phase = VoiceSessionPhase::Closed;
        self.active_turn_id = None;
    }

    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    pub fn phase(&self) -> VoiceSessionPhase {
        self.phase
    }

    pub fn active_turn_id(&self) -> Option<&str> {
        self.active_turn_id.as_deref()
    }

    pub fn last_interrupt_reason(&self) -> Option<&str> {
        self.last_interrupt_reason.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voice_session_tracks_turn_lifecycle() {
        let mut session = VoiceSession::default();

        session.start("session-1");
        session.begin_listening();
        session.begin_processing("turn-1");
        session.begin_speaking("turn-1");

        assert_eq!(session.session_id(), Some("session-1"));
        assert_eq!(session.phase(), VoiceSessionPhase::Speaking);
        assert_eq!(session.active_turn_id(), Some("turn-1"));
    }

    #[test]
    fn voice_session_interrupt_clears_active_turn_until_close() {
        let mut session = VoiceSession::default();

        session.start("session-1");
        session.begin_processing("turn-1");
        session.interrupt("barge_in");

        assert_eq!(session.phase(), VoiceSessionPhase::Interrupted);
        assert_eq!(session.active_turn_id(), None);
        assert_eq!(session.last_interrupt_reason(), Some("barge_in"));

        session.close();
        session.begin_listening();
        assert_eq!(session.phase(), VoiceSessionPhase::Closed);
    }
}
