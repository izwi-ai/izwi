//! The dictation session: what a hotkey press, release or timeout means.
//!
//! Pure and clock-free: every input carries the time in milliseconds, so the
//! rules are tested without threads or sleeps. The controller feeds inputs
//! and carries out the returned [`Action`].

use serde::{Deserialize, Serialize};

/// How the hotkey drives a recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Record while the hotkey is held.
    #[default]
    Hold,
    /// Press to start, press again to finish.
    Toggle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionConfig {
    pub mode: Mode,
    /// In hold mode, two quick presses keep recording until the next press.
    pub double_tap_lock: bool,
    /// A hold released sooner than this is a tap, not a recording.
    pub min_hold_ms: u64,
    /// How long after a tap a second press still counts as a double tap.
    pub double_tap_ms: u64,
    /// A release only counts once the key has stayed up this long. X11 key
    /// auto-repeat sends release and press pairs while the key is held.
    pub release_debounce_ms: u64,
    /// A recording finishes by itself at this length.
    pub max_recording_ms: u64,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            mode: Mode::Hold,
            double_tap_lock: true,
            min_hold_ms: 300,
            double_tap_ms: 300,
            release_debounce_ms: 0,
            max_recording_ms: 5 * 60 * 1000,
        }
    }
}

/// Something that happened, with the time it happened at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Input {
    /// The hotkey went down.
    Press,
    /// The hotkey went up.
    Release,
    /// Start or finish from the tray, the command line or the app window.
    Toggle,
    /// Start hands-free; ignored while recording.
    Start,
    /// Finish the recording; ignored when idle.
    Stop,
    /// Throw the recording away.
    Cancel,
    /// The time passed; fire any deadline that is due.
    Tick,
    /// Transcription and insertion ended (in success or failure).
    Finished,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscardReason {
    /// The hotkey was tapped, not held.
    TooShort,
    Cancelled,
}

/// What the controller must do next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    StartCapture,
    /// Stop capturing, transcribe and insert. Send [`Input::Finished`] after.
    FinishCapture,
    DiscardCapture(DiscardReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Idle,
    /// Recording while the key is down. `released_at` is a release waiting
    /// out the debounce.
    Held {
        started: u64,
        released_at: Option<u64>,
    },
    /// The key was tapped; a second press before `until` locks the recording.
    TapPending {
        started: u64,
        until: u64,
    },
    /// Recording hands-free until the next press.
    Locked {
        started: u64,
    },
    /// Transcribing or inserting; the hotkey does nothing.
    Busy,
}

#[derive(Debug, Clone)]
pub struct Session {
    config: SessionConfig,
    state: State,
}

impl Session {
    pub fn new(config: SessionConfig) -> Self {
        Self {
            config,
            state: State::Idle,
        }
    }

    pub fn config(&self) -> &SessionConfig {
        &self.config
    }

    /// New rules take effect at once; a recording in progress keeps going.
    pub fn set_config(&mut self, config: SessionConfig) {
        self.config = config;
    }

    pub fn is_idle(&self) -> bool {
        self.state == State::Idle
    }

    pub fn is_recording(&self) -> bool {
        matches!(
            self.state,
            State::Held { .. } | State::TapPending { .. } | State::Locked { .. }
        )
    }

    /// True while recording hands-free (the hotkey is not held down).
    pub fn is_locked(&self) -> bool {
        matches!(self.state, State::Locked { .. })
    }

    /// The time at which the controller must send [`Input::Tick`].
    pub fn deadline(&self) -> Option<u64> {
        let max = |started: u64| started.saturating_add(self.config.max_recording_ms);
        match self.state {
            State::Idle | State::Busy => None,
            State::Held {
                started,
                released_at,
            } => Some(match released_at {
                Some(released) => released
                    .saturating_add(self.config.release_debounce_ms)
                    .min(max(started)),
                None => max(started),
            }),
            State::TapPending { started, until } => Some(until.min(max(started))),
            State::Locked { started } => Some(max(started)),
        }
    }

    pub fn handle(&mut self, input: Input, now: u64) -> Option<Action> {
        match (self.state, input) {
            (State::Idle, Input::Press) => {
                self.state = match self.config.mode {
                    Mode::Hold => State::Held {
                        started: now,
                        released_at: None,
                    },
                    Mode::Toggle => State::Locked { started: now },
                };
                Some(Action::StartCapture)
            }
            (State::Idle, Input::Toggle | Input::Start) => {
                self.state = State::Locked { started: now };
                Some(Action::StartCapture)
            }
            (State::Idle, _) => None,

            (State::Busy, Input::Finished) => {
                self.state = State::Idle;
                None
            }
            (State::Busy, _) => None,

            (_, Input::Cancel) => {
                self.state = State::Idle;
                Some(Action::DiscardCapture(DiscardReason::Cancelled))
            }
            (_, Input::Toggle | Input::Stop) => self.finish(),
            // A recording never waits on `Finished`; only `Busy` does.
            (_, Input::Finished) => None,

            (State::Held { started, .. }, Input::Press) => {
                // Auto-repeat: the key never really went up.
                self.state = State::Held {
                    started,
                    released_at: None,
                };
                None
            }
            (
                State::Held {
                    started,
                    released_at,
                },
                Input::Release,
            ) => {
                if self.config.release_debounce_ms == 0 {
                    return self.released(started, now, now);
                }
                self.state = State::Held {
                    started,
                    released_at: released_at.or(Some(now)),
                };
                None
            }
            (
                State::Held {
                    started,
                    released_at,
                },
                Input::Tick,
            ) => {
                if let Some(released) = released_at {
                    if now >= released.saturating_add(self.config.release_debounce_ms) {
                        return self.released(started, released, now);
                    }
                }
                self.finish_at_limit(started, now)
            }

            (State::TapPending { started, until }, Input::Press) => {
                if now <= until {
                    self.state = State::Locked { started };
                }
                None
            }
            (State::TapPending { started, until }, Input::Tick) => {
                if now >= until {
                    self.state = State::Idle;
                    return Some(Action::DiscardCapture(DiscardReason::TooShort));
                }
                self.finish_at_limit(started, now)
            }
            (State::TapPending { .. }, Input::Release | Input::Start) => None,

            (State::Locked { .. }, Input::Press) => self.finish(),
            (State::Locked { started }, Input::Tick) => self.finish_at_limit(started, now),
            (State::Locked { .. }, Input::Release | Input::Start) => None,

            (State::Held { .. }, Input::Start) => None,
        }
    }

    /// The key went up at `released` for good.
    fn released(&mut self, started: u64, released: u64, now: u64) -> Option<Action> {
        if released.saturating_sub(started) >= self.config.min_hold_ms {
            return self.finish();
        }
        if self.config.double_tap_lock {
            self.state = State::TapPending {
                started,
                until: released.saturating_add(self.config.double_tap_ms),
            };
            // The debounce may already have used up the double-tap window.
            return self.handle(Input::Tick, now);
        }
        self.state = State::Idle;
        Some(Action::DiscardCapture(DiscardReason::TooShort))
    }

    fn finish(&mut self) -> Option<Action> {
        self.state = State::Busy;
        Some(Action::FinishCapture)
    }

    fn finish_at_limit(&mut self, started: u64, now: u64) -> Option<Action> {
        if now >= started.saturating_add(self.config.max_recording_ms) {
            return self.finish();
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hold() -> Session {
        Session::new(SessionConfig::default())
    }

    fn toggle() -> Session {
        Session::new(SessionConfig {
            mode: Mode::Toggle,
            ..SessionConfig::default()
        })
    }

    #[test]
    fn a_hold_records_until_release() {
        let mut session = hold();
        assert_eq!(session.handle(Input::Press, 0), Some(Action::StartCapture));
        assert!(session.is_recording());
        assert_eq!(
            session.handle(Input::Release, 1_200),
            Some(Action::FinishCapture)
        );
        assert!(!session.is_recording() && !session.is_idle());
        // The hotkey does nothing while the text is on its way.
        assert_eq!(session.handle(Input::Press, 1_300), None);
        assert_eq!(session.handle(Input::Finished, 1_500), None);
        assert!(session.is_idle());
    }

    #[test]
    fn a_tap_is_discarded_once_the_double_tap_window_closes() {
        let mut session = hold();
        session.handle(Input::Press, 0);
        assert_eq!(session.handle(Input::Release, 120), None);
        assert_eq!(session.deadline(), Some(420));
        assert_eq!(session.handle(Input::Tick, 300), None);
        assert_eq!(
            session.handle(Input::Tick, 420),
            Some(Action::DiscardCapture(DiscardReason::TooShort))
        );
        assert!(session.is_idle());
    }

    #[test]
    fn a_tap_is_discarded_at_once_without_double_tap_lock() {
        let mut session = Session::new(SessionConfig {
            double_tap_lock: false,
            ..SessionConfig::default()
        });
        session.handle(Input::Press, 0);
        assert_eq!(
            session.handle(Input::Release, 120),
            Some(Action::DiscardCapture(DiscardReason::TooShort))
        );
    }

    #[test]
    fn a_double_tap_locks_until_the_next_press() {
        let mut session = hold();
        session.handle(Input::Press, 0);
        session.handle(Input::Release, 100);
        assert_eq!(session.handle(Input::Press, 250), None);
        assert!(session.is_locked());
        assert_eq!(session.handle(Input::Release, 330), None);
        assert_eq!(session.handle(Input::Tick, 5_000), None);
        assert_eq!(
            session.handle(Input::Press, 9_000),
            Some(Action::FinishCapture)
        );
    }

    #[test]
    fn toggle_mode_ignores_releases() {
        let mut session = toggle();
        assert_eq!(session.handle(Input::Press, 0), Some(Action::StartCapture));
        assert_eq!(session.handle(Input::Release, 80), None);
        assert_eq!(
            session.handle(Input::Press, 4_000),
            Some(Action::FinishCapture)
        );
    }

    #[test]
    fn cancel_discards_any_recording() {
        for mut session in [hold(), toggle()] {
            session.handle(Input::Press, 0);
            assert_eq!(
                session.handle(Input::Cancel, 900),
                Some(Action::DiscardCapture(DiscardReason::Cancelled))
            );
            assert!(session.is_idle());
        }
        assert_eq!(hold().handle(Input::Cancel, 0), None);
    }

    #[test]
    fn a_recording_finishes_at_the_length_limit() {
        let mut session = Session::new(SessionConfig {
            max_recording_ms: 10_000,
            ..SessionConfig::default()
        });
        session.handle(Input::Press, 1_000);
        assert_eq!(session.deadline(), Some(11_000));
        assert_eq!(session.handle(Input::Tick, 10_999), None);
        assert_eq!(
            session.handle(Input::Tick, 11_000),
            Some(Action::FinishCapture)
        );
        // The late release of the key is not a new recording.
        assert_eq!(session.handle(Input::Release, 11_500), None);
    }

    #[test]
    fn external_toggle_starts_hands_free_and_finishes() {
        let mut session = hold();
        assert_eq!(session.handle(Input::Toggle, 0), Some(Action::StartCapture));
        assert!(session.is_locked());
        assert_eq!(session.handle(Input::Start, 10), None);
        assert_eq!(
            session.handle(Input::Toggle, 3_000),
            Some(Action::FinishCapture)
        );
        session.handle(Input::Finished, 3_100);
        assert_eq!(session.handle(Input::Stop, 3_200), None);
        assert_eq!(
            session.handle(Input::Start, 3_300),
            Some(Action::StartCapture)
        );
        assert_eq!(
            session.handle(Input::Stop, 5_000),
            Some(Action::FinishCapture)
        );
    }

    #[test]
    fn auto_repeat_does_not_end_a_hold() {
        let mut session = Session::new(SessionConfig {
            release_debounce_ms: 40,
            ..SessionConfig::default()
        });
        session.handle(Input::Press, 0);
        // X11 auto-repeat: release and press arrive together.
        for at in [600, 640, 680] {
            assert_eq!(session.handle(Input::Release, at), None);
            assert_eq!(session.handle(Input::Press, at + 1), None);
            assert_eq!(session.handle(Input::Tick, at + 39), None);
        }
        assert!(session.is_recording());
        assert_eq!(session.handle(Input::Release, 2_000), None);
        assert_eq!(session.deadline(), Some(2_040));
        assert_eq!(
            session.handle(Input::Tick, 2_040),
            Some(Action::FinishCapture)
        );
    }

    #[test]
    fn a_debounced_tap_still_allows_a_double_tap() {
        let mut session = Session::new(SessionConfig {
            release_debounce_ms: 40,
            ..SessionConfig::default()
        });
        session.handle(Input::Press, 0);
        session.handle(Input::Release, 100);
        assert_eq!(session.handle(Input::Tick, 140), None);
        // The window runs from the release, not from the debounce.
        assert_eq!(session.deadline(), Some(400));
        session.handle(Input::Press, 300);
        assert!(session.is_locked());
    }
}
