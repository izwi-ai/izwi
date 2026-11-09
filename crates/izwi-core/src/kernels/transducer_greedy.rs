//! Greedy transducer decoding, one joint step per row: the TDT rule of
//! Parakeet's retained decode and the RNNT rule of Nemotron ASR's.
//!
//! A joint step reads one encoder frame for each row, picks a label (and for
//! TDT a duration) from the joint's logits, and moves the row's cursor. The
//! eager decode loops apply these transitions on the host after reading the
//! picks back; the device decode applies the same transitions in a kernel,
//! so both are one definition: [`tdt_step`] and [`rnnt_step`].
//!
//! Picks follow [`crate::kernels::greedy`]: the largest logit, ties to the
//! lowest id, NaN never picked.

/// Sentinel for "no frame yet" in [`TdtRow::last_emit_t`].
pub const NO_FRAME: usize = usize::MAX;

/// The fixed rules of a TDT decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TdtRules {
    pub blank: usize,
    /// Duration logits after the token logits; durations are `0..n`.
    pub num_durations: usize,
    /// Labels a frame may emit before the cursor is pushed past it.
    pub max_symbols: usize,
}

/// One row's TDT cursor between joint steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TdtRow {
    /// The encoder frame the next joint step reads.
    pub t: usize,
    /// The frame of the row's last emission ([`NO_FRAME`] before any).
    pub last_emit_t: usize,
    /// Labels emitted at `last_emit_t` so far.
    pub emit_count_at_t: usize,
    /// Joint steps taken.
    pub guard_steps: usize,
}

impl TdtRow {
    /// A row before its first joint step.
    pub const START: Self = Self {
        t: 0,
        last_emit_t: NO_FRAME,
        emit_count_at_t: 0,
        guard_steps: 0,
    };
}

/// What one joint step did to a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TdtStep {
    /// A blank: the cursor moved by the duration (at least one frame).
    Blank,
    /// A label emitted at `frame` with its (clamped) duration index; the
    /// predictor consumes it.
    Emit {
        label: usize,
        frame: usize,
        duration: usize,
    },
    /// A label past the frame's `max_symbols`: dropped, and the cursor moved
    /// one frame past the frame the step read.
    Suppressed,
}

/// Parakeet's retained TDT transition for one joint step that picked
/// `label` and duration index `duration`: the cursor moves by the duration
/// (a blank of duration 0 by one frame), a label counts against its frame's
/// `max_symbols`, and one past them is dropped with the cursor pushed one
/// frame on.
pub fn tdt_step(row: &mut TdtRow, label: usize, duration: usize, rules: TdtRules) -> TdtStep {
    let duration = duration.min(rules.num_durations.saturating_sub(1));
    let mut jump = duration;
    if label == rules.blank && jump == 0 {
        jump = 1;
    }
    let frame = row.t;
    row.t = row.t.saturating_add(jump);
    row.guard_steps = row.guard_steps.saturating_add(1);
    if label == rules.blank {
        return TdtStep::Blank;
    }
    if frame == row.last_emit_t {
        row.emit_count_at_t = row.emit_count_at_t.saturating_add(1);
    } else {
        row.last_emit_t = frame;
        row.emit_count_at_t = 1;
    }
    if row.emit_count_at_t > rules.max_symbols {
        row.t = frame.saturating_add(1);
        return TdtStep::Suppressed;
    }
    TdtStep::Emit {
        label,
        frame,
        duration,
    }
}

/// The fixed rules of an RNNT decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RnntRules {
    pub blank: usize,
    /// Labels a frame may emit before the cursor moves on.
    pub max_symbols: usize,
}

/// One row's RNNT cursor between joint steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RnntRow {
    /// The encoder frame the next joint step reads.
    pub t: usize,
    /// Labels emitted on frame `t` so far.
    pub symbols: usize,
}

/// What one RNNT joint step did to a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RnntStep {
    /// A blank: the cursor moved to the next frame.
    Blank,
    /// A label emitted; `frame_full` when it was the frame's last allowed
    /// label and the cursor moved on.
    Emit { label: usize, frame_full: bool },
    /// A label past the blank: the joint's output is malformed.
    Invalid { label: usize },
}

/// Nemotron's RNNT transition for one joint step that picked `label`: a
/// blank moves to the next frame; a label is emitted, and the frame's
/// `max_symbols`-th moves the cursor on.
pub fn rnnt_step(row: &mut RnntRow, label: usize, rules: RnntRules) -> RnntStep {
    if label > rules.blank {
        return RnntStep::Invalid { label };
    }
    if label == rules.blank {
        row.t = row.t.saturating_add(1);
        row.symbols = 0;
        return RnntStep::Blank;
    }
    row.symbols = row.symbols.saturating_add(1);
    let frame_full = row.symbols >= rules.max_symbols;
    if frame_full {
        row.t = row.t.saturating_add(1);
        row.symbols = 0;
    }
    RnntStep::Emit { label, frame_full }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TDT: TdtRules = TdtRules {
        blank: 5,
        num_durations: 3,
        max_symbols: 2,
    };

    #[test]
    fn tdt_blank_of_duration_zero_moves_one_frame() {
        let mut row = TdtRow::START;
        assert_eq!(tdt_step(&mut row, 5, 0, TDT), TdtStep::Blank);
        assert_eq!((row.t, row.guard_steps), (1, 1));
        assert_eq!(tdt_step(&mut row, 5, 2, TDT), TdtStep::Blank);
        assert_eq!(row.t, 3);
        // Durations past the last clamp to it.
        assert_eq!(tdt_step(&mut row, 5, 9, TDT), TdtStep::Blank);
        assert_eq!(row.t, 5);
    }

    #[test]
    fn tdt_labels_count_per_frame_and_the_extra_one_is_dropped() {
        let mut row = TdtRow::START;
        let emit = |frame, duration| TdtStep::Emit {
            label: 1,
            frame,
            duration,
        };
        assert_eq!(tdt_step(&mut row, 1, 0, TDT), emit(0, 0));
        assert_eq!(tdt_step(&mut row, 1, 0, TDT), emit(0, 0));
        assert_eq!((row.t, row.emit_count_at_t), (0, 2));
        assert_eq!(tdt_step(&mut row, 1, 0, TDT), TdtStep::Suppressed);
        assert_eq!(row.t, 1);
        // A new frame restarts the count; the duration moves the cursor.
        assert_eq!(tdt_step(&mut row, 1, 2, TDT), emit(1, 2));
        assert_eq!((row.t, row.last_emit_t, row.emit_count_at_t), (3, 1, 1));
        assert_eq!(row.guard_steps, 4);
    }

    #[test]
    fn rnnt_moves_on_after_a_blank_or_a_full_frame() {
        let rules = RnntRules {
            blank: 4,
            max_symbols: 2,
        };
        let mut row = RnntRow::default();
        assert_eq!(
            rnnt_step(&mut row, 1, rules),
            RnntStep::Emit {
                label: 1,
                frame_full: false
            }
        );
        assert_eq!(
            rnnt_step(&mut row, 2, rules),
            RnntStep::Emit {
                label: 2,
                frame_full: true
            }
        );
        assert_eq!(row, RnntRow { t: 1, symbols: 0 });
        assert_eq!(rnnt_step(&mut row, 4, rules), RnntStep::Blank);
        assert_eq!(row, RnntRow { t: 2, symbols: 0 });
        assert_eq!(
            rnnt_step(&mut row, 5, rules),
            RnntStep::Invalid { label: 5 }
        );
        assert_eq!(row, RnntRow { t: 2, symbols: 0 });
    }
}
