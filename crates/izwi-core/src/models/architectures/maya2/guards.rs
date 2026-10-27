//! Maya2's per-piece frame budget and loop guard, ported from upstream's
//! engine (`maya3/guards.py`, `maya3/engine.py:405-617`, GitHub `bbfdb152`).
//!
//! - **Budget.** A per-language prior over the piece's characters gives the
//!   expected frames `ef` and its 99th-percentile residual; a second prior
//!   over (Indic letters, Latin letters, pauses, digits) gives the expected
//!   duration. The cap is `min(min(1500, 2(ef + resid) + 40),
//!   max(⌊0.5·ef⌋ + 8, ⌊(2.2·dur + 1.2) / 0.08⌋))`; the stop may fire from
//!   `max(4, ⌊0.5·ef⌋)` frames on.
//! - **Loop guard.** Every second frame from frame 32, outside suspend
//!   windows, the last 96 cb0 codes are checked for a 32-frame stall or a
//!   period of 2–16 repeating on ≥ 70% of the last 64. A hit adds +4 to the
//!   end-of-audio logit so the model ends the phrase itself; a loop still
//!   there 24 frames later stops the piece. A clean check removes the boost.
//! - **Suspend windows** cover where a loop is legitimate: around `[tags]`,
//!   fillers (hmm, umm, …) and `Name: ` dialogue switches.
//!
//! Upstream counts letters with Python's `str.isalpha` (`L*`, as here) and
//! digits with `str.isdigit`, for which decimal digits (`Nd`) stand in: they
//! differ only on superscript digits.

use std::sync::OnceLock;

use icu_properties::props::GeneralCategory;
use icu_properties::CodePointMapData;
use regex::Regex;

use super::config::{FRAME_SECONDS, HARD_CAP_FRAMES};

/// `frames = a + b · chars` and the residual's 99th percentile, per
/// language (`configs/text_frames_prior.json`).
const FRAME_PRIOR: [(&str, f64, f64, f64); 12] = [
    ("as", -1.36, 0.7786, 91.6),
    ("bn", 3.16, 0.7887, 76.2),
    ("en", 0.6, 0.7416, 75.0),
    ("gu", 1.74, 0.8834, 83.1),
    ("hi", 5.65, 0.7644, 79.8),
    ("kn", 1.91, 0.7675, 69.5),
    ("ml", 1.25, 0.6788, 105.6),
    ("mr", 1.12, 0.8235, 60.1),
    ("or", 1.06, 0.7936, 68.4),
    ("pa", 1.86, 0.8079, 73.5),
    ("ta", 2.24, 0.6425, 70.8),
    ("te", 1.2, 0.7927, 89.6),
];
/// Upstream's fallback for a language without a prior.
const FRAME_PRIOR_DEFAULT: (f64, f64, f64) = (2.0, 0.72, 90.0);

/// `dur_s = c0 + c1·indic_letters + c2·latin_letters + c3·pauses +
/// c4·digits` (`maya3/assets/duration_prior.json`).
const DURATION_PRIOR: [(&str, [f64; 5]); 12] = [
    ("as", [0.0, 0.07962, 0.17764, 0.53121, 0.62254]),
    ("bn", [0.71369, 0.06604, 0.04537, 0.49869, 0.23288]),
    ("en", [2.87006, 0.01, 0.04601, 0.40336, 0.06541]),
    ("gu", [0.54378, 0.07044, 0.01, 0.50786, 0.24873]),
    ("hi", [2.44169, 0.05329, 0.02187, 0.43341, 0.37052]),
    ("kn", [1.71201, 0.06651, 0.01, 0.45228, 0.5053]),
    ("ml", [0.92428, 0.05706, 0.11842, 0.52651, 0.41106]),
    ("mr", [1.44052, 0.0719, 0.03206, 0.30526, 0.28705]),
    ("or", [0.0, 0.08134, 0.12578, 0.43209, 0.23588]),
    ("pa", [0.57673, 0.07431, 0.01, 0.39355, 0.37283]),
    ("ta", [2.05273, 0.05244, 0.01, 0.47524, 0.45486]),
    ("te", [0.45026, 0.07269, 0.0576, 0.46093, 0.39692]),
];

const DURATION_K: f64 = 2.2;
const DURATION_MARGIN_S: f64 = 1.2;
const LOOP_START: usize = 32;
const LOOP_EVERY: usize = 2;
const STALL: usize = 32;
const PERIOD_WINDOW: usize = 64;
const PERIOD_THRESHOLD: f64 = 0.7;
const PERIOD_MIN: usize = 2;
const PERIOD_MAX: usize = 16;
const LOOP_MIN_FRAMES: usize = 12;
/// cb0 codes the loop guard keeps per piece.
pub const LOOP_HISTORY: usize = 96;
pub const LOOP_BOOST: f32 = 4.0;
const LOOP_BACKSTOP: usize = 24;
const TAG_HOLD_S: f64 = 4.0;
const STYLE_HOLD_S: f64 = 4.0;
const FILLER_HOLD_S: f64 = 2.5;
const SWITCH_HOLD_S: f64 = 2.0;

/// Tags upstream treats as styles colouring the speech after them.
const STYLE_TAGS: &[&str] = &[
    "whispers",
    "whispering",
    "sad",
    "excited",
    "curious",
    "shouting",
    "angry",
    "frustrated",
    "sarcastic",
    "crying",
    "nervous",
    "scared",
    "happy",
    "panicked",
    "amazed",
    "worried",
    "calm",
    "serious",
    "surprised",
    "cheerful",
    "tired",
    "fearful",
    "disgusted",
    "confused",
    "sleepy",
    "annoyed",
    "softly",
    "loudly",
];

fn pause_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[,;:.!?।॥…\-]").expect("pause pattern"))
}

fn tag_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\[([a-z _]+)\]").expect("tag pattern"))
}

fn filler_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Python's `(?<!\w)…(?!\w)` lookarounds become explicit boundary checks
    // in `filler_starts`.
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)(?:hmm+|hm+|umm+|um|uh+|uhh|erm|ह्म्म?|हम्म|उम्म?|ಹ್ಮ್|ಉಮ್|ஹ்ம்|ம்ம்|உம்|উমম?|হুম|ଉମ୍|ହମ୍|ਹਮ|ਉਮ|હમ્મ|ઉમ|ఉమ్|హ్మ్|ഉം|ഹ്മ്)",
        )
        .expect("filler pattern")
    })
}

fn switch_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // `(?:(?<=[.!?।॥])\s+|\n)NAME:\s`; the lookbehind is checked by hand.
    RE.get_or_init(|| Regex::new(r"(\s+)([^\s:,.!?।]{1,24}):\s").expect("switch pattern"))
}

fn category(ch: char) -> GeneralCategory {
    CodePointMapData::<GeneralCategory>::new().get(ch)
}

/// Python's `str.isalpha`: a letter (`L*`); combining marks such as Indic
/// vowel signs are not.
fn is_alpha(ch: char) -> bool {
    matches!(
        category(ch),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

/// Python's `str.isdigit`, as decimal digits (`Nd`), Indic digits included.
fn is_digit(ch: char) -> bool {
    category(ch) == GeneralCategory::DecimalNumber
}

/// Python's `\w` (`str.isalnum` or `_`) for the filler lookarounds.
fn is_word_char(ch: char) -> bool {
    ch == '_'
        || is_alpha(ch)
        || matches!(
            category(ch),
            GeneralCategory::DecimalNumber
                | GeneralCategory::LetterNumber
                | GeneralCategory::OtherNumber
        )
}

fn frame_prior(language: &str) -> (f64, f64, f64) {
    FRAME_PRIOR
        .iter()
        .find(|(code, ..)| *code == language)
        .map_or(FRAME_PRIOR_DEFAULT, |&(_, a, b, residual)| (a, b, residual))
}

fn duration_coefficients(language: &str) -> [f64; 5] {
    if let Some((_, coef)) = DURATION_PRIOR.iter().find(|(code, _)| *code == language) {
        return *coef;
    }
    let mut mean = [0.0; 5];
    for (_, coef) in DURATION_PRIOR {
        for (sum, value) in mean.iter_mut().zip(coef) {
            *sum += value / DURATION_PRIOR.len() as f64;
        }
    }
    mean
}

/// `[1, Indic letters, Latin letters, pauses, digits]` of `text`.
fn text_features(text: &str) -> [f64; 5] {
    let mut letters = 0usize;
    let mut latin = 0usize;
    let mut digits = 0usize;
    for ch in text.chars() {
        let code = ch as u32;
        if is_alpha(ch) || (0x0900..=0x0DFF).contains(&code) {
            letters += 1;
        }
        if ch.is_ascii_alphabetic() {
            latin += 1;
        }
        if is_digit(ch) {
            digits += 1;
        }
    }
    let pauses = pause_re().find_iter(text).count();
    [
        1.0,
        (letters - latin) as f64,
        latin as f64,
        pauses as f64,
        digits as f64,
    ]
}

/// Expected seconds of speech for `text`, at least 0.35.
pub fn expected_seconds(text: &str, language: &str) -> f64 {
    let features = text_features(text);
    let coef = duration_coefficients(language);
    let dot = features.iter().zip(coef).map(|(f, c)| f * c).sum::<f64>();
    dot.max(0.35)
}

/// A piece's frame budget and the earliest frame its stop may fire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Maya2PieceBudget {
    /// Frames the piece may generate (the cap frame is emitted).
    pub cap: usize,
    /// The cap before the duration guard (stops below it are guard stops).
    pub loose_cap: usize,
    /// Frames before the controlled stop may fire.
    pub minimum: usize,
}

/// The budget of a piece `text` in `language`, within `max_frames` when the
/// caller set one.
pub fn piece_budget(text: &str, language: &str, max_frames: Option<usize>) -> Maya2PieceBudget {
    let chars = text.chars().count() as f64;
    let (a, b, residual) = frame_prior(language);
    let expected_frames = a + b * chars;
    let loose = (HARD_CAP_FRAMES as f64).min((2.0 * (expected_frames + residual) + 40.0).trunc());
    let loose = loose.max(1.0) as usize;
    let duration_cap = ((DURATION_K * expected_seconds(text, language) + DURATION_MARGIN_S)
        / f64::from(FRAME_SECONDS))
    .trunc() as usize;
    let half = (expected_frames * 0.5).trunc().max(0.0) as usize;
    let mut cap = loose.min((half + 8).max(duration_cap));
    if let Some(limit) = max_frames {
        cap = cap.min(limit.max(1));
    }
    Maya2PieceBudget {
        cap: cap.max(1),
        loose_cap: loose,
        minimum: half.max(4),
    }
}

/// Inclusive frame windows where the loop guard takes no action.
pub fn suspend_windows(text: &str, language: &str) -> Vec<(usize, usize)> {
    let mut events: Vec<(usize, f64)> = Vec::new();
    for captures in tag_re().captures_iter(text) {
        let whole = captures.get(0).expect("match");
        let name = captures.get(1).expect("group").as_str().trim();
        let hold = if STYLE_TAGS.contains(&name) {
            STYLE_HOLD_S
        } else {
            TAG_HOLD_S
        };
        events.push((whole.start(), hold));
    }
    for found in filler_re().find_iter(text) {
        let before = text[..found.start()].chars().next_back();
        let after = text[found.end()..].chars().next();
        if before.is_none_or(|ch| !is_word_char(ch)) && after.is_none_or(|ch| !is_word_char(ch)) {
            events.push((found.start(), FILLER_HOLD_S));
        }
    }
    for captures in switch_re().captures_iter(text) {
        let space = captures.get(1).expect("space");
        let name = captures.get(2).expect("name");
        let after_terminal = text[..space.start()]
            .chars()
            .next_back()
            .is_some_and(|ch| matches!(ch, '.' | '!' | '?' | '।' | '॥'));
        let after_newline = space.as_str().ends_with('\n');
        if after_terminal || after_newline {
            events.push((name.start(), SWITCH_HOLD_S));
        }
    }
    events.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
    events
        .into_iter()
        .map(|(position, hold)| {
            let t = if position > 0 {
                expected_seconds(&text[..position], language)
            } else {
                0.0
            };
            let frame = f64::from(FRAME_SECONDS);
            let start = ((0.8 * t - 0.5) / frame).trunc().max(0.0) as usize;
            let end = ((1.5 * t + hold) / frame).trunc() as usize;
            (start, end)
        })
        .collect()
}

/// What the loop check found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopHit {
    Stall,
    Period(usize),
}

/// The loop check over the last cb0 codes `history` (oldest first, at most
/// [`LOOP_HISTORY`]) after `frames` frames.
pub fn loop_detect(history: &[u32], frames: usize) -> Option<LoopHit> {
    if frames < LOOP_MIN_FRAMES.max(STALL) || history.len() < STALL {
        return None;
    }
    let tail = &history[history.len() - STALL..];
    if tail.iter().all(|&code| code == tail[0]) {
        return Some(LoopHit::Stall);
    }
    if frames >= PERIOD_WINDOW && history.len() >= PERIOD_WINDOW {
        let window = &history[history.len() - PERIOD_WINDOW..];
        for period in PERIOD_MIN..=PERIOD_MAX {
            let pairs = PERIOD_WINDOW - period;
            let same = (0..pairs)
                .filter(|&i| window[i + period] == window[i])
                .count();
            if same as f64 / pairs as f64 >= PERIOD_THRESHOLD {
                return Some(LoopHit::Period(period));
            }
        }
    }
    None
}

/// Why a piece stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Maya2StopReason {
    /// End of audio was the most likely class.
    AeosArgmax,
    /// p(end of audio) reached 0.35.
    AeosTau,
    /// The cumulative stop hazard reached 0.6.
    AeosHazard,
    /// The duration guard's cap.
    CapGuard,
    /// The prior's loose cap.
    Cap,
    /// The caller's frame limit.
    MaxFrames,
    /// A loop persisted after the end-of-audio boost.
    LoopBackstop,
}

/// One piece's loop-guard state.
#[derive(Debug, Clone, PartialEq)]
pub struct LoopGuard {
    history: Vec<u32>,
    suspend: Vec<(usize, usize)>,
    boost_at: Option<usize>,
    pub boosts: usize,
}

impl LoopGuard {
    pub fn new(suspend: Vec<(usize, usize)>) -> Self {
        Self {
            history: Vec::with_capacity(LOOP_HISTORY),
            suspend,
            boost_at: None,
            boosts: 0,
        }
    }

    /// The end-of-audio logit bias the next step applies.
    pub fn aeos_bias(&self) -> f32 {
        if self.boost_at.is_some() {
            LOOP_BOOST
        } else {
            0.0
        }
    }

    /// Record the cb0 code of emitted frame number `frames` (1-based) and run
    /// the check when due; `true` when the loop backstop stops the piece.
    pub fn observe(&mut self, cb0: u32, frames: usize) -> bool {
        if self.history.len() == LOOP_HISTORY {
            self.history.remove(0);
        }
        self.history.push(cb0);
        let due = frames >= LOOP_START
            && frames.is_multiple_of(LOOP_EVERY)
            && !self
                .suspend
                .iter()
                .any(|&(start, end)| start <= frames && frames <= end);
        if !due {
            return false;
        }
        match loop_detect(&self.history, frames) {
            Some(_) => {
                let boosted = *self.boost_at.get_or_insert_with(|| {
                    self.boosts += 1;
                    frames
                });
                frames - boosted >= LOOP_BACKSTOP
            }
            None => {
                self.boost_at = None;
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets_follow_the_priors() {
        let budget = piece_budget("Hello there, how are you?", "en", None);
        assert!(budget.cap > 20, "{budget:?}");
        assert!(budget.cap <= budget.loose_cap);
        assert_eq!(budget.minimum, 9); // ⌊0.5 · (0.6 + 0.7416 · 25)⌋
        let limited = piece_budget("Hello there, how are you?", "en", Some(5));
        assert_eq!(limited.cap, 5);
        assert!(piece_budget(&"a".repeat(5000), "en", None).cap <= HARD_CAP_FRAMES);
        // Unknown languages fall back to upstream's defaults.
        assert!(piece_budget("hello", "xx", None).cap > 0);
    }

    #[test]
    fn loop_detection_finds_stalls_and_periods() {
        let stalled = [vec![1; 10], vec![7; 40]].concat();
        assert_eq!(loop_detect(&stalled, 50), Some(LoopHit::Stall));
        let periodic = (0..96).map(|i| [3, 9, 4][i % 3]).collect::<Vec<u32>>();
        assert_eq!(loop_detect(&periodic, 96), Some(LoopHit::Period(3)));
        let varied = (0..96u32)
            .map(|i| crate::models::shared::noise::mix(i as u64) as u32 % 50)
            .collect::<Vec<_>>();
        assert_eq!(loop_detect(&varied, 96), None);
        // Too early for any rule.
        assert_eq!(loop_detect(&stalled[..20], 20), None);
    }

    #[test]
    fn the_guard_boosts_then_backstops() {
        let mut guard = LoopGuard::new(Vec::new());
        let mut stopped_at = None;
        for frame in 1..=100 {
            if guard.observe(7, frame) {
                stopped_at = Some(frame);
                break;
            }
            if frame == 32 {
                assert_eq!(guard.aeos_bias(), LOOP_BOOST);
            }
        }
        assert_eq!(stopped_at, Some(56));
        assert_eq!(guard.boosts, 1);
        // A clean check removes the boost.
        let mut guard = LoopGuard::new(Vec::new());
        for frame in 1..=32 {
            guard.observe(7, frame);
        }
        assert_eq!(guard.aeos_bias(), LOOP_BOOST);
        for frame in 33..=100u32 {
            guard.observe(frame * 13 % 41, frame as usize);
        }
        assert_eq!(guard.aeos_bias(), 0.0);
    }

    #[test]
    fn suspend_windows_mirror_upstream_shapes() {
        // A style tag at t = 0: [0, 4 s].
        let windows = suspend_windows(
            "[whispers] arre sun, maine tujhe paise bhej diye hain",
            "hi",
        );
        assert_eq!(windows[0], (0, 50));
        assert_eq!(
            suspend_windows("I checked it, [laughs] and it is fine now.", "en").len(),
            1
        );
        // A filler.
        assert_eq!(
            suspend_windows("ನಮಸ್ಕಾರ, ನಾನು ರಮೇಶ್. ಹ್ಮ್, ನಿಮ್ಮ ಕಾರ್ಡ್ ನಂಬರ್", "kn").len(),
            1
        );
        // A speaker switch (the leading name is not one).
        assert_eq!(
            suspend_windows("Rohan: are you coming? Priya: yes, I am.", "en").len(),
            1
        );
        // "um" inside a word is not a filler.
        assert!(suspend_windows("The museum opens at nine.", "en").is_empty());
    }

    #[test]
    fn suspend_windows_stop_the_check() {
        let mut guard = LoopGuard::new(vec![(0, 200)]);
        for frame in 1..=100 {
            assert!(!guard.observe(7, frame));
        }
        assert_eq!(guard.aeos_bias(), 0.0);
    }
}
