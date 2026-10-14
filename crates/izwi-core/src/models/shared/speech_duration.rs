//! Spoken-duration estimates for automatic TTS output budgets.

/// How long `text` takes to speak at a deliberately slow pace: whitespace
/// separated words at `words_per_second`, and CJK characters (which carry
/// no spaces) at `cjk_chars_per_second`. Non-CJK text counts at least one
/// word per five letters, so unspaced runs are not one short word.
pub(crate) fn estimated_speech_secs(
    text: &str,
    words_per_second: f32,
    cjk_chars_per_second: f32,
) -> f32 {
    let cjk = text.chars().filter(|ch| is_cjk(*ch)).count();
    let words = text
        .split_whitespace()
        .filter(|word| word.chars().any(|ch| !is_cjk(ch)))
        .count();
    let letters = text
        .chars()
        .filter(|ch| !ch.is_whitespace() && !is_cjk(*ch))
        .count();
    let non_cjk = words.max(letters.div_ceil(5));
    non_cjk as f32 / words_per_second + cjk as f32 / cjk_chars_per_second
}

fn is_cjk(ch: char) -> bool {
    matches!(ch as u32,
        0x3040..=0x30ff | 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xac00..=0xd7af | 0x20000..=0x3134f
    )
}

#[cfg(test)]
mod tests {
    use super::estimated_speech_secs;

    #[test]
    fn words_and_cjk_characters_both_take_time() {
        assert_eq!(estimated_speech_secs("one two three four", 2.0, 3.0), 2.0);
        // A long unspaced word is not one short word.
        assert_eq!(estimated_speech_secs("abcdefghij", 2.0, 3.0), 1.0);
        // Mandarin carries no spaces: 30 characters take 10 s at 3 chars/s.
        assert_eq!(
            estimated_speech_secs(&"你好世界很".repeat(6), 2.0, 3.0),
            10.0
        );
        assert_eq!(estimated_speech_secs("", 2.0, 3.0), 0.0);
    }
}
