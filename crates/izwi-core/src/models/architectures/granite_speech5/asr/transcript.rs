//! Granite Speech 5.0 output cleanup. 5.0 emits plain text (no speaker
//! tags or timestamps), so only stray special tokens are removed.

pub fn clean_granite_speech5_output(raw: &str) -> String {
    raw.replace("<|end_of_text|>", "")
        .replace("<|endoftext|>", "")
        .replace("<|pad|>", "")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_special_tokens_and_surrounding_whitespace() {
        assert_eq!(
            clean_granite_speech5_output(" Hello, world.<|end_of_text|>\n"),
            "Hello, world."
        );
        assert_eq!(clean_granite_speech5_output("<|pad|>"), "");
        // Brackets are ordinary text for 5.0.
        assert_eq!(
            clean_granite_speech5_output("[Speaker 1]: hi"),
            "[Speaker 1]: hi"
        );
    }
}
