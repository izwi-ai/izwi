//! How a row's decode ends, and its transcript text.
//!
//! The model card notes that in rare cases the model repeats a character or
//! phrase up to the token limit. The online detector stops a row once its
//! last tokens are one block repeated [`LOOP_REPEATS`] times and drops the
//! repeats, keeping one copy; the shared transcript cleanup
//! (`izwi_asr_toolkit::clean_transcript_text`, applied by the runtime to
//! every ASR family) then cuts runs of a repeated symbol.

/// Copies of one block (in tokens) that stop decoding.
pub const LOOP_REPEATS: usize = 20;
/// The longest token block the online detector looks for.
pub const LOOP_MAX_BLOCK_TOKENS: usize = 64;

/// Why decoding stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SovFinishReason {
    /// A stop token.
    Stop,
    /// The token budget.
    Length,
    /// The online loop detector.
    Repetition,
    /// The request was cancelled.
    Cancelled,
}

impl SovFinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::Repetition => "repetition",
            Self::Cancelled => "cancelled",
        }
    }
}

/// The length of the block `tokens` end in, when they end in one block of at
/// most [`LOOP_MAX_BLOCK_TOKENS`] repeated [`LOOP_REPEATS`] times.
pub fn trailing_loop(tokens: &[u32]) -> Option<usize> {
    let n = tokens.len();
    let max_block = LOOP_MAX_BLOCK_TOKENS.min(n / LOOP_REPEATS);
    (1..=max_block).find(|&block| {
        let start = n - block * LOOP_REPEATS;
        (start + block..n).all(|i| tokens[i] == tokens[i - block])
    })
}

/// `tokens` without the repeats of a trailing loop (one copy stays).
pub fn without_trailing_loop(tokens: &[u32]) -> &[u32] {
    match trailing_loop(tokens) {
        Some(block) => &tokens[..tokens.len() - block * (LOOP_REPEATS - 1)],
        None => tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_repeated_block_is_a_loop_and_one_copy_stays() {
        let mut tokens = vec![1, 2, 3];
        for _ in 0..LOOP_REPEATS {
            tokens.extend([7, 8]);
        }
        assert_eq!(trailing_loop(&tokens), Some(2));
        assert_eq!(without_trailing_loop(&tokens), &[1, 2, 3, 7, 8]);
        tokens.pop();
        assert_eq!(trailing_loop(&tokens), None);
        assert_eq!(without_trailing_loop(&tokens).len(), tokens.len());
    }

    #[test]
    fn ordinary_repetition_is_not_a_loop() {
        // Nineteen identical tokens (a long digit string) do not stop a row.
        let tokens = vec![5; LOOP_REPEATS - 1];
        assert_eq!(trailing_loop(&tokens), None);
        assert_eq!(trailing_loop(&[]), None);
    }
}
