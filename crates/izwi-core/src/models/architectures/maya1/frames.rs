//! Audio tokens to SNAC codes.
//!
//! A frame is seven tokens. Slot `j` of a frame should come from band `j`
//! (`128266 + 4096·j + code`), and unpacks into the three SNAC levels as the
//! reference's `unpack_snac_from_7` does:
//!
//! | slot | 0  | 1  | 2  | 3  | 4  | 5  | 6  |
//! |------|----|----|----|----|----|----|----|
//! | level| L1 | L2 | L3 | L3 | L2 | L3 | L3 |
//!
//! The reference takes each code as `(token − 128266) mod 4096`, accepting a
//! token from the wrong band; so does this port, and it counts them.

use super::config::{AUDIO_CODE_END, AUDIO_CODE_START, CODEBOOK_SIZE, TOKENS_PER_FRAME};

/// The three SNAC levels of whole frames: `n`, `2n` and `4n` codes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SnacCodes {
    pub coarse: Vec<u32>,
    pub medium: Vec<u32>,
    pub fine: Vec<u32>,
}

impl SnacCodes {
    pub fn frames(&self) -> usize {
        self.coarse.len()
    }

    /// Frames `start..start + count`.
    pub fn slice(&self, start: usize, count: usize) -> Self {
        Self {
            coarse: self.coarse[start..start + count].to_vec(),
            medium: self.medium[2 * start..2 * (start + count)].to_vec(),
            fine: self.fine[4 * start..4 * (start + count)].to_vec(),
        }
    }
}

pub fn is_audio_code(token: u32) -> bool {
    (AUDIO_CODE_START..AUDIO_CODE_END).contains(&token)
}

/// Unpack every whole frame of `tokens` (audio codes only); a trailing
/// partial frame is dropped, as upstream drops it.
pub fn unpack_frames(tokens: &[u32]) -> SnacCodes {
    let frames = tokens.len() / TOKENS_PER_FRAME;
    let mut codes = SnacCodes {
        coarse: Vec::with_capacity(frames),
        medium: Vec::with_capacity(2 * frames),
        fine: Vec::with_capacity(4 * frames),
    };
    for frame in tokens.as_chunks::<TOKENS_PER_FRAME>().0 {
        let code = |slot: usize| (frame[slot] - AUDIO_CODE_START) % CODEBOOK_SIZE;
        codes.coarse.push(code(0));
        codes.medium.extend([code(1), code(4)]);
        codes.fine.extend([code(2), code(3), code(5), code(6)]);
    }
    codes
}

/// Audio tokens sampled in a slot other than their band's.
pub fn out_of_band_tokens(tokens: &[u32]) -> usize {
    tokens
        .iter()
        .enumerate()
        .filter(|&(position, &token)| {
            let band = ((token - AUDIO_CODE_START) / CODEBOOK_SIZE) as usize;
            band != position % TOKENS_PER_FRAME
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(slot: u32, code: u32) -> u32 {
        AUDIO_CODE_START + slot * CODEBOOK_SIZE + code
    }

    #[test]
    fn frames_unpack_like_the_reference() {
        let tokens = (0..2u32)
            .flat_map(|frame| (0..7u32).map(move |slot| token(slot, 10 * frame + slot)))
            .chain([token(0, 99)])
            .collect::<Vec<_>>();
        let codes = unpack_frames(&tokens);
        assert_eq!(codes.coarse, vec![0, 10]);
        assert_eq!(codes.medium, vec![1, 4, 11, 14]);
        assert_eq!(codes.fine, vec![2, 3, 5, 6, 12, 13, 15, 16]);
        assert_eq!(codes.slice(1, 1).fine, vec![12, 13, 15, 16]);
        assert_eq!(out_of_band_tokens(&tokens[..14]), 0);
    }

    #[test]
    fn wrong_band_tokens_wrap_and_are_counted() {
        let mut tokens = (0..7u32).map(|slot| token(slot, 5)).collect::<Vec<_>>();
        tokens[2] = token(6, 7);
        assert_eq!(unpack_frames(&tokens).fine[0], 7);
        assert_eq!(out_of_band_tokens(&tokens), 1);
        assert!(is_audio_code(AUDIO_CODE_END - 1) && !is_audio_code(AUDIO_CODE_END));
    }
}
