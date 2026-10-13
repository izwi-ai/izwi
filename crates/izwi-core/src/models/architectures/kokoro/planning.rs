//! Chunk planning from phonemized clauses (KOK-11/KOK-12).
//!
//! The text is phonemized once (one espeak-ng call, see
//! `phonemizer::phonemize_units_with`), and its clauses are packed into
//! chunks that fit the voice pack's phoneme limit and the caller's
//! character budget, preferring to end a chunk at a sentence. A chunk's
//! phonemes go to the model as they are, so nothing is phonemized twice.

use super::phonemizer::{KokoroBoundary, KokoroPhonemeUnit};

/// Character budgets for planned chunks; every chunk also stays within the
/// voice pack's phoneme limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KokoroChunkPolicy {
    /// The first chunk's character budget: small when streaming, so audio
    /// starts after one short clause (KOK-12).
    pub first_chunk_max_chars: usize,
    /// Every later chunk's character budget.
    pub max_chunk_chars: usize,
}

impl KokoroChunkPolicy {
    /// One chunk when the text fits the voice pack, else the fewest chunks.
    pub const WHOLE: Self = Self {
        first_chunk_max_chars: usize::MAX,
        max_chunk_chars: usize::MAX,
    };

    /// A streaming plan: a first chunk of one short clause (up to 80
    /// characters), then chunks of up to `max_chunk_chars`.
    pub const fn streaming(max_chunk_chars: usize) -> Self {
        Self {
            first_chunk_max_chars: if max_chunk_chars < 80 {
                max_chunk_chars
            } else {
                80
            },
            max_chunk_chars,
        }
    }
}

/// One planned chunk: its source text and the phonemes the model reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KokoroPlannedChunk {
    pub text: String,
    pub phonemes: String,
}

fn chars(text: &str) -> usize {
    text.chars().count()
}

/// Pack `units` into chunks of at most `max_phonemes` phonemes (spaces
/// between clauses included) and the policy's character budgets. A chunk
/// that overflows is closed after its last sentence when it has one, else
/// after its last clause; a single clause over a budget is split at word
/// boundaries.
pub fn plan_chunks(
    units: Vec<KokoroPhonemeUnit>,
    policy: KokoroChunkPolicy,
    max_phonemes: usize,
) -> Vec<KokoroPlannedChunk> {
    let max_phonemes = max_phonemes.max(1);
    let mut pending = std::collections::VecDeque::new();
    for unit in units {
        pending.extend(split_oversized(unit, policy.max_chunk_chars, max_phonemes));
    }
    let mut chunks = Vec::new();
    let mut current: Vec<KokoroPhonemeUnit> = Vec::new();
    while let Some(unit) = pending.pop_front() {
        let budget = if chunks.is_empty() {
            policy.first_chunk_max_chars
        } else {
            policy.max_chunk_chars
        };
        let fits = current.is_empty() || {
            let text =
                current.iter().map(|u| chars(&u.text) + 1).sum::<usize>() + chars(&unit.text);
            let phonemes = current
                .iter()
                .map(|u| chars(&u.phonemes) + 1)
                .sum::<usize>()
                + chars(&unit.phonemes);
            text <= budget && phonemes <= max_phonemes
        };
        if fits {
            current.push(unit);
            continue;
        }
        // Close after the last sentence in the chunk, if any, and carry the
        // clauses after it into the next chunk.
        let cut = current
            .iter()
            .rposition(|u| u.boundary == KokoroBoundary::Sentence)
            .map_or(current.len(), |index| index + 1);
        let carried = current.split_off(cut);
        chunks.push(join(std::mem::take(&mut current)));
        pending.push_front(unit);
        for unit in carried.into_iter().rev() {
            pending.push_front(unit);
        }
    }
    if !current.is_empty() {
        chunks.push(join(current));
    }
    chunks
}

fn join(units: Vec<KokoroPhonemeUnit>) -> KokoroPlannedChunk {
    KokoroPlannedChunk {
        text: units
            .iter()
            .map(|u| u.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
        phonemes: units
            .iter()
            .map(|u| u.phonemes.as_str())
            .collect::<Vec<_>>()
            .join(" "),
    }
}

/// Split one clause over `max_chars` or `max_phonemes` at word boundaries
/// in its phonemes. The text is split into the same number of word groups,
/// so each piece's text is approximate (it is used for budgeting and logs;
/// the model reads the phonemes).
fn split_oversized(
    unit: KokoroPhonemeUnit,
    max_chars: usize,
    max_phonemes: usize,
) -> Vec<KokoroPhonemeUnit> {
    if chars(&unit.phonemes) <= max_phonemes && chars(&unit.text) <= max_chars {
        return vec![unit];
    }
    let pieces_for = |length: usize, max: usize| length.div_ceil(max.max(1)).max(1);
    let pieces = pieces_for(chars(&unit.phonemes), max_phonemes)
        .max(pieces_for(chars(&unit.text), max_chars));
    let phoneme_words = unit.phonemes.split(' ').collect::<Vec<_>>();
    let text_words = unit.text.split_whitespace().collect::<Vec<_>>();
    let pieces = pieces.min(phoneme_words.len()).max(1);
    let phonemes = word_groups(&phoneme_words, pieces, max_phonemes);
    let texts = even_groups(&text_words, phonemes.len());
    let count = phonemes.len();
    phonemes
        .into_iter()
        .zip(texts)
        .enumerate()
        .map(|(index, (phonemes, text))| KokoroPhonemeUnit {
            text,
            phonemes,
            boundary: if index + 1 == count {
                unit.boundary
            } else {
                KokoroBoundary::End
            },
        })
        .collect()
}

/// At least `pieces` groups of consecutive words, each within `max` chars
/// (a single word over `max` stays whole and is cut by the model's limit).
fn word_groups(words: &[&str], pieces: usize, max: usize) -> Vec<String> {
    let total = words.iter().map(|w| chars(w) + 1).sum::<usize>();
    let target = total.div_ceil(pieces).min(max);
    let mut groups = Vec::new();
    let mut current = String::new();
    for word in words {
        let next = if current.is_empty() {
            chars(word)
        } else {
            chars(&current) + 1 + chars(word)
        };
        if !current.is_empty() && next > target {
            groups.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
    }
    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

/// `words` in `count` nearly equal consecutive groups.
fn even_groups(words: &[&str], count: usize) -> Vec<String> {
    let count = count.max(1);
    (0..count)
        .map(|index| {
            let start = index * words.len() / count;
            let end = (index + 1) * words.len() / count;
            words[start..end].join(" ")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(text: &str, phonemes: &str, boundary: KokoroBoundary) -> KokoroPhonemeUnit {
        KokoroPhonemeUnit {
            text: text.into(),
            phonemes: phonemes.into(),
            boundary,
        }
    }

    fn units() -> Vec<KokoroPhonemeUnit> {
        use KokoroBoundary::*;
        vec![
            unit("Good morning.", "ɡʊd mˈɔɹnɪŋ.", Sentence),
            unit("Today we test,", "tədˈA wi tˈɛst,", Clause),
            unit("and we measure.", "ænd wi mˈɛʒəɹ.", Sentence),
            unit("Then we stop", "ðɛn wi stˈɑp", End),
        ]
    }

    /// KOK-11/12: the whole text is one chunk when it fits; a streaming
    /// plan starts with one short clause and then packs whole sentences.
    #[test]
    fn plans_whole_text_or_a_short_first_chunk() {
        let whole = plan_chunks(units(), KokoroChunkPolicy::WHOLE, 510);
        assert_eq!(whole.len(), 1);
        assert_eq!(
            whole[0].text,
            "Good morning. Today we test, and we measure. Then we stop"
        );
        assert_eq!(
            whole[0].phonemes,
            "ɡʊd mˈɔɹnɪŋ. tədˈA wi tˈɛst, ænd wi mˈɛʒəɹ. ðɛn wi stˈɑp"
        );

        let stream = plan_chunks(
            units(),
            KokoroChunkPolicy {
                first_chunk_max_chars: 20,
                max_chunk_chars: 40,
            },
            510,
        );
        let texts = stream.iter().map(|c| c.text.as_str()).collect::<Vec<_>>();
        assert_eq!(
            texts,
            [
                "Good morning.",
                "Today we test, and we measure.",
                "Then we stop"
            ]
        );

        // Only the first chunk is held to the small budget.
        use KokoroBoundary::Sentence;
        let short = vec![
            unit("Hi.", "hˈI.", Sentence),
            unit("Yes.", "jˈɛs.", Sentence),
            unit("Go now.", "ɡˈO nˈW.", Sentence),
        ];
        let stream = plan_chunks(
            short,
            KokoroChunkPolicy {
                first_chunk_max_chars: 5,
                max_chunk_chars: 40,
            },
            510,
        );
        let texts = stream.iter().map(|c| c.text.as_str()).collect::<Vec<_>>();
        assert_eq!(texts, ["Hi.", "Yes. Go now."]);
    }

    /// The phoneme limit closes a chunk after its last sentence, carrying
    /// the clauses after it on; an oversized clause splits at words.
    #[test]
    fn phoneme_limit_cuts_after_sentences_and_splits_long_clauses() {
        let chunks = plan_chunks(units(), KokoroChunkPolicy::WHOLE, 40);
        for chunk in &chunks {
            assert!(chunk.phonemes.chars().count() <= 40, "{chunk:?}");
        }
        let phonemes = chunks
            .iter()
            .map(|c| c.phonemes.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            phonemes,
            [
                "ɡʊd mˈɔɹnɪŋ.",
                "tədˈA wi tˈɛst, ænd wi mˈɛʒəɹ.",
                "ðɛn wi stˈɑp"
            ]
        );

        let long = unit(
            "one two three four five six",
            "wˈʌn tˈu θɹˈi fˈɔɹ fˈIv sˈɪks",
            KokoroBoundary::Sentence,
        );
        let split = plan_chunks(vec![long], KokoroChunkPolicy::WHOLE, 12);
        assert!(split.len() >= 3, "{split:?}");
        assert!(
            split.iter().all(|c| c.phonemes.chars().count() <= 12),
            "{split:?}"
        );
        assert_eq!(
            split
                .iter()
                .map(|c| c.phonemes.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            "wˈʌn tˈu θɹˈi fˈɔɹ fˈIv sˈɪks"
        );
        assert_eq!(
            split
                .iter()
                .map(|c| c.text.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            "one two three four five six"
        );
    }
}
