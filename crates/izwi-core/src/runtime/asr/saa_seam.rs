//! Linking Granite speaker-attributed chunks at their seams.
//!
//! Granite numbers speakers by first appearance within each decode, so a
//! chunk's "Speaker 1" says nothing on its own about the transcript so far.
//! The audio two consecutive chunks share is the evidence: align the words
//! both decodes wrote for it, let every aligned word vote for which
//! transcript speaker each chunk label is, and keep one copy of the shared
//! words (the chunk's, which saw the audio with context on both sides).

use std::collections::{HashMap, HashSet};

use super::{SpeakerAttributedAsrTurn, UNKNOWN_SAA_SPEAKER};

/// Words of the transcript's tail and of the chunk's head compared at a
/// seam: well over what the overlap holds at conversational rates.
const SEAM_WINDOW_WORDS: usize = 200;
/// Aligned words needed before an alignment is trusted.
const MIN_ALIGNED_WORDS: usize = 4;
/// Most transcript words a seam may hand over to the chunk's decode of the
/// same audio. More than this means the alignment latched onto an earlier
/// repeat, so it is not trusted.
const MAX_REPLACED_TRANSCRIPT_WORDS: usize = 120;
/// Votes a chunk label needs before it is linked by the alignment.
const MIN_LABEL_VOTES: usize = 2;

/// How one chunk joins the transcript.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct SeamLink {
    /// Chunk speaker label to transcript speaker label.
    pub(super) labels: HashMap<String, String>,
    /// Words the two decodes aligned on; zero when the seam is unaligned.
    pub(super) aligned_words: usize,
    /// The chunk's last word (turn, token) the transcript already holds.
    pub(super) chunk_known_through: Option<(usize, usize)>,
    /// The transcript's last word (turn, token) to keep: what follows is the
    /// cut-off end of the previous chunk, which the chunk decoded again.
    pub(super) transcript_keep_through: Option<(usize, usize)>,
}

impl SeamLink {
    pub(super) fn is_aligned(&self) -> bool {
        self.aligned_words > 0
    }
}

/// A word, where it sits in its turns, and who said it.
#[derive(Debug, Clone)]
struct SeamWord<'a> {
    turn: usize,
    token: usize,
    speaker: &'a str,
    normalized: String,
}

/// Link `chunk` onto `transcript`. With no transcript yet, labels are kept.
pub(super) fn link_seam(
    transcript: &[SpeakerAttributedAsrTurn],
    chunk: &[SpeakerAttributedAsrTurn],
) -> SeamLink {
    let transcript_words = seam_words(transcript);
    if transcript_words.is_empty() {
        return SeamLink {
            labels: local_speakers(chunk)
                .into_iter()
                .map(|speaker| (speaker.to_string(), speaker.to_string()))
                .collect(),
            ..SeamLink::default()
        };
    }
    let tail = &transcript_words[transcript_words.len().saturating_sub(SEAM_WINDOW_WORDS)..];
    let chunk_words = seam_words(chunk);
    let head = &chunk_words[..chunk_words.len().min(SEAM_WINDOW_WORDS)];

    let anchors = aligned_runs(tail, head);
    let replaced_transcript_words = anchors
        .last()
        .map_or(usize::MAX, |&(tail_index, _)| tail.len() - 1 - tail_index);
    let aligned = anchors.len() >= MIN_ALIGNED_WORDS
        && replaced_transcript_words <= MAX_REPLACED_TRANSCRIPT_WORDS;

    let mut votes = HashMap::<(&str, &str), usize>::new();
    if aligned {
        for &(tail_index, head_index) in &anchors {
            let (global, local) = (tail[tail_index].speaker, head[head_index].speaker);
            if global != UNKNOWN_SAA_SPEAKER && local != UNKNOWN_SAA_SPEAKER {
                *votes.entry((local, global)).or_default() += 1;
            }
        }
    }
    let labels = assign_labels(transcript, chunk, &votes, aligned);

    if !aligned {
        return SeamLink {
            labels,
            ..SeamLink::default()
        };
    }
    let &(last_tail, last_head) = anchors.last().expect("aligned seams have anchors");
    SeamLink {
        labels,
        aligned_words: anchors.len(),
        chunk_known_through: Some((head[last_head].turn, head[last_head].token)),
        transcript_keep_through: Some((tail[last_tail].turn, tail[last_tail].token)),
    }
}

/// Map every chunk label to a transcript label. Labels the alignment backs
/// (most votes first, one chunk label per transcript speaker) come first.
/// Without an alignment, the chunk's first speaker continues the transcript's
/// last one: the chunk opens on audio the transcript just ended with. Other
/// labels take the transcript speaker heard most recently that no label has
/// claimed yet, and a new number once every known speaker is taken.
fn assign_labels(
    transcript: &[SpeakerAttributedAsrTurn],
    chunk: &[SpeakerAttributedAsrTurn],
    votes: &HashMap<(&str, &str), usize>,
    aligned: bool,
) -> HashMap<String, String> {
    let local_order = local_speakers(chunk);
    let mut labels = HashMap::<String, String>::new();
    let mut claimed = HashSet::<String>::new();

    let mut totals = HashMap::<&str, usize>::new();
    for (&(local, _), &count) in votes {
        *totals.entry(local).or_default() += count;
    }
    let order_of = |speaker: &str| local_order.iter().position(|local| *local == speaker);
    let mut ranked = votes.iter().collect::<Vec<_>>();
    ranked.sort_by(|((left_local, left_global), left_count), ((right_local, right_global), right_count)| {
        right_count
            .cmp(left_count)
            .then_with(|| order_of(left_local).cmp(&order_of(right_local)))
            .then_with(|| left_global.cmp(right_global))
    });
    for (&(local, global), &count) in ranked {
        if labels.contains_key(local) || claimed.contains(global) {
            continue;
        }
        if count >= MIN_LABEL_VOTES && count * 2 > totals[local] {
            labels.insert(local.to_string(), global.to_string());
            claimed.insert(global.to_string());
        }
    }

    let recent = recent_speakers(transcript);
    if !aligned {
        if let (Some(first), Some(last)) = (local_order.first(), recent.first()) {
            if !labels.contains_key(*first) && !claimed.contains(*last) {
                labels.insert(first.to_string(), last.to_string());
                claimed.insert(last.to_string());
            }
        }
    }

    let mut next_number = transcript
        .iter()
        .filter_map(|turn| speaker_number(&turn.speaker))
        .max()
        .unwrap_or(0)
        + 1;
    for local in local_order {
        if labels.contains_key(local) {
            continue;
        }
        let global = match recent.iter().find(|speaker| !claimed.contains(**speaker)) {
            Some(speaker) => speaker.to_string(),
            None => {
                let label = format!("Speaker {next_number}");
                next_number += 1;
                label
            }
        };
        claimed.insert(global.clone());
        labels.insert(local.to_string(), global);
    }
    labels
}

/// Aligned word pairs (tail index, head index) from a longest common
/// subsequence of the normalized words, keeping only pairs inside runs of
/// two or more consecutive matches: a lone "the" matching across is noise.
fn aligned_runs(tail: &[SeamWord<'_>], head: &[SeamWord<'_>]) -> Vec<(usize, usize)> {
    let (rows, cols) = (tail.len(), head.len());
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    // lengths[i][j]: LCS length of tail[i..] and head[j..].
    let mut lengths = vec![0u16; (rows + 1) * (cols + 1)];
    let at = |i: usize, j: usize| i * (cols + 1) + j;
    for i in (0..rows).rev() {
        for j in (0..cols).rev() {
            lengths[at(i, j)] = if tail[i].normalized == head[j].normalized {
                lengths[at(i + 1, j + 1)] + 1
            } else {
                lengths[at(i + 1, j)].max(lengths[at(i, j + 1)])
            };
        }
    }
    let mut pairs = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < rows && j < cols {
        if tail[i].normalized == head[j].normalized {
            pairs.push((i, j));
            i += 1;
            j += 1;
        } else if lengths[at(i + 1, j)] >= lengths[at(i, j + 1)] {
            i += 1;
        } else {
            j += 1;
        }
    }
    let adjacent = |left: (usize, usize), right: (usize, usize)| {
        right.0 == left.0 + 1 && right.1 == left.1 + 1
    };
    pairs
        .iter()
        .enumerate()
        .filter(|(index, pair)| {
            (*index > 0 && adjacent(pairs[index - 1], **pair))
                || pairs.get(index + 1).is_some_and(|next| adjacent(**pair, *next))
        })
        .map(|(_, pair)| *pair)
        .collect()
}

fn seam_words(turns: &[SpeakerAttributedAsrTurn]) -> Vec<SeamWord<'_>> {
    turns
        .iter()
        .enumerate()
        .flat_map(|(turn_index, turn)| {
            turn.text
                .split_whitespace()
                .enumerate()
                .filter_map(move |(token, word)| {
                    normalize_word(word).map(|normalized| SeamWord {
                        turn: turn_index,
                        token,
                        speaker: turn.speaker.as_str(),
                        normalized,
                    })
                })
        })
        .collect()
}

fn normalize_word(word: &str) -> Option<String> {
    let normalized = word
        .chars()
        .filter(|ch| ch.is_alphanumeric() || *ch == '\'')
        .flat_map(char::to_lowercase)
        .collect::<String>();
    (!normalized.is_empty()).then_some(normalized)
}

/// The chunk's speaker labels in order of first appearance.
fn local_speakers(chunk: &[SpeakerAttributedAsrTurn]) -> Vec<&str> {
    let mut speakers = Vec::<&str>::new();
    for turn in chunk {
        let speaker = turn.speaker.as_str();
        if speaker != UNKNOWN_SAA_SPEAKER && !speakers.contains(&speaker) {
            speakers.push(speaker);
        }
    }
    speakers
}

/// The transcript's speakers, most recently heard first.
fn recent_speakers(transcript: &[SpeakerAttributedAsrTurn]) -> Vec<&str> {
    let mut speakers = Vec::<&str>::new();
    for turn in transcript.iter().rev() {
        let speaker = turn.speaker.as_str();
        if speaker != UNKNOWN_SAA_SPEAKER && !speakers.contains(&speaker) {
            speakers.push(speaker);
        }
    }
    speakers
}

fn speaker_number(label: &str) -> Option<usize> {
    label.strip_prefix("Speaker ")?.trim().parse().ok()
}

/// Remove every word after `(turn, token)`, and turns left empty.
pub(super) fn truncate_turns_after(
    turns: &mut Vec<SpeakerAttributedAsrTurn>,
    (turn_index, token): (usize, usize),
) {
    turns.truncate(turn_index + 1);
    if let Some(turn) = turns.get_mut(turn_index) {
        turn.text = turn
            .text
            .split_whitespace()
            .take(token + 1)
            .collect::<Vec<_>>()
            .join(" ");
    }
    turns.retain(|turn| !turn.text.trim().is_empty());
}

/// Remove every word up to and including `(turn, token)`, and turns left
/// empty.
pub(super) fn drop_turn_words_through(
    turns: Vec<SpeakerAttributedAsrTurn>,
    (turn_index, token): (usize, usize),
) -> Vec<SpeakerAttributedAsrTurn> {
    turns
        .into_iter()
        .enumerate()
        .filter_map(|(index, mut turn)| {
            if index < turn_index {
                return None;
            }
            if index == turn_index {
                turn.text = turn
                    .text
                    .split_whitespace()
                    .skip(token + 1)
                    .collect::<Vec<_>>()
                    .join(" ");
            }
            (!turn.text.trim().is_empty()).then_some(turn)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turns(spec: &[(&str, &str)]) -> Vec<SpeakerAttributedAsrTurn> {
        spec.iter()
            .map(|(speaker, text)| SpeakerAttributedAsrTurn {
                speaker: speaker.to_string(),
                text: text.to_string(),
                start_secs: None,
                end_secs: None,
            })
            .collect()
    }

    fn label<'a>(link: &'a SeamLink, local: &str) -> &'a str {
        link.labels[local].as_str()
    }

    #[test]
    fn swapped_chunk_labels_follow_the_overlap_votes() {
        // The chunk opens on the guest (its "Speaker 1"), who is the
        // transcript's "Speaker 2"; the interviewer comes second.
        let transcript = turns(&[
            ("Speaker 1", "how did it all start for you back then"),
            ("Speaker 2", "well I started as a trading company in nineteen seventy eight"),
        ]);
        let chunk = turns(&[
            ("Speaker 1", "as a trading company in nineteen seventy eight"),
            ("Speaker 2", "what kind of things did you trade"),
        ]);
        let link = link_seam(&transcript, &chunk);

        assert!(link.is_aligned());
        assert_eq!(label(&link, "Speaker 1"), "Speaker 2");
        assert_eq!(label(&link, "Speaker 2"), "Speaker 1");
        // The chunk's copy of the shared words is dropped through "eight".
        assert_eq!(link.chunk_known_through, Some((0, 7)));
        assert_eq!(link.transcript_keep_through, Some((1, 10)));
    }

    #[test]
    fn misheard_words_at_the_cut_still_align() {
        // The previous chunk ends on a cut word ("fewer things" was "few
        // things" re-decoded with context); the chunk opens differently.
        let transcript = turns(&[
            ("Speaker 1", "we import sugar and rice and we export to fourteen countries fewer things"),
        ]);
        let chunk = turns(&[
            ("Speaker 1", "on a normal day we export to fourteen countries few things"),
            ("Speaker 2", "which countries"),
        ]);
        let link = link_seam(&transcript, &chunk);

        assert!(link.is_aligned());
        assert_eq!(label(&link, "Speaker 1"), "Speaker 1");
        // "Speaker 2" was not in the overlap: it takes the transcript
        // speaker no chunk label claimed, or a new number when none is left.
        assert_eq!(label(&link, "Speaker 2"), "Speaker 2");
        // The transcript keeps through "countries"; the chunk's "few things"
        // replaces the cut-off "fewer things".
        assert_eq!(link.transcript_keep_through, Some((0, 10)));
        assert_eq!(link.chunk_known_through, Some((0, 8)));
    }

    #[test]
    fn unaligned_chunk_continues_the_last_speaker() {
        let transcript = turns(&[
            ("Speaker 1", "question one"),
            ("Speaker 2", "a long answer about cement"),
        ]);
        let chunk = turns(&[
            ("Speaker 1", "completely different words here"),
            ("Speaker 2", "next question"),
        ]);
        let link = link_seam(&transcript, &chunk);

        assert!(!link.is_aligned());
        assert_eq!(label(&link, "Speaker 1"), "Speaker 2");
        assert_eq!(label(&link, "Speaker 2"), "Speaker 1");
        assert_eq!(link.chunk_known_through, None);
        assert_eq!(link.transcript_keep_through, None);
    }

    #[test]
    fn extra_chunk_speakers_get_new_numbers() {
        let transcript = turns(&[("Speaker 1", "only one person spoke so far in this talk")]);
        let chunk = turns(&[
            ("Speaker 1", "spoke so far in this talk"),
            ("Speaker 2", "hello"),
            ("Speaker 3", "hi"),
        ]);
        let link = link_seam(&transcript, &chunk);

        assert_eq!(label(&link, "Speaker 1"), "Speaker 1");
        assert_eq!(label(&link, "Speaker 2"), "Speaker 2");
        assert_eq!(label(&link, "Speaker 3"), "Speaker 3");
    }

    #[test]
    fn a_single_vote_does_not_link_a_label() {
        // "the" and "and" align alone; runs of one are dropped, so nothing
        // links and the seam falls back to continuity.
        let transcript = turns(&[("Speaker 2", "x the y and z")]);
        let chunk = turns(&[("Speaker 1", "p the q and r")]);
        let link = link_seam(&transcript, &chunk);
        assert!(!link.is_aligned());
        assert_eq!(label(&link, "Speaker 1"), "Speaker 2");
    }

    #[test]
    fn an_alignment_far_back_in_the_transcript_is_not_trusted() {
        let mut transcript_text = String::from("we buy from libya and egypt every year ");
        for index in 0..150 {
            transcript_text.push_str(&format!("word{index} "));
        }
        let transcript = turns(&[("Speaker 1", transcript_text.as_str())]);
        let chunk = turns(&[("Speaker 1", "we buy from libya and egypt every year")]);
        let link = link_seam(&transcript, &chunk);
        assert!(!link.is_aligned());
    }

    #[test]
    fn unknown_labels_neither_vote_nor_map() {
        let transcript = turns(&[("Speaker 1", "the shared words are right here now")]);
        let chunk = turns(&[("UNKNOWN", "the shared words are right here now")]);
        let link = link_seam(&transcript, &chunk);
        assert!(link.is_aligned());
        assert!(!link.labels.contains_key("UNKNOWN"));
    }

    #[test]
    fn first_chunk_keeps_its_labels() {
        let link = link_seam(&[], &turns(&[("Speaker 2", "a"), ("Speaker 1", "b")]));
        assert_eq!(label(&link, "Speaker 1"), "Speaker 1");
        assert_eq!(label(&link, "Speaker 2"), "Speaker 2");
    }

    #[test]
    fn truncation_and_dropping_work_on_word_positions() {
        let mut transcript = turns(&[("Speaker 1", "a b c"), ("Speaker 2", "d e f")]);
        truncate_turns_after(&mut transcript, (1, 0));
        assert_eq!(transcript, turns(&[("Speaker 1", "a b c"), ("Speaker 2", "d")]));
        truncate_turns_after(&mut transcript, (0, 2));
        assert_eq!(transcript, turns(&[("Speaker 1", "a b c")]));

        let chunk = turns(&[("Speaker 1", "a b"), ("Speaker 2", "c d")]);
        assert_eq!(
            drop_turn_words_through(chunk.clone(), (0, 1)),
            turns(&[("Speaker 2", "c d")])
        );
        assert_eq!(
            drop_turn_words_through(chunk, (1, 0)),
            turns(&[("Speaker 2", "d")])
        );
    }
}
