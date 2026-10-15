//! OpenAI `stop` strings for chat. They are applied where the executor
//! publishes detokenized text, so every chat family, streamed or not, ends at
//! the first stop string and never emits it.

use crate::engine::TokenLogprob;
use crate::models::registry::NativeChatDecodeStep;

/// Filters one request's detokenized text against its stop strings. Text
/// that could still be the start of a stop string is held back until a later
/// delta settles it, so a published delta never has to be retracted.
#[derive(Debug, Clone)]
pub(super) struct StopSequenceFilter {
    stops: Vec<String>,
    /// Text not yet published because it may begin a stop string.
    held: String,
    /// Logprob entries of steps whose text is still held.
    held_logprobs: Vec<TokenLogprob>,
    /// Every byte published so far.
    released: String,
    matched: bool,
}

impl StopSequenceFilter {
    /// A filter for `stops`, or `None` when there is no non-empty stop
    /// string to look for.
    pub(super) fn new(stops: &[String]) -> Option<Self> {
        let stops: Vec<String> = stops
            .iter()
            .filter(|stop| !stop.is_empty())
            .cloned()
            .collect();
        (!stops.is_empty()).then(|| Self {
            stops,
            held: String::new(),
            held_logprobs: Vec::new(),
            released: String::new(),
            matched: false,
        })
    }

    /// Feed one delta and return the text that is now safe to publish. After
    /// a stop string matches, everything from it on is discarded.
    fn push(&mut self, delta: &str) -> String {
        if self.matched {
            return String::new();
        }
        self.held.push_str(delta);
        // Held text never starts inside published text (the longest suffix
        // that could begin a stop string is always held), so a match can
        // only lie inside `held`.
        if let Some(start) = self
            .stops
            .iter()
            .filter_map(|stop| self.held.find(stop.as_str()))
            .min()
        {
            self.held.truncate(start);
            self.matched = true;
            return self.release(start);
        }
        let keep = self.longest_held_stop_prefix();
        self.release(self.held.len() - keep)
    }

    /// Byte length of the longest suffix of `held` that is a proper prefix
    /// of some stop string.
    fn longest_held_stop_prefix(&self) -> usize {
        let longest_stop = self.stops.iter().map(String::len).max().unwrap_or(0);
        let earliest = self
            .held
            .len()
            .saturating_sub(longest_stop.saturating_sub(1));
        self.held
            .char_indices()
            .map(|(index, _)| index)
            .filter(|&index| index >= earliest)
            .find(|&index| {
                let suffix = &self.held[index..];
                self.stops
                    .iter()
                    .any(|stop| stop.len() > suffix.len() && stop.starts_with(suffix))
            })
            .map_or(0, |index| self.held.len() - index)
    }

    fn release(&mut self, end: usize) -> String {
        let released: String = self.held.drain(..end).collect();
        self.released.push_str(&released);
        released
    }

    /// Apply the stop strings to one committed decode step. The step's delta
    /// becomes the text that is safe to publish; a matched stop string
    /// finishes the row with the text before it as its terminal text, and a
    /// natural finish publishes whatever was held. Returns the logprob
    /// entries to publish with the new delta: entries travel with the text
    /// that releases them, and the step's own list is left whole for the
    /// terminal output.
    pub(super) fn apply(&mut self, step: &mut NativeChatDecodeStep) -> Vec<TokenLogprob> {
        self.held_logprobs.extend_from_slice(&step.logprobs);
        let mut delta = self.push(&step.delta);
        if self.matched {
            step.finished = true;
            step.text = self.released.clone();
        } else if step.finished {
            let end = self.held.len();
            delta.push_str(&self.release(end));
        }
        step.delta = delta;
        if step.delta.is_empty() && !step.finished {
            Vec::new()
        } else {
            std::mem::take(&mut self.held_logprobs)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filter(stops: &[&str]) -> StopSequenceFilter {
        StopSequenceFilter::new(
            &stops
                .iter()
                .map(|stop| stop.to_string())
                .collect::<Vec<_>>(),
        )
        .expect("non-empty stops")
    }

    fn step(delta: &str, finished: bool) -> NativeChatDecodeStep {
        NativeChatDecodeStep {
            delta: delta.to_string(),
            text: format!("model terminal text {delta}"),
            tokens_generated: 1,
            input_tokens_committed: 1,
            finished,
            logprobs: Vec::new(),
        }
    }

    fn entry(token: &str) -> TokenLogprob {
        TokenLogprob {
            token: token.to_string(),
            logprob: -0.5,
            bytes: token.as_bytes().to_vec(),
            top_logprobs: Vec::new(),
        }
    }

    /// Feed `deltas` and return what each published.
    fn feed(filter: &mut StopSequenceFilter, deltas: &[&str]) -> Vec<String> {
        deltas.iter().map(|delta| filter.push(delta)).collect()
    }

    #[test]
    fn no_stop_strings_means_no_filter() {
        assert!(StopSequenceFilter::new(&[]).is_none());
        assert!(StopSequenceFilter::new(&[String::new()]).is_none());
    }

    #[test]
    fn text_before_the_stop_is_published_and_the_stop_is_not() {
        let mut stop = filter(&["END"]);
        assert_eq!(feed(&mut stop, &["hello ", "wENDorld"]), ["hello ", "w"]);
        assert!(stop.matched);
        assert_eq!(stop.released, "hello w");
        assert_eq!(stop.push("more"), "", "nothing is published after a match");
    }

    /// A stop string split across two deltas: the first delta's possible
    /// stop prefix is held, then dropped once the stop completes.
    #[test]
    fn a_stop_split_across_deltas_is_never_published() {
        let mut stop = filter(&["</answer>"]);
        assert_eq!(feed(&mut stop, &["42</ans", "wer> trailing"]), ["42", ""]);
        assert!(stop.matched);
        assert_eq!(stop.released, "42");
    }

    /// A held prefix that turns out not to be a stop is published later,
    /// unchanged, so the concatenated deltas equal the unfiltered text.
    #[test]
    fn held_text_is_released_when_it_stops_matching() {
        let mut stop = filter(&["STOP"]);
        assert_eq!(feed(&mut stop, &["a ST", "O", "x"]), ["a ", "", "STOx"]);
        assert!(!stop.matched);
        assert!(stop.held.is_empty());
        assert_eq!(stop.released, "a STOx");
    }

    #[test]
    fn the_earliest_of_several_stops_wins() {
        let mut stop = filter(&["two", "one"]);
        assert_eq!(stop.push("zero one two"), "zero ");
        let mut overlapping = filter(&["abc", "bc"]);
        assert_eq!(overlapping.push("xabcd"), "x");
    }

    #[test]
    fn multibyte_text_is_split_on_character_boundaries() {
        let mut stop = filter(&["é!"]);
        assert_eq!(feed(&mut stop, &["caféé", "!"]), ["café", ""]);
        assert_eq!(stop.released, "café");
        let mut stop = filter(&["©x"]);
        assert_eq!(feed(&mut stop, &["Ã©", "y"]), ["Ã", "©y"]);
    }

    #[test]
    fn a_matched_stop_finishes_the_row_with_the_text_before_it() {
        let mut stop = filter(&["END"]);
        let mut first = step("abc E", false);
        first.logprobs = vec![entry("abc"), entry(" E")];
        let published = stop.apply(&mut first);
        assert_eq!(first.delta, "abc ");
        assert!(!first.finished);
        assert_eq!(published.len(), 2, "logprobs travel with released text");

        let mut second = step("ND tail", false);
        second.logprobs = vec![entry("ND"), entry(" tail")];
        let published = stop.apply(&mut second);
        assert!(second.finished, "the stop string ends generation");
        assert_eq!(second.delta, "");
        assert_eq!(second.text, "abc ");
        assert_eq!(published.len(), 2);
        assert_eq!(second.logprobs.len(), 2, "the step's own list stays whole");
    }

    #[test]
    fn a_natural_finish_publishes_held_text_and_keeps_the_model_terminal_text() {
        let mut stop = filter(&["END"]);
        let mut first = step("abc E", false);
        first.logprobs = vec![entry("abc E")];
        stop.apply(&mut first);
        assert_eq!(first.delta, "abc ");

        let mut held = step("N", false);
        held.logprobs = vec![entry("N")];
        assert!(
            stop.apply(&mut held).is_empty(),
            "held text keeps its logprobs"
        );
        assert_eq!(held.delta, "");

        let mut last = step("", true);
        let published = stop.apply(&mut last);
        assert!(last.finished);
        assert_eq!(last.delta, "EN");
        assert_eq!(last.text, "model terminal text ");
        assert_eq!(published.len(), 1, "held entries are published at the end");
    }
}
