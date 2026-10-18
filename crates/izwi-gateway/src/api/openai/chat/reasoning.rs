//! Opt-in split of `<think>…</think>` reasoning into `reasoning_content`
//! (`reasoning_format: "parsed"`).
//!
//! Two conventions reach the server:
//! * the model writes its own opener (LFM2.5 Thinking): `<think>r</think>a`;
//! * the prompt already opened the block (Qwen3.5/3.6/3.8 with thinking on),
//!   so the output is `r</think>a`.
//!
//! By default nothing changes: reasoning stays inside `content`, which is what
//! the web UI parses.

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// Split a complete output into `(reasoning, content)`.
pub fn split_reasoning(text: &str, opened_in_prompt: bool) -> (Option<String>, String) {
    let mut splitter = ReasoningSplitter::new(opened_in_prompt);
    let (mut reasoning, mut content) = splitter.push(text);
    let (tail_reasoning, tail_content) = splitter.finish();
    reasoning.push_str(&tail_reasoning);
    content.push_str(&tail_content);
    let reasoning = splitter.saw_reasoning.then(|| reasoning.trim().to_string());
    (reasoning, content)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Nothing but whitespace so far; an opener may still arrive.
    Start,
    Reasoning,
    /// After `</think>`: leading whitespace before the answer is dropped.
    AnswerStart,
    Answer,
}

/// Incremental splitter for streamed deltas. Text that could still be the
/// beginning of a tag is held back until the next delta decides it.
pub struct ReasoningSplitter {
    phase: Phase,
    pending: String,
    saw_reasoning: bool,
}

impl ReasoningSplitter {
    pub fn new(opened_in_prompt: bool) -> Self {
        Self {
            phase: if opened_in_prompt {
                Phase::Reasoning
            } else {
                Phase::Start
            },
            pending: String::new(),
            saw_reasoning: opened_in_prompt,
        }
    }

    /// Feed one delta; returns the `(reasoning, content)` text now final.
    pub fn push(&mut self, delta: &str) -> (String, String) {
        self.pending.push_str(delta);
        let mut reasoning = String::new();
        let mut content = String::new();
        loop {
            match self.phase {
                Phase::Start => {
                    let trimmed = self.pending.trim_start();
                    if trimmed.is_empty()
                        || (trimmed.len() < OPEN.len() && OPEN.starts_with(trimmed))
                    {
                        break;
                    }
                    if let Some(rest) = trimmed.strip_prefix(OPEN) {
                        self.pending = rest.to_string();
                        self.phase = Phase::Reasoning;
                        self.saw_reasoning = true;
                    } else {
                        self.phase = Phase::Answer;
                    }
                }
                Phase::Reasoning => {
                    if let Some(at) = self.pending.find(CLOSE) {
                        reasoning.push_str(&self.pending[..at]);
                        self.pending = self.pending[at + CLOSE.len()..].to_string();
                        self.phase = Phase::AnswerStart;
                        continue;
                    }
                    let keep = partial_tag_suffix(&self.pending, CLOSE);
                    let emit = self.pending.len() - keep;
                    reasoning.push_str(&self.pending[..emit]);
                    self.pending.drain(..emit);
                    break;
                }
                Phase::AnswerStart => {
                    let trimmed = self.pending.trim_start();
                    if trimmed.is_empty() {
                        self.pending.clear();
                        break;
                    }
                    self.pending = trimmed.to_string();
                    self.phase = Phase::Answer;
                }
                Phase::Answer => {
                    content.push_str(&self.pending);
                    self.pending.clear();
                    break;
                }
            }
        }
        (reasoning, content)
    }

    /// Flush what is held back at the end of the output.
    pub fn finish(&mut self) -> (String, String) {
        let pending = std::mem::take(&mut self.pending);
        match self.phase {
            Phase::Reasoning => (pending, String::new()),
            Phase::Start | Phase::Answer => (String::new(), pending),
            Phase::AnswerStart => (String::new(), String::new()),
        }
    }
}

/// Length of the longest suffix of `text` that is a proper prefix of `tag`.
fn partial_tag_suffix(text: &str, tag: &str) -> usize {
    (1..tag.len())
        .rev()
        .find(|&len| text.ends_with(&tag[..len]))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_a_model_written_think_block() {
        assert_eq!(
            split_reasoning("<think>\nr1\nr2\n</think>\n\nAnswer.", false),
            (Some("r1\nr2".to_string()), "Answer.".to_string())
        );
        // A direct answer has no reasoning.
        assert_eq!(
            split_reasoning("Paris.", false),
            (None, "Paris.".to_string())
        );
        // A block cut off by max_tokens is all reasoning.
        assert_eq!(
            split_reasoning("<think>still going", false),
            (Some("still going".to_string()), String::new())
        );
    }

    #[test]
    fn splits_a_block_the_prompt_opened() {
        assert_eq!(
            split_reasoning("r</think>\n\nA", true),
            (Some("r".to_string()), "A".to_string())
        );
    }

    #[test]
    fn streamed_tags_split_across_deltas_match_the_whole_text() {
        let text = "<think>reason <b> here</think>\n\nThe answer </thi is here.";
        for chunk in 1..=text.len() {
            let mut splitter = ReasoningSplitter::new(false);
            let (mut reasoning, mut content) = (String::new(), String::new());
            let mut at = 0;
            while at < text.len() {
                let end = (at + chunk).min(text.len());
                let (r, c) = splitter.push(&text[at..end]);
                reasoning.push_str(&r);
                content.push_str(&c);
                at = end;
            }
            let (r, c) = splitter.finish();
            reasoning.push_str(&r);
            content.push_str(&c);
            assert_eq!(reasoning, "reason <b> here", "chunk {chunk}");
            assert_eq!(content, "The answer </thi is here.", "chunk {chunk}");
        }
    }

    #[test]
    fn a_partial_opener_that_turns_out_to_be_text_is_content() {
        let mut splitter = ReasoningSplitter::new(false);
        assert_eq!(splitter.push("<thi"), (String::new(), String::new()));
        assert_eq!(
            splitter.push("s is html"),
            (String::new(), "<this is html".to_string())
        );
    }
}
