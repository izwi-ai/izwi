//! Split of `<think>…</think>` reasoning from a chat model's answer.
//!
//! Two conventions reach the server:
//! * the model writes its own opener (LFM2.5 Thinking): `<think>r</think>a`;
//! * the prompt already opened the block (Qwen3.5/3.6/3.8 with thinking on),
//!   so the output is `r</think>a`.
//!
//! The OpenAI routes split only on request (`reasoning_format: "parsed"`);
//! unsplit, they write back the opener the prompt held ([`PromptOpener`]).
//! The server's own generations (summaries, titles, memory) always split, so
//! reasoning never reaches stored text.

use crate::catalog::{ModelFamily, ModelVariant};

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// Whether a request's prompt opens the `<think>` block itself: chat families
/// whose template opens it when thinking is on (Qwen3.5/3.6/3.8). The LFM2.5
/// Thinking model writes its own opener.
pub fn reasoning_opened_in_prompt(variant: ModelVariant, enable_thinking: Option<bool>) -> bool {
    variant.family() != ModelFamily::Lfm2Chat
        && variant.chat_capabilities().is_some_and(|caps| {
            caps.supports_thinking && enable_thinking.unwrap_or(caps.default_thinking_enabled)
        })
}

/// Split a complete output into `(reasoning, content)`.
pub fn split_reasoning(text: &str, opened_in_prompt: bool) -> (Option<String>, String) {
    split_reasoning_before_calls(text, opened_in_prompt, None)
}

/// [`split_reasoning`] for an answer that may call tools: `call_opener` (the
/// family's tool-call marker) ends an unclosed reasoning block, so a call a
/// model writes without first closing `</think>` is still a call.
pub fn split_reasoning_before_calls(
    text: &str,
    opened_in_prompt: bool,
    call_opener: Option<&'static str>,
) -> (Option<String>, String) {
    let mut splitter = ReasoningSplitter::new(opened_in_prompt).with_call_opener(call_opener);
    let (mut reasoning, mut content) = splitter.push(text);
    let (tail_reasoning, tail_content) = splitter.finish();
    reasoning.push_str(&tail_reasoning);
    content.push_str(&tail_content);
    let reasoning = splitter.saw_reasoning.then(|| reasoning.trim().to_string());
    (reasoning, content)
}

/// Unsplit output as a client reads it: the `<think>` the prompt opened is
/// written back in front of the first text, so reasoning reads
/// `<think>…</think>` whichever side opened the block.
pub struct PromptOpener {
    owed: bool,
}

impl PromptOpener {
    pub fn new(opened_in_prompt: bool) -> Self {
        Self {
            owed: opened_in_prompt,
        }
    }

    /// One delta, or a whole output, with the opener in front if still owed.
    pub fn push(&mut self, delta: &str) -> String {
        if self.owed && !delta.is_empty() {
            self.owed = false;
            format!("{OPEN}{delta}")
        } else {
            delta.to_string()
        }
    }
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
    /// A tool-call marker that also ends the reasoning block.
    call_opener: Option<&'static str>,
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
            call_opener: None,
        }
    }

    /// Treat `call_opener` inside the reasoning block as its end: models
    /// sometimes start a tool call without closing `</think>`, and the call
    /// must reach the tool-call parser instead of the reasoning text.
    pub fn with_call_opener(mut self, call_opener: Option<&'static str>) -> Self {
        self.call_opener = call_opener;
        self
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
                    let close = self.pending.find(CLOSE);
                    let call = self
                        .call_opener
                        .and_then(|opener| self.pending.find(opener))
                        .filter(|&call| close.is_none_or(|close| call < close));
                    if let Some(at) = call {
                        // The call is the answer; the marker stays for the
                        // tool-call parser.
                        reasoning.push_str(&self.pending[..at]);
                        self.pending.drain(..at);
                        self.phase = Phase::Answer;
                        continue;
                    }
                    if let Some(at) = close {
                        reasoning.push_str(&self.pending[..at]);
                        self.pending = self.pending[at + CLOSE.len()..].to_string();
                        self.phase = Phase::AnswerStart;
                        continue;
                    }
                    let keep = partial_tag_suffix(&self.pending, CLOSE).max(
                        self.call_opener
                            .map_or(0, |opener| partial_tag_suffix(&self.pending, opener)),
                    );
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
    fn unsplit_output_gets_back_the_opener_the_prompt_held() {
        let mut opener = PromptOpener::new(true);
        let streamed: String = ["", "r", "</think>", "A"]
            .iter()
            .map(|delta| opener.push(delta))
            .collect();
        assert_eq!(streamed, "<think>r</think>A");
        assert_eq!(
            PromptOpener::new(false).push("<think>r</think>A"),
            "<think>r</think>A"
        );
        assert_eq!(PromptOpener::new(true).push(""), "");
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
    fn a_tool_call_ends_reasoning_the_model_did_not_close() {
        let opener = Some("<tool_call>");
        let text =
            "I should call it <too late.\n<tool_call>\n<function=f>\n</function>\n</tool_call>";
        let whole = split_reasoning_before_calls(text, true, opener);
        assert_eq!(
            whole,
            (
                Some("I should call it <too late.".to_string()),
                "<tool_call>\n<function=f>\n</function>\n</tool_call>".to_string()
            )
        );
        for chunk in 1..text.len() {
            let mut splitter = ReasoningSplitter::new(true).with_call_opener(opener);
            let (mut reasoning, mut content) = (String::new(), String::new());
            for piece in text.as_bytes().chunks(chunk) {
                let (r, c) = splitter.push(std::str::from_utf8(piece).unwrap());
                reasoning.push_str(&r);
                content.push_str(&c);
            }
            let (r, c) = splitter.finish();
            reasoning.push_str(&r);
            content.push_str(&c);
            assert_eq!(
                reasoning.trim(),
                whole.0.as_deref().unwrap(),
                "chunk {chunk}"
            );
            assert_eq!(content, whole.1, "chunk {chunk}");
        }
        // A closed block is unchanged, and so is a call mentioned after it.
        assert_eq!(
            split_reasoning_before_calls("r</think>\n\n<tool_call>x", true, opener),
            (Some("r".to_string()), "<tool_call>x".to_string())
        );
        // Without tools the marker is reasoning text.
        assert_eq!(
            split_reasoning("r <tool_call> r</think>A", true),
            (Some("r <tool_call> r".to_string()), "A".to_string())
        );
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
