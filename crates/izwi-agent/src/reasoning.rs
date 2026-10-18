//! Reasoning ("thinking") text a model emits inside `<think>` tags.

const OPEN_TAG: &str = "<think>";
const CLOSE_TAG: &str = "</think>";

/// `text` without its `<think>…</think>` blocks, trimmed. An unclosed block
/// drops everything after its opening tag. Conversation history stores this,
/// never the reasoning: a later prompt would otherwise re-read (and pay for)
/// every earlier turn's thinking.
pub fn strip_reasoning(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find(OPEN_TAG) {
        out.push_str(&rest[..start]);
        match rest[start + OPEN_TAG.len()..].find(CLOSE_TAG) {
            Some(end) => rest = &rest[start + OPEN_TAG.len() + end + CLOSE_TAG.len()..],
            None => {
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    out.trim().to_string()
}

/// Removes `<think>…</think>` blocks from text that arrives in pieces (a
/// streamed reply), with tags that may be split across pieces. Text that
/// could still begin a tag is held back until the next piece decides it.
#[derive(Debug, Default)]
pub struct ReasoningFilter {
    pending: String,
    in_reasoning: bool,
}

impl ReasoningFilter {
    /// Feed one piece; returns the visible text it completes.
    pub fn push(&mut self, piece: &str) -> String {
        self.pending.push_str(piece);
        let mut visible = String::new();
        loop {
            let tag = if self.in_reasoning {
                CLOSE_TAG
            } else {
                OPEN_TAG
            };
            if let Some(index) = self.pending.find(tag) {
                if !self.in_reasoning {
                    visible.push_str(&self.pending[..index]);
                }
                self.pending.drain(..index + tag.len());
                self.in_reasoning = !self.in_reasoning;
                continue;
            }
            // Keep the longest suffix that is a prefix of the tag.
            let keep = (1..tag.len())
                .rev()
                .find(|len| {
                    self.pending.len() >= *len
                        && self.pending.is_char_boundary(self.pending.len() - len)
                        && tag.starts_with(&self.pending[self.pending.len() - len..])
                })
                .unwrap_or(0);
            let emit = self.pending.len() - keep;
            if !self.in_reasoning {
                visible.push_str(&self.pending[..emit]);
            }
            self.pending.drain(..emit);
            return visible;
        }
    }

    /// The visible text still held back at the end of the stream.
    pub fn finish(&mut self) -> String {
        let rest = std::mem::take(&mut self.pending);
        if self.in_reasoning {
            String::new()
        } else {
            rest
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_reasoning_drops_blocks_and_unclosed_tails() {
        assert_eq!(strip_reasoning("<think>plan</think> Hello."), "Hello.");
        assert_eq!(
            strip_reasoning("A <think>x</think>B <think>y</think>C"),
            "A B C"
        );
        assert_eq!(strip_reasoning("Answer. <think>never closed"), "Answer.");
        assert_eq!(strip_reasoning("<think>\n\n</think>\n\nHi"), "Hi");
        assert_eq!(strip_reasoning("plain"), "plain");
    }

    /// Every way of cutting a reply into pieces filters to the same text as
    /// stripping it whole.
    #[test]
    fn reasoning_filter_matches_whole_text_stripping_at_every_split() {
        let reply =
            "<think>let me think</think>Sure. It costs 3 <b> units.<think>more</think> Bye é.";
        let expected = "Sure. It costs 3 <b> units. Bye é.";
        let chars: Vec<(usize, char)> = reply.char_indices().collect();
        for (first, _) in &chars {
            for (second, _) in chars.iter().filter(|(index, _)| index >= first) {
                let mut filter = ReasoningFilter::default();
                let mut out = filter.push(&reply[..*first]);
                out.push_str(&filter.push(&reply[*first..*second]));
                out.push_str(&filter.push(&reply[*second..]));
                out.push_str(&filter.finish());
                assert_eq!(out, expected, "split at {first}/{second}");
            }
        }
    }

    #[test]
    fn reasoning_filter_holds_back_only_a_possible_tag() {
        let mut filter = ReasoningFilter::default();
        assert_eq!(filter.push("Hi <th"), "Hi ");
        assert_eq!(filter.push("ere"), "<there");
        assert_eq!(filter.push("<think>secret"), "");
        assert_eq!(filter.finish(), "");
    }
}
