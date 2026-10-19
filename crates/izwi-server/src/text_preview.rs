//! One-line plain-text previews of stored chat content for list views.

/// Maximum preview length in characters, before the ellipsis.
pub const PREVIEW_MAX_CHARS: usize = 160;

/// Reduce message content to the text a reader would see on one line:
/// reasoning in `<think>` blocks, code fences, heading/list/quote markers,
/// emphasis markers and link targets are removed, whitespace is collapsed,
/// and the result is cut to [`PREVIEW_MAX_CHARS`] characters.
pub fn plain_text_preview(content: &str) -> String {
    let visible = strip_thinking(content);
    let mut text = String::with_capacity(visible.len());
    let mut in_fence = false;
    for line in visible.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        let line = if in_fence {
            trimmed
        } else {
            strip_line_marker(trimmed)
        };
        text.push_str(line);
        text.push(' ');
    }
    let text = strip_inline_markdown(&text);
    truncate_chars(&collapse_whitespace(&text), PREVIEW_MAX_CHARS)
}

/// Drop reasoning: a leading close-only block (the template opened
/// `<think>` in the prompt) and every complete `<think>…</think>` span.
fn strip_thinking(content: &str) -> String {
    let lowered = content.to_ascii_lowercase();
    let mut rest = content;
    if let Some(close) = lowered.find("</think>") {
        let opens_first = lowered.find("<think>").is_some_and(|open| open < close);
        if !opens_first {
            rest = &content[close + "</think>".len()..];
        }
    }
    let mut output = String::with_capacity(rest.len());
    let mut remaining = rest;
    loop {
        let lowered = remaining.to_ascii_lowercase();
        let Some(open) = lowered.find("<think>") else {
            output.push_str(remaining);
            break;
        };
        output.push_str(&remaining[..open]);
        match lowered[open..].find("</think>") {
            Some(close) => remaining = &remaining[open + close + "</think>".len()..],
            // An unclosed block is still reasoning in progress.
            None => break,
        }
    }
    output
}

fn strip_line_marker(line: &str) -> &str {
    let heading = line.trim_start_matches('#');
    if heading.len() < line.len() && heading.starts_with(' ') {
        return heading.trim_start();
    }
    for marker in ["> ", "- ", "* ", "+ "] {
        if let Some(rest) = line.strip_prefix(marker) {
            return rest;
        }
    }
    if line == ">" {
        return "";
    }
    let digits = line.bytes().take_while(u8::is_ascii_digit).count();
    if digits > 0 {
        let rest = &line[digits..];
        if let Some(rest) = rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") ")) {
            return rest;
        }
    }
    line
}

/// Remove `**`, `__`, `~~` and backticks, and replace `[text](url)` and
/// `![alt](url)` with their text.
fn strip_inline_markdown(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut output = String::with_capacity(text.len());
    let mut index = 0;
    while index < chars.len() {
        let current = chars[index];
        let next = chars.get(index + 1).copied();
        match (current, next) {
            ('*', Some('*')) | ('_', Some('_')) | ('~', Some('~')) => {
                index += 2;
                continue;
            }
            ('`', _) => {
                index += 1;
                continue;
            }
            ('!', Some('[')) => {
                if let Some((label, end)) = link_at(&chars, index + 1) {
                    output.push_str(&label);
                    index = end;
                    continue;
                }
            }
            ('[', _) => {
                if let Some((label, end)) = link_at(&chars, index) {
                    output.push_str(&label);
                    index = end;
                    continue;
                }
            }
            _ => {}
        }
        output.push(current);
        index += 1;
    }
    output
}

/// Parse `[label](target)` starting at `start` (which holds `[`), returning
/// the label and the index just past the closing parenthesis.
fn link_at(chars: &[char], start: usize) -> Option<(String, usize)> {
    let close = (start + 1..chars.len()).find(|&i| chars[i] == ']')?;
    if chars.get(close + 1) != Some(&'(') {
        return None;
    }
    let end = (close + 2..chars.len()).find(|&i| chars[i] == ')')?;
    Some((chars[start + 1..close].iter().collect(), end + 1))
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    match text.char_indices().nth(max_chars) {
        Some((cut, _)) => format!("{}…", text[..cut].trim_end()),
        None => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{plain_text_preview, PREVIEW_MAX_CHARS};

    #[test]
    fn strips_markdown_to_reader_text() {
        assert_eq!(
            plain_text_preview(
                "**MLOps** stands for **Machine Learning Operations**. See [the docs](https://x.test)."
            ),
            "MLOps stands for Machine Learning Operations. See the docs."
        );
        assert_eq!(
            plain_text_preview("## Steps\n- Use `npm ci`\n1. Run __tests__\n> quoted"),
            "Steps Use npm ci Run tests quoted"
        );
        assert_eq!(
            plain_text_preview("Look: ![chart](a.png) and ~~old~~ text"),
            "Look: chart and old text"
        );
    }

    #[test]
    fn removes_reasoning_blocks() {
        assert_eq!(
            plain_text_preview("<think>plan the answer</think>The answer is 4."),
            "The answer is 4."
        );
        assert_eq!(
            plain_text_preview("implicit reasoning</think>\nVisible reply"),
            "Visible reply"
        );
        assert_eq!(
            plain_text_preview("Partial <think>still thinking"),
            "Partial"
        );
    }

    #[test]
    fn keeps_code_block_text_without_fences() {
        assert_eq!(
            plain_text_preview("Run this:\n```bash\nnpm test\n```"),
            "Run this: npm test"
        );
    }

    #[test]
    fn truncates_on_character_boundaries() {
        let long = "é".repeat(PREVIEW_MAX_CHARS + 20);
        let preview = plain_text_preview(&long);
        assert_eq!(preview.chars().count(), PREVIEW_MAX_CHARS + 1);
        assert!(preview.ends_with('…'));
    }

    #[test]
    fn leaves_plain_text_alone() {
        assert_eq!(
            plain_text_preview("snake_case and 2*3"),
            "snake_case and 2*3"
        );
    }
}
