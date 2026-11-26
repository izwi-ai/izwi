//! The model's reply, turned from its own message tokens into the text the
//! serving stack reads.
//!
//! The model answers in messages: `<|message_model|>`, an optional name, a
//! content type token, text, `<|end_message|>`; the turn ends with
//! `<|content_model_end_sampling|>`. The server splits reasoning and tool
//! calls out of plain text by markers, so this writes:
//!
//! * a thinking message as `<think>…</think>`;
//! * a text message as its text;
//! * a tool call (`{"name": f, "args": {…}}`) as a `<tool_call>` block in
//!   the Hermes JSON form, `{"name": f, "arguments": {…}}`.
//!
//! Text streams as it is decoded. A tool call is held until its message
//! ends, because its JSON is rewritten.
//!
//! Every token advances the state, whether or not it produces text.

use serde_json::Value;

use super::prompt::{
    special_id, CONTENT_IMAGE, CONTENT_TEXT, CONTENT_THINKING, CONTENT_TOOL_ERROR,
    CONTENT_TOOL_JSON, CONTENT_TOOL_TEXT, CONTENT_XML, END_MESSAGE, END_SAMPLING, MESSAGE_MODEL,
    MESSAGE_SYSTEM, MESSAGE_TOOL, MESSAGE_USER, THINK_CLOSE, THINK_OPEN, TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
};
use crate::error::Result;
use crate::tokenizer::{IncrementalDecoder, Tokenizer};

/// The special tokens a reply is made of.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inkling2ReplyTokens {
    pub message_model: u32,
    pub other_roles: [u32; 3],
    pub text: u32,
    pub thinking: u32,
    pub tool_json: u32,
    /// Content types shown as plain text.
    pub plain: [u32; 4],
    pub end_message: u32,
    pub end_sampling: u32,
}

impl Inkling2ReplyTokens {
    pub fn load(tokenizer: &Tokenizer) -> Result<Self> {
        let id = |token: &str| special_id(tokenizer, token);
        Ok(Self {
            message_model: id(MESSAGE_MODEL)?,
            other_roles: [id(MESSAGE_USER)?, id(MESSAGE_SYSTEM)?, id(MESSAGE_TOOL)?],
            text: id(CONTENT_TEXT)?,
            thinking: id(CONTENT_THINKING)?,
            tool_json: id(CONTENT_TOOL_JSON)?,
            plain: [
                id(CONTENT_TOOL_TEXT)?,
                id(CONTENT_TOOL_ERROR)?,
                id(CONTENT_XML)?,
                id(CONTENT_IMAGE)?,
            ],
            end_message: id(END_MESSAGE)?,
            end_sampling: id(END_SAMPLING)?,
        })
    }

    /// Whether `token` ends the turn: the end-of-sampling token, or another
    /// speaker's header (the model must not write a turn that is not its own).
    pub fn ends_turn(&self, token: u32) -> bool {
        token == self.end_sampling || self.other_roles.contains(&token)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Section {
    /// Between messages.
    Idle,
    /// After `<|message_model|>`, before the content type (the optional
    /// name is not shown).
    Header,
    Thinking,
    Text,
    /// The JSON of a tool call, held until the message ends.
    ToolCall(String),
}

/// Streaming state of one reply.
#[derive(Debug, Clone)]
pub struct Inkling2Reply {
    tokens: Inkling2ReplyTokens,
    section: Section,
    /// Decoder of the open message's text.
    decoder: IncrementalDecoder,
    /// Everything emitted so far.
    text: String,
    finished: bool,
}

impl Inkling2Reply {
    /// A reply that starts at the model's own header. `forced_call` is the
    /// JSON the prompt already wrote for a forced tool call, if any: the
    /// reply continues inside that call.
    pub fn new(tokens: Inkling2ReplyTokens, forced_call: Option<&str>) -> Self {
        Self {
            tokens,
            section: match forced_call {
                Some(prefix) => Section::ToolCall(prefix.to_string()),
                None => Section::Idle,
            },
            decoder: IncrementalDecoder::new(false),
            text: String::new(),
            finished: false,
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn finished(&self) -> bool {
        self.finished
    }

    fn emit(&mut self, delta: &str, out: &mut String) {
        self.text.push_str(delta);
        out.push_str(delta);
    }

    fn close_message(&mut self, tokenizer: &Tokenizer, out: &mut String) -> Result<()> {
        let tail = tokenizer.finish_incremental_decode(&mut self.decoder)?;
        self.decoder = IncrementalDecoder::new(false);
        match std::mem::replace(&mut self.section, Section::Idle) {
            Section::Thinking => {
                self.emit(&tail, out);
                self.emit(THINK_CLOSE, out);
            }
            Section::Text => self.emit(&tail, out),
            Section::ToolCall(mut json) => {
                json.push_str(&tail);
                let block = tool_call_block(&json);
                self.emit(&block, out);
            }
            Section::Idle | Section::Header => {}
        }
        Ok(())
    }

    /// Consumes one generated token and returns the text it adds.
    pub fn push(&mut self, tokenizer: &Tokenizer, token: u32) -> Result<String> {
        let mut out = String::new();
        if self.finished {
            return Ok(out);
        }
        let t = &self.tokens;
        if t.ends_turn(token) {
            self.close_message(tokenizer, &mut out)?;
            self.finished = true;
        } else if token == t.end_message {
            self.close_message(tokenizer, &mut out)?;
        } else if token == t.message_model {
            // A header inside an open message closes it.
            self.close_message(tokenizer, &mut out)?;
            self.section = Section::Header;
        } else if token == t.thinking {
            self.close_message(tokenizer, &mut out)?;
            self.section = Section::Thinking;
            self.emit(THINK_OPEN, &mut out);
        } else if token == t.text || t.plain.contains(&token) {
            self.close_message(tokenizer, &mut out)?;
            self.section = Section::Text;
        } else if token == t.tool_json {
            self.close_message(tokenizer, &mut out)?;
            self.section = Section::ToolCall(String::new());
        } else {
            match &self.section {
                // The name between the header and the content type, or
                // stray text between messages: consumed, not shown.
                Section::Header | Section::Idle => {}
                Section::Thinking | Section::Text => {
                    let delta = tokenizer.decode_incrementally(&mut self.decoder, token)?;
                    self.emit(&delta, &mut out);
                }
                Section::ToolCall(_) => {
                    let delta = tokenizer.decode_incrementally(&mut self.decoder, token)?;
                    if let Section::ToolCall(json) = &mut self.section {
                        json.push_str(&delta);
                    }
                }
            }
        }
        Ok(out)
    }

    /// Ends a reply that stopped without its end token (a token limit).
    pub fn finish(&mut self, tokenizer: &Tokenizer) -> Result<String> {
        let mut out = String::new();
        if !self.finished {
            self.close_message(tokenizer, &mut out)?;
            self.finished = true;
        }
        Ok(out)
    }
}

/// A native call body as a Hermes `<tool_call>` block. A body that is not
/// the expected JSON is passed through unchanged inside the block, for the
/// server's parser to report.
pub fn tool_call_block(native: &str) -> String {
    let body = serde_json::from_str::<Value>(native.trim())
        .ok()
        .and_then(|value| {
            let name = value.get("name")?.as_str()?.to_string();
            let arguments = match value.get("args").or_else(|| value.get("arguments")) {
                Some(Value::Object(map)) => Value::Object(map.clone()),
                None | Some(Value::Null) => Value::Object(serde_json::Map::new()),
                Some(_) => return None,
            };
            let mut call = serde_json::Map::new();
            call.insert("name".into(), Value::String(name));
            call.insert("arguments".into(), arguments);
            serde_json::to_string(&Value::Object(call)).ok()
        })
        .unwrap_or_else(|| native.trim().to_string());
    format!("{TOOL_CALL_OPEN}\n{body}\n{TOOL_CALL_CLOSE}")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A byte-level test tokenizer: ids 0..=255 are bytes, then the special
    /// tokens.
    pub(crate) fn test_tokenizer() -> Tokenizer {
        let specials = [
            MESSAGE_USER,
            MESSAGE_MODEL,
            MESSAGE_SYSTEM,
            MESSAGE_TOOL,
            CONTENT_TEXT,
            CONTENT_IMAGE,
            CONTENT_THINKING,
            CONTENT_XML,
            CONTENT_TOOL_JSON,
            CONTENT_TOOL_TEXT,
            CONTENT_TOOL_ERROR,
            END_MESSAGE,
            END_SAMPLING,
        ];
        let mut vocab = serde_json::Map::new();
        for byte in 0u32..256 {
            // The ByteLevel alphabet: each byte as one character.
            let ch = byte_level_char(byte as u8);
            vocab.insert(ch.to_string(), Value::from(byte));
        }
        let added = specials
            .iter()
            .enumerate()
            .map(|(index, token)| {
                serde_json::json!({
                    "id": 256 + index, "content": token, "single_word": false, "lstrip": false,
                    "rstrip": false, "normalized": false, "special": true
                })
            })
            .collect::<Vec<_>>();
        for (index, token) in specials.iter().enumerate() {
            vocab.insert(token.to_string(), Value::from(256 + index as u32));
        }
        let json = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": null,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false},
            "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                      "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false,
                      "ignore_merges": false, "vocab": vocab, "merges": []}
        });
        Tokenizer::from_hf_json_bytes(&serde_json::to_vec(&json).unwrap()).unwrap()
    }

    /// The character the ByteLevel pre-tokenizer maps a byte to (GPT-2's
    /// table).
    fn byte_level_char(byte: u8) -> char {
        let printable =
            (b'!'..=b'~').contains(&byte) || (0xA1..=0xAC).contains(&byte) || byte >= 0xAE;
        if printable {
            char::from_u32(byte as u32).unwrap()
        } else {
            let shifted = (0u32..256)
                .filter(|b| {
                    let b = *b as u8;
                    !((b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || b >= 0xAE)
                })
                .position(|b| b == byte as u32)
                .unwrap() as u32;
            char::from_u32(256 + shifted).unwrap()
        }
    }

    fn ids(tokenizer: &Tokenizer, pieces: &[&str]) -> Vec<u32> {
        let mut out = Vec::new();
        for piece in pieces {
            match tokenizer.token_to_id(piece) {
                Some(id) if piece.starts_with("<|") => out.push(id),
                _ => out.extend(tokenizer.encode(piece).unwrap()),
            }
        }
        out
    }

    fn run(pieces: &[&str]) -> (String, Vec<String>, bool) {
        let tokenizer = test_tokenizer();
        let tokens = Inkling2ReplyTokens::load(&tokenizer).unwrap();
        let mut reply = Inkling2Reply::new(tokens, None);
        let mut deltas = Vec::new();
        for id in ids(&tokenizer, pieces) {
            deltas.push(reply.push(&tokenizer, id).unwrap());
        }
        let finished = reply.finished();
        deltas.push(reply.finish(&tokenizer).unwrap());
        assert_eq!(deltas.concat(), reply.text());
        (reply.text().to_string(), deltas, finished)
    }

    #[test]
    fn thinking_text_and_end() {
        let (text, deltas, finished) = run(&[
            MESSAGE_MODEL,
            CONTENT_THINKING,
            "Hmm.",
            END_MESSAGE,
            MESSAGE_MODEL,
            CONTENT_TEXT,
            "Four.",
            END_MESSAGE,
            END_SAMPLING,
        ]);
        assert_eq!(text, "<think>Hmm.</think>Four.");
        assert!(finished);
        // Text streams token by token.
        assert!(deltas.iter().filter(|delta| !delta.is_empty()).count() > 6);
    }

    #[test]
    fn a_tool_call_becomes_a_hermes_block() {
        let (text, deltas, _) = run(&[
            MESSAGE_MODEL,
            CONTENT_TEXT,
            "Checking.",
            END_MESSAGE,
            MESSAGE_MODEL,
            "get_weather",
            CONTENT_TOOL_JSON,
            "{\"name\":\"get_weather\",\"args\":{\"city\":\"Harare\"}}",
            END_MESSAGE,
            END_SAMPLING,
        ]);
        assert_eq!(
            text,
            "Checking.<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Harare\"}}\n</tool_call>"
        );
        // The call arrives whole, and the header's name is not shown.
        assert_eq!(
            deltas
                .iter()
                .filter(|delta| delta.contains("<tool_call>"))
                .count(),
            1
        );
        assert!(!text.contains("Checking.get_weather"));
    }

    #[test]
    fn a_forced_call_continues_the_prompt() {
        let tokenizer = test_tokenizer();
        let tokens = Inkling2ReplyTokens::load(&tokenizer).unwrap();
        let mut reply = Inkling2Reply::new(tokens, Some("{\"name\":\"f\",\"args\":"));
        for id in ids(&tokenizer, &["{\"k\":1}}", END_MESSAGE, END_SAMPLING]) {
            reply.push(&tokenizer, id).unwrap();
        }
        assert_eq!(
            reply.text(),
            "<tool_call>\n{\"name\":\"f\",\"arguments\":{\"k\":1}}\n</tool_call>"
        );
    }

    #[test]
    fn split_characters_stream_whole() {
        let (text, deltas, _) = run(&[MESSAGE_MODEL, CONTENT_TEXT, "27 °C é", END_MESSAGE]);
        assert_eq!(text, "27 °C é");
        assert!(deltas.iter().all(|delta| !delta.contains('\u{FFFD}')));
    }

    #[test]
    fn a_reply_cut_short_closes_what_is_open() {
        let (text, _, finished) = run(&[MESSAGE_MODEL, CONTENT_THINKING, "half a th"]);
        assert_eq!(text, "<think>half a th</think>");
        assert!(!finished);
        let (text, _, _) = run(&[
            MESSAGE_MODEL,
            "f",
            CONTENT_TOOL_JSON,
            "{\"name\":\"f\",\"ar",
        ]);
        assert_eq!(text, "<tool_call>\n{\"name\":\"f\",\"ar\n</tool_call>");
    }

    #[test]
    fn another_speakers_header_ends_the_turn() {
        let (text, _, finished) = run(&[
            MESSAGE_MODEL,
            CONTENT_TEXT,
            "Hi",
            END_MESSAGE,
            MESSAGE_USER,
            CONTENT_TEXT,
            "x",
        ]);
        assert_eq!(text, "Hi");
        assert!(finished);
    }
}
