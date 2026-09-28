//! DS9.2: hand-rolled JSON grammar FSM for constrained decoding.
//!
//! The machine tracks one JSON value's parse state. A token is a legal
//! continuation in a state when feeding its surface text through the machine
//! succeeds; per-state masks over the vocabulary are derived from that
//! predicate and cached by the caller. The FSM is deliberately complete over
//! RFC 8259's value grammar (including `\u` hex escapes and the leading-zero
//! rule) so that greedy generation under its masks parses as JSON by
//! construction.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum JsonState {
    /// Expecting any JSON value (or insignificant whitespace).
    ValueStart,
    /// Directly after `{`: a key string or the closing brace.
    ObjectKeyOrEnd,
    /// After a comma inside an object: a key string is required.
    ObjectKeyRequired,
    /// Inside a key string.
    KeyString,
    /// Directly after a backslash inside a key string.
    KeyEscape,
    /// `\u` hex digits inside a key string: how many are still missing.
    KeyUnicode(u8),
    /// After a key string: the separating colon.
    ObjectColon,
    /// Inside an object after a value: a comma or the closing brace.
    ObjectCommaOrEnd,
    /// Directly after `[`: a value or the closing bracket.
    ArrayValueOrEnd,
    /// Inside an array after a value: a comma or the closing bracket.
    ArrayCommaOrEnd,
    /// Inside a value string.
    StringBody,
    /// Directly after a backslash inside a value string.
    StringEscape,
    /// `\u` hex digits inside a value string: how many are still missing.
    StringUnicode(u8),
    /// Inside a number's integer part (no sign, nonzero lead allowed).
    NumberInt,
    /// After a `0` or `-0`: only fraction, exponent, or end may follow.
    NumberZero,
    /// After the `-` sign.
    NumberSign,
    /// After a decimal point.
    NumberFracStart,
    /// Inside a fractional part.
    NumberFrac,
    /// After `e`/`E` before the exponent digits (sign optional).
    NumberExpSign,
    /// Inside an exponent.
    NumberExp,
    /// Partial `true` literal: how many letters are still missing.
    TrueLiteral(u8),
    /// Partial `false` literal.
    FalseLiteral(u8),
    /// Partial `null` literal.
    NullLiteral(u8),
    /// The JSON value is complete; only insignificant whitespace remains.
    Complete,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum JsonContainer {
    Object,
    Array,
}

/// Cache key for per-state masks: the state plus the container the machine
/// would resolve against. Only the stack top and emptiness affect whether a
/// token is allowed, so this key fully determines mask equivalence.
pub(crate) type JsonStateKey = (JsonState, Option<JsonContainer>);

#[derive(Debug, Clone)]
pub struct JsonGrammarMachine {
    state: JsonState,
    stack: Vec<JsonContainer>,
    root_seen_value: bool,
}

impl Default for JsonGrammarMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonGrammarMachine {
    pub fn new() -> Self {
        Self {
            state: JsonState::ValueStart,
            stack: Vec::new(),
            root_seen_value: false,
        }
    }

    /// Whether the document is complete; callers usually restrict the mask to
    /// whitespace and stop tokens in this state.
    pub fn is_complete(&self) -> bool {
        self.state == JsonState::Complete
    }

    /// Whether the machine may stand at the END of the input: complete, or
    /// inside a number (numbers end implicitly without a delimiter).
    pub fn is_document_terminal(&self) -> bool {
        matches!(
            self.state,
            JsonState::Complete
                | JsonState::NumberInt
                | JsonState::NumberZero
                | JsonState::NumberFrac
                | JsonState::NumberExp
        )
    }

    /// The mask-cache key for the current machine position.
    pub fn state_key(&self) -> JsonStateKey {
        (self.state, self.stack.last().copied())
    }

    /// Feed one token's surface text. Fails when the text is not a legal
    /// continuation; the machine is left untouched on failure.
    pub fn feed(&mut self, text: &str) -> Result<(), GrammarError> {
        let mut probe = self.clone();
        for ch in text.chars() {
            probe.step(ch)?;
        }
        *self = probe;
        Ok(())
    }

    fn step(&mut self, ch: char) -> Result<(), GrammarError> {
        use JsonState as S;
        let (state, stack) = (&mut self.state, &mut self.stack);
        let next = match *state {
            S::ValueStart => match ch {
                ' ' | '\t' | '\n' | '\r' => S::ValueStart,
                _ => value_char(ch, stack)?,
            },
            S::ObjectKeyOrEnd | S::ObjectKeyRequired => match ch {
                ' ' | '\t' | '\n' | '\r' => *state,
                '"' => S::KeyString,
                '}' if *state == S::ObjectKeyOrEnd => {
                    end_container(stack, &mut self.root_seen_value)?
                }
                _ => return Err(GrammarError),
            },
            S::KeyString => match ch {
                '"' => S::ObjectColon,
                '\\' => S::KeyEscape,
                _ => S::KeyString,
            },
            S::KeyEscape => match ch {
                '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => S::KeyString,
                'u' => S::KeyUnicode(4),
                _ => return Err(GrammarError),
            },
            S::KeyUnicode(remaining) => unicode_digit(ch, remaining, S::KeyString)?,
            S::ObjectColon => match ch {
                ' ' | '\t' | '\n' | '\r' => S::ObjectColon,
                ':' => S::ValueStart,
                _ => return Err(GrammarError),
            },
            S::ObjectCommaOrEnd => match ch {
                ' ' | '\t' | '\n' | '\r' => S::ObjectCommaOrEnd,
                ',' => S::ObjectKeyRequired,
                '}' => end_container(stack, &mut self.root_seen_value)?,
                _ => return Err(GrammarError),
            },
            S::ArrayValueOrEnd => match ch {
                ']' => end_container(stack, &mut self.root_seen_value)?,
                _ => value_char(ch, stack)?,
            },
            S::ArrayCommaOrEnd => match ch {
                ' ' | '\t' | '\n' | '\r' => S::ArrayCommaOrEnd,
                ',' => S::ValueStart,
                ']' => end_container(stack, &mut self.root_seen_value)?,
                _ => return Err(GrammarError),
            },
            S::StringBody => match ch {
                '"' => end_value(stack, &mut self.root_seen_value),
                '\\' => S::StringEscape,
                _ => S::StringBody,
            },
            S::StringEscape => match ch {
                '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => S::StringBody,
                'u' => S::StringUnicode(4),
                _ => return Err(GrammarError),
            },
            S::StringUnicode(remaining) => unicode_digit(ch, remaining, S::StringBody)?,
            S::NumberInt => match ch {
                '0'..='9' => S::NumberInt,
                '.' => S::NumberFracStart,
                'e' | 'E' => S::NumberExpSign,
                _ => number_end(ch, stack, &mut self.root_seen_value)?,
            },
            S::NumberZero => match ch {
                '.' => S::NumberFracStart,
                'e' | 'E' => S::NumberExpSign,
                _ => number_end(ch, stack, &mut self.root_seen_value)?,
            },
            S::NumberSign => match ch {
                '0' => S::NumberZero,
                '1'..='9' => S::NumberInt,
                _ => return Err(GrammarError),
            },
            S::NumberFracStart => {
                if ch.is_ascii_digit() {
                    S::NumberFrac
                } else {
                    return Err(GrammarError);
                }
            }
            S::NumberFrac => match ch {
                '0'..='9' => S::NumberFrac,
                'e' | 'E' => S::NumberExpSign,
                _ => number_end(ch, stack, &mut self.root_seen_value)?,
            },
            S::NumberExpSign => match ch {
                '+' | '-' => S::NumberExp,
                '0'..='9' => S::NumberExp,
                _ => return Err(GrammarError),
            },
            S::NumberExp => match ch {
                '0'..='9' => S::NumberExp,
                _ => number_end(ch, stack, &mut self.root_seen_value)?,
            },
            S::TrueLiteral(missing) => {
                literal_step(ch, "true", missing, stack, &mut self.root_seen_value)?
            }
            S::FalseLiteral(missing) => {
                literal_step(ch, "false", missing, stack, &mut self.root_seen_value)?
            }
            S::NullLiteral(missing) => {
                literal_step(ch, "null", missing, stack, &mut self.root_seen_value)?
            }
            S::Complete => match ch {
                ' ' | '\t' | '\n' | '\r' => S::Complete,
                _ => return Err(GrammarError),
            },
        };
        *state = next;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrammarError;

fn unicode_digit(ch: char, remaining: u8, back: JsonState) -> Result<JsonState, GrammarError> {
    if !ch.is_ascii_hexdigit() {
        return Err(GrammarError);
    }
    if remaining == 1 {
        Ok(back)
    } else {
        Ok(match back {
            JsonState::KeyString => JsonState::KeyUnicode(remaining - 1),
            _ => JsonState::StringUnicode(remaining - 1),
        })
    }
}

/// The value-start character set, shared by `ValueStart` and the array
/// value-or-end position.
fn value_char(ch: char, stack: &mut Vec<JsonContainer>) -> Result<JsonState, GrammarError> {
    Ok(match ch {
        '{' => {
            push_object(stack)?;
            JsonState::ObjectKeyOrEnd
        }
        '[' => {
            push_array(stack)?;
            JsonState::ArrayValueOrEnd
        }
        '"' => JsonState::StringBody,
        '0' => JsonState::NumberZero,
        '1'..='9' => JsonState::NumberInt,
        '-' => JsonState::NumberSign,
        't' => JsonState::TrueLiteral(3),
        'f' => JsonState::FalseLiteral(4),
        'n' => JsonState::NullLiteral(3),
        _ => return Err(GrammarError),
    })
}

fn push_object(stack: &mut Vec<JsonContainer>) -> Result<(), GrammarError> {
    if stack.len() >= 32 {
        // Bound nesting so pathological prompts cannot grow the stack.
        return Err(GrammarError);
    }
    stack.push(JsonContainer::Object);
    Ok(())
}

fn push_array(stack: &mut Vec<JsonContainer>) -> Result<(), GrammarError> {
    if stack.len() >= 32 {
        // Bound nesting so pathological prompts cannot grow the stack.
        return Err(GrammarError);
    }
    stack.push(JsonContainer::Array);
    Ok(())
}

/// Resolve after a VALUE finishes (string close, literal end): the container
/// it sits in dictates the next state.
fn end_value(stack: &mut Vec<JsonContainer>, root_seen_value: &mut bool) -> JsonState {
    *root_seen_value = true;
    match stack.last() {
        Some(JsonContainer::Object) => JsonState::ObjectCommaOrEnd,
        Some(JsonContainer::Array) => JsonState::ArrayCommaOrEnd,
        None => JsonState::Complete,
    }
}

/// Numbers end without an explicit terminator: the delimiter that ends them
/// must itself be processed (`,`/`}`/`]`), while whitespace just ends the
/// number.
fn number_end(
    ch: char,
    stack: &mut Vec<JsonContainer>,
    root_seen_value: &mut bool,
) -> Result<JsonState, GrammarError> {
    match ch {
        ' ' | '\t' | '\n' | '\r' => Ok(end_value(stack, root_seen_value)),
        ',' => match stack.last() {
            Some(JsonContainer::Object) => Ok(JsonState::ObjectKeyRequired),
            Some(JsonContainer::Array) => Ok(JsonState::ValueStart),
            None => Err(GrammarError),
        },
        '}' if matches!(stack.last(), Some(JsonContainer::Object)) => {
            end_container(stack, root_seen_value)
        }
        ']' if matches!(stack.last(), Some(JsonContainer::Array)) => {
            end_container(stack, root_seen_value)
        }
        _ => Err(GrammarError),
    }
}

fn literal_step(
    ch: char,
    literal: &str,
    missing: u8,
    stack: &mut Vec<JsonContainer>,
    root_seen_value: &mut bool,
) -> Result<JsonState, GrammarError> {
    let expected = literal
        .chars()
        .nth(literal.len() - usize::from(missing))
        .ok_or(GrammarError)?;
    if ch != expected {
        return Err(GrammarError);
    }
    if missing == 1 {
        Ok(end_value(stack, root_seen_value))
    } else if ch == 't' {
        Ok(JsonState::TrueLiteral(missing - 1))
    } else if ch == 'f' {
        Ok(JsonState::FalseLiteral(missing - 1))
    } else if ch == 'n' {
        Ok(JsonState::NullLiteral(missing - 1))
    } else {
        // Continue the same literal with one fewer missing letter.
        Ok(match literal {
            "true" => JsonState::TrueLiteral(missing - 1),
            "false" => JsonState::FalseLiteral(missing - 1),
            _ => JsonState::NullLiteral(missing - 1),
        })
    }
}

/// Resolve after a CONTAINER closes: pop it and treat the container as the
/// finished value.
fn end_container(
    stack: &mut Vec<JsonContainer>,
    root_seen_value: &mut bool,
) -> Result<JsonState, GrammarError> {
    stack.pop().ok_or(GrammarError)?;
    Ok(end_value(stack, root_seen_value))
}

/// DS9.2: per-state vocabulary masks, built lazily from token surfaces and
/// cached per (state, container-top) key. Cheap to clone; the cache is shared
/// across a sampler's clones.
pub struct JsonGrammarMasks {
    cache: Mutex<HashMap<JsonStateKey, Arc<Vec<bool>>>>,
}

impl Default for JsonGrammarMasks {
    fn default() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }
}

impl Clone for JsonGrammarMasks {
    fn clone(&self) -> Self {
        Self {
            cache: Mutex::new(self.cache.lock().expect("grammar mask cache").clone()),
        }
    }
}

impl JsonGrammarMasks {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build (or fetch) the allowed-token mask for the machine's state.
    /// `surfaces` maps token id → decoded surface text and must cover every
    /// masked position.
    pub fn mask_for(&self, machine: &JsonGrammarMachine, surfaces: &[String]) -> Arc<Vec<bool>> {
        let key = machine.state_key();
        let mut cache = self.cache.lock().expect("grammar mask cache");
        cache
            .entry(key)
            .or_insert_with(|| {
                let mut mask = Vec::with_capacity(surfaces.len());
                for surface in surfaces {
                    let mut probe = machine.clone();
                    mask.push(probe.feed(surface).is_ok());
                }
                Arc::new(mask)
            })
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine_over(text: &str) -> Result<JsonGrammarMachine, GrammarError> {
        let mut machine = JsonGrammarMachine::new();
        machine.feed(text)?;
        Ok(machine)
    }

    #[test]
    fn accepts_json_values_and_completes() {
        for text in [
            "42",
            "1.5e-3",
            "-0.5",
            "0",
            "\"hi\"",
            "true",
            "false",
            "null",
            "{}",
            "[]",
            r#"{"a": 1, "b": [true, null, "x"], "c": {"d": []}}"#,
            "\"\\u00e9\"",
            "[1, 2.5, -3e2]",
            r#"{"a":[1,{"b":2}]}"#,
        ] {
            let machine = machine_over(text).unwrap_or_else(|_| panic!("rejected {text}"));
            assert!(
                machine.is_document_terminal(),
                "{text} must be document-terminal"
            );
        }
    }

    #[test]
    fn rejects_malformed_documents() {
        for text in [
            "01",
            ".5",
            "-.5",
            "{\"a\" 1}",
            "{a:1}",
            "[1,]",
            "\"\\q\"",
            "\"\\u00\"",
            "[1}]",
            "{\"a\":1,}",
            "{\"a\":}",
            "1,2",
            "{\"a\":1}}",
            "[}",
            "[1]]",
            "{}}",
        ] {
            assert!(machine_over(text).is_err(), "accepted {text}");
        }
    }

    #[test]
    fn incomplete_prefixes_are_not_document_terminal() {
        for text in [
            "1.",
            "-",
            "tru",
            "fals",
            "\"unterminated",
            "{",
            "[",
            "1e",
            "0.",
        ] {
            let machine = machine_over(text).unwrap_or_else(|_| panic!("rejected {text}"));
            assert!(
                !machine.is_document_terminal(),
                "{text} must not end the document"
            );
        }
    }

    #[test]
    fn sampled_generation_under_the_mask_parses_as_json() {
        use crate::models::shared::chat::ChatGenerationConfig;
        use crate::models::shared::sampling::ChatSampler;
        use candle_core::{Device, Tensor};

        let vocab_json = r#"{
            "version":"1.0","truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,
            "model":{"type":"WordLevel","vocab":{
                "{":0,"\"":1,"a":2,":":3,"1":4,"}":5,",":6,"[":7,"]":8," ":9,"b":10,
                "<eos>":11,"<unk>":12},
                "unk_token":"<unk>"}
        }"#;
        let tokenizer =
            crate::tokenizer::Tokenizer::from_hf_json_bytes(vocab_json.as_bytes()).unwrap();
        let eos: u32 = 11;
        let surfaces = (0..13usize)
            .map(|id| tokenizer.decode(&[id as u32]).unwrap())
            .collect::<Vec<_>>();
        let masks = JsonGrammarMasks::new();
        // Closing characters are slightly preferred so greedy-style seeds
        // terminate quickly; openers stay available for sampled seeds.
        let weight = |surface: &str| -> f32 {
            match surface {
                "}" | "]" | "\"" | ":" | "," => 3.0,
                "<eos>" => 2.0,
                _ => 2.0,
            }
        };

        for seed in 0..24u64 {
            let config = ChatGenerationConfig {
                temperature: 1.0,
                top_p: 1.0,
                seed: seed + 1,
                constrain_json_object: true,
                ..ChatGenerationConfig::default()
            };
            let mut sampler = ChatSampler::new(config, &[])
                .with_json_object_constraint(std::sync::Arc::new(tokenizer.clone()), vec![eos]);
            let mut machine = JsonGrammarMachine::new();
            let mut assembled = String::new();
            for _ in 0..200 {
                if machine.is_document_terminal() {
                    break;
                }
                let mask = masks.mask_for(&machine, &surfaces);
                let values = surfaces
                    .iter()
                    .enumerate()
                    .map(|(index, surface)| {
                        // EOS is only proposed where the grammar can end.
                        if index == eos as usize {
                            if machine.is_document_terminal() {
                                2.0
                            } else {
                                f32::NEG_INFINITY
                            }
                        } else if mask.get(index).copied().unwrap_or(false) {
                            weight(surface)
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                    .collect::<Vec<_>>();
                let logits = Tensor::from_vec(values, surfaces.len(), &Device::Cpu).unwrap();
                let token = sampler.sample(&logits, surfaces.len()).unwrap();
                if token == eos {
                    break;
                }
                let surface = &surfaces[token as usize];
                machine.feed(surface).unwrap();
                assembled.push_str(surface);
            }
            assert!(
                machine.is_document_terminal(),
                "seed {seed} never completed: {assembled:?}"
            );
            let parsed: Result<serde_json::Value, _> = assembled.parse();
            assert!(
                parsed.is_ok(),
                "seed {seed} produced non-JSON output {assembled:?}"
            );
        }
    }

    #[test]
    fn masks_only_legal_continuations() {
        let surfaces = vec![
            "{\"".to_string(),
            "\"a\"".to_string(),
            "\":".to_string(),
            "1".to_string(),
            "}".to_string(),
            "x".to_string(),
        ];
        let masks = JsonGrammarMasks::new();

        let start = JsonGrammarMachine::new();
        let mask = masks.mask_for(&start, &surfaces);
        assert_eq!(*mask, vec![true, true, true, true, false, false]);

        let mut machine = JsonGrammarMachine::new();
        machine.feed("{").unwrap();
        let mask = masks.mask_for(&machine, &surfaces);
        assert_eq!(*mask, vec![false, true, true, false, true, false]);

        let mut machine = JsonGrammarMachine::new();
        machine.feed("{\"a\"").unwrap();
        let colon_surfaces = vec![":".to_string(), "x".to_string()];
        let mask = masks.mask_for(&machine, &colon_surfaces);
        assert_eq!(*mask, vec![true, false], "only the colon may follow a key");
    }
}
