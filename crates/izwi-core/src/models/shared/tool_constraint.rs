//! Sampler-side constraint for tool calls: only declared tool names, and
//! arguments that satisfy each tool's parameter schema.
//!
//! The request carries the grammars
//! ([`izwi_model_catalog::chat::tools::ToolConstraint`], llguidance's Lark
//! dialect). The model samples freely until it writes the call's opening
//! marker, one token in every tool-calling vocabulary; from there the grammar
//! masks the sampler until the call closes, and the model is free again. A
//! request that must make a call starts inside one, after the prefix the
//! prompt wrote.
//!
//! A grammar that does not compile, or a tokenizer without the marker, makes
//! no constraint: the request then runs as it would without one, and its
//! output is parsed leniently. A finished generation is never failed here.

use std::sync::{Arc, OnceLock};

use izwi_model_catalog::chat::tools::ToolConstraint;
use llguidance::api::TopLevelGrammar;
use llguidance::toktrie::{ApproximateTokEnv, SimpleVob, TokEnv, TokRxInfo, TokTrie};
use llguidance::{Matcher, ParserFactory};
use tracing::warn;

use crate::tokenizer::Tokenizer;

/// What a tokenizer contributes to every constraint over it: the token trie
/// and the grammar compiler. Built once per tokenizer, on first use.
pub(crate) struct ConstraintVocabulary {
    factory: ParserFactory,
    tok_env: TokEnv,
}

impl ConstraintVocabulary {
    fn new(tokenizer: &Tokenizer, end_of_turn: u32) -> Result<Self, String> {
        let words: Vec<Vec<u8>> = tokenizer
            .token_bytes()
            .into_iter()
            .enumerate()
            .map(|(id, bytes)| match bytes {
                Some(bytes) if !bytes.is_empty() => bytes,
                // A control token: named, and never part of grammar text.
                _ => {
                    let mut name = format!(".<[{id}]>").into_bytes();
                    name[0] = TokTrie::SPECIAL_TOKEN_MARKER;
                    name
                }
            })
            .collect();
        if end_of_turn as usize >= words.len() {
            return Err("the end-of-turn token is outside the vocabulary".to_string());
        }
        let info = TokRxInfo::new(words.len() as u32, end_of_turn);
        let tok_env: TokEnv = Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &words)));
        let mut factory = ParserFactory::new_simple(&tok_env).map_err(|err| err.to_string())?;
        factory.quiet();
        Ok(Self { factory, tok_env })
    }

    fn matcher(&self, grammar: &str) -> Result<Matcher, String> {
        let parser = self
            .factory
            .create_parser(TopLevelGrammar::from_lark(grammar.to_string()))
            .map_err(|err| err.to_string())?;
        Ok(Matcher::new(Ok(parser)))
    }

    fn vocab_size(&self) -> usize {
        self.tok_env.tok_trie().vocab_size()
    }
}

/// A tokenizer's [`ConstraintVocabulary`], shared by its clones.
#[derive(Clone, Default)]
pub(crate) struct ConstraintVocabularyCell(
    Arc<OnceLock<Result<Arc<ConstraintVocabulary>, String>>>,
);

impl ConstraintVocabularyCell {
    fn get(
        &self,
        tokenizer: &Tokenizer,
        end_of_turn: u32,
    ) -> Result<Arc<ConstraintVocabulary>, String> {
        self.0
            .get_or_init(|| ConstraintVocabulary::new(tokenizer, end_of_turn).map(Arc::new))
            .clone()
    }
}

impl std::fmt::Debug for ConstraintVocabularyCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ConstraintVocabularyCell")
    }
}

#[derive(Clone)]
enum Phase {
    /// The model writes what it likes; the opening marker starts a call.
    Free,
    /// Inside a call, under its grammar.
    InCall(Matcher),
    /// The answer made its last allowed call: only the end of the turn.
    Done,
}

/// What the sampler may pick next.
pub(crate) enum Allowed {
    /// Any token.
    Any,
    /// The tokens of this mask.
    Mask(SimpleVob),
    /// Only the tokens that end the turn.
    EndOfTurn,
}

/// One request's tool-call constraint, advanced one sampled token at a time.
#[derive(Clone)]
pub(crate) struct ToolConstraintRuntime {
    vocabulary: Arc<ConstraintVocabulary>,
    constraint: Arc<ToolConstraint>,
    opener: u32,
    phase: Phase,
    calls: u32,
}

impl ToolConstraintRuntime {
    /// The runtime for `constraint` over `tokenizer`, or `None` (with a
    /// warning) when it cannot be enforced: the request then runs
    /// unconstrained.
    pub(crate) fn new(
        tokenizer: &Tokenizer,
        end_of_turn: u32,
        constraint: &ToolConstraint,
    ) -> Option<Self> {
        match Self::try_new(tokenizer, end_of_turn, constraint) {
            Ok(runtime) => Some(runtime),
            Err(reason) => {
                warn!(%reason, "tool calls are not constrained for this request");
                None
            }
        }
    }

    fn try_new(
        tokenizer: &Tokenizer,
        end_of_turn: u32,
        constraint: &ToolConstraint,
    ) -> Result<Self, String> {
        let opener = tokenizer
            .token_to_id(&constraint.opener)
            .ok_or_else(|| format!("{:?} is not one token of this model", constraint.opener))?;
        let vocabulary = tokenizer
            .constraint_vocabulary()
            .get(tokenizer, end_of_turn)?;
        // Compile both grammars now, so a bad one is found before sampling.
        vocabulary.matcher(&constraint.call)?;
        let phase = match &constraint.forced {
            Some(forced) => Phase::InCall(vocabulary.matcher(forced)?),
            None => Phase::Free,
        };
        Ok(Self {
            vocabulary,
            constraint: Arc::new(constraint.clone()),
            opener,
            phase,
            calls: 0,
        })
    }

    /// What may be sampled next. A matcher that fails releases the
    /// constraint rather than the generation.
    pub(crate) fn allowed(&mut self) -> Allowed {
        match &mut self.phase {
            Phase::Free => Allowed::Any,
            Phase::Done => Allowed::EndOfTurn,
            Phase::InCall(matcher) => match matcher.compute_mask() {
                Ok(mask) if !mask.is_zero() => Allowed::Mask(mask),
                Ok(_) => {
                    self.finish_call();
                    self.allowed()
                }
                Err(err) => {
                    self.release(&err.to_string());
                    Allowed::Any
                }
            },
        }
    }

    /// Record the sampled token.
    pub(crate) fn commit(&mut self, token: u32) {
        match &mut self.phase {
            Phase::Free => {
                if token == self.opener {
                    match self.vocabulary.matcher(&self.constraint.call) {
                        Ok(matcher) => self.phase = Phase::InCall(matcher),
                        Err(reason) => self.release(&reason),
                    }
                }
            }
            Phase::InCall(matcher) => {
                if token as usize >= self.vocabulary.vocab_size() {
                    return;
                }
                match matcher.consume_token(token) {
                    Ok(()) => {
                        if matcher.is_stopped() {
                            self.finish_call();
                        }
                    }
                    Err(err) => self.release(&err.to_string()),
                }
            }
            Phase::Done => {}
        }
    }

    /// Whether the sampler is inside a call, where the turn cannot end.
    pub(crate) fn in_call(&self) -> bool {
        matches!(self.phase, Phase::InCall(_))
    }

    fn finish_call(&mut self) {
        self.calls += 1;
        self.phase = match self.constraint.max_calls {
            Some(max) if self.calls >= max => Phase::Done,
            _ => Phase::Free,
        };
    }

    fn release(&mut self, reason: &str) {
        warn!(%reason, "tool-call constraint released mid-answer");
        self.phase = Phase::Free;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use izwi_model_catalog::chat::tools::{ToolChoice, ToolDialect, ToolPlan};
    use serde_json::{json, Value};

    const END_OF_TURN: u32 = 256;

    /// A byte-level tokenizer: one token per byte, the end-of-turn control
    /// token, and the tool-call markers as single tokens, as in the real
    /// vocabularies.
    fn tokenizer() -> Tokenizer {
        let mut next = 0x100u32;
        let mut tokens: Vec<String> = (0..=255u8)
            .map(|byte| {
                let own = byte as char;
                if matches!(own, '!'..='~' | '\u{00A1}'..='\u{00AC}' | '\u{00AE}'..='\u{00FF}') {
                    own.to_string()
                } else {
                    let mapped = char::from_u32(next).unwrap();
                    next += 1;
                    mapped.to_string()
                }
            })
            .collect();
        let mut types = vec![1u32; tokens.len()];
        for (marker, kind) in [
            ("<|im_end|>", 3),
            ("<tool_call>", 4),
            ("</tool_call>", 4),
            ("<|tool_call_start|>", 4),
            ("<|tool_call_end|>", 4),
        ] {
            tokens.push(marker.to_string());
            types.push(kind);
        }
        let mut tokenizer = Tokenizer::from_gguf_bpe(&tokens, &[], Some("qwen35"), false).unwrap();
        tokenizer
            .register_gguf_token_types(&tokens, &types)
            .unwrap();
        assert_eq!(tokenizer.token_to_id("<|im_end|>"), Some(END_OF_TURN));
        tokenizer
    }

    fn tools() -> Vec<Value> {
        vec![
            json!({"type": "function", "function": {"name": "get_weather", "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    "days": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["city"]
            }}}),
            json!({"type": "function", "function": {"name": "now"}}),
        ]
    }

    fn runtime(
        tokenizer: &Tokenizer,
        dialect: ToolDialect,
        choice: ToolChoice,
        parallel: bool,
    ) -> ToolConstraintRuntime {
        let constraint = ToolPlan::new(dialect, tools(), choice, parallel)
            .unwrap()
            .constraint(true)
            .unwrap();
        ToolConstraintRuntime::try_new(tokenizer, END_OF_TURN, &constraint)
            .unwrap_or_else(|reason| panic!("{dialect:?}: {reason}\n{constraint:#?}"))
    }

    /// Feed `text` token by token; `Err` holds the text accepted before the
    /// first token the constraint forbids.
    fn feed(
        tokenizer: &Tokenizer,
        runtime: &mut ToolConstraintRuntime,
        text: &str,
    ) -> Result<(), String> {
        let ids = tokenizer.encode(text).unwrap();
        for (index, &id) in ids.iter().enumerate() {
            let allowed = match runtime.allowed() {
                Allowed::Any => true,
                Allowed::Mask(mask) => mask.is_allowed(id),
                Allowed::EndOfTurn => id == END_OF_TURN,
            };
            if !allowed {
                return Err(tokenizer.decode_with_special_tokens(&ids[..index]).unwrap());
            }
            runtime.commit(id);
        }
        Ok(())
    }

    #[test]
    fn token_bytes_spell_bytes_and_markers_and_hide_control_tokens() {
        let tokenizer = tokenizer();
        let bytes = tokenizer.token_bytes();
        assert_eq!(bytes[b'a' as usize].as_deref(), Some(&b"a"[..]));
        assert_eq!(bytes[b'\n' as usize].as_deref(), Some(&b"\n"[..]));
        assert_eq!(bytes[0xC3].as_deref(), Some(&[0xC3u8][..]));
        assert_eq!(bytes[END_OF_TURN as usize], None);
        let call = tokenizer.token_to_id("<tool_call>").unwrap() as usize;
        assert_eq!(bytes[call].as_deref(), Some(&b"<tool_call>"[..]));
    }

    #[test]
    fn a_model_opened_xml_call_must_name_a_tool_and_fit_its_schema() {
        let tokenizer = tokenizer();
        let free = || runtime(&tokenizer, ToolDialect::QwenXml, ToolChoice::Auto, true);
        let valid = "Let me check, a < b.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nSão Paulo\nline two\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n<parameter=days>\n2\n</parameter>\n<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=now>\n</function>\n</tool_call> done<|im_end|>";
        let mut runtime = free();
        assert_eq!(feed(&tokenizer, &mut runtime, valid), Ok(()));
        assert!(!runtime.in_call());

        // An undeclared name, a missing required parameter, a wrong enum
        // value, a wrong type and an ended turn are all stopped where they
        // start.
        for (text, accepted) in [
            ("<tool_call>\n<function=delete>", "<tool_call>\n<function="),
            (
                "<tool_call>\n<function=get_weather>\n<parameter=unit>",
                "<tool_call>\n<function=get_weather>\n<parameter=",
            ),
            (
                "<tool_call>\n<function=get_weather>\n<parameter=city>\nx\n</parameter>\n<parameter=unit>\nkelvin",
                "<tool_call>\n<function=get_weather>\n<parameter=city>\nx\n</parameter>\n<parameter=unit>\n",
            ),
            (
                "<tool_call>\n<function=get_weather>\n<parameter=city>\nx\n</parameter>\n<parameter=days>\ntwo",
                "<tool_call>\n<function=get_weather>\n<parameter=city>\nx\n</parameter>\n<parameter=days>\n",
            ),
            (
                "<tool_call>\n<function=now>\n<|im_end|>",
                "<tool_call>\n<function=now>\n",
            ),
        ] {
            assert_eq!(feed(&tokenizer, &mut free(), text), Err(accepted.to_string()), "{text}");
        }
    }

    #[test]
    fn a_forced_call_starts_inside_the_grammar_and_one_call_ends_the_turn() {
        let tokenizer = tokenizer();
        // The prompt wrote `<tool_call>\n<function=get_weather>\n`.
        let mut named = runtime(
            &tokenizer,
            ToolDialect::QwenXml,
            ToolChoice::Named("get_weather".into()),
            false,
        );
        assert!(named.in_call());
        assert_eq!(
            feed(&tokenizer, &mut named, "plain text"),
            Err(String::new())
        );
        assert_eq!(
            feed(
                &tokenizer,
                &mut named,
                "<parameter=city>\nHarare\n</parameter>\n</function>\n</tool_call>"
            ),
            Ok(())
        );
        // One call at most: nothing but the end of the turn may follow.
        assert_eq!(
            feed(&tokenizer, &mut named.clone(), "\n"),
            Err(String::new())
        );
        assert_eq!(feed(&tokenizer, &mut named, "<|im_end|>"), Ok(()));

        // "required" leaves the name to the model, among the declared ones.
        let mut required = runtime(&tokenizer, ToolDialect::QwenXml, ToolChoice::Required, true);
        assert_eq!(
            feed(&tokenizer, &mut required.clone(), "delete>"),
            Err(String::new())
        );
        assert_eq!(
            feed(
                &tokenizer,
                &mut required,
                "now>\n</function>\n</tool_call> and text"
            ),
            Ok(())
        );
    }

    #[test]
    fn hermes_calls_are_json_objects_of_a_declared_tool() {
        let tokenizer = tokenizer();
        let free = || runtime(&tokenizer, ToolDialect::HermesJson, ToolChoice::Auto, true);
        assert_eq!(
            feed(
                &tokenizer,
                &mut free(),
                "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Harare\", \"days\": 2}}\n</tool_call>"
            ),
            Ok(())
        );
        assert_eq!(
            feed(
                &tokenizer,
                &mut free(),
                "<tool_call>\n{\"name\": \"delete\""
            ),
            Err("<tool_call>\n{\"name\": \"".to_string())
        );
        // The required parameter must be there before the object closes.
        assert!(feed(
            &tokenizer,
            &mut free(),
            "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {}}"
        )
        .is_err());
        // Forced by name: the prompt wrote up to `"arguments": `.
        let mut named = runtime(
            &tokenizer,
            ToolDialect::HermesJson,
            ToolChoice::Named("get_weather".into()),
            true,
        );
        assert_eq!(
            feed(
                &tokenizer,
                &mut named,
                "{\"city\": \"Harare\"}}\n</tool_call>"
            ),
            Ok(())
        );
    }

    #[test]
    fn pythonic_call_lists_follow_the_declared_arguments() {
        let tokenizer = tokenizer();
        let free = |parallel| {
            runtime(
                &tokenizer,
                ToolDialect::LfmPythonic,
                ToolChoice::Auto,
                parallel,
            )
        };
        assert_eq!(
            feed(
                &tokenizer,
                &mut free(true),
                "Checking.<|tool_call_start|>[get_weather(city='Harare', unit='celsius', days=2), now()]<|tool_call_end|> Done."
            ),
            Ok(())
        );
        for text in [
            "<|tool_call_start|>[delete(",
            "<|tool_call_start|>[get_weather(unit=",
            "<|tool_call_start|>[get_weather(city=5",
            "<|tool_call_start|>[get_weather(city='x', unit='kelvin'",
        ] {
            assert!(feed(&tokenizer, &mut free(true), text).is_err(), "{text}");
        }
        // One call at most: no second call in the list.
        assert!(feed(
            &tokenizer,
            &mut free(false),
            "<|tool_call_start|>[now(), now()]<|tool_call_end|>"
        )
        .is_err());
        // Forced by name: the prompt wrote `<|tool_call_start|>[get_weather(`.
        let mut named = runtime(
            &tokenizer,
            ToolDialect::LfmPythonic,
            ToolChoice::Named("get_weather".into()),
            true,
        );
        assert_eq!(
            feed(&tokenizer, &mut named, "city=\"Harare\")]<|tool_call_end|>"),
            Ok(())
        );
    }

    /// The chat sampler under a forced call: whatever the logits prefer, the
    /// sampled text is a call to the named tool with schema-valid arguments,
    /// and then the turn ends.
    #[test]
    fn the_sampler_can_only_write_a_valid_forced_call() {
        use crate::models::shared::chat::ChatGenerationConfig;
        use crate::models::shared::sampling::ChatSampler;
        use candle_core::{Device, Tensor};
        use izwi_model_catalog::chat::tools::parse_answer;
        use izwi_model_catalog::chat::ChatRequestConfig;

        let tokenizer = std::sync::Arc::new(tokenizer());
        let vocab = tokenizer.vocab_size();
        let tools = vec![
            json!({"type": "function", "function": {"name": "set_mode", "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["eco", "boost"]},
                    "level": {"type": "integer", "minimum": 0, "maximum": 9},
                    "on": {"type": "boolean"}
                },
                "required": ["mode", "level"],
                "additionalProperties": false
            }}}),
        ];
        for dialect in [
            ToolDialect::QwenXml,
            ToolDialect::HermesJson,
            ToolDialect::LfmPythonic,
        ] {
            let plan = ToolPlan::new(
                dialect,
                tools.clone(),
                ToolChoice::Named("set_mode".into()),
                false,
            )
            .unwrap();
            for seed in 1..=12u64 {
                let config = ChatGenerationConfig {
                    temperature: 1.0,
                    top_p: 1.0,
                    seed,
                    request: ChatRequestConfig {
                        tool_constraint: plan.constraint(false).map(Box::new),
                        ..ChatRequestConfig::default()
                    },
                    ..ChatGenerationConfig::default()
                };
                let mut sampler = ChatSampler::new(config, &[])
                    .with_json_object_constraint(tokenizer.clone(), vec![END_OF_TURN]);
                // Flat logits: every token is as likely as any other.
                let logits = Tensor::new(vec![0f32; vocab].as_slice(), &Device::Cpu).unwrap();
                let mut ids = Vec::new();
                loop {
                    let token = sampler.sample(&logits, vocab).unwrap();
                    if token == END_OF_TURN {
                        break;
                    }
                    ids.push(token);
                    assert!(ids.len() < 400, "{dialect:?} seed {seed}: no end");
                }
                let text = tokenizer.decode_with_special_tokens(&ids).unwrap();
                let (content, calls) = parse_answer(plan.parser(), &text);
                assert_eq!(content, None, "{dialect:?} seed {seed}: {text:?}");
                assert_eq!(calls.len(), 1, "{dialect:?} seed {seed}: {text:?}");
                assert_eq!(calls[0].function.name, "set_mode");
                let arguments: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
                assert!(
                    matches!(arguments["mode"].as_str(), Some("eco" | "boost")),
                    "{dialect:?} seed {seed}: {text:?}"
                );
                assert!(
                    arguments["level"]
                        .as_i64()
                        .is_some_and(|level| (0..=9).contains(&level)),
                    "{dialect:?} seed {seed}: {text:?}"
                );
                assert!(arguments.get("on").is_none_or(Value::is_boolean));
            }
        }
    }

    #[test]
    fn a_grammar_that_does_not_compile_leaves_the_request_unconstrained() {
        let tokenizer = tokenizer();
        let broken = ToolConstraint {
            opener: "<tool_call>".to_string(),
            call: "start: undefined_rule".to_string(),
            forced: None,
            max_calls: None,
            strict: false,
        };
        assert!(ToolConstraintRuntime::new(&tokenizer, END_OF_TURN, &broken).is_none());
        let unknown_marker = ToolConstraint {
            opener: "<call>".to_string(),
            call: "start: \"x\"".to_string(),
            forced: None,
            max_calls: None,
            strict: false,
        };
        assert!(ToolConstraintRuntime::new(&tokenizer, END_OF_TURN, &unknown_marker).is_none());
    }
}
