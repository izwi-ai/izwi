//! Chat request resolution shared by every entry path (the local server, the
//! gateway and the worker), so the same request samples the same way on
//! each of them.

use crate::catalog::{ModelFamily, ModelVariant};

/// The output budget (izwi-core's `GenerationParams::max_tokens`) of a chat
/// request that did not set one: it runs with the rest of the model's
/// context.
pub const CHAT_MAX_TOKENS_UNSET: usize = usize::MAX;

/// A chat request's output budget from OpenAI's `max_completion_tokens`
/// (preferred) and legacy `max_tokens`, or [`CHAT_MAX_TOKENS_UNSET`].
pub fn resolve_chat_max_tokens(
    max_completion_tokens: Option<usize>,
    max_tokens: Option<usize>,
) -> usize {
    max_completion_tokens
        .or(max_tokens)
        .unwrap_or(CHAT_MAX_TOKENS_UNSET)
        .max(1)
}

/// Sampling controls a chat request set explicitly. `None` keeps the model
/// family's published profile, or the engine default for models without one.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ChatSamplingRequest {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}

/// The sampling a chat request runs with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedChatSampling {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub presence_penalty: f32,
}

/// A chat request's controls besides sampling and its output budget.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChatGenerationControls {
    /// OpenAI `seed`.
    pub seed: Option<u64>,
    /// OpenAI `stop` strings.
    pub stop: Vec<String>,
    /// OpenAI `logprobs: true` with `top_logprobs` alternatives (0 when the
    /// request set none); `None` collects no logprobs.
    pub logprobs: Option<usize>,
    /// OpenAI `response_format: json_object`.
    pub constrain_json_object: bool,
    /// vLLM's `skip_special_tokens: false`: the reply text keeps the
    /// tokenizer's special tokens. They are skipped by default.
    pub keep_special_tokens: bool,
}

impl ResolvedChatSampling {
    /// The engine's default sampling (izwi-core's
    /// `GenerationParams::default()`), for models without a published
    /// profile.
    pub const ENGINE_DEFAULT: Self = Self {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 0,
        repetition_penalty: 1.1,
        presence_penalty: 0.0,
    };
}

/// Whether a chat model's sampler can constrain output to one JSON value
/// (`response_format: json_object`, DS9.2). Other models reject the
/// request rather than ignore it.
pub fn chat_supports_json_object(variant: ModelVariant) -> bool {
    matches!(
        variant.family(),
        ModelFamily::Qwen3Chat
            | ModelFamily::Gemma3Chat
            | ModelFamily::Lfm2Chat
            | ModelFamily::Lfm3Chat
            | ModelFamily::Qwen35MoeChat
    )
}

/// Whether a chat model's prompt renders `ChatRequestConfig::tools` with a
/// tool-calling template (the Qwen3.5 family's and LFM3's XML calls,
/// LFM2.5's marker list). Other chat models have no tool template.
pub fn chat_supports_tools(variant: ModelVariant) -> bool {
    matches!(
        variant.family(),
        ModelFamily::Qwen35Chat
            | ModelFamily::Qwen35MoeChat
            | ModelFamily::Qwen38Chat
            | ModelFamily::Lfm2Chat
            | ModelFamily::Lfm3Chat
    )
}

/// Resolve a chat request's sampling. Each value comes from the request when
/// it set one, else from the model family's published profile, else from the
/// engine default. `enable_thinking` selects the profile's mode and falls back
/// to the model's own default mode exactly as its prompt renderer does.
pub fn resolve_chat_sampling(
    variant: ModelVariant,
    enable_thinking: Option<bool>,
    request: ChatSamplingRequest,
) -> ResolvedChatSampling {
    // Qwen chat models get Qwen's published sampling profile. The engine-wide
    // default (T 0.7, top_p 0.9, no top-k, repetition penalty 1.1 over prompt
    // + output) penalizes `<|im_end|>`, newlines, and `</think>` on these
    // models.
    let profile = if let Some(profile) = qwen_sampling_profile(variant, enable_thinking) {
        ResolvedChatSampling {
            temperature: profile.temperature,
            top_p: profile.top_p,
            top_k: 20,
            repetition_penalty: 1.0,
            presence_penalty: profile.presence_penalty,
        }
    } else if let Some(temperature) = lfm25_sampling_temperature(variant) {
        // Liquid's model cards: top-k 50 and repetition penalty 1.05, with
        // no top-p or presence penalty.
        ResolvedChatSampling {
            temperature,
            top_p: 1.0,
            top_k: 50,
            repetition_penalty: 1.05,
            presence_penalty: 0.0,
        }
    } else if variant.family() == ModelFamily::Lfm3Chat {
        // LFM3-1.7B's card and `generation_config.json` (general use: chat,
        // tools, agents): temperature 0.1, top-k 50, repetition penalty 1.1.
        ResolvedChatSampling {
            temperature: 0.1,
            top_p: 1.0,
            top_k: 50,
            repetition_penalty: 1.1,
            presence_penalty: 0.0,
        }
    } else if variant.family() == ModelFamily::Lfm3Audio {
        // LFM3-Audio-2B's `generation_config.json` decodes text greedily.
        ResolvedChatSampling {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            presence_penalty: 0.0,
        }
    } else {
        ResolvedChatSampling::ENGINE_DEFAULT
    };
    ResolvedChatSampling {
        temperature: request.temperature.unwrap_or(profile.temperature),
        top_p: request.top_p.unwrap_or(profile.top_p),
        top_k: request.top_k.unwrap_or(profile.top_k),
        repetition_penalty: request
            .repetition_penalty
            .unwrap_or(profile.repetition_penalty),
        presence_penalty: request.presence_penalty.unwrap_or(profile.presence_penalty),
    }
}

/// Sampling values that differ between Qwen's published per-mode profiles.
/// Every profile also uses top-k 20 and repetition penalty 1.0.
#[derive(Debug, Clone, Copy, PartialEq)]
struct QwenSamplingProfile {
    temperature: f32,
    top_p: f32,
    presence_penalty: f32,
}

/// Qwen's published sampling profile for a chat model, or `None` for models
/// that use the engine default. The values come from each model's card or
/// `generation_config.json`.
fn qwen_sampling_profile(
    variant: ModelVariant,
    enable_thinking: Option<bool>,
) -> Option<QwenSamplingProfile> {
    let profile = |temperature, top_p, presence_penalty| QwenSamplingProfile {
        temperature,
        top_p,
        presence_penalty,
    };
    match variant {
        // The Qwen3.6-35B-A3B-FP8 `generation_config.json` is exactly the
        // thinking branch.
        ModelVariant::Qwen3827BFp8 | ModelVariant::Qwen36Moe35BA3BFp8 => {
            Some(if enable_thinking.unwrap_or(true) {
                profile(1.0, 0.95, 0.0)
            } else {
                profile(0.7, 0.80, 1.5)
            })
        }
        // Qwen3.5-4B and -9B think by default. Their cards give the general
        // thinking and instruct profiles below.
        ModelVariant::Qwen354BGguf | ModelVariant::Qwen359BGguf => {
            Some(if enable_thinking.unwrap_or(true) {
                profile(1.0, 0.95, 1.5)
            } else {
                profile(0.7, 0.80, 1.5)
            })
        }
        // Qwen3.5-0.8B and -2B answer without thinking by default. Their
        // cards give these text profiles for each mode.
        ModelVariant::Qwen3508BGguf | ModelVariant::Qwen352BGguf => {
            Some(if enable_thinking.unwrap_or(false) {
                profile(1.0, 0.95, 1.5)
            } else {
                profile(1.0, 1.0, 2.0)
            })
        }
        _ => None,
    }
}

/// Temperature from Liquid's LFM2.5-1.2B model cards (Instruct 0.1, Thinking
/// 0.05), or `None` for other models. Neither the GGUF nor
/// `generation_config.json` carries sampling values.
fn lfm25_sampling_temperature(variant: ModelVariant) -> Option<f32> {
    match variant {
        ModelVariant::Lfm2512BInstructGguf => Some(0.1),
        ModelVariant::Lfm2512BThinkingGguf => Some(0.05),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN_THINKING_VARIANTS: [ModelVariant; 2] =
        [ModelVariant::Qwen3827BFp8, ModelVariant::Qwen36Moe35BA3BFp8];

    const QWEN35_VARIANTS: [ModelVariant; 4] = [
        ModelVariant::Qwen3508BGguf,
        ModelVariant::Qwen352BGguf,
        ModelVariant::Qwen354BGguf,
        ModelVariant::Qwen359BGguf,
    ];

    fn sampling(
        temperature: f32,
        top_p: f32,
        top_k: usize,
        repetition_penalty: f32,
        presence_penalty: f32,
    ) -> ResolvedChatSampling {
        ResolvedChatSampling {
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            presence_penalty,
        }
    }

    /// An omitted budget stays unset until the exact prompt is known, and an
    /// explicit one is not capped at an API-layer ceiling.
    #[test]
    fn chat_output_budgets_prefer_max_completion_tokens() {
        assert_eq!(resolve_chat_max_tokens(None, None), CHAT_MAX_TOKENS_UNSET);
        assert_eq!(resolve_chat_max_tokens(None, Some(32_768)), 32_768);
        assert_eq!(resolve_chat_max_tokens(Some(65_536), Some(32_768)), 65_536);
    }

    #[test]
    fn lfm25_models_use_their_model_card_sampling_profile() {
        for (variant, temperature) in [
            (ModelVariant::Lfm2512BInstructGguf, 0.1),
            (ModelVariant::Lfm2512BThinkingGguf, 0.05),
        ] {
            assert_eq!(
                resolve_chat_sampling(variant, None, ChatSamplingRequest::default()),
                sampling(temperature, 1.0, 50, 1.05, 0.0),
                "{variant:?}"
            );
        }
    }

    #[test]
    fn lfm3_uses_its_model_card_sampling_and_supports_tools_and_json() {
        for variant in [ModelVariant::Lfm317BGguf, ModelVariant::Lfm317BQ8Gguf] {
            for enable_thinking in [None, Some(true), Some(false)] {
                assert_eq!(
                    resolve_chat_sampling(variant, enable_thinking, ChatSamplingRequest::default()),
                    sampling(0.1, 1.0, 50, 1.1, 0.0),
                    "{variant:?} enable_thinking={enable_thinking:?}"
                );
            }
            assert!(chat_supports_tools(variant) && chat_supports_json_object(variant));
            assert!(crate::chat::reasoning::reasoning_opened_in_prompt(
                variant, None
            ));
            assert!(!crate::chat::reasoning::reasoning_opened_in_prompt(
                variant,
                Some(false)
            ));
        }
    }

    #[test]
    fn lfm3_audio_chat_decodes_text_greedily_unless_asked_otherwise() {
        for variant in [ModelVariant::Lfm3Audio2BGguf, ModelVariant::Lfm3Audio2BQ8Gguf] {
            assert_eq!(
                resolve_chat_sampling(variant, None, ChatSamplingRequest::default()),
                sampling(0.0, 1.0, 0, 1.0, 0.0)
            );
            assert!(variant.serves_chat_requests() && !variant.is_chat());
            assert!(!chat_supports_json_object(variant));
        }
    }

    #[test]
    fn qwen_thinking_models_select_the_profile_for_the_requested_mode() {
        for variant in QWEN_THINKING_VARIANTS {
            for (enable_thinking, expected) in [
                (None, sampling(1.0, 0.95, 20, 1.0, 0.0)),
                (Some(true), sampling(1.0, 0.95, 20, 1.0, 0.0)),
                (Some(false), sampling(0.7, 0.80, 20, 1.0, 1.5)),
            ] {
                assert_eq!(
                    resolve_chat_sampling(variant, enable_thinking, ChatSamplingRequest::default()),
                    expected,
                    "{variant:?} enable_thinking={enable_thinking:?}"
                );
            }
        }
    }

    /// Expected `(temperature, top_p, presence_penalty)` for each Qwen3.5
    /// variant and thinking request, from the Qwen3.5 model cards.
    const QWEN35_PROFILES: [(ModelVariant, Option<bool>, (f32, f32, f32)); 12] = [
        (ModelVariant::Qwen354BGguf, None, (1.0, 0.95, 1.5)),
        (ModelVariant::Qwen354BGguf, Some(true), (1.0, 0.95, 1.5)),
        (ModelVariant::Qwen354BGguf, Some(false), (0.7, 0.80, 1.5)),
        (ModelVariant::Qwen359BGguf, None, (1.0, 0.95, 1.5)),
        (ModelVariant::Qwen359BGguf, Some(true), (1.0, 0.95, 1.5)),
        (ModelVariant::Qwen359BGguf, Some(false), (0.7, 0.80, 1.5)),
        (ModelVariant::Qwen3508BGguf, None, (1.0, 1.0, 2.0)),
        (ModelVariant::Qwen3508BGguf, Some(false), (1.0, 1.0, 2.0)),
        (ModelVariant::Qwen3508BGguf, Some(true), (1.0, 0.95, 1.5)),
        (ModelVariant::Qwen352BGguf, None, (1.0, 1.0, 2.0)),
        (ModelVariant::Qwen352BGguf, Some(false), (1.0, 1.0, 2.0)),
        (ModelVariant::Qwen352BGguf, Some(true), (1.0, 0.95, 1.5)),
    ];

    #[test]
    fn qwen35_models_use_their_published_sampling_profile_per_mode() {
        for (variant, enable_thinking, (temperature, top_p, presence)) in QWEN35_PROFILES {
            assert_eq!(
                resolve_chat_sampling(variant, enable_thinking, ChatSamplingRequest::default()),
                sampling(temperature, top_p, 20, 1.0, presence),
                "{variant:?} enable_thinking={enable_thinking:?}"
            );
        }
    }

    #[test]
    fn explicit_values_win_over_every_profile() {
        let request = ChatSamplingRequest {
            temperature: Some(0.2),
            top_p: Some(0.3),
            top_k: Some(7),
            repetition_penalty: Some(1.2),
            presence_penalty: Some(-0.4),
        };
        for variant in QWEN_THINKING_VARIANTS
            .into_iter()
            .chain(QWEN35_VARIANTS)
            .chain([ModelVariant::Lfm2512BInstructGguf, ModelVariant::Gemma34BIt])
        {
            assert_eq!(
                resolve_chat_sampling(variant, Some(true), request),
                sampling(0.2, 0.3, 7, 1.2, -0.4),
                "{variant:?}"
            );
        }
    }

    #[test]
    fn json_object_needs_a_grammar_aware_sampler() {
        assert!(chat_supports_json_object(
            ModelVariant::Lfm2512BInstructGguf
        ));
        assert!(chat_supports_json_object(ModelVariant::Qwen36Moe35BA3BFp8));
        assert!(chat_supports_json_object(ModelVariant::Gemma34BIt));
        assert!(!chat_supports_json_object(ModelVariant::Qwen3827BFp8));
        assert!(chat_supports_tools(ModelVariant::Qwen3827BFp8));
        assert!(chat_supports_tools(ModelVariant::Lfm2512BInstructGguf));
        assert!(!chat_supports_tools(ModelVariant::Qwen314BGguf));
        assert!(!chat_supports_tools(ModelVariant::Gemma34BIt));
    }

    #[test]
    fn models_without_a_profile_keep_the_engine_default() {
        assert_eq!(
            resolve_chat_sampling(
                ModelVariant::Gemma34BIt,
                None,
                ChatSamplingRequest::default()
            ),
            ResolvedChatSampling::ENGINE_DEFAULT
        );
        // An unset value falls through to the profile while a set one wins.
        let resolved = resolve_chat_sampling(
            ModelVariant::Qwen354BGguf,
            None,
            ChatSamplingRequest {
                temperature: Some(0.0),
                ..Default::default()
            },
        );
        assert_eq!(resolved, sampling(0.0, 0.95, 20, 1.0, 1.5));
    }
}
