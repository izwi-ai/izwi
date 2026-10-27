//! Maya1's pinned checkpoint, token layout, frame geometry and defaults.

use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::models::shared::rope::{Llama3RopeScaling, RopeScaling};

/// <https://huggingface.co/maya-research/maya1>; weights and tokenizer have
/// not changed since `9ba3f6f9` (2025-11-03), later commits touch the card.
pub const MAYA1_REPO: &str = "maya-research/maya1";
pub const MAYA1_REVISION: &str = "b1b9b38ca9decbfcfd6c98ea93dd502bfd0f9572";

pub const CONFIG_FILE: &str = "config.json";
pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const WEIGHT_SHARDS: [&str; 2] = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
];
/// `hubertsiuzdak/snac_24khz` at `d73ad176` (MIT), its `pytorch_model.bin`
/// stored under this name beside the LM so it cannot collide with Maya1's
/// own `config.json`; its configuration is [`SnacConfig::snac_24khz`].
pub const SNAC_WEIGHTS_FILE: &str = "snac_24khz.pth";

// Token layout (Orpheus's): Llama 3's 128,256 ids, then control tokens and
// seven bands of 4,096 SNAC codes, one band per slot of a frame.
pub const BOS: u32 = 128_000;
pub const TEXT_EOT: u32 = 128_009;
pub const START_OF_SPEECH: u32 = 128_257;
pub const END_OF_SPEECH: u32 = 128_258;
pub const START_OF_HUMAN: u32 = 128_259;
pub const END_OF_HUMAN: u32 = 128_260;
pub const START_OF_AI: u32 = 128_261;
pub const AUDIO_CODE_START: u32 = 128_266;
pub const CODEBOOK_SIZE: u32 = 4_096;
pub const TOKENS_PER_FRAME: usize = 7;
/// One past the last audio code (156,937).
pub const AUDIO_CODE_END: u32 = AUDIO_CODE_START + TOKENS_PER_FRAME as u32 * CODEBOOK_SIZE;

/// The rows of the (tied) output head a decode step can sample: the stop
/// token, the seven control ids after it (always masked) and every audio
/// code. Logits are computed for these rows only.
pub const HEAD_FIRST_TOKEN: u32 = END_OF_SPEECH;
pub const HEAD_ROWS: usize = (AUDIO_CODE_END - HEAD_FIRST_TOKEN) as usize;

/// 24 kHz mono; a frame is 7 tokens = 1 + 2 + 4 SNAC codes = 4 fine codes of
/// 512 samples.
pub const SAMPLE_RATE: u32 = 24_000;
pub const SAMPLES_PER_FRAME: usize = 2_048;
pub const FRAME_RATE_HZ: f32 = SAMPLE_RATE as f32 / SAMPLES_PER_FRAME as f32;
/// The reference drops the first frame's samples (its "warm-up") in both its
/// whole-utterance and its streamed decode.
pub const LEADING_TRIM_SAMPLES: usize = SAMPLES_PER_FRAME;

/// The 20 inline tags the tokenizer registers as single tokens
/// (156,940..=156,959, sorted like their ids). `emotions.txt` lists only 17
/// of them; the catalog carries the list so every surface can offer it.
pub use crate::catalog::MAYA1_INLINE_TAGS;

/// Sampling as the card's Transformers example and the official Space run
/// it: temperature 0.4, nucleus 0.9, repetition penalty 1.1, at least 28
/// tokens (4 frames) before the stop token, and Transformers' implicit
/// top-k of 50.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Maya1SamplingDefaults {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub min_new_tokens: usize,
}

pub const MAYA1_DEFAULTS: Maya1SamplingDefaults = Maya1SamplingDefaults {
    temperature: 0.4,
    top_p: 0.9,
    top_k: 50,
    repetition_penalty: 1.1,
    min_new_tokens: 28,
};

/// The most tokens one request may generate (the card's `max_new_tokens`,
/// about 25 s of audio). Requests get a smaller budget sized to their text
/// ([`maya1_auto_max_new_tokens`]).
pub const MAX_NEW_TOKENS: usize = 2_048;
/// The longest prompt one request may hold: framing, description and text.
pub const MAX_PROMPT_TOKENS: usize = 512;
/// Positions one sequence may reach.
pub const STATE_CONTEXT_TOKENS: usize = MAX_PROMPT_TOKENS + MAX_NEW_TOKENS;
/// The longest voice description accepted, in characters.
pub const MAX_DESCRIPTION_CHARS: usize = 1_000;

/// A request's generation budget when it sets none, sized from its text
/// alone (the server sets it before any tokenizer is at hand): about twice
/// the tokens unhurried speech of its characters takes, a second and a half
/// per inline tag, in whole frames within [`MAX_NEW_TOKENS`].
pub fn maya1_auto_max_new_tokens(text: &str) -> usize {
    let tags = crate::models::architectures::maya1::prompt::inline_tags(text).count();
    let tag_chars = crate::models::architectures::maya1::prompt::inline_tags(text)
        .map(str::len)
        .sum::<usize>();
    let chars = text.chars().count().saturating_sub(tag_chars);
    let budget = AUTO_BUDGET_BASE_TOKENS
        + AUTO_BUDGET_TOKENS_PER_CHAR * chars
        + AUTO_BUDGET_TOKENS_PER_TAG * tags;
    let frames = budget
        .div_ceil(TOKENS_PER_FRAME)
        .min(MAX_NEW_TOKENS / TOKENS_PER_FRAME);
    frames * TOKENS_PER_FRAME
}

/// One second of audio.
const AUTO_BUDGET_BASE_TOKENS: usize = 84;
/// Speech runs near 14 characters a second (5.9 tokens a character);
/// twice that leaves room for slow and expressive delivery.
const AUTO_BUDGET_TOKENS_PER_CHAR: usize = 12;
const AUTO_BUDGET_TOKENS_PER_TAG: usize = 126;

/// A built-in voice: a name and the description it stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Maya1VoicePreset {
    pub name: &'static str,
    pub description: &'static str,
}

/// Built-in voices, each a description in the card's verbose template
/// (`prompt.txt`), which keeps the model from speaking the description.
pub const MAYA1_VOICE_PRESETS: [Maya1VoicePreset; 9] = [
    Maya1VoicePreset {
        name: "ava",
        description: "Realistic female voice in the 30s age with a american accent. Normal pitch, \
                      warm timbre, conversational pacing, neutral tone delivery at med intensity, \
                      podcast domain, podcast_host role, neutral delivery.",
    },
    Maya1VoicePreset {
        name: "leo",
        description: "Realistic male voice in the 30s age with a american accent. Normal pitch, \
                      warm timbre, conversational pacing, neutral tone delivery at med intensity, \
                      podcast domain, podcast_host role, neutral delivery.",
    },
    Maya1VoicePreset {
        name: "nora",
        description: "Realistic female voice in the 30s age with a british accent. Normal pitch, \
                      smooth timbre, conversational pacing, neutral tone delivery at med \
                      intensity, corporate domain, explainer_video_voice role, formal delivery.",
    },
    Maya1VoicePreset {
        name: "james",
        description: "Realistic male voice in the 40s age with a british accent. Low pitch, deep \
                      timbre, slow pacing, neutral tone delivery at med intensity, entertainment \
                      domain, storyteller role, neutral delivery.",
    },
    Maya1VoicePreset {
        name: "priya",
        description: "Realistic female voice in the 30s age with a indian accent. Normal pitch, \
                      smooth timbre, conversational pacing, neutral tone delivery at med \
                      intensity, support domain, customer_support_agent role, neutral delivery.",
    },
    Maya1VoicePreset {
        name: "arjun",
        description: "Realistic male voice in the 20s age with a indian accent. Normal pitch, \
                      warm timbre, conversational pacing, energetic tone delivery at med \
                      intensity, education domain, elearning_instructor role, casual delivery.",
    },
    Maya1VoicePreset {
        name: "zoe",
        description: "Realistic female voice in the 20s age with a american accent. High pitch, \
                      smooth timbre, brisk pacing, energetic tone delivery at med intensity, \
                      social_content domain, youtube_vlogger role, casual delivery.",
    },
    Maya1VoicePreset {
        name: "omar",
        description: "Realistic male voice in the 30s age with a middle_eastern accent. Low \
                      pitch, deep timbre, conversational pacing, neutral tone delivery at med \
                      intensity, corporate domain, event_host role, formal delivery.",
    },
    Maya1VoicePreset {
        name: "robot",
        description: "Creative, ai_machine_voice character. Male voice in their 20s with a \
                      american accent. Normal pitch, robotic timbre, conversational pacing, \
                      neutral tone at med intensity.",
    },
];

/// The preset a request gets when it names none.
pub const MAYA1_DEFAULT_VOICE: &str = "ava";

/// OpenAI's voice names, mapped to the preset closest in register.
pub const MAYA1_OPENAI_VOICE_ALIASES: [(&str, &str); 13] = [
    ("alloy", "ava"),
    ("ash", "leo"),
    ("ballad", "james"),
    ("cedar", "omar"),
    ("coral", "zoe"),
    ("echo", "leo"),
    ("fable", "james"),
    ("marin", "priya"),
    ("nova", "ava"),
    ("onyx", "omar"),
    ("sage", "nora"),
    ("shimmer", "zoe"),
    ("verse", "arjun"),
];

/// The preset `name` selects: a preset name or an OpenAI alias, ignoring
/// case.
pub fn maya1_voice_preset(name: &str) -> Option<&'static Maya1VoicePreset> {
    let name = name.trim();
    let target = MAYA1_OPENAI_VOICE_ALIASES
        .iter()
        .find(|(alias, _)| alias.eq_ignore_ascii_case(name))
        .map_or(name, |(_, preset)| preset);
    MAYA1_VOICE_PRESETS
        .iter()
        .find(|preset| preset.name.eq_ignore_ascii_case(target))
}

/// The Llama decoder's geometry from `config.json`, checked against the
/// pinned checkpoint's.
#[derive(Debug, Clone, PartialEq)]
pub struct Maya1LmConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_scaling: RopeScaling,
    pub vocab_size: usize,
}

#[derive(Debug, Deserialize)]
struct RawLmConfig {
    architectures: Vec<String>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    rope_scaling: Option<Value>,
    max_position_embeddings: usize,
    vocab_size: usize,
    tie_word_embeddings: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default)]
    hidden_act: Option<String>,
}

impl Maya1LmConfig {
    /// The pinned checkpoint's geometry (`config.json` at the pin).
    pub fn pinned() -> Self {
        Self {
            hidden_size: 3_072,
            intermediate_size: 8_192,
            num_hidden_layers: 28,
            num_attention_heads: 24,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            rope_scaling: RopeScaling::Llama3(Llama3RopeScaling {
                factor: 32.0,
                low_freq_factor: 1.0,
                high_freq_factor: 4.0,
                original_max_position_embeddings: 8_192,
            }),
            vocab_size: 156_960,
        }
    }

    /// Read `config.json` in `model_dir` and refuse anything but the pinned
    /// geometry: the port, its kernels and its state contract assume it.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join(CONFIG_FILE);
        let text = std::fs::read_to_string(&path).map_err(|error| {
            Error::ModelLoadError(format!(
                "Maya1 config {} unreadable: {error}",
                path.display()
            ))
        })?;
        let raw: RawLmConfig = serde_json::from_str(&text).map_err(|error| {
            Error::ModelLoadError(format!(
                "Maya1 config {} malformed: {error}",
                path.display()
            ))
        })?;
        let parsed = Self::from_raw(&raw)?;
        let pinned = Self::pinned();
        if parsed != pinned {
            return Err(Error::ModelLoadError(format!(
                "Maya1 config {} is not the pinned checkpoint's ({MAYA1_REVISION}): {parsed:?}",
                path.display()
            )));
        }
        Ok(parsed)
    }

    fn from_raw(raw: &RawLmConfig) -> Result<Self> {
        if raw.architectures != ["LlamaForCausalLM"]
            || !raw.tie_word_embeddings
            || raw.attention_bias
            || raw.mlp_bias
            || raw.hidden_act.as_deref().is_some_and(|act| act != "silu")
        {
            return Err(Error::ModelLoadError(
                "Maya1 expects a tied, bias-free SiLU LlamaForCausalLM".into(),
            ));
        }
        let rope_scaling =
            RopeScaling::from_config_value(raw.rope_scaling.as_ref(), raw.max_position_embeddings)?;
        Ok(Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta: raw.rope_theta,
            rope_scaling,
            vocab_size: raw.vocab_size,
        })
    }

    pub fn q_out(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_out(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    /// This geometry with only its first `layers` layers (layer-truncated
    /// reference fixtures).
    pub fn truncated(&self, layers: usize) -> Self {
        Self {
            num_hidden_layers: layers,
            ..self.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_layout_matches_the_tokenizer() {
        assert_eq!(AUDIO_CODE_END - 1, 156_937);
        assert_eq!(HEAD_ROWS, 28_680);
        assert_eq!(SAMPLES_PER_FRAME, 4 * 512);
        assert!((FRAME_RATE_HZ - 11.718_75).abs() < 1e-6);
    }

    #[test]
    fn auto_budget_is_whole_frames_within_the_cap() {
        for text in [
            "",
            "Hi.",
            "Hello there, how are you?",
            "Ha! <laugh> really?",
        ] {
            let budget = maya1_auto_max_new_tokens(text);
            assert_eq!(budget % TOKENS_PER_FRAME, 0);
            assert!(budget <= MAX_NEW_TOKENS);
            assert!(budget >= MAYA1_DEFAULTS.min_new_tokens);
        }
        // A tag costs its allowance, not its characters.
        assert!(
            maya1_auto_max_new_tokens("Ha! <laugh_harder> really?")
                > maya1_auto_max_new_tokens("Ha! really?") + 100
        );
        assert_eq!(maya1_auto_max_new_tokens(&"word ".repeat(200)), 2_044);
    }

    #[test]
    fn presets_resolve_by_name_and_openai_alias() {
        assert_eq!(maya1_voice_preset("Ava").unwrap().name, "ava");
        assert_eq!(maya1_voice_preset("alloy").unwrap().name, "ava");
        assert_eq!(maya1_voice_preset(" ONYX ").unwrap().name, "omar");
        assert!(maya1_voice_preset("unknown").is_none());
        assert!(maya1_voice_preset(MAYA1_DEFAULT_VOICE).is_some());
        for (_, preset) in MAYA1_OPENAI_VOICE_ALIASES {
            assert!(maya1_voice_preset(preset).is_some(), "{preset}");
        }
        for preset in MAYA1_VOICE_PRESETS {
            assert!(preset.description.ends_with('.'));
            assert!(!preset.description.contains('"'));
            assert!(!preset.description.contains("  "), "{}", preset.name);
        }
    }

    #[test]
    fn pinned_config_parses_from_the_published_json() {
        let json = r#"{"architectures":["LlamaForCausalLM"],"attention_bias":false,
            "head_dim":128,"hidden_act":"silu","hidden_size":3072,"intermediate_size":8192,
            "max_position_embeddings":131072,"mlp_bias":false,"num_attention_heads":24,
            "num_hidden_layers":28,"num_key_value_heads":8,"rms_norm_eps":1e-05,
            "rope_scaling":{"factor":32.0,"high_freq_factor":4.0,"low_freq_factor":1.0,
            "original_max_position_embeddings":8192,"rope_type":"llama3"},
            "rope_theta":500000.0,"tie_word_embeddings":true,"vocab_size":156960}"#;
        let raw: RawLmConfig = serde_json::from_str(json).unwrap();
        assert_eq!(
            Maya1LmConfig::from_raw(&raw).unwrap(),
            Maya1LmConfig::pinned()
        );
    }
}
