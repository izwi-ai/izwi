//! Maya 2 Native's pinned release, token layout, frame geometry and the
//! serving defaults its release numbers were measured with (upstream
//! `maya3` engine at GitHub `bbfdb152`).

use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};

/// <https://huggingface.co/maya-research/maya2-native> (Apache-2.0).
pub const MAYA2_REPO: &str = "maya-research/maya2-native";
pub const MAYA2_REVISION: &str = "1403b841a6af0f920781319e851d4d6bdeb596c6";
/// Kyutai's Mimi codec (CC-BY-4.0), kept in `mimi/` beside the model.
pub const MIMI_REPO: &str = "kyutai/mimi";
pub const MIMI_REVISION: &str = "89091b3e466eb6a9d11e537bf26b144f194978f7";
pub const MIMI_DIR: &str = "mimi";
/// SpeechBrain's ECAPA-TDNN speaker encoder (Apache-2.0), in `ecapa/`.
pub const ECAPA_REPO: &str = "speechbrain/spkrec-ecapa-voxceleb";
pub const ECAPA_REVISION: &str = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286";
pub const ECAPA_DIR: &str = "ecapa";
pub const ECAPA_WEIGHTS_FILE: &str = "ecapa/embedding_model.ckpt";

pub const RELEASE_CONFIG_FILE: &str = "maya3_config.json";
pub const BACKBONE_CONFIG_FILE: &str = "backbone/config.json";
pub const TOKENIZER_FILE: &str = "backbone/tokenizer.json";
pub const VOICES_FILE: &str = "voices/voices.json";
pub const WEIGHT_SHARDS: [&str; 2] = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
];

// Added tokens (the BPE vocabulary ends at 194,475; Qwen's specials follow,
// then the TTS contract). Checked against the tokenizer at load.
pub const END_OF_TEXT: u32 = 194_476;
pub const AUDIO_SENTINEL: u32 = 194_499;
/// `<|lang:as|>`; the twelve language tokens follow in
/// [`crate::catalog::MAYA2_LANGUAGES`] order.
pub const FIRST_LANGUAGE_TOKEN: u32 = 194_504;
pub const AUDIO_SENTINEL_TEXT: &str = "<|audio|>";

/// 32 Mimi codebooks of 2,048 codes; the backbone predicts codebook 0 plus
/// an end-of-audio class, the depth decoder the other 31.
pub const CODEBOOKS: usize = 32;
pub const CODEBOOK_SIZE: usize = 2_048;
pub const AEOS_CLASS: usize = CODEBOOK_SIZE;
pub const CB0_CLASSES: usize = CODEBOOK_SIZE + 1;
pub const DEPTH_LEVELS: usize = CODEBOOKS - 1;

/// 24 kHz mono at 12.5 frames per second.
pub const SAMPLE_RATE: u32 = 24_000;
pub const SAMPLES_PER_FRAME: usize = 1_920;
pub const FRAME_RATE_HZ: f32 = 12.5;
pub const FRAME_SECONDS: f32 = 0.08;

/// Positions one sequence may reach: upstream's per-seat capacity (prompt
/// plus generated frames).
pub const STATE_CONTEXT_TOKENS: usize = 2_048;
/// The absolute frame ceiling per piece (120 s).
pub const HARD_CAP_FRAMES: usize = 1_500;
/// The fewest frames a prompt must leave room for (upstream refuses less).
pub const MIN_FRAME_BUDGET: usize = 8;

/// Reference clips for cloning: 0.5–30 s (upstream's server bounds); 3–10 s
/// is what the release recommends.
pub const MIN_REFERENCE_SECONDS: f32 = 0.5;
pub const MAX_REFERENCE_SECONDS: f32 = 30.0;

/// The cb0 sampler upstream ships (a sweep's best stability setting):
/// temperature 0.9, top-k 64 over all 2,049 classes, no nucleus, a presence
/// penalty of 0.8 for each distinct code of the last 12 frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Maya2SamplingDefaults {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub presence_penalty: f32,
    pub presence_window: usize,
    pub depth_temperature: f32,
    pub depth_top_k: usize,
}

pub const MAYA2_DEFAULTS: Maya2SamplingDefaults = Maya2SamplingDefaults {
    temperature: 0.9,
    top_k: 64,
    top_p: 1.0,
    presence_penalty: 0.8,
    presence_window: 12,
    depth_temperature: 0.7,
    depth_top_k: 64,
};

/// Request bounds on the cb0 sampler (upstream's server ranges).
pub const TEMPERATURE_RANGE: (f32, f32) = (0.1, 1.5);
pub const PRESENCE_PENALTY_RANGE: (f32, f32) = (0.0, 5.0);

/// Controlled stop: once the minimum length is reached, stop when AEOS is
/// the argmax, when p(AEOS) ≥ 0.35, or when the cumulative hazard
/// 1 − Π(1 − p(AEOS)) ≥ 0.6.
pub const STOP_TAU: f32 = 0.35;
pub const STOP_HAZARD: f32 = 0.6;

/// The backbone's geometry (`backbone/config.json`, a Qwen3-1.7B with an
/// extended vocabulary), checked against the pinned checkpoint's.
#[derive(Debug, Clone, PartialEq)]
pub struct Maya2BackboneConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
}

#[derive(Debug, Deserialize)]
struct RawRopeParameters {
    rope_theta: f64,
    rope_type: String,
}

#[derive(Debug, Deserialize)]
struct RawBackboneConfig {
    architectures: Vec<String>,
    attention_bias: bool,
    head_dim: usize,
    hidden_act: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f64,
    rope_parameters: RawRopeParameters,
    tie_word_embeddings: bool,
    #[serde(default)]
    use_sliding_window: bool,
    vocab_size: usize,
}

impl Maya2BackboneConfig {
    /// The pinned checkpoint's backbone.
    pub fn pinned() -> Self {
        Self {
            hidden_size: 2_048,
            intermediate_size: 6_144,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            vocab_size: 194_548,
        }
    }

    /// Read `backbone/config.json` and refuse anything but the pinned
    /// geometry: the port implements exactly this Qwen3.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join(BACKBONE_CONFIG_FILE);
        let raw: RawBackboneConfig = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        let rope = &raw.rope_parameters;
        if raw.architectures != ["Qwen3ForCausalLM"]
            || raw.attention_bias
            || raw.hidden_act != "silu"
            || !raw.tie_word_embeddings
            || raw.use_sliding_window
            || rope.rope_type != "default"
        {
            return Err(Error::ModelLoadError(format!(
                "{} is not the Qwen3 backbone Maya2 pins",
                path.display()
            )));
        }
        let cfg = Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            rms_norm_eps: raw.rms_norm_eps,
            rope_theta: rope.rope_theta,
            vocab_size: raw.vocab_size,
        };
        if cfg != Self::pinned() {
            return Err(Error::ModelLoadError(format!(
                "Maya2 backbone geometry {cfg:?} differs from the pinned release"
            )));
        }
        Ok(cfg)
    }

    /// The first `layers` layers only: the layer-truncated parity fixtures.
    #[cfg(test)]
    pub(crate) fn truncated(mut self, layers: usize) -> Self {
        self.num_hidden_layers = layers;
        self
    }

    pub fn q_out(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_out(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

#[derive(Debug, Deserialize)]
struct RawSpeakerAnchor {
    enabled: bool,
    hidden: usize,
    input_dim: usize,
}

#[derive(Debug, Deserialize)]
struct RawReleaseModel {
    speaker_anchor: RawSpeakerAnchor,
    depth_grouped: bool,
}

#[derive(Debug, Deserialize)]
struct RawReleaseAudio {
    sample_rate: u32,
    n_codebooks: usize,
    codebook_size: usize,
    codec: String,
}

#[derive(Debug, Deserialize)]
struct RawReleaseConfig {
    format: String,
    format_version: u32,
    model_type: String,
    key_prefix: String,
    model: RawReleaseModel,
    audio: RawReleaseAudio,
    languages: Vec<String>,
}

/// Check `maya3_config.json` describes the release this port implements:
/// its format, the speaker anchor (ECAPA-192 → 512 → backbone), the grouped
/// depth decoder, Mimi's 32 × 2,048 codes at 24 kHz and the twelve
/// languages in token order.
pub fn check_release_config(model_dir: &Path) -> Result<()> {
    let path = model_dir.join(RELEASE_CONFIG_FILE);
    let raw: RawReleaseConfig = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let languages = crate::catalog::MAYA2_LANGUAGES
        .iter()
        .map(|language| language.code.to_string())
        .collect::<Vec<_>>();
    let anchor = &raw.model.speaker_anchor;
    let ok = raw.format == "maya3-safetensors"
        && raw.format_version == 1
        && raw.model_type == "maya3"
        && raw.key_prefix == "model."
        && anchor.enabled
        && anchor.hidden == 512
        && anchor.input_dim == 192
        && raw.model.depth_grouped
        && raw.audio.sample_rate == SAMPLE_RATE
        && raw.audio.n_codebooks == CODEBOOKS
        && raw.audio.codebook_size == CODEBOOK_SIZE
        && raw.audio.codec == MIMI_REPO
        && raw.languages == languages;
    if !ok {
        return Err(Error::ModelLoadError(format!(
            "{} does not describe the Maya2 release this port implements",
            path.display()
        )));
    }
    Ok(())
}

/// The `<|lang:xx|>` token of `code` (a model language).
pub fn language_token(code: &str) -> Result<u32> {
    crate::catalog::MAYA2_LANGUAGES
        .iter()
        .position(|language| language.code == code)
        .map(|index| FIRST_LANGUAGE_TOKEN + index as u32)
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "Maya 2 Native does not speak `{code}`; it speaks {}",
                crate::catalog::MAYA2_LANGUAGES
                    .iter()
                    .map(|language| language.code)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_tokens_follow_the_release_order() {
        assert_eq!(language_token("as").unwrap(), 194_504);
        assert_eq!(language_token("te").unwrap(), 194_515);
        assert!(language_token("fr").is_err());
    }
}
