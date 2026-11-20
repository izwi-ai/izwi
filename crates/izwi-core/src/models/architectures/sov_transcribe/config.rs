//! Config parsing for SOV-Transcribe-1.
//!
//! `config.json` carries the Whisper encoder (`audio_config`), the Gemma 4
//! MoE decoder (`text_config`) and the projector's frame stacking;
//! `preprocessor_config.json` the log-mel front end and
//! `generation_config.json` the stop tokens.
//!
//! Every switch the family implements one way is checked here and fails
//! closed: a checkpoint that changed an activation, the routing, the layer
//! pattern or a Gemma 4 feature this family does not build (per-layer
//! embeddings, shared KV layers, the double-wide MLP) would otherwise load
//! and produce plausible garbage. Dimensions are read, checked for internal
//! consistency, and then pinned tensor by tensor by the checkpoint census
//! (`checkpoint.rs`).

use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::error::{Error, Result};

pub const MODEL_TYPE: &str = "sovereignai_transcribe";
pub const ARCHITECTURE: &str = "SovereignAITranscribeForConditionalGeneration";
pub const CONFIG_FILE: &str = "config.json";
pub const PREPROCESSOR_CONFIG_FILE: &str = "preprocessor_config.json";
pub const GENERATION_CONFIG_FILE: &str = "generation_config.json";
pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const TOKENIZER_CONFIG_FILE: &str = "tokenizer_config.json";

/// The log-mel front end (`WhisperFeatureExtractor`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SovFrontendConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    /// Samples of one window (30 s).
    pub window_samples: usize,
    /// Mel frames of one window.
    pub window_frames: usize,
}

/// The Whisper encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SovEncoderConfig {
    pub n_mels: usize,
    pub width: usize,
    pub layers: usize,
    pub heads: usize,
    pub ffn: usize,
    /// Encoder frames of one window (the learned position table's rows).
    pub positions: usize,
}

/// One attention geometry of the decoder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SovAttentionGeometry {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Whether the layer has its own `v_proj` (full layers reuse `k_proj`).
    pub own_values: bool,
    pub rope_theta: f64,
    /// Leading frequency pairs that rotate; the rest pass through.
    pub rotary_pairs: usize,
    /// The denominator of the frequency exponent.
    pub rope_dim: usize,
    /// Visible keys per query, the query's own position included.
    pub window: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SovLayerKind {
    Sliding,
    Full,
}

/// The Gemma 4 MoE decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct SovDecoderConfig {
    pub hidden: usize,
    pub vocab: usize,
    pub layer_kinds: Vec<SovLayerKind>,
    pub sliding: SovAttentionGeometry,
    pub full: SovAttentionGeometry,
    pub mlp_width: usize,
    pub experts: usize,
    pub top_k: usize,
    pub expert_width: usize,
    pub norm_eps: f64,
    pub final_logit_softcap: Option<f32>,
}

impl SovDecoderConfig {
    pub fn layers(&self) -> usize {
        self.layer_kinds.len()
    }

    pub fn geometry(&self, layer: usize) -> &SovAttentionGeometry {
        match self.layer_kinds[layer] {
            SovLayerKind::Sliding => &self.sliding,
            SovLayerKind::Full => &self.full,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SovTranscribeConfig {
    pub frontend: SovFrontendConfig,
    pub encoder: SovEncoderConfig,
    pub decoder: SovDecoderConfig,
    /// Encoder frames stacked into one audio token.
    pub downsample_factor: usize,
    pub audio_token_id: u32,
    pub bos_token_id: u32,
    pub eos_token_ids: Vec<u32>,
}

fn invalid(message: impl std::fmt::Display) -> Error {
    Error::ModelLoadError(format!("SOV-Transcribe config: {message}"))
}

fn read_json(dir: &Path, file: &str) -> Result<Value> {
    let path = dir.join(file);
    let bytes = fs::read(&path).map_err(|err| invalid(format!("{}: {err}", path.display())))?;
    serde_json::from_slice(&bytes).map_err(|err| invalid(format!("{}: {err}", path.display())))
}

fn usize_of(value: &Value, key: &str) -> Result<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| invalid(format!("`{key}` is missing or not an unsigned integer")))
}

fn f64_of(value: &Value, key: &str) -> Result<f64> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid(format!("`{key}` is missing or not a number")))
}

fn str_of<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("`{key}` is missing or not a string")))
}

/// Requires `value[key]` to equal `expected` when present, and to be present
/// unless `default_ok`.
fn require(value: &Value, key: &str, expected: Value, default_ok: bool) -> Result<()> {
    match value.get(key) {
        Some(found) if *found == expected => Ok(()),
        Some(found) => Err(invalid(format!(
            "`{key}` is {found}; this family implements {expected} only"
        ))),
        None if default_ok => Ok(()),
        None => Err(invalid(format!("`{key}` is missing"))),
    }
}

impl SovTranscribeConfig {
    pub fn load(dir: &Path) -> Result<Self> {
        Self::parse(
            &read_json(dir, CONFIG_FILE)?,
            &read_json(dir, PREPROCESSOR_CONFIG_FILE)?,
            &read_json(dir, GENERATION_CONFIG_FILE)?,
        )
    }

    pub fn parse(config: &Value, preprocessor: &Value, generation: &Value) -> Result<Self> {
        require(config, "model_type", MODEL_TYPE.into(), false)?;
        require(config, "router_dtype", "float32".into(), false)?;
        require(config, "projector_hidden_act", "gelu".into(), true)?;
        let frontend = Self::parse_frontend(preprocessor)?;
        let audio = config
            .get("audio_config")
            .ok_or_else(|| invalid("`audio_config` is missing"))?;
        let encoder = Self::parse_encoder(audio)?;
        let text = config
            .get("text_config")
            .ok_or_else(|| invalid("`text_config` is missing"))?;
        let decoder = Self::parse_decoder(text)?;
        let downsample_factor = usize_of(config, "downsample_factor")?;
        if encoder.n_mels != frontend.n_mels {
            return Err(invalid(
                "the encoder and the front end disagree on mel bins",
            ));
        }
        if frontend.window_frames != encoder.positions * 2 {
            return Err(invalid(format!(
                "a {}-frame window does not fill the encoder's {} positions",
                frontend.window_frames, encoder.positions
            )));
        }
        if downsample_factor == 0 || !encoder.positions.is_multiple_of(downsample_factor) {
            return Err(invalid("the frame stacking does not divide a window"));
        }
        let eos_token_ids = match generation.get("eos_token_id") {
            Some(Value::Array(ids)) => ids
                .iter()
                .map(|id| id.as_u64().map(|id| id as u32))
                .collect::<Option<Vec<_>>>(),
            Some(Value::Number(id)) => id.as_u64().map(|id| vec![id as u32]),
            _ => None,
        }
        .filter(|ids| !ids.is_empty())
        .ok_or_else(|| invalid("`eos_token_id` is missing from the generation config"))?;
        let config = Self {
            frontend,
            encoder,
            decoder,
            downsample_factor,
            audio_token_id: usize_of(config, "audio_token_id")? as u32,
            bos_token_id: usize_of(config, "bos_token_id")? as u32,
            eos_token_ids,
        };
        if config.audio_token_id as usize >= config.decoder.vocab {
            return Err(invalid("the audio token is outside the vocabulary"));
        }
        Ok(config)
    }

    fn parse_frontend(p: &Value) -> Result<SovFrontendConfig> {
        require(
            p,
            "feature_extractor_type",
            "WhisperFeatureExtractor".into(),
            true,
        )?;
        require(p, "padding_value", 0.0.into(), true)?;
        require(p, "dither", 0.0.into(), true)?;
        let frontend = SovFrontendConfig {
            sample_rate: usize_of(p, "sampling_rate")?,
            n_fft: usize_of(p, "n_fft")?,
            hop_length: usize_of(p, "hop_length")?,
            n_mels: usize_of(p, "feature_size")?,
            window_samples: usize_of(p, "n_samples")?,
            window_frames: usize_of(p, "nb_max_frames")?,
        };
        if frontend.hop_length == 0
            || frontend.window_frames == 0
            || frontend.window_samples != frontend.window_frames * frontend.hop_length
        {
            return Err(invalid("the window is not a whole number of hops"));
        }
        Ok(frontend)
    }

    fn parse_encoder(a: &Value) -> Result<SovEncoderConfig> {
        require(a, "model_type", "whisper".into(), true)?;
        require(a, "activation_function", "gelu".into(), false)?;
        require(a, "scale_embedding", false.into(), true)?;
        let encoder = SovEncoderConfig {
            n_mels: usize_of(a, "num_mel_bins")?,
            width: usize_of(a, "d_model")?,
            layers: usize_of(a, "encoder_layers")?,
            heads: usize_of(a, "encoder_attention_heads")?,
            ffn: usize_of(a, "encoder_ffn_dim")?,
            positions: usize_of(a, "max_source_positions")?,
        };
        if encoder.heads == 0 || !encoder.width.is_multiple_of(encoder.heads) || encoder.layers == 0
        {
            return Err(invalid("the encoder width is not a whole number of heads"));
        }
        Ok(encoder)
    }

    fn parse_decoder(t: &Value) -> Result<SovDecoderConfig> {
        require(t, "model_type", "gemma4_text".into(), true)?;
        require(t, "hidden_activation", "gelu_pytorch_tanh".into(), false)?;
        require(t, "enable_moe_block", true.into(), false)?;
        require(t, "attention_k_eq_v", true.into(), false)?;
        require(t, "use_double_wide_mlp", false.into(), true)?;
        require(t, "num_kv_shared_layers", 0.into(), true)?;
        require(t, "hidden_size_per_layer_input", 0.into(), true)?;
        require(t, "attention_bias", false.into(), true)?;
        require(t, "tie_word_embeddings", true.into(), true)?;
        // "vision": only image tokens attend bidirectionally. The model has
        // none; "all" would make the decoder non-causal.
        match t.get("use_bidirectional_attention") {
            None | Some(Value::Null) => {}
            Some(Value::String(mode)) if mode == "vision" => {}
            Some(other) => {
                return Err(invalid(format!(
                    "`use_bidirectional_attention` is {other}; the decoder is causal"
                )))
            }
        }
        let hidden = usize_of(t, "hidden_size")?;
        let heads = usize_of(t, "num_attention_heads")?;
        let head_dim = usize_of(t, "head_dim")?;
        let window = usize_of(t, "sliding_window")?;
        let layer_types = t
            .get("layer_types")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("`layer_types` is missing"))?;
        let layer_kinds = layer_types
            .iter()
            .map(|kind| match kind.as_str() {
                Some("sliding_attention") => Ok(SovLayerKind::Sliding),
                Some("full_attention") => Ok(SovLayerKind::Full),
                _ => Err(invalid(format!("unknown layer type {kind}"))),
            })
            .collect::<Result<Vec<_>>>()?;
        if layer_kinds.len() != usize_of(t, "num_hidden_layers")? || layer_kinds.is_empty() {
            return Err(invalid("`layer_types` does not list every layer"));
        }
        let rope = t
            .get("rope_parameters")
            .ok_or_else(|| invalid("`rope_parameters` is missing"))?;
        let sliding_rope = rope
            .get("sliding_attention")
            .ok_or_else(|| invalid("`rope_parameters.sliding_attention` is missing"))?;
        let full_rope = rope
            .get("full_attention")
            .ok_or_else(|| invalid("`rope_parameters.full_attention` is missing"))?;
        require(sliding_rope, "rope_type", "default".into(), false)?;
        require(full_rope, "rope_type", "proportional".into(), false)?;
        let sliding = SovAttentionGeometry {
            heads,
            kv_heads: usize_of(t, "num_key_value_heads")?,
            head_dim,
            own_values: true,
            rope_theta: f64_of(sliding_rope, "rope_theta")?,
            rotary_pairs: head_dim / 2,
            rope_dim: head_dim,
            window: Some(window),
        };
        let global_head_dim = usize_of(t, "global_head_dim")?;
        let partial = f64_of(full_rope, "partial_rotary_factor")?;
        // transformers' `_compute_proportional_rope_parameters`.
        let rotary_pairs = (partial * global_head_dim as f64 / 2.0).floor() as usize;
        let full = SovAttentionGeometry {
            heads,
            kv_heads: usize_of(t, "num_global_key_value_heads")?,
            head_dim: global_head_dim,
            own_values: false,
            rope_theta: f64_of(full_rope, "rope_theta")?,
            rotary_pairs,
            rope_dim: global_head_dim,
            window: None,
        };
        for geometry in [&sliding, &full] {
            if geometry.heads == 0
                || geometry.kv_heads == 0
                || !geometry.heads.is_multiple_of(geometry.kv_heads)
                || geometry.head_dim == 0
                || !geometry.head_dim.is_multiple_of(2)
                || geometry.rotary_pairs == 0
                || geometry.rotary_pairs > geometry.head_dim / 2
                || geometry.window == Some(0)
            {
                return Err(invalid("an attention geometry is inconsistent"));
            }
        }
        let softcap = match t.get("final_logit_softcapping") {
            None | Some(Value::Null) => None,
            Some(value) => Some(
                value
                    .as_f64()
                    .filter(|cap| *cap > 0.0)
                    .ok_or_else(|| invalid("`final_logit_softcapping` is not positive"))?
                    as f32,
            ),
        };
        let decoder = SovDecoderConfig {
            hidden,
            vocab: usize_of(t, "vocab_size")?,
            layer_kinds,
            sliding,
            full,
            mlp_width: usize_of(t, "intermediate_size")?,
            experts: usize_of(t, "num_experts")?,
            top_k: usize_of(t, "top_k_experts")?,
            expert_width: usize_of(t, "moe_intermediate_size")?,
            norm_eps: f64_of(t, "rms_norm_eps")?,
            final_logit_softcap: softcap,
        };
        if decoder.hidden == 0
            || decoder.vocab == 0
            || decoder.mlp_width == 0
            || decoder.expert_width == 0
            || decoder.top_k == 0
            || decoder.top_k > decoder.experts
        {
            return Err(invalid("the decoder geometry is inconsistent"));
        }
        let _ = str_of(t, "hidden_activation")?;
        Ok(decoder)
    }

    /// Encoder frames of one window after the stride-2 convolution.
    pub fn tokens_per_window(&self) -> usize {
        self.encoder.positions / self.downsample_factor
    }

    /// 30 s windows covering `samples` (at least one).
    pub fn windows_for_samples(&self, samples: usize) -> usize {
        samples.div_ceil(self.frontend.window_samples).max(1)
    }

    pub fn audio_tokens_for_samples(&self, samples: usize) -> usize {
        self.windows_for_samples(samples) * self.tokens_per_window()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The published configs (`tests/fixtures/sov_transcribe/`).
    pub(crate) fn published() -> SovTranscribeConfig {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sov_transcribe");
        SovTranscribeConfig::load(&dir).expect("the published config parses")
    }

    #[test]
    fn the_published_config_has_the_gemma4_moe_geometry() {
        let config = published();
        assert_eq!(config.frontend.n_mels, 128);
        assert_eq!(config.frontend.window_samples, 480_000);
        assert_eq!(
            (
                config.encoder.layers,
                config.encoder.width,
                config.encoder.positions
            ),
            (32, 1280, 1500)
        );
        assert_eq!(config.tokens_per_window(), 375);
        assert_eq!(config.audio_tokens_for_samples(1), 375);
        assert_eq!(config.audio_tokens_for_samples(480_000), 375);
        assert_eq!(config.audio_tokens_for_samples(480_001), 750);
        assert_eq!(config.audio_token_id, 258_881);
        assert_eq!(config.eos_token_ids, vec![1, 106, 50]);
        let d = &config.decoder;
        assert_eq!((d.hidden, d.vocab, d.layers()), (2816, 262_144, 30));
        let full = d
            .layer_kinds
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == SovLayerKind::Full)
            .map(|(layer, _)| layer)
            .collect::<Vec<_>>();
        assert_eq!(full, vec![5, 11, 17, 23, 29]);
        assert_eq!(
            (d.sliding.heads, d.sliding.kv_heads, d.sliding.head_dim),
            (16, 8, 256)
        );
        assert_eq!(
            (d.sliding.rotary_pairs, d.sliding.window),
            (128, Some(1024))
        );
        assert_eq!(
            (d.full.heads, d.full.kv_heads, d.full.head_dim),
            (16, 2, 512)
        );
        // Proportional RoPE: 0.25 of the 256 pairs rotate, theta 1e6 over 512.
        assert_eq!((d.full.rotary_pairs, d.full.rope_dim), (64, 512));
        assert!(!d.full.own_values && d.sliding.own_values);
        assert_eq!(
            (d.experts, d.top_k, d.expert_width, d.mlp_width),
            (128, 8, 704, 2112)
        );
        assert_eq!(d.final_logit_softcap, Some(30.0));
    }

    #[test]
    fn a_changed_switch_fails_closed() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/sov_transcribe");
        let base = read_json(&dir, CONFIG_FILE).unwrap();
        let preprocessor = read_json(&dir, PREPROCESSOR_CONFIG_FILE).unwrap();
        let generation = read_json(&dir, GENERATION_CONFIG_FILE).unwrap();
        for (key, value) in [
            ("enable_moe_block", Value::from(false)),
            ("attention_k_eq_v", Value::from(false)),
            ("use_double_wide_mlp", Value::from(true)),
            ("num_kv_shared_layers", Value::from(2)),
            ("hidden_size_per_layer_input", Value::from(256)),
            ("hidden_activation", Value::from("silu")),
            ("use_bidirectional_attention", Value::from("all")),
        ] {
            let mut config = base.clone();
            config["text_config"][key] = value;
            assert!(
                SovTranscribeConfig::parse(&config, &preprocessor, &generation).is_err(),
                "{key} must fail closed"
            );
        }
        let mut config = base.clone();
        config["router_dtype"] = "bfloat16".into();
        assert!(SovTranscribeConfig::parse(&config, &preprocessor, &generation).is_err());
    }
}
