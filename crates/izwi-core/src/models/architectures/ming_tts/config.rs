//! Ming-3.0-tts-0.5B's published configuration and the constants of its
//! reference pipeline.
//!
//! `config.json` nests four parts: `llm_config` (Qwen2.5-0.5B),
//! `aggregator_config` and `ditar_config` (the two small bidirectional
//! transformers of the flow path) and `audio_tokenizer_config` (the VAE,
//! whose encoder and decoder are Qwen2 stacks of the language model's
//! shape). Everything the port relies on is checked here, so a sibling
//! checkpoint with another layout is refused at load instead of producing
//! worse audio.

use std::path::Path;

use serde_json::Value;

use crate::error::{Error, Result};

pub const CONFIG_FILE: &str = "config.json";
pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const WEIGHTS_FILE: &str = "model.safetensors";
/// CAM++'s PyTorch checkpoint, fetched from `funasr/campplus`.
pub const SPEAKER_WEIGHTS_FILE: &str = "campplus_cn_common.bin";

/// Output sample rate.
pub const SAMPLE_RATE: u32 = 44_100;
/// Samples of one VAE frame (20 ms).
pub const FRAME_SAMPLES: usize = 882;
/// VAE frames per latent.
pub const FRAMES_PER_LATENT: usize = 4;
/// Samples one latent decodes to (80 ms).
pub const LATENT_SAMPLES: usize = FRAME_SAMPLES * FRAMES_PER_LATENT;
/// Width of an audio latent.
pub const LATENT_DIM: usize = 64;
/// Latents per second of audio.
pub const LATENT_RATE_HZ: f32 = SAMPLE_RATE as f32 / LATENT_SAMPLES as f32;
/// Inverse STFT size and window: four frames.
pub const ISTFT_SIZE: usize = LATENT_SAMPLES;
/// Positive-frequency bins of the inverse STFT.
pub const ISTFT_BINS: usize = ISTFT_SIZE / 2 + 1;

/// Positions one request's sequence may reach: the reference decodes on a
/// 512-position cache (`MAX_CACHE_LEN`).
pub const CONTEXT_POSITIONS: usize = 512;
/// The context the engine leases for one request's retained state.
pub const STATE_CONTEXT_TOKENS: usize = CONTEXT_POSITIONS;
/// The most latents one request generates (about 40 s); the context bounds
/// it further by the prompt's length.
pub const MAX_NEW_LATENTS: usize = 500;
/// The stop head is read only after this many latents (`step > 3`).
pub const MIN_LATENTS: usize = 5;
/// The longest reference clip kept, in latents (20 s).
pub const MAX_REFERENCE_LATENTS: usize = 250;
/// The shortest reference clip, in seconds.
pub const MIN_REFERENCE_SECONDS: f32 = 1.0;

/// Euler steps of the flow sampler (`FlowLoss.sample`'s default).
pub const FLOW_STEPS: usize = 10;
/// The sampler's time grid for ten steps, in 32nds (`get_epss_timesteps`).
pub const FLOW_GRID_32: [u32; FLOW_STEPS + 1] = [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32];
/// Default guidance scale.
pub const DEFAULT_GUIDANCE: f32 = 2.0;
/// The stop head's threshold.
pub const STOP_THRESHOLD: f32 = 0.5;

/// Keys a VAE frame attends to, itself included.
///
/// The published config pins the VAE stacks to flash-attn with
/// `sliding_window: 64`; upstream's own requirements leave flash-attn out
/// and run the SDPA path, whose mask hides keys `j <= i - 64`: a frame sees
/// itself and the 63 before it. The flash path would see one more.
/// `IZWI_MING_TTS_VAE_WINDOW_KEYS` overrides the count.
pub const VAE_WINDOW_KEYS: usize = 64;
pub const VAE_WINDOW_KEYS_ENV: &str = "IZWI_MING_TTS_VAE_WINDOW_KEYS";

// Token ids the checkpoint was trained with (`prompt_contract.py`); they
// index embedding rows, so the names alone are not enough.
pub const TOKEN_ENDOFTEXT: u32 = 151_643;
pub const TOKEN_IM_START: u32 = 151_644;
pub const TOKEN_IM_END: u32 = 151_645;
pub const TOKEN_SPEAKER_START: u32 = 151_652;
pub const TOKEN_SPEAKER_END: u32 = 151_653;
pub const TOKEN_SPEAKER_PAD: u32 = 151_654;
pub const TOKEN_AUDIO_PATCH: u32 = 151_705;
pub const TOKEN_AUDIO_START: u32 = 151_706;

/// The (name, id) pairs a tokenizer must reproduce.
pub const PINNED_TOKENS: [(&str, u32); 8] = [
    ("<|endoftext|>", TOKEN_ENDOFTEXT),
    ("<|im_start|>", TOKEN_IM_START),
    ("<|im_end|>", TOKEN_IM_END),
    ("<|vision_start|>", TOKEN_SPEAKER_START),
    ("<|vision_end|>", TOKEN_SPEAKER_END),
    ("<|vision_pad|>", TOKEN_SPEAKER_PAD),
    ("<audioPatch>", TOKEN_AUDIO_PATCH),
    ("<audio>", TOKEN_AUDIO_START),
];

/// A Qwen2 stack: the language model, or one of the VAE's transformers.
#[derive(Debug, Clone, PartialEq)]
pub struct MingStackConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

impl MingStackConfig {
    pub fn q_out(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_out(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    fn pinned(layers: usize) -> Self {
        Self {
            hidden_size: 896,
            num_hidden_layers: layers,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            head_dim: 64,
            intermediate_size: 4_864,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }

    fn parse(value: &Value, what: &str) -> Result<Self> {
        let hidden_size = usize_field(value, "hidden_size", what)?;
        let num_attention_heads = usize_field(value, "num_attention_heads", what)?;
        if num_attention_heads == 0 || !hidden_size.is_multiple_of(num_attention_heads) {
            return Err(invalid(format!(
                "{what} width {hidden_size} is not a multiple of its {num_attention_heads} heads"
            )));
        }
        let cfg = Self {
            hidden_size,
            num_hidden_layers: usize_field(value, "num_hidden_layers", what)?,
            num_attention_heads,
            num_key_value_heads: usize_field(value, "num_key_value_heads", what)?,
            head_dim: hidden_size / num_attention_heads,
            intermediate_size: usize_field(value, "intermediate_size", what)?,
            rms_norm_eps: f64_field(value, "rms_norm_eps", what)?,
            rope_theta: f64_field(value, "rope_theta", what)?,
        };
        if value.get("hidden_act").and_then(Value::as_str) != Some("silu") {
            return Err(invalid(format!("{what} must use the SiLU activation")));
        }
        if cfg.num_key_value_heads == 0
            || !cfg
                .num_attention_heads
                .is_multiple_of(cfg.num_key_value_heads)
            || !cfg.head_dim.is_multiple_of(2)
            || cfg.num_hidden_layers == 0
        {
            return Err(invalid(format!(
                "{what} has an unsupported attention layout"
            )));
        }
        Ok(cfg)
    }
}

/// One of the flow path's bidirectional transformers.
#[derive(Debug, Clone, PartialEq)]
pub struct MingFlowConfig {
    pub hidden_size: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub mlp_size: usize,
}

impl MingFlowConfig {
    fn pinned() -> Self {
        Self {
            hidden_size: 1_024,
            depth: 8,
            num_heads: 16,
            head_dim: 64,
            mlp_size: 4_096,
        }
    }

    fn parse(value: &Value, what: &str) -> Result<Self> {
        let hidden_size = usize_field(value, "hidden_size", what)?;
        let num_heads = usize_field(value, "num_heads", what)?;
        let mlp_ratio = f64_field(value, "mlp_ratio", what)?;
        if num_heads == 0 || !hidden_size.is_multiple_of(num_heads) {
            return Err(invalid(format!("{what} has an unsupported head count")));
        }
        let head_dim = hidden_size / num_heads;
        if !head_dim.is_multiple_of(2) || mlp_ratio <= 0.0 {
            return Err(invalid(format!("{what} has an unsupported layout")));
        }
        Ok(Self {
            hidden_size,
            depth: usize_field(value, "depth", what)?,
            num_heads,
            head_dim,
            mlp_size: (hidden_size as f64 * mlp_ratio) as usize,
        })
    }
}

/// The whole model's configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct MingTtsConfig {
    pub llm: MingStackConfig,
    pub vocab_size: usize,
    pub aggregator: MingFlowConfig,
    pub flow: MingFlowConfig,
    /// The VAE encoder's and decoder's 24-layer stacks.
    pub vae_stack: MingStackConfig,
    /// The VAE encoder's 4-layer stack over groups of frames.
    pub vae_group_stack: MingStackConfig,
}

impl MingTtsConfig {
    /// The configuration of the pinned checkpoint.
    pub fn pinned() -> Self {
        Self {
            llm: MingStackConfig::pinned(24),
            vocab_size: 151_936,
            aggregator: MingFlowConfig::pinned(),
            flow: MingFlowConfig::pinned(),
            vae_stack: MingStackConfig::pinned(24),
            vae_group_stack: MingStackConfig::pinned(4),
        }
    }

    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join(CONFIG_FILE);
        let text = std::fs::read_to_string(&path).map_err(|error| {
            Error::ModelLoadError(format!("Failed to read {}: {error}", path.display()))
        })?;
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Self> {
        let root: Value = serde_json::from_str(text)
            .map_err(|error| invalid(format!("config.json is not JSON: {error}")))?;
        if root.get("model_type").and_then(Value::as_str) != Some("dense") {
            return Err(invalid(
                "only the dense Ming-3.0-tts-0.5B layout is supported (model_type \"dense\")"
                    .into(),
            ));
        }
        let llm_value = object(&root, "llm_config")?;
        let llm = MingStackConfig::parse(llm_value, "llm_config")?;
        let vocab_size = usize_field(llm_value, "vocab_size", "llm_config")?;
        if llm_value
            .get("tie_word_embeddings")
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err(invalid("llm_config must tie its word embeddings".into()));
        }

        let aggregator =
            MingFlowConfig::parse(object(&root, "aggregator_config")?, "aggregator_config")?;
        let ditar = object(&root, "ditar_config")?;
        let flow = MingFlowConfig::parse(ditar, "ditar_config")?;
        let patch = usize_field(ditar, "patch_size", "ditar_config")?;
        let history = ditar
            .get("history_patch_size")
            .and_then(Value::as_u64)
            .map_or(patch, |value| value as usize);
        if patch != 1 || history != 1 {
            return Err(invalid(format!(
                "ditar_config generates {patch} latent(s) a step from {history} of history; \
                 this port supports 1 and 1"
            )));
        }

        let audio = object(&root, "audio_tokenizer_config")?;
        let enc = object(audio, "enc_kwargs")?;
        let dec = object(audio, "dec_kwargs")?;
        if !audio
            .get("semantic_module_kwargs")
            .is_none_or(Value::is_null)
        {
            return Err(invalid(
                "audio_tokenizer_config carries a semantic module this port does not run".into(),
            ));
        }
        let expect = |value: &Value, key: &str, want: usize, what: &str| -> Result<()> {
            let got = usize_field(value, key, what)?;
            if got != want {
                return Err(invalid(format!("{what}.{key} is {got}, expected {want}")));
            }
            Ok(())
        };
        expect(
            audio,
            "sample_rate",
            SAMPLE_RATE as usize,
            "audio_tokenizer_config",
        )?;
        expect(
            audio,
            "patch_size",
            FRAMES_PER_LATENT,
            "audio_tokenizer_config",
        )?;
        expect(enc, "hop_size", FRAME_SAMPLES, "enc_kwargs")?;
        expect(enc, "input_dim", FRAME_SAMPLES, "enc_kwargs")?;
        expect(enc, "latent_dim", LATENT_DIM, "enc_kwargs")?;
        expect(dec, "latent_dim", LATENT_DIM, "dec_kwargs")?;
        expect(dec, "output_dim", FRAME_SAMPLES, "dec_kwargs")?;

        let enc_backbone = object(enc, "backbone")?;
        let dec_backbone = object(dec, "backbone")?;
        let vae_stack = MingStackConfig::parse(dec_backbone, "dec_kwargs.backbone")?;
        if MingStackConfig::parse(enc_backbone, "enc_kwargs.backbone")? != vae_stack {
            return Err(invalid(
                "the VAE encoder and decoder stacks must share one shape".into(),
            ));
        }
        for (backbone, what) in [
            (enc_backbone, "enc_kwargs.backbone"),
            (dec_backbone, "dec_kwargs.backbone"),
        ] {
            expect(backbone, "sliding_window", 64, what)?;
        }
        let mut vae_group_stack = vae_stack.clone();
        vae_group_stack.num_hidden_layers = 4;

        let cfg = Self {
            llm,
            vocab_size,
            aggregator,
            flow,
            vae_stack,
            vae_group_stack,
        };
        if cfg.llm.num_hidden_layers == 0 || cfg.vocab_size <= TOKEN_AUDIO_START as usize {
            return Err(invalid(
                "llm_config is too small for the prompt tokens".into(),
            ));
        }
        Ok(cfg)
    }
}

/// The VAE attention window in keys, the env override applied.
pub fn vae_window_keys() -> usize {
    static KEYS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *KEYS.get_or_init(|| {
        crate::env::value(VAE_WINDOW_KEYS_ENV)
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&keys| (2..=4_096).contains(&keys))
            .unwrap_or(VAE_WINDOW_KEYS)
    })
}

fn invalid(message: String) -> Error {
    Error::ModelLoadError(format!("Ming TTS config: {message}"))
}

fn object<'a>(value: &'a Value, key: &str) -> Result<&'a Value> {
    value
        .get(key)
        .filter(|value| value.is_object())
        .ok_or_else(|| invalid(format!("{key} is missing")))
}

fn usize_field(value: &Value, key: &str, what: &str) -> Result<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .ok_or_else(|| invalid(format!("{what}.{key} is missing")))
}

fn f64_field(value: &Value, key: &str, what: &str) -> Result<f64> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid(format!("{what}.{key} is missing")))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The published `config.json`, reduced to the fields the port reads.
    pub(crate) fn published_config_json() -> String {
        let stack = |extra: &str| {
            format!(
                r#"{{"hidden_act":"silu","hidden_size":896,"intermediate_size":4864,
                "num_attention_heads":14,"num_hidden_layers":24,"num_key_value_heads":2,
                "rms_norm_eps":1e-06,"rope_theta":1000000.0{extra}}}"#
            )
        };
        let flow = r#"{"depth":8,"hidden_size":1024,"num_heads":16,"mlp_ratio":4,"patch_size":1,"history_patch_size":1}"#;
        format!(
            r#"{{"model_type":"dense",
            "llm_config":{},
            "aggregator_config":{flow},
            "ditar_config":{flow},
            "audio_tokenizer_config":{{"sample_rate":44100,"patch_size":4,"semantic_module_kwargs":null,
              "enc_kwargs":{{"hop_size":882,"input_dim":882,"latent_dim":64,"backbone":{}}},
              "dec_kwargs":{{"latent_dim":64,"output_dim":882,"backbone":{}}}}}}}"#,
            stack(r#","vocab_size":151936,"tie_word_embeddings":true,"sliding_window":32768"#),
            stack(r#","sliding_window":64,"vocab_size":1"#),
            stack(r#","sliding_window":64,"vocab_size":1"#),
        )
    }

    #[test]
    fn published_config_parses_to_the_pinned_layout() {
        let cfg = MingTtsConfig::parse(&published_config_json()).unwrap();
        assert_eq!(cfg, MingTtsConfig::pinned());
        assert_eq!(cfg.llm.q_out(), 896);
        assert_eq!(cfg.llm.kv_out(), 128);
        assert_eq!(cfg.flow.mlp_size, 4_096);
    }

    #[test]
    fn a_different_patch_layout_is_refused() {
        let json = published_config_json()
            .replace(r#""history_patch_size":1"#, r#""history_patch_size":32"#);
        let error = MingTtsConfig::parse(&json).unwrap_err().to_string();
        assert!(error.contains("history"), "{error}");
        let json =
            published_config_json().replace(r#""model_type":"dense""#, r#""model_type":"moe""#);
        assert!(MingTtsConfig::parse(&json).is_err());
        let json = published_config_json().replace(r#""hop_size":882"#, r#""hop_size":320"#);
        assert!(MingTtsConfig::parse(&json).is_err());
    }

    #[test]
    fn audio_constants_agree() {
        assert_eq!(LATENT_SAMPLES, 3_528);
        assert_eq!(ISTFT_BINS, 1_765);
        assert!((LATENT_RATE_HZ - 12.5).abs() < 1e-6);
        assert_eq!(FLOW_GRID_32.len(), FLOW_STEPS + 1);
    }
}
