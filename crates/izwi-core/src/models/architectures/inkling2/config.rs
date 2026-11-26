//! The published `config.json` and the geometry derived from it.
//!
//! The trunk has `layers` decoder layers of weights. Layers
//! `loop_start..loop_end` run `loop_passes` times, so a token passes through
//! [`Inkling2Config::applications`] layer applications; each application has
//! its own attention and convolution state.

use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};

pub const CONFIG_FILE: &str = "config.json";
/// Convolution states of one layer application: key, value, after
/// attention, after the MLP.
pub const CONV_STATES_PER_APPLICATION: usize = 4;

fn config_error(message: impl std::fmt::Display) -> Error {
    Error::ModelLoadError(format!("Inkling 2 config: {message}"))
}

#[derive(Debug, Clone, Deserialize)]
struct RawLoop {
    core_start: usize,
    core_end: usize,
    num_passes: usize,
    gate_rank: usize,
    #[serde(default)]
    is_varloop: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct RawText {
    hidden_size: usize,
    num_hidden_layers: usize,
    vocab_size: usize,
    unpadded_vocab_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    swa_num_attention_heads: usize,
    swa_num_key_value_heads: usize,
    swa_head_dim: usize,
    sliding_window_size: usize,
    d_rel: usize,
    rel_extent: usize,
    log_scaling_n_floor: f64,
    log_scaling_alpha: f64,
    rms_norm_eps: f64,
    qk_l2_clip_threshold: f64,
    rel_l2_clip_threshold: f64,
    local_layer_ids: Vec<usize>,
    sconv_kernel_size: usize,
    final_logit_softcapping: f64,
    n_layers_per_attn_res_block: usize,
    n_routed_experts: usize,
    num_experts_per_tok: usize,
    #[serde(default)]
    n_shared_experts: usize,
    intermediate_size: usize,
    mlp_gate_softcap: f64,
    mlp_up_softcap: f64,
    route_scale: f64,
    loop_config: RawLoop,
    #[serde(default)]
    model_max_length: Option<usize>,
    #[serde(default = "enabled")]
    use_value_norm_no_gamma: bool,
    #[serde(default = "enabled")]
    attn_res_normalize_values: bool,
    #[serde(default = "enabled")]
    use_gate_bias: bool,
    #[serde(default)]
    q_bias: bool,
    #[serde(default)]
    o_bias: bool,
}

fn enabled() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
struct RawVision {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    mlp_ratio: f64,
    native_patch_size: usize,
    #[serde(default = "one")]
    temporal_patch_size: usize,
    adapter_merge_size: usize,
    projection_hidden_dim: usize,
    #[serde(default)]
    projector_bias: bool,
    text_hidden_size: usize,
    posemb_grid_size: usize,
    layer_norm_eps: f64,
}

fn one() -> usize {
    1
}

#[derive(Debug, Clone, Deserialize)]
struct RawConfig {
    model_type: String,
    eos_token_id: u32,
    text_config: RawText,
    #[serde(default)]
    vision_config: Option<RawVision>,
    #[serde(default)]
    image_token_id: Option<u32>,
}

/// Attention form of a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inkling2LayerKind {
    /// Sliding-window attention with static key/value convolutions.
    Sliding,
    /// Full attention with per-token key/value convolution taps and log
    /// scaling.
    Full,
}

/// Head geometry of one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inkling2Heads {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Backward distances the relative-position bias covers.
    pub rel_extent: usize,
    /// Keys a query sees, itself included (`None`: all of them).
    pub window: Option<usize>,
}

impl Inkling2Heads {
    pub fn query_width(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

/// One pass of a token through one layer's weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inkling2Application {
    /// Index into the layer weights.
    pub layer: usize,
    /// Loop pass (0 outside the loop and on its first pass).
    pub pass: usize,
    /// Position of the layer inside the loop, for loop layers.
    pub loop_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2VisionConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub mlp_width: usize,
    pub patch: usize,
    pub temporal_patch: usize,
    pub merge: usize,
    pub projection_hidden: usize,
    pub grid: usize,
    pub layer_norm_eps: f64,
}

impl Inkling2VisionConfig {
    /// Pixels per image token along one side.
    pub fn cell(&self) -> usize {
        self.patch * self.merge
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2Config {
    pub hidden: usize,
    pub layers: usize,
    /// Rows of the stored embedding and unembedding tables.
    pub vocab: usize,
    /// Rows the model uses; the rest of the unembedding is padding.
    pub unpadded_vocab: usize,
    pub full: Inkling2Heads,
    pub sliding: Inkling2Heads,
    pub layer_kinds: Vec<Inkling2LayerKind>,
    pub d_rel: usize,
    pub log_scaling_n_floor: f64,
    pub log_scaling_alpha: f64,
    pub rms_norm_eps: f64,
    pub qk_l2_clip_threshold: f64,
    pub rel_l2_clip_threshold: f64,
    pub conv_kernel: usize,
    pub final_logit_softcap: f64,
    /// Layers per depth-residual block.
    pub block_layers: usize,
    pub experts: usize,
    pub experts_per_token: usize,
    pub expert_width: usize,
    pub gate_softcap: f64,
    pub up_softcap: f64,
    pub route_scale: f64,
    pub loop_start: usize,
    pub loop_end: usize,
    pub loop_passes: usize,
    pub loop_gate_rank: usize,
    pub eos_token_id: u32,
    pub image_token_id: u32,
    /// Context the checkpoint was trained for.
    pub max_context_tokens: usize,
    pub vision: Option<Inkling2VisionConfig>,
}

/// The image placeholder of the published tokenizer (`<|unused_200054|>`).
pub const DEFAULT_IMAGE_TOKEN_ID: u32 = 200_054;
const DEFAULT_MAX_CONTEXT_TOKENS: usize = 1_048_576;

impl Inkling2Config {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join(CONFIG_FILE);
        let bytes = std::fs::read(&path)
            .map_err(|err| config_error(format!("{}: {err}", path.display())))?;
        Self::from_json(&bytes)
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self> {
        let raw: RawConfig = serde_json::from_slice(bytes).map_err(config_error)?;
        if raw.model_type != "inkling2_loop" {
            return Err(config_error(format!(
                "model_type is {:?}, expected \"inkling2_loop\"",
                raw.model_type
            )));
        }
        let t = raw.text_config;
        // Features the trunk has no code for must fail the load, not be
        // ignored.
        for (name, ok) in [
            ("n_shared_experts == 0", t.n_shared_experts == 0),
            ("use_value_norm_no_gamma", t.use_value_norm_no_gamma),
            ("attn_res_normalize_values", t.attn_res_normalize_values),
            ("use_gate_bias", t.use_gate_bias),
            ("q_bias == false", !t.q_bias),
            ("o_bias == false", !t.o_bias),
            ("loop_config.is_varloop == false", !t.loop_config.is_varloop),
        ] {
            if !ok {
                return Err(config_error(format!("unsupported setting: not {name}")));
            }
        }
        let layers = t.num_hidden_layers;
        if let Some(layer) = t.local_layer_ids.iter().find(|layer| **layer >= layers) {
            return Err(config_error(format!(
                "local_layer_ids names layer {layer} of {layers}"
            )));
        }
        let layer_kinds = (0..layers)
            .map(|layer| {
                if t.local_layer_ids.contains(&layer) {
                    Inkling2LayerKind::Sliding
                } else {
                    Inkling2LayerKind::Full
                }
            })
            .collect::<Vec<_>>();
        let config = Self {
            hidden: t.hidden_size,
            layers,
            vocab: t.vocab_size,
            unpadded_vocab: t.unpadded_vocab_size,
            full: Inkling2Heads {
                heads: t.num_attention_heads,
                kv_heads: t.num_key_value_heads,
                head_dim: t.head_dim,
                rel_extent: t.rel_extent,
                window: None,
            },
            sliding: Inkling2Heads {
                heads: t.swa_num_attention_heads,
                kv_heads: t.swa_num_key_value_heads,
                head_dim: t.swa_head_dim,
                rel_extent: t.sliding_window_size,
                window: Some(t.sliding_window_size),
            },
            layer_kinds,
            d_rel: t.d_rel,
            log_scaling_n_floor: t.log_scaling_n_floor,
            log_scaling_alpha: t.log_scaling_alpha,
            rms_norm_eps: t.rms_norm_eps,
            qk_l2_clip_threshold: t.qk_l2_clip_threshold,
            rel_l2_clip_threshold: t.rel_l2_clip_threshold,
            conv_kernel: t.sconv_kernel_size,
            final_logit_softcap: t.final_logit_softcapping,
            block_layers: t.n_layers_per_attn_res_block,
            experts: t.n_routed_experts,
            experts_per_token: t.num_experts_per_tok,
            expert_width: t.intermediate_size,
            gate_softcap: t.mlp_gate_softcap,
            up_softcap: t.mlp_up_softcap,
            route_scale: t.route_scale,
            loop_start: t.loop_config.core_start,
            loop_end: t.loop_config.core_end,
            loop_passes: t.loop_config.num_passes,
            loop_gate_rank: t.loop_config.gate_rank,
            eos_token_id: raw.eos_token_id,
            image_token_id: raw.image_token_id.unwrap_or(DEFAULT_IMAGE_TOKEN_ID),
            max_context_tokens: t.model_max_length.unwrap_or(DEFAULT_MAX_CONTEXT_TOKENS),
            vision: raw
                .vision_config
                .map(|v| {
                    if v.text_hidden_size != t.hidden_size {
                        return Err(config_error(
                            "vision_config.text_hidden_size differs from the text hidden size",
                        ));
                    }
                    if v.projector_bias {
                        return Err(config_error("unsupported setting: projector_bias"));
                    }
                    Ok(Inkling2VisionConfig {
                        hidden: v.hidden_size,
                        layers: v.num_hidden_layers,
                        heads: v.num_attention_heads,
                        mlp_width: (v.hidden_size as f64 * v.mlp_ratio).round() as usize,
                        patch: v.native_patch_size,
                        temporal_patch: v.temporal_patch_size,
                        merge: v.adapter_merge_size,
                        projection_hidden: v.projection_hidden_dim,
                        grid: v.posemb_grid_size,
                        layer_norm_eps: v.layer_norm_eps,
                    })
                })
                .transpose()?,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        let positive = [
            ("hidden_size", self.hidden),
            ("num_hidden_layers", self.layers),
            ("unpadded_vocab_size", self.unpadded_vocab),
            ("sconv_kernel_size", self.conv_kernel),
            ("n_layers_per_attn_res_block", self.block_layers),
            ("num_experts_per_tok", self.experts_per_token),
            ("intermediate_size", self.expert_width),
            ("d_rel", self.d_rel),
            ("loop num_passes", self.loop_passes),
            ("loop gate_rank", self.loop_gate_rank),
        ];
        if let Some((name, _)) = positive.iter().find(|(_, value)| *value == 0) {
            return Err(config_error(format!("{name} is zero")));
        }
        if self.unpadded_vocab > self.vocab {
            return Err(config_error("unpadded_vocab_size exceeds vocab_size"));
        }
        if self.experts_per_token > self.experts {
            return Err(config_error("num_experts_per_tok exceeds n_routed_experts"));
        }
        for (name, heads) in [("full", self.full), ("sliding", self.sliding)] {
            if heads.heads == 0
                || heads.kv_heads == 0
                || heads.head_dim == 0
                || heads.rel_extent == 0
                || !heads.heads.is_multiple_of(heads.kv_heads)
            {
                return Err(config_error(format!("bad {name} attention geometry")));
            }
        }
        // The loop is a whole number of depth-residual blocks and does not
        // start at layer 0 (the reference checks the same).
        if self.loop_start == 0
            || self.loop_start >= self.loop_end
            || self.loop_end > self.layers
            || !self.loop_start.is_multiple_of(self.block_layers)
            || !self.loop_end.is_multiple_of(self.block_layers)
        {
            return Err(config_error(format!(
                "loop {}..{} does not fit {} layers in blocks of {}",
                self.loop_start, self.loop_end, self.layers, self.block_layers
            )));
        }
        if (self.eos_token_id as usize) >= self.unpadded_vocab {
            return Err(config_error("eos_token_id is outside the vocabulary"));
        }
        Ok(())
    }

    pub fn heads(&self, layer: usize) -> Inkling2Heads {
        match self.layer_kinds[layer] {
            Inkling2LayerKind::Sliding => self.sliding,
            Inkling2LayerKind::Full => self.full,
        }
    }

    pub fn is_loop_layer(&self, layer: usize) -> bool {
        (self.loop_start..self.loop_end).contains(&layer)
    }

    pub fn loop_layers(&self) -> usize {
        self.loop_end - self.loop_start
    }

    /// Whether `layer` opens a depth-residual block.
    pub fn is_block_boundary(&self, layer: usize) -> bool {
        layer.is_multiple_of(self.block_layers)
    }

    /// Layer applications of one token, in execution order.
    pub fn applications(&self) -> Vec<Inkling2Application> {
        let mut out = Vec::new();
        let plain = |layer| Inkling2Application {
            layer,
            pass: 0,
            loop_index: None,
        };
        out.extend((0..self.loop_start).map(plain));
        for pass in 0..self.loop_passes {
            out.extend(
                (self.loop_start..self.loop_end).map(|layer| Inkling2Application {
                    layer,
                    pass,
                    loop_index: Some(layer - self.loop_start),
                }),
            );
        }
        out.extend((self.loop_end..self.layers).map(plain));
        out
    }

    pub fn application_count(&self) -> usize {
        self.layers + (self.loop_passes - 1) * self.loop_layers()
    }

    /// Rows of an expert's first projection: gate and up rows, interleaved.
    pub fn gate_up_rows(&self) -> usize {
        2 * self.expert_width
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn fixture(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    pub(crate) fn published() -> Inkling2Config {
        Inkling2Config::load(&fixture("inkling2")).unwrap()
    }

    pub(crate) fn tiny() -> Inkling2Config {
        Inkling2Config::load(&fixture("inkling2_golden")).unwrap()
    }

    #[test]
    fn the_published_config_parses_to_the_documented_geometry() {
        let c = published();
        assert_eq!((c.hidden, c.layers), (2304, 24));
        assert_eq!((c.vocab, c.unpadded_vocab), (201_024, 200_058));
        assert_eq!(c.full.heads, 24);
        assert_eq!(c.full.kv_heads, 8);
        assert_eq!(c.full.head_dim, 128);
        assert_eq!(c.full.window, None);
        assert_eq!(c.sliding.window, Some(1024));
        assert_eq!(c.sliding.rel_extent, 1024);
        let full = c
            .layer_kinds
            .iter()
            .enumerate()
            .filter(|(_, kind)| **kind == Inkling2LayerKind::Full)
            .map(|(layer, _)| layer)
            .collect::<Vec<_>>();
        assert_eq!(full, vec![5, 11, 17, 23]);
        assert_eq!(
            (c.experts, c.experts_per_token, c.expert_width),
            (128, 8, 1152)
        );
        assert_eq!((c.loop_start, c.loop_end, c.loop_passes), (6, 18, 3));
        assert_eq!(c.application_count(), 48);
        assert_eq!(c.applications().len(), 48);
        assert_eq!(c.eos_token_id, 200_006);
        assert_eq!(c.image_token_id, 200_054);
        assert_eq!(c.max_context_tokens, 1_048_576);
        let vision = c.vision.unwrap();
        assert_eq!((vision.hidden, vision.layers, vision.heads), (1024, 23, 16));
        assert_eq!(vision.cell(), 40);
        assert_eq!(vision.mlp_width, 4096);
    }

    #[test]
    fn applications_repeat_the_loop_layers_once_per_pass() {
        let c = tiny();
        let layers = c
            .applications()
            .iter()
            .map(|a| (a.layer, a.pass))
            .collect::<Vec<_>>();
        let mut want = vec![(0, 0), (1, 0)];
        for pass in 0..3 {
            want.extend((2..6).map(|layer| (layer, pass)));
        }
        want.extend([(6, 0), (7, 0)]);
        assert_eq!(layers, want);
        assert_eq!(c.application_count(), 16);
        assert_eq!(c.applications()[6].loop_index, Some(0));
        assert_eq!(c.applications()[6].pass, 1);
        assert_eq!(c.applications()[14].loop_index, None);
    }

    fn with_text(edit: impl Fn(&mut serde_json::Value)) -> Result<Inkling2Config> {
        let mut value: serde_json::Value =
            serde_json::from_slice(&std::fs::read(fixture("inkling2").join(CONFIG_FILE)).unwrap())
                .unwrap();
        edit(&mut value["text_config"]);
        Inkling2Config::from_json(&serde_json::to_vec(&value).unwrap())
    }

    #[test]
    fn settings_the_trunk_does_not_implement_fail_the_load() {
        assert!(with_text(|_| {}).is_ok());
        for (key, value) in [
            ("n_shared_experts", serde_json::json!(1)),
            ("use_value_norm_no_gamma", serde_json::json!(false)),
            ("attn_res_normalize_values", serde_json::json!(false)),
            ("use_gate_bias", serde_json::json!(false)),
            ("q_bias", serde_json::json!(true)),
            ("o_bias", serde_json::json!(true)),
        ] {
            let result = with_text(|text| text[key] = value.clone());
            assert!(result.is_err(), "{key} was accepted");
        }
        assert!(
            with_text(|text| text["loop_config"]["is_varloop"] = serde_json::json!(true)).is_err()
        );
        // A loop that is not a whole number of blocks.
        assert!(
            with_text(|text| text["loop_config"]["core_start"] = serde_json::json!(5)).is_err()
        );
        assert!(
            with_text(|text| text["loop_config"]["core_start"] = serde_json::json!(0)).is_err()
        );
    }
}
