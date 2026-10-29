//! Config parsing for Nemotron 3.5 Transcribe.
//!
//! The checkpoint is a NeMo SpeechLM bundle (`model_type: nemo_speechlm`):
//! `config.json` carries the perception front end, the Parallel Expert
//! Encoder (an ASR TransformerEncoder and the Nemotron-3-Diarization
//! Sortformer), the Nemotron-H decoder (`llm_config`) and the ModelOpt FP8
//! quantization; `hf_quant_config.json` repeats the quantization and
//! `generation_config.json` the stop token.
//!
//! Every switch the family implements one way is checked here and fails
//! closed: a checkpoint that changed an activation, a norm, the routing or
//! the quantization would otherwise load and produce plausible garbage.
//! Dimensions are read, checked for internal consistency, and then pinned
//! tensor by tensor by the checkpoint census (`checkpoint.rs`).

use std::fs;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use crate::error::{Error, Result};

pub const MODEL_TYPE: &str = "nemo_speechlm";
pub const ARCHITECTURE: &str = "Nemotron3_5TranscribeForConditionalGeneration";
pub const CONFIG_FILE: &str = "config.json";
pub const QUANT_CONFIG_FILE: &str = "hf_quant_config.json";
pub const GENERATION_CONFIG_FILE: &str = "generation_config.json";
pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const TOKENIZER_CONFIG_FILE: &str = "tokenizer_config.json";

/// What the serving path runs, which the config does not say: NeMo's online
/// Parallel Expert Encoder windows (`parallel_expert_encoder.py:395-400`)
/// and its streaming Sortformer overrides (`:2167-2170`). The config's
/// `chunk_size_seconds: 30` is the training path.
pub const ONLINE_WINDOW_TOKENS: usize = 500;
pub const ONLINE_CONTEXT_TOKENS: usize = 50;
pub const SORTFORMER_CHUNK_LEN: usize = 500;
pub const SORTFORMER_FIFO_LEN: usize = 40;
pub const SORTFORMER_UPDATE_PERIOD: usize = 300;
pub const SORTFORMER_CACHE_LEN: usize = 188;

/// The front end (NeMo `AudioToMelSpectrogramPreprocessor` defaults that
/// apply to this checkpoint).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NemotronTranscribeFrontendConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub preemphasis: f32,
    pub log_guard: f32,
}

/// One NeMo `TransformerEncoder` (feature stacking, pre-norm blocks).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeechTransformerConfig {
    pub feat_in: usize,
    pub d_model: usize,
    pub heads: usize,
    pub layers: usize,
    pub ff_dim: usize,
    pub qk_norm: bool,
    /// Rotated dims per head (the first `rotary_dims` of `head_dim`).
    pub rotary_dims: usize,
    pub rope_base: f64,
    /// Mel frames stacked into one encoder frame (1 for the CTC trunk, which
    /// bypasses pre-encoding).
    pub stacking: usize,
    pub layer_norm_eps: f64,
}

impl SpeechTransformerConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.heads
    }
}

/// The Sortformer head and its streaming settings (serving overrides).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SortformerConfig {
    pub speakers: usize,
    pub fc_d_model: usize,
    pub tf_d_model: usize,
    pub upsample: usize,
    pub chunk_len: usize,
    pub fifo_len: usize,
    pub update_period: usize,
    pub cache_len: usize,
    pub sil_frames_per_speaker: usize,
    pub pred_score_threshold: f32,
    pub scores_boost_latest: f32,
    pub strong_boost_rate: f32,
    pub weak_boost_rate: f32,
    pub min_pos_scores_rate: f32,
}

/// What one decoder layer mixes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemotronLayerKind {
    Mamba,
    Moe,
    Attention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MambaConfig {
    pub heads: usize,
    pub head_dim: usize,
    pub groups: usize,
    pub state: usize,
    pub conv_kernel: usize,
    pub chunk: usize,
}

impl MambaConfig {
    pub fn inner(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn conv_channels(&self) -> usize {
        self.inner() + 2 * self.groups * self.state
    }

    /// `in_proj` rows: `z | xBC | dt`.
    pub fn in_proj_rows(&self) -> usize {
        self.inner() + self.conv_channels() + self.heads
    }

    /// The gated norm's group width.
    pub fn norm_group(&self) -> usize {
        self.inner() / self.groups
    }

    pub fn geometry(&self) -> crate::kernels::mamba2::Mamba2Geometry {
        crate::kernels::mamba2::Mamba2Geometry {
            heads: self.heads,
            head_dim: self.head_dim,
            groups: self.groups,
            state: self.state,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoeConfig {
    pub experts: usize,
    pub top_k: usize,
    pub intermediate: usize,
    pub shared_intermediate: usize,
    pub routed_scaling: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionConfig {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

/// The Nemotron-H decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct NemotronDecoderConfig {
    pub hidden: usize,
    pub vocab: usize,
    pub layers: Vec<NemotronLayerKind>,
    pub mamba: MambaConfig,
    pub moe: MoeConfig,
    pub attention: AttentionConfig,
    pub norm_eps: f64,
}

impl NemotronDecoderConfig {
    /// Model-layer indices of each kind, in order.
    pub fn layers_of(&self, kind: NemotronLayerKind) -> Vec<usize> {
        self.layers
            .iter()
            .enumerate()
            .filter_map(|(index, layer)| (*layer == kind).then_some(index))
            .collect()
    }
}

/// Everything the family reads from the checkpoint's configs.
#[derive(Debug, Clone, PartialEq)]
pub struct NemotronTranscribeConfig {
    pub frontend: NemotronTranscribeFrontendConfig,
    pub asr_encoder: SpeechTransformerConfig,
    pub diar_encoder: SpeechTransformerConfig,
    pub sortformer: SortformerConfig,
    pub speaker_activity_threshold: f32,
    pub speaker_kernel_scale: f32,
    pub decoder: NemotronDecoderConfig,
    pub audio_token: String,
    pub eos_token_ids: Vec<u32>,
    /// The FP8 exclude list (`hf_quant_config.json`), as published.
    pub fp8_excluded: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    model_type: String,
    #[serde(default)]
    architectures: Vec<String>,
    audio_locator_tag: String,
    prompt_format: String,
    perception: RawPerception,
    pe_encoder_config: RawPeEncoder,
    llm_config: RawLlm,
    quantization_config: RawQuantization,
    #[serde(default)]
    eos_token_id: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct RawPerception {
    output_dim: usize,
    preprocessor: RawPreprocessor,
    modality_adapter: RawAdapter,
}

#[derive(Debug, Deserialize)]
struct RawPreprocessor {
    sample_rate: usize,
    #[serde(default)]
    normalize: Option<String>,
    window_size: f64,
    window_stride: f64,
    window: String,
    features: usize,
    n_fft: usize,
    #[serde(default = "default_true")]
    log: bool,
    #[serde(default = "default_one")]
    frame_splicing: usize,
    #[serde(default)]
    pad_to: usize,
    #[serde(default)]
    pad_value: f64,
}

#[derive(Debug, Deserialize)]
struct RawAdapter {
    #[serde(rename = "_target_")]
    target: String,
    d_model: usize,
}

#[derive(Debug, Deserialize)]
struct RawPeEncoder {
    asr_encoder_type: String,
    asr_encoder_cfg: RawTransformer,
    diarization_model_cfg: RawDiarization,
    asr_normalize_type: String,
    diar_normalize_type: String,
    speaker_feature_mode: String,
    speaker_activity_threshold: f32,
    spk_kernel_scale: f32,
}

#[derive(Debug, Deserialize)]
struct RawTransformer {
    feat_in: usize,
    d_model: usize,
    n_heads: usize,
    n_layers: usize,
    #[serde(default)]
    qkv_bias: bool,
    #[serde(default)]
    qk_norm: bool,
    ff_expansion: f64,
    #[serde(default = "default_true")]
    pre_block_norm: bool,
    subsampling_factor: usize,
    #[serde(default = "default_full")]
    attn_mode: String,
    self_attention_model: String,
    #[serde(default = "default_rope_base")]
    rope_base: f64,
    #[serde(default = "default_rotary_fraction")]
    rotary_fraction: f64,
    #[serde(default)]
    xscaling: bool,
    #[serde(default = "default_subsampling")]
    subsampling: String,
}

#[derive(Debug, Deserialize)]
struct RawDiarization {
    max_num_of_spks: usize,
    high_resolution: bool,
    sortformer_modules: RawSortformerModules,
    encoder: RawTransformer,
}

#[derive(Debug, Deserialize)]
struct RawSortformerModules {
    num_spks: usize,
    fc_d_model: usize,
    tf_d_model: usize,
    spkcache_sil_frames_per_spk: usize,
    pred_score_threshold: f32,
    scores_boost_latest: f32,
    strong_boost_rate: f32,
    weak_boost_rate: f32,
    min_pos_scores_rate: f32,
    #[serde(default)]
    scores_add_rnd: f64,
    use_learnable_sil_emb: bool,
}

#[derive(Debug, Deserialize)]
struct RawLlm {
    model_type: String,
    hidden_size: usize,
    vocab_size: usize,
    layers_block_type: Vec<String>,
    mamba_num_heads: usize,
    mamba_head_dim: usize,
    n_groups: usize,
    ssm_state_size: usize,
    conv_kernel: usize,
    chunk_size: usize,
    mamba_hidden_act: String,
    use_conv_bias: bool,
    #[serde(default)]
    mamba_proj_bias: bool,
    mamba_ssm_cache_dtype: String,
    time_step_limit: Value,
    n_routed_experts: usize,
    num_experts_per_tok: usize,
    moe_intermediate_size: usize,
    moe_shared_expert_intermediate_size: usize,
    n_shared_experts: usize,
    routed_scaling_factor: f32,
    norm_topk_prob: bool,
    n_group: usize,
    topk_group: usize,
    mlp_hidden_act: String,
    #[serde(default)]
    mlp_bias: bool,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    #[serde(default)]
    attention_bias: bool,
    layer_norm_epsilon: f64,
    tie_word_embeddings: bool,
    #[serde(default)]
    moe_latent_size: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RawQuantization {
    quant_algo: String,
    quant_method: String,
    #[serde(default)]
    ignore: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawQuantFile {
    quantization: RawQuantFileBody,
}

#[derive(Debug, Deserialize)]
struct RawQuantFileBody {
    quant_algo: String,
    #[serde(default)]
    kv_cache_quant_algo: Option<String>,
    #[serde(default)]
    exclude_modules: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawGeneration {
    #[serde(default)]
    eos_token_id: Value,
}

fn default_true() -> bool {
    true
}

fn default_one() -> usize {
    1
}

fn default_full() -> String {
    "full".into()
}

fn default_rope_base() -> f64 {
    10_000.0
}

fn default_rotary_fraction() -> f64 {
    1.0
}

fn default_subsampling() -> String {
    "feature_stacking".into()
}

fn invalid(message: impl Into<String>) -> Error {
    Error::ModelLoadError(format!(
        "Nemotron 3.5 Transcribe config: {}",
        message.into()
    ))
}

fn require<T: PartialEq + std::fmt::Debug>(field: &str, found: T, expected: T) -> Result<()> {
    if found == expected {
        Ok(())
    } else {
        Err(invalid(format!(
            "{field} is {found:?}; this family implements {expected:?}"
        )))
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let raw =
        fs::read(path).map_err(|err| invalid(format!("cannot read {}: {err}", path.display())))?;
    serde_json::from_slice(&raw)
        .map_err(|err| invalid(format!("cannot parse {}: {err}", path.display())))
}

impl NemotronTranscribeConfig {
    /// Reads and validates `config.json`, `hf_quant_config.json` and
    /// `generation_config.json` from `model_dir`.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let raw: RawConfig = read_json(&model_dir.join(CONFIG_FILE))?;
        let quant: RawQuantFile = read_json(&model_dir.join(QUANT_CONFIG_FILE))?;
        let generation: RawGeneration = read_json(&model_dir.join(GENERATION_CONFIG_FILE))?;
        Self::from_raw(raw, quant, generation)
    }

    /// Parses the three configs from JSON text (fixtures and tests).
    pub fn from_json(config: &str, quant: &str, generation: &str) -> Result<Self> {
        let parse = |what: &str, text: &str| -> Result<Value> {
            serde_json::from_str(text).map_err(|err| invalid(format!("{what}: {err}")))
        };
        let raw = serde_json::from_value(parse(CONFIG_FILE, config)?)
            .map_err(|err| invalid(format!("{CONFIG_FILE}: {err}")))?;
        let quant = serde_json::from_value(parse(QUANT_CONFIG_FILE, quant)?)
            .map_err(|err| invalid(format!("{QUANT_CONFIG_FILE}: {err}")))?;
        let generation = serde_json::from_value(parse(GENERATION_CONFIG_FILE, generation)?)
            .map_err(|err| invalid(format!("{GENERATION_CONFIG_FILE}: {err}")))?;
        Self::from_raw(raw, quant, generation)
    }

    fn from_raw(raw: RawConfig, quant: RawQuantFile, generation: RawGeneration) -> Result<Self> {
        require("model_type", raw.model_type.as_str(), MODEL_TYPE)?;
        if !raw.architectures.iter().any(|a| a == ARCHITECTURE) {
            return Err(invalid(format!(
                "architectures {:?} do not name {ARCHITECTURE}",
                raw.architectures
            )));
        }
        require("prompt_format", raw.prompt_format.as_str(), "nemotron3p5")?;

        let frontend = frontend_config(&raw.perception.preprocessor)?;
        require(
            "perception.modality_adapter",
            raw.perception.modality_adapter.target.rsplit('.').next(),
            Some("IdentityConnector"),
        )?;

        let pe = &raw.pe_encoder_config;
        require(
            "asr_encoder_type",
            pe.asr_encoder_type.as_str(),
            "transformer",
        )?;
        require(
            "asr_normalize_type",
            pe.asr_normalize_type.as_str(),
            "per_feature",
        )?;
        require("diar_normalize_type", pe.diar_normalize_type.as_str(), "NA")?;
        require(
            "speaker_feature_mode",
            pe.speaker_feature_mode.as_str(),
            "thresholded",
        )?;
        let asr_encoder =
            transformer_config("asr_encoder_cfg", &pe.asr_encoder_cfg, frontend.n_mels)?;
        require("asr_encoder_cfg.qk_norm", asr_encoder.qk_norm, true)?;
        let diar = &pe.diarization_model_cfg;
        let diar_encoder = transformer_config(
            "diarization_model_cfg.encoder",
            &diar.encoder,
            frontend.n_mels,
        )?;
        require(
            "perception.modality_adapter.d_model",
            raw.perception.modality_adapter.d_model,
            asr_encoder.d_model,
        )?;
        let modules = &diar.sortformer_modules;
        require("sortformer high_resolution", diar.high_resolution, true)?;
        require(
            "sortformer use_learnable_sil_emb",
            modules.use_learnable_sil_emb,
            true,
        )?;
        require("sortformer scores_add_rnd", modules.scores_add_rnd, 0.0)?;
        if modules.num_spks != diar.max_num_of_spks
            || modules.fc_d_model != diar_encoder.d_model
            || modules.num_spks == 0
            || modules.tf_d_model == 0
        {
            return Err(invalid("inconsistent Sortformer speaker or width settings"));
        }
        let sortformer = SortformerConfig {
            speakers: modules.num_spks,
            fc_d_model: modules.fc_d_model,
            tf_d_model: modules.tf_d_model,
            upsample: diar_encoder.stacking,
            chunk_len: SORTFORMER_CHUNK_LEN,
            fifo_len: SORTFORMER_FIFO_LEN,
            update_period: SORTFORMER_UPDATE_PERIOD,
            cache_len: SORTFORMER_CACHE_LEN,
            sil_frames_per_speaker: modules.spkcache_sil_frames_per_spk,
            pred_score_threshold: modules.pred_score_threshold,
            scores_boost_latest: modules.scores_boost_latest,
            strong_boost_rate: modules.strong_boost_rate,
            weak_boost_rate: modules.weak_boost_rate,
            min_pos_scores_rate: modules.min_pos_scores_rate,
        };
        if sortformer.cache_len / sortformer.speakers <= sortformer.sil_frames_per_speaker {
            return Err(invalid("the Sortformer cache leaves no frames per speaker"));
        }
        if asr_encoder.stacking != diar_encoder.stacking {
            return Err(invalid(
                "the ASR encoder and the diarizer stack mel frames differently",
            ));
        }

        let decoder = decoder_config(&raw.llm_config)?;
        require(
            "perception.output_dim",
            raw.perception.output_dim,
            decoder.hidden,
        )?;

        require(
            "quantization_config.quant_method",
            raw.quantization_config.quant_method.as_str(),
            "modelopt",
        )?;
        require(
            "quantization_config.quant_algo",
            raw.quantization_config.quant_algo.as_str(),
            "FP8",
        )?;
        require(
            "hf_quant_config quant_algo",
            quant.quantization.quant_algo.as_str(),
            "FP8",
        )?;
        if let Some(kv) = quant.quantization.kv_cache_quant_algo.as_deref() {
            // Izwi stores dense 16-bit or F32 KV; the published FP8 KV scales
            // are validated and only the `vllm-fp8` parity mode uses them.
            require("hf_quant_config kv_cache_quant_algo", kv, "FP8")?;
        }
        let mut excluded = quant.quantization.exclude_modules.clone();
        let mut ignored = raw.quantization_config.ignore.clone();
        excluded.sort();
        ignored.sort();
        if excluded != ignored {
            return Err(invalid(
                "config.json and hf_quant_config.json disagree on the FP8 exclude list",
            ));
        }

        let eos_token_ids = match generation.eos_token_id {
            Value::Number(n) => vec![n.as_u64().ok_or_else(|| invalid("eos_token_id"))? as u32],
            Value::Array(values) => values
                .iter()
                .map(|v| {
                    v.as_u64()
                        .map(|v| v as u32)
                        .ok_or_else(|| invalid("eos_token_id"))
                })
                .collect::<Result<Vec<_>>>()?,
            Value::Null => raw.eos_token_id.clone(),
            _ => return Err(invalid("eos_token_id is neither a number nor a list")),
        };
        if eos_token_ids.is_empty() {
            return Err(invalid("no stop token"));
        }

        Ok(Self {
            frontend,
            asr_encoder,
            diar_encoder,
            sortformer,
            speaker_activity_threshold: pe.speaker_activity_threshold,
            speaker_kernel_scale: pe.spk_kernel_scale,
            decoder,
            audio_token: raw.audio_locator_tag,
            eos_token_ids,
            fp8_excluded: quant.quantization.exclude_modules,
        })
    }

    /// The perception output frames (= audio tokens) for `samples` of 16 kHz
    /// audio: `max(1, ceil((samples // hop) / stacking))`.
    pub fn audio_tokens_for_samples(&self, samples: usize) -> usize {
        (samples / self.frontend.hop_length)
            .div_ceil(self.asr_encoder.stacking)
            .max(1)
    }

    /// Whether ModelOpt left `module` (a weight's module path) unquantized.
    pub fn fp8_excludes(&self, module: &str) -> bool {
        self.fp8_excluded
            .iter()
            .any(|pattern| match pattern.strip_suffix('*') {
                Some(prefix) => module.starts_with(prefix),
                None => module == pattern,
            })
    }
}

fn frontend_config(raw: &RawPreprocessor) -> Result<NemotronTranscribeFrontendConfig> {
    require("preprocessor.window", raw.window.as_str(), "hann")?;
    require("preprocessor.log", raw.log, true)?;
    require("preprocessor.frame_splicing", raw.frame_splicing, 1)?;
    require("preprocessor.pad_to", raw.pad_to, 0)?;
    require("preprocessor.pad_value", raw.pad_value, 0.0)?;
    // The perception preprocessor must not normalize: the encoder normalizes
    // per feature itself (vLLM forces this to None as well).
    if raw
        .normalize
        .as_deref()
        .is_some_and(|value| !value.is_empty() && value != "None")
    {
        return Err(invalid(format!(
            "preprocessor.normalize is {:?}; the encoder normalizes",
            raw.normalize
        )));
    }
    let win_length = (raw.window_size * raw.sample_rate as f64).round() as usize;
    let hop_length = (raw.window_stride * raw.sample_rate as f64).round() as usize;
    if raw.sample_rate != 16_000 || win_length == 0 || hop_length == 0 || win_length > raw.n_fft {
        return Err(invalid("unsupported front-end framing"));
    }
    Ok(NemotronTranscribeFrontendConfig {
        sample_rate: raw.sample_rate,
        n_fft: raw.n_fft,
        win_length,
        hop_length,
        n_mels: raw.features,
        preemphasis: 0.97,
        log_guard: 2f32.powi(-24),
    })
}

fn transformer_config(
    field: &str,
    raw: &RawTransformer,
    n_mels: usize,
) -> Result<SpeechTransformerConfig> {
    require(
        &format!("{field}.self_attention_model"),
        raw.self_attention_model.as_str(),
        "rope",
    )?;
    require(
        &format!("{field}.attn_mode"),
        raw.attn_mode.as_str(),
        "full",
    )?;
    require(&format!("{field}.pre_block_norm"), raw.pre_block_norm, true)?;
    require(&format!("{field}.qkv_bias"), raw.qkv_bias, false)?;
    require(&format!("{field}.xscaling"), raw.xscaling, false)?;
    require(
        &format!("{field}.subsampling"),
        raw.subsampling.as_str(),
        "feature_stacking",
    )?;
    require(&format!("{field}.feat_in"), raw.feat_in, n_mels)?;
    if raw.n_heads == 0 || !raw.d_model.is_multiple_of(raw.n_heads) || raw.n_layers == 0 {
        return Err(invalid(format!("{field} has an invalid head layout")));
    }
    let head_dim = raw.d_model / raw.n_heads;
    let rotary = head_dim as f64 * raw.rotary_fraction;
    let rotary_dims = rotary.round() as usize;
    if (rotary - rotary_dims as f64).abs() > 1e-9
        || rotary_dims == 0
        || !rotary_dims.is_multiple_of(2)
        || rotary_dims > head_dim
    {
        return Err(invalid(format!(
            "{field} has an unsupported rotary fraction"
        )));
    }
    let ff = raw.d_model as f64 * raw.ff_expansion;
    if (ff - ff.round()).abs() > 1e-9 || ff < 1.0 {
        return Err(invalid(format!("{field} has a fractional FFN width")));
    }
    Ok(SpeechTransformerConfig {
        feat_in: raw.feat_in,
        d_model: raw.d_model,
        heads: raw.n_heads,
        layers: raw.n_layers,
        ff_dim: ff.round() as usize,
        qk_norm: raw.qk_norm,
        rotary_dims,
        rope_base: raw.rope_base,
        stacking: raw.subsampling_factor,
        layer_norm_eps: 1e-5,
    })
}

fn decoder_config(raw: &RawLlm) -> Result<NemotronDecoderConfig> {
    require(
        "llm_config.model_type",
        raw.model_type.as_str(),
        "nemotron_h",
    )?;
    require(
        "llm_config.mamba_hidden_act",
        raw.mamba_hidden_act.as_str(),
        "silu",
    )?;
    require("llm_config.use_conv_bias", raw.use_conv_bias, true)?;
    require("llm_config.mamba_proj_bias", raw.mamba_proj_bias, false)?;
    require(
        "llm_config.mamba_ssm_cache_dtype",
        raw.mamba_ssm_cache_dtype.as_str(),
        "float32",
    )?;
    require(
        "llm_config.mlp_hidden_act",
        raw.mlp_hidden_act.as_str(),
        "relu2",
    )?;
    require("llm_config.mlp_bias", raw.mlp_bias, false)?;
    require("llm_config.attention_bias", raw.attention_bias, false)?;
    require("llm_config.norm_topk_prob", raw.norm_topk_prob, true)?;
    require("llm_config.n_group", raw.n_group, 1)?;
    require("llm_config.topk_group", raw.topk_group, 1)?;
    require("llm_config.n_shared_experts", raw.n_shared_experts, 1)?;
    require(
        "llm_config.tie_word_embeddings",
        raw.tie_word_embeddings,
        false,
    )?;
    require("llm_config.moe_latent_size", raw.moe_latent_size, None)?;
    // No dt clamp: time_step_limit must be [0, inf).
    let limit_ok = match &raw.time_step_limit {
        Value::Array(values) if values.len() == 2 => {
            values[0].as_f64() == Some(0.0)
                && match &values[1] {
                    Value::Object(map) => {
                        map.get("__float__").and_then(Value::as_str) == Some("Infinity")
                    }
                    Value::String(text) => text.eq_ignore_ascii_case("infinity") || text == "inf",
                    Value::Number(n) => n.as_f64().is_some_and(f64::is_infinite),
                    _ => false,
                }
        }
        _ => false,
    };
    if !limit_ok {
        return Err(invalid(format!(
            "llm_config.time_step_limit is {}; this family runs dt unclamped",
            raw.time_step_limit
        )));
    }
    let layers = raw
        .layers_block_type
        .iter()
        .map(|kind| match kind.as_str() {
            "mamba" => Ok(NemotronLayerKind::Mamba),
            "moe" => Ok(NemotronLayerKind::Moe),
            "attention" => Ok(NemotronLayerKind::Attention),
            other => Err(invalid(format!("unsupported decoder layer kind {other:?}"))),
        })
        .collect::<Result<Vec<_>>>()?;
    let mamba = MambaConfig {
        heads: raw.mamba_num_heads,
        head_dim: raw.mamba_head_dim,
        groups: raw.n_groups,
        state: raw.ssm_state_size,
        conv_kernel: raw.conv_kernel,
        chunk: raw.chunk_size,
    };
    mamba
        .geometry()
        .validate()
        .map_err(|err| invalid(format!("Mamba2 geometry: {err}")))?;
    if mamba.conv_kernel < 2 || !mamba.inner().is_multiple_of(mamba.groups) {
        return Err(invalid("unsupported Mamba2 convolution or norm grouping"));
    }
    let moe = MoeConfig {
        experts: raw.n_routed_experts,
        top_k: raw.num_experts_per_tok,
        intermediate: raw.moe_intermediate_size,
        shared_intermediate: raw.moe_shared_expert_intermediate_size,
        routed_scaling: raw.routed_scaling_factor,
    };
    if moe.experts == 0 || moe.top_k == 0 || moe.top_k > moe.experts || moe.intermediate == 0 {
        return Err(invalid("invalid MoE routing geometry"));
    }
    let attention = AttentionConfig {
        heads: raw.num_attention_heads,
        kv_heads: raw.num_key_value_heads,
        head_dim: raw.head_dim,
    };
    if attention.kv_heads == 0 || !attention.heads.is_multiple_of(attention.kv_heads) {
        return Err(invalid("invalid attention head grouping"));
    }
    let config = NemotronDecoderConfig {
        hidden: raw.hidden_size,
        vocab: raw.vocab_size,
        layers,
        mamba,
        moe,
        attention,
        norm_eps: raw.layer_norm_epsilon,
    };
    for kind in [
        NemotronLayerKind::Mamba,
        NemotronLayerKind::Moe,
        NemotronLayerKind::Attention,
    ] {
        if config.layers_of(kind).is_empty() {
            return Err(invalid(format!("the decoder has no {kind:?} layer")));
        }
    }
    Ok(config)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) const PUBLISHED_CONFIG: &str =
        include_str!("../../../../tests/fixtures/nemotron_transcribe/config.json");
    pub(crate) const PUBLISHED_QUANT: &str =
        include_str!("../../../../tests/fixtures/nemotron_transcribe/hf_quant_config.json");
    pub(crate) const PUBLISHED_GENERATION: &str =
        include_str!("../../../../tests/fixtures/nemotron_transcribe/generation_config.json");

    pub(crate) fn published() -> NemotronTranscribeConfig {
        NemotronTranscribeConfig::from_json(PUBLISHED_CONFIG, PUBLISHED_QUANT, PUBLISHED_GENERATION)
            .unwrap()
    }

    #[test]
    fn the_published_config_parses_to_the_census_geometry() {
        let c = published();
        assert_eq!(c.frontend.n_fft, 512);
        assert_eq!(c.frontend.win_length, 400);
        assert_eq!(c.frontend.hop_length, 160);
        assert_eq!(c.frontend.n_mels, 128);
        assert_eq!(
            (
                c.asr_encoder.d_model,
                c.asr_encoder.heads,
                c.asr_encoder.layers,
                c.asr_encoder.ff_dim
            ),
            (1280, 10, 32, 5120)
        );
        assert_eq!(c.asr_encoder.rotary_dims, 64);
        assert!(c.asr_encoder.qk_norm);
        assert_eq!(
            (
                c.diar_encoder.d_model,
                c.diar_encoder.heads,
                c.diar_encoder.layers,
                c.diar_encoder.ff_dim
            ),
            (512, 8, 31, 2048)
        );
        assert_eq!(c.diar_encoder.rotary_dims, 64);
        assert!(!c.diar_encoder.qk_norm);
        assert_eq!(
            (
                c.sortformer.speakers,
                c.sortformer.tf_d_model,
                c.sortformer.upsample
            ),
            (8, 192, 8)
        );
        assert_eq!(
            (
                c.sortformer.chunk_len,
                c.sortformer.fifo_len,
                c.sortformer.update_period,
                c.sortformer.cache_len
            ),
            (500, 40, 300, 188)
        );
        let d = &c.decoder;
        assert_eq!((d.hidden, d.vocab, d.layers.len()), (2688, 131_072, 52));
        assert_eq!(
            d.layers_of(NemotronLayerKind::Attention),
            vec![5, 12, 19, 26, 33, 42]
        );
        assert_eq!(d.layers_of(NemotronLayerKind::Mamba).len(), 23);
        assert_eq!(d.layers_of(NemotronLayerKind::Moe).len(), 23);
        assert_eq!(d.mamba.in_proj_rows(), 10_304);
        assert_eq!(d.mamba.conv_channels(), 6144);
        assert_eq!(d.mamba.norm_group(), 512);
        assert_eq!(
            (
                d.moe.experts,
                d.moe.top_k,
                d.moe.intermediate,
                d.moe.shared_intermediate
            ),
            (128, 6, 1856, 3712)
        );
        assert_eq!(d.moe.routed_scaling, 2.5);
        assert_eq!(
            (
                d.attention.heads,
                d.attention.kv_heads,
                d.attention.head_dim
            ),
            (32, 2, 128)
        );
        assert_eq!(c.audio_token, "<|audio|>");
        assert_eq!(c.eos_token_ids, vec![11]);
        assert!(c.fp8_excludes("lm_head"));
        assert!(c.fp8_excludes("perception.encoder.diarization_model.encoder.layers.0.attn.w_qkv"));
        assert!(c.fp8_excludes("backbone.layers.1.mixer.gate"));
        assert!(!c.fp8_excludes("backbone.layers.1.mixer.experts.0.up_proj"));
    }

    #[test]
    fn audio_token_counts_follow_the_estimator() {
        let c = published();
        assert_eq!(c.audio_tokens_for_samples(30 * 16_000), 375);
        assert_eq!(c.audio_tokens_for_samples(3600 * 16_000), 45_000);
        assert_eq!(c.audio_tokens_for_samples(0), 1);
        assert_eq!(c.audio_tokens_for_samples(161), 1);
        assert_eq!(c.audio_tokens_for_samples(160 * 9), 2);
    }

    fn mutated(edit: impl Fn(&mut Value)) -> Result<NemotronTranscribeConfig> {
        let mut config: Value = serde_json::from_str(PUBLISHED_CONFIG).unwrap();
        edit(&mut config);
        NemotronTranscribeConfig::from_json(
            &config.to_string(),
            PUBLISHED_QUANT,
            PUBLISHED_GENERATION,
        )
    }

    #[test]
    fn changed_conventions_fail_closed() {
        assert!(mutated(|c| c["llm_config"]["mlp_hidden_act"] = "silu".into()).is_err());
        assert!(
            mutated(|c| c["llm_config"]["time_step_limit"] = serde_json::json!([0.001, 0.1]))
                .is_err()
        );
        assert!(mutated(|c| c["llm_config"]["norm_topk_prob"] = false.into()).is_err());
        assert!(mutated(|c| c["llm_config"]["layers_block_type"][3] = "mlp".into()).is_err());
        assert!(
            mutated(|c| c["pe_encoder_config"]["asr_normalize_type"] = "all_features".into())
                .is_err()
        );
        assert!(mutated(
            |c| c["pe_encoder_config"]["asr_encoder_cfg"]["rotary_fraction"] = 1.0.into()
        )
        .is_ok());
        assert!(mutated(
            |c| c["pe_encoder_config"]["asr_encoder_cfg"]["self_attention_model"] =
                "abs_pos".into()
        )
        .is_err());
        assert!(mutated(|c| c["perception"]["preprocessor"]["window"] = "hamming".into()).is_err());
        assert!(mutated(|c| c["quantization_config"]["quant_algo"] = "NVFP4".into()).is_err());
        assert!(mutated(|c| c["model_type"] = "salm".into()).is_err());
        assert!(mutated(|_| {}).is_ok());
    }
}
