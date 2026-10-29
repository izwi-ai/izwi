//! A tiny checkpoint in the published layout, for tests: every real tensor
//! name and dtype (FP8 linears with ModelOpt scalar scales, BF16 elsewhere,
//! the F32 router bias near +40.7), the real front-end tables, configs that
//! pass the same validation, and a word-level tokenizer whose special ids
//! sit where the real ones do (`<|audio|>` one past the 256-row table).
//!
//! Geometry: hidden 64; layers `M E * M E`; Mamba 4 heads × 16, 2 groups,
//! state 8; 8 experts (top-2, width 96, so CUDA pads it) and a shared
//! expert; attention 4/2 heads × 16; encoders of width 32 (2 heads, 2
//! layers); 8 speakers.

use std::collections::BTreeMap;
use std::path::Path;

use half::bf16;
use safetensors::tensor::{Dtype, TensorView};
use serde_json::json;

use super::checkpoint::{CTC_FILE, INDEX_FILE, PERCEPTION_FILE};
use crate::kernels::cuda::fp8::encode_e4m3fn;
use crate::kernels::numeric_check::xorshift_values;

pub const HIDDEN: usize = 64;
pub const VOCAB: usize = 256;
pub const LAYERS: [&str; 5] = ["mamba", "moe", "attention", "mamba", "moe"];
pub const ENCODER_WIDTH: usize = 32;
pub const EXPERTS: usize = 8;
pub const EXPERT_WIDTH: usize = 96;
pub const SHARED_WIDTH: usize = 64;
pub const SPEAKERS: usize = 8;
pub const TF_WIDTH: usize = 16;

/// Tensors to write, by file.
struct Writer {
    files: BTreeMap<&'static str, Vec<(String, Dtype, Vec<usize>, Vec<u8>)>>,
    seed: u64,
}

impl Writer {
    fn values(&mut self, len: usize, scale: f32) -> Vec<f32> {
        self.seed += 1;
        xorshift_values(len, self.seed, scale)
    }

    fn push(
        &mut self,
        file: &'static str,
        name: impl Into<String>,
        dtype: Dtype,
        shape: &[usize],
        bytes: Vec<u8>,
    ) {
        self.files
            .entry(file)
            .or_default()
            .push((name.into(), dtype, shape.to_vec(), bytes));
    }

    fn bf16(
        &mut self,
        file: &'static str,
        name: impl Into<String>,
        shape: &[usize],
        values: &[f32],
    ) {
        let bytes = values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
            .collect();
        self.push(file, name, Dtype::BF16, shape, bytes);
    }

    fn f32(
        &mut self,
        file: &'static str,
        name: impl Into<String>,
        shape: &[usize],
        values: &[f32],
    ) {
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.push(file, name, Dtype::F32, shape, bytes);
    }

    fn random_bf16(
        &mut self,
        file: &'static str,
        name: impl Into<String>,
        shape: &[usize],
        scale: f32,
    ) {
        let values = self.values(shape.iter().product(), scale);
        self.bf16(file, name, shape, &values);
    }

    fn norm(&mut self, file: &'static str, name: &str, width: usize, bias: bool) {
        let gains = self
            .values(width, 0.2)
            .into_iter()
            .map(|v| 1.0 + v)
            .collect::<Vec<_>>();
        self.bf16(file, format!("{name}.weight"), &[width], &gains);
        if bias {
            self.random_bf16(file, format!("{name}.bias"), &[width], 0.05);
        }
    }

    /// A ModelOpt FP8 linear: E4M3 bytes with a per-tensor scale (amax/448)
    /// and a static input scale.
    fn fp8(&mut self, file: &'static str, module: &str, rows: usize, cols: usize, bias: bool) {
        let scale = (cols as f32).sqrt().recip();
        let values = self.values(rows * cols, scale);
        let amax = values.iter().fold(0f32, |m, v| m.max(v.abs()));
        let weight_scale = amax / 448.0;
        let bytes = values
            .iter()
            .map(|v| encode_e4m3fn(v / weight_scale))
            .collect();
        self.push(
            file,
            format!("{module}.weight"),
            Dtype::F8_E4M3,
            &[rows, cols],
            bytes,
        );
        self.f32(file, format!("{module}.weight_scale"), &[], &[weight_scale]);
        self.f32(file, format!("{module}.input_scale"), &[], &[0.02]);
        if bias {
            self.random_bf16(file, format!("{module}.bias"), &[rows], 0.02);
        }
    }

    fn dense(&mut self, file: &'static str, module: &str, rows: usize, cols: usize, bias: bool) {
        self.random_bf16(
            file,
            format!("{module}.weight"),
            &[rows, cols],
            (cols as f32).sqrt().recip(),
        );
        if bias {
            self.random_bf16(file, format!("{module}.bias"), &[rows], 0.02);
        }
    }

    fn transformer(
        &mut self,
        file: &'static str,
        prefix: &str,
        layers: usize,
        ff: usize,
        fp8: bool,
        qk_norm: bool,
    ) {
        let d = ENCODER_WIDTH;
        self.dense(
            file,
            &format!("{prefix}.pre_encode.proj"),
            d,
            128 * 8,
            false,
        );
        self.norm(file, &format!("{prefix}.embed_norm"), d, true);
        self.norm(file, &format!("{prefix}.final_norm"), d, true);
        for layer in 0..layers {
            let l = format!("{prefix}.layers.{layer}");
            self.norm(file, &format!("{l}.norm1"), d, true);
            self.norm(file, &format!("{l}.norm2"), d, true);
            for (name, rows, cols, bias) in [
                ("attn.w_qkv", 3 * d, d, false),
                ("attn.out_proj", d, d, true),
                ("ffn.net.0", ff, d, true),
                ("ffn.net.3", d, ff, true),
            ] {
                if fp8 {
                    self.fp8(file, &format!("{l}.{name}"), rows, cols, bias);
                } else {
                    self.dense(file, &format!("{l}.{name}"), rows, cols, bias);
                }
            }
            if qk_norm {
                self.norm(file, &format!("{l}.attn.q_norm"), d / 2, true);
                self.norm(file, &format!("{l}.attn.k_norm"), d / 2, true);
            }
        }
    }
}

fn config_json() -> serde_json::Value {
    let encoder = |layers: usize, qk_norm: bool, rotary: f64| {
        json!({
            "_target_": "nemo.collections.asr.modules.transformer_encoder.TransformerEncoder",
            "feat_in": 128, "d_model": ENCODER_WIDTH, "n_heads": 2, "n_layers": layers,
            "qkv_bias": false, "qk_norm": qk_norm, "ff_expansion": 4, "pre_block_norm": true,
            "subsampling_factor": 8, "attn_mode": "full", "self_attention_model": "rope",
            "rope_base": 10000.0, "rotary_fraction": rotary, "xscaling": false,
            "subsampling": "feature_stacking"
        })
    };
    let mut diar_encoder = encoder(2, false, 1.0);
    diar_encoder["ff_expansion"] = json!(2);
    json!({
        "model_type": "nemo_speechlm",
        "architectures": ["Nemotron3_5TranscribeForConditionalGeneration"],
        "audio_locator_tag": "<|audio|>",
        "prompt_format": "nemotron3p5",
        "eos_token_id": [11],
        "perception": {
            "output_dim": HIDDEN,
            "preprocessor": {
                "sample_rate": 16000, "normalize": null, "window_size": 0.025, "window_stride": 0.01,
                "window": "hann", "features": 128, "n_fft": 512, "log": true, "frame_splicing": 1,
                "dither": 1e-05, "pad_to": 0, "pad_value": 0.0
            },
            "modality_adapter": {"_target_": "nemo.collections.speechlm2.modules.perception.IdentityConnector", "d_model": ENCODER_WIDTH}
        },
        "pe_encoder_config": {
            "asr_encoder_type": "transformer",
            "asr_encoder_cfg": encoder(2, true, 0.5),
            "diarization_model_cfg": {
                "max_num_of_spks": SPEAKERS,
                "high_resolution": true,
                "sortformer_modules": {
                    "num_spks": SPEAKERS, "fc_d_model": ENCODER_WIDTH, "tf_d_model": TF_WIDTH,
                    "spkcache_sil_frames_per_spk": 1, "pred_score_threshold": 0.25,
                    "scores_boost_latest": 0.05, "strong_boost_rate": 0.75, "weak_boost_rate": 1.5,
                    "min_pos_scores_rate": 0.5, "scores_add_rnd": 0, "use_learnable_sil_emb": true
                },
                "encoder": diar_encoder
            },
            "asr_normalize_type": "per_feature",
            "diar_normalize_type": "NA",
            "speaker_feature_mode": "thresholded",
            "speaker_activity_threshold": 0.5,
            "spk_kernel_scale": 1.0
        },
        "llm_config": {
            "model_type": "nemotron_h",
            "hidden_size": HIDDEN, "vocab_size": VOCAB, "layers_block_type": LAYERS,
            "mamba_num_heads": 4, "mamba_head_dim": 16, "n_groups": 2, "ssm_state_size": 8,
            "conv_kernel": 4, "chunk_size": 8, "mamba_hidden_act": "silu", "use_conv_bias": true,
            "mamba_proj_bias": false, "mamba_ssm_cache_dtype": "float32",
            "time_step_limit": [0.0, {"__float__": "Infinity"}],
            "n_routed_experts": EXPERTS, "num_experts_per_tok": 2,
            "moe_intermediate_size": EXPERT_WIDTH, "moe_shared_expert_intermediate_size": SHARED_WIDTH,
            "n_shared_experts": 1, "routed_scaling_factor": 2.5, "norm_topk_prob": true,
            "n_group": 1, "topk_group": 1, "mlp_hidden_act": "relu2", "mlp_bias": false,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "attention_bias": false, "layer_norm_epsilon": 1e-5, "tie_word_embeddings": false,
            "moe_latent_size": null
        },
        "quantization_config": {"quant_algo": "FP8", "quant_method": "modelopt", "ignore": exclude_list()}
    })
}

fn exclude_list() -> Vec<String> {
    let mut out = vec!["backbone.embeddings".to_string(), "lm_head".into()];
    for (index, kind) in LAYERS.iter().enumerate() {
        match *kind {
            "mamba" => out.push(format!("backbone.layers.{index}.mixer.conv1d")),
            "moe" => out.push(format!("backbone.layers.{index}.mixer.gate")),
            _ => {}
        }
    }
    out.extend([
        "ctc_timestamp*".into(),
        "perception.encoder.diarization_model*".into(),
        "perception.encoder.asr_encoder.pre_encode*".into(),
        "perception.proj".into(),
    ]);
    out
}

/// The word-level tokenizer: specials at the published ids, words after.
pub fn tokenizer_json() -> serde_json::Value {
    let mut vocab = serde_json::Map::new();
    let mut added = Vec::new();
    let mut special =
        |id: usize, content: String, vocab: &mut serde_json::Map<String, serde_json::Value>| {
            if id < VOCAB {
                vocab.insert(content.clone(), json!(id));
            }
            added.push(
                json!({"id": id, "content": content, "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true}),
            );
        };
    let names = |id: usize| -> String {
        match id {
            0 => "<unk>".into(),
            1 => "<s>".into(),
            2 => "</s>".into(),
            10 => "<|im_start|>".into(),
            11 => "<|im_end|>".into(),
            12 => "<think>".into(),
            13 => "</think>".into(),
            100..=109 => format!("<spk:{}>", id - 100),
            110..=119 => format!("<ev:e{}>", id - 110),
            120..=130 => format!("<sty:stt:s{}>", id - 120),
            131..=141 => format!("<sty:end:s{}>", id - 131),
            _ => format!("<reserved_{id}>"),
        }
    };
    for id in 0..142 {
        special(id, names(id), &mut vocab);
    }
    special(VOCAB, "<|audio|>".into(), &mut vocab);
    for id in 142..VOCAB {
        vocab.insert(format!("w{id}"), json!(id));
    }
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}
    })
}

/// Writes the synthetic checkpoint into `dir`.
pub fn write_checkpoint(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    let mut w = Writer {
        files: BTreeMap::new(),
        seed: 7,
    };
    let p = PERCEPTION_FILE;
    let read = |bytes: &[u8]| {
        bytes
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect::<Vec<_>>()
    };
    let fb = read(include_bytes!(
        "../../../../tests/fixtures/nemotron_transcribe/featurizer_fb.bf16"
    ));
    let window = read(include_bytes!(
        "../../../../tests/fixtures/nemotron_transcribe/featurizer_window.bf16"
    ));
    for prefix in [
        "perception.preprocessor.featurizer",
        "perception.encoder.diarization_model.preprocessor.featurizer",
    ] {
        w.bf16(p, format!("{prefix}.fb"), &[1, 128, 257], &fb);
        w.bf16(p, format!("{prefix}.window"), &[400], &window);
    }
    w.transformer(
        p,
        "perception.encoder.asr_encoder",
        2,
        4 * ENCODER_WIDTH,
        true,
        true,
    );
    w.transformer(
        p,
        "perception.encoder.diarization_model.encoder",
        2,
        2 * ENCODER_WIDTH,
        false,
        false,
    );
    let m = "perception.encoder.diarization_model.sortformer_modules";
    w.dense(
        p,
        &format!("{m}.encoder_proj"),
        TF_WIDTH,
        ENCODER_WIDTH,
        true,
    );
    w.dense(
        p,
        &format!("{m}.first_hidden_to_hidden"),
        TF_WIDTH,
        TF_WIDTH,
        true,
    );
    w.dense(
        p,
        &format!("{m}.hidden_to_spks"),
        SPEAKERS,
        2 * TF_WIDTH,
        true,
    );
    w.dense(
        p,
        &format!("{m}.single_hidden_to_spks"),
        SPEAKERS,
        TF_WIDTH,
        true,
    );
    w.random_bf16(
        p,
        format!("{m}.subpixel_upsample.weight"),
        &[TF_WIDTH * 8, TF_WIDTH, 3],
        0.3,
    );
    w.random_bf16(
        p,
        format!("{m}.subpixel_upsample.bias"),
        &[TF_WIDTH * 8],
        0.3,
    );
    w.random_bf16(p, format!("{m}.learnable_sil_emb"), &[ENCODER_WIDTH], 0.5);
    w.norm(p, "perception.encoder.asr_norm", ENCODER_WIDTH, true);
    w.norm(p, "perception.encoder.diar_norm", SPEAKERS, true);
    w.dense(p, "perception.proj", HIDDEN, ENCODER_WIDTH, true);

    let (s1, s2, s3, s4) = (
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
    );
    w.random_bf16(s1, "backbone.embeddings.weight", &[VOCAB, HIDDEN], 1.0);
    for (index, kind) in LAYERS.iter().enumerate() {
        let file = if index < 3 { s2 } else { s3 };
        let x = format!("backbone.layers.{index}.mixer");
        w.norm(
            file,
            &format!("backbone.layers.{index}.norm"),
            HIDDEN,
            false,
        );
        match *kind {
            "mamba" => {
                let (heads, inner, channels) = (4, 64, 64 + 2 * 2 * 8);
                let a_log = w
                    .values(heads, 1.0)
                    .into_iter()
                    .map(|v| (2.0 + v).ln())
                    .collect::<Vec<_>>();
                w.bf16(file, format!("{x}.A_log"), &[heads], &a_log);
                let d = w
                    .values(heads, 0.5)
                    .into_iter()
                    .map(|v| 1.0 + v)
                    .collect::<Vec<_>>();
                w.bf16(file, format!("{x}.D"), &[heads], &d);
                w.random_bf16(file, format!("{x}.dt_bias"), &[heads], 1.0);
                w.random_bf16(file, format!("{x}.conv1d.weight"), &[channels, 1, 4], 0.5);
                w.random_bf16(file, format!("{x}.conv1d.bias"), &[channels], 0.1);
                w.norm(file, &format!("{x}.norm"), inner, false);
                w.fp8(
                    file,
                    &format!("{x}.in_proj"),
                    inner + channels + heads,
                    HIDDEN,
                    false,
                );
                w.fp8(file, &format!("{x}.out_proj"), HIDDEN, inner, false);
            }
            "moe" => {
                w.random_bf16(file, format!("{x}.gate.weight"), &[EXPERTS, HIDDEN], 0.3);
                let bias = w
                    .values(EXPERTS, 0.05)
                    .into_iter()
                    .map(|v| 40.69 + v)
                    .collect::<Vec<_>>();
                w.f32(
                    file,
                    format!("{x}.gate.e_score_correction_bias"),
                    &[EXPERTS],
                    &bias,
                );
                for e in 0..EXPERTS {
                    w.fp8(
                        file,
                        &format!("{x}.experts.{e}.up_proj"),
                        EXPERT_WIDTH,
                        HIDDEN,
                        false,
                    );
                    w.fp8(
                        file,
                        &format!("{x}.experts.{e}.down_proj"),
                        HIDDEN,
                        EXPERT_WIDTH,
                        false,
                    );
                }
                w.fp8(
                    file,
                    &format!("{x}.shared_experts.up_proj"),
                    SHARED_WIDTH,
                    HIDDEN,
                    false,
                );
                w.fp8(
                    file,
                    &format!("{x}.shared_experts.down_proj"),
                    HIDDEN,
                    SHARED_WIDTH,
                    false,
                );
            }
            _ => {
                for (name, rows, cols) in [
                    ("q_proj", 64, HIDDEN),
                    ("k_proj", 32, HIDDEN),
                    ("v_proj", 32, HIDDEN),
                    ("o_proj", HIDDEN, 64),
                ] {
                    w.fp8(file, &format!("{x}.{name}"), rows, cols, false);
                }
                w.f32(file, format!("{x}.k_proj.k_scale"), &[], &[0.02]);
                w.f32(file, format!("{x}.v_proj.v_scale"), &[], &[0.004]);
            }
        }
    }
    w.norm(s4, "backbone.norm_f", HIDDEN, false);
    w.random_bf16(s4, "lm_head.weight", &[VOCAB, HIDDEN], 0.5);

    let mut weight_map = BTreeMap::new();
    for (file, tensors) in &w.files {
        let views = tensors
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                (
                    name.clone(),
                    TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        for (name, _) in &views {
            weight_map.insert(name.clone(), file.to_string());
        }
        safetensors::serialize_to_file(views, None, &dir.join(file)).unwrap();
    }
    let _ = CTC_FILE;
    std::fs::write(
        dir.join(INDEX_FILE),
        serde_json::to_vec_pretty(&json!({"metadata": {}, "weight_map": weight_map})).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("config.json"),
        serde_json::to_vec_pretty(&config_json()).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("hf_quant_config.json"),
        serde_json::to_vec_pretty(&json!({
            "producer": {"name": "modelopt", "version": "synthetic"},
            "quantization": {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8", "exclude_modules": exclude_list()}
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("generation_config.json"),
        br#"{"eos_token_id": [11], "pad_token_id": 0}"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("tokenizer.json"),
        serde_json::to_vec(&tokenizer_json()).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        br#"{"audio_token": "<|audio|>", "eos_token": "<|im_end|>"}"#,
    )
    .unwrap();
}

/// A directory removed when dropped.
pub struct TempCheckpoint(std::path::PathBuf);

impl TempCheckpoint {
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempCheckpoint {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A synthetic checkpoint in a fresh temporary directory.
pub fn temp_checkpoint() -> TempCheckpoint {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "izwi-nemotron-transcribe-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    write_checkpoint(&dir);
    TempCheckpoint(dir)
}

/// Speech-like test audio: a few harmonics with a slow envelope and noise.
pub fn test_audio(seconds: f32, seed: u64) -> Vec<f32> {
    let n = (seconds * 16_000.0) as usize;
    let noise = xorshift_values(n, seed, 0.01);
    (0..n)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            let envelope = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * 0.7 * t).sin();
            let f0 = 140.0 + 30.0 * (seed % 5) as f32;
            envelope
                * (0.3 * (2.0 * std::f32::consts::PI * f0 * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * 2.0 * f0 * t).sin())
                + noise[i]
        })
        .collect()
}
