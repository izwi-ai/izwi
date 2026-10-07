//! Shared fixture helpers for worker process tests: a tiny genuine LFM2 GGUF
//! that exercises the real RuntimeService loader, scheduler, managed memory,
//! and decode path without downloading the 1.2B artifact.

use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use izwi_core::{artifacts::ArtifactManifest, ModelVariant};
use std::path::{Path, PathBuf};

// Shared helpers: each integration binary links this module whole, so any
// helper it does not use locally is dead code in that one binary only.
#[allow(dead_code)]
pub fn id<T: TryFrom<&'static str>>(value: &'static str) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(value).unwrap()
}

pub fn write_tiny_lfm_fixture(models_dir: &Path) -> PathBuf {
    let model_dir = models_dir.join("LFM2.5-1.2B-Instruct-GGUF");
    std::fs::create_dir_all(&model_dir).unwrap();
    let tokenizer = tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [
                    ("<|pad|>".to_string(), 0),
                    ("<|im_start|>".to_string(), 1),
                    ("<|im_end|>".to_string(), 2),
                    ("a".to_string(), 3),
                    ("b".to_string(), 4),
                    ("Ã".to_string(), 5),
                    ("©".to_string(), 6),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("<|pad|>".to_string())
            .build()
            .unwrap(),
    );
    tokenizer
        .save(model_dir.join("tokenizer.json"), false)
        .unwrap();
    std::fs::write(
        model_dir.join("tokenizer_config.json"),
        r#"{"added_tokens_decoder":{"0":{"content":"<|pad|>","special":true},"1":{"content":"<|im_start|>","special":true},"2":{"content":"<|im_end|>","special":true}}}"#,
    )
    .unwrap();
    for (name, contents) in [
        ("chat_template.jinja", "{{ messages }}"),
        ("config.json", "{}"),
        ("generation_config.json", "{}"),
        ("special_tokens_map.json", "{}"),
    ] {
        std::fs::write(model_dir.join(name), contents).unwrap();
    }

    use gguf_file::Value;
    let metadata = [
        ("general.architecture", Value::String("lfm2".into())),
        ("lfm2.block_count", Value::U32(2)),
        ("lfm2.context_length", Value::U32(32)),
        ("lfm2.embedding_length", Value::U32(4)),
        ("lfm2.attention.head_count", Value::U32(1)),
        (
            "lfm2.attention.head_count_kv",
            Value::Array(vec![Value::U32(1), Value::U32(0)]),
        ),
        ("lfm2.attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
        ("lfm2.shortconv.l_cache", Value::U32(3)),
    ];
    let mut weights = Vec::new();
    let mut embeddings = vec![0.25f32; 7 * 4];
    embeddings[3 * 4..4 * 4].fill(2.0);
    let embeddings = Tensor::from_vec(embeddings, (7, 4), &Device::Cpu).unwrap();
    weights.push((
        "token_embd.weight".into(),
        QTensor::quantize(&embeddings, GgmlDType::F32).unwrap(),
    ));
    let mut add = |name: String, shape: &[usize]| {
        let tensor = Tensor::ones(shape, DType::F32, &Device::Cpu).unwrap();
        weights.push((name, QTensor::quantize(&tensor, GgmlDType::F32).unwrap()));
    };
    add("output_norm.weight".into(), &[4]);
    for layer in 0..2 {
        for name in ["attn_norm", "ffn_norm"] {
            add(format!("blk.{layer}.{name}.weight"), &[4]);
        }
        for name in ["ffn_gate", "ffn_up", "ffn_down"] {
            add(format!("blk.{layer}.{name}.weight"), &[4, 4]);
        }
    }
    for name in ["attn_q_norm", "attn_k_norm"] {
        add(format!("blk.0.{name}.weight"), &[4]);
    }
    for name in ["attn_q", "attn_k", "attn_v", "attn_output"] {
        add(format!("blk.0.{name}.weight"), &[4, 4]);
    }
    add("blk.1.shortconv.in_proj.weight".into(), &[12, 4]);
    add("blk.1.shortconv.out_proj.weight".into(), &[4, 4]);
    add("blk.1.shortconv.conv.weight".into(), &[4, 3]);
    let mut file =
        std::fs::File::create(model_dir.join("LFM2.5-1.2B-Instruct-Q4_K_M.gguf")).unwrap();
    gguf_file::write(
        &mut file,
        &metadata
            .iter()
            .map(|(name, value)| (*name, value))
            .collect::<Vec<_>>(),
        &weights
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let manifest = ArtifactManifest {
        schema_version: 1,
        variant: ModelVariant::Lfm2512BInstructGguf,
        repo_id: ModelVariant::Lfm2512BInstructGguf.repo_id().into(),
        revision: "tiny-lfm-fixture-v1".into(),
        files: vec![
            "LFM2.5-1.2B-Instruct-Q4_K_M.gguf".into(),
            "tokenizer.json".into(),
            "tokenizer_config.json".into(),
            "chat_template.jinja".into(),
            "config.json".into(),
            "generation_config.json".into(),
            "special_tokens_map.json".into(),
        ],
    };
    std::fs::write(
        model_dir.join("izwi-artifact.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    model_dir
}

/// Tiny genuine Qwen3-MoE GGUF: one decoder layer whose FFN is a 2-expert
/// sparse block with llama.cpp fused expert tensors (`ffn_gate_inp` router,
/// `ffn_{gate,up,down}_exps` fused `[n_expert, n_ff, hidden]`), the
/// `qwen3moe.*` metadata prefix, and the im_start/im_end chat tokenizer. This
/// exercises the real sparse-expert load and dispatch path (DS10 groundwork)
/// without downloading the 30.5B artifact.
#[allow(dead_code)]
pub fn write_tiny_qwen3_moe_fixture(models_dir: &Path) -> PathBuf {
    let model_dir = models_dir.join("Qwen3-30B-A3B-GGUF");
    std::fs::create_dir_all(&model_dir).unwrap();
    let tokenizer = tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [
                    ("<|pad|>".to_string(), 0),
                    ("<|im_start|>".to_string(), 1),
                    ("<|im_end|>".to_string(), 2),
                    ("a".to_string(), 3),
                    ("b".to_string(), 4),
                    ("Ã".to_string(), 5),
                    ("©".to_string(), 6),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("<|pad|>".to_string())
            .build()
            .unwrap(),
    );
    tokenizer
        .save(model_dir.join("tokenizer.json"), false)
        .unwrap();
    std::fs::write(
        model_dir.join("tokenizer_config.json"),
        r#"{"added_tokens_decoder":{"0":{"content":"<|pad|>","special":true},"1":{"content":"<|im_start|>","special":true},"2":{"content":"<|im_end|>","special":true}}}"#,
    )
    .unwrap();

    use gguf_file::Value;
    let hidden = 4_usize;
    let n_experts = 2_usize;
    let n_ff = 4_usize;
    let fixture_vocab = ["<|pad|>", "<|im_start|>", "<|im_end|>", "a", "b", "Ã", "©"];
    let metadata = vec![
        ("general.architecture", Value::String("qwen3moe".into())),
        ("qwen3moe.block_count", Value::U32(1)),
        ("qwen3moe.context_length", Value::U32(32)),
        ("qwen3moe.embedding_length", Value::U32(hidden as u32)),
        ("qwen3moe.feed_forward_length", Value::U32(4)),
        ("qwen3moe.attention.head_count", Value::U32(2)),
        ("qwen3moe.attention.head_count_kv", Value::U32(1)),
        ("qwen3moe.attention.key_length", Value::U32(2)),
        (
            "qwen3moe.attention.layer_norm_rms_epsilon",
            Value::F32(1e-5),
        ),
        ("qwen3moe.rope.freq_base", Value::F32(10_000.0)),
        ("qwen3moe.expert_count", Value::U32(n_experts as u32)),
        ("qwen3moe.expert_used_count", Value::U32(2)),
        (
            "qwen3moe.expert_feed_forward_length",
            Value::U32(n_ff as u32),
        ),
        ("qwen3moe.norm_topk_prob", Value::F32(1.0)),
        // Embedded tokenizer metadata: the qwen3 GGUF config parser derives
        // vocab_size from the token array (real Qwen3 GGUFs embed it).
        (
            "tokenizer.ggml.tokens",
            Value::Array(
                fixture_vocab
                    .iter()
                    .map(|token| Value::String((*token).into()))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.scores",
            Value::Array(vec![Value::F32(0.0); fixture_vocab.len()]),
        ),
    ];
    let mut weights = Vec::new();
    let embeddings = vec![0.25f32; 7 * hidden];
    weights.push((
        "token_embd.weight".into(),
        QTensor::quantize(
            &Tensor::from_vec(embeddings, (7, hidden), &Device::Cpu).unwrap(),
            GgmlDType::F32,
        )
        .unwrap(),
    ));
    let mut add = |name: String, shape: &[usize]| {
        let tensor = Tensor::from_vec(
            (0..shape.iter().product::<usize>())
                .map(|idx| ((idx * 7 % 13) as f32 - 6.0) / 16.0)
                .collect::<Vec<_>>(),
            shape,
            &Device::Cpu,
        )
        .unwrap();
        weights.push((name, QTensor::quantize(&tensor, GgmlDType::F32).unwrap()));
    };
    add("output_norm.weight".into(), &[hidden]);
    add("blk.0.attn_norm.weight".into(), &[hidden]);
    add("blk.0.ffn_norm.weight".into(), &[hidden]);
    // q projects 2 heads x head_dim 2 = 4; k/v project 1 kv-head x 2 = 2.
    add("blk.0.attn_q.weight".into(), &[hidden, hidden]);
    add("blk.0.attn_k.weight".into(), &[2, hidden]);
    add("blk.0.attn_v.weight".into(), &[2, hidden]);
    add("blk.0.attn_output.weight".into(), &[hidden, hidden]);
    // Router: [num_experts, hidden]; distinct logits so top-2 of 2 is stable.
    add("blk.0.ffn_gate_inp.weight".into(), &[n_experts, hidden]);
    // Fused experts: [num_experts, n_ff, hidden] for gate/up, [num_experts,
    // hidden, n_ff] for down — the llama.cpp MoE tensor convention.
    add(
        "blk.0.ffn_gate_exps.weight".into(),
        &[n_experts, n_ff, hidden],
    );
    add(
        "blk.0.ffn_up_exps.weight".into(),
        &[n_experts, n_ff, hidden],
    );
    add(
        "blk.0.ffn_down_exps.weight".into(),
        &[n_experts, hidden, n_ff],
    );
    let mut file = std::fs::File::create(model_dir.join("Qwen3-30B-A3B-Q4_K_M.gguf")).unwrap();
    gguf_file::write(
        &mut file,
        &metadata
            .iter()
            .map(|(name, value)| (*name, value))
            .collect::<Vec<_>>(),
        &weights
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let manifest = ArtifactManifest {
        schema_version: 1,
        variant: ModelVariant::Qwen3Moe30bA3bGguf,
        repo_id: ModelVariant::Qwen3Moe30bA3bGguf.repo_id().into(),
        revision: "tiny-qwen3-moe-fixture-v1".into(),
        files: vec![
            "Qwen3-30B-A3B-Q4_K_M.gguf".into(),
            "tokenizer.json".into(),
            "tokenizer_config.json".into(),
        ],
    };
    std::fs::write(
        model_dir.join("izwi-artifact.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    model_dir
}

/// Explicit fixture generation for benchmark runs (DS0.7): run with
/// `IZWI_BENCH_FIXTURE_DIR=<models root> cargo test -p izwi-serving-worker \
///  --test backend_parity generate_benchmark_fixture -- --ignored`
#[test]
#[ignore = "explicit benchmark fixture generation"]
fn generate_benchmark_fixture() {
    let root = std::env::var("IZWI_BENCH_FIXTURE_DIR").expect("set IZWI_BENCH_FIXTURE_DIR");
    let dir = write_tiny_lfm_fixture(std::path::Path::new(&root));
    println!("fixture model dir: {}", dir.display());
}

pub const QWEN38_FIXTURE_REVISION: &str = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a";

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
        .collect()
}

struct Qwen38FixtureTensor {
    name: String,
    dtype: safetensors::Dtype,
    shape: Vec<usize>,
    data: Vec<u8>,
}

fn qwen38_dense(name: impl Into<String>, shape: &[usize], values: &[f32]) -> Qwen38FixtureTensor {
    Qwen38FixtureTensor {
        name: name.into(),
        dtype: safetensors::Dtype::BF16,
        shape: shape.to_vec(),
        data: bf16_bytes(values),
    }
}

/// A zero block-FP8 projection weight plus its unit `weight_scale_inv`.
fn qwen38_fp8_projection(
    name: impl Into<String>,
    shape: [usize; 2],
    block: [usize; 2],
) -> Vec<Qwen38FixtureTensor> {
    let name = name.into();
    let scale_shape = [shape[0].div_ceil(block[0]), shape[1].div_ceil(block[1])];
    vec![
        Qwen38FixtureTensor {
            name: name.clone(),
            dtype: safetensors::Dtype::F8_E4M3,
            shape: shape.to_vec(),
            data: vec![0; shape[0] * shape[1]],
        },
        qwen38_dense(
            name.replace(".weight", ".weight_scale_inv"),
            &scale_shape,
            &vec![1.0; scale_shape[0] * scale_shape[1]],
        ),
    ]
}

/// Tiny synthetic hybrid Qwen3.8 checkpoint (DS1.5): one linear-attention block
/// plus one full-attention block, so the DS1.2b committed-snapshot sharing gate
/// accepts the sealed managed contract. The geometry requires
/// `IZWI_ALLOW_SYNTHETIC_QWEN38_GEOMETRY=1` in the loading process; the values
/// mirror the in-process recovery fixtures in izwi-core exactly.
pub fn write_tiny_qwen38_hybrid_fixture(models_dir: &Path) -> PathBuf {
    use safetensors::tensor::TensorView;
    use std::collections::BTreeMap;

    const HIDDEN: usize = 4;
    const FF: usize = 4;
    const Q_WIDTH: usize = 8; // 2 heads * head_dim 2 * 2 (gated)
    const KV_WIDTH: usize = 2; // 1 kv head * head_dim 2
    const OUTPUT_WIDTH: usize = 4; // 2 heads * head_dim 2
    const BLOCK: [usize; 2] = [2, 2];

    let model_dir = models_dir.join("Qwen3.8-27B-FP8");
    std::fs::create_dir_all(&model_dir).unwrap();

    // --- config.json: synthetic hybrid geometry (structural checks still run).
    let text_config = serde_json::json!({
        "attention_bias": false,
        "attention_dropout": 0.0,
        "attn_output_gate": true,
        "bos_token_id": 2,
        "dtype": "bfloat16",
        "eos_token_id": 2,
        "full_attention_interval": 2,
        "head_dim": 2,
        "hidden_act": "silu",
        "hidden_size": HIDDEN,
        "intermediate_size": FF,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 3,
        "linear_key_head_dim": 1,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 1,
        "linear_value_head_dim": 1,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 512,
        "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": false,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 1,
        "output_gate_type": "swish",
        "partial_rotary_factor": 1.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": false,
        "use_cache": true,
        "vocab_size": 8,
        "rope_parameters": {
            "mrope_interleaved": true,
            "mrope_section": [1, 0, 0],
            "partial_rotary_factor": 1.0,
            "rope_theta": 10000.0,
            "rope_type": "default"
        }
    });
    let config = serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "language_model_only": false,
        "model_type": "qwen3_5",
        "tie_word_embeddings": false,
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [2, 2]
        },
        "text_config": text_config
    });
    std::fs::write(model_dir.join("config.json"), config.to_string()).unwrap();

    // --- mtp.safetensors: the full MTP checkpoint contract (validated on load,
    // executed only when MTP is enabled).
    let mut fc = vec![0.0f32; HIDDEN * HIDDEN * 2];
    for row in 0..HIDDEN {
        fc[row * HIDDEN * 2 + row] = 1.0;
    }
    let mut mtp_tensors = vec![
        qwen38_dense("mtp.fc.weight", &[HIDDEN, HIDDEN * 2], &fc),
        qwen38_dense(
            "mtp.pre_fc_norm_embedding.weight",
            &[HIDDEN],
            &[0.0; HIDDEN],
        ),
        qwen38_dense("mtp.pre_fc_norm_hidden.weight", &[HIDDEN], &[0.0; HIDDEN]),
        qwen38_dense("mtp.norm.weight", &[HIDDEN], &[0.0; HIDDEN]),
        qwen38_dense(
            "mtp.layers.0.input_layernorm.weight",
            &[HIDDEN],
            &[0.0; HIDDEN],
        ),
        qwen38_dense(
            "mtp.layers.0.post_attention_layernorm.weight",
            &[HIDDEN],
            &[0.0; HIDDEN],
        ),
        qwen38_dense("mtp.layers.0.self_attn.q_norm.weight", &[2], &[0.0; 2]),
        qwen38_dense("mtp.layers.0.self_attn.k_norm.weight", &[2], &[0.0; 2]),
    ];
    for (name, shape) in [
        ("mtp.layers.0.mlp.gate_proj.weight", [FF, HIDDEN]),
        ("mtp.layers.0.mlp.up_proj.weight", [FF, HIDDEN]),
        ("mtp.layers.0.mlp.down_proj.weight", [HIDDEN, FF]),
        ("mtp.layers.0.self_attn.q_proj.weight", [Q_WIDTH, HIDDEN]),
        ("mtp.layers.0.self_attn.k_proj.weight", [KV_WIDTH, HIDDEN]),
        ("mtp.layers.0.self_attn.v_proj.weight", [KV_WIDTH, HIDDEN]),
        (
            "mtp.layers.0.self_attn.o_proj.weight",
            [HIDDEN, OUTPUT_WIDTH],
        ),
    ] {
        mtp_tensors.extend(qwen38_fp8_projection(name, shape, BLOCK));
    }
    assert_eq!(
        mtp_tensors.len(),
        22,
        "MTP tensor count must match the loader's manifest"
    );

    // --- target.safetensors: the text backbone. Block 1 (full attention)
    // reuses the MTP layer tensors; BLOCK 0 (linear attention) gets the MTP
    // MLP/norms plus a nonzero DeltaNet set so attach state depends on conv
    // history and recurrent state, not only the token cursor.
    let mut target_tensors: Vec<Qwen38FixtureTensor> = Vec::new();
    for tensor in &mtp_tensors {
        if tensor.name.starts_with("mtp.layers.0.") {
            target_tensors.push(Qwen38FixtureTensor {
                name: tensor
                    .name
                    .replacen("mtp.layers.0.", "model.language_model.layers.1.", 1),
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
                data: tensor.data.clone(),
            });
        }
    }
    for tensor in &mtp_tensors {
        if tensor.name.contains(".mlp.")
            || tensor.name.ends_with(".input_layernorm.weight")
            || tensor.name.ends_with(".post_attention_layernorm.weight")
        {
            target_tensors.push(Qwen38FixtureTensor {
                name: tensor
                    .name
                    .replacen("mtp.layers.0.", "model.language_model.layers.0.", 1),
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
                data: tensor.data.clone(),
            });
        }
    }
    // Dense linear-attention math tensors (the loader requires these dense,
    // without scale companions).
    for (name, shape, value) in [
        ("dt_bias", vec![1usize], 0.1f32),
        ("A_log", vec![1], -1.0),
        ("conv1d.weight", vec![3, 1, 3], 0.25),
        ("norm.weight", vec![1], 1.0),
        ("in_proj_a.weight", vec![1, 4], 0.125),
        ("in_proj_b.weight", vec![1, 4], 0.125),
    ] {
        target_tensors.push(qwen38_dense(
            format!("model.language_model.layers.0.linear_attn.{name}"),
            &shape,
            &vec![value; shape.iter().product::<usize>()],
        ));
    }
    // Linear-attention projections are block-FP8 pairs in the checkpoint
    // contract; the loader rejects dense weights that carry a scale companion
    // and requires the scale name for every projection.
    for (name, shape) in [
        ("in_proj_qkv", [3usize, 4]),
        ("in_proj_z", [1, 4]),
        ("out_proj", [4, 1]),
    ] {
        target_tensors.extend(qwen38_fp8_projection(
            format!("model.language_model.layers.0.linear_attn.{name}.weight"),
            shape,
            BLOCK,
        ));
    }
    let mut embedding = vec![0.125f32; 8 * HIDDEN];
    for row in 0..8 {
        if row % 4 == row / 4 % 4 {
            for column in 0..HIDDEN {
                embedding[row * HIDDEN + column] = 1.0;
            }
        }
    }
    target_tensors.push(qwen38_dense(
        "model.language_model.embed_tokens.weight",
        &[8, HIDDEN],
        &embedding,
    ));
    target_tensors.push(qwen38_dense("lm_head.weight", &[8, HIDDEN], &embedding));
    target_tensors.push(qwen38_dense(
        "model.language_model.norm.weight",
        &[HIDDEN],
        &[0.0; HIDDEN],
    ));

    // --- write shards + index.
    let write_shard = |path: &Path, tensors: &[Qwen38FixtureTensor]| {
        let views = tensors
            .iter()
            .map(|tensor| {
                (
                    tensor.name.clone(),
                    TensorView::new(tensor.dtype, tensor.shape.clone(), &tensor.data).unwrap(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        safetensors::serialize_to_file(&views, &None, path).unwrap();
    };
    write_shard(&model_dir.join("mtp.safetensors"), &mtp_tensors);
    write_shard(&model_dir.join("target.safetensors"), &target_tensors);

    let mut weight_map = serde_json::Map::new();
    for tensor in mtp_tensors.iter().chain(target_tensors.iter()) {
        let shard = if tensor.name.starts_with("mtp.") {
            "mtp.safetensors"
        } else {
            "target.safetensors"
        };
        weight_map.insert(tensor.name.clone(), serde_json::json!(shard));
    }
    std::fs::write(
        model_dir.join("model.safetensors.index.json"),
        serde_json::json!({ "weight_map": weight_map }).to_string(),
    )
    .unwrap();

    // --- tokenizer: WordLevel over an 8-token vocab (ids 0-7 cover every
    // model output). Specials live in tokenizer_config.json; qwen38 chat
    // encoding splits them out as literals before the inner tokenizer runs.
    let tokenizer = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {
                "<|pad|>": 0,
                "<|im_start|>": 1,
                "<|im_end|>": 2,
                "<|image_pad|>": 3,
                "<|video_pad|>": 4,
                "a": 5,
                "b": 6,
                "c": 7
            },
            "unk_token": "a"
        }
    });
    std::fs::write(model_dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    let tokenizer_config = serde_json::json!({
        "added_tokens_decoder": {
            "0": { "content": "<|pad|>", "special": true },
            "1": { "content": "<|im_start|>", "special": true },
            "2": { "content": "<|im_end|>", "special": true },
            "3": { "content": "<|image_pad|>", "special": true },
            "4": { "content": "<|video_pad|>", "special": true }
        },
        "eos_token": "<|im_end|>",
        "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    });
    std::fs::write(
        model_dir.join("tokenizer_config.json"),
        tokenizer_config.to_string(),
    )
    .unwrap();

    // The downloader's qwen38 bundle-completeness gate requires every metadata
    // file of the published bundle to exist, even when unused (tokenizer.json
    // wins over vocab.json+merges; render_prompt builds the template inline).
    std::fs::write(model_dir.join("generation_config.json"), "{}").unwrap();
    std::fs::write(
        model_dir.join("chat_template.jinja"),
        "{% for message in messages %}{{ message['content'] }}{% endfor %}",
    )
    .unwrap();
    std::fs::write(model_dir.join("vocab.json"), "{}").unwrap();
    std::fs::write(model_dir.join("merges.txt"), "").unwrap();
    std::fs::write(model_dir.join("preprocessor_config.json"), "{}").unwrap();
    std::fs::write(model_dir.join("video_preprocessor_config.json"), "{}").unwrap();

    let manifest = ArtifactManifest {
        schema_version: 1,
        variant: ModelVariant::Qwen3827BFp8,
        repo_id: ModelVariant::Qwen3827BFp8.repo_id().into(),
        revision: QWEN38_FIXTURE_REVISION.into(),
        files: vec![
            "chat_template.jinja".into(),
            "config.json".into(),
            "generation_config.json".into(),
            "merges.txt".into(),
            "model.safetensors.index.json".into(),
            "mtp.safetensors".into(),
            "preprocessor_config.json".into(),
            "target.safetensors".into(),
            "tokenizer.json".into(),
            "tokenizer_config.json".into(),
            "video_preprocessor_config.json".into(),
            "vocab.json".into(),
        ],
    };
    std::fs::write(
        model_dir.join("izwi-artifact.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    model_dir
}

/// Explicit fixture generation for the DS1.5 prefix benchmark runs: run with
/// `IZWI_BENCH_FIXTURE_DIR=<models root> cargo test -p izwi-serving-worker \
///  --test backend_parity generate_qwen38_benchmark_fixture -- --ignored`
#[test]
#[ignore = "explicit benchmark fixture generation"]
fn generate_qwen38_benchmark_fixture() {
    let root = std::env::var("IZWI_BENCH_FIXTURE_DIR").expect("set IZWI_BENCH_FIXTURE_DIR");
    let dir = write_tiny_qwen38_hybrid_fixture(std::path::Path::new(&root));
    println!("fixture model dir: {}", dir.display());
}
/// Tiny synthetic native block-FP8 Qwen3.5/3.6-MoE checkpoint: a forward-capable
/// 4-layer hybrid trunk (3 DeltaNet + 1 gated full attention, interval 4) with
/// 2 routed experts (top-1) plus a shared expert. Values mirror the in-process
/// `qwen36moe::native` recovery fixtures exactly: all-ones weights, 0x38
/// E4M3 bytes (= 1.0), unit block scales, and an lm_head row bias so greedy
/// decode emits a plain-letter token deterministically. The geometry requires
/// `IZWI_ALLOW_SYNTHETIC_QWEN36_MOE_GEOMETRY=1` in the loading process; the
/// downloader bundle gate requires the real pinned artifact revision.
#[allow(dead_code)]
pub fn write_tiny_qwen35_moe_fixture(models_dir: &Path) -> PathBuf {
    let model_dir = models_dir.join("Qwen3.6-35B-A3B-FP8");
    std::fs::create_dir_all(&model_dir).unwrap();

    const HIDDEN: usize = 32;
    const VOCAB: usize = 32;
    const QUERY_WIDTH: usize = 2 * 16; // 2 attention heads × head_dim 16
    const KV_WIDTH: usize = 16; // 1 kv head × head_dim 16
    const MOE_INTERMEDIATE: usize = 32;
    const SHARED_INTERMEDIATE: usize = 32;
    const NUM_EXPERTS: usize = 2;
    const SSM_TIME_STEP_RANK: usize = 4; // linear_num_value_heads
    const SSM_QK_WIDTH: usize = 2 * 16; // linear_num_key_heads × linear_key_head_dim
    const SSM_V_WIDTH: usize = 4 * 16; // linear_num_value_heads × linear_value_head_dim
    const SSM_CONV_CHANNELS: usize = 2 * SSM_QK_WIDTH + SSM_V_WIDTH;
    const CONV_KERNEL: usize = 2;
    const BLOCK: [usize; 2] = [4, 4];

    let config = serde_json::json!({
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "num_hidden_layers": 4,
            "full_attention_interval": 4,
            "hidden_size": HIDDEN,
            "vocab_size": VOCAB,
            "max_position_embeddings": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "num_experts": NUM_EXPERTS,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": MOE_INTERMEDIATE,
            "shared_expert_intermediate_size": SHARED_INTERMEDIATE,
            "linear_num_value_heads": SSM_TIME_STEP_RANK,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": CONV_KERNEL,
            "rms_norm_eps": 1e-6,
            "mamba_ssm_dtype": "float32",
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention"
            ],
            "rope_parameters": {
                "rope_type": "mrope",
                "mrope_interleaved": true,
                "mrope_section": [2, 2, 0],
                "rope_theta": 1_000_000.0,
                "partial_rotary_factor": 0.5
            }
        },
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": BLOCK
        }
    });
    std::fs::write(model_dir.join("config.json"), config.to_string()).unwrap();

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect()
    }

    struct Qwen36MoeFixtureTensor {
        name: String,
        dtype: safetensors::Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    let dense = |name: String, shape: &[usize], value: f32| Qwen36MoeFixtureTensor {
        name,
        dtype: safetensors::Dtype::BF16,
        shape: shape.to_vec(),
        data: bf16_bytes(&vec![value; shape.iter().product::<usize>()]),
    };

    // Block-FP8 projection: an all-ones (0x38 = 1.0 in E4M3) weight with a
    // unit `weight_scale_inv` companion at [ceil(rows/block), ceil(cols/block)].
    let fp8_projection = |name: String, rows: usize, cols: usize| {
        let name: String = name;
        let (scale_rows, scale_cols) = (rows.div_ceil(BLOCK[0]), cols.div_ceil(BLOCK[1]));
        vec![
            Qwen36MoeFixtureTensor {
                name: name.clone(),
                dtype: safetensors::Dtype::F8_E4M3,
                shape: vec![rows, cols],
                data: vec![0x38; rows * cols],
            },
            Qwen36MoeFixtureTensor {
                name: format!("{}.weight_scale_inv", name.strip_suffix(".weight").unwrap()),
                dtype: safetensors::Dtype::BF16,
                shape: vec![scale_rows, scale_cols],
                data: bf16_bytes(&vec![1.0; scale_rows * scale_cols]),
            },
        ]
    };

    let mut tensors: Vec<Qwen36MoeFixtureTensor> = Vec::new();
    // Embeddings and lm_head are FP8-excluded dense tensors. lm_head row 7
    // ("c" in the fixture tokenizer) carries a larger value so greedy decode
    // deterministically emits a plain-letter token instead of a special.
    let mut lm_head = vec![0.5f32; VOCAB * HIDDEN];
    for column in 0..HIDDEN {
        lm_head[7 * HIDDEN + column] = 4.0;
    }
    tensors.push(Qwen36MoeFixtureTensor {
        name: "model.language_model.embed_tokens.weight".into(),
        dtype: safetensors::Dtype::BF16,
        shape: vec![VOCAB, HIDDEN],
        data: bf16_bytes(&vec![1.0; VOCAB * HIDDEN]),
    });
    tensors.push(Qwen36MoeFixtureTensor {
        name: "lm_head.weight".into(),
        dtype: safetensors::Dtype::BF16,
        shape: vec![VOCAB, HIDDEN],
        data: bf16_bytes(&lm_head),
    });
    tensors.push(dense(
        "model.language_model.norm.weight".to_string(),
        &[HIDDEN],
        1.0,
    ));

    for layer in 0..4usize {
        let prefix = format!("model.language_model.layers.{layer}");
        tensors.push(dense(
            format!("{prefix}.input_layernorm.weight"),
            &[HIDDEN],
            1.0,
        ));
        tensors.push(dense(
            format!("{prefix}.post_attention_layernorm.weight"),
            &[HIDDEN],
            1.0,
        ));
        // Router: dense FP8-excluded [num_experts, hidden].
        tensors.push(dense(
            format!("{prefix}.mlp.gate.weight"),
            &[NUM_EXPERTS, HIDDEN],
            1.0,
        ));
        for expert in 0..NUM_EXPERTS {
            tensors.extend(fp8_projection(
                format!("{prefix}.mlp.experts.{expert}.gate_proj.weight"),
                MOE_INTERMEDIATE,
                HIDDEN,
            ));
            tensors.extend(fp8_projection(
                format!("{prefix}.mlp.experts.{expert}.up_proj.weight"),
                MOE_INTERMEDIATE,
                HIDDEN,
            ));
            tensors.extend(fp8_projection(
                format!("{prefix}.mlp.experts.{expert}.down_proj.weight"),
                HIDDEN,
                MOE_INTERMEDIATE,
            ));
        }
        tensors.extend(fp8_projection(
            format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
            SHARED_INTERMEDIATE,
            HIDDEN,
        ));
        tensors.extend(fp8_projection(
            format!("{prefix}.mlp.shared_expert.up_proj.weight"),
            SHARED_INTERMEDIATE,
            HIDDEN,
        ));
        tensors.extend(fp8_projection(
            format!("{prefix}.mlp.shared_expert.down_proj.weight"),
            HIDDEN,
            SHARED_INTERMEDIATE,
        ));
        tensors.push(dense(
            format!("{prefix}.mlp.shared_expert_gate.weight"),
            &[1, HIDDEN],
            1.0,
        ));
        if layer == 3 {
            // Gated full attention: q_proj fuses the per-head output gate.
            tensors.extend(fp8_projection(
                format!("{prefix}.self_attn.q_proj.weight"),
                QUERY_WIDTH * 2,
                HIDDEN,
            ));
            tensors.extend(fp8_projection(
                format!("{prefix}.self_attn.k_proj.weight"),
                KV_WIDTH,
                HIDDEN,
            ));
            tensors.extend(fp8_projection(
                format!("{prefix}.self_attn.v_proj.weight"),
                KV_WIDTH,
                HIDDEN,
            ));
            tensors.extend(fp8_projection(
                format!("{prefix}.self_attn.o_proj.weight"),
                HIDDEN,
                QUERY_WIDTH,
            ));
            tensors.push(dense(
                format!("{prefix}.self_attn.q_norm.weight"),
                &[16],
                1.0,
            ));
            tensors.push(dense(
                format!("{prefix}.self_attn.k_norm.weight"),
                &[16],
                1.0,
            ));
        } else {
            // Gated DeltaNet: in_proj_qkv/in_proj_z and out_proj are
            // block-FP8 pairs, the per-head tensors stay dense — the
            // checkpoint contract.
            tensors.extend(fp8_projection(
                format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                SSM_CONV_CHANNELS,
                HIDDEN,
            ));
            tensors.extend(fp8_projection(
                format!("{prefix}.linear_attn.in_proj_z.weight"),
                SSM_V_WIDTH,
                HIDDEN,
            ));
            tensors.push(dense(
                format!("{prefix}.linear_attn.in_proj_b.weight"),
                &[SSM_TIME_STEP_RANK, HIDDEN],
                1.0,
            ));
            tensors.push(dense(
                format!("{prefix}.linear_attn.in_proj_a.weight"),
                &[SSM_TIME_STEP_RANK, HIDDEN],
                1.0,
            ));
            tensors.push(dense(
                format!("{prefix}.linear_attn.A_log"),
                &[SSM_TIME_STEP_RANK],
                -1.0,
            ));
            tensors.push(dense(
                format!("{prefix}.linear_attn.dt_bias"),
                &[SSM_TIME_STEP_RANK],
                0.1,
            ));
            tensors.push(dense(
                format!("{prefix}.linear_attn.conv1d.weight"),
                &[SSM_CONV_CHANNELS, 1, CONV_KERNEL],
                1.0,
            ));
            tensors.push(dense(
                format!("{prefix}.linear_attn.norm.weight"),
                &[16],
                1.0,
            ));
            tensors.extend(fp8_projection(
                format!("{prefix}.linear_attn.out_proj.weight"),
                HIDDEN,
                SSM_V_WIDTH,
            ));
        }
    }
    // Vision tower: an auxiliary scope the loader must skip.
    tensors.push(dense(
        "model.visual.blocks.0.attn.qkv.weight".to_string(),
        &[4, 4],
        1.0,
    ));
    // The full MTP draft manifest at this fixture's geometry (24 fixed
    // tensors + 6 per routed expert), mirroring the loader's MoE contract:
    // the fixture therefore also loads with MTP validation enabled, not
    // only on the skip path.
    tensors.push(dense(
        "mtp.fc.weight".to_string(),
        &[HIDDEN, HIDDEN * 2],
        1.0,
    ));
    tensors.push(dense("mtp.norm.weight".to_string(), &[HIDDEN], 1.0));
    tensors.push(dense(
        "mtp.pre_fc_norm_embedding.weight".to_string(),
        &[HIDDEN],
        1.0,
    ));
    tensors.push(dense(
        "mtp.pre_fc_norm_hidden.weight".to_string(),
        &[HIDDEN],
        1.0,
    ));
    tensors.push(dense(
        "mtp.layers.0.input_layernorm.weight".to_string(),
        &[HIDDEN],
        1.0,
    ));
    tensors.push(dense(
        "mtp.layers.0.post_attention_layernorm.weight".to_string(),
        &[HIDDEN],
        1.0,
    ));
    tensors.push(dense(
        "mtp.layers.0.mlp.gate.weight".to_string(),
        &[NUM_EXPERTS, HIDDEN],
        1.0,
    ));
    for expert in 0..NUM_EXPERTS {
        tensors.extend(fp8_projection(
            format!("mtp.layers.0.mlp.experts.{expert}.gate_proj.weight"),
            MOE_INTERMEDIATE,
            HIDDEN,
        ));
        tensors.extend(fp8_projection(
            format!("mtp.layers.0.mlp.experts.{expert}.up_proj.weight"),
            MOE_INTERMEDIATE,
            HIDDEN,
        ));
        tensors.extend(fp8_projection(
            format!("mtp.layers.0.mlp.experts.{expert}.down_proj.weight"),
            HIDDEN,
            MOE_INTERMEDIATE,
        ));
    }
    tensors.extend(fp8_projection(
        "mtp.layers.0.mlp.shared_expert.gate_proj.weight".to_string(),
        SHARED_INTERMEDIATE,
        HIDDEN,
    ));
    tensors.extend(fp8_projection(
        "mtp.layers.0.mlp.shared_expert.up_proj.weight".to_string(),
        SHARED_INTERMEDIATE,
        HIDDEN,
    ));
    tensors.extend(fp8_projection(
        "mtp.layers.0.mlp.shared_expert.down_proj.weight".to_string(),
        HIDDEN,
        SHARED_INTERMEDIATE,
    ));
    tensors.push(dense(
        "mtp.layers.0.mlp.shared_expert_gate.weight".to_string(),
        &[1, HIDDEN],
        1.0,
    ));
    tensors.push(dense(
        "mtp.layers.0.self_attn.q_norm.weight".to_string(),
        &[16],
        1.0,
    ));
    tensors.push(dense(
        "mtp.layers.0.self_attn.k_norm.weight".to_string(),
        &[16],
        1.0,
    ));
    tensors.extend(fp8_projection(
        "mtp.layers.0.self_attn.q_proj.weight".to_string(),
        QUERY_WIDTH * 2,
        HIDDEN,
    ));
    tensors.extend(fp8_projection(
        "mtp.layers.0.self_attn.k_proj.weight".to_string(),
        KV_WIDTH,
        HIDDEN,
    ));
    tensors.extend(fp8_projection(
        "mtp.layers.0.self_attn.v_proj.weight".to_string(),
        KV_WIDTH,
        HIDDEN,
    ));
    tensors.extend(fp8_projection(
        "mtp.layers.0.self_attn.o_proj.weight".to_string(),
        HIDDEN,
        QUERY_WIDTH,
    ));

    let write_shard = |path: &Path, tensors: &[Qwen36MoeFixtureTensor]| {
        let views = tensors
            .iter()
            .map(|tensor| {
                (
                    tensor.name.clone(),
                    safetensors::tensor::TensorView::new(
                        tensor.dtype,
                        tensor.shape.clone(),
                        &tensor.data,
                    )
                    .unwrap(),
                )
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        safetensors::serialize_to_file(&views, &None, path).unwrap();
    };
    write_shard(&model_dir.join("layers.safetensors"), &tensors);

    let mut weight_map = serde_json::Map::new();
    for tensor in &tensors {
        weight_map.insert(tensor.name.clone(), serde_json::json!("layers.safetensors"));
    }
    std::fs::write(
        model_dir.join("model.safetensors.index.json"),
        serde_json::json!({ "weight_map": weight_map }).to_string(),
    )
    .unwrap();

    // WordLevel tokenizer over the 32-id model vocabulary; ids 5-7 are plain
    // letters so prompt/decode text stays inside the model output space.
    let tokenizer = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {
                "<|im_start|>": 0,
                "<|im_end|>": 1,
                "<|image_pad|>": 2,
                "<|video_pad|>": 3,
                "<|endoftext|>": 4,
                "a": 5,
                "b": 6,
                "c": 7
            },
            "unk_token": "a"
        }
    });
    std::fs::write(model_dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    let tokenizer_config = serde_json::json!({
        "added_tokens_decoder": {
            "0": { "content": "<|im_start|>", "special": true },
            "1": { "content": "<|im_end|>", "special": true },
            "2": { "content": "<|image_pad|>", "special": true },
            "3": { "content": "<|video_pad|>", "special": true },
            "4": { "content": "<|endoftext|>", "special": true }
        },
        "eos_token": "<|endoftext|>",
        "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    });
    std::fs::write(
        model_dir.join("tokenizer_config.json"),
        tokenizer_config.to_string(),
    )
    .unwrap();

    // The downloader bundle-completeness gate requires every metadata file of
    // the published bundle to exist, even when unused by the loader.
    std::fs::write(model_dir.join("generation_config.json"), "{}").unwrap();
    std::fs::write(
        model_dir.join("chat_template.jinja"),
        "{% for message in messages %}{{ message['content'] }}{% endfor %}",
    )
    .unwrap();
    std::fs::write(model_dir.join("vocab.json"), "{}").unwrap();
    std::fs::write(model_dir.join("merges.txt"), "").unwrap();
    std::fs::write(model_dir.join("preprocessor_config.json"), "{}").unwrap();
    std::fs::write(model_dir.join("video_preprocessor_config.json"), "{}").unwrap();

    let manifest = ArtifactManifest {
        schema_version: 1,
        variant: ModelVariant::Qwen36Moe35BA3BFp8,
        repo_id: ModelVariant::Qwen36Moe35BA3BFp8.repo_id().into(),
        revision: ModelVariant::Qwen36Moe35BA3BFp8
            .artifact_revision()
            .expect("Qwen3.6 MoE artifact revision is catalog-pinned")
            .into(),
        files: vec![
            "chat_template.jinja".into(),
            "config.json".into(),
            "generation_config.json".into(),
            "layers.safetensors".into(),
            "merges.txt".into(),
            "model.safetensors.index.json".into(),
            "preprocessor_config.json".into(),
            "tokenizer.json".into(),
            "tokenizer_config.json".into(),
            "video_preprocessor_config.json".into(),
            "vocab.json".into(),
        ],
    };
    std::fs::write(
        model_dir.join("izwi-artifact.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    model_dir
}
