//! Shared fixture helpers for worker process tests: a tiny genuine LFM2 GGUF
//! that exercises the real RuntimeService loader, scheduler, managed memory,
//! and decode path without downloading the 1.2B artifact.

use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use izwi_core::{artifacts::ArtifactManifest, ModelVariant};
use std::path::{Path, PathBuf};

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
    let mut metadata = vec![
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
    add("blk.0.ffn_gate_exps.weight".into(), &[n_experts, n_ff, hidden]);
    add("blk.0.ffn_up_exps.weight".into(), &[n_experts, n_ff, hidden]);
    add("blk.0.ffn_down_exps.weight".into(), &[n_experts, hidden, n_ff]);
    let mut file =
        std::fs::File::create(model_dir.join("Qwen3-30B-A3B-Q4_K_M.gguf")).unwrap();
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
