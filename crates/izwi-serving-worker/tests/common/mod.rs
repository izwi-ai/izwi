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
