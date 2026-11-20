//! Ming's checkpoint file and its published tensors.

use std::path::Path;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use super::config::{
    MingFlowConfig, MingStackConfig, MingTtsConfig, FRAME_SAMPLES, ISTFT_BINS, ISTFT_SIZE,
    LATENT_DIM, WEIGHTS_FILE,
};
use super::speaker::SPEAKER_EMBEDDING_DIM;
use crate::error::{Error, Result};
use crate::models::shared::weights::staged::metal_staged_safetensors;

/// The checkpoint memory-mapped as a `VarBuilder` producing `dtype` on
/// `device`; on Metal a tensor is cast on the host and uploaded at its
/// final size, so a half-precision load never holds a second copy.
pub(crate) fn mmap_checkpoint(
    model_dir: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let path = model_dir.join(WEIGHTS_FILE);
    if !path.is_file() {
        return Err(Error::ModelLoadError(format!(
            "Ming weights {} are missing",
            path.display()
        )));
    }
    // SAFETY: the file is a read-only model artifact izwi verified on
    // download and does not modify while loaded.
    unsafe { metal_staged_safetensors(&[path], dtype, device) }
        .map_err(|error| Error::ModelLoadError(format!("Failed to map Ming weights: {error}")))
}

fn stack_tensors(prefix: &str, cfg: &MingStackConfig, out: &mut Vec<(String, Vec<usize>)>) {
    let hidden = cfg.hidden_size;
    out.push((format!("{prefix}norm.weight"), vec![hidden]));
    for layer in 0..cfg.num_hidden_layers {
        let name = |suffix: &str| format!("{prefix}layers.{layer}.{suffix}");
        out.extend([
            (name("input_layernorm.weight"), vec![hidden]),
            (name("post_attention_layernorm.weight"), vec![hidden]),
            (name("self_attn.q_proj.weight"), vec![cfg.q_out(), hidden]),
            (name("self_attn.q_proj.bias"), vec![cfg.q_out()]),
            (name("self_attn.k_proj.weight"), vec![cfg.kv_out(), hidden]),
            (name("self_attn.k_proj.bias"), vec![cfg.kv_out()]),
            (name("self_attn.v_proj.weight"), vec![cfg.kv_out(), hidden]),
            (name("self_attn.v_proj.bias"), vec![cfg.kv_out()]),
            (name("self_attn.o_proj.weight"), vec![hidden, cfg.q_out()]),
            (
                name("mlp.gate_proj.weight"),
                vec![cfg.intermediate_size, hidden],
            ),
            (
                name("mlp.up_proj.weight"),
                vec![cfg.intermediate_size, hidden],
            ),
            (
                name("mlp.down_proj.weight"),
                vec![hidden, cfg.intermediate_size],
            ),
        ]);
    }
}

fn flow_tensors(
    prefix: &str,
    cfg: &MingFlowConfig,
    output: usize,
    out: &mut Vec<(String, Vec<usize>)>,
) {
    let hidden = cfg.hidden_size;
    let mut linear = |name: String, input: usize, output: usize| {
        out.push((format!("{name}.weight"), vec![output, input]));
        out.push((format!("{name}.bias"), vec![output]));
    };
    for block in 0..cfg.depth {
        let name = |suffix: &str| format!("{prefix}blocks.{block}.{suffix}");
        for part in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"] {
            linear(name(part), hidden, hidden);
        }
        linear(name("mlp.ff.0.0"), hidden, cfg.mlp_size);
        linear(name("mlp.ff.2"), cfg.mlp_size, hidden);
    }
    linear(format!("{prefix}final_layer.linear"), hidden, output);
    linear(format!("{prefix}x_embedder"), LATENT_DIM, hidden);
    for block in 0..cfg.depth {
        out.push((format!("{prefix}blocks.{block}.norm1.weight"), vec![hidden]));
        out.push((format!("{prefix}blocks.{block}.norm2.weight"), vec![hidden]));
    }
    out.push((
        format!("{prefix}final_layer.norm_final.weight"),
        vec![hidden],
    ));
    out.push((
        format!("{prefix}rotary_embed.inv_freq"),
        vec![cfg.head_dim / 2],
    ));
}

/// Every tensor of the published checkpoint (name, shape), sorted by name.
/// All are BF16 but the two `rotary_embed.inv_freq` buffers (F32).
pub fn published_tensors(cfg: &MingTtsConfig) -> Vec<(String, Vec<usize>)> {
    let hidden = cfg.llm.hidden_size;
    let vae = cfg.vae_stack.hidden_size;
    let mut tensors = vec![
        (
            "model.model.embed_tokens.weight".to_string(),
            vec![cfg.vocab_size, hidden],
        ),
        ("stop_head.weight".to_string(), vec![2, hidden]),
        ("stop_head.bias".to_string(), vec![2]),
        (
            "spk_head.weight".to_string(),
            vec![hidden, SPEAKER_EMBEDDING_DIM],
        ),
        ("spk_head.bias".to_string(), vec![hidden]),
        (
            "linear_proj_audio.word_embedder.weight".to_string(),
            vec![1, cfg.aggregator.hidden_size],
        ),
        // The decoder.
        (
            "audio.decoder.fc1.weight".to_string(),
            vec![vae, LATENT_DIM],
        ),
        ("audio.decoder.fc1.bias".to_string(), vec![vae]),
        (
            "audio.decoder.head.out.weight".to_string(),
            vec![2 * ISTFT_BINS, vae],
        ),
        (
            "audio.decoder.head.out.bias".to_string(),
            vec![2 * ISTFT_BINS],
        ),
        // The encoder.
        (
            "audio.encoder.fc1.weight".to_string(),
            vec![vae, FRAME_SAMPLES],
        ),
        ("audio.encoder.fc2.weight".to_string(), vec![vae, vae]),
        ("audio.encoder.fc2.bias".to_string(), vec![vae]),
        (
            "audio.encoder.fc3.weight".to_string(),
            vec![2 * LATENT_DIM, vae],
        ),
        ("audio.encoder.fc3.bias".to_string(), vec![2 * LATENT_DIM]),
        ("audio.encoder.cls_embed".to_string(), vec![1, 1, vae]),
    ];
    tensors.extend(unused_tensors(cfg));
    stack_tensors("model.model.", &cfg.llm, &mut tensors);
    stack_tensors("audio.decoder.decoder.", &cfg.vae_stack, &mut tensors);
    stack_tensors("audio.encoder.encoder.", &cfg.vae_stack, &mut tensors);
    stack_tensors(
        "audio.encoder.aggregator.",
        &cfg.vae_group_stack,
        &mut tensors,
    );
    flow_tensors("linear_proj_audio.", &cfg.aggregator, hidden, &mut tensors);
    flow_tensors("flowloss.cfm.model.", &cfg.flow, LATENT_DIM, &mut tensors);
    let time = "flowloss.cfm.model.t_embedder.time_mlp";
    let flow = cfg.flow.hidden_size;
    tensors.extend([
        (format!("{time}.0.weight"), vec![flow, 256]),
        (format!("{time}.0.bias"), vec![flow]),
        (format!("{time}.2.weight"), vec![flow, flow]),
        (format!("{time}.2.bias"), vec![flow]),
        (
            "flowloss.cfm.model.c_embedder.cond_embedder.weight".to_string(),
            vec![flow, hidden],
        ),
        (
            "flowloss.cfm.model.c_embedder.cond_embedder.bias".to_string(),
            vec![flow],
        ),
    ]);
    tensors.sort();
    tensors
}

/// Tensors the checkpoint carries and inference never reads: the VAE
/// stacks' one-row token tables (they take embeddings), a layer norm the
/// encoder's forward skips, and the stored inverse-STFT window (rebuilt in
/// F32; the stored one is BF16). The rotary buffers are rebuilt too.
pub fn unused_tensors(cfg: &MingTtsConfig) -> Vec<(String, Vec<usize>)> {
    let vae = cfg.vae_stack.hidden_size;
    vec![
        (
            "audio.decoder.decoder.embed_tokens.weight".to_string(),
            vec![1, vae],
        ),
        (
            "audio.encoder.encoder.embed_tokens.weight".to_string(),
            vec![1, vae],
        ),
        (
            "audio.encoder.aggregator.embed_tokens.weight".to_string(),
            vec![1, vae],
        ),
        ("audio.encoder.norm.weight".to_string(), vec![vae]),
        ("audio.encoder.norm.bias".to_string(), vec![vae]),
        (
            "audio.decoder.head.istft.window".to_string(),
            vec![ISTFT_SIZE],
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The census of the published safetensors header at the pin (read over
    /// HTTP range on 2026-11-19): 1,180 tensors.
    const CENSUS: &str = include_str!("../../../../tests/fixtures/ming_tts/tensor_census.txt");

    #[test]
    fn published_tensor_list_matches_the_census() {
        let census = CENSUS
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| {
                let mut fields = line.split('\t');
                let name = fields.next().unwrap().to_string();
                let dtype = fields.next().unwrap().to_string();
                let shape = fields
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|dim| dim.parse::<usize>().unwrap())
                    .collect::<Vec<_>>();
                (name, dtype, shape)
            })
            .collect::<Vec<_>>();
        assert_eq!(census.len(), 1_180);
        let expected = published_tensors(&MingTtsConfig::pinned());
        assert_eq!(expected.len(), census.len());
        for ((name, shape), (census_name, dtype, census_shape)) in expected.iter().zip(&census) {
            assert_eq!(name, census_name);
            assert_eq!(shape, census_shape, "{name}");
            let want = if name.ends_with("rotary_embed.inv_freq") {
                "F32"
            } else {
                "BF16"
            };
            assert_eq!(dtype, want, "{name}");
        }
        // The language model ties its output head: none is stored.
        assert!(!census.iter().any(|(name, _, _)| name.contains("lm_head")));
    }

    #[test]
    fn unused_tensors_are_part_of_the_checkpoint() {
        let cfg = MingTtsConfig::pinned();
        let published = published_tensors(&cfg);
        for tensor in unused_tensors(&cfg) {
            assert!(published.contains(&tensor), "{}", tensor.0);
        }
    }
}
