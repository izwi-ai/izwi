//! Kyutai's Mimi neural audio codec (<https://huggingface.co/kyutai/mimi>,
//! CC-BY-4.0), ported natively from transformers' `MimiModel`: 24 kHz mono,
//! 12.5 frames per second, a split RVQ of 1 semantic + 31 acoustic codebooks
//! of 2,048 entries.
//!
//! The decoder streams: every layer is causal, so [`MimiDecoder::decode_stream`]
//! turns a few frames at a time into exactly the samples a whole-sequence
//! decode gives them, carrying each layer's left context in a
//! [`MimiDecoderState`]. Its transformer attends over a causal window of
//! `sliding_window` (250) steps, as Kyutai's streaming Mimi does. The
//! encoder runs offline over a whole clip (voice-clone references).
//!
//! Weights load in F32 from the transformers-format `model.safetensors`.

mod config;
mod conv;
mod decoder;
mod encoder;
mod quantizer;
mod transformer;

use std::path::Path;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;

pub use config::MimiConfig;
pub use decoder::{MimiDecoder, MimiDecoderState};
pub use encoder::MimiEncoder;

use crate::error::{Error, Result};

/// The weights file beside `config.json`.
pub const MIMI_WEIGHTS_FILE: &str = "model.safetensors";
pub const MIMI_CONFIG_FILE: &str = "config.json";

fn var_builder(dir: &Path, device: &Device) -> Result<VarBuilder<'static>> {
    let path = dir.join(MIMI_WEIGHTS_FILE);
    if !path.is_file() {
        return Err(Error::ModelLoadError(format!(
            "Mimi weights {} are missing",
            path.display()
        )));
    }
    // SAFETY: a read-only model artifact izwi verified on download and does
    // not modify while mapped.
    unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device) }
        .map_err(|error| Error::ModelLoadError(format!("Failed to map Mimi weights: {error}")))
}

/// Load the decoder (quantizer tables, upsample, transformer, SEANet) from
/// the Mimi directory `dir`.
pub fn load_decoder(dir: &Path, device: &Device) -> Result<MimiDecoder> {
    let cfg = MimiConfig::load(&dir.join(MIMI_CONFIG_FILE))?;
    MimiDecoder::load(&cfg, var_builder(dir, device)?)
}

/// Load the encoder (SEANet, transformer, downsample, quantizer) from the
/// Mimi directory `dir`.
pub fn load_encoder(dir: &Path, device: &Device) -> Result<MimiEncoder> {
    let cfg = MimiConfig::load(&dir.join(MIMI_CONFIG_FILE))?;
    MimiEncoder::load(&cfg, var_builder(dir, device)?)
}

#[cfg(test)]
mod tests {
    use candle_core::{Tensor, Var};
    use candle_nn::VarMap;

    use super::*;

    /// A tiny Mimi: hop 10 at 100 Hz (10 steps/s, 5 frames/s), 4 codebooks
    /// of 16, a 3-step window so short clips exercise the window.
    fn tiny_config() -> MimiConfig {
        serde_json::from_value(serde_json::json!({
            "audio_channels": 1, "codebook_dim": 4, "codebook_size": 16, "compress": 2,
            "dilation_growth_rate": 2, "frame_rate": 5.0, "head_dim": 4, "hidden_act": "gelu",
            "hidden_size": 8, "intermediate_size": 16, "kernel_size": 7, "last_kernel_size": 3,
            "norm_eps": 1e-5, "num_attention_heads": 2, "num_filters": 2,
            "num_hidden_layers": 2, "num_key_value_heads": 2, "num_quantizers": 4,
            "num_residual_layers": 1, "num_semantic_quantizers": 1, "pad_mode": "constant",
            "residual_kernel_size": 3, "rope_theta": 10000.0, "sampling_rate": 100,
            "sliding_window": 3, "trim_right_ratio": 1.0, "upsample_groups": 8,
            "upsampling_ratios": [2, 5], "use_causal_conv": true, "use_conv_shortcut": false,
            "vector_quantization_hidden_dimension": 4
        }))
        .unwrap()
    }

    /// Random weights for every tensor `build` reads (cluster usage kept
    /// positive), then `build` again over them.
    fn random<T>(build: impl Fn(VarBuilder<'static>) -> Result<T>) -> T {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        build(vb).unwrap();
        let mut seed = 1u64;
        for (name, var) in varmap.data().lock().unwrap().iter() {
            let shape = var.shape().clone();
            let count = shape.elem_count();
            let values = (0..count)
                .map(|_| {
                    seed = crate::models::shared::noise::mix(seed.wrapping_add(1));
                    let unit = (seed >> 40) as f32 / (1u64 << 24) as f32;
                    if name.ends_with("cluster_usage") {
                        0.5 + unit
                    } else {
                        (unit - 0.5) * 0.6
                    }
                })
                .collect::<Vec<_>>();
            set(var, Tensor::from_vec(values, shape, &Device::Cpu).unwrap());
        }
        build(VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu)).unwrap()
    }

    fn set(var: &Var, value: Tensor) {
        var.set(&value).unwrap();
    }

    fn codes(frames: usize, books: usize, size: u32) -> Vec<u32> {
        (0..frames * books)
            .map(|i| (crate::models::shared::noise::mix(i as u64 + 7) % size as u64) as u32)
            .collect()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| {
                let d = (x - y).abs();
                if d.is_finite() {
                    d
                } else {
                    f32::INFINITY
                }
            })
            .fold(0.0, f32::max)
    }

    #[test]
    fn streamed_frames_equal_the_whole_decode() {
        let cfg = tiny_config();
        cfg.validate().unwrap();
        let decoder = random(|vb| MimiDecoder::load(&cfg, vb));
        let codes = codes(9, 4, 16);
        let whole = decoder.decode(&codes).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(whole.len(), 9 * cfg.samples_per_frame());
        let scale = whole.iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!(scale > 1e-4, "decoder output is degenerate");
        for chunks in [vec![1; 9], vec![2, 3, 1, 3], vec![9]] {
            let mut state = decoder.new_state().unwrap();
            let mut streamed = Vec::new();
            let mut frame = 0;
            for size in chunks {
                let chunk = &codes[frame * 4..(frame + size) * 4];
                streamed.extend(
                    decoder
                        .decode_stream(&mut state, chunk)
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap(),
                );
                frame += size;
            }
            assert_eq!(state.frames(), 9);
            let diff = max_abs_diff(&streamed, &whole);
            assert!(diff <= 1e-5 * scale.max(1.0), "stream differs by {diff}");
        }
    }

    #[test]
    fn a_stream_depends_only_on_its_own_frames() {
        let cfg = tiny_config();
        let decoder = random(|vb| MimiDecoder::load(&cfg, vb));
        let codes = codes(3, 4, 16);
        let mut a = decoder.new_state().unwrap();
        let mut b = decoder.new_state().unwrap();
        let first = decoder.decode_stream(&mut a, &codes).unwrap();
        // Another stream in between must not disturb `a`'s successor.
        decoder.decode_stream(&mut b, &codes[..4]).unwrap();
        let again = decoder.decode_stream(&mut a.clone(), &codes[..4]).unwrap();
        let fresh = decoder.decode_stream(&mut a, &codes[..4]).unwrap();
        assert_eq!(
            again.to_vec1::<f32>().unwrap(),
            fresh.to_vec1::<f32>().unwrap()
        );
        assert_eq!(first.dim(0).unwrap(), 3 * cfg.samples_per_frame());
    }

    #[test]
    fn windowed_encoder_stack_equals_one_pass() {
        let cfg = tiny_config();
        let encoder = random(|vb| MimiEncoder::load(&cfg, vb));
        // 3,000 samples = 300 steps, past the 128-step window.
        let samples = (0..3000)
            .map(|i| ((i as f32) * 0.013).sin() * 0.5)
            .collect::<Vec<_>>();
        let x = Tensor::from_slice(&samples, (1, 1, samples.len()), &Device::Cpu).unwrap();
        let whole = encoder.seanet_for_test(&x).unwrap();
        let windowed = encoder.seanet_windowed_for_test(&x).unwrap();
        let diff = max_abs_diff(
            &whole.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            &windowed.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        assert!(diff < 1e-5, "windowed encoder differs by {diff}");
        let codes = encoder.encode(&samples).unwrap();
        // 300 steps → 150 frames of 4 codes.
        assert_eq!(codes.len(), 150 * 4);
        assert!(codes.iter().all(|&code| code < 16));
        assert_eq!(codes, encoder.encode(&samples).unwrap());
    }
}
