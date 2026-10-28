//! The LFM3 trunk on the LFM2 backbone against an independent F64 forward
//! pass written from HF `modeling_lfm3.py` (LiquidAI/LFM3-1.7B @ 40ebe0aa).

use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use gguf_file::Value;

use crate::models::architectures::lfm2::backbone::QuantizedLfm2Backbone;
use crate::models::shared::weights::gguf::GgufLoader;

use super::parse_lfm3_backbone_config;

const HIDDEN: usize = 8;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const HEAD_DIM: usize = HIDDEN / HEADS;
const FFN: usize = 12;
const VOCAB: usize = 11;
const L_CACHE: usize = 7;
const WINDOW: usize = 3;
const EPS: f64 = 1e-5;
/// conv, conv, global attention, conv, sliding attention.
const KV_PER_LAYER: [usize; 5] = [0, 0, KV_HEADS, 0, KV_HEADS];
const SLIDING: [bool; 5] = [false, false, false, false, true];
const ROPE_BASE: f64 = 1_000_000.0;
const ROPE_BASE_SWA: f64 = 10_000.0;
/// The global layers rotate their first frequency pair and leave the second.
const ROPE_FACTORS: [f32; HEAD_DIM / 2] = [1.0, 1e30];

/// A tiny `lfm3` GGUF with every LFM3 feature: the per-head gate, sandwich
/// norms, a width-7 ShortConv, global and sliding attention with their own
/// RoPE bases, and `rope_freqs` partial rotation. Weights are xorshift noise
/// with norm gains away from 1, so each feature changes the output.
pub(super) fn write_tiny_lfm3_gguf(path: &std::path::Path, seed: u64) {
    let metadata = vec![
        ("general.architecture", Value::String("lfm3".into())),
        ("lfm3.block_count", Value::U32(5)),
        ("lfm3.context_length", Value::U32(256)),
        ("lfm3.embedding_length", Value::U32(HIDDEN as u32)),
        ("lfm3.feed_forward_length", Value::U32(FFN as u32)),
        ("lfm3.attention.head_count", Value::U32(HEADS as u32)),
        (
            "lfm3.attention.head_count_kv",
            Value::Array(
                KV_PER_LAYER
                    .iter()
                    .map(|kv| Value::U32(*kv as u32))
                    .collect(),
            ),
        ),
        (
            "lfm3.attention.layer_norm_rms_epsilon",
            Value::F32(EPS as f32),
        ),
        ("lfm3.attention.key_length", Value::U32(HEAD_DIM as u32)),
        ("lfm3.rope.dimension_count", Value::U32(HEAD_DIM as u32)),
        ("lfm3.rope.freq_base", Value::F32(ROPE_BASE as f32)),
        ("lfm3.rope.freq_base_swa", Value::F32(ROPE_BASE_SWA as f32)),
        ("lfm3.attention.sliding_window", Value::U32(WINDOW as u32)),
        (
            "lfm3.attention.sliding_window_pattern",
            Value::Array(SLIDING.iter().map(|flag| Value::Bool(*flag)).collect()),
        ),
        ("lfm3.shortconv.l_cache", Value::U32(L_CACHE as u32)),
    ];
    let mut weights = Vec::new();
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    let mut add = |name: String, shape: &[usize], offset: f32, scale: f32| {
        let values = (0..shape.iter().product::<usize>())
            .map(|_| offset + scale * next())
            .collect::<Vec<_>>();
        let tensor = Tensor::from_vec(values, shape, &Device::Cpu).unwrap();
        weights.push((name, QTensor::quantize(&tensor, GgmlDType::F32).unwrap()));
    };
    add("token_embd.weight".into(), &[VOCAB, HIDDEN], 0.0, 2.0);
    add("output_norm.weight".into(), &[HIDDEN], 1.0, 1.0);
    for (layer, kv) in KV_PER_LAYER.iter().enumerate() {
        for name in [
            "attn_norm",
            "ffn_norm",
            "post_attention_norm",
            "post_ffw_norm",
        ] {
            add(format!("blk.{layer}.{name}.weight"), &[HIDDEN], 1.0, 1.0);
        }
        add(
            format!("blk.{layer}.ffn_gate.weight"),
            &[FFN, HIDDEN],
            0.0,
            1.0,
        );
        add(
            format!("blk.{layer}.ffn_up.weight"),
            &[FFN, HIDDEN],
            0.0,
            1.0,
        );
        add(
            format!("blk.{layer}.ffn_down.weight"),
            &[HIDDEN, FFN],
            0.0,
            1.0,
        );
        if *kv > 0 {
            for name in ["attn_q_norm", "attn_k_norm"] {
                add(format!("blk.{layer}.{name}.weight"), &[HEAD_DIM], 1.0, 1.0);
            }
            add(
                format!("blk.{layer}.attn_q.weight"),
                &[HIDDEN, HIDDEN],
                0.0,
                1.0,
            );
            add(
                format!("blk.{layer}.attn_k.weight"),
                &[KV_HEADS * HEAD_DIM, HIDDEN],
                0.0,
                1.0,
            );
            add(
                format!("blk.{layer}.attn_v.weight"),
                &[KV_HEADS * HEAD_DIM, HIDDEN],
                0.0,
                1.0,
            );
            add(
                format!("blk.{layer}.attn_output.weight"),
                &[HIDDEN, HIDDEN],
                0.0,
                1.0,
            );
            add(
                format!("blk.{layer}.attn_gate.weight"),
                &[HEADS, HIDDEN],
                0.0,
                4.0,
            );
        } else {
            add(
                format!("blk.{layer}.shortconv.in_proj.weight"),
                &[3 * HIDDEN, HIDDEN],
                0.0,
                1.0,
            );
            add(
                format!("blk.{layer}.shortconv.out_proj.weight"),
                &[HIDDEN, HIDDEN],
                0.0,
                1.0,
            );
            add(
                format!("blk.{layer}.shortconv.conv.weight"),
                &[HIDDEN, L_CACHE],
                0.0,
                1.0,
            );
        }
    }
    let factors = Tensor::new(&ROPE_FACTORS, &Device::Cpu).unwrap();
    weights.push((
        "rope_freqs.weight".into(),
        QTensor::quantize(&factors, GgmlDType::F32).unwrap(),
    ));
    let mut file = std::fs::File::create(path).unwrap();
    gguf_file::write(
        &mut file,
        &metadata.iter().map(|(k, v)| (*k, v)).collect::<Vec<_>>(),
        &weights
            .iter()
            .map(|(n, t)| (n.as_str(), t))
            .collect::<Vec<_>>(),
    )
    .unwrap();
}

/// A tiny checkpoint on disk; the loader reopens it for every tensor, so it
/// lives as long as this guard.
pub(super) struct TinyGguf {
    pub(super) path: std::path::PathBuf,
    pub(super) loader: GgufLoader,
}

impl Drop for TinyGguf {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

pub(super) fn tiny_gguf(seed: u64) -> TinyGguf {
    let path = std::env::temp_dir().join(format!("izwi-lfm3-tiny-{}.gguf", uuid::Uuid::new_v4()));
    write_tiny_lfm3_gguf(&path, seed);
    let loader = GgufLoader::from_path(&path).unwrap();
    TinyGguf { path, loader }
}

#[test]
fn parses_every_lfm3_feature_from_the_gguf() {
    let tiny = tiny_gguf(1);
    let config = parse_lfm3_backbone_config(&tiny.loader).unwrap();
    let features = config.lfm3.as_ref().unwrap();
    assert_eq!(config.attention_sliding_window, None);
    assert_eq!(config.shortconv_l_cache, L_CACHE);
    assert_eq!(
        features.layer_sliding_windows,
        vec![None, None, None, None, Some(WINDOW)]
    );
    assert_eq!(features.rope_freq_base_swa, Some(ROPE_BASE_SWA));
    assert_eq!(
        features.rope_freq_factors.as_deref(),
        Some(&[1.0, f64::from(1e30_f32)][..])
    );
    assert!(features.attention_output_gate && features.sandwich_norm);
    assert!(config.layer_uses_swa_rope(4) && !config.layer_uses_swa_rope(2));
}

#[test]
fn refuses_rope_scaling_other_than_partial_rotation() {
    let path = std::env::temp_dir().join(format!("izwi-lfm3-bad-{}.gguf", uuid::Uuid::new_v4()));
    write_tiny_lfm3_gguf(&path, 1);
    // Rewrite `rope_freqs` with a linear-scaling factor.
    let loader = GgufLoader::from_path(&path).unwrap();
    let mut file = std::fs::File::open(&path).unwrap();
    let content = gguf_file::Content::read(&mut file).unwrap();
    let mut tensors = content
        .tensor_infos
        .keys()
        .map(|name| {
            let tensor = if name == "rope_freqs.weight" {
                QTensor::quantize(
                    &Tensor::new(&[1.0f32, 4.0], &Device::Cpu).unwrap(),
                    GgmlDType::F32,
                )
                .unwrap()
            } else {
                loader.load_qtensor(name, &Device::Cpu).unwrap()
            };
            (name.clone(), tensor)
        })
        .collect::<Vec<_>>();
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    let metadata = content.metadata.clone();
    drop(file);
    let mut out = std::fs::File::create(&path).unwrap();
    gguf_file::write(
        &mut out,
        &metadata
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>(),
        &tensors
            .iter()
            .map(|(n, t)| (n.as_str(), t))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    drop(out);
    let error = parse_lfm3_backbone_config(&GgufLoader::from_path(&path).unwrap()).unwrap_err();
    std::fs::remove_file(&path).unwrap();
    assert!(error.to_string().contains("partial rotation"), "{error}");
}

/// Dense F64 weights of the tiny checkpoint, keyed by GGUF name.
struct Reference {
    weights: std::collections::HashMap<String, (Vec<usize>, Vec<f64>)>,
}

impl Reference {
    fn new(loader: &GgufLoader) -> Self {
        let weights = loader
            .tensor_names()
            .into_iter()
            .map(|name| {
                let tensor = loader.load_tensor(&name, DType::F64, &Device::Cpu).unwrap();
                let shape = tensor.dims().to_vec();
                let values = tensor.flatten_all().unwrap().to_vec1::<f64>().unwrap();
                (name, (shape, values))
            })
            .collect();
        Self { weights }
    }

    fn w(&self, name: &str) -> &[f64] {
        &self.weights[name].1
    }

    /// `[rows, cols] x input[cols]`.
    fn linear(&self, name: &str, input: &[f64]) -> Vec<f64> {
        let (shape, values) = &self.weights[name];
        let (rows, cols) = (shape[0], shape[1]);
        assert_eq!(cols, input.len(), "{name}");
        (0..rows)
            .map(|row| {
                (0..cols)
                    .map(|col| values[row * cols + col] * input[col])
                    .sum()
            })
            .collect()
    }

    fn rms_norm(&self, name: &str, input: &[f64]) -> Vec<f64> {
        let mean = input.iter().map(|x| x * x).sum::<f64>() / input.len() as f64;
        let scale = 1.0 / (mean + EPS).sqrt();
        input
            .iter()
            .zip(self.w(name))
            .map(|(x, w)| w * x * scale)
            .collect()
    }

    /// NEOX rotation of one head at `position` with HF's zero-tail partial
    /// RoPE: the unrotated half has an inverse frequency of zero.
    fn rope(head: &mut [f64], position: usize, sliding: bool) {
        let half = HEAD_DIM / 2;
        for pair in 0..half {
            let base = if sliding { ROPE_BASE_SWA } else { ROPE_BASE };
            let rotated = sliding || ROPE_FACTORS[pair] == 1.0;
            let inv_freq = if rotated {
                1.0 / base.powf((2 * pair) as f64 / HEAD_DIM as f64)
            } else {
                0.0
            };
            let (sin, cos) = (position as f64 * inv_freq).sin_cos();
            let (x1, x2) = (head[pair], head[pair + half]);
            head[pair] = x1 * cos - x2 * sin;
            head[pair + half] = x2 * cos + x1 * sin;
        }
    }

    /// The final-norm hidden states of every position of `tokens`.
    fn forward(&self, tokens: &[u32]) -> Vec<Vec<f64>> {
        let embed = self.w("token_embd.weight");
        let mut hidden = tokens
            .iter()
            .map(|&token| embed[token as usize * HIDDEN..][..HIDDEN].to_vec())
            .collect::<Vec<_>>();
        for (layer, kv) in KV_PER_LAYER.iter().enumerate() {
            let p = |name: &str| format!("blk.{layer}.{name}.weight");
            let normed = hidden
                .iter()
                .map(|h| self.rms_norm(&p("attn_norm"), h))
                .collect::<Vec<_>>();
            let operator = if *kv > 0 {
                self.attention(layer, &normed)
            } else {
                self.short_conv(layer, &normed)
            };
            for (h, o) in hidden.iter_mut().zip(operator) {
                let o = self.rms_norm(&p("post_attention_norm"), &o);
                h.iter_mut().zip(o).for_each(|(h, o)| *h += o);
            }
            for h in hidden.iter_mut() {
                let x = self.rms_norm(&p("ffn_norm"), h);
                let gate = self.linear(&p("ffn_gate"), &x);
                let up = self.linear(&p("ffn_up"), &x);
                let act = gate
                    .iter()
                    .zip(&up)
                    .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                    .collect::<Vec<_>>();
                let ffn = self.rms_norm(&p("post_ffw_norm"), &self.linear(&p("ffn_down"), &act));
                h.iter_mut().zip(ffn).for_each(|(h, f)| *h += f);
            }
        }
        hidden
            .iter()
            .map(|h| self.rms_norm("output_norm.weight", h))
            .collect()
    }

    fn short_conv(&self, layer: usize, normed: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let p = |name: &str| format!("blk.{layer}.shortconv.{name}.weight");
        let projected = normed
            .iter()
            .map(|x| self.linear(&p("in_proj"), x))
            .collect::<Vec<_>>();
        let bx = projected
            .iter()
            .map(|bcx| {
                (0..HIDDEN)
                    .map(|c| bcx[c] * bcx[2 * HIDDEN + c])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let conv = self.w(&p("conv"));
        (0..normed.len())
            .map(|t| {
                let y = (0..HIDDEN)
                    .map(|c| {
                        let mut sum = 0.0;
                        for k in 0..L_CACHE {
                            // `F.conv1d(padding=L-1)[..., :T]`: tap k reads t - (L-1) + k.
                            if let Some(source) = (t + k).checked_sub(L_CACHE - 1) {
                                sum += conv[c * L_CACHE + k] * bx[source][c];
                            }
                        }
                        projected[t][HIDDEN + c] * sum
                    })
                    .collect::<Vec<_>>();
                self.linear(&p("out_proj"), &y)
            })
            .collect()
    }

    fn attention(&self, layer: usize, normed: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let p = |name: &str| format!("blk.{layer}.{name}.weight");
        let sliding = SLIDING[layer];
        let heads = |name: &str, norm: &str, count: usize| {
            normed
                .iter()
                .enumerate()
                .map(|(position, x)| {
                    let flat = self.linear(&p(name), x);
                    (0..count)
                        .map(|head| {
                            let mut head =
                                self.rms_norm(&p(norm), &flat[head * HEAD_DIM..][..HEAD_DIM]);
                            Self::rope(&mut head, position, sliding);
                            head
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        };
        let q = heads("attn_q", "attn_q_norm", HEADS);
        let k = heads("attn_k", "attn_k_norm", KV_HEADS);
        let v = normed
            .iter()
            .map(|x| self.linear(&p("attn_v"), x))
            .collect::<Vec<_>>();
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        (0..normed.len())
            .map(|t| {
                let gate = self.linear(&p("attn_gate"), &normed[t]);
                let mut out = Vec::with_capacity(HIDDEN);
                for head in 0..HEADS {
                    let kv_head = head / (HEADS / KV_HEADS);
                    let visible = (0..=t)
                        .filter(|s| !sliding || t - s < WINDOW)
                        .collect::<Vec<_>>();
                    let scores = visible
                        .iter()
                        .map(|&s| {
                            q[t][head]
                                .iter()
                                .zip(&k[s][kv_head])
                                .map(|(a, b)| a * b)
                                .sum::<f64>()
                                * scale
                        })
                        .collect::<Vec<_>>();
                    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exp = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
                    let total = exp.iter().sum::<f64>();
                    let sigmoid = 1.0 / (1.0 + (-gate[head]).exp());
                    for d in 0..HEAD_DIM {
                        let value = visible
                            .iter()
                            .zip(&exp)
                            .map(|(&s, e)| e / total * v[s][kv_head * HEAD_DIM + d])
                            .sum::<f64>();
                        out.push(value * sigmoid);
                    }
                }
                self.linear(&p("attn_output"), &out)
            })
            .collect()
    }
}

fn max_abs_diff(actual: &Tensor, expected: &[Vec<f64>]) -> f64 {
    let actual = actual
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F64)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f64>()
        .unwrap();
    let expected = expected.iter().flatten().copied().collect::<Vec<_>>();
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(&expected)
        .map(|(a, e)| {
            let diff = (a - e).abs();
            if diff.is_finite() {
                diff
            } else {
                f64::INFINITY
            }
        })
        .fold(0.0, f64::max)
}

/// Retained prefill in two spans (crossing the 3-token window and the
/// 7-tap ShortConv), then a batched decode step, match the reference.
fn assert_trunk_matches_reference(device: &Device) {
    let tiny = tiny_gguf(7);
    let loader = &tiny.loader;
    let config = parse_lfm3_backbone_config(loader).unwrap();
    let backbone = QuantizedLfm2Backbone::load(loader, config, device).unwrap();
    let reference = Reference::new(loader);
    let tokens: Vec<u32> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
    let expected = reference.forward(&tokens);

    let mut caches =
        crate::models::shared::attention::physical::test_physical_caches_for_model_layers_on(
            device,
            DType::F32,
            &[2, 4],
            KV_HEADS,
            HEAD_DIM,
            32,
            1,
        );
    let mut cache = caches.pop().unwrap();
    let mut shortconv = backbone.new_shortconv_state();
    let mut hidden = Vec::new();
    for (start, end) in [(0, 4), (4, 9)] {
        let input = Tensor::from_slice(&tokens[start..end], (1, end - start), device).unwrap();
        let embeds = backbone.embed_tokens(&input).unwrap();
        hidden.push(
            backbone
                .forward_embeds_retained(&embeds, start, &mut cache, &mut shortconv)
                .unwrap(),
        );
    }
    let prefill = Tensor::cat(&hidden, 1).unwrap();
    let error = max_abs_diff(&prefill, &expected[..9]);
    assert!(
        error < 1e-4,
        "prefill differs from the reference by {error}"
    );

    // The tenth token through the batched decode path, as chat decodes.
    let mut states = [&mut shortconv];
    let mut cache_refs = [&mut cache];
    let logits = backbone
        .forward_token_ids_retained_batch(&tokens[9..], &[9], &mut states, &mut cache_refs)
        .unwrap();
    let embed = reference.w("token_embd.weight");
    let expected_logits = (0..VOCAB)
        .map(|token| {
            expected[9]
                .iter()
                .zip(&embed[token * HIDDEN..][..HIDDEN])
                .map(|(h, e)| h * e)
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let error = max_abs_diff(&logits, &[expected_logits]);
    assert!(
        error < 1e-3,
        "decode logits differ from the reference by {error}"
    );
}

#[test]
fn lfm3_trunk_matches_the_reference_on_cpu() {
    assert_trunk_matches_reference(&Device::Cpu);
}

#[cfg(feature = "metal")]
#[test]
fn lfm3_trunk_matches_the_reference_on_metal() {
    if let Ok(device) = Device::new_metal(0) {
        assert_trunk_matches_reference(&device);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn lfm3_trunk_matches_the_reference_on_cuda() {
    if let Ok(device) = Device::new_cuda(0) {
        assert_trunk_matches_reference(&device);
    }
}
