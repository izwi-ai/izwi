//! The block drafter against the reference fixture
//! (`tests/fixtures/inkling2_drafter_golden`): a tiny random drafter in the
//! published tensor layout and one draft round as a torch implementation
//! computes it in F32 (`scripts/fixtures/generate_inkling2_drafter_golden.py`).
//!
//! Errors are relative to the largest reference value of the tensor
//! compared.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;

use super::checkpoint::MTP_FILE;
use super::config::tests::fixture;
use super::config::CONFIG_FILE;
use super::drafter::{DenseDrafterContext, DrafterContext, Inkling2Drafter, Inkling2DrafterConfig};

const FIXTURE: &str = "inkling2_drafter_golden";
/// Host bound on every compared tensor. Both sides are F32; they differ in
/// the order of their sums.
const CPU_TOLERANCE: f32 = 2e-5;
/// Candle's Metal kernels use fast math and its products sum in another
/// order; everything is still F32.
#[cfg(feature = "metal")]
const METAL_TOLERANCE: f32 = 2e-4;

struct Golden {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
    integers: HashMap<String, Vec<u32>>,
}

impl Golden {
    fn load() -> Self {
        let bytes = std::fs::read(fixture(FIXTURE).join("golden.safetensors")).unwrap();
        let file = SafeTensors::deserialize(&bytes).unwrap();
        let mut golden = Self {
            tensors: HashMap::new(),
            integers: HashMap::new(),
        };
        for (name, view) in file.tensors() {
            match view.dtype() {
                safetensors::Dtype::F32 => {
                    let values = view
                        .data()
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|quad| f32::from_le_bytes(*quad))
                        .collect();
                    golden.tensors.insert(name, (view.shape().to_vec(), values));
                }
                safetensors::Dtype::I64 => {
                    let values = view
                        .data()
                        .as_chunks::<8>()
                        .0
                        .iter()
                        .map(|bytes| i64::from_le_bytes(*bytes) as u32)
                        .collect();
                    golden.integers.insert(name, values);
                }
                other => panic!("{name}: {other:?}"),
            }
        }
        golden
    }

    fn tensor(&self, name: &str, device: &Device) -> Tensor {
        let (shape, values) = &self.tensors[name];
        Tensor::from_slice(values, shape.as_slice(), device).unwrap()
    }

    /// The error of `got` against `name`, relative to the reference's
    /// largest value; shapes must agree.
    fn error(&self, name: &str, got: &Tensor) -> f32 {
        let (shape, want) = &self.tensors[name];
        assert_eq!(got.dims(), shape.as_slice(), "{name}");
        assert_eq!(got.dtype(), DType::F32, "{name}");
        let got = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let scale = want
            .iter()
            .fold(0f32, |worst, value| worst.max(value.abs()));
        assert!(scale > 0.0, "{name}");
        let worst = got.iter().zip(want).fold(0f32, |worst, (got, want)| {
            let error = (got - want).abs();
            if error.is_finite() {
                worst.max(error)
            } else {
                f32::INFINITY
            }
        });
        worst / scale
    }
}

fn tiny_drafter(device: &Device) -> Inkling2Drafter {
    Inkling2Drafter::load(&fixture(FIXTURE), device).unwrap()
}

/// The context of all the fixture's committed positions.
fn committed_context(drafter: &Inkling2Drafter, golden: &Golden) -> DenseDrafterContext {
    let mut context = DenseDrafterContext::new(drafter.config());
    context
        .extend(drafter, &golden.tensor("taps", drafter.device()))
        .unwrap();
    context
}

/// Runs the whole round on `device` and holds every stage to `tolerance`.
/// Returns the worst error seen.
fn round_matches_the_reference(device: &Device, tolerance: f32) -> f32 {
    let golden = Golden::load();
    let drafter = tiny_drafter(device);
    let layers = drafter.config().layers;
    let mut worst = 0f32;
    let mut hold = |name: &str, got: &Tensor| {
        let error = golden.error(name, got);
        assert!(error <= tolerance, "{name}: error {error:.3e}");
        worst = worst.max(error);
    };

    let taps = golden.tensor("taps", device);
    let committed = taps.dim(0).unwrap();
    let fused = drafter.fuse(&taps).unwrap();
    assert_eq!(fused.positions(), committed);
    hold("fused.head", &fused.head().unwrap());
    for layer in 0..layers {
        let states = fused.layer(layer).unwrap();
        hold(&format!("fused.layer.{layer}"), &states);
        let (keys, values) = drafter.context_kv(layer, &states, 0).unwrap();
        hold(&format!("context.{layer}.keys"), &keys);
        hold(&format!("context.{layer}.values"), &values);
    }
    assert!(fused.layer(layers).is_err());

    let context = committed_context(&drafter, &golden);
    let anchor = golden.tensor("anchor_embedding", device);
    let hidden = drafter.draft_block(&anchor, committed, &context).unwrap();
    hold("block_hidden", &hidden);
    let unembed = golden.tensor("unembed", device);
    let mask_rows = drafter.mask_rows(&hidden).unwrap();
    let base = drafter.base_logits(&mask_rows, &unembed).unwrap();
    hold("base_logits", &base);

    let f_rnn = context.last_head().unwrap();
    let anchor_token = golden.integers["anchor_token"][0];
    let (tokens, logits) = drafter
        .draft_greedy(&hidden, f_rnn, anchor_token, &unembed)
        .unwrap();
    assert_eq!(tokens, golden.integers["greedy.tokens"]);
    hold("greedy.logits", &logits);

    // A chain whose tokens come from outside (a sampler's, say): each
    // step's logits follow the token given, not the best one.
    let driven = &golden.integers["driven.tokens"];
    assert!(driven.iter().zip(&tokens).all(|(a, b)| a != b));
    let mut chain = drafter.markov_chain(f_rnn, anchor_token).unwrap();
    let mut rows = Vec::new();
    for (row, token) in driven.iter().enumerate() {
        let logits = chain
            .step(
                row,
                &mask_rows.get(row).unwrap(),
                &base.get(row).unwrap(),
                &unembed,
            )
            .unwrap();
        chain.advance(*token).unwrap();
        rows.push(logits);
    }
    hold("driven.logits", &Tensor::stack(&rows, 0).unwrap());
    worst
}

#[test]
fn a_draft_round_matches_the_reference() {
    let worst = round_matches_the_reference(&Device::Cpu, CPU_TOLERANCE);
    eprintln!("host drafter: worst relative error {worst:.3e}");
}

#[cfg(feature = "metal")]
#[test]
fn a_draft_round_matches_the_reference_on_metal() {
    let Some((device, _lease)) =
        crate::backends::metal_test_device_or_skip("a_draft_round_matches_the_reference_on_metal")
    else {
        return;
    };
    let worst = round_matches_the_reference(&device, METAL_TOLERANCE);
    eprintln!("Metal drafter: worst relative error {worst:.3e}");
}

/// Positions committed a few at a time (a prompt, then each verified round)
/// give the keys and values of one pass: a position's rotation is its own.
fn context_grows_in_pieces(device: &Device, tolerance: f32) {
    let golden = Golden::load();
    let drafter = tiny_drafter(device);
    let taps = golden.tensor("taps", device);
    let committed = taps.dim(0).unwrap();
    let whole = committed_context(&drafter, &golden);
    let mut pieces = DenseDrafterContext::new(drafter.config());
    let mut at = 0;
    for count in [13, 1, committed - 14] {
        pieces
            .extend(&drafter, &taps.narrow(0, at, count).unwrap())
            .unwrap();
        at += count;
        assert_eq!(pieces.committed(), at);
    }
    // The global layer still holds every position.
    let (keys, values) = pieces.span(1, 0).unwrap().unwrap();
    assert!(golden.error("context.1.keys", &keys) <= tolerance);
    assert!(golden.error("context.1.values", &values) <= tolerance);

    let anchor = golden.tensor("anchor_embedding", device);
    let hidden = drafter.draft_block(&anchor, committed, &pieces).unwrap();
    assert!(golden.error("block_hidden", &hidden) <= tolerance);
    // Either way the head reads the last committed position.
    let (shape, head) = &golden.tensors["fused.head"];
    let want = &head[(shape[0] - 1) * shape[1]..];
    for context in [&pieces, &whole] {
        let got = context.last_head().unwrap().to_vec1::<f32>().unwrap();
        let error = super::tests::max_error(&got, want);
        assert!(error <= tolerance * 10.0, "f_rnn error {error}");
    }
}

#[test]
fn context_committed_in_pieces_is_the_same_context() {
    context_grows_in_pieces(&Device::Cpu, CPU_TOLERANCE);
}

#[cfg(feature = "metal")]
#[test]
fn context_committed_in_pieces_is_the_same_context_on_metal() {
    let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
        "context_committed_in_pieces_is_the_same_context_on_metal",
    ) else {
        return;
    };
    context_grows_in_pieces(&device, METAL_TOLERANCE);
}

/// A local layer sees the window before the anchor and a global one
/// everything; the dense context keeps no more than that.
#[test]
fn local_layers_see_only_the_window() {
    let golden = Golden::load();
    let device = Device::Cpu;
    let drafter = tiny_drafter(&device);
    let config = drafter.config().clone();
    assert_eq!(config.local, [true, false, true]);
    let context = committed_context(&drafter, &golden);
    let committed = context.committed();
    // The fixture commits more than a window, so the window matters.
    assert!(committed > config.window);
    assert_eq!(
        config.first_visible(0, committed),
        committed - config.window
    );
    assert_eq!(config.first_visible(1, committed), 0);
    assert_eq!(config.first_visible(0, 3), 0);
    assert_eq!(context.held(0), config.window);
    assert_eq!(context.held(1), committed);
    assert_eq!(context.held(2), config.window);

    let first = committed - config.window;
    let (keys, values) = context.span(0, first).unwrap().unwrap();
    assert_eq!(
        keys.dims(),
        [config.window, config.kv_heads, config.head_dim]
    );
    assert_eq!(values.dims(), keys.dims());
    // The rows kept are the last ones of the reference's.
    let (shape, want) = &golden.tensors["context.0.keys"];
    let row = shape[1] * shape[2];
    let got = keys.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(super::tests::max_error(&got, &want[first * row..]) <= 1e-5);
    // Older positions are gone from a local layer, and nothing lies past
    // the committed ones.
    assert!(context.span(0, first - 1).is_err());
    assert!(context.span(0, committed).unwrap().is_none());
    assert!(context.span(1, 0).unwrap().is_some());
    assert!(context.span(3, 0).is_err());

    // A context that hands a local layer every position, whatever span it
    // was asked for.
    struct Everything {
        committed: usize,
        layers: Vec<(Tensor, Tensor)>,
    }
    impl DrafterContext for Everything {
        fn committed(&self) -> usize {
            self.committed
        }
        fn span(
            &self,
            layer: usize,
            _first: usize,
        ) -> crate::error::Result<Option<(Tensor, Tensor)>> {
            Ok(Some(self.layers[layer].clone()))
        }
    }
    let everything = Everything {
        committed,
        layers: (0..config.layers)
            .map(|layer| {
                (
                    golden.tensor(&format!("context.{layer}.keys"), &device),
                    golden.tensor(&format!("context.{layer}.values"), &device),
                )
            })
            .collect(),
    };
    let anchor = golden.tensor("anchor_embedding", &device);
    // The drafter refuses a span that is not the one it asked for.
    assert!(drafter
        .draft_block(&anchor, committed, &everything)
        .is_err());

    // The anchor must sit directly after the committed positions.
    assert!(drafter
        .draft_block(&anchor, committed + 1, &context)
        .is_err());
    assert!(drafter
        .draft_block(&anchor, committed - 1, &context)
        .is_err());
}

/// With nothing committed the block attends to itself alone.
#[test]
fn a_block_runs_without_context() {
    let golden = Golden::load();
    let device = Device::Cpu;
    let drafter = tiny_drafter(&device);
    let context = DenseDrafterContext::new(drafter.config());
    assert!(context.last_head().is_err());
    let anchor = golden.tensor("anchor_embedding", &device);
    let hidden = drafter.draft_block(&anchor, 0, &context).unwrap();
    assert_eq!(
        hidden.dims(),
        [drafter.config().block, drafter.config().hidden]
    );
    let values = hidden.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!(values.iter().all(|value| value.is_finite()));
    // And it is not the block the context produces.
    assert!(golden.error("block_hidden", &hidden) > 1e-2);
}

#[test]
fn the_chain_takes_rows_and_tokens_in_turn() {
    let golden = Golden::load();
    let device = Device::Cpu;
    let drafter = tiny_drafter(&device);
    let context = committed_context(&drafter, &golden);
    let unembed = golden.tensor("unembed", &device);
    let hidden = golden.tensor("block_hidden", &device);
    let rows = drafter.mask_rows(&hidden).unwrap();
    let base = drafter.base_logits(&rows, &unembed).unwrap();
    let f_rnn = context.last_head().unwrap();
    let config = drafter.config();
    assert_eq!(config.drafted(), 7);

    assert!(drafter
        .markov_chain(f_rnn, config.padded_vocab as u32)
        .is_err());
    let mut chain = drafter.markov_chain(f_rnn, 1).unwrap();
    let step = |chain: &mut super::drafter::MarkovChain<'_>, row: usize| {
        chain.step(
            row,
            &rows.get(row.min(6)).unwrap(),
            &base.get(row.min(6)).unwrap(),
            &unembed,
        )
    };
    assert!(chain.advance(3).is_err());
    assert!(step(&mut chain, 1).is_err());
    let logits = step(&mut chain, 0).unwrap();
    // Logits cover the tokens, not the padding of the vocabulary.
    assert_eq!(logits.dims(), [config.vocab]);
    assert!(config.vocab < config.padded_vocab);
    assert!(step(&mut chain, 1).is_err());
    assert!(chain.advance(config.padded_vocab as u32).is_err());
    chain.advance(3).unwrap();
    assert!(step(&mut chain, 0).is_err());
    for row in 1..7 {
        step(&mut chain, row).unwrap();
        chain.advance(3).unwrap();
    }
    assert!(step(&mut chain, 7).is_err());

    // A wrong-shaped unembedding is refused before any arithmetic.
    let narrow = unembed.narrow(0, 0, config.vocab).unwrap();
    assert!(drafter.base_logits(&rows, &narrow).is_err());
}

#[test]
fn the_published_config_parses() {
    let config = Inkling2DrafterConfig::load(&fixture("inkling2")).unwrap();
    assert_eq!(
        (config.hidden, config.layers, config.intermediate),
        (2304, 3, 9216)
    );
    assert_eq!(
        (config.heads, config.kv_heads, config.head_dim),
        (24, 8, 128)
    );
    assert_eq!(config.local, [true, false, true]);
    assert_eq!(
        (config.window, config.block, config.drafted()),
        (1024, 8, 7)
    );
    assert_eq!(config.markov_rank, 256);
    // 24 layers, 12 of them run three times: 48 applications, and the last
    // tap is the final-norm output.
    assert_eq!(config.taps, [5, 23, 41, 48]);
    assert_eq!((config.padded_vocab, config.vocab), (201024, 200058));
    assert_eq!(config.softcap, Some(30.0));
    assert_eq!(config.rope_theta, 1e6);
    assert_eq!(config.max_positions, Some(1048576));
}

fn fixture_config() -> serde_json::Value {
    let text = std::fs::read_to_string(fixture(FIXTURE).join(CONFIG_FILE)).unwrap();
    serde_json::from_str(&text).unwrap()
}

#[test]
fn a_config_the_drafter_cannot_follow_is_refused() {
    let parse = |edit: &dyn Fn(&mut serde_json::Value)| {
        let mut config = fixture_config();
        edit(&mut config);
        Inkling2DrafterConfig::from_json(&config.to_string())
    };
    assert!(parse(&|_| {}).is_ok());
    let refused: [&dyn Fn(&mut serde_json::Value); 9] = [
        &|c| {
            c.as_object_mut().unwrap().remove("mtp_config");
        },
        &|c| c["mtp_config"]["markov_head_type"] = "rnn".into(),
        &|c| c["mtp_config"]["head_dim"] = 7.into(),
        &|c| c["mtp_config"]["num_key_value_heads"] = 3.into(),
        &|c| c["mtp_config"]["hidden_size"] = 32.into(),
        &|c| c["mtp_config"]["local_layer_ids"] = serde_json::json!([0, 3]),
        // The last tap must be the final-norm output: 16 applications.
        &|c| c["mtp_config"]["target_layer_ids"] = serde_json::json!([3, 7, 13, 15]),
        &|c| c["mtp_config"]["block_size"] = 1.into(),
        &|c| c["mtp_config"]["q_bias"] = true.into(),
    ];
    for (index, edit) in refused.iter().enumerate() {
        assert!(parse(edit).is_err(), "edit {index} was accepted");
    }
}

/// A copy of the fixture's model directory whose weight file went through
/// `edit` (name, shape, bytes of every tensor).
fn edited_fixture(
    edit: impl Fn(&mut Vec<(String, Vec<usize>, Vec<u8>)>),
) -> (PathBuf, crate::error::Result<Inkling2Drafter>) {
    let source = fixture(FIXTURE);
    let bytes = std::fs::read(source.join(MTP_FILE)).unwrap();
    let file = SafeTensors::deserialize(&bytes).unwrap();
    let mut tensors = file
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            assert_eq!(view.dtype(), safetensors::Dtype::BF16);
            (name, view.shape().to_vec(), view.data().to_vec())
        })
        .collect::<Vec<_>>();
    tensors.sort();
    edit(&mut tensors);
    let views = tensors
        .iter()
        .map(|(name, shape, data)| {
            (
                name.clone(),
                TensorView::new(safetensors::Dtype::BF16, shape.clone(), data).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let dir = std::env::temp_dir().join(format!("izwi-inkling2-drafter-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(source.join(CONFIG_FILE), dir.join(CONFIG_FILE)).unwrap();
    std::fs::write(
        dir.join(MTP_FILE),
        safetensors::serialize(views, None).unwrap(),
    )
    .unwrap();
    let loaded = Inkling2Drafter::load(&dir, &Device::Cpu);
    (dir, loaded)
}

#[test]
fn a_weight_file_that_does_not_fit_the_config_is_refused() {
    let message = |edit: &dyn Fn(&mut Vec<(String, Vec<usize>, Vec<u8>)>)| {
        let (dir, loaded) = edited_fixture(edit);
        std::fs::remove_dir_all(&dir).unwrap();
        loaded.map(|_| ()).map_err(|err| err.to_string())
    };
    // The copy itself loads.
    assert_eq!(message(&|_| {}), Ok(()));

    let missing = message(&|tensors| {
        tensors.retain(|(name, _, _)| name != "layers.1.self_attn.k_norm_ctx.weight");
    })
    .unwrap_err();
    assert!(
        missing.contains("`layers.1.self_attn.k_norm_ctx.weight` is missing"),
        "{missing}"
    );

    // The same bytes under a transposed shape.
    let shaped = message(&|tensors| {
        let entry = tensors
            .iter_mut()
            .find(|(name, _, _)| name == "layers.0.self_attn.o_proj.weight")
            .unwrap();
        entry.1.reverse();
    })
    .unwrap_err();
    assert!(
        shaped.contains("`layers.0.self_attn.o_proj.weight` has shape [32, 64]"),
        "{shaped}"
    );

    let extra = message(&|tensors| {
        tensors.push(("layers.3.dflare_norm.weight".into(), vec![1], vec![0, 0]));
    })
    .unwrap_err();
    assert!(extra.contains("is not a drafter tensor"), "{extra}");

    // No weight file at all.
    let dir = std::env::temp_dir().join(format!("izwi-inkling2-drafter-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::copy(fixture(FIXTURE).join(CONFIG_FILE), dir.join(CONFIG_FILE)).unwrap();
    assert!(Inkling2Drafter::load(&dir, &Device::Cpu).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The head through a projection of the corrected hidden state gives the
/// logits of base plus correction, up to rounding, and the same greedy
/// tokens.
#[test]
fn the_head_through_a_projection_matches_base_plus_correction() {
    let device = Device::Cpu;
    let golden = Golden::load();
    let drafter = tiny_drafter(&device);
    let unembed = golden.tensor("unembed", &device);
    let context = committed_context(&drafter, &golden);
    let block = drafter
        .draft_block(
            &golden.tensor("anchor_embedding", &device),
            context.committed(),
            &context,
        )
        .unwrap();
    let anchor = golden.integers["anchor_token"][0];
    let f_rnn = context.last_head().unwrap().clone();
    let (want_tokens, want_logits) = drafter
        .draft_greedy(&block, &f_rnn, anchor, &unembed)
        .unwrap();
    let rows = drafter.mask_rows(&block).unwrap();
    // A served model's unembedding has no padded rows.
    let unpadded = unembed.narrow(0, 0, drafter.config().vocab).unwrap();
    let project =
        |hidden: &Tensor| -> crate::error::Result<Tensor> { Ok(hidden.matmul(&unpadded.t()?)?) };
    let mut chain = drafter.markov_chain(&f_rnn, anchor).unwrap();
    for row in 0..drafter.config().drafted() {
        let logits = chain
            .step_projected(row, &rows.get(row).unwrap(), &project)
            .unwrap();
        let got = logits.to_vec1::<f32>().unwrap();
        let want = want_logits.get(row).unwrap().to_vec1::<f32>().unwrap();
        let error = got
            .iter()
            .zip(&want)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(error <= 1e-4, "row {row}: {error}");
        let token = logits.argmax(0).unwrap().to_scalar::<u32>().unwrap();
        assert_eq!(token, want_tokens[row]);
        chain.advance(token).unwrap();
    }
}
