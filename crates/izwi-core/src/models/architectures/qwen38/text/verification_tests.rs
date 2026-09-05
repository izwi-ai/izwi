use super::*;
use crate::backends::kv::{CpuKvArena, KvArenaConfig, KvLayerConfig};
use crate::backends::BackendKind;
use crate::engine::ModelInstanceId;
use crate::kv::{CacheBlockRef, KvArenaId, KvGroupId, KvLayerBinding};

fn dense(output: usize, input: usize, scale: f32) -> Qwen38Projection {
    Qwen38Projection::Dense(Linear::new(
        Tensor::from_vec(
            (0..output * input)
                .map(|i| ((i * 7 % 13) as f32 - 6.0) * scale)
                .collect(),
            (output, input),
            &Device::Cpu,
        )
        .unwrap(),
        None,
    ))
}
fn norm(width: usize) -> Qwen38RmsNorm {
    Qwen38RmsNorm {
        weight: Tensor::ones(width, DType::F32, &Device::Cpu).unwrap(),
        eps: 1e-6,
    }
}
fn mlp() -> Qwen38Mlp {
    Qwen38Mlp {
        graphs: Arc::new(TensorIsland::default()),
        graphs_enabled: false,
        graph_generation: 0,
        fused_decode: false,
        gate_up: Qwen38ProjectionGroup::Separate(vec![dense(8, 4, 0.04), dense(8, 4, 0.03)]),
        down: dense(4, 8, 0.04),
    }
}
fn model() -> Qwen38TextModel {
    model_with_value_heads(1)
}

fn model_with_value_heads(value_heads: usize) -> Qwen38TextModel {
    let device = &Device::Cpu;
    let conv_dim = 4 + 2 * value_heads;
    let conv_kernel = Tensor::from_vec(
        (0..conv_dim * 4)
            .map(|i| ((i % 4) + 1) as f32 * 0.1)
            .collect(),
        (conv_dim, 4),
        device,
    )
    .unwrap();
    let linear = Qwen38LinearAttention {
        fused_decode: false,
        qkv_z_proj: Qwen38ProjectionGroup::Separate(vec![
            dense(conv_dim, 4, 0.07),
            dense(2 * value_heads, 4, 0.06),
        ]),
        alpha_beta_proj: Qwen38ProjectionGroup::Separate(vec![
            dense(value_heads, 4, 0.03),
            dense(value_heads, 4, -0.02),
        ]),
        dt_bias: Tensor::zeros((1, 1, value_heads), DType::F32, device).unwrap(),
        a: Tensor::full(-0.5f32, (1, 1, value_heads), device).unwrap(),
        conv_kernel_slices: pre_slice_conv_kernel(&conv_kernel, 4).unwrap(),
        conv_kernel,
        norm: Qwen38GatedRmsNorm {
            fused_decode: false,
            weight: Tensor::ones(2, DType::F32, device).unwrap(),
            eps: 1e-6,
        },
        out_proj: dense(4, 2 * value_heads, 0.04),
        num_k_heads: 1,
        num_v_heads: value_heads,
        head_k_dim: 2,
        head_v_dim: 2,
        conv_dim,
        kernel_size: 4,
        tiled_recurrence_enabled: false,
        tiled_recurrence_tile_size_override: None,
    };
    let full = Qwen38FullAttention {
        qkv_proj: Qwen38ProjectionGroup::Separate(vec![
            dense(8, 4, 0.08),
            dense(2, 4, 0.05),
            dense(2, 4, 0.04),
        ]),
        o_proj: dense(4, 4, 0.03),
        q_norm: norm(2),
        k_norm: norm(2),
        num_heads: 2,
        num_kv_heads: 1,
        head_dim: 2,
        rope_dim: 2,
        rope_theta: 10_000.0,
        mrope_sections: vec![1, 0, 0],
        rope_kernel_enabled: false,
        rope_inv_freqs: build_rope_inv_freqs(2, 10_000.0).unwrap(),
    };
    Qwen38TextModel {
        device: device.clone(),
        projection_representation: Qwen38ProjectionRepresentation::ExpandedF32,
        token_embeddings: Embedding::new(
            Tensor::from_vec(
                (0..32).map(|i| ((i * 3 % 11) as f32 - 5.0) * 0.1).collect(),
                (8, 4),
                device,
            )
            .unwrap(),
            4,
        ),
        layers: vec![
            Qwen38Layer {
                graphs: Arc::new(TensorIsland::default()),
                graphs_enabled: false,
                graph_generation: 0,
                attn_norm: norm(4),
                mixer: Qwen38Mixer::Linear(linear),
                post_attention_norm: norm(4),
                mlp: mlp(),
            },
            Qwen38Layer {
                graphs: Arc::new(TensorIsland::default()),
                graphs_enabled: false,
                graph_generation: 0,
                attn_norm: norm(4),
                mixer: Qwen38Mixer::Full(full),
                post_attention_norm: norm(4),
                mlp: mlp(),
            },
        ],
        output_norm: norm(4),
        output: dense(8, 4, 0.11),
        finite_diagnostics_enabled: false,
    }
}
fn cache() -> PhysicalPagedKvCache {
    let id = KvArenaId {
        model_instance: ModelInstanceId::new(20),
        backend: BackendKind::Cpu,
        device_ordinal: None,
        generation: 1,
    };
    let group = KvGroupId::new(1);
    let bindings = vec![KvLayerBinding {
        model_layer: 1,
        physical_layer: 0,
    }];
    let arena = Arc::new(
        CpuKvArena::new(KvArenaConfig {
            id,
            group,
            page_tokens: 4,
            capacity_pages: 4,
            growth: None,
            dtype: DType::F32,
            layers: vec![KvLayerConfig {
                binding: bindings[0],
                num_kv_heads: 1,
                key_head_dim: 2,
                value_head_dim: 2,
            }],
        })
        .unwrap(),
    );
    let blocks = (0..4)
        .map(|index| CacheBlockRef {
            arena: id,
            group,
            index,
            slot_generation: 1,
        })
        .collect();
    PhysicalPagedKvCache::new(arena, bindings, blocks, 0).unwrap()
}
fn close(a: &Tensor, b: &Tensor) {
    assert_eq!(a.dims(), b.dims());
    let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() < 2e-5, "{a} != {b}");
    }
}
fn assert_state(a: &Qwen38TextRuntimeState, b: &Qwen38TextRuntimeState) {
    for (a, b) in a.layers.iter().zip(&b.layers) {
        if let (
            Qwen38LayerRuntimeState::Linear {
                conv_state: Some(ac),
                recurrent_state: Some(ar),
            },
            Qwen38LayerRuntimeState::Linear {
                conv_state: Some(bc),
                recurrent_state: Some(br),
            },
        ) = (a, b)
        {
            close(
                &ac.contiguous_history().unwrap(),
                &bc.contiguous_history().unwrap(),
            );
            close(ar, br);
        }
    }
}
#[test]
fn every_verified_prefix_matches_scalar_hybrid_state_and_next_logits() {
    for value_heads in [1, 2, 3] {
        let model = model_with_value_heads(value_heads);
        for width in 2..=4 {
            for prefix in 0..=width {
                let mut state = model.new_state();
                let mut kv = cache();
                let mut reference = model.new_state();
                let mut reference_kv = cache();
                model
                    .forward_token_id_hidden_at_physical(1, [0; 3], &mut state, &mut kv)
                    .unwrap();
                model
                    .forward_token_id_hidden_at_physical(
                        1,
                        [0; 3],
                        &mut reference,
                        &mut reference_kv,
                    )
                    .unwrap();
                let before = state.clone();
                let ids = (2..2 + width as u32).collect::<Vec<_>>();
                let positions = (1..=width).map(|p| [p; 3]).collect::<Vec<_>>();
                let verified = model
                    .verify_token_ids_physical(&ids, &positions, &mut state, &mut kv)
                    .unwrap();
                // The target pass cannot mutate the retained initial checkpoint.
                assert_state(&before, &reference);
                let committed = verified.commit_prefix(prefix, &mut state, &mut kv).unwrap();
                for position in 0..prefix {
                    let hidden = model
                        .forward_token_id_hidden_at_physical(
                            ids[position],
                            positions[position],
                            &mut reference,
                            &mut reference_kv,
                        )
                        .unwrap();
                    close(&committed.narrow(1, position, 1).unwrap(), &hidden);
                }
                assert_eq!(kv.context_len(), reference_kv.context_len());
                assert_state(&state, &reference);
                // Fixed-size escaping storage, including deep-copied conv history.
                assert_eq!(
                    state.allocated_session_bytes(),
                    Some(((4 + 2 * value_heads) * 3 + value_heads * 2 * 2) as u64 * 4)
                );
                let next = model
                    .forward_token_id_at_physical(7, [prefix + 1; 3], &mut state, &mut kv)
                    .unwrap();
                let expected = model
                    .forward_token_id_at_physical(
                        7,
                        [prefix + 1; 3],
                        &mut reference,
                        &mut reference_kv,
                    )
                    .unwrap();
                close(&next, &expected);
                assert_eq!(
                    next.argmax(0).unwrap().to_scalar::<u32>().unwrap(),
                    expected.argmax(0).unwrap().to_scalar::<u32>().unwrap()
                );
            }
        }
    }
}

#[test]
fn rejected_prefix_commit_refuses_foreign_view_and_same_view_generation_aba() {
    let model = model();
    let mut state = model.new_state();
    let mut kv = cache();
    let verified = model
        .verify_token_ids_physical(&[1, 2], &[[0; 3], [1; 3]], &mut state, &mut kv)
        .unwrap();
    let mut foreign = cache();
    let mut foreign_state = model.new_state();
    model
        .prefill_token_ids_with_hidden_physical(
            &[1, 2],
            &[[0; 3], [1; 3]],
            &mut foreign_state,
            &mut foreign,
            false,
        )
        .unwrap();
    assert!(verified
        .commit_prefix(1, &mut foreign_state, &mut foreign)
        .is_err());
    let checkpoint = kv.logical_checkpoint();
    let verified = model
        .verify_token_ids_physical(&[3, 4], &[[2; 3], [3; 3]], &mut state, &mut kv)
        .unwrap();
    kv.restore_logical_checkpoint(checkpoint).unwrap();
    model
        .prefill_token_ids_with_hidden_physical(
            &[5, 6],
            &[[2; 3], [3; 3]],
            &mut state,
            &mut kv,
            false,
        )
        .unwrap();
    assert!(verified.commit_prefix(1, &mut state, &mut kv).is_err());
}
#[test]
fn cancellation_restores_checkpoint_and_verification_budget_is_bounded() {
    let model = model();
    let mut state = model.new_state();
    let mut kv = cache();
    model
        .forward_token_id_hidden_at_physical(1, [0; 3], &mut state, &mut kv)
        .unwrap();
    let base = state.clone();
    let checkpoint = kv.logical_checkpoint();
    let verified = model
        .verify_token_ids_physical(&[2, 3, 4], &[[1; 3], [2; 3], [3; 3]], &mut state, &mut kv)
        .unwrap();
    drop(verified);
    kv.restore_logical_checkpoint(checkpoint).unwrap();
    state = base.clone();
    assert_eq!(kv.context_len(), 1);
    assert_state(&state, &base);
    assert!(model
        .verify_token_ids_physical(&[1; 5], &[[0; 3]; 5], &mut state, &mut kv)
        .is_err());
    assert_eq!(kv.context_len(), 1);
}

#[test]
fn native_mlp_graph_closure_matches_eager_and_retains_every_intermediate() {
    fn raw(n: usize, k: usize) -> Qwen38Projection {
        Qwen38Projection::NativeFp8(super::super::native::RawBlockFp8Projection {
            weights: Tensor::from_vec(
                (0..n * k).map(|i| 0x30u8 + (i % 8) as u8).collect(),
                (n, k),
                &Device::Cpu,
            )
            .unwrap(),
            scales: Tensor::full(0.02f32, (1, 1), &Device::Cpu).unwrap(),
            shape: [n, k],
            scale_shape: [1, 1],
            block_shape: [128, 128],
        })
    }
    let mlp = Qwen38Mlp {
        graphs: Arc::new(TensorIsland::default()),
        graphs_enabled: true,
        graph_generation: 0,
        fused_decode: false,
        gate_up: Qwen38ProjectionGroup::Separate(vec![raw(8, 4), raw(8, 4)]),
        down: raw(4, 8),
    };
    let input = Tensor::from_slice(&[0.4f32, -0.3, 0.1, 0.2], (1, 1, 4), &Device::Cpu).unwrap();
    let owners = mlp.native_graph_owners().unwrap();
    assert_eq!(owners.len(), 6);
    let captured = mlp.native_graph_forward(&input).unwrap();
    assert_eq!(captured.intermediates.len(), 4);
    close(&captured.outputs[0], &mlp.forward_eager(&input).unwrap());
    close(&mlp.forward(&input).unwrap(), &captured.outputs[0]);
    assert!(
        model().layers[0].mlp.native_graph_owners().is_none(),
        "Q8/dense hidden scratch must not enter this region"
    );
}
