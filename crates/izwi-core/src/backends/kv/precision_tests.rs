use super::*;
use crate::engine::ModelInstanceId;
use candle_core::Device;

fn causal_overflow_config(backend: BackendKind, dtype: DType, head_dim: u32) -> KvArenaConfig {
    KvArenaConfig {
        id: KvArenaId {
            model_instance: ModelInstanceId::new(92),
            backend,
            device_ordinal: (backend == BackendKind::Cuda).then_some(0),
            generation: 1,
        },
        group: KvGroupId::new(0),
        page_tokens: 64,
        capacity_pages: 34,
        growth: None,
        dtype,
        layers: vec![KvLayerConfig {
            binding: KvLayerBinding {
                model_layer: 0,
                physical_layer: 0,
            },
            num_kv_heads: 1,
            key_head_dim: head_dim,
            value_head_dim: head_dim,
        }],
    }
}

// Four verification queries followed by a scalar decode of the final row.
// The last V is finite in BF16 but overflows F16 before attention runs.
fn causal_overflow_outputs(
    arena: &dyn KvArena,
    device: &Device,
    context_len: u32,
    portable_reference: bool,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let config = arena.config();
    let binding = config.layers[0].binding;
    let head_dim = config.layers[0].key_head_dim as usize;
    let capacity = (config.capacity_pages * config.page_tokens) as usize;
    assert!(context_len >= 4 && (context_len as usize) < capacity);
    // 13 is coprime to 34: every physical page occurs once, out of order.
    let blocks = (0..config.capacity_pages)
        .map(|page| CacheBlockRef {
            arena: config.id,
            group: config.group,
            index: (page * 13 + 7) % config.capacity_pages,
            slot_generation: 1,
        })
        .collect::<Vec<_>>();
    let slots = (0..capacity)
        .map(|token| KvSlotRef {
            block: blocks[token / config.page_tokens as usize],
            offset: (token % config.page_tokens as usize) as u32,
        })
        .collect::<Vec<_>>();
    let slots = arena.lower_slots(&slots)?;
    let mut keys = vec![f32::NAN; capacity * head_dim];
    let mut values = vec![f32::NAN; capacity * head_dim];
    for token in 0..context_len as usize {
        for dim in 0..head_dim {
            keys[token * head_dim + dim] = 0.0;
            values[token * head_dim + dim] = if token + 1 == context_len as usize {
                65_536.0
            } else {
                // Exact BF16 values distinguish pages and output columns.
                1.0 + ((token / 64) % 7) as f32 / 4.0 + (dim % 8) as f32 / 32.0
            };
        }
    }
    // Model the actual BF16 activation -> cache dtype conversion. NaNs
    // exist only outside the declared context, simulating discarded slots.
    let keys = Tensor::from_vec(keys, (capacity, 1, head_dim), device)?
        .to_dtype(DType::BF16)?
        .to_dtype(config.dtype)?;
    let values = Tensor::from_vec(values, (capacity, 1, head_dim), device)?
        .to_dtype(DType::BF16)?
        .to_dtype(config.dtype)?;
    arena
        .write_slots(
            binding,
            KvWriteArgs {
                keys: &keys,
                values: &values,
                slots: slots.as_ref(),
            },
        )?
        .wait()?;
    let blocks = blocks[..context_len.div_ceil(config.page_tokens) as usize].to_vec();
    let rows = [PagedKvPrefillRow {
        blocks: blocks.clone(),
        first_page_offset: 0,
        query_start: 0,
        query_len: 4,
        context_len,
    }];
    let queries = Tensor::zeros((4, 2, head_dim), config.dtype, device)?;
    let args = PagedKvPrefillArgs {
        queries: &queries,
        rows: &rows,
        softmax_scale: 1.0 / (head_dim as f32).sqrt(),
        softcap: None,
        window_tokens: None,
    };
    let prefill = if portable_reference {
        portable_paged_prefill(arena, binding, args)?
    } else {
        arena.paged_prefill(binding, args)?
    };
    let batch = KvDecodeBatchMetadata {
        sequences: vec![crate::kv::KvSequenceBlockTable {
            blocks,
            first_page_offset: 0,
            context_len,
        }],
    };
    let query = Tensor::zeros((1, 2, head_dim), config.dtype, device)?;
    let decode = arena.paged_decode(
        binding,
        PagedKvDecodeArgs {
            queries: &query,
            batch: &batch,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            softcap: None,
        },
    )?;
    let host = |output: Tensor| -> Result<Vec<f32>> {
        Ok(output
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?)
    };
    Ok((host(prefill)?, host(decode)?))
}

fn assert_finite_causal_overflow_match(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite(), "non-finite attention output at {index}");
        assert!(expected.is_finite());
        assert!(
            (actual - expected).abs() <= 0.015 * expected.abs().max(1.0),
            "attention output {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn portable_causal_bf16_kv_preserves_values_that_overflow_f16() -> Result<()> {
    let reference = CpuKvArena::new(causal_overflow_config(BackendKind::Cpu, DType::F32, 8))?;
    let (expected, expected_decode) = causal_overflow_outputs(&reference, &Device::Cpu, 65, true)?;
    // All visible-prefix averages fit F16; any failure is from storing V,
    // not from an attention result whose magnitude exceeds F16's range.
    assert!(expected.iter().all(|v| v.is_finite() && *v < 65_504.0));
    for (index, &value) in expected[..16].iter().enumerate() {
        assert_eq!(value, 1.0 + (index % 8) as f32 / 32.0);
    }
    for dtype in [DType::F16, DType::BF16] {
        let arena = CpuKvArena::new(causal_overflow_config(BackendKind::Cpu, dtype, 8))?;
        let (actual, decode) = causal_overflow_outputs(&arena, &Device::Cpu, 65, true)?;
        if dtype == DType::F16 {
            // The portable path never reads causally excluded future V.
            assert_finite_causal_overflow_match(&actual[..48], &expected[..48]);
            assert!(actual[48..].iter().all(|v| !v.is_finite()));
            assert!(decode.iter().all(|v| !v.is_finite()));
        } else {
            assert_finite_causal_overflow_match(&actual, &expected);
            assert_finite_causal_overflow_match(&decode, &expected_decode);
        }
    }
    Ok(())
}

#[cfg(all(feature = "cuda", feature = "flash-attn"))]
#[test]
#[ignore = "required CUDA FlashAttention evidence: run explicitly on SM80+; absence or fallback must fail"]
fn cuda_flash_paged_bf16_preserves_finite_kv_range() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let reference = CpuKvArena::new(causal_overflow_config(BackendKind::Cpu, DType::F32, 256))?;
    for dtype in [DType::F16, DType::BF16] {
        let arena = CandleAcceleratorKvArena::new_mutation_only(
            causal_overflow_config(BackendKind::Cuda, dtype, 256),
            device.clone(),
        )?;
        arena.disable_cuda_split_decode();
        let (keys, values) = arena.layer_tensors(arena.config().layers[0].binding)?;
        assert_eq!(keys.dtype(), dtype, "test requires dense KV storage");
        assert_eq!(values.dtype(), dtype, "test requires dense KV storage");
        for context in [63, 64, 65, 127, 128, 129, 2047, 2048, 2049] {
            let (expected, expected_decode) =
                causal_overflow_outputs(&reference, &Device::Cpu, context, true)?;
            let before = arena.operation_stats();
            let (actual, decode) = causal_overflow_outputs(&arena, &device, context, false)?;
            let after = arena.operation_stats();
            assert_eq!(
                after.cuda_flash_attention_dispatches - before.cuda_flash_attention_dispatches,
                2,
                "{dtype:?} context={context}: prefill and decode must both use FlashAttention"
            );
            assert_eq!(after.cuda_native_attention_dispatches, 0);
            assert_eq!(after.portable_attention_dispatches, 0);
            assert_eq!(
                after.last_attention_provider,
                Some(KvAttentionProvider::CudaFlashAttention)
            );
            if dtype == DType::BF16 {
                assert_finite_causal_overflow_match(&actual, &expected);
                assert_finite_causal_overflow_match(&decode, &expected_decode);
            } else {
                // Flash masks scores but still multiplies masked P by V:
                // 0 * Inf poisons a past query in the verification tile.
                assert!(
                    actual[..3 * 2 * 256].iter().any(|v| !v.is_finite()),
                    "F16 context={context}: expected masked future-V overflow"
                );
                assert!(decode.iter().all(|v| !v.is_finite()));
            }
        }
    }
    Ok(())
}
