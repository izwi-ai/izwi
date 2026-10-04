//! Host transfer codec for whole pages (DS4), used by the CPU and accelerator
//! arenas' `capture_page`/`restore_page` implementations.
//!
//! Layout: per layer in `KvArenaConfig::layers` order, the key block then the
//! value block, each a contiguous run of
//! `page_tokens * kv_heads * head_dim` elements in the arena dtype's native
//! bit representation. Capture and restore are symmetric on one host, so
//! native-endian element bytes are sufficient.

use candle_core::{DType, Device, Tensor, WithDType};

use crate::error::Error;
use crate::Result;

use super::KvArenaConfig;

/// Byte size of one whole page across every layer of an arena config, in the
/// layout produced by [`capture_block`] and consumed by [`decoded_page`].
/// Matches the per-page term of `KvArena::resident_bytes`.
pub fn arena_page_bytes(config: &KvArenaConfig) -> u64 {
    config.layers.iter().fold(0_u64, |total, layer| {
        let elements = u64::from(config.page_tokens)
            .saturating_mul(u64::from(layer.num_kv_heads))
            .saturating_mul(
                u64::from(layer.key_head_dim).saturating_add(u64::from(layer.value_head_dim)),
            );
        total.saturating_add(elements.saturating_mul(config.dtype.size_in_bytes() as u64))
    })
}

fn encode_block<T: WithDType>(
    tensor: &Tensor,
    page: usize,
    destination: &mut [u8],
    element_size: usize,
    encode: impl Fn(T) -> Vec<u8>,
) -> Result<usize> {
    let values = tensor.narrow(0, page, 1)?.flatten_all()?.to_vec1::<T>()?;
    let needed = values.len() * element_size;
    if destination.len() < needed {
        return Err(Error::InferenceError(format!(
            "KV host page buffer is too small: need {needed} bytes, got {}",
            destination.len()
        )));
    }
    for (index, value) in values.iter().enumerate() {
        let bytes = encode(*value);
        destination[index * element_size..(index + 1) * element_size].copy_from_slice(&bytes);
    }
    Ok(needed)
}

/// Appends one tensor's page slice to `destination`, returning the bytes
/// written. `tensor` must have shape `[capacity_pages, ...]`.
pub fn capture_block(tensor: &Tensor, page: usize, destination: &mut [u8]) -> Result<usize> {
    match tensor.dtype() {
        DType::F32 => encode_block(tensor, page, destination, 4, |v: f32| {
            v.to_ne_bytes().to_vec()
        }),
        DType::F16 => encode_block(tensor, page, destination, 2, |v: half::f16| {
            v.to_ne_bytes().to_vec()
        }),
        DType::BF16 => encode_block(tensor, page, destination, 2, |v: half::bf16| {
            v.to_ne_bytes().to_vec()
        }),
        other => Err(Error::InferenceError(format!(
            "KV host page transfer does not support {other:?} storage"
        ))),
    }
}

fn decode_block<T: WithDType>(
    source: &[u8],
    elements: usize,
    shape: (usize, usize, usize, usize),
    element_size: usize,
    decode: impl Fn(&[u8]) -> T,
) -> Result<Tensor> {
    let needed = elements * element_size;
    if source.len() < needed {
        return Err(Error::InferenceError(format!(
            "KV host page buffer is truncated: need {needed} bytes, got {}",
            source.len()
        )));
    }
    let mut values = Vec::with_capacity(elements);
    for index in 0..elements {
        values.push(decode(
            &source[index * element_size..(index + 1) * element_size],
        ));
    }
    let (_, page_tokens, heads, head_dim) = shape;
    Ok(Tensor::from_vec(
        values,
        (page_tokens, heads, head_dim),
        &Device::Cpu,
    )?)
}

/// Decodes one tensor's page bytes from the head of `source` into a new host
/// tensor shaped like `tensor`'s per-page slice
/// (`[page_tokens, heads, head_dim]`). The caller uploads it to the arena's
/// device before writing it back.
pub fn decoded_page(tensor: &Tensor, source: &[u8]) -> Result<Tensor> {
    let dims = tensor.dims();
    if dims.len() != 4 {
        return Err(Error::InferenceError(format!(
            "KV host page transfer expects a [capacity_pages, tokens, heads, dim] tensor, got {dims:?}"
        )));
    }
    let elements: usize = dims[1..].iter().product();
    let shape = (dims[0], dims[1], dims[2], dims[3]);
    match tensor.dtype() {
        DType::F32 => decode_block(source, elements, shape, 4, |b: &[u8]| {
            f32::from_ne_bytes(b.try_into().expect("f32 element size"))
        }),
        DType::F16 => decode_block(source, elements, shape, 2, |b: &[u8]| {
            half::f16::from_ne_bytes(b.try_into().expect("f16 element size"))
        }),
        DType::BF16 => decode_block(source, elements, shape, 2, |b: &[u8]| {
            half::bf16::from_ne_bytes(b.try_into().expect("bf16 element size"))
        }),
        other => Err(Error::InferenceError(format!(
            "KV host page transfer does not support {other:?} storage"
        ))),
    }
}

/// Writes one tensor's page slice from the head of `source`, returning the
/// bytes consumed. `tensor` must have shape `[capacity_pages, ...]`; the
/// restored page replaces the existing rows for `page`. The decoded page is
/// uploaded to `tensor`'s device before the scatter.
pub fn restore_block(tensor: &Tensor, page: usize, source: &[u8]) -> Result<usize> {
    let dims = tensor.dims();
    if dims.len() != 4 || dims[0] <= page {
        return Err(Error::InferenceError(format!(
            "KV host page restore expects a [capacity_pages, tokens, heads, dim] tensor, got {dims:?}"
        )));
    }
    let elements: usize = dims[1..].iter().product();
    let page_tensor = decoded_page(tensor, source)?
        .reshape((1, dims[1], dims[2], dims[3]))?
        .to_device(tensor.device())?;
    let destination = tensor.narrow(0, page, 1)?;
    let index_shape = vec![1; page_tensor.rank()];
    let indices = Tensor::from_vec(vec![0_u32], (1,), destination.device())?
        .reshape(index_shape)?
        .broadcast_as(page_tensor.shape())?
        .contiguous()?;
    destination.scatter_set(&indices, &page_tensor, 0)?;
    Ok(elements * tensor.dtype().size_in_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::kv::{KvArenaId, KvLayerConfig};
    use crate::backends::BackendKind;
    use crate::engine::ModelInstanceId;
    use crate::kv::{KvGroupId, KvLayerBinding};

    const ARENA: KvArenaId = KvArenaId {
        model_instance: ModelInstanceId::new(41),
        backend: BackendKind::Cpu,
        device_ordinal: None,
        generation: 3,
    };
    const GROUP: KvGroupId = KvGroupId::new(5);

    /// Deterministic xorshift-derived values in [-4, 4) with special bit
    /// patterns (±inf, NaN, -0.0) injected at fixed strides so the round trip
    /// must be bit-preserving, not approximately preserving.
    fn seeded_values(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|index| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if index % 89 == 0 {
                    -0.0_f32
                } else if index % 97 == 0 {
                    f32::INFINITY
                } else if index % 101 == 0 {
                    f32::NEG_INFINITY
                } else if index % 103 == 0 {
                    f32::NAN
                } else {
                    let unit = (state >> 40) as f32 / (1 << 24) as f32;
                    (unit * 8.0) - 4.0
                }
            })
            .collect()
    }

    /// A `[capacity_pages, page_tokens, heads, head_dim]` tensor of seeded
    /// values converted to the arena dtype, mirroring a real arena layer's
    /// per-page slice shape.
    fn layer_tensor(
        dtype: DType,
        device: &Device,
        capacity_pages: usize,
        page_tokens: usize,
        heads: usize,
        head_dim: usize,
        seed: u64,
    ) -> Result<Tensor> {
        let values = seeded_values(seed, capacity_pages * page_tokens * heads * head_dim);
        let tensor = Tensor::from_vec(values, (capacity_pages, page_tokens, heads, head_dim), device)?;
        if dtype == DType::F32 {
            Ok(tensor)
        } else {
            Ok(tensor.to_dtype(dtype)?)
        }
    }

    /// Two layers with asymmetric head dims, so the page layout (per layer:
    /// key block then value block) is exercised non-uniformly.
    fn arena_config(dtype: DType) -> KvArenaConfig {
        KvArenaConfig {
            id: ARENA,
            group: GROUP,
            page_tokens: 8,
            capacity_pages: 3,
            growth: None,
            dtype,
            layers: vec![
                KvLayerConfig {
                    binding: KvLayerBinding {
                        model_layer: 0,
                        physical_layer: 0,
                    },
                    num_kv_heads: 2,
                    key_head_dim: 4,
                    value_head_dim: 6,
                },
                KvLayerConfig {
                    binding: KvLayerBinding {
                        model_layer: 1,
                        physical_layer: 1,
                    },
                    num_kv_heads: 3,
                    key_head_dim: 2,
                    value_head_dim: 1,
                },
            ],
        }
    }

    fn layer_dims(config: &KvArenaConfig) -> Vec<(usize, usize, usize)> {
        config
            .layers
            .iter()
            .map(|layer| {
                (
                    layer.num_kv_heads as usize,
                    layer.key_head_dim as usize,
                    layer.value_head_dim as usize,
                )
            })
            .collect()
    }

    /// Full page round trip on one device: capture every layer block into one
    /// page buffer, restore into fresh zero arenas, then recapture and demand
    /// bitwise-identical bytes (NaN and -0.0 included).
    fn round_trip_page(device: &Device, dtype: DType, seed: u64, page: usize) -> Result<()> {
        let config = arena_config(dtype);
        let capacity = config.capacity_pages as usize;
        let page_tokens = config.page_tokens as usize;
        let dims = layer_dims(&config);

        let mut sources = Vec::new();
        for (layer, (heads, key_dim, value_dim)) in dims.iter().enumerate() {
            sources.push(layer_tensor(
                dtype,
                device,
                capacity,
                page_tokens,
                *heads,
                *key_dim,
                seed + layer as u64,
            )?);
            sources.push(layer_tensor(
                dtype,
                device,
                capacity,
                page_tokens,
                *heads,
                *value_dim,
                seed + 100 + layer as u64,
            )?);
        }

        let page_bytes = arena_page_bytes(&config) as usize;
        let mut captured = vec![0_u8; page_bytes];
        let mut offset = 0usize;
        for tensor in &sources {
            offset += capture_block(tensor, page, &mut captured[offset..])?;
        }
        assert_eq!(offset, page_bytes, "captured bytes must match the arena page size");

        let mut restored = Vec::new();
        for tensor in &sources {
            let dims = tensor.dims();
            restored.push(Tensor::zeros(
                (dims[0], dims[1], dims[2], dims[3]),
                tensor.dtype(),
                device,
            )?);
        }
        offset = 0usize;
        for tensor in &restored {
            offset += restore_block(tensor, page, &captured[offset..])?;
        }
        assert_eq!(offset, page_bytes, "restored bytes must consume the whole page buffer");

        // Bitwise proof: recapturing the restored pages must reproduce the
        // original capture byte-for-byte, so the codec is lossless including
        // NaN payloads, infinities, and negative zero.
        let mut recaptured = vec![0_u8; page_bytes];
        offset = 0usize;
        for tensor in &restored {
            offset += capture_block(tensor, page, &mut recaptured[offset..])?;
        }
        assert_eq!(recaptured, captured, "page round trip must be bit-preserving");

        // The decoded page is a host (CPU) tensor shaped [tokens, heads, dim].
        let decoded = decoded_page(&sources[0], &captured)?;
        assert_eq!(decoded.device().location(), candle_core::DeviceLocation::Cpu);
        assert_eq!(decoded.dims(), &[page_tokens, dims[0].0, dims[0].1]);
        Ok(())
    }

    #[test]
    fn page_round_trip_is_bit_preserving_across_dtypes_and_pages() -> Result<()> {
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for seed in 0..4_u64 {
                for page in 0..3_usize {
                    round_trip_page(&Device::Cpu, dtype, seed, page)?;
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn page_round_trip_is_bit_preserving_through_metal_arenas() -> Result<()> {
        let Some(device) = crate::backends::metal_device_if_available(0) else {
            eprintln!("skipping Metal page-transfer round trip: no Metal device");
            return Ok(());
        };
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            for seed in 0..2_u64 {
                for page in 0..3_usize {
                    round_trip_page(&device, dtype, seed, page)?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn capture_rejects_undersized_destination() -> Result<()> {
        let tensor = layer_tensor(DType::F32, &Device::Cpu, 2, 4, 2, 3, 7)?;
        let mut destination = vec![0_u8; 4 * 4 * 2 * 3 - 1];
        let error = capture_block(&tensor, 0, &mut destination).unwrap_err();
        assert!(format!("{error}").contains("too small"));
        Ok(())
    }

    #[test]
    fn decode_and_restore_reject_truncated_and_out_of_range_input() -> Result<()> {
        let tensor = layer_tensor(DType::F16, &Device::Cpu, 2, 4, 2, 3, 9)?;
        let truncated = vec![0_u8; 4 * 2 * 3 * 2 - 1];
        let error = decoded_page(&tensor, &truncated).unwrap_err();
        assert!(format!("{error}").contains("truncated"));

        let mut full = vec![0_u8; 4 * 2 * 3 * 2];
        let error = restore_block(&tensor, 2, &full).unwrap_err();
        assert!(format!("{error}").contains("expects a [capacity_pages"));

        let flat = Tensor::zeros((4, 2, 3), DType::F32, &Device::Cpu)?;
        let error = decoded_page(&flat, &full).unwrap_err();
        assert!(format!("{error}").contains("expects a [capacity_pages, tokens, heads, dim]"));

        let quantized = Tensor::zeros((2, 4, 2, 3), DType::I64, &Device::Cpu)?;
        let error = capture_block(&quantized, 0, &mut full).unwrap_err();
        assert!(format!("{error}").contains("does not support"));
        Ok(())
    }
}
