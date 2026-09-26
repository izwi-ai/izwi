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
