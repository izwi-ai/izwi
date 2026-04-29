//! Shared paged-KV helpers for decode-time attention.

use candle_core::{DType, Tensor, D};

use crate::error::{Error, Result};
use crate::models::shared::telemetry::{record_decode_attention_path, DecodeAttentionPath};

const Q4_0_BLOCK_SIZE: usize = 32;

/// Default KV page size used when model-specific config is unavailable.
pub const DEFAULT_KV_PAGE_SIZE: usize = 64;

/// Supported KV cache quantization modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheQuantization {
    /// Keep KV pages in the model's native dtype.
    None,
    /// Store KV pages in int8 with per-page symmetric scale.
    Int8,
    /// Store KV pages in Q4_0 format (4-bit, 32-element blocks).
    /// This reduces memory by 75% compared to F16 at minor quality loss.
    Q4_0,
}

impl KvCacheQuantization {
    /// Parse quantization mode from a KV cache dtype hint.
    ///
    /// Accepts values such as `int8`, `i8`, `q8`, `q4_0`, `q4`, `float16`, `float32`, `bf16`.
    pub fn from_dtype_hint(dtype: &str) -> Self {
        match dtype.trim().to_ascii_lowercase().as_str() {
            "int8" | "i8" | "q8" | "q8_0" => Self::Int8,
            "q4_0" | "q4" => Self::Q4_0,
            _ => Self::None,
        }
    }

    /// Get the compression ratio compared to F16.
    pub fn compression_ratio(&self) -> f32 {
        match self {
            Self::None => 1.0,
            Self::Int8 => 2.0,
            Self::Q4_0 => 4.0,
        }
    }
}

/// Resolve default page size, optionally overridden by env.
pub fn default_kv_page_size() -> usize {
    std::env::var("IZWI_KV_PAGE_SIZE")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_KV_PAGE_SIZE)
}

/// Resolve default KV quantization from env.
pub fn default_kv_quantization() -> KvCacheQuantization {
    std::env::var("IZWI_KV_CACHE_DTYPE")
        .ok()
        .map(|raw| KvCacheQuantization::from_dtype_hint(&raw))
        .unwrap_or(KvCacheQuantization::None)
}

/// Resolve KV quantization for the target append device.
///
/// CUDA Q4_0 currently uses host-vector pack/unpack. Keep that fallback explicit
/// until a Rust-owned CUDA kernel exists for this format.
pub fn resolve_kv_cache_quantization(
    device: &candle_core::Device,
    requested: KvCacheQuantization,
) -> Result<KvCacheQuantization> {
    kv_cache_quantization_policy(
        device.is_cuda(),
        requested,
        env_bool("IZWI_CUDA_KV_Q4_0_HOST_FALLBACK", false),
    )
}

fn kv_cache_quantization_policy(
    is_cuda: bool,
    requested: KvCacheQuantization,
    cuda_q4_0_host_fallback_enabled: bool,
) -> Result<KvCacheQuantization> {
    if is_cuda
        && requested == KvCacheQuantization::Q4_0
        && !cuda_q4_0_host_fallback_enabled
    {
        return Err(Error::InvalidInput(
            "CUDA Q4_0 KV cache quantization requires a CUDA kernel or explicit \
             IZWI_CUDA_KV_Q4_0_HOST_FALLBACK=1 host fallback"
                .to_string(),
        ));
    }
    Ok(requested)
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

/// Single KV page storage (dense or quantized).
#[derive(Debug, Clone)]
pub enum KvPage {
    Dense(Tensor),
    Int8 {
        values: Tensor,
        scale: f32,
        target_dtype: DType,
    },
    /// Q4_0 quantized page: 4-bit weights with per-32-element block scales.
    /// Values are packed 2x4-bit per byte, with a block scale (f16/f32) every 32 values.
    Q4_0 {
        values: Tensor,      // U8 packed 4-bit values
        scales: Tensor,      // F32 scale per block
        seq_len: usize,      // Original sequence length
        num_elements: usize, // Total flat elements represented by the page
        target_dtype: DType,
    },
}

impl KvPage {
    fn from_dense(tensor: Tensor, quantization: KvCacheQuantization) -> Result<Self> {
        match quantization {
            KvCacheQuantization::None => Ok(Self::Dense(tensor)),
            KvCacheQuantization::Int8 => {
                let (values, scale, target_dtype) = quantize_tensor_int8(&tensor)?;
                Ok(Self::Int8 {
                    values,
                    scale,
                    target_dtype,
                })
            }
            KvCacheQuantization::Q4_0 => {
                let seq_len = tensor.dim(1)?;
                let num_elements = tensor.elem_count();
                let (values, scales, target_dtype) = quantize_tensor_q4_0(&tensor)?;
                let num_blocks = scales.elem_count();
                let expected_blocks = (num_elements + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE;
                if num_elements == 0 || num_blocks != expected_blocks {
                    return Err(Error::InvalidInput(
                        "Invalid Q4_0 quantization metadata".to_string(),
                    ));
                }
                Ok(Self::Q4_0 {
                    values,
                    scales,
                    seq_len,
                    num_elements,
                    target_dtype,
                })
            }
        }
    }

    fn seq_len(&self) -> Result<usize> {
        match self {
            Self::Dense(t) => t.dim(1).map_err(Error::from),
            Self::Int8 { values, .. } => values.dim(1).map_err(Error::from),
            Self::Q4_0 { seq_len, .. } => Ok(*seq_len),
        }
    }

    fn to_dense(&self) -> Result<Tensor> {
        match self {
            Self::Dense(t) => Ok(t.clone()),
            Self::Int8 {
                values,
                scale,
                target_dtype,
            } => dequantize_tensor_int8(values, *scale, *target_dtype),
            Self::Q4_0 {
                values,
                scales,
                seq_len: _,
                num_elements,
                target_dtype,
            } => dequantize_tensor_q4_0(values, scales, *num_elements, *target_dtype),
        }
    }
}

fn quantize_tensor_int8(tensor: &Tensor) -> Result<(Tensor, f32, DType)> {
    let target_dtype = tensor.dtype();
    let max_abs = tensor
        .abs()?
        .max_all()?
        .to_dtype(DType::F32)?
        .to_scalar::<f32>()?;
    let scale = if max_abs > 0.0 {
        (max_abs / 127.0).max(1e-8)
    } else {
        1.0
    };
    let inv_scale = 1.0f32 / scale;
    let inv_scale_t =
        Tensor::from_vec(vec![inv_scale], (1,), tensor.device())?.to_dtype(target_dtype)?;
    let offset_t =
        Tensor::from_vec(vec![128.0f32], (1,), tensor.device())?.to_dtype(target_dtype)?;
    let quantized = tensor
        .broadcast_mul(&inv_scale_t)?
        .clamp(-127.0f64, 127.0f64)?
        .round()?
        .broadcast_add(&offset_t)?
        .to_dtype(DType::U8)?;
    Ok((quantized, scale, target_dtype))
}

fn dequantize_tensor_int8(values: &Tensor, scale: f32, target_dtype: DType) -> Result<Tensor> {
    let dense = values.to_dtype(target_dtype)?;
    let offset_t =
        Tensor::from_vec(vec![128.0f32], (1,), values.device())?.to_dtype(target_dtype)?;
    let dense = dense.broadcast_sub(&offset_t)?;
    if (scale - 1.0).abs() < f32::EPSILON {
        return Ok(dense);
    }
    let scale_t = Tensor::from_vec(vec![scale], (1,), values.device())?.to_dtype(target_dtype)?;
    dense.broadcast_mul(&scale_t).map_err(Error::from)
}

/// Q4_0 quantization: 4-bit weights with per-32-element block scales.
///
/// Each block contains:
/// - 32 4-bit values packed into 16 bytes (2 values per byte)
/// - 1 f16/f32 scale factor for the block
///
/// Returns: (packed_values: U8, scales: F32, target_dtype)
fn quantize_tensor_q4_0(tensor: &Tensor) -> Result<(Tensor, Tensor, DType)> {
    let target_dtype = tensor.dtype();
    let device = tensor.device();

    // Convert to F32 for quantization
    let tensor_f32 = tensor.to_dtype(DType::F32)?;
    let flat = tensor_f32.flatten_all()?;
    let numel = flat.elem_count();

    let num_blocks = (numel + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE;

    // Get raw data
    let data = flat.to_vec1::<f32>()?;

    // Allocate output buffers
    let mut packed_values: Vec<u8> = Vec::with_capacity(num_blocks * Q4_0_BLOCK_SIZE / 2);
    let mut scales: Vec<f32> = Vec::with_capacity(num_blocks);

    for block_idx in 0..num_blocks {
        let start = block_idx * Q4_0_BLOCK_SIZE;
        let end = (start + Q4_0_BLOCK_SIZE).min(numel);

        // Find max abs value in block for scale
        let max_abs = data[start..end]
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 {
            max_abs / 7.0 // Q4 range is -7 to +7 (we use 3 bits for magnitude + 1 for sign)
        } else {
            1.0
        };
        scales.push(scale);

        // Quantize and pack values
        let inv_scale = 1.0 / scale;
        for i in (0..Q4_0_BLOCK_SIZE).step_by(2) {
            let v0_idx = start + i;
            let v1_idx = start + i + 1;

            // Quantize first value
            let v0 = if v0_idx < numel {
                let q = (data[v0_idx] * inv_scale).round().clamp(-7.0, 7.0) as i8;
                (q + 8) as u8 & 0x0F // Shift to 0-15 range, keep lower 4 bits
            } else {
                0u8
            };

            // Quantize second value
            let v1 = if v1_idx < numel {
                let q = (data[v1_idx] * inv_scale).round().clamp(-7.0, 7.0) as i8;
                ((q + 8) as u8 & 0x0F) << 4 // Shift to 0-15 range, put in upper 4 bits
            } else {
                0u8
            };

            packed_values.push(v0 | v1);
        }
    }

    let values_len = packed_values.len();
    let values_tensor = Tensor::from_vec(packed_values, (values_len,), device)?;
    let scales_tensor = Tensor::from_vec(scales, (num_blocks,), device)?;

    Ok((values_tensor, scales_tensor, target_dtype))
}

/// Dequantize Q4_0 format back to dense tensor.
fn dequantize_tensor_q4_0(
    values: &Tensor,
    scales: &Tensor,
    num_elements: usize,
    target_dtype: DType,
) -> Result<Tensor> {
    let device = values.device();
    if num_elements == 0 {
        return Ok(Tensor::zeros(0, target_dtype, device).map_err(Error::from)?);
    }

    let packed = values.to_vec1::<u8>()?;
    let scales_f32 = scales.to_vec1::<f32>()?;
    let expected_num_blocks = (num_elements + Q4_0_BLOCK_SIZE - 1) / Q4_0_BLOCK_SIZE;
    let expected_packed_len = expected_num_blocks * (Q4_0_BLOCK_SIZE / 2);

    if scales_f32.len() != expected_num_blocks {
        return Err(Error::InvalidInput(format!(
            "Q4_0 scale length mismatch: expected {}, got {}",
            expected_num_blocks,
            scales_f32.len()
        )));
    }
    if packed.len() != expected_packed_len {
        return Err(Error::InvalidInput(format!(
            "Q4_0 packed length mismatch: expected {}, got {}",
            expected_packed_len,
            packed.len()
        )));
    }

    let mut dequantized: Vec<f32> = Vec::with_capacity(num_elements);

    for block_idx in 0..expected_num_blocks {
        let scale = scales_f32[block_idx];
        let packed_offset = block_idx * (Q4_0_BLOCK_SIZE / 2);
        for i in 0..Q4_0_BLOCK_SIZE {
            let out_idx = block_idx * Q4_0_BLOCK_SIZE + i;
            if out_idx >= num_elements {
                break;
            }
            let packed_idx = packed_offset + i / 2;
            let byte = packed[packed_idx];

            // Extract 4-bit value
            let q = if i % 2 == 0 {
                (byte & 0x0F) as i8
            } else {
                ((byte >> 4) & 0x0F) as i8
            };

            // Convert from unsigned 0-15 to signed -7 to +7
            let q_signed = q as f32 - 8.0;
            let v = q_signed * scale;
            dequantized.push(v);
        }
    }

    if dequantized.len() != num_elements {
        return Err(Error::InvalidInput(format!(
            "Q4_0 dequantization length mismatch: expected {}, got {}",
            num_elements,
            dequantized.len()
        )));
    }

    let dequantized_tensor = Tensor::from_vec(dequantized, (num_elements,), device)?;
    dequantized_tensor
        .to_dtype(target_dtype)
        .map_err(Error::from)
}

/// Append a `[batch, seq, heads, dim]` tensor into fixed-size pages along `seq`.
///
/// The last existing page is filled first (if not full), then new pages are pushed.
pub fn append_to_pages(
    page_size: usize,
    pages: &mut Vec<KvPage>,
    append: &Tensor,
    quantization: KvCacheQuantization,
) -> Result<()> {
    let quantization = resolve_kv_cache_quantization(append.device(), quantization)?;
    if page_size == 0 {
        return Err(Error::InvalidInput(
            "KV page size must be greater than zero".to_string(),
        ));
    }
    let seq_len = append.dim(1)?;
    if seq_len == 0 {
        return Ok(());
    }

    let mut cursor = 0usize;
    while cursor < seq_len {
        let mut consumed = false;
        if let Some(last) = pages.last_mut() {
            let last_len = last.seq_len()?;
            if last_len < page_size {
                let take = (page_size - last_len).min(seq_len - cursor);
                let chunk = append.narrow(1, cursor, take)?;
                let last_dense = last.to_dense()?;
                let merged = Tensor::cat(&[&last_dense, &chunk], 1)?;
                *last = KvPage::from_dense(merged, quantization)?;
                cursor += take;
                consumed = true;
            }
        }

        if consumed {
            continue;
        }

        let take = page_size.min(seq_len - cursor);
        let chunk = append.narrow(1, cursor, take)?;
        pages.push(KvPage::from_dense(chunk, quantization)?);
        cursor += take;
    }
    Ok(())
}

/// Materialize paged tensors into a single contiguous `[batch, total_seq, heads, dim]` tensor.
pub fn materialize_pages(pages: &[KvPage]) -> Result<Tensor> {
    if pages.is_empty() {
        return Err(Error::InferenceError(
            "Attempted to materialize empty KV pages".to_string(),
        ));
    }
    if pages.len() == 1 {
        return pages[0].to_dense();
    }
    let mut dense_pages = Vec::with_capacity(pages.len());
    for page in pages {
        dense_pages.push(page.to_dense()?);
    }
    let refs: Vec<&Tensor> = dense_pages.iter().collect();
    Tensor::cat(&refs, 1).map_err(Error::from)
}

pub fn repeat_kv(x: &Tensor, num_heads: usize, num_kv_heads: usize) -> Result<Tensor> {
    if num_heads == num_kv_heads {
        return Ok(x.clone());
    }
    let repeats = num_heads / num_kv_heads;
    let (batch, seq_len, n_kv_heads, head_dim) = x.dims4()?;
    let x = x
        .unsqueeze(3)?
        .expand((batch, seq_len, n_kv_heads, repeats, head_dim))?
        .reshape((batch, seq_len, num_heads, head_dim))?;
    Ok(x)
}

fn parse_env_positive_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn paged_decode_page_group_size(device: &candle_core::Device, page_count: usize) -> usize {
    let override_pages = parse_env_positive_usize("IZWI_PAGED_DECODE_GROUP_PAGES");
    paged_decode_page_group_size_policy(
        device.is_metal(),
        device.is_cuda(),
        page_count,
        override_pages,
    )
}

fn paged_decode_page_group_size_policy(
    is_metal: bool,
    is_cuda: bool,
    page_count: usize,
    override_pages: Option<usize>,
) -> usize {
    if let Some(override_pages) = override_pages {
        return override_pages.max(1);
    }
    if is_cuda {
        if page_count >= 64 {
            return 16;
        }
        if page_count >= 24 {
            return 8;
        }
        if page_count >= 8 {
            return 4;
        }
        return 1;
    }
    if !is_metal || page_count < 8 {
        return 1;
    }
    if page_count >= 64 {
        return 8;
    }
    if page_count >= 24 {
        return 4;
    }
    2
}

fn paged_decode_attention_with_group_size(
    q: &Tensor,
    k_pages: &[KvPage],
    v_pages: &[KvPage],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    page_group_size: usize,
) -> Result<Tensor> {
    let page_group_size = page_group_size.max(1);
    let scale = (head_dim as f64).sqrt();
    let batch_size = q.dim(0)?;
    let q_heads = q.transpose(1, 2)?.contiguous()?; // [bs, h, 1, d]
    let q_flat = q_heads.reshape((batch_size * num_heads, 1, head_dim))?; // [bs*h, 1, d]

    let mut running_max: Option<Tensor> = None; // [bs*h, 1, 1]
    let mut running_sum: Option<Tensor> = None; // [bs*h, 1, 1]
    let mut running_out: Option<Tensor> = None; // [bs*h, 1, d]

    for page_start in (0..k_pages.len()).step_by(page_group_size) {
        let page_end = (page_start + page_group_size).min(k_pages.len());
        let k_page = materialize_pages(&k_pages[page_start..page_end])?;
        let v_page = materialize_pages(&v_pages[page_start..page_end])?;
        let page_len = k_page.dim(1)?;
        if page_len == 0 {
            continue;
        }

        let (k_heads, v_heads) = if num_heads == num_kv_heads {
            (
                k_page.transpose(1, 2)?.contiguous()?,
                v_page.transpose(1, 2)?.contiguous()?,
            )
        } else {
            (
                repeat_kv(&k_page, num_heads, num_kv_heads)?
                    .transpose(1, 2)?
                    .contiguous()?,
                repeat_kv(&v_page, num_heads, num_kv_heads)?
                    .transpose(1, 2)?
                    .contiguous()?,
            )
        };
        let k_flat = k_heads.reshape((batch_size * num_heads, page_len, head_dim))?;
        let v_flat = v_heads.reshape((batch_size * num_heads, page_len, head_dim))?;

        let mut scores = q_flat.matmul(&k_flat.transpose(1, 2)?)?;
        scores = (scores / scale)?;
        let page_max = scores.max_keepdim(D::Minus1)?;
        let exp_scores = scores.broadcast_sub(&page_max)?.exp()?;
        let page_sum = exp_scores.sum_keepdim(D::Minus1)?;
        let page_out = exp_scores.matmul(&v_flat)?;

        match (&running_max, &running_sum, &running_out) {
            (None, None, None) => {
                running_max = Some(page_max);
                running_sum = Some(page_sum);
                running_out = Some(page_out);
            }
            (Some(cur_max), Some(cur_sum), Some(cur_out)) => {
                let new_max = cur_max.broadcast_maximum(&page_max)?;
                let cur_scale = cur_max.broadcast_sub(&new_max)?.exp()?;
                let page_scale = page_max.broadcast_sub(&new_max)?.exp()?;

                let new_sum = cur_sum
                    .broadcast_mul(&cur_scale)?
                    .broadcast_add(&page_sum.broadcast_mul(&page_scale)?)?;
                let new_out = cur_out
                    .broadcast_mul(&cur_scale)?
                    .broadcast_add(&page_out.broadcast_mul(&page_scale)?)?;

                running_max = Some(new_max);
                running_sum = Some(new_sum);
                running_out = Some(new_out);
            }
            _ => {
                return Err(Error::InferenceError(
                    "Paged decode attention entered inconsistent running state".to_string(),
                ));
            }
        }
    }

    let running_sum = running_sum.ok_or_else(|| {
        Error::InferenceError("Paged decode attention produced no valid page sum".to_string())
    })?;
    let running_out = running_out.ok_or_else(|| {
        Error::InferenceError("Paged decode attention produced no valid page output".to_string())
    })?;

    let out = running_out.broadcast_div(&running_sum)?;
    out.reshape((batch_size, num_heads, 1, head_dim))?
        .transpose(1, 2)
        .map_err(Error::from)
}

/// Compute exact single-token attention over paged K/V without materializing full K/V.
///
/// `q` is `[batch, 1, heads, head_dim]` and page tensors are `[batch, page_seq, kv_heads, head_dim]`.
/// Returns `[batch, 1, heads, head_dim]`.
pub fn paged_decode_attention(
    q: &Tensor,
    k_pages: &[KvPage],
    v_pages: &[KvPage],
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    if k_pages.is_empty() || v_pages.is_empty() || k_pages.len() != v_pages.len() {
        return Err(Error::InferenceError(
            "Paged decode attention received invalid KV pages".to_string(),
        ));
    }
    let q_len = q.dim(1)?;
    if q_len != 1 {
        return Err(Error::InvalidInput(format!(
            "Paged decode attention expects q_len=1, got {}",
            q_len
        )));
    }
    if num_kv_heads == 0 || num_heads == 0 || !num_heads.is_multiple_of(num_kv_heads) {
        return Err(Error::InvalidInput(format!(
            "Paged decode attention received invalid head layout: num_heads={}, num_kv_heads={}",
            num_heads, num_kv_heads
        )));
    }
    record_decode_attention_path(DecodeAttentionPath::Paged);
    if q.device().is_cuda() {
        if let Some(out) = crate::kernels::cuda::try_cuda_paged_decode_attention(
            q,
            head_dim,
            num_heads,
            num_kv_heads,
        )? {
            return Ok(out);
        }
    }
    let page_group_size = paged_decode_page_group_size(q.device(), k_pages.len());
    paged_decode_attention_with_group_size(
        q,
        k_pages,
        v_pages,
        num_heads,
        num_kv_heads,
        head_dim,
        page_group_size,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::ops;

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = b
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        a.iter()
            .zip(b.iter())
            .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    #[test]
    fn test_append_to_pages_respects_page_size() {
        let device = Device::Cpu;
        let mut pages = Vec::new();
        let tensor = Tensor::randn(0.0f32, 1.0f32, (1, 10, 2, 4), &device).unwrap();
        append_to_pages(4, &mut pages, &tensor, KvCacheQuantization::None).unwrap();
        assert_eq!(pages.len(), 3);
        assert_eq!(pages[0].seq_len().unwrap(), 4);
        assert_eq!(pages[1].seq_len().unwrap(), 4);
        assert_eq!(pages[2].seq_len().unwrap(), 2);

        let next = Tensor::randn(0.0f32, 1.0f32, (1, 3, 2, 4), &device).unwrap();
        append_to_pages(4, &mut pages, &next, KvCacheQuantization::None).unwrap();
        assert_eq!(pages.len(), 4);
        assert_eq!(pages[2].seq_len().unwrap(), 4);
        assert_eq!(pages[3].seq_len().unwrap(), 1);
    }

    #[test]
    fn test_quantized_materialize_close_to_dense() {
        let device = Device::Cpu;
        let full = Tensor::randn(0.0f32, 1.0f32, (1, 17, 4, 8), &device).unwrap();
        let mut pages = Vec::new();
        append_to_pages(5, &mut pages, &full, KvCacheQuantization::Int8).unwrap();
        let materialized = materialize_pages(&pages).unwrap();
        let diff = max_abs_diff(&materialized, &full);
        assert!(diff < 0.08, "max abs diff was {}", diff);
    }

    #[test]
    fn test_q4_0_quantized_materialize_close_to_dense() {
        let device = Device::Cpu;
        let full = Tensor::randn(0.0f32, 1.0f32, (1, 17, 4, 8), &device).unwrap();
        let mut pages = Vec::new();
        append_to_pages(5, &mut pages, &full, KvCacheQuantization::Q4_0).unwrap();
        let materialized = materialize_pages(&pages).unwrap();
        let diff = max_abs_diff(&materialized, &full);
        assert!(diff < 0.20, "max abs diff was {}", diff);

        assert_eq!(pages.len(), 4, "unexpected page count");
        assert_eq!(pages[0].seq_len().unwrap(), 5);
        assert_eq!(pages[1].seq_len().unwrap(), 5);
        assert_eq!(pages[2].seq_len().unwrap(), 5);
        assert_eq!(pages[3].seq_len().unwrap(), 2);
    }

    #[test]
    fn test_paged_decode_matches_dense_single_token() {
        let device = Device::Cpu;
        let bsz = 2usize;
        let num_heads = 4usize;
        let head_dim = 8usize;
        let total_len = 11usize;

        let q = Tensor::randn(0.0f32, 1.0f32, (bsz, 1, num_heads, head_dim), &device).unwrap();
        let k_full = Tensor::randn(
            0.0f32,
            1.0f32,
            (bsz, total_len, num_heads, head_dim),
            &device,
        )
        .unwrap();
        let v_full = Tensor::randn(
            0.0f32,
            1.0f32,
            (bsz, total_len, num_heads, head_dim),
            &device,
        )
        .unwrap();

        let mut k_pages = Vec::new();
        let mut v_pages = Vec::new();
        append_to_pages(3, &mut k_pages, &k_full, KvCacheQuantization::None).unwrap();
        append_to_pages(3, &mut v_pages, &v_full, KvCacheQuantization::None).unwrap();

        let paged =
            paged_decode_attention(&q, &k_pages, &v_pages, num_heads, num_heads, head_dim).unwrap();

        // Dense reference implementation.
        let q_ref = q
            .transpose(1, 2)
            .unwrap()
            .reshape((bsz * num_heads, 1, head_dim))
            .unwrap();
        let k_ref = k_full
            .transpose(1, 2)
            .unwrap()
            .reshape((bsz * num_heads, total_len, head_dim))
            .unwrap();
        let v_ref = v_full
            .transpose(1, 2)
            .unwrap()
            .reshape((bsz * num_heads, total_len, head_dim))
            .unwrap();
        let scale = (head_dim as f64).sqrt();
        let mut scores = q_ref.matmul(&k_ref.transpose(1, 2).unwrap()).unwrap();
        let scale_t = Tensor::from_vec(vec![scale as f32], (1,), &device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        scores = scores.broadcast_div(&scale_t).unwrap();
        let weights = ops::softmax(&scores, D::Minus1).unwrap();
        let dense = weights
            .matmul(&v_ref)
            .unwrap()
            .reshape((bsz, num_heads, 1, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();

        let diff = max_abs_diff(&paged, &dense);
        assert!(diff < 1e-4, "max abs diff was {}", diff);
    }

    #[test]
    fn test_quantized_paged_decode_matches_dense_single_token() {
        let device = Device::Cpu;
        let bsz = 2usize;
        let num_heads = 4usize;
        let head_dim = 8usize;
        let total_len = 13usize;

        let q = Tensor::randn(0.0f32, 1.0f32, (bsz, 1, num_heads, head_dim), &device).unwrap();
        let k_full = Tensor::randn(
            0.0f32,
            1.0f32,
            (bsz, total_len, num_heads, head_dim),
            &device,
        )
        .unwrap();
        let v_full = Tensor::randn(
            0.0f32,
            1.0f32,
            (bsz, total_len, num_heads, head_dim),
            &device,
        )
        .unwrap();

        let mut k_pages = Vec::new();
        let mut v_pages = Vec::new();
        append_to_pages(4, &mut k_pages, &k_full, KvCacheQuantization::Int8).unwrap();
        append_to_pages(4, &mut v_pages, &v_full, KvCacheQuantization::Int8).unwrap();

        let paged =
            paged_decode_attention(&q, &k_pages, &v_pages, num_heads, num_heads, head_dim).unwrap();

        // Dense reference implementation.
        let q_ref = q
            .transpose(1, 2)
            .unwrap()
            .reshape((bsz * num_heads, 1, head_dim))
            .unwrap();
        let k_ref = k_full
            .transpose(1, 2)
            .unwrap()
            .reshape((bsz * num_heads, total_len, head_dim))
            .unwrap();
        let v_ref = v_full
            .transpose(1, 2)
            .unwrap()
            .reshape((bsz * num_heads, total_len, head_dim))
            .unwrap();
        let scale = (head_dim as f64).sqrt();
        let mut scores = q_ref.matmul(&k_ref.transpose(1, 2).unwrap()).unwrap();
        let scale_t = Tensor::from_vec(vec![scale as f32], (1,), &device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        scores = scores.broadcast_div(&scale_t).unwrap();
        let weights = ops::softmax(&scores, D::Minus1).unwrap();
        let dense = weights
            .matmul(&v_ref)
            .unwrap()
            .reshape((bsz, num_heads, 1, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();

        let diff = max_abs_diff(&paged, &dense);
        assert!(diff < 0.12, "max abs diff was {}", diff);
    }

    #[test]
    fn test_grouped_paged_decode_matches_single_page_loop() {
        let device = Device::Cpu;
        let bsz = 1usize;
        let num_heads = 8usize;
        let num_kv_heads = 2usize;
        let head_dim = 8usize;
        let total_len = 23usize;

        let q = Tensor::randn(0.0f32, 1.0f32, (bsz, 1, num_heads, head_dim), &device).unwrap();
        let k_full = Tensor::randn(
            0.0f32,
            1.0f32,
            (bsz, total_len, num_kv_heads, head_dim),
            &device,
        )
        .unwrap();
        let v_full = Tensor::randn(
            0.0f32,
            1.0f32,
            (bsz, total_len, num_kv_heads, head_dim),
            &device,
        )
        .unwrap();

        let mut k_pages = Vec::new();
        let mut v_pages = Vec::new();
        append_to_pages(3, &mut k_pages, &k_full, KvCacheQuantization::None).unwrap();
        append_to_pages(3, &mut v_pages, &v_full, KvCacheQuantization::None).unwrap();

        let single = paged_decode_attention_with_group_size(
            &q,
            &k_pages,
            &v_pages,
            num_heads,
            num_kv_heads,
            head_dim,
            1,
        )
        .unwrap();
        let grouped = paged_decode_attention_with_group_size(
            &q,
            &k_pages,
            &v_pages,
            num_heads,
            num_kv_heads,
            head_dim,
            4,
        )
        .unwrap();

        let diff = max_abs_diff(&single, &grouped);
        assert!(diff < 1e-4, "max abs diff was {}", diff);
    }

    #[test]
    fn test_paged_decode_page_group_policy_scales_on_metal() {
        assert_eq!(paged_decode_page_group_size_policy(true, false, 1, None), 1);
        assert_eq!(paged_decode_page_group_size_policy(true, false, 7, None), 1);
        assert_eq!(paged_decode_page_group_size_policy(true, false, 8, None), 2);
        assert_eq!(paged_decode_page_group_size_policy(true, false, 23, None), 2);
        assert_eq!(paged_decode_page_group_size_policy(true, false, 24, None), 4);
        assert_eq!(paged_decode_page_group_size_policy(true, false, 63, None), 4);
        assert_eq!(paged_decode_page_group_size_policy(true, false, 64, None), 8);
    }

    #[test]
    fn test_paged_decode_page_group_policy_scales_on_cuda() {
        assert_eq!(paged_decode_page_group_size_policy(false, true, 1, None), 1);
        assert_eq!(paged_decode_page_group_size_policy(false, true, 7, None), 1);
        assert_eq!(paged_decode_page_group_size_policy(false, true, 8, None), 4);
        assert_eq!(paged_decode_page_group_size_policy(false, true, 23, None), 4);
        assert_eq!(paged_decode_page_group_size_policy(false, true, 24, None), 8);
        assert_eq!(paged_decode_page_group_size_policy(false, true, 63, None), 8);
        assert_eq!(paged_decode_page_group_size_policy(false, true, 64, None), 16);
    }

    #[test]
    fn test_paged_decode_page_group_policy_respects_overrides() {
        assert_eq!(paged_decode_page_group_size_policy(false, false, 128, None), 1);
        assert_eq!(paged_decode_page_group_size_policy(true, false, 128, Some(3)), 3);
        assert_eq!(paged_decode_page_group_size_policy(false, true, 128, Some(5)), 5);
    }

    #[test]
    fn test_kv_quantization_policy_blocks_implicit_cuda_q4_host_fallback() {
        assert!(kv_cache_quantization_policy(false, KvCacheQuantization::Q4_0, false).is_ok());
        assert!(kv_cache_quantization_policy(true, KvCacheQuantization::None, false).is_ok());
        assert!(kv_cache_quantization_policy(true, KvCacheQuantization::Int8, false).is_ok());
        assert!(kv_cache_quantization_policy(true, KvCacheQuantization::Q4_0, false).is_err());
    }

    #[test]
    fn test_kv_quantization_policy_allows_explicit_cuda_q4_host_fallback() {
        let selected =
            kv_cache_quantization_policy(true, KvCacheQuantization::Q4_0, true).unwrap();
        assert_eq!(selected, KvCacheQuantization::Q4_0);
    }

    #[test]
    fn test_paged_decode_gqa_matches_dense_single_token() {
        let device = Device::Cpu;
        let bsz = 2usize;
        let num_heads = 8usize;
        let num_kv_heads = 2usize;
        let head_dim = 8usize;
        let total_len = 11usize;

        let q = Tensor::randn(0.0f32, 1.0f32, (bsz, 1, num_heads, head_dim), &device).unwrap();
        let k_full = Tensor::randn(
            0.0f32,
            1.0f32,
            (bsz, total_len, num_kv_heads, head_dim),
            &device,
        )
        .unwrap();
        let v_full = Tensor::randn(
            0.0f32,
            1.0f32,
            (bsz, total_len, num_kv_heads, head_dim),
            &device,
        )
        .unwrap();

        let mut k_pages = Vec::new();
        let mut v_pages = Vec::new();
        append_to_pages(3, &mut k_pages, &k_full, KvCacheQuantization::None).unwrap();
        append_to_pages(3, &mut v_pages, &v_full, KvCacheQuantization::None).unwrap();

        let paged =
            paged_decode_attention(&q, &k_pages, &v_pages, num_heads, num_kv_heads, head_dim)
                .unwrap();

        let q_ref = q
            .transpose(1, 2)
            .unwrap()
            .reshape((bsz * num_heads, 1, head_dim))
            .unwrap();
        let k_ref = repeat_kv(&k_full, num_heads, num_kv_heads)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .reshape((bsz * num_heads, total_len, head_dim))
            .unwrap();
        let v_ref = repeat_kv(&v_full, num_heads, num_kv_heads)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .reshape((bsz * num_heads, total_len, head_dim))
            .unwrap();
        let scale = (head_dim as f64).sqrt();
        let mut scores = q_ref.matmul(&k_ref.transpose(1, 2).unwrap()).unwrap();
        let scale_t = Tensor::from_vec(vec![scale as f32], (1,), &device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        scores = scores.broadcast_div(&scale_t).unwrap();
        let weights = ops::softmax(&scores, D::Minus1).unwrap();
        let dense = weights
            .matmul(&v_ref)
            .unwrap()
            .reshape((bsz, num_heads, 1, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap();

        let diff = max_abs_diff(&paged, &dense);
        assert!(diff < 1e-4, "max abs diff was {}", diff);
    }
}
