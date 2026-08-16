//! Portable scaled dot-product attention for compact GQA tensors.
//!
//! Accelerator-specific fused attention should be attempted before this
//! helper. Unlike the compatibility pattern of repeating K/V heads until they
//! match the query-head count, this fallback keeps K/V compact. Query heads
//! are grouped into the row dimension of a batched matrix multiplication so a
//! single compact K/V head serves every query head in its group.

use candle_core::{Tensor, D};

use crate::error::{Error, Result};

/// Mask forms used by compact GQA attention.
#[derive(Clone, Copy)]
pub(crate) enum CompactGqaMask<'a> {
    /// An additive attention mask, normally zero for visible positions and
    /// negative infinity for masked positions.
    Additive(&'a Tensor),
    /// A boolean/U8 mask where true selects `masked_value`.
    Boolean {
        mask: &'a Tensor,
        masked_value: &'a Tensor,
    },
}

/// Compute scaled dot-product attention without materializing repeated K/V.
///
/// Inputs and output use `[batch, heads, sequence, dimension]` (BHSD). Query
/// heads must be an integer multiple of KV heads. Masks may be `[q, k]`,
/// `[batch, q, k]`, `[batch, 1|kv_heads|query_heads, q, k]`, or the internal
/// grouped shape `[batch, kv_heads, groups, q, k]`.
pub(crate) fn compact_gqa_sdpa_bhsd(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<CompactGqaMask<'_>>,
    softmax_scale: f64,
) -> Result<Tensor> {
    let (batch, query_heads, query_len, query_dim) = query.dims4()?;
    let (key_batch, kv_heads, key_len, key_dim) = key.dims4()?;
    let (value_batch, value_heads, value_len, value_dim) = value.dims4()?;

    if batch == 0 || query_heads == 0 || query_len == 0 || query_dim == 0 {
        return Err(Error::InvalidInput(format!(
            "compact GQA query dimensions must be non-zero, got {:?}",
            query.dims()
        )));
    }
    if key_batch != batch || value_batch != batch {
        return Err(Error::InvalidInput(format!(
            "compact GQA batch mismatch: q={batch}, k={key_batch}, v={value_batch}"
        )));
    }
    if kv_heads == 0 || value_heads != kv_heads || !query_heads.is_multiple_of(kv_heads) {
        return Err(Error::InvalidInput(format!(
            "invalid compact GQA head geometry: query_heads={query_heads}, key_heads={kv_heads}, value_heads={value_heads}"
        )));
    }
    if key_len == 0 || value_len != key_len {
        return Err(Error::InvalidInput(format!(
            "compact GQA sequence mismatch: key_len={key_len}, value_len={value_len}"
        )));
    }
    if key_dim != query_dim || value_dim == 0 {
        return Err(Error::InvalidInput(format!(
            "compact GQA dimension mismatch: query_dim={query_dim}, key_dim={key_dim}, value_dim={value_dim}"
        )));
    }
    if !softmax_scale.is_finite() || softmax_scale <= 0.0 {
        return Err(Error::InvalidInput(format!(
            "compact GQA softmax scale must be finite and positive, got {softmax_scale}"
        )));
    }

    let groups = query_heads / kv_heads;

    // A contiguous Q-head ordering maps heads [g * groups, (g + 1) * groups)
    // to KV head g, which is the same mapping produced by repeat_interleave.
    // Folding the group and query sequence into the GEMM row dimension lets
    // every compact KV head be consumed once without a broadcasted K/V copy.
    let query = query
        .contiguous()?
        .reshape((batch * kv_heads, groups * query_len, query_dim))?;
    let key = key
        .contiguous()?
        .reshape((batch * kv_heads, key_len, key_dim))?;
    let value = value
        .contiguous()?
        .reshape((batch * kv_heads, key_len, value_dim))?;

    let scores = (query.matmul(&key.transpose(1, 2)?.contiguous()?)? * softmax_scale)?
        .reshape((batch, kv_heads, groups, query_len, key_len))?;
    let scores = match mask {
        None => scores,
        Some(CompactGqaMask::Additive(mask)) => {
            let mask = normalize_mask(
                mask,
                batch,
                query_heads,
                kv_heads,
                groups,
                query_len,
                key_len,
            )?;
            scores.broadcast_add(&mask)?
        }
        Some(CompactGqaMask::Boolean { mask, masked_value }) => {
            let mask = normalize_mask(
                mask,
                batch,
                query_heads,
                kv_heads,
                groups,
                query_len,
                key_len,
            )?
            .broadcast_as(scores.shape())?;
            mask.where_cond(&masked_value.broadcast_as(scores.shape().dims())?, &scores)?
        }
    };

    let probabilities = candle_nn::ops::softmax(&scores, D::Minus1)?.reshape((
        batch * kv_heads,
        groups * query_len,
        key_len,
    ))?;
    probabilities
        .contiguous()?
        .matmul(&value)?
        .reshape((batch, query_heads, query_len, value_dim))
        .map_err(Error::from)
}

fn normalize_mask(
    mask: &Tensor,
    batch: usize,
    query_heads: usize,
    kv_heads: usize,
    groups: usize,
    query_len: usize,
    key_len: usize,
) -> Result<Tensor> {
    let invalid = || {
        Error::InvalidInput(format!(
            "compact GQA mask shape {:?} is incompatible with batch={batch}, query_heads={query_heads}, kv_heads={kv_heads}, query_len={query_len}, key_len={key_len}",
            mask.dims()
        ))
    };

    let normalized = match mask.rank() {
        2 => {
            let (mask_query, mask_key) = mask.dims2()?;
            if mask_query != query_len || mask_key != key_len {
                return Err(invalid());
            }
            mask.unsqueeze(0)?.unsqueeze(0)?.unsqueeze(0)?
        }
        3 => {
            let (mask_batch, mask_query, mask_key) = mask.dims3()?;
            if !(mask_batch == 1 || mask_batch == batch)
                || mask_query != query_len
                || mask_key != key_len
            {
                return Err(invalid());
            }
            mask.unsqueeze(1)?.unsqueeze(1)?
        }
        4 => {
            let (mask_batch, mask_heads, mask_query, mask_key) = mask.dims4()?;
            if !(mask_batch == 1 || mask_batch == batch)
                || mask_query != query_len
                || mask_key != key_len
            {
                return Err(invalid());
            }
            if mask_heads == 1 || mask_heads == kv_heads {
                mask.unsqueeze(2)?
            } else if mask_heads == query_heads {
                mask.contiguous()?
                    .reshape((mask_batch, kv_heads, groups, query_len, key_len))?
            } else {
                return Err(invalid());
            }
        }
        5 => {
            let dims = mask.dims();
            if !(dims[0] == 1 || dims[0] == batch)
                || !(dims[1] == 1 || dims[1] == kv_heads)
                || !(dims[2] == 1 || dims[2] == groups)
                || dims[3] != query_len
                || dims[4] != key_len
            {
                return Err(invalid());
            }
            mask.clone()
        }
        _ => return Err(invalid()),
    };
    Ok(normalized)
}

#[cfg(test)]
mod tests {
    use super::{compact_gqa_sdpa_bhsd, CompactGqaMask};
    use candle_core::{Device, Tensor, D};

    fn repeat_heads(tensor: &Tensor, query_heads: usize) -> candle_core::Result<Tensor> {
        let kv_heads = tensor.dim(1)?;
        let groups = query_heads / kv_heads;
        let mut heads = Vec::with_capacity(query_heads);
        for kv_head in 0..kv_heads {
            let head = tensor.narrow(1, kv_head, 1)?;
            for _ in 0..groups {
                heads.push(head.clone());
            }
        }
        Tensor::cat(&heads.iter().collect::<Vec<_>>(), 1)
    }

    fn dense_repeat_oracle(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        additive_mask: Option<&Tensor>,
        boolean_mask: Option<(&Tensor, &Tensor)>,
        scale: f64,
    ) -> candle_core::Result<Tensor> {
        let query_heads = query.dim(1)?;
        let key = repeat_heads(key, query_heads)?;
        let value = repeat_heads(value, query_heads)?;
        let mut scores = (query
            .contiguous()?
            .matmul(&key.transpose(2, 3)?.contiguous()?)?
            * scale)?;
        if let Some(mask) = additive_mask {
            scores = scores.broadcast_add(mask)?;
        }
        if let Some((mask, masked_value)) = boolean_mask {
            let mask = mask.broadcast_as(scores.shape())?;
            scores =
                mask.where_cond(&masked_value.broadcast_as(scores.shape().dims())?, &scores)?;
        }
        candle_nn::ops::softmax(&scores, D::Minus1)?
            .contiguous()?
            .matmul(&value.contiguous()?)
    }

    fn assert_close(actual: &Tensor, expected: &Tensor) {
        let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "element {index}: compact={actual}, repeated={expected}"
            );
        }
    }

    #[test]
    fn compact_gqa_matches_repeated_kv_with_additive_mask_on_cpu() {
        let device = Device::Cpu;
        let query = Tensor::from_iter((0..2 * 4 * 3 * 2).map(|v| v as f32 / 17.0), &device)
            .unwrap()
            .reshape((2, 4, 3, 2))
            .unwrap();
        let key = Tensor::from_iter((0..2 * 2 * 3 * 2).map(|v| (v as f32 - 7.0) / 13.0), &device)
            .unwrap()
            .reshape((2, 2, 3, 2))
            .unwrap();
        let value = Tensor::from_iter((0..2 * 2 * 3 * 3).map(|v| v as f32 / 11.0), &device)
            .unwrap()
            .reshape((2, 2, 3, 3))
            .unwrap();
        let mask = Tensor::from_slice(
            &[
                0.0f32,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                0.0,
            ],
            (3, 3),
            &device,
        )
        .unwrap();
        let scale = 1.0 / 2.0f64.sqrt();

        let compact = compact_gqa_sdpa_bhsd(
            &query,
            &key,
            &value,
            Some(CompactGqaMask::Additive(&mask)),
            scale,
        )
        .unwrap();
        let repeated = dense_repeat_oracle(&query, &key, &value, Some(&mask), None, scale).unwrap();
        assert_close(&compact, &repeated);
    }

    #[test]
    fn compact_gqa_matches_repeated_kv_with_boolean_mask_on_cpu() {
        let device = Device::Cpu;
        let query = Tensor::from_iter((0..6 * 3 * 2).map(|v| (v as f32 - 9.0) / 7.0), &device)
            .unwrap()
            .reshape((1, 6, 3, 2))
            .unwrap();
        let key = Tensor::from_iter((0..2 * 3 * 2).map(|v| v as f32 / 5.0), &device)
            .unwrap()
            .reshape((1, 2, 3, 2))
            .unwrap();
        let value = Tensor::from_iter((0..2 * 3 * 2).map(|v| (v as f32 - 4.0) / 3.0), &device)
            .unwrap()
            .reshape((1, 2, 3, 2))
            .unwrap();
        let mask = Tensor::from_slice(&[0u8, 1, 1, 0, 0, 1, 0, 0, 0], (3, 3), &device).unwrap();
        let masked_value = Tensor::new(f32::NEG_INFINITY, &device).unwrap();
        let scale = 1.0 / 2.0f64.sqrt();

        let compact = compact_gqa_sdpa_bhsd(
            &query,
            &key,
            &value,
            Some(CompactGqaMask::Boolean {
                mask: &mask,
                masked_value: &masked_value,
            }),
            scale,
        )
        .unwrap();
        let repeated = dense_repeat_oracle(
            &query,
            &key,
            &value,
            None,
            Some((&mask, &masked_value)),
            scale,
        )
        .unwrap();
        assert_close(&compact, &repeated);
    }

    #[test]
    fn compact_gqa_rejects_non_integral_head_groups() {
        let query = Tensor::zeros((1, 3, 2, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        let key = Tensor::zeros((1, 2, 2, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        let value = key.clone();
        let error = compact_gqa_sdpa_bhsd(&query, &key, &value, None, 0.5).unwrap_err();
        assert!(error
            .to_string()
            .contains("invalid compact GQA head geometry"));
    }
}
