//! Quantized projections assembled from GGUF tensors as stored.
//!
//! A family that fuses projections (q/k/v, gate/up) joins the rows of
//! several GGUF tensors into one `QMatMul`, and llama.cpp's conversion of a
//! Llama-style model reorders each head's Q and K rows for interleaved RoPE
//! (`permute` in `convert_hf_to_gguf.py`), which a half-split RoPE must put
//! back. Rows are whole quantized blocks, so both are byte copies: nothing
//! is dequantized.

use std::borrow::Cow;
use std::ops::Range;

use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::Device;

use super::gguf::GgufLoader;
use crate::error::{Error, Result};

/// One tensor joined into a projection: its GGUF name and, for Q and K, the
/// heads whose rows llama.cpp reordered for interleaved RoPE.
pub(crate) struct QuantizedPart<'a> {
    pub(crate) name: &'a str,
    pub(crate) rope_heads: Option<usize>,
}

/// The parts' rows joined, in order, into one quantized projection on
/// `device`, with Q and K rows put back in half-split RoPE order.
pub(crate) fn joined_qmatmul(
    gguf: &GgufLoader,
    parts: &[QuantizedPart<'_>],
    device: &Device,
) -> Result<QMatMul> {
    QMatMul::from_qtensor(joined_qtensor(gguf, parts, device)?).map_err(Error::from)
}

/// [`joined_qmatmul`]'s tensor. Each part is read on the host and the joined
/// rows uploaded once.
pub(crate) fn joined_qtensor(
    gguf: &GgufLoader,
    parts: &[QuantizedPart<'_>],
    device: &Device,
) -> Result<QTensor> {
    let mut joined = Vec::new();
    let mut rows = 0;
    let mut layout: Option<(GgmlDType, usize)> = None;
    for part in parts {
        let tensor = gguf.load_qtensor(part.name, &Device::Cpu)?;
        let (part_rows, columns) = tensor.shape().dims2()?;
        match layout {
            None => layout = Some((tensor.dtype(), columns)),
            Some(expected) if expected == (tensor.dtype(), columns) => {}
            Some((dtype, width)) => {
                return Err(Error::ModelLoadError(format!(
                    "GGUF tensor {} is {:?} x {columns}, unlike the {dtype:?} x {width} it joins",
                    part.name,
                    tensor.dtype()
                )))
            }
        }
        let data = tensor.data()?;
        let order = match part.rope_heads {
            Some(heads) => half_split_row_order(part_rows, heads).ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "GGUF tensor {} has {part_rows} rows, which do not split into {heads} heads of even width",
                    part.name
                ))
            })?,
            None => (0..part_rows).collect(),
        };
        append_rows(&mut joined, &data, part_rows, &order)?;
        rows += part_rows;
    }
    let (dtype, columns) =
        layout.ok_or_else(|| Error::ModelLoadError("GGUF projection has no parts".into()))?;
    let storage = QStorage::from_data(Cow::Owned(joined), device, dtype)?;
    QTensor::new(storage, (rows, columns)).map_err(Error::from)
}

/// Rows `rows` of the GGUF tensor `name` as their own quantized tensor on
/// `device` (a slice of a vocabulary, for a head that scores only part of
/// it).
pub(crate) fn row_slice_qtensor(
    gguf: &GgufLoader,
    name: &str,
    rows: Range<usize>,
    device: &Device,
) -> Result<QTensor> {
    let tensor = gguf.load_qtensor(name, &Device::Cpu)?;
    let (total, columns) = tensor.shape().dims2()?;
    if rows.start >= rows.end || rows.end > total {
        return Err(Error::ModelLoadError(format!(
            "GGUF tensor {name} has {total} rows; rows {rows:?} are out of range"
        )));
    }
    let data = tensor.data()?;
    let order = rows.clone().collect::<Vec<_>>();
    let mut sliced = Vec::new();
    append_rows(&mut sliced, &data, total, &order)?;
    let storage = QStorage::from_data(Cow::Owned(sliced), device, tensor.dtype())?;
    QTensor::new(storage, (rows.len(), columns)).map_err(Error::from)
}

/// The GGUF row that holds each half-split row, head by head. llama.cpp's
/// conversion stores row `i` of a head's first half at `2i` and row `i` of
/// its second half at `2i + 1`.
pub(crate) fn half_split_row_order(rows: usize, heads: usize) -> Option<Vec<usize>> {
    if heads == 0 || !rows.is_multiple_of(heads) {
        return None;
    }
    let head_rows = rows / heads;
    if !head_rows.is_multiple_of(2) {
        return None;
    }
    let half = head_rows / 2;
    Some(
        (0..heads)
            .flat_map(|head| {
                (0..2).flat_map(move |second| {
                    (0..half).map(move |index| head * head_rows + 2 * index + second)
                })
            })
            .collect(),
    )
}

/// Appends `data`'s rows (`rows` of equal byte length) in `order`.
pub(crate) fn append_rows(
    out: &mut Vec<u8>,
    data: &[u8],
    rows: usize,
    order: &[usize],
) -> Result<()> {
    if rows == 0 || !data.len().is_multiple_of(rows) {
        return Err(Error::ModelLoadError(format!(
            "a GGUF tensor of {} bytes does not hold {rows} equal rows",
            data.len()
        )));
    }
    let row_bytes = data.len() / rows;
    out.reserve(order.len() * row_bytes);
    for &row in order {
        out.extend_from_slice(&data[row * row_bytes..(row + 1) * row_bytes]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_split_order_undoes_llamas_interleave() {
        // llama.cpp's `permute` on one head of 6 rows: [a0 a1 a2 | b0 b1 b2]
        // is stored as [a0 b0 a1 b1 a2 b2].
        let interleaved = ["a0", "b0", "a1", "b1", "a2", "b2"];
        let order = half_split_row_order(6, 1).unwrap();
        let restored = order
            .iter()
            .map(|row| interleaved[*row])
            .collect::<Vec<_>>();
        assert_eq!(restored, ["a0", "a1", "a2", "b0", "b1", "b2"]);
        assert_eq!(
            half_split_row_order(8, 2).unwrap(),
            [0, 2, 1, 3, 4, 6, 5, 7]
        );
        assert!(half_split_row_order(6, 4).is_none());
        assert!(half_split_row_order(3, 1).is_none());
    }

    #[test]
    fn joined_rows_follow_the_order() {
        let mut out = Vec::new();
        append_rows(&mut out, &[1, 1, 2, 2, 3, 3], 3, &[2, 0]).unwrap();
        assert_eq!(out, [3, 3, 1, 1]);
        assert!(append_rows(&mut out, &[1, 2, 3], 2, &[0]).is_err());
    }
}
