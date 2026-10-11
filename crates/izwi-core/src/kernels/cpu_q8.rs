//! Q8_0 prefill on the CPU.
//!
//! Candle's CPU `QMatMul` for Q8_0 loops over activation rows and streams the
//! complete weight matrix once per row, so a prompt of `m` tokens reads every
//! weight `m` times. Above a few rows it is faster to expand a tile of weight
//! rows to F32 once (in parallel) and hand the multiplication to Candle's F32
//! matmul, which reads that tile once for every row. Generic: any `[n, k]`
//! Q8_0 tensor.
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Result, Tensor, D};

/// Q8_0 block layout: an F16 scale followed by 32 signed bytes.
const BLOCK_BYTES: usize = 34;
const BLOCK_VALUES: usize = 32;

/// `x @ dequantize(q).T` for a CPU Q8_0 matrix `q` (`[n, k]`), expanding
/// `tile_rows` weight rows to F32 at a time. `x` is `[.., k]` F32; the
/// result is `[.., n]` F32.
pub fn dequant_tile_matmul(x: &Tensor, q: &QTensor, tile_rows: usize) -> Result<Tensor> {
    use rayon::prelude::*;
    let (n, k) = q.shape().dims2()?;
    if q.dtype() != GgmlDType::Q8_0
        || !matches!(q.device(), Device::Cpu)
        || !x.device().is_cpu()
        || x.dim(D::Minus1)? != k
        || !k.is_multiple_of(BLOCK_VALUES)
        || tile_rows == 0
    {
        candle_core::bail!("Q8_0 tile matmul needs CPU tensors and a [n, k] Q8_0 matrix")
    }
    let data = q.data()?;
    let row_bytes = k / BLOCK_VALUES * BLOCK_BYTES;
    if data.len() != n * row_bytes {
        candle_core::bail!("Q8_0 payload of {} bytes is not [{n}, {k}]", data.len())
    }
    let mut dims = x.dims().to_vec();
    let m = x.elem_count() / k;
    let x2 = x
        .to_dtype(candle_core::DType::F32)?
        .reshape((m, k))?
        .contiguous()?;
    let mut tiles = Vec::with_capacity(n.div_ceil(tile_rows));
    for start in (0..n).step_by(tile_rows) {
        let rows = tile_rows.min(n - start);
        let mut values = vec![0f32; rows * k];
        values
            .par_chunks_mut(k)
            .zip(data[start * row_bytes..(start + rows) * row_bytes].par_chunks(row_bytes))
            .for_each(|(out, row)| dequantize_row(row, out));
        let weights = Tensor::from_vec(values, (rows, k), &Device::Cpu)?;
        tiles.push(x2.matmul(&weights.t()?)?);
    }
    let y = Tensor::cat(&tiles, 1)?;
    *dims.last_mut().expect("rank checked above") = n;
    y.reshape(dims)
}

/// One row of Q8_0 blocks into F32: `d * q` per element, as Candle's
/// `BlockQ8_0::to_float`.
fn dequantize_row(row: &[u8], out: &mut [f32]) {
    for (block, values) in row
        .chunks_exact(BLOCK_BYTES)
        .zip(out.chunks_exact_mut(BLOCK_VALUES))
    {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        for (value, &q) in values.iter_mut().zip(&block[2..]) {
            *value = f32::from(q as i8) * d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::QMatMul;
    use candle_core::Module;

    fn matrix(rows: usize, cols: usize, seed: f32) -> Tensor {
        Tensor::from_vec(
            (0..rows * cols)
                .map(|i| ((i as f32 + seed) * 0.618_034).sin() * (1.0 + (i % 7) as f32 * 0.1))
                .collect(),
            (rows, cols),
            &Device::Cpu,
        )
        .unwrap()
    }

    fn values(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn relative(a: &[f32], b: &[f32]) -> f32 {
        let e: f32 = a.iter().zip(b).map(|(a, b)| (a - b) * (a - b)).sum();
        let n: f32 = b.iter().map(|b| b * b).sum();
        let r = (e / n).sqrt();
        if r.is_finite() {
            r
        } else {
            f32::INFINITY
        }
    }

    #[test]
    fn tiles_match_dequantize_then_matmul_and_track_qmatmul() {
        // 300 rows: tiles of 128 leave a partial tile.
        let weights = matrix(300, 256, 1.0);
        let q = QTensor::quantize(&weights, GgmlDType::Q8_0).unwrap();
        let x = matrix(2 * 9, 256, 2.0).reshape((2, 9, 256)).unwrap();
        let tiled = dequant_tile_matmul(&x, &q, 128).unwrap();
        assert_eq!(tiled.dims(), [2, 9, 300]);
        let dense = q.dequantize(&Device::Cpu).unwrap();
        let reference = x.broadcast_matmul(&dense.t().unwrap()).unwrap();
        let (tiled, reference) = (values(&tiled), values(&reference));
        let max = tiled
            .iter()
            .zip(&reference)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(max <= 1e-5 * 256.0, "max |diff| {max}");
        // Candle's QMatMul also quantizes the activations: close, not equal.
        let qmatmul = QMatMul::QTensor(std::sync::Arc::new(q))
            .forward(&x)
            .unwrap();
        assert!(relative(&tiled, &values(&qmatmul)) <= 1e-2);
        assert!(dequant_tile_matmul(
            &x.narrow(2, 0, 128).unwrap(),
            &QTensor::quantize(&weights, GgmlDType::Q8_0).unwrap(),
            128
        )
        .is_err());
    }

    #[test]
    fn row_decode_matches_candle_dequantization() {
        let weights = matrix(4, 64, 3.0);
        let q = QTensor::quantize(&weights, GgmlDType::Q8_0).unwrap();
        let data = q.data().unwrap();
        let mut out = vec![0f32; 4 * 64];
        for (row, chunk) in data.chunks(2 * BLOCK_BYTES).enumerate() {
            dequantize_row(chunk, &mut out[row * 64..(row + 1) * 64]);
        }
        assert_eq!(out, values(&q.dequantize(&Device::Cpu).unwrap()));
    }
}
