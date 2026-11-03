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
///
/// Each tile expands into one buffer the calling thread keeps between
/// calls, and its product lands straight in its columns of the one output
/// buffer: no fresh pages per tile and no final concatenation. Accelerate's
/// SGEMM multiplies on macOS builds with the `accelerate` feature (Candle's
/// own F32 matmul there), the `gemm` crate elsewhere.
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
    *dims.last_mut().expect("rank checked above") = n;
    let x = x.to_dtype(candle_core::DType::F32)?.flatten_all()?;
    let x = x.to_vec1::<f32>()?;
    let mut out = vec![0f32; m * n];
    if m > 0 {
        // Taken, not borrowed: a nested call on this thread (a rayon worker
        // stealing another call's task) expands into its own buffer.
        let mut tile = TILE.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
        for start in (0..n).step_by(tile_rows) {
            let rows = tile_rows.min(n - start);
            tile.resize(rows * k, 0.0);
            tile[..rows * k]
                .par_chunks_mut(k)
                .zip(data[start * row_bytes..(start + rows) * row_bytes].par_chunks(row_bytes))
                .for_each(|(out, row)| dequantize_row(row, out));
            tile_product(&x, m, k, &tile[..rows * k], rows, &mut out, n, start);
        }
        TILE.with(|cell| {
            let mut kept = cell.borrow_mut();
            if kept.capacity() < tile.capacity() {
                *kept = tile;
            }
        });
    }
    Tensor::from_vec(out, dims, &Device::Cpu)
}

thread_local! {
    /// The calling thread's F32 tile, kept between calls.
    static TILE: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// `out[i, start + j] = sum_p x[i, p] * tile[j, p]` for `i < m`, `j < rows`,
/// with `out` `[m, n]` row-major.
#[allow(clippy::too_many_arguments)]
fn tile_product(
    x: &[f32],
    m: usize,
    k: usize,
    tile: &[f32],
    rows: usize,
    out: &mut [f32],
    n: usize,
    start: usize,
) {
    debug_assert!(x.len() >= m * k && tile.len() >= rows * k && out.len() >= m * n);
    debug_assert!(start + rows <= n);
    #[cfg(all(target_os = "macos", feature = "accelerate"))]
    {
        const ROW_MAJOR: i32 = 101;
        const NO_TRANS: i32 = 111;
        const TRANS: i32 = 112;
        #[link(name = "Accelerate", kind = "framework")]
        extern "C" {
            fn cblas_sgemm(
                order: i32,
                trans_a: i32,
                trans_b: i32,
                m: i32,
                n: i32,
                k: i32,
                alpha: f32,
                a: *const f32,
                lda: i32,
                b: *const f32,
                ldb: i32,
                beta: f32,
                c: *mut f32,
                ldc: i32,
            );
        }
        // SAFETY: the slices cover every element the call reads and writes
        // (checked above); the dimensions fit i32 for any Q8_0 projection.
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                NO_TRANS,
                TRANS,
                m as i32,
                rows as i32,
                k as i32,
                1.0,
                x.as_ptr(),
                k as i32,
                tile.as_ptr(),
                k as i32,
                0.0,
                out.as_mut_ptr().add(start),
                n as i32,
            );
        }
    }
    #[cfg(not(all(target_os = "macos", feature = "accelerate")))]
    {
        let threads = rayon::current_num_threads();
        let parallelism = if threads > 1 {
            gemm::Parallelism::Rayon(threads)
        } else {
            gemm::Parallelism::None
        };
        // SAFETY: as above; `out`'s block starts at column `start` with row
        // stride `n`, and `tile` is read as its transpose (column stride k).
        unsafe {
            gemm::gemm(
                m,
                rows,
                k,
                out.as_mut_ptr().add(start),
                1,
                n as isize,
                false,
                x.as_ptr(),
                1,
                k as isize,
                tile.as_ptr(),
                k as isize,
                1,
                0.0,
                1.0,
                false,
                false,
                false,
                parallelism,
            );
        }
    }
}

/// One row of Q8_0 blocks into F32: `d * q` per element, as Candle's
/// `BlockQ8_0::to_float`.
fn dequantize_row(row: &[u8], out: &mut [f32]) {
    let (blocks, _) = row.as_chunks::<BLOCK_BYTES>();
    let (outputs, _) = out.as_chunks_mut::<BLOCK_VALUES>();
    for (block, values) in blocks.iter().zip(outputs.iter_mut()) {
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

    /// The kept tile serves calls of other shapes, and an empty input gives
    /// an empty output.
    #[test]
    fn the_kept_tile_serves_calls_of_any_shape() {
        for (n, k, m) in [
            (96usize, 64usize, 7usize),
            (40, 128, 3),
            (200, 32, 11),
            (64, 64, 0),
        ] {
            let weights = matrix(n, k, n as f32);
            let q = QTensor::quantize(&weights, GgmlDType::Q8_0).unwrap();
            let x = matrix(m.max(1), k, 1.0).narrow(0, 0, m).unwrap();
            let tiled = dequant_tile_matmul(&x, &q, 48).unwrap();
            assert_eq!(tiled.dims(), [m, n]);
            let dense = q.dequantize(&Device::Cpu).unwrap();
            let reference = x.matmul(&dense.t().unwrap()).unwrap();
            let max = values(&tiled)
                .iter()
                .zip(&values(&reference))
                .fold(0f32, |acc, (a, b)| acc.max((a - b).abs()));
            assert!(max <= 1e-5 * k as f32, "[{n}, {k}] x {m}: max |diff| {max}");
        }
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
