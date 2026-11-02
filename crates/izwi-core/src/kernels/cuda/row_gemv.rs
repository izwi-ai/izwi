//! Row-invariant multi-row GEMV: `y = x @ W.T` for decode steps whose rows
//! must not depend on how many rows share the call.
//!
//! A batched decode feeds several requests' rows through one projection. A
//! tile GEMM (cuBLAS) picks its kernel, its split of the reduction and its
//! blocking by the row count, so a request's bits change with the batch it
//! happens to share. This kernel (`row_gemv.cu`) gives every row one fixed
//! reduction order, independent of the row count:
//!
//! ```text
//! lane l sums chunks k0 = 8l, 8l + 256, ...:  part = fma chain over the chunk's 8 products
//!                                             acc  = fma(part, block scale or 1, acc)
//! y[r, c] = xor butterfly (16, 8, 4, 2, 1) over the 32 lane sums, lane 0's
//! ```
//!
//! One warp owns one output channel and reads its weight row exactly once per
//! call, folding each loaded chunk into every row; up to [`MAX_ROWS`] rows go
//! in one launch, and longer calls split into launches of that many rows,
//! which changes nothing for a row. Weights are dense (the activations'
//! 16-bit dtype, or F32) or E4M3FN with one F32 scale per 128x128 block; the
//! activations are F16 or BF16 and the output their dtype or F32.
//!
//! [`reference`] is the same arithmetic on the host, operation for operation
//! (`f32::mul_add` is the kernel's `fmaf`), so on IEEE hardware the device
//! and the reference agree bit for bit. [`verify`] runs the device kernel
//! against it, and checks that every row's bits are the same at every row
//! count, before [`verify_and_enable`] lets a caller use it.

use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{DType, Device, Result, Tensor};

/// Most rows one launch takes.
pub const MAX_ROWS: usize = 64;
/// Rows of the small accumulator file (`_r8` kernels).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
const SMALL_ROWS: usize = 8;
/// Weights a lane loads at a time.
const CHUNK: usize = 8;
const LANES: usize = 32;
/// FP8 block edge.
const BLOCK: usize = 128;

/// The weight operand.
#[derive(Debug, Clone, Copy)]
pub enum RowGemvWeight<'a> {
    /// `[n, k]` in the activations' 16-bit dtype, or F32.
    Dense(&'a Tensor),
    /// `[n, k]` E4M3FN bytes (U8) and their `[ceil(n / 128), k / 128]` F32
    /// block scales.
    BlockFp8 {
        codes: &'a Tensor,
        scales: &'a Tensor,
    },
}

impl RowGemvWeight<'_> {
    fn matrix(&self) -> &Tensor {
        match self {
            Self::Dense(weights) => weights,
            Self::BlockFp8 { codes, .. } => codes,
        }
    }
}

/// Whether the kernel covers `x_dtype` activations against `weight` into
/// `out`: F16/BF16 activations; dense weights in their dtype or F32 with K a
/// multiple of 8, E4M3 weights with K a multiple of 128; output in the
/// activations' dtype or F32.
pub fn supported(x_dtype: DType, weight: RowGemvWeight<'_>, out: DType) -> bool {
    let Ok((n, k)) = weight.matrix().dims2() else {
        return false;
    };
    let activations = matches!(x_dtype, DType::F16 | DType::BF16);
    let output = out == x_dtype || out == DType::F32;
    let shape = n > 0 && k > 0 && k.is_multiple_of(CHUNK) && n * k <= i32::MAX as usize;
    let weights = match weight {
        RowGemvWeight::Dense(w) => w.dtype() == x_dtype || w.dtype() == DType::F32,
        RowGemvWeight::BlockFp8 { codes, scales } => {
            codes.dtype() == DType::U8
                && scales.dtype() == DType::F32
                && k.is_multiple_of(BLOCK)
                && scales.dims() == [n.div_ceil(BLOCK), k / BLOCK]
        }
    };
    activations && output && shape && weights
}

/// `(m, n, k)` of a checked call.
fn validate(x: &Tensor, weight: RowGemvWeight<'_>, out: DType) -> Result<(usize, usize, usize)> {
    let (m, width) = x.dims2()?;
    let (n, k) = weight.matrix().dims2()?;
    let same_device = |t: &Tensor| t.device().same_device(x.device());
    let devices = match weight {
        RowGemvWeight::Dense(w) => same_device(w),
        RowGemvWeight::BlockFp8 { codes, scales } => same_device(codes) && same_device(scales),
    };
    if m == 0 || width != k || !devices || !supported(x.dtype(), weight, out) {
        candle_core::bail!(
            "invalid row-invariant GEMV contract: x {:?} {:?}, weights {:?} {:?}, output {out:?}",
            x.dims(),
            x.dtype(),
            weight.matrix().dims(),
            weight.matrix().dtype()
        )
    }
    Ok((m, n, k))
}

/// `x @ W.T`, `[m, n]` in `out`, every row computed in the one order the
/// module documents: the CUDA kernel on CUDA, [`reference`] on the CPU.
pub fn row_gemv(x: &Tensor, weight: RowGemvWeight<'_>, out: DType) -> Result<Tensor> {
    let (m, n, k) = validate(x, weight, out)?;
    #[cfg(feature = "cuda")]
    if x.device().is_cuda() {
        return cuda_impl::launch(x, weight, out, [m, n, k]);
    }
    let _ = (m, n, k);
    if !x.device().is_cpu() {
        candle_core::bail!("the row-invariant GEMV runs on CUDA, and its reference on the CPU")
    }
    reference(x, weight, out)
}

/// One row of the kernel's arithmetic: `x` and `w` exact F32 copies of the
/// row's activations and the channel's weights, `scales` the channel's
/// block-row of FP8 scales.
pub fn reference_dot(x: &[f32], w: &[f32], scales: Option<&[f32]>) -> f32 {
    let k = x.len();
    let mut lanes = [0f32; LANES];
    for (lane, total) in lanes.iter_mut().enumerate() {
        let mut acc = 0f32;
        let mut k0 = lane * CHUNK;
        while k0 < k {
            let scale = scales.map_or(1.0, |s| s[k0 / BLOCK]);
            let mut part = 0f32;
            for (a, b) in x[k0..k0 + CHUNK].iter().zip(&w[k0..k0 + CHUNK]) {
                part = a.mul_add(*b, part);
            }
            acc = part.mul_add(scale, acc);
            k0 += LANES * CHUNK;
        }
        *total = acc;
    }
    // `total += __shfl_xor_sync(total, offset)` for 16, 8, 4, 2, 1.
    let mut offset = LANES / 2;
    while offset > 0 {
        let previous = lanes;
        for (lane, total) in lanes.iter_mut().enumerate() {
            *total = previous[lane] + previous[lane ^ offset];
        }
        offset /= 2;
    }
    lanes[0]
}

fn host_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
}

/// The kernel's arithmetic on the host (see the module docs), for any
/// device's tensors; the result lives on `x`'s device.
pub fn reference(x: &Tensor, weight: RowGemvWeight<'_>, out: DType) -> Result<Tensor> {
    let (m, n, k) = validate(x, weight, out)?;
    let xs = host_f32(x)?;
    let (w, scales) = match weight {
        RowGemvWeight::Dense(w) => (host_f32(w)?, None),
        RowGemvWeight::BlockFp8 { codes, scales } => {
            let bytes = codes
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<u8>()?;
            (
                bytes.into_iter().map(super::fp8::decode_e4m3fn).collect(),
                Some(host_f32(scales)?),
            )
        }
    };
    let blocks = k / BLOCK;
    let mut y = Vec::with_capacity(m * n);
    for row in xs.chunks_exact(k) {
        for c in 0..n {
            let s = scales
                .as_ref()
                .map(|s| &s[(c / BLOCK) * blocks..(c / BLOCK + 1) * blocks]);
            y.push(reference_dot(row, &w[c * k..(c + 1) * k], s));
        }
    }
    // One rounding to the output dtype, as the kernel's store.
    Tensor::from_vec(y, (m, n), &Device::Cpu)?
        .to_dtype(out)?
        .to_device(x.device())
}

static VERIFIED: AtomicBool = AtomicBool::new(false);

/// Whether [`verify_and_enable`] has passed on this process's device.
pub fn enabled() -> bool {
    VERIFIED.load(Ordering::Relaxed)
}

/// [`verify`] on `device`, then the process-wide switch. Idempotent.
pub fn verify_and_enable(device: &Device) -> Result<()> {
    if enabled() {
        return Ok(());
    }
    verify(device)?;
    VERIFIED.store(true, Ordering::Relaxed);
    Ok(())
}

/// The device kernel against [`reference`] at row counts 1, 2, 3, 8, 9 and
/// 64 (both accumulator files), over dense BF16/F16/F32 and E4M3 weights,
/// 16-bit and F32 outputs, a ragged channel count and xorshift inputs: every
/// value within the output's rounding of the reference (F32: 1e-5 of the
/// largest magnitude; 16-bit: 2 units of its last place), and every row's
/// bits the same at every row count. Sets no flag.
pub fn verify(device: &Device) -> Result<()> {
    if !device.is_cuda() {
        candle_core::bail!("the row-invariant GEMV kernel runs on CUDA only")
    }
    use crate::kernels::numeric_check::xorshift_values;
    const COUNTS: [usize; 6] = [1, 2, 3, 8, 9, MAX_ROWS];
    let (n, k) = (200usize, 1024usize);
    let codes = (0..n * k)
        .map(|i| {
            let mixed = (i.wrapping_mul(0x9E37_79B1) >> 17) as u8;
            if mixed & 0x7f == 0x7f {
                mixed ^ 0x41
            } else {
                mixed
            }
        })
        .collect::<Vec<_>>();
    let scales = (0..n.div_ceil(BLOCK) * (k / BLOCK))
        .map(|i| 1e-3 * (1.0 + (i % 17) as f32 * 0.37))
        .collect::<Vec<_>>();
    let codes = Tensor::from_vec(codes, (n, k), device)?;
    let scales = Tensor::from_vec(scales, (n.div_ceil(BLOCK), k / BLOCK), device)?;
    let dense = Tensor::from_vec(xorshift_values(n * k, 0x6E37, 0.05), (n, k), device)?;
    let inputs = Tensor::from_vec(
        xorshift_values(MAX_ROWS * k, 0x6E38, 1.0),
        (MAX_ROWS, k),
        device,
    )?;
    for x_dtype in [DType::BF16, DType::F16] {
        let x = inputs.to_dtype(x_dtype)?;
        let half = dense.to_dtype(x_dtype)?;
        let weights = [
            ("dense", RowGemvWeight::Dense(&half)),
            ("dense F32", RowGemvWeight::Dense(&dense)),
            (
                "E4M3",
                RowGemvWeight::BlockFp8 {
                    codes: &codes,
                    scales: &scales,
                },
            ),
        ];
        for (name, weight) in weights {
            for out in [x_dtype, DType::F32] {
                let label = format!("{x_dtype:?} x {name} -> {out:?}");
                let expected = host_f32(&reference(&x, weight, out)?)?;
                let peak = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
                let mut top = Vec::new();
                for &m in COUNTS.iter().rev() {
                    let y = host_f32(&row_gemv(&x.narrow(0, 0, m)?, weight, out)?)?;
                    for (i, (a, e)) in y.iter().zip(&expected).enumerate() {
                        let limit = if out == DType::F32 {
                            1e-5 * peak
                        } else {
                            // Two units of the 16-bit output's last place.
                            let ulp = if out == DType::BF16 {
                                2f32.powi(-7)
                            } else {
                                2f32.powi(-10)
                            };
                            2.0 * ulp * e.abs().max(f32::MIN_POSITIVE)
                        };
                        if !a.is_finite() || (a - e).abs() > limit {
                            candle_core::bail!(
                                "row-invariant GEMV {label} at m={m}, index {i}: {a} vs reference {e}"
                            )
                        }
                    }
                    if top.is_empty() {
                        top = y;
                    } else if let Some(i) = y
                        .iter()
                        .zip(&top)
                        .position(|(a, b)| a.to_bits() != b.to_bits())
                    {
                        candle_core::bail!(
                            "row-invariant GEMV {label}: row {} changes bits between m={m} and m={MAX_ROWS}",
                            i / n
                        )
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core::cuda_backend::cudarc::driver::{CudaView, LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaDType, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{CudaStorage, DType, Result, Storage, Tensor};

    use super::{RowGemvWeight, MAX_ROWS, SMALL_ROWS};

    const MODULE: &str = "izwi_row_gemv";
    const WARPS: usize = 8;

    /// `tensor` contiguous with its start aligned to `align` bytes (a copy
    /// when it is not): the kernel loads rows in 8- and 16-byte vectors.
    fn aligned(tensor: &Tensor, align: usize) -> Result<Tensor> {
        let tensor = tensor.contiguous()?;
        let start = tensor
            .layout()
            .contiguous_offsets()
            .map(|(start, _)| start * tensor.dtype().size_in_bytes());
        if start.is_some_and(|start| start.is_multiple_of(align)) {
            Ok(tensor)
        } else {
            tensor.force_contiguous()
        }
    }

    fn view<'a, T: CudaDType>(storage: &'a Storage, tensor: &Tensor) -> Result<CudaView<'a, T>> {
        let Storage::Cuda(cuda) = storage else {
            candle_core::bail!("row-invariant GEMV operands must be CUDA tensors")
        };
        let Some((start, end)) = tensor.layout().contiguous_offsets() else {
            candle_core::bail!("row-invariant GEMV operands must be contiguous")
        };
        Ok(cuda.as_cuda_slice::<T>()?.slice(start..end))
    }

    fn int(value: usize) -> Result<i32> {
        i32::try_from(value)
            .map_err(|_| candle_core::Error::Msg(format!("row-invariant GEMV argument {value}")))
    }

    fn suffix(dtype: DType) -> &'static str {
        match dtype {
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            _ => "f32",
        }
    }

    pub(super) fn launch(
        x: &Tensor,
        weight: RowGemvWeight<'_>,
        out: DType,
        [m, n, k]: [usize; 3],
    ) -> Result<Tensor> {
        let device = x.device().as_cuda_device()?;
        let x = aligned(x, 16)?;
        let (weights, scales, weight_name) = match weight {
            RowGemvWeight::Dense(w) => (aligned(w, 16)?, None, suffix(w.dtype())),
            RowGemvWeight::BlockFp8 { codes, scales } => {
                (aligned(codes, 8)?, Some(scales.contiguous()?), "e4m3")
            }
        };
        let config = LaunchConfig {
            grid_dim: (
                u32::try_from(n.div_ceil(WARPS))
                    .map_err(|_| candle_core::Error::Msg("row-invariant GEMV grid".into()))?,
                1,
                1,
            ),
            block_dim: ((WARPS * 32) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (w_storage, _) = weights.storage_and_layout();
        let scale_storage = scales.as_ref().map(|s| (s.storage_and_layout().0, s));
        let mut outputs = Vec::with_capacity(m.div_ceil(MAX_ROWS));
        for start in (0..m).step_by(MAX_ROWS) {
            let rows = (m - start).min(MAX_ROWS);
            // K is a multiple of 8, so every chunk of rows starts as aligned
            // as `x` does.
            let chunk = x.narrow(0, start, rows)?;
            let (x_storage, _) = chunk.storage_and_layout();
            let name = format!(
                "izwi_row_gemv_{}_{weight_name}_{}_r{}",
                suffix(x.dtype()),
                suffix(out),
                if rows <= SMALL_ROWS {
                    SMALL_ROWS
                } else {
                    MAX_ROWS
                }
            );
            let function =
                device.get_or_load_custom_func(&name, MODULE, super::super::cuda_ptx::ROW_GEMV)?;
            // A null scale pointer selects dense weights.
            let null = 0u64;
            macro_rules! run {
                ($tx:ty, $ty:ty, $tw:ty) => {{
                    let xv = view::<$tx>(&x_storage, &chunk)?;
                    let wv = view::<$tw>(&w_storage, &weights)?;
                    let sv = match &scale_storage {
                        Some((storage, tensor)) => Some(view::<f32>(storage, tensor)?),
                        None => None,
                    };
                    // SAFETY: the kernel writes every element of `y`.
                    let y = unsafe { device.alloc::<$ty>(rows * n)? };
                    let mut builder = function.builder();
                    builder.arg(&xv);
                    builder.arg(&wv);
                    match &sv {
                        Some(sv) => builder.arg(sv),
                        None => builder.arg(&null),
                    };
                    builder.arg(&y);
                    candle_core::builder_arg!(builder, int(rows)?, int(n)?, int(k)?);
                    // SAFETY: argument order and types match the kernel, and
                    // the host checked every shape and alignment.
                    unsafe { builder.launch(config) }.w()?;
                    CudaStorage::wrap_cuda_slice(y, device.clone())
                }};
            }
            macro_rules! by_weight {
                ($tx:ty, $ty:ty) => {
                    match weights.dtype() {
                        DType::U8 => run!($tx, $ty, u8),
                        DType::F32 => run!($tx, $ty, f32),
                        _ => run!($tx, $ty, $tx),
                    }
                };
            }
            let storage = match (x.dtype(), out) {
                (DType::BF16, DType::BF16) => by_weight!(half::bf16, half::bf16),
                (DType::BF16, DType::F32) => by_weight!(half::bf16, f32),
                (DType::F16, DType::F16) => by_weight!(half::f16, half::f16),
                (DType::F16, DType::F32) => by_weight!(half::f16, f32),
                (x_dtype, out) => {
                    candle_core::bail!("row-invariant GEMV does not take {x_dtype:?} -> {out:?}")
                }
            };
            outputs.push(Tensor::from_storage(
                Storage::Cuda(storage),
                (rows, n),
                BackpropOp::none(),
                false,
            ));
        }
        if outputs.len() == 1 {
            return Ok(outputs.remove(0));
        }
        Tensor::cat(&outputs, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::numeric_check::xorshift_values;

    fn f64_reference(x: &[f32], w: &[f32], k: usize) -> Vec<f64> {
        x.chunks_exact(k)
            .flat_map(|row| {
                w.chunks_exact(k).map(move |channel| {
                    row.iter()
                        .zip(channel)
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum::<f64>()
                })
            })
            .collect()
    }

    fn host(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// The reference against F64 at the published Drama 3 shapes' K, for
    /// dense BF16 and F32 weights and E4M3 with block scales.
    #[test]
    fn the_reference_computes_the_projection() {
        let (n, k, m) = (130usize, 2048usize, 3usize);
        let x = Tensor::from_vec(xorshift_values(m * k, 1, 1.0), (m, k), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w = Tensor::from_vec(xorshift_values(n * k, 2, 0.05), (n, k), &Device::Cpu).unwrap();
        let x32 = host(&x);
        for weights in [w.to_dtype(DType::BF16).unwrap(), w.clone()] {
            let y = host(&reference(&x, RowGemvWeight::Dense(&weights), DType::F32).unwrap());
            let expected = f64_reference(&x32, &host(&weights), k);
            for (a, e) in y.iter().zip(&expected) {
                assert!((f64::from(*a) - e).abs() < 1e-5, "{a} vs {e}");
            }
        }
        // E4M3: every finite byte, a scale per 128x128 block.
        let codes = (0..n * k)
            .map(|i| {
                let b = (i.wrapping_mul(0x9E37_79B1) >> 17) as u8;
                if b & 0x7f == 0x7f {
                    b ^ 0x41
                } else {
                    b
                }
            })
            .collect::<Vec<_>>();
        let scales = (0..n.div_ceil(128) * (k / 128))
            .map(|i| 1e-3 * (1.0 + (i % 17) as f32 * 0.37))
            .collect::<Vec<_>>();
        let dequant = codes
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                let (c, j) = (i / k, i % k);
                super::super::fp8::decode_e4m3fn(b) * scales[(c / 128) * (k / 128) + j / 128]
            })
            .collect::<Vec<_>>();
        let codes = Tensor::from_vec(codes, (n, k), &Device::Cpu).unwrap();
        let scales = Tensor::from_vec(scales, (n.div_ceil(128), k / 128), &Device::Cpu).unwrap();
        let weight = RowGemvWeight::BlockFp8 {
            codes: &codes,
            scales: &scales,
        };
        let y = host(&row_gemv(&x, weight, DType::F32).unwrap());
        let expected = f64_reference(&x32, &dequant, k);
        let peak = expected.iter().fold(0f64, |m, v| m.max(v.abs()));
        for (a, e) in y.iter().zip(&expected) {
            assert!((f64::from(*a) - e).abs() < 1e-5 * peak, "{a} vs {e}");
        }
        // A 16-bit output is the F32 result rounded once.
        let rounded = host(&reference(&x, weight, DType::BF16).unwrap());
        let once = host(
            &Tensor::new(y.as_slice(), &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap(),
        );
        assert_eq!(rounded, once);
    }

    /// Row r's bits are the same at every row count; RED: a reduction split
    /// by the row count (a batch-sized split-K) changes them.
    #[test]
    fn a_rows_bits_do_not_depend_on_the_row_count() {
        let (n, k) = (24usize, 512usize);
        let x = xorshift_values(MAX_ROWS * k, 3, 1.0);
        let w = xorshift_values(n * k, 4, 0.05);
        let run = |m: usize, split: bool| -> Vec<u32> {
            x[..m * k]
                .chunks_exact(k)
                .flat_map(|row| {
                    w.chunks_exact(k).map(move |channel| {
                        if split && m > 4 {
                            let half = k / 2;
                            (reference_dot(&row[..half], &channel[..half], None)
                                + reference_dot(&row[half..], &channel[half..], None))
                            .to_bits()
                        } else {
                            reference_dot(row, channel, None).to_bits()
                        }
                    })
                })
                .collect()
        };
        let all = run(MAX_ROWS, false);
        for m in [1, 2, 4, 8, 32] {
            assert_eq!(run(m, false), all[..m * n], "m={m}");
        }
        let split = run(MAX_ROWS, true);
        assert_ne!(
            split[..8 * n],
            run(8, false)[..],
            "the RED variant must change bits"
        );
        // The facade is the reference on the CPU, row by row.
        let xt = Tensor::from_vec(x.clone(), (MAX_ROWS, k), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let wt = Tensor::from_vec(w, (n, k), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let full = host(&row_gemv(&xt, RowGemvWeight::Dense(&wt), DType::F32).unwrap());
        for m in [1, 5, 9] {
            let part = host(
                &row_gemv(
                    &xt.narrow(0, 0, m).unwrap(),
                    RowGemvWeight::Dense(&wt),
                    DType::F32,
                )
                .unwrap(),
            );
            assert_eq!(part[..], full[..m * n]);
        }
    }

    #[test]
    fn the_contract_refuses_what_the_kernel_cannot_take() {
        let w = Tensor::zeros((8, 64), DType::BF16, &Device::Cpu).unwrap();
        assert!(supported(DType::BF16, RowGemvWeight::Dense(&w), DType::F32));
        assert!(supported(
            DType::BF16,
            RowGemvWeight::Dense(&w),
            DType::BF16
        ));
        assert!(!supported(DType::F16, RowGemvWeight::Dense(&w), DType::F16));
        assert!(!supported(DType::F32, RowGemvWeight::Dense(&w), DType::F32));
        assert!(!supported(
            DType::BF16,
            RowGemvWeight::Dense(&w),
            DType::F16
        ));
        let ragged = Tensor::zeros((8, 60), DType::BF16, &Device::Cpu).unwrap();
        assert!(!supported(
            DType::BF16,
            RowGemvWeight::Dense(&ragged),
            DType::F32
        ));
        // E4M3 needs whole 128-column blocks and matching scales.
        let codes = Tensor::zeros((8, 64), DType::U8, &Device::Cpu).unwrap();
        let scales = Tensor::zeros((1, 0), DType::F32, &Device::Cpu).unwrap();
        assert!(!supported(
            DType::BF16,
            RowGemvWeight::BlockFp8 {
                codes: &codes,
                scales: &scales
            },
            DType::F32
        ));
        let x = Tensor::zeros((2, 32), DType::BF16, &Device::Cpu).unwrap();
        assert!(row_gemv(&x, RowGemvWeight::Dense(&w), DType::F32).is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_row_gemv_matches_its_reference_and_keeps_row_bits() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        verify(&device).unwrap();
        // The device and the reference agree bit for bit at a published
        // shape (IEEE fma and adds in the same order).
        let (n, k, m) = (520usize, 2048usize, 8usize);
        let x = Tensor::from_vec(xorshift_values(m * k, 5, 1.0), (m, k), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w = Tensor::from_vec(xorshift_values(n * k, 6, 0.05), (n, k), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let got = host(
            &row_gemv(&x, RowGemvWeight::Dense(&w), DType::F32)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap(),
        );
        let expected = host(
            &reference(&x, RowGemvWeight::Dense(&w), DType::F32)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap(),
        );
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}
