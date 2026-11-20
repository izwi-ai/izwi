//! Dense multi-row mat-vec for Metal ("skinny GEMM"): `y = x Wᵀ + b` for a
//! few activation rows, `x` `[M, K]` and `W` `[N, K]` in F16 or F32.
//!
//! Candle sends a matmul with two or more rows to a tiled GEMM, which at a
//! handful of rows reads the weight far below the memory bandwidth (M1 Pro,
//! F16 4096 -> 1024: ~180 µs at 6 rows, 1.5x Candle's one-row GEMV), and one
//! row to a GEMV that rounds differently. This kernel follows the layout of
//! llama.cpp's `mul_mv_ext` (as `metal_kquant_mv` does): eight lanes share a
//! weight row (four rows per simdgroup, two simdgroups per threadgroup),
//! each lane dots one 16-weight chunk per step with every activation row of
//! its group, read straight from device memory, and a row's lanes reduce
//! with `simd_shuffle_down`. A threadgroup takes up to [`GROUP_ROWS`]
//! activation rows, so the weight is read once per that many rows.
//! Accumulation is F32; the bias is added in F32 before the one rounding to
//! the activations' dtype.
//!
//! Measured on an M1 Pro, F16, µs per call (Candle matmul and bias add ->
//! this; `tests::bench_flow_shapes`):
//!
//! | `K x N` | 6 rows | 12 rows | 24 rows | 48 rows |
//! |---|---|---|---|---|
//! | 1024 x 1024 | 123 -> 18 | 75 -> 24 | 79 -> 50 | 83 -> 70 |
//! | 1024 x 4096 | 108 -> 37 | 110 -> 66 | 189 -> 161 | 215 -> 244 |
//! | 4096 x 1024 | 193 -> 41 | 198 -> 78 | 224 -> 169 | 233 -> 249 |
//!
//! A lane's time grows with the rows it dots, so from about 48 rows a
//! wide weight takes longer than Candle's tiled GEMM. Larger row groups
//! and four weight rows a lane were both slower.
//!
//! An output's sum runs over its 16-weight chunks in an order fixed by the
//! lane layout and `K`, never by `M` or by how many rows share a
//! threadgroup, so a row gets the same bits alone as beside any other rows:
//! batched decode stays batch-invariant with no padding.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::{bail, CpuStorage, CustomOp3, DType, Layout, MetalStorage, Shape, Tensor};
use candle_metal_kernels::metal::{ComputePipeline, Device as MetalDevice};

use super::metal_encoder::IzwiMetalCommandEncoderExt;

/// Most activation rows one threadgroup takes.
pub(crate) const GROUP_ROWS: usize = 16;
/// Weights one lane dots per step.
const CHUNK: usize = 16;
/// Lanes sharing one weight row.
const LANES_PER_ROW: usize = 8;
/// Simdgroups per threadgroup.
const SIMDGROUPS: usize = 2;
/// Weight rows one lane serves (each activation chunk it loads is dotted
/// with all of them).
const WEIGHT_ROWS_PER_LANE: usize = 1;
/// Weight rows one threadgroup produces.
const WEIGHT_ROWS_PER_GROUP: usize = (32 / LANES_PER_ROW) * WEIGHT_ROWS_PER_LANE * SIMDGROUPS;
/// Row counts a threadgroup is compiled for; a call takes the smallest that
/// holds its rows, up to [`GROUP_ROWS`].
const ROW_VARIANTS: [usize; 8] = [1, 2, 4, 6, 8, 12, 16, 24];

fn dtype_tag(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::F16 => Some("f16"),
        DType::F32 => Some("f32"),
        _ => None,
    }
}

/// Whether the kernel serves `x` `[..., K]` through `weight` `[N, K]` and
/// `bias` `[N]` (all on Metal, F16 or F32, contiguous and 16-byte aligned,
/// weight and bias in one dtype, `K` a multiple of 16).
pub(crate) fn serves(x: &Tensor, weight: &Tensor, bias: &Tensor) -> bool {
    let Ok((n, k)) = weight.dims2() else {
        return false;
    };
    x.device().is_metal()
        && weight.device().is_metal()
        && dtype_tag(x.dtype()).is_some()
        && dtype_tag(weight.dtype()).is_some()
        && bias.dtype() == weight.dtype()
        && bias.dims() == [n]
        && k > 0
        && k.is_multiple_of(CHUNK)
        && x.dims().last() == Some(&k)
        && x.elem_count() >= k
        && aligned(weight)
        && aligned(bias)
        && k <= u32::MAX as usize
        && n <= u32::MAX as usize
        && x.elem_count() / k <= u32::MAX as usize
}

/// Whether `tensor` is contiguous and starts on a 16-byte boundary of its
/// buffer (the kernel reads it four values at a time).
fn aligned(tensor: &Tensor) -> bool {
    let (_, layout) = tensor.storage_and_layout();
    layout.is_contiguous()
        && (layout.start_offset() * tensor.dtype().size_in_bytes()).is_multiple_of(16)
}

/// `x Wᵀ + b` for `x` `[..., K]`; the result is `[..., N]` in `x`'s dtype.
/// The caller checks [`serves`].
pub(crate) fn linear(x: &Tensor, weight: &Tensor, bias: &Tensor) -> candle_core::Result<Tensor> {
    let (n, k) = weight.dims2()?;
    let rows = x.elem_count() / k;
    let mut dims = x.dims().to_vec();
    *dims.last_mut().expect("a linear input has a last dim") = n;
    x.contiguous()?
        .reshape((rows, k))?
        .apply_op3_no_bwd(
            weight,
            bias,
            &DenseMatVec {
                max_group: GROUP_ROWS,
            },
        )?
        .reshape(dims)
}

/// The op, taking at most `max_group` rows a threadgroup (a row's bits do
/// not depend on it; the benchmark compares caps).
struct DenseMatVec {
    max_group: usize,
}

impl CustomOp3 for DenseMatVec {
    fn name(&self) -> &'static str {
        "izwi-dense-mat-vec"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        bail!("izwi-dense-mat-vec runs on Metal only")
    }

    fn metal_fwd(
        &self,
        x: &MetalStorage,
        x_layout: &Layout,
        weight: &MetalStorage,
        weight_layout: &Layout,
        bias: &MetalStorage,
        bias_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let (rows, k) = x_layout.shape().dims2()?;
        let (n, weight_k) = weight_layout.shape().dims2()?;
        let (Some(x_tag), Some(w_tag)) = (dtype_tag(x.dtype()), dtype_tag(weight.dtype())) else {
            bail!("izwi-dense-mat-vec takes F16 or F32 operands")
        };
        if weight_k != k
            || bias_layout.shape().elem_count() != n
            || bias.dtype() != weight.dtype()
            || !k.is_multiple_of(CHUNK)
            || rows == 0
            || !x_layout.is_contiguous()
            || !weight_layout.is_contiguous()
            || !bias_layout.is_contiguous()
        {
            bail!(
                "izwi-dense-mat-vec needs contiguous x [M, K], weight [N, K] and bias [N] with K % {CHUNK} == 0, got {:?}, {:?}, {:?}",
                x_layout.shape(),
                weight_layout.shape(),
                bias_layout.shape()
            )
        }
        let offset = |storage: &MetalStorage, layout: &Layout| {
            layout.start_offset() * storage.dtype().size_in_bytes()
        };
        if [
            offset(x, x_layout),
            offset(weight, weight_layout),
            offset(bias, bias_layout),
        ]
        .iter()
        .any(|bytes| !bytes.is_multiple_of(16))
        {
            bail!("izwi-dense-mat-vec needs 16-byte aligned operands")
        }
        let group = ROW_VARIANTS
            .into_iter()
            .filter(|&variant| variant <= self.max_group)
            .find(|&variant| variant >= rows)
            .unwrap_or(self.max_group);
        let function = format!("izwi_dense_mv_x{x_tag}_w{w_tag}_r{group}");
        let device = x.device().clone();
        let elements = rows * n;
        let output = device.new_buffer(elements, x.dtype(), "izwi-dense-mat-vec")?;
        let pipeline = pipeline(device.metal_device(), &function)?;
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-dense-mat-vec");
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_input_buffer(0, Some(weight.buffer()), offset(weight, weight_layout));
        encoder.set_input_buffer(1, Some(x.buffer()), offset(x, x_layout));
        encoder.set_input_buffer(2, Some(bias.buffer()), offset(bias, bias_layout));
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &(k as u32));
        encoder.set_bytes(5, &(n as u32));
        encoder.set_bytes(6, &(rows as u32));
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: n.div_ceil(WEIGHT_ROWS_PER_GROUP),
                height: rows.div_ceil(group),
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 32 * SIMDGROUPS,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        Ok((
            MetalStorage::new(output, device, elements, x.dtype()),
            Shape::from((rows, n)),
        ))
    }
}

fn pipeline(device: &MetalDevice, function: &str) -> candle_core::Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<(u64, String), ComputePipeline>>> = OnceLock::new();
    let key = (device.registry_id(), function.to_string());
    let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(pipeline) = pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .get(&key)
        .cloned()
    {
        return Ok(pipeline);
    }
    let library = device
        .new_library_with_source(SOURCE, None)
        .map_err(candle_core::Error::wrap)?;
    let function = library
        .get_function(function, None)
        .map_err(candle_core::Error::wrap)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(candle_core::Error::wrap)?;
    pipelines
        .lock()
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?
        .insert(key, pipeline.clone());
    Ok(pipeline)
}

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant constexpr short LANES = 8;   // lanes sharing a weight row
constant constexpr short NSG = 2;     // simdgroups per threadgroup
constant constexpr short ROWS_PER_SG = 32 / LANES;
constant constexpr short CHUNK = 16;  // weights a lane dots per step
constant constexpr short NW = 1;      // weight rows a lane serves

// y[m, n] = sum_k x[m, k] * w[n, k] + b[n]. A lane serves NW weight rows:
// each activation chunk it loads is dotted with all of them, so activations
// are read once per NW weight rows. A threadgroup makes ROWS_PER_SG * NW * NSG
// weight rows for R activation rows. Lane l of a weight row takes chunks
// l, l + 8, ..., and the row's lanes reduce in a fixed tree, so an output's
// bits depend on K and its own operands only.
template <typename TX, typename TW, short R>
kernel void izwi_dense_mv(
        device const TW * w   [[buffer(0)]],
        device const TX * x   [[buffer(1)]],
        device const TW * b   [[buffer(2)]],
        device       TX * y   [[buffer(3)]],
        constant uint & K     [[buffer(4)]],
        constant uint & N     [[buffer(5)]],
        constant uint & M     [[buffer(6)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    typedef vec<TW, 4> tw4;
    typedef vec<TX, 4> tx4;
    const short lane = tiisg % LANES;
    const short slot = tiisg / LANES;
    // This lane's first weight row; its NW rows are ROWS_PER_SG apart.
    const uint n0 = tgpig.x * (ROWS_PER_SG * NW * NSG) + ROWS_PER_SG * NW * sgitg + slot;
    const uint m0 = tgpig.y * R;
    const uint stride = K / 4;
    device const tw4 * wr[NW];
    for (short j = 0; j < NW; ++j) {
        const uint n = n0 + ROWS_PER_SG * j;
        wr[j] = (device const tw4 *)(w + (ulong)(n < N ? n : 0) * K);
    }
    // Rows past M read row 0 and are never written.
    const uint rows = M - m0 < uint(R) ? M - m0 : uint(R);
    device const tx4 * x0 = (device const tx4 *)(x + (ulong)m0 * K);
    float sum[R][NW];
    for (short r = 0; r < R; ++r) {
        for (short j = 0; j < NW; ++j) {
            sum[r][j] = 0.0f;
        }
    }
    for (uint c = lane; c * CHUNK < K; c += LANES) {
        const uint q = c * (CHUNK / 4);
        float4 wv[NW][4];
        for (short j = 0; j < NW; ++j) {
            wv[j][0] = float4(wr[j][q]);
            wv[j][1] = float4(wr[j][q + 1]);
            wv[j][2] = float4(wr[j][q + 2]);
            wv[j][3] = float4(wr[j][q + 3]);
        }
        for (short r = 0; r < R; ++r) {
            device const tx4 * xr = x0 + (uint(r) < rows ? uint(r) * stride : 0) + q;
            const float4 a0 = float4(xr[0]);
            const float4 a1 = float4(xr[1]);
            const float4 a2 = float4(xr[2]);
            const float4 a3 = float4(xr[3]);
            for (short j = 0; j < NW; ++j) {
                sum[r][j] += dot(wv[j][0], a0) + dot(wv[j][1], a1)
                           + dot(wv[j][2], a2) + dot(wv[j][3], a3);
            }
        }
    }
    for (short r = 0; r < R; ++r) {
        for (short j = 0; j < NW; ++j) {
            float s = sum[r][j];
            s += simd_shuffle_down(s, 4);
            s += simd_shuffle_down(s, 2);
            s += simd_shuffle_down(s, 1);
            sum[r][j] = s;
        }
    }
    if (lane == 0) {
        for (short j = 0; j < NW; ++j) {
            const uint n = n0 + ROWS_PER_SG * j;
            if (n < N) {
                const float bias = float(b[n]);
                for (short r = 0; r < R; ++r) {
                    if (uint(r) < rows) {
                        y[(ulong)(m0 + r) * N + n] = TX(sum[r][j] + bias);
                    }
                }
            }
        }
    }
}

#define IZWI_DENSE_MV(XT, XN, WT, WN, R) \
template [[host_name("izwi_dense_mv_x" #XN "_w" #WN "_r" #R)]] kernel void izwi_dense_mv<XT, WT, R>( \
    device const WT *, device const XT *, device const WT *, device XT *, \
    constant uint &, constant uint &, constant uint &, uint3, ushort, ushort);

#define IZWI_DENSE_MV_ROWS(XT, XN, WT, WN) \
IZWI_DENSE_MV(XT, XN, WT, WN, 1) \
IZWI_DENSE_MV(XT, XN, WT, WN, 2) \
IZWI_DENSE_MV(XT, XN, WT, WN, 4) \
IZWI_DENSE_MV(XT, XN, WT, WN, 6) \
IZWI_DENSE_MV(XT, XN, WT, WN, 8) \
IZWI_DENSE_MV(XT, XN, WT, WN, 12) \
IZWI_DENSE_MV(XT, XN, WT, WN, 16) \
IZWI_DENSE_MV(XT, XN, WT, WN, 24)

IZWI_DENSE_MV_ROWS(half, f16, half, f16)
IZWI_DENSE_MV_ROWS(float, f32, half, f16)
IZWI_DENSE_MV_ROWS(half, f16, float, f32)
IZWI_DENSE_MV_ROWS(float, f32, float, f32)
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::numeric_check::xorshift_tensor;

    fn reference(x: &Tensor, w: &Tensor, b: &Tensor) -> Vec<Vec<f32>> {
        let f32 = |t: &Tensor| t.to_dtype(DType::F32).unwrap();
        f32(x)
            .matmul(&f32(w).t().unwrap())
            .unwrap()
            .broadcast_add(&f32(b))
            .unwrap()
            .to_device(&candle_core::Device::Cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()
    }

    fn host(t: &Tensor) -> Vec<Vec<f32>> {
        t.to_dtype(DType::F32)
            .unwrap()
            .to_device(&candle_core::Device::Cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()
    }

    #[test]
    fn matches_an_f32_matmul_and_gives_a_row_its_bits_alone() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "matches_an_f32_matmul_and_gives_a_row_its_bits_alone",
        ) else {
            return;
        };
        for (x_dtype, w_dtype, tolerance) in [
            (DType::F32, DType::F32, 2e-4f32),
            (DType::F16, DType::F16, 2e-2),
            (DType::F32, DType::F16, 2e-4),
            (DType::F16, DType::F32, 2e-2),
        ] {
            for (k, n) in [
                (64usize, 1024usize),
                (1024, 1024),
                (896, 1024),
                (4096, 1024),
                (1024, 64),
            ] {
                let w = xorshift_tensor(&[n, k], 11, 0.05, &device)
                    .unwrap()
                    .to_dtype(w_dtype)
                    .unwrap();
                let b = xorshift_tensor(&[n], 12, 0.5, &device)
                    .unwrap()
                    .to_dtype(w_dtype)
                    .unwrap();
                let x = xorshift_tensor(&[100, k], 13, 1.0, &device)
                    .unwrap()
                    .to_dtype(x_dtype)
                    .unwrap();
                assert!(serves(&x, &w, &b));
                let all = host(&linear(&x, &w, &b).unwrap());
                let want = reference(&x, &w, &b);
                for (row, (got, want)) in all.iter().zip(&want).enumerate() {
                    for (g, e) in got.iter().zip(want) {
                        let scale = e.abs().max(1.0);
                        assert!(
                            (g - e).abs() <= tolerance * scale,
                            "{x_dtype:?}/{w_dtype:?} K {k} N {n} row {row}: {g} vs {e}"
                        );
                    }
                }
                // Every count of rows, one included, gives each row the
                // bits it gets in the 100-row call (three groups).
                for count in [1usize, 2, 3, 5, 6, 7, 9, 13, 16, 17, 25, 33] {
                    let part = host(&linear(&x.narrow(0, 0, count).unwrap(), &w, &b).unwrap());
                    assert_eq!(
                        part[..],
                        all[..count],
                        "{x_dtype:?}/{w_dtype:?} K {k} N {n}, {count} rows"
                    );
                }
                let alone = host(&linear(&x.narrow(0, 21, 1).unwrap(), &w, &b).unwrap());
                assert_eq!(alone[0], all[21]);
            }
        }
    }

    /// µs per call for the flow head's shapes (F16) at its row counts, by
    /// row-group cap, against Candle's matmul and bias add (a benchmark).
    #[test]
    #[ignore = "benchmark"]
    fn bench_flow_shapes() {
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("bench_flow_shapes")
        else {
            return;
        };
        let time = |f: &dyn Fn() -> Tensor| {
            for _ in 0..5 {
                f();
            }
            device.synchronize().unwrap();
            let started = std::time::Instant::now();
            for _ in 0..200 {
                f();
            }
            device.synchronize().unwrap();
            started.elapsed().as_secs_f64() * 1e6 / 200.0
        };
        for (k, n) in [(1024usize, 1024usize), (1024, 4096), (4096, 1024)] {
            let w = xorshift_tensor(&[n, k], 1, 0.05, &device)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();
            let b = xorshift_tensor(&[n], 2, 0.5, &device)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();
            for rows in [6usize, 12, 24, 48, 96] {
                let x = xorshift_tensor(&[rows, k], 3, 1.0, &device)
                    .unwrap()
                    .to_dtype(DType::F16)
                    .unwrap();
                let candle = time(&|| {
                    x.matmul(&w.t().unwrap())
                        .unwrap()
                        .broadcast_add(&b)
                        .unwrap()
                });
                let caps = [8usize, 16, 24]
                    .map(|cap| {
                        let op = DenseMatVec { max_group: cap };
                        let t = time(&|| x.apply_op3_no_bwd(&w, &b, &op).unwrap());
                        format!("cap {cap}: {t:6.1}")
                    })
                    .join("  ");
                eprintln!("K {k:4} N {n:4} rows {rows:2}: candle {candle:6.1}  {caps}");
            }
        }
    }

    #[test]
    fn declines_what_it_cannot_serve() {
        let cpu = candle_core::Device::Cpu;
        let w = Tensor::zeros((8, 32), DType::F32, &cpu).unwrap();
        let b = Tensor::zeros(8, DType::F32, &cpu).unwrap();
        let x = Tensor::zeros((2, 32), DType::F32, &cpu).unwrap();
        assert!(!serves(&x, &w, &b));
    }
}
