//! Small-batch Q4_K / Q6_K mat-vec for Metal.
//!
//! Candle's Metal mat-vec issues one dispatch per activation row, and each
//! one streams the whole quantized weight again, so a decode batch of B rows
//! reads the weights B times. These kernels walk each weight block once for
//! up to [`MAX_ROWS`] rows. They are Candle's `kernel_mul_mv_q4_K_f32` and
//! `kernel_mul_mv_q6_K_f32` with the activation rows looped inside the block
//! walk.
//! - Q4_K: one simdgroup per four weight rows, as in Candle. The per-row
//!   weight words are reread from cache for each activation row, and the
//!   results are bit-identical to Candle's.
//! - Q6_K: two simdgroups per threadgroup, one weight row each. Each block's
//!   sixteen quants per lane are decoded once and reused for every row; the
//!   results match Candle's to an ulp.
//!
//! Decoding the Q4_K quants once per block as well (scales folded into F32
//! weights) was slower: 215 against 121 µs at two rows of an 8192 x 2048
//! weight, from register pressure.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::quantized::GgmlDType;
use candle_core::{bail, CpuStorage, CustomOp1, DType, Layout, MetalStorage, Shape, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputePipeline, Device as MetalDevice};

use super::metal_encoder::IzwiMetalCommandEncoderExt;

/// Most activation rows one dispatch serves.
pub(crate) const MAX_ROWS: usize = 8;
const QK_K: usize = 256;

/// A Q4_K or Q6_K weight `[rows, cols]` resident in a Metal buffer, in GGUF
/// block layout. The buffer is the one the weight's `QTensor` owns, shared.
#[derive(Clone)]
pub(crate) struct MetalKQuantMatVec {
    buffer: Buffer,
    dtype: GgmlDType,
    rows: usize,
    cols: usize,
}

impl std::fmt::Debug for MetalKQuantMatVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalKQuantMatVec")
            .field("dtype", &self.dtype)
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .finish()
    }
}

impl MetalKQuantMatVec {
    /// `None` for a type or geometry the kernels do not cover.
    pub(crate) fn new(buffer: Buffer, dtype: GgmlDType, rows: usize, cols: usize) -> Option<Self> {
        let rows_per_group = match dtype {
            GgmlDType::Q4K => 4,
            GgmlDType::Q6K => 2,
            _ => return None,
        };
        (cols > 0 && cols.is_multiple_of(QK_K) && rows > 0 && rows.is_multiple_of(rows_per_group))
            .then_some(Self {
                buffer,
                dtype,
                rows,
                cols,
            })
    }

    /// Whether `x` (`[..., cols]`, F32, on Metal) has 2..=[`MAX_ROWS`] rows.
    pub(crate) fn accepts(&self, x: &Tensor) -> bool {
        let rows = x.elem_count() / self.cols.max(1);
        x.device().is_metal()
            && x.dtype() == DType::F32
            && x.dims().last() == Some(&self.cols)
            && (2..=MAX_ROWS).contains(&rows)
    }

    /// `x @ weightᵀ` for `x` of shape `[..., cols]`; the result is
    /// `[..., rows]`.
    pub(crate) fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        if !self.accepts(x) {
            bail!(
                "izwi k-quant mat-vec needs 2..={MAX_ROWS} F32 Metal rows of {}, got {:?} {:?}",
                self.cols,
                x.dims(),
                x.dtype()
            )
        }
        let mut dims = x.dims().to_vec();
        let batch = x.elem_count() / self.cols;
        *dims.last_mut().expect("checked above") = self.rows;
        x.contiguous()?
            .reshape((batch, self.cols))?
            .apply_op1_no_bwd(self)?
            .reshape(dims)
    }
}

impl CustomOp1 for MetalKQuantMatVec {
    fn name(&self) -> &'static str {
        "izwi-kquant-mat-vec"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        bail!("izwi-kquant-mat-vec runs on Metal only")
    }

    fn metal_fwd(
        &self,
        storage: &MetalStorage,
        layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let (batch, cols) = layout.shape().dims2()?;
        if storage.dtype() != DType::F32 || cols != self.cols || !layout.is_contiguous() {
            bail!(
                "izwi-kquant-mat-vec needs contiguous F32 rows of {}",
                self.cols
            )
        }
        if !(2..=MAX_ROWS).contains(&batch) {
            bail!("izwi-kquant-mat-vec serves 2..={MAX_ROWS} rows, got {batch}")
        }
        let (kind, rows_per_group, threads) = match self.dtype {
            GgmlDType::Q4K => ("q4_K", 4, 32),
            GgmlDType::Q6K => ("q6_K", 2, 64),
            dtype => bail!("izwi-kquant-mat-vec does not serve {dtype:?}"),
        };
        let device = storage.device().clone();
        let elements = batch * self.rows;
        let output = device.new_buffer(elements, DType::F32, "izwi-kquant-mat-vec")?;
        let pipeline = pipeline(
            device.metal_device(),
            &format!("izwi_mul_mv_ext_{kind}_f32_r{batch}"),
        )?;
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-kquant-mat-vec");
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_input_buffer(0, Some(&self.buffer), 0);
        encoder.set_input_buffer(
            1,
            Some(storage.buffer()),
            layout.start_offset() * DType::F32.size_in_bytes(),
        );
        encoder.set_output_buffer(2, Some(&output), 0);
        encoder.set_bytes(3, &(self.cols as u32));
        encoder.set_bytes(4, &(self.rows as u32));
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: self.rows / rows_per_group,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        Ok((
            MetalStorage::new(output, device, elements, DType::F32),
            Shape::from((batch, self.rows)),
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

#define QK_K 256
#define K_SCALE_SIZE 12
#define N_DST 4

typedef struct {
    union {
        struct {
            half d;
            half dmin;
        };
        half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

typedef struct {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t  scales[QK_K/16];
    half d;
} block_q6_K;

// Candle's kernel_mul_mv_q4_K_f32 with R activation rows: one simdgroup
// computes N_DST weight rows for every activation row.
template <int R>
kernel void izwi_mul_mv_ext_q4_K_f32(
        device const void  * src0 [[buffer(0)]],
        device const float * src1 [[buffer(1)]],
        device       float * dst  [[buffer(2)]],
        constant uint & ne00 [[buffer(3)]],
        constant uint & ne01 [[buffer(4)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint  tiisg [[thread_index_in_simdgroup]]) {
    const uint16_t kmask1 = 0x3f3f;
    const uint16_t kmask2 = 0x0f0f;
    const uint16_t kmask3 = 0xc0c0;

    const int ix = tiisg/8;
    const int it = tiisg%8;
    const int iq = it/4;
    const int ir = it%4;

    const int nb = ne00/QK_K;
    const int first_row = tgpig.x * N_DST;

    device const block_q4_K * x = (device const block_q4_K *) src0 + first_row * nb;

    float yl[16];
    float yh[16];
    float sumf[R][N_DST];
    for (int b = 0; b < R; ++b) {
        for (int row = 0; row < N_DST; ++row) {
            sumf[b][row] = 0.f;
        }
    }

    const int step = sizeof(block_q4_K) * nb / 2;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        for (int b = 0; b < R; ++b) {
            device const float * y4 = src1 + b * ne00 + ib * QK_K + 64 * iq + 8 * ir;

            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; ++i) {
                yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
                yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
                yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
                yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
            }

            device const uint16_t * sc = (device const uint16_t *)x[ib].scales + iq;
            device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &x[ib].d;

            for (int row = 0; row < N_DST; row++) {
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                device const uint16_t * q2 = q1 + 32;

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                for (int i = 0; i < 8; i += 2) {
                    acc1[0] += yl[i+0] * (q1[i/2] & 0x000F);
                    acc1[1] += yl[i+1] * (q1[i/2] & 0x0F00);
                    acc1[2] += yl[i+8] * (q1[i/2] & 0x00F0);
                    acc1[3] += yl[i+9] * (q1[i/2] & 0xF000);
                    acc2[0] += yh[i+0] * (q2[i/2] & 0x000F);
                    acc2[1] += yh[i+1] * (q2[i/2] & 0x0F00);
                    acc2[2] += yh[i+8] * (q2[i/2] & 0x00F0);
                    acc2[3] += yh[i+9] * (q2[i/2] & 0xF000);
                }

                float dall = dh[0];
                float dmin = dh[1];
                sumf[b][row] += dall * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                        (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                        (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                        (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                                dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

                q1 += step;
                sc += step;
                dh += step;
            }
        }
    }

    for (int b = 0; b < R; ++b) {
        for (int row = 0; row < N_DST; ++row) {
            const float all_sum = simd_sum(sumf[b][row]);
            if (tiisg == 0) {
                dst[b * ne01 + first_row + row] = all_sum;
            }
        }
    }
}

// Candle's kernel_mul_mv_q6_K_f32 with R activation rows: one simdgroup per
// weight row; each lane decodes its sixteen quants of a block once.
template <int R>
kernel void izwi_mul_mv_ext_q6_K_f32(
        device const void  * src0 [[buffer(0)]],
        device const float * src1 [[buffer(1)]],
        device       float * dst  [[buffer(2)]],
        constant uint & ne00 [[buffer(3)]],
        constant uint & ne01 [[buffer(4)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint  tiisg [[thread_index_in_simdgroup]],
        uint  sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint8_t kmask1 = 0x03;
    const uint8_t kmask2 = 0x0C;
    const uint8_t kmask3 = 0x30;
    const uint8_t kmask4 = 0xC0;

    const int nb = ne00/QK_K;
    const int row = 2 * tgpig.x + sgitg;

    device const block_q6_K * x = (device const block_q6_K *) src0 + row * nb;

    float sumf[R];
    for (int b = 0; b < R; ++b) {
        sumf[b] = 0.f;
    }

    const int tid  = tiisg/2;
    const int ix   = tiisg%2;
    const int ip   = tid/8;
    const int il   = tid%8;
    const int n    = 4;
    const int l0   = n*il;
    const int is   = 8*ip + l0/16;

    const int y_offset = 128*ip + l0;
    const int q_offset_l = 64*ip + l0;
    const int q_offset_h = 32*ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;

        const float dall = x[i].d;

        float w[4][4];
        for (int l = 0; l < n; ++l) {
            w[0][l] = (int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32;
            w[1][l] = (int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32;
            w[2][l] = (int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32;
            w[3][l] = (int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32;
        }
        const float s0 = sc[0];
        const float s2 = sc[2];
        const float s4 = sc[4];
        const float s6 = sc[6];

        for (int b = 0; b < R; ++b) {
            device const float * y = src1 + b * ne00 + i * QK_K + y_offset;
            float4 sums = {0.f, 0.f, 0.f, 0.f};
            for (int l = 0; l < n; ++l) {
                sums[0] += y[l+ 0] * w[0][l];
                sums[1] += y[l+32] * w[1][l];
                sums[2] += y[l+64] * w[2][l];
                sums[3] += y[l+96] * w[3][l];
            }
            sumf[b] += dall * (sums[0] * s0 + sums[1] * s2 + sums[2] * s4 + sums[3] * s6);
        }
    }

    for (int b = 0; b < R; ++b) {
        const float tot = simd_sum(sumf[b]);
        if (tiisg == 0) {
            dst[b * ne01 + row] = tot;
        }
    }
}

#define IZWI_KQUANT_MV_EXT(R) \
template [[host_name("izwi_mul_mv_ext_q4_K_f32_r" #R)]] kernel void izwi_mul_mv_ext_q4_K_f32<R>( \
    device const void *, device const float *, device float *, constant uint &, constant uint &, \
    uint3, uint); \
template [[host_name("izwi_mul_mv_ext_q6_K_f32_r" #R)]] kernel void izwi_mul_mv_ext_q6_K_f32<R>( \
    device const void *, device const float *, device float *, constant uint &, constant uint &, \
    uint3, uint, uint);

IZWI_KQUANT_MV_EXT(2)
IZWI_KQUANT_MV_EXT(3)
IZWI_KQUANT_MV_EXT(4)
IZWI_KQUANT_MV_EXT(5)
IZWI_KQUANT_MV_EXT(6)
IZWI_KQUANT_MV_EXT(7)
IZWI_KQUANT_MV_EXT(8)
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::{QStorage, QTensor};
    use candle_core::Module;

    fn xorshift(seed: u32, len: usize, scale: f32) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32 - 0.5) * scale
            })
            .collect()
    }

    /// A quantized `[rows, cols]` weight on Metal, its buffer, and the
    /// Candle `QMatMul` over the same buffer.
    fn weight(
        device: &candle_core::Device,
        dtype: GgmlDType,
        rows: usize,
        cols: usize,
    ) -> (MetalKQuantMatVec, candle_core::quantized::QMatMul) {
        let values = Tensor::from_vec(
            xorshift(0x1234_5678 ^ rows as u32, rows * cols, 2.0),
            (rows, cols),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let cpu = QTensor::quantize(&values, dtype).unwrap();
        let storage = QStorage::from_data(cpu.data().unwrap(), device, dtype).unwrap();
        let buffer = match &storage {
            QStorage::Metal(metal) => metal.buffer().clone(),
            _ => unreachable!("Metal storage"),
        };
        let qtensor = QTensor::new(storage, (rows, cols)).unwrap();
        (
            MetalKQuantMatVec::new(buffer, dtype, rows, cols).unwrap(),
            candle_core::quantized::QMatMul::from_qtensor(qtensor).unwrap(),
        )
    }

    #[test]
    fn small_batches_match_candle_per_row_mat_vec() {
        let Some(device) = crate::backends::metal_device_if_available(0) else {
            return;
        };
        for (dtype, rows, cols) in [
            (GgmlDType::Q4K, 64, 512),
            (GgmlDType::Q4K, 12, 768),
            (GgmlDType::Q6K, 64, 512),
            (GgmlDType::Q6K, 6, 1024),
        ] {
            let (ext, matmul) = weight(&device, dtype, rows, cols);
            for batch in 2..=MAX_ROWS {
                let x = Tensor::from_vec(
                    xorshift(batch as u32 * 977, batch * cols, 4.0),
                    (batch, 1, cols),
                    &device,
                )
                .unwrap();
                // Candle's mat-vec path: one dispatch per (row, 1, cols).
                let expected = matmul.forward(&x).unwrap();
                let actual = ext.forward(&x).unwrap();
                assert_eq!(actual.dims(), &[batch, 1, rows]);
                let expected = expected.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        (a - e).abs() <= 1e-4 * e.abs().max(1.0),
                        "{dtype:?} {rows}x{cols} batch {batch} [{index}]: {a} vs {e}"
                    );
                }
            }
        }
    }

    /// Timing against Candle's per-row mat-vec on an LFM2.5-sized weight.
    #[test]
    #[ignore = "timing"]
    fn small_batch_timing() {
        let Some(device) = crate::backends::metal_device_if_available(0) else {
            return;
        };
        for (dtype, rows, cols) in [(GgmlDType::Q4K, 8192, 2048), (GgmlDType::Q6K, 2048, 8192)] {
            let (ext, matmul) = weight(&device, dtype, rows, cols);
            for batch in [1usize, 2, 4, 8] {
                let x = Tensor::from_vec(xorshift(7, batch * cols, 2.0), (batch, 1, cols), &device)
                    .unwrap();
                let time = |f: &dyn Fn() -> Tensor| {
                    for _ in 0..8 {
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
                let candle = time(&|| matmul.forward(&x).unwrap());
                let small = if batch >= 2 {
                    time(&|| ext.forward(&x).unwrap())
                } else {
                    f64::NAN
                };
                eprintln!("{dtype:?} {rows}x{cols} batch {batch}: candle {candle:7.1} us, small-batch {small:7.1} us");
            }
        }
    }

    #[test]
    fn unsupported_shapes_are_refused() {
        let Some(device) = crate::backends::metal_device_if_available(0) else {
            return;
        };
        let (ext, _) = weight(&device, GgmlDType::Q4K, 8, 256);
        let one = Tensor::zeros((1, 256), DType::F32, &device).unwrap();
        let nine = Tensor::zeros((9, 256), DType::F32, &device).unwrap();
        let wide = Tensor::zeros((2, 512), DType::F32, &device).unwrap();
        let half = Tensor::zeros((2, 256), DType::F16, &device).unwrap();
        for x in [&one, &nine, &wide, &half] {
            assert!(!ext.accepts(x));
            assert!(ext.forward(x).is_err());
        }
        let buffer = ext.buffer.clone();
        assert!(MetalKQuantMatVec::new(buffer.clone(), GgmlDType::Q8_0, 8, 256).is_none());
        assert!(MetalKQuantMatVec::new(buffer.clone(), GgmlDType::Q4K, 6, 256).is_none());
        assert!(MetalKQuantMatVec::new(buffer, GgmlDType::Q6K, 8, 300).is_none());
    }
}
