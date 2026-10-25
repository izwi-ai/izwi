//! Q4_K / Q5_K / Q6_K / Q8_0 mat-vec for Metal decode batches (2..=8
//! activation rows).
//!
//! Candle's Metal mat-vec issues one dispatch per activation row, and each
//! one streams the whole quantized weight again. This kernel follows
//! llama.cpp's `mul_mv_ext` structure (ggml, MIT): eight lanes share a weight
//! row (four rows per simdgroup), each lane dequantizes one 16-weight chunk
//! per step into a float4x4 with ggml's dequantizers and dots it with up to
//! five activation rows read straight from device memory, and the row's
//! lanes reduce with simd_shuffle_down. Six to eight rows run as two
//! threadgroups of vectors.
//!
//! Measured on an M1 Pro (µs per call, Candle per-row mat-vec -> this):
//! - Q4_K 8192 x 2048: 1 row 131-229 -> 85-109, 4 rows 275 -> 180,
//!   8 rows 546 -> 337;
//! - Q6_K 2048 x 8192: 1 row 213-217 -> 117-156, 4 rows 468 -> 225;
//! - Q6_K 65536 x 2048 (LM head): 1 row 885-922 -> 808, 4 rows 3,336 ->
//!   1,626, 8 rows 6,670 -> 3,215.
//!
//! Two, sixteen and thirty-two lanes per row were slower (four was close),
//! and the earlier Candle-derived kernel that looped the rows inside
//! Candle's block walk reached only 4 rows 242, 8 rows 480 on Q4_K.
//! In the LFM2.5 decode step (ms per step, Candle -> this): 2 rows 14.0 ->
//! 12.8, 4 rows 23.9 -> 16.9; a single row was slower in the model (7.3 ->
//! 7.9) despite the isolated timing, so one row stays on Candle.
//! Results match Candle's to float rounding (sums regroup).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::backend::BackendStorage;
use candle_core::quantized::GgmlDType;
use candle_core::{bail, CpuStorage, CustomOp1, DType, Layout, MetalStorage, Shape, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputePipeline, Device as MetalDevice};

use super::metal_encoder::IzwiMetalCommandEncoderExt;

/// Most activation rows one dispatch serves.
pub(crate) const MAX_ROWS: usize = 8;
/// Simdgroups per threadgroup (`IZWI_KQ_NSG`).
const SIMDGROUPS: usize = 2;
/// Lanes sharing one weight row (`IZWI_KQ_NXPSG`).
const LANES_PER_ROW: usize = 8;

/// Activation vectors one threadgroup takes: all of them up to five, else
/// half (rounded up), so a batch of six to eight runs as two groups.
fn vectors_per_group(batch: usize) -> usize {
    if batch <= 5 {
        batch
    } else {
        batch.div_ceil(2)
    }
}

const QK_K: usize = 256;

/// Whether the kernels serve a `[_, cols]` weight of `dtype`.
pub(crate) fn serves(dtype: GgmlDType, cols: usize) -> bool {
    block_weights(dtype).is_some_and(|block| cols > 0 && cols.is_multiple_of(block))
}

/// Weights per quantization block of the types the kernels serve.
fn block_weights(dtype: GgmlDType) -> Option<usize> {
    match dtype {
        GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => Some(QK_K),
        GgmlDType::Q8_0 => Some(32),
        _ => None,
    }
}

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
        let block = block_weights(dtype)?;
        (cols > 0 && cols.is_multiple_of(block) && rows > 0).then_some(Self {
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

/// Give a Metal Q4_K / Q5_K / Q6_K / Q8_0 `QMatMul` a small-batch mat-vec over the same
/// weight. Candle's `QTensor` does not expose its buffer, so the weight is
/// read back once and uploaded into a buffer both share; `matmul` is
/// rebuilt over it (the old buffer is freed). `None`, leaving `matmul`
/// untouched, for another type, device or geometry, or a weight another
/// holder shares (a tied head would be duplicated).
pub(crate) fn share_qmatmul_weight(
    matmul: &mut candle_core::quantized::QMatMul,
) -> candle_core::Result<Option<MetalKQuantMatVec>> {
    use candle_core::quantized::{QMatMul, QStorage, QTensor};
    let QMatMul::QTensor(qtensor) = matmul else {
        return Ok(None);
    };
    let device = qtensor.device();
    let dtype = qtensor.dtype();
    let Some(block) = block_weights(dtype) else {
        return Ok(None);
    };
    if !device.is_metal() || std::sync::Arc::strong_count(qtensor) != 1 {
        return Ok(None);
    }
    let Ok((rows, cols)) = qtensor.shape().dims2() else {
        return Ok(None);
    };
    if cols == 0 || !cols.is_multiple_of(block) {
        return Ok(None);
    }
    let storage = QStorage::from_data(qtensor.data()?, &device, dtype)?;
    let QStorage::Metal(metal) = &storage else {
        return Ok(None);
    };
    let kernel = MetalKQuantMatVec::new(metal.buffer().clone(), dtype, rows, cols);
    *matmul = QMatMul::from_qtensor(QTensor::new(storage, (rows, cols))?)?;
    Ok(kernel)
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
        let kind = match self.dtype {
            GgmlDType::Q4K => "q4_K",
            GgmlDType::Q5K => "q5_K",
            GgmlDType::Q6K => "q6_K",
            GgmlDType::Q8_0 => "q8_0",
            dtype => bail!("izwi-kquant-mat-vec does not serve {dtype:?}"),
        };
        let vectors = vectors_per_group(batch);
        let rows_per_group = (32 / LANES_PER_ROW) * SIMDGROUPS;
        let function = format!("izwi_kq_ext_{kind}_r{vectors}");
        let device = storage.device().clone();
        let elements = batch * self.rows;
        let output = device.new_buffer(elements, DType::F32, "izwi-kquant-mat-vec")?;
        let pipeline = pipeline(device.metal_device(), &function)?;
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
        encoder.set_bytes(5, &(batch as u32));
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: self.rows.div_ceil(rows_per_group),
                height: batch.div_ceil(vectors),
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

typedef struct {
    union {
        struct {
            half d;
            half dmin;
        };
        half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
} block_q5_K;

#define QK8_0 32
typedef struct {
    half d;
    int8_t qs[QK8_0];
} block_q8_0;

// llama.cpp's `mul_mv_ext` structure (ggml, MIT): NXPSG lanes share one
// weight row, each lane dequantizes one 16-weight chunk per step into a
// float4x4 and dots it with R activation vectors read straight from device
// memory; a threadgroup covers R vectors (grid y covers the rest), and the
// lanes of a row reduce with simd_shuffle_down. The dequantizers are ggml's
// (as vendored by Candle): chunk `il` of a 256-weight block is its weights
// 16 il .. 16 il + 15.
constant constexpr short IZWI_KQ_NSG = 2;

static inline uchar2 izwi_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                          uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

void izwi_dequantize_q4_K(device const block_q4_K * xb, short il, thread float4x4 & reg) {
    device const uchar * q = xb->qs;
    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = izwi_scale_min_k4_just2(is, il/2, xb->scales);
    // In F32: ggml divides the half scale by 16 in half precision, which
    // lost bits on the LFM2.5 checkpoint's scales (logit error 0.47 instead
    // of 0.19 against the F32 reference; see the real-weight parity test).
    const float d   = il < 2 ? float(xb->d) : float(xb->d) / 16.0f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - ml;
    }
}

// ggml's Q5_K dequantizer (chunk `il` of 16), with the scale divide in F32
// as for Q4_K above.
void izwi_dequantize_q5_K(device const block_q5_K * xb, short il, thread float4x4 & reg) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;
    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    const uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = izwi_scale_min_k4_just2(is, il/2, xb->scales);
    const float d   = il < 2 ? float(xb->d) : float(xb->d) / 16.0f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    const float qh_val = il < 2 ? 16.0f : 256.0f;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0.0f)) - ml;
    }
}

// ggml's Q8_0 dequantizer: chunk `il` (0 or 1) of a 32-weight block.
void izwi_dequantize_q8_0(device const block_q8_0 * xb, short il, thread float4x4 & reg) {
    device const int8_t * qs = xb->qs;
    const float d = xb->d;
    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = qs[i + 16*il] * d;
    }
}

void izwi_dequantize_q6_K(device const block_q6_K * xb, short il, thread float4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * ql = (device const uint8_t *)xb->ql;
    device const uint8_t * qh = (device const uint8_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;
    ql = ql + 64*(il/8) + 32*((il/2)&1) + 16*(il&1);
    qh = qh + 32*(il/8) + 16*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;
    const uint16_t kmask1 = il>1 ? (il>2 ? 192 : 48) : (il>0 ? 12 : 3);
    const uint16_t kmask2 = il>1 ? 0xF0 : 0x0F;
    const float coef = il>1 ? 1.f/16.f : 1.f;
    const float ml = d_all * sc * 32.f;
    const float dl = d_all * sc * coef;
    for (int i = 0; i < 16; ++i) {
        const half q = il&1 ? ((ql[i] & kmask2) | ((qh[i] & kmask1) << 2))
                            : ((ql[i] & kmask2) | ((qh[i] & kmask1) << 4));
        reg[i/4][i%4] = dl * q - ml;
    }
}

template <typename block_q, short R, short NXPSG, short NL,
          void (*deq)(device const block_q *, short, thread float4x4 &)>
void izwi_kq_ext_impl(
        device const char  * src0,
        device const float * src1,
        device       float * dst,
        uint ne00, uint ne01, uint ne11,
        uint3 tgpig, ushort tiisg, ushort sgitg) {
    constexpr short NYPSG = 32 / NXPSG;
    const short tx = tiisg % NXPSG;
    const short ty = tiisg / NXPSG;
    const uint i01 = tgpig.x * (NYPSG * IZWI_KQ_NSG) + NYPSG * sgitg + ty;
    const uint i11 = tgpig.y * R;
    const uint nb = ne00 / (16 * NL);

    device const block_q * xq = (device const block_q *)src0 + (i01 < ne01 ? i01 : 0) * nb;
    device const float4x4 * y4x4[R];
    for (short ir = 0; ir < R; ++ir) {
        const uint vector = i11 + ir < ne11 ? i11 + ir : 0;
        y4x4[ir] = (device const float4x4 *)(src1 + (ulong)vector * ne00) + tx;
    }
    float sumf[R];
    for (short ir = 0; ir < R; ++ir) {
        sumf[ir] = 0.0f;
    }
    short cch = tx % NL;
    xq += tx / NL;
    for (uint ich = tx; 16 * ich < ne00; ich += NXPSG) {
        float4x4 lx;
        deq(xq, cch, lx);
        cch += NXPSG;
        if (cch >= NL) {
            xq += cch / NL;
            cch %= NL;
        }
        for (short ir = 0; ir < R; ++ir) {
            const float4x4 y = *y4x4[ir];
            sumf[ir] += dot(lx[0], y[0]) + dot(lx[1], y[1]) + dot(lx[2], y[2]) + dot(lx[3], y[3]);
            y4x4[ir] += NXPSG;
        }
    }
    for (short ir = 0; ir < R; ++ir) {
        float s = sumf[ir];
        if (NXPSG >= 32) { s += simd_shuffle_down(s, 16); }
        if (NXPSG >= 16) { s += simd_shuffle_down(s,  8); }
        if (NXPSG >=  8) { s += simd_shuffle_down(s,  4); }
        if (NXPSG >=  4) { s += simd_shuffle_down(s,  2); }
        if (NXPSG >=  2) { s += simd_shuffle_down(s,  1); }
        sumf[ir] = s;
    }
    if (tx == 0 && i01 < ne01) {
        for (short ir = 0; ir < R; ++ir) {
            if (i11 + ir < ne11) {
                dst[(ulong)(i11 + ir) * ne01 + i01] = sumf[ir];
            }
        }
    }
}

template <typename block_q, short R, short NXPSG, short NL,
          void (*deq)(device const block_q *, short, thread float4x4 &)>
kernel void izwi_kq_ext(
        device const char  * src0 [[buffer(0)]],
        device const float * src1 [[buffer(1)]],
        device       float * dst  [[buffer(2)]],
        constant uint & ne00 [[buffer(3)]],
        constant uint & ne01 [[buffer(4)]],
        constant uint & ne11 [[buffer(5)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    izwi_kq_ext_impl<block_q, R, NXPSG, NL, deq>(src0, src1, dst, ne00, ne01, ne11, tgpig, tiisg, sgitg);
}

typedef decltype(izwi_kq_ext<block_q4_K, 2, 8, 16, izwi_dequantize_q4_K>) izwi_kq_ext_t;

constant constexpr short IZWI_KQ_NXPSG = 8;

// NL: 16-weight chunks per block (16 for the K-quants, 2 for Q8_0).
#define IZWI_KQ_EXT(R) \
template [[host_name("izwi_kq_ext_q4_K_r" #R)]] kernel izwi_kq_ext_t izwi_kq_ext<block_q4_K, R, IZWI_KQ_NXPSG, 16, izwi_dequantize_q4_K>; \
template [[host_name("izwi_kq_ext_q5_K_r" #R)]] kernel izwi_kq_ext_t izwi_kq_ext<block_q5_K, R, IZWI_KQ_NXPSG, 16, izwi_dequantize_q5_K>; \
template [[host_name("izwi_kq_ext_q6_K_r" #R)]] kernel izwi_kq_ext_t izwi_kq_ext<block_q6_K, R, IZWI_KQ_NXPSG, 16, izwi_dequantize_q6_K>; \
template [[host_name("izwi_kq_ext_q8_0_r" #R)]] kernel izwi_kq_ext_t izwi_kq_ext<block_q8_0, R, IZWI_KQ_NXPSG, 2, izwi_dequantize_q8_0>;

IZWI_KQ_EXT(2)
IZWI_KQ_EXT(3)
IZWI_KQ_EXT(4)
IZWI_KQ_EXT(5)
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
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "small_batches_match_candle_per_row_mat_vec",
        ) else {
            return;
        };
        for (dtype, rows, cols) in [
            (GgmlDType::Q4K, 64, 512),
            (GgmlDType::Q4K, 12, 768),
            (GgmlDType::Q5K, 64, 512),
            (GgmlDType::Q5K, 10, 1024),
            (GgmlDType::Q6K, 64, 512),
            (GgmlDType::Q6K, 6, 1024),
            (GgmlDType::Q8_0, 64, 512),
            (GgmlDType::Q8_0, 9, 96),
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
                // Relative to the output scale: the sums only regroup, so
                // the difference is float rounding.
                let scale = expected.iter().fold(0f32, |max, e| max.max(e.abs()));
                for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        (a - e).abs() <= 2e-6 * scale,
                        "{dtype:?} {rows}x{cols} batch {batch} [{index}]: {a} vs {e} (scale {scale})"
                    );
                }
            }
        }
    }

    /// A `QMatMul` loaded the ordinary way gets a kernel over its own
    /// (re-uploaded) weight that matches it; a shared or unsupported weight
    /// is left alone.
    #[test]
    fn a_loaded_qmatmul_shares_its_weight_with_the_small_batch_kernel() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "a_loaded_qmatmul_shares_its_weight_with_the_small_batch_kernel",
        ) else {
            return;
        };
        let (rows, cols) = (48, 512);
        let values = Tensor::from_vec(
            xorshift(0x51A2, rows * cols, 2.0),
            (rows, cols),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        let loaded = || QTensor::quantize_onto(&values, GgmlDType::Q4K, &device).unwrap();
        let reference = candle_core::quantized::QMatMul::from_qtensor(loaded()).unwrap();
        let mut matmul = candle_core::quantized::QMatMul::from_qtensor(loaded()).unwrap();
        let kernel = share_qmatmul_weight(&mut matmul)
            .unwrap()
            .expect("a Q4_K Metal weight takes the kernel");
        for batch in [2usize, 5, MAX_ROWS] {
            let x = Tensor::from_vec(
                xorshift(batch as u32, batch * cols, 3.0),
                (batch, 1, cols),
                &device,
            )
            .unwrap();
            let expected = reference.forward(&x).unwrap().flatten_all().unwrap();
            let expected = expected.to_vec1::<f32>().unwrap();
            for actual in [kernel.forward(&x).unwrap(), matmul.forward(&x).unwrap()] {
                let actual = actual.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let scale = expected.iter().fold(0f32, |max, e| max.max(e.abs()));
                for (a, e) in actual.iter().zip(&expected) {
                    assert!((a - e).abs() <= 2e-6 * scale, "batch {batch}: {a} vs {e}");
                }
            }
        }
        // A weight with another holder is not duplicated.
        let shared = std::sync::Arc::new(loaded());
        let mut tied = candle_core::quantized::QMatMul::from_arc(shared.clone()).unwrap();
        assert!(share_qmatmul_weight(&mut tied).unwrap().is_none());
        // Unsupported types keep Candle.
        let mut q4_0 = candle_core::quantized::QMatMul::from_qtensor(
            QTensor::quantize_onto(&values, GgmlDType::Q4_0, &device).unwrap(),
        )
        .unwrap();
        assert!(share_qmatmul_weight(&mut q4_0).unwrap().is_none());
    }

    /// Timing against Candle's per-row mat-vec on LFM2.5-sized weights.
    #[test]
    #[ignore = "timing"]
    fn small_batch_timing() {
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("small_batch_timing")
        else {
            return;
        };
        for (dtype, rows, cols) in [
            (GgmlDType::Q4K, 8192, 2048),
            (GgmlDType::Q4K, 2048, 8192),
            (GgmlDType::Q6K, 2048, 8192),
            (GgmlDType::Q6K, 65536, 2048),
        ] {
            let (ext, matmul) = weight(&device, dtype, rows, cols);
            for batch in [2usize, 4, 8] {
                let x = Tensor::from_vec(xorshift(7, batch * cols, 2.0), (batch, 1, cols), &device)
                    .unwrap();
                let time = |f: &dyn Fn() -> Tensor| {
                    for _ in 0..8 {
                        f();
                    }
                    device.synchronize().unwrap();
                    let started = std::time::Instant::now();
                    for _ in 0..100 {
                        f();
                    }
                    device.synchronize().unwrap();
                    started.elapsed().as_secs_f64() * 1e6 / 100.0
                };
                let candle = time(&|| matmul.forward(&x).unwrap());
                let ext = time(&|| ext.forward(&x).unwrap());
                eprintln!(
                    "{dtype:?} {rows}x{cols} batch {batch}: candle {candle:7.1} us, izwi {ext:7.1} us"
                );
            }
        }
    }

    #[test]
    fn unsupported_shapes_are_refused() {
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("unsupported_shapes_are_refused")
        else {
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
        assert!(MetalKQuantMatVec::new(buffer.clone(), GgmlDType::Q4_0, 8, 256).is_none());
        assert!(MetalKQuantMatVec::new(buffer.clone(), GgmlDType::Q8_0, 8, 48).is_none());
        assert!(MetalKQuantMatVec::new(buffer, GgmlDType::Q6K, 8, 300).is_none());
    }
}
