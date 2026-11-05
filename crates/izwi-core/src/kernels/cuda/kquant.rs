//! GGUF K-quant matrix-vector products with caller-owned activations.
//!
//! On CUDA, Candle's `QMatMul` runs llama.cpp's MMVQ kernels for up to eight
//! input rows. It first quantizes the input to Q8_1 into one process-global,
//! grow-only scratch buffer (`candle-core` `quantized/fast_mmvq.rs`). That
//! buffer has two costs:
//! - a captured CUDA graph would bake in its address, and a later growth frees
//!   it, so the projection cannot be captured;
//! - every projection quantizes its input again, even when several weights
//!   read the same input.
//!
//! Here the Q8_1 activations are an ordinary allocation owned by the caller:
//! one [`quantize`] per input, then one [`mat_vec`] per weight. Both call
//! candle-kernels' public launchers with the arguments `QMatMul` uses, so the
//! outputs match it bit for bit. Allocations are stream-ordered, so they
//! become graph memory under capture.
//!
//! Above eight rows `QMatMul` runs llama.cpp's tiled MMQ kernels
//! (`quantized/fast_mmq.rs`) with the same two costs: the input is quantized
//! into Q8_1 tiles (`block_q8_1_mmq`, 128 values and a scale layout that
//! follows the weight type) in a second process-global, grow-only workspace,
//! and the stream-k launch writes partial tiles to a third one for its fixup
//! pass. [`quantize_tiles`] and [`mat_mul`] call the same launchers with the
//! same arguments over caller-owned, stream-ordered buffers: one tile
//! quantization per input and scale layout, one launch pair per weight, and
//! a fixup buffer per launch that is freed behind it.
use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Result, Tensor};

/// Input rows one MMVQ launch serves (the kernels' batch limit).
pub const MAX_ROWS: usize = 8;

/// llama.cpp pads every Q8_1 activation row to this many values.
const ROW_PADDING: usize = 512;
const Q8_1_VALUES: usize = 32;
const Q8_1_BYTES: usize = 36;

/// A `block_q8_1_mmq` tile: 128 quantized values and 16 bytes of scales.
const TILE_VALUES: usize = 4 * Q8_1_VALUES;
const TILE_BYTES: usize = 4 * Q8_1_VALUES + 16;
/// The MMQ kernels read up to one widest column tile (`mmq_x` = 128) past
/// the last input row; `QMatMul` pads its workspace by that many tiles.
const TILE_PADDING: usize = 128 * TILE_BYTES;
/// A stream-k fixup holds one `mmq_x` x `mmq_y` (at most 128 x 128) F32
/// partial tile per SM.
const FIXUP_TILE_VALUES: usize = 128 * 128;

/// A GGUF weight on CUDA in a format the MMVQ launchers read directly.
#[derive(Clone)]
pub struct KQuantWeight {
    weight: Arc<QTensor>,
    dtype: GgmlDType,
    rows: usize,
    cols: usize,
}

impl KQuantWeight {
    /// The quantized weight behind `matmul` when it is a CUDA Q4_K, Q5_K,
    /// Q6_K or Q8_0 tensor whose rows hold whole blocks; `None` otherwise
    /// (dense weights, other formats, other devices).
    pub fn from_qmatmul(matmul: &QMatMul) -> Option<Self> {
        let QMatMul::QTensor(weight) = matmul else {
            return None;
        };
        let dtype = weight.dtype();
        if !weight.device().is_cuda()
            || !matches!(
                dtype,
                GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K | GgmlDType::Q8_0
            )
        {
            return None;
        }
        let (rows, cols) = weight.shape().dims2().ok()?;
        if rows == 0 || cols == 0 || !cols.is_multiple_of(dtype.block_size()) {
            return None;
        }
        Some(Self {
            weight: weight.clone(),
            dtype,
            rows,
            cols,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> candle_core::Device {
        self.weight.device()
    }
}

/// Q8_1 activations for up to [`MAX_ROWS`] input rows, shared by every
/// weight that reads them.
pub struct Q8Activations {
    #[cfg(feature = "cuda")]
    data: candle_core::cuda_backend::cudarc::driver::CudaSlice<u8>,
    device: candle_core::Device,
    /// Leading dims of the input; the product is the row count.
    leading: Vec<usize>,
    rows: usize,
    cols: usize,
}

impl Q8Activations {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }
}

/// Whether [`quantize`] accepts `x`: a contiguous F32 CUDA tensor whose
/// leading dims hold 1 to [`MAX_ROWS`] rows.
pub fn accepts(x: &Tensor) -> bool {
    let dims = x.dims();
    let rows = dims[..dims.len().saturating_sub(1)]
        .iter()
        .product::<usize>();
    x.device().is_cuda()
        && x.dtype() == candle_core::DType::F32
        && !dims.is_empty()
        && (1..=MAX_ROWS).contains(&rows)
        && x.is_contiguous()
}

fn padded(cols: usize) -> usize {
    cols.div_ceil(ROW_PADDING) * ROW_PADDING
}

/// Quantize `x` (`[.., cols]`, see [`accepts`]) to Q8_1 once for every weight
/// that reads it.
pub fn quantize(x: &Tensor) -> Result<Q8Activations> {
    if !accepts(x) {
        candle_core::bail!(
            "K-quant MMVQ input must be contiguous F32 CUDA with 1..={MAX_ROWS} rows, got {:?} {:?} on {:?}",
            x.dims(),
            x.dtype(),
            x.device().location()
        )
    }
    let dims = x.dims();
    let cols = dims[dims.len() - 1];
    let leading = dims[..dims.len() - 1].to_vec();
    let rows = leading.iter().product::<usize>();
    #[cfg(feature = "cuda")]
    {
        let data = cuda_impl::quantize(x, rows, cols)?;
        Ok(Q8Activations {
            data,
            device: x.device().clone(),
            leading,
            rows,
            cols,
        })
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (leading, rows, cols);
        candle_core::bail!("K-quant MMVQ needs the cuda feature")
    }
}

/// The scale layout of the Q8_1 tiles an MMQ launch for a weight type reads
/// (llama.cpp's `mmq_get_q8_1_ds_layout`): inputs quantized once per layout
/// serve every weight of that layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TileScales {
    /// One scale per 32 values (Q6_K, Q8_0).
    D4,
    /// A scale and a sum per 32 values (Q4_K, Q5_K).
    Ds4,
}

impl KQuantWeight {
    /// The tile layout [`mat_mul`] reads for this weight.
    pub fn tile_scales(&self) -> TileScales {
        match self.dtype {
            GgmlDType::Q4K | GgmlDType::Q5K => TileScales::Ds4,
            _ => TileScales::D4,
        }
    }
}

/// Q8_1 tiles of any number of input rows in one scale layout, shared by
/// every weight of that layout ([`KQuantWeight::tile_scales`]).
pub struct Q8Tiles {
    #[cfg(feature = "cuda")]
    data: candle_core::cuda_backend::cudarc::driver::CudaSlice<u8>,
    device: candle_core::Device,
    leading: Vec<usize>,
    rows: usize,
    cols: usize,
    scales: TileScales,
}

impl Q8Tiles {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn scales(&self) -> TileScales {
        self.scales
    }
}

/// Whether [`quantize_tiles`] accepts `x`: a contiguous F32 CUDA tensor with
/// at least one row, of any count.
pub fn accepts_tiles(x: &Tensor) -> bool {
    let dims = x.dims();
    let rows = dims[..dims.len().saturating_sub(1)]
        .iter()
        .product::<usize>();
    x.device().is_cuda()
        && x.dtype() == candle_core::DType::F32
        && !dims.is_empty()
        && rows >= 1
        && i32::try_from(rows).is_ok()
        && x.is_contiguous()
}

/// Bytes of Q8_1 tiles for `rows` inputs of `cols` values, padding included
/// (`QMatMul`'s workspace size for the same input).
fn tile_bytes(rows: usize, cols: usize) -> usize {
    rows * (padded(cols) / TILE_VALUES) * TILE_BYTES + TILE_PADDING
}

/// Quantize `x` (`[.., cols]`, see [`accepts_tiles`]) to Q8_1 tiles in the
/// `scales` layout, once for every weight of that layout that reads it.
pub fn quantize_tiles(x: &Tensor, scales: TileScales) -> Result<Q8Tiles> {
    if !accepts_tiles(x) {
        candle_core::bail!(
            "K-quant MMQ input must be contiguous F32 CUDA with at least one row, got {:?} {:?} on {:?}",
            x.dims(),
            x.dtype(),
            x.device().location()
        )
    }
    let dims = x.dims();
    let cols = dims[dims.len() - 1];
    let leading = dims[..dims.len() - 1].to_vec();
    let rows = leading.iter().product::<usize>();
    #[cfg(feature = "cuda")]
    {
        let data = cuda_impl::quantize_tiles(x, rows, cols, scales)?;
        Ok(Q8Tiles {
            data,
            device: x.device().clone(),
            leading,
            rows,
            cols,
            scales,
        })
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (leading, rows, cols, scales);
        candle_core::bail!("K-quant MMQ needs the cuda feature")
    }
}

/// `x · weightᵀ` for every row of the tiles: F32 `[leading.., weight.rows()]`,
/// bit for bit what `QMatMul` returns for the same rows above eight.
pub fn mat_mul(weight: &KQuantWeight, x: &Q8Tiles) -> Result<Tensor> {
    if weight.cols != x.cols
        || weight.tile_scales() != x.scales
        || !weight.weight.device().same_device(&x.device)
    {
        candle_core::bail!(
            "K-quant MMQ weight {:?} [{}, {}] does not match {} input columns in {:?} tiles",
            weight.dtype,
            weight.rows,
            weight.cols,
            x.cols,
            x.scales
        )
    }
    let mut shape = x.leading.clone();
    shape.push(weight.rows);
    #[cfg(feature = "cuda")]
    {
        cuda_impl::mat_mul(weight, x, shape)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = shape;
        candle_core::bail!("K-quant MMQ needs the cuda feature")
    }
}

/// `x · weightᵀ` for every row of `x`: F32 `[leading.., weight.rows()]`.
pub fn mat_vec(weight: &KQuantWeight, x: &Q8Activations) -> Result<Tensor> {
    if weight.cols != x.cols || !weight.weight.device().same_device(&x.device) {
        candle_core::bail!(
            "K-quant MMVQ weight [{}, {}] does not match {} input columns",
            weight.rows,
            weight.cols,
            x.cols
        )
    }
    let mut shape = x.leading.clone();
    shape.push(weight.rows);
    #[cfg(feature = "cuda")]
    {
        cuda_impl::mat_vec(weight, x, shape)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = shape;
        candle_core::bail!("K-quant MMVQ needs the cuda feature")
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use std::ffi::c_void;

    use candle_core::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr};
    use candle_core::op::BackpropOp;
    use candle_core::quantized::GgmlDType;
    use candle_core::{CudaStorage, Result, Shape, Storage, Tensor};

    use super::{
        padded, tile_bytes, KQuantWeight, Q8Activations, Q8Tiles, TileScales, FIXUP_TILE_VALUES,
        Q8_1_BYTES, Q8_1_VALUES,
    };

    type Launcher = unsafe extern "C" fn(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );

    extern "C" {
        // The candle-kernels launchers use the runtime launch syntax and
        // return nothing; the runtime's last error is the only report of a
        // failed launch. candle-kernels links cudart.
        fn cudaGetLastError() -> i32;
    }

    fn launcher(dtype: GgmlDType) -> Option<Launcher> {
        use candle_kernels::ffi;
        Some(match dtype {
            GgmlDType::Q4K => ffi::launch_mmvq_gguf_q4_k_f32_plain,
            GgmlDType::Q5K => ffi::launch_mmvq_gguf_q5_k_f32_plain,
            GgmlDType::Q6K => ffi::launch_mmvq_gguf_q6_k_f32_plain,
            GgmlDType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_f32_plain,
            _ => return None,
        })
    }

    fn check_launch(what: &str) -> Result<()> {
        // SAFETY: plain query of (and reset of) the calling thread's runtime
        // error state.
        let code = unsafe { cudaGetLastError() };
        if code != 0 {
            candle_core::bail!("{what} launch failed with CUDA runtime error {code}")
        }
        Ok(())
    }

    pub(super) fn quantize(x: &Tensor, rows: usize, cols: usize) -> Result<CudaSlice<u8>> {
        let device = x.device().as_cuda_device()?;
        let stream = device.cuda_stream();
        let cols_padded = padded(cols);
        let bytes = rows * (cols_padded / Q8_1_VALUES) * Q8_1_BYTES;
        // SAFETY: the quantize kernel writes every byte, padding included.
        let data = unsafe { device.alloc::<u8>(bytes)? };
        let (storage, layout) = x.storage_and_layout();
        let Storage::Cuda(cuda) = &*storage else {
            candle_core::bail!("K-quant MMVQ input must be a CUDA tensor")
        };
        let Some((start, end)) = layout.contiguous_offsets() else {
            candle_core::bail!("K-quant MMVQ input must be contiguous")
        };
        let input = cuda.as_cuda_slice::<f32>()?.slice(start..end);
        let input_ptr = input.device_ptr(&stream).0 as *const c_void;
        let data_ptr = data.device_ptr(&stream).0 as *mut c_void;
        // SAFETY: both buffers live on this device and stream, `input` holds
        // `rows * cols` F32 values and `data` the padded Q8_1 rows.
        unsafe {
            candle_kernels::ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                input_ptr,
                data_ptr,
                cols as i32,
                cols_padded as i32,
                rows as i32,
                stream.cu_stream() as *mut c_void,
            );
        }
        check_launch("Q8_1 quantize")?;
        Ok(data)
    }

    pub(super) fn mat_vec(
        weight: &KQuantWeight,
        x: &Q8Activations,
        shape: Vec<usize>,
    ) -> Result<Tensor> {
        let Some(launch) = launcher(weight.dtype) else {
            candle_core::bail!("no MMVQ launcher for {:?}", weight.dtype)
        };
        let device = x.device.as_cuda_device()?;
        let stream = device.cuda_stream();
        // SAFETY: the launcher writes all `rows * weight.rows` outputs.
        let out = unsafe { device.alloc::<f32>(x.rows * weight.rows)? };
        let weight_ptr = weight.weight.device_ptr()? as *const c_void;
        let input_ptr = x.data.device_ptr(&stream).0 as *const c_void;
        let out_ptr = out.device_ptr(&stream).0 as *mut c_void;
        // SAFETY: the weight holds `rows * cols / block` whole blocks, the
        // Q8_1 rows are `padded(cols) / 32` blocks apart, and `out` holds
        // `x.rows` rows of `weight.rows` values. `b_size` is 1..=8, the only
        // range the launcher serves.
        unsafe {
            launch(
                weight_ptr,
                input_ptr,
                out_ptr,
                weight.cols as i32,
                weight.rows as i32,
                (padded(weight.cols) / Q8_1_VALUES) as i32,
                weight.rows as i32,
                x.rows as i32,
                stream.cu_stream() as *mut c_void,
            );
        }
        check_launch("K-quant MMVQ")?;
        Ok(Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, device.clone())),
            Shape::from(shape),
            BackpropOp::none(),
            false,
        ))
    }

    type TileQuantizeLauncher = unsafe extern "C" fn(
        x: *const c_void,
        ids: *const i32,
        vy: *mut c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        s02: i64,
        s03: i64,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        stream: *mut c_void,
    );

    type MmqLauncher = unsafe extern "C" fn(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );

    fn mmq_launcher(dtype: GgmlDType) -> Option<MmqLauncher> {
        use candle_kernels::ffi;
        Some(match dtype {
            GgmlDType::Q4K => ffi::launch_mmq_gguf_q4_k,
            GgmlDType::Q5K => ffi::launch_mmq_gguf_q5_k,
            GgmlDType::Q6K => ffi::launch_mmq_gguf_q6_k,
            GgmlDType::Q8_0 => ffi::launch_mmq_gguf_q8_0,
            _ => return None,
        })
    }

    fn tile_quantizer(scales: TileScales) -> TileQuantizeLauncher {
        use candle_kernels::ffi;
        match scales {
            TileScales::D4 => ffi::launch_mmq_quantize_q8_1_D4,
            TileScales::Ds4 => ffi::launch_mmq_quantize_q8_1_DS4,
        }
    }

    /// The device attributes the MMQ launchers take, as `QMatMul` reads them.
    #[derive(Clone, Copy)]
    struct DeviceInfo {
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
    }

    fn device_info(device: &candle_core::CudaDevice) -> DeviceInfo {
        use candle_core::cuda_backend::cudarc::driver::{result, sys};
        use std::collections::HashMap;
        use std::sync::{Mutex, OnceLock};

        static INFO: OnceLock<Mutex<HashMap<i32, DeviceInfo>>> = OnceLock::new();
        let cu_device = device.cuda_stream().context().cu_device();
        let mut cache = INFO
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *cache.entry(cu_device).or_insert_with(|| {
            let attribute = |attribute, fallback| {
                // SAFETY: a read-only attribute query of a valid device.
                unsafe { result::device::get_attribute(cu_device, attribute) }.unwrap_or(fallback)
            };
            use sys::CUdevice_attribute as A;
            let major = attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 8);
            let minor = attribute(A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0);
            DeviceInfo {
                cc: major * 100 + minor * 10,
                nsm: attribute(A::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 1),
                smpbo: attribute(
                    A::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                    49152,
                ) as i64,
                warp_size: attribute(A::CU_DEVICE_ATTRIBUTE_WARP_SIZE, 32),
            }
        })
    }

    pub(super) fn quantize_tiles(
        x: &Tensor,
        rows: usize,
        cols: usize,
        scales: TileScales,
    ) -> Result<CudaSlice<u8>> {
        let device = x.device().as_cuda_device()?;
        let stream = device.cuda_stream();
        let cols_padded = padded(cols);
        // SAFETY: the quantizer writes every tile of the `rows` inputs; the
        // trailing padding is read only for columns no output keeps, as in
        // `QMatMul`'s workspace.
        let data = unsafe { device.alloc::<u8>(tile_bytes(rows, cols))? };
        let (storage, layout) = x.storage_and_layout();
        let Storage::Cuda(cuda) = &*storage else {
            candle_core::bail!("K-quant MMQ input must be a CUDA tensor")
        };
        let Some((start, end)) = layout.contiguous_offsets() else {
            candle_core::bail!("K-quant MMQ input must be contiguous")
        };
        let input = cuda.as_cuda_slice::<f32>()?.slice(start..end);
        let input_ptr = input.device_ptr(&stream).0 as *const c_void;
        let data_ptr = data.device_ptr(&stream).0 as *mut c_void;
        // SAFETY: `input` holds `rows * cols` F32 values and `data` the
        // padded tiles; the arguments are `QMatMul`'s for the same input.
        unsafe {
            tile_quantizer(scales)(
                input_ptr,
                std::ptr::null(),
                data_ptr,
                0,
                cols as i64,
                cols as i64,
                0,
                0,
                cols_padded as i64,
                rows as i64,
                1,
                1,
                stream.cu_stream() as *mut c_void,
            );
        }
        check_launch("Q8_1 tile quantize")?;
        Ok(data)
    }

    pub(super) fn mat_mul(weight: &KQuantWeight, x: &Q8Tiles, shape: Vec<usize>) -> Result<Tensor> {
        let Some(launch) = mmq_launcher(weight.dtype) else {
            candle_core::bail!("no MMQ launcher for {:?}", weight.dtype)
        };
        let device = x.device.as_cuda_device()?;
        let stream = device.cuda_stream();
        let info = device_info(device);
        // SAFETY: the launcher writes all `rows * weight.rows` outputs.
        let out = unsafe { device.alloc::<f32>(x.rows * weight.rows)? };
        // SAFETY: the stream-k pass writes a partial tile per SM before its
        // fixup pass reads it; nothing reads it uninitialized.
        let fixup = unsafe { device.alloc::<f32>(info.nsm.max(1) as usize * FIXUP_TILE_VALUES)? };
        let weight_ptr = weight.weight.device_ptr()? as *const c_void;
        let input_ptr = x.data.device_ptr(&stream).0 as *const c_void;
        let out_ptr = out.device_ptr(&stream).0 as *mut c_void;
        let fixup_ptr = fixup.device_ptr(&stream).0 as *mut c_void;
        // SAFETY: the weight holds `rows * cols / block` whole blocks
        // (`from_qmatmul`), the tiles hold `x.rows` padded inputs, `out`
        // `x.rows` rows of `weight.rows` values and `fixup` one 128 x 128
        // tile per SM. The arguments are `QMatMul`'s for the same product.
        unsafe {
            launch(
                fixup_ptr,
                weight_ptr,
                input_ptr,
                out_ptr,
                weight.cols as i64,
                weight.rows as i64,
                x.rows as i64,
                (weight.cols / weight.dtype.block_size()) as i64,
                weight.rows as i64,
                info.cc,
                info.nsm,
                info.smpbo,
                info.warp_size,
                stream.cu_stream() as *mut c_void,
            );
        }
        check_launch("K-quant MMQ")?;
        // Freed behind the launch on the same stream.
        drop(fixup);
        Ok(Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, device.clone())),
            Shape::from(shape),
            BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn rows_are_padded_to_whole_q8_1_rows() {
        assert_eq!(padded(1024), 1024);
        assert_eq!(padded(2560), 2560);
        assert_eq!(padded(3584), 3584);
        assert_eq!(padded(100), 512);
        assert_eq!(padded(513), 1024);
    }

    #[test]
    fn non_cuda_weights_and_inputs_are_refused() {
        let weights = Tensor::zeros((4, 256), candle_core::DType::F32, &Device::Cpu).unwrap();
        let quantized = QTensor::quantize(&weights, GgmlDType::Q4K).unwrap();
        assert!(KQuantWeight::from_qmatmul(&QMatMul::QTensor(Arc::new(quantized))).is_none());
        assert!(KQuantWeight::from_qmatmul(&QMatMul::Tensor(weights)).is_none());
        let x = Tensor::zeros((1, 256), candle_core::DType::F32, &Device::Cpu).unwrap();
        assert!(!accepts(&x));
        assert!(quantize(&x).is_err());
    }
}
