//! SwiGLU over a packed `[gate | up]` projection, on CPU, Metal and CUDA:
//! `out[r, i] = O(silu(gate[r, i]) * up[r, i])` with the arithmetic in F32
//! and `silu(v) = v / (1 + exp(-v))`, as the eager chain computes it.
//!
//! A gated MLP that packs its gate and up weights into one projection runs
//! one GEMV and this one launch, where the eager form ran two GEMVs, two
//! casts, SiLU, a product and a cast back.
use candle_core::{DType, Device, Result, Tensor};

fn tag(dtype: DType) -> Result<&'static str> {
    Ok(match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        other => candle_core::bail!("SwiGLU does not take {other:?}"),
    })
}

/// The kernels' operands: `packed`'s rows and width, and the dtype mix's
/// name.
#[cfg(any(feature = "metal", feature = "cuda"))]
fn launch(packed: &Tensor, out: DType) -> Result<(usize, usize, String)> {
    let (rows, columns) = packed.dims2()?;
    let name = format!("{}_{}", tag(packed.dtype())?, tag(out)?);
    Ok((rows, columns / 2, name))
}

/// SwiGLU of `packed` `[rows, 2 * width]` (gate columns first) into `out`.
pub fn swiglu(packed: &Tensor, out: DType) -> Result<Tensor> {
    let (_, columns) = packed.dims2()?;
    if columns == 0 || !columns.is_multiple_of(2) {
        candle_core::bail!("SwiGLU takes [rows, 2 * width], got {:?}", packed.dims())
    }
    tag(packed.dtype())?;
    tag(out)?;
    let packed = packed.contiguous()?;
    match packed.device() {
        #[cfg(feature = "metal")]
        Device::Metal(_) => {
            if let Some(y) = metal_impl::run(&packed, out)? {
                return Ok(y);
            }
            reference(&packed, out)
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::run(&packed, out),
        Device::Cpu => reference(&packed, out),
        #[allow(unreachable_patterns)]
        _ => reference(&packed, out),
    }
}

/// The eager chain: both halves to F32, SiLU of the gate, the product.
pub fn reference(packed: &Tensor, out: DType) -> Result<Tensor> {
    let width = packed.dim(1)? / 2;
    let gate = packed.narrow(1, 0, width)?.to_dtype(DType::F32)?;
    let up = packed.narrow(1, width, width)?.to_dtype(DType::F32)?;
    (candle_nn::ops::silu(&gate)? * up)?.to_dtype(out)
}

#[cfg(feature = "metal")]
mod metal_impl {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    use candle_core::backend::BackendStorage;
    use candle_core::op::BackpropOp;
    use candle_core::{DType, MetalStorage, Result, Shape, Storage, Tensor};
    use candle_metal_kernels::metal::{ComputePipeline, Device as MetalDevice};

    use crate::kernels::metal_encoder::IzwiMetalCommandEncoderExt;

    /// `None` for a dtype mix Metal has no form of (BF16).
    pub(super) fn run(packed: &Tensor, out: DType) -> Result<Option<Tensor>> {
        if packed.dtype() == DType::BF16 || out == DType::BF16 {
            return Ok(None);
        }
        let (rows, width, name) = super::launch(packed, out)?;
        let (storage, layout) = packed.storage_and_layout();
        let Storage::Metal(m) = &*storage else {
            candle_core::bail!("the Metal SwiGLU needs a Metal tensor")
        };
        let device = m.device().clone();
        let Some(pipeline) = pipeline(device.metal_device(), &format!("izwi_swiglu_{name}"))?
        else {
            return Ok(None);
        };
        let y = device.new_buffer(rows * width, out, "izwi-swiglu")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-swiglu");
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_input_buffer(
            0,
            Some(m.buffer()),
            layout.start_offset() * packed.dtype().size_in_bytes(),
        );
        encoder.set_output_buffer(1, Some(&y), 0);
        encoder.set_bytes(2, &(width as u32));
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: width.div_ceil(256),
                height: rows,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        Ok(Some(Tensor::from_storage(
            Storage::Metal(MetalStorage::new(y, device.clone(), rows * width, out)),
            Shape::from((rows, width)),
            BackpropOp::none(),
            false,
        )))
    }

    fn pipeline(device: &MetalDevice, function: &str) -> Result<Option<ComputePipeline>> {
        static PIPELINES: OnceLock<Mutex<HashMap<(u64, String), ComputePipeline>>> =
            OnceLock::new();
        let key = (device.registry_id(), function.to_string());
        let pipelines = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(pipeline) = pipelines
            .lock()
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?
            .get(&key)
            .cloned()
        {
            return Ok(Some(pipeline));
        }
        let library = device
            .new_library_with_source(SOURCE, None)
            .map_err(candle_core::Error::wrap)?;
        let Ok(function) = library.get_function(function, None) else {
            return Ok(None);
        };
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(candle_core::Error::wrap)?;
        pipelines
            .lock()
            .map_err(|err| candle_core::Error::Msg(err.to_string()))?
            .insert(key, pipeline.clone());
        Ok(Some(pipeline))
    }

    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// out[r, i] = O(silu(gate) * up) over packed [gate | up] rows; grid
// (ceil(width / 256), rows).
#define IZWI_SWIGLU(NAME, I, O)                                                   \
kernel void izwi_swiglu_##NAME(                                                   \
    device const I* packed [[buffer(0)]],                                         \
    device O* out [[buffer(1)]],                                                  \
    constant uint& width [[buffer(2)]],                                           \
    uint2 local [[thread_position_in_threadgroup]],                               \
    uint2 group [[threadgroup_position_in_grid]]                                  \
) {                                                                               \
    const uint i = group.x * 256 + local.x;                                       \
    if (i >= width) { return; }                                                   \
    const ulong row = group.y;                                                    \
    const float g = float(packed[row * 2 * width + i]);                           \
    const float u = float(packed[row * 2 * width + width + i]);                   \
    out[row * width + i] = O(g / (1.0f + precise::exp(-g)) * u);                  \
}

IZWI_SWIGLU(f32_f32, float, float)
IZWI_SWIGLU(f16_f16, half, half)
IZWI_SWIGLU(f32_f16, float, half)
"#;
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaStorage, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{DType, Result, Shape, Storage, Tensor};

    pub(super) fn run(packed: &Tensor, out: DType) -> Result<Tensor> {
        let device = packed.device().as_cuda_device()?;
        let (rows, width, name) = super::launch(packed, out)?;
        let exported = ["f32_f32", "f16_f16", "bf16_bf16", "f32_f16", "f32_bf16"];
        if !exported.contains(&name.as_str()) {
            candle_core::bail!("the CUDA SwiGLU has no {name} form")
        }
        let function = device.get_or_load_custom_func(
            &format!("izwi_swiglu_{name}"),
            "izwi_swiglu",
            super::super::cuda::cuda_ptx_swiglu(),
        )?;
        let (storage, layout) = packed.storage_and_layout();
        let Storage::Cuda(cuda) = &*storage else {
            candle_core::bail!("the CUDA SwiGLU needs a CUDA tensor")
        };
        let start = layout.start_offset();
        let count = rows * 2 * width;
        let config = LaunchConfig {
            grid_dim: (width.div_ceil(256) as u32, rows as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        macro_rules! go {
            ($i:ty, $o:ty) => {{
                let x = cuda.as_cuda_slice::<$i>()?.slice(start..start + count);
                // SAFETY: the kernel writes every output element.
                let y = unsafe { device.alloc::<$o>(rows * width)? };
                let mut builder = function.builder();
                builder.arg(&x);
                builder.arg(&y);
                candle_core::builder_arg!(builder, width as i32);
                // SAFETY: argument order and types match the kernel.
                unsafe { builder.launch(config) }.w()?;
                Tensor::from_storage(
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(y, device.clone())),
                    Shape::from((rows, width)),
                    BackpropOp::none(),
                    false,
                )
            }};
        }
        use half::{bf16, f16};
        Ok(match (packed.dtype(), out) {
            (DType::F32, DType::F32) => go!(f32, f32),
            (DType::F16, DType::F16) => go!(f16, f16),
            (DType::BF16, DType::BF16) => go!(bf16, bf16),
            (DType::F32, DType::F16) => go!(f32, f16),
            _ => go!(f32, bf16),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::numeric_check::xorshift_values;

    fn packed(rows: usize, width: usize, dtype: DType, device: &Device) -> Tensor {
        Tensor::from_vec(
            xorshift_values(rows * 2 * width, 41, 6.0),
            (rows, 2 * width),
            &Device::Cpu,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
        .to_device(device)
        .unwrap()
    }

    fn host(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    #[allow(dead_code)]
    fn matches_the_eager_chain(device: &Device, mixes: &[(DType, DType)]) {
        for &(input, out) in mixes {
            let x = packed(3, 6144, input, device);
            let got = swiglu(&x, out).unwrap();
            assert_eq!(got.dims(), [3, 6144]);
            assert_eq!(got.dtype(), out);
            let want = reference(&packed(3, 6144, input, &Device::Cpu), out).unwrap();
            let tolerance = if out == DType::F32 { 1e-5 } else { 1e-2 };
            for (a, e) in host(&got).iter().zip(&host(&want)) {
                assert!(
                    (a - e).abs() <= tolerance * e.abs().max(1.0),
                    "{input:?}->{out:?}: {a} {e}"
                );
            }
        }
    }

    #[test]
    fn the_cpu_form_is_the_eager_chain() {
        matches_the_eager_chain(
            &Device::Cpu,
            &[(DType::F32, DType::F32), (DType::BF16, DType::BF16)],
        );
        assert!(swiglu(
            &packed(1, 3, DType::F32, &Device::Cpu)
                .narrow(1, 0, 5)
                .unwrap(),
            DType::F32
        )
        .is_err());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_matches_the_eager_chain() {
        let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("swiglu_metal_matches_the_eager_chain")
        else {
            return;
        };
        matches_the_eager_chain(
            &device,
            &[
                (DType::F32, DType::F32),
                (DType::F16, DType::F16),
                (DType::F32, DType::F16),
            ],
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_matches_the_eager_chain() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        matches_the_eager_chain(
            &device,
            &[
                (DType::F32, DType::F32),
                (DType::BF16, DType::BF16),
                (DType::F16, DType::F16),
            ],
        );
    }
}
