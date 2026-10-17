//! Block-FP8 weight expansion to F16 for Metal prefill.
//!
//! Metal keeps a model's block-FP8 projections resident as the checkpoint's
//! E4M3FN bytes plus one F32 scale per 128x128 block. Decode rows run the
//! compact FP8 GEMV (`kernels::cuda::fp8::block_fp8_projection`); a prefill
//! span is faster as one F16 matrix multiplication, so this kernel expands the
//! raw weights into a transient F16 matrix first: `half(decode(b) * scale)`,
//! the product taken in F32 before the single rounding to F16, exactly as
//! the expanded F16 load computes each weight. Generic: geometry comes from
//! the tensors, and no model family is assumed. A tiled simdgroup FP8 GEMM can
//! replace this later.
use candle_core::{CpuStorage, CustomOp2, DType, Layout, Result, Shape, Tensor};

/// `[n, k]` F16 weights from `weights` (`[n, k]` E4M3FN bytes) and `scales`
/// (`[ceil(n/128), ceil(k/128)]` F32). The CPU reference runs on host tensors;
/// Metal tensors run the MSL kernel.
pub fn dequantize_f16(weights: &Tensor, scales: &Tensor) -> Result<Tensor> {
    let (n, k) = weights.dims2()?;
    if weights.dtype() != DType::U8
        || scales.dtype() != DType::F32
        || scales.dims() != [n.div_ceil(128), k.div_ceil(128)]
        || !weights.device().same_device(scales.device())
    {
        candle_core::bail!("invalid block-FP8 dequantization contract")
    }
    weights
        .contiguous()?
        .apply_op2_no_bwd(&scales.contiguous()?, &DequantF16 { n, k })
}

struct DequantF16 {
    n: usize,
    k: usize,
}

impl CustomOp2 for DequantF16 {
    fn name(&self) -> &'static str {
        "izwi-fp8-dequant-f16"
    }

    fn cpu_fwd(
        &self,
        w: &CpuStorage,
        wl: &Layout,
        s: &CpuStorage,
        sl: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::U8(w), CpuStorage::F32(s)) = (w, s) else {
            candle_core::bail!("block-FP8 dequantization needs U8 weights and F32 scales")
        };
        let (w, s) = (&w[wl.start_offset()..], &s[sl.start_offset()..]);
        let column_blocks = self.k.div_ceil(128);
        let out = (0..self.n * self.k)
            .map(|i| {
                let (row, column) = (i / self.k, i % self.k);
                let scale = s[(row / 128) * column_blocks + column / 128];
                half::f16::from_f32(super::cuda::fp8::decode_e4m3fn(w[i]) * scale)
            })
            .collect();
        Ok((CpuStorage::F16(out), Shape::from((self.n, self.k))))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        w: &candle_core::MetalStorage,
        wl: &Layout,
        s: &candle_core::MetalStorage,
        sl: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        metal::dequantize(w, wl, s, sl, self.n, self.k)
    }
}

#[cfg(feature = "metal")]
mod metal {
    use super::super::metal_encoder::IzwiMetalCommandEncoderExt;
    use candle_core::backend::BackendStorage;
    use candle_core::{DType, Layout, MetalStorage, Result, Shape};
    use candle_metal_kernels::metal::{ComputePipeline, Device as MetalDevice};
    use objc2_metal::MTLSize;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// E4M3FN byte -> value * 2^-8 through an F16 lane (exact for finite bytes).
inline float izwi_e4m3_scaled(uint b) {
    ushort h = ushort(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
    return float(as_type<half>(h));
}

// out[n, k] = half(decode(w[n, k]) * s[n / 128, k / 128]); four bytes per
// thread (K is a multiple of 4).
kernel void izwi_fp8_dequant_f16(
    device const uchar4* w [[buffer(0)]], device const float* s [[buffer(1)]],
    device half4* out [[buffer(2)]], constant uint& N [[buffer(3)]],
    constant uint& K [[buffer(4)]], uint2 gid [[thread_position_in_grid]]) {
    const uint quad = gid.x;
    const uint n = gid.y;
    const uint quads = K >> 2;
    if (quad >= quads || n >= N) {
        return;
    }
    const float scale = s[(n >> 7) * ((K + 127) >> 7) + ((quad << 2) >> 7)] * 256.0f;
    const uchar4 b = w[(ulong)n * quads + quad];
    out[(ulong)n * quads + quad] = half4(
        half(izwi_e4m3_scaled(b.x) * scale), half(izwi_e4m3_scaled(b.y) * scale),
        half(izwi_e4m3_scaled(b.z) * scale), half(izwi_e4m3_scaled(b.w) * scale));
}
"#;

    fn pipeline(device: &MetalDevice, name: &'static str) -> Result<ComputePipeline> {
        static PIPELINES: OnceLock<Mutex<HashMap<(u64, &'static str), ComputePipeline>>> =
            OnceLock::new();
        let key = (device.registry_id(), name);
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
            .get_function(name, None)
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

    pub(super) fn dequantize(
        w: &MetalStorage,
        wl: &Layout,
        s: &MetalStorage,
        sl: &Layout,
        n: usize,
        k: usize,
    ) -> Result<(MetalStorage, Shape)> {
        if !k.is_multiple_of(4) || !wl.start_offset().is_multiple_of(4) {
            candle_core::bail!("Metal block-FP8 dequantization needs 4-byte aligned rows")
        }
        let u32_arg = |value: usize, name: &str| {
            u32::try_from(value)
                .map_err(|_| candle_core::Error::Msg(format!("{name} {value} exceeds u32")))
        };
        let device = w.device().clone();
        let out = device.new_buffer(n * k, DType::F16, "izwi-fp8-dequant-f16")?;
        let encoder = device.command_encoder()?;
        encoder.set_label("izwi-fp8-dequant-f16");
        let pipeline = pipeline(device.metal_device(), "izwi_fp8_dequant_f16")?;
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_input_buffer(0, Some(w.buffer()), wl.start_offset());
        encoder.set_input_buffer(1, Some(s.buffer()), sl.start_offset() * 4);
        encoder.set_output_buffer(2, Some(&out), 0);
        encoder.set_bytes(3, &u32_arg(n, "rows")?);
        encoder.set_bytes(4, &u32_arg(k, "columns")?);
        let quads = k / 4;
        encoder.dispatch_threads(
            MTLSize {
                width: quads,
                height: n,
                depth: 1,
            },
            MTLSize {
                width: 64.min(quads),
                height: 4,
                depth: 1,
            },
        );
        drop(encoder);
        Ok((
            MetalStorage::new(out, device, n * k, DType::F16),
            Shape::from((n, k)),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn raw(n: usize, k: usize) -> (Tensor, Tensor) {
        let bytes: Vec<u8> = (0..n * k)
            .map(|i| match (i.wrapping_mul(0x9E37_79B1) >> 13) as u8 {
                b if b & 0x7f == 0x7f => b ^ 0x10,
                b => b,
            })
            .collect();
        let scales: Vec<f32> = (0..n.div_ceil(128) * k.div_ceil(128))
            .map(|i| 1e-3 * (1.0 + (i % 13) as f32 * 0.29))
            .collect();
        (
            Tensor::from_vec(bytes, (n, k), &Device::Cpu).unwrap(),
            Tensor::from_vec(scales, (n.div_ceil(128), k.div_ceil(128)), &Device::Cpu).unwrap(),
        )
    }

    #[test]
    fn cpu_reference_applies_each_block_scale_before_one_f16_rounding() {
        let (w, s) = raw(192, 384);
        let dense = dequantize_f16(&w, &s).unwrap();
        assert_eq!(dense.dtype(), DType::F16);
        assert_eq!(dense.dims(), [192, 384]);
        let bytes = w.flatten_all().unwrap().to_vec1::<u8>().unwrap();
        let scales = s.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let values = dense.flatten_all().unwrap().to_vec1::<half::f16>().unwrap();
        for (i, value) in values.iter().enumerate() {
            let scale = scales[(i / 384 / 128) * 3 + (i % 384) / 128];
            let expected =
                half::f16::from_f32(crate::kernels::cuda::fp8::decode_e4m3fn(bytes[i]) * scale);
            assert_eq!(value.to_bits(), expected.to_bits(), "element {i}");
        }
        assert!(dequantize_f16(&w, &s.narrow(1, 0, 2).unwrap()).is_err());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_dequantization_is_bit_exact_against_the_cpu_reference() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "metal_dequantization_is_bit_exact_against_the_cpu_reference",
        ) else {
            return;
        };
        for (n, k) in [(192usize, 384usize), (1280, 256), (256, 768)] {
            let (w, s) = raw(n, k);
            let expected = dequantize_f16(&w, &s)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<half::f16>()
                .unwrap();
            let actual = dequantize_f16(
                &w.to_device(&device).unwrap(),
                &s.to_device(&device).unwrap(),
            )
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<half::f16>()
            .unwrap();
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "[{n}, {k}]"
            );
        }
    }
}
