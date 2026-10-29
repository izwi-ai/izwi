//! Fused LFM2 ShortConv decode step: one kernel per row and layer.
//!
//! From the row's in-projection output `[b | c | x]` (each `hidden` wide) and
//! its ring of the last `capacity` gated inputs, one pass computes
//!
//! ```text
//! bx   = b * x
//! conv = sum_tap ring_or_bx(cursor + 1 - capacity + tap) * weight[h, tap]
//! y    = c * conv
//! ```
//!
//! and writes the next ring (the old slots with `bx` in slot
//! `cursor % capacity`) beside `y` in one `[capacity + 1, hidden]` buffer:
//! rows `0..capacity` are the next ring and row `capacity` is `y`. It
//! replaces the gate multiply, the ring kernel, the three-slot ring rebuild,
//! the output gate and two layout copies of the op-by-op path (about eight
//! dispatches per layer and token).
//!
//! The next ring is a fresh buffer, never the input ring, so a checkpointed
//! ring stays valid for rollback. Ring slots older than `cursor - valid` are
//! never read, matching the ring-sequence kernels.

use candle_core::{
    bail, CpuStorage, CustomOp3, Layout, Result as CandleResult, Shape, Tensor,
};

use crate::error::{Error, Result};

/// `(y [1, 1, hidden], next_ring [capacity, 1, hidden])` for one row.
pub(crate) fn lfm_shortconv_gated_ring_step(
    projected: &Tensor,
    ring: &Tensor,
    weight: &Tensor,
    cursor: u64,
    valid: u64,
) -> Result<(Tensor, Tensor)> {
    let (capacity, ring_rows, hidden) = ring.dims3()?;
    if ring_rows != 1
        || projected.elem_count() != 3 * hidden
        || weight.dims2()? != (hidden, capacity)
        || capacity == 0
        || valid > capacity as u64
        || valid > cursor
    {
        return Err(Error::InvalidInput(
            "LFM2 fused ShortConv step geometry mismatch".into(),
        ));
    }
    let projected = projected.contiguous()?;
    let ring = ring.contiguous()?;
    let weight = weight.contiguous()?;
    let packed = projected.apply_op3_no_bwd(
        &ring,
        &weight,
        &GatedRingStepOp {
            hidden,
            capacity,
            cursor,
            valid,
        },
    )?;
    let y = packed.narrow(0, capacity, 1)?.reshape((1, 1, hidden))?;
    let next_ring = packed
        .narrow(0, 0, capacity)?
        .reshape((capacity, 1, hidden))?;
    Ok((y, next_ring))
}

struct GatedRingStepOp {
    hidden: usize,
    capacity: usize,
    cursor: u64,
    valid: u64,
}

impl GatedRingStepOp {
    fn output_shape(&self) -> Shape {
        Shape::from((self.capacity + 1, self.hidden))
    }

    fn check(&self, l1: &Layout, l2: &Layout, l3: &Layout) -> CandleResult<()> {
        if !l1.is_contiguous() || !l2.is_contiguous() || !l3.is_contiguous() {
            bail!("lfm-shortconv-gated-ring-step requires contiguous tensors")
        }
        if l1.shape().elem_count() != 3 * self.hidden
            || l2.shape().elem_count() != self.capacity * self.hidden
            || l3.shape().elem_count() != self.capacity * self.hidden
        {
            bail!("lfm-shortconv-gated-ring-step shape mismatch")
        }
        if self.hidden > u32::MAX as usize || self.capacity > u32::MAX as usize {
            bail!("lfm-shortconv-gated-ring-step tensor is too large")
        }
        Ok(())
    }
}

/// The exact reference every backend matches.
fn gated_ring_step_reference(
    projected: &[f32],
    ring: &[f32],
    weight: &[f32],
    hidden: usize,
    capacity: usize,
    cursor: u64,
    valid: u64,
    output: &mut [f32],
) {
    let slot = (cursor % capacity as u64) as usize;
    let oldest = cursor - valid;
    for h in 0..hidden {
        let bx = projected[h] * projected[2 * hidden + h];
        let mut conv = bx * weight[h * capacity + capacity - 1];
        for tap in 0..capacity - 1 {
            let source = cursor as i128 + 1 - capacity as i128 + tap as i128;
            if source < 0 || (source as u64) < oldest {
                continue;
            }
            let physical = (source as u64 % capacity as u64) as usize;
            conv += ring[physical * hidden + h] * weight[h * capacity + tap];
        }
        for s in 0..capacity {
            output[s * hidden + h] = if s == slot { bx } else { ring[s * hidden + h] };
        }
        output[capacity * hidden + h] = projected[hidden + h] * conv;
    }
}

impl CustomOp3 for GatedRingStepOp {
    fn name(&self) -> &'static str {
        "lfm-shortconv-gated-ring-step"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> CandleResult<(CpuStorage, Shape)> {
        self.check(l1, l2, l3)?;
        // The inputs are read in place (`check` holds them contiguous).
        fn slice<'a>(storage: &'a CpuStorage, layout: &Layout) -> CandleResult<&'a [f32]> {
            let CpuStorage::F32(values) = storage else {
                bail!("lfm-shortconv-gated-ring-step only supports F32 tensors")
            };
            let start = layout.start_offset();
            Ok(&values[start..start + layout.shape().elem_count()])
        }
        let (projected, ring, weight) = (slice(s1, l1)?, slice(s2, l2)?, slice(s3, l3)?);
        let mut output = vec![0f32; (self.capacity + 1) * self.hidden];
        gated_ring_step_reference(
            projected,
            ring,
            weight,
            self.hidden,
            self.capacity,
            self.cursor,
            self.valid,
            &mut output,
        );
        Ok((CpuStorage::F32(output), self.output_shape()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
        s3: &candle_core::MetalStorage,
        l3: &Layout,
    ) -> CandleResult<(candle_core::MetalStorage, Shape)> {
        use super::metal_encoder::IzwiMetalCommandEncoderExt;
        use candle_core::backend::BackendStorage;
        use candle_core::DType;
        self.check(l1, l2, l3)?;
        if [s1.dtype(), s2.dtype(), s3.dtype()]
            .iter()
            .any(|dtype| *dtype != DType::F32)
        {
            bail!("lfm-shortconv-gated-ring-step only supports F32 tensors")
        }
        let device = s1.device().clone();
        let elements = (self.capacity + 1) * self.hidden;
        let output = device.new_buffer(elements, DType::F32, "lfm-shortconv-gated-ring-step")?;
        let pipeline = metal::pipeline(device.metal_device())?;
        let encoder = device.command_encoder()?;
        encoder.set_label("lfm-shortconv-gated-ring-step");
        encoder.set_compute_pipeline_state(&pipeline);
        let bytes = DType::F32.size_in_bytes();
        encoder.set_input_buffer(0, Some(s1.buffer()), l1.start_offset() * bytes);
        encoder.set_input_buffer(1, Some(s2.buffer()), l2.start_offset() * bytes);
        encoder.set_input_buffer(2, Some(s3.buffer()), l3.start_offset() * bytes);
        encoder.set_output_buffer(3, Some(&output), 0);
        encoder.set_bytes(4, &(self.hidden as u32));
        encoder.set_bytes(5, &(self.capacity as u32));
        encoder.set_bytes(6, &self.cursor);
        encoder.set_bytes(7, &self.valid);
        let width = pipeline.max_total_threads_per_threadgroup().clamp(1, 256);
        encoder.dispatch_threads(
            objc2_metal::MTLSize {
                width: self.hidden,
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        drop(encoder);
        Ok((
            candle_core::MetalStorage::new(output, device, elements, DType::F32),
            self.output_shape(),
        ))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> CandleResult<(candle_core::CudaStorage, Shape)> {
        self.check(l1, l2, l3)?;
        let output = super::cuda::launch_lfm_shortconv_gated_ring_step(
            s1,
            l1,
            s2,
            l2,
            s3,
            l3,
            self.hidden,
            self.capacity,
            self.cursor,
            self.valid,
        )?;
        Ok((output, self.output_shape()))
    }
}

#[cfg(feature = "metal")]
mod metal {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    use candle_core::Result as CandleResult;
    use candle_metal_kernels::metal::{ComputePipeline, Device as MetalDevice};

    pub(super) fn pipeline(device: &MetalDevice) -> CandleResult<ComputePipeline> {
        static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
        let key = device.registry_id();
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
            .get_function("izwi_lfm_shortconv_gated_ring_step_f32", None)
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

kernel void izwi_lfm_shortconv_gated_ring_step_f32(
    device const float* projected [[buffer(0)]],
    device const float* ring [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant uint& capacity [[buffer(5)]],
    constant ulong& cursor [[buffer(6)]],
    constant ulong& valid [[buffer(7)]],
    uint h [[thread_position_in_grid]]
) {
    if (h >= hidden) {
        return;
    }
    const float bx = projected[h] * projected[2 * hidden + h];
    const ulong slot = cursor % ulong(capacity);
    const ulong oldest = cursor - valid;
    float conv = bx * weight[h * capacity + capacity - 1];
    for (uint tap = 0; tap + 1 < capacity; ++tap) {
        const long source = long(cursor) + 1 - long(capacity) + long(tap);
        if (source < 0 || ulong(source) < oldest) {
            continue;
        }
        conv += ring[(ulong(source) % ulong(capacity)) * hidden + h] *
            weight[h * capacity + tap];
    }
    for (uint s = 0; s < capacity; ++s) {
        output[s * hidden + h] = (ulong(s) == slot) ? bx : ring[s * hidden + h];
    }
    output[capacity * hidden + h] = projected[hidden + h] * conv;
}
"#;
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// Deterministic unstructured values (xorshift), per tasks/lessons.md.
    fn xorshift(seed: u32, count: usize) -> Vec<f32> {
        let mut state = seed.max(1);
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// The op-by-op definition: gate, depthwise causal conv over the ring
    /// window and the current input, output gate; then the ring update.
    fn op_by_op(
        projected: &[f32],
        ring: &[f32],
        weight: &[f32],
        hidden: usize,
        capacity: usize,
        cursor: u64,
        valid: u64,
    ) -> (Vec<f32>, Vec<f32>) {
        let (b, rest) = projected.split_at(hidden);
        let (c, x) = rest.split_at(hidden);
        let bx: Vec<f32> = b.iter().zip(x).map(|(b, x)| b * x).collect();
        let mut y = vec![0f32; hidden];
        for h in 0..hidden {
            let mut conv = 0f32;
            for tap in 0..capacity {
                let source = cursor as i64 + 1 - capacity as i64 + tap as i64;
                let value = if source == cursor as i64 {
                    bx[h]
                } else if source < 0 || (source as u64) < cursor - valid {
                    0.0
                } else {
                    ring[(source as usize % capacity) * hidden + h]
                };
                conv += value * weight[h * capacity + tap];
            }
            y[h] = c[h] * conv;
        }
        let mut next = ring.to_vec();
        let slot = (cursor % capacity as u64) as usize;
        next[slot * hidden..(slot + 1) * hidden].copy_from_slice(&bx);
        (y, next)
    }

    pub(crate) fn assert_matches_op_by_op(device: &Device, tolerance: f32) {
        let hidden = 37;
        let capacity = 3;
        for (cursor, valid) in [(0u64, 0u64), (1, 1), (2, 2), (3, 3), (7, 3), (8, 2)] {
            let projected = xorshift(11 + cursor as u32, 3 * hidden);
            let ring = xorshift(97 + cursor as u32, capacity * hidden);
            let weight = xorshift(5, hidden * capacity);
            let (want_y, want_ring) =
                op_by_op(&projected, &ring, &weight, hidden, capacity, cursor, valid);
            let (y, next) = lfm_shortconv_gated_ring_step(
                &Tensor::from_vec(projected, (1, 1, 3 * hidden), device).unwrap(),
                &Tensor::from_vec(ring, (capacity, 1, hidden), device).unwrap(),
                &Tensor::from_vec(weight, (hidden, capacity), device).unwrap(),
                cursor,
                valid,
            )
            .unwrap();
            assert_eq!(y.dims(), [1, 1, hidden]);
            assert_eq!(next.dims(), [capacity, 1, hidden]);
            let y = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let next = next.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (got, want) in y.iter().zip(&want_y).chain(next.iter().zip(&want_ring)) {
                // `<=` is false for NaN, so a non-finite output fails.
                assert!(
                    (got - want).abs() <= tolerance,
                    "cursor {cursor} valid {valid}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn fused_step_matches_op_by_op_on_cpu() {
        assert_matches_op_by_op(&Device::Cpu, 1e-6);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn fused_step_matches_op_by_op_on_metal() {
        if let Some((device, _lease)) =
            crate::backends::metal_test_device_or_skip("fused_step_matches_op_by_op_on_metal")
        {
            assert_matches_op_by_op(&device, 1e-5);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an NVIDIA GPU; run explicitly with --ignored"]
    fn fused_step_matches_op_by_op_on_cuda() {
        assert_matches_op_by_op(&Device::new_cuda(0).expect("CUDA device required"), 1e-5);
    }

    #[test]
    fn rejects_inconsistent_cursor_and_geometry() {
        let device = Device::Cpu;
        let projected = Tensor::zeros((1, 1, 12), DType::F32, &device).unwrap();
        let ring = Tensor::zeros((3, 1, 4), DType::F32, &device).unwrap();
        let weight = Tensor::zeros((4, 3), DType::F32, &device).unwrap();
        assert!(lfm_shortconv_gated_ring_step(&projected, &ring, &weight, 1, 2).is_err());
        assert!(lfm_shortconv_gated_ring_step(&projected, &ring, &weight, 9, 4).is_err());
        let wrong = Tensor::zeros((4, 2), DType::F32, &device).unwrap();
        assert!(lfm_shortconv_gated_ring_step(&projected, &ring, &wrong, 3, 3).is_err());
    }
}
