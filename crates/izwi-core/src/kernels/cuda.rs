//! CUDA kernel dispatch.
//!
//! This module wires CUDA-only fused-operation entry points to Candle CUDA
//! tensor kernels where Candle provides the primitive. These paths stay guarded
//! by `Device::is_cuda()` and fall back to the caller's existing implementation
//! when a shape, dtype, or build does not support the operation.

use candle_core::{CpuStorage, CustomOp3, DType, Layout, Shape, Tensor, D};

use crate::kernels::FusedSiluMulResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaKernelStatus {
    pub compiled: bool,
    pub available: bool,
    pub reason: &'static str,
}

pub fn cuda_kernels_compiled() -> bool {
    cfg!(feature = "cuda")
}

pub fn fused_kernels_available() -> bool {
    cuda_kernels_compiled()
}

pub fn use_block_fusion() -> bool {
    false
}

pub fn status() -> CudaKernelStatus {
    if !cuda_kernels_compiled() {
        return CudaKernelStatus {
            compiled: false,
            available: false,
            reason: "binary was not built with CUDA support",
        };
    }

    CudaKernelStatus {
        compiled: true,
        available: true,
        reason: "Candle CUDA kernel dispatch is enabled",
    }
}

pub fn try_fused_silu_mul(gate: &Tensor, up: &Tensor) -> Option<Tensor> {
    try_fused_silu_mul_with_status(gate, up).map(|result| result.tensor)
}

pub fn try_fused_silu_mul_with_status(gate: &Tensor, up: &Tensor) -> Option<FusedSiluMulResult> {
    if !cuda_tensor_pair_supported(gate, up) {
        return None;
    }

    let silu_gate = candle_nn::ops::silu(gate).ok()?;
    let tensor = silu_gate.broadcast_mul(up).ok()?;
    Some(FusedSiluMulResult {
        tensor,
        used_custom_kernel: false,
    })
}

pub fn try_fused_l2_norm(input: &Tensor, eps: f64) -> Option<Tensor> {
    if !cuda_tensor_supported(input) || input.dtype() != DType::F32 {
        return None;
    }

    input
        .broadcast_div(
            &(input.sqr().ok()?.sum_keepdim(D::Minus1).ok()? + eps)
                .ok()?
                .sqrt()
                .ok()?,
        )
        .ok()
}

pub fn try_fused_rms_norm(input: &Tensor, weight: &Tensor, eps: f64) -> Option<Tensor> {
    if !cuda_tensor_pair_supported(input, weight) {
        return None;
    }

    candle_nn::ops::rms_norm(input, weight, eps as f32).ok()
}

pub fn try_fused_gated_rms_norm(
    hidden: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> Option<Tensor> {
    if !cuda_tensor_pair_supported(hidden, gate) || !cuda_tensor_pair_supported(hidden, weight) {
        return None;
    }

    let normalized = try_fused_rms_norm(hidden, weight, eps)?;
    let silu_gate = candle_nn::ops::silu(gate).ok()?;
    normalized.broadcast_mul(&silu_gate).ok()
}

pub fn try_qwen35_causal_conv_sequence(
    input: &Tensor,
    weight: &Tensor,
    history: &Tensor,
) -> Option<(Tensor, Tensor)> {
    if !cuda_tensor_pair_supported(input, weight)
        || !cuda_tensor_pair_supported(input, history)
        || input.dtype() != DType::F32
    {
        return None;
    }
    let (batch, sequence, conv_dim) = input.dims3().ok()?;
    let (weight_channels, kernel_size) = weight.dims2().ok()?;
    let (history_channels, history_len) = history.dims2().ok()?;
    if batch != 1
        || sequence == 0
        || conv_dim == 0
        || kernel_size < 2
        || weight_channels != conv_dim
        || history_channels != conv_dim
        || history_len != kernel_size - 1
    {
        return None;
    }

    let input = input.contiguous().ok()?;
    let weight = weight.contiguous().ok()?;
    let history = history.contiguous().ok()?;
    let output_elements = sequence.checked_mul(conv_dim)?;
    let state_elements = history_len.checked_mul(conv_dim)?;
    let packed = input
        .apply_op3_no_bwd(
            &weight,
            &history,
            &CudaCausalConvSequenceOp {
                conv_dim,
                sequence,
                kernel_size,
            },
        )
        .ok()?;
    let output = packed
        .narrow(0, 0, output_elements)
        .ok()?
        .reshape((1, sequence, conv_dim))
        .ok()?;
    let final_history = packed
        .narrow(0, output_elements, state_elements)
        .ok()?
        .reshape((conv_dim, history_len))
        .ok()?
        .force_contiguous()
        .ok()?;
    Some((output, final_history))
}

pub fn try_fused_gated_delta_recurrent(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &Tensor,
) -> Option<(Tensor, Tensor)> {
    if !cuda_tensor_pair_supported(query, key)
        || !cuda_tensor_pair_supported(query, value)
        || !cuda_tensor_pair_supported(query, g)
        || !cuda_tensor_pair_supported(query, beta)
        || !cuda_tensor_pair_supported(query, state)
        || query.dtype() != DType::F32
    {
        return None;
    }

    let queries = query.unsqueeze(1).ok()?;
    let keys = key.unsqueeze(1).ok()?;
    let values = value.unsqueeze(1).ok()?;
    let gates = g.unsqueeze(1).ok()?;
    let betas = beta.unsqueeze(1).ok()?;
    let (outputs, next_state) =
        cuda_gated_delta_sequence(&queries, &keys, &values, &gates, &betas, state)?;
    Some((outputs.squeeze(1).ok()?, next_state))
}

pub fn try_tiled_deltanet_recurrence(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    initial_state: &Tensor,
    _tile_size: usize,
) -> Option<(Tensor, Tensor)> {
    if !cuda_tensor_pair_supported(queries, keys)
        || !cuda_tensor_pair_supported(queries, values)
        || !cuda_tensor_pair_supported(queries, g)
        || !cuda_tensor_pair_supported(queries, beta)
        || !cuda_tensor_pair_supported(queries, initial_state)
        || queries.dtype() != DType::F32
    {
        return None;
    }

    let (batch, seq_len, num_heads, head_k_dim) = queries.dims4().ok()?;
    let (k_batch, k_seq_len, k_num_heads, k_head_k_dim) = keys.dims4().ok()?;
    let (v_batch, v_seq_len, v_num_heads, v_head_dim) = values.dims4().ok()?;
    let (g_batch, g_seq_len, g_heads) = g.dims3().ok()?;
    let (b_batch, b_seq_len, b_heads) = beta.dims3().ok()?;
    let (s_batch, s_heads, s_head_k_dim, s_head_v_dim) = initial_state.dims4().ok()?;

    if batch != 1
        || k_batch != batch
        || v_batch != batch
        || g_batch != batch
        || b_batch != batch
        || s_batch != batch
    {
        return None;
    }
    if k_seq_len != seq_len || v_seq_len != seq_len || g_seq_len != seq_len || b_seq_len != seq_len
    {
        return None;
    }
    if k_num_heads != num_heads || v_num_heads != num_heads || g_heads != num_heads {
        return None;
    }
    if b_heads != num_heads || k_head_k_dim != head_k_dim || s_heads != num_heads {
        return None;
    }
    if s_head_k_dim != head_k_dim || s_head_v_dim != v_head_dim {
        return None;
    }

    cuda_gated_delta_sequence(queries, keys, values, g, beta, initial_state)
}

fn cuda_tensor_supported(tensor: &Tensor) -> bool {
    cuda_kernels_compiled() && tensor.device().is_cuda()
}

fn cuda_tensor_pair_supported(lhs: &Tensor, rhs: &Tensor) -> bool {
    cuda_tensor_supported(lhs) && rhs.device().is_cuda() && lhs.dtype() == rhs.dtype()
}

fn cuda_gated_delta_sequence(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    initial_state: &Tensor,
) -> Option<(Tensor, Tensor)> {
    let (batch, sequence, heads, key_dim) = queries.dims4().ok()?;
    let (key_batch, key_sequence, key_heads, key_width) = keys.dims4().ok()?;
    let (value_batch, value_sequence, value_heads, value_dim) = values.dims4().ok()?;
    let (g_batch, g_sequence, g_heads) = g.dims3().ok()?;
    let (beta_batch, beta_sequence, beta_heads) = beta.dims3().ok()?;
    let (state_batch, state_heads, state_key_dim, state_value_dim) = initial_state.dims4().ok()?;
    if sequence == 0 || heads == 0 || key_dim == 0 || value_dim == 0 {
        return None;
    }
    if (key_batch, key_sequence, key_heads, key_width) != (batch, sequence, heads, key_dim)
        || (value_batch, value_sequence, value_heads) != (batch, sequence, heads)
        || (g_batch, g_sequence, g_heads) != (batch, sequence, heads)
        || (beta_batch, beta_sequence, beta_heads) != (batch, sequence, heads)
        || (state_batch, state_heads, state_key_dim, state_value_dim)
            != (batch, heads, key_dim, value_dim)
    {
        return None;
    }

    let qkv = Tensor::cat(&[queries, keys, values], D::Minus1)
        .ok()?
        .contiguous()
        .ok()?;
    let gates = Tensor::cat(
        &[
            &g.unsqueeze(D::Minus1).ok()?,
            &beta.unsqueeze(D::Minus1).ok()?,
        ],
        D::Minus1,
    )
    .ok()?
    .contiguous()
    .ok()?;
    let initial_state = initial_state.contiguous().ok()?;
    let packed = qkv
        .apply_op3_no_bwd(
            &gates,
            &initial_state,
            &CudaGatedDeltaSequenceOp {
                batch,
                sequence,
                heads,
                key_dim,
                value_dim,
            },
        )
        .ok()?;

    let output_elements = batch * sequence * heads * value_dim;
    let state_elements = batch * heads * key_dim * value_dim;
    let outputs = packed
        .narrow(0, 0, output_elements)
        .ok()?
        .reshape((batch, sequence, heads, value_dim))
        .ok()?;
    let next_state = packed
        .narrow(0, output_elements, state_elements)
        .ok()?
        .reshape((batch, heads, key_dim, value_dim))
        .ok()?
        // The packed custom-op result also contains sequence outputs. Detach
        // the persistent state so it retains only its exact live allocation.
        .force_contiguous()
        .ok()?;
    Some((outputs, next_state))
}

struct CudaCausalConvSequenceOp {
    conv_dim: usize,
    sequence: usize,
    kernel_size: usize,
}

impl CustomOp3 for CudaCausalConvSequenceOp {
    fn name(&self) -> &'static str {
        "qwen35-causal-conv-sequence"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _input_layout: &Layout,
        _weight: &CpuStorage,
        _weight_layout: &Layout,
        _history: &CpuStorage,
        _history_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("Qwen3.5 CUDA causal convolution has no CPU implementation")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        input: &candle_core::CudaStorage,
        input_layout: &Layout,
        weight: &candle_core::CudaStorage,
        weight_layout: &Layout,
        history: &candle_core::CudaStorage,
        history_layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};

        fn contiguous_slice<'a>(
            storage: &'a CudaStorageSlice,
            layout: &Layout,
            name: &str,
        ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, f32>>
        {
            let CudaStorageSlice::F32(slice) = storage else {
                candle_core::bail!("{name} must use F32 storage")
            };
            let Some((start, end)) = layout.contiguous_offsets() else {
                candle_core::bail!("{name} must be contiguous")
            };
            Ok(slice.slice(start..end))
        }

        let input_slice = contiguous_slice(&input.slice, input_layout, "input")?;
        let weight_slice = contiguous_slice(&weight.slice, weight_layout, "weight")?;
        let history_slice = contiguous_slice(&history.slice, history_layout, "history")?;
        let output_elements = self.sequence.checked_mul(self.conv_dim).ok_or_else(|| {
            candle_core::Error::Msg("Qwen3.5 CUDA convolution output overflow".to_string())
        })?;
        let state_elements = (self.kernel_size - 1)
            .checked_mul(self.conv_dim)
            .ok_or_else(|| {
                candle_core::Error::Msg("Qwen3.5 CUDA convolution state overflow".to_string())
            })?;
        let total_elements = output_elements.checked_add(state_elements).ok_or_else(|| {
            candle_core::Error::Msg("Qwen3.5 CUDA convolution allocation overflow".to_string())
        })?;
        if total_elements > i32::MAX as usize {
            candle_core::bail!("Qwen3.5 CUDA convolution tensor is too large")
        }
        let device = input.device();
        // SAFETY: the custom kernel writes every element before the storage is observed.
        let output = unsafe { device.alloc::<f32>(total_elements)? };
        let function = device.get_or_load_custom_func(
            "qwen35_causal_conv_sequence_f32",
            "izwi_qwen35_causal_conv_sequence",
            qwen35_cuda_ptx::QWEN35,
        )?;
        let config = LaunchConfig::for_num_elems(total_elements as u32);
        let mut builder = function.builder();
        builder.arg(&input_slice);
        builder.arg(&weight_slice);
        builder.arg(&history_slice);
        builder.arg(&output);
        candle_core::builder_arg!(
            builder,
            self.conv_dim as i32,
            self.sequence as i32,
            self.kernel_size as i32,
            output_elements as i32,
            total_elements as i32
        );
        // SAFETY: argument types and element bounds match the CUDA kernel signature.
        unsafe { builder.launch(config) }.w()?;

        Ok((
            candle_core::CudaStorage {
                slice: CudaStorageSlice::F32(output),
                device: device.clone(),
            },
            Shape::from_dims(&[total_elements]),
        ))
    }
}

struct CudaGatedDeltaSequenceOp {
    batch: usize,
    sequence: usize,
    heads: usize,
    key_dim: usize,
    value_dim: usize,
}

impl CustomOp3 for CudaGatedDeltaSequenceOp {
    fn name(&self) -> &'static str {
        "qwen35-gated-delta-sequence"
    }

    fn cpu_fwd(
        &self,
        _qkv: &CpuStorage,
        _qkv_layout: &Layout,
        _gates: &CpuStorage,
        _gates_layout: &Layout,
        _state: &CpuStorage,
        _state_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("Qwen3.5 CUDA recurrence has no CPU implementation")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        qkv: &candle_core::CudaStorage,
        qkv_layout: &Layout,
        gates: &candle_core::CudaStorage,
        gates_layout: &Layout,
        state: &candle_core::CudaStorage,
        state_layout: &Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};

        fn contiguous_slice<'a>(
            storage: &'a CudaStorageSlice,
            layout: &Layout,
            name: &str,
        ) -> candle_core::Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, f32>>
        {
            let CudaStorageSlice::F32(slice) = storage else {
                candle_core::bail!("{name} must use F32 storage")
            };
            let Some((start, end)) = layout.contiguous_offsets() else {
                candle_core::bail!("{name} must be contiguous")
            };
            Ok(slice.slice(start..end))
        }

        let qkv_slice = contiguous_slice(&qkv.slice, qkv_layout, "qkv")?;
        let gates_slice = contiguous_slice(&gates.slice, gates_layout, "gates")?;
        let state_slice = contiguous_slice(&state.slice, state_layout, "initial_state")?;
        let device = qkv.device();
        let output_elements = self.batch * self.sequence * self.heads * self.value_dim;
        let state_elements = self.batch * self.heads * self.key_dim * self.value_dim;
        // SAFETY: the custom kernel writes every element before the storage is observed.
        let output = unsafe { device.alloc::<f32>(output_elements + state_elements)? };
        let function = device.get_or_load_custom_func(
            "qwen35_gated_delta_sequence_f32",
            "izwi_qwen35_gated_delta_sequence",
            qwen35_cuda_ptx::QWEN35,
        )?;
        let block_size = self.value_dim.next_power_of_two().clamp(32, 256) as u32;
        let config = LaunchConfig {
            grid_dim: ((self.batch * self.heads) as u32, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = function.builder();
        builder.arg(&qkv_slice);
        builder.arg(&gates_slice);
        builder.arg(&state_slice);
        builder.arg(&output);
        candle_core::builder_arg!(
            builder,
            self.batch as i32,
            self.sequence as i32,
            self.heads as i32,
            self.key_dim as i32,
            self.value_dim as i32
        );
        // SAFETY: argument types and launch dimensions match the CUDA kernel signature.
        unsafe { builder.launch(config) }.w()?;

        Ok((
            candle_core::CudaStorage {
                slice: CudaStorageSlice::F32(output),
                device: device.clone(),
            },
            Shape::from_dims(&[output_elements + state_elements]),
        ))
    }
}

#[cfg(feature = "cuda")]
mod qwen35_cuda_ptx {
    include!(concat!(env!("OUT_DIR"), "/qwen35_ptx.rs"));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_kernel_status_is_explicit() {
        let status = status();
        assert_eq!(status.compiled, cfg!(feature = "cuda"));
        assert_eq!(status.available, cfg!(feature = "cuda"));
        assert!(!status.reason.trim().is_empty());
    }

    #[test]
    fn cuda_candle_dispatch_rejects_cpu_tensors() {
        let device = candle_core::Device::Cpu;
        let lhs = Tensor::zeros((1, 2), DType::F32, &device).expect("lhs");
        let rhs = Tensor::zeros((1, 2), DType::F32, &device).expect("rhs");

        assert!(try_fused_silu_mul(&lhs, &rhs).is_none());
        assert!(try_fused_l2_norm(&lhs, 1e-6).is_none());
        assert!(try_fused_rms_norm(&lhs, &rhs, 1e-6).is_none());
        assert!(try_fused_gated_rms_norm(&lhs, &rhs, &rhs, 1e-6).is_none());
    }
}
