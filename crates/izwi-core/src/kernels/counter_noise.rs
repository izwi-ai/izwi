//! Counter-based Gumbel noise generated where it is used: element `j` of
//! row `r` is a function of `keys[r]` and `j` alone,
//!
//! ```text
//! z = splitmix64_finalize(key + (j + 1) * 0x9E3779B97F4A7C15)
//! u = ((z >> 41) + 0.5) / 2^23          in (0, 1), exact in F32
//! g = -ln(-ln(u))
//! ```
//!
//! so a row's noise never depends on which rows share the call, and a
//! sampler that needs fresh noise every step uploads one 64-bit key a row
//! instead of the noise. On CUDA one thread computes each element (its `logf`
//! rounds as the device's does, so values match the host's reference to a
//! few ulps, and always match themselves); elsewhere the host computes and
//! uploads them.
use candle_core::{DType, Device, Result, Tensor};

/// The uniform of element `index` under `key`.
pub fn uniform(key: u64, index: u64) -> f32 {
    let mut z = key.wrapping_add(index.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 41) as u32 as f32 + 0.5) * (1.0 / 8_388_608.0)
}

/// The host's Gumbel value of element `index` under `key`.
pub fn host_gumbel(key: u64, index: u64) -> f32 {
    -(-uniform(key, index).ln()).ln()
}

/// `[keys.len(), per_row]` F32 Gumbel noise on `device`, row `r` from
/// `keys[r]`.
pub fn gumbel(keys: &[u64], per_row: usize, device: &Device) -> Result<Tensor> {
    if keys.is_empty() || per_row == 0 {
        candle_core::bail!("counter noise needs rows and columns")
    }
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda_impl::run(keys, per_row, device),
        _ => {
            let values = keys
                .iter()
                .flat_map(|&key| (0..per_row as u64).map(move |index| host_gumbel(key, index)))
                .collect::<Vec<_>>();
            Tensor::from_vec(values, (keys.len(), per_row), device)?.to_dtype(DType::F32)
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::{CudaStorage, WrapErr};
    use candle_core::op::BackpropOp;
    use candle_core::{Device, Result, Shape, Storage, Tensor};

    const MODULE: &str = "izwi_counter_noise";

    pub(super) fn run(keys: &[u64], per_row: usize, device: &Device) -> Result<Tensor> {
        let device = device.as_cuda_device()?;
        let total = keys.len() * per_row;
        let keys_on_device = device.clone_htod(keys)?;
        // SAFETY: the kernel writes every element.
        let out = unsafe { device.alloc::<f32>(total)? };
        let function = device.get_or_load_custom_func(
            "izwi_counter_gumbel",
            MODULE,
            super::super::cuda::cuda_ptx_counter_noise(),
        )?;
        let mut builder = function.builder();
        builder.arg(&keys_on_device);
        builder.arg(&out);
        candle_core::builder_arg!(builder, per_row as i64, total as i64);
        let config = LaunchConfig {
            grid_dim: (total.div_ceil(256) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: argument order and types match the kernel.
        unsafe { builder.launch(config) }.w()?;
        Ok(Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(out, device.clone())),
            Shape::from((keys.len(), per_row)),
            BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_a_function_of_the_key_and_index_alone() {
        let device = Device::Cpu;
        let together = gumbel(&[3, 9, 3], 1_000, &device)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let alone = gumbel(&[9], 1_000, &device)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(together[1], alone[0]);
        assert_eq!(together[0], together[2]);
        assert_ne!(together[0], together[1]);
        // Finite everywhere, with Gumbel(0, 1)'s mean (Euler's constant).
        let values = gumbel(&[77], 200_000, &device)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()
            .remove(0);
        assert!(values.iter().all(|v| v.is_finite()));
        let mean = values.iter().map(|&v| f64::from(v)).sum::<f64>() / values.len() as f64;
        assert!((mean - std::f64::consts::EULER_GAMMA).abs() < 0.02, "{mean}");
    }

    #[test]
    fn uniforms_stay_strictly_inside_the_unit_interval() {
        assert_eq!(uniform(0, u64::MAX), uniform(0, u64::MAX));
        let extremes = [0.5f32 / 8_388_608.0, 1.0 - 0.5 / 8_388_608.0];
        assert!(extremes[1] < 1.0 && extremes[0] > 0.0);
        for key in 0..2_000u64 {
            let u = uniform(key.wrapping_mul(0x1234_5678_9ABC_DEF1), key);
            assert!(u >= extremes[0] && u <= extremes[1]);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_noise_tracks_the_host_reference() {
        let Some(device) = crate::kernels::cuda::cuda_test_device() else {
            return;
        };
        let keys = [5u64, 1 << 40, u64::MAX];
        let got = gumbel(&keys, 4_096, &device)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for (row, &key) in got.iter().zip(&keys) {
            for (index, &value) in row.iter().enumerate() {
                let expected = host_gumbel(key, index as u64);
                assert!((value - expected).abs() <= 1e-5 * expected.abs().max(1.0));
            }
        }
    }
}
