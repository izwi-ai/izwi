//! Opt-in activation checks and mandatory finite-logit validation.
use candle_core::Tensor;

use crate::error::{Error, Result};

pub(crate) fn enabled() -> bool {
    static ENABLED: crate::env::ReadOnce<bool> =
        crate::env::ReadOnce::new(|| crate::env::flag_or("IZWI_LFM2_DIAGNOSTICS", false));
    ENABLED.get()
}

/// Reduce on the tensor's device; only one byte crosses the device boundary.
/// NaN comparisons are false, and abs(infinity) is not less than infinity.
pub(crate) fn validate_finite(tensor: &Tensor, stage: &str) -> Result<()> {
    if finite_flag(tensor)?.to_vec1::<f32>()?[0] == 0.0 {
        return Err(non_finite_error(tensor, stage));
    }
    Ok(())
}

/// The finite check left on the device: a `[1]` F32 tensor that is 1 when
/// every value of `tensor` is finite and 0 otherwise, for a caller to read
/// back together with other per-step values in one synchronization.
pub(crate) fn finite_flag(tensor: &Tensor) -> Result<Tensor> {
    if tensor.elem_count() == 0 {
        return Err(Error::InferenceError("finite check of an empty tensor".into()));
    }
    tensor
        .abs()?
        .lt(f64::INFINITY)?
        .min_all()?
        .to_dtype(candle_core::DType::F32)?
        .reshape(1)
        .map_err(Error::from)
}

pub(crate) fn non_finite_error(tensor: &Tensor, stage: &str) -> Error {
    Error::InferenceError(format!(
        "{stage}: non-finite values (NaN or infinity), shape {:?}, dtype {:?}, device {:?}",
        tensor.dims(),
        tensor.dtype(),
        tensor.device().location()
    ))
}

pub(crate) fn check(tensor: &Tensor, stage: &str, position: usize) -> Result<()> {
    if enabled() {
        validate_finite(tensor, &format!("LFM2 {stage}, position {position}"))?;
        let bounds = Tensor::stack(&[tensor.min_all()?, tensor.max_all()?], 0)?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1::<f32>()?;
        tracing::info!(stage, position, shape = ?tensor.dims(), dtype = ?tensor.dtype(),
            device = ?tensor.device().location(), min = bounds[0], max = bounds[1],
            "LFM2 activation diagnostics");
    }
    Ok(())
}

pub(crate) fn check_layer(
    tensor: &Tensor,
    stage: &str,
    layer: usize,
    position: usize,
) -> Result<()> {
    if enabled() {
        check(tensor, &format!("layer {layer} {stage}"), position)?;
    }
    Ok(())
}

pub(crate) fn check_batch_layer(
    tensor: &Tensor,
    stage: &str,
    layer: usize,
    positions: &[usize],
) -> Result<()> {
    if enabled() {
        for (row, &position) in positions.iter().enumerate() {
            check(
                &tensor.narrow(0, row, 1)?,
                &format!("layer {layer} row {row} {stage}"),
                position,
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_finite;
    use candle_core::{Device, Tensor};

    fn check_device(device: &Device) {
        let finite = Tensor::new(&[0f32, -1., f32::MAX, f32::MIN], device).unwrap();
        validate_finite(&finite, "finite logits").unwrap();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            for values in [[value, 1., 2.], [1., value, 2.], [1., 2., value]] {
                let tensor = Tensor::new(&values, device).unwrap();
                let error = validate_finite(&tensor, "layer 3 Q projection, position 42")
                    .unwrap_err()
                    .to_string();
                assert!(error.contains("non-finite"), "{error}");
                assert!(
                    error.contains("layer 3 Q projection, position 42"),
                    "{error}"
                );
            }
        }
        let strided = Tensor::new(&[[1f32, 2.], [3., f32::NAN]], device)
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        assert!(validate_finite(&strided, "strided").is_err());
        let empty = Tensor::new(&[] as &[f32], device).unwrap();
        assert!(validate_finite(&empty, "empty logits").is_err());
    }

    #[test]
    fn lfm2_finite_validation_rejects_nonfinite_values() {
        check_device(&Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires an NVIDIA GPU; run explicitly with --ignored"]
    fn cuda_lfm2_finite_validation_rejects_nonfinite_values() {
        check_device(&Device::new_cuda(0).expect("CUDA device required"));
    }
}
