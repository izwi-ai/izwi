use candle_core::{DType, Shape, Tensor};
use candle_nn::{Linear, Module};

use crate::error::{Error, Result};

/// Exact (erf) GELU, the activation HF transformers and vLLM use for
/// Voxtral's conv stem, audio adapter and Ada-norm (`nn.functional.gelu`,
/// `ACT2FN["gelu"]`). Candle's `gelu` is the tanh approximation. Computed in
/// F32 and returned in the input dtype.
pub(super) fn exact_gelu(x: &Tensor) -> candle_core::Result<Tensor> {
    x.to_dtype(DType::F32)?.gelu_erf()?.to_dtype(x.dtype())
}

pub(super) fn linear_forward_last_dim(linear: &Linear, x: &Tensor) -> Result<Tensor> {
    if x.rank() <= 2 {
        let input = x.contiguous()?;
        let dims = input.dims().to_vec();
        return linear.forward(&input).map_err(|err| {
            Error::InferenceError(format!("Voxtral linear failed for {:?}: {}", dims, err))
        });
    }

    let dims = x.dims();
    let Some((&in_dim, prefix_dims)) = dims.split_last() else {
        return Err(Error::InferenceError(
            "Voxtral linear received a scalar tensor".to_string(),
        ));
    };
    let flat_rows = prefix_dims.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            Error::InferenceError("Voxtral linear prefix shape overflowed".to_string())
        })
    })?;
    let flat = x.contiguous()?.reshape((flat_rows, in_dim))?.contiguous()?;
    let flat_shape = flat.dims().to_vec();
    let projected = linear.forward(&flat).map_err(|err| {
        Error::InferenceError(format!(
            "Voxtral linear failed after flattening {:?} to {:?}: {}",
            dims, flat_shape, err
        ))
    })?;
    let out_dim = projected.dim(1)?;
    let mut out_shape = prefix_dims.to_vec();
    out_shape.push(out_dim);
    projected
        .reshape(Shape::from(out_shape))
        .map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, IndexOp, Tensor};

    #[test]
    fn exact_gelu_matches_erf_not_tanh() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, -1.0, 0.0, 3.0], (4,), &device).unwrap();

        let y = exact_gelu(&x).unwrap().to_vec1::<f32>().unwrap();

        // 0.5 * x * (1 + erf(x / sqrt(2))); the tanh form gives 0.8411920 at 1.
        for (got, want) in y
            .iter()
            .zip([0.841_344_746, -0.158_655_254, 0.0, 2.995_950_306])
        {
            assert!((f64::from(*got) - want).abs() < 1e-6, "{got} != {want}");
        }

        let half = x.to_dtype(DType::F16).unwrap();
        let y = exact_gelu(&half).unwrap();
        assert_eq!(y.dtype(), DType::F16);
        let y = y.to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
        assert!((f64::from(y[0]) - 0.841_344_746).abs() < 1e-3);
    }

    #[test]
    fn linear_forward_last_dim_flattens_rank3_inputs() {
        let device = Device::Cpu;
        let linear = Linear::new(Tensor::ones((4, 2), DType::F32, &device).unwrap(), None);
        let input = Tensor::ones((3, 2), DType::F32, &device)
            .unwrap()
            .unsqueeze(1)
            .unwrap();

        let output = linear_forward_last_dim(&linear, &input).unwrap();

        assert_eq!(output.dims(), &[3, 1, 4]);
        assert_eq!(
            output.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![2.0; 12]
        );
    }

    #[test]
    fn linear_forward_last_dim_contiguous_rank2_slices() {
        let device = Device::Cpu;
        let linear = Linear::new(Tensor::ones((4, 2), DType::F32, &device).unwrap(), None);
        let input = Tensor::ones((3, 5, 2), DType::F32, &device)
            .unwrap()
            .i((.., 0, ..))
            .unwrap();

        let output = linear_forward_last_dim(&linear, &input).unwrap();

        assert_eq!(output.dims(), &[3, 4]);
        assert_eq!(
            output.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![2.0; 12]
        );
    }

    #[cfg(feature = "metal")]
    #[test]
    fn linear_forward_last_dim_handles_voxtral_mlp_shape_on_metal() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "linear_forward_last_dim_handles_voxtral_mlp_shape_on_metal",
        ) else {
            return;
        };
        let linear = Linear::new(
            Tensor::zeros((9_216, 3_072), DType::F32, &device).unwrap(),
            None,
        );
        let input = Tensor::zeros((2, 36, 3_072), DType::F32, &device).unwrap();

        let output = linear_forward_last_dim(&linear, &input).unwrap();

        assert_eq!(output.dims(), &[2, 36, 9_216]);
    }
}
