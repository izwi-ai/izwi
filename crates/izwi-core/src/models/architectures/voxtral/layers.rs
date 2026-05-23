use candle_core::{Shape, Tensor};
use candle_nn::{Linear, Module};

use crate::error::{Error, Result};

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
    use candle_core::{DType, Device, IndexOp, Tensor};

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
        let device = match std::panic::catch_unwind(|| Device::new_metal(0)) {
            Ok(Ok(device)) => device,
            _ => return,
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
