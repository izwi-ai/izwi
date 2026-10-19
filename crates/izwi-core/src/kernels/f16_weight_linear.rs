//! Weight-only F16 linear layers: F16 weights, F32 activations, F32
//! accumulation and F32 output.
//!
//! This is whisper.cpp's recipe for F32 NeMo checkpoints (its Parakeet
//! converter stores only the 2-D linear and pointwise weights in F16 and keeps
//! activations in F32). On Metal the weight is an F16 ggml `QTensor` applied as
//! a Candle `CustomOp1`, which dispatches ggml's `kernel_mul_mm_f16_f32`:
//! weights staged as half, activations read as float, accumulation and output
//! in float. Candle 0.11 sends a single-row input to `kernel_mul_mv_f16_f32`
//! with zero weight strides (`nb01 = 0`), so every output would read weight
//! row 0; a single row is padded to two rows so it also takes `mul_mm`.
//! `QMatMul::from_arc` must not be used here: it dequantizes F16 QTensors to a
//! dense F32 tensor and keeps none of the memory saving.
//!
//! Elsewhere (CPU, CUDA, and Metal shapes the ggml kernel cannot take) the
//! weight is a dense F16 tensor cast to F32 per call, which gives the same
//! numbers. Candle's CPU `QTensor` op would quantize the activations to F16,
//! so it is not used.

use std::borrow::Cow;
use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Conv1d, Linear, VarBuilder};

use super::numeric_check::{host_f32, relative_l2, xorshift_tensor, NumericSelfCheckReport};
use crate::backends::DTypeSelection;

/// ggml's `mul_mm` reads the reduction dimension in blocks of 32 with no tail.
const GGML_MUL_MM_K_BLOCK: usize = 32;

/// How an [`F16WeightLinear`] multiplies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F16WeightLinearPath {
    /// ggml `mul_mm_f16_f32` on Metal.
    GgmlF16F32,
    /// A dense F16 weight cast to F32 per call.
    DenseF16Upcast,
}

impl F16WeightLinearPath {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GgmlF16F32 => "ggml_f16_f32",
            Self::DenseF16Upcast => "dense_f16_upcast",
        }
    }
}

#[derive(Debug, Clone)]
enum F16Weight {
    Ggml(Arc<QTensor>),
    Dense(Tensor),
}

/// `y = x W^T + b` with `W` stored in F16, `x`, `b` and `y` in F32.
#[derive(Debug, Clone)]
pub struct F16WeightLinear {
    weight: F16Weight,
    bias: Option<Tensor>,
    out_features: usize,
    in_features: usize,
}

impl F16WeightLinear {
    /// Converts a `[out, in]` weight (any float dtype, any device) to F16 on
    /// the host and uploads it to `device` once, at its exact size. `bias`
    /// stays F32 on `device`.
    pub fn new(weight: &Tensor, bias: Option<&Tensor>, device: &Device) -> Result<Self> {
        let (out_features, in_features) = weight.dims2()?;
        let host = weight.to_device(&Device::Cpu)?.to_dtype(DType::F16)?;
        let weight = if device.is_metal() && in_features % GGML_MUL_MM_K_BLOCK == 0 {
            // One exact-size upload of the host F16 bytes (`quantize_onto`
            // would first allocate and clear a pooled device buffer).
            let values = host.flatten_all()?.to_vec1::<half::f16>()?;
            let mut bytes = Vec::with_capacity(values.len() * DType::F16.size_in_bytes());
            for value in values {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let storage = QStorage::from_data(Cow::Owned(bytes), device, GgmlDType::F16)?;
            F16Weight::Ggml(Arc::new(QTensor::new(
                storage,
                (out_features, in_features),
            )?))
        } else {
            F16Weight::Dense(host.to_device(device)?)
        };
        let bias = bias
            .map(|bias| bias.to_dtype(DType::F32)?.to_device(device))
            .transpose()?;
        Ok(Self {
            weight,
            bias,
            out_features,
            in_features,
        })
    }

    /// Converts a pointwise (kernel-size 1) `Conv1d` weight `[out, in, 1]`.
    pub fn from_pointwise_conv1d(
        weight: &Tensor,
        bias: Option<&Tensor>,
        device: &Device,
    ) -> Result<Self> {
        let (out_features, in_features, kernel) = weight.dims3()?;
        if kernel != 1 {
            candle_core::bail!("pointwise conv1d weight must have kernel size 1, got {kernel}");
        }
        Self::new(&weight.reshape((out_features, in_features))?, bias, device)
    }

    pub fn path(&self) -> F16WeightLinearPath {
        match self.weight {
            F16Weight::Ggml(_) => F16WeightLinearPath::GgmlF16F32,
            F16Weight::Dense(_) => F16WeightLinearPath::DenseF16Upcast,
        }
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Resident bytes of the F16 weight and the F32 bias.
    pub fn resident_bytes(&self) -> usize {
        self.out_features * self.in_features * DType::F16.size_in_bytes()
            + self
                .bias
                .as_ref()
                .map_or(0, |bias| bias.elem_count() * DType::F32.size_in_bytes())
    }

    /// The stored weight widened to F32, `[out, in]`.
    pub fn dequantized_weight(&self) -> Result<Tensor> {
        match &self.weight {
            F16Weight::Ggml(weight) => weight.dequantize(&weight.device()),
            F16Weight::Dense(weight) => weight.to_dtype(DType::F32),
        }
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for F16WeightLinear {
    /// `x` is `[..., in]` in any float dtype; the output is F32 `[..., out]`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let Some((&in_features, leading)) = dims.split_last() else {
            candle_core::bail!("F16WeightLinear input must have at least one dimension");
        };
        if in_features != self.in_features {
            candle_core::bail!(
                "F16WeightLinear input has {in_features} features, weight expects {}",
                self.in_features
            );
        }
        let rows = leading.iter().product::<usize>();
        let x = x
            .to_dtype(DType::F32)?
            .reshape((rows, in_features))?
            .contiguous()?;
        let y = match &self.weight {
            // The ggml op reads the weight as `[n, k]` and the input as
            // `[m, k]` row-major, writing F32 `[m, n]`.
            F16Weight::Ggml(weight) if rows == 1 => Tensor::cat(&[&x, &x], 0)?
                .apply_op1_no_bwd(weight.as_ref())?
                .narrow(0, 0, 1)?,
            F16Weight::Ggml(weight) => x.apply_op1_no_bwd(weight.as_ref())?,
            F16Weight::Dense(weight) => x.matmul(&weight.to_dtype(DType::F32)?.t()?)?,
        };
        let y = match &self.bias {
            Some(bias) => y.broadcast_add(bias)?,
            None => y,
        };
        let mut out_dims = leading.to_vec();
        out_dims.push(self.out_features);
        y.reshape(out_dims)
    }
}

/// The largest relative L2 an F16 projection may differ from its F32
/// original by in [`F16WeightLinear::check_against`]. F16 rounding gives
/// about 3e-4 on unstructured input; a wrong transform gives order 1.
pub const F16_PROJECTION_MAX_RELATIVE_L2: f64 = 1e-2;

impl F16WeightLinear {
    /// Compares this layer with the F32 `[out, in]` weight it was built from
    /// on `rows` xorshift rows, on this layer's device, and returns the
    /// relative L2 error. A self-check that compares only the converted
    /// weights with themselves cannot see a bad conversion.
    pub fn check_against(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        rows: usize,
    ) -> Result<f64> {
        let device = match &self.weight {
            F16Weight::Ggml(weight) => weight.device(),
            F16Weight::Dense(weight) => weight.device().clone(),
        };
        let x = xorshift_tensor(&[rows, self.in_features], 0x5EED, 1.0, &device)?;
        let reference = Linear::new(
            weight.to_device(&Device::Cpu)?.to_dtype(DType::F32)?,
            bias.map(|bias| bias.to_device(&Device::Cpu)?.to_dtype(DType::F32))
                .transpose()?,
        )
        .forward(&x.to_device(&Device::Cpu)?)?;
        Ok(relative_l2(
            &host_f32(&self.forward(&x)?)?,
            &host_f32(&reference)?,
        ))
    }
}

/// The projection precision a model loaded with, and why.
#[derive(Debug, Clone)]
pub struct WeightOnlyF16Load {
    pub projections_f16: bool,
    pub reason: String,
    pub self_check: Option<NumericSelfCheckReport>,
}

impl WeightOnlyF16Load {
    pub fn dense(reason: impl Into<String>) -> Self {
        Self {
            projections_f16: false,
            reason: reason.into(),
            self_check: None,
        }
    }

    pub fn diagnostics(&self) -> serde_json::Value {
        serde_json::json!({
            "projection_weights": if self.projections_f16 { "F16" } else { "F32" },
            "activations": "F32",
            "reason": self.reason,
            "self_check": self.self_check.as_ref().map(NumericSelfCheckReport::diagnostics),
        })
    }
}

/// Loads a model at `precision` (F16 = weight-only F16 projections on
/// `device`, from `DeviceProfile::resolve_metal_weight_only_f16`). `load`
/// builds the model with its projections in F16 on the given device, or all
/// dense with `None`. The F16 model is kept only if `self_check` passes;
/// otherwise it is dropped, the model reloads dense, and the reason is
/// recorded.
pub fn load_weight_only_f16<T, E: From<crate::error::Error>>(
    label: &str,
    precision: &DTypeSelection,
    device: &Device,
    mut load: impl FnMut(Option<&Device>) -> std::result::Result<T, E>,
    self_check: impl FnOnce(&T) -> NumericSelfCheckReport,
) -> std::result::Result<(T, WeightOnlyF16Load), E> {
    if precision.dtype != DType::F16 {
        return Ok((
            load(None)?,
            WeightOnlyF16Load::dense(precision.reason.clone()),
        ));
    }
    let model = load(Some(device))?;
    let report = self_check(&model);
    if report.passed {
        return Ok((
            model,
            WeightOnlyF16Load {
                projections_f16: true,
                reason: precision.reason.to_string(),
                self_check: Some(report),
            },
        ));
    }
    tracing::warn!(
        "{label} weight-only F16 failed its load-time self-check; reloading in F32: {}",
        report.reason
    );
    drop(model);
    // The load's gate frees the dropped model and admits the dense reload's
    // memory (each family's `precision_reload_plan`).
    crate::kernels::numeric_check::admit_precision_reload(label, DType::F32).map_err(E::from)?;
    let model = load(None)?;
    Ok((
        model,
        WeightOnlyF16Load {
            projections_f16: false,
            reason: format!("{}; fell back to F32", report.reason),
            self_check: Some(report),
        },
    ))
}

/// The builder a projection that may load as weight-only F16 reads from:
/// with `f16` set, the host, so the weight is read and converted there and
/// its F32 form never reaches the GPU; otherwise `vb` itself.
pub fn projection_var_builder<'a>(vb: &VarBuilder<'a>, f16: Option<&Device>) -> VarBuilder<'a> {
    match f16 {
        Some(_) => vb.clone().set_device(Device::Cpu),
        None => vb.clone(),
    }
}

/// A projection held dense (Candle's `Linear`, any dtype) or weight-only F16.
#[derive(Debug, Clone)]
pub enum ProjectionLinear {
    Dense(Linear),
    F16(F16WeightLinear),
}

impl ProjectionLinear {
    /// Keeps `linear` dense, or converts it to weight-only F16 on `f16_device`.
    pub fn new(linear: Linear, f16_device: Option<&Device>) -> Result<Self> {
        match f16_device {
            None => Ok(Self::Dense(linear)),
            Some(device) => Ok(Self::F16(F16WeightLinear::new(
                linear.weight(),
                linear.bias(),
                device,
            )?)),
        }
    }

    pub fn as_f16(&self) -> Option<&F16WeightLinear> {
        match self {
            Self::Dense(_) => None,
            Self::F16(linear) => Some(linear),
        }
    }
}

impl From<Linear> for ProjectionLinear {
    fn from(linear: Linear) -> Self {
        Self::Dense(linear)
    }
}

impl Module for ProjectionLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(linear) => linear.forward(x),
            Self::F16(linear) => linear.forward(x),
        }
    }
}

/// A pointwise (kernel-size 1) `Conv1d` over `[B, C, T]`, held dense or as a
/// weight-only F16 projection over the channels.
#[derive(Debug, Clone)]
pub enum PointwiseConv1d {
    Dense(Conv1d),
    F16(F16WeightLinear),
}

impl PointwiseConv1d {
    /// Keeps `conv` dense, or converts it to weight-only F16 on `f16_device`.
    /// Only a plain pointwise convolution (kernel 1, stride 1, no padding,
    /// dilation or groups) converts.
    pub fn new(conv: Conv1d, f16_device: Option<&Device>) -> Result<Self> {
        let Some(device) = f16_device else {
            return Ok(Self::Dense(conv));
        };
        let config = conv.config();
        if config.padding != 0 || config.stride != 1 || config.dilation != 1 || config.groups != 1 {
            candle_core::bail!("only a plain pointwise conv1d can load as an F16 projection");
        }
        Ok(Self::F16(F16WeightLinear::from_pointwise_conv1d(
            conv.weight(),
            conv.bias(),
            device,
        )?))
    }

    pub fn as_f16(&self) -> Option<&F16WeightLinear> {
        match self {
            Self::Dense(_) => None,
            Self::F16(linear) => Some(linear),
        }
    }
}

impl From<Conv1d> for PointwiseConv1d {
    fn from(conv: Conv1d) -> Self {
        Self::Dense(conv)
    }
}

impl Module for PointwiseConv1d {
    /// `x` is `[B, C_in, T]`; the output is `[B, C_out, T]` (a transposed
    /// view on the F16 path).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(conv) => conv.forward(x),
            Self::F16(linear) => linear.forward(&x.transpose(1, 2)?)?.transpose(1, 2),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::numeric_check::max_abs_diff;
    use candle_nn::Conv1dConfig;

    /// A dense F32 reference and the F16 layer built from the same weights.
    fn layers(
        out_features: usize,
        in_features: usize,
        with_bias: bool,
        device: &Device,
    ) -> (Linear, F16WeightLinear) {
        let weight = xorshift_tensor(&[out_features, in_features], 11, 0.2, &Device::Cpu).unwrap();
        let bias =
            with_bias.then(|| xorshift_tensor(&[out_features], 12, 0.5, &Device::Cpu).unwrap());
        let dense = Linear::new(weight.clone(), bias.clone());
        let f16 = F16WeightLinear::new(&weight, bias.as_ref(), device).unwrap();
        (dense, f16)
    }

    fn compare(dense: &Linear, f16: &F16WeightLinear, x: &Tensor) -> (f64, f32) {
        let reference =
            host_f32(&dense.forward(&x.to_device(&Device::Cpu).unwrap()).unwrap()).unwrap();
        let y = f16.forward(x).unwrap();
        assert_eq!(y.dtype(), DType::F32);
        let values = host_f32(&y).unwrap();
        (
            relative_l2(&values, &reference),
            max_abs_diff(&values, &reference),
        )
    }

    #[test]
    fn cpu_matches_the_dense_f32_linear_within_f16_weight_rounding() {
        let device = Device::Cpu;
        for (out_features, in_features, with_bias) in
            [(96, 64, true), (40, 24, false), (7, 1024, true)]
        {
            let (dense, f16) = layers(out_features, in_features, with_bias, &device);
            assert_eq!(f16.path(), F16WeightLinearPath::DenseF16Upcast);
            for shape in [
                vec![1, 5, in_features],
                vec![3, in_features],
                vec![1, 1, in_features],
            ] {
                let x = xorshift_tensor(&shape, 21, 1.0, &device).unwrap();
                let (rel, _) = compare(&dense, &f16, &x);
                assert!(
                    rel < 1e-3,
                    "{out_features}x{in_features} {shape:?}: {rel:e}"
                );
                // F16 rounds weights, so the result is close but not exact.
                assert!(rel > 0.0, "the weight was not rounded to F16");
            }
            let y = f16
                .forward(&xorshift_tensor(&[2, 3, in_features], 22, 1.0, &device).unwrap())
                .unwrap();
            assert_eq!(y.dims(), &[2, 3, out_features]);
        }
    }

    #[test]
    fn stores_the_weight_in_half_and_widens_it_exactly() {
        let (dense, f16) = layers(32, 64, true, &Device::Cpu);
        assert_eq!(f16.resident_bytes(), 32 * 64 * 2 + 32 * 4);
        let widened = host_f32(&f16.dequantized_weight().unwrap()).unwrap();
        let rounded = host_f32(
            &dense
                .weight()
                .to_dtype(DType::F16)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(widened, rounded);
    }

    #[test]
    fn pointwise_conv1d_weights_load_as_linear() {
        let weight = xorshift_tensor(&[8, 4, 1], 3, 1.0, &Device::Cpu).unwrap();
        let layer = F16WeightLinear::from_pointwise_conv1d(&weight, None, &Device::Cpu).unwrap();
        assert_eq!((layer.out_features(), layer.in_features()), (8, 4));
        let bad = xorshift_tensor(&[8, 4, 3], 3, 1.0, &Device::Cpu).unwrap();
        assert!(F16WeightLinear::from_pointwise_conv1d(&bad, None, &Device::Cpu).is_err());
    }

    #[test]
    fn rejects_mismatched_input_features() {
        let (_, f16) = layers(8, 32, false, &Device::Cpu);
        let x = xorshift_tensor(&[2, 16], 1, 1.0, &Device::Cpu).unwrap();
        assert!(f16.forward(&x).is_err());
    }

    #[test]
    fn check_against_measures_the_conversion_against_the_original() {
        let (dense, f16) = layers(48, 64, true, &Device::Cpu);
        let rel = f16.check_against(dense.weight(), dense.bias(), 8).unwrap();
        assert!(rel > 0.0 && rel < 1e-3, "{rel:e}");
        // A transform that lost the weights is far outside the bound.
        let wrong = (dense.weight() * -1.0).unwrap();
        let rel = f16.check_against(&wrong, dense.bias(), 8).unwrap();
        assert!(rel > F16_PROJECTION_MAX_RELATIVE_L2, "{rel:e}");
    }

    #[test]
    fn weight_only_f16_load_keeps_f16_only_when_the_self_check_passes() {
        use crate::kernels::numeric_check::{run_numeric_self_check, F16_SAFE_MAX_ABS};

        let device = Device::Cpu;
        let (dense, _) = layers(8, 32, false, &device);
        let precision = |dtype| DTypeSelection {
            dtype,
            reason: "test policy".into(),
        };
        let load = |f16: Option<&Device>| {
            ProjectionLinear::new(dense.clone(), f16).map_err(crate::error::Error::from)
        };
        let check = |scale: f32| {
            move |layer: &ProjectionLinear| {
                run_numeric_self_check("probe", F16_SAFE_MAX_ABS, || {
                    let x = xorshift_tensor(&[2, 32], 1, scale, &Device::Cpu)?;
                    Ok::<_, candle_core::Error>(vec![("y".into(), layer.forward(&x)?)])
                })
            }
        };

        let (layer, outcome) =
            load_weight_only_f16("test", &precision(DType::F16), &device, load, check(1.0))
                .unwrap();
        assert!(outcome.projections_f16 && layer.as_f16().is_some());
        assert!(outcome.self_check.unwrap().passed);

        let mut loads = Vec::new();
        let (layer, outcome) = load_weight_only_f16(
            "test",
            &precision(DType::F16),
            &device,
            |f16| {
                loads.push(f16.is_some());
                load(f16)
            },
            check(1.0e9),
        )
        .unwrap();
        assert_eq!(loads, [true, false]);
        assert!(!outcome.projections_f16 && layer.as_f16().is_none());
        assert!(
            outcome.reason.ends_with("fell back to F32"),
            "{}",
            outcome.reason
        );
        assert_eq!(outcome.diagnostics()["projection_weights"], "F32");

        let (layer, outcome) =
            load_weight_only_f16("test", &precision(DType::F32), &device, load, |_| {
                unreachable!("no self-check for a dense load")
            })
            .unwrap();
        assert!(layer.as_f16().is_none() && outcome.self_check.is_none());
        assert_eq!(outcome.reason, "test policy");
    }

    #[test]
    fn projection_wrappers_match_their_dense_forms() {
        let device = Device::Cpu;
        let (dense, _) = layers(16, 32, true, &device);
        let x = xorshift_tensor(&[2, 5, 32], 4, 1.0, &device).unwrap();
        let reference = host_f32(&dense.forward(&x).unwrap()).unwrap();
        let kept = ProjectionLinear::new(dense.clone(), None).unwrap();
        assert!(kept.as_f16().is_none());
        assert_eq!(host_f32(&kept.forward(&x).unwrap()).unwrap(), reference);
        let f16 = ProjectionLinear::new(dense.clone(), Some(&device)).unwrap();
        assert!(f16.as_f16().is_some());
        let rel = relative_l2(&host_f32(&f16.forward(&x).unwrap()).unwrap(), &reference);
        assert!(rel > 0.0 && rel < 1e-3, "{rel:e}");

        let weight = xorshift_tensor(&[24, 32, 1], 6, 0.2, &device).unwrap();
        let conv = Conv1d::new(weight, None, Conv1dConfig::default());
        let x = xorshift_tensor(&[2, 32, 7], 7, 1.0, &device).unwrap();
        let reference = conv.forward(&x).unwrap();
        let pointwise = PointwiseConv1d::new(conv.clone(), Some(&device)).unwrap();
        let y = pointwise.forward(&x).unwrap();
        assert_eq!(y.dims(), reference.dims());
        let rel = relative_l2(&host_f32(&y).unwrap(), &host_f32(&reference).unwrap());
        assert!(rel > 0.0 && rel < 1e-3, "{rel:e}");
        assert!(PointwiseConv1d::new(conv, None).unwrap().as_f16().is_none());

        let grouped = Conv1d::new(
            xorshift_tensor(&[8, 4, 1], 8, 0.2, &device).unwrap(),
            None,
            Conv1dConfig {
                groups: 2,
                ..Default::default()
            },
        );
        assert!(PointwiseConv1d::new(grouped, Some(&device)).is_err());
    }

    /// The Metal layer must run ggml's f16 x f32 kernel (not a dense
    /// fallback) for matrix shapes (partial 64x32 tiles included) and for a
    /// single row (which Candle would send to its broken `mul_mv`), and agree
    /// with the F32 reference and the CPU path.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_uses_the_ggml_f16_f32_kernel_and_matches_f32() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "metal_uses_the_ggml_f16_f32_kernel_and_matches_f32",
        ) else {
            return;
        };
        for (out_features, in_features, with_bias) in
            [(1024, 1024, false), (96, 64, true), (640, 4096, false)]
        {
            let (dense, f16) = layers(out_features, in_features, with_bias, &device);
            assert_eq!(f16.path(), F16WeightLinearPath::GgmlF16F32);
            let cpu = F16WeightLinear::new(dense.weight(), dense.bias(), &Device::Cpu).unwrap();
            for shape in [
                vec![1, 37, in_features],
                vec![130, in_features],
                vec![1, 1, in_features],
                vec![2, 1, in_features],
            ] {
                let x = xorshift_tensor(&shape, 31, 1.0, &device).unwrap();
                let (rel, max) = compare(&dense, &f16, &x);
                assert!(
                    rel < 1e-3,
                    "{out_features}x{in_features} {shape:?}: rel {rel:e} max {max:e}"
                );
                let metal = host_f32(&f16.forward(&x).unwrap()).unwrap();
                let host =
                    host_f32(&cpu.forward(&x.to_device(&Device::Cpu).unwrap()).unwrap()).unwrap();
                let rel_cpu = relative_l2(&metal, &host);
                assert!(
                    rel_cpu < 1e-5,
                    "Metal vs CPU F16 path {shape:?}: {rel_cpu:e}"
                );
            }
        }
        // A reduction dimension ggml's mul_mm cannot take falls back densely.
        let (dense, f16) = layers(16, 40, true, &device);
        assert_eq!(f16.path(), F16WeightLinearPath::DenseF16Upcast);
        let x = xorshift_tensor(&[3, 40], 5, 1.0, &device).unwrap();
        let (rel, _) = compare(&dense, &f16, &x);
        assert!(rel < 1e-3, "{rel:e}");
    }
}
