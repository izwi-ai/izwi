//! Opt-in BF16 copies of audio decoders and vocoders on CUDA.
//!
//! A decoder that runs F32 may load a BF16 copy on CUDA when its own
//! environment switch asks for it (each defaults off). The copy serves only
//! after a load-time check: the decoder's probe through both copies, every
//! probed output within the stated relative L2 of the F32 copy's. Otherwise
//! the F32 copy serves, and the diagnostics say why. Making any copy the
//! default needs H100 parity, an ASR round-trip WER and a listening check.

use candle_core::{DType, Device};
use serde_json::json;

use crate::error::Result;
use crate::kernels::numeric_check::relative_l2;

/// Which copy of a decoder serves, and why.
#[derive(Debug, Clone)]
pub(crate) struct DecoderPrecision {
    pub(crate) dtype: DType,
    pub(crate) reason: String,
    /// The probe's outputs and their relative L2 against F32, when it ran.
    pub(crate) relative_l2: Vec<(String, f64)>,
    pub(crate) tolerance: f64,
}

impl DecoderPrecision {
    fn f32(reason: String, tolerance: f64) -> Self {
        Self {
            dtype: DType::F32,
            reason,
            relative_l2: Vec::new(),
            tolerance,
        }
    }

    pub(crate) fn diagnostics(&self) -> serde_json::Value {
        json!({
            "dtype": format!("{:?}", self.dtype),
            "reason": self.reason,
            "tolerance_relative_l2": self.tolerance,
            "relative_l2": self
                .relative_l2
                .iter()
                .map(|(output, value)| json!({ "output": output, "relative_l2": value }))
                .collect::<Vec<_>>(),
        })
    }
}

/// The copy of a decoder to serve: `f32_copy`, unless `switch` is set on a
/// CUDA `device`; then the BF16 copy `load_bf16` builds, when each output
/// `probe` names (host F32) is within `tolerance` relative L2 of the F32
/// copy's. A failed load or probe keeps F32, with the reason.
pub(crate) fn opt_in_bf16<T>(
    label: &str,
    switch: &str,
    device: &Device,
    tolerance: f64,
    f32_copy: T,
    load_bf16: impl FnOnce() -> Result<T>,
    probe: impl Fn(&T) -> Result<Vec<(String, Vec<f32>)>>,
) -> (T, DecoderPrecision) {
    if !crate::env::flag_or(switch, false) {
        return (
            f32_copy,
            DecoderPrecision::f32(format!("{switch} is off"), tolerance),
        );
    }
    if !device.is_cuda() {
        return (
            f32_copy,
            DecoderPrecision::f32(format!("{switch} applies on CUDA only"), tolerance),
        );
    }
    let (copy, precision) = check_bf16(label, tolerance, f32_copy, load_bf16, probe);
    match precision.dtype {
        DType::BF16 => tracing::info!(
            relative_l2 = ?precision.relative_l2,
            tolerance,
            "{label} runs in BF16 ({switch})"
        ),
        _ => tracing::warn!(
            reason = %precision.reason,
            "{label} stays F32 although {switch} is set"
        ),
    }
    (copy, precision)
}

/// [`opt_in_bf16`]'s check, on any device.
pub(crate) fn check_bf16<T>(
    label: &str,
    tolerance: f64,
    f32_copy: T,
    load_bf16: impl FnOnce() -> Result<T>,
    probe: impl Fn(&T) -> Result<Vec<(String, Vec<f32>)>>,
) -> (T, DecoderPrecision) {
    let compared = (|| -> Result<(T, Vec<(String, f64)>)> {
        let reference = probe(&f32_copy)?;
        let bf16_copy = load_bf16()?;
        let outputs = probe(&bf16_copy)?;
        if outputs.len() != reference.len() {
            return Err(crate::error::Error::InferenceError(format!(
                "{label} BF16 probe returned {} outputs, F32 {}",
                outputs.len(),
                reference.len()
            )));
        }
        let distances = outputs
            .iter()
            .zip(&reference)
            .map(|((name, values), (_, expected))| {
                let distance = if values.len() == expected.len() {
                    relative_l2(values, expected)
                } else {
                    f64::INFINITY
                };
                (name.clone(), distance)
            })
            .collect();
        Ok((bf16_copy, distances))
    })();
    match compared {
        Ok((bf16_copy, distances))
            if distances
                .iter()
                .all(|(_, distance)| distance.is_finite() && *distance <= tolerance) =>
        {
            (
                bf16_copy,
                DecoderPrecision {
                    dtype: DType::BF16,
                    reason: format!("BF16 within {tolerance} relative L2 of F32"),
                    relative_l2: distances,
                    tolerance,
                },
            )
        }
        Ok((_, distances)) => (
            f32_copy,
            DecoderPrecision {
                dtype: DType::F32,
                reason: format!("BF16 past {tolerance} relative L2 of F32"),
                relative_l2: distances,
                tolerance,
            },
        ),
        Err(error) => (
            f32_copy,
            DecoderPrecision::f32(format!("BF16 check failed: {error}"), tolerance),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::numeric_check::xorshift_tensor;
    use candle_core::Tensor;

    /// A toy decoder: one projection, its weight in its copy's dtype.
    fn probe(weight: &Tensor) -> Result<Vec<(String, Vec<f32>)>> {
        let x = xorshift_tensor(&[4, 64], 3, 1.0, weight.device())?.to_dtype(weight.dtype())?;
        let y = x.matmul(&weight.t()?)?.to_dtype(DType::F32)?;
        Ok(vec![(
            "projection".into(),
            y.flatten_all()?.to_vec1::<f32>()?,
        )])
    }

    /// The BF16 copy serves within the tolerance, the F32 copy past it or
    /// when the copy fails to load; the distances are reported either way.
    #[test]
    fn bf16_copies_serve_only_within_the_tolerance() {
        let device = Device::Cpu;
        let weight = xorshift_tensor(&[32, 64], 7, 0.5, &device).unwrap();
        let bf16 = || Ok(weight.to_dtype(DType::BF16)?);

        let (copy, precision) = check_bf16("toy", 0.05, weight.clone(), bf16, probe);
        assert_eq!(copy.dtype(), DType::BF16);
        assert_eq!(precision.dtype, DType::BF16);
        let distance = precision.relative_l2[0].1;
        assert!(distance > 0.0 && distance <= 0.05, "{distance}");

        let (copy, precision) = check_bf16("toy", distance / 2.0, weight.clone(), bf16, probe);
        assert_eq!(copy.dtype(), DType::F32);
        assert_eq!(precision.dtype, DType::F32);
        assert_eq!(precision.relative_l2.len(), 1);
        assert!(precision.reason.contains("past"), "{}", precision.reason);

        let failing =
            || -> Result<Tensor> { Err(crate::error::Error::ModelLoadError("no copy".into())) };
        let (copy, precision) = check_bf16("toy", 0.05, weight.clone(), failing, probe);
        assert_eq!(copy.dtype(), DType::F32);
        assert!(precision.reason.contains("no copy"), "{}", precision.reason);
        assert_eq!(
            precision.diagnostics()["dtype"],
            "F32",
            "{}",
            precision.diagnostics()
        );
    }

    /// Off by default and off the GPU: the F32 copy serves and the reason
    /// names the switch; nothing loads a BF16 copy.
    #[test]
    fn bf16_copies_need_the_switch_and_cuda() {
        let _env = crate::env_test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // A registered switch (an unregistered read panics in debug builds).
        const SWITCH: &str = "IZWI_CHATTERBOX_ACOUSTIC_BF16";
        let previous = std::env::var_os(SWITCH);
        let device = Device::Cpu;
        let weight = xorshift_tensor(&[8, 64], 9, 0.5, &device).unwrap();
        let never = || -> Result<Tensor> { panic!("no BF16 copy may load") };
        std::env::remove_var(SWITCH);
        let (copy, off) = opt_in_bf16("toy", SWITCH, &device, 0.05, weight.clone(), never, probe);
        std::env::set_var(SWITCH, "1");
        let (cpu_copy, cpu) = opt_in_bf16("toy", SWITCH, &device, 0.05, weight, never, probe);
        match previous {
            Some(value) => std::env::set_var(SWITCH, value),
            None => std::env::remove_var(SWITCH),
        }
        assert_eq!(copy.dtype(), DType::F32);
        assert!(off.reason.contains(SWITCH), "{}", off.reason);
        assert_eq!(cpu_copy.dtype(), DType::F32);
        assert!(cpu.reason.contains("CUDA only"), "{}", cpu.reason);
    }
}
