//! Where the model runs and in which dtype.
//!
//! SOV-Transcribe-1 is served on CUDA only, in the checkpoint's own BF16
//! with the routers and every norm in F32, as the reference does. The CPU
//! runs the same code in F32 as the test reference for a tiny synthetic
//! checkpoint; it is not a serving tier, and the published model is refused
//! there before any weight is read ([`require_serving_backend`]). Metal is
//! not supported at all.

use candle_core::{DType, Device};
use half::bf16;

use crate::backends::BackendKind;
use crate::error::{Error, Result};

pub fn backend_of(device: &Device) -> BackendKind {
    if device.is_cuda() {
        BackendKind::Cuda
    } else if device.is_metal() {
        BackendKind::Metal
    } else {
        BackendKind::Cpu
    }
}

/// The message every entry point gives off CUDA.
pub fn cuda_only_message(backend: BackendKind) -> String {
    let here = match backend {
        BackendKind::Cpu => "the CPU",
        BackendKind::Metal => "Metal",
        BackendKind::Cuda => "CUDA",
    };
    format!("SOV-Transcribe-1 needs an NVIDIA GPU (CUDA); this server runs on {here}")
}

/// Refuses the published model off CUDA.
pub fn require_serving_backend(device: &Device) -> Result<()> {
    match backend_of(device) {
        BackendKind::Cuda => Ok(()),
        other => Err(Error::InvalidInput(cuda_only_message(other))),
    }
}

/// Marks a test's synthetic checkpoint (`synthetic::write_checkpoint`).
#[cfg(test)]
pub(crate) const SYNTHETIC_MARKER: &str = ".izwi-synthetic-checkpoint";

/// [`require_serving_backend`] for the checkpoint in `model_dir`. Tests run
/// the tiny synthetic checkpoint (marked by its writer) on the CPU through
/// the real loader; nothing else passes off CUDA.
pub fn require_serving_backend_for(model_dir: &std::path::Path, device: &Device) -> Result<()> {
    #[cfg(test)]
    if model_dir.join(SYNTHETIC_MARKER).is_file() {
        return Ok(());
    }
    let _ = model_dir;
    require_serving_backend(device)
}

/// The dtype of weights and activations on `device`.
pub fn compute_dtype(device: &Device) -> Result<DType> {
    match backend_of(device) {
        BackendKind::Cuda => Ok(DType::BF16),
        BackendKind::Cpu => Ok(DType::F32),
        BackendKind::Metal => Err(Error::InvalidInput(cuda_only_message(BackendKind::Metal))),
    }
}

/// The text-embedding scale `sqrt(hidden)` as the reference applies it: cast
/// to the embedding's dtype first, so a BF16 checkpoint multiplies by the
/// BF16 rounding (53.0 for hidden 2816, not 53.066).
pub fn embedding_scale(hidden: usize, checkpoint_is_bf16: bool) -> f32 {
    let scale = (hidden as f32).sqrt();
    if checkpoint_is_bf16 {
        bf16::from_f32(scale).to_f32()
    } else {
        scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_published_embedding_scale_rounds_to_bf16() {
        assert_eq!(embedding_scale(2816, true), 53.0);
        assert!((embedding_scale(2816, false) - 53.066).abs() < 1e-3);
    }

    #[test]
    fn only_cuda_serves_the_model() {
        let error = require_serving_backend(&Device::Cpu).unwrap_err();
        assert!(matches!(error, Error::InvalidInput(_)));
        assert_eq!(error.kind(), crate::error::ErrorKind::InvalidInput);
        // An unmarked directory is the published model: refused off CUDA.
        let dir = std::env::temp_dir().join(format!("izwi-sov-gate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(require_serving_backend_for(&dir, &Device::Cpu).is_err());
        let _ = std::fs::remove_dir_all(&dir);
        assert!(error.to_string().contains("needs an NVIDIA GPU (CUDA)"));
        assert_eq!(compute_dtype(&Device::Cpu).unwrap(), DType::F32);
    }
}
