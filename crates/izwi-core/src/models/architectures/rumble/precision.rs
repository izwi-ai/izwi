//! Which dtypes Rumble's components load at on each device, and the
//! load-time probes behind a lowered dtype.
//!
//! The checkpoint is BF16. CPU and Metal run both components in F32: F16
//! text on the shared Qwen3 trunk drifts on long audio on Metal. CUDA runs
//! the decoder in the device's half precision (BF16 where supported) and
//! the audio tower in F32: batched BF16 audio encoders made a row's
//! transcript depend on the clips beside it on the H100, and were slower.

use candle_core::DType;

use crate::backends::{DTypeSelection, DeviceKind, DeviceProfile};
use crate::catalog::ModelFamily;
use crate::error::Result;

/// Overrides the decoder dtype (`f32`, `f16`, `bf16`).
pub(crate) const RUMBLE_TEXT_DTYPE_ENV: &str = "IZWI_RUMBLE_TEXT_DTYPE";
/// Overrides the audio tower and projector dtype.
pub(crate) const RUMBLE_AUDIO_DTYPE_ENV: &str = "IZWI_RUMBLE_AUDIO_DTYPE";

fn env_override(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|raw| raw.trim().to_string())
        .filter(|raw| !raw.is_empty())
}

/// The decoder and audio dtypes selected on `device`, with the reason for
/// each. Load admission prices the checkpoint with the same selection.
pub(crate) fn rumble_weight_precisions(
    device: &DeviceProfile,
) -> Result<(DTypeSelection, DTypeSelection)> {
    let text = match env_override(RUMBLE_TEXT_DTYPE_ENV) {
        Some(raw) => DTypeSelection {
            dtype: device.select_model_dtype_checked(
                ModelFamily::RumbleAsr,
                Some(&raw),
                "Rumble text",
            )?,
            reason: format!("text dtype override {raw:?}").into(),
        },
        None => match device.kind {
            DeviceKind::Cuda => DTypeSelection {
                dtype: device.select_model_dtype(ModelFamily::RumbleAsr, None),
                reason: "CUDA runs the dense decoder in the device's half precision".into(),
            },
            DeviceKind::Metal => DTypeSelection {
                dtype: DType::F32,
                reason: "Metal keeps the dense decoder in F32".into(),
            },
            DeviceKind::Cpu => DTypeSelection {
                dtype: DType::F32,
                reason: "CPU runs the dense decoder in F32".into(),
            },
        },
    };
    let audio = match env_override(RUMBLE_AUDIO_DTYPE_ENV) {
        Some(raw) => DTypeSelection {
            dtype: device.select_model_dtype_checked(
                ModelFamily::RumbleAsr,
                Some(&raw),
                "Rumble audio",
            )?,
            reason: format!("audio dtype override {raw:?}").into(),
        },
        None => DTypeSelection {
            dtype: DType::F32,
            reason: "the audio tower and projector run in F32".into(),
        },
    };
    Ok((text, audio))
}

/// The dtypes alone, for load admission.
pub(crate) fn rumble_weight_dtypes(device: &DeviceProfile) -> Result<(DType, DType)> {
    rumble_weight_precisions(device).map(|(text, audio)| (text.dtype, audio.dtype))
}
