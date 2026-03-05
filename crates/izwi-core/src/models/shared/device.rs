//! Compatibility shim for legacy imports.
//!
//! Canonical backend/device implementation now lives under `crate::backends`.

#[allow(unused_imports)]
pub use crate::backends::device::{
    DeviceCapabilities, DeviceKind, DeviceProfile, DeviceSelector,
};
