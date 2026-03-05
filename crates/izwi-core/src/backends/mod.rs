//! Backend routing, device probing, and execution policy primitives.

pub mod capabilities;
pub mod device;
pub mod policy;
pub mod router;
pub mod types;

pub use capabilities::BackendCapabilities;
pub use device::{DeviceCapabilities, DeviceKind, DeviceProfile, DeviceSelector};
pub use policy::{can_parallelize_requests, default_dtype_for_device, kv_dtype_bytes};
pub use router::{BackendPlan, BackendRouter};
pub use types::{BackendKind, BackendPreference, BackendSelectionSource, ExecutionBackend};
