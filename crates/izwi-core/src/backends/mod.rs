//! Backend routing, device probing, and execution policy primitives.

pub mod capabilities;
pub mod cuda_plan;
pub mod cuda_runtime;
pub mod device;
pub mod device_thread;
pub mod kv;
pub mod model_io;
pub mod policy;
pub mod router;
pub(crate) mod state;
pub mod types;

pub use capabilities::BackendCapabilities;
pub use cuda_plan::{
    evaluate_cuda_plan_eligibility, resolve_cuda_execution_plan, CudaExecutionPlan,
    CudaPlanBlocker, CudaPlanEligibility, CudaRuntimeObservation,
};
pub use cuda_runtime::{
    prepend_cuda_loader_paths, private_cuda_runtime_active, private_cuda_runtime_binary_env_key,
    private_cuda_runtime_candidates, private_cuda_runtime_env_key, resolve_private_cuda_runtime,
    CudaRuntimeDiagnostics,
};
#[cfg(all(test, feature = "metal"))]
pub(crate) use device::metal_test_device_or_skip;
pub use device::{
    metal_apple_gpu_family, metal_bf16_checkpoint_lm_tier, metal_device_if_available,
    metal_runtime_supported, metal_supports_fast_bf16, parse_dtype_name, AppleGpuFamily,
    DTypeSelection, DTypeSelectionError, DTypeSelectionPolicy, DTypeSelectionRequest,
    DeviceCapabilities, DeviceKind, DeviceProfile, DeviceSelector, MetalDTypeOverride,
    METAL_DTYPE_ENV, METAL_PAGED_ATTENTION_SUPPORTS_BF16,
};
pub use device_thread::{
    device_ops_off_thread_total, probe_device_op, DeviceJobTag, DeviceScope, DeviceThread,
    DeviceThreadKey,
};
pub use model_io::{
    auto_gguf_mmap_for_backend, backend_kind_for_device, gguf_mmap_enabled,
    gguf_mmap_mode_from_env, open_gguf_reader, open_gguf_reader_with_mode, resolve_gguf_mmap_mode,
    GgufMmapMode, GgufReader, GgufReaderKind,
};
pub use policy::{can_parallelize_requests, default_dtype_for_device, kv_dtype_bytes};
pub use router::{BackendPlan, BackendRouter};
pub use types::{
    BackendContext, BackendKind, BackendPreference, BackendSelectionSource, ExecutionBackend,
    RuntimeDeviceAssignment,
};
