//! Device selection for native inference with Metal optimizations.
//!
//! This module provides optimized device selection with Metal-specific improvements:
//! - Optimized dtype selection based on device capabilities
//! - Memory pool integration for reduced allocation overhead
//! - Unified memory awareness for Apple Silicon

use candle_core::{DType, Device};
use std::any::Any;
use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{debug, info, warn};

use super::types::{BackendKind, BackendPreference};
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::models::shared::memory::metal::{metal_pool_for_device, MetalMemoryPool};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Cuda,
    Metal,
    Cpu,
}

impl DeviceKind {
    pub fn is_cpu(&self) -> bool {
        matches!(self, DeviceKind::Cpu)
    }

    pub fn is_metal(&self) -> bool {
        matches!(self, DeviceKind::Metal)
    }

    pub fn is_cuda(&self) -> bool {
        matches!(self, DeviceKind::Cuda)
    }
}

impl From<DeviceKind> for BackendKind {
    fn from(kind: DeviceKind) -> Self {
        match kind {
            DeviceKind::Cpu => BackendKind::Cpu,
            DeviceKind::Metal => BackendKind::Metal,
            DeviceKind::Cuda => BackendKind::Cuda,
        }
    }
}

/// Device capabilities and optimization hints
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Whether the device prefers float32 (Metal on Apple Silicon)
    pub prefers_f32: bool,
    /// Whether the device supports bfloat16
    pub supports_bf16: bool,
    /// Whether the device supports float16 compute/storage for model activations
    pub supports_f16: bool,
    /// Whether the device has int8 tensor cores
    pub supports_int8_tensor_cores: bool,
    /// Whether the device has unified memory (Apple Silicon)
    pub has_unified_memory: bool,
    /// Recommended batch size for this device
    pub recommended_batch_size: usize,
    /// Available memory in bytes (if detectable)
    pub available_memory_bytes: Option<usize>,
    /// CUDA compute capability for NVIDIA devices
    pub cuda_compute_capability: Option<(u32, u32)>,
    /// CUDA device name when reported by the runtime
    pub cuda_device_name: Option<String>,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            prefers_f32: false,
            supports_bf16: false,
            supports_f16: false,
            supports_int8_tensor_cores: false,
            has_unified_memory: false,
            recommended_batch_size: 1,
            available_memory_bytes: None,
            cuda_compute_capability: None,
            cuda_device_name: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DTypeSelectionPolicy {
    Default,
    PreferF32,
    PreferF16,
    PreferBf16,
}

impl Default for DTypeSelectionPolicy {
    fn default() -> Self {
        Self::Default
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DTypeSelectionRequest<'a> {
    pub requested: Option<&'a str>,
    pub model_family: Option<ModelFamily>,
    pub checkpoint_dtype: Option<DType>,
    pub quantized: bool,
    pub policy: DTypeSelectionPolicy,
}

impl<'a> DTypeSelectionRequest<'a> {
    pub fn new(requested: Option<&'a str>) -> Self {
        Self {
            requested,
            model_family: None,
            checkpoint_dtype: None,
            quantized: false,
            policy: DTypeSelectionPolicy::Default,
        }
    }

    pub fn with_model_family(mut self, model_family: ModelFamily) -> Self {
        self.model_family = Some(model_family);
        self
    }

    pub fn with_checkpoint_dtype(mut self, checkpoint_dtype: DType) -> Self {
        self.checkpoint_dtype = Some(checkpoint_dtype);
        self
    }

    pub fn with_quantized(mut self, quantized: bool) -> Self {
        self.quantized = quantized;
        self
    }

    pub fn with_policy(mut self, policy: DTypeSelectionPolicy) -> Self {
        self.policy = policy;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DTypeSelection {
    pub dtype: DType,
    pub reason: Cow<'static, str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DTypeSelectionError {
    pub requested: String,
    pub reason: Cow<'static, str>,
}

impl std::fmt::Display for DTypeSelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unsupported dtype override {:?}: {}",
            self.requested, self.reason
        )
    }
}

impl std::error::Error for DTypeSelectionError {}

#[derive(Debug, Clone)]
pub struct DeviceProfile {
    pub device: Device,
    pub kind: DeviceKind,
    pub capabilities: DeviceCapabilities,
    /// Optional memory pool for this device (Metal only)
    pub memory_pool: Option<Arc<MetalMemoryPool>>,
}

impl DeviceProfile {
    pub fn cpu() -> Self {
        Self {
            device: Device::Cpu,
            kind: DeviceKind::Cpu,
            capabilities: DeviceCapabilities::default(),
            memory_pool: None,
        }
    }

    /// Select optimal dtype based on device kind and requested preference
    ///
    /// # Optimization Notes:
    /// - Metal on Apple Silicon: F32 is preferred over F16/BF16 for better performance
    ///   and numerical stability. Apple's GPU architecture doesn't benefit from F16
    ///   the same way NVIDIA GPUs do with Tensor Cores.
    /// - CUDA: BF16 is preferred for compute, F16 for memory-constrained scenarios
    /// - CPU: Always F32 since most CPUs don't have efficient F16/BF16 paths
    pub fn select_dtype(&self, requested: Option<&str>) -> DType {
        let request = DTypeSelectionRequest::new(requested);
        let selection = match self.try_resolve_dtype(request) {
            Ok(selection) => selection,
            Err(err) => {
                warn!(
                    "Falling back from unsupported dtype override on {:?}: {}",
                    self.kind, err
                );
                self.resolve_dtype(DTypeSelectionRequest::new(None))
            }
        };

        debug!(
            "Selected dtype {:?} for device {:?} (requested: {:?}, reason: {})",
            selection.dtype, self.kind, requested, selection.reason
        );

        selection.dtype
    }

    pub fn try_select_dtype(
        &self,
        requested: Option<&str>,
    ) -> std::result::Result<DType, DTypeSelectionError> {
        self.try_resolve_dtype(DTypeSelectionRequest::new(requested))
            .map(|selection| selection.dtype)
    }

    pub fn select_model_dtype(&self, model_family: ModelFamily, requested: Option<&str>) -> DType {
        let request = DTypeSelectionRequest::new(requested).with_model_family(model_family);
        let selection = match self.try_resolve_dtype(request) {
            Ok(selection) => selection,
            Err(err) => {
                warn!(
                    "Falling back from unsupported dtype override for {:?} on {:?}: {}",
                    model_family, self.kind, err
                );
                self.resolve_dtype(DTypeSelectionRequest::new(None).with_model_family(model_family))
            }
        };

        debug!(
            "Selected dtype {:?} for {:?} on {:?} (requested: {:?}, reason: {})",
            selection.dtype, model_family, self.kind, requested, selection.reason
        );

        selection.dtype
    }

    pub fn select_model_dtype_with_checkpoint(
        &self,
        model_family: ModelFamily,
        checkpoint_dtype: Option<DType>,
    ) -> DType {
        let mut request = DTypeSelectionRequest::new(None).with_model_family(model_family);
        if let Some(dtype) = checkpoint_dtype {
            request = request.with_checkpoint_dtype(dtype);
        }
        let selection = self.resolve_dtype(request);

        debug!(
            "Selected dtype {:?} for {:?} on {:?} (checkpoint: {:?}, reason: {})",
            selection.dtype, model_family, self.kind, checkpoint_dtype, selection.reason
        );

        selection.dtype
    }

    pub fn try_select_model_dtype(
        &self,
        model_family: ModelFamily,
        requested: Option<&str>,
    ) -> std::result::Result<DType, DTypeSelectionError> {
        self.try_resolve_dtype(
            DTypeSelectionRequest::new(requested).with_model_family(model_family),
        )
        .map(|selection| selection.dtype)
    }

    pub fn select_model_dtype_checked(
        &self,
        model_family: ModelFamily,
        requested: Option<&str>,
        context: &str,
    ) -> Result<DType> {
        let requested = requested.map(str::trim).filter(|raw| !raw.is_empty());
        if self.kind.is_cuda() && requested.is_some() {
            return self
                .try_select_model_dtype(model_family, requested)
                .map_err(|err| {
                    Error::InvalidInput(format!("Invalid CUDA {context} dtype override: {err}"))
                });
        }

        Ok(self.select_model_dtype(model_family, requested))
    }

    pub fn resolve_dtype(&self, request: DTypeSelectionRequest<'_>) -> DTypeSelection {
        self.try_resolve_dtype(request)
            .unwrap_or_else(|_| self.default_dtype_selection(request))
    }

    pub fn try_resolve_dtype(
        &self,
        request: DTypeSelectionRequest<'_>,
    ) -> std::result::Result<DTypeSelection, DTypeSelectionError> {
        if let Some(raw) = request.requested {
            let raw = raw.trim();
            if !raw.is_empty() {
                if let Some(dtype) = parse_dtype_name(raw) {
                    return self.resolve_requested_dtype(raw, dtype);
                }

                return Err(DTypeSelectionError {
                    requested: raw.to_string(),
                    reason: "expected one of f32, f16, or bf16".into(),
                });
            }
        }

        Ok(self.default_dtype_selection(request))
    }

    fn resolve_requested_dtype(
        &self,
        raw: &str,
        requested: DType,
    ) -> std::result::Result<DTypeSelection, DTypeSelectionError> {
        match self.kind {
            DeviceKind::Cpu => Ok(DTypeSelection {
                dtype: DType::F32,
                reason: "CPU inference uses F32 regardless of lower-precision override".into(),
            }),
            DeviceKind::Metal => Ok(DTypeSelection {
                dtype: DType::F32,
                reason: "Metal policy keeps F32 for existing stability/performance behavior".into(),
            }),
            DeviceKind::Cuda => match requested {
                DType::F32 => Ok(DTypeSelection {
                    dtype: DType::F32,
                    reason: "explicit CUDA F32 override".into(),
                }),
                DType::F16 if self.capabilities.supports_f16 => Ok(DTypeSelection {
                    dtype: DType::F16,
                    reason: "explicit CUDA F16 override".into(),
                }),
                DType::BF16 if self.capabilities.supports_bf16 => Ok(DTypeSelection {
                    dtype: DType::BF16,
                    reason: "explicit CUDA BF16 override supported by device capability".into(),
                }),
                DType::BF16 => Err(DTypeSelectionError {
                    requested: raw.to_string(),
                    reason: "CUDA device does not report BF16 support".into(),
                }),
                DType::F16 => Err(DTypeSelectionError {
                    requested: raw.to_string(),
                    reason: "CUDA device does not report F16 support".into(),
                }),
                _ => Err(DTypeSelectionError {
                    requested: raw.to_string(),
                    reason: "dtype override is not supported for CUDA inference".into(),
                }),
            },
        }
    }

    fn default_dtype_selection(&self, request: DTypeSelectionRequest<'_>) -> DTypeSelection {
        match self.kind {
            DeviceKind::Cpu => DTypeSelection {
                dtype: DType::F32,
                reason: "CPU default dtype is F32".into(),
            },
            DeviceKind::Metal => DTypeSelection {
                dtype: DType::F32,
                reason: "Metal default dtype remains F32".into(),
            },
            DeviceKind::Cuda => self.default_cuda_dtype_selection(request),
        }
    }

    fn default_cuda_dtype_selection(&self, request: DTypeSelectionRequest<'_>) -> DTypeSelection {
        match request.policy {
            DTypeSelectionPolicy::PreferF32 => {
                return DTypeSelection {
                    dtype: DType::F32,
                    reason: "model policy prefers CUDA F32".into(),
                };
            }
            DTypeSelectionPolicy::PreferF16 if self.capabilities.supports_f16 => {
                return DTypeSelection {
                    dtype: DType::F16,
                    reason: "model policy prefers CUDA F16".into(),
                };
            }
            DTypeSelectionPolicy::PreferBf16 if self.capabilities.supports_bf16 => {
                return DTypeSelection {
                    dtype: DType::BF16,
                    reason: "model policy prefers CUDA BF16 and device supports it".into(),
                };
            }
            _ => {}
        }

        if let Some(selection) = self.cuda_model_family_dtype_selection(request) {
            return selection;
        }

        if request.checkpoint_dtype == Some(DType::F32) {
            return DTypeSelection {
                dtype: DType::F32,
                reason: "checkpoint dtype is F32".into(),
            };
        }

        if request.checkpoint_dtype == Some(DType::BF16) && self.capabilities.supports_bf16 {
            return DTypeSelection {
                dtype: DType::BF16,
                reason: "checkpoint dtype is BF16 and CUDA device supports BF16".into(),
            };
        }

        if request.checkpoint_dtype == Some(DType::F16) && self.capabilities.supports_f16 {
            return DTypeSelection {
                dtype: DType::F16,
                reason: "checkpoint dtype is F16 and CUDA device supports F16".into(),
            };
        }

        if self.capabilities.supports_bf16 {
            DTypeSelection {
                dtype: DType::BF16,
                reason: "CUDA default uses BF16 on devices with BF16 support".into(),
            }
        } else if self.capabilities.supports_f16 {
            DTypeSelection {
                dtype: DType::F16,
                reason: "CUDA default falls back to F16 when BF16 is unavailable".into(),
            }
        } else {
            DTypeSelection {
                dtype: DType::F32,
                reason:
                    "CUDA default falls back to F32 because no half precision support was reported"
                        .into(),
            }
        }
    }

    fn cuda_model_family_dtype_selection(
        &self,
        request: DTypeSelectionRequest<'_>,
    ) -> Option<DTypeSelection> {
        match request.model_family? {
            ModelFamily::WhisperAsr => {
                if self.capabilities.supports_f16 {
                    Some(DTypeSelection {
                        dtype: DType::F16,
                        reason: "Whisper CUDA policy defaults to F16".into(),
                    })
                } else {
                    Some(DTypeSelection {
                        dtype: DType::F32,
                        reason: "Whisper CUDA policy falls back to F32 without F16 support".into(),
                    })
                }
            }
            ModelFamily::KokoroTts
            | ModelFamily::ParakeetAsr
            | ModelFamily::SortformerDiarization => Some(DTypeSelection {
                dtype: DType::F32,
                reason: "model family policy keeps CUDA default in F32".into(),
            }),
            _ => None,
        }
    }

    /// Get the optimal dtype for this device without any specific request
    pub fn optimal_dtype(&self) -> DType {
        self.select_dtype(None)
    }

    /// Check if this device supports memory pooling (Metal only)
    pub fn supports_memory_pool(&self) -> bool {
        self.kind.is_metal() && self.memory_pool.is_some()
    }

    /// Get memory pool statistics if available
    pub fn memory_pool_stats(
        &self,
    ) -> Option<crate::models::shared::memory::metal::MetalPoolStats> {
        self.memory_pool.as_ref().map(|pool| pool.stats())
    }

    /// Returns true if the device has unified memory architecture (Apple Silicon)
    pub fn has_unified_memory(&self) -> bool {
        self.capabilities.has_unified_memory
    }
}

pub struct DeviceSelector;

static METAL_PROBE_PANICKED: AtomicBool = AtomicBool::new(false);

fn panic_payload_to_string(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "unknown panic payload".to_string()
}

impl DeviceSelector {
    fn try_metal() -> Option<DeviceProfile> {
        if METAL_PROBE_PANICKED.load(Ordering::Relaxed) {
            return None;
        }

        let device = match std::panic::catch_unwind(|| Device::metal_if_available(0)) {
            Ok(Ok(device)) => device,
            Ok(Err(_)) => return None,
            Err(payload) => {
                METAL_PROBE_PANICKED.store(true, Ordering::Relaxed);
                warn!(
                    "Metal probe panicked; disabling Metal for this process: {}",
                    panic_payload_to_string(payload.as_ref())
                );
                return None;
            }
        };
        if device.is_metal() {
            // Initialize memory pool for Metal
            let memory_pool = metal_pool_for_device(&device);

            if memory_pool.is_some() {
                info!("Metal memory pool initialized");
            }

            Some(DeviceProfile {
                device,
                kind: DeviceKind::Metal,
                capabilities: DeviceCapabilities {
                    prefers_f32: true,    // Metal on Apple Silicon prefers F32
                    supports_bf16: false, // Metal doesn't have good BF16 support
                    supports_f16: false,  // Current Metal policy intentionally uses F32
                    supports_int8_tensor_cores: false,
                    has_unified_memory: true, // Apple Silicon has unified memory
                    recommended_batch_size: 4, // Conservative for unified memory
                    available_memory_bytes: None, // Could be detected via system APIs
                    cuda_compute_capability: None,
                    cuda_device_name: None,
                },
                memory_pool,
            })
        } else {
            None
        }
    }

    fn try_cuda() -> Option<DeviceProfile> {
        let device = std::panic::catch_unwind(|| Device::cuda_if_available(0))
            .ok()?
            .ok()?;
        if device.is_cuda() {
            let cuda_capabilities = Self::detect_cuda_capabilities(0);
            let supports_bf16 = cuda_capabilities
                .compute_capability
                .is_some_and(cuda_compute_capability_supports_bf16);
            let supports_f16 = cuda_capabilities
                .compute_capability
                .map_or(true, cuda_compute_capability_supports_f16);
            let supports_int8_tensor_cores = cuda_capabilities
                .compute_capability
                .is_some_and(cuda_compute_capability_supports_int8_tensor_cores);

            Some(DeviceProfile {
                device,
                kind: DeviceKind::Cuda,
                capabilities: DeviceCapabilities {
                    prefers_f32: false,
                    supports_bf16,
                    supports_f16,
                    supports_int8_tensor_cores,
                    has_unified_memory: false,
                    recommended_batch_size: 8, // CUDA can handle larger batches
                    available_memory_bytes: cuda_capabilities.total_memory_bytes,
                    cuda_compute_capability: cuda_capabilities.compute_capability,
                    cuda_device_name: cuda_capabilities.device_name,
                },
                memory_pool: None, // CUDA uses its own memory management
            })
        } else {
            None
        }
    }

    fn detect_cuda_capabilities(ordinal: usize) -> CudaProbe {
        detect_cuda_capabilities(ordinal)
    }

    pub fn detect() -> Result<DeviceProfile> {
        if cfg!(target_os = "macos") {
            if let Some(profile) = Self::try_metal() {
                info!(
                    "Using Metal device for inference (unified memory: {})",
                    profile.has_unified_memory()
                );
                return Ok(profile);
            }
        } else if let Some(profile) = Self::try_cuda() {
            info!("Using CUDA device for inference");
            return Ok(profile);
        }

        if let Some(profile) = Self::try_metal() {
            info!("Using Metal device for inference");
            return Ok(profile);
        }

        info!("Falling back to CPU for inference");
        Ok(DeviceProfile::cpu())
    }

    pub fn detect_for_preference(preference: BackendPreference) -> Result<DeviceProfile> {
        match preference {
            BackendPreference::Auto => Self::detect(),
            BackendPreference::Cpu => Ok(DeviceProfile::cpu()),
            BackendPreference::Metal => {
                if let Some(profile) = Self::try_metal() {
                    Ok(profile)
                } else {
                    Self::detect()
                }
            }
            BackendPreference::Cuda => {
                if cfg!(target_os = "macos") {
                    return Self::detect();
                }
                if let Some(profile) = Self::try_cuda() {
                    Ok(profile)
                } else {
                    Self::detect()
                }
            }
        }
    }

    pub fn detect_with_preference(preference: Option<&str>) -> Result<DeviceProfile> {
        preference
            .and_then(BackendPreference::parse)
            .map(Self::detect_for_preference)
            .unwrap_or_else(Self::detect)
    }
}

#[derive(Debug, Clone, Default)]
struct CudaProbe {
    compute_capability: Option<(u32, u32)>,
    total_memory_bytes: Option<usize>,
    device_name: Option<String>,
}

pub fn parse_dtype_name(raw: &str) -> Option<DType> {
    let normalized = raw.trim().to_ascii_lowercase();
    let normalized = normalized
        .strip_prefix("torch.")
        .unwrap_or(normalized.as_str());
    match normalized {
        "bfloat16" | "bf16" => Some(DType::BF16),
        "float16" | "f16" | "fp16" | "half" => Some(DType::F16),
        "float32" | "float" | "f32" | "fp32" => Some(DType::F32),
        _ => None,
    }
}

pub fn cuda_compute_capability_supports_bf16((major, _minor): (u32, u32)) -> bool {
    major >= 8
}

pub fn cuda_compute_capability_supports_f16((major, minor): (u32, u32)) -> bool {
    major > 5 || (major == 5 && minor >= 3)
}

pub fn cuda_compute_capability_supports_int8_tensor_cores((major, minor): (u32, u32)) -> bool {
    major > 7 || (major == 7 && minor >= 5)
}

#[cfg(feature = "cuda")]
fn detect_cuda_capabilities(ordinal: usize) -> CudaProbe {
    use candle_core::cuda_backend::cudarc::driver::{result, CudaContext};

    match CudaContext::new(ordinal) {
        Ok(context) => {
            let compute_capability = context
                .compute_capability()
                .ok()
                .map(|(major, minor)| (major.max(0) as u32, minor.max(0) as u32));
            let total_memory_bytes = unsafe { result::device::total_mem(context.cu_device()).ok() };
            let device_name = context.name().ok().map(|name| name.trim().to_string());

            CudaProbe {
                compute_capability,
                total_memory_bytes,
                device_name: device_name.filter(|name| !name.is_empty()),
            }
        }
        Err(err) => {
            warn!("Unable to query CUDA device properties: {err}");
            CudaProbe::default()
        }
    }
}

#[cfg(not(feature = "cuda"))]
fn detect_cuda_capabilities(_ordinal: usize) -> CudaProbe {
    CudaProbe::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_with_cpu_preference_returns_cpu() {
        let profile = DeviceSelector::detect_with_preference(Some("cpu")).unwrap();
        assert_eq!(profile.kind, DeviceKind::Cpu);
        assert!(profile.device.is_cpu());
        assert!(!profile.has_unified_memory());
    }

    #[test]
    fn test_detect_for_cpu_preference_returns_cpu() {
        let profile = DeviceSelector::detect_for_preference(BackendPreference::Cpu).unwrap();
        assert_eq!(profile.kind, DeviceKind::Cpu);
        assert!(profile.device.is_cpu());
    }

    #[test]
    fn test_detect_kind_matches_device() {
        let profile = DeviceSelector::detect().unwrap();
        match profile.kind {
            DeviceKind::Cpu => assert!(profile.device.is_cpu()),
            DeviceKind::Metal => {
                assert!(profile.device.is_metal());
                assert!(profile.has_unified_memory());
                assert!(profile.capabilities.prefers_f32);
            }
            DeviceKind::Cuda => assert!(profile.device.is_cuda()),
        }
    }

    #[test]
    fn test_device_kind_maps_to_backend_kind() {
        assert_eq!(BackendKind::from(DeviceKind::Cpu), BackendKind::Cpu);
        assert_eq!(BackendKind::from(DeviceKind::Metal), BackendKind::Metal);
        assert_eq!(BackendKind::from(DeviceKind::Cuda), BackendKind::Cuda);
    }

    #[test]
    fn test_metal_prefers_f32() {
        // Test that Metal devices prefer F32
        let metal_profile = DeviceProfile {
            device: Device::Cpu, // Use CPU for testing
            kind: DeviceKind::Metal,
            capabilities: DeviceCapabilities {
                prefers_f32: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        // Default should be F32 for Metal
        assert_eq!(metal_profile.select_dtype(None), DType::F32);

        // Explicit BF16 request should still give F32 for Metal
        assert_eq!(metal_profile.select_dtype(Some("bf16")), DType::F32);

        // Explicit F16 request should give F32 for Metal
        assert_eq!(metal_profile.select_dtype(Some("f16")), DType::F32);

        // F32 request should give F32
        assert_eq!(metal_profile.select_dtype(Some("f32")), DType::F32);
    }

    #[test]
    fn test_cuda_dtype_selection() {
        let cuda_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: DeviceCapabilities {
                supports_bf16: true,
                supports_f16: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        // CUDA should prefer BF16 by default
        assert_eq!(cuda_profile.select_dtype(None), DType::BF16);

        // Explicit requests should be respected
        assert_eq!(cuda_profile.select_dtype(Some("f32")), DType::F32);
        assert_eq!(cuda_profile.select_dtype(Some("f16")), DType::F16);
        assert_eq!(cuda_profile.select_dtype(Some("bf16")), DType::BF16);
    }

    #[test]
    fn test_cuda_without_bf16_defaults_to_f16() {
        let cuda_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: DeviceCapabilities {
                supports_bf16: false,
                supports_f16: true,
                cuda_compute_capability: Some((7, 5)),
                ..Default::default()
            },
            memory_pool: None,
        };

        assert_eq!(cuda_profile.select_dtype(None), DType::F16);
        assert_eq!(
            cuda_profile
                .try_select_dtype(Some("bf16"))
                .unwrap_err()
                .requested,
            "bf16"
        );
    }

    #[test]
    fn cuda_model_family_defaults_are_specific() {
        let cuda_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: DeviceCapabilities {
                supports_bf16: true,
                supports_f16: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        assert_eq!(
            cuda_profile.select_model_dtype(ModelFamily::WhisperAsr, None),
            DType::F16
        );
        assert_eq!(
            cuda_profile.select_model_dtype(ModelFamily::KokoroTts, None),
            DType::F32
        );
        assert_eq!(
            cuda_profile.select_model_dtype(ModelFamily::ParakeetAsr, None),
            DType::F32
        );
        assert_eq!(
            cuda_profile.select_model_dtype(ModelFamily::SortformerDiarization, None),
            DType::F32
        );
        assert_eq!(
            cuda_profile.select_model_dtype(ModelFamily::Qwen3Chat, None),
            DType::BF16
        );
        assert_eq!(
            cuda_profile.select_model_dtype(ModelFamily::Qwen3Tts, None),
            DType::BF16
        );
        assert_eq!(
            cuda_profile.select_model_dtype(ModelFamily::Gemma3Chat, None),
            DType::BF16
        );
        assert_eq!(
            cuda_profile.select_model_dtype(ModelFamily::Voxtral, None),
            DType::BF16
        );
    }

    #[test]
    fn parse_dtype_name_handles_config_aliases() {
        assert_eq!(parse_dtype_name("torch.bfloat16"), Some(DType::BF16));
        assert_eq!(parse_dtype_name("torch.float16"), Some(DType::F16));
        assert_eq!(parse_dtype_name("fp16"), Some(DType::F16));
        assert_eq!(parse_dtype_name("float"), Some(DType::F32));
        assert_eq!(parse_dtype_name("float8"), None);
    }

    #[test]
    fn cuda_dense_checkpoint_dtype_beats_generic_bf16_default() {
        let cuda_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: DeviceCapabilities {
                supports_bf16: true,
                supports_f16: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        for family in [
            ModelFamily::Qwen3Chat,
            ModelFamily::Gemma3Chat,
            ModelFamily::Voxtral,
        ] {
            assert_eq!(
                cuda_profile.select_model_dtype_with_checkpoint(family, Some(DType::F16)),
                DType::F16
            );
            assert_eq!(
                cuda_profile.select_model_dtype_with_checkpoint(family, Some(DType::F32)),
                DType::F32
            );
        }
    }

    #[test]
    fn cuda_quantized_checkpoint_dtype_respects_capabilities() {
        let cuda_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: DeviceCapabilities {
                supports_bf16: false,
                supports_f16: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        let selection = cuda_profile.resolve_dtype(
            DTypeSelectionRequest::new(None)
                .with_model_family(ModelFamily::Qwen3Asr)
                .with_checkpoint_dtype(DType::BF16)
                .with_quantized(true),
        );
        assert_eq!(selection.dtype, DType::F16);

        let selection = cuda_profile.resolve_dtype(
            DTypeSelectionRequest::new(None)
                .with_model_family(ModelFamily::Qwen3Asr)
                .with_checkpoint_dtype(DType::F32)
                .with_quantized(true),
        );
        assert_eq!(selection.dtype, DType::F32);
    }

    #[test]
    fn cuda_checked_model_dtype_rejects_bad_overrides() {
        let cuda_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: DeviceCapabilities {
                supports_bf16: false,
                supports_f16: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        let err = cuda_profile
            .select_model_dtype_checked(ModelFamily::Qwen3Chat, Some("bf16"), "Qwen3 chat")
            .unwrap_err();
        assert!(err.to_string().contains("Invalid CUDA Qwen3 chat"));

        let err = cuda_profile
            .select_model_dtype_checked(ModelFamily::Qwen3Chat, Some("float8"), "Qwen3 chat")
            .unwrap_err();
        assert!(err.to_string().contains("expected one of"));
    }

    #[test]
    fn non_cuda_checked_model_dtype_keeps_legacy_fallbacks() {
        let cpu_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cpu,
            capabilities: DeviceCapabilities::default(),
            memory_pool: None,
        };
        let metal_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Metal,
            capabilities: DeviceCapabilities {
                prefers_f32: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        assert_eq!(
            cpu_profile
                .select_model_dtype_checked(ModelFamily::Qwen3Chat, Some("float8"), "Qwen3 chat")
                .unwrap(),
            DType::F32
        );
        assert_eq!(
            metal_profile
                .select_model_dtype_checked(ModelFamily::Qwen3Chat, Some("bf16"), "Qwen3 chat")
                .unwrap(),
            DType::F32
        );
    }

    #[test]
    fn cpu_and_metal_whisper_model_dtype_stays_f32() {
        let cpu_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cpu,
            capabilities: DeviceCapabilities::default(),
            memory_pool: None,
        };
        let metal_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Metal,
            capabilities: DeviceCapabilities {
                prefers_f32: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        assert_eq!(
            cpu_profile.select_model_dtype(ModelFamily::WhisperAsr, None),
            DType::F32
        );
        assert_eq!(
            cpu_profile
                .select_model_dtype_checked(ModelFamily::WhisperAsr, Some("f16"), "Whisper")
                .unwrap(),
            DType::F32
        );
        assert_eq!(
            metal_profile.select_model_dtype(ModelFamily::WhisperAsr, None),
            DType::F32
        );
        assert_eq!(
            metal_profile
                .select_model_dtype_checked(ModelFamily::WhisperAsr, Some("bf16"), "Whisper")
                .unwrap(),
            DType::F32
        );
    }

    #[test]
    fn cuda_compute_capability_gates_bf16() {
        assert!(!cuda_compute_capability_supports_bf16((7, 5)));
        assert!(cuda_compute_capability_supports_bf16((8, 0)));
        assert!(cuda_compute_capability_supports_bf16((9, 0)));
    }

    #[test]
    fn cuda_compute_capability_gates_int8_tensor_cores() {
        assert!(!cuda_compute_capability_supports_int8_tensor_cores((7, 0)));
        assert!(cuda_compute_capability_supports_int8_tensor_cores((7, 5)));
        assert!(cuda_compute_capability_supports_int8_tensor_cores((8, 0)));
    }

    #[test]
    fn test_cpu_always_f32() {
        let cpu_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cpu,
            capabilities: DeviceCapabilities::default(),
            memory_pool: None,
        };

        // CPU should always use F32 regardless of request
        assert_eq!(cpu_profile.select_dtype(None), DType::F32);
        assert_eq!(cpu_profile.select_dtype(Some("bf16")), DType::F32);
        assert_eq!(cpu_profile.select_dtype(Some("f16")), DType::F32);
        assert_eq!(cpu_profile.select_dtype(Some("f32")), DType::F32);
    }
}
