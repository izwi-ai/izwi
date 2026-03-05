//! Device selection for native inference with Metal optimizations.
//!
//! This module provides optimized device selection with Metal-specific improvements:
//! - Optimized dtype selection based on device capabilities
//! - Memory pool integration for reduced allocation overhead
//! - Unified memory awareness for Apple Silicon

use candle_core::{DType, Device};
use std::any::Any;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::error::Result;
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

/// Device capabilities and optimization hints
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Whether the device prefers float32 (Metal on Apple Silicon)
    pub prefers_f32: bool,
    /// Whether the device supports bfloat16
    pub supports_bf16: bool,
    /// Whether the device has unified memory (Apple Silicon)
    pub has_unified_memory: bool,
    /// Recommended batch size for this device
    pub recommended_batch_size: usize,
    /// Available memory in bytes (if detectable)
    pub available_memory_bytes: Option<usize>,
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            prefers_f32: false,
            supports_bf16: false,
            has_unified_memory: false,
            recommended_batch_size: 1,
            available_memory_bytes: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceProfile {
    pub device: Device,
    pub kind: DeviceKind,
    pub capabilities: DeviceCapabilities,
    /// Optional memory pool for this device (Metal only)
    pub memory_pool: Option<Arc<MetalMemoryPool>>,
}

impl DeviceProfile {
    /// Select optimal dtype based on device kind and requested preference
    ///
    /// # Optimization Notes:
    /// - Metal on Apple Silicon: F32 is preferred over F16/BF16 for better performance
    ///   and numerical stability. Apple's GPU architecture doesn't benefit from F16
    ///   the same way NVIDIA GPUs do with Tensor Cores.
    /// - CUDA: BF16 is preferred for compute, F16 for memory-constrained scenarios
    /// - CPU: Always F32 since most CPUs don't have efficient F16/BF16 paths
    pub fn select_dtype(&self, requested: Option<&str>) -> DType {
        let dtype = match requested.unwrap_or("") {
            "bfloat16" | "bf16" => match self.kind {
                DeviceKind::Cpu => DType::F32,
                DeviceKind::Metal => {
                    // Metal on Apple Silicon actually performs better with F32
                    // due to lack of Tensor Cores and better F32 optimization
                    debug!("Metal device: using F32 instead of BF16 for better performance");
                    DType::F32
                }
                DeviceKind::Cuda => {
                    if self.capabilities.supports_bf16 {
                        DType::BF16
                    } else {
                        DType::F32
                    }
                }
            },
            "float16" | "f16" => match self.kind {
                DeviceKind::Cpu => DType::F32,
                DeviceKind::Metal => {
                    // Same reasoning as BF16 - F32 is better on Metal
                    debug!("Metal device: using F32 instead of F16 for better performance");
                    DType::F32
                }
                _ => DType::F16,
            },
            "float32" | "f32" => DType::F32,
            // Default selection based on device optimization
            _ => match self.kind {
                DeviceKind::Cpu => DType::F32,
                DeviceKind::Metal => {
                    // Research shows F32 performs better than F16 on Apple Silicon
                    // due to unified memory architecture and lack of Tensor Cores
                    DType::F32
                }
                DeviceKind::Cuda => {
                    if self.capabilities.supports_bf16 {
                        DType::BF16
                    } else {
                        DType::F32
                    }
                }
            },
        };

        debug!(
            "Selected dtype {:?} for device {:?} (requested: {:?})",
            dtype, self.kind, requested
        );

        dtype
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
                    prefers_f32: true,            // Metal on Apple Silicon prefers F32
                    supports_bf16: false,         // Metal doesn't have good BF16 support
                    has_unified_memory: true,     // Apple Silicon has unified memory
                    recommended_batch_size: 4,    // Conservative for unified memory
                    available_memory_bytes: None, // Could be detected via system APIs
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
            // Detect CUDA capabilities
            let supports_bf16 = Self::detect_cuda_bf16_support();

            Some(DeviceProfile {
                device,
                kind: DeviceKind::Cuda,
                capabilities: DeviceCapabilities {
                    prefers_f32: false,
                    supports_bf16,
                    has_unified_memory: false,
                    recommended_batch_size: 8, // CUDA can handle larger batches
                    available_memory_bytes: Self::detect_cuda_memory(),
                },
                memory_pool: None, // CUDA uses its own memory management
            })
        } else {
            None
        }
    }

    fn detect_cuda_bf16_support() -> bool {
        // BF16 support requires Compute Capability 8.0+ (Ampere)
        // This is a simplified check - in production, query the device properties
        true // Assume modern CUDA supports BF16
    }

    fn detect_cuda_memory() -> Option<usize> {
        // Could use CUDA APIs to detect available memory
        None
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
        Ok(DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cpu,
            capabilities: DeviceCapabilities::default(),
            memory_pool: None,
        })
    }

    pub fn detect_with_preference(preference: Option<&str>) -> Result<DeviceProfile> {
        match preference.unwrap_or("") {
            "cuda" => {
                if cfg!(target_os = "macos") {
                    return Self::detect();
                }
                if let Some(profile) = Self::try_cuda() {
                    Ok(profile)
                } else {
                    Self::detect()
                }
            }
            "metal" | "mps" => {
                if let Some(profile) = Self::try_metal() {
                    Ok(profile)
                } else {
                    Self::detect()
                }
            }
            "cpu" => Ok(DeviceProfile {
                device: Device::Cpu,
                kind: DeviceKind::Cpu,
                capabilities: DeviceCapabilities::default(),
                memory_pool: None,
            }),
            _ => Self::detect(),
        }
    }
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
