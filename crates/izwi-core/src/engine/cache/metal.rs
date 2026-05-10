//! Metal-specific KV cache optimizations.
//!
//! This module provides optimizations for KV cache management on Metal/Apple Silicon:
//! - Unified memory awareness (no CPU/GPU transfers needed)
//! - Memory pressure monitoring
//! - Efficient residency management for shared memory architecture
//! - Integration with Metal memory pools

use super::kv::{KVCacheConfig, KVCacheManager, KVCacheStats};
use crate::backends::DeviceProfile;
use crate::error::Result;
use tracing::{debug, info, warn};

/// Metal-optimized KV cache configuration
#[derive(Debug, Clone)]
pub struct MetalKVCacheConfig {
    /// Base KV cache configuration
    pub base_config: KVCacheConfig,
    /// Enable unified memory optimizations
    pub enable_unified_memory: bool,
    /// Memory pressure threshold (0.0 - 1.0)
    pub memory_pressure_threshold: f32,
    /// Automatic garbage collection interval (number of operations)
    pub gc_interval: usize,
    /// Prefetch blocks ahead of current position
    pub prefetch_blocks: usize,
    /// Enable aggressive memory compaction under pressure
    pub aggressive_compaction: bool,
}

impl Default for MetalKVCacheConfig {
    fn default() -> Self {
        let base_config = KVCacheConfig {
            // Metal performs better with F32, so adjust dtype_bytes
            dtype_bytes: 4, // F32 instead of F16
            ..Default::default()
        };

        Self {
            base_config,
            enable_unified_memory: true,
            memory_pressure_threshold: 0.80,
            gc_interval: 100,
            prefetch_blocks: 2,
            aggressive_compaction: true,
        }
    }
}

/// Memory pressure level for the system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressure {
    /// Normal memory usage
    Normal,
    /// Warning level - should start reducing memory
    Warning,
    /// Critical level - must free memory immediately
    Critical,
}

impl MemoryPressure {
    /// Check if we should trigger garbage collection
    pub fn should_gc(&self) -> bool {
        matches!(self, MemoryPressure::Warning | MemoryPressure::Critical)
    }

    /// Check if we should use aggressive compaction
    pub fn should_compact(&self) -> bool {
        matches!(self, MemoryPressure::Critical)
    }
}

/// Metal-optimized KV cache manager wrapper
///
/// This wrapper adds Metal-specific optimizations on top of the base KVCacheManager:
/// - Monitors unified memory usage
/// - Handles memory pressure events
/// - Optimizes for shared memory architecture
pub struct MetalKVCacheManager {
    /// Base KV cache manager
    pub inner: KVCacheManager,
    /// Metal-specific configuration
    pub config: MetalKVCacheConfig,
    /// Device profile for memory queries
    device_profile: DeviceProfile,
    /// Operation counter for periodic maintenance
    operation_count: usize,
    /// Current memory pressure level
    memory_pressure: MemoryPressure,
    /// Peak memory usage seen (bytes)
    peak_memory_bytes: usize,
    /// Number of memory pressure events handled
    pressure_events: u64,
}

impl MetalKVCacheManager {
    /// Create a new Metal-optimized KV cache manager
    pub fn new(config: MetalKVCacheConfig, device_profile: DeviceProfile) -> Result<Self> {
        if !device_profile.has_unified_memory() {
            warn!("MetalKVCacheManager created without unified memory device");
        }

        let inner = KVCacheManager::new(config.base_config.clone());

        info!(
            "Initialized MetalKVCacheManager with {}MB capacity, unified_memory={}",
            config.base_config.total_memory_bytes() / (1024 * 1024),
            device_profile.has_unified_memory()
        );

        Ok(Self {
            inner,
            config,
            device_profile,
            operation_count: 0,
            memory_pressure: MemoryPressure::Normal,
            peak_memory_bytes: 0,
            pressure_events: 0,
        })
    }

    /// Check and update memory pressure status
    pub fn update_memory_pressure(&mut self) -> MemoryPressure {
        let stats = self.inner.stats();
        let usage_ratio = stats.utilization() as f32;

        let new_pressure = if usage_ratio > 0.95 {
            MemoryPressure::Critical
        } else if usage_ratio > self.config.memory_pressure_threshold {
            MemoryPressure::Warning
        } else {
            MemoryPressure::Normal
        };

        if new_pressure != self.memory_pressure {
            self.pressure_events += 1;
            info!(
                "Memory pressure changed: {:?} -> {:?} (usage: {:.1}%)",
                self.memory_pressure,
                new_pressure,
                usage_ratio * 100.0
            );
            self.memory_pressure = new_pressure;
        }

        self.memory_pressure
    }

    /// Perform periodic maintenance tasks
    pub fn maintenance(&mut self) -> Result<()> {
        self.operation_count += 1;

        // Check memory pressure
        let pressure = self.update_memory_pressure();

        // Handle memory pressure
        if pressure.should_gc() {
            self.handle_memory_pressure()?;
        }

        // Periodic maintenance every N operations
        if self.operation_count % self.config.gc_interval == 0 {
            self.periodic_maintenance()?;
        }

        // Update peak memory tracking
        let stats = self.inner.stats();
        self.peak_memory_bytes = self.peak_memory_bytes.max(stats.memory_used_bytes);

        Ok(())
    }

    /// Handle memory pressure by freeing resources
    fn handle_memory_pressure(&mut self) -> Result<()> {
        debug!("Handling memory pressure: {:?}", self.memory_pressure);

        match self.memory_pressure {
            MemoryPressure::Warning => {
                // Reduce soft limit to encourage more aggressive eviction
                let current_soft = self.inner.soft_max_blocks();
                let new_soft = (current_soft as f32 * 0.9) as usize;
                self.inner.set_soft_max_blocks(new_soft);
                debug!("Reduced soft limit: {} -> {}", current_soft, new_soft);
            }
            MemoryPressure::Critical => {
                // Aggressive cleanup
                if self.config.aggressive_compaction {
                    // Compact the cache by removing unused shared prefixes
                    self.inner.compact_shared_prefixes();
                }

                // Further reduce soft limit
                let current_soft = self.inner.soft_max_blocks();
                let new_soft = (current_soft as f32 * 0.7) as usize;
                self.inner.set_soft_max_blocks(new_soft);
                warn!("Critical memory: reduced soft limit to {}", new_soft);
            }
            _ => {}
        }

        Ok(())
    }

    /// Periodic maintenance tasks
    fn periodic_maintenance(&mut self) -> Result<()> {
        debug!("Running periodic KV cache maintenance");

        // Reset soft limit to default if pressure is normal
        if self.memory_pressure == MemoryPressure::Normal {
            let max_blocks = self.config.base_config.max_blocks;
            let current_soft = self.inner.soft_max_blocks();
            if current_soft < max_blocks {
                let new_soft = (current_soft + max_blocks / 20).min(max_blocks);
                self.inner.set_soft_max_blocks(new_soft);
            }
        }

        Ok(())
    }

    /// Get optimized stats with Metal-specific metrics
    pub fn metal_stats(&self) -> MetalKVCacheStats {
        let base_stats = self.inner.stats();

        MetalKVCacheStats {
            base_stats,
            memory_pressure: self.memory_pressure,
            peak_memory_bytes: self.peak_memory_bytes,
            pressure_events: self.pressure_events,
            unified_memory_enabled: self.config.enable_unified_memory,
            operation_count: self.operation_count,
        }
    }

    /// Reset peak memory tracking
    pub fn reset_peak_memory(&mut self) {
        self.peak_memory_bytes = 0;
    }

    /// Check if prefetching should be enabled for a request
    pub fn should_prefetch(&self) -> bool {
        // Don't prefetch under memory pressure
        self.memory_pressure == MemoryPressure::Normal
    }

    /// Get recommended batch size based on memory pressure
    pub fn recommended_batch_size(&self, base_batch_size: usize) -> usize {
        match self.memory_pressure {
            MemoryPressure::Normal => base_batch_size,
            MemoryPressure::Warning => base_batch_size / 2,
            MemoryPressure::Critical => 1,
        }
    }
}

/// Metal-specific KV cache statistics
#[derive(Debug, Clone)]
pub struct MetalKVCacheStats {
    /// Base KV cache statistics
    pub base_stats: KVCacheStats,
    /// Current memory pressure level
    pub memory_pressure: MemoryPressure,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Number of memory pressure events
    pub pressure_events: u64,
    /// Whether unified memory optimizations are enabled
    pub unified_memory_enabled: bool,
    /// Operation count since creation
    pub operation_count: usize,
}

/// Helper functions for Metal memory management
pub mod utils {
    use super::*;

    /// Calculate optimal block size for Metal devices
    ///
    /// Metal benefits from larger blocks due to unified memory
    pub fn optimal_block_size(seq_len: usize, _num_layers: usize) -> usize {
        // Metal can handle larger blocks efficiently
        if seq_len <= 128 {
            16
        } else if seq_len <= 1024 {
            32
        } else {
            64
        }
    }

    /// Estimate memory usage for a given sequence configuration
    pub fn estimate_memory_bytes(
        num_sequences: usize,
        seq_len: usize,
        config: &KVCacheConfig,
    ) -> usize {
        let blocks_per_seq = config.blocks_for_tokens(seq_len);
        let total_blocks = num_sequences * blocks_per_seq;
        total_blocks * config.block_memory_bytes()
    }

    /// Check if configuration is suitable for Metal
    pub fn is_suitable_for_metal(config: &KVCacheConfig) -> bool {
        // Metal works best with F32 and larger block sizes
        config.dtype_bytes == 4 && config.block_size >= 16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::{DeviceCapabilities, DeviceKind};
    use candle_core::Device;

    #[test]
    fn test_memory_pressure_transitions() {
        let device_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Metal,
            capabilities: DeviceCapabilities {
                has_unified_memory: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        let config = MetalKVCacheConfig::default();
        let mut manager = MetalKVCacheManager::new(config, device_profile).unwrap();

        // Initially normal
        assert_eq!(manager.memory_pressure, MemoryPressure::Normal);

        // Test pressure transitions
        manager.memory_pressure = MemoryPressure::Warning;
        assert!(manager.memory_pressure.should_gc());
        assert!(!manager.memory_pressure.should_compact());

        manager.memory_pressure = MemoryPressure::Critical;
        assert!(manager.memory_pressure.should_gc());
        assert!(manager.memory_pressure.should_compact());
    }

    #[test]
    fn test_recommended_batch_size() {
        let device_profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Metal,
            capabilities: DeviceCapabilities {
                has_unified_memory: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        let config = MetalKVCacheConfig::default();
        let mut manager = MetalKVCacheManager::new(config, device_profile).unwrap();

        let base_batch = 8;

        // Normal pressure - use full batch
        manager.memory_pressure = MemoryPressure::Normal;
        assert_eq!(manager.recommended_batch_size(base_batch), base_batch);

        // Warning pressure - halve batch
        manager.memory_pressure = MemoryPressure::Warning;
        assert_eq!(manager.recommended_batch_size(base_batch), base_batch / 2);

        // Critical pressure - batch of 1
        manager.memory_pressure = MemoryPressure::Critical;
        assert_eq!(manager.recommended_batch_size(base_batch), 1);
    }

    #[test]
    fn test_utils_optimal_block_size() {
        assert_eq!(utils::optimal_block_size(64, 24), 16);
        assert_eq!(utils::optimal_block_size(512, 24), 32);
        assert_eq!(utils::optimal_block_size(2048, 24), 64);
    }
}
