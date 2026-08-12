//! Metal-specific memory management optimizations.
//!
//! This module provides memory pooling and management optimizations specifically
//! for Metal on Apple Silicon. Key features:
//! - Tensor buffer reuse to reduce allocation overhead
//! - Memory pressure monitoring for unified memory architecture
//! - Bucket-based allocation strategy for common tensor sizes
//! - Integration with system memory pressure callbacks

use candle_core::{DType, Device, Shape, Tensor};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use tracing::{debug, info};

use crate::error::{Error, Result};

/// Memory pool configuration for Metal devices
#[derive(Debug, Clone)]
pub struct MetalMemoryPoolConfig {
    /// Maximum memory pool size in bytes (default: 4GB)
    pub max_pool_size_bytes: usize,
    /// Minimum alignment (in bytes) required for pooled allocations
    pub bucket_increment: usize,
    /// Minimum tensor size to pool (smaller tensors are allocated directly)
    pub min_pool_size: usize,
    /// Maximum tensor size to pool (larger tensors are allocated directly)
    pub max_pool_size: usize,
    /// Enable memory pressure monitoring
    pub enable_pressure_monitoring: bool,
    /// Memory pressure threshold (0.0 - 1.0, default: 0.85)
    pub pressure_threshold: f32,
}

impl Default for MetalMemoryPoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size_bytes: 4 * 1024 * 1024 * 1024, // 4GB
            bucket_increment: 1,                         // No alignment requirement by default
            min_pool_size: 4096,                         // 4KB minimum
            max_pool_size: 256 * 1024 * 1024,            // 256MB maximum
            enable_pressure_monitoring: true,
            pressure_threshold: 0.85,
        }
    }
}

/// A pooled tensor buffer that can be reused
#[derive(Debug)]
struct PooledBuffer {
    tensor: Tensor,
    size_bytes: usize,
    last_used: std::time::Instant,
    use_count: u32,
}

/// Memory pool for Metal device tensors
///
/// This pool reduces allocation overhead by reusing tensor buffers
/// across inference calls. It's particularly effective for Metal
/// where allocation has higher overhead than CUDA.
pub struct MetalMemoryPool {
    config: MetalMemoryPoolConfig,
    device: Device,
    /// Buckets of available buffers indexed by element count
    buckets: Mutex<HashMap<usize, Vec<PooledBuffer>>>,
    /// Current total memory usage
    current_usage: AtomicUsize,
    /// Total allocations served from pool
    hits: AtomicUsize,
    /// Total allocations that missed pool
    misses: AtomicUsize,
    /// Number of buffers currently checked out
    checked_out: AtomicUsize,
}

impl std::fmt::Debug for MetalMemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalMemoryPool")
            .field("config", &self.config)
            .field("device", &self.device)
            .field("current_usage", &self.current_usage.load(Ordering::Relaxed))
            .field("hits", &self.hits.load(Ordering::Relaxed))
            .field("misses", &self.misses.load(Ordering::Relaxed))
            .field("checked_out", &self.checked_out.load(Ordering::Relaxed))
            .finish()
    }
}

impl MetalMemoryPool {
    /// Create a new memory pool for the given Metal device
    pub fn new(device: Device, config: MetalMemoryPoolConfig) -> Result<Self> {
        if !device.is_metal() {
            return Err(Error::InvalidInput(
                "MetalMemoryPool can only be used with Metal devices".to_string(),
            ));
        }

        info!(
            "Initializing MetalMemoryPool with {} MB capacity",
            config.max_pool_size_bytes / (1024 * 1024)
        );

        Ok(Self {
            config,
            device,
            buckets: Mutex::new(HashMap::new()),
            current_usage: AtomicUsize::new(0),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
            checked_out: AtomicUsize::new(0),
        })
    }

    /// Determine the pool key for a given tensor size.
    ///
    /// Returns element count when poolable; otherwise None.
    fn pool_key(&self, elem_count: usize, size_bytes: usize) -> Option<usize> {
        if size_bytes < self.config.min_pool_size {
            return None; // Don't pool very small tensors
        }
        if size_bytes > self.config.max_pool_size {
            return None; // Don't pool very large tensors
        }
        if self.config.bucket_increment > 1
            && !size_bytes.is_multiple_of(self.config.bucket_increment)
        {
            return None; // Enforce alignment if configured
        }

        Some(elem_count)
    }

    /// Acquire a tensor from the pool or allocate a new one
    ///
    /// The returned tensor will have the requested shape and dtype.
    /// Pooled tensors are reshaped when element counts match.
    pub fn acquire(&self, shape: &Shape, dtype: DType) -> Result<Tensor> {
        let element_size = dtype.size_in_bytes();
        let elem_count = shape.elem_count();
        let requested_size = elem_count * element_size;
        let pool_key = self.pool_key(elem_count, requested_size);

        // Don't pool if outside pooling range
        let pool_key = if let Some(pool_key) = pool_key {
            pool_key
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            let tensor = Tensor::zeros(shape.clone(), dtype, &self.device).map_err(Error::from)?;
            self.checked_out.fetch_add(1, Ordering::Relaxed);
            return Ok(tensor);
        };

        // Try to get from pool
        {
            let mut buckets = self.buckets.lock().unwrap();

            if let Some(buffers) = buckets.get_mut(&pool_key) {
                // Find a buffer with compatible dtype
                if let Some(idx) = buffers.iter().position(|buf| buf.tensor.dtype() == dtype) {
                    let mut buffer = buffers.remove(idx);
                    buffer.use_count += 1;
                    buffer.last_used = std::time::Instant::now();

                    self.checked_out.fetch_add(1, Ordering::Relaxed);
                    self.hits.fetch_add(1, Ordering::Relaxed);

                    debug!(
                        "Pool hit: elem_count={}, use_count={}",
                        elem_count, buffer.use_count
                    );

                    // Return the buffer's tensor, reshaped if needed
                    if buffer.tensor.shape() == shape {
                        return Ok(buffer.tensor);
                    }

                    return buffer.tensor.reshape(shape.clone()).map_err(Error::from);
                }
            }
        }

        // Pool miss - allocate new tensor
        self.misses.fetch_add(1, Ordering::Relaxed);

        let tensor = Tensor::zeros(shape.clone(), dtype, &self.device).map_err(Error::from)?;
        self.current_usage
            .fetch_add(requested_size, Ordering::Relaxed);
        self.checked_out.fetch_add(1, Ordering::Relaxed);

        debug!(
            "Pool miss: allocated elem_count={}, current_usage={}MB",
            elem_count,
            self.current_usage.load(Ordering::Relaxed) / (1024 * 1024)
        );

        Ok(tensor)
    }

    /// Return a tensor to the pool for reuse
    ///
    /// The tensor should be zeroed or considered uninitialized after return.
    pub fn release(&self, tensor: Tensor) {
        // Check if tensor is on the same device by comparing device types
        let tensor_device_type = format!("{:?}", tensor.device());
        let pool_device_type = format!("{:?}", &self.device);
        if tensor_device_type != pool_device_type {
            // Can't pool tensors from different devices
            return;
        }

        let elem_count = tensor.elem_count();
        let size_bytes = tensor.dtype().size_in_bytes() * elem_count;
        let pool_key = self.pool_key(elem_count, size_bytes);

        // Don't pool if outside pooling range
        let pool_key = if let Some(pool_key) = pool_key {
            pool_key
        } else {
            self.checked_out
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.checked_sub(1)
                })
                .ok();
            return;
        };

        // Check if we're under memory limit
        let current = self.current_usage.load(Ordering::Relaxed);
        if current > self.config.max_pool_size_bytes {
            debug!("Pool at capacity, dropping tensor");
            self.current_usage
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.checked_sub(size_bytes)
                })
                .ok();
            self.checked_out
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.checked_sub(1)
                })
                .ok();
            return;
        }

        let buffer = PooledBuffer {
            tensor,
            size_bytes,
            last_used: std::time::Instant::now(),
            use_count: 1,
        };

        let mut buckets = self.buckets.lock().unwrap();
        buckets.entry(pool_key).or_default().push(buffer);
        self.checked_out
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_sub(1)
            })
            .ok();

        debug!(
            "Released tensor to pool: elem_count={}, pool_size={}",
            elem_count,
            buckets.get(&pool_key).map(|v| v.len()).unwrap_or(0)
        );
    }

    /// Get pool statistics
    pub fn stats(&self) -> MetalPoolStats {
        let buckets = self.buckets.lock().unwrap();
        let total_buffers: usize = buckets.values().map(|v| v.len()).sum();

        MetalPoolStats {
            total_buffers,
            checked_out: self.checked_out.load(Ordering::Relaxed),
            current_usage_bytes: self.current_usage.load(Ordering::Relaxed),
            max_usage_bytes: self.config.max_pool_size_bytes,
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            bucket_count: buckets.len(),
        }
    }

    /// Clear all pooled buffers
    pub fn clear(&self) {
        let mut buckets = self.buckets.lock().unwrap();
        let total_cleared: usize = buckets.values().map(|v| v.len()).sum();
        buckets.clear();
        self.current_usage.store(0, Ordering::Relaxed);

        info!("Cleared {} buffers from MetalMemoryPool", total_cleared);
    }

    /// Trim old buffers to reduce memory usage
    ///
    /// Removes buffers that haven't been used for longer than the threshold
    pub fn trim_old_buffers(&self, max_age: std::time::Duration) -> usize {
        let mut buckets = self.buckets.lock().unwrap();
        let now = std::time::Instant::now();
        let mut total_freed = 0;

        for buffers in buckets.values_mut() {
            let before_len = buffers.len();
            buffers.retain(|buf| {
                let keep = now.duration_since(buf.last_used) < max_age;
                if !keep {
                    total_freed += 1;
                    self.current_usage
                        .fetch_sub(buf.size_bytes, Ordering::Relaxed);
                }
                keep
            });

            debug!("Trimmed {} old buffers", before_len - buffers.len());
        }

        // Remove empty buckets
        buckets.retain(|_, v| !v.is_empty());

        total_freed
    }

    /// Check if memory pressure is high
    pub fn is_under_pressure(&self) -> bool {
        let usage = self.current_usage.load(Ordering::Relaxed) as f32;
        let max = self.config.max_pool_size_bytes as f32;
        (usage / max) > self.config.pressure_threshold
    }

    /// Get the hit rate (0.0 - 1.0)
    pub fn hit_rate(&self) -> f32 {
        let hits = self.hits.load(Ordering::Relaxed) as f32;
        let misses = self.misses.load(Ordering::Relaxed) as f32;
        let total = hits + misses;

        if total == 0.0 {
            0.0
        } else {
            hits / total
        }
    }
}

/// Statistics for the Metal memory pool
#[derive(Debug, Clone)]
pub struct MetalPoolStats {
    pub total_buffers: usize,
    pub checked_out: usize,
    pub current_usage_bytes: usize,
    pub max_usage_bytes: usize,
    pub hits: usize,
    pub misses: usize,
    pub bucket_count: usize,
}

impl MetalPoolStats {
    /// Get utilization ratio (0.0 - 1.0)
    pub fn utilization(&self) -> f32 {
        self.current_usage_bytes as f32 / self.max_usage_bytes.max(1) as f32
    }

    /// Get hit rate (0.0 - 1.0)
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f32 / total as f32
        }
    }
}

/// Global memory pool manager for Metal devices
///
/// Uses device location as key since Device doesn't implement Hash
pub struct MetalPoolManager {
    pools: Mutex<HashMap<String, Arc<MetalMemoryPool>>>,
    default_config: MetalMemoryPoolConfig,
}

/// Get a unique key for a device
fn device_key(device: &Device) -> String {
    // Use device type and ordinal as key
    format!("{:?}", device)
}

impl MetalPoolManager {
    /// Create a new pool manager
    pub fn new() -> Self {
        Self {
            pools: Mutex::new(HashMap::new()),
            default_config: MetalMemoryPoolConfig::default(),
        }
    }

    /// Get or create a memory pool for a device
    pub fn get_pool(&self, device: &Device) -> Result<Arc<MetalMemoryPool>> {
        let key = device_key(device);
        let mut pools = self.pools.lock().unwrap();

        if let Some(pool) = pools.get(&key) {
            return Ok(pool.clone());
        }

        // Create new pool
        let pool = Arc::new(MetalMemoryPool::new(
            device.clone(),
            self.default_config.clone(),
        )?);

        pools.insert(key, pool.clone());
        Ok(pool)
    }

    /// Set the default configuration for new pools
    pub fn set_default_config(&mut self, config: MetalMemoryPoolConfig) {
        self.default_config = config;
    }

    /// Clear all pools
    pub fn clear_all(&self) {
        let pools = self.pools.lock().unwrap();
        for pool in pools.values() {
            pool.clear();
        }
    }

    /// Get statistics for all pools
    pub fn all_stats(&self) -> Vec<(String, MetalPoolStats)> {
        let pools = self.pools.lock().unwrap();
        pools
            .iter()
            .map(|(key, pool)| (key.clone(), pool.stats()))
            .collect()
    }
}

impl Default for MetalPoolManager {
    fn default() -> Self {
        Self::new()
    }
}

static GLOBAL_POOL_MANAGER: OnceLock<MetalPoolManager> = OnceLock::new();

impl MetalPoolManager {
    /// Get the global Metal pool manager.
    pub fn global() -> &'static MetalPoolManager {
        GLOBAL_POOL_MANAGER.get_or_init(MetalPoolManager::new)
    }
}

/// Fetch (or create) the shared Metal memory pool for a device.
pub fn metal_pool_for_device(device: &Device) -> Option<Arc<MetalMemoryPool>> {
    if device.is_metal() {
        MetalPoolManager::global().get_pool(device).ok()
    } else {
        None
    }
}

/// Tensor wrapper that returns pooled tensors on drop.
pub struct PooledTensor {
    tensor: Option<Tensor>,
    pool: Option<Arc<MetalMemoryPool>>,
}

impl PooledTensor {
    pub fn new(tensor: Tensor, pool: Option<Arc<MetalMemoryPool>>) -> Self {
        Self {
            tensor: Some(tensor),
            pool,
        }
    }

    pub fn tensor(&self) -> &Tensor {
        self.tensor.as_ref().expect("PooledTensor missing tensor")
    }

    pub fn into_inner(mut self) -> Tensor {
        self.pool = None;
        self.tensor.take().expect("PooledTensor missing tensor")
    }
}

impl Drop for PooledTensor {
    fn drop(&mut self) {
        if let (Some(pool), Some(tensor)) = (self.pool.take(), self.tensor.take()) {
            pool.release(tensor);
        }
    }
}

#[cfg(test)]
impl MetalMemoryPool {
    fn new_for_tests(device: Device, config: MetalMemoryPoolConfig) -> Self {
        Self {
            config,
            device,
            buckets: Mutex::new(HashMap::new()),
            current_usage: AtomicUsize::new(0),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
            checked_out: AtomicUsize::new(0),
        }
    }
}

/// Convenience function to get the size of a tensor in bytes
pub fn tensor_size_bytes(tensor: &Tensor) -> usize {
    tensor.dtype().size_in_bytes() * tensor.elem_count()
}

/// Check if tensor pooling should be used for this tensor size
pub fn should_pool_tensor(shape: &Shape, dtype: DType, config: &MetalMemoryPoolConfig) -> bool {
    let size_bytes = shape.elem_count() * dtype.size_in_bytes();
    size_bytes >= config.min_pool_size && size_bytes <= config.max_pool_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_key_calculation() {
        let config = MetalMemoryPoolConfig::default();
        let device = Device::Cpu; // Use CPU for testing
        let pool = MetalMemoryPool::new_for_tests(device, config);

        let elem_count = 1024;
        let size_bytes = elem_count * DType::F32.size_in_bytes();
        assert_eq!(pool.pool_key(elem_count, 1024), None); // Below minimum
        assert_eq!(pool.pool_key(elem_count, size_bytes), Some(elem_count));
    }

    #[test]
    fn test_pool_stats() {
        let config = MetalMemoryPoolConfig::default();
        let device = Device::Cpu;
        let pool = MetalMemoryPool::new_for_tests(device, config);

        let stats = pool.stats();
        assert_eq!(stats.total_buffers, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.hit_rate(), 0.0);
    }
}
