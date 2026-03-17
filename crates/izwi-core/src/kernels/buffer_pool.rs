//! Static buffer pool for eliminating allocations during inference.
//!
//! This module provides a pre-allocated scratchpad for intermediate tensors,
//! eliminating Arc clones and memory allocations during the forward pass.
//! This is critical for small models where CPU/GPU synchronization overhead
//! dominates performance.

use candle_core::{DType, Device, Tensor};
use std::sync::{Arc, Mutex};

use crate::error::{Error, Result};

/// A pooled tensor buffer that can be reused across inference steps.
#[derive(Debug, Clone)]
pub struct PooledBuffer {
    /// The underlying tensor storage
    tensor: Tensor,
    /// Maximum capacity in elements
    capacity: usize,
    /// Current logical shape
    shape: Vec<usize>,
    /// Data type
    dtype: DType,
}

impl PooledBuffer {
    /// Create a new pooled buffer with the given capacity.
    pub fn new(capacity: usize, dtype: DType, device: &Device) -> Result<Self> {
        let tensor = Tensor::zeros(capacity, dtype, device)
            .map_err(|e| Error::InferenceError(format!("Failed to allocate buffer: {}", e)))?;

        Ok(Self {
            tensor,
            capacity,
            shape: vec![capacity],
            dtype,
        })
    }

    /// Get a view of this buffer reshaped to the requested dimensions.
    /// The total element count must not exceed capacity.
    pub fn view(&self, shape: &[usize]) -> Result<Tensor> {
        let numel: usize = shape.iter().product();
        if numel > self.capacity {
            return Err(Error::InvalidInput(format!(
                "Buffer view requested {} elements but capacity is {}",
                numel, self.capacity
            )));
        }

        self.tensor
            .reshape(shape)
            .map_err(|e| Error::InferenceError(format!("Failed to reshape buffer: {}", e)))
    }

    /// Get a mutable reference to the underlying tensor for in-place operations.
    pub fn as_tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Reset the buffer to zeros (for clean state between uses).
    pub fn clear(&mut self) -> Result<()> {
        self.tensor = Tensor::zeros(self.capacity, self.dtype, self.tensor.device())
            .map_err(|e| Error::InferenceError(format!("Failed to clear buffer: {}", e)))?;
        Ok(())
    }

    /// Get the buffer capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the data type.
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

/// Configuration for buffer pool sizing.
#[derive(Debug, Clone)]
pub struct BufferPoolConfig {
    /// Number of small buffers (for activations, 1K-4K elements)
    pub num_small: usize,
    /// Size of small buffers in elements
    pub small_size: usize,
    /// Number of medium buffers (for attention heads, 8K-32K elements)
    pub num_medium: usize,
    /// Size of medium buffers in elements
    pub medium_size: usize,
    /// Number of large buffers (for full layers, 64K+ elements)
    pub num_large: usize,
    /// Size of large buffers in elements
    pub large_size: usize,
    /// Data type for buffers
    pub dtype: DType,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        // Sizing for Qwen 3.5 4B model
        // - Small: individual head activations
        // - Medium: attention head groups
        // - Large: full layer outputs
        Self {
            num_small: 16,
            small_size: 4096,
            num_medium: 8,
            medium_size: 32768,
            num_large: 4,
            large_size: 262144,
            dtype: DType::F32,
        }
    }
}

impl BufferPoolConfig {
    /// Create config from environment variables.
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(val) = std::env::var("IZWI_BUFFER_POOL_SMALL") {
            if let Ok((num, size)) = parse_pool_config(&val) {
                config.num_small = num;
                config.small_size = size;
            }
        }

        if let Ok(val) = std::env::var("IZWI_BUFFER_POOL_MEDIUM") {
            if let Ok((num, size)) = parse_pool_config(&val) {
                config.num_medium = num;
                config.medium_size = size;
            }
        }

        if let Ok(val) = std::env::var("IZWI_BUFFER_POOL_LARGE") {
            if let Ok((num, size)) = parse_pool_config(&val) {
                config.num_large = num;
                config.large_size = size;
            }
        }

        config
    }

    /// Total memory footprint in bytes.
    pub fn total_memory_bytes(&self) -> usize {
        let bytes_per_element = match self.dtype {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            _ => 4,
        };

        (self.num_small * self.small_size
            + self.num_medium * self.medium_size
            + self.num_large * self.large_size)
            * bytes_per_element
    }
}

/// A pool of pre-allocated buffers for inference.
pub struct BufferPool {
    small: Vec<PooledBuffer>,
    medium: Vec<PooledBuffer>,
    large: Vec<PooledBuffer>,
    /// Track which buffers are currently in use
    small_in_use: Vec<bool>,
    medium_in_use: Vec<bool>,
    large_in_use: Vec<bool>,
}

impl BufferPool {
    /// Create a new buffer pool with the given configuration.
    pub fn new(config: &BufferPoolConfig, device: &Device) -> Result<Self> {
        let mut small = Vec::with_capacity(config.num_small);
        let mut medium = Vec::with_capacity(config.num_medium);
        let mut large = Vec::with_capacity(config.num_large);

        for _ in 0..config.num_small {
            small.push(PooledBuffer::new(config.small_size, config.dtype, device)?);
        }

        for _ in 0..config.num_medium {
            medium.push(PooledBuffer::new(config.medium_size, config.dtype, device)?);
        }

        for _ in 0..config.num_large {
            large.push(PooledBuffer::new(config.large_size, config.dtype, device)?);
        }

        tracing::info!(
            "Created buffer pool: {} small ({}B each), {} medium ({}B each), {} large ({}B each). Total: {}MB",
            config.num_small, config.small_size * 4,
            config.num_medium, config.medium_size * 4,
            config.num_large, config.large_size * 4,
            config.total_memory_bytes() / (1024 * 1024)
        );

        Ok(Self {
            small,
            medium,
            large,
            small_in_use: vec![false; config.num_small],
            medium_in_use: vec![false; config.num_medium],
            large_in_use: vec![false; config.num_large],
        })
    }

    /// Acquire a small buffer. Returns None if all are in use.
    pub fn acquire_small(&mut self) -> Option<(usize, &PooledBuffer)> {
        for (i, in_use) in self.small_in_use.iter_mut().enumerate() {
            if !*in_use {
                *in_use = true;
                return Some((i, &self.small[i]));
            }
        }
        tracing::warn!("All small buffers in use - consider increasing pool size");
        None
    }

    /// Acquire a medium buffer. Returns None if all are in use.
    pub fn acquire_medium(&mut self) -> Option<(usize, &PooledBuffer)> {
        for (i, in_use) in self.medium_in_use.iter_mut().enumerate() {
            if !*in_use {
                *in_use = true;
                return Some((i, &self.medium[i]));
            }
        }
        tracing::warn!("All medium buffers in use - consider increasing pool size");
        None
    }

    /// Acquire a large buffer. Returns None if all are in use.
    pub fn acquire_large(&mut self) -> Option<(usize, &PooledBuffer)> {
        for (i, in_use) in self.large_in_use.iter_mut().enumerate() {
            if !*in_use {
                *in_use = true;
                return Some((i, &self.large[i]));
            }
        }
        tracing::warn!("All large buffers in use - consider increasing pool size");
        None
    }

    /// Release a small buffer back to the pool.
    pub fn release_small(&mut self, index: usize) {
        if index < self.small_in_use.len() {
            self.small_in_use[index] = false;
        }
    }

    /// Release a medium buffer back to the pool.
    pub fn release_medium(&mut self, index: usize) {
        if index < self.medium_in_use.len() {
            self.medium_in_use[index] = false;
        }
    }

    /// Release a large buffer back to the pool.
    pub fn release_large(&mut self, index: usize) {
        if index < self.large_in_use.len() {
            self.large_in_use[index] = false;
        }
    }

    /// Release all buffers.
    pub fn release_all(&mut self) {
        self.small_in_use.fill(false);
        self.medium_in_use.fill(false);
        self.large_in_use.fill(false);
    }

    /// Get usage statistics.
    pub fn usage_stats(&self) -> BufferPoolStats {
        let small_used = self.small_in_use.iter().filter(|&&x| x).count();
        let medium_used = self.medium_in_use.iter().filter(|&&x| x).count();
        let large_used = self.large_in_use.iter().filter(|&&x| x).count();

        BufferPoolStats {
            small_available: self.small.len() - small_used,
            small_total: self.small.len(),
            medium_available: self.medium.len() - medium_used,
            medium_total: self.medium.len(),
            large_available: self.large.len() - large_used,
            large_total: self.large.len(),
        }
    }
}

/// Usage statistics for the buffer pool.
#[derive(Debug, Clone, Copy)]
pub struct BufferPoolStats {
    pub small_available: usize,
    pub small_total: usize,
    pub medium_available: usize,
    pub medium_total: usize,
    pub large_available: usize,
    pub large_total: usize,
}

/// Thread-safe wrapper around BufferPool using std::sync::OnceLock for safe initialization.
#[derive(Clone)]
pub struct SharedBufferPool {
    inner: Arc<Mutex<BufferPool>>,
}

impl SharedBufferPool {
    pub fn new(pool: BufferPool) -> Self {
        Self {
            inner: Arc::new(Mutex::new(pool)),
        }
    }

    /// Acquire a buffer with automatic size selection.
    pub fn acquire(&self, num_elements: usize) -> Option<(BufferSize, usize, PooledBuffer)> {
        let mut pool = self.inner.lock().ok()?;

        if num_elements <= pool.small.first()?.capacity() {
            pool.acquire_small()
                .map(|(idx, buf)| (BufferSize::Small, idx, buf.clone()))
        } else if num_elements <= pool.medium.first()?.capacity() {
            pool.acquire_medium()
                .map(|(idx, buf)| (BufferSize::Medium, idx, buf.clone()))
        } else if num_elements <= pool.large.first()?.capacity() {
            pool.acquire_large()
                .map(|(idx, buf)| (BufferSize::Large, idx, buf.clone()))
        } else {
            tracing::warn!(
                "Requested {} elements exceeds largest buffer capacity {}",
                num_elements,
                pool.large.first()?.capacity()
            );
            None
        }
    }

    /// Release a buffer back to the pool.
    pub fn release(&self, size: BufferSize, index: usize) {
        if let Ok(mut pool) = self.inner.lock() {
            match size {
                BufferSize::Small => pool.release_small(index),
                BufferSize::Medium => pool.release_medium(index),
                BufferSize::Large => pool.release_large(index),
            }
        }
    }

    /// Get usage statistics.
    pub fn stats(&self) -> Option<BufferPoolStats> {
        self.inner.lock().ok().map(|p| p.usage_stats())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferSize {
    Small,
    Medium,
    Large,
}

/// Parse pool config string like "8,4096" (num,size).
fn parse_pool_config(s: &str) -> Result<(usize, usize)> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        return Err(Error::InvalidInput(format!(
            "Pool config must be 'num,size', got: {}",
            s
        )));
    }

    let num = parts[0]
        .trim()
        .parse()
        .map_err(|_| Error::InvalidInput(format!("Invalid number: {}", parts[0])))?;
    let size = parts[1]
        .trim()
        .parse()
        .map_err(|_| Error::InvalidInput(format!("Invalid size: {}", parts[1])))?;

    Ok((num, size))
}

/// Global buffer pool for inference using OnceLock for thread-safe initialization.
use std::sync::OnceLock;
static GLOBAL_BUFFER_POOL: OnceLock<SharedBufferPool> = OnceLock::new();

/// Initialize the global buffer pool.
pub fn init_global_buffer_pool(config: &BufferPoolConfig, device: &Device) -> Result<()> {
    let pool = BufferPool::new(config, device)?;
    let shared = SharedBufferPool::new(pool);

    GLOBAL_BUFFER_POOL
        .set(shared)
        .map_err(|_| Error::InferenceError("Global buffer pool already initialized".to_string()))?;

    Ok(())
}

/// Get the global buffer pool if initialized.
pub fn global_buffer_pool() -> Option<SharedBufferPool> {
    GLOBAL_BUFFER_POOL.get().cloned()
}

/// Check if buffer pooling is enabled.
pub fn buffer_pooling_enabled() -> bool {
    GLOBAL_BUFFER_POOL.get().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_buffer_pool_basic() {
        let device = Device::Cpu;
        let config = BufferPoolConfig {
            num_small: 2,
            small_size: 1024,
            num_medium: 1,
            medium_size: 4096,
            num_large: 1,
            large_size: 16384,
            dtype: DType::F32,
        };

        let mut pool = BufferPool::new(&config, &device).unwrap();

        // Acquire all small buffers
        let (idx1, _) = pool.acquire_small().unwrap();
        let (idx2, _) = pool.acquire_small().unwrap();

        // Third acquisition should fail
        assert!(pool.acquire_small().is_none());

        // Release and reacquire
        pool.release_small(idx1);
        let (idx3, buf) = pool.acquire_small().unwrap();
        assert_eq!(idx1, idx3);

        // Test view
        let view = buf.view(&[256, 4]).unwrap();
        assert_eq!(view.shape().dims(), &[256, 4]);

        // Test oversized view
        assert!(buf.view(&[2048]).is_err());
    }

    #[test]
    fn test_shared_pool() {
        let device = Device::Cpu;
        let config = BufferPoolConfig::default();
        let pool = BufferPool::new(&config, &device).unwrap();
        let shared = SharedBufferPool::new(pool);

        // Acquire a buffer
        let (size, idx, buf) = shared.acquire(1024).unwrap();
        assert_eq!(size, BufferSize::Small);
        assert!(buf.capacity() >= 1024);

        // Release it
        shared.release(size, idx);

        // Check stats
        let stats = shared.stats().unwrap();
        assert!(stats.small_available > 0);
    }
}
