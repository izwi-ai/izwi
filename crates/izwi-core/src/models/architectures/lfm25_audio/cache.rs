use candle_core::Tensor;
use candle_nn::kv_cache::KvCache;

use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub(super) struct DenseKvCache {
    inner: Option<KvCache>,
    min_initial_capacity: usize,
}

#[derive(Debug)]
pub(super) struct DenseKvCacheView {
    pub current_k: Tensor,
    pub current_v: Tensor,
    pub full_k: Tensor,
    pub full_v: Tensor,
    pub valid_len: usize,
}

impl DenseKvCache {
    pub fn new(min_initial_capacity: usize) -> Self {
        Self {
            inner: None,
            min_initial_capacity: min_initial_capacity.max(1),
        }
    }

    pub fn reset(&mut self) {
        self.inner = None;
    }

    pub fn len(&self) -> usize {
        self.inner
            .as_ref()
            .map(KvCache::current_seq_len)
            .unwrap_or(0)
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<DenseKvCacheView> {
        let append_tokens = k.dim(2)?;
        if append_tokens == 0 {
            return Err(Error::InferenceError(
                "Cannot append empty LFM2.5 Audio K/V cache chunk".to_string(),
            ));
        }

        if self.inner.is_none() {
            let initial_capacity = append_tokens.max(self.min_initial_capacity);
            self.inner = Some(KvCache::new(2, initial_capacity));
        }

        let cache = self.inner.as_mut().ok_or_else(|| {
            Error::InferenceError("LFM2.5 Audio K/V cache was not initialized".to_string())
        })?;
        let (current_k, current_v) = cache.append(k, v)?;
        let full_k = cache
            .k_cache()
            .all_data()
            .as_ref()
            .cloned()
            .ok_or_else(|| Error::InferenceError("LFM2.5 key cache is empty".to_string()))?;
        let full_v = cache
            .v_cache()
            .all_data()
            .as_ref()
            .cloned()
            .ok_or_else(|| Error::InferenceError("LFM2.5 value cache is empty".to_string()))?;

        Ok(DenseKvCacheView {
            current_k,
            current_v,
            full_k,
            full_v,
            valid_len: cache.current_seq_len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_kv_cache_uses_observed_first_append_capacity() {
        let device = candle_core::Device::Cpu;
        let mut cache = DenseKvCache::new(1);
        let k = Tensor::zeros((1, 2, 3, 4), candle_core::DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 2, 3, 4), candle_core::DType::F32, &device).unwrap();
        let view = cache.append(&k, &v).unwrap();

        assert_eq!(view.valid_len, 3);
        assert_eq!(view.current_k.dims(), &[1, 2, 3, 4]);
        assert_eq!(view.full_k.dims(), &[1, 2, 3, 4]);
    }

    #[test]
    fn dense_kv_cache_can_reserve_small_inner_loop_capacity() {
        let device = candle_core::Device::Cpu;
        let mut cache = DenseKvCache::new(8);
        let k = Tensor::zeros((1, 2, 1, 4), candle_core::DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 2, 1, 4), candle_core::DType::F32, &device).unwrap();
        let view = cache.append(&k, &v).unwrap();

        assert_eq!(view.valid_len, 1);
        assert_eq!(view.current_k.dims(), &[1, 2, 1, 4]);
        assert_eq!(view.full_k.dims(), &[1, 2, 8, 4]);
    }
}
