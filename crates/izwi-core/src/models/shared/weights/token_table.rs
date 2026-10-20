//! A vocabulary table kept once, in its stored quantized form.
//!
//! Dequantizing a GGUF token table to a dense F32 `Embedding` multiplies its
//! size by 4-8 (Qwen3.5's 248,320 x 2,560 Q6_K table is 0.52 GB stored and
//! 2.54 GB as F32), and a tied output head then reads the same table a
//! second time. Gathering rows straight from the quantized tensor
//! (`QTensor::embedding`, on CPU, Metal and CUDA) produces the same F32 rows,
//! and a tied head shares the one tensor.

use std::sync::Arc;

use candle_core::quantized::{QMatMul, QTensor};
use candle_core::Tensor;

use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub(crate) struct QuantizedTokenTable {
    tensor: Arc<QTensor>,
    vocab_size: usize,
    hidden_size: usize,
}

impl QuantizedTokenTable {
    /// `tensor` is `[vocab, hidden]`.
    pub(crate) fn new(tensor: QTensor) -> Result<Self> {
        Self::from_arc(Arc::new(tensor))
    }

    /// A table over a tensor another owner may share (a quantized head).
    pub(crate) fn from_arc(tensor: Arc<QTensor>) -> Result<Self> {
        let (vocab_size, hidden_size) = tensor.shape().dims2()?;
        Ok(Self {
            tensor,
            vocab_size,
            hidden_size,
        })
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub(crate) fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// The F32 rows for `ids` (any shape): `[..ids, hidden]`.
    pub(crate) fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        self.tensor.embedding(ids).map_err(Error::from)
    }

    /// An output head over the same tensor, for a model whose head is tied
    /// to its embedding: no second copy of the table.
    pub(crate) fn tied_head(&self) -> Result<QMatMul> {
        QMatMul::from_arc(self.tensor.clone()).map_err(Error::from)
    }

    /// Stored bytes of the table (what it keeps resident).
    pub(crate) fn storage_bytes(&self) -> usize {
        self.tensor.storage_size_in_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;
    use candle_core::{DType, Device, Module};

    fn table(dtype: GgmlDType) -> (QuantizedTokenTable, Tensor) {
        let device = Device::Cpu;
        let values: Vec<f32> = (0..8 * 256)
            .map(|index| ((index * 37 % 101) as f32 - 50.0) / 17.0)
            .collect();
        let dense = Tensor::from_vec(values, (8, 256), &device).unwrap();
        let quantized = QTensor::quantize(&dense, dtype).unwrap();
        let dequantized = quantized.dequantize(&device).unwrap();
        (QuantizedTokenTable::new(quantized).unwrap(), dequantized)
    }

    /// The gathered rows are bit-identical to a dense lookup in the
    /// dequantized table, which is what the loaders kept before.
    #[test]
    fn gathered_rows_equal_the_dequantized_dense_lookup() {
        for dtype in [GgmlDType::Q6K, GgmlDType::Q4K, GgmlDType::Q8_0] {
            let (table, dense) = table(dtype);
            let ids = Tensor::new(&[[3u32, 0, 7], [7, 1, 3]], &Device::Cpu).unwrap();
            let gathered = table.embed(&ids).unwrap();
            assert_eq!(gathered.dims(), &[2, 3, 256]);
            assert_eq!(gathered.dtype(), DType::F32);
            let expected = candle_nn::Embedding::new(dense, 256).forward(&ids).unwrap();
            let difference = (gathered - expected)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert_eq!(difference, 0.0, "{dtype:?}");
            assert_eq!((table.vocab_size(), table.hidden_size()), (8, 256));
        }
    }

    #[test]
    fn the_tied_head_shares_the_table() {
        let (table, dense) = table(GgmlDType::Q6K);
        let head = table.tied_head().unwrap();
        let hidden = Tensor::ones((1, 256), DType::F32, &Device::Cpu).unwrap();
        let logits = head.forward(&hidden).unwrap();
        assert_eq!(logits.dims(), &[1, 8]);
        let expected = hidden.matmul(&dense.t().unwrap()).unwrap();
        let difference = (logits - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(difference < 1e-2, "{difference}");
    }
}
