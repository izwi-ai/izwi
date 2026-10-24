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

    /// Whether `head` is this table's tied head (it shares the tensor).
    pub(crate) fn ties(&self, head: &QMatMul) -> bool {
        matches!(head, QMatMul::QTensor(tensor) if Arc::ptr_eq(tensor, &self.tensor))
    }

    /// On Metal, move the table into a buffer the small-batch head kernel
    /// can share, and return the tied head and that kernel over it: still
    /// one copy of the table (the old one is freed once its head is
    /// replaced). `None` for a type or device the kernel does not serve.
    #[cfg(feature = "metal")]
    pub(crate) fn rehome_tied_head(
        &mut self,
    ) -> Result<Option<(QMatMul, crate::kernels::metal_kquant_mv::MetalKQuantMatVec)>> {
        use candle_core::quantized::QStorage;
        let device = self.tensor.device();
        let dtype = self.tensor.dtype();
        if !device.is_metal() || !crate::kernels::metal_kquant_mv::serves(dtype, self.hidden_size) {
            return Ok(None);
        }
        let storage = QStorage::from_data(self.tensor.data()?, &device, dtype)?;
        let QStorage::Metal(metal) = &storage else {
            return Ok(None);
        };
        let Some(kernel) = crate::kernels::metal_kquant_mv::MetalKQuantMatVec::new(
            metal.buffer().clone(),
            dtype,
            self.vocab_size,
            self.hidden_size,
        ) else {
            return Ok(None);
        };
        let tensor = Arc::new(QTensor::new(storage, (self.vocab_size, self.hidden_size))?);
        self.tensor = tensor.clone();
        Ok(Some((QMatMul::from_arc(tensor)?, kernel)))
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

    /// On Metal the re-homed table still gathers the same rows, its head
    /// gives the same logits, the small-batch kernel matches the head, and
    /// the table and head share one tensor.
    #[cfg(feature = "metal")]
    #[test]
    fn a_rehomed_table_shares_one_tensor_with_its_head_and_kernel() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "a_rehomed_table_shares_one_tensor_with_its_head_and_kernel",
        ) else {
            return;
        };
        let values: Vec<f32> = (0..40 * 256)
            .map(|index| ((index * 37 % 101) as f32 - 50.0) / 17.0)
            .collect();
        let dense = Tensor::from_vec(values, (40, 256), &Device::Cpu).unwrap();
        let mut table = QuantizedTokenTable::new(
            QTensor::quantize_onto(&dense, GgmlDType::Q6K, &device).unwrap(),
        )
        .unwrap();
        let ids = Tensor::new(&[3u32, 39, 0], &device).unwrap();
        let rows_before = table.embed(&ids).unwrap().to_vec2::<f32>().unwrap();
        let hidden = Tensor::from_vec(
            (0..3 * 256)
                .map(|index| (index % 7) as f32 - 3.0)
                .collect::<Vec<_>>(),
            (3, 1, 256),
            &device,
        )
        .unwrap();
        let logits_before = table.tied_head().unwrap().forward(&hidden).unwrap();
        let logits_before = logits_before
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let (head, kernel) = table.rehome_tied_head().unwrap().expect("Q6_K on Metal");
        assert!(table.ties(&head));
        assert_eq!(
            table.embed(&ids).unwrap().to_vec2::<f32>().unwrap(),
            rows_before
        );
        let scale = logits_before.iter().fold(0f32, |max, e| max.max(e.abs()));
        for logits in [
            head.forward(&hidden).unwrap(),
            kernel.forward(&hidden).unwrap(),
        ] {
            let logits = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (a, e) in logits.iter().zip(&logits_before) {
                assert!((a - e).abs() <= 2e-6 * scale, "{a} vs {e}");
            }
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
