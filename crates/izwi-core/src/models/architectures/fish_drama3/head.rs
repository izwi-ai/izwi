//! The compact main head (R2).
//!
//! Every engine samples the main token from 4,097 candidates only:
//! `<|im_end|>` and the 4,096 `<|semantic:N|>` ids. So only those rows of
//! the untied `lm_head` are read at load, through the checkpoint's row-slice
//! reader, and the 252,288-row head never exists in memory. Logits come out
//! F32 in candidate order, eos first: logit index 0 is eos and index
//! `1 + code` is `<|semantic:code|>`.
//!
//! On the CPU the weights are the BF16 rows widened to F32 (about 32 MiB);
//! on Metal they are F16 (16 MiB) and the projection runs in F16, widened
//! to F32 logits (F32 throughout once the load-time range probe breached
//! the head); on CUDA they stay BF16 and the projection writes F32 logits.
//! Each hidden row is projected on its own (on CUDA through the
//! row-invariant multi-row GEMV once it is enabled), so a row's logits
//! never depend on how many rows share the call.

use candle_core::{DType, Device, Tensor};

use super::checkpoint::{FishDrama3Checkpoint, LM_HEAD};
use super::config::{FishDrama3Config, FishDrama3TokenIds};
use crate::error::{Error, Result};

pub struct FishDrama3CompactHead {
    /// `[candidates, hidden]`: F32 on the CPU, F16 on Metal, BF16 on CUDA.
    weight: Tensor,
    /// The projection's dtype: F32 on the CPU, F16 on Metal unless the
    /// range probe moved it to F32, BF16 on CUDA.
    act: DType,
    /// Logit index → token id: eos, then the semantic band.
    token_ids: Vec<u32>,
    ids: FishDrama3TokenIds,
    hidden: usize,
}

impl std::fmt::Debug for FishDrama3CompactHead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FishDrama3CompactHead")
            .field("candidates", &self.token_ids.len())
            .field("hidden", &self.hidden)
            .finish()
    }
}

impl FishDrama3CompactHead {
    pub fn load(
        checkpoint: &FishDrama3Checkpoint,
        config: &FishDrama3Config,
        device: &Device,
    ) -> Result<Self> {
        let rows = config.ids.compact_head_rows();
        let dtype = match device {
            Device::Metal(_) => DType::F16,
            Device::Cuda(_) => DType::BF16,
            Device::Cpu => DType::F32,
        };
        let weight = checkpoint.bf16_rows_tensor(LM_HEAD, &rows, dtype, device)?;
        let (count, hidden) = weight.dims2()?;
        if count != rows.len() || hidden != config.trunk.hidden {
            return Err(Error::ModelLoadError(format!(
                "Fish Audio Drama 3 compact head is [{count}, {hidden}]; expected [{}, {}]",
                rows.len(),
                config.trunk.hidden
            )));
        }
        Ok(Self {
            weight,
            act: dtype,
            token_ids: rows.into_iter().map(|row| row as u32).collect(),
            ids: config.ids,
            hidden,
        })
    }

    /// Candidates: eos and the semantic band.
    pub fn len(&self) -> usize {
        self.token_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.token_ids.is_empty()
    }

    /// The logit index of eos.
    pub fn eos_index(&self) -> usize {
        0
    }

    /// Logit index → token id.
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn token_id(&self, index: usize) -> Result<u32> {
        self.token_ids.get(index).copied().ok_or_else(|| {
            Error::InferenceError(format!(
                "compact head index {index} is past its {} candidates",
                self.len()
            ))
        })
    }

    /// Token id → logit index, for eos and the semantic band only.
    pub fn logit_index(&self, token: u32) -> Option<usize> {
        if token == self.ids.eos {
            Some(0)
        } else {
            self.ids.semantic_code(token).map(|code| 1 + code as usize)
        }
    }

    /// Resident bytes of the weights.
    pub fn resident_bytes(&self) -> usize {
        self.weight.elem_count() * self.weight.dtype().size_in_bytes()
    }

    /// The projection's dtype.
    pub fn act(&self) -> DType {
        self.act
    }

    /// Runs the projection in `act` (the range probe's F32 fallback).
    pub(crate) fn set_act(&mut self, act: DType) {
        self.act = act;
    }

    /// F32 logits `[n, candidates]` of final-norm hidden rows `[n, hidden]`.
    pub fn logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let (n, width) = hidden.dims2()?;
        if width != self.hidden || n == 0 {
            return Err(Error::InferenceError(format!(
                "the compact head takes [n, {}] rows, got {:?}",
                self.hidden,
                hidden.dims()
            )));
        }
        let hidden = hidden.to_dtype(self.act)?;
        if hidden.device().is_cuda() {
            // BF16 rows summed into F32 logits (BF16 logits would lose two
            // decimal digits): the row-invariant GEMV once enabled, else one
            // F32 GEMV per row over the widened values.
            use crate::kernels::cuda::row_gemv::{row_gemv, supported, RowGemvWeight};
            let weight = RowGemvWeight::Dense(&self.weight);
            if crate::kernels::cuda::row_gemv::enabled()
                && supported(hidden.dtype(), weight, DType::F32)
            {
                return Ok(row_gemv(&hidden, weight, DType::F32)?);
            }
            let w = self.weight.to_dtype(DType::F32)?.t()?;
            let hidden = hidden.to_dtype(DType::F32)?;
            let rows = (0..n)
                .map(|row| hidden.narrow(0, row, 1)?.contiguous()?.matmul(&w))
                .collect::<candle_core::Result<Vec<_>>>()?;
            return Ok(Tensor::cat(&rows, 0)?);
        }
        let w = if self.weight.dtype() == self.act {
            self.weight.t()?
        } else {
            self.weight.to_dtype(self.act)?.t()?
        };
        let rows = (0..n)
            .map(|row| hidden.narrow(0, row, 1)?.contiguous()?.matmul(&w))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Tensor::cat(&rows, 0)?.to_dtype(DType::F32)?)
    }
}

#[cfg(test)]
mod tests {
    use super::super::synthetic::SyntheticCheckpoint;
    use super::*;
    use crate::kernels::numeric_check::{host_f32, relative_l2, xorshift_values};

    #[test]
    fn the_head_holds_eos_then_the_band_and_maps_indices_both_ways() {
        let synthetic = SyntheticCheckpoint::write(2);
        let config = &synthetic.config;
        let checkpoint = synthetic.open();
        let head = FishDrama3CompactHead::load(&checkpoint, config, &Device::Cpu).unwrap();
        let ids = config.ids;
        let band = ids.semantic_codes();
        assert_eq!(head.len(), band + 1);
        assert!(head.len() < config.trunk.vocab / 4);
        assert_eq!(head.token_id(head.eos_index()).unwrap(), ids.eos);
        assert_eq!(head.token_id(1).unwrap(), ids.semantic_start);
        assert_eq!(head.token_id(band).unwrap(), ids.semantic_end);
        assert!(head.token_id(band + 1).is_err());
        for index in 0..head.len() {
            let token = head.token_id(index).unwrap();
            assert_eq!(head.logit_index(token), Some(index));
        }
        assert_eq!(head.logit_index(ids.voice), None);
        assert_eq!(head.logit_index(ids.semantic_end + 1), None);
        assert_eq!(head.resident_bytes(), head.len() * config.trunk.hidden * 4);

        // Logits are the selected rows' dot products, row by row.
        let hidden = config.trunk.hidden;
        let full = checkpoint
            .bf16_to_f32(LM_HEAD, &[config.trunk.vocab, hidden])
            .unwrap();
        let x = xorshift_values(3 * hidden, 9, 1.0);
        let logits = head
            .logits(&Tensor::from_vec(x.clone(), (3, hidden), &Device::Cpu).unwrap())
            .unwrap();
        assert_eq!(logits.dims(), [3, head.len()]);
        let mut expected = Vec::new();
        for r in 0..3 {
            for &token in head.token_ids() {
                let w = &full[token as usize * hidden..(token as usize + 1) * hidden];
                expected.push(
                    x[r * hidden..(r + 1) * hidden]
                        .iter()
                        .zip(w)
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum::<f64>() as f32,
                );
            }
        }
        let got = host_f32(&logits).unwrap();
        assert!(relative_l2(&got, &expected) < 1e-6);
        // A row alone gets the same bits as beside others.
        let alone = head
            .logits(
                &Tensor::from_vec(x[hidden..2 * hidden].to_vec(), (1, hidden), &Device::Cpu)
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(host_f32(&alone).unwrap(), got[head.len()..2 * head.len()]);
    }
}
