//! The backbone's input side and its codebook-0 head
//! (`tts/model/audio_io.py`, `anchors.py`, `backbone.py`).
//!
//! - Text positions read the token embedding.
//! - An audio frame reads the sum of its 32 codebook embeddings (a private
//!   `[32·2048, 2048]` table) divided by √32, RMS-normalised.
//! - The position after a reference clip reads the learned end-of-audio
//!   input, RMS-normalised the same way.
//! - The speaker anchor `fc2(GELU(fc1(e)))` of the clip's ECAPA embedding
//!   (or the learned null input) is added at every `<|audio|>` sentinel.
//!
//! The audio norm is torch's `nn.RMSNorm` without an explicit eps, which
//! uses the compute dtype's machine epsilon: 2⁻⁷ in the BF16 the model was
//! trained and served in. That value is kept on every backend.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::backbone::rows_apart_matmul;
use super::config::{CB0_CLASSES, CODEBOOKS, CODEBOOK_SIZE};
use super::prompt::{Maya2Prompt, PromptItem};
use crate::error::{Error, Result};

/// `torch.finfo(torch.bfloat16).eps`.
pub const AUDIO_NORM_EPS: f32 = 0.007_812_5;
const ECAPA_DIM: usize = 192;
const ANCHOR_HIDDEN: usize = 512;

pub struct Maya2Inputs {
    hidden: usize,
    dtype: DType,
    device: Device,
    /// Token embedding (gather only; kept 16-bit on the CPU).
    tokens: Tensor,
    audio_table: Tensor,
    audio_norm: Tensor,
    aeos_in: Tensor,
    anchor_fc1: (Tensor, Tensor),
    anchor_fc2: (Tensor, Tensor),
    anchor_null: Tensor,
    cb0_head: Tensor,
}

impl Maya2Inputs {
    /// `vb` holds the release's tensors in the model dtype; `tokens_vb` the
    /// same files in the dtype the token table is stored in.
    pub fn load(
        vb: &VarBuilder,
        tokens_vb: &VarBuilder,
        hidden: usize,
        vocab: usize,
    ) -> Result<Self> {
        let f32 = |t: Tensor| -> Result<Tensor> { Ok(t.to_dtype(DType::F32)?) };
        let anchor = vb.pp("speaker_anchor");
        Ok(Self {
            hidden,
            dtype: vb.dtype(),
            device: vb.device().clone(),
            tokens: tokens_vb.get((vocab, hidden), "backbone.lm.model.embed_tokens.weight")?,
            audio_table: vb.get((CODEBOOKS * CODEBOOK_SIZE, hidden), "audio_in.table.weight")?,
            audio_norm: f32(vb.get(hidden, "audio_in.norm.weight")?)?,
            aeos_in: f32(vb.get(hidden, "audio_in.aeos_in")?)?,
            anchor_fc1: (
                f32(anchor.get((ANCHOR_HIDDEN, ECAPA_DIM), "fc1.weight")?)?,
                f32(anchor.get(ANCHOR_HIDDEN, "fc1.bias")?)?,
            ),
            anchor_fc2: (
                f32(anchor.get((hidden, ANCHOR_HIDDEN), "fc2.weight")?)?,
                f32(anchor.get(hidden, "fc2.bias")?)?,
            ),
            anchor_null: f32(anchor.get(ECAPA_DIM, "null_vec")?)?,
            cb0_head: vb.get((CB0_CLASSES, hidden), "backbone.cb0_head.weight")?,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn rms_norm(&self, x: &Tensor) -> Result<Tensor> {
        // x [n, hidden] F32 → normalised, in the model dtype.
        Ok(candle_nn::ops::rms_norm(x, &self.audio_norm, AUDIO_NORM_EPS)?.to_dtype(self.dtype)?)
    }

    /// Embeddings `[n, hidden]` of `n` frames of codes (frame-major,
    /// 32 per frame).
    pub fn audio_frames(&self, codes: &Tensor) -> Result<Tensor> {
        let (frames, books) = codes.dims2()?;
        if books != CODEBOOKS {
            return Err(Error::InvalidInput(format!(
                "Maya2 audio frames take {CODEBOOKS} codes, got {books}"
            )));
        }
        let offsets = Tensor::arange(0u32, CODEBOOKS as u32, &self.device)?
            .affine(CODEBOOK_SIZE as f64, 0.0)?
            .unsqueeze(0)?;
        let rows = codes
            .to_dtype(DType::U32)?
            .broadcast_add(&offsets)?
            .flatten_all()?;
        let summed = self
            .audio_table
            .index_select(&rows, 0)?
            .to_dtype(DType::F32)?
            .reshape((frames, CODEBOOKS, self.hidden))?
            .sum(1)?;
        self.rms_norm(&(summed / (CODEBOOKS as f64).sqrt())?)
    }

    /// The end-of-audio input `[1, hidden]`.
    pub fn aeos(&self) -> Result<Tensor> {
        self.rms_norm(&self.aeos_in.unsqueeze(0)?)
    }

    /// The speaker anchor `[hidden]` (model dtype) of an L2-normalised
    /// ECAPA embedding, or of the learned null input.
    pub fn anchor(&self, ecapa: Option<&[f32]>) -> Result<Tensor> {
        let x = match ecapa {
            Some(values) if values.len() == ECAPA_DIM => {
                Tensor::from_slice(values, (1, ECAPA_DIM), &self.device)?
            }
            Some(values) => {
                return Err(Error::InvalidInput(format!(
                    "Maya2 speaker embeddings have {ECAPA_DIM} values, got {}",
                    values.len()
                )))
            }
            None => self.anchor_null.unsqueeze(0)?,
        };
        let h = x
            .matmul(&self.anchor_fc1.0.t()?)?
            .broadcast_add(&self.anchor_fc1.1)?
            .gelu_erf()?;
        let out = h
            .matmul(&self.anchor_fc2.0.t()?)?
            .broadcast_add(&self.anchor_fc2.1)?;
        Ok(out.squeeze(0)?.to_dtype(self.dtype)?)
    }

    /// The prompt's input embeddings `[positions, hidden]`: text gathered,
    /// reference frames summed from `reference_codes` (frame-major), the
    /// end-of-audio input, and `anchor` added at each sentinel.
    pub fn compose(
        &self,
        prompt: &Maya2Prompt,
        reference_codes: &[u32],
        anchor: &Tensor,
    ) -> Result<Tensor> {
        // Runs of each kind are embedded together, in prompt order.
        enum Run {
            Text(Vec<u32>),
            Frames(Vec<usize>),
            Aeos,
        }
        let mut runs: Vec<Run> = Vec::new();
        for item in &prompt.items {
            match (*item, runs.last_mut()) {
                (PromptItem::Text(id), Some(Run::Text(ids))) => ids.push(id),
                (PromptItem::Text(id), _) => runs.push(Run::Text(vec![id])),
                (PromptItem::Audio(frame), Some(Run::Frames(frames))) => frames.push(frame),
                (PromptItem::Audio(frame), _) => runs.push(Run::Frames(vec![frame])),
                (PromptItem::Aeos, _) => runs.push(Run::Aeos),
            }
        }
        let mut parts = Vec::with_capacity(runs.len());
        for run in runs {
            parts.push(match run {
                Run::Text(ids) => {
                    let count = ids.len();
                    let ids = Tensor::from_vec(ids, count, &self.device)?;
                    self.tokens.index_select(&ids, 0)?.to_dtype(self.dtype)?
                }
                Run::Frames(frames) => {
                    let mut codes = Vec::with_capacity(frames.len() * CODEBOOKS);
                    for frame in &frames {
                        let start = frame * CODEBOOKS;
                        let slice =
                            reference_codes
                                .get(start..start + CODEBOOKS)
                                .ok_or_else(|| {
                                    Error::InvalidInput(
                                        "Maya2 prompt names a missing reference frame".into(),
                                    )
                                })?;
                        codes.extend_from_slice(slice);
                    }
                    let codes = Tensor::from_vec(codes, (frames.len(), CODEBOOKS), &self.device)?;
                    self.audio_frames(&codes)?
                }
                Run::Aeos => self.aeos()?,
            });
        }
        let mut embeds = Tensor::cat(&parts, 0)?;
        let positions = embeds.dim(0)?;
        // Add the anchor at the sentinels: a one-hot [positions, 1] mask
        // times the anchor row.
        let mut mask = vec![0f32; positions];
        for &sentinel in &prompt.sentinels {
            *mask.get_mut(sentinel).ok_or_else(|| {
                Error::InvalidInput("Maya2 prompt sentinel is out of range".into())
            })? = 1.0;
        }
        let mask = Tensor::from_vec(mask, (positions, 1), &self.device)?;
        let added = mask.broadcast_mul(&anchor.to_dtype(DType::F32)?.unsqueeze(0)?)?;
        embeds = (embeds.to_dtype(DType::F32)? + added)?.to_dtype(self.dtype)?;
        Ok(embeds)
    }

    /// Codebook-0 logits `[rows, 2049]` (F32) of backbone states `[rows,
    /// hidden]`.
    pub fn cb0_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        // Rows apart, so a request's logits do not move with its batch.
        Ok(
            rows_apart_matmul(&hidden.to_dtype(self.dtype)?, &self.cb0_head)?
                .to_dtype(DType::F32)?,
        )
    }
}
