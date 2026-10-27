//! Voice conditioning: the bundled default voice (`conds.pt`) and the shape
//! every prepared voice takes.
//!
//! `conds.pt` is upstream's `Conditionals.save`: a pickled dict with a `t3`
//! dict (`speaker_emb` [1, 256], `cond_prompt_speech_tokens` [1, 150] int64,
//! `emotion_adv` [1, 1, 1], and `None` for `clap_emb` and
//! `cond_prompt_speech_emb`) and a `gen` dict (`prompt_token` [1, n] int64,
//! `prompt_token_len` [1], `prompt_feat` [1, 2n, 80], `embedding` [1, 192],
//! and `None` for `prompt_feat_len`). The `None` entries are not tensors and
//! are not read; nothing here depends on them.

use std::path::Path;

use candle_core::{DType, Device, Tensor};

use crate::error::{Error, Result};
use crate::models::shared::weights::pytorch::PthTensorMap;

/// T3's voice conditioning, before exaggeration is applied. Exaggeration is
/// a per-request scalar and never stored here, so one prepared voice serves
/// every exaggeration.
#[derive(Debug, Clone)]
pub struct T3VoiceConditioning {
    /// Voice-encoder speaker embedding, [256].
    pub speaker_embedding: Vec<f32>,
    /// Reference speech tokens the perceiver reads (up to 150).
    pub prompt_speech_tokens: Vec<u32>,
}

/// S3Gen's reference: the speaker x-vector and the reference's speech tokens
/// and mel frames the flow continues from.
#[derive(Debug, Clone)]
pub struct S3GenVoiceReference {
    /// Reference speech tokens (25 Hz).
    pub prompt_tokens: Vec<u32>,
    /// Reference mel frames, row-major [frames, 80], two per token.
    pub prompt_mel: Vec<f32>,
    pub prompt_mel_frames: usize,
    /// CAMPPlus x-vector, [192].
    pub speaker_xvector: Vec<f32>,
}

/// One voice, as T3 and S3Gen consume it.
#[derive(Debug, Clone)]
pub struct ChatterboxVoice {
    pub t3: T3VoiceConditioning,
    pub s3gen: S3GenVoiceReference,
}

impl ChatterboxVoice {
    /// The bundled default voice. `conds.pt` also records an exaggeration
    /// (0.5); it is ignored, because the request supplies its own.
    pub fn load_default(path: &Path) -> Result<Self> {
        let cpu = Device::Cpu;
        let t3 = PthTensorMap::open(path, Some("t3"))?;
        let gen = PthTensorMap::open(path, Some("gen"))?;
        let read = |map: &PthTensorMap, name: &str| -> Result<Tensor> {
            map.get(name, &cpu, None)?.ok_or_else(|| {
                Error::ModelLoadError(format!(
                    "Chatterbox default voice {} has no {name}",
                    path.display()
                ))
            })
        };
        let floats = |tensor: Tensor| -> Result<Vec<f32>> {
            Ok(tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
        };
        let tokens = |tensor: Tensor| -> Result<Vec<u32>> {
            tensor
                .to_dtype(DType::I64)?
                .flatten_all()?
                .to_vec1::<i64>()?
                .into_iter()
                .map(|token| {
                    u32::try_from(token).map_err(|_| {
                        Error::ModelLoadError(format!(
                            "Chatterbox default voice has an invalid speech token {token}"
                        ))
                    })
                })
                .collect()
        };

        let speaker_embedding = floats(read(&t3, "speaker_emb")?)?;
        let t3_prompt = tokens(read(&t3, "cond_prompt_speech_tokens")?)?;
        let prompt_tokens = tokens(read(&gen, "prompt_token")?)?;
        let prompt_len = tokens(read(&gen, "prompt_token_len")?)?;
        let prompt_feat = read(&gen, "prompt_feat")?;
        let speaker_xvector = floats(read(&gen, "embedding")?)?;

        let (batch, frames, bins) = prompt_feat.dims3()?;
        if batch != 1 || bins != 80 {
            return Err(Error::ModelLoadError(format!(
                "Chatterbox default voice prompt_feat has shape {:?}, expected [1, frames, 80]",
                prompt_feat.dims()
            )));
        }
        if prompt_len.as_slice() != [prompt_tokens.len() as u32] {
            return Err(Error::ModelLoadError(format!(
                "Chatterbox default voice prompt_token_len {prompt_len:?} does not match its {} tokens",
                prompt_tokens.len()
            )));
        }
        let voice = Self {
            t3: T3VoiceConditioning {
                speaker_embedding,
                prompt_speech_tokens: t3_prompt,
            },
            s3gen: S3GenVoiceReference {
                prompt_tokens,
                prompt_mel: floats(prompt_feat)?,
                prompt_mel_frames: frames,
                speaker_xvector,
            },
        };
        voice.validate()?;
        Ok(voice)
    }

    /// Shape and range checks every voice must pass, default or cloned.
    pub fn validate(&self) -> Result<()> {
        let fail = |message: String| Err(Error::InvalidInput(message));
        if self.t3.speaker_embedding.len() != 256 {
            return fail(format!(
                "Chatterbox speaker embedding has {} values, expected 256",
                self.t3.speaker_embedding.len()
            ));
        }
        if self.t3.prompt_speech_tokens.len() > 150 {
            return fail(format!(
                "Chatterbox T3 prompt has {} speech tokens, at most 150",
                self.t3.prompt_speech_tokens.len()
            ));
        }
        if self.s3gen.speaker_xvector.len() != 192 {
            return fail(format!(
                "Chatterbox x-vector has {} values, expected 192",
                self.s3gen.speaker_xvector.len()
            ));
        }
        if self.s3gen.prompt_mel.len() != self.s3gen.prompt_mel_frames * 80 {
            return fail("Chatterbox reference mel does not hold whole 80-bin frames".into());
        }
        if self.s3gen.prompt_mel_frames != 2 * self.s3gen.prompt_tokens.len() {
            return fail(format!(
                "Chatterbox reference has {} mel frames for {} tokens, expected two per token",
                self.s3gen.prompt_mel_frames,
                self.s3gen.prompt_tokens.len()
            ));
        }
        let codes = self
            .t3
            .prompt_speech_tokens
            .iter()
            .chain(&self.s3gen.prompt_tokens);
        if let Some(code) = codes.copied().find(|&code| code >= 6561) {
            return fail(format!(
                "Chatterbox reference has speech token {code}, beyond the 6561 codes"
            ));
        }
        let finite = self
            .t3
            .speaker_embedding
            .iter()
            .chain(&self.s3gen.prompt_mel)
            .chain(&self.s3gen.speaker_xvector)
            .all(|value| value.is_finite());
        if !finite {
            return fail("Chatterbox voice conditioning is not finite".into());
        }
        Ok(())
    }

    /// Host bytes the voice holds.
    pub fn host_bytes(&self) -> usize {
        4 * (self.t3.speaker_embedding.len()
            + self.t3.prompt_speech_tokens.len()
            + self.s3gen.prompt_tokens.len()
            + self.s3gen.prompt_mel.len()
            + self.s3gen.speaker_xvector.len())
    }
}
