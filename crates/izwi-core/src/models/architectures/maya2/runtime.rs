//! The loaded Maya2 networks and one utterance end to end (the engine runs
//! the same steps quantum by quantum, many requests at once: see
//! [`super::retained`]).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use serde::Serialize;

use super::backbone::{Maya2Backbone, Maya2BackboneRow};
use super::config::{
    self, check_release_config, Maya2BackboneConfig, CODEBOOKS, CODEBOOK_SIZE, DEPTH_LEVELS,
    MAYA2_DEFAULTS, MIN_FRAME_BUDGET, SAMPLE_RATE, STATE_CONTEXT_TOKENS,
};
use super::depth::{DepthGeometry, Maya2Depth};
use super::ecapa::Maya2Ecapa;
use super::guards::{piece_budget, suspend_windows, LoopGuard, Maya2PieceBudget, Maya2StopReason};
use super::inputs::Maya2Inputs;
use super::prompt::{Maya2Prompt, Maya2Reference, Maya2Tokenizer};
use super::sampling::{depth_gumbel, sample_cb0, Cb0State, Maya2SamplingParams};
use super::weights::mmap_release;
use crate::error::{Error, Result};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::codecs::mimi::{self, MimiDecoder, MimiEncoder};
use crate::models::shared::noise::SplitMix64;

/// Independent random streams of one request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Maya2Stream {
    /// Codebook-0 draws.
    Cb0 = 1,
    /// The depth decoder's Gumbel noise.
    Depth = 2,
}

impl From<Maya2Stream> for u64 {
    fn from(stream: Maya2Stream) -> Self {
        stream as u64
    }
}

/// A prepared clone voice: the reference's span and its ECAPA embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct Maya2ClonedVoice {
    pub reference: Maya2Reference,
    /// L2-normalised, 192 values.
    pub ecapa: Vec<f32>,
}

/// Who speaks a request.
#[derive(Debug, Clone, PartialEq)]
pub enum Maya2Voice {
    /// A built-in voice, by its exact name.
    Named(&'static str),
    Cloned(Arc<Maya2ClonedVoice>),
}

/// Everything one utterance's generation is parameterised by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Maya2GenerationParams {
    pub sampling: Maya2SamplingParams,
    /// The caller's frame limit for the piece.
    pub max_frames: Option<usize>,
    /// Positive; drives every draw.
    pub seed: u64,
}

impl Maya2GenerationParams {
    pub fn validate(&self) -> Result<()> {
        self.sampling.validate()?;
        if self.seed == 0 {
            return Err(Error::InvalidInput(
                "Maya2 generation needs a resolved positive seed".into(),
            ));
        }
        if self.max_frames == Some(0) {
            return Err(Error::InvalidInput(
                "Maya2 max_frames must be positive".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Maya2GenerationOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Every emitted frame's 32 codes, frame-major.
    pub codes: Vec<u32>,
    pub frames: usize,
    pub stop_reason: Maya2StopReason,
    pub prompt_tokens: usize,
    pub loop_boosts: usize,
    pub lm_ms: f32,
    pub codec_ms: f32,
}

/// The loaded networks of one Maya2 model.
pub struct Maya2Runtime {
    model_dir: PathBuf,
    tokenizer: Maya2Tokenizer,
    backbone: Maya2Backbone,
    inputs: Maya2Inputs,
    depth: Maya2Depth,
    mimi: MimiDecoder,
    /// Loaded on first use: only clone references need them.
    mimi_encoder: Mutex<Option<Arc<MimiEncoder>>>,
    ecapa: Mutex<Option<Arc<Maya2Ecapa>>>,
    device: Device,
}

impl Maya2Runtime {
    /// Load the backbone in `backbone_dtype`, the depth decoder in
    /// `depth_dtype`, and Mimi's decoder in F32, from `model_dir`.
    pub fn load(
        model_dir: &Path,
        device: &Device,
        backbone_dtype: DType,
        depth_dtype: DType,
    ) -> Result<Self> {
        let cfg = Maya2BackboneConfig::load(model_dir)?;
        Self::load_with_config(model_dir, cfg, device, backbone_dtype, depth_dtype)
    }

    /// [`Self::load`] with only the first `layers` backbone layers: the
    /// layer-truncated parity fixtures.
    #[cfg(test)]
    pub(crate) fn load_truncated(
        model_dir: &Path,
        device: &Device,
        dtype: DType,
        layers: usize,
    ) -> Result<Self> {
        let cfg = Maya2BackboneConfig::load(model_dir)?.truncated(layers);
        Self::load_with_config(model_dir, cfg, device, dtype, dtype)
    }

    fn load_with_config(
        model_dir: &Path,
        cfg: Maya2BackboneConfig,
        device: &Device,
        backbone_dtype: DType,
        depth_dtype: DType,
    ) -> Result<Self> {
        check_release_config(model_dir)?;
        let tokenizer = Maya2Tokenizer::load(&model_dir.join(config::TOKENIZER_FILE))?;
        let vb = mmap_release(model_dir, backbone_dtype, device)?;
        // The token table is only gathered: on the CPU it stays BF16 as
        // stored rather than widening 0.8 GB to F32.
        let token_dtype = if device.is_cpu() {
            DType::BF16
        } else {
            backbone_dtype
        };
        let tokens_vb = if token_dtype == backbone_dtype {
            vb.clone()
        } else {
            mmap_release(model_dir, token_dtype, device)?
        };
        let inputs = Maya2Inputs::load(&vb, &tokens_vb, cfg.hidden_size, cfg.vocab_size)?;
        let backbone = Maya2Backbone::load(cfg, vb.pp("backbone.lm.model"), STATE_CONTEXT_TOKENS)?;
        let depth_vb = if depth_dtype == backbone_dtype {
            vb
        } else {
            mmap_release(model_dir, depth_dtype, device)?
        };
        let depth = Maya2Depth::load(depth_vb.pp("depth"), DepthGeometry::RELEASE)?;
        let mimi = mimi::load_decoder(&model_dir.join(config::MIMI_DIR), device)?;
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            tokenizer,
            backbone,
            inputs,
            depth,
            mimi,
            mimi_encoder: Mutex::new(None),
            ecapa: Mutex::new(None),
            device: device.clone(),
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn backbone(&self) -> &Maya2Backbone {
        &self.backbone
    }

    pub fn inputs(&self) -> &Maya2Inputs {
        &self.inputs
    }

    pub fn depth(&self) -> &Maya2Depth {
        &self.depth
    }

    pub fn mimi(&self) -> &MimiDecoder {
        &self.mimi
    }

    pub fn tokenizer(&self) -> &Maya2Tokenizer {
        &self.tokenizer
    }

    fn mimi_encoder(&self) -> Result<Arc<MimiEncoder>> {
        let mut slot = self
            .mimi_encoder
            .lock()
            .map_err(|_| Error::InferenceError("Maya2 Mimi encoder lock poisoned".into()))?;
        if let Some(encoder) = slot.as_ref() {
            return Ok(encoder.clone());
        }
        let encoder = Arc::new(mimi::load_encoder(
            &self.model_dir.join(config::MIMI_DIR),
            &self.device,
        )?);
        *slot = Some(encoder.clone());
        Ok(encoder)
    }

    fn ecapa(&self) -> Result<Arc<Maya2Ecapa>> {
        let mut slot = self
            .ecapa
            .lock()
            .map_err(|_| Error::InferenceError("Maya2 ECAPA lock poisoned".into()))?;
        if let Some(ecapa) = slot.as_ref() {
            return Ok(ecapa.clone());
        }
        let ecapa = Arc::new(Maya2Ecapa::load(
            &self.model_dir.join(config::ECAPA_WEIGHTS_FILE),
        )?);
        *slot = Some(ecapa.clone());
        Ok(ecapa)
    }

    /// Prepare a clone voice from a reference clip: 24 kHz mono `samples`
    /// (0.5–30 s), its exact `transcript` and its `language`.
    pub fn prepare_clone(
        &self,
        samples: &[f32],
        transcript: &str,
        language: &'static str,
    ) -> Result<Maya2ClonedVoice> {
        let seconds = samples.len() as f32 / SAMPLE_RATE as f32;
        if !(config::MIN_REFERENCE_SECONDS..=config::MAX_REFERENCE_SECONDS).contains(&seconds) {
            return Err(Error::InvalidInput(format!(
                "Maya2 reference audio must last {}–{} s, got {seconds:.2} s",
                config::MIN_REFERENCE_SECONDS,
                config::MAX_REFERENCE_SECONDS
            )));
        }
        if transcript.trim().is_empty() {
            return Err(Error::InvalidInput(
                "Maya2 clones need the reference audio's exact transcript".into(),
            ));
        }
        let codes = self.mimi_encoder()?.encode(samples)?;
        let ecapa = self.ecapa()?.embed_24k(samples)?;
        Ok(Maya2ClonedVoice {
            reference: Maya2Reference {
                text: transcript.trim().to_string(),
                language,
                codes,
            },
            ecapa,
        })
    }

    /// The prompt of `voice` speaking `text` in `language`.
    pub fn prompt(&self, voice: &Maya2Voice, language: &str, text: &str) -> Result<Maya2Prompt> {
        match voice {
            Maya2Voice::Named(name) => self.tokenizer.named_prompt(name, language, text),
            Maya2Voice::Cloned(clone) => {
                self.tokenizer
                    .clone_prompt(&clone.reference, language, text)
            }
        }
    }

    /// The prompt's embeddings, the speaker's anchor and its depth cross
    /// constants `[4, 1024]`.
    pub fn conditioning(
        &self,
        voice: &Maya2Voice,
        prompt: &Maya2Prompt,
    ) -> Result<(Tensor, Tensor)> {
        let (ecapa, codes): (Option<&[f32]>, &[u32]) = match voice {
            Maya2Voice::Named(_) => (None, &[]),
            Maya2Voice::Cloned(clone) => (Some(&clone.ecapa), &clone.reference.codes),
        };
        let anchor = self.inputs.anchor(ecapa)?;
        let embeds = self.inputs.compose(prompt, codes, &anchor)?;
        let ecapa = ecapa
            .map(|values| Tensor::from_slice(values, values.len(), &self.device))
            .transpose()?;
        let cross = self.depth.cross_constants(ecapa.as_ref())?;
        Ok((embeds, cross))
    }

    /// A scratch cache on this model's device for `tokens` positions.
    pub(crate) fn scratch_cache(&self, tokens: usize) -> Result<PhysicalPagedKvCache> {
        let cfg = self.backbone.config();
        PhysicalPagedKvCache::scratch_for_probe(
            &self.device,
            self.backbone.dtype(),
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            tokens,
        )
    }

    /// One utterance end to end on `cache`: prefill, frames to the stop or
    /// the budget, then Mimi over the frames in chunks.
    pub fn generate_physical(
        &self,
        voice: &Maya2Voice,
        language: &str,
        text: &str,
        params: &Maya2GenerationParams,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<Maya2GenerationOutput> {
        params.validate()?;
        let prompt = self.prompt(voice, language, text)?;
        let budget = frame_budget(text, language, params.max_frames, prompt.items.len())?;
        let (embeds, cross) = self.conditioning(voice, &prompt)?;
        let cross = cross.unsqueeze(0)?;
        let started = Instant::now();
        let mut hidden = self.backbone.forward(&mut [Maya2BackboneRow {
            embeds,
            cache: &mut *cache,
        }])?;
        let mut cb0_rng = SplitMix64::for_stream(params.seed, Maya2Stream::Cb0);
        let mut depth_rng = SplitMix64::for_stream(params.seed, Maya2Stream::Depth);
        let mut cb0_state = Cb0State::default();
        let mut guard = LoopGuard::new(suspend_windows(text, language));
        let mut codes = Vec::new();
        let mut frames = 0usize;
        let stop_reason = loop {
            let logits = self.inputs.cb0_logits(&hidden)?.to_vec2::<f32>()?;
            let draw = sample_cb0(
                &logits[0],
                &params.sampling,
                guard.aeos_bias(),
                frames,
                budget.minimum,
                &mut cb0_state,
                &mut cb0_rng,
            )?;
            if let Some(reason) = draw.stop {
                break reason;
            }
            let gumbel = depth_gumbel(
                &mut [&mut depth_rng],
                DEPTH_LEVELS,
                CODEBOOK_SIZE,
                &self.device,
            )?;
            let cb0 = Tensor::new(&[draw.code], &self.device)?;
            let frame = self.depth.generate_frame(
                &hidden,
                &cb0,
                &cross,
                &gumbel,
                MAYA2_DEFAULTS.depth_temperature,
                MAYA2_DEFAULTS.depth_top_k,
            )?;
            codes.extend(frame.to_vec2::<u32>()?.remove(0));
            frames += 1;
            if frames >= budget.cap {
                break cap_reason(&budget, params.max_frames);
            }
            if guard.observe(draw.code, frames) {
                break Maya2StopReason::LoopBackstop;
            }
            let embeds = self.inputs.audio_frames(&frame)?;
            hidden = self.backbone.forward(&mut [Maya2BackboneRow {
                embeds,
                cache: &mut *cache,
            }])?;
        };
        let lm_ms = started.elapsed().as_secs_f32() * 1000.0;
        if frames == 0 {
            return Err(Error::InferenceError(
                "Maya2 generated no audio frames".into(),
            ));
        }
        let codec_started = Instant::now();
        let mut state = self.mimi.new_state()?;
        let mut samples = Vec::with_capacity(frames * self.mimi.samples_per_frame());
        for chunk in codes.chunks(AUDIO_CHUNK_FRAMES * CODEBOOKS) {
            samples.extend(
                self.mimi
                    .decode_stream(&mut state, chunk)?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(|sample| sample.clamp(-1.0, 1.0)),
            );
        }
        Ok(Maya2GenerationOutput {
            samples,
            sample_rate: SAMPLE_RATE,
            frames,
            codes,
            stop_reason,
            prompt_tokens: prompt.items.len(),
            loop_boosts: guard.boosts,
            lm_ms,
            codec_ms: codec_started.elapsed().as_secs_f32() * 1000.0,
        })
    }
}

/// Frames Mimi decodes per call when the whole utterance is at hand.
const AUDIO_CHUNK_FRAMES: usize = 8;

/// A piece's budget, clamped to what its prompt leaves of the context.
pub(crate) fn frame_budget(
    text: &str,
    language: &str,
    max_frames: Option<usize>,
    prompt_positions: usize,
) -> Result<Maya2PieceBudget> {
    let mut budget = piece_budget(text, language, max_frames);
    let room = STATE_CONTEXT_TOKENS.saturating_sub(prompt_positions + 2);
    if room < MIN_FRAME_BUDGET {
        return Err(Error::InvalidInput(format!(
            "Maya2 prompt of {prompt_positions} positions leaves no room for audio in its \
             {STATE_CONTEXT_TOKENS}-position context (shorten the text or the reference)"
        )));
    }
    budget.cap = budget.cap.min(room);
    Ok(budget)
}

/// Why a piece that reached its cap stopped.
pub(crate) fn cap_reason(budget: &Maya2PieceBudget, max_frames: Option<usize>) -> Maya2StopReason {
    if max_frames.is_some_and(|limit| budget.cap >= limit) {
        Maya2StopReason::MaxFrames
    } else if budget.cap < budget.loose_cap {
        Maya2StopReason::CapGuard
    } else {
        Maya2StopReason::Cap
    }
}
