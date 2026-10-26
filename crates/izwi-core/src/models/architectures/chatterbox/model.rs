//! The engine-facing Chatterbox model: load, state contract, and the
//! retained prefill / decode / acoustic steps the executor drives.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use candle_core::DType;
use serde::Serialize;

use super::physical::{
    chatterbox_physical_state_spec, chatterbox_retained_contract, ChatterboxPhysicalStateSpec,
};
use super::retained::{
    chatterbox_decode_batch, chatterbox_prefill_batch, ChatterboxDecodeStep, ChatterboxPhase,
    ChatterboxPrefillStep, ChatterboxPreparedArtifact, ChatterboxRetainedState, ChatterboxStepRow,
};
use super::{
    config, ChatterboxAcoustic, ChatterboxGenerationParams, ChatterboxPreparedVoice,
    ChatterboxRuntime, S3GEN_SAMPLE_RATE,
};
use crate::backends::DeviceProfile;
use crate::catalog::ModelVariant;
use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::models::shared::attention::paged::default_kv_page_size;

/// The name of the one built-in voice (`conds.pt`).
pub const CHATTERBOX_DEFAULT_VOICE: &str = "default";

/// Voices prepared from reference audio that a model keeps for reuse, least
/// recently used first out. Each holds a few hundred KB on the device.
pub const CHATTERBOX_REFERENCE_VOICE_CACHE: usize = 16;

static NEXT_CHATTERBOX_MODEL_IDENTITY: AtomicU64 = AtomicU64::new(1);

pub struct ChatterboxTtsModel {
    model_identity: u64,
    variant: ModelVariant,
    runtime: ChatterboxRuntime,
    default_voice: Arc<ChatterboxPreparedVoice>,
    /// Voices prepared from reference audio, keyed by a digest of the
    /// reference payload. The model is one checkpoint on one device at one
    /// dtype, so unloading it drops every entry.
    reference_voices: Mutex<VecDeque<([u8; 32], Arc<ChatterboxPreparedVoice>)>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatterboxTtsDiagnostics {
    pub model_family: &'static str,
    pub variant: String,
    pub sample_rate: u32,
    pub dtype: String,
    pub context_tokens: usize,
    pub max_speech_tokens: usize,
    /// Rendered audio carries the PerTh watermark.
    pub watermarked: bool,
}

impl ChatterboxTtsModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        if variant != ModelVariant::Chatterbox500M {
            return Err(Error::InvalidInput(format!(
                "Unsupported Chatterbox variant: {variant}"
            )));
        }
        // F32 everywhere until the per-component precision gates pass.
        let watermark = crate::env::flag_or("IZWI_CHATTERBOX_WATERMARK", true);
        if !watermark {
            tracing::warn!(
                "IZWI_CHATTERBOX_WATERMARK is false: Chatterbox audio is served without its PerTh watermark"
            );
        }
        let runtime = ChatterboxRuntime::load(model_dir, &device.device, DType::F32, watermark)?;
        let default_voice = Arc::new(runtime.default_voice().clone());
        Ok(Self {
            model_identity: NEXT_CHATTERBOX_MODEL_IDENTITY.fetch_add(1, Ordering::Relaxed),
            variant,
            runtime,
            default_voice,
            reference_voices: Mutex::new(VecDeque::new()),
        })
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    pub fn model_identity(&self) -> u64 {
        self.model_identity
    }

    pub fn runtime(&self) -> &ChatterboxRuntime {
        &self.runtime
    }

    pub fn available_speakers(&self) -> Vec<String> {
        vec![CHATTERBOX_DEFAULT_VOICE.to_string()]
    }

    pub fn default_voice(&self) -> Arc<ChatterboxPreparedVoice> {
        self.default_voice.clone()
    }

    /// The voice of a reference payload with `digest`: cached, or prepared
    /// from the audio `decode` returns (mono samples and their rate).
    pub(crate) fn reference_voice(
        &self,
        digest: [u8; 32],
        decode: impl FnOnce() -> Result<(Vec<f32>, u32)>,
    ) -> Result<Arc<ChatterboxPreparedVoice>> {
        if let Some(voice) = self.take_cached_voice(&digest) {
            return Ok(voice);
        }
        let (samples, sample_rate) = decode()?;
        let voice = Arc::new(
            self.runtime
                .prepare_voice_from_audio(&samples, sample_rate)?,
        );
        let mut voices = self
            .reference_voices
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !voices.iter().any(|(cached, _)| *cached == digest) {
            if voices.len() == CHATTERBOX_REFERENCE_VOICE_CACHE {
                voices.pop_front();
            }
            voices.push_back((digest, voice.clone()));
        }
        Ok(voice)
    }

    /// A cached voice, moved to the most recently used end.
    fn take_cached_voice(&self, digest: &[u8; 32]) -> Option<Arc<ChatterboxPreparedVoice>> {
        let mut voices = self
            .reference_voices
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let index = voices.iter().position(|(cached, _)| cached == digest)?;
        let entry = voices.remove(index)?;
        let voice = entry.1.clone();
        voices.push_back(entry);
        Some(voice)
    }

    /// Positions one T3 sequence may reach: conditioning, the longest text
    /// one request takes, both start tokens and the speech budget.
    pub fn context_tokens(&self) -> usize {
        config::STATE_CONTEXT_TOKENS
    }

    pub fn diagnostics(&self) -> ChatterboxTtsDiagnostics {
        ChatterboxTtsDiagnostics {
            model_family: "chatterbox_tts",
            variant: self.variant.to_string(),
            sample_rate: S3GEN_SAMPLE_RATE,
            dtype: format!("{:?}", self.runtime.dtype()),
            context_tokens: self.context_tokens(),
            max_speech_tokens: ModelVariant::CHATTERBOX_MAX_OUTPUT_FRAMES,
            watermarked: self.runtime.watermarked(),
        }
    }

    /// Bytes of one retained K/V element.
    pub(crate) fn state_element_bytes(&self) -> usize {
        self.runtime.dtype().size_in_bytes()
    }

    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<ChatterboxPhysicalStateSpec> {
        let retained = chatterbox_retained_contract(
            self.runtime.t3().config(),
            self.runtime.dtype(),
            default_kv_page_size(),
        )?;
        chatterbox_physical_state_spec(stage_graphs, retained, self.context_tokens())
    }

    /// Text and voice to prefill rows, checked against the context.
    pub(crate) fn prepare_retained_artifact(
        &self,
        text: &str,
        voice: Option<Arc<ChatterboxPreparedVoice>>,
        params: &ChatterboxGenerationParams,
    ) -> Result<Arc<ChatterboxPreparedArtifact>> {
        if text.trim().is_empty() {
            return Err(Error::InvalidInput(
                "Chatterbox needs non-empty text".into(),
            ));
        }
        let voice = voice.unwrap_or_else(|| self.default_voice.clone());
        let artifact = self.runtime.prepare_artifact(text, voice, params)?;
        if artifact.text_tokens > config::MAX_TEXT_TOKENS {
            return Err(Error::InvalidInput(format!(
                "Chatterbox text has {} tokens; one request takes at most {} (split longer text)",
                artifact.text_tokens,
                config::MAX_TEXT_TOKENS
            )));
        }
        Ok(Arc::new(artifact))
    }

    pub(crate) fn new_retained_state(
        &self,
        artifact: Arc<ChatterboxPreparedArtifact>,
        params: ChatterboxGenerationParams,
    ) -> Result<ChatterboxRetainedState> {
        ChatterboxRetainedState::new(
            artifact,
            params,
            self.runtime.t3().config().start_speech_token,
            self.context_tokens(),
        )
    }

    pub(crate) fn retained_prefill_batch(
        &self,
        rows: &mut [ChatterboxStepRow<'_>],
        tokens: &[usize],
    ) -> Result<Vec<ChatterboxPrefillStep>> {
        chatterbox_prefill_batch(self.runtime.t3(), rows, tokens)
    }

    pub(crate) fn retained_decode_batch(
        &self,
        rows: &mut [ChatterboxStepRow<'_>],
    ) -> Result<Vec<ChatterboxDecodeStep>> {
        chatterbox_decode_batch(self.runtime.t3(), rows)
    }

    /// One acoustic unit (S3Gen encoder, an Euler step, or HiFT) for a
    /// request whose T3 has stopped. Returns true once its waveform is
    /// rendered.
    pub(crate) fn retained_acoustic_step(
        &self,
        state: &mut ChatterboxRetainedState,
    ) -> Result<bool> {
        if state.phase() != ChatterboxPhase::Acoustic {
            return Err(Error::InvalidInput(
                "Chatterbox acoustic stage is not ready".into(),
            ));
        }
        let tokens = state.renderable_tokens(self.runtime.t3())?;
        self.runtime.acoustic_step(
            &state.artifact.voice,
            &tokens,
            state.params.seed,
            &mut state.acoustic,
        )
    }

    /// Hand over a rendered waveform (24 kHz) and mark the request finished.
    pub(crate) fn retained_acoustic_finalize(
        &self,
        state: &mut ChatterboxRetainedState,
    ) -> Result<Vec<f32>> {
        if state.phase() != ChatterboxPhase::Acoustic
            || !matches!(state.acoustic, ChatterboxAcoustic::Rendered(_))
        {
            return Err(Error::InvalidInput(
                "Chatterbox finalized before its waveform was rendered".into(),
            ));
        }
        let ChatterboxAcoustic::Rendered(samples) =
            std::mem::replace(&mut state.acoustic, ChatterboxAcoustic::Pending)
        else {
            unreachable!("checked above");
        };
        state.mark_finished()?;
        Ok(samples)
    }
}
