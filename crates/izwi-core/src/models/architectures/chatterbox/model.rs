//! The engine-facing Chatterbox model: load, state contract, and the
//! retained prefill / decode / acoustic steps the executor drives.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

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
    config, ChatterboxGenerationParams, ChatterboxPreparedVoice, ChatterboxRuntime,
    S3GEN_SAMPLE_RATE,
};
use crate::backends::DeviceProfile;
use crate::catalog::ModelVariant;
use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::models::shared::attention::paged::default_kv_page_size;

/// The name of the one built-in voice (`conds.pt`).
pub const CHATTERBOX_DEFAULT_VOICE: &str = "default";

static NEXT_CHATTERBOX_MODEL_IDENTITY: AtomicU64 = AtomicU64::new(1);

pub struct ChatterboxTtsModel {
    model_identity: u64,
    variant: ModelVariant,
    runtime: ChatterboxRuntime,
    default_voice: Arc<ChatterboxPreparedVoice>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatterboxTtsDiagnostics {
    pub model_family: &'static str,
    pub variant: String,
    pub sample_rate: u32,
    pub dtype: String,
    pub context_tokens: usize,
    pub max_speech_tokens: usize,
}

impl ChatterboxTtsModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        if variant != ModelVariant::Chatterbox500M {
            return Err(Error::InvalidInput(format!(
                "Unsupported Chatterbox variant: {variant}"
            )));
        }
        // F32 everywhere until the per-component precision gates pass.
        let runtime = ChatterboxRuntime::load(model_dir, &device.device, DType::F32)?;
        let default_voice = Arc::new(runtime.default_voice().clone());
        Ok(Self {
            model_identity: NEXT_CHATTERBOX_MODEL_IDENTITY.fetch_add(1, Ordering::Relaxed),
            variant,
            runtime,
            default_voice,
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

    /// S3Gen and HiFT for a request whose T3 has stopped; marks it
    /// finished. Returns the samples at 24 kHz.
    pub(crate) fn retained_acoustic_finalize(
        &self,
        state: &mut ChatterboxRetainedState,
    ) -> Result<Vec<f32>> {
        if state.phase() != ChatterboxPhase::Acoustic {
            return Err(Error::InvalidInput(
                "Chatterbox acoustic stage is not ready".into(),
            ));
        }
        let tokens = state.renderable_tokens(self.runtime.t3())?;
        let (samples, _) =
            self.runtime
                .render(&state.artifact.voice, &tokens, state.params.seed)?;
        state.mark_finished()?;
        Ok(samples)
    }
}
