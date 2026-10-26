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
use super::stream::{ChatterboxStreamUnit, CHATTERBOX_STREAM_PLAN};
use super::{
    config, ChatterboxAcoustic, ChatterboxEdition, ChatterboxGenerationParams,
    ChatterboxPreparedVoice, ChatterboxRuntime, S3GEN_SAMPLE_RATE,
};
use crate::backends::{DTypeSelection, DTypeSelectionRequest, DeviceProfile};
use crate::catalog::{ModelFamily, ModelVariant};
use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::kernels::numeric_check::load_with_precision_fallback;
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

/// What one acoustic unit did.
pub(crate) enum ChatterboxAcousticProgress {
    /// More units of this window or utterance follow.
    Working,
    /// A streamed window finished; these samples are settled.
    Window(Vec<f32>),
    /// The final-only waveform is rendered.
    Rendered,
}

/// The dtype T3 loads in on `device`, and why: the device's Chatterbox
/// policy, which `IZWI_METAL_DTYPE` overrides on Metal.
pub fn chatterbox_t3_precision(device: &DeviceProfile) -> DTypeSelection {
    device.resolve_dtype(
        DTypeSelectionRequest::new(None).with_model_family(ModelFamily::ChatterboxTts),
    )
}

impl ChatterboxTtsModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        let edition = ChatterboxEdition::of(variant).ok_or_else(|| {
            Error::InvalidInput(format!("Unsupported Chatterbox variant: {variant}"))
        })?;
        let watermark = crate::env::flag_or("IZWI_CHATTERBOX_WATERMARK", true);
        if !watermark {
            tracing::warn!(
                "IZWI_CHATTERBOX_WATERMARK is false: Chatterbox audio is served without its PerTh watermark"
            );
        }
        // T3 at the device's Chatterbox dtype (F16 on Metal, BF16 on CUDA
        // with BF16) when its load-time self-check passes, else F32; every
        // other network stays F32.
        let selection = chatterbox_t3_precision(&device);
        let (runtime, precision) = load_with_precision_fallback(
            "Chatterbox T3",
            &selection,
            true,
            DType::F32,
            |t3_dtype| {
                ChatterboxRuntime::load_edition(
                    model_dir,
                    edition,
                    &device.device,
                    t3_dtype,
                    watermark,
                )
            },
            ChatterboxRuntime::numeric_self_check,
        )?;
        tracing::info!(
            "Chatterbox T3 dtype on {:?}: {:?} ({})",
            device.kind,
            precision.loaded,
            precision.reason
        );
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
            dtype: format!("{:?}", self.runtime.t3_dtype()),
            context_tokens: self.context_tokens(),
            max_speech_tokens: ModelVariant::CHATTERBOX_MAX_OUTPUT_FRAMES,
            watermarked: self.runtime.watermarked(),
        }
    }

    /// Bytes of one retained K/V element.
    pub(crate) fn state_element_bytes(&self) -> usize {
        self.runtime.t3_dtype().size_in_bytes()
    }

    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<ChatterboxPhysicalStateSpec> {
        let retained = chatterbox_retained_contract(
            self.runtime.t3().config(),
            self.runtime.t3_dtype(),
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

    /// A request's state; a `streaming` one renders windows while T3 runs.
    pub(crate) fn new_retained_state(
        &self,
        artifact: Arc<ChatterboxPreparedArtifact>,
        params: ChatterboxGenerationParams,
        streaming: bool,
    ) -> Result<ChatterboxRetainedState> {
        let stream = streaming.then(|| self.runtime.stream(CHATTERBOX_STREAM_PLAN, params.seed));
        ChatterboxRetainedState::new(
            artifact,
            params,
            self.runtime.t3().config().start_speech_token,
            self.context_tokens(),
            stream,
        )
    }

    /// Whether a streamed request has a window to render now.
    pub(crate) fn retained_window_due(&self, state: &ChatterboxRetainedState) -> bool {
        state.stream.as_ref().is_some_and(|stream| {
            let available = state.streamable_tokens(self.runtime.t3()).len();
            self.runtime
                .stream_window_due(stream, available, state.stopped())
        })
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

    /// One acoustic unit (S3Gen encoder, an Euler step, or HiFT): of a
    /// streamed request's current window, or of a final-only request's
    /// whole utterance once T3 has stopped.
    pub(crate) fn retained_acoustic_step(
        &self,
        state: &mut ChatterboxRetainedState,
    ) -> Result<ChatterboxAcousticProgress> {
        if let Some(mut stream) = state.stream.take() {
            let tokens = state.streamable_tokens(self.runtime.t3());
            let unit = self.runtime.stream_step(
                &state.artifact.voice,
                &tokens,
                &mut stream,
                state.stopped(),
            );
            state.stream = Some(stream);
            return Ok(match unit? {
                ChatterboxStreamUnit::Rendering => ChatterboxAcousticProgress::Working,
                ChatterboxStreamUnit::Rendered(samples) => {
                    ChatterboxAcousticProgress::Window(samples)
                }
            });
        }
        if state.phase() != ChatterboxPhase::Acoustic {
            return Err(Error::InvalidInput(
                "Chatterbox acoustic stage is not ready".into(),
            ));
        }
        let tokens = state.renderable_tokens(self.runtime.t3())?;
        let rendered = self.runtime.acoustic_step(
            &state.artifact.voice,
            &tokens,
            state.params.seed,
            &mut state.acoustic,
        )?;
        Ok(if rendered {
            ChatterboxAcousticProgress::Rendered
        } else {
            ChatterboxAcousticProgress::Working
        })
    }

    /// Hand over a rendered waveform (24 kHz) and mark the request finished;
    /// a stream has handed over everything with its last window.
    pub(crate) fn retained_acoustic_finalize(
        &self,
        state: &mut ChatterboxRetainedState,
    ) -> Result<Vec<f32>> {
        if let Some(stream) = &state.stream {
            if state.phase() != ChatterboxPhase::Acoustic || !stream.done() {
                return Err(Error::InvalidInput(
                    "Chatterbox stream finalized before its last window".into(),
                ));
            }
            state.mark_finished()?;
            return Ok(Vec::new());
        }
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
