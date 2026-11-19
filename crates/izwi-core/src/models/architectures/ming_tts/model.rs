//! The engine-facing Ming model: load, state contract, and the retained
//! prefill / decode / audio-window steps the executor drives.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use candle_core::DType;
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::config::{vae_window_keys, CONTEXT_POSITIONS, MAX_NEW_LATENTS, SAMPLE_RATE};
use super::physical::{ming_physical_state_spec, ming_retained_contract, MingPhysicalStateSpec};
use super::retained::{
    ming_decode_batch, ming_prefill_batch, MingDecodeStep, MingPhase, MingPrefillStep,
    MingPreparedArtifact, MingRetainedState, MingStepRow,
};
use super::{MingGenerationParams, MingReferenceVoice, MingRequest, MingRuntime};
use crate::backends::{BackendKind, DTypeSelection, DTypeSelectionRequest, DeviceProfile};
use crate::catalog::{ModelFamily, ModelVariant, MING_TTS_VOICES};
use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::kernels::numeric_check::load_with_precision_fallback;
use crate::models::shared::attention::paged::default_kv_page_size;

static NEXT_MING_MODEL_IDENTITY: AtomicU64 = AtomicU64::new(1);

/// Encoded reference voices kept for reuse: a saved voice is encoded once.
const REFERENCE_CACHE_VOICES: usize = 16;

pub struct MingTtsModel {
    model_identity: u64,
    variant: ModelVariant,
    runtime: MingRuntime,
    /// (digest of the clip and its transcript, the encoded voice), most
    /// recently used last.
    references: Mutex<Vec<([u8; 32], Arc<MingReferenceVoice>)>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MingTtsDiagnostics {
    pub model_family: &'static str,
    pub variant: String,
    pub sample_rate: u32,
    /// The language model's and flow path's dtype.
    pub dtype: String,
    /// The VAE's dtype.
    pub vae_dtype: String,
    pub context_positions: usize,
    pub max_new_tokens: usize,
    /// Keys a VAE frame attends to, itself included.
    pub vae_window_keys: usize,
    pub voice_cloning: bool,
}

/// What one audio-window quantum did.
pub(crate) enum MingAudioProgress {
    /// A window ran: these samples are settled (possibly none).
    Window(Vec<f32>),
    /// No window was due.
    Idle,
}

/// The dtype the language model and the flow path load in on `device`, and
/// why: the device's Ming policy, which `IZWI_METAL_DTYPE` overrides on
/// Metal.
pub fn ming_lm_precision(device: &DeviceProfile) -> DTypeSelection {
    device.resolve_dtype(DTypeSelectionRequest::new(None).with_model_family(ModelFamily::MingTts))
}

/// Whether a model loaded in `dtype` reloads in F32 when its probe fails:
/// on every backend (the F32 weights of the probed parts are 2.8 GB), and
/// never from F32 itself.
pub(crate) fn ming_reloads_in_f32(_device: &DeviceProfile, dtype: DType) -> bool {
    dtype != DType::F32
}

impl MingTtsModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        if variant.family() != ModelFamily::MingTts {
            return Err(Error::InvalidInput(format!(
                "Unsupported Ming TTS variant: {variant}"
            )));
        }
        // The language model, aggregator and flow head at the device's Ming
        // dtype (F16 on Metal, BF16 on CUDA) when the load-time probe
        // passes, else F32. The VAE is F32 everywhere: its output is a
        // spectrum whose magnitudes span several orders, and it runs once
        // per audio window.
        let selection = ming_lm_precision(&device);
        let (runtime, precision) = load_with_precision_fallback(
            "Ming language model and flow path",
            &selection,
            ming_reloads_in_f32(&device, selection.dtype),
            DType::F32,
            |dtype| MingRuntime::load(model_dir, &device.device, dtype, DType::F32),
            MingRuntime::numeric_self_check,
        )?;
        tracing::info!(
            "Ming TTS dtype on {:?}: {:?} ({}); VAE F32, {} keys per frame",
            BackendKind::from(device.kind),
            precision.loaded,
            precision.reason,
            vae_window_keys(),
        );
        Ok(Self {
            model_identity: NEXT_MING_MODEL_IDENTITY.fetch_add(1, Ordering::Relaxed),
            variant,
            runtime,
            references: Mutex::new(Vec::new()),
        })
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    pub fn model_identity(&self) -> u64 {
        self.model_identity
    }

    pub fn runtime(&self) -> &MingRuntime {
        &self.runtime
    }

    /// The built-in voices' request names.
    pub fn available_speakers(&self) -> Vec<String> {
        MING_TTS_VOICES
            .iter()
            .map(|voice| voice.name.to_string())
            .collect()
    }

    pub fn context_tokens(&self) -> usize {
        CONTEXT_POSITIONS
    }

    pub fn diagnostics(&self) -> MingTtsDiagnostics {
        MingTtsDiagnostics {
            model_family: "ming_tts",
            variant: self.variant.to_string(),
            sample_rate: SAMPLE_RATE,
            dtype: format!("{:?}", self.runtime.lm_dtype()),
            vae_dtype: format!("{:?}", self.runtime.vae_dtype()),
            context_positions: self.context_tokens(),
            max_new_tokens: MAX_NEW_LATENTS,
            vae_window_keys: self.runtime.decoder().window_keys(),
            voice_cloning: self.runtime.can_clone_voices(),
        }
    }

    /// Bytes of one retained K/V element.
    pub(crate) fn state_element_bytes(&self) -> usize {
        self.runtime.lm_dtype().size_in_bytes()
    }

    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<MingPhysicalStateSpec> {
        let retained = ming_retained_contract(
            self.runtime.trunk().config(),
            self.runtime.lm_dtype(),
            default_kv_page_size(),
        )?;
        ming_physical_state_spec(stage_graphs, retained, self.context_tokens())
    }

    /// Encode a reference clip for cloning (mono samples at `sample_rate`).
    pub fn encode_reference(
        &self,
        samples: &[f32],
        sample_rate: u32,
        transcript: Option<&str>,
    ) -> Result<MingReferenceVoice> {
        self.runtime
            .encode_reference(samples, sample_rate, transcript)
    }

    /// The encoded voice of a reference clip given as the request carries
    /// it (`audio`, base64) with its optional transcript. `decode` turns the
    /// clip into mono samples and their rate; it runs only for a clip not
    /// seen before, and the VAE and speaker encoder with it.
    pub(crate) fn reference_voice(
        &self,
        audio: &str,
        transcript: Option<&str>,
        decode: impl FnOnce() -> Result<(Vec<f32>, u32)>,
    ) -> Result<Arc<MingReferenceVoice>> {
        let transcript = transcript.map(str::trim).filter(|text| !text.is_empty());
        let mut hasher = Sha256::new();
        hasher.update(audio.as_bytes());
        hasher.update([0u8]);
        hasher.update(transcript.unwrap_or("").as_bytes());
        let digest: [u8; 32] = hasher.finalize().into();
        let cached = |references: &mut Vec<([u8; 32], Arc<MingReferenceVoice>)>| {
            let index = references.iter().position(|(key, _)| *key == digest)?;
            let entry = references.remove(index);
            let voice = entry.1.clone();
            references.push(entry);
            Some(voice)
        };
        let lock = || {
            self.references.lock().map_err(|_| {
                Error::InferenceError("Ming reference voice cache mutex poisoned".into())
            })
        };
        if let Some(voice) = cached(&mut *lock()?) {
            return Ok(voice);
        }
        let (samples, sample_rate) = decode()?;
        let voice = Arc::new(self.encode_reference(&samples, sample_rate, transcript)?);
        let mut references = lock()?;
        if let Some(voice) = cached(&mut references) {
            return Ok(voice);
        }
        if references.len() >= REFERENCE_CACHE_VOICES {
            references.remove(0);
        }
        references.push((digest, voice.clone()));
        Ok(voice)
    }

    /// A request to its prompt, checked against the context.
    pub(crate) fn prepare_retained_artifact(
        &self,
        request: &MingRequest,
    ) -> Result<Arc<MingPreparedArtifact>> {
        Ok(Arc::new(MingPreparedArtifact {
            prompt: self.runtime.prompt(request)?,
        }))
    }

    pub(crate) fn new_retained_state(
        &self,
        artifact: Arc<MingPreparedArtifact>,
        params: MingGenerationParams,
    ) -> Result<MingRetainedState> {
        MingRetainedState::new(artifact, params, self.context_tokens())
    }

    /// Whether a request has an audio window to decode now.
    pub(crate) fn retained_window_due(&self, state: &MingRetainedState) -> bool {
        state.next_window().is_some()
    }

    pub(crate) fn retained_prefill_batch(
        &self,
        rows: &mut [MingStepRow<'_>],
        tokens: &[usize],
    ) -> Result<Vec<MingPrefillStep>> {
        ming_prefill_batch(&self.runtime, rows, tokens)
    }

    pub(crate) fn retained_decode_batch(
        &self,
        rows: &mut [MingStepRow<'_>],
    ) -> Result<Vec<MingDecodeStep>> {
        ming_decode_batch(&self.runtime, rows)
    }

    /// Decode the request's next audio window, waiting for the device so
    /// the window's cost lands in its own quantum.
    pub(crate) fn retained_audio_step(
        &self,
        state: &mut MingRetainedState,
    ) -> Result<MingAudioProgress> {
        let samples = state.decode_window(self.runtime.decoder())?;
        self.runtime.device().synchronize()?;
        Ok(match samples {
            Some(samples) => MingAudioProgress::Window(samples),
            None => MingAudioProgress::Idle,
        })
    }

    /// [`Self::retained_audio_step`] for several requests: each request's
    /// window decodes on its own (the VAE keeps per-request attention
    /// state), so every request gets the samples it would alone.
    pub(crate) fn retained_audio_batch(
        &self,
        states: &mut [&mut MingRetainedState],
    ) -> Result<Vec<MingAudioProgress>> {
        states
            .iter_mut()
            .map(|state| self.retained_audio_step(state))
            .collect()
    }

    /// Mark a request whose last window is out as finished.
    pub(crate) fn retained_finalize(&self, state: &mut MingRetainedState) -> Result<()> {
        if state.phase() != MingPhase::Audio {
            return Err(Error::InvalidInput(
                "Ming finalized before the model stopped".into(),
            ));
        }
        if state.audio.samples() == 0 {
            return Err(Error::InferenceError("Ming generated no audio".into()));
        }
        state.mark_finished()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_voice_list_is_the_catalog_table() {
        assert_eq!(MING_TTS_VOICES.len(), 85);
        assert!(CONTEXT_POSITIONS > crate::models::architectures::ming_tts::MIN_LATENTS);
    }
}
