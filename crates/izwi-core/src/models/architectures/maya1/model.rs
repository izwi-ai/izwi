//! The engine-facing Maya1 model: load, state contract, and the retained
//! prefill / decode / audio-window steps the executor drives.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use candle_core::DType;
use serde::Serialize;

use super::config::{self, MAYA1_VOICE_PRESETS, SAMPLE_RATE, STATE_CONTEXT_TOKENS};
use super::physical::{maya1_physical_state_spec, maya1_retained_contract, Maya1PhysicalStateSpec};
use super::retained::{
    maya1_decode_batch, maya1_prefill_batch, Maya1DecodeStep, Maya1Phase, Maya1PrefillStep,
    Maya1PreparedArtifact, Maya1RetainedState, Maya1StepRow,
};
use super::{Maya1GenerationParams, Maya1Runtime};
use crate::backends::{BackendKind, DTypeSelection, DTypeSelectionRequest, DeviceProfile};
use crate::catalog::{ModelFamily, ModelVariant};
use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::kernels::numeric_check::load_with_precision_fallback;
use crate::models::shared::attention::paged::default_kv_page_size;

static NEXT_MAYA1_MODEL_IDENTITY: AtomicU64 = AtomicU64::new(1);

pub struct Maya1TtsModel {
    model_identity: u64,
    variant: ModelVariant,
    runtime: Maya1Runtime,
}

#[derive(Debug, Clone, Serialize)]
pub struct Maya1TtsDiagnostics {
    pub model_family: &'static str,
    pub variant: String,
    pub sample_rate: u32,
    pub dtype: String,
    pub context_tokens: usize,
    pub max_new_tokens: usize,
}

/// What one audio-window quantum did.
pub(crate) enum Maya1AudioProgress {
    /// A window ran: these samples are settled (possibly none).
    Window(Vec<f32>),
    /// No window was due.
    Idle,
}

/// The dtype the decoder loads in on `device`, and why: the device's Maya1
/// policy, which `IZWI_METAL_DTYPE` overrides on Metal.
pub fn maya1_lm_precision(device: &DeviceProfile) -> DTypeSelection {
    device.resolve_dtype(DTypeSelectionRequest::new(None).with_model_family(ModelFamily::Maya1Tts))
}

/// Whether a decoder loaded in `dtype` on `device` reloads in F32 when its
/// probe fails (everywhere but Metal, and never from F32 itself).
pub(crate) fn maya1_reloads_in_f32(device: &DeviceProfile, dtype: DType) -> bool {
    dtype != DType::F32 && BackendKind::from(device.kind) != BackendKind::Metal
}

impl Maya1TtsModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        if variant.family() != ModelFamily::Maya1Tts {
            return Err(Error::InvalidInput(format!(
                "Unsupported Maya1 variant: {variant}"
            )));
        }
        // The decoder at the device's Maya1 dtype (F16 on Metal, BF16 on
        // CUDA) when its load-time probe passes; SNAC is F32. Where an F32
        // reload is affordable it is the fallback. On Metal it would hold
        // 13.2 GB of weights, past what most Macs give the GPU, so a decoder
        // failing its F16 probe is refused there instead of reloaded.
        let selection = maya1_lm_precision(&device);
        let runtime = if variant.is_maya1_gguf() {
            // Q8_0 weights with F32 activations everywhere: nothing narrower
            // than F32 to probe.
            let runtime = Maya1Runtime::load_gguf(model_dir, &device.device)?;
            tracing::info!(
                "Maya1 decoder on {:?}: Q8_0 weights, F32 activations",
                device.kind
            );
            runtime
        } else if maya1_reloads_in_f32(&device, selection.dtype) {
            let (runtime, precision) = load_with_precision_fallback(
                "Maya1 decoder",
                &selection,
                true,
                DType::F32,
                |dtype| Maya1Runtime::load(model_dir, &device.device, dtype),
                Maya1Runtime::numeric_self_check,
            )?;
            tracing::info!(
                "Maya1 decoder dtype on {:?}: {:?} ({})",
                device.kind,
                precision.loaded,
                precision.reason
            );
            runtime
        } else {
            let runtime = Maya1Runtime::load(model_dir, &device.device, selection.dtype)?;
            let report = runtime.numeric_self_check();
            if !report.passed {
                return Err(Error::ModelLoadError(format!(
                    "Maya1's decoder failed its {:?} range probe on {:?} ({}); its F32 weights \
                     (13.2 GB) are not loaded on this device",
                    selection.dtype, device.kind, report.reason
                )));
            }
            tracing::info!(
                "Maya1 decoder dtype on {:?}: {:?} ({}; probe passed)",
                device.kind,
                selection.dtype,
                selection.reason
            );
            runtime
        };
        Ok(Self {
            model_identity: NEXT_MAYA1_MODEL_IDENTITY.fetch_add(1, Ordering::Relaxed),
            variant,
            runtime,
        })
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    pub fn model_identity(&self) -> u64 {
        self.model_identity
    }

    pub fn runtime(&self) -> &Maya1Runtime {
        &self.runtime
    }

    /// The built-in voices: named descriptions.
    pub fn available_speakers(&self) -> Vec<String> {
        MAYA1_VOICE_PRESETS
            .iter()
            .map(|preset| preset.name.to_string())
            .collect()
    }

    pub fn context_tokens(&self) -> usize {
        STATE_CONTEXT_TOKENS
    }

    pub fn diagnostics(&self) -> Maya1TtsDiagnostics {
        Maya1TtsDiagnostics {
            model_family: "maya1_tts",
            variant: self.variant.to_string(),
            sample_rate: SAMPLE_RATE,
            dtype: if self.variant.is_maya1_gguf() {
                "Q8_0 weights, F32 activations".to_string()
            } else {
                format!("{:?}", self.runtime.lm_dtype())
            },
            context_tokens: self.context_tokens(),
            max_new_tokens: config::MAX_NEW_TOKENS,
        }
    }

    /// Bytes of one retained K/V element.
    pub(crate) fn state_element_bytes(&self) -> usize {
        self.runtime.lm_dtype().size_in_bytes()
    }

    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<Maya1PhysicalStateSpec> {
        let retained = maya1_retained_contract(
            self.runtime.trunk().config(),
            self.runtime.lm_dtype(),
            default_kv_page_size(),
        )?;
        maya1_physical_state_spec(stage_graphs, retained, self.context_tokens())
    }

    /// A voice description and text to a prompt, checked against the
    /// context.
    pub(crate) fn prepare_retained_artifact(
        &self,
        description: &str,
        text: &str,
    ) -> Result<Arc<Maya1PreparedArtifact>> {
        let prompt = self.runtime.prompt(description, text)?;
        Ok(Arc::new(Maya1PreparedArtifact {
            prompt: prompt.ids,
            text_tokens: prompt.text_tokens,
        }))
    }

    pub(crate) fn new_retained_state(
        &self,
        artifact: Arc<Maya1PreparedArtifact>,
        params: Maya1GenerationParams,
    ) -> Result<Maya1RetainedState> {
        Maya1RetainedState::new(artifact, params, self.context_tokens())
    }

    /// Whether a request has an audio window to decode now.
    pub(crate) fn retained_window_due(&self, state: &Maya1RetainedState) -> bool {
        state.next_window().is_some()
    }

    pub(crate) fn retained_prefill_batch(
        &self,
        rows: &mut [Maya1StepRow<'_>],
        tokens: &[usize],
    ) -> Result<Vec<Maya1PrefillStep>> {
        maya1_prefill_batch(self.runtime.trunk(), rows, tokens)
    }

    pub(crate) fn retained_decode_batch(
        &self,
        rows: &mut [Maya1StepRow<'_>],
    ) -> Result<Vec<Maya1DecodeStep>> {
        maya1_decode_batch(self.runtime.trunk(), rows)
    }

    /// Decode the request's next audio window, waiting for the device so the
    /// window's cost lands in its own quantum.
    pub(crate) fn retained_audio_step(
        &self,
        state: &mut Maya1RetainedState,
    ) -> Result<Maya1AudioProgress> {
        let samples = state.decode_window(self.runtime.snac())?;
        self.runtime.device().synchronize()?;
        Ok(match samples {
            Some(samples) => Maya1AudioProgress::Window(samples),
            None => Maya1AudioProgress::Idle,
        })
    }

    /// Mark a request whose last window is out as finished.
    pub(crate) fn retained_finalize(&self, state: &mut Maya1RetainedState) -> Result<()> {
        if state.phase() != Maya1Phase::Audio {
            return Err(Error::InvalidInput(
                "Maya1 finalized before the decoder stopped".into(),
            ));
        }
        if state.audio.samples() == 0 {
            return Err(Error::InferenceError(
                "Maya1 generated no audio frames".into(),
            ));
        }
        state.mark_finished()
    }
}
