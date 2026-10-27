//! The engine-facing Maya2 model: load, state contract, voice preparation,
//! and the retained prefill / decode / audio-window steps the executor
//! drives.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use candle_core::{DType, Tensor};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::config::{SAMPLE_RATE, STATE_CONTEXT_TOKENS};
use super::guards::suspend_windows;
use super::physical::{maya2_physical_state_spec, maya2_retained_contract, Maya2PhysicalStateSpec};
use super::retained::{
    maya2_decode_batch, maya2_prefill_batch, Maya2DecodeStep, Maya2Networks, Maya2Phase,
    Maya2PrefillStep, Maya2PreparedArtifact, Maya2RetainedState, Maya2StepRow,
};
use super::runtime::{
    frame_budget, Maya2ClonedVoice, Maya2GenerationParams, Maya2Runtime, Maya2Voice,
};
use crate::backends::{BackendKind, DTypeSelection, DTypeSelectionRequest, DeviceProfile};
use crate::catalog::{ModelFamily, ModelVariant, MAYA2_VOICES};
use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::kernels::numeric_check::{
    load_with_precision_fallback, run_numeric_self_check, NumericSelfCheckReport, F16_SAFE_MAX_ABS,
};
use crate::models::shared::attention::paged::default_kv_page_size;

static NEXT_MAYA2_MODEL_IDENTITY: AtomicU64 = AtomicU64::new(1);

/// Prepared clone voices kept per model, least recently used first out (a
/// voice is a few hundred frames of codes and 192 floats).
const CLONE_CACHE_ENTRIES: usize = 16;

pub struct Maya2TtsModel {
    model_identity: u64,
    variant: ModelVariant,
    runtime: Maya2Runtime,
    clones: Mutex<VecDeque<([u8; 32], Arc<Maya2ClonedVoice>)>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Maya2TtsDiagnostics {
    pub model_family: &'static str,
    pub variant: String,
    pub sample_rate: u32,
    pub backbone_dtype: String,
    pub depth_dtype: String,
    pub context_tokens: usize,
}

/// What one audio-window quantum did.
pub(crate) enum Maya2AudioProgress {
    /// A window ran: these samples are settled.
    Window(Vec<f32>),
    /// No window was due.
    Idle,
}

/// The dtype the backbone loads in on `device`, and why: the device's Maya2
/// policy (F16 on Metal, BF16 on CUDA, F32 on the CPU), which
/// `IZWI_METAL_DTYPE` overrides on Metal.
pub fn maya2_backbone_precision(device: &DeviceProfile) -> DTypeSelection {
    device.resolve_dtype(DTypeSelectionRequest::new(None).with_model_family(ModelFamily::Maya2Tts))
}

/// The depth decoder's dtype: F16 on accelerators (upstream serves it in
/// F16, the closest to its F32 reference), F32 on the CPU.
pub fn maya2_depth_dtype(device: &DeviceProfile, backbone: DType) -> DType {
    if backbone == DType::F32 || BackendKind::from(device.kind) == BackendKind::Cpu {
        DType::F32
    } else {
        DType::F16
    }
}

/// Whether a backbone loaded in `dtype` on `device` reloads in F32 when its
/// probe fails (everywhere but Metal, and never from F32 itself).
pub(crate) fn maya2_reloads_in_f32(device: &DeviceProfile, dtype: DType) -> bool {
    dtype != DType::F32 && BackendKind::from(device.kind) != BackendKind::Metal
}

impl Maya2TtsModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        if variant.family() != ModelFamily::Maya2Tts {
            return Err(Error::InvalidInput(format!(
                "Unsupported Maya2 variant: {variant}"
            )));
        }
        // The backbone at the device's Maya2 dtype behind a load-time probe.
        // Off Metal an F32 reload is the fallback; on Metal F32 weights
        // (10 GB) are refused instead.
        let selection = maya2_backbone_precision(&device);
        let reloads = maya2_reloads_in_f32(&device, selection.dtype);
        let load = |dtype: DType| {
            Maya2Runtime::load(
                model_dir,
                &device.device,
                dtype,
                maya2_depth_dtype(&device, dtype),
            )
        };
        let runtime = if reloads {
            let (runtime, precision) = load_with_precision_fallback(
                "Maya2 backbone",
                &selection,
                true,
                DType::F32,
                load,
                Maya2Runtime::numeric_self_check,
            )?;
            tracing::info!(
                "Maya2 backbone dtype on {:?}: {:?} ({})",
                device.kind,
                precision.loaded,
                precision.reason
            );
            runtime
        } else {
            let runtime = load(selection.dtype)?;
            if selection.dtype != DType::F32 {
                let report = runtime.numeric_self_check();
                if !report.passed {
                    return Err(Error::ModelLoadError(format!(
                        "Maya2's backbone failed its {:?} range probe on {:?} ({}); its F32 \
                         weights (10 GB) are not loaded on this device",
                        selection.dtype, device.kind, report.reason
                    )));
                }
            }
            tracing::info!(
                "Maya2 backbone dtype on {:?}: {:?} ({})",
                device.kind,
                selection.dtype,
                selection.reason
            );
            runtime
        };
        Ok(Self {
            model_identity: NEXT_MAYA2_MODEL_IDENTITY.fetch_add(1, Ordering::Relaxed),
            variant,
            runtime,
            clones: Mutex::new(VecDeque::with_capacity(CLONE_CACHE_ENTRIES)),
        })
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    pub fn model_identity(&self) -> u64 {
        self.model_identity
    }

    pub fn runtime(&self) -> &Maya2Runtime {
        &self.runtime
    }

    /// The built-in voices.
    pub fn available_speakers(&self) -> Vec<String> {
        MAYA2_VOICES
            .iter()
            .map(|voice| voice.name.to_string())
            .collect()
    }

    pub fn context_tokens(&self) -> usize {
        STATE_CONTEXT_TOKENS
    }

    pub fn diagnostics(&self) -> Maya2TtsDiagnostics {
        Maya2TtsDiagnostics {
            model_family: "maya2_tts",
            variant: self.variant.to_string(),
            sample_rate: SAMPLE_RATE,
            backbone_dtype: format!("{:?}", self.runtime.backbone().dtype()),
            depth_dtype: format!("{:?}", self.runtime.depth().dtype()),
            context_tokens: self.context_tokens(),
        }
    }

    /// Bytes of one retained K/V element.
    pub(crate) fn state_element_bytes(&self) -> usize {
        self.runtime.backbone().dtype().size_in_bytes()
    }

    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<Maya2PhysicalStateSpec> {
        let retained = maya2_retained_contract(
            self.runtime.backbone().config(),
            self.runtime.backbone().dtype(),
            default_kv_page_size(),
        )?;
        maya2_physical_state_spec(stage_graphs, retained, self.context_tokens())
    }

    /// A clone voice from a reference clip (24 kHz mono), its transcript and
    /// language; prepared once per distinct clip and kept in a small cache.
    pub(crate) fn prepare_clone(
        &self,
        samples: &[f32],
        transcript: &str,
        language: &'static str,
    ) -> Result<Arc<Maya2ClonedVoice>> {
        let mut hasher = Sha256::new();
        for sample in samples {
            hasher.update(sample.to_le_bytes());
        }
        hasher.update([0xff]);
        hasher.update(transcript.trim().as_bytes());
        hasher.update([0xff]);
        hasher.update(language.as_bytes());
        let key: [u8; 32] = hasher.finalize().into();
        let poisoned = || Error::InferenceError("Maya2 clone cache lock poisoned".into());
        {
            let mut cache = self.clones.lock().map_err(|_| poisoned())?;
            if let Some(position) = cache.iter().position(|(cached, _)| *cached == key) {
                let entry = cache.remove(position).expect("found");
                let voice = entry.1.clone();
                cache.push_back(entry);
                return Ok(voice);
            }
        }
        let voice = Arc::new(self.runtime.prepare_clone(samples, transcript, language)?);
        let mut cache = self.clones.lock().map_err(|_| poisoned())?;
        if cache.len() == CLONE_CACHE_ENTRIES {
            cache.pop_front();
        }
        cache.push_back((key, voice.clone()));
        Ok(voice)
    }

    /// The prepared request: prompt, budget and guard windows for `voice`
    /// speaking `text` in `language`, checked against the context.
    pub(crate) fn prepare_retained_artifact(
        &self,
        voice: Maya2Voice,
        language: &'static str,
        text: &str,
        max_frames: Option<usize>,
    ) -> Result<Arc<Maya2PreparedArtifact>> {
        let prompt = self.runtime.prompt(&voice, language, text)?;
        let budget = frame_budget(text, language, max_frames, prompt.items.len())?;
        Ok(Arc::new(Maya2PreparedArtifact {
            suspend: suspend_windows(text, language),
            voice,
            language,
            text: text.to_string(),
            prompt,
            budget,
        }))
    }

    pub(crate) fn new_retained_state(
        &self,
        artifact: Arc<Maya2PreparedArtifact>,
        params: Maya2GenerationParams,
    ) -> Result<Maya2RetainedState> {
        Maya2RetainedState::new(artifact, params, self.context_tokens())
    }

    /// Whether a request has an audio window to decode now.
    pub(crate) fn retained_window_due(&self, state: &Maya2RetainedState) -> bool {
        state.next_window().is_some()
    }

    fn networks<'a>(
        &'a self,
        conditioning: &'a dyn Fn(&Maya2PreparedArtifact) -> Result<(Tensor, Tensor)>,
    ) -> Maya2Networks<'a> {
        Maya2Networks {
            backbone: self.runtime.backbone(),
            inputs: self.runtime.inputs(),
            depth: self.runtime.depth(),
            conditioning,
        }
    }

    pub(crate) fn retained_prefill_batch(
        &self,
        rows: &mut [Maya2StepRow<'_>],
        tokens: &[usize],
    ) -> Result<Vec<Maya2PrefillStep>> {
        let conditioning = |artifact: &Maya2PreparedArtifact| {
            self.runtime.conditioning(&artifact.voice, &artifact.prompt)
        };
        maya2_prefill_batch(&self.networks(&conditioning), rows, tokens)
    }

    pub(crate) fn retained_decode_batch(
        &self,
        rows: &mut [Maya2StepRow<'_>],
    ) -> Result<Vec<Maya2DecodeStep>> {
        let conditioning = |artifact: &Maya2PreparedArtifact| {
            self.runtime.conditioning(&artifact.voice, &artifact.prompt)
        };
        maya2_decode_batch(&self.networks(&conditioning), rows, self.runtime.device())
    }

    /// Decode the request's next audio window, waiting for the device so the
    /// window's cost lands in its own quantum.
    pub(crate) fn retained_audio_step(
        &self,
        state: &mut Maya2RetainedState,
    ) -> Result<Maya2AudioProgress> {
        let samples = state.decode_window(self.runtime.mimi())?;
        self.runtime.device().synchronize()?;
        Ok(match samples {
            Some(samples) => Maya2AudioProgress::Window(samples),
            None => Maya2AudioProgress::Idle,
        })
    }

    /// Mark a request whose last window is out as finished.
    pub(crate) fn retained_finalize(&self, state: &mut Maya2RetainedState) -> Result<()> {
        if state.phase() != Maya2Phase::Audio {
            return Err(Error::InvalidInput(
                "Maya2 finalized before the model stopped".into(),
            ));
        }
        if state.samples() == 0 {
            return Err(Error::InferenceError(
                "Maya2 generated no audio frames".into(),
            ));
        }
        state.mark_finished()
    }
}

impl Maya2Runtime {
    /// The load-time check of a backbone below F32: a built-in voice's
    /// prompt and a few frames on a probe cache. Every probed tensor must be
    /// finite and inside F16's safe range, or the backbone reloads in F32.
    pub fn numeric_self_check(&self) -> NumericSelfCheckReport {
        const FRAMES: usize = 3;
        run_numeric_self_check(
            format!("Maya2 backbone {:?}", self.backbone().dtype()),
            F16_SAFE_MAX_ABS,
            || -> Result<Vec<(String, Tensor)>> {
                let voice = Maya2Voice::Named(MAYA2_VOICES[0].name);
                let prompt = self.prompt(
                    &voice,
                    "hi",
                    "नमस्ते! The quick brown fox jumps over the lazy dog, ₹1,500 twice.",
                )?;
                let (embeds, cross) = self.conditioning(&voice, &prompt)?;
                let mut cache = self.scratch_cache(prompt.items.len() + FRAMES)?;
                let mut hidden =
                    self.backbone()
                        .forward(&mut [super::backbone::Maya2BackboneRow {
                            embeds,
                            cache: &mut cache,
                        }])?;
                let mut probed = vec![("prefill state".to_string(), hidden.clone())];
                let cross = cross.unsqueeze(0)?;
                let mut rng = crate::models::shared::noise::SplitMix64::new(1);
                for _ in 0..FRAMES {
                    let logits = self.inputs().cb0_logits(&hidden)?;
                    let cb0 = logits
                        .narrow(1, 0, super::config::CODEBOOK_SIZE)?
                        .argmax(1)?
                        .to_dtype(DType::U32)?;
                    let gumbel = super::sampling::depth_gumbel(
                        &mut [&mut rng],
                        super::config::DEPTH_LEVELS,
                        super::config::CODEBOOK_SIZE,
                        self.device(),
                    )?;
                    let frame = self
                        .depth()
                        .generate_frame(&hidden, &cb0, &cross, &gumbel, 0.7, 64)?;
                    hidden = self
                        .backbone()
                        .forward(&mut [super::backbone::Maya2BackboneRow {
                            embeds: self.inputs().audio_frames(&frame)?,
                            cache: &mut cache,
                        }])?;
                }
                probed.push(("decode state".to_string(), hidden));
                Ok(probed)
            },
        )
    }
}

/// The release checkpoint through the engine's own quanta: a request's codes
/// alone against the same request beside another, their caches in one
/// arena as the engine allocates them.
#[cfg(all(test, feature = "metal"))]
mod real_weight_tests {
    use super::*;
    use crate::backends::{BackendPreference, DeviceSelector};
    use crate::models::architectures::maya2::retained::{Maya2Phase, Maya2StepRow};
    use crate::models::architectures::maya2::runtime::{Maya2GenerationParams, Maya2Voice};
    use crate::models::architectures::maya2::sampling::Maya2SamplingParams;
    use crate::models::shared::attention::physical::test_physical_caches_with_dtype_on;

    /// Each request's codes, its prompt prefilled in spans of at most
    /// `chunk` positions (all at once when `None`).
    fn run(
        model: &Maya2TtsModel,
        requests: &[(&'static str, u64)],
        chunk: Option<usize>,
    ) -> Vec<Vec<u32>> {
        let text =
            "The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun.";
        let cfg = model.runtime().backbone().config().clone();
        let mut caches = test_physical_caches_with_dtype_on(
            model.runtime().device(),
            model.runtime().backbone().dtype(),
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            STATE_CONTEXT_TOKENS,
            requests.len(),
        );
        let mut states = requests
            .iter()
            .map(|&(voice, seed)| {
                let artifact = model
                    .prepare_retained_artifact(Maya2Voice::Named(voice), "en", text, None)
                    .unwrap();
                let params = Maya2GenerationParams {
                    sampling: Maya2SamplingParams::default(),
                    max_frames: None,
                    seed,
                };
                model.new_retained_state(artifact, params).unwrap()
            })
            .collect::<Vec<_>>();
        loop {
            let spans = states
                .iter()
                .map(|s| s.prompt_tokens() - s.prefill_cursor())
                .map(|left| chunk.map_or(left, |chunk| left.min(chunk)))
                .filter(|&span| span > 0)
                .collect::<Vec<_>>();
            if spans.is_empty() {
                break;
            }
            let mut rows = states
                .iter_mut()
                .zip(caches.iter_mut())
                .filter(|(state, _)| state.prefill_cursor() < state.prompt_tokens())
                .map(|(state, cache)| Maya2StepRow { state, cache })
                .collect::<Vec<_>>();
            model.retained_prefill_batch(&mut rows, &spans).unwrap();
        }
        loop {
            let mut rows = states
                .iter_mut()
                .zip(caches.iter_mut())
                .filter(|(state, _)| state.phase() == Maya2Phase::Decode && !state.stopped())
                .map(|(state, cache)| Maya2StepRow { state, cache })
                .collect::<Vec<_>>();
            if rows.is_empty() {
                break;
            }
            model.retained_decode_batch(&mut rows).unwrap();
        }
        states.iter().map(|s| s.codes().to_vec()).collect()
    }

    #[test]
    #[ignore = "needs the Maya2 release (IZWI_MAYA2_MODEL_DIR) and Metal"]
    fn a_requests_codes_do_not_depend_on_its_batch_on_metal() {
        let _lease = crate::backends::metal_test_device_or_skip(
            "a_requests_codes_do_not_depend_on_its_batch_on_metal",
        );
        let profile = DeviceSelector::detect_for_preference(BackendPreference::Metal).unwrap();
        if !matches!(profile.device, candle_core::Device::Metal(_)) {
            return;
        }
        let dir = std::path::PathBuf::from(std::env::var("IZWI_MAYA2_MODEL_DIR").unwrap());
        let model = Maya2TtsModel::load(&dir, ModelVariant::Maya2Native25B, profile).unwrap();
        let alone = run(&model, &[("Ahana", 20)], None).remove(0);
        let compare = |label: &str, other: &[u32]| {
            let frame = alone
                .chunks(32)
                .zip(other.chunks(32))
                .position(|(a, b)| a != b);
            eprintln!(
                "{label}: {} frames alone, {} here, first differing frame {frame:?}",
                alone.len() / 32,
                other.len() / 32
            );
            frame.is_none() && alone.len() == other.len()
        };
        // Second in the arena, so its KV blocks sit after another request's.
        let second = run(&model, &[("Anirban", 21), ("Ahana", 20)], None).remove(1);
        // Its prompt prefilled in chunks, as the scheduler may split it.
        let chunked = run(&model, &[("Ahana", 20)], Some(8)).remove(0);
        let second_ok = compare("second in the arena", &second);
        let chunked_ok = compare("prefilled in chunks of 8", &chunked);
        assert!(second_ok && chunked_ok);
    }
}
