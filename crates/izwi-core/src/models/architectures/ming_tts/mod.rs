//! inclusionAI's Ming-3.0-tts-0.5B, ported natively.
//!
//! Upstream: weights <https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B>
//! at ffd36ab5 (MIT); reference code
//! <https://github.com/inclusionAI/Ming-3.0-tts> at c88c023e (MIT); speaker
//! encoder <https://huggingface.co/funasr/campplus> at e4b6ede7
//! (Apache-2.0).
//!
//! Text, a voice and an optional control, framed as one prompt → a
//! Qwen2.5-0.5B language model whose last hidden state, once per 80 ms of
//! speech, conditions a flow-matching head that samples one 64-wide audio
//! latent → the latent's embedding goes back in as the next input → a
//! transformer VAE turns the latents into 44.1 kHz mono through an inverse
//! STFT. There are no audio tokens and no token sampling: the only
//! randomness is each latent's starting noise.
//!
//! The voice is a built-in one named in the caption, a described one, or a
//! reference clip (its VAE latents with its transcript, and its CAM++
//! speaker embedding).
//!
//! The family imports no other architecture (see `isolation_tests`).

pub mod config;
pub mod flow;
mod graphs;
mod model;
mod physical;
pub mod prompt;
pub(crate) mod retained;
pub mod speaker;
pub mod stack;
pub mod stream;
pub mod vae;
mod weights;

#[cfg(test)]
mod reference_tests;
#[cfg(test)]
pub(crate) mod synthetic;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use serde::Serialize;

pub use config::{
    MingTtsConfig, CONTEXT_POSITIONS, DEFAULT_GUIDANCE, LATENT_DIM, LATENT_RATE_HZ, LATENT_SAMPLES,
    MAX_NEW_LATENTS, MAX_REFERENCE_LATENTS, MIN_LATENTS, MIN_REFERENCE_SECONDS, SAMPLE_RATE,
    SPEAKER_WEIGHTS_FILE, WEIGHTS_FILE,
};
pub(crate) use model::{ming_audio_batching_enabled, ming_reloads_in_f32, MingAudioProgress};
pub use model::{ming_lm_precision, MingTtsDiagnostics, MingTtsModel};
pub(crate) use physical::MING_TTS_DOMAIN;
pub use prompt::{
    target_language_name, MingControl, MingDialect, MingEmotion, MingLevel, MingSpeaker,
    MING_INLINE_TAGS,
};
pub use weights::{published_tensors, unused_tensors};

use crate::error::{Error, Result};
use crate::kernels::numeric_check::{
    run_numeric_self_check, NumericSelfCheckReport, F16_SAFE_MAX_ABS,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::noise::SplitMix64;
use config::{STOP_THRESHOLD, TOKENIZER_FILE};
use flow::{MingAggregator, MingFlowHead};
pub(crate) use graphs::ming_tts_graph_device_bytes;
use graphs::{MingFlowGraphs, MingGraphKey};
use prompt::MingTokenizer;
use speaker::{sinc_resample, CampPlus, SPEAKER_EMBEDDING_DIM};
use stack::{MingForwardRow, MingTrunk};
use stream::MingAudioStream;
use vae::{MingVaeDecoder, MingVaeEncoder};

/// What each part of the model runs in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MingPrecision {
    /// The language model.
    pub lm: DType,
    /// The aggregator and the flow head.
    pub flow: DType,
    /// The VAE.
    pub vae: DType,
}

impl MingPrecision {
    /// Everything in F32.
    pub const F32: Self = Self {
        lm: DType::F32,
        flow: DType::F32,
        vae: DType::F32,
    };

    /// The flow path's dtype beside a language model in `dtype`: the same.
    pub fn flow_dtype_for(dtype: DType) -> DType {
        dtype
    }

    /// The order loads try for a device whose policy picks `dtype`: the
    /// language model and flow path in it; then the flow path in F32; then
    /// everything in F32. The VAE is F32 throughout.
    pub fn ladder(dtype: DType) -> Vec<Self> {
        let first = Self {
            lm: dtype,
            flow: Self::flow_dtype_for(dtype),
            vae: DType::F32,
        };
        let mut steps = vec![first];
        if first.flow != DType::F32 {
            steps.push(Self {
                flow: DType::F32,
                ..first
            });
        }
        if first != Self::F32 {
            steps.push(Self::F32);
        }
        steps
    }
}

/// The noise streams of one request's seed.
#[derive(Debug, Clone, Copy)]
pub(crate) enum MingNoiseStream {
    /// Each latent's starting noise.
    Flow,
}

impl From<MingNoiseStream> for u64 {
    fn from(stream: MingNoiseStream) -> u64 {
        match stream {
            MingNoiseStream::Flow => 0x4d49_4e47_464c_4f57,
        }
    }
}

/// A reference clip, encoded once: what a cloned voice is made of.
#[derive(Debug, Clone, PartialEq)]
pub struct MingReferenceVoice {
    /// The clip's VAE latents, 64 values each (empty without a transcript:
    /// the model continues a reference only together with its words).
    pub latents: Vec<f32>,
    /// The clip's transcript, when it came with one.
    pub transcript: Option<String>,
    /// CAM++'s 192-wide speaker embedding.
    pub embedding: Vec<f32>,
}

impl MingReferenceVoice {
    pub fn latent_count(&self) -> usize {
        self.latents.len() / LATENT_DIM
    }
}

/// Whose voice speaks.
#[derive(Debug, Clone, PartialEq)]
pub enum MingVoice {
    /// A voice the caption names or describes (built-in voices, voice
    /// design, free-form instructions): a speaker block of zeros.
    Captioned,
    /// A cloned voice.
    Reference(Arc<MingReferenceVoice>),
    /// No speaker block at all.
    Unconditioned,
}

/// One utterance to speak.
#[derive(Debug, Clone, PartialEq)]
pub struct MingRequest {
    pub text: String,
    /// A language code or name; required for Japanese, Korean, German,
    /// French and Spanish.
    pub language: Option<String>,
    pub voice: MingVoice,
    pub control: MingControl,
}

/// A request's prompt, ready for the language model.
#[derive(Debug, Clone, PartialEq)]
pub struct MingPrompt {
    /// The input embeddings, row-major [positions, hidden], F32.
    pub embeds: Vec<f32>,
    pub positions: usize,
    /// The latent the flow head sees as history at the first step: the
    /// reference's last latent, or zeros.
    pub history: Vec<f32>,
    /// Tokens the text alone takes.
    pub text_tokens: usize,
}

impl MingPrompt {
    /// Positions the prompt takes in the language model's context.
    pub fn len(&self) -> usize {
        self.positions
    }

    pub fn is_empty(&self) -> bool {
        self.positions == 0
    }
}

/// Everything one utterance's generation is parameterised by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MingGenerationParams {
    /// Classifier-free guidance scale of the flow head.
    pub guidance: f32,
    /// The most latents to generate (80 ms each): the engine's tokens.
    pub max_new_tokens: usize,
    /// Positive; drives each latent's starting noise.
    pub seed: u64,
}

impl MingGenerationParams {
    pub fn validate(&self) -> Result<()> {
        if !self.guidance.is_finite() || !(0.0..=10.0).contains(&self.guidance) {
            return Err(Error::InvalidInput(
                "Ming guidance scale must be between 0 and 10".into(),
            ));
        }
        if self.max_new_tokens < MIN_LATENTS || self.max_new_tokens > MAX_NEW_LATENTS {
            return Err(Error::InvalidInput(format!(
                "Ming generates {MIN_LATENTS}..={MAX_NEW_LATENTS} latents per request"
            )));
        }
        if self.seed == 0 {
            return Err(Error::InvalidInput(
                "Ming generation needs a resolved positive seed".into(),
            ));
        }
        Ok(())
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MingStopReason {
    /// The stop head fired.
    EndOfSpeech,
    /// The latent budget or the context ran out first: the audio is cut.
    LatentBudget,
}

#[derive(Debug, Clone, Serialize)]
pub struct MingGenerationOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Latents generated, the stop step's included.
    pub latents: usize,
    pub stop_reason: MingStopReason,
    pub prompt_positions: usize,
    pub lm_ms: f32,
    pub codec_ms: f32,
}

/// The loaded networks of one Ming model.
pub struct MingRuntime {
    model_dir: PathBuf,
    cfg: MingTtsConfig,
    tokenizer: MingTokenizer,
    trunk: MingTrunk,
    aggregator: MingAggregator,
    flow: MingFlowHead,
    /// `stop_head`: [2, hidden] row-major and its bias, on the host.
    stop_weight: Vec<f32>,
    stop_bias: [f32; 2],
    /// `spk_head`: [hidden, 192] row-major and its bias, on the host.
    speaker_weight: Vec<f32>,
    speaker_bias: Vec<f32>,
    decoder: MingVaeDecoder,
    encoder: MingVaeEncoder,
    speaker: Option<CampPlus>,
    /// The flow path's CUDA graphs (off until [`MingRuntime::resolve_flow_graphs`]).
    graphs: MingFlowGraphs,
    device: Device,
}

/// One request's inputs to one latent: the language model's hidden state,
/// the previous latent, the starting noise and the guidance.
pub(crate) struct MingLatentInput<'a> {
    pub(crate) hidden: &'a [f32],
    pub(crate) history: &'a [f32],
    pub(crate) noise: &'a [f32],
    pub(crate) guidance: f32,
}

/// Requests the flow head and the aggregator take on CUDA: one zero-padded
/// shape for every call, the shape their CUDA graphs capture.
pub(crate) const FLOW_GROUP_CUDA: usize = 32;
/// Most requests one call takes on Metal, unpadded.
pub(crate) const FLOW_GROUP_METAL: usize = 32;

/// How requests share the flow head's and the aggregator's calls.
///
/// A request's values must be the ones it gets alone. Every op on the flow
/// path keeps rows apart except the matmuls, whose kernel Candle picks by
/// shape. On CUDA each call is padded to [`FLOW_GROUP_CUDA`] requests: one
/// shape, the one the CUDA graphs capture. On Metal a call takes the
/// requests it has: Metal's matmul rounds a row the same for every row
/// count from two up, and every matmul of the flow head has at least two
/// rows; an aggregator call is padded to two rows, since its first matmul
/// has one row a request (a lone request padded to eight would cost 60%
/// more, M1 Pro). On the CPU the matmul's rounding changes with the row count,
/// and padding would cost a lone request about 1.5x, so each request runs
/// alone. The tests check both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlowGroup {
    /// Most requests one call takes.
    pub(crate) requests: usize,
    /// Whether a call is padded to `requests` with zero rows.
    pub(crate) padded: bool,
    /// Fewest rows an aggregator call has, zero-padded: its first matmul
    /// takes one row a request, and Candle's Metal matmul runs one row as a
    /// GEMV that rounds differently from the GEMM of two or more. (The flow
    /// path's own Metal kernel rounds a row the same at any count; the pad
    /// keeps that true with it switched off.)
    pub(crate) min_aggregator_rows: usize,
}

pub(crate) fn flow_group(device: &Device) -> FlowGroup {
    if device.is_cuda() {
        FlowGroup {
            requests: FLOW_GROUP_CUDA,
            padded: true,
            min_aggregator_rows: FLOW_GROUP_CUDA,
        }
    } else if device.is_metal() {
        FlowGroup {
            requests: FLOW_GROUP_METAL,
            padded: false,
            min_aggregator_rows: 2,
        }
    } else {
        FlowGroup {
            requests: 1,
            padded: false,
            min_aggregator_rows: 1,
        }
    }
}

fn host_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1()?)
}

impl MingRuntime {
    /// Load the language model in `precision.lm`, the aggregator and flow
    /// head in `precision.flow` and the VAE in `precision.vae` from
    /// `model_dir`.
    pub fn load(model_dir: &Path, device: &Device, precision: MingPrecision) -> Result<Self> {
        let cfg = MingTtsConfig::load(model_dir)?;
        let tokenizer = MingTokenizer::load(&model_dir.join(TOKENIZER_FILE))?;
        let lm = weights::mmap_checkpoint(model_dir, precision.lm, device)?;
        let flow = if precision.flow == precision.lm {
            lm.clone()
        } else {
            weights::mmap_checkpoint(model_dir, precision.flow, device)?
        };
        let vae = weights::mmap_checkpoint(model_dir, precision.vae, device)?;
        let host = weights::mmap_checkpoint(model_dir, DType::F32, &Device::Cpu)?;
        let speaker_path = model_dir.join(SPEAKER_WEIGHTS_FILE);
        let speaker = if speaker_path.is_file() {
            Some(CampPlus::open(&speaker_path)?)
        } else {
            tracing::warn!(
                "Ming speaker encoder weights {} are missing; voice cloning is unavailable",
                speaker_path.display()
            );
            None
        };
        Self::assemble(
            model_dir, cfg, tokenizer, &lm, &flow, &vae, &host, speaker, device,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn assemble(
        model_dir: &Path,
        cfg: MingTtsConfig,
        tokenizer: MingTokenizer,
        lm: &candle_nn::VarBuilder,
        flow: &candle_nn::VarBuilder,
        vae: &candle_nn::VarBuilder,
        host: &candle_nn::VarBuilder,
        speaker: Option<CampPlus>,
        device: &Device,
    ) -> Result<Self> {
        let hidden = cfg.llm.hidden_size;
        let window = config::vae_window_keys();
        let stop = host.pp("stop_head");
        let stop_bias = host_f32(&stop.get(2, "bias")?)?;
        let spk = host.pp("spk_head");
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            trunk: MingTrunk::load(
                cfg.llm.clone(),
                cfg.vocab_size,
                lm.pp("model").pp("model"),
                CONTEXT_POSITIONS,
            )?,
            aggregator: MingAggregator::load(
                cfg.aggregator.clone(),
                flow.pp("linear_proj_audio"),
                hidden,
            )?,
            flow: MingFlowHead::load(
                cfg.flow.clone(),
                flow.pp("flowloss").pp("cfm").pp("model"),
                hidden,
            )?,
            stop_weight: host_f32(&stop.get((2, hidden), "weight")?)?,
            stop_bias: [stop_bias[0], stop_bias[1]],
            speaker_weight: host_f32(&spk.get((hidden, SPEAKER_EMBEDDING_DIM), "weight")?)?,
            speaker_bias: host_f32(&spk.get(hidden, "bias")?)?,
            decoder: MingVaeDecoder::load(&cfg, vae.pp("audio").pp("decoder"), window)?,
            encoder: MingVaeEncoder::load(&cfg, vae.pp("audio").pp("encoder"), window)?,
            speaker,
            tokenizer,
            cfg,
            graphs: MingFlowGraphs::off("not resolved"),
            device: device.clone(),
        })
    }

    /// On CUDA, the flow path's and the VAE window's graphs ([`graphs`]):
    /// resolved once the load's range probe has passed, and checked through
    /// capture and verification on xorshift inputs before the model serves.
    /// A failure leaves that part eager, with the reason in the diagnostics.
    pub(crate) fn resolve_graphs(&mut self) {
        self.resolve_flow_graphs();
        self.decoder.resolve_graphs();
    }

    fn resolve_flow_graphs(&mut self) {
        let mut graphs = MingFlowGraphs::resolve(&self.device);
        if graphs.is_on() {
            match self.self_check_flow_graphs(&graphs) {
                Ok(()) if graphs.is_on() => {}
                Ok(()) => {
                    let reason = graphs.diagnostics()["off"].to_string();
                    graphs.turn_off(format!("self-check failed: {reason}"));
                }
                Err(error) => graphs.turn_off(format!("self-check failed: {error}")),
            }
            if graphs.is_on() {
                tracing::info!(
                    requests = FLOW_GROUP_CUDA,
                    "Ming TTS flow head and aggregator run as CUDA graphs"
                );
            } else {
                tracing::warn!(
                    reason = %graphs.diagnostics()["off"],
                    "Ming TTS flow head and aggregator run eagerly"
                );
            }
        }
        self.graphs = graphs;
    }

    /// Two sampler and two aggregator calls at the padded group through
    /// `graphs`: the first captures each graph and checks a replay against
    /// the eager run, the second checks a replay over new inputs.
    fn self_check_flow_graphs(&self, graphs: &MingFlowGraphs) -> Result<()> {
        use crate::kernels::numeric_check::xorshift_tensor;
        let rows = FLOW_GROUP_CUDA;
        let hidden = self.cfg.llm.hidden_size;
        for seed in [0x31A6_F10Au64, 0x31A6_F10B] {
            let inputs = [
                xorshift_tensor(&[rows, hidden], seed, 4.0, &self.device)?,
                xorshift_tensor(&[rows, LATENT_DIM], seed ^ 0x1, 1.0, &self.device)?,
                xorshift_tensor(&[rows, LATENT_DIM], seed ^ 0x2, 1.0, &self.device)?,
                Tensor::full(DEFAULT_GUIDANCE, (rows, 1), &self.device)?,
            ];
            self.run_sampler(graphs, &inputs)?;
            let latents = xorshift_tensor(&[rows, LATENT_DIM], seed ^ 0x3, 1.0, &self.device)?;
            self.run_aggregator(graphs, &latents)?;
        }
        Ok(())
    }

    /// The flow head over one group's `inputs` (hidden, history, noise,
    /// guidance), through its graph at the padded CUDA group.
    fn run_sampler(&self, graphs: &MingFlowGraphs, inputs: &[Tensor; 4]) -> Result<Tensor> {
        let body = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
            let [hidden, history, noise, guidance] = inputs else {
                return Err(Error::InferenceError(format!(
                    "Ming flow graph received {} inputs, expected 4",
                    inputs.len()
                )));
            };
            Ok(vec![self
                .flow
                .sample_batch(hidden, history, noise, guidance)?])
        };
        if graphs.takes(inputs[0].dim(0)?) {
            graphs.run(MingGraphKey::Sampler(self.flow.dtype()), inputs, &body)
        } else {
            Ok(body(inputs)?.pop().expect("one output"))
        }
    }

    /// The aggregator over one group's `latents`, through its graph at the
    /// padded CUDA group.
    fn run_aggregator(&self, graphs: &MingFlowGraphs, latents: &Tensor) -> Result<Tensor> {
        let body = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
            Ok(vec![self.aggregator.forward(&inputs[0])?])
        };
        if graphs.takes(latents.dim(0)?) {
            graphs.run(
                MingGraphKey::Aggregator(self.flow.dtype()),
                std::slice::from_ref(latents),
                &body,
            )
        } else {
            self.aggregator.forward(latents)
        }
    }

    /// The flow path's CUDA graphs: their state and counters.
    pub fn flow_graphs_diagnostics(&self) -> serde_json::Value {
        self.graphs.diagnostics()
    }

    /// The language model's decode-step CUDA graphs: their state and
    /// counters.
    pub fn decode_graphs_diagnostics(&self) -> serde_json::Value {
        self.trunk.decode_graphs_diagnostics()
    }

    /// Run the language model's decode steps eagerly (a test's reference
    /// for the graphs).
    #[cfg(test)]
    pub(crate) fn turn_off_decode_graphs(&mut self) {
        self.trunk.turn_off_graphs();
    }

    /// The VAE window's CUDA graph: its state and counters.
    pub fn vae_graphs_diagnostics(&self) -> serde_json::Value {
        self.decoder.graphs_diagnostics()
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn config(&self) -> &MingTtsConfig {
        &self.cfg
    }

    pub fn lm_dtype(&self) -> DType {
        self.trunk.dtype()
    }

    pub fn vae_dtype(&self) -> DType {
        self.decoder.dtype()
    }

    pub fn flow_dtype(&self) -> DType {
        self.flow.dtype()
    }

    pub fn precision(&self) -> MingPrecision {
        MingPrecision {
            lm: self.lm_dtype(),
            flow: self.flow_dtype(),
            vae: self.vae_dtype(),
        }
    }

    pub fn trunk(&self) -> &MingTrunk {
        &self.trunk
    }

    pub fn decoder(&self) -> &MingVaeDecoder {
        &self.decoder
    }

    pub fn can_clone_voices(&self) -> bool {
        self.speaker.is_some()
    }

    /// Encode a reference clip (mono samples in [-1, 1] at `sample_rate`).
    ///
    /// The speaker embedding reads the whole clip. With a transcript the
    /// clip is also encoded to latents, padded with silence at its start to
    /// a whole number of latents: its end is what the model continues from.
    pub fn encode_reference(
        &self,
        samples: &[f32],
        sample_rate: u32,
        transcript: Option<&str>,
    ) -> Result<MingReferenceVoice> {
        let speaker = self.speaker.as_ref().ok_or_else(|| {
            Error::InvalidInput(format!(
                "Ming voice cloning needs {SPEAKER_WEIGHTS_FILE} in the model folder; \
                 download the model again to fetch it"
            ))
        })?;
        if sample_rate == 0 || samples.is_empty() {
            return Err(Error::InvalidInput("Ming reference audio is empty".into()));
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(Error::InvalidInput(
                "Ming reference audio is not finite".into(),
            ));
        }
        let seconds = samples.len() as f32 / sample_rate as f32;
        if seconds < MIN_REFERENCE_SECONDS {
            return Err(Error::InvalidInput(format!(
                "Ming needs at least {MIN_REFERENCE_SECONDS} s of reference audio"
            )));
        }
        let transcript = transcript
            .map(str::trim)
            .filter(|transcript| !transcript.is_empty());
        let embedding = speaker.embed_clip(samples, sample_rate)?;
        let latents = match transcript {
            None => Vec::new(),
            Some(_) => {
                let wav = sinc_resample(samples, sample_rate, SAMPLE_RATE);
                let count = wav.len().div_ceil(LATENT_SAMPLES);
                if count > MAX_REFERENCE_LATENTS {
                    return Err(Error::InvalidInput(format!(
                        "a Ming reference clip with a transcript is at most {:.0} s; trim the \
                         clip and its transcript, or send the clip without a transcript",
                        MAX_REFERENCE_LATENTS as f32 / LATENT_RATE_HZ
                    )));
                }
                let mut padded = vec![0f32; count * LATENT_SAMPLES - wav.len()];
                padded.extend_from_slice(&wav);
                // The posterior's noise follows the clip, so a voice encodes
                // the same way every time.
                let seed = padded
                    .iter()
                    .fold(0xcbf2_9ce4_8422_2325u64, |hash, sample| {
                        (hash ^ u64::from(sample.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
                    });
                let noise = SplitMix64::new(seed).standard_normals(count * LATENT_DIM);
                self.encoder.encode(&padded, &noise[..count * LATENT_DIM])?
            }
        };
        Ok(MingReferenceVoice {
            latents,
            transcript: transcript.map(str::to_string),
            embedding,
        })
    }

    /// `spk_head` of a 192-wide embedding.
    fn project_speaker(&self, embedding: &[f32]) -> Result<Vec<f32>> {
        if embedding.len() != SPEAKER_EMBEDDING_DIM {
            return Err(Error::InvalidInput(format!(
                "Ming speaker embeddings are {SPEAKER_EMBEDDING_DIM} wide"
            )));
        }
        Ok(self
            .speaker_weight
            .as_chunks::<SPEAKER_EMBEDDING_DIM>()
            .0
            .iter()
            .zip(&self.speaker_bias)
            .map(|(row, bias)| row.iter().zip(embedding).map(|(w, e)| w * e).sum::<f32>() + bias)
            .collect())
    }

    /// The prompt of `request`, its speaker and reference rows filled in.
    pub fn prompt(&self, request: &MingRequest) -> Result<MingPrompt> {
        let hidden = self.cfg.llm.hidden_size;
        let language = match request.language.as_deref().map(str::trim) {
            None | Some("") => None,
            Some(language) => Some(target_language_name(language).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Ming does not speak '{language}'; it takes Chinese, English, Japanese, \
                     Korean, German, French and Spanish"
                ))
            })?),
        };
        let text = prompt::prepare_text(&request.text, language, &request.control)?;
        let instruction = request.control.instruction()?;
        let (speaker_row, reference) = match &request.voice {
            MingVoice::Unconditioned => (None, None),
            MingVoice::Captioned => (Some(vec![0f32; hidden]), None),
            MingVoice::Reference(voice) => (
                Some(self.project_speaker(&voice.embedding)?),
                voice
                    .transcript
                    .as_deref()
                    .filter(|_| voice.latent_count() > 0)
                    .map(|transcript| (transcript, voice.as_ref())),
            ),
        };
        let ids = self.tokenizer.prompt_ids(
            &text,
            language,
            speaker_row.is_some(),
            instruction.as_deref(),
            reference.map(|(transcript, voice)| (transcript, voice.latent_count())),
        )?;
        let positions = ids.ids.len();
        if positions + MIN_LATENTS > CONTEXT_POSITIONS {
            return Err(Error::InvalidInput(format!(
                "this Ming request takes {positions} of {CONTEXT_POSITIONS} positions before any \
                 speech; shorten the text, the instruction or the reference clip"
            )));
        }
        let mut embeds = host_f32(&self.trunk.embed_tokens(&ids.ids)?)?;
        if let (Some(row), Some(values)) = (ids.speaker_row, &speaker_row) {
            embeds[row * hidden..(row + 1) * hidden].copy_from_slice(values);
        }
        let mut history = vec![0f32; LATENT_DIM];
        if let (Some(row), Some((_, voice))) = (ids.latent_row, reference) {
            let count = voice.latent_count();
            let latents =
                Tensor::from_vec(voice.latents.clone(), (count, LATENT_DIM), &self.device)?;
            let rows = host_f32(&self.aggregator.forward(&latents)?)?;
            embeds[row * hidden..(row + count) * hidden].copy_from_slice(&rows);
            history.copy_from_slice(&voice.latents[(count - 1) * LATENT_DIM..]);
        }
        if embeds.iter().any(|value| !value.is_finite()) {
            return Err(Error::InferenceError(
                "Ming prompt embeddings are not finite".into(),
            ));
        }
        Ok(MingPrompt {
            embeds,
            positions,
            history,
            text_tokens: ids.text_tokens,
        })
    }

    /// The probability the stop head gives to "this was the last latent".
    pub(crate) fn stop_probability(&self, hidden: &[f32]) -> f32 {
        let logit = |row: usize| {
            self.stop_weight[row * hidden.len()..(row + 1) * hidden.len()]
                .iter()
                .zip(hidden)
                .map(|(w, h)| w * h)
                .sum::<f32>()
                + self.stop_bias[row]
        };
        let (no, yes) = (logit(0), logit(1));
        1.0 / (1.0 + (no - yes).exp())
    }

    /// Whether the model stops after the latent of 0-based `step`.
    pub(crate) fn stops_at(&self, hidden: &[f32], step: usize) -> bool {
        step + 1 >= MIN_LATENTS && self.stop_probability(hidden) > STOP_THRESHOLD
    }

    /// One latent from the language model's state.
    pub(crate) fn sample_latent(
        &self,
        hidden: &[f32],
        history: &[f32],
        noise: &[f32],
        guidance: f32,
    ) -> Result<Vec<f32>> {
        let mut latents = self.sample_latents(&[MingLatentInput {
            hidden,
            history,
            noise,
            guidance,
        }])?;
        Ok(latents.pop().expect("one request, one latent"))
    }

    /// One latent for each request, every request's in the same flow-head
    /// calls ([`flow_group`] at a time). A request's latent depends on its
    /// own inputs only: it is the one it gets alone.
    pub(crate) fn sample_latents(&self, rows: &[MingLatentInput<'_>]) -> Result<Vec<Vec<f32>>> {
        let latents = self
            .sample_latents_device(rows)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;
        if latents.iter().flatten().any(|value| !value.is_finite()) {
            return Err(Error::InferenceError(
                "Ming flow head produced a non-finite latent".into(),
            ));
        }
        Ok(latents)
    }

    /// [`Self::sample_latents`] left on the device, unchecked: [rows, 64]
    /// F32, the caller reads it back (and checks it) when it needs it. Each
    /// call's inputs go up as one upload.
    pub(crate) fn sample_latents_device(&self, rows: &[MingLatentInput<'_>]) -> Result<Tensor> {
        let hidden_size = self.cfg.llm.hidden_size;
        let group = flow_group(&self.device);
        let columns = hidden_size + 2 * LATENT_DIM + 1;
        let mut out = Vec::with_capacity(rows.len().div_ceil(group.requests));
        for chunk in rows.chunks(group.requests) {
            let width = if group.padded {
                group.requests
            } else {
                chunk.len()
            };
            // Per request: hidden | history | noise | guidance.
            let mut packed = vec![0f32; width * columns];
            for (row, values) in chunk.iter().zip(packed.chunks_exact_mut(columns)) {
                if row.hidden.len() != hidden_size
                    || row.history.len() != LATENT_DIM
                    || row.noise.len() != LATENT_DIM
                {
                    return Err(Error::InvalidInput(
                        "Ming flow sampling needs a hidden state, a 64-wide history and \
                         64 noise values per request"
                            .into(),
                    ));
                }
                let (hidden, rest) = values.split_at_mut(hidden_size);
                let (history, rest) = rest.split_at_mut(LATENT_DIM);
                let (noise, guidance) = rest.split_at_mut(LATENT_DIM);
                hidden.copy_from_slice(row.hidden);
                history.copy_from_slice(row.history);
                noise.copy_from_slice(row.noise);
                guidance[0] = row.guidance;
            }
            let packed = Tensor::from_vec(packed, (width, columns), &self.device)?;
            let part = |from: usize, cols: usize| -> Result<Tensor> {
                Ok(packed.narrow(1, from, cols)?.contiguous()?)
            };
            let inputs = [
                part(0, hidden_size)?,
                part(hidden_size, LATENT_DIM)?,
                part(hidden_size + LATENT_DIM, LATENT_DIM)?,
                part(hidden_size + 2 * LATENT_DIM, 1)?,
            ];
            out.push(
                self.run_sampler(&self.graphs, &inputs)?
                    .narrow(0, 0, chunk.len())?,
            );
        }
        Ok(if out.len() == 1 {
            out.pop().expect("one group")
        } else {
            Tensor::cat(&out, 0)?
        })
    }

    /// The embeddings the language model reads for `latents` [rows * 64],
    /// [rows, hidden] in the model dtype on the device. The aggregator runs
    /// [`flow_group`] rows at a time, as the flow head does.
    pub(crate) fn latent_embeddings(&self, latents: &[f32]) -> Result<Tensor> {
        let rows = latents.len() / LATENT_DIM;
        if rows == 0 || latents.len() != rows * LATENT_DIM {
            return Err(Error::InvalidInput(
                "Ming aggregator input must be whole 64-wide latents".into(),
            ));
        }
        self.latent_embeddings_device(&Tensor::from_vec(
            latents.to_vec(),
            (rows, LATENT_DIM),
            &self.device,
        )?)
    }

    /// [`Self::latent_embeddings`] of `latents` [rows, 64] F32 already on
    /// the device (a flow-head call's output): each group zero-padded on the
    /// device as [`Self::latent_embeddings`] pads it on the host, so the
    /// values are the same.
    pub(crate) fn latent_embeddings_device(&self, latents: &Tensor) -> Result<Tensor> {
        let (rows, width) = latents.dims2()?;
        if rows == 0 || width != LATENT_DIM {
            return Err(Error::InvalidInput(
                "Ming aggregator input must be whole 64-wide latents".into(),
            ));
        }
        let group = flow_group(&self.device);
        let mut parts = Vec::with_capacity(rows.div_ceil(group.requests));
        let mut start = 0;
        while start < rows {
            let count = group.requests.min(rows - start);
            let width =
                if group.padded { group.requests } else { count }.max(group.min_aggregator_rows);
            let padded = latents
                .narrow(0, start, count)?
                .pad_with_zeros(0, 0, width - count)?
                .contiguous()?;
            parts.push(
                self.run_aggregator(&self.graphs, &padded)?
                    .narrow(0, 0, count)?,
            );
            start += count;
        }
        Ok(if parts.len() == 1 {
            parts.pop().expect("one group")
        } else {
            Tensor::cat(&parts, 0)?
        })
    }

    /// `rows` of prompt embeddings (row-major, F32) as a device tensor.
    pub(crate) fn prompt_rows(&self, embeds: &[f32]) -> Result<Tensor> {
        let hidden = self.cfg.llm.hidden_size;
        Ok(Tensor::from_vec(
            embeds.to_vec(),
            (embeds.len() / hidden, hidden),
            &self.device,
        )?
        .to_dtype(self.trunk.dtype())?)
    }

    /// A scratch cache on this model's device for `positions` positions.
    pub(crate) fn scratch_cache(&self, positions: usize) -> Result<PhysicalPagedKvCache> {
        let cfg = self.trunk.config();
        PhysicalPagedKvCache::scratch_for_probe(
            &self.device,
            self.trunk.dtype(),
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            positions,
        )
    }

    /// The most latents a prompt of `positions` leaves room for.
    pub fn latent_room(positions: usize) -> usize {
        // As the reference: one language-model step per latent, the
        // prompt's own step included.
        CONTEXT_POSITIONS.saturating_sub(positions)
    }

    /// The load-time check of a model below F32: a built-in voice's prompt
    /// and a few steps on a probe cache. Every probed tensor must be finite
    /// and inside F16's safe range.
    pub fn numeric_self_check(&self) -> NumericSelfCheckReport {
        const STEPS: usize = 3;
        run_numeric_self_check(
            format!(
                "Ming language model {:?}, flow path {:?}",
                self.trunk.dtype(),
                self.flow.dtype()
            ),
            F16_SAFE_MAX_ABS,
            || -> Result<Vec<(String, Tensor)>> {
                let prompt = self.prompt(&MingRequest {
                    text: "The quick brown fox jumps over the lazy dog, twice!".into(),
                    language: None,
                    voice: MingVoice::Captioned,
                    control: MingControl::BuiltInVoice("US_Emma".into()),
                })?;
                let mut cache = self.scratch_cache(prompt.positions + STEPS)?;
                let embeds = self.prompt_rows(&prompt.embeds)?;
                let mut hidden = self.forward_one(&embeds, &mut cache)?;
                let mut probed = vec![("prefill hidden".to_string(), hidden.clone())];
                let mut history = prompt.history.clone();
                let mut rng = SplitMix64::new(0x5eed);
                for _ in 0..STEPS {
                    let state = host_f32(&hidden)?;
                    let noise = rng.standard_normals(LATENT_DIM);
                    let latent = self.sample_latent(
                        &state,
                        &history,
                        &noise[..LATENT_DIM],
                        DEFAULT_GUIDANCE,
                    )?;
                    let next = self.latent_embeddings(&latent)?;
                    probed.push(("latent embedding".to_string(), next.clone()));
                    hidden = self.forward_one(&next, &mut cache)?;
                    history = latent;
                }
                probed.push(("decode hidden".to_string(), hidden));
                probed.push((
                    "latent".to_string(),
                    Tensor::from_vec(history, LATENT_DIM, &Device::Cpu)?,
                ));
                Ok(probed)
            },
        )
    }

    fn forward_one(&self, embeds: &Tensor, cache: &mut PhysicalPagedKvCache) -> Result<Tensor> {
        self.trunk.forward(&mut [MingForwardRow { embeds, cache }])
    }

    /// One utterance end to end on `cache`: prefill, generate latents until
    /// the stop head fires or the budget ends, then decode the audio.
    pub fn generate_physical(
        &self,
        request: &MingRequest,
        params: &MingGenerationParams,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<MingGenerationOutput> {
        params.validate()?;
        let prompt = self.prompt(request)?;
        let budget = params
            .max_new_tokens
            .min(Self::latent_room(prompt.positions));
        let started = Instant::now();
        let embeds = self.prompt_rows(&prompt.embeds)?;
        let mut hidden = host_f32(&self.forward_one(&embeds, cache)?)?;
        let mut rng = SplitMix64::for_stream(params.seed, MingNoiseStream::Flow);
        let mut history = prompt.history.clone();
        let mut latents = Vec::with_capacity(budget * LATENT_DIM);
        let mut stop_reason = MingStopReason::LatentBudget;
        for step in 0..budget {
            let noise = rng.standard_normals(LATENT_DIM);
            let latent =
                self.sample_latent(&hidden, &history, &noise[..LATENT_DIM], params.guidance)?;
            latents.extend_from_slice(&latent);
            if self.stops_at(&hidden, step) {
                stop_reason = MingStopReason::EndOfSpeech;
                break;
            }
            if step + 1 < budget {
                let next = self.latent_embeddings(&latent)?;
                hidden = host_f32(&self.forward_one(&next, cache)?)?;
            }
            history = latent;
        }
        let lm_ms = started.elapsed().as_secs_f32() * 1000.0;
        let codec_started = Instant::now();
        let count = latents.len() / LATENT_DIM;
        let mut stream = MingAudioStream::default();
        let mut samples = Vec::new();
        while let Some(window) = stream.next_window(count, true) {
            samples.extend(stream.decode(&self.decoder, &latents, window)?);
        }
        Ok(MingGenerationOutput {
            samples,
            sample_rate: SAMPLE_RATE,
            latents: count,
            stop_reason,
            prompt_positions: prompt.positions,
            lm_ms,
            codec_ms: codec_started.elapsed().as_secs_f32() * 1000.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_params_validate_their_ranges() {
        let mut params = MingGenerationParams {
            guidance: DEFAULT_GUIDANCE,
            max_new_tokens: 100,
            seed: 3,
        };
        assert!(params.validate().is_ok());
        params.seed = 0;
        assert!(params.validate().is_err());
        params.seed = 3;
        params.max_new_tokens = MAX_NEW_LATENTS + 1;
        assert!(params.validate().is_err());
        params.max_new_tokens = MIN_LATENTS - 1;
        assert!(params.validate().is_err());
        params.max_new_tokens = MIN_LATENTS;
        params.guidance = f32::NAN;
        assert!(params.validate().is_err());
    }

    #[test]
    fn the_precision_ladder_steps_down_to_f32() {
        assert_eq!(MingPrecision::ladder(DType::F32), vec![MingPrecision::F32]);
        let f16 = MingPrecision::ladder(DType::F16);
        assert_eq!(f16.len(), 3);
        assert_eq!((f16[0].lm, f16[0].flow), (DType::F16, DType::F16));
        assert_eq!((f16[1].lm, f16[1].flow), (DType::F16, DType::F32));
        assert_eq!(f16[2], MingPrecision::F32);
        let bf16 = MingPrecision::ladder(DType::BF16);
        assert_eq!(bf16.len(), 3);
        assert_eq!((bf16[0].lm, bf16[0].flow), (DType::BF16, DType::BF16));
        assert_eq!((bf16[1].lm, bf16[1].flow), (DType::BF16, DType::F32));
        assert!(bf16.iter().chain(&f16).all(|step| step.vae == DType::F32));
    }

    /// Every count from two to eleven requests' latents and embeddings in
    /// shared calls against each request alone, bit for bit, with mixed
    /// guidance: a shared call's rounding must not depend on how many
    /// requests share it.
    fn requests_together_match_each_alone(runtime: &MingRuntime) {
        let hidden_size = runtime.config().llm.hidden_size;
        let mut rng = SplitMix64::new(3);
        let mut normals = |count: usize| rng.standard_normals(count)[..count].to_vec();
        let rows = (0..11)
            .map(|index| {
                (
                    normals(hidden_size),
                    normals(LATENT_DIM),
                    normals(LATENT_DIM),
                    if index % 3 == 0 {
                        1.5
                    } else {
                        DEFAULT_GUIDANCE
                    },
                )
            })
            .collect::<Vec<_>>();
        let alone = rows
            .iter()
            .map(|(hidden, history, noise, guidance)| {
                runtime
                    .sample_latent(hidden, history, noise, *guidance)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_ne!(alone[0], alone[1]);
        let embeddings = |latents: &[f32]| {
            runtime
                .latent_embeddings(latents)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap()
        };
        let embedded_alone = alone
            .iter()
            .map(|latent| embeddings(latent).remove(0))
            .collect::<Vec<_>>();
        for count in 2..=rows.len() {
            let inputs = rows[..count]
                .iter()
                .map(|(hidden, history, noise, guidance)| MingLatentInput {
                    hidden,
                    history,
                    noise,
                    guidance: *guidance,
                })
                .collect::<Vec<_>>();
            let together = runtime.sample_latents(&inputs).unwrap();
            assert_eq!(together, alone[..count], "sampler, {count} requests");
            assert_eq!(
                embeddings(&alone[..count].concat()),
                embedded_alone[..count],
                "aggregator, {count} requests"
            );
        }
    }

    #[test]
    fn requests_sampled_together_match_each_alone_on_the_cpu() {
        requests_together_match_each_alone(&synthetic::tiny_runtime());
    }

    #[cfg(feature = "metal")]
    #[test]
    fn requests_sampled_together_match_each_alone_on_metal() {
        let Some((device, _lease)) = crate::backends::metal_test_device_or_skip(
            "requests_sampled_together_match_each_alone_on_metal",
        ) else {
            return;
        };
        requests_together_match_each_alone(&synthetic::tiny_runtime_on(&device, DType::F16));
    }

    #[test]
    fn latent_room_is_what_the_context_leaves() {
        assert_eq!(MingRuntime::latent_room(CONTEXT_POSITIONS), 0);
        assert_eq!(MingRuntime::latent_room(100), CONTEXT_POSITIONS - 100);
        assert_eq!(MingRuntime::latent_room(CONTEXT_POSITIONS + 5), 0);
    }
}

#[cfg(test)]
mod isolation_tests {
    use std::path::{Path, PathBuf};

    /// Walks the family's sources: the only architecture path any of them
    /// may name is this family's own.
    #[test]
    fn the_family_imports_no_other_architecture() {
        let root =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/models/architectures/ming_tts");
        fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    sources(&path, out);
                } else if path.extension().is_some_and(|ext| ext == "rs") {
                    out.push(path);
                }
            }
        }
        let mut files = Vec::new();
        sources(&root, &mut files);
        assert!(files.len() >= 8);
        // Every sibling family, so a relative path into one is caught too.
        let others = std::fs::read_dir(root.parent().unwrap())
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter_map(|path| Some(path.file_stem()?.to_str()?.to_string()))
            .filter(|name| name != "mod" && name != "ming_tts")
            .collect::<Vec<_>>();
        assert!(others.iter().any(|name| name == "chatterbox"));
        let needle = concat!("architectures", "::");
        let own = concat!("architectures", "::", "ming_tts");
        for file in files {
            let source = std::fs::read_to_string(&file).unwrap();
            for (offset, _) in source.match_indices(needle) {
                assert!(
                    source[offset..].starts_with(own),
                    "{} names another architecture at byte {offset}",
                    file.display()
                );
            }
            assert!(
                !source.contains(concat!("super::super::super::", "architectures")),
                "{}",
                file.display()
            );
            for other in &others {
                let path = format!("{other}::");
                for (offset, _) in source.match_indices(&path) {
                    let before = source[..offset].chars().next_back();
                    assert!(
                        before.is_some_and(|c| c.is_alphanumeric() || c == '_'),
                        "{} names the {other} family at byte {offset}",
                        file.display()
                    );
                }
            }
        }
    }
}
