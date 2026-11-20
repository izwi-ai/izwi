//! The loaded model and its request rows.
//!
//! A request is prepared on the host (resample, whole-clip log-mel, the
//! window plan, prompt ids, the decode budget) and then runs as a retained
//! row: resumable prefill spans, the first of which encodes the row's audio
//! windows, then decode steps batched with other rows. A row's state is its
//! paged KV, its audio embeddings and its host decode loop; a quantum
//! checkpoints the first and the last, and the audio embeddings only move
//! forward (a retried span reuses what was already encoded).
//!
//! The windows of every row of a prefill quantum are encoded together: the
//! encoder has one input shape, so windows of different requests share its
//! batches.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use serde_json::json;
use tokenizers::Tokenizer;

use super::checkpoint::SovCheckpoint;
use super::config::{SovTranscribeConfig, TOKENIZER_FILE};
use super::decoder::moe::{ExpertPath, MoeBackend};
use super::decoder::{DecoderTrace, SovDecoder, SpanRow, TraceRequest};
use super::encoder::{max_abs, SovEncoder, SovProjector};
use super::frontend::{SovMel, SovMelFrontend};
use super::precision::{backend_of, compute_dtype, require_serving_backend_for};
use super::prompt::{
    render_instruction, tokenize_prompt, PromptLayout, PromptSlot, SovControlTokens,
    SovPromptOptions,
};
use super::transcript::{trailing_loop, without_trailing_loop, SovFinishReason};
use crate::backends::{BackendKind, DeviceProfile};
use crate::catalog::ModelVariant;
use crate::error::{Error, Result};
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PhysicalPagedKvCheckpoint};
use crate::models::shared::detokenize::IncrementalDetokenizer;

/// Prompt and completion share this many positions (the reference server's
/// `max_model_len`).
pub const MAX_CONTEXT_TOKENS: usize = 32_768;
/// The card's completion budget.
pub const MAX_NEW_TOKENS: usize = 8_192;
/// The planned decode: what a request that omits its budget decodes at most
/// and books KV for. The card measures about 4 transcript tokens per second
/// of audio; 12 covers dense speech with room, plus a fixed margin.
pub const PLANNED_TOKENS_PER_SECOND: f64 = 12.0;
pub const PLANNED_DECODE_MARGIN: usize = 64;
/// The longest recording one request takes (the model's tested maximum).
pub const MAX_AUDIO_SECONDS: f32 = 1_800.0;

static NEXT_MODEL_IDENTITY: AtomicU64 = AtomicU64::new(1);

/// Relative L2 bound of the device-routed experts against the host-routed
/// reference at load (BF16 on CUDA; the two sum the same products in a
/// different order).
const MOE_CHECK_LIMIT: f64 = 2e-2;

/// `IZWI_SOV_TRANSCRIBE_PREFILL_SPAN`: the prompt tokens one row prefills in
/// one engine quantum (16 to 16,384).
pub const PREFILL_SPAN_ENV: &str = "IZWI_SOV_TRANSCRIBE_PREFILL_SPAN";

/// The prefill span per row and quantum: 512 tokens on CUDA (the whole
/// prompt of a one-window request, about 410 tokens), 192 on the CPU test
/// reference, or the environment's. Every span but a prompt's last is
/// exactly this long, alone or beside other rows.
pub fn prefill_span(backend: BackendKind) -> usize {
    std::env::var(PREFILL_SPAN_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|span| (16..=16_384).contains(span))
        .unwrap_or(match backend {
            BackendKind::Cuda => 512,
            BackendKind::Cpu | BackendKind::Metal => 192,
        })
}

/// Kill switch for packed prefill (`0`, `off`, `false`; read at load): the
/// rows of a prefill quantum then run their spans one after another, each
/// exactly as it runs alone. Packed, the rows share the dense products, so
/// a near-tied greedy token can depend on what else prefills beside it.
pub const BATCHED_PREFILL_ENV: &str = "IZWI_SOV_TRANSCRIBE_BATCHED_PREFILL";

/// `IZWI_SOV_TRANSCRIBE_DECODE_QUANTUM`: the most decode steps a row runs in
/// one engine quantum (1 to 64).
pub const DECODE_QUANTUM_ENV: &str = "IZWI_SOV_TRANSCRIBE_DECODE_QUANTUM";

/// Decode steps per engine quantum while nothing waits: 8 on CUDA, 4 on the
/// CPU test reference, or the environment's.
pub fn decode_quantum(backend: BackendKind) -> usize {
    std::env::var(DECODE_QUANTUM_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|steps| (1..=64).contains(steps))
        .unwrap_or(match backend {
            BackendKind::Cpu => 4,
            BackendKind::Cuda | BackendKind::Metal => 8,
        })
}

/// `IZWI_SOV_TRANSCRIBE_ENCODER_BATCH`: the most 30 s windows one encoder
/// call takes (1 to 64, default 8, the reference's chunk).
pub const ENCODER_BATCH_ENV: &str = "IZWI_SOV_TRANSCRIBE_ENCODER_BATCH";

pub fn encoder_batch() -> usize {
    std::env::var(ENCODER_BATCH_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|windows| (1..=64).contains(windows))
        .unwrap_or(8)
}

/// `min(8192, ceil(12 * seconds) + 64)`, the decode a request books.
pub fn planned_decode_tokens(audio_seconds: f32) -> usize {
    ((f64::from(audio_seconds.max(0.0)) * PLANNED_TOKENS_PER_SECOND).ceil() as usize
        + PLANNED_DECODE_MARGIN)
        .min(MAX_NEW_TOKENS)
}

/// What a request asks for.
#[derive(Debug, Clone, Default)]
pub struct SovTranscribeRequest {
    pub prompt: SovPromptOptions,
    /// The completion budget; the planned decode when omitted.
    pub max_new_tokens: Option<usize>,
}

/// A request prepared on the host.
pub struct SovPreparedArtifact {
    pub request: SovTranscribeRequest,
    pub instruction: String,
    pub layout: PromptLayout,
    pub max_new_tokens: usize,
    pub planned_decode: usize,
    pub duration_secs: f32,
    pub samples: usize,
    pub(crate) mel: SovMel,
    pub preparation_ms: f64,
}

impl std::fmt::Debug for SovPreparedArtifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SovPreparedArtifact")
            .field("prompt_tokens", &self.layout.len())
            .field("audio_tokens", &self.layout.audio_tokens)
            .field("duration_secs", &self.duration_secs)
            .finish()
    }
}

impl SovPreparedArtifact {
    pub fn prompt_tokens(&self) -> usize {
        self.layout.len()
    }

    pub fn windows(&self) -> usize {
        self.mel.windows
    }

    /// The context a row books: its prompt and its planned decode.
    pub fn planned_context(&self) -> usize {
        (self.layout.len() + self.planned_decode).min(MAX_CONTEXT_TOKENS)
    }

    /// The context the request may use at most.
    pub fn max_context(&self) -> usize {
        (self.layout.len() + self.max_new_tokens).min(MAX_CONTEXT_TOKENS)
    }

    pub fn host_bytes(&self) -> usize {
        self.mel.host_bytes() + self.layout.len() * 4
    }
}

/// Where a row's time went.
#[derive(Debug, Clone, Default)]
pub struct SovRowTiming {
    pub preparation_ms: f64,
    pub encoder_ms: f64,
    pub prefill_ms: f64,
    pub prefill_spans: u64,
    pub decode_ms: f64,
    pub decode_steps: u64,
}

impl SovRowTiming {
    pub fn to_json(&self) -> serde_json::Value {
        json!({
            "preparation_ms": self.preparation_ms,
            "encoder_ms": self.encoder_ms,
            "prefill_ms": self.prefill_ms,
            "prefill_spans": self.prefill_spans,
            "decode_ms": self.decode_ms,
            "decode_steps": self.decode_steps,
        })
    }
}

#[derive(Clone)]
pub struct SovDecodeLoop {
    generated: Vec<u32>,
    finished: Option<SovFinishReason>,
    max_new_tokens: usize,
    detokenizer: IncrementalDetokenizer,
    pending_text: String,
    /// Steps whose greedy margin was below 1e-4 (near-ties).
    near_ties: u64,
    /// Generated tokens fed back to the decoder.
    fed: usize,
}

impl SovDecodeLoop {
    pub fn generated(&self) -> &[u32] {
        &self.generated
    }

    pub fn finish_reason(&self) -> Option<SovFinishReason> {
        self.finished
    }

    pub fn finished(&self) -> bool {
        self.finished.is_some()
    }

    pub fn near_ties(&self) -> u64 {
        self.near_ties
    }
}

/// A rollback point of a row.
pub struct SovRetainedCheckpoint {
    kv: PhysicalPagedKvCheckpoint,
    prefill_cursor: usize,
    decode: Option<SovDecodeLoop>,
}

/// One request's retained state.
pub struct SovRetainedState {
    artifact: Arc<SovPreparedArtifact>,
    /// The row's audio embeddings `[audio_tokens, hidden]`, once encoded.
    audio: Option<Tensor>,
    prefill_cursor: usize,
    decode: Option<SovDecodeLoop>,
    context: usize,
    /// Kept across rollbacks: a rolled-back quantum's time was still spent.
    timing: SovRowTiming,
}

impl std::fmt::Debug for SovRetainedState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SovRetainedState")
            .field("prefill_cursor", &self.prefill_cursor)
            .field("prompt_tokens", &self.prompt_tokens())
            .field("decoding", &self.decode.is_some())
            .finish()
    }
}

impl SovRetainedState {
    pub fn prompt_tokens(&self) -> usize {
        self.artifact.prompt_tokens()
    }

    pub fn prefill_cursor(&self) -> usize {
        self.prefill_cursor
    }

    pub fn artifact(&self) -> &Arc<SovPreparedArtifact> {
        &self.artifact
    }

    pub fn finished(&self) -> bool {
        self.decode.as_ref().is_some_and(SovDecodeLoop::finished)
    }

    pub fn decode_loop(&self) -> Option<&SovDecodeLoop> {
        self.decode.as_ref()
    }

    pub fn timing(&self) -> &SovRowTiming {
        &self.timing
    }

    pub fn audio_encoded(&self) -> bool {
        self.audio.is_some()
    }

    /// Text decoded since the last call.
    pub fn take_text(&mut self) -> String {
        self.decode
            .as_mut()
            .map(|d| std::mem::take(&mut d.pending_text))
            .unwrap_or_default()
    }

    pub fn begin_quantum(&self, cache: &PhysicalPagedKvCache) -> SovRetainedCheckpoint {
        SovRetainedCheckpoint {
            kv: cache.logical_checkpoint(),
            prefill_cursor: self.prefill_cursor,
            decode: self.decode.clone(),
        }
    }

    pub fn rollback_quantum(
        &mut self,
        cache: &mut PhysicalPagedKvCache,
        checkpoint: &SovRetainedCheckpoint,
    ) -> Result<()> {
        self.prefill_cursor = checkpoint.prefill_cursor;
        self.decode = checkpoint.decode.clone();
        cache.restore_logical_checkpoint(checkpoint.kv.clone())
    }

    /// Validates the committed quantum: the KV and the row's clocks agree.
    pub fn commit_quantum(&self, cache: &PhysicalPagedKvCache) -> Result<()> {
        if cache.context_len() != self.fed_tokens() {
            return Err(Error::InferenceError(
                "SOV-Transcribe KV and row clocks diverged at commit".into(),
            ));
        }
        Ok(())
    }

    /// Tokens the decoder has consumed: the prefilled prompt and every
    /// generated token that was fed back.
    fn fed_tokens(&self) -> usize {
        self.prefill_cursor + self.decode.as_ref().map_or(0, |decode| decode.fed)
    }

    /// Stops the row (cancellation).
    pub fn stop(&mut self) {
        if let Some(decode) = self.decode.as_mut() {
            decode.finished.get_or_insert(SovFinishReason::Cancelled);
        }
    }
}

/// A row in a batched decode step.
pub struct SovStepRow<'a> {
    pub state: &'a mut SovRetainedState,
    pub cache: &'a mut PhysicalPagedKvCache,
}

/// A row in a prefill quantum, taking `span` prompt tokens.
pub struct SovPrefillRow<'a> {
    pub state: &'a mut SovRetainedState,
    pub cache: &'a mut PhysicalPagedKvCache,
    pub span: usize,
}

/// A finished row.
#[derive(Debug, Clone)]
pub struct SovTranscription {
    pub text: String,
    pub finish_reason: SovFinishReason,
    pub generated_tokens: usize,
    pub prompt_tokens: usize,
    pub audio_tokens: usize,
    pub windows: usize,
    pub timing: SovRowTiming,
}

#[derive(Default)]
struct SovDiagnostics {
    load_seconds: f64,
    head_tied: bool,
    /// The device-routed experts' relative error against the host-routed
    /// reference at load, when the check ran.
    moe_check: Option<f64>,
    windows_encoded: AtomicU64,
    encoder_batches: AtomicU64,
    encoder_us: AtomicU64,
    prefill_spans: AtomicU64,
    packed_prefills: AtomicU64,
    packed_prefill_rows: AtomicU64,
    prefill_us: AtomicU64,
    decode_steps: AtomicU64,
    decode_rows: AtomicU64,
    decode_us: AtomicU64,
    near_ties: AtomicU64,
    loop_stops: AtomicU64,
}

pub struct SovTranscribeModel {
    variant: Option<ModelVariant>,
    model_dir: PathBuf,
    device: Device,
    dtype: DType,
    config: SovTranscribeConfig,
    frontend: SovMelFrontend,
    encoder: SovEncoder,
    projector: SovProjector,
    decoder: SovDecoder,
    tokenizer: Tokenizer,
    control: SovControlTokens,
    identity: u64,
    prefill_span: usize,
    decode_quantum: usize,
    encoder_batch: usize,
    encoder_graphs:
        std::result::Result<crate::kernels::cuda::shape_graphs::ShapeGraphs<usize>, String>,
    batched_prefill: bool,
    diagnostics: SovDiagnostics,
}

impl std::fmt::Debug for SovTranscribeModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SovTranscribeModel")
            .field("model_dir", &self.model_dir)
            .field("device", &backend_of(&self.device))
            .field("dtype", &self.dtype)
            .finish()
    }
}

impl SovTranscribeModel {
    /// Loads the published model. It serves on CUDA only: any other device
    /// is refused before a weight is read.
    pub fn load(
        model_dir: &Path,
        variant: Option<ModelVariant>,
        device: &DeviceProfile,
    ) -> Result<Self> {
        require_serving_backend_for(model_dir, &device.device)?;
        Self::load_on(model_dir, variant, &device.device)
    }

    /// Loads on `device` without the serving gate: CUDA, or the CPU as the
    /// test reference for a synthetic checkpoint.
    pub fn load_on(
        model_dir: &Path,
        variant: Option<ModelVariant>,
        device: &Device,
    ) -> Result<Self> {
        let started = Instant::now();
        let dtype = compute_dtype(device)?;
        let config = SovTranscribeConfig::load(model_dir)?;
        let checkpoint = SovCheckpoint::open(model_dir, &config)?;
        let head_tied = checkpoint.head_is_tied()?;
        if !head_tied {
            return Err(Error::ModelLoadError(
                "SOV-Transcribe: `lm_head.weight` differs from the embedding; the family ties them"
                    .into(),
            ));
        }
        let tokenizer = Tokenizer::from_file(model_dir.join(TOKENIZER_FILE))
            .map_err(|err| Error::ModelLoadError(format!("SOV-Transcribe tokenizer: {err}")))?;
        let control =
            SovControlTokens::resolve(&tokenizer, config.audio_token_id, config.bos_token_id)?;
        for &eos in &config.eos_token_ids {
            if eos as usize >= config.decoder.vocab {
                return Err(Error::ModelLoadError(format!(
                    "stop token {eos} is outside the vocabulary"
                )));
            }
        }
        if !config.eos_token_ids.contains(&control.turn_end) {
            return Err(Error::ModelLoadError(
                "SOV-Transcribe: the end-of-turn token is not a stop token".into(),
            ));
        }
        let frontend = SovMelFrontend::new(config.frontend)?;
        let encoder = SovEncoder::load(&checkpoint, &config, device, dtype)?;
        let projector = SovProjector::load(&checkpoint, &config, device, dtype)?;
        let mut decoder = SovDecoder::load(&checkpoint, &config, device, dtype)?;
        drop(checkpoint);
        // The device-routed experts serve only after they agree with the
        // host-routed reference on a probe; otherwise the host path serves
        // every call and the diagnostics say why.
        let checked_path = match decoder.moe_backend() {
            MoeBackend::Host => None,
            MoeBackend::Device => Some(ExpertPath::Gathered),
            MoeBackend::Fused => Some(ExpertPath::Fused),
            MoeBackend::Auto if device.is_cuda() => Some(ExpertPath::Fused),
            MoeBackend::Auto => None,
        };
        let moe_check = match checked_path {
            Some(ExpertPath::Fused) if !decoder.fused_experts_supported() => {
                tracing::warn!(
                    "SOV-Transcribe: the fused expert kernels do not take this checkpoint's geometry; routing on the host"
                );
                decoder.set_moe_backend(MoeBackend::Host);
                None
            }
            Some(path) => {
                let error = decoder.check_expert_path(path, 4)?;
                if error.is_nan() || error > MOE_CHECK_LIMIT {
                    tracing::warn!(
                        "SOV-Transcribe: the {path:?} experts deviate from the host-routed reference by {error:.3e}; routing on the host"
                    );
                    decoder.set_moe_backend(MoeBackend::Host);
                }
                Some(error)
            }
            None => None,
        };
        let load_seconds = started.elapsed().as_secs_f64();
        let model = Self {
            variant,
            model_dir: model_dir.to_path_buf(),
            device: device.clone(),
            dtype,
            frontend,
            encoder,
            projector,
            decoder,
            tokenizer,
            control,
            identity: NEXT_MODEL_IDENTITY.fetch_add(1, Ordering::Relaxed),
            prefill_span: prefill_span(backend_of(device)),
            decode_quantum: decode_quantum(backend_of(device)),
            encoder_batch: encoder_batch(),
            encoder_graphs: super::graphs::resolve_encoder(device),
            batched_prefill: crate::env::flag_or(BATCHED_PREFILL_ENV, true),
            diagnostics: SovDiagnostics {
                load_seconds,
                head_tied,
                moe_check,
                ..Default::default()
            },
            config,
        };
        tracing::info!(
            "Loaded SOV-Transcribe from {:?} on {:?} ({:?}), {:.1} GiB resident, in {:.1} s",
            model_dir,
            backend_of(device),
            dtype,
            model.resident_bytes() as f64 / (1u64 << 30) as f64,
            load_seconds
        );
        Ok(model)
    }

    pub fn batched_prefill(&self) -> bool {
        self.batched_prefill
    }

    /// Switches packed prefill, for tests that compare the two.
    #[cfg(test)]
    pub(crate) fn set_batched_prefill(&mut self, on: bool) {
        self.batched_prefill = on;
    }

    pub fn prefill_span(&self) -> usize {
        self.prefill_span
    }

    pub fn decode_quantum(&self) -> usize {
        self.decode_quantum
    }

    pub fn config(&self) -> &SovTranscribeConfig {
        &self.config
    }

    pub fn variant(&self) -> Option<ModelVariant> {
        self.variant
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn model_identity(&self) -> u64 {
        self.identity
    }

    pub fn decoder(&self) -> &SovDecoder {
        &self.decoder
    }

    #[cfg(test)]
    pub(crate) fn decoder_mut(&mut self) -> &mut SovDecoder {
        &mut self.decoder
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn control_tokens(&self) -> &SovControlTokens {
        &self.control
    }

    pub fn max_context_tokens(&self) -> usize {
        MAX_CONTEXT_TOKENS
    }

    /// Seconds of one encoder window (30).
    pub fn window_seconds(&self) -> f32 {
        self.config.frontend.window_samples as f32 / self.config.frontend.sample_rate as f32
    }

    /// Audio tokens per second of a full window (12.5).
    pub fn audio_token_rate(&self) -> f32 {
        self.config.tokens_per_window() as f32 / self.window_seconds()
    }

    pub fn resident_bytes(&self) -> usize {
        self.encoder.resident_bytes()
            + self.projector.resident_bytes()
            + self.decoder.resident_bytes()
    }

    pub fn diagnostics_summary(&self) -> serde_json::Value {
        let d = &self.diagnostics;
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        json!({
            "family": "sov_transcribe_asr",
            "device_kind": match backend_of(&self.device) {
                BackendKind::Cpu => "cpu",
                BackendKind::Metal => "metal",
                BackendKind::Cuda => "cuda",
            },
            "dtype": match self.dtype {
                DType::BF16 => "bf16",
                _ => "f32",
            },
            "resident_bytes": self.resident_bytes(),
            "load_seconds": d.load_seconds,
            "head_tied": d.head_tied,
            "moe_backend": self.decoder.moe_backend().name(),
            "moe_fused_max_pairs": super::decoder::moe::FUSED_MAX_PAIRS,
            "decode_graphs": self.decoder.graph_diagnostics(),
            "moe_check": d.moe_check,
            "prefill_span": self.prefill_span,
            "decode_quantum": self.decode_quantum,
            "encoder_batch": self.encoder_batch,
            "encoder_graphs": match &self.encoder_graphs {
                Ok(graphs) => graphs.diagnostics(),
                Err(reason) => json!({ "off": reason }),
            },
            "batched_prefill": self.batched_prefill,
            "windows_encoded": load(&d.windows_encoded),
            "encoder_batches": load(&d.encoder_batches),
            "encoder_ms": load(&d.encoder_us) as f64 / 1e3,
            "prefill_spans": load(&d.prefill_spans),
            "packed_prefills": load(&d.packed_prefills),
            "packed_prefill_rows": load(&d.packed_prefill_rows),
            "prefill_ms": load(&d.prefill_us) as f64 / 1e3,
            "decode_steps": load(&d.decode_steps),
            "decode_rows": load(&d.decode_rows),
            "decode_ms": load(&d.decode_us) as f64 / 1e3,
            "near_ties": load(&d.near_ties),
            "loop_stops": load(&d.loop_stops),
        })
    }

    /// Device workspace of one row of a decode step.
    pub fn decode_workspace_per_row_bytes(&self) -> u64 {
        self.decoder.workspace_per_token_bytes() + (self.config.decoder.vocab * 4 * 2) as u64
    }

    /// Device workspace a prefill quantum of `spans` (each row's tokens)
    /// holds: the activations of every token, and per row one batch share
    /// of encoder windows.
    pub fn prefill_workspace_bytes(&self, spans: &[usize]) -> u64 {
        let tokens = spans.iter().sum::<usize>() as u64;
        tokens * self.decoder.workspace_per_token_bytes()
            + spans.len() as u64 * self.encoder.workspace_per_window_bytes()
    }

    /// [`Self::prepare`] for audio at any sample rate (resampled to 16 kHz).
    pub fn prepare_audio(
        &self,
        samples: &[f32],
        sample_rate: u32,
        request: SovTranscribeRequest,
    ) -> Result<SovPreparedArtifact> {
        let target = self.config.frontend.sample_rate as u32;
        if sample_rate == target {
            return self.prepare(samples, request);
        }
        let resampled = crate::audio::resample_mono_high_quality(samples, sample_rate, target)?;
        self.prepare(&resampled, request)
    }

    /// Host-side preparation of `samples` (mono, 16 kHz).
    pub fn prepare(
        &self,
        samples: &[f32],
        request: SovTranscribeRequest,
    ) -> Result<SovPreparedArtifact> {
        let started = Instant::now();
        let sample_rate = self.config.frontend.sample_rate;
        let duration_secs = samples.len() as f32 / sample_rate as f32;
        if duration_secs > MAX_AUDIO_SECONDS + 1.0 {
            return Err(Error::InvalidInput(format!(
                "SOV-Transcribe-1 takes at most {} minutes of audio in one request",
                MAX_AUDIO_SECONDS / 60.0
            )));
        }
        let instruction = render_instruction(&request.prompt)?;
        let mel = self.frontend.compute(samples)?;
        let audio_tokens = mel.windows * self.config.tokens_per_window();
        let layout = tokenize_prompt(&self.tokenizer, &self.control, &instruction, audio_tokens)?;
        let room = MAX_CONTEXT_TOKENS.saturating_sub(layout.len());
        let planned = planned_decode_tokens(duration_secs);
        let max_new_tokens = request
            .max_new_tokens
            .unwrap_or(planned)
            .min(MAX_NEW_TOKENS)
            .min(room);
        if max_new_tokens == 0 {
            return Err(Error::ContextLengthExceeded(format!(
                "the {}-token prompt leaves no room in the {MAX_CONTEXT_TOKENS}-token context",
                layout.len()
            )));
        }
        Ok(SovPreparedArtifact {
            planned_decode: planned.min(max_new_tokens),
            request,
            instruction,
            layout,
            max_new_tokens,
            duration_secs,
            samples: samples.len(),
            mel,
            preparation_ms: started.elapsed().as_secs_f64() * 1e3,
        })
    }

    /// A row for a prepared request, holding at most `context` positions.
    pub fn new_row(
        &self,
        artifact: Arc<SovPreparedArtifact>,
        context: usize,
    ) -> Result<SovRetainedState> {
        let context = context.min(MAX_CONTEXT_TOKENS);
        if artifact.prompt_tokens() >= context {
            return Err(Error::InvalidInput(format!(
                "a {}-token prompt does not fit its {context}-token row",
                artifact.prompt_tokens()
            )));
        }
        Ok(SovRetainedState {
            audio: None,
            prefill_cursor: 0,
            decode: None,
            context,
            timing: SovRowTiming {
                preparation_ms: artifact.preparation_ms,
                ..SovRowTiming::default()
            },
            artifact,
        })
    }

    /// Audio embeddings `[windows * tokens_per_window, hidden]` of each
    /// clip, every clip's windows sharing encoder batches of at most
    /// [`ENCODER_BATCH_ENV`] windows.
    pub fn encode_audio(&self, clips: &[&SovMel]) -> Result<Vec<Tensor>> {
        let started = Instant::now();
        let hidden = self.config.decoder.hidden;
        let per_window = self.config.tokens_per_window();
        let windows = clips
            .iter()
            .enumerate()
            .flat_map(|(clip, mel)| (0..mel.windows).map(move |window| (clip, window)))
            .collect::<Vec<_>>();
        let mut embedded = Vec::with_capacity(windows.len());
        for batch in windows.chunks(self.encoder_batch) {
            let features = batch
                .iter()
                .map(|&(clip, window)| clips[clip].window_tensor(window, 1, &self.device))
                .collect::<Result<Vec<_>>>()?;
            let projected = self.encode_windows(Tensor::cat(&features, 0)?)?;
            for row in 0..batch.len() {
                embedded.push(projected.get(row)?);
            }
            self.diagnostics
                .encoder_batches
                .fetch_add(1, Ordering::Relaxed);
        }
        let mut out = Vec::with_capacity(clips.len());
        let mut next = 0;
        for mel in clips {
            let clip = Tensor::cat(&embedded[next..next + mel.windows], 0)?;
            next += mel.windows;
            if clip.dims() != [mel.windows * per_window, hidden] {
                return Err(Error::InferenceError(format!(
                    "the audio tower produced {:?} for {} windows",
                    clip.dims(),
                    mel.windows
                )));
            }
            out.push(clip);
        }
        if let Some(first) = out.first() {
            if !max_abs(first)?.is_finite() {
                return Err(Error::InferenceError(
                    "the audio tower produced non-finite embeddings".into(),
                ));
            }
        }
        self.diagnostics
            .windows_encoded
            .fetch_add(windows.len() as u64, Ordering::Relaxed);
        self.diagnostics
            .encoder_us
            .fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        Ok(out)
    }

    /// The encoder and projector over `features` `[windows, n_mels,
    /// frames]`, through the batch size's CUDA graph when the graphs are on.
    fn encode_windows(&self, features: Tensor) -> Result<Tensor> {
        let body = |inputs: &[Tensor]| -> Result<Vec<Tensor>> {
            Ok(vec![self
                .projector
                .forward(&self.encoder.forward(&inputs[0])?)?])
        };
        let Ok(graphs) = &self.encoder_graphs else {
            return Ok(body(&[features])?.remove(0));
        };
        let windows = features.dim(0)?;
        let retained = (features.elem_count() * 4) as u64
            + (windows
                * self.config.tokens_per_window()
                * self.config.decoder.hidden
                * self.dtype.size_in_bytes()) as u64;
        graphs
            .run(windows, retained, &[features], &body)?
            .pop()
            .ok_or_else(|| Error::InferenceError("the encoder graph returned nothing".into()))
    }

    /// Encodes the audio of every row that has not been encoded yet.
    fn ensure_audio(&self, states: &mut [&mut SovRetainedState]) -> Result<()> {
        let pending = states
            .iter()
            .enumerate()
            .filter(|(_, state)| state.audio.is_none())
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if pending.is_empty() {
            return Ok(());
        }
        let started = Instant::now();
        let artifacts = pending
            .iter()
            .map(|&index| states[index].artifact.clone())
            .collect::<Vec<_>>();
        let clips = artifacts.iter().map(|a| &a.mel).collect::<Vec<_>>();
        let embedded = self.encode_audio(&clips)?;
        let elapsed = started.elapsed().as_secs_f64() * 1e3;
        for (&index, audio) in pending.iter().zip(embedded) {
            states[index].audio = Some(audio);
            states[index].timing.encoder_ms += elapsed;
        }
        Ok(())
    }

    /// Embeddings `[len, hidden]` of prompt positions `start..start + len`
    /// of a row whose audio is encoded.
    fn span_embeddings(
        &self,
        state: &SovRetainedState,
        start: usize,
        len: usize,
    ) -> Result<Tensor> {
        let layout = &state.artifact.layout;
        let audio = state.audio.as_ref().ok_or_else(|| {
            Error::InferenceError("a SOV-Transcribe row prefills before its audio".into())
        })?;
        let mut parts = Vec::new();
        let mut text = Vec::new();
        let mut span: Option<(usize, usize)> = None;
        for position in start..start + len {
            match layout.slot(position) {
                PromptSlot::Text(id) => {
                    if let Some((first, count)) = span.take() {
                        parts.push(audio.narrow(0, first, count)?);
                    }
                    text.push(id);
                }
                PromptSlot::Audio(index) => {
                    if !text.is_empty() {
                        parts.push(self.decoder.embed_text(&text)?);
                        text.clear();
                    }
                    span = Some(match span {
                        Some((first, count)) => (first, count + 1),
                        None => (index, 1),
                    });
                }
            }
        }
        if let Some((first, count)) = span {
            parts.push(audio.narrow(0, first, count)?);
        }
        if !text.is_empty() {
            parts.push(self.decoder.embed_text(&text)?);
        }
        Ok(Tensor::cat(&parts, 0)?)
    }

    /// Prefills the next `span` prompt tokens of a row; the last span starts
    /// its decode loop.
    pub fn retained_prefill(
        &self,
        state: &mut SovRetainedState,
        cache: &mut PhysicalPagedKvCache,
        span: usize,
    ) -> Result<()> {
        self.prefill_rows(&mut [SovPrefillRow { state, cache, span }], None)
    }

    /// [`Self::retained_prefill`] recording `trace`.
    pub fn retained_prefill_traced(
        &self,
        state: &mut SovRetainedState,
        cache: &mut PhysicalPagedKvCache,
        span: usize,
        trace: (&TraceRequest, &mut DecoderTrace),
    ) -> Result<()> {
        self.prefill_rows(&mut [SovPrefillRow { state, cache, span }], Some(trace))
    }

    /// Prefills each row's next `span` prompt tokens; a row's last span
    /// starts its decode loop. The rows' audio windows are encoded together
    /// first. With [`BATCHED_PREFILL_ENV`] on, two or more rows then run as
    /// one packed forward; an error leaves every row's KV and cursor where
    /// they were. One row, or the switch off, runs each span alone.
    pub fn retained_prefill_batch(&self, rows: &mut [SovPrefillRow<'_>]) -> Result<()> {
        if rows.len() < 2 || self.batched_prefill {
            return self.prefill_rows(rows, None);
        }
        {
            let mut states = rows
                .iter_mut()
                .map(|row| &mut *row.state)
                .collect::<Vec<_>>();
            self.ensure_audio(&mut states)?;
        }
        for row in rows.iter_mut() {
            self.prefill_rows(std::slice::from_mut(row), None)?;
        }
        Ok(())
    }

    fn prefill_rows(
        &self,
        rows: &mut [SovPrefillRow<'_>],
        trace: Option<(&TraceRequest, &mut DecoderTrace)>,
    ) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        for row in rows.iter() {
            let (start, prompt) = (row.state.prefill_cursor, row.state.prompt_tokens());
            if row.state.decode.is_some() || row.span == 0 || start + row.span > prompt {
                return Err(Error::InvalidInput(format!(
                    "SOV-Transcribe prefill span {start}+{} does not fit its {prompt}-token prompt",
                    row.span
                )));
            }
        }
        {
            let mut states = rows
                .iter_mut()
                .map(|row| &mut *row.state)
                .collect::<Vec<_>>();
            self.ensure_audio(&mut states)?;
        }
        let started = Instant::now();
        let mut spans = Vec::with_capacity(rows.len());
        for row in rows.iter_mut() {
            let embeds = self.span_embeddings(row.state, row.state.prefill_cursor, row.span)?;
            spans.push(SpanRow {
                embeds,
                cache: &mut *row.cache,
            });
        }
        let hidden = self.decoder.forward_rows(&mut spans, trace)?;
        drop(spans);
        let ending = rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row.state.prefill_cursor + row.span == row.state.prompt_tokens())
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let picks = if ending.is_empty() {
            Vec::new()
        } else {
            let index = Tensor::from_vec(
                ending.iter().map(|&row| row as u32).collect::<Vec<_>>(),
                ending.len(),
                &self.device,
            )?;
            self.decoder.greedy(&hidden.index_select(&index, 0)?)?
        };
        for row in rows.iter_mut() {
            row.state.prefill_cursor += row.span;
        }
        for (&index, (token, margin)) in ending.iter().zip(picks) {
            self.start_decode(rows[index].state, token, margin)?;
        }
        let elapsed = started.elapsed();
        for row in rows.iter_mut() {
            row.state.timing.prefill_ms += elapsed.as_secs_f64() * 1e3;
            row.state.timing.prefill_spans += 1;
        }
        let d = &self.diagnostics;
        d.prefill_spans
            .fetch_add(rows.len() as u64, Ordering::Relaxed);
        if rows.len() > 1 {
            d.packed_prefills.fetch_add(1, Ordering::Relaxed);
            d.packed_prefill_rows
                .fetch_add(rows.len() as u64, Ordering::Relaxed);
        }
        d.prefill_us
            .fetch_add(elapsed.as_micros() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// Starts a row's decode loop from the token its prompt's last span
    /// picked.
    fn start_decode(&self, state: &mut SovRetainedState, token: u32, margin: f32) -> Result<()> {
        let budget = state
            .artifact
            .max_new_tokens
            .min(state.context.saturating_sub(state.prompt_tokens()));
        let mut decode = SovDecodeLoop {
            generated: Vec::new(),
            finished: None,
            max_new_tokens: budget,
            detokenizer: IncrementalDetokenizer::new(),
            pending_text: String::new(),
            near_ties: 0,
            fed: 0,
        };
        self.accept(&mut decode, token, margin)?;
        state.decode = Some(decode);
        Ok(())
    }

    fn decode_text(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|err| Error::TokenizationError(err.to_string()))
    }

    /// Takes `token` into a row's loop: stop, budget and loop checks, and
    /// the streamed text delta.
    fn accept(&self, decode: &mut SovDecodeLoop, token: u32, margin: f32) -> Result<()> {
        if margin < 1e-4 {
            decode.near_ties += 1;
            self.diagnostics.near_ties.fetch_add(1, Ordering::Relaxed);
        }
        if self.config.eos_token_ids.contains(&token) {
            decode.finished = Some(SovFinishReason::Stop);
        } else {
            decode.generated.push(token);
            if decode.generated.len() >= decode.max_new_tokens {
                decode.finished = Some(SovFinishReason::Length);
            } else if trailing_loop(&decode.generated).is_some() {
                decode.finished = Some(SovFinishReason::Repetition);
                self.diagnostics.loop_stops.fetch_add(1, Ordering::Relaxed);
            }
        }
        let decode_fn = |ids: &[u32]| self.decode_text(ids);
        let delta = if decode.finished.is_some() {
            decode.detokenizer.finish(&decode.generated, decode_fn)?
        } else {
            decode.detokenizer.step(&decode.generated, decode_fn)?
        };
        decode.pending_text.push_str(&delta);
        Ok(())
    }

    /// One decode step for every unfinished row, batched. Returns whether
    /// each row has finished.
    pub fn retained_decode_batch(&self, rows: &mut [SovStepRow<'_>]) -> Result<Vec<bool>> {
        let mut active = Vec::new();
        for (index, row) in rows.iter().enumerate() {
            let decode = row.state.decode.as_ref().ok_or_else(|| {
                Error::InvalidInput("a SOV-Transcribe row decodes before its prefill".into())
            })?;
            if !decode.finished() {
                active.push(index);
            }
        }
        if !active.is_empty() {
            let started = Instant::now();
            let mut tokens = Vec::with_capacity(active.len());
            for &index in &active {
                tokens.push(
                    *rows[index]
                        .state
                        .decode
                        .as_ref()
                        .and_then(|d| d.generated.last())
                        .ok_or_else(|| {
                            Error::InferenceError("an unfinished row has no token to feed".into())
                        })?,
                );
            }
            let embeds = self.decoder.embed_text(&tokens)?;
            let mut spans = Vec::with_capacity(active.len());
            let mut position = 0;
            for (index, row) in rows.iter_mut().enumerate() {
                if !active.contains(&index) {
                    continue;
                }
                spans.push(SpanRow {
                    embeds: embeds.narrow(0, position, 1)?,
                    cache: &mut *row.cache,
                });
                position += 1;
            }
            let hidden = self.decoder.forward_rows(&mut spans, None)?;
            drop(spans);
            let next = self.decoder.greedy(&hidden)?;
            for (&index, (token, margin)) in active.iter().zip(next) {
                let decode = rows[index].state.decode.as_mut().expect("checked above");
                decode.fed += 1;
                self.accept(decode, token, margin)?;
            }
            let elapsed = started.elapsed();
            for &index in &active {
                let timing = &mut rows[index].state.timing;
                timing.decode_ms += elapsed.as_secs_f64() * 1e3;
                timing.decode_steps += 1;
            }
            let d = &self.diagnostics;
            d.decode_steps.fetch_add(1, Ordering::Relaxed);
            d.decode_rows
                .fetch_add(active.len() as u64, Ordering::Relaxed);
            d.decode_us
                .fetch_add(elapsed.as_micros() as u64, Ordering::Relaxed);
        }
        Ok(rows.iter().map(|row| row.state.finished()).collect())
    }

    /// The row's transcription once finished.
    pub fn finish_row(&self, state: &SovRetainedState) -> Result<SovTranscription> {
        let decode = state
            .decode
            .as_ref()
            .filter(|d| d.finished())
            .ok_or_else(|| Error::InferenceError("the row has not finished".into()))?;
        let finish_reason = decode.finished.expect("finished");
        let tokens = if finish_reason == SovFinishReason::Repetition {
            without_trailing_loop(&decode.generated)
        } else {
            &decode.generated
        };
        Ok(SovTranscription {
            text: self.decode_text(tokens)?.trim().to_string(),
            finish_reason,
            generated_tokens: decode.generated.len(),
            prompt_tokens: state.prompt_tokens(),
            audio_tokens: state.artifact.layout.audio_tokens,
            windows: state.artifact.windows(),
            timing: state.timing.clone(),
        })
    }

    /// One request end to end on a caller-owned cache, in prefill spans of
    /// `span` tokens (tests and parity runs).
    pub fn transcribe_with_cache(
        &self,
        samples: &[f32],
        request: SovTranscribeRequest,
        cache: &mut PhysicalPagedKvCache,
        span: usize,
    ) -> Result<SovTranscription> {
        let artifact = Arc::new(self.prepare(samples, request)?);
        let context = artifact.max_context().min(cache.capacity_tokens());
        let mut state = self.new_row(artifact, context)?;
        while state.prefill_cursor < state.prompt_tokens() {
            let n = span
                .max(1)
                .min(state.prompt_tokens() - state.prefill_cursor);
            self.retained_prefill(&mut state, cache, n)?;
        }
        while !state.finished() {
            self.retained_decode_batch(&mut [SovStepRow {
                state: &mut state,
                cache,
            }])?;
        }
        self.finish_row(&state)
    }
}
