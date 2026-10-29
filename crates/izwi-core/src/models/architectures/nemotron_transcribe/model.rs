//! The loaded model and its request rows.
//!
//! A request is prepared on the host (decode and resample, log-mel and
//! normalization statistics, the window plan, prompt ids, the decode
//! budget) and then runs as a retained row: resumable prefill spans, each
//! encoding the perception windows its audio positions need, then decode
//! steps batched with other rows. A row's state is its Mamba state, its
//! paged KV, its perception stream and its host decode loop; a quantum
//! checkpoints the first, second and fourth, and the stream only moves
//! forward (a retried span reuses the windows it already encoded).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use candle_core::{Device, Tensor};
use serde_json::json;
use tokenizers::Tokenizer;

use super::checkpoint::NemotronCheckpoint;
use super::config::{NemotronTranscribeConfig, TOKENIZER_FILE};
use super::ctc::{
    align, clean_sot_transcript, diarization_segments, prepare as prepare_alignment,
    CtcAlignmentConfig, CtcAlignmentInputs, CtcTimestampError, CtcTimestamps, CtcTokenizer,
    SpeakerActivity,
};
use super::ctc_head::NemotronCtcHead;
use super::decoder::{DecodeRow, DecoderTrace, NemotronDecoder, NemotronMambaState, TraceRequest};
use super::frontend::NemotronMelFrontend;
use super::linear::Residency;
use super::perception::{NemotronPerception, PerceptionStream};
use super::precision::Fp8Mode;
use super::prompt::{
    render_instruction, tokenize_prompt, NemotronPromptOptions, PromptLayout, PromptSlot,
};
use super::transcript::{ends_in_loop, parse_transcript, NemotronFinishReason, ParsedTranscript};
use crate::backends::state::{
    PhysicalStateSequenceId, PhysicalStateTransactionId, TensorStateArena,
};
use crate::backends::DeviceProfile;
use crate::catalog::ModelVariant;
use crate::error::{Error, Result};
use crate::models::shared::attention::physical::{PhysicalPagedKvCache, PhysicalPagedKvCheckpoint};
use crate::models::shared::detokenize::IncrementalDetokenizer;

/// The model's context: prompt and completion share 65,536 positions.
pub const MAX_CONTEXT_TOKENS: usize = 65_536;
/// The card's completion budget.
pub const MAX_NEW_TOKENS: usize = 16_384;
/// The planned decode for KV booking: 8 tokens per second of audio covers
/// CJK and tag-heavy output, plus a fixed margin.
pub const PLANNED_TOKENS_PER_SECOND: f64 = 8.0;
pub const PLANNED_DECODE_MARGIN: usize = 128;
/// The longest recording one request takes (the model's trained maximum and
/// izwi's ASR cap).
pub const MAX_AUDIO_SECONDS: f32 = 3_600.0;
/// Relative L2 bound of the load-time checks of resident FP8 projections
/// (Q8_0 re-quantization on the CPU is the loosest).
const PROJECTION_CHECK_LIMIT: f64 = 2e-2;
/// Relative L2 bound of the device Mamba2 scan against the CPU scan.
const SCAN_CHECK_LIMIT: f64 = 1e-4;

static NEXT_MODEL_IDENTITY: AtomicU64 = AtomicU64::new(1);

/// `IZWI_NEMOTRON_TRANSCRIBE_PREFILL_SPAN`: the most prompt tokens one row
/// prefills in one engine quantum (16 to 16,384).
pub const PREFILL_SPAN_ENV: &str = "IZWI_NEMOTRON_TRANSCRIBE_PREFILL_SPAN";

/// The prefill span per row and quantum: 2,048 tokens on CUDA, 512 on CPU
/// and Metal (a bounded dispatch between other users' decode steps; an
/// hour's 45,000-token prompt is 23 or 88 spans), or the environment's.
pub fn prefill_span(backend: crate::backends::BackendKind) -> usize {
    std::env::var(PREFILL_SPAN_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|span| (16..=16_384).contains(span))
        .unwrap_or(match backend {
            crate::backends::BackendKind::Cuda => 2_048,
            crate::backends::BackendKind::Cpu | crate::backends::BackendKind::Metal => 512,
        })
}

/// `min(16384, ceil(8 * seconds) + 128)`, the decode a request books.
pub fn planned_decode_tokens(audio_seconds: f32) -> usize {
    ((f64::from(audio_seconds.max(0.0)) * PLANNED_TOKENS_PER_SECOND).ceil() as usize
        + PLANNED_DECODE_MARGIN)
        .min(MAX_NEW_TOKENS)
}

/// What a request asks for.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NemotronTranscribeRequest {
    pub prompt: NemotronPromptOptions,
    /// The completion budget (the card's 16,384 when absent).
    pub max_new_tokens: Option<usize>,
    /// Keep what the CTC timestamp stage needs.
    pub word_timestamps: bool,
}

/// A request prepared on the host.
pub struct NemotronPreparedArtifact {
    pub request: NemotronTranscribeRequest,
    pub instruction: String,
    pub layout: PromptLayout,
    pub max_new_tokens: usize,
    pub planned_decode: usize,
    pub duration_secs: f32,
    pub samples: usize,
    pub(crate) mel: super::frontend::NemotronMel,
}

impl std::fmt::Debug for NemotronPreparedArtifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NemotronPreparedArtifact")
            .field("prompt_tokens", &self.layout.len())
            .field("audio_tokens", &self.layout.audio_tokens)
            .field("max_new_tokens", &self.max_new_tokens)
            .finish()
    }
}

impl NemotronPreparedArtifact {
    pub fn prompt_tokens(&self) -> usize {
        self.layout.len()
    }

    /// Positions the row books: the prompt and its planned decode.
    pub fn planned_context(&self) -> usize {
        self.layout.len() + self.planned_decode
    }

    /// Positions the row may reach: the prompt and its whole budget.
    pub fn max_context(&self) -> usize {
        self.layout.len() + self.max_new_tokens
    }

    pub fn host_bytes(&self) -> usize {
        self.mel.host_bytes()
    }
}

/// The host decode loop of one row.
#[derive(Debug, Clone)]
pub struct NemotronDecodeLoop {
    generated: Vec<u32>,
    finished: Option<NemotronFinishReason>,
    max_new_tokens: usize,
    detokenizer: IncrementalDetokenizer,
    pending_text: String,
    /// Steps whose greedy margin was below 1e-4 (near-ties).
    near_ties: u64,
}

impl NemotronDecodeLoop {
    pub fn generated(&self) -> &[u32] {
        &self.generated
    }

    pub fn finish_reason(&self) -> Option<NemotronFinishReason> {
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
pub struct NemotronRetainedCheckpoint {
    mamba: NemotronMambaState,
    kv: PhysicalPagedKvCheckpoint,
    prefill_cursor: usize,
    decode: Option<NemotronDecodeLoop>,
}

/// One request's retained state.
pub struct NemotronRetainedState {
    artifact: Arc<NemotronPreparedArtifact>,
    mamba: NemotronMambaState,
    perception: PerceptionStream,
    prefill_cursor: usize,
    decode: Option<NemotronDecodeLoop>,
    context: usize,
    tensor_sequence: Option<PhysicalStateSequenceId>,
}

impl std::fmt::Debug for NemotronRetainedState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NemotronRetainedState")
            .field("prefill_cursor", &self.prefill_cursor)
            .field("prompt_tokens", &self.artifact.prompt_tokens())
            .field("finished", &self.finished())
            .finish()
    }
}

impl NemotronRetainedState {
    pub fn prompt_tokens(&self) -> usize {
        self.artifact.prompt_tokens()
    }

    pub fn prefill_cursor(&self) -> usize {
        self.prefill_cursor
    }

    pub fn artifact(&self) -> &Arc<NemotronPreparedArtifact> {
        &self.artifact
    }

    pub fn finished(&self) -> bool {
        self.decode
            .as_ref()
            .is_some_and(NemotronDecodeLoop::finished)
    }

    pub fn decode_loop(&self) -> Option<&NemotronDecodeLoop> {
        self.decode.as_ref()
    }

    pub fn perception(&self) -> &PerceptionStream {
        &self.perception
    }

    /// Text decoded since the last call (tags removed).
    pub fn take_text(&mut self) -> String {
        self.decode
            .as_mut()
            .map(|d| std::mem::take(&mut d.pending_text))
            .unwrap_or_default()
    }

    pub fn begin_quantum(&self, cache: &PhysicalPagedKvCache) -> NemotronRetainedCheckpoint {
        NemotronRetainedCheckpoint {
            mamba: self.mamba.clone(),
            kv: cache.logical_checkpoint(),
            prefill_cursor: self.prefill_cursor,
            decode: self.decode.clone(),
        }
    }

    pub fn rollback_quantum(
        &mut self,
        cache: &mut PhysicalPagedKvCache,
        checkpoint: &NemotronRetainedCheckpoint,
    ) -> Result<()> {
        self.mamba = checkpoint.mamba.clone();
        self.prefill_cursor = checkpoint.prefill_cursor;
        self.decode = checkpoint.decode.clone();
        cache.restore_logical_checkpoint(checkpoint.kv.clone())
    }

    /// Validates the committed quantum: the KV, Mamba and prefill clocks
    /// agree.
    pub fn commit_quantum(&self, cache: &PhysicalPagedKvCache) -> Result<()> {
        if cache.context_len() as u64 != self.mamba.cursor() {
            return Err(Error::InferenceError(
                "Nemotron 3.5 Transcribe KV and Mamba clocks diverged at commit".into(),
            ));
        }
        Ok(())
    }

    pub fn bind_tensor_sequence(&mut self, sequence: u64) -> Result<()> {
        let sequence = PhysicalStateSequenceId::new(sequence)?;
        if self.tensor_sequence.is_some_and(|bound| bound != sequence) {
            return Err(Error::InferenceError(
                "Nemotron 3.5 Transcribe tensor-state sequence changed".into(),
            ));
        }
        self.tensor_sequence = Some(sequence);
        Ok(())
    }

    pub(crate) fn restore_mamba(
        &mut self,
        arena: &TensorStateArena,
        model: &NemotronTranscribeModel,
    ) -> Result<()> {
        let sequence = self.tensor_sequence.ok_or_else(|| {
            Error::InferenceError("Nemotron 3.5 Transcribe row has no tensor-state sequence".into())
        })?;
        self.mamba.restore(arena, sequence, model.decoder.config())
    }

    pub(crate) fn stage_mamba(&self, arena: &TensorStateArena, transaction: u64) -> Result<()> {
        self.mamba
            .stage(arena, PhysicalStateTransactionId::new(transaction)?)
    }

    /// Stops the row (cancellation).
    pub fn stop(&mut self) {
        if let Some(decode) = self.decode.as_mut() {
            decode
                .finished
                .get_or_insert(NemotronFinishReason::Cancelled);
        }
    }
}

/// A row in a batched decode step.
pub struct NemotronStepRow<'a> {
    pub state: &'a mut NemotronRetainedState,
    pub cache: &'a mut PhysicalPagedKvCache,
}

/// A finished transcription.
#[derive(Debug, Clone)]
pub struct NemotronTranscription {
    pub transcript: ParsedTranscript,
    pub finish_reason: NemotronFinishReason,
    pub generated_tokens: usize,
    pub prompt_tokens: usize,
    pub audio_tokens: usize,
    pub windows: usize,
    /// Word timestamps and diarization segments, when the request asked.
    pub timestamps: Option<CtcTimestamps>,
}

/// One speaker turn of a finished transcript, timed when the CTC stage
/// aligned its words.
#[derive(Debug, Clone, PartialEq)]
pub struct NemotronTimedTurn {
    /// The `<spk:N>` tag; `None` for text before the first tag.
    pub speaker: Option<u32>,
    pub text: String,
    pub start: Option<f64>,
    pub end: Option<f64>,
}

impl NemotronTranscription {
    /// The transcript's speaker turns, each timed from its own words: the
    /// aligner's words of the turn's speaker, taken in order while their
    /// text (whitespace removed, so character units of Chinese and Japanese
    /// match too) continues the turn's text. A turn whose words do not match
    /// stays untimed.
    pub fn timed_turns(&self) -> Vec<NemotronTimedTurn> {
        // Words carry upstream's `str(tag)`: "None" before the first tag,
        // and "0" throughout a transcript without tags (one untagged turn).
        let tagged = self
            .transcript
            .turns
            .iter()
            .any(|turn| turn.speaker.is_some());
        let mut streams: std::collections::HashMap<Option<u32>, std::collections::VecDeque<_>> =
            std::collections::HashMap::new();
        for word in self
            .timestamps
            .iter()
            .filter(|timestamps| timestamps.error.is_none())
            .flat_map(|timestamps| timestamps.words.iter())
            // Event and style tags are aligned as units but are not text.
            .filter(|word| !(word.word.starts_with('<') && word.word.ends_with('>')))
        {
            let speaker = tagged.then(|| word.speaker.parse::<u32>().ok()).flatten();
            streams.entry(speaker).or_default().push_back(word);
        }
        let squash = |text: &str| {
            text.chars()
                .filter(|c| !c.is_whitespace())
                .collect::<String>()
        };
        self.transcript
            .turns
            .iter()
            .map(|turn| {
                let target = squash(&turn.text);
                let mut matched = String::new();
                let (mut start, mut end) = (None, None);
                if let Some(stream) = streams.get_mut(&turn.speaker) {
                    while let Some(word) = stream.front() {
                        let next = format!("{matched}{}", squash(&word.word));
                        if next.is_empty() || !target.starts_with(&next) {
                            break;
                        }
                        start = start.or(Some(word.start));
                        end = Some(word.end);
                        matched = next;
                        stream.pop_front();
                    }
                }
                NemotronTimedTurn {
                    speaker: turn.speaker,
                    text: turn.text.clone(),
                    start,
                    end,
                }
            })
            .collect()
    }
}

/// The CTC timestamp stage: the head, its tokenizer and NeMo's aligner
/// settings.
struct CtcStage {
    head: NemotronCtcHead,
    tokenizer: CtcTokenizer,
    config: CtcAlignmentConfig,
}

/// Load-time checks and counters.
#[derive(Debug, Default)]
struct NemotronDiagnostics {
    load_seconds: f64,
    projection_checks: Vec<(String, f64)>,
    scan_check: Option<f64>,
    frontend_checked: bool,
    prefill_spans: AtomicU64,
    decode_steps: AtomicU64,
    windows_encoded: AtomicU64,
    near_ties: AtomicU64,
    loop_stops: AtomicU64,
}

pub struct NemotronTranscribeModel {
    variant: Option<ModelVariant>,
    model_dir: PathBuf,
    config: NemotronTranscribeConfig,
    device: Device,
    residency: Residency,
    fp8_mode: Fp8Mode,
    frontend: NemotronMelFrontend,
    perception: NemotronPerception,
    decoder: NemotronDecoder,
    tokenizer: Tokenizer,
    /// The tokenizer's special tokens, removed from decoded text.
    special_tokens: Vec<String>,
    /// Present when the checkpoint has the CTC artifact.
    ctc: Option<CtcStage>,
    audio_token_id: u32,
    /// Added `<…>` tokens, left out of streamed text.
    tag_ids: std::collections::HashSet<u32>,
    identity: u64,
    diagnostics: NemotronDiagnostics,
}

impl std::fmt::Debug for NemotronTranscribeModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NemotronTranscribeModel")
            .field("model_dir", &self.model_dir)
            .field("residency", &self.residency)
            .field("fp8_mode", &self.fp8_mode)
            .finish()
    }
}

impl NemotronTranscribeModel {
    pub fn load(
        model_dir: &Path,
        variant: Option<ModelVariant>,
        device: &DeviceProfile,
    ) -> Result<Self> {
        Self::load_on(model_dir, variant, &device.device)
    }

    /// Loads on `device` (CPU, Metal, or CUDA SM80+).
    pub fn load_on(
        model_dir: &Path,
        variant: Option<ModelVariant>,
        device: &Device,
    ) -> Result<Self> {
        let started = Instant::now();
        let residency = Residency::of(device);
        if residency == Residency::Cuda
            && !crate::kernels::cuda::fp8::device_is_sm80_or_newer(device)
        {
            return Err(Error::ModelLoadError(
                "Nemotron 3.5 Transcribe needs a CUDA GPU of compute capability 8.0 or newer"
                    .into(),
            ));
        }
        let fp8_mode = Fp8Mode::from_env()?;
        let config = NemotronTranscribeConfig::load(model_dir)?;
        let checkpoint = NemotronCheckpoint::open(model_dir)?;
        checkpoint.validate_census(&config)?;
        let fe = config.frontend;
        let window = checkpoint.f32_values(
            "perception.preprocessor.featurizer.window",
            &[fe.win_length],
        )?;
        let fb = checkpoint.f32_values(
            "perception.preprocessor.featurizer.fb",
            &[1, fe.n_mels, fe.n_fft / 2 + 1],
        )?;
        let frontend = NemotronMelFrontend::new(fe, window, fb)?;
        let perception = NemotronPerception::load(&checkpoint, &config, device, fp8_mode)?;
        let decoder = NemotronDecoder::load(&checkpoint, &config, device, fp8_mode)?;
        let tokenizer = Tokenizer::from_file(model_dir.join(TOKENIZER_FILE)).map_err(|err| {
            Error::ModelLoadError(format!("Nemotron 3.5 Transcribe tokenizer: {err}"))
        })?;
        let audio_token_id = tokenizer.token_to_id(&config.audio_token).ok_or_else(|| {
            Error::ModelLoadError(format!("the tokenizer has no {} token", config.audio_token))
        })?;
        for &eos in &config.eos_token_ids {
            if eos as usize >= config.decoder.vocab {
                return Err(Error::ModelLoadError(format!(
                    "stop token {eos} is outside the vocabulary"
                )));
            }
        }
        let tag_ids = tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter(|(_, token)| token.content.starts_with('<') && token.content.ends_with('>'))
            .map(|(id, _)| id)
            .collect();
        // Decoded text is cleaned of these as vLLM cleans it (speaker, event
        // and style tags stay for the transcript parser and the aligner).
        let special_tokens = tokenizer
            .get_added_tokens_decoder()
            .into_values()
            .filter(|token| token.special)
            .map(|token| token.content)
            .collect::<Vec<_>>();

        // Self-checks: each resident projection kind against the exactly
        // dequantized checkpoint weights (before the mapping is dropped),
        // and the device scan against the CPU scan.
        let mut projection_checks = Vec::new();
        if fp8_mode == Fp8Mode::W8A16 {
            for (module, linear) in decoder.sample_projections() {
                let original =
                    checkpoint.fp8(&module, linear.out_features(), linear.in_features())?;
                let error = linear.check_against(&original, device)?;
                // A NaN error fails the check too.
                if error.is_nan() || error > PROJECTION_CHECK_LIMIT {
                    return Err(Error::ModelLoadError(format!(
                        "Nemotron 3.5 Transcribe: resident {module} deviates from the checkpoint by {error:.3e}"
                    )));
                }
                projection_checks.push((module, error));
            }
        }
        let scan_check = if residency == Residency::Cpu {
            None
        } else {
            let error =
                crate::kernels::mamba2::verify_device(device, config.decoder.mamba.state.min(128))?;
            if error.is_nan() || error > SCAN_CHECK_LIMIT {
                return Err(Error::ModelLoadError(format!(
                    "Nemotron 3.5 Transcribe: the device Mamba2 scan deviates from the CPU scan by {error:.3e}"
                )));
            }
            Some(error)
        };
        let ctc = checkpoint
            .ctc_geometry()?
            .map(|geometry| -> Result<CtcStage> {
                let head =
                    NemotronCtcHead::load(&checkpoint, &config, &geometry, device, fp8_mode)?;
                let tokenizer = CtcTokenizer::from_model_proto(head.model_proto())?;
                if tokenizer.len() + 1 != head.classes() {
                    return Err(Error::ModelLoadError(format!(
                        "the CTC tokenizer has {} pieces for a {}-class head",
                        tokenizer.len(),
                        head.classes()
                    )));
                }
                Ok(CtcStage {
                    head,
                    tokenizer,
                    config: CtcAlignmentConfig::default(),
                })
            })
            .transpose()?;
        drop(checkpoint);
        let load_seconds = started.elapsed().as_secs_f64();
        tracing::info!(
            "Loaded Nemotron 3.5 Transcribe from {:?} on {:?} ({}), {:.1} GiB resident, in {:.1} s",
            model_dir,
            residency,
            fp8_mode.name(),
            (decoder.resident_bytes() + perception.resident_bytes()) as f64 / (1u64 << 30) as f64,
            load_seconds
        );
        Ok(Self {
            variant,
            model_dir: model_dir.to_path_buf(),
            device: device.clone(),
            residency,
            fp8_mode,
            frontend,
            perception,
            decoder,
            tokenizer,
            special_tokens,
            ctc,
            audio_token_id,
            tag_ids,
            identity: NEXT_MODEL_IDENTITY.fetch_add(1, Ordering::Relaxed),
            diagnostics: NemotronDiagnostics {
                load_seconds,
                projection_checks,
                scan_check,
                frontend_checked: true,
                ..Default::default()
            },
            config,
        })
    }

    pub fn config(&self) -> &NemotronTranscribeConfig {
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

    pub fn frontend(&self) -> &NemotronMelFrontend {
        &self.frontend
    }

    pub fn perception(&self) -> &NemotronPerception {
        &self.perception
    }

    pub fn audio_token_id(&self) -> u32 {
        self.audio_token_id
    }

    pub fn decoder(&self) -> &NemotronDecoder {
        &self.decoder
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn fp8_mode(&self) -> Fp8Mode {
        self.fp8_mode
    }

    pub fn max_context_tokens(&self) -> usize {
        MAX_CONTEXT_TOKENS
    }

    /// Resident device bytes of the weights.
    pub fn resident_bytes(&self) -> usize {
        self.decoder.resident_bytes()
            + self.perception.resident_bytes()
            + self.ctc.as_ref().map_or(0, |ctc| ctc.head.resident_bytes())
    }

    /// Whether word timestamps can be computed (the checkpoint has the CTC
    /// artifact).
    pub fn supports_word_timestamps(&self) -> bool {
        self.ctc.is_some()
    }

    pub fn diagnostics_summary(&self) -> serde_json::Value {
        let d = &self.diagnostics;
        json!({
            "family": "nemotron_transcribe_asr",
            "device_kind": match self.residency {
                Residency::Cpu => "cpu",
                Residency::Metal => "metal",
                Residency::Cuda => "cuda",
            },
            "dtype": match (self.residency, self.fp8_mode) {
                (Residency::Cpu, _) => "q8_0",
                (_, Fp8Mode::W8A16) => "fp8_w8a16",
                (_, Fp8Mode::VllmFp8) => "fp8_vllm_parity",
            },
            "residency": format!("{:?}", self.residency),
            "fp8_mode": self.fp8_mode.name(),
            "load_seconds": d.load_seconds,
            "resident_bytes": self.resident_bytes(),
            "embedding_host_bytes": self.decoder.embeddings().host_bytes(),
            "frontend_tables_checked": d.frontend_checked,
            "projection_checks": d.projection_checks.iter().map(|(m, e)| json!({"module": m, "relative_l2": e})).collect::<Vec<_>>(),
            "mamba2_scan_check": d.scan_check,
            "moe_path": if self.decoder.moe_kernels().0 { "device_routed" } else { "per_expert" },
            "moe_stacked_check": self.decoder.moe_kernels().1,
            "ctc_timestamp_head": self.ctc.is_some(),
            "prefill_spans": d.prefill_spans.load(Ordering::Relaxed),
            "decode_steps": d.decode_steps.load(Ordering::Relaxed),
            "windows_encoded": d.windows_encoded.load(Ordering::Relaxed),
            "greedy_near_ties": d.near_ties.load(Ordering::Relaxed),
            "loop_stops": d.loop_stops.load(Ordering::Relaxed),
        })
    }

    /// Device workspace one decode row holds during a step: the stacked
    /// Mamba states it replaces (about one row's state) and its activations
    /// and logits.
    pub fn decode_workspace_per_row_bytes(&self) -> u64 {
        let d = self.decoder.config();
        super::decoder::state::recurrent_state_bytes(d)
            + (d.vocab * 4 + d.mamba.in_proj_rows() * 8 + d.hidden * 64) as u64
    }

    /// Device workspace one prefill span of `tokens` holds: its activations
    /// through the widest projection, one perception window, and a copy of
    /// the row's Mamba state.
    pub fn prefill_workspace_bytes(&self, tokens: usize) -> u64 {
        let d = self.decoder.config();
        let widest = d
            .mamba
            .in_proj_rows()
            .max(d.moe.shared_intermediate)
            .max(d.attention.heads * d.attention.head_dim);
        let activations = tokens as u64 * (widest as u64 + 8 * d.hidden as u64) * 4;
        let window = 600 * self.config.asr_encoder.ff_dim as u64 * 4 * 4
            + (self.config.asr_encoder.heads * 600 * 600 * 4) as u64;
        activations + window + super::decoder::state::recurrent_state_bytes(d)
    }

    /// [`Self::prepare`] for audio at any sample rate (resampled to 16 kHz).
    pub fn prepare_audio(
        &self,
        samples: &[f32],
        sample_rate: u32,
        request: NemotronTranscribeRequest,
    ) -> Result<NemotronPreparedArtifact> {
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
        request: NemotronTranscribeRequest,
    ) -> Result<NemotronPreparedArtifact> {
        let sample_rate = self.config.frontend.sample_rate;
        let duration_secs = samples.len() as f32 / sample_rate as f32;
        if duration_secs > MAX_AUDIO_SECONDS + 1.0 {
            return Err(Error::InvalidInput(format!(
                "Nemotron 3.5 Transcribe takes at most {} minutes of audio in one request",
                MAX_AUDIO_SECONDS / 60.0
            )));
        }
        let instruction = render_instruction(&request.prompt)?;
        let mel = self.frontend.compute(samples)?;
        let audio_tokens = self.config.audio_tokens_for_samples(samples.len());
        let layout = tokenize_prompt(
            &self.tokenizer,
            &instruction,
            &self.config.audio_token,
            self.audio_token_id,
            audio_tokens,
        )?;
        let room = MAX_CONTEXT_TOKENS.saturating_sub(layout.len());
        let max_new_tokens = request.max_new_tokens.unwrap_or(MAX_NEW_TOKENS).min(room);
        if max_new_tokens == 0 {
            return Err(Error::ContextLengthExceeded(format!(
                "the {}-token prompt leaves no room in the {MAX_CONTEXT_TOKENS}-token context",
                layout.len()
            )));
        }
        Ok(NemotronPreparedArtifact {
            planned_decode: planned_decode_tokens(duration_secs).min(max_new_tokens),
            request,
            instruction,
            layout,
            max_new_tokens,
            duration_secs,
            samples: samples.len(),
            mel,
        })
    }

    /// A row for a prepared request, holding at most `context` positions.
    pub fn new_row(
        &self,
        artifact: Arc<NemotronPreparedArtifact>,
        context: usize,
    ) -> Result<NemotronRetainedState> {
        let context = context.min(MAX_CONTEXT_TOKENS);
        if artifact.prompt_tokens() >= context {
            return Err(Error::InvalidInput(format!(
                "a {}-token prompt does not fit its {context}-token row",
                artifact.prompt_tokens()
            )));
        }
        let perception = PerceptionStream::new(
            artifact.mel.clone(),
            self.perception.stacking(),
            artifact.request.word_timestamps && self.ctc.is_some(),
        );
        if perception.audio_tokens() != artifact.layout.audio_tokens {
            return Err(Error::InferenceError(
                "the perception plan and the prompt disagree on audio tokens".into(),
            ));
        }
        Ok(NemotronRetainedState {
            mamba: NemotronMambaState::zeros(self.decoder.config(), &self.device)?,
            perception,
            prefill_cursor: 0,
            decode: None,
            context,
            tensor_sequence: None,
            artifact,
        })
    }

    /// Embeddings `[len, hidden]` of prompt positions `start..start + len`.
    fn span_embeddings(
        &self,
        state: &mut NemotronRetainedState,
        start: usize,
        len: usize,
    ) -> Result<Tensor> {
        let layout = &state.artifact.layout;
        let mut parts = Vec::new();
        let mut text = Vec::new();
        let mut audio: Option<(usize, usize)> = None;
        let flush_text = |parts: &mut Vec<Tensor>, text: &mut Vec<u32>| -> Result<()> {
            if !text.is_empty() {
                parts.push(self.decoder.embeddings().gather(text, &self.device)?);
                text.clear();
            }
            Ok(())
        };
        let before = state.perception.windows_encoded();
        for position in start..start + len {
            match layout.slot(position) {
                PromptSlot::Text(id) => {
                    if let Some((first, count)) = audio.take() {
                        parts.push(
                            state
                                .perception
                                .embeddings(&self.perception, first, count)?,
                        );
                    }
                    text.push(id);
                }
                PromptSlot::Audio(index) => {
                    flush_text(&mut parts, &mut text)?;
                    audio = Some(match audio {
                        Some((first, count)) => (first, count + 1),
                        None => (index, 1),
                    });
                }
            }
        }
        if let Some((first, count)) = audio {
            parts.push(
                state
                    .perception
                    .embeddings(&self.perception, first, count)?,
            );
        }
        flush_text(&mut parts, &mut text)?;
        self.diagnostics.windows_encoded.fetch_add(
            (state.perception.windows_encoded() - before) as u64,
            Ordering::Relaxed,
        );
        Ok(Tensor::cat(&parts, 0)?)
    }

    /// Prefills the next `span` prompt tokens of a row; the last span starts
    /// its decode loop.
    pub fn retained_prefill(
        &self,
        state: &mut NemotronRetainedState,
        cache: &mut PhysicalPagedKvCache,
        span: usize,
    ) -> Result<()> {
        self.retained_prefill_traced(state, cache, span, None)
    }

    pub fn retained_prefill_traced(
        &self,
        state: &mut NemotronRetainedState,
        cache: &mut PhysicalPagedKvCache,
        span: usize,
        trace: Option<(&TraceRequest, &mut DecoderTrace)>,
    ) -> Result<()> {
        let start = state.prefill_cursor;
        let prompt = state.prompt_tokens();
        if state.decode.is_some() || span == 0 || start + span > prompt {
            return Err(Error::InvalidInput(format!(
                "Nemotron 3.5 Transcribe prefill span {start}+{span} does not fit its {prompt}-token prompt"
            )));
        }
        let embeds = self.span_embeddings(state, start, span)?;
        let mut trace = trace;
        let hidden = self.decoder.forward_span(
            &embeds,
            &mut state.mamba,
            cache,
            trace
                .as_mut()
                .map(|(request, trace)| (*request, &mut **trace)),
        )?;
        if start + span == prompt {
            if let Some((_, trace)) = trace.as_mut() {
                trace.final_logits =
                    super::speech_transformer::host_values(&self.decoder.logits(&hidden)?)?;
            }
        }
        state.prefill_cursor = start + span;
        self.diagnostics
            .prefill_spans
            .fetch_add(1, Ordering::Relaxed);
        if state.prefill_cursor == prompt {
            let budget = state
                .artifact
                .max_new_tokens
                .min(state.context.saturating_sub(prompt));
            let mut decode = NemotronDecodeLoop {
                generated: Vec::new(),
                finished: None,
                max_new_tokens: budget,
                detokenizer: IncrementalDetokenizer::new(),
                pending_text: String::new(),
                near_ties: 0,
            };
            let (token, margin) = self.decoder.greedy(&hidden)?[0];
            self.accept(&mut decode, token, margin)?;
            state.decode = Some(decode);
        }
        Ok(())
    }

    /// Takes `token` into a row's loop: stop, budget and loop checks, and the
    /// streamed text delta.
    fn accept(&self, decode: &mut NemotronDecodeLoop, token: u32, margin: f32) -> Result<()> {
        if margin < 1e-4 {
            decode.near_ties += 1;
            self.diagnostics.near_ties.fetch_add(1, Ordering::Relaxed);
        }
        if self.config.eos_token_ids.contains(&token) {
            decode.finished = Some(NemotronFinishReason::Stop);
        } else {
            decode.generated.push(token);
            if decode.generated.len() >= decode.max_new_tokens {
                decode.finished = Some(NemotronFinishReason::Length);
            } else if ends_in_loop(&decode.generated) {
                decode.finished = Some(NemotronFinishReason::Repetition);
                self.diagnostics.loop_stops.fetch_add(1, Ordering::Relaxed);
            }
        }
        let tags = &self.tag_ids;
        let tokenizer = &self.tokenizer;
        let decode_fn = |ids: &[u32]| -> Result<String> {
            let ids = ids
                .iter()
                .copied()
                .filter(|id| !tags.contains(id))
                .collect::<Vec<_>>();
            tokenizer
                .decode(&ids, false)
                .map_err(|err| Error::TokenizationError(err.to_string()))
        };
        let delta = if decode.finished.is_some() {
            decode.detokenizer.finish(&decode.generated, decode_fn)?
        } else {
            decode.detokenizer.step(&decode.generated, decode_fn)?
        };
        decode.pending_text.push_str(&delta);
        if decode.max_new_tokens == 0 && decode.finished.is_none() {
            decode.finished = Some(NemotronFinishReason::Length);
        }
        Ok(())
    }

    /// One decode step for every unfinished row, batched. Returns whether
    /// each row has finished.
    pub fn retained_decode_batch(&self, rows: &mut [NemotronStepRow<'_>]) -> Result<Vec<bool>> {
        let mut active = Vec::new();
        for (index, row) in rows.iter().enumerate() {
            let decode = row.state.decode.as_ref().ok_or_else(|| {
                Error::InvalidInput("Nemotron 3.5 Transcribe row decodes before its prefill".into())
            })?;
            if !decode.finished() {
                active.push(index);
            }
        }
        if !active.is_empty() {
            let mut step_rows = Vec::with_capacity(active.len());
            for (index, row) in rows.iter_mut().enumerate() {
                if !active.contains(&index) {
                    continue;
                }
                let NemotronStepRow { state, cache } = row;
                let token = *state
                    .decode
                    .as_ref()
                    .and_then(|d| d.generated.last())
                    .ok_or_else(|| {
                        Error::InferenceError("an unfinished row has no token to feed".into())
                    })?;
                step_rows.push(DecodeRow {
                    state: &mut state.mamba,
                    cache,
                    token,
                });
            }
            let hidden = self.decoder.forward_step(&mut step_rows)?;
            drop(step_rows);
            let next = self.decoder.greedy(&hidden)?;
            for (&index, (token, margin)) in active.iter().zip(next) {
                let decode = rows[index].state.decode.as_mut().expect("checked above");
                self.accept(decode, token, margin)?;
            }
            self.diagnostics
                .decode_steps
                .fetch_add(1, Ordering::Relaxed);
        }
        Ok(rows.iter().map(|row| row.state.finished()).collect())
    }

    /// The row's transcription once finished.
    pub fn finish_row(&self, state: &NemotronRetainedState) -> Result<NemotronTranscription> {
        let decode = state
            .decode
            .as_ref()
            .filter(|d| d.finished())
            .ok_or_else(|| Error::InferenceError("the row has not finished".into()))?;
        let decoded = self
            .tokenizer
            .decode(&decode.generated, false)
            .map_err(|err| Error::TokenizationError(err.to_string()))?;
        let special_tokens = self
            .special_tokens
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let transcript = parse_transcript(&clean_sot_transcript(&decoded, &special_tokens));
        let timestamps = state
            .artifact
            .request
            .word_timestamps
            .then(|| self.word_timestamps(state, &transcript.raw));
        Ok(NemotronTranscription {
            transcript,
            finish_reason: decode.finished.expect("finished"),
            generated_tokens: decode.generated.len(),
            prompt_tokens: state.prompt_tokens(),
            audio_tokens: state.artifact.layout.audio_tokens,
            windows: state.perception.windows_encoded(),
            timestamps,
        })
    }

    /// The CTC stage over a finished row: the head over the stored encoder
    /// output (only the transcript's columns leave the device), then NeMo's
    /// aligner with the diarizer's probabilities. A failure is reported in
    /// the result's `error`, as upstream does, and never fails the
    /// transcript.
    fn word_timestamps(&self, state: &NemotronRetainedState, transcript: &str) -> CtcTimestamps {
        let Some(ctc) = &self.ctc else {
            return CtcTimestamps::failed(CtcTimestampError::NotEnabled);
        };
        let windows = state.perception.retained_windows();
        if windows.is_empty() {
            return CtcTimestamps::failed(CtcTimestampError::CaptureUnavailable);
        }
        let run = || -> Result<CtcTimestamps> {
            let flatten = |rows: &mut dyn Iterator<Item = &Vec<f32>>| {
                let mut values = Vec::new();
                let mut frames = 0;
                let mut speakers = 0;
                for row in rows {
                    speakers = row.len();
                    values.extend_from_slice(row);
                    frames += 1;
                }
                (values, frames, speakers)
            };
            let (speaker_values, speaker_frames, speakers) =
                flatten(&mut windows.iter().flat_map(|w| w.speaker_preds.iter()));
            // The native predictions past the recording's valid mel frames
            // are padding.
            let valid = state.perception.valid_frames();
            let (diar_values, diar_frames, diar_speakers) = flatten(
                &mut windows
                    .iter()
                    .flat_map(|w| w.speaker_preds_10ms.iter())
                    .take(valid),
            );
            let diarization = (diar_frames > 0).then_some(SpeakerActivity {
                values: &diar_values,
                frames: diar_frames,
                speakers: diar_speakers,
            });
            let audio_duration = f64::from(state.artifact.duration_secs);
            let inputs = windows
                .iter()
                .map(|window| window.ctc_input.clone())
                .collect::<Vec<_>>();
            let states = Tensor::cat(&inputs, 0)?;
            let ctc_frames = states.dim(0)? * ctc.head.factor();
            // No language: NeMo serves with per-transcript script detection.
            let prepared = match prepare_alignment(
                &ctc.tokenizer,
                transcript,
                None,
                ctc_frames,
                ctc.head.blank(),
                &ctc.config,
            ) {
                Ok(prepared) => prepared,
                // Upstream keeps the diarization when the text cannot align.
                Err(error) => {
                    return Ok(CtcTimestamps::without_words(
                        Some(error),
                        diarization.map_or_else(Vec::new, |activity| {
                            diarization_segments(activity, audio_duration, &ctc.config)
                        }),
                    ))
                }
            };
            let log_probs = ctc.head.log_prob_columns(&states, &prepared.columns)?;
            Ok(align(
                &prepared,
                CtcAlignmentInputs {
                    log_probs: &log_probs,
                    speaker_probs: (speaker_frames > 0).then_some(SpeakerActivity {
                        values: &speaker_values,
                        frames: speaker_frames,
                        speakers,
                    }),
                    diarization,
                    audio_duration,
                },
                &ctc.config,
            ))
        };
        run().unwrap_or_else(|error| {
            tracing::warn!("Nemotron 3.5 Transcribe word timestamps failed: {error}");
            CtcTimestamps::failed(CtcTimestampError::AlignmentFailed)
        })
    }

    /// One request end to end on a caller-owned cache, in prefill spans of
    /// `span` tokens (tests, the CLI reference path and parity runs).
    pub fn transcribe_with_cache(
        &self,
        samples: &[f32],
        request: NemotronTranscribeRequest,
        cache: &mut PhysicalPagedKvCache,
        span: usize,
    ) -> Result<NemotronTranscription> {
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
            self.retained_decode_batch(&mut [NemotronStepRow {
                state: &mut state,
                cache,
            }])?;
        }
        self.finish_row(&state)
    }
}
