//! LFM3-Audio-2B: an LFM3 language model that reads speech through the
//! causal encoder and writes speech as 12-codebook frames through the
//! MaskGIT depth decoder and the LiquidCodec decoder.
//!
//! Generation follows `Lfm3AudioForConditionalGeneration._generate`:
//! * text mode samples tokens; `<|audio_start|>` switches to speech and
//!   `<|mixed_start|>` to interleaved blocks;
//! * speech mode samples a frame per step until a frame holds the
//!   end-of-audio code, which returns to text;
//! * interleaved mode alternates 6 text tokens and 12 frames until
//!   `<|text_end|>` closes the text, then speaks until end-of-audio;
//! * `<|im_end|>` ends the turn from any text slot, and `<|text_end|>` does
//!   outside an interleaved block.
//!
//! Every slot, a text token or a frame (fed back as the sum of its codes'
//! embeddings), is one language-model step. Speech streams through the codec
//! as frames arrive.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_core::quantized::QTensor;
use candle_core::{DType, Device, IndexOp, Tensor};
use tracing::info;

use crate::backends::{BackendKind, DeviceProfile};
use crate::engine::InvocationTensorLease;
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::architectures::lfm2::backbone::{
    Lfm2ShortConvRuntimeState, QuantizedLfm2Backbone,
};
use crate::models::architectures::lfm2::config::Lfm2BackboneConfig;
use crate::models::architectures::lfm3::parse_lfm3_backbone_config;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::noise::SplitMix64;
use crate::models::shared::weights::gguf::GgufLoader;
use crate::tokenizer::{IncrementalDecoder, Tokenizer};

use super::codec::{LiquidCodecDecoder, LiquidCodecStream};
use super::config::*;
use super::depth::{FrameHistory, Lfm3DepthDecoder};
use super::encoder::Lfm3AudioEncoder;
use super::preprocessor::Lfm3AudioMel;
use super::prompts::{
    asr_prompt, audio_slots, live_asr_prompt, reply_prompt, tts_prompt, Lfm3AudioReplyOutput,
    Lfm3AudioVoice, PromptPart,
};
use super::sampling::{apply_repetition_penalty, sample};
use crate::models::shared::chat::ChatMessage;
use izwi_asr_toolkit::{plan_audio_chunks, AsrLongFormConfig, TranscriptAssembler};

/// The served context, the length on Liquid's model card (the GGUF says
/// 128,000).
pub(crate) const LFM3_AUDIO_CONTEXT_TOKENS: usize = 32_768;
const MMPROJ: &str = "mmproj-LFM3-Audio-2B-Q8_0.gguf";
/// Frames decoded per codec call while speech streams: the first call
/// emits after two frames (160 ms), later ones every six (480 ms).
const FIRST_STREAM_FRAMES: usize = 2;
const STREAM_FRAMES: usize = 6;

/// The GGUF files of an LFM3-Audio variant: the language model and the
/// mmproj holding the encoder, depth decoder and codec decoder.
pub(crate) fn lfm3_audio_gguf_files(variant: ModelVariant) -> Result<[&'static str; 2]> {
    match variant {
        ModelVariant::Lfm3Audio2BGguf => Ok(["LFM3-Audio-2B-Q4_K_M.gguf", MMPROJ]),
        ModelVariant::Lfm3Audio2BQ8Gguf => Ok(["LFM3-Audio-2B-Q8_0.gguf", MMPROJ]),
        other => Err(Error::ModelLoadError(format!(
            "Unsupported LFM3-Audio variant: {other}"
        ))),
    }
}

/// What generation starts as: the mode the prompt's last token opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lfm3AudioMode {
    Text,
    Audio,
    Interleaved,
}

/// One generation's inputs.
pub(crate) struct Lfm3AudioRequest {
    pub(crate) prompt: Vec<PromptPart>,
    /// Encoder embeddings `[1, tokens, 2048]` of the prompt's speech clips,
    /// one per [`PromptPart::Audio`], in order.
    pub(crate) clips: Vec<Tensor>,
    pub(crate) mode: Lfm3AudioMode,
    /// Keep the reply in text: forbid `<|audio_start|>` and
    /// `<|mixed_start|>`.
    pub(crate) text_only: bool,
    /// Slots (text tokens and frames) to generate at most.
    pub(crate) max_slots: usize,
    pub(crate) text_sampling: Lfm3TextSampling,
    pub(crate) audio_sampling: Lfm3AudioSampling,
    pub(crate) seed: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Lfm3AudioOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub text_tokens: usize,
    pub audio_frames: usize,
    pub samples: Vec<f32>,
    /// The speech frames, in order (end-of-audio frames excluded).
    pub(crate) frames: Vec<[u32; CODEBOOKS]>,
    /// Why generation stopped: `eos`, `length` or `cancelled`.
    pub stop_reason: &'static str,
}

/// One request's generation outside the language model: its mode, sampling
/// state, histories and codec stream, and the hidden state its next slot
/// reads. [`Lfm3AudioModel::generate`] drives one row; a batched step drives
/// many through [`Lfm3AudioModel::choose_slots`], each row drawing from its
/// own generator, so a row samples the same alone or beside others.
#[derive(Debug, Clone)]
pub(crate) struct Lfm3AudioRow {
    text_sampling: Lfm3TextSampling,
    audio_sampling: Lfm3AudioSampling,
    max_slots: usize,
    /// The next slot to fill.
    slot: usize,
    /// The final-norm hidden state `[1, 2048]` the next slot reads.
    last: Tensor,
    rng: SplitMix64,
    decoder: IncrementalDecoder,
    text_history: std::collections::VecDeque<u32>,
    audio_history: FrameHistory,
    codec: LiquidCodecStream,
    pending_frames: Vec<[u32; CODEBOOKS]>,
    mode: Lfm3AudioMode,
    slots_left: usize,
    speaking: bool,
    text_done: bool,
    suppress: Vec<u32>,
    output: Lfm3AudioOutput,
    done: bool,
    /// Text and 24 kHz samples decoded since the caller last took them.
    new_text: String,
    new_samples: Vec<f32>,
}

impl Lfm3AudioRow {
    /// Whether the row has stopped: its output is complete.
    pub(crate) fn finished(&self) -> bool {
        self.done
    }

    pub(crate) fn output(&self) -> &Lfm3AudioOutput {
        &self.output
    }

    pub(crate) fn into_output(self) -> Lfm3AudioOutput {
        self.output
    }

    /// Text decoded since the last call.
    pub(crate) fn take_text(&mut self) -> String {
        std::mem::take(&mut self.new_text)
    }

    /// Samples decoded since the last call.
    pub(crate) fn take_samples(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.new_samples)
    }

    /// The language model's hidden state `[1, 2048]` after the embedding
    /// [`Lfm3AudioModel::choose_slots`] returned.
    pub(crate) fn accept_hidden(&mut self, hidden: Tensor) {
        self.last = hidden;
    }

    fn speaks_next(&self) -> bool {
        self.mode == Lfm3AudioMode::Audio
            || (self.mode == Lfm3AudioMode::Interleaved && self.speaking)
    }

    fn push_text(&mut self, delta: &str) {
        self.output.text.push_str(delta);
        self.new_text.push_str(delta);
    }
}

/// The language model's sequence state: a forward over new embeddings at
/// the sequence's current length returns their final-norm hidden states.
pub(crate) trait Lfm3AudioLmState {
    /// Empty the state before a generation; it may hold an earlier one's.
    fn reset(&mut self) -> Result<()>;

    /// Positions the state can hold: the context this request leased.
    fn capacity_tokens(&self) -> usize;

    fn forward(
        &mut self,
        lm: &QuantizedLfm2Backbone,
        embeds: &Tensor,
        position: usize,
    ) -> Result<Tensor>;
}

/// The engine's invocation workspace: a paged KV lease and a ShortConv ring
/// lease.
pub(crate) struct InvocationLmState<'a> {
    pub(crate) cache: &'a mut PhysicalPagedKvCache,
    pub(crate) shortconv: &'a mut InvocationTensorLease,
}

impl Lfm3AudioLmState for InvocationLmState<'_> {
    fn reset(&mut self) -> Result<()> {
        if self.cache.context_len() > 0 {
            self.cache.reset_invocation()?;
            self.shortconv.reset_invocation()?;
        }
        Ok(())
    }

    fn capacity_tokens(&self) -> usize {
        self.cache.capacity_tokens()
    }

    fn forward(
        &mut self,
        lm: &QuantizedLfm2Backbone,
        embeds: &Tensor,
        position: usize,
    ) -> Result<Tensor> {
        lm.forward_embeds_physical(embeds, position, self.cache, self.shortconv)
    }
}

/// A retained paged cache and ShortConv state (tests and benchmarks).
pub(crate) struct RetainedLmState {
    pub(crate) cache: PhysicalPagedKvCache,
    pub(crate) shortconv: Lfm2ShortConvRuntimeState,
}

impl Lfm3AudioLmState for RetainedLmState {
    fn reset(&mut self) -> Result<()> {
        if self.cache.context_len() > 0 {
            return Err(Error::InvalidInput(
                "a retained LFM3-Audio test state serves one generation".into(),
            ));
        }
        Ok(())
    }

    fn capacity_tokens(&self) -> usize {
        self.cache.capacity_tokens()
    }

    fn forward(
        &mut self,
        lm: &QuantizedLfm2Backbone,
        embeds: &Tensor,
        position: usize,
    ) -> Result<Tensor> {
        lm.forward_embeds_retained(embeds, position, &mut self.cache, &mut self.shortconv)
    }
}

/// A transcription of one clip's encoder embeddings `audio` (at most 30 s),
/// greedy, in `language` or the one the model hears, at most `max_tokens`
/// text tokens.
pub(crate) fn asr_request(
    audio: Tensor,
    language: Option<&str>,
    max_tokens: usize,
) -> Result<Lfm3AudioRequest> {
    Ok(Lfm3AudioRequest {
        prompt: asr_prompt(language)?,
        clips: vec![audio],
        mode: Lfm3AudioMode::Text,
        text_only: true,
        max_slots: max_tokens.max(1),
        text_sampling: Lfm3TextSampling::default(),
        audio_sampling: Lfm3AudioSampling::default(),
        seed: 0,
    })
}

/// A speech request: `text` in `voice`, at most `max_frames` frames.
pub(crate) fn tts_request(
    text: &str,
    voice: Lfm3AudioVoice,
    max_frames: usize,
    sampling: Lfm3AudioSampling,
    seed: u64,
) -> Result<Lfm3AudioRequest> {
    if text.trim().is_empty() {
        return Err(Error::InvalidInput("LFM3-Audio TTS needs text".into()));
    }
    Ok(Lfm3AudioRequest {
        prompt: tts_prompt(text, voice),
        clips: Vec::new(),
        mode: Lfm3AudioMode::Audio,
        text_only: false,
        // The end-of-audio frame and the closing `<|im_end|>`.
        max_slots: max_frames.max(1) + 2,
        text_sampling: Lfm3TextSampling::default(),
        audio_sampling: sampling,
        seed,
    })
}

/// A reply to `messages` (see [`reply_prompt`]): text, or interleaved text
/// and speech, the encoder embeddings `clips` filling the prompt's speech
/// slots in order, at most `max_slots` slots. A reply that may call tools
/// starts in text and moves to speech at `<|mixed_start|>`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn reply_request(
    messages: &[ChatMessage],
    clips: Vec<Tensor>,
    tools: &[serde_json::Value],
    output: Lfm3AudioReplyOutput,
    preserve_thinking: bool,
    max_slots: usize,
    text_sampling: Lfm3TextSampling,
    audio_sampling: Lfm3AudioSampling,
    seed: u64,
) -> Result<Lfm3AudioRequest> {
    let prompt = reply_prompt(messages, tools, output, preserve_thinking);
    let slots = audio_slots(&prompt);
    if slots != clips.len() {
        return Err(Error::InvalidInput(format!(
            "LFM3-Audio reply has {slots} audio placeholders for {} audio clips",
            clips.len()
        )));
    }
    let (mode, text_only) = match output {
        Lfm3AudioReplyOutput::Text => (Lfm3AudioMode::Text, true),
        Lfm3AudioReplyOutput::Interleaved if !tools.is_empty() => (Lfm3AudioMode::Text, false),
        Lfm3AudioReplyOutput::Interleaved => (Lfm3AudioMode::Interleaved, false),
    };
    Ok(Lfm3AudioRequest {
        prompt,
        clips,
        mode,
        text_only,
        max_slots: max_slots.max(1),
        text_sampling,
        audio_sampling,
        seed,
    })
}

fn check_prompt_fits(prompt_tokens: usize, context: usize) -> Result<()> {
    if prompt_tokens >= context {
        return Err(Error::InvalidInput(format!(
            "LFM3-Audio prompt of {prompt_tokens} tokens does not fit this request's {context}-token context"
        )));
    }
    Ok(())
}

pub struct Lfm3AudioModel {
    device: DeviceProfile,
    tokenizer: Tokenizer,
    lm_config: Lfm2BackboneConfig,
    lm: QuantizedLfm2Backbone,
    encoder: Lfm3AudioEncoder,
    depth: Lfm3DepthDecoder,
    /// `[12 x 2017, 2048]`: a frame's feedback embedding sums one row per
    /// codebook.
    feedback: Arc<QTensor>,
    codec: LiquidCodecDecoder,
    mel: Lfm3AudioMel,
    /// Additive mask of the ids text slots never take: the speech
    /// placeholders, pad and BOS, and ids past the tokenizer.
    text_mask: Vec<f32>,
}

impl Lfm3AudioModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        let [lm_file, mmproj_file] = lfm3_audio_gguf_files(variant)?;
        let path = |file: &str| -> Result<PathBuf> {
            let path = model_dir.join(file);
            if path.exists() {
                Ok(path)
            } else {
                Err(Error::ModelLoadError(format!(
                    "LFM3-Audio file not found: {}",
                    path.display()
                )))
            }
        };
        let backend = BackendKind::from(device.kind);
        let lm_loader = GgufLoader::from_path_with_backend(&path(lm_file)?, backend)?;
        let mut lm_config = parse_lfm3_backbone_config(&lm_loader)?;
        lm_config.context_length = lm_config.context_length.min(LFM3_AUDIO_CONTEXT_TOKENS);
        let mut lm = QuantizedLfm2Backbone::load(&lm_loader, lm_config.clone(), &device.device)?;
        // Rows decode alone or batched; sampled speech must not depend on
        // which.
        lm.make_decode_rows_invariant();
        let mmproj = GgufLoader::from_path_with_backend(&path(mmproj_file)?, backend)?;
        let encoder = Lfm3AudioEncoder::load(&mmproj, &device.device)?;
        let depth = Lfm3DepthDecoder::load(&mmproj, &device.device)?;
        let feedback = Arc::new(mmproj.load_qtensor("a.gen.code.out_embd.weight", &device.device)?);
        if feedback.shape().dims() != [CODEBOOKS * CODEBOOK_SIZE, lm.hidden_size()] {
            return Err(Error::ModelLoadError(format!(
                "LFM3-Audio feedback table is {:?}",
                feedback.shape().dims()
            )));
        }
        let codec = LiquidCodecDecoder::load(&mmproj, &device.device)?;
        let tokenizer = Tokenizer::from_path(model_dir)?;
        let vocab = tokenizer.vocab_size();
        let mut text_mask = vec![0f32; lm.vocab_size()];
        for (id, value) in text_mask.iter_mut().enumerate() {
            let id = id as u32;
            if id as usize >= vocab
                || matches!(
                    id,
                    AUDIO_PLACEHOLDER | AUDIO_OUT_PLACEHOLDER | BOS | IM_START
                )
                || id == 124_893
            {
                *value = f32::NEG_INFINITY;
            }
        }
        info!(
            "Loaded {variant} on {:?} ({} context tokens)",
            device.kind, lm_config.context_length
        );
        Ok(Self {
            device,
            tokenizer,
            lm_config,
            lm,
            encoder,
            depth,
            feedback,
            codec,
            mel: Lfm3AudioMel::new(),
            text_mask,
        })
    }

    pub fn device(&self) -> &DeviceProfile {
        &self.device
    }

    pub(crate) fn lm_config(&self) -> &Lfm2BackboneConfig {
        &self.lm_config
    }

    /// Positions one row of `workload` may hold: that workload's context.
    pub(crate) fn row_context(&self, workload: super::physical::Lfm3AudioWorkload) -> usize {
        workload.context_tokens().min(self.max_context_tokens())
    }

    pub(crate) fn lm(&self) -> &QuantizedLfm2Backbone {
        &self.lm
    }

    pub(crate) fn new_lm_shortconv_state(&self) -> Lfm2ShortConvRuntimeState {
        self.lm.new_shortconv_state()
    }

    /// A request's prompt embeddings, computed once before its row runs.
    pub(crate) fn prepare_row(
        &self,
        request: Lfm3AudioRequest,
    ) -> Result<std::sync::Arc<super::retained::Lfm3AudioPreparedArtifact>> {
        let (embeds, prompt_tokens) = self.prompt_embeds(&request)?;
        Ok(std::sync::Arc::new(
            super::retained::Lfm3AudioPreparedArtifact {
                request,
                embeds,
                prompt_tokens,
            },
        ))
    }

    /// A transcription row of one clip (at most 30 s) at any rate, in
    /// `language` or the one the model hears: its prompt with the clip's
    /// encoder embeddings, decoding at most `max_tokens` text tokens (from
    /// the clip's length when `None`).
    pub(crate) fn prepare_asr_row(
        &self,
        samples: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        max_tokens: Option<usize>,
    ) -> Result<std::sync::Arc<super::retained::Lfm3AudioPreparedArtifact>> {
        let seconds = samples.len() as f32 / sample_rate.max(1) as f32;
        let budget = max_tokens.unwrap_or_else(|| lfm3_audio_asr_max_tokens(seconds));
        let audio = self.encode_audio(samples, sample_rate)?;
        self.prepare_row(asr_request(audio, language, budget)?)
    }

    /// The reply row of a conversation (see [`reply_request`]): its prompt
    /// with each audio clip's encoder embeddings, the clips at any rate.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare_reply_row(
        &self,
        clips: &[(Vec<f32>, u32)],
        messages: &[ChatMessage],
        tools: &[serde_json::Value],
        output: Lfm3AudioReplyOutput,
        preserve_thinking: bool,
        max_slots: usize,
        text_sampling: Lfm3TextSampling,
        audio_sampling: Lfm3AudioSampling,
        seed: u64,
    ) -> Result<std::sync::Arc<super::retained::Lfm3AudioPreparedArtifact>> {
        let clips = clips
            .iter()
            .map(|(samples, sample_rate)| self.encode_audio(samples, *sample_rate))
            .collect::<Result<Vec<_>>>()?;
        self.prepare_row(reply_request(
            messages,
            clips,
            tools,
            output,
            preserve_thinking,
            max_slots,
            text_sampling,
            audio_sampling,
            seed,
        )?)
    }

    pub fn max_context_tokens(&self) -> usize {
        self.lm_config.context_length
    }

    /// Encoder embeddings `[1, tokens, 2048]` of a mono clip at any rate.
    pub(crate) fn encode_audio(&self, samples: &[f32], sample_rate: u32) -> Result<Tensor> {
        if samples.is_empty() {
            return Err(Error::InvalidInput("LFM3-Audio needs audio".into()));
        }
        let samples = if sample_rate == INPUT_SAMPLE_RATE {
            samples.to_vec()
        } else {
            crate::audio::resample_mono_high_quality(samples, sample_rate, INPUT_SAMPLE_RATE)?
        };
        let (features, frames) = self.mel.features(&samples);
        self.encoder.encode(&features, frames)
    }

    /// Prompt embeddings `[1, tokens, 2048]` and their token count.
    fn prompt_embeds(&self, request: &Lfm3AudioRequest) -> Result<(Tensor, usize)> {
        let device = &self.device.device;
        let mut pieces = Vec::new();
        let mut clips = request.clips.iter();
        for part in &request.prompt {
            match part {
                PromptPart::Text(text) => {
                    let ids = self.tokenizer.encode(text)?;
                    if !ids.is_empty() {
                        let input = Tensor::from_vec(ids.clone(), (1, ids.len()), device)?;
                        pieces.push(self.lm.embed_tokens(&input)?);
                    }
                }
                PromptPart::Audio => {
                    let clip = clips.next().ok_or_else(|| {
                        Error::InvalidInput(
                            "LFM3-Audio prompt has a speech slot but no audio".into(),
                        )
                    })?;
                    pieces.push(clip.to_dtype(DType::F32)?);
                }
            }
        }
        if clips.next().is_some() {
            return Err(Error::InvalidInput(
                "LFM3-Audio prompt has more audio clips than speech slots".into(),
            ));
        }
        let embeds = Tensor::cat(&pieces, 1)?;
        let tokens = embeds.dim(1)?;
        Ok((embeds, tokens))
    }

    fn feedback_embed(&self, frame: &[u32; CODEBOOKS]) -> Result<Tensor> {
        let ids = frame
            .iter()
            .enumerate()
            .map(|(codebook, code)| codebook as u32 * CODEBOOK_SIZE as u32 + code)
            .collect::<Vec<_>>();
        let ids = Tensor::from_vec(ids, CODEBOOKS, &self.device.device)?;
        Ok(self
            .feedback
            .embedding(&ids)?
            .sum_keepdim(0)?
            .unsqueeze(0)?)
    }

    /// Run one request. `on_text` receives text deltas, `on_audio` 24 kHz
    /// samples as they decode; `should_stop` cancels between slots.
    pub(crate) fn generate(
        &self,
        request: Lfm3AudioRequest,
        state: &mut dyn Lfm3AudioLmState,
        should_stop: &mut dyn FnMut() -> bool,
        on_text: &mut dyn FnMut(&str),
        on_audio: &mut dyn FnMut(&[f32]),
    ) -> Result<Lfm3AudioOutput> {
        let (embeds, prompt_tokens) = self.prompt_embeds(&request)?;
        state.reset()?;
        // The request leases a context sized for its workload: a prompt
        // that fills it is refused, and the slots stop where it ends.
        let context = state.capacity_tokens().min(self.lm_config.context_length);
        check_prompt_fits(prompt_tokens, context)?;
        let hidden = state.forward(&self.lm, &embeds, 0)?;
        let mut row = self.begin_row(&request, prompt_tokens, context, &hidden)?;
        loop {
            if should_stop() {
                self.stop_row(&mut row, "cancelled")?;
            }
            let next = self.choose_slots(&mut [&mut row])?.pop().flatten();
            let text = row.take_text();
            if !text.is_empty() {
                on_text(&text);
            }
            let samples = row.take_samples();
            if !samples.is_empty() {
                on_audio(&samples);
            }
            let Some((embed, position)) = next else {
                break;
            };
            let hidden = state.forward(&self.lm, &embed, position)?;
            row.accept_hidden(hidden.squeeze(1)?);
        }
        Ok(row.output)
    }

    /// A row after its prompt's prefill: `hidden` holds the language-model
    /// hidden states `[1, span, 2048]` of the prompt's last prefill span, and
    /// the row's slots stop where its `context` ends.
    pub(crate) fn begin_row(
        &self,
        request: &Lfm3AudioRequest,
        prompt_tokens: usize,
        context: usize,
        hidden: &Tensor,
    ) -> Result<Lfm3AudioRow> {
        check_prompt_fits(prompt_tokens, context)?;
        let mut suppress = vec![AUDIO_PLACEHOLDER, AUDIO_OUT_PLACEHOLDER];
        if request.text_only {
            suppress.extend([AUDIO_START, MIXED_START]);
        }
        Ok(Lfm3AudioRow {
            text_sampling: request.text_sampling,
            audio_sampling: request.audio_sampling,
            max_slots: request.max_slots.min(context - prompt_tokens),
            slot: 0,
            last: hidden.i((.., hidden.dim(1)? - 1, ..))?,
            rng: SplitMix64::new(request.seed),
            decoder: IncrementalDecoder::new(true),
            text_history: std::collections::VecDeque::new(),
            audio_history: FrameHistory::default(),
            codec: LiquidCodecStream::default(),
            pending_frames: Vec::new(),
            mode: request.mode,
            slots_left: INTERLEAVED_TEXT,
            speaking: false,
            text_done: false,
            suppress,
            output: Lfm3AudioOutput {
                prompt_tokens,
                stop_reason: "length",
                ..Lfm3AudioOutput::default()
            },
            done: false,
            new_text: String::new(),
            new_samples: Vec::new(),
        })
    }

    /// Fill each live row's next slot, a speech frame or a text token: the
    /// speaking rows draw their frames in one depth batch, each from its own
    /// sampling and generator. Returns, per row, the embedding to feed back
    /// and its position, or `None` once the row has finished (its text and
    /// audio are then complete).
    pub(crate) fn choose_slots(
        &self,
        rows: &mut [&mut Lfm3AudioRow],
    ) -> Result<Vec<Option<(Tensor, usize)>>> {
        let speaking = rows
            .iter()
            .enumerate()
            .filter(|(_, row)| !row.done && row.speaks_next())
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let mut frames = vec![None; rows.len()];
        if !speaking.is_empty() {
            let hidden = Tensor::cat(
                &speaking
                    .iter()
                    .map(|&index| &rows[index].last)
                    .collect::<Vec<_>>(),
                0,
            )?;
            let (samplings, (histories, mut rngs)): (Vec<_>, (Vec<_>, Vec<_>)) = rows
                .iter_mut()
                .enumerate()
                .filter(|(index, _)| speaking.contains(index))
                .map(|(_, row)| {
                    let Lfm3AudioRow {
                        audio_sampling,
                        audio_history,
                        rng,
                        ..
                    } = &mut **row;
                    (&*audio_sampling, (&*audio_history, rng))
                })
                .unzip();
            let drawn = self
                .depth
                .sample_frames(&hidden, &samplings, &histories, &mut rngs)?;
            for (&index, frame) in speaking.iter().zip(drawn) {
                frames[index] = Some(frame);
            }
        }
        rows.iter_mut()
            .zip(frames)
            .map(|(row, frame)| {
                if row.done {
                    return Ok(None);
                }
                self.fill_slot(row, frame)
            })
            .collect()
    }

    /// One slot of one row: the generate loop's body.
    fn fill_slot(
        &self,
        row: &mut Lfm3AudioRow,
        frame: Option<[u32; CODEBOOKS]>,
    ) -> Result<Option<(Tensor, usize)>> {
        if row.slot >= row.max_slots {
            self.stop_row(row, "length")?;
            return Ok(None);
        }
        let interleaving = row.mode == Lfm3AudioMode::Interleaved;
        let mut finished = false;
        let next = if let Some(frame) = frame {
            if frame[0] == AUDIO_EOS_CODE {
                row.mode = Lfm3AudioMode::Text;
                self.flush_row(row, true)?;
                row.codec = LiquidCodecStream::default();
            } else {
                row.audio_history
                    .push(frame, row.audio_sampling.repetition_window);
                row.output.audio_frames += 1;
                row.output.frames.push(frame);
                row.pending_frames.push(frame);
                let due = if row.output.audio_frames <= FIRST_STREAM_FRAMES {
                    FIRST_STREAM_FRAMES
                } else {
                    STREAM_FRAMES
                };
                if row.pending_frames.len() >= due {
                    self.flush_row(row, false)?;
                }
            }
            self.feedback_embed(&frame)?
        } else {
            let mut scores = self
                .lm
                .project_hidden(&row.last)?
                .squeeze(0)?
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec1::<f32>()?;
            for (score, mask) in scores.iter_mut().zip(&self.text_mask) {
                *score += mask;
            }
            apply_repetition_penalty(
                &mut scores,
                row.text_history.iter().copied(),
                row.text_sampling.repetition_penalty,
            );
            for id in &row.suppress {
                scores[*id as usize] = f32::NEG_INFINITY;
            }
            let token = sample(
                &mut scores,
                row.text_sampling.temperature,
                row.text_sampling.top_k,
                row.text_sampling.top_p,
                &mut row.rng,
            );
            if row.text_sampling.repetition_window > 0 {
                if row.text_history.len() == row.text_sampling.repetition_window {
                    row.text_history.pop_front();
                }
                row.text_history.push_back(token);
            }
            row.output.text_tokens += 1;
            let delta = self
                .tokenizer
                .decode_incrementally(&mut row.decoder, token)?;
            row.push_text(&delta);
            if token == TEXT_END {
                if interleaving {
                    row.text_done = true;
                    row.speaking = true;
                } else {
                    finished = true;
                }
            } else if token == IM_END || token == END_OF_TEXT {
                finished = true;
            } else if interleaving {
                // Inside a block the mode markers are ordinary text.
            } else if token == AUDIO_START {
                row.mode = Lfm3AudioMode::Audio;
            } else if token == MIXED_START {
                row.mode = Lfm3AudioMode::Interleaved;
                row.slots_left = INTERLEAVED_TEXT;
                row.speaking = false;
                row.text_done = false;
            }
            let input = Tensor::from_vec(vec![token], (1, 1), &self.device.device)?;
            self.lm.embed_tokens(&input)?
        };
        if interleaving && !row.text_done {
            row.slots_left -= 1;
            if row.slots_left == 0 {
                row.speaking = !row.speaking;
                row.slots_left = if row.speaking {
                    INTERLEAVED_AUDIO
                } else {
                    INTERLEAVED_TEXT
                };
            }
        }
        let position = row.output.prompt_tokens + row.slot;
        row.slot += 1;
        if finished {
            self.stop_row(row, "eos")?;
            return Ok(None);
        }
        if row.slot == row.max_slots {
            self.stop_row(row, "length")?;
            return Ok(None);
        }
        Ok(Some((next, position)))
    }

    /// End a row: the tail of its text and its last speech.
    pub(crate) fn stop_row(&self, row: &mut Lfm3AudioRow, reason: &'static str) -> Result<()> {
        if row.done {
            return Ok(());
        }
        row.done = true;
        row.output.stop_reason = reason;
        let tail = self.tokenizer.finish_incremental_decode(&mut row.decoder)?;
        row.push_text(&tail);
        if row.codec.has_rows() || !row.pending_frames.is_empty() {
            self.flush_row(row, true)?;
        }
        Ok(())
    }

    /// Decode the row's pending frames (all of them with `finalize`).
    fn flush_row(&self, row: &mut Lfm3AudioRow, finalize: bool) -> Result<()> {
        if row.pending_frames.is_empty() && !finalize {
            return Ok(());
        }
        let samples = self
            .codec
            .decode_stream(&mut row.codec, &row.pending_frames, finalize)?;
        row.pending_frames.clear();
        row.output.samples.extend_from_slice(&samples);
        row.new_samples.extend_from_slice(&samples);
        Ok(())
    }

    /// Transcribe one clip of at most 30 s (the training clips' length),
    /// greedily, in `language` or the one the model hears.
    pub(crate) fn transcribe_clip(
        &self,
        samples: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        max_tokens: usize,
        state: &mut dyn Lfm3AudioLmState,
        should_stop: &mut dyn FnMut() -> bool,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Lfm3AudioOutput> {
        let audio = self.encode_audio(samples, sample_rate)?;
        self.transcribe_encoded(audio, language, max_tokens, state, should_stop, on_text)
    }

    /// [`Self::transcribe_clip`] over encoder embeddings already computed.
    pub(crate) fn transcribe_encoded(
        &self,
        audio: Tensor,
        language: Option<&str>,
        max_tokens: usize,
        state: &mut dyn Lfm3AudioLmState,
        should_stop: &mut dyn FnMut() -> bool,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<Lfm3AudioOutput> {
        self.generate(
            asr_request(audio, language, max_tokens)?,
            state,
            should_stop,
            on_text,
            &mut |_| {},
        )
    }

    /// Speak `text` in `voice`: at most `max_frames` frames of 80 ms.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn synthesize(
        &self,
        text: &str,
        voice: Lfm3AudioVoice,
        max_frames: usize,
        sampling: Lfm3AudioSampling,
        seed: u64,
        state: &mut dyn Lfm3AudioLmState,
        should_stop: &mut dyn FnMut() -> bool,
        on_audio: &mut dyn FnMut(&[f32]),
    ) -> Result<Lfm3AudioOutput> {
        self.generate(
            tts_request(text, voice, max_frames, sampling, seed)?,
            state,
            should_stop,
            &mut |_| {},
            on_audio,
        )
    }
}

impl Lfm3AudioModel {
    /// Transcribe audio of any length: one pass up to 30 s, else chunks of
    /// at most 30 s cut at pauses, joined without the words their overlaps
    /// repeat. Returns the text and the chunking diagnostics.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transcribe(
        &self,
        samples: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        max_tokens: Option<usize>,
        state: &mut dyn Lfm3AudioLmState,
        should_stop: &mut dyn FnMut() -> bool,
        on_text: &mut dyn FnMut(&str),
    ) -> Result<(String, serde_json::Value)> {
        let seconds = |samples: usize| samples as f32 / sample_rate.max(1) as f32;
        let config = AsrLongFormConfig::default();
        let chunks = plan_audio_chunks(samples, sample_rate, &config, None);
        if chunks.len() <= 1 {
            let budget =
                max_tokens.unwrap_or_else(|| lfm3_audio_asr_max_tokens(seconds(samples.len())));
            let output = self.transcribe_clip(
                samples,
                sample_rate,
                language,
                budget,
                state,
                should_stop,
                on_text,
            )?;
            return Ok((
                output.text.trim().to_string(),
                serde_json::json!({
                    "model": "lfm3_audio",
                    "task": "asr",
                    "chunks": 1,
                    "max_new_tokens": budget,
                    "stop_reason": output.stop_reason,
                }),
            ));
        }
        let mut assembler = TranscriptAssembler::new(config.clone());
        let mut chunk_diagnostics = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            if should_stop() {
                return Err(Error::Cancelled(
                    "LFM3-Audio transcription cancelled".into(),
                ));
            }
            let clip = &samples[chunk.start_sample..chunk.end_sample.min(samples.len())];
            let budget =
                max_tokens.unwrap_or_else(|| lfm3_audio_asr_max_tokens(seconds(clip.len())));
            let output = self.transcribe_clip(
                clip,
                sample_rate,
                language,
                budget,
                state,
                should_stop,
                &mut |_| {},
            )?;
            let delta = assembler.push_chunk_text(&output.text);
            if !delta.is_empty() {
                on_text(&delta);
            }
            chunk_diagnostics.push(serde_json::json!({
                "start_seconds": seconds(chunk.start_sample),
                "end_seconds": seconds(chunk.end_sample),
                "max_new_tokens": budget,
                "stop_reason": output.stop_reason,
            }));
        }
        Ok((
            assembler.text().trim().to_string(),
            serde_json::json!({
                "model": "lfm3_audio",
                "task": "asr",
                "chunks": chunks.len(),
                "chunk_transcriptions": chunk_diagnostics,
            }),
        ))
    }
}

/// Frames a TTS request without its own budget may speak: about 1.5 times
/// a slow reading of the text (11 characters a second, CJK 4) plus 2 s,
/// between 4 s and the per-request cap.
pub fn lfm3_audio_tts_auto_max_frames(text: &str) -> usize {
    let cjk = text
        .chars()
        .filter(|ch| {
            matches!(*ch as u32,
                0x3040..=0x30ff | 0x3400..=0x4dbf | 0x4e00..=0x9fff | 0xac00..=0xd7af)
        })
        .count();
    let other = text.chars().filter(|ch| !ch.is_whitespace()).count() - cjk;
    let seconds = (other as f32 / 11.0 + cjk as f32 / 4.0) * 1.5 + 2.0;
    let frames = (seconds * FRAMES_PER_SECOND).ceil() as usize;
    frames.clamp(
        (4.0 * FRAMES_PER_SECOND) as usize,
        crate::model::ModelVariant::LFM3_AUDIO_MAX_OUTPUT_FRAMES,
    )
}

/// The longest clip one transcription covers: the training clips' length.
/// Longer audio is transcribed in chunks.
pub(crate) const LFM3_AUDIO_MAX_CLIP_SECS: f32 = 30.0;

/// ASR tokens a clip of `seconds` may need: 12 per second (well above
/// speech in any of the 18 languages) plus 32, between 64 and 512.
pub(crate) fn lfm3_audio_asr_max_tokens(seconds: f32) -> usize {
    let seconds = if seconds.is_finite() {
        seconds.max(0.0)
    } else {
        0.0
    };
    ((seconds * 12.0).ceil() as usize + 32).clamp(64, 512)
}

#[path = "live_asr.rs"]
mod live_asr;
pub(crate) use live_asr::{
    lfm3_live_asr_operation_bounds, Lfm3LiveAsr, Lfm3LiveTranscript, LFM3_LIVE_ASR_OVERLAP_SECONDS,
    LFM3_LIVE_ASR_SEGMENT_SECONDS,
};

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
