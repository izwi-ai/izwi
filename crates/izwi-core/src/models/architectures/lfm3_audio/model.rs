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
    asr_prompt, tts_prompt, Lfm3AudioVoice, PromptBuilder, PromptPart,
    LFM3_AUDIO_INTERLEAVED_SYSTEM_PROMPT,
};
use super::sampling::{apply_repetition_penalty, sample};
use crate::models::shared::chat::{ChatMessage, ChatRole};
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
    /// Encoder embeddings `[1, tokens, 2048]` for the prompt's speech clip.
    pub(crate) audio: Option<Tensor>,
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
        let lm = QuantizedLfm2Backbone::load(&lm_loader, lm_config.clone(), &device.device)?;
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
        let mut clips = request.audio.iter();
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
        if prompt_tokens >= context {
            return Err(Error::InvalidInput(format!(
                "LFM3-Audio prompt of {prompt_tokens} tokens does not fit this request's {context}-token context"
            )));
        }
        let max_slots = request.max_slots.min(context - prompt_tokens);
        let hidden = state.forward(&self.lm, &embeds, 0)?;
        let mut last = hidden
            .i((.., prompt_tokens - 1..prompt_tokens, ..))?
            .squeeze(1)?;
        let mut rng = SplitMix64::new(request.seed);
        let mut decoder = IncrementalDecoder::new(true);
        let mut output = Lfm3AudioOutput {
            prompt_tokens,
            stop_reason: "length",
            ..Lfm3AudioOutput::default()
        };
        let mut text_history = std::collections::VecDeque::new();
        let mut audio_history = FrameHistory::default();
        let mut codec = LiquidCodecStream::default();
        let mut pending_frames = Vec::<[u32; CODEBOOKS]>::new();
        let mut mode = request.mode;
        let mut slots_left = INTERLEAVED_TEXT;
        let (mut speaking, mut text_done) = (false, false);
        let mut suppress = vec![AUDIO_PLACEHOLDER, AUDIO_OUT_PLACEHOLDER];
        if request.text_only {
            suppress.extend([AUDIO_START, MIXED_START]);
        }

        let mut flush = |pending: &mut Vec<[u32; CODEBOOKS]>,
                         codec: &mut LiquidCodecStream,
                         output: &mut Lfm3AudioOutput,
                         finalize: bool|
         -> Result<()> {
            if pending.is_empty() && !finalize {
                return Ok(());
            }
            let samples = self.codec.decode_stream(codec, pending, finalize)?;
            pending.clear();
            if !samples.is_empty() {
                on_audio(&samples);
                output.samples.extend_from_slice(&samples);
            }
            Ok(())
        };

        for slot in 0..max_slots {
            if should_stop() {
                output.stop_reason = "cancelled";
                break;
            }
            let interleaving = mode == Lfm3AudioMode::Interleaved;
            let mut finished = false;
            let next = if mode == Lfm3AudioMode::Audio || (interleaving && speaking) {
                let frame = self.depth.sample_frames(
                    &last,
                    &request.audio_sampling,
                    &[&audio_history],
                    &mut [&mut rng],
                )?[0];
                if frame[0] == AUDIO_EOS_CODE {
                    mode = Lfm3AudioMode::Text;
                    flush(&mut pending_frames, &mut codec, &mut output, true)?;
                    codec = LiquidCodecStream::default();
                } else {
                    audio_history.push(frame, request.audio_sampling.repetition_window);
                    output.audio_frames += 1;
                    output.frames.push(frame);
                    pending_frames.push(frame);
                    let due = if output.audio_frames <= FIRST_STREAM_FRAMES {
                        FIRST_STREAM_FRAMES
                    } else {
                        STREAM_FRAMES
                    };
                    if pending_frames.len() >= due {
                        flush(&mut pending_frames, &mut codec, &mut output, false)?;
                    }
                }
                self.feedback_embed(&frame)?
            } else {
                let mut scores = self
                    .lm
                    .project_hidden(&last)?
                    .squeeze(0)?
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<f32>()?;
                for (score, mask) in scores.iter_mut().zip(&self.text_mask) {
                    *score += mask;
                }
                apply_repetition_penalty(
                    &mut scores,
                    text_history.iter().copied(),
                    request.text_sampling.repetition_penalty,
                );
                for id in &suppress {
                    scores[*id as usize] = f32::NEG_INFINITY;
                }
                let token = sample(
                    &mut scores,
                    request.text_sampling.temperature,
                    request.text_sampling.top_k,
                    request.text_sampling.top_p,
                    &mut rng,
                );
                if request.text_sampling.repetition_window > 0 {
                    if text_history.len() == request.text_sampling.repetition_window {
                        text_history.pop_front();
                    }
                    text_history.push_back(token);
                }
                output.text_tokens += 1;
                let delta = self.tokenizer.decode_incrementally(&mut decoder, token)?;
                if !delta.is_empty() {
                    on_text(&delta);
                    output.text.push_str(&delta);
                }
                if token == TEXT_END {
                    if interleaving {
                        text_done = true;
                        speaking = true;
                    } else {
                        finished = true;
                    }
                } else if token == IM_END || token == END_OF_TEXT {
                    finished = true;
                } else if interleaving {
                    // Inside a block the mode markers are ordinary text.
                } else if token == AUDIO_START {
                    mode = Lfm3AudioMode::Audio;
                } else if token == MIXED_START {
                    mode = Lfm3AudioMode::Interleaved;
                    slots_left = INTERLEAVED_TEXT;
                    speaking = false;
                    text_done = false;
                }
                let input = Tensor::from_vec(vec![token], (1, 1), &self.device.device)?;
                self.lm.embed_tokens(&input)?
            };
            if interleaving && !text_done {
                slots_left -= 1;
                if slots_left == 0 {
                    speaking = !speaking;
                    slots_left = if speaking {
                        INTERLEAVED_AUDIO
                    } else {
                        INTERLEAVED_TEXT
                    };
                }
            }
            if finished {
                output.stop_reason = "eos";
                break;
            }
            if slot + 1 == max_slots {
                break;
            }
            let hidden = state.forward(&self.lm, &next, prompt_tokens + slot)?;
            last = hidden.squeeze(1)?;
        }
        let tail = self.tokenizer.finish_incremental_decode(&mut decoder)?;
        if !tail.is_empty() {
            on_text(&tail);
            output.text.push_str(&tail);
        }
        if codec.has_rows() || !pending_frames.is_empty() {
            flush(&mut pending_frames, &mut codec, &mut output, true)?;
        }
        Ok(output)
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
        let prompt = asr_prompt(language)?;
        self.generate(
            Lfm3AudioRequest {
                prompt,
                audio: Some(audio),
                mode: Lfm3AudioMode::Text,
                text_only: true,
                max_slots: max_tokens.max(1),
                text_sampling: Lfm3TextSampling::default(),
                audio_sampling: Lfm3AudioSampling::default(),
                seed: 0,
            },
            state,
            should_stop,
            on_text,
            &mut |_| {},
        )
    }

    /// The transcript of a spoken turn whose encoder embeddings `audio` the
    /// reply will reuse: one pass over them when the turn fits one ASR
    /// window, else the chunked long form over `samples`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn transcribe_turn(
        &self,
        samples: &[f32],
        sample_rate: u32,
        audio: &Tensor,
        state: &mut dyn Lfm3AudioLmState,
        should_stop: &mut dyn FnMut() -> bool,
    ) -> Result<String> {
        let config = AsrLongFormConfig::default();
        if plan_audio_chunks(samples, sample_rate, &config, None).len() > 1 {
            let (text, _) = self.transcribe(
                samples,
                sample_rate,
                None,
                None,
                state,
                should_stop,
                &mut |_| {},
            )?;
            return Ok(text);
        }
        let seconds = samples.len() as f32 / sample_rate.max(1) as f32;
        let output = self.transcribe_encoded(
            audio.clone(),
            None,
            lfm3_audio_asr_max_tokens(seconds),
            state,
            should_stop,
            &mut |_| {},
        )?;
        if output.stop_reason == "cancelled" {
            return Err(Error::Cancelled(
                "LFM3-Audio transcription cancelled".into(),
            ));
        }
        Ok(output.text.trim().to_string())
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
        if text.trim().is_empty() {
            return Err(Error::InvalidInput("LFM3-Audio TTS needs text".into()));
        }
        self.generate(
            Lfm3AudioRequest {
                prompt: tts_prompt(text, voice),
                audio: None,
                mode: Lfm3AudioMode::Audio,
                text_only: false,
                // The end-of-audio frame and the closing `<|im_end|>`.
                max_slots: max_frames.max(1) + 2,
                text_sampling: Lfm3TextSampling::default(),
                audio_sampling: sampling,
                seed,
            },
            state,
            should_stop,
            &mut |_| {},
            on_audio,
        )
    }

    /// Answer a spoken turn, after `history`'s text turns, with interleaved
    /// text and speech (`Respond with interleaved text and audio.`, the
    /// reply prefilled with `<|mixed_start|>`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn respond(
        &self,
        history: &[ChatMessage],
        audio: Tensor,
        system_prompt: Option<&str>,
        max_slots: usize,
        text_sampling: Lfm3TextSampling,
        audio_sampling: Lfm3AudioSampling,
        seed: u64,
        state: &mut dyn Lfm3AudioLmState,
        should_stop: &mut dyn FnMut() -> bool,
        on_text: &mut dyn FnMut(&str),
        on_audio: &mut dyn FnMut(&[f32]),
    ) -> Result<Lfm3AudioOutput> {
        let system = match system_prompt.map(str::trim).filter(|text| !text.is_empty()) {
            Some(extra) => format!("{LFM3_AUDIO_INTERLEAVED_SYSTEM_PROMPT}\n\n{extra}"),
            None => LFM3_AUDIO_INTERLEAVED_SYSTEM_PROMPT.to_string(),
        };
        let mut prompt = PromptBuilder::new().turn("system", &system);
        for message in history {
            let role = match message.role {
                ChatRole::System => continue,
                ChatRole::User | ChatRole::Tool => "user",
                ChatRole::Assistant => "assistant",
            };
            prompt = prompt.turn(role, &message.content);
        }
        let prompt = prompt.audio_turn("user", "").assistant("<|mixed_start|>");
        self.generate(
            Lfm3AudioRequest {
                prompt,
                audio: Some(audio),
                mode: Lfm3AudioMode::Interleaved,
                text_only: false,
                max_slots: max_slots.max(1),
                text_sampling,
                audio_sampling,
                seed,
            },
            state,
            should_stop,
            on_text,
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

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
