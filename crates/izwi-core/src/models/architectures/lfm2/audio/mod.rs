//! Native Candle implementation for LiquidAI LFM2-Audio.

mod audio_detokenizer;
mod config;
mod conformer;
mod depthformer;
mod lfm_backbone;
mod mimi_decoder;
mod preprocessor;
mod tokenizer;

use std::cmp::Ordering;
use std::path::{Path, PathBuf};

use audio_detokenizer::AudioDetokenizer;
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};
use config::Lfm2AudioConfig;
use conformer::ConformerEncoder;
use depthformer::Depthformer;
use lfm_backbone::{LfmBackbone, LfmCache};
use mimi_decoder::MimiDecoder;
use preprocessor::{resample_audio, Lfm2AudioPreprocessor};
use tokenizer::{ChatState, Lfm2Tokenizer, LfmModality};
use tracing::{info, warn};

use crate::error::{Error, Result};
use crate::models::shared::device::DeviceProfile;

pub const LFM2_DEFAULT_S2S_PROMPT: &str = "Respond with interleaved text and audio.";

const TTS_US_MALE_PROMPT: &str = "Perform TTS. Use the US male voice.";
const TTS_US_FEMALE_PROMPT: &str = "Perform TTS. Use the US female voice.";
const TTS_UK_MALE_PROMPT: &str = "Perform TTS. Use the UK male voice.";
const TTS_UK_FEMALE_PROMPT: &str = "Perform TTS. Use the UK female voice.";

const END_OF_AUDIO_TOKEN: u32 = 2048;
const ASR_SYSTEM_PROMPT: &str = "Perform ASR.";
const MIMI_TOKENIZER_CHECKPOINT: &str = "tokenizer-e351c8d8-checkpoint125.safetensors";
const DETOKENIZER_CONFIG_FILE: &str = "audio_detokenizer/config.json";
const DETOKENIZER_MODEL_FILE: &str = "audio_detokenizer/model.safetensors";

enum WaveDecoder {
    Mimi(MimiDecoder),
    Lfm25Detokenizer(AudioDetokenizer),
}

impl WaveDecoder {
    fn decode_tokens(&self, codebooks: &[Vec<u32>]) -> Result<Vec<f32>> {
        match self {
            Self::Mimi(decoder) => decoder.decode_tokens(codebooks),
            Self::Lfm25Detokenizer(decoder) => decoder.decode_tokens(codebooks),
        }
    }
}

pub struct Lfm2AudioModel {
    model_dir: PathBuf,
    device: DeviceProfile,
    cfg: Lfm2AudioConfig,
    tokenizer: Lfm2Tokenizer,
    preprocessor: Lfm2AudioPreprocessor,
    conformer: ConformerEncoder,
    audio_adapter: AudioAdapter,
    lfm: LfmBackbone,
    depthformer: Depthformer,
    wave_decoder: WaveDecoder,
}

pub struct SpeechToSpeechDecodeState {
    cache: LfmCache,
    in_emb: Tensor,
    current_modality: LfmModality,
    modality_left: usize,
    text_done: bool,
    max_new_tokens: usize,
    steps: usize,
    text_tokens: Vec<u32>,
    assembled: String,
    rng: SimpleRng,
    text_temperature: Option<f32>,
    text_top_k: Option<usize>,
    audio_temperature: Option<f32>,
    audio_top_k: Option<usize>,
    finished: bool,
}

pub struct TtsDecodeState {
    cache: LfmCache,
    in_emb: Tensor,
    current_modality: LfmModality,
    max_new_tokens: usize,
    steps: usize,
    text_tokens: Vec<u32>,
    assembled: String,
    rng: SimpleRng,
    text_temperature: Option<f32>,
    text_top_k: Option<usize>,
    audio_temperature: Option<f32>,
    audio_top_k: Option<usize>,
    finished: bool,
}

#[derive(Debug, Clone)]
pub struct TtsDecodeStep {
    pub delta: String,
    pub text: String,
    pub audio_frame: Option<Vec<u32>>,
    pub tokens_generated: usize,
    pub finished: bool,
}

#[derive(Debug, Clone)]
pub struct SpeechToSpeechDecodeStep {
    pub delta: String,
    pub text: String,
    pub audio_frame: Option<Vec<u32>>,
    pub tokens_generated: usize,
    pub finished: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Lfm2TtsReference<'a> {
    pub audio: &'a [f32],
    pub sample_rate: u32,
    pub text: &'a str,
}

struct AudioAdapter {
    norm: LayerNorm,
    linear1: Linear,
    linear2: Linear,
}

impl AudioAdapter {
    fn load(vb: VarBuilder) -> Result<Self> {
        let layers_prefix = if vb.contains_tensor("model.0.weight") {
            "model"
        } else if vb.contains_tensor("layers.0.weight") {
            "layers"
        } else {
            return Err(Error::ModelLoadError(
                "Unsupported LFM2 audio_adapter layout: expected model.0 or layers.0 tensors"
                    .to_string(),
            ));
        };

        let ln_dim = vb
            .pp(format!("{layers_prefix}.0"))
            .get(1, "weight")
            .map(|t| t.dim(0).unwrap_or(0))
            .unwrap_or(512);

        let norm = candle_nn::layer_norm(ln_dim, 1e-5, vb.pp(format!("{layers_prefix}.0")))?;
        let linear1 = crate::models::shared::weights::mlx::load_linear(
            ln_dim,
            2048,
            vb.pp(format!("{layers_prefix}.1")),
        )?;
        let linear2 = crate::models::shared::weights::mlx::load_linear(
            2048,
            2048,
            vb.pp(format!("{layers_prefix}.3")),
        )?;

        Ok(Self {
            norm,
            linear1,
            linear2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.norm.forward(x)?;
        let x = self.linear1.forward(&x)?;
        let x = x.gelu_erf()?;
        self.linear2.forward(&x).map_err(Error::from)
    }
}

impl Lfm2AudioModel {
    pub fn load(model_dir: &Path, device: DeviceProfile) -> Result<Self> {
        validate_model_dir(model_dir)?;

        let cfg: Lfm2AudioConfig =
            serde_json::from_str(std::fs::read_to_string(model_dir.join("config.json"))?.as_str())
                .map_err(|e| Error::ModelLoadError(format!("Invalid LFM2 config.json: {e}")))?;

        let dtype = match device.kind {
            crate::models::shared::device::DeviceKind::Cuda
                if device.capabilities.supports_bf16 =>
            {
                DType::BF16
            }
            _ => DType::F32,
        };

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_dir.join("model.safetensors")],
                dtype,
                &device.device,
            )?
        };

        let conformer_vb = if vb.contains_tensor("conformer.pre_encode.conv.0.weight") {
            vb.pp("conformer")
        } else if vb.contains_tensor("audio_encoder.pre_encode.conv.0.weight") {
            vb.pp("audio_encoder")
        } else {
            return Err(Error::ModelLoadError(
                "Unsupported LFM2 checkpoint layout: missing conformer/audio_encoder tensors"
                    .to_string(),
            ));
        };

        let tokenizer = Lfm2Tokenizer::load(model_dir)?;
        let preprocessor = Lfm2AudioPreprocessor::new(cfg.preprocessor.clone())?;
        let conformer = ConformerEncoder::load(cfg.encoder.clone(), conformer_vb)?;
        let audio_adapter = AudioAdapter::load(vb.pp("audio_adapter"))?;
        let lfm = LfmBackbone::load(cfg.lfm.clone(), vb.pp("lfm"))?;
        let depthformer = Depthformer::load(&cfg, vb.clone())?;
        let has_detokenizer = model_dir.join("audio_detokenizer").exists();
        let has_mimi = model_dir.join(MIMI_TOKENIZER_CHECKPOINT).exists();
        let allow_mimi_fallback = env_flag_enabled("IZWI_LFM2_ALLOW_MIMI_FALLBACK") && has_mimi;

        let wave_decoder = if has_detokenizer {
            match AudioDetokenizer::load(model_dir, &device.device) {
                Ok(detokenizer) => {
                    info!("Using native LFM2.5 audio detokenizer");
                    WaveDecoder::Lfm25Detokenizer(detokenizer)
                }
                Err(err) => {
                    if allow_mimi_fallback {
                        warn!(
                            "Failed to load LFM2.5 audio detokenizer ({}); IZWI_LFM2_ALLOW_MIMI_FALLBACK=1 is set, falling back to Mimi",
                            err
                        );
                        WaveDecoder::Mimi(MimiDecoder::load(model_dir, &device.device)?)
                    } else {
                        return Err(Error::ModelLoadError(format!(
                            "Failed to load LFM2.5 audio detokenizer: {err}. \
Set IZWI_LFM2_ALLOW_MIMI_FALLBACK=1 to force legacy Mimi fallback."
                        )));
                    }
                }
            }
        } else if has_mimi {
            info!("audio_detokenizer is unavailable, using legacy Mimi decoder");
            WaveDecoder::Mimi(MimiDecoder::load(model_dir, &device.device)?)
        } else {
            return Err(Error::ModelLoadError(
                "LFM2 model is missing both audio_detokenizer and Mimi tokenizer weights"
                    .to_string(),
            ));
        };

        info!("Loaded native LFM2 model from {:?}", model_dir);

        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            device,
            cfg,
            tokenizer,
            preprocessor,
            conformer,
            audio_adapter,
            lfm,
            depthformer,
            wave_decoder,
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn available_voices(&self) -> Vec<String> {
        vec![
            "US Male".to_string(),
            "US Female".to_string(),
            "UK Male".to_string(),
            "UK Female".to_string(),
        ]
    }

    pub fn transcribe_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let mel = self.prepare_audio_mel(audio, sample_rate)?;

        let mut state = ChatState::new(
            &self.tokenizer,
            self.cfg.codebooks,
            self.preprocessor.features(),
        );
        state.new_turn(&self.tokenizer, "system")?;
        let asr_prompt = asr_system_prompt(language);
        state.add_text(&self.tokenizer, asr_prompt.as_str())?;
        state.end_turn(&self.tokenizer)?;

        state.new_turn(&self.tokenizer, "user")?;
        state.add_audio_mel(&mel.0, mel.1);
        state.end_turn(&self.tokenizer)?;

        state.new_turn(&self.tokenizer, "assistant")?;

        let mut tokens = Vec::new();
        let mut assembled = String::new();
        let mut rng = SimpleRng::new();

        self.generate_sequential(
            &state,
            512,
            false,
            None,
            None,
            Some(1),
            None,
            None,
            &mut rng,
            &mut |token| {
                tokens.push(token);
                if let Ok(decoded) = self.tokenizer.decode_text(&tokens) {
                    let delta = text_delta(&assembled, &decoded);
                    if !delta.is_empty() {
                        on_delta(delta.as_str());
                    }
                    assembled = decoded;
                }
            },
            &mut |_frame| {},
        )?;

        Ok(assembled.trim().to_string())
    }

    pub fn synthesize_with_callback(
        &self,
        text: &str,
        speaker_prompt: &str,
        temperature: Option<f32>,
        top_k: Option<usize>,
        max_new_tokens: usize,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<Vec<f32>> {
        self.synthesize_with_callback_with_reference(
            text,
            speaker_prompt,
            None,
            temperature,
            top_k,
            max_new_tokens,
            on_delta,
        )
    }

    pub fn synthesize_with_callback_with_reference(
        &self,
        text: &str,
        speaker_prompt: &str,
        reference: Option<Lfm2TtsReference<'_>>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        max_new_tokens: usize,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<Vec<f32>> {
        let state = self.build_tts_chat_state(text, speaker_prompt, reference)?;

        let mut text_tokens = Vec::new();
        let mut assembled = String::new();
        let mut audio_frames: Vec<Vec<u32>> = vec![Vec::new(); self.cfg.codebooks];
        let mut rng = SimpleRng::new();

        self.generate_sequential(
            &state,
            max_new_tokens.max(256),
            true,
            None,
            None,
            None,
            temperature,
            top_k,
            &mut rng,
            &mut |token| {
                text_tokens.push(token);
                if let Ok(decoded) = self.tokenizer.decode_text(&text_tokens) {
                    let delta = text_delta(&assembled, &decoded);
                    if !delta.is_empty() {
                        on_delta(delta.as_str());
                    }
                    assembled = decoded;
                }
            },
            &mut |frame| {
                if is_end_of_audio_frame(frame) {
                    return;
                }
                for (i, &tok) in frame.iter().enumerate() {
                    if i < audio_frames.len() {
                        audio_frames[i].push(tok);
                    }
                }
            },
        )?;

        trim_audio_frames(&mut audio_frames);
        self.wave_decoder.decode_tokens(&audio_frames)
    }

    pub fn start_tts_decode(
        &self,
        text: &str,
        speaker_prompt: &str,
        temperature: Option<f32>,
        top_k: Option<usize>,
        max_new_tokens: usize,
    ) -> Result<TtsDecodeState> {
        self.start_tts_decode_with_reference(
            text,
            speaker_prompt,
            None,
            temperature,
            top_k,
            max_new_tokens,
        )
    }

    pub fn start_tts_decode_with_reference(
        &self,
        text: &str,
        speaker_prompt: &str,
        reference: Option<Lfm2TtsReference<'_>>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        max_new_tokens: usize,
    ) -> Result<TtsDecodeState> {
        let state = self.build_tts_chat_state(text, speaker_prompt, reference)?;

        Ok(TtsDecodeState {
            cache: LfmCache::new(self.lfm.config()),
            in_emb: self.build_prefill_embeddings(&state)?,
            current_modality: LfmModality::Text,
            max_new_tokens: max_new_tokens.max(256),
            steps: 0,
            text_tokens: Vec::new(),
            assembled: String::new(),
            rng: SimpleRng::new(),
            text_temperature: None,
            text_top_k: None,
            audio_temperature: temperature,
            audio_top_k: top_k,
            finished: false,
        })
    }

    pub fn tts_decode_step(&self, state: &mut TtsDecodeState) -> Result<TtsDecodeStep> {
        if state.finished || state.steps >= state.max_new_tokens {
            state.finished = true;
            return Ok(TtsDecodeStep {
                delta: String::new(),
                text: state.assembled.trim().to_string(),
                audio_frame: None,
                tokens_generated: state.steps,
                finished: true,
            });
        }

        let out = self
            .lfm
            .forward_embeds_cached(&state.in_emb, &mut state.cache)?;
        let last = out.i((0, out.dim(1)? - 1, ..))?;

        let mut delta = String::new();
        let mut audio_frame = None;

        match state.current_modality {
            LfmModality::Text => {
                let logits = last
                    .reshape((1, last.dim(0)?))?
                    .matmul(&self.lfm.embed_tokens_weight().t()?)?
                    .squeeze(0)?;
                let token = sample_token(
                    &logits,
                    state.text_temperature.unwrap_or(0.0) <= 0.0 || state.text_top_k == Some(1),
                    state.text_temperature,
                    state.text_top_k,
                    &mut state.rng,
                )?;

                if token == self.tokenizer.specials().im_end {
                    state.finished = true;
                    return Ok(TtsDecodeStep {
                        delta,
                        text: state.assembled.trim().to_string(),
                        audio_frame,
                        tokens_generated: state.steps,
                        finished: true,
                    });
                }

                state.text_tokens.push(token);
                if let Ok(decoded) = self.tokenizer.decode_text(&state.text_tokens) {
                    delta = text_delta(&state.assembled, &decoded);
                    state.assembled = decoded;
                }

                if token == self.tokenizer.specials().audio_start {
                    state.current_modality = LfmModality::AudioOut;
                }

                state.in_emb = self.lfm.embed_tokens(token)?;
            }
            LfmModality::AudioOut => {
                let mut frame = self.depthformer.sample_audio_frame(
                    &last,
                    state.audio_temperature,
                    state.audio_top_k,
                    &mut state.rng,
                )?;

                if is_end_of_audio_frame(&frame) {
                    for t in &mut frame {
                        *t = END_OF_AUDIO_TOKEN;
                    }
                    state.finished = true;
                }

                audio_frame = Some(frame.clone());
                let frame_t = Tensor::from_vec(
                    frame,
                    self.cfg.codebooks,
                    self.lfm.embed_tokens_weight().device(),
                )?
                .to_dtype(DType::U32)?;
                state.in_emb = self.depthformer.audio_embedding_sum(&frame_t)?;
            }
            LfmModality::AudioIn => {}
        }

        state.steps = state.steps.saturating_add(1);
        if state.steps >= state.max_new_tokens {
            state.finished = true;
        }

        Ok(TtsDecodeStep {
            delta,
            text: state.assembled.trim().to_string(),
            audio_frame,
            tokens_generated: state.steps,
            finished: state.finished,
        })
    }

    pub fn speech_to_speech_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        system_prompt: Option<&str>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        max_new_tokens: usize,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<(String, Vec<f32>)> {
        let mel = self.prepare_audio_mel(audio, sample_rate)?;

        let mut state = ChatState::new(
            &self.tokenizer,
            self.cfg.codebooks,
            self.preprocessor.features(),
        );
        state.new_turn(&self.tokenizer, "system")?;
        let s2s_prompt = s2s_system_prompt(system_prompt, language);
        state.add_text(&self.tokenizer, s2s_prompt.as_str())?;
        state.end_turn(&self.tokenizer)?;

        state.new_turn(&self.tokenizer, "user")?;
        state.add_audio_mel(&mel.0, mel.1);
        state.end_turn(&self.tokenizer)?;

        state.new_turn(&self.tokenizer, "assistant")?;

        let mut text_tokens = Vec::new();
        let mut assembled = String::new();
        let mut audio_frames: Vec<Vec<u32>> = vec![Vec::new(); self.cfg.codebooks];
        let mut rng = SimpleRng::new();

        self.generate_interleaved(
            &state,
            max_new_tokens.max(768),
            None,
            None,
            temperature,
            top_k,
            &mut rng,
            &mut |token| {
                text_tokens.push(token);
                if let Ok(decoded) = self.tokenizer.decode_text(&text_tokens) {
                    let delta = text_delta(&assembled, &decoded);
                    if !delta.is_empty() {
                        on_delta(delta.as_str());
                    }
                    assembled = decoded;
                }
            },
            &mut |frame| {
                if is_end_of_audio_frame(frame) {
                    return;
                }
                for (i, &tok) in frame.iter().enumerate() {
                    if i < audio_frames.len() {
                        audio_frames[i].push(tok);
                    }
                }
            },
        )?;

        trim_audio_frames(&mut audio_frames);
        let wav = self.wave_decoder.decode_tokens(&audio_frames)?;
        Ok((assembled.trim().to_string(), wav))
    }

    pub fn start_speech_to_speech_decode(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        system_prompt: Option<&str>,
        audio_temperature: Option<f32>,
        audio_top_k: Option<usize>,
        max_new_tokens: usize,
    ) -> Result<SpeechToSpeechDecodeState> {
        let mel = self.prepare_audio_mel(audio, sample_rate)?;

        let mut state = ChatState::new(
            &self.tokenizer,
            self.cfg.codebooks,
            self.preprocessor.features(),
        );
        state.new_turn(&self.tokenizer, "system")?;
        let s2s_prompt = s2s_system_prompt(system_prompt, language);
        state.add_text(&self.tokenizer, s2s_prompt.as_str())?;
        state.end_turn(&self.tokenizer)?;

        state.new_turn(&self.tokenizer, "user")?;
        state.add_audio_mel(&mel.0, mel.1);
        state.end_turn(&self.tokenizer)?;

        state.new_turn(&self.tokenizer, "assistant")?;

        Ok(SpeechToSpeechDecodeState {
            cache: LfmCache::new(self.lfm.config()),
            in_emb: self.build_prefill_embeddings(&state)?,
            current_modality: LfmModality::Text,
            modality_left: self.cfg.interleaved_n_text,
            text_done: false,
            max_new_tokens: max_new_tokens.max(1),
            steps: 0,
            text_tokens: Vec::new(),
            assembled: String::new(),
            rng: SimpleRng::new(),
            text_temperature: None,
            text_top_k: None,
            audio_temperature,
            audio_top_k,
            finished: false,
        })
    }

    pub fn speech_to_speech_decode_step(
        &self,
        state: &mut SpeechToSpeechDecodeState,
    ) -> Result<SpeechToSpeechDecodeStep> {
        if state.finished || state.steps >= state.max_new_tokens {
            state.finished = true;
            return Ok(SpeechToSpeechDecodeStep {
                delta: String::new(),
                text: state.assembled.trim().to_string(),
                audio_frame: None,
                tokens_generated: state.steps,
                finished: true,
            });
        }

        state.modality_left = state.modality_left.saturating_sub(1);
        let out = self
            .lfm
            .forward_embeds_cached(&state.in_emb, &mut state.cache)?;
        let last = out.i((0, out.dim(1)? - 1, ..))?;

        let mut delta = String::new();
        let mut audio_frame = None;

        match state.current_modality {
            LfmModality::Text => {
                let logits = last
                    .reshape((1, last.dim(0)?))?
                    .matmul(&self.lfm.embed_tokens_weight().t()?)?
                    .squeeze(0)?;
                let token = sample_token(
                    &logits,
                    state.text_temperature.unwrap_or(0.0) <= 0.0 || state.text_top_k == Some(1),
                    state.text_temperature,
                    state.text_top_k,
                    &mut state.rng,
                )?;

                if token == self.tokenizer.specials().im_end {
                    state.finished = true;
                    return Ok(SpeechToSpeechDecodeStep {
                        delta,
                        text: state.assembled.trim().to_string(),
                        audio_frame,
                        tokens_generated: state.steps,
                        finished: true,
                    });
                }

                state.text_tokens.push(token);
                if let Ok(decoded) = self.tokenizer.decode_text(&state.text_tokens) {
                    delta = text_delta(&state.assembled, &decoded);
                    state.assembled = decoded;
                }

                if token == self.tokenizer.specials().text_end {
                    state.text_done = true;
                }

                if state.modality_left == 0 || state.text_done {
                    state.current_modality = LfmModality::AudioOut;
                    state.modality_left = self.cfg.interleaved_n_audio;
                }

                state.in_emb = self.lfm.embed_tokens(token)?;
            }
            LfmModality::AudioOut => {
                let mut frame = self.depthformer.sample_audio_frame(
                    &last,
                    state.audio_temperature,
                    state.audio_top_k,
                    &mut state.rng,
                )?;

                if state.modality_left == 0 && !state.text_done {
                    state.current_modality = LfmModality::Text;
                    state.modality_left = self.cfg.interleaved_n_text;
                }

                if is_end_of_audio_frame(&frame) {
                    for t in &mut frame {
                        *t = END_OF_AUDIO_TOKEN;
                    }
                    state.current_modality = LfmModality::Text;
                    state.modality_left = self.cfg.interleaved_n_text;
                }

                audio_frame = Some(frame.clone());
                let frame_t = Tensor::from_vec(
                    frame,
                    self.cfg.codebooks,
                    self.lfm.embed_tokens_weight().device(),
                )?
                .to_dtype(DType::U32)?;
                state.in_emb = self.depthformer.audio_embedding_sum(&frame_t)?;
            }
            LfmModality::AudioIn => {}
        }

        state.steps = state.steps.saturating_add(1);
        if state.steps >= state.max_new_tokens {
            state.finished = true;
        }

        Ok(SpeechToSpeechDecodeStep {
            delta,
            text: state.assembled.trim().to_string(),
            audio_frame,
            tokens_generated: state.steps,
            finished: state.finished,
        })
    }

    pub fn decode_audio_frame(&self, frame: &[u32]) -> Result<Vec<f32>> {
        if frame.is_empty() || is_end_of_audio_frame(frame) {
            return Ok(Vec::new());
        }

        let mut codebooks: Vec<Vec<u32>> = vec![Vec::new(); self.cfg.codebooks];
        for (i, &tok) in frame.iter().enumerate() {
            if i >= codebooks.len() {
                break;
            }
            codebooks[i].push(tok);
        }
        self.wave_decoder.decode_tokens(&codebooks)
    }

    pub fn decode_audio_frames(&self, frames: &[Vec<u32>]) -> Result<Vec<f32>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        let mut codebooks: Vec<Vec<u32>> = vec![Vec::new(); self.cfg.codebooks];
        for frame in frames {
            if frame.is_empty() || is_end_of_audio_frame(frame) {
                break;
            }

            if frame.len() < self.cfg.codebooks {
                return Err(Error::InferenceError(format!(
                    "LFM2 audio frame length {} is smaller than expected codebooks {}",
                    frame.len(),
                    self.cfg.codebooks
                )));
            }

            for (i, &tok) in frame.iter().take(self.cfg.codebooks).enumerate() {
                codebooks[i].push(tok);
            }
        }

        trim_audio_frames(&mut codebooks);
        self.wave_decoder.decode_tokens(&codebooks)
    }

    fn prepare_audio_mel(&self, audio: &[f32], sample_rate: u32) -> Result<(Vec<f32>, usize)> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let target_sr = self.preprocessor.sample_rate();
        let mono = if sample_rate == target_sr {
            audio.to_vec()
        } else {
            resample_audio(audio, sample_rate, target_sr)?
        };

        let (mel, frames) = self.preprocessor.compute_features(&mono)?;
        let mel = mel.squeeze(0)?.flatten_all()?.to_vec1::<f32>()?;
        Ok((mel, frames))
    }

    fn build_tts_chat_state(
        &self,
        text: &str,
        speaker_prompt: &str,
        reference: Option<Lfm2TtsReference<'_>>,
    ) -> Result<ChatState> {
        let mut state = ChatState::new(
            &self.tokenizer,
            self.cfg.codebooks,
            self.preprocessor.features(),
        );
        state.new_turn(&self.tokenizer, "system")?;
        state.add_text(&self.tokenizer, speaker_prompt)?;
        if reference.is_some() {
            state.add_text(
                &self.tokenizer,
                " Match the speaker characteristics in the reference input.",
            )?;
        }
        state.end_turn(&self.tokenizer)?;

        if let Some(reference) = reference {
            let reference_text = reference.text.trim();
            if reference_text.is_empty() {
                return Err(Error::InvalidInput(
                    "reference_text cannot be empty".to_string(),
                ));
            }
            if reference.sample_rate == 0 {
                return Err(Error::InvalidInput(
                    "reference_audio sample rate must be > 0".to_string(),
                ));
            }

            let (reference_mel, reference_frames) =
                self.prepare_audio_mel(reference.audio, reference.sample_rate)?;
            state.new_turn(&self.tokenizer, "user")?;
            state.add_text(&self.tokenizer, reference_text)?;
            state.add_audio_mel(&reference_mel, reference_frames);
            state.end_turn(&self.tokenizer)?;
        }

        state.new_turn(&self.tokenizer, "user")?;
        state.add_text(&self.tokenizer, text)?;
        state.end_turn(&self.tokenizer)?;

        state.new_turn(&self.tokenizer, "assistant")?;
        Ok(state)
    }

    fn build_prefill_embeddings(&self, state: &ChatState) -> Result<Tensor> {
        let text_ids = Tensor::from_vec(
            state.text.clone(),
            state.text.len(),
            self.lfm.embed_tokens_weight().device(),
        )?
        .to_dtype(DType::U32)?;
        let text_emb = self.lfm.embed_sequence(&text_ids)?; // [T_text, D]

        let mut audio_in_rows: Vec<Tensor> = Vec::new();
        if !state.audio_in_lens.is_empty() {
            let total_frames: usize = state.audio_in_lens.iter().sum();
            let audio_in = Tensor::from_vec(
                state.audio_in.clone(),
                (state.features, total_frames),
                self.lfm.embed_tokens_weight().device(),
            )?;

            let mut start = 0usize;
            for &len in &state.audio_in_lens {
                let seg = audio_in.narrow(1, start, len)?;
                let seg = seg.unsqueeze(0)?;
                let (enc, enc_len) = self.conformer.encode(&seg, len)?;
                let enc = enc.i((0, ..enc_len, ..))?;
                let enc = self.audio_adapter.forward(&enc)?;
                for i in 0..enc_len {
                    audio_in_rows.push(enc.i((i, ..))?);
                }
                start += len;
            }
        }

        let mut audio_out_rows: Vec<Tensor> = Vec::new();
        if !state.audio_out.is_empty() {
            let frames = state.audio_out.len() / self.cfg.codebooks;
            let audio_out = Tensor::from_vec(
                state.audio_out.clone(),
                (self.cfg.codebooks, frames),
                self.lfm.embed_tokens_weight().device(),
            )?
            .to_dtype(DType::U32)?;
            for t in 0..frames {
                let frame = audio_out.i((.., t))?;
                let emb = self.depthformer.audio_embedding_sum(&frame)?;
                audio_out_rows.push(emb.squeeze(0)?.squeeze(0)?);
            }
        }

        let mut text_i = 0usize;
        let mut audio_in_i = 0usize;
        let mut audio_out_i = 0usize;
        let mut rows = Vec::with_capacity(state.modality_flag.len());

        for &m in &state.modality_flag {
            match m {
                x if x == LfmModality::Text as u32 => {
                    rows.push(text_emb.i((text_i, ..))?);
                    text_i += 1;
                }
                x if x == LfmModality::AudioIn as u32 => {
                    let row = audio_in_rows
                        .get(audio_in_i)
                        .ok_or_else(|| {
                            Error::InferenceError("audio_in/modality mismatch".to_string())
                        })?
                        .clone();
                    rows.push(row);
                    audio_in_i += 1;
                }
                x if x == LfmModality::AudioOut as u32 => {
                    let row = audio_out_rows
                        .get(audio_out_i)
                        .ok_or_else(|| {
                            Error::InferenceError("audio_out/modality mismatch".to_string())
                        })?
                        .clone();
                    rows.push(row);
                    audio_out_i += 1;
                }
                _ => {
                    return Err(Error::InferenceError(
                        "Unsupported LFM2 modality flag".to_string(),
                    ));
                }
            }
        }

        if rows.is_empty() {
            return Err(Error::InferenceError(
                "Empty LFM2 prefill embedding sequence".to_string(),
            ));
        }

        Ok(Tensor::stack(&rows, 0)?.unsqueeze(0)?)
    }

    fn generate_sequential(
        &self,
        state: &ChatState,
        max_new_tokens: usize,
        stop_on_audio_end: bool,
        max_audio_frames: Option<usize>,
        text_temperature: Option<f32>,
        text_top_k: Option<usize>,
        audio_temperature: Option<f32>,
        audio_top_k: Option<usize>,
        rng: &mut SimpleRng,
        on_text: &mut dyn FnMut(u32),
        on_audio: &mut dyn FnMut(&[u32]),
    ) -> Result<()> {
        let mut in_emb = self.build_prefill_embeddings(state)?;
        let mut current_modality = LfmModality::Text;
        let mut emitted_audio_frames = 0usize;
        let mut cache = LfmCache::new(self.lfm.config());

        for _ in 0..max_new_tokens {
            let out = self.lfm.forward_embeds_cached(&in_emb, &mut cache)?;
            let last = out.i((0, out.dim(1)? - 1, ..))?;

            match current_modality {
                LfmModality::Text => {
                    let logits = last
                        .reshape((1, last.dim(0)?))?
                        .matmul(&self.lfm.embed_tokens_weight().t()?)?
                        .squeeze(0)?;
                    let token = sample_token(
                        &logits,
                        text_temperature.unwrap_or(0.0) <= 0.0 || text_top_k == Some(1),
                        text_temperature,
                        text_top_k,
                        rng,
                    )?;

                    if token == self.tokenizer.specials().im_end {
                        break;
                    }

                    on_text(token);

                    if token == self.tokenizer.specials().audio_start {
                        current_modality = LfmModality::AudioOut;
                    }

                    in_emb = self.lfm.embed_tokens(token)?;
                }
                LfmModality::AudioOut => {
                    let frame = self.depthformer.sample_audio_frame(
                        &last,
                        audio_temperature,
                        audio_top_k,
                        rng,
                    )?;
                    let mut frame = frame;
                    if is_end_of_audio_frame(&frame) {
                        for t in &mut frame {
                            *t = END_OF_AUDIO_TOKEN;
                        }
                        if stop_on_audio_end {
                            break;
                        }
                        current_modality = LfmModality::Text;
                    }

                    on_audio(&frame);
                    emitted_audio_frames = emitted_audio_frames.saturating_add(1);
                    if max_audio_frames
                        .map(|limit| emitted_audio_frames >= limit)
                        .unwrap_or(false)
                    {
                        break;
                    }
                    let frame_t = Tensor::from_vec(
                        frame,
                        self.cfg.codebooks,
                        self.lfm.embed_tokens_weight().device(),
                    )?
                    .to_dtype(DType::U32)?;
                    in_emb = self.depthformer.audio_embedding_sum(&frame_t)?;
                }
                LfmModality::AudioIn => {}
            }
        }

        Ok(())
    }

    fn generate_interleaved(
        &self,
        state: &ChatState,
        max_new_tokens: usize,
        text_temperature: Option<f32>,
        text_top_k: Option<usize>,
        audio_temperature: Option<f32>,
        audio_top_k: Option<usize>,
        rng: &mut SimpleRng,
        on_text: &mut dyn FnMut(u32),
        on_audio: &mut dyn FnMut(&[u32]),
    ) -> Result<()> {
        let mut in_emb = self.build_prefill_embeddings(state)?;
        let mut current_modality = LfmModality::Text;
        let mut modality_left = self.cfg.interleaved_n_text;
        let mut text_done = false;
        let mut cache = LfmCache::new(self.lfm.config());

        for _ in 0..max_new_tokens {
            modality_left = modality_left.saturating_sub(1);

            let out = self.lfm.forward_embeds_cached(&in_emb, &mut cache)?;
            let last = out.i((0, out.dim(1)? - 1, ..))?;

            match current_modality {
                LfmModality::Text => {
                    let logits = last
                        .reshape((1, last.dim(0)?))?
                        .matmul(&self.lfm.embed_tokens_weight().t()?)?
                        .squeeze(0)?;
                    let token = sample_token(
                        &logits,
                        text_temperature.unwrap_or(0.0) <= 0.0 || text_top_k == Some(1),
                        text_temperature,
                        text_top_k,
                        rng,
                    )?;

                    if token == self.tokenizer.specials().im_end {
                        break;
                    }

                    on_text(token);

                    if token == self.tokenizer.specials().text_end {
                        text_done = true;
                    }

                    if modality_left == 0 || text_done {
                        current_modality = LfmModality::AudioOut;
                        modality_left = self.cfg.interleaved_n_audio;
                    }

                    in_emb = self.lfm.embed_tokens(token)?;
                }
                LfmModality::AudioOut => {
                    let mut frame = self.depthformer.sample_audio_frame(
                        &last,
                        audio_temperature,
                        audio_top_k,
                        rng,
                    )?;

                    if modality_left == 0 && !text_done {
                        current_modality = LfmModality::Text;
                        modality_left = self.cfg.interleaved_n_text;
                    }

                    if is_end_of_audio_frame(&frame) {
                        for t in &mut frame {
                            *t = END_OF_AUDIO_TOKEN;
                        }
                        current_modality = LfmModality::Text;
                        modality_left = self.cfg.interleaved_n_text;
                    }

                    on_audio(&frame);

                    let frame_t = Tensor::from_vec(
                        frame,
                        self.cfg.codebooks,
                        self.lfm.embed_tokens_weight().device(),
                    )?
                    .to_dtype(DType::U32)?;
                    in_emb = self.depthformer.audio_embedding_sum(&frame_t)?;
                }
                LfmModality::AudioIn => {}
            }
        }

        Ok(())
    }
}

pub fn lfm2_tts_voice_prompt(speaker: Option<&str>) -> &'static str {
    let normalized = speaker
        .unwrap_or("")
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_lowercase())
        .collect::<String>();

    if normalized.contains("ukmale")
        || normalized == "dylan"
        || normalized == "unclefu"
        || normalized == "ukm"
    {
        return TTS_UK_MALE_PROMPT;
    }

    if normalized.contains("ukfemale") || normalized == "vivian" {
        return TTS_UK_FEMALE_PROMPT;
    }

    if normalized.contains("usmale")
        || normalized == "ryan"
        || normalized == "aiden"
        || normalized == "eric"
        || normalized.contains("male")
    {
        return TTS_US_MALE_PROMPT;
    }

    if normalized.contains("usfemale")
        || normalized == "serena"
        || normalized == "sohee"
        || normalized == "onoanna"
        || normalized == "anna"
    {
        return TTS_US_FEMALE_PROMPT;
    }

    TTS_US_FEMALE_PROMPT
}

fn normalize_language(language: Option<&str>) -> Option<String> {
    language
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn asr_system_prompt(language: Option<&str>) -> String {
    if let Some(language) = normalize_language(language) {
        format!("{ASR_SYSTEM_PROMPT} Transcribe in {language}.")
    } else {
        ASR_SYSTEM_PROMPT.to_string()
    }
}

fn s2s_system_prompt(system_prompt: Option<&str>, language: Option<&str>) -> String {
    let base = system_prompt.unwrap_or(LFM2_DEFAULT_S2S_PROMPT).trim();
    if let Some(language) = normalize_language(language) {
        if base.is_empty() {
            format!("Respond in {language}.")
        } else {
            format!("{base} Respond in {language}.")
        }
    } else {
        base.to_string()
    }
}

fn validate_model_dir(model_dir: &Path) -> Result<()> {
    let required_files = ["config.json", "model.safetensors", "tokenizer.json"];
    for file in &required_files {
        let path = model_dir.join(file);
        if !path.exists() {
            return Err(Error::ModelLoadError(format!(
                "LFM2 model is missing required file: {}",
                path.display()
            )));
        }
    }

    let detok_cfg = model_dir.join(DETOKENIZER_CONFIG_FILE);
    let detok_model = model_dir.join(DETOKENIZER_MODEL_FILE);
    let has_detokenizer = detok_cfg.exists() && detok_model.exists();

    let mimi = model_dir.join(MIMI_TOKENIZER_CHECKPOINT);
    let has_mimi = mimi.exists();

    if !has_detokenizer && !has_mimi {
        return Err(Error::ModelLoadError(format!(
            "LFM2 model is missing audio decoder weights: expected either [{} + {}] or {}",
            detok_cfg.display(),
            detok_model.display(),
            mimi.display()
        )));
    }

    Ok(())
}

fn env_flag_enabled(key: &str) -> bool {
    std::env::var(key)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn is_end_of_audio_frame(frame: &[u32]) -> bool {
    frame.first().copied() == Some(END_OF_AUDIO_TOKEN)
}

fn trim_audio_frames(codebooks: &mut [Vec<u32>]) {
    if codebooks.is_empty() || codebooks[0].is_empty() {
        return;
    }

    let frames = codebooks[0].len();
    let mut cut_at = frames;
    for t in 0..frames {
        if codebooks[0].get(t).copied() == Some(END_OF_AUDIO_TOKEN) {
            cut_at = t;
            break;
        }
    }

    if cut_at < frames {
        for codes in codebooks {
            codes.truncate(cut_at);
        }
    }
}

fn text_delta(previous: &str, current: &str) -> String {
    if let Some(delta) = current.strip_prefix(previous) {
        return delta.to_string();
    }

    let common = previous
        .chars()
        .zip(current.chars())
        .take_while(|(a, b)| a == b)
        .count();
    current.chars().skip(common).collect()
}

pub(crate) fn sample_token(
    logits: &Tensor,
    greedy: bool,
    temperature: Option<f32>,
    top_k: Option<usize>,
    rng: &mut SimpleRng,
) -> Result<u32> {
    let values = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;

    if greedy {
        return argmax_from_slice(&values)
            .map(|idx| idx as u32)
            .ok_or_else(|| Error::InferenceError("Empty logits".to_string()));
    }

    let temp = temperature.unwrap_or(1.0).max(1e-5);
    let mut candidates: Vec<(usize, f32)> = values
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| {
            if v.is_finite() {
                Some((i, v / temp))
            } else {
                None
            }
        })
        .collect();

    if candidates.is_empty() {
        return Err(Error::InferenceError(
            "No valid sampling candidates".to_string(),
        ));
    }

    if let Some(k) = top_k {
        if k > 0 && k < candidates.len() {
            candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            candidates.truncate(k);
        }
    }

    let max_logit = candidates
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);

    let mut probs: Vec<(usize, f32)> = candidates
        .iter()
        .map(|(idx, v)| (*idx, (*v - max_logit).exp()))
        .collect();

    let sum: f32 = probs.iter().map(|(_, p)| *p).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return argmax_from_slice(&values)
            .map(|idx| idx as u32)
            .ok_or_else(|| Error::InferenceError("Empty logits".to_string()));
    }
    for (_, p) in &mut probs {
        *p /= sum;
    }

    let r = rng.next_f32();
    let mut acc = 0.0f32;
    for (idx, p) in probs {
        acc += p;
        if r <= acc {
            return Ok(idx as u32);
        }
    }

    argmax_from_slice(&values)
        .map(|idx| idx as u32)
        .ok_or_else(|| Error::InferenceError("Sampling fallback failed".to_string()))
}

fn argmax_from_slice(values: &[f32]) -> Option<usize> {
    let mut best = None;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best = Some(i);
            best_val = v;
        }
    }
    best
}

pub(crate) struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x1234_5678_9abc_def0);
        Self { state: nanos }
    }

    fn next_u32(&mut self) -> u32 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        ((x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) & 0xffff_ffff) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    static TEST_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn maps_known_speakers_to_expected_prompts() {
        assert_eq!(lfm2_tts_voice_prompt(Some("Ryan")), TTS_US_MALE_PROMPT);
        assert_eq!(lfm2_tts_voice_prompt(Some("Serena")), TTS_US_FEMALE_PROMPT);
        assert_eq!(lfm2_tts_voice_prompt(Some("Dylan")), TTS_UK_MALE_PROMPT);
        assert_eq!(lfm2_tts_voice_prompt(Some("Vivian")), TTS_UK_FEMALE_PROMPT);
    }

    #[test]
    fn asr_prompt_includes_language_hint_when_provided() {
        let prompt = asr_system_prompt(Some("fr"));
        assert_eq!(prompt, "Perform ASR. Transcribe in fr.");
    }

    #[test]
    fn s2s_prompt_appends_language_hint() {
        let prompt = s2s_system_prompt(Some("Respond helpfully."), Some("es"));
        assert_eq!(prompt, "Respond helpfully. Respond in es.");
    }

    #[test]
    fn detects_end_of_audio_when_first_codebook_marks_end() {
        assert!(is_end_of_audio_frame(&[END_OF_AUDIO_TOKEN, 2, 3]));
        assert!(!is_end_of_audio_frame(&[1, END_OF_AUDIO_TOKEN, 3]));
        assert!(!is_end_of_audio_frame(&[1, 2, 3]));
    }

    #[test]
    fn trims_audio_frames_using_first_codebook_end_marker() {
        let mut codebooks = vec![
            vec![10, 11, END_OF_AUDIO_TOKEN, 13],
            vec![20, 21, 22, 23],
            vec![30, END_OF_AUDIO_TOKEN, 32, 33],
        ];

        trim_audio_frames(&mut codebooks);

        assert_eq!(codebooks[0], vec![10, 11]);
        assert_eq!(codebooks[1], vec![20, 21]);
        assert_eq!(codebooks[2], vec![30, END_OF_AUDIO_TOKEN]);
    }

    #[test]
    fn validate_model_dir_accepts_detokenizer_without_mimi() {
        let dir = make_temp_dir("lfm2_detok_only");
        write_empty_file(dir.join("config.json"));
        write_empty_file(dir.join("model.safetensors"));
        write_empty_file(dir.join("tokenizer.json"));
        write_empty_file(dir.join("audio_detokenizer/config.json"));
        write_empty_file(dir.join("audio_detokenizer/model.safetensors"));

        let result = validate_model_dir(&dir);
        cleanup_temp_dir(&dir);

        assert!(
            result.is_ok(),
            "expected detokenizer-only model dir to be valid"
        );
    }

    #[test]
    fn validate_model_dir_rejects_missing_audio_decoder_assets() {
        let dir = make_temp_dir("lfm2_missing_decoder");
        write_empty_file(dir.join("config.json"));
        write_empty_file(dir.join("model.safetensors"));
        write_empty_file(dir.join("tokenizer.json"));

        let result = validate_model_dir(&dir);
        cleanup_temp_dir(&dir);

        assert!(result.is_err(), "expected missing decoder assets to fail");
    }

    fn make_temp_dir(prefix: &str) -> PathBuf {
        let unique = TEST_DIR_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
        let path = std::env::temp_dir().join(format!("{prefix}_{}_{}", std::process::id(), unique));
        fs::create_dir_all(&path).expect("create temp test dir");
        path
    }

    fn write_empty_file(path: PathBuf) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent dir");
        }
        fs::write(path, []).expect("write test file");
    }

    fn cleanup_temp_dir(path: &PathBuf) {
        let _ = fs::remove_dir_all(path);
    }
}
