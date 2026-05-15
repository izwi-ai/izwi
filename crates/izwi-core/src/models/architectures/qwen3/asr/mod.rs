//! Native Qwen speech-family loader and inference shared with Qwen3-ForcedAligner.

mod audio;
mod config;
mod tokenizer;

use std::path::{Path, PathBuf};
use std::time::Instant;
use std::{fs, io::Read};

use candle_core::quantized::gguf_file::Value as GgufValue;
use candle_core::{DType, IndexOp, Tensor, D};
use candle_nn::VarBuilder;
use serde::Deserialize;
use serde_json::json;
use tracing::{debug, info, warn};

use crate::audio::{MelConfig, MelSpectrogram};
use crate::backends::{backend_kind_for_device, DTypeSelectionRequest, DeviceKind, DeviceProfile};
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::kernels::buffer_pool::maybe_init_global_buffer_pool;
use crate::model::ModelVariant;
use crate::models::architectures::qwen3::core::{
    Qwen3Cache, Qwen3Config, Qwen3Model, RopeScalingConfig,
};
use crate::models::shared::attention::paged::default_kv_page_size;
use crate::models::shared::memory::metal::metal_pool_for_device;
use crate::models::shared::weights::gguf::{var_builder_from_gguf, GgufLoader};

use audio::AudioTower;
use config::Qwen3AsrConfig;
use tokenizer::{AsrTokenizer, SpecialTokenIds};

#[derive(Debug, Deserialize)]
struct PreprocessorConfig {
    #[serde(default)]
    feature_size: usize,
    #[serde(default)]
    n_fft: usize,
    #[serde(default)]
    hop_length: usize,
    #[serde(default)]
    n_samples: usize,
    #[serde(default)]
    nb_max_frames: usize,
}

pub struct Qwen3AsrModel {
    device: DeviceProfile,
    audio_dtype: DType,
    text_dtype: DType,
    is_forced_aligner: bool,
    timestamp_token_id: Option<u32>,
    timestamp_segment_time_ms: Option<u32>,
    tokenizer: AsrTokenizer,
    specials: SpecialTokenIds,
    audio_tower: AudioTower,
    text_model: Qwen3Model,
    mel: MelSpectrogram,
    preprocessor: PreprocessorConfig,
}

pub struct AsrDecodeState {
    cache: Qwen3Cache,
    embeds: Tensor,
    pos: usize,
    language: Option<String>,
    generated_ids: Vec<u32>,
    visible_generated_ids: Vec<u32>,
    decoded_visible_tokens: usize,
    tokens_since_resync: usize,
    assembled: String,
    diagnostics: Qwen3AsrDiagnostics,
    stop_tokens: Vec<u32>,
    max_new_tokens: usize,
    finished: bool,
}

#[derive(Debug, Clone)]
pub struct AsrDecodeStep {
    pub delta: String,
    pub text: String,
    pub tokens_generated: usize,
    pub finished: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsrTranscriptionOutput {
    pub text: String,
    pub language: Option<String>,
    pub diagnostics: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Default)]
struct Qwen3AsrTimingDiagnostics {
    resample_ms: f64,
    mel_ms: f64,
    audio_encode_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    total_ms: f64,
}

#[derive(Debug, Clone, Default)]
struct Qwen3AsrDiagnostics {
    input_sample_rate: u32,
    input_samples: usize,
    resampled_sample_rate: u32,
    resampled_samples: usize,
    mel_frames: usize,
    mel_frames_before_drop: usize,
    mel_frames_before_truncate: usize,
    mel_last_frame_dropped: bool,
    mel_frames_truncated: bool,
    audio_tokens: usize,
    prompt_tokens: usize,
    system_prompt_requested: bool,
    system_prompt_token_count: usize,
    forced_language: Option<String>,
    forced_language_token_count: usize,
    asr_text_token_available: bool,
    asr_text_token_appended: bool,
    max_new_tokens: usize,
    generated_tokens: usize,
    timings: Qwen3AsrTimingDiagnostics,
}

impl Qwen3AsrDiagnostics {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "model_family": "qwen3_asr",
            "audio": {
                "input_sample_rate": self.input_sample_rate,
                "input_samples": self.input_samples,
                "resampled_sample_rate": self.resampled_sample_rate,
                "resampled_samples": self.resampled_samples,
                "mel_frames": self.mel_frames,
                "mel_frames_before_drop": self.mel_frames_before_drop,
                "mel_frames_before_truncate": self.mel_frames_before_truncate,
                "mel_last_frame_dropped": self.mel_last_frame_dropped,
                "mel_frames_truncated": self.mel_frames_truncated,
                "audio_tokens": self.audio_tokens,
            },
            "prompt": {
                "prompt_tokens": self.prompt_tokens,
                "system_prompt_requested": self.system_prompt_requested,
                "system_prompt_token_count": self.system_prompt_token_count,
                "forced_language": self.forced_language,
                "forced_language_token_count": self.forced_language_token_count,
                "asr_text_token_available": self.asr_text_token_available,
                "asr_text_token_appended": self.asr_text_token_appended,
            },
            "decode": {
                "max_new_tokens": self.max_new_tokens,
                "generated_tokens": self.generated_tokens,
            },
            "timings_ms": {
                "resample": self.timings.resample_ms,
                "mel": self.timings.mel_ms,
                "audio_encode": self.timings.audio_encode_ms,
                "prefill": self.timings.prefill_ms,
                "decode": self.timings.decode_ms,
                "model_total": self.timings.total_ms,
            }
        })
    }
}

const DEFAULT_TRANSCRIBE_MAX_NEW_TOKENS: usize = 512;
const ASR_INCREMENTAL_DECODE_RESYNC_TOKENS: usize = 32;

fn parse_env_positive_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn qwen3_asr_kv_page_size(device: &candle_core::Device, prompt_tokens: usize) -> usize {
    let default_page_size = default_kv_page_size().max(1);
    let global_override = std::env::var_os("IZWI_KV_PAGE_SIZE").is_some();
    let asr_override = parse_env_positive_usize("IZWI_QWEN3_ASR_KV_PAGE_SIZE");
    qwen3_asr_kv_page_size_policy(
        default_page_size,
        device.is_metal(),
        prompt_tokens,
        global_override,
        asr_override,
    )
}

fn qwen3_asr_kv_page_size_policy(
    default_page_size: usize,
    is_metal: bool,
    prompt_tokens: usize,
    global_override: bool,
    asr_override: Option<usize>,
) -> usize {
    if let Some(override_page_size) = asr_override {
        return override_page_size.max(1);
    }
    if global_override || !is_metal {
        return default_page_size.max(1);
    }

    // Long ASR prompts are heavily page-bound in decode; larger pages reduce
    // per-token page-loop/kernel-launch overhead on Metal.
    if prompt_tokens >= 4096 {
        return default_page_size.max(128);
    }
    default_page_size.max(1)
}

impl Qwen3AsrModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        let gguf_path = resolve_qwen_asr_gguf_path(model_dir, variant);
        let gguf_loader = if let Some(path) = gguf_path.as_ref() {
            Some(GgufLoader::from_path_with_backend(
                path,
                backend_kind_for_device(&device.device),
            )?)
        } else {
            None
        };
        let config = if let Some(loader) = gguf_loader.as_ref() {
            parse_qwen3_asr_config_from_gguf(loader)?
        } else {
            let config_path = model_dir.join("config.json");
            let config_str = fs::read_to_string(config_path)?;
            serde_json::from_str(&config_str)?
        };
        let timestamp_token_id = config.timestamp_token_id;
        let timestamp_segment_time_ms = config.timestamp_segment_time.map(|v| v as u32);
        let is_forced_aligner = config
            .thinker_config
            .model_type
            .as_deref()
            .map(|name| name.to_ascii_lowercase().contains("forced_aligner"))
            .unwrap_or(false)
            || config.thinker_config.classify_num.is_some();
        let model_family = if is_forced_aligner {
            ModelFamily::Qwen3ForcedAligner
        } else {
            ModelFamily::Qwen3Asr
        };

        let mut text_cfg = config.thinker_config.text_config.clone();
        let inferred_lm_head_size = if let Some(gguf_path) = gguf_path.as_ref() {
            infer_lm_head_size_from_gguf(gguf_path, Some("thinker.lm_head.weight"))?.or(
                infer_lm_head_size_from_gguf(gguf_path, Some("lm_head.weight"))?,
            )
        } else {
            infer_lm_head_size_from_checkpoint(model_dir, Some("thinker.lm_head.weight"))?.or(
                infer_lm_head_size_from_checkpoint(model_dir, Some("lm_head.weight"))?,
            )
        };
        if let Some(inferred_lm_head_size) = inferred_lm_head_size {
            if inferred_lm_head_size != text_cfg.vocab_size {
                debug!(
                    "Overriding Qwen speech-family lm_head output size from {} to {}",
                    text_cfg.vocab_size, inferred_lm_head_size
                );
                text_cfg.lm_head_size = Some(inferred_lm_head_size);
            }
        } else if let Some(classify_num) = config.thinker_config.classify_num {
            if classify_num != text_cfg.vocab_size {
                text_cfg.lm_head_size = Some(classify_num);
            }
        }

        let tokenizer = if let Some(loader) = gguf_loader.as_ref() {
            AsrTokenizer::load_from_gguf(model_dir, loader, text_cfg.vocab_size)?
        } else {
            AsrTokenizer::load(model_dir, text_cfg.vocab_size)?
        };
        let specials = tokenizer.specials().clone();

        let preprocessor: PreprocessorConfig = if let Some(loader) = gguf_loader.as_ref() {
            preprocessor_config_from_gguf(loader)?
        } else {
            let path = model_dir.join("preprocessor_config.json");
            let data = fs::read_to_string(path)?;
            serde_json::from_str(&data)?
        };

        let mel_cfg = MelConfig {
            sample_rate: 16_000,
            n_fft: preprocessor.n_fft.max(1),
            hop_length: preprocessor.hop_length.max(1),
            n_mels: preprocessor.feature_size.max(1),
            f_min: 0.0,
            f_max: 8_000.0,
            normalize: true,
        };
        let mel = MelSpectrogram::new(mel_cfg)?;

        // Quantized checkpoints are trained/evaluated in bf16 and can degrade
        // badly when forced through fp32 dequant paths. Audio conditioning is
        // especially sensitive to precision, so keep the audio tower in F32
        // by default and select the text dtype with backend-aware rules.
        let is_gguf = gguf_loader.is_some();
        let is_quantized =
            is_gguf || config.quantization.is_some() || config.quantization_config.is_some();
        if !is_gguf && is_quantized {
            validate_quantization_config(&config)?;
        }
        let text_dtype_override = std::env::var("IZWI_QWEN_ASR_TEXT_DTYPE")
            .ok()
            .or_else(|| std::env::var("IZWI_QWEN_DTYPE").ok())
            .as_deref()
            .map(str::trim)
            .filter(|raw| !raw.is_empty())
            .map(str::to_string);
        let text_dtype = if let Some(raw) = text_dtype_override.as_deref() {
            device.select_model_dtype_checked(model_family, Some(raw), "Qwen ASR text")?
        } else if is_quantized {
            let requested =
                parse_asr_dtype(config.thinker_config.dtype.as_deref()).unwrap_or(DType::BF16);
            let selected = select_quantized_text_dtype(&device, model_family, requested);
            debug!(
                "Qwen speech-family quantized dtype selection: requested={:?}, selected={:?} on {:?}",
                requested, selected, device.kind
            );
            selected
        } else if device.kind.is_metal() {
            // This Qwen speech-family text decode path can drift badly on Metal in fp16 on longer
            // utterances. Prefer f32 unless the user explicitly overrides it.
            DType::F32
        } else {
            device.select_model_dtype(model_family, None)
        };
        let audio_dtype_override = std::env::var("IZWI_QWEN_ASR_AUDIO_DTYPE").ok();
        let audio_dtype = if is_gguf {
            match audio_dtype_override.as_deref().map(str::trim) {
                Some(raw) if !raw.is_empty() => {
                    let requested = device.select_model_dtype_checked(
                        model_family,
                        Some(raw),
                        "Qwen ASR audio",
                    )?;
                    if requested != text_dtype {
                        warn!(
                            "Qwen ASR GGUF requested audio dtype {:?} differs from text dtype {:?}; loading a second GGUF VarBuilder",
                            requested, text_dtype
                        );
                    }
                    requested
                }
                _ => text_dtype,
            }
        } else {
            match audio_dtype_override.as_deref().map(str::trim) {
                Some(raw) if !raw.is_empty() => {
                    device.select_model_dtype_checked(model_family, Some(raw), "Qwen ASR audio")?
                }
                _ => DType::F32,
            }
        };

        // Pre-initialize shared scratch pools up front to keep first-use
        // allocation costs out of ASR prefill/decode hot loops.
        let _ = maybe_init_global_buffer_pool(&device.device);
        let _ = metal_pool_for_device(&device.device);

        let (vb_text, vb_audio) = if let Some(gguf_path) = gguf_path.as_ref() {
            let vb_text = var_builder_from_gguf(gguf_path, text_dtype, &device.device)?;
            if audio_dtype == text_dtype {
                (vb_text.clone(), vb_text)
            } else {
                let vb_audio = var_builder_from_gguf(gguf_path, audio_dtype, &device.device)?;
                (vb_text, vb_audio)
            }
        } else {
            load_safetensors_var_builders(model_dir, &device, text_dtype, audio_dtype)?
        };

        let has_thinker_prefix = vb_text.contains_tensor("thinker.audio_tower.conv2d1.weight");
        let vb_text = if has_thinker_prefix {
            vb_text.pp("thinker")
        } else {
            vb_text
        };
        let vb_audio = if has_thinker_prefix {
            vb_audio.pp("thinker")
        } else {
            vb_audio
        };

        let audio_cfg = config.thinker_config.audio_config.clone();
        let audio_tower = AudioTower::load(audio_cfg, vb_audio.pp("audio_tower"))?;
        let text_model = Qwen3Model::load(text_cfg, vb_text)?;

        info!(
            "Loaded Qwen speech-family model on {:?} (audio_dtype={:?}, text_dtype={:?})",
            device.kind, audio_dtype, text_dtype
        );

        Ok(Self {
            device,
            audio_dtype,
            text_dtype,
            is_forced_aligner,
            timestamp_token_id,
            timestamp_segment_time_ms,
            tokenizer,
            specials,
            audio_tower,
            text_model,
            mel,
            preprocessor,
        })
    }

    fn build_decode_cache(&self, prompt_tokens: usize) -> Qwen3Cache {
        let page_size = qwen3_asr_kv_page_size(&self.device.device, prompt_tokens);
        Qwen3Cache::with_page_size_and_dense_decode(
            self.text_model.num_layers(),
            page_size,
            &self.device.device,
        )
    }

    pub fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<String> {
        self.transcribe_with_prompt(audio, sample_rate, language, None)
    }

    pub fn transcribe_with_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        system_prompt: Option<&str>,
    ) -> Result<String> {
        let mut no_op = |_delta: &str| {};
        Ok(self
            .transcribe_impl(audio, sample_rate, language, system_prompt, &mut no_op)?
            .text)
    }

    pub fn transcribe_with_details(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<AsrTranscriptionOutput> {
        self.transcribe_with_details_and_prompt(audio, sample_rate, language, None)
    }

    pub fn transcribe_with_details_and_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        system_prompt: Option<&str>,
    ) -> Result<AsrTranscriptionOutput> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_impl(audio, sample_rate, language, system_prompt, &mut no_op)
    }

    /// Upper-bound hint for chunk sizing in long-form ASR orchestration.
    pub fn max_audio_seconds_hint(&self) -> Option<f32> {
        let sample_rate = self.mel.config().sample_rate.max(1) as f32;
        if self.preprocessor.nb_max_frames > 0 {
            let hop = self.mel.config().hop_length.max(1) as f32;
            return Some(self.preprocessor.nb_max_frames as f32 * hop / sample_rate);
        }
        if self.preprocessor.n_samples > 0 {
            return Some(self.preprocessor.n_samples as f32 / sample_rate);
        }
        None
    }

    pub fn transcribe_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        self.transcribe_with_callback_and_prompt(audio, sample_rate, language, None, on_delta)
    }

    pub fn transcribe_with_callback_and_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        system_prompt: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        Ok(self
            .transcribe_impl(audio, sample_rate, language, system_prompt, on_delta)?
            .text)
    }

    fn transcribe_impl(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        system_prompt: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<AsrTranscriptionOutput> {
        if self.is_forced_aligner {
            return Err(Error::InvalidInput(
                "Qwen3-ForcedAligner models do not support transcription. Use forced alignment instead."
                    .to_string(),
            ));
        }
        let mut state = self.start_decode_with_prompt(
            audio,
            sample_rate,
            language,
            system_prompt,
            DEFAULT_TRANSCRIBE_MAX_NEW_TOKENS,
        )?;
        let decode_started = Instant::now();
        loop {
            let step = self.decode_step(&mut state)?;
            if !step.delta.is_empty() {
                for ch in step.delta.chars() {
                    let mut buf = [0u8; 4];
                    on_delta(ch.encode_utf8(&mut buf));
                }
            }
            if step.finished {
                state.diagnostics.timings.decode_ms = elapsed_ms(decode_started);
                state.diagnostics.timings.total_ms += state.diagnostics.timings.decode_ms;
                state.diagnostics.generated_tokens = step.tokens_generated;
                let (detected_language, text) = parse_asr_output(&step.text, language);
                return Ok(AsrTranscriptionOutput {
                    text,
                    language: detected_language,
                    diagnostics: Some(state.diagnostics.to_json()),
                });
            }
        }
    }

    pub fn start_decode(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        max_new_tokens: usize,
    ) -> Result<AsrDecodeState> {
        self.start_decode_with_prompt(audio, sample_rate, language, None, max_new_tokens)
    }

    pub fn start_decode_with_prompt(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        system_prompt: Option<&str>,
        max_new_tokens: usize,
    ) -> Result<AsrDecodeState> {
        if self.is_forced_aligner {
            return Err(Error::InvalidInput(
                "Qwen3-ForcedAligner models do not support transcription decode state.".to_string(),
            ));
        }
        let input_samples = audio.len();
        let total_started = Instant::now();
        let resample_started = Instant::now();
        let audio = if sample_rate != 16_000 {
            resample(audio, sample_rate, 16_000)?
        } else {
            audio.to_vec()
        };
        let resample_ms = elapsed_ms(resample_started);

        let mel_started = Instant::now();
        let mut mel_spec = self.mel.compute(&audio)?;
        let mel_frames_before_drop = mel_spec.len();
        let mel_last_frame_dropped =
            apply_qwen_mel_frame_policy(&mut mel_spec, qwen_asr_drop_last_mel_frame());
        let mel_frames_before_truncate = mel_spec.len();
        if self.preprocessor.nb_max_frames > 0 && mel_spec.len() > self.preprocessor.nb_max_frames {
            mel_spec.truncate(self.preprocessor.nb_max_frames);
        }
        let mel_ms = elapsed_ms(mel_started);

        let n_mels = self.mel.config().n_mels;
        if mel_spec.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let frames = mel_spec.len();
        let mut flat = Vec::with_capacity(frames * n_mels);
        for frame in &mel_spec {
            flat.extend_from_slice(frame);
        }

        let mel = Tensor::from_vec(flat, (frames, n_mels), &self.device.device)?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .to_dtype(self.audio_dtype)?;

        let feature_lens = vec![frames];
        let audio_started = Instant::now();
        let mut audio_embeds = self.audio_tower.forward(&mel, Some(&feature_lens))?;
        if audio_embeds.dtype() != self.text_dtype {
            audio_embeds = audio_embeds.to_dtype(self.text_dtype)?;
        }
        let audio_encode_ms = elapsed_ms(audio_started);
        let audio_len = audio_embeds.dim(1)?;

        let prompt = self.build_prompt(audio_len, language, system_prompt)?;
        let input_ids = Tensor::from_vec(
            prompt.ids.clone(),
            (1, prompt.ids.len()),
            &self.device.device,
        )?;

        let mut cache = self.build_decode_cache(prompt.ids.len());
        let prefill_started = Instant::now();
        let embeds = self.forward_with_audio(
            &input_ids,
            &audio_embeds,
            prompt.audio_pad_start,
            prompt.audio_pad_len,
            &mut cache,
        )?;
        let prefill_ms = elapsed_ms(prefill_started);
        let pos = embeds.dim(1)?;
        let max_new_tokens = max_new_tokens.max(1);
        let diagnostics = Qwen3AsrDiagnostics {
            input_sample_rate: sample_rate,
            input_samples,
            resampled_sample_rate: 16_000,
            resampled_samples: audio.len(),
            mel_frames: frames,
            mel_frames_before_drop,
            mel_frames_before_truncate,
            mel_last_frame_dropped,
            mel_frames_truncated: mel_frames_before_truncate > frames,
            audio_tokens: audio_len,
            prompt_tokens: prompt.ids.len(),
            system_prompt_requested: prompt.system_prompt_requested,
            system_prompt_token_count: prompt.system_prompt_token_count,
            forced_language: prompt.forced_language,
            forced_language_token_count: prompt.forced_language_token_count,
            asr_text_token_available: prompt.asr_text_token_available,
            asr_text_token_appended: prompt.asr_text_token_appended,
            max_new_tokens,
            generated_tokens: 0,
            timings: Qwen3AsrTimingDiagnostics {
                resample_ms,
                mel_ms,
                audio_encode_ms,
                prefill_ms,
                decode_ms: 0.0,
                total_ms: elapsed_ms(total_started),
            },
        };

        Ok(AsrDecodeState {
            cache,
            embeds,
            pos,
            language: language.map(ToString::to_string),
            generated_ids: Vec::new(),
            visible_generated_ids: Vec::new(),
            decoded_visible_tokens: 0,
            tokens_since_resync: 0,
            assembled: String::new(),
            diagnostics,
            stop_tokens: collect_stop_token_ids(&self.specials),
            max_new_tokens,
            finished: false,
        })
    }

    fn finalized_text(&self, state: &AsrDecodeState) -> String {
        let (_, text) = parse_asr_output(&state.assembled, state.language.as_deref());
        text
    }

    fn refresh_full_decoded_text(&self, state: &mut AsrDecodeState) -> Result<()> {
        state.assembled = self.decode_generated_untrimmed(&state.generated_ids)?;
        state.visible_generated_ids = state
            .generated_ids
            .iter()
            .copied()
            .filter(|id| !is_special_generation_token(&self.specials, *id))
            .collect();
        state.decoded_visible_tokens = state.visible_generated_ids.len();
        state.tokens_since_resync = 0;
        Ok(())
    }

    fn decode_incremental_delta(&self, state: &mut AsrDecodeState) -> Result<String> {
        if state.visible_generated_ids.len() <= state.decoded_visible_tokens {
            return Ok(String::new());
        }

        let new_ids = &state.visible_generated_ids[state.decoded_visible_tokens..];
        let mut delta = self.tokenizer.decode_text(new_ids)?;
        state.decoded_visible_tokens = state.visible_generated_ids.len();
        state.tokens_since_resync = state.tokens_since_resync.saturating_add(new_ids.len());

        if state.tokens_since_resync >= ASR_INCREMENTAL_DECODE_RESYNC_TOKENS {
            let full = self.tokenizer.decode_text(&state.visible_generated_ids)?;
            delta = text_delta(&state.assembled, &full);
            state.assembled = full;
            state.tokens_since_resync = 0;
            return Ok(delta);
        }

        state.assembled.push_str(&delta);
        Ok(delta)
    }

    pub fn decode_step(&self, state: &mut AsrDecodeState) -> Result<AsrDecodeStep> {
        if state.finished || state.generated_ids.len() >= state.max_new_tokens {
            state.finished = true;
            self.refresh_full_decoded_text(state)?;
            return Ok(AsrDecodeStep {
                delta: String::new(),
                text: self.finalized_text(state),
                tokens_generated: state.generated_ids.len(),
                finished: true,
            });
        }

        let logits = state.embeds.i((0, state.embeds.dim(1)? - 1))?;
        let next = argmax(&logits)?;
        if state.stop_tokens.contains(&next) {
            state.finished = true;
            self.refresh_full_decoded_text(state)?;
            return Ok(AsrDecodeStep {
                delta: String::new(),
                text: self.finalized_text(state),
                tokens_generated: state.generated_ids.len(),
                finished: true,
            });
        }

        state.generated_ids.push(next);
        if !is_special_generation_token(&self.specials, next) {
            state.visible_generated_ids.push(next);
        }
        let delta = self.decode_incremental_delta(state)?;

        let next_tensor = Tensor::from_vec(vec![next], (1, 1), &self.device.device)?;
        state.embeds = self
            .text_model
            .forward(&next_tensor, state.pos, Some(&mut state.cache))?;
        state.pos += 1;

        if state.generated_ids.len() >= state.max_new_tokens {
            state.finished = true;
            self.refresh_full_decoded_text(state)?;
        }

        Ok(AsrDecodeStep {
            delta,
            text: if state.finished {
                self.finalized_text(state)
            } else {
                state.assembled.trim().to_string()
            },
            tokens_generated: state.generated_ids.len(),
            finished: state.finished,
        })
    }

    /// Forced alignment: align reference text with audio timestamps.
    /// Returns a vector of (word, start_time_ms, end_time_ms) tuples.
    pub fn force_align(
        &self,
        audio: &[f32],
        sample_rate: u32,
        reference_text: &str,
        _language: Option<&str>,
    ) -> Result<Vec<(String, u32, u32)>> {
        if self.is_forced_aligner {
            return self.force_align_with_nar_head(audio, sample_rate, reference_text);
        }

        let audio = if sample_rate != 16_000 {
            resample(audio, sample_rate, 16_000)?
        } else {
            audio.to_vec()
        };

        let mut mel_spec = self.mel.compute(&audio)?;
        if self.preprocessor.nb_max_frames > 0 && mel_spec.len() > self.preprocessor.nb_max_frames {
            mel_spec.truncate(self.preprocessor.nb_max_frames);
        }

        let n_mels = self.mel.config().n_mels;
        if mel_spec.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let frames = mel_spec.len();
        let mut flat = Vec::with_capacity(frames * n_mels);
        for frame in mel_spec.iter() {
            flat.extend_from_slice(frame);
        }

        let mel = Tensor::from_vec(flat, (frames, n_mels), &self.device.device)?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .unsqueeze(0)? // [1, 1, n_mels, frames]
            .to_dtype(self.audio_dtype)?;

        let feature_lens = vec![frames];
        let mut audio_embeds = self.audio_tower.forward(&mel, Some(&feature_lens))?;
        if audio_embeds.dtype() != self.text_dtype {
            audio_embeds = audio_embeds.to_dtype(self.text_dtype)?;
        }
        let audio_len = audio_embeds.dim(1)?;

        // Build alignment prompt with reference text
        let prompt = self.build_alignment_prompt(audio_len, reference_text)?;
        let input_ids = Tensor::from_vec(
            prompt.ids.clone(),
            (1, prompt.ids.len()),
            &self.device.device,
        )?;

        let mut cache = self.build_decode_cache(prompt.ids.len());
        let mut embeds = self.forward_with_audio(
            &input_ids,
            &audio_embeds,
            prompt.audio_pad_start,
            prompt.audio_pad_len,
            &mut cache,
        )?;

        let mut pos = embeds.dim(1)?;

        let mut generated: Vec<u32> = Vec::new();

        let max_tokens = 2048usize;
        for _ in 0..max_tokens {
            let logits = embeds.i((0, embeds.dim(1)? - 1))?;
            let next = argmax(&logits)?;

            if next == self.specials.im_end || next == self.specials.eos {
                break;
            }
            generated.push(next);

            let next_tensor = Tensor::from_vec(vec![next], (1, 1), &self.device.device)?;
            embeds = self
                .text_model
                .forward(&next_tensor, pos, Some(&mut cache))?;
            pos += 1;
        }

        self.parse_alignment(&generated, reference_text, audio.len() as u32 / 16)
    }

    fn force_align_with_nar_head(
        &self,
        audio: &[f32],
        sample_rate: u32,
        reference_text: &str,
    ) -> Result<Vec<(String, u32, u32)>> {
        let audio = if sample_rate != 16_000 {
            resample(audio, sample_rate, 16_000)?
        } else {
            audio.to_vec()
        };

        let mut mel_spec = self.mel.compute(&audio)?;
        if self.preprocessor.nb_max_frames > 0 && mel_spec.len() > self.preprocessor.nb_max_frames {
            mel_spec.truncate(self.preprocessor.nb_max_frames);
        }

        let n_mels = self.mel.config().n_mels;
        if mel_spec.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let frames = mel_spec.len();
        let mut flat = Vec::with_capacity(frames * n_mels);
        for frame in mel_spec.iter() {
            flat.extend_from_slice(frame);
        }

        let mel = Tensor::from_vec(flat, (frames, n_mels), &self.device.device)?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .to_dtype(self.audio_dtype)?;

        let feature_lens = vec![frames];
        let mut audio_embeds = self.audio_tower.forward(&mel, Some(&feature_lens))?;
        if audio_embeds.dtype() != self.text_dtype {
            audio_embeds = audio_embeds.to_dtype(self.text_dtype)?;
        }
        let audio_len = audio_embeds.dim(1)?;

        let words = extract_alignment_words(reference_text);
        if words.is_empty() {
            return Err(Error::InvalidInput(
                "Reference text produced no alignable words".to_string(),
            ));
        }

        let prompt = self.build_forced_aligner_prompt(audio_len, &words)?;
        let timestamp_positions: Vec<usize> = prompt
            .ids
            .iter()
            .enumerate()
            .filter_map(|(idx, token_id)| (*token_id == prompt.timestamp_token_id).then_some(idx))
            .collect();
        if timestamp_positions.is_empty() {
            return Err(Error::InferenceError(
                "Forced aligner prompt contains no timestamp markers".to_string(),
            ));
        }

        let input_ids = Tensor::from_vec(
            prompt.ids.clone(),
            (1, prompt.ids.len()),
            &self.device.device,
        )?;

        let mut cache = self.build_decode_cache(prompt.ids.len());
        let logits = self.forward_with_audio(
            &input_ids,
            &audio_embeds,
            prompt.audio_pad_start,
            prompt.audio_pad_len,
            &mut cache,
        )?;

        let segment_time_ms = self.timestamp_segment_time_ms.unwrap_or(20).max(1);
        let mut timestamp_ms = Vec::with_capacity(timestamp_positions.len());
        for &position in &timestamp_positions {
            let token_logits = logits.i((0, position))?;
            let cls_idx = argmax(&token_logits)?;
            timestamp_ms.push(cls_idx.saturating_mul(segment_time_ms));
        }

        let mut fixed = fix_timestamp_sequence(&timestamp_ms);
        let required = words.len().saturating_mul(2);
        if fixed.len() < required {
            let fill = fixed.last().copied().unwrap_or(0);
            fixed.resize(required, fill);
        }

        let mut alignments = Vec::with_capacity(words.len());
        for (idx, word) in words.iter().enumerate() {
            let start = fixed.get(idx * 2).copied().unwrap_or(0);
            let mut end = fixed
                .get(idx * 2 + 1)
                .copied()
                .unwrap_or_else(|| start.saturating_add(segment_time_ms));
            if end <= start {
                end = start.saturating_add(1);
            }
            alignments.push((word.clone(), start, end));
        }

        let audio_duration_ms = ((audio.len() as f32 / 16_000.0) * 1000.0).round() as u32;
        normalize_alignment_bounds(&mut alignments, audio_duration_ms);
        if alignment_distribution_is_degenerate(&alignments, audio_duration_ms) {
            warn!(
                "Forced aligner produced degenerate timestamp distribution; falling back to interval-based alignment"
            );
            alignments = distribute_words_over_interval(&words, 0, audio_duration_ms.max(1));
            normalize_alignment_bounds(&mut alignments, audio_duration_ms);
        }
        Ok(alignments)
    }

    fn build_forced_aligner_prompt(
        &self,
        audio_len: usize,
        words: &[String],
    ) -> Result<ForcedAlignerPrompt> {
        let timestamp_token_id = self
            .specials
            .timestamp
            .or(self.timestamp_token_id)
            .ok_or_else(|| {
                Error::InvalidInput(
                    "Forced aligner model is missing timestamp token metadata".to_string(),
                )
            })?;

        let mut ids = Vec::new();
        ids.push(self.specials.audio_start);
        let audio_pad_start = ids.len();
        ids.extend(std::iter::repeat_n(self.specials.audio_token, audio_len));
        ids.push(self.specials.audio_end);

        for word in words {
            ids.extend(self.tokenizer.encode_text(word)?);
            ids.push(timestamp_token_id);
            ids.push(timestamp_token_id);
        }

        Ok(ForcedAlignerPrompt {
            ids,
            audio_pad_start,
            audio_pad_len: audio_len,
            timestamp_token_id,
        })
    }

    fn decode_generated_untrimmed(&self, tokens: &[u32]) -> Result<String> {
        let filtered: Vec<u32> = tokens
            .iter()
            .copied()
            .filter(|id| !is_special_generation_token(&self.specials, *id))
            .collect();
        let text = self.tokenizer.decode_text(&filtered)?;
        Ok(text)
    }

    fn forward_with_audio(
        &self,
        input_ids: &Tensor,
        audio_embeds: &Tensor,
        audio_pad_start: usize,
        audio_pad_len: usize,
        cache: &mut Qwen3Cache,
    ) -> Result<Tensor> {
        let embeds = self.text_model.embeddings(input_ids)?;
        let seq_len = embeds.dim(1)?;
        let model_audio_len = audio_embeds.dim(1)?;
        if audio_pad_len == 0 {
            return Err(Error::InvalidInput(
                "Audio placeholder length must be at least 1".to_string(),
            ));
        }
        if model_audio_len != audio_pad_len {
            return Err(Error::InvalidInput(format!(
                "Audio placeholder mismatch: prompt has {audio_pad_len}, embeddings have {model_audio_len}"
            )));
        }

        if audio_pad_start + audio_pad_len > seq_len {
            return Err(Error::InvalidInput(
                "Audio placeholder span is out of prompt bounds".to_string(),
            ));
        }

        // Replace the contiguous <|audio_pad|> span with projected audio embeddings.
        let before = if audio_pad_start > 0 {
            embeds.narrow(1, 0, audio_pad_start)?
        } else {
            Tensor::zeros((1, 0, embeds.dim(2)?), embeds.dtype(), embeds.device())?
        };

        let after_start = audio_pad_start + audio_pad_len;
        let after = if after_start < seq_len {
            embeds.narrow(1, after_start, seq_len - after_start)?
        } else {
            Tensor::zeros((1, 0, embeds.dim(2)?), embeds.dtype(), embeds.device())?
        };

        let embeds = Tensor::cat(&[before, audio_embeds.clone(), after], 1)?;

        let position_ids = if self.text_model.uses_mrope() {
            Some(self.build_position_ids(
                embeds.dim(1)?,
                0,
                Some((audio_pad_start, audio_pad_len)),
            )?)
        } else {
            None
        };
        self.text_model
            .forward_with_embeds(&embeds, 0, Some(cache), position_ids.as_ref())
    }

    fn build_prompt(
        &self,
        audio_len: usize,
        language: Option<&str>,
        system_prompt: Option<&str>,
    ) -> Result<PromptTokens> {
        build_qwen_asr_prompt_tokens(
            &self.specials,
            audio_len,
            language,
            system_prompt,
            |text| self.tokenizer.encode_text(text),
        )
    }

    fn build_alignment_prompt(
        &self,
        audio_len: usize,
        reference_text: &str,
    ) -> Result<PromptTokens> {
        let mut ids = Vec::new();

        ids.push(self.specials.im_start);
        ids.extend(self.tokenizer.encode_text("system\n")?);
        ids.push(self.specials.im_end);
        ids.extend(self.tokenizer.encode_text("\n")?);

        ids.push(self.specials.im_start);
        ids.extend(self.tokenizer.encode_text("user\n")?);

        ids.push(self.specials.audio_start);
        let audio_pad_start = ids.len();
        ids.extend(std::iter::repeat_n(self.specials.audio_token, audio_len));
        ids.push(self.specials.audio_end);
        ids.extend(
            self.tokenizer
                .encode_text(&format!("Reference: {}\n", reference_text))?,
        );

        ids.push(self.specials.im_end);
        ids.extend(self.tokenizer.encode_text("\n")?);

        ids.push(self.specials.im_start);
        ids.extend(self.tokenizer.encode_text("assistant\n")?);

        Ok(PromptTokens {
            ids,
            audio_pad_start,
            audio_pad_len: audio_len,
            system_prompt_requested: false,
            system_prompt_token_count: 0,
            forced_language: None,
            forced_language_token_count: 0,
            asr_text_token_available: self.specials.asr_text.is_some(),
            asr_text_token_appended: false,
        })
    }

    fn parse_alignment(
        &self,
        generated_ids: &[u32],
        reference_text: &str,
        audio_duration_ms: u32,
    ) -> Result<Vec<(String, u32, u32)>> {
        let mut alignments = self.parse_alignment_from_timestamp_tokens(generated_ids)?;

        if alignments.is_empty() {
            let decoded = self
                .tokenizer
                .decode_text_with_special_tokens(generated_ids)
                .unwrap_or_default();
            alignments = fallback_alignment_from_text(&decoded, audio_duration_ms);
        }

        if alignments.is_empty() {
            alignments = fallback_alignment_from_text(reference_text, audio_duration_ms);
        }

        if alignments.is_empty() {
            return Err(Error::InferenceError(
                "Forced alignment produced no aligned words".to_string(),
            ));
        }

        normalize_alignment_bounds(&mut alignments, audio_duration_ms);
        if alignment_distribution_is_degenerate(&alignments, audio_duration_ms) {
            warn!(
                "Parsed forced alignment timestamps looked degenerate; falling back to text interval distribution"
            );
            alignments = fallback_alignment_from_text(reference_text, audio_duration_ms.max(1));
            normalize_alignment_bounds(&mut alignments, audio_duration_ms);
        }
        Ok(alignments)
    }

    fn parse_alignment_from_timestamp_tokens(
        &self,
        generated_ids: &[u32],
    ) -> Result<Vec<(String, u32, u32)>> {
        let mut results = Vec::new();
        let mut text_ids = Vec::new();
        let mut last_ts_ms = 0u32;

        let segment_time_ms = self
            .timestamp_segment_time_ms
            .or_else(|| self.timestamp_token_id.map(|_| 20))
            .unwrap_or(20)
            .max(1);

        for token_id in generated_ids.iter().copied() {
            if let Some(timestamp_index) = self.tokenizer.timestamp_index_for_token(token_id) {
                let ts_ms = timestamp_index.saturating_mul(segment_time_ms);
                if !text_ids.is_empty() {
                    let chunk_text = self.tokenizer.decode_text(&text_ids)?;
                    let words = extract_alignment_words(&chunk_text);
                    results.extend(distribute_words_over_interval(
                        &words,
                        last_ts_ms,
                        ts_ms.max(last_ts_ms.saturating_add(1)),
                    ));
                    text_ids.clear();
                }
                last_ts_ms = ts_ms;
                continue;
            }

            if is_special_generation_token(&self.specials, token_id) {
                continue;
            }
            text_ids.push(token_id);
        }

        if !text_ids.is_empty() {
            let chunk_text = self.tokenizer.decode_text(&text_ids)?;
            let words = extract_alignment_words(&chunk_text);
            let default_end = last_ts_ms
                .saturating_add((words.len() as u32).saturating_mul(segment_time_ms.max(1)))
                .max(last_ts_ms.saturating_add(1));
            results.extend(distribute_words_over_interval(
                &words,
                last_ts_ms,
                default_end,
            ));
        }

        Ok(results)
    }

    fn build_position_ids(
        &self,
        seq_len: usize,
        start_pos: usize,
        audio_span: Option<(usize, usize)>,
    ) -> Result<Tensor> {
        let positions = build_mrope_positions(seq_len, start_pos, audio_span);

        let mut data = Vec::with_capacity(3 * seq_len);
        for _axis in 0..3 {
            data.extend_from_slice(&positions);
        }

        Tensor::from_vec(data, (3, seq_len), &self.device.device).map_err(Error::from)
    }
}

fn extract_alignment_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();

    let flush = |buffer: &mut String, out: &mut Vec<String>| {
        if buffer.is_empty() {
            return;
        }
        out.push(buffer.clone());
        buffer.clear();
    };

    for ch in text.chars() {
        if is_east_asian_char(ch) {
            flush(&mut current, &mut words);
            words.push(ch.to_string());
            continue;
        }

        if is_alignment_word_char(ch) {
            current.push(ch);
        } else {
            flush(&mut current, &mut words);
        }
    }
    flush(&mut current, &mut words);

    words
}

fn is_alignment_word_char(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '\'' || ch == '-'
}

fn is_east_asian_char(ch: char) -> bool {
    let code = ch as u32;
    (0x4E00..=0x9FFF).contains(&code)
        || (0x3400..=0x4DBF).contains(&code)
        || (0x20000..=0x2A6DF).contains(&code)
        || (0x2A700..=0x2B73F).contains(&code)
        || (0x2B740..=0x2B81F).contains(&code)
        || (0x2B820..=0x2CEAF).contains(&code)
        || (0xF900..=0xFAFF).contains(&code)
        || (0x3040..=0x309F).contains(&code) // Hiragana
        || (0x30A0..=0x30FF).contains(&code) // Katakana
        || (0xAC00..=0xD7AF).contains(&code) // Hangul syllables
        || (0x1100..=0x11FF).contains(&code) // Hangul Jamo
        || (0x3130..=0x318F).contains(&code) // Hangul Compatibility Jamo
}

fn distribute_words_over_interval(
    words: &[String],
    start_ms: u32,
    end_ms: u32,
) -> Vec<(String, u32, u32)> {
    if words.is_empty() {
        return Vec::new();
    }

    let start = start_ms.min(end_ms);
    let mut end = end_ms.max(start.saturating_add(1));
    let min_span = words.len() as u32;
    if end.saturating_sub(start) < min_span {
        end = start.saturating_add(min_span);
    }

    let span = end.saturating_sub(start).max(1);
    let step = span as f32 / words.len() as f32;

    words
        .iter()
        .enumerate()
        .map(|(idx, word)| {
            let ws = start.saturating_add((idx as f32 * step).floor() as u32);
            let mut we = if idx + 1 == words.len() {
                end
            } else {
                start.saturating_add(((idx + 1) as f32 * step).floor() as u32)
            };
            if we <= ws {
                we = ws.saturating_add(1);
            }
            (word.clone(), ws, we)
        })
        .collect()
}

fn fallback_alignment_from_text(text: &str, audio_duration_ms: u32) -> Vec<(String, u32, u32)> {
    let words = extract_alignment_words(text);
    distribute_words_over_interval(&words, 0, audio_duration_ms.max(1))
}

fn normalize_alignment_bounds(alignments: &mut [(String, u32, u32)], audio_duration_ms: u32) {
    if alignments.is_empty() {
        return;
    }

    let max_end = audio_duration_ms.max(1);
    let mut cursor = 0u32;

    for (_, start, end) in alignments.iter_mut() {
        let mut s = (*start).min(max_end.saturating_sub(1));
        let mut e = (*end).min(max_end);

        if s < cursor {
            s = cursor;
        }
        if e <= s {
            e = s.saturating_add(1).min(max_end);
        }
        if e <= s {
            // audio may be fully exhausted; preserve monotonic order with best effort.
            s = max_end.saturating_sub(1);
            e = max_end;
        }

        *start = s;
        *end = e;
        cursor = e;
    }
}

fn alignment_distribution_is_degenerate(
    alignments: &[(String, u32, u32)],
    audio_duration_ms: u32,
) -> bool {
    if alignments.is_empty() {
        return true;
    }
    if alignments.len() < 8 {
        return false;
    }

    let audio_duration_ms = audio_duration_ms.max(1);
    let tail_start = audio_duration_ms.saturating_sub(250);
    let mut min_start = u32::MAX;
    let mut max_end = 0u32;
    let mut tiny_spans = 0usize;
    let mut tail_heavy = 0usize;

    for (_, start, end) in alignments {
        min_start = min_start.min((*start).min(audio_duration_ms));
        max_end = max_end.max((*end).min(audio_duration_ms));
        if *end <= start.saturating_add(1) {
            tiny_spans += 1;
        }
        if *start >= tail_start {
            tail_heavy += 1;
        }
    }

    let span = max_end.saturating_sub(min_start);
    let len = alignments.len();
    tiny_spans * 10 >= len * 8
        || tail_heavy * 10 >= len * 8
        || span <= (audio_duration_ms / 20).max(1)
}

fn resolve_qwen_asr_gguf_path(model_dir: &Path, variant: ModelVariant) -> Option<PathBuf> {
    if let Some(filename) = qwen_asr_gguf_filename(variant) {
        let candidate = model_dir.join(filename);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    for candidate in [
        model_dir.join("qwen3_asr_0.6b_q8_0.gguf"),
        model_dir.join("qwen3_asr_1.7b_q8_0.gguf"),
    ] {
        if candidate.exists() {
            return Some(candidate);
        }
    }

    None
}

fn qwen_asr_gguf_filename(variant: ModelVariant) -> Option<&'static str> {
    match variant {
        ModelVariant::Qwen3Asr06BGguf => Some("qwen3_asr_0.6b_q8_0.gguf"),
        ModelVariant::Qwen3Asr17BGguf => Some("qwen3_asr_1.7b_q8_0.gguf"),
        _ => None,
    }
}

fn parse_qwen3_asr_config_from_gguf(loader: &GgufLoader) -> Result<Qwen3AsrConfig> {
    let rope_theta = required_f64_metadata(loader, "qwen3_asr.text.rope_theta")?;
    let mrope_section = required_usize_array_metadata(loader, "qwen3_asr.text.mrope_section")?;

    let text_config = Qwen3Config {
        hidden_size: required_usize_metadata(loader, "qwen3_asr.text.hidden_size")?,
        intermediate_size: required_usize_metadata(loader, "qwen3_asr.text.intermediate_size")?,
        num_attention_heads: required_usize_metadata(loader, "qwen3_asr.text.num_attention_heads")?,
        num_hidden_layers: required_usize_metadata(loader, "qwen3_asr.text.num_hidden_layers")?,
        num_key_value_heads: required_usize_metadata(loader, "qwen3_asr.text.num_key_value_heads")?,
        head_dim: Some(required_usize_metadata(loader, "qwen3_asr.text.head_dim")?),
        rms_norm_eps: required_f64_metadata(loader, "qwen3_asr.text.rms_norm_eps")?,
        rope_theta,
        vocab_size: required_usize_metadata(loader, "qwen3_asr.text.vocab_size")?,
        lm_head_size: None,
        rope_scaling: Some(RopeScalingConfig {
            rope_type: Some("mrope".to_string()),
            rope_theta: Some(rope_theta),
            mrope_interleaved: Some(true),
            interleaved: Some(true),
            mrope_section: Some(mrope_section),
        }),
    };

    let audio_config = config::AudioConfig {
        d_model: required_usize_metadata(loader, "qwen3_asr.audio.d_model")?,
        encoder_attention_heads: required_usize_metadata(
            loader,
            "qwen3_asr.audio.encoder_attention_heads",
        )?,
        encoder_ffn_dim: required_usize_metadata(loader, "qwen3_asr.audio.encoder_ffn_dim")?,
        encoder_layers: required_usize_metadata(loader, "qwen3_asr.audio.encoder_layers")?,
        num_mel_bins: required_usize_metadata(loader, "qwen3_asr.audio.num_mel_bins")?,
        downsample_hidden_size: infer_downsample_hidden_size(loader)?,
        output_dim: required_usize_metadata(loader, "qwen3_asr.audio.output_dim")?,
        conv_chunksize: optional_usize_metadata(loader, "qwen3_asr.audio.conv_chunksize"),
        n_window: optional_usize_metadata(loader, "qwen3_asr.audio.n_window"),
        n_window_infer: optional_usize_metadata(loader, "qwen3_asr.audio.n_window_infer"),
    };

    Ok(Qwen3AsrConfig {
        model_type: loader.get_metadata_string("general.architecture"),
        quantization: None,
        quantization_config: None,
        thinker_config: config::ThinkerConfig {
            model_type: loader.get_metadata_string("general.architecture"),
            classify_num: None,
            audio_config,
            text_config,
            audio_start_token_id: optional_u32_metadata(loader, "qwen3_asr.audio_start_token_id"),
            audio_end_token_id: optional_u32_metadata(loader, "qwen3_asr.audio_end_token_id"),
            audio_token_id: optional_u32_metadata(loader, "qwen3_asr.audio_token_id"),
            dtype: Some("bf16".to_string()),
        },
        support_languages: Vec::new(),
        timestamp_token_id: None,
        timestamp_segment_time: None,
    })
}

fn preprocessor_config_from_gguf(loader: &GgufLoader) -> Result<PreprocessorConfig> {
    let feature_size = required_usize_metadata(loader, "qwen3_asr.audio.num_mel_bins")?;
    let n_fft = optional_usize_metadata(loader, "qwen3_asr.audio.n_fft").unwrap_or(400);
    let hop_length = optional_usize_metadata(loader, "qwen3_asr.audio.hop_length").unwrap_or(160);
    let nb_max_frames = optional_usize_metadata(loader, "qwen3_asr.audio.max_source_positions")
        .map(|value| value.saturating_mul(8))
        .unwrap_or(0);
    let n_samples = if nb_max_frames > 0 {
        nb_max_frames.saturating_mul(hop_length)
    } else {
        0
    };

    Ok(PreprocessorConfig {
        feature_size,
        n_fft,
        hop_length,
        n_samples,
        nb_max_frames,
    })
}

fn infer_downsample_hidden_size(loader: &GgufLoader) -> Result<usize> {
    for tensor_name in [
        "thinker.audio_tower.conv2d1.weight",
        "audio_tower.conv2d1.weight",
    ] {
        if let Some(shape) = loader.tensor_shape(tensor_name) {
            if let Some(channels) = shape.first().copied() {
                return Ok(channels);
            }
        }
    }

    Err(Error::ModelLoadError(
        "Unable to infer qwen3_asr.audio.downsample_hidden_size from GGUF tensors".to_string(),
    ))
}

fn load_safetensors_var_builders(
    model_dir: &Path,
    device: &DeviceProfile,
    text_dtype: DType,
    audio_dtype: DType,
) -> Result<(VarBuilder<'static>, VarBuilder<'static>)> {
    let shard_paths = resolve_safetensors_shards(model_dir)?;
    let vb_text =
        unsafe { VarBuilder::from_mmaped_safetensors(&shard_paths, text_dtype, &device.device)? };
    let vb_audio =
        unsafe { VarBuilder::from_mmaped_safetensors(&shard_paths, audio_dtype, &device.device)? };
    Ok((vb_text, vb_audio))
}

fn resolve_safetensors_shards(model_dir: &Path) -> Result<Vec<PathBuf>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index_data = fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_data)?;
        let weight_map = index
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| {
                Error::InvalidInput("Invalid model.safetensors.index.json format".to_string())
            })?;

        let mut shard_files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        shard_files.sort();
        shard_files.dedup();
        let shard_paths: Vec<PathBuf> = shard_files.iter().map(|f| model_dir.join(f)).collect();
        info!(
            "Loading sharded ASR model with {} shard files",
            shard_paths.len()
        );
        return Ok(shard_paths);
    }

    let weights_path = model_dir.join("model.safetensors");
    if !weights_path.exists() {
        return Err(Error::ModelNotFound(format!(
            "Missing ASR checkpoint file: {}",
            weights_path.display()
        )));
    }
    Ok(vec![weights_path])
}

fn select_quantized_text_dtype(
    device: &DeviceProfile,
    model_family: ModelFamily,
    requested: DType,
) -> DType {
    match device.kind {
        DeviceKind::Metal => DType::F16,
        DeviceKind::Cpu => DType::F32,
        DeviceKind::Cuda => {
            device
                .resolve_dtype(
                    DTypeSelectionRequest::new(None)
                        .with_model_family(model_family)
                        .with_checkpoint_dtype(requested)
                        .with_quantized(true),
                )
                .dtype
        }
    }
}

fn infer_lm_head_size_from_gguf(
    gguf_path: &Path,
    tensor_name: Option<&str>,
) -> Result<Option<usize>> {
    let tensor_name = tensor_name.unwrap_or("lm_head.weight");
    let loader = GgufLoader::from_path(gguf_path)?;
    Ok(loader
        .tensor_shape(tensor_name)
        .and_then(|shape| shape.first().copied()))
}

fn required_usize_metadata(loader: &GgufLoader, key: &str) -> Result<usize> {
    loader
        .metadata_value(key)
        .and_then(gguf_to_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))
}

fn optional_usize_metadata(loader: &GgufLoader, key: &str) -> Option<usize> {
    loader
        .metadata_value(key)
        .and_then(gguf_to_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn optional_u32_metadata(loader: &GgufLoader, key: &str) -> Option<u32> {
    loader
        .metadata_value(key)
        .and_then(gguf_to_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn required_f64_metadata(loader: &GgufLoader, key: &str) -> Result<f64> {
    loader
        .metadata_value(key)
        .and_then(gguf_to_f64)
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))
}

fn required_usize_array_metadata(loader: &GgufLoader, key: &str) -> Result<Vec<usize>> {
    let value = loader
        .metadata_value(key)
        .ok_or_else(|| Error::ModelLoadError(format!("Missing or invalid GGUF metadata: {key}")))?;
    let GgufValue::Array(values) = value else {
        return Err(Error::ModelLoadError(format!(
            "Expected GGUF metadata array for {key}"
        )));
    };

    let mut out = Vec::with_capacity(values.len());
    for value in values {
        let raw = gguf_to_u64(value).ok_or_else(|| {
            Error::ModelLoadError(format!("Expected integer metadata values for {key}"))
        })?;
        let parsed = usize::try_from(raw).map_err(|_| {
            Error::ModelLoadError(format!("GGUF metadata value out of range for {key}: {raw}"))
        })?;
        out.push(parsed);
    }
    Ok(out)
}

fn gguf_to_u64(value: &GgufValue) -> Option<u64> {
    match value {
        GgufValue::U64(v) => Some(*v),
        GgufValue::I64(v) => Some(*v as u64),
        GgufValue::U32(v) => Some(*v as u64),
        GgufValue::I32(v) => Some(*v as u64),
        GgufValue::U16(v) => Some(*v as u64),
        GgufValue::I16(v) => Some(*v as u64),
        GgufValue::U8(v) => Some(*v as u64),
        GgufValue::I8(v) => Some(*v as u64),
        _ => None,
    }
}

fn gguf_to_f64(value: &GgufValue) -> Option<f64> {
    match value {
        GgufValue::F64(v) => Some(*v),
        GgufValue::F32(v) => Some(*v as f64),
        GgufValue::U64(v) => Some(*v as f64),
        GgufValue::I64(v) => Some(*v as f64),
        GgufValue::U32(v) => Some(*v as f64),
        GgufValue::I32(v) => Some(*v as f64),
        GgufValue::U16(v) => Some(*v as f64),
        GgufValue::I16(v) => Some(*v as f64),
        GgufValue::U8(v) => Some(*v as f64),
        GgufValue::I8(v) => Some(*v as f64),
        _ => None,
    }
}

const MAX_SAFE_TENSORS_HEADER_SIZE: usize = 100_000_000;

fn infer_lm_head_size_from_checkpoint(
    model_dir: &Path,
    tensor_name: Option<&str>,
) -> Result<Option<usize>> {
    let tensor_name = tensor_name.unwrap_or("lm_head.weight");
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index_data = fs::read_to_string(&index_path)?;
        let index: serde_json::Value = serde_json::from_str(&index_data)?;
        let weight_map = index
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| {
                Error::InvalidInput("Invalid model.safetensors.index.json format".to_string())
            })?;
        let Some(shard_name) = weight_map.get(tensor_name).and_then(|v| v.as_str()) else {
            return Ok(None);
        };
        let shard_path = model_dir.join(shard_name);
        return infer_tensor_first_dim_from_safetensors(&shard_path, tensor_name);
    }

    let weights_path = model_dir.join("model.safetensors");
    if !weights_path.exists() {
        return Ok(None);
    }
    infer_tensor_first_dim_from_safetensors(&weights_path, tensor_name)
}

fn infer_tensor_first_dim_from_safetensors(
    safetensors_path: &Path,
    tensor_name: &str,
) -> Result<Option<usize>> {
    let shape = match tensor_shape_from_safetensors_header(safetensors_path, tensor_name)? {
        Some(shape) => shape,
        None => return Ok(None),
    };
    Ok(shape.first().copied())
}

fn tensor_shape_from_safetensors_header(
    safetensors_path: &Path,
    tensor_name: &str,
) -> Result<Option<Vec<usize>>> {
    let mut file = fs::File::open(safetensors_path)?;

    let mut n_buf = [0u8; 8];
    file.read_exact(&mut n_buf)?;
    let header_len_u64 = u64::from_le_bytes(n_buf);
    let header_len: usize = header_len_u64
        .try_into()
        .map_err(|_| Error::InvalidInput("Invalid safetensors header length".to_string()))?;
    if header_len > MAX_SAFE_TENSORS_HEADER_SIZE {
        return Err(Error::InvalidInput(format!(
            "Safetensors header too large: {header_len}"
        )));
    }

    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;

    let metadata: serde_json::Value = serde_json::from_slice(&header_buf)?;
    let Some(tensor_entry) = metadata.get(tensor_name) else {
        return Ok(None);
    };
    let Some(shape) = tensor_entry.get("shape").and_then(|shape| shape.as_array()) else {
        return Ok(None);
    };

    let dims = shape
        .iter()
        .map(|dim| dim.as_u64().map(|value| value as usize))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "Invalid shape metadata for tensor {tensor_name} in {}",
                safetensors_path.display()
            ))
        })?;
    Ok(Some(dims))
}

fn fix_timestamp_sequence(data: &[u32]) -> Vec<u32> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }

    let mut dp = vec![1usize; n];
    let mut parent = vec![usize::MAX; n];
    for i in 1..n {
        for j in 0..i {
            if data[j] <= data[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
        }
    }

    let mut max_idx = 0usize;
    for i in 1..n {
        if dp[i] > dp[max_idx] {
            max_idx = i;
        }
    }

    let mut lis_indices = Vec::new();
    let mut idx = max_idx;
    while idx != usize::MAX {
        lis_indices.push(idx);
        idx = parent[idx];
    }
    lis_indices.reverse();

    let mut is_normal = vec![false; n];
    for idx in lis_indices {
        is_normal[idx] = true;
    }

    let mut result: Vec<f32> = data.iter().map(|&v| v as f32).collect();
    let mut i = 0usize;
    while i < n {
        if is_normal[i] {
            i += 1;
            continue;
        }

        let mut j = i;
        while j < n && !is_normal[j] {
            j += 1;
        }
        let anomaly_count = j - i;

        if anomaly_count <= 2 {
            let left_val = (0..i).rev().find(|&k| is_normal[k]).map(|k| result[k]);
            let right_val = (j..n).find(|&k| is_normal[k]).map(|k| result[k]);

            for (offset, value) in result[i..j].iter_mut().enumerate() {
                let k = i + offset;
                *value = match (left_val, right_val) {
                    (None, Some(right)) => right,
                    (Some(left), None) => left,
                    (Some(left), Some(right)) => {
                        let left_distance = k as isize - (i as isize - 1);
                        let right_distance = j as isize - k as isize;
                        if left_distance <= right_distance {
                            left
                        } else {
                            right
                        }
                    }
                    (None, None) => *value,
                };
            }
        } else {
            let left_val = (0..i).rev().find(|&k| is_normal[k]).map(|k| result[k]);
            let right_val = (j..n).find(|&k| is_normal[k]).map(|k| result[k]);

            match (left_val, right_val) {
                (Some(left), Some(right)) => {
                    let step = (right - left) / (anomaly_count as f32 + 1.0);
                    for (offset, value) in result[i..j].iter_mut().enumerate() {
                        *value = left + step * (offset as f32 + 1.0);
                    }
                }
                (Some(left), None) => {
                    for value in &mut result[i..j] {
                        *value = left;
                    }
                }
                (None, Some(right)) => {
                    for value in &mut result[i..j] {
                        *value = right;
                    }
                }
                (None, None) => {}
            }
        }

        i = j;
    }

    result
        .into_iter()
        .map(|value| value.max(0.0) as u32)
        .collect()
}

fn validate_quantization_config(config: &Qwen3AsrConfig) -> Result<()> {
    let quant = config
        .quantization_config
        .as_ref()
        .or_else(|| config.quantization.as_ref());
    let Some(quant) = quant else {
        return Ok(());
    };

    if let Some(mode) = quant.get("mode").and_then(|v| v.as_str()) {
        if mode != "affine" {
            return Err(Error::InvalidInput(format!(
                "Unsupported MLX quantization mode '{mode}'. Only affine is supported."
            )));
        }
    }

    if let Some(bits) = quant.get("bits").and_then(|v| v.as_u64()) {
        if bits == 0 || bits > 8 {
            return Err(Error::InvalidInput(format!(
                "Unsupported MLX quantization bit-width {bits}. Only 1-8 bits are supported."
            )));
        }
    }

    Ok(())
}

struct PromptTokens {
    ids: Vec<u32>,
    audio_pad_start: usize,
    audio_pad_len: usize,
    system_prompt_requested: bool,
    system_prompt_token_count: usize,
    forced_language: Option<String>,
    forced_language_token_count: usize,
    asr_text_token_available: bool,
    asr_text_token_appended: bool,
}

struct ForcedAlignerPrompt {
    ids: Vec<u32>,
    audio_pad_start: usize,
    audio_pad_len: usize,
    timestamp_token_id: u32,
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn build_qwen_asr_prompt_tokens(
    specials: &SpecialTokenIds,
    audio_len: usize,
    language: Option<&str>,
    system_prompt: Option<&str>,
    mut encode_text: impl FnMut(&str) -> Result<Vec<u32>>,
) -> Result<PromptTokens> {
    // Match the upstream Qwen speech-family prompt contract:
    // <|im_start|>system\n
    // [optional system prompt text]
    // <|im_end|>\n
    // <|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n
    // <|im_start|>assistant\n
    // If language is explicitly forced, append: "language {Lang}<asr_text>".
    let system_prompt = non_empty_trimmed(system_prompt);
    let forced_language = forced_language_name(language);
    let mut ids = Vec::new();

    ids.push(specials.im_start);
    ids.extend(encode_text("system\n")?);
    let system_prompt_token_count = if let Some(prompt) = system_prompt {
        let prompt_tokens = encode_text(prompt)?;
        let count = prompt_tokens.len();
        ids.extend(prompt_tokens);
        count
    } else {
        0
    };
    ids.push(specials.im_end);
    ids.extend(encode_text("\n")?);

    ids.push(specials.im_start);
    ids.extend(encode_text("user\n")?);
    ids.push(specials.audio_start);

    let audio_pad_start = ids.len();
    ids.extend(std::iter::repeat_n(specials.audio_token, audio_len));

    ids.push(specials.audio_end);
    ids.push(specials.im_end);
    ids.extend(encode_text("\n")?);
    ids.push(specials.im_start);
    ids.extend(encode_text("assistant\n")?);

    let mut forced_language_token_count = 0usize;
    let mut asr_text_token_appended = false;
    if let Some(lang) = forced_language.as_deref() {
        let language_prefix = encode_text("language ")?;
        forced_language_token_count += language_prefix.len();
        ids.extend(language_prefix);

        let language_tokens = encode_text(lang)?;
        forced_language_token_count += language_tokens.len();
        ids.extend(language_tokens);

        if let Some(asr_text) = specials.asr_text {
            ids.push(asr_text);
            forced_language_token_count += 1;
            asr_text_token_appended = true;
        }
    }

    Ok(PromptTokens {
        ids,
        audio_pad_start,
        audio_pad_len: audio_len,
        system_prompt_requested: system_prompt.is_some(),
        system_prompt_token_count,
        forced_language,
        forced_language_token_count,
        asr_text_token_available: specials.asr_text.is_some(),
        asr_text_token_appended,
    })
}

fn non_empty_trimmed(value: Option<&str>) -> Option<&str> {
    let trimmed = value?.trim();
    (!trimmed.is_empty()).then_some(trimmed)
}

fn parse_asr_dtype(dtype: Option<&str>) -> Option<DType> {
    match dtype.map(|d| d.trim().to_ascii_lowercase()) {
        Some(d) if d == "bfloat16" || d == "bf16" => Some(DType::BF16),
        Some(d) if d == "float16" || d == "f16" || d == "fp16" => Some(DType::F16),
        Some(d) if d == "float32" || d == "f32" || d == "fp32" => Some(DType::F32),
        _ => None,
    }
}

fn normalized_language_name(language: &str) -> String {
    let lang = language.trim();
    if lang.eq_ignore_ascii_case("auto") {
        return "Auto".to_string();
    }

    let mut out = String::with_capacity(lang.len());
    let mut new_word = true;
    for ch in lang.chars() {
        if ch.is_ascii_alphabetic() {
            if new_word {
                out.push(ch.to_ascii_uppercase());
                new_word = false;
            } else {
                out.push(ch.to_ascii_lowercase());
            }
        } else {
            out.push(ch);
            new_word = ch == ' ' || ch == '-' || ch == '_';
        }
    }
    out
}

fn forced_language_name(language: Option<&str>) -> Option<String> {
    let lang = language?.trim();
    if lang.is_empty() || lang.eq_ignore_ascii_case("auto") {
        return None;
    }
    Some(normalized_language_name(lang))
}

fn parse_asr_output(raw: &str, user_language: Option<&str>) -> (Option<String>, String) {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return (forced_language_name(user_language), String::new());
    }

    if let Some(language) = forced_language_name(user_language) {
        if let Some(rest) = trimmed.strip_prefix("language ") {
            if let Some((_, text)) = split_language_prefixed_output(rest) {
                return (Some(language), text);
            }
        }
        return (Some(language), trimmed.to_string());
    }

    let asr_text_token = "<asr_text>";
    if let Some((meta_part, text_part)) = trimmed.split_once(asr_text_token) {
        let meta_trimmed = meta_part.trim();
        let text_trimmed = text_part.trim();
        if meta_trimmed.to_ascii_lowercase().contains("language none") {
            return if text_trimmed.is_empty() {
                (None, String::new())
            } else {
                (None, text_trimmed.to_string())
            };
        }

        for line in meta_trimmed.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(value) = line.strip_prefix("language ") {
                let value = value.trim();
                if !value.is_empty() {
                    return (
                        Some(normalized_language_name(value)),
                        text_trimmed.to_string(),
                    );
                }
            }
        }

        return (None, text_trimmed.to_string());
    }

    if let Some(rest) = trimmed.strip_prefix("language ") {
        if let Some((language, text)) = split_language_prefixed_output(rest) {
            return (language, text);
        }
    }

    (None, trimmed.to_string())
}

fn split_language_prefixed_output(rest: &str) -> Option<(Option<String>, String)> {
    let rest = rest.trim();
    if rest.is_empty() {
        return None;
    }
    if rest.eq_ignore_ascii_case("none") {
        return Some((None, String::new()));
    }

    // Preferred split: explicit whitespace between language and transcript.
    if let Some((lang, text)) = rest.split_once(char::is_whitespace) {
        let lang = lang.trim();
        if !lang.is_empty()
            && lang.chars().all(|ch| ch.is_ascii_alphabetic() || ch == '-')
            && !has_lower_to_upper_transition(lang)
        {
            return Some((
                Some(normalized_language_name(lang)),
                text.trim().to_string(),
            ));
        }
    }

    // Fallback for GGUF decode where `<asr_text>` is not in tokenizer vocab and
    // the language token glues directly to text, e.g. "EnglishThe quick ...".
    let mut chars = rest.char_indices().peekable();
    let mut split_idx = None;
    let mut prev_is_lower = false;
    while let Some((idx, ch)) = chars.next() {
        if idx == 0 {
            prev_is_lower = ch.is_ascii_lowercase();
            continue;
        }
        if prev_is_lower && ch.is_ascii_uppercase() {
            split_idx = Some(idx);
            break;
        }
        prev_is_lower = ch.is_ascii_lowercase();
    }
    let split_idx = split_idx?;
    let lang = rest[..split_idx].trim();
    if lang.is_empty() || !lang.chars().all(|ch| ch.is_ascii_alphabetic() || ch == '-') {
        return None;
    }
    let text = rest[split_idx..].trim().to_string();
    Some((Some(normalized_language_name(lang)), text))
}

fn has_lower_to_upper_transition(value: &str) -> bool {
    let mut prev_is_lower = false;
    for ch in value.chars() {
        if prev_is_lower && ch.is_ascii_uppercase() {
            return true;
        }
        prev_is_lower = ch.is_ascii_lowercase();
    }
    false
}

fn build_mrope_positions(
    seq_len: usize,
    start_pos: usize,
    audio_span: Option<(usize, usize)>,
) -> Vec<i64> {
    if let Some((audio_start, audio_len)) = audio_span {
        let mut pos = Vec::with_capacity(seq_len);
        let mut st = 0usize;
        let mut st_idx = start_pos as i64;

        if audio_start > 0 && audio_start <= seq_len {
            let text_len = audio_start - st;
            for i in 0..text_len {
                pos.push(st_idx + i as i64);
            }
            st = audio_start;
            st_idx += text_len as i64;
        }

        if audio_len > 0 && st < seq_len {
            let audio_take = audio_len.min(seq_len - st);
            for i in 0..audio_take {
                pos.push(st_idx + i as i64);
            }
            st += audio_take;
            st_idx += audio_take as i64;
        }

        if st < seq_len {
            let tail = seq_len - st;
            for i in 0..tail {
                pos.push(st_idx + i as i64);
            }
        }

        pos
    } else {
        (start_pos..start_pos + seq_len).map(|p| p as i64).collect()
    }
}

fn is_special_generation_token(specials: &SpecialTokenIds, id: u32) -> bool {
    if id == specials.im_start
        || id == specials.im_end
        || id == specials.audio_start
        || id == specials.audio_end
        || id == specials.audio_token
        || id == specials.pad
        || id == specials.eos
        || specials.eos_alt == Some(id)
    {
        return true;
    }
    if let Some(asr_text) = specials.asr_text {
        if id == asr_text {
            return true;
        }
    }
    if let Some(id0) = specials.fim_prefix {
        if id == id0 {
            return true;
        }
    }
    if let Some(id0) = specials.fim_middle {
        if id == id0 {
            return true;
        }
    }
    if let Some(id0) = specials.fim_suffix {
        if id == id0 {
            return true;
        }
    }
    if let Some(id0) = specials.fim_pad {
        if id == id0 {
            return true;
        }
    }
    false
}

fn qwen_asr_drop_last_mel_frame() -> bool {
    env_bool("IZWI_QWEN_ASR_DROP_LAST_MEL_FRAME").unwrap_or(false)
}

fn env_bool(name: &str) -> Option<bool> {
    std::env::var(name).ok().and_then(|raw| {
        let normalized = raw.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        }
    })
}

fn apply_qwen_mel_frame_policy(mel_spec: &mut Vec<Vec<f32>>, drop_last: bool) -> bool {
    if drop_last && !mel_spec.is_empty() {
        mel_spec.pop();
        return true;
    }
    false
}

fn resample(audio: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if src_rate == dst_rate {
        return Ok(audio.to_vec());
    }
    if src_rate == 0 || dst_rate == 0 {
        return Err(Error::InvalidInput(format!(
            "Invalid sample rate for resampling: {src_rate} -> {dst_rate}"
        )));
    }
    if audio.is_empty() {
        return Ok(Vec::new());
    }

    // Antirez/qwen-asr uses a bounded windowed-sinc resampler with a Kaiser
    // window for anti-aliasing when downsampling. Keep this Qwen-local so the
    // broader audio stack remains unchanged.
    const SINC_HALF: isize = 16;
    const KAISER_BETA: f64 = 6.0;

    let ratio = dst_rate as f64 / src_rate as f64;
    let cutoff = ratio.min(1.0);
    let out_len = ((audio.len() as u64)
        .saturating_mul(dst_rate as u64)
        .checked_div(src_rate as u64)
        .unwrap_or(0) as usize)
        .max(1);
    let inv_i0_beta = 1.0 / bessel_i0(KAISER_BETA);
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let center = src_pos.floor() as isize;
        let mut acc = 0.0f64;
        let mut weight_sum = 0.0f64;

        for j in (center - SINC_HALF + 1)..=(center + SINC_HALF) {
            let d = j as f64 - src_pos;
            let coeff = sinc(d * cutoff)
                * kaiser_window(d, SINC_HALF as f64, KAISER_BETA, inv_i0_beta)
                * cutoff;
            if let Some(sample) = (j >= 0)
                .then_some(j as usize)
                .and_then(|idx| audio.get(idx))
            {
                acc += *sample as f64 * coeff;
                weight_sum += coeff;
            }
        }

        out.push(if weight_sum.abs() > 1e-9 {
            (acc / weight_sum) as f32
        } else {
            0.0
        });
    }
    Ok(out)
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-9 {
        1.0
    } else {
        let pix = std::f64::consts::PI * x;
        pix.sin() / pix
    }
}

fn kaiser_window(distance: f64, half_width: f64, beta: f64, inv_i0_beta: f64) -> f64 {
    let normalized = distance / half_width;
    if normalized <= -1.0 || normalized >= 1.0 {
        return 0.0;
    }
    let inner = (1.0 - normalized * normalized).max(0.0).sqrt();
    bessel_i0(beta * inner) * inv_i0_beta
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let xx = x * x;
    for k in 1..=20 {
        let k = k as f64;
        term *= xx / (4.0 * k * k);
        sum += term;
    }
    sum
}

fn argmax(logits: &Tensor) -> Result<u32> {
    let logits = match logits.rank() {
        1 => logits.clone(),
        2 => {
            let (batch, _vocab) = logits.dims2()?;
            if batch != 1 {
                return Err(Error::InferenceError(format!(
                    "Unexpected batched logits for argmax: expected batch=1, got {batch}"
                )));
            }
            logits.i(0)?
        }
        rank => {
            return Err(Error::InferenceError(format!(
                "Unexpected logits rank for argmax: {rank}"
            )))
        }
    };
    let idx = logits.argmax(D::Minus1)?;
    let idx = if idx.rank() == 0 {
        idx
    } else {
        idx.squeeze(0)?
    };
    idx.to_dtype(DType::U32)?
        .to_scalar::<u32>()
        .map_err(Error::from)
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

fn collect_stop_token_ids(specials: &SpecialTokenIds) -> Vec<u32> {
    let mut stop_ids = vec![specials.im_end, specials.eos];
    if let Some(alt) = specials.eos_alt {
        if alt != specials.im_end && alt != specials.eos {
            stop_ids.push(alt);
        }
    }
    stop_ids
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use std::path::PathBuf;

    use crate::backends::DeviceSelector;

    #[test]
    fn collect_stop_token_ids_deduplicates_alt_eos() {
        let specials = SpecialTokenIds {
            im_start: 1,
            im_end: 2,
            audio_start: 3,
            audio_end: 4,
            audio_token: 5,
            timestamp: Some(12),
            asr_text: Some(7),
            fim_prefix: Some(8),
            fim_middle: Some(9),
            fim_suffix: Some(10),
            fim_pad: Some(11),
            eos: 6,
            eos_alt: Some(6),
            pad: 0,
        };
        let stop_ids = collect_stop_token_ids(&specials);
        assert_eq!(stop_ids, vec![2, 6]);
    }

    fn test_specials(asr_text: Option<u32>) -> SpecialTokenIds {
        SpecialTokenIds {
            im_start: 1,
            im_end: 2,
            audio_start: 3,
            audio_end: 4,
            audio_token: 5,
            timestamp: Some(12),
            asr_text,
            fim_prefix: Some(8),
            fim_middle: Some(9),
            fim_suffix: Some(10),
            fim_pad: Some(11),
            eos: 6,
            eos_alt: Some(13),
            pad: 0,
        }
    }

    fn encode_prompt_piece_for_test(text: &str) -> Result<Vec<u32>> {
        match text {
            "system\n" => Ok(vec![20]),
            "\n" => Ok(vec![21]),
            "user\n" => Ok(vec![22]),
            "assistant\n" => Ok(vec![23]),
            "language " => Ok(vec![24]),
            "English" => Ok(vec![25]),
            "spell Izwi correctly" => Ok(vec![26, 27, 28]),
            other => Err(Error::TokenizationError(format!(
                "unexpected test prompt piece: {other}"
            ))),
        }
    }

    #[test]
    fn qwen_asr_prompt_inserts_system_prompt_and_asr_text_marker() {
        let prompt = build_qwen_asr_prompt_tokens(
            &test_specials(Some(7)),
            2,
            Some("english"),
            Some("  spell Izwi correctly  "),
            encode_prompt_piece_for_test,
        )
        .expect("prompt");

        assert_eq!(
            prompt.ids,
            vec![
                1, 20, 26, 27, 28, 2, 21, 1, 22, 3, 5, 5, 4, 2, 21, 1, 23, 24, 25, 7
            ]
        );
        assert_eq!(prompt.audio_pad_start, 10);
        assert_eq!(prompt.audio_pad_len, 2);
        assert!(prompt.system_prompt_requested);
        assert_eq!(prompt.system_prompt_token_count, 3);
        assert_eq!(prompt.forced_language.as_deref(), Some("English"));
        assert_eq!(prompt.forced_language_token_count, 3);
        assert!(prompt.asr_text_token_available);
        assert!(prompt.asr_text_token_appended);
    }

    #[test]
    fn qwen_asr_prompt_preserves_markerless_forced_language_fallback() {
        let prompt = build_qwen_asr_prompt_tokens(
            &test_specials(None),
            1,
            Some("english"),
            Some("   "),
            encode_prompt_piece_for_test,
        )
        .expect("prompt");

        assert_eq!(
            prompt.ids,
            vec![1, 20, 2, 21, 1, 22, 3, 5, 4, 2, 21, 1, 23, 24, 25]
        );
        assert_eq!(prompt.audio_pad_start, 7);
        assert!(!prompt.system_prompt_requested);
        assert_eq!(prompt.system_prompt_token_count, 0);
        assert_eq!(prompt.forced_language.as_deref(), Some("English"));
        assert_eq!(prompt.forced_language_token_count, 2);
        assert!(!prompt.asr_text_token_available);
        assert!(!prompt.asr_text_token_appended);
    }

    #[test]
    fn qwen_asr_diagnostics_json_shape_is_stable() {
        let diagnostics = Qwen3AsrDiagnostics {
            input_sample_rate: 44_100,
            input_samples: 44_100,
            resampled_sample_rate: 16_000,
            resampled_samples: 16_000,
            mel_frames: 100,
            mel_frames_before_drop: 121,
            mel_frames_before_truncate: 120,
            mel_last_frame_dropped: true,
            mel_frames_truncated: true,
            audio_tokens: 50,
            prompt_tokens: 64,
            system_prompt_requested: true,
            system_prompt_token_count: 4,
            forced_language: Some("English".to_string()),
            forced_language_token_count: 3,
            asr_text_token_available: true,
            asr_text_token_appended: true,
            max_new_tokens: 128,
            generated_tokens: 12,
            timings: Qwen3AsrTimingDiagnostics {
                resample_ms: 1.0,
                mel_ms: 2.0,
                audio_encode_ms: 3.0,
                prefill_ms: 4.0,
                decode_ms: 5.0,
                total_ms: 15.0,
            },
        }
        .to_json();

        assert_eq!(diagnostics["model_family"], "qwen3_asr");
        assert_eq!(diagnostics["audio"]["mel_frames"], 100);
        assert_eq!(diagnostics["audio"]["mel_frames_before_drop"], 121);
        assert_eq!(diagnostics["audio"]["mel_last_frame_dropped"], true);
        assert_eq!(diagnostics["prompt"]["system_prompt_requested"], true);
        assert_eq!(diagnostics["prompt"]["asr_text_token_appended"], true);
        assert_eq!(diagnostics["decode"]["generated_tokens"], 12);
        assert_eq!(diagnostics["timings_ms"]["model_total"], 15.0);
    }

    #[test]
    fn forced_language_name_ignores_auto_and_empty() {
        assert_eq!(forced_language_name(None), None);
        assert_eq!(forced_language_name(Some("")), None);
        assert_eq!(forced_language_name(Some("Auto")), None);
        assert_eq!(
            forced_language_name(Some("english")),
            Some("English".to_string())
        );
    }

    #[test]
    fn parse_asr_output_extracts_detected_language_and_text() {
        let (language, text) = parse_asr_output(
            "language Chinese<asr_text>甚至出现交易几乎停滞的情况。",
            None,
        );
        assert_eq!(language.as_deref(), Some("Chinese"));
        assert_eq!(text, "甚至出现交易几乎停滞的情况。");
    }

    #[test]
    fn parse_asr_output_respects_forced_language() {
        let (language, text) = parse_asr_output("hello world", Some("english"));
        assert_eq!(language.as_deref(), Some("English"));
        assert_eq!(text, "hello world");
    }

    #[test]
    fn parse_asr_output_respects_forced_language_and_strips_prefix() {
        let (language, text) =
            parse_asr_output("language EnglishThe quick brown fox.", Some("english"));
        assert_eq!(language.as_deref(), Some("English"));
        assert_eq!(text, "The quick brown fox.");
    }

    #[test]
    fn parse_asr_output_handles_empty_audio_marker() {
        let (language, text) = parse_asr_output("language None<asr_text>", None);
        assert_eq!(language, None);
        assert!(text.is_empty());
    }

    #[test]
    fn parse_asr_output_handles_markerless_language_prefix_with_space() {
        let (language, text) = parse_asr_output("language English The quick brown fox.", None);
        assert_eq!(language.as_deref(), Some("English"));
        assert_eq!(text, "The quick brown fox.");
    }

    #[test]
    fn parse_asr_output_handles_markerless_language_prefix_without_space() {
        let (language, text) = parse_asr_output("language EnglishThe quick brown fox.", None);
        assert_eq!(language.as_deref(), Some("English"));
        assert_eq!(text, "The quick brown fox.");
    }

    #[test]
    fn parse_asr_dtype_handles_common_aliases() {
        assert_eq!(parse_asr_dtype(Some("bf16")), Some(DType::BF16));
        assert_eq!(parse_asr_dtype(Some("bfloat16")), Some(DType::BF16));
        assert_eq!(parse_asr_dtype(Some("fp16")), Some(DType::F16));
        assert_eq!(parse_asr_dtype(Some("float32")), Some(DType::F32));
        assert_eq!(parse_asr_dtype(Some("unknown")), None);
    }

    #[test]
    fn qwen_mel_frame_policy_is_opt_in() {
        let mut mel = vec![vec![1.0], vec![2.0], vec![3.0]];
        assert!(!apply_qwen_mel_frame_policy(&mut mel, false));
        assert_eq!(mel.len(), 3);

        assert!(apply_qwen_mel_frame_policy(&mut mel, true));
        assert_eq!(mel, vec![vec![1.0], vec![2.0]]);

        let mut empty: Vec<Vec<f32>> = Vec::new();
        assert!(!apply_qwen_mel_frame_policy(&mut empty, true));
    }

    #[test]
    fn resample_preserves_identity_when_rates_match() {
        let input = vec![0.0f32, 0.25, -0.5, 1.0];
        let output = resample(&input, 16_000, 16_000).expect("resample");
        assert_eq!(output, input);
    }

    #[test]
    fn resample_uses_expected_duration_for_up_and_down_sampling() {
        let input = (0..100).map(|idx| idx as f32 / 100.0).collect::<Vec<_>>();
        let down = resample(&input, 100, 50).expect("downsample");
        let up = resample(&input, 50, 100).expect("upsample");

        assert_eq!(down.len(), 50);
        assert_eq!(up.len(), 200);
        assert!(down.iter().all(|value| value.is_finite()));
        assert!(up.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn resample_rejects_zero_sample_rates() {
        let input = vec![0.0f32, 1.0];
        let err = resample(&input, 0, 16_000).expect_err("zero source rate");
        assert!(format!("{err}").contains("Invalid sample rate"));

        let err = resample(&input, 16_000, 0).expect_err("zero destination rate");
        assert!(format!("{err}").contains("Invalid sample rate"));
    }

    #[test]
    fn resample_keeps_constant_signal_stable() {
        let input = vec![0.75f32; 128];
        let output = resample(&input, 48_000, 16_000).expect("resample");

        assert_eq!(output.len(), 42);
        for value in output {
            assert!((value - 0.75).abs() < 1e-3, "unexpected value: {value}");
        }
    }

    #[test]
    fn text_delta_finds_suffix_when_prefix_changes() {
        assert_eq!(text_delta("Hello", "Hello world"), " world");
        assert_eq!(text_delta("abcd", "abXY"), "XY");
    }

    #[test]
    fn extract_alignment_words_strips_markers() {
        let words = extract_alignment_words("hello,  world! it's me.");
        assert_eq!(words, vec!["hello", "world", "it's", "me"]);
    }

    #[test]
    fn extract_alignment_words_splits_cjk_and_hangul() {
        let words = extract_alignment_words("你好 world 안녕");
        assert_eq!(words, vec!["你", "好", "world", "안", "녕"]);
    }

    #[test]
    fn distribute_words_over_interval_is_monotonic() {
        let words = vec!["one".to_string(), "two".to_string(), "three".to_string()];
        let aligned = distribute_words_over_interval(&words, 100, 160);
        assert_eq!(aligned.len(), 3);
        assert!(aligned[0].1 < aligned[0].2);
        assert!(aligned[1].1 >= aligned[0].2);
        assert!(aligned[2].2 >= aligned[1].2);
    }

    #[test]
    fn normalize_alignment_bounds_clamps_to_duration() {
        let mut alignments = vec![
            ("one".to_string(), 0, 20),
            ("two".to_string(), 10, 12),
            ("three".to_string(), 100, 140),
        ];
        normalize_alignment_bounds(&mut alignments, 60);
        assert_eq!(alignments[0].0, "one");
        assert!(alignments[0].1 < alignments[0].2);
        assert!(alignments[1].1 >= alignments[0].2);
        assert!(alignments[2].2 <= 60);
    }

    #[test]
    fn alignment_distribution_is_degenerate_detects_tail_collapse() {
        let alignments = (0..20)
            .map(|idx| (format!("w{idx}"), 27_302u32, 27_303u32))
            .collect::<Vec<_>>();
        assert!(alignment_distribution_is_degenerate(&alignments, 27_303));
    }

    #[test]
    fn alignment_distribution_is_degenerate_accepts_spread_alignment() {
        let alignments = (0..20)
            .map(|idx| {
                let start = idx * 120;
                let end = start + 80;
                (format!("w{idx}"), start, end)
            })
            .collect::<Vec<_>>();
        assert!(!alignment_distribution_is_degenerate(&alignments, 3_000));
    }

    #[test]
    fn fix_timestamp_sequence_recovers_monotonicity() {
        let repaired = fix_timestamp_sequence(&[100, 80, 120, 60, 140]);
        assert_eq!(repaired.len(), 5);
        for pair in repaired.windows(2) {
            assert!(pair[0] <= pair[1]);
        }
    }

    #[test]
    fn qwen_asr_gguf_filename_matches_new_variants() {
        assert_eq!(
            qwen_asr_gguf_filename(ModelVariant::Qwen3Asr06BGguf),
            Some("qwen3_asr_0.6b_q8_0.gguf")
        );
        assert_eq!(
            qwen_asr_gguf_filename(ModelVariant::Qwen3Asr17BGguf),
            Some("qwen3_asr_1.7b_q8_0.gguf")
        );
        assert_eq!(
            qwen_asr_gguf_filename(ModelVariant::WhisperLargeV3Turbo),
            None
        );
    }

    #[test]
    fn argmax_selects_max_from_rank1_logits() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::from_vec(vec![0.1f32, 3.2, -1.0, 2.7], 4, &device).expect("logits");
        let idx = argmax(&logits).expect("argmax");
        assert_eq!(idx, 1);
    }

    #[test]
    fn argmax_rejects_rank2_batched_logits() {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::from_vec(vec![0.1f32; 8], (2, 4), &device).expect("logits");
        let err = argmax(&logits).expect_err("argmax should reject batched rank2");
        assert!(format!("{err}").contains("Unexpected batched logits for argmax"));
    }

    #[test]
    fn qwen3_asr_kv_page_size_policy_upsizes_long_metal_prompts() {
        let size = qwen3_asr_kv_page_size_policy(64, true, 9306, false, None);
        assert_eq!(size, 128);

        let size = qwen3_asr_kv_page_size_policy(96, true, 9306, false, None);
        assert_eq!(size, 128);
    }

    #[test]
    fn qwen3_asr_kv_page_size_policy_respects_env_override_priority() {
        let asr_override = qwen3_asr_kv_page_size_policy(64, true, 9306, false, Some(256));
        assert_eq!(asr_override, 256);

        let global_override = qwen3_asr_kv_page_size_policy(192, true, 9306, true, None);
        assert_eq!(global_override, 192);

        let non_metal = qwen3_asr_kv_page_size_policy(64, false, 9306, false, None);
        assert_eq!(non_metal, 64);
    }

    #[test]
    #[ignore = "requires local Qwen3-ForcedAligner checkpoint"]
    fn forced_aligner_local_checkpoint_loads_and_aligns() {
        let models_root = std::env::var("IZWI_MODELS_DIR")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                dirs::data_local_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("izwi")
                    .join("models")
            });
        let model_dir = models_root.join("Qwen3-ForcedAligner-0.6B");
        if !model_dir.join("model.safetensors").exists() {
            eprintln!(
                "Skipping forced aligner local checkpoint test, model not found at {}",
                model_dir.display()
            );
            return;
        }

        let device = DeviceSelector::detect_with_preference(Some("cpu")).expect("cpu device");
        let model = Qwen3AsrModel::load(&model_dir, ModelVariant::Qwen3ForcedAligner06B, device)
            .expect("forced aligner should load");
        let audio = vec![0f32; 16_000];
        let alignment = model
            .force_align(&audio, 16_000, "hello world", None)
            .expect("forced align should run");
        assert!(
            !alignment.is_empty(),
            "forced aligner should return timestamps"
        );
    }

    #[test]
    #[ignore = "requires local Qwen3-ASR-0.6B-GGUF assets and fox.wav fixture"]
    fn qwen3_asr_local_fox_transcription_smoke_if_available() {
        let models_root = std::env::var("IZWI_MODELS_DIR")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                dirs::data_local_dir()
                    .unwrap_or_else(|| PathBuf::from("."))
                    .join("izwi")
                    .join("models")
            });
        let model_dir = models_root.join("Qwen3-ASR-0.6B-GGUF");
        let gguf_path = model_dir.join("qwen3_asr_0.6b_q8_0.gguf");
        if !gguf_path.exists() {
            eprintln!(
                "Skipping local ASR smoke test, model not found at {}",
                gguf_path.display()
            );
            return;
        }

        let fox_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/fox.wav");
        if !fox_path.exists() {
            eprintln!(
                "Skipping local ASR smoke test, fixture not found at {}",
                fox_path.display()
            );
            return;
        }

        let wav_bytes = std::fs::read(&fox_path).expect("read fox.wav");
        let (samples, sample_rate) =
            crate::runtime::audio_io::decode_wav_bytes(&wav_bytes).expect("decode fox.wav");

        let device = DeviceSelector::detect_with_preference(Some("cpu")).expect("cpu device");
        let model = Qwen3AsrModel::load(&model_dir, ModelVariant::Qwen3Asr06BGguf, device)
            .expect("qwen3 asr model should load");
        let raw = model
            .transcribe(&samples, sample_rate, None)
            .expect("transcription should run");
        let (_, text) = parse_asr_output(&raw, None);
        assert!(
            !text.trim().is_empty(),
            "local ASR output should not be empty"
        );
        assert!(
            text.to_ascii_lowercase().contains("quick brown fox"),
            "unexpected local ASR transcript: {text}"
        );
    }
}
