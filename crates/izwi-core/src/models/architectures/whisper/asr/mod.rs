//! Native Whisper Large v3 Turbo ASR model loader and inference.
//!
//! This implementation follows Whisper prompting/decoding conventions used in:
//! - `whisper.cpp` (llama.cpp ecosystem): SOT/lang/task/no-timestamps prefix and
//!   timestamp suppression for text-only decode.
//! - Hugging Face `transformers`: language/task prompt handling and suppress token
//!   masks from `generation_config.json`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{self, model::Whisper, Config as WhisperConfig};
use serde::Deserialize;
use tracing::info;

use crate::audio::{MelConfig, MelSpectrogram};
use crate::error::{Error, Result};
use crate::models::shared::device::DeviceProfile;
use crate::tokenizer::Tokenizer;

const SAMPLE_RATE: u32 = whisper::SAMPLE_RATE as u32;
const DEFAULT_MAX_NEW_TOKENS: usize = 448;
const MAX_AUDIO_SECONDS_HINT: f32 = whisper::CHUNK_LENGTH as f32;
const REPETITION_GUARD_MIN_SPAN_TOKENS: usize = 8;
const REPETITION_GUARD_MAX_SPAN_TOKENS: usize = 96;
const REPETITION_GUARD_MIN_TOTAL_TOKENS: usize = 20;

#[derive(Debug, Clone, Deserialize, Default)]
struct WhisperGenerationConfig {
    #[serde(default)]
    begin_suppress_tokens: Vec<u32>,
    #[serde(default)]
    suppress_tokens: Vec<u32>,
    #[serde(default)]
    lang_to_id: HashMap<String, u32>,
    #[serde(default)]
    task_to_id: HashMap<String, u32>,
    #[serde(default)]
    no_timestamps_token_id: Option<u32>,
    #[serde(default)]
    max_length: Option<usize>,
    #[serde(default)]
    eos_token_id: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
struct WhisperSpecialTokens {
    sot: u32,
    transcribe: u32,
    eot: u32,
    no_timestamps: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsrTranscriptionOutput {
    pub text: String,
    pub language: Option<String>,
}

pub struct WhisperTurboAsrModel {
    device: DeviceProfile,
    model_dtype: DType,
    whisper: Mutex<Whisper>,
    config: WhisperConfig,
    generation: WhisperGenerationConfig,
    tokenizer: Tokenizer,
    special: WhisperSpecialTokens,
    mel: MelSpectrogram,
    suppress_tokens: Vec<u32>,
    language_token_ids: Vec<u32>,
    token_id_to_language_code: HashMap<u32, String>,
}

impl WhisperTurboAsrModel {
    pub fn load(model_dir: &Path, device: DeviceProfile) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config_data = fs::read_to_string(config_path)?;
        let config: WhisperConfig = serde_json::from_str(&config_data)?;

        let generation = read_generation_config(model_dir)?;
        let tokenizer = Tokenizer::from_path(model_dir)?;

        let model_dtype = std::env::var("IZWI_WHISPER_DTYPE")
            .ok()
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| device.select_dtype(Some(value)))
            .unwrap_or_else(|| device.select_dtype(None));

        let index_path = model_dir.join("model.safetensors.index.json");
        let vb = if index_path.exists() {
            let index_data = fs::read_to_string(index_path)?;
            let index: serde_json::Value = serde_json::from_str(&index_data)?;
            let weight_map = index
                .get("weight_map")
                .and_then(|value| value.as_object())
                .ok_or_else(|| {
                    Error::InvalidInput("Invalid Whisper safetensors index format".to_string())
                })?;

            let mut shard_files: Vec<String> = weight_map
                .values()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect();
            shard_files.sort();
            shard_files.dedup();

            let shard_paths: Vec<std::path::PathBuf> = shard_files
                .iter()
                .map(|file| model_dir.join(file))
                .collect();

            unsafe {
                VarBuilder::from_mmaped_safetensors(&shard_paths, model_dtype, &device.device)?
            }
        } else {
            let model_path = model_dir.join("model.safetensors");
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[model_path], model_dtype, &device.device)?
            }
        };

        let whisper = Whisper::load(&vb, config.clone())?;
        let special = resolve_special_tokens(&tokenizer, &generation)?;
        let (language_token_ids, token_id_to_language_code) =
            build_language_token_maps(&tokenizer, &generation);

        let mut suppress_tokens = generation.suppress_tokens.clone();
        suppress_tokens.sort_unstable();
        suppress_tokens.dedup();

        let mel = MelSpectrogram::new(MelConfig {
            sample_rate: whisper::SAMPLE_RATE,
            n_fft: whisper::N_FFT,
            hop_length: whisper::HOP_LENGTH,
            n_mels: config.num_mel_bins,
            f_min: 0.0,
            f_max: (whisper::SAMPLE_RATE / 2) as f32,
            normalize: true,
        })?;

        info!(
            "Loaded Whisper Large v3 Turbo ASR on {:?} (dtype={:?})",
            device.kind, model_dtype
        );

        Ok(Self {
            device,
            model_dtype,
            whisper: Mutex::new(whisper),
            config,
            generation,
            tokenizer,
            special,
            mel,
            suppress_tokens,
            language_token_ids,
            token_id_to_language_code,
        })
    }

    pub fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<String> {
        let mut no_op = |_delta: &str| {};
        Ok(self
            .transcribe_impl(audio, sample_rate, language, &mut no_op)?
            .text)
    }

    pub fn transcribe_with_details(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<AsrTranscriptionOutput> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_impl(audio, sample_rate, language, &mut no_op)
    }

    pub fn transcribe_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        Ok(self
            .transcribe_impl(audio, sample_rate, language, on_delta)?
            .text)
    }

    pub fn max_audio_seconds_hint(&self) -> Option<f32> {
        Some(MAX_AUDIO_SECONDS_HINT)
    }

    fn transcribe_impl(
        &self,
        audio: &[f32],
        sample_rate: u32,
        language: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<AsrTranscriptionOutput> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let mel = self.prepare_mel(audio, sample_rate)?;
        let mut whisper = self
            .whisper
            .lock()
            .map_err(|_| Error::InferenceError("Whisper model mutex poisoned".to_string()))?;

        whisper.reset_kv_cache();
        let audio_features = whisper.encoder.forward(&mel, true)?;

        let mut resolved_language = if let Some(language) = language {
            self.resolve_language_token(language)?
        } else {
            None
        };
        if resolved_language.is_none() {
            resolved_language = self.detect_language_token(&mut whisper, &audio_features)?;
        }

        let mut prompt = Vec::with_capacity(4);
        prompt.push(self.special.sot);
        if let Some((language_token, _language_code)) = resolved_language.as_ref() {
            prompt.push(*language_token);
        }
        prompt.push(self.special.transcribe);
        if let Some(no_timestamps) = self.special.no_timestamps {
            prompt.push(no_timestamps);
        }

        let mut generated_tokens = Vec::<u32>::new();
        let mut assembled = String::new();

        let max_steps = decode_step_budget(
            prompt.len(),
            self.config.max_target_positions,
            self.generation.max_length.unwrap_or(DEFAULT_MAX_NEW_TOKENS),
        )?;

        for step_idx in 0..max_steps {
            let tokens_t = Tensor::new(prompt.as_slice(), &self.device.device)?.unsqueeze(0)?;
            let ys = whisper
                .decoder
                .forward(&tokens_t, &audio_features, step_idx == 0)?;
            let (_, seq_len, _) = ys.dims3()?;
            let logits = whisper
                .decoder
                .final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;

            let mut logits_vec = logits.to_vec1::<f32>()?;
            self.apply_decode_constraints(&mut logits_vec, step_idx == 0);
            let next = select_next_token(&logits_vec, self.special.eot);
            if next == self.special.eot {
                break;
            }

            generated_tokens.push(next);
            prompt.push(next);

            if let Some((span, repeats)) = find_suffix_token_repetition(&generated_tokens) {
                // Degenerate greedy loops can repeat an entire phrase verbatim.
                // Keep the first span and stop decoding before the transcript explodes.
                let trim = span.saturating_mul(repeats.saturating_sub(1));
                if trim > 0 && trim <= generated_tokens.len() {
                    generated_tokens.truncate(generated_tokens.len() - trim);
                    prompt.truncate(prompt.len() - trim);
                    assembled = self.decode_generated_text(&generated_tokens)?;
                }
                break;
            }

            let decoded = self.decode_generated_text(&generated_tokens)?;
            let delta = text_delta(&assembled, &decoded);
            if !delta.is_empty() {
                for ch in delta.chars() {
                    let mut buf = [0u8; 4];
                    on_delta(ch.encode_utf8(&mut buf));
                }
            }
            assembled = decoded;
        }

        let text = assembled.trim().to_string();
        let language = resolved_language.map(|(_token_id, code)| code);
        Ok(AsrTranscriptionOutput { text, language })
    }

    fn prepare_mel(&self, audio: &[f32], sample_rate: u32) -> Result<Tensor> {
        let mono_16khz = if sample_rate == SAMPLE_RATE {
            audio.to_vec()
        } else {
            resample_linear(audio, sample_rate, SAMPLE_RATE)
        };

        let mut mel_spec = self.mel.compute(&mono_16khz)?;
        if mel_spec.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        // Whisper encoder downsamples by 2 before positional embeddings.
        let max_input_frames = self.config.max_source_positions.saturating_mul(2).max(1);
        if mel_spec.len() > max_input_frames {
            mel_spec.truncate(max_input_frames);
        }

        let n_mels = self.config.num_mel_bins;
        let frames = mel_spec.len();
        let mut flat = vec![0f32; frames * n_mels];
        for (frame_idx, frame) in mel_spec.iter().enumerate() {
            for mel_idx in 0..n_mels {
                flat[mel_idx * frames + frame_idx] = frame[mel_idx];
            }
        }

        let mel = Tensor::from_vec(flat, (1, n_mels, frames), &self.device.device)?;
        if mel.dtype() != self.model_dtype {
            return Ok(mel.to_dtype(self.model_dtype)?);
        }
        Ok(mel)
    }

    fn resolve_language_token(&self, language: &str) -> Result<Option<(u32, String)>> {
        let normalized = language.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return Ok(None);
        }

        let language_code = if let Some(code) = normalized
            .strip_prefix("<|")
            .and_then(|inner| inner.strip_suffix("|>"))
        {
            code.to_string()
        } else if has_whisper_language_token(
            &self.generation.lang_to_id,
            &normalized,
            &self.tokenizer,
        ) {
            normalized
        } else if let Some(code) = language_name_to_code(&normalized) {
            code.to_string()
        } else if let Some(code) = language_alias_to_code(&normalized) {
            code.to_string()
        } else {
            return Err(Error::InvalidInput(format!(
                "Unsupported Whisper language '{}'",
                language
            )));
        };

        let token = format!("<|{}|>", language_code);
        let token_id = self
            .generation
            .lang_to_id
            .get(&token)
            .copied()
            .or_else(|| self.tokenizer.token_to_id(&token))
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Whisper model does not support language token '{}'",
                    token
                ))
            })?;

        Ok(Some((token_id, language_code)))
    }

    fn detect_language_token(
        &self,
        whisper: &mut Whisper,
        audio_features: &Tensor,
    ) -> Result<Option<(u32, String)>> {
        if self.language_token_ids.is_empty() {
            return Ok(None);
        }

        let tokens = Tensor::new(&[[self.special.sot]], &self.device.device)?;
        let ys = whisper.decoder.forward(&tokens, audio_features, true)?;
        let logits = whisper.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
        let logits_vec = logits.to_vec1::<f32>()?;

        let mut best_token: Option<u32> = None;
        let mut best_score = f32::NEG_INFINITY;
        for token_id in &self.language_token_ids {
            let idx = *token_id as usize;
            if idx >= logits_vec.len() {
                continue;
            }
            let score = logits_vec[idx];
            if score > best_score {
                best_score = score;
                best_token = Some(*token_id);
            }
        }

        let Some(token_id) = best_token else {
            return Ok(None);
        };
        let Some(code) = self.token_id_to_language_code.get(&token_id).cloned() else {
            return Ok(None);
        };

        Ok(Some((token_id, code)))
    }

    fn decode_generated_text(&self, token_ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(token_ids)
    }

    fn apply_decode_constraints(&self, logits: &mut [f32], at_begin: bool) {
        for token_id in &self.suppress_tokens {
            mask_token(logits, *token_id);
        }
        if at_begin {
            for token_id in &self.generation.begin_suppress_tokens {
                mask_token(logits, *token_id);
            }
        }

        mask_token(logits, self.special.sot);
        mask_token(logits, self.special.transcribe);
        for token_id in &self.language_token_ids {
            mask_token(logits, *token_id);
        }

        if let Some(no_timestamps_token_id) = self.special.no_timestamps {
            // whisper.cpp / transformers text-only decode behavior.
            mask_token(logits, no_timestamps_token_id);
            let timestamp_begin = no_timestamps_token_id.saturating_add(1) as usize;
            if timestamp_begin < logits.len() {
                logits[timestamp_begin..].fill(f32::NEG_INFINITY);
            }
        }
    }
}

fn read_generation_config(model_dir: &Path) -> Result<WhisperGenerationConfig> {
    let generation_path = model_dir.join("generation_config.json");
    if !generation_path.exists() {
        return Ok(WhisperGenerationConfig::default());
    }
    let generation_data = fs::read_to_string(generation_path)?;
    Ok(serde_json::from_str::<WhisperGenerationConfig>(
        &generation_data,
    )?)
}

fn resolve_special_tokens(
    tokenizer: &Tokenizer,
    generation: &WhisperGenerationConfig,
) -> Result<WhisperSpecialTokens> {
    let sot = tokenizer.token_to_id(whisper::SOT_TOKEN).ok_or_else(|| {
        Error::TokenizationError("Missing <|startoftranscript|> token".to_string())
    })?;
    let transcribe = tokenizer
        .token_to_id(whisper::TRANSCRIBE_TOKEN)
        .or_else(|| generation.task_to_id.get("transcribe").copied())
        .ok_or_else(|| Error::TokenizationError("Missing <|transcribe|> token".to_string()))?;
    let eot = tokenizer
        .token_to_id(whisper::EOT_TOKEN)
        .or(generation.eos_token_id)
        .ok_or_else(|| Error::TokenizationError("Missing <|endoftext|> token".to_string()))?;
    let no_timestamps = generation
        .no_timestamps_token_id
        .or_else(|| tokenizer.token_to_id(whisper::NO_TIMESTAMPS_TOKEN));

    Ok(WhisperSpecialTokens {
        sot,
        transcribe,
        eot,
        no_timestamps,
    })
}

fn build_language_token_maps(
    tokenizer: &Tokenizer,
    generation: &WhisperGenerationConfig,
) -> (Vec<u32>, HashMap<u32, String>) {
    let mut token_to_lang = HashMap::new();
    let mut lang_ids = Vec::new();

    if generation.lang_to_id.is_empty() {
        for (code, _name) in WHISPER_LANGUAGES {
            let token = format!("<|{}|>", code);
            if let Some(token_id) = tokenizer.token_to_id(&token) {
                lang_ids.push(token_id);
                token_to_lang.insert(token_id, (*code).to_string());
            }
        }
    } else {
        for (token, token_id) in &generation.lang_to_id {
            if let Some(code) = token
                .strip_prefix("<|")
                .and_then(|inner| inner.strip_suffix("|>"))
            {
                lang_ids.push(*token_id);
                token_to_lang.insert(*token_id, code.to_string());
            }
        }
    }

    lang_ids.sort_unstable();
    lang_ids.dedup();
    (lang_ids, token_to_lang)
}

fn has_whisper_language_token(
    generation_lang_to_id: &HashMap<String, u32>,
    code: &str,
    tokenizer: &Tokenizer,
) -> bool {
    let token = format!("<|{}|>", code);
    generation_lang_to_id.contains_key(&token) || tokenizer.token_to_id(&token).is_some()
}

fn mask_token(logits: &mut [f32], token_id: u32) {
    let idx = token_id as usize;
    if idx < logits.len() {
        logits[idx] = f32::NEG_INFINITY;
    }
}

fn select_next_token(logits: &[f32], fallback_token: u32) -> u32 {
    let mut best_idx = fallback_token as usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, value) in logits.iter().enumerate() {
        if *value > best_val {
            best_idx = idx;
            best_val = *value;
        }
    }
    if !best_val.is_finite() {
        fallback_token
    } else {
        best_idx as u32
    }
}

fn text_delta(previous: &str, current: &str) -> String {
    if let Some(delta) = current.strip_prefix(previous) {
        return delta.to_string();
    }
    let common = previous
        .chars()
        .zip(current.chars())
        .take_while(|(left, right)| left == right)
        .count();
    current.chars().skip(common).collect()
}

fn find_suffix_token_repetition(ids: &[u32]) -> Option<(usize, usize)> {
    if ids.len() < REPETITION_GUARD_MIN_TOTAL_TOKENS {
        return None;
    }

    let max_span = (ids.len() / 2).min(REPETITION_GUARD_MAX_SPAN_TOKENS);
    if max_span < REPETITION_GUARD_MIN_SPAN_TOKENS {
        return None;
    }

    for span in (REPETITION_GUARD_MIN_SPAN_TOKENS..=max_span).rev() {
        let tail_start = ids.len() - span;
        let tail = &ids[tail_start..];
        let mut repeats = 1usize;

        while ids.len() >= span.saturating_mul(repeats + 1) {
            let start = ids.len() - span * (repeats + 1);
            let end = start + span;
            if &ids[start..end] == tail {
                repeats += 1;
            } else {
                break;
            }
        }

        if repeats >= 2 {
            return Some((span, repeats));
        }
    }

    None
}

fn decode_step_budget(
    prompt_len: usize,
    max_target_positions: usize,
    generation_max_length: usize,
) -> Result<usize> {
    if max_target_positions == 0 || prompt_len >= max_target_positions {
        return Err(Error::InvalidInput(format!(
            "Whisper decode prompt length {} exceeds decoder context {}",
            prompt_len, max_target_positions
        )));
    }

    let prompt_budget = max_target_positions - prompt_len;

    // Whisper decoder positional embeddings are bounded by max_target_positions.
    // Keep generated tokens within remaining context budget to avoid narrow() overflow.
    Ok(generation_max_length.max(1).min(prompt_budget))
}

#[cfg(test)]
mod tests {
    use super::{decode_step_budget, find_suffix_token_repetition};

    #[test]
    fn decode_step_budget_clamps_generation_to_remaining_context() {
        let budget = decode_step_budget(4, 448, 448).expect("budget");
        assert_eq!(budget, 444);
    }

    #[test]
    fn decode_step_budget_rejects_prompt_overflow() {
        assert!(decode_step_budget(448, 448, 448).is_err());
        assert!(decode_step_budget(449, 448, 448).is_err());
    }

    #[test]
    fn detects_suffix_token_repetition() {
        let mut ids = Vec::new();
        ids.extend(1u32..=12);
        ids.extend(1u32..=12);
        let repetition = find_suffix_token_repetition(&ids);
        assert_eq!(repetition, Some((12, 2)));
    }

    #[test]
    fn ignores_short_or_non_repeating_suffixes() {
        let ids: Vec<u32> = (1..=16).collect();
        assert_eq!(find_suffix_token_repetition(&ids), None);
    }
}

fn resample_linear(audio: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || audio.len() < 2 {
        return audio.to_vec();
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((audio.len() as f64) * ratio).round().max(1.0) as usize;

    let mut out = Vec::with_capacity(out_len);
    for idx in 0..out_len {
        let src_pos = idx as f64 / ratio;
        let left = src_pos.floor() as usize;
        let right = left.saturating_add(1).min(audio.len() - 1);
        let frac = (src_pos - left as f64) as f32;
        out.push(audio[left] * (1.0 - frac) + audio[right] * frac);
    }
    out
}

fn language_name_to_code(language: &str) -> Option<&'static str> {
    WHISPER_LANGUAGES
        .iter()
        .find(|(_code, name)| *name == language)
        .map(|(code, _name)| *code)
}

fn language_alias_to_code(language: &str) -> Option<&'static str> {
    match language {
        "burmese" => Some("my"),
        "valencian" => Some("ca"),
        "flemish" => Some("nl"),
        "haitian" => Some("ht"),
        "letzeburgesch" => Some("lb"),
        "pushto" => Some("ps"),
        "panjabi" => Some("pa"),
        "moldavian" | "moldovan" => Some("ro"),
        "sinhalese" => Some("si"),
        "castilian" => Some("es"),
        "mandarin" => Some("zh"),
        _ => None,
    }
}

// Mirrors Whisper multilingual language table from upstream implementations.
const WHISPER_LANGUAGES: [(&str, &str); 100] = [
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jw", "javanese"),
    ("su", "sundanese"),
    ("yue", "cantonese"),
];
