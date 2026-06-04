//! High-level Voxtral TTS model contract.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use tracing::{info, warn};

use crate::backends::{DeviceKind, DeviceProfile};
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::models::architectures::qwen3::core::Qwen3Cache;
use crate::models::architectures::voxtral::lm::VoxtralLM;
use crate::models::shared::attention::paged::{KvCacheQuantization, DEFAULT_KV_PAGE_SIZE};

use super::acoustic::{AudioSpecialToken, FlowMatchingAudioTransformer, AUDIO_SPECIAL_TOKEN_COUNT};
use super::codec::{VoxtralCodecConfig, VoxtralCodecDecoder, VoxtralCodecTimeline};
use super::config::VoxtralTtsConfig;
use super::sampling::VoxtralTtsGenerationParams;
use super::tokenizer::VoxtralTtsTokenizer;
use super::voice::{voice_embedding_path, VoxtralVoiceCatalog, VoxtralVoiceEmbeddingLibrary};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoxtralTtsDTypePlan {
    pub language_model: DType,
    pub acoustic_transformer: DType,
    pub codec: DType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoxtralTtsAssets {
    pub params_path: PathBuf,
    pub tekken_path: PathBuf,
    pub weights_path: PathBuf,
    pub voice_embedding_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct VoxtralTtsOutput {
    pub samples: Vec<f32>,
    pub sample_rate: usize,
    pub frames_generated: usize,
}

pub struct VoxtralTtsModel {
    pub model_dir: PathBuf,
    pub config: VoxtralTtsConfig,
    pub voices: VoxtralVoiceCatalog,
    pub voice_embeddings: VoxtralVoiceEmbeddingLibrary,
    pub codec_config: VoxtralCodecConfig,
    pub dtype_plan: VoxtralTtsDTypePlan,
    pipeline: Option<VoxtralTtsPipeline>,
}

struct VoxtralTtsPipeline {
    tokenizer: VoxtralTtsTokenizer,
    language_model: VoxtralLM,
    acoustic_transformer: FlowMatchingAudioTransformer,
    codec_decoder: VoxtralCodecDecoder,
    audio_embeddings: VoxtralAudioTokenEmbeddings,
    device: Device,
    device_kind: DeviceKind,
}

struct VoxtralAudioTokenEmbeddings {
    embeddings: Embedding,
    offsets: Vec<u32>,
    offsets_tensor: Option<Tensor>,
    codebook_sizes: Vec<u32>,
    num_codebooks: usize,
}

impl VoxtralTtsAssets {
    pub fn from_config(model_dir: &Path, config: &VoxtralTtsConfig) -> Self {
        Self {
            params_path: model_dir.join("params.json"),
            tekken_path: model_dir.join("tekken.json"),
            weights_path: model_dir.join("consolidated.safetensors"),
            voice_embedding_paths: config
                .voice_names_by_id()
                .iter()
                .map(|voice| voice_embedding_path(model_dir, voice))
                .collect(),
        }
    }

    pub fn missing_paths(&self) -> Vec<PathBuf> {
        let mut missing = Vec::new();
        for path in [&self.params_path, &self.tekken_path, &self.weights_path] {
            if !path.exists() {
                missing.push(path.clone());
            }
        }
        missing.extend(
            self.voice_embedding_paths
                .iter()
                .filter(|path| !path.exists())
                .cloned(),
        );
        missing
    }

    pub fn validate_present(&self) -> Result<()> {
        let missing = self.missing_paths();
        if missing.is_empty() {
            return Ok(());
        }
        let rendered = missing
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        Err(Error::ModelLoadError(format!(
            "Voxtral TTS model directory is incomplete; missing {rendered}"
        )))
    }
}

impl VoxtralTtsModel {
    pub fn load(model_dir: &Path, device: DeviceProfile) -> Result<Self> {
        let mut model = Self::load_metadata(model_dir, device.clone())?;
        info!(
            "Loading Voxtral TTS generation pipeline from {:?}",
            model_dir
        );
        info!(
            "Voxtral TTS dtype plan on {:?}: language_model={:?}, acoustic_transformer={:?}, codec={:?}",
            device.kind,
            model.dtype_plan.language_model,
            model.dtype_plan.acoustic_transformer,
            model.dtype_plan.codec
        );
        let language_vb =
            load_voxtral_tts_weights(model_dir, model.dtype_plan.language_model, &device)?;
        let acoustic_vb =
            load_voxtral_tts_weights(model_dir, model.dtype_plan.acoustic_transformer, &device)?;
        let codec_vb = load_voxtral_tts_weights(model_dir, model.dtype_plan.codec, &device)?;
        let tokenizer = VoxtralTtsTokenizer::load(model_dir, &model.config)?;
        let language_model = VoxtralLM::load(model.config.text_config(), language_vb.clone())?;
        let acoustic_transformer = FlowMatchingAudioTransformer::load(
            &model.config,
            acoustic_vb.pp("acoustic_transformer"),
        )?;
        let codec_decoder =
            VoxtralCodecDecoder::load(&model.config, codec_vb.pp("audio_tokenizer"))?;
        let audio_embeddings =
            VoxtralAudioTokenEmbeddings::load(&model.config, model.config.text_dim, language_vb)?;
        model.pipeline = Some(VoxtralTtsPipeline {
            tokenizer,
            language_model,
            acoustic_transformer,
            codec_decoder,
            audio_embeddings,
            device: device.device.clone(),
            device_kind: device.kind,
        });
        Ok(model)
    }

    pub fn load_metadata(model_dir: &Path, device: DeviceProfile) -> Result<Self> {
        info!("Loading Voxtral TTS metadata from {:?}", model_dir);
        let config = VoxtralTtsConfig::load(model_dir)?;
        let assets = VoxtralTtsAssets::from_config(model_dir, &config);
        assets.validate_present()?;
        let voices = VoxtralVoiceCatalog::from_config(model_dir, &config)?;
        voices.validate_embedding_files()?;
        let codec_config = VoxtralCodecConfig::from_config(&config)?;
        let dtype_plan =
            select_voxtral_tts_dtypes(&device, voxtral_tts_dtype_override().as_deref())?;
        let voice_embeddings = VoxtralVoiceEmbeddingLibrary::new(
            voices.clone(),
            device.device.clone(),
            dtype_plan.language_model,
            config.text_dim,
        );
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            config,
            voices,
            voice_embeddings,
            codec_config,
            dtype_plan,
            pipeline: None,
        })
    }

    pub fn available_speakers(&self) -> Vec<String> {
        self.voices.names_by_id()
    }

    pub fn generate_with_voice(
        &self,
        text: &str,
        voice: &str,
        params: VoxtralTtsGenerationParams,
    ) -> Result<VoxtralTtsOutput> {
        self.voices.resolve(voice)?;
        let pipeline = self.pipeline.as_ref().ok_or_else(|| {
            Error::InferenceError(
                "Voxtral TTS generation requires the full model loader, not metadata-only loading"
                    .to_string(),
            )
        })?;
        pipeline.generate(text, voice, params, self)
    }
}

impl VoxtralTtsPipeline {
    fn generate(
        &self,
        text: &str,
        voice: &str,
        params: VoxtralTtsGenerationParams,
        model: &VoxtralTtsModel,
    ) -> Result<VoxtralTtsOutput> {
        if text.trim().is_empty() {
            return Err(Error::InvalidInput(
                "Voxtral TTS text input cannot be empty".to_string(),
            ));
        }
        let total_start = Instant::now();
        let prompt_start = Instant::now();
        let voice_embedding = model.voice_embeddings.load(voice)?;
        let voice_frames = voice_embedding.dim(1)?;
        let prompt = self.tokenizer.build_speech_prompt(text, voice_frames)?;
        let prompt_embeds = self
            .prompt_embeddings(
                &prompt.input_ids,
                &voice_embedding,
                prompt.voice_token_range.as_ref(),
            )
            .map_err(|err| {
                Error::InferenceError(format!("Voxtral TTS prompt embedding failed: {err}"))
            })?;
        let prompt_duration = prompt_start.elapsed();
        let max_frames = params.max_frames.max(1);
        let dense_decode_tokens = prompt.input_ids.len().saturating_add(max_frames);
        let mut cache = build_voxtral_tts_decode_cache(
            self.language_model.num_layers(),
            self.device_kind,
            dense_decode_tokens,
        );
        let lm_prefill_start = Instant::now();
        let prefill_hidden = self
            .language_model
            .forward_hidden_with_embeds(&prompt_embeds, 0, Some(&mut cache), None, None)
            .map_err(|err| {
                Error::InferenceError(format!("Voxtral TTS LM prefill failed: {err}"))
            })?;
        let lm_prefill_duration = lm_prefill_start.elapsed();
        let mut pos = prompt.input_ids.len();
        let mut last_hidden = last_sequence_hidden(&prefill_hidden, "Voxtral TTS LM prefill")?;
        let mut frames = Vec::new();
        let mut acoustic_duration = Duration::ZERO;
        let mut lm_decode_duration = Duration::ZERO;

        for frame_idx in 0..max_frames {
            let acoustic_start = Instant::now();
            let generated = self
                .acoustic_transformer
                .forward_audio_codes_with_feedback_tensor(
                    &last_hidden,
                    params.cfg_alpha,
                    params.n_decoding_steps,
                    !frames.is_empty(),
                )
                .map_err(|err| {
                    Error::InferenceError(format!("Voxtral TTS acoustic generation failed: {err}"))
                })?;
            let feedback_code_tensor = generated.shifted_code_tensor;
            acoustic_duration += acoustic_start.elapsed();
            let frame = generated.frames.into_iter().next().ok_or_else(|| {
                Error::InferenceError("Voxtral acoustic transformer returned no frames".to_string())
            })?;
            if frame.first().copied() == Some(AudioSpecialToken::End.id()) {
                break;
            }
            let feedback_frame = feedback_code_tensor.is_none().then(|| frame.clone());
            frames.push(frame);
            if params.auto_frame_budget
                && (frames.len() == 1 || frames.len() % 8 == 0 || frame_idx + 1 >= max_frames)
            {
                info!(
                    "Voxtral TTS generated {}/{} acoustic frame(s) (~{:.2}s audio budget)",
                    frames.len(),
                    max_frames,
                    frames.len() as f32 / model.config.frame_rate()
                );
            }
            if frame_idx + 1 >= max_frames {
                break;
            }
            let next_embed = if let Some(feedback_code_tensor) = feedback_code_tensor.as_ref() {
                self.audio_embeddings
                    .embedding_for_shifted_code_tensor(feedback_code_tensor)?
            } else {
                let feedback_frame = feedback_frame.as_ref().ok_or_else(|| {
                    Error::InferenceError(
                        "Voxtral TTS feedback frame missing host and tensor codes".to_string(),
                    )
                })?;
                self.audio_embeddings
                    .embedding_for_shifted_codes(feedback_frame)?
            };
            let lm_decode_start = Instant::now();
            let hidden = self
                .language_model
                .forward_hidden_with_embeds(&next_embed, pos, Some(&mut cache), None, None)
                .map_err(|err| {
                    Error::InferenceError(format!("Voxtral TTS LM decode failed: {err}"))
                })?;
            lm_decode_duration += lm_decode_start.elapsed();
            pos += 1;
            last_hidden = last_sequence_hidden(&hidden, "Voxtral TTS LM decode")?;
        }

        if frames.is_empty() {
            return Err(Error::InferenceError(
                "Voxtral TTS generated no audio frames".to_string(),
            ));
        }
        if params.auto_frame_budget && frames.len() >= max_frames {
            warn!(
                "Voxtral TTS reached auto frame budget of {} frame(s) before the model emitted end-of-audio",
                max_frames
            );
        }
        let frames_generated = frames.len();
        let timeline = VoxtralCodecTimeline::new(frames_to_codebooks(frames)?).map_err(|err| {
            Error::InferenceError(format!("Voxtral TTS timeline construction failed: {err}"))
        })?;
        let codec_start = Instant::now();
        let samples = self
            .codec_decoder
            .decode_timeline(&timeline)
            .map_err(|err| {
                Error::InferenceError(format!("Voxtral TTS codec decode failed: {err}"))
            })?;
        let codec_duration = codec_start.elapsed();
        let total_duration = total_start.elapsed();
        info!(
            "Voxtral TTS timings: frames={}, samples={}, prompt={:.2}ms, lm_prefill={:.2}ms, acoustic={:.2}ms, lm_decode={:.2}ms, codec={:.2}ms, total={:.2}ms",
            frames_generated,
            samples.len(),
            duration_ms(prompt_duration),
            duration_ms(lm_prefill_duration),
            duration_ms(acoustic_duration),
            duration_ms(lm_decode_duration),
            duration_ms(codec_duration),
            duration_ms(total_duration)
        );
        Ok(VoxtralTtsOutput {
            samples,
            sample_rate: model.codec_config.sample_rate,
            frames_generated,
        })
    }

    fn prompt_embeddings(
        &self,
        input_ids: &[u32],
        voice_embedding: &Tensor,
        voice_range: Option<&std::ops::Range<usize>>,
    ) -> Result<Tensor> {
        let ids = Tensor::from_vec(input_ids.to_vec(), (1, input_ids.len()), &self.device)?;
        let embeds = self.language_model.embeddings(&ids)?;
        let Some(range) = voice_range else {
            return Ok(embeds);
        };
        let expected_frames = range.end.saturating_sub(range.start);
        if voice_embedding.dim(1)? != expected_frames {
            return Err(Error::InferenceError(format!(
                "Voxtral voice embedding has {} frames but prompt reserved {expected_frames}",
                voice_embedding.dim(1)?
            )));
        }
        let mut parts = Vec::new();
        if range.start > 0 {
            parts.push(embeds.narrow(1, 0, range.start)?);
        }
        parts.push(voice_embedding.to_dtype(embeds.dtype())?);
        if range.end < input_ids.len() {
            parts.push(embeds.narrow(1, range.end, input_ids.len() - range.end)?);
        }
        Tensor::cat(&parts, 1).map_err(Error::from)
    }
}

fn last_sequence_hidden(hidden: &Tensor, context: &str) -> Result<Tensor> {
    if hidden.rank() != 3 {
        return Err(Error::InferenceError(format!(
            "{context} returned rank-{} hidden states; expected [batch, seq, hidden]",
            hidden.rank()
        )));
    }
    if hidden.dim(0)? != 1 {
        return Err(Error::InferenceError(format!(
            "{context} returned batch size {}; Voxtral TTS only supports batch size 1",
            hidden.dim(0)?
        )));
    }
    let seq_len = hidden.dim(1)?;
    if seq_len == 0 {
        return Err(Error::InferenceError(format!(
            "{context} returned no sequence positions"
        )));
    }
    hidden.i((0, seq_len - 1, ..)).map_err(Error::from)
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

impl VoxtralAudioTokenEmbeddings {
    fn load(config: &VoxtralTtsConfig, embedding_dim: usize, vb: VarBuilder) -> Result<Self> {
        let codebook_sizes = voxtral_audio_embedding_codebook_sizes(config)?;
        let offsets = codebook_offsets(&codebook_sizes)?;
        let total_size = codebook_sizes
            .iter()
            .try_fold(0usize, |acc, size| acc.checked_add(*size as usize))
            .ok_or_else(|| {
                Error::ConfigError("Voxtral audio embedding table size overflowed".to_string())
            })?;
        let padded_size = 128 * total_size.div_ceil(128);
        let num_codebooks = config.num_codebooks();
        let offsets_tensor = if vb.device().is_cuda() {
            Some(Tensor::from_vec(
                offsets.clone(),
                (1, num_codebooks),
                vb.device(),
            )?)
        } else {
            None
        };
        for candidate in [
            "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight",
            "audio_tokenizer.audio_token_embedding.embeddings.weight",
            "audio_generation.audio_tokenizer.audio_token_embedding.embeddings.weight",
        ] {
            if vb.contains_tensor(candidate) {
                let weights = vb.get((padded_size, embedding_dim), candidate)?;
                return Ok(Self {
                    embeddings: Embedding::new(weights, embedding_dim),
                    offsets,
                    offsets_tensor: offsets_tensor.clone(),
                    codebook_sizes,
                    num_codebooks,
                });
            }
        }
        Err(Error::ModelLoadError(
            "Voxtral TTS checkpoint is missing audio codebook embedding weights".to_string(),
        ))
    }

    fn embedding_for_shifted_codes(&self, shifted_codes: &[u32]) -> Result<Tensor> {
        if shifted_codes.len() != self.num_codebooks {
            return Err(Error::InferenceError(format!(
                "Voxtral audio embedding expected {} codebooks, got {}",
                self.num_codebooks,
                shifted_codes.len()
            )));
        }
        let mut ids = Vec::with_capacity(shifted_codes.len());
        for (idx, token) in shifted_codes.iter().enumerate() {
            if *token >= self.codebook_sizes[idx] {
                return Err(Error::InferenceError(format!(
                    "Voxtral audio codebook {idx} token {token} exceeds size {}",
                    self.codebook_sizes[idx]
                )));
            }
            ids.push(self.offsets[idx] + *token);
        }
        let ids = Tensor::from_vec(
            ids,
            (1, shifted_codes.len()),
            self.embeddings.embeddings().device(),
        )?;
        self.embeddings
            .forward(&ids)?
            .sum(1)?
            .unsqueeze(1)
            .map_err(Error::from)
    }

    fn embedding_for_shifted_code_tensor(&self, shifted_codes: &Tensor) -> Result<Tensor> {
        let shifted_codes = match shifted_codes.rank() {
            1 => shifted_codes.unsqueeze(0)?,
            2 => shifted_codes.clone(),
            rank => {
                return Err(Error::InferenceError(format!(
                    "Voxtral audio code tensor expected rank 1 or 2, got rank {rank}"
                )));
            }
        };
        if shifted_codes.dim(1)? != self.num_codebooks {
            return Err(Error::InferenceError(format!(
                "Voxtral audio embedding expected {} codebooks, got {}",
                self.num_codebooks,
                shifted_codes.dim(1)?
            )));
        }
        let offsets = self.offsets_tensor.as_ref().ok_or_else(|| {
            Error::InferenceError(
                "Voxtral audio tensor embedding requires CUDA offsets tensor".to_string(),
            )
        })?;
        let ids = shifted_codes.to_dtype(DType::U32)?.broadcast_add(offsets)?;
        self.embeddings
            .forward(&ids)?
            .sum(1)?
            .unsqueeze(1)
            .map_err(Error::from)
    }
}

fn frames_to_codebooks(frames: Vec<Vec<u32>>) -> Result<Vec<Vec<u32>>> {
    let Some(first) = frames.first() else {
        return Err(Error::InferenceError(
            "Voxtral generated frame list is empty".to_string(),
        ));
    };
    let codebooks = first.len();
    if codebooks == 0 {
        return Err(Error::InferenceError(
            "Voxtral generated frames have no codebooks".to_string(),
        ));
    }
    let mut out = vec![Vec::with_capacity(frames.len()); codebooks];
    for frame in frames {
        if frame.len() != codebooks {
            return Err(Error::InferenceError(
                "Voxtral generated frame codebook count changed during decoding".to_string(),
            ));
        }
        for (idx, token) in frame.into_iter().enumerate() {
            out[idx].push(token);
        }
    }
    Ok(out)
}

fn voxtral_audio_embedding_codebook_sizes(config: &VoxtralTtsConfig) -> Result<Vec<u32>> {
    let mut sizes = Vec::with_capacity(config.num_codebooks());
    sizes.push(
        config
            .semantic_codebook_size()
            .checked_add(AUDIO_SPECIAL_TOKEN_COUNT as usize)
            .ok_or_else(|| {
                Error::ConfigError("Voxtral semantic codebook size overflowed".to_string())
            })? as u32,
    );
    let acoustic_size = config
        .acoustic_codebook_size()
        .checked_add(AUDIO_SPECIAL_TOKEN_COUNT as usize)
        .ok_or_else(|| {
            Error::ConfigError("Voxtral acoustic codebook size overflowed".to_string())
        })? as u32;
    sizes.extend(std::iter::repeat(acoustic_size).take(config.n_acoustic_codebooks()));
    Ok(sizes)
}

fn codebook_offsets(sizes: &[u32]) -> Result<Vec<u32>> {
    let mut offsets = Vec::with_capacity(sizes.len());
    let mut current = 0u32;
    for size in sizes {
        offsets.push(current);
        current = current.checked_add(*size).ok_or_else(|| {
            Error::ConfigError("Voxtral audio embedding offsets overflowed".to_string())
        })?;
    }
    Ok(offsets)
}

fn load_voxtral_tts_weights<'a>(
    model_dir: &'a Path,
    dtype: DType,
    device: &'a DeviceProfile,
) -> Result<VarBuilder<'a>> {
    let weights_path = model_dir.join("consolidated.safetensors");
    unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device.device).map_err(|err| {
            Error::ModelLoadError(format!("Failed to load Voxtral TTS weights: {err}"))
        })
    }
}

fn build_voxtral_tts_decode_cache(
    num_layers: usize,
    device_kind: DeviceKind,
    max_decode_tokens: usize,
) -> Qwen3Cache {
    Qwen3Cache::with_page_size_quantization_and_dense_decode_tokens(
        num_layers,
        DEFAULT_KV_PAGE_SIZE,
        KvCacheQuantization::None,
        voxtral_tts_dense_decode_max_tokens(device_kind, max_decode_tokens),
    )
}

fn voxtral_tts_dense_decode_max_tokens(device_kind: DeviceKind, max_decode_tokens: usize) -> usize {
    if device_kind.is_cuda() {
        max_decode_tokens.max(1)
    } else {
        0
    }
}

pub fn select_voxtral_tts_dtypes(
    device: &DeviceProfile,
    dtype_override: Option<&str>,
) -> Result<VoxtralTtsDTypePlan> {
    if let Some(raw) = dtype_override.map(str::trim).filter(|raw| !raw.is_empty()) {
        let dtype =
            device.select_model_dtype_checked(ModelFamily::VoxtralTts, Some(raw), "Voxtral TTS")?;
        return Ok(VoxtralTtsDTypePlan {
            language_model: dtype,
            acoustic_transformer: dtype,
            codec: dtype,
        });
    }

    let transformer_dtype = device.select_model_dtype(ModelFamily::VoxtralTts, None);
    let codec_dtype = if device.kind.is_cuda() {
        transformer_dtype
    } else {
        DType::F32
    };
    Ok(VoxtralTtsDTypePlan {
        language_model: transformer_dtype,
        acoustic_transformer: transformer_dtype,
        codec: codec_dtype,
    })
}

fn voxtral_tts_dtype_override() -> Option<String> {
    std::env::var("IZWI_VOXTRAL_TTS_DTYPE")
        .ok()
        .or_else(|| std::env::var("IZWI_VOXTRAL_DTYPE").ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use candle_nn::VarBuilder;
    use serde_json::json;
    use std::collections::HashMap;

    use crate::backends::{DeviceCapabilities, DeviceKind};
    use crate::models::architectures::voxtral::tts::config::{fixture_json, VoxtralTtsConfig};

    fn profile(kind: DeviceKind, supports_bf16: bool, supports_f16: bool) -> DeviceProfile {
        DeviceProfile {
            device: Device::Cpu,
            kind,
            capabilities: DeviceCapabilities {
                supports_bf16,
                supports_f16,
                ..Default::default()
            },
            memory_pool: None,
        }
    }

    #[test]
    fn asset_contract_uses_hf_file_layout() {
        let config = VoxtralTtsConfig::from_json_str(fixture_json()).unwrap();
        let assets =
            VoxtralTtsAssets::from_config(Path::new("/models/Voxtral-4B-TTS-2603"), &config);
        assert_eq!(
            assets.params_path,
            Path::new("/models/Voxtral-4B-TTS-2603").join("params.json")
        );
        assert_eq!(
            assets.weights_path,
            Path::new("/models/Voxtral-4B-TTS-2603").join("consolidated.safetensors")
        );
        assert_eq!(assets.voice_embedding_paths.len(), 20);
        assert_eq!(
            assets.voice_embedding_paths[1],
            Path::new("/models/Voxtral-4B-TTS-2603")
                .join("voice_embedding")
                .join("casual_male.pt")
        );
    }

    #[test]
    fn dtype_plan_keeps_cpu_and_metal_in_f32_and_allows_cuda_bf16() {
        let cpu = profile(DeviceKind::Cpu, false, false);
        let cpu_plan = select_voxtral_tts_dtypes(&cpu, None).unwrap();
        assert_eq!(cpu_plan.language_model, DType::F32);
        assert_eq!(cpu_plan.acoustic_transformer, DType::F32);
        assert_eq!(cpu_plan.codec, DType::F32);

        let metal = profile(DeviceKind::Metal, false, true);
        let metal_plan = select_voxtral_tts_dtypes(&metal, None).unwrap();
        assert_eq!(metal_plan.language_model, DType::F32);
        assert_eq!(metal_plan.acoustic_transformer, DType::F32);
        assert_eq!(metal_plan.codec, DType::F32);

        let cuda = profile(DeviceKind::Cuda, true, true);
        let cuda_plan = select_voxtral_tts_dtypes(&cuda, None).unwrap();
        assert_eq!(cuda_plan.language_model, DType::BF16);
        assert_eq!(cuda_plan.acoustic_transformer, DType::BF16);
        assert_eq!(cuda_plan.codec, DType::BF16);
    }

    #[test]
    fn tts_dense_decode_cache_is_cuda_only() {
        assert_eq!(voxtral_tts_dense_decode_max_tokens(DeviceKind::Cpu, 128), 0);
        assert_eq!(
            voxtral_tts_dense_decode_max_tokens(DeviceKind::Metal, 128),
            0
        );
        assert_eq!(
            voxtral_tts_dense_decode_max_tokens(DeviceKind::Cuda, 128),
            128
        );

        let cpu_cache = build_voxtral_tts_decode_cache(2, DeviceKind::Cpu, 128);
        assert_eq!(cpu_cache.dense_decode_max_tokens(), 0);

        let metal_cache = build_voxtral_tts_decode_cache(2, DeviceKind::Metal, 128);
        assert_eq!(metal_cache.dense_decode_max_tokens(), 0);

        let cuda_cache = build_voxtral_tts_decode_cache(2, DeviceKind::Cuda, 128);
        assert_eq!(cuda_cache.dense_decode_max_tokens(), 128);
    }

    #[test]
    fn dtype_override_applies_to_all_voxtral_tts_stages() {
        let cuda = profile(DeviceKind::Cuda, true, true);
        let plan = select_voxtral_tts_dtypes(&cuda, Some("f16")).unwrap();
        assert_eq!(plan.language_model, DType::F16);
        assert_eq!(plan.acoustic_transformer, DType::F16);
        assert_eq!(plan.codec, DType::F16);
    }

    #[test]
    fn audio_embedding_codebook_sizes_include_special_tokens() {
        let config = tiny_audio_embedding_config();
        assert_eq!(
            voxtral_audio_embedding_codebook_sizes(&config).unwrap(),
            vec![6, 5, 5]
        );
        assert_eq!(codebook_offsets(&[6, 5, 5]).unwrap(), vec![0, 6, 11]);
    }

    #[test]
    fn audio_embedding_sums_shifted_codebook_embeddings() {
        let device = Device::Cpu;
        let config = tiny_audio_embedding_config();
        let mut rows = Vec::new();
        for row in 0..128 {
            rows.extend([
                row as f32,
                row as f32 + 0.25,
                row as f32 + 0.5,
                row as f32 + 0.75,
            ]);
        }
        let tensors = HashMap::from([(
            "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight".to_string(),
            Tensor::from_vec(rows, (128, 4), &device).unwrap(),
        )]);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let embeddings = VoxtralAudioTokenEmbeddings::load(&config, 4, vb).unwrap();
        let embed = embeddings.embedding_for_shifted_codes(&[2, 3, 4]).unwrap();
        assert_eq!(embed.dims(), &[1, 1, 4]);
        let values = embed.to_vec3::<f32>().unwrap();
        assert_eq!(values[0][0], vec![26.0, 26.75, 27.5, 28.25]);
    }

    #[test]
    fn audio_embedding_accepts_shifted_code_tensors() {
        let device = Device::Cpu;
        let config = tiny_audio_embedding_config();
        let mut rows = Vec::new();
        for row in 0..128 {
            rows.extend([
                row as f32,
                row as f32 + 0.25,
                row as f32 + 0.5,
                row as f32 + 0.75,
            ]);
        }
        let tensors = HashMap::from([(
            "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight".to_string(),
            Tensor::from_vec(rows, (128, 4), &device).unwrap(),
        )]);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let mut embeddings = VoxtralAudioTokenEmbeddings::load(&config, 4, vb).unwrap();
        embeddings.offsets_tensor = Some(
            Tensor::from_vec(
                embeddings.offsets.clone(),
                (1, embeddings.num_codebooks),
                &device,
            )
            .unwrap(),
        );
        let codes = Tensor::from_vec(vec![2u32, 3, 4], (1, 3), &device).unwrap();

        let embed = embeddings
            .embedding_for_shifted_code_tensor(&codes)
            .unwrap();

        assert_eq!(embed.dims(), &[1, 1, 4]);
        let values = embed.to_vec3::<f32>().unwrap();
        assert_eq!(values[0][0], vec![26.0, 26.75, 27.5, 28.25]);
    }

    #[test]
    fn generated_frames_transpose_to_codec_codebooks() {
        let codebooks = frames_to_codebooks(vec![vec![2, 3, 4], vec![5, 6, 7]]).unwrap();
        assert_eq!(codebooks, vec![vec![2, 5], vec![3, 6], vec![4, 7]]);
        assert!(frames_to_codebooks(vec![vec![1], vec![1, 2]]).is_err());
    }

    #[test]
    fn last_sequence_hidden_extracts_decode_step_output() {
        let device = Device::Cpu;
        let hidden =
            Tensor::from_vec(vec![0.0f32, 0.1, 1.0, 1.1, 2.0, 2.1], (1, 3, 2), &device).unwrap();

        let last = last_sequence_hidden(&hidden, "test").unwrap();

        assert_eq!(last.dims(), &[2]);
        assert_eq!(last.to_vec1::<f32>().unwrap(), vec![2.0, 2.1]);
    }

    #[test]
    #[ignore = "requires IZWI_VOXTRAL_TTS_SMOKE_MODEL_DIR pointing at a full Voxtral TTS checkpoint"]
    fn voxtral_tts_local_generate_smoke_if_env_set() {
        let model_dir = std::env::var("IZWI_VOXTRAL_TTS_SMOKE_MODEL_DIR")
            .map(PathBuf::from)
            .expect("set IZWI_VOXTRAL_TTS_SMOKE_MODEL_DIR to run the local Voxtral TTS smoke");
        let max_frames = std::env::var("IZWI_VOXTRAL_TTS_SMOKE_MAX_FRAMES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(4)
            .max(1);
        let model = VoxtralTtsModel::load(&model_dir, DeviceProfile::cpu()).unwrap();
        let output = model
            .generate_with_voice(
                "Testing Voxtral TTS.",
                "casual_male",
                VoxtralTtsGenerationParams {
                    max_frames,
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(output.sample_rate, 24_000);
        assert!(output.frames_generated > 0);
        assert_eq!(
            output.samples.len(),
            output.frames_generated * model.codec_config.downsample_factor().unwrap()
        );
        assert!(output.samples.iter().all(|sample| sample.is_finite()));
        assert!(output.samples.iter().any(|sample| sample.abs() > 1e-6));
    }

    fn tiny_audio_embedding_config() -> VoxtralTtsConfig {
        let mut value: serde_json::Value = serde_json::from_str(fixture_json()).unwrap();
        let audio = &mut value["multimodal"]["audio_model_args"];
        audio["semantic_codebook_size"] = json!(4);
        audio["acoustic_codebook_size"] = json!(3);
        audio["n_acoustic_codebook"] = json!(2);
        audio["audio_encoding_args"]["num_codebooks"] = json!(3);
        value["multimodal"]["audio_tokenizer_args"]["semantic_codebook_size"] = json!(4);
        value["multimodal"]["audio_tokenizer_args"]["acoustic_codebook_size"] = json!(3);
        VoxtralTtsConfig::from_json_str(&value.to_string()).unwrap()
    }
}
