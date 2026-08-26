//! High-level Voxtral TTS model contract.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Module, VarBuilder};
use tracing::{info, warn};

use crate::backends::{BackendKind, DeviceProfile};
use crate::catalog::{ModelFamily, ModelVariant};
use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::kv::CacheDomainId;
use crate::models::architectures::voxtral::lm::VoxtralLM;
use crate::models::architectures::voxtral::{
    voxtral_invocation_contract, voxtral_physical_state_spec, VoxtralPhysicalStateSpec,
};
use crate::models::shared::attention::paged::default_kv_page_size;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

use super::acoustic::{AudioSpecialToken, FlowMatchingAudioTransformer, AUDIO_SPECIAL_TOKEN_COUNT};
use super::codec::{VoxtralCodecConfig, VoxtralCodecDecoder, VoxtralCodecTimeline};
use super::config::VoxtralTtsConfig;
use super::retained::{
    validate_acoustic_cohort, voxtral_tts_stage_resource_envelope, VoxtralTtsDecodeBatch,
    VoxtralTtsDecodeStep, VoxtralTtsPrefillBatch, VoxtralTtsPrefillStep,
    VoxtralTtsPreparedArtifact, VoxtralTtsQuantumCheckpoint, VoxtralTtsRetainedPhase,
    VoxtralTtsRetainedState, VoxtralTtsStageCeiling, VoxtralTtsStageResourceEnvelope,
};
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

    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<VoxtralPhysicalStateSpec> {
        let pipeline = self.pipeline.as_ref().ok_or_else(|| {
            Error::ModelLoadError(
                "Voxtral TTS physical state requires the full model loader".into(),
            )
        })?;
        let invocation = voxtral_invocation_contract(
            &pipeline.language_model,
            self.dtype_plan.language_model,
            default_kv_page_size(),
            &[CacheDomainId::new(1)],
        )?;
        let max_context_tokens = pipeline
            .language_model
            .physical_context_limit()
            .ok_or_else(|| Error::ModelLoadError("Voxtral TTS has no context limit".into()))?;
        voxtral_physical_state_spec(stage_graphs, invocation, max_context_tokens)
    }

    pub fn generate_with_voice(
        &self,
        text: &str,
        voice: &str,
        params: VoxtralTtsGenerationParams,
    ) -> Result<VoxtralTtsOutput> {
        let _ = (text, voice, params);
        Err(Error::InferenceError(
            "Voxtral TTS requires a lifecycle-owned physical invocation cache".into(),
        ))
    }

    pub(crate) fn generate_with_voice_physical(
        &self,
        text: &str,
        voice: &str,
        params: VoxtralTtsGenerationParams,
        cache: &mut PhysicalPagedKvCache,
    ) -> Result<VoxtralTtsOutput> {
        self.voices.resolve(voice)?;
        let pipeline = self.pipeline.as_ref().ok_or_else(|| {
            Error::InferenceError(
                "Voxtral TTS generation requires the full model loader, not metadata-only loading"
                    .to_string(),
            )
        })?;
        pipeline.generate(text, voice, params, self, cache)
    }

    pub(crate) fn prepare_retained_artifact(
        &self,
        text: &str,
        voice: &str,
    ) -> Result<Arc<VoxtralTtsPreparedArtifact>> {
        if text.trim().is_empty() {
            return Err(Error::InvalidInput(
                "Voxtral TTS text input cannot be empty".into(),
            ));
        }
        self.voices.resolve(voice)?;
        let pipeline = self.pipeline.as_ref().ok_or_else(|| {
            Error::InferenceError("Voxtral TTS retained preparation requires loaded weights".into())
        })?;
        let voice_embedding = self.voice_embeddings.load(voice)?;
        let prompt = pipeline
            .tokenizer
            .build_speech_prompt(text, voice_embedding.dim(1)?)?;
        let prompt_embeddings = pipeline.prompt_embeddings(
            &prompt.input_ids,
            &voice_embedding,
            prompt.voice_token_range.as_ref(),
        )?;
        let retained_resident_bytes = u64::try_from(prompt_embeddings.elem_count())
            .ok()
            .and_then(|elements| {
                elements.checked_mul(u64::try_from(prompt_embeddings.dtype().size_in_bytes()).ok()?)
            })
            .ok_or_else(|| Error::InvalidInput("Voxtral TTS prompt bytes overflowed".into()))?;
        Ok(Arc::new(VoxtralTtsPreparedArtifact {
            prompt_embeddings,
            prompt_tokens: prompt.input_ids.len(),
            source_text: Arc::from(text.trim()),
            voice: Arc::from(voice),
            retained_resident_bytes,
        }))
    }

    pub(crate) fn new_retained_state(
        &self,
        artifact: Arc<VoxtralTtsPreparedArtifact>,
        params: VoxtralTtsGenerationParams,
    ) -> Result<VoxtralTtsRetainedState> {
        let context = self
            .pipeline
            .as_ref()
            .and_then(|pipeline| pipeline.language_model.model_context_limit())
            .ok_or_else(|| Error::ModelLoadError("Voxtral TTS has no context limit".into()))?;
        VoxtralTtsRetainedState::new(artifact, params, context)
    }

    pub(crate) fn retained_stage_ceiling(&self) -> Result<VoxtralTtsStageCeiling> {
        let max_prompt_tokens = self
            .pipeline
            .as_ref()
            .and_then(|pipeline| pipeline.language_model.model_context_limit())
            .ok_or_else(|| Error::ModelLoadError("Voxtral TTS has no context limit".into()))?;
        Ok(VoxtralTtsStageCeiling {
            max_prompt_tokens,
            max_frames: ModelVariant::VOXTRAL_TTS_MAX_OUTPUT_FRAMES,
            hidden_size: self.config.text_dim,
            num_codebooks: self.config.num_codebooks(),
        })
    }

    pub(crate) fn retained_prefill_resource_envelope(
        &self,
        backend: BackendKind,
        start: usize,
        tokens: usize,
        prompt_tokens: usize,
    ) -> Result<VoxtralTtsStageResourceEnvelope> {
        let ceiling = self.retained_stage_ceiling()?;
        let end = start
            .checked_add(tokens)
            .ok_or_else(|| Error::InvalidInput("Voxtral TTS prefill span overflowed".into()))?;
        if tokens == 0 || end > prompt_tokens || prompt_tokens > ceiling.max_prompt_tokens {
            return Err(Error::InvalidInput(
                "Voxtral TTS prefill span exceeds its stage ceiling".into(),
            ));
        }
        let elements = u64::try_from(tokens)
            .ok()
            .and_then(|tokens| tokens.checked_mul(u64::try_from(ceiling.hidden_size).ok()?))
            .ok_or_else(|| {
                Error::InvalidInput("Voxtral TTS prefill workspace overflowed".into())
            })?;
        let workspace = elements.checked_mul(4).ok_or_else(|| {
            Error::InvalidInput("Voxtral TTS prefill workspace bytes overflowed".into())
        })?;
        voxtral_tts_stage_resource_envelope(backend, tokens, elements, workspace)
    }

    pub(crate) fn retained_decode_resource_envelope(
        &self,
        backend: BackendKind,
        batch_width: usize,
        decoding_steps: usize,
    ) -> Result<VoxtralTtsStageResourceEnvelope> {
        let ceiling = self.retained_stage_ceiling()?;
        if batch_width == 0 || decoding_steps == 0 {
            return Err(Error::InvalidInput(
                "Voxtral TTS decode resource shape cannot be empty".into(),
            ));
        }
        let work_units = batch_width
            .checked_mul(decoding_steps.saturating_add(1))
            .ok_or_else(|| Error::InvalidInput("Voxtral TTS decode work overflowed".into()))?;
        let row_elements = ceiling
            .hidden_size
            .checked_add(ceiling.num_codebooks)
            .ok_or_else(|| Error::InvalidInput("Voxtral TTS decode shape overflowed".into()))?;
        let elements = u64::try_from(batch_width)
            .ok()
            .and_then(|width| width.checked_mul(u64::try_from(row_elements).ok()?))
            .ok_or_else(|| Error::InvalidInput("Voxtral TTS decode workspace overflowed".into()))?;
        let workspace = elements.checked_mul(4).ok_or_else(|| {
            Error::InvalidInput("Voxtral TTS decode workspace bytes overflowed".into())
        })?;
        voxtral_tts_stage_resource_envelope(backend, work_units, elements, workspace)
    }

    pub(crate) fn retained_codec_resource_envelope(
        &self,
        backend: BackendKind,
        frames: usize,
    ) -> Result<VoxtralTtsStageResourceEnvelope> {
        let ceiling = self.retained_stage_ceiling()?;
        if frames == 0 || frames > ceiling.max_frames {
            return Err(Error::InvalidInput(
                "Voxtral TTS codec frames exceed its stage ceiling".into(),
            ));
        }
        let elements = u64::try_from(frames)
            .ok()
            .and_then(|frames| frames.checked_mul(u64::try_from(ceiling.num_codebooks).ok()?))
            .ok_or_else(|| Error::InvalidInput("Voxtral TTS codec workspace overflowed".into()))?;
        let workspace = elements.checked_mul(4).ok_or_else(|| {
            Error::InvalidInput("Voxtral TTS codec workspace bytes overflowed".into())
        })?;
        voxtral_tts_stage_resource_envelope(backend, frames, elements, workspace)
    }

    pub(crate) fn retained_prefill_step(
        &self,
        state: &mut VoxtralTtsRetainedState,
        cache: &mut PhysicalPagedKvCache,
        _checkpoint: &VoxtralTtsQuantumCheckpoint,
        max_tokens: usize,
    ) -> Result<VoxtralTtsPrefillStep> {
        let pipeline = self.pipeline.as_ref().ok_or_else(|| {
            Error::InferenceError("Voxtral TTS retained prefill requires loaded weights".into())
        })?;
        if state.phase != VoxtralTtsRetainedPhase::Prefill || max_tokens == 0 {
            return Err(Error::InvalidInput(
                "Voxtral TTS retained prefill quantum is invalid".into(),
            ));
        }
        let consumed = max_tokens.min(state.artifact.prompt_tokens - state.prefill_cursor);
        let input = state
            .artifact
            .prompt_embeddings
            .narrow(1, state.prefill_cursor, consumed)?;
        let hidden = pipeline.language_model.forward_managed_hidden_with_embeds(
            &input,
            state.lm_position,
            cache,
            None,
            None,
        )?;
        state.prefill_cursor += consumed;
        state.lm_position += consumed;
        if state.prefill_cursor == state.artifact.prompt_tokens {
            state.last_hidden = Some(last_sequence_hidden(
                &hidden,
                "Voxtral TTS retained prefill",
            )?);
            state.phase = VoxtralTtsRetainedPhase::Decode;
        }
        Ok(VoxtralTtsPrefillStep {
            consumed_tokens: consumed,
            prefill_cursor: state.prefill_cursor,
            prompt_tokens: state.artifact.prompt_tokens,
            complete: state.phase == VoxtralTtsRetainedPhase::Decode,
        })
    }

    /// Advance a ragged retained prompt cohort in one-token wavefronts. Each
    /// wave with more than one active row is one physical paged LM launch.
    pub(crate) fn retained_prefill_batch(
        &self,
        states: &mut [&mut VoxtralTtsRetainedState],
        caches: &mut [&mut PhysicalPagedKvCache],
        checkpoints: &[&VoxtralTtsQuantumCheckpoint],
        max_tokens: usize,
    ) -> Result<VoxtralTtsPrefillBatch> {
        if states.is_empty()
            || states.len() != caches.len()
            || states.len() != checkpoints.len()
            || max_tokens == 0
        {
            return Err(Error::InvalidInput(
                "Voxtral TTS retained prefill batch widths are invalid".into(),
            ));
        }
        if states
            .iter()
            .any(|state| state.phase != VoxtralTtsRetainedPhase::Prefill)
        {
            return Err(Error::InvalidInput(
                "Voxtral TTS retained prefill cohort contains a non-prefill row".into(),
            ));
        }
        let pipeline = self.pipeline.as_ref().ok_or_else(|| {
            Error::InferenceError("Voxtral TTS retained prefill requires loaded weights".into())
        })?;
        let mut consumed = vec![0usize; states.len()];
        let mut max_lm_launch_width = 0usize;
        let mut scalar_lm_launches = 0usize;
        loop {
            let active = states
                .iter()
                .enumerate()
                .filter_map(|(row, state)| {
                    (consumed[row] < max_tokens
                        && state.prefill_cursor < state.artifact.prompt_tokens)
                        .then_some(row)
                })
                .collect::<Vec<_>>();
            if active.is_empty() {
                break;
            }
            let embeddings = active
                .iter()
                .map(|&row| {
                    states[row]
                        .artifact
                        .prompt_embeddings
                        .narrow(1, states[row].prefill_cursor, 1)
                })
                .collect::<candle_core::Result<Vec<_>>>()?;
            let embeds = Tensor::cat(&embeddings.iter().collect::<Vec<_>>(), 0)?;
            let positions = active
                .iter()
                .map(|&row| states[row].lm_position)
                .collect::<Vec<_>>();
            let mut cache_refs = caches
                .iter_mut()
                .enumerate()
                .filter_map(|(row, cache)| active.contains(&row).then_some(&mut **cache))
                .collect::<Vec<_>>();
            let hidden = pipeline
                .language_model
                .forward_managed_decode_batch_hidden_with_embeds(
                    &embeds,
                    &positions,
                    &mut cache_refs,
                    None,
                )?;
            max_lm_launch_width = max_lm_launch_width.max(active.len());
            scalar_lm_launches += usize::from(active.len() == 1);
            for (wave_row, &state_row) in active.iter().enumerate() {
                let state = &mut states[state_row];
                state.prefill_cursor += 1;
                state.lm_position += 1;
                consumed[state_row] += 1;
                state.last_hidden = Some(last_sequence_hidden(
                    &hidden.narrow(0, wave_row, 1)?,
                    "Voxtral TTS retained batched prefill",
                )?);
                if state.prefill_cursor == state.artifact.prompt_tokens {
                    state.phase = VoxtralTtsRetainedPhase::Decode;
                }
            }
        }
        let steps = states
            .iter()
            .enumerate()
            .map(|(row, state)| VoxtralTtsPrefillStep {
                consumed_tokens: consumed[row],
                prefill_cursor: state.prefill_cursor,
                prompt_tokens: state.artifact.prompt_tokens,
                complete: state.phase == VoxtralTtsRetainedPhase::Decode,
            })
            .collect();
        Ok(VoxtralTtsPrefillBatch {
            steps,
            max_lm_launch_width,
            scalar_lm_launches,
        })
    }

    pub(crate) fn retained_decode_step(
        &self,
        state: &mut VoxtralTtsRetainedState,
        cache: &mut PhysicalPagedKvCache,
        _checkpoint: &VoxtralTtsQuantumCheckpoint,
    ) -> Result<VoxtralTtsDecodeStep> {
        let pipeline = self.pipeline.as_ref().ok_or_else(|| {
            Error::InferenceError("Voxtral TTS retained decode requires loaded weights".into())
        })?;
        if state.phase != VoxtralTtsRetainedPhase::Decode {
            return Err(Error::InvalidInput(
                "Voxtral TTS retained decode is not active".into(),
            ));
        }
        let hidden = state.last_hidden.as_ref().ok_or_else(|| {
            Error::InferenceError("Voxtral TTS retained decode has no hidden state".into())
        })?;
        let generated = pipeline
            .acoustic_transformer
            .forward_audio_codes_with_feedback_tensor(
                hidden,
                state.params.cfg_alpha,
                state.params.n_decoding_steps,
                !state.frames.is_empty(),
            )?;
        let feedback_tensor = generated.shifted_code_tensor;
        let frame = generated.frames.into_iter().next().ok_or_else(|| {
            Error::InferenceError("Voxtral TTS acoustic decode returned no frame".into())
        })?;
        if frame.first().copied() == Some(AudioSpecialToken::End.id()) {
            state.phase = VoxtralTtsRetainedPhase::Codec;
            return Ok(VoxtralTtsDecodeStep {
                frame: None,
                frames_generated: state.frames.len(),
                finished: true,
            });
        }
        state.frames.push(frame.clone());
        if state.frames.len() >= state.params.max_frames {
            state.phase = VoxtralTtsRetainedPhase::Codec;
            return Ok(VoxtralTtsDecodeStep {
                frame: Some(frame),
                frames_generated: state.frames.len(),
                finished: true,
            });
        }
        let next_embed = match feedback_tensor.as_ref() {
            Some(codes) => pipeline
                .audio_embeddings
                .embedding_for_shifted_code_tensor(codes)?,
            None => pipeline
                .audio_embeddings
                .embedding_for_shifted_codes(&frame)?,
        };
        let hidden = pipeline.language_model.forward_managed_hidden_with_embeds(
            &next_embed,
            state.lm_position,
            cache,
            None,
            None,
        )?;
        state.lm_position += 1;
        state.last_hidden = Some(last_sequence_hidden(
            &hidden,
            "Voxtral TTS retained decode",
        )?);
        Ok(VoxtralTtsDecodeStep {
            frame: Some(frame),
            frames_generated: state.frames.len(),
            finished: false,
        })
    }

    /// Decode one acoustic frame for a compatible retained cohort.
    ///
    /// Both the acoustic transformer and non-terminal LM feedback cohort are
    /// genuinely invoked at `B > 1`; width-one cohorts retain the scalar path.
    pub(crate) fn retained_decode_batch(
        &self,
        states: &mut [&mut VoxtralTtsRetainedState],
        caches: &mut [&mut PhysicalPagedKvCache],
        _checkpoints: &[&VoxtralTtsQuantumCheckpoint],
    ) -> Result<VoxtralTtsDecodeBatch> {
        if states.len() != caches.len() || states.len() != _checkpoints.len() {
            return Err(Error::InvalidInput(
                "Voxtral TTS retained decode batch widths do not match".into(),
            ));
        }
        validate_acoustic_cohort(states)?;
        if states.len() == 1 {
            let step = self.retained_decode_step(states[0], caches[0], _checkpoints[0])?;
            let scalar_lm_rows = usize::from(!step.finished);
            return Ok(VoxtralTtsDecodeBatch {
                steps: vec![step],
                acoustic_launch_width: 1,
                lm_launch_width: scalar_lm_rows,
                scalar_lm_launches: scalar_lm_rows,
            });
        }
        let pipeline = self.pipeline.as_ref().ok_or_else(|| {
            Error::InferenceError("Voxtral TTS retained decode requires loaded weights".into())
        })?;
        let hidden_rows = states
            .iter()
            .map(|state| {
                state.last_hidden.as_ref().ok_or_else(|| {
                    Error::InferenceError("Voxtral TTS retained decode has no hidden state".into())
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let hidden = Tensor::cat(&hidden_rows, 0)?;
        let first = &states[0];
        let generated = pipeline
            .acoustic_transformer
            .forward_audio_codes_with_feedback_tensor(
                &hidden,
                first.params.cfg_alpha,
                first.params.n_decoding_steps,
                !first.frames.is_empty(),
            )?;
        if generated.frames.len() != states.len() {
            return Err(Error::InferenceError(format!(
                "Voxtral TTS acoustic batch returned {} rows for width {}",
                generated.frames.len(),
                states.len()
            )));
        }

        let mut steps = Vec::with_capacity(states.len());
        let mut feedback_rows = Vec::new();
        let mut feedback_embeddings = Vec::new();
        for (row, state) in states.iter_mut().enumerate() {
            let frame = generated.frames[row].clone();
            if frame.first().copied() == Some(AudioSpecialToken::End.id()) {
                state.phase = VoxtralTtsRetainedPhase::Codec;
                steps.push(VoxtralTtsDecodeStep {
                    frame: None,
                    frames_generated: state.frames.len(),
                    finished: true,
                });
                continue;
            }
            state.frames.push(frame.clone());
            if state.frames.len() >= state.params.max_frames {
                state.phase = VoxtralTtsRetainedPhase::Codec;
                steps.push(VoxtralTtsDecodeStep {
                    frame: Some(frame),
                    frames_generated: state.frames.len(),
                    finished: true,
                });
                continue;
            }
            let next_embed = match generated.shifted_code_tensor.as_ref() {
                Some(codes) => pipeline
                    .audio_embeddings
                    .embedding_for_shifted_code_tensor(&codes.narrow(0, row, 1)?)?,
                None => pipeline
                    .audio_embeddings
                    .embedding_for_shifted_codes(&frame)?,
            };
            feedback_rows.push(row);
            feedback_embeddings.push(next_embed);
            steps.push(VoxtralTtsDecodeStep {
                frame: Some(frame),
                frames_generated: state.frames.len(),
                finished: false,
            });
        }
        let lm_launch_width = feedback_rows.len();
        let scalar_lm_launches = usize::from(lm_launch_width == 1);
        if !feedback_rows.is_empty() {
            let embeds = Tensor::cat(&feedback_embeddings.iter().collect::<Vec<_>>(), 0)?;
            let positions = feedback_rows
                .iter()
                .map(|&row| states[row].lm_position)
                .collect::<Vec<_>>();
            let mut cache_refs = caches
                .iter_mut()
                .enumerate()
                .filter_map(|(row, cache)| feedback_rows.contains(&row).then_some(&mut **cache))
                .collect::<Vec<_>>();
            let hidden = pipeline
                .language_model
                .forward_managed_decode_batch_hidden_with_embeds(
                    &embeds,
                    &positions,
                    &mut cache_refs,
                    None,
                )?;
            for (batch_row, &state_row) in feedback_rows.iter().enumerate() {
                let state = &mut states[state_row];
                state.lm_position += 1;
                state.last_hidden = Some(last_sequence_hidden(
                    &hidden.narrow(0, batch_row, 1)?,
                    "Voxtral TTS retained batched decode",
                )?);
            }
        }
        Ok(VoxtralTtsDecodeBatch {
            steps,
            acoustic_launch_width: states.len(),
            lm_launch_width,
            scalar_lm_launches,
        })
    }

    pub(crate) fn retained_codec_finalize(
        &self,
        state: &mut VoxtralTtsRetainedState,
    ) -> Result<VoxtralTtsOutput> {
        let pipeline = self.pipeline.as_ref().ok_or_else(|| {
            Error::InferenceError("Voxtral TTS retained codec requires loaded weights".into())
        })?;
        if state.phase != VoxtralTtsRetainedPhase::Codec || state.frames.is_empty() {
            return Err(Error::InvalidInput(
                "Voxtral TTS retained codec is not ready".into(),
            ));
        }
        let frames_generated = state.frames.len();
        let timeline = VoxtralCodecTimeline::new(frames_to_codebooks(state.frames.clone())?)?;
        let samples = pipeline.codec_decoder.decode_timeline(&timeline)?;
        state.phase = VoxtralTtsRetainedPhase::Finished;
        Ok(VoxtralTtsOutput {
            samples,
            sample_rate: self.codec_config.sample_rate,
            frames_generated,
        })
    }
}

impl VoxtralTtsPipeline {
    fn generate(
        &self,
        text: &str,
        voice: &str,
        params: VoxtralTtsGenerationParams,
        model: &VoxtralTtsModel,
        cache: &mut PhysicalPagedKvCache,
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
        let required_cache_tokens = prompt
            .input_ids
            .len()
            .checked_add(max_frames.saturating_sub(1))
            .ok_or_else(|| Error::InvalidInput("Voxtral TTS cache length overflow".into()))?;
        if cache.context_len() != 0 {
            return Err(Error::InvalidInput(
                "Voxtral TTS invocation cache must start empty".into(),
            ));
        }
        if required_cache_tokens > cache.capacity_tokens() {
            return Err(Error::InvalidInput(format!(
                "Voxtral TTS needs {required_cache_tokens} cache tokens, but its invocation lease has capacity {}",
                cache.capacity_tokens()
            )));
        }
        let lm_prefill_start = Instant::now();
        let prefill_hidden = self
            .language_model
            .forward_managed_hidden_with_embeds(&prompt_embeds, 0, cache, None, None)
            .map_err(|err| {
                Error::InferenceError(format!("Voxtral TTS LM prefill failed: {err}"))
            })?;
        let lm_prefill_duration = lm_prefill_start.elapsed();
        let mut pos = prompt.input_ids.len();
        let mut last_hidden = last_sequence_hidden(&prefill_hidden, "Voxtral TTS LM prefill")?;
        let mut frames = Vec::new();
        let mut acoustic_duration = Duration::ZERO;
        let mut lm_decode_duration = Duration::ZERO;
        let mut tensor_feedback_frames = 0usize;
        let mut host_feedback_frames = 0usize;

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
                tensor_feedback_frames += 1;
                self.audio_embeddings
                    .embedding_for_shifted_code_tensor(feedback_code_tensor)?
            } else {
                host_feedback_frames += 1;
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
                .forward_managed_hidden_with_embeds(&next_embed, pos, cache, None, None)
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
            "Voxtral TTS timings: frames={}, samples={}, dense_decode_tokens={}, tensor_feedback_frames={}, host_feedback_frames={}, prompt={:.2}ms, lm_prefill={:.2}ms, acoustic={:.2}ms, lm_decode={:.2}ms, codec={:.2}ms, total={:.2}ms",
            frames_generated,
            samples.len(),
            0,
            tensor_feedback_frames,
            host_feedback_frames,
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
    sizes.extend(std::iter::repeat_n(
        acoustic_size,
        config.n_acoustic_codebooks(),
    ));
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
