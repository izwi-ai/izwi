use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use candle_core::{IndexOp, Tensor};
use tracing::info;

use crate::backends::state::TensorStateArena;
use crate::backends::{BackendKind, DeviceProfile};
use crate::engine::{InvocationTensorLease, StageDescriptor};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::shared::attention::physical::PhysicalPagedKvCache;
use crate::models::shared::chat::{ChatMessage, ChatRole};

use super::asr_retained::{
    Lfm25AudioAsrDecodeStep, Lfm25AudioAsrPrefillStep, Lfm25AudioAsrQuantumCheckpoint,
    Lfm25AudioAsrRetainedState,
};
use super::audio_output::{Lfm25AudioHead, Lfm25SampledAudioFrame};
use super::bundle::{Lfm25AudioBundle, Lfm25AudioBundleInfo};
use super::config::{
    parse_audio_decoder_config, parse_audio_encoder_config, parse_detokenizer_config,
    parse_main_backbone_config, Lfm25AudioDecoderConfig, Lfm25AudioEncoderConfig,
    Lfm2BackboneConfig,
};
use super::conformer::subsampled_len_3x;
use super::conformer::Lfm25AudioEncoder;
use super::detokenizer::Lfm25AudioDetokenizer;
use super::physical::{
    lfm25_audio_physical_state_spec, lfm25_audio_retained_state_spec, Lfm25AudioPhysicalStateSpec,
    Lfm25AudioRetainedStateSpec, Lfm25AudioStateMode,
};
use super::preprocessor::Lfm25AudioPreprocessor;
use super::sampling::{
    greedy_from_logits, greedy_token_tensor_from_logits, sample_from_logits,
    Lfm25AudioGenerationConfig, SimpleRng,
};
use super::state::{Lfm25AudioRetainedCheckpoint, Lfm25AudioRetainedMode, Lfm25AudioRetainedState};
use super::tokenizer::{Lfm25SpecialTokenIds, Lfm25TextTokenizer};
use super::LFM25_AUDIO_DEFAULT_INTERLEAVED_SYSTEM_PROMPT;
use crate::models::architectures::lfm2::backbone::QuantizedLfm2Backbone;

const DEFAULT_AUDIO_STREAM_DECODE_STRIDE_FRAMES: usize = 6;
const DEFAULT_AUDIO_STREAM_HOLDBACK_FRAMES: usize = 2;
const DEFAULT_ASR_STOP_CHECK_INTERVAL: usize = 92;
const DEFAULT_TTS_AUDIO_STOP_CHECK_INTERVAL: usize = 20;
static NEXT_LFM25_AUDIO_MODEL_LOAD_NONCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct Lfm25AudioTextOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub tokens_generated: usize,
    pub diagnostics: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct Lfm25AudioGenerationOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub tokens_generated: usize,
    pub audio_frames_generated: usize,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub diagnostics: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub(crate) struct Lfm25AudioPreparedAsrArtifact {
    model_load_nonce: u64,
    pub(crate) prompt_embeddings: Tensor,
    pub(crate) prefix_tokens: usize,
    pub(crate) suffix_tokens: usize,
    pub(crate) source_samples: usize,
    pub(crate) source_sample_rate: u32,
    pub(crate) resampled_samples: usize,
    pub(crate) mel_frames: usize,
    pub(crate) effective_feature_frames: usize,
    pub(crate) audio_tokens: usize,
    pub(crate) prompt_tokens: usize,
    /// Retained mixed-prompt tensor elements only.
    pub(crate) materialized_tensor_elements: u64,
    pub(crate) retained_resident_bytes: u64,
    pub(crate) retained_host_bytes: u64,
    preparation_timings: Lfm25AsrPreparationTimings,
}

impl Lfm25AudioPreparedAsrArtifact {
    pub(crate) const fn model_load_nonce(&self) -> u64 {
        self.model_load_nonce
    }

    pub(crate) fn hidden_size(&self) -> Result<usize> {
        self.prompt_embeddings.dim(2).map_err(Error::from)
    }

    pub(crate) fn prompt_slice(&self, start: usize, tokens: usize) -> Result<Tensor> {
        let end = start.checked_add(tokens).ok_or_else(|| {
            Error::InvalidInput("LFM2.5 Audio retained ASR prefill range overflowed".into())
        })?;
        if tokens == 0 || end > self.prompt_tokens {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio retained ASR prefill range is outside the prompt".into(),
            ));
        }
        self.prompt_embeddings
            .narrow(1, start, tokens)
            .map_err(Error::from)
    }

    pub(crate) fn device(&self) -> &candle_core::Device {
        self.prompt_embeddings.device()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Lfm25AudioAsrPreparationGeometry {
    pub(crate) source_samples: usize,
    pub(crate) source_sample_rate: u32,
    pub(crate) resampled_samples: usize,
    pub(crate) padded_samples: usize,
    pub(crate) total_mel_frames: usize,
    pub(crate) effective_feature_frames: usize,
    pub(crate) encoder_frames: usize,
    pub(crate) prompt_tokens: usize,
    /// Useful unpadded feature input plus the retained mixed prompt.
    pub(crate) preparation_useful_tensor_elements: u64,
    /// Retained mixed-prompt tensor elements only.
    pub(crate) materialized_tensor_elements: u64,
    pub(crate) retained_resident_bytes: u64,
    pub(crate) host_workspace_bytes: u64,
    pub(crate) device_workspace_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Lfm25AudioAsrPreparationResourceEnvelope {
    pub(crate) backend: BackendKind,
    pub(crate) geometry: Lfm25AudioAsrPreparationGeometry,
    pub(crate) max_work_units: u64,
    pub(crate) max_materialized_tensor_elements: u64,
    pub(crate) max_retained_resident_bytes: u64,
    pub(crate) max_host_workspace_bytes: u64,
    pub(crate) max_device_workspace_bytes: u64,
    pub(crate) max_unified_workspace_bytes: u64,
    pub(crate) max_workspace_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Lfm25AudioAsrPreparationStageCeiling {
    pub(crate) backend: BackendKind,
    pub(crate) max_source_samples: usize,
    pub(crate) max_source_sample_rate: u32,
    pub(crate) max_resampled_samples: usize,
    pub(crate) max_prompt_tokens: usize,
    pub(crate) max_work_units: u64,
    pub(crate) max_materialized_tensor_elements: u64,
    pub(crate) max_retained_resident_bytes: u64,
    pub(crate) max_host_workspace_bytes: u64,
    pub(crate) max_device_workspace_bytes: u64,
    pub(crate) max_unified_workspace_bytes: u64,
    pub(crate) max_workspace_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Lfm25AudioAsrStepResourceEnvelope {
    pub(crate) backend: BackendKind,
    pub(crate) work_units: u64,
    pub(crate) materialized_tensor_elements: u64,
    pub(crate) host_workspace_bytes: u64,
    pub(crate) device_workspace_bytes: u64,
    pub(crate) unified_workspace_bytes: u64,
    pub(crate) workspace_bytes: u64,
}

const LFM25_AUDIO_MAX_SOURCE_SAMPLE_RATE: u32 = 192_000;

#[derive(Debug, Clone, Copy, Default)]
struct Lfm25AsrPreparationTimings {
    resample_ms: f64,
    feature_extract_ms: f64,
    encoder_forward_ms: f64,
    prompt_build_ms: f64,
    prompt_embed_ms: f64,
    prompt_concat_ms: f64,
    total_ms: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Lfm25AudioStreamConfig {
    pub decode_stride_frames: usize,
    pub holdback_frames: usize,
}

impl Default for Lfm25AudioStreamConfig {
    fn default() -> Self {
        Self {
            decode_stride_frames: DEFAULT_AUDIO_STREAM_DECODE_STRIDE_FRAMES,
            holdback_frames: DEFAULT_AUDIO_STREAM_HOLDBACK_FRAMES,
        }
    }
}

pub struct Lfm25AudioModel {
    model_load_nonce: u64,
    device: DeviceProfile,
    bundle_info: Lfm25AudioBundleInfo,
    tokenizer: Lfm25TextTokenizer,
    main_config: Lfm2BackboneConfig,
    detokenizer_config: Lfm2BackboneConfig,
    encoder_config: Lfm25AudioEncoderConfig,
    decoder_config: Lfm25AudioDecoderConfig,
    preprocessor: Lfm25AudioPreprocessor,
    encoder: Lfm25AudioEncoder,
    audio_head: Lfm25AudioHead,
    detokenizer: Lfm25AudioDetokenizer,
    main_backbone: Mutex<QuantizedLfm2Backbone>,
}

#[derive(Debug, Clone, Copy, Default)]
struct Lfm25AsrProfile {
    main_prefill_ms: f64,
    decode_loop_ms: f64,
    decode_argmax_ms: f64,
    decode_host_read_ms: f64,
    decode_token_tensor_ms: f64,
    decode_forward_ms: f64,
    tokenizer_decode_ms: f64,
    token_select_reads: u64,
    host_token_reads: u64,
    host_read_chunks: u64,
    device_token_steps: u64,
    token_repetition_loop: bool,
    text_repetition_loop: bool,
    stop_reason: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, Default)]
struct Lfm25TtsProfile {
    prompt_embed_ms: f64,
    main_prefill_ms: f64,
    text_sampling_ms: f64,
    tokenizer_decode_ms: f64,
    text_forward_ms: f64,
    audio_head_ms: f64,
    audio_head_depth_linear_ms: f64,
    audio_head_depth_reshape_ms: f64,
    audio_head_cache_setup_ms: f64,
    audio_head_codebook_input_ms: f64,
    audio_head_depthformer_ms: f64,
    audio_head_sample_ms: f64,
    audio_head_embed_step_ms: f64,
    audio_head_materialize_ms: f64,
    audio_head_materialize_pack_ms: f64,
    audio_head_materialize_readback_ms: f64,
    audio_embed_ms: f64,
    audio_forward_ms: f64,
    detokenizer_embedding_ms: f64,
    detokenizer_upsample_ms: f64,
    detokenizer_backbone_ms: f64,
    detokenizer_projection_ms: f64,
    detokenizer_waveform_prepare_ms: f64,
    detokenizer_readback_ms: f64,
    detokenizer_istft_ms: f64,
    audio_head_calls: u64,
    audio_head_codebook_steps: u64,
    text_sample_calls: u64,
}

impl Lfm25AudioModel {
    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        if !matches!(variant, ModelVariant::Lfm25Audio15BGguf) {
            return Err(Error::ModelLoadError(format!(
                "Unsupported LFM2.5 Audio variant: {variant}"
            )));
        }

        let backend = BackendKind::from(device.kind);
        let bundle = Lfm25AudioBundle::load(model_dir, backend)?;
        let bundle_info = bundle.info();

        let tokenizer = Lfm25TextTokenizer::load(&bundle.main)?;
        let main_config = parse_main_backbone_config(&bundle.main)?;
        let detokenizer_config = parse_detokenizer_config(&bundle.tokenizer)?;
        let encoder_config = parse_audio_encoder_config(&bundle.mmproj)?;
        let decoder_config = parse_audio_decoder_config(&bundle.vocoder)?;
        let preprocessor = Lfm25AudioPreprocessor::load()?;

        let main_backbone =
            QuantizedLfm2Backbone::load(&bundle.main, main_config.clone(), &device.device)?;
        let encoder =
            Lfm25AudioEncoder::load(&bundle.mmproj, encoder_config.clone(), &device.device)?;
        let audio_head = Lfm25AudioHead::load(
            &bundle.vocoder,
            &decoder_config,
            main_config.embedding_length,
            &device.device,
        )?;
        let detokenizer = Lfm25AudioDetokenizer::load(
            &bundle.tokenizer,
            &bundle.vocoder,
            detokenizer_config.clone(),
            &decoder_config,
            &device.device,
        )?;

        info!(
            "Loaded LFM2.5 Audio GGUF bundle on {:?} from {}",
            device.kind,
            model_dir.display()
        );

        Ok(Self {
            model_load_nonce: NEXT_LFM25_AUDIO_MODEL_LOAD_NONCE.fetch_add(1, Ordering::Relaxed),
            device,
            bundle_info,
            tokenizer,
            main_config,
            detokenizer_config,
            encoder_config,
            decoder_config,
            preprocessor,
            encoder,
            audio_head,
            detokenizer,
            main_backbone: Mutex::new(main_backbone),
        })
    }

    pub fn device(&self) -> &DeviceProfile {
        &self.device
    }

    pub fn bundle_info(&self) -> &Lfm25AudioBundleInfo {
        &self.bundle_info
    }

    pub fn tokenizer(&self) -> &Lfm25TextTokenizer {
        &self.tokenizer
    }

    pub fn main_config(&self) -> &Lfm2BackboneConfig {
        &self.main_config
    }

    pub fn detokenizer_config(&self) -> &Lfm2BackboneConfig {
        &self.detokenizer_config
    }

    pub fn encoder_config(&self) -> &Lfm25AudioEncoderConfig {
        &self.encoder_config
    }

    pub fn decoder_config(&self) -> &Lfm25AudioDecoderConfig {
        &self.decoder_config
    }

    pub(crate) fn physical_state_spec(
        &self,
        mode: Lfm25AudioStateMode,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<Lfm25AudioPhysicalStateSpec> {
        lfm25_audio_physical_state_spec(&self.main_config, &self.decoder_config, mode, stage_graphs)
    }

    pub(crate) fn retained_state_spec(
        &self,
        mode: Lfm25AudioRetainedMode,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<Lfm25AudioRetainedStateSpec> {
        lfm25_audio_retained_state_spec(&self.main_config, &self.decoder_config, mode, stage_graphs)
    }

    pub(crate) fn new_retained_state(
        &self,
        mode: Lfm25AudioRetainedMode,
    ) -> Result<Lfm25AudioRetainedState> {
        let backbone = self
            .main_backbone
            .lock()
            .map_err(|_| Error::InferenceError("LFM2.5 Audio backbone mutex poisoned".into()))?;
        Ok(Lfm25AudioRetainedState::new(
            mode,
            backbone.new_shortconv_state(),
            self.decoder_config.codebooks,
        ))
    }

    pub(crate) fn new_retained_asr_state(
        &self,
        artifact: Arc<Lfm25AudioPreparedAsrArtifact>,
        requested_max_new_tokens: usize,
    ) -> Result<Lfm25AudioAsrRetainedState> {
        validate_prepared_asr_identity(self.model_load_nonce, artifact.model_load_nonce())?;
        validate_prepared_asr_prompt_shape(
            artifact.prompt_embeddings.dims(),
            artifact.prompt_tokens,
            self.main_config.embedding_length,
        )?;
        Lfm25AudioAsrRetainedState::new(
            artifact,
            self.new_retained_state(Lfm25AudioRetainedMode::Asr)?,
            self.model_load_nonce,
            self.tokenizer.vocab_size(),
            self.tokenizer.specials().clone(),
            requested_max_new_tokens,
            self.main_config.context_length,
        )
    }

    pub(crate) fn retained_asr_prefill_step(
        &self,
        state: &mut Lfm25AudioAsrRetainedState,
        cache: &mut PhysicalPagedKvCache,
        checkpoint: &Lfm25AudioAsrQuantumCheckpoint,
        max_tokens: usize,
    ) -> Result<Lfm25AudioAsrPrefillStep> {
        self.with_main_backbone(|backbone| {
            state.prefill_step(backbone, cache, checkpoint, max_tokens)
        })
    }

    pub(crate) fn retained_asr_decode_step(
        &self,
        state: &mut Lfm25AudioAsrRetainedState,
        cache: &mut PhysicalPagedKvCache,
        checkpoint: &Lfm25AudioAsrQuantumCheckpoint,
    ) -> Result<Lfm25AudioAsrDecodeStep> {
        self.with_main_backbone(|backbone| {
            state.decode_step(backbone, &self.tokenizer, cache, checkpoint)
        })
    }

    pub(crate) fn retained_asr_decode_will_append(
        &self,
        state: &Lfm25AudioAsrRetainedState,
    ) -> Result<bool> {
        state.decode_will_append(&self.tokenizer)
    }

    pub(crate) fn retained_asr_decode_append_batch(
        &self,
        states: &mut [&mut Lfm25AudioAsrRetainedState],
        caches: &mut [&mut PhysicalPagedKvCache],
        checkpoints: &[&Lfm25AudioAsrQuantumCheckpoint],
    ) -> Result<Vec<Lfm25AudioAsrDecodeStep>> {
        self.with_main_backbone(|backbone| {
            Lfm25AudioAsrRetainedState::decode_append_batch(
                backbone,
                &self.tokenizer,
                states,
                caches,
                checkpoints,
            )
        })
    }

    pub(crate) fn begin_retained_main_quantum(
        &self,
        state: &mut Lfm25AudioRetainedState,
        main: &PhysicalPagedKvCache,
    ) -> Result<Lfm25AudioRetainedCheckpoint> {
        state.begin_main_quantum(main)
    }

    pub(crate) fn begin_retained_depthformer_quantum(
        &self,
        state: &mut Lfm25AudioRetainedState,
        main: &PhysicalPagedKvCache,
        depthformer: &PhysicalPagedKvCache,
    ) -> Result<Lfm25AudioRetainedCheckpoint> {
        state.begin_depthformer_quantum(main, depthformer)
    }

    pub(crate) fn commit_retained_quantum(
        &self,
        state: &mut Lfm25AudioRetainedState,
        main: &PhysicalPagedKvCache,
        depthformer: Option<&PhysicalPagedKvCache>,
        checkpoint: &Lfm25AudioRetainedCheckpoint,
    ) -> Result<()> {
        state.commit_quantum(main, depthformer, checkpoint)
    }

    pub(crate) fn rollback_retained_quantum(
        &self,
        state: &mut Lfm25AudioRetainedState,
        main: &mut PhysicalPagedKvCache,
        depthformer: Option<&mut PhysicalPagedKvCache>,
        checkpoint: &Lfm25AudioRetainedCheckpoint,
    ) -> Result<()> {
        state.rollback_quantum(main, depthformer, checkpoint)
    }

    pub(crate) fn bind_retained_tensor_sequence(
        &self,
        state: &mut Lfm25AudioRetainedState,
        sequence: u64,
    ) -> Result<()> {
        state.bind_tensor_sequence(sequence)
    }

    pub(crate) fn restore_retained_shortconv(
        &self,
        state: &mut Lfm25AudioRetainedState,
        arena: &TensorStateArena,
    ) -> Result<()> {
        state.restore_shortconv(arena)
    }

    pub(crate) fn stage_retained_shortconv(
        &self,
        state: &mut Lfm25AudioRetainedState,
        arena: &TensorStateArena,
        transaction: u64,
    ) -> Result<()> {
        state.stage_shortconv(arena, transaction)
    }

    pub(crate) fn reset_retained_depthformer_frame(
        &self,
        state: &mut Lfm25AudioRetainedState,
        depthformer: &mut PhysicalPagedKvCache,
    ) -> Result<()> {
        state.reset_depthformer_frame(depthformer)
    }

    pub(crate) fn transcribe_to_output_with_callback_physical(
        &self,
        audio: &[f32],
        sample_rate: u32,
        max_new_tokens: usize,
        cache: &mut PhysicalPagedKvCache,
        shortconv: &mut InvocationTensorLease,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<Lfm25AudioTextOutput> {
        let prepared = self.prepare_asr_artifact(audio, sample_rate)?;
        self.transcribe_prepared_asr_with_callback_physical(
            &prepared,
            max_new_tokens,
            cache,
            shortconv,
            on_delta,
        )
    }

    pub(crate) fn prepare_asr_artifact(
        &self,
        audio: &[f32],
        sample_rate: u32,
    ) -> Result<Arc<Lfm25AudioPreparedAsrArtifact>> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }
        if sample_rate == 0 {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio sample rate must be non-zero".to_string(),
            ));
        }
        let ceiling = self.asr_preparation_stage_ceiling()?;
        let envelope = self.asr_preparation_resource_envelope(audio.len(), sample_rate)?;

        let total_started = Instant::now();
        let resample_started = Instant::now();
        let mono_16khz = if sample_rate == super::config::LFM25_AUDIO_INPUT_SAMPLE_RATE {
            audio.to_vec()
        } else {
            resample_linear(
                audio,
                sample_rate,
                super::config::LFM25_AUDIO_INPUT_SAMPLE_RATE,
            )?
        };
        let resample_ms = elapsed_ms(resample_started);

        let feature_started = Instant::now();
        let (features, feature_frames) = self
            .preprocessor
            .compute_features(&mono_16khz, &self.device.device)?;
        let feature_extract_ms = elapsed_ms(feature_started);

        let encoder_started = Instant::now();
        let audio_embeds = self.encoder.encode(&features, feature_frames)?;
        let encoder_forward_ms = elapsed_ms(encoder_started);
        let audio_tokens = audio_embeds.dim(1)?;

        let prompt_started = Instant::now();
        let (prefix_ids, suffix_ids) = self.build_asr_prompt_segments()?;
        let prompt_build_ms = elapsed_ms(prompt_started);
        let prompt_tokens = checked_asr_prompt_tokens(
            prefix_ids.len(),
            audio_tokens,
            suffix_ids.len(),
            self.main_config.context_length,
        )?;
        let prompt_embed_started = Instant::now();
        let (prefix_embeds, suffix_embeds) = self.with_main_backbone(|main_backbone| {
            Ok((
                embed_token_ids(main_backbone, &self.device.device, &prefix_ids)?,
                embed_token_ids(main_backbone, &self.device.device, &suffix_ids)?,
            ))
        })?;
        let prompt_embed_ms = elapsed_ms(prompt_embed_started);

        let prompt_concat_started = Instant::now();
        let prompt_embeddings = Tensor::cat(&[&prefix_embeds, &audio_embeds, &suffix_embeds], 1)?;
        validate_prepared_asr_prompt_shape(
            prompt_embeddings.dims(),
            prompt_tokens,
            self.main_config.embedding_length,
        )?;
        let prompt_concat_ms = elapsed_ms(prompt_concat_started);
        let materialized_tensor_elements =
            checked_asr_prompt_tensor_elements(prompt_tokens, self.main_config.embedding_length)?;
        if u64::try_from(prompt_embeddings.elem_count()).ok() != Some(materialized_tensor_elements)
        {
            return Err(Error::InferenceError(
                "Prepared LFM2.5 Audio prompt tensor accounting does not match its shape"
                    .to_string(),
            ));
        }
        let retained_resident_bytes = materialized_tensor_elements
            .checked_mul(
                u64::try_from(prompt_embeddings.dtype().size_in_bytes()).map_err(|_| {
                    Error::InvalidInput("LFM2.5 Audio prompt dtype size exceeds u64".to_string())
                })?,
            )
            .ok_or_else(|| {
                Error::InvalidInput("LFM2.5 Audio retained tensor bytes overflowed u64".to_string())
            })?;
        if mono_16khz.len() != envelope.geometry.resampled_samples
            || feature_frames != envelope.geometry.effective_feature_frames
            || prompt_tokens != envelope.geometry.prompt_tokens
            || audio_tokens != envelope.geometry.encoder_frames
            || materialized_tensor_elements != envelope.geometry.materialized_tensor_elements
            || retained_resident_bytes > envelope.max_retained_resident_bytes
            || envelope.max_workspace_bytes > ceiling.max_workspace_bytes
        {
            return Err(Error::InferenceError(
                "Prepared LFM2.5 Audio artifact exceeded its authenticated preparation seal"
                    .to_string(),
            ));
        }

        Ok(Arc::new(Lfm25AudioPreparedAsrArtifact {
            model_load_nonce: self.model_load_nonce,
            prompt_embeddings,
            prefix_tokens: prefix_ids.len(),
            suffix_tokens: suffix_ids.len(),
            source_samples: audio.len(),
            source_sample_rate: sample_rate,
            resampled_samples: mono_16khz.len(),
            mel_frames: envelope.geometry.total_mel_frames,
            effective_feature_frames: feature_frames,
            audio_tokens,
            prompt_tokens,
            materialized_tensor_elements,
            retained_resident_bytes,
            retained_host_bytes: prepared_asr_retained_host_bytes(),
            preparation_timings: Lfm25AsrPreparationTimings {
                resample_ms,
                feature_extract_ms,
                encoder_forward_ms,
                prompt_build_ms,
                prompt_embed_ms,
                prompt_concat_ms,
                total_ms: elapsed_ms(total_started),
            },
        }))
    }

    pub(crate) fn asr_preparation_geometry(
        &self,
        source_samples: usize,
        source_sample_rate: u32,
    ) -> Result<Lfm25AudioAsrPreparationGeometry> {
        if source_samples == 0 || source_sample_rate == 0 {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio preparation requires non-empty audio and a non-zero sample rate"
                    .to_string(),
            ));
        }
        if source_sample_rate > LFM25_AUDIO_MAX_SOURCE_SAMPLE_RATE {
            return Err(Error::InvalidInput(format!(
                "LFM2.5 Audio source sample rate {source_sample_rate} exceeds the sealed maximum {LFM25_AUDIO_MAX_SOURCE_SAMPLE_RATE}"
            )));
        }
        let resampled_samples = checked_resampled_len(
            source_samples,
            source_sample_rate,
            super::config::LFM25_AUDIO_INPUT_SAMPLE_RATE,
        )?;
        let padded_samples = resampled_samples
            .checked_add(super::config::LFM25_AUDIO_INPUT_N_FFT)
            .ok_or_else(|| Error::InvalidInput("LFM2.5 Audio padding overflowed usize".into()))?;
        let padded_feature_frames = resampled_samples
            .checked_div(super::config::LFM25_AUDIO_INPUT_HOP_LENGTH)
            .and_then(|frames| frames.checked_add(1))
            .ok_or_else(|| Error::InvalidInput("LFM2.5 Audio frame count overflowed".into()))?;
        let effective_frames = super::preprocessor::effective_frame_count(resampled_samples);
        let encoder_frames = subsampled_len_3x(effective_frames);
        let (prefix, suffix) = self.build_asr_prompt_segments()?;
        let prompt_tokens = checked_asr_prompt_tokens(
            prefix.len(),
            encoder_frames,
            suffix.len(),
            self.main_config.context_length,
        )?;
        let materialized_tensor_elements =
            checked_asr_prompt_tensor_elements(prompt_tokens, self.main_config.embedding_length)?;
        let preparation_useful_tensor_elements = u64::try_from(padded_feature_frames)
            .ok()
            .and_then(|frames| {
                frames.checked_mul(u64::try_from(self.encoder_config.num_mel_bins).ok()?)
            })
            .and_then(|features| features.checked_add(materialized_tensor_elements))
            .ok_or_else(|| {
                Error::InvalidInput(
                    "LFM2.5 Audio useful preparation elements overflowed u64".into(),
                )
            })?;
        let retained_resident_bytes =
            materialized_tensor_elements.checked_mul(4).ok_or_else(|| {
                Error::InvalidInput("LFM2.5 Audio retained resident bytes overflowed".into())
            })?;
        let (host_workspace_bytes, device_workspace_bytes) = checked_asr_preparation_workspace(
            resampled_samples,
            padded_samples,
            padded_feature_frames,
            encoder_frames,
            prompt_tokens,
            &self.encoder_config,
            self.main_config.embedding_length,
        )?;
        Ok(Lfm25AudioAsrPreparationGeometry {
            source_samples,
            source_sample_rate,
            resampled_samples,
            padded_samples,
            total_mel_frames: padded_feature_frames,
            effective_feature_frames: effective_frames,
            encoder_frames,
            prompt_tokens,
            preparation_useful_tensor_elements,
            materialized_tensor_elements,
            retained_resident_bytes,
            host_workspace_bytes,
            device_workspace_bytes,
        })
    }

    pub(crate) fn asr_preparation_resource_envelope(
        &self,
        source_samples: usize,
        source_sample_rate: u32,
    ) -> Result<Lfm25AudioAsrPreparationResourceEnvelope> {
        let backend = BackendKind::from(self.device.kind);
        let geometry = self.asr_preparation_geometry(source_samples, source_sample_rate)?;
        let (max_host_workspace_bytes, max_device_workspace_bytes, max_unified_workspace_bytes) =
            map_asr_workspace_domains(
                backend,
                geometry.host_workspace_bytes,
                geometry.device_workspace_bytes,
            )?;
        let max_workspace_bytes = geometry
            .host_workspace_bytes
            .checked_add(geometry.device_workspace_bytes)
            .ok_or_else(|| Error::InvalidInput("LFM2.5 Audio workspace bytes overflowed".into()))?;
        Ok(Lfm25AudioAsrPreparationResourceEnvelope {
            backend,
            geometry,
            max_work_units: u64::try_from(geometry.encoder_frames)
                .map_err(|_| Error::InvalidInput("LFM2.5 Audio work units exceed u64".into()))?,
            max_materialized_tensor_elements: geometry.preparation_useful_tensor_elements,
            max_retained_resident_bytes: geometry.retained_resident_bytes,
            max_host_workspace_bytes,
            max_device_workspace_bytes,
            max_unified_workspace_bytes,
            max_workspace_bytes,
        })
    }

    pub(crate) fn asr_preparation_stage_ceiling(
        &self,
    ) -> Result<Lfm25AudioAsrPreparationStageCeiling> {
        let (prefix, suffix) = self.build_asr_prompt_segments()?;
        let fixed_tokens = prefix
            .len()
            .checked_add(suffix.len())
            .ok_or_else(|| Error::InvalidInput("LFM2.5 Audio fixed prompt overflowed".into()))?;
        let max_prompt_tokens =
            self.main_config
                .context_length
                .checked_sub(1)
                .ok_or_else(|| {
                    Error::ModelLoadError(
                        "LFM2.5 Audio context leaves no ASR prompt capacity".into(),
                    )
                })?;
        let max_encoder_frames = max_prompt_tokens.checked_sub(fixed_tokens).ok_or_else(|| {
            Error::ModelLoadError("LFM2.5 Audio fixed ASR prompt exceeds model context".into())
        })?;
        if max_encoder_frames == 0 {
            return Err(Error::ModelLoadError(
                "LFM2.5 Audio context leaves no audio-token capacity".into(),
            ));
        }
        let max_effective_frames = max_encoder_frames.checked_mul(8).ok_or_else(|| {
            Error::ModelLoadError("LFM2.5 Audio maximum feature frames overflowed".into())
        })?;
        let max_resampled_samples = max_effective_frames
            .checked_add(1)
            .and_then(|frames| frames.checked_mul(super::config::LFM25_AUDIO_INPUT_HOP_LENGTH))
            .and_then(|samples| samples.checked_sub(1))
            .ok_or_else(|| {
                Error::ModelLoadError("LFM2.5 Audio maximum resampled samples overflowed".into())
            })?;
        let max_source_samples = max_resampled_samples
            .checked_mul(LFM25_AUDIO_MAX_SOURCE_SAMPLE_RATE as usize)
            .and_then(|samples| {
                samples.checked_div(super::config::LFM25_AUDIO_INPUT_SAMPLE_RATE as usize)
            })
            .ok_or_else(|| {
                Error::ModelLoadError("LFM2.5 Audio maximum source samples overflowed".into())
            })?;
        let padded_samples = max_resampled_samples
            .checked_add(super::config::LFM25_AUDIO_INPUT_N_FFT)
            .ok_or_else(|| {
                Error::ModelLoadError("LFM2.5 Audio maximum padding overflowed".into())
            })?;
        let total_mel_frames = max_resampled_samples
            .checked_div(super::config::LFM25_AUDIO_INPUT_HOP_LENGTH)
            .and_then(|frames| frames.checked_add(1))
            .ok_or_else(|| {
                Error::ModelLoadError("LFM2.5 Audio maximum mel frames overflowed".into())
            })?;
        let (logical_host, logical_device) = checked_asr_preparation_workspace(
            max_resampled_samples,
            padded_samples,
            total_mel_frames,
            max_encoder_frames,
            max_prompt_tokens,
            &self.encoder_config,
            self.main_config.embedding_length,
        )?;
        let backend = BackendKind::from(self.device.kind);
        let (max_host_workspace_bytes, max_device_workspace_bytes, max_unified_workspace_bytes) =
            map_asr_workspace_domains(backend, logical_host, logical_device)?;
        let max_workspace_bytes = logical_host.checked_add(logical_device).ok_or_else(|| {
            Error::ModelLoadError("LFM2.5 Audio maximum workspace overflowed".into())
        })?;
        let max_materialized_tensor_elements = checked_asr_prompt_tensor_elements(
            max_prompt_tokens,
            self.main_config.embedding_length,
        )?;
        let max_retained_resident_bytes = max_materialized_tensor_elements
            .checked_mul(4)
            .ok_or_else(|| {
                Error::ModelLoadError("LFM2.5 Audio resident ceiling overflowed".into())
            })?;
        let max_materialized_tensor_elements = u64::try_from(total_mel_frames)
            .ok()
            .and_then(|frames| {
                frames.checked_mul(u64::try_from(self.encoder_config.num_mel_bins).ok()?)
            })
            .and_then(|features| features.checked_add(max_materialized_tensor_elements))
            .ok_or_else(|| {
                Error::ModelLoadError("LFM2.5 Audio useful preparation ceiling overflowed".into())
            })?;
        Ok(Lfm25AudioAsrPreparationStageCeiling {
            backend,
            max_source_samples,
            max_source_sample_rate: LFM25_AUDIO_MAX_SOURCE_SAMPLE_RATE,
            max_resampled_samples,
            max_prompt_tokens,
            max_work_units: u64::try_from(max_encoder_frames).map_err(|_| {
                Error::ModelLoadError("LFM2.5 Audio maximum work units exceed u64".into())
            })?,
            max_materialized_tensor_elements,
            max_retained_resident_bytes,
            max_host_workspace_bytes,
            max_device_workspace_bytes,
            max_unified_workspace_bytes,
            max_workspace_bytes,
        })
    }

    pub(crate) fn asr_prefill_resource_envelope(
        &self,
        start: usize,
        tokens: usize,
        prompt_tokens: usize,
    ) -> Result<Lfm25AudioAsrStepResourceEnvelope> {
        let end = start
            .checked_add(tokens)
            .ok_or_else(|| Error::InvalidInput("LFM2.5 Audio prefill span overflowed".into()))?;
        if tokens == 0 || end > prompt_tokens || prompt_tokens >= self.main_config.context_length {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio prefill span is outside its sealed prompt".into(),
            ));
        }
        self.asr_main_step_resource_envelope(tokens, end, end == prompt_tokens)
    }

    pub(crate) fn asr_decode_resource_envelope(
        &self,
        position: usize,
    ) -> Result<Lfm25AudioAsrStepResourceEnvelope> {
        let visible = position
            .checked_add(1)
            .ok_or_else(|| Error::InvalidInput("LFM2.5 Audio decode position overflowed".into()))?;
        if visible > self.main_config.context_length {
            return Err(Error::InvalidInput(
                "LFM2.5 Audio decode position exceeds model context".into(),
            ));
        }
        self.asr_main_step_resource_envelope(1, visible, true)
    }

    fn asr_main_step_resource_envelope(
        &self,
        query_tokens: usize,
        visible_tokens: usize,
        include_logits: bool,
    ) -> Result<Lfm25AudioAsrStepResourceEnvelope> {
        let tensor_workspace = checked_asr_main_step_workspace(
            query_tokens,
            visible_tokens,
            self.tokenizer.vocab_size(),
            include_logits,
            &self.main_config,
        )?;
        let backend = BackendKind::from(self.device.kind);
        let (host_workspace_bytes, device_workspace_bytes, unified_workspace_bytes) =
            map_asr_workspace_domains(backend, 0, tensor_workspace)?;
        let materialized_tensor_elements = u64::try_from(query_tokens)
            .ok()
            .and_then(|tokens| {
                tokens.checked_mul(u64::try_from(self.main_config.embedding_length).ok()?)
            })
            .ok_or_else(|| Error::InvalidInput("LFM2.5 Audio step elements overflowed".into()))?;
        Ok(Lfm25AudioAsrStepResourceEnvelope {
            backend,
            work_units: u64::try_from(query_tokens)
                .map_err(|_| Error::InvalidInput("LFM2.5 Audio step work exceeds u64".into()))?,
            materialized_tensor_elements,
            host_workspace_bytes,
            device_workspace_bytes,
            unified_workspace_bytes,
            workspace_bytes: tensor_workspace,
        })
    }

    pub(crate) fn transcribe_prepared_asr_with_callback_physical(
        &self,
        prepared: &Lfm25AudioPreparedAsrArtifact,
        max_new_tokens: usize,
        cache: &mut PhysicalPagedKvCache,
        shortconv: &mut InvocationTensorLease,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<Lfm25AudioTextOutput> {
        validate_prepared_asr_identity(self.model_load_nonce, prepared.model_load_nonce)?;
        validate_prepared_asr_prompt_shape(
            prepared.prompt_embeddings.dims(),
            prepared.prompt_tokens,
            self.main_config.embedding_length,
        )?;
        let total_started = Instant::now();
        let prompt_embeds = &prepared.prompt_embeddings;
        let prompt_tokens = prepared.prompt_tokens;
        let vocab_limit = self.tokenizer.vocab_size();
        let specials = self.tokenizer.specials().clone();

        let main_started = Instant::now();
        let (mut output, profile) = self.with_main_backbone(|main_backbone| {
            let mut profile = Lfm25AsrProfile::default();
            reset_main_state(cache, shortconv)?;

            let prefill_started = Instant::now();
            let hidden =
                main_backbone.forward_embeds_physical(prompt_embeds, 0, cache, shortconv)?;
            let mut logits = main_backbone.project_last_hidden(&hidden)?;
            profile.main_prefill_ms = elapsed_ms(prefill_started);
            let mut position = prompt_tokens;
            let mut generated_ids = Vec::new();
            let mut assembled = String::new();
            let max_new_tokens = max_new_tokens
                .max(1)
                .min(self.main_config.context_length - prompt_tokens);
            let mut stop_reason = "max_tokens";
            let stop_check_interval = lfm25_asr_stop_check_interval();
            let use_deferred_device_decode = (self.device.device.is_metal()
                || self.device.device.is_cuda())
                && stop_check_interval > 1;

            let decode_started = Instant::now();
            if use_deferred_device_decode {
                while generated_ids.len() < max_new_tokens {
                    let remaining = max_new_tokens.saturating_sub(generated_ids.len());
                    let chunk_len = remaining.min(stop_check_interval);
                    let mut chunk_tokens = Vec::with_capacity(chunk_len);

                    for _ in 0..chunk_len {
                        let argmax_started = Instant::now();
                        let next_token = greedy_token_tensor_from_logits(&logits, vocab_limit)?
                            .ok_or_else(|| {
                                Error::InferenceError(
                                    "Device ASR argmax returned no token tensor".to_string(),
                                )
                            })?;
                        profile.decode_argmax_ms += elapsed_ms(argmax_started);
                        profile.token_select_reads = profile.token_select_reads.saturating_add(1);
                        profile.device_token_steps = profile.device_token_steps.saturating_add(1);

                        let token_tensor_started = Instant::now();
                        let next_tensor = next_token.reshape((1, 1))?;
                        profile.decode_token_tensor_ms += elapsed_ms(token_tensor_started);

                        let decode_forward_started = Instant::now();
                        logits = main_backbone.forward_tokens_physical(
                            &next_tensor,
                            position,
                            cache,
                            shortconv,
                        )?;
                        profile.decode_forward_ms += elapsed_ms(decode_forward_started);
                        position += 1;
                        chunk_tokens.push(next_token);
                    }

                    let read_started = Instant::now();
                    let token_refs = chunk_tokens.iter().collect::<Vec<_>>();
                    let host_tokens = Tensor::cat(&token_refs, 0)?
                        .to_vec1::<u32>()
                        .map_err(Error::from)?;
                    profile.decode_host_read_ms += elapsed_ms(read_started);
                    profile.host_read_chunks = profile.host_read_chunks.saturating_add(1);
                    profile.host_token_reads = profile
                        .host_token_reads
                        .saturating_add(u64::try_from(host_tokens.len()).unwrap_or(u64::MAX));

                    let mut should_stop = false;
                    for next in host_tokens {
                        if is_asr_stop_token(next, &specials) {
                            stop_reason = "stop_token";
                            should_stop = true;
                            break;
                        }

                        let append_status = append_asr_text_token(
                            &self.tokenizer,
                            &mut generated_ids,
                            &mut assembled,
                            next,
                            &mut profile,
                            on_delta,
                        )?;
                        if append_status.should_stop() {
                            profile.token_repetition_loop |= append_status.token_repetition_loop;
                            profile.text_repetition_loop |= append_status.text_repetition_loop;
                            stop_reason = if append_status.text_repetition_loop {
                                "text_repetition_loop"
                            } else {
                                "token_repetition_loop"
                            };
                            should_stop = true;
                            break;
                        }
                    }
                    if should_stop {
                        break;
                    }
                }
            } else {
                while generated_ids.len() < max_new_tokens {
                    let argmax_started = Instant::now();
                    let next = greedy_from_logits(&logits, vocab_limit)?;
                    profile.decode_argmax_ms += elapsed_ms(argmax_started);
                    profile.token_select_reads = profile.token_select_reads.saturating_add(1);
                    profile.host_token_reads = profile.host_token_reads.saturating_add(1);
                    if is_asr_stop_token(next, &specials) {
                        stop_reason = "stop_token";
                        break;
                    }

                    let append_status = append_asr_text_token(
                        &self.tokenizer,
                        &mut generated_ids,
                        &mut assembled,
                        next,
                        &mut profile,
                        on_delta,
                    )?;
                    if append_status.should_stop() {
                        profile.token_repetition_loop |= append_status.token_repetition_loop;
                        profile.text_repetition_loop |= append_status.text_repetition_loop;
                        stop_reason = if append_status.text_repetition_loop {
                            "text_repetition_loop"
                        } else {
                            "token_repetition_loop"
                        };
                        break;
                    }

                    let token_tensor_started = Instant::now();
                    let next_tensor = Tensor::from_vec(vec![next], (1, 1), &self.device.device)?;
                    profile.decode_token_tensor_ms += elapsed_ms(token_tensor_started);
                    let decode_forward_started = Instant::now();
                    logits = main_backbone.forward_tokens_physical(
                        &next_tensor,
                        position,
                        cache,
                        shortconv,
                    )?;
                    profile.decode_forward_ms += elapsed_ms(decode_forward_started);
                    position += 1;
                }
            }
            profile.decode_loop_ms = elapsed_ms(decode_started);
            profile.stop_reason = Some(stop_reason);

            Ok((
                Lfm25AudioTextOutput {
                    text: assembled.trim().to_string(),
                    prompt_tokens,
                    tokens_generated: generated_ids.len(),
                    diagnostics: None,
                },
                profile,
            ))
        })?;
        let main_backbone_ms = prepared.preparation_timings.prompt_embed_ms
            + prepared.preparation_timings.prompt_concat_ms
            + elapsed_ms(main_started);
        let model_total_ms = prepared.preparation_timings.total_ms + elapsed_ms(total_started);
        let device_token_select_reads = profile.device_token_steps;
        let host_argmax_reads = if device_token_select_reads > 0 {
            0
        } else {
            profile.host_token_reads
        };
        let stop_check_interval = lfm25_asr_stop_check_interval();
        let deferred_stop_check = device_token_select_reads > 0 && stop_check_interval > 1;
        output.diagnostics = Some(serde_json::json!({
            "model": "lfm25_audio",
            "task": "asr",
            "timings_ms": {
                "resample": prepared.preparation_timings.resample_ms,
                "feature_extract": prepared.preparation_timings.feature_extract_ms,
                "mel": prepared.preparation_timings.feature_extract_ms,
                "encoder_forward": prepared.preparation_timings.encoder_forward_ms,
                "audio_encode": prepared.preparation_timings.encoder_forward_ms,
                "prompt_build": prepared.preparation_timings.prompt_build_ms,
                "prompt_embed": prepared.preparation_timings.prompt_embed_ms,
                "prompt_concat": prepared.preparation_timings.prompt_concat_ms,
                "prefill": profile.main_prefill_ms,
                "main_prefill": profile.main_prefill_ms,
                "decode": profile.decode_loop_ms,
                "decode_argmax": profile.decode_argmax_ms,
                "decode_host_read": profile.decode_host_read_ms,
                "decode_token_tensor": profile.decode_token_tensor_ms,
                "decode_forward": profile.decode_forward_ms,
                "tokenizer_decode": profile.tokenizer_decode_ms,
                "main_backbone": main_backbone_ms,
                "model_total": model_total_ms
            },
            "prompt": {
                "prompt_tokens": output.prompt_tokens,
                "prefix_tokens": prepared.prefix_tokens,
                "suffix_tokens": prepared.suffix_tokens
            },
            "audio": {
                "input_samples": prepared.source_samples,
                "input_sample_rate": prepared.source_sample_rate,
                "resampled_samples": prepared.resampled_samples,
                "mel_frames": prepared.mel_frames,
                "feature_frames": prepared.effective_feature_frames,
                "audio_tokens": prepared.audio_tokens,
                "materialized_tensor_elements": prepared.materialized_tensor_elements,
                "retained_resident_bytes": prepared.retained_resident_bytes,
                "retained_host_bytes": prepared.retained_host_bytes
            },
            "decode": {
                "generated_tokens": output.tokens_generated,
                "max_new_tokens": max_new_tokens,
                "stop_reason": profile.stop_reason.unwrap_or("unknown"),
                "repetition_loop": profile.token_repetition_loop || profile.text_repetition_loop,
                "token_repetition_loop": profile.token_repetition_loop,
                "text_repetition_loop": profile.text_repetition_loop,
                "token_select_reads": profile.token_select_reads,
                "device_argmax_reads": device_token_select_reads,
                "host_argmax_reads": host_argmax_reads,
                "host_read_chunks": profile.host_read_chunks,
                "host_token_reads": profile.host_token_reads,
                "device_token_steps": profile.device_token_steps,
                "profile": {
                    "enabled": true,
                    "sampling_ms": profile.decode_argmax_ms + profile.decode_host_read_ms,
                    "argmax_ms": profile.decode_argmax_ms,
                    "host_read_ms": profile.decode_host_read_ms,
                    "host_read_chunks": profile.host_read_chunks,
                    "token_tensor_ms": profile.decode_token_tensor_ms,
                    "decoder_forward_ms": profile.decode_forward_ms,
                    "tokenizer_decode_ms": profile.tokenizer_decode_ms,
                    "step_total_ms": profile.decode_loop_ms
                }
            },
            "execution": {
                "deferred_stop_check": deferred_stop_check,
                "chunked_stop_check": deferred_stop_check,
                "stop_check_interval": stop_check_interval
            }
        }));
        Ok(output)
    }

    pub(crate) fn generate_sequential_with_config_and_callback_physical(
        &self,
        messages: &[ChatMessage],
        max_new_tokens: usize,
        generation_config: &Lfm25AudioGenerationConfig,
        cache: &mut PhysicalPagedKvCache,
        shortconv: &mut InvocationTensorLease,
        depthformer_cache: &mut PhysicalPagedKvCache,
        on_text_delta: &mut dyn FnMut(&str),
    ) -> Result<Lfm25AudioGenerationOutput> {
        let total_started = Instant::now();
        let prompt_build_started = Instant::now();
        let prompt_ids = self.build_chat_prompt(messages)?;
        let prompt_build_ms = elapsed_ms(prompt_build_started);
        let vocab_limit = self.tokenizer.vocab_size();
        let specials = self.tokenizer.specials().clone();
        let codebooks = self.decoder_config.codebooks;
        let audio_stop_check_interval = lfm25_tts_audio_stop_check_interval();
        let chunked_audio_stop_check =
            generation_config.audio.temperature <= 1e-5 && audio_stop_check_interval > 1;

        let main_started = Instant::now();
        let (
            text,
            prompt_tokens,
            tokens_generated,
            audio_codes,
            device_audio_codes,
            audio_frames_generated,
            mut profile,
        ) = self.with_main_backbone(|main_backbone| {
            let mut profile = Lfm25TtsProfile::default();
            let mut rng = SimpleRng::new(generation_config.seed);
            reset_main_state(cache, shortconv)?;

            let prompt_embed_started = Instant::now();
            let prompt_embeds = embed_token_ids(main_backbone, &self.device.device, &prompt_ids)?;
            profile.prompt_embed_ms = elapsed_ms(prompt_embed_started);
            let prompt_tokens = prompt_embeds.dim(1)?;
            if prompt_tokens >= self.main_config.context_length {
                return Err(Error::InvalidInput(format!(
                    "LFM2.5 Audio TTS prompt has {prompt_tokens} tokens and leaves no generation capacity in the {}-token context",
                    self.main_config.context_length
                )));
            }
            let prefill_started = Instant::now();
            let prompt_hidden =
                main_backbone.forward_embeds_physical(&prompt_embeds, 0, cache, shortconv)?;
            let mut last_hidden = last_hidden_state(&prompt_hidden)?;
            let mut logits = main_backbone.project_last_hidden(&prompt_hidden)?;
            profile.main_prefill_ms = elapsed_ms(prefill_started);
            let mut position = prompt_tokens;
            let mut visible_text_ids = Vec::new();
            let mut visible_text = String::new();
            let mut audio_codes = vec![Vec::new(); codebooks];
            let mut tokens_generated = 0usize;
            let mut in_audio = false;
            let mut generation_done = false;
            let mut sampled_audio_frames: Vec<Lfm25SampledAudioFrame> = Vec::new();
            let mut audio_stop_check_chunk: Vec<Lfm25SampledAudioFrame> = Vec::new();
            let max_new_tokens = max_new_tokens
                .max(1)
                .min(self.main_config.context_length - prompt_tokens);

            while tokens_generated < max_new_tokens && !generation_done {
                if !in_audio {
                    let sampling_started = Instant::now();
                    let next = sample_from_logits(
                        &logits,
                        vocab_limit,
                        &generation_config.text,
                        &mut rng,
                    )?;
                    profile.text_sampling_ms += elapsed_ms(sampling_started);
                    profile.text_sample_calls = profile.text_sample_calls.saturating_add(1);
                    tokens_generated += 1;

                    if next == specials.im_end
                        || next == specials.eos
                        || specials.eos_alt == Some(next)
                    {
                        break;
                    }

                    if next == specials.audio_start {
                        in_audio = true;
                    } else if next != specials.text_end {
                        visible_text_ids.push(next);
                        let tokenizer_started = Instant::now();
                        let decoded = self.tokenizer.decode_text(&visible_text_ids)?;
                        profile.tokenizer_decode_ms += elapsed_ms(tokenizer_started);
                        let delta = text_delta(&visible_text, &decoded);
                        if !delta.is_empty() {
                            for ch in delta.chars() {
                                let mut buf = [0u8; 4];
                                on_text_delta(ch.encode_utf8(&mut buf));
                            }
                        }
                        visible_text = decoded;
                    }

                    let text_forward_started = Instant::now();
                    let next_embed = embed_token_ids(main_backbone, &self.device.device, &[next])?;
                    let step_hidden = main_backbone.forward_embeds_physical(
                        &next_embed,
                        position,
                        cache,
                        shortconv,
                    )?;
                    position += 1;
                    last_hidden = last_hidden_state(&step_hidden)?;
                    logits = main_backbone.project_last_hidden(&step_hidden)?;
                    profile.text_forward_ms += elapsed_ms(text_forward_started);

                    if has_token_repetition_loop(&visible_text_ids) {
                        break;
                    }
                } else if chunked_audio_stop_check {
                    let audio_head_started = Instant::now();
                    let (frame, audio_head_profile) =
                        self.audio_head.sample_audio_frame_embedded_with_profile(
                            &last_hidden,
                            &generation_config.audio,
                            &mut rng,
                            depthformer_cache,
                        )?;
                    profile.audio_head_ms += elapsed_ms(audio_head_started);
                    profile.audio_head_depth_linear_ms += audio_head_profile.depth_linear_ms;
                    profile.audio_head_depth_reshape_ms += audio_head_profile.depth_reshape_ms;
                    profile.audio_head_cache_setup_ms += audio_head_profile.cache_setup_ms;
                    profile.audio_head_codebook_input_ms += audio_head_profile.codebook_input_ms;
                    profile.audio_head_depthformer_ms += audio_head_profile.depthformer_ms;
                    profile.audio_head_sample_ms += audio_head_profile.sample_ms;
                    profile.audio_head_embed_step_ms += audio_head_profile.embed_ms;
                    profile.audio_head_materialize_ms += audio_head_profile.materialize_ms;
                    profile.audio_head_materialize_pack_ms +=
                        audio_head_profile.materialize_pack_ms;
                    profile.audio_head_materialize_readback_ms +=
                        audio_head_profile.materialize_readback_ms;
                    profile.audio_head_calls = profile.audio_head_calls.saturating_add(1);
                    profile.audio_head_codebook_steps = profile
                        .audio_head_codebook_steps
                        .saturating_add(audio_head_profile.codebook_steps);
                    tokens_generated += 1;

                    let audio_embed_started = Instant::now();
                    let audio_embed = frame.embedding().clone();
                    profile.audio_embed_ms += elapsed_ms(audio_embed_started);
                    let audio_forward_started = Instant::now();
                    let step_hidden = main_backbone.forward_embeds_physical(
                        &audio_embed,
                        position,
                        cache,
                        shortconv,
                    )?;
                    position += 1;
                    last_hidden = last_hidden_state(&step_hidden)?;
                    profile.audio_forward_ms += elapsed_ms(audio_forward_started);

                    audio_stop_check_chunk.push(frame);
                    let should_check_audio_stop = audio_stop_check_chunk.len()
                        >= audio_stop_check_interval
                        || tokens_generated >= max_new_tokens;
                    if should_check_audio_stop {
                        let first_tokens = self
                            .audio_head
                            .first_tokens_with_profile(&audio_stop_check_chunk)?;
                        profile.audio_head_ms += first_tokens.materialize_ms;
                        profile.audio_head_materialize_ms += first_tokens.materialize_ms;
                        profile.audio_head_materialize_pack_ms += first_tokens.pack_ms;
                        profile.audio_head_materialize_readback_ms += first_tokens.readback_ms;

                        if let Some(end_idx) = first_tokens
                            .tokens
                            .iter()
                            .position(|token| *token == self.audio_head.audio_end_token_id())
                        {
                            let speculative_after_end =
                                audio_stop_check_chunk.len().saturating_sub(end_idx + 1);
                            tokens_generated =
                                tokens_generated.saturating_sub(speculative_after_end);
                            sampled_audio_frames.extend(audio_stop_check_chunk.drain(..end_idx));
                            audio_stop_check_chunk.clear();
                            in_audio = false;
                            generation_done = true;
                        } else {
                            sampled_audio_frames.append(&mut audio_stop_check_chunk);
                        }
                    }
                } else {
                    let audio_head_started = Instant::now();
                    let (frame, audio_head_profile) =
                        self.audio_head.sample_audio_frame_with_profile(
                            &last_hidden,
                            &generation_config.audio,
                            &mut rng,
                            depthformer_cache,
                        )?;
                    profile.audio_head_ms += elapsed_ms(audio_head_started);
                    profile.audio_head_depth_linear_ms += audio_head_profile.depth_linear_ms;
                    profile.audio_head_depth_reshape_ms += audio_head_profile.depth_reshape_ms;
                    profile.audio_head_cache_setup_ms += audio_head_profile.cache_setup_ms;
                    profile.audio_head_codebook_input_ms += audio_head_profile.codebook_input_ms;
                    profile.audio_head_depthformer_ms += audio_head_profile.depthformer_ms;
                    profile.audio_head_sample_ms += audio_head_profile.sample_ms;
                    profile.audio_head_embed_step_ms += audio_head_profile.embed_ms;
                    profile.audio_head_materialize_ms += audio_head_profile.materialize_ms;
                    profile.audio_head_materialize_pack_ms +=
                        audio_head_profile.materialize_pack_ms;
                    profile.audio_head_materialize_readback_ms +=
                        audio_head_profile.materialize_readback_ms;
                    profile.audio_head_calls = profile.audio_head_calls.saturating_add(1);
                    profile.audio_head_codebook_steps = profile
                        .audio_head_codebook_steps
                        .saturating_add(audio_head_profile.codebook_steps);
                    tokens_generated += 1;
                    let is_end =
                        frame.first().copied() == Some(self.audio_head.audio_end_token_id());
                    if !is_end {
                        for (codebook_idx, token) in frame.iter().copied().enumerate() {
                            audio_codes[codebook_idx].push(token);
                        }
                    }

                    let audio_embed_started = Instant::now();
                    let audio_embed = self
                        .audio_head
                        .embed_audio_frame(&frame, &self.device.device)?;
                    profile.audio_embed_ms += elapsed_ms(audio_embed_started);
                    let audio_forward_started = Instant::now();
                    let step_hidden = main_backbone.forward_embeds_physical(
                        &audio_embed,
                        position,
                        cache,
                        shortconv,
                    )?;
                    position += 1;
                    last_hidden = last_hidden_state(&step_hidden)?;
                    profile.audio_forward_ms += elapsed_ms(audio_forward_started);

                    if is_end {
                        in_audio = false;
                        logits = main_backbone.project_last_hidden(&step_hidden)?;
                    }
                }
            }

            let (device_audio_codes, audio_frames_generated) = if chunked_audio_stop_check {
                if !audio_stop_check_chunk.is_empty() {
                    let first_tokens = self
                        .audio_head
                        .first_tokens_with_profile(&audio_stop_check_chunk)?;
                    profile.audio_head_ms += first_tokens.materialize_ms;
                    profile.audio_head_materialize_ms += first_tokens.materialize_ms;
                    profile.audio_head_materialize_pack_ms += first_tokens.pack_ms;
                    profile.audio_head_materialize_readback_ms += first_tokens.readback_ms;
                    if let Some(end_idx) = first_tokens
                        .tokens
                        .iter()
                        .position(|token| *token == self.audio_head.audio_end_token_id())
                    {
                        sampled_audio_frames.extend(audio_stop_check_chunk.drain(..end_idx));
                    } else {
                        sampled_audio_frames.append(&mut audio_stop_check_chunk);
                    }
                }

                let stacked = self
                    .audio_head
                    .stack_frame_tokens_with_profile(&sampled_audio_frames, &self.device.device)?;
                profile.audio_head_ms += stacked.materialize_ms;
                profile.audio_head_materialize_ms += stacked.materialize_ms;
                profile.audio_head_materialize_pack_ms += stacked.pack_ms;
                (Some(stacked.tokens), sampled_audio_frames.len())
            } else {
                (None, audio_codes.first().map(Vec::len).unwrap_or(0))
            };

            Ok((
                visible_text.trim().to_string(),
                prompt_tokens,
                tokens_generated,
                audio_codes,
                device_audio_codes,
                audio_frames_generated,
                profile,
            ))
        })?;
        let main_backbone_ms = elapsed_ms(main_started);

        let detokenizer_started = Instant::now();
        let (samples, detokenizer_profile) = if let Some(device_audio_codes) = device_audio_codes {
            self.detokenizer
                .decode_token_ids_with_profile(&device_audio_codes)?
        } else {
            self.detokenizer
                .decode_with_profile(&audio_codes, &self.device.device)?
        };
        let detokenizer_ms = elapsed_ms(detokenizer_started);
        profile.detokenizer_embedding_ms = detokenizer_profile.embedding_ms;
        profile.detokenizer_upsample_ms = detokenizer_profile.upsample_ms;
        profile.detokenizer_backbone_ms = detokenizer_profile.backbone_forward_ms;
        profile.detokenizer_projection_ms = detokenizer_profile.projection_ms;
        profile.detokenizer_waveform_prepare_ms = detokenizer_profile.waveform_prepare_ms;
        profile.detokenizer_readback_ms = detokenizer_profile.readback_ms;
        profile.detokenizer_istft_ms = detokenizer_profile.istft_ms;
        let model_total_ms = elapsed_ms(total_started);
        let samples_len = samples.len();
        Ok(Lfm25AudioGenerationOutput {
            text,
            prompt_tokens,
            tokens_generated,
            audio_frames_generated,
            samples,
            sample_rate: self.decoder_config.output_sample_rate,
            diagnostics: Some(serde_json::json!({
                "model": "lfm25_audio",
                "task": "tts",
                "timings_ms": {
                    "prompt_build": prompt_build_ms,
                    "prompt_embed": profile.prompt_embed_ms,
                    "prefill": profile.main_prefill_ms,
                    "main_prefill": profile.main_prefill_ms,
                    "text_sampling": profile.text_sampling_ms,
                    "tokenizer_decode": profile.tokenizer_decode_ms,
                    "text_forward": profile.text_forward_ms,
                    "audio_head": profile.audio_head_ms,
                    "audio_head_depth_linear": profile.audio_head_depth_linear_ms,
                    "audio_head_depth_reshape": profile.audio_head_depth_reshape_ms,
                    "audio_head_cache_setup": profile.audio_head_cache_setup_ms,
                    "audio_head_codebook_input": profile.audio_head_codebook_input_ms,
                    "audio_head_depthformer": profile.audio_head_depthformer_ms,
                    "audio_head_sample": profile.audio_head_sample_ms,
                    "audio_head_embed_step": profile.audio_head_embed_step_ms,
                    "audio_head_materialize": profile.audio_head_materialize_ms,
                    "audio_head_materialize_pack": profile.audio_head_materialize_pack_ms,
                    "audio_head_materialize_readback": profile.audio_head_materialize_readback_ms,
                    "audio_embed": profile.audio_embed_ms,
                    "audio_forward": profile.audio_forward_ms,
                    "main_backbone": main_backbone_ms,
                    "detokenizer": detokenizer_ms,
                    "detokenizer_embedding": profile.detokenizer_embedding_ms,
                    "detokenizer_upsample": profile.detokenizer_upsample_ms,
                    "detokenizer_backbone": profile.detokenizer_backbone_ms,
                    "detokenizer_projection": profile.detokenizer_projection_ms,
                    "detokenizer_waveform_prepare": profile.detokenizer_waveform_prepare_ms,
                    "detokenizer_readback": profile.detokenizer_readback_ms,
                    "detokenizer_istft": profile.detokenizer_istft_ms,
                    "model_total": model_total_ms
                },
                "prompt": {
                    "prompt_tokens": prompt_tokens
                },
                "decode": {
                    "generated_tokens": tokens_generated,
                    "max_new_tokens": max_new_tokens,
                    "text_sample_calls": profile.text_sample_calls,
                    "audio_head_calls": profile.audio_head_calls,
                    "audio_head_codebook_steps": profile.audio_head_codebook_steps,
                    "chunked_audio_stop_check": chunked_audio_stop_check,
                    "audio_stop_check_interval": audio_stop_check_interval
                },
                "audio": {
                    "audio_frames": audio_frames_generated,
                    "sample_rate": self.decoder_config.output_sample_rate,
                    "samples": samples_len
                }
            })),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_interleaved_with_config_and_callback_physical(
        &self,
        history_messages: &[ChatMessage],
        audio: &[f32],
        sample_rate: u32,
        max_new_tokens: usize,
        system_prompt: Option<&str>,
        generation_config: &Lfm25AudioGenerationConfig,
        stream_config: &Lfm25AudioStreamConfig,
        cache: &mut PhysicalPagedKvCache,
        shortconv: &mut InvocationTensorLease,
        depthformer_cache: &mut PhysicalPagedKvCache,
        on_text_delta: &mut dyn FnMut(&str),
        on_audio_samples: &mut dyn FnMut(&[f32]),
    ) -> Result<Lfm25AudioGenerationOutput> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let audio_embeds = self.encode_audio_input(audio, sample_rate)?;
        let (prefix_ids, suffix_ids) =
            self.build_audio_chat_prompt(history_messages, system_prompt)?;
        let vocab_limit = self.tokenizer.vocab_size();
        let specials = self.tokenizer.specials().clone();
        let codebooks = self.decoder_config.codebooks;
        let stride_frames = stream_config.decode_stride_frames.max(1);
        let holdback_samples = self.audio_stream_holdback_samples(stream_config);

        let (text, prompt_tokens, tokens_generated, audio_codes, samples) = self
            .with_main_backbone(|main_backbone| {
                let mut rng = SimpleRng::new(generation_config.seed);
                let mut emitted_audio_samples = 0usize;

                reset_main_state(cache, shortconv)?;
                let prefix_embeds =
                    embed_token_ids(main_backbone, &self.device.device, &prefix_ids)?;
                let suffix_embeds =
                    embed_token_ids(main_backbone, &self.device.device, &suffix_ids)?;
                let prompt_embeds =
                    Tensor::cat(&[&prefix_embeds, &audio_embeds, &suffix_embeds], 1)?;
                let prompt_tokens = prompt_embeds.dim(1)?;
                let prompt_hidden =
                    main_backbone.forward_embeds_physical(&prompt_embeds, 0, cache, shortconv)?;
                let mut last_hidden = last_hidden_state(&prompt_hidden)?;
                let mut logits = main_backbone.project_last_hidden(&prompt_hidden)?;
                let mut position = prompt_tokens;
                let mut visible_text_ids = Vec::new();
                let mut visible_text = String::new();
                let mut audio_codes = vec![Vec::new(); codebooks];
                let mut tokens_generated = 0usize;
                let mut in_audio = false;
                let mut text_done = false;
                let mut modality_left = self.decoder_config.interleaved_n_text.max(1);
                let max_new_tokens = max_new_tokens.max(1);

                while tokens_generated < max_new_tokens {
                    modality_left = modality_left.saturating_sub(1);
                    if !in_audio {
                        let next = sample_from_logits(
                            &logits,
                            vocab_limit,
                            &generation_config.text,
                            &mut rng,
                        )?;
                        tokens_generated += 1;

                        if next == specials.im_end
                            || next == specials.eos
                            || specials.eos_alt == Some(next)
                        {
                            break;
                        }

                        if next == specials.text_end {
                            text_done = true;
                        } else {
                            visible_text_ids.push(next);
                            let decoded = self.tokenizer.decode_text(&visible_text_ids)?;
                            let delta = text_delta(&visible_text, &decoded);
                            if !delta.is_empty() {
                                for ch in delta.chars() {
                                    let mut buf = [0u8; 4];
                                    on_text_delta(ch.encode_utf8(&mut buf));
                                }
                            }
                            visible_text = decoded;
                        }

                        if modality_left == 0 || text_done {
                            in_audio = true;
                            modality_left = self.decoder_config.interleaved_n_audio.max(1);
                        }

                        let next_embed =
                            embed_token_ids(main_backbone, &self.device.device, &[next])?;
                        let step_hidden = main_backbone.forward_embeds_physical(
                            &next_embed,
                            position,
                            cache,
                            shortconv,
                        )?;
                        position += 1;
                        last_hidden = last_hidden_state(&step_hidden)?;
                        logits = main_backbone.project_last_hidden(&step_hidden)?;

                        if has_token_repetition_loop(&visible_text_ids) {
                            break;
                        }
                    } else {
                        let mut frame = self.audio_head.sample_audio_frame(
                            &last_hidden,
                            &generation_config.audio,
                            &mut rng,
                            depthformer_cache,
                        )?;
                        tokens_generated += 1;
                        let is_end =
                            frame.first().copied() == Some(self.audio_head.audio_end_token_id());
                        if is_end {
                            frame.fill(self.audio_head.audio_end_token_id());
                            in_audio = false;
                        } else {
                            for (codebook_idx, token) in frame.iter().copied().enumerate() {
                                audio_codes[codebook_idx].push(token);
                            }
                            if modality_left == 0 && !text_done {
                                in_audio = false;
                                modality_left = self.decoder_config.interleaved_n_text.max(1);
                            }
                        }

                        let audio_embed = self
                            .audio_head
                            .embed_audio_frame(&frame, &self.device.device)?;
                        let step_hidden = main_backbone.forward_embeds_physical(
                            &audio_embed,
                            position,
                            cache,
                            shortconv,
                        )?;
                        position += 1;
                        last_hidden = last_hidden_state(&step_hidden)?;
                        logits = main_backbone.project_last_hidden(&step_hidden)?;

                        let should_decode_partial = !audio_codes[0].is_empty()
                            && (is_end
                                || !in_audio
                                || audio_codes[0].len() % stride_frames == 0
                                || tokens_generated >= max_new_tokens);
                        if should_decode_partial {
                            let partial =
                                self.detokenizer.decode(&audio_codes, &self.device.device)?;
                            let delta = next_audio_delta_stable(
                                &partial,
                                &mut emitted_audio_samples,
                                if is_end || !in_audio {
                                    0
                                } else {
                                    holdback_samples
                                },
                                is_end || tokens_generated >= max_new_tokens,
                            );
                            if !delta.is_empty() {
                                on_audio_samples(&delta);
                            }
                        }
                    }
                }

                let samples = self.detokenizer.decode(&audio_codes, &self.device.device)?;
                let final_delta =
                    next_audio_delta_stable(&samples, &mut emitted_audio_samples, 0, true);
                if !final_delta.is_empty() {
                    on_audio_samples(&final_delta);
                }

                Ok((
                    visible_text
                        .trim()
                        .trim_end_matches(super::config::LFM25_AUDIO_TEXT_END_TOKEN)
                        .trim()
                        .to_string(),
                    prompt_tokens,
                    tokens_generated,
                    audio_codes,
                    samples,
                ))
            })?;

        Ok(Lfm25AudioGenerationOutput {
            text,
            prompt_tokens,
            tokens_generated,
            audio_frames_generated: audio_codes.first().map(Vec::len).unwrap_or(0),
            samples,
            sample_rate: self.decoder_config.output_sample_rate,
            diagnostics: None,
        })
    }

    pub fn with_main_backbone<T>(
        &self,
        f: impl FnOnce(&mut QuantizedLfm2Backbone) -> Result<T>,
    ) -> Result<T> {
        let mut guard = self.main_backbone.lock().map_err(|_| {
            Error::InferenceError("LFM2.5 Audio backbone mutex poisoned".to_string())
        })?;
        f(&mut guard)
    }

    fn encode_audio_input(&self, audio: &[f32], sample_rate: u32) -> Result<Tensor> {
        let mono_16khz = if sample_rate == super::config::LFM25_AUDIO_INPUT_SAMPLE_RATE {
            audio.to_vec()
        } else {
            resample_linear(
                audio,
                sample_rate,
                super::config::LFM25_AUDIO_INPUT_SAMPLE_RATE,
            )?
        };

        let (features, feature_frames) = self
            .preprocessor
            .compute_features(&mono_16khz, &self.device.device)?;
        self.encoder.encode(&features, feature_frames)
    }

    fn audio_stream_holdback_samples(&self, stream_config: &Lfm25AudioStreamConfig) -> usize {
        self.decoder_config
            .output_hop_length
            .saturating_mul(self.decoder_config.detokenizer_upsample_factor)
            .saturating_mul(stream_config.holdback_frames)
    }

    fn build_asr_prompt_segments(&self) -> Result<(Vec<u32>, Vec<u32>)> {
        let specials = self.tokenizer.specials();
        let mut prefix = Vec::new();
        if let Some(bos) = specials.bos {
            prefix.push(bos);
        }
        prefix.push(specials.im_start);
        prefix.extend(self.tokenizer.encode_text("system\n")?);
        prefix.extend(self.tokenizer.encode_text("Perform ASR.")?);
        prefix.push(specials.im_end);
        prefix.extend(self.tokenizer.encode_text("\n")?);
        prefix.push(specials.im_start);
        prefix.extend(self.tokenizer.encode_text("user\n")?);

        let mut suffix = Vec::new();
        suffix.push(specials.im_end);
        suffix.extend(self.tokenizer.encode_text("\n")?);
        suffix.push(specials.im_start);
        suffix.extend(self.tokenizer.encode_text("assistant\n")?);

        Ok((prefix, suffix))
    }

    fn build_audio_chat_prompt(
        &self,
        history_messages: &[ChatMessage],
        system_prompt: Option<&str>,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        let specials = self.tokenizer.specials();
        let mut prompt_messages = history_messages.to_vec();
        let explicit_system_prompt = system_prompt
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);

        if let Some(prompt) = explicit_system_prompt {
            if let Some(first) = prompt_messages.first_mut() {
                if matches!(first.role, ChatRole::System) {
                    first.content = prompt;
                } else {
                    prompt_messages.insert(
                        0,
                        ChatMessage {
                            role: ChatRole::System,
                            content: prompt,
                        },
                    );
                }
            } else {
                prompt_messages.insert(
                    0,
                    ChatMessage {
                        role: ChatRole::System,
                        content: prompt,
                    },
                );
            }
        } else if !matches!(
            prompt_messages.first().map(|message| &message.role),
            Some(ChatRole::System)
        ) {
            prompt_messages.insert(
                0,
                ChatMessage {
                    role: ChatRole::System,
                    content: LFM25_AUDIO_DEFAULT_INTERLEAVED_SYSTEM_PROMPT.to_string(),
                },
            );
        }

        let last_assistant_index = prompt_messages
            .iter()
            .rposition(|message| matches!(message.role, ChatRole::Assistant));

        let mut prefix = Vec::new();
        if let Some(bos) = specials.bos {
            prefix.push(bos);
        }

        for (idx, message) in prompt_messages.iter().enumerate() {
            let content = if matches!(message.role, ChatRole::Assistant) {
                if Some(idx) == last_assistant_index {
                    message.content.trim().to_string()
                } else {
                    strip_past_assistant_thinking(message.content.trim())
                }
            } else {
                message.content.trim().to_string()
            };
            if content.is_empty() {
                continue;
            }

            prefix.push(specials.im_start);
            prefix.extend(
                self.tokenizer
                    .encode_text(&format!("{}\n", message.role.as_prompt_role()))?,
            );
            prefix.extend(self.tokenizer.encode_text(&content)?);
            prefix.push(specials.im_end);
            prefix.extend(self.tokenizer.encode_text("\n")?);
        }

        prefix.push(specials.im_start);
        prefix.extend(self.tokenizer.encode_text("user\n")?);

        let mut suffix = Vec::new();
        suffix.push(specials.im_end);
        suffix.extend(self.tokenizer.encode_text("\n")?);
        suffix.push(specials.im_start);
        suffix.extend(self.tokenizer.encode_text("assistant\n")?);

        Ok((prefix, suffix))
    }

    fn build_chat_prompt(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        if messages.is_empty() {
            return Err(Error::InvalidInput(
                "Chat request must include at least one message".to_string(),
            ));
        }

        let mut prompt_messages = messages.to_vec();
        if !matches!(
            prompt_messages.first().map(|message| &message.role),
            Some(ChatRole::System)
        ) {
            prompt_messages.insert(
                0,
                ChatMessage {
                    role: ChatRole::System,
                    content: "You are a helpful assistant.".to_string(),
                },
            );
        }

        let specials = self.tokenizer.specials();
        let last_assistant_index = prompt_messages
            .iter()
            .rposition(|message| matches!(message.role, ChatRole::Assistant));

        let mut ids = Vec::new();
        if let Some(bos) = specials.bos {
            ids.push(bos);
        }

        for (idx, message) in prompt_messages.iter().enumerate() {
            let content = if matches!(message.role, ChatRole::Assistant) {
                if Some(idx) == last_assistant_index {
                    message.content.trim().to_string()
                } else {
                    strip_past_assistant_thinking(message.content.trim())
                }
            } else {
                message.content.trim().to_string()
            };
            if content.is_empty() {
                continue;
            }

            ids.push(specials.im_start);
            ids.extend(
                self.tokenizer
                    .encode_text(&format!("{}\n", message.role.as_prompt_role()))?,
            );
            ids.extend(self.tokenizer.encode_text(&content)?);
            ids.push(specials.im_end);
            ids.extend(self.tokenizer.encode_text("\n")?);
        }

        ids.push(specials.im_start);
        ids.extend(self.tokenizer.encode_text("assistant\n")?);

        Ok(ids)
    }
}

fn embed_token_ids(
    backbone: &QuantizedLfm2Backbone,
    device: &candle_core::Device,
    token_ids: &[u32],
) -> Result<Tensor> {
    let ids = Tensor::from_vec(token_ids.to_vec(), (1, token_ids.len()), device)?;
    backbone.embed_tokens(&ids)
}

fn reset_main_state(
    cache: &mut PhysicalPagedKvCache,
    shortconv: &mut InvocationTensorLease,
) -> Result<()> {
    cache.reset_invocation()?;
    shortconv.reset_invocation()?;
    Ok(())
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn last_hidden_state(hidden_states: &Tensor) -> Result<Tensor> {
    let seq_len = hidden_states.dim(1)?;
    hidden_states
        .i((0, seq_len.saturating_sub(1)))
        .map_err(Error::from)
}

pub(super) fn text_delta(previous: &str, current: &str) -> String {
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

#[derive(Debug, Clone, Copy, Default)]
struct Lfm25AsrAppendStatus {
    token_repetition_loop: bool,
    text_repetition_loop: bool,
}

impl Lfm25AsrAppendStatus {
    fn should_stop(self) -> bool {
        self.token_repetition_loop || self.text_repetition_loop
    }
}

fn append_asr_text_token(
    tokenizer: &Lfm25TextTokenizer,
    generated_ids: &mut Vec<u32>,
    assembled: &mut String,
    next: u32,
    profile: &mut Lfm25AsrProfile,
    on_delta: &mut dyn FnMut(&str),
) -> Result<Lfm25AsrAppendStatus> {
    generated_ids.push(next);
    let tokenizer_started = Instant::now();
    let mut decoded = tokenizer.decode_text(generated_ids)?;
    profile.tokenizer_decode_ms += elapsed_ms(tokenizer_started);
    let token_repetition_loop = has_token_repetition_loop(generated_ids);
    let text_repetition_loop = if let Some(trimmed) = trim_repeated_phrase_tail(&decoded) {
        decoded = trimmed;
        true
    } else {
        false
    };
    let delta = text_delta(assembled, &decoded);
    if !delta.is_empty() {
        for ch in delta.chars() {
            let mut buf = [0u8; 4];
            on_delta(ch.encode_utf8(&mut buf));
        }
    }
    *assembled = decoded;
    Ok(Lfm25AsrAppendStatus {
        token_repetition_loop,
        text_repetition_loop,
    })
}

pub(super) fn is_asr_stop_token(next: u32, specials: &Lfm25SpecialTokenIds) -> bool {
    next == specials.im_end
        || next == specials.eos
        || specials.eos_alt == Some(next)
        || next == specials.text_end
        || next == specials.audio_start
}

fn lfm25_asr_stop_check_interval() -> usize {
    lfm25_asr_stop_check_interval_from_env(std::env::var("IZWI_LFM25_ASR_STOP_CHECK_INTERVAL").ok())
}

fn lfm25_asr_stop_check_interval_from_env(value: Option<String>) -> usize {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_ASR_STOP_CHECK_INTERVAL)
        .clamp(1, 128)
}

fn lfm25_tts_audio_stop_check_interval() -> usize {
    lfm25_tts_audio_stop_check_interval_from_env(
        std::env::var("IZWI_LFM25_TTS_AUDIO_STOP_CHECK_INTERVAL").ok(),
    )
}

fn lfm25_tts_audio_stop_check_interval_from_env(value: Option<String>) -> usize {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_TTS_AUDIO_STOP_CHECK_INTERVAL)
        .clamp(1, 32)
}

fn next_audio_delta_stable(
    all_samples: &[f32],
    emitted_samples: &mut usize,
    holdback_samples: usize,
    is_final: bool,
) -> Vec<f32> {
    let stable_end = if is_final {
        all_samples.len()
    } else {
        all_samples.len().saturating_sub(holdback_samples)
    };
    let start = (*emitted_samples).min(stable_end);
    let delta = all_samples[start..stable_end].to_vec();
    *emitted_samples = stable_end;
    delta
}

fn strip_past_assistant_thinking(input: &str) -> String {
    if let Some((_reasoning, tail)) = input.rsplit_once("</think>") {
        tail.trim().to_string()
    } else {
        input.trim().to_string()
    }
}

fn has_suffix_repeat(ids: &[u32], span: usize, repeats: usize) -> bool {
    if span == 0 || repeats < 2 || ids.len() < span * repeats {
        return false;
    }
    let tail_start = ids.len() - span;
    let tail = &ids[tail_start..];
    (2..=repeats).all(|rep| {
        let start = ids.len() - (span * rep);
        &ids[start..start + span] == tail
    })
}

pub(super) fn has_token_repetition_loop(ids: &[u32]) -> bool {
    if ids.len() < 48 {
        return false;
    }
    const PATTERNS: &[(usize, usize)] = &[(24, 3), (16, 3), (12, 3), (8, 4), (6, 5)];
    PATTERNS
        .iter()
        .any(|(span, repeats)| has_suffix_repeat(ids, *span, *repeats))
}

#[derive(Debug, Clone)]
struct NormalizedTextWord {
    normalized: String,
    end: usize,
}

pub(super) fn trim_repeated_phrase_tail(input: &str) -> Option<String> {
    let words = normalized_text_words(input);
    const MIN_REPEAT_COUNT: usize = 4;
    const MIN_PHRASE_WORDS: usize = 2;
    const MAX_PHRASE_WORDS: usize = 8;
    let max_phrase_words = MAX_PHRASE_WORDS.min(words.len() / MIN_REPEAT_COUNT);
    for phrase_words in MIN_PHRASE_WORDS..=max_phrase_words {
        let mut repeat_count = 1usize;
        while words.len() >= phrase_words.saturating_mul(repeat_count + 1) {
            let right_start = words.len() - phrase_words * repeat_count;
            let left_start = right_start.saturating_sub(phrase_words);
            if !normalized_word_ranges_equal(&words, left_start, right_start, phrase_words) {
                break;
            }
            repeat_count += 1;
        }

        if repeat_count >= MIN_REPEAT_COUNT {
            let keep_words = words.len() - phrase_words * (repeat_count - 1);
            let end = words[keep_words - 1].end;
            return Some(input[..end].trim_end().to_string());
        }
    }
    None
}

fn normalized_word_ranges_equal(
    words: &[NormalizedTextWord],
    left_start: usize,
    right_start: usize,
    len: usize,
) -> bool {
    (0..len).all(|offset| {
        words[left_start + offset].normalized == words[right_start + offset].normalized
    })
}

fn normalized_text_words(input: &str) -> Vec<NormalizedTextWord> {
    let mut words = Vec::new();
    let mut start = None;
    for (idx, ch) in input.char_indices() {
        if ch.is_whitespace() {
            if let Some(word_start) = start.take() {
                push_normalized_text_word(input, word_start, idx, &mut words);
            }
        } else if start.is_none() {
            start = Some(idx);
        }
    }
    if let Some(word_start) = start {
        push_normalized_text_word(input, word_start, input.len(), &mut words);
    }
    words
}

fn push_normalized_text_word(
    input: &str,
    start: usize,
    end: usize,
    words: &mut Vec<NormalizedTextWord>,
) {
    let raw = &input[start..end];
    let normalized = raw
        .trim_matches(|ch: char| !ch.is_alphanumeric())
        .to_ascii_lowercase();
    if normalized.is_empty() {
        return;
    }
    words.push(NormalizedTextWord { normalized, end });
}

fn checked_asr_prompt_tokens(
    prefix_tokens: usize,
    audio_tokens: usize,
    suffix_tokens: usize,
    context_length: usize,
) -> Result<usize> {
    let prompt_tokens = prefix_tokens
        .checked_add(audio_tokens)
        .and_then(|tokens| tokens.checked_add(suffix_tokens))
        .ok_or_else(|| {
            Error::InvalidInput("LFM2.5 Audio prompt token count overflowed usize".to_string())
        })?;
    if prompt_tokens >= context_length {
        return Err(Error::InvalidInput(format!(
            "LFM2.5 Audio prompt has {prompt_tokens} tokens and leaves no generation capacity in the {context_length}-token context"
        )));
    }
    Ok(prompt_tokens)
}

fn checked_asr_prompt_tensor_elements(prompt_tokens: usize, hidden_size: usize) -> Result<u64> {
    u64::try_from(prompt_tokens)
        .ok()
        .zip(u64::try_from(hidden_size).ok())
        .and_then(|(tokens, hidden)| tokens.checked_mul(hidden))
        .ok_or_else(|| {
            Error::InvalidInput("LFM2.5 Audio prompt tensor elements overflowed u64".to_string())
        })
}

fn prepared_asr_retained_host_bytes() -> u64 {
    // Prefix and suffix token IDs are consumed during preparation. The retained
    // artifact owns no dynamically-sized host payload, only its device tensor.
    0
}

fn validate_prepared_asr_prompt_shape(
    dims: &[usize],
    prompt_tokens: usize,
    hidden_size: usize,
) -> Result<()> {
    if dims != [1, prompt_tokens, hidden_size] {
        return Err(Error::InvalidInput(format!(
            "Prepared LFM2.5 Audio prompt tensor has shape {dims:?}, expected [1, {prompt_tokens}, {hidden_size}]"
        )));
    }
    Ok(())
}

fn checked_asr_preparation_workspace(
    resampled_samples: usize,
    padded_samples: usize,
    feature_frames: usize,
    encoder_frames: usize,
    prompt_tokens: usize,
    encoder: &Lfm25AudioEncoderConfig,
    main_hidden: usize,
) -> Result<(u64, u64)> {
    // This deliberately sums conservative logical allocations rather than
    // claiming an allocator-observed peak. In particular, four N×N attention
    // planes overbound the score, relative N×(2N-1), rel-shift, and mask
    // materializations for every Conformer block on every backend.
    let u = |value: usize, label: &str| {
        u64::try_from(value).map_err(|_| {
            Error::InvalidInput(format!("LFM2.5 Audio {label} exceeds u64 during sealing"))
        })
    };
    let mul = |left: u64, right: u64, label: &str| {
        left.checked_mul(right).ok_or_else(|| {
            Error::InvalidInput(format!("LFM2.5 Audio {label} overflowed during sealing"))
        })
    };
    let add = |left: u64, right: u64, label: &str| {
        left.checked_add(right).ok_or_else(|| {
            Error::InvalidInput(format!("LFM2.5 Audio {label} overflowed during sealing"))
        })
    };
    let f32_bytes = 4u64;
    let mel = u(encoder.num_mel_bins, "mel bins")?;
    let frames = u(feature_frames, "feature frames")?;
    let encoded = u(encoder_frames, "encoder frames")?;
    let hidden = u(encoder.embedding_length, "encoder hidden size")?;
    let ffn = u(encoder.feed_forward_length, "encoder FFN size")?;
    let heads = u(encoder.attention_head_count, "encoder attention heads")?;

    // Host frontend peak includes the resampled waveform, reflected padding,
    // row-major log-mel, transposed tensor upload buffer, and complex FFT frame.
    let host_elements = add(
        add(
            u(resampled_samples, "resampled samples")?,
            u(padded_samples, "padded samples")?,
            "host waveforms",
        )?,
        add(
            mul(mul(frames, mel, "log-mel elements")?, 2, "dual mel buffers")?,
            mul(
                u(super::config::LFM25_AUDIO_INPUT_N_FFT, "FFT width")?,
                2,
                "complex FFT frame",
            )?,
            "host feature scratch",
        )?,
        "host frontend peak",
    )?;

    let feature_elements = mul(frames, mel, "feature tensor")?;
    let conv_elements = mul(
        mul(frames, mel, "conv spatial elements")?,
        u(encoder.subsampling_channels, "subsampling channels")?,
        "conv activation",
    )?;
    let encoded_elements = mul(encoded, hidden, "encoder hidden")?;
    let ffn_elements = mul(encoded, ffn, "encoder FFN")?;
    let attention_elements = mul(
        mul(heads, encoded, "attention rows")?,
        encoded,
        "attention scores",
    )?;
    let positional_elements = mul(
        encoded
            .checked_mul(2)
            .and_then(|v| v.checked_sub(1))
            .ok_or_else(|| {
                Error::InvalidInput("LFM2.5 Audio positional width overflowed".into())
            })?,
        hidden,
        "relative positional embedding",
    )?;
    let per_layer = add(
        mul(encoded_elements, 12, "Conformer hidden intermediates")?,
        add(
            mul(ffn_elements, 4, "Conformer FFN intermediates")?,
            mul(attention_elements, 4, "Conformer attention intermediates")?,
            "Conformer layer scratch",
        )?,
        "Conformer layer total",
    )?;
    let all_layers = mul(
        per_layer,
        u(encoder.block_count, "encoder block count")?,
        "Conformer blocks",
    )?;
    let adapter_elements = mul(
        encoded,
        u(
            encoder.projection_dim.max(encoder.embedding_length),
            "adapter width",
        )?,
        "adapter activation",
    )?;
    let prompt_elements = mul(
        u(prompt_tokens, "prompt tokens")?,
        u(main_hidden, "main hidden")?,
        "full prompt",
    )?;
    let device_elements = [
        feature_elements,
        conv_elements,
        encoded_elements,
        positional_elements,
        all_layers,
        mul(adapter_elements, 3, "adapter intermediates")?,
        mul(prompt_elements, 3, "mixed prompt construction")?,
    ]
    .into_iter()
    .try_fold(0u64, |total, value| {
        add(total, value, "device preparation peak")
    })?;
    Ok((
        mul(host_elements, f32_bytes, "host workspace bytes")?,
        mul(device_elements, f32_bytes, "device workspace bytes")?,
    ))
}

fn map_asr_workspace_domains(
    backend: BackendKind,
    host_frontend_bytes: u64,
    tensor_bytes: u64,
) -> Result<(u64, u64, u64)> {
    let unified = host_frontend_bytes
        .checked_add(tensor_bytes)
        .ok_or_else(|| Error::InvalidInput("LFM2.5 Audio workspace domain overflowed".into()))?;
    Ok(match backend {
        BackendKind::Cpu => (unified, 0, 0),
        BackendKind::Metal => (0, 0, unified),
        BackendKind::Cuda => (host_frontend_bytes, tensor_bytes, 0),
    })
}

fn checked_asr_main_step_workspace(
    query_tokens: usize,
    visible_tokens: usize,
    vocab_size: usize,
    include_logits: bool,
    config: &Lfm2BackboneConfig,
) -> Result<u64> {
    let u = |value: usize| {
        u64::try_from(value)
            .map_err(|_| Error::InvalidInput("LFM2.5 Audio step geometry exceeds u64".into()))
    };
    let mul = |left: u64, right: u64| {
        left.checked_mul(right)
            .ok_or_else(|| Error::InvalidInput("LFM2.5 Audio step workspace overflowed".into()))
    };
    let add = |left: u64, right: u64| {
        left.checked_add(right)
            .ok_or_else(|| Error::InvalidInput("LFM2.5 Audio step workspace overflowed".into()))
    };
    let q = u(query_tokens)?;
    let visible = u(visible_tokens)?;
    let hidden = u(config.embedding_length)?;
    let ffn = u(config
        .feed_forward_length
        .unwrap_or(config.embedding_length))?;
    let heads = u(config.attention_head_count)?;
    let layers = u(config.block_count)?;
    let ring = u(config.shortconv_l_cache)?;

    // Conservative logical ceiling. Every model block is priced as if it
    // simultaneously owns residual/norm/QKV/output tensors, both FFN arms,
    // dense attention through the full visible prefix, and a cloned ShortConv
    // ring. Actual mixed attention/ShortConv layouts allocate a strict subset.
    let hidden_planes = mul(mul(q, hidden)?, 12)?;
    let ffn_planes = mul(mul(q, ffn)?, 4)?;
    let attention_planes = mul(mul(mul(heads, q)?, visible)?, 4)?;
    let ring_clone = mul(mul(ring, hidden)?, layers)?;
    let per_layer = add(add(hidden_planes, ffn_planes)?, attention_planes)?;
    let mut elements = add(mul(per_layer, layers)?, ring_clone)?;
    if include_logits {
        elements = add(elements, mul(q, u(vocab_size)?)?)?;
    }
    mul(elements, 4)
}

fn validate_prepared_asr_identity(expected: u64, actual: u64) -> Result<()> {
    if expected != actual {
        return Err(Error::InvalidInput(
            "Prepared LFM2.5 Audio ASR artifact belongs to a different model load".to_string(),
        ));
    }
    Ok(())
}

fn checked_resampled_len(input_len: usize, src_rate: u32, dst_rate: u32) -> Result<usize> {
    if src_rate == 0 || dst_rate == 0 {
        return Err(Error::InvalidInput(
            "LFM2.5 Audio resampling rates must be non-zero".into(),
        ));
    }
    if input_len < 2 || src_rate == dst_rate {
        return Ok(input_len);
    }
    let numerator = (input_len as u128)
        .checked_mul(dst_rate as u128)
        .and_then(|value| value.checked_add((src_rate / 2) as u128))
        .ok_or_else(|| Error::InvalidInput("LFM2.5 Audio resampled length overflowed".into()))?;
    usize::try_from((numerator / src_rate as u128).max(1))
        .map_err(|_| Error::InvalidInput("LFM2.5 Audio resampled length exceeds usize".into()))
}

fn resample_linear(audio: &[f32], src_rate: u32, dst_rate: u32) -> Result<Vec<f32>> {
    if src_rate == dst_rate || audio.len() < 2 {
        return Ok(audio.to_vec());
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = checked_resampled_len(audio.len(), src_rate, dst_rate)?;
    let mut out = Vec::with_capacity(out_len);

    for idx in 0..out_len {
        let src_pos = idx as f64 / ratio;
        let left = src_pos.floor() as usize;
        let right = left
            .min(audio.len() - 1)
            .saturating_add(1)
            .min(audio.len() - 1);
        let frac = (src_pos - left as f64) as f32;
        let left_sample = audio[left.min(audio.len() - 1)];
        let right_sample = audio[right];
        out.push(left_sample + (right_sample - left_sample) * frac);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::backends::DeviceProfile;
    use crate::model::ModelVariant;

    fn local_model_dir(name: &str) -> PathBuf {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
        PathBuf::from(home)
            .join("Library/Application Support/izwi/models")
            .join(name)
    }

    #[test]
    fn next_audio_delta_stable_holds_back_tail_until_final() {
        let mut emitted = 0usize;
        let all = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let delta = next_audio_delta_stable(&all, &mut emitted, 2, false);
        assert_eq!(delta, vec![0.1, 0.2, 0.3]);
        assert_eq!(emitted, 3);

        let delta_final = next_audio_delta_stable(&all, &mut emitted, 0, true);
        assert_eq!(delta_final, vec![0.4, 0.5]);
        assert_eq!(emitted, 5);
    }

    #[test]
    fn strip_past_assistant_thinking_keeps_final_answer_only() {
        assert_eq!(
            strip_past_assistant_thinking("<think>plan</think>final answer"),
            "final answer"
        );
    }

    #[test]
    fn repeated_phrase_tail_trim_keeps_one_tail_occurrence() {
        let text = "So, human spaceflight is very expensive, and it's very expensive. It's very expensive. It's very expensive. It's very expensive.";
        let trimmed = trim_repeated_phrase_tail(text).expect("repeated phrase tail");

        assert_eq!(
            trimmed,
            "So, human spaceflight is very expensive, and it's very expensive."
        );
    }

    #[test]
    fn repeated_phrase_tail_trim_ignores_short_repeats() {
        assert!(trim_repeated_phrase_tail(
            "This is very expensive. It is very expensive, but it may be worth it."
        )
        .is_none());
        assert!(trim_repeated_phrase_tail("yes yes yes yes yes yes").is_none());
    }

    #[test]
    fn asr_stop_check_interval_defaults_to_ninety_two() {
        assert_eq!(lfm25_asr_stop_check_interval_from_env(None), 92);
        assert_eq!(
            lfm25_asr_stop_check_interval_from_env(Some("bad".to_string())),
            92
        );
    }

    #[test]
    fn asr_stop_check_interval_clamps_override() {
        assert_eq!(
            lfm25_asr_stop_check_interval_from_env(Some("0".to_string())),
            1
        );
        assert_eq!(
            lfm25_asr_stop_check_interval_from_env(Some("128".to_string())),
            128
        );
        assert_eq!(
            lfm25_asr_stop_check_interval_from_env(Some("256".to_string())),
            128
        );
        assert_eq!(
            lfm25_asr_stop_check_interval_from_env(Some("96".to_string())),
            96
        );
    }

    #[test]
    fn prepared_asr_prompt_layout_is_exact_and_leaves_decode_capacity() {
        assert_eq!(checked_asr_prompt_tokens(3, 7, 2, 13).unwrap(), 12);
        assert!(checked_asr_prompt_tokens(3, 7, 2, 12).is_err());
        assert!(checked_asr_prompt_tokens(usize::MAX, 1, 0, usize::MAX).is_err());
        assert_eq!(checked_asr_prompt_tensor_elements(12, 8).unwrap(), 96);
        assert_eq!(prepared_asr_retained_host_bytes(), 0);
    }

    #[test]
    fn prepared_asr_prompt_shape_authenticates_full_mixed_tensor() {
        validate_prepared_asr_prompt_shape(&[1, 12, 8], 12, 8).unwrap();
        assert!(validate_prepared_asr_prompt_shape(&[2, 12, 8], 12, 8).is_err());
        assert!(validate_prepared_asr_prompt_shape(&[1, 11, 8], 12, 8).is_err());
        assert!(validate_prepared_asr_prompt_shape(&[1, 12, 7], 12, 8).is_err());
    }

    #[test]
    fn asr_preparation_geometry_uses_exact_checked_resample_rounding() {
        assert_eq!(
            checked_resampled_len(48_000, 48_000, 16_000).unwrap(),
            16_000
        );
        assert_eq!(checked_resampled_len(3, 2, 1).unwrap(), 2);
        assert!(checked_resampled_len(16, 0, 16_000).is_err());
    }

    #[test]
    fn asr_preparation_workspace_maps_backend_domains_without_double_counting() {
        assert_eq!(
            map_asr_workspace_domains(BackendKind::Cpu, 11, 29).unwrap(),
            (40, 0, 0)
        );
        assert_eq!(
            map_asr_workspace_domains(BackendKind::Metal, 11, 29).unwrap(),
            (0, 0, 40)
        );
        assert_eq!(
            map_asr_workspace_domains(BackendKind::Cuda, 11, 29).unwrap(),
            (11, 29, 0)
        );
        assert!(map_asr_workspace_domains(BackendKind::Cpu, u64::MAX, 1).is_err());
    }

    #[test]
    fn main_step_workspace_prices_logits_context_and_shortconv_clones() {
        let config = Lfm2BackboneConfig {
            architecture: "lfm2".into(),
            block_count: 4,
            context_length: 128,
            embedding_length: 16,
            embedding_length_out: None,
            feed_forward_length: Some(32),
            attention_head_count: 4,
            attention_head_count_kv: vec![2; 4],
            attention_layer_norm_rms_epsilon: 1e-5,
            attention_sliding_window: Some(32),
            rope_freq_base: 10_000.0,
            shortconv_l_cache: 3,
        };
        let without_logits = checked_asr_main_step_workspace(2, 11, 101, false, &config).unwrap();
        let with_logits = checked_asr_main_step_workspace(2, 11, 101, true, &config).unwrap();
        assert_eq!(with_logits - without_logits, 2 * 101 * 4);
        assert!(
            checked_asr_main_step_workspace(2, 12, 101, false, &config).unwrap() > without_logits
        );
        assert!(checked_asr_main_step_workspace(usize::MAX, 2, 3, true, &config).is_err());
    }

    #[test]
    fn four_attention_planes_bound_dense_relative_attention_materialization() {
        for frames in [1u64, 2, 17, 511] {
            let dense_and_relative = frames
                .checked_mul(frames)
                .and_then(|dense| {
                    frames
                        .checked_mul(frames * 2 - 1)
                        .and_then(|relative| dense.checked_add(relative))
                })
                .unwrap();
            assert!(dense_and_relative <= 4 * frames * frames);
        }
    }

    #[test]
    fn prepared_asr_artifact_identity_is_model_load_bound() {
        validate_prepared_asr_identity(17, 17).unwrap();
        let error = validate_prepared_asr_identity(17, 18).unwrap_err();
        assert!(error.to_string().contains("different model load"));
    }

    #[test]
    fn tts_audio_stop_check_interval_defaults_to_twenty() {
        assert_eq!(lfm25_tts_audio_stop_check_interval_from_env(None), 20);
        assert_eq!(
            lfm25_tts_audio_stop_check_interval_from_env(Some("bad".to_string())),
            20
        );
    }

    #[test]
    fn tts_audio_stop_check_interval_clamps_override() {
        assert_eq!(
            lfm25_tts_audio_stop_check_interval_from_env(Some("0".to_string())),
            1
        );
        assert_eq!(
            lfm25_tts_audio_stop_check_interval_from_env(Some("32".to_string())),
            32
        );
        assert_eq!(
            lfm25_tts_audio_stop_check_interval_from_env(Some("64".to_string())),
            32
        );
        assert_eq!(
            lfm25_tts_audio_stop_check_interval_from_env(Some("4".to_string())),
            4
        );
    }

    #[test]
    fn load_local_lfm25_audio_model_smoke_if_available() {
        let model_dir = local_model_dir("LFM2.5-Audio-1.5B-GGUF");
        if !model_dir.exists() {
            return;
        }

        let model = Lfm25AudioModel::load(
            &model_dir,
            ModelVariant::Lfm25Audio15BGguf,
            DeviceProfile::cpu(),
        )
        .expect("lfm2.5 audio assets should load");

        assert_eq!(model.main_config().architecture, "lfm2");
        assert_eq!(model.encoder_config().embedding_length, 512);
        assert_eq!(model.encoder_config().feed_forward_length, 2048);
        assert_eq!(model.decoder_config().codebooks, 8);
    }
}
