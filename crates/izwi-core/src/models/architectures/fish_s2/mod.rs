//! Fish Audio S2 Pro TTS architecture boundary.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use candle_core::{DType, IndexOp, Tensor};
use serde::Serialize;

use sampling::FishS2SamplingDistribution;

use crate::backends::{DTypeSelection, DeviceProfile};
use crate::catalog::ModelVariant;
use crate::engine::StageDescriptor;
use crate::error::{Error, Result};
use crate::kernels::numeric_check::{
    load_with_precision_fallback, run_numeric_self_check, xorshift_values, NumericSelfCheckReport,
    PrecisionLoad, F16_SAFE_MAX_ABS,
};
use crate::models::shared::attention::physical::PhysicalPagedKvCache;

pub mod artifacts;
mod batch;
pub mod codec;
pub mod config;
pub mod contracts;
pub mod dac;
pub mod fast;
mod physical;
mod reference_cache;
mod retained;
mod rotary;
mod sampling;
pub mod slow;
pub mod tokenizer;
pub mod weights;

pub use artifacts::FishS2ArtifactManifest;
pub use codec::{FishS2CodecArtifact, FishS2CodecSupport};
pub use config::{FishS2AudioDecoderConfig, FishS2Config, FishS2TextConfig};
pub use contracts::{
    build_semantic_allowed_mask, remap_fish_qwen3_omni_key, semantic_code_from_token_id,
    semantic_token_id, FishS2DacContract, FishS2PromptTensorShape,
};
pub use dac::{FishS2DacConfig, FishS2DacDecoder};
pub use fast::{FishS2FastConfig, FishS2FastDecoder, FishS2GeneratedFrame, FishS2Sampler};
#[cfg(test)]
pub(crate) use physical::fish_s2_physical_state_spec;
pub(crate) use physical::{FishS2PhysicalStateSpec, FISH_S2_SLOW_STATE_GROUP};
#[allow(unused_imports)]
pub(crate) use retained::{
    FishS2PreparationMemory, FishS2PreparedArtifact, FishS2RetainedCheckpoint, FishS2RetainedState,
    FishS2RetainedStep,
};
pub use slow::{FishS2SlowConfig, FishS2SlowOutput, FishS2SlowTransformer};
pub use tokenizer::{
    FishS2ConditioningPrompt, FishS2PromptTokenizer, FishS2SpecialTokens, FishS2VqCodes,
};
pub use weights::{FishS2TensorSpec, FishS2WeightIndex, FishS2Weights};

pub(crate) use reference_cache::FISH_S2_REFERENCE_CACHE_BYTES;

pub(crate) const FISH_S2_TTS_PREPARATION_STAGE: &str = "tts.prepare.fish_s2";
pub(crate) const FISH_S2_TTS_PREFILL_STAGE: &str = "tts.prefill.fish_s2";
pub(crate) const FISH_S2_TTS_DECODE_STAGE: &str = "tts.decode.fish_s2";
pub(crate) const FISH_S2_AUDIO_CHUNK_FRAMES: usize = 16;
pub(crate) const FISH_S2_FIRST_AUDIO_FRAMES: usize = 4;
pub(crate) const FISH_S2_TTS_AUDIO_DECODE_STAGE: &str = "tts.audio_decode.fish_s2";
pub(crate) const FISH_S2_TTS_FINALIZE_STAGE: &str = "tts.codec.fish_s2.scalar";

pub struct FishS2TtsModel {
    model_identity: u64,
    artifact_fingerprint: Option<String>,
    variant: ModelVariant,
    config: FishS2Config,
    artifacts: FishS2ArtifactManifest,
    codec: FishS2CodecArtifact,
    runtime: Option<FishS2NativeRuntime>,
    reference_cache: reference_cache::ReferenceCodeCache,
}

struct FishS2NativeRuntime {
    tokenizer: FishS2PromptTokenizer,
    slow: FishS2SlowTransformer,
    fast: FishS2FastDecoder,
    dac: FishS2DacDecoder,
    semantic_allowed_mask: Vec<bool>,
    dtype: DType,
    /// The transformers' dtype after the load-time check, and why.
    precision: PrecisionLoad,
}

#[derive(Debug, Clone)]
pub struct FishS2Reference {
    pub audio_samples: Vec<f32>,
    pub sample_rate: u32,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FishS2GenerationParams {
    pub max_frames: usize,
    pub temperature: f32,
    pub top_p: f32,
    /// Zero disables the top-k cap. The upstream serving default is 30.
    pub top_k: usize,
    /// Reproducible Rust sampling seed; fast-codebook sampling uses seed + 1.
    pub seed: u64,
    /// Retry recent semantic repeats using the upstream high-temperature policy.
    /// A zero temperature remains greedy and bypasses this retry.
    pub repetition_aware: bool,
}

#[derive(Debug, Clone)]
pub struct FishS2GenerationOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub frames_generated: usize,
    pub diagnostics: FishS2TtsGenerationDiagnostics,
}

#[derive(Debug, Clone, Serialize)]
pub struct FishS2TtsDiagnostics {
    pub model_family: &'static str,
    pub variant: String,
    pub native_runtime_loaded: bool,
    pub sample_rate: u32,
    pub num_codebooks: usize,
    pub codebook_size: usize,
    pub semantic_start_token_id: u32,
    pub semantic_end_token_id: u32,
    pub max_seq_len: usize,
    pub codec_support: &'static str,
    /// The slow/fast transformers' load precision and self-check (FSH-11);
    /// `None` without the native runtime.
    pub precision: Option<serde_json::Value>,
}

/// Model-authored terminal reason. Exhausting a budget is incomplete speech.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum FishS2StopReason {
    #[serde(rename = "im_end")]
    EndOfSpeech,
    #[serde(rename = "max_frames")]
    FrameLimit,
}

impl FishS2StopReason {
    pub(crate) fn require_complete(self) -> Result<()> {
        match self {
            Self::EndOfSpeech => Ok(()),
            Self::FrameLimit => Err(Error::InferenceError(
                "Fish S2 generation incomplete: frame_limit reached before end of speech".into(),
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct FishS2TtsGenerationDiagnostics {
    pub model_family: &'static str,
    pub sample_rate: u32,
    pub prompt_tokens: usize,
    pub max_frames: usize,
    pub frames_generated: usize,
    pub stop_reason: FishS2StopReason,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    pub repetition_aware: bool,
    pub reference_encode_ms: f32,
    pub prompt_build_ms: f32,
    pub slow_prefill_ms: f32,
    pub ar_decode_ms: f32,
    pub dac_decode_ms: f32,
    pub total_model_ms: f32,
}

const FISH_S2_DTYPE_ENV: &str = "IZWI_FISH_S2_DTYPE";
const RAS_WIN_SIZE: usize = 10;
const RAS_HIGH_TEMP: f32 = 1.0;
const RAS_HIGH_TOP_P: f32 = 0.9;
static NEXT_FISH_S2_MODEL_IDENTITY: AtomicU64 = AtomicU64::new(1);

impl FishS2TtsModel {
    /// Metadata-only fixture for request preparation and admission tests.
    /// Native inference remains unavailable, as for `load_metadata`.
    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        Self {
            model_identity: 1,
            artifact_fingerprint: None,
            variant: ModelVariant::FishAudioS2Pro,
            config: config::current_config(),
            artifacts: FishS2ArtifactManifest {
                model_dir: Default::default(),
                shard_files: Vec::new(),
                tensor_count: 0,
                text_tensor_count: 0,
                audio_decoder_tensor_count: 0,
                codec_path: Default::default(),
            },
            codec: FishS2CodecArtifact {
                path: Default::default(),
                support: codec::FishS2CodecSupport::NativePthStateDict,
            },
            runtime: None,
            reference_cache: Default::default(),
        }
    }

    pub fn load_metadata(model_dir: &Path, variant: ModelVariant) -> Result<Self> {
        if variant != ModelVariant::FishAudioS2Pro {
            return Err(Error::InvalidInput(format!(
                "Unsupported Fish S2 TTS variant: {variant}"
            )));
        }
        let config = FishS2Config::load(model_dir)?;
        let artifacts = FishS2ArtifactManifest::load(model_dir)?;
        let codec = FishS2CodecArtifact::load(model_dir)?;
        Ok(Self {
            model_identity: next_fish_s2_model_identity()?,
            artifact_fingerprint: None,
            variant,
            config,
            artifacts,
            codec,
            runtime: None,
            reference_cache: Default::default(),
        })
    }

    pub fn load(model_dir: &Path, variant: ModelVariant, device: DeviceProfile) -> Result<Self> {
        let mut model = Self::load_metadata(model_dir, variant)?;
        model.artifact_fingerprint = Some(model.artifacts.content_fingerprint()?);
        model.runtime = Some(FishS2NativeRuntime::load(
            model_dir,
            &model.config,
            &model.codec,
            device,
        )?);
        Ok(model)
    }

    pub fn artifact_fingerprint(&self) -> Option<&str> {
        self.artifact_fingerprint.as_deref()
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    pub fn config(&self) -> &FishS2Config {
        &self.config
    }

    pub fn artifacts(&self) -> &FishS2ArtifactManifest {
        &self.artifacts
    }

    pub fn codec(&self) -> &FishS2CodecArtifact {
        &self.codec
    }

    pub fn diagnostics(&self) -> FishS2TtsDiagnostics {
        let sample_rate = self
            .runtime
            .as_ref()
            .map(|runtime| runtime.dac.config().sample_rate)
            .unwrap_or_else(|| FishS2DacConfig::current().sample_rate);
        FishS2TtsDiagnostics {
            model_family: "fish_s2_tts",
            variant: self.variant.to_string(),
            native_runtime_loaded: self.runtime.is_some(),
            sample_rate,
            num_codebooks: self.config.num_codebooks,
            codebook_size: self.config.codebook_size,
            semantic_start_token_id: self.config.semantic_start_token_id,
            semantic_end_token_id: self.config.semantic_end_token_id,
            max_seq_len: self.config.max_seq_len,
            codec_support: fish_s2_codec_support_name(self.codec.support),
            precision: self
                .runtime
                .as_ref()
                .map(|runtime| runtime.precision.diagnostics()),
        }
    }

    pub fn generate_with_reference(
        &self,
        _text: &str,
        _reference: FishS2Reference,
        _params: FishS2GenerationParams,
    ) -> Result<FishS2GenerationOutput> {
        Err(Error::InferenceError(
            "Fish S2 generation requires lifecycle-owned slow and fast physical page leases"
                .to_string(),
        ))
    }

    /// Bytes of one retained state element at the precision the model runs.
    pub(crate) fn state_element_bytes(&self) -> usize {
        self.runtime
            .as_ref()
            .map_or(4, |runtime| runtime.dtype.size_in_bytes())
    }

    pub(crate) fn physical_state_spec(
        &self,
        stage_graphs: &[&[StageDescriptor]],
    ) -> Result<FishS2PhysicalStateSpec> {
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            Error::ModelLoadError(
                "Fish Audio S2 Pro metadata was loaded without native inference modules"
                    .to_string(),
            )
        })?;
        physical::fish_s2_physical_state_spec(&self.config, runtime.dtype, stage_graphs)
    }

    pub(crate) fn generate_with_reference_physical(
        &self,
        text: &str,
        reference: FishS2Reference,
        params: FishS2GenerationParams,
        slow_cache: &mut PhysicalPagedKvCache,
        fast_cache: &mut PhysicalPagedKvCache,
    ) -> Result<FishS2GenerationOutput> {
        self.codec.ensure_native_supported()?;
        let runtime = self.runtime.as_ref().ok_or_else(|| {
            Error::ModelLoadError(
                "Fish Audio S2 Pro metadata was loaded without native inference modules"
                    .to_string(),
            )
        })?;
        runtime.generate_with_reference_physical(
            &self.config,
            text,
            reference,
            params,
            slow_cache,
            fast_cache,
        )
    }
}

/// Slow-stack positions the load-time self-check runs.
const FISH_S2_PROBE_TOKENS: usize = 16;

/// The load-time self-check (report 13, FSH-11): see
/// [`fish_s2_numeric_probe`]. Every probed tensor must be finite and below
/// the F16-safe range.
fn fish_s2_numeric_self_check(
    slow: &FishS2SlowTransformer,
    fast: &FishS2FastDecoder,
    device: &candle_core::Device,
) -> NumericSelfCheckReport {
    run_numeric_self_check("Fish S2", F16_SAFE_MAX_ABS, || {
        fish_s2_numeric_probe(slow, fast, device)
    })
}

/// Xorshift text token ids (below the semantic range) for
/// [`FISH_S2_PROBE_TOKENS`] positions through every slow layer, on a
/// scratch cache in the weights' dtype, and the semantic head; then one
/// fast-codebook frame from the last slow hidden state with an xorshift
/// semantic code. Returns the slow hidden states and logits and the fast
/// logits of every depth.
fn fish_s2_numeric_probe(
    slow: &FishS2SlowTransformer,
    fast: &FishS2FastDecoder,
    device: &candle_core::Device,
) -> Result<Vec<(String, Tensor)>> {
    let cfg = slow.config();
    let text_tokens = cfg.semantic_start_token_id.max(1);
    let draws = xorshift_values(FISH_S2_PROBE_TOKENS + 1, 0xF15A, 1.0);
    let pick = |draw: f32, limit: u32| ((draw.abs() * limit as f32) as u32).min(limit - 1);
    let ids = draws[..FISH_S2_PROBE_TOKENS]
        .iter()
        .map(|draw| pick(*draw, text_tokens))
        .collect::<Vec<_>>();
    let mut values = vec![ids];
    values.extend((0..cfg.num_codebooks).map(|_| vec![0u32; FISH_S2_PROBE_TOKENS]));
    let prompt = tokenizer::FishS2ConditioningPrompt {
        values,
        vq_mask: vec![false; FISH_S2_PROBE_TOKENS],
        prompt_length: FISH_S2_PROBE_TOKENS,
    };
    let embeds = slow.embed_prompt(&prompt)?;
    let dtype = embeds.dtype();
    let mut slow_cache = PhysicalPagedKvCache::scratch_for_probe(
        device,
        dtype,
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.head_dim,
        FISH_S2_PROBE_TOKENS,
    )?;
    let output = slow.forward_embeds(&embeds, 0, &mut slow_cache, true)?;
    let fast_cfg = fast.config();
    let mut fast_cache = PhysicalPagedKvCache::scratch_for_probe(
        device,
        dtype,
        fast_cfg.num_hidden_layers,
        fast_cfg.num_key_value_heads,
        fast_cfg.head_dim,
        fast_cfg.num_codebooks,
    )?;
    let last = output
        .hidden_states
        .narrow(1, FISH_S2_PROBE_TOKENS - 1, 1)?;
    let semantic_code = pick(draws[FISH_S2_PROBE_TOKENS], fast_cfg.codebook_size as u32);
    let fast_logits = fast.numeric_probe(&last, semantic_code, &mut fast_cache)?;
    Ok(vec![
        ("slow_hidden".to_string(), output.hidden_states),
        ("slow_logits".to_string(), output.logits),
        ("fast_logits".to_string(), fast_logits),
    ])
}

fn next_fish_s2_model_identity() -> Result<u64> {
    NEXT_FISH_S2_MODEL_IDENTITY
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| Error::ModelLoadError("Fish S2 model identity space exhausted".into()))
}

impl FishS2NativeRuntime {
    fn load(
        model_dir: &Path,
        config: &FishS2Config,
        codec: &FishS2CodecArtifact,
        device: DeviceProfile,
    ) -> Result<Self> {
        codec.ensure_native_supported()?;
        let tokenizer = FishS2PromptTokenizer::load(model_dir, config)?;
        let dtype_override = std::env::var(FISH_S2_DTYPE_ENV).ok();
        let selection = DTypeSelection {
            dtype: FishS2Weights::resolve_dtype(model_dir, &device, dtype_override.as_deref())?,
            reason: match dtype_override.as_deref().map(str::trim) {
                Some(raw) if !raw.is_empty() => format!("{FISH_S2_DTYPE_ENV}={raw}").into(),
                _ => format!("Fish S2 {:?} dtype policy", device.kind).into(),
            },
        };
        let eos = tokenizer.specials().eos;
        // On Metal a dtype below F32 must keep the trunk's activations finite
        // and in range (FSH-11: norm gains reach 48), or the slow and fast
        // transformers reload in F32.
        let ((slow, fast, device), precision) = load_with_precision_fallback(
            "Fish S2",
            &selection,
            device.kind.is_metal(),
            DType::F32,
            |dtype| -> Result<_> {
                let weights = FishS2Weights::load_with_dtype(model_dir, device.clone(), dtype)?;
                let mut slow = FishS2SlowTransformer::load(
                    FishS2SlowConfig::from_config(config)?,
                    weights.var_builder(),
                )?;
                slow.configure_semantic_head(eos)?;
                let fast = FishS2FastDecoder::load(
                    FishS2FastConfig::from_config(config)?,
                    weights.var_builder(),
                )?;
                Ok((slow, fast, weights.device().clone()))
            },
            |(slow, fast, device)| fish_s2_numeric_self_check(slow, fast, device),
        )?;
        tracing::info!(
            "Fish S2 precision: {:?} ({})",
            precision.loaded,
            precision.reason
        );
        let codec_weights = codec.load_weights(&device, DType::F32)?;
        let dac = FishS2DacDecoder::load(FishS2DacConfig::current(), codec_weights.var_builder())?;
        let semantic_allowed_mask = slow.semantic_allowed_mask(eos)?;

        Ok(Self {
            tokenizer,
            slow,
            fast,
            dac,
            semantic_allowed_mask,
            dtype: precision.loaded,
            precision,
        })
    }

    fn generate_with_reference_physical(
        &self,
        config: &FishS2Config,
        text: &str,
        reference: FishS2Reference,
        params: FishS2GenerationParams,
        slow_cache: &mut PhysicalPagedKvCache,
        fast_cache: &mut PhysicalPagedKvCache,
    ) -> Result<FishS2GenerationOutput> {
        let text = text.trim();
        let reference_text = reference.text.trim();
        if text.is_empty() {
            return Err(Error::InvalidInput(
                "Fish S2 TTS text input cannot be empty".to_string(),
            ));
        }
        if reference_text.is_empty() {
            return Err(Error::InvalidInput(
                "Fish S2 TTS reference_text cannot be empty".to_string(),
            ));
        }
        if reference.audio_samples.is_empty() {
            return Err(Error::InvalidInput(
                "Fish S2 TTS reference_audio cannot be empty".to_string(),
            ));
        }
        params.validate()?;

        let total_started = Instant::now();
        let started = Instant::now();
        let reference_codes = self
            .dac
            .encode_reference_audio(&reference.audio_samples, reference.sample_rate)?;
        let reference_encode_ms = elapsed_ms(started);

        let started = Instant::now();
        let prompt = self.tokenizer.build_reference_voice_prompt(
            config,
            reference_text,
            reference_codes,
            text,
        )?;
        let prompt_build_ms = elapsed_ms(started);
        if prompt.prompt_length >= config.max_seq_len {
            return Err(Error::InvalidInput(format!(
                "Fish S2 prompt length {} exceeds max_seq_len {}",
                prompt.prompt_length, config.max_seq_len
            )));
        }

        let max_frames = effective_frame_budget(
            prompt.prompt_length,
            config.max_seq_len,
            slow_cache.capacity_tokens(),
            params.max_frames,
        )?;

        let mut semantic_sampler = FishS2SemanticSampler::from_params(&params);
        let mut fast_sampler = FishS2Sampler::with_top_k(
            params.temperature,
            params.top_p,
            params.top_k,
            params.seed.wrapping_add(1),
        );
        let im_end_token_id = self.tokenizer.specials().eos;

        if slow_cache.context_len() != 0 {
            return Err(Error::InvalidInput(format!(
                "Fish S2 slow invocation cache must begin empty, got context {}",
                slow_cache.context_len()
            )));
        }
        let started = Instant::now();
        let mut slow_output = self.slow.forward_prompt(&prompt, slow_cache, false)?;
        let slow_prefill_ms = elapsed_ms(started);
        let mut generated_codebooks = vec![Vec::new(); config.num_codebooks];
        let mut recent_semantic_tokens = Vec::with_capacity(RAS_WIN_SIZE);
        let mut stop_reason = FishS2StopReason::FrameLimit;

        let started = Instant::now();
        for frame_index in 0..max_frames {
            let allow_im_end = generated_codebooks
                .first()
                .map(|codebook| !codebook.is_empty())
                .unwrap_or(false);
            let semantic_index = sample_semantic_token(
                &slow_output.logits,
                &self.semantic_allowed_mask,
                self.slow.eos_logit_index(im_end_token_id),
                allow_im_end,
                &recent_semantic_tokens,
                &mut semantic_sampler,
            )?;
            let semantic_token_id = self.slow.token_id_from_logit(semantic_index)?;
            if semantic_token_id == im_end_token_id {
                stop_reason = FishS2StopReason::EndOfSpeech;
                break;
            }

            let frame = self.fast.generate_frame(
                semantic_token_id,
                &slow_output.hidden_states,
                &mut fast_sampler,
                fast_cache,
            )?;
            append_generated_frame(&mut generated_codebooks, &frame)?;

            recent_semantic_tokens.push(semantic_index);
            if recent_semantic_tokens.len() > RAS_WIN_SIZE {
                recent_semantic_tokens.remove(0);
            }

            if frame_index + 1 == max_frames {
                break;
            }
            let frame_prompt = generated_frame_prompt(config.num_codebooks, &frame)?;
            let frame_embeds = self.slow.embed_prompt(&frame_prompt)?;
            let start_pos = slow_cache.context_len();
            slow_output = self
                .slow
                .forward_embeds(&frame_embeds, start_pos, slow_cache, false)?;
        }
        let ar_decode_ms = elapsed_ms(started);
        stop_reason.require_complete()?;

        let frames_generated = generated_codebooks.first().map(Vec::len).unwrap_or(0);
        if frames_generated == 0 {
            return Err(Error::InferenceError(
                "Fish S2 generation produced no audio frames".to_string(),
            ));
        }

        let started = Instant::now();
        let samples = self.dac.decode_vq_codes(&FishS2VqCodes {
            codebooks: generated_codebooks,
        })?;
        let dac_decode_ms = elapsed_ms(started);
        if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
            return Err(Error::InferenceError(
                "Fish S2 DAC produced empty or non-finite audio".to_string(),
            ));
        }

        let sample_rate = self.dac.config().sample_rate;
        Ok(FishS2GenerationOutput {
            samples,
            sample_rate,
            frames_generated,
            diagnostics: FishS2TtsGenerationDiagnostics {
                model_family: "fish_s2_tts",
                sample_rate,
                prompt_tokens: prompt.prompt_length,
                max_frames,
                frames_generated,
                stop_reason,
                temperature: params.temperature,
                top_p: params.top_p,
                top_k: params.top_k,
                seed: params.seed,
                repetition_aware: params.repetition_aware,
                reference_encode_ms,
                prompt_build_ms,
                slow_prefill_ms,
                ar_decode_ms,
                dac_decode_ms,
                total_model_ms: elapsed_ms(total_started),
            },
        })
    }
}

impl Default for FishS2GenerationParams {
    fn default() -> Self {
        Self {
            max_frames: ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES,
            // fish-speech serving defaults at 1ff3c3ed (2026-10-13).
            temperature: 0.7,
            top_p: 0.7,
            top_k: 30,
            seed: 0,
            repetition_aware: true,
        }
    }
}

impl FishS2GenerationParams {
    pub fn validate(&self) -> Result<()> {
        if self.max_frames == 0 {
            return Err(Error::InvalidInput(
                "Fish S2 max_frames must be greater than zero".into(),
            ));
        }
        sampling::validate_policy(self.temperature, self.top_p)
    }
}

#[derive(Clone)]
struct FishS2SemanticSampler {
    sampler: FishS2Sampler,
    repetition_aware: bool,
}

impl FishS2SemanticSampler {
    fn from_params(params: &FishS2GenerationParams) -> Self {
        Self {
            sampler: FishS2Sampler::with_top_k(
                params.temperature,
                params.top_p,
                params.top_k,
                params.seed,
            ),
            repetition_aware: params.repetition_aware,
        }
    }

    #[cfg(test)]
    fn new(temperature: f32, top_p: f32, seed: u64) -> Self {
        Self::from_params(&FishS2GenerationParams {
            temperature,
            top_p,
            seed,
            ..Default::default()
        })
    }
}

fn sample_semantic_token(
    logits: &Tensor,
    allowed_mask: &[bool],
    im_end_token_id: u32,
    allow_im_end: bool,
    previous_semantic_tokens: &[u32],
    sampler: &mut FishS2SemanticSampler,
) -> Result<u32> {
    sampling::validate_policy(sampler.sampler.temperature, sampler.sampler.top_p)?;
    let row = last_logits_row(logits)?;
    if row.dim(0)? != allowed_mask.len() {
        return Err(Error::InferenceError(format!(
            "Fish S2 semantic mask length {} does not match logits length {}",
            allowed_mask.len(),
            row.dim(0)?,
        )));
    }
    let mut first_frame_mask;
    let sampling_mask = if allow_im_end {
        allowed_mask
    } else {
        first_frame_mask = allowed_mask.to_vec();
        if let Some(allowed) = first_frame_mask.get_mut(im_end_token_id as usize) {
            *allowed = false;
        }
        &first_frame_mask
    };
    let use_ras = sampler.repetition_aware && sampler.sampler.temperature > 0.0;
    let required_top_p = if use_ras {
        sampler.sampler.top_p.max(RAS_HIGH_TOP_P)
    } else {
        sampler.sampler.top_p
    };
    let distribution = FishS2SamplingDistribution::from_logits(
        &row,
        Some(sampling_mask),
        sampler.sampler.top_k,
        required_top_p,
        "semantic",
    )?;
    let normal = sampler.sampler.sample(&distribution)?;
    if use_ras {
        // Upstream draws both policies each frame, then chooses the retry only
        // when a semantic token repeats in the last ten frames. RNG algorithms
        // differ from Torch; this records and replays the native Rust policy.
        let high =
            sampler
                .sampler
                .sample_with_policy(&distribution, RAS_HIGH_TEMP, RAS_HIGH_TOP_P)?;
        let recent = &previous_semantic_tokens
            [previous_semantic_tokens.len().saturating_sub(RAS_WIN_SIZE)..];
        if normal != im_end_token_id && recent.contains(&normal) {
            return Ok(high);
        }
    }
    Ok(normal)
}

fn last_logits_row(logits: &Tensor) -> Result<Tensor> {
    match logits.rank() {
        3 => {
            let (batch, seq_len, _) = logits.dims3()?;
            if batch != 1 || seq_len == 0 {
                return Err(Error::InferenceError(format!(
                    "Fish S2 semantic logits require one non-empty sequence, got {:?}",
                    logits.dims()
                )));
            }
            logits.i((0, seq_len - 1)).map_err(Error::from)
        }
        2 => logits.i(0).map_err(Error::from),
        1 => Ok(logits.clone()),
        _ => Err(Error::InferenceError(format!(
            "Fish S2 expected semantic logits rank 1, 2, or 3, got {:?}",
            logits.dims()
        ))),
    }
}

fn append_generated_frame(
    generated_codebooks: &mut [Vec<u32>],
    frame: &FishS2GeneratedFrame,
) -> Result<()> {
    if frame.codebooks.len() != generated_codebooks.len() {
        return Err(Error::InferenceError(format!(
            "Fish S2 fast decoder returned {} codebooks, expected {}",
            frame.codebooks.len(),
            generated_codebooks.len()
        )));
    }
    for (row, code) in generated_codebooks.iter_mut().zip(&frame.codebooks) {
        row.push(*code);
    }
    Ok(())
}

fn generated_frame_prompt(
    num_codebooks: usize,
    frame: &FishS2GeneratedFrame,
) -> Result<FishS2ConditioningPrompt> {
    if frame.codebooks.len() != num_codebooks {
        return Err(Error::InferenceError(format!(
            "Fish S2 frame has {} codebooks, expected {num_codebooks}",
            frame.codebooks.len()
        )));
    }
    let mut values = vec![vec![0u32; 1]; num_codebooks + 1];
    values[0][0] = frame.semantic_token_id;
    for (idx, code) in frame.codebooks.iter().enumerate() {
        values[idx + 1][0] = *code;
    }
    Ok(FishS2ConditioningPrompt {
        values,
        vq_mask: vec![true],
        prompt_length: 1,
    })
}

pub(crate) fn effective_frame_budget(
    prompt_tokens: usize,
    model_context: usize,
    cache_capacity: usize,
    requested: usize,
) -> Result<usize> {
    let context = model_context.min(cache_capacity);
    let available = context.checked_sub(prompt_tokens).filter(|available| *available > 0)
        .ok_or_else(|| Error::ContextLengthExceeded(format!(
            "Fish S2 prompt length {prompt_tokens} leaves no output room in effective context {context}",
        )))?;
    // Reserve one context position beyond the full output budget for EOS. Never
    // silently shrink the admitted output budget after exact tokenization.
    if requested == 0 || requested >= available {
        return Err(Error::ContextLengthExceeded(format!(
            "Fish S2 segment does not fit effective context: prompt_tokens={prompt_tokens}, requested_frames={requested}, eos_margin=1, context={context}; split the text segment or shorten the voice reference before generating audio",
        )));
    }
    Ok(requested)
}

fn elapsed_ms(started: Instant) -> f32 {
    started.elapsed().as_secs_f32() * 1000.0
}

fn fish_s2_codec_support_name(support: FishS2CodecSupport) -> &'static str {
    match support {
        FishS2CodecSupport::NativePthStateDict => "native_pth_state_dict",
    }
}

#[cfg(test)]
mod smoke;

#[cfg(test)]
mod streaming_smoke;

#[cfg(test)]
mod precision_tests {
    use super::*;
    use crate::kernels::numeric_check::xorshift_tensor;
    use candle_core::Device;
    use candle_nn::VarBuilder;
    use std::collections::BTreeMap;

    /// A tiny two-layer slow transformer and one-layer fast decoder with
    /// xorshift weights and every RMSNorm gain set to `norm_gain`, in `dtype`.
    fn tiny_random_parts(
        device: &Device,
        dtype: DType,
        norm_gain: f32,
    ) -> (FishS2SlowTransformer, FishS2FastDecoder) {
        let slow_cfg = FishS2SlowConfig {
            vocab_size: 32,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            max_seq_len: 32,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            codebook_size: 8,
            num_codebooks: 2,
            semantic_start_token_id: 20,
            semantic_end_token_id: 27,
            scale_codebook_embeddings: true,
        };
        let fast_cfg = FishS2FastConfig {
            input_hidden_size: 4,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 2,
            codebook_size: 8,
            num_codebooks: 3,
            semantic_start_token_id: 20,
            semantic_end_token_id: 27,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
        };
        let mut shapes = BTreeMap::new();
        shapes.insert("embed_tokens.weight".to_string(), vec![32, 4]);
        shapes.insert("codebook_embeddings.weight".to_string(), vec![16, 4]);
        shapes.insert("fast_project_in.weight".to_string(), vec![4, 4]);
        shapes.insert("fast_project_in.bias".to_string(), vec![4]);
        shapes.insert("fast_embeddings.weight".to_string(), vec![8, 4]);
        shapes.insert("fast_output.weight".to_string(), vec![8, 4]);
        let layer = |prefix: &str, shapes: &mut BTreeMap<String, Vec<usize>>| {
            for (name, shape) in [
                ("self_attn.qkv_proj.weight", vec![8, 4]),
                ("self_attn.o_proj.weight", vec![4, 4]),
                ("mlp.gate_proj.weight", vec![8, 4]),
                ("mlp.up_proj.weight", vec![8, 4]),
                ("mlp.down_proj.weight", vec![4, 8]),
            ] {
                shapes.insert(format!("{prefix}.{name}"), shape);
            }
        };
        layer("layers.0", &mut shapes);
        layer("layers.1", &mut shapes);
        layer("fast_layers.0", &mut shapes);
        let mut tensors = shapes
            .into_iter()
            .enumerate()
            .map(|(index, (name, shape))| {
                let tensor = xorshift_tensor(&shape, 0xF500 + index as u64, 0.5, device).unwrap();
                (name, tensor)
            })
            .collect::<std::collections::HashMap<_, _>>();
        let gain = |dim: usize| Tensor::full(norm_gain, dim, device).unwrap();
        for name in [
            "norm.weight",
            "layers.0.input_layernorm.weight",
            "layers.0.post_attention_layernorm.weight",
            "layers.1.input_layernorm.weight",
            "layers.1.post_attention_layernorm.weight",
            "fast_norm.weight",
            "fast_layers.0.input_layernorm.weight",
            "fast_layers.0.post_attention_layernorm.weight",
        ] {
            tensors.insert(name.to_string(), gain(4));
        }
        for name in [
            "layers.0.self_attn.q_norm.weight",
            "layers.0.self_attn.k_norm.weight",
            "layers.1.self_attn.q_norm.weight",
            "layers.1.self_attn.k_norm.weight",
        ] {
            tensors.insert(
                name.to_string(),
                Tensor::ones(2, DType::F32, device).unwrap(),
            );
        }
        let vb = VarBuilder::from_tensors(tensors, dtype, device);
        let mut slow = FishS2SlowTransformer::load(slow_cfg, vb.clone()).unwrap();
        slow.configure_semantic_head(1).unwrap();
        let fast = FishS2FastDecoder::load(fast_cfg, vb).unwrap();
        (slow, fast)
    }

    /// FSH-11: the probe runs every slow layer and the semantic head, then a
    /// fast frame; on a healthy tiny model it passes in F16 and tracks F32.
    #[test]
    fn fish_probe_passes_in_f16_and_tracks_f32() {
        let device = Device::Cpu;
        let report = |dtype| {
            let (slow, fast) = tiny_random_parts(&device, dtype, 1.0);
            fish_s2_numeric_self_check(&slow, &fast, &device)
        };
        let full = report(DType::F32);
        let half = report(DType::F16);
        assert!(full.passed, "{}", full.reason);
        assert!(half.passed, "{}", half.reason);
        let names = half
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, ["slow_hidden", "slow_logits", "fast_logits"]);
        for (half, full) in half.tensors.iter().zip(&full.tensors) {
            assert!(
                (half.max_abs - full.max_abs).abs() <= 0.05 * full.max_abs.max(1.0),
                "{}: F16 max |x| {} vs F32 {}",
                half.name,
                half.max_abs,
                full.max_abs
            );
        }
    }

    /// FSH-11: when large norm gains push F16 activations out of range, the
    /// self-check fails and the transformers reload in F32; healthy gains
    /// keep F16.
    #[test]
    fn fish_self_check_falls_back_to_f32_when_f16_activations_overflow() {
        let device = Device::Cpu;
        let selection = DTypeSelection {
            dtype: DType::F16,
            reason: "test policy".into(),
        };
        for (norm_gain, expected) in [(1.0f32, DType::F16), (3.0e3, DType::F32)] {
            let (_, precision) = load_with_precision_fallback(
                "tiny Fish S2",
                &selection,
                true,
                DType::F32,
                |dtype| Ok::<_, Error>(tiny_random_parts(&device, dtype, norm_gain)),
                |(slow, fast)| fish_s2_numeric_self_check(slow, fast, &device),
            )
            .unwrap();
            assert_eq!(precision.loaded, expected, "{}", precision.reason);
            assert_eq!(precision.fell_back(), expected == DType::F32);
        }
    }

    /// FSH-11: the probe runs on Metal in F16 and passes on the healthy
    /// tiny model, tracking CPU F32.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_fish_probe_passes_in_f16() {
        let Some((metal, _lease)) =
            crate::backends::metal_test_device_or_skip("metal_fish_probe_passes_in_f16")
        else {
            return;
        };
        let (slow, fast) = tiny_random_parts(&metal, DType::F16, 1.0);
        let report = fish_s2_numeric_self_check(&slow, &fast, &metal);
        assert!(report.passed, "{}", report.reason);
        let (slow, fast) = tiny_random_parts(&Device::Cpu, DType::F32, 1.0);
        let cpu = fish_s2_numeric_self_check(&slow, &fast, &Device::Cpu);
        for (metal, cpu) in report.tensors.iter().zip(&cpu.tensors) {
            assert!(
                (metal.max_abs - cpu.max_abs).abs() <= 0.05 * cpu.max_abs.max(1.0),
                "{}: Metal F16 {} vs CPU F32 {}",
                metal.name,
                metal.max_abs,
                cpu.max_abs
            );
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn full_output_budget_and_eos_must_fit_exact_context() {
        assert_eq!(
            super::effective_frame_budget(224, 8192, 256, 31).unwrap(),
            31
        );
        for requested in [32, 512] {
            let error = super::effective_frame_budget(224, 8192, 256, requested).unwrap_err();
            assert!(error
                .to_string()
                .contains("segment does not fit effective context"));
            assert_eq!(error.kind(), crate::error::ErrorKind::ContextLengthExceeded);
        }
        let error = super::effective_frame_budget(256, 8192, 256, 1).unwrap_err();
        assert_eq!(error.kind(), crate::error::ErrorKind::ContextLengthExceeded);
    }

    #[test]
    fn frame_limit_is_incomplete_speech_while_eos_is_success() {
        use super::FishS2StopReason;
        FishS2StopReason::EndOfSpeech.require_complete().unwrap();
        assert!(FishS2StopReason::FrameLimit
            .require_complete()
            .unwrap_err()
            .to_string()
            .contains("generation incomplete: frame_limit"));
        assert_eq!(
            serde_json::to_string(&FishS2StopReason::EndOfSpeech).unwrap(),
            "\"im_end\""
        );
        assert_eq!(
            serde_json::to_string(&FishS2StopReason::FrameLimit).unwrap(),
            "\"max_frames\""
        );
    }

    use super::*;
    use candle_core::Device;

    #[test]
    fn generation_params_default_to_s2_frame_budget() {
        let params = FishS2GenerationParams::default();
        assert_eq!(
            params.max_frames,
            ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES
        );
        // fish-speech serving defaults at 1ff3c3ed.
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_p, 0.7);
    }

    #[test]
    fn semantic_sampler_obeys_allowed_mask_for_argmax() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![100.0f32, 2.0, 3.0, 4.0], (1, 1, 4), &device).unwrap();
        let allowed = vec![false, true, false, true];
        let mut sampler = FishS2SemanticSampler::new(0.0, 1.0, 0);
        let token = sample_semantic_token(&logits, &allowed, 1, true, &[], &mut sampler).unwrap();
        assert_eq!(token, 3);
    }

    #[test]
    fn semantic_sampler_masks_im_end_before_first_audio_frame() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.0f32, 100.0, 98.0, 4.0], (1, 1, 4), &device).unwrap();
        let allowed = vec![false, true, true, true];
        let mut sampler = FishS2SemanticSampler::new(0.0, 1.0, 0);
        let token = sample_semantic_token(&logits, &allowed, 1, false, &[], &mut sampler).unwrap();
        assert_eq!(token, 2);
    }

    #[test]
    fn semantic_sampler_allows_im_end_after_audio_frame() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.0f32, 100.0, 98.0, 4.0], (1, 1, 4), &device).unwrap();
        let allowed = vec![false, true, true, true];
        let mut sampler = FishS2SemanticSampler::new(0.0, 1.0, 0);
        let token = sample_semantic_token(&logits, &allowed, 1, true, &[2], &mut sampler).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn generated_frame_prompt_has_upstream_codebook_shape() {
        let frame = FishS2GeneratedFrame {
            semantic_token_id: 42,
            codebooks: vec![4, 5, 6],
        };
        let prompt = generated_frame_prompt(3, &frame).unwrap();
        assert_eq!(prompt.prompt_length, 1);
        assert_eq!(prompt.vq_mask, vec![true]);
        assert_eq!(prompt.values, vec![vec![42], vec![4], vec![5], vec![6]]);
    }
}
