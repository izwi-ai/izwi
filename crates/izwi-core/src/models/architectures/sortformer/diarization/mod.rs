mod nemo;

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::ops;
use candle_nn::{
    batch_norm, layer_norm, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, LayerNorm, Linear, Module,
    ModuleT, VarBuilder,
};
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::backends::{DeviceKind, DeviceProfile};
use crate::error::{Error, Result};
use crate::model::ModelVariant;
use crate::models::shared::weights::mlx;
use crate::runtime::{DiarizationConfig, DiarizationResult, DiarizationSegment};

use nemo::{ensure_sortformer_artifacts, SortformerArtifacts};

const TARGET_SAMPLE_RATE: u32 = 16_000;
const MAX_SUPPORTED_SPEAKERS: usize = 4;
const DEFAULT_MIN_SPEECH_MS: f32 = 240.0;
const DEFAULT_MIN_SILENCE_MS: f32 = 200.0;
const PREEMPH: f32 = 0.97;
const LOG_GUARD: f32 = 5.960_464_5e-8;
const NORMALIZE_EPS: f32 = 1e-5;
const REALTIME_VAD_THRESHOLD: f32 = 0.02;
const TS_VAD_FRAME_LENGTH_SECS: f32 = 0.01;
const TS_VAD_UNIT_FRAME_COUNT: usize = 8;

#[derive(Debug, Clone, serde::Deserialize)]
struct SortformerModelConfig {
    sample_rate: Option<u32>,
    max_num_of_spks: Option<usize>,
    streaming_mode: Option<bool>,
    preprocessor: Option<SortformerPreprocessorConfig>,
    encoder: Option<SortformerEncoderConfig>,
    sortformer_modules: Option<SortformerModulesConfig>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct SortformerPreprocessorConfig {
    sample_rate: Option<u32>,
    window_size: Option<f32>,
    window_stride: Option<f32>,
    features: Option<usize>,
    n_fft: Option<usize>,
    normalize: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct SortformerEncoderConfig {
    xscaling: Option<bool>,
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
struct SortformerModulesConfig {
    fc_d_model: Option<usize>,
    subsampling_factor: Option<usize>,
    spkcache_len: Option<usize>,
    fifo_len: Option<usize>,
    chunk_len: Option<usize>,
    spkcache_update_period: Option<usize>,
    chunk_left_context: Option<usize>,
    chunk_right_context: Option<usize>,
    spkcache_sil_frames_per_spk: Option<usize>,
    pred_score_threshold: Option<f32>,
    scores_boost_latest: Option<f32>,
    sil_threshold: Option<f32>,
    strong_boost_rate: Option<f32>,
    weak_boost_rate: Option<f32>,
    min_pos_scores_rate: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SortformerStreamingProfile {
    Model,
    LowLatency,
    HighLatency,
}

#[derive(Debug, Clone, Copy)]
struct SortformerStreamingConfig {
    fc_d_model: usize,
    subsampling_factor: usize,
    spkcache_len: usize,
    fifo_len: usize,
    chunk_len: usize,
    spkcache_update_period: usize,
    chunk_left_context: usize,
    chunk_right_context: usize,
    spkcache_sil_frames_per_spk: usize,
    pred_score_threshold: f32,
    scores_boost_latest: f32,
    sil_threshold: f32,
    strong_boost_rate: f32,
    weak_boost_rate: f32,
    min_pos_scores_rate: f32,
}

impl SortformerStreamingConfig {
    fn validate(self) -> Result<Self> {
        let min_spkcache_len = (1 + self.spkcache_sil_frames_per_spk) * MAX_SUPPORTED_SPEAKERS;
        if self.subsampling_factor == 0
            || self.fc_d_model == 0
            || self.chunk_len == 0
            || self.spkcache_update_period == 0
        {
            return Err(Error::ModelLoadError(
                "Sortformer streaming config contains zero-valued required fields".to_string(),
            ));
        }
        if self.spkcache_len < min_spkcache_len {
            return Err(Error::ModelLoadError(format!(
                "Sortformer spkcache_len {} is smaller than the required minimum {}",
                self.spkcache_len, min_spkcache_len
            )));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone)]
struct SortformerStreamingChunkPlan {
    feature_start: usize,
    feature_end: usize,
    left_offset: usize,
    right_offset: usize,
}

#[derive(Debug, Clone)]
struct SortformerStreamingState {
    spkcache: Vec<Vec<f32>>,
    spkcache_preds: Option<Vec<[f32; MAX_SUPPORTED_SPEAKERS]>>,
    fifo: Vec<Vec<f32>>,
    fifo_preds: Vec<[f32; MAX_SUPPORTED_SPEAKERS]>,
    mean_sil_emb: Vec<f32>,
    n_sil_frames: usize,
}

impl SortformerStreamingState {
    fn new(emb_dim: usize) -> Self {
        Self {
            spkcache: Vec::new(),
            spkcache_preds: None,
            fifo: Vec::new(),
            fifo_preds: Vec::new(),
            mean_sil_emb: vec![0.0; emb_dim],
            n_sil_frames: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct SortformerCacheCandidate {
    flat_index: usize,
    frame_index: Option<usize>,
    score: f32,
}

const SORTFORMER_SCORE_BOOST_DELTA: f32 = 0.693_147_2;

pub struct SortformerDiarizerModel {
    variant: ModelVariant,
    _artifacts: SortformerArtifacts,
    _checkpoint_tensor_count: usize,
    model: SortformerInferenceModel,
}

impl SortformerDiarizerModel {
    pub fn load(
        model_dir: &Path,
        variant: ModelVariant,
        device_profile: DeviceProfile,
    ) -> Result<Self> {
        if !variant.is_diarization() {
            return Err(Error::InvalidInput(format!(
                "Variant {} is not a Sortformer diarization model",
                variant.dir_name()
            )));
        }

        let artifacts = ensure_sortformer_artifacts(model_dir, variant)?;
        let tensor_info =
            candle_core::pickle::read_pth_tensor_info(&artifacts.checkpoint_path, false, None)
                .map_err(|e| {
                    Error::ModelLoadError(format!(
                        "Failed to inspect Sortformer checkpoint {}: {}",
                        artifacts.checkpoint_path.display(),
                        e
                    ))
                })?;

        let config: SortformerModelConfig = serde_yaml::from_str(
            &std::fs::read_to_string(&artifacts.model_config_path).map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed reading Sortformer config {}: {}",
                    artifacts.model_config_path.display(),
                    e
                ))
            })?,
        )
        .map_err(|e| {
            Error::ModelLoadError(format!(
                "Failed parsing Sortformer config {}: {}",
                artifacts.model_config_path.display(),
                e
            ))
        })?;

        let sample_rate = config.sample_rate.unwrap_or(TARGET_SAMPLE_RATE);
        if sample_rate != TARGET_SAMPLE_RATE {
            return Err(Error::ModelLoadError(format!(
                "Unsupported Sortformer sample rate {sample_rate}; expected {TARGET_SAMPLE_RATE}"
            )));
        }

        let num_spks = config.max_num_of_spks.unwrap_or(MAX_SUPPORTED_SPEAKERS);
        if num_spks != MAX_SUPPORTED_SPEAKERS {
            return Err(Error::ModelLoadError(format!(
                "Unsupported Sortformer speaker count {num_spks}; expected {MAX_SUPPORTED_SPEAKERS}"
            )));
        }

        let device = sortformer_model_device(&device_profile);
        let vb =
            VarBuilder::from_pth(&artifacts.checkpoint_path, DType::F32, &device).map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed to load Sortformer checkpoint {}: {}",
                    artifacts.checkpoint_path.display(),
                    e
                ))
            })?;

        let preprocessor_cfg =
            config
                .preprocessor
                .clone()
                .unwrap_or(SortformerPreprocessorConfig {
                    sample_rate: Some(TARGET_SAMPLE_RATE),
                    window_size: Some(0.025),
                    window_stride: Some(0.01),
                    features: Some(128),
                    n_fft: Some(512),
                    normalize: Some("NA".to_string()),
                });

        let modules_cfg = config.sortformer_modules.clone().unwrap_or_default();
        let streaming_mode = config.streaming_mode.unwrap_or(false);
        let model = SortformerInferenceModel::load(
            &vb,
            preprocessor_cfg,
            variant,
            streaming_mode,
            config.encoder.clone(),
            modules_cfg.clone(),
            device.clone(),
        )?;

        Ok(Self {
            variant,
            _artifacts: artifacts,
            _checkpoint_tensor_count: tensor_info.len(),
            model,
        })
    }

    pub fn diarize(
        &self,
        audio: &[f32],
        sample_rate: u32,
        config: &DiarizationConfig,
    ) -> Result<DiarizationResult> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }
        if sample_rate == 0 {
            return Err(Error::InvalidInput("Invalid sample rate: 0".to_string()));
        }

        let samples = if sample_rate == TARGET_SAMPLE_RATE {
            audio.to_vec()
        } else {
            resample_linear(audio, sample_rate, TARGET_SAMPLE_RATE)
        };

        let duration_secs = samples.len() as f32 / TARGET_SAMPLE_RATE as f32;
        if samples.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                duration_secs,
                speaker_count: 0,
            });
        }

        let (speaker_probs, frame_stride_samples) =
            self.model.infer_speaker_probabilities(&samples)?;
        if speaker_probs.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                duration_secs,
                speaker_count: 0,
            });
        }

        let explicit_min_speech_ms = config
            .min_speech_duration_ms
            .filter(|value| value.is_finite())
            .map(|value| {
                value.clamp(
                    frame_stride_samples as f32 * 1000.0 / TARGET_SAMPLE_RATE as f32,
                    5000.0,
                )
            });
        let explicit_min_silence_ms = config
            .min_silence_duration_ms
            .filter(|value| value.is_finite())
            .map(|value| {
                value.clamp(
                    frame_stride_samples as f32 * 1000.0 / TARGET_SAMPLE_RATE as f32,
                    5000.0,
                )
            });
        let min_speech_ms = explicit_min_speech_ms
            .unwrap_or(DEFAULT_MIN_SPEECH_MS)
            .clamp(
                frame_stride_samples as f32 * 1000.0 / TARGET_SAMPLE_RATE as f32,
                5000.0,
            );
        let min_silence_ms = explicit_min_silence_ms
            .unwrap_or(DEFAULT_MIN_SILENCE_MS)
            .clamp(
                frame_stride_samples as f32 * 1000.0 / TARGET_SAMPLE_RATE as f32,
                5000.0,
            );

        let mut gated_probs = speaker_probs;
        if sortformer_rms_gating_enabled() {
            let frame_ms = frame_stride_samples as f32 * 1000.0 / TARGET_SAMPLE_RATE as f32;
            let min_speech_frames = ((min_speech_ms / frame_ms).round() as usize).max(1);
            let min_silence_frames = ((min_silence_ms / frame_ms).round() as usize).max(1);

            let frame_count = gated_probs.len();
            let mut vad_mask = realtime_voice_vad_frame_mask(
                &samples,
                frame_count,
                frame_stride_samples,
                REALTIME_VAD_THRESHOLD,
            );
            smooth_activity_mask(&mut vad_mask, min_speech_frames, min_silence_frames);

            for (frame_idx, active) in vad_mask.iter().copied().enumerate() {
                if !active {
                    for spk in 0..MAX_SUPPORTED_SPEAKERS {
                        gated_probs[frame_idx][spk] = 0.0;
                    }
                }
            }
        }

        let requested_max = config.max_speakers.unwrap_or(MAX_SUPPORTED_SPEAKERS);
        let max_speakers = requested_max.clamp(1, MAX_SUPPORTED_SPEAKERS);
        let requested_min = config.min_speakers.unwrap_or(1);
        let min_speakers = requested_min.clamp(1, max_speakers);
        let limit_speaker_channels = should_limit_speaker_channels(config);

        let postprocessing_params =
            resolve_postprocessing_params(config, explicit_min_speech_ms, explicit_min_silence_ms);

        let mut raw_segments = Vec::<RawSegment>::new();
        let mut speaker_stats = Vec::<SpeakerActivityStats>::new();
        for speaker_idx in 0..MAX_SUPPORTED_SPEAKERS {
            let speaker_segments =
                ts_vad_post_processing(&gated_probs, speaker_idx, &postprocessing_params);
            if speaker_segments.is_empty() {
                if limit_speaker_channels {
                    speaker_stats.push(SpeakerActivityStats {
                        speaker_idx,
                        total_duration_secs: 0.0,
                        peak_probability: 0.0,
                        segment_count: 0,
                    });
                }
                continue;
            }

            let peak_probability = gated_probs
                .iter()
                .map(|row| row[speaker_idx])
                .fold(0.0f32, f32::max);
            let total_duration_secs = speaker_segments
                .iter()
                .map(|(start_secs, end_secs)| (end_secs - start_secs).max(0.0))
                .sum::<f32>();

            for (start_secs, end_secs) in speaker_segments {
                let start_secs = start_secs.clamp(0.0, duration_secs);
                let end_secs = end_secs.clamp(0.0, duration_secs);
                if end_secs <= start_secs {
                    continue;
                }
                let confidence = average_speaker_probability_for_range(
                    &gated_probs,
                    speaker_idx,
                    start_secs,
                    end_secs,
                    frame_stride_samples,
                );
                raw_segments.push(RawSegment {
                    speaker_idx,
                    start_secs,
                    end_secs,
                    confidence,
                });
            }

            if limit_speaker_channels {
                speaker_stats.push(SpeakerActivityStats {
                    speaker_idx,
                    total_duration_secs,
                    peak_probability,
                    segment_count: raw_segments
                        .iter()
                        .filter(|segment| segment.speaker_idx == speaker_idx)
                        .count(),
                });
            }
        }

        if raw_segments.is_empty() {
            return Ok(DiarizationResult {
                segments: Vec::new(),
                duration_secs,
                speaker_count: 0,
            });
        }

        if limit_speaker_channels {
            let selected_speakers =
                select_speaker_channels(&speaker_stats, min_speakers, max_speakers);
            raw_segments.retain(|segment| selected_speakers.contains(&segment.speaker_idx));
        }

        raw_segments.sort_by(|a, b| {
            a.start_secs
                .total_cmp(&b.start_secs)
                .then(a.speaker_idx.cmp(&b.speaker_idx))
        });

        let mut speaker_first_start = BTreeMap::<usize, f32>::new();
        for segment in &raw_segments {
            speaker_first_start
                .entry(segment.speaker_idx)
                .and_modify(|cur| {
                    if segment.start_secs < *cur {
                        *cur = segment.start_secs;
                    }
                })
                .or_insert(segment.start_secs);
        }

        let mut ordered = speaker_first_start.into_iter().collect::<Vec<_>>();
        ordered.sort_by(|a, b| a.1.total_cmp(&b.1));
        let speaker_remap = ordered
            .iter()
            .enumerate()
            .map(|(i, (speaker_idx, _))| (*speaker_idx, i))
            .collect::<HashMap<_, _>>();

        let speaker_labels = (0..ordered.len())
            .map(|idx| format!("SPEAKER_{idx:02}"))
            .collect::<Vec<_>>();

        let mut segments = raw_segments
            .into_iter()
            .map(|segment| {
                let remapped = speaker_remap
                    .get(&segment.speaker_idx)
                    .copied()
                    .unwrap_or(0);
                DiarizationSegment {
                    speaker: speaker_labels
                        .get(remapped)
                        .cloned()
                        .unwrap_or_else(|| format!("SPEAKER_{remapped:02}")),
                    start_secs: segment.start_secs,
                    end_secs: segment.end_secs,
                    confidence: segment.confidence,
                }
            })
            .collect::<Vec<_>>();

        merge_adjacent_segments(&mut segments, 0.0);
        segments.sort_by(|a, b| {
            a.start_secs
                .total_cmp(&b.start_secs)
                .then(a.speaker.cmp(&b.speaker))
        });

        let speaker_count = segments
            .iter()
            .map(|segment| segment.speaker.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len();

        Ok(DiarizationResult {
            segments,
            duration_secs,
            speaker_count,
        })
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }
}

fn sortformer_model_device(device_profile: &DeviceProfile) -> Device {
    if sortformer_uses_selected_model_device(device_profile.kind) {
        device_profile.device.clone()
    } else {
        Device::Cpu
    }
}

fn sortformer_uses_selected_model_device(kind: DeviceKind) -> bool {
    kind.is_cuda()
}

#[derive(Debug, Clone)]
struct RawSegment {
    speaker_idx: usize,
    start_secs: f32,
    end_secs: f32,
    confidence: Option<f32>,
}

#[derive(Debug, Clone, Copy)]
struct SpeakerActivityStats {
    speaker_idx: usize,
    total_duration_secs: f32,
    peak_probability: f32,
    segment_count: usize,
}

#[derive(Debug, Clone, Copy)]
struct PostProcessingParams {
    onset: f32,
    offset: f32,
    pad_onset: f32,
    pad_offset: f32,
    min_duration_on: f32,
    min_duration_off: f32,
    filter_speech_first: bool,
}

struct SortformerInferenceModel {
    device: Device,
    preprocessor: SortformerPreprocessor,
    encoder: SortformerConformerEncoder,
    encoder_proj: Linear,
    transformer: SortformerTransformerEncoder,
    head: SortformerSpeakerHead,
    streaming: Option<SortformerStreamingConfig>,
}

impl SortformerInferenceModel {
    fn load(
        vb: &VarBuilder,
        preprocessor_cfg: SortformerPreprocessorConfig,
        variant: ModelVariant,
        streaming_mode: bool,
        encoder_cfg: Option<SortformerEncoderConfig>,
        modules_cfg: SortformerModulesConfig,
        device: Device,
    ) -> Result<Self> {
        let preprocessor = SortformerPreprocessor::load(vb, preprocessor_cfg)?;
        let encoder = SortformerConformerEncoder::load(
            vb.pp("encoder"),
            encoder_cfg.and_then(|cfg| cfg.xscaling).unwrap_or(true),
        )?;

        let encoder_proj_w = vb
            .pp("sortformer_modules.encoder_proj")
            .get_unchecked_dtype("weight", DType::F32)?;
        let (proj_out, proj_in) = encoder_proj_w.dims2()?;
        let encoder_proj =
            mlx::load_linear(proj_in, proj_out, vb.pp("sortformer_modules.encoder_proj"))?;

        let transformer = SortformerTransformerEncoder::load(vb.pp("transformer_encoder"))?;
        let head = SortformerSpeakerHead::load(vb.pp("sortformer_modules"))?;
        let streaming = if streaming_mode {
            Some(resolve_streaming_config(
                variant,
                &modules_cfg,
                encoder.d_model(),
            )?)
        } else {
            None
        };

        Ok(Self {
            device,
            preprocessor,
            encoder,
            encoder_proj,
            transformer,
            head,
            streaming,
        })
    }

    fn infer_speaker_probabilities(
        &self,
        samples: &[f32],
    ) -> Result<(Vec<[f32; MAX_SUPPORTED_SPEAKERS]>, usize)> {
        let normalized_storage;
        let feature_input = if self.streaming.is_some() {
            samples
        } else {
            normalized_storage = {
                let mut normalized = samples.to_vec();
                let max_abs = normalized
                    .iter()
                    .copied()
                    .map(f32::abs)
                    .fold(0.0f32, f32::max)
                    .max(1e-6);
                for sample in &mut normalized {
                    *sample /= max_abs;
                }
                normalized
            };
            &normalized_storage
        };

        let (features, feature_frames) = self.preprocessor.compute_features(feature_input)?;
        if feature_frames == 0 {
            return Ok((Vec::new(), self.encoder.frame_stride_samples()));
        }
        let features = features
            .narrow(2, 0, feature_frames)?
            .to_device(&self.device)?;

        let out = if let Some(streaming_cfg) = self.streaming {
            self.infer_speaker_probabilities_streaming(&features, feature_frames, streaming_cfg)?
        } else {
            self.infer_speaker_probabilities_offline(&features, feature_frames)?
        };

        Ok((out, self.encoder.frame_stride_samples()))
    }

    fn infer_speaker_probabilities_offline(
        &self,
        features: &Tensor,
        feature_frames: usize,
    ) -> Result<Vec<[f32; MAX_SUPPORTED_SPEAKERS]>> {
        let (encoded, encoded_len) = self.encoder.forward(features, feature_frames)?;
        if encoded_len == 0 {
            return Ok(Vec::new());
        }
        let probs = self.forward_probabilities(&encoded, encoded_len)?;
        tensor_to_probability_rows(&probs, encoded_len)
    }

    fn infer_speaker_probabilities_streaming(
        &self,
        features: &Tensor,
        feature_frames: usize,
        cfg: SortformerStreamingConfig,
    ) -> Result<Vec<[f32; MAX_SUPPORTED_SPEAKERS]>> {
        let mut state = SortformerStreamingState::new(cfg.fc_d_model);
        let mut total_preds = Vec::new();
        for plan in plan_streaming_feature_chunks(feature_frames, cfg) {
            let chunk = features
                .i((.., .., plan.feature_start..plan.feature_end))?
                .transpose(1, 2)?
                .contiguous()?;
            let (chunk_pre_encoded, chunk_pre_encoded_len) =
                self.encoder.pre_encode(&chunk, chunk.dim(1)?)?;
            if chunk_pre_encoded_len == 0 {
                continue;
            }

            let chunk_rows = tensor_to_embedding_rows(&chunk_pre_encoded, chunk_pre_encoded_len)?;
            let mut composite_rows =
                Vec::with_capacity(state.spkcache.len() + state.fifo.len() + chunk_rows.len());
            composite_rows.extend(state.spkcache.iter().cloned());
            composite_rows.extend(state.fifo.iter().cloned());
            composite_rows.extend(chunk_rows.iter().cloned());
            if composite_rows.is_empty() {
                continue;
            }

            let composite = tensor_from_embedding_rows(
                &composite_rows,
                cfg.fc_d_model,
                chunk_pre_encoded.device(),
            )?;
            let (encoded, encoded_len) = self.encoder.forward_pre_encoded(
                &composite,
                state.spkcache.len() + state.fifo.len() + chunk_rows.len(),
            )?;
            let probs = self.forward_probabilities(&encoded, encoded_len)?;
            let pred_rows = tensor_to_probability_rows(&probs, encoded_len)?;
            let (updated_state, chunk_preds) = update_streaming_state(
                state,
                &chunk_rows,
                &pred_rows,
                pre_encoded_left_offset(plan.left_offset, cfg.subsampling_factor),
                pre_encoded_right_offset(plan.right_offset, cfg.subsampling_factor),
                cfg,
            )?;
            state = updated_state;
            total_preds.extend(chunk_preds);
        }

        Ok(total_preds)
    }

    fn forward_probabilities(&self, encoded: &Tensor, encoded_len: usize) -> Result<Tensor> {
        let mut x = encoded.i((.., ..encoded_len, ..))?;
        x = x.apply(&self.encoder_proj)?;
        x = self.transformer.forward(&x)?;
        let probs = self.head.forward(&x)?;
        let (_, _, speaker_dim) = probs.dims3()?;
        if speaker_dim != MAX_SUPPORTED_SPEAKERS {
            return Err(Error::InferenceError(format!(
                "Unexpected Sortformer speaker dimension {}; expected {}",
                speaker_dim, MAX_SUPPORTED_SPEAKERS
            )));
        }
        Ok(probs)
    }
}

fn resolve_streaming_config(
    variant: ModelVariant,
    modules_cfg: &SortformerModulesConfig,
    encoder_d_model: usize,
) -> Result<SortformerStreamingConfig> {
    let mut cfg = SortformerStreamingConfig {
        fc_d_model: modules_cfg.fc_d_model.unwrap_or(encoder_d_model),
        subsampling_factor: modules_cfg
            .subsampling_factor
            .unwrap_or(TS_VAD_UNIT_FRAME_COUNT),
        spkcache_len: modules_cfg.spkcache_len.unwrap_or(188),
        fifo_len: modules_cfg.fifo_len.unwrap_or(0),
        chunk_len: modules_cfg.chunk_len.unwrap_or(188),
        spkcache_update_period: modules_cfg.spkcache_update_period.unwrap_or(188),
        chunk_left_context: modules_cfg.chunk_left_context.unwrap_or(1),
        chunk_right_context: modules_cfg.chunk_right_context.unwrap_or(1),
        spkcache_sil_frames_per_spk: modules_cfg.spkcache_sil_frames_per_spk.unwrap_or(3),
        pred_score_threshold: modules_cfg
            .pred_score_threshold
            .unwrap_or(0.25)
            .clamp(1e-4, 0.99),
        scores_boost_latest: modules_cfg.scores_boost_latest.unwrap_or(0.05).max(0.0),
        sil_threshold: modules_cfg.sil_threshold.unwrap_or(0.2).max(0.0),
        strong_boost_rate: modules_cfg.strong_boost_rate.unwrap_or(0.75).max(0.0),
        weak_boost_rate: modules_cfg.weak_boost_rate.unwrap_or(1.5).max(0.0),
        min_pos_scores_rate: modules_cfg
            .min_pos_scores_rate
            .unwrap_or(0.5)
            .clamp(0.0, 1.0),
    };

    match resolve_streaming_profile(variant) {
        SortformerStreamingProfile::Model => {}
        SortformerStreamingProfile::LowLatency => {
            cfg.chunk_len = 6;
            cfg.chunk_right_context = 7;
            cfg.fifo_len = 188;
            cfg.spkcache_update_period = 144;
            cfg.spkcache_len = 188;
        }
        SortformerStreamingProfile::HighLatency => {
            cfg.chunk_len = 340;
            cfg.chunk_right_context = 40;
            cfg.fifo_len = 40;
            cfg.spkcache_update_period = 300;
            cfg.spkcache_len = 188;
        }
    }

    cfg.validate()
}

fn resolve_streaming_profile(variant: ModelVariant) -> SortformerStreamingProfile {
    if let Ok(value) = std::env::var("IZWI_SORTFORMER_STREAMING_PROFILE") {
        match value.trim().to_ascii_lowercase().as_str() {
            "model" => return SortformerStreamingProfile::Model,
            "low_latency" | "low-latency" | "low" => return SortformerStreamingProfile::LowLatency,
            "high_latency" | "high-latency" | "high" => {
                return SortformerStreamingProfile::HighLatency
            }
            _ => {}
        }
    }

    let _ = variant;
    SortformerStreamingProfile::Model
}

fn plan_streaming_feature_chunks(
    feature_frames: usize,
    cfg: SortformerStreamingConfig,
) -> Vec<SortformerStreamingChunkPlan> {
    if feature_frames == 0 {
        return Vec::new();
    }

    let context_frames = cfg.subsampling_factor;
    let chunk_width = cfg.chunk_len * context_frames;
    let left_context = cfg.chunk_left_context * context_frames;
    let right_context = cfg.chunk_right_context * context_frames;

    let mut plans = Vec::new();
    let mut start = 0usize;
    while start < feature_frames {
        let left_offset = left_context.min(start);
        let end = (start + chunk_width).min(feature_frames);
        let right_offset = right_context.min(feature_frames.saturating_sub(end));
        plans.push(SortformerStreamingChunkPlan {
            feature_start: start - left_offset,
            feature_end: end + right_offset,
            left_offset,
            right_offset,
        });
        start = end;
    }
    plans
}

fn pre_encoded_left_offset(left_offset: usize, subsampling_factor: usize) -> usize {
    ((left_offset as f32) / (subsampling_factor as f32)).round() as usize
}

fn pre_encoded_right_offset(right_offset: usize, subsampling_factor: usize) -> usize {
    ((right_offset as f32) / (subsampling_factor as f32)).ceil() as usize
}

fn tensor_to_embedding_rows(tensor: &Tensor, row_count: usize) -> Result<Vec<Vec<f32>>> {
    if row_count == 0 {
        return Ok(Vec::new());
    }

    let view = tensor.i((0, ..row_count, ..))?;
    let (_, emb_dim) = view.dims2()?;
    let values = view.flatten_all()?.to_vec1::<f32>()?;
    Ok(values
        .chunks(emb_dim)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>())
}

fn tensor_to_probability_rows(
    tensor: &Tensor,
    row_count: usize,
) -> Result<Vec<[f32; MAX_SUPPORTED_SPEAKERS]>> {
    if row_count == 0 {
        return Ok(Vec::new());
    }

    let view = tensor.i((0, ..row_count, ..))?;
    let (_, speaker_dim) = view.dims2()?;
    if speaker_dim != MAX_SUPPORTED_SPEAKERS {
        return Err(Error::InferenceError(format!(
            "Unexpected Sortformer probability tensor width {}; expected {}",
            speaker_dim, MAX_SUPPORTED_SPEAKERS
        )));
    }
    let values = view.flatten_all()?.to_vec1::<f32>()?;
    Ok(values
        .chunks(MAX_SUPPORTED_SPEAKERS)
        .map(|chunk| [chunk[0], chunk[1], chunk[2], chunk[3]])
        .collect::<Vec<_>>())
}

fn tensor_from_embedding_rows(
    rows: &[Vec<f32>],
    emb_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    if rows.is_empty() {
        return Tensor::zeros((1, 0, emb_dim), DType::F32, device).map_err(Error::from);
    }
    let mut flat = Vec::with_capacity(rows.len() * emb_dim);
    for row in rows {
        if row.len() != emb_dim {
            return Err(Error::InferenceError(format!(
                "Inconsistent Sortformer embedding row size {}; expected {}",
                row.len(),
                emb_dim
            )));
        }
        flat.extend_from_slice(row);
    }
    Tensor::from_vec(flat, (1, rows.len(), emb_dim), device).map_err(Error::from)
}

fn update_streaming_state(
    mut state: SortformerStreamingState,
    chunk_rows: &[Vec<f32>],
    preds: &[[f32; MAX_SUPPORTED_SPEAKERS]],
    lc: usize,
    rc: usize,
    cfg: SortformerStreamingConfig,
) -> Result<(SortformerStreamingState, Vec<[f32; MAX_SUPPORTED_SPEAKERS]>)> {
    let spkcache_len = state.spkcache.len();
    let fifo_len = state.fifo.len();
    if preds.len() < spkcache_len + fifo_len + chunk_rows.len() {
        return Err(Error::InferenceError(format!(
            "Streaming Sortformer prediction rows {} do not cover spkcache ({spkcache_len}) + fifo ({fifo_len}) + chunk ({})",
            preds.len(),
            chunk_rows.len()
        )));
    }

    state.fifo_preds = preds[spkcache_len..spkcache_len + fifo_len].to_vec();

    let chunk_valid_len = chunk_rows.len().saturating_sub(lc + rc);
    let chunk_start = lc.min(chunk_rows.len());
    let chunk_end = (chunk_start + chunk_valid_len).min(chunk_rows.len());
    let chunk_payload = chunk_rows[chunk_start..chunk_end].to_vec();
    let chunk_preds =
        preds[spkcache_len + fifo_len + chunk_start..spkcache_len + fifo_len + chunk_end].to_vec();

    state.fifo.extend(chunk_payload.clone());
    state.fifo_preds.extend(chunk_preds.clone());

    if fifo_len + chunk_payload.len() > cfg.fifo_len {
        let pop_out_len = cfg
            .spkcache_update_period
            .max(
                chunk_payload
                    .len()
                    .saturating_sub(cfg.fifo_len)
                    .saturating_add(fifo_len),
            )
            .min(fifo_len + chunk_payload.len());
        let pop_out_embs = state.fifo[..pop_out_len].to_vec();
        let pop_out_preds = state.fifo_preds[..pop_out_len].to_vec();

        update_silence_profile(&mut state, &pop_out_embs, &pop_out_preds, cfg.sil_threshold);
        state.fifo.drain(..pop_out_len);
        state.fifo_preds.drain(..pop_out_len);

        let prev_spkcache_len = state.spkcache.len();
        state.spkcache.extend(pop_out_embs);
        if let Some(spkcache_preds) = state.spkcache_preds.as_mut() {
            spkcache_preds.extend(pop_out_preds);
        } else if state.spkcache.len() > cfg.spkcache_len {
            let mut seeded_preds = preds[..prev_spkcache_len].to_vec();
            seeded_preds.extend(pop_out_preds);
            state.spkcache_preds = Some(seeded_preds);
        }

        if state.spkcache.len() > cfg.spkcache_len {
            let spkcache_preds = state.spkcache_preds.as_ref().ok_or_else(|| {
                Error::InferenceError(
                    "Sortformer speaker cache predictions were not initialized".to_string(),
                )
            })?;
            let (compressed_cache, compressed_preds) =
                compress_spkcache(&state.spkcache, spkcache_preds, &state.mean_sil_emb, cfg)?;
            state.spkcache = compressed_cache;
            state.spkcache_preds = Some(compressed_preds);
        }
    }

    Ok((state, chunk_preds))
}

fn update_silence_profile(
    state: &mut SortformerStreamingState,
    emb_seq: &[Vec<f32>],
    preds: &[[f32; MAX_SUPPORTED_SPEAKERS]],
    sil_threshold: f32,
) {
    for (emb, pred) in emb_seq.iter().zip(preds.iter()) {
        let is_silence = pred.iter().copied().sum::<f32>() < sil_threshold;
        if !is_silence {
            continue;
        }
        let total = state.n_sil_frames as f32;
        for (idx, value) in emb.iter().copied().enumerate() {
            state.mean_sil_emb[idx] = if state.n_sil_frames == 0 {
                value
            } else {
                (state.mean_sil_emb[idx] * total + value) / (total + 1.0)
            };
        }
        state.n_sil_frames += 1;
    }
}

fn compress_spkcache(
    emb_seq: &[Vec<f32>],
    preds: &[[f32; MAX_SUPPORTED_SPEAKERS]],
    mean_sil_emb: &[f32],
    cfg: SortformerStreamingConfig,
) -> Result<(Vec<Vec<f32>>, Vec<[f32; MAX_SUPPORTED_SPEAKERS]>)> {
    if emb_seq.len() != preds.len() {
        return Err(Error::InferenceError(format!(
            "Sortformer speaker cache compression length mismatch: {} embeddings vs {} prediction rows",
            emb_seq.len(),
            preds.len()
        )));
    }

    let spkcache_len_per_spk =
        cfg.spkcache_len / MAX_SUPPORTED_SPEAKERS - cfg.spkcache_sil_frames_per_spk;
    let strong_boost_per_spk =
        ((spkcache_len_per_spk as f32) * cfg.strong_boost_rate).floor() as usize;
    let weak_boost_per_spk = ((spkcache_len_per_spk as f32) * cfg.weak_boost_rate).floor() as usize;
    let min_pos_scores_per_spk =
        ((spkcache_len_per_spk as f32) * cfg.min_pos_scores_rate).floor() as usize;

    let mut scores = get_log_pred_scores(preds, cfg.pred_score_threshold);
    disable_low_scores(preds, &mut scores, min_pos_scores_per_spk);

    if cfg.scores_boost_latest > 0.0 && emb_seq.len() > cfg.spkcache_len {
        for row in scores.iter_mut().skip(cfg.spkcache_len) {
            for score in row.iter_mut().filter(|score| score.is_finite()) {
                *score += cfg.scores_boost_latest;
            }
        }
    }

    boost_topk_scores(&mut scores, strong_boost_per_spk, 2.0);
    boost_topk_scores(&mut scores, weak_boost_per_spk, 1.0);

    let speaker_frame_span = emb_seq.len() + cfg.spkcache_sil_frames_per_spk;
    let mut candidates = Vec::with_capacity(speaker_frame_span * MAX_SUPPORTED_SPEAKERS);
    for speaker_idx in 0..MAX_SUPPORTED_SPEAKERS {
        let base = speaker_idx * speaker_frame_span;
        for (frame_idx, frame_scores) in scores.iter().enumerate() {
            candidates.push(SortformerCacheCandidate {
                flat_index: base + frame_idx,
                frame_index: Some(frame_idx),
                score: frame_scores[speaker_idx],
            });
        }
        for silence_idx in 0..cfg.spkcache_sil_frames_per_spk {
            candidates.push(SortformerCacheCandidate {
                flat_index: base + emb_seq.len() + silence_idx,
                frame_index: None,
                score: f32::INFINITY,
            });
        }
    }

    candidates.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.flat_index.cmp(&b.flat_index))
    });
    let mut selected = candidates
        .into_iter()
        .take(cfg.spkcache_len)
        .collect::<Vec<_>>();
    selected.sort_by(|a, b| a.flat_index.cmp(&b.flat_index));

    let mut spkcache = Vec::with_capacity(cfg.spkcache_len);
    let mut spkcache_preds = Vec::with_capacity(cfg.spkcache_len);
    for candidate in selected {
        if candidate.score.is_finite() {
            if let Some(frame_idx) = candidate.frame_index {
                spkcache.push(emb_seq[frame_idx].clone());
                spkcache_preds.push(preds[frame_idx]);
            } else {
                spkcache.push(mean_sil_emb.to_vec());
                spkcache_preds.push([0.0; MAX_SUPPORTED_SPEAKERS]);
            }
        } else {
            spkcache.push(mean_sil_emb.to_vec());
            spkcache_preds.push([0.0; MAX_SUPPORTED_SPEAKERS]);
        }
    }

    Ok((spkcache, spkcache_preds))
}

fn get_log_pred_scores(
    preds: &[[f32; MAX_SUPPORTED_SPEAKERS]],
    pred_score_threshold: f32,
) -> Vec<[f32; MAX_SUPPORTED_SPEAKERS]> {
    preds
        .iter()
        .map(|frame| {
            let log_one_minus =
                frame.map(|prob| (1.0 - prob).clamp(pred_score_threshold, 1.0).ln());
            let log_one_minus_sum = log_one_minus.iter().copied().sum::<f32>();
            let mut scores = [0.0; MAX_SUPPORTED_SPEAKERS];
            for speaker_idx in 0..MAX_SUPPORTED_SPEAKERS {
                let log_prob = frame[speaker_idx].clamp(pred_score_threshold, 1.0).ln();
                scores[speaker_idx] =
                    log_prob - log_one_minus[speaker_idx] + log_one_minus_sum - 0.5f32.ln();
            }
            scores
        })
        .collect()
}

fn disable_low_scores(
    preds: &[[f32; MAX_SUPPORTED_SPEAKERS]],
    scores: &mut [[f32; MAX_SUPPORTED_SPEAKERS]],
    min_pos_scores_per_spk: usize,
) {
    let mut positive_counts = [0usize; MAX_SUPPORTED_SPEAKERS];
    for (pred_row, score_row) in preds.iter().zip(scores.iter_mut()) {
        for speaker_idx in 0..MAX_SUPPORTED_SPEAKERS {
            if pred_row[speaker_idx] <= 0.5 {
                score_row[speaker_idx] = f32::NEG_INFINITY;
            } else if score_row[speaker_idx] > 0.0 {
                positive_counts[speaker_idx] += 1;
            }
        }
    }

    for (pred_row, score_row) in preds.iter().zip(scores.iter_mut()) {
        for speaker_idx in 0..MAX_SUPPORTED_SPEAKERS {
            if pred_row[speaker_idx] > 0.5
                && score_row[speaker_idx].is_finite()
                && score_row[speaker_idx] <= 0.0
                && positive_counts[speaker_idx] >= min_pos_scores_per_spk
            {
                score_row[speaker_idx] = f32::NEG_INFINITY;
            }
        }
    }
}

fn boost_topk_scores(
    scores: &mut [[f32; MAX_SUPPORTED_SPEAKERS]],
    n_boost_per_spk: usize,
    scale_factor: f32,
) {
    if n_boost_per_spk == 0 {
        return;
    }

    for speaker_idx in 0..MAX_SUPPORTED_SPEAKERS {
        let mut ranked = scores
            .iter()
            .enumerate()
            .filter_map(|(frame_idx, frame_scores)| {
                frame_scores[speaker_idx]
                    .is_finite()
                    .then_some((frame_scores[speaker_idx], frame_idx))
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        for (_, frame_idx) in ranked.into_iter().take(n_boost_per_spk) {
            scores[frame_idx][speaker_idx] += scale_factor * SORTFORMER_SCORE_BOOST_DELTA;
        }
    }
}

struct SortformerPreprocessor {
    sample_rate: usize,
    n_fft: usize,
    win_length: usize,
    hop_length: usize,
    _window: Vec<f32>,
    padded_window: Vec<f32>,
    fb: Vec<f32>,
    n_mels: usize,
    n_freqs: usize,
    normalize: SortformerFeatureNormalize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SortformerFeatureNormalize {
    None,
    PerFeature,
    AllFeatures,
}

impl SortformerPreprocessor {
    fn load(vb: &VarBuilder, cfg: SortformerPreprocessorConfig) -> Result<Self> {
        let sample_rate = cfg.sample_rate.unwrap_or(TARGET_SAMPLE_RATE) as usize;
        let n_fft = cfg.n_fft.unwrap_or(512);
        let win_length =
            ((cfg.window_size.unwrap_or(0.025) * sample_rate as f32).round() as usize).max(1);
        let hop_length =
            ((cfg.window_stride.unwrap_or(0.01) * sample_rate as f32).round() as usize).max(1);
        let n_mels = cfg.features.unwrap_or(128);
        let normalize = match cfg
            .normalize
            .as_deref()
            .map(|value| value.trim().to_ascii_lowercase())
        {
            Some(value) if value == "per_feature" => SortformerFeatureNormalize::PerFeature,
            Some(value) if value == "all_features" => SortformerFeatureNormalize::AllFeatures,
            _ => SortformerFeatureNormalize::None,
        };

        let preproc_vb = vb.pp("preprocessor.featurizer");
        let window = match preproc_vb.get_unchecked_dtype("window", DType::F32) {
            Ok(window_tensor) => window_tensor.to_vec1::<f32>()?,
            Err(_) => hann_window(win_length),
        };

        let (fb, loaded_mels, loaded_freqs) = match preproc_vb.get_unchecked_dtype("fb", DType::F32)
        {
            Ok(fb_tensor) => {
                let (_, mels, freqs) = fb_tensor.dims3()?;
                let fb = fb_tensor.squeeze(0)?.flatten_all()?.to_vec1::<f32>()?;
                (fb, mels, freqs)
            }
            Err(_) => {
                let generated =
                    mel_filterbank(sample_rate, n_fft, n_mels, 0.0, sample_rate as f32 / 2.0);
                (generated, n_mels, n_fft / 2 + 1)
            }
        };

        let n_freqs = n_fft / 2 + 1;
        if loaded_freqs != n_freqs {
            return Err(Error::ModelLoadError(format!(
                "Unexpected Sortformer filterbank bins: expected {}, got {}",
                n_freqs, loaded_freqs
            )));
        }

        let mut padded_window = vec![0.0f32; n_fft];
        let src_len = window.len().min(n_fft);
        let offset = (n_fft - src_len) / 2;
        padded_window[offset..offset + src_len].copy_from_slice(&window[..src_len]);

        Ok(Self {
            sample_rate,
            n_fft,
            win_length,
            hop_length,
            _window: window,
            padded_window,
            fb,
            n_mels: loaded_mels,
            n_freqs,
            normalize,
        })
    }

    fn compute_features(&self, audio: &[f32]) -> Result<(Tensor, usize)> {
        if audio.is_empty() {
            return Ok((
                Tensor::zeros((1, self.n_mels, 1), DType::F32, &Device::Cpu)?,
                0,
            ));
        }

        let mut x = audio.to_vec();
        preemphasis(&mut x, PREEMPH);

        let center_pad = self.n_fft / 2;
        let mut padded = Vec::with_capacity(x.len() + center_pad * 2);
        padded.extend(std::iter::repeat(0.0).take(center_pad));
        padded.extend_from_slice(&x);
        padded.extend(std::iter::repeat(0.0).take(center_pad));

        let frame_count = if padded.len() >= self.n_fft {
            (padded.len() - self.n_fft) / self.hop_length + 1
        } else {
            1
        };

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.n_fft);

        let mut spectrum = vec![0f32; frame_count * self.n_freqs];
        let mut buffer = vec![Complex::<f32>::new(0.0, 0.0); self.n_fft];
        for frame_idx in 0..frame_count {
            let start = frame_idx * self.hop_length;
            let slice = &padded[start..start + self.n_fft];
            for i in 0..self.n_fft {
                buffer[i].re = slice[i] * self.padded_window[i];
                buffer[i].im = 0.0;
            }
            fft.process(&mut buffer);
            for k in 0..self.n_freqs {
                let mag = (buffer[k].re * buffer[k].re + buffer[k].im * buffer[k].im).sqrt();
                spectrum[frame_idx * self.n_freqs + k] = mag * mag;
            }
        }

        let mut mel = vec![0f32; self.n_mels * frame_count];
        for m in 0..self.n_mels {
            for t in 0..frame_count {
                let mut acc = 0f32;
                let spec_row = &spectrum[t * self.n_freqs..(t + 1) * self.n_freqs];
                let fb_row = &self.fb[m * self.n_freqs..(m + 1) * self.n_freqs];
                for f in 0..self.n_freqs {
                    acc += spec_row[f] * fb_row[f];
                }
                mel[m * frame_count + t] = (acc + LOG_GUARD).ln();
            }
        }

        let valid_frames = audio.len() / self.hop_length;
        let normalized_frames = valid_frames.min(frame_count);
        match self.normalize {
            SortformerFeatureNormalize::None => {}
            SortformerFeatureNormalize::PerFeature => {
                normalize_per_feature(&mut mel, self.n_mels, frame_count, normalized_frames)
            }
            SortformerFeatureNormalize::AllFeatures => {
                normalize_all_features(&mut mel, self.n_mels, frame_count, normalized_frames)
            }
        }

        if valid_frames < frame_count {
            for m in 0..self.n_mels {
                for t in valid_frames..frame_count {
                    mel[m * frame_count + t] = 0.0;
                }
            }
        }

        let features = Tensor::from_vec(mel, (1, self.n_mels, frame_count), &Device::Cpu)?;
        Ok((features, valid_frames.min(frame_count)))
    }
}

struct SortformerConformerEncoder {
    pre_encode: ConvSubsamplingDw,
    layers: Vec<ConformerLayer>,
    d_model: usize,
    input_scale: f64,
    frame_stride_samples: usize,
}

impl SortformerConformerEncoder {
    fn load(vb: VarBuilder, xscaling: bool) -> Result<Self> {
        let pre_encode = ConvSubsamplingDw::load(vb.pp("pre_encode"))?;

        let mut layers = Vec::new();
        let mut idx = 0usize;
        loop {
            let layer_vb = vb.pp(format!("layers.{idx}"));
            if !layer_vb.contains_tensor("norm_out.weight") {
                break;
            }
            layers.push(ConformerLayer::load(layer_vb)?);
            idx += 1;
        }
        if layers.is_empty() {
            return Err(Error::ModelLoadError(
                "Sortformer Conformer encoder has no layers".to_string(),
            ));
        }

        let d_model = layers[0].d_model();
        Ok(Self {
            pre_encode,
            layers,
            d_model,
            input_scale: if xscaling {
                (d_model as f64).sqrt()
            } else {
                1.0
            },
            frame_stride_samples: 160 * 8,
        })
    }

    fn forward(&self, features: &Tensor, feature_frames: usize) -> Result<(Tensor, usize)> {
        let features_t = features.transpose(1, 2)?;
        let (x, encoded_len) = self.pre_encode(&features_t, feature_frames)?;
        self.forward_pre_encoded(&x, encoded_len)
    }

    fn pre_encode(&self, features_t: &Tensor, feature_frames: usize) -> Result<(Tensor, usize)> {
        self.pre_encode.forward(features_t, feature_frames)
    }

    fn forward_pre_encoded(
        &self,
        pre_encoded: &Tensor,
        encoded_len: usize,
    ) -> Result<(Tensor, usize)> {
        let mut x = if self.input_scale != 1.0 {
            pre_encoded.affine(self.input_scale, 0.0)?
        } else {
            pre_encoded.clone()
        };
        let pos_len = x.dim(1)?;
        let pos_emb = build_rel_positional_embedding(pos_len, self.d_model, x.device())?;
        for layer in &self.layers {
            x = layer.forward(&x, &pos_emb)?;
        }
        Ok((x, encoded_len))
    }

    fn frame_stride_samples(&self) -> usize {
        self.frame_stride_samples
    }

    fn d_model(&self) -> usize {
        self.d_model
    }
}

struct ConvSubsamplingDw {
    conv0: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv5: Conv2d,
    conv6: Conv2d,
    out: Linear,
}

impl ConvSubsamplingDw {
    fn load(vb: VarBuilder) -> Result<Self> {
        let conv0_w = vb.pp("conv.0").get_unchecked_dtype("weight", DType::F32)?;
        let (out_channels, _, _, _) = conv0_w.dims4()?;

        let stride_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let point_cfg = Conv2dConfig {
            stride: 1,
            padding: 0,
            ..Default::default()
        };

        let conv0 = mlx::load_conv2d(1, out_channels, 3, stride_cfg, vb.pp("conv.0"))?;

        let mut dw_stride_cfg = stride_cfg;
        dw_stride_cfg.groups = out_channels;
        let conv2 = mlx::load_conv2d(1, out_channels, 3, dw_stride_cfg, vb.pp("conv.2"))?;
        let conv3 = mlx::load_conv2d(out_channels, out_channels, 1, point_cfg, vb.pp("conv.3"))?;
        let conv5 = mlx::load_conv2d(1, out_channels, 3, dw_stride_cfg, vb.pp("conv.5"))?;
        let conv6 = mlx::load_conv2d(out_channels, out_channels, 1, point_cfg, vb.pp("conv.6"))?;

        let out_w = vb.pp("out").get_unchecked_dtype("weight", DType::F32)?;
        let (out_dim, in_dim) = out_w.dims2()?;
        let out = mlx::load_linear(in_dim, out_dim, vb.pp("out"))?;

        Ok(Self {
            conv0,
            conv2,
            conv3,
            conv5,
            conv6,
            out,
        })
    }

    fn forward(&self, features_t: &Tensor, feature_frames: usize) -> Result<(Tensor, usize)> {
        let mut x = features_t.unsqueeze(1)?; // [B,1,T,F]

        x = self.conv0.forward(&x)?;
        x = x.relu()?;

        x = self.conv2.forward(&x)?;
        x = self.conv3.forward(&x)?;
        x = x.relu()?;

        x = self.conv5.forward(&x)?;
        x = self.conv6.forward(&x)?;
        x = x.relu()?;

        let (b, c, t, f) = x.dims4()?;
        let x = x
            .transpose(1, 2)?
            .reshape((b, t, c * f))?
            .apply(&self.out)?;
        let encoded_len = subsampled_len_3x(feature_frames).min(t);
        Ok((x, encoded_len))
    }
}

fn subsampled_len_3x(mut len: usize) -> usize {
    for _ in 0..3 {
        len = len.div_ceil(2);
    }
    len
}

struct ConformerLayer {
    norm_ff1: LayerNorm,
    ff1: FeedForward,
    norm_self_att: LayerNorm,
    self_attn: RelPosSelfAttention,
    norm_conv: LayerNorm,
    conv: ConformerConv,
    norm_ff2: LayerNorm,
    ff2: FeedForward,
    norm_out: LayerNorm,
    d_model: usize,
}

impl ConformerLayer {
    fn load(vb: VarBuilder) -> Result<Self> {
        let d_model = vb
            .pp("norm_out")
            .get_unchecked_dtype("weight", DType::F32)?
            .dim(0)?;

        let ff_dim = vb
            .pp("feed_forward1.linear1")
            .get_unchecked_dtype("weight", DType::F32)?
            .dims2()?
            .0;

        let norm_ff1 = layer_norm(d_model, 1e-5, vb.pp("norm_feed_forward1"))?;
        let ff1 = FeedForward::load(vb.pp("feed_forward1"), d_model, ff_dim)?;

        let norm_self_att = layer_norm(d_model, 1e-5, vb.pp("norm_self_att"))?;
        let self_attn = RelPosSelfAttention::load(vb.pp("self_attn"), d_model)?;

        let norm_conv = layer_norm(d_model, 1e-5, vb.pp("norm_conv"))?;
        let conv = ConformerConv::load(vb.pp("conv"), d_model)?;

        let norm_ff2 = layer_norm(d_model, 1e-5, vb.pp("norm_feed_forward2"))?;
        let ff2 = FeedForward::load(vb.pp("feed_forward2"), d_model, ff_dim)?;

        let norm_out = layer_norm(d_model, 1e-5, vb.pp("norm_out"))?;

        Ok(Self {
            norm_ff1,
            ff1,
            norm_self_att,
            self_attn,
            norm_conv,
            conv,
            norm_ff2,
            ff2,
            norm_out,
            d_model,
        })
    }

    fn d_model(&self) -> usize {
        self.d_model
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let mut residual = x.clone();

        let ff1 = self.ff1.forward(&self.norm_ff1.forward(&residual)?)?;
        residual = residual.broadcast_add(&ff1.affine(0.5, 0.0)?)?;

        let attn = self
            .self_attn
            .forward(&self.norm_self_att.forward(&residual)?, pos_emb)?;
        residual = residual.broadcast_add(&attn)?;

        let conv = self.conv.forward(&self.norm_conv.forward(&residual)?)?;
        residual = residual.broadcast_add(&conv)?;

        let ff2 = self.ff2.forward(&self.norm_ff2.forward(&residual)?)?;
        residual = residual.broadcast_add(&ff2.affine(0.5, 0.0)?)?;

        self.norm_out
            .forward(&residual)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl FeedForward {
    fn load(vb: VarBuilder, d_model: usize, ff_dim: usize) -> Result<Self> {
        let linear1 = mlx::load_linear(d_model, ff_dim, vb.pp("linear1"))?;
        let linear2 = mlx::load_linear(ff_dim, d_model, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear1.forward(x)?;
        let x = swish(&x)?;
        self.linear2
            .forward(&x)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

struct ConformerConv {
    pointwise_conv1: Conv1d,
    depthwise_conv: Conv1d,
    batch_norm: candle_nn::BatchNorm,
    pointwise_conv2: Conv1d,
    d_model: usize,
}

impl ConformerConv {
    fn load(vb: VarBuilder, d_model: usize) -> Result<Self> {
        let kernel_size = vb
            .pp("depthwise_conv")
            .get_unchecked_dtype("weight", DType::F32)?
            .dims3()?
            .2;

        let pointwise_conv1 = mlx::load_conv1d(
            d_model,
            d_model * 2,
            1,
            Conv1dConfig::default(),
            vb.pp("pointwise_conv1"),
        )?;

        let depthwise_conv = mlx::load_conv1d(
            d_model,
            d_model,
            kernel_size,
            Conv1dConfig {
                padding: (kernel_size - 1) / 2,
                groups: d_model,
                ..Default::default()
            },
            vb.pp("depthwise_conv"),
        )?;

        let batch_norm = batch_norm(d_model, 1e-5, vb.pp("batch_norm"))?;

        let pointwise_conv2 = mlx::load_conv1d(
            d_model,
            d_model,
            1,
            Conv1dConfig::default(),
            vb.pp("pointwise_conv2"),
        )?;

        Ok(Self {
            pointwise_conv1,
            depthwise_conv,
            batch_norm,
            pointwise_conv2,
            d_model,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.transpose(1, 2)?;

        x = self.pointwise_conv1.forward(&x)?;
        let x_a = x.i((.., ..self.d_model, ..))?;
        let x_b = x.i((.., self.d_model.., ..))?;
        x = x_a.broadcast_mul(&ops::sigmoid(&x_b)?)?;

        x = self.depthwise_conv.forward(&x)?;
        x = self.batch_norm.forward_t(&x, false)?;
        x = swish(&x)?;
        x = self.pointwise_conv2.forward(&x)?;

        x.transpose(1, 2).map_err(Error::from)
    }
}

struct RelPosSelfAttention {
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    linear_pos: Linear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
    num_heads: usize,
    head_dim: usize,
    d_model: usize,
}

impl RelPosSelfAttention {
    fn load(vb: VarBuilder, d_model: usize) -> Result<Self> {
        let pos_bias_u = vb.get_unchecked_dtype("pos_bias_u", DType::F32)?;
        let (num_heads, head_dim) = pos_bias_u.dims2()?;
        let pos_bias_v = vb.get((num_heads, head_dim), "pos_bias_v")?;

        if num_heads * head_dim != d_model {
            return Err(Error::ModelLoadError(format!(
                "Sortformer attention head dims mismatch: heads={num_heads}, head_dim={head_dim}, d_model={d_model}"
            )));
        }

        let linear_q = mlx::load_linear(d_model, d_model, vb.pp("linear_q"))?;
        let linear_k = mlx::load_linear(d_model, d_model, vb.pp("linear_k"))?;
        let linear_v = mlx::load_linear(d_model, d_model, vb.pp("linear_v"))?;
        let linear_out = mlx::load_linear(d_model, d_model, vb.pp("linear_out"))?;
        let linear_pos = mlx::load_linear_no_bias(d_model, d_model, vb.pp("linear_pos"))?;

        Ok(Self {
            linear_q,
            linear_k,
            linear_v,
            linear_out,
            linear_pos,
            pos_bias_u,
            pos_bias_v,
            num_heads,
            head_dim,
            d_model,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;

        let q = self
            .linear_q
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .linear_k
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .linear_v
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let p = self
            .linear_pos
            .forward(pos_emb)?
            .reshape((1, 2 * t - 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let pos_bias_u = self
            .pos_bias_u
            .reshape((1, self.num_heads, 1, self.head_dim))?;
        let pos_bias_v = self
            .pos_bias_v
            .reshape((1, self.num_heads, 1, self.head_dim))?;

        let q_u = q.broadcast_add(&pos_bias_u)?.contiguous()?;
        let q_v = q.broadcast_add(&pos_bias_v)?.contiguous()?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let p_t = p.transpose(2, 3)?.contiguous()?;
        let matrix_ac = q_u.matmul(&k_t)?;
        let matrix_bd = rel_shift(&q_v.matmul(&p_t)?)?;
        let matrix_bd = matrix_bd.narrow(3, 0, t)?;

        let scores = matrix_ac
            .broadcast_add(&matrix_bd)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let attn = ops::softmax(&scores, 3)?;

        let out = attn.contiguous()?.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, t, self.d_model))?;

        self.linear_out
            .forward(&out)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

fn rel_shift(x: &Tensor) -> Result<Tensor> {
    let (b, h, qlen, pos_len) = x.dims4()?;
    let x = x.pad_with_zeros(3, 1, 0)?;
    let x = x.reshape((b, h, pos_len + 1, qlen))?;
    let x = x.narrow(2, 1, pos_len)?;
    x.reshape((b, h, qlen, pos_len)).map_err(Error::from)
}

struct SortformerTransformerEncoder {
    layers: Vec<SortformerTransformerLayer>,
}

impl SortformerTransformerEncoder {
    fn load(vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        let mut idx = 0usize;
        loop {
            let layer_vb = vb.pp(format!("layers.{idx}"));
            if !layer_vb.contains_tensor("layer_norm_1.weight") {
                break;
            }
            layers.push(SortformerTransformerLayer::load(layer_vb)?);
            idx += 1;
        }
        if layers.is_empty() {
            return Err(Error::ModelLoadError(
                "Sortformer transformer encoder has no layers".to_string(),
            ));
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out)?;
        }
        Ok(out)
    }
}

struct SortformerTransformerLayer {
    norm1: LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out_proj: Linear,
    norm2: LayerNorm,
    dense_in: Linear,
    dense_out: Linear,
    d_model: usize,
    num_heads: usize,
    head_dim: usize,
}

impl SortformerTransformerLayer {
    fn load(vb: VarBuilder) -> Result<Self> {
        let d_model = vb
            .pp("layer_norm_1")
            .get_unchecked_dtype("weight", DType::F32)?
            .dim(0)?;

        let q_w = vb
            .pp("first_sub_layer.query_net")
            .get_unchecked_dtype("weight", DType::F32)?;
        let (_, q_in) = q_w.dims2()?;
        if q_in != d_model {
            return Err(Error::ModelLoadError(format!(
                "Sortformer transformer query input dim mismatch: expected {d_model}, got {q_in}"
            )));
        }

        let dense_in_w = vb
            .pp("second_sub_layer.dense_in")
            .get_unchecked_dtype("weight", DType::F32)?;
        let (inner_size, dense_in) = dense_in_w.dims2()?;
        if dense_in != d_model {
            return Err(Error::ModelLoadError(format!(
                "Sortformer transformer FFN input dim mismatch: expected {d_model}, got {dense_in}"
            )));
        }

        let num_heads = 8usize;
        if d_model % num_heads != 0 {
            return Err(Error::ModelLoadError(format!(
                "Sortformer transformer hidden size {d_model} is not divisible by {num_heads} heads"
            )));
        }
        let head_dim = d_model / num_heads;

        Ok(Self {
            norm1: layer_norm(d_model, 1e-5, vb.pp("layer_norm_1"))?,
            q: mlx::load_linear(d_model, d_model, vb.pp("first_sub_layer.query_net"))?,
            k: mlx::load_linear(d_model, d_model, vb.pp("first_sub_layer.key_net"))?,
            v: mlx::load_linear(d_model, d_model, vb.pp("first_sub_layer.value_net"))?,
            out_proj: mlx::load_linear(d_model, d_model, vb.pp("first_sub_layer.out_projection"))?,
            norm2: layer_norm(d_model, 1e-5, vb.pp("layer_norm_2"))?,
            dense_in: mlx::load_linear(d_model, inner_size, vb.pp("second_sub_layer.dense_in"))?,
            dense_out: mlx::load_linear(inner_size, d_model, vb.pp("second_sub_layer.dense_out"))?,
            d_model,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let attn = self.self_attention(x)?;
        let h = self.norm1.forward(&x.broadcast_add(&attn)?)?;
        let ff = self
            .dense_out
            .forward(&self.dense_in.forward(&h)?.relu()?)?;
        self.norm2
            .forward(&h.broadcast_add(&ff)?)
            .map_err(Error::from)
    }

    fn self_attention(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;

        let q = self
            .q
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v
            .forward(x)?
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = q
            .matmul(&k_t)?
            .affine(1.0 / (self.head_dim as f64).sqrt(), 0.0)?;
        let attn = ops::softmax(&scores, 3)?;
        let ctx = attn.contiguous()?.matmul(&v)?;

        let ctx = ctx.transpose(1, 2)?.reshape((b, t, self.d_model))?;
        self.out_proj.forward(&ctx).map_err(Error::from)
    }
}

struct SortformerSpeakerHead {
    first_hidden_to_hidden: Linear,
    single_hidden_to_spks: Linear,
}

impl SortformerSpeakerHead {
    fn load(vb: VarBuilder) -> Result<Self> {
        let first_w = vb
            .pp("first_hidden_to_hidden")
            .get_unchecked_dtype("weight", DType::F32)?;
        let (first_out, first_in) = first_w.dims2()?;
        if first_out != first_in {
            return Err(Error::ModelLoadError(format!(
                "Unexpected Sortformer hidden projection shape: [{first_out}, {first_in}]"
            )));
        }

        let second_w = vb
            .pp("single_hidden_to_spks")
            .get_unchecked_dtype("weight", DType::F32)?;
        let (spk_out, spk_in) = second_w.dims2()?;
        if spk_out != MAX_SUPPORTED_SPEAKERS {
            return Err(Error::ModelLoadError(format!(
                "Unexpected Sortformer speaker head output dim {spk_out}; expected {MAX_SUPPORTED_SPEAKERS}"
            )));
        }
        if spk_in != first_out {
            return Err(Error::ModelLoadError(format!(
                "Sortformer speaker head dim mismatch: hidden={first_out}, input={spk_in}"
            )));
        }

        let first_hidden_to_hidden =
            mlx::load_linear(first_in, first_out, vb.pp("first_hidden_to_hidden"))?;
        let single_hidden_to_spks =
            mlx::load_linear(spk_in, spk_out, vb.pp("single_hidden_to_spks"))?;
        Ok(Self {
            first_hidden_to_hidden,
            single_hidden_to_spks,
        })
    }

    fn forward(&self, hidden_out: &Tensor) -> Result<Tensor> {
        let hidden_out = hidden_out.relu()?;
        let hidden_out = self.first_hidden_to_hidden.forward(&hidden_out)?;
        let hidden_out = hidden_out.relu()?;
        let spk_logits = self.single_hidden_to_spks.forward(&hidden_out)?;
        ops::sigmoid(&spk_logits).map_err(Error::from)
    }
}

fn resolve_postprocessing_params(
    _config: &DiarizationConfig,
    min_duration_on_ms: Option<f32>,
    min_duration_off_ms: Option<f32>,
) -> PostProcessingParams {
    let preset = std::env::var("IZWI_SORTFORMER_PP_PRESET")
        .unwrap_or_else(|_| "model".to_string())
        .to_ascii_lowercase();

    let mut params = match preset.as_str() {
        "callhome" | "callhome_v2" => PostProcessingParams {
            onset: 0.641,
            offset: 0.561,
            pad_onset: 0.229,
            pad_offset: 0.079,
            min_duration_on: 0.511,
            min_duration_off: 0.296,
            filter_speech_first: true,
        },
        "dihard3" | "dihard3_v2" => PostProcessingParams {
            onset: 0.56,
            offset: 1.0,
            pad_onset: 0.063,
            pad_offset: 0.002,
            min_duration_on: 0.007,
            min_duration_off: 0.151,
            filter_speech_first: true,
        },
        "legacy" | "legacy_model" => PostProcessingParams {
            onset: 0.25,
            offset: 0.25,
            pad_onset: 0.0,
            pad_offset: 0.0,
            min_duration_on: min_duration_on_ms.unwrap_or(0.0).max(0.0) / 1000.0,
            min_duration_off: min_duration_off_ms.unwrap_or(0.0).max(0.0) / 1000.0,
            filter_speech_first: true,
        },
        _ => PostProcessingParams {
            onset: 0.5,
            offset: 0.5,
            pad_onset: 0.0,
            pad_offset: 0.0,
            min_duration_on: min_duration_on_ms.unwrap_or(0.0).max(0.0) / 1000.0,
            min_duration_off: min_duration_off_ms.unwrap_or(0.0).max(0.0) / 1000.0,
            filter_speech_first: true,
        },
    };

    if let Some(value) = env_postprocessing_value("IZWI_SORTFORMER_PP_ONSET") {
        params.onset = value.clamp(0.0, 1.0);
    }
    if let Some(value) = env_postprocessing_value("IZWI_SORTFORMER_PP_OFFSET") {
        params.offset = value.clamp(0.0, 1.0);
    }
    if let Some(value) = env_postprocessing_value("IZWI_SORTFORMER_PP_PAD_ONSET") {
        params.pad_onset = value.max(0.0);
    }
    if let Some(value) = env_postprocessing_value("IZWI_SORTFORMER_PP_PAD_OFFSET") {
        params.pad_offset = value.max(0.0);
    }
    if let Some(value) = env_postprocessing_value("IZWI_SORTFORMER_PP_MIN_DURATION_ON") {
        params.min_duration_on = value.max(0.0);
    }
    if let Some(value) = env_postprocessing_value("IZWI_SORTFORMER_PP_MIN_DURATION_OFF") {
        params.min_duration_off = value.max(0.0);
    }
    if let Some(value) = env_flag("IZWI_SORTFORMER_PP_FILTER_SPEECH_FIRST") {
        params.filter_speech_first = value;
    }

    params
}

fn should_limit_speaker_channels(config: &DiarizationConfig) -> bool {
    config.min_speakers.is_some() || config.max_speakers.is_some()
}

fn env_postprocessing_value(key: &str) -> Option<f32> {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite())
}

fn env_flag(key: &str) -> Option<bool> {
    std::env::var(key)
        .ok()
        .and_then(|raw| match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
}

fn sortformer_rms_gating_enabled() -> bool {
    env_flag("IZWI_SORTFORMER_ENABLE_RMS_GATING").unwrap_or(false)
}

fn select_speaker_channels(
    stats: &[SpeakerActivityStats],
    min_speakers: usize,
    max_speakers: usize,
) -> Vec<usize> {
    let keep = max_speakers.clamp(min_speakers, MAX_SUPPORTED_SPEAKERS);
    let mut ranked = stats.to_vec();
    ranked.sort_by(|a, b| {
        b.total_duration_secs
            .total_cmp(&a.total_duration_secs)
            .then(b.segment_count.cmp(&a.segment_count))
            .then(b.peak_probability.total_cmp(&a.peak_probability))
            .then(a.speaker_idx.cmp(&b.speaker_idx))
    });

    let active = ranked
        .iter()
        .filter(|stat| stat.segment_count > 0 && stat.total_duration_secs > 0.0)
        .map(|stat| stat.speaker_idx)
        .take(keep)
        .collect::<Vec<_>>();

    if active.len() >= min_speakers {
        return active;
    }

    ranked
        .into_iter()
        .take(keep)
        .map(|stat| stat.speaker_idx)
        .collect()
}

fn ts_vad_post_processing(
    probs: &[[f32; MAX_SUPPORTED_SPEAKERS]],
    speaker_idx: usize,
    params: &PostProcessingParams,
) -> Vec<(f32, f32)> {
    let mut repeated = Vec::with_capacity(probs.len() * TS_VAD_UNIT_FRAME_COUNT);
    for row in probs {
        let value = row[speaker_idx].clamp(0.0, 1.0);
        for _ in 0..TS_VAD_UNIT_FRAME_COUNT {
            repeated.push(value);
        }
    }

    filtering(&binarization(&repeated, params), params)
}

fn binarization(sequence: &[f32], params: &PostProcessingParams) -> Vec<(f32, f32)> {
    let mut speech = false;
    let mut start = 0.0f32;
    let mut segments = Vec::new();
    let mut last_index = 0usize;

    for (idx, &value) in sequence.iter().enumerate() {
        last_index = idx;
        if speech {
            if value < params.offset {
                let seg_start = (start - params.pad_onset).max(0.0);
                let seg_end = idx as f32 * TS_VAD_FRAME_LENGTH_SECS + params.pad_offset;
                if seg_end > seg_start {
                    segments.push((seg_start, seg_end));
                }
                start = idx as f32 * TS_VAD_FRAME_LENGTH_SECS;
                speech = false;
            }
        } else if value > params.onset {
            start = idx as f32 * TS_VAD_FRAME_LENGTH_SECS;
            speech = true;
        }
    }

    if speech {
        let seg_start = (start - params.pad_onset).max(0.0);
        let seg_end = last_index as f32 * TS_VAD_FRAME_LENGTH_SECS + params.pad_offset;
        if seg_end > seg_start {
            segments.push((seg_start, seg_end));
        }
    }

    merge_overlap_ranges(&segments)
}

fn filtering(segments: &[(f32, f32)], params: &PostProcessingParams) -> Vec<(f32, f32)> {
    if segments.is_empty() {
        return Vec::new();
    }

    let mut speech_segments = segments.to_vec();
    if params.filter_speech_first {
        if params.min_duration_on > 0.0 {
            speech_segments = filter_short_segments(&speech_segments, params.min_duration_on);
        }
        if params.min_duration_off > 0.0 && speech_segments.len() > 1 {
            let non_speech_segments = get_gap_segments(&speech_segments);
            let short_non_speech_segments = remove_ranges(
                &non_speech_segments,
                &filter_short_segments(&non_speech_segments, params.min_duration_off),
            );
            if !short_non_speech_segments.is_empty() {
                speech_segments.extend(short_non_speech_segments);
                speech_segments = merge_overlap_ranges(&speech_segments);
            }
        }
    } else {
        if params.min_duration_off > 0.0 && speech_segments.len() > 1 {
            let non_speech_segments = get_gap_segments(&speech_segments);
            let short_non_speech_segments = remove_ranges(
                &non_speech_segments,
                &filter_short_segments(&non_speech_segments, params.min_duration_off),
            );
            if !short_non_speech_segments.is_empty() {
                speech_segments.extend(short_non_speech_segments);
                speech_segments = merge_overlap_ranges(&speech_segments);
            }
        }
        if params.min_duration_on > 0.0 {
            speech_segments = filter_short_segments(&speech_segments, params.min_duration_on);
        }
    }
    speech_segments
}

fn remove_ranges(
    original_segments: &[(f32, f32)],
    to_be_removed_segments: &[(f32, f32)],
) -> Vec<(f32, f32)> {
    if original_segments.is_empty() || to_be_removed_segments.is_empty() {
        return original_segments.to_vec();
    }

    original_segments
        .iter()
        .copied()
        .filter(|segment| {
            !to_be_removed_segments.iter().any(|removed| {
                (segment.0 - removed.0).abs() <= f32::EPSILON
                    && (segment.1 - removed.1).abs() <= f32::EPSILON
            })
        })
        .collect()
}

fn filter_short_segments(segments: &[(f32, f32)], threshold: f32) -> Vec<(f32, f32)> {
    segments
        .iter()
        .copied()
        .filter(|(start, end)| (end - start) >= threshold)
        .collect()
}

fn get_gap_segments(segments: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if segments.len() <= 1 {
        return Vec::new();
    }

    let sorted = sort_ranges(segments);
    sorted
        .windows(2)
        .filter_map(|window| {
            let (_, left_end) = window[0];
            let (right_start, _) = window[1];
            (right_start > left_end).then_some((left_end, right_start))
        })
        .collect()
}

fn merge_overlap_ranges(segments: &[(f32, f32)]) -> Vec<(f32, f32)> {
    if segments.len() <= 1 {
        return segments.to_vec();
    }

    let mut sorted = sort_ranges(segments);
    let mut merged = Vec::with_capacity(sorted.len());
    let mut current = sorted.remove(0);
    for segment in sorted {
        if current.1 >= segment.0 {
            current.1 = current.1.max(segment.1);
        } else {
            merged.push(current);
            current = segment;
        }
    }
    merged.push(current);
    merged
}

fn sort_ranges(segments: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut sorted = segments.to_vec();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    sorted
}

fn average_speaker_probability_for_range(
    probs: &[[f32; MAX_SUPPORTED_SPEAKERS]],
    speaker_idx: usize,
    start_secs: f32,
    end_secs: f32,
    frame_stride_samples: usize,
) -> Option<f32> {
    if probs.is_empty() || end_secs <= start_secs || frame_stride_samples == 0 {
        return None;
    }

    let frame_stride_secs = frame_stride_samples as f32 / TARGET_SAMPLE_RATE as f32;
    let start_frame = (start_secs / frame_stride_secs).floor().max(0.0) as usize;
    let end_frame = ((end_secs / frame_stride_secs).ceil().max(0.0) as usize).min(probs.len());
    if start_frame >= end_frame {
        return None;
    }

    let mut sum = 0.0f32;
    let mut count = 0usize;
    for row in probs.iter().take(end_frame).skip(start_frame) {
        sum += row[speaker_idx];
        count += 1;
    }

    (count > 0).then_some((sum / count as f32).clamp(0.0, 1.0))
}

fn realtime_voice_vad_frame_mask(
    samples: &[f32],
    frame_count: usize,
    frame_stride_samples: usize,
    vad_threshold: f32,
) -> Vec<bool> {
    if frame_count == 0 || frame_stride_samples == 0 {
        return Vec::new();
    }

    let threshold = vad_threshold.clamp(0.001, 1.0);
    let mut mask = vec![false; frame_count];
    for (frame_idx, active) in mask.iter_mut().enumerate().take(frame_count) {
        let start = frame_idx * frame_stride_samples;
        if start >= samples.len() {
            break;
        }
        let end = ((frame_idx + 1) * frame_stride_samples).min(samples.len());
        *active = rms_f32(&samples[start..end]) >= threshold;
    }
    mask
}

fn rms_f32(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_squares: f32 = samples.iter().map(|s| s * s).sum();
    (sum_squares / samples.len() as f32).sqrt()
}

fn smooth_activity_mask(active: &mut [bool], min_speech_frames: usize, min_silence_frames: usize) {
    if active.is_empty() {
        return;
    }

    let mut idx = 0usize;
    while idx < active.len() {
        if active[idx] {
            idx += 1;
            continue;
        }
        let start = idx;
        while idx < active.len() && !active[idx] {
            idx += 1;
        }
        let end = idx;
        let gap_len = end - start;
        let has_left_speech = start > 0 && active[start - 1];
        let has_right_speech = end < active.len() && active[end];
        if has_left_speech && has_right_speech && gap_len <= min_silence_frames {
            for value in &mut active[start..end] {
                *value = true;
            }
        }
    }

    idx = 0;
    while idx < active.len() {
        if !active[idx] {
            idx += 1;
            continue;
        }
        let start = idx;
        while idx < active.len() && active[idx] {
            idx += 1;
        }
        let end = idx;
        if end - start < min_speech_frames {
            for value in &mut active[start..end] {
                *value = false;
            }
        }
    }
}

fn collect_active_regions(active: &[bool]) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    let mut idx = 0usize;
    while idx < active.len() {
        if !active[idx] {
            idx += 1;
            continue;
        }
        let start = idx;
        while idx < active.len() && active[idx] {
            idx += 1;
        }
        let end = idx.saturating_sub(1);
        regions.push((start, end));
    }
    regions
}

fn merge_adjacent_segments(segments: &mut Vec<DiarizationSegment>, merge_gap_secs: f32) {
    if segments.len() <= 1 {
        return;
    }

    let mut by_speaker: BTreeMap<String, Vec<DiarizationSegment>> = BTreeMap::new();
    for segment in segments.drain(..) {
        by_speaker
            .entry(segment.speaker.clone())
            .or_default()
            .push(segment);
    }

    let mut merged_all = Vec::new();
    for (_, mut speaker_segments) in by_speaker {
        speaker_segments.sort_by(|a, b| a.start_secs.total_cmp(&b.start_secs));
        let mut iter = speaker_segments.into_iter();
        let Some(mut current) = iter.next() else {
            continue;
        };

        for segment in iter {
            let gap = (segment.start_secs - current.end_secs).max(0.0);
            if gap <= merge_gap_secs {
                current.end_secs = current.end_secs.max(segment.end_secs);
                current.confidence = match (current.confidence, segment.confidence) {
                    (Some(a), Some(b)) => Some((a + b) / 2.0),
                    (Some(a), None) => Some(a),
                    (None, Some(b)) => Some(b),
                    (None, None) => None,
                };
            } else {
                merged_all.push(current);
                current = segment;
            }
        }
        merged_all.push(current);
    }

    merged_all.sort_by(|a, b| {
        a.start_secs
            .total_cmp(&b.start_secs)
            .then(a.speaker.cmp(&b.speaker))
    });
    *segments = merged_all;
}

fn build_rel_positional_embedding(len: usize, d_model: usize, device: &Device) -> Result<Tensor> {
    if len == 0 {
        return Err(Error::InvalidInput(
            "Cannot build positional embedding for empty sequence".to_string(),
        ));
    }

    let pos_len = 2 * len - 1;
    let mut positions = Vec::with_capacity(pos_len);
    for p in (-(len as isize - 1))..=(len as isize - 1) {
        positions.push((-p) as f32);
    }

    let mut emb = vec![0f32; pos_len * d_model];
    let denom = (10_000f32).ln() / d_model as f32;

    for (pi, p) in positions.iter().enumerate() {
        for i in (0..d_model).step_by(2) {
            let div = (-denom * i as f32).exp();
            let angle = p * div;
            emb[pi * d_model + i] = angle.sin();
            if i + 1 < d_model {
                emb[pi * d_model + i + 1] = angle.cos();
            }
        }
    }

    Tensor::from_vec(emb, (1, pos_len, d_model), device).map_err(Error::from)
}

fn swish(x: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&ops::sigmoid(x)?).map_err(Error::from)
}

fn resample_linear(audio: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if audio.is_empty() || src_rate == 0 || dst_rate == 0 {
        return Vec::new();
    }
    if src_rate == dst_rate {
        return audio.to_vec();
    }

    let src_len = audio.len();
    let dst_len = ((src_len as u64) * (dst_rate as u64) / (src_rate as u64))
        .max(1)
        .min(usize::MAX as u64) as usize;
    let mut out = Vec::with_capacity(dst_len);

    let scale = src_rate as f64 / dst_rate as f64;
    for i in 0..dst_len {
        let src_pos = i as f64 * scale;
        let idx0 = src_pos.floor() as usize;
        let idx1 = (idx0 + 1).min(src_len.saturating_sub(1));
        let frac = (src_pos - idx0 as f64) as f32;
        let sample0 = audio[idx0];
        let sample1 = audio[idx1];
        out.push(sample0 + (sample1 - sample0) * frac);
    }

    out
}

fn hann_window(win_length: usize) -> Vec<f32> {
    if win_length <= 1 {
        return vec![1.0; win_length.max(1)];
    }

    (0..win_length)
        .map(|i| {
            let x = (2.0 * std::f32::consts::PI * i as f32) / (win_length as f32 - 1.0);
            0.5 - 0.5 * x.cos()
        })
        .collect()
}

fn hz_to_mel_slaney(hz: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;

    if hz < min_log_hz {
        hz / f_sp
    } else {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    }
}

fn mel_to_hz_slaney(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;

    if mel < min_log_mel {
        mel * f_sp
    } else {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    }
}

fn mel_filterbank(
    sample_rate: usize,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let nyquist = sample_rate as f32 / 2.0;
    let mel_min = hz_to_mel_slaney(fmin.max(0.0));
    let mel_max = hz_to_mel_slaney(fmax.min(nyquist).max(fmin));

    let mel_points: Vec<f32> = (0..(n_mels + 2))
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.into_iter().map(mel_to_hz_slaney).collect();
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| nyquist * i as f32 / (n_freqs.saturating_sub(1).max(1)) as f32)
        .collect();

    let mut fb = vec![0f32; n_mels * n_freqs];
    for m in 0..n_mels {
        let left = hz_points[m];
        let center = hz_points[m + 1];
        let right = hz_points[m + 2];
        let lower_width = (center - left).max(1e-12);
        let upper_width = (right - center).max(1e-12);
        let enorm = if right > left {
            2.0 / (right - left)
        } else {
            0.0
        };

        for (k, &freq) in fft_freqs.iter().enumerate() {
            let lower = (freq - left) / lower_width;
            let upper = (right - freq) / upper_width;
            fb[m * n_freqs + k] = lower.min(upper).max(0.0) * enorm;
        }
    }

    fb
}

fn preemphasis(x: &mut [f32], preemph: f32) {
    if x.len() < 2 {
        return;
    }

    let mut prev = x[0];
    for sample in x.iter_mut().skip(1) {
        let cur = *sample;
        *sample = cur - preemph * prev;
        prev = cur;
    }
}

fn normalize_per_feature(mel: &mut [f32], n_mels: usize, frames: usize, valid_frames: usize) {
    if valid_frames == 0 {
        return;
    }

    for m in 0..n_mels {
        let row = &mut mel[m * frames..(m + 1) * frames];

        let mean = row[..valid_frames].iter().copied().sum::<f32>() / valid_frames as f32;

        let var = if valid_frames > 1 {
            row[..valid_frames]
                .iter()
                .map(|v| {
                    let d = *v - mean;
                    d * d
                })
                .sum::<f32>()
                / (valid_frames as f32 - 1.0)
        } else {
            0.0
        };

        let std = var.sqrt() + NORMALIZE_EPS;
        for v in row[..valid_frames].iter_mut() {
            *v = (*v - mean) / std;
        }
    }
}

fn normalize_all_features(mel: &mut [f32], n_mels: usize, frames: usize, valid_frames: usize) {
    if valid_frames == 0 {
        return;
    }

    let total = n_mels * valid_frames;
    if total == 0 {
        return;
    }

    let mut sum = 0.0f32;
    for m in 0..n_mels {
        let row = &mel[m * frames..(m + 1) * frames];
        sum += row[..valid_frames].iter().copied().sum::<f32>();
    }
    let mean = sum / total as f32;

    let var = if total > 1 {
        let mut accum = 0.0f32;
        for m in 0..n_mels {
            let row = &mel[m * frames..(m + 1) * frames];
            for value in &row[..valid_frames] {
                let delta = *value - mean;
                accum += delta * delta;
            }
        }
        accum / (total as f32 - 1.0)
    } else {
        0.0
    };

    let std = var.sqrt() + NORMALIZE_EPS;
    for m in 0..n_mels {
        let row = &mut mel[m * frames..(m + 1) * frames];
        for value in &mut row[..valid_frames] {
            *value = (*value - mean) / std;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::DeviceKind;
    use crate::runtime::audio_io::decode_audio_bytes;
    use std::path::PathBuf;

    fn streaming_cfg_for_test() -> SortformerStreamingConfig {
        SortformerStreamingConfig {
            fc_d_model: 2,
            subsampling_factor: 8,
            spkcache_len: 4,
            fifo_len: 2,
            chunk_len: 2,
            spkcache_update_period: 2,
            chunk_left_context: 1,
            chunk_right_context: 1,
            spkcache_sil_frames_per_spk: 0,
            pred_score_threshold: 0.25,
            scores_boost_latest: 0.0,
            sil_threshold: 0.2,
            strong_boost_rate: 0.75,
            weak_boost_rate: 1.5,
            min_pos_scores_rate: 0.5,
        }
    }

    #[test]
    fn sortformer_only_uses_selected_model_device_for_cuda() {
        assert!(!sortformer_uses_selected_model_device(DeviceKind::Cpu));
        assert!(!sortformer_uses_selected_model_device(DeviceKind::Metal));
        assert!(sortformer_uses_selected_model_device(DeviceKind::Cuda));
    }

    #[test]
    fn realtime_voice_vad_frame_mask_uses_rms_threshold() {
        let samples = vec![
            0.0, 0.0, 0.0, 0.0, // silence
            0.3, 0.3, 0.3, 0.3, // speech
            0.0, 0.0, 0.0, 0.0, // silence
            0.4, 0.4, 0.4, 0.4, // speech
        ];
        let mask = realtime_voice_vad_frame_mask(&samples, 4, 4, 0.02);
        assert_eq!(mask, vec![false, true, false, true]);
    }

    #[test]
    fn select_speaker_channels_prefers_active_speakers_by_duration() {
        let stats = vec![
            SpeakerActivityStats {
                speaker_idx: 0,
                total_duration_secs: 8.0,
                peak_probability: 0.70,
                segment_count: 2,
            },
            SpeakerActivityStats {
                speaker_idx: 1,
                total_duration_secs: 2.0,
                peak_probability: 0.90,
                segment_count: 3,
            },
            SpeakerActivityStats {
                speaker_idx: 2,
                total_duration_secs: 5.0,
                peak_probability: 0.60,
                segment_count: 1,
            },
            SpeakerActivityStats {
                speaker_idx: 3,
                total_duration_secs: 0.0,
                peak_probability: 0.99,
                segment_count: 0,
            },
        ];

        let selected = select_speaker_channels(&stats, 1, 2);
        assert_eq!(selected, vec![0, 2]);
    }

    #[test]
    fn select_speaker_channels_backfills_when_min_exceeds_active() {
        let stats = vec![
            SpeakerActivityStats {
                speaker_idx: 0,
                total_duration_secs: 0.0,
                peak_probability: 0.40,
                segment_count: 0,
            },
            SpeakerActivityStats {
                speaker_idx: 1,
                total_duration_secs: 3.0,
                peak_probability: 0.60,
                segment_count: 2,
            },
            SpeakerActivityStats {
                speaker_idx: 2,
                total_duration_secs: 0.0,
                peak_probability: 0.80,
                segment_count: 0,
            },
            SpeakerActivityStats {
                speaker_idx: 3,
                total_duration_secs: 0.0,
                peak_probability: 0.20,
                segment_count: 0,
            },
        ];

        let selected = select_speaker_channels(&stats, 2, 2);
        assert_eq!(selected, vec![1, 2]);
    }

    #[test]
    fn select_speaker_channels_keeps_all_four_active_speakers() {
        let stats = vec![
            SpeakerActivityStats {
                speaker_idx: 0,
                total_duration_secs: 8.0,
                peak_probability: 0.81,
                segment_count: 3,
            },
            SpeakerActivityStats {
                speaker_idx: 1,
                total_duration_secs: 6.0,
                peak_probability: 0.75,
                segment_count: 3,
            },
            SpeakerActivityStats {
                speaker_idx: 2,
                total_duration_secs: 4.0,
                peak_probability: 0.72,
                segment_count: 2,
            },
            SpeakerActivityStats {
                speaker_idx: 3,
                total_duration_secs: 2.0,
                peak_probability: 0.68,
                segment_count: 2,
            },
        ];

        let selected = select_speaker_channels(&stats, 1, 4);
        assert_eq!(selected, vec![0, 1, 2, 3]);
    }

    #[test]
    fn resolve_streaming_config_uses_model_profile_by_default() {
        let cfg = resolve_streaming_config(
            ModelVariant::DiarStreamingSortformer4SpkV21,
            &SortformerModulesConfig::default(),
            512,
        )
        .unwrap();

        assert_eq!(cfg.chunk_len, 188);
        assert_eq!(cfg.chunk_right_context, 1);
        assert_eq!(cfg.fifo_len, 0);
        assert_eq!(cfg.spkcache_update_period, 188);
        assert_eq!(cfg.spkcache_len, 188);
    }

    #[test]
    fn resolve_streaming_config_honors_high_latency_override() {
        let key = "IZWI_SORTFORMER_STREAMING_PROFILE";
        let previous = std::env::var(key).ok();
        std::env::set_var(key, "high");

        let cfg = resolve_streaming_config(
            ModelVariant::DiarStreamingSortformer4SpkV21,
            &SortformerModulesConfig::default(),
            512,
        )
        .unwrap();

        match previous {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }

        assert_eq!(cfg.chunk_len, 340);
        assert_eq!(cfg.chunk_right_context, 40);
        assert_eq!(cfg.fifo_len, 40);
        assert_eq!(cfg.spkcache_update_period, 300);
        assert_eq!(cfg.spkcache_len, 188);
    }

    #[test]
    fn resolve_postprocessing_params_defaults_to_reference_binarization() {
        let params = resolve_postprocessing_params(&DiarizationConfig::default(), None, None);
        assert_eq!(params.onset, 0.5);
        assert_eq!(params.offset, 0.5);
        assert_eq!(params.pad_onset, 0.0);
        assert_eq!(params.pad_offset, 0.0);
        assert_eq!(params.min_duration_on, 0.0);
        assert_eq!(params.min_duration_off, 0.0);
        assert!(params.filter_speech_first);
    }

    #[test]
    fn should_limit_speaker_channels_only_when_requested() {
        assert!(!should_limit_speaker_channels(&DiarizationConfig::default()));
        assert!(should_limit_speaker_channels(&DiarizationConfig {
            max_speakers: Some(2),
            ..DiarizationConfig::default()
        }));
    }

    #[test]
    fn plan_streaming_feature_chunks_matches_nemo_style_context_windows() {
        let mut cfg = streaming_cfg_for_test();
        cfg.chunk_len = 6;
        cfg.fifo_len = 188;
        cfg.chunk_right_context = 7;

        let plans = plan_streaming_feature_chunks(120, cfg);
        assert_eq!(plans.len(), 3);
        assert_eq!(plans[0].feature_start, 0);
        assert_eq!(plans[0].feature_end, 104);
        assert_eq!(plans[0].left_offset, 0);
        assert_eq!(plans[0].right_offset, 56);

        assert_eq!(plans[1].feature_start, 40);
        assert_eq!(plans[1].feature_end, 120);
        assert_eq!(plans[1].left_offset, 8);
        assert_eq!(plans[1].right_offset, 24);

        assert_eq!(plans[2].feature_start, 88);
        assert_eq!(plans[2].feature_end, 120);
        assert_eq!(plans[2].left_offset, 8);
        assert_eq!(plans[2].right_offset, 0);
    }

    #[test]
    fn streaming_chunk_offsets_follow_nemo_round_and_ceil_rules() {
        assert_eq!(pre_encoded_left_offset(3, 8), 0);
        assert_eq!(pre_encoded_left_offset(4, 8), 1);
        assert_eq!(pre_encoded_left_offset(8, 8), 1);

        assert_eq!(pre_encoded_right_offset(1, 8), 1);
        assert_eq!(pre_encoded_right_offset(9, 8), 2);
        assert_eq!(pre_encoded_right_offset(56, 8), 7);
    }

    #[test]
    fn update_streaming_state_moves_oldest_fifo_frames_into_speaker_cache() {
        let cfg = streaming_cfg_for_test();
        let state = SortformerStreamingState::new(2);
        let first_chunk = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
        let first_preds = vec![[0.9, 0.0, 0.0, 0.0], [0.8, 0.0, 0.0, 0.0]];
        let (state, first_chunk_preds) =
            update_streaming_state(state, &first_chunk, &first_preds, 0, 0, cfg).unwrap();
        assert_eq!(first_chunk_preds, first_preds);
        assert!(state.spkcache.is_empty());
        assert_eq!(state.fifo, first_chunk);

        let second_chunk = vec![vec![3.0, 3.0], vec![4.0, 4.0]];
        let second_preds = vec![
            [0.9, 0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.0, 0.0],
            [0.0, 0.8, 0.0, 0.0],
        ];
        let (state, second_chunk_preds) =
            update_streaming_state(state, &second_chunk, &second_preds, 0, 0, cfg).unwrap();

        assert_eq!(state.spkcache, vec![vec![1.0, 1.0], vec![2.0, 2.0]]);
        assert_eq!(state.fifo, vec![vec![3.0, 3.0], vec![4.0, 4.0]]);
        assert!(state.spkcache_preds.is_none());
        assert_eq!(
            second_chunk_preds,
            vec![[0.0, 0.9, 0.0, 0.0], [0.0, 0.8, 0.0, 0.0]]
        );
    }

    #[test]
    fn binarization_matches_nemo_threshold_transitions() {
        let params = PostProcessingParams {
            onset: 0.5,
            offset: 0.5,
            pad_onset: 0.0,
            pad_offset: 0.0,
            min_duration_on: 0.0,
            min_duration_off: 0.0,
            filter_speech_first: true,
        };

        let sequence = vec![0.1, 0.6, 0.7, 0.2, 0.1];
        let segments = binarization(&sequence, &params);

        assert_eq!(segments, vec![(0.01, 0.03)]);
    }

    #[test]
    fn filtering_merges_short_non_speech_gaps_like_nemo_default_order() {
        let params = PostProcessingParams {
            onset: 0.5,
            offset: 0.5,
            pad_onset: 0.0,
            pad_offset: 0.0,
            min_duration_on: 0.0,
            min_duration_off: 0.15,
            filter_speech_first: true,
        };

        let segments = vec![(0.0, 0.5), (0.55, 1.0), (1.3, 1.7)];
        let filtered = filtering(&segments, &params);

        assert_eq!(filtered, vec![(0.0, 1.0), (1.3, 1.7)]);
    }

    #[test]
    fn filtering_respects_filter_speech_first_toggle() {
        let segments = vec![(0.0, 0.10), (0.14, 0.22)];
        let speech_first = PostProcessingParams {
            onset: 0.5,
            offset: 0.5,
            pad_onset: 0.0,
            pad_offset: 0.0,
            min_duration_on: 0.12,
            min_duration_off: 0.08,
            filter_speech_first: true,
        };
        let nonspeech_first = PostProcessingParams {
            filter_speech_first: false,
            ..speech_first
        };

        assert!(filtering(&segments, &speech_first).is_empty());
        assert_eq!(filtering(&segments, &nonspeech_first), vec![(0.0, 0.22)]);
    }

    #[test]
    fn smooth_activity_mask_fills_gaps_and_removes_short_bursts() {
        let mut active = vec![
            true, true, false, true, true, false, false, false, true, false, false,
        ];
        smooth_activity_mask(&mut active, 2, 1);
        assert_eq!(
            active,
            vec![true, true, true, true, true, false, false, false, false, false, false]
        );
    }

    #[test]
    fn merge_adjacent_segments_merges_per_speaker_with_overlap_present() {
        let mut segments = vec![
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 0.0,
                end_secs: 1.0,
                confidence: Some(0.8),
            },
            DiarizationSegment {
                speaker: "SPEAKER_01".to_string(),
                start_secs: 0.8,
                end_secs: 1.4,
                confidence: Some(0.9),
            },
            DiarizationSegment {
                speaker: "SPEAKER_00".to_string(),
                start_secs: 1.05,
                end_secs: 2.0,
                confidence: Some(0.6),
            },
        ];

        merge_adjacent_segments(&mut segments, 0.1);

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].speaker, "SPEAKER_00");
        assert!((segments[0].start_secs - 0.0).abs() < 1e-6);
        assert!((segments[0].end_secs - 2.0).abs() < 1e-6);
        assert_eq!(segments[1].speaker, "SPEAKER_01");
    }

    #[test]
    #[ignore = "requires local Sortformer checkpoint"]
    fn sortformer_local_checkpoint_matches_diarization_2_reference_segments() {
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
        let model_dir = models_root.join(ModelVariant::DiarStreamingSortformer4SpkV21.dir_name());
        if !model_dir
            .join("diar_streaming_sortformer_4spk-v2.1.nemo")
            .exists()
        {
            eprintln!(
                "Skipping Sortformer checkpoint test, model not found at {}",
                model_dir.display()
            );
            return;
        }

        let audio_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../data/diarization-2.mp3");
        let audio_bytes = std::fs::read(&audio_path).expect("sample audio should exist");
        let (samples, sample_rate) = decode_audio_bytes(&audio_bytes).expect("audio should decode");

        let model = SortformerDiarizerModel::load(
            &model_dir,
            ModelVariant::DiarStreamingSortformer4SpkV21,
            DeviceProfile::cpu(),
        )
        .expect("sortformer checkpoint should load");
        let diarization = model
            .diarize(&samples, sample_rate, &DiarizationConfig::default())
            .expect("sortformer diarization should run");

        let expected = vec![
            (0.40, 2.72, 0usize),
            (3.20, 4.96, 0),
            (5.44, 10.08, 0),
            (10.72, 15.52, 0),
            (15.60, 18.32, 0),
            (19.60, 20.08, 0),
            (20.88, 22.88, 0),
            (23.28, 27.84, 0),
            (28.40, 29.84, 0),
            (30.16, 36.96, 0),
            (38.40, 42.64, 0),
            (42.80, 62.16, 1),
            (62.40, 68.80, 2),
            (69.44, 92.80, 2),
            (92.88, 97.60, 3),
            (97.92, 104.00, 0),
            (104.16, 116.55, 0),
        ];

        assert_eq!(
            diarization.segments.len(),
            expected.len(),
            "unexpected segment count: {:#?}",
            diarization.segments
        );

        for (segment, (expected_start, expected_end, expected_speaker)) in
            diarization.segments.iter().zip(expected.iter())
        {
            let actual_speaker = parse_test_speaker_id(&segment.speaker);
            assert!(
                (segment.start_secs - expected_start).abs() <= 0.02,
                "segment start mismatch for {:?}: expected {}, got {}",
                segment,
                expected_start,
                segment.start_secs
            );
            assert!(
                (segment.end_secs - expected_end).abs() <= 0.02,
                "segment end mismatch for {:?}: expected {}, got {}",
                segment,
                expected_end,
                segment.end_secs
            );
            assert_eq!(
                actual_speaker, *expected_speaker,
                "speaker mismatch for {:?}",
                segment
            );
        }
    }

    fn parse_test_speaker_id(label: &str) -> usize {
        label
            .chars()
            .filter(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse::<usize>()
            .unwrap_or(0)
    }
}
