use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::ops;
use candle_nn::{
    layer_norm, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder,
};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use serde_json::json;

use super::config::NemotronConfigInventory;
use crate::error::{Error, Result};
use crate::models::shared::weights::mlx;

const SAMPLE_RATE: u32 = 16_000;
const ENCODER_LAYERS: usize = 24;
const ENCODER_DIM: usize = 1024;
const ENCODER_HEADS: usize = 8;
const ENCODER_HEAD_DIM: usize = ENCODER_DIM / ENCODER_HEADS;
const FF_DIM: usize = ENCODER_DIM * 4;
const PRED_HIDDEN: usize = 640;
const JOINT_HIDDEN: usize = 640;
const PROMPT_DIM: usize = 128;
const PROMPT_HIDDEN: usize = 2048;
const CONV_SUB_CHANNELS: usize = 256;
const CONV_KERNEL_1D: usize = 9;
const SUBSAMPLING_FACTOR: usize = 8;
const N_FFT: usize = 512;
const WIN_LENGTH: usize = 400;
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 128;
const PREEMPH: f32 = 0.97;
const LOG_GUARD: f32 = 5.960_464_5e-8;
const NORMALIZE_EPS: f32 = 1e-5;
const DEFAULT_MAX_SYMBOLS_PER_FRAME: usize = 10;

#[derive(Debug, Clone)]
pub(super) struct NemotronDecodeStats {
    pub encoded_frames: usize,
    pub emitted_tokens: usize,
    pub blank_frames: usize,
    pub guard_exits: usize,
    pub max_symbols_per_frame: usize,
}

impl NemotronDecodeStats {
    pub(super) fn diagnostics(&self) -> serde_json::Value {
        json!({
            "encoded_frames": self.encoded_frames,
            "emitted_tokens": self.emitted_tokens,
            "blank_frames": self.blank_frames,
            "guard_exits": self.guard_exits,
            "max_symbols_per_frame": self.max_symbols_per_frame,
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct NemotronDecodedTokens {
    pub token_ids: Vec<usize>,
    pub stats: NemotronDecodeStats,
}

pub(super) struct NemotronNetwork {
    preprocessor: NemotronPreprocessor,
    pre_encode: ConvSubsamplingDw,
    layers: Vec<ConformerLayer>,
    predictor: Predictor,
    joint: Joint,
    prompt_kernel: PromptKernel,
    prompt_dictionary: HashMap<String, usize>,
    left_context_frames: usize,
    right_context_frames: usize,
    blank_idx: usize,
    max_symbols_per_frame: usize,
    rel_pos_cache: Mutex<HashMap<RelPosCacheKey, Tensor>>,
    att_mask_cache: Mutex<HashMap<LimitedMaskCacheKey, Tensor>>,
    dtype: DType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RelPosCacheKey {
    len: usize,
    dtype: DType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct LimitedMaskCacheKey {
    len: usize,
    left_context: usize,
    right_context: usize,
    dtype: DType,
}

impl NemotronNetwork {
    pub(super) fn load(vb: &VarBuilder, inventory: &NemotronConfigInventory) -> Result<Self> {
        validate_inventory_for_forward(inventory)?;

        let preprocessor = NemotronPreprocessor::load(vb, inventory)?;
        let pre_encode = ConvSubsamplingDw::load(vb.pp("encoder.pre_encode"))?;

        let mut layers = Vec::with_capacity(ENCODER_LAYERS);
        for idx in 0..ENCODER_LAYERS {
            layers.push(ConformerLayer::load(
                vb.pp(format!("encoder.layers.{idx}")),
            )?);
        }

        let predictor = Predictor::load(vb.pp("decoder.prediction"))?;
        let blank_idx = inventory
            .vocab_size
            .unwrap_or_else(|| predictor.blank_idx.saturating_sub(0));
        if predictor.blank_idx != blank_idx {
            return Err(Error::ModelLoadError(format!(
                "Nemotron predictor blank index mismatch: embed_blank={}, config_blank={blank_idx}",
                predictor.blank_idx
            )));
        }

        let joint = Joint::load(vb.pp("joint"), ENCODER_DIM, blank_idx + 1)?;
        let prompt_kernel = PromptKernel::load(vb.pp("prompt_kernel"))?;
        let prompt_dictionary = inventory
            .prompt_dictionary
            .iter()
            .cloned()
            .collect::<HashMap<_, _>>();
        if prompt_dictionary.is_empty() {
            return Err(Error::ModelLoadError(
                "Nemotron config does not include prompt_dictionary entries".to_string(),
            ));
        }

        Ok(Self {
            preprocessor,
            pre_encode,
            layers,
            predictor,
            joint,
            prompt_kernel,
            prompt_dictionary,
            left_context_frames: inventory.left_context_frames.unwrap_or(56),
            right_context_frames: inventory
                .right_context_frames
                .iter()
                .copied()
                .max()
                .unwrap_or(13),
            blank_idx,
            max_symbols_per_frame: DEFAULT_MAX_SYMBOLS_PER_FRAME,
            rel_pos_cache: Mutex::new(HashMap::new()),
            att_mask_cache: Mutex::new(HashMap::new()),
            dtype: vb.dtype(),
        })
    }

    pub(super) fn prompt_id(&self, target_lang: &str) -> Result<usize> {
        self.prompt_dictionary
            .get(target_lang)
            .copied()
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "Nemotron prompt dictionary does not contain target_lang '{target_lang}'"
                ))
            })
    }

    pub(super) fn encode_with_prompt(
        &self,
        audio_16khz: &[f32],
        prompt_id: usize,
    ) -> Result<(Tensor, usize)> {
        let (features, feature_frames) = self.preprocessor.compute_features(audio_16khz)?;
        let features = if features.dtype() == self.dtype {
            features
        } else {
            features.to_dtype(self.dtype)?
        };
        let (mut x, encoded_len) = self.pre_encode.forward(&features, feature_frames)?;
        if encoded_len == 0 {
            return Err(Error::InferenceError(
                "Nemotron encoder produced zero frames".to_string(),
            ));
        }

        let pos_len = x.dim(1)?;
        let pos_emb = self.rel_positional_embedding(pos_len, x.dtype(), x.device())?;
        let att_mask = self.limited_context_mask(
            pos_len,
            self.left_context_frames,
            self.right_context_frames,
            x.dtype(),
            x.device(),
        )?;
        for layer in &self.layers {
            x = layer.forward(&x, &pos_emb, &att_mask)?;
        }

        let x = self.prompt_kernel.forward(&x, prompt_id)?;
        Ok((x, encoded_len.min(pos_len)))
    }

    pub(super) fn decode_rnnt_greedy(
        &self,
        encoded: &Tensor,
        encoded_len: usize,
        on_token: &mut dyn FnMut(usize),
    ) -> Result<NemotronDecodedTokens> {
        if encoded.device().is_cuda() {
            return self.decode_rnnt_greedy_cuda_cached(encoded, encoded_len, on_token);
        }

        let encoded = encoded.i((0, ..encoded_len, ..))?; // [T, D]
        let mut predictor_state = self.predictor.initial_state(1, encoded.device())?;
        let mut predictor_out =
            self.predictor
                .step(self.blank_idx, &mut predictor_state, encoded.device())?;

        let mut token_ids = Vec::new();
        let mut blank_frames = 0usize;
        let mut guard_exits = 0usize;

        for t in 0..encoded_len {
            let enc_t = encoded.i((t, ..))?.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, D]
            let mut symbols_this_frame = 0usize;

            loop {
                let logits = self
                    .joint
                    .joint_after_projection(&enc_t, &predictor_out)?
                    .squeeze(0)?
                    .squeeze(0)?
                    .squeeze(0)?; // [V + blank]
                let label = argmax_1d(&logits)?;

                if label == self.blank_idx {
                    blank_frames = blank_frames.saturating_add(1);
                    break;
                }
                if label > self.blank_idx {
                    return Err(Error::InferenceError(format!(
                        "Nemotron RNNT emitted invalid label {label}; blank_idx={}",
                        self.blank_idx
                    )));
                }

                token_ids.push(label);
                on_token(label);
                symbols_this_frame = symbols_this_frame.saturating_add(1);
                if symbols_this_frame >= self.max_symbols_per_frame {
                    guard_exits = guard_exits.saturating_add(1);
                    break;
                }

                predictor_out =
                    self.predictor
                        .step(label, &mut predictor_state, encoded.device())?;
            }
        }

        Ok(NemotronDecodedTokens {
            stats: NemotronDecodeStats {
                encoded_frames: encoded_len,
                emitted_tokens: token_ids.len(),
                blank_frames,
                guard_exits,
                max_symbols_per_frame: self.max_symbols_per_frame,
            },
            token_ids,
        })
    }

    fn decode_rnnt_greedy_cuda_cached(
        &self,
        encoded: &Tensor,
        encoded_len: usize,
        on_token: &mut dyn FnMut(usize),
    ) -> Result<NemotronDecodedTokens> {
        let encoded = encoded.i((0, ..encoded_len, ..))?; // [T, D]
        let encoded_projection = self.joint.project_encoder(&encoded)?; // [T, H]
        let mut predictor_state = self.predictor.initial_state(1, encoded.device())?;
        let mut predictor_out =
            self.predictor
                .step(self.blank_idx, &mut predictor_state, encoded.device())?;
        let mut predictor_projection = self.joint.project_predictor(&predictor_out)?;

        let mut token_ids = Vec::new();
        let mut blank_frames = 0usize;
        let mut guard_exits = 0usize;

        for t in 0..encoded_len {
            let enc_t = encoded_projection.i((t, ..))?.unsqueeze(0)?.unsqueeze(0)?;
            let mut symbols_this_frame = 0usize;

            loop {
                let logits = self
                    .joint
                    .joint_from_projections(&enc_t, &predictor_projection)?
                    .squeeze(0)?
                    .squeeze(0)?
                    .squeeze(0)?;
                let label = argmax_1d(&logits)?;

                if label == self.blank_idx {
                    blank_frames = blank_frames.saturating_add(1);
                    break;
                }
                if label > self.blank_idx {
                    return Err(Error::InferenceError(format!(
                        "Nemotron RNNT emitted invalid label {label}; blank_idx={}",
                        self.blank_idx
                    )));
                }

                token_ids.push(label);
                on_token(label);
                symbols_this_frame = symbols_this_frame.saturating_add(1);
                if symbols_this_frame >= self.max_symbols_per_frame {
                    guard_exits = guard_exits.saturating_add(1);
                    break;
                }

                predictor_out =
                    self.predictor
                        .step(label, &mut predictor_state, encoded.device())?;
                predictor_projection = self.joint.project_predictor(&predictor_out)?;
            }
        }

        Ok(NemotronDecodedTokens {
            stats: NemotronDecodeStats {
                encoded_frames: encoded_len,
                emitted_tokens: token_ids.len(),
                blank_frames,
                guard_exits,
                max_symbols_per_frame: self.max_symbols_per_frame,
            },
            token_ids,
        })
    }

    pub(super) fn blank_idx(&self) -> usize {
        self.blank_idx
    }

    pub(super) fn dtype(&self) -> DType {
        self.dtype
    }

    fn rel_positional_embedding(
        &self,
        len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        if !device.is_cuda() {
            return build_rel_positional_embedding(len, ENCODER_DIM, device);
        }

        let key = RelPosCacheKey { len, dtype };
        if let Some(cached) = self
            .rel_pos_cache
            .lock()
            .map_err(|_| Error::InferenceError("Nemotron rel-pos cache lock poisoned".into()))?
            .get(&key)
            .cloned()
        {
            return Ok(cached);
        }

        let tensor = build_rel_positional_embedding_for_dtype(len, ENCODER_DIM, device, dtype)?;
        self.rel_pos_cache
            .lock()
            .map_err(|_| Error::InferenceError("Nemotron rel-pos cache lock poisoned".into()))?
            .insert(key, tensor.clone());
        Ok(tensor)
    }

    fn limited_context_mask(
        &self,
        len: usize,
        left_context: usize,
        right_context: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        if !device.is_cuda() {
            return build_limited_context_mask(len, left_context, right_context, device);
        }

        let key = LimitedMaskCacheKey {
            len,
            left_context,
            right_context,
            dtype,
        };
        if let Some(cached) = self
            .att_mask_cache
            .lock()
            .map_err(|_| Error::InferenceError("Nemotron mask cache lock poisoned".into()))?
            .get(&key)
            .cloned()
        {
            return Ok(cached);
        }

        let tensor =
            build_limited_context_mask_for_dtype(len, left_context, right_context, device, dtype)?;
        self.att_mask_cache
            .lock()
            .map_err(|_| Error::InferenceError("Nemotron mask cache lock poisoned".into()))?
            .insert(key, tensor.clone());
        Ok(tensor)
    }
}

fn validate_inventory_for_forward(inventory: &NemotronConfigInventory) -> Result<()> {
    expect_dim("sample_rate", inventory.sample_rate, SAMPLE_RATE as usize)?;
    expect_dim("features", inventory.features, N_MELS)?;
    expect_dim("n_fft", inventory.n_fft, N_FFT)?;
    expect_dim("window_length", inventory.window_length, WIN_LENGTH)?;
    expect_dim("hop_length", inventory.hop_length, HOP_LENGTH)?;
    expect_dim("encoder_layers", inventory.encoder_layers, ENCODER_LAYERS)?;
    expect_dim("encoder_dim", inventory.encoder_dim, ENCODER_DIM)?;
    expect_dim("encoder_heads", inventory.encoder_heads, ENCODER_HEADS)?;
    expect_dim(
        "subsampling_factor",
        inventory.subsampling_factor,
        SUBSAMPLING_FACTOR,
    )?;
    expect_dim(
        "subsampling_conv_channels",
        inventory.subsampling_conv_channels,
        CONV_SUB_CHANNELS,
    )?;
    expect_dim("ff_expansion_factor", inventory.ff_expansion_factor, 4)?;
    expect_dim(
        "conv_kernel_size",
        inventory.conv_kernel_size,
        CONV_KERNEL_1D,
    )?;
    expect_dim("predictor_hidden", inventory.predictor_hidden, PRED_HIDDEN)?;
    expect_dim("predictor_layers", inventory.predictor_layers, 2)?;
    expect_dim("joint_hidden", inventory.joint_hidden, JOINT_HIDDEN)?;
    expect_dim("prompt_dim", inventory.prompt_dim, PROMPT_DIM)?;
    if inventory.vocab_size.is_none() {
        return Err(Error::ModelLoadError(
            "Nemotron config is missing vocab_size/num_classes".to_string(),
        ));
    }
    Ok(())
}

fn expect_dim(name: &str, actual: Option<usize>, expected: usize) -> Result<()> {
    if actual.is_some_and(|actual| actual != expected) {
        return Err(Error::ModelLoadError(format!(
            "Nemotron config {name} mismatch: expected {expected}, got {}",
            actual.unwrap()
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FeatureNormalize {
    None,
    PerFeature,
}

impl FeatureNormalize {
    fn from_config(value: Option<&str>) -> Self {
        match value
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| value.to_ascii_lowercase())
            .as_deref()
        {
            Some("per_feature") | Some("true") => Self::PerFeature,
            _ => Self::None,
        }
    }
}

struct NemotronPreprocessor {
    device: Device,
    padded_window: Vec<f32>,
    fb: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
    normalize: FeatureNormalize,
}

pub(super) struct NemotronStreamingFeatureState {
    preemphasized: Vec<f32>,
    last_raw_sample: Option<f32>,
    next_frame: usize,
    input_finished: bool,
}

pub(super) struct NemotronStreamingFeatureChunk {
    pub features: Tensor,
    pub start_frame: usize,
    pub frames: usize,
    pub total_ready_frames: usize,
    pub is_final: bool,
}

pub(super) struct NemotronStreamingPreEncodeState {
    features: Option<Tensor>,
    feature_frames: usize,
    emitted_encoded_frames: usize,
    input_finished: bool,
}

pub(super) struct NemotronStreamingEncodedChunk {
    pub encoded: Tensor,
    pub start_frame: usize,
    pub frames: usize,
    pub total_stable_frames: usize,
    pub is_final: bool,
}

impl NemotronStreamingFeatureState {
    pub(super) fn new() -> Self {
        Self {
            preemphasized: Vec::new(),
            last_raw_sample: None,
            next_frame: 0,
            input_finished: false,
        }
    }

    pub(super) fn push_samples(&mut self, samples: &[f32]) -> Result<()> {
        if self.input_finished {
            return Err(Error::InvalidInput(
                "Cannot push audio into a finalized Nemotron feature stream".to_string(),
            ));
        }
        if samples.is_empty() {
            return Ok(());
        }

        self.preemphasized.reserve(samples.len());
        for &sample in samples {
            let value = if let Some(prev) = self.last_raw_sample {
                sample - PREEMPH * prev
            } else {
                sample
            };
            self.preemphasized.push(value);
            self.last_raw_sample = Some(sample);
        }
        Ok(())
    }

    pub(super) fn finish_input(&mut self) {
        self.input_finished = true;
    }

    fn ready_frames(&self) -> usize {
        if self.preemphasized.is_empty() {
            return 0;
        }
        if self.input_finished {
            return ((self.preemphasized.len() + HOP_LENGTH - 1) / HOP_LENGTH).max(1);
        }

        let center_pad = N_FFT / 2;
        if self.preemphasized.len() < center_pad {
            return 0;
        }
        ((self.preemphasized.len() - center_pad) / HOP_LENGTH) + 1
    }
}

impl NemotronStreamingPreEncodeState {
    pub(super) fn new() -> Self {
        Self {
            features: None,
            feature_frames: 0,
            emitted_encoded_frames: 0,
            input_finished: false,
        }
    }

    pub(super) fn push_features(
        &mut self,
        chunk: NemotronStreamingFeatureChunk,
    ) -> Result<()> {
        if self.input_finished {
            return Err(Error::InvalidInput(
                "Cannot push features into a finalized Nemotron pre-encode stream".to_string(),
            ));
        }
        if chunk.start_frame != self.feature_frames {
            return Err(Error::InvalidInput(format!(
                "Nemotron feature stream expected start frame {}, got {}",
                self.feature_frames, chunk.start_frame
            )));
        }
        if chunk.frames == 0 {
            return Ok(());
        }

        self.features = Some(if let Some(existing) = self.features.as_ref() {
            Tensor::cat(&[existing, &chunk.features], 2)?
        } else {
            chunk.features
        });
        self.feature_frames = self.feature_frames.saturating_add(chunk.frames);
        self.input_finished = chunk.is_final;
        Ok(())
    }

    pub(super) fn finish_input(&mut self) {
        self.input_finished = true;
    }
}

impl NemotronPreprocessor {
    fn load(vb: &VarBuilder, inventory: &NemotronConfigInventory) -> Result<Self> {
        let device = vb.device().clone();
        let preproc_vb = vb.pp("preprocessor.featurizer");

        let window_tensor = preproc_vb
            .get_unchecked_dtype("window", DType::F32)
            .map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed to load Nemotron preprocessor window tensor: {e}"
                ))
            })?;
        let window = window_tensor.to_vec1::<f32>()?;
        if window.len() != WIN_LENGTH {
            return Err(Error::ModelLoadError(format!(
                "Unexpected Nemotron window length: expected {WIN_LENGTH}, got {}",
                window.len()
            )));
        }

        let fb_tensor = preproc_vb
            .get_unchecked_dtype("fb", DType::F32)
            .map_err(|e| {
                Error::ModelLoadError(format!(
                    "Failed to load Nemotron preprocessor filterbank tensor: {e}"
                ))
            })?;
        let (_, n_mels, n_freqs) = fb_tensor.dims3().map_err(|e| {
            Error::ModelLoadError(format!("Unexpected Nemotron filterbank tensor shape: {e}"))
        })?;
        if n_mels != N_MELS || n_freqs != N_FFT / 2 + 1 {
            return Err(Error::ModelLoadError(format!(
                "Unexpected Nemotron filterbank shape: expected (1,{N_MELS},{}), got (1,{n_mels},{n_freqs})",
                N_FFT / 2 + 1
            )));
        }
        let fb = fb_tensor.squeeze(0)?.flatten_all()?.to_vec1::<f32>()?;

        let mut padded_window = vec![0f32; N_FFT];
        let offset = (N_FFT - WIN_LENGTH) / 2;
        padded_window[offset..offset + WIN_LENGTH].copy_from_slice(&window);

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);

        Ok(Self {
            device,
            padded_window,
            fb,
            fft,
            normalize: FeatureNormalize::from_config(inventory.normalize.as_deref()),
        })
    }

    fn compute_features(&self, audio: &[f32]) -> Result<(Tensor, usize)> {
        if audio.is_empty() {
            return Err(Error::InvalidInput("Empty audio input".to_string()));
        }

        let mut x = audio.to_vec();
        preemphasis(&mut x, PREEMPH);

        let center_pad = N_FFT / 2;
        let mut padded = Vec::with_capacity(x.len() + center_pad * 2);
        padded.extend(std::iter::repeat(0.0).take(center_pad));
        padded.extend_from_slice(&x);
        padded.extend(std::iter::repeat(0.0).take(center_pad));

        let frame_count = if padded.len() >= N_FFT {
            (padded.len() - N_FFT) / HOP_LENGTH + 1
        } else {
            1
        };

        let mut spectrum = vec![0f32; frame_count * (N_FFT / 2 + 1)];
        let mut buffer = vec![Complex::<f32>::new(0.0, 0.0); N_FFT];

        for frame_idx in 0..frame_count {
            let start = frame_idx * HOP_LENGTH;
            let slice = &padded[start..start + N_FFT];
            for i in 0..N_FFT {
                buffer[i].re = slice[i] * self.padded_window[i];
                buffer[i].im = 0.0;
            }

            self.fft.process(&mut buffer);

            for k in 0..(N_FFT / 2 + 1) {
                let mag = (buffer[k].re * buffer[k].re + buffer[k].im * buffer[k].im).sqrt();
                spectrum[frame_idx * (N_FFT / 2 + 1) + k] = mag * mag;
            }
        }

        let n_freqs = N_FFT / 2 + 1;
        let mut mel = vec![0f32; N_MELS * frame_count];
        for m in 0..N_MELS {
            for t in 0..frame_count {
                let mut acc = 0f32;
                let spec_row = &spectrum[t * n_freqs..(t + 1) * n_freqs];
                let fb_row = &self.fb[m * n_freqs..(m + 1) * n_freqs];
                for f in 0..n_freqs {
                    acc += spec_row[f] * fb_row[f];
                }
                mel[m * frame_count + t] = (acc + LOG_GUARD).ln();
            }
        }

        let valid_frames = ((audio.len() + HOP_LENGTH - 1) / HOP_LENGTH)
            .max(1)
            .min(frame_count);
        if self.normalize == FeatureNormalize::PerFeature {
            normalize_per_feature(&mut mel, N_MELS, frame_count, valid_frames);
        }

        if valid_frames < frame_count {
            for m in 0..N_MELS {
                for t in valid_frames..frame_count {
                    mel[m * frame_count + t] = 0.0;
                }
            }
        }

        let features = Tensor::from_vec(mel, (1, N_MELS, frame_count), &self.device)?;
        Ok((features, valid_frames))
    }

    pub(super) fn compute_streaming_features(
        &self,
        state: &mut NemotronStreamingFeatureState,
    ) -> Result<Option<NemotronStreamingFeatureChunk>> {
        if self.normalize == FeatureNormalize::PerFeature {
            return Err(Error::InferenceError(
                "Nemotron streaming frontend does not support per-feature normalization".to_string(),
            ));
        }

        let ready_frames = state.ready_frames();
        if ready_frames <= state.next_frame {
            return Ok(None);
        }

        let start_frame = state.next_frame;
        let frames = ready_frames - start_frame;
        let mel = self.compute_mel_frames(
            &state.preemphasized,
            start_frame,
            frames,
            state.input_finished,
        )?;
        let features = Tensor::from_vec(mel, (1, N_MELS, frames), &self.device)?;
        state.next_frame = ready_frames;

        Ok(Some(NemotronStreamingFeatureChunk {
            features,
            start_frame,
            frames,
            total_ready_frames: ready_frames,
            is_final: state.input_finished,
        }))
    }

    fn compute_mel_frames(
        &self,
        preemphasized: &[f32],
        start_frame: usize,
        frames: usize,
        allow_right_padding: bool,
    ) -> Result<Vec<f32>> {
        if frames == 0 {
            return Ok(Vec::new());
        }

        let n_freqs = N_FFT / 2 + 1;
        let center_pad = N_FFT / 2;
        let mut spectrum = vec![0f32; frames * n_freqs];
        let mut buffer = vec![Complex::<f32>::new(0.0, 0.0); N_FFT];

        for local_frame in 0..frames {
            let frame_idx = start_frame + local_frame;
            let center = frame_idx * HOP_LENGTH;
            for i in 0..N_FFT {
                let sample = match center.checked_add(i).and_then(|v| v.checked_sub(center_pad)) {
                    Some(src_idx) if src_idx < preemphasized.len() => preemphasized[src_idx],
                    Some(_) if allow_right_padding => 0.0,
                    Some(src_idx) => {
                        return Err(Error::InferenceError(format!(
                            "Nemotron streaming frontend frame {frame_idx} requires sample {src_idx}, but only {} samples are available",
                            preemphasized.len()
                        )));
                    }
                    None => 0.0,
                };
                buffer[i].re = sample * self.padded_window[i];
                buffer[i].im = 0.0;
            }

            self.fft.process(&mut buffer);

            for k in 0..n_freqs {
                let mag = (buffer[k].re * buffer[k].re + buffer[k].im * buffer[k].im).sqrt();
                spectrum[local_frame * n_freqs + k] = mag * mag;
            }
        }

        let mut mel = vec![0f32; N_MELS * frames];
        for m in 0..N_MELS {
            for t in 0..frames {
                let mut acc = 0f32;
                let spec_row = &spectrum[t * n_freqs..(t + 1) * n_freqs];
                let fb_row = &self.fb[m * n_freqs..(m + 1) * n_freqs];
                for f in 0..n_freqs {
                    acc += spec_row[f] * fb_row[f];
                }
                mel[m * frames + t] = (acc + LOG_GUARD).ln();
            }
        }

        Ok(mel)
    }
}

struct ConvSubsamplingDw {
    conv0: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    conv5: Conv2d,
    conv6: Conv2d,
    out: Linear,
    out_feature_bins: usize,
}

impl ConvSubsamplingDw {
    fn load(vb: VarBuilder) -> Result<Self> {
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

        let conv0 = mlx::load_conv2d(1, CONV_SUB_CHANNELS, 3, stride_cfg, vb.pp("conv.0"))?;

        let mut dw_stride_cfg = stride_cfg;
        dw_stride_cfg.groups = CONV_SUB_CHANNELS;
        let conv2 = mlx::load_conv2d(1, CONV_SUB_CHANNELS, 3, dw_stride_cfg, vb.pp("conv.2"))?;
        let conv3 = mlx::load_conv2d(
            CONV_SUB_CHANNELS,
            CONV_SUB_CHANNELS,
            1,
            point_cfg,
            vb.pp("conv.3"),
        )?;
        let conv5 = mlx::load_conv2d(1, CONV_SUB_CHANNELS, 3, dw_stride_cfg, vb.pp("conv.5"))?;
        let conv6 = mlx::load_conv2d(
            CONV_SUB_CHANNELS,
            CONV_SUB_CHANNELS,
            1,
            point_cfg,
            vb.pp("conv.6"),
        )?;
        let (out, out_feature_bins) = load_subsampling_out_projection(vb.pp("out"))?;

        Ok(Self {
            conv0,
            conv2,
            conv3,
            conv5,
            conv6,
            out,
            out_feature_bins,
        })
    }

    fn forward(&self, features: &Tensor, feature_frames: usize) -> Result<(Tensor, usize)> {
        let mut x = features.transpose(1, 2)?.unsqueeze(1)?;

        x = self.conv0.forward(&x)?.relu()?;
        x = self.conv2.forward(&x)?;
        x = self.conv3.forward(&x)?.relu()?;
        x = self.conv5.forward(&x)?;
        x = self.conv6.forward(&x)?.relu()?;

        let (b, c, t, mut f) = x.dims4()?;
        if c != CONV_SUB_CHANNELS {
            return Err(Error::InferenceError(format!(
                "Nemotron subsampling channel mismatch: expected {CONV_SUB_CHANNELS}, got {c}"
            )));
        }
        if f < self.out_feature_bins {
            x = x.pad_with_zeros(3, 0, self.out_feature_bins - f)?;
            f = self.out_feature_bins;
        } else if f > self.out_feature_bins {
            x = x.narrow(3, 0, self.out_feature_bins)?;
            f = self.out_feature_bins;
        }
        let x = x
            .transpose(1, 2)?
            .reshape((b, t, c * f))?
            .apply(&self.out)?;
        let encoded_len = subsampled_len_3x(feature_frames).min(t).max(1);
        Ok((x, encoded_len))
    }

    pub(super) fn forward_streaming_chunk(
        &self,
        state: &mut NemotronStreamingPreEncodeState,
    ) -> Result<Option<NemotronStreamingEncodedChunk>> {
        let Some(features) = state.features.as_ref() else {
            return Ok(None);
        };
        if state.feature_frames == 0 {
            return Ok(None);
        }

        let stable_frames = if state.input_finished {
            subsampled_len_3x(state.feature_frames)
        } else {
            stable_subsampled_len_3x(state.feature_frames)
        };
        if stable_frames <= state.emitted_encoded_frames {
            return Ok(None);
        }

        let (encoded, encoded_len) = self.forward(features, state.feature_frames)?;
        let stable_frames = stable_frames.min(encoded_len);
        if stable_frames <= state.emitted_encoded_frames {
            return Ok(None);
        }

        let start_frame = state.emitted_encoded_frames;
        let frames = stable_frames - start_frame;
        let encoded = encoded.narrow(1, start_frame, frames)?.contiguous()?;
        state.emitted_encoded_frames = stable_frames;

        Ok(Some(NemotronStreamingEncodedChunk {
            encoded,
            start_frame,
            frames,
            total_stable_frames: stable_frames,
            is_final: state.input_finished && stable_frames == encoded_len,
        }))
    }
}

fn load_subsampling_out_projection(vb: VarBuilder) -> Result<(Linear, usize)> {
    let ws = vb.get_unchecked_dtype("weight", vb.dtype())?;
    let (out_dim, in_dim) = ws.dims2()?;
    if out_dim != ENCODER_DIM || in_dim % CONV_SUB_CHANNELS != 0 {
        return Err(Error::ModelLoadError(format!(
            "Unexpected Nemotron subsampling projection shape: expected out_dim={ENCODER_DIM} and input multiple of {CONV_SUB_CHANNELS}, got ({out_dim},{in_dim})"
        )));
    }
    let bias = if vb.contains_tensor("bias") {
        Some(vb.get(out_dim, "bias")?)
    } else {
        None
    };
    Ok((Linear::new(ws, bias), in_dim / CONV_SUB_CHANNELS))
}

fn subsampled_len_3x(mut len: usize) -> usize {
    for _ in 0..3 {
        len = len.div_ceil(2);
    }
    len
}

fn stable_subsampled_len_3x(feature_frames: usize) -> usize {
    if feature_frames < SUBSAMPLING_FACTOR {
        return 0;
    }
    ((feature_frames - SUBSAMPLING_FACTOR) / SUBSAMPLING_FACTOR) + 1
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
}

impl ConformerLayer {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm_ff1: layer_norm(ENCODER_DIM, 1e-5, vb.pp("norm_feed_forward1"))?,
            ff1: FeedForward::load(vb.pp("feed_forward1"))?,
            norm_self_att: layer_norm(ENCODER_DIM, 1e-5, vb.pp("norm_self_att"))?,
            self_attn: RelPosSelfAttention::load(vb.pp("self_attn"))?,
            norm_conv: layer_norm(ENCODER_DIM, 1e-5, vb.pp("norm_conv"))?,
            conv: ConformerConv::load(vb.pp("conv"))?,
            norm_ff2: layer_norm(ENCODER_DIM, 1e-5, vb.pp("norm_feed_forward2"))?,
            ff2: FeedForward::load(vb.pp("feed_forward2"))?,
            norm_out: layer_norm(ENCODER_DIM, 1e-5, vb.pp("norm_out"))?,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor, att_mask: &Tensor) -> Result<Tensor> {
        let mut residual = x.clone();

        let ff1 = self.ff1.forward(&self.norm_ff1.forward(&residual)?)?;
        residual = residual.broadcast_add(&ff1.affine(0.5, 0.0)?)?;

        let attn =
            self.self_attn
                .forward(&self.norm_self_att.forward(&residual)?, pos_emb, att_mask)?;
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
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear1: mlx::load_linear_no_bias(ENCODER_DIM, FF_DIM, vb.pp("linear1"))?,
            linear2: mlx::load_linear_no_bias(FF_DIM, ENCODER_DIM, vb.pp("linear2"))?,
        })
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
    conv_norm: LayerNorm,
    pointwise_conv2: Conv1d,
}

impl ConformerConv {
    fn load(vb: VarBuilder) -> Result<Self> {
        let pointwise_conv1 = mlx::load_conv1d_no_bias(
            ENCODER_DIM,
            ENCODER_DIM * 2,
            1,
            Conv1dConfig::default(),
            vb.pp("pointwise_conv1"),
        )?;

        let depthwise_conv = mlx::load_conv1d_no_bias(
            ENCODER_DIM,
            ENCODER_DIM,
            CONV_KERNEL_1D,
            Conv1dConfig {
                padding: 0,
                groups: ENCODER_DIM,
                ..Default::default()
            },
            vb.pp("depthwise_conv"),
        )?;

        Ok(Self {
            pointwise_conv1,
            depthwise_conv,
            conv_norm: load_conv_channel_layer_norm(vb.pp("batch_norm"))?,
            pointwise_conv2: mlx::load_conv1d_no_bias(
                ENCODER_DIM,
                ENCODER_DIM,
                1,
                Conv1dConfig::default(),
                vb.pp("pointwise_conv2"),
            )?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.transpose(1, 2)?;
        x = self.pointwise_conv1.forward(&x)?;
        let x_a = x.i((.., ..ENCODER_DIM, ..))?;
        let x_b = x.i((.., ENCODER_DIM.., ..))?;
        x = x_a.broadcast_mul(&ops::sigmoid(&x_b)?)?;

        x = x.pad_with_zeros(2, CONV_KERNEL_1D - 1, 0)?;
        x = self.depthwise_conv.forward(&x)?;
        x = self
            .conv_norm
            .forward(&x.transpose(1, 2)?)?
            .transpose(1, 2)?;
        x = swish(&x)?;
        x = self.pointwise_conv2.forward(&x)?;

        x.transpose(1, 2).map_err(Error::from)
    }
}

fn load_conv_channel_layer_norm(vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get(ENCODER_DIM, "weight")?;
    let bias = vb.get(ENCODER_DIM, "bias")?;
    Ok(LayerNorm::new(weight, bias, 1e-5))
}

struct RelPosSelfAttention {
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    linear_pos: Linear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
}

impl RelPosSelfAttention {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear_q: mlx::load_linear_no_bias(ENCODER_DIM, ENCODER_DIM, vb.pp("linear_q"))?,
            linear_k: mlx::load_linear_no_bias(ENCODER_DIM, ENCODER_DIM, vb.pp("linear_k"))?,
            linear_v: mlx::load_linear_no_bias(ENCODER_DIM, ENCODER_DIM, vb.pp("linear_v"))?,
            linear_out: mlx::load_linear_no_bias(ENCODER_DIM, ENCODER_DIM, vb.pp("linear_out"))?,
            linear_pos: mlx::load_linear_no_bias(ENCODER_DIM, ENCODER_DIM, vb.pp("linear_pos"))?,
            pos_bias_u: vb.get((ENCODER_HEADS, ENCODER_HEAD_DIM), "pos_bias_u")?,
            pos_bias_v: vb.get((ENCODER_HEADS, ENCODER_HEAD_DIM), "pos_bias_v")?,
        })
    }

    fn forward(&self, x: &Tensor, pos_emb: &Tensor, att_mask: &Tensor) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;

        let q = self
            .linear_q
            .forward(x)?
            .reshape((b, t, ENCODER_HEADS, ENCODER_HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .linear_k
            .forward(x)?
            .reshape((b, t, ENCODER_HEADS, ENCODER_HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .linear_v
            .forward(x)?
            .reshape((b, t, ENCODER_HEADS, ENCODER_HEAD_DIM))?
            .transpose(1, 2)?
            .contiguous()?;

        let p = self
            .linear_pos
            .forward(pos_emb)?
            .reshape((1, 2 * t - 1, ENCODER_HEADS, ENCODER_HEAD_DIM))?
            .transpose(1, 2)?;

        let pos_bias_u = self.pos_bias_u.to_dtype(q.dtype())?.reshape((
            1,
            ENCODER_HEADS,
            1,
            ENCODER_HEAD_DIM,
        ))?;
        let pos_bias_v = self.pos_bias_v.to_dtype(q.dtype())?.reshape((
            1,
            ENCODER_HEADS,
            1,
            ENCODER_HEAD_DIM,
        ))?;
        let matrix_ac = q
            .broadcast_add(&pos_bias_u)?
            .matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let matrix_bd = rel_shift(
            &q.broadcast_add(&pos_bias_v)?
                .matmul(&p.transpose(2, 3)?.contiguous()?)?,
        )?
        .narrow(3, 0, t)?;

        let scores = matrix_ac
            .broadcast_add(&matrix_bd)?
            .affine(1.0 / (ENCODER_HEAD_DIM as f64).sqrt(), 0.0)?
            .broadcast_add(att_mask)?;
        let attn = ops::softmax(&scores, 3)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, t, ENCODER_DIM))?;

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

struct PromptKernel {
    linear0: Linear,
    linear2: Linear,
    prompt_cache: Mutex<HashMap<PromptCacheKey, Tensor>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PromptCacheKey {
    prompt_id: usize,
    dtype: DType,
}

impl PromptKernel {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            linear0: mlx::load_linear(ENCODER_DIM + PROMPT_DIM, PROMPT_HIDDEN, vb.pp("0"))?,
            linear2: mlx::load_linear(PROMPT_HIDDEN, ENCODER_DIM, vb.pp("2"))?,
            prompt_cache: Mutex::new(HashMap::new()),
        })
    }

    fn forward(&self, encoded: &Tensor, prompt_id: usize) -> Result<Tensor> {
        if prompt_id >= PROMPT_DIM {
            return Err(Error::InvalidInput(format!(
                "Nemotron prompt id {prompt_id} exceeds prompt feature dimension {PROMPT_DIM}"
            )));
        }

        let (b, t, _d) = encoded.dims3()?;
        let prompt = self.prompt_tensor(encoded, prompt_id, b, t)?;
        let x = Tensor::cat(&[encoded, &prompt], 2)?;
        let x = self.linear0.forward(&x)?.relu()?;
        self.linear2
            .forward(&x)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }

    fn prompt_tensor(
        &self,
        encoded: &Tensor,
        prompt_id: usize,
        b: usize,
        t: usize,
    ) -> Result<Tensor> {
        if !encoded.device().is_cuda() {
            let mut prompt = vec![0f32; b * t * PROMPT_DIM];
            for bi in 0..b {
                for ti in 0..t {
                    prompt[(bi * t + ti) * PROMPT_DIM + prompt_id] = 1.0;
                }
            }
            return Tensor::from_vec(prompt, (b, t, PROMPT_DIM), encoded.device())
                .map_err(Error::from);
        }

        let key = PromptCacheKey {
            prompt_id,
            dtype: encoded.dtype(),
        };
        let base = if let Some(cached) = self
            .prompt_cache
            .lock()
            .map_err(|_| Error::InferenceError("Nemotron prompt cache lock poisoned".into()))?
            .get(&key)
            .cloned()
        {
            cached
        } else {
            let tensor = build_prompt_one_hot(prompt_id, encoded.device(), encoded.dtype())?;
            self.prompt_cache
                .lock()
                .map_err(|_| Error::InferenceError("Nemotron prompt cache lock poisoned".into()))?
                .insert(key, tensor.clone());
            tensor
        };
        base.broadcast_as((b, t, PROMPT_DIM)).map_err(Error::from)
    }
}

struct Predictor {
    embed: Tensor,
    lstm_l0: LstmCell,
    lstm_l1: LstmCell,
    blank_idx: usize,
}

#[derive(Clone)]
struct PredictorState {
    h0: Tensor,
    c0: Tensor,
    h1: Tensor,
    c1: Tensor,
}

impl Predictor {
    fn load(vb: VarBuilder) -> Result<Self> {
        let embed = vb.pp("embed").get_unchecked_dtype("weight", vb.dtype())?;
        let vocab_plus_blank = embed.dim(0)?;
        let hidden = embed.dim(1)?;
        if hidden != PRED_HIDDEN {
            return Err(Error::ModelLoadError(format!(
                "Unexpected Nemotron predictor embedding hidden size: expected {PRED_HIDDEN}, got {hidden}"
            )));
        }

        Ok(Self {
            embed,
            lstm_l0: LstmCell::load(vb.pp("dec_rnn.lstm"), 0)?,
            lstm_l1: LstmCell::load(vb.pp("dec_rnn.lstm"), 1)?,
            blank_idx: vocab_plus_blank.saturating_sub(1),
        })
    }

    fn initial_state(&self, batch: usize, device: &Device) -> Result<PredictorState> {
        let dtype = self.embed.dtype();
        let zeros = |dim| Tensor::zeros((batch, dim), dtype, device).map_err(Error::from);
        Ok(PredictorState {
            h0: zeros(PRED_HIDDEN)?,
            c0: zeros(PRED_HIDDEN)?,
            h1: zeros(PRED_HIDDEN)?,
            c1: zeros(PRED_HIDDEN)?,
        })
    }

    fn step(&self, label: usize, state: &mut PredictorState, device: &Device) -> Result<Tensor> {
        let x = if label == self.blank_idx {
            Tensor::zeros((1, PRED_HIDDEN), self.embed.dtype(), device)?
        } else {
            self.embed.i((label, ..))?.unsqueeze(0)?
        };

        let (h0, c0) = self.lstm_l0.step(&x, &state.h0, &state.c0)?;
        state.h0 = h0;
        state.c0 = c0;

        let (h1, c1) = self.lstm_l1.step(&state.h0, &state.h1, &state.c1)?;
        state.h1 = h1;
        state.c1 = c1;

        Ok(state.h1.unsqueeze(1)?)
    }
}

struct LstmCell {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
    cuda_views: Option<LstmCellCudaViews>,
}

struct LstmCellCudaViews {
    w_ih_t: Tensor,
    w_hh_t: Tensor,
    b_ih_batched: Tensor,
    b_hh_batched: Tensor,
}

impl LstmCell {
    fn load(vb: VarBuilder, layer: usize) -> Result<Self> {
        let w_ih_name = format!("weight_ih_l{layer}");
        let w_hh_name = format!("weight_hh_l{layer}");
        let b_ih_name = format!("bias_ih_l{layer}");
        let b_hh_name = format!("bias_hh_l{layer}");

        let w_ih = vb.get((PRED_HIDDEN * 4, PRED_HIDDEN), &w_ih_name)?;
        let w_hh = vb.get((PRED_HIDDEN * 4, PRED_HIDDEN), &w_hh_name)?;
        let b_ih = vb.get(PRED_HIDDEN * 4, &b_ih_name)?;
        let b_hh = vb.get(PRED_HIDDEN * 4, &b_hh_name)?;
        let cuda_views = if w_ih.device().is_cuda() {
            Some(LstmCellCudaViews {
                w_ih_t: w_ih.transpose(0, 1)?.contiguous()?,
                w_hh_t: w_hh.transpose(0, 1)?.contiguous()?,
                b_ih_batched: b_ih.unsqueeze(0)?,
                b_hh_batched: b_hh.unsqueeze(0)?,
            })
        } else {
            None
        };

        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            cuda_views,
        })
    }

    fn step(&self, x: &Tensor, h_prev: &Tensor, c_prev: &Tensor) -> Result<(Tensor, Tensor)> {
        let gates = if let Some(views) = &self.cuda_views {
            x.matmul(&views.w_ih_t)?
                .broadcast_add(&views.b_ih_batched)?
                .broadcast_add(&h_prev.matmul(&views.w_hh_t)?)?
                .broadcast_add(&views.b_hh_batched)?
        } else {
            x.matmul(&self.w_ih.transpose(0, 1)?)?
                .broadcast_add(&self.b_ih.unsqueeze(0)?)?
                .broadcast_add(&h_prev.matmul(&self.w_hh.transpose(0, 1)?)?)?
                .broadcast_add(&self.b_hh.unsqueeze(0)?)?
        };

        let i = ops::sigmoid(&gates.i((.., 0..PRED_HIDDEN))?)?;
        let f = ops::sigmoid(&gates.i((.., PRED_HIDDEN..(PRED_HIDDEN * 2)))?)?;
        let g = gates
            .i((.., (PRED_HIDDEN * 2)..(PRED_HIDDEN * 3)))?
            .tanh()?;
        let o = ops::sigmoid(&gates.i((.., (PRED_HIDDEN * 3)..))?)?;

        let c = f
            .broadcast_mul(c_prev)?
            .broadcast_add(&i.broadcast_mul(&g)?)?;
        let h = o.broadcast_mul(&c.tanh()?)?;

        Ok((h, c))
    }
}

struct Joint {
    pred: Linear,
    enc: Linear,
    out: Linear,
}

impl Joint {
    fn load(vb: VarBuilder, enc_hidden: usize, num_classes_with_blank: usize) -> Result<Self> {
        let pred = mlx::load_linear(PRED_HIDDEN, JOINT_HIDDEN, vb.pp("pred"))?;
        let enc = mlx::load_linear(enc_hidden, JOINT_HIDDEN, vb.pp("enc"))?;
        let out_bias = vb
            .pp("joint_net.2")
            .get_unchecked_dtype("bias", vb.dtype())?;
        let out_dim = out_bias.dim(0)?;
        if out_dim != num_classes_with_blank {
            return Err(Error::ModelLoadError(format!(
                "Nemotron RNNT joint output mismatch: out_dim={out_dim}, expected classes_with_blank={num_classes_with_blank}"
            )));
        }
        let out = mlx::load_linear(JOINT_HIDDEN, out_dim, vb.pp("joint_net.2"))?;
        Ok(Self { pred, enc, out })
    }

    fn joint_after_projection(&self, f: &Tensor, g: &Tensor) -> Result<Tensor> {
        let f = self.project_encoder(f)?;
        let g = self.project_predictor(g)?;
        self.joint_from_projections(&f, &g)
    }

    fn project_encoder(&self, f: &Tensor) -> Result<Tensor> {
        self.enc
            .forward(f)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }

    fn project_predictor(&self, g: &Tensor) -> Result<Tensor> {
        self.pred
            .forward(g)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }

    fn joint_from_projections(&self, f: &Tensor, g: &Tensor) -> Result<Tensor> {
        let inp = f.unsqueeze(2)?.broadcast_add(&g.unsqueeze(1)?)?;
        let inp = inp.relu()?;
        self.out
            .forward(&inp)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

pub(super) fn resample_linear(audio: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || audio.len() < 2 {
        return audio.to_vec();
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((audio.len() as f64) * ratio).round().max(1.0) as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = (i as f64) / ratio;
        let left = src_pos.floor() as usize;
        let right = left.saturating_add(1).min(audio.len() - 1);
        let frac = (src_pos - left as f64) as f32;
        out.push(audio[left] * (1.0 - frac) + audio[right] * frac);
    }
    out
}

fn build_rel_positional_embedding(len: usize, d_model: usize, device: &Device) -> Result<Tensor> {
    if len == 0 {
        return Err(Error::InvalidInput(
            "Cannot build positional embedding for empty sequence".to_string(),
        ));
    }

    let pos_len = 2 * len - 1;
    let mut emb = vec![0f32; pos_len * d_model];
    let denom = (10_000f32).ln() / d_model as f32;

    for (pi, p) in (-(len as isize - 1)..=(len as isize - 1)).enumerate() {
        let p = (-p) as f32;
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

fn build_rel_positional_embedding_for_dtype(
    len: usize,
    d_model: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    build_rel_positional_embedding(len, d_model, device)?
        .to_dtype(dtype)
        .map_err(Error::from)
}

fn build_limited_context_mask(
    len: usize,
    left_context: usize,
    right_context: usize,
    device: &Device,
) -> Result<Tensor> {
    let mut mask = vec![0f32; len * len];
    for q in 0..len {
        for k in 0..len {
            if k + left_context < q || k > q.saturating_add(right_context) {
                mask[q * len + k] = -1.0e9;
            }
        }
    }
    Tensor::from_vec(mask, (1, 1, len, len), device).map_err(Error::from)
}

fn build_limited_context_mask_for_dtype(
    len: usize,
    left_context: usize,
    right_context: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    build_limited_context_mask(len, left_context, right_context, device)?
        .to_dtype(dtype)
        .map_err(Error::from)
}

fn build_prompt_one_hot(prompt_id: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    if prompt_id >= PROMPT_DIM {
        return Err(Error::InvalidInput(format!(
            "Nemotron prompt id {prompt_id} exceeds prompt feature dimension {PROMPT_DIM}"
        )));
    }
    let mut prompt = vec![0f32; PROMPT_DIM];
    prompt[prompt_id] = 1.0;
    Tensor::from_vec(prompt, (1, 1, PROMPT_DIM), device)?
        .to_dtype(dtype)
        .map_err(Error::from)
}

fn swish(x: &Tensor) -> Result<Tensor> {
    x.broadcast_mul(&ops::sigmoid(x)?).map_err(Error::from)
}

fn argmax_1d(x: &Tensor) -> Result<usize> {
    if x.rank() != 1 {
        return Err(Error::InferenceError(format!(
            "Nemotron RNNT argmax expected rank-1 logits, got shape {:?}",
            x.shape().dims()
        )));
    }
    if x.device().is_cuda() {
        return argmax_1d_device(x);
    }

    argmax_1d_host(x)
}

fn argmax_1d_device(x: &Tensor) -> Result<usize> {
    let idx = x.argmax(D::Minus1)?;
    let idx = if idx.rank() == 0 {
        idx
    } else {
        idx.squeeze(0)?
    };
    Ok(idx.to_dtype(DType::U32)?.to_scalar::<u32>()? as usize)
}

fn argmax_1d_host(x: &Tensor) -> Result<usize> {
    let x = x.to_dtype(DType::F32)?;
    let v = x.to_vec1::<f32>()?;
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &val) in v.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }
    Ok(best_idx)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn test_preprocessor(normalize: FeatureNormalize) -> NemotronPreprocessor {
        let device = Device::Cpu;
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let mut fb = vec![0f32; N_MELS * (N_FFT / 2 + 1)];
        let n_freqs = N_FFT / 2 + 1;
        for m in 0..N_MELS {
            fb[m * n_freqs + (m % n_freqs)] = 1.0;
        }

        NemotronPreprocessor {
            device,
            padded_window: vec![1.0; N_FFT],
            fb,
            fft,
            normalize,
        }
    }

    fn zero_subsampling() -> ConvSubsamplingDw {
        let device = Device::Cpu;
        let stride_cfg = Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let mut dw_stride_cfg = stride_cfg;
        dw_stride_cfg.groups = CONV_SUB_CHANNELS;
        let point_cfg = Conv2dConfig {
            stride: 1,
            padding: 0,
            ..Default::default()
        };
        let conv0 = Conv2d::new(
            Tensor::zeros((CONV_SUB_CHANNELS, 1, 3, 3), DType::F32, &device).unwrap(),
            None,
            stride_cfg,
        );
        let conv2 = Conv2d::new(
            Tensor::zeros((CONV_SUB_CHANNELS, 1, 3, 3), DType::F32, &device).unwrap(),
            None,
            dw_stride_cfg,
        );
        let conv3 = Conv2d::new(
            Tensor::zeros(
                (CONV_SUB_CHANNELS, CONV_SUB_CHANNELS, 1, 1),
                DType::F32,
                &device,
            )
            .unwrap(),
            None,
            point_cfg,
        );
        let conv5 = Conv2d::new(
            Tensor::zeros((CONV_SUB_CHANNELS, 1, 3, 3), DType::F32, &device).unwrap(),
            None,
            dw_stride_cfg,
        );
        let conv6 = Conv2d::new(
            Tensor::zeros(
                (CONV_SUB_CHANNELS, CONV_SUB_CHANNELS, 1, 1),
                DType::F32,
                &device,
            )
            .unwrap(),
            None,
            point_cfg,
        );
        let out_feature_bins = 16;
        let out = Linear::new(
            Tensor::zeros(
                (ENCODER_DIM, CONV_SUB_CHANNELS * out_feature_bins),
                DType::F32,
                &device,
            )
            .unwrap(),
            Some(Tensor::zeros(ENCODER_DIM, DType::F32, &device).unwrap()),
        );

        ConvSubsamplingDw {
            conv0,
            conv2,
            conv3,
            conv5,
            conv6,
            out,
            out_feature_bins,
        }
    }

    fn feature_chunk(
        features: &Tensor,
        start_frame: usize,
        frames: usize,
        is_final: bool,
    ) -> NemotronStreamingFeatureChunk {
        NemotronStreamingFeatureChunk {
            features: features.narrow(2, start_frame, frames).unwrap(),
            start_frame,
            frames,
            total_ready_frames: start_frame + frames,
            is_final,
        }
    }

    fn assert_tensor_close(lhs: &Tensor, rhs: &Tensor, tolerance: f32) {
        assert_eq!(lhs.dims(), rhs.dims());
        let lhs = lhs.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let rhs = rhs.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (idx, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= tolerance,
                "tensor mismatch at {idx}: lhs={a} rhs={b} diff={diff}"
            );
        }
    }

    #[test]
    fn resample_linear_downsamples_24khz_to_16khz_length() {
        let audio = vec![0.0f32; 24_000];
        let out = resample_linear(&audio, 24_000, 16_000);

        assert_eq!(out.len(), 16_000);
    }

    #[test]
    fn subsampled_length_matches_three_stride_two_layers() {
        assert_eq!(subsampled_len_3x(1), 1);
        assert_eq!(subsampled_len_3x(8), 1);
        assert_eq!(subsampled_len_3x(9), 2);
        assert_eq!(subsampled_len_3x(100), 13);
    }

    #[test]
    fn stable_subsampled_length_waits_for_full_stride_receptive_field() {
        assert_eq!(stable_subsampled_len_3x(0), 0);
        assert_eq!(stable_subsampled_len_3x(7), 0);
        assert_eq!(stable_subsampled_len_3x(8), 1);
        assert_eq!(stable_subsampled_len_3x(15), 1);
        assert_eq!(stable_subsampled_len_3x(16), 2);
        assert_eq!(stable_subsampled_len_3x(41), 5);
    }

    #[test]
    fn limited_context_mask_blocks_out_of_window_positions() {
        let mask = build_limited_context_mask(4, 1, 1, &Device::Cpu).unwrap();
        let values = mask
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        assert_eq!(values[0][0], 0.0);
        assert_eq!(values[0][1], 0.0);
        assert!(values[0][2] < -1.0e8);
        assert_eq!(values[2][1], 0.0);
        assert_eq!(values[2][3], 0.0);
        assert!(values[3][1] < -1.0e8);
    }

    #[test]
    fn typed_limited_context_mask_matches_f32_builder_values() {
        let mask = build_limited_context_mask_for_dtype(4, 1, 1, &Device::Cpu, DType::F32).unwrap();
        let values = mask
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        assert_eq!(mask.dtype(), DType::F32);
        assert_eq!(values[0][0], 0.0);
        assert!(values[0][2] < -1.0e8);
    }

    #[test]
    fn typed_rel_positional_embedding_matches_f32_builder_shape() {
        let pos = build_rel_positional_embedding_for_dtype(3, 8, &Device::Cpu, DType::F32).unwrap();

        assert_eq!(pos.dtype(), DType::F32);
        assert_eq!(pos.dims(), &[1, 5, 8]);
    }

    #[test]
    fn normalize_mode_respects_nemo_na_value() {
        assert_eq!(
            FeatureNormalize::from_config(Some("NA")),
            FeatureNormalize::None
        );
        assert_eq!(
            FeatureNormalize::from_config(Some("per_feature")),
            FeatureNormalize::PerFeature
        );
    }

    #[test]
    fn argmax_1d_selects_max_from_rank1_logits() {
        let logits =
            Tensor::from_vec(vec![0.1f32, 3.2, -4.0, 2.7], 4, &Device::Cpu).expect("logits");

        assert_eq!(argmax_1d(&logits).expect("argmax"), 1);
    }

    #[test]
    fn argmax_1d_rejects_non_vector_logits() {
        let logits =
            Tensor::from_vec(vec![0.1f32; 8], (2, 4), &Device::Cpu).expect("batched logits");
        let err = argmax_1d(&logits).expect_err("rank-2 logits should fail");

        assert!(err.to_string().contains("expected rank-1 logits"));
    }

    #[test]
    fn prompt_one_hot_builder_sets_only_requested_prompt_id() {
        let prompt = build_prompt_one_hot(3, &Device::Cpu, DType::F32).unwrap();
        let values = prompt.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        assert_eq!(prompt.dims(), &[1, 1, PROMPT_DIM]);
        assert_eq!(values[3], 1.0);
        assert_eq!(values.iter().filter(|value| **value == 1.0).count(), 1);
    }

    #[test]
    fn streaming_feature_state_waits_for_centered_frame_samples() {
        let preprocessor = test_preprocessor(FeatureNormalize::None);
        let mut state = NemotronStreamingFeatureState::new();

        state.push_samples(&vec![0.25; (N_FFT / 2) - 1]).unwrap();
        assert!(preprocessor
            .compute_streaming_features(&mut state)
            .unwrap()
            .is_none());

        state.push_samples(&[0.5]).unwrap();
        let chunk = preprocessor
            .compute_streaming_features(&mut state)
            .unwrap()
            .expect("first centered frame");

        assert_eq!(chunk.start_frame, 0);
        assert_eq!(chunk.frames, 1);
        assert_eq!(chunk.total_ready_frames, 1);
        assert!(!chunk.is_final);
    }

    #[test]
    fn streaming_features_match_full_valid_features_after_chunked_pushes() {
        let preprocessor = test_preprocessor(FeatureNormalize::None);
        let audio = (0..4_013)
            .map(|idx| ((idx as f32) * 0.017).sin() * 0.3)
            .collect::<Vec<_>>();
        let (full, valid_frames) = preprocessor.compute_features(&audio).unwrap();
        let full = full.narrow(2, 0, valid_frames).unwrap();

        let mut state = NemotronStreamingFeatureState::new();
        let mut chunks = Vec::<Tensor>::new();
        let mut offset = 0usize;
        for size in [17usize, 439, 811, 1_003, 571, 1_172] {
            let end = offset.saturating_add(size).min(audio.len());
            if end == offset {
                break;
            }
            state.push_samples(&audio[offset..end]).unwrap();
            if let Some(chunk) = preprocessor.compute_streaming_features(&mut state).unwrap() {
                chunks.push(chunk.features);
            }
            offset = end;
        }
        if offset < audio.len() {
            state.push_samples(&audio[offset..]).unwrap();
            if let Some(chunk) = preprocessor.compute_streaming_features(&mut state).unwrap() {
                chunks.push(chunk.features);
            }
        }
        state.finish_input();
        if let Some(chunk) = preprocessor.compute_streaming_features(&mut state).unwrap() {
            assert!(chunk.is_final);
            chunks.push(chunk.features);
        }

        let refs = chunks.iter().collect::<Vec<_>>();
        let streamed = Tensor::cat(&refs, 2).unwrap();
        assert_tensor_close(&streamed, &full, 1e-4);
    }

    #[test]
    fn streaming_feature_state_rejects_push_after_finish() {
        let mut state = NemotronStreamingFeatureState::new();
        state.finish_input();

        let err = state.push_samples(&[0.0]).unwrap_err();
        assert!(err.to_string().contains("finalized"));
    }

    #[test]
    fn streaming_pre_encode_delays_tail_and_matches_full_output() {
        let subsampling = zero_subsampling();
        let device = Device::Cpu;
        let feature_frames = 41usize;
        let values = (0..(N_MELS * feature_frames))
            .map(|idx| (idx as f32) / 1000.0)
            .collect::<Vec<_>>();
        let features = Tensor::from_vec(values, (1, N_MELS, feature_frames), &device).unwrap();
        let (full, encoded_len) = subsampling.forward(&features, feature_frames).unwrap();

        let mut state = NemotronStreamingPreEncodeState::new();
        state
            .push_features(feature_chunk(&features, 0, 7, false))
            .unwrap();
        assert!(subsampling
            .forward_streaming_chunk(&mut state)
            .unwrap()
            .is_none());

        state
            .push_features(feature_chunk(&features, 7, 1, false))
            .unwrap();
        let first = subsampling
            .forward_streaming_chunk(&mut state)
            .unwrap()
            .expect("first stable encoded frame");
        assert_eq!(first.start_frame, 0);
        assert_eq!(first.frames, 1);
        assert_eq!(first.total_stable_frames, 1);
        assert!(!first.is_final);

        state
            .push_features(feature_chunk(&features, 8, 24, false))
            .unwrap();
        let middle = subsampling
            .forward_streaming_chunk(&mut state)
            .unwrap()
            .expect("middle stable frames");
        assert_eq!(middle.start_frame, 1);
        assert_eq!(middle.total_stable_frames, 4);

        state
            .push_features(feature_chunk(&features, 32, 9, false))
            .unwrap();
        state.finish_input();
        let tail = subsampling
            .forward_streaming_chunk(&mut state)
            .unwrap()
            .expect("final tail frames");
        assert!(tail.is_final);
        assert_eq!(tail.total_stable_frames, encoded_len);

        let streamed = Tensor::cat(&[&first.encoded, &middle.encoded, &tail.encoded], 1).unwrap();
        let full = full.narrow(1, 0, encoded_len).unwrap();
        assert_tensor_close(&streamed, &full, 0.0);
    }

    #[test]
    fn streaming_pre_encode_rejects_out_of_order_features() {
        let features = Tensor::zeros((1, N_MELS, 2), DType::F32, &Device::Cpu).unwrap();
        let mut state = NemotronStreamingPreEncodeState::new();

        let err = state
            .push_features(feature_chunk(&features, 1, 1, false))
            .unwrap_err();
        assert!(err.to_string().contains("expected start frame 0"));
    }

    #[test]
    fn joint_projection_helpers_match_uncached_joint_logits() {
        let device = Device::Cpu;
        let joint = Joint {
            pred: Linear::new(
                Tensor::ones((JOINT_HIDDEN, PRED_HIDDEN), DType::F32, &device).unwrap(),
                Some(Tensor::zeros(JOINT_HIDDEN, DType::F32, &device).unwrap()),
            ),
            enc: Linear::new(
                Tensor::ones((JOINT_HIDDEN, ENCODER_DIM), DType::F32, &device).unwrap(),
                Some(Tensor::zeros(JOINT_HIDDEN, DType::F32, &device).unwrap()),
            ),
            out: Linear::new(
                Tensor::ones((4, JOINT_HIDDEN), DType::F32, &device).unwrap(),
                Some(Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], 4, &device).unwrap()),
            ),
        };
        let enc = Tensor::ones((1, 1, ENCODER_DIM), DType::F32, &device).unwrap();
        let pred = Tensor::ones((1, 1, PRED_HIDDEN), DType::F32, &device).unwrap();

        let uncached = joint
            .joint_after_projection(&enc, &pred)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cached = joint
            .joint_from_projections(
                &joint.project_encoder(&enc).unwrap(),
                &joint.project_predictor(&pred).unwrap(),
            )
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(cached, uncached);
    }
}
