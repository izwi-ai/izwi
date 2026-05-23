//! Flow-matching acoustic-token shape helpers.

use candle_core::{DType, IndexOp, Tensor, D};
use candle_nn::{ops, Linear, Module, RmsNorm, VarBuilder};

use crate::error::{Error, Result};
use crate::models::architectures::qwen3::core::repeat_kv;

use super::super::layers::linear_forward_last_dim;
use super::config::{
    VoxtralTtsAcousticTransformerArgs, VoxtralTtsConfig, DEFAULT_CFG_ALPHA,
    DEFAULT_N_DECODING_STEPS,
};

pub const AUDIO_SPECIAL_TOKEN_COUNT: u32 = 2;
pub const ACOUSTIC_CODEBOOK_OFFSET: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioSpecialToken {
    Empty,
    End,
}

impl AudioSpecialToken {
    pub fn id(self) -> u32 {
        match self {
            Self::Empty => 0,
            Self::End => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodeValue {
    Empty,
    End,
    Code(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcousticCodeFrame {
    pub semantic: u32,
    pub acoustic: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticGenerationConfig {
    pub semantic_codebook_size: usize,
    pub acoustic_codebook_size: usize,
    pub n_acoustic_codebooks: usize,
    pub num_codebooks: usize,
    pub n_decoding_steps: usize,
    pub cfg_alpha: f32,
    pub sigma: f32,
}

pub struct FlowMatchingAudioTransformer {
    input_projection: Linear,
    time_projection: Linear,
    llm_projection: Linear,
    semantic_codebook_output: Linear,
    acoustic_codebook_output: Linear,
    layers: Vec<AcousticTransformerBlock>,
    norm: RmsNorm,
    time_embedding: TimeEmbedding,
    args: VoxtralTtsAcousticTransformerArgs,
    generation: AcousticGenerationConfig,
}

struct AcousticTransformerBlock {
    attention: BidirectionalAttention,
    feed_forward: AcousticFeedForward,
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

struct BidirectionalAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

struct AcousticFeedForward {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

struct TimeEmbedding {
    inv_freq: Tensor,
}

impl AcousticGenerationConfig {
    pub fn from_config(config: &VoxtralTtsConfig) -> Self {
        let transformer = &config.multimodal.audio_model_args.acoustic_transformer_args;
        Self {
            semantic_codebook_size: config.semantic_codebook_size(),
            acoustic_codebook_size: config.acoustic_codebook_size(),
            n_acoustic_codebooks: config.n_acoustic_codebooks(),
            num_codebooks: config.num_codebooks(),
            n_decoding_steps: transformer
                .n_decoding_steps
                .unwrap_or(DEFAULT_N_DECODING_STEPS),
            cfg_alpha: transformer.cfg_alpha.unwrap_or(DEFAULT_CFG_ALPHA),
            sigma: transformer.sigma,
        }
    }

    pub fn validate_frame(&self, frame: &AcousticCodeFrame) -> Result<()> {
        if frame.semantic as usize >= self.semantic_codebook_size {
            return Err(Error::InferenceError(format!(
                "Voxtral semantic code {} exceeds codebook size {}",
                frame.semantic, self.semantic_codebook_size
            )));
        }
        if frame.acoustic.len() != self.n_acoustic_codebooks {
            return Err(Error::InferenceError(format!(
                "Voxtral acoustic frame has {} codebooks, expected {}",
                frame.acoustic.len(),
                self.n_acoustic_codebooks
            )));
        }
        for (idx, code) in frame.acoustic.iter().enumerate() {
            if *code as usize >= self.acoustic_codebook_size {
                return Err(Error::InferenceError(format!(
                    "Voxtral acoustic codebook {idx} value {code} exceeds codebook size {}",
                    self.acoustic_codebook_size
                )));
            }
        }
        Ok(())
    }
}

impl FlowMatchingAudioTransformer {
    pub fn load(config: &VoxtralTtsConfig, vb: VarBuilder) -> Result<Self> {
        let generation = AcousticGenerationConfig::from_config(config);
        let args = config
            .multimodal
            .audio_model_args
            .acoustic_transformer_args
            .clone();
        let input_projection = candle_nn::linear_no_bias(
            generation.n_acoustic_codebooks,
            args.dim,
            vb.pp("input_projection"),
        )?;
        let time_projection =
            candle_nn::linear_no_bias(args.dim, args.dim, vb.pp("time_projection"))?;
        let llm_projection =
            candle_nn::linear_no_bias(args.input_dim, args.dim, vb.pp("llm_projection"))?;
        let semantic_output_size =
            padded_codebook_size(generation.semantic_codebook_size, true, 128);
        let semantic_codebook_output = linear_maybe_bias(
            args.dim,
            semantic_output_size,
            args.use_biases,
            vb.pp("semantic_codebook_output"),
        )?;
        let acoustic_codebook_output = candle_nn::linear_no_bias(
            args.dim,
            generation.n_acoustic_codebooks,
            vb.pp("acoustic_codebook_output"),
        )?;
        let mut layers = Vec::with_capacity(args.n_layers);
        for layer_idx in 0..args.n_layers {
            layers.push(AcousticTransformerBlock::load(
                &args,
                vb.pp(format!("layers.{layer_idx}")),
            )?);
        }
        let norm = candle_nn::rms_norm(args.dim, args.norm_eps, vb.pp("norm"))?;
        let time_embedding = TimeEmbedding::new(args.dim, args.rope_theta, vb.device())?;
        Ok(Self {
            input_projection,
            time_projection,
            llm_projection,
            semantic_codebook_output,
            acoustic_codebook_output,
            layers,
            norm,
            time_embedding,
            args,
            generation,
        })
    }

    pub fn forward_audio_codes(
        &self,
        llm_hidden: &Tensor,
        cfg_alpha: f32,
    ) -> Result<Vec<Vec<u32>>> {
        self.forward_audio_codes_with_steps(llm_hidden, cfg_alpha, self.generation.n_decoding_steps)
    }

    pub fn forward_audio_codes_with_steps(
        &self,
        llm_hidden: &Tensor,
        cfg_alpha: f32,
        n_decoding_steps: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let llm_hidden = match llm_hidden.rank() {
            1 => llm_hidden.unsqueeze(0)?,
            2 => llm_hidden.clone(),
            rank => {
                return Err(Error::InferenceError(format!(
                    "Voxtral acoustic transformer expected rank-1 or rank-2 hidden states, got rank {rank}"
                )));
            }
        };
        if llm_hidden.dim(1)? != self.args.input_dim {
            return Err(Error::InferenceError(format!(
                "Voxtral acoustic transformer hidden size mismatch: got {}, expected {}",
                llm_hidden.dim(1)?,
                self.args.input_dim
            )));
        }

        let semantic_logits = linear_forward_last_dim(&self.semantic_codebook_output, &llm_hidden)
            .map_err(|err| Error::InferenceError(format!("semantic projection failed: {err}")))?;
        let semantic_codes =
            semantic_codes_from_logits(&semantic_logits, self.generation.semantic_codebook_size)
                .map_err(|err| Error::InferenceError(format!("semantic sampling failed: {err}")))?;
        let acoustic_codes = self
            .decode_one_frame(&semantic_codes, &llm_hidden, cfg_alpha, n_decoding_steps)
            .map_err(|err| Error::InferenceError(format!("acoustic frame decode failed: {err}")))?;
        let mut frames = Vec::with_capacity(semantic_codes.len());
        for (semantic, acoustic) in semantic_codes.into_iter().zip(acoustic_codes) {
            let mut frame = Vec::with_capacity(self.generation.num_codebooks);
            frame.push(semantic);
            frame.extend(acoustic);
            frames.push(frame);
        }
        Ok(frames)
    }

    fn decode_one_frame(
        &self,
        semantic_codes: &[u32],
        llm_hidden: &Tensor,
        cfg_alpha: f32,
        n_decoding_steps: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let batch = semantic_codes.len();
        let dtype = llm_hidden.dtype();
        let device = llm_hidden.device();
        let mut sampled = Tensor::randn(
            0.0f32,
            1.0f32,
            (batch, self.generation.n_acoustic_codebooks),
            device,
        )?
        .to_dtype(dtype)?;
        let llm_hidden_zero = Tensor::zeros(llm_hidden.shape(), dtype, device)?;
        let n_steps = n_decoding_steps.max(1);
        for step in 0..n_steps {
            let t = step as f32 / n_steps as f32;
            let dt = 1.0f64 / n_steps as f64;
            let t_values = Tensor::from_vec(vec![t; batch], (batch, 1), device)?.to_dtype(dtype)?;
            let t_emb = (|| -> Result<Tensor> {
                Ok(self.time_embedding.forward(&t_values)?.to_dtype(dtype)?)
            })()
            .map_err(|err| {
                Error::InferenceError(format!("time embedding step {step} failed: {err}"))
            })?;
            let x_batched = Tensor::cat(&[sampled.clone(), sampled.clone()], 0)?;
            let llm_batched = Tensor::cat(&[llm_hidden.clone(), llm_hidden_zero.clone()], 0)?;
            let t_emb_batched = Tensor::cat(&[t_emb.clone(), t_emb], 0)?;
            let velocity = self
                .predict_velocity(&x_batched, &llm_batched, &t_emb_batched)
                .map_err(|err| {
                    Error::InferenceError(format!("velocity prediction step {step} failed: {err}"))
                })?;
            let cond = velocity.narrow(0, 0, batch)?;
            let uncond = velocity.narrow(0, batch, batch)?;
            let blended = blend_cfg_tensors(&cond, &uncond, cfg_alpha)?;
            sampled = sampled.broadcast_add(&(blended * dt)?)?;
        }

        let sampled = sampled.clamp(-1.0, 1.0)?.to_dtype(DType::F32)?;
        let rows = sampled.to_vec2::<f32>()?;
        let mut output = Vec::with_capacity(batch);
        for (semantic, row) in semantic_codes.iter().zip(rows) {
            let should_decode = *semantic != AudioSpecialToken::End.id();
            let mut acoustic = Vec::with_capacity(row.len());
            for value in row {
                let raw = if should_decode {
                    fsq_unit_to_code(value, self.generation.acoustic_codebook_size)
                } else {
                    AudioSpecialToken::Empty.id()
                };
                acoustic.push(apply_audio_token_offset(raw));
            }
            output.push(acoustic);
        }
        Ok(output)
    }

    fn predict_velocity(
        &self,
        x_t: &Tensor,
        llm_output: &Tensor,
        t_emb: &Tensor,
    ) -> Result<Tensor> {
        let x_t = x_t.to_dtype(llm_output.dtype())?;
        let projected_x = linear_forward_last_dim(&self.input_projection, &x_t.unsqueeze(1)?)?;
        let projected_t = linear_forward_last_dim(&self.time_projection, t_emb)?.unsqueeze(1)?;
        let projected_llm =
            linear_forward_last_dim(&self.llm_projection, llm_output)?.unsqueeze(1)?;
        let mut hidden = Tensor::cat(&[projected_x, projected_t, projected_llm], 1)?;
        for (idx, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden).map_err(|err| {
                Error::InferenceError(format!(
                    "Voxtral acoustic transformer layer {idx} failed: {err}"
                ))
            })?;
        }
        let hidden = self.norm.forward(&hidden)?;
        let first = hidden.i((.., 0, ..))?;
        linear_forward_last_dim(&self.acoustic_codebook_output, &first)
    }
}

impl AcousticTransformerBlock {
    fn load(args: &VoxtralTtsAcousticTransformerArgs, vb: VarBuilder) -> Result<Self> {
        let attention = BidirectionalAttention::load(args, vb.pp("attention"))?;
        let feed_forward = AcousticFeedForward::load(args, vb.pp("feed_forward"))?;
        let attention_norm = candle_nn::rms_norm(args.dim, args.norm_eps, vb.pp("attention_norm"))?;
        let ffn_norm = candle_nn::rms_norm(args.dim, args.norm_eps, vb.pp("ffn_norm"))?;
        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let attn = self
            .attention
            .forward(&self.attention_norm.forward(x)?)
            .map_err(|err| Error::InferenceError(format!("attention failed: {err}")))?;
        let hidden = x.broadcast_add(&attn)?;
        let ff = self
            .feed_forward
            .forward(&self.ffn_norm.forward(&hidden)?)
            .map_err(|err| Error::InferenceError(format!("feed-forward failed: {err}")))?;
        hidden.broadcast_add(&ff).map_err(Error::from)
    }
}

impl BidirectionalAttention {
    fn load(args: &VoxtralTtsAcousticTransformerArgs, vb: VarBuilder) -> Result<Self> {
        let wq = linear_maybe_bias(
            args.dim,
            args.n_heads * args.head_dim,
            args.use_biases,
            vb.pp("wq"),
        )?;
        let wk = candle_nn::linear_no_bias(args.dim, args.n_kv_heads * args.head_dim, vb.pp("wk"))?;
        let wv = linear_maybe_bias(
            args.dim,
            args.n_kv_heads * args.head_dim,
            args.use_biases,
            vb.pp("wv"),
        )?;
        let wo = linear_maybe_bias(
            args.n_heads * args.head_dim,
            args.dim,
            args.use_biases,
            vb.pp("wo"),
        )?;
        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            n_heads: args.n_heads,
            n_kv_heads: args.n_kv_heads,
            head_dim: args.head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let batch = x.dim(0)?;
        let seq_len = x.dim(1)?;
        let q = linear_forward_last_dim(&self.wq, x)?.reshape((
            batch,
            seq_len,
            self.n_heads,
            self.head_dim,
        ))?;
        let k = linear_forward_last_dim(&self.wk, x)?.reshape((
            batch,
            seq_len,
            self.n_kv_heads,
            self.head_dim,
        ))?;
        let v = linear_forward_last_dim(&self.wv, x)?.reshape((
            batch,
            seq_len,
            self.n_kv_heads,
            self.head_dim,
        ))?;
        let k = repeat_kv(&k, self.n_heads, self.n_kv_heads)?;
        let v = repeat_kv(&v, self.n_heads, self.n_kv_heads)?;
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let q = q.reshape((batch * self.n_heads, seq_len, self.head_dim))?;
        let k = k.reshape((batch * self.n_heads, seq_len, self.head_dim))?;
        let v = v.reshape((batch * self.n_heads, seq_len, self.head_dim))?;
        let mut attn = q.matmul(&k.transpose(1, 2)?)?;
        let scale = Tensor::from_vec(vec![(self.head_dim as f32).sqrt()], (1,), attn.device())?
            .to_dtype(attn.dtype())?;
        attn = attn.broadcast_div(&scale)?;
        let attn = ops::softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v)?;
        let out = out
            .reshape((batch, self.n_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.n_heads * self.head_dim))?;
        linear_forward_last_dim(&self.wo, &out)
    }
}

impl AcousticFeedForward {
    fn load(args: &VoxtralTtsAcousticTransformerArgs, vb: VarBuilder) -> Result<Self> {
        let w1 = candle_nn::linear_no_bias(args.dim, args.hidden_dim, vb.pp("w1"))?;
        let w2 = linear_maybe_bias(args.hidden_dim, args.dim, args.use_biases, vb.pp("w2"))?;
        let w3 = candle_nn::linear_no_bias(args.dim, args.hidden_dim, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = linear_forward_last_dim(&self.w1, x)?;
        let up = linear_forward_last_dim(&self.w3, x)?;
        let hidden = ops::silu(&gate)?.broadcast_mul(&up)?;
        linear_forward_last_dim(&self.w2, &hidden)
    }
}

impl TimeEmbedding {
    fn new(dim: usize, theta: f64, device: &candle_core::Device) -> Result<Self> {
        let half_dim = dim / 2;
        let inv_freq = (0..half_dim)
            .map(|idx| (-theta.ln() * idx as f64 / half_dim as f64).exp() as f32)
            .collect::<Vec<_>>();
        Ok(Self {
            inv_freq: Tensor::from_vec(inv_freq, (half_dim,), device)?,
        })
    }

    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let emb = t.matmul(&self.inv_freq.unsqueeze(0)?)?;
        Tensor::cat(&[emb.cos()?, emb.sin()?], 1).map_err(Error::from)
    }
}

impl AcousticCodeFrame {
    pub fn new(
        semantic: u32,
        acoustic: Vec<u32>,
        config: &AcousticGenerationConfig,
    ) -> Result<Self> {
        let frame = Self { semantic, acoustic };
        config.validate_frame(&frame)?;
        Ok(frame)
    }

    pub fn shifted_codes(&self) -> Vec<u32> {
        let mut codes = Vec::with_capacity(self.acoustic.len() + ACOUSTIC_CODEBOOK_OFFSET);
        codes.push(apply_audio_token_offset(self.semantic));
        codes.extend(
            self.acoustic
                .iter()
                .map(|code| apply_audio_token_offset(*code)),
        );
        codes
    }
}

pub fn apply_audio_token_offset(code: u32) -> u32 {
    code + AUDIO_SPECIAL_TOKEN_COUNT
}

pub fn strip_audio_token_offset(shifted: u32) -> AudioCodeValue {
    match shifted {
        0 => AudioCodeValue::Empty,
        1 => AudioCodeValue::End,
        value => AudioCodeValue::Code(value - AUDIO_SPECIAL_TOKEN_COUNT),
    }
}

pub fn fsq_unit_to_code(value: f32, codebook_size: usize) -> u32 {
    if codebook_size <= 1 {
        return 0;
    }
    let clamped = value.clamp(-1.0, 1.0);
    let scaled = (clamped + 1.0) * 0.5 * (codebook_size as f32 - 1.0);
    scaled.round() as u32
}

pub fn fsq_code_to_unit(code: u32, codebook_size: usize) -> f32 {
    if codebook_size <= 1 {
        return 0.0;
    }
    let clamped = code.min(codebook_size as u32 - 1) as f32;
    (clamped / (codebook_size as f32 - 1.0)) * 2.0 - 1.0
}

pub fn cfg_velocity_blend(conditional: f32, unconditional: f32, alpha: f32) -> f32 {
    alpha.mul_add(conditional, (1.0 - alpha) * unconditional)
}

pub fn padded_codebook_size(
    codebook_size: usize,
    include_special_tokens: bool,
    pad_to_multiple: usize,
) -> usize {
    let mut size = codebook_size;
    if include_special_tokens {
        size += AUDIO_SPECIAL_TOKEN_COUNT as usize;
    }
    if pad_to_multiple > 0 {
        size = pad_to_multiple * size.div_ceil(pad_to_multiple);
    }
    size
}

fn linear_maybe_bias(
    in_dim: usize,
    out_dim: usize,
    use_bias: bool,
    vb: VarBuilder,
) -> Result<Linear> {
    if use_bias {
        candle_nn::linear(in_dim, out_dim, vb).map_err(Error::from)
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb).map_err(Error::from)
    }
}

fn semantic_codes_from_logits(logits: &Tensor, semantic_codebook_size: usize) -> Result<Vec<u32>> {
    let logits = match logits.rank() {
        1 => logits.unsqueeze(0)?,
        2 => logits.clone(),
        rank => {
            return Err(Error::InferenceError(format!(
                "Voxtral semantic logits expected rank 1 or 2, got rank {rank}"
            )));
        }
    };
    let allowed = (semantic_codebook_size + AUDIO_SPECIAL_TOKEN_COUNT as usize).min(logits.dim(1)?);
    let rows = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    rows.into_iter()
        .map(|row| {
            let mut best_idx = AudioSpecialToken::End.id() as usize;
            let mut best_value = f32::NEG_INFINITY;
            for (idx, value) in row.iter().take(allowed).enumerate() {
                if idx == AudioSpecialToken::Empty.id() as usize {
                    continue;
                }
                if *value > best_value {
                    best_idx = idx;
                    best_value = *value;
                }
            }
            Ok(best_idx as u32)
        })
        .collect()
}

fn blend_cfg_tensors(cond: &Tensor, uncond: &Tensor, alpha: f32) -> Result<Tensor> {
    let cond_scaled = (cond * alpha as f64)?;
    let uncond_scaled = (uncond * (1.0f64 - alpha as f64))?;
    cond_scaled
        .broadcast_add(&uncond_scaled)
        .map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::architectures::voxtral::tts::config::{fixture_json, VoxtralTtsConfig};
    use candle_core::{Device, Shape};
    use candle_nn::VarBuilder;
    use serde_json::json;
    use std::collections::HashMap;

    #[test]
    fn derives_generation_config_from_params() {
        let config = VoxtralTtsConfig::from_json_str(fixture_json()).unwrap();
        let acoustic = AcousticGenerationConfig::from_config(&config);
        assert_eq!(acoustic.semantic_codebook_size, 8192);
        assert_eq!(acoustic.acoustic_codebook_size, 21);
        assert_eq!(acoustic.n_acoustic_codebooks, 36);
        assert_eq!(acoustic.num_codebooks, 37);
        assert_eq!(acoustic.n_decoding_steps, 7);
        assert_eq!(acoustic.cfg_alpha, 1.2);
    }

    #[test]
    fn applies_and_strips_vllm_omni_audio_token_offsets() {
        assert_eq!(apply_audio_token_offset(0), 2);
        assert_eq!(strip_audio_token_offset(0), AudioCodeValue::Empty);
        assert_eq!(strip_audio_token_offset(1), AudioCodeValue::End);
        assert_eq!(strip_audio_token_offset(8193), AudioCodeValue::Code(8191));
    }

    #[test]
    fn quantizes_fsq_unit_range_to_21_levels() {
        assert_eq!(fsq_unit_to_code(-2.0, 21), 0);
        assert_eq!(fsq_unit_to_code(-1.0, 21), 0);
        assert_eq!(fsq_unit_to_code(0.0, 21), 10);
        assert_eq!(fsq_unit_to_code(1.0, 21), 20);
        assert_eq!(fsq_unit_to_code(2.0, 21), 20);
        assert!((fsq_code_to_unit(10, 21) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn validates_code_frame_shapes() {
        let config = VoxtralTtsConfig::from_json_str(fixture_json()).unwrap();
        let acoustic = AcousticGenerationConfig::from_config(&config);
        let frame = AcousticCodeFrame::new(7, vec![3; 36], &acoustic).unwrap();
        assert_eq!(frame.shifted_codes().len(), 37);
        assert!(AcousticCodeFrame::new(8192, vec![3; 36], &acoustic).is_err());
        assert!(AcousticCodeFrame::new(7, vec![3; 35], &acoustic).is_err());
        assert!(AcousticCodeFrame::new(7, vec![21; 36], &acoustic).is_err());
    }

    #[test]
    fn blends_classifier_free_guidance_velocity() {
        assert!((cfg_velocity_blend(2.0, 0.5, 1.2) - 2.3).abs() < 1e-6);
    }

    #[test]
    fn pads_codebook_sizes_like_vllm_omni() {
        assert_eq!(padded_codebook_size(8192, true, 128), 8320);
        assert_eq!(padded_codebook_size(21, true, 128), 128);
        assert_eq!(padded_codebook_size(21, false, 0), 21);
    }

    #[test]
    fn acoustic_linear_flattens_rank3_inputs_over_last_dim() {
        let device = Device::Cpu;
        let linear = Linear::new(Tensor::ones((4, 2), DType::F32, &device).unwrap(), None);
        let input = Tensor::ones((3, 2), DType::F32, &device)
            .unwrap()
            .unsqueeze(1)
            .unwrap();

        let output = linear_forward_last_dim(&linear, &input).unwrap();

        assert_eq!(output.dims(), &[3, 1, 4]);
        assert_eq!(
            output.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![2.0; 12]
        );
    }

    #[test]
    fn semantic_logits_mask_empty_and_out_of_range_codes() {
        let device = Device::Cpu;
        let logits =
            Tensor::from_vec(vec![100.0, 0.0, 1.0, 2.0, 99.0, 98.0], (1, 6), &device).unwrap();
        let codes = semantic_codes_from_logits(&logits, 2).unwrap();
        assert_eq!(codes, vec![3]);
    }

    #[test]
    fn tiny_acoustic_transformer_loads_and_outputs_shifted_codes() {
        let device = Device::Cpu;
        let config = tiny_acoustic_config();
        let vb = VarBuilder::from_tensors(tiny_acoustic_tensors(&device), DType::F32, &device);
        let transformer = FlowMatchingAudioTransformer::load(&config, vb).unwrap();
        let hidden = Tensor::zeros((1, 4), DType::F32, &device).unwrap();
        let codes = transformer.forward_audio_codes(&hidden, 1.2).unwrap();
        assert_eq!(codes.len(), 1);
        assert_eq!(codes[0].len(), 3);
        assert_eq!(codes[0][0], AudioSpecialToken::End.id());
        assert!(codes[0][1] >= AUDIO_SPECIAL_TOKEN_COUNT);
        assert!(codes[0][2] >= AUDIO_SPECIAL_TOKEN_COUNT);
    }

    fn tiny_acoustic_config() -> VoxtralTtsConfig {
        let mut value: serde_json::Value = serde_json::from_str(fixture_json()).unwrap();
        value["dim"] = json!(4);
        value["hidden_dim"] = json!(8);
        value["n_heads"] = json!(2);
        value["n_kv_heads"] = json!(1);
        value["head_dim"] = json!(2);
        let audio = &mut value["multimodal"]["audio_model_args"];
        audio["semantic_codebook_size"] = json!(4);
        audio["acoustic_codebook_size"] = json!(3);
        audio["n_acoustic_codebook"] = json!(2);
        audio["audio_encoding_args"]["num_codebooks"] = json!(3);
        let transformer = &mut audio["acoustic_transformer_args"];
        transformer["input_dim"] = json!(4);
        transformer["dim"] = json!(4);
        transformer["n_layers"] = json!(1);
        transformer["head_dim"] = json!(2);
        transformer["hidden_dim"] = json!(8);
        transformer["n_heads"] = json!(2);
        transformer["n_kv_heads"] = json!(1);
        transformer["norm_eps"] = json!(1e-5);
        transformer["n_decoding_steps"] = json!(1);
        value["multimodal"]["audio_tokenizer_args"]["semantic_codebook_size"] = json!(4);
        value["multimodal"]["audio_tokenizer_args"]["acoustic_codebook_size"] = json!(3);
        VoxtralTtsConfig::from_json_str(&value.to_string()).unwrap()
    }

    fn tiny_acoustic_tensors(device: &Device) -> HashMap<String, Tensor> {
        fn zeros(shape: impl Into<Shape>, device: &Device) -> Tensor {
            Tensor::zeros(shape, DType::F32, device).unwrap()
        }
        HashMap::from([
            ("input_projection.weight".to_string(), zeros((4, 2), device)),
            ("time_projection.weight".to_string(), zeros((4, 4), device)),
            ("llm_projection.weight".to_string(), zeros((4, 4), device)),
            (
                "semantic_codebook_output.weight".to_string(),
                zeros((128, 4), device),
            ),
            (
                "acoustic_codebook_output.weight".to_string(),
                zeros((2, 4), device),
            ),
            (
                "layers.0.attention.wq.weight".to_string(),
                zeros((4, 4), device),
            ),
            (
                "layers.0.attention.wk.weight".to_string(),
                zeros((2, 4), device),
            ),
            (
                "layers.0.attention.wv.weight".to_string(),
                zeros((2, 4), device),
            ),
            (
                "layers.0.attention.wo.weight".to_string(),
                zeros((4, 4), device),
            ),
            (
                "layers.0.feed_forward.w1.weight".to_string(),
                zeros((8, 4), device),
            ),
            (
                "layers.0.feed_forward.w2.weight".to_string(),
                zeros((4, 8), device),
            ),
            (
                "layers.0.feed_forward.w3.weight".to_string(),
                zeros((8, 4), device),
            ),
            (
                "layers.0.attention_norm.weight".to_string(),
                Tensor::ones((4,), DType::F32, device).unwrap(),
            ),
            (
                "layers.0.ffn_norm.weight".to_string(),
                Tensor::ones((4,), DType::F32, device).unwrap(),
            ),
            (
                "norm.weight".to_string(),
                Tensor::ones((4,), DType::F32, device).unwrap(),
            ),
        ])
    }
}
