//! Flow-matching acoustic-token shape helpers.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use candle_core::{DType, IndexOp, Tensor, D};
use candle_nn::{ops, Linear, Module, RmsNorm, VarBuilder};

use crate::error::{Error, Result};
use crate::models::shared::attention::flash::try_fused_self_attention;
use crate::models::shared::attention::gqa::compact_gqa_sdpa_bhsd;

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

#[derive(Debug)]
pub struct VoxtralGeneratedAudioCodes {
    pub frames: Vec<Vec<u32>>,
    pub shifted_code_tensor: Option<Tensor>,
}

#[derive(Debug)]
struct SemanticCodeSelection {
    codes: Vec<u32>,
    code_tensor: Option<Tensor>,
}

#[derive(Debug)]
struct DecodedAcousticCodes {
    shifted_codes: Vec<Vec<u32>>,
    shifted_code_tensor: Option<Tensor>,
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
    /// `time_projection(time_embedding(step / n_steps))` for every step of
    /// an `n_steps` schedule, per (n_steps, dtype, device).
    time_tables: Mutex<Vec<AcousticTimeTable>>,
}

struct AcousticTimeTable {
    n_steps: usize,
    dtype: DType,
    device: candle_core::DeviceLocation,
    /// `[n_steps, dim]`.
    projected: Tensor,
}

static VOXTRAL_ACOUSTIC_FRAME_READBACKS: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
thread_local! {
    static THREAD_FRAME_READBACKS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Process-wide count of device-to-host copies made while selecting frame
/// codes (semantic and acoustic). The packed CUDA path makes one per frame.
pub fn voxtral_acoustic_frame_readbacks() -> u64 {
    VOXTRAL_ACOUSTIC_FRAME_READBACKS.load(Ordering::Relaxed)
}

#[cfg(test)]
fn thread_frame_readbacks() -> u64 {
    THREAD_FRAME_READBACKS.with(|reads| reads.get())
}

fn record_frame_readback() {
    VOXTRAL_ACOUSTIC_FRAME_READBACKS.fetch_add(1, Ordering::Relaxed);
    #[cfg(test)]
    THREAD_FRAME_READBACKS.with(|reads| reads.set(reads.get() + 1));
}

static VOXTRAL_ACOUSTIC_STEP_UPLOADS: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
thread_local! {
    static THREAD_ACOUSTIC_STEP_UPLOADS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Process-wide count of host uploads of flow-matching step times. The
/// projected time table is built once per decoding schedule; before, every
/// step of every frame uploaded its `t` and re-ran the time embedding and
/// projection.
pub fn voxtral_acoustic_step_uploads() -> u64 {
    VOXTRAL_ACOUSTIC_STEP_UPLOADS.load(Ordering::Relaxed)
}

#[cfg(test)]
fn thread_acoustic_step_uploads() -> u64 {
    THREAD_ACOUSTIC_STEP_UPLOADS.with(|uploads| uploads.get())
}

fn record_acoustic_step_upload() {
    VOXTRAL_ACOUSTIC_STEP_UPLOADS.fetch_add(1, Ordering::Relaxed);
    #[cfg(test)]
    THREAD_ACOUSTIC_STEP_UPLOADS.with(|uploads| uploads.set(uploads.get() + 1));
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
        let time_embedding = TimeEmbedding::new(args.dim, TIME_EMBEDDING_THETA, vb.device())?;
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
            time_tables: Mutex::new(Vec::new()),
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
        self.forward_audio_codes_with_steps_and_eos_policy(
            llm_hidden,
            cfg_alpha,
            n_decoding_steps,
            true,
            None,
        )
    }

    pub fn forward_audio_codes_with_steps_and_eos_policy(
        &self,
        llm_hidden: &Tensor,
        cfg_alpha: f32,
        n_decoding_steps: usize,
        allow_end_audio: bool,
        initial_noise: Option<&Tensor>,
    ) -> Result<Vec<Vec<u32>>> {
        Ok(self
            .forward_audio_codes_with_feedback_tensor(
                llm_hidden,
                cfg_alpha,
                n_decoding_steps,
                allow_end_audio,
                initial_noise,
            )?
            .frames)
    }

    pub fn forward_audio_codes_with_feedback_tensor(
        &self,
        llm_hidden: &Tensor,
        cfg_alpha: f32,
        n_decoding_steps: usize,
        allow_end_audio: bool,
        initial_noise: Option<&Tensor>,
    ) -> Result<VoxtralGeneratedAudioCodes> {
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
        if llm_hidden.device().is_cuda() {
            return self
                .packed_frame(
                    &llm_hidden,
                    &semantic_logits,
                    cfg_alpha,
                    n_decoding_steps,
                    allow_end_audio,
                    initial_noise,
                )
                .map_err(|err| {
                    Error::InferenceError(format!("acoustic frame decode failed: {err}"))
                });
        }
        let semantic_selection = semantic_code_selection_from_logits_with_eos_policy(
            &semantic_logits,
            self.generation.semantic_codebook_size,
            allow_end_audio,
        )
        .map_err(|err| Error::InferenceError(format!("semantic sampling failed: {err}")))?;
        let acoustic_codes = self
            .decode_one_frame(
                &semantic_selection.codes,
                &llm_hidden,
                cfg_alpha,
                n_decoding_steps,
                initial_noise,
            )
            .map_err(|err| Error::InferenceError(format!("acoustic frame decode failed: {err}")))?;
        let mut frames = Vec::with_capacity(semantic_selection.codes.len());
        for (semantic, acoustic) in semantic_selection
            .codes
            .iter()
            .copied()
            .zip(acoustic_codes.shifted_codes)
        {
            let mut frame = Vec::with_capacity(self.generation.num_codebooks);
            frame.push(semantic);
            frame.extend(acoustic);
            frames.push(frame);
        }
        let shifted_code_tensor = match (
            semantic_selection.code_tensor,
            acoustic_codes.shifted_code_tensor,
        ) {
            (Some(semantic), Some(acoustic)) => Some(Tensor::cat(&[semantic, acoustic], 1)?),
            _ => None,
        };
        Ok(VoxtralGeneratedAudioCodes {
            frames,
            shifted_code_tensor,
        })
    }

    fn decode_one_frame(
        &self,
        semantic_codes: &[u32],
        llm_hidden: &Tensor,
        cfg_alpha: f32,
        n_decoding_steps: usize,
        initial_noise: Option<&Tensor>,
    ) -> Result<DecodedAcousticCodes> {
        let sampled = self.initial_flow_state(semantic_codes.len(), llm_hidden, initial_noise)?;
        let sampled = self.integrate_flow(sampled, llm_hidden, cfg_alpha, n_decoding_steps)?;
        shifted_acoustic_codes_from_sampled_tensor(
            &sampled,
            semantic_codes,
            self.generation.acoustic_codebook_size,
        )
    }

    /// The flow's starting state for `batch` rows: `initial_noise` when given
    /// (it must be `[batch, n_acoustic_codebooks]`), else device `randn`.
    fn initial_flow_state(
        &self,
        batch: usize,
        llm_hidden: &Tensor,
        initial_noise: Option<&Tensor>,
    ) -> Result<Tensor> {
        let dtype = llm_hidden.dtype();
        let device = llm_hidden.device();
        let shape = (batch, self.generation.n_acoustic_codebooks);
        match initial_noise {
            Some(noise) if noise.dims() == [shape.0, shape.1] => {
                Ok(noise.to_device(device)?.to_dtype(dtype)?)
            }
            Some(noise) => Err(Error::InvalidInput(format!(
                "Voxtral flow-matching noise {:?} does not match {:?}",
                noise.dims(),
                shape
            ))),
            None => Ok(Tensor::randn(0.0f32, 1.0f32, shape, device)?.to_dtype(dtype)?),
        }
    }

    /// One frame with a single device readback (used on CUDA): the semantic
    /// argmax and its score stay on the device while the flow runs (it does
    /// not depend on them), then the argmax, the score and the quantized
    /// acoustic codes come back in one packed `[rows, 2 + n_acoustic]` F32
    /// copy. The per-path code took three copies per frame. A non-finite
    /// selected score falls back to the host semantic selection, as before.
    fn packed_frame(
        &self,
        llm_hidden: &Tensor,
        semantic_logits: &Tensor,
        cfg_alpha: f32,
        n_decoding_steps: usize,
        allow_end_audio: bool,
        initial_noise: Option<&Tensor>,
    ) -> Result<VoxtralGeneratedAudioCodes> {
        let batch = llm_hidden.dim(0)?;
        let allowed = (self.generation.semantic_codebook_size + AUDIO_SPECIAL_TOKEN_COUNT as usize)
            .min(semantic_logits.dim(1)?);
        let (indices, scores) = semantic_argmax_tensors(semantic_logits, allowed, allow_end_audio)?;
        let sampled = self.initial_flow_state(batch, llm_hidden, initial_noise)?;
        let sampled = self.integrate_flow(sampled, llm_hidden, cfg_alpha, n_decoding_steps)?;
        let acoustic =
            shifted_acoustic_code_tensor(&sampled, self.generation.acoustic_codebook_size)?;
        let packed = Tensor::cat(
            &[
                indices.to_dtype(DType::F32)?.unsqueeze(1)?,
                scores,
                acoustic.to_dtype(DType::F32)?,
            ],
            1,
        )?;
        record_frame_readback();
        let rows = packed.to_vec2::<f32>()?;
        let semantic = if rows.iter().all(|row| row[1].is_finite()) {
            SemanticCodeSelection {
                codes: rows.iter().map(|row| row[0] as u32).collect(),
                code_tensor: Some(indices.unsqueeze(1)?),
            }
        } else {
            semantic_code_selection_from_logits_host(semantic_logits, allowed, allow_end_audio)?
        };
        let end = AudioSpecialToken::End.id();
        let empty = apply_audio_token_offset(AudioSpecialToken::Empty.id());
        let contains_end = semantic.codes.contains(&end);
        let frames = rows
            .iter()
            .zip(&semantic.codes)
            .map(|(row, &code)| {
                let mut frame = Vec::with_capacity(self.generation.num_codebooks);
                frame.push(code);
                if code == end {
                    frame.extend(std::iter::repeat_n(empty, row.len() - 2));
                } else {
                    frame.extend(row[2..].iter().map(|value| *value as u32));
                }
                frame
            })
            .collect();
        let shifted_code_tensor = match (semantic.code_tensor, contains_end) {
            (Some(semantic), false) => Some(Tensor::cat(&[semantic, acoustic], 1)?),
            _ => None,
        };
        Ok(VoxtralGeneratedAudioCodes {
            frames,
            shifted_code_tensor,
        })
    }

    /// The load-time precision probe for one frame: the semantic logits and
    /// the integrated acoustic state from `llm_hidden` (`[rows, input_dim]`)
    /// and `noise` (`[rows, n_acoustic_codebooks]`), at the model's default
    /// guidance and step count.
    pub(crate) fn numeric_probe(
        &self,
        llm_hidden: &Tensor,
        noise: &Tensor,
    ) -> Result<Vec<(String, Tensor)>> {
        let semantic = linear_forward_last_dim(&self.semantic_codebook_output, llm_hidden)?;
        let state = self.integrate_flow(
            noise.to_dtype(llm_hidden.dtype())?,
            llm_hidden,
            self.generation.cfg_alpha,
            self.generation.n_decoding_steps,
        )?;
        Ok(vec![
            ("acoustic_semantic_logits".to_string(), semantic),
            ("acoustic_state".to_string(), state),
        ])
    }

    /// Euler-integrates the classifier-free-guided flow from `sampled` over
    /// `n_decoding_steps` steps and returns the continuous acoustic state.
    fn integrate_flow(
        &self,
        mut sampled: Tensor,
        llm_hidden: &Tensor,
        cfg_alpha: f32,
        n_decoding_steps: usize,
    ) -> Result<Tensor> {
        let batch = llm_hidden.dim(0)?;
        let dtype = llm_hidden.dtype();
        let device = llm_hidden.device();
        let n_steps = n_decoding_steps.max(1);
        // The conditional and unconditional (zero) LLM rows and the step
        // times are the same for every step: project them once.
        let projected_llm = {
            let llm_hidden_zero = Tensor::zeros(llm_hidden.shape(), dtype, device)?;
            let llm_batched = Tensor::cat(&[llm_hidden.clone(), llm_hidden_zero], 0)?;
            linear_forward_last_dim(&self.llm_projection, &llm_batched)?.unsqueeze(1)?
        };
        let time_table = self
            .projected_time_table(n_steps, dtype, device)
            .map_err(|err| Error::InferenceError(format!("time embedding failed: {err}")))?;
        let dt = 1.0f64 / n_steps as f64;
        for step in 0..n_steps {
            let projected_t = time_table.narrow(0, step, 1)?.unsqueeze(1)?.broadcast_as((
                2 * batch,
                1,
                self.args.dim,
            ))?;
            let x_batched = Tensor::cat(&[sampled.clone(), sampled.clone()], 0)?;
            let velocity = self
                .predict_velocity(&x_batched, &projected_t, &projected_llm)
                .map_err(|err| {
                    Error::InferenceError(format!("velocity prediction step {step} failed: {err}"))
                })?;
            let cond = velocity.narrow(0, 0, batch)?;
            let uncond = velocity.narrow(0, batch, batch)?;
            let blended = blend_cfg_tensors(&cond, &uncond, cfg_alpha)?;
            sampled = sampled.broadcast_add(&(blended * dt)?)?;
        }

        Ok(sampled)
    }

    /// `time_projection(time_embedding(t))` for `t = step / n_steps`, one row
    /// per step, built on first use and cached for the model's lifetime.
    fn projected_time_table(
        &self,
        n_steps: usize,
        dtype: DType,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        let mut tables = self
            .time_tables
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(table) = tables.iter().find(|table| {
            table.n_steps == n_steps && table.dtype == dtype && table.device == device.location()
        }) {
            return Ok(table.projected.clone());
        }
        let times = (0..n_steps)
            .map(|step| step as f32 / n_steps as f32)
            .collect::<Vec<_>>();
        record_acoustic_step_upload();
        // `t` rounds through the activation dtype exactly as a per-step
        // `[batch, 1]` upload did.
        let t_values = Tensor::from_vec(times, (n_steps, 1), device)?.to_dtype(dtype)?;
        let t_emb = self.time_embedding.forward(&t_values)?.to_dtype(dtype)?;
        let projected = linear_forward_last_dim(&self.time_projection, &t_emb)?;
        tables.push(AcousticTimeTable {
            n_steps,
            dtype,
            device: device.location(),
            projected: projected.clone(),
        });
        Ok(projected)
    }

    /// One flow-matching velocity for `x_t` given the already projected time
    /// (`[rows, 1, dim]`) and LLM (`[rows, 1, dim]`) inputs.
    fn predict_velocity(
        &self,
        x_t: &Tensor,
        projected_t: &Tensor,
        projected_llm: &Tensor,
    ) -> Result<Tensor> {
        let x_t = x_t.to_dtype(projected_llm.dtype())?;
        let projected_x = linear_forward_last_dim(&self.input_projection, &x_t.unsqueeze(1)?)?;
        let mut hidden = Tensor::cat(&[&projected_x, projected_t, projected_llm], 1)?;
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
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        if let Some(out) = try_fused_self_attention(&q, &k, &v, None, self.head_dim, false)? {
            let out =
                out.transpose(1, 2)?
                    .reshape((batch, seq_len, self.n_heads * self.head_dim))?;
            return linear_forward_last_dim(&self.wo, &out);
        }
        let out = compact_gqa_sdpa_bhsd(&q, &k, &v, None, 1.0 / (self.head_dim as f64).sqrt())?
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

/// Initial flow-matching noise for rows `(seed, frame)`: standard normal
/// values from a counter-based stream keyed by the row's seed and frame
/// index, so a frame retried or re-batched gets the same noise. `None` when
/// every row is unseeded (seed 0), which keeps Candle's device `randn`; an
/// unseeded row in a seeded batch draws a time-based seed.
pub(crate) fn seeded_flow_noise(
    rows: &[(u64, usize)],
    codebooks: usize,
    device: &candle_core::Device,
) -> Result<Option<Tensor>> {
    if rows.iter().all(|(seed, _)| *seed == 0) {
        return Ok(None);
    }
    let mut values = Vec::with_capacity(rows.len() * codebooks);
    for &(seed, frame) in rows {
        let seed = if seed == 0 {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_nanos() as u64)
                .unwrap_or(0x9E37_79B9_7F4A_7C15)
        } else {
            seed
        };
        values.extend(gaussian_stream(seed, frame as u64, codebooks));
    }
    Ok(Some(Tensor::from_vec(
        values,
        (rows.len(), codebooks),
        device,
    )?))
}

/// `count` standard normal values (Box-Muller over a splitmix64-seeded
/// xorshift64*) for one (seed, frame) key.
fn gaussian_stream(seed: u64, frame: u64, count: usize) -> Vec<f32> {
    let mut state = seed ^ frame.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut next_u64 = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut uniform = || ((next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    let mut values = Vec::with_capacity(count + 1);
    while values.len() < count {
        let (u1, u2) = (uniform(), uniform());
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        values.push((radius * angle.cos()) as f32);
        values.push((radius * angle.sin()) as f32);
    }
    values.truncate(count);
    values
}

/// The reference `TimeEmbedding(dim, theta=10000.0)` (vllm-omni
/// voxtral_tts_audio_generation.py): a fixed constant, not the acoustic
/// transformer's `rope_theta` (equal today, but independent).
const TIME_EMBEDDING_THETA: f64 = 10_000.0;

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
        let t = t.to_dtype(self.inv_freq.dtype())?;
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

fn shifted_acoustic_codes_from_sampled_tensor(
    sampled: &Tensor,
    semantic_codes: &[u32],
    acoustic_codebook_size: usize,
) -> Result<DecodedAcousticCodes> {
    if sampled.device().is_cuda() {
        if let Some(decoded) = shifted_acoustic_codes_from_sampled_tensor_cuda(
            sampled,
            semantic_codes,
            acoustic_codebook_size,
        )? {
            return Ok(decoded);
        }
    }
    shifted_acoustic_codes_from_sampled_tensor_host(sampled, semantic_codes, acoustic_codebook_size)
}

/// The flow state quantized to shifted acoustic codes on the device
/// (`round((clamp(x) + 1) / 2 * (size - 1)) + 2`, as [`fsq_unit_to_code`] and
/// [`apply_audio_token_offset`] compute on the host), as U32.
fn shifted_acoustic_code_tensor(sampled: &Tensor, acoustic_codebook_size: usize) -> Result<Tensor> {
    let shifted = sampled
        .clamp(-1.0, 1.0)?
        .to_dtype(DType::F32)?
        .broadcast_add(&Tensor::full(
            1.0f32,
            sampled.shape().clone(),
            sampled.device(),
        )?)?;
    let scale = 0.5f64 * (acoustic_codebook_size as f64 - 1.0);
    let shifted = (shifted * scale)?
        .round()?
        .broadcast_add(&Tensor::full(
            AUDIO_SPECIAL_TOKEN_COUNT as f32,
            sampled.shape().clone(),
            sampled.device(),
        )?)?
        .to_dtype(DType::U32)?;
    Ok(shifted)
}

fn shifted_acoustic_codes_from_sampled_tensor_cuda(
    sampled: &Tensor,
    semantic_codes: &[u32],
    acoustic_codebook_size: usize,
) -> Result<Option<DecodedAcousticCodes>> {
    let shifted = shifted_acoustic_code_tensor(sampled, acoustic_codebook_size)?;
    record_frame_readback();
    let mut rows = shifted.to_vec2::<u32>()?;
    let contains_end = semantic_codes
        .iter()
        .any(|semantic| *semantic == AudioSpecialToken::End.id());
    if contains_end {
        let empty = apply_audio_token_offset(AudioSpecialToken::Empty.id());
        for (semantic, row) in semantic_codes.iter().zip(rows.iter_mut()) {
            if *semantic == AudioSpecialToken::End.id() {
                row.fill(empty);
            }
        }
    }
    Ok(Some(DecodedAcousticCodes {
        shifted_codes: rows,
        shifted_code_tensor: (!contains_end).then_some(shifted),
    }))
}

fn shifted_acoustic_codes_from_sampled_tensor_host(
    sampled: &Tensor,
    semantic_codes: &[u32],
    acoustic_codebook_size: usize,
) -> Result<DecodedAcousticCodes> {
    let sampled = sampled.clamp(-1.0, 1.0)?.to_dtype(DType::F32)?;
    record_frame_readback();
    let rows = sampled.to_vec2::<f32>()?;
    let mut output = Vec::with_capacity(semantic_codes.len());
    for (semantic, row) in semantic_codes.iter().zip(rows) {
        let should_decode = *semantic != AudioSpecialToken::End.id();
        let mut acoustic = Vec::with_capacity(row.len());
        for value in row {
            let raw = if should_decode {
                fsq_unit_to_code(value, acoustic_codebook_size)
            } else {
                AudioSpecialToken::Empty.id()
            };
            acoustic.push(apply_audio_token_offset(raw));
        }
        output.push(acoustic);
    }
    Ok(DecodedAcousticCodes {
        shifted_codes: output,
        shifted_code_tensor: None,
    })
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
    semantic_codes_from_logits_with_eos_policy(logits, semantic_codebook_size, true)
}

fn semantic_codes_from_logits_with_eos_policy(
    logits: &Tensor,
    semantic_codebook_size: usize,
    allow_end_audio: bool,
) -> Result<Vec<u32>> {
    Ok(semantic_code_selection_from_logits_with_eos_policy(
        logits,
        semantic_codebook_size,
        allow_end_audio,
    )?
    .codes)
}

fn semantic_code_selection_from_logits_with_eos_policy(
    logits: &Tensor,
    semantic_codebook_size: usize,
    allow_end_audio: bool,
) -> Result<SemanticCodeSelection> {
    // Non-empty TTS requests should produce at least one audible frame. Some
    // voices can rank end-of-audio first, so the generation loop masks EOS only
    // for frame zero and restores the normal stop token immediately after.
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
    if logits.device().is_cuda() {
        if let Some(selection) =
            semantic_code_selection_from_logits_cuda(&logits, allowed, allow_end_audio)?
        {
            return Ok(selection);
        }
    }
    semantic_code_selection_from_logits_host(&logits, allowed, allow_end_audio)
}

fn semantic_code_selection_from_logits_cuda(
    logits: &Tensor,
    allowed: usize,
    allow_end_audio: bool,
) -> Result<Option<SemanticCodeSelection>> {
    if allowed == 0 {
        return Ok(None);
    }
    let (indices, selected_scores) = semantic_argmax_tensors(logits, allowed, allow_end_audio)?;
    record_frame_readback();
    let selected_scores = selected_scores.to_vec2::<f32>()?;
    if selected_scores
        .iter()
        .flatten()
        .any(|score| !score.is_finite())
    {
        return Ok(None);
    }
    record_frame_readback();
    let codes = indices.to_vec1::<u32>()?;
    Ok(Some(SemanticCodeSelection {
        codes,
        code_tensor: Some(indices.unsqueeze(1)?),
    }))
}

/// The device argmax over the allowed semantic codes (`[rows]` U32) and the
/// selected scores (`[rows, 1]` F32), left on the device.
fn semantic_argmax_tensors(
    logits: &Tensor,
    allowed: usize,
    allow_end_audio: bool,
) -> Result<(Tensor, Tensor)> {
    let batch = logits.dim(0)?;
    let token_logits = logits.narrow(1, 0, allowed)?;
    let mask = semantic_validity_mask(batch, allowed, allow_end_audio);
    let mask = Tensor::from_vec(mask, (batch, allowed), logits.device())?;
    let neg_inf = Tensor::full(f32::NEG_INFINITY, token_logits.shape(), logits.device())?
        .to_dtype(token_logits.dtype())?;
    let masked = mask.where_cond(&token_logits, &neg_inf)?;
    let indices = masked.argmax(D::Minus1)?.to_dtype(DType::U32)?;
    let scores = masked
        .gather(&indices.unsqueeze(1)?, 1)?
        .to_dtype(DType::F32)?;
    Ok((indices, scores))
}

fn semantic_validity_mask(batch: usize, allowed: usize, allow_end_audio: bool) -> Vec<u8> {
    let mut row = vec![1u8; allowed];
    if (AudioSpecialToken::Empty.id() as usize) < allowed {
        row[AudioSpecialToken::Empty.id() as usize] = 0;
    }
    if !allow_end_audio && (AudioSpecialToken::End.id() as usize) < allowed {
        row[AudioSpecialToken::End.id() as usize] = 0;
    }
    let mut mask = Vec::with_capacity(batch * allowed);
    for _ in 0..batch {
        mask.extend_from_slice(&row);
    }
    mask
}

fn semantic_code_selection_from_logits_host(
    logits: &Tensor,
    allowed: usize,
    allow_end_audio: bool,
) -> Result<SemanticCodeSelection> {
    record_frame_readback();
    let rows = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let codes = rows
        .into_iter()
        .map(|row| {
            let mut best_idx = None;
            let mut best_value = f32::NEG_INFINITY;
            for (idx, value) in row.iter().take(allowed).enumerate() {
                if idx == AudioSpecialToken::Empty.id() as usize {
                    continue;
                }
                if !allow_end_audio && idx == AudioSpecialToken::End.id() as usize {
                    continue;
                }
                if !value.is_finite() {
                    continue;
                }
                if *value > best_value {
                    best_idx = Some(idx);
                    best_value = *value;
                }
            }
            let best_idx = best_idx.ok_or_else(|| {
                Error::InferenceError(
                    "Voxtral semantic logits contained no finite valid audio tokens".to_string(),
                )
            })?;
            Ok(best_idx as u32)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(SemanticCodeSelection {
        codes,
        code_tensor: None,
    })
}

fn blend_cfg_tensors(cond: &Tensor, uncond: &Tensor, alpha: f32) -> Result<Tensor> {
    let cond_scaled = (cond * alpha as f64)?;
    let uncond_scaled = (uncond * (1.0f64 - alpha as f64))?;
    cond_scaled
        .broadcast_add(&uncond_scaled)
        .map_err(Error::from)
}

#[cfg(test)]
pub(crate) mod tests {
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
    fn tensorized_fsq_quantization_matches_shifted_codes() {
        let device = Device::Cpu;
        let sampled = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0], (1, 3), &device).unwrap();

        let decoded = shifted_acoustic_codes_from_sampled_tensor_cuda(&sampled, &[2], 21)
            .unwrap()
            .unwrap();

        assert_eq!(decoded.shifted_codes, vec![vec![2, 12, 22]]);
        assert_eq!(
            decoded
                .shifted_code_tensor
                .unwrap()
                .to_vec2::<u32>()
                .unwrap(),
            vec![vec![2, 12, 22]]
        );
    }

    #[test]
    fn tensorized_fsq_omits_feedback_tensor_for_end_audio() {
        let device = Device::Cpu;
        let sampled = Tensor::from_vec(vec![-1.0f32, 0.0, 1.0], (1, 3), &device).unwrap();

        let decoded = shifted_acoustic_codes_from_sampled_tensor_cuda(
            &sampled,
            &[AudioSpecialToken::End.id()],
            21,
        )
        .unwrap()
        .unwrap();

        assert_eq!(decoded.shifted_codes, vec![vec![2, 2, 2]]);
        assert!(decoded.shifted_code_tensor.is_none());
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
    fn time_embedding_accepts_bf16_step_values() {
        let device = Device::Cpu;
        let embedding = TimeEmbedding::new(4, 10000.0, &device).unwrap();
        let t = Tensor::from_vec(vec![0.0f32], (1, 1), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let output = embedding.forward(&t).unwrap();

        assert_eq!(output.dtype(), DType::F32);
        assert_eq!(output.dims(), &[1, 4]);
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
    fn semantic_logits_can_mask_end_audio_for_first_frame() {
        let device = Device::Cpu;
        let logits =
            Tensor::from_vec(vec![100.0, 99.0, 1.0, 2.0, 98.0, 97.0], (1, 6), &device).unwrap();

        let eos_allowed = semantic_codes_from_logits_with_eos_policy(&logits, 2, true).unwrap();
        let eos_masked = semantic_codes_from_logits_with_eos_policy(&logits, 2, false).unwrap();

        assert_eq!(eos_allowed, vec![1]);
        assert_eq!(eos_masked, vec![3]);
    }

    #[test]
    fn tensorized_semantic_selection_masks_empty_and_eos() {
        let device = Device::Cpu;
        let logits =
            Tensor::from_vec(vec![100.0f32, 99.0, 1.0, 2.0, 98.0], (1, 5), &device).unwrap();

        let selection = semantic_code_selection_from_logits_cuda(&logits, 4, false)
            .unwrap()
            .unwrap();

        assert_eq!(selection.codes, vec![3]);
        assert_eq!(
            selection.code_tensor.unwrap().to_vec2::<u32>().unwrap(),
            vec![vec![3]]
        );
    }

    #[test]
    fn tensorized_semantic_selection_falls_back_on_non_finite_selected_score() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.0f32, f32::NAN, f32::NAN], (1, 3), &device).unwrap();

        let selection = semantic_code_selection_from_logits_cuda(&logits, 3, true).unwrap();

        assert!(selection.is_none());
    }

    #[test]
    fn semantic_logits_reject_non_finite_rows() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![100.0, f32::NAN, f32::NEG_INFINITY, f32::NAN, 99.0],
            (1, 5),
            &device,
        )
        .unwrap();

        let err = semantic_codes_from_logits_with_eos_policy(&logits, 2, true).unwrap_err();

        assert!(format!("{err}").contains("no finite valid audio tokens"));
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

    /// VXT-12: seeded flow-matching noise is a pure function of (seed,
    /// frame): reproducible, independent of batch position, standard normal,
    /// and actually used as the ODE's starting point.
    #[test]
    fn seeded_flow_noise_is_reproducible_and_drives_the_frame() {
        let device = Device::Cpu;
        assert!(seeded_flow_noise(&[(0, 3), (0, 4)], 2, &device)
            .unwrap()
            .is_none());
        let single = seeded_flow_noise(&[(7, 3)], 2, &device).unwrap().unwrap();
        let batched = seeded_flow_noise(&[(9, 0), (7, 3)], 2, &device)
            .unwrap()
            .unwrap();
        assert_eq!(
            single.to_vec2::<f32>().unwrap()[0],
            batched.to_vec2::<f32>().unwrap()[1]
        );
        assert_ne!(gaussian_stream(7, 3, 2), gaussian_stream(7, 4, 2));
        let values = gaussian_stream(42, 0, 20_000);
        let mean = values.iter().map(|v| f64::from(*v)).sum::<f64>() / values.len() as f64;
        let var = values
            .iter()
            .map(|v| (f64::from(*v) - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        assert!(
            mean.abs() < 0.03 && (var - 1.0).abs() < 0.05,
            "mean {mean}, var {var}"
        );

        // The tiny transformer has zero weights, so the velocity is zero and
        // the frame's codes are the quantized starting noise (end-of-audio is
        // masked so the acoustic codes are decoded at all).
        let config = tiny_acoustic_config();
        let vb = VarBuilder::from_tensors(tiny_acoustic_tensors(&device), DType::F32, &device);
        let transformer = FlowMatchingAudioTransformer::load(&config, vb).unwrap();
        let hidden = Tensor::zeros((1, 4), DType::F32, &device).unwrap();
        let run = |noise: &Tensor| {
            transformer
                .forward_audio_codes_with_steps_and_eos_policy(&hidden, 1.2, 1, false, Some(noise))
                .unwrap()
        };
        let frames = (0..8)
            .map(|frame| {
                let noise = seeded_flow_noise(&[(11, frame)], 2, &device)
                    .unwrap()
                    .unwrap();
                let codes = run(&noise);
                assert_eq!(codes, run(&noise), "frame {frame} is not reproducible");
                codes
            })
            .collect::<Vec<_>>();
        assert!(frames.windows(2).any(|pair| pair[0] != pair[1]));
    }

    /// VXT-10: the packed frame path (CUDA's) gives the same frames and
    /// feedback tensor as the per-path selection with one readback per
    /// frame (the per-path code takes two on the host path and three on
    /// CUDA), including end-of-audio rows.
    #[test]
    fn packed_frame_matches_per_path_selection_in_one_readback() {
        use crate::kernels::numeric_check::xorshift_tensor;

        let device = Device::Cpu;
        let config = tiny_acoustic_config();
        let random = tiny_acoustic_tensors(&device)
            .into_iter()
            .collect::<std::collections::BTreeMap<_, _>>()
            .into_iter()
            .enumerate()
            .map(|(index, (name, tensor))| {
                let value = if name.contains("norm") {
                    tensor
                } else {
                    xorshift_tensor(tensor.dims(), 0x9ACC + index as u64, 0.5, &device).unwrap()
                };
                (name, value)
            })
            .collect::<HashMap<_, _>>();
        let zero = tiny_acoustic_tensors(&device);
        for (tensors, expect_end) in [(random, false), (zero, true)] {
            let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
            let transformer = FlowMatchingAudioTransformer::load(&config, vb).unwrap();
            let hidden = xorshift_tensor(&[3, 4], 0xC0DE, 1.0, &device).unwrap();
            let noise = xorshift_tensor(&[3, 2], 0xD00D, 1.0, &device).unwrap();
            for allow_end in [true, false] {
                let before = thread_frame_readbacks();
                let expected = transformer
                    .forward_audio_codes_with_feedback_tensor(
                        &hidden,
                        1.2,
                        4,
                        allow_end,
                        Some(&noise),
                    )
                    .unwrap();
                assert_eq!(thread_frame_readbacks() - before, 2);
                let logits =
                    linear_forward_last_dim(&transformer.semantic_codebook_output, &hidden)
                        .unwrap();
                let before = thread_frame_readbacks();
                let packed = transformer
                    .packed_frame(&hidden, &logits, 1.2, 4, allow_end, Some(&noise))
                    .unwrap();
                assert_eq!(thread_frame_readbacks() - before, 1);
                assert_eq!(packed.frames, expected.frames);
                let has_end = packed
                    .frames
                    .iter()
                    .any(|frame| frame[0] == AudioSpecialToken::End.id());
                if expect_end || !allow_end {
                    assert_eq!(has_end, expect_end && allow_end);
                }
                assert_eq!(packed.shifted_code_tensor.is_some(), !has_end);
                if let Some(tensor) = &packed.shifted_code_tensor {
                    assert_eq!(tensor.to_vec2::<u32>().unwrap(), packed.frames);
                }
            }
        }
    }

    /// The pre-VXT-10 flow-matching loop: every step re-uploads `t`, re-runs
    /// the time embedding and projection, and re-projects the conditional
    /// and zero LLM rows.
    fn decode_reference(
        transformer: &FlowMatchingAudioTransformer,
        llm_hidden: &Tensor,
        noise: &Tensor,
        cfg_alpha: f32,
        n_steps: usize,
    ) -> Tensor {
        let batch = llm_hidden.dim(0).unwrap();
        let dtype = llm_hidden.dtype();
        let device = llm_hidden.device();
        let mut sampled = noise.clone();
        let llm_hidden_zero = Tensor::zeros(llm_hidden.shape(), dtype, device).unwrap();
        for step in 0..n_steps {
            let t = step as f32 / n_steps as f32;
            let dt = 1.0f64 / n_steps as f64;
            let t_values = Tensor::from_vec(vec![t; batch], (batch, 1), device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let t_emb = transformer
                .time_embedding
                .forward(&t_values)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            let x_batched = Tensor::cat(&[sampled.clone(), sampled.clone()], 0).unwrap();
            let llm_batched =
                Tensor::cat(&[llm_hidden.clone(), llm_hidden_zero.clone()], 0).unwrap();
            let t_emb_batched = Tensor::cat(&[t_emb.clone(), t_emb], 0).unwrap();
            let projected_t = linear_forward_last_dim(&transformer.time_projection, &t_emb_batched)
                .unwrap()
                .unsqueeze(1)
                .unwrap();
            let projected_llm = linear_forward_last_dim(&transformer.llm_projection, &llm_batched)
                .unwrap()
                .unsqueeze(1)
                .unwrap();
            let velocity = transformer
                .predict_velocity(&x_batched, &projected_t, &projected_llm)
                .unwrap();
            let cond = velocity.narrow(0, 0, batch).unwrap();
            let uncond = velocity.narrow(0, batch, batch).unwrap();
            let blended = blend_cfg_tensors(&cond, &uncond, cfg_alpha).unwrap();
            sampled = sampled.broadcast_add(&(blended * dt).unwrap()).unwrap();
        }
        sampled
    }

    /// VXT-10: the hoisted loop (LLM rows projected once per frame, the
    /// projected step times cached per schedule) matches the per-step loop
    /// on a random tiny transformer, and uploads step times once per
    /// schedule instead of once per step.
    #[test]
    fn hoisted_flow_matching_matches_the_per_step_loop() {
        use crate::kernels::numeric_check::{host_f32, max_abs_diff, xorshift_tensor};

        let device = Device::Cpu;
        let config = tiny_acoustic_config();
        let tensors = tiny_acoustic_tensors(&device)
            .into_iter()
            .enumerate()
            .map(|(index, (name, tensor))| {
                let random = if name.contains("norm") {
                    tensor
                } else {
                    xorshift_tensor(tensor.dims(), 0x7A11 + index as u64, 0.5, &device).unwrap()
                };
                (name, random)
            })
            .collect::<HashMap<_, _>>();
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let transformer = FlowMatchingAudioTransformer::load(&config, vb).unwrap();
        let llm_hidden = xorshift_tensor(&[3, 4], 0xBEEF, 1.0, &device).unwrap();
        let noise = xorshift_tensor(&[3, 2], 0xF00D, 1.0, &device).unwrap();
        let n_steps = 7;
        let before = thread_acoustic_step_uploads();
        for _frame in 0..3 {
            let codes = transformer
                .decode_one_frame(&[5, 6, 7], &llm_hidden, 1.2, n_steps, Some(&noise))
                .unwrap();
            assert_eq!(codes.shifted_codes.len(), 3);
        }
        // Three frames of seven steps: one upload (it was 21).
        assert_eq!(thread_acoustic_step_uploads() - before, 1);

        // Compare the continuous ODE state, before quantization.
        let expected = decode_reference(&transformer, &llm_hidden, &noise, 1.2, n_steps);
        let actual = transformer
            .integrate_flow(noise.clone(), &llm_hidden, 1.2, n_steps)
            .unwrap();
        let diff = max_abs_diff(&host_f32(&actual).unwrap(), &host_f32(&expected).unwrap());
        assert!(diff == 0.0, "bit-exact on CPU; max |diff| {diff}");
    }

    /// VXT-12: the time embedding uses the reference's fixed theta 10000,
    /// whatever `rope_theta` the acoustic transformer config carries.
    #[test]
    fn time_embedding_theta_is_the_reference_constant() {
        let device = Device::Cpu;
        let mut config = tiny_acoustic_config();
        config
            .multimodal
            .audio_model_args
            .acoustic_transformer_args
            .rope_theta = 1_000_000.0;
        let vb = VarBuilder::from_tensors(tiny_acoustic_tensors(&device), DType::F32, &device);
        let transformer = FlowMatchingAudioTransformer::load(&config, vb).unwrap();
        let inv_freq = transformer
            .time_embedding
            .inv_freq
            .to_vec1::<f32>()
            .unwrap();
        // dim 4: half 2, inv_freq[1] = 10000^(-1/2).
        assert!((inv_freq[1] - 0.01).abs() < 1e-7, "{inv_freq:?}");
    }

    pub(crate) fn tiny_acoustic_config() -> VoxtralTtsConfig {
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

    pub(crate) fn tiny_acoustic_tensors(device: &Device) -> HashMap<String, Tensor> {
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
