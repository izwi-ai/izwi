//! Main Voxtral Realtime model implementation.

use std::path::Path;

use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{Module, VarBuilder};
use tracing::info;

use crate::audio::{MelConfig, MelSpectrogram};
use crate::backends::DeviceProfile;
use crate::catalog::ModelFamily;
use crate::error::{Error, Result};
use crate::models::architectures::qwen3::core::Qwen3Cache;
use crate::models::architectures::voxtral::lm::VoxtralLM;
use crate::models::shared::attention::flash::{
    flash_attention_requested, try_fused_self_attention,
};
use crate::models::shared::config::checkpoint_dtype_from_config_json;

use super::audio::{AudioLanguageAdapter, TimeEmbedding};
use super::config::VoxtralConfig;
use super::tokenizer::{AudioConfig, VoxtralTokenizer};

/// Voxtral Realtime Model
pub struct VoxtralRealtimeModel {
    device: DeviceProfile,
    dtype: DType,
    tokenizer: VoxtralTokenizer,
    config: VoxtralConfig,
    whisper_encoder: WhisperEncoder,
    audio_adapter: AudioLanguageAdapter,
    language_model: VoxtralLM,
    time_embedding: TimeEmbedding,
    mel: MelSpectrogram,
}

impl VoxtralRealtimeModel {
    /// Load model from directory
    pub fn load(model_dir: &Path, device: DeviceProfile) -> Result<Self> {
        // Try params.json (Voxtral format) first, then config.json (standard)
        let config_path = if model_dir.join("params.json").exists() {
            model_dir.join("params.json")
        } else {
            model_dir.join("config.json")
        };
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| Error::ModelLoadError(format!("Failed to read config: {}", e)))?;
        let config: VoxtralConfig = serde_json::from_str(&config_str)
            .map_err(|e| Error::ModelLoadError(format!("Failed to parse config: {}", e)))?;
        let checkpoint_dtype = checkpoint_dtype_from_config_json(&config_str);

        // Setup audio processing
        let audio_cfg = config.audio_config();
        let mel_cfg = MelConfig {
            sample_rate: audio_cfg.sampling_rate,
            n_fft: audio_cfg.window_size,
            hop_length: audio_cfg.hop_length,
            n_mels: audio_cfg.num_mel_bins,
            f_min: 0.0,
            f_max: 8000.0,
            normalize: true,
        };
        let mel = MelSpectrogram::new(mel_cfg)?;

        // Setup tokenizer
        let audio_config = AudioConfig {
            sampling_rate: audio_cfg.sampling_rate,
            frame_rate: config.frame_rate(),
            window_size: audio_cfg.window_size,
            hop_length: audio_cfg.hop_length,
            num_mel_bins: audio_cfg.num_mel_bins,
            n_delay_tokens: config.num_delay_tokens(),
        };
        let tokenizer = VoxtralTokenizer::load(model_dir, audio_config)?;

        let dtype = select_voxtral_dtype(&device, checkpoint_dtype);

        // Load weights - clone device to a local binding for lifetime
        let device_clone = device.clone();
        let vb = load_weights(model_dir, dtype, &device_clone)?;

        // Load components
        // Note: Checkpoint uses mm_streams_embeddings.embedding_module prefix for audio components
        let whisper_prefix = "mm_streams_embeddings.embedding_module.whisper_encoder";
        let whisper_encoder = WhisperEncoder::load_voxtral(&audio_cfg, vb.pp(whisper_prefix))?;

        let hidden_size = audio_cfg.d_model * config.downsample_factor();
        let audio_adapter = AudioLanguageAdapter::load(
            hidden_size,
            config.text_config().hidden_size,
            vb.pp("mm_streams_embeddings.embedding_module.audio_language_projection"),
        )?;

        // Language model uses root-level layers.* and norm (Mistral-style)
        let language_model = VoxtralLM::load(config.text_config().into(), vb.clone())?;

        let time_embedding =
            TimeEmbedding::new(config.text_config().hidden_size, 10000.0, &device.device)?;

        info!("Loaded Voxtral Realtime model on {:?}", device.kind);

        Ok(Self {
            device,
            dtype,
            tokenizer,
            config,
            whisper_encoder,
            audio_adapter,
            language_model,
            time_embedding,
            mel,
        })
    }

    /// Transcribe audio (non-streaming)
    pub fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        _language: Option<&str>,
    ) -> Result<String> {
        let mut no_op = |_delta: &str| {};
        self.transcribe_with_callback(audio, sample_rate, _language, &mut no_op)
    }

    pub fn transcribe_with_callback(
        &self,
        audio: &[f32],
        sample_rate: u32,
        _language: Option<&str>,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<String> {
        // Resample to 16kHz if needed
        let audio = if sample_rate != 16_000 {
            resample_audio(audio, sample_rate, 16_000)?
        } else {
            audio.to_vec()
        };

        // Compute mel spectrogram
        let mel_spec = self.mel.compute(&audio)?;
        let n_mels = self.mel.config().n_mels;
        let frames = mel_spec.len();

        // Convert to tensor: [n_mels, frames] -> [1, n_mels, frames]
        let mut flat = Vec::with_capacity(frames * n_mels);
        for frame in mel_spec.iter() {
            flat.extend_from_slice(frame);
        }

        let mel = Tensor::from_vec(flat, (n_mels, frames), &self.device.device)?
            .transpose(0, 1)?
            .unsqueeze(0)? // [1, frames, n_mels]
            .to_dtype(self.dtype)?;

        // Process through whisper encoder
        let audio_embeds = self.whisper_encoder.forward(&mel)?;

        // Apply pooling
        let audio_embeds = self.pool_audio_embeddings(&audio_embeds)?;

        // Project to language model dimension
        let audio_embeds = self.audio_adapter.forward(&audio_embeds)?;

        // Build prompt
        let audio_tokens = audio_embeds.dim(1)?;
        let prompt_ids = self.tokenizer.build_transcription_prompt(audio_tokens);
        let input_ids = Tensor::from_vec(
            prompt_ids.clone(),
            (1, prompt_ids.len()),
            &self.device.device,
        )?;

        // Get text embeddings
        let text_embeds = self.language_model.embeddings(&input_ids)?;

        // Sum audio and text embeddings
        let combined_embeds = audio_embeds.broadcast_add(&text_embeds)?;

        // Apply time conditioning
        let time_tensor = Tensor::from_vec(
            vec![self.config.num_delay_tokens() as f32],
            (1,),
            &self.device.device,
        )?
        .to_dtype(self.dtype)?;
        let t_cond = self.time_embedding.forward(&time_tensor)?;

        // Generate
        let mut cache = Qwen3Cache::new(self.language_model.num_layers());
        let mut generated = Vec::new();
        let mut assembled = String::new();
        let max_tokens = 1024usize;

        // Forward with audio - use forward_with_embeds for custom embeddings
        let combined_embeds = combined_embeds.broadcast_add(&t_cond.unsqueeze(0)?)?;
        let mut logits =
            self.language_model
                .forward_with_embeds(&combined_embeds, 0, Some(&mut cache), None)?;

        let specials = self.tokenizer.specials();
        let mut pos = combined_embeds.dim(1)?;

        for _ in 0..max_tokens {
            let next_logits = logits.i((0, logits.dim(1)? - 1))?;
            let next = argmax(&next_logits)?;

            if next == specials.eos || Some(next) == specials.end_audio {
                break;
            }

            generated.push(next);
            let decoded = self.tokenizer.decode_text(&generated)?;
            let delta = text_delta(&assembled, &decoded);
            for ch in delta.chars() {
                let mut buf = [0u8; 4];
                on_delta(ch.encode_utf8(&mut buf));
            }
            assembled = decoded;

            let next_tensor = Tensor::from_vec(vec![next], (1, 1), &self.device.device)?;
            logits = self
                .language_model
                .forward(&next_tensor, pos, Some(&mut cache))?;
            pos += 1;
        }

        Ok(assembled.trim().to_string())
    }

    /// Pool audio embeddings by block_size
    fn pool_audio_embeddings(&self, audio_embeds: &Tensor) -> Result<Tensor> {
        let (bsz, seq_len, hidden) = audio_embeds.dims3()?;
        let pool_size = self.config.block_pool_size();

        // Ensure seq_len is divisible by pool_size
        let new_len = seq_len / pool_size;
        let truncated_len = new_len * pool_size;

        if truncated_len < seq_len {
            let audio_embeds = audio_embeds.narrow(1, seq_len - truncated_len, truncated_len)?;
            let reshaped = audio_embeds.reshape((bsz, new_len, hidden * pool_size))?;
            Ok(reshaped)
        } else {
            let reshaped = audio_embeds.reshape((bsz, new_len, hidden * pool_size))?;
            Ok(reshaped)
        }
    }
}

/// Whisper encoder for audio processing
pub struct WhisperEncoder {
    conv1: candle_nn::Conv1d,
    conv2: candle_nn::Conv1d,
    layers: Vec<WhisperEncoderLayer>,
    ln_post: Option<candle_nn::LayerNorm>,
    ln_post_rms: Option<candle_nn::RmsNorm>,
    embed_positions: Tensor,
    is_causal: bool,
}

impl WhisperEncoder {
    /// Load for standard Whisper format
    pub fn load(cfg: &super::config::AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Self::load_internal(cfg, vb, false)
    }

    /// Load for Voxtral checkpoint format (different tensor naming)
    pub fn load_voxtral(cfg: &super::config::AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Self::load_internal(cfg, vb, true)
    }

    fn load_internal(
        cfg: &super::config::AudioEncoderConfig,
        vb: VarBuilder,
        is_voxtral: bool,
    ) -> Result<Self> {
        let conv1_config = candle_nn::Conv1dConfig {
            stride: cfg.conv1_stride,
            padding: 1,
            groups: 1,
            dilation: 1,
            ..Default::default()
        };
        let conv2_config = candle_nn::Conv1dConfig {
            stride: cfg.conv2_stride,
            padding: 1,
            groups: 1,
            dilation: 1,
            ..Default::default()
        };

        // Voxtral uses conv_layers.0.conv and conv_layers.1.conv
        let (conv1, conv2) = if is_voxtral {
            let conv1 = candle_nn::conv1d(
                cfg.num_mel_bins,
                cfg.d_model,
                cfg.conv1_kernel_size,
                conv1_config,
                vb.pp("conv_layers.0.conv"),
            )?;
            let conv2 = candle_nn::conv1d(
                cfg.d_model,
                cfg.d_model,
                cfg.conv2_kernel_size,
                conv2_config,
                vb.pp("conv_layers.1.conv"),
            )?;
            (conv1, conv2)
        } else {
            let conv1 = candle_nn::conv1d(
                cfg.num_mel_bins,
                cfg.d_model,
                cfg.conv1_kernel_size,
                conv1_config,
                vb.pp("conv1"),
            )?;
            let conv2 = candle_nn::conv1d(
                cfg.d_model,
                cfg.d_model,
                cfg.conv2_kernel_size,
                conv2_config,
                vb.pp("conv2"),
            )?;
            (conv1, conv2)
        };

        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for i in 0..cfg.encoder_layers {
            // Voxtral uses transformer.layers.{i} instead of layers.{i}
            let layer_vb = if is_voxtral {
                vb.pp(format!("transformer.layers.{i}"))
            } else {
                vb.pp(format!("layers.{i}"))
            };
            layers.push(WhisperEncoderLayer::load(cfg, layer_vb, is_voxtral)?);
        }

        let (ln_post, ln_post_rms) = if is_voxtral {
            // Voxtral uses transformer.norm (RMSNorm)
            let norm = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("transformer.norm"))?;
            (None, Some(norm))
        } else {
            let norm = candle_nn::layer_norm(cfg.d_model, 1e-5, vb.pp("ln_post"))?;
            (Some(norm), None)
        };

        // Sinusoidal position embeddings
        let max_len = cfg.max_source_positions;
        let hidden = cfg.d_model;
        let half_hidden = hidden / 2;
        let log_theta = (10000f32).ln() / (half_hidden as f32 - 1.0);
        let inv_freq: Vec<f32> = (0..half_hidden)
            .map(|i| (-log_theta * i as f32).exp())
            .collect();

        let mut pos_embed_data = Vec::with_capacity(max_len * hidden);
        for pos in 0..max_len {
            for i in 0..half_hidden {
                let timescale = inv_freq[i];
                pos_embed_data.push((pos as f32 * timescale).sin());
            }
            for i in 0..half_hidden {
                let timescale = inv_freq[i];
                pos_embed_data.push((pos as f32 * timescale).cos());
            }
        }

        let embed_positions = Tensor::from_vec(pos_embed_data, (max_len, hidden), vb.device())?;

        Ok(Self {
            conv1,
            conv2,
            layers,
            ln_post,
            ln_post_rms,
            embed_positions,
            is_causal: cfg.is_causal,
        })
    }

    pub fn forward(&self, input_features: &Tensor) -> Result<Tensor> {
        // input_features: [batch, frames, n_mels]
        // Transpose to [batch, n_mels, frames] for conv1d
        let x = input_features.transpose(1, 2)?;

        // Conv layers with gelu
        let x = self.conv1.forward(&x)?;
        let x = gelu(&x)?;
        let x = self.conv2.forward(&x)?;
        let x = gelu(&x)?;

        // Transpose back: [batch, hidden, frames] -> [batch, frames, hidden]
        let x = x.transpose(1, 2)?;

        // Add positional embeddings
        let seq_len = x.dim(1)?;
        let pos_embed = self.embed_positions.narrow(0, 0, seq_len)?;
        let pos_embed = pos_embed.unsqueeze(0)?.broadcast_as(x.shape())?;
        let x = x.broadcast_add(&pos_embed)?;

        // Transformer layers
        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, self.is_causal)?;
        }

        // Final layer norm
        if self.ln_post_rms.is_some() {
            self.ln_post_rms.as_ref().unwrap().forward(&x)
        } else {
            self.ln_post.as_ref().unwrap().forward(&x)
        }
        .map_err(|e| Error::InferenceError(e.to_string()))
    }

    /// Forward only conv layers (for realtime processing)
    pub fn forward_conv(&self, mel_features: &[Tensor]) -> Result<Tensor> {
        let mut outputs = Vec::with_capacity(mel_features.len());

        for mel in mel_features {
            // mel: [n_mels, seq_len] -> [1, n_mels, seq_len]
            let x = mel.unsqueeze(0)?.transpose(1, 2)?;
            let x = self.conv1.forward(&x)?;
            let x = gelu(&x)?;
            let x = self.conv2.forward(&x)?;
            let x = gelu(&x)?;
            // [1, hidden, seq_len] -> [hidden, seq_len]
            let x = x.squeeze(0)?;
            outputs.push(x);
        }

        // Concatenate along sequence dimension
        let outputs_refs: Vec<&Tensor> = outputs.iter().collect();
        Tensor::cat(&outputs_refs, 1).map_err(|e| Error::InferenceError(e.to_string()))
    }
}

/// Whisper encoder layer
struct WhisperEncoderLayer {
    self_attn_layer_norm: Option<candle_nn::LayerNorm>,
    final_layer_norm: Option<candle_nn::LayerNorm>,
    self_attn_rms_norm: Option<candle_nn::RmsNorm>,
    final_rms_norm: Option<candle_nn::RmsNorm>,
    self_attn: WhisperAttention,
    fc1: Option<candle_nn::Linear>,
    fc2: Option<candle_nn::Linear>,
    ffn_w1: Option<candle_nn::Linear>,
    ffn_w2: Option<candle_nn::Linear>,
    ffn_w3: Option<candle_nn::Linear>,
    is_voxtral: bool,
}

impl WhisperEncoderLayer {
    pub fn load(
        cfg: &super::config::AudioEncoderConfig,
        vb: VarBuilder,
        is_voxtral: bool,
    ) -> Result<Self> {
        let (self_attn_layer_norm, self_attn_rms_norm, self_attn, final_layer_norm, final_rms_norm) =
            if is_voxtral {
                let norm = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("attention_norm"))?;
                let attn = WhisperAttention::load_voxtral(cfg, vb.pp("attention"))?;
                let ffn_norm = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("ffn_norm"))?;
                (None, Some(norm), attn, None, Some(ffn_norm))
            } else {
                let norm = candle_nn::layer_norm(cfg.d_model, 1e-5, vb.pp("self_attn_layer_norm"))?;
                let attn = WhisperAttention::load(cfg, vb.pp("self_attn"))?;
                let ffn_norm = candle_nn::layer_norm(cfg.d_model, 1e-5, vb.pp("final_layer_norm"))?;
                (Some(norm), None, attn, Some(ffn_norm), None)
            };

        let (fc1, fc2, ffn_w1, ffn_w2, ffn_w3) = if is_voxtral {
            // Voxtral uses w1, w2, w3 for feed-forward (no bias)
            let w1 = candle_nn::linear_no_bias(
                cfg.d_model,
                cfg.encoder_ffn_dim,
                vb.pp("feed_forward.w1"),
            )?;
            let w2 = candle_nn::linear_no_bias(
                cfg.encoder_ffn_dim,
                cfg.d_model,
                vb.pp("feed_forward.w2"),
            )?;
            let w3 = candle_nn::linear_no_bias(
                cfg.d_model,
                cfg.encoder_ffn_dim,
                vb.pp("feed_forward.w3"),
            )?;
            (None, None, Some(w1), Some(w2), Some(w3))
        } else {
            let fc1 = candle_nn::linear(cfg.d_model, cfg.encoder_ffn_dim, vb.pp("fc1"))?;
            let fc2 = candle_nn::linear(cfg.encoder_ffn_dim, cfg.d_model, vb.pp("fc2"))?;
            (Some(fc1), Some(fc2), None, None, None)
        };

        Ok(Self {
            self_attn_layer_norm,
            self_attn_rms_norm,
            self_attn,
            final_layer_norm,
            final_rms_norm,
            fc1,
            fc2,
            ffn_w1,
            ffn_w2,
            ffn_w3,
            is_voxtral,
        })
    }

    pub fn forward(&self, x: &Tensor, is_causal: bool) -> Result<Tensor> {
        let residual = x;
        let x = if self.is_voxtral {
            self.self_attn_rms_norm.as_ref().unwrap().forward(x)?
        } else {
            self.self_attn_layer_norm.as_ref().unwrap().forward(x)?
        };
        let x = self.self_attn.forward(&x, is_causal)?;
        let x = (residual + x)?;

        let residual = &x;
        let x = if self.is_voxtral {
            self.final_rms_norm.as_ref().unwrap().forward(&x)?
        } else {
            self.final_layer_norm.as_ref().unwrap().forward(&x)?
        };

        // FFN: Voxtral uses SwiGLU (w1, w2, w3), standard uses GELU (fc1, fc2)
        let x = if self.is_voxtral {
            let w1_out = self.ffn_w1.as_ref().unwrap().forward(&x)?;
            let w3_out = self.ffn_w3.as_ref().unwrap().forward(&x)?;
            // SwiGLU: silu(w1) * w3
            let silu_w1 = candle_nn::ops::silu(&w1_out)?;
            let gated = silu_w1.broadcast_mul(&w3_out)?;
            self.ffn_w2.as_ref().unwrap().forward(&gated)?
        } else {
            let x = self.fc1.as_ref().unwrap().forward(&x)?;
            let x = gelu(&x)?;
            self.fc2.as_ref().unwrap().forward(&x)?
        };

        residual
            .broadcast_add(&x)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

/// Whisper attention with optional causal masking
struct WhisperAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    out_proj: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl WhisperAttention {
    /// Load for standard Whisper format  
    pub fn load(cfg: &super::config::AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Self::load_internal(cfg, vb, false)
    }

    /// Load for Voxtral checkpoint format
    pub fn load_voxtral(cfg: &super::config::AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Self::load_internal(cfg, vb, true)
    }

    fn load_internal(
        cfg: &super::config::AudioEncoderConfig,
        vb: VarBuilder,
        is_voxtral: bool,
    ) -> Result<Self> {
        // Use explicit head_dim from config if available, otherwise compute
        let head_dim = if cfg.head_dim > 0 {
            cfg.head_dim
        } else {
            cfg.d_model / cfg.encoder_attention_heads
        };
        let qkv_proj_dim = cfg.encoder_attention_heads * head_dim;

        let (q_proj, k_proj, v_proj, out_proj) = if is_voxtral {
            // Voxtral uses wq, wk, wv with shape [n_heads * head_dim, d_model]
            // Note: wk has no bias, others have bias
            let q = candle_nn::linear(cfg.d_model, qkv_proj_dim, vb.pp("wq"))?;
            let k = candle_nn::linear_no_bias(cfg.d_model, qkv_proj_dim, vb.pp("wk"))?;
            let v = candle_nn::linear(cfg.d_model, qkv_proj_dim, vb.pp("wv"))?;
            let out = candle_nn::linear(qkv_proj_dim, cfg.d_model, vb.pp("wo"))?;
            (q, k, v, out)
        } else {
            // Standard Whisper uses q_proj, k_proj, v_proj, out_proj
            let q = candle_nn::linear(cfg.d_model, qkv_proj_dim, vb.pp("q_proj"))?;
            let k = candle_nn::linear(cfg.d_model, qkv_proj_dim, vb.pp("k_proj"))?;
            let v = candle_nn::linear(cfg.d_model, qkv_proj_dim, vb.pp("v_proj"))?;
            let out = candle_nn::linear(qkv_proj_dim, cfg.d_model, vb.pp("out_proj"))?;
            (q, k, v, out)
        };

        let scale = (head_dim as f64).powf(-0.5);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: cfg.encoder_attention_heads,
            head_dim,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor, is_causal: bool) -> Result<Tensor> {
        let (bsz, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention
        let q = q
            .reshape((bsz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((bsz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((bsz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        if flash_attention_requested() {
            if let Some(fused_out) =
                try_fused_self_attention(&q, &k, &v, None, self.head_dim, is_causal)?
            {
                let attn_output = fused_out.transpose(1, 2)?.reshape((
                    bsz,
                    seq_len,
                    self.num_heads * self.head_dim,
                ))?;
                return self
                    .out_proj
                    .forward(&attn_output)
                    .map_err(|e| Error::InferenceError(e.to_string()));
            }
        }

        // Scaled dot-product attention
        let q = q.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;
        let k = k.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;
        let v = v.reshape((bsz * self.num_heads, seq_len, self.head_dim))?;

        let attn_weights = q.matmul(&k.transpose(1, 2)?)?;
        let attn_weights = attn_weights.broadcast_mul(&Tensor::from_vec(
            vec![self.scale as f32],
            (1, 1, 1),
            attn_weights.device(),
        )?)?;

        // Apply causal mask if needed
        let attn_weights = if is_causal {
            let mask = create_causal_mask(seq_len, x.device(), attn_weights.dtype())?;
            attn_weights.broadcast_add(&mask)?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax(&attn_weights, 2)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output
            .reshape((bsz, self.num_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((bsz, seq_len, self.num_heads * self.head_dim))?;

        self.out_proj
            .forward(&attn_output)
            .map_err(|e| Error::InferenceError(e.to_string()))
    }
}

fn create_causal_mask(
    seq_len: usize,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[i * seq_len + j] = f32::MIN;
        }
    }
    let mask_tensor = Tensor::from_vec(mask, (seq_len, seq_len), device)?;
    mask_tensor
        .to_dtype(dtype)
        .map_err(|e| Error::InferenceError(e.to_string()))
}

fn argmax(logits: &Tensor) -> Result<u32> {
    let logits = logits.to_vec1::<f32>()?;
    let mut max_idx = 0;
    let mut max_val = logits[0];
    for (i, &v) in logits.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    Ok(max_idx as u32)
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

fn resample_audio(audio: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(audio.to_vec());
    }

    let ratio = to_rate as f64 / from_rate as f64;
    let new_len = (audio.len() as f64 * ratio) as usize;
    let mut resampled = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_idx = i as f64 / ratio;
        let src_idx_floor = src_idx.floor() as usize;
        let src_idx_ceil = (src_idx_floor + 1).min(audio.len() - 1);
        let frac = src_idx - src_idx_floor as f64;

        let val = audio[src_idx_floor] as f64 * (1.0 - frac) + audio[src_idx_ceil] as f64 * frac;
        resampled.push(val as f32);
    }

    Ok(resampled)
}

fn load_weights<'a>(
    model_dir: &'a Path,
    dtype: DType,
    device: &'a DeviceProfile,
) -> Result<VarBuilder<'a>> {
    // Voxtral uses consolidated.safetensors (single file)
    let consolidated_path = model_dir.join("consolidated.safetensors");
    if consolidated_path.exists() {
        info!("Loading Voxtral from consolidated.safetensors");
        unsafe {
            return VarBuilder::from_mmaped_safetensors(
                &[consolidated_path],
                dtype,
                &device.device,
            )
            .map_err(|e| Error::ModelLoadError(format!("Failed to load weights: {}", e)));
        }
    }

    let index_path = model_dir.join("model.safetensors.index.json");

    if index_path.exists() {
        let index_data = std::fs::read_to_string(&index_path)
            .map_err(|e| Error::ModelLoadError(format!("Failed to read index: {}", e)))?;
        let index: serde_json::Value = serde_json::from_str(&index_data)
            .map_err(|e| Error::ModelLoadError(format!("Failed to parse index: {}", e)))?;

        let weight_map = index
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| Error::InvalidInput("Invalid weight map".to_string()))?;

        let mut shard_files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        shard_files.sort();
        shard_files.dedup();

        let shard_paths: Vec<std::path::PathBuf> =
            shard_files.iter().map(|f| model_dir.join(f)).collect();

        info!("Loading Voxtral with {} shard files", shard_paths.len());

        unsafe {
            VarBuilder::from_mmaped_safetensors(&shard_paths, dtype, &device.device)
                .map_err(|e| Error::ModelLoadError(format!("Failed to load shards: {}", e)))
        }
    } else {
        let weights_path = model_dir.join("model.safetensors");
        unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, &device.device)
                .map_err(|e| Error::ModelLoadError(format!("Failed to load weights: {}", e)))
        }
    }
}

fn select_voxtral_dtype(device: &DeviceProfile, checkpoint_dtype: Option<DType>) -> DType {
    device.select_model_dtype_with_checkpoint(ModelFamily::Voxtral, checkpoint_dtype)
}

/// GELU activation function
fn gelu(x: &Tensor) -> Result<Tensor> {
    let coeff = 0.044715f32;
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let dtype = x.dtype();
    let x_f32 = x.to_dtype(candle_core::DType::F32)?;
    let x3 = x_f32.powf(3.0)?;
    let coeff_t = Tensor::from_vec(vec![coeff], (1,), x.device())?;
    let x3 = x3.broadcast_mul(&coeff_t)?;
    let sqrt_t = Tensor::from_vec(vec![sqrt_2_over_pi], (1,), x.device())?;
    let inner = (&x_f32 + x3)?.broadcast_mul(&sqrt_t)?;
    let tanh = inner.tanh()?;
    let one = Tensor::from_vec(vec![1.0f32], (1,), x.device())?;
    let half = Tensor::from_vec(vec![0.5f32], (1,), x.device())?;
    let out = x_f32.broadcast_mul(&one.broadcast_add(&tanh)?)?;
    let out = out.broadcast_mul(&half)?;
    out.to_dtype(dtype)
        .map_err(|e| Error::InferenceError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::select_voxtral_dtype;
    use crate::backends::{DeviceCapabilities, DeviceKind, DeviceProfile};
    use candle_core::{DType, Device};

    #[test]
    fn voxtral_dense_cuda_uses_checkpoint_dtype() {
        let profile = DeviceProfile {
            device: Device::Cpu,
            kind: DeviceKind::Cuda,
            capabilities: DeviceCapabilities {
                supports_bf16: true,
                supports_f16: true,
                ..Default::default()
            },
            memory_pool: None,
        };

        assert_eq!(select_voxtral_dtype(&profile, Some(DType::F16)), DType::F16);
    }
}
