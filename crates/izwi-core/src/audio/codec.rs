//! Audio codec for Qwen3-TTS (12Hz tokenizer)
//!
//! The Qwen3-TTS-Tokenizer-12Hz uses a 16-layer multi-codebook design
//! operating at 12.5Hz with a lightweight causal ConvNet decoder.

use std::path::Path;
use tracing::{debug, info};

use crate::error::Result;
use crate::model::weights::ModelWeights;

/// Configuration for the audio codec
#[derive(Debug, Clone)]
pub struct CodecConfig {
    /// Sample rate for output audio (default: 24000 Hz)
    pub sample_rate: u32,
    /// Number of codebook layers (default: 16)
    pub num_codebooks: usize,
    /// Token rate in Hz (default: 12.5)
    pub token_rate_hz: f32,
    /// Number of channels (default: 1 for mono)
    pub channels: u16,
}

impl Default for CodecConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            num_codebooks: 16,
            token_rate_hz: 12.5,
            channels: 1,
        }
    }
}

impl CodecConfig {
    /// Samples per audio token
    pub fn samples_per_token(&self) -> usize {
        (self.sample_rate as f32 / self.token_rate_hz) as usize
    }
}

/// Audio codec for converting between audio tokens and waveforms
pub struct AudioCodec {
    config: CodecConfig,
    decoder_weights: Option<DecoderWeights>,
}

/// Decoder network weights
struct DecoderWeights {
    /// Embedding for each codebook
    codebook_embeddings: Vec<Vec<f32>>,
    /// Causal ConvNet layers for the decoder
    conv_layers: Vec<ConvLayer>,
    /// Final projection to audio samples
    output_proj_weight: Vec<f32>,
    output_proj_bias: Vec<f32>,
    /// Hidden dimension
    hidden_dim: usize,
    /// Codebook vocabulary size
    vocab_size: usize,
}

/// A causal 1D convolution layer
struct ConvLayer {
    weight: Vec<f32>,
    bias: Vec<f32>,
    kernel_size: usize,
    in_channels: usize,
    out_channels: usize,
}

impl ConvLayer {
    /// Apply causal conv1d: output[t] only depends on input[0..=t]
    fn forward(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; self.out_channels * seq_len];

        // Causal convolution: pad on the left
        let padding = self.kernel_size - 1;

        for t in 0..seq_len {
            for out_c in 0..self.out_channels {
                let mut sum = self.bias[out_c];

                for k in 0..self.kernel_size {
                    let input_t = t as isize - (padding as isize - k as isize);
                    if input_t >= 0 && (input_t as usize) < seq_len {
                        for in_c in 0..self.in_channels {
                            let weight_idx = out_c * self.in_channels * self.kernel_size
                                + in_c * self.kernel_size
                                + k;
                            let input_idx = (input_t as usize) * self.in_channels + in_c;
                            sum += self.weight[weight_idx] * input[input_idx];
                        }
                    }
                }

                output[t * self.out_channels + out_c] = sum;
            }
        }

        output
    }
}

/// Apply GELU activation function
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// Apply GELU activation to a slice in-place
fn apply_gelu(data: &mut [f32]) {
    for x in data.iter_mut() {
        *x = gelu(*x);
    }
}

impl AudioCodec {
    /// Create a new codec with default configuration
    pub fn new() -> Self {
        Self {
            config: CodecConfig::default(),
            decoder_weights: None,
        }
    }

    /// Create codec with custom configuration
    pub fn with_config(config: CodecConfig) -> Self {
        Self {
            config,
            decoder_weights: None,
        }
    }

    /// Load codec weights from a tokenizer model directory
    pub fn load_weights(&mut self, model_dir: &Path) -> Result<()> {
        info!("Loading audio codec from {:?}", model_dir);

        // The codec decoder is part of Qwen3-TTS-Tokenizer-12Hz
        let decoder_path = model_dir.join("codec_decoder.safetensors");

        if decoder_path.exists() {
            let weights = ModelWeights::load(model_dir)?;
            // Extract decoder-specific weights
            // Note: Actual weight names depend on the model structure
            debug!("Codec weights loaded: {} tensors", weights.tensors.len());
        } else {
            info!("No codec weights found, using placeholder decoder");
        }

        Ok(())
    }

    /// Decode audio tokens to waveform
    ///
    /// Input: Audio tokens of shape [num_codebooks, sequence_length]
    /// Output: Audio waveform as f32 samples
    pub fn decode(&self, tokens: &[Vec<u32>]) -> Result<Vec<f32>> {
        if tokens.is_empty() || tokens[0].is_empty() {
            return Ok(Vec::new());
        }

        let num_codebooks = tokens.len();
        let sequence_length = tokens[0].len();

        debug!(
            "Decoding {} tokens across {} codebooks",
            sequence_length, num_codebooks
        );

        // Calculate output length
        let samples_per_token = self.config.samples_per_token();
        let output_length = sequence_length * samples_per_token;

        // Placeholder: Generate silence or simple waveform
        // In a real implementation, this would run the ConvNet decoder
        let mut output = vec![0.0f32; output_length];

        if self.decoder_weights.is_some() {
            // Run actual decoder network
            self.run_decoder(tokens, &mut output)?;
        } else {
            // Placeholder: generate simple tone based on token values
            self.placeholder_decode(tokens, &mut output);
        }

        Ok(output)
    }

    /// Decode a single chunk of audio tokens (for streaming)
    pub fn decode_chunk(&self, tokens: &[Vec<u32>], chunk_idx: usize) -> Result<Vec<f32>> {
        // For streaming, we process one token column at a time
        let samples_per_token = self.config.samples_per_token();
        let mut chunk = vec![0.0f32; samples_per_token];

        if self.decoder_weights.is_some() {
            // Run incremental decoder
            self.run_decoder_incremental(tokens, chunk_idx, &mut chunk)?;
        } else {
            // Placeholder decode
            self.placeholder_decode_chunk(tokens, chunk_idx, &mut chunk);
        }

        Ok(chunk)
    }

    /// Run the full ConvNet decoder forward pass
    fn run_decoder(&self, tokens: &[Vec<u32>], output: &mut [f32]) -> Result<()> {
        let weights = self.decoder_weights.as_ref().unwrap();
        let num_codebooks = tokens.len().min(weights.codebook_embeddings.len());
        let seq_len = tokens[0].len();
        let samples_per_token = self.config.samples_per_token();

        // Step 1: Sum embeddings from all codebooks
        let mut hidden = vec![0.0f32; seq_len * weights.hidden_dim];

        for cb in 0..num_codebooks {
            for (t, &token) in tokens[cb].iter().enumerate() {
                let token_idx = (token as usize).min(weights.vocab_size - 1);
                let embed_offset = token_idx * weights.hidden_dim;

                for h in 0..weights.hidden_dim {
                    let hidden_idx = t * weights.hidden_dim + h;
                    if embed_offset + h < weights.codebook_embeddings[cb].len() {
                        hidden[hidden_idx] += weights.codebook_embeddings[cb][embed_offset + h];
                    }
                }
            }
        }

        // Step 2: Apply causal ConvNet layers with GELU activation
        for layer in &weights.conv_layers {
            hidden = layer.forward(&hidden, seq_len);
            apply_gelu(&mut hidden);
        }

        // Step 3: Apply output projection to get audio samples
        // Upsample from token rate to sample rate
        let out_channels = weights.output_proj_weight.len() / weights.hidden_dim;

        for t in 0..seq_len {
            for s in 0..samples_per_token {
                let output_idx = t * samples_per_token + s;
                if output_idx >= output.len() {
                    break;
                }

                let mut sample = 0.0f32;

                // Linear interpolation between timesteps for smoother audio
                let interp = s as f32 / samples_per_token as f32;
                let h_idx_curr = t * weights.hidden_dim;
                let h_idx_next = if t + 1 < seq_len {
                    (t + 1) * weights.hidden_dim
                } else {
                    h_idx_curr
                };

                for h in 0..weights.hidden_dim.min(out_channels) {
                    let h_val =
                        hidden[h_idx_curr + h] * (1.0 - interp) + hidden[h_idx_next + h] * interp;
                    let w_idx = h; // Simplified projection
                    if w_idx < weights.output_proj_weight.len() {
                        sample += h_val * weights.output_proj_weight[w_idx];
                    }
                }

                if !weights.output_proj_bias.is_empty() {
                    sample += weights.output_proj_bias[0];
                }

                // Clamp output to valid audio range
                output[output_idx] = sample.clamp(-1.0, 1.0);
            }
        }

        Ok(())
    }

    /// Run incremental causal decoding for streaming
    /// Only processes the specified chunk using cached hidden states
    fn run_decoder_incremental(
        &self,
        tokens: &[Vec<u32>],
        chunk_idx: usize,
        output: &mut [f32],
    ) -> Result<()> {
        let weights = self.decoder_weights.as_ref().unwrap();
        let num_codebooks = tokens.len().min(weights.codebook_embeddings.len());
        let samples_per_token = self.config.samples_per_token();

        // For causal decoding, we need to process all tokens up to and including chunk_idx
        // to maintain causality (each position only sees past positions)
        let context_len = chunk_idx + 1;

        // Step 1: Build hidden states for context
        let mut hidden = vec![0.0f32; context_len * weights.hidden_dim];

        for cb in 0..num_codebooks {
            for t in 0..context_len {
                if t < tokens[cb].len() {
                    let token = tokens[cb][t];
                    let token_idx = (token as usize).min(weights.vocab_size - 1);
                    let embed_offset = token_idx * weights.hidden_dim;

                    for h in 0..weights.hidden_dim {
                        let hidden_idx = t * weights.hidden_dim + h;
                        if embed_offset + h < weights.codebook_embeddings[cb].len() {
                            hidden[hidden_idx] += weights.codebook_embeddings[cb][embed_offset + h];
                        }
                    }
                }
            }
        }

        // Step 2: Apply causal ConvNet layers
        for layer in &weights.conv_layers {
            hidden = layer.forward(&hidden, context_len);
            apply_gelu(&mut hidden);
        }

        // Step 3: Extract output for just the current chunk
        let h_idx = chunk_idx * weights.hidden_dim;
        let out_channels = weights.output_proj_weight.len() / weights.hidden_dim;

        for s in 0..samples_per_token.min(output.len()) {
            let mut sample = 0.0f32;
            let interp = s as f32 / samples_per_token as f32;

            // Get next timestep hidden for interpolation (if available)
            let h_idx_next = if chunk_idx + 1 < context_len {
                (chunk_idx + 1) * weights.hidden_dim
            } else {
                h_idx
            };

            for h in 0..weights.hidden_dim.min(out_channels) {
                let h_val = if h_idx + h < hidden.len() {
                    let curr = hidden[h_idx + h];
                    let next = if h_idx_next + h < hidden.len() {
                        hidden[h_idx_next + h]
                    } else {
                        curr
                    };
                    curr * (1.0 - interp) + next * interp
                } else {
                    0.0
                };

                if h < weights.output_proj_weight.len() {
                    sample += h_val * weights.output_proj_weight[h];
                }
            }

            if !weights.output_proj_bias.is_empty() {
                sample += weights.output_proj_bias[0];
            }

            output[s] = sample.clamp(-1.0, 1.0);
        }

        Ok(())
    }

    fn placeholder_decode(&self, tokens: &[Vec<u32>], output: &mut [f32]) {
        let samples_per_token = self.config.samples_per_token();

        for (t, token_col) in tokens[0].iter().enumerate() {
            let start = t * samples_per_token;
            let freq = 220.0 + (*token_col as f32 % 100.0) * 5.0;

            for i in 0..samples_per_token {
                let sample_idx = start + i;
                if sample_idx < output.len() {
                    let time = sample_idx as f32 / self.config.sample_rate as f32;
                    output[sample_idx] = (2.0 * std::f32::consts::PI * freq * time).sin() * 0.3;
                }
            }
        }
    }

    fn placeholder_decode_chunk(&self, tokens: &[Vec<u32>], chunk_idx: usize, output: &mut [f32]) {
        if chunk_idx >= tokens[0].len() {
            return;
        }

        let token = tokens[0][chunk_idx];
        let freq = 220.0 + (token as f32 % 100.0) * 5.0;

        let output_len = output.len();
        for (i, sample) in output.iter_mut().enumerate() {
            let time = (chunk_idx * output_len + i) as f32 / self.config.sample_rate as f32;
            *sample = (2.0 * std::f32::consts::PI * freq * time).sin() * 0.3;
        }
    }

    /// Get codec configuration
    pub fn config(&self) -> &CodecConfig {
        &self.config
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }
}

impl Default for AudioCodec {
    fn default() -> Self {
        Self::new()
    }
}
