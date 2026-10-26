//! Chatterbox's fixed geometry. Upstream ships no config files: the values
//! are the constructor defaults at resemble-ai/chatterbox@5de7a54a
//! (`t3/modules/t3_config.py`, `t3/llama_configs.py`, `s3gen/s3gen.py`,
//! `s3gen/flow.py`, `s3gen/hifigan.py`) for the English checkpoint at HF
//! revision 5bb1f6ee.

/// Files the English loader reads, all from the model directory.
pub const T3_WEIGHTS_FILE: &str = "t3_cfg.safetensors";
pub const S3GEN_WEIGHTS_FILE: &str = "s3gen.safetensors";
pub const VOICE_ENCODER_WEIGHTS_FILE: &str = "ve.safetensors";
pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const DEFAULT_VOICE_FILE: &str = "conds.pt";

/// Text tokens one request may carry (framing included); longer text is
/// split across requests. Well inside the learned 2,050-position table.
pub const MAX_TEXT_TOKENS: usize = 1024;

/// Positions one T3 sequence may reach: 34 conditioning positions, the
/// longest text, both start tokens and the 1,000-token speech budget.
pub const STATE_CONTEXT_TOKENS: usize = 34 + MAX_TEXT_TOKENS + 2 + 1000;

/// Output sample rate of S3Gen's vocoder.
pub const S3GEN_SAMPLE_RATE: u32 = 24_000;
/// Sample rate of the S3 speech tokenizer and the voice encoder.
pub const S3_TOKENIZER_SAMPLE_RATE: u32 = 16_000;
/// Speech tokens per second of audio.
pub const SPEECH_TOKEN_RATE_HZ: u32 = 25;
/// Mel frames S3Gen produces per speech token.
pub const MEL_FRAMES_PER_TOKEN: usize = 2;
/// Waveform samples per mel frame (`8 * 5 * 3` upsampling × iSTFT hop 4).
pub const SAMPLES_PER_MEL_FRAME: usize = 480;

#[derive(Debug, Clone, PartialEq)]
pub struct T3Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_factor: f64,
    pub rope_low_freq_factor: f64,
    pub rope_high_freq_factor: f64,
    pub rope_original_max_position_embeddings: usize,
    pub text_vocab_size: usize,
    pub speech_vocab_size: usize,
    pub start_text_token: u32,
    pub stop_text_token: u32,
    pub start_speech_token: u32,
    pub stop_speech_token: u32,
    /// Learned text and speech position tables: `max_text_tokens + 2` and
    /// `max_speech_tokens + 4` rows.
    pub text_positions: usize,
    pub speech_positions: usize,
    pub speaker_embed_size: usize,
    /// Reference speech tokens the perceiver reads (6 s at 25 Hz).
    pub speech_cond_prompt_len: usize,
    pub perceiver_queries: usize,
    pub perceiver_heads: usize,
}

impl T3Config {
    /// Llama-520M with the 704-token English text vocabulary.
    pub fn english() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 30,
            num_attention_heads: 16,
            head_dim: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            rope_factor: 8.0,
            rope_low_freq_factor: 1.0,
            rope_high_freq_factor: 4.0,
            rope_original_max_position_embeddings: 8192,
            text_vocab_size: 704,
            speech_vocab_size: 8194,
            start_text_token: 255,
            stop_text_token: 0,
            start_speech_token: 6561,
            stop_speech_token: 6562,
            text_positions: 2050,
            speech_positions: 4100,
            speaker_embed_size: 256,
            speech_cond_prompt_len: 150,
            perceiver_queries: 32,
            perceiver_heads: 4,
        }
    }

    /// Conditioning positions ahead of the text: speaker, perceiver queries,
    /// exaggeration.
    pub fn conditioning_len(&self) -> usize {
        1 + self.perceiver_queries + 1
    }

    /// Speech codes S3Gen can render; ids from here up are BOS, EOS and
    /// unused rows.
    pub fn acoustic_code_count(&self) -> u32 {
        self.start_speech_token
    }

    pub fn rope_scaling(&self) -> crate::models::shared::rope::RopeScaling {
        crate::models::shared::rope::RopeScaling::Llama3(
            crate::models::shared::rope::Llama3RopeScaling {
                factor: self.rope_factor,
                low_freq_factor: self.rope_low_freq_factor,
                high_freq_factor: self.rope_high_freq_factor,
                original_max_position_embeddings: self.rope_original_max_position_embeddings,
            },
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct S3GenConfig {
    /// Speech-token vocabulary the flow embeds (codes `0..6561`).
    pub vocab_size: usize,
    pub encoder_dim: usize,
    pub encoder_heads: usize,
    pub encoder_ffn: usize,
    pub encoder_layers: usize,
    pub up_encoder_layers: usize,
    pub pre_lookahead_len: usize,
    pub mel_bins: usize,
    pub speaker_embed_size: usize,
    /// Flow-matching estimator (causal U-Net) geometry.
    pub estimator_channels: usize,
    pub estimator_in_channels: usize,
    pub estimator_heads: usize,
    pub estimator_head_dim: usize,
    pub estimator_blocks: usize,
    pub estimator_mid_blocks: usize,
    pub time_embed_dim: usize,
    pub cfm_steps: usize,
    pub cfm_cfg_rate: f32,
}

impl S3GenConfig {
    pub fn current() -> Self {
        Self {
            vocab_size: 6561,
            encoder_dim: 512,
            encoder_heads: 8,
            encoder_ffn: 2048,
            encoder_layers: 6,
            up_encoder_layers: 4,
            pre_lookahead_len: 3,
            mel_bins: 80,
            speaker_embed_size: 192,
            estimator_channels: 256,
            estimator_in_channels: 320,
            estimator_heads: 8,
            estimator_head_dim: 64,
            estimator_blocks: 4,
            estimator_mid_blocks: 12,
            time_embed_dim: 1024,
            cfm_steps: 10,
            cfm_cfg_rate: 0.7,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HiftConfig {
    pub mel_bins: usize,
    pub base_channels: usize,
    pub harmonics: usize,
    pub upsample_rates: [usize; 3],
    pub upsample_kernel_sizes: [usize; 3],
    pub resblock_kernel_sizes: [usize; 3],
    pub source_resblock_kernel_sizes: [usize; 3],
    pub resblock_dilations: [usize; 3],
    pub istft_n_fft: usize,
    pub istft_hop: usize,
    pub nsf_alpha: f32,
    pub nsf_sigma: f32,
    pub nsf_voiced_threshold: f32,
    pub lrelu_slope: f64,
    pub audio_limit: f32,
}

impl HiftConfig {
    pub fn current() -> Self {
        Self {
            mel_bins: 80,
            base_channels: 512,
            harmonics: 8,
            upsample_rates: [8, 5, 3],
            upsample_kernel_sizes: [16, 11, 7],
            resblock_kernel_sizes: [3, 7, 11],
            source_resblock_kernel_sizes: [7, 7, 11],
            resblock_dilations: [1, 3, 5],
            istft_n_fft: 16,
            istft_hop: 4,
            nsf_alpha: 0.1,
            nsf_sigma: 0.003,
            nsf_voiced_threshold: 10.0,
            lrelu_slope: 0.1,
            audio_limit: 0.99,
        }
    }

    /// Waveform samples per mel frame.
    pub fn upsample_factor(&self) -> usize {
        self.upsample_rates.iter().product::<usize>() * self.istft_hop
    }
}

/// Upstream's generation defaults (`tts.py`, `ChatterboxTTS.generate`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChatterboxSamplingDefaults {
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub min_p: f32,
    pub top_p: f32,
    pub exaggeration: f32,
    pub cfg_weight: f32,
    /// The wrapper's hard-coded `max_new_tokens`, about 40 s of speech.
    pub max_speech_tokens: usize,
}

/// A speech-token budget for `text` when the request sets none: 50 tokens
/// plus two per character (12.5 characters a second at 25 tokens a second,
/// slower than read speech), capped at the per-utterance limit. It bounds
/// the runaway generations upstream sees (resemble-ai/chatterbox#531, the
/// budget its PR #570 proposes) without cutting real speech.
pub fn chatterbox_auto_max_speech_tokens(text: &str) -> usize {
    let chars = text.chars().filter(|ch| !ch.is_whitespace()).count()
        + text.split_whitespace().count().saturating_sub(1);
    (50 + 2 * chars).min(CHATTERBOX_DEFAULTS.max_speech_tokens)
}

pub const CHATTERBOX_DEFAULTS: ChatterboxSamplingDefaults = ChatterboxSamplingDefaults {
    temperature: 0.8,
    repetition_penalty: 1.2,
    min_p: 0.05,
    top_p: 1.0,
    exaggeration: 0.5,
    cfg_weight: 0.5,
    max_speech_tokens: 1000,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_budget_scales_with_text_and_caps() {
        assert_eq!(chatterbox_auto_max_speech_tokens("hi"), 54);
        // Runs of whitespace count once.
        assert_eq!(chatterbox_auto_max_speech_tokens("a  b"), 56);
        assert_eq!(
            chatterbox_auto_max_speech_tokens(&"word ".repeat(400)),
            1000
        );
    }

    #[test]
    fn geometry_matches_the_published_tables() {
        let t3 = T3Config::english();
        assert_eq!(t3.conditioning_len(), 34);
        assert_eq!(t3.num_attention_heads * t3.head_dim, t3.hidden_size);
        let hift = HiftConfig::current();
        assert_eq!(hift.upsample_factor(), SAMPLES_PER_MEL_FRAME);
        assert_eq!(
            SAMPLES_PER_MEL_FRAME * MEL_FRAMES_PER_TOKEN * SPEECH_TOKEN_RATE_HZ as usize,
            S3GEN_SAMPLE_RATE as usize
        );
    }
}
