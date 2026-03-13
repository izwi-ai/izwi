//! Model information and metadata

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Available TTS model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelVariant {
    /// 0.6B parameter base model
    #[serde(rename = "Qwen3-TTS-12Hz-0.6B-Base")]
    Qwen3Tts12Hz06BBase,
    /// 0.6B parameter base model (MLX 4-bit)
    #[serde(rename = "Qwen3-TTS-12Hz-0.6B-Base-4bit")]
    Qwen3Tts12Hz06BBase4Bit,
    /// 0.6B parameter base model (MLX 8-bit)
    #[serde(rename = "Qwen3-TTS-12Hz-0.6B-Base-8bit")]
    Qwen3Tts12Hz06BBase8Bit,
    /// 0.6B parameter base model (MLX bf16)
    #[serde(rename = "Qwen3-TTS-12Hz-0.6B-Base-bf16")]
    Qwen3Tts12Hz06BBaseBf16,
    /// 0.6B parameter custom voice model
    #[serde(rename = "Qwen3-TTS-12Hz-0.6B-CustomVoice")]
    Qwen3Tts12Hz06BCustomVoice,
    /// 0.6B parameter custom voice model (MLX 4-bit)
    #[serde(rename = "Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit")]
    Qwen3Tts12Hz06BCustomVoice4Bit,
    /// 0.6B parameter custom voice model (MLX 8-bit)
    #[serde(rename = "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit")]
    Qwen3Tts12Hz06BCustomVoice8Bit,
    /// 0.6B parameter custom voice model (MLX bf16)
    #[serde(rename = "Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16")]
    Qwen3Tts12Hz06BCustomVoiceBf16,
    /// 1.7B parameter base model
    #[serde(rename = "Qwen3-TTS-12Hz-1.7B-Base")]
    Qwen3Tts12Hz17BBase,
    /// 1.7B parameter base model (MLX 4-bit)
    #[serde(rename = "Qwen3-TTS-12Hz-1.7B-Base-4bit")]
    Qwen3Tts12Hz17BBase4Bit,
    /// 1.7B parameter custom voice model  
    #[serde(rename = "Qwen3-TTS-12Hz-1.7B-CustomVoice")]
    Qwen3Tts12Hz17BCustomVoice,
    /// 1.7B parameter custom voice model (MLX 4-bit)
    #[serde(rename = "Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit")]
    Qwen3Tts12Hz17BCustomVoice4Bit,
    /// 1.7B parameter voice design model
    #[serde(rename = "Qwen3-TTS-12Hz-1.7B-VoiceDesign")]
    Qwen3Tts12Hz17BVoiceDesign,
    /// 1.7B parameter voice design model (MLX 4-bit)
    #[serde(rename = "Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit")]
    Qwen3Tts12Hz17BVoiceDesign4Bit,
    /// 1.7B parameter voice design model (MLX 8-bit)
    #[serde(rename = "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit")]
    Qwen3Tts12Hz17BVoiceDesign8Bit,
    /// 1.7B parameter voice design model (MLX bf16)
    #[serde(rename = "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")]
    Qwen3Tts12Hz17BVoiceDesignBf16,
    /// Tokenizer for 12Hz codec
    #[serde(rename = "Qwen3-TTS-Tokenizer-12Hz")]
    Qwen3TtsTokenizer12Hz,
    /// LFM2.5-Audio 1.5B model from Liquid AI
    #[serde(rename = "LFM2.5-Audio-1.5B")]
    Lfm25Audio15B,
    /// LFM2.5-Audio 1.5B model from Liquid AI (MLX 4-bit)
    #[serde(rename = "LFM2.5-Audio-1.5B-4bit")]
    Lfm25Audio15B4Bit,
    /// LFM2.5 1.2B Instruct model from Liquid AI (GGUF Q4_K_M)
    #[serde(
        rename = "LFM2.5-1.2B-Instruct-GGUF",
        alias = "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
        alias = "LFM2.5-1.2B-Instruct-Q4_K_M.gguf"
    )]
    Lfm2512BInstructGguf,
    /// LFM2.5 1.2B Thinking model from Liquid AI (GGUF Q4_K_M)
    #[serde(
        rename = "LFM2.5-1.2B-Thinking-GGUF",
        alias = "LiquidAI/LFM2.5-1.2B-Thinking-GGUF",
        alias = "LFM2.5-1.2B-Thinking-Q4_K_M.gguf"
    )]
    Lfm2512BThinkingGguf,
    /// Kokoro-82M TTS model from hexgrad
    #[serde(rename = "Kokoro-82M")]
    Kokoro82M,
    /// Qwen3-ASR 0.6B model
    #[serde(rename = "Qwen3-ASR-0.6B")]
    Qwen3Asr06B,
    /// Qwen3-ASR 0.6B model (MLX 4-bit)
    #[serde(rename = "Qwen3-ASR-0.6B-4bit")]
    Qwen3Asr06B4Bit,
    /// Qwen3-ASR 0.6B model (MLX 8-bit)
    #[serde(rename = "Qwen3-ASR-0.6B-8bit")]
    Qwen3Asr06B8Bit,
    /// Qwen3-ASR 0.6B model (MLX bf16)
    #[serde(rename = "Qwen3-ASR-0.6B-bf16")]
    Qwen3Asr06BBf16,
    /// Qwen3-ASR 1.7B model
    #[serde(rename = "Qwen3-ASR-1.7B")]
    Qwen3Asr17B,
    /// Qwen3-ASR 1.7B model (MLX 4-bit)
    #[serde(rename = "Qwen3-ASR-1.7B-4bit")]
    Qwen3Asr17B4Bit,
    /// Qwen3-ASR 1.7B model (MLX 8-bit)
    #[serde(rename = "Qwen3-ASR-1.7B-8bit")]
    Qwen3Asr17B8Bit,
    /// Qwen3-ASR 1.7B model (MLX bf16)
    #[serde(rename = "Qwen3-ASR-1.7B-bf16")]
    Qwen3Asr17BBf16,
    /// Parakeet TDT 0.6B v2 ASR model (.nemo)
    #[serde(rename = "Parakeet-TDT-0.6B-v2", alias = "Parakeet-TDT-0.6B-v2-4bit")]
    ParakeetTdt06BV2,
    /// Parakeet TDT 0.6B v3 ASR model (.nemo)
    #[serde(rename = "Parakeet-TDT-0.6B-v3", alias = "Parakeet-TDT-0.6B-v3-4bit")]
    ParakeetTdt06BV3,
    /// Whisper Large v3 Turbo ASR model
    #[serde(
        rename = "Whisper-Large-v3-Turbo",
        alias = "whisper-large-v3-turbo",
        alias = "openai/whisper-large-v3-turbo"
    )]
    WhisperLargeV3Turbo,
    /// Streaming Sortformer 4-speaker diarization model (.nemo)
    #[serde(rename = "diar_streaming_sortformer_4spk-v2.1")]
    DiarStreamingSortformer4SpkV21,
    /// Qwen3 0.6B text model
    #[serde(rename = "Qwen3-0.6B")]
    Qwen306B,
    /// Qwen3 0.6B text model (MLX 4-bit)
    #[serde(rename = "Qwen3-0.6B-4bit")]
    Qwen306B4Bit,
    /// Qwen3 1.7B text model
    #[serde(rename = "Qwen3-1.7B")]
    Qwen317B,
    /// Qwen3 1.7B text model (MLX 4-bit)
    #[serde(rename = "Qwen3-1.7B-4bit")]
    Qwen317B4Bit,
    /// Qwen3 0.6B text model (GGUF Q8_0)
    #[serde(rename = "Qwen3-0.6B-GGUF")]
    Qwen306BGguf,
    /// Qwen3 1.7B text model (GGUF Q8_0)
    #[serde(rename = "Qwen3-1.7B-GGUF")]
    Qwen317BGguf,
    /// Qwen3 4B text model (GGUF Q4_K_M)
    #[serde(rename = "Qwen3-4B-GGUF")]
    Qwen34BGguf,
    /// Qwen3 8B text model (GGUF Q4_K_M)
    #[serde(rename = "Qwen3-8B-GGUF")]
    Qwen38BGguf,
    /// Qwen3 14B text model (GGUF Q4_K_M)
    #[serde(rename = "Qwen3-14B-GGUF")]
    Qwen314BGguf,
    /// Qwen3.5 0.8B multimodal chat model (GGUF Q4_K_M only)
    #[serde(
        rename = "Qwen3.5-0.8B",
        alias = "Qwen3.5-0.8B-GGUF",
        alias = "unsloth/Qwen3.5-0.8B-GGUF",
        alias = "Qwen3.5-0.8B-Q4_K_M.gguf"
    )]
    Qwen3508BGguf,
    /// Qwen3.5 2B multimodal chat model (GGUF Q4_K_M only)
    #[serde(
        rename = "Qwen3.5-2B",
        alias = "Qwen3.5-2B-GGUF",
        alias = "unsloth/Qwen3.5-2B-GGUF",
        alias = "Qwen3.5-2B-Q4_K_M.gguf"
    )]
    Qwen352BGguf,
    /// Qwen3.5 4B multimodal chat model (GGUF Q4_K_M only)
    #[serde(
        rename = "Qwen3.5-4B",
        alias = "Qwen3.5-4B-GGUF",
        alias = "unsloth/Qwen3.5-4B-GGUF",
        alias = "Qwen3.5-4B-Q4_K_M.gguf"
    )]
    Qwen354BGguf,
    /// Qwen3.5 9B multimodal chat model (GGUF Q4_K_M only)
    #[serde(
        rename = "Qwen3.5-9B",
        alias = "Qwen3.5-9B-GGUF",
        alias = "unsloth/Qwen3.5-9B-GGUF",
        alias = "Qwen3.5-9B-Q4_K_M.gguf"
    )]
    Qwen359BGguf,
    /// Gemma 3 1B instruction-tuned chat model
    #[serde(rename = "Gemma-3-1b-it")]
    Gemma31BIt,
    /// Gemma 3 4B instruction-tuned chat model
    #[serde(rename = "Gemma-3-4b-it")]
    Gemma34BIt,
    /// Qwen3-ForcedAligner 0.6B model
    #[serde(rename = "Qwen3-ForcedAligner-0.6B")]
    Qwen3ForcedAligner06B,
    /// Qwen3-ForcedAligner 0.6B model (MLX 4-bit)
    #[serde(rename = "Qwen3-ForcedAligner-0.6B-4bit")]
    Qwen3ForcedAligner06B4Bit,
    /// Voxtral Mini 4B Realtime model from Mistral AI
    #[serde(rename = "Voxtral-Mini-4B-Realtime-2602")]
    VoxtralMini4BRealtime2602,
}

impl ModelVariant {
    /// Official Qwen3-TTS generation limit from upstream generation configs
    /// (`max_new_tokens` in Hugging Face `generation_config.json`).
    pub const QWEN3_TTS_MAX_OUTPUT_FRAMES: usize = 8192;
    /// Qwen3-TTS codec frame rate encoded in model variant naming ("12Hz").
    pub const QWEN3_TTS_FRAME_RATE_HZ: f32 = 12.0;
    pub const QWEN_CUSTOMVOICE_BUILT_IN_VOICE_COUNT: usize = 9;
    pub const LFM2_AUDIO_BUILT_IN_VOICE_COUNT: usize = 4;
    pub const KOKORO_BUILT_IN_VOICE_COUNT: usize = 54;

    /// Get HuggingFace repository ID
    pub fn repo_id(&self) -> &'static str {
        match self {
            Self::Qwen3Tts12Hz06BBase => "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            Self::Qwen3Tts12Hz06BBase4Bit => "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-4bit",
            Self::Qwen3Tts12Hz06BBase8Bit => "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
            Self::Qwen3Tts12Hz06BBaseBf16 => "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            Self::Qwen3Tts12Hz06BCustomVoice => "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            Self::Qwen3Tts12Hz06BCustomVoice4Bit => {
                "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
            }
            Self::Qwen3Tts12Hz06BCustomVoice8Bit => {
                "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
            }
            Self::Qwen3Tts12Hz06BCustomVoiceBf16 => {
                "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"
            }
            Self::Qwen3Tts12Hz17BBase => "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            Self::Qwen3Tts12Hz17BBase4Bit => "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit",
            Self::Qwen3Tts12Hz17BCustomVoice => "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            Self::Qwen3Tts12Hz17BCustomVoice4Bit => {
                "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit"
            }
            Self::Qwen3Tts12Hz17BVoiceDesign => "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            Self::Qwen3Tts12Hz17BVoiceDesign4Bit => {
                "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit"
            }
            Self::Qwen3Tts12Hz17BVoiceDesign8Bit => {
                "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"
            }
            Self::Qwen3Tts12Hz17BVoiceDesignBf16 => {
                "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
            }
            Self::Qwen3TtsTokenizer12Hz => "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            Self::Lfm25Audio15B => "LiquidAI/LFM2.5-Audio-1.5B",
            Self::Lfm25Audio15B4Bit => "mlx-community/LFM2.5-Audio-1.5B-4bit",
            Self::Lfm2512BInstructGguf => "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
            Self::Lfm2512BThinkingGguf => "LiquidAI/LFM2.5-1.2B-Thinking-GGUF",
            Self::Kokoro82M => "hexgrad/Kokoro-82M",
            Self::Qwen3Asr06B => "Qwen/Qwen3-ASR-0.6B",
            Self::Qwen3Asr06B4Bit => "mlx-community/Qwen3-ASR-0.6B-4bit",
            Self::Qwen3Asr06B8Bit => "mlx-community/Qwen3-ASR-0.6B-8bit",
            Self::Qwen3Asr06BBf16 => "mlx-community/Qwen3-ASR-0.6B-bf16",
            Self::Qwen3Asr17B => "Qwen/Qwen3-ASR-1.7B",
            Self::Qwen3Asr17B4Bit => "mlx-community/Qwen3-ASR-1.7B-4bit",
            Self::Qwen3Asr17B8Bit => "mlx-community/Qwen3-ASR-1.7B-8bit",
            Self::Qwen3Asr17BBf16 => "mlx-community/Qwen3-ASR-1.7B-bf16",
            Self::ParakeetTdt06BV2 => "nvidia/parakeet-tdt-0.6b-v2",
            Self::ParakeetTdt06BV3 => "nvidia/parakeet-tdt-0.6b-v3",
            Self::WhisperLargeV3Turbo => "openai/whisper-large-v3-turbo",
            Self::DiarStreamingSortformer4SpkV21 => "nvidia/diar_streaming_sortformer_4spk-v2.1",
            Self::Qwen306B => "Qwen/Qwen3-0.6B",
            Self::Qwen306B4Bit => "mlx-community/Qwen3-0.6B-4bit",
            Self::Qwen317B => "Qwen/Qwen3-1.7B",
            Self::Qwen317B4Bit => "mlx-community/Qwen3-1.7B-4bit",
            Self::Qwen306BGguf => "Qwen/Qwen3-0.6B-GGUF",
            Self::Qwen317BGguf => "Qwen/Qwen3-1.7B-GGUF",
            Self::Qwen34BGguf => "Qwen/Qwen3-4B-GGUF",
            Self::Qwen38BGguf => "Qwen/Qwen3-8B-GGUF",
            Self::Qwen314BGguf => "Qwen/Qwen3-14B-GGUF",
            Self::Qwen3508BGguf => "unsloth/Qwen3.5-0.8B-GGUF",
            Self::Qwen352BGguf => "unsloth/Qwen3.5-2B-GGUF",
            Self::Qwen354BGguf => "unsloth/Qwen3.5-4B-GGUF",
            Self::Qwen359BGguf => "unsloth/Qwen3.5-9B-GGUF",
            Self::Gemma31BIt => "google/gemma-3-1b-it",
            Self::Gemma34BIt => "google/gemma-3-4b-it",
            Self::Qwen3ForcedAligner06B => "Qwen/Qwen3-ForcedAligner-0.6B",
            Self::Qwen3ForcedAligner06B4Bit => "mlx-community/Qwen3-ForcedAligner-0.6B-4bit",
            Self::VoxtralMini4BRealtime2602 => "mistralai/Voxtral-Mini-4B-Realtime-2602",
        }
    }

    /// Get human-readable name
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Qwen3Tts12Hz06BBase => "Qwen3-TTS 0.6B Base",
            Self::Qwen3Tts12Hz06BBase4Bit => "Qwen3-TTS 0.6B Base 4-bit",
            Self::Qwen3Tts12Hz06BBase8Bit => "Qwen3-TTS 0.6B Base 8-bit",
            Self::Qwen3Tts12Hz06BBaseBf16 => "Qwen3-TTS 0.6B Base bf16",
            Self::Qwen3Tts12Hz06BCustomVoice => "Qwen3-TTS 0.6B CustomVoice",
            Self::Qwen3Tts12Hz06BCustomVoice4Bit => "Qwen3-TTS 0.6B CustomVoice 4-bit",
            Self::Qwen3Tts12Hz06BCustomVoice8Bit => "Qwen3-TTS 0.6B CustomVoice 8-bit",
            Self::Qwen3Tts12Hz06BCustomVoiceBf16 => "Qwen3-TTS 0.6B CustomVoice bf16",
            Self::Qwen3Tts12Hz17BBase => "Qwen3-TTS 1.7B Base",
            Self::Qwen3Tts12Hz17BBase4Bit => "Qwen3-TTS 1.7B Base 4-bit",
            Self::Qwen3Tts12Hz17BCustomVoice => "Qwen3-TTS 1.7B CustomVoice",
            Self::Qwen3Tts12Hz17BCustomVoice4Bit => "Qwen3-TTS 1.7B CustomVoice 4-bit",
            Self::Qwen3Tts12Hz17BVoiceDesign => "Qwen3-TTS 1.7B VoiceDesign",
            Self::Qwen3Tts12Hz17BVoiceDesign4Bit => "Qwen3-TTS 1.7B VoiceDesign 4-bit",
            Self::Qwen3Tts12Hz17BVoiceDesign8Bit => "Qwen3-TTS 1.7B VoiceDesign 8-bit",
            Self::Qwen3Tts12Hz17BVoiceDesignBf16 => "Qwen3-TTS 1.7B VoiceDesign bf16",
            Self::Qwen3TtsTokenizer12Hz => "Qwen3-TTS Tokenizer 12Hz",
            Self::Lfm25Audio15B => "LFM2.5-Audio 1.5B",
            Self::Lfm25Audio15B4Bit => "LFM2.5-Audio 1.5B 4-bit",
            Self::Lfm2512BInstructGguf => "LFM2.5 1.2B Instruct GGUF",
            Self::Lfm2512BThinkingGguf => "LFM2.5 1.2B Thinking GGUF",
            Self::Kokoro82M => "Kokoro 82M",
            Self::Qwen3Asr06B => "Qwen3-ASR 0.6B",
            Self::Qwen3Asr06B4Bit => "Qwen3-ASR 0.6B 4-bit",
            Self::Qwen3Asr06B8Bit => "Qwen3-ASR 0.6B 8-bit",
            Self::Qwen3Asr06BBf16 => "Qwen3-ASR 0.6B bf16",
            Self::Qwen3Asr17B => "Qwen3-ASR 1.7B",
            Self::Qwen3Asr17B4Bit => "Qwen3-ASR 1.7B 4-bit",
            Self::Qwen3Asr17B8Bit => "Qwen3-ASR 1.7B 8-bit",
            Self::Qwen3Asr17BBf16 => "Qwen3-ASR 1.7B bf16",
            Self::ParakeetTdt06BV2 => "Parakeet TDT 0.6B v2",
            Self::ParakeetTdt06BV3 => "Parakeet TDT 0.6B v3",
            Self::WhisperLargeV3Turbo => "Whisper Large v3 Turbo",
            Self::DiarStreamingSortformer4SpkV21 => "Streaming Sortformer 4spk v2.1",
            Self::Qwen306B => "Qwen3 0.6B",
            Self::Qwen306B4Bit => "Qwen3 0.6B 4-bit",
            Self::Qwen317B => "Qwen3 1.7B",
            Self::Qwen317B4Bit => "Qwen3 1.7B 4-bit",
            Self::Qwen306BGguf => "Qwen3 0.6B GGUF",
            Self::Qwen317BGguf => "Qwen3 1.7B GGUF",
            Self::Qwen34BGguf => "Qwen3 4B GGUF",
            Self::Qwen38BGguf => "Qwen3 8B GGUF",
            Self::Qwen314BGguf => "Qwen3 14B GGUF",
            Self::Qwen3508BGguf => "Qwen3.5 0.8B GGUF",
            Self::Qwen352BGguf => "Qwen3.5 2B GGUF",
            Self::Qwen354BGguf => "Qwen3.5 4B GGUF",
            Self::Qwen359BGguf => "Qwen3.5 9B GGUF",
            Self::Gemma31BIt => "Gemma 3 1B Instruct",
            Self::Gemma34BIt => "Gemma 3 4B Instruct",
            Self::Qwen3ForcedAligner06B => "Qwen3-ForcedAligner 0.6B",
            Self::Qwen3ForcedAligner06B4Bit => "Qwen3-ForcedAligner 0.6B 4-bit",
            Self::VoxtralMini4BRealtime2602 => "Voxtral Mini 4B Realtime",
        }
    }

    /// Get local directory name
    pub fn dir_name(&self) -> &'static str {
        match self {
            Self::Qwen3Tts12Hz06BBase => "Qwen3-TTS-12Hz-0.6B-Base",
            Self::Qwen3Tts12Hz06BBase4Bit => "Qwen3-TTS-12Hz-0.6B-Base-4bit",
            Self::Qwen3Tts12Hz06BBase8Bit => "Qwen3-TTS-12Hz-0.6B-Base-8bit",
            Self::Qwen3Tts12Hz06BBaseBf16 => "Qwen3-TTS-12Hz-0.6B-Base-bf16",
            Self::Qwen3Tts12Hz06BCustomVoice => "Qwen3-TTS-12Hz-0.6B-CustomVoice",
            Self::Qwen3Tts12Hz06BCustomVoice4Bit => "Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
            Self::Qwen3Tts12Hz06BCustomVoice8Bit => "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
            Self::Qwen3Tts12Hz06BCustomVoiceBf16 => "Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
            Self::Qwen3Tts12Hz17BBase => "Qwen3-TTS-12Hz-1.7B-Base",
            Self::Qwen3Tts12Hz17BBase4Bit => "Qwen3-TTS-12Hz-1.7B-Base-4bit",
            Self::Qwen3Tts12Hz17BCustomVoice => "Qwen3-TTS-12Hz-1.7B-CustomVoice",
            Self::Qwen3Tts12Hz17BCustomVoice4Bit => "Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit",
            Self::Qwen3Tts12Hz17BVoiceDesign => "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            Self::Qwen3Tts12Hz17BVoiceDesign4Bit => "Qwen3-TTS-12Hz-1.7B-VoiceDesign-4bit",
            Self::Qwen3Tts12Hz17BVoiceDesign8Bit => "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
            Self::Qwen3Tts12Hz17BVoiceDesignBf16 => "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
            Self::Qwen3TtsTokenizer12Hz => "Qwen3-TTS-Tokenizer-12Hz",
            Self::Lfm25Audio15B => "LFM2.5-Audio-1.5B",
            Self::Lfm25Audio15B4Bit => "LFM2.5-Audio-1.5B-4bit",
            Self::Lfm2512BInstructGguf => "LFM2.5-1.2B-Instruct-GGUF",
            Self::Lfm2512BThinkingGguf => "LFM2.5-1.2B-Thinking-GGUF",
            Self::Kokoro82M => "Kokoro-82M",
            Self::Qwen3Asr06B => "Qwen3-ASR-0.6B",
            Self::Qwen3Asr06B4Bit => "Qwen3-ASR-0.6B-4bit",
            Self::Qwen3Asr06B8Bit => "Qwen3-ASR-0.6B-8bit",
            Self::Qwen3Asr06BBf16 => "Qwen3-ASR-0.6B-bf16",
            Self::Qwen3Asr17B => "Qwen3-ASR-1.7B",
            Self::Qwen3Asr17B4Bit => "Qwen3-ASR-1.7B-4bit",
            Self::Qwen3Asr17B8Bit => "Qwen3-ASR-1.7B-8bit",
            Self::Qwen3Asr17BBf16 => "Qwen3-ASR-1.7B-bf16",
            Self::ParakeetTdt06BV2 => "Parakeet-TDT-0.6B-v2",
            Self::ParakeetTdt06BV3 => "Parakeet-TDT-0.6B-v3",
            Self::WhisperLargeV3Turbo => "Whisper-Large-v3-Turbo",
            Self::DiarStreamingSortformer4SpkV21 => "diar_streaming_sortformer_4spk-v2.1",
            Self::Qwen306B => "Qwen3-0.6B",
            Self::Qwen306B4Bit => "Qwen3-0.6B-4bit",
            Self::Qwen317B => "Qwen3-1.7B",
            Self::Qwen317B4Bit => "Qwen3-1.7B-4bit",
            Self::Qwen306BGguf => "Qwen3-0.6B-GGUF",
            Self::Qwen317BGguf => "Qwen3-1.7B-GGUF",
            Self::Qwen34BGguf => "Qwen3-4B-GGUF",
            Self::Qwen38BGguf => "Qwen3-8B-GGUF",
            Self::Qwen314BGguf => "Qwen3-14B-GGUF",
            Self::Qwen3508BGguf => "Qwen3.5-0.8B",
            Self::Qwen352BGguf => "Qwen3.5-2B",
            Self::Qwen354BGguf => "Qwen3.5-4B",
            Self::Qwen359BGguf => "Qwen3.5-9B",
            Self::Gemma31BIt => "Gemma-3-1b-it",
            Self::Gemma34BIt => "Gemma-3-4b-it",
            Self::Qwen3ForcedAligner06B => "Qwen3-ForcedAligner-0.6B",
            Self::Qwen3ForcedAligner06B4Bit => "Qwen3-ForcedAligner-0.6B-4bit",
            Self::VoxtralMini4BRealtime2602 => "Voxtral-Mini-4B-Realtime-2602",
        }
    }

    /// Estimated model size in bytes
    pub fn estimated_size(&self) -> u64 {
        match self {
            Self::Qwen3Tts12Hz06BBase => 2_516_106_051, // ~2.34 GB
            Self::Qwen3Tts12Hz06BBase4Bit => 1_711_328_624, // ~1.59 GB
            Self::Qwen3Tts12Hz06BBase8Bit => 1_991_299_138, // ~1.85 GB
            Self::Qwen3Tts12Hz06BBaseBf16 => 2_516_143_009, // ~2.34 GB
            Self::Qwen3Tts12Hz06BCustomVoice => 2_498_388_392, // ~2.33 GB
            Self::Qwen3Tts12Hz06BCustomVoice4Bit => 1_693_604_738, // ~1.58 GB
            Self::Qwen3Tts12Hz06BCustomVoice8Bit => 1_973_575_388, // ~1.84 GB
            Self::Qwen3Tts12Hz06BCustomVoiceBf16 => 2_498_419_405, // ~2.33 GB
            Self::Qwen3Tts12Hz17BBase => 4_544_229_700, // ~4.23 GB
            Self::Qwen3Tts12Hz17BBase4Bit => 2_335_000_000, // ~2.17 GB (est)
            Self::Qwen3Tts12Hz17BCustomVoice => 4_520_218_951, // ~4.21 GB
            Self::Qwen3Tts12Hz17BCustomVoice4Bit => 2_330_000_000, // ~2.17 GB (est)
            Self::Qwen3Tts12Hz17BVoiceDesign => 4_520_163_832, // ~4.21 GB
            Self::Qwen3Tts12Hz17BVoiceDesign4Bit => 2_312_058_795, // ~2.15 GB
            Self::Qwen3Tts12Hz17BVoiceDesign8Bit => 3_080_140_867, // ~2.87 GB
            Self::Qwen3Tts12Hz17BVoiceDesignBf16 => 4_520_194_992, // ~4.21 GB
            Self::Qwen3TtsTokenizer12Hz => 682_300_739, // ~0.64 GB
            Self::Lfm25Audio15B => 3_200_000_000,       // ~2.98 GB (est)
            Self::Lfm25Audio15B4Bit => 884_000_000,     // ~0.82 GB
            Self::Lfm2512BInstructGguf => 730_895_168,  // ~0.68 GB (GGUF Q4_K_M, HF tree)
            Self::Lfm2512BThinkingGguf => 730_895_360,  // ~0.68 GB (GGUF Q4_K_M, HF tree)
            Self::Kokoro82M => 363_323_757,             // ~346 MB (HF tree total, Apr 2025)
            Self::Qwen3Asr06B => 1_880_619_678,         // ~1.75 GB
            Self::Qwen3Asr06B4Bit => 712_781_279,       // ~0.66 GB
            Self::Qwen3Asr06B8Bit => 1_010_773_761,     // ~0.94 GB
            Self::Qwen3Asr06BBf16 => 1_569_438_434,     // ~1.46 GB
            Self::Qwen3Asr17B => 4_703_114_308,         // ~4.38 GB
            Self::Qwen3Asr17B4Bit => 1_607_633_106,     // ~1.50 GB
            Self::Qwen3Asr17B8Bit => 2_467_859_030,     // ~2.30 GB
            Self::Qwen3Asr17BBf16 => 4_080_710_353,     // ~3.80 GB
            Self::ParakeetTdt06BV2 => 4_926_457_088,    // ~4.59 GB
            Self::ParakeetTdt06BV3 => 10_036_761_167,   // ~9.35 GB
            Self::WhisperLargeV3Turbo => 1_617_824_864, // ~1.51 GB (HF x-linked-size)
            Self::DiarStreamingSortformer4SpkV21 => 510_000_000, // ~0.47 GB (est)
            Self::Qwen306B => 1_520_000_000,            // ~1.42 GB (est)
            Self::Qwen306B4Bit => 900_000_000,          // ~0.84 GB (est)
            Self::Qwen317B => 4_080_000_000,            // ~3.80 GB (actual: 3.44GB + 622MB shards)
            Self::Qwen317B4Bit => 1_115_700_000,        // ~1.04 GB
            Self::Qwen306BGguf => 1_100_000_000,        // ~1.02 GB (Q8_0 est)
            Self::Qwen317BGguf => 2_400_000_000,        // ~2.24 GB (Q8_0 est)
            Self::Qwen34BGguf => 2_500_000_000, // ~2.33 GB (Q4_K_M GGUF, HF file size, Feb 2026)
            Self::Qwen38BGguf => 5_200_000_000, // ~4.84 GB (Q4_K_M est)
            Self::Qwen314BGguf => 9_200_000_000, // ~8.57 GB (Q4_K_M est)
            Self::Qwen3508BGguf => 737_504_352, // local GGUF + mmproj + tokenizer assets
            Self::Qwen352BGguf => 1_949_063_104, // local GGUF + mmproj + tokenizer assets
            Self::Qwen354BGguf => 3_413_361_504, // local GGUF + mmproj + tokenizer assets
            Self::Qwen359BGguf => 6_598_688_544, // local GGUF + mmproj + tokenizer assets
            Self::Gemma31BIt => 2_200_000_000,  // ~2.05 GB (est)
            Self::Gemma34BIt => 8_600_000_000,  // ~8.01 GB (est)
            Self::Qwen3ForcedAligner06B => 1_840_072_459, // ~1.71 GB
            Self::Qwen3ForcedAligner06B4Bit => 703_200_000, // ~0.65 GB
            Self::VoxtralMini4BRealtime2602 => 8_000_000_000, // ~7.45 GB (est)
        }
    }

    /// Memory required for inference
    pub fn memory_required_gb(&self) -> f32 {
        match self {
            Self::Qwen3Tts12Hz06BBase
            | Self::Qwen3Tts12Hz06BBase4Bit
            | Self::Qwen3Tts12Hz06BBase8Bit
            | Self::Qwen3Tts12Hz06BBaseBf16
            | Self::Qwen3Tts12Hz06BCustomVoice
            | Self::Qwen3Tts12Hz06BCustomVoice4Bit
            | Self::Qwen3Tts12Hz06BCustomVoice8Bit
            | Self::Qwen3Tts12Hz06BCustomVoiceBf16 => 2.5,
            Self::Qwen3Tts12Hz17BBase
            | Self::Qwen3Tts12Hz17BBase4Bit
            | Self::Qwen3Tts12Hz17BCustomVoice
            | Self::Qwen3Tts12Hz17BCustomVoice4Bit
            | Self::Qwen3Tts12Hz17BVoiceDesign
            | Self::Qwen3Tts12Hz17BVoiceDesign4Bit
            | Self::Qwen3Tts12Hz17BVoiceDesign8Bit
            | Self::Qwen3Tts12Hz17BVoiceDesignBf16 => 6.0,
            Self::Qwen3TtsTokenizer12Hz => 1.0,
            Self::Lfm25Audio15B => 6.0,
            Self::Lfm25Audio15B4Bit => 4.5,
            Self::Lfm2512BInstructGguf | Self::Lfm2512BThinkingGguf => 2.0,
            Self::Kokoro82M => 2.0,
            Self::Qwen3Asr06B
            | Self::Qwen3Asr06B4Bit
            | Self::Qwen3Asr06B8Bit
            | Self::Qwen3Asr06BBf16 => 2.5,
            Self::Qwen3Asr17B
            | Self::Qwen3Asr17B4Bit
            | Self::Qwen3Asr17B8Bit
            | Self::Qwen3Asr17BBf16 => 6.0,
            Self::ParakeetTdt06BV2 => 8.0,
            Self::ParakeetTdt06BV3 => 12.0,
            Self::WhisperLargeV3Turbo => 4.0,
            Self::DiarStreamingSortformer4SpkV21 => 3.0,
            Self::Qwen306B => 3.0,
            Self::Qwen306B4Bit => 2.0,
            Self::Qwen317B => 5.0,
            Self::Qwen317B4Bit => 3.0,
            Self::Qwen306BGguf => 2.5,
            Self::Qwen317BGguf => 4.0,
            Self::Qwen34BGguf => 6.0,
            Self::Qwen38BGguf => 10.0,
            Self::Qwen314BGguf => 16.0,
            Self::Qwen3508BGguf => 3.5,
            Self::Qwen352BGguf => 5.5,
            Self::Qwen354BGguf => 9.0,
            Self::Qwen359BGguf => 16.0,
            Self::Gemma31BIt => 3.5,
            Self::Gemma34BIt => 11.0,
            Self::Qwen3ForcedAligner06B => 2.5,
            Self::Qwen3ForcedAligner06B4Bit => 1.5,
            Self::VoxtralMini4BRealtime2602 => 16.0,
        }
    }

    /// Whether this is a tokenizer/codec model
    pub fn is_tokenizer(&self) -> bool {
        matches!(self.family(), crate::catalog::ModelFamily::Tokenizer)
    }

    /// Whether this is an LFM2-Audio model
    pub fn is_lfm2(&self) -> bool {
        matches!(self.family(), crate::catalog::ModelFamily::Lfm2Audio)
    }

    /// Whether this is a Kokoro TTS model
    pub fn is_kokoro(&self) -> bool {
        matches!(self.family(), crate::catalog::ModelFamily::KokoroTts)
    }

    /// Whether this is a Qwen3-ASR model
    pub fn is_asr(&self) -> bool {
        matches!(
            self.family(),
            crate::catalog::ModelFamily::Qwen3Asr
                | crate::catalog::ModelFamily::ParakeetAsr
                | crate::catalog::ModelFamily::WhisperAsr
        )
    }

    /// Whether this is a diarization model.
    pub fn is_diarization(&self) -> bool {
        matches!(
            self.family(),
            crate::catalog::ModelFamily::SortformerDiarization
        )
    }

    /// Whether this is a forced aligner model
    pub fn is_forced_aligner(&self) -> bool {
        matches!(
            self.family(),
            crate::catalog::ModelFamily::Qwen3ForcedAligner
        )
    }

    /// Whether this is a text chat model
    pub fn is_chat(&self) -> bool {
        matches!(
            self.family(),
            crate::catalog::ModelFamily::Qwen3Chat
                | crate::catalog::ModelFamily::Qwen35Chat
                | crate::catalog::ModelFamily::Gemma3Chat
                | crate::catalog::ModelFamily::Lfm2Chat
        )
    }

    pub fn is_tts(&self) -> bool {
        matches!(
            self.family(),
            crate::catalog::ModelFamily::Qwen3Tts | crate::catalog::ModelFamily::KokoroTts
        )
    }

    pub fn speech_capabilities(&self) -> Option<SpeechModelCapabilities> {
        let capabilities = match self {
            Self::Qwen3Tts12Hz06BBase
            | Self::Qwen3Tts12Hz06BBase4Bit
            | Self::Qwen3Tts12Hz06BBase8Bit
            | Self::Qwen3Tts12Hz06BBaseBf16
            | Self::Qwen3Tts12Hz17BBase
            | Self::Qwen3Tts12Hz17BBase4Bit => SpeechModelCapabilities {
                supports_builtin_voices: false,
                built_in_voice_count: None,
                supports_reference_voice: true,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: true,
                supports_auto_long_form: true,
            },
            Self::Qwen3Tts12Hz06BCustomVoice
            | Self::Qwen3Tts12Hz06BCustomVoice4Bit
            | Self::Qwen3Tts12Hz06BCustomVoice8Bit
            | Self::Qwen3Tts12Hz06BCustomVoiceBf16 => SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(Self::QWEN_CUSTOMVOICE_BUILT_IN_VOICE_COUNT),
                supports_reference_voice: false,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: true,
                supports_auto_long_form: true,
            },
            Self::Qwen3Tts12Hz17BCustomVoice | Self::Qwen3Tts12Hz17BCustomVoice4Bit => {
                SpeechModelCapabilities {
                    supports_builtin_voices: true,
                    built_in_voice_count: Some(Self::QWEN_CUSTOMVOICE_BUILT_IN_VOICE_COUNT),
                    supports_reference_voice: false,
                    supports_voice_description: true,
                    supports_streaming: true,
                    supports_speed_control: true,
                    supports_auto_long_form: true,
                }
            }
            Self::Qwen3Tts12Hz17BVoiceDesign
            | Self::Qwen3Tts12Hz17BVoiceDesign4Bit
            | Self::Qwen3Tts12Hz17BVoiceDesign8Bit
            | Self::Qwen3Tts12Hz17BVoiceDesignBf16 => SpeechModelCapabilities {
                supports_builtin_voices: false,
                built_in_voice_count: None,
                supports_reference_voice: false,
                supports_voice_description: true,
                supports_streaming: true,
                supports_speed_control: true,
                supports_auto_long_form: true,
            },
            Self::Lfm25Audio15B | Self::Lfm25Audio15B4Bit => SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(Self::LFM2_AUDIO_BUILT_IN_VOICE_COUNT),
                supports_reference_voice: true,
                supports_voice_description: true,
                supports_streaming: true,
                supports_speed_control: true,
                supports_auto_long_form: false,
            },
            Self::Kokoro82M => SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(Self::KOKORO_BUILT_IN_VOICE_COUNT),
                supports_reference_voice: false,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: true,
                supports_auto_long_form: false,
            },
            _ => return None,
        };

        Some(capabilities)
    }

    /// Max output codec frames for this TTS variant, if known.
    pub fn tts_max_output_frames_hint(&self) -> Option<usize> {
        if matches!(self.family(), crate::catalog::ModelFamily::Qwen3Tts) {
            Some(Self::QWEN3_TTS_MAX_OUTPUT_FRAMES)
        } else {
            None
        }
    }

    /// Codec frame-rate hint for this TTS variant, if known.
    pub fn tts_output_frame_rate_hz_hint(&self) -> Option<f32> {
        if matches!(self.family(), crate::catalog::ModelFamily::Qwen3Tts) {
            Some(Self::QWEN3_TTS_FRAME_RATE_HZ)
        } else {
            None
        }
    }

    /// Max output duration hint in seconds for this TTS variant, if known.
    pub fn tts_max_output_seconds_hint(&self) -> Option<f32> {
        let frames = self.tts_max_output_frames_hint()?;
        let hz = self.tts_output_frame_rate_hz_hint()?;
        if hz > 0.0 {
            Some(frames as f32 / hz)
        } else {
            None
        }
    }

    /// Whether this is a Voxtral model
    pub fn is_voxtral(&self) -> bool {
        matches!(self.family(), crate::catalog::ModelFamily::Voxtral)
    }

    /// Whether this is a Parakeet ASR model.
    pub fn is_parakeet(&self) -> bool {
        matches!(self.family(), crate::catalog::ModelFamily::ParakeetAsr)
    }

    /// Whether this is a legacy .nemo Parakeet variant.
    pub fn is_parakeet_nemo(&self) -> bool {
        matches!(self, Self::ParakeetTdt06BV2 | Self::ParakeetTdt06BV3)
    }

    /// Whether this is an MLX-exported Parakeet variant.
    pub fn is_parakeet_mlx(&self) -> bool {
        false
    }

    /// Whether this variant uses reduced-precision or quantized weights.
    pub fn is_quantized(&self) -> bool {
        matches!(
            self,
            Self::Qwen3Tts12Hz06BBase4Bit
                | Self::Qwen3Tts12Hz06BBase8Bit
                | Self::Qwen3Tts12Hz06BBaseBf16
                | Self::Qwen3Tts12Hz06BCustomVoice4Bit
                | Self::Qwen3Tts12Hz06BCustomVoice8Bit
                | Self::Qwen3Tts12Hz06BCustomVoiceBf16
                | Self::Qwen3Tts12Hz17BBase4Bit
                | Self::Qwen3Tts12Hz17BCustomVoice4Bit
                | Self::Qwen3Tts12Hz17BVoiceDesign4Bit
                | Self::Qwen3Tts12Hz17BVoiceDesign8Bit
                | Self::Qwen3Tts12Hz17BVoiceDesignBf16
                | Self::Lfm25Audio15B4Bit
                | Self::Lfm2512BInstructGguf
                | Self::Lfm2512BThinkingGguf
                | Self::Qwen3Asr06B4Bit
                | Self::Qwen3Asr06B8Bit
                | Self::Qwen3Asr06BBf16
                | Self::Qwen3Asr17B4Bit
                | Self::Qwen3Asr17B8Bit
                | Self::Qwen3Asr17BBf16
                | Self::Qwen306B4Bit
                | Self::Qwen317B4Bit
                | Self::Qwen306BGguf
                | Self::Qwen317BGguf
                | Self::Qwen34BGguf
                | Self::Qwen38BGguf
                | Self::Qwen314BGguf
                | Self::Qwen3508BGguf
                | Self::Qwen352BGguf
                | Self::Qwen354BGguf
                | Self::Qwen359BGguf
                | Self::Qwen3ForcedAligner06B4Bit
        )
    }

    /// Whether this variant uses GGUF weights.
    pub fn is_gguf(&self) -> bool {
        matches!(
            self,
            Self::Qwen306BGguf
                | Self::Qwen317BGguf
                | Self::Qwen34BGguf
                | Self::Qwen38BGguf
                | Self::Qwen314BGguf
                | Self::Qwen3508BGguf
                | Self::Qwen352BGguf
                | Self::Qwen354BGguf
                | Self::Qwen359BGguf
                | Self::Lfm2512BInstructGguf
                | Self::Lfm2512BThinkingGguf
        )
    }

    /// Whether this is a Qwen3 chat GGUF variant.
    pub fn is_qwen_chat_gguf(&self) -> bool {
        matches!(
            self,
            Self::Qwen306BGguf
                | Self::Qwen317BGguf
                | Self::Qwen34BGguf
                | Self::Qwen38BGguf
                | Self::Qwen314BGguf
        )
    }

    /// Whether this is a Qwen3.5 chat GGUF variant.
    pub fn is_qwen35_chat_gguf(&self) -> bool {
        matches!(
            self,
            Self::Qwen3508BGguf
                | Self::Qwen352BGguf
                | Self::Qwen354BGguf
                | Self::Qwen359BGguf
        )
    }

    /// Whether this is an LFM2.5 chat GGUF variant.
    pub fn is_lfm2_chat_gguf(&self) -> bool {
        matches!(
            self,
            Self::Lfm2512BInstructGguf | Self::Lfm2512BThinkingGguf
        )
    }

    /// Whether this variant is currently enabled in the application catalog.
    pub fn is_enabled(&self) -> bool {
        match self {
            Self::Qwen306B
            | Self::Qwen306B4Bit
            | Self::Qwen317B
            | Self::Qwen317B4Bit
            | Self::Qwen314BGguf => false,
            Self::Qwen306BGguf
            | Self::Qwen317BGguf
            | Self::Qwen34BGguf
            | Self::Qwen38BGguf
            | Self::Qwen3508BGguf
            | Self::Qwen352BGguf
            | Self::Qwen354BGguf
            | Self::Qwen359BGguf
            | Self::Gemma31BIt
            | Self::Qwen3Asr17B4Bit
            | Self::Qwen3Tts12Hz06BBase4Bit
            | Self::Qwen3Tts12Hz06BCustomVoice4Bit
            | Self::Qwen3Tts12Hz17BBase4Bit
            | Self::Qwen3Tts12Hz17BCustomVoice4Bit
            | Self::Qwen3Tts12Hz17BVoiceDesign4Bit
            | Self::Qwen3ForcedAligner06B4Bit
            | Self::Lfm25Audio15B4Bit
            | Self::Lfm2512BInstructGguf
            | Self::Lfm2512BThinkingGguf
            | Self::Kokoro82M => true,
            Self::Gemma34BIt => false,
            Self::Lfm25Audio15B => true,
            Self::VoxtralMini4BRealtime2602 => false,
            Self::ParakeetTdt06BV2 | Self::ParakeetTdt06BV3 => true,
            Self::WhisperLargeV3Turbo => true,
            Self::DiarStreamingSortformer4SpkV21 => true,
            Self::Qwen3ForcedAligner06B => true,
            _ => !self.is_quantized(),
        }
    }

    /// Get all available variants
    pub fn all() -> &'static [ModelVariant] {
        &[
            Self::Qwen3Tts12Hz06BBase,
            Self::Qwen3Tts12Hz06BBase4Bit,
            Self::Qwen3Tts12Hz06BBase8Bit,
            Self::Qwen3Tts12Hz06BBaseBf16,
            Self::Qwen3Tts12Hz06BCustomVoice,
            Self::Qwen3Tts12Hz06BCustomVoice4Bit,
            Self::Qwen3Tts12Hz06BCustomVoice8Bit,
            Self::Qwen3Tts12Hz06BCustomVoiceBf16,
            Self::Qwen3Tts12Hz17BBase,
            Self::Qwen3Tts12Hz17BBase4Bit,
            Self::Qwen3Tts12Hz17BCustomVoice,
            Self::Qwen3Tts12Hz17BCustomVoice4Bit,
            Self::Qwen3Tts12Hz17BVoiceDesign,
            Self::Qwen3Tts12Hz17BVoiceDesign4Bit,
            Self::Qwen3Tts12Hz17BVoiceDesign8Bit,
            Self::Qwen3Tts12Hz17BVoiceDesignBf16,
            Self::Qwen3TtsTokenizer12Hz,
            Self::Lfm25Audio15B,
            Self::Lfm25Audio15B4Bit,
            Self::Lfm2512BInstructGguf,
            Self::Lfm2512BThinkingGguf,
            Self::Kokoro82M,
            Self::Qwen3Asr06B,
            Self::Qwen3Asr06B4Bit,
            Self::Qwen3Asr06B8Bit,
            Self::Qwen3Asr06BBf16,
            Self::Qwen3Asr17B,
            Self::Qwen3Asr17B4Bit,
            Self::Qwen3Asr17B8Bit,
            Self::Qwen3Asr17BBf16,
            Self::ParakeetTdt06BV2,
            Self::ParakeetTdt06BV3,
            Self::WhisperLargeV3Turbo,
            Self::DiarStreamingSortformer4SpkV21,
            Self::Qwen306B,
            Self::Qwen306B4Bit,
            Self::Qwen306BGguf,
            Self::Qwen317B,
            Self::Qwen317B4Bit,
            Self::Qwen317BGguf,
            Self::Qwen34BGguf,
            Self::Qwen38BGguf,
            Self::Qwen314BGguf,
            Self::Qwen3508BGguf,
            Self::Qwen352BGguf,
            Self::Qwen354BGguf,
            Self::Qwen359BGguf,
            Self::Gemma31BIt,
            Self::Gemma34BIt,
            Self::Qwen3ForcedAligner06B,
            Self::Qwen3ForcedAligner06B4Bit,
            Self::VoxtralMini4BRealtime2602,
        ]
    }
}

impl std::fmt::Display for ModelVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Model download/load status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    /// Not downloaded
    NotDownloaded,
    /// Currently downloading
    Downloading,
    /// Downloaded but not loaded
    Downloaded,
    /// Currently loading into memory
    Loading,
    /// Loaded and ready for inference
    Ready,
    /// Error state
    Error,
}

/// Machine-readable speech feature contract for voice workflows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeechModelCapabilities {
    pub supports_builtin_voices: bool,
    pub built_in_voice_count: Option<usize>,
    pub supports_reference_voice: bool,
    pub supports_voice_description: bool,
    pub supports_streaming: bool,
    pub supports_speed_control: bool,
    pub supports_auto_long_form: bool,
}

/// Complete model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub variant: ModelVariant,
    pub enabled: bool,
    pub status: ModelStatus,
    pub local_path: Option<PathBuf>,
    pub size_bytes: Option<u64>,
    pub download_progress: Option<f32>,
    pub error_message: Option<String>,
    pub speech_capabilities: Option<SpeechModelCapabilities>,
}

impl ModelInfo {
    pub fn new(variant: ModelVariant) -> Self {
        Self {
            variant,
            enabled: variant.is_enabled(),
            status: ModelStatus::NotDownloaded,
            local_path: None,
            size_bytes: None,
            download_progress: None,
            error_message: None,
            speech_capabilities: variant.speech_capabilities(),
        }
    }

    pub fn with_path(mut self, path: PathBuf) -> Self {
        self.local_path = Some(path);
        self.status = ModelStatus::Downloaded;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{ModelVariant, SpeechModelCapabilities};

    #[test]
    fn qwen3_tts_variants_expose_output_hints() {
        let variant = ModelVariant::Qwen3Tts12Hz17BVoiceDesign;
        assert_eq!(
            variant.tts_max_output_frames_hint(),
            Some(ModelVariant::QWEN3_TTS_MAX_OUTPUT_FRAMES)
        );
        assert_eq!(
            variant.tts_output_frame_rate_hz_hint(),
            Some(ModelVariant::QWEN3_TTS_FRAME_RATE_HZ)
        );
        assert!(
            variant
                .tts_max_output_seconds_hint()
                .is_some_and(|seconds| seconds > 600.0),
            "expected >10 minutes max-output hint"
        );
    }

    #[test]
    fn non_qwen3_tts_variants_do_not_expose_qwen_output_hints() {
        let variant = ModelVariant::Lfm25Audio15B;
        assert_eq!(variant.tts_max_output_frames_hint(), None);
        assert_eq!(variant.tts_output_frame_rate_hz_hint(), None);
        assert_eq!(variant.tts_max_output_seconds_hint(), None);
    }

    #[test]
    fn qwen3_asr_06b_quantized_variants_are_disabled() {
        let q4 = ModelVariant::Qwen3Asr06B4Bit;
        let q8 = ModelVariant::Qwen3Asr06B8Bit;
        assert_eq!(q4.repo_id(), "mlx-community/Qwen3-ASR-0.6B-4bit");
        assert_eq!(q8.repo_id(), "mlx-community/Qwen3-ASR-0.6B-8bit");
        assert!(!q4.is_enabled());
        assert!(!q8.is_enabled());
    }

    #[test]
    fn retired_qwen3_chat_variants_are_disabled() {
        let retired = [
            ModelVariant::Qwen306B,
            ModelVariant::Qwen306B4Bit,
            ModelVariant::Qwen317B,
            ModelVariant::Qwen317B4Bit,
            ModelVariant::Qwen314BGguf,
        ];
        for variant in retired {
            assert!(
                !variant.is_enabled(),
                "{} should be disabled",
                variant.dir_name()
            );
        }
    }

    #[test]
    fn qwen35_chat_variants_are_enabled_gguf_models() {
        let active = [
            ModelVariant::Qwen3508BGguf,
            ModelVariant::Qwen352BGguf,
            ModelVariant::Qwen354BGguf,
            ModelVariant::Qwen359BGguf,
        ];

        for variant in active {
            assert!(variant.is_enabled(), "{} should be enabled", variant.dir_name());
            assert!(variant.is_chat(), "{} should be chat", variant.dir_name());
            assert!(variant.is_quantized(), "{} should be quantized", variant.dir_name());
            assert!(variant.is_gguf(), "{} should be gguf", variant.dir_name());
            assert!(
                variant.is_qwen35_chat_gguf(),
                "{} should be a qwen3.5 gguf",
                variant.dir_name()
            );
        }
    }

    #[test]
    fn qwen_base_capabilities_match_saved_voice_workflow() {
        assert_eq!(
            ModelVariant::Qwen3Tts12Hz17BBase.speech_capabilities(),
            Some(SpeechModelCapabilities {
                supports_builtin_voices: false,
                built_in_voice_count: None,
                supports_reference_voice: true,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: true,
                supports_auto_long_form: true,
            })
        );
    }

    #[test]
    fn custom_voice_capabilities_distinguish_instruction_support() {
        assert_eq!(
            ModelVariant::Qwen3Tts12Hz06BCustomVoice.speech_capabilities(),
            Some(SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(ModelVariant::QWEN_CUSTOMVOICE_BUILT_IN_VOICE_COUNT),
                supports_reference_voice: false,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: true,
                supports_auto_long_form: true,
            })
        );
        assert_eq!(
            ModelVariant::Qwen3Tts12Hz17BCustomVoice.speech_capabilities(),
            Some(SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(ModelVariant::QWEN_CUSTOMVOICE_BUILT_IN_VOICE_COUNT),
                supports_reference_voice: false,
                supports_voice_description: true,
                supports_streaming: true,
                supports_speed_control: true,
                supports_auto_long_form: true,
            })
        );
    }

    #[test]
    fn kokoro_and_lfm2_capabilities_reflect_runtime_constraints() {
        assert_eq!(
            ModelVariant::Kokoro82M.speech_capabilities(),
            Some(SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(ModelVariant::KOKORO_BUILT_IN_VOICE_COUNT),
                supports_reference_voice: false,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: true,
                supports_auto_long_form: false,
            })
        );
        assert_eq!(
            ModelVariant::Lfm25Audio15B.speech_capabilities(),
            Some(SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(ModelVariant::LFM2_AUDIO_BUILT_IN_VOICE_COUNT),
                supports_reference_voice: true,
                supports_voice_description: true,
                supports_streaming: true,
                supports_speed_control: true,
                supports_auto_long_form: false,
            })
        );
    }

    #[test]
    fn non_speech_variants_do_not_expose_speech_capabilities() {
        assert_eq!(ModelVariant::Qwen3Asr06B.speech_capabilities(), None);
        assert_eq!(ModelVariant::Qwen38BGguf.speech_capabilities(), None);
    }
}
