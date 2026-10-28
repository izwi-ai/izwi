//! Model information and metadata

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::{CudaQuantizationInfo, CudaSupportInfo};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatReasoningEffort {
    #[default]
    Xhigh,
    Medium,
    Low,
}

/// Revision-pinned model limits used when converted artifacts omit otherwise
/// authoritative configuration metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactContextManifest {
    pub native_context_tokens: usize,
    pub evidence: &'static str,
}

/// Available TTS model variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelVariant {
    /// 0.6B parameter base model
    #[serde(rename = "Qwen3-TTS-12Hz-0.6B-Base")]
    Qwen3Tts12Hz06BBase,
    /// 0.6B parameter base model (MLX 8-bit)
    #[serde(rename = "Qwen3-TTS-12Hz-0.6B-Base-8bit")]
    Qwen3Tts12Hz06BBase8Bit,
    /// 0.6B parameter base model (MLX bf16)
    #[serde(rename = "Qwen3-TTS-12Hz-0.6B-Base-bf16")]
    Qwen3Tts12Hz06BBaseBf16,
    /// 0.6B parameter custom voice model
    #[serde(rename = "Qwen3-TTS-12Hz-0.6B-CustomVoice")]
    Qwen3Tts12Hz06BCustomVoice,
    /// 0.6B parameter custom voice model (MLX 8-bit)
    #[serde(rename = "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit")]
    Qwen3Tts12Hz06BCustomVoice8Bit,
    /// 0.6B parameter custom voice model (MLX bf16)
    #[serde(rename = "Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16")]
    Qwen3Tts12Hz06BCustomVoiceBf16,
    /// 1.7B parameter base model
    #[serde(rename = "Qwen3-TTS-12Hz-1.7B-Base")]
    Qwen3Tts12Hz17BBase,
    /// 1.7B parameter custom voice model  
    #[serde(rename = "Qwen3-TTS-12Hz-1.7B-CustomVoice")]
    Qwen3Tts12Hz17BCustomVoice,
    /// 1.7B parameter voice design model
    #[serde(rename = "Qwen3-TTS-12Hz-1.7B-VoiceDesign")]
    Qwen3Tts12Hz17BVoiceDesign,
    /// 1.7B parameter voice design model (MLX 8-bit)
    #[serde(rename = "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit")]
    Qwen3Tts12Hz17BVoiceDesign8Bit,
    /// 1.7B parameter voice design model (MLX bf16)
    #[serde(rename = "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")]
    Qwen3Tts12Hz17BVoiceDesignBf16,
    /// Tokenizer for 12Hz codec
    #[serde(rename = "Qwen3-TTS-Tokenizer-12Hz")]
    Qwen3TtsTokenizer12Hz,
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
    /// LFM2.5 Audio 1.5B model from Liquid AI (GGUF Q4_0 bundle)
    #[serde(
        rename = "LFM2.5-Audio-1.5B-GGUF",
        alias = "LiquidAI/LFM2.5-Audio-1.5B-GGUF",
        alias = "LFM2.5-Audio-1.5B-Q4_0.gguf"
    )]
    Lfm25Audio15BGguf,
    /// LFM3 1.7B from Liquid AI (GGUF, quantization-aware distilled Q4_0)
    #[serde(
        rename = "LFM3-1.7B-GGUF",
        alias = "LiquidAI/LFM3-1.7B-GGUF",
        alias = "LFM3-1.7B-QAD-Q4_0.gguf"
    )]
    Lfm317BGguf,
    /// LFM3 1.7B from Liquid AI (GGUF Q8_0)
    #[serde(rename = "LFM3-1.7B-Q8_0-GGUF", alias = "LFM3-1.7B-Q8_0.gguf")]
    Lfm317BQ8Gguf,
    /// LFM3 Audio 2B from Liquid AI (GGUF Q4_K_M language model and Q8_0
    /// encoder, depth decoder and codec)
    #[serde(
        rename = "LFM3-Audio-2B-GGUF",
        alias = "LiquidAI/LFM3-Audio-2B-GGUF",
        alias = "LFM3-Audio-2B-Q4_K_M.gguf"
    )]
    Lfm3Audio2BGguf,
    /// LFM3 Audio 2B from Liquid AI (GGUF Q8_0 language model)
    #[serde(rename = "LFM3-Audio-2B-Q8_0-GGUF", alias = "LFM3-Audio-2B-Q8_0.gguf")]
    Lfm3Audio2BQ8Gguf,
    /// Kokoro-82M TTS model from hexgrad
    #[serde(rename = "Kokoro-82M")]
    Kokoro82M,
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
    /// Qwen3 ASR 0.6B model (GGUF Q8_0)
    #[serde(
        rename = "Qwen3-ASR-0.6B-GGUF",
        alias = "Qwen3-ASR-0.6B",
        alias = "qwen3-asr-0.6b",
        alias = "qwen3_asr_0.6b_q8_0.gguf"
    )]
    Qwen3Asr06BGguf,
    /// Qwen3 ASR 1.7B model (GGUF Q8_0)
    #[serde(
        rename = "Qwen3-ASR-1.7B-GGUF",
        alias = "Qwen3-ASR-1.7B",
        alias = "qwen3-asr-1.7b",
        alias = "qwen3_asr_1.7b_q8_0.gguf"
    )]
    Qwen3Asr17BGguf,
    /// VibeVoice ASR model from Microsoft
    #[serde(
        rename = "VibeVoice-ASR",
        alias = "microsoft/VibeVoice-ASR",
        alias = "vibevoice-asr"
    )]
    VibeVoiceAsr,
    /// NVIDIA Nemotron 3.5 streaming multilingual ASR model (.nemo)
    #[serde(
        rename = "Nemotron-3.5-ASR-Streaming-0.6B",
        alias = "nvidia/nemotron-3.5-asr-streaming-0.6b",
        alias = "nemotron-3.5-asr-streaming-0.6b",
        alias = "Nemotron 3.5 ASR Streaming 0.6B"
    )]
    Nemotron35AsrStreaming06B,
    /// IBM Granite Speech 4.1 2B: punctuated, capitalised ASR and speech
    /// translation; IBM's default Granite Speech model.
    #[serde(
        rename = "Granite-Speech-4.1-2B",
        alias = "ibm-granite/granite-speech-4.1-2b",
        alias = "granite-speech-4.1-2b",
        alias = "Granite Speech 4.1 2B"
    )]
    GraniteSpeech412B,
    /// IBM Granite Speech 4.1 2B Plus rich transcription model: speaker
    /// turns and word timestamps, without punctuation by design.
    #[serde(
        rename = "Granite-Speech-4.1-2B-Plus",
        alias = "ibm-granite/granite-speech-4.1-2b-plus",
        alias = "granite-speech-4.1-2b-plus",
        alias = "Granite Speech 4.1 2B Plus"
    )]
    GraniteSpeech412BPlus,
    /// IBM Granite Speech 5.0 2B: punctuated multilingual ASR, speech
    /// translation, keyword biasing and language hints. Its own family: the
    /// language model uses YaRN RoPE over a 16k context.
    #[serde(
        rename = "Granite-Speech-5.0-2B",
        alias = "ibm-granite/granite-speech-5.0-2b",
        alias = "granite-speech-5.0-2b",
        alias = "Granite Speech 5.0 2B"
    )]
    GraniteSpeech502B,
    /// Granite Speech 5.0 2B as IBM's llama.cpp GGUF: the Q8_0 language
    /// model and the F16 audio encoder and projector (`mmproj`), with the
    /// configs and tokenizer of the safetensors release. The same model, so
    /// the same family.
    #[serde(
        rename = "Granite-Speech-5.0-2B-GGUF",
        alias = "ibm-granite/granite-speech-5.0-2b-GGUF",
        alias = "granite-speech-5.0-2b-gguf",
        alias = "Granite Speech 5.0 2B GGUF"
    )]
    GraniteSpeech502BGguf,
    /// Streaming Sortformer 4-speaker diarization model (.nemo)
    #[serde(rename = "diar_streaming_sortformer_4spk-v2.1")]
    DiarStreamingSortformer4SpkV21,
    /// NVIDIA Nemotron 3 8-speaker Sortformer-family diarization model (.nemo)
    #[serde(
        rename = "Nemotron-3-Diarization",
        alias = "nvidia/nemotron-3-diarization",
        alias = "nemotron-3-diarization",
        alias = "nemotron3-diarization",
        alias = "Nemotron 3 Diarization"
    )]
    Nemotron3Diarization,
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
    /// Qwen3 14B text model (GGUF Q4_K_M)
    #[serde(rename = "Qwen3-14B-GGUF")]
    Qwen314BGguf,
    /// Qwen3 MoE 30B-A3B text model (GGUF; sparse experts, ~3B active params).
    /// Catalog-disabled until real-checkpoint activation evidence exists
    /// (DS10 groundwork posture).
    #[serde(
        rename = "Qwen3-30B-A3B-GGUF",
        alias = "Qwen3-30B-A3B",
        alias = "unsloth/Qwen3-30B-A3B-GGUF",
        alias = "Qwen3-30B-A3B-Q4_K_M.gguf"
    )]
    Qwen3Moe30bA3bGguf,
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
    /// Qwen3.8 27B hybrid chat model with block-scaled FP8 Safetensors weights.
    #[serde(
        rename = "Qwen3.8-27B-FP8",
        alias = "Qwen/Qwen3.8-27B-FP8",
        alias = "qwen3.8-27b-fp8",
        alias = "Qwen3.8-27B"
    )]
    Qwen3827BFp8,
    /// Qwen3.6 35B-A3B hybrid sparse-MoE chat model with block-scaled FP8
    /// Safetensors weights (256 experts, 8 routed + 1 shared). Architecture
    /// twin of the originally served Qwen3.5-35B-A3B-FP8 checkpoint (same
    /// `Qwen3_5MoeForConditionalGeneration` geometry, tokenizer, and tensor
    /// map); only post-training differs, so the catalog carries exactly one
    /// checkpoint for the qwen3_5_moe architecture.
    #[serde(
        rename = "Qwen3.6-35B-A3B-FP8",
        alias = "Qwen/Qwen3.6-35B-A3B-FP8",
        alias = "qwen3.6-35b-a3b-fp8",
        alias = "Qwen3.6-35B-A3B"
    )]
    Qwen36Moe35BA3BFp8,
    /// Gemma 3 4B instruction-tuned chat model
    #[serde(rename = "Gemma-3-4b-it")]
    Gemma34BIt,
    /// Qwen3-ForcedAligner 0.6B model
    #[serde(rename = "Qwen3-ForcedAligner-0.6B")]
    Qwen3ForcedAligner06B,
    /// Voxtral Mini 4B Realtime model from Mistral AI
    #[serde(rename = "Voxtral-Mini-4B-Realtime-2602")]
    VoxtralMini4BRealtime2602,
    /// Voxtral 4B TTS model from Mistral AI
    #[serde(
        rename = "Voxtral-4B-TTS-2603",
        alias = "mistralai/Voxtral-4B-TTS-2603",
        alias = "voxtral-4b-tts-2603",
        alias = "Voxtral 4B TTS 2603"
    )]
    Voxtral4BTts2603,
    /// VibeVoice 1.5B long-form TTS model from Microsoft
    #[serde(
        rename = "VibeVoice-1.5B",
        alias = "microsoft/VibeVoice-1.5B",
        alias = "VibeVoice 1.5B",
        alias = "vibevoice-1.5b"
    )]
    VibeVoice15BTts,
    /// Fish Audio S2 Pro multilingual TTS model
    #[serde(
        rename = "FishAudio-S2-Pro",
        alias = "fishaudio/s2-pro",
        alias = "s2-pro",
        alias = "Fish Audio S2 Pro",
        alias = "FishAudio S2 Pro"
    )]
    FishAudioS2Pro,
    /// Resemble AI's Chatterbox (English, 0.5B Llama T3 + S3Gen)
    #[serde(
        rename = "Chatterbox-500M",
        alias = "ResembleAI/chatterbox",
        alias = "chatterbox",
        alias = "Chatterbox",
        alias = "Chatterbox 500M"
    )]
    Chatterbox500M,
    /// Resemble AI's Chatterbox Multilingual (23 languages, the same T3
    /// with a grapheme vocabulary, the English S3Gen)
    #[serde(
        rename = "Chatterbox-Multilingual-500M",
        alias = "chatterbox-multilingual",
        alias = "Chatterbox Multilingual",
        alias = "Chatterbox-Multilingual"
    )]
    ChatterboxMultilingual500M,
    /// Maya Research's Maya1 (3B Llama voice-design TTS over SNAC 24 kHz)
    #[serde(
        rename = "Maya1-3B",
        alias = "maya-research/maya1",
        alias = "maya1",
        alias = "Maya1",
        alias = "Maya-1",
        alias = "Maya1 3B"
    )]
    Maya13B,
    /// Maya1 with its decoder quantized to Q8_0 by llama.cpp's conversion
    /// (`mradermacher/maya1-GGUF`); config, tokenizer and SNAC as Maya1's
    #[serde(
        rename = "Maya1-3B-GGUF",
        alias = "mradermacher/maya1-GGUF",
        alias = "maya1-gguf",
        alias = "Maya1 GGUF",
        alias = "Maya1-GGUF"
    )]
    Maya13BGguf,
    /// Maya Research's Maya 2 Native (Qwen3-1.7B backbone + depth decoder
    /// over Kyutai Mimi; 12 Indic and English languages, 20 voices, cloning)
    #[serde(
        rename = "Maya2-Native-2.5B",
        alias = "maya-research/maya2-native",
        alias = "maya2-native",
        alias = "Maya2-Native",
        alias = "maya2",
        alias = "Maya2",
        alias = "Maya 2 Native"
    )]
    Maya2Native25B,
}

impl ModelVariant {
    /// Revision used for the first supported Qwen3.8-27B-FP8 artifact contract.
    pub const QWEN38_27B_FP8_ARTIFACT_REVISION: &'static str =
        "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a";
    /// Revision used for the first supported Qwen3.6-35B-A3B-FP8 artifact
    /// contract (HF repo state 2026-04-24).
    pub const QWEN36_MOE_35B_A3B_FP8_ARTIFACT_REVISION: &'static str =
        "95a723d08a9490559dae23d0cff1d9466213d989";
    /// Official Qwen3-TTS generation limit from upstream generation configs
    /// (`max_new_tokens` in Hugging Face `generation_config.json`).
    pub const QWEN3_TTS_MAX_OUTPUT_FRAMES: usize = 8192;
    /// Qwen3-TTS codec frame rate: the speech tokenizer's 24 kHz output over
    /// its 1,920-sample decode upsampling (`speech_tokenizer/config.json`).
    /// The "12Hz" in the variant name is rounded; budgets at 12 Hz cut 4% of
    /// the audio they meant to allow (QTTS-21).
    pub const QWEN3_TTS_FRAME_RATE_HZ: f32 = 24_000.0 / 1_920.0;
    pub const QWEN_CUSTOMVOICE_BUILT_IN_VOICE_COUNT: usize = 9;
    /// Kokoro's 54 voices less the 13 Japanese and Mandarin ones, which
    /// have no G2P yet and are hidden and refused.
    pub const KOKORO_BUILT_IN_VOICE_COUNT: usize = 41;
    pub const LFM25_AUDIO_BUILT_IN_VOICE_COUNT: usize = 4;
    /// LFM3-Audio speaks with a female or a male voice.
    pub const LFM3_AUDIO_BUILT_IN_VOICE_COUNT: usize = 2;
    /// LFM3-Audio frames per second: one frame is 1,920 samples at 24 kHz.
    pub const LFM3_AUDIO_FRAME_RATE_HZ: f32 = 12.5;
    /// Frames one LFM3-Audio TTS request speaks at most (about 164 s);
    /// longer text is split into sentences first.
    pub const LFM3_AUDIO_MAX_OUTPUT_FRAMES: usize = 2048;
    pub const VOXTRAL_TTS_BUILT_IN_VOICE_COUNT: usize = 20;
    pub const VOXTRAL_TTS_MAX_OUTPUT_FRAMES: usize = 1500;
    pub const VOXTRAL_TTS_CUDA_MAX_OUTPUT_FRAMES: usize = 2048;
    pub const VOXTRAL_TTS_FRAME_RATE_HZ: f32 = 12.5;
    pub const VIBEVOICE_TTS_MAX_OUTPUT_FRAMES: usize = 40_500;
    pub const VIBEVOICE_TTS_FRAME_RATE_HZ: f32 = 7.5;
    pub const FISH_S2_PRO_ARTIFACT_REVISION: &'static str =
        "1de9996b6be38b745688de084d87a5633f714e4e";
    pub const FISH_S2_PRO_MAX_OUTPUT_FRAMES: usize = 4096;
    /// Chatterbox's English checkpoint as published 2026-06-10; its five
    /// English files are unchanged since 2025-05-30.
    pub const CHATTERBOX_ARTIFACT_REVISION: &'static str =
        "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18";
    /// Upstream's per-utterance speech-token budget (`max_new_tokens`),
    /// about 40 s at 25 tokens per second.
    pub const CHATTERBOX_MAX_OUTPUT_FRAMES: usize = 1000;
    pub const CHATTERBOX_FRAME_RATE_HZ: f32 = 25.0;
    /// One bundled voice (`conds.pt`).
    pub const CHATTERBOX_BUILT_IN_VOICE_COUNT: usize = 1;
    /// Maya1 as published 2026-10-13; its weights and tokenizer are
    /// unchanged since 9ba3f6f9 (2025-11-03).
    pub const MAYA1_ARTIFACT_REVISION: &'static str = "b1b9b38ca9decbfcfd6c98ea93dd502bfd0f9572";
    /// The card's per-request budget (`max_new_tokens`), in decoder tokens:
    /// about 25 s, seven tokens to an 85 ms SNAC frame.
    pub const MAYA1_MAX_OUTPUT_FRAMES: usize = 2048;
    /// Decoder tokens per second of audio (7 per 2,048 samples at 24 kHz).
    pub const MAYA1_FRAME_RATE_HZ: f32 = 24_000.0 * 7.0 / 2_048.0;
    /// Named voice descriptions.
    pub const MAYA1_BUILT_IN_VOICE_COUNT: usize = 9;
    /// Maya 2 Native's release 1.0 (2026-10-25) as of its card rewrite.
    pub const MAYA2_ARTIFACT_REVISION: &'static str = "1403b841a6af0f920781319e851d4d6bdeb596c6";
    /// LiquidAI/LFM3-1.7B-GGUF (QAD-Q4_0 and Q8_0 files).
    pub const LFM3_17B_GGUF_ARTIFACT_REVISION: &'static str =
        "b3f2ae5d73cd3c91bf40e8ba752c648aa05dffdf";
    /// LiquidAI/LFM3-1.7B: the tokenizer, chat template and configs the GGUF
    /// repo does not ship.
    pub const LFM3_17B_COMPANION_REVISION: &'static str =
        "40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd";
    /// LiquidAI/LFM3-Audio-2B-GGUF (Q4_K_M and Q8_0 language models, Q8_0
    /// mmproj).
    pub const LFM3_AUDIO_2B_GGUF_ARTIFACT_REVISION: &'static str =
        "fcfae9027c00a6f3f872c48a841cfeb77e2b5a1c";
    /// LiquidAI/LFM3-Audio-2B: the tokenizer, chat template and configs the
    /// GGUF repo does not ship.
    pub const LFM3_AUDIO_2B_COMPANION_REVISION: &'static str =
        "9d49701c372d6011d900f6abb40c6b296fb706dd";
    /// Upstream's absolute per-piece ceiling: 1,500 frames (120 s).
    pub const MAYA2_MAX_OUTPUT_FRAMES: usize = 1_500;
    /// Mimi frames per second.
    pub const MAYA2_FRAME_RATE_HZ: f32 = 12.5;
    /// The release's named voices.
    pub const MAYA2_BUILT_IN_VOICE_COUNT: usize = 20;
    /// mradermacher/maya1-GGUF, whose `maya1.Q8_0.gguf` converts Maya1's
    /// unchanged weights.
    pub const MAYA1_GGUF_ARTIFACT_REVISION: &'static str =
        "595e0193e6ca6d4a2e5a7ea18e3bf8ada39aae53";
    /// The `.nemo` revisions izwi's Sortformer loader was verified against
    /// (sha256 `8abd3283…2ef8` and `867c53f5…e96d`); a re-published archive
    /// changes graph config and weights together (NDZ-5).
    pub const SORTFORMER_4SPK_V21_ARTIFACT_REVISION: &'static str =
        "cd03eee90fbec18297ac31b8c21546e596b7f71c";
    pub const NEMOTRON3_DIARIZATION_ARTIFACT_REVISION: &'static str =
        "f667ed73aee57d40cc39428eb768b4fd87a0a29e";
    /// The Granite Speech 4.1 revisions new downloads fetch. Plus is IBM's
    /// v2610 weights (2026-10-21: better speaker attribution, timestamps and
    /// long-form), with the language model under `language_model.model.model.*`
    /// and YaRN RoPE; the loader also reads installs of the earlier revision
    /// (1454e6e1), so they keep working without a re-download.
    pub const GRANITE_SPEECH_4_1_2B_ARTIFACT_REVISION: &'static str =
        "de575db64086f84fdc79da4932d1076e965bc546";
    pub const GRANITE_SPEECH_4_1_2B_PLUS_ARTIFACT_REVISION: &'static str =
        "e69efe37f97b6283c61f5a8dadda2b3eb67765d6";
    /// Granite Speech 5.0 2B as published 2026-10-23. The weights are from
    /// 818a37c6 (2026-10-07); later commits changed only the card and the
    /// config's `_name_or_path`.
    pub const GRANITE_SPEECH_5_0_2B_ARTIFACT_REVISION: &'static str =
        "0555e70346a8992da6b5df98ae26ec50c43958b8";
    /// The GGUF release as published 2026-10-15 (`granite-speech-5.0-2b-GGUF`).
    pub const GRANITE_SPEECH_5_0_2B_GGUF_ARTIFACT_REVISION: &'static str =
        "df3833f5e22e7d51c3f5dfedd4726260f7ec7cb5";
    pub const FISH_S2_PRO_NATIVE_CONTEXT_TOKENS: usize = 32_768;
    pub const LFM25_AUDIO_NATIVE_CONTEXT_TOKENS: usize = 32_768;
    pub const FISH_S2_PRO_FRAME_RATE_HZ: f32 = 21.5;

    /// Return a revision-pinned native context for artifacts whose conversion
    /// format is known to omit it. Embedded and sidecar metadata must still be
    /// checked for conflicts before this value is used.
    pub fn artifact_context_manifest(self) -> Option<ArtifactContextManifest> {
        match self {
            Self::Qwen3Asr06BGguf => Some(ArtifactContextManifest {
                native_context_tokens: 65_536,
                evidence: "Qwen/Qwen3-ASR-0.6B@9ba1d4a/config.json",
            }),
            Self::Qwen3Asr17BGguf => Some(ArtifactContextManifest {
                native_context_tokens: 65_536,
                evidence: "Qwen/Qwen3-ASR-1.7B@c85e7e3/config.json",
            }),
            _ => None,
        }
    }

    /// Pinned Hugging Face artifact revision, when mutable `main` is not an
    /// acceptable source for this variant.
    pub fn artifact_revision(self) -> Option<&'static str> {
        match self {
            Self::Qwen3827BFp8 => Some(Self::QWEN38_27B_FP8_ARTIFACT_REVISION),
            Self::Qwen36Moe35BA3BFp8 => Some(Self::QWEN36_MOE_35B_A3B_FP8_ARTIFACT_REVISION),
            Self::FishAudioS2Pro => Some(Self::FISH_S2_PRO_ARTIFACT_REVISION),
            Self::Chatterbox500M | Self::ChatterboxMultilingual500M => {
                Some(Self::CHATTERBOX_ARTIFACT_REVISION)
            }
            Self::Maya13B => Some(Self::MAYA1_ARTIFACT_REVISION),
            Self::Maya13BGguf => Some(Self::MAYA1_GGUF_ARTIFACT_REVISION),
            Self::Maya2Native25B => Some(Self::MAYA2_ARTIFACT_REVISION),
            Self::Lfm317BGguf | Self::Lfm317BQ8Gguf => Some(Self::LFM3_17B_GGUF_ARTIFACT_REVISION),
            Self::Lfm3Audio2BGguf | Self::Lfm3Audio2BQ8Gguf => {
                Some(Self::LFM3_AUDIO_2B_GGUF_ARTIFACT_REVISION)
            }
            Self::DiarStreamingSortformer4SpkV21 => {
                Some(Self::SORTFORMER_4SPK_V21_ARTIFACT_REVISION)
            }
            Self::Nemotron3Diarization => Some(Self::NEMOTRON3_DIARIZATION_ARTIFACT_REVISION),
            Self::GraniteSpeech412B => Some(Self::GRANITE_SPEECH_4_1_2B_ARTIFACT_REVISION),
            Self::GraniteSpeech412BPlus => Some(Self::GRANITE_SPEECH_4_1_2B_PLUS_ARTIFACT_REVISION),
            Self::GraniteSpeech502B => Some(Self::GRANITE_SPEECH_5_0_2B_ARTIFACT_REVISION),
            Self::GraniteSpeech502BGguf => Some(Self::GRANITE_SPEECH_5_0_2B_GGUF_ARTIFACT_REVISION),
            _ => None,
        }
    }

    /// Get HuggingFace repository ID
    pub fn repo_id(&self) -> &'static str {
        match self {
            Self::Qwen3Tts12Hz06BBase => "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            Self::Qwen3Tts12Hz06BBase8Bit => "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
            Self::Qwen3Tts12Hz06BBaseBf16 => "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
            Self::Qwen3Tts12Hz06BCustomVoice => "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            Self::Qwen3Tts12Hz06BCustomVoice8Bit => {
                "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"
            }
            Self::Qwen3Tts12Hz06BCustomVoiceBf16 => {
                "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16"
            }
            Self::Qwen3Tts12Hz17BBase => "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            Self::Qwen3Tts12Hz17BCustomVoice => "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            Self::Qwen3Tts12Hz17BVoiceDesign => "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            Self::Qwen3Tts12Hz17BVoiceDesign8Bit => {
                "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"
            }
            Self::Qwen3Tts12Hz17BVoiceDesignBf16 => {
                "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
            }
            Self::Qwen3TtsTokenizer12Hz => "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            Self::Lfm2512BInstructGguf => "LiquidAI/LFM2.5-1.2B-Instruct-GGUF",
            Self::Lfm2512BThinkingGguf => "LiquidAI/LFM2.5-1.2B-Thinking-GGUF",
            Self::Lfm25Audio15BGguf => "LiquidAI/LFM2.5-Audio-1.5B-GGUF",
            Self::Lfm317BGguf | Self::Lfm317BQ8Gguf => "LiquidAI/LFM3-1.7B-GGUF",
            Self::Lfm3Audio2BGguf | Self::Lfm3Audio2BQ8Gguf => "LiquidAI/LFM3-Audio-2B-GGUF",
            Self::Kokoro82M => "hexgrad/Kokoro-82M",
            Self::ParakeetTdt06BV3 => "nvidia/parakeet-tdt-0.6b-v3",
            Self::WhisperLargeV3Turbo => "openai/whisper-large-v3-turbo",
            Self::Qwen3Asr06BGguf | Self::Qwen3Asr17BGguf => "Alkd/qwen3-asr-gguf",
            Self::VibeVoiceAsr => "microsoft/VibeVoice-ASR",
            Self::Nemotron35AsrStreaming06B => "nvidia/nemotron-3.5-asr-streaming-0.6b",
            Self::GraniteSpeech412B => "ibm-granite/granite-speech-4.1-2b",
            Self::GraniteSpeech412BPlus => "ibm-granite/granite-speech-4.1-2b-plus",
            Self::GraniteSpeech502B => "ibm-granite/granite-speech-5.0-2b",
            Self::GraniteSpeech502BGguf => "ibm-granite/granite-speech-5.0-2b-GGUF",
            Self::DiarStreamingSortformer4SpkV21 => "nvidia/diar_streaming_sortformer_4spk-v2.1",
            Self::Nemotron3Diarization => "nvidia/Nemotron-3-Diarization",
            Self::Qwen306B => "Qwen/Qwen3-0.6B",
            Self::Qwen306B4Bit => "mlx-community/Qwen3-0.6B-4bit",
            Self::Qwen317B => "Qwen/Qwen3-1.7B",
            Self::Qwen317B4Bit => "mlx-community/Qwen3-1.7B-4bit",
            Self::Qwen314BGguf => "Qwen/Qwen3-14B-GGUF",
            Self::Qwen3Moe30bA3bGguf => "Qwen/Qwen3-30B-A3B-GGUF",
            Self::Qwen3508BGguf => "unsloth/Qwen3.5-0.8B-GGUF",
            Self::Qwen352BGguf => "unsloth/Qwen3.5-2B-GGUF",
            Self::Qwen354BGguf => "unsloth/Qwen3.5-4B-GGUF",
            Self::Qwen359BGguf => "unsloth/Qwen3.5-9B-GGUF",
            Self::Qwen3827BFp8 => "Qwen/Qwen3.8-27B-FP8",
            Self::Qwen36Moe35BA3BFp8 => "Qwen/Qwen3.6-35B-A3B-FP8",
            Self::Gemma34BIt => "google/gemma-3-4b-it",
            Self::Qwen3ForcedAligner06B => "Qwen/Qwen3-ForcedAligner-0.6B",
            Self::VoxtralMini4BRealtime2602 => "mistralai/Voxtral-Mini-4B-Realtime-2602",
            Self::Voxtral4BTts2603 => "mistralai/Voxtral-4B-TTS-2603",
            Self::VibeVoice15BTts => "microsoft/VibeVoice-1.5B",
            Self::FishAudioS2Pro => "fishaudio/s2-pro",
            Self::Chatterbox500M | Self::ChatterboxMultilingual500M => "ResembleAI/chatterbox",
            Self::Maya13B => "maya-research/maya1",
            Self::Maya13BGguf => "mradermacher/maya1-GGUF",
            Self::Maya2Native25B => "maya-research/maya2-native",
        }
    }

    /// Get human-readable name
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Qwen3Tts12Hz06BBase => "Qwen3-TTS 0.6B Base",
            Self::Qwen3Tts12Hz06BBase8Bit => "Qwen3-TTS 0.6B Base 8-bit",
            Self::Qwen3Tts12Hz06BBaseBf16 => "Qwen3-TTS 0.6B Base bf16",
            Self::Qwen3Tts12Hz06BCustomVoice => "Qwen3-TTS 0.6B CustomVoice",
            Self::Qwen3Tts12Hz06BCustomVoice8Bit => "Qwen3-TTS 0.6B CustomVoice 8-bit",
            Self::Qwen3Tts12Hz06BCustomVoiceBf16 => "Qwen3-TTS 0.6B CustomVoice bf16",
            Self::Qwen3Tts12Hz17BBase => "Qwen3-TTS 1.7B Base",
            Self::Qwen3Tts12Hz17BCustomVoice => "Qwen3-TTS 1.7B CustomVoice",
            Self::Qwen3Tts12Hz17BVoiceDesign => "Qwen3-TTS 1.7B VoiceDesign",
            Self::Qwen3Tts12Hz17BVoiceDesign8Bit => "Qwen3-TTS 1.7B VoiceDesign 8-bit",
            Self::Qwen3Tts12Hz17BVoiceDesignBf16 => "Qwen3-TTS 1.7B VoiceDesign bf16",
            Self::Qwen3TtsTokenizer12Hz => "Qwen3-TTS Tokenizer 12Hz",
            Self::Lfm2512BInstructGguf => "LFM2.5 1.2B Instruct GGUF",
            Self::Lfm2512BThinkingGguf => "LFM2.5 1.2B Thinking GGUF",
            Self::Lfm25Audio15BGguf => "LFM2.5 Audio 1.5B GGUF",
            Self::Lfm317BGguf => "LFM3 1.7B GGUF",
            Self::Lfm317BQ8Gguf => "LFM3 1.7B Q8_0 GGUF",
            Self::Lfm3Audio2BGguf => "LFM3 Audio 2B GGUF",
            Self::Lfm3Audio2BQ8Gguf => "LFM3 Audio 2B Q8_0 GGUF",
            Self::Kokoro82M => "Kokoro 82M",
            Self::ParakeetTdt06BV3 => "Parakeet TDT 0.6B v3",
            Self::WhisperLargeV3Turbo => "Whisper Large v3 Turbo",
            Self::Qwen3Asr06BGguf => "Qwen3 ASR 0.6B GGUF",
            Self::Qwen3Asr17BGguf => "Qwen3 ASR 1.7B GGUF",
            Self::VibeVoiceAsr => "VibeVoice ASR",
            Self::Nemotron35AsrStreaming06B => "Nemotron 3.5 ASR Streaming 0.6B",
            Self::GraniteSpeech412B => "Granite Speech 4.1 2B",
            Self::GraniteSpeech412BPlus => "Granite Speech 4.1 2B Plus",
            Self::GraniteSpeech502B => "Granite Speech 5.0 2B",
            Self::GraniteSpeech502BGguf => "Granite Speech 5.0 2B GGUF",
            Self::DiarStreamingSortformer4SpkV21 => "Streaming Sortformer 4spk v2.1",
            Self::Nemotron3Diarization => "Nemotron 3 Diarization",
            Self::Qwen306B => "Qwen3 0.6B",
            Self::Qwen306B4Bit => "Qwen3 0.6B 4-bit",
            Self::Qwen317B => "Qwen3 1.7B",
            Self::Qwen317B4Bit => "Qwen3 1.7B 4-bit",
            Self::Qwen314BGguf => "Qwen3 14B GGUF",
            Self::Qwen3Moe30bA3bGguf => "Qwen3 MoE 30B-A3B GGUF",
            Self::Qwen3508BGguf => "Qwen3.5 0.8B GGUF",
            Self::Qwen352BGguf => "Qwen3.5 2B GGUF",
            Self::Qwen354BGguf => "Qwen3.5 4B GGUF",
            Self::Qwen359BGguf => "Qwen3.5 9B GGUF",
            Self::Qwen3827BFp8 => "Qwen3.8 27B FP8",
            Self::Qwen36Moe35BA3BFp8 => "Qwen3.6 35B-A3B FP8",
            Self::Gemma34BIt => "Gemma 3 4B Instruct",
            Self::Qwen3ForcedAligner06B => "Qwen3-ForcedAligner 0.6B",
            Self::VoxtralMini4BRealtime2602 => "Voxtral Mini 4B Realtime",
            Self::Voxtral4BTts2603 => "Voxtral 4B TTS",
            Self::VibeVoice15BTts => "VibeVoice 1.5B TTS",
            Self::FishAudioS2Pro => "Fish Audio S2 Pro",
            Self::Chatterbox500M => "Chatterbox",
            Self::ChatterboxMultilingual500M => "Chatterbox Multilingual",
            Self::Maya13B => "Maya1",
            Self::Maya13BGguf => "Maya1 GGUF",
            Self::Maya2Native25B => "Maya 2 Native",
        }
    }

    /// Get local directory name
    pub fn dir_name(&self) -> &'static str {
        match self {
            Self::Qwen3Tts12Hz06BBase => "Qwen3-TTS-12Hz-0.6B-Base",
            Self::Qwen3Tts12Hz06BBase8Bit => "Qwen3-TTS-12Hz-0.6B-Base-8bit",
            Self::Qwen3Tts12Hz06BBaseBf16 => "Qwen3-TTS-12Hz-0.6B-Base-bf16",
            Self::Qwen3Tts12Hz06BCustomVoice => "Qwen3-TTS-12Hz-0.6B-CustomVoice",
            Self::Qwen3Tts12Hz06BCustomVoice8Bit => "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
            Self::Qwen3Tts12Hz06BCustomVoiceBf16 => "Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16",
            Self::Qwen3Tts12Hz17BBase => "Qwen3-TTS-12Hz-1.7B-Base",
            Self::Qwen3Tts12Hz17BCustomVoice => "Qwen3-TTS-12Hz-1.7B-CustomVoice",
            Self::Qwen3Tts12Hz17BVoiceDesign => "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            Self::Qwen3Tts12Hz17BVoiceDesign8Bit => "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
            Self::Qwen3Tts12Hz17BVoiceDesignBf16 => "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
            Self::Qwen3TtsTokenizer12Hz => "Qwen3-TTS-Tokenizer-12Hz",
            Self::Lfm2512BInstructGguf => "LFM2.5-1.2B-Instruct-GGUF",
            Self::Lfm2512BThinkingGguf => "LFM2.5-1.2B-Thinking-GGUF",
            Self::Lfm25Audio15BGguf => "LFM2.5-Audio-1.5B-GGUF",
            Self::Lfm317BGguf => "LFM3-1.7B-GGUF",
            Self::Lfm317BQ8Gguf => "LFM3-1.7B-Q8_0-GGUF",
            Self::Lfm3Audio2BGguf => "LFM3-Audio-2B-GGUF",
            Self::Lfm3Audio2BQ8Gguf => "LFM3-Audio-2B-Q8_0-GGUF",
            Self::Kokoro82M => "Kokoro-82M",
            Self::ParakeetTdt06BV3 => "Parakeet-TDT-0.6B-v3",
            Self::WhisperLargeV3Turbo => "Whisper-Large-v3-Turbo",
            Self::Qwen3Asr06BGguf => "Qwen3-ASR-0.6B-GGUF",
            Self::Qwen3Asr17BGguf => "Qwen3-ASR-1.7B-GGUF",
            Self::VibeVoiceAsr => "VibeVoice-ASR",
            Self::Nemotron35AsrStreaming06B => "Nemotron-3.5-ASR-Streaming-0.6B",
            Self::GraniteSpeech412B => "Granite-Speech-4.1-2B",
            Self::GraniteSpeech412BPlus => "Granite-Speech-4.1-2B-Plus",
            Self::GraniteSpeech502B => "Granite-Speech-5.0-2B",
            Self::GraniteSpeech502BGguf => "Granite-Speech-5.0-2B-GGUF",
            Self::DiarStreamingSortformer4SpkV21 => "diar_streaming_sortformer_4spk-v2.1",
            Self::Nemotron3Diarization => "Nemotron-3-Diarization",
            Self::Qwen306B => "Qwen3-0.6B",
            Self::Qwen306B4Bit => "Qwen3-0.6B-4bit",
            Self::Qwen317B => "Qwen3-1.7B",
            Self::Qwen317B4Bit => "Qwen3-1.7B-4bit",
            Self::Qwen314BGguf => "Qwen3-14B-GGUF",
            Self::Qwen3Moe30bA3bGguf => "Qwen3-30B-A3B-GGUF",
            Self::Qwen3508BGguf => "Qwen3.5-0.8B",
            Self::Qwen352BGguf => "Qwen3.5-2B",
            Self::Qwen354BGguf => "Qwen3.5-4B",
            Self::Qwen359BGguf => "Qwen3.5-9B",
            Self::Qwen3827BFp8 => "Qwen3.8-27B-FP8",
            Self::Qwen36Moe35BA3BFp8 => "Qwen3.6-35B-A3B-FP8",
            Self::Gemma34BIt => "Gemma-3-4b-it",
            Self::Qwen3ForcedAligner06B => "Qwen3-ForcedAligner-0.6B",
            Self::VoxtralMini4BRealtime2602 => "Voxtral-Mini-4B-Realtime-2602",
            Self::Voxtral4BTts2603 => "Voxtral-4B-TTS-2603",
            Self::VibeVoice15BTts => "VibeVoice-1.5B",
            Self::FishAudioS2Pro => "FishAudio-S2-Pro",
            Self::Chatterbox500M => "Chatterbox-500M",
            Self::ChatterboxMultilingual500M => "Chatterbox-Multilingual-500M",
            Self::Maya13B => "Maya1-3B",
            Self::Maya13BGguf => "Maya1-3B-GGUF",
            Self::Maya2Native25B => "Maya2-Native-2.5B",
        }
    }

    /// Estimated model size in bytes
    pub fn estimated_size(&self) -> u64 {
        match self {
            Self::Qwen3Tts12Hz06BBase => 2_516_106_051, // ~2.34 GB
            Self::Qwen3Tts12Hz06BBase8Bit => 1_991_299_138, // ~1.85 GB
            Self::Qwen3Tts12Hz06BBaseBf16 => 2_516_143_009, // ~2.34 GB
            Self::Qwen3Tts12Hz06BCustomVoice => 2_498_388_392, // ~2.33 GB
            Self::Qwen3Tts12Hz06BCustomVoice8Bit => 1_973_575_388, // ~1.84 GB
            Self::Qwen3Tts12Hz06BCustomVoiceBf16 => 2_498_419_405, // ~2.33 GB
            Self::Qwen3Tts12Hz17BBase => 4_544_229_700, // ~4.23 GB
            Self::Qwen3Tts12Hz17BCustomVoice => 4_520_218_951, // ~4.21 GB
            Self::Qwen3Tts12Hz17BVoiceDesign => 4_520_163_832, // ~4.21 GB
            Self::Qwen3Tts12Hz17BVoiceDesign8Bit => 3_080_140_867, // ~2.87 GB
            Self::Qwen3Tts12Hz17BVoiceDesignBf16 => 4_520_194_992, // ~4.21 GB
            Self::Qwen3TtsTokenizer12Hz => 682_300_739, // ~0.64 GB
            Self::Lfm2512BInstructGguf => 730_895_168,  // ~0.68 GB (GGUF Q4_K_M, HF tree)
            Self::Lfm2512BThinkingGguf => 730_895_360,  // ~0.68 GB (GGUF Q4_K_M, HF tree)
            Self::Lfm25Audio15BGguf => 1_220_000_000,   // ~1.14 GB (four-file GGUF bundle, est)
            // QAD-Q4_0 GGUF 1,049,615,680 B + companion files 17,924,183 B (HF tree @ pins)
            Self::Lfm317BGguf => 1_067_539_863,
            // Q8_0 GGUF 1,846,238,496 B + companion files 17,924,183 B
            Self::Lfm317BQ8Gguf => 1_864_162_679,
            // Q4_K_M GGUF 1,099,626,016 B + mmproj Q8_0 337,862,080 B +
            // companion files 17,925,353 B (HF tree @ pins)
            Self::Lfm3Audio2BGguf => 1_455_413_449,
            // Q8_0 GGUF 1,846,236,704 B + the same mmproj and companions
            Self::Lfm3Audio2BQ8Gguf => 2_202_024_137,
            Self::Kokoro82M => 363_323_757,             // ~346 MB (HF tree total, Apr 2025)
            Self::ParakeetTdt06BV3 => 2_509_332_480,    // .nemo 2,509,332,480 B (HF tree, Oct 2026)
            Self::WhisperLargeV3Turbo => 1_617_824_864, // ~1.51 GB (HF x-linked-size)
            Self::Qwen3Asr06BGguf => 1_012_824_608,     // HF x-linked-size, Mar 2026
            Self::Qwen3Asr17BGguf => 2_512_740_320,     // HF x-linked-size, Mar 2026
            Self::VibeVoiceAsr => 17_348_198_410,       // safetensors index metadata total_size
            Self::Nemotron35AsrStreaming06B => 2_368_284_501, // .nemo bytes, HF tree, Oct 2026
            Self::GraniteSpeech412B => 4_626_414_392,   // safetensors index metadata total_size
            Self::GraniteSpeech412BPlus => 4_223_757_112, // safetensors index metadata total_size
            Self::GraniteSpeech502B => 4_626_414_392,   // model.safetensors tensor bytes
            Self::GraniteSpeech502BGguf => 3_122_684_382, // Q8_0 + mmproj GGUFs and companions
            Self::DiarStreamingSortformer4SpkV21 => 471_367_680, // .nemo bytes, HF tree, Oct 2026
            Self::Nemotron3Diarization => 198_676_480,  // .nemo bytes, HF tree, Oct 2026
            Self::Qwen306B => 1_520_000_000,            // ~1.42 GB (est)
            Self::Qwen306B4Bit => 900_000_000,          // ~0.84 GB (est)
            Self::Qwen317B => 4_080_000_000,            // ~3.80 GB (actual: 3.44GB + 622MB shards)
            Self::Qwen317B4Bit => 1_115_700_000,        // ~1.04 GB
            Self::Qwen314BGguf => 9_200_000_000, // ~8.57 GB (Q4_K_M est)
            Self::Qwen3Moe30bA3bGguf => 18_600_000_000, // ~17.3 GB Q4_K_M estimate (128 routed experts, ~3B active)
            Self::Qwen3508BGguf => 737_504_352,         // local GGUF + mmproj + tokenizer assets
            Self::Qwen352BGguf => 1_949_063_104,        // local GGUF + mmproj + tokenizer assets
            Self::Qwen354BGguf => 3_413_361_504,        // local GGUF + mmproj + tokenizer assets
            Self::Qwen359BGguf => 6_598_688_544,        // local GGUF + mmproj + tokenizer assets
            Self::Qwen3827BFp8 => 30_889_968_808, // pinned indexed FP8 bundle + chat/tokenizer metadata
            Self::Qwen36Moe35BA3BFp8 => 37_470_000_000, // ~34.9 GiB indexed FP8 safetensors + chat/tokenizer metadata (HF model-card total; re-pinned at download)
            Self::Gemma34BIt => 8_600_000_000,          // ~8.01 GB (est)
            Self::Qwen3ForcedAligner06B => 1_840_072_459, // ~1.71 GB
            Self::VoxtralMini4BRealtime2602 => 8_874_376_007, // resolved download total, Oct 2026
            Self::Voxtral4BTts2603 => 8_650_000_000,    // ~8.04 GB plus voice assets
            Self::VibeVoice15BTts => 5_408_043_974,     // safetensors index metadata total_size
            Self::FishAudioS2Pro => 11_520_000_000, // ~10.7 GiB current HF bundle, including codec.pth
            Self::Chatterbox500M => 3_229_396_676, // five pinned HF files (3,191,966,992 B) + PerTh checkpoint (37,429,684 B)
            // Its T3 and tokenizer (2,144,059,741 B), the English S3Gen,
            // voice encoder and default voice (1,062,287,778 B), PerTh, and
            // the Chinese Cangjie table and pkuseg segmenter (41,013,265 B).
            Self::ChatterboxMultilingual500M => 3_284_790_468,
            // Two BF16 shards, their index, config and tokenizer at the pin
            // (6,624,761,513 B), and SNAC 24 kHz's pinned checkpoint
            // (79,488,254 B).
            Self::Maya13B => 6_704_249_767,
            // The Q8_0 GGUF (3,516,496,480 B), Maya1's config and tokenizer
            // (22,854,118 B) and SNAC.
            Self::Maya13BGguf => 3_618_838_852,
            // Both BF16 shards, their index, the release config, backbone
            // config and tokenizer and the voice list at the pin
            // (5,021,048,540 B), Mimi's config and weights (384,650,945 B)
            // and ECAPA's embedding model (83,316,686 B).
            Self::Maya2Native25B => 5_489_016_171,
        }
    }

    /// Memory required for inference
    pub fn memory_required_gb(&self) -> f32 {
        match self {
            Self::Qwen3Tts12Hz06BBase
            | Self::Qwen3Tts12Hz06BBase8Bit
            | Self::Qwen3Tts12Hz06BBaseBf16
            | Self::Qwen3Tts12Hz06BCustomVoice
            | Self::Qwen3Tts12Hz06BCustomVoice8Bit
            | Self::Qwen3Tts12Hz06BCustomVoiceBf16 => 2.5,
            Self::Qwen3Tts12Hz17BBase
            | Self::Qwen3Tts12Hz17BCustomVoice
            | Self::Qwen3Tts12Hz17BVoiceDesign
            | Self::Qwen3Tts12Hz17BVoiceDesign8Bit
            | Self::Qwen3Tts12Hz17BVoiceDesignBf16 => 6.0,
            Self::Qwen3TtsTokenizer12Hz => 1.0,
            Self::Lfm2512BInstructGguf | Self::Lfm2512BThinkingGguf => 2.0,
            Self::Lfm25Audio15BGguf => 5.0,
            Self::Lfm317BGguf => 2.0,
            Self::Lfm317BQ8Gguf => 3.0,
            Self::Lfm3Audio2BGguf => 3.0,
            Self::Lfm3Audio2BQ8Gguf => 4.0,
            Self::Kokoro82M => 2.0,
            Self::ParakeetTdt06BV3 => 12.0,
            Self::WhisperLargeV3Turbo => 4.0,
            Self::Qwen3Asr06BGguf => 4.0,
            Self::Qwen3Asr17BGguf => 8.0,
            Self::VibeVoiceAsr => 36.0,
            Self::Nemotron35AsrStreaming06B => 6.0,
            // Same architecture as plus; base's untied head adds 0.4 GB BF16.
            Self::GraniteSpeech412B | Self::GraniteSpeech412BPlus => 10.0,
            // The same weights as 4.1 with a 16k-token decoder context.
            Self::GraniteSpeech502B => 12.0,
            Self::GraniteSpeech502BGguf => 8.0,
            // The diarizers and the forced aligner are priced by their
            // residency plans (`asr_memory`); these figures cover the largest
            // plan resident on any backend. Sortformer loads F32 weights:
            // 0.44 GiB (v2.1) and 0.37 GiB (Nemotron-3) on every backend.
            Self::DiarStreamingSortformer4SpkV21 => 0.5,
            Self::Nemotron3Diarization => 0.4,
            Self::Qwen306B => 3.0,
            Self::Qwen306B4Bit => 2.0,
            Self::Qwen317B => 5.0,
            Self::Qwen317B4Bit => 3.0,
            Self::Qwen314BGguf => 16.0,
            Self::Qwen3Moe30bA3bGguf => 32.0,
            Self::Qwen3508BGguf => 3.5,
            Self::Qwen352BGguf => 5.5,
            Self::Qwen354BGguf => 9.0,
            Self::Qwen359BGguf => 16.0,
            // Backend-specific admission replaces this coarse catalog hint:
            // packed FP8 CUDA needs less, while the CPU F32 fallback needs more.
            // No load path reads it as Qwen3.8's footprint; it is not shown to
            // users either (`ModelInfo` carries no memory figure).
            Self::Qwen3827BFp8 => 80.0,
            // Same contract as Qwen38_27B_FP8: backend-specific admission
            // replaces this hint. 140.0 is the worst-case CPU F32 expanded
            // envelope (34.45B FP8 params + 1.5B BF16 params, dequantized);
            // the qwen3_5_moe representation plan replaces it per backend.
            Self::Qwen36Moe35BA3BFp8 => 140.0,
            Self::Gemma34BIt => 11.0,
            // The aligner expands to F32 on the CPU (3.42 GiB); Metal (F16
            // text) holds 2.69 GiB and CUDA (BF16 text) 2.30 GiB.
            Self::Qwen3ForcedAligner06B => 3.5,
            Self::VoxtralMini4BRealtime2602 => 16.0,
            Self::Voxtral4BTts2603 => 16.0,
            Self::VibeVoice15BTts => 12.0,
            Self::FishAudioS2Pro => 24.0,
            // 3.2 GB of F32 weights, two guided KV branches (~0.6 GB per
            // 1,000-token request at F32) and the flow's attention workspace.
            Self::Chatterbox500M | Self::ChatterboxMultilingual500M => 6.0,
            // 6.6 GB of decoder weights at F16/BF16 (13.2 GB in F32 on the
            // CPU, priced by its residency plan), ~0.25 GB of KV per
            // 2,300-position request and SNAC.
            Self::Maya13B => 9.0,
            // 3.5 GB of Q8_0 weights, ~0.5 GB of F32 KV per 2,300-position
            // request and SNAC.
            Self::Maya13BGguf => 5.0,
            // 5.0 GB of weights at F16/BF16 (10 GB in F32 on the CPU, priced
            // by its residency plan), Mimi in F32, ~0.22 GB of KV per
            // 2,048-position request.
            Self::Maya2Native25B => 7.0,
        }
    }

    /// Whether this is a tokenizer/codec model
    pub fn is_tokenizer(&self) -> bool {
        matches!(self.family(), crate::catalog::ModelFamily::Tokenizer)
    }

    /// Whether this is a Kokoro TTS model
    pub fn is_kokoro(&self) -> bool {
        matches!(self.family(), crate::catalog::ModelFamily::KokoroTts)
    }

    /// Whether this is an ASR model
    pub fn is_asr(&self) -> bool {
        matches!(
            self.family(),
            crate::catalog::ModelFamily::ParakeetAsr
                | crate::catalog::ModelFamily::WhisperAsr
                | crate::catalog::ModelFamily::Qwen3Asr
                | crate::catalog::ModelFamily::VibeVoiceAsr
                | crate::catalog::ModelFamily::NemotronAsr
                | crate::catalog::ModelFamily::GraniteSpeechAsr
                | crate::catalog::ModelFamily::GraniteSpeech5Asr
        )
    }

    /// Whether this is a diarization model.
    pub fn is_diarization(&self) -> bool {
        matches!(
            self.family(),
            crate::catalog::ModelFamily::SortformerDiarization
        )
    }

    /// The speaker channels a diarization checkpoint emits: the most speakers
    /// it can tell apart, and the bound on `min_speakers`/`max_speakers`.
    pub fn diarization_speaker_channels(&self) -> Option<usize> {
        match self {
            Self::DiarStreamingSortformer4SpkV21 => Some(4),
            Self::Nemotron3Diarization => Some(8),
            _ => None,
        }
    }

    /// Whether this model can produce persisted diarization records.
    pub fn supports_diarization_records(&self) -> bool {
        self.is_diarization()
    }

    /// Whether this model can produce model-native speaker-attributed ASR text.
    pub fn supports_speaker_attributed_asr(&self) -> bool {
        matches!(self, Self::GraniteSpeech412BPlus)
    }

    /// Whether a transcription request may ask this model to translate the
    /// speech into another language instead of transcribing it.
    pub fn supports_asr_translation(&self) -> bool {
        matches!(self, Self::GraniteSpeech502B | Self::GraniteSpeech502BGguf)
    }

    /// Whether a transcription request may bias this model toward a list of
    /// keywords (names, product terms, acronyms).
    pub fn supports_asr_keyword_biasing(&self) -> bool {
        matches!(self, Self::GraniteSpeech502B | Self::GraniteSpeech502BGguf)
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
                | crate::catalog::ModelFamily::Qwen3MoeChat
                | crate::catalog::ModelFamily::Qwen35Chat
                | crate::catalog::ModelFamily::Qwen35MoeChat
                | crate::catalog::ModelFamily::Qwen38Chat
                | crate::catalog::ModelFamily::Gemma3Chat
                | crate::catalog::ModelFamily::Lfm2Chat
                | crate::catalog::ModelFamily::Lfm3Chat
        )
    }

    /// Whether this is an audio-chat model.
    pub fn is_audio_chat(&self) -> bool {
        matches!(
            self.family(),
            crate::catalog::ModelFamily::Lfm25Audio | crate::catalog::ModelFamily::Lfm3Audio
        )
    }

    pub fn is_tts(&self) -> bool {
        matches!(
            self.family(),
            crate::catalog::ModelFamily::Qwen3Tts
                | crate::catalog::ModelFamily::KokoroTts
                | crate::catalog::ModelFamily::VoxtralTts
                | crate::catalog::ModelFamily::VibeVoiceTts
                | crate::catalog::ModelFamily::FishS2Tts
                | crate::catalog::ModelFamily::ChatterboxTts
                | crate::catalog::ModelFamily::Maya1Tts
                | crate::catalog::ModelFamily::Maya2Tts
        )
    }

    /// Model-weight license label, when the model has a license that should be
    /// surfaced independently from the Izwi application license.
    pub fn license_label(&self) -> Option<&'static str> {
        match self {
            Self::Voxtral4BTts2603 => Some("CC BY-NC 4.0"),
            Self::VibeVoiceAsr
            | Self::VibeVoice15BTts
            | Self::Chatterbox500M
            | Self::ChatterboxMultilingual500M => Some("MIT"),
            Self::FishAudioS2Pro => Some("Fish Audio Research License"),
            Self::Nemotron35AsrStreaming06B | Self::Nemotron3Diarization => Some("OpenMDW-1.1"),
            Self::GraniteSpeech412B
            | Self::GraniteSpeech412BPlus
            | Self::GraniteSpeech502B
            | Self::GraniteSpeech502BGguf
            | Self::Qwen3827BFp8
            | Self::Qwen36Moe35BA3BFp8
            | Self::Maya13B
            | Self::Maya13BGguf
            | Self::Maya2Native25B => Some("Apache-2.0"),
            Self::Lfm317BGguf
            | Self::Lfm317BQ8Gguf
            | Self::Lfm3Audio2BGguf
            | Self::Lfm3Audio2BQ8Gguf => Some("LFM Open License v1.0"),
            _ => None,
        }
    }

    /// Whether a streamed request runs on the TTS capability's own
    /// native-streaming graph instead of a separately loaded streaming
    /// capability. The engine binds every TTS request to its TTS adapter, so
    /// these families stream without a second copy of their retained state.
    pub fn streams_on_its_tts_route(&self) -> bool {
        matches!(
            self.family(),
            crate::catalog::ModelFamily::Lfm25Audio
                | crate::catalog::ModelFamily::Lfm3Audio
                | crate::catalog::ModelFamily::VibeVoiceTts
                | crate::catalog::ModelFamily::VoxtralTts
                | crate::catalog::ModelFamily::ChatterboxTts
                | crate::catalog::ModelFamily::Maya1Tts
                | crate::catalog::ModelFamily::Maya2Tts
        )
    }

    pub fn speech_capabilities(&self) -> Option<SpeechModelCapabilities> {
        let capabilities = match self {
            Self::Qwen3Tts12Hz06BBase
            | Self::Qwen3Tts12Hz06BBase8Bit
            | Self::Qwen3Tts12Hz06BBaseBf16
            | Self::Qwen3Tts12Hz17BBase => SpeechModelCapabilities {
                supports_builtin_voices: false,
                built_in_voice_count: None,
                supports_reference_voice: true,
                reference_transcript_required: true,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: false,
                supports_auto_long_form: true,
            },
            Self::Qwen3Tts12Hz06BCustomVoice
            | Self::Qwen3Tts12Hz06BCustomVoice8Bit
            | Self::Qwen3Tts12Hz06BCustomVoiceBf16 => SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(Self::QWEN_CUSTOMVOICE_BUILT_IN_VOICE_COUNT),
                supports_reference_voice: false,
                reference_transcript_required: false,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: false,
                supports_auto_long_form: true,
            },
            Self::Qwen3Tts12Hz17BCustomVoice => SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(Self::QWEN_CUSTOMVOICE_BUILT_IN_VOICE_COUNT),
                supports_reference_voice: false,
                reference_transcript_required: false,
                supports_voice_description: true,
                supports_streaming: true,
                supports_speed_control: false,
                supports_auto_long_form: true,
            },
            Self::Qwen3Tts12Hz17BVoiceDesign
            | Self::Qwen3Tts12Hz17BVoiceDesign8Bit
            | Self::Qwen3Tts12Hz17BVoiceDesignBf16 => SpeechModelCapabilities {
                supports_builtin_voices: false,
                built_in_voice_count: None,
                supports_reference_voice: false,
                reference_transcript_required: false,
                supports_voice_description: true,
                supports_streaming: true,
                supports_speed_control: false,
                supports_auto_long_form: true,
            },
            Self::Kokoro82M => SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(Self::KOKORO_BUILT_IN_VOICE_COUNT),
                supports_reference_voice: false,
                reference_transcript_required: false,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: true,
                supports_auto_long_form: false,
            },
            Self::Lfm25Audio15BGguf => SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(Self::LFM25_AUDIO_BUILT_IN_VOICE_COUNT),
                supports_reference_voice: false,
                reference_transcript_required: false,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: false,
                supports_auto_long_form: false,
            },
            Self::Lfm3Audio2BGguf | Self::Lfm3Audio2BQ8Gguf => SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(Self::LFM3_AUDIO_BUILT_IN_VOICE_COUNT),
                supports_reference_voice: false,
                reference_transcript_required: false,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: false,
                supports_auto_long_form: true,
            },
            Self::Voxtral4BTts2603 => SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(Self::VOXTRAL_TTS_BUILT_IN_VOICE_COUNT),
                supports_reference_voice: false,
                reference_transcript_required: false,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: false,
                supports_auto_long_form: false,
            },
            Self::VibeVoice15BTts => SpeechModelCapabilities {
                supports_builtin_voices: false,
                built_in_voice_count: None,
                supports_reference_voice: true,
                reference_transcript_required: true,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: false,
                supports_auto_long_form: true,
            },
            Self::FishAudioS2Pro => SpeechModelCapabilities {
                supports_builtin_voices: false,
                built_in_voice_count: None,
                supports_reference_voice: true,
                reference_transcript_required: true,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: false,
                supports_auto_long_form: true,
            },
            Self::Chatterbox500M | Self::ChatterboxMultilingual500M => SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(Self::CHATTERBOX_BUILT_IN_VOICE_COUNT),
                supports_reference_voice: true,
                reference_transcript_required: false,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: false,
                supports_auto_long_form: true,
            },
            // Built-in voices are named descriptions; a request's own
            // description replaces the voice's.
            // Named voices, or a clone of a reference clip with its exact
            // transcript (the reference's language may differ).
            Self::Maya2Native25B => SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(Self::MAYA2_BUILT_IN_VOICE_COUNT),
                supports_reference_voice: true,
                reference_transcript_required: true,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: false,
                supports_auto_long_form: true,
            },
            Self::Maya13B | Self::Maya13BGguf => SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(Self::MAYA1_BUILT_IN_VOICE_COUNT),
                supports_reference_voice: false,
                reference_transcript_required: false,
                supports_voice_description: true,
                supports_streaming: true,
                supports_speed_control: false,
                supports_auto_long_form: true,
            },
            _ => return None,
        };

        Some(capabilities)
    }

    /// Machine-readable chat controls that the selected runtime actually
    /// consumes. Keep reasoning effort narrower than generic thinking support:
    /// today only Qwen3.8 implements the three-level effort contract. The
    /// LFM2.5 Thinking checkpoint always reasons (the upstream template never
    /// gates thinking on a kwarg), so its thinking is not toggleable.
    pub fn chat_capabilities(&self) -> Option<ChatModelCapabilities> {
        let (default_thinking_enabled, reasoning_efforts, supports_preserve_thinking) = match self {
            Self::Qwen3827BFp8 => (
                true,
                vec![
                    ChatReasoningEffort::Xhigh,
                    ChatReasoningEffort::Medium,
                    ChatReasoningEffort::Low,
                ],
                true,
            ),
            Self::Qwen3508BGguf | Self::Qwen352BGguf => (false, Vec::new(), false),
            Self::Qwen354BGguf | Self::Qwen359BGguf => (true, Vec::new(), false),
            Self::Qwen36Moe35BA3BFp8 => (true, Vec::new(), false),
            Self::Lfm2512BThinkingGguf => (true, Vec::new(), false),
            // LFM3 reasons by default; `enable_thinking=false` closes an empty
            // `<think></think>` instead.
            Self::Lfm317BGguf | Self::Lfm317BQ8Gguf => (true, Vec::new(), true),
            _ => return None,
        };

        Some(ChatModelCapabilities {
            supports_thinking: true,
            thinking_toggleable: !matches!(self, Self::Lfm2512BThinkingGguf),
            default_thinking_enabled,
            default_reasoning_effort: reasoning_efforts.first().copied(),
            reasoning_efforts,
            supports_preserve_thinking,
        })
    }

    /// The variant that answers like this one without reasoning, for a model
    /// whose thinking can't be turned off. Server-side generations that must
    /// not think (chat titles) use it when it is loaded.
    pub fn non_thinking_sibling(&self) -> Option<ModelVariant> {
        match self {
            Self::Lfm2512BThinkingGguf => Some(Self::Lfm2512BInstructGguf),
            _ => None,
        }
    }

    /// Max output codec frames for this TTS variant, if known.
    pub fn tts_max_output_frames_hint(&self) -> Option<usize> {
        match self.family() {
            crate::catalog::ModelFamily::Qwen3Tts => Some(Self::QWEN3_TTS_MAX_OUTPUT_FRAMES),
            crate::catalog::ModelFamily::VoxtralTts => Some(Self::VOXTRAL_TTS_MAX_OUTPUT_FRAMES),
            crate::catalog::ModelFamily::VibeVoiceTts => {
                Some(Self::VIBEVOICE_TTS_MAX_OUTPUT_FRAMES)
            }
            crate::catalog::ModelFamily::FishS2Tts => Some(Self::FISH_S2_PRO_MAX_OUTPUT_FRAMES),
            crate::catalog::ModelFamily::ChatterboxTts => Some(Self::CHATTERBOX_MAX_OUTPUT_FRAMES),
            crate::catalog::ModelFamily::Maya1Tts => Some(Self::MAYA1_MAX_OUTPUT_FRAMES),
            crate::catalog::ModelFamily::Maya2Tts => Some(Self::MAYA2_MAX_OUTPUT_FRAMES),
            crate::catalog::ModelFamily::Lfm3Audio => Some(Self::LFM3_AUDIO_MAX_OUTPUT_FRAMES),
            _ => None,
        }
    }

    /// Codec frame-rate hint for this TTS variant, if known.
    pub fn tts_output_frame_rate_hz_hint(&self) -> Option<f32> {
        match self.family() {
            crate::catalog::ModelFamily::Qwen3Tts => Some(Self::QWEN3_TTS_FRAME_RATE_HZ),
            crate::catalog::ModelFamily::VoxtralTts => Some(Self::VOXTRAL_TTS_FRAME_RATE_HZ),
            crate::catalog::ModelFamily::VibeVoiceTts => Some(Self::VIBEVOICE_TTS_FRAME_RATE_HZ),
            crate::catalog::ModelFamily::FishS2Tts => Some(Self::FISH_S2_PRO_FRAME_RATE_HZ),
            crate::catalog::ModelFamily::ChatterboxTts => Some(Self::CHATTERBOX_FRAME_RATE_HZ),
            crate::catalog::ModelFamily::Maya1Tts => Some(Self::MAYA1_FRAME_RATE_HZ),
            crate::catalog::ModelFamily::Maya2Tts => Some(Self::MAYA2_FRAME_RATE_HZ),
            crate::catalog::ModelFamily::Lfm3Audio => Some(Self::LFM3_AUDIO_FRAME_RATE_HZ),
            _ => None,
        }
    }

    /// The sample rate this TTS variant's audio comes out at, for surfaces
    /// that announce it before the first chunk (realtime sessions).
    pub fn tts_output_sample_rate_hint(&self) -> Option<u32> {
        match self.family() {
            crate::catalog::ModelFamily::FishS2Tts => Some(44_100),
            crate::catalog::ModelFamily::Qwen3Tts
            | crate::catalog::ModelFamily::KokoroTts
            | crate::catalog::ModelFamily::VoxtralTts
            | crate::catalog::ModelFamily::VibeVoiceTts
            | crate::catalog::ModelFamily::ChatterboxTts
            | crate::catalog::ModelFamily::Maya1Tts
            | crate::catalog::ModelFamily::Maya2Tts
            | crate::catalog::ModelFamily::Lfm25Audio
            | crate::catalog::ModelFamily::Lfm3Audio => Some(24_000),
            _ => None,
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
        matches!(self, Self::ParakeetTdt06BV3)
    }

    /// Whether this is an MLX-exported Parakeet variant.
    pub fn is_parakeet_mlx(&self) -> bool {
        false
    }

    /// Whether this variant uses reduced-precision or quantized weights.
    pub fn is_quantized(&self) -> bool {
        matches!(
            self,
            Self::Qwen3Tts12Hz06BBase8Bit
                | Self::Qwen3Tts12Hz06BBaseBf16
                | Self::Qwen3Tts12Hz06BCustomVoice8Bit
                | Self::Qwen3Tts12Hz06BCustomVoiceBf16
                | Self::Qwen3Tts12Hz17BVoiceDesign8Bit
                | Self::Qwen3Tts12Hz17BVoiceDesignBf16
                | Self::Lfm2512BInstructGguf
                | Self::Lfm2512BThinkingGguf
                | Self::Lfm25Audio15BGguf
                | Self::Lfm317BGguf
                | Self::Lfm317BQ8Gguf
                | Self::Lfm3Audio2BGguf
                | Self::Lfm3Audio2BQ8Gguf
                | Self::Qwen306B4Bit
                | Self::Qwen317B4Bit
                | Self::Qwen3Asr06BGguf
                | Self::Qwen3Asr17BGguf
                | Self::GraniteSpeech502BGguf
                | Self::Maya13BGguf
                | Self::Qwen314BGguf
                | Self::Qwen3Moe30bA3bGguf
                | Self::Qwen3508BGguf
                | Self::Qwen352BGguf
                | Self::Qwen354BGguf
                | Self::Qwen359BGguf
                | Self::Qwen3827BFp8
                | Self::Qwen36Moe35BA3BFp8
        )
    }

    /// Whether this variant uses GGUF weights.
    pub fn is_gguf(&self) -> bool {
        matches!(
            self,
            Self::Qwen314BGguf
                | Self::Qwen3Moe30bA3bGguf
                | Self::Qwen3508BGguf
                | Self::Qwen352BGguf
                | Self::Qwen354BGguf
                | Self::Qwen359BGguf
                | Self::Lfm2512BInstructGguf
                | Self::Lfm2512BThinkingGguf
                | Self::Lfm25Audio15BGguf
                | Self::Lfm317BGguf
                | Self::Lfm317BQ8Gguf
                | Self::Lfm3Audio2BGguf
                | Self::Lfm3Audio2BQ8Gguf
                | Self::Qwen3Asr06BGguf
                | Self::Qwen3Asr17BGguf
                | Self::GraniteSpeech502BGguf
                | Self::Maya13BGguf
        )
    }

    /// Whether this is a Qwen3 chat GGUF variant (dense or sparse/MoE).
    pub fn is_qwen_chat_gguf(&self) -> bool {
        matches!(
            self,
            Self::Qwen314BGguf
                | Self::Qwen3Moe30bA3bGguf
        )
    }

    /// Whether this is the sparse-expert (MoE) Qwen3 chat GGUF variant.
    pub fn is_qwen3_moe_chat_gguf(&self) -> bool {
        matches!(self, Self::Qwen3Moe30bA3bGguf)
    }

    /// Whether this is a Qwen3 ASR GGUF variant.
    pub fn is_qwen_asr_gguf(&self) -> bool {
        matches!(self, Self::Qwen3Asr06BGguf | Self::Qwen3Asr17BGguf)
    }

    /// Whether this is Granite Speech 5.0's GGUF release (Q8_0 language
    /// model and F16 `mmproj`).
    pub fn is_granite_speech5_gguf(&self) -> bool {
        matches!(self, Self::GraniteSpeech502BGguf)
    }

    /// Whether this is Maya1's GGUF release (a Q8_0 decoder).
    pub fn is_maya1_gguf(&self) -> bool {
        matches!(self, Self::Maya13BGguf)
    }

    /// Whether this is a Qwen3.5 chat GGUF variant.
    pub fn is_qwen35_chat_gguf(&self) -> bool {
        matches!(
            self,
            Self::Qwen3508BGguf | Self::Qwen352BGguf | Self::Qwen354BGguf | Self::Qwen359BGguf
        )
    }

    /// Whether this is the revision-pinned native Qwen3.8 FP8 checkpoint.
    pub fn is_qwen38_fp8(&self) -> bool {
        matches!(self, Self::Qwen3827BFp8)
    }

    /// Whether this is the native qwen3_5_moe MoE FP8 checkpoint. The served
    /// release is Qwen3.6-35B-A3B-FP8 (architecture twin of the original
    /// Qwen3.5-35B-A3B-FP8 rollout); the name follows the architecture module.
    pub fn is_qwen35_moe_fp8(&self) -> bool {
        matches!(self, Self::Qwen36Moe35BA3BFp8)
    }

    /// Whether this is an LFM2.5 chat GGUF variant.
    pub fn is_lfm2_chat_gguf(&self) -> bool {
        matches!(
            self,
            Self::Lfm2512BInstructGguf | Self::Lfm2512BThinkingGguf
        )
    }

    /// Whether this is an LFM3 chat GGUF variant.
    pub fn is_lfm3_chat_gguf(&self) -> bool {
        matches!(self, Self::Lfm317BGguf | Self::Lfm317BQ8Gguf)
    }

    /// Whether this is an LFM3-Audio GGUF variant.
    pub fn is_lfm3_audio_gguf(&self) -> bool {
        matches!(self, Self::Lfm3Audio2BGguf | Self::Lfm3Audio2BQ8Gguf)
    }

    /// Whether this is an LFM2.5 Audio GGUF bundle variant.
    pub fn is_lfm25_audio_gguf(&self) -> bool {
        matches!(self, Self::Lfm25Audio15BGguf)
    }

    /// Whether this variant is currently enabled in the application catalog.
    pub fn is_enabled(&self) -> bool {
        match self {
            Self::Qwen306B
            | Self::Qwen306B4Bit
            | Self::Qwen317B
            | Self::Qwen317B4Bit
            | Self::Qwen314BGguf
            // DS10 groundwork posture: catalog-disabled until a
            // real MoE checkpoint is validated on every buildable lane.
            | Self::Qwen3Moe30bA3bGguf => false,
            Self::Qwen3Asr06BGguf
            | Self::Qwen3Asr17BGguf
            | Self::GraniteSpeech502BGguf
            | Self::Qwen3508BGguf
            | Self::Qwen352BGguf
            | Self::Qwen354BGguf
            | Self::Qwen359BGguf
            // Enabled by product decision (2026-09-30): visible and usable by
            // default. Runtime/performance certification against the real
            // checkpoint remains an exact-SHA hardware handoff gate.
            | Self::Qwen36Moe35BA3BFp8
            | Self::Qwen3827BFp8
            | Self::Lfm2512BInstructGguf
            | Self::Lfm2512BThinkingGguf
            | Self::Lfm25Audio15BGguf
            | Self::Lfm317BGguf
            | Self::Lfm317BQ8Gguf
            | Self::Lfm3Audio2BGguf
            | Self::Lfm3Audio2BQ8Gguf
            | Self::Kokoro82M => true,
            Self::Gemma34BIt => false,
            Self::VoxtralMini4BRealtime2602 => true,
            Self::Voxtral4BTts2603 => true,
            Self::ParakeetTdt06BV3 => true,
            Self::WhisperLargeV3Turbo => true,
            Self::DiarStreamingSortformer4SpkV21 => true,
            // Enabled by product decision (2026-09-30) after real-checkpoint
            // CPU evidence: 8-channel load + end-to-end diarization of the
            // conformance clip through the RoPE encoder path (commit
            // 2e35f632). Visible in the catalog and downloadable.
            Self::Nemotron3Diarization => true,
            Self::Qwen3ForcedAligner06B => true,
            Self::FishAudioS2Pro => true,
            Self::Chatterbox500M | Self::ChatterboxMultilingual500M => true,
            Self::Maya13B | Self::Maya13BGguf => true,
            Self::Maya2Native25B => true,
            _ => !self.is_quantized(),
        }
    }

    /// Get all available variants
    pub fn all() -> &'static [ModelVariant] {
        &[
            Self::Qwen3Tts12Hz06BBase,
            Self::Qwen3Tts12Hz06BBase8Bit,
            Self::Qwen3Tts12Hz06BBaseBf16,
            Self::Qwen3Tts12Hz06BCustomVoice,
            Self::Qwen3Tts12Hz06BCustomVoice8Bit,
            Self::Qwen3Tts12Hz06BCustomVoiceBf16,
            Self::Qwen3Tts12Hz17BBase,
            Self::Qwen3Tts12Hz17BCustomVoice,
            Self::Qwen3Tts12Hz17BVoiceDesign,
            Self::Qwen3Tts12Hz17BVoiceDesign8Bit,
            Self::Qwen3Tts12Hz17BVoiceDesignBf16,
            Self::Qwen3TtsTokenizer12Hz,
            Self::Lfm2512BInstructGguf,
            Self::Lfm2512BThinkingGguf,
            Self::Lfm25Audio15BGguf,
            Self::Lfm317BGguf,
            Self::Lfm317BQ8Gguf,
            Self::Lfm3Audio2BGguf,
            Self::Lfm3Audio2BQ8Gguf,
            Self::Kokoro82M,
            Self::ParakeetTdt06BV3,
            Self::WhisperLargeV3Turbo,
            Self::Qwen3Asr06BGguf,
            Self::Qwen3Asr17BGguf,
            Self::VibeVoiceAsr,
            Self::Nemotron35AsrStreaming06B,
            Self::GraniteSpeech412B,
            Self::GraniteSpeech412BPlus,
            Self::GraniteSpeech502B,
            Self::GraniteSpeech502BGguf,
            Self::DiarStreamingSortformer4SpkV21,
            Self::Nemotron3Diarization,
            Self::Qwen306B,
            Self::Qwen306B4Bit,
            Self::Qwen317B,
            Self::Qwen317B4Bit,
            Self::Qwen314BGguf,
            Self::Qwen3Moe30bA3bGguf,
            Self::Qwen3508BGguf,
            Self::Qwen352BGguf,
            Self::Qwen354BGguf,
            Self::Qwen359BGguf,
            Self::Qwen3827BFp8,
            Self::Qwen36Moe35BA3BFp8,
            Self::Gemma34BIt,
            Self::Qwen3ForcedAligner06B,
            Self::VoxtralMini4BRealtime2602,
            Self::Voxtral4BTts2603,
            Self::VibeVoice15BTts,
            Self::FishAudioS2Pro,
            Self::Chatterbox500M,
            Self::ChatterboxMultilingual500M,
            Self::Maya13B,
            Self::Maya13BGguf,
            Self::Maya2Native25B,
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

/// A model-specific speech control a request may set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechControl {
    /// Emotion intensity (Chatterbox).
    Exaggeration,
    /// Classifier-free guidance weight; 0 turns guidance off (Chatterbox).
    CfgWeight,
    /// Min-p sampling threshold (Chatterbox).
    MinP,
}

impl SpeechControl {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Exaggeration => "exaggeration",
            Self::CfgWeight => "cfg_weight",
            Self::MinP => "min_p",
        }
    }
}

/// A control's accepted range and the model's value when it is unset.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpeechControlRange {
    pub control: SpeechControl,
    pub min: f32,
    pub max: f32,
    pub default: f32,
}

const CHATTERBOX_SPEECH_CONTROLS: &[SpeechControlRange] = &[
    SpeechControlRange {
        control: SpeechControl::Exaggeration,
        min: 0.0,
        max: 2.0,
        default: 0.5,
    },
    SpeechControlRange {
        control: SpeechControl::CfgWeight,
        min: 0.0,
        max: 1.0,
        default: 0.5,
    },
    SpeechControlRange {
        control: SpeechControl::MinP,
        min: 0.0,
        max: 1.0,
        default: 0.05,
    },
];

/// Maya1's inline tags, each a single token of its tokenizer
/// (156,940..=156,959, in this order): the text may place them where the
/// voice should laugh, sigh, whisper and so on.
pub const MAYA1_INLINE_TAGS: &[&str] = &[
    "<angry>",
    "<appalled>",
    "<chuckle>",
    "<cry>",
    "<curious>",
    "<disappointed>",
    "<excited>",
    "<exhale>",
    "<gasp>",
    "<giggle>",
    "<gulp>",
    "<laugh>",
    "<laugh_harder>",
    "<mischievous>",
    "<sarcastic>",
    "<scream>",
    "<sigh>",
    "<sing>",
    "<snort>",
    "<whisper>",
];

impl ModelVariant {
    /// The model-specific controls a speech request may set; a control not
    /// listed is refused rather than ignored.
    pub fn speech_controls(&self) -> &'static [SpeechControlRange] {
        match self.family() {
            crate::catalog::ModelFamily::ChatterboxTts => CHATTERBOX_SPEECH_CONTROLS,
            _ => &[],
        }
    }

    /// The inline tags a speech request's text may carry, as they are
    /// written: Maya1's `<tag>`s (a tag this list lacks is refused, see
    /// [`Self::refuses_unknown_inline_tags`]) and the `[tag]`s Maya 2 Native
    /// performs best effort (any lowercase `[tag]` passes through; these are
    /// the ones it names). Text for any other model has none.
    pub fn speech_inline_tags(&self) -> &'static [&'static str] {
        match self.family() {
            crate::catalog::ModelFamily::Maya1Tts => MAYA1_INLINE_TAGS,
            crate::catalog::ModelFamily::Maya2Tts => crate::catalog::MAYA2_SUGGESTED_TAGS,
            _ => &[],
        }
    }

    /// Whether a speech request naming an inline tag outside
    /// [`Self::speech_inline_tags`] is refused (Maya1, whose tokenizer would
    /// read an unknown tag aloud).
    pub fn refuses_unknown_inline_tags(&self) -> bool {
        self.family() == crate::catalog::ModelFamily::Maya1Tts
    }
}

/// Machine-readable speech feature contract for voice workflows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpeechModelCapabilities {
    pub supports_builtin_voices: bool,
    pub built_in_voice_count: Option<usize>,
    pub supports_reference_voice: bool,
    /// Whether a reference voice needs its transcript. Every reference
    /// family but Chatterbox conditions on the reference's words; Chatterbox
    /// takes the audio alone.
    pub reference_transcript_required: bool,
    pub supports_voice_description: bool,
    pub supports_streaming: bool,
    pub supports_speed_control: bool,
    pub supports_auto_long_form: bool,
}

impl SpeechModelCapabilities {
    /// The `speed` a model with speed control honours, inclusive. A request
    /// outside it (or any speed but 1.0 for a model without speed control)
    /// is refused rather than silently clamped or dropped (TPIPE-10).
    pub const SPEED_RANGE: (f32, f32) = (0.5, 2.0);

    /// The longest reference audio a voice-clone request or saved voice
    /// carries, in seconds; longer audio is refused with 400 at upload and at
    /// synthesis alike (TPIPE-23).
    pub const MAX_REFERENCE_AUDIO_SECS: u64 = 30;

    /// Whether this model honours `speed`.
    pub fn honours_speed(&self, speed: f32) -> bool {
        speed == 1.0
            || (self.supports_speed_control
                && (Self::SPEED_RANGE.0..=Self::SPEED_RANGE.1).contains(&speed))
    }
}

/// Machine-readable controls exposed by chat-capable model runtimes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatModelCapabilities {
    pub supports_thinking: bool,
    /// Whether `enable_thinking: false` turns thinking off. False for a model
    /// that always reasons; its non-thinking sibling, when the catalog names
    /// one, is `ModelVariant::non_thinking_sibling`.
    pub thinking_toggleable: bool,
    pub default_thinking_enabled: bool,
    pub reasoning_efforts: Vec<ChatReasoningEffort>,
    pub default_reasoning_effort: Option<ChatReasoningEffort>,
    pub supports_preserve_thinking: bool,
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
    #[serde(default)]
    pub chat_capabilities: Option<ChatModelCapabilities>,
    pub speech_capabilities: Option<SpeechModelCapabilities>,
    #[serde(skip_deserializing)]
    pub cuda_support: CudaSupportInfo,
    #[serde(skip_deserializing)]
    pub cuda_quantization: CudaQuantizationInfo,
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
            chat_capabilities: variant.chat_capabilities(),
            speech_capabilities: variant.speech_capabilities(),
            cuda_support: variant.cuda_support(),
            cuda_quantization: variant.cuda_quantization(),
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
    use crate::catalog::ModelTask;

    use super::{ChatReasoningEffort, ModelVariant, SpeechModelCapabilities};

    /// TPIPE-22: the announced realtime rate is each family's real output
    /// rate (Fish's codec runs at 44.1 kHz; izwi-core checks the hint
    /// against the codec's own configuration).
    #[test]
    fn tts_output_sample_rate_hints_match_the_model_codecs() {
        assert_eq!(
            ModelVariant::FishAudioS2Pro.tts_output_sample_rate_hint(),
            Some(44_100)
        );
        assert_eq!(
            ModelVariant::Lfm25Audio15BGguf.tts_output_sample_rate_hint(),
            Some(24_000)
        );
        assert_eq!(
            ModelVariant::Kokoro82M.tts_output_sample_rate_hint(),
            Some(24_000)
        );
        assert_eq!(
            ModelVariant::WhisperLargeV3Turbo.tts_output_sample_rate_hint(),
            None
        );
    }

    #[test]
    fn chat_capabilities_expose_reasoning_effort_only_for_qwen38() {
        let qwen38 = ModelVariant::Qwen3827BFp8
            .chat_capabilities()
            .expect("Qwen3.8 chat capabilities");
        assert!(qwen38.supports_thinking);
        assert!(qwen38.default_thinking_enabled);
        assert_eq!(
            qwen38.reasoning_efforts,
            vec![
                ChatReasoningEffort::Xhigh,
                ChatReasoningEffort::Medium,
                ChatReasoningEffort::Low,
            ]
        );
        assert_eq!(
            qwen38.default_reasoning_effort,
            Some(ChatReasoningEffort::Xhigh)
        );
        assert!(qwen38.supports_preserve_thinking);

        let qwen35 = ModelVariant::Qwen354BGguf
            .chat_capabilities()
            .expect("Qwen3.5 chat capabilities");
        assert!(qwen35.supports_thinking);
        assert!(qwen35.default_thinking_enabled);
        assert!(qwen35.reasoning_efforts.is_empty());
        assert_eq!(qwen35.default_reasoning_effort, None);

        assert_eq!(ModelVariant::Lfm2512BInstructGguf.chat_capabilities(), None);
        let lfm25_thinking = ModelVariant::Lfm2512BThinkingGguf
            .chat_capabilities()
            .expect("LFM2.5 Thinking chat capabilities");
        assert!(lfm25_thinking.supports_thinking);
        assert!(lfm25_thinking.default_thinking_enabled);
        assert!(lfm25_thinking.reasoning_efforts.is_empty());
        assert_eq!(lfm25_thinking.default_reasoning_effort, None);
        assert!(!lfm25_thinking.supports_preserve_thinking);
        assert!(!lfm25_thinking.thinking_toggleable);
        assert_eq!(
            ModelVariant::Lfm2512BThinkingGguf.non_thinking_sibling(),
            Some(ModelVariant::Lfm2512BInstructGguf)
        );
        assert!(qwen35.thinking_toggleable);
        assert!(qwen38.thinking_toggleable);
        assert_eq!(ModelVariant::Qwen354BGguf.non_thinking_sibling(), None);
        assert_eq!(ModelVariant::Lfm2512BInstructGguf.chat_capabilities(), None);
    }

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
        let variant = ModelVariant::Kokoro82M;
        assert_eq!(variant.tts_max_output_frames_hint(), None);
        assert_eq!(variant.tts_output_frame_rate_hz_hint(), None);
        assert_eq!(variant.tts_max_output_seconds_hint(), None);
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
            assert!(
                variant.is_enabled(),
                "{} should be enabled",
                variant.dir_name()
            );
            assert!(variant.is_chat(), "{} should be chat", variant.dir_name());
            assert!(
                variant.is_quantized(),
                "{} should be quantized",
                variant.dir_name()
            );
            assert!(variant.is_gguf(), "{} should be gguf", variant.dir_name());
            assert!(
                variant.is_qwen35_chat_gguf(),
                "{} should be a qwen3.5 gguf",
                variant.dir_name()
            );
        }
    }

    #[test]
    fn nemotron3_diarization_catalog_contract_is_enabled_eight_speakers() {
        let variant = ModelVariant::Nemotron3Diarization;

        // Enabled by product decision (2026-09-30) after real-checkpoint CPU
        // evidence; the SortformerDiarization family shares v2.1's runtime
        // surface with a per-checkpoint 8-speaker envelope.
        assert!(variant.is_enabled());
        assert!(variant.is_diarization());
        assert!(variant.supports_diarization_records());
        assert!(!variant.is_quantized());
        assert_eq!(
            variant.family(),
            crate::catalog::ModelFamily::SortformerDiarization
        );
        assert_eq!(variant.primary_task(), ModelTask::Diarization);
        assert_eq!(variant.repo_id(), "nvidia/Nemotron-3-Diarization");
        assert_eq!(variant.dir_name(), "Nemotron-3-Diarization");
        assert_eq!(variant.display_name(), "Nemotron 3 Diarization");
        assert_eq!(variant.license_label(), Some("OpenMDW-1.1"));
        assert_eq!(variant.estimated_size(), 198_676_480);
        assert_eq!(variant.memory_required_gb(), 0.4);
    }

    #[test]
    fn qwen36_moe_fp8_catalog_contract_is_enabled_moe_family() {
        let variant = ModelVariant::Qwen36Moe35BA3BFp8;

        // Enabled by product decision (2026-09-30); runtime certification
        // against the real checkpoint stays an exact-SHA hardware gate.
        assert!(variant.is_enabled());
        assert!(variant.is_chat());
        assert!(variant.is_quantized());
        assert!(!variant.is_gguf());
        assert!(variant.is_qwen35_moe_fp8());
        assert!(!variant.is_qwen38_fp8());
        assert_eq!(variant.family(), crate::catalog::ModelFamily::Qwen35MoeChat);
        assert_eq!(variant.primary_task(), ModelTask::Chat);
        assert_eq!(variant.repo_id(), "Qwen/Qwen3.6-35B-A3B-FP8");
        assert_eq!(variant.dir_name(), "Qwen3.6-35B-A3B-FP8");
        assert_eq!(variant.display_name(), "Qwen3.6 35B-A3B FP8");
        assert_eq!(variant.license_label(), Some("Apache-2.0"));
        assert_eq!(
            variant.artifact_revision(),
            Some(ModelVariant::QWEN36_MOE_35B_A3B_FP8_ARTIFACT_REVISION)
        );
        let caps = variant
            .chat_capabilities()
            .expect("qwen3.6 moe chat capabilities");
        assert!(caps.supports_thinking);
        assert!(caps.default_thinking_enabled);
        assert!(caps.reasoning_efforts.is_empty());
        assert_eq!(caps.default_reasoning_effort, None);
        assert!(!caps.supports_preserve_thinking);
    }

    #[test]
    fn qwen38_fp8_catalog_contract_has_distinct_chat_family() {
        let variant = ModelVariant::Qwen3827BFp8;

        assert!(variant.is_enabled());
        assert!(variant.is_chat());
        assert!(variant.is_quantized());
        assert!(!variant.is_gguf());
        assert!(variant.is_qwen38_fp8());
        assert!(!variant.is_qwen35_chat_gguf());
        assert_eq!(variant.family(), crate::catalog::ModelFamily::Qwen38Chat);
        assert_eq!(variant.repo_id(), "Qwen/Qwen3.8-27B-FP8");
        assert_eq!(variant.dir_name(), "Qwen3.8-27B-FP8");
        assert_eq!(variant.display_name(), "Qwen3.8 27B FP8");
        assert_eq!(variant.license_label(), Some("Apache-2.0"));
        assert_eq!(
            variant.artifact_revision(),
            Some(ModelVariant::QWEN38_27B_FP8_ARTIFACT_REVISION)
        );
    }

    #[test]
    fn qwen38_fp8_serde_accepts_canonical_and_repository_identifiers() {
        for identifier in [
            "Qwen3.8-27B-FP8",
            "Qwen/Qwen3.8-27B-FP8",
            "qwen3.8-27b-fp8",
            "Qwen3.8-27B",
        ] {
            let encoded = serde_json::to_string(identifier).expect("serialize identifier");
            let parsed: ModelVariant =
                serde_json::from_str(&encoded).expect("deserialize Qwen3.8 alias");
            assert_eq!(parsed, ModelVariant::Qwen3827BFp8, "alias {identifier}");
        }

        assert_eq!(
            serde_json::to_string(&ModelVariant::Qwen3827BFp8).expect("serialize variant"),
            "\"Qwen3.8-27B-FP8\""
        );
    }

    #[test]
    fn lfm25_audio_variant_is_enabled() {
        let variant = ModelVariant::Lfm25Audio15BGguf;
        assert!(
            variant.is_enabled(),
            "{} should be enabled",
            variant.dir_name()
        );
        assert!(
            variant.is_audio_chat(),
            "{} should be audio-chat",
            variant.dir_name()
        );
        assert!(variant.is_gguf(), "{} should be gguf", variant.dir_name());
    }

    #[test]
    fn voxtral_enabled_contract_is_asr_only_until_realtime_lands() {
        let variant = ModelVariant::VoxtralMini4BRealtime2602;
        assert!(variant.is_voxtral());
        assert_eq!(variant.primary_task(), ModelTask::Asr);
        assert!(variant.is_enabled());
        assert!(!variant.is_asr());
        assert!(!variant.is_audio_chat());
        assert_eq!(variant.speech_capabilities(), None);
    }

    #[test]
    fn voxtral_tts_contract_exposes_builtin_voice_tts_only() {
        let variant = ModelVariant::Voxtral4BTts2603;
        assert!(variant.is_tts());
        assert_eq!(variant.primary_task(), ModelTask::Tts);
        assert!(variant.is_enabled());
        assert!(!variant.is_voxtral());
        assert_eq!(variant.repo_id(), "mistralai/Voxtral-4B-TTS-2603");
        assert_eq!(variant.license_label(), Some("CC BY-NC 4.0"));
        assert_eq!(
            variant.tts_output_frame_rate_hz_hint(),
            Some(ModelVariant::VOXTRAL_TTS_FRAME_RATE_HZ)
        );
        assert_eq!(
            variant.tts_max_output_frames_hint(),
            Some(ModelVariant::VOXTRAL_TTS_MAX_OUTPUT_FRAMES)
        );
        assert_eq!(
            variant.speech_capabilities(),
            Some(SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(ModelVariant::VOXTRAL_TTS_BUILT_IN_VOICE_COUNT),
                supports_reference_voice: false,
                reference_transcript_required: false,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: false,
                supports_auto_long_form: false,
            })
        );
    }

    #[test]
    fn vibevoice_variants_expose_long_form_contracts() {
        let asr = ModelVariant::VibeVoiceAsr;
        assert!(asr.is_asr());
        assert!(!asr.is_tts());
        assert_eq!(asr.primary_task(), ModelTask::Asr);
        assert_eq!(asr.repo_id(), "microsoft/VibeVoice-ASR");
        assert_eq!(asr.license_label(), Some("MIT"));

        let tts = ModelVariant::VibeVoice15BTts;
        assert!(tts.is_tts());
        assert_eq!(tts.primary_task(), ModelTask::Tts);
        assert_eq!(tts.repo_id(), "microsoft/VibeVoice-1.5B");
        assert_eq!(tts.license_label(), Some("MIT"));
        assert_eq!(
            tts.tts_max_output_frames_hint(),
            Some(ModelVariant::VIBEVOICE_TTS_MAX_OUTPUT_FRAMES)
        );
        assert_eq!(
            tts.tts_output_frame_rate_hz_hint(),
            Some(ModelVariant::VIBEVOICE_TTS_FRAME_RATE_HZ)
        );
        assert!(tts
            .tts_max_output_seconds_hint()
            .is_some_and(|seconds| seconds >= 5_400.0));
        assert_eq!(
            tts.speech_capabilities(),
            Some(SpeechModelCapabilities {
                supports_builtin_voices: false,
                built_in_voice_count: None,
                supports_reference_voice: true,
                reference_transcript_required: true,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: false,
                supports_auto_long_form: true,
            })
        );
    }

    #[test]
    fn fish_s2_pro_exposes_reference_voice_tts_contract() {
        let variant = ModelVariant::FishAudioS2Pro;
        assert!(variant.is_tts());
        assert!(variant.is_enabled());
        assert_eq!(variant.primary_task(), ModelTask::Tts);
        assert_eq!(variant.repo_id(), "fishaudio/s2-pro");
        assert_eq!(variant.dir_name(), "FishAudio-S2-Pro");
        assert_eq!(variant.license_label(), Some("Fish Audio Research License"));
        assert_eq!(variant.memory_required_gb(), 24.0);
        assert_eq!(
            variant.tts_max_output_frames_hint(),
            Some(ModelVariant::FISH_S2_PRO_MAX_OUTPUT_FRAMES)
        );
        assert_eq!(
            variant.tts_output_frame_rate_hz_hint(),
            Some(ModelVariant::FISH_S2_PRO_FRAME_RATE_HZ)
        );
        assert_eq!(
            variant.speech_capabilities(),
            Some(SpeechModelCapabilities {
                supports_builtin_voices: false,
                built_in_voice_count: None,
                supports_reference_voice: true,
                reference_transcript_required: true,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: false,
                supports_auto_long_form: true,
            })
        );
    }

    #[test]
    fn chatterbox_exposes_default_voice_audio_only_clones_and_streaming() {
        let variant = ModelVariant::Chatterbox500M;
        assert!(variant.is_tts());
        assert!(variant.is_enabled());
        assert_eq!(variant.primary_task(), ModelTask::Tts);
        assert_eq!(variant.repo_id(), "ResembleAI/chatterbox");
        assert_eq!(variant.dir_name(), "Chatterbox-500M");
        assert_eq!(
            variant.artifact_revision(),
            Some(ModelVariant::CHATTERBOX_ARTIFACT_REVISION)
        );
        assert_eq!(variant.license_label(), Some("MIT"));
        // A stream runs on its TTS route's retained graph.
        assert!(variant.streams_on_its_tts_route());
        assert_eq!(variant.tts_output_sample_rate_hint(), Some(24_000));
        assert_eq!(variant.tts_max_output_seconds_hint(), Some(40.0));
        let controls = variant
            .speech_controls()
            .iter()
            .map(|range| (range.control.as_str(), range.default))
            .collect::<Vec<_>>();
        assert_eq!(
            controls,
            vec![("exaggeration", 0.5), ("cfg_weight", 0.5), ("min_p", 0.05)]
        );
        assert!(ModelVariant::Kokoro82M.speech_controls().is_empty());
        // Every other reference family conditions on the reference's words.
        for other in [
            ModelVariant::FishAudioS2Pro,
            ModelVariant::VibeVoice15BTts,
            ModelVariant::Qwen3Tts12Hz06BBase,
        ] {
            let capabilities = other.speech_capabilities().unwrap();
            assert!(capabilities.supports_reference_voice, "{other}");
            assert!(capabilities.reference_transcript_required, "{other}");
        }
        assert_eq!(
            variant.speech_capabilities(),
            Some(SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(1),
                supports_reference_voice: true,
                reference_transcript_required: false,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: false,
                supports_auto_long_form: true,
            })
        );
    }

    #[test]
    fn nemotron_asr_catalog_contract_matches_hf_artifact() {
        let variant = ModelVariant::Nemotron35AsrStreaming06B;
        assert!(variant.is_asr());
        assert_eq!(variant.primary_task(), ModelTask::Asr);
        assert_eq!(variant.repo_id(), "nvidia/nemotron-3.5-asr-streaming-0.6b");
        assert_eq!(variant.dir_name(), "Nemotron-3.5-ASR-Streaming-0.6B");
        assert_eq!(variant.license_label(), Some("OpenMDW-1.1"));
        assert!(!variant.is_quantized());
        assert_eq!(variant.memory_required_gb(), 6.0);
    }

    #[test]
    fn granite_speech_base_variant_is_enabled_punctuating_asr_metadata() {
        let variant = ModelVariant::GraniteSpeech412B;
        assert!(variant.is_asr());
        assert!(variant.is_enabled());
        assert_eq!(
            variant.family(),
            crate::catalog::ModelFamily::GraniteSpeechAsr
        );
        assert_eq!(variant.primary_task(), ModelTask::Asr);
        assert_eq!(variant.repo_id(), "ibm-granite/granite-speech-4.1-2b");
        assert_eq!(variant.dir_name(), "Granite-Speech-4.1-2B");
        assert_eq!(variant.estimated_size(), 4_626_414_392);
        assert_eq!(
            variant.artifact_revision(),
            Some("de575db64086f84fdc79da4932d1076e965bc546")
        );
        assert_eq!(
            variant.memory_required_gb(),
            ModelVariant::GraniteSpeech412BPlus.memory_required_gb()
        );
        assert_eq!(variant.license_label(), Some("Apache-2.0"));
        // Speaker turns are a plus-only capability.
        assert!(!variant.supports_speaker_attributed_asr());
        assert!(ModelVariant::GraniteSpeech412BPlus.supports_speaker_attributed_asr());
        assert!(!variant.is_quantized());
        assert!(!variant.is_gguf());
    }

    #[test]
    fn granite_speech_5_is_an_enabled_asr_variant_of_its_own_family() {
        let variant = ModelVariant::GraniteSpeech502B;
        assert!(variant.is_asr());
        assert!(variant.is_enabled());
        assert_eq!(
            variant.family(),
            crate::catalog::ModelFamily::GraniteSpeech5Asr
        );
        assert_ne!(
            variant.family(),
            ModelVariant::GraniteSpeech412B.family(),
            "5.0 is an architecture of its own, not a 4.1 checkpoint"
        );
        assert_eq!(variant.primary_task(), ModelTask::Asr);
        assert_eq!(variant.repo_id(), "ibm-granite/granite-speech-5.0-2b");
        assert_eq!(variant.dir_name(), "Granite-Speech-5.0-2B");
        assert_eq!(variant.display_name(), "Granite Speech 5.0 2B");
        assert_eq!(variant.estimated_size(), 4_626_414_392);
        assert_eq!(
            variant.artifact_revision(),
            Some("0555e70346a8992da6b5df98ae26ec50c43958b8")
        );
        assert_eq!(variant.license_label(), Some("Apache-2.0"));
        assert!(variant.supports_asr_translation());
        assert!(variant.supports_asr_keyword_biasing());
        assert!(!variant.supports_speaker_attributed_asr());
        assert!(!variant.is_quantized());
        assert!(!variant.is_gguf());
        assert!(ModelVariant::all().contains(&variant));
        for other in [
            ModelVariant::GraniteSpeech412B,
            ModelVariant::GraniteSpeech412BPlus,
            ModelVariant::WhisperLargeV3Turbo,
        ] {
            assert!(!other.supports_asr_translation(), "{other}");
            assert!(!other.supports_asr_keyword_biasing(), "{other}");
        }
    }

    #[test]
    fn granite_speech_variant_is_enabled_asr_metadata() {
        let variant = ModelVariant::GraniteSpeech412BPlus;
        assert!(variant.is_asr());
        assert!(variant.is_enabled());
        assert_eq!(variant.primary_task(), ModelTask::Asr);
        assert_eq!(variant.repo_id(), "ibm-granite/granite-speech-4.1-2b-plus");
        assert_eq!(variant.dir_name(), "Granite-Speech-4.1-2B-Plus");
        assert_eq!(variant.estimated_size(), 4_223_757_112);
        // IBM's v2610 weights; the loader reads the earlier layout too.
        assert_eq!(
            variant.artifact_revision(),
            Some("e69efe37f97b6283c61f5a8dadda2b3eb67765d6")
        );
        assert_eq!(variant.license_label(), Some("Apache-2.0"));
        assert!(!variant.is_quantized());
        assert!(!variant.is_gguf());
    }

    #[test]
    fn qwen_base_capabilities_match_saved_voice_workflow() {
        assert_eq!(
            ModelVariant::Qwen3Tts12Hz17BBase.speech_capabilities(),
            Some(SpeechModelCapabilities {
                supports_builtin_voices: false,
                built_in_voice_count: None,
                supports_reference_voice: true,
                reference_transcript_required: true,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: false,
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
                reference_transcript_required: false,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: false,
                supports_auto_long_form: true,
            })
        );
        assert_eq!(
            ModelVariant::Qwen3Tts12Hz17BCustomVoice.speech_capabilities(),
            Some(SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(ModelVariant::QWEN_CUSTOMVOICE_BUILT_IN_VOICE_COUNT),
                supports_reference_voice: false,
                reference_transcript_required: false,
                supports_voice_description: true,
                supports_streaming: true,
                supports_speed_control: false,
                supports_auto_long_form: true,
            })
        );
    }

    #[test]
    fn kokoro_capabilities_reflect_runtime_constraints() {
        assert_eq!(
            ModelVariant::Kokoro82M.speech_capabilities(),
            Some(SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(ModelVariant::KOKORO_BUILT_IN_VOICE_COUNT),
                supports_reference_voice: false,
                reference_transcript_required: false,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: true,
                supports_auto_long_form: false,
            })
        );
    }

    #[test]
    fn lfm25_audio_capabilities_surface_default_tts_voice() {
        assert_eq!(
            ModelVariant::Lfm25Audio15BGguf.speech_capabilities(),
            Some(SpeechModelCapabilities {
                supports_builtin_voices: true,
                built_in_voice_count: Some(ModelVariant::LFM25_AUDIO_BUILT_IN_VOICE_COUNT),
                supports_reference_voice: false,
                reference_transcript_required: false,
                supports_voice_description: false,
                supports_streaming: true,
                supports_speed_control: false,
                supports_auto_long_form: false,
            })
        );
    }

    #[test]
    fn non_speech_variants_do_not_expose_speech_capabilities() {
        assert_eq!(
            ModelVariant::WhisperLargeV3Turbo.speech_capabilities(),
            None
        );
        assert_eq!(ModelVariant::Qwen359BGguf.speech_capabilities(), None);
    }
}
