import type { ModelInfo } from "@/api";
import { withQwen3Prefix } from "@/utils/modelDisplay";

export const PROVIDER_ORDER = [
  "Qwen",
  "OpenAI",
  "Microsoft",
  "IBM",
  "Liquid AI",
  "Google",
  "NVIDIA",
  "Mistral AI",
  "hexgrad",
  "Fish Audio",
  "Resemble AI",
  "Maya Research",
  "Other",
] as const;

/**
 * Major.minor version carried by a Qwen variant id ("Qwen3.5-4B" → [3, 5],
 * "Qwen3-TTS-12Hz-..." → [3, 0]), or null for non-versioned ids. Drives both
 * Qwen provider labeling and version-number ordering so future Qwen3.x
 * releases classify correctly without touching the prefix lists again.
 */
export function getQwenVersionRank(variant: string): [number, number] | null {
  const match = variant.match(/^Qwen(\d+)(?:\.(\d+))?-/);
  if (!match) return null;
  return [Number(match[1]), match[2] ? Number(match[2]) : 0];
}

/**
 * Version-number comparison for a pair of Qwen variants. Returns 0 when
 * either id carries no Qwen version so callers fall through to their own
 * (size-based) ordering for mixed or non-Qwen sections.
 */
export function compareQwenVersionRank(left: string, right: string): number {
  const leftRank = getQwenVersionRank(left);
  const rightRank = getQwenVersionRank(right);
  if (!leftRank || !rightRank) return 0;
  return leftRank[0] - rightRank[0] || leftRank[1] - rightRank[1];
}

export function getModelProviderLabel(variant: string): string {
  if (/^Qwen\d+(\.\d+)?-/.test(variant)) {
    return "Qwen";
  }
  if (variant.startsWith("Whisper-")) return "OpenAI";
  if (
    variant.startsWith("VibeVoice-") ||
    variant.startsWith("microsoft/VibeVoice-")
  ) {
    return "Microsoft";
  }
  if (
    variant.startsWith("Granite-Speech-") ||
    variant.startsWith("ibm-granite/granite-speech-")
  ) {
    return "IBM";
  }
  if (variant.startsWith("LFM2")) return "Liquid AI";
  if (variant.startsWith("Gemma-")) return "Google";
  if (
    variant.startsWith("Parakeet-") ||
    variant.startsWith("Nemotron-") ||
    variant.startsWith("nvidia/nemotron-") ||
    variant.startsWith("diar_streaming_sortformer")
  ) {
    return "NVIDIA";
  }
  if (variant.startsWith("Voxtral-")) return "Mistral AI";
  if (variant.startsWith("Kokoro-")) return "hexgrad";
  if (variant.startsWith("FishAudio-") || variant.startsWith("fishaudio/")) {
    return "Fish Audio";
  }
  if (
    variant.startsWith("Chatterbox-") ||
    variant.startsWith("ResembleAI/chatterbox")
  ) {
    return "Resemble AI";
  }
  if (variant.startsWith("Maya1-") || variant.startsWith("maya-research/")) {
    return "Maya Research";
  }
  return "Other";
}

export interface ModelDetail {
  shortName: string;
  fullName: string;
  description: string;
  category: "tts" | "asr" | "chat";
  categories?: Array<"tts" | "asr" | "chat">;
  capabilities: string[];
  size: string;
}

export const MODEL_DETAILS: Record<string, ModelDetail> = {
  // TTS 0.6B Base models
  "Qwen3-TTS-12Hz-0.6B-Base": {
    shortName: "0.6B Base",
    fullName: "Qwen3-TTS 12Hz 0.6B Base",
    description: "Voice cloning with reference audio samples",
    category: "tts",
    capabilities: ["Voice Cloning"],
    size: "2.3 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-Base-8bit": {
    shortName: "0.6B Base 8-bit",
    fullName: "Qwen3-TTS 12Hz 0.6B Base (MLX 8-bit)",
    description:
      "8-bit MLX weights for better quality while staying memory friendly",
    category: "tts",
    capabilities: ["Voice Cloning", "8-bit"],
    size: "1.9 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-Base-bf16": {
    shortName: "0.6B Base BF16",
    fullName: "Qwen3-TTS 12Hz 0.6B Base (MLX bf16)",
    description: "BF16 MLX weights for highest fidelity base voices",
    category: "tts",
    capabilities: ["Voice Cloning", "BF16"],
    size: "2.3 GB",
  },
  // TTS 0.6B CustomVoice models
  "Qwen3-TTS-12Hz-0.6B-CustomVoice": {
    shortName: "0.6B CustomVoice",
    fullName: "Qwen3-TTS 12Hz 0.6B CustomVoice",
    description: "Pre-trained with 9 built-in voice profiles",
    category: "tts",
    capabilities: ["Text to Speech"],
    size: "2.3 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit": {
    shortName: "0.6B Custom 8-bit",
    fullName: "Qwen3-TTS 12Hz 0.6B CustomVoice (MLX 8-bit)",
    description:
      "Balanced 8-bit CustomVoice for better quality with reduced VRAM",
    category: "tts",
    capabilities: ["Text to Speech", "8-bit"],
    size: "1.8 GB",
  },
  "Qwen3-TTS-12Hz-0.6B-CustomVoice-bf16": {
    shortName: "0.6B Custom BF16",
    fullName: "Qwen3-TTS 12Hz 0.6B CustomVoice (MLX bf16)",
    description: "BF16 precision for premium CustomVoice rendering",
    category: "tts",
    capabilities: ["Text to Speech", "BF16"],
    size: "2.3 GB",
  },
  // TTS 1.7B Base models
  "Qwen3-TTS-12Hz-1.7B-Base": {
    shortName: "1.7B Base",
    fullName: "Qwen3-TTS 12Hz 1.7B Base",
    description: "Higher quality voice cloning capabilities",
    category: "tts",
    capabilities: ["Voice Cloning"],
    size: "4.2 GB",
  },
  // TTS 1.7B CustomVoice
  "Qwen3-TTS-12Hz-1.7B-CustomVoice": {
    shortName: "1.7B CustomVoice",
    fullName: "Qwen3-TTS 12Hz 1.7B CustomVoice",
    description: "Premium quality with 9 built-in voices",
    category: "tts",
    capabilities: ["Text to Speech"],
    size: "4.2 GB",
  },
  // TTS 1.7B VoiceDesign models
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign": {
    shortName: "1.7B VoiceDesign",
    fullName: "Qwen3-TTS 12Hz 1.7B VoiceDesign",
    description: "Generate voices from text descriptions",
    category: "tts",
    capabilities: ["Voice Design"],
    size: "4.2 GB",
  },
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit": {
    shortName: "1.7B Design 8-bit",
    fullName: "Qwen3-TTS 12Hz 1.7B VoiceDesign (MLX 8-bit)",
    description: "8-bit VoiceDesign balancing VRAM usage and quality",
    category: "tts",
    capabilities: ["Voice Design", "8-bit"],
    size: "2.9 GB",
  },
  "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16": {
    shortName: "1.7B Design BF16",
    fullName: "Qwen3-TTS 12Hz 1.7B VoiceDesign (MLX bf16)",
    description: "BF16 VoiceDesign with best timbre quality",
    category: "tts",
    capabilities: ["Voice Design", "BF16"],
    size: "4.2 GB",
  },
  "Kokoro-82M": {
    shortName: "Kokoro 82M",
    fullName: "Kokoro-82M by hexgrad",
    description:
      "Lightweight multilingual TTS model with 41 built-in voices in English, Spanish, French, Hindi, Italian and Portuguese",
    category: "tts",
    capabilities: ["Text to Speech", "41 Voices", "24 kHz"],
    size: "346 MB",
  },
  "VibeVoice-1.5B": {
    shortName: "VibeVoice 1.5B TTS",
    fullName: "Microsoft VibeVoice 1.5B",
    description:
      "Long-form Microsoft text-to-speech model with reference voice cloning",
    category: "tts",
    capabilities: ["Text to Speech", "Voice Cloning", "Long-form", "24 kHz"],
    size: "5.04 GB",
  },
  "FishAudio-S2-Pro": {
    shortName: "Fish S2 Pro",
    fullName: "Fish Audio S2 Pro",
    description:
      "Multilingual Fish Audio text-to-speech model with reference voice cloning",
    category: "tts",
    capabilities: ["Text to Speech", "Voice Cloning", "Reference Voice"],
    size: "10.7 GB",
  },
  "Chatterbox-500M": {
    shortName: "Chatterbox",
    fullName: "Chatterbox by Resemble AI",
    description:
      "Expressive English text-to-speech with a bundled voice, cloning from a few seconds of audio and an emotion-exaggeration control. Its audio carries Resemble AI's imperceptible PerTh watermark.",
    category: "tts",
    capabilities: [
      "Text to Speech",
      "Voice Cloning",
      "Exaggeration",
      "Streaming",
      "Watermarked",
      "24 kHz",
    ],
    size: "3.23 GB",
  },
  "Chatterbox-Multilingual-500M": {
    shortName: "Chatterbox Multilingual",
    fullName: "Chatterbox Multilingual by Resemble AI",
    description:
      "Chatterbox in 22 languages, chosen per request, with the same bundled voice, cloning and exaggeration control. Its audio carries Resemble AI's imperceptible PerTh watermark.",
    category: "tts",
    capabilities: [
      "Text to Speech",
      "Multilingual",
      "Voice Cloning",
      "Exaggeration",
      "Streaming",
      "Watermarked",
      "24 kHz",
    ],
    size: "3.28 GB",
  },
  "Maya1-3B": {
    shortName: "Maya1",
    fullName: "Maya1 by Maya Research",
    description:
      "Voice design from a written description, with inline emotion tags such as <laugh>, <sigh> and <whisper> placed in the text. Nine described voices are built in. A 3B model: slower than real time on Apple Silicon (about 4.5x the length of its speech on an M1 Pro).",
    category: "tts",
    capabilities: [
      "Text to Speech",
      "Voice Design",
      "Emotion Tags",
      "Streaming",
      "24 kHz",
    ],
    size: "6.70 GB",
  },
  "Maya1-3B-GGUF": {
    shortName: "Maya1 GGUF",
    fullName: "Maya1 by Maya Research (GGUF Q8_0)",
    description:
      "Maya1 with a Q8_0 decoder: the same described voices and inline emotion tags in about half the download and weight memory, and faster on Apple Silicon (about 3x the length of its speech on an M1 Pro).",
    category: "tts",
    capabilities: [
      "Text to Speech",
      "Voice Design",
      "Emotion Tags",
      "Streaming",
      "24 kHz",
      "GGUF",
      "Q8_0",
    ],
    size: "3.62 GB",
  },
  // Text Chat
  "Qwen3.5-0.8B": {
    shortName: "Qwen3.5 0.8B",
    fullName: "Qwen3.5 0.8B (GGUF Q4_K_M)",
    description: "Compact hybrid multimodal Qwen3.5 checkpoint",
    category: "chat",
    capabilities: ["Chat", "Multimodal", "GGUF", "Q4_K_M"],
    size: "0.74 GB",
  },
  "Qwen3.5-2B": {
    shortName: "Qwen3.5 2B",
    fullName: "Qwen3.5 2B (GGUF Q4_K_M)",
    description: "Balanced hybrid multimodal Qwen3.5 checkpoint",
    category: "chat",
    capabilities: ["Chat", "Multimodal", "GGUF", "Q4_K_M"],
    size: "1.95 GB",
  },
  "Qwen3.5-4B": {
    shortName: "Qwen3.5 4B",
    fullName: "Qwen3.5 4B (GGUF Q4_K_M)",
    description: "Higher-quality hybrid multimodal Qwen3.5 checkpoint",
    category: "chat",
    capabilities: ["Chat", "Multimodal", "GGUF", "Q4_K_M"],
    size: "3.41 GB",
  },
  "Qwen3.5-9B": {
    shortName: "Qwen3.5 9B",
    fullName: "Qwen3.5 9B (GGUF Q4_K_M)",
    description: "Largest hybrid multimodal Qwen3.5 checkpoint, for the best answers",
    category: "chat",
    capabilities: ["Chat", "Multimodal", "GGUF", "Q4_K_M"],
    size: "6.60 GB",
  },
  "Qwen3.8-27B-FP8": {
    shortName: "Qwen3.8 27B",
    fullName: "Qwen3.8 27B (Block FP8)",
    description:
      "Large dedicated Qwen3.8 text-chat architecture with block-scaled FP8 weights",
    category: "chat",
    capabilities: [
      "Text Chat",
      "Thinking",
      "Reasoning Effort",
      "Tool Calling",
      "FP8",
    ],
    size: "28.74 GiB",
  },
  "Qwen3.6-35B-A3B-FP8": {
    shortName: "Qwen3.6 35B-A3B",
    fullName: "Qwen3.6 35B-A3B (Block FP8 MoE)",
    description:
      "Hybrid linear-attention sparse-MoE checkpoint (256 experts, 8 routed + 1 shared) with block-scaled FP8 weights",
    category: "chat",
    capabilities: ["Text Chat", "Thinking", "MoE", "FP8"],
    size: "34.89 GiB",
  },
  "LFM2.5-1.2B-Instruct-GGUF": {
    shortName: "LFM2.5 1.2B Instruct",
    fullName: "LFM2.5 1.2B Instruct (GGUF Q4_K_M)",
    description: "On-device instruction chat model from Liquid AI",
    category: "chat",
    capabilities: ["Text Chat", "GGUF", "Q4_K_M"],
    size: "0.68 GB",
  },
  "LFM2.5-1.2B-Thinking-GGUF": {
    shortName: "LFM2.5 1.2B Thinking",
    fullName: "LFM2.5 1.2B Thinking (GGUF Q4_K_M)",
    description: "Reasoning-tuned on-device chat model from Liquid AI",
    category: "chat",
    capabilities: ["Text Chat", "GGUF", "Q4_K_M", "Thinking"],
    size: "0.68 GB",
  },
  "LFM2.5-Audio-1.5B-GGUF": {
    shortName: "LFM2.5 Audio 1.5B",
    fullName: "LFM2.5 Audio 1.5B (GGUF Q4_0 bundle)",
    description:
      "Native end-to-end speech model from Liquid AI with unified audio-chat generation",
    category: "asr",
    categories: ["asr", "tts"],
    capabilities: [
      "Speech-to-speech",
      "Transcription",
      "Text to Speech",
      "GGUF",
      "Q4_0",
      "Unified Voice",
    ],
    size: "1.14 GB",
  },
  "Gemma-3-4b-it": {
    shortName: "Gemma 3 4B",
    fullName: "Gemma 3 4B Instruct",
    description: "Higher-quality Gemma 3 instruction model for local chat",
    category: "chat",
    capabilities: ["Text Chat", "Instruction Tuned"],
    size: "8.0 GB",
  },
  "Whisper-Large-v3-Turbo": {
    shortName: "Whisper v3 Turbo",
    fullName: "Whisper Large v3 Turbo",
    description:
      "OpenAI Whisper Turbo model for fast multilingual local transcription",
    category: "asr",
    capabilities: [
      "Transcription",
      "Multilingual",
      "Offline",
      "Streaming deltas",
    ],
    size: "1.5 GB",
  },
  "Qwen3-ASR-0.6B-GGUF": {
    shortName: "Qwen3-ASR 0.6B",
    fullName: "Qwen3-ASR 0.6B (GGUF Q8_0)",
    description:
      "Compact Qwen3-ASR speech recognition model with native GGUF loading",
    category: "asr",
    capabilities: ["Transcription", "GGUF", "Q8_0"],
    size: "1.0 GB",
  },
  "Qwen3-ASR-1.7B-GGUF": {
    shortName: "Qwen3-ASR 1.7B",
    fullName: "Qwen3-ASR 1.7B (GGUF Q8_0)",
    description:
      "Higher-accuracy Qwen3-ASR speech recognition model in GGUF format",
    category: "asr",
    capabilities: ["Transcription", "GGUF", "Q8_0"],
    size: "2.5 GB",
  },
  "VibeVoice-ASR": {
    shortName: "VibeVoice ASR",
    fullName: "Microsoft VibeVoice ASR",
    description:
      "Native Microsoft speech recognition model for long-form audio transcription",
    category: "asr",
    capabilities: ["Transcription", "Long-form", "24 kHz"],
    size: "16.16 GB",
  },
  "Nemotron-3.5-ASR-Streaming-0.6B": {
    shortName: "Nemotron 3.5 ASR",
    fullName: "NVIDIA Nemotron 3.5 ASR Streaming 0.6B",
    description:
      "Multilingual cache-aware FastConformer-RNNT ASR model with native Rust/Candle offline transcription and language-ID prompt conditioning",
    category: "asr",
    capabilities: [
      "Transcription",
      "40 locales",
      "16 kHz",
      "RNNT",
      "Streaming profiles",
    ],
    size: "2.37 GB",
  },
  "Granite-Speech-4.1-2B": {
    shortName: "Granite Speech 4.1",
    fullName: "IBM Granite Speech 4.1 2B",
    description:
      "IBM Granite Speech ASR with punctuated, capitalized transcripts in English, French, German, Spanish, Portuguese, and Japanese, plus keyword biasing",
    category: "asr",
    capabilities: [
      "Transcription",
      "Punctuation",
      "Keyword biasing",
      "16 kHz",
    ],
    size: "4.63 GB",
  },
  "Granite-Speech-5.0-2B": {
    shortName: "Granite Speech 5.0",
    fullName: "IBM Granite Speech 5.0 2B",
    description:
      "IBM Granite Speech 5.0 with punctuated transcripts in English, French, German, Spanish, Portuguese, and Japanese, speech translation to and from English, and keyword lists of up to 2,000 terms",
    category: "asr",
    capabilities: [
      "Transcription",
      "Translation",
      "Punctuation",
      "Keyword biasing",
      "16 kHz",
    ],
    size: "4.63 GB",
  },
  "Granite-Speech-5.0-2B-GGUF": {
    shortName: "Granite Speech 5.0 GGUF",
    fullName: "IBM Granite Speech 5.0 2B (GGUF Q8_0)",
    description:
      "IBM Granite Speech 5.0 with a Q8_0 language model: the same transcription, speech translation and keyword biasing in less memory",
    category: "asr",
    capabilities: [
      "Transcription",
      "Translation",
      "Punctuation",
      "Keyword biasing",
      "GGUF",
      "Q8_0",
    ],
    size: "3.12 GB",
  },
  "Granite-Speech-4.1-2B-Plus": {
    shortName: "Granite Speech 4.1 Plus",
    fullName: "IBM Granite Speech 4.1 2B Plus",
    description:
      "IBM Granite Speech rich transcription model with speaker-attributed transcripts, word timestamps, prefix prompting, and keyword biasing; transcripts are unpunctuated by design",
    category: "asr",
    capabilities: [
      "Transcription",
      "Speaker attribution",
      "Word timestamps",
      "Prefix text",
      "Keyword biasing",
      "16 kHz",
    ],
    size: "4.22 GB",
  },
  "Qwen3-ForcedAligner-0.6B": {
    shortName: "ForcedAligner 0.6B",
    fullName: "Qwen3-ForcedAligner 0.6B",
    description: "Aligns transcript text to precise speech timestamps",
    category: "asr",
    capabilities: ["Forced Alignment", "Word timestamps"],
    size: "1.7 GB",
  },
  "Parakeet-TDT-0.6B-v3": {
    shortName: "Parakeet v3",
    fullName: "Parakeet-TDT 0.6B v3",
    description: "Multilingual FastConformer-TDT ASR model in .nemo format",
    category: "asr",
    capabilities: ["Transcription", "25 EU languages"],
    size: "9.3 GB",
  },
  "diar_streaming_sortformer_4spk-v2.1": {
    shortName: "Sortformer 4spk",
    fullName: "Streaming Sortformer 4spk v2.1",
    description:
      "Streaming speaker diarization model from NVIDIA in .nemo format",
    category: "asr",
    capabilities: ["Diarization", "Up to 4 speakers", "Streaming"],
    size: "0.5 GB",
  },
  "Nemotron-3-Diarization": {
    shortName: "Nemotron 3 Diarization",
    fullName: "NVIDIA Nemotron 3 Diarization",
    description:
      "8-speaker streaming Sortformer-family diarization model from NVIDIA in .nemo format",
    category: "asr",
    capabilities: ["Diarization", "Up to 8 speakers", "Streaming"],
    size: "0.2 GB",
  },
  // Voxtral
  "Voxtral-4B-TTS-2603": {
    shortName: "Voxtral 4B TTS",
    fullName: "Voxtral 4B TTS by Mistral AI",
    description: "Mistral AI text to speech model with 20 built-in voices",
    category: "tts",
    capabilities: ["Text to Speech", "20 Voices", "Streaming"],
    size: "8.65 GB",
  },
  "Voxtral-Mini-4B-Realtime-2602": {
    shortName: "Voxtral 4B",
    fullName: "Voxtral Mini 4B Realtime",
    description: "Mistral AI speech transcription model",
    category: "asr",
    capabilities: ["Transcription"],
    size: "8.0 GB",
  },
};

const TASK_CATEGORY: Record<string, ModelDetail["category"]> = {
  tts: "tts",
  chat: "chat",
  asr: "asr",
  diarization: "asr",
  forced_align: "asr",
  audio_chat: "asr",
};

/**
 * UI metadata for a model: the curated entry when there is one, otherwise a
 * minimal entry built from the API's display name and task, so models added
 * to the server catalog still appear before the UI table lists them.
 */
export function resolveModelDetail(
  model: Pick<ModelInfo, "variant" | "display_name" | "task">,
): ModelDetail {
  const curated = MODEL_DETAILS[model.variant];
  if (curated) {
    return curated;
  }
  const name = model.display_name?.trim() || model.variant;
  return {
    shortName: name,
    fullName: name,
    description: "",
    category: TASK_CATEGORY[model.task ?? ""] ?? "asr",
    capabilities: [],
    size: "",
  };
}

/**
 * Friendly display name for a model variant ("Qwen3 0.6B CustomVoice"):
 * the curated short name, then the API display name, then the raw id.
 */
export function getModelDisplayName(
  variant: string,
  apiDisplayName?: string | null,
): string {
  const details = MODEL_DETAILS[variant];
  if (details) {
    return withQwen3Prefix(details.shortName, variant);
  }
  return apiDisplayName?.trim() || variant;
}
