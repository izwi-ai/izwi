export type ViewMode =
  | "custom-voice"
  | "voice-clone"
  | "voice-design"
  | "transcription"
  | "chat";

export interface ViewConfig {
  id: ViewMode;
  label: string;
  description: string;
  icon: string;
  modelFilter: (variant: string) => boolean;
  emptyStateTitle: string;
  emptyStateDescription: string;
  disabled?: boolean;
}

export function isKokoroVariant(variant: string): boolean {
  const normalized = variant.toLowerCase();
  return normalized === "kokoro-82m" || normalized.includes("kokoro-82m");
}

export function isLfm25AudioVariant(variant: string): boolean {
  const normalized = variant.trim().toLowerCase();
  return (
    normalized === "lfm2.5-audio-1.5b-gguf" ||
    normalized === "liquidai/lfm2.5-audio-1.5b-gguf" ||
    normalized === "liquidai/lfm2.5-audio-1.5b"
  );
}

export function isVoxtralTtsVariant(variant: string): boolean {
  const normalized = variant.trim().toLowerCase();
  return (
    normalized === "voxtral-4b-tts-2603" ||
    normalized === "mistralai/voxtral-4b-tts-2603" ||
    normalized === "voxtral 4b tts 2603"
  );
}

export function isVibeVoiceTtsVariant(variant: string): boolean {
  const normalized = variant.trim().toLowerCase();
  return (
    normalized === "vibevoice-1.5b" ||
    normalized === "microsoft/vibevoice-1.5b" ||
    normalized === "vibevoice 1.5b" ||
    normalized === "vibevoice 1.5b tts"
  );
}

function isVoxtralAsrVariant(variant: string): boolean {
  const normalized = variant.trim().toLowerCase();
  return (
    normalized === "voxtral-mini-4b-realtime-2602" ||
    normalized === "mistralai/voxtral-mini-4b-realtime-2602"
  );
}

function isVibeVoiceAsrVariant(variant: string): boolean {
  const normalized = variant.trim().toLowerCase();
  return (
    normalized === "vibevoice-asr" ||
    normalized === "microsoft/vibevoice-asr"
  );
}

function isQwenAsrVariant(variant: string): boolean {
  const normalized = variant.trim().toLowerCase();
  return (
    normalized === "qwen3-asr-0.6b-gguf" ||
    normalized === "qwen3-asr-1.7b-gguf" ||
    normalized === "qwen3-asr-0.6b" ||
    normalized === "qwen3-asr-1.7b" ||
    normalized === "qwen3_asr_0.6b_q8_0.gguf" ||
    normalized === "qwen3_asr_1.7b_q8_0.gguf"
  );
}

export function getSpeakerProfilesForVariant(variant: string | null): SpeakerProfile[] {
  if (!variant) {
    return QWEN_SPEAKERS;
  }
  if (isLfm25AudioVariant(variant)) {
    return LFM25_AUDIO_SPEAKERS;
  }
  if (isVoxtralTtsVariant(variant)) {
    return VOXTRAL_TTS_SPEAKERS;
  }
  if (isVibeVoiceTtsVariant(variant)) {
    return [];
  }
  if (isKokoroVariant(variant)) {
    return KOKORO_SPEAKERS;
  }
  return QWEN_SPEAKERS;
}

export const VIEW_CONFIGS: Record<ViewMode, ViewConfig> = {
  "custom-voice": {
    id: "custom-voice",
    label: "Text to Speech",
    description:
      "Generate speech with built-in voice profiles (Qwen3, Kokoro, Voxtral)",
    icon: "Volume2",
    modelFilter: (variant) =>
      variant.includes("CustomVoice") ||
      isKokoroVariant(variant) ||
      isVoxtralTtsVariant(variant),
    emptyStateTitle: "No TTS Model Loaded",
    emptyStateDescription:
      "Load a CustomVoice, Kokoro, or Voxtral TTS model to generate speech",
  },
  "voice-clone": {
    id: "voice-clone",
    label: "Voice Cloning",
    description: "Clone any voice with a reference audio sample",
    icon: "Users",
    modelFilter: (variant) =>
      (variant.includes("Base") && !variant.includes("Tokenizer")) ||
      isVibeVoiceTtsVariant(variant),
    emptyStateTitle: "No Base Model Loaded",
    emptyStateDescription:
      "Load a Base or VibeVoice model to clone voices from reference audio",
  },
  "voice-design": {
    id: "voice-design",
    label: "Voice Design",
    description: "Create unique voices from text descriptions",
    icon: "Wand2",
    modelFilter: (variant) => variant.includes("VoiceDesign"),
    emptyStateTitle: "No Voice Design Model Loaded",
    emptyStateDescription:
      "Load the VoiceDesign model to create voices from descriptions",
  },
  transcription: {
    id: "transcription",
    label: "Transcription",
    description:
      "Speech-to-text with Qwen3-ASR, VibeVoice-ASR, Whisper, Parakeet-TDT, Voxtral, and LFM2.5 Audio models",
    icon: "FileText",
    modelFilter: (variant) =>
      isQwenAsrVariant(variant) ||
      isVibeVoiceAsrVariant(variant) ||
      variant.includes("Whisper-Large-v3-Turbo") ||
      variant.includes("Parakeet-TDT") ||
      isVoxtralAsrVariant(variant) ||
      isLfm25AudioVariant(variant),
    emptyStateTitle: "No ASR Model Loaded",
    emptyStateDescription:
      "Download and load a Qwen3-ASR, VibeVoice-ASR, Whisper, Parakeet-TDT, Voxtral, or LFM2.5 Audio model for speech transcription",
  },
  chat: {
    id: "chat",
    label: "Chat",
    description: "Text and multimodal chat with Qwen3, Qwen3.5, LFM2.5, and Gemma 3",
    icon: "MessageSquare",
    modelFilter: (variant) =>
      variant === "Qwen3-0.6B-GGUF" ||
      variant === "Qwen3-1.7B-GGUF" ||
      variant === "Qwen3-4B-GGUF" ||
      variant === "Qwen3-8B-GGUF" ||
      variant === "Qwen3.5-0.8B" ||
      variant === "Qwen3.5-2B" ||
      variant === "Qwen3.5-4B" ||
      variant === "Qwen3.5-9B" ||
      variant === "LFM2.5-1.2B-Instruct-GGUF" ||
      variant === "LFM2.5-1.2B-Thinking-GGUF" ||
      variant === "Gemma-3-1b-it" ||
      variant === "Gemma-3-4b-it",
    emptyStateTitle: "No Chat Model Loaded",
    emptyStateDescription:
      "Load a Qwen3, Qwen3.5, LFM2.5, or Gemma chat model (including GGUF variants) to start chatting",
  },
};

export interface SpeakerProfile {
  id: string;
  name: string;
  language: string;
  description: string;
}

const QWEN_SPEAKERS: SpeakerProfile[] = [
  {
    id: "Vivian",
    name: "Vivian",
    language: "Chinese",
    description: "Warm and expressive female voice",
  },
  {
    id: "Serena",
    name: "Serena",
    language: "English",
    description: "Clear and professional female voice",
  },
  {
    id: "Ryan",
    name: "Ryan",
    language: "English",
    description: "Confident and friendly male voice",
  },
  {
    id: "Aiden",
    name: "Aiden",
    language: "English",
    description: "Young and energetic male voice",
  },
  {
    id: "Dylan",
    name: "Dylan",
    language: "English",
    description: "Deep and authoritative male voice",
  },
  {
    id: "Eric",
    name: "Eric",
    language: "English",
    description: "Calm and measured male voice",
  },
  {
    id: "Sohee",
    name: "Sohee",
    language: "Korean",
    description: "Gentle and melodic female voice",
  },
  {
    id: "Ono_anna",
    name: "Anna",
    language: "Japanese",
    description: "Soft and pleasant female voice",
  },
  {
    id: "Uncle_fu",
    name: "Uncle Fu",
    language: "Chinese",
    description: "Mature and wise male voice",
  },
];

const LFM25_AUDIO_SPEAKERS: SpeakerProfile[] = [
  {
    id: "US Female",
    name: "US Female",
    language: "English",
    description: "US female preset voice",
  },
  {
    id: "US Male",
    name: "US Male",
    language: "English",
    description: "US male preset voice",
  },
  {
    id: "UK Female",
    name: "UK Female",
    language: "English",
    description: "UK female preset voice",
  },
  {
    id: "UK Male",
    name: "UK Male",
    language: "English",
    description: "UK male preset voice",
  },
];

const VOXTRAL_TTS_SPEAKERS: SpeakerProfile[] = [
  {
    id: "casual_female",
    name: "Casual Female",
    language: "English",
    description: "Casual English female preset voice (Voxtral)",
  },
  {
    id: "casual_male",
    name: "Casual Male",
    language: "English",
    description: "Casual English male preset voice (Voxtral)",
  },
  {
    id: "cheerful_female",
    name: "Cheerful Female",
    language: "English",
    description: "Cheerful English female preset voice (Voxtral)",
  },
  {
    id: "neutral_female",
    name: "Neutral Female",
    language: "English",
    description: "Neutral English female preset voice (Voxtral)",
  },
  {
    id: "neutral_male",
    name: "Neutral Male",
    language: "English",
    description: "Neutral English male preset voice (Voxtral)",
  },
  {
    id: "pt_male",
    name: "Portuguese Male",
    language: "Portuguese",
    description: "Portuguese male preset voice (Voxtral)",
  },
  {
    id: "pt_female",
    name: "Portuguese Female",
    language: "Portuguese",
    description: "Portuguese female preset voice (Voxtral)",
  },
  {
    id: "nl_male",
    name: "Dutch Male",
    language: "Dutch",
    description: "Dutch male preset voice (Voxtral)",
  },
  {
    id: "nl_female",
    name: "Dutch Female",
    language: "Dutch",
    description: "Dutch female preset voice (Voxtral)",
  },
  {
    id: "it_male",
    name: "Italian Male",
    language: "Italian",
    description: "Italian male preset voice (Voxtral)",
  },
  {
    id: "it_female",
    name: "Italian Female",
    language: "Italian",
    description: "Italian female preset voice (Voxtral)",
  },
  {
    id: "fr_male",
    name: "French Male",
    language: "French",
    description: "French male preset voice (Voxtral)",
  },
  {
    id: "fr_female",
    name: "French Female",
    language: "French",
    description: "French female preset voice (Voxtral)",
  },
  {
    id: "es_male",
    name: "Spanish Male",
    language: "Spanish",
    description: "Spanish male preset voice (Voxtral)",
  },
  {
    id: "es_female",
    name: "Spanish Female",
    language: "Spanish",
    description: "Spanish female preset voice (Voxtral)",
  },
  {
    id: "de_male",
    name: "German Male",
    language: "German",
    description: "German male preset voice (Voxtral)",
  },
  {
    id: "de_female",
    name: "German Female",
    language: "German",
    description: "German female preset voice (Voxtral)",
  },
  {
    id: "ar_male",
    name: "Arabic Male",
    language: "Arabic",
    description: "Arabic male preset voice (Voxtral)",
  },
  {
    id: "hi_male",
    name: "Hindi Male",
    language: "Hindi",
    description: "Hindi male preset voice (Voxtral)",
  },
  {
    id: "hi_female",
    name: "Hindi Female",
    language: "Hindi",
    description: "Hindi female preset voice (Voxtral)",
  },
];

const KOKORO_VOICE_IDS = [
  "af_alloy",
  "af_aoede",
  "af_bella",
  "af_heart",
  "af_jessica",
  "af_kore",
  "af_nicole",
  "af_nova",
  "af_river",
  "af_sarah",
  "af_sky",
  "am_adam",
  "am_echo",
  "am_eric",
  "am_fenrir",
  "am_liam",
  "am_michael",
  "am_onyx",
  "am_puck",
  "am_santa",
  "bf_alice",
  "bf_emma",
  "bf_isabella",
  "bf_lily",
  "bm_daniel",
  "bm_fable",
  "bm_george",
  "bm_lewis",
  "ef_dora",
  "em_alex",
  "em_santa",
  "ff_siwis",
  "hf_alpha",
  "hf_beta",
  "hm_omega",
  "hm_psi",
  "if_sara",
  "im_nicola",
  "jf_alpha",
  "jf_gongitsune",
  "jf_nezumi",
  "jf_tebukuro",
  "jm_kumo",
  "pf_dora",
  "pm_alex",
  "pm_santa",
  "zf_xiaobei",
  "zf_xiaoni",
  "zf_xiaoxiao",
  "zf_xiaoyi",
  "zm_yunjian",
  "zm_yunxi",
  "zm_yunxia",
  "zm_yunyang",
] as const;

function kokoroLanguageForVoice(voiceId: string): string {
  const prefix = voiceId.slice(0, 1);
  switch (prefix) {
    case "a":
      return "American English";
    case "b":
      return "British English";
    case "j":
      return "Japanese";
    case "z":
      return "Mandarin Chinese";
    case "e":
      return "Spanish";
    case "f":
      return "French";
    case "h":
      return "Hindi";
    case "i":
      return "Italian";
    case "p":
      return "Brazilian Portuguese";
    default:
      return "Multilingual";
  }
}

function kokoroDisplayName(voiceId: string): string {
  const [, rawName = voiceId] = voiceId.split("_", 2);
  return rawName
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function kokoroGenderLabel(voiceId: string): string {
  const genderCode = voiceId.slice(1, 2);
  if (genderCode === "f") return "Female";
  if (genderCode === "m") return "Male";
  return "Voice";
}

const KOKORO_SPEAKERS: SpeakerProfile[] = KOKORO_VOICE_IDS.map((id) => ({
  id,
  name: kokoroDisplayName(id),
  language: kokoroLanguageForVoice(id),
  description: `${kokoroLanguageForVoice(id)} ${kokoroGenderLabel(id)} voice (Kokoro)`,
}));

export const LANGUAGES = [
  { id: "Auto", name: "Auto Detect" },
  { id: "Chinese", name: "Chinese" },
  { id: "English", name: "English" },
  { id: "Japanese", name: "Japanese" },
  { id: "Korean", name: "Korean" },
  { id: "German", name: "German" },
  { id: "French", name: "French" },
  { id: "Russian", name: "Russian" },
  { id: "Portuguese", name: "Portuguese" },
  { id: "Spanish", name: "Spanish" },
  { id: "Italian", name: "Italian" },
];

export const VOICE_DESIGN_PRESETS = [
  {
    name: "Professional Female",
    description:
      "A clear, confident, professional female voice with neutral accent. Suitable for business presentations and narration.",
  },
  {
    name: "Warm Storyteller",
    description:
      "A warm, gentle male voice with a storytelling quality. Perfect for audiobooks and bedtime stories.",
  },
  {
    name: "Energetic Youth",
    description:
      "A young, energetic voice full of enthusiasm. Great for advertisements and exciting content.",
  },
  {
    name: "Wise Elder",
    description:
      "A mature, thoughtful voice conveying wisdom and experience. Ideal for documentaries and educational content.",
  },
  {
    name: "Playful Character",
    description:
      "A playful, animated voice with expressive range. Perfect for character voices and entertainment.",
  },
  {
    name: "Calm Meditation",
    description:
      "A soft, soothing voice that promotes relaxation. Ideal for meditation guides and ASMR content.",
  },
];
