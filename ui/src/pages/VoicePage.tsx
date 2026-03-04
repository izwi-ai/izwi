import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Mic,
  Loader2,
  PhoneOff,
  AudioLines,
  Settings2,
  Download,
  Play,
  Square,
  X,
} from "lucide-react";
import clsx from "clsx";

import { api, ModelInfo } from "../api";
import {
  getSpeakerProfilesForVariant,
  isKokoroVariant,
  isLfmAudioVariant,
} from "../types";

import { Slider } from "../components/ui/slider";
import { Button } from "../components/ui/button";
import { PageShell } from "../components/PageShell";
import { cn } from "@/lib/utils";

type RuntimeStatus =
  | "idle"
  | "listening"
  | "user_speaking"
  | "processing"
  | "assistant_speaking";
type PipelineMode = "s2s" | "stt_chat_tts";

type VoiceRealtimeServerEvent =
  | { type: "connected"; protocol: string; server_time_ms?: number }
  | { type: "session_ready"; protocol: string }
  | {
      type: "input_stream_ready";
      vad?: {
        threshold?: number;
        min_speech_ms?: number;
        silence_duration_ms?: number;
      };
    }
  | { type: "input_stream_stopped" }
  | { type: "listening"; utterance_id: string; utterance_seq: number }
  | { type: "user_speech_start"; utterance_id: string; utterance_seq: number }
  | {
      type: "user_speech_end";
      utterance_id: string;
      utterance_seq: number;
      reason?: "silence" | "max_duration" | "stream_stopped";
    }
  | { type: "turn_processing"; utterance_id: string; utterance_seq: number }
  | {
      type: "user_transcript_start";
      utterance_id: string;
      utterance_seq: number;
    }
  | {
      type: "user_transcript_delta";
      utterance_id: string;
      utterance_seq: number;
      delta: string;
    }
  | {
      type: "user_transcript_final";
      utterance_id: string;
      utterance_seq: number;
      text: string;
      language?: string | null;
      audio_duration_secs?: number;
    }
  | {
      type: "assistant_text_start";
      utterance_id: string;
      utterance_seq: number;
    }
  | {
      type: "assistant_text_final";
      utterance_id: string;
      utterance_seq: number;
      text: string;
      raw_text?: string;
    }
  | {
      type: "assistant_audio_start";
      utterance_id: string;
      utterance_seq: number;
      sample_rate: number;
      audio_format: "pcm_i16" | "pcm_f32" | "wav";
    }
  | {
      type: "assistant_audio_done";
      utterance_id: string;
      utterance_seq: number;
    }
  | {
      type: "turn_done";
      utterance_id: string;
      utterance_seq: number;
      status: "ok" | "error" | "timeout" | "interrupted" | "no_input";
      reason?: string;
    }
  | {
      type: "error";
      utterance_id?: string | null;
      utterance_seq?: number | null;
      message: string;
    }
  | { type: "pong"; timestamp_ms?: number; server_time_ms?: number };

type VoiceRealtimeClientMessage =
  | { type: "session_start"; system_prompt?: string }
  | {
      type: "input_stream_start";
      asr_model_id: string;
      text_model_id: string;
      tts_model_id: string;
      speaker?: string;
      asr_language?: string;
      max_output_tokens?: number;
      vad_threshold?: number;
      min_speech_ms?: number;
      silence_duration_ms?: number;
      max_utterance_ms?: number;
      pre_roll_ms?: number;
      input_sample_rate?: number;
    }
  | { type: "input_stream_stop" }
  | { type: "interrupt"; reason?: string }
  | { type: "ping"; timestamp_ms?: number };

const VOICE_WS_BIN_MAGIC = "IVWS";
const VOICE_WS_BIN_VERSION = 1;
const VOICE_WS_BIN_KIND_CLIENT_PCM16 = 1;
const VOICE_WS_BIN_KIND_ASSISTANT_PCM16 = 2;
const VOICE_WS_BIN_CLIENT_HEADER_LEN = 16;
const VOICE_WS_BIN_ASSISTANT_HEADER_LEN = 24;

interface VoiceRealtimeAssistantAudioBinaryChunk {
  utteranceSeq: number;
  sequence: number;
  sampleRate: number;
  isFinal: boolean;
  pcm16Bytes: Uint8Array;
}

interface TranscriptEntry {
  id: string;
  role: "user" | "assistant";
  text: string;
  timestamp: number;
}

interface VoicePageProps {
  models: ModelInfo[];
  loading: boolean;
  downloadProgress: Record<
    string,
    {
      percent: number;
      currentFile: string;
      status: string;
      downloadedBytes: number;
      totalBytes: number;
    }
  >;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onError?: (message: string) => void;
}

const VOICE_AGENT_SYSTEM_PROMPT =
  "You are a helpful voice assistant. Reply with concise spoken-friendly language. Avoid markdown. Do not output <think> tags or internal reasoning. Return only the final spoken answer. Keep responses brief unless asked for details.";

const PIPELINE_LABELS: Record<PipelineMode, string> = {
  s2s: "Unified Speech Model",
  stt_chat_tts: "Modular Voice Stack",
};

const MODULAR_STACK_VARIANTS = {
  asr: "Parakeet-TDT-0.6B-v3",
  text: "Qwen3-1.7B-GGUF",
  tts: "Kokoro-82M",
} as const;

function parseFinalAnswer(content: string): string {
  const openTag = "<think>";
  const closeTag = "</think>";
  let out = content;

  while (true) {
    const start = out.indexOf(openTag);
    if (start === -1) break;
    const end = out.indexOf(closeTag, start + openTag.length);
    if (end === -1) {
      out = out.slice(0, start);
      break;
    }
    out = `${out.slice(0, start)}${out.slice(end + closeTag.length)}`;
  }

  return out.trim();
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function isAsrVariant(variant: string): boolean {
  return (
    variant.includes("Qwen3-ASR") ||
    variant.includes("Whisper-Large-v3-Turbo") ||
    variant.includes("Parakeet-TDT") ||
    variant.includes("Voxtral") ||
    isLfmAudioVariant(variant)
  );
}

function isLfm2Variant(variant: string): boolean {
  return isLfmAudioVariant(variant);
}

function formatModelVariantLabel(variant: string): string {
  const normalized = variant
    .replace(/-4bit\b/g, "-4-bit")
    .replace(/-8bit\b/g, "-8-bit");

  if (normalized.startsWith("Qwen3-ASR-")) {
    return normalized.replace("Qwen3-ASR-", "ASR ");
  }

  if (normalized.startsWith("Parakeet-TDT-")) {
    return normalized.replace("Parakeet-TDT-", "Parakeet ");
  }

  if (normalized.startsWith("Whisper-Large-v3-Turbo")) {
    return "Whisper Large v3 Turbo";
  }

  if (normalized.startsWith("Qwen3-TTS-12Hz-")) {
    return normalized.replace("Qwen3-TTS-12Hz-", "TTS ");
  }

  if (normalized.startsWith("Qwen3-ForcedAligner-")) {
    return normalized.replace("Qwen3-ForcedAligner-", "ForcedAligner ");
  }

  if (normalized.startsWith("Qwen3-")) {
    return normalized.replace("Qwen3-", "Qwen3 ");
  }

  if (normalized.startsWith("Gemma-3-")) {
    return normalized
      .replace("Gemma-3-1b-it", "Gemma 3 1B Instruct")
      .replace("Gemma-3-4b-it", "Gemma 3 4B Instruct");
  }

  if (isLfmAudioVariant(normalized)) {
    return normalized.replace("LFM2.5-Audio-", "LFM2.5 Audio ");
  }

  if (isKokoroVariant(normalized)) {
    return "Kokoro 82M";
  }

  return normalized.replace(/-/g, " ");
}

function isRunnableModelStatus(status: ModelInfo["status"]): boolean {
  return status === "ready";
}

function encodeWavPcm16(samples: Float32Array, sampleRate: number): Blob {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    const int16 = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    view.setInt16(offset, int16, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

function decodePcmI16Base64(base64Data: string): Float32Array {
  const binary = atob(base64Data);
  const sampleCount = Math.floor(binary.length / 2);
  const out = new Float32Array(sampleCount);

  for (let i = 0; i < sampleCount; i += 1) {
    const lo = binary.charCodeAt(i * 2);
    const hi = binary.charCodeAt(i * 2 + 1);
    let value = (hi << 8) | lo;
    if (value & 0x8000) {
      value -= 0x10000;
    }
    out[i] = value / 0x8000;
  }

  return out;
}

function mergeSampleChunks(chunks: Float32Array[]): Float32Array {
  const totalSamples = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(totalSamples);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function encodeFloat32ToPcm16Bytes(samples: Float32Array): Uint8Array {
  const out = new Uint8Array(samples.length * 2);
  const view = new DataView(out.buffer);
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    const int16 = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    view.setInt16(i * 2, int16, true);
  }
  return out;
}

function encodeVoiceRealtimeClientPcm16Frame(
  pcm16Bytes: Uint8Array,
  sampleRate: number,
  frameSeq: number,
): Uint8Array {
  const frame = new Uint8Array(
    VOICE_WS_BIN_CLIENT_HEADER_LEN + pcm16Bytes.length,
  );
  frame[0] = VOICE_WS_BIN_MAGIC.charCodeAt(0);
  frame[1] = VOICE_WS_BIN_MAGIC.charCodeAt(1);
  frame[2] = VOICE_WS_BIN_MAGIC.charCodeAt(2);
  frame[3] = VOICE_WS_BIN_MAGIC.charCodeAt(3);
  frame[4] = VOICE_WS_BIN_VERSION;
  frame[5] = VOICE_WS_BIN_KIND_CLIENT_PCM16;
  frame[6] = 0;
  frame[7] = 0;
  const view = new DataView(frame.buffer);
  view.setUint32(8, sampleRate >>> 0, true);
  view.setUint32(12, frameSeq >>> 0, true);
  frame.set(pcm16Bytes, VOICE_WS_BIN_CLIENT_HEADER_LEN);
  return frame;
}

function parseVoiceRealtimeAssistantAudioBinaryChunk(
  data: ArrayBuffer,
): VoiceRealtimeAssistantAudioBinaryChunk | null {
  if (data.byteLength < VOICE_WS_BIN_ASSISTANT_HEADER_LEN) {
    return null;
  }
  const bytes = new Uint8Array(data);
  if (
    String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]) !==
    VOICE_WS_BIN_MAGIC
  ) {
    return null;
  }
  const view = new DataView(data);
  const version = view.getUint8(4);
  const kind = view.getUint8(5);
  if (
    version !== VOICE_WS_BIN_VERSION ||
    kind !== VOICE_WS_BIN_KIND_ASSISTANT_PCM16
  ) {
    return null;
  }
  const flags = view.getUint16(6, true);
  const utteranceSeq = Number(view.getBigUint64(8, true));
  const sequence = view.getUint32(16, true);
  const sampleRate = view.getUint32(20, true);
  const pcm16Bytes = bytes.slice(VOICE_WS_BIN_ASSISTANT_HEADER_LEN);
  return {
    utteranceSeq,
    sequence,
    sampleRate,
    isFinal: (flags & 1) === 1,
    pcm16Bytes,
  };
}

function decodePcmI16Bytes(pcm16Bytes: Uint8Array): Float32Array {
  const sampleCount = Math.floor(pcm16Bytes.length / 2);
  const out = new Float32Array(sampleCount);
  const view = new DataView(
    pcm16Bytes.buffer,
    pcm16Bytes.byteOffset,
    pcm16Bytes.byteLength,
  );
  for (let i = 0; i < sampleCount; i += 1) {
    out[i] = view.getInt16(i * 2, true) / 0x8000;
  }
  return out;
}

function buildVoiceRealtimeWebSocketUrl(apiBaseUrl: string): string {
  const base = new URL(apiBaseUrl, window.location.origin);
  base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  base.pathname = `${base.pathname.replace(/\/$/, "")}/voice/realtime/ws`;
  base.search = "";
  base.hash = "";
  return base.toString();
}

function isVoiceRealtimeServerEvent(
  value: unknown,
): value is VoiceRealtimeServerEvent {
  return (
    !!value &&
    typeof value === "object" &&
    "type" in value &&
    typeof (value as { type?: unknown }).type === "string"
  );
}

function makeTranscriptEntryId(role: "user" | "assistant"): string {
  return `${role}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function transcodeToWav(
  inputBlob: Blob,
  targetSampleRate = 16000,
): Promise<Blob> {
  if (inputBlob.type === "audio/wav" || inputBlob.type === "audio/x-wav") {
    return inputBlob;
  }

  const decodeContext = new AudioContext();
  try {
    const sourceBytes = await inputBlob.arrayBuffer();
    const decoded = await decodeContext.decodeAudioData(sourceBytes.slice(0));

    const monoBuffer = decodeContext.createBuffer(
      1,
      decoded.length,
      decoded.sampleRate,
    );
    const mono = monoBuffer.getChannelData(0);

    for (let i = 0; i < decoded.length; i += 1) {
      let sum = 0;
      for (let ch = 0; ch < decoded.numberOfChannels; ch += 1) {
        sum += decoded.getChannelData(ch)[i] ?? 0;
      }
      mono[i] = sum / decoded.numberOfChannels;
    }

    const rendered = await (() => {
      if (decoded.sampleRate === targetSampleRate) {
        return Promise.resolve(monoBuffer);
      }

      const targetLength = Math.ceil(
        (monoBuffer.length * targetSampleRate) / monoBuffer.sampleRate,
      );
      const offline = new OfflineAudioContext(
        1,
        targetLength,
        targetSampleRate,
      );
      const source = offline.createBufferSource();
      source.buffer = monoBuffer;
      source.connect(offline.destination);
      source.start(0);
      return offline.startRendering();
    })();

    return encodeWavPcm16(rendered.getChannelData(0), targetSampleRate);
  } finally {
    decodeContext.close().catch(() => {});
  }
}

export function VoicePage({
  models,
  loading,
  downloadProgress,
  onDownload,
  onCancelDownload,
  onLoad,
  onUnload,
  onError,
}: VoicePageProps) {
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [audioLevel, setAudioLevel] = useState(0);

  const [pipelineMode, setPipelineMode] =
    useState<PipelineMode>("stt_chat_tts");
  const [selectedS2sModel, setSelectedS2sModel] = useState<string | null>(null);
  const [selectedSpeaker, setSelectedSpeaker] = useState("Serena");

  const [vadThreshold, setVadThreshold] = useState(0.02);
  const [silenceDurationMs, setSilenceDurationMs] = useState(900);
  const [minSpeechMs, setMinSpeechMs] = useState(300);
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const [isLoadAllRequested, setIsLoadAllRequested] = useState(false);

  const streamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamingProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const streamingProcessorSinkRef = useRef<GainNode | null>(null);
  const vadTimerRef = useRef<number | null>(null);
  const speechStartRef = useRef<number | null>(null);
  const silenceMsRef = useRef(0);
  const processingRef = useRef(false);
  const runtimeStatusRef = useRef<RuntimeStatus>("idle");
  const isSessionActiveRef = useRef(false);
  const turnIdRef = useRef(0);
  const agentSessionIdRef = useRef<string | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioUrlRef = useRef<string | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);
  const asrStreamAbortRef = useRef<AbortController | null>(null);
  const chatStreamAbortRef = useRef<AbortController | null>(null);
  const ttsStreamAbortRef = useRef<AbortController | null>(null);
  const ttsPlaybackContextRef = useRef<AudioContext | null>(null);
  const ttsPlaybackSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const ttsNextPlaybackTimeRef = useRef(0);
  const ttsSampleRateRef = useRef(24000);
  const ttsSamplesRef = useRef<Float32Array[]>([]);
  const ttsStreamSessionRef = useRef(0);
  const voiceWsRef = useRef<WebSocket | null>(null);
  const voiceWsConnectingRef = useRef<Promise<WebSocket> | null>(null);
  const voiceWsSessionReadyRef = useRef(false);
  const voiceWsInputStreamStartedRef = useRef(false);
  const voiceWsInputStreamStartingRef = useRef<Promise<void> | null>(null);
  const voiceWsInputStreamReadyPromiseRef = useRef<Promise<void> | null>(null);
  const voiceWsInputStreamReadySettleRef = useRef<
    ((error?: Error) => void) | null
  >(null);
  const voiceWsInputFrameSeqRef = useRef(0);
  const voiceMinAcceptedAssistantSeqRef = useRef(0);
  const voiceUserEntryIdsRef = useRef<Map<string, string>>(new Map());
  const voiceAssistantEntryIdsRef = useRef<Map<string, string>>(new Map());
  const loadAllDownloadRequestedRef = useRef<Set<string>>(new Set());
  const loadAllLoadRequestedRef = useRef<Set<string>>(new Set());
  const voiceWsPlaybackRef = useRef<{
    utteranceId: string;
    utteranceSeq: number;
    streamSession: number;
    streamDone: boolean;
    playbackStarted: boolean;
  } | null>(null);

  const sortedModels = useMemo(() => {
    return [...models]
      .filter((m) => !m.variant.includes("Tokenizer"))
      .sort((a, b) => a.variant.localeCompare(b.variant));
  }, [models]);

  const asrModels = useMemo(
    () => sortedModels.filter((m) => isAsrVariant(m.variant)),
    [sortedModels],
  );
  const s2sModels = useMemo(
    () => asrModels.filter((m) => isLfm2Variant(m.variant)),
    [asrModels],
  );
  const lfm25UnifiedInfo = useMemo(
    () =>
      s2sModels.find((m) => m.variant === "LFM2.5-Audio-1.5B") ??
      s2sModels.find((m) => m.variant === "LFM2.5-Audio-1.5B-4bit") ??
      s2sModels.find((m) => m.variant.startsWith("LFM2.5-Audio-")) ??
      null,
    [s2sModels],
  );
  const unifiedModelOptions = useMemo(
    () => [
      {
        key: "lfm2_5",
        label: "LFM2.5",
        description:
          "Improved speech quality with the same single-model runtime flow.",
        model: lfm25UnifiedInfo,
      },
    ],
    [lfm25UnifiedInfo],
  );
  const selectedAsrInfo = useMemo(
    () =>
      sortedModels.find((m) => m.variant === MODULAR_STACK_VARIANTS.asr) ??
      null,
    [sortedModels],
  );
  const selectedTextInfo = useMemo(
    () =>
      sortedModels.find((m) => m.variant === MODULAR_STACK_VARIANTS.text) ??
      null,
    [sortedModels],
  );
  const selectedTtsInfo = useMemo(
    () =>
      sortedModels.find((m) => m.variant === MODULAR_STACK_VARIANTS.tts) ??
      null,
    [sortedModels],
  );
  const selectedAsrModel = selectedAsrInfo?.variant ?? null;
  const selectedTextModel = selectedTextInfo?.variant ?? null;
  const selectedTtsModel = selectedTtsInfo?.variant ?? null;
  const modularStackModels = useMemo(
    () => [
      {
        key: "asr",
        role: "ASR",
        model: selectedAsrInfo,
        requiredVariant: MODULAR_STACK_VARIANTS.asr,
      },
      {
        key: "text",
        role: "Text",
        model: selectedTextInfo,
        requiredVariant: MODULAR_STACK_VARIANTS.text,
      },
      {
        key: "tts",
        role: "TTS",
        model: selectedTtsInfo,
        requiredVariant: MODULAR_STACK_VARIANTS.tts,
      },
    ],
    [selectedAsrInfo, selectedTextInfo, selectedTtsInfo],
  );
  const assistantSpeakers = useMemo(
    () => getSpeakerProfilesForVariant(selectedTtsModel),
    [selectedTtsModel],
  );
  const selectedS2sInfo = useMemo(
    () => s2sModels.find((m) => m.variant === selectedS2sModel) ?? null,
    [s2sModels, selectedS2sModel],
  );
  const lfm2DirectMode = pipelineMode === "s2s";
  const currentPipelineLabel = PIPELINE_LABELS[pipelineMode];

  useEffect(() => {
    if (!assistantSpeakers.some((speaker) => speaker.id === selectedSpeaker)) {
      setSelectedSpeaker(assistantSpeakers[0]?.id ?? "Serena");
    }
  }, [assistantSpeakers, selectedSpeaker]);

  useEffect(() => {
    runtimeStatusRef.current = runtimeStatus;
  }, [runtimeStatus]);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcript, runtimeStatus]);

  useEffect(() => {
    if (
      pipelineMode === "s2s" &&
      unifiedModelOptions.every((option) => !option.model)
    ) {
      setPipelineMode("stt_chat_tts");
    }
  }, [pipelineMode, unifiedModelOptions]);

  useEffect(() => {
    const candidates = unifiedModelOptions
      .map((option) => option.model)
      .filter((model): model is ModelInfo => model !== null);
    if (candidates.length === 0) {
      setSelectedS2sModel(null);
      return;
    }
    if (
      selectedS2sModel &&
      candidates.some((model) => model.variant === selectedS2sModel)
    ) {
      return;
    }
    const preferredS2s =
      candidates.find(
        (model) =>
          model.variant.startsWith("LFM2.5-Audio-") && model.status === "ready",
      ) ||
      candidates.find((model) => model.variant.startsWith("LFM2.5-Audio-")) ||
      candidates[0];

    setSelectedS2sModel(preferredS2s?.variant ?? null);
  }, [selectedS2sModel, unifiedModelOptions]);

  const hasRunnableConfig = useMemo(() => {
    if (lfm2DirectMode) {
      return !!selectedS2sInfo && isRunnableModelStatus(selectedS2sInfo.status);
    }

    if (!selectedAsrInfo || !isRunnableModelStatus(selectedAsrInfo.status)) {
      return false;
    }

    return (
      !!selectedTextInfo &&
      !!selectedTtsInfo &&
      isRunnableModelStatus(selectedTextInfo.status) &&
      isRunnableModelStatus(selectedTtsInfo.status)
    );
  }, [
    lfm2DirectMode,
    selectedAsrInfo,
    selectedS2sInfo,
    selectedTextInfo,
    selectedTtsInfo,
  ]);

  const hasRequiredModularModels = useMemo(
    () => modularStackModels.every((item) => item.model !== null),
    [modularStackModels],
  );

  const hasLoadableModularModels = useMemo(
    () =>
      modularStackModels.some((item) => {
        const model = item.model;
        return (
          !!model &&
          (model.status === "downloaded" ||
            model.status === "not_downloaded" ||
            model.status === "error")
        );
      }),
    [modularStackModels],
  );

  const isLoadAllBusy = useMemo(
    () =>
      modularStackModels.some((item) => {
        const model = item.model;
        return (
          !!model &&
          (model.status === "downloading" || model.status === "loading")
        );
      }),
    [modularStackModels],
  );

  const getModelProgress = useCallback(
    (model: ModelInfo | null) => {
      if (!model) return null;
      const progressValue = downloadProgress[model.variant];
      const progress = progressValue?.percent ?? model.download_progress ?? 0;
      return { progressValue, progress };
    },
    [downloadProgress],
  );

  const handleLoadAllModularStack = useCallback(() => {
    loadAllDownloadRequestedRef.current.clear();
    loadAllLoadRequestedRef.current.clear();
    setIsLoadAllRequested(true);
  }, []);

  useEffect(() => {
    if (!isLoadAllRequested) return;

    let allReady = true;
    let encounteredError = false;
    for (const item of modularStackModels) {
      const model = item.model;
      if (!model) {
        allReady = false;
        continue;
      }
      if (model.status === "ready") {
        continue;
      }
      if (
        model.status === "error" &&
        loadAllDownloadRequestedRef.current.has(model.variant)
      ) {
        encounteredError = true;
      }

      allReady = false;
      if (
        (model.status === "not_downloaded" || model.status === "error") &&
        !loadAllDownloadRequestedRef.current.has(model.variant)
      ) {
        loadAllDownloadRequestedRef.current.add(model.variant);
        onDownload(model.variant);
      }
      if (
        model.status === "downloaded" &&
        !loadAllLoadRequestedRef.current.has(model.variant)
      ) {
        loadAllLoadRequestedRef.current.add(model.variant);
        onLoad(model.variant);
      }
    }

    if (allReady || encounteredError) {
      setIsLoadAllRequested(false);
      loadAllDownloadRequestedRef.current.clear();
      loadAllLoadRequestedRef.current.clear();
    }
  }, [isLoadAllRequested, modularStackModels, onDownload, onLoad]);

  const onModelAction = useCallback(
    (model: ModelInfo) => {
      if (model.status === "not_downloaded" || model.status === "error") {
        onDownload(model.variant);
        return;
      }
      if (model.status === "downloaded") {
        onLoad(model.variant);
        return;
      }
      if (model.status === "ready") {
        onUnload(model.variant);
      }
    },
    [onDownload, onLoad, onUnload],
  );

  const handleUseUnifiedModel = useCallback(
    (model: ModelInfo) => {
      setSelectedS2sModel(model.variant);
      if (model.status === "downloaded") {
        onLoad(model.variant);
        return;
      }
      if (model.status === "not_downloaded" || model.status === "error") {
        onDownload(model.variant);
      }
    },
    [onDownload, onLoad],
  );

  const getUnifiedModelButtonLabel = useCallback(
    (model: ModelInfo | null, isSelected: boolean) => {
      if (!model) return "Unavailable";
      if (model.status === "ready") {
        return isSelected ? "Selected" : "Use model";
      }
      if (model.status === "downloaded") return "Load & use";
      if (model.status === "not_downloaded" || model.status === "error") {
        return "Download";
      }
      if (model.status === "downloading") return "Downloading";
      if (model.status === "loading") return "Loading";
      return "Unavailable";
    },
    [],
  );

  const isUnifiedModelButtonDisabled = useCallback(
    (model: ModelInfo | null, isSelected: boolean) => {
      if (!model) return true;
      if (model.status === "downloading" || model.status === "loading") {
        return true;
      }
      return model.status === "ready" && isSelected;
    },
    [],
  );

  const stopTtsStreamingPlayback = useCallback(() => {
    ttsStreamSessionRef.current += 1;

    if (ttsStreamAbortRef.current) {
      ttsStreamAbortRef.current.abort();
      ttsStreamAbortRef.current = null;
    }

    for (const source of ttsPlaybackSourcesRef.current) {
      try {
        source.stop();
      } catch {
        // Ignore already-stopped sources.
      }
    }
    ttsPlaybackSourcesRef.current.clear();

    if (ttsPlaybackContextRef.current) {
      ttsPlaybackContextRef.current.close().catch(() => {});
      ttsPlaybackContextRef.current = null;
    }

    ttsNextPlaybackTimeRef.current = 0;
    ttsSampleRateRef.current = 24000;
    ttsSamplesRef.current = [];
  }, []);

  const clearAudioPlayback = useCallback(() => {
    stopTtsStreamingPlayback();

    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
      audio.src = "";
    }

    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
  }, [stopTtsStreamingPlayback]);

  const settleVoiceRealtimeInputStreamReady = useCallback((error?: Error) => {
    const settle = voiceWsInputStreamReadySettleRef.current;
    if (!settle) return;

    voiceWsInputStreamReadySettleRef.current = null;
    voiceWsInputStreamReadyPromiseRef.current = null;
    settle(error);
  }, []);

  const waitForVoiceRealtimeInputStreamReady = useCallback(
    (timeoutMs = 8000): Promise<void> => {
      if (voiceWsInputStreamStartedRef.current) {
        return Promise.resolve();
      }
      if (voiceWsInputStreamReadyPromiseRef.current) {
        return voiceWsInputStreamReadyPromiseRef.current;
      }

      const promise = new Promise<void>((resolve, reject) => {
        const settle = (error?: Error) => {
          window.clearTimeout(timeoutId);
          if (error) {
            reject(error);
            return;
          }
          resolve();
        };

        const timeoutId = window.setTimeout(() => {
          if (voiceWsInputStreamReadySettleRef.current === settle) {
            voiceWsInputStreamReadySettleRef.current = null;
            voiceWsInputStreamReadyPromiseRef.current = null;
          }
          reject(
            new Error(
              "Voice realtime input stream did not become ready in time",
            ),
          );
        }, timeoutMs);

        voiceWsInputStreamReadySettleRef.current = settle;
      });

      voiceWsInputStreamReadyPromiseRef.current = promise;
      return promise;
    },
    [],
  );

  const closeVoiceRealtimeSocket = useCallback(
    (reason?: string) => {
      voiceWsSessionReadyRef.current = false;
      voiceWsInputStreamStartedRef.current = false;
      voiceWsInputFrameSeqRef.current = 0;
      voiceWsInputStreamStartingRef.current = null;
      voiceWsConnectingRef.current = null;
      settleVoiceRealtimeInputStreamReady(
        new Error("Voice realtime input stream stopped"),
      );

      const socket = voiceWsRef.current;
      voiceWsRef.current = null;
      if (socket) {
        try {
          if (
            socket.readyState === WebSocket.OPEN ||
            socket.readyState === WebSocket.CONNECTING
          ) {
            socket.close(1000, reason || "session_closed");
          }
        } catch {
          // Ignore close failures.
        }
      }
    },
    [settleVoiceRealtimeInputStreamReady],
  );

  const stopSession = useCallback(() => {
    isSessionActiveRef.current = false;
    turnIdRef.current += 1;
    processingRef.current = false;
    silenceMsRef.current = 0;
    speechStartRef.current = null;
    setRuntimeStatus("idle");
    setAudioLevel(0);

    if (vadTimerRef.current != null) {
      window.clearInterval(vadTimerRef.current);
      vadTimerRef.current = null;
    }

    if (
      voiceWsRef.current &&
      voiceWsRef.current.readyState === WebSocket.OPEN
    ) {
      try {
        voiceWsRef.current.send(
          JSON.stringify({
            type: "input_stream_stop",
          } satisfies VoiceRealtimeClientMessage),
        );
      } catch {
        // Best-effort during shutdown.
      }
    }

    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state === "recording") {
      recorder.stop();
    }
    mediaRecorderRef.current = null;

    if (streamingProcessorRef.current) {
      try {
        streamingProcessorRef.current.disconnect();
      } catch {
        // Ignore.
      }
      streamingProcessorRef.current.onaudioprocess = null;
      streamingProcessorRef.current = null;
    }
    if (streamingProcessorSinkRef.current) {
      try {
        streamingProcessorSinkRef.current.disconnect();
      } catch {
        // Ignore.
      }
      streamingProcessorSinkRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }

    if (asrStreamAbortRef.current) {
      asrStreamAbortRef.current.abort();
      asrStreamAbortRef.current = null;
    }

    if (chatStreamAbortRef.current) {
      chatStreamAbortRef.current.abort();
      chatStreamAbortRef.current = null;
    }

    closeVoiceRealtimeSocket("session_stopped");
    voiceWsPlaybackRef.current = null;
    voiceMinAcceptedAssistantSeqRef.current = 0;
    voiceUserEntryIdsRef.current.clear();
    voiceAssistantEntryIdsRef.current.clear();

    analyserRef.current = null;
    clearAudioPlayback();
  }, [clearAudioPlayback, closeVoiceRealtimeSocket]);

  useEffect(() => {
    return () => stopSession();
  }, [stopSession]);

  const appendTranscriptEntry = useCallback((entry: TranscriptEntry) => {
    setTranscript((prev) => [...prev, entry]);
  }, []);

  const setTranscriptEntryText = useCallback(
    (entryId: string, text: string) => {
      setTranscript((prev) => {
        const index = prev.findIndex((entry) => entry.id === entryId);
        if (index === -1) {
          return prev;
        }
        const next = [...prev];
        next[index] = {
          ...next[index],
          text,
        };
        return next;
      });
    },
    [],
  );

  const removeTranscriptEntry = useCallback((entryId: string) => {
    setTranscript((prev) => prev.filter((entry) => entry.id !== entryId));
  }, []);

  const sendVoiceRealtimeJson = useCallback(
    (message: VoiceRealtimeClientMessage) => {
      const socket = voiceWsRef.current;
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        throw new Error("Voice realtime websocket is not connected");
      }
      socket.send(JSON.stringify(message));
    },
    [],
  );

  const sendVoiceRealtimeBinary = useCallback((data: Uint8Array) => {
    const socket = voiceWsRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      throw new Error("Voice realtime websocket is not connected");
    }
    socket.send(data);
  }, []);

  const finalizeVoiceWsPlaybackIfComplete = useCallback(
    (utteranceSeq: number, streamSession: number) => {
      const active = voiceWsPlaybackRef.current;
      if (!active) return;
      if (active.utteranceSeq !== utteranceSeq) return;
      if (active.streamSession !== streamSession) return;
      if (!active.streamDone || ttsPlaybackSourcesRef.current.size > 0) {
        return;
      }

      if (ttsStreamSessionRef.current === streamSession) {
        const merged = mergeSampleChunks(ttsSamplesRef.current);
        if (merged.length > 0) {
          const wavBlob = encodeWavPcm16(merged, ttsSampleRateRef.current);
          const nextUrl = URL.createObjectURL(wavBlob);
          if (audioUrlRef.current) {
            URL.revokeObjectURL(audioUrlRef.current);
          }
          audioUrlRef.current = nextUrl;
        }

        if (ttsPlaybackContextRef.current) {
          ttsPlaybackContextRef.current.close().catch(() => {});
          ttsPlaybackContextRef.current = null;
        }

        ttsPlaybackSourcesRef.current.clear();
        ttsNextPlaybackTimeRef.current = 0;
        ttsSamplesRef.current = [];
        ttsStreamAbortRef.current = null;
      }

      if (
        voiceWsPlaybackRef.current &&
        voiceWsPlaybackRef.current.streamSession === streamSession
      ) {
        voiceWsPlaybackRef.current = null;
      }

      processingRef.current = false;
      if (
        isSessionActiveRef.current &&
        runtimeStatusRef.current !== "user_speaking"
      ) {
        setRuntimeStatus("listening");
      }
    },
    [],
  );

  const handleVoiceRealtimeAssistantAudioBinaryChunk = useCallback(
    (chunk: VoiceRealtimeAssistantAudioBinaryChunk) => {
      const playback = voiceWsPlaybackRef.current;
      if (!playback) return;
      if (playback.utteranceSeq !== chunk.utteranceSeq) {
        return;
      }

      const context = ttsPlaybackContextRef.current;
      if (!context) return;

      if (
        chunk.sampleRate > 0 &&
        ttsSampleRateRef.current !== chunk.sampleRate
      ) {
        ttsSampleRateRef.current = chunk.sampleRate;
      }

      const samples = decodePcmI16Bytes(chunk.pcm16Bytes);
      if (samples.length === 0) return;

      if (!playback.playbackStarted) {
        playback.playbackStarted = true;
        processingRef.current = false;
        if (isSessionActiveRef.current) {
          setRuntimeStatus("assistant_speaking");
        }
      }

      ttsSamplesRef.current.push(samples);

      const buffer = context.createBuffer(
        1,
        samples.length,
        ttsSampleRateRef.current,
      );
      const samplesForPlayback = new Float32Array(samples.length);
      samplesForPlayback.set(samples);
      buffer.copyToChannel(samplesForPlayback, 0);

      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);

      const scheduledAt = Math.max(
        context.currentTime + 0.02,
        ttsNextPlaybackTimeRef.current,
      );
      source.start(scheduledAt);
      ttsNextPlaybackTimeRef.current = scheduledAt + buffer.duration;

      const streamSession = playback.streamSession;
      const utteranceSeq = playback.utteranceSeq;
      ttsPlaybackSourcesRef.current.add(source);
      source.onended = () => {
        ttsPlaybackSourcesRef.current.delete(source);
        finalizeVoiceWsPlaybackIfComplete(utteranceSeq, streamSession);
      };

      if (chunk.isFinal) {
        playback.streamDone = true;
      }

      if (context.state === "suspended") {
        context.resume().catch(() => {});
      }
    },
    [finalizeVoiceWsPlaybackIfComplete],
  );

  const handleVoiceRealtimeServerEvent = useCallback(
    (event: VoiceRealtimeServerEvent) => {
      const eventSeq =
        "utterance_seq" in event && typeof event.utterance_seq === "number"
          ? event.utterance_seq
          : null;
      const ignoreAssistantEvent =
        eventSeq != null &&
        eventSeq < voiceMinAcceptedAssistantSeqRef.current &&
        (event.type.startsWith("assistant_") ||
          (event.type === "turn_done" && event.status === "interrupted"));

      if (ignoreAssistantEvent) {
        return;
      }

      switch (event.type) {
        case "connected":
          return;
        case "session_ready":
          voiceWsSessionReadyRef.current = true;
          if (
            isSessionActiveRef.current &&
            runtimeStatusRef.current === "idle"
          ) {
            setRuntimeStatus("listening");
          }
          return;
        case "input_stream_ready":
          voiceWsInputStreamStartedRef.current = true;
          settleVoiceRealtimeInputStreamReady();
          return;
        case "input_stream_stopped":
          voiceWsInputStreamStartedRef.current = false;
          settleVoiceRealtimeInputStreamReady(
            new Error("Voice realtime input stream stopped"),
          );
          return;
        case "listening":
          if (
            isSessionActiveRef.current &&
            runtimeStatusRef.current !== "user_speaking"
          ) {
            setRuntimeStatus("listening");
          }
          return;
        case "user_speech_start":
          voiceMinAcceptedAssistantSeqRef.current = Math.max(
            voiceMinAcceptedAssistantSeqRef.current,
            event.utterance_seq,
          );
          if (isSessionActiveRef.current) {
            setRuntimeStatus("user_speaking");
          }
          return;
        case "user_speech_end":
          processingRef.current = true;
          if (isSessionActiveRef.current) {
            setRuntimeStatus("processing");
          }
          return;
        case "turn_processing":
          processingRef.current = true;
          if (
            isSessionActiveRef.current &&
            runtimeStatusRef.current !== "user_speaking"
          ) {
            setRuntimeStatus("processing");
          }
          return;
        case "user_transcript_start": {
          const existingId = voiceUserEntryIdsRef.current.get(
            event.utterance_id,
          );
          if (!existingId) {
            const entryId = makeTranscriptEntryId("user");
            voiceUserEntryIdsRef.current.set(event.utterance_id, entryId);
            appendTranscriptEntry({
              id: entryId,
              role: "user",
              text: "",
              timestamp: Date.now(),
            });
          }
          return;
        }
        case "user_transcript_delta": {
          const entryId = voiceUserEntryIdsRef.current.get(event.utterance_id);
          if (!entryId) return;
          setTranscript((prev) => {
            const index = prev.findIndex((entry) => entry.id === entryId);
            if (index === -1) return prev;
            const next = [...prev];
            next[index] = {
              ...next[index],
              text: `${next[index].text}${event.delta}`,
            };
            return next;
          });
          return;
        }
        case "user_transcript_final": {
          const entryId = voiceUserEntryIdsRef.current.get(event.utterance_id);
          if (!entryId) return;
          const finalText = (event.text ?? "").trim();
          if (finalText) {
            setTranscriptEntryText(entryId, finalText);
          } else {
            removeTranscriptEntry(entryId);
            voiceUserEntryIdsRef.current.delete(event.utterance_id);
          }
          return;
        }
        case "assistant_text_start": {
          const existingId = voiceAssistantEntryIdsRef.current.get(
            event.utterance_id,
          );
          if (!existingId) {
            const entryId = makeTranscriptEntryId("assistant");
            voiceAssistantEntryIdsRef.current.set(event.utterance_id, entryId);
            appendTranscriptEntry({
              id: entryId,
              role: "assistant",
              text: "",
              timestamp: Date.now(),
            });
          }
          return;
        }
        case "assistant_text_final": {
          const entryId = voiceAssistantEntryIdsRef.current.get(
            event.utterance_id,
          );
          const finalText = parseFinalAnswer((event.text ?? "").trim());
          if (entryId) {
            if (finalText) {
              setTranscriptEntryText(entryId, finalText);
            } else {
              removeTranscriptEntry(entryId);
              voiceAssistantEntryIdsRef.current.delete(event.utterance_id);
            }
          }
          return;
        }
        case "assistant_audio_start": {
          if (event.audio_format !== "pcm_i16") {
            const message = `Unsupported streamed audio format '${event.audio_format}'. Expected pcm_i16.`;
            setError(message);
            onError?.(message);
            return;
          }

          clearAudioPlayback();

          const playbackContext = new AudioContext();
          ttsPlaybackContextRef.current = playbackContext;
          ttsNextPlaybackTimeRef.current = playbackContext.currentTime + 0.05;
          ttsSampleRateRef.current = event.sample_rate || 24000;
          ttsSamplesRef.current = [];

          const streamSession = ++ttsStreamSessionRef.current;
          voiceWsPlaybackRef.current = {
            utteranceId: event.utterance_id,
            utteranceSeq: event.utterance_seq,
            streamSession,
            streamDone: false,
            playbackStarted: false,
          };
          return;
        }
        case "assistant_audio_done": {
          const playback = voiceWsPlaybackRef.current;
          if (!playback) {
            if (
              isSessionActiveRef.current &&
              runtimeStatusRef.current !== "user_speaking"
            ) {
              setRuntimeStatus("listening");
            }
            processingRef.current = false;
            return;
          }
          if (
            playback.utteranceId !== event.utterance_id ||
            playback.utteranceSeq !== event.utterance_seq
          ) {
            return;
          }
          playback.streamDone = true;
          finalizeVoiceWsPlaybackIfComplete(
            playback.utteranceSeq,
            playback.streamSession,
          );
          return;
        }
        case "turn_done": {
          if (event.status !== "ok" && event.status !== "interrupted") {
            processingRef.current = false;
          }

          if (event.status === "interrupted") {
            if (
              voiceWsPlaybackRef.current?.utteranceSeq === event.utterance_seq
            ) {
              voiceWsPlaybackRef.current = null;
            }
            return;
          }

          if (!voiceWsPlaybackRef.current && isSessionActiveRef.current) {
            if (runtimeStatusRef.current !== "user_speaking") {
              setRuntimeStatus("listening");
            }
            processingRef.current = false;
          }
          return;
        }
        case "error": {
          const message = event.message || "Voice realtime error";
          if (!voiceWsInputStreamStartedRef.current) {
            settleVoiceRealtimeInputStreamReady(new Error(message));
          }
          setError(message);
          onError?.(message);
          processingRef.current = false;
          if (
            isSessionActiveRef.current &&
            runtimeStatusRef.current !== "user_speaking" &&
            runtimeStatusRef.current !== "assistant_speaking"
          ) {
            setRuntimeStatus("listening");
          }
          return;
        }
        case "pong":
          return;
      }
    },
    [
      appendTranscriptEntry,
      clearAudioPlayback,
      finalizeVoiceWsPlaybackIfComplete,
      onError,
      removeTranscriptEntry,
      settleVoiceRealtimeInputStreamReady,
      setTranscriptEntryText,
    ],
  );

  const ensureVoiceRealtimeSocket =
    useCallback(async (): Promise<WebSocket> => {
      const existing = voiceWsRef.current;
      if (existing && existing.readyState === WebSocket.OPEN) {
        return existing;
      }
      if (voiceWsConnectingRef.current) {
        return voiceWsConnectingRef.current;
      }

      const url = buildVoiceRealtimeWebSocketUrl(api.baseUrl);
      const promise = new Promise<WebSocket>((resolve, reject) => {
        let settled = false;
        const ws = new WebSocket(url);
        ws.binaryType = "arraybuffer";

        const settle = (fn: () => void) => {
          if (settled) return;
          settled = true;
          fn();
        };

        ws.onopen = () => {
          voiceWsRef.current = ws;
          voiceWsSessionReadyRef.current = false;
          voiceWsInputStreamStartedRef.current = false;
          voiceWsInputFrameSeqRef.current = 0;
          try {
            ws.send(
              JSON.stringify({
                type: "session_start",
                system_prompt: VOICE_AGENT_SYSTEM_PROMPT,
              } satisfies VoiceRealtimeClientMessage),
            );
          } catch (error) {
            settle(() => {
              voiceWsConnectingRef.current = null;
              reject(
                error instanceof Error
                  ? error
                  : new Error("Failed to initialize voice realtime websocket"),
              );
            });
            return;
          }
          settle(() => resolve(ws));
        };

        ws.onmessage = (messageEvent) => {
          if (messageEvent.data instanceof ArrayBuffer) {
            const chunk = parseVoiceRealtimeAssistantAudioBinaryChunk(
              messageEvent.data,
            );
            if (chunk) {
              handleVoiceRealtimeAssistantAudioBinaryChunk(chunk);
            }
            return;
          }
          if (messageEvent.data instanceof Blob) {
            void messageEvent.data.arrayBuffer().then((buffer) => {
              const chunk = parseVoiceRealtimeAssistantAudioBinaryChunk(buffer);
              if (chunk) {
                handleVoiceRealtimeAssistantAudioBinaryChunk(chunk);
              }
            });
            return;
          }
          if (typeof messageEvent.data !== "string") return;
          try {
            const parsed: unknown = JSON.parse(messageEvent.data);
            if (!isVoiceRealtimeServerEvent(parsed)) {
              return;
            }
            handleVoiceRealtimeServerEvent(parsed);
          } catch {
            // Ignore malformed events.
          }
        };

        ws.onerror = () => {
          const message = "Voice realtime websocket error";
          if (!settled) {
            settle(() => {
              voiceWsConnectingRef.current = null;
              reject(new Error(message));
            });
          }
        };

        ws.onclose = () => {
          if (!settled) {
            settle(() => {
              voiceWsConnectingRef.current = null;
              reject(
                new Error("Voice realtime connection closed during setup"),
              );
            });
          }
          const wasActive = isSessionActiveRef.current;
          const wasCurrent = voiceWsRef.current === ws;
          if (wasCurrent) {
            voiceWsRef.current = null;
          }
          voiceWsSessionReadyRef.current = false;
          voiceWsInputStreamStartedRef.current = false;
          voiceWsInputFrameSeqRef.current = 0;
          voiceWsInputStreamStartingRef.current = null;
          voiceWsConnectingRef.current = null;
          settleVoiceRealtimeInputStreamReady(
            new Error("Voice realtime connection closed"),
          );
          if (wasActive && wasCurrent) {
            processingRef.current = false;
            if (runtimeStatusRef.current !== "idle") {
              setRuntimeStatus("idle");
            }
            const message = "Voice realtime connection closed";
            setError(message);
            onError?.(message);
          }
        };
      });

      voiceWsConnectingRef.current = promise;
      try {
        return await promise;
      } finally {
        if (voiceWsConnectingRef.current === promise) {
          voiceWsConnectingRef.current = null;
        }
      }
    }, [
      handleVoiceRealtimeAssistantAudioBinaryChunk,
      handleVoiceRealtimeServerEvent,
      onError,
      settleVoiceRealtimeInputStreamReady,
    ]);

  const ensureVoiceRealtimeInputStreamStarted = useCallback(
    async (inputSampleRate: number) => {
      if (voiceWsInputStreamStartedRef.current) {
        return;
      }
      if (voiceWsInputStreamStartingRef.current) {
        return voiceWsInputStreamStartingRef.current;
      }
      if (!selectedAsrModel || !selectedTextModel || !selectedTtsModel) {
        throw new Error(
          "Required modular stack models are unavailable. Open Config.",
        );
      }

      const startPromise = (async () => {
        const socket = await ensureVoiceRealtimeSocket();
        if (socket.readyState !== WebSocket.OPEN) {
          throw new Error("Voice realtime websocket is not connected");
        }
        const readyPromise = waitForVoiceRealtimeInputStreamReady();
        sendVoiceRealtimeJson({
          type: "input_stream_start",
          asr_model_id: selectedAsrModel,
          text_model_id: selectedTextModel,
          tts_model_id: selectedTtsModel,
          speaker: selectedSpeaker,
          asr_language: "Auto",
          max_output_tokens: 1536,
          vad_threshold: vadThreshold,
          min_speech_ms: minSpeechMs,
          silence_duration_ms: silenceDurationMs,
          max_utterance_ms: 20_000,
          pre_roll_ms: 160,
          input_sample_rate: Math.round(inputSampleRate),
        });
        await readyPromise;
      })();

      voiceWsInputStreamStartingRef.current = startPromise.finally(() => {
        if (voiceWsInputStreamStartingRef.current === startPromise) {
          voiceWsInputStreamStartingRef.current = null;
        }
      });

      return voiceWsInputStreamStartingRef.current;
    },
    [
      ensureVoiceRealtimeSocket,
      minSpeechMs,
      selectedAsrModel,
      selectedSpeaker,
      selectedTextModel,
      selectedTtsModel,
      sendVoiceRealtimeJson,
      silenceDurationMs,
      vadThreshold,
      waitForVoiceRealtimeInputStreamReady,
    ],
  );

  const streamUserTranscription = useCallback(
    (audioBlob: Blob, modelId: string): Promise<string> =>
      new Promise((resolve, reject) => {
        const entryId = makeTranscriptEntryId("user");
        let assembledText = "";
        let settled = false;

        appendTranscriptEntry({
          id: entryId,
          role: "user",
          text: "",
          timestamp: Date.now(),
        });

        const settle = (fn: () => void) => {
          if (settled) return;
          settled = true;
          fn();
        };

        asrStreamAbortRef.current = api.asrTranscribeStream(
          {
            audio_file: audioBlob,
            audio_filename: "voice-turn.wav",
            model_id: modelId,
            language: "Auto",
          },
          {
            onDelta: (delta) => {
              assembledText += delta;
              setTranscriptEntryText(entryId, assembledText);
            },
            onPartial: (text) => {
              assembledText = text;
              setTranscriptEntryText(entryId, assembledText);
            },
            onFinal: (text) => {
              assembledText = text;
              setTranscriptEntryText(entryId, assembledText);
            },
            onError: (errorMessage) => {
              settle(() => {
                asrStreamAbortRef.current = null;
                const finalText = assembledText.trim();
                if (finalText) {
                  setTranscriptEntryText(entryId, finalText);
                } else {
                  removeTranscriptEntry(entryId);
                }
                reject(new Error(errorMessage));
              });
            },
            onDone: () => {
              settle(() => {
                asrStreamAbortRef.current = null;
                const finalText = assembledText.trim();
                if (finalText) {
                  setTranscriptEntryText(entryId, finalText);
                } else {
                  removeTranscriptEntry(entryId);
                }
                resolve(finalText);
              });
            },
          },
        );
      }),
    [appendTranscriptEntry, removeTranscriptEntry, setTranscriptEntryText],
  );

  const ensureAgentSession = useCallback(async (modelId: string) => {
    if (agentSessionIdRef.current) {
      return agentSessionIdRef.current;
    }

    const session = await api.createAgentSession({
      agent_id: "voice-agent",
      model_id: modelId,
      system_prompt: VOICE_AGENT_SYSTEM_PROMPT,
      planning_mode: "auto",
      title: "Voice Session",
    });

    agentSessionIdRef.current = session.id;
    return session.id;
  }, []);

  const streamAssistantResponse = useCallback(
    (userText: string, modelId: string): Promise<string> =>
      new Promise((resolve, reject) => {
        const entryId = makeTranscriptEntryId("assistant");
        let rawText = "";
        let settled = false;

        appendTranscriptEntry({
          id: entryId,
          role: "assistant",
          text: "",
          timestamp: Date.now(),
        });

        const settle = (fn: () => void) => {
          if (settled) return;
          settled = true;
          fn();
        };

        const updateVisibleText = () => {
          const visible = parseFinalAnswer(rawText);
          setTranscriptEntryText(entryId, visible);
        };

        const abortController = new AbortController();
        chatStreamAbortRef.current = abortController;

        const run = async () => {
          try {
            const sessionId = await ensureAgentSession(modelId);
            const response = await api.createAgentTurn(
              sessionId,
              {
                input: userText,
                model_id: modelId,
                max_output_tokens: 1536,
              },
              abortController.signal,
            );

            rawText = response.assistant_text ?? "";
            updateVisibleText();
            settle(() => {
              chatStreamAbortRef.current = null;
              const finalText = parseFinalAnswer(rawText) || rawText.trim();
              if (finalText) {
                setTranscriptEntryText(entryId, finalText);
              } else {
                removeTranscriptEntry(entryId);
              }
              resolve(finalText);
            });
          } catch (error) {
            if ((error as Error).name === "AbortError") {
              settle(() => {
                chatStreamAbortRef.current = null;
                const finalText = parseFinalAnswer(rawText) || rawText.trim();
                if (finalText) {
                  setTranscriptEntryText(entryId, finalText);
                } else {
                  removeTranscriptEntry(entryId);
                }
                reject(error as Error);
              });
              return;
            }

            settle(() => {
              chatStreamAbortRef.current = null;
              const finalText = parseFinalAnswer(rawText) || rawText.trim();
              if (finalText) {
                setTranscriptEntryText(entryId, finalText);
              } else {
                removeTranscriptEntry(entryId);
              }
              reject(
                error instanceof Error
                  ? error
                  : new Error("Agent response failed"),
              );
            });
          }
        };

        run();
      }),
    [
      appendTranscriptEntry,
      ensureAgentSession,
      removeTranscriptEntry,
      setTranscriptEntryText,
    ],
  );

  const streamAssistantSpeech = useCallback(
    (text: string, modelId: string, speaker: string, turnId: number) =>
      new Promise<void>((resolve, reject) => {
        clearAudioPlayback();

        const playbackContext = new AudioContext();
        ttsPlaybackContextRef.current = playbackContext;
        ttsNextPlaybackTimeRef.current = playbackContext.currentTime + 0.05;
        ttsSampleRateRef.current = 24000;
        ttsSamplesRef.current = [];

        const streamSession = ++ttsStreamSessionRef.current;
        let settled = false;
        let streamDone = false;
        let playbackStarted = false;

        const settle = (fn: () => void) => {
          if (settled) return;
          settled = true;
          fn();
        };

        const finalizeIfComplete = () => {
          if (!streamDone || ttsPlaybackSourcesRef.current.size > 0) {
            return;
          }

          if (ttsStreamSessionRef.current === streamSession) {
            const merged = mergeSampleChunks(ttsSamplesRef.current);
            if (merged.length > 0) {
              const wavBlob = encodeWavPcm16(merged, ttsSampleRateRef.current);
              const nextUrl = URL.createObjectURL(wavBlob);
              if (audioUrlRef.current) {
                URL.revokeObjectURL(audioUrlRef.current);
              }
              audioUrlRef.current = nextUrl;
            }

            if (ttsPlaybackContextRef.current) {
              ttsPlaybackContextRef.current.close().catch(() => {});
              ttsPlaybackContextRef.current = null;
            }

            ttsPlaybackSourcesRef.current.clear();
            ttsNextPlaybackTimeRef.current = 0;
            ttsSamplesRef.current = [];
            ttsStreamAbortRef.current = null;

            if (turnId === turnIdRef.current) {
              if (isSessionActiveRef.current) {
                setRuntimeStatus("listening");
              } else {
                setRuntimeStatus("idle");
              }
            }
          }

          settle(() => resolve());
        };

        ttsStreamAbortRef.current = api.generateTTSStream(
          {
            text,
            model_id: modelId,
            speaker,
            max_tokens: 0,
            format: "pcm",
          },
          {
            onStart: ({ sampleRate, audioFormat }) => {
              if (ttsStreamSessionRef.current !== streamSession) return;
              ttsSampleRateRef.current = sampleRate;

              if (audioFormat !== "pcm_i16") {
                stopTtsStreamingPlayback();
                settle(() => {
                  reject(
                    new Error(
                      `Unsupported streamed audio format '${audioFormat}'. Expected pcm_i16.`,
                    ),
                  );
                });
              }
            },
            onChunk: ({ audioBase64 }) => {
              if (ttsStreamSessionRef.current !== streamSession) return;

              const context = ttsPlaybackContextRef.current;
              if (!context) return;

              const samples = decodePcmI16Base64(audioBase64);
              if (samples.length === 0) return;

              if (!playbackStarted) {
                playbackStarted = true;
                processingRef.current = false;
                if (turnId === turnIdRef.current) {
                  setRuntimeStatus("assistant_speaking");
                }
              }

              ttsSamplesRef.current.push(samples);

              const buffer = context.createBuffer(
                1,
                samples.length,
                ttsSampleRateRef.current,
              );
              const samplesForPlayback = new Float32Array(samples.length);
              samplesForPlayback.set(samples);
              buffer.copyToChannel(samplesForPlayback, 0);

              const source = context.createBufferSource();
              source.buffer = buffer;
              source.connect(context.destination);

              const scheduledAt = Math.max(
                context.currentTime + 0.02,
                ttsNextPlaybackTimeRef.current,
              );
              source.start(scheduledAt);
              ttsNextPlaybackTimeRef.current = scheduledAt + buffer.duration;

              ttsPlaybackSourcesRef.current.add(source);
              source.onended = () => {
                ttsPlaybackSourcesRef.current.delete(source);
                finalizeIfComplete();
              };

              if (context.state === "suspended") {
                context.resume().catch(() => {});
              }
            },
            onError: (errorMessage) => {
              if (ttsStreamSessionRef.current !== streamSession) {
                settle(() => resolve());
                return;
              }

              stopTtsStreamingPlayback();
              settle(() => reject(new Error(errorMessage)));
            },
            onDone: () => {
              if (ttsStreamSessionRef.current !== streamSession) {
                settle(() => resolve());
                return;
              }

              streamDone = true;
              if (!playbackStarted) {
                processingRef.current = false;
              }
              finalizeIfComplete();
            },
          },
        );
      }),
    [clearAudioPlayback, stopTtsStreamingPlayback],
  );

  const playAssistantBlob = useCallback(
    (audioBlob: Blob, turnId: number) =>
      new Promise<void>((resolve, reject) => {
        clearAudioPlayback();

        const nextUrl = URL.createObjectURL(audioBlob);
        if (audioUrlRef.current) {
          URL.revokeObjectURL(audioUrlRef.current);
        }
        audioUrlRef.current = nextUrl;

        let audio = audioRef.current;
        if (!audio) {
          audio = new Audio();
          audioRef.current = audio;
        }

        const finalize = (error?: Error) => {
          audio!.onended = null;
          audio!.onerror = null;

          if (turnId === turnIdRef.current) {
            if (isSessionActiveRef.current) {
              setRuntimeStatus("listening");
            } else {
              setRuntimeStatus("idle");
            }
          }

          if (error) {
            reject(error);
          } else {
            resolve();
          }
        };

        audio.src = nextUrl;
        audio.onended = () => finalize();
        audio.onerror = () =>
          finalize(new Error("Failed to play assistant audio"));

        if (turnId === turnIdRef.current) {
          setRuntimeStatus("assistant_speaking");
        }

        audio.play().catch((error) => {
          finalize(
            error instanceof Error
              ? error
              : new Error("Failed to start assistant audio playback"),
          );
        });
      }),
    [clearAudioPlayback],
  );

  const processUtterance = useCallback(
    async (audioBlob: Blob) => {
      if (!isSessionActiveRef.current) {
        return;
      }

      if (
        (lfm2DirectMode && !selectedS2sModel) ||
        (!lfm2DirectMode && !selectedAsrModel) ||
        (!lfm2DirectMode && (!selectedTextModel || !selectedTtsModel))
      ) {
        setError(
          lfm2DirectMode
            ? "Select a speech-to-speech model before starting voice mode."
            : "Required modular stack models are unavailable. Open Config.",
        );
        setIsConfigOpen(true);
        setRuntimeStatus("listening");
        processingRef.current = false;
        return;
      }

      if (!hasRunnableConfig) {
        setError(
          "Required models are not loaded. Open Config and load the stack.",
        );
        setIsConfigOpen(true);
        setRuntimeStatus("listening");
        processingRef.current = false;
        return;
      }

      const turnId = turnIdRef.current + 1;
      turnIdRef.current = turnId;

      try {
        setRuntimeStatus("processing");

        if (lfm2DirectMode) {
          const wavBlob = await transcodeToWav(audioBlob, 24000);
          if (turnId !== turnIdRef.current || !isSessionActiveRef.current)
            return;

          const response = await api.speechToSpeech({
            audio_file: wavBlob,
            audio_filename: "voice-turn.wav",
            model_id: selectedS2sModel!,
            language: "English",
          });

          if (turnId !== turnIdRef.current || !isSessionActiveRef.current)
            return;

          let userText = response.transcription?.trim() || "";
          if (!userText) {
            try {
              const fallback = await api.asrTranscribe({
                audio_file: wavBlob,
                audio_filename: "voice-turn.wav",
                model_id: selectedS2sModel!,
                language: "English",
              });
              userText = fallback.transcription.trim();
            } catch {
              // Keep the turn visible even if transcription fallback fails.
            }
          }
          if (!userText) {
            userText = "User speech captured (transcription unavailable).";
          }
          if (userText) {
            appendTranscriptEntry({
              id: makeTranscriptEntryId("user"),
              role: "user",
              text: userText,
              timestamp: Date.now(),
            });
          }

          const assistantText = response.text.trim();
          if (assistantText) {
            appendTranscriptEntry({
              id: makeTranscriptEntryId("assistant"),
              role: "assistant",
              text: assistantText,
              timestamp: Date.now(),
            });
          }

          await playAssistantBlob(response.audioBlob, turnId);
          return;
        }

        const wavBlob = await transcodeToWav(audioBlob, 16000);
        if (turnId !== turnIdRef.current || !isSessionActiveRef.current) return;
        const userText = await streamUserTranscription(
          wavBlob,
          selectedAsrModel!,
        );

        if (turnId !== turnIdRef.current || !isSessionActiveRef.current) return;
        if (!userText) {
          processingRef.current = false;
          if (isSessionActiveRef.current) {
            setRuntimeStatus("listening");
          }
          return;
        }

        const assistantText = await streamAssistantResponse(
          userText,
          selectedTextModel!,
        );
        if (turnId !== turnIdRef.current || !isSessionActiveRef.current) return;
        if (!assistantText) {
          processingRef.current = false;
          if (isSessionActiveRef.current) {
            setRuntimeStatus("listening");
          }
          return;
        }

        await streamAssistantSpeech(
          assistantText,
          selectedTtsModel!,
          selectedSpeaker,
          turnId,
        );
      } catch (err) {
        if (turnId !== turnIdRef.current) {
          return;
        }

        const message =
          err instanceof Error ? err.message : "Voice turn failed";
        setError(message);
        onError?.(message);
        if (isSessionActiveRef.current) {
          setRuntimeStatus("listening");
        } else {
          setRuntimeStatus("idle");
        }
      } finally {
        if (turnId === turnIdRef.current) {
          processingRef.current = false;
          if (
            isSessionActiveRef.current &&
            runtimeStatusRef.current === "processing"
          ) {
            setRuntimeStatus("listening");
          }
        }
      }
    },
    [
      appendTranscriptEntry,
      hasRunnableConfig,
      lfm2DirectMode,
      onError,
      playAssistantBlob,
      selectedAsrModel,
      selectedS2sModel,
      selectedSpeaker,
      selectedTextModel,
      selectedTtsModel,
      streamAssistantResponse,
      streamAssistantSpeech,
      streamUserTranscription,
    ],
  );

  const startSession = useCallback(async () => {
    if (
      (lfm2DirectMode && !selectedS2sModel) ||
      (!lfm2DirectMode && !selectedAsrModel) ||
      (!lfm2DirectMode && (!selectedTextModel || !selectedTtsModel))
    ) {
      const message = lfm2DirectMode
        ? "Select a speech-to-speech model before starting voice mode."
        : "Required modular stack models are unavailable. Open Config.";
      setError(message);
      onError?.(message);
      setIsConfigOpen(true);
      return;
    }

    if (!hasRunnableConfig) {
      const message =
        "Required models are not loaded. Open Config and load the stack.";
      setError(message);
      onError?.(message);
      setIsConfigOpen(true);
      return;
    }

    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.2;
      source.connect(analyser);
      analyserRef.current = analyser;

      let recorder: MediaRecorder | null = null;
      if (lfm2DirectMode) {
        const mimeCandidates = [
          "audio/webm;codecs=opus",
          "audio/webm",
          "audio/mp4",
        ];
        for (const mimeType of mimeCandidates) {
          if (MediaRecorder.isTypeSupported(mimeType)) {
            recorder = new MediaRecorder(stream, { mimeType });
            break;
          }
        }
        if (!recorder) {
          recorder = new MediaRecorder(stream);
        }

        recorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            chunksRef.current.push(event.data);
          }
        };

        recorder.onstop = () => {
          const blob = new Blob(chunksRef.current, {
            type: recorder?.mimeType || "audio/webm",
          });
          chunksRef.current = [];

          if (blob.size < 1200) {
            processingRef.current = false;
            if (
              isSessionActiveRef.current &&
              runtimeStatusRef.current !== "assistant_speaking"
            ) {
              setRuntimeStatus("listening");
            }
            return;
          }

          void processUtterance(blob);
        };
      } else {
        const processor = audioContext.createScriptProcessor(2048, 1, 1);
        const sink = audioContext.createGain();
        sink.gain.value = 0;

        processor.onaudioprocess = (event) => {
          if (!isSessionActiveRef.current) return;
          if (!voiceWsInputStreamStartedRef.current) return;

          const inputBuffer = event.inputBuffer;
          const channelCount = inputBuffer.numberOfChannels;
          const frameCount = inputBuffer.length;
          if (frameCount <= 0 || channelCount <= 0) return;

          const mono = new Float32Array(frameCount);
          for (let ch = 0; ch < channelCount; ch += 1) {
            const channel = inputBuffer.getChannelData(ch);
            for (let i = 0; i < frameCount; i += 1) {
              mono[i] += (channel[i] ?? 0) / channelCount;
            }
          }

          const pcm16 = encodeFloat32ToPcm16Bytes(mono);
          const nextSeq = (voiceWsInputFrameSeqRef.current + 1) >>> 0;
          voiceWsInputFrameSeqRef.current = nextSeq;

          try {
            sendVoiceRealtimeBinary(
              encodeVoiceRealtimeClientPcm16Frame(
                pcm16,
                Math.round(inputBuffer.sampleRate),
                nextSeq,
              ),
            );
          } catch {
            // Best-effort; websocket lifecycle handles reconnect/error surfaces.
          }
        };

        source.connect(processor);
        processor.connect(sink);
        sink.connect(audioContext.destination);
        streamingProcessorRef.current = processor;
        streamingProcessorSinkRef.current = sink;
      }

      mediaRecorderRef.current = recorder;
      isSessionActiveRef.current = true;
      processingRef.current = false;
      silenceMsRef.current = 0;
      speechStartRef.current = null;
      setRuntimeStatus("listening");

      if (!lfm2DirectMode) {
        // Warm up realtime websocket + server-side VAD stream without blocking mic startup.
        void ensureVoiceRealtimeInputStreamStarted(
          audioContext.sampleRate,
        ).catch((err) => {
          const message =
            err instanceof Error
              ? err.message
              : "Voice realtime connection failed";
          if (!isSessionActiveRef.current) {
            return;
          }
          setError(message);
          onError?.(message);
        });
      }

      const VAD_INTERVAL = 80;
      vadTimerRef.current = window.setInterval(() => {
        const analyserNode = analyserRef.current;
        const recorderNode = mediaRecorderRef.current;
        if (!analyserNode || !isSessionActiveRef.current) return;

        const data = new Uint8Array(analyserNode.fftSize);
        analyserNode.getByteTimeDomainData(data);

        let sumSquares = 0;
        for (let i = 0; i < data.length; i += 1) {
          const centered = (data[i] - 128) / 128;
          sumSquares += centered * centered;
        }
        const rms = Math.sqrt(sumSquares / data.length);
        setAudioLevel(rms);

        const isSpeech = rms >= vadThreshold;
        const now = Date.now();

        if (!lfm2DirectMode) {
          if (isSpeech && runtimeStatusRef.current === "assistant_speaking") {
            const nextAccepted =
              (voiceWsPlaybackRef.current?.utteranceSeq ?? 0) + 1;
            voiceMinAcceptedAssistantSeqRef.current = Math.max(
              voiceMinAcceptedAssistantSeqRef.current,
              nextAccepted,
            );
            try {
              sendVoiceRealtimeJson({ type: "interrupt", reason: "barge_in" });
            } catch {
              // Best-effort; local playback is stopped immediately.
            }
            clearAudioPlayback();
            setRuntimeStatus("listening");
          }
          return;
        }

        if (!recorderNode) return;
        const isRecording = recorderNode.state === "recording";

        if (isSpeech) {
          silenceMsRef.current = 0;

          if (runtimeStatusRef.current === "assistant_speaking") {
            clearAudioPlayback();
            setRuntimeStatus("listening");
          }

          if (!isRecording && !processingRef.current) {
            chunksRef.current = [];
            recorderNode.start();
            speechStartRef.current = now;
            setRuntimeStatus("user_speaking");
          }
          return;
        }

        if (isRecording) {
          silenceMsRef.current += VAD_INTERVAL;
          const speechDuration = speechStartRef.current
            ? now - speechStartRef.current
            : 0;
          if (
            speechDuration >= minSpeechMs &&
            silenceMsRef.current >= silenceDurationMs
          ) {
            processingRef.current = true;
            setRuntimeStatus("processing");
            recorderNode.stop();
            silenceMsRef.current = 0;
            speechStartRef.current = null;
          }
        }
      }, VAD_INTERVAL);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Failed to start microphone session";
      setError(message);
      onError?.(message);
      stopSession();
    }
  }, [
    clearAudioPlayback,
    ensureVoiceRealtimeInputStreamStarted,
    hasRunnableConfig,
    lfm2DirectMode,
    minSpeechMs,
    onError,
    processUtterance,
    selectedAsrModel,
    selectedS2sModel,
    selectedTextModel,
    selectedTtsModel,
    sendVoiceRealtimeBinary,
    sendVoiceRealtimeJson,
    silenceDurationMs,
    stopSession,
    vadThreshold,
  ]);

  const toggleSession = () => {
    if (runtimeStatus === "idle") {
      void startSession();
    } else {
      stopSession();
    }
  };

  const statusLabel = {
    idle: "Idle",
    listening: "Listening",
    user_speaking: "User speaking",
    processing: "Thinking",
    assistant_speaking: "Assistant speaking",
  }[runtimeStatus];
  const formatTranscriptTimestamp = (timestamp: number) =>
    new Date(timestamp).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });

  const vadPercent = Math.min(
    100,
    Math.round((audioLevel / Math.max(vadThreshold, 0.001)) * 40),
  );

  const getStatusClass = (status: ModelInfo["status"]) => {
    switch (status) {
      case "ready":
        return "bg-white/10 border-white/20 text-[var(--text-secondary)]";
      case "loading":
      case "downloading":
        return "bg-amber-500/15 border-amber-500/40 text-[var(--text-muted)]";
      case "downloaded":
        return "bg-white/10 border-white/20 text-[var(--text-secondary)]";
      case "error":
        return "bg-red-500/15 border-red-500/40 text-red-300";
      default:
        return "bg-[var(--bg-surface-2)] border-[var(--border-muted)] text-[var(--text-muted)]";
    }
  };

  const getStatusLabel = (status: ModelInfo["status"]) => {
    switch (status) {
      case "not_downloaded":
        return "Not downloaded";
      case "downloading":
        return "Downloading";
      case "downloaded":
        return "Downloaded";
      case "loading":
        return "Loading";
      case "ready":
        return "Loaded";
      case "error":
        return "Error";
      default:
        return status;
    }
  };

  const startDisabled =
    (lfm2DirectMode && !selectedS2sModel) ||
    (!lfm2DirectMode &&
      (!selectedAsrModel || !selectedTextModel || !selectedTtsModel)) ||
    !hasRunnableConfig;
  const showTranscriptPanel = runtimeStatus !== "idle" || transcript.length > 0;

  if (loading) {
    return (
      <PageShell>
        <div className="flex flex-col items-center justify-center py-24 gap-3">
          <motion.div
            className="w-8 h-8 border-2 border-white border-t-transparent rounded-full"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          />
          <p className="text-sm text-[var(--text-muted)]">Loading models...</p>
        </div>
      </PageShell>
    );
  }

  return (
    <PageShell className="flex flex-col h-full min-h-[500px] relative overflow-hidden">
      <div className="absolute top-4 right-4 z-10 flex items-center gap-3">
        <div className="flex items-center gap-2">
          <span
            className={cn(
              "inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-medium transition-colors",
              hasRunnableConfig
                ? "bg-green-500/10 text-green-500 border border-green-500/20"
                : "bg-yellow-500/10 text-yellow-500 border border-yellow-500/20",
            )}
          >
            {hasRunnableConfig ? "Ready" : "Models Required"}
          </span>
          <span className="text-[11px] font-medium text-[var(--text-muted)] bg-[var(--bg-surface-2)] border border-[var(--border-muted)] px-2.5 py-1 rounded-full">
            {currentPipelineLabel}
          </span>
        </div>
        <button
          onClick={() => setIsConfigOpen(true)}
          className="btn btn-secondary h-8 px-3 rounded-full text-[11px] gap-1.5"
        >
          <Settings2 className="w-3.5 h-3.5" />
          Settings
        </button>
      </div>

      <div className="flex-1 flex flex-col items-center justify-center relative min-h-[400px]">
        {/* Dynamic Background Glow */}
        {runtimeStatus !== "idle" && (
          <motion.div
            className={cn(
              "absolute inset-0 z-0 pointer-events-none transition-opacity duration-1000",
              runtimeStatus === "user_speaking" ? "opacity-20" : "opacity-0",
            )}
            style={{
              background: `radial-gradient(circle at center, rgba(250,250,250,${Math.min(vadPercent / 100, 0.4)}) 0%, transparent 50%)`,
            }}
          />
        )}

        {/* Central Orb / Main Visualizer */}
        <div
          className={cn(
            "relative z-10 flex flex-col items-center justify-center transition-transform duration-300",
            showTranscriptPanel
              ? "mb-16 -translate-y-16 sm:-translate-y-20"
              : "mb-8",
          )}
        >
          <motion.div
            className="relative flex items-center justify-center"
            animate={{
              scale:
                runtimeStatus === "user_speaking"
                  ? 1 + vadPercent / 150
                  : runtimeStatus === "assistant_speaking"
                    ? [1, 1.1, 1]
                    : 1,
            }}
            transition={
              runtimeStatus === "assistant_speaking"
                ? { duration: 1.5, repeat: Infinity, ease: "easeInOut" }
                : { type: "spring", stiffness: 300, damping: 20 }
            }
          >
            <div
              className={cn(
                "w-48 h-48 sm:w-64 sm:h-64 rounded-full flex items-center justify-center transition-all duration-500",
                runtimeStatus === "idle"
                  ? "bg-[var(--bg-surface-1)] border-2 border-[var(--border-muted)]"
                  : runtimeStatus === "listening"
                    ? "bg-[var(--bg-surface-2)] border-2 border-[var(--border-strong)] shadow-[0_0_40px_rgba(255,255,255,0.05)]"
                    : runtimeStatus === "user_speaking"
                      ? "bg-white shadow-[0_0_60px_rgba(255,255,255,0.2)]"
                      : runtimeStatus === "processing"
                        ? "bg-[var(--bg-surface-3)] border-2 border-[var(--border-strong)]"
                        : runtimeStatus === "assistant_speaking"
                          ? "bg-black shadow-[0_0_60px_rgba(255,255,255,0.15)] border-2 border-white/20"
                          : "",
              )}
            >
              {runtimeStatus === "idle" ? (
                <button
                  onClick={toggleSession}
                  disabled={startDisabled}
                  className="w-full h-full rounded-full flex flex-col items-center justify-center gap-3 group hover:bg-[var(--bg-surface-2)] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <div className="w-16 h-16 rounded-full bg-[var(--bg-surface-3)] flex items-center justify-center group-hover:scale-110 transition-transform">
                    <Mic className="w-8 h-8 text-[var(--text-primary)]" />
                  </div>
                  <span className="text-sm font-medium text-[var(--text-secondary)]">
                    Start Conversation
                  </span>
                </button>
              ) : (
                <button
                  onClick={toggleSession}
                  className="w-full h-full rounded-full flex items-center justify-center group relative"
                >
                  {runtimeStatus === "assistant_speaking" ? (
                    <div className="flex gap-1.5 items-center justify-center h-12">
                      {[0, 1, 2, 3, 4].map((i) => (
                        <motion.div
                          key={i}
                          className="w-2 bg-white rounded-full"
                          animate={{ height: ["20%", "100%", "20%"] }}
                          transition={{
                            duration: 1,
                            repeat: Infinity,
                            delay: i * 0.1,
                            ease: "easeInOut",
                          }}
                        />
                      ))}
                    </div>
                  ) : runtimeStatus === "user_speaking" ? (
                    <AudioLines className="w-16 h-16 text-black" />
                  ) : runtimeStatus === "processing" ? (
                    <Loader2 className="w-12 h-12 text-white animate-spin" />
                  ) : (
                    <Mic className="w-12 h-12 text-white" />
                  )}

                  {/* Hover Overlay for stopping */}
                  <div className="absolute inset-0 bg-black/60 rounded-full opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity backdrop-blur-sm">
                    <Square className="w-10 h-10 text-white fill-white" />
                  </div>
                </button>
              )}
            </div>
          </motion.div>

          <div className="mt-8 text-center h-16 flex flex-col justify-center">
            <h2 className="text-2xl font-medium tracking-tight text-[var(--text-primary)]">
              {statusLabel}
            </h2>
            {runtimeStatus !== "idle" && (
              <p className="text-sm text-[var(--text-muted)] mt-1 max-w-md mx-auto line-clamp-1">
                {transcript.length > 0
                  ? transcript[transcript.length - 1].text
                  : "Listening..."}
              </p>
            )}
          </div>
        </div>

        {showTranscriptPanel && (
          <div className="absolute inset-x-4 bottom-24 sm:bottom-28 z-10 mx-auto w-full max-w-3xl">
            <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)]/95 backdrop-blur p-3 sm:p-4 shadow-2xl">
              <div className="flex items-center justify-between gap-2 mb-2">
                <span className="text-xs font-medium text-[var(--text-secondary)]">
                  Conversation Transcript
                </span>
                <span className="text-[11px] text-[var(--text-muted)]">
                  {transcript.length}{" "}
                  {transcript.length === 1 ? "turn" : "turns"}
                </span>
              </div>

              <div className="max-h-48 overflow-y-auto pr-1 space-y-2">
                {transcript.length === 0 ? (
                  <p className="text-xs text-[var(--text-muted)]">
                    Listening...
                  </p>
                ) : (
                  transcript.map((entry) => {
                    const isUser = entry.role === "user";
                    return (
                      <div
                        key={entry.id}
                        className={clsx(
                          "rounded-lg border px-3 py-2 text-sm whitespace-pre-wrap",
                          isUser
                            ? "bg-white text-black border-white"
                            : "bg-[var(--bg-surface-2)] text-[var(--text-primary)] border-[var(--border-muted)]",
                        )}
                      >
                        <div
                          className={clsx(
                            "text-[10px] mb-1 uppercase tracking-wide flex items-center justify-between gap-2",
                            isUser
                              ? "text-black/60"
                              : "text-[var(--text-muted)]",
                          )}
                        >
                          <span>{isUser ? "User" : "Assistant"}</span>
                          <span className="normal-case tracking-normal opacity-80">
                            {formatTranscriptTimestamp(entry.timestamp)}
                          </span>
                        </div>
                        {entry.text || (
                          <span className="opacity-70">
                            Waiting for text...
                          </span>
                        )}
                      </div>
                    );
                  })
                )}
                <div ref={transcriptEndRef} />
              </div>
            </div>
          </div>
        )}

        {/* Floating Action Bar */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ y: -20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: -20, opacity: 0 }}
              className="absolute top-16 left-1/2 -translate-x-1/2 z-20 w-full max-w-sm"
            >
              <div className="bg-red-500/10 border border-red-500/20 text-red-500 text-xs px-4 py-2.5 rounded-xl shadow-lg backdrop-blur-md text-center mx-4">
                {error}
              </div>
            </motion.div>
          )}
          {runtimeStatus !== "idle" && (
            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: 20, opacity: 0 }}
              className="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-3 p-2 bg-[var(--bg-surface-1)] border border-[var(--border-muted)] rounded-full shadow-2xl backdrop-blur-xl"
            >
              <button
                onClick={toggleSession}
                className="w-12 h-12 rounded-full bg-red-500/10 hover:bg-red-500/20 text-red-500 flex items-center justify-center transition-colors"
              >
                <PhoneOff className="w-5 h-5" />
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Slide-out Config Panel */}
      <AnimatePresence>
        {isConfigOpen && (
          <motion.div
            className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm p-4 sm:p-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsConfigOpen(false)}
          >
            <motion.div
              initial={{ y: 16, opacity: 0, scale: 0.98 }}
              animate={{ y: 0, opacity: 1, scale: 1 }}
              exit={{ y: 16, opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.2 }}
              className="mx-auto max-w-5xl max-h-[90vh] overflow-hidden card"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="px-4 sm:px-5 py-4 border-b border-[var(--border-muted)] flex items-center justify-between gap-3">
                <div>
                  <h2 className="text-base font-semibold text-white">
                    Voice Configuration
                  </h2>
                  <p className="text-xs text-[var(--text-muted)] mt-1">
                    Configure realtime model stack and manage model lifecycle.
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="gap-2"
                  onClick={() => setIsConfigOpen(false)}
                >
                  <X className="w-3.5 h-3.5" />
                  Close
                </Button>
              </div>

              <div className="p-4 sm:p-5 overflow-y-auto max-h-[calc(90vh-88px)] space-y-5">
                <section className="space-y-4">
                  <div className="flex items-center justify-between gap-2">
                    <h3 className="text-sm font-medium text-white">
                      Runtime Profile
                    </h3>
                    <span className="text-[11px] text-[var(--text-muted)]">
                      Choose how inference is orchestrated.
                    </span>
                  </div>
                  <div className="grid md:grid-cols-2 gap-3">
                    <button
                      className={cn(
                        "rounded-lg border p-3 text-left transition-colors",
                        lfm2DirectMode
                          ? "border-primary bg-primary/5 ring-1 ring-primary/20"
                          : "border-border bg-card hover:bg-accent hover:text-accent-foreground",
                      )}
                      onClick={() => setPipelineMode("s2s")}
                    >
                      <div className="text-sm font-medium">
                        Unified Speech Model
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        One LFM2.5 model handles user speech understanding and
                        assistant speech output.
                      </p>
                    </button>
                    <button
                      className={cn(
                        "rounded-lg border p-3 text-left transition-colors",
                        !lfm2DirectMode
                          ? "border-primary bg-primary/5 ring-1 ring-primary/20"
                          : "border-border bg-card hover:bg-accent hover:text-accent-foreground",
                      )}
                      onClick={() => setPipelineMode("stt_chat_tts")}
                    >
                      <div className="text-sm font-medium">
                        Modular Voice Stack
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        Uses a fixed local stack: Parakeet ASR, Qwen3-1.7B-GGUF,
                        and Kokoro-82M.
                      </p>
                    </button>
                  </div>
                  <div className="text-[11px] text-[var(--text-muted)]">
                    Current mode: {currentPipelineLabel}
                  </div>
                </section>

                <section className="space-y-3">
                  {lfm2DirectMode ? (
                    <>
                      <div className="flex items-center justify-between gap-2">
                        <h3 className="text-sm font-medium text-white">
                          Unified Models
                        </h3>
                        <span className="text-[11px] text-[var(--text-muted)]">
                          Choose an LFM2.5 model.
                        </span>
                      </div>
                      <div className="grid md:grid-cols-2 gap-3">
                        {unifiedModelOptions.map((option) => {
                          const model = option.model;
                          const isSelected =
                            !!model && selectedS2sModel === model.variant;
                          const progressMeta = getModelProgress(model);
                          return (
                            <div
                              key={option.key}
                              className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3 space-y-3"
                            >
                              <div className="flex items-start justify-between gap-2">
                                <div className="min-w-0">
                                  <div className="text-sm font-medium text-white">
                                    {option.label}
                                  </div>
                                  <div className="mt-0.5 text-xs text-[var(--text-muted)] truncate">
                                    {model
                                      ? formatModelVariantLabel(model.variant)
                                      : "Model not available in current catalog"}
                                  </div>
                                </div>
                                <span
                                  className={clsx(
                                    "text-[10px] px-1.5 py-0.5 rounded border whitespace-nowrap",
                                    model
                                      ? getStatusClass(model.status)
                                      : "bg-[var(--bg-surface-2)] border-[var(--border-muted)] text-[var(--text-muted)]",
                                  )}
                                >
                                  {model
                                    ? getStatusLabel(model.status)
                                    : "Unavailable"}
                                </span>
                              </div>
                              <p className="text-xs text-[var(--text-muted)]">
                                {option.description}
                              </p>
                              <div className="flex flex-wrap items-center gap-2">
                                {model?.status === "downloading" &&
                                  onCancelDownload && (
                                    <Button
                                      onClick={() =>
                                        onCancelDownload(model.variant)
                                      }
                                      variant="destructive"
                                      size="sm"
                                      className="text-xs h-8 gap-2"
                                    >
                                      <X className="w-3.5 h-3.5" />
                                      Cancel
                                    </Button>
                                  )}
                                <Button
                                  onClick={() =>
                                    model && handleUseUnifiedModel(model)
                                  }
                                  size="sm"
                                  variant={
                                    model?.status === "ready" && isSelected
                                      ? "outline"
                                      : "default"
                                  }
                                  disabled={isUnifiedModelButtonDisabled(
                                    model,
                                    isSelected,
                                  )}
                                  className="text-xs h-8 gap-2"
                                >
                                  {(model?.status === "loading" ||
                                    model?.status === "downloading") && (
                                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                  )}
                                  {(model?.status === "not_downloaded" ||
                                    model?.status === "error") && (
                                    <Download className="w-3.5 h-3.5" />
                                  )}
                                  {(model?.status === "downloaded" ||
                                    (model?.status === "ready" &&
                                      !isSelected)) && (
                                    <Play className="w-3.5 h-3.5" />
                                  )}
                                  {getUnifiedModelButtonLabel(
                                    model,
                                    isSelected,
                                  )}
                                </Button>
                              </div>
                              {model?.status === "downloading" &&
                                progressMeta && (
                                  <div>
                                    <div className="h-1.5 rounded bg-[var(--bg-surface-3)] overflow-hidden">
                                      <div
                                        className="h-full rounded bg-white transition-all duration-300"
                                        style={{
                                          width: `${progressMeta.progress}%`,
                                        }}
                                      />
                                    </div>
                                    <div className="mt-1 text-[11px] text-[var(--text-muted)]">
                                      Downloading{" "}
                                      {Math.round(progressMeta.progress)}%
                                      {progressMeta.progressValue &&
                                        progressMeta.progressValue.totalBytes >
                                          0 && (
                                          <>
                                            {" "}
                                            (
                                            {formatBytes(
                                              progressMeta.progressValue
                                                .downloadedBytes,
                                            )}{" "}
                                            /{" "}
                                            {formatBytes(
                                              progressMeta.progressValue
                                                .totalBytes,
                                            )}
                                            )
                                          </>
                                        )}
                                    </div>
                                  </div>
                                )}
                            </div>
                          );
                        })}
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <h3 className="text-sm font-medium text-white">
                            Modular Stack Models
                          </h3>
                          <p className="text-[11px] text-[var(--text-muted)] mt-1">
                            Fixed stack: Parakeet-TDT-0.6B-v3, Qwen3-1.7B-GGUF,
                            Kokoro-82M.
                          </p>
                        </div>
                        <Button
                          onClick={handleLoadAllModularStack}
                          size="sm"
                          className="text-xs h-8 gap-2"
                          disabled={
                            !hasRequiredModularModels ||
                            !hasLoadableModularModels ||
                            isLoadAllRequested ||
                            isLoadAllBusy
                          }
                        >
                          {isLoadAllRequested ? (
                            <Loader2 className="w-3.5 h-3.5 animate-spin" />
                          ) : (
                            <Play className="w-3.5 h-3.5" />
                          )}
                          {isLoadAllRequested ? "Loading all..." : "Load all"}
                        </Button>
                      </div>
                      <div className="space-y-3">
                        {modularStackModels.map((item) => {
                          const model = item.model;
                          const progressMeta = getModelProgress(model);
                          return (
                            <div
                              key={item.key}
                              className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-2.5 space-y-2"
                            >
                              <div className="flex items-start justify-between gap-2">
                                <div className="min-w-0">
                                  <div className="text-sm font-medium text-white">
                                    {item.role}
                                  </div>
                                  <div className="mt-0.5 text-xs text-[var(--text-muted)] truncate">
                                    {model
                                      ? formatModelVariantLabel(model.variant)
                                      : item.requiredVariant}
                                  </div>
                                </div>
                                <span
                                  className={clsx(
                                    "text-[10px] px-1.5 py-0.5 rounded border whitespace-nowrap",
                                    model
                                      ? getStatusClass(model.status)
                                      : "bg-[var(--bg-surface-2)] border-[var(--border-muted)] text-[var(--text-muted)]",
                                  )}
                                >
                                  {model
                                    ? getStatusLabel(model.status)
                                    : "Unavailable"}
                                </span>
                              </div>

                              <div className="flex flex-wrap items-center justify-end gap-2">
                                {model?.status === "downloading" &&
                                  onCancelDownload && (
                                    <Button
                                      onClick={() =>
                                        onCancelDownload(model.variant)
                                      }
                                      variant="destructive"
                                      size="sm"
                                      className="text-xs h-7 gap-2"
                                    >
                                      <X className="w-3.5 h-3.5" />
                                      Cancel
                                    </Button>
                                  )}
                                {model?.status === "loading" && (
                                  <Button
                                    disabled
                                    variant="outline"
                                    size="sm"
                                    className="text-xs h-7 gap-2"
                                  >
                                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                    Loading
                                  </Button>
                                )}
                                {model &&
                                  model.status !== "loading" &&
                                  model.status !== "downloading" && (
                                    <Button
                                      onClick={() => onModelAction(model)}
                                      variant={
                                        model.status === "ready"
                                          ? "outline"
                                          : "default"
                                      }
                                      size="sm"
                                      className="text-xs h-7 gap-2"
                                    >
                                      {(model.status === "not_downloaded" ||
                                        model.status === "error") && (
                                        <Download className="w-3.5 h-3.5" />
                                      )}
                                      {model.status === "downloaded" && (
                                        <Play className="w-3.5 h-3.5" />
                                      )}
                                      {model.status === "ready" && (
                                        <Square className="w-3.5 h-3.5" />
                                      )}
                                      {model.status === "not_downloaded" ||
                                      model.status === "error"
                                        ? "Download"
                                        : model.status === "downloaded"
                                          ? "Load"
                                          : "Unload"}
                                    </Button>
                                  )}
                              </div>

                              {model?.status === "downloading" &&
                                progressMeta && (
                                  <div>
                                    <div className="h-1.5 rounded bg-[var(--bg-surface-3)] overflow-hidden">
                                      <div
                                        className="h-full rounded bg-white transition-all duration-300"
                                        style={{
                                          width: `${progressMeta.progress}%`,
                                        }}
                                      />
                                    </div>
                                    <div className="mt-1 text-[11px] text-[var(--text-muted)]">
                                      Downloading{" "}
                                      {Math.round(progressMeta.progress)}%
                                      {progressMeta.progressValue &&
                                        progressMeta.progressValue.totalBytes >
                                          0 && (
                                          <>
                                            {" "}
                                            (
                                            {formatBytes(
                                              progressMeta.progressValue
                                                .downloadedBytes,
                                            )}{" "}
                                            /{" "}
                                            {formatBytes(
                                              progressMeta.progressValue
                                                .totalBytes,
                                            )}
                                            )
                                          </>
                                        )}
                                    </div>
                                  </div>
                                )}
                            </div>
                          );
                        })}
                      </div>
                    </>
                  )}
                </section>

                <section className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3">
                  <details>
                    <summary className="cursor-pointer text-sm text-white">
                      Advanced Speech Detection
                    </summary>
                    <div className="mt-3 grid md:grid-cols-3 gap-4">
                      <div>
                        <label className="text-xs text-[var(--text-muted)]">
                          VAD Sensitivity ({vadThreshold.toFixed(3)})
                        </label>
                        <Slider
                          aria-label="VAD sensitivity"
                          min={0.005}
                          max={0.08}
                          step={0.001}
                          value={[vadThreshold]}
                          onValueChange={(value) => {
                            const next = value[0];
                            if (typeof next === "number") {
                              setVadThreshold(next);
                            }
                          }}
                          className="mt-2"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-[var(--text-muted)]">
                          End Silence (ms): {silenceDurationMs}
                        </label>
                        <Slider
                          aria-label="End silence duration"
                          min={400}
                          max={1800}
                          step={50}
                          value={[silenceDurationMs]}
                          onValueChange={(value) => {
                            const next = value[0];
                            if (typeof next === "number") {
                              setSilenceDurationMs(next);
                            }
                          }}
                          className="mt-2"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-[var(--text-muted)]">
                          Minimum Speech (ms): {minSpeechMs}
                        </label>
                        <Slider
                          aria-label="Minimum speech duration"
                          min={150}
                          max={1200}
                          step={50}
                          value={[minSpeechMs]}
                          onValueChange={(value) => {
                            const next = value[0];
                            if (typeof next === "number") {
                              setMinSpeechMs(next);
                            }
                          }}
                          className="mt-2"
                        />
                      </div>
                    </div>
                  </details>
                </section>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <audio
        ref={audioRef}
        className="hidden"
        onEnded={() => {
          clearAudioPlayback();
          if (isSessionActiveRef.current && !processingRef.current) {
            setRuntimeStatus("listening");
          } else if (!isSessionActiveRef.current) {
            setRuntimeStatus("idle");
          }
        }}
      />
    </PageShell>
  );
}
