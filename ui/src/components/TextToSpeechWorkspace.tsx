import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertCircle,
  CheckCircle2,
  Download,
  Loader2,
  Mic2,
  Radio,
  RotateCcw,
  Settings2,
  Sparkles,
  Square,
  Waves,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  api,
  type ModelInfo,
  type SavedVoiceSummary,
  type SpeechHistoryRecord,
  type TTSGenerationStats,
} from "@/api";
import {
  TEXT_TO_SPEECH_PREFERRED_MODELS,
  VOICE_CLONING_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import type { VoicePickerItem } from "@/components/VoicePicker";
import { VoiceSelect } from "@/components/VoiceSelect";
import { RouteModelSelect } from "@/components/RouteModelSelect";
import { GenerationStats } from "@/components/GenerationStats";
import { TextToSpeechProjectsWorkspace } from "@/components/TextToSpeechProjectsWorkspace";
import {
  VOICE_ROUTE_BODY_COPY_CLASS,
  VOICE_ROUTE_META_COPY_CLASS,
  VOICE_ROUTE_PANEL_TITLE_CLASS,
  VOICE_ROUTE_SECTION_LABEL_CLASS,
  VOICE_ROUTE_WORKSPACE_DESCRIPTION_CLASS,
  VOICE_ROUTE_WORKSPACE_TITLE_CLASS,
} from "@/components/voiceRouteTypography";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SpeechHistoryPanel } from "@/components/SpeechHistoryPanel";
import { useDownloadIndicator } from "@/utils/useDownloadIndicator";
import { getSpeakerProfilesForVariant } from "@/types";

interface ModelOption {
  value: string;
  label: string;
  statusLabel: string;
  isReady: boolean;
}

interface TextToSpeechWorkspaceProps {
  selectedModel: string | null;
  selectedModelInfo: ModelInfo | null;
  selectedModelReady?: boolean;
  availableModels: ModelInfo[];
  modelOptions?: ModelOption[];
  onSelectModel?: (variant: string) => void;
  onOpenModelManager?: () => void;
  onModelRequired: () => void;
  onError: (message: string) => void;
  historyActionContainer?: HTMLElement | null;
  initialSavedVoiceId?: string | null;
  initialSpeaker?: string | null;
}

type VoiceMode = "saved" | "built_in";
type WorkspaceMode = "quick" | "projects";

const MAX_BUFFERED_PCM_BYTES = 256 * 1024 * 1024;
const ABORT_ERROR_NAME = "AbortError";
const SAVED_VOICE_RENDERER_PREFERRED_MODELS = [
  ...VOICE_CLONING_PREFERRED_MODELS,
] as const;

function createAbortError(message: string): Error {
  const error = new Error(message);
  error.name = ABORT_ERROR_NAME;
  return error;
}

function decodePcmI16Base64(base64Data: string): Int16Array {
  const binary = atob(base64Data);
  const sampleCount = Math.floor(binary.length / 2);
  const out = new Int16Array(sampleCount);

  for (let i = 0; i < sampleCount; i += 1) {
    const lo = binary.charCodeAt(i * 2);
    const hi = binary.charCodeAt(i * 2 + 1);
    let value = (hi << 8) | lo;
    if (value & 0x8000) {
      value -= 0x10000;
    }
    out[i] = value;
  }

  return out;
}

function pcmI16ToFloat32(samples: Int16Array): Float32Array<ArrayBuffer> {
  const floatSamples = new Float32Array(
    samples.length,
  ) as Float32Array<ArrayBuffer>;
  for (let i = 0; i < samples.length; i += 1) {
    floatSamples[i] = samples[i] / 0x8000;
  }
  return floatSamples;
}

function wavHeader(sampleRate: number, dataSize: number): Uint8Array {
  const bytesPerSample = 2;
  const buffer = new ArrayBuffer(44);
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
  view.setUint32(28, sampleRate * bytesPerSample, true);
  view.setUint16(32, bytesPerSample, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  return new Uint8Array(buffer, 0, 44);
}

function copyToArrayBuffer(view: Uint8Array): ArrayBuffer {
  const copied = new Uint8Array(view.byteLength);
  copied.set(view);
  return copied.buffer;
}

function encodeWavPcm16Chunks(
  sampleRate: number,
  pcmChunks: Uint8Array[],
  totalPcmBytes: number,
): Blob {
  const parts: BlobPart[] = [
    copyToArrayBuffer(wavHeader(sampleRate, totalPcmBytes)),
  ];
  for (const chunk of pcmChunks) {
    parts.push(copyToArrayBuffer(chunk));
  }

  return new Blob(parts, {
    type: "audio/wav",
  });
}

function revokeObjectUrlIfNeeded(url: string | null): void {
  if (url && url.startsWith("blob:")) {
    URL.revokeObjectURL(url);
  }
}

function mapRecordToStats(record: SpeechHistoryRecord): TTSGenerationStats {
  return {
    generation_time_ms: record.generation_time_ms,
    audio_duration_secs: record.audio_duration_secs ?? 0,
    rtf: record.rtf ?? 0,
    tokens_generated: record.tokens_generated ?? 0,
  };
}

function formatSavedVoiceDate(timestampMs: number): string {
  const value = new Date(timestampMs);
  if (Number.isNaN(value.getTime())) {
    return "Unknown date";
  }
  return value.toLocaleDateString([], {
    month: "short",
    day: "numeric",
  });
}

function savedVoiceSourceLabel(
  source: SavedVoiceSummary["source_route_kind"],
): string {
  switch (source) {
    case "voice_cloning":
      return "Cloned voice";
    case "voice_design":
      return "Designed voice";
    default:
      return "Saved voice";
  }
}

export function TextToSpeechWorkspace({
  selectedModel,
  selectedModelInfo,
  selectedModelReady = false,
  availableModels,
  modelOptions = [],
  onSelectModel,
  onOpenModelManager,
  onModelRequired,
  onError,
  historyActionContainer = null,
  initialSavedVoiceId,
  initialSpeaker,
}: TextToSpeechWorkspaceProps) {
  const [workspaceMode, setWorkspaceMode] = useState<WorkspaceMode>("quick");
  const [text, setText] = useState("");
  const [voiceMode, setVoiceMode] = useState<VoiceMode>(
    initialSavedVoiceId ? "saved" : "built_in",
  );
  const [speaker, setSpeaker] = useState(initialSpeaker || "Vivian");
  const [selectedSavedVoiceId, setSelectedSavedVoiceId] = useState(
    initialSavedVoiceId || "",
  );
  const [instructions, setInstructions] = useState("");
  const [speed, setSpeed] = useState(1);
  const [streamingEnabled, setStreamingEnabled] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [generationStats, setGenerationStats] =
    useState<TTSGenerationStats | null>(null);
  const [latestRecord, setLatestRecord] = useState<SpeechHistoryRecord | null>(
    null,
  );
  const [savedVoices, setSavedVoices] = useState<SavedVoiceSummary[]>([]);
  const [savedVoicesLoading, setSavedVoicesLoading] = useState(false);
  const [savedVoicesError, setSavedVoicesError] = useState<string | null>(null);
  const {
    downloadState,
    downloadMessage,
    isDownloading,
    beginDownload,
    completeDownload,
    failDownload,
  } = useDownloadIndicator();

  const audioRef = useRef<HTMLAudioElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const audioUrlRef = useRef<string | null>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const playbackContextRef = useRef<AudioContext | null>(null);
  const playbackSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const nextPlaybackTimeRef = useRef(0);
  const streamSampleRateRef = useRef(24000);
  const streamPcmChunksRef = useRef<Uint8Array[]>([]);
  const bufferedPcmBytesRef = useRef(0);
  const mergeSuppressedRef = useRef(false);
  const generationSessionRef = useRef(0);
  const appliedInitialSavedVoiceRef = useRef(false);
  const appliedInitialSpeakerRef = useRef(false);
  const alignedInitialModelIntentRef = useRef(false);

  const selectedOption = useMemo(() => {
    if (!selectedModel) {
      return null;
    }
    return (
      modelOptions.find((option) => option.value === selectedModel) || null
    );
  }, [selectedModel, modelOptions]);

  const capabilities = selectedModelInfo?.speech_capabilities ?? null;
  const supportsBuiltInVoices = capabilities?.supports_builtin_voices ?? false;
  const supportsReferenceVoices =
    capabilities?.supports_reference_voice ?? false;
  const supportsVoiceDescription =
    capabilities?.supports_voice_description ?? false;
  const supportsStreaming = capabilities?.supports_streaming ?? false;

  const availableSpeakers = useMemo(
    () =>
      supportsBuiltInVoices ? getSpeakerProfilesForVariant(selectedModel) : [],
    [selectedModel, supportsBuiltInVoices],
  );
  const defaultSpeaker = availableSpeakers[0]?.id ?? "Vivian";
  const selectedSavedVoice = useMemo(
    () =>
      savedVoices.find((voice) => voice.id === selectedSavedVoiceId) ?? null,
    [savedVoices, selectedSavedVoiceId],
  );

  useEffect(() => {
    if (!availableSpeakers.some((candidate) => candidate.id === speaker)) {
      setSpeaker(defaultSpeaker);
    }
  }, [availableSpeakers, defaultSpeaker, speaker]);

  useEffect(() => {
    if (!supportsStreaming && streamingEnabled) {
      setStreamingEnabled(false);
    }
  }, [streamingEnabled, supportsStreaming]);

  useEffect(() => {
    if (appliedInitialSavedVoiceRef.current || !initialSavedVoiceId) {
      return;
    }
    setVoiceMode("saved");
    setSelectedSavedVoiceId(initialSavedVoiceId);
    appliedInitialSavedVoiceRef.current = true;
  }, [initialSavedVoiceId]);

  useEffect(() => {
    if (appliedInitialSpeakerRef.current || !initialSpeaker) {
      return;
    }
    setVoiceMode("built_in");
    setSpeaker(initialSpeaker);
    appliedInitialSpeakerRef.current = true;
  }, [initialSpeaker]);

  useEffect(() => {
    if (alignedInitialModelIntentRef.current) {
      return;
    }

    if (initialSavedVoiceId && !supportsReferenceVoices) {
      const nextModel = resolvePreferredRouteModel({
        models: availableModels.filter(
          (model) => model.speech_capabilities?.supports_reference_voice,
        ),
        selectedModel,
        preferredVariants: SAVED_VOICE_RENDERER_PREFERRED_MODELS,
      });
      if (nextModel && nextModel !== selectedModel) {
        onSelectModel?.(nextModel);
        return;
      }
    }

    if (initialSpeaker && !supportsBuiltInVoices) {
      const nextModel = resolvePreferredRouteModel({
        models: availableModels.filter(
          (model) => model.speech_capabilities?.supports_builtin_voices,
        ),
        selectedModel,
        preferredVariants: TEXT_TO_SPEECH_PREFERRED_MODELS,
      });
      if (nextModel && nextModel !== selectedModel) {
        onSelectModel?.(nextModel);
        return;
      }
    }

    alignedInitialModelIntentRef.current = true;
  }, [
    availableModels,
    initialSavedVoiceId,
    initialSpeaker,
    onSelectModel,
    selectedModel,
    supportsBuiltInVoices,
    supportsReferenceVoices,
  ]);

  useEffect(() => {
    if (voiceMode === "saved" && !supportsReferenceVoices && supportsBuiltInVoices) {
      setVoiceMode("built_in");
      return;
    }

    if (voiceMode === "built_in" && !supportsBuiltInVoices && supportsReferenceVoices) {
      setVoiceMode("saved");
    }
  }, [
    supportsBuiltInVoices,
    supportsReferenceVoices,
    voiceMode,
  ]);

  useEffect(() => {
    if (
      voiceMode === "saved" &&
      supportsReferenceVoices &&
      !selectedSavedVoiceId &&
      savedVoices.length > 0
    ) {
      setSelectedSavedVoiceId(savedVoices[0]?.id ?? "");
    }
  }, [
    savedVoices,
    selectedSavedVoiceId,
    supportsReferenceVoices,
    voiceMode,
  ]);

  const replaceAudioUrl = useCallback((nextUrl: string | null) => {
    revokeObjectUrlIfNeeded(audioUrlRef.current);
    audioUrlRef.current = nextUrl;
    setAudioUrl(nextUrl);
  }, []);

  const reportError = useCallback(
    (message: string) => {
      setError(message);
      onError(message);
    },
    [onError],
  );

  const stopStreamingSession = useCallback(() => {
    generationSessionRef.current += 1;
    if (streamAbortRef.current) {
      streamAbortRef.current.abort();
      streamAbortRef.current = null;
    }

    for (const source of playbackSourcesRef.current) {
      try {
        source.stop();
      } catch {
        // Ignore already-stopped sources.
      }
    }
    playbackSourcesRef.current.clear();

    if (playbackContextRef.current) {
      playbackContextRef.current.close().catch(() => {});
      playbackContextRef.current = null;
    }

    nextPlaybackTimeRef.current = 0;
    streamPcmChunksRef.current = [];
    bufferedPcmBytesRef.current = 0;
    mergeSuppressedRef.current = false;
  }, []);

  useEffect(() => {
    return () => {
      stopStreamingSession();
      revokeObjectUrlIfNeeded(audioUrlRef.current);
      audioUrlRef.current = null;
    };
  }, [stopStreamingSession]);

  const loadSavedVoices = useCallback(async () => {
    setSavedVoicesLoading(true);
    setSavedVoicesError(null);
    try {
      const records = await api.listSavedVoices();
      setSavedVoices(records);
    } catch (err) {
      setSavedVoicesError(
        err instanceof Error ? err.message : "Failed to load saved voices.",
      );
    } finally {
      setSavedVoicesLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadSavedVoices();
  }, [loadSavedVoices]);

  const compatibilityNotice = useMemo(() => {
    if (!selectedModelInfo?.variant) {
      return "Choose a model to see the voice options it supports on this route.";
    }

    if (voiceMode === "saved") {
      if (!supportsReferenceVoices) {
        return `${selectedModelInfo.variant} does not support reusable saved voices. Pick a renderer with saved-voice support.`;
      }
      if (selectedSavedVoice) {
        return `Rendering with reusable voice "${selectedSavedVoice.name}".`;
      }
      if (savedVoicesLoading) {
        return "Loading your saved voices for this model.";
      }
      return "Choose a saved voice to reuse an existing cloned or designed profile.";
    }

    if (!supportsBuiltInVoices) {
      return `${selectedModelInfo.variant} does not expose built-in speakers on this route.`;
    }

    if (availableSpeakers.length === 0) {
      return `No built-in speakers are currently mapped for ${selectedModelInfo.variant}.`;
    }

    return `Using built-in voice "${speaker}" on ${selectedModelInfo.variant}.`;
  }, [
    availableSpeakers.length,
    savedVoicesLoading,
    selectedModelInfo?.variant,
    selectedSavedVoice,
    speaker,
    supportsBuiltInVoices,
    supportsReferenceVoices,
    voiceMode,
  ]);

  const voiceAvailabilitySummary = useMemo(() => {
    if (!selectedModelInfo?.variant) {
      return "Choose a model";
    }

    if (supportsBuiltInVoices && supportsReferenceVoices) {
      return `${availableSpeakers.length} built-in voices plus saved voices`;
    }

    if (supportsBuiltInVoices) {
      return availableSpeakers.length > 0
        ? `${availableSpeakers.length} built-in voices`
        : "Built-in voices unavailable";
    }

    if (supportsReferenceVoices) {
      return "Saved voices only";
    }

    return "No voices on this route";
  }, [
    availableSpeakers.length,
    selectedModelInfo?.variant,
    supportsBuiltInVoices,
    supportsReferenceVoices,
  ]);

  const savedVoiceItems: VoicePickerItem[] = savedVoices.map((voice) => ({
    id: voice.id,
    name: voice.name,
    categoryLabel: savedVoiceSourceLabel(voice.source_route_kind),
    description: voice.reference_text_preview,
    meta: [
      `${voice.reference_text_chars} chars`,
      formatSavedVoiceDate(voice.updated_at || voice.created_at),
    ],
    previewUrl: api.savedVoiceAudioUrl(voice.id),
    selected: voiceMode === "saved" && selectedSavedVoiceId === voice.id,
    onSelect: () => {
      setVoiceMode("saved");
      setSelectedSavedVoiceId(voice.id);
    },
  }));

  const builtInVoiceItems: VoicePickerItem[] = availableSpeakers.map((voice) => ({
    id: voice.id,
    name: voice.name,
    categoryLabel: selectedModelInfo?.variant ?? "Built-in voice",
    description: voice.description,
    meta: [voice.language],
    selected: voiceMode === "built_in" && speaker === voice.id,
    onSelect: () => {
      setVoiceMode("built_in");
      setSpeaker(voice.id);
    },
  }));

  const selectedVoiceItem = useMemo(() => {
    if (voiceMode === "saved") {
      return (
        savedVoiceItems.find((item) => item.id === selectedSavedVoiceId) ?? null
      );
    }
    return builtInVoiceItems.find((item) => item.id === speaker) ?? null;
  }, [
    builtInVoiceItems,
    savedVoiceItems,
    selectedSavedVoiceId,
    speaker,
    voiceMode,
  ]);

  const canGenerate =
    selectedModelReady &&
    (voiceMode !== "saved" || Boolean(selectedSavedVoiceId)) &&
    (voiceMode !== "built_in" || Boolean(speaker));

  const handleGenerate = async () => {
    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      return;
    }

    if (!text.trim()) {
      reportError("Please enter some text");
      return;
    }

    if (voiceMode === "saved" && !selectedSavedVoiceId) {
      reportError("Choose a saved voice before generating audio.");
      return;
    }

    if (voiceMode === "built_in" && !speaker) {
      reportError("Choose a built-in voice before generating audio.");
      return;
    }

    const trimmedText = text.trim();

    try {
      setGenerating(true);
      setIsStreaming(false);
      setError(null);
      setGenerationStats(null);
      stopStreamingSession();
      replaceAudioUrl(null);
      const generationSession = generationSessionRef.current;

      const requestBase = {
        model_id: selectedModel,
        max_tokens: 0,
        speed,
        speaker: voiceMode === "built_in" ? speaker : undefined,
        saved_voice_id:
          voiceMode === "saved" ? selectedSavedVoiceId : undefined,
        voice_description:
          supportsVoiceDescription && instructions.trim()
            ? instructions.trim()
            : undefined,
      };

      if (!streamingEnabled || !supportsStreaming) {
        const record = await api.createTextToSpeechRecord({
          ...requestBase,
          text: trimmedText,
        });
        replaceAudioUrl(api.textToSpeechRecordAudioUrl(record.id));
        setGenerationStats(mapRecordToStats(record));
        setLatestRecord(record);

        setTimeout(() => {
          audioRef.current?.play().catch(() => {});
        }, 100);

        setGenerating(false);
        return;
      }

      const audioContext = new AudioContext();
      playbackContextRef.current = audioContext;
      nextPlaybackTimeRef.current = audioContext.currentTime + 0.05;
      streamSampleRateRef.current = 24000;
      streamPcmChunksRef.current = [];
      bufferedPcmBytesRef.current = 0;
      mergeSuppressedRef.current = false;
      setIsStreaming(true);

      const appendPcmChunk = (pcmSamples: Int16Array) => {
        if (mergeSuppressedRef.current) {
          return;
        }
        const pcmBytes = new Uint8Array(pcmSamples.buffer);
        const nextTotal = bufferedPcmBytesRef.current + pcmBytes.byteLength;
        if (nextTotal > MAX_BUFFERED_PCM_BYTES) {
          mergeSuppressedRef.current = true;
          streamPcmChunksRef.current = [];
          bufferedPcmBytesRef.current = 0;
          return;
        }
        streamPcmChunksRef.current.push(pcmBytes);
        bufferedPcmBytesRef.current = nextTotal;
      };

      const schedulePlayback = (pcmSamples: Int16Array) => {
        const context = playbackContextRef.current;
        if (!context) {
          return;
        }

        const floatSamples = pcmI16ToFloat32(pcmSamples);
        const buffer = context.createBuffer(
          1,
          floatSamples.length,
          streamSampleRateRef.current,
        );
        buffer.copyToChannel(floatSamples, 0);

        const source = context.createBufferSource();
        source.buffer = buffer;
        source.connect(context.destination);

        const scheduledAt = Math.max(
          context.currentTime + 0.02,
          nextPlaybackTimeRef.current,
        );
        source.start(scheduledAt);
        nextPlaybackTimeRef.current = scheduledAt + buffer.duration;

        playbackSourcesRef.current.add(source);
        source.onended = () => {
          playbackSourcesRef.current.delete(source);
        };

        if (context.state === "suspended") {
          context.resume().catch(() => {});
        }
      };

      const finalRecordRef = { current: null as SpeechHistoryRecord | null };
      const finalStatsRef = {
        current: null as TTSGenerationStats | null,
      };

      const streamRequest = (): Promise<void> =>
        new Promise((resolve, reject) => {
          let settled = false;
          const resolveOnce = () => {
            if (settled) return;
            settled = true;
            resolve();
          };
          const rejectOnce = (streamError: Error) => {
            if (settled) return;
            settled = true;
            reject(streamError);
          };

          streamAbortRef.current = api.createTextToSpeechRecordStream(
            {
              ...requestBase,
              text: trimmedText,
            },
            {
              onStart: ({ sampleRate, audioFormat }) => {
                if (generationSession !== generationSessionRef.current) {
                  return;
                }
                streamSampleRateRef.current = sampleRate;
                if (audioFormat !== "pcm_i16") {
                  const message = `Unsupported streamed audio format '${audioFormat}'. Expected pcm_i16.`;
                  reportError(message);
                  streamAbortRef.current?.abort();
                  rejectOnce(new Error(message));
                }
              },
              onChunk: ({ audioBase64 }) => {
                if (generationSession !== generationSessionRef.current) {
                  return;
                }
                const pcmSamples = decodePcmI16Base64(audioBase64);
                if (pcmSamples.length === 0) {
                  return;
                }
                appendPcmChunk(pcmSamples);
                schedulePlayback(pcmSamples);
              },
              onFinal: ({ record, stats }) => {
                if (generationSession !== generationSessionRef.current) {
                  return;
                }
                finalRecordRef.current = record;
                finalStatsRef.current = stats;
              },
              onError: (errorMessage) => {
                if (generationSession !== generationSessionRef.current) {
                  return;
                }
                reportError(errorMessage);
                rejectOnce(new Error(errorMessage));
              },
              onDone: () => {
                streamAbortRef.current = null;
                resolveOnce();
              },
            },
          );
        });

      await streamRequest();

      if (generationSession !== generationSessionRef.current) {
        throw createAbortError("Generation cancelled");
      }

      if (finalStatsRef.current) {
        setGenerationStats(finalStatsRef.current);
      }
      const finalRecord = finalRecordRef.current;
      if (finalRecord) {
        setLatestRecord(finalRecord);
      }

      if (
        !mergeSuppressedRef.current &&
        bufferedPcmBytesRef.current > 0 &&
        streamPcmChunksRef.current.length > 0
      ) {
        const wavBlob = encodeWavPcm16Chunks(
          streamSampleRateRef.current,
          streamPcmChunksRef.current,
          bufferedPcmBytesRef.current,
        );
        replaceAudioUrl(URL.createObjectURL(wavBlob));
      } else if (finalRecord?.id) {
        replaceAudioUrl(api.textToSpeechRecordAudioUrl(finalRecord.id));
      }

      setIsStreaming(false);
      setGenerating(false);
    } catch (err) {
      if ((err as Error).name === ABORT_ERROR_NAME) {
        setGenerating(false);
        setIsStreaming(false);
        return;
      }
      reportError(err instanceof Error ? err.message : "Generation failed");
      setGenerating(false);
      setIsStreaming(false);
    }
  };

  const handleStop = () => {
    stopStreamingSession();
    setGenerating(false);
    setIsStreaming(false);

    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
  };

  const handleDownload = async () => {
    const record = latestRecord;
    const localAudioUrl = !record ? audioUrl : null;
    if ((!record && !localAudioUrl) || isDownloading) {
      return;
    }

    beginDownload();
    try {
      if (record) {
        const downloadUrl = api.textToSpeechRecordAudioUrl(record.id, {
          download: true,
        });
        const filename =
          record.audio_filename ||
          `izwi-tts-${voiceMode === "built_in" ? speaker.toLowerCase() : "saved-voice"}-${Date.now()}.wav`;
        await api.downloadAudioFile(downloadUrl, filename);
        completeDownload();
        return;
      }

      if (!localAudioUrl) {
        return;
      }
      await api.downloadAudioFile(localAudioUrl, `izwi-tts-${Date.now()}.wav`);
      completeDownload();
    } catch (downloadError) {
      failDownload(downloadError);
    }
  };

  const handleReset = () => {
    stopStreamingSession();
    setText("");
    setInstructions("");
    setSpeed(1);
    setError(null);
    setGenerationStats(null);
    setGenerating(false);
    setIsStreaming(false);
    replaceAudioUrl(null);
    textareaRef.current?.focus();
  };

  const renderModelSelector = () => (
    <RouteModelSelect
      value={selectedModel}
      options={modelOptions}
      onSelect={onSelectModel}
      className="w-full"
    />
  );

  const renderWorkflowTabs = () => (
    <Tabs
      value={workspaceMode}
      onValueChange={(value) => setWorkspaceMode(value as WorkspaceMode)}
      className="w-full max-w-sm"
    >
      <TabsList className="grid w-full grid-cols-2 border-[var(--border-strong)] bg-[var(--bg-surface-2)] p-1 shadow-sm">
        <TabsTrigger
          value="quick"
          className="text-[var(--text-muted)] data-[state=active]:bg-[var(--accent-solid)] data-[state=active]:text-[var(--text-on-accent)] data-[state=active]:shadow-[0_8px_20px_-14px_rgba(17,17,17,0.55)]"
        >
          Quick
        </TabsTrigger>
        <TabsTrigger
          value="projects"
          className="text-[var(--text-muted)] data-[state=active]:bg-[var(--accent-solid)] data-[state=active]:text-[var(--text-on-accent)] data-[state=active]:shadow-[0_8px_20px_-14px_rgba(17,17,17,0.55)]"
        >
          Projects
        </TabsTrigger>
      </TabsList>
    </Tabs>
  );

  if (workspaceMode === "projects") {
    return (
      <div className="grid gap-4 items-stretch xl:h-[calc(100dvh-11.75rem)]">
        <div className="card p-4 flex min-h-0 flex-col">
          <div className="mb-4 flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="rounded bg-[var(--bg-surface-2)] border border-[var(--border-muted)] p-2">
                <Waves className="h-5 w-5 text-[var(--text-muted)]" />
              </div>
              <div>
                <h2 className={VOICE_ROUTE_WORKSPACE_TITLE_CLASS}>
                  TTS Projects
                </h2>
                <p className={VOICE_ROUTE_WORKSPACE_DESCRIPTION_CLASS}>
                  Import scripts, render segments, and export merged narration.
                </p>
              </div>
            </div>

            {renderWorkflowTabs()}
          </div>

          <div className="flex-1 min-h-0 overflow-y-auto pr-1 scrollbar-thin">
            <TextToSpeechProjectsWorkspace
              selectedModel={selectedModel}
              selectedModelInfo={selectedModelInfo}
              availableModels={availableModels}
              modelOptions={modelOptions}
              headerActionContainer={historyActionContainer}
              onSelectModel={onSelectModel}
              onOpenModelManager={onOpenModelManager}
              onModelRequired={onModelRequired}
              onError={onError}
            />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="grid gap-4 items-stretch xl:h-[calc(100dvh-11.75rem)]">
      <div className="card p-4 flex min-h-0 flex-col">
        <div className="flex-1 min-h-0 overflow-y-auto pr-1 scrollbar-thin">
          <div className="mb-4 flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="rounded bg-[var(--bg-surface-2)] border border-[var(--border-muted)] p-2">
                <Mic2 className="h-5 w-5 text-[var(--text-muted)]" />
              </div>
              <div>
                <h2 className={VOICE_ROUTE_WORKSPACE_TITLE_CLASS}>
                  Text to Speech
                </h2>
                <p className={VOICE_ROUTE_WORKSPACE_DESCRIPTION_CLASS}>
                  Choose a model, then a compatible voice, then render.
                </p>
              </div>
            </div>

            {renderWorkflowTabs()}
          </div>

          <div className="mb-4 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
            <div className="grid gap-x-5 gap-y-4 xl:grid-cols-[minmax(0,360px)_minmax(0,360px)_minmax(0,1fr)_auto]">
              <div className="min-w-0">
                <div className={VOICE_ROUTE_SECTION_LABEL_CLASS}>
                  Active Model
                </div>
                <div className="mt-3 w-full max-w-[360px]">{renderModelSelector()}</div>
                <div
                  className={cn(
                    "mt-2 text-xs",
                    selectedModelReady
                      ? "text-[var(--text-secondary)]"
                      : "text-amber-500",
                  )}
                >
                  {selectedOption?.statusLabel ||
                    "Select and load a TTS model to continue"}
                </div>
              </div>

              <div className="min-w-0">
                <div className={VOICE_ROUTE_SECTION_LABEL_CLASS}>Voice</div>
                <div className="mt-3 w-full max-w-[360px]">
                  <VoiceSelect
                    voiceMode={voiceMode}
                    onVoiceModeChange={setVoiceMode}
                    savedVoiceItems={savedVoiceItems}
                    builtInVoiceItems={builtInVoiceItems}
                    selectedItem={selectedVoiceItem}
                    savedVoicesLoading={savedVoicesLoading}
                    savedVoicesError={savedVoicesError}
                    savedEnabled={supportsReferenceVoices}
                    builtInEnabled={supportsBuiltInVoices}
                    disabled={!selectedModel}
                    modelLabel={selectedModelInfo?.variant ?? selectedModel}
                    compact
                  />
                </div>
                <p className="mt-2 text-[11px] font-medium text-[var(--text-muted)]">
                  Voice availability follows the selected model instead of
                  switching models automatically.
                </p>
              </div>

              {onOpenModelManager ? (
                <div className="xl:col-start-4 xl:row-start-1 xl:mt-[1.8rem] xl:justify-self-end">
                  <Button
                    variant="outline"
                    onClick={onOpenModelManager}
                    className="h-11 shrink-0 rounded-xl px-4"
                  >
                    <Settings2 className="h-4 w-4" />
                    Models
                  </Button>
                </div>
              ) : null}
            </div>
          </div>

          <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_420px]">
            <div className="space-y-6">
              <div>
                <label className={cn(VOICE_ROUTE_SECTION_LABEL_CLASS, "mb-2 block")}>
                  Script
                </label>
                <textarea
                  ref={textareaRef}
                  value={text}
                  onChange={(event) => setText(event.target.value)}
                  rows={8}
                  placeholder="Paste the text you want this voice to speak..."
                  className="textarea min-h-[220px] w-full bg-[var(--bg-surface-1)] border-[var(--border-muted)] py-4 text-base leading-relaxed"
                />
              </div>

              <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
                <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                  <div>
                    <div className={cn(VOICE_ROUTE_SECTION_LABEL_CLASS, "mb-2")}>
                      Delivery Controls
                    </div>
                    <div className={VOICE_ROUTE_BODY_COPY_CLASS}>
                      Speed is saved with the generation history. Streaming
                      appears only when the selected model exposes it.
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowAdvanced((current) => !current)}
                  >
                    <Sparkles className="h-4 w-4" />
                    {showAdvanced ? "Hide" : "Show"} advanced
                  </Button>
                </div>

                <div className="mt-4 grid gap-4 sm:grid-cols-[minmax(0,1fr)_320px]">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium text-[var(--text-primary)]">
                        Speed
                      </span>
                      <span className="text-[var(--text-muted)]">
                        {speed.toFixed(2)}x
                      </span>
                    </div>
                    <Slider
                      value={[speed]}
                      min={0.5}
                      max={1.5}
                      step={0.05}
                      onValueChange={([value]) => setSpeed(value ?? 1)}
                    />
                  </div>

                  <label className="flex items-center gap-3 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3.5 py-3 text-sm">
                    <input
                      type="checkbox"
                      checked={streamingEnabled && supportsStreaming}
                      onChange={(event) =>
                        setStreamingEnabled(event.target.checked)
                      }
                      disabled={!supportsStreaming}
                      className="h-4 w-4 rounded border-border"
                    />
                    <span className="min-w-0 flex-1">
                      <span className="font-medium text-[var(--text-primary)]">
                        Stream audio
                      </span>
                      <span className="block text-xs text-[var(--text-muted)]">
                        {supportsStreaming
                          ? "Play chunks as they arrive."
                          : "Current model does not expose streaming on this route."}
                      </span>
                    </span>
                    <Radio className="h-4 w-4 text-[var(--text-muted)]" />
                  </label>
                </div>

                <AnimatePresence>
                  {showAdvanced ? (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      className="overflow-hidden"
                    >
                      <div className="space-y-2 pt-4">
                        <label className={VOICE_ROUTE_SECTION_LABEL_CLASS}>
                          Style prompt
                        </label>
                        <Input
                          value={instructions}
                          onChange={(event) => setInstructions(event.target.value)}
                          disabled={!supportsVoiceDescription}
                          placeholder={
                            supportsVoiceDescription
                              ? "Optional style guidance such as calm, energetic, or formal"
                              : "This renderer does not support style prompts"
                          }
                        />
                      </div>
                    </motion.div>
                  ) : null}
                </AnimatePresence>
              </div>

              <AnimatePresence>
                {error ? (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="flex items-start gap-2 rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] p-3 text-sm text-[var(--danger-text)]">
                      <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                      <p>{error}</p>
                    </div>
                  </motion.div>
                ) : null}
              </AnimatePresence>

              <div className="flex flex-wrap items-center gap-3">
                <Button
                  onClick={handleGenerate}
                  disabled={generating || !canGenerate}
                  className="min-w-[190px]"
                >
                  {generating ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <Mic2 className="h-4 w-4" />
                      Generate audio
                    </>
                  )}
                </Button>

                {(audioUrl || isStreaming) ? (
                  <Button variant="outline" onClick={handleStop}>
                    <Square className="h-4 w-4" />
                    Stop
                  </Button>
                ) : null}

                {audioUrl ? (
                  <>
                    <Button
                      variant="outline"
                      onClick={handleDownload}
                      disabled={isDownloading}
                    >
                      {isDownloading ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Download className="h-4 w-4" />
                      )}
                      Download
                    </Button>
                    <Button variant="ghost" onClick={handleReset}>
                      <RotateCcw className="h-4 w-4" />
                      Reset
                    </Button>
                  </>
                ) : null}
              </div>

              {downloadState !== "idle" && downloadMessage ? (
                <div
                  className={cn(
                    "flex items-center gap-2 rounded-lg border px-3 py-2.5 text-xs font-medium",
                    downloadState === "downloading" &&
                      "bg-[var(--status-warning-bg)] border-[var(--status-warning-border)] text-[var(--status-warning-text)]",
                    downloadState === "success" &&
                      "bg-[var(--status-positive-bg)] border-[var(--status-positive-border)] text-[var(--status-positive-text)]",
                    downloadState === "error" &&
                      "bg-[var(--danger-bg)] border-[var(--danger-border)] text-[var(--danger-text)]",
                  )}
                >
                  {downloadState === "downloading" ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : downloadState === "success" ? (
                    <CheckCircle2 className="h-4 w-4" />
                  ) : (
                    <AlertCircle className="h-4 w-4" />
                  )}
                  {downloadMessage}
                </div>
              ) : null}
            </div>

            <div className="space-y-4">
              <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
                <div className={cn(VOICE_ROUTE_SECTION_LABEL_CLASS, "mb-3")}>
                  Voice Summary
                </div>
                <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-4">
                  <div className={VOICE_ROUTE_PANEL_TITLE_CLASS}>
                    {selectedVoiceItem?.name || "Select a voice"}
                  </div>
                  <div className={cn(VOICE_ROUTE_META_COPY_CLASS, "mt-1")}>
                    {voiceMode === "saved" ? "Saved voice" : "Built-in voice"}
                  </div>
                  <p className={cn(VOICE_ROUTE_BODY_COPY_CLASS, "mt-3")}>
                    {compatibilityNotice}
                  </p>
                  <div className="mt-3 grid gap-2 sm:grid-cols-2 text-xs">
                    <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2 text-[var(--text-secondary)]">
                      {selectedOption?.label || "No model selected"}
                    </div>
                    <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2 text-[var(--text-secondary)]">
                      {voiceAvailabilitySummary}
                    </div>
                  </div>
                </div>
              </div>

              <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
                <div className={cn(VOICE_ROUTE_SECTION_LABEL_CLASS, "mb-3")}>
                  Output
                </div>

                {!audioUrl && !isStreaming ? (
                  <div className="rounded-lg border border-dashed border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-8 text-center text-sm text-[var(--text-muted)]">
                    Generate audio to review the rendered result here.
                  </div>
                ) : (
                  <div className="space-y-4 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-4">
                    {audioUrl ? (
                      <audio
                        ref={audioRef}
                        src={audioUrl}
                        className="h-11 w-full"
                        controls
                      />
                    ) : null}
                    {isStreaming && !audioUrl ? (
                      <div className="rounded-lg border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] px-3 py-2 text-xs font-medium text-[var(--status-warning-text)]">
                        Streaming live audio...
                      </div>
                    ) : null}
                    {generationStats ? (
                      <GenerationStats stats={generationStats} type="tts" />
                    ) : null}
                  </div>
                )}
              </div>

              <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
                <div className={cn(VOICE_ROUTE_SECTION_LABEL_CLASS, "mb-2")}>
                  Quick Workflow
                </div>
                <div className="grid gap-2 text-xs">
                  <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-2 text-[var(--text-secondary)]">
                    1. Choose a TTS model
                  </div>
                  <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-2 text-[var(--text-secondary)]">
                    2. Pick a compatible voice
                  </div>
                  <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-2 text-[var(--text-secondary)]">
                    3. Generate and review the result
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      {workspaceMode === "quick" ? (
        <SpeechHistoryPanel
          route="text-to-speech"
          title="Text to Speech History"
          emptyMessage="No saved text-to-speech generations yet."
          latestRecord={latestRecord}
          historyActionContainer={historyActionContainer}
        />
      ) : null}
    </div>
  );
}
