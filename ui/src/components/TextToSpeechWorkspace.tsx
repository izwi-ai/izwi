import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AlertCircle,
  CheckCircle2,
  ChevronDown,
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
import { VoicePicker, type VoicePickerItem } from "@/components/VoicePicker";
import { GenerationStats } from "@/components/GenerationStats";
import { TextToSpeechProjectsWorkspace } from "@/components/TextToSpeechProjectsWorkspace";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
  "LFM2.5-Audio-1.5B-4bit",
  "LFM2.5-Audio-1.5B",
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
  const [voiceSearch, setVoiceSearch] = useState("");
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
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);
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
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const appliedInitialSavedVoiceRef = useRef(false);
  const appliedInitialSpeakerRef = useRef(false);

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

  const builtInCompatibleModels = useMemo(
    () =>
      availableModels.filter(
        (model) => model.speech_capabilities?.supports_builtin_voices,
      ),
    [availableModels],
  );
  const savedVoiceCompatibleModels = useMemo(
    () =>
      availableModels.filter(
        (model) => model.speech_capabilities?.supports_reference_voice,
      ),
    [availableModels],
  );

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
    const onPointerDown = (event: MouseEvent) => {
      if (
        modelMenuRef.current &&
        event.target instanceof Node &&
        !modelMenuRef.current.contains(event.target)
      ) {
        setIsModelMenuOpen(false);
      }
    };
    window.addEventListener("mousedown", onPointerDown);
    return () => window.removeEventListener("mousedown", onPointerDown);
  }, []);

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

  useEffect(() => {
    if (
      voiceMode === "saved" &&
      selectedSavedVoiceId &&
      !supportsReferenceVoices
    ) {
      const nextModel = resolvePreferredRouteModel({
        models: savedVoiceCompatibleModels,
        selectedModel,
        preferredVariants: SAVED_VOICE_RENDERER_PREFERRED_MODELS,
      });
      if (nextModel && nextModel !== selectedModel) {
        onSelectModel?.(nextModel);
      }
    }
  }, [
    onSelectModel,
    savedVoiceCompatibleModels,
    selectedModel,
    selectedSavedVoiceId,
    supportsReferenceVoices,
    voiceMode,
  ]);

  useEffect(() => {
    if (voiceMode === "built_in" && !supportsBuiltInVoices) {
      const nextModel = resolvePreferredRouteModel({
        models: builtInCompatibleModels,
        selectedModel,
        preferredVariants: TEXT_TO_SPEECH_PREFERRED_MODELS,
      });
      if (nextModel && nextModel !== selectedModel) {
        onSelectModel?.(nextModel);
      }
    }
  }, [
    builtInCompatibleModels,
    onSelectModel,
    selectedModel,
    supportsBuiltInVoices,
    voiceMode,
  ]);

  const compatibilityNotice = useMemo(() => {
    if (voiceMode === "saved") {
      if (selectedSavedVoiceId && !supportsReferenceVoices) {
        if (savedVoiceCompatibleModels.length === 0) {
          return "Load a Qwen Base or LFM2 Audio model to render saved voices.";
        }
        return "Switching to a saved-voice compatible model.";
      }
      if (selectedSavedVoice) {
        return `Rendering with reusable voice "${selectedSavedVoice.name}".`;
      }
      return "Choose a saved voice to reuse an existing cloned or designed profile.";
    }

    if (!supportsBuiltInVoices) {
      if (builtInCompatibleModels.length === 0) {
        return "Load a CustomVoice, Kokoro, or LFM2 Audio model to use built-in voices.";
      }
      return "Switching to a built-in voice model.";
    }

    return `Using built-in voice "${speaker}".`;
  }, [
    builtInCompatibleModels.length,
    savedVoiceCompatibleModels.length,
    selectedSavedVoice,
    selectedSavedVoiceId,
    speaker,
    supportsBuiltInVoices,
    supportsReferenceVoices,
    voiceMode,
  ]);

  const filteredSavedVoices = useMemo(() => {
    const normalizedQuery = voiceSearch.trim().toLowerCase();
    return savedVoices.filter((voice) => {
      if (!normalizedQuery) {
        return true;
      }
      return (
        voice.name.toLowerCase().includes(normalizedQuery) ||
        voice.reference_text_preview.toLowerCase().includes(normalizedQuery)
      );
    });
  }, [savedVoices, voiceSearch]);

  const filteredBuiltInVoices = useMemo(() => {
    const normalizedQuery = voiceSearch.trim().toLowerCase();
    return availableSpeakers.filter((voice) => {
      if (!normalizedQuery) {
        return true;
      }
      return (
        voice.name.toLowerCase().includes(normalizedQuery) ||
        voice.language.toLowerCase().includes(normalizedQuery) ||
        voice.description.toLowerCase().includes(normalizedQuery)
      );
    });
  }, [availableSpeakers, voiceSearch]);

  const savedVoiceItems: VoicePickerItem[] = filteredSavedVoices.map(
    (voice) => ({
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
    }),
  );

  const builtInVoiceItems: VoicePickerItem[] = filteredBuiltInVoices.map(
    (voice) => ({
      id: voice.id,
      name: voice.name,
      categoryLabel: selectedModelInfo?.variant ?? "Built-in voice",
      description: voice.description,
      meta: [voice.language],
      previewMessage: "Select this voice, then generate speech to audition it.",
      selected: voiceMode === "built_in" && speaker === voice.id,
      onSelect: () => {
        setVoiceMode("built_in");
        setSpeaker(voice.id);
      },
    }),
  );

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
    <div className="relative inline-block w-full" ref={modelMenuRef}>
      <button
        onClick={() => setIsModelMenuOpen((prev) => !prev)}
        className={cn(
          "flex h-10 w-full items-center justify-between gap-2 rounded-lg border px-3 text-sm transition-colors",
          selectedOption?.isReady
            ? "border-border bg-background/75 text-foreground"
            : "border-border/80 bg-muted/35 text-muted-foreground hover:border-border",
        )}
      >
        <span className="min-w-0 flex-1 truncate text-left">
          {selectedOption?.label || "Select model"}
        </span>
        <ChevronDown
          className={cn(
            "h-4 w-4 transition-transform",
            isModelMenuOpen && "rotate-180",
          )}
        />
      </button>

      <AnimatePresence>
        {isModelMenuOpen ? (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            transition={{ duration: 0.15 }}
            className="absolute left-0 right-0 top-full z-50 mt-2 rounded-xl border border-border bg-popover p-1.5 shadow-xl"
          >
            <div className="max-h-64 space-y-1 overflow-y-auto pr-1">
              {modelOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => {
                    onSelectModel?.(option.value);
                    setIsModelMenuOpen(false);
                  }}
                  className={cn(
                    "w-full rounded-lg px-3 py-2 text-left transition-colors",
                    selectedOption?.value === option.value
                      ? "bg-accent text-accent-foreground"
                      : "hover:bg-accent/70",
                  )}
                >
                  <div className="truncate text-sm font-medium">
                    {option.label}
                  </div>
                  <div className="mt-1 text-[11px] text-muted-foreground">
                    {option.statusLabel}
                  </div>
                </button>
              ))}
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );

  const renderWorkspaceModeToggle = () => (
    <Card>
      <CardContent className="flex flex-col gap-4 p-5 lg:flex-row lg:items-center lg:justify-between">
        <div className="space-y-1">
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
            Workflow
          </div>
          <h2 className="text-lg font-semibold text-foreground">
            Quick render or project workflow
          </h2>
          <p className="text-sm text-muted-foreground">
            Use quick mode for one-off generations, or switch to projects for
            reusable scripts and merged export.
          </p>
        </div>

        <Tabs
          value={workspaceMode}
          onValueChange={(value) => setWorkspaceMode(value as WorkspaceMode)}
          className="w-full max-w-sm"
        >
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="quick">Quick</TabsTrigger>
            <TabsTrigger value="projects">Projects</TabsTrigger>
          </TabsList>
        </Tabs>
      </CardContent>
    </Card>
  );

  if (workspaceMode === "projects") {
    return (
      <div className="space-y-4">
        {renderWorkspaceModeToggle()}
        <TextToSpeechProjectsWorkspace
          selectedModel={selectedModel}
          selectedModelInfo={selectedModelInfo}
          availableModels={availableModels}
          modelOptions={modelOptions}
          onSelectModel={onSelectModel}
          onOpenModelManager={onOpenModelManager}
          onModelRequired={onModelRequired}
          onError={onError}
        />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {renderWorkspaceModeToggle()}
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.1fr)_420px]">
        <Card>
          <CardContent className="space-y-6 p-6">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
              <div className="space-y-1">
                <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                  Voice-first TTS
                </div>
                <h2 className="text-lg font-semibold text-foreground">
                  Render with reusable or built-in voices
                </h2>
                <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">
                  Pick the voice first, then generate. The route will use a
                  compatible renderer for reusable saved voices versus built-in
                  speaker libraries.
                </p>
              </div>

              <div className="flex w-full max-w-md flex-col gap-3 lg:items-end">
                {renderModelSelector()}
                <div className="flex w-full items-center justify-between rounded-xl border border-border/60 bg-muted/20 px-3.5 py-2.5 text-xs">
                  <span className="text-muted-foreground">Model status</span>
                  <span
                    className={cn(
                      "font-semibold",
                      selectedModelReady ? "text-foreground" : "text-amber-500",
                    )}
                  >
                    {selectedOption?.statusLabel || "No model selected"}
                  </span>
                </div>
                {onOpenModelManager ? (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={onOpenModelManager}
                  >
                    <Settings2 className="h-4 w-4" />
                    Models
                  </Button>
                ) : null}
              </div>
            </div>

            <div className="rounded-2xl border border-border/60 bg-muted/20 p-4.5 px-5 py-4 text-sm text-muted-foreground">
              {compatibilityNotice}
            </div>

            <div className="space-y-2.5">
              <label className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                Script
              </label>
              <textarea
                ref={textareaRef}
                value={text}
                onChange={(event) => setText(event.target.value)}
                rows={7}
                placeholder="Paste the text you want this voice to speak..."
                className="min-h-[180px] w-full rounded-xl border border-input/85 bg-background/70 px-4 py-3 text-sm leading-relaxed shadow-sm transition-[border-color,box-shadow,background-color] placeholder:text-muted-foreground/85 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/35 focus-visible:border-ring/50"
              />
            </div>

            <div className="rounded-2xl border border-border/60 bg-muted/20 p-5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                    Delivery controls
                  </div>
                  <div className="mt-1.5 text-sm text-muted-foreground">
                    Speed is persisted with the generation history. Streaming is
                    available only on compatible models.
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

              <div className="mt-5 space-y-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium text-foreground">Speed</span>
                    <span className="text-muted-foreground">
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

                <label className="flex items-center gap-3 rounded-xl border border-border/60 bg-background/50 px-3.5 py-3 text-sm">
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
                    <span className="font-medium text-foreground">
                      Stream audio
                    </span>
                    <span className="block text-xs text-muted-foreground">
                      {supportsStreaming
                        ? "Play chunks as they arrive."
                        : "Current model does not expose streaming on this route."}
                    </span>
                  </span>
                  <Radio className="h-4 w-4 text-muted-foreground" />
                </label>

                <AnimatePresence>
                  {showAdvanced ? (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      className="overflow-hidden"
                    >
                      <div className="space-y-2 pt-1">
                        <label className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                          Style prompt
                        </label>
                        <Input
                          value={instructions}
                          onChange={(event) =>
                            setInstructions(event.target.value)
                          }
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
            </div>

            <AnimatePresence>
              {error ? (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="flex items-start gap-2 rounded-xl border border-destructive/45 bg-destructive/5 px-3 py-3 text-sm text-destructive">
                    <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
                    <p>{error}</p>
                  </div>
                </motion.div>
              ) : null}
            </AnimatePresence>

            <div className="flex flex-wrap items-center gap-3">
              <Button
                onClick={handleGenerate}
                disabled={
                  generating ||
                  !selectedModelReady ||
                  (voiceMode === "saved" && !selectedSavedVoiceId)
                }
                className="min-w-[180px]"
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

              {(audioUrl || isStreaming) && (
                <Button variant="outline" onClick={handleStop}>
                  <Square className="h-4 w-4" />
                  Stop
                </Button>
              )}

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

            <AnimatePresence>
              {downloadState !== "idle" && downloadMessage ? (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div
                    className={cn(
                      "flex items-center gap-2 rounded-xl border px-3 py-2 text-xs font-medium",
                      downloadState === "downloading" &&
                        "border-amber-500/30 bg-amber-500/10 text-amber-700",
                      downloadState === "success" &&
                        "border-emerald-500/25 bg-emerald-500/10 text-emerald-700",
                      downloadState === "error" &&
                        "border-destructive/35 bg-destructive/5 text-destructive",
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
                </motion.div>
              ) : null}
            </AnimatePresence>

            {audioUrl ? (
              <div className="space-y-4 rounded-2xl border border-border/75 bg-muted/25 p-4">
                <audio
                  ref={audioRef}
                  src={audioUrl}
                  className="h-11 w-full"
                  controls
                />
                {generationStats ? (
                  <GenerationStats stats={generationStats} type="tts" />
                ) : null}
              </div>
            ) : null}
          </CardContent>
        </Card>

        <Card>
          <CardContent className="space-y-5 p-6">
            <div className="space-y-1">
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                Voice picker
              </div>
              <h3 className="text-lg font-semibold text-foreground">
                Choose the voice before you type
              </h3>
            </div>

            <Tabs
              value={voiceMode}
              onValueChange={(value) => setVoiceMode(value as VoiceMode)}
              className="space-y-5"
            >
              <TabsList>
                <TabsTrigger value="saved">
                  <Waves className="h-4 w-4" />
                  My voices
                </TabsTrigger>
                <TabsTrigger value="built_in">
                  <Sparkles className="h-4 w-4" />
                  Built-in
                </TabsTrigger>
              </TabsList>

              <div className="space-y-2.5">
                <label className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                  Search voices
                </label>
                <Input
                  value={voiceSearch}
                  onChange={(event) => setVoiceSearch(event.target.value)}
                  placeholder="Search by name, transcript, or language"
                  className="bg-muted/20"
                />
              </div>

              <TabsContent value="saved" className="mt-0">
                {savedVoicesError ? (
                  <div className="mb-3 rounded-xl border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs text-destructive">
                    {savedVoicesError}
                  </div>
                ) : null}
                <VoicePicker
                  items={savedVoiceItems}
                  emptyTitle={
                    savedVoicesLoading
                      ? "Loading saved voices"
                      : "No saved voices yet"
                  }
                  emptyDescription={
                    savedVoicesLoading
                      ? "Fetching your cloned and designed voices."
                      : "Save a voice from cloning or design to reuse it directly in text-to-speech."
                  }
                />
              </TabsContent>

              <TabsContent value="built_in" className="mt-0">
                <VoicePicker
                  items={builtInVoiceItems}
                  emptyTitle="No built-in voices available"
                  emptyDescription="Load a CustomVoice, Kokoro, or LFM2 Audio model to browse its speaker library."
                />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>

      <SpeechHistoryPanel
        route="text-to-speech"
        title="Text to Speech History"
        emptyMessage="No saved text-to-speech generations yet."
        latestRecord={latestRecord}
        historyActionContainer={historyActionContainer}
      />
    </div>
  );
}
