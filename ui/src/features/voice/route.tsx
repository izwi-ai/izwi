import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Mic,
  Loader2,
  PhoneOff,
  AudioLines,
  Volume2,
  VolumeX,
  Settings2,
  Download,
  Play,
  Square,
  X,
} from "lucide-react";
import clsx from "clsx";
import { cn } from "@/lib/utils";
import { api, type ModelInfo, type VoiceObservation } from "@/api";
import { PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import {
  MODULAR_STACK_VARIANTS,
  UNIFIED_VOICE_PIPELINE_LABEL,
  VOICE_PIPELINE_LABEL,
  VOICE_AGENT_SYSTEM_PROMPT,
  buildVoiceRealtimeWebSocketUrl,
  decodePcmI16Bytes,
  encodeLiveMicPcm16,
  encodeVoiceRealtimeClientPcm16Frame,
  encodeWavPcm16,
  formatBytes,
  formatModelVariantLabel,
  isRunnableModelStatus,
  isUnifiedAudioChatVariant,
  isVoiceRealtimeServerEvent,
  makeTranscriptEntryId,
  mergeSampleChunks,
  parseFinalAnswer,
  type VoiceRealtimeMode,
  parseVoiceRealtimeAssistantAudioBinaryChunk,
  type RuntimeStatus,
  type TranscriptEntry,
  type VoicePageProps,
  type VoiceRealtimeAssistantAudioBinaryChunk,
  type VoiceRealtimeClientMessage,
  type VoiceRealtimeServerEvent,
} from "@/features/voice/realtime/support";
import { getSpeakerProfilesForVariant } from "@/types";

const VOICE_OUTPUT_MUTED_STORAGE_KEY = "izwi.voice.output_muted";
const VOICE_PLAYBACK_SPEED_STORAGE_KEY = "izwi.voice.playback_speed";

function clampPlaybackSpeed(value: number) {
  return Math.min(1.75, Math.max(0.75, value));
}

type VoiceConfigTab = "setup" | "models" | "agent" | "memory";

function isEditableShortcutTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) {
    return false;
  }

  const tagName = target.tagName.toLowerCase();
  return (
    tagName === "input" ||
    tagName === "textarea" ||
    tagName === "select" ||
    target.isContentEditable
  );
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

  const [selectedSpeaker, setSelectedSpeaker] = useState("Serena");
  const [voiceMode, setVoiceMode] = useState<VoiceRealtimeMode>("modular");

  const [vadThreshold, setVadThreshold] = useState(0.02);
  const [silenceDurationMs, setSilenceDurationMs] = useState(900);
  const [minSpeechMs, setMinSpeechMs] = useState(300);
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const [configTab, setConfigTab] = useState<VoiceConfigTab>("setup");
  const [isLoadAllRequested, setIsLoadAllRequested] = useState(false);
  const [isOutputMuted, setIsOutputMuted] = useState(() => {
    if (typeof window === "undefined") return false;
    return window.localStorage.getItem(VOICE_OUTPUT_MUTED_STORAGE_KEY) === "1";
  });
  const [playbackSpeed, setPlaybackSpeed] = useState(() => {
    if (typeof window === "undefined") return 1;
    const raw = Number.parseFloat(
      window.localStorage.getItem(VOICE_PLAYBACK_SPEED_STORAGE_KEY) ?? "1",
    );
    return Number.isFinite(raw) ? clampPlaybackSpeed(raw) : 1;
  });
  const [savedSystemPrompt, setSavedSystemPrompt] = useState(
    VOICE_AGENT_SYSTEM_PROMPT,
  );
  const [systemPromptDraft, setSystemPromptDraft] = useState(
    VOICE_AGENT_SYSTEM_PROMPT,
  );
  const [defaultSystemPrompt, setDefaultSystemPrompt] = useState(
    VOICE_AGENT_SYSTEM_PROMPT,
  );
  const [isVoiceProfileLoading, setIsVoiceProfileLoading] = useState(true);
  const [isVoiceProfileSaving, setIsVoiceProfileSaving] = useState(false);
  const [observationalMemoryEnabled, setObservationalMemoryEnabled] =
    useState(true);
  const [voiceObservations, setVoiceObservations] = useState<VoiceObservation[]>(
    [],
  );
  const [isVoiceObservationsLoading, setIsVoiceObservationsLoading] =
    useState(false);
  const [isVoiceObservationsMutating, setIsVoiceObservationsMutating] =
    useState(false);

  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamingProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const streamingProcessorSinkRef = useRef<GainNode | null>(null);
  const vadTimerRef = useRef<number | null>(null);
  const processingRef = useRef(false);
  const runtimeStatusRef = useRef<RuntimeStatus>("idle");
  const isSessionActiveRef = useRef(false);
  const turnIdRef = useRef(0);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioUrlRef = useRef<string | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);
  const ttsPlaybackContextRef = useRef<AudioContext | null>(null);
  const ttsPlaybackGainRef = useRef<GainNode | null>(null);
  const ttsPlaybackSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const ttsNextPlaybackTimeRef = useRef(0);
  const ttsSampleRateRef = useRef(24000);
  const ttsSamplesRef = useRef<Float32Array[]>([]);
  const ttsStreamSessionRef = useRef(0);
  const playbackSpeedRef = useRef(1);
  const isOutputMutedRef = useRef(false);
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
  const unifiedAudioModels = useMemo(
    () => sortedModels.filter((model) => isUnifiedAudioChatVariant(model.variant)),
    [sortedModels],
  );
  const selectedUnifiedInfo = useMemo(
    () => unifiedAudioModels[0] ?? null,
    [unifiedAudioModels],
  );
  const selectedUnifiedModel = selectedUnifiedInfo?.variant ?? null;
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
  const speakerModelVariant =
    voiceMode === "unified" ? selectedUnifiedModel : selectedTtsModel;
  const assistantSpeakers = useMemo(
    () => getSpeakerProfilesForVariant(speakerModelVariant),
    [speakerModelVariant],
  );
  const selectedSpeakerProfile = useMemo(
    () => assistantSpeakers.find((speaker) => speaker.id === selectedSpeaker) ?? null,
    [assistantSpeakers, selectedSpeaker],
  );
  const currentPipelineLabel =
    voiceMode === "unified"
      ? UNIFIED_VOICE_PIPELINE_LABEL
      : VOICE_PIPELINE_LABEL;
  const activeVoiceSystemPrompt =
    savedSystemPrompt.trim() ||
    defaultSystemPrompt.trim() ||
    VOICE_AGENT_SYSTEM_PROMPT;
  const isSystemPromptDirty =
    systemPromptDraft.trim() !== activeVoiceSystemPrompt.trim();

  useEffect(() => {
    if (!assistantSpeakers.some((speaker) => speaker.id === selectedSpeaker)) {
      setSelectedSpeaker(assistantSpeakers[0]?.id ?? "Serena");
    }
  }, [assistantSpeakers, selectedSpeaker]);

  useEffect(() => {
    runtimeStatusRef.current = runtimeStatus;
  }, [runtimeStatus]);

  useEffect(() => {
    let cancelled = false;

    const loadVoiceProfile = async () => {
      try {
        const profile = await api.getVoiceProfile();
        if (cancelled) return;

        setSavedSystemPrompt(profile.system_prompt || VOICE_AGENT_SYSTEM_PROMPT);
        setSystemPromptDraft(
          profile.system_prompt || profile.default_system_prompt,
        );
        setDefaultSystemPrompt(
          profile.default_system_prompt || VOICE_AGENT_SYSTEM_PROMPT,
        );
        setObservationalMemoryEnabled(
          profile.observational_memory_enabled ?? true,
        );
      } catch (err) {
        if (cancelled) return;
        const message =
          err instanceof Error
            ? err.message
            : "Failed to load voice profile settings";
        setError(message);
        onError?.(message);
      } finally {
        if (!cancelled) {
          setIsVoiceProfileLoading(false);
        }
      }
    };

    void loadVoiceProfile();

    return () => {
      cancelled = true;
    };
  }, [onError]);

  const loadVoiceObservations = useCallback(async () => {
    try {
      setIsVoiceObservationsLoading(true);
      const observations = await api.listVoiceObservations(25);
      setVoiceObservations(observations);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Failed to load observational memory";
      setError(message);
      onError?.(message);
    } finally {
      setIsVoiceObservationsLoading(false);
    }
  }, [onError]);

  useEffect(() => {
    if (!isConfigOpen) return;
    void loadVoiceObservations();
  }, [isConfigOpen, loadVoiceObservations]);

  useEffect(() => {
    playbackSpeedRef.current = playbackSpeed;
    if (typeof window !== "undefined") {
      window.localStorage.setItem(
        VOICE_PLAYBACK_SPEED_STORAGE_KEY,
        playbackSpeed.toString(),
      );
    }

    const audio = audioRef.current;
    if (audio) {
      audio.playbackRate = playbackSpeed;
      audio.defaultPlaybackRate = playbackSpeed;
    }
  }, [playbackSpeed]);

  useEffect(() => {
    isOutputMutedRef.current = isOutputMuted;
    if (typeof window !== "undefined") {
      window.localStorage.setItem(
        VOICE_OUTPUT_MUTED_STORAGE_KEY,
        isOutputMuted ? "1" : "0",
      );
    }

    const audio = audioRef.current;
    if (audio) {
      audio.muted = isOutputMuted;
    }

    if (ttsPlaybackGainRef.current) {
      ttsPlaybackGainRef.current.gain.value = isOutputMuted ? 0 : 1;
    }
  }, [isOutputMuted]);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcript, runtimeStatus]);

  const hasRunnableConfig = useMemo(() => {
    if (voiceMode === "unified") {
      return (
        !!selectedUnifiedInfo &&
        isRunnableModelStatus(selectedUnifiedInfo.status)
      );
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
    selectedAsrInfo,
    selectedTextInfo,
    selectedTtsInfo,
    selectedUnifiedInfo,
    voiceMode,
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

  const stopTtsStreamingPlayback = useCallback(() => {
    ttsStreamSessionRef.current += 1;

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
    ttsPlaybackGainRef.current = null;

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
      source.playbackRate.value = playbackSpeedRef.current;
      source.connect(ttsPlaybackGainRef.current ?? context.destination);

      const scheduledAt = Math.max(
        context.currentTime + 0.02,
        ttsNextPlaybackTimeRef.current,
      );
      source.start(scheduledAt);
      ttsNextPlaybackTimeRef.current =
        scheduledAt + buffer.duration / playbackSpeedRef.current;

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
        case "assistant_text_delta": {
          const entryId = voiceAssistantEntryIdsRef.current.get(
            event.utterance_id,
          );
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
          const playbackGain = playbackContext.createGain();
          playbackGain.gain.value = isOutputMutedRef.current ? 0 : 1;
          playbackGain.connect(playbackContext.destination);
          ttsPlaybackContextRef.current = playbackContext;
          ttsPlaybackGainRef.current = playbackGain;
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
                system_prompt: activeVoiceSystemPrompt,
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
      activeVoiceSystemPrompt,
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

      const startPromise = (async () => {
        const socket = await ensureVoiceRealtimeSocket();
        if (socket.readyState !== WebSocket.OPEN) {
          throw new Error("Voice realtime websocket is not connected");
        }
        const readyPromise = waitForVoiceRealtimeInputStreamReady();
        if (voiceMode === "unified") {
          if (!selectedUnifiedModel) {
            throw new Error(
              "A unified LFM2.5 Audio model is unavailable. Open Config.",
            );
          }
          sendVoiceRealtimeJson({
            type: "input_stream_start",
            mode: "unified",
            s2s_model_id: selectedUnifiedModel,
            speaker: selectedSpeaker,
            max_output_tokens: 1536,
            vad_threshold: vadThreshold,
            min_speech_ms: minSpeechMs,
            silence_duration_ms: silenceDurationMs,
            max_utterance_ms: 20_000,
            pre_roll_ms: 160,
            input_sample_rate: Math.round(inputSampleRate),
          });
        } else {
          if (!selectedAsrModel || !selectedTextModel || !selectedTtsModel) {
            throw new Error(
              "Required modular stack models are unavailable. Open Config.",
            );
          }
          sendVoiceRealtimeJson({
            type: "input_stream_start",
            mode: "modular",
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
        }
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
      selectedUnifiedModel,
      sendVoiceRealtimeJson,
      silenceDurationMs,
      vadThreshold,
      voiceMode,
      waitForVoiceRealtimeInputStreamReady,
    ],
  );

  const handleSaveSystemPrompt = useCallback(async () => {
    const nextPrompt =
      systemPromptDraft.trim() ||
      defaultSystemPrompt.trim() ||
      VOICE_AGENT_SYSTEM_PROMPT;

    try {
      setIsVoiceProfileSaving(true);
      const profile = await api.updateVoiceProfile({
        system_prompt: nextPrompt,
      });
      setSavedSystemPrompt(profile.system_prompt || nextPrompt);
      setSystemPromptDraft(profile.system_prompt || nextPrompt);
      setDefaultSystemPrompt(
        profile.default_system_prompt || defaultSystemPrompt,
      );
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to save system prompt";
      setError(message);
      onError?.(message);
    } finally {
      setIsVoiceProfileSaving(false);
    }
  }, [defaultSystemPrompt, onError, systemPromptDraft]);

  const handleSetObservationalMemoryEnabled = useCallback(
    async (nextValue: boolean) => {
      try {
        setIsVoiceObservationsMutating(true);
        const profile = await api.updateVoiceProfile({
          observational_memory_enabled: nextValue,
        });
        setObservationalMemoryEnabled(
          profile.observational_memory_enabled ?? nextValue,
        );
      } catch (err) {
        const message =
          err instanceof Error
            ? err.message
            : "Failed to update observational memory";
        setError(message);
        onError?.(message);
      } finally {
        setIsVoiceObservationsMutating(false);
      }
    },
    [onError],
  );

  const handleForgetObservation = useCallback(
    async (observationId: string) => {
      try {
        setIsVoiceObservationsMutating(true);
        await api.deleteVoiceObservation(observationId);
        setVoiceObservations((current) =>
          current.filter((item) => item.id !== observationId),
        );
      } catch (err) {
        const message =
          err instanceof Error
            ? err.message
            : "Failed to forget voice memory";
        setError(message);
        onError?.(message);
      } finally {
        setIsVoiceObservationsMutating(false);
      }
    },
    [onError],
  );

  const handleClearObservations = useCallback(async () => {
    try {
      setIsVoiceObservationsMutating(true);
      await api.clearVoiceObservations();
      setVoiceObservations([]);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Failed to clear voice memories";
      setError(message);
      onError?.(message);
    } finally {
      setIsVoiceObservationsMutating(false);
    }
  }, [onError]);

  const openConfig = useCallback((tab: VoiceConfigTab = "setup") => {
    setConfigTab(tab);
    setIsConfigOpen(true);
  }, []);

  const startSession = useCallback(async () => {
    if (!selectedAsrModel || !selectedTextModel || !selectedTtsModel) {
      const message = "Required modular stack models are unavailable. Open Config.";
      setError(message);
      onError?.(message);
      openConfig("models");
      return;
    }

    if (!hasRunnableConfig) {
      const message =
        "Required models are not loaded. Open Config and load the stack.";
      setError(message);
      onError?.(message);
      openConfig("models");
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

        const pcm16 = encodeLiveMicPcm16(mono);
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

      isSessionActiveRef.current = true;
      processingRef.current = false;
      setRuntimeStatus("listening");

      // Warm up realtime websocket + server-side VAD stream without blocking mic startup.
      void ensureVoiceRealtimeInputStreamStarted(audioContext.sampleRate).catch(
        (err) => {
          const message =
            err instanceof Error ? err.message : "Voice realtime connection failed";
          if (!isSessionActiveRef.current) {
            return;
          }
          setError(message);
          onError?.(message);
        },
      );

      const VAD_INTERVAL = 80;
      vadTimerRef.current = window.setInterval(() => {
        const analyserNode = analyserRef.current;
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
        if (isSpeech && runtimeStatusRef.current === "assistant_speaking") {
          const nextAccepted = (voiceWsPlaybackRef.current?.utteranceSeq ?? 0) + 1;
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
    onError,
    openConfig,
    selectedAsrModel,
    selectedTextModel,
    selectedTtsModel,
    sendVoiceRealtimeBinary,
    sendVoiceRealtimeJson,
    stopSession,
    vadThreshold,
  ]);

  const toggleSession = useCallback(() => {
    if (runtimeStatus === "idle") {
      void startSession();
    } else {
      stopSession();
    }
  }, [runtimeStatus, startSession, stopSession]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (
        event.defaultPrevented ||
        event.repeat ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        isEditableShortcutTarget(event.target)
      ) {
        return;
      }

      if (event.code === "Space") {
        event.preventDefault();
        toggleSession();
        return;
      }

      if (event.key === "Escape") {
        if (runtimeStatus !== "idle") {
          event.preventDefault();
          stopSession();
        }
        return;
      }

      if (event.key.toLowerCase() === "m") {
        event.preventDefault();
        setIsOutputMuted((current) => !current);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [runtimeStatus, stopSession, toggleSession]);

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

  const promptStatusLabel = isVoiceProfileLoading
    ? "Loading prompt"
    : isSystemPromptDirty
      ? "Unsaved edits"
      : savedSystemPrompt === defaultSystemPrompt
        ? "Default prompt"
        : "Custom prompt";

  const promptSummaryText = isVoiceProfileLoading
    ? "Fetching the saved voice agent instructions."
    : isSystemPromptDirty
      ? "You have local prompt edits that have not been saved yet."
      : savedSystemPrompt === defaultSystemPrompt
        ? "Voice mode is using the default assistant behavior."
        : "A custom saved prompt will apply to the next voice session.";

  const memoryStatusLabel = observationalMemoryEnabled
    ? voiceObservations.length === 0
      ? "Enabled"
      : `${voiceObservations.length} saved`
    : "Disabled";

  const memorySummaryText = observationalMemoryEnabled
    ? voiceObservations.length === 0
      ? "Memory is ready to capture stable preferences from modular conversations."
      : "Review or remove stored voice memories without leaving this modal."
    : "Memory capture is currently turned off for voice mode.";

  const modelStatusLabel = hasRunnableConfig ? "Ready to start" : "Needs model action";

  const modelSummaryText =
    voiceMode === "unified"
      ? "LFM2.5 Audio handles transcription, reasoning, and response speech in one native GGUF stack."
      : "Parakeet, Qwen, and Kokoro stay separate so each stage is inspectable.";

  const renderSetupTab = () => (
    <div className="space-y-4">
      <section className="grid gap-3 md:grid-cols-3">
        <button
          type="button"
          onClick={() => setConfigTab("models")}
          className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4 text-left transition-colors hover:border-[var(--border-strong)] hover:bg-[var(--bg-surface-2)]"
        >
          <span className="text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--text-muted)]">
            Models
          </span>
          <div className="mt-3 flex items-start justify-between gap-3">
            <div>
              <p className="text-sm font-medium text-[var(--text-primary)]">
                {modelStatusLabel}
              </p>
              <p className="mt-1 text-xs leading-relaxed text-[var(--text-muted)]">
                {modelSummaryText}
              </p>
            </div>
            <span className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-2 py-1 text-[10px] text-[var(--text-muted)]">
              {currentPipelineLabel}
            </span>
          </div>
        </button>

        <button
          type="button"
          onClick={() => setConfigTab("agent")}
          className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4 text-left transition-colors hover:border-[var(--border-strong)] hover:bg-[var(--bg-surface-2)]"
        >
          <span className="text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--text-muted)]">
            Agent
          </span>
          <div className="mt-3">
            <p className="text-sm font-medium text-[var(--text-primary)]">
              {promptStatusLabel}
            </p>
            <p className="mt-1 text-xs leading-relaxed text-[var(--text-muted)]">
              {promptSummaryText}
            </p>
          </div>
        </button>

        <button
          type="button"
          onClick={() => setConfigTab("memory")}
          className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4 text-left transition-colors hover:border-[var(--border-strong)] hover:bg-[var(--bg-surface-2)]"
        >
          <span className="text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--text-muted)]">
            Memory
          </span>
          <div className="mt-3">
            <p className="text-sm font-medium text-[var(--text-primary)]">
              {memoryStatusLabel}
            </p>
            <p className="mt-1 text-xs leading-relaxed text-[var(--text-muted)]">
              {memorySummaryText}
            </p>
          </div>
        </button>
      </section>

      <section className="space-y-4">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-sm font-medium text-white">Runtime Profile</h3>
          <span className="text-[11px] text-[var(--text-muted)]">
            Choose how inference is orchestrated.
          </span>
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <button
            type="button"
            onClick={() => setVoiceMode("modular")}
            className={cn(
              "rounded-lg border p-3 text-left transition-colors",
              voiceMode === "modular"
                ? "border-primary bg-primary/5 ring-1 ring-primary/20"
                : "border-[var(--border-muted)] bg-[var(--bg-surface-1)] hover:border-[var(--border-strong)]",
            )}
          >
            <div className="text-sm font-medium">Modular Voice Stack</div>
            <p className="mt-1 text-xs text-[var(--text-muted)]">
              Uses a fixed local stack: Parakeet ASR, Qwen3-1.7B-GGUF, and
              Kokoro-82M.
            </p>
          </button>
          <button
            type="button"
            onClick={() => setVoiceMode("unified")}
            className={cn(
              "rounded-lg border p-3 text-left transition-colors",
              voiceMode === "unified"
                ? "border-primary bg-primary/5 ring-1 ring-primary/20"
                : "border-[var(--border-muted)] bg-[var(--bg-surface-1)] hover:border-[var(--border-strong)]",
            )}
          >
            <div className="text-sm font-medium">Unified LFM2.5 Audio</div>
            <p className="mt-1 text-xs text-[var(--text-muted)]">
              Runs transcription, response planning, and synthesized speech in a
              single native GGUF model.
            </p>
          </button>
        </div>
        <div className="text-[11px] text-[var(--text-muted)]">
          Current mode: {currentPipelineLabel}
        </div>
      </section>

      <section className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3 space-y-3">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-medium text-white">Assistant Voice</h3>
            <p className="mt-1 text-xs text-[var(--text-muted)]">
              {voiceMode === "unified"
                ? "LFM2.5 Audio uses its native built-in voice inventory."
                : "Choose the built-in speaker exposed by the active realtime TTS model."}
            </p>
          </div>
          <span className="text-[11px] text-[var(--text-muted)]">
            {speakerModelVariant
              ? formatModelVariantLabel(speakerModelVariant)
              : "No voice model selected"}
          </span>
        </div>
        <Select
          value={selectedSpeaker}
          onValueChange={setSelectedSpeaker}
          disabled={assistantSpeakers.length === 0}
        >
          <SelectTrigger className="border-[var(--border-muted)] bg-[var(--bg-surface-2)]">
            <SelectValue placeholder="Select assistant voice" />
          </SelectTrigger>
          <SelectContent>
            {assistantSpeakers.map((speaker) => (
              <SelectItem key={speaker.id} value={speaker.id}>
                {speaker.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <div className="flex flex-wrap items-center gap-2 text-[11px] text-[var(--text-muted)]">
          {selectedSpeakerProfile ? (
            <>
              <span>{selectedSpeakerProfile.language}</span>
              <span>{selectedSpeakerProfile.description}</span>
            </>
          ) : (
            <span>No built-in voices are available for the selected realtime model.</span>
          )}
        </div>
        {runtimeStatus !== "idle" && (
          <p className="text-[11px] text-[var(--text-muted)]">
            Voice changes apply the next time you start listening.
          </p>
        )}
      </section>

      {renderAudioOutputSettings()}
      {renderAdvancedSpeechSettings()}
    </div>
  );

  const renderPromptSettings = () => (
    <section className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4 space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-medium text-white">Voice Agent Prompt</h3>
          <p className="text-xs text-[var(--text-muted)] mt-1">
            Customize how the assistant responds in voice mode. Prompt updates
            apply the next time you start a voice session.
          </p>
        </div>
        {runtimeStatus !== "idle" && (
          <span className="text-[11px] rounded-full border border-amber-500/30 bg-amber-500/10 px-2.5 py-1 text-amber-200">
            Restart required
          </span>
        )}
      </div>

      {isVoiceProfileLoading ? (
        <p className="text-xs text-[var(--text-muted)]">
          Loading saved prompt...
        </p>
      ) : (
        <>
          <Textarea
            value={systemPromptDraft}
            onChange={(event) => setSystemPromptDraft(event.target.value)}
            placeholder={defaultSystemPrompt}
            className="min-h-[220px] bg-[var(--bg-surface-2)] border-[var(--border-muted)]"
          />
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-[11px] text-[var(--text-muted)]">
              Saved prompt is used for websocket `session_start` and the legacy
              agent-session fallback path.
            </p>
            <div className="flex flex-wrap items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSystemPromptDraft(defaultSystemPrompt)}
                disabled={
                  isVoiceProfileSaving ||
                  systemPromptDraft === defaultSystemPrompt
                }
              >
                Restore Default
              </Button>
              <Button
                size="sm"
                onClick={() => void handleSaveSystemPrompt()}
                disabled={isVoiceProfileSaving || !isSystemPromptDirty}
              >
                {isVoiceProfileSaving ? "Saving..." : "Save Prompt"}
              </Button>
            </div>
          </div>
        </>
      )}
    </section>
  );

  const renderMemorySettings = () => (
    <section className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4 space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-sm font-medium text-white">
            Observational Memory
          </h3>
          <p className="text-xs text-[var(--text-muted)] mt-1">
            Save stable user preferences and facts from modular voice turns so
            future responses can stay grounded.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-[11px] text-[var(--text-muted)]">
            {observationalMemoryEnabled ? "Enabled" : "Disabled"}
          </span>
          <Switch
            checked={observationalMemoryEnabled}
            onCheckedChange={(checked) => {
              void handleSetObservationalMemoryEnabled(checked);
            }}
            disabled={isVoiceObservationsMutating}
            aria-label="Toggle observational memory"
          />
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-[11px] text-[var(--text-muted)]">
          Memory is currently applied only to modular `/voice` conversations
          and can be reviewed or deleted here.
        </p>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => void loadVoiceObservations()}
            disabled={isVoiceObservationsLoading}
          >
            Refresh
          </Button>
          <Button
            variant="destructive"
            size="sm"
            onClick={() => void handleClearObservations()}
            disabled={
              isVoiceObservationsMutating || voiceObservations.length === 0
            }
          >
            Clear All
          </Button>
        </div>
      </div>

      <div className="space-y-2 max-h-[360px] overflow-y-auto pr-1">
        {isVoiceObservationsLoading ? (
          <p className="text-xs text-[var(--text-muted)]">
            Loading voice memories...
          </p>
        ) : voiceObservations.length === 0 ? (
          <p className="text-xs text-[var(--text-muted)]">
            No voice memories stored yet.
          </p>
        ) : (
          voiceObservations.map((observation) => (
            <div
              key={observation.id}
              className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-3 py-2.5"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-[10px] uppercase tracking-wide text-[var(--text-muted)]">
                      {observation.category}
                    </span>
                    <span className="text-[10px] text-[var(--text-muted)]">
                      confidence {observation.confidence.toFixed(2)}
                    </span>
                    <span className="text-[10px] text-[var(--text-muted)]">
                      seen {observation.times_seen}x
                    </span>
                  </div>
                  <p className="mt-1 text-sm text-[var(--text-primary)]">
                    {observation.summary}
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="shrink-0"
                  onClick={() => void handleForgetObservation(observation.id)}
                  disabled={isVoiceObservationsMutating}
                >
                  Forget
                </Button>
              </div>
            </div>
          ))
        )}
      </div>
    </section>
  );

  const renderModelSettings = () => (
    <section className="space-y-3">
      <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3 space-y-3">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-medium text-white">Unified Model</h3>
            <p className="text-[11px] text-[var(--text-muted)] mt-1">
              Single-stack speech model for end-to-end voice replies.
            </p>
          </div>
          <span className="text-[11px] text-[var(--text-muted)]">
            {voiceMode === "unified" ? "Selected mode" : "Available mode"}
          </span>
        </div>
        {selectedUnifiedInfo ? (
          <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-2)] p-3 space-y-2">
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="text-sm font-medium text-white">
                  {formatModelVariantLabel(selectedUnifiedInfo.variant)}
                </div>
                <div className="mt-0.5 text-xs text-[var(--text-muted)] truncate">
                  {selectedUnifiedInfo.variant}
                </div>
              </div>
              <span
                className={clsx(
                  "text-[10px] px-1.5 py-0.5 rounded border whitespace-nowrap",
                  getStatusClass(selectedUnifiedInfo.status),
                )}
              >
                {getStatusLabel(selectedUnifiedInfo.status)}
              </span>
            </div>
            <p className="text-xs text-[var(--text-muted)]">
              Native LFM2.5 Audio GGUF bundle with speech-to-speech generation.
            </p>
            <div className="flex flex-wrap items-center justify-end gap-2">
              {selectedUnifiedInfo.status === "downloading" && onCancelDownload && (
                <Button
                  onClick={() => onCancelDownload(selectedUnifiedInfo.variant)}
                  variant="destructive"
                  size="sm"
                  className="text-xs h-7 gap-2"
                >
                  <X className="w-3.5 h-3.5" />
                  Cancel
                </Button>
              )}
              {selectedUnifiedInfo.status === "loading" && (
                <Button disabled variant="outline" size="sm" className="text-xs h-7 gap-2">
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  Loading
                </Button>
              )}
              {selectedUnifiedInfo.status !== "loading" &&
                selectedUnifiedInfo.status !== "downloading" && (
                  <Button
                    onClick={() => onModelAction(selectedUnifiedInfo)}
                    variant={selectedUnifiedInfo.status === "ready" ? "outline" : "default"}
                    size="sm"
                    className="text-xs h-7 gap-2"
                  >
                    {selectedUnifiedInfo.status === "not_downloaded" ||
                    selectedUnifiedInfo.status === "error" ? (
                      <Download className="w-3.5 h-3.5" />
                    ) : selectedUnifiedInfo.status === "downloaded" ? (
                      <Play className="w-3.5 h-3.5" />
                    ) : (
                      <Square className="w-3.5 h-3.5" />
                    )}
                    {selectedUnifiedInfo.status === "not_downloaded" ||
                    selectedUnifiedInfo.status === "error"
                      ? "Download"
                      : selectedUnifiedInfo.status === "downloaded"
                        ? "Load"
                        : "Unload"}
                  </Button>
                )}
            </div>
          </div>
        ) : (
          <p className="text-xs text-[var(--text-muted)]">
            No unified LFM2.5 Audio model is available in the current catalog.
          </p>
        )}
      </div>
      <div className="flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-medium text-white">Modular Stack Models</h3>
          <p className="text-[11px] text-[var(--text-muted)] mt-1">
            Fixed stack: Parakeet-TDT-0.6B-v3, Qwen3-1.7B-GGUF, Kokoro-82M.
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
                  <div className="text-sm font-medium text-white">{item.role}</div>
                  <div className="mt-0.5 text-xs text-[var(--text-muted)] truncate">
                    {model ? formatModelVariantLabel(model.variant) : item.requiredVariant}
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
                  {model ? getStatusLabel(model.status) : "Unavailable"}
                </span>
              </div>

              <div className="flex flex-wrap items-center justify-end gap-2">
                {model?.status === "downloading" && onCancelDownload && (
                  <Button
                    onClick={() => onCancelDownload(model.variant)}
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
                      variant={model.status === "ready" ? "outline" : "default"}
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
                      {model.status === "ready" && <Square className="w-3.5 h-3.5" />}
                      {model.status === "not_downloaded" || model.status === "error"
                        ? "Download"
                        : model.status === "downloaded"
                          ? "Load"
                          : "Unload"}
                    </Button>
                  )}
              </div>

              {model?.status === "downloading" && progressMeta && (
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
                    Downloading {Math.round(progressMeta.progress)}%
                    {progressMeta.progressValue &&
                      progressMeta.progressValue.totalBytes > 0 && (
                        <>
                          {" "}
                          (
                          {formatBytes(progressMeta.progressValue.downloadedBytes)} /{" "}
                          {formatBytes(progressMeta.progressValue.totalBytes)})
                        </>
                      )}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );

  const renderAudioOutputSettings = () => (
    <section className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3 space-y-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-medium text-white">Audio Output</h3>
          <p className="text-xs text-[var(--text-muted)] mt-1">
            Control assistant playback without leaving voice mode.
          </p>
        </div>
        <button
          onClick={() => setIsOutputMuted((current) => !current)}
          className="btn btn-secondary h-8 px-3 rounded-full text-[11px] gap-1.5"
        >
          {isOutputMuted ? (
            <VolumeX className="w-3.5 h-3.5" />
          ) : (
            <Volume2 className="w-3.5 h-3.5" />
          )}
          {isOutputMuted ? "Muted" : "Unmuted"}
        </button>
      </div>

      <div>
        <label className="text-xs text-[var(--text-muted)]">
          Playback Speed ({playbackSpeed.toFixed(2)}x)
        </label>
        <Slider
          aria-label="Playback speed"
          min={0.75}
          max={1.75}
          step={0.05}
          value={[playbackSpeed]}
          onValueChange={(value) => {
            const next = value[0];
            if (typeof next === "number") {
              setPlaybackSpeed(clampPlaybackSpeed(next));
            }
          }}
          className="mt-2"
        />
      </div>

      <div className="text-[11px] text-[var(--text-muted)]">
        Shortcuts: `Space` start or stop, `Escape` stop, `M` mute.
      </div>
    </section>
  );

  const renderAdvancedSpeechSettings = () => (
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
  );

  const startDisabled =
    (voiceMode === "unified"
      ? !selectedUnifiedModel
      : !selectedAsrModel || !selectedTextModel || !selectedTtsModel) ||
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
        <Button
          onClick={() => setIsOutputMuted((current) => !current)}
          variant="outline"
          size="sm"
          className="h-8 rounded-full border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-3 text-[11px] font-medium text-[var(--text-primary)] hover:bg-[var(--bg-surface-3)]"
          aria-label={isOutputMuted ? "Unmute assistant audio" : "Mute assistant audio"}
          aria-pressed={isOutputMuted}
          title={isOutputMuted ? "Unmute assistant audio (M)" : "Mute assistant audio (M)"}
        >
          {isOutputMuted ? (
            <VolumeX className="w-3.5 h-3.5" />
          ) : (
            <Volume2 className="w-3.5 h-3.5" />
          )}
          <span>{isOutputMuted ? "Muted" : "Audio"}</span>
        </Button>
        <button
          onClick={() => openConfig("setup")}
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

              <div className="p-4 sm:p-5 overflow-y-auto max-h-[calc(90vh-88px)]">
                <Tabs
                  value={configTab}
                  onValueChange={(value) =>
                    setConfigTab(value as VoiceConfigTab)
                  }
                  className="space-y-4"
                >
                  <TabsList className="grid w-full grid-cols-2 gap-1 border-[var(--border-strong)] bg-[var(--bg-surface-2)] p-1 shadow-sm sm:grid-cols-4">
                    <TabsTrigger
                      value="setup"
                      className="text-[var(--text-muted)] data-[state=active]:bg-[var(--accent-solid)] data-[state=active]:text-[var(--text-on-accent)] data-[state=active]:shadow-[0_8px_20px_-14px_rgba(17,17,17,0.55)]"
                    >
                      Setup
                    </TabsTrigger>
                    <TabsTrigger
                      value="models"
                      className="text-[var(--text-muted)] data-[state=active]:bg-[var(--accent-solid)] data-[state=active]:text-[var(--text-on-accent)] data-[state=active]:shadow-[0_8px_20px_-14px_rgba(17,17,17,0.55)]"
                    >
                      Models
                    </TabsTrigger>
                    <TabsTrigger
                      value="agent"
                      className="text-[var(--text-muted)] data-[state=active]:bg-[var(--accent-solid)] data-[state=active]:text-[var(--text-on-accent)] data-[state=active]:shadow-[0_8px_20px_-14px_rgba(17,17,17,0.55)]"
                    >
                      Agent
                    </TabsTrigger>
                    <TabsTrigger
                      value="memory"
                      className="text-[var(--text-muted)] data-[state=active]:bg-[var(--accent-solid)] data-[state=active]:text-[var(--text-on-accent)] data-[state=active]:shadow-[0_8px_20px_-14px_rgba(17,17,17,0.55)]"
                    >
                      Memory
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="setup" className="mt-0">
                    {renderSetupTab()}
                  </TabsContent>
                  <TabsContent value="models" className="mt-0">
                    {renderModelSettings()}
                  </TabsContent>
                  <TabsContent value="agent" className="mt-0">
                    {renderPromptSettings()}
                  </TabsContent>
                  <TabsContent value="memory" className="mt-0">
                    {renderMemorySettings()}
                  </TabsContent>
                </Tabs>
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
