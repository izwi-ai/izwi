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
import { api, type ModelInfo } from "@/api";
import { PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import {
  ENABLE_LEGACY_LOCAL_UNIFIED_PATH,
  MODULAR_STACK_VARIANTS,
  PIPELINE_LABELS,
  VOICE_AGENT_SYSTEM_PROMPT,
  buildVoiceRealtimeWebSocketUrl,
  decodePcmI16Base64,
  decodePcmI16Bytes,
  encodeLiveMicPcm16,
  encodeVoiceRealtimeClientPcm16Frame,
  encodeWavPcm16,
  formatBytes,
  formatModelVariantLabel,
  isAsrVariant,
  isLfm2Variant,
  isRunnableModelStatus,
  isVoiceRealtimeServerEvent,
  makeTranscriptEntryId,
  mergeSampleChunks,
  parseFinalAnswer,
  parseVoiceRealtimeAssistantAudioBinaryChunk,
  transcodeToWav,
  type PipelineMode,
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

  const [pipelineMode, setPipelineMode] =
    useState<PipelineMode>("stt_chat_tts");
  const [selectedS2sModel, setSelectedS2sModel] = useState<string | null>(null);
  const [selectedSpeaker, setSelectedSpeaker] = useState("Serena");

  const [vadThreshold, setVadThreshold] = useState(0.02);
  const [silenceDurationMs, setSilenceDurationMs] = useState(900);
  const [minSpeechMs, setMinSpeechMs] = useState(300);
  const [isConfigOpen, setIsConfigOpen] = useState(false);
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
    agentSessionIdRef.current = null;
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
        if (lfm2DirectMode) {
          if (!selectedS2sModel) {
            throw new Error(
              "Select a speech-to-speech model before starting voice mode.",
            );
          }
          sendVoiceRealtimeJson({
            type: "input_stream_start",
            mode: "unified",
            s2s_model_id: selectedS2sModel,
            language: "English",
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
      lfm2DirectMode,
      minSpeechMs,
      selectedAsrModel,
      selectedS2sModel,
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
      system_prompt: activeVoiceSystemPrompt,
      planning_mode: "auto",
      title: "Voice Session",
    });

    agentSessionIdRef.current = session.id;
    return session.id;
  }, [activeVoiceSystemPrompt]);

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
        const playbackGain = playbackContext.createGain();
        playbackGain.gain.value = isOutputMutedRef.current ? 0 : 1;
        playbackGain.connect(playbackContext.destination);
        ttsPlaybackContextRef.current = playbackContext;
        ttsPlaybackGainRef.current = playbackGain;
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
            ttsPlaybackGainRef.current = null;

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
              source.playbackRate.value = playbackSpeedRef.current;
              source.connect(ttsPlaybackGainRef.current ?? context.destination);

              const scheduledAt = Math.max(
                context.currentTime + 0.02,
                ttsNextPlaybackTimeRef.current,
              );
              source.start(scheduledAt);
              ttsNextPlaybackTimeRef.current =
                scheduledAt + buffer.duration / playbackSpeedRef.current;

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
        audio.muted = isOutputMutedRef.current;
        audio.playbackRate = playbackSpeedRef.current;
        audio.defaultPlaybackRate = playbackSpeedRef.current;
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

      const useLegacyLocalUnifiedPath =
        lfm2DirectMode && ENABLE_LEGACY_LOCAL_UNIFIED_PATH;
      let recorder: MediaRecorder | null = null;
      if (useLegacyLocalUnifiedPath) {
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
      }

      mediaRecorderRef.current = recorder;
      isSessionActiveRef.current = true;
      processingRef.current = false;
      silenceMsRef.current = 0;
      speechStartRef.current = null;
      setRuntimeStatus("listening");

      if (!useLegacyLocalUnifiedPath) {
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

        if (!useLegacyLocalUnifiedPath) {
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
          onClick={() => setIsOutputMuted((current) => !current)}
          className="btn btn-secondary h-8 w-8 rounded-full p-0"
          aria-label={isOutputMuted ? "Unmute assistant audio" : "Mute assistant audio"}
          title={isOutputMuted ? "Unmute assistant audio (M)" : "Mute assistant audio (M)"}
        >
          {isOutputMuted ? (
            <VolumeX className="w-3.5 h-3.5" />
          ) : (
            <Volume2 className="w-3.5 h-3.5" />
          )}
        </button>
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

                <section className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3 space-y-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <h3 className="text-sm font-medium text-white">
                        Voice Agent Prompt
                      </h3>
                      <p className="text-xs text-[var(--text-muted)] mt-1">
                        Customize how the assistant responds in voice mode.
                        Prompt updates apply the next time you start a voice
                        session.
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
                        onChange={(event) =>
                          setSystemPromptDraft(event.target.value)
                        }
                        placeholder={defaultSystemPrompt}
                        className="min-h-[140px] bg-[var(--bg-surface-2)] border-[var(--border-muted)]"
                      />
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <p className="text-[11px] text-[var(--text-muted)]">
                          Saved prompt is used for websocket `session_start` and
                          the legacy agent-session fallback path.
                        </p>
                        <div className="flex flex-wrap items-center gap-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() =>
                              setSystemPromptDraft(defaultSystemPrompt)
                            }
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

                <section className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3 space-y-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <h3 className="text-sm font-medium text-white">
                        Audio Output
                      </h3>
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
                    Shortcuts: `Space` start or stop, `Escape` stop, `M`
                    mute.
                  </div>
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
