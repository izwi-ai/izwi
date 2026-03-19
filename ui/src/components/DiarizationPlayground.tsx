import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Check,
  Copy,
  Download,
  Loader2,
  Mic,
  RotateCcw,
  Settings2,
  Upload,
  Square,
  AlertTriangle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { GenerationStats, type ASRStats } from "@/components/GenerationStats";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { MiniWaveform } from "@/components/ui/Waveform";
import { StatusBadge } from "@/components/ui/status-badge";
import { useWorkspaceShortcuts } from "@/hooks/useWorkspaceShortcuts";
import {
  api,
  type DiarizationRecord,
  type DiarizationRecordRerunRequest,
} from "../api";
import { formattedTranscriptFromResult } from "../utils/diarizationTranscript";
import { DiarizationExportDialog } from "./DiarizationExportDialog";
import { DiarizationHistoryPanel } from "./DiarizationHistoryPanel";
import { DiarizationQualityPanel } from "./DiarizationQualityPanel";
import { DiarizationReviewWorkspace } from "./DiarizationReviewWorkspace";
import { DiarizationSpeakerManager } from "./DiarizationSpeakerManager";

function revokeObjectUrlIfNeeded(url: string | null): void {
  if (url && url.startsWith("blob:")) {
    URL.revokeObjectURL(url);
  }
}

function formatDraftValue(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "";
  }
  if (Number.isInteger(value)) {
    return String(value);
  }
  return value.toFixed(2).replace(/\.?0+$/, "");
}

function parseOptionalInteger(value: string): number | undefined {
  const trimmed = value.trim();
  if (!trimmed) {
    return undefined;
  }
  const parsed = Number.parseInt(trimmed, 10);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function clampIntegerDraft(
  value: string,
  fallback: number,
  min: number,
  max: number,
): number {
  const parsed = parseOptionalInteger(value) ?? fallback;
  return Math.max(min, Math.min(max, parsed));
}

interface DiarizationPlaygroundProps {
  selectedModel: string | null;
  selectedModelReady?: boolean;
  onOpenModelManager?: () => void;
  onTogglePipelineLoadAll?: () => void;
  pipelineAllLoaded?: boolean;
  pipelineLoadAllBusy?: boolean;
  onModelRequired: () => void;
  pipelineAsrModelId?: string | null;
  pipelineAlignerModelId?: string | null;
  pipelineLlmModelId?: string | null;
  pipelineLlmModelReady?: boolean;
  pipelineModelsReady?: boolean;
  onPipelineModelsRequired?: () => void;
  historyActionContainer?: HTMLElement | null;
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

function isWavMimeType(mimeType: string | null | undefined): boolean {
  if (!mimeType) {
    return false;
  }
  const normalized = mimeType.toLowerCase();
  return (
    normalized === "audio/wav" ||
    normalized === "audio/x-wav" ||
    normalized === "audio/wave" ||
    normalized === "audio/vnd.wave"
  );
}

async function transcodeToWav(
  inputBlob: Blob,
  targetSampleRate = 16000,
  sourceFileName?: string,
): Promise<Blob> {
  const filenameLooksWav = sourceFileName
    ? sourceFileName.toLowerCase().endsWith(".wav")
    : false;
  if (isWavMimeType(inputBlob.type) || filenameLooksWav) {
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

export function DiarizationPlayground({
  selectedModel,
  selectedModelReady = false,
  onOpenModelManager,
  onTogglePipelineLoadAll,
  pipelineAllLoaded = false,
  pipelineLoadAllBusy = false,
  onModelRequired,
  pipelineAsrModelId = null,
  pipelineAlignerModelId = null,
  pipelineLlmModelId = null,
  pipelineModelsReady = true,
  onPipelineModelsRequired,
  historyActionContainer = null,
}: DiarizationPlaygroundProps) {
  const [speakerTranscript, setSpeakerTranscript] = useState("");
  const [isDiarizationSessionActive, setIsDiarizationSessionActive] =
    useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [latestRecord, setLatestRecord] = useState<DiarizationRecord | null>(
    null,
  );
  const [workspaceTab, setWorkspaceTab] = useState("transcript");
  const [speakerUpdatePending, setSpeakerUpdatePending] = useState(false);
  const [speakerUpdateError, setSpeakerUpdateError] = useState<string | null>(
    null,
  );
  const [rerunPending, setRerunPending] = useState(false);
  const [rerunError, setRerunError] = useState<string | null>(null);
  const [minSpeakers, setMinSpeakers] = useState("1");
  const [maxSpeakers, setMaxSpeakers] = useState("4");
  const [minSpeechMs, setMinSpeechMs] = useState("240");
  const [minSilenceMs, setMinSilenceMs] = useState("200");

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const requireReadyModel = useCallback(() => {
    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      return false;
    }
    return true;
  }, [selectedModel, selectedModelReady, onModelRequired]);

  const requireReadyPipelineModels = useCallback(() => {
    if (
      !pipelineAsrModelId ||
      !pipelineAlignerModelId ||
      !pipelineModelsReady
    ) {
      onPipelineModelsRequired?.();
      if (!onPipelineModelsRequired) {
        setError("Load ASR and forced aligner models before diarization.");
      }
      return false;
    }
    return true;
  }, [
    onPipelineModelsRequired,
    pipelineAlignerModelId,
    pipelineAsrModelId,
    pipelineModelsReady,
  ]);

  const normalizeSidebarSettings = useCallback(() => {
    let nextMinSpeakers = clampIntegerDraft(minSpeakers, 1, 1, 4);
    const nextMaxSpeakers = clampIntegerDraft(maxSpeakers, 4, 1, 4);
    const nextMinSpeechMs = clampIntegerDraft(minSpeechMs, 240, 40, 5000);
    const nextMinSilenceMs = clampIntegerDraft(minSilenceMs, 200, 40, 5000);

    if (nextMinSpeakers > nextMaxSpeakers) {
      nextMinSpeakers = nextMaxSpeakers;
    }

    const nextMinSpeakersText = formatDraftValue(nextMinSpeakers);
    const nextMaxSpeakersText = formatDraftValue(nextMaxSpeakers);
    const nextMinSpeechText = formatDraftValue(nextMinSpeechMs);
    const nextMinSilenceText = formatDraftValue(nextMinSilenceMs);

    if (minSpeakers !== nextMinSpeakersText) {
      setMinSpeakers(nextMinSpeakersText);
    }
    if (maxSpeakers !== nextMaxSpeakersText) {
      setMaxSpeakers(nextMaxSpeakersText);
    }
    if (minSpeechMs !== nextMinSpeechText) {
      setMinSpeechMs(nextMinSpeechText);
    }
    if (minSilenceMs !== nextMinSilenceText) {
      setMinSilenceMs(nextMinSilenceText);
    }

    return {
      minSpeakers: nextMinSpeakers,
      maxSpeakers: nextMaxSpeakers,
      minSpeechMs: nextMinSpeechMs,
      minSilenceMs: nextMinSilenceMs,
    };
  }, [maxSpeakers, minSilenceMs, minSpeakers, minSpeechMs]);

  const processAudio = useCallback(
    async (audioBlob: Blob) => {
      if (!requireReadyModel()) {
        return;
      }
      if (!requireReadyPipelineModels()) {
        return;
      }

      const captureSettings = normalizeSidebarSettings();

      setIsDiarizationSessionActive(true);
      setIsProcessing(true);
      setError(null);
      setSpeakerUpdateError(null);
      setRerunError(null);
      revokeObjectUrlIfNeeded(audioUrl);
      setAudioUrl(null);
      setLatestRecord(null);
      setWorkspaceTab("transcript");
      setCopied(false);
      setSpeakerTranscript("");

      try {
        const sourceFileName =
          audioBlob instanceof File && audioBlob.name
            ? audioBlob.name
            : "audio.wav";
        const wavBlob = await transcodeToWav(
          audioBlob,
          16000,
          sourceFileName,
        ).catch(() => audioBlob);

        const uploadFilename =
          wavBlob === audioBlob ? sourceFileName : "audio.wav";

        const record = await api.createDiarizationRecord({
          audio_file: wavBlob,
          audio_filename: uploadFilename,
          model_id: selectedModel || undefined,
          asr_model_id: pipelineAsrModelId || undefined,
          aligner_model_id: pipelineAlignerModelId || undefined,
          llm_model_id: pipelineLlmModelId || undefined,
          min_speakers: captureSettings.minSpeakers,
          max_speakers: captureSettings.maxSpeakers,
          min_speech_duration_ms: captureSettings.minSpeechMs,
          min_silence_duration_ms: captureSettings.minSilenceMs,
        });

        setLatestRecord(record);
        setWorkspaceTab("transcript");
        setAudioUrl(api.diarizationRecordAudioUrl(record.id));
        setSpeakerTranscript(formattedTranscriptFromResult(record));
      } catch (err) {
        setError(err instanceof Error ? err.message : "Diarization failed");
      } finally {
        setIsProcessing(false);
      }
    },
    [
      audioUrl,
      normalizeSidebarSettings,
      pipelineAlignerModelId,
      pipelineAsrModelId,
      pipelineLlmModelId,
      requireReadyModel,
      requireReadyPipelineModels,
      selectedModel,
    ],
  );

  const startRecording = useCallback(async () => {
    if (!requireReadyModel()) {
      return;
    }
    if (!requireReadyPipelineModels()) {
      return;
    }

    setIsDiarizationSessionActive(true);
    revokeObjectUrlIfNeeded(audioUrl);
    setAudioUrl(null);
    setLatestRecord(null);
    setSpeakerTranscript("");
    setWorkspaceTab("transcript");
    setSpeakerUpdateError(null);
    setRerunError(null);
    setCopied(false);
    setError(null);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      let mediaRecorder: MediaRecorder | null = null;
      const mimeCandidates = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/mp4",
      ];
      for (const mimeType of mimeCandidates) {
        if (MediaRecorder.isTypeSupported(mimeType)) {
          mediaRecorder = new MediaRecorder(stream, { mimeType });
          break;
        }
      }
      if (!mediaRecorder) {
        mediaRecorder = new MediaRecorder(stream);
      }
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: mediaRecorder?.mimeType || "audio/webm",
        });
        stream.getTracks().forEach((track) => track.stop());
        await processAudio(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch {
      setError("Could not access microphone. Please grant permission.");
    }
  }, [
    audioUrl,
    processAudio,
    requireReadyModel,
    requireReadyPipelineModels,
  ]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const input = event.currentTarget;
    const file = input.files?.[0];
    if (!file) {
      return;
    }
    input.value = "";
    await processAudio(file);
  };

  const openFilePicker = useCallback(() => {
    if (!requireReadyModel()) {
      return;
    }
    if (!requireReadyPipelineModels()) {
      return;
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
      fileInputRef.current.click();
    }
  }, [requireReadyModel, requireReadyPipelineModels]);

  const handleReset = () => {
    revokeObjectUrlIfNeeded(audioUrl);
    setSpeakerTranscript("");
    setIsDiarizationSessionActive(false);
    setAudioUrl(null);
    setLatestRecord(null);
    setWorkspaceTab("transcript");
    setSpeakerUpdateError(null);
    setRerunError(null);
    setError(null);
    setIsProcessing(false);
    setCopied(false);
  };

  const asText = useMemo(() => speakerTranscript.trim(), [speakerTranscript]);

  const handleCopy = async () => {
    if (!asText) {
      return;
    }
    await navigator.clipboard.writeText(asText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleSpeakerCorrectionsSave = useCallback(
    async (speakerNameOverrides: Record<string, string>) => {
      if (!latestRecord || speakerUpdatePending) {
        return;
      }

      setSpeakerUpdatePending(true);
      setSpeakerUpdateError(null);
      try {
        const updatedRecord = await api.updateDiarizationRecord(latestRecord.id, {
          speaker_name_overrides: speakerNameOverrides,
        });
        setLatestRecord(updatedRecord);
        setSpeakerTranscript(formattedTranscriptFromResult(updatedRecord));
      } catch (err) {
        setSpeakerUpdateError(
          err instanceof Error ? err.message : "Failed to save speaker corrections.",
        );
      } finally {
        setSpeakerUpdatePending(false);
      }
    },
    [latestRecord, speakerUpdatePending],
  );

  const handleRerunRecord = useCallback(
    async (request: DiarizationRecordRerunRequest) => {
      if (!latestRecord || rerunPending) {
        return;
      }

      setRerunPending(true);
      setRerunError(null);
      setSpeakerUpdateError(null);
      setError(null);

      try {
        const rerunRecord = await api.rerunDiarizationRecord(latestRecord.id, request);
        revokeObjectUrlIfNeeded(audioUrl);
        setLatestRecord(rerunRecord);
        setSpeakerTranscript(formattedTranscriptFromResult(rerunRecord));
        setAudioUrl(api.diarizationRecordAudioUrl(rerunRecord.id));
        setWorkspaceTab("transcript");
        setMinSpeakers(formatDraftValue(rerunRecord.min_speakers ?? 1));
        setMaxSpeakers(formatDraftValue(rerunRecord.max_speakers ?? 4));
        setMinSpeechMs(
          formatDraftValue(rerunRecord.min_speech_duration_ms ?? 240),
        );
        setMinSilenceMs(
          formatDraftValue(rerunRecord.min_silence_duration_ms ?? 200),
        );
      } catch (err) {
        setRerunError(
          err instanceof Error ? err.message : "Failed to rerun diarization.",
        );
      } finally {
        setRerunPending(false);
      }
    },
    [
      audioUrl,
      latestRecord,
      rerunPending,
    ],
  );

  useEffect(() => {
    return () => {
      revokeObjectUrlIfNeeded(audioUrl);
    };
  }, [audioUrl]);

  const canRunInput =
    !isProcessing && !isRecording && selectedModelReady && pipelineModelsReady;
  const hasOutput = speakerTranscript.trim().length > 0;
  const canResetSession = isDiarizationSessionActive && !isRecording;
  const processingStats = useMemo<ASRStats | null>(
    () =>
      latestRecord
        ? {
            processing_time_ms: latestRecord.processing_time_ms,
            audio_duration_secs: latestRecord.duration_secs,
            rtf: latestRecord.rtf,
          }
        : null,
    [latestRecord],
  );
  const activeSpeakerCount =
    latestRecord?.corrected_speaker_count ?? latestRecord?.speaker_count ?? null;
  const renderErrorAlert = (className?: string) =>
    error ? (
      <motion.div
        initial={{ opacity: 0, height: 0, y: 10 }}
        animate={{ opacity: 1, height: "auto", y: 0 }}
        exit={{ opacity: 0, height: 0, y: 10 }}
        className={cn(
          "p-3.5 rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] text-[var(--danger-text)] text-sm font-medium flex items-start gap-3",
          className,
        )}
      >
        <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
        {error}
      </motion.div>
    ) : null;

  const workspaceShortcuts = useMemo(
    () => [
      {
        key: "Enter",
        metaKey: true,
        enabled: selectedModelReady && pipelineModelsReady && !isProcessing,
        action: () => {
          if (isRecording) {
            stopRecording();
            return;
          }
          void startRecording();
        },
      },
      {
        key: "Escape",
        enabled: isRecording,
        action: stopRecording,
      },
      {
        key: "Escape",
        shiftKey: true,
        enabled: isDiarizationSessionActive && !isRecording,
        action: handleReset,
      },
    ],
    [
      handleReset,
      isDiarizationSessionActive,
      isProcessing,
      isRecording,
      pipelineModelsReady,
      selectedModelReady,
      startRecording,
      stopRecording,
    ],
  );

  useWorkspaceShortcuts(workspaceShortcuts);

  return (
    <div
      className={cn(
        "grid gap-5 lg:gap-6",
        isDiarizationSessionActive
          ? "xl:grid-cols-[340px,minmax(0,1fr)] xl:h-[calc(100dvh-11.75rem)]"
          : "mx-auto w-full max-w-3xl",
      )}
    >
      <div className="space-y-4">
        <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-4 pt-5 sm:p-5 xl:pt-7 space-y-4">
          <div className="space-y-2.5">
            <div className="flex items-center justify-between gap-3">
              <div className="inline-flex items-center gap-2 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
                <Settings2 className="w-3.5 h-3.5" />
                Session
              </div>
              <div className="flex shrink-0 items-center gap-2">
                {onTogglePipelineLoadAll ? (
                  <Button
                    onClick={onTogglePipelineLoadAll}
                    variant="outline"
                    size="sm"
                    className="h-8 gap-1.5 text-xs bg-[var(--bg-surface-1)] border-[var(--border-muted)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-surface-2)] shadow-sm"
                    disabled={pipelineLoadAllBusy}
                  >
                    {pipelineLoadAllBusy ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Loading...
                      </>
                    ) : pipelineAllLoaded ? (
                      "Unload Models"
                    ) : (
                      "Load Models"
                    )}
                  </Button>
                ) : null}
                {onOpenModelManager ? (
                  <Button
                    onClick={onOpenModelManager}
                    variant="outline"
                    size="sm"
                    className="h-8 gap-1.5 text-xs bg-[var(--bg-surface-1)] border-[var(--border-muted)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-surface-2)] shadow-sm"
                  >
                    <Settings2 className="w-4 h-4" />
                    Models
                  </Button>
                ) : null}
              </div>
            </div>
            <div>
              <h2 className="text-sm font-semibold text-[var(--text-primary)] sm:text-base">
                Diarization Settings
              </h2>
            </div>
          </div>

          <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3.5 sm:p-4 space-y-3.5">
            <div className="space-y-3.5">
              <div>
                <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                  Speaker Range
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <label className="text-xs font-medium text-[var(--text-muted)] space-y-2 block">
                    <span>Min speakers</span>
                    <input
                      aria-label="Min speakers"
                      type="text"
                      inputMode="numeric"
                      value={minSpeakers}
                      onChange={(event) => setMinSpeakers(event.target.value)}
                      onBlur={normalizeSidebarSettings}
                      className="flex h-10 w-full rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-1 text-sm text-[var(--text-primary)] transition-colors placeholder:text-[var(--text-subtle)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--bg-surface-1)] disabled:cursor-not-allowed disabled:opacity-50"
                    />
                  </label>
                  <label className="text-xs font-medium text-[var(--text-muted)] space-y-2 block">
                    <span>Max speakers</span>
                    <input
                      aria-label="Max speakers"
                      type="text"
                      inputMode="numeric"
                      value={maxSpeakers}
                      onChange={(event) => setMaxSpeakers(event.target.value)}
                      onBlur={normalizeSidebarSettings}
                      className="flex h-10 w-full rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-1 text-sm text-[var(--text-primary)] transition-colors placeholder:text-[var(--text-subtle)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--bg-surface-1)] disabled:cursor-not-allowed disabled:opacity-50"
                    />
                  </label>
                </div>
              </div>

              <div>
                <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                  Timing Windows
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <label className="text-xs font-medium text-[var(--text-muted)] space-y-2 block">
                    <span>Min speech (ms)</span>
                    <input
                      aria-label="Min speech (ms)"
                      type="text"
                      inputMode="numeric"
                      value={minSpeechMs}
                      onChange={(event) => setMinSpeechMs(event.target.value)}
                      onBlur={normalizeSidebarSettings}
                      className="flex h-10 w-full rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-1 text-sm text-[var(--text-primary)] transition-colors placeholder:text-[var(--text-subtle)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--bg-surface-1)] disabled:cursor-not-allowed disabled:opacity-50"
                    />
                  </label>
                  <label className="text-xs font-medium text-[var(--text-muted)] space-y-2 block">
                    <span>Min silence (ms)</span>
                    <input
                      aria-label="Min silence (ms)"
                      type="text"
                      inputMode="numeric"
                      value={minSilenceMs}
                      onChange={(event) => setMinSilenceMs(event.target.value)}
                      onBlur={normalizeSidebarSettings}
                      className="flex h-10 w-full rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-1 text-sm text-[var(--text-primary)] transition-colors placeholder:text-[var(--text-subtle)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--bg-surface-1)] disabled:cursor-not-allowed disabled:opacity-50"
                    />
                  </label>
                </div>
              </div>
            </div>

          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <button
              onClick={() => {
                if (isRecording) {
                  stopRecording();
                } else {
                  void startRecording();
                }
              }}
              className={cn(
                "rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4 sm:p-5 text-center transition-all duration-300",
                "flex min-h-[176px] flex-col items-center justify-center gap-3 hover:border-[var(--border-strong)]",
                isRecording
                  ? "border-red-300 bg-red-500/5 shadow-[0_18px_40px_-24px_rgba(239,68,68,0.55)]"
                  : "hover:bg-[var(--bg-surface-2)] shadow-sm",
                (!selectedModelReady || !pipelineModelsReady || isProcessing) &&
                  "opacity-50 cursor-not-allowed hover:border-[var(--border-muted)] hover:bg-[var(--bg-surface-1)]",
              )}
              disabled={!selectedModelReady || !pipelineModelsReady || isProcessing}
            >
              <div
                className={cn(
                  "relative flex h-20 w-20 items-center justify-center rounded-full border shadow-md",
                  isRecording
                    ? "border-red-400 bg-red-500 text-white shadow-red-500/20"
                    : "border-[var(--border-muted)] bg-[var(--bg-surface-0)] text-[var(--text-primary)]",
                )}
              >
                {isRecording ? (
                  <>
                    <div
                      className="absolute inset-0 rounded-full bg-red-500/20 animate-ping"
                      style={{ animationDuration: "1.5s" }}
                    />
                    <div
                      className="absolute inset-[-8px] rounded-full bg-red-500/10 animate-ping"
                      style={{ animationDuration: "2s" }}
                    />
                    <Square className="relative z-10 h-8 w-8 fill-current" />
                  </>
                ) : (
                  <Mic className="h-8 w-8" />
                )}
              </div>
              <div className="space-y-1">
                <p className="text-sm font-semibold text-[var(--text-primary)]">
                  {isRecording ? "Recording" : "Record audio"}
                </p>
                <p className="text-xs text-[var(--text-muted)]">
                  {isRecording ? "Tap to stop" : "Use your microphone"}
                </p>
              </div>
            </button>

            <div
              onClick={openFilePicker}
              className={cn(
                "rounded-xl border-2 border-dashed p-4 sm:p-5 transition-all duration-200 cursor-pointer group",
                "flex min-h-[176px] flex-col items-center justify-center gap-3 text-center",
                canRunInput
                  ? "border-[var(--border-strong)] bg-[var(--bg-surface-1)] hover:border-primary hover:bg-[var(--bg-surface-2)] hover:shadow-sm"
                  : "border-[var(--border-muted)] bg-[var(--bg-surface-1)] opacity-50 cursor-not-allowed",
              )}
            >
              <div className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-3 shadow-sm transition-transform duration-200 group-hover:scale-105">
                <Upload className="h-5 w-5 text-muted-foreground group-hover:text-primary transition-colors" />
              </div>
              <div className="space-y-1">
                <p className="text-sm font-semibold text-[var(--text-primary)] group-hover:text-primary transition-colors">
                  Upload audio
                </p>
                <p className="text-xs text-[var(--text-muted)]">
                  WAV, MP3, M4A, AAC
                </p>
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileUpload}
                className="hidden"
                disabled={!canRunInput}
              />
            </div>
          </div>

          <div className="flex items-center justify-end border-t border-[var(--border-muted)] pt-3">
            {canResetSession ? (
              <Button
                onClick={handleReset}
                variant="ghost"
                size="sm"
                className="h-8 gap-2 text-xs border border-transparent hover:border-[var(--border-muted)] bg-transparent hover:bg-[var(--bg-surface-1)]"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Reset Session
              </Button>
            ) : null}
          </div>

          {!isDiarizationSessionActive ? (
            <AnimatePresence>{renderErrorAlert()}</AnimatePresence>
          ) : null}
        </div>
      </div>

      {isDiarizationSessionActive ? (
        <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] flex flex-col min-h-[460px] lg:min-h-[560px] xl:h-full xl:min-h-0 overflow-hidden">
          <div className="px-4 sm:px-5 py-3 border-b border-[var(--border-muted)] flex items-center justify-between gap-3 bg-[var(--bg-surface-1)]">
            <h3 className="text-base font-semibold text-[var(--text-primary)]">
              Transcript
            </h3>
            <div className="flex flex-wrap items-center justify-end gap-2">
              {isRecording ? <StatusBadge tone="success">Recording</StatusBadge> : null}
              {!isRecording && activeSpeakerCount ? (
                <StatusBadge>{activeSpeakerCount} speakers</StatusBadge>
              ) : null}
              <Button
                onClick={handleCopy}
                variant="outline"
                size="icon"
                className="h-9 w-9 bg-[var(--bg-surface-1)] border-[var(--border-muted)] text-[var(--text-secondary)] hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)]"
                disabled={!hasOutput || isProcessing || isRecording}
                title="Copy transcript"
              >
                {copied ? (
                  <Check className="w-4 h-4 text-green-500" />
                ) : (
                  <Copy className="w-4 h-4" />
                )}
              </Button>
              <DiarizationExportDialog record={latestRecord}>
                <Button
                  variant="outline"
                  size="icon"
                  className="h-9 w-9 bg-[var(--bg-surface-1)] border-[var(--border-muted)] text-[var(--text-secondary)] hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)]"
                  disabled={!latestRecord || !hasOutput || isProcessing || isRecording}
                  title="Export transcript"
                >
                  <Download className="w-4 h-4" />
                </Button>
              </DiarizationExportDialog>
            </div>
          </div>

          <div className="flex flex-1 min-h-0 flex-col bg-[var(--bg-surface-0)]">
            <div className="flex-1 min-h-0 overflow-y-auto p-4 sm:p-6 scrollbar-thin">
              {isProcessing ? (
                <div className="h-full flex flex-col items-center justify-center text-sm font-medium text-[var(--text-muted)] gap-3">
                  <Loader2 className="w-5 h-5 animate-spin text-[var(--text-primary)]" />
                  Running diarization and transcript pipeline...
                </div>
              ) : latestRecord ? (
                <Tabs
                  value={workspaceTab}
                  onValueChange={setWorkspaceTab}
                  className="flex min-h-full flex-col gap-4"
                >
                  <TabsList className="w-full justify-start bg-[var(--bg-surface-1)]">
                    <TabsTrigger value="transcript">Transcript</TabsTrigger>
                    <TabsTrigger value="speakers">Speakers</TabsTrigger>
                    <TabsTrigger value="quality">Quality</TabsTrigger>
                  </TabsList>

                  <TabsContent value="transcript" className="mt-0 flex-1">
                    <DiarizationReviewWorkspace
                      record={latestRecord}
                      audioUrl={audioUrl}
                      emptyTitle="Ready to diarize"
                      emptyMessage="Record audio from your microphone or upload an audio file to start diarization. Your speaker-segmented transcript will appear here."
                    />
                  </TabsContent>

                  <TabsContent value="speakers" className="mt-0">
                    <DiarizationSpeakerManager
                      record={latestRecord}
                      isSaving={speakerUpdatePending}
                      error={speakerUpdateError}
                      onSave={handleSpeakerCorrectionsSave}
                    />
                  </TabsContent>

                  <TabsContent value="quality" className="mt-0">
                    <DiarizationQualityPanel
                      record={latestRecord}
                      isRerunning={rerunPending}
                      error={rerunError}
                      onRerun={handleRerunRecord}
                    />
                  </TabsContent>
                </Tabs>
              ) : (
                <div className="flex min-h-full flex-col gap-4">
                  {isRecording ? (
                    <div className="rounded-xl border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] px-4 py-3 text-sm text-[var(--status-warning-text)] flex items-center gap-3">
                      <MiniWaveform isActive={true} />
                      <span>Recording audio...</span>
                    </div>
                  ) : null}

                  <div className="flex-1 min-h-0">
                    <DiarizationReviewWorkspace
                      record={null}
                      audioUrl={null}
                      emptyTitle="Ready to diarize"
                      emptyMessage="Record audio from your microphone or upload an audio file to start diarization. Your speaker-segmented transcript will appear here."
                    />
                  </div>
                </div>
              )}
            </div>

            {processingStats && !isProcessing && !isRecording ? (
              <div
                data-testid="diarization-stats-footer"
                className="border-t border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-4 sm:px-6"
              >
                <GenerationStats
                  stats={processingStats}
                  type="asr"
                  surface="plain"
                  className="w-full justify-between gap-2 bg-transparent px-0 py-0 sm:justify-start sm:gap-3"
                />
              </div>
            ) : null}
          </div>

          <AnimatePresence>{renderErrorAlert("m-4")}</AnimatePresence>
        </div>
      ) : null}

      <div
        className={cn(
          "text-xs text-[var(--text-muted)]",
          isDiarizationSessionActive ? "xl:col-start-2" : "mx-1",
        )}
      >
        Shortcut: <span className="app-kbd">Ctrl/Cmd + Enter</span> start or stop capture, <span className="app-kbd">Esc</span> stop recording, <span className="app-kbd">Shift + Esc</span> reset.
      </div>

      <DiarizationHistoryPanel
        latestRecord={latestRecord}
        historyActionContainer={historyActionContainer}
      />
    </div>
  );
}
