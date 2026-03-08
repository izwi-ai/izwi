import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Check,
  Copy,
  Download,
  FileAudio,
  Loader2,
  Mic,
  RotateCcw,
  Settings2,
  Upload,
  Square,
  Users,
  AlertTriangle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { api, type DiarizationRecord } from "../api";
import {
  formattedTranscriptFromResult,
  transcriptEntriesFromResult,
} from "../utils/diarizationTranscript";
import { DiarizationHistoryPanel } from "./DiarizationHistoryPanel";

function revokeObjectUrlIfNeeded(url: string | null): void {
  if (url && url.startsWith("blob:")) {
    URL.revokeObjectURL(url);
  }
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
}: DiarizationPlaygroundProps) {
  const [speakerTranscript, setSpeakerTranscript] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [latestRecord, setLatestRecord] = useState<DiarizationRecord | null>(
    null,
  );
  const [minSpeakers, setMinSpeakers] = useState(1);
  const [maxSpeakers, setMaxSpeakers] = useState(4);
  const [minSpeechMs, setMinSpeechMs] = useState(240);
  const [minSilenceMs, setMinSilenceMs] = useState(200);

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

  const processAudio = useCallback(
    async (audioBlob: Blob) => {
      if (!requireReadyModel()) {
        return;
      }
      if (!requireReadyPipelineModels()) {
        return;
      }

      setIsProcessing(true);
      setError(null);
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
        revokeObjectUrlIfNeeded(audioUrl);
        setAudioUrl(null);

        const uploadFilename =
          wavBlob === audioBlob ? sourceFileName : "audio.wav";

        const record = await api.createDiarizationRecord({
          audio_file: wavBlob,
          audio_filename: uploadFilename,
          model_id: selectedModel || undefined,
          asr_model_id: pipelineAsrModelId || undefined,
          aligner_model_id: pipelineAlignerModelId || undefined,
          llm_model_id: pipelineLlmModelId || undefined,
          min_speakers: minSpeakers,
          max_speakers: maxSpeakers,
          min_speech_duration_ms: minSpeechMs,
          min_silence_duration_ms: minSilenceMs,
        });

        setLatestRecord(record);
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
      maxSpeakers,
      minSilenceMs,
      minSpeakers,
      minSpeechMs,
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
      setError(null);
    } catch {
      setError("Could not access microphone. Please grant permission.");
    }
  }, [processAudio, requireReadyModel]);

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
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
      fileInputRef.current.click();
    }
  }, [requireReadyModel]);

  const handleReset = () => {
    revokeObjectUrlIfNeeded(audioUrl);
    setSpeakerTranscript("");
    setAudioUrl(null);
    setLatestRecord(null);
    setError(null);
    setIsProcessing(false);
  };

  const transcriptEntries = useMemo(
    () =>
      latestRecord
        ? transcriptEntriesFromResult(latestRecord)
        : transcriptEntriesFromResult({
            utterances: [],
            transcript: speakerTranscript,
            raw_transcript: speakerTranscript,
          }),
    [latestRecord, speakerTranscript],
  );

  const asText = useMemo(() => speakerTranscript.trim(), [speakerTranscript]);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(asText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([asText], {
      type: "text/plain; charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `diarization-${Date.now()}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    if (minSpeakers > maxSpeakers) {
      setMinSpeakers(maxSpeakers);
    }
  }, [minSpeakers, maxSpeakers]);

  useEffect(() => {
    return () => {
      revokeObjectUrlIfNeeded(audioUrl);
    };
  }, [audioUrl]);

  const canRunInput = !isProcessing && !isRecording && selectedModelReady;
  const hasOutput = speakerTranscript.trim().length > 0;

  return (
    <div className="grid gap-4 lg:gap-6 xl:grid-cols-[340px,minmax(0,1fr)] xl:h-[calc(100dvh-11.75rem)]">
      <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-4 sm:p-5 space-y-4 xl:h-full xl:min-h-0 xl:overflow-y-auto">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="inline-flex items-center gap-2 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
              <FileAudio className="w-3.5 h-3.5" />
              Capture
            </div>
            <h2 className="text-base font-semibold text-[var(--text-primary)] mt-1.5">
              Audio Input
            </h2>
          </div>
          <div className="flex items-center gap-2">
            {onTogglePipelineLoadAll && (
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
                  "Unload All"
                ) : (
                  "Load All"
                )}
              </Button>
            )}
            {onOpenModelManager && (
              <Button
                onClick={onOpenModelManager}
                variant="outline"
                size="sm"
                className="h-8 gap-1.5 text-xs bg-[var(--bg-surface-1)] border-[var(--border-muted)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-surface-2)] shadow-sm"
              >
                <Settings2 className="w-4 h-4" />
                Models
              </Button>
            )}
          </div>
        </div>

        <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4 space-y-3">
          <div className="grid grid-cols-2 gap-4">
            <label className="text-xs font-semibold text-[var(--text-primary)] space-y-2 block">
              <span className="text-[var(--text-muted)] uppercase tracking-wider">
                Min Speakers
              </span>
              <input
                type="number"
                min={1}
                max={4}
                value={minSpeakers}
                onChange={(event) =>
                  setMinSpeakers(
                    Math.max(1, Math.min(4, Number(event.target.value) || 1)),
                  )
                }
                className="flex h-10 w-full rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-1 text-sm transition-colors placeholder:text-[var(--text-subtle)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--bg-surface-1)] disabled:cursor-not-allowed disabled:opacity-50"
              />
            </label>
            <label className="text-xs font-semibold text-[var(--text-primary)] space-y-2 block">
              <span className="text-[var(--text-muted)] uppercase tracking-wider">
                Max Speakers
              </span>
              <input
                type="number"
                min={1}
                max={4}
                value={maxSpeakers}
                onChange={(event) =>
                  setMaxSpeakers(
                    Math.max(1, Math.min(4, Number(event.target.value) || 4)),
                  )
                }
                className="flex h-10 w-full rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-1 text-sm transition-colors placeholder:text-[var(--text-subtle)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--bg-surface-1)] disabled:cursor-not-allowed disabled:opacity-50"
              />
            </label>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <label className="text-xs font-semibold text-[var(--text-primary)] space-y-2 block">
              <span className="text-[var(--text-muted)] uppercase tracking-wider">
                Min Speech (ms)
              </span>
              <input
                type="number"
                min={40}
                max={5000}
                value={minSpeechMs}
                onChange={(event) =>
                  setMinSpeechMs(
                    Math.max(
                      40,
                      Math.min(5000, Number(event.target.value) || 240),
                    ),
                  )
                }
                className="flex h-10 w-full rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-1 text-sm transition-colors placeholder:text-[var(--text-subtle)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--bg-surface-1)] disabled:cursor-not-allowed disabled:opacity-50"
              />
            </label>
            <label className="text-xs font-semibold text-[var(--text-primary)] space-y-2 block">
              <span className="text-[var(--text-muted)] uppercase tracking-wider">
                Min Silence (ms)
              </span>
              <input
                type="number"
                min={40}
                max={5000}
                value={minSilenceMs}
                onChange={(event) =>
                  setMinSilenceMs(
                    Math.max(
                      40,
                      Math.min(5000, Number(event.target.value) || 200),
                    ),
                  )
                }
                className="flex h-10 w-full rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-1 text-sm transition-colors placeholder:text-[var(--text-subtle)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--bg-surface-1)] disabled:cursor-not-allowed disabled:opacity-50"
              />
            </label>
          </div>
        </div>

        <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5">
          <div className="flex flex-col items-center">
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={cn(
                "w-24 h-24 rounded-full flex items-center justify-center transition-all duration-300 shadow-md",
                isRecording
                  ? "bg-red-500 hover:bg-red-600 scale-110 shadow-red-500/20 shadow-xl"
                  : "bg-[var(--bg-surface-3)] hover:bg-[var(--border-muted)] border-2 border-[var(--border-strong)] hover:border-[var(--text-muted)]",
                (!selectedModelReady || isProcessing) &&
                  "opacity-50 cursor-not-allowed",
              )}
              disabled={!selectedModelReady || isProcessing}
            >
              {isRecording ? (
                <Square className="w-10 h-10 text-white fill-current" />
              ) : (
                <Mic className="w-10 h-10 text-[var(--text-primary)]" />
              )}
            </button>
            <p className="mt-4 text-sm font-medium text-[var(--text-secondary)]">
              {isRecording
                ? "Recording... click to stop"
                : "Tap to record audio"}
            </p>

            <div className="w-full mt-6">
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-[var(--border-muted)]" />
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-[var(--bg-surface-0)] px-2 text-[var(--text-muted)]">
                    Or
                  </span>
                </div>
              </div>

              <div
                onClick={openFilePicker}
                className={cn(
                  "mt-4 flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 transition-colors cursor-pointer",
                  canRunInput
                    ? "border-[var(--border-strong)] hover:border-primary/50 hover:bg-[var(--bg-surface-2)] bg-[var(--bg-surface-1)]"
                    : "border-[var(--border-muted)] bg-[var(--bg-surface-1)] opacity-50 cursor-not-allowed",
                )}
              >
                <Upload className="w-6 h-6 text-[var(--text-muted)] mb-2" />
                <p className="text-sm font-medium text-[var(--text-primary)]">
                  Upload audio file
                </p>
                <p className="text-xs text-[var(--text-muted)] mt-1">
                  WAV, MP3, M4A, AAC
                </p>
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
          </div>
        </div>

        {audioUrl && (
          <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
            <div className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider mb-3">
              Latest input
            </div>
            <audio src={audioUrl} controls className="w-full h-10" />
          </div>
        )}

        {(hasOutput || audioUrl || error) && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleReset}
            className="w-full h-9 gap-2 text-xs border border-transparent hover:border-[var(--border-muted)] bg-transparent hover:bg-[var(--bg-surface-1)] mt-2"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Session
          </Button>
        )}
      </div>

      <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] flex flex-col min-h-[460px] lg:min-h-[560px] xl:min-h-0 xl:h-full overflow-hidden">
        <div className="px-4 sm:px-5 py-4 border-b border-[var(--border-muted)] flex items-center justify-between gap-3 bg-[var(--bg-surface-1)]">
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-[var(--text-primary)]">
              Diarized Transcript
            </h3>
          </div>
          <div className="flex items-center gap-2">
            <Button
              onClick={handleCopy}
              variant="outline"
              size="icon"
              className="h-9 w-9 bg-[var(--bg-surface-1)] border-[var(--border-muted)] text-[var(--text-secondary)] hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)]"
              disabled={!hasOutput || isProcessing}
              title="Copy transcript"
            >
              {copied ? (
                <Check className="w-4 h-4 text-green-500" />
              ) : (
                <Copy className="w-4 h-4" />
              )}
            </Button>
            <Button
              onClick={handleDownload}
              variant="outline"
              size="icon"
              className="h-9 w-9 bg-[var(--bg-surface-1)] border-[var(--border-muted)] text-[var(--text-secondary)] hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)]"
              disabled={!hasOutput || isProcessing}
              title="Download transcript text"
            >
              <Download className="w-4 h-4" />
            </Button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 sm:p-6 bg-[var(--bg-surface-0)] scrollbar-thin">
          {isProcessing ? (
            <div className="h-full flex flex-col items-center justify-center text-sm font-medium text-[var(--text-muted)] gap-3">
              <Loader2 className="w-5 h-5 animate-spin text-[var(--text-primary)]" />
              Running diarization and transcript pipeline...
            </div>
          ) : hasOutput ? (
            <div className="space-y-4">
              {transcriptEntries.length > 0 ? (
                <div className="space-y-4">
                  {transcriptEntries.map((entry, index) => (
                    <div
                      key={`${entry.speaker}-${entry.start}-${entry.end}-${index}`}
                      className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-5 shadow-sm"
                    >
                      <div className="flex items-center justify-between gap-3 mb-3">
                        <div className="flex items-center gap-2.5">
                          <div className="flex h-6 w-6 items-center justify-center rounded-full bg-[var(--bg-surface-2)] border border-[var(--border-strong)] text-[10px] font-semibold text-[var(--text-primary)] shadow-sm">
                            {entry.speaker?.charAt(0) ?? "S"}
                          </div>
                          <span className="text-sm font-semibold text-[var(--text-primary)]">
                            {entry.speaker}
                          </span>
                        </div>
                        <span className="text-[11px] font-medium text-[var(--text-secondary)] bg-[var(--bg-surface-2)] px-2 py-0.5 rounded-md border border-[var(--border-muted)]">
                          {entry.start.toFixed(2)}s - {entry.end.toFixed(2)}s
                        </span>
                      </div>
                      <p className="text-base text-[var(--text-secondary)] whitespace-pre-wrap break-words leading-relaxed selection:bg-[var(--accent-soft)]">
                        {entry.text}
                      </p>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-5 sm:p-6 shadow-sm">
                  <pre className="text-base text-[var(--text-secondary)] whitespace-pre-wrap break-words leading-relaxed font-sans selection:bg-[var(--accent-soft)]">
                    {speakerTranscript}
                  </pre>
                </div>
              )}
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-center px-6">
              <div className="max-w-sm">
                <div className="w-16 h-16 rounded-full bg-[var(--bg-surface-2)] flex items-center justify-center mx-auto mb-4 border border-[var(--border-muted)]">
                  <Users className="w-8 h-8 text-[var(--text-subtle)]" />
                </div>
                <p className="text-base font-semibold text-[var(--text-secondary)] mb-2">
                  Ready to diarize
                </p>
                <p className="text-sm text-[var(--text-muted)] leading-relaxed">
                  Record audio from your microphone or upload an audio file to
                  start diarization. Your speaker-segmented transcript will
                  appear here.
                </p>
              </div>
            </div>
          )}
        </div>

        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0, y: 10 }}
              animate={{ opacity: 1, height: "auto", y: 0 }}
              exit={{ opacity: 0, height: 0, y: 10 }}
              className="m-4 p-3.5 rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] text-[var(--danger-text)] text-sm font-medium flex items-start gap-3"
            >
              <AlertTriangle className="w-4 h-4 mt-0.5 shrink-0" />
              {error}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <DiarizationHistoryPanel latestRecord={latestRecord} />
    </div>
  );
}
