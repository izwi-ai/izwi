import { useCallback, useEffect, useRef, useState, useMemo } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle,
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  Download,
  FileAudio,
  FileText,
  History,
  Loader2,
  Mic,
  MicOff,
  Pause,
  Play,
  Radio,
  RefreshCw,
  RotateCcw,
  Settings2,
  SkipBack,
  SkipForward,
  Trash2,
  Upload,
  X,
  ChevronDown,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  api,
  type TranscriptionRecord,
  type TranscriptionRecordSummary,
} from "../api";
import { ASRStats, GenerationStats } from "./GenerationStats";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";

interface ModelOption {
  value: string;
  label: string;
  statusLabel: string;
  isReady: boolean;
}

interface TranscriptionPlaygroundProps {
  selectedModel: string | null;
  selectedModelReady?: boolean;
  modelOptions?: ModelOption[];
  onSelectModel?: (variant: string) => void;
  onOpenModelManager?: () => void;
  onModelRequired: () => void;
}

interface ProcessAudioOptions {
  filename?: string;
  transcode?: boolean;
  preserveTranscript?: boolean;
}

const LANGUAGE_OPTIONS = [
  "English",
  "Chinese",
  "Cantonese",
  "Arabic",
  "German",
  "French",
  "Spanish",
  "Portuguese",
  "Indonesian",
  "Italian",
  "Korean",
  "Russian",
  "Thai",
  "Vietnamese",
  "Japanese",
  "Turkish",
  "Hindi",
  "Malay",
  "Dutch",
  "Swedish",
  "Danish",
  "Finnish",
  "Polish",
  "Czech",
  "Filipino",
  "Persian",
  "Greek",
  "Romanian",
  "Hungarian",
  "Macedonian",
];

const TRANSCRIPTION_WS_BIN_MAGIC = "ITRW";
const TRANSCRIPTION_WS_BIN_VERSION = 1;
const TRANSCRIPTION_WS_BIN_KIND_CLIENT_PCM16 = 1;
const TRANSCRIPTION_WS_BIN_CLIENT_HEADER_LEN = 16;
const LIVE_MIC_PCM_FRAME_SIZE = 2048;

type TranscriptionRealtimeServerEvent =
  | { type: "session_ready"; protocol?: string }
  | { type: "session_started" }
  | {
      type: "transcript_partial";
      sequence: number;
      text: string;
      language?: string | null;
    }
  | { type: "error"; message?: string }
  | { type: "session_done" }
  | { type: "pong"; timestamp_ms?: number | null };

function buildTranscriptionRealtimeWebSocketUrl(apiBaseUrl: string): string {
  const base = new URL(apiBaseUrl, window.location.origin);
  base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  base.pathname = `${base.pathname.replace(/\/$/, "")}/transcription/realtime/ws`;
  base.search = "";
  base.hash = "";
  return base.toString();
}

function isTranscriptionRealtimeServerEvent(
  value: unknown,
): value is TranscriptionRealtimeServerEvent {
  return (
    !!value &&
    typeof value === "object" &&
    "type" in value &&
    typeof (value as { type?: unknown }).type === "string"
  );
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

function encodeTranscriptionRealtimePcm16Frame(
  pcm16Bytes: Uint8Array,
  sampleRate: number,
  frameSeq: number,
): Uint8Array {
  const frame = new Uint8Array(
    TRANSCRIPTION_WS_BIN_CLIENT_HEADER_LEN + pcm16Bytes.length,
  );
  frame[0] = TRANSCRIPTION_WS_BIN_MAGIC.charCodeAt(0);
  frame[1] = TRANSCRIPTION_WS_BIN_MAGIC.charCodeAt(1);
  frame[2] = TRANSCRIPTION_WS_BIN_MAGIC.charCodeAt(2);
  frame[3] = TRANSCRIPTION_WS_BIN_MAGIC.charCodeAt(3);
  frame[4] = TRANSCRIPTION_WS_BIN_VERSION;
  frame[5] = TRANSCRIPTION_WS_BIN_KIND_CLIENT_PCM16;
  frame[6] = 0;
  frame[7] = 0;
  const view = new DataView(frame.buffer);
  view.setUint32(8, sampleRate >>> 0, true);
  view.setUint32(12, frameSeq >>> 0, true);
  frame.set(pcm16Bytes, TRANSCRIPTION_WS_BIN_CLIENT_HEADER_LEN);
  return frame;
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

function normalizeTranscript(text: string): string {
  return text.trim().replace(/\s+/g, " ");
}

function buildTranscriptPreview(text: string, maxChars = 160): string {
  const normalized = normalizeTranscript(text);
  if (!normalized) {
    return "No transcript";
  }
  if (normalized.length <= maxChars) {
    return normalized;
  }
  return `${normalized.slice(0, maxChars)}...`;
}

function summarizeRecord(
  record: TranscriptionRecord,
): TranscriptionRecordSummary {
  return {
    id: record.id,
    created_at: record.created_at,
    model_id: record.model_id,
    language: record.language,
    duration_secs: record.duration_secs,
    processing_time_ms: record.processing_time_ms,
    rtf: record.rtf,
    audio_mime_type: record.audio_mime_type,
    audio_filename: record.audio_filename,
    transcription_preview: buildTranscriptPreview(record.transcription),
    transcription_chars: Array.from(record.transcription).length,
  };
}

function formatCreatedAt(timestampMs: number): string {
  if (!Number.isFinite(timestampMs)) {
    return "Unknown time";
  }
  const value = new Date(timestampMs);
  if (Number.isNaN(value.getTime())) {
    return "Unknown time";
  }
  return value.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatAudioDuration(durationSecs: number | null): string {
  if (
    durationSecs === null ||
    !Number.isFinite(durationSecs) ||
    durationSecs < 0
  ) {
    return "Unknown length";
  }
  if (durationSecs < 60) {
    return `${durationSecs.toFixed(1)}s`;
  }
  const minutes = Math.floor(durationSecs / 60);
  const seconds = Math.floor(durationSecs % 60);
  return `${minutes}m ${seconds}s`;
}

function formatClockTime(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds < 0) {
    return "0:00";
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

export function TranscriptionPlayground({
  selectedModel,
  selectedModelReady = false,
  modelOptions = [],
  onSelectModel,
  onOpenModelManager,
  onModelRequired,
}: TranscriptionPlaygroundProps) {
  const [transcription, setTranscription] = useState("");
  const [detectedLanguage, setDetectedLanguage] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [processingStats, setProcessingStats] = useState<ASRStats | null>(null);
  const [streamingEnabled, setStreamingEnabled] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState("English");
  const [isModelMenuOpen, setIsModelMenuOpen] = useState(false);
  const [historyRecords, setHistoryRecords] = useState<
    TranscriptionRecordSummary[]
  >([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [selectedHistoryRecordId, setSelectedHistoryRecordId] = useState<
    string | null
  >(null);
  const [selectedHistoryRecord, setSelectedHistoryRecord] =
    useState<TranscriptionRecord | null>(null);
  const [selectedHistoryLoading, setSelectedHistoryLoading] = useState(false);
  const [selectedHistoryError, setSelectedHistoryError] = useState<
    string | null
  >(null);
  const [isHistoryModalOpen, setIsHistoryModalOpen] = useState(false);
  const [historyCurrentTime, setHistoryCurrentTime] = useState(0);
  const [historyDuration, setHistoryDuration] = useState(0);
  const [historyIsPlaying, setHistoryIsPlaying] = useState(false);
  const [historyPlaybackRate, setHistoryPlaybackRate] = useState(1);
  const [historyAudioError, setHistoryAudioError] = useState<string | null>(
    null,
  );
  const [historyTranscriptCopied, setHistoryTranscriptCopied] = useState(false);
  const [deleteTargetRecordId, setDeleteTargetRecordId] = useState<
    string | null
  >(null);
  const [deleteRecordPending, setDeleteRecordPending] = useState(false);
  const [deleteRecordError, setDeleteRecordError] = useState<string | null>(
    null,
  );

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const liveMicWsRef = useRef<WebSocket | null>(null);
  const liveMicWsReadyRef = useRef(false);
  const liveMicSessionRef = useRef(0);
  const liveMicInputFrameSeqRef = useRef(0);
  const liveMicAudioContextRef = useRef<AudioContext | null>(null);
  const liveMicAudioSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const liveMicProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const liveMicProcessorSinkRef = useRef<GainNode | null>(null);
  const modelMenuRef = useRef<HTMLDivElement | null>(null);
  const historyAudioRef = useRef<HTMLAudioElement | null>(null);

  const selectedOption = useMemo(() => {
    if (!selectedModel) {
      return null;
    }
    return (
      modelOptions.find((option) => option.value === selectedModel) || null
    );
  }, [selectedModel, modelOptions]);

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

  const requireReadyModel = useCallback(() => {
    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      return false;
    }
    return true;
  }, [selectedModel, selectedModelReady, onModelRequired]);

  const mergeHistorySummary = useCallback(
    (summary: TranscriptionRecordSummary) => {
      setHistoryRecords((previous) => {
        const next = [
          summary,
          ...previous.filter((item) => item.id !== summary.id),
        ];
        next.sort((a, b) => b.created_at - a.created_at);
        return next;
      });
    },
    [],
  );

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const records = await api.listTranscriptionRecords();
      setHistoryRecords(records);
      setSelectedHistoryRecordId((current) => {
        if (current && records.some((item) => item.id === current)) {
          return current;
        }
        return records[0]?.id ?? null;
      });
    } catch (err) {
      setHistoryError(
        err instanceof Error
          ? err.message
          : "Failed to load transcription history.",
      );
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);

  useEffect(() => {
    if (!selectedHistoryRecordId) {
      setSelectedHistoryRecord(null);
      setSelectedHistoryError(null);
      return;
    }

    if (selectedHistoryRecord?.id === selectedHistoryRecordId) {
      return;
    }

    let cancelled = false;
    setSelectedHistoryLoading(true);
    setSelectedHistoryError(null);

    api
      .getTranscriptionRecord(selectedHistoryRecordId)
      .then((record) => {
        if (cancelled) {
          return;
        }
        setSelectedHistoryRecord(record);
        mergeHistorySummary(summarizeRecord(record));
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        setSelectedHistoryError(
          err instanceof Error
            ? err.message
            : "Failed to load transcription record details.",
        );
      })
      .finally(() => {
        if (!cancelled) {
          setSelectedHistoryLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [mergeHistorySummary, selectedHistoryRecord, selectedHistoryRecordId]);

  const closeHistoryModal = useCallback(() => {
    setIsHistoryModalOpen(false);
  }, []);

  const openHistoryRecord = useCallback((recordId: string) => {
    setSelectedHistoryRecordId(recordId);
    setSelectedHistoryError(null);
    setIsHistoryModalOpen(true);
  }, []);

  const openDeleteRecordConfirm = useCallback((recordId: string) => {
    setDeleteTargetRecordId(recordId);
    setDeleteRecordError(null);
  }, []);

  const closeDeleteRecordConfirm = useCallback(() => {
    if (deleteRecordPending) {
      return;
    }
    setDeleteTargetRecordId(null);
    setDeleteRecordError(null);
  }, [deleteRecordPending]);

  const confirmDeleteRecord = useCallback(async () => {
    if (!deleteTargetRecordId || deleteRecordPending) {
      return;
    }

    setDeleteRecordPending(true);
    setDeleteRecordError(null);

    try {
      await api.deleteTranscriptionRecord(deleteTargetRecordId);

      const previous = historyRecords;
      const deletedIndex = previous.findIndex(
        (record) => record.id === deleteTargetRecordId,
      );
      const remaining = previous.filter(
        (record) => record.id !== deleteTargetRecordId,
      );

      setHistoryRecords(remaining);

      if (selectedHistoryRecordId === deleteTargetRecordId) {
        const fallbackIndex =
          deletedIndex >= 0 ? Math.min(deletedIndex, remaining.length - 1) : 0;
        const fallbackId = remaining[fallbackIndex]?.id ?? null;
        setSelectedHistoryRecordId(fallbackId);
        if (!fallbackId) {
          setSelectedHistoryRecord(null);
          setIsHistoryModalOpen(false);
        }
      }

      if (selectedHistoryRecord?.id === deleteTargetRecordId) {
        setSelectedHistoryRecord(null);
      }

      setDeleteTargetRecordId(null);
      setDeleteRecordError(null);
    } catch (err) {
      setDeleteRecordError(
        err instanceof Error
          ? err.message
          : "Failed to delete transcription record.",
      );
    } finally {
      setDeleteRecordPending(false);
    }
  }, [
    deleteRecordPending,
    deleteTargetRecordId,
    historyRecords,
    selectedHistoryRecord,
    selectedHistoryRecordId,
  ]);

  useEffect(() => {
    if (!isHistoryModalOpen) {
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closeHistoryModal();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [closeHistoryModal, isHistoryModalOpen]);

  useEffect(() => {
    if (!isHistoryModalOpen) {
      return;
    }
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [isHistoryModalOpen]);

  useEffect(() => {
    if (isHistoryModalOpen) {
      return;
    }
    const audio = historyAudioRef.current;
    if (audio) {
      audio.pause();
    }
    setHistoryIsPlaying(false);
  }, [isHistoryModalOpen]);

  const stopLiveMicAudioPipeline = useCallback(() => {
    const processor = liveMicProcessorRef.current;
    liveMicProcessorRef.current = null;
    if (processor) {
      processor.onaudioprocess = null;
      try {
        processor.disconnect();
      } catch {
        // Best effort cleanup.
      }
    }

    const source = liveMicAudioSourceRef.current;
    liveMicAudioSourceRef.current = null;
    if (source) {
      try {
        source.disconnect();
      } catch {
        // Best effort cleanup.
      }
    }

    const sink = liveMicProcessorSinkRef.current;
    liveMicProcessorSinkRef.current = null;
    if (sink) {
      try {
        sink.disconnect();
      } catch {
        // Best effort cleanup.
      }
    }

    const context = liveMicAudioContextRef.current;
    liveMicAudioContextRef.current = null;
    if (context) {
      void context.close().catch(() => {});
    }

    liveMicInputFrameSeqRef.current = 0;
  }, []);

  const abortLiveMicStream = useCallback(() => {
    stopLiveMicAudioPipeline();
    liveMicWsReadyRef.current = false;
    liveMicInputFrameSeqRef.current = 0;
    const ws = liveMicWsRef.current;
    liveMicWsRef.current = null;
    if (
      ws &&
      (ws.readyState === WebSocket.OPEN ||
        ws.readyState === WebSocket.CONNECTING)
    ) {
      try {
        ws.close(1000, "transcription_reset");
      } catch {
        // Best effort cleanup.
      }
    }
  }, [stopLiveMicAudioPipeline]);

  const processAudio = useCallback(
    async (audioBlob: Blob, options: ProcessAudioOptions = {}) => {
      if (!requireReadyModel()) {
        return;
      }

      setIsProcessing(true);
      setError(null);
      setProcessingStats(null);
      if (!options.preserveTranscript) {
        setTranscription("");
      }

      try {
        const shouldTranscode =
          options.transcode ?? !(audioBlob instanceof File);
        const uploadBlob = shouldTranscode
          ? await transcodeToWav(audioBlob, 16000)
          : audioBlob;
        const uploadFilename =
          options.filename?.trim() ||
          (audioBlob instanceof File && audioBlob.name
            ? audioBlob.name
            : "audio.wav");

        const url = URL.createObjectURL(uploadBlob);
        setAudioUrl((previousUrl) => {
          if (previousUrl) {
            URL.revokeObjectURL(previousUrl);
          }
          return url;
        });

        if (streamingEnabled) {
          setIsStreaming(true);
          let finalRecordId: string | null = null;

          streamAbortRef.current = api.createTranscriptionRecordStream(
            {
              audio_file: uploadBlob,
              audio_filename: uploadFilename,
              model_id: selectedModel || undefined,
              language: selectedLanguage,
            },
            {
              onStart: () => {},
              onDelta: (delta) => {
                setTranscription((previous) => `${previous}${delta}`);
              },
              onFinal: (record) => {
                finalRecordId = record.id;
                setTranscription(record.transcription);
                setDetectedLanguage(record.language || null);
                setProcessingStats({
                  processing_time_ms: record.processing_time_ms,
                  audio_duration_secs: record.duration_secs,
                  rtf: record.rtf,
                });
                mergeHistorySummary(summarizeRecord(record));
                setSelectedHistoryRecord(record);
                setSelectedHistoryRecordId(record.id);
                setSelectedHistoryError(null);
              },
              onError: (errorMsg) => {
                setError(errorMsg);
              },
              onDone: () => {
                setIsStreaming(false);
                setIsProcessing(false);
                streamAbortRef.current = null;
                if (!finalRecordId) {
                  void loadHistory();
                }
              },
            },
          );
        } else {
          const record = await api.createTranscriptionRecord({
            audio_file: uploadBlob,
            audio_filename: uploadFilename,
            model_id: selectedModel || undefined,
            language: selectedLanguage,
          });

          setTranscription(record.transcription);
          setDetectedLanguage(record.language || null);
          setProcessingStats({
            processing_time_ms: record.processing_time_ms,
            audio_duration_secs: record.duration_secs,
            rtf: record.rtf,
          });
          mergeHistorySummary(summarizeRecord(record));
          setSelectedHistoryRecord(record);
          setSelectedHistoryRecordId(record.id);
          setSelectedHistoryError(null);
          setIsProcessing(false);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "Transcription failed");
        setIsProcessing(false);
        setIsStreaming(false);
      }
    },
    [
      loadHistory,
      mergeHistorySummary,
      requireReadyModel,
      selectedModel,
      selectedLanguage,
      streamingEnabled,
    ],
  );

  const startRecording = useCallback(async () => {
    if (!requireReadyModel()) {
      return;
    }

    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const liveStream = stream;
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
      const recordingSession = liveMicSessionRef.current + 1;
      liveMicSessionRef.current = recordingSession;
      abortLiveMicStream();
      liveMicSessionRef.current = recordingSession;
      liveMicInputFrameSeqRef.current = 0;

      const ws = new WebSocket(
        buildTranscriptionRealtimeWebSocketUrl(api.baseUrl),
      );
      ws.binaryType = "arraybuffer";
      liveMicWsRef.current = ws;
      liveMicWsReadyRef.current = false;

      ws.onopen = () => {
        if (liveMicSessionRef.current !== recordingSession) {
          try {
            ws.close(1000, "stale_session");
          } catch {
            // noop
          }
          return;
        }
        ws.send(
          JSON.stringify({
            type: "session_start",
            model_id: selectedModel || undefined,
            language: selectedLanguage,
          }),
        );
      };

      ws.onmessage = (messageEvent) => {
        if (liveMicSessionRef.current !== recordingSession) {
          return;
        }
        if (typeof messageEvent.data !== "string") {
          return;
        }
        let parsed: unknown;
        try {
          parsed = JSON.parse(messageEvent.data);
        } catch {
          return;
        }
        if (!isTranscriptionRealtimeServerEvent(parsed)) {
          return;
        }

        switch (parsed.type) {
          case "session_ready":
            liveMicWsReadyRef.current = true;
            break;
          case "session_started":
            break;
          case "transcript_partial":
            setTranscription(parsed.text || "");
            setDetectedLanguage(parsed.language || null);
            break;
          case "error":
            setError(parsed.message || "Realtime transcription error");
            break;
          case "session_done":
          case "pong":
            break;
        }
      };

      ws.onclose = () => {
        if (liveMicWsRef.current === ws) {
          liveMicWsRef.current = null;
        }
        liveMicWsReadyRef.current = false;
      };

      ws.onerror = () => {
        if (liveMicSessionRef.current !== recordingSession) {
          return;
        }
        setError("Live transcription connection error");
      };

      if (streamAbortRef.current) {
        streamAbortRef.current.abort();
        streamAbortRef.current = null;
      }

      setTranscription("");
      setDetectedLanguage(null);
      setProcessingStats(null);
      setIsStreaming(true);

      const audioContext = new AudioContext();
      await audioContext.resume();
      liveMicAudioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(liveStream);
      liveMicAudioSourceRef.current = source;

      const processor = audioContext.createScriptProcessor(
        LIVE_MIC_PCM_FRAME_SIZE,
        1,
        1,
      );
      liveMicProcessorRef.current = processor;

      const sink = audioContext.createGain();
      sink.gain.value = 0;
      liveMicProcessorSinkRef.current = sink;

      processor.onaudioprocess = (event) => {
        if (liveMicSessionRef.current !== recordingSession) {
          return;
        }
        const socket = liveMicWsRef.current;
        if (
          !socket ||
          socket.readyState !== WebSocket.OPEN ||
          !liveMicWsReadyRef.current
        ) {
          return;
        }

        const inputBuffer = event.inputBuffer;
        const channelCount = inputBuffer.numberOfChannels;
        const frameCount = inputBuffer.length;
        if (frameCount <= 0 || channelCount <= 0) {
          return;
        }

        const mono = new Float32Array(frameCount);
        for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
          const channel = inputBuffer.getChannelData(channelIndex);
          for (let sampleIndex = 0; sampleIndex < frameCount; sampleIndex += 1) {
            mono[sampleIndex] +=
              (channel[sampleIndex] ?? 0) / channelCount;
          }
        }

        const pcm16 = encodeFloat32ToPcm16Bytes(mono);
        const frameSeq = (liveMicInputFrameSeqRef.current + 1) >>> 0;
        liveMicInputFrameSeqRef.current = frameSeq;

        try {
          socket.send(
            encodeTranscriptionRealtimePcm16Frame(
              pcm16,
              Math.round(inputBuffer.sampleRate),
              frameSeq,
            ),
          );
        } catch {
          // Best effort send while websocket is open.
        }
      };

      source.connect(processor);
      processor.connect(sink);
      sink.connect(audioContext.destination);

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        liveMicSessionRef.current = 0;
        abortLiveMicStream();
        setIsStreaming(false);
        const audioBlob = new Blob(audioChunksRef.current, {
          type: mediaRecorder?.mimeType || "audio/webm",
        });
        liveStream.getTracks().forEach((track) => track.stop());
        await processAudio(audioBlob, { preserveTranscript: true });
      };

      mediaRecorder.start(1000);
      setIsRecording(true);
      setError(null);
    } catch {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
      abortLiveMicStream();
      setError("Could not access microphone. Please grant permission.");
    }
  }, [
    abortLiveMicStream,
    processAudio,
    requireReadyModel,
    selectedLanguage,
    selectedModel,
  ]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      liveMicSessionRef.current = 0;
      const ws = liveMicWsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify({ type: "session_stop" }));
        } catch {
          // Best effort.
        }
      }
      abortLiveMicStream();
      setIsStreaming(false);
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [abortLiveMicStream, isRecording]);

  const handleFileUpload = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    await processAudio(file, {
      filename: file.name,
      transcode: false,
    });
    event.target.value = "";
  };

  const handleReset = () => {
    if (streamAbortRef.current) {
      streamAbortRef.current.abort();
      streamAbortRef.current = null;
    }
    liveMicSessionRef.current = 0;
    abortLiveMicStream();
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setTranscription("");
    setDetectedLanguage(null);
    setAudioUrl(null);
    setError(null);
    setProcessingStats(null);
    setIsStreaming(false);
    setIsProcessing(false);
  };

  const handleCopy = async () => {
    await navigator.clipboard.writeText(transcription);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([transcription], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `transcription-${Date.now()}.txt`;
    link.click();
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    return () => {
      if (streamAbortRef.current) {
        streamAbortRef.current.abort();
      }
      abortLiveMicStream();
    };
  }, [abortLiveMicStream]);

  const canRunInput = !isProcessing && !isRecording && selectedModelReady;
  const showResult = Boolean(transcription || isStreaming || isProcessing);
  const hasDraft = Boolean(transcription || audioUrl || error);
  const selectedHistorySummary = useMemo(
    () =>
      selectedHistoryRecordId
        ? (historyRecords.find(
            (record) => record.id === selectedHistoryRecordId,
          ) ?? null)
        : null,
    [historyRecords, selectedHistoryRecordId],
  );
  const activeHistoryRecord =
    selectedHistoryRecord &&
    selectedHistoryRecord.id === selectedHistoryRecordId
      ? selectedHistoryRecord
      : null;
  const deleteTargetRecord = useMemo(() => {
    if (!deleteTargetRecordId) {
      return null;
    }
    const fromSummary = historyRecords.find(
      (record) => record.id === deleteTargetRecordId,
    );
    if (fromSummary) {
      return fromSummary;
    }
    if (
      activeHistoryRecord &&
      activeHistoryRecord.id === deleteTargetRecordId
    ) {
      return summarizeRecord(activeHistoryRecord);
    }
    return null;
  }, [activeHistoryRecord, deleteTargetRecordId, historyRecords]);
  const selectedHistoryAudioUrl = useMemo(
    () =>
      selectedHistoryRecordId
        ? api.transcriptionRecordAudioUrl(selectedHistoryRecordId)
        : null,
    [selectedHistoryRecordId],
  );
  const selectedHistoryIndex = useMemo(
    () =>
      selectedHistoryRecordId
        ? historyRecords.findIndex(
            (record) => record.id === selectedHistoryRecordId,
          )
        : -1,
    [historyRecords, selectedHistoryRecordId],
  );
  const historyViewerDuration =
    historyDuration > 0
      ? historyDuration
      : activeHistoryRecord?.duration_secs &&
          activeHistoryRecord.duration_secs > 0
        ? activeHistoryRecord.duration_secs
        : 0;
  const canOpenNewerHistory = selectedHistoryIndex > 0;
  const canOpenOlderHistory =
    selectedHistoryIndex >= 0 &&
    selectedHistoryIndex < historyRecords.length - 1;

  const openAdjacentHistoryRecord = useCallback(
    (direction: "newer" | "older") => {
      if (selectedHistoryIndex < 0) {
        return;
      }
      const targetIndex =
        direction === "newer"
          ? selectedHistoryIndex - 1
          : selectedHistoryIndex + 1;
      if (targetIndex < 0 || targetIndex >= historyRecords.length) {
        return;
      }
      const target = historyRecords[targetIndex];
      if (!target) {
        return;
      }
      setSelectedHistoryRecordId(target.id);
      setSelectedHistoryError(null);
      setIsHistoryModalOpen(true);
    },
    [historyRecords, selectedHistoryIndex],
  );

  const toggleHistoryPlayback = useCallback(async () => {
    const audio = historyAudioRef.current;
    if (!audio) {
      return;
    }
    try {
      if (audio.paused) {
        await audio.play();
      } else {
        audio.pause();
      }
    } catch {
      setHistoryAudioError("Unable to start playback for this audio.");
    }
  }, []);

  const seekHistoryAudio = useCallback(
    (nextTime: number) => {
      const audio = historyAudioRef.current;
      if (!audio) {
        return;
      }
      const duration = Number.isFinite(audio.duration)
        ? audio.duration
        : historyViewerDuration;
      const clamped = Math.max(0, Math.min(nextTime, duration || 0));
      audio.currentTime = clamped;
      setHistoryCurrentTime(clamped);
    },
    [historyViewerDuration],
  );

  const skipHistoryAudio = useCallback(
    (deltaSeconds: number) => {
      const audio = historyAudioRef.current;
      if (!audio) {
        return;
      }
      const duration = Number.isFinite(audio.duration)
        ? audio.duration
        : historyViewerDuration;
      const next = Math.max(
        0,
        Math.min(audio.currentTime + deltaSeconds, duration || 0),
      );
      audio.currentTime = next;
      setHistoryCurrentTime(next);
    },
    [historyViewerDuration],
  );

  const handleHistoryRateChange = useCallback((rate: number) => {
    const audio = historyAudioRef.current;
    if (!audio) {
      return;
    }
    audio.playbackRate = rate;
    setHistoryPlaybackRate(rate);
  }, []);

  const handleCopyHistoryTranscript = useCallback(async () => {
    if (!activeHistoryRecord?.transcription) {
      return;
    }
    await navigator.clipboard.writeText(activeHistoryRecord.transcription);
    setHistoryTranscriptCopied(true);
    window.setTimeout(() => setHistoryTranscriptCopied(false), 1800);
  }, [activeHistoryRecord]);

  const handleDownloadHistoryTranscript = useCallback(() => {
    if (!activeHistoryRecord) {
      return;
    }
    const blob = new Blob([activeHistoryRecord.transcription], {
      type: "text/plain",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `transcription-${activeHistoryRecord.id}.txt`;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [activeHistoryRecord]);

  useEffect(() => {
    const audio = historyAudioRef.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
      audio.playbackRate = 1;
    }
    setHistoryCurrentTime(0);
    setHistoryDuration(0);
    setHistoryIsPlaying(false);
    setHistoryPlaybackRate(1);
    setHistoryAudioError(null);
    setHistoryTranscriptCopied(false);
  }, [selectedHistoryRecordId]);

  const getStatusTone = (option: ModelOption): string => {
    if (option.isReady) {
      return "text-green-500 bg-green-500/10";
    }
    if (
      option.statusLabel.toLowerCase().includes("downloading") ||
      option.statusLabel.toLowerCase().includes("loading")
    ) {
      return "text-amber-500 bg-amber-500/10";
    }
    if (option.statusLabel.toLowerCase().includes("error")) {
      return "text-destructive bg-destructive/10";
    }
    return "text-muted-foreground bg-muted";
  };

  const handleOpenModels = () => {
    setIsModelMenuOpen(false);
    onOpenModelManager?.();
  };

  const renderModelSelector = () => (
    <div className="relative w-full" ref={modelMenuRef}>
      <Button
        variant="outline"
        onClick={() => setIsModelMenuOpen((prev) => !prev)}
        className={cn(
          "w-full justify-between font-normal h-9",
          selectedOption?.isReady ? "border-primary/20 bg-primary/5" : "",
        )}
      >
        <span className="flex-1 min-w-0 truncate text-left">
          {selectedOption?.label || "Select model"}
        </span>
        <ChevronDown
          className={cn(
            "w-3.5 h-3.5 transition-transform shrink-0 opacity-50",
            isModelMenuOpen && "rotate-180",
          )}
        />
      </Button>

      <AnimatePresence>
        {isModelMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: 6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 6, scale: 0.98 }}
            transition={{ duration: 0.16 }}
            className="absolute left-0 right-0 top-full mt-2 rounded-md border bg-popover text-popover-foreground p-1 shadow-md z-[90]"
          >
            <div className="max-h-64 overflow-y-auto">
              {modelOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => {
                    onSelectModel?.(option.value);
                    setIsModelMenuOpen(false);
                  }}
                  className={cn(
                    "relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 px-2 text-sm outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
                    selectedOption?.value === option.value &&
                      "bg-accent text-accent-foreground",
                  )}
                >
                  <div className="flex flex-col items-start min-w-0 w-full">
                    <span className="truncate w-full text-left font-medium">
                      {option.label}
                    </span>
                    <span
                      className={cn(
                        "mt-1 text-[10px] uppercase tracking-wider font-semibold px-1.5 py-0.5 rounded-sm",
                        getStatusTone(option),
                      )}
                    >
                      {option.statusLabel}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );

  return (
    <div className="grid gap-4 lg:gap-6 xl:grid-cols-[340px,minmax(0,1fr),320px] xl:h-[calc(100dvh-11.75rem)]">
      <div className="rounded-xl border border-[var(--border-muted)] bg-card text-card-foreground shadow-sm p-4 sm:p-5 space-y-4 xl:h-full xl:min-h-0 xl:overflow-y-auto">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="inline-flex items-center gap-2 text-xs text-muted-foreground font-medium uppercase tracking-wider">
              <FileAudio className="w-3.5 h-3.5" />
              Capture
            </div>
            <h2 className="text-sm font-semibold mt-1">Audio Input</h2>
          </div>
          {onOpenModelManager && (
            <Button
              onClick={handleOpenModels}
              variant="outline"
              size="sm"
              className="h-8 gap-1.5 text-xs shadow-sm"
            >
              <Settings2 className="w-4 h-4" />
              Models
            </Button>
          )}
        </div>

        <div className="rounded-xl border border-[var(--border-muted)] bg-muted/30 p-4 space-y-3 shadow-inner">
          <div>
            <div className="text-[11px] text-[var(--text-subtle)] uppercase tracking-wide mb-2">
              Active Model
            </div>
            {modelOptions.length > 0 && renderModelSelector()}
          </div>

          <div className="pt-2 border-t border-[var(--border-muted)]">
            <div
              className={cn(
                "text-xs",
                selectedModelReady
                  ? "text-[var(--text-secondary)]"
                  : "text-amber-500",
              )}
            >
              {selectedModelReady
                ? "Loaded and ready"
                : "Select and load a transcription model"}
            </div>
          </div>
        </div>

        <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5">
          <div className="flex items-center justify-center">
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={cn(
                "h-24 w-24 rounded-full border transition-all duration-150 flex items-center justify-center",
                isRecording
                  ? "bg-[var(--accent-solid)] border-[var(--accent-solid)] text-[var(--text-on-accent)] shadow-[0_0_0_8px_rgba(255,255,255,0.08)]"
                  : "bg-[var(--bg-surface-2)] border-[var(--border-muted)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:border-[var(--border-strong)]",
              )}
              disabled={!selectedModelReady || isProcessing}
            >
              {isRecording ? (
                <MicOff className="w-8 h-8" />
              ) : (
                <Mic className="w-8 h-8" />
              )}
            </button>
          </div>
          <p className="text-center text-xs text-[var(--text-subtle)] mt-3">
            {isRecording
              ? "Recording... click again to stop"
              : "Tap to record from microphone"}
          </p>

          <div className="mt-4">
            <button
              onClick={() => {
                if (!requireReadyModel()) {
                  return;
                }
                fileInputRef.current?.click();
              }}
              className="btn btn-secondary w-full text-sm"
              disabled={!canRunInput}
            >
              <Upload className="w-4 h-4" />
              Upload Audio File
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              onChange={handleFileUpload}
              className="hidden"
            />
          </div>
        </div>

        {audioUrl && (
          <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-2)] p-3">
            <div className="text-xs text-[var(--text-subtle)] mb-2">
              Latest input
            </div>
            <audio src={audioUrl} controls className="w-full h-9" />
          </div>
        )}

        {hasDraft && (
          <button
            onClick={handleReset}
            className="btn btn-ghost w-full text-xs"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Session
          </button>
        )}
      </div>

      <div className="card p-4 sm:p-5 min-h-[460px] lg:min-h-[560px] xl:min-h-0 flex flex-col xl:h-full">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2 min-w-0">
              <FileText className="w-4 h-4 text-muted-foreground" />
              <h3 className="text-sm font-medium">Transcript</h3>
              {isStreaming && (
                <span className="text-[10px] px-1.5 py-0.5 rounded flex items-center gap-1 bg-green-500/10 text-green-500">
                  <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                  Live
                </span>
              )}
              {detectedLanguage && !isStreaming && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground font-medium border border-[var(--border-muted)] shadow-sm">
                  {detectedLanguage}
                </span>
              )}
            </div>
            <p className="text-[11px] text-muted-foreground mt-1">
              Saved automatically to transcription history.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Select
              value={selectedLanguage}
              onValueChange={setSelectedLanguage}
              disabled={isProcessing}
            >
              <SelectTrigger className="h-8 w-[140px] sm:w-[160px] text-xs">
                <SelectValue placeholder="Language" />
              </SelectTrigger>
              <SelectContent>
                {LANGUAGE_OPTIONS.map((language) => (
                  <SelectItem
                    key={language}
                    value={language}
                    className="text-xs"
                  >
                    {language}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <label className="flex items-center gap-2 rounded-md border border-[var(--border-muted)] bg-muted/30 px-2 py-1.5 text-xs text-muted-foreground shadow-sm">
              <Radio className="w-3.5 h-3.5" />
              <span className="font-medium">Stream</span>
              <input
                type="checkbox"
                checked={streamingEnabled}
                onChange={(event) => setStreamingEnabled(event.target.checked)}
                className="app-checkbox w-3.5 h-3.5 disabled:opacity-50 ml-1"
                disabled={isProcessing}
              />
            </label>
            <Button
              onClick={handleCopy}
              variant="outline"
              size="sm"
              className="h-8 w-8 p-0"
              disabled={!transcription || isStreaming}
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
              size="sm"
              className="h-8 w-8 p-0"
              disabled={!transcription || isStreaming}
              title="Download transcript"
            >
              <Download className="w-4 h-4" />
            </Button>
          </div>
        </div>

        <div className="flex-1 rounded-lg border border-[var(--border-muted)] bg-background/50 p-4 sm:p-5 overflow-y-auto shadow-inner">
          {showResult ? (
            <>
              {isProcessing && !transcription ? (
                <div className="h-full flex items-center justify-center text-sm text-muted-foreground gap-2">
                  <Loader2 className="w-4 h-4 animate-spin text-primary" />
                  {isStreaming
                    ? "Streaming transcription..."
                    : "Transcribing..."}
                </div>
              ) : (
                <p className="text-sm text-foreground/90 whitespace-pre-wrap min-h-[2em] font-medium leading-relaxed">
                  {transcription ||
                    (isStreaming ? "Listening for speech..." : "")}
                </p>
              )}
            </>
          ) : (
            <div className="h-full flex items-center justify-center text-center px-6">
              <div className="max-w-xs">
                <FileText className="w-8 h-8 mx-auto mb-3 opacity-20 text-muted-foreground" />
                <p className="text-sm font-medium text-muted-foreground">
                  Record audio or upload a file to start.
                </p>
                <p className="text-xs text-muted-foreground/70 mt-1">
                  The transcript appears live and is stored automatically.
                </p>
              </div>
            </div>
          )}
        </div>

        {processingStats && !isStreaming && (
          <div className="mt-4 pt-3 border-t border-[var(--border-muted)]">
            <GenerationStats stats={processingStats} type="asr" />
          </div>
        )}

        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0, y: 10 }}
              animate={{ opacity: 1, height: "auto", y: 0 }}
              exit={{ opacity: 0, height: 0, y: 10 }}
              className="mt-3 p-3 rounded-lg border border-destructive/20 bg-destructive/10 text-destructive text-xs font-medium flex items-center gap-2"
            >
              <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
              {error}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <aside className="card border-[var(--border-muted)] p-4 sm:p-5 h-[440px] xl:h-full flex flex-col overflow-hidden">
        <div className="flex items-start justify-between gap-3 mb-3">
          <div>
            <div className="inline-flex items-center gap-2 text-xs text-[var(--text-muted)]">
              <History className="w-3.5 h-3.5" />
              History
            </div>
            <h3 className="text-sm font-medium text-[var(--text-primary)] mt-1">
              Transcription History
            </h3>
            <p className="text-xs text-[var(--text-subtle)] mt-1">
              {historyRecords.length}{" "}
              {historyRecords.length === 1 ? "record" : "records"}
            </p>
          </div>
          <button
            onClick={() => void loadHistory()}
            className="btn btn-ghost px-2.5 py-1.5 text-xs"
            disabled={historyLoading}
            title="Refresh history"
          >
            <RefreshCw
              className={cn("w-3.5 h-3.5", historyLoading && "animate-spin")}
            />
            Refresh
          </button>
        </div>

        <div className="mt-1 flex-1 min-h-0 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-2 overflow-y-auto">
          {historyLoading ? (
            <div className="h-full min-h-full flex items-center justify-center gap-2 text-xs text-[var(--text-muted)]">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              Loading history...
            </div>
          ) : historyRecords.length === 0 ? (
            <div className="h-full min-h-full flex items-center justify-center text-center px-3 text-xs text-[var(--text-subtle)]">
              No saved transcriptions yet.
            </div>
          ) : (
            <div className="flex flex-col gap-2.5">
              {historyRecords.map((record) => {
                const isActive = record.id === selectedHistoryRecordId;
                return (
                  <div
                    key={record.id}
                    onClick={() => openHistoryRecord(record.id)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        openHistoryRecord(record.id);
                      }
                    }}
                    role="button"
                    tabIndex={0}
                    className={cn(
                      "group w-full h-[102px] text-left rounded-lg border px-3 py-2.5 transition-colors overflow-hidden cursor-pointer relative focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                      isActive
                        ? "border-[var(--border-strong)] bg-[var(--bg-surface-3)]"
                        : "border-[var(--border-muted)] bg-[var(--bg-surface-2)] hover:border-[var(--border-strong)]",
                    )}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[11px] text-[var(--text-secondary)] truncate">
                        {record.audio_filename ||
                          record.model_id ||
                          "Audio input"}
                      </span>
                      <div className="inline-flex items-center gap-1.5 shrink-0">
                        <span className="text-[10px] text-[var(--text-subtle)]">
                          {formatCreatedAt(record.created_at)}
                        </span>
                        <button
                          onClick={(event) => {
                            event.preventDefault();
                            event.stopPropagation();
                            openDeleteRecordConfirm(record.id);
                          }}
                          className="inline-flex h-6 w-6 items-center justify-center rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-1)] text-[var(--text-subtle)] transition-colors hover:border-[var(--danger-border)] hover:bg-[var(--danger-bg)] hover:text-[var(--danger-text)]"
                          title="Delete record"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                    <p
                      className="text-xs text-[var(--text-primary)] mt-1.5 leading-[1.35]"
                      style={{
                        display: "-webkit-box",
                        WebkitLineClamp: 3,
                        WebkitBoxOrient: "vertical",
                        overflow: "hidden",
                      }}
                    >
                      {record.transcription_preview}
                    </p>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <AnimatePresence>
          {historyError && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="p-2 rounded border text-xs mt-2 bg-[var(--danger-bg)] border-[var(--danger-border)] text-[var(--danger-text)]"
            >
              {historyError}
            </motion.div>
          )}
        </AnimatePresence>
      </aside>

      <AnimatePresence>
        {isHistoryModalOpen && (
          <motion.div
            className="fixed inset-0 z-50 bg-background/80 p-3 backdrop-blur-sm sm:p-6"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={closeHistoryModal}
          >
            <motion.div
              initial={{ y: 18, opacity: 0, scale: 0.985 }}
              animate={{ y: 0, opacity: 1, scale: 1 }}
              exit={{ y: 18, opacity: 0, scale: 0.985 }}
              transition={{ duration: 0.18 }}
              onClick={(event) => event.stopPropagation()}
              className="mx-auto flex max-h-[92vh] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-[var(--border-muted)] bg-card text-card-foreground shadow-2xl"
            >
              <div className="flex items-center justify-between gap-3 border-b border-[var(--border-muted)] px-4 py-3 sm:px-6 bg-muted/20">
                <div className="min-w-0">
                  <p className="text-[11px] uppercase tracking-wide text-muted-foreground font-medium">
                    Transcription Record
                  </p>
                  <h3 className="truncate text-sm font-semibold mt-1">
                    {selectedHistorySummary?.audio_filename ||
                      selectedHistorySummary?.model_id ||
                      "Audio transcription"}
                  </h3>
                  <p className="text-xs text-muted-foreground mt-1">
                    {selectedHistorySummary
                      ? formatCreatedAt(selectedHistorySummary.created_at)
                      : "No record selected"}
                  </p>
                </div>
                <div className="flex items-center gap-1.5">
                  {activeHistoryRecord && (
                    <Button
                      onClick={() =>
                        openDeleteRecordConfirm(activeHistoryRecord.id)
                      }
                      variant="destructive"
                      size="sm"
                      className="gap-1.5 h-8 text-xs shadow-sm"
                      title="Delete this record"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                      Delete
                    </Button>
                  )}
                  <Button
                    onClick={() => openAdjacentHistoryRecord("newer")}
                    variant="outline"
                    size="icon"
                    className="h-8 w-8 shadow-sm"
                    disabled={!canOpenNewerHistory}
                    title="Open newer record"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </Button>
                  <Button
                    onClick={() => openAdjacentHistoryRecord("older")}
                    variant="outline"
                    size="icon"
                    className="h-8 w-8 shadow-sm"
                    disabled={!canOpenOlderHistory}
                    title="Open older record"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </Button>
                  <Button
                    onClick={closeHistoryModal}
                    variant="outline"
                    size="icon"
                    className="h-8 w-8 shadow-sm"
                    title="Close"
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>
              </div>

              <div className="grid flex-1 overflow-hidden lg:grid-cols-[360px,minmax(0,1fr)]">
                <div className="border-b border-[var(--border-muted)] p-4 sm:p-5 lg:border-b-0 lg:border-r bg-muted/10">
                  {selectedHistoryLoading ? (
                    <div className="h-full min-h-[220px] flex items-center justify-center gap-2 text-sm text-muted-foreground">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Loading record...
                    </div>
                  ) : selectedHistoryError ? (
                    <div className="rounded-lg border border-destructive/20 bg-destructive/10 px-3 py-2 text-xs text-destructive font-medium">
                      {selectedHistoryError}
                    </div>
                  ) : activeHistoryRecord ? (
                    <>
                      <div className="flex flex-wrap gap-1.5 mb-5">
                        <span className="rounded-md border border-[var(--border-muted)] bg-card px-2 py-1 text-[11px] text-muted-foreground shadow-sm">
                          {activeHistoryRecord.model_id || "Unknown model"}
                        </span>
                        <span className="rounded-md border border-[var(--border-muted)] bg-card px-2 py-1 text-[11px] text-muted-foreground shadow-sm">
                          {activeHistoryRecord.language || "Unknown language"}
                        </span>
                        <span className="rounded-md border border-[var(--border-muted)] bg-card px-2 py-1 text-[11px] text-muted-foreground shadow-sm">
                          {formatAudioDuration(
                            activeHistoryRecord.duration_secs,
                          )}
                        </span>
                      </div>

                      <audio
                        ref={historyAudioRef}
                        src={selectedHistoryAudioUrl ?? undefined}
                        preload="metadata"
                        onLoadedMetadata={(event) => {
                          const durationSeconds = Number.isFinite(
                            event.currentTarget.duration,
                          )
                            ? event.currentTarget.duration
                            : 0;
                          setHistoryDuration(durationSeconds);
                          setHistoryAudioError(null);
                        }}
                        onTimeUpdate={(event) => {
                          setHistoryCurrentTime(
                            event.currentTarget.currentTime,
                          );
                        }}
                        onPlay={() => setHistoryIsPlaying(true)}
                        onPause={() => setHistoryIsPlaying(false)}
                        onRateChange={(event) =>
                          setHistoryPlaybackRate(
                            event.currentTarget.playbackRate,
                          )
                        }
                        onError={() =>
                          setHistoryAudioError(
                            "Unable to load audio for this transcription.",
                          )
                        }
                        className="hidden"
                      />

                      <div className="rounded-xl border border-[var(--border-muted)] bg-card shadow-sm p-3">
                        <div className="flex items-center gap-2">
                          <Button
                            onClick={() => void toggleHistoryPlayback()}
                            variant="secondary"
                            size="icon"
                            className="h-9 w-9 rounded-full shadow-sm"
                            disabled={!selectedHistoryAudioUrl}
                            title={historyIsPlaying ? "Pause" : "Play"}
                          >
                            {historyIsPlaying ? (
                              <Pause className="w-4 h-4" />
                            ) : (
                              <Play className="w-4 h-4 ml-0.5" />
                            )}
                          </Button>
                          <Button
                            onClick={() => skipHistoryAudio(-10)}
                            variant="ghost"
                            size="icon"
                            className="h-9 w-9 text-muted-foreground hover:text-foreground"
                            disabled={!selectedHistoryAudioUrl}
                            title="Back 10 seconds"
                          >
                            <SkipBack className="w-4 h-4" />
                          </Button>
                          <Button
                            onClick={() => skipHistoryAudio(10)}
                            variant="ghost"
                            size="icon"
                            className="h-9 w-9 text-muted-foreground hover:text-foreground"
                            disabled={!selectedHistoryAudioUrl}
                            title="Forward 10 seconds"
                          >
                            <SkipForward className="w-4 h-4" />
                          </Button>
                          <div className="ml-auto text-[11px] font-medium text-muted-foreground tracking-wide">
                            {formatClockTime(historyCurrentTime)} /{" "}
                            {formatClockTime(historyViewerDuration)}
                          </div>
                        </div>

                        <div className="mt-4">
                          <input
                            type="range"
                            min={0}
                            max={historyViewerDuration || 0}
                            step={0.05}
                            value={Math.min(
                              historyCurrentTime,
                              historyViewerDuration || 0,
                            )}
                            onChange={(event) =>
                              seekHistoryAudio(Number(event.target.value))
                            }
                            className="w-full accent-primary h-1.5 bg-muted rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary"
                            disabled={
                              !selectedHistoryAudioUrl ||
                              historyViewerDuration <= 0
                            }
                          />
                        </div>

                        <div className="mt-4 flex items-center gap-2">
                          <label className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">
                            Speed
                          </label>
                          <select
                            value={historyPlaybackRate}
                            onChange={(event) =>
                              handleHistoryRateChange(
                                Number(event.target.value),
                              )
                            }
                            className="h-8 rounded-md border border-[var(--border-muted)] bg-background px-2 text-xs font-medium text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary shadow-sm"
                          >
                            <option value={0.75}>0.75x</option>
                            <option value={1}>1.0x</option>
                            <option value={1.25}>1.25x</option>
                            <option value={1.5}>1.5x</option>
                            <option value={2}>2.0x</option>
                          </select>
                          {selectedHistoryAudioUrl && (
                            <Button
                              asChild
                              variant="outline"
                              size="sm"
                              className="ml-auto h-8 gap-1.5 text-xs shadow-sm"
                            >
                              <a
                                href={selectedHistoryAudioUrl}
                                download={
                                  activeHistoryRecord.audio_filename ||
                                  `${activeHistoryRecord.id}.wav`
                                }
                              >
                                <Download className="w-3.5 h-3.5" />
                                Audio
                              </a>
                            </Button>
                          )}
                        </div>
                      </div>

                      <AnimatePresence>
                        {historyAudioError && (
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: "auto" }}
                            exit={{ opacity: 0, height: 0 }}
                            className="mt-4 rounded-lg border border-destructive/20 bg-destructive/10 px-3 py-2 text-xs font-medium text-destructive"
                          >
                            {historyAudioError}
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </>
                  ) : (
                    <div className="h-full min-h-[220px] flex items-center justify-center text-sm text-muted-foreground text-center px-4">
                      Select a history record to inspect playback and
                      transcript.
                    </div>
                  )}
                </div>

                <div className="p-4 sm:p-5 flex flex-col min-h-0 bg-background/50">
                  <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                    <h4 className="text-sm font-semibold">Transcript</h4>
                    <div className="flex items-center gap-2">
                      <Button
                        onClick={() => void handleCopyHistoryTranscript()}
                        variant="outline"
                        size="sm"
                        className="h-8 gap-1.5 text-xs shadow-sm"
                        disabled={!activeHistoryRecord?.transcription}
                      >
                        {historyTranscriptCopied ? (
                          <>
                            <Check className="w-3.5 h-3.5 text-green-500" />
                            Copied
                          </>
                        ) : (
                          <>
                            <Copy className="w-3.5 h-3.5" />
                            Copy
                          </>
                        )}
                      </Button>
                      <Button
                        onClick={handleDownloadHistoryTranscript}
                        variant="outline"
                        size="sm"
                        className="h-8 gap-1.5 text-xs shadow-sm"
                        disabled={!activeHistoryRecord?.transcription}
                      >
                        <Download className="w-3.5 h-3.5" />
                        TXT
                      </Button>
                    </div>
                  </div>
                  <div className="flex-1 rounded-xl border border-[var(--border-muted)] bg-card p-4 sm:p-5 overflow-y-auto shadow-inner min-h-[320px]">
                    {selectedHistoryLoading ? (
                      <div className="h-full flex items-center justify-center gap-2 text-sm text-muted-foreground">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Loading transcript...
                      </div>
                    ) : (
                      <p className="text-sm text-foreground/90 whitespace-pre-wrap leading-relaxed font-medium">
                        {activeHistoryRecord?.transcription || (
                          <span className="text-muted-foreground italic">
                            No transcript text available for this record.
                          </span>
                        )}
                      </p>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {deleteTargetRecord && (
          <motion.div
            className="fixed inset-0 z-[60] bg-background/80 p-4 backdrop-blur-sm flex items-center justify-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={closeDeleteRecordConfirm}
          >
            <motion.div
              initial={{ y: 10, opacity: 0, scale: 0.98 }}
              animate={{ y: 0, opacity: 1, scale: 1 }}
              exit={{ y: 10, opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.16 }}
              className="w-full max-w-md rounded-xl border border-[var(--border-muted)] bg-card text-card-foreground p-5 shadow-2xl"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-start gap-4">
                <div className="mt-0.5 rounded-full border border-destructive/20 bg-destructive/10 p-2.5 text-destructive">
                  <AlertTriangle className="h-5 w-5" />
                </div>
                <div className="min-w-0 flex-1">
                  <h3 className="text-base font-semibold">
                    Delete transcription?
                  </h3>
                  <p className="mt-1.5 text-sm text-muted-foreground leading-relaxed">
                    This permanently removes the saved audio and transcript from
                    history.
                  </p>
                  <div className="mt-3 rounded-md bg-muted/50 p-2.5 border border-[var(--border-muted)]">
                    <p className="truncate text-xs font-medium text-muted-foreground">
                      {deleteTargetRecord.audio_filename ||
                        deleteTargetRecord.model_id ||
                        deleteTargetRecord.id}
                    </p>
                  </div>
                </div>
              </div>

              <AnimatePresence>
                {deleteRecordError && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-4 rounded-md border border-destructive/20 bg-destructive/10 px-3 py-2 text-xs font-medium text-destructive"
                  >
                    {deleteRecordError}
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="mt-6 flex items-center justify-end gap-3">
                <Button
                  onClick={closeDeleteRecordConfirm}
                  variant="outline"
                  disabled={deleteRecordPending}
                >
                  Cancel
                </Button>
                <Button
                  onClick={() => void confirmDeleteRecord()}
                  variant="destructive"
                  disabled={deleteRecordPending}
                  className="gap-2"
                >
                  {deleteRecordPending ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Trash2 className="h-4 w-4" />
                  )}
                  Delete record
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
