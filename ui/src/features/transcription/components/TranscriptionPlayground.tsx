import { useCallback, useEffect, useRef, useState, useMemo } from "react";
import { createPortal } from "react-dom";
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
  Radio,
  RotateCcw,
  Settings2,
  Square,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { RouteHistoryDrawer } from "@/components/RouteHistoryDrawer";
import {
  LANGUAGE_OPTIONS,
  LIVE_MIC_PCM_FRAME_SIZE,
  type ProcessAudioOptions,
  type TranscriptionPlaygroundProps,
  buildTranscriptionRealtimeWebSocketUrl,
  encodeLiveMicChunk,
  encodeTranscriptionRealtimePcm16Frame,
  formatAudioDuration,
  formatCreatedAt,
  isTranscriptionRealtimeServerEvent,
  summarizeRecord,
  transcodeToWav,
} from "@/features/transcription/playground/support";
import {
  api,
  type TranscriptionRecord,
  type TranscriptionRecordSummary,
} from "@/api";
import { ASRStats, GenerationStats } from "@/components/GenerationStats";
import { MiniWaveform } from "@/components/ui/Waveform";
import { TranscriptionExportDialog } from "@/features/transcription/components/TranscriptionExportDialog";
import { TranscriptionReviewWorkspace } from "@/features/transcription/components/TranscriptionReviewWorkspace";
import { formatTranscriptionText } from "@/features/transcription/transcript";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { StatusBadge } from "@/components/ui/status-badge";
import { useWorkspaceShortcuts } from "@/hooks/useWorkspaceShortcuts";
import { RouteModelSelect } from "@/components/RouteModelSelect";

export function TranscriptionPlayground({
  selectedModel,
  selectedModelReady = false,
  modelOptions = [],
  onSelectModel,
  onOpenModelManager,
  onModelRequired,
  timestampAlignerModelId = null,
  timestampAlignerReady = false,
  onTimestampAlignerRequired,
  historyActionContainer,
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
  const [includeTimestamps, setIncludeTimestamps] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState("English");
  const [currentOutputRecord, setCurrentOutputRecord] =
    useState<TranscriptionRecord | null>(null);
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
  const [isHistoryDrawerOpen, setIsHistoryDrawerOpen] = useState(false);
  const [isHistoryModalOpen, setIsHistoryModalOpen] = useState(false);
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

  const requireReadyModel = useCallback(() => {
    if (!selectedModel || !selectedModelReady) {
      onModelRequired();
      return false;
    }
    return true;
  }, [selectedModel, selectedModelReady, onModelRequired]);

  const requireTimestampAligner = useCallback(() => {
    if (!includeTimestamps) {
      return true;
    }
    if (timestampAlignerModelId && timestampAlignerReady) {
      return true;
    }
    onTimestampAlignerRequired?.();
    setError("Load the timestamp aligner model to enable timestamps.");
    return false;
  }, [
    includeTimestamps,
    onTimestampAlignerRequired,
    timestampAlignerModelId,
    timestampAlignerReady,
  ]);

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

  const handleHistoryDrawerOpenChange = useCallback(
    (nextOpen: boolean) => {
      if (!nextOpen && deleteTargetRecordId) {
        return;
      }
      setIsHistoryDrawerOpen(nextOpen);
    },
    [deleteTargetRecordId],
  );

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
      if (!requireTimestampAligner()) {
        return;
      }

      setIsProcessing(true);
      setError(null);
      setProcessingStats(null);
      setCurrentOutputRecord(null);
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
              aligner_model_id: includeTimestamps
                ? timestampAlignerModelId || undefined
                : undefined,
              language: selectedLanguage,
              include_timestamps: includeTimestamps,
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
                setCurrentOutputRecord(record);
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
            aligner_model_id: includeTimestamps
              ? timestampAlignerModelId || undefined
              : undefined,
            language: selectedLanguage,
            include_timestamps: includeTimestamps,
          });

          setTranscription(record.transcription);
          setDetectedLanguage(record.language || null);
          setProcessingStats({
            processing_time_ms: record.processing_time_ms,
            audio_duration_secs: record.duration_secs,
            rtf: record.rtf,
          });
          setCurrentOutputRecord(record);
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
      requireTimestampAligner,
      includeTimestamps,
      selectedModel,
      selectedLanguage,
      streamingEnabled,
      timestampAlignerModelId,
    ],
  );

  const startRecording = useCallback(async () => {
    if (!requireReadyModel()) {
      return;
    }
    if (!requireTimestampAligner()) {
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
      setCurrentOutputRecord(null);
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
        for (
          let channelIndex = 0;
          channelIndex < channelCount;
          channelIndex += 1
        ) {
          const channel = inputBuffer.getChannelData(channelIndex);
          for (
            let sampleIndex = 0;
            sampleIndex < frameCount;
            sampleIndex += 1
          ) {
            mono[sampleIndex] += (channel[sampleIndex] ?? 0) / channelCount;
          }
        }

        const pcm16 = encodeLiveMicChunk(mono);
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
    requireTimestampAligner,
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
    setCurrentOutputRecord(null);
    setAudioUrl(null);
    setError(null);
    setProcessingStats(null);
    setIsStreaming(false);
    setIsProcessing(false);
  };

  const workspaceShortcuts = useMemo(
    () => [
      {
        key: "Enter",
        metaKey: true,
        enabled:
          !isHistoryModalOpen &&
          !deleteTargetRecordId &&
          selectedModelReady &&
          !isProcessing,
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
        enabled: !isHistoryModalOpen && isRecording,
        action: stopRecording,
      },
      {
        key: "Escape",
        shiftKey: true,
        enabled: !isHistoryModalOpen && Boolean(transcription || audioUrl || error),
        action: handleReset,
      },
    ],
    [
      audioUrl,
      deleteTargetRecordId,
      error,
      handleReset,
      isHistoryModalOpen,
      isProcessing,
      isRecording,
      selectedModelReady,
      startRecording,
      stopRecording,
      transcription,
    ],
  );

  useWorkspaceShortcuts(workspaceShortcuts);

  const handleIncludeTimestampsChange = useCallback(
    (nextValue: boolean) => {
      if (!nextValue) {
        setIncludeTimestamps(false);
        return;
      }

      if (timestampAlignerModelId && timestampAlignerReady) {
        setIncludeTimestamps(true);
        setStreamingEnabled(false);
        return;
      }

      onTimestampAlignerRequired?.();
      setError("Load the timestamp aligner model to enable timestamps.");
    },
    [
      onTimestampAlignerRequired,
      timestampAlignerModelId,
      timestampAlignerReady,
    ],
  );

  const handleStreamingEnabledChange = useCallback((nextValue: boolean) => {
    setStreamingEnabled(nextValue);
    if (nextValue) {
      setIncludeTimestamps(false);
    }
  }, []);

  const handleCopy = async () => {
    const exportText = currentOutputExportText;
    if (!exportText) {
      return;
    }
    await navigator.clipboard.writeText(exportText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  useEffect(() => {
    return () => {
      if (streamAbortRef.current) {
        streamAbortRef.current.abort();
      }
      abortLiveMicStream();
    };
  }, [abortLiveMicStream]);

  const outputRecord = useMemo<
    Pick<
      TranscriptionRecord,
      | "created_at"
      | "model_id"
      | "id"
      | "aligner_model_id"
      | "audio_filename"
      | "duration_secs"
      | "language"
      | "transcription"
      | "segments"
      | "words"
    > | null
  >(() => {
    if (currentOutputRecord) {
      return currentOutputRecord;
    }
    if (!transcription && !isStreaming && !isProcessing) {
      return null;
    }
    return {
      created_at: Date.now(),
      model_id: selectedModel,
      id: "transcription-draft",
      aligner_model_id: includeTimestamps ? timestampAlignerModelId : null,
      audio_filename: null,
      duration_secs: processingStats?.audio_duration_secs ?? null,
      language: detectedLanguage ?? selectedLanguage,
      transcription,
      segments: [],
      words: [],
    };
  }, [
    currentOutputRecord,
    detectedLanguage,
    includeTimestamps,
    isProcessing,
    isStreaming,
    processingStats?.audio_duration_secs,
    selectedLanguage,
    selectedModel,
    timestampAlignerModelId,
    transcription,
  ]);
  const currentOutputExportText = useMemo(() => {
    return formatTranscriptionText(outputRecord);
  }, [outputRecord]);
  const showResult = Boolean(outputRecord || isStreaming || isProcessing);
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
  const activeHistoryExportText = useMemo(
    () => formatTranscriptionText(activeHistoryRecord),
    [activeHistoryRecord],
  );
  const activeHistoryHasTimestamps = useMemo(
    () =>
      Boolean(
        activeHistoryRecord &&
          ((activeHistoryRecord.segments ?? []).length > 0 ||
            (activeHistoryRecord.words ?? []).length > 0),
      ),
    [activeHistoryRecord],
  );
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

  const handleCopyHistoryTranscript = useCallback(async () => {
    if (!activeHistoryExportText) {
      return;
    }
    await navigator.clipboard.writeText(activeHistoryExportText);
    setHistoryTranscriptCopied(true);
    window.setTimeout(() => setHistoryTranscriptCopied(false), 1800);
  }, [activeHistoryExportText]);

  useEffect(() => {
    setHistoryTranscriptCopied(false);
  }, [selectedHistoryRecordId]);

  const handleOpenModels = () => {
    onOpenModelManager?.();
  };

  const renderModelSelector = () => (
    <RouteModelSelect
      value={selectedModel}
      options={modelOptions}
      onSelect={onSelectModel}
      className="w-full"
    />
  );

  const historyDrawer = (
    <RouteHistoryDrawer
      title="Transcriptions"
      countLabel={`${historyRecords.length} ${historyRecords.length === 1 ? "record" : "records"}`}
      triggerCount={historyRecords.length}
      open={isHistoryDrawerOpen}
      onOpenChange={handleHistoryDrawerOpenChange}
    >
      {({ close }) => (
        <>
          <div className="app-sidebar-list scrollbar-thin">
            {historyLoading ? (
              <div className="app-sidebar-loading">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Loading history...
              </div>
            ) : historyRecords.length === 0 ? (
              <div className="flex flex-col items-center justify-center p-6 text-center opacity-60">
                <History className="mb-3 h-10 w-10 text-muted-foreground" />
                <p className="text-sm font-medium text-muted-foreground">
                  No history yet
                </p>
                <p className="mt-1 text-xs text-muted-foreground/70">
                  Transcriptions will appear here
                </p>
              </div>
            ) : (
              <div className="flex flex-col gap-2.5">
                {historyRecords.map((record) => {
                  const isActive = record.id === selectedHistoryRecordId;
                  return (
                    <div
                      key={record.id}
                      onClick={() => {
                        openHistoryRecord(record.id);
                        close();
                      }}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          openHistoryRecord(record.id);
                          close();
                        }
                      }}
                      role="button"
                      tabIndex={0}
                      className={cn(
                        "group app-sidebar-row relative focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-[var(--bg-surface-1)]",
                        isActive
                          ? "app-sidebar-row-active"
                          : "app-sidebar-row-idle",
                      )}
                    >
                      <div className="mb-1.5 flex items-center justify-between gap-2">
                        <span className="app-sidebar-row-label truncate font-medium transition-colors group-hover:text-primary">
                          {record.audio_filename ||
                            record.model_id ||
                            "Audio input"}
                        </span>
                        <div className="inline-flex items-center gap-1.5 shrink-0 opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100">
                          <span className="app-sidebar-row-meta mr-1 hidden group-hover:hidden sm:block">
                            {formatCreatedAt(record.created_at)}
                          </span>
                          <button
                            onPointerDown={(event) => {
                              event.stopPropagation();
                            }}
                            onClick={(event) => {
                              event.preventDefault();
                              event.stopPropagation();
                              openDeleteRecordConfirm(record.id);
                            }}
                            className="rounded-md p-1.5 text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive focus:opacity-100"
                            title="Delete record"
                            aria-label={`Delete ${record.audio_filename || record.model_id || "audio input"}`}
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                        <span className="app-sidebar-row-meta group-hover:hidden sm:hidden">
                          {formatCreatedAt(record.created_at)}
                        </span>
                      </div>
                      <div className="mt-1 mb-1.5 flex items-center justify-between text-[10px] font-medium uppercase tracking-wide opacity-60">
                        <span>{formatCreatedAt(record.created_at)}</span>
                        {record.duration_secs && (
                          <span>{formatAudioDuration(record.duration_secs)}</span>
                        )}
                      </div>
                      <p
                        className="app-sidebar-row-preview text-[13px] leading-snug"
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
                className="mt-1 flex items-start gap-2 rounded-lg border bg-[var(--danger-bg)] p-3 text-sm text-[var(--danger-text)] border-[var(--danger-border)]"
              >
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                <p>{historyError}</p>
              </motion.div>
            )}
          </AnimatePresence>
        </>
      )}
    </RouteHistoryDrawer>
  );

  return (
    <div className="grid gap-5 lg:gap-6 xl:grid-cols-[340px,minmax(0,1fr)] xl:h-[calc(100dvh-11.75rem)]">
      <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-4 sm:p-5 space-y-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="inline-flex items-center gap-2 text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
              <FileAudio className="w-3.5 h-3.5" />
              Capture
            </div>
            <h2 className="mt-1.5 text-base font-semibold text-[var(--text-primary)]">
              Audio Input
            </h2>
          </div>
          {onOpenModelManager ? (
            <Button
              onClick={handleOpenModels}
              variant="outline"
              size="sm"
              className="h-8 gap-1.5 text-xs bg-[var(--bg-surface-1)] border-[var(--border-muted)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-surface-2)] shadow-sm"
            >
              <Settings2 className="w-4 h-4" />
              Models
            </Button>
          ) : null}
        </div>

        <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4 space-y-3">
          <div>
            <div className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">
              Active Model
            </div>
            {modelOptions.length > 0 ? (
              <div className="mt-3">{renderModelSelector()}</div>
            ) : null}
          </div>

          <div className="pt-2 border-t border-[var(--border-muted)]">
            <StatusBadge tone={selectedModelReady ? "success" : "warning"}>
              {selectedModelReady
                ? "Loaded and ready"
                : "Select and load a transcription model"}
            </StatusBadge>
          </div>
        </div>

        <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5">
          <div className="flex flex-col items-center">
            <button
              onClick={() => {
                if (isRecording) {
                  stopRecording();
                } else {
                  void startRecording();
                }
              }}
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
                <div className="relative flex items-center justify-center">
                  <div
                    className="absolute inset-0 rounded-full bg-red-500/20 animate-ping"
                    style={{ animationDuration: "1.5s" }}
                  />
                  <div
                    className="absolute inset-[-10px] rounded-full bg-red-500/10 animate-ping"
                    style={{ animationDuration: "2s" }}
                  />
                  <Square className="w-10 h-10 text-white fill-current relative z-10" />
                </div>
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
                onClick={() => {
                  if (!requireReadyModel()) return;
                  fileInputRef.current?.click();
                }}
                className={cn(
                  "mt-4 flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 transition-all duration-200 cursor-pointer group",
                  selectedModelReady && !isProcessing
                    ? "border-[var(--border-strong)] hover:border-primary hover:bg-[var(--bg-surface-2)] bg-[var(--bg-surface-1)] hover:shadow-sm"
                    : "border-[var(--border-muted)] bg-[var(--bg-surface-1)] opacity-50 cursor-not-allowed",
                )}
              >
                <div className="p-3 bg-background rounded-full mb-3 shadow-sm group-hover:scale-105 transition-transform duration-200 border border-[var(--border-muted)]">
                  <Upload className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors" />
                </div>
                <p className="text-sm font-medium text-[var(--text-primary)] group-hover:text-primary transition-colors">
                  Upload audio file
                </p>
                <p className="text-xs text-[var(--text-muted)] mt-1.5">
                  WAV, MP3, M4A, AAC
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*"
                  onChange={handleFileUpload}
                  className="hidden"
                  disabled={!selectedModelReady || isProcessing}
                />
              </div>
            </div>
          </div>
        </div>

        {hasDraft ? (
          <Button
            onClick={handleReset}
            variant="ghost"
            size="sm"
            className="w-full h-9 gap-2 text-xs border border-transparent hover:border-[var(--border-muted)] bg-transparent hover:bg-[var(--bg-surface-1)]"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Session
          </Button>
        ) : null}
      </div>

      <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] flex flex-col min-h-[460px] lg:min-h-[560px] xl:h-full xl:min-h-0 overflow-hidden">
        <div className="px-4 sm:px-5 py-4 border-b border-[var(--border-muted)] flex items-center justify-between gap-3 bg-[var(--bg-surface-1)]">
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-[var(--text-primary)]">
              Transcript
            </h3>
          </div>
          <div className="flex flex-wrap items-center justify-end gap-2">
            {isStreaming ? (
              <StatusBadge tone="success">Live</StatusBadge>
            ) : detectedLanguage ? (
              <StatusBadge>{detectedLanguage}</StatusBadge>
            ) : null}
            <Select
              value={selectedLanguage}
              onValueChange={setSelectedLanguage}
              disabled={isProcessing}
            >
              <SelectTrigger className="h-9 w-[140px] sm:w-[160px] text-xs bg-[var(--bg-surface-1)]">
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
            <label className="flex items-center gap-2 rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2 text-xs text-[var(--text-muted)]">
              <span className="font-medium">Timestamps</span>
              <input
                type="checkbox"
                checked={includeTimestamps}
                onChange={(event) =>
                  handleIncludeTimestampsChange(event.target.checked)
                }
                className="app-checkbox h-3.5 w-3.5 disabled:opacity-50"
                disabled={isProcessing}
              />
            </label>
            <label className="flex items-center gap-2 rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2 text-xs text-[var(--text-muted)]">
              <Radio className="w-3.5 h-3.5" />
              <span className="font-medium">Stream</span>
              <input
                type="checkbox"
                checked={streamingEnabled}
                onChange={(event) =>
                  handleStreamingEnabledChange(event.target.checked)
                }
                className="app-checkbox w-3.5 h-3.5 disabled:opacity-50 ml-1"
                disabled={isProcessing}
              />
            </label>
            <Button
              onClick={handleCopy}
              variant="outline"
              size="icon"
              className="h-9 w-9 bg-[var(--bg-surface-1)] border-[var(--border-muted)] text-[var(--text-secondary)] hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)]"
              disabled={!currentOutputExportText || isStreaming}
              title="Copy transcript"
            >
              {copied ? (
                <Check className="w-4 h-4 text-green-500" />
              ) : (
                <Copy className="w-4 h-4" />
              )}
            </Button>
            <TranscriptionExportDialog record={outputRecord}>
              <Button
                variant="outline"
                size="icon"
                className="h-9 w-9 bg-[var(--bg-surface-1)] border-[var(--border-muted)] text-[var(--text-secondary)] hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)]"
                disabled={!currentOutputExportText || isStreaming}
                title="Export transcript"
              >
                <Download className="w-4 h-4" />
              </Button>
            </TranscriptionExportDialog>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 sm:p-6 bg-[var(--bg-surface-0)] scrollbar-thin">
          {isProcessing && !transcription ? (
            <div className="h-full flex flex-col items-center justify-center text-sm font-medium text-[var(--text-muted)] gap-3">
              <Loader2 className="w-5 h-5 animate-spin text-[var(--text-primary)]" />
              {isStreaming
                ? "Streaming transcription..."
                : "Transcribing audio..."}
            </div>
          ) : showResult ? (
            <div className="space-y-4">
              {isStreaming ? (
                <div className="rounded-xl border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] px-4 py-3 text-sm text-[var(--status-warning-text)] flex items-center gap-3">
                  <MiniWaveform isActive={true} />
                  <span>Listening for speech...</span>
                </div>
              ) : null}

              <TranscriptionReviewWorkspace
                record={outputRecord}
                audioUrl={audioUrl}
                emptyMessage="Record audio or upload a file to start."
              />

              {processingStats && !isStreaming ? (
                <GenerationStats stats={processingStats} type="asr" />
              ) : null}
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-center px-6">
              <div className="max-w-sm">
                <div className="w-16 h-16 rounded-full bg-[var(--bg-surface-2)] flex items-center justify-center mx-auto mb-4 border border-[var(--border-muted)]">
                  <FileText className="w-8 h-8 text-[var(--text-subtle)]" />
                </div>
                <p className="text-base font-semibold text-[var(--text-secondary)] mb-2">
                  Ready to transcribe
                </p>
                <p className="text-sm text-[var(--text-muted)] leading-relaxed">
                  Record audio from your microphone or upload an audio file to
                  start transcription. The transcript will appear here.
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

      <div className="xl:col-start-2 text-xs text-[var(--text-muted)]">
        Shortcut: <span className="app-kbd">Ctrl/Cmd + Enter</span> start or stop capture, <span className="app-kbd">Esc</span> stop recording, <span className="app-kbd">Shift + Esc</span> reset.
      </div>

      {historyActionContainer === undefined
        ? historyDrawer
        : historyActionContainer
          ? createPortal(historyDrawer, historyActionContainer)
          : null}

      <AnimatePresence>
        {isHistoryModalOpen && (
          <motion.div
            className="fixed inset-0 z-50 bg-black/60 p-4 backdrop-blur-sm sm:p-6 flex items-center justify-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={closeHistoryModal}
          >
            <motion.div
              initial={{ y: 20, opacity: 0, scale: 0.95 }}
              animate={{ y: 0, opacity: 1, scale: 1 }}
              exit={{ y: 20, opacity: 0, scale: 0.95 }}
              transition={{ type: "spring", duration: 0.5, bounce: 0.3 }}
              onClick={(event) => event.stopPropagation()}
              className="mx-auto flex max-h-[92vh] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-[var(--border-strong)] bg-[var(--bg-surface-0)] shadow-2xl"
            >
              <div className="flex items-center justify-between gap-4 border-b border-[var(--border-muted)] px-5 py-4 sm:px-6 bg-[var(--bg-surface-1)]">
                <div className="min-w-0 flex-1">
                  <p className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-subtle)]">
                    Transcription Record
                  </p>
                  <div className="mt-1 flex items-center gap-3">
                    <h2 className="truncate text-[1.95rem] font-semibold leading-none tracking-[-0.03em] text-[var(--text-primary)]">
                      {selectedHistorySummary?.audio_filename ||
                        selectedHistorySummary?.model_id ||
                        "Transcription transcript"}
                    </h2>
                  </div>
                  <div className="mt-2.5 flex flex-wrap items-center gap-2">
                    <span className="text-xs text-[var(--text-muted)]">
                      {selectedHistorySummary
                        ? formatCreatedAt(selectedHistorySummary.created_at)
                        : "No record selected"}
                    </span>
                    {activeHistoryRecord ? (
                      <>
                        <span className="text-[var(--text-subtle)]">•</span>
                        <span className="inline-flex items-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-0.5 text-[10px] font-medium tracking-[0.08em] text-[var(--text-secondary)]">
                          {formatAudioDuration(activeHistoryRecord.duration_secs)}
                        </span>
                        <span className="inline-flex items-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-0.5 text-[10px] font-medium tracking-[0.08em] text-[var(--text-secondary)]">
                          {activeHistoryRecord.language || "Unknown language"}
                        </span>
                        <span className="inline-flex items-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-0.5 text-[10px] font-medium tracking-[0.08em] text-[var(--text-secondary)]">
                          {activeHistoryHasTimestamps
                            ? "Timed transcript"
                            : "Plain transcript"}
                        </span>
                        <span className="inline-flex max-w-[220px] items-center truncate rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-0.5 text-[10px] font-medium tracking-[0.02em] text-[var(--text-secondary)]">
                          {activeHistoryRecord.model_id || "Unknown model"}
                        </span>
                      </>
                    ) : null}
                  </div>
                </div>
                <div className="flex items-center gap-2 shrink-0 self-start">
                  {activeHistoryRecord && (
                    <button
                      onClick={() =>
                        openDeleteRecordConfirm(activeHistoryRecord.id)
                      }
                      className="inline-flex h-8 items-center gap-1 rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 text-[11px] font-medium text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)]"
                      title="Delete this record"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                      Delete
                    </button>
                  )}
                  <button
                    onClick={() => openAdjacentHistoryRecord("newer")}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] text-[var(--text-muted)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] disabled:opacity-40"
                    disabled={!canOpenNewerHistory}
                    title="Open newer record"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => openAdjacentHistoryRecord("older")}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] text-[var(--text-muted)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] disabled:opacity-40"
                    disabled={!canOpenOlderHistory}
                    title="Open older record"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                  <button
                    onClick={closeHistoryModal}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-muted)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]"
                    title="Close"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="flex-1 overflow-y-auto flex flex-col">
                {selectedHistoryLoading ? (
                  <div className="h-full min-h-[220px] flex items-center justify-center gap-2 text-sm text-[var(--text-muted)]">
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Loading record...
                  </div>
                ) : selectedHistoryError ? (
                  <div className="p-4 sm:p-5">
                    <div className="rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]">
                      {selectedHistoryError}
                    </div>
                  </div>
                ) : activeHistoryRecord ? (
                  <div className="p-3 sm:p-4 space-y-3">
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => void handleCopyHistoryTranscript()}
                          className="inline-flex h-8 items-center gap-1.5 rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 text-[12px] font-medium text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)] disabled:opacity-45"
                          disabled={!activeHistoryExportText}
                        >
                          {historyTranscriptCopied ? (
                            <>
                              <Check className="w-3.5 h-3.5" />
                              Copied
                            </>
                          ) : (
                            <>
                              <Copy className="w-3.5 h-3.5" />
                              Copy
                            </>
                          )}
                        </button>
                        <TranscriptionExportDialog record={activeHistoryRecord}>
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-8 rounded-full border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 text-[12px] font-medium text-[var(--text-secondary)] hover:bg-[var(--bg-surface-2)]"
                            disabled={!activeHistoryExportText}
                          >
                            <Download className="mr-1.5 w-3.5 h-3.5" />
                            Export
                          </Button>
                        </TranscriptionExportDialog>
                      </div>
                      <p className="text-[11px] text-[var(--text-subtle)]">
                        Playback, transcript review, and export for this saved transcription.
                      </p>
                    </div>

                    <TranscriptionReviewWorkspace
                      record={activeHistoryRecord}
                      audioUrl={selectedHistoryAudioUrl}
                      loading={selectedHistoryLoading}
                      emptyMessage="No transcript text available for this record."
                    />
                  </div>
                ) : (
                  <div className="h-full min-h-[220px] flex items-center justify-center text-sm text-[var(--text-subtle)] text-center">
                    Select a history record to inspect playback and transcript.
                  </div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <Dialog
        open={!!deleteTargetRecord}
        onOpenChange={(open) => {
          if (!open) {
            closeDeleteRecordConfirm();
          }
        }}
      >
        {deleteTargetRecord ? (
          <DialogContent className="max-w-md border-[var(--border-strong)] bg-[var(--bg-surface-0)] p-6">
            <DialogTitle className="sr-only">Delete transcription?</DialogTitle>
            <div className="flex items-start gap-4">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-[var(--danger-bg)] text-[var(--danger-text)]">
                <AlertTriangle className="h-5 w-5" />
              </div>
              <div className="min-w-0 flex-1">
                <h3 className="text-base font-semibold text-[var(--text-primary)]">
                  Delete transcription?
                </h3>
                <DialogDescription className="mt-1.5 text-sm leading-relaxed text-[var(--text-muted)]">
                  This permanently removes the saved audio and transcript from
                  history. This action cannot be undone.
                </DialogDescription>
                <div className="mt-4 rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3">
                  <p className="truncate text-xs font-medium text-[var(--text-secondary)]">
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
                  initial={{ opacity: 0, height: 0, marginTop: 0 }}
                  animate={{ opacity: 1, height: "auto", marginTop: 16 }}
                  exit={{ opacity: 0, height: 0, marginTop: 0 }}
                  className="overflow-hidden"
                >
                  <div className="flex items-start gap-2 rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] px-4 py-3 text-sm text-[var(--danger-text)]">
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                    <p>{deleteRecordError}</p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="mt-6 flex items-center justify-end gap-3 border-t border-[var(--border-muted)] pt-5">
              <Button
                onClick={closeDeleteRecordConfirm}
                variant="outline"
                disabled={deleteRecordPending}
                className="h-10 border-[var(--border-muted)]"
              >
                Cancel
              </Button>
              <Button
                onClick={() => void confirmDeleteRecord()}
                variant="destructive"
                disabled={deleteRecordPending}
                className="h-10 gap-2"
              >
                {deleteRecordPending ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Deleting...
                  </>
                ) : (
                  <>
                    <Trash2 className="h-4 w-4" />
                    Delete record
                  </>
                )}
              </Button>
            </div>
          </DialogContent>
        ) : null}
      </Dialog>
    </div>
  );
}
