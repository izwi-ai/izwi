import { useCallback, useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertCircle,
  AlertTriangle,
  BookmarkPlus,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Copy,
  Download,
  Loader2,
  Pause,
  Play,
  RefreshCw,
  SkipBack,
  SkipForward,
  Trash2,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { RouteHistoryDrawer } from "@/components/RouteHistoryDrawer";
import {
  api,
  type SpeechHistoryRecord,
  type SpeechHistoryRecordSummary,
  type SpeechHistoryRoute,
} from "../api";
import { useDownloadIndicator } from "../utils/useDownloadIndicator";
import { blobToBase64Payload } from "../utils/audioBase64";

interface SpeechHistoryPanelProps {
  route: SpeechHistoryRoute;
  title: string;
  emptyMessage: string;
  latestRecord?: SpeechHistoryRecord | null;
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

function routeKindFor(
  route: SpeechHistoryRoute,
): SpeechHistoryRecordSummary["route_kind"] {
  switch (route) {
    case "text-to-speech":
      return "text_to_speech";
    case "voice-design":
      return "voice_design";
    case "voice-cloning":
      return "voice_cloning";
    default:
      return "text_to_speech";
  }
}

function buildInputPreview(text: string, maxChars = 180): string {
  const normalized = text.trim().replace(/\s+/g, " ");
  if (!normalized) {
    return "No text input";
  }
  if (normalized.length <= maxChars) {
    return normalized;
  }
  return `${normalized.slice(0, maxChars)}...`;
}

function summarizeRecord(
  route: SpeechHistoryRoute,
  record: SpeechHistoryRecord,
): SpeechHistoryRecordSummary {
  return {
    id: record.id,
    created_at: record.created_at,
    route_kind: routeKindFor(route),
    model_id: record.model_id,
    speaker: record.speaker,
    language: record.language,
    input_preview: buildInputPreview(record.input_text),
    input_chars: Array.from(record.input_text ?? "").length,
    generation_time_ms: record.generation_time_ms,
    audio_duration_secs: record.audio_duration_secs,
    rtf: record.rtf,
    tokens_generated: record.tokens_generated,
    audio_mime_type: record.audio_mime_type,
    audio_filename: record.audio_filename,
  };
}

export function SpeechHistoryPanel({
  route,
  title,
  emptyMessage,
  latestRecord = null,
}: SpeechHistoryPanelProps) {
  const [records, setRecords] = useState<SpeechHistoryRecordSummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [selectedRecordId, setSelectedRecordId] = useState<string | null>(null);
  const [selectedRecord, setSelectedRecord] =
    useState<SpeechHistoryRecord | null>(null);
  const [selectedLoading, setSelectedLoading] = useState(false);
  const [selectedError, setSelectedError] = useState<string | null>(null);
  const [isHistoryModalOpen, setIsHistoryModalOpen] = useState(false);
  const [historyCurrentTime, setHistoryCurrentTime] = useState(0);
  const [historyDuration, setHistoryDuration] = useState(0);
  const [historyIsPlaying, setHistoryIsPlaying] = useState(false);
  const [historyPlaybackRate, setHistoryPlaybackRate] = useState(1);
  const [historyAudioError, setHistoryAudioError] = useState<string | null>(
    null,
  );
  const [copiedInput, setCopiedInput] = useState(false);
  const [deleteTargetRecordId, setDeleteTargetRecordId] = useState<
    string | null
  >(null);
  const [deleteRecordPending, setDeleteRecordPending] = useState(false);
  const [deleteRecordError, setDeleteRecordError] = useState<string | null>(
    null,
  );
  const [historyVoiceName, setHistoryVoiceName] = useState("");
  const [historyVoiceReferenceText, setHistoryVoiceReferenceText] =
    useState("");
  const [historyVoiceSaving, setHistoryVoiceSaving] = useState(false);
  const [historyVoiceStatus, setHistoryVoiceStatus] = useState<{
    tone: "success" | "error";
    message: string;
  } | null>(null);
  const {
    downloadState,
    downloadMessage,
    isDownloading,
    beginDownload,
    completeDownload,
    failDownload,
  } = useDownloadIndicator();

  const mergeHistorySummary = useCallback(
    (summary: SpeechHistoryRecordSummary) => {
      setRecords((previous) => {
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
      const nextRecords = await api.listSpeechHistoryRecords(route);
      setRecords(nextRecords);
      setSelectedRecordId((current) => {
        if (current && nextRecords.some((item) => item.id === current)) {
          return current;
        }
        return nextRecords[0]?.id ?? null;
      });
    } catch (err) {
      setHistoryError(
        err instanceof Error ? err.message : "Failed to load speech history.",
      );
    } finally {
      setHistoryLoading(false);
    }
  }, [route]);

  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);

  useEffect(() => {
    if (!selectedRecordId) {
      setSelectedRecord(null);
      setSelectedError(null);
      return;
    }

    if (selectedRecord?.id === selectedRecordId) {
      return;
    }

    let cancelled = false;
    setSelectedLoading(true);
    setSelectedError(null);

    api
      .getSpeechHistoryRecord(route, selectedRecordId)
      .then((record) => {
        if (cancelled) {
          return;
        }
        setSelectedRecord(record);
        mergeHistorySummary(summarizeRecord(route, record));
      })
      .catch((err) => {
        if (cancelled) {
          return;
        }
        setSelectedError(
          err instanceof Error
            ? err.message
            : "Failed to load speech history record details.",
        );
      })
      .finally(() => {
        if (!cancelled) {
          setSelectedLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [mergeHistorySummary, route, selectedRecord, selectedRecordId]);

  useEffect(() => {
    if (!latestRecord) {
      return;
    }
    mergeHistorySummary(summarizeRecord(route, latestRecord));
    setSelectedRecord(latestRecord);
    setSelectedRecordId(latestRecord.id);
    setSelectedError(null);
  }, [latestRecord?.id, latestRecord, mergeHistorySummary, route]);

  const closeHistoryModal = useCallback(() => {
    setIsHistoryModalOpen(false);
  }, []);

  const openHistoryRecord = useCallback((recordId: string) => {
    setSelectedRecordId(recordId);
    setSelectedError(null);
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
      await api.deleteSpeechHistoryRecord(route, deleteTargetRecordId);

      const previous = records;
      const deletedIndex = previous.findIndex(
        (record) => record.id === deleteTargetRecordId,
      );
      const remaining = previous.filter(
        (record) => record.id !== deleteTargetRecordId,
      );

      setRecords(remaining);

      if (selectedRecordId === deleteTargetRecordId) {
        const fallbackIndex =
          deletedIndex >= 0 ? Math.min(deletedIndex, remaining.length - 1) : 0;
        const fallbackId = remaining[fallbackIndex]?.id ?? null;
        setSelectedRecordId(fallbackId);
        if (!fallbackId) {
          setSelectedRecord(null);
          setIsHistoryModalOpen(false);
        }
      }

      if (selectedRecord?.id === deleteTargetRecordId) {
        setSelectedRecord(null);
      }

      setDeleteTargetRecordId(null);
      setDeleteRecordError(null);
    } catch (err) {
      setDeleteRecordError(
        err instanceof Error ? err.message : "Failed to delete speech record.",
      );
    } finally {
      setDeleteRecordPending(false);
    }
  }, [
    deleteRecordPending,
    deleteTargetRecordId,
    records,
    route,
    selectedRecord,
    selectedRecordId,
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
    const audio = document.getElementById(
      "speech-history-audio",
    ) as HTMLAudioElement | null;
    if (audio) {
      audio.pause();
    }
    setHistoryIsPlaying(false);
  }, [isHistoryModalOpen]);

  const selectedSummary = useMemo(
    () =>
      selectedRecordId
        ? (records.find((record) => record.id === selectedRecordId) ?? null)
        : null,
    [records, selectedRecordId],
  );

  const activeRecord =
    selectedRecord && selectedRecord.id === selectedRecordId
      ? selectedRecord
      : null;

  useEffect(() => {
    if (route !== "voice-cloning" || !activeRecord) {
      setHistoryVoiceName("");
      setHistoryVoiceReferenceText("");
      setHistoryVoiceStatus(null);
      return;
    }

    setHistoryVoiceName("");
    setHistoryVoiceReferenceText(
      activeRecord.input_text?.trim() ||
        activeRecord.reference_text?.trim() ||
        "",
    );
    setHistoryVoiceStatus(null);
  }, [activeRecord?.id, route]);

  const deleteTargetRecord = useMemo(() => {
    if (!deleteTargetRecordId) {
      return null;
    }
    const fromSummary = records.find(
      (record) => record.id === deleteTargetRecordId,
    );
    if (fromSummary) {
      return fromSummary;
    }
    if (activeRecord && activeRecord.id === deleteTargetRecordId) {
      return summarizeRecord(route, activeRecord);
    }
    return null;
  }, [activeRecord, deleteTargetRecordId, records, route]);

  const selectedAudioUrl = useMemo(
    () =>
      selectedRecordId
        ? api.speechHistoryRecordAudioUrl(route, selectedRecordId)
        : null,
    [route, selectedRecordId],
  );

  const selectedRecordIndex = useMemo(
    () =>
      selectedRecordId
        ? records.findIndex((record) => record.id === selectedRecordId)
        : -1,
    [records, selectedRecordId],
  );

  const historyViewerDuration =
    historyDuration > 0
      ? historyDuration
      : activeRecord?.audio_duration_secs &&
          activeRecord.audio_duration_secs > 0
        ? activeRecord.audio_duration_secs
        : 0;

  const canOpenNewerRecord = selectedRecordIndex > 0;
  const canOpenOlderRecord =
    selectedRecordIndex >= 0 && selectedRecordIndex < records.length - 1;

  const openAdjacentRecord = useCallback(
    (direction: "newer" | "older") => {
      if (selectedRecordIndex < 0) {
        return;
      }
      const targetIndex =
        direction === "newer"
          ? selectedRecordIndex - 1
          : selectedRecordIndex + 1;
      if (targetIndex < 0 || targetIndex >= records.length) {
        return;
      }
      const target = records[targetIndex];
      if (!target) {
        return;
      }
      setSelectedRecordId(target.id);
      setSelectedError(null);
      setIsHistoryModalOpen(true);
    },
    [records, selectedRecordIndex],
  );

  const toggleHistoryPlayback = useCallback(async () => {
    const audio = document.getElementById(
      "speech-history-audio",
    ) as HTMLAudioElement | null;
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
      const audio = document.getElementById(
        "speech-history-audio",
      ) as HTMLAudioElement | null;
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
      const audio = document.getElementById(
        "speech-history-audio",
      ) as HTMLAudioElement | null;
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
    const audio = document.getElementById(
      "speech-history-audio",
    ) as HTMLAudioElement | null;
    if (!audio) {
      return;
    }
    audio.playbackRate = rate;
    setHistoryPlaybackRate(rate);
  }, []);

  const handleCopyInput = useCallback(async () => {
    if (!activeRecord?.input_text) {
      return;
    }
    await navigator.clipboard.writeText(activeRecord.input_text);
    setCopiedInput(true);
    window.setTimeout(() => setCopiedInput(false), 1800);
  }, [activeRecord]);

  const handleDownloadInput = useCallback(() => {
    if (!activeRecord?.input_text) {
      return;
    }
    const blob = new Blob([activeRecord.input_text], {
      type: "text/plain",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `speech-input-${activeRecord.id}.txt`;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [activeRecord]);

  const handleDownloadAudio = useCallback(async () => {
    if (!activeRecord || isDownloading) {
      return;
    }

    beginDownload();
    try {
      const downloadUrl = api.speechHistoryRecordAudioUrl(
        route,
        activeRecord.id,
        {
          download: true,
        },
      );
      const filename = activeRecord.audio_filename || `${activeRecord.id}.wav`;
      await api.downloadAudioFile(downloadUrl, filename);
      completeDownload();
    } catch (error) {
      failDownload(error);
    }
  }, [
    activeRecord,
    beginDownload,
    completeDownload,
    failDownload,
    isDownloading,
    route,
  ]);

  const handleSaveHistoryVoice = useCallback(async () => {
    if (route !== "voice-cloning" || !activeRecord || historyVoiceSaving) {
      return;
    }

    const trimmedName = historyVoiceName.trim();
    if (!trimmedName) {
      setHistoryVoiceStatus({
        tone: "error",
        message: "Enter a voice name before saving.",
      });
      return;
    }

    const trimmedReferenceText = historyVoiceReferenceText.trim();
    if (!trimmedReferenceText) {
      setHistoryVoiceStatus({
        tone: "error",
        message: "Reference text is required to save a voice profile.",
      });
      return;
    }

    setHistoryVoiceSaving(true);
    setHistoryVoiceStatus(null);

    try {
      const audioResponse = await fetch(
        api.speechHistoryRecordAudioUrl(route, activeRecord.id),
      );
      if (!audioResponse.ok) {
        throw new Error(
          `Failed to load history audio (${audioResponse.status}).`,
        );
      }

      const audioBlob = await audioResponse.blob();
      const audioBase64 = await blobToBase64Payload(audioBlob);

      await api.createSavedVoice({
        name: trimmedName,
        reference_text: trimmedReferenceText,
        audio_base64: audioBase64,
        audio_mime_type:
          activeRecord.audio_mime_type || audioBlob.type || "audio/wav",
        audio_filename:
          activeRecord.audio_filename ||
          `voice-from-history-${activeRecord.id}.wav`,
        source_route_kind: "voice_cloning",
        source_record_id: activeRecord.id,
      });

      setHistoryVoiceName("");
      setHistoryVoiceStatus({
        tone: "success",
        message: `Saved voice profile "${trimmedName}".`,
      });
    } catch (err) {
      setHistoryVoiceStatus({
        tone: "error",
        message: err instanceof Error ? err.message : "Failed to save voice.",
      });
    } finally {
      setHistoryVoiceSaving(false);
    }
  }, [
    activeRecord,
    historyVoiceName,
    historyVoiceReferenceText,
    historyVoiceSaving,
    route,
  ]);

  useEffect(() => {
    const audio = document.getElementById(
      "speech-history-audio",
    ) as HTMLAudioElement | null;
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
    setCopiedInput(false);
  }, [selectedRecordId]);

  return (
    <>
      <RouteHistoryDrawer
        title={title}
        countLabel={`${records.length} ${records.length === 1 ? "record" : "records"}`}
        triggerCount={records.length}
        headerActions={
          <button
            onClick={() => void loadHistory()}
            className="btn btn-ghost app-sidebar-refresh-btn"
            disabled={historyLoading}
            title="Refresh history"
          >
            <RefreshCw
              className={cn("w-3.5 h-3.5", historyLoading && "animate-spin")}
            />
            Refresh
          </button>
        }
      >
        {({ close }) => (
          <>
            <div className="app-sidebar-list">
              {historyLoading ? (
                <div className="app-sidebar-loading">
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  Loading history...
                </div>
              ) : records.length === 0 ? (
                <div className="app-sidebar-empty">{emptyMessage}</div>
              ) : (
                <div className="flex flex-col gap-2.5">
                  {records.map((record) => {
                    const isActive = record.id === selectedRecordId;
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
                          "group app-sidebar-row relative focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                          isActive
                            ? "app-sidebar-row-active"
                            : "app-sidebar-row-idle",
                        )}
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="app-sidebar-row-label truncate">
                            {record.audio_filename ||
                              record.model_id ||
                              record.speaker ||
                              "Speech generation"}
                          </span>
                          <div className="inline-flex items-center gap-1.5 shrink-0">
                            <span className="app-sidebar-row-meta">
                              {formatCreatedAt(record.created_at)}
                            </span>
                            <button
                              onClick={(event) => {
                                event.preventDefault();
                                event.stopPropagation();
                                openDeleteRecordConfirm(record.id);
                              }}
                              className="app-sidebar-delete-btn"
                              title="Delete record"
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          </div>
                        </div>
                        <p
                          className="app-sidebar-row-preview"
                          style={{
                            display: "-webkit-box",
                            WebkitLineClamp: 3,
                            WebkitBoxOrient: "vertical",
                            overflow: "hidden",
                          }}
                        >
                          {record.input_preview}
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
                  className="rounded border bg-[var(--danger-bg)] p-2 text-xs text-[var(--danger-text)] border-[var(--danger-border)]"
                >
                  {historyError}
                </motion.div>
              )}
            </AnimatePresence>
          </>
        )}
      </RouteHistoryDrawer>

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
                    Saved Audio Record
                  </p>
                  <h3 className="truncate text-sm font-semibold mt-1">
                    {selectedSummary?.audio_filename ||
                      selectedSummary?.model_id ||
                      "Speech generation"}
                  </h3>
                  <p className="text-xs text-muted-foreground mt-1">
                    {selectedSummary
                      ? formatCreatedAt(selectedSummary.created_at)
                      : "No record selected"}
                  </p>
                </div>
                <div className="flex items-center gap-1.5">
                  {activeRecord && (
                    <Button
                      onClick={() => openDeleteRecordConfirm(activeRecord.id)}
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
                    onClick={() => openAdjacentRecord("newer")}
                    variant="outline"
                    size="icon"
                    className="h-8 w-8 shadow-sm"
                    disabled={!canOpenNewerRecord}
                    title="Open newer record"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </Button>
                  <Button
                    onClick={() => openAdjacentRecord("older")}
                    variant="outline"
                    size="icon"
                    className="h-8 w-8 shadow-sm"
                    disabled={!canOpenOlderRecord}
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

              <div className="grid flex-1 overflow-hidden lg:grid-cols-[350px,minmax(0,1fr)]">
                <div className="border-b border-[var(--border-muted)] p-4 sm:p-5 lg:border-b-0 lg:border-r">
                  {selectedLoading ? (
                    <div className="h-full min-h-[220px] flex items-center justify-center gap-2 text-sm text-[var(--text-muted)]">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Loading record...
                    </div>
                  ) : selectedError ? (
                    <div className="rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]">
                      {selectedError}
                    </div>
                  ) : activeRecord ? (
                    <>
                      <div className="flex flex-wrap gap-1.5 mb-4">
                        <span className="rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-2 py-1 text-[11px] text-[var(--text-secondary)]">
                          {activeRecord.model_id || "Unknown model"}
                        </span>
                        {activeRecord.speaker && (
                          <span className="rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-2 py-1 text-[11px] text-[var(--text-secondary)]">
                            {activeRecord.speaker}
                          </span>
                        )}
                        {activeRecord.language && (
                          <span className="rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-2 py-1 text-[11px] text-[var(--text-secondary)]">
                            {activeRecord.language}
                          </span>
                        )}
                        <span className="rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-2 py-1 text-[11px] text-[var(--text-secondary)]">
                          {formatAudioDuration(
                            activeRecord.audio_duration_secs,
                          )}
                        </span>
                      </div>

                      <audio
                        id="speech-history-audio"
                        src={selectedAudioUrl ?? undefined}
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
                            "Unable to load audio for this speech generation.",
                          )
                        }
                        className="hidden"
                      />

                      <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3">
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => void toggleHistoryPlayback()}
                            className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] disabled:opacity-45"
                            disabled={!selectedAudioUrl}
                            title={historyIsPlaying ? "Pause" : "Play"}
                          >
                            {historyIsPlaying ? (
                              <Pause className="w-4 h-4" />
                            ) : (
                              <Play className="w-4 h-4" />
                            )}
                          </button>
                          <button
                            onClick={() => skipHistoryAudio(-10)}
                            className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-muted)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] disabled:opacity-45"
                            disabled={!selectedAudioUrl}
                            title="Back 10 seconds"
                          >
                            <SkipBack className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => skipHistoryAudio(10)}
                            className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-muted)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] disabled:opacity-45"
                            disabled={!selectedAudioUrl}
                            title="Forward 10 seconds"
                          >
                            <SkipForward className="w-4 h-4" />
                          </button>
                          <div className="ml-auto text-[11px] text-[var(--text-muted)]">
                            {formatClockTime(historyCurrentTime)} /{" "}
                            {formatClockTime(historyViewerDuration)}
                          </div>
                        </div>

                        <div className="mt-3">
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
                            className="w-full accent-[var(--accent-solid)]"
                            disabled={
                              !selectedAudioUrl || historyViewerDuration <= 0
                            }
                          />
                        </div>

                        <div className="mt-3 flex items-center gap-2">
                          <label className="text-[11px] text-[var(--text-subtle)]">
                            Speed
                          </label>
                          <select
                            value={historyPlaybackRate}
                            onChange={(event) =>
                              handleHistoryRateChange(
                                Number(event.target.value),
                              )
                            }
                            className="h-8 rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-2 text-xs text-[var(--text-secondary)] focus:outline-none focus:ring-1 focus:ring-[var(--border-strong)]"
                          >
                            <option value={0.75}>0.75x</option>
                            <option value={1}>1.0x</option>
                            <option value={1.25}>1.25x</option>
                            <option value={1.5}>1.5x</option>
                            <option value={2}>2.0x</option>
                          </select>
                          {selectedAudioUrl && (
                            <button
                              onClick={handleDownloadAudio}
                              className="ml-auto inline-flex items-center gap-1 rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-2.5 py-1.5 text-xs text-[var(--text-secondary)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] disabled:opacity-45"
                              disabled={isDownloading}
                            >
                              {isDownloading ? (
                                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                              ) : (
                                <Download className="w-3.5 h-3.5" />
                              )}
                              {isDownloading ? "Downloading..." : "Audio"}
                            </button>
                          )}
                        </div>

                        <AnimatePresence>
                          {downloadState !== "idle" && downloadMessage && (
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: "auto" }}
                              exit={{ opacity: 0, height: 0 }}
                              className={cn(
                                "mt-3 rounded-lg border px-3 py-2 text-xs flex items-center gap-2",
                                downloadState === "downloading" &&
                                  "border-amber-500/20 bg-amber-500/10 text-[var(--text-muted)]",
                                downloadState === "success" &&
                                  "border-green-500/20 bg-green-500/10 text-green-500",
                                downloadState === "error" &&
                                  "border-destructive/20 bg-destructive/10 text-destructive",
                              )}
                            >
                              {downloadState === "downloading" ? (
                                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                              ) : downloadState === "success" ? (
                                <CheckCircle2 className="h-3.5 w-3.5" />
                              ) : (
                                <AlertCircle className="h-3.5 w-3.5" />
                              )}
                              {downloadMessage}
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>

                      <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                        <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-2">
                          <div className="text-[10px] text-[var(--text-subtle)] uppercase">
                            Tokens
                          </div>
                          <div className="mt-1 text-[var(--text-secondary)]">
                            {activeRecord.tokens_generated ?? "Unknown"}
                          </div>
                        </div>
                        <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-2">
                          <div className="text-[10px] text-[var(--text-subtle)] uppercase">
                            Runtime
                          </div>
                          <div className="mt-1 text-[var(--text-secondary)]">
                            {Math.max(
                              0,
                              Math.round(activeRecord.generation_time_ms),
                            )}{" "}
                            ms
                          </div>
                        </div>
                      </div>

                      <AnimatePresence>
                        {historyAudioError && (
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: "auto" }}
                            exit={{ opacity: 0, height: 0 }}
                            className="mt-3 rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]"
                          >
                            {historyAudioError}
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </>
                  ) : (
                    <div className="h-full min-h-[220px] flex items-center justify-center text-sm text-[var(--text-subtle)] text-center">
                      Select a history record to inspect playback and metadata.
                    </div>
                  )}
                </div>

                <div className="p-4 sm:p-5 overflow-y-auto">
                  <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                    <h4 className="text-sm font-medium text-[var(--text-primary)]">
                      Input Text
                    </h4>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => void handleCopyInput()}
                        className="inline-flex items-center gap-1 rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-2.5 py-1.5 text-xs text-[var(--text-secondary)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] disabled:opacity-45"
                        disabled={!activeRecord?.input_text}
                      >
                        {copiedInput ? (
                          <>
                            <Copy className="w-3.5 h-3.5" />
                            Copied
                          </>
                        ) : (
                          <>
                            <Copy className="w-3.5 h-3.5" />
                            Copy
                          </>
                        )}
                      </button>
                      <button
                        onClick={handleDownloadInput}
                        className="inline-flex items-center gap-1 rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-2.5 py-1.5 text-xs text-[var(--text-secondary)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] disabled:opacity-45"
                        disabled={!activeRecord?.input_text}
                      >
                        <Download className="w-3.5 h-3.5" />
                        TXT
                      </button>
                    </div>
                  </div>

                  <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-3 min-h-[200px]">
                    {selectedLoading ? (
                      <div className="h-[200px] flex items-center justify-center gap-2 text-sm text-[var(--text-muted)]">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Loading input text...
                      </div>
                    ) : (
                      <p className="text-sm text-[var(--text-secondary)] whitespace-pre-wrap leading-relaxed">
                        {activeRecord?.input_text ||
                          "No input text available for this record."}
                      </p>
                    )}
                  </div>

                  {activeRecord?.voice_description && (
                    <div className="mt-4">
                      <h5 className="text-xs font-medium text-[var(--text-muted)] mb-2 uppercase tracking-wide">
                        Voice Description
                      </h5>
                      <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-3">
                        <p className="text-sm text-[var(--text-secondary)] whitespace-pre-wrap leading-relaxed">
                          {activeRecord.voice_description}
                        </p>
                      </div>
                    </div>
                  )}

                  {activeRecord?.reference_text && (
                    <div className="mt-4">
                      <h5 className="text-xs font-medium text-[var(--text-muted)] mb-2 uppercase tracking-wide">
                        Voice Clone Reference
                      </h5>
                      <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-3">
                        <p className="text-sm text-[var(--text-secondary)] whitespace-pre-wrap leading-relaxed">
                          {activeRecord.reference_text}
                        </p>
                      </div>
                    </div>
                  )}

                  {route === "voice-cloning" && activeRecord && (
                    <div className="mt-4">
                      <h5 className="text-xs font-medium text-[var(--text-muted)] mb-2 uppercase tracking-wide">
                        Save as Voice
                      </h5>
                      <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-3 space-y-2.5">
                        <p className="text-xs text-[var(--text-subtle)]">
                          Save this cloned output as a reusable voice profile.
                        </p>
                        <input
                          value={historyVoiceName}
                          onChange={(event) =>
                            setHistoryVoiceName(event.target.value)
                          }
                          placeholder="Voice name"
                          className="input text-sm"
                          disabled={historyVoiceSaving}
                        />
                        <textarea
                          value={historyVoiceReferenceText}
                          onChange={(event) =>
                            setHistoryVoiceReferenceText(event.target.value)
                          }
                          rows={3}
                          className="textarea text-sm"
                          disabled={historyVoiceSaving}
                          placeholder="Transcript matching this audio"
                        />
                        <div className="flex justify-end">
                          <button
                            onClick={() => void handleSaveHistoryVoice()}
                            className="btn btn-secondary text-xs min-h-[34px]"
                            disabled={
                              historyVoiceSaving ||
                              !historyVoiceName.trim() ||
                              !historyVoiceReferenceText.trim()
                            }
                          >
                            {historyVoiceSaving ? (
                              <>
                                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                                Saving...
                              </>
                            ) : (
                              <>
                                <BookmarkPlus className="w-3.5 h-3.5" />
                                Save Voice
                              </>
                            )}
                          </button>
                        </div>
                        <AnimatePresence>
                          {historyVoiceStatus && (
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: "auto" }}
                              exit={{ opacity: 0, height: 0 }}
                              className={cn(
                                "rounded-md border px-3 py-2 text-xs",
                                historyVoiceStatus.tone === "success"
                                  ? "border-green-500/20 bg-green-500/10 text-green-500"
                                  : "border-destructive/20 bg-destructive/10 text-destructive",
                              )}
                            >
                              {historyVoiceStatus.message}
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {deleteTargetRecord && (
          <motion.div
            className="fixed inset-0 z-[60] bg-black/75 p-4 backdrop-blur-sm"
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
              className="mx-auto mt-[18vh] max-w-md rounded-xl border border-[var(--border-strong)] bg-[var(--bg-surface-1)] p-5"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-start gap-3">
                <div className="mt-0.5 rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2 text-[var(--danger-text)]">
                  <AlertTriangle className="h-4 w-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                    Delete speech record?
                  </h3>
                  <p className="mt-1 text-sm text-[var(--text-muted)]">
                    This permanently removes the saved audio and metadata from
                    history.
                  </p>
                  <p className="mt-2 truncate text-xs text-[var(--text-subtle)]">
                    {deleteTargetRecord.audio_filename ||
                      deleteTargetRecord.model_id ||
                      deleteTargetRecord.id}
                  </p>
                </div>
              </div>

              <AnimatePresence>
                {deleteRecordError && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-4 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]"
                  >
                    {deleteRecordError}
                  </motion.div>
                )}
              </AnimatePresence>

              <div className="mt-5 flex items-center justify-end gap-2">
                <button
                  onClick={closeDeleteRecordConfirm}
                  className="rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-surface-3)] disabled:opacity-50"
                  disabled={deleteRecordPending}
                >
                  Cancel
                </button>
                <button
                  onClick={() => void confirmDeleteRecord()}
                  className="flex items-center gap-1.5 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-1.5 text-xs font-medium text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)] disabled:opacity-50"
                  disabled={deleteRecordPending}
                >
                  {deleteRecordPending ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Trash2 className="h-3.5 w-3.5" />
                  )}
                  Delete record
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
