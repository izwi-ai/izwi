import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle,
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  Download,
  Loader2,
  Pause,
  Play,
  SkipBack,
  SkipForward,
  Trash2,
  X,
} from "lucide-react";
import clsx from "clsx";

import { RouteHistoryDrawer } from "@/components/RouteHistoryDrawer";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  api,
  type DiarizationRecord,
  type DiarizationRecordSummary,
} from "../api";
import {
  formattedTranscriptFromRecord,
  previewTranscript,
  transcriptEntriesFromRecord,
} from "../utils/diarizationTranscript";
import { DiarizationSpeakerManager } from "./DiarizationSpeakerManager";

interface DiarizationHistoryPanelProps {
  latestRecord?: DiarizationRecord | null;
  historyActionContainer?: HTMLElement | null;
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

function summarizeRecord(record: DiarizationRecord): DiarizationRecordSummary {
  const entries = transcriptEntriesFromRecord(record);
  const preview = previewTranscript(
    entries,
    record.transcript ?? "",
    record.raw_transcript ?? "",
  );
  const formatted = formattedTranscriptFromRecord(record);

  return {
    id: record.id,
    created_at: record.created_at,
    model_id: record.model_id,
    speaker_count: record.speaker_count ?? 0,
    corrected_speaker_count:
      record.corrected_speaker_count ?? record.speaker_count ?? 0,
    speaker_name_override_count: Object.keys(
      record.speaker_name_overrides ?? {},
    ).length,
    duration_secs: record.duration_secs,
    processing_time_ms: record.processing_time_ms,
    rtf: record.rtf,
    audio_mime_type: record.audio_mime_type,
    audio_filename: record.audio_filename,
    transcript_preview: preview || "No transcript",
    transcript_chars: Array.from(formatted).length,
  };
}

export function DiarizationHistoryPanel({
  latestRecord = null,
  historyActionContainer,
}: DiarizationHistoryPanelProps) {
  const [historyRecords, setHistoryRecords] = useState<
    DiarizationRecordSummary[]
  >([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [selectedHistoryRecordId, setSelectedHistoryRecordId] = useState<
    string | null
  >(null);
  const [selectedHistoryRecord, setSelectedHistoryRecord] =
    useState<DiarizationRecord | null>(null);
  const [selectedHistoryLoading, setSelectedHistoryLoading] = useState(false);
  const [selectedHistoryError, setSelectedHistoryError] = useState<
    string | null
  >(null);
  const [isHistoryDrawerOpen, setIsHistoryDrawerOpen] = useState(false);
  const [isHistoryModalOpen, setIsHistoryModalOpen] = useState(false);
  const [historyCurrentTime, setHistoryCurrentTime] = useState(0);
  const [historyDuration, setHistoryDuration] = useState(0);
  const [historyIsPlaying, setHistoryIsPlaying] = useState(false);
  const [historyPlaybackRate, setHistoryPlaybackRate] = useState(1);
  const [historyAudioError, setHistoryAudioError] = useState<string | null>(
    null,
  );
  const [historyTranscriptCopied, setHistoryTranscriptCopied] = useState(false);
  const [recordWorkspaceTab, setRecordWorkspaceTab] = useState("transcript");
  const [speakerUpdatePending, setSpeakerUpdatePending] = useState(false);
  const [speakerUpdateError, setSpeakerUpdateError] = useState<string | null>(
    null,
  );
  const [deleteTargetRecordId, setDeleteTargetRecordId] = useState<
    string | null
  >(null);
  const [deleteRecordPending, setDeleteRecordPending] = useState(false);
  const [deleteRecordError, setDeleteRecordError] = useState<string | null>(
    null,
  );

  const historyAudioRef = useRef<HTMLAudioElement | null>(null);

  const mergeHistorySummary = useCallback(
    (summary: DiarizationRecordSummary) => {
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
      const records = await api.listDiarizationRecords();
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
          : "Failed to load diarization history.",
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
      .getDiarizationRecord(selectedHistoryRecordId)
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
            : "Failed to load diarization record details.",
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

  useEffect(() => {
    if (!latestRecord) {
      return;
    }
    setSelectedHistoryRecord(latestRecord);
    setSelectedHistoryRecordId(latestRecord.id);
    setSelectedHistoryError(null);
    mergeHistorySummary(summarizeRecord(latestRecord));
  }, [latestRecord?.id, latestRecord, mergeHistorySummary]);

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
      await api.deleteDiarizationRecord(deleteTargetRecordId);

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
          : "Failed to delete diarization record.",
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
        ? api.diarizationRecordAudioUrl(selectedHistoryRecordId)
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

  const normalizedActiveTranscript = useMemo(
    () => (activeHistoryRecord ? formattedTranscriptFromRecord(activeHistoryRecord) : ""),
    [activeHistoryRecord],
  );

  const activeEntries = useMemo(
    () =>
      activeHistoryRecord ? transcriptEntriesFromRecord(activeHistoryRecord) : [],
    [activeHistoryRecord],
  );

  const handleCopyHistoryTranscript = useCallback(async () => {
    if (!normalizedActiveTranscript) {
      return;
    }
    await navigator.clipboard.writeText(normalizedActiveTranscript);
    setHistoryTranscriptCopied(true);
    window.setTimeout(() => setHistoryTranscriptCopied(false), 1800);
  }, [normalizedActiveTranscript]);

  const handleDownloadHistoryTranscript = useCallback(() => {
    if (!activeHistoryRecord) {
      return;
    }
    const content =
      normalizedActiveTranscript ||
      activeHistoryRecord.transcript ||
      activeHistoryRecord.raw_transcript ||
      "";
    const blob = new Blob([content], {
      type: "text/plain",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `diarization-${activeHistoryRecord.id}.txt`;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [activeHistoryRecord, normalizedActiveTranscript]);

  const handleSaveSpeakerCorrections = useCallback(
    async (speakerNameOverrides: Record<string, string>) => {
      if (!activeHistoryRecord || speakerUpdatePending) {
        return;
      }

      setSpeakerUpdatePending(true);
      setSpeakerUpdateError(null);
      try {
        const updatedRecord = await api.updateDiarizationRecord(activeHistoryRecord.id, {
          speaker_name_overrides: speakerNameOverrides,
        });
        setSelectedHistoryRecord(updatedRecord);
        mergeHistorySummary(summarizeRecord(updatedRecord));
      } catch (err) {
        setSpeakerUpdateError(
          err instanceof Error ? err.message : "Failed to save speaker corrections.",
        );
      } finally {
        setSpeakerUpdatePending(false);
      }
    },
    [activeHistoryRecord, mergeHistorySummary, speakerUpdatePending],
  );

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
    setRecordWorkspaceTab("transcript");
    setSpeakerUpdateError(null);
  }, [selectedHistoryRecordId]);

  const historyDrawer = (
    <RouteHistoryDrawer
      title="Diarization History"
      countLabel={`${historyRecords.length} ${historyRecords.length === 1 ? "record" : "records"}`}
      triggerCount={historyRecords.length}
      open={isHistoryDrawerOpen}
      onOpenChange={handleHistoryDrawerOpenChange}
    >
      {({ close }) => (
        <>
          <div className="app-sidebar-list">
            {historyLoading ? (
              <div className="app-sidebar-loading">
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                Loading history...
              </div>
            ) : historyRecords.length === 0 ? (
              <div className="app-sidebar-empty">
                No saved diarization records yet.
              </div>
            ) : (
              <div className="flex flex-col gap-2.5">
                {historyRecords.map((record) => {
                  const isActive = record.id === selectedHistoryRecordId;
                  return (
                    <div
                      key={record.id}
                      role="button"
                      tabIndex={0}
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
                      className={clsx(
                        "app-sidebar-row",
                        isActive
                          ? "app-sidebar-row-active"
                          : "app-sidebar-row-idle",
                      )}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="app-sidebar-row-label truncate">
                          {record.audio_filename ||
                            record.model_id ||
                            "Diarization run"}
                        </span>
                        <div className="inline-flex items-center gap-1.5 shrink-0">
                          <span className="app-sidebar-row-meta">
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
                            className="app-sidebar-delete-btn"
                            title="Delete record"
                            aria-label={`Delete ${record.audio_filename || record.model_id || "diarization transcript"}`}
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
                        {record.transcript_preview}
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
  );

  return (
    <>
      {historyActionContainer === undefined
        ? historyDrawer
        : historyActionContainer
          ? createPortal(historyDrawer, historyActionContainer)
          : null}

      <AnimatePresence>
        {isHistoryModalOpen && (
          <motion.div
            className="fixed inset-0 z-50 bg-black/70 p-3 backdrop-blur-sm sm:p-6"
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
              className="mx-auto flex max-h-[92vh] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] shadow-2xl"
            >
              <div className="flex items-center justify-between gap-3 border-b border-[var(--border-muted)] px-4 py-3 sm:px-6">
                <div className="min-w-0">
                  <p className="text-[11px] uppercase tracking-wide text-[var(--text-subtle)]">
                    Diarization Record
                  </p>
                  <h3 className="truncate text-sm font-medium text-[var(--text-primary)] mt-1">
                    {selectedHistorySummary?.audio_filename ||
                      selectedHistorySummary?.model_id ||
                      "Diarization transcript"}
                  </h3>
                  <p className="text-xs text-[var(--text-muted)] mt-1">
                    {selectedHistorySummary
                      ? formatCreatedAt(selectedHistorySummary.created_at)
                      : "No record selected"}
                  </p>
                </div>
                <div className="flex items-center gap-1.5">
                  {activeHistoryRecord && (
                    <button
                      onClick={() =>
                        openDeleteRecordConfirm(activeHistoryRecord.id)
                      }
                      className="inline-flex items-center gap-1 rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] px-2.5 py-1.5 text-xs text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)]"
                      title="Delete this record"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                      Delete
                    </button>
                  )}
                  <button
                    onClick={() => openAdjacentHistoryRecord("newer")}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-muted)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] disabled:opacity-40"
                    disabled={!canOpenNewerHistory}
                    title="Open newer record"
                  >
                    <ChevronLeft className="w-4 h-4" />
                  </button>
                  <button
                    onClick={() => openAdjacentHistoryRecord("older")}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-muted)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] disabled:opacity-40"
                    disabled={!canOpenOlderHistory}
                    title="Open older record"
                  >
                    <ChevronRight className="w-4 h-4" />
                  </button>
                  <button
                    onClick={closeHistoryModal}
                    className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-muted)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]"
                    title="Close"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="grid flex-1 overflow-hidden lg:grid-cols-[360px,minmax(0,1fr)]">
                <div className="border-b border-[var(--border-muted)] p-4 sm:p-5 lg:border-b-0 lg:border-r">
                  {selectedHistoryLoading ? (
                    <div className="h-full min-h-[220px] flex items-center justify-center gap-2 text-sm text-[var(--text-muted)]">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Loading record...
                    </div>
                  ) : selectedHistoryError ? (
                    <div className="rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]">
                      {selectedHistoryError}
                    </div>
                  ) : activeHistoryRecord ? (
                    <>
                      <div className="flex flex-wrap gap-1.5 mb-4">
                        <span className="rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-2 py-1 text-[11px] text-[var(--text-secondary)]">
                          {activeHistoryRecord.model_id || "Unknown model"}
                        </span>
                        <span className="rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-2 py-1 text-[11px] text-[var(--text-secondary)]">
                          {activeHistoryRecord.corrected_speaker_count ??
                            activeHistoryRecord.speaker_count}{" "}
                          speakers
                        </span>
                        <span className="rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-2 py-1 text-[11px] text-[var(--text-secondary)]">
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
                            "Unable to load audio for this diarization record.",
                          )
                        }
                        className="hidden"
                      />

                      <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3">
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => void toggleHistoryPlayback()}
                            className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] disabled:opacity-45"
                            disabled={!selectedHistoryAudioUrl}
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
                            disabled={!selectedHistoryAudioUrl}
                            title="Back 10 seconds"
                          >
                            <SkipBack className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => skipHistoryAudio(10)}
                            className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-muted)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] disabled:opacity-45"
                            disabled={!selectedHistoryAudioUrl}
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
                              !selectedHistoryAudioUrl ||
                              historyViewerDuration <= 0
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
                          {selectedHistoryAudioUrl && (
                            <a
                              href={selectedHistoryAudioUrl}
                              download={
                                activeHistoryRecord.audio_filename ||
                                `${activeHistoryRecord.id}.wav`
                              }
                              className="ml-auto inline-flex items-center gap-1 rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-2.5 py-1.5 text-xs text-[var(--text-secondary)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)]"
                            >
                              <Download className="w-3.5 h-3.5" />
                              Audio
                            </a>
                          )}
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
                      Select a history record to inspect playback and
                      transcript.
                    </div>
                  )}
                </div>

                <div className="p-4 sm:p-5 overflow-y-auto">
                  <Tabs
                    value={recordWorkspaceTab}
                    onValueChange={setRecordWorkspaceTab}
                    className="space-y-4"
                  >
                    <TabsList className="w-full justify-start bg-[var(--bg-surface-1)]">
                      <TabsTrigger value="transcript">Transcript</TabsTrigger>
                      <TabsTrigger value="speakers">Speakers</TabsTrigger>
                    </TabsList>

                    <TabsContent value="transcript" className="mt-0 space-y-4">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <h4 className="text-sm font-medium text-[var(--text-primary)]">
                          Transcript
                        </h4>
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => void handleCopyHistoryTranscript()}
                            className="inline-flex items-center gap-1 rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-2.5 py-1.5 text-xs text-[var(--text-secondary)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] disabled:opacity-45"
                            disabled={!normalizedActiveTranscript}
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
                          <button
                            onClick={handleDownloadHistoryTranscript}
                            className="inline-flex items-center gap-1 rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-2.5 py-1.5 text-xs text-[var(--text-secondary)] transition-colors hover:border-[var(--border-strong)] hover:text-[var(--text-primary)] disabled:opacity-45"
                            disabled={!normalizedActiveTranscript}
                          >
                            <Download className="w-3.5 h-3.5" />
                            TXT
                          </button>
                        </div>
                      </div>

                      <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-3 min-h-[320px]">
                        {selectedHistoryLoading ? (
                          <div className="h-[320px] flex items-center justify-center gap-2 text-sm text-[var(--text-muted)]">
                            <Loader2 className="w-4 h-4 animate-spin" />
                            Loading transcript...
                          </div>
                        ) : activeEntries.length > 0 ? (
                          <div className="space-y-2">
                            {activeEntries.map((entry, index) => (
                              <div
                                key={`${entry.speaker}-${entry.start}-${entry.end}-${index}`}
                                className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-2)] p-3"
                              >
                                <div className="flex items-center justify-between gap-2 mb-1">
                                  <span className="text-xs font-medium text-[var(--text-primary)]">
                                    {entry.speaker}
                                  </span>
                                  <span className="text-[11px] text-[var(--text-subtle)]">
                                    {entry.start.toFixed(2)}s -{" "}
                                    {entry.end.toFixed(2)}s
                                  </span>
                                </div>
                                <p className="text-sm text-[var(--text-secondary)] whitespace-pre-wrap break-words">
                                  {entry.text}
                                </p>
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="text-sm text-[var(--text-secondary)] whitespace-pre-wrap leading-relaxed">
                            {normalizedActiveTranscript ||
                              "No transcript text available for this record."}
                          </p>
                        )}
                      </div>
                    </TabsContent>

                    <TabsContent value="speakers" className="mt-0">
                      {activeHistoryRecord ? (
                        <DiarizationSpeakerManager
                          record={activeHistoryRecord}
                          isSaving={speakerUpdatePending}
                          error={speakerUpdateError}
                          onSave={handleSaveSpeakerCorrections}
                        />
                      ) : null}
                    </TabsContent>
                  </Tabs>
                </div>
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
          <DialogContent className="max-w-md border-[var(--border-strong)] bg-[var(--bg-surface-1)] p-5">
            <DialogTitle className="sr-only">
              Delete diarization record?
            </DialogTitle>
            <div className="flex items-start gap-3">
              <div className="mt-0.5 rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2 text-[var(--danger-text)]">
                <AlertTriangle className="h-4 w-4" />
              </div>
              <div className="min-w-0 flex-1">
                <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                  Delete diarization record?
                </h3>
                <DialogDescription className="mt-1 text-sm text-[var(--text-muted)]">
                  This permanently removes the saved audio and transcript from
                  history.
                </DialogDescription>
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
              <Button
                onClick={closeDeleteRecordConfirm}
                variant="outline"
                size="sm"
                className="h-8 border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)] hover:bg-[var(--bg-surface-3)]"
                disabled={deleteRecordPending}
              >
                Cancel
              </Button>
              <Button
                onClick={() => void confirmDeleteRecord()}
                variant="destructive"
                size="sm"
                className="h-8 gap-1.5"
                disabled={deleteRecordPending}
              >
                {deleteRecordPending ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Trash2 className="h-3.5 w-3.5" />
                )}
                Delete record
              </Button>
            </div>
          </DialogContent>
        ) : null}
      </Dialog>
    </>
  );
}
