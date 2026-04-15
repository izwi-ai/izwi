import { useMemo, useState } from "react";
import {
  AlertTriangle,
  Copy,
  Download,
  ExternalLink,
  Loader2,
  MoreVertical,
  Trash2,
} from "lucide-react";

import {
  api,
  type DiarizationRecord,
  type SpeechTextDiarizationSummary,
  type SpeechTextJobSummary,
} from "@/api";
import { useNotifications } from "@/app/providers/NotificationProvider";
import { DiarizationExportDialog } from "@/components/DiarizationExportDialog";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { TranscriptionExportDialog } from "@/features/transcription/components/TranscriptionExportDialog";
import {
  formatAudioDuration,
  formatCreatedAt,
} from "@/features/transcription/playground/support";
import { formatTranscriptionText } from "@/features/transcription/transcript";
import { formattedTranscriptFromRecord } from "@/utils/diarizationTranscript";
import type { ExportableTranscriptionRecord } from "@/utils/transcriptionExport";

interface TranscriptionHistoryTableProps {
  records: SpeechTextJobSummary[];
  loading?: boolean;
  error?: string | null;
  loadMore?: {
    canLoadMore: boolean;
    loading: boolean;
    onLoadMore: () => void;
  };
  onOpenRecord: (record: SpeechTextJobSummary) => void;
  onDeleteRecord?: (record: SpeechTextJobSummary) => Promise<void>;
  onRefresh?: () => void;
}

type ExportableDiarizationRecord = Pick<
  DiarizationRecord,
  | "id"
  | "created_at"
  | "model_id"
  | "speaker_count"
  | "corrected_speaker_count"
  | "duration_secs"
  | "audio_filename"
  | "speaker_name_overrides"
  | "utterances"
  | "transcript"
  | "raw_transcript"
>;

function rowKind(record: SpeechTextJobSummary): "transcription" | "diarization" {
  return record.kind === "diarization" ? "diarization" : "transcription";
}

function isDiarizationSummary(
  record: SpeechTextJobSummary,
): record is SpeechTextDiarizationSummary {
  return record.kind === "diarization";
}

function rowKindLabel(record: SpeechTextJobSummary): string {
  return rowKind(record) === "diarization" ? "Diarization" : "Transcription";
}

function rowPreview(record: SpeechTextJobSummary): string {
  if (isDiarizationSummary(record)) {
    return record.transcript_preview;
  }
  return record.transcription_preview;
}

function rowLabel(record: SpeechTextJobSummary): string {
  return record.audio_filename || record.model_id || record.id;
}

function rowSummaryLabel(record: SpeechTextJobSummary): string {
  if (record.summary_preview) {
    return `Summary: ${record.summary_preview}`;
  }
  if (record.summary_status === "pending") {
    return "Summary: Generating";
  }
  if (record.summary_status === "failed") {
    return "Summary: Failed";
  }
  return "Summary: Not requested";
}

export function TranscriptionHistoryTable({
  records,
  loading = false,
  error = null,
  loadMore,
  onOpenRecord,
  onDeleteRecord,
  onRefresh,
}: TranscriptionHistoryTableProps) {
  const { notify } = useNotifications();
  const [busyRecordId, setBusyRecordId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<SpeechTextJobSummary | null>(
    null,
  );
  const [deletePending, setDeletePending] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [exportTranscriptionRecord, setExportTranscriptionRecord] =
    useState<ExportableTranscriptionRecord | null>(null);
  const [exportDiarizationRecord, setExportDiarizationRecord] =
    useState<ExportableDiarizationRecord | null>(null);

  const deleteTargetLabel = useMemo(
    () => (deleteTarget ? rowLabel(deleteTarget) : "Speech-text job"),
    [deleteTarget],
  );

  const deleteTargetKindLabel = useMemo(
    () => (deleteTarget ? rowKindLabel(deleteTarget).toLowerCase() : "record"),
    [deleteTarget],
  );

  async function handleCopy(record: SpeechTextJobSummary): Promise<void> {
    if (busyRecordId || deletePending) {
      return;
    }

    setBusyRecordId(record.id);
    const kind = rowKind(record);
    try {
      const transcript =
        kind === "diarization"
          ? formattedTranscriptFromRecord(
              await api.getDiarizationRecord(record.id),
            )
          : formatTranscriptionText(await api.getTranscriptionRecord(record.id));

      if (!transcript.trim()) {
        notify({
          title: "Nothing to copy",
          description: `This ${kind} does not have transcript text yet.`,
          tone: "warning",
        });
        return;
      }

      await navigator.clipboard.writeText(transcript);
      notify({
        title: "Transcript copied",
        description: rowLabel(record),
        tone: "success",
      });
    } catch (err) {
      notify({
        title: "Could not copy transcript",
        description:
          err instanceof Error
            ? err.message
            : `Failed to load ${kind} record.`,
        tone: "warning",
      });
    } finally {
      setBusyRecordId(null);
    }
  }

  async function handleExport(record: SpeechTextJobSummary): Promise<void> {
    if (busyRecordId || deletePending) {
      return;
    }

    setBusyRecordId(record.id);
    const kind = rowKind(record);
    try {
      if (kind === "diarization") {
        setExportTranscriptionRecord(null);
        setExportDiarizationRecord(await api.getDiarizationRecord(record.id));
      } else {
        setExportDiarizationRecord(null);
        setExportTranscriptionRecord(await api.getTranscriptionRecord(record.id));
      }
      setExportDialogOpen(true);
    } catch (err) {
      notify({
        title: "Could not open export",
        description:
          err instanceof Error
            ? err.message
            : `Failed to load ${kind} record.`,
        tone: "warning",
      });
    } finally {
      setBusyRecordId(null);
    }
  }

  async function handleConfirmDelete(): Promise<void> {
    if (!deleteTarget || !onDeleteRecord || deletePending) {
      return;
    }

    setDeletePending(true);
    setDeleteError(null);
    try {
      await onDeleteRecord(deleteTarget);
      notify({
        title: `${rowKindLabel(deleteTarget)} deleted`,
        description: deleteTargetLabel,
        tone: "success",
      });
      setDeleteTarget(null);
    } catch (err) {
      setDeleteError(
        err instanceof Error ? err.message : `Failed to delete ${deleteTargetKindLabel}.`,
      );
    } finally {
      setDeletePending(false);
    }
  }

  const activeBusyRecordId = busyRecordId;

  if (loading) {
    return (
      <div className="mb-6 flex min-h-[20rem] items-center justify-center rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] text-sm text-[var(--text-muted)]">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        Loading speech-text history...
      </div>
    );
  }

  if (error) {
    return (
      <div className="mb-6 rounded-2xl border border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-text)]">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <p>{error}</p>
          </div>
          {onRefresh ? (
            <Button type="button" variant="outline" size="sm" onClick={onRefresh}>
              Retry
            </Button>
          ) : null}
        </div>
      </div>
    );
  }

  if (records.length === 0) {
    return (
      <div className="mb-6 rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-10 text-center">
        <h3 className="text-lg font-semibold text-[var(--text-primary)]">
          No speech-text jobs yet
        </h3>
        <p className="mt-2 text-sm text-[var(--text-muted)]">
          Queued, processing, and completed transcription and diarization jobs will appear here.
        </p>
      </div>
    );
  }

  return (
    <>
      <div className="mb-6 overflow-hidden rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)]">
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse text-sm">
            <thead className="bg-[var(--bg-surface-1)] text-left text-xs uppercase tracking-[0.14em] text-[var(--text-muted)]">
              <tr>
                <th className="px-4 py-3 font-semibold sm:px-5">Created</th>
                <th className="px-4 py-3 font-semibold">Type</th>
                <th className="px-4 py-3 font-semibold">File</th>
                <th className="px-4 py-3 font-semibold">Duration</th>
                <th className="px-4 py-3 font-semibold">Preview</th>
                <th className="w-[56px] px-3 py-3 text-right font-semibold sm:px-4">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {records.map((record) => {
                const kind = rowKind(record);
                const isBusy = activeBusyRecordId === record.id;
                const speakerCount = isDiarizationSummary(record)
                  ? (record.corrected_speaker_count ?? record.speaker_count)
                  : null;

                return (
                  <tr
                    key={record.id}
                    aria-label={`Open ${kind} ${rowLabel(record)}`}
                    className="cursor-pointer border-t border-[var(--border-muted)] transition-colors hover:bg-[var(--bg-surface-1)]"
                    onClick={(event) => {
                      if ((event.target as HTMLElement).closest("[data-row-action]")) {
                        return;
                      }
                      onOpenRecord(record);
                    }}
                    onKeyDown={(event) => {
                      if ((event.target as HTMLElement).closest("[data-row-action]")) {
                        return;
                      }
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        onOpenRecord(record);
                      }
                    }}
                    tabIndex={0}
                  >
                    <td className="px-4 py-3 align-top text-[var(--text-secondary)] sm:px-5">
                      {formatCreatedAt(record.created_at)}
                    </td>
                    <td className="px-4 py-3 align-top text-[var(--text-secondary)]">
                      {rowKindLabel(record)}
                    </td>
                    <td className="px-4 py-3 align-top">
                      <div className="font-medium text-[var(--text-primary)]">
                        {record.audio_filename || "Audio input"}
                      </div>
                    </td>
                    <td className="px-4 py-3 align-top text-[var(--text-secondary)]">
                      {formatAudioDuration(record.duration_secs)}
                    </td>
                    <td className="px-4 py-3 align-top text-[var(--text-secondary)]">
                      <div className="max-w-[34rem]">
                        <div className="line-clamp-2 text-[var(--text-primary)]">
                          {rowPreview(record) || "No transcript preview yet."}
                        </div>
                        {speakerCount != null ? (
                          <div className="mt-1 line-clamp-1 text-xs text-[var(--text-muted)]">
                            Speakers: {speakerCount}
                          </div>
                        ) : null}
                        <div className="mt-1 line-clamp-1 text-xs text-[var(--text-muted)]">
                          {rowSummaryLabel(record)}
                        </div>
                      </div>
                    </td>
                    <td className="px-3 py-2 align-top text-right sm:px-4">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon"
                            data-row-action
                            className="h-8 w-8 rounded-full text-[var(--text-muted)] hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)]"
                            aria-label={`More actions for ${rowLabel(record)}`}
                            onClick={(event) => event.stopPropagation()}
                            onKeyDown={(event) => event.stopPropagation()}
                          >
                            {isBusy ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <MoreVertical className="h-4 w-4" />
                            )}
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent
                          align="end"
                          className="w-48"
                          onClick={(event) => event.stopPropagation()}
                        >
                          <DropdownMenuItem onSelect={() => onOpenRecord(record)}>
                            <ExternalLink className="mr-2 h-4 w-4" />
                            Open record
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            disabled={isBusy || deletePending}
                            onSelect={() => void handleCopy(record)}
                          >
                            <Copy className="mr-2 h-4 w-4" />
                            Copy transcript
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            disabled={isBusy || deletePending}
                            onSelect={() => void handleExport(record)}
                          >
                            <Download className="mr-2 h-4 w-4" />
                            Export
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem
                            disabled={!onDeleteRecord || deletePending}
                            onSelect={() => {
                              setDeleteError(null);
                              setDeleteTarget(record);
                            }}
                            className="text-[var(--danger-text)] focus:text-[var(--danger-text)]"
                          >
                            <Trash2 className="mr-2 h-4 w-4" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {loadMore?.canLoadMore ? (
          <div className="flex justify-center border-t border-[var(--border-muted)] px-4 py-3 sm:px-5">
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 gap-2"
              onClick={loadMore.onLoadMore}
              disabled={loadMore.loading}
            >
              {loadMore.loading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
              Load more
            </Button>
          </div>
        ) : null}
      </div>

      <TranscriptionExportDialog
        record={exportTranscriptionRecord}
        open={exportDialogOpen && Boolean(exportTranscriptionRecord)}
        onOpenChange={(open) => {
          setExportDialogOpen(open);
          if (!open) {
            setExportTranscriptionRecord(null);
          }
        }}
      />

      <DiarizationExportDialog
        record={exportDiarizationRecord}
        open={exportDialogOpen && Boolean(exportDiarizationRecord)}
        onOpenChange={(open) => {
          setExportDialogOpen(open);
          if (!open) {
            setExportDiarizationRecord(null);
          }
        }}
      />

      <Dialog
        open={Boolean(deleteTarget)}
        onOpenChange={(open) => {
          if (!deletePending) {
            setDeleteTarget(open ? deleteTarget : null);
            if (!open) {
              setDeleteError(null);
            }
          }
        }}
      >
        <DialogContent className="max-w-md border-[var(--border-strong)] bg-[var(--bg-surface-1)] p-5">
          <DialogTitle className="sr-only">Delete record?</DialogTitle>
          {deleteTarget ? (
            <>
              <div className="flex items-start gap-3">
                <div className="mt-0.5 rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2 text-[var(--danger-text)]">
                  <AlertTriangle className="h-4 w-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                    Delete {deleteTargetKindLabel}?
                  </h3>
                  <DialogDescription className="mt-1 text-sm text-[var(--text-muted)]">
                    This permanently removes the saved audio and transcript from
                    history.
                  </DialogDescription>
                  <p className="mt-2 truncate text-xs text-[var(--text-subtle)]">
                    {deleteTargetLabel}
                  </p>
                </div>
              </div>

              {deleteError ? (
                <div className="mt-4 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]">
                  {deleteError}
                </div>
              ) : null}

              <div className="mt-5 flex items-center justify-end gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="h-8 border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)] hover:bg-[var(--bg-surface-3)]"
                  onClick={() => {
                    setDeleteTarget(null);
                    setDeleteError(null);
                  }}
                  disabled={deletePending}
                >
                  Cancel
                </Button>
                <Button
                  type="button"
                  variant="destructive"
                  size="sm"
                  className="h-8 gap-1.5"
                  onClick={() => void handleConfirmDelete()}
                  disabled={deletePending}
                >
                  {deletePending ? (
                    <>
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      Deleting
                    </>
                  ) : (
                    `Delete ${deleteTargetKindLabel}`
                  )}
                </Button>
              </div>
            </>
          ) : null}
        </DialogContent>
      </Dialog>
    </>
  );
}
