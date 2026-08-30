import { useMemo, useState } from "react";
import {
  AlertTriangle,
  Copy,
  Download,
  ExternalLink,
  Loader2,
  MoreVertical,
  OctagonX,
  Trash2,
} from "lucide-react";

import { api, type SpeechHistoryRecordSummary } from "@/api";
import { useNotifications } from "@/app/providers/NotificationProvider";
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
import {
  formatSpeechCreatedAt,
  formatSpeechDuration,
  normalizeSpeechProcessingStatus,
  resolveSpeechVoiceLabel,
} from "@/features/text-to-speech/support";

interface TextToSpeechHistoryTableProps {
  records: SpeechHistoryRecordSummary[];
  savedVoiceNameById?: Record<string, string>;
  loading?: boolean;
  error?: string | null;
  loadMore?: {
    canLoadMore: boolean;
    loading: boolean;
    onLoadMore: () => void;
  };
  onOpenRecord: (recordId: string) => void;
  onCancelRecord?: (recordId: string) => Promise<void>;
  onDeleteRecord?: (recordId: string) => Promise<void>;
  onRefresh?: () => void;
}

function rowLabel(
  record: SpeechHistoryRecordSummary,
  voiceLabel: string,
): string {
  return record.audio_filename || voiceLabel || record.model_id || record.id;
}

export function TextToSpeechHistoryTable({
  records,
  savedVoiceNameById = {},
  loading = false,
  error = null,
  loadMore,
  onOpenRecord,
  onCancelRecord,
  onDeleteRecord,
  onRefresh,
}: TextToSpeechHistoryTableProps) {
  const { notify } = useNotifications();
  const [busyRecordId, setBusyRecordId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<SpeechHistoryRecordSummary | null>(
    null,
  );
  const [deletePending, setDeletePending] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const deleteTargetLabel = useMemo(() => {
    if (!deleteTarget) {
      return "Speech generation";
    }

    const voiceLabel = resolveSpeechVoiceLabel({
      savedVoiceId: deleteTarget.saved_voice_id,
      speaker: deleteTarget.speaker,
      modelId: deleteTarget.model_id,
      savedVoiceNameById,
    });

    return rowLabel(deleteTarget, voiceLabel);
  }, [deleteTarget, savedVoiceNameById]);

  async function handleCopy(record: SpeechHistoryRecordSummary): Promise<void> {
    if (busyRecordId || deletePending) {
      return;
    }

    setBusyRecordId(record.id);
    try {
      const fullRecord = await api.getTextToSpeechRecord(record.id);
      if (!fullRecord.input_text.trim()) {
        notify({
          title: "Nothing to copy",
          description: "This generation does not have input text yet.",
          tone: "warning",
        });
        return;
      }

      await navigator.clipboard.writeText(fullRecord.input_text);
      notify({
        title: "Input text copied",
        description: rowLabel(
          record,
          resolveSpeechVoiceLabel({
            savedVoiceId: record.saved_voice_id,
            speaker: record.speaker,
            modelId: record.model_id,
            savedVoiceNameById,
          }),
        ),
        tone: "success",
      });
    } catch (err) {
      notify({
        title: "Could not copy text",
        description:
          err instanceof Error ? err.message : "Failed to load generation details.",
        tone: "warning",
      });
    } finally {
      setBusyRecordId(null);
    }
  }

  async function handleDownload(
    record: SpeechHistoryRecordSummary,
    voiceLabel: string,
  ): Promise<void> {
    if (busyRecordId || deletePending) {
      return;
    }

    setBusyRecordId(record.id);
    try {
      const audioUrl = api.textToSpeechRecordAudioUrl(record.id);
      const filename =
        record.audio_filename ||
        `tts-${record.id}.${record.audio_mime_type.includes("wav") ? "wav" : "audio"}`;
      await api.downloadAudioFile(audioUrl, filename);
      notify({
        title: "Download started",
        description: rowLabel(record, voiceLabel),
        tone: "success",
      });
    } catch (err) {
      notify({
        title: "Could not download audio",
        description:
          err instanceof Error ? err.message : "Failed to download audio.",
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
      await onDeleteRecord(deleteTarget.id);
      notify({
        title: "Generation deleted",
        description: deleteTargetLabel,
        tone: "success",
      });
      setDeleteTarget(null);
    } catch (err) {
      setDeleteError(
        err instanceof Error ? err.message : "Failed to delete generation.",
      );
    } finally {
      setDeletePending(false);
    }
  }

  async function handleCancel(record: SpeechHistoryRecordSummary): Promise<void> {
    if (!onCancelRecord || busyRecordId || deletePending) {
      return;
    }

    setBusyRecordId(record.id);
    try {
      await onCancelRecord(record.id);
      notify({
        title: "Generation cancelled",
        description: "The speech job was stopped and its model can now be unloaded.",
        tone: "success",
      });
    } catch (err) {
      notify({
        title: "Could not cancel generation",
        description: err instanceof Error ? err.message : "Failed to cancel generation.",
        tone: "warning",
      });
    } finally {
      setBusyRecordId(null);
    }
  }

  const activeBusyRecordId = busyRecordId;

  if (loading) {
    return (
      <div className="mb-6 flex min-h-[20rem] items-center justify-center rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] text-sm text-[var(--text-muted)]">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        Loading text-to-speech history...
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
          No text-to-speech jobs yet
        </h3>
        <p className="mt-2 text-sm text-[var(--text-muted)]">
          Queued, processing, and completed generations will appear here.
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
                <th className="px-4 py-3 font-semibold">Voice</th>
                <th className="px-4 py-3 font-semibold">Duration</th>
                <th className="px-4 py-3 font-semibold">Preview</th>
                <th className="w-[56px] px-3 py-3 text-right font-semibold sm:px-4">
                  <span className="sr-only">Actions</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {records.map((record) => {
                const voiceLabel = resolveSpeechVoiceLabel({
                  savedVoiceId: record.saved_voice_id,
                  speaker: record.speaker,
                  modelId: record.model_id,
                  savedVoiceNameById,
                });
                const isBusy = activeBusyRecordId === record.id;
                const processingStatus = normalizeSpeechProcessingStatus(
                  record.processing_status,
                  record.processing_error,
                );
                const canDownload = processingStatus === "ready";
                const canCancel =
                  processingStatus === "pending" || processingStatus === "processing";

                return (
                  <tr
                    key={record.id}
                    aria-label={`Open text-to-speech ${rowLabel(record, voiceLabel)}`}
                    className="cursor-pointer border-t border-[var(--border-muted)] transition-colors hover:bg-[var(--bg-surface-1)]"
                    onClick={(event) => {
                      if ((event.target as HTMLElement).closest("[data-row-action]")) {
                        return;
                      }
                      onOpenRecord(record.id);
                    }}
                    onKeyDown={(event) => {
                      if ((event.target as HTMLElement).closest("[data-row-action]")) {
                        return;
                      }
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        onOpenRecord(record.id);
                      }
                    }}
                    tabIndex={0}
                  >
                    <td className="px-4 py-3 align-top text-[var(--text-secondary)] sm:px-5">
                      {formatSpeechCreatedAt(record.created_at)}
                    </td>
                    <td className="px-4 py-3 align-top">
                      <div className="font-medium text-[var(--text-primary)]">
                        {voiceLabel}
                      </div>
                    </td>
                    <td className="px-4 py-3 align-top text-[var(--text-secondary)]">
                      {formatSpeechDuration(record.audio_duration_secs)}
                    </td>
                    <td className="px-4 py-3 align-top text-[var(--text-secondary)]">
                      <div className="max-w-[34rem] line-clamp-2 text-[var(--text-primary)]">
                        {record.input_preview}
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
                            aria-label={`More actions for ${rowLabel(record, voiceLabel)}`}
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
                          <DropdownMenuItem onSelect={() => onOpenRecord(record.id)}>
                            <ExternalLink className="mr-2 h-4 w-4" />
                            Open record
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            disabled={isBusy || deletePending}
                            onSelect={() => void handleCopy(record)}
                          >
                            <Copy className="mr-2 h-4 w-4" />
                            Copy text
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            disabled={isBusy || deletePending || !canDownload}
                            onSelect={() => void handleDownload(record, voiceLabel)}
                          >
                            <Download className="mr-2 h-4 w-4" />
                            Download
                          </DropdownMenuItem>
                          {canCancel ? (
                            <DropdownMenuItem
                              disabled={isBusy || deletePending || !onCancelRecord}
                              onSelect={() => void handleCancel(record)}
                            >
                              <OctagonX className="mr-2 h-4 w-4" />
                              Cancel generation
                            </DropdownMenuItem>
                          ) : null}
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
          <DialogTitle className="sr-only">Delete generation?</DialogTitle>
          {deleteTarget ? (
            <>
              <div className="flex items-start gap-3">
                <div className="mt-0.5 rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2 text-[var(--danger-text)]">
                  <AlertTriangle className="h-4 w-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                    Delete generation?
                  </h3>
                  <DialogDescription className="mt-1 text-sm text-[var(--text-muted)]">
                    This permanently removes the saved audio and prompt text from
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
                    "Delete generation"
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
