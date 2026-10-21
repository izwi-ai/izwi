import { useState } from "react";
import {
  Copy,
  Download,
  ExternalLink,
  OctagonX,
  Pause,
  Play,
  Trash2,
} from "lucide-react";

import { api, type SpeechHistoryRecordSummary } from "@/api";
import { useNotifications } from "@/app/providers/NotificationProvider";
import { useAudioPreview } from "@/components/audio/useAudioPreview";
import { ConfirmDeleteDialog } from "@/components/history/ConfirmDeleteDialog";
import {
  formatClockDuration,
  formatHistoryDate,
} from "@/components/history/format";
import { HistoryTable, type HistoryColumn } from "@/components/history/HistoryTable";
import { RowBadge } from "@/components/history/RowBadge";
import { RowMenu } from "@/components/history/RowMenu";
import { Button } from "@/components/ui/button";
import { getModelDisplayName } from "@/features/models/catalog/modelMetadata";
import {
  normalizeSpeechProcessingStatus,
  resolveSpeechVoiceLabel,
  speechDownloadFilename,
} from "@/features/text-to-speech/support";
import { cn } from "@/lib/utils";

interface TextToSpeechHistoryTableProps {
  records: SpeechHistoryRecordSummary[];
  /** Records matching the filters, if the server reports it. */
  total?: number | null;
  /** Search or a filter narrows the list. */
  filtered?: boolean;
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

const LABEL_MAX_CHARS = 48;

/**
 * How a take is named in labels and notices: the opening of its text (as
 * its record page title is), not the generated audio filename.
 */
function rowLabel(
  record: SpeechHistoryRecordSummary,
  voiceLabel: string,
): string {
  const text = record.input_preview.replace(/\s+/g, " ").trim();
  if (text) {
    return text.length > LABEL_MAX_CHARS
      ? `${text.slice(0, LABEL_MAX_CHARS).trimEnd()}…`
      : text;
  }
  return voiceLabel || record.model_id || record.id;
}

function initials(label: string): string {
  return (
    label
      .split(/\s+/)
      .filter(Boolean)
      .map((part) => part[0])
      .join("")
      .slice(0, 2)
      .toUpperCase() || "?"
  );
}

const QUICK_ACTION_CLASS =
  "h-7 w-7 rounded-lg text-(--text-subtle) opacity-0 transition-opacity hover:bg-(--bg-surface-2) hover:text-(--text-primary) focus-visible:opacity-100 disabled:opacity-0 group-hover:opacity-100 disabled:group-hover:opacity-40 group-focus-within:opacity-100";

export function TextToSpeechHistoryTable({
  records,
  total = null,
  filtered = false,
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
  const preview = useAudioPreview(() =>
    notify({ title: "Could not play this generation", tone: "warning" }),
  );

  const voiceLabelFor = (record: SpeechHistoryRecordSummary) =>
    resolveSpeechVoiceLabel({
      savedVoiceId: record.saved_voice_id,
      speaker: record.speaker,
      modelId: record.model_id,
      savedVoiceNameById,
    });
  const deleteTargetLabel = deleteTarget
    ? rowLabel(deleteTarget, voiceLabelFor(deleteTarget))
    : "Speech generation";

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
      const saved = await api.downloadAudioFile(
        audioUrl,
        speechDownloadFilename(record),
      );
      // The desktop app's save dialog can be cancelled.
      if (saved) {
        notify({
          title: "Download started",
          description: rowLabel(record, voiceLabel),
          tone: "success",
        });
      }
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

  const columns: HistoryColumn<SpeechHistoryRecordSummary>[] = [
    {
      id: "play",
      header: <span className="sr-only">Play</span>,
      className: "w-13 pr-0",
      cell: (record) => {
        const ready =
          normalizeSpeechProcessingStatus(
            record.processing_status,
            record.processing_error,
          ) === "ready";
        const playing = preview.playingId === record.id;
        return (
          <button
            type="button"
            data-row-action
            disabled={!ready}
            onClick={() =>
              preview.toggle(record.id, api.textToSpeechRecordAudioUrl(record.id))
            }
            aria-label={`${playing ? "Pause" : "Play"} ${rowLabel(record, voiceLabelFor(record))}`}
            className={cn(
              "inline-flex h-[30px] w-[30px] items-center justify-center rounded-full border border-(--border-default) bg-(--surface-panel-alt) text-(--text-primary) transition-colors hover:border-(--border-strong) disabled:cursor-not-allowed disabled:opacity-40",
              playing && "border-(--text-primary)",
            )}
          >
            {playing ? (
              <Pause className="h-3 w-3" fill="currentColor" />
            ) : (
              <Play className="ml-0.5 h-3 w-3" fill="currentColor" />
            )}
          </button>
        );
      },
    },
    {
      id: "text",
      header: "Text",
      className: "max-w-0",
      cell: (record) => {
        const status = normalizeSpeechProcessingStatus(
          record.processing_status,
          record.processing_error,
        );
        return (
          <div className="min-w-0">
            <div className="truncate text-(--text-primary)">
              {record.input_preview || "Untitled generation"}
            </div>
            <div className="mt-0.5 flex items-center gap-1.5 text-xs text-(--text-subtle)">
              {status === "failed" ? (
                <RowBadge tone="danger" className="h-[18px] text-[11px]">
                  Failed
                </RowBadge>
              ) : status === "pending" || status === "processing" ? (
                <RowBadge tone="info" dot className="h-[18px] text-[11px]">
                  {status === "pending" ? "Queued" : "Generating"}
                </RowBadge>
              ) : null}
              <span className="truncate">
                {record.model_id ? getModelDisplayName(record.model_id) : ""}
              </span>
            </div>
          </div>
        );
      },
    },
    {
      id: "voice",
      header: "Voice",
      className: "hidden w-40 md:table-cell",
      cell: (record) => {
        const label = voiceLabelFor(record);
        return (
          <span className="inline-flex max-w-full items-center gap-2 text-[13px] text-(--text-secondary)">
            <span
              aria-hidden="true"
              className="inline-flex h-[22px] w-[22px] shrink-0 items-center justify-center rounded-full border border-(--speaker-avatar-border) bg-(--speaker-avatar-bg) text-[10px] font-semibold text-(--speaker-avatar-text)"
            >
              {initials(label)}
            </span>
            <span className="truncate">{label}</span>
          </span>
        );
      },
    },
    {
      id: "length",
      header: "Length",
      className: "hidden w-22 lg:table-cell",
      cellClassName: "whitespace-nowrap tabular-nums text-(--text-muted)",
      cell: (record) => formatClockDuration(record.audio_duration_secs),
    },
    {
      id: "created",
      header: "Created",
      className: "hidden w-32 lg:table-cell",
      cellClassName: "whitespace-nowrap text-[13px] text-(--text-subtle)",
      cell: (record) => formatHistoryDate(record.created_at),
    },
  ];

  return (
    <>
      <HistoryTable
        rows={records}
        columns={columns}
        getRowId={(record) => record.id}
        getRowLabel={(record) =>
          `speech "${rowLabel(record, voiceLabelFor(record))}"`
        }
        onOpenRow={(record) => onOpenRecord(record.id)}
        loading={loading}
        error={error}
        onRetry={onRefresh}
        filtered={filtered}
        total={total}
        noun="generations"
        empty={{
          title: "No speech yet",
          description:
            "Generate speech from text. Queued, running and finished takes appear here.",
        }}
        loadMore={loadMore}
        rowActions={(record) => {
          const voiceLabel = voiceLabelFor(record);
          const label = rowLabel(record, voiceLabel);
          const status = normalizeSpeechProcessingStatus(
            record.processing_status,
            record.processing_error,
          );
          const isBusy = busyRecordId === record.id;
          const canDownload = status === "ready";
          const canCancel = status === "pending" || status === "processing";
          return (
            <>
              {canDownload ? (
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className={QUICK_ACTION_CLASS}
                  disabled={isBusy || deletePending}
                  onClick={() => void handleDownload(record, voiceLabel)}
                  aria-label={`Download ${label}`}
                  title="Download"
                >
                  <Download className="h-4 w-4" />
                </Button>
              ) : null}
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className={QUICK_ACTION_CLASS}
                disabled={isBusy || deletePending}
                onClick={() => void handleCopy(record)}
                aria-label={`Copy text of ${label}`}
                title="Copy text"
              >
                <Copy className="h-4 w-4" />
              </Button>
              <RowMenu
                label={label}
                busy={isBusy}
                items={[
                  {
                    id: "open",
                    label: "Open record",
                    icon: ExternalLink,
                    onSelect: () => onOpenRecord(record.id),
                  },
                  {
                    id: "copy",
                    label: "Copy text",
                    icon: Copy,
                    disabled: isBusy || deletePending,
                    onSelect: () => void handleCopy(record),
                  },
                  {
                    id: "download",
                    label: "Download",
                    icon: Download,
                    disabled: isBusy || deletePending || !canDownload,
                    onSelect: () => void handleDownload(record, voiceLabel),
                  },
                  ...(canCancel
                    ? [
                        {
                          id: "cancel",
                          label: "Cancel generation",
                          icon: OctagonX,
                          disabled: isBusy || deletePending || !onCancelRecord,
                          onSelect: () => void handleCancel(record),
                        },
                      ]
                    : []),
                  {
                    id: "delete",
                    label: "Delete",
                    icon: Trash2,
                    danger: true,
                    separatorBefore: true,
                    disabled: !onDeleteRecord || deletePending,
                    onSelect: () => {
                      setDeleteError(null);
                      setDeleteTarget(record);
                    },
                  },
                ]}
              />
            </>
          );
        }}
      />

      <ConfirmDeleteDialog
        noun={deleteTarget ? "generation" : null}
        itemLabel={deleteTargetLabel}
        description="This permanently removes the generated audio and its text from history."
        pending={deletePending}
        error={deleteError}
        onConfirm={() => void handleConfirmDelete()}
        onCancel={() => {
          setDeleteTarget(null);
          setDeleteError(null);
        }}
      />
    </>
  );
}
