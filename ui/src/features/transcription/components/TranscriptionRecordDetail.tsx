import { useMemo, useState } from "react";
import {
  AlertTriangle,
  CalendarDays,
  Check,
  Clock,
  Copy,
  Cpu,
  Download,
  Languages,
  Loader2,
  RotateCcw,
  Trash2,
  Type,
} from "lucide-react";

import {
  type TranscriptionProcessingProgress,
  type TranscriptionRecord,
} from "@/api";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { TranscriptionExportDialog } from "@/features/transcription/components/TranscriptionExportDialog";
import { TranscriptionReviewWorkspace } from "@/features/transcription/components/TranscriptionReviewWorkspace";
import {
  formatClockDuration,
  formatHistoryDate,
} from "@/components/history/format";
import type { RowMenuItem } from "@/components/history/RowMenu";
import type { Breadcrumb } from "@/components/PageShell";
import { RecordPage, type RecordMetaItem } from "@/components/RecordPage";
import { getModelDisplayName } from "@/features/models/catalog/modelMetadata";
import { normalizeProcessingStatus } from "@/features/transcription/playground/support";
import {
  formatTranscriptionText,
  transcriptionWordCount,
} from "@/features/transcription/transcript";

interface TranscriptionRecordDetailProps {
  record: TranscriptionRecord | null;
  audioUrl: string | null;
  loading?: boolean;
  error?: string | null;
  deleteError?: string | null;
  summaryModelGuidance?: string | null;
  onDelete?: () => Promise<void> | void;
  onRegenerateSummary?: () => void;
  deletePending?: boolean;
  summaryRefreshPending?: boolean;
  summaryRefreshError?: string | null;
  /** The list this record belongs to, for the breadcrumb trail. */
  parentCrumb: Breadcrumb;
  /** Route actions for the header menu (Models). */
  extraMenuItems?: RowMenuItem[];
}

function progressPhaseLabel(
  progress: TranscriptionProcessingProgress | null | undefined,
): string {
  switch (progress?.phase) {
    case "chunk_started":
    case "chunk_finished":
      return "Transcribing audio";
    case "aligning":
      return "Aligning timestamps";
    case "complete":
      return "Finalizing transcript";
    case "processing":
    default:
      return "Preparing transcription";
  }
}

function clampProgressPercent(
  progress: TranscriptionProcessingProgress | null | undefined,
): number | null {
  const percent = progress?.percent;
  if (typeof percent !== "number" || !Number.isFinite(percent)) {
    return null;
  }
  return Math.min(100, Math.max(0, percent));
}

function progressDetailLabel(
  progress: TranscriptionProcessingProgress | null | undefined,
): string | null {
  const percent = clampProgressPercent(progress);
  if (percent !== null) {
    return `${Math.round(percent)}%`;
  }

  const processedAudioSecs = progress?.processed_audio_secs;
  const totalAudioSecs = progress?.total_audio_secs;
  if (
    typeof processedAudioSecs === "number" &&
    Number.isFinite(processedAudioSecs) &&
    processedAudioSecs >= 0 &&
    typeof totalAudioSecs === "number" &&
    Number.isFinite(totalAudioSecs) &&
    totalAudioSecs > 0
  ) {
    const derivedPercent = Math.min(
      100,
      Math.max(0, (processedAudioSecs / totalAudioSecs) * 100),
    );
    return `${Math.round(derivedPercent)}%`;
  }

  return null;
}

function ProcessingProgressCard({
  progress,
}: {
  progress: TranscriptionProcessingProgress | null | undefined;
}) {
  const percent = clampProgressPercent(progress);
  const detail = progressDetailLabel(progress);

  return (
    <Card
      className="border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] p-4 text-sm text-[var(--status-warning-text)]"
      role="status"
      aria-live="polite"
      aria-atomic="true"
    >
      <div className="flex items-start gap-3">
        <Loader2 className="mt-0.5 h-4 w-4 shrink-0 animate-spin" />
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-3">
            <p className="font-medium">{progressPhaseLabel(progress)}</p>
            {detail ? (
              <span className="shrink-0 text-xs text-[var(--text-muted)]">
                {detail}
              </span>
            ) : null}
          </div>
          <div
            className="mt-3 h-2 overflow-hidden rounded-full bg-[var(--border-muted)]"
            role="progressbar"
            aria-label="ASR processing progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={percent ?? undefined}
          >
            <div
              className={
                percent === null
                  ? "h-full w-1/3 animate-pulse rounded-full bg-[var(--status-warning-text)]"
                  : "h-full rounded-full bg-[var(--status-warning-text)] transition-[width] duration-300"
              }
              style={percent === null ? undefined : { width: `${percent}%` }}
            />
          </div>
        </div>
      </div>
    </Card>
  );
}

export function TranscriptionRecordDetail({
  record,
  audioUrl,
  loading = false,
  error = null,
  deleteError = null,
  summaryModelGuidance = null,
  onDelete,
  onRegenerateSummary,
  deletePending = false,
  summaryRefreshPending = false,
  summaryRefreshError = null,
  parentCrumb,
  extraMenuItems = [],
}: TranscriptionRecordDetailProps) {
  const [copied, setCopied] = useState(false);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);

  const processingStatus = useMemo(
    () =>
      normalizeProcessingStatus(
        record?.processing_status,
        record?.processing_error,
      ),
    [record?.processing_error, record?.processing_status],
  );
  const exportText = useMemo(() => formatTranscriptionText(record), [record]);
  const hasTranscript = useMemo(
    () => (record?.transcription ?? "").trim().length > 0,
    [record?.transcription],
  );
  const saaWarning = useMemo(() => {
    const warnings = record?.saa_warnings ?? [];
    if (record?.saa_status !== "warning" || warnings.length === 0) {
      return null;
    }
    return warnings.join(" ");
  }, [record?.saa_status, record?.saa_warnings]);
  const alignmentWarning = useMemo(() => {
    const warnings = record?.alignment_warnings ?? [];
    return warnings.length > 0 ? warnings.join(" ") : null;
  }, [record?.alignment_warnings]);
  const statusMessage = useMemo(
    () =>
      processingStatus === "failed"
        ? record?.processing_error || "Transcription processing failed."
        : null,
    [processingStatus, record?.processing_error],
  );

  async function handleCopy(): Promise<void> {
    if (!exportText) {
      return;
    }
    await navigator.clipboard.writeText(exportText);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  }

  async function handleConfirmDelete(): Promise<void> {
    if (!onDelete || deletePending) {
      return;
    }

    await onDelete();
  }

  const title =
    record?.audio_filename || record?.model_id || "Transcription record";
  const wordCount = record ? transcriptionWordCount(record) : 0;
  const meta: RecordMetaItem[] = record
    ? [
        { icon: CalendarDays, label: formatHistoryDate(record.created_at) },
        ...(record.duration_secs != null
          ? [{ icon: Clock, label: formatClockDuration(record.duration_secs) }]
          : []),
        ...(record.language ? [{ icon: Languages, label: record.language }] : []),
        ...(record.model_id
          ? [
              {
                icon: Cpu,
                label: getModelDisplayName(record.model_id),
                title: "Model",
              },
            ]
          : []),
        ...(wordCount > 0
          ? [
              {
                icon: Type,
                label: `${wordCount} word${wordCount === 1 ? "" : "s"}`,
              },
            ]
          : []),
      ]
    : [];
  const menuItems: RowMenuItem[] = [
    ...(onRegenerateSummary
      ? [
          {
            id: "summary",
            label: summaryRefreshPending
              ? "Regenerating summary…"
              : "Regenerate summary",
            icon: summaryRefreshPending ? Loader2 : RotateCcw,
            disabled:
              !record ||
              processingStatus !== "ready" ||
              summaryRefreshPending ||
              !hasTranscript,
            onSelect: onRegenerateSummary,
          },
        ]
      : []),
    ...extraMenuItems,
    ...(onDelete
      ? [
          {
            id: "delete",
            label: "Delete",
            icon: Trash2,
            danger: true,
            separatorBefore: true,
            disabled: deletePending,
            onSelect: () => setDeleteConfirmOpen(true),
          },
        ]
      : []),
  ];

  return (
    <RecordPage
      title={title}
      breadcrumbs={[parentCrumb, { label: title }]}
      meta={meta}
      menuItems={menuItems}
      actions={
        <>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-9 gap-2"
            onClick={() => void handleCopy()}
            disabled={!hasTranscript}
          >
            {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
            Copy
          </Button>
          <TranscriptionExportDialog record={record}>
            <Button
              type="button"
              size="sm"
              className="h-9 gap-2"
              disabled={!hasTranscript}
            >
              <Download className="h-4 w-4" />
              Export
            </Button>
          </TranscriptionExportDialog>
        </>
      }
      alerts={
        <>
          {error ? (
            <Card
              className="border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-text)]"
              role="alert"
              aria-live="assertive"
            >
              {error}
            </Card>
          ) : null}

          {processingStatus === "pending" || processingStatus === "processing" ? (
            <ProcessingProgressCard progress={record?.processing_progress} />
          ) : null}

          {statusMessage ? (
            <Card
              className="border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-text)]"
              role="alert"
              aria-live="assertive"
            >
              <div className="flex items-start gap-3">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                <p>{statusMessage}</p>
              </div>
            </Card>
          ) : null}

          {saaWarning ? (
            <Card
              className="border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] p-4 text-sm text-[var(--status-warning-text)]"
              role="status"
              aria-live="polite"
            >
              <div className="flex items-start gap-3">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                <p>{saaWarning}</p>
              </div>
            </Card>
          ) : null}

          {alignmentWarning ? (
            <Card
              className="border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] p-4 text-sm text-[var(--status-warning-text)]"
              role="status"
              aria-live="polite"
            >
              <div className="flex items-start gap-3">
                <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                <p>{alignmentWarning}</p>
              </div>
            </Card>
          ) : null}

          {summaryRefreshError ? (
            <Card
              className="border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-text)]"
              role="alert"
              aria-live="assertive"
            >
              {summaryRefreshError}
            </Card>
          ) : null}
        </>
      }
    >
      <TranscriptionReviewWorkspace
        record={record}
        audioUrl={audioUrl}
        loading={loading}
        autoScrollActiveEntry={true}
        fixedPlaybackFooter={true}
        summaryModelGuidance={
          processingStatus === "ready" ? summaryModelGuidance : null
        }
        emptyTitle={
          processingStatus === "pending" || processingStatus === "processing"
            ? "Transcription in progress"
            : "No transcript"
        }
        emptyMessage={
          processingStatus === "failed"
            ? "This job failed before producing a transcript."
            : processingStatus === "pending" || processingStatus === "processing"
              ? "The transcript will appear here as soon as this record is ready."
              : "No speech was recognised in this recording."
        }
      />

      <Dialog
        open={deleteConfirmOpen}
        onOpenChange={(open) => {
          if (!deletePending) {
            setDeleteConfirmOpen(open);
          }
        }}
      >
        <DialogContent className="max-w-md border-[var(--border-strong)] bg-[var(--bg-surface-1)] p-5">
          <DialogTitle className="sr-only">Delete transcription?</DialogTitle>
          <div className="flex items-start gap-3">
            <div className="mt-0.5 rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2 text-[var(--danger-text)]">
              <AlertTriangle className="h-4 w-4" />
            </div>
            <div className="min-w-0 flex-1">
              <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                Delete transcription?
              </h3>
              <DialogDescription className="mt-1 text-sm text-[var(--text-muted)]">
                This permanently removes the saved audio and transcript from
                history.
              </DialogDescription>
              <p className="mt-2 truncate text-xs text-[var(--text-subtle)]">
                {record?.audio_filename ||
                  record?.model_id ||
                  record?.id ||
                  "Transcription record"}
              </p>
            </div>
          </div>

          {deleteError ? (
            <div
              className="mt-4 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]"
              role="alert"
              aria-live="assertive"
            >
              {deleteError}
            </div>
          ) : null}

          <div className="mt-5 flex items-center justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => setDeleteConfirmOpen(false)}
              size="sm"
              className="h-8 border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)] hover:bg-[var(--bg-surface-3)]"
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
                "Delete transcription"
              )}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </RecordPage>
  );
}
