import { useMemo, useState } from "react";
import {
  AlertTriangle,
  CalendarDays,
  Check,
  Clock,
  Copy,
  Cpu,
  Download,
  Loader2,
  Mic,
  OctagonX,
  Trash2,
} from "lucide-react";

import { api, type SpeechHistoryRecord } from "@/api";
import { RecordAudioPlayer } from "@/components/audio/RecordAudioPlayer";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  formatClockDuration,
  formatHistoryDate,
} from "@/components/history/format";
import type { RowMenuItem } from "@/components/history/RowMenu";
import type { Breadcrumb } from "@/components/PageShell";
import { RecordPage, type RecordMetaItem } from "@/components/RecordPage";
import { getModelDisplayName } from "@/features/models/catalog/modelMetadata";
import { normalizeSpeechProcessingStatus } from "@/features/text-to-speech/support";

const TITLE_MAX_CHARS = 60;

/** The opening of the spoken text, which names a take better than its model. */
function speechRecordTitle(record: SpeechHistoryRecord | null): string {
  const text = record?.input_text?.replace(/\s+/g, " ").trim() ?? "";
  if (!text) {
    return "Speech generation";
  }
  return text.length > TITLE_MAX_CHARS
    ? `${text.slice(0, TITLE_MAX_CHARS).trimEnd()}…`
    : text;
}

interface TextToSpeechRecordDetailProps {
  record: SpeechHistoryRecord | null;
  audioUrl: string | null;
  voiceLabel?: string | null;
  loading?: boolean;
  error?: string | null;
  deleteError?: string | null;
  deletePending?: boolean;
  cancelPending?: boolean;
  onCancel?: () => Promise<void> | void;
  onDelete?: () => Promise<void> | void;
  /** The list this record belongs to, for the breadcrumb trail. */
  parentCrumb: Breadcrumb;
  /** Route actions for the header menu (Models). */
  extraMenuItems?: RowMenuItem[];
}

export function TextToSpeechRecordDetail({
  record,
  audioUrl,
  voiceLabel = null,
  loading = false,
  error = null,
  deleteError = null,
  deletePending = false,
  cancelPending = false,
  onCancel,
  onDelete,
  parentCrumb,
  extraMenuItems = [],
}: TextToSpeechRecordDetailProps) {
  const [copied, setCopied] = useState(false);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);

  const processingStatus = useMemo(
    () =>
      normalizeSpeechProcessingStatus(
        record?.processing_status,
        record?.processing_error,
      ),
    [record?.processing_error, record?.processing_status],
  );
  const hasAudio = Boolean(audioUrl) && processingStatus === "ready";
  const canCancel = processingStatus === "pending" || processingStatus === "processing";
  const statusMessage = useMemo(() => {
    switch (processingStatus) {
      case "pending":
        return "This generation is queued and will start shortly.";
      case "processing":
        return "This generation is actively rendering audio. The player and metrics will update automatically.";
      case "failed":
        return record?.processing_error || "Speech generation failed.";
      case "ready":
      default:
        return null;
    }
  }, [processingStatus, record?.processing_error]);

  async function handleCopyInput(): Promise<void> {
    if (!record?.input_text) {
      return;
    }
    await navigator.clipboard.writeText(record.input_text);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  }

  async function handleDownloadAudio(): Promise<void> {
    if (!hasAudio || !audioUrl || !record) {
      return;
    }
    const filename =
      record.audio_filename ||
      `tts-${record.id}.${record.audio_mime_type.includes("wav") ? "wav" : "audio"}`;
    await api.downloadAudioFile(audioUrl, filename);
  }

  async function handleConfirmDelete(): Promise<void> {
    if (!onDelete || deletePending) {
      return;
    }
    await onDelete();
  }

  const title = speechRecordTitle(record);
  const meta: RecordMetaItem[] = record
    ? [
        { icon: CalendarDays, label: formatHistoryDate(record.created_at) },
        ...(record.audio_duration_secs != null
          ? [
              {
                icon: Clock,
                label: formatClockDuration(record.audio_duration_secs),
              },
            ]
          : []),
        ...(voiceLabel ? [{ icon: Mic, label: voiceLabel, title: "Voice" }] : []),
        ...(record.model_id
          ? [
              {
                icon: Cpu,
                label: getModelDisplayName(record.model_id),
                title: "Model",
              },
            ]
          : []),
      ]
    : [];
  const menuItems: RowMenuItem[] = [
    ...(onCancel && canCancel
      ? [
          {
            id: "cancel",
            label: cancelPending ? "Cancelling…" : "Cancel generation",
            icon: OctagonX,
            disabled: cancelPending,
            onSelect: () => void onCancel(),
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
            onClick={() => void handleCopyInput()}
            disabled={!record?.input_text}
          >
            {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
            Copy text
          </Button>
          <Button
            type="button"
            size="sm"
            className="h-9 gap-2"
            onClick={() => void handleDownloadAudio()}
            disabled={!hasAudio}
          >
            <Download className="h-4 w-4" />
            Download
          </Button>
        </>
      }
    >
      {error ? (
        <Card
          className="border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-text)]"
          role="alert"
          aria-live="assertive"
        >
          {error}
        </Card>
      ) : null}

      {statusMessage ? (
        <Card
          role={processingStatus === "failed" ? "alert" : "status"}
          aria-live={processingStatus === "failed" ? "assertive" : "polite"}
          aria-atomic="true"
          className={
            processingStatus === "failed"
              ? "border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-text)]"
              : "border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] p-4 text-sm text-[var(--status-warning-text)]"
          }
        >
          <div className="flex items-start gap-3">
            {processingStatus === "failed" ? (
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            ) : (
              <Loader2 className="mt-0.5 h-4 w-4 shrink-0 animate-spin" />
            )}
            <p>{statusMessage}</p>
          </div>
        </Card>
      ) : null}

      <Card className="border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-[var(--text-muted)]">
          Audio
        </h3>
        {loading ? (
          <p
            className="mt-3 text-sm text-[var(--text-muted)]"
            role="status"
            aria-live="polite"
          >
            Loading record...
          </p>
        ) : hasAudio ? (
          <RecordAudioPlayer
            src={audioUrl ?? ""}
            durationHint={record?.audio_duration_secs}
            className="mt-3"
          />
        ) : (
          <p className="mt-3 text-sm text-[var(--text-muted)]">
            Audio is not available yet.
          </p>
        )}
      </Card>

      <Card className="border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-[var(--text-muted)]">
          Input text
        </h3>
        <p className="mt-3 whitespace-pre-wrap rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4 text-sm leading-6 text-[var(--text-primary)]">
          {record?.input_text || "No input text available."}
        </p>
      </Card>

      <Card className="border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5">
        <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-[var(--text-muted)]">
          Generation stats
        </h3>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3">
            <div className="text-xs text-[var(--text-muted)]">Generation time</div>
            <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
              {record ? `${record.generation_time_ms.toFixed(0)} ms` : "n/a"}
            </div>
          </div>
          <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3">
            <div className="text-xs text-[var(--text-muted)]">Audio duration</div>
            <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
              {formatClockDuration(record?.audio_duration_secs ?? null)}
            </div>
          </div>
          <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3">
            <div className="text-xs text-[var(--text-muted)]">RTF</div>
            <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
              {record?.rtf != null ? record.rtf.toFixed(2) : "n/a"}
            </div>
          </div>
          <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3">
            <div className="text-xs text-[var(--text-muted)]">Tokens generated</div>
            <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
              {record?.tokens_generated ?? "n/a"}
            </div>
          </div>
        </div>
      </Card>

      <Dialog
        open={deleteConfirmOpen}
        onOpenChange={(open) => {
          if (!deletePending) {
            setDeleteConfirmOpen(open);
          }
        }}
      >
        <DialogContent className="max-w-md border-[var(--border-strong)] bg-[var(--bg-surface-1)] p-5">
          <DialogTitle className="sr-only">Delete speech generation?</DialogTitle>
          <div className="flex items-start gap-3">
            <div className="mt-0.5 rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2 text-[var(--danger-text)]">
              <AlertTriangle className="h-4 w-4" />
            </div>
            <div className="min-w-0 flex-1">
              <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                Delete generation?
              </h3>
              <DialogDescription className="mt-1 text-sm text-[var(--text-muted)]">
                This permanently removes the saved audio and metadata from
                history.
              </DialogDescription>
              <p className="mt-2 truncate text-xs text-[var(--text-subtle)]">
                {record?.model_id || record?.id || "Record"}
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
                "Delete generation"
              )}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </RecordPage>
  );
}
