import { useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowLeft,
  Check,
  Copy,
  Download,
  Loader2,
  OctagonX,
  Trash2,
} from "lucide-react";

import { api, type SpeechHistoryRecord } from "@/api";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  formatSpeechCreatedAt,
  formatSpeechDuration,
  normalizeSpeechProcessingStatus,
} from "@/features/text-to-speech/support";

interface TextToSpeechRecordDetailProps {
  record: SpeechHistoryRecord | null;
  audioUrl: string | null;
  voiceLabel?: string | null;
  loading?: boolean;
  error?: string | null;
  deleteError?: string | null;
  deletePending?: boolean;
  cancelPending?: boolean;
  onBack?: () => void;
  onCancel?: () => Promise<void> | void;
  onDelete?: () => Promise<void> | void;
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
  onBack,
  onCancel,
  onDelete,
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

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          {onBack ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="mb-4 h-10 gap-2 rounded-full border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 text-sm font-medium text-[var(--text-secondary)] shadow-sm hover:bg-[var(--bg-surface-1)]"
              onClick={onBack}
            >
              <ArrowLeft className="h-4 w-4" />
              Back to history
            </Button>
          ) : null}
          <h2 className="truncate text-2xl font-semibold tracking-tight text-[var(--text-primary)]">
            {record?.model_id || "Speech generation"}
          </h2>
          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-[var(--text-muted)]">
            {record ? <span>{formatSpeechCreatedAt(record.created_at)}</span> : null}
            {record?.audio_duration_secs != null ? (
              <span>{formatSpeechDuration(record.audio_duration_secs)}</span>
            ) : null}
            {record?.model_id ? <span>{record.model_id}</span> : null}
            {voiceLabel ? <span>Voice: {voiceLabel}</span> : null}
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-end gap-2">
          {onCancel && canCancel ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 gap-2 border-[var(--status-warning-border)] text-[var(--status-warning-text)]"
              onClick={() => void onCancel()}
              disabled={cancelPending}
            >
              {cancelPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <OctagonX className="h-4 w-4" />
              )}
              Cancel generation
            </Button>
          ) : null}
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
            variant="outline"
            size="sm"
            className="h-9 gap-2"
            onClick={() => void handleDownloadAudio()}
            disabled={!hasAudio}
          >
            <Download className="h-4 w-4" />
            Download
          </Button>
          {onDelete ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 gap-2 border-[var(--danger-border)] text-[var(--danger-text)] hover:bg-[var(--danger-bg)]"
              onClick={() => setDeleteConfirmOpen(true)}
              disabled={deletePending}
            >
              {deletePending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Trash2 className="h-4 w-4" />
              )}
              Delete
            </Button>
          ) : null}
        </div>
      </div>

      {error ? (
        <Card className="border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-text)]">
          {error}
        </Card>
      ) : null}

      {statusMessage ? (
        <Card
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
          <p className="mt-3 text-sm text-[var(--text-muted)]">Loading record...</p>
        ) : hasAudio ? (
          <audio src={audioUrl || undefined} className="mt-3 h-11 w-full" controls />
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
              {formatSpeechDuration(record?.audio_duration_secs ?? null)}
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
            <div className="mt-4 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]">
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
    </div>
  );
}
