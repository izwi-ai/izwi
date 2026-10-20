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
import { FactsList, formatProcessingFact } from "@/components/FactsList";
import { RecordPage, type RecordMetaItem } from "@/components/RecordPage";
import { getModelDisplayName } from "@/features/models/catalog/modelMetadata";
import {
  normalizeSpeechProcessingStatus,
  speechDownloadFilename,
} from "@/features/text-to-speech/support";

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
  /** Why an action on the page (cancel) failed. */
  actionError?: string | null;
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
  actionError = null,
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
  // A failed attempt that will run again is still in progress.
  const retrying = processingStatus === "failed" && Boolean(record?.retry_pending);
  const canCancel =
    processingStatus === "pending" || processingStatus === "processing" || retrying;
  const truncated =
    processingStatus === "ready" &&
    (record?.stop_reason === "length" || record?.stop_reason === "context_length");
  const statusMessage = useMemo(() => {
    if (retrying) {
      return `Retrying after an error: ${record?.processing_error || "the last attempt failed"}.`;
    }
    if (truncated) {
      return "The model reached its length limit before the end of the script, so the audio may stop early. Split the script or use a model with a longer limit.";
    }
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
  }, [processingStatus, record?.processing_error, retrying, truncated]);
  const statusIsError = processingStatus === "failed" && !retrying;

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
    await api.downloadAudioFile(audioUrl, speechDownloadFilename(record));
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
      {error || actionError ? (
        <Card
          className="border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-text)]"
          role="alert"
          aria-live="assertive"
        >
          {error ?? actionError}
        </Card>
      ) : null}

      {statusMessage ? (
        <Card
          role={statusIsError ? "alert" : "status"}
          aria-live={statusIsError ? "assertive" : "polite"}
          aria-atomic="true"
          className={
            statusIsError
              ? "border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-text)]"
              : "border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] p-4 text-sm text-[var(--status-warning-text)]"
          }
        >
          <div className="flex items-start gap-3">
            {canCancel ? (
              <Loader2 className="mt-0.5 h-4 w-4 shrink-0 animate-spin" />
            ) : (
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            )}
            <p>{statusMessage}</p>
          </div>
        </Card>
      ) : null}

            <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr),280px]">
        <div className="space-y-5">
    <Card className="rounded-[14px] border-[var(--border-subtle)] bg-[var(--surface-panel)] p-4">
            <h3 className="text-sm font-semibold text-[var(--text-primary)]">
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

          <Card className="rounded-[14px] border-[var(--border-subtle)] bg-[var(--surface-panel)] p-4">
            <h3 className="text-sm font-semibold text-[var(--text-primary)]">
              Input text
            </h3>
            <p className="mt-3 whitespace-pre-wrap rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4 text-sm leading-6 text-[var(--text-primary)]">
              {record?.input_text || "No input text available."}
            </p>
          </Card>

      
        </div>
        <FactsList
          label="Generation"
          className="self-start"
          facts={[
            {
              label: "Generated in",
              value:
                formatProcessingFact(record?.generation_time_ms, record?.rtf) ??
                "—",
            },
            {
              label: "Length",
              value: formatClockDuration(record?.audio_duration_secs ?? null),
            },
            { label: "Tokens", value: record?.tokens_generated ?? "—" },
            ...(record?.speed != null && record.speed !== 1
              ? [{ label: "Speed", value: `${record.speed}×` }]
              : []),
            ...(record?.language
              ? [{ label: "Language", value: record.language }]
              : []),
          ]}
        />
      </div>

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
