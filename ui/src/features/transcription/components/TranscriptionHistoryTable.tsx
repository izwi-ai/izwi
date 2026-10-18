import { useState } from "react";
import {
  AudioLines,
  Copy,
  Download,
  ExternalLink,
  Loader2,
  Sparkles,
  Trash2,
  Users,
} from "lucide-react";

import {
  api,
  type DiarizationRecord,
  type SpeechTextDiarizationSummary,
  type SpeechTextSpeakerAttributedAsrSummary,
  type SpeechTextJobSummary,
} from "@/api";
import { useNotifications } from "@/app/providers/NotificationProvider";
import { DiarizationExportDialog } from "@/components/DiarizationExportDialog";
import { ConfirmDeleteDialog } from "@/components/history/ConfirmDeleteDialog";
import {
  formatClockDuration,
  formatHistoryDate,
} from "@/components/history/format";
import { HistoryTable, type HistoryColumn } from "@/components/history/HistoryTable";
import { RowBadge } from "@/components/history/RowBadge";
import { RowMenu } from "@/components/history/RowMenu";
import { TranscriptionExportDialog } from "@/features/transcription/components/TranscriptionExportDialog";
import { presentSpeechTextHistoryStatus } from "@/features/transcription/historyStatus";
import { formatTranscriptionText } from "@/features/transcription/transcript";
import { formattedTranscriptFromRecord } from "@/utils/diarizationTranscript";
import type { ExportableTranscriptionRecord } from "@/utils/transcriptionExport";

interface TranscriptionHistoryTableProps {
  records: SpeechTextJobSummary[];
  /** Records matching the filters, if the server reports it. */
  total?: number | null;
  /** Search or a filter narrows the list. */
  filtered?: boolean;
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

function rowKind(
  record: SpeechTextJobSummary,
): "transcription" | "speaker_attributed_asr" | "diarization" {
  return record.kind;
}

function isDiarizationSummary(
  record: SpeechTextJobSummary,
): record is SpeechTextDiarizationSummary {
  return record.kind === "diarization";
}

function isSpeakerAttributedAsrSummary(
  record: SpeechTextJobSummary,
): record is SpeechTextSpeakerAttributedAsrSummary {
  return record.kind === "speaker_attributed_asr";
}

function rowKindLabel(record: SpeechTextJobSummary): string {
  switch (rowKind(record)) {
    case "diarization":
      return "Diarization";
    case "speaker_attributed_asr":
      return "Speaker Attributed ASR";
    case "transcription":
    default:
      return "Transcription";
  }
}

function rowPreview(record: SpeechTextJobSummary): string {
  if (isDiarizationSummary(record)) {
    return record.transcript_preview;
  }
  if (isSpeakerAttributedAsrSummary(record)) {
    return record.speaker_attributed_text_preview || record.transcription_preview;
  }
  return record.transcription_preview;
}

function rowLabel(record: SpeechTextJobSummary): string {
  return record.audio_filename || record.model_id || record.id;
}

function speakerCount(record: SpeechTextJobSummary): number | null {
  if (isDiarizationSummary(record)) {
    return record.corrected_speaker_count ?? record.speaker_count;
  }
  return null;
}

function StatusCell({ record }: { record: SpeechTextJobSummary }) {
  const status = presentSpeechTextHistoryStatus(
    record.processing_status,
    record.processing_error,
    record.kind,
  ).status;
  if (status === "processing") {
    return (
      <RowBadge tone="info" dot>
        {record.kind === "diarization" ? "Diarizing" : "Transcribing"}
      </RowBadge>
    );
  }
  if (status === "queued") {
    return <RowBadge dot>Queued</RowBadge>;
  }
  if (status === "failed") {
    return (
      <RowBadge tone="danger" dot>
        Failed
      </RowBadge>
    );
  }
  if (status === "canceled") {
    return <RowBadge>Cancelled</RowBadge>;
  }
  if (status === "unknown") {
    return <RowBadge>Status unknown</RowBadge>;
  }

  const speakers = speakerCount(record);
  const badges = [];
  if (speakers != null) {
    badges.push(
      <RowBadge key="speakers" icon={<Users className="h-3 w-3" />}>
        {speakers} speaker{speakers === 1 ? "" : "s"}
      </RowBadge>,
    );
  } else if (record.kind === "speaker_attributed_asr") {
    badges.push(
      <RowBadge key="speakers" icon={<Users className="h-3 w-3" />}>
        Speakers
      </RowBadge>,
    );
  }
  if (record.summary_status === "ready") {
    badges.push(
      <RowBadge key="summary" icon={<Sparkles className="h-3 w-3" />}>
        Summary
      </RowBadge>,
    );
  } else if (record.summary_status === "pending") {
    badges.push(
      <RowBadge
        key="summary"
        tone="info"
        icon={<Loader2 className="h-3 w-3 animate-spin" />}
      >
        Summarising
      </RowBadge>
    );
  }
  return badges.length > 0 ? (
    <div className="flex flex-wrap gap-1">{badges}</div>
  ) : null;
}

function PreviewCell({ record }: { record: SpeechTextJobSummary }) {
  const presentation = presentSpeechTextHistoryStatus(
    record.processing_status,
    record.processing_error,
    record.kind,
  );
  const preview = rowPreview(record);
  if (presentation.status === "failed") {
    return (
      <span className="text-[13px] text-[var(--danger-text)]">
        {record.processing_error?.trim() || presentation.description}
      </span>
    );
  }
  if (presentation.status === "queued" || presentation.status === "processing") {
    return (
      <span className="text-[13px] text-[var(--text-subtle)]">
        {preview ||
          (presentation.status === "queued"
            ? "Waiting to start…"
            : "Preparing transcript…")}
      </span>
    );
  }
  if (presentation.status !== "ready") {
    // Never present output from an unknown or cancelled job as finished.
    return (
      <span className="text-[13px] text-[var(--text-subtle)]">
        {presentation.description}
      </span>
    );
  }
  return (
    <span className="text-[13px] text-[var(--text-muted)]">
      {preview || "No transcript text."}
    </span>
  );
}

const COLUMNS: HistoryColumn<SpeechTextJobSummary>[] = [
  {
    id: "file",
    header: "File",
    className: "w-[15rem] max-w-[15rem]",
    cell: (record) => {
      const active =
        record.processing_status === "pending" ||
        record.processing_status === "processing";
      return (
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-[9px] border border-[var(--border-subtle)] bg-[var(--surface-panel-alt)] text-[var(--text-muted)]">
            {active ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <AudioLines className="h-4 w-4" />
            )}
          </span>
          <span className="truncate font-medium text-[var(--text-primary)]">
            {record.audio_filename || "Audio input"}
          </span>
        </div>
      );
    },
  },
  {
    id: "transcript",
    header: "Transcript",
    className: "max-w-0",
    cell: (record) => (
      <div className="truncate">
        <PreviewCell record={record} />
      </div>
    ),
  },
  {
    id: "length",
    header: "Length",
    className: "w-[5.5rem] whitespace-nowrap tabular-nums text-[var(--text-muted)]",
    cell: (record) => formatClockDuration(record.duration_secs),
  },
  {
    id: "status",
    header: "Status",
    className: "w-[10rem]",
    cell: (record) => <StatusCell record={record} />,
  },
  {
    id: "created",
    header: "Created",
    className: "w-[8rem] whitespace-nowrap text-[13px] text-[var(--text-subtle)]",
    cell: (record) => formatHistoryDate(record.created_at),
  },
];

export function TranscriptionHistoryTable({
  records,
  total = null,
  filtered = false,
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

  const deleteTargetLabel = deleteTarget ? rowLabel(deleteTarget) : undefined;
  const deleteTargetKindLabel = deleteTarget
    ? rowKindLabel(deleteTarget).toLowerCase()
    : "record";

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
          : formatTranscriptionText(
              kind === "speaker_attributed_asr"
                ? await api.getSpeakerAttributedAsrRecord(record.id)
                : await api.getTranscriptionRecord(record.id),
            );

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
        setExportTranscriptionRecord(
          kind === "speaker_attributed_asr"
            ? await api.getSpeakerAttributedAsrRecord(record.id)
            : await api.getTranscriptionRecord(record.id),
        );
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

  return (
    <>
      <HistoryTable
        rows={records}
        columns={COLUMNS}
        getRowId={(record) => record.id}
        getRowLabel={(record) => `${rowKind(record)} ${rowLabel(record)}`}
        onOpenRow={onOpenRecord}
        loading={loading}
        error={error}
        onRetry={onRefresh}
        filtered={filtered}
        total={total}
        noun="transcripts"
        empty={{
          title: "No transcripts yet",
          description:
            "Transcribe a recording or file. Queued, running and finished jobs appear here.",
        }}
        loadMore={loadMore}
        rowActions={(record) => {
          const outputReady =
            presentSpeechTextHistoryStatus(
              record.processing_status,
              record.processing_error,
              record.kind,
            ).status === "ready";
          const isBusy = busyRecordId === record.id;
          return (
            <RowMenu
              label={rowLabel(record)}
              busy={isBusy}
              items={[
                {
                  id: "open",
                  label: "Open record",
                  icon: ExternalLink,
                  onSelect: () => onOpenRecord(record),
                },
                {
                  id: "copy",
                  label: "Copy transcript",
                  icon: Copy,
                  disabled: isBusy || deletePending || !outputReady,
                  onSelect: () => void handleCopy(record),
                },
                {
                  id: "export",
                  label: "Export",
                  icon: Download,
                  disabled: isBusy || deletePending || !outputReady,
                  onSelect: () => void handleExport(record),
                },
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
          );
        }}
      />

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

      <ConfirmDeleteDialog
        noun={deleteTarget ? deleteTargetKindLabel : null}
        itemLabel={deleteTargetLabel}
        description="This permanently removes the saved audio and transcript from history."
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
