import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowLeft,
  Check,
  Copy,
  Download,
  Loader2,
  RotateCcw,
  Trash2,
} from "lucide-react";

import { type DiarizationRecord, type DiarizationRecordRerunRequest } from "@/api";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { DiarizationExportDialog } from "@/components/DiarizationExportDialog";
import { DiarizationQualityPanel } from "@/components/DiarizationQualityPanel";
import { DiarizationReviewWorkspace } from "@/components/DiarizationReviewWorkspace";
import { DiarizationSpeakerManager } from "@/components/DiarizationSpeakerManager";
import { normalizeDiarizationSummaryStatus } from "@/utils/diarizationSummary";
import { normalizeDiarizationProcessingStatus } from "@/utils/diarizationProcessing";
import { formattedTranscriptFromRecord } from "@/utils/diarizationTranscript";

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

interface DiarizationRecordDetailProps {
  record: DiarizationRecord | null;
  audioUrl: string | null;
  loading?: boolean;
  error?: string | null;
  summaryModelGuidance?: string | null;
  onBack?: () => void;
  onDelete?: (recordId: string) => Promise<void> | void;
  onSaveSpeakerCorrections?: (
    recordId: string,
    speakerNameOverrides: Record<string, string>,
  ) => Promise<void> | void;
  onRerun?: (
    recordId: string,
    request: DiarizationRecordRerunRequest,
  ) => Promise<void> | void;
  onCancelProcessing?: (recordId: string) => Promise<void> | void;
  onRegenerateSummary?: (recordId: string) => Promise<void> | void;
}

export function DiarizationRecordDetail({
  record,
  audioUrl,
  loading = false,
  error = null,
  summaryModelGuidance = null,
  onBack,
  onDelete,
  onSaveSpeakerCorrections,
  onRerun,
  onCancelProcessing,
  onRegenerateSummary,
}: DiarizationRecordDetailProps) {
  const [workspaceTab, setWorkspaceTab] = useState("transcript");
  const [copied, setCopied] = useState(false);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [deletePending, setDeletePending] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [speakerUpdatePending, setSpeakerUpdatePending] = useState(false);
  const [speakerUpdateError, setSpeakerUpdateError] = useState<string | null>(
    null,
  );
  const [rerunPending, setRerunPending] = useState(false);
  const [rerunError, setRerunError] = useState<string | null>(null);
  const [cancelPending, setCancelPending] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);
  const [summaryRefreshPending, setSummaryRefreshPending] = useState(false);
  const [summaryRefreshError, setSummaryRefreshError] = useState<string | null>(
    null,
  );

  const transcriptText = useMemo(
    () => (record ? formattedTranscriptFromRecord(record) : ""),
    [record],
  );
  const hasTranscript = transcriptText.trim().length > 0;
  const processingStatus = useMemo(
    () =>
      normalizeDiarizationProcessingStatus(
        record?.processing_status,
        record?.processing_error,
      ),
    [record?.processing_error, record?.processing_status],
  );
  const summaryStatus = useMemo(
    () =>
      normalizeDiarizationSummaryStatus(
        record?.summary_status,
        record?.summary_text,
        record?.summary_error,
      ),
    [record?.summary_error, record?.summary_status, record?.summary_text],
  );
  const processingStatusMessage = useMemo(() => {
    switch (processingStatus) {
      case "pending":
        return "This diarization run is queued and will begin processing shortly.";
      case "processing":
        return "This diarization run is being processed. The transcript will appear here automatically.";
      case "failed":
        return record?.processing_error || "Diarization processing failed.";
      case "ready":
      default:
        return null;
    }
  }, [processingStatus, record?.processing_error]);
  const summaryStatusMessage = useMemo(() => {
    switch (summaryStatus) {
      case "pending":
        return "Summary generation is still running in the background. The transcript is ready to review now.";
      case "failed":
        return record?.summary_error || "Summary generation failed for this diarization record.";
      case "ready":
      case "not_requested":
      default:
        return null;
    }
  }, [record?.summary_error, summaryStatus]);
  const canCancelProcessing =
    processingStatus === "pending" || processingStatus === "processing";

  useEffect(() => {
    setWorkspaceTab("transcript");
    setCopied(false);
    setDeleteConfirmOpen(false);
    setDeletePending(false);
    setDeleteError(null);
    setSpeakerUpdatePending(false);
    setSpeakerUpdateError(null);
    setRerunPending(false);
    setRerunError(null);
    setCancelPending(false);
    setCancelError(null);
    setSummaryRefreshPending(false);
    setSummaryRefreshError(null);
  }, [record?.id]);

  async function handleCopy(): Promise<void> {
    if (!transcriptText) {
      return;
    }
    await navigator.clipboard.writeText(transcriptText);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  }

  async function handleDelete(): Promise<void> {
    if (!record || !onDelete || deletePending) {
      return;
    }

    setDeletePending(true);
    setDeleteError(null);
    try {
      await onDelete(record.id);
    } catch (err) {
      setDeleteError(
        err instanceof Error ? err.message : "Failed to delete diarization record.",
      );
    } finally {
      setDeletePending(false);
    }
  }

  async function handleSaveSpeakerCorrections(
    speakerNameOverrides: Record<string, string>,
  ): Promise<void> {
    if (!record || !onSaveSpeakerCorrections || speakerUpdatePending) {
      return;
    }

    setSpeakerUpdatePending(true);
    setSpeakerUpdateError(null);
    try {
      await onSaveSpeakerCorrections(record.id, speakerNameOverrides);
    } catch (err) {
      setSpeakerUpdateError(
        err instanceof Error ? err.message : "Failed to save speaker corrections.",
      );
    } finally {
      setSpeakerUpdatePending(false);
    }
  }

  async function handleRerun(
    request: DiarizationRecordRerunRequest,
  ): Promise<void> {
    if (!record || !onRerun || rerunPending) {
      return;
    }

    setRerunPending(true);
    setRerunError(null);
    setSpeakerUpdateError(null);
    setSummaryRefreshError(null);
    try {
      await onRerun(record.id, request);
    } catch (err) {
      setRerunError(
        err instanceof Error ? err.message : "Failed to rerun diarization.",
      );
    } finally {
      setRerunPending(false);
    }
  }

  async function handleCancelProcessing(): Promise<void> {
    if (
      !record ||
      !onCancelProcessing ||
      cancelPending ||
      !canCancelProcessing
    ) {
      return;
    }

    setCancelPending(true);
    setCancelError(null);
    try {
      await onCancelProcessing(record.id);
    } catch (err) {
      setCancelError(
        err instanceof Error
          ? err.message
          : "Failed to cancel diarization processing.",
      );
    } finally {
      setCancelPending(false);
    }
  }

  async function handleRegenerateSummary(): Promise<void> {
    if (!record || !onRegenerateSummary || summaryRefreshPending) {
      return;
    }

    setSummaryRefreshPending(true);
    setSummaryRefreshError(null);
    try {
      await onRegenerateSummary(record.id);
    } catch (err) {
      setSummaryRefreshError(
        err instanceof Error
          ? err.message
          : "Failed to regenerate diarization summary.",
      );
    } finally {
      setSummaryRefreshPending(false);
    }
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
              Back to diarization
            </Button>
          ) : null}
          <h2 className="truncate text-2xl font-semibold tracking-tight text-[var(--text-primary)]">
            {record?.audio_filename || record?.model_id || "Diarization record"}
          </h2>
          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-[var(--text-muted)]">
            {record ? <span>{formatCreatedAt(record.created_at)}</span> : null}
            {record?.duration_secs != null ? (
              <span>{formatAudioDuration(record.duration_secs)}</span>
            ) : null}
            {record ? (
              <span>
                {record.corrected_speaker_count ?? record.speaker_count} speakers
              </span>
            ) : null}
            {record?.model_id ? <span>{record.model_id}</span> : null}
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-end gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-9 gap-2"
            onClick={() => void handleRegenerateSummary()}
            disabled={
              !record || summaryRefreshPending || !hasTranscript || cancelPending
            }
          >
            {summaryRefreshPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <RotateCcw className="h-4 w-4" />
            )}
            Regenerate summary
          </Button>
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
          <DiarizationExportDialog record={record}>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 gap-2"
              disabled={!hasTranscript}
            >
              <Download className="h-4 w-4" />
              Export
            </Button>
          </DiarizationExportDialog>
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
          {onCancelProcessing && canCancelProcessing ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 gap-2 border-[var(--status-warning-border)] text-[var(--status-warning-text)] hover:bg-[var(--status-warning-bg)]"
              onClick={() => void handleCancelProcessing()}
              disabled={cancelPending}
            >
              {cancelPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : null}
              {cancelPending ? "Cancelling" : "Cancel run"}
            </Button>
          ) : null}
        </div>
      </div>

      {error ? (
        <Card className="border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-text)]">
          {error}
        </Card>
      ) : null}

      {summaryRefreshError ? (
        <Card className="border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-text)]">
          {summaryRefreshError}
        </Card>
      ) : null}

      {cancelError ? (
        <Card className="border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-text)]">
          {cancelError}
        </Card>
      ) : null}

      {processingStatusMessage ? (
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
            <p>{processingStatusMessage}</p>
          </div>
        </Card>
      ) : null}

      {summaryStatusMessage ? (
        <Card
          className={
            summaryStatus === "failed"
              ? "border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-text)]"
              : "border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] p-4 text-sm text-[var(--status-warning-text)]"
          }
        >
          <div className="flex items-start gap-3">
            {summaryStatus === "failed" ? (
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            ) : (
              <Loader2 className="mt-0.5 h-4 w-4 shrink-0 animate-spin" />
            )}
            <p>{summaryStatusMessage}</p>
          </div>
        </Card>
      ) : null}

      {summaryModelGuidance ? (
        <Card className="border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] p-4 text-sm text-[var(--status-warning-text)]">
          {summaryModelGuidance}
        </Card>
      ) : null}

      <Tabs
        value={workspaceTab}
        onValueChange={setWorkspaceTab}
        className="space-y-4"
      >
        <TabsList className="inline-flex w-auto justify-start rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-0.5">
          <TabsTrigger
            value="transcript"
            className="h-9 rounded-full px-4 text-sm font-medium"
          >
            Transcript
          </TabsTrigger>
          <TabsTrigger
            value="speakers"
            className="h-9 rounded-full px-4 text-sm font-medium"
          >
            Speakers
          </TabsTrigger>
          <TabsTrigger
            value="quality"
            className="h-9 rounded-full px-4 text-sm font-medium"
          >
            Quality
          </TabsTrigger>
        </TabsList>

        <TabsContent value="transcript" className="mt-0">
          <DiarizationReviewWorkspace
            record={record}
            audioUrl={audioUrl}
            loading={loading}
            autoScrollActiveEntry={true}
            fixedPlaybackFooter={true}
            summaryModelGuidance={summaryModelGuidance}
            emptyTitle="Diarization record loading"
            emptyMessage="The transcript will appear here once the diarization record is ready."
          />
        </TabsContent>

        <TabsContent value="speakers" className="mt-0">
          {record ? (
            <DiarizationSpeakerManager
              record={record}
              isSaving={speakerUpdatePending}
              error={speakerUpdateError}
              onSave={(speakerNameOverrides) =>
                void handleSaveSpeakerCorrections(speakerNameOverrides)
              }
            />
          ) : null}
        </TabsContent>

        <TabsContent value="quality" className="mt-0">
          {record ? (
            <DiarizationQualityPanel
              record={record}
              isRerunning={rerunPending}
              error={rerunError}
              onRerun={(request) => void handleRerun(request)}
            />
          ) : null}
        </TabsContent>
      </Tabs>

      <Dialog
        open={deleteConfirmOpen}
        onOpenChange={(open) => {
          if (!deletePending) {
            setDeleteConfirmOpen(open);
          }
        }}
      >
        <DialogContent className="max-w-md border-[var(--border-strong)] bg-[var(--bg-surface-1)] p-5">
          <DialogTitle className="sr-only">Delete diarization record?</DialogTitle>
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
                {record?.audio_filename || record?.model_id || record?.id}
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
              onClick={() => void handleDelete()}
              disabled={deletePending}
            >
              {deletePending ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Deleting
                </>
              ) : (
                "Delete record"
              )}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
