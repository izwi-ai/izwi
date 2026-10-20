import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  CalendarDays,
  Check,
  Clock,
  Copy,
  Cpu,
  Download,
  Loader2,
  OctagonX,
  RotateCcw,
  Trash2,
  Users,
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
import { DiarizationSpeakersTable } from "@/features/diarization/components/DiarizationSpeakersTable";
import {
  formatClockDuration,
  formatHistoryDate,
} from "@/components/history/format";
import type { RowMenuItem } from "@/components/history/RowMenu";
import type { Breadcrumb } from "@/components/PageShell";
import { RecordPage, type RecordMetaItem } from "@/components/RecordPage";
import { getModelDisplayName } from "@/features/models/catalog/modelMetadata";
import { normalizeDiarizationSummaryStatus } from "@/utils/diarizationSummary";
import { normalizeDiarizationProcessingStatus } from "@/utils/diarizationProcessing";
import { formattedTranscriptFromRecord } from "@/utils/diarizationTranscript";

interface DiarizationRecordDetailProps {
  record: DiarizationRecord | null;
  audioUrl: string | null;
  loading?: boolean;
  error?: string | null;
  summaryModelGuidance?: string | null;
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
  /** The list this record belongs to, for the breadcrumb trail. */
  parentCrumb: Breadcrumb;
  /** Route actions for the header menu (Models). */
  extraMenuItems?: RowMenuItem[];
}

export function DiarizationRecordDetail({
  record,
  audioUrl,
  loading = false,
  error = null,
  summaryModelGuidance = null,
  onDelete,
  onSaveSpeakerCorrections,
  onRerun,
  onCancelProcessing,
  onRegenerateSummary,
  parentCrumb,
  extraMenuItems = [],
}: DiarizationRecordDetailProps) {
  const [workspaceTab, setWorkspaceTab] = useState("transcript");
  const [copied, setCopied] = useState(false);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [deletePending, setDeletePending] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
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
    if (!record || !onSaveSpeakerCorrections) {
      return;
    }

    setSpeakerUpdateError(null);
    try {
      await onSaveSpeakerCorrections(record.id, speakerNameOverrides);
    } catch (err) {
      setSpeakerUpdateError(
        err instanceof Error ? err.message : "Failed to save speaker names.",
      );
      throw err;
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

  const title =
    record?.audio_filename || record?.id || "Diarization record";
  const isReady = processingStatus === "ready";
  // The hint to load the summary model only helps a finished run whose
  // summary failed: an existing summary needs nothing, and an unrequested
  // one is asked for from the menu, which explains what to load.
  const visibleSummaryGuidance =
    isReady && record?.summary_status === "failed" ? summaryModelGuidance : null;
  const emptyState =
    processingStatus === "failed"
      ? {
          title: "No transcript",
          message:
            "This run failed before producing a transcript. Re-run it from the Quality tab, or delete it.",
        }
      : processingStatus === "pending" || processingStatus === "processing"
        ? {
            title: "Diarization in progress",
            message: "The transcript will appear here when the run finishes.",
          }
        : {
            title: "No transcript",
            message: "This run did not find any speech.",
          };
  const speakerCount = record
    ? (record.corrected_speaker_count ?? record.speaker_count)
    : null;
  const meta: RecordMetaItem[] = record
    ? [
        { icon: CalendarDays, label: formatHistoryDate(record.created_at) },
        ...(record.duration_secs != null
          ? [{ icon: Clock, label: formatClockDuration(record.duration_secs) }]
          : []),
        ...(speakerCount != null && (isReady || speakerCount > 0)
          ? [
              {
                icon: Users,
                label: `${speakerCount} speaker${speakerCount === 1 ? "" : "s"}`,
              },
            ]
          : []),
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
    {
      id: "summary",
      label: summaryRefreshPending
        ? "Regenerating summary…"
        : "Regenerate summary",
      icon: summaryRefreshPending ? Loader2 : RotateCcw,
      disabled:
        !record || summaryRefreshPending || !hasTranscript || cancelPending,
      onSelect: () => void handleRegenerateSummary(),
    },
    ...(onCancelProcessing && canCancelProcessing
      ? [
          {
            id: "cancel",
            label: cancelPending ? "Cancelling…" : "Cancel run",
            icon: OctagonX,
            disabled: cancelPending,
            onSelect: () => void handleCancelProcessing(),
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
          <DiarizationExportDialog record={record}>
            <Button
              type="button"
              size="sm"
              className="h-9 gap-2"
              disabled={!hasTranscript}
            >
              <Download className="h-4 w-4" />
              Export
            </Button>
          </DiarizationExportDialog>
        </>
      }
      alerts={
        <>
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

          {visibleSummaryGuidance ? (
            <Card className="border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] p-4 text-sm text-[var(--status-warning-text)]">
              {visibleSummaryGuidance}
            </Card>
          ) : null}
        </>
      }
    >
      <Tabs
        value={workspaceTab}
        onValueChange={setWorkspaceTab}
        className="space-y-4"
      >
        <TabsList className="inline-flex h-auto w-auto justify-start gap-0.5 rounded-[10px] border border-[var(--border-subtle)] bg-transparent p-[3px]">
          {[
            ["transcript", "Transcript"],
            ["speakers", "Speakers"],
            ["quality", "Quality"],
          ].map(([value, label]) => (
            <TabsTrigger
              key={value}
              value={value}
              className="h-[26px] rounded-[7px] px-3 text-[12.5px] font-medium text-[var(--text-muted)] shadow-none data-[state=active]:bg-[var(--surface-raised)] data-[state=active]:text-[var(--text-primary)] data-[state=active]:shadow-none"
            >
              {label}
            </TabsTrigger>
          ))}
        </TabsList>

        <TabsContent value="transcript" className="mt-0">
          <DiarizationReviewWorkspace
            record={record}
            audioUrl={audioUrl}
            loading={loading}
            autoScrollActiveEntry={true}
            fixedPlaybackFooter={true}
            summaryModelGuidance={visibleSummaryGuidance}
            emptyTitle={emptyState.title}
            emptyMessage={emptyState.message}
          />
        </TabsContent>

        <TabsContent value="speakers" className="mt-0">
          {record ? (
            <DiarizationSpeakersTable
              record={record}
              audioUrl={audioUrl}
              error={speakerUpdateError}
              onSave={handleSaveSpeakerCorrections}
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
                {record?.audio_filename || record?.id}
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
    </RecordPage>
  );
}
