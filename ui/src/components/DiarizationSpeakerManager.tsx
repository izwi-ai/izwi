import { useEffect, useMemo, useState } from "react";
import { ArrowRightLeft, Check, Loader2, RotateCcw, Save } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import type { DiarizationRecord } from "../api";
import {
  correctedSpeakerCount,
  rawSpeakerSummariesFromRecord,
} from "../utils/diarizationTranscript";

interface DiarizationSpeakerManagerProps {
  record: Pick<
    DiarizationRecord,
    | "speaker_count"
    | "corrected_speaker_count"
    | "segments"
    | "utterances"
    | "words"
    | "speaker_name_overrides"
  >;
  isSaving?: boolean;
  error?: string | null;
  onSave: (speakerNameOverrides: Record<string, string>) => Promise<void> | void;
}

function normalizeSpeakerName(value: string): string {
  return value.trim().replace(/\s+/g, " ");
}

function buildSpeakerNameOverrides(
  drafts: Record<string, string>,
): Record<string, string> {
  return Object.fromEntries(
    Object.entries(drafts)
      .map(([rawSpeaker, displaySpeaker]) => [
        rawSpeaker,
        normalizeSpeakerName(displaySpeaker),
      ])
      .filter(
        ([rawSpeaker, displaySpeaker]) =>
          displaySpeaker.length > 0 && displaySpeaker !== rawSpeaker,
      ),
  );
}

function equalOverrides(
  left: Record<string, string>,
  right: Record<string, string>,
): boolean {
  const leftEntries = Object.entries(left).sort(([a], [b]) => a.localeCompare(b));
  const rightEntries = Object.entries(right).sort(([a], [b]) => a.localeCompare(b));
  return JSON.stringify(leftEntries) === JSON.stringify(rightEntries);
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "0s";
  }
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  return `${minutes}m ${remainingSeconds}s`;
}

export function DiarizationSpeakerManager({
  record,
  isSaving = false,
  error = null,
  onSave,
}: DiarizationSpeakerManagerProps) {
  const rawSummaries = useMemo(
    () => rawSpeakerSummariesFromRecord(record),
    [record],
  );
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [mergeSourceSpeaker, setMergeSourceSpeaker] = useState<string | null>(null);
  const [mergeTargetSpeaker, setMergeTargetSpeaker] = useState<string>("");

  useEffect(() => {
    const nextDrafts = Object.fromEntries(
      rawSummaries.map((summary) => [summary.rawSpeaker, summary.displaySpeaker]),
    );
    setDrafts(nextDrafts);
  }, [rawSummaries]);

  const currentOverrides = useMemo(
    () => buildSpeakerNameOverrides(drafts),
    [drafts],
  );
  const persistedOverrides = useMemo(
    () => buildSpeakerNameOverrides(record.speaker_name_overrides ?? {}),
    [record.speaker_name_overrides],
  );
  const hasDraftChanges = !equalOverrides(currentOverrides, persistedOverrides);
  const correctedCount = correctedSpeakerCount(record);

  const duplicateCounts = useMemo(() => {
    const counts = new Map<string, number>();
    Object.values(drafts)
      .map(normalizeSpeakerName)
      .filter(Boolean)
      .forEach((speakerName) => {
        counts.set(speakerName, (counts.get(speakerName) ?? 0) + 1);
      });
    return counts;
  }, [drafts]);

  const mergeOptions = useMemo(() => {
    if (!mergeSourceSpeaker) {
      return [];
    }
    return rawSummaries
      .filter((summary) => summary.rawSpeaker !== mergeSourceSpeaker)
      .map((summary) => ({
        rawSpeaker: summary.rawSpeaker,
        displaySpeaker:
          normalizeSpeakerName(drafts[summary.rawSpeaker] ?? summary.displaySpeaker) ||
          summary.rawSpeaker,
      }));
  }, [drafts, mergeSourceSpeaker, rawSummaries]);

  return (
    <div className="space-y-4">
      <Card className="border-[var(--border-muted)] bg-[var(--bg-surface-1)]">
        <CardHeader className="pb-4">
          <CardTitle className="text-sm text-[var(--text-primary)]">
            Speaker Corrections
          </CardTitle>
          <CardDescription className="text-[var(--text-muted)]">
            Rename raw labels or merge multiple detected speakers into one display name.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-3">
              <div className="text-[11px] uppercase tracking-wider text-[var(--text-subtle)]">
                Detected
              </div>
              <div className="mt-1 text-lg font-semibold text-[var(--text-primary)]">
                {record.speaker_count}
              </div>
            </div>
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-3">
              <div className="text-[11px] uppercase tracking-wider text-[var(--text-subtle)]">
                Corrected
              </div>
              <div className="mt-1 text-lg font-semibold text-[var(--text-primary)]">
                {correctedCount}
              </div>
            </div>
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-3">
              <div className="text-[11px] uppercase tracking-wider text-[var(--text-subtle)]">
                Overrides
              </div>
              <div className="mt-1 text-lg font-semibold text-[var(--text-primary)]">
                {Object.keys(currentOverrides).length}
              </div>
            </div>
          </div>

          <Separator className="bg-[var(--border-muted)]" />

          <div className="space-y-3">
            {rawSummaries.length > 0 ? (
              rawSummaries.map((summary) => {
                const currentDisplaySpeaker =
                  normalizeSpeakerName(drafts[summary.rawSpeaker] ?? summary.displaySpeaker) ||
                  summary.rawSpeaker;
                const isMerged = (duplicateCounts.get(currentDisplaySpeaker) ?? 0) > 1;

                return (
                  <div
                    key={summary.rawSpeaker}
                    className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-4"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-3">
                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-2 py-1 text-[11px] font-semibold text-[var(--text-secondary)]">
                            {summary.rawSpeaker}
                          </span>
                          {isMerged ? (
                            <span className="rounded-md border border-[var(--accent-solid)]/35 bg-[var(--accent-soft)] px-2 py-1 text-[11px] font-medium text-[var(--text-primary)]">
                              merged
                            </span>
                          ) : null}
                        </div>
                        <div className="mt-2 text-xs text-[var(--text-muted)]">
                          {summary.utteranceCount} utterances · {summary.wordCount} words ·{" "}
                          {formatDuration(summary.totalDuration)}
                        </div>
                      </div>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="h-8 gap-1.5 border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)]"
                        onClick={() => {
                          setMergeSourceSpeaker(summary.rawSpeaker);
                          setMergeTargetSpeaker("");
                        }}
                      >
                        <ArrowRightLeft className="h-3.5 w-3.5" />
                        Merge
                      </Button>
                    </div>

                    <div className="mt-4 space-y-2">
                      <Label
                        htmlFor={`speaker-name-${summary.rawSpeaker}`}
                        className="text-xs uppercase tracking-wider text-[var(--text-subtle)]"
                      >
                        Display name
                      </Label>
                      <div className="flex flex-col gap-2 sm:flex-row">
                        <Input
                          id={`speaker-name-${summary.rawSpeaker}`}
                          value={drafts[summary.rawSpeaker] ?? summary.displaySpeaker}
                          onChange={(event) =>
                            setDrafts((current) => ({
                              ...current,
                              [summary.rawSpeaker]: event.target.value,
                            }))
                          }
                          className="border-[var(--border-muted)] bg-[var(--bg-surface-1)]"
                          placeholder={summary.rawSpeaker}
                        />
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="h-10 shrink-0 border border-transparent text-[var(--text-secondary)] hover:border-[var(--border-muted)] hover:bg-[var(--bg-surface-2)]"
                          onClick={() =>
                            setDrafts((current) => ({
                              ...current,
                              [summary.rawSpeaker]: summary.rawSpeaker,
                            }))
                          }
                        >
                          <RotateCcw className="mr-1.5 h-3.5 w-3.5" />
                          Reset
                        </Button>
                      </div>
                    </div>
                  </div>
                );
              })
            ) : (
              <div className="rounded-xl border border-dashed border-[var(--border-muted)] px-4 py-6 text-sm text-[var(--text-muted)]">
                No detected speakers available for correction.
              </div>
            )}
          </div>

          {error ? (
            <div className="rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-sm text-[var(--danger-text)]">
              {error}
            </div>
          ) : null}

          <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 border-[var(--border-muted)] bg-[var(--bg-surface-0)] text-[var(--text-secondary)]"
              onClick={() =>
                setDrafts(
                  Object.fromEntries(
                    rawSummaries.map((summary) => [
                      summary.rawSpeaker,
                      summary.displaySpeaker,
                    ]),
                  ),
                )
              }
              disabled={isSaving || !hasDraftChanges}
            >
              Reset changes
            </Button>
            <Button
              type="button"
              size="sm"
              className="h-9 gap-1.5"
              onClick={() => void onSave(currentOverrides)}
              disabled={isSaving || !hasDraftChanges}
            >
              {isSaving ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Save className="h-3.5 w-3.5" />
              )}
              Save corrections
            </Button>
          </div>
        </CardContent>
      </Card>

      <Dialog
        open={mergeSourceSpeaker !== null}
        onOpenChange={(open) => {
          if (!open && !isSaving) {
            setMergeSourceSpeaker(null);
            setMergeTargetSpeaker("");
          }
        }}
      >
        {mergeSourceSpeaker ? (
          <DialogContent className="max-w-md border-[var(--border-strong)] bg-[var(--bg-surface-0)]">
            <DialogHeader>
              <DialogTitle className="text-base text-[var(--text-primary)]">
                Merge speaker label
              </DialogTitle>
              <DialogDescription className="text-[var(--text-muted)]">
                Assign <strong>{mergeSourceSpeaker}</strong> to another speaker name without
                changing the original diarization output.
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-2">
              <Label className="text-xs uppercase tracking-wider text-[var(--text-subtle)]">
                Merge into
              </Label>
              <Select value={mergeTargetSpeaker} onValueChange={setMergeTargetSpeaker}>
                <SelectTrigger className="border-[var(--border-muted)] bg-[var(--bg-surface-1)]">
                  <SelectValue placeholder="Select a target speaker" />
                </SelectTrigger>
                <SelectContent>
                  {mergeOptions.map((option) => (
                    <SelectItem key={option.rawSpeaker} value={option.rawSpeaker}>
                      {option.displaySpeaker}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <DialogFooter className="gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-9 border-[var(--border-muted)] bg-[var(--bg-surface-1)]"
                onClick={() => {
                  setMergeSourceSpeaker(null);
                  setMergeTargetSpeaker("");
                }}
                disabled={isSaving}
              >
                Cancel
              </Button>
              <Button
                type="button"
                size="sm"
                className="h-9 gap-1.5"
                onClick={() => {
                  if (!mergeSourceSpeaker || !mergeTargetSpeaker) {
                    return;
                  }
                  const targetDisplaySpeaker =
                    mergeOptions.find(
                      (option) => option.rawSpeaker === mergeTargetSpeaker,
                    )?.displaySpeaker ?? mergeTargetSpeaker;
                  setDrafts((current) => ({
                    ...current,
                    [mergeSourceSpeaker]: targetDisplaySpeaker,
                  }));
                  setMergeSourceSpeaker(null);
                  setMergeTargetSpeaker("");
                }}
                disabled={!mergeTargetSpeaker || isSaving}
              >
                <Check className="h-3.5 w-3.5" />
                Apply merge
              </Button>
            </DialogFooter>
          </DialogContent>
        ) : null}
      </Dialog>
    </div>
  );
}
