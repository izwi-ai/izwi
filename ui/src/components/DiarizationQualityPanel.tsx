import { type FormEvent, useEffect, useMemo, useState } from "react";
import { AlertTriangle, Loader2, RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import type { DiarizationRecord, DiarizationRecordRerunRequest } from "../api";

interface DiarizationQualityPanelProps {
  record: Pick<
    DiarizationRecord,
    | "alignment_coverage"
    | "corrected_speaker_count"
    | "enable_llm_refinement"
    | "llm_refined"
    | "max_speakers"
    | "min_silence_duration_ms"
    | "min_speakers"
    | "min_speech_duration_ms"
    | "speaker_count"
    | "unattributed_words"
  > | null;
  isRerunning?: boolean;
  error?: string | null;
  onRerun: (request: DiarizationRecordRerunRequest) => Promise<void> | void;
}

function formatCoverage(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Unavailable";
  }
  return `${Math.round(value * 100)}%`;
}

function formatDraftValue(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "";
  }
  if (Number.isInteger(value)) {
    return String(value);
  }
  return value.toFixed(2).replace(/\.?0+$/, "");
}

function parseOptionalInteger(value: string): number | undefined {
  const trimmed = value.trim();
  if (!trimmed) {
    return undefined;
  }
  const parsed = Number.parseInt(trimmed, 10);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function parseOptionalNumber(value: string): number | undefined {
  const trimmed = value.trim();
  if (!trimmed) {
    return undefined;
  }
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : undefined;
}

type QualityWarning = {
  title: string;
  description: string;
};

export function DiarizationQualityPanel({
  record,
  isRerunning = false,
  error = null,
  onRerun,
}: DiarizationQualityPanelProps) {
  const [minSpeakers, setMinSpeakers] = useState("");
  const [maxSpeakers, setMaxSpeakers] = useState("");
  const [minSpeechDurationMs, setMinSpeechDurationMs] = useState("");
  const [minSilenceDurationMs, setMinSilenceDurationMs] = useState("");
  const [enableLlmRefinement, setEnableLlmRefinement] = useState(true);

  useEffect(() => {
    setMinSpeakers(formatDraftValue(record?.min_speakers));
    setMaxSpeakers(formatDraftValue(record?.max_speakers));
    setMinSpeechDurationMs(formatDraftValue(record?.min_speech_duration_ms));
    setMinSilenceDurationMs(formatDraftValue(record?.min_silence_duration_ms));
    setEnableLlmRefinement(record?.enable_llm_refinement ?? true);
  }, [record]);

  const parsedMinSpeakers = useMemo(
    () => parseOptionalInteger(minSpeakers),
    [minSpeakers],
  );
  const parsedMaxSpeakers = useMemo(
    () => parseOptionalInteger(maxSpeakers),
    [maxSpeakers],
  );
  const parsedMinSpeechDurationMs = useMemo(
    () => parseOptionalNumber(minSpeechDurationMs),
    [minSpeechDurationMs],
  );
  const parsedMinSilenceDurationMs = useMemo(
    () => parseOptionalNumber(minSilenceDurationMs),
    [minSilenceDurationMs],
  );

  const validationError = useMemo(() => {
    if (minSpeakers.trim() && parsedMinSpeakers === undefined) {
      return "Enter a whole number for minimum speakers.";
    }
    if (maxSpeakers.trim() && parsedMaxSpeakers === undefined) {
      return "Enter a whole number for maximum speakers.";
    }
    if (
      parsedMinSpeakers !== undefined &&
      (parsedMinSpeakers < 1 || !Number.isInteger(parsedMinSpeakers))
    ) {
      return "Minimum speakers must be at least 1.";
    }
    if (
      parsedMaxSpeakers !== undefined &&
      (parsedMaxSpeakers < 1 || !Number.isInteger(parsedMaxSpeakers))
    ) {
      return "Maximum speakers must be at least 1.";
    }
    if (
      parsedMinSpeakers !== undefined &&
      parsedMaxSpeakers !== undefined &&
      parsedMinSpeakers > parsedMaxSpeakers
    ) {
      return "Minimum speakers cannot exceed maximum speakers.";
    }
    if (
      minSpeechDurationMs.trim() &&
      parsedMinSpeechDurationMs === undefined
    ) {
      return "Enter a number for minimum speech duration.";
    }
    if (
      minSilenceDurationMs.trim() &&
      parsedMinSilenceDurationMs === undefined
    ) {
      return "Enter a number for minimum silence duration.";
    }
    if (
      parsedMinSpeechDurationMs !== undefined &&
      parsedMinSpeechDurationMs < 0
    ) {
      return "Minimum speech duration cannot be negative.";
    }
    if (
      parsedMinSilenceDurationMs !== undefined &&
      parsedMinSilenceDurationMs < 0
    ) {
      return "Minimum silence duration cannot be negative.";
    }
    return null;
  }, [
    maxSpeakers,
    minSilenceDurationMs,
    minSpeakers,
    minSpeechDurationMs,
    parsedMaxSpeakers,
    parsedMinSilenceDurationMs,
    parsedMinSpeakers,
    parsedMinSpeechDurationMs,
  ]);

  const warnings = useMemo<QualityWarning[]>(() => {
    if (!record) {
      return [];
    }

    const nextWarnings: QualityWarning[] = [];
    if (
      typeof record.alignment_coverage === "number" &&
      Number.isFinite(record.alignment_coverage) &&
      record.alignment_coverage < 0.9
    ) {
      nextWarnings.push({
        title: "Alignment coverage needs review",
        description: `${formatCoverage(record.alignment_coverage)} of aligned words were matched to speaker turns. Expect more speaker drift near overlaps or short interruptions.`,
      });
    }
    if (record.unattributed_words > 0) {
      nextWarnings.push({
        title: "Unattributed words detected",
        description: `${record.unattributed_words} word${record.unattributed_words === 1 ? "" : "s"} could not be assigned to a speaker. A rerun with tighter speaker bounds can help.`,
      });
    }
    if (!record.llm_refined && record.enable_llm_refinement) {
      nextWarnings.push({
        title: "LLM refinement did not apply",
        description: "The transcript was configured to use refinement, but the saved result remained unrefined. Review the raw speaker turns before exporting.",
      });
    }
    return nextWarnings;
  }, [record]);

  if (!record) {
    return (
      <Card className="border-[var(--border-muted)] bg-[var(--bg-surface-1)]">
        <CardContent className="py-12 text-center text-sm text-[var(--text-muted)]">
          No quality data is available yet.
        </CardContent>
      </Card>
    );
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (validationError || isRerunning) {
      return;
    }

    await onRerun({
      min_speakers: parsedMinSpeakers,
      max_speakers: parsedMaxSpeakers,
      min_speech_duration_ms: parsedMinSpeechDurationMs,
      min_silence_duration_ms: parsedMinSilenceDurationMs,
      enable_llm_refinement: enableLlmRefinement,
    });
  }

  function resetToCurrentSettings(): void {
    if (!record) {
      return;
    }
    setMinSpeakers(formatDraftValue(record.min_speakers));
    setMaxSpeakers(formatDraftValue(record.max_speakers));
    setMinSpeechDurationMs(formatDraftValue(record.min_speech_duration_ms));
    setMinSilenceDurationMs(formatDraftValue(record.min_silence_duration_ms));
    setEnableLlmRefinement(record.enable_llm_refinement);
  }

  return (
    <div className="space-y-4">
      <Card className="border-[var(--border-muted)] bg-[var(--bg-surface-1)]">
        <CardHeader className="pb-4">
          <CardTitle className="text-sm text-[var(--text-primary)]">
            Quality Signals
          </CardTitle>
          <CardDescription className="text-[var(--text-muted)]">
            Review diarization confidence indicators before exporting or saving speaker corrections.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-3">
              <div className="text-[11px] uppercase tracking-wider text-[var(--text-subtle)]">
                Alignment coverage
              </div>
              <div className="mt-1 text-lg font-semibold text-[var(--text-primary)]">
                {formatCoverage(record.alignment_coverage)}
              </div>
            </div>
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-3">
              <div className="text-[11px] uppercase tracking-wider text-[var(--text-subtle)]">
                Unattributed words
              </div>
              <div className="mt-1 text-lg font-semibold text-[var(--text-primary)]">
                {record.unattributed_words}
              </div>
            </div>
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-3">
              <div className="text-[11px] uppercase tracking-wider text-[var(--text-subtle)]">
                Speakers
              </div>
              <div className="mt-1 text-lg font-semibold text-[var(--text-primary)]">
                {record.speaker_count}
              </div>
              <div className="mt-1 text-xs text-[var(--text-muted)]">
                {record.corrected_speaker_count ?? record.speaker_count} corrected
              </div>
            </div>
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-3">
              <div className="text-[11px] uppercase tracking-wider text-[var(--text-subtle)]">
                Refinement
              </div>
              <div className="mt-1 text-lg font-semibold text-[var(--text-primary)]">
                {record.llm_refined ? "Applied" : "Off"}
              </div>
              <div className="mt-1 text-xs text-[var(--text-muted)]">
                {record.enable_llm_refinement ? "Enabled for this run" : "Disabled"}
              </div>
            </div>
          </div>

          <Separator className="bg-[var(--border-muted)]" />

          {warnings.length > 0 ? (
            <div className="grid gap-3">
              {warnings.map((warning) => (
                <div
                  key={warning.title}
                  className="rounded-xl border border-[var(--warning-border)] bg-[var(--warning-bg)] px-4 py-3 text-[var(--warning-text)]"
                >
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                    <div>
                      <div className="text-sm font-semibold">{warning.title}</div>
                      <p className="mt-1 text-sm leading-relaxed">
                        {warning.description}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-3 text-sm text-[var(--text-secondary)]">
              No major diarization warnings surfaced for this run.
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="border-[var(--border-muted)] bg-[var(--bg-surface-1)]">
        <CardHeader className="pb-4">
          <CardTitle className="text-sm text-[var(--text-primary)]">
            Rerun Saved Audio
          </CardTitle>
          <CardDescription className="text-[var(--text-muted)]">
            Adjust speaker bounds or segmentation thresholds, then rerun without re-uploading the file.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form className="space-y-4" onSubmit={(event) => void handleSubmit(event)}>
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="quality-min-speakers">Min speakers</Label>
                <Input
                  id="quality-min-speakers"
                  type="number"
                  min={1}
                  step={1}
                  value={minSpeakers}
                  onChange={(event) => setMinSpeakers(event.target.value)}
                  className="border-[var(--border-muted)] bg-[var(--bg-surface-0)]"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="quality-max-speakers">Max speakers</Label>
                <Input
                  id="quality-max-speakers"
                  type="number"
                  min={1}
                  step={1}
                  value={maxSpeakers}
                  onChange={(event) => setMaxSpeakers(event.target.value)}
                  className="border-[var(--border-muted)] bg-[var(--bg-surface-0)]"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="quality-min-speech">Min speech (ms)</Label>
                <Input
                  id="quality-min-speech"
                  type="number"
                  min={0}
                  step={10}
                  value={minSpeechDurationMs}
                  onChange={(event) => setMinSpeechDurationMs(event.target.value)}
                  className="border-[var(--border-muted)] bg-[var(--bg-surface-0)]"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="quality-min-silence">Min silence (ms)</Label>
                <Input
                  id="quality-min-silence"
                  type="number"
                  min={0}
                  step={10}
                  value={minSilenceDurationMs}
                  onChange={(event) => setMinSilenceDurationMs(event.target.value)}
                  className="border-[var(--border-muted)] bg-[var(--bg-surface-0)]"
                />
              </div>
            </div>

            <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 py-3">
              <div className="flex items-center justify-between gap-3">
                <div className="space-y-1">
                  <Label htmlFor="quality-llm-refinement">
                    LLM transcript refinement
                  </Label>
                  <p className="text-sm text-[var(--text-muted)]">
                    Keep it on for cleaner transcript wording, or turn it off to inspect the raw diarized output.
                  </p>
                </div>
                <Switch
                  id="quality-llm-refinement"
                  checked={enableLlmRefinement}
                  onCheckedChange={setEnableLlmRefinement}
                />
              </div>
            </div>

            {validationError ? (
              <div className="rounded-xl border border-[var(--danger-border)] bg-[var(--danger-bg)] px-4 py-3 text-sm text-[var(--danger-text)]">
                {validationError}
              </div>
            ) : null}

            {error ? (
              <div className="rounded-xl border border-[var(--danger-border)] bg-[var(--danger-bg)] px-4 py-3 text-sm text-[var(--danger-text)]">
                {error}
              </div>
            ) : null}

            <div className="flex flex-wrap items-center justify-between gap-3">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="h-9 gap-2 text-[var(--text-secondary)]"
                onClick={resetToCurrentSettings}
                disabled={isRerunning}
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Reset to current
              </Button>
              <Button
                type="submit"
                size="sm"
                className="h-9 gap-2"
                disabled={Boolean(validationError) || isRerunning}
              >
                {isRerunning ? (
                  <>
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    Rerunning...
                  </>
                ) : (
                  <>
                    <RotateCcw className="h-3.5 w-3.5" />
                    Rerun saved audio
                  </>
                )}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
