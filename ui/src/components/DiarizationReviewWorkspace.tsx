import { useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Loader2, Users } from "lucide-react";

import {
  AudioPlayer,
  formatClockTime,
} from "@/components/audio/AudioPlayer";
import { useAudioPlayback } from "@/components/audio/useAudioPlayback";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { StatusBadge } from "@/components/ui/status-badge";
import type { DiarizationRecord } from "../api";
import {
  speakerSummariesFromRecord,
  transcriptEntriesFromRecord,
} from "../utils/diarizationTranscript";
import {
  diarizationSummaryStatusLabel,
  diarizationSummaryStatusTone,
  normalizeDiarizationSummaryStatus,
} from "../utils/diarizationSummary";

interface DiarizationReviewWorkspaceProps {
  record: Pick<
    DiarizationRecord,
    | "id"
    | "duration_secs"
    | "speaker_count"
    | "corrected_speaker_count"
    | "audio_filename"
    | "segments"
    | "utterances"
    | "words"
    | "speaker_name_overrides"
    | "transcript"
    | "raw_transcript"
    | "summary_status"
    | "summary_model_id"
    | "summary_text"
    | "summary_error"
    | "summary_updated_at"
  > | null;
  audioUrl?: string | null;
  loading?: boolean;
  emptyTitle?: string;
  emptyMessage?: string;
  autoScrollActiveEntry?: boolean;
  showPlayback?: boolean;
  stickyPlaybackFooter?: boolean;
  fixedPlaybackFooter?: boolean;
  summaryModelGuidance?: string | null;
}

type SpeakerAccent = {
  solid: string;
  soft: string;
  border: string;
};

type ConfidenceFlag = {
  entryIndex: number;
  start: number;
  speaker: string;
  averageConfidence: number | null;
  reason: string;
};

const LOW_CONFIDENCE_THRESHOLD = 0.72;

function formatDurationLabel(seconds: number): string {
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

function speakerAccent(index: number): SpeakerAccent {
  const hue = (index * 67 + 154) % 360;
  return {
    solid: `hsl(${hue} 72% 52%)`,
    soft: `hsla(${hue} 72% 52% / 0.16)`,
    border: `hsla(${hue} 72% 52% / 0.4)`,
  };
}

function isEntryActive(
  currentTime: number,
  start: number,
  end: number,
  epsilon = 0.08,
): boolean {
  return (
    currentTime >= Math.max(0, start - epsilon) && currentTime < end + epsilon
  );
}

function resolveEntryWords(
  record: NonNullable<DiarizationReviewWorkspaceProps["record"]>,
  entryIndex: number,
  start: number,
  end: number,
) {
  const utterance = record.utterances[entryIndex];
  if (
    utterance &&
    Number.isInteger(utterance.word_start) &&
    Number.isInteger(utterance.word_end)
  ) {
    const min = Math.max(0, utterance.word_start);
    const max = Math.min(record.words.length - 1, utterance.word_end);
    if (max >= min) {
      return record.words.slice(min, max + 1);
    }
  }

  return record.words.filter(
    (word) =>
      Number.isFinite(word.start) &&
      Number.isFinite(word.end) &&
      word.end > start &&
      word.start < end,
  );
}

function computeConfidenceFlags(
  record: NonNullable<DiarizationReviewWorkspaceProps["record"]>,
  entries: ReturnType<typeof transcriptEntriesFromRecord>,
): ConfidenceFlag[] {
  return entries.flatMap((entry, entryIndex) => {
    const words = resolveEntryWords(record, entryIndex, entry.start, entry.end);
    if (words.length === 0) {
      return [];
    }

    const confidenceValues = words
      .map((word) =>
        typeof word.speaker_confidence === "number" &&
        Number.isFinite(word.speaker_confidence)
          ? word.speaker_confidence
          : null,
      )
      .filter((value): value is number => value !== null);

    const averageConfidence =
      confidenceValues.length > 0
        ? confidenceValues.reduce((sum, value) => sum + value, 0) /
          confidenceValues.length
        : null;
    const overlapMismatches = words.filter(
      (word) => word.overlaps_segment === false,
    ).length;
    const lowConfidence =
      averageConfidence !== null && averageConfidence < LOW_CONFIDENCE_THRESHOLD;

    if (!lowConfidence && overlapMismatches === 0) {
      return [];
    }

    const reason =
      overlapMismatches > 0
        ? `${overlapMismatches} word${overlapMismatches === 1 ? "" : "s"} drifted from segment boundaries.`
        : `Average speaker confidence ${Math.round((averageConfidence ?? 0) * 100)}%.`;

    return [
      {
        entryIndex,
        start: entry.start,
        speaker: entry.speaker,
        averageConfidence,
        reason,
      },
    ];
  });
}

export function DiarizationReviewWorkspace({
  record,
  audioUrl = null,
  loading = false,
  emptyTitle = "Ready to diarize",
  emptyMessage = "No diarization transcript is available yet.",
  autoScrollActiveEntry = false,
  showPlayback = true,
  stickyPlaybackFooter = false,
  fixedPlaybackFooter = false,
  summaryModelGuidance = null,
}: DiarizationReviewWorkspaceProps) {
  const transcriptEntryRefs = useRef(new Map<number, HTMLButtonElement>());
  const lastAutoScrolledEntryRef = useRef<number | null>(null);
  const [focusedFlagIndex, setFocusedFlagIndex] = useState<number | null>(null);

  const transcriptEntries = useMemo(
    () => (record ? transcriptEntriesFromRecord(record) : []),
    [record],
  );
  const speakerSummaries = useMemo(
    () => (record ? speakerSummariesFromRecord(record) : []),
    [record],
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
  const summaryText = useMemo(
    () => record?.summary_text?.trim() || null,
    [record?.summary_text],
  );
  const summaryError = useMemo(
    () => record?.summary_error?.trim() || null,
    [record?.summary_error],
  );
  const normalizedSummaryGuidance = useMemo(
    () => summaryModelGuidance?.trim() || null,
    [summaryModelGuidance],
  );
  const summaryUpdatedLabel = record?.summary_updated_at
    ? new Date(record.summary_updated_at).toLocaleString([], {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      })
    : null;

  const knownDuration = useMemo(() => {
    const transcriptDuration = transcriptEntries.reduce(
      (max, entry) => Math.max(max, entry.end),
      0,
    );
    const recordDuration =
      record?.duration_secs && record.duration_secs > 0
        ? record.duration_secs
        : 0;
    return Math.max(recordDuration, transcriptDuration);
  }, [record, transcriptEntries]);
  const playback = useAudioPlayback(audioUrl, {
    resetKey: record?.id,
    durationHint: knownDuration,
    loadErrorMessage: "Unable to load audio for this diarization record.",
    playErrorMessage: "Unable to start playback for this diarization audio.",
  });
  const { currentTime, isPlaying, seek } = playback;

  const totalTalkTime = useMemo(
    () =>
      speakerSummaries.reduce(
        (sum, summary) => sum + Math.max(summary.totalDuration, 0),
        0,
      ),
    [speakerSummaries],
  );
  const activeEntryIndex = transcriptEntries.findIndex((entry) =>
    isEntryActive(currentTime, entry.start, entry.end),
  );
  const activeSpeaker =
    activeEntryIndex >= 0
      ? (transcriptEntries[activeEntryIndex]?.speaker ?? null)
      : null;
  const confidenceFlags = useMemo(
    () => (record ? computeConfidenceFlags(record, transcriptEntries) : []),
    [record, transcriptEntries],
  );
  const confidenceFlagsByEntry = useMemo(
    () =>
      new Map(confidenceFlags.map((flag) => [flag.entryIndex, flag] as const)),
    [confidenceFlags],
  );
  const activeFlagIndex = useMemo(
    () =>
      confidenceFlags.findIndex((flag) => flag.entryIndex === activeEntryIndex),
    [activeEntryIndex, confidenceFlags],
  );
  const selectedFlagIndex = useMemo(() => {
    if (confidenceFlags.length === 0) {
      return null;
    }
    if (
      focusedFlagIndex !== null &&
      focusedFlagIndex >= 0 &&
      focusedFlagIndex < confidenceFlags.length
    ) {
      return focusedFlagIndex;
    }
    if (activeFlagIndex >= 0) {
      return activeFlagIndex;
    }
    return 0;
  }, [activeFlagIndex, confidenceFlags, focusedFlagIndex]);

  useEffect(() => {
    setFocusedFlagIndex(null);
    lastAutoScrolledEntryRef.current = null;
  }, [audioUrl, record?.id]);

  useEffect(() => {
    if (!isPlaying) {
      lastAutoScrolledEntryRef.current = null;
    }
  }, [isPlaying]);

  useEffect(() => {
    if (!autoScrollActiveEntry || !isPlaying || activeEntryIndex < 0) {
      return;
    }
    if (lastAutoScrolledEntryRef.current === activeEntryIndex) {
      return;
    }

    const activeEntry = transcriptEntryRefs.current.get(activeEntryIndex);
    activeEntry?.scrollIntoView({
      block: "center",
      behavior: "smooth",
      inline: "nearest",
    });
    lastAutoScrolledEntryRef.current = activeEntryIndex;
  }, [activeEntryIndex, autoScrollActiveEntry, isPlaying]);

  function jumpToConfidenceFlag(flagIndex: number): void {
    if (confidenceFlags.length === 0) {
      return;
    }

    const normalizedIndex =
      ((flagIndex % confidenceFlags.length) + confidenceFlags.length) %
      confidenceFlags.length;
    const target = confidenceFlags[normalizedIndex];
    setFocusedFlagIndex(normalizedIndex);
    seek(target.start);
  }

  function stepConfidenceFlag(step: number): void {
    if (confidenceFlags.length === 0) {
      return;
    }
    const baseIndex = selectedFlagIndex ?? 0;
    jumpToConfidenceFlag(baseIndex + step);
  }

  if (loading) {
    return (
      <Card className="border-[var(--border-muted)] bg-[var(--bg-surface-1)]">
        <CardContent className="flex min-h-[320px] items-center justify-center gap-2 py-12 text-sm text-[var(--text-muted)]">
          <Loader2 className="h-4 w-4 animate-spin" />
          Loading transcript...
        </CardContent>
      </Card>
    );
  }

  if (!record || transcriptEntries.length === 0) {
    return (
      <div className="flex min-h-[320px] items-center justify-center px-6 py-12 text-center">
        <div className="max-w-sm">
          <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] shadow-sm">
            <Users className="h-8 w-8 text-[var(--text-subtle)]" />
          </div>
          <p className="mb-2 text-base font-semibold text-[var(--text-secondary)]">
            {emptyTitle}
          </p>
          <p className="text-sm text-[var(--text-muted)] leading-relaxed">
            {emptyMessage}
          </p>
        </div>
      </div>
    );
  }

  const floatingPlaybackFooter = stickyPlaybackFooter || fixedPlaybackFooter;
  const rootClassName = floatingPlaybackFooter
    ? "relative flex min-h-0 flex-col"
    : "flex h-full min-h-0 flex-col overflow-hidden rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)]";
  const contentClassName = fixedPlaybackFooter
    ? "grid gap-6 pb-32 xl:grid-cols-[minmax(0,1fr),248px]"
    : stickyPlaybackFooter
      ? "grid gap-6 pb-20 xl:grid-cols-[minmax(0,1fr),248px]"
      : "grid gap-6 xl:grid-cols-[minmax(0,1fr),248px] pb-20";
  const playbackClassName = fixedPlaybackFooter
    ? "fixed inset-x-0 bottom-0 z-40 border-t border-[var(--border-muted)] bg-[color-mix(in_srgb,var(--bg-surface-0)_96%,transparent)] shadow-[0_-18px_48px_-30px_rgba(15,23,42,0.45)] backdrop-blur lg:left-[var(--app-shell-left)]"
    : stickyPlaybackFooter
      ? "sticky bottom-0 -mx-4 -mb-4 mt-auto border-t border-[var(--border-muted)] bg-[color-mix(in_srgb,var(--bg-surface-0)_95%,transparent)] p-3 backdrop-blur sm:-mx-5 sm:-mb-5 sm:px-5 sm:py-3"
      : "sticky bottom-0 -mx-4 -mb-4 mt-auto border-t border-[var(--border-muted)] bg-[color-mix(in_srgb,var(--bg-surface-0)_95%,transparent)] p-3 backdrop-blur sm:-mx-5 sm:-mb-5 sm:px-5 sm:py-3";
  const playbackInnerClassName = fixedPlaybackFooter
    ? "mx-auto flex w-full max-w-[calc(100vw-var(--app-shell-left))] flex-col gap-2.5 px-4 py-3 sm:px-6"
    : "flex flex-col gap-2.5";

  return (
    <div className={rootClassName}>
      <audio {...playback.audioProps} className="hidden" />

      <div className={contentClassName}>
        <div className="space-y-5">
          <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3.5 py-3.5">
            <div className="flex items-center justify-between gap-3">
              <h3 className="text-[13px] font-semibold tracking-wide text-[var(--text-primary)]">
                Summary
              </h3>
              <StatusBadge tone={diarizationSummaryStatusTone(summaryStatus)}>
                {diarizationSummaryStatusLabel(summaryStatus)}
              </StatusBadge>
            </div>
            {summaryText ? (
              <p className="mt-2 text-[15px] leading-7 text-[var(--text-secondary)]">
                {summaryText}
              </p>
            ) : summaryStatus === "pending" ? (
              <p className="mt-2 text-[12px] text-[var(--text-muted)]">
                Generating summary...
              </p>
            ) : summaryStatus === "failed" ? (
              <p className="mt-2 text-[12px] text-[var(--danger-text)]">
                {summaryError || "Summary generation failed."}
              </p>
            ) : (
              <p className="mt-2 text-[12px] text-[var(--text-muted)]">
                No summary available yet.
              </p>
            )}
            {normalizedSummaryGuidance && summaryStatus !== "ready" ? (
              <p className="mt-2 text-[11px] leading-4 text-[var(--status-warning-text)]">
                {normalizedSummaryGuidance}
              </p>
            ) : null}
            <div className="mt-2 text-[11px] text-[var(--text-muted)]">
              {record.summary_model_id || "Qwen3.5-4B"}
              {summaryUpdatedLabel ? ` • Updated ${summaryUpdatedLabel}` : ""}
            </div>
          </div>
          <div>
            <h3 className="mb-3 text-[13px] font-semibold tracking-wide text-[var(--text-primary)]">
              Transcript
            </h3>
            {confidenceFlags.length > 0 ? (
              <div
                data-testid="diarization-confidence-nav"
                className="mb-3 rounded-lg border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] px-3 py-2.5"
              >
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2 text-[12px] font-semibold text-[var(--status-warning-text)]">
                    <AlertTriangle className="h-4 w-4" />
                    Uncertain turns
                    <span className="rounded-full bg-[var(--bg-surface-0)] px-2 py-0.5 text-[11px] font-medium text-[var(--text-primary)]">
                      {confidenceFlags.length} flagged
                    </span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="h-7 border-[var(--status-warning-border)] text-[var(--status-warning-text)] hover:bg-[var(--status-warning-bg)]"
                      onClick={() => stepConfidenceFlag(-1)}
                      disabled={confidenceFlags.length === 0}
                      aria-label="Previous flagged turn"
                    >
                      Prev flagged
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="h-7 border-[var(--status-warning-border)] text-[var(--status-warning-text)] hover:bg-[var(--status-warning-bg)]"
                      onClick={() => stepConfidenceFlag(1)}
                      disabled={confidenceFlags.length === 0}
                      aria-label="Next flagged turn"
                    >
                      Next flagged
                    </Button>
                  </div>
                </div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {confidenceFlags.map((flag, flagIndex) => {
                    const selected = selectedFlagIndex === flagIndex;
                    const confidenceLabel =
                      flag.averageConfidence !== null
                        ? `${Math.round(flag.averageConfidence * 100)}%`
                        : "Review";
                    return (
                      <button
                        key={`${flag.entryIndex}-${flag.start}`}
                        type="button"
                        className="rounded-full border px-2.5 py-1 text-[11px] font-medium transition-colors"
                        style={{
                          borderColor: "var(--status-warning-border)",
                          backgroundColor: selected
                            ? "var(--bg-surface-0)"
                            : "transparent",
                          color: "var(--status-warning-text)",
                        }}
                        aria-label={`Jump to flagged turn ${flagIndex + 1}`}
                        onClick={() => jumpToConfidenceFlag(flagIndex)}
                      >
                        {formatClockTime(flag.start)} · {flag.speaker} ·{" "}
                        {confidenceLabel}
                      </button>
                    );
                  })}
                </div>
              </div>
            ) : null}
            <div className="space-y-2.5">
              {transcriptEntries.map((entry, index) => {
                const active = index === activeEntryIndex;
                const confidenceFlag = confidenceFlagsByEntry.get(index) ?? null;
                const speakerIndex = speakerSummaries.findIndex(
                  (s) => s.displaySpeaker === entry.speaker,
                );
                const accent = speakerAccent(
                  speakerIndex >= 0 ? speakerIndex : index,
                );

                return (
                  <button
                    key={`${entry.speaker}-${entry.start}-${entry.end}-${index}`}
                    type="button"
                    onClick={() => seek(entry.start)}
                    ref={(element) => {
                      if (element) {
                        transcriptEntryRefs.current.set(index, element);
                      } else {
                        transcriptEntryRefs.current.delete(index);
                      }
                    }}
                    className="w-full rounded-lg border px-3.5 py-3 text-left transition-colors"
                    data-active={active ? "true" : "false"}
                    style={{
                      backgroundColor: active ? accent.soft : "transparent",
                      borderColor: active ? accent.solid : "transparent",
                      boxShadow: active
                        ? `0 0 0 1px ${accent.solid} inset`
                        : undefined,
                    }}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="min-w-0 flex items-center gap-2">
                        <div className="truncate text-[13px] font-semibold text-[var(--text-primary)]">
                          {entry.speaker}
                        </div>
                        <div className="text-[11px] font-medium text-[var(--text-muted)]">
                          {formatClockTime(entry.start)}
                        </div>
                      </div>
                      {active ? (
                        <span
                          className="rounded-full px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.14em]"
                          style={{
                            color: accent.solid,
                            backgroundColor: accent.soft,
                          }}
                        >
                          Live
                        </span>
                      ) : null}
                    </div>
                    {confidenceFlag ? (
                      <div className="mt-1 flex flex-wrap items-center gap-1.5">
                        <span className="rounded-full border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-[0.1em] text-[var(--status-warning-text)]">
                          Review
                        </span>
                        <span className="text-[11px] text-[var(--status-warning-text)]">
                          {confidenceFlag.reason}
                        </span>
                      </div>
                    ) : null}
                    <p className="mt-1.5 text-[15px] leading-7 text-[var(--text-secondary)]">
                      {entry.text}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        <div className="space-y-5">
          <div>
            <h3 className="mb-3 text-[13px] font-semibold tracking-wide text-[var(--text-primary)]">
              Talk Time
            </h3>
            <div className="space-y-2.5">
              {speakerSummaries.map((summary, index) => {
                const accent = speakerAccent(index);
                const share =
                  totalTalkTime > 0
                    ? (summary.totalDuration / totalTalkTime) * 100
                    : 0;
                const isCurrentSpeaker =
                  summary.displaySpeaker === activeSpeaker;

                return (
                  <div
                    key={summary.displaySpeaker}
                    className="rounded-lg border px-3 py-2.5 transition-colors"
                    style={{
                      backgroundColor: isCurrentSpeaker
                        ? accent.soft
                        : "var(--bg-surface-0)",
                      borderColor: isCurrentSpeaker
                        ? accent.solid
                        : "var(--border-muted)",
                    }}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="min-w-0">
                        <div className="truncate text-sm font-semibold text-[var(--text-primary)]">
                          {summary.displaySpeaker}
                        </div>
                        <div className="mt-1 text-[11px] leading-5 text-[var(--text-muted)]">
                          {summary.utteranceCount} turns • {summary.wordCount}{" "}
                          words
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-base font-semibold text-[var(--text-primary)]">
                          {formatDurationLabel(summary.totalDuration)}
                        </div>
                        <div className="text-[11px] text-[var(--text-muted)]">
                          {share.toFixed(0)}%
                        </div>
                      </div>
                    </div>
                    <div className="mt-2.5 h-1.5 rounded-full bg-[var(--bg-surface-2)]">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${Math.max(share, 6)}%`,
                          backgroundColor: accent.solid,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {showPlayback ? (
        <div
          data-testid="diarization-review-player"
          className={playbackClassName}
        >
          <AudioPlayer
            playback={playback}
            className={playbackInnerClassName}
            segments={transcriptEntries.map((entry, index) => {
              const speakerIndex = speakerSummaries.findIndex(
                (summary) => summary.displaySpeaker === entry.speaker,
              );
              const accent = speakerAccent(
                speakerIndex >= 0 ? speakerIndex : index,
              );
              return {
                start: entry.start,
                end: entry.end,
                active: index === activeEntryIndex,
                solid: accent.solid,
                soft: accent.soft,
              };
            })}
            trailing={activeSpeaker ?? ""}
          />
        </div>
      ) : null}
    </div>
  );
}
