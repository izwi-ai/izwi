import { useEffect, useMemo, useRef } from "react";
import { Loader2 } from "lucide-react";

import {
  AudioPlayer,
  formatClockTime,
} from "@/components/audio/AudioPlayer";
import { useAudioPlayback } from "@/components/audio/useAudioPlayback";
import { FactsList, formatProcessingFact } from "@/components/FactsList";
import { formatClockDuration } from "@/components/history/format";
import { Card, CardContent } from "@/components/ui/card";
import type { TranscriptionRecord } from "@/api";
import {
  normalizeSummaryStatus,
  summaryStatusLabel,
} from "@/features/transcription/playground/support";
import {
  transcriptionEntriesFromRecord,
  transcriptionHasTimestamps,
  transcriptionWordCount,
} from "@/features/transcription/transcript";

const ACCENT_SOLID = "hsl(189 76% 42%)";
const ACCENT_SOFT = "hsla(189 76% 42% / 0.16)";

interface TranscriptionReviewWorkspaceProps {
  record: Pick<
    TranscriptionRecord,
    | "id"
    | "aligner_model_id"
    | "audio_filename"
    | "duration_secs"
    | "language"
    | "transcription"
    | "segments"
    | "words"
    | "speaker_attributed_text"
    | "speaker_turns"
    | "summary_status"
    | "summary_model_id"
    | "summary_text"
    | "summary_error"
    | "summary_updated_at"
  > &
    Partial<Pick<TranscriptionRecord, "processing_time_ms" | "rtf">> | null;
  audioUrl?: string | null;
  loading?: boolean;
  emptyTitle?: string;
  emptyMessage?: string;
  showPlayback?: boolean;
  autoScrollActiveEntry?: boolean;
  stickyPlaybackFooter?: boolean;
  fixedPlaybackFooter?: boolean;
  summaryModelGuidance?: string | null;
}

function isEntryActive(
  currentTime: number,
  start: number,
  end: number,
  epsilon = 0.08,
): boolean {
  return (
    Number.isFinite(start) &&
    Number.isFinite(end) &&
    end > start &&
    currentTime >= Math.max(0, start - epsilon) &&
    currentTime < end + epsilon
  );
}

export function TranscriptionReviewWorkspace({
  record,
  audioUrl = null,
  loading = false,
  emptyTitle = "Ready to transcribe",
  emptyMessage =
    "Record audio from your microphone or upload an audio file to start transcription. The transcript will appear here.",
  showPlayback = true,
  autoScrollActiveEntry = false,
  stickyPlaybackFooter = false,
  fixedPlaybackFooter = false,
  summaryModelGuidance = null,
}: TranscriptionReviewWorkspaceProps) {
  const transcriptEntryRefs = useRef(new Map<number, HTMLElement>());
  const lastAutoScrolledEntryRef = useRef<number | null>(null);

  const transcriptEntries = useMemo(
    () => transcriptionEntriesFromRecord(record),
    [record],
  );
  const timestamped = useMemo(
    () => transcriptionHasTimestamps(record),
    [record],
  );
  const wordCount = useMemo(() => transcriptionWordCount(record), [record]);
  const summaryStatus = useMemo(
    () =>
      normalizeSummaryStatus(
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
      record?.duration_secs && record.duration_secs > 0 ? record.duration_secs : 0;
    return Math.max(recordDuration, transcriptDuration);
  }, [record, transcriptEntries]);
  const playback = useAudioPlayback(audioUrl, {
    resetKey: record?.id,
    durationHint: knownDuration,
    loadErrorMessage: "Unable to load audio for this transcription record.",
    playErrorMessage: "Unable to start playback for this transcription audio.",
  });
  const { currentTime, isPlaying, seek } = playback;

  const activeEntryIndex = transcriptEntries.findIndex(
    (entry) => entry.timed && isEntryActive(currentTime, entry.start, entry.end),
  );
  const activeEntry =
    activeEntryIndex >= 0 ? transcriptEntries[activeEntryIndex] ?? null : null;

  useEffect(() => {
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

    const activeEntryElement = transcriptEntryRefs.current.get(activeEntryIndex);
    activeEntryElement?.scrollIntoView({
      block: "center",
      behavior: "smooth",
      inline: "nearest",
    });
    lastAutoScrolledEntryRef.current = activeEntryIndex;
  }, [activeEntryIndex, autoScrollActiveEntry, isPlaying]);

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
          <div className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-0)] shadow-sm">
            <svg
              aria-hidden="true"
              viewBox="0 0 24 24"
              className="h-8 w-8 text-[var(--text-subtle)]"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7z" />
              <path d="M14 2v5h5" />
              <path d="M9 13h6" />
              <path d="M9 17h6" />
              <path d="M9 9h2" />
            </svg>
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
    ? "grid gap-5 pb-32 xl:grid-cols-[minmax(0,1fr),260px]"
    : stickyPlaybackFooter
    ? "grid gap-5 pb-20 xl:grid-cols-[minmax(0,1fr),260px]"
    : "grid flex-1 min-h-0 gap-5 overflow-y-auto px-4 py-4 sm:px-5 sm:py-5 xl:grid-cols-[minmax(0,1fr),260px]";
  const playbackClassName = fixedPlaybackFooter
    ? "fixed inset-x-0 bottom-0 z-40 border-t border-[var(--border-muted)] bg-[color-mix(in_srgb,var(--bg-surface-0)_96%,transparent)] shadow-[0_-18px_48px_-30px_rgba(15,23,42,0.45)] backdrop-blur lg:left-[var(--app-shell-left)]"
    : stickyPlaybackFooter
    ? "sticky bottom-0 -mx-4 -mb-4 mt-auto border-t border-[var(--border-muted)] bg-[color-mix(in_srgb,var(--bg-surface-0)_95%,transparent)] px-4 py-3 backdrop-blur sm:-mx-5 sm:-mb-5 sm:px-5 sm:py-3"
    : "border-t border-[var(--border-muted)] bg-[color-mix(in_srgb,var(--bg-surface-0)_95%,transparent)] px-4 py-3 backdrop-blur sm:px-5";
  const playbackInnerClassName = fixedPlaybackFooter
    ? "mx-auto flex w-full max-w-[calc(100vw-var(--app-shell-left))] flex-col gap-2.5 px-4 py-3 sm:px-6"
    : "flex flex-col gap-2.5";

  return (
    <div className={rootClassName}>
      <audio {...playback.audioProps} className="hidden" />

      <div className={contentClassName}>
        <div className="space-y-4">
          <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3.5 py-3.5">
            <div className="flex items-center justify-between gap-3">
              <h3 className="text-[13px] font-semibold tracking-wide text-[var(--text-primary)]">
                Summary
              </h3>
              {summaryStatus !== "ready" ? (
                <span className="text-xs text-[var(--text-muted)]">
                  {summaryStatusLabel(summaryStatus)}
                </span>
              ) : null}
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
            <div className="space-y-2.5">
              {transcriptEntries.map((entry, index) => {
                const active = index === activeEntryIndex;
                const sharedClassName =
                  "w-full rounded-lg border px-3.5 py-3 text-left transition-colors";
                const style = {
                  backgroundColor: active ? ACCENT_SOFT : "var(--bg-surface-0)",
                  borderColor: active
                    ? ACCENT_SOLID
                    : "var(--border-muted)",
                  boxShadow: active
                    ? `0 0 0 1px ${ACCENT_SOLID} inset`
                    : undefined,
                };
                const content = (
                  <>
                    <div className="flex items-center justify-between gap-3">
                      <div className="min-w-0 flex items-center gap-2">
                        <div className="truncate text-[13px] font-semibold text-[var(--text-primary)]">
                          {entry.timed
                            ? `${formatClockTime(entry.start)} - ${formatClockTime(entry.end)}`
                            : "Transcript"}
                        </div>
                      </div>
                      {active ? (
                        <span
                          className="rounded-full px-2 py-0.5 text-[11px] font-medium"
                          style={{
                            color: ACCENT_SOLID,
                            backgroundColor: ACCENT_SOFT,
                          }}
                        >
                          Live
                        </span>
                      ) : null}
                    </div>
                    <p className="mt-1.5 text-[15px] leading-7 text-[var(--text-secondary)]">
                      {entry.text}
                    </p>
                  </>
                );

                if (!entry.timed) {
                  return (
                    <div
                      key={`entry-${index}`}
                      ref={(element) => {
                        if (element) {
                          transcriptEntryRefs.current.set(index, element);
                        } else {
                          transcriptEntryRefs.current.delete(index);
                        }
                      }}
                      className={sharedClassName}
                      style={style}
                      data-active={active ? "true" : "false"}
                    >
                      {content}
                    </div>
                  );
                }

                return (
                  <button
                    key={`entry-${entry.start}-${entry.end}-${index}`}
                    type="button"
                    onClick={() => seek(entry.start)}
                    ref={(element) => {
                      if (element) {
                        transcriptEntryRefs.current.set(index, element);
                      } else {
                        transcriptEntryRefs.current.delete(index);
                      }
                    }}
                    className={sharedClassName}
                    style={style}
                    data-active={active ? "true" : "false"}
                  >
                    {content}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        <FactsList
          label="Overview"
          className="self-start"
          facts={[
            { label: "Duration", value: formatClockDuration(record.duration_secs) },
            { label: "Words", value: wordCount },
            { label: "Language", value: record.language || "Unknown" },
            {
              label: "Timestamps",
              value: timestamped
                ? record.words && record.words.length > 0
                  ? "Word-level"
                  : "Segment-level"
                : "Off",
            },
            {
              label: "Speakers",
              value:
                record.speaker_turns && record.speaker_turns.length > 0
                  ? `${new Set(record.speaker_turns.map((turn) => turn.speaker)).size} identified`
                  : "Not requested",
            },
            ...(formatProcessingFact(record.processing_time_ms, record.rtf)
              ? [
                  {
                    label: "Processed in",
                    value: formatProcessingFact(
                      record.processing_time_ms,
                      record.rtf,
                    ),
                  },
                ]
              : []),
          ]}
        />
      </div>

      {showPlayback ? (
        <div
          data-testid="transcription-review-player"
          className={playbackClassName}
        >
          <AudioPlayer
            playback={playback}
            className={playbackInnerClassName}
            segments={transcriptEntries.map((entry, index) => ({
              start: entry.start,
              end: entry.end,
              active: index === activeEntryIndex,
            }))}
            trailing={activeEntry?.timed ? formatClockTime(activeEntry.start) : ""}
          />
        </div>
      ) : null}
    </div>
  );
}
