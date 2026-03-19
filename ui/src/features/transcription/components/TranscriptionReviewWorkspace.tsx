import { useEffect, useMemo, useRef, useState } from "react";
import { Loader2, Pause, Play, SkipBack, SkipForward } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import type { TranscriptionRecord } from "@/api";
import {
  transcriptionEntriesFromRecord,
  transcriptionHasTimestamps,
  transcriptionWordCount,
} from "@/features/transcription/transcript";

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
  > | null;
  audioUrl?: string | null;
  loading?: boolean;
  emptyTitle?: string;
  emptyMessage?: string;
  showPlayback?: boolean;
}

const PLAYBACK_SPEEDS = [0.75, 1, 1.25, 1.5, 2];
const ACCENT_SOLID = "hsl(189 76% 42%)";
const ACCENT_SOFT = "hsla(189 76% 42% / 0.16)";

function formatClockTime(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds < 0) {
    return "0:00";
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function formatDurationLabel(seconds: number | null | undefined): string {
  if (!Number.isFinite(seconds) || seconds === null || seconds === undefined) {
    return "Unknown";
  }
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.round(seconds % 60);
  return `${minutes}m ${remainingSeconds}s`;
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
}: TranscriptionReviewWorkspaceProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [audioError, setAudioError] = useState<string | null>(null);

  const transcriptEntries = useMemo(
    () => transcriptionEntriesFromRecord(record),
    [record],
  );
  const timestamped = useMemo(
    () => transcriptionHasTimestamps(record),
    [record],
  );
  const wordCount = useMemo(() => transcriptionWordCount(record), [record]);

  const viewerDuration = useMemo(() => {
    const transcriptDuration = transcriptEntries.reduce(
      (max, entry) => Math.max(max, entry.end),
      0,
    );
    const recordDuration =
      record?.duration_secs && record.duration_secs > 0 ? record.duration_secs : 0;
    return Math.max(duration, recordDuration, transcriptDuration, 0);
  }, [duration, record, transcriptEntries]);

  const activeEntryIndex = transcriptEntries.findIndex(
    (entry) => entry.timed && isEntryActive(currentTime, entry.start, entry.end),
  );
  const activeEntry =
    activeEntryIndex >= 0 ? transcriptEntries[activeEntryIndex] ?? null : null;

  useEffect(() => {
    const audio = audioRef.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
      audio.playbackRate = 1;
    }
    setCurrentTime(0);
    setDuration(0);
    setIsPlaying(false);
    setPlaybackRate(1);
    setAudioError(null);
  }, [audioUrl, record?.id]);

  async function togglePlayback(): Promise<void> {
    const audio = audioRef.current;
    if (!audio || !audioUrl) {
      return;
    }

    try {
      if (audio.paused) {
        await audio.play();
      } else {
        audio.pause();
      }
    } catch {
      setAudioError("Unable to start playback for this transcription audio.");
    }
  }

  function seek(nextTime: number): void {
    const audio = audioRef.current;
    const clamped = Math.max(0, Math.min(nextTime, viewerDuration || 0));
    if (audio) {
      audio.currentTime = clamped;
    }
    setCurrentTime(clamped);
  }

  function skip(deltaSeconds: number): void {
    seek(currentTime + deltaSeconds);
  }

  function updatePlaybackRate(nextRate: number): void {
    const audio = audioRef.current;
    if (audio) {
      audio.playbackRate = nextRate;
    }
    setPlaybackRate(nextRate);
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
          <p className="mb-2 text-[1.75rem] font-semibold leading-none tracking-[-0.03em] text-[var(--text-primary)]">
            {emptyTitle}
          </p>
          <p className="text-sm leading-relaxed text-[var(--text-muted)]">
            {emptyMessage}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)]">
      <audio
        ref={audioRef}
        src={audioUrl ?? undefined}
        preload="metadata"
        onLoadedMetadata={(event) => {
          const nextDuration = Number.isFinite(event.currentTarget.duration)
            ? event.currentTarget.duration
            : 0;
          setDuration(nextDuration);
          setAudioError(null);
        }}
        onTimeUpdate={(event) => {
          setCurrentTime(event.currentTarget.currentTime);
        }}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onRateChange={(event) =>
          setPlaybackRate(event.currentTarget.playbackRate)
        }
        onError={() =>
          setAudioError("Unable to load audio for this transcription record.")
        }
        className="hidden"
      />

      <div className="grid flex-1 min-h-0 gap-6 overflow-y-auto px-4 py-4 sm:px-5 sm:py-5 xl:grid-cols-[minmax(0,1fr),248px]">
        <div className="space-y-4">
          <div>
            <h3 className="mb-3 text-[13px] font-semibold tracking-wide text-[var(--text-primary)]">
              Transcript
            </h3>
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
                          className="rounded-full px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.14em]"
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

        <div className="space-y-5">
          <div>
            <h3 className="mb-3 text-[13px] font-semibold tracking-wide text-[var(--text-primary)]">
              Overview
            </h3>
            <div className="space-y-2.5">
              <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-3">
                <div className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-subtle)]">
                  Duration
                </div>
                <div className="mt-1 text-base font-semibold text-[var(--text-primary)]">
                  {formatDurationLabel(record.duration_secs)}
                </div>
              </div>

              <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-3">
                <div className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-subtle)]">
                  Words
                </div>
                <div className="mt-1 text-base font-semibold text-[var(--text-primary)]">
                  {wordCount}
                </div>
              </div>

              <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-3">
                <div className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-subtle)]">
                  Timestamps
                </div>
                <div className="mt-1 text-base font-semibold text-[var(--text-primary)]">
                  {timestamped ? "Enabled" : "Disabled"}
                </div>
                <div className="mt-1 text-[11px] text-[var(--text-muted)]">
                  {timestamped
                    ? record.aligner_model_id || "Forced aligner"
                    : "Plain transcript only"}
                </div>
              </div>

              <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-3">
                <div className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-subtle)]">
                  Language
                </div>
                <div className="mt-1 text-base font-semibold text-[var(--text-primary)]">
                  {record.language || "Unknown"}
                </div>
                {record.audio_filename ? (
                  <div className="mt-1 truncate text-[11px] text-[var(--text-muted)]">
                    {record.audio_filename}
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </div>
      </div>

      {showPlayback ? (
        <div className="border-t border-[var(--border-muted)] bg-[var(--bg-surface-0)]/95 px-4 py-3 backdrop-blur sm:px-5">
          <div className="flex flex-col gap-2.5">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  className="h-8 w-8 rounded-full border-[var(--border-muted)] bg-[var(--bg-surface-1)] hover:bg-[var(--bg-surface-2)]"
                  onClick={() => skip(-10)}
                  disabled={!audioUrl}
                  title="Rewind 10 seconds"
                >
                  <SkipBack className="h-4 w-4" />
                </Button>
                <Button
                  type="button"
                  size="icon"
                  className="h-9 w-9 rounded-full bg-[var(--text-primary)] text-[var(--bg-surface-0)] hover:bg-[var(--text-secondary)] shadow-md"
                  onClick={() => void togglePlayback()}
                  disabled={!audioUrl}
                >
                  {isPlaying ? (
                    <Pause className="h-4 w-4" fill="currentColor" />
                  ) : (
                    <Play className="ml-0.5 h-4 w-4" fill="currentColor" />
                  )}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  className="h-8 w-8 rounded-full border-[var(--border-muted)] bg-[var(--bg-surface-1)] hover:bg-[var(--bg-surface-2)]"
                  onClick={() => skip(10)}
                  disabled={!audioUrl}
                  title="Forward 10 seconds"
                >
                  <SkipForward className="h-4 w-4" />
                </Button>
              </div>

              <div className="font-mono text-[13px] font-medium tabular-nums tracking-tight text-[var(--text-primary)]">
                {formatClockTime(currentTime)}{" "}
                <span className="font-normal text-[var(--text-muted)]">
                  / {formatClockTime(viewerDuration)}
                </span>
              </div>

              <div className="group relative mx-1.5 flex h-9 flex-1 items-center">
                <div className="pointer-events-none absolute inset-x-0 top-1/2 h-7 -translate-y-1/2 overflow-hidden rounded bg-[var(--bg-surface-2)]/50">
                  {viewerDuration > 0
                    ? transcriptEntries.map((entry, index) => {
                        const left = (entry.start / viewerDuration) * 100;
                        const width =
                          entry.end > entry.start
                            ? ((entry.end - entry.start) / viewerDuration) * 100
                            : 100;
                        const active = index === activeEntryIndex;

                        return (
                          <div
                            key={`timeline-${index}`}
                            className="absolute bottom-0 top-0 border-r border-[var(--bg-surface-0)] transition-all duration-200"
                            style={{
                              left: `${Math.max(left, 0)}%`,
                              width: `${Math.max(width, 0.75)}%`,
                              backgroundColor: active ? ACCENT_SOLID : ACCENT_SOFT,
                              opacity: active ? 1 : 0.8,
                            }}
                          />
                        );
                      })
                    : null}
                </div>

                <input
                  type="range"
                  min={0}
                  max={viewerDuration || 0}
                  step={0.05}
                  value={Math.min(currentTime, viewerDuration || 0)}
                  onChange={(event) => seek(Number(event.target.value))}
                  aria-label="Seek audio timeline"
                  disabled={!audioUrl || viewerDuration <= 0}
                  className="relative z-10 h-7 w-full cursor-pointer appearance-none bg-transparent accent-[var(--text-primary)] focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-45 [&::-moz-range-progress]:bg-transparent [&::-moz-range-thumb]:h-3.5 [&::-moz-range-thumb]:w-3.5 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-none [&::-moz-range-thumb]:bg-[var(--text-primary)] [&::-moz-range-thumb]:shadow-md [&::-moz-range-track]:h-1.5 [&::-moz-range-track]:rounded-full [&::-moz-range-track]:bg-transparent [&::-webkit-slider-runnable-track]:h-1.5 [&::-webkit-slider-runnable-track]:rounded-full [&::-webkit-slider-runnable-track]:bg-transparent [&::-webkit-slider-thumb]:-mt-1 [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-none [&::-webkit-slider-thumb]:bg-[var(--text-primary)] [&::-webkit-slider-thumb]:shadow-md"
                />
              </div>

              <div className="flex items-center gap-2">
                <div className="min-w-[68px] truncate text-right text-[13px] font-semibold text-[var(--text-primary)]">
                  {activeEntry?.timed ? formatClockTime(activeEntry.start) : ""}
                </div>

                <select
                  className="h-7 rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 text-[11px] font-medium text-[var(--text-primary)] outline-none"
                  value={playbackRate}
                  onChange={(event) =>
                    updatePlaybackRate(Number(event.target.value))
                  }
                >
                  {PLAYBACK_SPEEDS.map((rate) => (
                    <option key={rate} value={rate}>
                      {rate}x
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {audioError ? (
              <div className="rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]">
                {audioError}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}
