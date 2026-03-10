import { useEffect, useMemo, useRef, useState } from "react";
import { Loader2, Pause, Play, SkipBack, SkipForward } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import type { DiarizationRecord } from "../api";
import {
  speakerSummariesFromRecord,
  transcriptEntriesFromRecord,
} from "../utils/diarizationTranscript";

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
  > | null;
  audioUrl?: string | null;
  loading?: boolean;
  emptyMessage?: string;
}

type SpeakerAccent = {
  solid: string;
  soft: string;
  border: string;
};

const PLAYBACK_SPEEDS = [0.75, 1, 1.25, 1.5, 2];

function formatClockTime(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds < 0) {
    return "0:00";
  }
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

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

export function DiarizationReviewWorkspace({
  record,
  audioUrl = null,
  loading = false,
  emptyMessage = "No diarization transcript is available yet.",
}: DiarizationReviewWorkspaceProps) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [audioError, setAudioError] = useState<string | null>(null);

  const transcriptEntries = useMemo(
    () => (record ? transcriptEntriesFromRecord(record) : []),
    [record],
  );
  const speakerSummaries = useMemo(
    () => (record ? speakerSummariesFromRecord(record) : []),
    [record],
  );

  const viewerDuration = useMemo(() => {
    const transcriptDuration = transcriptEntries.reduce(
      (max, entry) => Math.max(max, entry.end),
      0,
    );
    const recordDuration =
      record?.duration_secs && record.duration_secs > 0
        ? record.duration_secs
        : 0;
    return Math.max(duration, recordDuration, transcriptDuration, 0);
  }, [duration, record, transcriptEntries]);

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
      setAudioError("Unable to start playback for this diarization audio.");
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
      <Card className="border-[var(--border-muted)] bg-[var(--bg-surface-1)]">
        <CardContent className="py-12 text-center text-sm text-[var(--text-muted)]">
          {emptyMessage}
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="flex flex-col h-full relative">
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
          setAudioError("Unable to load audio for this diarization record.")
        }
        className="hidden"
      />

      <div className="grid gap-8 xl:grid-cols-[minmax(0,1fr),280px] pb-24">
        <div className="space-y-6">
          <div>
            <h3 className="text-sm font-medium text-[var(--text-primary)] mb-4">
              Transcript
            </h3>
            <div className="space-y-3">
              {transcriptEntries.map((entry, index) => {
                const active = index === activeEntryIndex;
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
                    className="w-full rounded-xl border p-4 text-left transition-colors"
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
                        <div className="truncate text-sm font-semibold text-[var(--text-primary)]">
                          {entry.speaker}
                        </div>
                        <div className="text-[12px] text-[var(--text-muted)]">
                          {formatClockTime(entry.start)}
                        </div>
                      </div>
                      {active ? (
                        <span
                          className="rounded-md px-2 py-1 text-[10px] font-semibold"
                          style={{
                            color: accent.solid,
                            backgroundColor: accent.soft,
                          }}
                        >
                          Live
                        </span>
                      ) : null}
                    </div>
                    <p className="mt-2 text-base leading-relaxed text-[var(--text-secondary)]">
                      {entry.text}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div>
            <h3 className="text-sm font-medium text-[var(--text-primary)] mb-4">
              Talk Time
            </h3>
            <div className="space-y-3">
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
                    className="rounded-xl border p-3 transition-colors"
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
                        <div className="mt-1 text-[11px] text-[var(--text-muted)]">
                          {summary.utteranceCount} turns • {summary.wordCount}{" "}
                          words
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-semibold text-[var(--text-primary)]">
                          {formatDurationLabel(summary.totalDuration)}
                        </div>
                        <div className="text-[11px] text-[var(--text-muted)]">
                          {share.toFixed(0)}%
                        </div>
                      </div>
                    </div>
                    <div className="mt-3 h-2 rounded-full bg-[var(--bg-surface-2)]">
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

      <div className="sticky bottom-0 -mx-4 -mb-4 mt-auto border-t border-[var(--border-muted)] bg-[var(--bg-surface-0)]/95 p-4 backdrop-blur sm:-mx-5 sm:-mb-5 sm:px-6 sm:py-4">
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <Button
                type="button"
                variant="outline"
                size="icon"
                className="h-9 w-9 rounded-full border-[var(--border-muted)] bg-[var(--bg-surface-1)] hover:bg-[var(--bg-surface-2)]"
                onClick={() => skip(-10)}
                disabled={!audioUrl}
                title="Rewind 10 seconds"
              >
                <SkipBack className="h-4 w-4" />
              </Button>
              <Button
                type="button"
                size="icon"
                className="h-10 w-10 rounded-full bg-[var(--text-primary)] text-[var(--bg-surface-0)] hover:bg-[var(--text-secondary)] shadow-md"
                onClick={() => void togglePlayback()}
                disabled={!audioUrl}
              >
                {isPlaying ? (
                  <Pause className="h-5 w-5" fill="currentColor" />
                ) : (
                  <Play className="h-5 w-5 ml-0.5" fill="currentColor" />
                )}
              </Button>
              <Button
                type="button"
                variant="outline"
                size="icon"
                className="h-9 w-9 rounded-full border-[var(--border-muted)] bg-[var(--bg-surface-1)] hover:bg-[var(--bg-surface-2)]"
                onClick={() => skip(10)}
                disabled={!audioUrl}
                title="Forward 10 seconds"
              >
                <SkipForward className="h-4 w-4" />
              </Button>
            </div>

            <div className="text-sm font-medium text-[var(--text-primary)] font-mono tabular-nums tracking-tight">
              {formatClockTime(currentTime)}{" "}
              <span className="text-[var(--text-muted)] font-normal">
                / {formatClockTime(viewerDuration)}
              </span>
            </div>

            <div className="flex-1 mx-2 relative flex items-center h-10 group">
              <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-8 pointer-events-none overflow-hidden rounded bg-[var(--bg-surface-2)]/50">
                {viewerDuration > 0
                  ? transcriptEntries.map((entry, index) => {
                      const left = (entry.start / viewerDuration) * 100;
                      const width =
                        ((entry.end - entry.start) / viewerDuration) * 100;
                      const speakerIndex = speakerSummaries.findIndex(
                        (s) => s.displaySpeaker === entry.speaker,
                      );
                      const accent = speakerAccent(
                        speakerIndex >= 0 ? speakerIndex : index,
                      );
                      const active = index === activeEntryIndex;

                      return (
                        <div
                          key={`wave-${index}`}
                          className="absolute top-0 bottom-0 transition-all duration-200 border-r border-[var(--bg-surface-0)]"
                          style={{
                            left: `${left}%`,
                            width: `${Math.max(width, 0.5)}%`,
                            backgroundColor: active
                              ? accent.solid
                              : accent.soft,
                            opacity: active ? 1 : 0.7,
                          }}
                        />
                      );
                    })
                  : null}
              </div>

              <Slider
                min={0}
                max={viewerDuration || 0}
                step={0.05}
                value={[Math.min(currentTime, viewerDuration || 0)]}
                onValueChange={(values) => seek(values[0] ?? 0)}
                disabled={!audioUrl || viewerDuration <= 0}
                className="cursor-pointer z-10 relative [&_[role=slider]]:h-4 [&_[role=slider]]:w-4 [&_[role=slider]]:border-[var(--text-primary)] [&_[role=slider]]:bg-[var(--text-primary)] [&_[role=slider]]:shadow-md [&_[data-orientation=horizontal]]:h-8 [&_.bg-primary]:bg-[var(--text-primary)]/20 [&_.bg-primary\/20]:bg-transparent"
              />
            </div>

            <div className="flex items-center gap-2">
              <div className="text-sm font-semibold min-w-[80px] text-right truncate text-[var(--text-primary)]">
                {activeSpeaker ?? ""}
              </div>

              <select
                className="h-8 rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2 text-xs text-[var(--text-primary)] outline-none"
                value={playbackRate}
                onChange={(e) => updatePlaybackRate(Number(e.target.value))}
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
            <div className="rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-sm text-[var(--danger-text)]">
              {audioError}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
