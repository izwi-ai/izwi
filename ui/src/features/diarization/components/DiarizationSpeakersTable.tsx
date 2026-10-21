import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { GitMerge, Play, RotateCcw, Square } from "lucide-react";

import type { DiarizationRecord } from "@/api";
import { RowMenu, type RowMenuItem } from "@/components/history/RowMenu";
import { cn } from "@/lib/utils";
import { rawSpeakerSummariesFromRecord } from "@/utils/diarizationTranscript";

type SpeakerRecord = Pick<
  DiarizationRecord,
  "id" | "segments" | "utterances" | "words" | "speaker_name_overrides"
>;

interface DiarizationSpeakersTableProps {
  record: SpeakerRecord;
  audioUrl: string | null;
  error?: string | null;
  onSave: (speakerNameOverrides: Record<string, string>) => Promise<void>;
}

interface SpeakerRow {
  rawSpeaker: string;
  turns: Array<[number, number]>;
  talkSeconds: number;
  wordCount: number;
}

const SAVE_DELAY_MS = 600;
const GRID =
  "grid grid-cols-[minmax(0,1fr)_minmax(8rem,14rem)_3.5rem_3.5rem_1.75rem] items-center gap-4";

function normalizeName(value: string): string {
  return value.trim().replace(/\s+/g, " ");
}

/** The names that differ from the detected labels: what the server stores. */
function namesToSave(names: Record<string, string>): Record<string, string> {
  return Object.fromEntries(
    Object.entries(names)
      .map(([raw, name]) => [raw, normalizeName(name)] as const)
      .filter(([raw, name]) => name.length > 0 && name !== raw),
  );
}

function sameNames(
  left: Record<string, string>,
  right: Record<string, string>,
): boolean {
  const keys = new Set([...Object.keys(left), ...Object.keys(right)]);
  return [...keys].every((key) => left[key] === right[key]);
}

/** Every detected speaker, words or not, in order of first appearance. */
function speakerRows(record: SpeakerRecord): SpeakerRow[] {
  const summaries = rawSpeakerSummariesFromRecord(record);
  const rows = new Map<string, SpeakerRow>();
  const ensure = (raw: string) => {
    const trimmed = raw.trim();
    if (!trimmed) {
      return null;
    }
    let row = rows.get(trimmed);
    if (!row) {
      const summary = summaries.find((entry) => entry.rawSpeaker === trimmed);
      row = {
        rawSpeaker: trimmed,
        turns: [],
        talkSeconds: summary?.totalDuration ?? 0,
        wordCount: summary?.wordCount ?? 0,
      };
      rows.set(trimmed, row);
    }
    return row;
  };
  for (const utterance of record.utterances ?? []) {
    const row = ensure(String(utterance.speaker ?? ""));
    const start = Number(utterance.start);
    const end = Number(utterance.end);
    if (row && Number.isFinite(start) && Number.isFinite(end) && end > start) {
      row.turns.push([start, end]);
    }
  }
  for (const segment of record.segments ?? []) {
    ensure(String(segment.speaker ?? ""));
  }
  for (const summary of summaries) {
    ensure(summary.rawSpeaker);
  }
  return [...rows.values()];
}

function formatTalkTime(seconds: number): string {
  if (seconds < 60) {
    return `${seconds.toFixed(1)} s`;
  }
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds % 60)}s`;
}

/** Plays one speaker's turns back to back on its own audio element. */
function useTurnPlayer(audioUrl: string | null) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const queueRef = useRef<Array<[number, number]>>([]);
  const endRef = useRef(0);
  const [playingSpeaker, setPlayingSpeaker] = useState<string | null>(null);

  const stop = useCallback(() => {
    queueRef.current = [];
    audioRef.current?.pause();
    setPlayingSpeaker(null);
  }, []);

  const playNext = useCallback(() => {
    const audio = audioRef.current;
    const next = queueRef.current.shift();
    if (!audio || !next) {
      stop();
      return;
    }
    endRef.current = next[1];
    audio.currentTime = next[0];
    void audio.play().catch(stop);
  }, [stop]);

  const play = useCallback(
    (speaker: string, turns: Array<[number, number]>) => {
      if (!audioUrl || turns.length === 0) {
        return;
      }
      if (!audioRef.current) {
        const audio = new Audio(audioUrl);
        audio.ontimeupdate = () => {
          if (audio.currentTime >= endRef.current) {
            playNext();
          }
        };
        audio.onended = () => playNext();
        audioRef.current = audio;
      }
      queueRef.current = [...turns];
      setPlayingSpeaker(speaker);
      playNext();
    },
    [audioUrl, playNext],
  );

  useEffect(() => () => audioRef.current?.pause(), []);
  return { playingSpeaker, play, stop };
}

export function DiarizationSpeakersTable({
  record,
  audioUrl,
  error = null,
  onSave,
}: DiarizationSpeakersTableProps) {
  const rows = useMemo(() => speakerRows(record), [record]);
  const [names, setNames] = useState<Record<string, string>>(
    () => record.speaker_name_overrides ?? {},
  );
  // Names are the user's from the first keystroke: a refresh of the record
  // does not overwrite what they are typing, and a new record starts fresh.
  useEffect(() => {
    setNames(record.speaker_name_overrides ?? {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [record.id]);

  // Saves run one at a time; edits made during a save are sent after it.
  const savedRef = useRef(namesToSave(record.speaker_name_overrides ?? {}));
  const savingRef = useRef(false);
  const latestRef = useRef(names);
  const flush = useCallback(async () => {
    if (savingRef.current) {
      return;
    }
    savingRef.current = true;
    try {
      // Until the names stop changing under the save.
      for (;;) {
        const next = namesToSave(latestRef.current);
        if (sameNames(next, savedRef.current)) {
          return;
        }
        try {
          await onSave(next);
        } catch {
          // The page shows the error; the next edit sends the names again.
          return;
        }
        savedRef.current = next;
      }
    } finally {
      savingRef.current = false;
    }
  }, [onSave]);

  // Declared before the save timer so the flush it schedules sees these names.
  useEffect(() => {
    latestRef.current = names;
  }, [names]);

  useEffect(() => {
    const timer = window.setTimeout(() => void flush(), SAVE_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [flush, names]);

  const { playingSpeaker, play, stop } = useTurnPlayer(audioUrl);
  const totalTalk = rows.reduce((sum, row) => sum + row.talkSeconds, 0);
  const displayName = (raw: string) => normalizeName(names[raw] ?? "") || raw;

  if (rows.length === 0) {
    return (
      <p className="rounded-[14px] border border-dashed border-(--border-subtle) px-4 py-6 text-sm text-(--text-muted)">
        No speakers were detected in this recording.
      </p>
    );
  }

  return (
    <div>
      <div className="overflow-x-auto rounded-[14px] border border-(--border-subtle) bg-(--surface-panel)">
        <div className="min-w-136">
          <div
            className={cn(
              GRID,
              "border-b border-(--border-subtle) px-3.5 py-2.5 text-xs font-medium text-(--text-subtle)",
            )}
          >
            <span>Speaker</span>
            <span>Talk time</span>
            <span className="text-right">Turns</span>
            <span className="text-right">Words</span>
            <span />
          </div>
          {rows.map((row, index) => {
            const name = names[row.rawSpeaker] ?? "";
            const renamed = normalizeName(name).length > 0;
            const silent = row.turns.length === 0 && row.wordCount === 0;
            const share = totalTalk > 0 ? row.talkSeconds / totalTalk : 0;
            const label = displayName(row.rawSpeaker);
            const menuItems: RowMenuItem[] = [
              playingSpeaker === row.rawSpeaker
                ? { id: "stop", label: "Stop playing", icon: Square, onSelect: stop }
                : {
                    id: "play",
                    label: "Play their turns",
                    icon: Play,
                    disabled: !audioUrl || row.turns.length === 0,
                    onSelect: () => play(row.rawSpeaker, row.turns),
                  },
              ...rows
                .filter((other) => other.rawSpeaker !== row.rawSpeaker)
                .filter((other) => displayName(other.rawSpeaker) !== label)
                .map((other) => ({
                  id: `merge-${other.rawSpeaker}`,
                  label: `Merge into ${displayName(other.rawSpeaker)}`,
                  icon: GitMerge,
                  onSelect: () =>
                    setNames((current) => ({
                      ...current,
                      [row.rawSpeaker]: displayName(other.rawSpeaker),
                    })),
                })),
              {
                id: "reset",
                label: "Reset name",
                icon: RotateCcw,
                disabled: !renamed,
                separatorBefore: true,
                onSelect: () =>
                  setNames((current) => ({ ...current, [row.rawSpeaker]: "" })),
              },
            ];
            return (
              <div
                key={row.rawSpeaker}
                className={cn(
                  GRID,
                  "px-3.5 py-2.5",
                  index < rows.length - 1 && "border-b border-(--border-subtle)",
                )}
              >
                <div className="flex min-w-0 items-center gap-3">
                  <span
                    aria-hidden="true"
                    className={cn(
                      "flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-(--speaker-avatar-border) bg-(--speaker-avatar-bg) text-[12.5px] font-medium",
                      silent
                        ? "text-(--text-subtle)"
                        : "text-(--speaker-avatar-text)",
                    )}
                  >
                    {String.fromCharCode(65 + (index % 26))}
                  </span>
                  <div className="flex min-w-0 flex-1 flex-col">
                    <input
                      type="text"
                      value={name}
                      placeholder={row.rawSpeaker}
                      aria-label={`Name for ${row.rawSpeaker}`}
                      onChange={(event) =>
                        setNames((current) => ({
                          ...current,
                          [row.rawSpeaker]: event.target.value,
                        }))
                      }
                      className={cn(
                        "-ml-2 h-7 w-full max-w-[16rem] rounded-[8px] border px-2 text-[13.5px] font-medium text-(--text-primary) outline-hidden placeholder:text-(--text-primary) focus:border-(--border-default) focus:bg-(--surface-canvas)",
                        renamed
                          ? "border-(--border-default) bg-(--surface-canvas)"
                          : "border-transparent bg-transparent hover:border-(--border-default)",
                      )}
                    />
                    <span className="text-[11.5px] text-(--text-subtle)">
                      {silent
                        ? "Detected, but no words"
                        : renamed
                          ? `Detected as ${row.rawSpeaker}`
                          : "Click the name to rename"}
                    </span>
                  </div>
                </div>
                {silent ? (
                  <span className="text-[13px] text-(--text-subtle)">—</span>
                ) : (
                  <div className="flex items-center gap-3">
                    <div className="h-1 flex-1 overflow-hidden rounded-full bg-(--bg-surface-3)">
                      <div
                        className="h-full rounded-full bg-(--accent-solid)"
                        style={{ width: `${Math.round(share * 100)}%` }}
                      />
                    </div>
                    <span className="w-26 shrink-0 whitespace-nowrap text-right text-[13px] tabular-nums text-(--text-muted)">
                      {formatTalkTime(row.talkSeconds)} · {Math.round(share * 100)}%
                    </span>
                  </div>
                )}
                <span className="text-right text-[13px] tabular-nums text-(--text-muted)">
                  {row.turns.length}
                </span>
                <span className="text-right text-[13px] tabular-nums text-(--text-muted)">
                  {row.wordCount}
                </span>
                <RowMenu
                  label={renamed ? `${label} (${row.rawSpeaker})` : label}
                  items={menuItems}
                />
              </div>
            );
          })}
        </div>
      </div>
      {error ? (
        <p role="alert" className="mt-3 text-xs text-(--danger-text)">
          {error}
        </p>
      ) : (
        <p className="mt-3 text-xs text-(--text-subtle)">
          Names save as you type. Copy, export and summaries use them.
        </p>
      )}
    </div>
  );
}
