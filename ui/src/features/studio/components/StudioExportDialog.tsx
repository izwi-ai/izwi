import { useEffect, useRef, useState } from "react";
import { AlertCircle, Download, Loader2 } from "lucide-react";

import { api, type StudioProjectRecord } from "@/api";
import { formatClockDuration } from "@/components/history/format";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
import {
  isPending,
  needsRender,
  projectDurationSecs,
  scriptFilename,
} from "@/features/studio/studio";

const PAUSES = [
  { label: "None", ms: 0 },
  { label: "0.5 s", ms: 500 },
  { label: "1 s", ms: 1000 },
] as const;

function downloadText(filename: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

interface StudioExportDialogProps {
  open: boolean;
  onClose: () => void;
  project: StudioProjectRecord;
  /** Export only these paragraphs; all when absent. */
  segmentIds?: string[];
  onRender: (segmentIds?: string[]) => Promise<boolean>;
}

/** One WAV of the project, rendering what's missing first. */
export function StudioExportDialog({
  open,
  onClose,
  project,
  segmentIds,
  onRender,
}: StudioExportDialogProps) {
  const [pauseMs, setPauseMs] = useState(500);
  const [includeScript, setIncludeScript] = useState(false);
  const [phase, setPhase] = useState<"idle" | "rendering" | "saving">("idle");
  const [error, setError] = useState<string | null>(null);
  const savingRef = useRef(false);

  useEffect(() => {
    if (open) {
      setPhase("idle");
      setError(null);
    }
  }, [open]);

  const segments = segmentIds
    ? project.segments.filter((segment) => segmentIds.includes(segment.id))
    : project.segments;
  const waiting = segments.filter((segment) => needsRender(segment) || isPending(segment));
  const failed = segments.filter((segment) => segment.render_state === "failed");
  const duration = segments.reduce((total, segment) => total + (segment.audio_duration_secs ?? 0), 0);

  const save = async () => {
    if (savingRef.current) return;
    savingRef.current = true;
    setPhase("saving");
    setError(null);
    try {
      await api.downloadAudioFile(
        api.studioProjectAudioUrl(project.id, {
          download: true,
          format: "wav",
          segment_ids: segmentIds,
          pause_ms: pauseMs,
        }),
        scriptFilename(project.name, "wav"),
      );
      if (includeScript) {
        downloadText(
          scriptFilename(`${project.name}-script`, "txt"),
          segments.map((segment) => segment.text).join("\n\n"),
        );
      }
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "The export couldn't be saved.");
      setPhase("idle");
    } finally {
      savingRef.current = false;
    }
  };

  // Once the paragraphs it waited for are rendered, save the file.
  useEffect(() => {
    if (phase !== "rendering") return;
    if (segments.some(isPending)) return;
    if (failed.length > 0) {
      setError(
        failed.length === 1
          ? "1 paragraph failed to render, so nothing was exported."
          : `${failed.length} paragraphs failed to render, so nothing was exported.`,
      );
      setPhase("idle");
      return;
    }
    if (waiting.length === 0) void save();
    // `save` reads the latest options; re-running on it would loop.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, project]);

  const start = async () => {
    setError(null);
    if (waiting.length === 0) {
      void save();
      return;
    }
    const toRender = waiting.filter(needsRender).map((segment) => segment.id);
    if (toRender.length > 0 && !(await onRender(toRender))) return;
    setPhase("rendering");
  };

  const busy = phase !== "idle";
  const remaining = segments.filter(isPending).length;

  return (
    <Dialog open={open} onOpenChange={(next) => !next && !busy && onClose()}>
      <DialogContent className="max-w-[460px] gap-0 border-[var(--border-strong)] bg-[var(--bg-surface-1)] p-0">
        <div className="px-6 pb-1 pt-5">
          <DialogTitle className="text-[17px] font-semibold text-[var(--text-primary)]">
            Export audio
          </DialogTitle>
          <DialogDescription className="mt-1 text-[13px] text-[var(--text-muted)]">
            One WAV file
            {waiting.length === 0
              ? ` · ${formatClockDuration(segmentIds ? duration : projectDurationSecs(project))}`
              : ""}
            {segmentIds ? ` · ${segments.length} selected paragraphs` : ""}
          </DialogDescription>
        </div>

        <div className="flex flex-col gap-3.5 px-6 pb-1 pt-4">
          {waiting.length > 0 && phase !== "rendering" ? (
            <p className="flex items-start gap-2.5 rounded-[12px] border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] px-3.5 py-3 text-[13px] leading-5 text-[var(--status-warning-text)]">
              <AlertCircle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" />
              {waiting.length === 1
                ? "1 paragraph isn't rendered yet. Izwi renders it first, then saves the file."
                : `${waiting.length} paragraphs aren't rendered yet. Izwi renders them first, then saves the file.`}
            </p>
          ) : null}
          {phase === "rendering" ? (
            <p className="flex items-center gap-2.5 rounded-[12px] border border-[var(--border-subtle)] px-3.5 py-3 text-[13px] text-[var(--text-secondary)]">
              <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" />
              Rendering, {remaining} left…
            </p>
          ) : null}

          <div className="flex flex-col rounded-[12px] border border-[var(--border-subtle)]">
            <label className="flex items-center justify-between gap-3 border-b border-[var(--border-subtle)] px-3.5 py-3">
              <span className="flex flex-col gap-0.5">
                <span className="text-[13px] text-[var(--text-primary)]">Pause between paragraphs</span>
                <span className="text-xs text-[var(--text-muted)]">A short breath at each break</span>
              </span>
              <select
                value={pauseMs}
                onChange={(event) => setPauseMs(Number(event.target.value))}
                disabled={busy}
                className="h-8 rounded-[8px] border border-[var(--border-default)] bg-[var(--surface-panel-alt)] px-2 text-xs text-[var(--text-primary)]"
              >
                {PAUSES.map((pause) => (
                  <option key={pause.ms} value={pause.ms}>
                    {pause.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center justify-between gap-3 px-3.5 py-3">
              <span className="flex flex-col gap-0.5">
                <span className="text-[13px] text-[var(--text-primary)]">Include script</span>
                <span className="text-xs text-[var(--text-muted)]">Saves a .txt next to the audio</span>
              </span>
              <Switch checked={includeScript} onCheckedChange={setIncludeScript} disabled={busy} />
            </label>
          </div>

          {error ? (
            <p role="alert" className="flex items-start gap-2 text-[13px] text-[var(--danger-text)]">
              <AlertCircle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" />
              {error}
            </p>
          ) : null}
        </div>

        <div className="flex justify-end gap-2 px-6 pb-5 pt-4">
          <button
            type="button"
            onClick={onClose}
            disabled={busy}
            className="h-9 rounded-full border border-[var(--border-default)] px-4 text-[13px] text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-surface-2)] disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => void start()}
            disabled={busy}
            className="inline-flex h-9 items-center gap-2 rounded-full bg-[var(--accent-solid)] px-4 text-[13px] font-medium text-[var(--text-on-accent)] transition-opacity hover:opacity-90 disabled:opacity-60"
          >
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
            {waiting.length > 0 ? "Render and export" : "Export"}
          </button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
