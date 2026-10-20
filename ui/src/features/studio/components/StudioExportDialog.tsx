import { useEffect, useRef, useState } from "react";
import { AlertCircle, Loader2 } from "lucide-react";

import { api, type StudioProjectRecord } from "@/api";
import { CreationDialog } from "@/components/CreationDialog";
import {
  EXPORT_DIALOG_CLASS,
  ExportChoiceRow,
  ExportError,
  ExportSettings,
  ExportSwitchRow,
  exportFooterNote,
} from "@/components/ExportDialog";
import { formatClockDuration } from "@/components/history/format";
import {
  isPending,
  needsRender,
  projectDurationSecs,
  scriptFilename,
} from "@/features/studio/studio";
import { saveFile } from "@/shared/saveFile";

const PAUSES = [
  { id: "0", label: "None" },
  { id: "500", label: "0.5 s" },
  { id: "1000", label: "1 s" },
];

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
      const saved = await api.downloadAudioFile(
        api.studioProjectAudioUrl(project.id, {
          download: true,
          format: "wav",
          segment_ids: segmentIds,
          pause_ms: pauseMs,
        }),
        scriptFilename(project.name, "wav"),
      );
      // A cancelled save dialog leaves this one open to try again.
      if (!saved) {
        setPhase("idle");
        return;
      }
      if (includeScript) {
        await saveFile(
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
    <CreationDialog
      open={open}
      onClose={onClose}
      blockDismiss={busy}
      className={EXPORT_DIALOG_CLASS}
      title="Export audio"
      description={
        <>
          One WAV file
          {waiting.length === 0
            ? ` · ${formatClockDuration(segmentIds ? duration : projectDurationSecs(project))}`
            : ""}
          {segmentIds ? ` · ${segments.length} selected paragraphs` : ""}
        </>
      }
      onSubmit={() => void start()}
      submitLabel={waiting.length > 0 ? "Render and export…" : "Export…"}
      submitting={busy}
      footerStart={exportFooterNote()}
    >
      <div className="flex flex-col gap-3 px-5 py-4 sm:px-6">
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

        <ExportSettings>
          <ExportChoiceRow
            label="Pause between paragraphs"
            hint="A short breath at each break"
            options={PAUSES}
            value={String(pauseMs)}
            onChange={(value) => setPauseMs(Number(value))}
            disabled={busy}
          />
          <ExportSwitchRow
            title="Include script"
            description="Also saves the script as a .txt file"
            checked={includeScript}
            onCheckedChange={setIncludeScript}
            disabled={busy}
          />
        </ExportSettings>

        {error ? <ExportError message={error} /> : null}
      </div>
    </CreationDialog>
  );
}
