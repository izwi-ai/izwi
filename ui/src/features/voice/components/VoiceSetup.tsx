import { Check } from "lucide-react";

import type { VoicePageProps } from "@/features/voice/realtime/support";
import {
  formatBytes,
  isUsableVoiceModelStatus,
} from "@/features/voice/realtime/support";
import {
  voiceEngineTotalBytes,
  type VoiceEngineModel,
} from "@/features/voice/engine";
import { cn } from "@/lib/utils";

type DownloadProgress = VoicePageProps["downloadProgress"];

function rowStatus(
  row: VoiceEngineModel,
  progress: DownloadProgress,
): { text: string; tone: "done" | "active" | "muted" | "error"; percent?: number } {
  const model = row.model;
  if (!model) return { text: "Unavailable", tone: "error" };
  if (isUsableVoiceModelStatus(model.status)) return { text: "Ready", tone: "done" };
  if (model.status === "downloading") {
    const percent = Math.round(
      progress[model.variant]?.percent ?? model.download_progress ?? 0,
    );
    return { text: `Downloading · ${percent}%`, tone: "active", percent };
  }
  if (model.status === "error") return { text: "Failed", tone: "error" };
  const bytes = model.estimated_size_bytes ?? model.size_bytes;
  return { text: bytes ? formatBytes(bytes) : "Not downloaded", tone: "muted" };
}

/**
 * First run: the engine's models, what each is for and one action that
 * downloads them all. The server loads them when a conversation starts.
 */
export function VoiceSetup({
  rows,
  downloadProgress,
  downloadBytes,
  onDownload,
  onCancel,
  onUseCompact,
}: {
  rows: VoiceEngineModel[];
  downloadProgress: DownloadProgress;
  /** Still to download, for the button label. */
  downloadBytes: number;
  onDownload: () => void;
  onCancel?: () => void;
  /** Offered while the Standard engine is chosen. */
  onUseCompact?: () => void;
}) {
  const isDownloading = rows.some((row) => row.model?.status === "downloading");
  const totalBytes = voiceEngineTotalBytes(rows);
  const doneBytes = rows.reduce((total, row) => {
    const model = row.model;
    if (!model) return total;
    if (isUsableVoiceModelStatus(model.status)) {
      return total + (model.size_bytes ?? model.estimated_size_bytes ?? 0);
    }
    return total + (downloadProgress[model.variant]?.downloadedBytes ?? 0);
  }, 0);
  const unavailable = rows.some((row) => !row.model);

  return (
    <div className="flex w-full max-w-[440px] flex-col gap-7">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-medium tracking-[-0.02em] text-[var(--text-primary)]">
          Set up voice
        </h1>
        <p className="text-sm leading-relaxed text-[var(--text-muted)]">
          {isDownloading
            ? "Voice runs entirely on this device. You can leave this page while it sets up."
            : rows.length === 1
              ? "Voice runs entirely on this device. It needs one model, downloaded once."
              : `Voice runs entirely on this device. It needs ${rows.length} models, downloaded once.`}
        </p>
      </div>

      <ul className="border-t border-[var(--border-muted)]">
        {rows.map((row) => {
          const status = rowStatus(row, downloadProgress);
          return (
            <li
              key={row.variant}
              className="flex flex-col gap-2.5 border-b border-[var(--border-muted)] py-3.5"
            >
              <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-4">
                <div className="flex min-w-0 flex-col gap-0.5">
                  <span className="text-sm font-medium text-[var(--text-primary)]">
                    {row.role}
                  </span>
                  <span className="truncate text-[13px] text-[var(--text-muted)]">
                    {row.label}
                  </span>
                </div>
                <span
                  className={cn(
                    "flex items-center gap-1.5 whitespace-nowrap text-[13px]",
                    status.tone === "done" && "text-[var(--status-positive-text)]",
                    status.tone === "active" && "text-[var(--text-primary)]",
                    status.tone === "muted" && "text-[var(--text-muted)]",
                    status.tone === "error" && "text-[var(--danger-text)]",
                  )}
                >
                  {status.tone === "done" && (
                    <Check className="h-[15px] w-[15px]" strokeWidth={2.5} />
                  )}
                  {status.text}
                </span>
              </div>
              {status.percent !== undefined && (
                <div
                  role="progressbar"
                  aria-label={`${row.role} download`}
                  aria-valuenow={status.percent}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  className="h-1 overflow-hidden rounded-full bg-[var(--bg-surface-3)]"
                >
                  <div
                    className="h-full rounded-full bg-[var(--accent-solid)] transition-[width] duration-300"
                    style={{ width: `${status.percent}%` }}
                  />
                </div>
              )}
            </li>
          );
        })}
      </ul>

      <div className="flex flex-col gap-3.5">
        <div className="flex items-center gap-2.5">
          {isDownloading ? (
            <button
              type="button"
              disabled
              className="h-11 flex-1 rounded-full bg-[var(--bg-surface-3)] px-5 text-sm font-medium text-[var(--text-muted)]"
            >
              Setting up… {formatBytes(doneBytes)} of {formatBytes(totalBytes)}
            </button>
          ) : (
            <button
              type="button"
              onClick={onDownload}
              disabled={unavailable}
              className="h-11 flex-1 rounded-full bg-[var(--accent-solid)] px-5 text-sm font-medium text-[var(--text-on-accent)] transition-opacity hover:opacity-90 disabled:opacity-50"
            >
              {downloadBytes > 0
                ? `Download and set up · ${formatBytes(downloadBytes)}`
                : "Download and set up"}
            </button>
          )}
          {isDownloading && onCancel && (
            <button
              type="button"
              onClick={onCancel}
              className="h-11 rounded-full border border-[var(--border-default)] px-4 text-sm text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-surface-2)]"
            >
              Cancel
            </button>
          )}
        </div>
        {onUseCompact && (
          <p className="text-[13px] text-[var(--text-muted)]">
            Prefer a smaller download?{" "}
            <button
              type="button"
              onClick={onUseCompact}
              className="text-[var(--text-primary)] underline decoration-[var(--border-strong)] underline-offset-[3px] hover:text-[var(--text-muted)]"
            >
              Use the compact engine
            </button>{" "}
            (LFM2.5 Audio).
          </p>
        )}
      </div>
    </div>
  );
}
