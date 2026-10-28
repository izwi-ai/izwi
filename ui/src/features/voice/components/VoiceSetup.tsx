import {
  downloadPercent,
  SetupRow,
  type SetupRowTone,
} from "@/components/SetupRow";
import type { VoicePageProps } from "@/features/voice/realtime/support";
import {
  formatBytes,
  isUsableVoiceModelStatus,
} from "@/features/voice/realtime/support";
import {
  voiceEngineTotalBytes,
  type VoiceEngineModel,
} from "@/features/voice/engine";

type DownloadProgress = VoicePageProps["downloadProgress"];

function rowStatus(
  row: VoiceEngineModel,
  progress: DownloadProgress,
): { text: string; tone: SetupRowTone; percent?: number } {
  const model = row.model;
  if (!model) return { text: "Unavailable", tone: "error" };
  if (isUsableVoiceModelStatus(model.status)) return { text: "Ready", tone: "done" };
  if (model.status === "downloading") {
    const percent = downloadPercent(model, progress);
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
  compactLabel,
}: {
  rows: VoiceEngineModel[];
  downloadProgress: DownloadProgress;
  /** Still to download, for the button label. */
  downloadBytes: number;
  onDownload: () => void;
  onCancel?: () => void;
  /** Offered while the Standard engine is chosen. */
  onUseCompact?: () => void;
  /** The model the compact engine would use. */
  compactLabel?: string;
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
        <h1 className="text-2xl font-medium tracking-[-0.02em] text-(--text-primary)">
          Set up voice
        </h1>
        <p className="text-sm leading-relaxed text-(--text-muted)">
          {isDownloading
            ? "Voice runs entirely on this device. You can leave this page while it sets up."
            : rows.length === 1
              ? "Voice runs entirely on this device. It needs one model, downloaded once."
              : `Voice runs entirely on this device. It needs ${rows.length} models, downloaded once.`}
        </p>
      </div>

      <ul className="border-t border-(--border-muted)">
        {rows.map((row) => {
          const status = rowStatus(row, downloadProgress);
          return (
            <SetupRow
              key={row.variant}
              title={row.role}
              model={row.label}
              status={status.text}
              tone={status.tone}
              percent={status.percent}
            />
          );
        })}
      </ul>

      <div className="flex flex-col gap-3.5">
        <div className="flex items-center gap-2.5">
          {isDownloading ? (
            <button
              type="button"
              disabled
              className="h-11 flex-1 rounded-full bg-(--bg-surface-3) px-5 text-sm font-medium text-(--text-muted)"
            >
              Setting up… {formatBytes(doneBytes)} of {formatBytes(totalBytes)}
            </button>
          ) : (
            <button
              type="button"
              onClick={onDownload}
              disabled={unavailable}
              className="h-11 flex-1 rounded-full bg-(--accent-solid) px-5 text-sm font-medium text-(--text-on-accent) transition-opacity hover:opacity-90 disabled:opacity-50"
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
              className="h-11 rounded-full border border-(--border-default) px-4 text-sm text-(--text-primary) transition-colors hover:bg-(--bg-surface-2)"
            >
              Cancel
            </button>
          )}
        </div>
        {onUseCompact && (
          <p className="text-[13px] text-(--text-muted)">
            Prefer a smaller download?{" "}
            <button
              type="button"
              onClick={onUseCompact}
              className="text-(--text-primary) underline decoration-(--border-strong) underline-offset-[3px] hover:text-(--text-muted)"
            >
              Use the compact engine
            </button>
            {compactLabel ? ` (${compactLabel}).` : "."}
          </p>
        )}
      </div>
    </div>
  );
}
