import {
  AlertTriangle,
  CheckCircle2,
  FileAudio,
  Loader2,
  X,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  formatUploadFileSize,
  type SpeechTextUploadPhase,
} from "@/features/speech-text/uploadProgress";

interface SpeechTextUploadProgressProps {
  fileName: string;
  fileSizeBytes: number;
  fileKind: string;
  phase: SpeechTextUploadPhase;
  loadedBytes: number;
  totalBytes: number | null;
  percent: number | null;
  errorMessage?: string;
  canCancel?: boolean;
  onCancel?: () => void;
}

const phaseLabels: Record<SpeechTextUploadPhase, string> = {
  idle: "Ready",
  preparing: "Preparing audio",
  uploading: "Uploading audio",
  accepted: "Upload accepted",
  opening: "Opening job",
  failed: "Upload failed",
  cancelled: "Upload cancelled",
};

function progressDetail({
  loadedBytes,
  percent,
  totalBytes,
}: Pick<
  SpeechTextUploadProgressProps,
  "loadedBytes" | "percent" | "totalBytes"
>): string {
  if (typeof percent === "number") {
    return `${Math.round(percent)}%`;
  }

  if (loadedBytes > 0) {
    const uploaded = formatUploadFileSize(loadedBytes);
    if (totalBytes && totalBytes > 0) {
      return `${uploaded} of ${formatUploadFileSize(totalBytes)}`;
    }
    return `${uploaded} uploaded`;
  }

  return "Starting";
}

export function SpeechTextUploadProgress({
  fileName,
  fileSizeBytes,
  fileKind,
  phase,
  loadedBytes,
  totalBytes,
  percent,
  errorMessage,
  canCancel = false,
  onCancel,
}: SpeechTextUploadProgressProps) {
  const isFailed = phase === "failed";
  const isAccepted = phase === "accepted" || phase === "opening";
  const isActive = phase === "preparing" || phase === "uploading";
  const normalizedPercent =
    typeof percent === "number" ? Math.min(100, Math.max(0, percent)) : null;
  const barWidth = normalizedPercent ?? (isAccepted ? 100 : 34);
  const statusText = errorMessage || phaseLabels[phase];

  return (
    <div
      className={cn(
        "rounded-lg border bg-[var(--bg-surface-1)] p-3.5",
        isFailed
          ? "border-[var(--danger-border)]"
          : "border-[var(--border-muted)]",
      )}
    >
      <div className="flex items-start gap-3">
        <div
          className={cn(
            "flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border",
            isFailed
              ? "border-[var(--danger-border)] bg-[var(--danger-bg)] text-[var(--danger-text)]"
              : isAccepted
                ? "border-[var(--status-positive-border)] bg-[var(--status-positive-bg)] text-[var(--status-positive-text)]"
                : "border-[var(--border-muted)] bg-[var(--bg-surface-0)] text-[var(--text-primary)]",
          )}
        >
          {isFailed ? (
            <AlertTriangle className="h-4 w-4" />
          ) : isAccepted ? (
            <CheckCircle2 className="h-4 w-4" />
          ) : isActive ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <FileAudio className="h-4 w-4" />
          )}
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div
                className="truncate text-sm font-semibold text-[var(--text-primary)]"
                title={fileName}
              >
                {fileName}
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] font-medium uppercase tracking-[0.12em] text-[var(--text-muted)]">
                <span>{fileKind}</span>
                <span aria-hidden="true">/</span>
                <span>{formatUploadFileSize(fileSizeBytes)}</span>
              </div>
            </div>

            {canCancel && onCancel ? (
              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="h-8 w-8 shrink-0 rounded-lg"
                aria-label="Cancel upload"
                onClick={onCancel}
              >
                <X className="h-4 w-4" />
              </Button>
            ) : null}
          </div>

          <div
            className="mt-3 h-2 overflow-hidden rounded-full bg-[var(--bg-surface-0)]"
            role="progressbar"
            aria-label="Upload progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={normalizedPercent ?? undefined}
            aria-valuetext={
              normalizedPercent === null ? progressDetail({ loadedBytes, percent, totalBytes }) : undefined
            }
          >
            <div
              className={cn(
                "h-full rounded-full bg-[var(--status-info-text)] transition-[width] duration-200",
                normalizedPercent === null && !isAccepted ? "animate-pulse" : null,
              )}
              style={{ width: `${barWidth}%` }}
            />
          </div>

          <div
            className={cn(
              "mt-2 flex items-center justify-between gap-3 text-xs",
              isFailed ? "text-[var(--danger-text)]" : "text-[var(--text-muted)]",
            )}
            aria-live="polite"
          >
            <span className="min-w-0 truncate">{statusText}</span>
            <span className="shrink-0 font-medium">
              {progressDetail({ loadedBytes, percent, totalBytes })}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
