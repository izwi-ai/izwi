import {
  Download,
  FolderOpen,
  Loader2,
  Pin,
  Play,
  RotateCcw,
  Square,
  Trash2,
  X,
} from "lucide-react";

import type { ModelInfo } from "@/api";
import { RowBadge } from "@/components/history/RowBadge";
import { RowMenu, type RowMenuItem } from "@/components/history/RowMenu";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const GIB = 1024 ** 3;

/** A size on disk, in the decimal units Finder and Explorer show. */
export function formatModelBytes(bytes: number): string {
  if (bytes < 1e3) return `${bytes} B`;
  if (bytes < 1e6) return `${(bytes / 1e3).toFixed(0)} KB`;
  if (bytes < 1e9) return `${(bytes / 1e6).toFixed(0)} MB`;
  return `${(bytes / 1e9).toFixed(1)} GB`;
}

/** Memory a model needs, for one that cannot load on this device. */
export function neededMemoryLabel(model: ModelInfo): string | null {
  if (model.device_fit !== "too_large") {
    return null;
  }
  const bytes =
    model.estimated_memory_bytes ?? (model.memory_required_gb ?? 0) * GIB;
  return bytes > 0
    ? `Needs ${Math.ceil(bytes / GIB)} GB memory`
    : "Too large for this device";
}

export function isModelOnDisk(model: ModelInfo): boolean {
  return model.status === "downloaded" || model.status === "ready";
}

export interface ModelRowActions {
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  /** Asks to delete the download; the caller confirms. */
  onDelete?: (variant: string) => void;
  /** "Keep loaded" on or off. */
  onSetPinned?: (variant: string, pinned: boolean) => void;
  /** Opens the model's folder in the file manager. */
  onReveal?: (model: ModelInfo) => void;
  revealLabel?: string;
}

/** The status column: a pill, download progress, or why it can't load. */
export function ModelStatusCell({
  model,
  progress,
}: {
  model: ModelInfo;
  progress: number;
}) {
  const live = {
    role: model.status === "error" ? "alert" : "status",
    "aria-live": model.status === "error" ? "assertive" : "polite",
    "aria-atomic": true,
  } as const;
  switch (model.status) {
    case "ready":
      return (
        <span {...live}>
          <RowBadge tone="positive" dot>
            Loaded
          </RowBadge>
        </span>
      );
    case "downloading":
      return (
        <div {...live} className="flex flex-col gap-1.5">
          <span className="whitespace-nowrap text-[12.5px] text-[var(--text-primary)]">
            Downloading · {Math.round(progress)}%
          </span>
          <div className="h-[3px] w-full max-w-28 overflow-hidden rounded-full bg-[var(--surface-raised)]">
            <div
              className="h-full rounded-full bg-[var(--text-primary)] transition-[width]"
              style={{ width: `${Math.min(Math.max(progress, 0), 100)}%` }}
            />
          </div>
        </div>
      );
    case "loading":
      return (
        <span {...live}>
          <RowBadge tone="info" icon={<Loader2 className="h-3 w-3 animate-spin" />}>
            Loading
          </RowBadge>
        </span>
      );
    case "downloaded":
      return (
        <span {...live}>
          <RowBadge>Downloaded</RowBadge>
        </span>
      );
    case "error":
      return (
        <span {...live} title={model.error_message ?? undefined}>
          <RowBadge tone="danger" dot>
            Didn't download
          </RowBadge>
        </span>
      );
    default: {
      const needs = neededMemoryLabel(model);
      return (
        <span {...live} className="whitespace-nowrap text-[12.5px] text-[var(--text-subtle)]">
          {needs ?? "Not downloaded"}
        </span>
      );
    }
  }
}

interface ModelListRowProps {
  model: ModelInfo;
  label: string;
  /** The second line: what it is, where it's used, and its provider. */
  subtitle: string;
  progress: number;
  actions: ModelRowActions;
  /** Replaces Unload as the main action of a loaded model (Use). */
  readyAction?: { label: string; onClick: () => void } | null;
  /** The model the page asked for. */
  highlighted?: boolean;
}

/**
 * One model: name and subtitle, status, size, one outline action and a
 * "⋯" menu. The Models page and every page's Models dialog use it.
 */
export function ModelListRow({
  model,
  label,
  subtitle,
  progress,
  actions,
  readyAction = null,
  highlighted = false,
}: ModelListRowProps) {
  const tooLarge = model.device_fit === "too_large" && !isModelOnDisk(model);
  const size = model.size_bytes ?? model.estimated_size_bytes ?? null;
  const common = "h-8 gap-1.5 px-2.5 text-[13px]";

  let primary = null;
  switch (model.status) {
    case "not_downloaded":
    case "error":
      primary = tooLarge ? null : (
        <Button
          type="button"
          variant="outline"
          size="sm"
          className={common}
          onClick={() => actions.onDownload(model.variant)}
          aria-label={`${model.status === "error" ? "Retry" : "Download"} ${label}`}
        >
          {model.status === "error" ? (
            <RotateCcw className="h-3.5 w-3.5" />
          ) : (
            <Download className="h-3.5 w-3.5" />
          )}
          {model.status === "error" ? "Retry" : "Download"}
        </Button>
      );
      break;
    case "downloading":
      primary = actions.onCancelDownload ? (
        <Button
          type="button"
          variant="outline"
          size="sm"
          className={common}
          onClick={() => actions.onCancelDownload?.(model.variant)}
          aria-label={`Cancel the download of ${label}`}
        >
          <X className="h-3.5 w-3.5" />
          Cancel
        </Button>
      ) : null;
      break;
    case "downloaded":
      primary = (
        <Button
          type="button"
          variant="outline"
          size="sm"
          className={common}
          onClick={() => actions.onLoad(model.variant)}
          aria-label={`Load ${label}`}
        >
          <Play className="h-3.5 w-3.5" />
          Load
        </Button>
      );
      break;
    case "loading":
      primary = (
        <Button
          type="button"
          variant="outline"
          size="sm"
          className={common}
          onClick={() => actions.onUnload(model.variant)}
          aria-label={`Cancel loading ${label}`}
        >
          <X className="h-3.5 w-3.5" />
          Cancel
        </Button>
      );
      break;
    case "ready":
      primary = readyAction ? (
        <Button
          type="button"
          variant="outline"
          size="sm"
          className={common}
          onClick={readyAction.onClick}
          aria-label={`${readyAction.label} ${label}`}
        >
          {readyAction.label}
        </Button>
      ) : (
        <Button
          type="button"
          variant="outline"
          size="sm"
          className={common}
          onClick={() => actions.onUnload(model.variant)}
          aria-label={`Unload ${label}`}
        >
          <Square className="h-3.5 w-3.5" />
          Unload
        </Button>
      );
      break;
  }

  const menu: RowMenuItem[] = [];
  if (model.status === "ready" && actions.onSetPinned) {
    menu.push({
      id: "keep-loaded",
      label: "Keep loaded",
      icon: Pin,
      checked: Boolean(model.pinned),
      onSelect: () => actions.onSetPinned?.(model.variant, !model.pinned),
    });
  }
  if (model.status === "ready" && readyAction) {
    menu.push({
      id: "unload",
      label: "Unload",
      icon: Square,
      onSelect: () => actions.onUnload(model.variant),
    });
  }
  if (actions.onReveal && model.local_path && isModelOnDisk(model)) {
    menu.push({
      id: "reveal",
      label: actions.revealLabel ?? "Show in folder",
      icon: FolderOpen,
      onSelect: () => actions.onReveal?.(model),
    });
  }
  if (
    actions.onDelete &&
    (isModelOnDisk(model) || (model.status === "error" && model.local_path))
  ) {
    menu.push({
      id: "delete",
      label: "Delete download",
      icon: Trash2,
      danger: true,
      separatorBefore: menu.length > 0,
      onSelect: () => actions.onDelete?.(model.variant),
    });
  }

  return (
    <li
      data-testid={`model-row-${model.variant}`}
      className={cn(
        "grid grid-cols-[minmax(0,1fr)_auto_auto_auto] items-center gap-x-3 gap-y-2 border-b border-[var(--border-subtle)] px-4 py-3 last:border-b-0 sm:grid-cols-[minmax(0,1fr)_8.5rem_4.5rem_6.5rem_1.75rem]",
        highlighted && "bg-[var(--accent-soft)]",
      )}
    >
      <div className="col-span-4 flex min-w-0 flex-col sm:col-span-1">
        <span
          className={cn(
            "truncate text-[13.5px] font-medium",
            tooLarge ? "text-[var(--text-subtle)]" : "text-[var(--text-primary)]",
          )}
        >
          {label}
        </span>
        <span
          className="truncate text-[12.5px] text-[var(--text-subtle)]"
          title={model.status === "error" ? (model.error_message ?? undefined) : undefined}
        >
          {model.status === "error" && model.error_message
            ? model.error_message
            : subtitle}
        </span>
      </div>
      <ModelStatusCell model={model} progress={progress} />
      <span
        className={cn(
          "whitespace-nowrap text-right text-[13px] tabular-nums",
          tooLarge ? "text-[var(--text-subtle)]" : "text-[var(--text-muted)]",
        )}
      >
        {size !== null ? formatModelBytes(size) : "—"}
      </span>
      <div className="flex justify-end">{primary}</div>
      <div className="flex justify-end">
        {menu.length > 0 ? <RowMenu label={label} items={menu} /> : null}
      </div>
    </li>
  );
}
