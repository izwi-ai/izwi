import clsx from "clsx";
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  Loader2,
  Play,
  Square,
  Trash2,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { ModelInfo } from "@/api";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  getModelProviderLabel,
  PROVIDER_ORDER,
} from "@/features/models/catalog/modelMetadata";
import type {
  ModelDownloadProgressEntry,
  ModelDownloadProgressMap,
} from "@/features/models/downloadProgress";
import { withQwen3Prefix } from "@/utils/modelDisplay";

interface RouteModelSection {
  key: string;
  title?: string;
  description?: string;
  models: ModelInfo[];
}

interface ProviderGroup {
  provider: string;
  models: ModelInfo[];
}

interface PreparedRouteModelSection extends RouteModelSection {
  providerGroups: ProviderGroup[];
}

interface RouteModelModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  description: string;
  models: ModelInfo[];
  loading: boolean;
  selectedVariant: string | null;
  intentVariant?: string | null;
  downloadProgress: ModelDownloadProgressMap;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onUseModel: (variant: string) => void;
  emptyMessage?: string;
  sections?: RouteModelSection[];
  canUseModel?: (variant: string) => boolean;
  getModelLabel?: (variant: string) => string;
  selectionMode?: "route" | "manage";
  zIndexClassName?: string;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function getStatusDotClass(status: ModelInfo["status"]): string {
  switch (status) {
    case "ready":
      return "bg-[var(--status-positive-solid)]";
    case "downloaded":
      return "bg-[var(--text-secondary)]";
    case "downloading":
    case "loading":
      return "bg-[var(--status-warning-text)]";
    case "error":
      return "bg-[var(--danger-text)]";
    default:
      return "bg-[var(--text-subtle)]";
  }
}

function getProviderLabel(variant: string): string {
  return getModelProviderLabel(variant);
}

function compareProviders(left: string, right: string): number {
  const leftRank = PROVIDER_ORDER.indexOf(
    left as (typeof PROVIDER_ORDER)[number],
  );
  const rightRank = PROVIDER_ORDER.indexOf(
    right as (typeof PROVIDER_ORDER)[number],
  );
  const normalizedLeftRank = leftRank === -1 ? Number.MAX_SAFE_INTEGER : leftRank;
  const normalizedRightRank =
    rightRank === -1 ? Number.MAX_SAFE_INTEGER : rightRank;
  if (normalizedLeftRank !== normalizedRightRank) {
    return normalizedLeftRank - normalizedRightRank;
  }
  return left.localeCompare(right);
}

function groupModelsByProvider(models: ModelInfo[]): ProviderGroup[] {
  const grouped = new Map<string, ModelInfo[]>();
  for (const model of models) {
    const provider = getProviderLabel(model.variant);
    const bucket = grouped.get(provider);
    if (bucket) {
      bucket.push(model);
    } else {
      grouped.set(provider, [model]);
    }
  }
  return Array.from(grouped.entries())
    .sort(([left], [right]) => compareProviders(left, right))
    .map(([provider, sectionModels]) => ({
      provider,
      models: sectionModels,
    }));
}

function defaultModelLabel(variant: string): string {
  return withQwen3Prefix(variant, variant);
}

function getModelSizeLabel(
  model: ModelInfo,
  progress: ModelDownloadProgressEntry | undefined,
): string {
  if (progress && progress.totalBytes > 0) {
    return formatBytes(progress.totalBytes);
  }
  if (model.size_bytes !== null) {
    return formatBytes(model.size_bytes);
  }
  return "Size unknown";
}

function requiresManualDownload(variant: string): boolean {
  return variant === "Gemma-3-1b-it";
}

const MANUAL_GEMMA_DOWNLOAD_GUIDE =
  "https://github.com/izwi-ai/izwi/blob/main/docs/user/models/manual-gemma-3-1b-download.md";

export function RouteModelModal({
  isOpen,
  onClose,
  title,
  description,
  models,
  loading,
  selectedVariant,
  intentVariant,
  downloadProgress,
  onDownload,
  onCancelDownload,
  onLoad,
  onUnload,
  onDelete,
  onUseModel,
  emptyMessage = "No models are available for this route.",
  sections,
  canUseModel,
  getModelLabel,
  selectionMode = "route",
  zIndexClassName = "z-50",
}: RouteModelModalProps) {
  const [deleteTargetVariant, setDeleteTargetVariant] = useState<string | null>(
    null,
  );
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const deleteReturnFocusRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!isOpen) {
      setDeleteTargetVariant(null);
    }
  }, [isOpen]);

  const modalSections = useMemo<PreparedRouteModelSection[]>(() => {
    const baseSections =
      sections && sections.length > 0 ? sections : [{ key: "models", models }];
    return baseSections.map((section) => ({
      ...section,
      providerGroups: groupModelsByProvider(section.models),
    }));
  }, [models, sections]);

  const orderedModels = useMemo(
    () => modalSections.flatMap((section) => section.models),
    [modalSections],
  );

  const deleteTargetModel = deleteTargetVariant
    ? orderedModels.find((model) => model.variant === deleteTargetVariant) ?? null
    : null;

  const resolveModelLabel = (variant: string): string => {
    if (getModelLabel) {
      return getModelLabel(variant);
    }
    return defaultModelLabel(variant);
  };

  const handleConfirmDelete = () => {
    if (!deleteTargetModel) {
      return;
    }
    onDelete(deleteTargetModel.variant);
    setDeleteTargetVariant(null);
  };

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(open) => {
        if (!open) {
          onClose();
        }
      }}
    >
      <DialogContent
        onOpenAutoFocus={() => {
          returnFocusRef.current =
            document.activeElement instanceof HTMLElement
              ? document.activeElement
              : null;
        }}
        onCloseAutoFocus={(event) => {
          if (!returnFocusRef.current) {
            return;
          }
          event.preventDefault();
          returnFocusRef.current.focus();
          returnFocusRef.current = null;
        }}
        className={clsx(
          "flex max-w-4xl flex-col gap-0 overflow-hidden border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-0",
          zIndexClassName,
        )}
      >
        <div className="border-b border-[var(--border-muted)] px-4 py-4 pr-14 sm:px-5 sm:pr-14">
          <DialogTitle className="text-base text-[var(--text-primary)]">
            {title}
          </DialogTitle>
          <DialogDescription className="mt-1 text-xs text-[var(--text-muted)]">
            {description}
          </DialogDescription>
        </div>

            <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4 sm:px-5">
              {loading ? (
                <div className="flex items-center gap-2 py-4 text-sm text-[var(--text-muted)]">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading models...
                </div>
              ) : orderedModels.length === 0 ? (
                <div className="py-4 text-sm text-[var(--text-muted)]">
                  {emptyMessage}
                </div>
              ) : (
                <div className="space-y-4">
                  {modalSections.map((section) => (
                    <section key={section.key} className="space-y-2">
                      {section.title && (
                        <div className="px-1">
                          <h3 className="text-xs font-semibold uppercase tracking-wide text-[var(--text-muted)]">
                            {section.title}
                          </h3>
                          {section.description && (
                            <p className="mt-0.5 text-[11px] text-[var(--text-subtle)]">
                              {section.description}
                            </p>
                          )}
                        </div>
                      )}

                      {section.models.length === 0 ? (
                        <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3 text-xs text-[var(--text-subtle)]">
                          No models in this group.
                        </div>
                      ) : (
                        <div className="space-y-2">
                          {section.providerGroups.map((providerGroup) => (
                            <div
                              key={`${section.key}-${providerGroup.provider}`}
                              className="space-y-2"
                            >
                              <div className="flex items-center gap-2 px-1">
                                <h4 className="text-[11px] font-semibold uppercase tracking-wide text-[var(--text-subtle)]">
                                  {providerGroup.provider}
                                </h4>
                                <span className="text-[10px] text-[var(--text-subtle)]">
                                  {providerGroup.models.length}
                                </span>
                                <div className="h-px flex-1 bg-[var(--border-muted)]" />
                              </div>

                              {providerGroup.models.map((model) => {
                                const selectionEnabled =
                                  selectionMode === "route";
                                const isSelected =
                                  selectionEnabled &&
                                  selectedVariant === model.variant;
                                const isIntent = intentVariant === model.variant;
                                const progressValue =
                                  downloadProgress[model.variant];
                                const progress =
                                  progressValue?.percent ??
                                  model.download_progress ??
                                  0;
                                const canSelect =
                                  selectionEnabled &&
                                  (canUseModel
                                    ? canUseModel(model.variant)
                                    : true);
                                const modelSizeLabel = getModelSizeLabel(
                                  model,
                                  progressValue,
                                );
                                const modelLabel = resolveModelLabel(
                                  model.variant,
                                );

                                return (
                                  <div
                                    key={model.variant}
                                    data-testid={`route-model-row-${model.variant}`}
                                    role="group"
                                    aria-label={`${modelLabel}. Status: ${model.status.replace(/_/g, " ")}.`}
                                    className={clsx(
                                      "rounded-xl border px-3 py-2.5 transition-colors",
                                      isIntent
                                        ? "border-[var(--border-strong)] bg-[var(--bg-surface-2)]"
                                        : isSelected
                                          ? "border-[var(--border-strong)] bg-[var(--bg-surface-1)]"
                                          : "border-[var(--border-muted)] bg-[var(--bg-surface-1)]",
                                    )}
                                  >
                                    <div className="flex items-center justify-between gap-3">
                                      <div className="min-w-0 flex items-center gap-2">
                                        {model.status === "downloading" ||
                                        model.status === "loading" ? (
                                          <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-[var(--status-warning-text)]" />
                                        ) : (
                                          <span
                                            className={clsx(
                                              "h-2 w-2 shrink-0 rounded-full",
                                              getStatusDotClass(model.status),
                                            )}
                                          />
                                        )}
                                        <h3 className="truncate text-sm font-medium text-[var(--text-primary)]">
                                          {modelLabel}
                                        </h3>
                                      </div>

                                      <div className="shrink-0 flex items-center gap-1.5">
                                        <span className="mr-1 text-xs text-[var(--text-subtle)] whitespace-nowrap">
                                          {modelSizeLabel}
                                        </span>
                                        {model.status === "downloading" && (
                                          <span className="text-xs text-[var(--status-warning-text)] whitespace-nowrap">
                                            {Math.round(progress)}%
                                          </span>
                                        )}

                                        {model.status === "downloading" &&
                                          onCancelDownload && (
                                            <button
                                              type="button"
                                              onClick={() =>
                                                onCancelDownload(model.variant)
                                              }
                                              className="flex items-center gap-1 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-2.5 py-1.5 text-xs font-medium text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)]"
                                            >
                                              <X aria-hidden="true" className="h-3.5 w-3.5" />
                                              Cancel
                                            </button>
                                          )}

                                        {(model.status === "not_downloaded" ||
                                          model.status === "error") &&
                                          (requiresManualDownload(
                                            model.variant,
                                          ) ? (
                                            <a
                                              href={MANUAL_GEMMA_DOWNLOAD_GUIDE}
                                              target="_blank"
                                              rel="noreferrer"
                                              className="flex items-center gap-1.5 rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-muted)]"
                                              aria-label={`Open manual download guide for ${modelLabel}`}
                                            >
                                              <Download aria-hidden="true" className="h-3.5 w-3.5" />
                                              Manual download guide
                                            </a>
                                          ) : (
                                            <button
                                              type="button"
                                              onClick={() =>
                                                onDownload(model.variant)
                                              }
                                              className="flex items-center gap-1.5 rounded-md bg-[var(--accent-solid)] px-3 py-1.5 text-xs font-medium text-[var(--text-on-accent)] transition-opacity hover:opacity-90"
                                            >
                                              <Download aria-hidden="true" className="h-3.5 w-3.5" />
                                              Download
                                            </button>
                                          ))}

                                        {model.status === "downloaded" && (
                                          <button
                                            type="button"
                                            onClick={() => onLoad(model.variant)}
                                            className="flex items-center gap-1.5 rounded-md bg-[var(--accent-solid)] px-3 py-1.5 text-xs font-medium text-[var(--text-on-accent)] transition-opacity hover:opacity-90"
                                          >
                                            <Play aria-hidden="true" className="h-3.5 w-3.5" />
                                            Load
                                          </button>
                                        )}

                                        {model.status === "loading" && (
                                          <button
                                            type="button"
                                            onClick={() =>
                                              onUnload(model.variant)
                                            }
                                            className="flex items-center gap-1.5 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-1.5 text-xs font-medium text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)]"
                                          >
                                            <X aria-hidden="true" className="h-3.5 w-3.5" />
                                            Cancel load
                                          </button>
                                        )}

                                        {selectionEnabled &&
                                          model.status === "ready" &&
                                          canSelect &&
                                          (isSelected ? (
                                            <button
                                              type="button"
                                              className="flex items-center gap-1.5 rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-muted)]"
                                              disabled
                                            >
                                              <CheckCircle2 aria-hidden="true" className="h-3.5 w-3.5" />
                                              Selected
                                            </button>
                                          ) : (
                                            <button
                                              type="button"
                                              onClick={() => {
                                                onUseModel(model.variant);
                                                onClose();
                                              }}
                                              className="flex items-center gap-1.5 rounded-md bg-[var(--accent-solid)] px-3 py-1.5 text-xs font-medium text-[var(--text-on-accent)] transition-opacity hover:opacity-90"
                                            >
                                              <CheckCircle2 aria-hidden="true" className="h-3.5 w-3.5" />
                                              Use model
                                            </button>
                                          ))}

                                        {model.status === "ready" && (
                                          <button
                                            type="button"
                                            onClick={() =>
                                              onUnload(model.variant)
                                            }
                                            className="flex items-center gap-1.5 rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-surface-3)]"
                                          >
                                            <Square aria-hidden="true" className="h-3.5 w-3.5" />
                                            Unload
                                          </button>
                                        )}

                                        {(model.status === "downloaded" ||
                                          model.status === "ready") && (
                                          <button
                                            type="button"
                                            onClick={(event) => {
                                              deleteReturnFocusRef.current =
                                                event.currentTarget;
                                              setDeleteTargetVariant(
                                                model.variant,
                                              );
                                            }}
                                            className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)]"
                                            title="Delete model"
                                            aria-label={`Delete ${modelLabel}`}
                                          >
                                            <Trash2 className="h-3.5 w-3.5" />
                                          </button>
                                        )}
                                      </div>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          ))}
                        </div>
                      )}
                    </section>
                  ))}
                </div>
              )}
            </div>
        <Dialog
          open={Boolean(deleteTargetModel)}
          onOpenChange={(open) => {
            if (!open) {
              setDeleteTargetVariant(null);
            }
          }}
        >
          {deleteTargetModel ? (
            <DialogContent
              onCloseAutoFocus={(event) => {
                if (!deleteReturnFocusRef.current?.isConnected) {
                  return;
                }
                event.preventDefault();
                deleteReturnFocusRef.current.focus();
                deleteReturnFocusRef.current = null;
              }}
              className="z-[80] max-w-md gap-0 border-[var(--border-strong)] bg-[var(--bg-surface-1)] p-5"
            >
                  <div className="flex items-start gap-3">
                    <div className="mt-0.5 rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2 text-[var(--danger-text)]">
                      <AlertTriangle aria-hidden="true" className="h-4 w-4" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <DialogTitle className="text-sm text-[var(--text-primary)]">
                        Delete model?
                      </DialogTitle>
                      <DialogDescription className="mt-1 text-sm text-[var(--text-muted)]">
                        This removes{" "}
                        <span className="mx-1 font-medium text-[var(--text-primary)]">
                          {resolveModelLabel(deleteTargetModel.variant)}
                        </span>
                        {" "}from local storage.
                      </DialogDescription>
                      <p className="mt-2 truncate text-xs text-[var(--text-subtle)]">
                        {deleteTargetModel.variant}
                      </p>
                    </div>
                  </div>

                  <div className="mt-5 flex items-center justify-end gap-2">
                    <button
                      type="button"
                      onClick={() => setDeleteTargetVariant(null)}
                      className="rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-surface-3)]"
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      onClick={handleConfirmDelete}
                      className="flex items-center gap-1.5 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-1.5 text-xs font-medium text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)]"
                    >
                      <Trash2 aria-hidden="true" className="h-3.5 w-3.5" />
                      Delete model
                    </button>
                  </div>
            </DialogContent>
          ) : null}
        </Dialog>
      </DialogContent>
    </Dialog>
  );
}
