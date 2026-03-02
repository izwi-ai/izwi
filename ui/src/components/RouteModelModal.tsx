import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
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
import { useEffect, useMemo, useState } from "react";
import { ModelInfo } from "../api";
import { MODEL_DETAILS } from "../pages/MyModelsPage";
import { withQwen3Prefix } from "../utils/modelDisplay";

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
  downloadProgress: Record<
    string,
    {
      percent: number;
      currentFile: string;
      status: string;
      downloadedBytes: number;
      totalBytes: number;
    }
  >;
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
}

const PROVIDER_ORDER = [
  "Qwen",
  "Liquid AI",
  "Google",
  "NVIDIA",
  "Mistral AI",
  "hexgrad",
  "Other",
] as const;

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
  if (variant.startsWith("Qwen3-") || variant.startsWith("Qwen3.5-")) {
    return "Qwen";
  }
  if (variant.startsWith("LFM2")) return "Liquid AI";
  if (variant.startsWith("Gemma-")) return "Google";
  if (
    variant.startsWith("Parakeet-") ||
    variant.startsWith("diar_streaming_sortformer")
  ) {
    return "NVIDIA";
  }
  if (variant.startsWith("Voxtral-")) return "Mistral AI";
  if (variant.startsWith("Kokoro-")) return "hexgrad";
  return "Other";
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
  const details = MODEL_DETAILS[variant];
  if (!details) {
    return variant;
  }
  return withQwen3Prefix(details.shortName, variant);
}

function getModelSizeLabel(
  model: ModelInfo,
  progress: {
    percent: number;
    currentFile: string;
    status: string;
    downloadedBytes: number;
    totalBytes: number;
  } | undefined,
): string {
  if (progress && progress.totalBytes > 0) {
    return formatBytes(progress.totalBytes);
  }
  if (model.size_bytes !== null) {
    return formatBytes(model.size_bytes);
  }
  const knownSize = MODEL_DETAILS[model.variant]?.size;
  if (knownSize) {
    return knownSize;
  }
  return "Size unknown";
}

function requiresManualDownload(variant: string): boolean {
  return variant === "Gemma-3-1b-it";
}

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
}: RouteModelModalProps) {
  const [deleteTargetVariant, setDeleteTargetVariant] = useState<string | null>(
    null,
  );

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
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="fixed inset-0 z-50 bg-black/70 p-4 backdrop-blur-sm sm:p-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            initial={{ y: 16, opacity: 0, scale: 0.98 }}
            animate={{ y: 0, opacity: 1, scale: 1 }}
            exit={{ y: 16, opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.2 }}
            className="mx-auto flex max-h-[90vh] max-w-4xl flex-col overflow-hidden rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)]"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-center justify-between gap-3 border-b border-[var(--border-muted)] px-4 py-4 sm:px-5">
              <div>
                <h2 className="text-base font-semibold text-[var(--text-primary)]">
                  {title}
                </h2>
                <p className="mt-1 text-xs text-[var(--text-muted)]">
                  {description}
                </p>
              </div>
              <button
                className="flex items-center gap-1 rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-1.5 text-xs text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)]"
                onClick={onClose}
              >
                <X className="h-3.5 w-3.5" />
                Close
              </button>
            </div>

            <div className="max-h-[calc(90vh-88px)] overflow-y-auto px-4 py-4 sm:px-5">
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
                                const isSelected =
                                  selectedVariant === model.variant;
                                const isIntent = intentVariant === model.variant;
                                const progressValue =
                                  downloadProgress[model.variant];
                                const progress =
                                  progressValue?.percent ??
                                  model.download_progress ??
                                  0;
                                const canSelect = canUseModel
                                  ? canUseModel(model.variant)
                                  : true;
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
                                              onClick={() =>
                                                onCancelDownload(model.variant)
                                              }
                                              className="flex items-center gap-1 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-2.5 py-1.5 text-xs font-medium text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)]"
                                            >
                                              <X className="h-3.5 w-3.5" />
                                              Cancel
                                            </button>
                                          )}

                                        {(model.status === "not_downloaded" ||
                                          model.status === "error") &&
                                          (requiresManualDownload(
                                            model.variant,
                                          ) ? (
                                            <button
                                              className="flex items-center gap-1.5 rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-muted)] disabled:cursor-not-allowed disabled:opacity-60"
                                              disabled
                                              title="Manual download required. See docs/user/manual-gemma-3-1b-download.md."
                                            >
                                              <Download className="h-3.5 w-3.5" />
                                              Manual download
                                            </button>
                                          ) : (
                                            <button
                                              onClick={() =>
                                                onDownload(model.variant)
                                              }
                                              className="flex items-center gap-1.5 rounded-md bg-[var(--accent-solid)] px-3 py-1.5 text-xs font-medium text-[var(--text-on-accent)] transition-opacity hover:opacity-90"
                                            >
                                              <Download className="h-3.5 w-3.5" />
                                              Download
                                            </button>
                                          ))}

                                        {model.status === "downloaded" && (
                                          <button
                                            onClick={() => onLoad(model.variant)}
                                            className="flex items-center gap-1.5 rounded-md bg-[var(--accent-solid)] px-3 py-1.5 text-xs font-medium text-[var(--text-on-accent)] transition-opacity hover:opacity-90"
                                          >
                                            <Play className="h-3.5 w-3.5" />
                                            Load
                                          </button>
                                        )}

                                        {model.status === "loading" && (
                                          <button
                                            className="flex items-center gap-1.5 rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-muted)]"
                                            disabled
                                          >
                                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                            Loading
                                          </button>
                                        )}

                                        {model.status === "ready" &&
                                          canSelect &&
                                          (isSelected ? (
                                            <button
                                              className="flex items-center gap-1.5 rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-muted)]"
                                              disabled
                                            >
                                              <CheckCircle2 className="h-3.5 w-3.5" />
                                              Selected
                                            </button>
                                          ) : (
                                            <button
                                              onClick={() => {
                                                onUseModel(model.variant);
                                                onClose();
                                              }}
                                              className="flex items-center gap-1.5 rounded-md bg-[var(--accent-solid)] px-3 py-1.5 text-xs font-medium text-[var(--text-on-accent)] transition-opacity hover:opacity-90"
                                            >
                                              <CheckCircle2 className="h-3.5 w-3.5" />
                                              Use model
                                            </button>
                                          ))}

                                        {model.status === "ready" && (
                                          <button
                                            onClick={() =>
                                              onUnload(model.variant)
                                            }
                                            className="flex items-center gap-1.5 rounded-md border border-[var(--border-strong)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-primary)] transition-colors hover:bg-[var(--bg-surface-3)]"
                                          >
                                            <Square className="h-3.5 w-3.5" />
                                            Unload
                                          </button>
                                        )}

                                        {(model.status === "downloaded" ||
                                          model.status === "ready") && (
                                          <button
                                            onClick={() =>
                                              setDeleteTargetVariant(
                                                model.variant,
                                              )
                                            }
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
          </motion.div>

          <AnimatePresence>
            {deleteTargetModel && (
              <motion.div
                className="fixed inset-0 z-[60] bg-black/75 p-4 backdrop-blur-sm"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setDeleteTargetVariant(null)}
              >
                <motion.div
                  initial={{ y: 10, opacity: 0, scale: 0.98 }}
                  animate={{ y: 0, opacity: 1, scale: 1 }}
                  exit={{ y: 10, opacity: 0, scale: 0.98 }}
                  transition={{ duration: 0.16 }}
                  className="mx-auto mt-[18vh] max-w-md rounded-xl border border-[var(--danger-border)] bg-[var(--bg-surface-1)] p-5"
                  onClick={(event) => event.stopPropagation()}
                >
                  <div className="flex items-start gap-3">
                    <div className="mt-0.5 rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2 text-[var(--danger-text)]">
                      <AlertTriangle className="h-4 w-4" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                        Delete model?
                      </h3>
                      <p className="mt-1 text-sm text-[var(--text-muted)]">
                        This removes
                        <span className="mx-1 font-medium text-[var(--text-primary)]">
                          {resolveModelLabel(deleteTargetModel.variant)}
                        </span>
                        from local storage.
                      </p>
                      <p className="mt-2 truncate text-xs text-[var(--text-subtle)]">
                        {deleteTargetModel.variant}
                      </p>
                    </div>
                  </div>

                  <div className="mt-5 flex items-center justify-end gap-2">
                    <button
                      onClick={() => setDeleteTargetVariant(null)}
                      className="rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-2)] px-3 py-1.5 text-xs font-medium text-[var(--text-secondary)] transition-colors hover:bg-[var(--bg-surface-3)]"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleConfirmDelete}
                      className="flex items-center gap-1.5 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-1.5 text-xs font-medium text-[var(--danger-text)] transition-colors hover:bg-[var(--danger-bg-hover)]"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                      Delete model
                    </button>
                  </div>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
