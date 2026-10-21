import clsx from "clsx";
import { Loader2 } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { Link, useInRouterContext } from "react-router";

import type { ModelInfo, ModelResidencySummary } from "@/api";
import { useOptionalModelCatalog } from "@/app/providers/ModelCatalogProvider";
import { ConfirmDeleteDialog } from "@/components/history/ConfirmDeleteDialog";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  compareQwenVersionRank,
  getModelDisplayName,
  getModelProviderLabel,
  PROVIDER_ORDER,
} from "@/features/models/catalog/modelMetadata";
import {
  formatModelBytes,
  ModelListRow,
  revealModelActions,
} from "@/features/models/components/ModelListRow";
import type { ModelDownloadProgressMap } from "@/features/models/downloadProgress";
import { useRuntimeHealth } from "@/shared/hooks/useRuntimeHealth";

interface RouteModelSection {
  key: string;
  title?: string;
  description?: string;
  models: ModelInfo[];
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
  residencySummary?: ModelResidencySummary | null;
  /** The page, for "Selected in Chat". */
  routeLabel?: string;
}

function providerRank(variant: string): number {
  const rank = PROVIDER_ORDER.indexOf(
    getModelProviderLabel(variant) as (typeof PROVIDER_ORDER)[number],
  );
  return rank === -1 ? Number.MAX_SAFE_INTEGER : rank;
}

/** Provider order, then Qwen by version (a smaller newer release does not
 * leapfrog an older larger one), then size. */
function compareModels(a: ModelInfo, b: ModelInfo): number {
  const providerDelta = providerRank(a.variant) - providerRank(b.variant);
  if (providerDelta !== 0) return providerDelta;
  const providerName = getModelProviderLabel(a.variant).localeCompare(
    getModelProviderLabel(b.variant),
  );
  if (providerName !== 0) return providerName;
  const versionDelta = compareQwenVersionRank(a.variant, b.variant);
  if (versionDelta !== 0) return versionDelta;
  const sizeDelta = (a.size_bytes ?? 0) - (b.size_bytes ?? 0);
  if (sizeDelta !== 0) return sizeDelta;
  return a.variant.localeCompare(b.variant);
}

const GIB = 1024 ** 3;

function formatMemory(bytes: number): string {
  const gigabytes = bytes / GIB;
  return `${gigabytes >= 10 ? gigabytes.toFixed(0) : gigabytes.toFixed(1)} GB`;
}

/** Memory in use, from the runtime; the residency counts when that's all. */
function MemoryInUse({
  readyCount,
  residencySummary,
}: {
  readyCount: number;
  residencySummary: ModelResidencySummary | null;
}) {
  const health = useRuntimeHealth(readyCount);
  const memory = health.status === "ready" ? health.health.runtime.memory : null;
  if (memory?.system_total_bytes) {
    return (
      <>
        Memory in use: {formatMemory(memory.reserved_bytes)} of{" "}
        {formatMemory(memory.system_total_bytes)}
      </>
    );
  }
  if (residencySummary) {
    return (
      <>
        {residencySummary.resident_count} model
        {residencySummary.resident_count === 1 ? "" : "s"} loaded
        {residencySummary.max_loaded_models
          ? ` of ${residencySummary.max_loaded_models}`
          : ""}
      </>
    );
  }
  return null;
}

/**
 * A page's models: download, load, use and unload them where they are
 * needed. Loaded models use memory, downloaded models use disk.
 */
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
  emptyMessage = "No models are available for this page.",
  sections,
  canUseModel,
  getModelLabel,
  selectionMode = "route",
  zIndexClassName = "z-50",
  residencySummary = null,
  routeLabel,
}: RouteModelModalProps) {
  const catalog = useOptionalModelCatalog();
  const inRouter = useInRouterContext();
  const [deleteTargetVariant, setDeleteTargetVariant] = useState<string | null>(
    null,
  );
  const returnFocusRef = useRef<HTMLElement | null>(null);

  const modalSections = useMemo(() => {
    const baseSections =
      sections && sections.length > 0 ? sections : [{ key: "models", models }];
    return baseSections.map((section) => ({
      ...section,
      models: [...section.models].sort(compareModels),
    }));
  }, [models, sections]);
  const orderedModels = modalSections.flatMap((section) => section.models);

  const labelFor = (variant: string): string =>
    getModelLabel
      ? getModelLabel(variant)
      : getModelDisplayName(
          variant,
          orderedModels.find((model) => model.variant === variant)?.display_name,
        );

  const deleteTarget = deleteTargetVariant
    ? (orderedModels.find((model) => model.variant === deleteTargetVariant) ??
      null)
    : null;
  const readyCount = models.filter((model) => model.status === "ready").length;

  const rowActions = {
    onDownload,
    onCancelDownload,
    onLoad,
    onUnload,
    onDelete: setDeleteTargetVariant,
    onSetPinned: catalog?.setModelPinned
      ? (variant: string, pinned: boolean) =>
          void catalog.setModelPinned(variant, pinned)
      : undefined,
    ...revealModelActions((message) => catalog?.reportError(message)),
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
          "flex max-h-[min(46rem,calc(100vh-3rem))] max-w-184 flex-col gap-0 overflow-hidden rounded-2xl border-(--border-strong) bg-(--bg-surface-0) p-0",
          zIndexClassName,
        )}
      >
        <header className="border-b border-(--border-subtle) px-6 py-5 pr-14">
          <DialogTitle className="text-xl font-semibold tracking-tight text-(--text-primary)">
            {title}
          </DialogTitle>
          <DialogDescription className="mt-1 text-sm text-(--text-muted)">
            {description}
          </DialogDescription>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto px-6 py-4">
          {loading ? (
            <div className="flex items-center gap-2 py-4 text-sm text-(--text-muted)">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading models…
            </div>
          ) : orderedModels.length === 0 ? (
            <p className="py-4 text-sm text-(--text-muted)">{emptyMessage}</p>
          ) : (
            <div className="space-y-5">
              {modalSections.map((section) => (
                <section key={section.key} aria-label={section.title}>
                  {section.title ? (
                    <div className="mb-2 px-0.5">
                      <h3 className="text-[13.5px] font-semibold text-(--text-primary)">
                        {section.title}
                      </h3>
                      {section.description ? (
                        <p className="mt-0.5 text-xs text-(--text-subtle)">
                          {section.description}
                        </p>
                      ) : null}
                    </div>
                  ) : null}
                  {section.models.length === 0 ? (
                    <p className="rounded-[14px] border border-(--border-subtle) px-4 py-3 text-xs text-(--text-subtle)">
                      No models in this group.
                    </p>
                  ) : (
                    <ul className="overflow-hidden rounded-[14px] border border-(--border-subtle) bg-(--bg-surface-1)">
                      {section.models.map((model) => {
                        const label = labelFor(model.variant);
                        const selectable =
                          selectionMode === "route" &&
                          (canUseModel ? canUseModel(model.variant) : true);
                        const isSelected =
                          selectionMode === "route" &&
                          selectedVariant === model.variant;
                        const subtitle = [
                          isSelected
                            ? routeLabel
                              ? `Selected in ${routeLabel}`
                              : "Selected"
                            : null,
                          model.status === "ready" && model.pinned
                            ? "Kept loaded"
                            : null,
                          getModelProviderLabel(model.variant),
                        ]
                          .filter(Boolean)
                          .join(" · ");
                        return (
                          <ModelListRow
                            key={model.variant}
                            model={model}
                            label={label}
                            subtitle={subtitle}
                            progress={
                              downloadProgress[model.variant]?.percent ??
                              model.download_progress ??
                              0
                            }
                            actions={rowActions}
                            highlighted={intentVariant === model.variant}
                            readyAction={
                              selectable && !isSelected
                                ? {
                                    label: "Use",
                                    onClick: () => {
                                      onUseModel(model.variant);
                                      onClose();
                                    },
                                  }
                                : null
                            }
                          />
                        );
                      })}
                    </ul>
                  )}
                </section>
              ))}
            </div>
          )}
        </div>

        <footer className="flex items-center justify-between gap-3 border-t border-(--border-subtle) px-6 py-3.5">
          <span
            data-testid="route-model-residency-summary"
            className="text-xs text-(--text-subtle)"
          >
            <MemoryInUse
              readyCount={readyCount}
              residencySummary={residencySummary}
            />
          </span>
          <div className="flex items-center gap-2">
            {inRouter ? (
              <Button asChild variant="ghost" size="sm" className="h-9">
                <Link to="/models" onClick={onClose}>
                  Open Models
                </Link>
              </Button>
            ) : null}
            <Button type="button" size="sm" className="h-9 px-4" onClick={onClose}>
              Done
            </Button>
          </div>
        </footer>

        <ConfirmDeleteDialog
          noun={deleteTarget ? "download" : null}
          itemLabel={deleteTarget ? labelFor(deleteTarget.variant) : undefined}
          description={
            deleteTarget
              ? `This removes ${deleteTarget.size_bytes ? `${formatModelBytes(deleteTarget.size_bytes)} of ` : ""}model files from this device, and stops any job using the model. Download it again to use it.`
              : ""
          }
          pending={false}
          error={null}
          onConfirm={() => {
            if (deleteTarget) {
              onDelete(deleteTarget.variant);
            }
            setDeleteTargetVariant(null);
          }}
          onCancel={() => setDeleteTargetVariant(null)}
        />
      </DialogContent>
    </Dialog>
  );
}
