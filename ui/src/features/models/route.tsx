import { useMemo, useState } from "react";
import { Loader2, RefreshCw, Trash2 } from "lucide-react";
import type { ModelInfo, RemovedModelInfo } from "@/api";
import { ConfirmDeleteDialog } from "@/components/history/ConfirmDeleteDialog";
import {
  HistoryToolbar,
  SearchField,
  SegmentedFilter,
} from "@/components/history/HistoryToolbar";
import { PageHeader, PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { useOptionalModelCatalog } from "@/app/providers/ModelCatalogProvider";
import {
  ModelListRow,
  revealModelActions,
} from "@/features/models/components/ModelListRow";
import type { ModelDownloadProgressMap } from "@/features/models/downloadProgress";
import {
  compareQwenVersionRank,
  getModelProviderLabel,
  MODEL_DETAILS,
  getModelDisplayName,
  resolveModelDetail,
  PROVIDER_ORDER,
} from "@/features/models/catalog/modelMetadata";
import {
  MODEL_TASK_GROUPS,
  modelTaskGroupId,
} from "@/features/models/taskGroups";
import { useRuntimeHealth } from "@/shared/hooks/useRuntimeHealth";
import { cn } from "@/lib/utils";

interface MyModelsPageProps {
  models: ModelInfo[];
  /** Downloads of removed variants still on disk; only deletion applies. */
  removedModels?: RemovedModelInfo[];
  loading: boolean;
  downloadProgress: ModelDownloadProgressMap;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  /** "Keep loaded" on or off. */
  onSetPinned?: (variant: string, pinned: boolean) => void;
  onRefresh: () => void;
}

type FilterType = "all" | "loaded" | "downloaded" | "available";
type GroupBy = "task" | "provider";
interface ModelSection {
  id: string;
  label: string;
  icon?: React.ComponentType<{ className?: string }>;
  usedBy?: string;
  models: ModelInfo[];
}

function parseSize(sizeStr: string): number {
  const match = sizeStr.match(/^([\d.]+)\s*(GB|MB|KB|B)?$/i);
  if (!match) return 0;
  const value = parseFloat(match[1]);
  const unit = (match[2] || "B").toUpperCase();
  const multipliers: Record<string, number> = {
    B: 1,
    KB: 1024,
    MB: 1024 * 1024,
    GB: 1024 * 1024 * 1024,
  };
  return value * (multipliers[unit] || 1);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024)
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function getModelSizeBytes(model: ModelInfo): number {
  if (model.size_bytes !== null && model.size_bytes > 0) {
    return model.size_bytes;
  }
  const fallback = MODEL_DETAILS[model.variant]?.size;
  if (fallback) {
    return parseSize(fallback);
  }
  return model.estimated_size_bytes ?? 0;
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
  const normalizedLeftRank =
    leftRank === -1 ? Number.MAX_SAFE_INTEGER : leftRank;
  const normalizedRightRank =
    rightRank === -1 ? Number.MAX_SAFE_INTEGER : rightRank;
  if (normalizedLeftRank !== normalizedRightRank) {
    return normalizedLeftRank - normalizedRightRank;
  }
  return left.localeCompare(right);
}

function isOnDisk(model: ModelInfo): boolean {
  return model.status === "downloaded" || model.status === "ready";
}

function SummaryCard({
  label,
  value,
  detail,
  meter,
}: {
  label: string;
  value: React.ReactNode;
  detail: React.ReactNode;
  meter?: number | null;
}) {
  return (
    <div className="flex min-w-0 flex-col gap-1 rounded-[14px] border border-(--border-subtle) bg-(--surface-panel) px-4 py-3.5">
      <div className="text-xs text-(--text-subtle)">{label}</div>
      <div className="truncate text-[15px] text-(--text-muted)">{value}</div>
      {meter != null ? (
        <div
          role="meter"
          aria-label={label}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(meter * 100)}
          className="mt-1.5 h-1 overflow-hidden rounded-full bg-(--surface-raised)"
        >
          <div
            className="h-full rounded-full bg-(--status-positive-solid)"
            style={{ width: `${Math.max(meter * 100, 2)}%` }}
          />
        </div>
      ) : (
        <div className="truncate text-xs text-(--text-subtle)">{detail}</div>
      )}
    </div>
  );
}

export function MyModelsPage({
  models,
  removedModels = [],
  loading,
  downloadProgress,
  onDownload,
  onCancelDownload,
  onLoad,
  onUnload,
  onDelete,
  onSetPinned,
  onRefresh,
}: MyModelsPageProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<FilterType>("all");
  const [groupBy, setGroupBy] = useState<GroupBy>("task");
  const [deleteModalVariant, setDeleteModalVariant] = useState<string | null>(
    null,
  );
  const [isRefreshing, setIsRefreshing] = useState(false);
  const catalog = useOptionalModelCatalog();
  const revealActions = revealModelActions((message) =>
    catalog?.reportError(message),
  );

  const visibleModels = useMemo(
    () => models.filter((m) => !m.variant.includes("Tokenizer")),
    [models],
  );
  const stats = useMemo(() => {
    const loaded = visibleModels.filter((m) => m.status === "ready");
    return {
      total: visibleModels.length,
      loaded,
      downloaded: visibleModels.filter(isOnDisk).length,
      available: visibleModels.filter((m) => m.status === "not_downloaded")
        .length,
      totalSize: visibleModels
        .filter(isOnDisk)
        .reduce((acc, m) => acc + getModelSizeBytes(m), 0),
    };
  }, [visibleModels]);
  const health = useRuntimeHealth(stats.loaded.length);

  const filteredModels = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    return visibleModels
      .filter((m) => {
        if (query) {
          const details = resolveModelDetail(m);
          const matchesSearch = [
            details.shortName,
            details.fullName,
            details.description,
            getModelDisplayName(m.variant, m.display_name),
            getProviderLabel(m.variant),
            ...details.capabilities,
          ].some((value) => value.toLowerCase().includes(query));
          if (!matchesSearch) return false;
        }
        switch (statusFilter) {
          case "loaded":
            return m.status === "ready";
          case "downloaded":
            return isOnDisk(m);
          case "available":
            return m.status === "not_downloaded";
          default:
            return true;
        }
      })
      .sort((a, b) => {
        // Stable sort independent of status so rows do not jump while
        // downloading/loading. Qwen variants follow version number first
        // (Qwen3 → 3.5 → 3.6 → …) so a smaller newer release does not
        // leapfrog an older larger one; everything else stays size-ordered.
        const versionDelta = compareQwenVersionRank(a.variant, b.variant);
        if (versionDelta !== 0) return versionDelta;
        const sizeA = getModelSizeBytes(a);
        const sizeB = getModelSizeBytes(b);
        if (sizeA !== sizeB) return sizeA - sizeB;
        return a.variant.localeCompare(b.variant);
      });
  }, [visibleModels, searchQuery, statusFilter]);

  const sections = useMemo<ModelSection[]>(() => {
    if (groupBy === "task") {
      return MODEL_TASK_GROUPS.map((group) => ({
        id: group.id,
        label: group.label,
        icon: group.icon,
        usedBy: group.usedBy,
        models: filteredModels.filter((m) => modelTaskGroupId(m) === group.id),
      })).filter((section) => section.models.length > 0);
    }
    const grouped = new Map<string, ModelInfo[]>();
    for (const model of filteredModels) {
      const provider = getProviderLabel(model.variant);
      grouped.set(provider, [...(grouped.get(provider) ?? []), model]);
    }
    return Array.from(grouped.entries())
      .sort(([left], [right]) => compareProviders(left, right))
      .map(([provider, groupedModels]) => ({
        id: provider,
        label: provider,
        models: groupedModels,
      }));
  }, [filteredModels, groupBy]);

  const deleteTargetDetails =
    deleteModalVariant && MODEL_DETAILS[deleteModalVariant]
      ? MODEL_DETAILS[deleteModalVariant]
      : null;
  const deleteTargetName = deleteModalVariant
    ? deleteTargetDetails
      ? getModelDisplayName(deleteModalVariant)
      : (removedModels.find((model) => model.variant === deleteModalVariant)
          ?.display_name ??
        models.find((model) => model.variant === deleteModalVariant)
          ?.display_name ??
        deleteModalVariant)
    : undefined;

  if (loading) {
    return (
      <PageShell>
        <div className="flex items-center justify-center gap-2 py-24 text-(--text-muted)">
          <Loader2 className="w-4 h-4 animate-spin" />
          <p className="text-sm">Loading models...</p>
        </div>
      </PageShell>
    );
  }

  const memory =
    health.status === "ready" ? health.health.runtime.memory : null;
  const memoryFraction =
    memory?.system_total_bytes && memory.system_total_bytes > 0
      ? Math.min(memory.reserved_bytes / memory.system_total_bytes, 1)
      : null;
  const hasActiveFilters =
    searchQuery.trim().length > 0 || statusFilter !== "all";

  return (
    <PageShell>
      <PageHeader
        title="Models"
        description="Download, load and manage local models. Loaded models use memory; downloaded models use disk."
        actions={
          onRefresh ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-9 gap-2"
              onClick={async () => {
                setIsRefreshing(true);
                await onRefresh();
                setIsRefreshing(false);
              }}
              disabled={isRefreshing}
            >
              <RefreshCw
                className={cn("h-4 w-4", isRefreshing && "animate-spin")}
              />
              Refresh
            </Button>
          ) : null
        }
      />

      <div className="mb-4 grid gap-3 md:grid-cols-3">
        <SummaryCard
          label="Memory in use by models"
          value={
            memory ? (
              <>
                <span className="text-xl font-semibold text-(--text-primary)">
                  {memory.reserved_bytes > 0
                    ? formatBytes(memory.reserved_bytes)
                    : "None"}
                </span>
                {memory.system_total_bytes ? (
                  <span className="text-[13px]">
                    {" "}
                    of {formatBytes(memory.system_total_bytes)}
                    {memory.system_available_bytes != null
                      ? ` · ${formatBytes(memory.system_available_bytes)} free`
                      : ""}
                  </span>
                ) : null}
              </>
            ) : (
              <span className="text-xl font-semibold text-(--text-primary)">
                —
              </span>
            )
          }
          detail="Memory figures are unavailable."
          meter={memoryFraction}
        />
        <SummaryCard
          label="Disk used"
          value={
            <span className="text-xl font-semibold text-(--text-primary)">
              {formatBytes(stats.totalSize)}
            </span>
          }
          detail={`${stats.downloaded} model${stats.downloaded === 1 ? "" : "s"} downloaded`}
        />
        <SummaryCard
          label="Loaded"
          value={
            <>
              <span className="text-xl font-semibold text-(--text-primary)">
                {stats.loaded.length}
              </span>
              {stats.loaded.length > 0 ? (
                <span className="text-[13px]">
                  {" · "}
                  {stats.loaded
                    .map((m) => getModelDisplayName(m.variant, m.display_name))
                    .join(", ")}
                </span>
              ) : null}
            </>
          }
          detail={
            stats.loaded.some((m) => m.pinned)
              ? "Models you keep loaded stay until you unload them."
              : stats.loaded.length > 0
                ? "Loaded models make room when another model needs it."
                : "No model loaded."
          }
        />
      </div>

      <HistoryToolbar>
        <SearchField
          value={searchQuery}
          onChange={setSearchQuery}
          placeholder="Search models…"
        />
        <SegmentedFilter
          label="Show"
          options={[
            { id: "all", label: "All", count: stats.total },
            { id: "loaded", label: "Loaded", count: stats.loaded.length },
            { id: "downloaded", label: "Downloaded", count: stats.downloaded },
            { id: "available", label: "Available", count: stats.available },
          ]}
          value={statusFilter}
          onChange={setStatusFilter}
        />
        <label className="ml-auto flex items-center gap-2 text-[12.5px] text-(--text-subtle)">
          Group by
          <select
            value={groupBy}
            onChange={(event) => setGroupBy(event.target.value as GroupBy)}
            className="h-[30px] rounded-[8px] border border-(--border-default) bg-(--surface-panel) px-2 text-[12.5px] text-(--text-secondary) outline-hidden"
          >
            <option value="task">Task</option>
            <option value="provider">Provider</option>
          </select>
        </label>
      </HistoryToolbar>

      {filteredModels.length === 0 ? (
        <div className="rounded-[14px] border border-(--border-subtle) bg-(--surface-panel) px-6 py-12 text-center">
          <h3 className="text-base font-semibold text-(--text-primary)">
            No models found
          </h3>
          <p className="mt-1.5 text-sm text-(--text-muted)">
            {hasActiveFilters
              ? "Try a different search or filter."
              : "Download a model to get started."}
          </p>
        </div>
      ) : (
        <div className="space-y-5">
          {sections.map((section) => (
            <section key={section.id} aria-label={section.label}>
              <div className="mb-2 flex items-center gap-2 px-0.5">
                {section.icon ? (
                  <section.icon className="h-4 w-4 text-(--text-subtle)" />
                ) : null}
                <h3 className="text-[13.5px] font-semibold text-(--text-primary)">
                  {section.label}
                </h3>
                <span className="text-xs text-(--text-subtle)">
                  {section.models.length}
                </span>
                {section.usedBy ? (
                  <span className="ml-auto text-xs text-(--text-subtle)">
                    Used by {section.usedBy}
                  </span>
                ) : null}
              </div>
              <ul className="overflow-hidden rounded-[14px] border border-(--border-subtle) bg-(--surface-panel)">
                {section.models.map((model) => {
                  const details = resolveModelDetail(model);
                  const displayName = getModelDisplayName(
                    model.variant,
                    model.display_name,
                  );
                  const progress =
                    downloadProgress[model.variant]?.percent ??
                    model.download_progress ??
                    0;
                  return (
                    <ModelListRow
                      key={model.variant}
                      model={model}
                      label={displayName}
                      subtitle={[
                        details.description || null,
                        model.status === "ready" && model.pinned
                          ? "Kept loaded"
                          : null,
                        getProviderLabel(model.variant),
                      ]
                        .filter(Boolean)
                        .join(" · ")}
                      progress={progress}
                      actions={{
                        onDownload,
                        onCancelDownload,
                        onLoad,
                        onUnload,
                        onDelete: setDeleteModalVariant,
                        onSetPinned,
                        ...revealActions,
                      }}
                    />
                  );
                })}
              </ul>
            </section>
          ))}
        </div>
      )}

      {removedModels.length > 0 && (
        <section
          aria-label="Removed models"
          className="mt-5 space-y-2 rounded-[14px] border border-(--status-warning-border) bg-(--surface-panel) p-4"
        >
          <div>
            <h3 className="text-sm font-semibold text-(--text-primary)">
              Removed models still on disk
            </h3>
            <p className="mt-1 text-xs text-(--text-muted)">
              Izwi no longer runs these downloads and never deletes them for
              you. Delete them to reclaim the space.
            </p>
          </div>
          <ul className="divide-y divide-(--border-subtle)">
            {removedModels.map((model) => (
              <li
                key={model.variant}
                className="flex flex-col gap-2 py-2 sm:flex-row sm:items-center sm:justify-between"
              >
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-(--text-primary)">
                    {model.display_name}
                  </p>
                  <p className="text-xs text-(--text-muted)">
                    {formatBytes(model.size_bytes)} · use {model.replacement}{" "}
                    instead
                  </p>
                  <p className="text-xs text-(--text-subtle)">
                    {model.reason}
                  </p>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => setDeleteModalVariant(model.variant)}
                  className="h-8 shrink-0 gap-1.5 self-start text-(--danger-text) sm:self-auto"
                  aria-label={`Delete ${model.display_name}`}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  Delete
                </Button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <ConfirmDeleteDialog
        noun={deleteModalVariant ? "model" : null}
        itemLabel={deleteTargetName}
        description="This removes the downloaded model files from local storage."
        pending={false}
        error={null}
        onConfirm={() => {
          if (deleteModalVariant) {
            onDelete(deleteModalVariant);
          }
          setDeleteModalVariant(null);
        }}
        onCancel={() => setDeleteModalVariant(null)}
      />
    </PageShell>
  );
}
