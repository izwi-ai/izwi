import { cn } from "@/lib/utils";
import { APP_VERSION } from "@/shared/config/runtime";
import { useRuntimeHealth } from "@/shared/hooks/useRuntimeHealth";

const GIB = 1024 ** 3;

const BACKEND_LABELS: Record<string, string> = {
  cpu: "CPU",
  cuda: "CUDA",
  metal: "Metal",
};

function formatGigabytes(bytes: number): string {
  const gigabytes = bytes / GIB;
  return `${gigabytes >= 10 ? gigabytes.toFixed(0) : gigabytes.toFixed(1)} GB`;
}

interface RuntimeCardProps {
  readyModelsCount: number;
  catalogError: string | null;
  collapsed: boolean;
}

/** Sidebar summary of the local runtime: backend, memory and loaded models. */
export function RuntimeCard({
  readyModelsCount,
  catalogError,
  collapsed,
}: RuntimeCardProps) {
  // Loading or unloading a model changes reservations: refresh at once.
  const state = useRuntimeHealth(readyModelsCount);

  const offline = state.status === "error" || catalogError !== null;
  const health = state.status === "ready" ? state.health : null;
  const memory = health?.runtime.memory ?? null;
  const backend = health
    ? (BACKEND_LABELS[health.runtime.selected_backend] ??
      health.runtime.selected_backend)
    : null;
  const usedFraction =
    memory?.system_total_bytes && memory.system_total_bytes > 0
      ? Math.min(1, memory.reserved_bytes / memory.system_total_bytes)
      : null;

  const statusLabel = offline
    ? "Runtime offline"
    : state.status === "loading"
      ? "Connecting…"
      : "Runtime ready";
  const modelsLabel = `${readyModelsCount} model${readyModelsCount === 1 ? "" : "s"}`;
  const memoryLabel =
    memory && memory.reserved_bytes > 0
      ? formatGigabytes(memory.reserved_bytes)
      : null;
  const memoryTitle =
    memory && memory.system_total_bytes
      ? `${formatGigabytes(memory.reserved_bytes)} of ${formatGigabytes(memory.system_total_bytes)} reserved`
      : undefined;

  const dot = (
    <span
      aria-hidden="true"
      className={cn(
        "h-[7px] w-[7px] shrink-0 rounded-full",
        offline
          ? "bg-(--danger-text)"
          : state.status === "ready"
            ? "bg-(--status-positive-solid) shadow-[0_0_0_3px_var(--status-positive-bg)]"
            : "bg-(--text-subtle)",
      )}
    />
  );

  if (collapsed) {
    return (
      <div
        className="flex justify-center py-1"
        role="status"
        aria-label={`${statusLabel}, ${modelsLabel} loaded`}
        title={`${statusLabel} · ${modelsLabel} loaded`}
        data-testid="runtime-card"
      >
        {dot}
      </div>
    );
  }

  return (
    <div
      role="status"
      data-testid="runtime-card"
      className="flex flex-col gap-2 rounded-xl border border-(--border-subtle) bg-(--surface-panel) px-3 py-2.5"
    >
      <div className="flex items-center gap-2 text-[12.5px] text-(--text-primary)">
        {dot}
        <span className="truncate">{statusLabel}</span>
        {backend && !offline ? (
          <span className="ml-auto text-[11.5px] text-(--text-subtle)">
            {backend}
          </span>
        ) : null}
      </div>
      {usedFraction !== null && !offline ? (
        <div
          className="h-1 overflow-hidden rounded-full bg-(--surface-raised)"
          title={memoryTitle}
          role="meter"
          aria-label="Memory reserved by models"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(usedFraction * 100)}
        >
          <div
            className="h-full rounded-full bg-(--text-muted)"
            style={{ width: `${Math.max(usedFraction * 100, 2)}%` }}
          />
        </div>
      ) : null}
      <div className="flex justify-between gap-2 text-[11px] text-(--text-subtle)">
        <span className="truncate">
          {modelsLabel}
          {memoryLabel ? ` · ${memoryLabel}` : ""}
        </span>
        <span title={`App version ${APP_VERSION}`}>v{APP_VERSION}</span>
      </div>
    </div>
  );
}
