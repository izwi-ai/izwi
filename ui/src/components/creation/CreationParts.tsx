import type { ReactNode } from "react";
import { Loader2 } from "lucide-react";

import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";

/** Body of a creation dialog: the work on the left, settings on the right. */
export function CreationBody({
  main,
  side,
}: {
  main: ReactNode;
  side: ReactNode;
}) {
  return (
    <div className="grid lg:grid-cols-[minmax(0,1fr)_19rem]">
      <div className="flex min-w-0 flex-col gap-3 px-5 py-4 sm:px-6">{main}</div>
      <div className="flex flex-col gap-4 border-t border-[var(--border-subtle)] bg-[var(--surface-panel-alt)] px-5 py-4 lg:border-l lg:border-t-0">
        {side}
      </div>
    </div>
  );
}

/** A labelled setting in the side panel. */
export function CreationField({
  label,
  htmlFor,
  hint,
  children,
}: {
  label: ReactNode;
  htmlFor?: string;
  hint?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label
        htmlFor={htmlFor}
        className="text-[12.5px] font-medium text-[var(--text-muted)]"
      >
        {label}
      </label>
      {children}
      {hint ? (
        <p className="text-xs leading-4 text-[var(--text-subtle)]">{hint}</p>
      ) : null}
    </div>
  );
}

/** A switch with a title and one line of explanation. */
export function CreationOption({
  title,
  description,
  checked,
  onCheckedChange,
  disabled = false,
}: {
  title: ReactNode;
  description?: ReactNode;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label
      className={cn(
        "flex items-start gap-3",
        disabled ? "cursor-not-allowed" : "cursor-pointer",
      )}
    >
      <span className="min-w-0 flex-1">
        <span className="block text-[13.5px] font-medium text-[var(--text-primary)]">
          {title}
        </span>
        {description ? (
          <span className="mt-0.5 block text-xs leading-4 text-[var(--text-subtle)]">
            {description}
          </span>
        ) : null}
      </span>
      <Switch
        className="mt-0.5"
        checked={checked}
        onCheckedChange={onCheckedChange}
        disabled={disabled}
      />
    </label>
  );
}

export type ModelRowStatus = "ready" | "busy" | "not_ready" | "missing";

/** One model's readiness: a status dot, its name, and what to do next. */
export function ModelRow({
  name,
  status,
  statusLabel,
  actions,
}: {
  name: ReactNode;
  status: ModelRowStatus;
  statusLabel: string;
  actions?: ReactNode;
}) {
  return (
    <div className="flex items-center gap-2 rounded-[10px] border border-[var(--border-subtle)] bg-[var(--surface-panel)] px-2.5 py-2 text-[13px]">
      {status === "busy" ? (
        <Loader2
          aria-hidden="true"
          className="h-3 w-3 shrink-0 animate-spin text-[var(--status-warning-text)]"
        />
      ) : (
        <span
          aria-hidden="true"
          className={cn(
            "h-1.5 w-1.5 shrink-0 rounded-full",
            status === "ready"
              ? "bg-[var(--status-positive-solid)]"
              : status === "not_ready"
                ? "bg-[var(--status-warning-text)]"
                : "bg-[var(--text-subtle)]",
          )}
        />
      )}
      <span className="min-w-0 flex-1">
        <span className="block truncate text-[var(--text-primary)]">{name}</span>
        <span className="block text-[11.5px] text-[var(--text-subtle)]">
          {statusLabel}
        </span>
      </span>
      {actions ? (
        <span className="flex shrink-0 items-center gap-1">{actions}</span>
      ) : null}
    </div>
  );
}

/** Small text button for ModelRow actions ("Load", "Change"). */
export function ModelRowAction({
  children,
  onClick,
  disabled,
  busy,
  "aria-label": ariaLabel,
}: {
  children: ReactNode;
  onClick: () => void;
  disabled?: boolean;
  busy?: boolean;
  "aria-label"?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      className="inline-flex h-7 items-center gap-1 rounded-md px-1.5 text-[12.5px] text-[var(--text-muted)] underline-offset-[3px] hover:text-[var(--text-primary)] hover:underline disabled:cursor-not-allowed disabled:opacity-50"
    >
      {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : null}
      {children}
    </button>
  );
}
