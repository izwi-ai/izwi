import { useEffect, useRef, type FocusEvent } from "react";
import { motion, useIsPresent } from "framer-motion";
import {
  Check,
  CircleAlert,
  Info,
  TriangleAlert,
  X,
  type LucideIcon,
} from "lucide-react";

import type {
  NotificationAction,
  NotificationTone,
} from "@/app/providers/NotificationProvider";
import { cn } from "@/lib/utils";

/** One row in the stack: a single notification or a combined group. */
export interface NotificationView {
  key: string;
  ids: string[];
  /** Members with progress: closing the row hides their later updates. */
  progressIds: string[];
  title: string;
  description?: string;
  tone: NotificationTone;
  icon?: LucideIcon;
  /** 0–100. */
  progress?: number;
  detail?: string;
  action?: NotificationAction;
}

const TONE_ICON: Record<NotificationTone, { icon: LucideIcon; className: string }> = {
  info: { icon: Info, className: "text-[var(--text-muted)]" },
  success: { icon: Check, className: "text-[var(--status-positive-text)]" },
  warning: { icon: TriangleAlert, className: "text-[var(--status-warning-text)]" },
  danger: { icon: CircleAlert, className: "text-[var(--danger-text)]" },
};

const textButtonClass =
  "rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--border-strong)]";

interface NotificationItemProps {
  view: NotificationView;
  onClose: (view: NotificationView) => void;
  /** True while the pointer or focus is on this row; timers wait meanwhile. */
  onHeldChange: (held: boolean) => void;
}

export function NotificationItem({
  view,
  onClose,
  onHeldChange,
}: NotificationItemProps) {
  const heldRef = useRef({ pointer: false, focus: false });
  const isPresent = useIsPresent();

  const setHeld = (kind: "pointer" | "focus", value: boolean) => {
    const held = heldRef.current;
    const wasHeld = held.pointer || held.focus;
    held[kind] = value;
    const isHeld = held.pointer || held.focus;
    if (wasHeld !== isHeld) {
      onHeldChange(isHeld);
    }
  };

  // A row can close while held (its own dismiss button): release the hold.
  useEffect(() => {
    const held = heldRef.current;
    return () => {
      if (held.pointer || held.focus) {
        held.pointer = false;
        held.focus = false;
        onHeldChange(false);
      }
    };
  }, [onHeldChange]);

  const tone = TONE_ICON[view.tone];
  const Icon = view.icon ?? tone.icon;
  const assertive = view.tone === "danger";
  // Errors and warnings carry a reason the user needs: never cut it off.
  const fullText = view.tone === "danger" || view.tone === "warning";
  const hasProgress = view.progress !== undefined;
  const percent = hasProgress ? Math.round(view.progress ?? 0) : null;
  const detail = view.detail ?? (percent !== null ? `${percent}%` : undefined);

  return (
    <motion.div
      layout="position"
      data-state={isPresent ? "open" : "closed"}
      role={assertive ? "alert" : "status"}
      aria-live={assertive ? "assertive" : "polite"}
      aria-atomic="true"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.98, transition: { duration: 0.12 } }}
      transition={{ duration: 0.16 }}
      onPointerEnter={() => setHeld("pointer", true)}
      onPointerLeave={() => setHeld("pointer", false)}
      onFocus={() => setHeld("focus", true)}
      onBlur={(event: FocusEvent<HTMLDivElement>) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
          setHeld("focus", false);
        }
      }}
      className={cn(
        "flex flex-col gap-2 rounded-[12px] border border-[var(--border-default)] bg-[var(--surface-panel)] px-3.5 py-3 text-[var(--text-primary)] shadow-[var(--shadow-raised)]",
        isPresent ? "pointer-events-auto" : "pointer-events-none",
      )}
    >
      <div
        className={cn(
          "grid grid-cols-[auto_minmax(0,1fr)_auto] gap-3",
          fullText ? "items-start" : "items-center",
        )}
      >
        <Icon
          aria-hidden="true"
          strokeWidth={view.tone === "success" && !view.icon ? 2.5 : 2}
          className={cn("h-4 w-4 shrink-0", fullText && "mt-0.5", tone.className)}
        />
        <span className="flex min-w-0 flex-col">
          <span
            title={fullText ? undefined : view.title}
            className={cn(
              "text-[13.5px] font-medium leading-5",
              fullText ? "break-words" : "truncate",
            )}
          >
            {view.title}
          </span>
          {view.description ? (
            <span
              title={fullText ? undefined : view.description}
              className={cn(
                "text-[12.5px] leading-[1.45] text-[var(--text-muted)]",
                fullText ? "whitespace-pre-line break-words" : "truncate",
              )}
            >
              {view.description}
            </span>
          ) : null}
        </span>
        <span className={cn("flex items-center gap-1.5", fullText && "-mt-0.5")}>
          {detail ? (
            <span
              // The progress bar reports this to assistive tech without re-announcing each tick.
              aria-hidden={hasProgress || undefined}
              className="whitespace-nowrap pl-1 text-[12.5px] tabular-nums text-[var(--text-muted)]"
            >
              {detail}
            </span>
          ) : null}
          {view.action ? (
            <button
              type="button"
              onClick={() => {
                view.action?.onClick();
                onClose(view);
              }}
              className={cn(
                textButtonClass,
                "whitespace-nowrap px-1 text-[13px] font-medium underline decoration-[var(--border-strong)] underline-offset-[3px] transition-colors hover:decoration-[var(--text-primary)]",
              )}
            >
              {view.action.label}
            </button>
          ) : null}
          <button
            type="button"
            onClick={() => onClose(view)}
            aria-label={`Dismiss ${view.title} notification`}
            className={cn(
              textButtonClass,
              "flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-[var(--text-subtle)] transition-colors hover:bg-[var(--accent-soft)] hover:text-[var(--text-primary)]",
            )}
          >
            <X aria-hidden="true" className="h-3.5 w-3.5" />
          </button>
        </span>
      </div>
      {percent !== null ? (
        <div
          role="progressbar"
          aria-label={view.title}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
          aria-valuetext={detail}
          className="ml-7 h-[3px] overflow-hidden rounded-full bg-[var(--bg-surface-3)]"
        >
          <div
            className="h-full rounded-full bg-[var(--accent-solid)] transition-[width] duration-300"
            style={{ width: `${percent}%` }}
          />
        </div>
      ) : null}
    </motion.div>
  );
}
