import type { ReactNode } from "react";
import { AlertTriangle, Square } from "lucide-react";

import {
  formatMessageDateTime,
  formatMessageTime,
} from "@/features/chat/playground/support";
import { cn } from "@/lib/utils";

export function DaySeparator({ label }: { label: string }) {
  return (
    <div
      role="separator"
      aria-label={label}
      className="flex items-center gap-3 text-[11.5px] text-(--text-subtle) before:h-px before:flex-1 before:bg-(--border-subtle) after:h-px after:flex-1 after:bg-(--border-subtle)"
    >
      {label}
    </div>
  );
}

export function MessageTime({ timestamp }: { timestamp: number }) {
  return (
    <time
      dateTime={new Date(timestamp).toISOString()}
      title={formatMessageDateTime(timestamp)}
      className="px-1.5 tabular-nums"
    >
      {formatMessageTime(timestamp)}
    </time>
  );
}

/** Marks a reply the user stopped before it finished. */
export function StoppedChip() {
  return (
    <span className="inline-flex h-[22px] items-center gap-1.5 self-start rounded-full border border-(--border-default) px-2 text-[11.5px] text-(--text-muted)">
      <Square className="h-2.5 w-2.5 fill-current" aria-hidden="true" />
      Stopped
    </span>
  );
}

/** Inline note under a reply that ended early. */
export function ReplyNotice({
  tone,
  children,
}: {
  tone: "warning" | "danger";
  children: ReactNode;
}) {
  return (
    <p
      role="note"
      className={cn(
        "flex items-center gap-2 rounded-[10px] border px-3 py-2 text-[12.5px] leading-snug",
        tone === "warning"
          ? "border-(--status-warning-border) bg-(--status-warning-bg) text-(--status-warning-text)"
          : "border-(--danger-border) bg-(--danger-bg) text-(--danger-text)",
      )}
    >
      <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
      {children}
    </p>
  );
}
