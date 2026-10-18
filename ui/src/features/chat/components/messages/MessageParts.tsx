import {
  formatMessageDateTime,
  formatMessageTime,
} from "@/features/chat/playground/support";

export function DaySeparator({ label }: { label: string }) {
  return (
    <div
      role="separator"
      aria-label={label}
      className="flex items-center gap-3 text-[11.5px] text-[var(--text-subtle)] before:h-px before:flex-1 before:bg-[var(--border-subtle)] after:h-px after:flex-1 after:bg-[var(--border-subtle)]"
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
