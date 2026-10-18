import { useMemo, useState } from "react";
import { ImageIcon, Loader2, SquarePen, Trash2 } from "lucide-react";

import type { ChatThread } from "@/api";
import { SearchField } from "@/components/history/HistoryToolbar";
import { Button } from "@/components/ui/button";
import { displayThreadTitle } from "@/features/chat/playground/support";
import { cn } from "@/lib/utils";

const DAY_MS = 24 * 60 * 60 * 1000;

function startOfDay(date: Date): number {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}

/** "Today", "Yesterday", "Previous 7 days", "Previous 30 days", then months. */
export function threadDateGroup(updatedAt: number, now = new Date()): string {
  const days = Math.floor((startOfDay(now) - startOfDay(new Date(updatedAt))) / DAY_MS);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return "Previous 7 days";
  if (days < 30) return "Previous 30 days";
  const date = new Date(updatedAt);
  return date.toLocaleDateString([], {
    month: "long",
    ...(date.getFullYear() === now.getFullYear() ? {} : { year: "numeric" }),
  });
}

interface ChatThreadListProps {
  threads: ChatThread[];
  activeThreadId: string | null;
  loading: boolean;
  /** Streaming or starting a thread: switching is paused. */
  busy: boolean;
  onSelect: (threadId: string) => void;
  onDelete: (threadId: string) => void;
  onNewChat: () => void;
  className?: string;
}

/** New chat, search, and the threads grouped by when they were last used. */
export function ChatThreadList({
  threads,
  activeThreadId,
  loading,
  busy,
  onSelect,
  onDelete,
  onNewChat,
  className,
}: ChatThreadListProps) {
  const [search, setSearch] = useState("");

  const groups = useMemo(() => {
    const term = search.trim().toLowerCase();
    const visible = term
      ? threads.filter((thread) =>
          [displayThreadTitle(thread.title), thread.last_message_preview ?? ""]
            .some((value) => value.toLowerCase().includes(term)),
        )
      : threads;
    const byGroup = new Map<string, ChatThread[]>();
    for (const thread of [...visible].sort((a, b) => b.updated_at - a.updated_at)) {
      const group = threadDateGroup(thread.updated_at);
      byGroup.set(group, [...(byGroup.get(group) ?? []), thread]);
    }
    return Array.from(byGroup.entries());
  }, [search, threads]);

  return (
    <div className={cn("flex min-h-0 flex-col gap-2.5", className)}>
      <Button
        type="button"
        size="sm"
        className="h-8 w-full gap-2"
        onClick={onNewChat}
        disabled={busy}
      >
        <SquarePen className="h-4 w-4" />
        New chat
      </Button>
      <SearchField
        value={search}
        onChange={setSearch}
        placeholder="Search chats…"
        className="h-8 sm:w-full"
      />

      <nav
        aria-label="Chats"
        className="-mx-1 min-h-0 flex-1 overflow-y-auto px-1 pb-2 scrollbar-thin"
      >
        {loading ? (
          <div className="flex items-center gap-2 px-2 py-3 text-xs text-[var(--text-muted)]">
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
            Loading chats…
          </div>
        ) : groups.length === 0 ? (
          <p className="px-2 py-3 text-xs text-[var(--text-muted)]">
            {search.trim() ? "No chats match." : "No chats yet. Send a message to start one."}
          </p>
        ) : (
          groups.map(([group, groupThreads]) => (
            <div key={group} className="mt-2 first:mt-0">
              <h3 className="px-2 pb-1 pt-1.5 text-[11px] font-medium text-[var(--text-subtle)]">
                {group}
              </h3>
              <ul className="space-y-0.5">
                {groupThreads.map((thread) => {
                  const title = displayThreadTitle(thread.title);
                  const active = thread.id === activeThreadId;
                  return (
                    <li key={thread.id} className="group relative">
                      <button
                        type="button"
                        onClick={() => onSelect(thread.id)}
                        disabled={busy && !active}
                        aria-current={active ? "page" : undefined}
                        title={title}
                        className={cn(
                          "flex h-8 w-full items-center gap-2 rounded-lg pl-2.5 pr-8 text-left text-[13.5px] transition-colors disabled:opacity-60",
                          active
                            ? "bg-[var(--sidebar-link-active-bg)] text-[var(--text-primary)]"
                            : "text-[var(--sidebar-link-title)] hover:bg-[var(--sidebar-link-hover-bg)] hover:text-[var(--text-primary)]",
                        )}
                      >
                        {thread.has_images ? (
                          <ImageIcon
                            aria-hidden="true"
                            className="h-3.5 w-3.5 shrink-0 text-[var(--text-subtle)]"
                          />
                        ) : null}
                        <span className="truncate">{title}</span>
                        {thread.has_images ? (
                          <span className="sr-only"> (has images)</span>
                        ) : null}
                      </button>
                      <button
                        type="button"
                        onClick={() => onDelete(thread.id)}
                        disabled={busy}
                        aria-label={`Delete ${title}`}
                        title="Delete chat"
                        className="absolute right-1 top-1/2 inline-flex h-6 w-6 -translate-y-1/2 items-center justify-center rounded-md text-[var(--text-subtle)] opacity-0 transition-opacity hover:bg-[var(--bg-surface-2)] hover:text-[var(--danger-text)] focus-visible:opacity-100 group-hover:opacity-100 disabled:hidden"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))
        )}
      </nav>
    </div>
  );
}
