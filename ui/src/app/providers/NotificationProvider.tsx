import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import { AnimatePresence } from "framer-motion";
import type { LucideIcon } from "lucide-react";

import {
  NotificationItem,
  type NotificationView,
} from "@/app/providers/NotificationItem";
import { createUuid } from "@/lib/ids";

export type NotificationTone = "info" | "success" | "warning" | "danger";

export interface NotificationAction {
  label: string;
  /** Runs, then the notification closes. */
  onClick: () => void;
}

/**
 * Notifications that share a group key show as one row while two or more of
 * them are on screen: the group's title, and the members' combined progress.
 * A combined row has no action; dismissing it dismisses every member.
 */
export interface NotificationGroup {
  key: string;
  /** Title of the combined row, e.g. (count) => `Downloading ${count} models`. */
  title: (count: number) => string;
}

export interface NotificationInput {
  title: string;
  /** Second line. Errors and warnings show it in full: pass the server's reason here. */
  description?: string;
  tone?: NotificationTone;
  /**
   * Auto-dismiss delay. By default success and info close after 5s (8s with an
   * action), warnings after 8s; errors and notifications with progress stay
   * until dismissed. `Infinity` keeps any notification open.
   */
  durationMs?: number;
  /** Pass the same id again to update that notification in place. */
  id?: string;
  /** 0–100: shows a thin progress bar. */
  progress?: number;
  /** Short text at the end of the title line ("1.2 of 3.4 GB"). Defaults to the percentage when there is progress. */
  detail?: string;
  /** This entry's share of its group's combined progress (e.g. total bytes). Defaults to 1. */
  weight?: number;
  action?: NotificationAction;
  group?: NotificationGroup;
  /** Replaces the tone's icon, e.g. a download arrow. */
  icon?: LucideIcon;
}

interface NotificationRecord extends NotificationInput {
  id: string;
  tone: NotificationTone;
}

interface NotificationContextValue {
  /**
   * Shows a notification and returns its id. Calling it again with that id
   * updates the notification in place. If the user dismissed a notification
   * with progress, its later progress updates stay hidden; its next update
   * without progress (done, failed) shows again.
   */
  notify: (input: NotificationInput) => string;
  dismiss: (id: string) => void;
}

const NotificationContext = createContext<NotificationContextValue | null>(null);

// undefined: no provider (render in place); null: the slot is not mounted yet.
const NotificationSlotContext = createContext<HTMLElement | null | undefined>(
  undefined,
);

const TONE_SEVERITY: Record<NotificationTone, number> = {
  info: 0,
  success: 1,
  warning: 2,
  danger: 3,
};

// The pointer may leave a held notification just as it closes; give the rest
// a moment before they go too.
const MIN_REMAINING_AFTER_HOLD_MS = 1500;

function autoDismissDelay(record: NotificationRecord): number | null {
  if (record.durationMs !== undefined) {
    return Number.isFinite(record.durationMs) ? record.durationMs : null;
  }
  if (record.tone === "danger" || record.progress !== undefined) {
    return null;
  }
  return record.tone === "warning" || record.action ? 8000 : 5000;
}

function clampPercent(value: number): number {
  return Math.min(100, Math.max(0, value));
}

function singleView(record: NotificationRecord): NotificationView {
  return {
    key: record.id,
    ids: [record.id],
    progressIds: record.progress !== undefined ? [record.id] : [],
    title: record.title,
    description: record.description,
    tone: record.tone,
    icon: record.icon,
    progress:
      record.progress !== undefined ? clampPercent(record.progress) : undefined,
    detail: record.detail,
    action: record.action,
  };
}

function groupView(members: NotificationRecord[]): NotificationView {
  const measured = members.filter((member) => member.progress !== undefined);
  const totalWeight = measured.reduce(
    (sum, member) => sum + (member.weight ?? 1),
    0,
  );
  const progress =
    measured.length > 0 && totalWeight > 0
      ? measured.reduce(
          (sum, member) =>
            sum + clampPercent(member.progress ?? 0) * (member.weight ?? 1),
          0,
        ) / totalWeight
      : undefined;
  const latest = members[members.length - 1];

  return {
    // Keyed by the first member so the row stays put as others join.
    key: members[0].id,
    ids: members.map((member) => member.id),
    progressIds: measured.map((member) => member.id),
    title: latest.group?.title(members.length) ?? latest.title,
    tone: members.reduce<NotificationTone>(
      (worst, member) =>
        TONE_SEVERITY[member.tone] > TONE_SEVERITY[worst] ? member.tone : worst,
      "info",
    ),
    icon: members[0].icon,
    progress,
  };
}

/** Rows to show, in order: single notifications and combined groups. */
function toViews(records: NotificationRecord[]): NotificationView[] {
  const groups = new Map<string, NotificationRecord[]>();
  for (const record of records) {
    if (record.group) {
      const members = groups.get(record.group.key) ?? [];
      members.push(record);
      groups.set(record.group.key, members);
    }
  }

  const views: NotificationView[] = [];
  const shownGroups = new Set<string>();
  for (const record of records) {
    const members = record.group ? groups.get(record.group.key) : undefined;
    if (!record.group || !members || members.length < 2) {
      views.push(singleView(record));
    } else if (!shownGroups.has(record.group.key)) {
      shownGroups.add(record.group.key);
      views.push(groupView(members));
    }
  }
  return views;
}

interface DismissTimer {
  handle: ReturnType<typeof setTimeout> | null;
  remaining: number;
  startedAt: number;
}

interface NotificationProviderProps {
  children: ReactNode;
}

export function NotificationProvider({ children }: NotificationProviderProps) {
  const [records, setRecords] = useState<NotificationRecord[]>([]);
  const [slot, setSlot] = useState<HTMLDivElement | null>(null);
  const timersRef = useRef(new Map<string, DismissTimer>());
  // Ids of progress notifications the user closed (see `notify`).
  const hiddenRef = useRef(new Set<string>());
  // Notifications under the pointer or holding focus; timers wait while > 0.
  const holdsRef = useRef(0);

  const clearTimer = useCallback((id: string) => {
    const timer = timersRef.current.get(id);
    if (timer?.handle) {
      clearTimeout(timer.handle);
    }
    timersRef.current.delete(id);
  }, []);

  const remove = useCallback(
    (ids: string[]) => {
      ids.forEach(clearTimer);
      setRecords((current) =>
        current.filter((record) => !ids.includes(record.id)),
      );
    },
    [clearTimer],
  );

  const runTimer = useCallback(
    (id: string, delay: number) => {
      timersRef.current.set(id, {
        handle: setTimeout(() => remove([id]), delay),
        remaining: delay,
        startedAt: Date.now(),
      });
    },
    [remove],
  );

  const scheduleDismiss = useCallback(
    (id: string, delay: number | null) => {
      clearTimer(id);
      if (delay === null) {
        return;
      }
      if (holdsRef.current > 0) {
        timersRef.current.set(id, { handle: null, remaining: delay, startedAt: 0 });
      } else {
        runTimer(id, delay);
      }
    },
    [clearTimer, runTimer],
  );

  const setHeld = useCallback(
    (held: boolean) => {
      const previous = holdsRef.current;
      const next = Math.max(0, previous + (held ? 1 : -1));
      holdsRef.current = next;
      const timers = timersRef.current;
      if (previous === 0 && next > 0) {
        for (const [id, timer] of timers) {
          if (timer.handle) {
            clearTimeout(timer.handle);
            timers.set(id, {
              handle: null,
              remaining: timer.remaining - (Date.now() - timer.startedAt),
              startedAt: 0,
            });
          }
        }
      } else if (previous > 0 && next === 0) {
        for (const [id, timer] of timers) {
          if (!timer.handle) {
            runTimer(id, Math.max(timer.remaining, MIN_REMAINING_AFTER_HOLD_MS));
          }
        }
      }
    },
    [runTimer],
  );

  const notify = useCallback(
    (input: NotificationInput) => {
      const id = input.id ?? createUuid();
      if (input.progress !== undefined && hiddenRef.current.has(id)) {
        return id;
      }
      hiddenRef.current.delete(id);

      const record: NotificationRecord = {
        ...input,
        id,
        tone: input.tone ?? "info",
      };
      setRecords((current) => {
        const index = current.findIndex((item) => item.id === id);
        if (index === -1) {
          return [...current, record];
        }
        const next = current.slice();
        next[index] = record;
        return next;
      });
      scheduleDismiss(id, autoDismissDelay(record));
      return id;
    },
    [scheduleDismiss],
  );

  const dismiss = useCallback(
    (id: string) => {
      hiddenRef.current.delete(id);
      remove([id]);
    },
    [remove],
  );

  const closeView = useCallback(
    (view: NotificationView) => {
      view.progressIds.forEach((id) => hiddenRef.current.add(id));
      remove(view.ids);
    },
    [remove],
  );

  useEffect(() => {
    const timers = timersRef.current;
    return () => {
      for (const timer of timers.values()) {
        if (timer.handle) {
          clearTimeout(timer.handle);
        }
      }
      timers.clear();
    };
  }, []);

  const value = useMemo<NotificationContextValue>(
    () => ({ notify, dismiss }),
    [dismiss, notify],
  );
  const views = useMemo(() => toViews(records), [records]);

  return (
    <NotificationContext.Provider value={value}>
      <NotificationSlotContext.Provider value={slot}>
        {children}
        <section
          aria-label="Notifications"
          className="pointer-events-none fixed bottom-4 right-4 z-80 flex w-[min(23.25rem,calc(100vw-2rem))] flex-col gap-2 sm:bottom-6 sm:right-6"
        >
          <AnimatePresence initial={false}>
            {views.map((view) => (
              <NotificationItem
                key={view.key}
                view={view}
                onClose={closeView}
                onHeldChange={setHeld}
              />
            ))}
          </AnimatePresence>
          <div ref={setSlot} className="contents" />
        </section>
      </NotificationSlotContext.Provider>
    </NotificationContext.Provider>
  );
}

export function useNotifications() {
  const context = useContext(NotificationContext);

  if (!context) {
    throw new Error("useNotifications must be used within NotificationProvider");
  }

  return context;
}

/**
 * Renders a persistent card (the update card) at the bottom of the
 * notification stack so the two never overlap. Children need
 * `pointer-events-auto`. Without a provider it renders in place.
 */
export function NotificationSlot({ children }: { children: ReactNode }) {
  const slot = useContext(NotificationSlotContext);
  if (slot === undefined) {
    return <>{children}</>;
  }
  return slot ? createPortal(children, slot) : null;
}
