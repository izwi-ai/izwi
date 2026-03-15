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
import { AnimatePresence, motion } from "framer-motion";
import { AlertCircle, CheckCircle2, Info, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type NotificationTone = "info" | "success" | "warning" | "danger";

interface NotificationInput {
  title: string;
  description?: string;
  tone?: NotificationTone;
  durationMs?: number;
}

interface NotificationRecord extends Required<Pick<NotificationInput, "title">> {
  id: string;
  description?: string;
  tone: NotificationTone;
  durationMs: number;
}

interface NotificationContextValue {
  notify: (input: NotificationInput) => string;
  dismiss: (id: string) => void;
}

const NotificationContext = createContext<NotificationContextValue | null>(null);

const toneConfig: Record<
  NotificationTone,
  { icon: typeof Info; className: string }
> = {
  info: {
    icon: Info,
    className:
      "border-[var(--status-info-border)] bg-[color-mix(in_srgb,var(--bg-surface-1)_94%,transparent)] text-[var(--text-primary)]",
  },
  success: {
    icon: CheckCircle2,
    className:
      "border-[var(--status-positive-border)] bg-[color-mix(in_srgb,var(--bg-surface-1)_94%,transparent)] text-[var(--text-primary)]",
  },
  warning: {
    icon: AlertCircle,
    className:
      "border-[var(--status-warning-border)] bg-[color-mix(in_srgb,var(--bg-surface-1)_94%,transparent)] text-[var(--text-primary)]",
  },
  danger: {
    icon: AlertCircle,
    className:
      "border-[var(--danger-border)] bg-[color-mix(in_srgb,var(--bg-surface-1)_94%,transparent)] text-[var(--text-primary)]",
  },
};

interface NotificationProviderProps {
  children: ReactNode;
}

function createNotificationId() {
  return `toast-${Math.random().toString(36).slice(2, 10)}`;
}

export function NotificationProvider({
  children,
}: NotificationProviderProps) {
  const [notifications, setNotifications] = useState<NotificationRecord[]>([]);
  const timersRef = useRef<Record<string, ReturnType<typeof setTimeout>>>({});

  const dismiss = useCallback((id: string) => {
    const timer = timersRef.current[id];
    if (timer) {
      clearTimeout(timer);
      delete timersRef.current[id];
    }
    setNotifications((current) => current.filter((item) => item.id !== id));
  }, []);

  const notify = useCallback(
    ({
      title,
      description,
      tone = "info",
      durationMs = 4200,
    }: NotificationInput) => {
      const id = createNotificationId();
      setNotifications((current) => [
        ...current,
        {
          id,
          title,
          description,
          tone,
          durationMs,
        },
      ]);

      timersRef.current[id] = setTimeout(() => {
        dismiss(id);
      }, durationMs);

      return id;
    },
    [dismiss],
  );

  useEffect(() => {
    return () => {
      Object.values(timersRef.current).forEach((timer) => clearTimeout(timer));
      timersRef.current = {};
    };
  }, []);

  const value = useMemo<NotificationContextValue>(
    () => ({
      notify,
      dismiss,
    }),
    [dismiss, notify],
  );

  return (
    <NotificationContext.Provider value={value}>
      {children}
      <div className="pointer-events-none fixed right-4 top-4 z-[80] flex w-[min(28rem,calc(100vw-2rem))] flex-col gap-3">
        <AnimatePresence initial={false}>
          {notifications.map((notification) => {
            const config = toneConfig[notification.tone];
            const Icon = config.icon;
            return (
              <motion.div
                key={notification.id}
                initial={{ opacity: 0, y: -8, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -8, scale: 0.98 }}
                transition={{ duration: 0.16 }}
                className={cn(
                  "pointer-events-auto rounded-[var(--radius-lg)] border px-4 py-3 shadow-[var(--shadow-overlay)] backdrop-blur-md",
                  config.className,
                )}
              >
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-[var(--radius-sm)] border border-border/70 bg-background/55">
                    <Icon className="h-4 w-4" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-semibold tracking-tight">
                      {notification.title}
                    </div>
                    {notification.description ? (
                      <div className="mt-1 text-sm leading-5 text-[var(--text-secondary)]">
                        {notification.description}
                      </div>
                    ) : null}
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 shrink-0 rounded-full"
                    onClick={() => dismiss(notification.id)}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </NotificationContext.Provider>
  );
}

export function useNotifications() {
  const context = useContext(NotificationContext);

  if (!context) {
    throw new Error(
      "useNotifications must be used within NotificationProvider",
    );
  }

  return context;
}
