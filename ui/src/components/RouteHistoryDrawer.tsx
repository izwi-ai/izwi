import { useMemo, useState, type ReactElement, type ReactNode } from "react";
import { History, type LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";

interface RouteHistoryDrawerControls {
  close: () => void;
}

interface RouteHistoryDrawerProps {
  title: string;
  countLabel: string;
  eyebrow?: string;
  headerIcon?: LucideIcon;
  triggerLabel?: string;
  triggerCount?: number;
  trigger?: ReactElement;
  headerActions?: ReactNode | ((controls: RouteHistoryDrawerControls) => ReactNode);
  children: ReactNode | ((controls: RouteHistoryDrawerControls) => ReactNode);
  bodyClassName?: string;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

export function RouteHistoryDrawer({
  title,
  countLabel,
  eyebrow = "History",
  headerIcon: HeaderIcon = History,
  triggerLabel = "History",
  triggerCount,
  trigger,
  headerActions,
  children,
  bodyClassName,
  open,
  onOpenChange,
}: RouteHistoryDrawerProps) {
  const [internalOpen, setInternalOpen] = useState(false);
  const isControlled = typeof open === "boolean";
  const resolvedOpen = isControlled ? open : internalOpen;

  const setDrawerOpen = (nextOpen: boolean) => {
    if (!isControlled) {
      setInternalOpen(nextOpen);
    }
    onOpenChange?.(nextOpen);
  };

  const triggerBadge = useMemo(() => {
    if (typeof triggerCount !== "number" || Number.isNaN(triggerCount)) {
      return null;
    }
    if (triggerCount > 99) {
      return "99+";
    }
    return String(Math.max(0, triggerCount));
  }, [triggerCount]);

  const drawerBody =
    typeof children === "function"
      ? children({ close: () => setDrawerOpen(false) })
      : children;
  const drawerHeaderActions =
    typeof headerActions === "function"
      ? headerActions({ close: () => setDrawerOpen(false) })
      : headerActions;

  return (
    <Sheet open={resolvedOpen} onOpenChange={setDrawerOpen}>
      <SheetTrigger asChild>
        {trigger ?? (
          <Button
            type="button"
            variant="outline"
            size="sm"
            className={cn(
              "h-9 shrink-0 gap-2 rounded-lg border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)] shadow-none hover:border-[var(--border-strong)] hover:bg-[var(--bg-surface-3)] hover:text-[var(--text-primary)]",
            )}
          >
            <History className="h-4 w-4 shrink-0" />
            <span>{triggerLabel}</span>
            {triggerBadge ? (
              <span className="inline-flex min-w-5 items-center justify-center rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-3)] px-1.5 py-0.5 text-[10px] font-semibold text-[var(--text-primary)]">
                {triggerBadge}
              </span>
            ) : null}
          </Button>
        )}
      </SheetTrigger>

      <SheetContent
        side="right"
        className="w-[min(92vw,28rem)] max-w-[28rem] gap-0 border-l border-border/70 bg-background/98 p-0"
      >
        <SheetHeader className="gap-0 border-b border-border/70 px-5 py-5 pr-14 sm:px-6">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0 flex-1">
              <div className="inline-flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                <HeaderIcon className="h-3.5 w-3.5" />
                {eyebrow}
              </div>
              <SheetTitle className="mt-2 text-base tracking-tight">
                {title}
              </SheetTitle>
              <SheetDescription className="mt-1 text-xs">
                {countLabel}
              </SheetDescription>
            </div>
            {drawerHeaderActions ? (
              <div className="shrink-0">{drawerHeaderActions}</div>
            ) : null}
          </div>
        </SheetHeader>

        <div
          className={cn(
            "flex min-h-0 flex-1 flex-col gap-3 overflow-hidden px-5 py-4 sm:px-6 sm:py-5",
            bodyClassName,
          )}
        >
          {drawerBody}
        </div>
      </SheetContent>
    </Sheet>
  );
}
