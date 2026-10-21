import { useMemo, useState, type ReactElement, type ReactNode } from "react";
import { type LucideIcon, History } from "lucide-react";

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
              "h-9 shrink-0 gap-2 rounded-lg border-(--border-muted) bg-(--bg-surface-2) text-(--text-secondary) shadow-none hover:border-(--border-strong) hover:bg-(--bg-surface-3) hover:text-(--text-primary)",
            )}
          >
            <HeaderIcon className="h-4 w-4 shrink-0" />
            <span>{triggerLabel}</span>
            {triggerBadge ? (
              <span className="inline-flex min-w-5 items-center justify-center rounded-full border border-(--border-muted) bg-(--bg-surface-3) px-1.5 py-0.5 text-[10px] font-semibold text-(--text-primary)">
                {triggerBadge}
              </span>
            ) : null}
          </Button>
        )}
      </SheetTrigger>

      <SheetContent
        side="right"
        className="w-[min(92vw,22rem)] max-w-88 gap-0 border-l border-(--border-subtle) bg-(--surface-panel) p-0"
      >
        <SheetHeader className="gap-0 border-b border-(--border-subtle) px-5 py-4 pr-14">
          <div className="flex items-start justify-between gap-4">
            <div className="min-w-0 flex-1">
              <SheetTitle className="text-[15px] font-semibold tracking-tight text-(--text-primary)">
                {title}
              </SheetTitle>
              <SheetDescription className="mt-0.5 text-xs text-(--text-muted)">
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
            "flex min-h-0 flex-1 flex-col gap-3 overflow-hidden px-4 py-4",
            bodyClassName,
          )}
        >
          {drawerBody}
        </div>
      </SheetContent>
    </Sheet>
  );
}
