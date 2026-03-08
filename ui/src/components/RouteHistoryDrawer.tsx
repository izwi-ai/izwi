import { useMemo, useState, type ReactNode } from "react";
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
  headerActions?: ReactNode | ((controls: RouteHistoryDrawerControls) => ReactNode);
  children: ReactNode | ((controls: RouteHistoryDrawerControls) => ReactNode);
  bodyClassName?: string;
}

export function RouteHistoryDrawer({
  title,
  countLabel,
  eyebrow = "History",
  headerIcon: HeaderIcon = History,
  triggerLabel = "History",
  triggerCount,
  headerActions,
  children,
  bodyClassName,
}: RouteHistoryDrawerProps) {
  const [open, setOpen] = useState(false);

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
    typeof children === "function" ? children({ close: () => setOpen(false) }) : children;
  const drawerHeaderActions =
    typeof headerActions === "function"
      ? headerActions({ close: () => setOpen(false) })
      : headerActions;

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button
          type="button"
          variant="outline"
          className={cn(
            "fixed bottom-4 right-4 z-30 h-11 rounded-full border-border/80 bg-background/94 px-4 shadow-lg backdrop-blur-xl hover:bg-background sm:bottom-6 sm:right-6",
            "lg:bottom-auto lg:top-1/2 lg:right-0 lg:h-auto lg:w-14 lg:-translate-y-1/2 lg:flex-col lg:gap-3 lg:rounded-l-2xl lg:rounded-r-none lg:px-2 lg:py-4",
          )}
        >
          <History className="h-4 w-4 shrink-0" />
          <span className="text-xs font-semibold lg:hidden">{triggerLabel}</span>
          <span className="hidden text-[10px] font-semibold uppercase tracking-[0.2em] lg:block [writing-mode:vertical-rl] rotate-180">
            {triggerLabel}
          </span>
          {triggerBadge ? (
            <span className="inline-flex min-w-5 items-center justify-center rounded-full border border-border/70 bg-primary/10 px-1.5 py-0.5 text-[10px] font-semibold text-foreground">
              {triggerBadge}
            </span>
          ) : null}
        </Button>
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
