import type { ReactNode } from "react";
import { Music4 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Card, CardContent } from "@/components/ui/card";

export interface VoicePickerItem {
  id: string;
  name: string;
  categoryLabel: string;
  description?: string;
  meta?: string[];
  previewUrl?: string | null;
  previewMessage?: string | null;
  selected?: boolean;
  onSelect?: () => void;
  actions?: ReactNode;
}

interface VoicePickerProps {
  items: VoicePickerItem[];
  emptyTitle: string;
  emptyDescription: string;
  className?: string;
}

export function VoicePicker({
  items,
  emptyTitle,
  emptyDescription,
  className,
}: VoicePickerProps) {
  if (items.length === 0) {
    return (
      <Card className={cn("border-dashed", className)}>
        <CardContent className="flex min-h-48 flex-col items-center justify-center gap-3 text-center">
          <div className="rounded-2xl border border-border/70 bg-muted/45 p-4">
            <Music4 className="h-6 w-6 text-muted-foreground" />
          </div>
          <div className="space-y-1">
            <p className="text-sm font-semibold text-foreground">
              {emptyTitle}
            </p>
            <p className="max-w-md text-sm text-muted-foreground">
              {emptyDescription}
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div
      className={cn(
        "grid gap-4 grid-cols-[repeat(auto-fill,minmax(280px,1fr))]",
        className,
      )}
    >
      {items.map((item) => {
        return (
          <Card
            key={item.id}
            className={cn(
              "h-full border-border/75 bg-card/90 transition-colors",
              item.selected &&
                "border-primary/60 shadow-[0_18px_40px_-34px_hsl(var(--primary)/0.9)]",
              item.onSelect && "cursor-pointer hover:border-primary/40",
            )}
            role={item.onSelect ? "button" : undefined}
            tabIndex={item.onSelect ? 0 : undefined}
            onClick={item.onSelect}
            onKeyDown={(event) => {
              if (!item.onSelect) {
                return;
              }
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                item.onSelect();
              }
            }}
          >
            <CardContent className="flex h-full flex-col gap-4 p-5">
              <div className="space-y-2">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                      {item.categoryLabel}
                    </div>
                    <div className="mt-1 text-base font-semibold text-foreground">
                      {item.name}
                    </div>
                  </div>
                  {item.selected ? (
                    <div className="rounded-full border border-primary/35 bg-primary/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-primary">
                      Selected
                    </div>
                  ) : null}
                </div>
                {item.description ? (
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    {item.description}
                  </p>
                ) : null}
              </div>

              {item.meta && item.meta.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {item.meta.map((meta) => (
                    <span
                      key={`${item.id}-${meta}`}
                      className="rounded-full border border-border/75 bg-muted/55 px-2.5 py-1 text-[11px] font-medium text-muted-foreground"
                    >
                      {meta}
                    </span>
                  ))}
                </div>
              ) : null}

              <div className="mt-auto space-y-3">
                {item.previewUrl ? (
                  <audio
                    src={item.previewUrl}
                    controls
                    preload="none"
                    onClick={(event) => event.stopPropagation()}
                    className="h-10 w-full"
                  />
                ) : item.previewMessage ? (
                  <div className="rounded-xl border border-dashed border-border/75 bg-muted/35 px-3 py-2 text-xs text-muted-foreground">
                    {item.previewMessage}
                  </div>
                ) : null}

                {item.actions ? (
                  <div
                    className="flex flex-wrap items-center gap-2"
                    onClick={(event) => event.stopPropagation()}
                  >
                    {item.actions}
                  </div>
                ) : null}
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
