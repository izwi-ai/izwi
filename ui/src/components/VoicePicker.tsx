import type { ReactNode } from "react";
import { Music4 } from "lucide-react";
import { cn } from "@/lib/utils";
import { Card, CardContent } from "@/components/ui/card";
import {
  VOICE_ROUTE_BODY_COPY_CLASS,
  VOICE_ROUTE_PANEL_TITLE_CLASS,
  VOICE_ROUTE_SECTION_LABEL_CLASS,
} from "@/components/voiceRouteTypography";

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

function itemInitial(name: string): string {
  return name.trim().charAt(0).toUpperCase() || "V";
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
            <p className={VOICE_ROUTE_PANEL_TITLE_CLASS}>{emptyTitle}</p>
            <p className={cn(VOICE_ROUTE_BODY_COPY_CLASS, "max-w-md")}>
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
        "grid gap-3 grid-cols-[repeat(auto-fill,minmax(250px,1fr))]",
        className,
      )}
    >
      {items.map((item) => {
        return (
          <Card
            key={item.id}
            className={cn(
              "h-full border-border/75 bg-card/90 transition-[border-color,transform,box-shadow] duration-150",
              item.selected &&
                "border-primary/60 shadow-[0_18px_40px_-34px_hsl(var(--primary)/0.9)]",
              item.onSelect &&
                "cursor-pointer hover:-translate-y-[1px] hover:border-primary/40 hover:shadow-[0_22px_44px_-34px_rgba(0,0,0,0.38)]",
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
              <div className="flex items-start gap-3">
                <div className="speaker-avatar flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl text-sm font-semibold">
                  {itemInitial(item.name)}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--text-muted)]">
                      {item.categoryLabel}
                    </span>
                    {item.selected ? (
                      <div className="rounded-full border border-primary/35 bg-primary/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-primary">
                        Selected
                      </div>
                    ) : null}
                  </div>
                  <div className={cn(VOICE_ROUTE_PANEL_TITLE_CLASS, "mt-2")}>
                    {item.name}
                  </div>

                  {item.meta && item.meta.length > 0 ? (
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {item.meta.map((meta) => (
                        <span
                          key={`${item.id}-${meta}`}
                          className="rounded-full border border-border/75 bg-muted/55 px-2.5 py-0.5 text-[11px] font-medium text-muted-foreground"
                        >
                          {meta}
                        </span>
                      ))}
                    </div>
                  ) : null}
                </div>
              </div>

              {item.description ? (
                <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-3.5">
                  <div className={cn(VOICE_ROUTE_SECTION_LABEL_CLASS, "mb-2")}>
                    Voice Notes
                  </div>
                  <p className={cn(VOICE_ROUTE_BODY_COPY_CLASS, "line-clamp-4")}>
                    {item.description}
                  </p>
                </div>
              ) : null}

              <div className="mt-auto space-y-3">
                <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-3.5">
                  <div className={cn(VOICE_ROUTE_SECTION_LABEL_CLASS, "mb-2")}>
                    Preview
                  </div>
                {item.previewUrl ? (
                  <audio
                    src={item.previewUrl}
                    controls
                    preload="none"
                    onClick={(event) => event.stopPropagation()}
                    className="h-10 w-full"
                  />
                ) : item.previewMessage ? (
                  <div className="rounded-xl border border-dashed border-border/75 bg-[var(--bg-surface-0)] px-3 py-2 text-xs text-muted-foreground">
                    {item.previewMessage}
                  </div>
                ) : null}
                </div>

                {item.actions ? (
                  <div
                    className="grid gap-2 sm:grid-cols-2 [&>*]:w-full [&>*]:justify-center [&>*:only-child]:sm:col-span-2"
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
