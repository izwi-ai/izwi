import { Fragment } from "react";
import { Loader2, MoreHorizontal, type LucideIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/utils";

export interface RowMenuItem {
  id: string;
  label: string;
  icon: LucideIcon;
  onSelect: () => void;
  disabled?: boolean;
  danger?: boolean;
  /** Draw a separator above this item. */
  separatorBefore?: boolean;
}

interface RowMenuProps {
  /** Names the row in the trigger's label: "More actions for {label}". */
  label: string;
  items: RowMenuItem[];
  busy?: boolean;
  triggerClassName?: string;
}

/** The "…" menu at the end of a history row. */
export function RowMenu({
  label,
  items,
  busy = false,
  triggerClassName,
}: RowMenuProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className={cn(
            "h-7 w-7 rounded-lg text-[var(--text-subtle)] hover:bg-[var(--bg-surface-2)] hover:text-[var(--text-primary)]",
            triggerClassName,
          )}
          aria-label={`More actions for ${label}`}
        >
          {busy ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <MoreHorizontal className="h-4 w-4" />
          )}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-48">
        {items.map((item) => (
          <Fragment key={item.id}>
            {item.separatorBefore ? <DropdownMenuSeparator /> : null}
            <DropdownMenuItem
              disabled={item.disabled}
              onSelect={item.onSelect}
              className={cn(
                item.danger &&
                  "text-[var(--danger-text)] focus:text-[var(--danger-text)]",
              )}
            >
              <item.icon className="mr-2 h-4 w-4" />
              {item.label}
            </DropdownMenuItem>
          </Fragment>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
