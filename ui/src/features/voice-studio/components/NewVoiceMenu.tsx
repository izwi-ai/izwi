import { useEffect, useRef } from "react";
import { Mic, Plus, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { newItemShortcutTitle } from "@/hooks/useNewItemShortcut";
import { isMacPlatform } from "@/lib/platform";

export type NewVoiceKind = "clone" | "design";

const ITEMS = [
  { kind: "clone", icon: Mic, title: "Clone a voice", detail: "From a short recording" },
  { kind: "design", icon: Sparkles, title: "Design a voice", detail: "From a description" },
] as const;

interface NewVoiceMenuProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSelect: (kind: NewVoiceKind) => void;
}

/** The Voices page's "New voice" button: a menu of the two ways to make one. */
export function NewVoiceMenu({ open, onOpenChange, onSelect }: NewVoiceMenuProps) {
  const triggerRef = useRef<HTMLButtonElement>(null);
  const chosenRef = useRef<NewVoiceKind | null>(null);
  const handedOffRef = useRef(false);
  const label = "New voice";

  // Once the menu has closed, put focus on the button and only then open the
  // chosen dialog: a dialog gives focus back to what had it when it opened.
  useEffect(() => {
    const kind = chosenRef.current;
    if (open || !kind) {
      return;
    }
    chosenRef.current = null;
    handedOffRef.current = true;
    triggerRef.current?.focus();
    onSelect(kind);
  }, [onSelect, open]);

  return (
    <DropdownMenu open={open} onOpenChange={onOpenChange}>
      <DropdownMenuTrigger asChild ref={triggerRef}>
        <Button
          type="button"
          size="sm"
          className="h-9 gap-2"
          title={newItemShortcutTitle(label)}
          aria-keyshortcuts={isMacPlatform() ? "Meta+N" : "Control+N"}
        >
          <Plus className="h-4 w-4" />
          {label}
          <kbd aria-hidden="true" className="font-sans text-[11px] opacity-70">
            {isMacPlatform() ? "⌘N" : "Ctrl N"}
          </kbd>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent
        align="end"
        className="w-[300px] rounded-[12px] border-(--border-default) bg-(--bg-surface-1) p-1"
        // The menu finishes closing after the dialog has opened; it must not
        // pull focus out of the dialog.
        onCloseAutoFocus={(event) => {
          if (handedOffRef.current) {
            handedOffRef.current = false;
            event.preventDefault();
          }
        }}
      >
        {ITEMS.map((item) => (
          <DropdownMenuItem
            key={item.kind}
            onSelect={() => {
              chosenRef.current = item.kind;
            }}
            className="items-start gap-3 rounded-[8px] px-2.5 py-2.5 focus:bg-(--accent-soft)"
          >
            <item.icon
              aria-hidden="true"
              className="mt-0.5 h-4 w-4 shrink-0 text-(--text-muted)"
            />
            <span className="flex min-w-0 flex-col">
              <span className="text-[13.5px] font-medium text-(--text-primary)">
                {item.title}
              </span>
              <span className="text-[12.5px] text-(--text-subtle)">
                {item.detail}
              </span>
            </span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
