import { Plus } from "lucide-react";

import { Button } from "@/components/ui/button";
import { newItemShortcutTitle } from "@/hooks/useNewItemShortcut";
import { isMacPlatform } from "@/lib/platform";

/** A page's primary "New …" action, which Cmd/Ctrl+N also runs. */
export function NewItemButton({
  label,
  onClick,
}: {
  label: string;
  onClick: () => void;
}) {
  return (
    <Button
      type="button"
      size="sm"
      className="h-9 gap-2"
      onClick={onClick}
      title={newItemShortcutTitle(label)}
      aria-keyshortcuts={isMacPlatform() ? "Meta+N" : "Control+N"}
    >
      <Plus className="h-4 w-4" />
      {label}
      <kbd aria-hidden="true" className="font-sans text-[11px] opacity-70">
        {isMacPlatform() ? "⌘N" : "Ctrl N"}
      </kbd>
    </Button>
  );
}
