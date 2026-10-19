import { useWorkspaceShortcuts } from "@/hooks/useWorkspaceShortcuts";
import { isMacPlatform } from "@/lib/platform";

/**
 * Cmd/Ctrl+N runs a page's "New …" action, unless a dialog (a creation
 * modal, the command palette) is already open.
 */
export function useNewItemShortcut(onNew: () => void) {
  useWorkspaceShortcuts([
    {
      key: "n",
      metaKey: true,
      allowInInputs: true,
      action: () => {
        if (document.querySelector('[role="dialog"][data-state="open"]')) {
          return;
        }
        onNew();
      },
    },
  ]);
}

/** Tooltip text for a "New …" button that the shortcut also triggers. */
export function newItemShortcutTitle(label: string): string {
  return `${label} (${isMacPlatform() ? "⌘N" : "Ctrl+N"})`;
}
