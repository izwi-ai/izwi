import { useEffect } from "react";

interface WorkspaceShortcut {
  key: string;
  action: () => void;
  metaKey?: boolean;
  shiftKey?: boolean;
  altKey?: boolean;
  enabled?: boolean;
  allowInInputs?: boolean;
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }

  if (target.isContentEditable) {
    return true;
  }

  const tagName = target.tagName;
  return (
    tagName === "INPUT" ||
    tagName === "TEXTAREA" ||
    tagName === "SELECT"
  );
}

export function useWorkspaceShortcuts(shortcuts: WorkspaceShortcut[]) {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      for (const shortcut of shortcuts) {
        if (shortcut.enabled === false) {
          continue;
        }

        if (event.key !== shortcut.key) {
          continue;
        }

        const expectsMeta = shortcut.metaKey ?? false;
        const expectsShift = shortcut.shiftKey ?? false;
        const expectsAlt = shortcut.altKey ?? false;

        if ((event.metaKey || event.ctrlKey) !== expectsMeta) {
          continue;
        }
        if (event.shiftKey !== expectsShift) {
          continue;
        }
        if (event.altKey !== expectsAlt) {
          continue;
        }
        if (!shortcut.allowInInputs && isEditableTarget(event.target)) {
          continue;
        }

        event.preventDefault();
        shortcut.action();
        return;
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [shortcuts]);
}
