import { useEffect, useLayoutEffect, useRef } from "react";

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

/**
 * Window keyboard shortcuts. The listener is registered once and reads the
 * latest list, so callers can pass a fresh array (and fresh handlers) on
 * every render.
 */
export function useWorkspaceShortcuts(shortcuts: WorkspaceShortcut[]) {
  const shortcutsRef = useRef(shortcuts);
  useLayoutEffect(() => {
    shortcutsRef.current = shortcuts;
  });

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      for (const shortcut of shortcutsRef.current) {
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
  }, []);
}
