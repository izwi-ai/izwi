import { useRef } from "react";

type AutoFocusHandler = ((event: Event) => void) | undefined;

/**
 * Radix dialogs return focus only to a DialogTrigger. Ours usually open from
 * a plain button (or a menu, or a shortcut), so remember what had focus when
 * the overlay opened and put focus back there when it closes.
 */
export function useReturnFocus(onOpenAutoFocus: AutoFocusHandler, onCloseAutoFocus: AutoFocusHandler) {
  const previous = useRef<HTMLElement | null>(null);
  return {
    onOpenAutoFocus: (event: Event) => {
      const active = document.activeElement;
      previous.current = active instanceof HTMLElement && active !== document.body ? active : null;
      onOpenAutoFocus?.(event);
    },
    onCloseAutoFocus: (event: Event) => {
      onCloseAutoFocus?.(event);
      const target = previous.current;
      previous.current = null;
      if (event.defaultPrevented || !target?.isConnected) return;
      event.preventDefault();
      target.focus();
    },
  };
}
