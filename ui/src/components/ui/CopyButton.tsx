import { useEffect, useRef, useState } from "react";
import { Check, Copy } from "lucide-react";

import { cn } from "@/lib/utils";

const COPIED_RESET_MS = 2000;

/** Copies text, falling back to a hidden textarea where the async clipboard
 * API is unavailable or refused. Resolves to whether the copy happened. */
export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    try {
      return document.execCommand("copy");
    } catch {
      return false;
    } finally {
      textarea.remove();
    }
  }
}

interface CopyButtonProps {
  /** Text to copy, or a function read at click time. */
  text: string | (() => string);
  /** Accessible name before copying, e.g. "Copy message". */
  label?: string;
  /** Show the label next to the icon instead of an icon-only button. */
  showLabel?: boolean;
  className?: string;
}

/** Copy control that confirms with a check and a "Copied" tooltip. */
export function CopyButton({
  text,
  label = "Copy",
  showLabel = false,
  className,
}: CopyButtonProps) {
  const [copied, setCopied] = useState(false);
  const resetTimerRef = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (resetTimerRef.current !== null) {
        window.clearTimeout(resetTimerRef.current);
      }
    },
    [],
  );

  const handleClick = async () => {
    const value = typeof text === "function" ? text() : text;
    if (!value || !(await copyText(value))) {
      return;
    }
    setCopied(true);
    if (resetTimerRef.current !== null) {
      window.clearTimeout(resetTimerRef.current);
    }
    resetTimerRef.current = window.setTimeout(() => {
      setCopied(false);
      resetTimerRef.current = null;
    }, COPIED_RESET_MS);
  };

  const Icon = copied ? Check : Copy;

  return (
    <span className="relative inline-flex">
      <button
        type="button"
        onClick={() => void handleClick()}
        aria-label={showLabel ? undefined : copied ? "Copied" : label}
        title={showLabel ? undefined : label}
        className={cn(
          "inline-flex items-center justify-center gap-1.5 rounded-lg text-[var(--text-subtle)] transition-colors hover:bg-[var(--surface-raised)] hover:text-[var(--text-primary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--border-strong)]",
          showLabel ? "h-7 px-2 text-xs" : "h-7 w-7",
          className,
        )}
      >
        <Icon className="h-[15px] w-[15px]" aria-hidden="true" />
        {showLabel ? (copied ? "Copied" : label) : null}
      </button>
      {copied && !showLabel ? (
        <span
          role="status"
          className="pointer-events-none absolute bottom-full left-1/2 mb-1.5 -translate-x-1/2 whitespace-nowrap rounded-md bg-[var(--text-primary)] px-2 py-1 text-[11.5px] font-medium text-[var(--bg-app)]"
        >
          Copied
        </span>
      ) : null}
    </span>
  );
}
