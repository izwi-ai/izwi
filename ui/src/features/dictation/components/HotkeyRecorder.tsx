import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";

import { dictationApi, dictationAvailable } from "@/features/dictation/api";
import {
  EMPTY_RECORDER_STATE,
  hotkeyParts,
  recordHotkeyEvent,
  type HotkeyRecorderState,
} from "@/features/dictation/hotkey";
import type { DictationOs } from "@/features/dictation/types";
import { cn } from "@/lib/utils";

const KBD_CLASS =
  "rounded-[6px] border border-(--border-default) px-1.5 py-0.5 font-sans text-xs text-(--text-secondary)";

/** A hotkey as one keycap per key. */
export function HotkeyKeys({
  hotkey,
  os,
  className,
}: {
  hotkey: string;
  os: DictationOs;
  className?: string;
}) {
  return (
    <span className={cn("inline-flex items-center gap-1", className)}>
      {hotkeyParts(hotkey, os).map((part, index) => (
        <kbd key={`${part}-${index}`} className={KBD_CLASS}>
          {part}
        </kbd>
      ))}
    </span>
  );
}

const REJECTION_HINTS = {
  needs_modifier: "Add Ctrl, Alt, Shift or another modifier key.",
  needs_key: "Hold the modifier keys and press one more key.",
} as const;

interface HotkeyRecorderProps {
  /** Names the control, e.g. "Dictation shortcut". */
  label: string;
  value: string | null;
  os: DictationOs;
  /** Whether modifiers alone ("Alt") can be a shortcut here. */
  modifierOnly: boolean;
  onChange: (hotkey: string) => void;
  /** Shows a clear button while a value is set. */
  onClear?: () => void;
  disabled?: boolean;
}

/**
 * Click, then press the keys: records a shortcut from the keyboard. Escape
 * or clicking away cancels and keeps the current shortcut.
 */
export function HotkeyRecorder({
  label,
  value,
  os,
  modifierOnly,
  onChange,
  onClear,
  disabled = false,
}: HotkeyRecorderProps) {
  const [recording, setRecording] = useState(false);
  const [held, setHeld] = useState("");
  const [hint, setHint] = useState<string | null>(null);
  const stateRef = useRef<HotkeyRecorderState>(EMPTY_RECORDER_STATE);
  const onChangeRef = useRef(onChange);

  useEffect(() => {
    onChangeRef.current = onChange;
  }, [onChange]);

  // The registered hotkeys are off for as long as one is being recorded.
  useEffect(() => {
    if (!recording || !dictationAvailable()) {
      return;
    }
    void dictationApi.pauseHotkeys(true).catch(() => {});
    return () => {
      void dictationApi.pauseHotkeys(false).catch(() => {});
    };
  }, [recording]);

  useEffect(() => {
    if (!recording) {
      return;
    }
    stateRef.current = EMPTY_RECORDER_STATE;

    const stop = () => {
      stateRef.current = EMPTY_RECORDER_STATE;
      setRecording(false);
      setHeld("");
    };

    const handleKey = (event: KeyboardEvent) => {
      // The keys belong to the recording: no app shortcut may act on them.
      event.preventDefault();
      event.stopPropagation();
      const step = recordHotkeyEvent(
        stateRef.current,
        {
          type: event.type === "keyup" ? "keyup" : "keydown",
          code: event.code,
        },
        { modifierOnly },
      );
      stateRef.current = step.state;
      setHeld(step.state.held.join("+"));
      if (!step.outcome) {
        return;
      }
      if (step.outcome.kind === "rejected") {
        setHint(REJECTION_HINTS[step.outcome.reason]);
        return;
      }
      setHint(null);
      stop();
      if (step.outcome.kind === "captured") {
        onChangeRef.current(step.outcome.hotkey);
      }
    };

    window.addEventListener("keydown", handleKey, true);
    window.addEventListener("keyup", handleKey, true);
    window.addEventListener("blur", stop);
    return () => {
      window.removeEventListener("keydown", handleKey, true);
      window.removeEventListener("keyup", handleKey, true);
      window.removeEventListener("blur", stop);
    };
  }, [recording, modifierOnly]);

  return (
    <div className="flex flex-col items-start gap-1.5 md:items-end">
      <div className="flex items-center gap-1.5">
        <button
          type="button"
          aria-label={
            recording
              ? `${label}: press the keys, or Escape to cancel`
              : value
                ? `${label}: ${hotkeyParts(value, os).join(" ")}. Change`
                : `${label}: not set. Record`
          }
          disabled={disabled}
          onClick={() => {
            setHint(null);
            setRecording((current) => !current);
          }}
          onBlur={() => setRecording(false)}
          className={cn(
            "inline-flex h-9 min-w-36 items-center justify-center gap-1 rounded-[10px] border px-3 text-[13px] transition-colors focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-(--border-strong) disabled:cursor-not-allowed disabled:opacity-50",
            recording
              ? "border-(--text-primary) bg-(--surface-panel) text-(--text-primary)"
              : "border-(--border-default) bg-(--surface-panel) text-(--text-secondary) hover:border-(--border-strong)",
          )}
        >
          {recording ? (
            held ? (
              <HotkeyKeys hotkey={held} os={os} />
            ) : (
              "Press keys…"
            )
          ) : value ? (
            <HotkeyKeys hotkey={value} os={os} />
          ) : (
            <span className="text-(--text-subtle)">Not set</span>
          )}
        </button>
        {onClear && value && !recording ? (
          <button
            type="button"
            aria-label={`Remove ${label.toLowerCase()}`}
            disabled={disabled}
            onClick={onClear}
            className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-(--text-subtle) transition-colors hover:bg-(--bg-surface-2) hover:text-(--text-primary) focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-(--border-strong)"
          >
            <X className="h-4 w-4" />
          </button>
        ) : null}
      </div>
      {recording || hint ? (
        <p className="text-[12.5px] text-(--text-subtle)" aria-live="polite">
          {hint ??
            (modifierOnly
              ? "Press a key combination, or press and release modifier keys alone. Esc cancels."
              : "Press a key combination. Esc cancels.")}
        </p>
      ) : null}
    </div>
  );
}
