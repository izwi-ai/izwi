import { pasteShortcutLabel } from "@/features/dictation/statusText";
import type { DictationOs, DictationStatus } from "@/features/dictation/types";

/** What the indicator pill shows. */
export type IndicatorView =
  | { kind: "hidden" }
  | { kind: "starting" }
  | { kind: "listening"; locked: boolean }
  | { kind: "transcribing" }
  /** The text went in: nothing more to say. */
  | { kind: "done" }
  | { kind: "notice"; text: string; tone: "neutral" | "danger" };

export const HIDDEN_VIEW: IndicatorView = { kind: "hidden" };

/**
 * The view after a status. Every dictation ends with its result followed at
 * once by `idle`, and the desktop app hides the window a moment later, so
 * `idle` leaves a result on screen; only new activity replaces it.
 */
export function nextIndicatorView(
  current: IndicatorView,
  status: DictationStatus,
  os: DictationOs,
): IndicatorView {
  switch (status.state) {
    case "idle":
      return current.kind === "notice" || current.kind === "done"
        ? current
        : HIDDEN_VIEW;
    case "starting":
      return { kind: "starting" };
    case "listening":
      return { kind: "listening", locked: status.locked };
    case "transcribing":
      return { kind: "transcribing" };
    case "failed":
      return { kind: "notice", text: status.message, tone: "danger" };
    case "discarded":
      return status.reason === "no_speech"
        ? { kind: "notice", text: "Didn't catch that", tone: "neutral" }
        : HIDDEN_VIEW;
    case "inserted":
      switch (status.outcome.kind) {
        case "clipboard":
          return {
            kind: "notice",
            text: `Copied — paste with ${pasteShortcutLabel(os)}`,
            tone: "neutral",
          };
        case "failed":
          return { kind: "notice", text: status.outcome.message, tone: "danger" };
        default:
          return { kind: "done" };
      }
  }
}

/** Bar heights (0..1) for the newest levels, oldest first. */
export function pushLevel(levels: number[], level: number, size: number): number[] {
  const clamped = Math.min(1, Math.max(0, Number.isFinite(level) ? level : 0));
  // Speech sits low in the range: the square root makes quiet words visible.
  const next = [...levels, Math.sqrt(clamped)];
  return next.length > size ? next.slice(next.length - size) : next;
}
