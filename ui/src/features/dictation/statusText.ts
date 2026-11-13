import { formatHotkey } from "@/features/dictation/hotkey";
import type {
  DictationOs,
  DictationResultStatus,
  DictationSettings,
  DictationStatus,
} from "@/features/dictation/types";

const PREVIEW_LENGTH = 80;

function preview(text: string): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > PREVIEW_LENGTH
    ? `${flat.slice(0, PREVIEW_LENGTH - 1)}…`
    : flat;
}

/** "⌘V" on macOS, "Ctrl+V" elsewhere. */
export function pasteShortcutLabel(os: DictationOs): string {
  return os === "macos" ? "⌘V" : "Ctrl+V";
}

/** How a finished dictation ended, as one line. */
export function resultText(
  status: DictationResultStatus,
  os: DictationOs,
): string {
  if (status.state === "failed") {
    return status.message;
  }
  if (status.state === "discarded") {
    switch (status.reason) {
      case "too_short":
        return "Too short. Keep the shortcut held while you speak.";
      case "cancelled":
        return "Cancelled.";
      case "no_speech":
        return "Didn't catch that.";
    }
  }
  const { outcome } = status;
  switch (outcome.kind) {
    case "pasted":
    case "typed":
      return `Inserted “${preview(status.text)}”`;
    case "failed":
      return outcome.message;
    case "clipboard": {
      const copied = `Copied. Paste with ${pasteShortcutLabel(os)}.`;
      switch (outcome.reason) {
        case "requested":
          return copied;
        case "not_permitted":
          return `${copied} Allow keyboard control to have Izwi paste for you.`;
        case "elevated":
          return `${copied} Windows administrator windows can't be typed into.`;
        case "unsupported":
          return `${copied} Pasting isn't available here.`;
        case "failed":
          return outcome.detail
            ? `${copied} Pasting failed: ${outcome.detail}`
            : `${copied} Pasting failed.`;
      }
    }
  }
}

/** What dictation is doing now, when it is doing something. */
export function activityText(status: DictationStatus): string | null {
  switch (status.state) {
    case "starting":
      return "Starting…";
    case "listening":
      return status.locked
        ? "Listening hands-free. Press the shortcut to stop."
        : "Listening…";
    case "transcribing":
      return "Transcribing…";
    default:
      return null;
  }
}

/** How to start dictating with the current shortcut. */
export function readyText(settings: DictationSettings, os: DictationOs): string {
  const hotkey = formatHotkey(settings.hotkey, os);
  return settings.mode === "hold"
    ? `Ready — hold ${hotkey} and speak`
    : `Ready — press ${hotkey}, speak, then press it again`;
}
