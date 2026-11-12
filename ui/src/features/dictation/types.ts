// Mirrors the desktop app's dictation types, in the snake_case Rust
// serializes them with.

export type DictationMode = "hold" | "toggle";
export type InsertMethod = "paste" | "type" | "clipboard";

export interface DictationReplacement {
  from: string;
  to: string;
}

export interface DictationCleanupSettings {
  enabled: boolean;
  /** Chat model variant; null picks one automatically. */
  model: string | null;
  deadline_ms: number;
}

export interface DictationSettings {
  enabled: boolean;
  /** e.g. "Alt+Space", "Ctrl+Shift+Space", "Alt", "Fn". */
  hotkey: string;
  mode: DictationMode;
  /** Hold mode: two quick presses keep recording until the next press. */
  double_tap_lock: boolean;
  paste_last_hotkey: string | null;
  /** ASR model variant; null is the default model. */
  model: string | null;
  /** null detects the language. */
  language: string | null;
  /** Device name; null is the system default. */
  microphone: string | null;
  keep_microphone_ready: boolean;
  show_indicator: boolean;
  sounds: boolean;
  mute_while_recording: boolean;
  insert_method: InsertMethod;
  trailing_space: boolean;
  remove_fillers: boolean;
  replacements: DictationReplacement[];
  vocabulary: string[];
  live_preview: boolean;
  cleanup: DictationCleanupSettings;
  keep_history: boolean;
}

export type ClipboardReason =
  | "requested"
  | "not_permitted"
  | "elevated"
  | "unsupported"
  | "failed";

export type InsertOutcome =
  | { kind: "pasted" }
  | { kind: "typed" }
  | { kind: "clipboard"; reason: ClipboardReason; detail: string | null }
  | { kind: "failed"; message: string };

export interface DictationTimings {
  audio_ms: number;
  capture_ms: number;
  transcribe_ms: number;
  cleanup_ms: number;
  insert_ms: number;
}

export type DictationStatus =
  | { state: "idle" }
  | { state: "starting" }
  | { state: "listening"; locked: boolean }
  | { state: "transcribing" }
  | {
      state: "inserted";
      text: string;
      outcome: InsertOutcome;
      timings: DictationTimings;
    }
  | { state: "discarded"; reason: "too_short" | "cancelled" | "no_speech" }
  | { state: "failed"; message: string };

/** A status that reports how a dictation ended. */
export type DictationResultStatus = Extract<
  DictationStatus,
  { state: "inserted" | "discarded" | "failed" }
>;

export type DictationOs = "macos" | "windows" | "linux";

export interface DictationPlatform {
  os: DictationOs;
  wayland: boolean;
  modifier_only_hotkeys: boolean;
  fn_key: boolean;
  default_hotkey: string;
  needs_keyboard_permission: boolean;
}

export type DictationModelState =
  | "ready"
  | "loading"
  | "not_downloaded"
  | "downloading"
  | "error"
  | "unknown";

export interface DictationSnapshot {
  settings: DictationSettings;
  status: DictationStatus;
  platform: DictationPlatform;
  permissions: { keyboard: boolean };
  hotkey_error: string | null;
  paste_last_hotkey_error: string | null;
  model: { id: string; state: DictationModelState; message: string | null };
  microphones: { name: string; is_default: boolean }[];
  server_is_remote: boolean;
}

export interface DictationHistoryEntry {
  id: string;
  at_ms: number;
  text: string;
  app: string | null;
  outcome: "pasted" | "typed" | "clipboard" | "failed";
  audio_ms: number;
  total_ms: number;
}

/** A change to some settings; `cleanup` may name only the fields it changes. */
export type DictationSettingsPatch = Partial<
  Omit<DictationSettings, "cleanup">
> & { cleanup?: Partial<DictationCleanupSettings> };

export const DEFAULT_DICTATION_MODEL = "Parakeet-TDT-0.6B-v3";

export function isResultStatus(
  status: DictationStatus,
): status is DictationResultStatus {
  return (
    status.state === "inserted" ||
    status.state === "discarded" ||
    status.state === "failed"
  );
}
