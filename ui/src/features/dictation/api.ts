import { invoke, isTauri } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

import type {
  DictationHistoryEntry,
  DictationSettings,
  DictationSnapshot,
  DictationStatus,
} from "@/features/dictation/types";

/** Dictation lives in the desktop app; the web UI has no commands for it. */
export function dictationAvailable(): boolean {
  return isTauri();
}

export type SystemSettingsPane = "accessibility" | "microphone";

export const dictationApi = {
  /** Also makes the app re-check a speech model that is not ready yet. */
  snapshot: () => invoke<DictationSnapshot>("dictation_snapshot"),
  /** Rejects with a readable string, e.g. for a hotkey that is taken. */
  updateSettings: (settings: DictationSettings) =>
    invoke<DictationSnapshot>("dictation_update_settings", { settings }),
  requestKeyboardPermission: () =>
    invoke<boolean>("dictation_request_keyboard_permission"),
  openSystemSettings: (pane: SystemSettingsPane) =>
    invoke<void>("dictation_open_system_settings", { pane }),
  /** Starts or stops a hands-free recording. */
  toggle: () => invoke<void>("dictation_toggle"),
  cancel: () => invoke<void>("dictation_cancel"),
  /** Newest first. */
  history: () => invoke<DictationHistoryEntry[]>("dictation_history"),
  clearHistory: () => invoke<void>("dictation_clear_history"),
  copyText: (text: string) => invoke<void>("dictation_copy_text", { text }),
  pasteLast: () => invoke<void>("dictation_paste_last"),
  /**
   * Suspends the global hotkeys while a new one is being recorded, so
   * pressing the current one does not start a dictation.
   */
  pauseHotkeys: (paused: boolean) =>
    invoke<void>("dictation_pause_hotkeys", { paused }),
};

function subscribe<T>(
  event: string,
  handler: (payload: T) => void,
): Promise<UnlistenFn> {
  return listen<T>(event, (message) => handler(message.payload));
}

export const dictationEvents = {
  status: (handler: (status: DictationStatus) => void) =>
    subscribe("dictation://status", handler),
  /** 0..1, about 20 times a second while listening. */
  level: (handler: (level: number) => void) =>
    subscribe("dictation://level", handler),
  /** Live words while speaking; may never fire. */
  partial: (handler: (text: string) => void) =>
    subscribe("dictation://partial", handler),
  /** Something other than the status changed. */
  snapshot: (handler: (snapshot: DictationSnapshot) => void) =>
    subscribe("dictation://snapshot", handler),
};

/** The message of a rejected command: Rust rejects with a plain string. */
export function commandErrorMessage(error: unknown, fallback: string): string {
  if (typeof error === "string" && error.trim()) {
    return error;
  }
  if (error instanceof Error && error.message.trim()) {
    return error.message;
  }
  return fallback;
}
