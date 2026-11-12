import type {
  DictationSettings,
  DictationSnapshot,
} from "@/features/dictation/types";

// Fixtures for dictation tests.

export function dictationSettings(
  overrides: Partial<DictationSettings> = {},
): DictationSettings {
  return {
    enabled: true,
    hotkey: "Alt+Space",
    mode: "hold",
    double_tap_lock: true,
    paste_last_hotkey: null,
    model: null,
    language: null,
    microphone: null,
    keep_microphone_ready: false,
    show_indicator: true,
    sounds: true,
    mute_while_recording: false,
    insert_method: "paste",
    trailing_space: true,
    remove_fillers: false,
    replacements: [],
    vocabulary: [],
    live_preview: true,
    cleanup: { enabled: false, model: null, deadline_ms: 1500 },
    keep_history: true,
    ...overrides,
  };
}

export function dictationSnapshot(
  overrides: Partial<DictationSnapshot> = {},
): DictationSnapshot {
  return {
    settings: dictationSettings(),
    status: { state: "idle" },
    platform: {
      os: "macos",
      wayland: false,
      modifier_only_hotkeys: true,
      fn_key: true,
      default_hotkey: "Alt+Space",
      needs_keyboard_permission: true,
    },
    permissions: { keyboard: true },
    hotkey_error: null,
    paste_last_hotkey_error: null,
    model: { id: "Parakeet-TDT-0.6B-v3", state: "ready", message: null },
    microphones: [{ name: "MacBook Pro Microphone", is_default: true }],
    server_is_remote: false,
    ...overrides,
  };
}
