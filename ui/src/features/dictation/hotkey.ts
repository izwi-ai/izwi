import type { DictationOs } from "@/features/dictation/types";

// Hotkey strings: modifiers (Ctrl, Alt, Shift, Meta, Fn) joined with "+",
// then at most one key named by KeyboardEvent.code. No key at all is a
// modifier-only shortcut ("Alt", "Ctrl+Shift").

export type HotkeyModifier = "Ctrl" | "Alt" | "Shift" | "Meta" | "Fn";

const MODIFIER_ORDER: HotkeyModifier[] = ["Ctrl", "Alt", "Shift", "Meta", "Fn"];

const MODIFIER_CODES: Record<string, HotkeyModifier> = {
  ControlLeft: "Ctrl",
  ControlRight: "Ctrl",
  AltLeft: "Alt",
  AltRight: "Alt",
  ShiftLeft: "Shift",
  ShiftRight: "Shift",
  MetaLeft: "Meta",
  MetaRight: "Meta",
  OSLeft: "Meta",
  OSRight: "Meta",
};

export interface ParsedHotkey {
  modifiers: HotkeyModifier[];
  key: string | null;
}

function isModifier(token: string): token is HotkeyModifier {
  return (MODIFIER_ORDER as string[]).includes(token);
}

/** The hotkey string for these modifiers (in any order) and key. */
export function buildHotkey(
  modifiers: Iterable<HotkeyModifier>,
  key: string | null,
): string {
  const held = new Set(modifiers);
  const parts: string[] = MODIFIER_ORDER.filter((modifier) =>
    held.has(modifier),
  );
  if (key) {
    parts.push(key);
  }
  return parts.join("+");
}

export function parseHotkey(hotkey: string): ParsedHotkey {
  const modifiers: HotkeyModifier[] = [];
  let key: string | null = null;
  for (const token of hotkey.split("+").map((part) => part.trim())) {
    if (!token) {
      continue;
    }
    if (isModifier(token)) {
      modifiers.push(token);
    } else {
      key = token;
    }
  }
  return {
    modifiers: MODIFIER_ORDER.filter((modifier) => modifiers.includes(modifier)),
    key,
  };
}

export function isModifierOnlyHotkey(hotkey: string): boolean {
  const parsed = parseHotkey(hotkey);
  return parsed.key === null && parsed.modifiers.length > 0;
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

export interface HotkeyRecorderState {
  /** Modifiers down right now. */
  held: HotkeyModifier[];
  /** Every modifier pressed since the keyboard was last fully released. */
  pressed: HotkeyModifier[];
}

export type HotkeyRecorderOutcome =
  | { kind: "captured"; hotkey: string }
  | { kind: "cancelled" }
  /** The keys so far cannot be a shortcut; recording goes on. */
  | { kind: "rejected"; reason: "needs_modifier" | "needs_key" };

export interface HotkeyRecorderStep {
  state: HotkeyRecorderState;
  outcome: HotkeyRecorderOutcome | null;
}

export const EMPTY_RECORDER_STATE: HotkeyRecorderState = {
  held: [],
  pressed: [],
};

const FUNCTION_KEY = /^F\d{1,2}$/;

/**
 * Feeds one key event to a recording. A key pressed with modifiers (or a
 * function key alone) is captured at once; modifiers released with no key
 * are captured as a modifier-only shortcut where the platform allows it.
 * Escape cancels.
 */
export function recordHotkeyEvent(
  state: HotkeyRecorderState,
  event: { type: "keydown" | "keyup"; code: string },
  options: { modifierOnly: boolean },
): HotkeyRecorderStep {
  const modifier = MODIFIER_CODES[event.code];

  if (event.type === "keydown") {
    if (event.code === "Escape") {
      return { state: EMPTY_RECORDER_STATE, outcome: { kind: "cancelled" } };
    }
    if (modifier) {
      return {
        state: {
          held: state.held.includes(modifier)
            ? state.held
            : [...state.held, modifier],
          pressed: state.pressed.includes(modifier)
            ? state.pressed
            : [...state.pressed, modifier],
        },
        outcome: null,
      };
    }
    if (!event.code) {
      return { state, outcome: null };
    }
    if (state.held.length === 0 && !FUNCTION_KEY.test(event.code)) {
      return { state, outcome: { kind: "rejected", reason: "needs_modifier" } };
    }
    return {
      state: EMPTY_RECORDER_STATE,
      outcome: { kind: "captured", hotkey: buildHotkey(state.held, event.code) },
    };
  }

  if (!modifier) {
    return { state, outcome: null };
  }
  const held = state.held.filter((entry) => entry !== modifier);
  if (held.length > 0) {
    return { state: { held, pressed: state.pressed }, outcome: null };
  }
  if (state.pressed.length === 0) {
    return { state: EMPTY_RECORDER_STATE, outcome: null };
  }
  return {
    state: EMPTY_RECORDER_STATE,
    outcome: options.modifierOnly
      ? { kind: "captured", hotkey: buildHotkey(state.pressed, null) }
      : { kind: "rejected", reason: "needs_key" },
  };
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

const MAC_MODIFIER_LABELS: Record<HotkeyModifier, string> = {
  Ctrl: "⌃",
  Alt: "⌥",
  Shift: "⇧",
  Meta: "⌘",
  Fn: "fn",
};

const KEY_LABELS: Record<string, string> = {
  Space: "Space",
  Enter: "Enter",
  NumpadEnter: "Num Enter",
  Escape: "Esc",
  Backspace: "Backspace",
  Delete: "Delete",
  ArrowUp: "↑",
  ArrowDown: "↓",
  ArrowLeft: "←",
  ArrowRight: "→",
  Slash: "/",
  Backslash: "\\",
  Comma: ",",
  Period: ".",
  Semicolon: ";",
  Quote: "'",
  BracketLeft: "[",
  BracketRight: "]",
  Minus: "-",
  Equal: "=",
  Backquote: "`",
  NumpadAdd: "Num +",
  NumpadSubtract: "Num -",
  NumpadMultiply: "Num *",
  NumpadDivide: "Num /",
  NumpadDecimal: "Num .",
  CapsLock: "Caps Lock",
  PageUp: "Page Up",
  PageDown: "Page Down",
};

function modifierLabel(modifier: HotkeyModifier, os: DictationOs): string {
  if (os === "macos") {
    return MAC_MODIFIER_LABELS[modifier];
  }
  if (modifier === "Meta") {
    return os === "windows" ? "Win" : "Super";
  }
  return modifier;
}

/** A key's label: `KeyD` is "D", `Digit1` is "1", `Slash` is "/". */
export function keyLabel(code: string): string {
  if (KEY_LABELS[code]) {
    return KEY_LABELS[code];
  }
  const letter = /^Key([A-Z])$/.exec(code);
  if (letter) {
    return letter[1];
  }
  const digit = /^Digit(\d)$/.exec(code);
  if (digit) {
    return digit[1];
  }
  const numpad = /^Numpad(\d)$/.exec(code);
  if (numpad) {
    return `Num ${numpad[1]}`;
  }
  return code;
}

/** The labels of a hotkey's keys, in order: ["⌥", "Space"]. */
export function hotkeyParts(hotkey: string, os: DictationOs): string[] {
  const { modifiers, key } = parseHotkey(hotkey);
  const parts = modifiers.map((modifier) => modifierLabel(modifier, os));
  if (key) {
    parts.push(keyLabel(key));
  }
  return parts;
}

/** "⌥ Space" on macOS, "Alt+Space" elsewhere. */
export function formatHotkey(hotkey: string, os: DictationOs): string {
  return hotkeyParts(hotkey, os).join(os === "macos" ? " " : "+");
}
