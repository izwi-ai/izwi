import { describe, expect, it } from "vitest";

import {
  EMPTY_RECORDER_STATE,
  buildHotkey,
  formatHotkey,
  hotkeyParts,
  isModifierOnlyHotkey,
  keyLabel,
  parseHotkey,
  recordHotkeyEvent,
  settleHotkeyRecording,
  type HotkeyRecorderOutcome,
} from "@/features/dictation/hotkey";

/** Feeds "down:Code" / "up:Code" events, and "wait" for the pause after the
 * keys are let go. Returns the outcome that ends the recording, or the last
 * one before it. */
function record(
  events: string[],
  modifierOnly = true,
): HotkeyRecorderOutcome | null {
  let state = EMPTY_RECORDER_STATE;
  let last: HotkeyRecorderOutcome | null = null;
  for (const entry of events) {
    const [direction, code] = entry.split(":");
    const step =
      entry === "wait"
        ? settleHotkeyRecording(state, { modifierOnly })
        : recordHotkeyEvent(state, {
            type: direction === "down" ? "keydown" : "keyup",
            code,
          });
    state = step.state;
    if (step.outcome) {
      last = step.outcome;
      if (last.kind === "captured" || last.kind === "cancelled") {
        return last;
      }
    }
  }
  return last;
}

describe("hotkey strings", () => {
  it("orders modifiers the same way whatever order they were pressed in", () => {
    expect(buildHotkey(["Shift", "Ctrl"], "Space")).toBe("Ctrl+Shift+Space");
    expect(buildHotkey(["Meta", "Alt", "Shift", "Ctrl"], "KeyD")).toBe(
      "Ctrl+Alt+Shift+Meta+KeyD",
    );
    expect(buildHotkey(["Alt"], null)).toBe("Alt");
  });

  it("parses modifiers and the key", () => {
    expect(parseHotkey("Ctrl+Shift+Space")).toEqual({
      modifiers: ["Ctrl", "Shift"],
      key: "Space",
    });
    expect(parseHotkey("Fn")).toEqual({ modifiers: ["Fn"], key: null });
    expect(isModifierOnlyHotkey("Ctrl+Shift")).toBe(true);
    expect(isModifierOnlyHotkey("Alt+Space")).toBe(false);
  });
});

describe("recording a hotkey", () => {
  it("captures a key pressed with modifiers", () => {
    expect(record(["down:AltLeft", "down:Space"])).toEqual({
      kind: "captured",
      hotkey: "Alt+Space",
    });
    expect(
      record(["down:ShiftRight", "down:ControlLeft", "down:KeyD"]),
    ).toEqual({ kind: "captured", hotkey: "Ctrl+Shift+KeyD" });
  });

  it("captures keys pressed one after another", () => {
    expect(
      record(["down:AltLeft", "up:AltLeft", "down:Space", "up:Space"]),
    ).toEqual({ kind: "captured", hotkey: "Alt+Space" });
    expect(
      record([
        "down:ControlLeft",
        "up:ControlLeft",
        "down:ShiftLeft",
        "up:ShiftLeft",
        "down:KeyD",
      ]),
    ).toEqual({ kind: "captured", hotkey: "Ctrl+Shift+KeyD" });
  });

  it("waits after modifiers are let go, then takes them alone where allowed", () => {
    expect(record(["down:AltRight", "up:AltRight"])).toEqual({
      kind: "waiting",
    });
    expect(record(["down:AltRight", "up:AltRight", "wait"])).toEqual({
      kind: "captured",
      hotkey: "Alt",
    });
    expect(
      record(["down:ControlLeft", "up:ControlLeft", "down:ShiftLeft", "up:ShiftLeft", "wait"]),
    ).toEqual({ kind: "captured", hotkey: "Ctrl+Shift" });
    // Nothing to settle while a key is down or before any was pressed.
    expect(record(["wait"])).toBeNull();
    expect(record(["down:AltLeft", "wait"])).toBeNull();
  });

  it("asks for a key where modifiers alone are not allowed, then starts over", () => {
    expect(record(["down:AltLeft", "up:AltLeft", "wait"], false)).toEqual({
      kind: "rejected",
      reason: "needs_key",
    });
    expect(
      record(
        ["down:AltLeft", "up:AltLeft", "wait", "down:MetaLeft", "down:KeyK"],
        false,
      ),
    ).toEqual({ kind: "captured", hotkey: "Meta+KeyK" });
  });

  it("refuses a plain key but takes a function key alone", () => {
    expect(record(["down:KeyA"])).toEqual({
      kind: "rejected",
      reason: "needs_modifier",
    });
    expect(record(["down:F5"])).toEqual({ kind: "captured", hotkey: "F5" });
  });

  it("drops a modifier that was released before the key", () => {
    expect(
      record(["down:ShiftLeft", "down:AltLeft", "up:ShiftLeft", "down:Space"]),
    ).toEqual({ kind: "captured", hotkey: "Alt+Space" });
  });

  it("cancels on Escape, even with modifiers held", () => {
    expect(record(["down:Escape"])).toEqual({ kind: "cancelled" });
    expect(record(["down:ControlLeft", "down:Escape"])).toEqual({
      kind: "cancelled",
    });
  });
});

describe("showing a hotkey", () => {
  it("uses glyphs on macOS and names elsewhere", () => {
    expect(formatHotkey("Alt+Space", "macos")).toBe("⌥ Space");
    expect(formatHotkey("Ctrl+Alt+Shift+Meta+KeyD", "macos")).toBe("⌃ ⌥ ⇧ ⌘ D");
    expect(formatHotkey("Ctrl+Shift+Space", "windows")).toBe("Ctrl+Shift+Space");
    expect(formatHotkey("Meta+Digit1", "windows")).toBe("Win+1");
    expect(formatHotkey("Meta+Digit1", "linux")).toBe("Super+1");
    expect(formatHotkey("Fn", "macos")).toBe("fn");
  });

  it("labels keys by what is printed on them", () => {
    expect(keyLabel("KeyD")).toBe("D");
    expect(keyLabel("Digit1")).toBe("1");
    expect(keyLabel("Slash")).toBe("/");
    expect(keyLabel("F5")).toBe("F5");
    expect(keyLabel("Numpad7")).toBe("Num 7");
    expect(hotkeyParts("Alt", "macos")).toEqual(["⌥"]);
  });
});
