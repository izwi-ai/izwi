import { describe, expect, it } from "vitest";

import {
  mergeDictationSettings,
  normalizeVocabulary,
  revertPatch,
} from "@/features/dictation/settings";
import { dictationSettings } from "@/features/dictation/testing";

describe("mergeDictationSettings", () => {
  it("changes only what the patch names", () => {
    const current = dictationSettings();
    const next = mergeDictationSettings(current, { hotkey: "Ctrl+KeyD" });

    expect(next).toEqual({ ...current, hotkey: "Ctrl+KeyD" });
    expect(current.hotkey).toBe("Alt+Space");
  });

  it("merges clean-up settings field by field", () => {
    const current = dictationSettings({
      cleanup: { enabled: false, model: "Qwen3.5-4B", deadline_ms: 2000 },
    });

    expect(
      mergeDictationSettings(current, { cleanup: { enabled: true } }).cleanup,
    ).toEqual({ enabled: true, model: "Qwen3.5-4B", deadline_ms: 2000 });
  });

  it("can set a value back to null", () => {
    const current = dictationSettings({ microphone: "USB Mic" });

    expect(
      mergeDictationSettings(current, { microphone: null }).microphone,
    ).toBeNull();
  });
});

describe("revertPatch", () => {
  it("restores what a patch changed and nothing else", () => {
    const previous = dictationSettings();
    const patch = { hotkey: "Ctrl+KeyD", cleanup: { enabled: true } };
    // A later edit that must survive the revert.
    const later = mergeDictationSettings(
      mergeDictationSettings(previous, patch),
      { sounds: false },
    );

    const reverted = mergeDictationSettings(later, revertPatch(previous, patch));

    expect(reverted).toEqual({ ...previous, sounds: false });
  });
});

describe("normalizeVocabulary", () => {
  it("trims, drops blanks and removes repeats", () => {
    expect(normalizeVocabulary([" Izwi ", "", "izwi", "Parakeet"])).toEqual([
      "Izwi",
      "Parakeet",
    ]);
  });
});
