import { describe, expect, it } from "vitest";

import type { DictationStatus } from "@/features/dictation/types";
import {
  HIDDEN_VIEW,
  nextIndicatorView,
  pushLevel,
  type IndicatorView,
} from "@/indicator/view";

const timings = {
  audio_ms: 1000,
  capture_ms: 5,
  transcribe_ms: 120,
  cleanup_ms: 0,
  insert_ms: 10,
};

function after(statuses: DictationStatus[], os: "macos" | "windows" = "macos") {
  return statuses.reduce<IndicatorView>(
    (view, status) => nextIndicatorView(view, status, os),
    HIDDEN_VIEW,
  );
}

describe("nextIndicatorView", () => {
  it("follows a dictation from start to transcribing", () => {
    expect(after([{ state: "starting" }])).toEqual({ kind: "starting" });
    expect(
      after([{ state: "starting" }, { state: "listening", locked: true }]),
    ).toEqual({ kind: "listening", locked: true });
    expect(after([{ state: "transcribing" }])).toEqual({ kind: "transcribing" });
  });

  it("keeps a notice when idle follows it", () => {
    const copied: DictationStatus = {
      state: "inserted",
      text: "hello",
      outcome: { kind: "clipboard", reason: "not_permitted", detail: null },
      timings,
    };

    expect(after([copied, { state: "idle" }])).toEqual({
      kind: "notice",
      text: "Copied — paste with ⌘V",
      tone: "neutral",
    });
    expect(after([copied, { state: "idle" }], "windows")).toMatchObject({
      text: "Copied — paste with Ctrl+V",
    });
    expect(
      after([{ state: "failed", message: "No microphone" }, { state: "idle" }]),
    ).toEqual({ kind: "notice", text: "No microphone", tone: "danger" });
    expect(
      after([{ state: "discarded", reason: "no_speech" }, { state: "idle" }]),
    ).toEqual({ kind: "notice", text: "Didn't catch that", tone: "neutral" });
  });

  it("replaces a notice when a new dictation starts", () => {
    expect(
      after([
        { state: "failed", message: "No microphone" },
        { state: "idle" },
        { state: "starting" },
      ]),
    ).toEqual({ kind: "starting" });
  });

  it("shows nothing for a cancelled or too-short dictation", () => {
    expect(
      after([
        { state: "listening", locked: false },
        { state: "discarded", reason: "cancelled" },
        { state: "idle" },
      ]),
    ).toEqual(HIDDEN_VIEW);
    expect(after([{ state: "listening", locked: false }, { state: "idle" }])).toEqual(
      HIDDEN_VIEW,
    );
  });

  it("marks pasted text as done", () => {
    expect(
      after([
        { state: "inserted", text: "hi", outcome: { kind: "pasted" }, timings },
        { state: "idle" },
      ]),
    ).toEqual({ kind: "done" });
  });
});

describe("pushLevel", () => {
  it("keeps the newest levels, clamped", () => {
    expect(pushLevel([0.5, 0.5], 4, 2)).toEqual([0.5, 1]);
    expect(pushLevel([], -1, 5)).toEqual([0]);
    expect(pushLevel([], Number.NaN, 5)).toEqual([0]);
  });
});
