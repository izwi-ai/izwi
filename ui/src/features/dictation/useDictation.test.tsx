import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { dictationSettings, dictationSnapshot } from "@/features/dictation/testing";
import type { DictationStatus } from "@/features/dictation/types";
import { RESULT_LINGER_MS, useDictation } from "@/features/dictation/useDictation";

const tauri = vi.hoisted(() => ({
  desktop: true,
  invoke: vi.fn(),
  listeners: new Map<string, (event: { payload: unknown }) => void>(),
}));

vi.mock("@tauri-apps/api/core", () => ({
  isTauri: () => tauri.desktop,
  invoke: tauri.invoke,
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(
    async (event: string, handler: (event: { payload: unknown }) => void) => {
      tauri.listeners.set(event, handler);
      return () => tauri.listeners.delete(event);
    },
  ),
}));

function emit(event: string, payload: unknown) {
  act(() => tauri.listeners.get(event)?.({ payload }));
}

const timings = {
  audio_ms: 1000,
  capture_ms: 5,
  transcribe_ms: 120,
  cleanup_ms: 0,
  insert_ms: 10,
};

const inserted: DictationStatus = {
  state: "inserted",
  text: "hello there",
  outcome: { kind: "pasted" },
  timings,
};

async function renderLoaded() {
  const view = renderHook(() => useDictation());
  await waitFor(() => expect(view.result.current.snapshot).not.toBeNull());
  await waitFor(() => expect(tauri.listeners.size).toBe(3));
  return view;
}

describe("useDictation", () => {
  beforeEach(() => {
    tauri.desktop = true;
    tauri.listeners.clear();
    tauri.invoke.mockReset();
    tauri.invoke.mockImplementation(async (command: string, args?: any) => {
      switch (command) {
        case "dictation_snapshot":
          return dictationSnapshot();
        case "dictation_update_settings":
          return dictationSnapshot({ settings: args.settings });
        case "dictation_history":
          return [];
        default:
          return undefined;
      }
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("does nothing in the web UI", () => {
    tauri.desktop = false;
    const { result } = renderHook(() => useDictation());

    expect(result.current.available).toBe(false);
    expect(tauri.invoke).not.toHaveBeenCalled();
  });

  it("loads the snapshot and saves a change merged into the settings", async () => {
    const { result } = await renderLoaded();

    let saved = false;
    await act(async () => {
      saved = await result.current.update({ mode: "toggle" });
    });

    expect(saved).toBe(true);
    expect(tauri.invoke).toHaveBeenCalledWith("dictation_update_settings", {
      settings: dictationSettings({ mode: "toggle" }),
    });
    expect(result.current.snapshot?.settings.mode).toBe("toggle");
    expect(result.current.errors).toEqual({});
  });

  it("puts a refused change back and reports why next to it", async () => {
    const { result } = await renderLoaded();
    tauri.invoke.mockImplementation(async (command: string) => {
      if (command === "dictation_update_settings") {
        throw "That shortcut is already used by another app.";
      }
      return command === "dictation_history" ? [] : dictationSnapshot();
    });

    let saved = true;
    await act(async () => {
      saved = await result.current.update({ hotkey: "Ctrl+KeyC" });
    });

    expect(saved).toBe(false);
    expect(result.current.snapshot?.settings.hotkey).toBe("Alt+Space");
    expect(result.current.errors.hotkey).toBe(
      "That shortcut is already used by another app.",
    );
  });

  it("clears a setting's error when it is changed again", async () => {
    const { result } = await renderLoaded();
    tauri.invoke.mockRejectedValueOnce("Not a valid shortcut.");
    await act(async () => {
      await result.current.update({ hotkey: "KeyA" });
    });
    expect(result.current.errors.hotkey).toBe("Not a valid shortcut.");

    await act(async () => {
      await result.current.update({ hotkey: "Ctrl+KeyD" });
    });

    expect(result.current.errors.hotkey).toBeUndefined();
    expect(result.current.snapshot?.settings.hotkey).toBe("Ctrl+KeyD");
  });

  it("keeps a second edit when the first save is refused", async () => {
    const { result } = await renderLoaded();
    let refuse: (reason: string) => void = () => {};
    tauri.invoke.mockImplementation(async (command: string, args?: any) => {
      if (command !== "dictation_update_settings") {
        return command === "dictation_history" ? [] : dictationSnapshot();
      }
      if (args.settings.hotkey === "Ctrl+KeyC" && args.settings.sounds) {
        return new Promise((_, reject) => {
          refuse = reject;
        });
      }
      return dictationSnapshot({ settings: args.settings });
    });

    let first: Promise<boolean> = Promise.resolve(true);
    await act(async () => {
      first = result.current.update({ hotkey: "Ctrl+KeyC" });
      await result.current.update({ sounds: false });
    });
    await act(async () => {
      refuse("Taken.");
      await first;
    });

    expect(result.current.snapshot?.settings.hotkey).toBe("Alt+Space");
    expect(result.current.snapshot?.settings.sounds).toBe(false);
  });

  it("follows status, partial and snapshot events", async () => {
    const { result } = await renderLoaded();

    emit("dictation://status", { state: "listening", locked: false });
    emit("dictation://partial", "hello th");
    expect(result.current.snapshot?.status).toEqual({
      state: "listening",
      locked: false,
    });
    expect(result.current.partial).toBe("hello th");

    emit(
      "dictation://snapshot",
      dictationSnapshot({ hotkey_error: "Shortcut unavailable." }),
    );
    expect(result.current.snapshot?.hotkey_error).toBe("Shortcut unavailable.");
  });

  it("keeps the last result after idle, then lets it go", async () => {
    const { result } = await renderLoaded();
    vi.useFakeTimers();

    emit("dictation://status", inserted);
    emit("dictation://status", { state: "idle" });

    expect(result.current.snapshot?.status).toEqual({ state: "idle" });
    expect(result.current.lastResult).toEqual(inserted);

    act(() => {
      vi.advanceTimersByTime(RESULT_LINGER_MS);
    });
    expect(result.current.lastResult).toBeNull();
  });

  it("drops the last result as soon as a new dictation starts", async () => {
    const { result } = await renderLoaded();

    emit("dictation://status", { state: "failed", message: "No microphone" });
    emit("dictation://status", { state: "idle" });
    emit("dictation://status", { state: "starting" });

    expect(result.current.lastResult).toBeNull();
  });

  it("reloads the history after text is inserted", async () => {
    const { result } = await renderLoaded();
    const entry = {
      id: "1",
      at_ms: 1,
      text: "hello there",
      app: "Notes",
      outcome: "pasted" as const,
      audio_ms: 1000,
      total_ms: 1200,
    };
    tauri.invoke.mockImplementation(async (command: string) =>
      command === "dictation_history" ? [entry] : dictationSnapshot(),
    );

    emit("dictation://status", inserted);

    await waitFor(() => expect(result.current.history).toEqual([entry]));
  });

  it("reads the snapshot again when the window regains focus", async () => {
    const { result } = await renderLoaded();
    tauri.invoke.mockImplementation(async (command: string) =>
      command === "dictation_history"
        ? []
        : dictationSnapshot({ permissions: { keyboard: false } }),
    );

    act(() => {
      window.dispatchEvent(new Event("focus"));
    });

    await waitFor(() =>
      expect(result.current.snapshot?.permissions.keyboard).toBe(false),
    );
  });
});
