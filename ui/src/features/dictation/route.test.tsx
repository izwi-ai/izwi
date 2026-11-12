import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { DictationPage } from "@/features/dictation/route";
import { dictationSettings, dictationSnapshot } from "@/features/dictation/testing";
import type { DictationSnapshot } from "@/features/dictation/types";

const tauri = vi.hoisted(() => ({
  desktop: true,
  invoke: vi.fn(),
}));

vi.mock("@tauri-apps/api/core", () => ({
  isTauri: () => tauri.desktop,
  invoke: tauri.invoke,
}));

vi.mock("@tauri-apps/api/event", () => ({
  listen: vi.fn(async () => () => undefined),
}));

function serve(snapshot: DictationSnapshot) {
  tauri.invoke.mockImplementation(async (command: string, args?: any) => {
    switch (command) {
      case "dictation_snapshot":
        return snapshot;
      case "dictation_update_settings":
        return { ...snapshot, settings: args.settings };
      case "dictation_history":
        return [];
      default:
        return undefined;
    }
  });
}

function renderPage(onDownload = vi.fn()) {
  render(
    <MemoryRouter>
      <DictationPage models={[]} downloadProgress={{}} onDownload={onDownload} />
    </MemoryRouter>,
  );
  return { onDownload };
}

describe("DictationPage", () => {
  beforeEach(() => {
    tauri.desktop = true;
    tauri.invoke.mockReset();
    window.localStorage.clear();
  });

  it("says dictation is part of the desktop app in the web UI", () => {
    tauri.desktop = false;
    renderPage();

    expect(
      screen.getByText("Dictation is part of the desktop app"),
    ).toBeInTheDocument();
    expect(tauri.invoke).not.toHaveBeenCalled();
  });

  it("shows how to dictate with the current shortcut", async () => {
    serve(dictationSnapshot());
    renderPage();

    expect(
      await screen.findByText("Ready — hold ⌥ Space and speak"),
    ).toBeInTheDocument();
  });

  it("offers the model download and the keyboard permission during setup", async () => {
    serve(
      dictationSnapshot({
        model: { id: "Parakeet-TDT-0.6B-v3", state: "not_downloaded", message: null },
        permissions: { keyboard: false },
      }),
    );
    const { onDownload } = renderPage();

    fireEvent.click(await screen.findByRole("button", { name: "Download" }));
    expect(onDownload).toHaveBeenCalledWith("Parakeet-TDT-0.6B-v3");

    fireEvent.click(screen.getByRole("button", { name: "Allow…" }));
    await waitFor(() =>
      expect(tauri.invoke).toHaveBeenCalledWith(
        "dictation_request_keyboard_permission",
      ),
    );
    expect(screen.getByRole("status")).toHaveTextContent(
      "Finish setup below to start dictating.",
    );
  });

  it("records a new shortcut and saves it", async () => {
    serve(dictationSnapshot());
    renderPage();

    fireEvent.click(
      await screen.findByRole("button", { name: /^Dictation shortcut:/ }),
    );
    fireEvent.keyDown(window, { code: "ControlLeft" });
    fireEvent.keyDown(window, { code: "KeyD" });

    await waitFor(() =>
      expect(tauri.invoke).toHaveBeenCalledWith("dictation_update_settings", {
        settings: dictationSettings({ hotkey: "Ctrl+KeyD" }),
      }),
    );
  });

  it("shows why a shortcut was refused and keeps the old one", async () => {
    const snapshot = dictationSnapshot();
    tauri.invoke.mockImplementation(async (command: string) => {
      if (command === "dictation_update_settings") {
        throw "That shortcut is used by another app.";
      }
      return command === "dictation_history" ? [] : snapshot;
    });
    renderPage();

    fireEvent.click(await screen.findByRole("button", { name: "Use Fn" }));

    expect(
      await screen.findByText("That shortcut is used by another app."),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /^Dictation shortcut: ⌥ Space/ }),
    ).toBeInTheDocument();
  });

  it("explains Wayland and remote servers only where they apply", async () => {
    const base = dictationSnapshot();
    serve(
      dictationSnapshot({
        platform: { ...base.platform, os: "linux", wayland: true },
        server_is_remote: true,
      }),
    );
    renderPage();

    expect(await screen.findByText("Wayland")).toBeInTheDocument();
    expect(screen.getByText("Remote server")).toBeInTheDocument();
    expect(screen.queryByText("Administrator window")).not.toBeInTheDocument();
  });
});
