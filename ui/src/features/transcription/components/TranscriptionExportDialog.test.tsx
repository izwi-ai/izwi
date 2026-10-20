import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const tauri = vi.hoisted(() => ({
  isTauri: vi.fn(() => false),
  invoke: vi.fn(),
}));
vi.mock("@tauri-apps/api/core", () => tauri);

import type { TranscriptionRecord } from "@/api";
import { TranscriptionExportDialog } from "@/features/transcription/components/TranscriptionExportDialog";

const record = {
  id: "transcription-export-dialog-1",
  created_at: 1,
  model_id: "Parakeet-TDT-0.6B-v3",
  aligner_model_id: "Qwen3-ForcedAligner-0.6B",
  language: "English",
  processing_status: "ready",
  processing_error: null,
  duration_secs: 2.5,
  processing_time_ms: 120,
  rtf: 0.5,
  audio_mime_type: "audio/wav",
  audio_filename: "meeting.wav",
  transcription: "Hello there. Hi back.",
  segments: [
    {
      start: 0,
      end: 1.25,
      text: "Hello there.",
      word_start: 0,
      word_end: 1,
    },
    {
      start: 1.25,
      end: 2.5,
      text: "Hi back.",
      word_start: 2,
      word_end: 3,
    },
  ],
  words: [
    { word: "Hello", start: 0, end: 0.5 },
    { word: "there.", start: 0.55, end: 1.25 },
    { word: "Hi", start: 1.25, end: 1.65 },
    { word: "back.", start: 1.7, end: 2.5 },
  ],
} satisfies TranscriptionRecord;

beforeEach(() => {
  tauri.isTauri.mockReturnValue(false);
  tauri.invoke.mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

function openDialog() {
  render(
    <TranscriptionExportDialog record={record}>
      <button type="button">Open export</button>
    </TranscriptionExportDialog>,
  );
  fireEvent.click(screen.getByRole("button", { name: "Open export" }));
}

describe("TranscriptionExportDialog", () => {
  it("downloads the default text export in a browser", async () => {
    const createObjectUrl = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:transcription-export");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    const clickSpy = vi
      .spyOn(window.HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    openDialog();
    expect(screen.queryByText("You choose where it's saved.")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /Export…/ }));

    await waitFor(() => expect(clickSpy).toHaveBeenCalledTimes(1));
    const blob = createObjectUrl.mock.calls[0]?.[0];
    expect(blob).toBeInstanceOf(Blob);
    await expect((blob as Blob).text()).resolves.toContain(
      "[00:00.000 - 00:01.250] Hello there.",
    );
    expect(
      (clickSpy.mock.instances[0] as unknown as HTMLAnchorElement).download,
    ).toBe("meeting.txt");
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("saves through the desktop app's save dialog", async () => {
    tauri.isTauri.mockReturnValue(true);
    tauri.invoke.mockResolvedValue("/Users/me/meeting.srt");

    openDialog();
    expect(screen.getByText("You choose where it's saved.")).toBeTruthy();
    fireEvent.click(screen.getByRole("radio", { name: "SRT" }));
    fireEvent.click(screen.getByRole("button", { name: /Export…/ }));

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    const [command, body, options] = tauri.invoke.mock.calls[0] ?? [];
    expect(command).toBe("save_file");
    expect(new TextDecoder().decode(body as Uint8Array)).toContain(
      "00:00:00,000 --> 00:00:01,250",
    );
    expect(options).toEqual({ headers: { "x-izwi-filename": "meeting.srt" } });
  });

  it("stays open when the save dialog is cancelled", async () => {
    tauri.isTauri.mockReturnValue(true);
    tauri.invoke.mockResolvedValue(null);

    openDialog();
    fireEvent.click(screen.getByRole("button", { name: /Export…/ }));

    await waitFor(() => expect(tauri.invoke).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /Export…/ }).hasAttribute("disabled"),
      ).toBe(false),
    );
    expect(screen.getByRole("dialog")).toBeTruthy();
  });

  it("shows why a save failed", async () => {
    tauri.isTauri.mockReturnValue(true);
    tauri.invoke.mockRejectedValue(new Error("failed writing /Volumes/USB/meeting.txt"));

    openDialog();
    fireEvent.click(screen.getByRole("button", { name: /Export…/ }));

    expect((await screen.findByRole("alert")).textContent).toContain(
      "failed writing /Volumes/USB/meeting.txt",
    );
    expect(screen.getByRole("dialog")).toBeTruthy();
  });

  it("offers details only for text and JSON files", () => {
    openDialog();
    const details = screen.getByRole("switch", { name: /Details/ });
    expect(details.hasAttribute("disabled")).toBe(false);

    fireEvent.click(screen.getByRole("radio", { name: "VTT" }));
    expect(
      screen.getByRole("switch", { name: /Details/ }).hasAttribute("disabled"),
    ).toBe(true);
    expect(screen.getByText("Only in Text and JSON files")).toBeTruthy();
  });
});
