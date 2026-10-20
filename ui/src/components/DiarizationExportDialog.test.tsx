import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const tauri = vi.hoisted(() => ({
  isTauri: vi.fn(() => false),
  invoke: vi.fn(),
}));
vi.mock("@tauri-apps/api/core", () => tauri);

import type { DiarizationRecord } from "../api";
import { DiarizationExportDialog } from "./DiarizationExportDialog";

const record = {
  id: "diar-export-dialog-1",
  created_at: 1,
  model_id: "diar_streaming_sortformer_4spk-v2.1",
  asr_model_id: "Parakeet-TDT-0.6B-v3",
  aligner_model_id: "Qwen3-ForcedAligner-0.6B",
  llm_model_id: null,
  processing_status: "ready" as const,
  processing_error: null,
  min_speakers: 1,
  max_speakers: 4,
  min_speech_duration_ms: 240,
  min_silence_duration_ms: 200,
  processing_time_ms: 120,
  duration_secs: 6.4,
  rtf: 0.5,
  speaker_count: 2,
  corrected_speaker_count: 2,
  alignment_coverage: 1,
  unattributed_words: 0,
  asr_text: "Hello there. Hi back.",
  raw_transcript: "",
  transcript: "",
  segments: [],
  words: [],
  utterances: [
    {
      speaker: "SPEAKER_00",
      start: 0,
      end: 1,
      text: "Hello there.",
      word_start: 0,
      word_end: 1,
    },
    {
      speaker: "SPEAKER_01",
      start: 1,
      end: 2,
      text: "Hi back.",
      word_start: 2,
      word_end: 3,
    },
  ],
  speaker_name_overrides: {
    SPEAKER_00: "Alice",
    SPEAKER_01: "Bob",
  },
  audio_mime_type: "audio/wav",
  audio_filename: "meeting.wav",
} satisfies DiarizationRecord;

beforeEach(() => {
  tauri.isTauri.mockReturnValue(false);
  tauri.invoke.mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
});

function openDialog(source: DiarizationRecord = record) {
  render(
    <DiarizationExportDialog record={source}>
      <button type="button">Open export</button>
    </DiarizationExportDialog>,
  );
  fireEvent.click(screen.getByRole("button", { name: "Open export" }));
}

async function exportedText(): Promise<string> {
  fireEvent.click(screen.getByRole("button", { name: /Export…/ }));
  await waitFor(() => expect(tauri.invoke).toHaveBeenCalledTimes(1));
  return new TextDecoder().decode(tauri.invoke.mock.calls[0]?.[1] as Uint8Array);
}

describe("DiarizationExportDialog", () => {
  it("downloads the default text export with corrected speaker labels", async () => {
    const createObjectUrl = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:diarization-export");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    const clickSpy = vi
      .spyOn(window.HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    openDialog();
    fireEvent.click(screen.getByRole("button", { name: /Export…/ }));

    await waitFor(() => expect(clickSpy).toHaveBeenCalledTimes(1));
    const blob = createObjectUrl.mock.calls[0]?.[0];
    expect(blob).toBeInstanceOf(Blob);
    await expect((blob as Blob).text()).resolves.toContain("Alice [0.00s - 1.00s]");
  });

  it("exports the labels diarization gave when speaker names are off", async () => {
    tauri.isTauri.mockReturnValue(true);
    tauri.invoke.mockResolvedValue("/Users/me/meeting.txt");

    openDialog();
    const names = screen.getByRole("switch", { name: /Speaker names/ });
    expect(names.getAttribute("aria-checked")).toBe("true");
    fireEvent.click(names);

    const text = await exportedText();
    expect(text).toContain("SPEAKER_00 [0.00s - 1.00s]");
    expect(text).not.toContain("Alice");
  });

  it("puts the speaker names in captions", async () => {
    tauri.isTauri.mockReturnValue(true);
    tauri.invoke.mockResolvedValue("/Users/me/meeting.vtt");

    openDialog();
    fireEvent.click(screen.getByRole("radio", { name: "VTT" }));

    const text = await exportedText();
    expect(text.startsWith("WEBVTT")).toBe(true);
    expect(text).toContain("Bob");
    expect(tauri.invoke.mock.calls[0]?.[2]).toEqual({
      headers: { "x-izwi-filename": "meeting.vtt" },
    });
  });

  it("hides the speaker names option when no speaker was named", () => {
    openDialog({ ...record, speaker_name_overrides: {} });
    expect(screen.queryByRole("switch", { name: /Speaker names/ })).toBeNull();
    expect(screen.getByRole("switch", { name: /Details/ })).toBeTruthy();
  });
});
