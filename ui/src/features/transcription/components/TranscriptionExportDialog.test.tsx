import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { TranscriptionRecord } from "@/api";
import { TranscriptionExportDialog } from "@/features/transcription/components/TranscriptionExportDialog";

const record = {
  id: "transcription-export-dialog-1",
  created_at: 1,
  model_id: "Qwen3-ASR-0.6B",
  aligner_model_id: "Qwen3-ForcedAligner-0.6B",
  language: "English",
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

afterEach(() => {
  vi.restoreAllMocks();
});

describe("TranscriptionExportDialog", () => {
  it("downloads the default txt export", async () => {
    const createObjectUrl = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:transcription-export");
    const revokeObjectUrl = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => {});
    const clickSpy = vi
      .spyOn(window.HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    render(
      <TranscriptionExportDialog record={record}>
        <button type="button">Open export</button>
      </TranscriptionExportDialog>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Open export" }));
    fireEvent.click(screen.getByRole("button", { name: "Export file" }));

    expect(createObjectUrl).toHaveBeenCalledTimes(1);
    const blob = createObjectUrl.mock.calls[0]?.[0];
    expect(blob).toBeInstanceOf(Blob);
    await expect((blob as Blob).text()).resolves.toContain(
      "[00:00.000 - 00:01.250] Hello there.",
    );
    expect(clickSpy).toHaveBeenCalledTimes(1);
    expect(revokeObjectUrl).toHaveBeenCalledWith("blob:transcription-export");
  });
});
