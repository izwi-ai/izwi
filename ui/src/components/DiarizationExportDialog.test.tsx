import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { DiarizationRecord } from "../api";
import { DiarizationExportDialog } from "./DiarizationExportDialog";

const record = {
  id: "diar-export-dialog-1",
  created_at: 1,
  model_id: "diar_streaming_sortformer_4spk-v2.1",
  asr_model_id: "Parakeet-TDT-0.6B-v3",
  aligner_model_id: "Qwen3-ForcedAligner-0.6B",
  llm_model_id: null,
  min_speakers: 1,
  max_speakers: 4,
  min_speech_duration_ms: 240,
  min_silence_duration_ms: 200,
  enable_llm_refinement: false,
  processing_time_ms: 120,
  duration_secs: 6.4,
  rtf: 0.5,
  speaker_count: 2,
  corrected_speaker_count: 2,
  alignment_coverage: 1,
  unattributed_words: 0,
  llm_refined: false,
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

afterEach(() => {
  vi.restoreAllMocks();
});

describe("DiarizationExportDialog", () => {
  it("downloads the default TXT export with corrected speaker labels", async () => {
    const createObjectUrl = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:diarization-export");
    const revokeObjectUrl = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => {});
    const clickSpy = vi
      .spyOn(window.HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});

    render(
      <DiarizationExportDialog record={record}>
        <button type="button">Open export</button>
      </DiarizationExportDialog>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Open export" }));
    fireEvent.click(screen.getByRole("button", { name: "Export file" }));

    expect(createObjectUrl).toHaveBeenCalledTimes(1);
    const blob = createObjectUrl.mock.calls[0]?.[0];
    expect(blob).toBeInstanceOf(Blob);
    await expect((blob as Blob).text()).resolves.toContain("Alice [0.00s - 1.00s]");
    expect(clickSpy).toHaveBeenCalledTimes(1);
    expect(revokeObjectUrl).toHaveBeenCalledWith("blob:diarization-export");
  });
});
