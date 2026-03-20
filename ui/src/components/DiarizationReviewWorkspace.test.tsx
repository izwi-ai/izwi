import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { DiarizationRecord } from "../api";
import { DiarizationReviewWorkspace } from "./DiarizationReviewWorkspace";

const record = {
  id: "diar-review-1",
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
  duration_secs: 6,
  rtf: 0.5,
  speaker_count: 2,
  corrected_speaker_count: 2,
  alignment_coverage: 1,
  unattributed_words: 0,
  llm_refined: false,
  asr_text: "Hello there. Follow up. Response.",
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
      speaker: "SPEAKER_00",
      start: 1,
      end: 3,
      text: "Follow up.",
      word_start: 2,
      word_end: 3,
    },
    {
      speaker: "SPEAKER_01",
      start: 3,
      end: 6,
      text: "Response.",
      word_start: 4,
      word_end: 5,
    },
  ],
  speaker_name_overrides: {
    SPEAKER_00: "Alice",
    SPEAKER_01: "Bob",
  },
  audio_mime_type: "audio/wav",
  audio_filename: "meeting.wav",
} satisfies DiarizationRecord;

describe("DiarizationReviewWorkspace", () => {
  beforeEach(() => {
    vi.spyOn(window.HTMLMediaElement.prototype, "pause").mockImplementation(
      () => {},
    );
  });

  it("supports transcript seeking and active-row highlighting", () => {
    const { container } = render(
      <DiarizationReviewWorkspace
        record={record}
        audioUrl="/audio/meeting.wav"
      />,
    );

    expect(screen.getByText("Talk Time")).toBeInTheDocument();
    expect(screen.getByText("2 turns • 0 words")).toBeInTheDocument();

    const audio = container.querySelector("audio");
    expect(audio).not.toBeNull();

    Object.defineProperty(audio, "duration", {
      configurable: true,
      value: 6,
    });
    Object.defineProperty(audio, "currentTime", {
      configurable: true,
      writable: true,
      value: 0,
    });

    fireEvent.loadedMetadata(audio!);

    const timeline = screen.getByRole("slider", {
      name: "Seek audio timeline",
    });
    fireEvent.change(timeline, {
      target: { value: "4.25" },
    });
    expect(audio!.currentTime).toBe(4.25);

    const secondRow = screen.getByText("Follow up.").closest("button");
    expect(secondRow).not.toBeNull();

    fireEvent.click(secondRow!);
    expect(audio!.currentTime).toBe(1);

    audio!.currentTime = 1.5;
    fireEvent.timeUpdate(audio!);

    expect(secondRow).toHaveAttribute("data-active", "true");
  });
});
