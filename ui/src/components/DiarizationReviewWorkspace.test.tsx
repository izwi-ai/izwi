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
  processing_status: "ready" as const,
  processing_error: null,
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
  summary_status: "ready" as const,
  summary_model_id: "Qwen3.5-4B",
  summary_text: "Alice greets, then continues the discussion before Bob responds.",
  summary_error: null,
  summary_updated_at: 1_711_728_000_000,
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

const lowConfidenceRecord = {
  ...record,
  words: [
    {
      word: "Hello",
      speaker: "SPEAKER_00",
      start: 0,
      end: 0.4,
      speaker_confidence: 0.95,
      overlaps_segment: true,
    },
    {
      word: "there",
      speaker: "SPEAKER_00",
      start: 0.4,
      end: 0.9,
      speaker_confidence: 0.9,
      overlaps_segment: true,
    },
    {
      word: "Follow",
      speaker: "SPEAKER_00",
      start: 1,
      end: 2,
      speaker_confidence: 0.45,
      overlaps_segment: true,
    },
    {
      word: "up",
      speaker: "SPEAKER_00",
      start: 2,
      end: 2.9,
      speaker_confidence: 0.48,
      overlaps_segment: true,
    },
    {
      word: "Response",
      speaker: "SPEAKER_01",
      start: 3,
      end: 4.5,
      speaker_confidence: 0.93,
      overlaps_segment: false,
    },
    {
      word: ".",
      speaker: "SPEAKER_01",
      start: 4.5,
      end: 6,
      speaker_confidence: 0.9,
      overlaps_segment: true,
    },
  ],
} satisfies DiarizationRecord;

describe("DiarizationReviewWorkspace", () => {
  beforeEach(() => {
    vi.spyOn(window.HTMLMediaElement.prototype, "pause").mockImplementation(
      () => {},
    );
    HTMLElement.prototype.scrollIntoView = vi.fn();
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

  it("renders summary content above transcript entries", () => {
    render(
      <DiarizationReviewWorkspace
        record={record}
        audioUrl="/audio/meeting.wav"
      />,
    );

    const summaryHeading = screen.getByRole("heading", { name: "Summary" });
    const transcriptHeading = screen.getByRole("heading", { name: "Transcript" });
    expect(
      summaryHeading.compareDocumentPosition(transcriptHeading) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      screen.getByText(
        "Alice greets, then continues the discussion before Bob responds.",
      ),
    ).toBeInTheDocument();
  });

  it("auto-scrolls the active transcript entry into view during playback when enabled", () => {
    const { container } = render(
      <DiarizationReviewWorkspace
        record={record}
        audioUrl="/audio/meeting.wav"
        autoScrollActiveEntry={true}
      />,
    );

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
    fireEvent.play(audio!);

    const scrollSpy = vi.spyOn(HTMLElement.prototype, "scrollIntoView");
    scrollSpy.mockClear();

    audio!.currentTime = 4.2;
    fireEvent.timeUpdate(audio!);

    expect(screen.getByText("Response.").closest("button")).toHaveAttribute(
      "data-active",
      "true",
    );
    expect(scrollSpy).toHaveBeenCalledWith({
      behavior: "smooth",
      block: "center",
      inline: "nearest",
    });
  });

  it("can pin playback controls in a fixed footer for record detail pages", () => {
    render(
      <DiarizationReviewWorkspace
        record={record}
        audioUrl="/audio/meeting.wav"
        fixedPlaybackFooter={true}
      />,
    );

    expect(screen.getByTestId("diarization-review-player")).toHaveClass("fixed");
    expect(screen.getByTestId("diarization-review-player")).toHaveClass(
      "lg:left-[var(--app-shell-left)]",
    );
  });

  it("flags uncertain turns and supports quick-jump navigation", () => {
    const { container } = render(
      <DiarizationReviewWorkspace
        record={lowConfidenceRecord}
        audioUrl="/audio/meeting.wav"
      />,
    );

    const audio = container.querySelector("audio");
    expect(audio).not.toBeNull();
    Object.defineProperty(audio, "currentTime", {
      configurable: true,
      writable: true,
      value: 0,
    });

    expect(screen.getByTestId("diarization-confidence-nav")).toBeInTheDocument();
    expect(screen.getByText("2 flagged")).toBeInTheDocument();
    expect(
      screen.getByText("Average speaker confidence 47%."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("1 word drifted from segment boundaries."),
    ).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Jump to flagged turn 1" }),
    );
    expect(audio!.currentTime).toBe(1);
  });
});
