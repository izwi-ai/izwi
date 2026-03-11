import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { TranscriptionReviewWorkspace } from "./TranscriptionReviewWorkspace";

const record = {
  id: "txr-review-1",
  aligner_model_id: "Qwen3-ForcedAligner-0.6B",
  audio_filename: "meeting.wav",
  duration_secs: 6,
  language: "English",
  transcription: "Hello there. Follow up.",
  segments: [
    {
      start: 0,
      end: 2,
      text: "Hello there.",
      word_start: 0,
      word_end: 1,
    },
    {
      start: 2.5,
      end: 6,
      text: "Follow up.",
      word_start: 2,
      word_end: 3,
    },
  ],
  words: [
    { word: "Hello", start: 0, end: 0.8 },
    { word: "there.", start: 0.9, end: 2 },
    { word: "Follow", start: 2.5, end: 4 },
    { word: "up.", start: 4.1, end: 6 },
  ],
};

describe("TranscriptionReviewWorkspace", () => {
  beforeEach(() => {
    vi.spyOn(window.HTMLMediaElement.prototype, "pause").mockImplementation(
      () => {},
    );
  });

  it("supports transcript seeking and active-row highlighting", () => {
    const { container } = render(
      <TranscriptionReviewWorkspace
        record={record}
        audioUrl="/audio/meeting.wav"
      />,
    );

    expect(screen.getByText("Timestamps")).toBeInTheDocument();
    expect(screen.getByText("Enabled")).toBeInTheDocument();

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
    expect(audio!.currentTime).toBe(2.5);

    audio!.currentTime = 3;
    fireEvent.timeUpdate(audio!);

    expect(secondRow).toHaveAttribute("data-active", "true");
  });

  it("can render transcript cards without playback controls", () => {
    render(
      <TranscriptionReviewWorkspace
        record={record}
        showPlayback={false}
      />,
    );

    expect(
      screen.queryByRole("slider", { name: "Seek audio timeline" }),
    ).not.toBeInTheDocument();
    expect(screen.getByText("Follow up.")).toBeInTheDocument();
  });
});
