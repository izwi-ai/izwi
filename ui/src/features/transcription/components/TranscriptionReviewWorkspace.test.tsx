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

const recordWithSummary = {
  ...record,
  summary_status: "ready" as const,
  summary_model_id: "Qwen3.5-4B",
  summary_text:
    "Aaron is worried about review load and is trying to keep up with assignments.",
  summary_error: null,
  summary_updated_at: 1_711_728_000_000,
};

describe("TranscriptionReviewWorkspace", () => {
  beforeEach(() => {
    vi.spyOn(window.HTMLMediaElement.prototype, "pause").mockImplementation(
      () => {},
    );
    HTMLElement.prototype.scrollIntoView = vi.fn();
  });

  it("supports transcript seeking and active-row highlighting", () => {
    const { container } = render(
      <TranscriptionReviewWorkspace
        record={record}
        audioUrl="/audio/meeting.wav"
      />,
    );

    expect(screen.getByText("Timestamps")).toBeInTheDocument();
    expect(screen.getByText("Word-level")).toBeInTheDocument();

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

  it("renders the summary section above transcript content", () => {
    render(
      <TranscriptionReviewWorkspace
        record={recordWithSummary}
        showPlayback={false}
      />,
    );

    const summaryHeading = screen.getByRole("heading", { name: "Summary" });
    const transcriptBodyText = screen.getByText("Hello there.");

    expect(
      summaryHeading.compareDocumentPosition(transcriptBodyText) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      screen.getByText(
        "Aaron is worried about review load and is trying to keep up with assignments.",
      ),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "Transcript" }),
    ).not.toBeInTheDocument();
    expect(screen.getAllByRole("heading", { name: "Summary" })).toHaveLength(1);
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

  it("shows summary model guidance when summary generation is unavailable", () => {
    render(
      <TranscriptionReviewWorkspace
        record={record}
        showPlayback={false}
        summaryModelGuidance="Download and load Qwen3.5-4B in Transcription Models to generate summaries."
      />,
    );

    expect(
      screen.getByText(
        "Download and load Qwen3.5-4B in Transcription Models to generate summaries.",
      ),
    ).toBeInTheDocument();
  });

  it("keeps the playback controls pinned in the sticky footer", () => {
    render(
      <TranscriptionReviewWorkspace
        record={record}
        audioUrl="/audio/meeting.wav"
        stickyPlaybackFooter={true}
      />,
    );

    expect(screen.getByTestId("transcription-review-player")).toHaveClass(
      "sticky",
    );
  });

  it("keeps the fixed playback footer clear of the expanded sidebar", () => {
    render(
      <TranscriptionReviewWorkspace
        record={record}
        audioUrl="/audio/meeting.wav"
        fixedPlaybackFooter={true}
      />,
    );

    expect(screen.getByTestId("transcription-review-player")).toHaveClass(
      "fixed",
    );
    expect(screen.getByTestId("transcription-review-player")).toHaveClass(
      "lg:left-[var(--app-shell-left)]",
    );
  });

  it("auto-scrolls the active transcript entry into view during playback when enabled", () => {
    const { container } = render(
      <TranscriptionReviewWorkspace
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

    audio!.currentTime = 3;
    fireEvent.timeUpdate(audio!);

    expect(screen.getByText("Follow up.").closest("button")).toHaveAttribute(
      "data-active",
      "true",
    );
    expect(scrollSpy).toHaveBeenCalledWith({
      behavior: "smooth",
      block: "center",
      inline: "nearest",
    });
  });
});
