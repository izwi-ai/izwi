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
        summaryModelGuidance="Download and load Qwen3.5-4B in Transcription models to generate summaries."
      />,
    );

    expect(
      screen.getByText(
        "Download and load Qwen3.5-4B in Transcription models to generate summaries.",
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
      "lg:left-(--app-shell-left)",
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

  it("reads speaker-attributed turns as one flowing transcript", () => {
    const { container } = render(
      <TranscriptionReviewWorkspace
        record={{
          ...record,
          transcription: "",
          segments: [],
          words: [],
          speaker_attributed_text:
            "[Speaker 1]: Hi everybody. [Speaker 2]: Thank you. [Speaker 1]: How did it start?",
          speaker_turns: [
            { speaker: "Speaker 1", text: "Hi everybody." },
            { speaker: "Speaker 2", text: "Thank you." },
            { speaker: "Speaker 2", text: "It's great to be here." },
            { speaker: "Speaker 1", text: "How did it start?" },
          ],
        }}
        audioUrl="/audio/meeting.wav"
      />,
    );

    expect(screen.getAllByRole("heading", { name: "Transcript" })).toHaveLength(1);
    expect(screen.getAllByText("Speaker 1")).toHaveLength(2);
    expect(screen.getAllByText("Speaker 2")).toHaveLength(1);
    expect(
      screen.getByText("Thank you. It's great to be here."),
    ).toBeInTheDocument();
    expect(container.textContent).not.toContain("[Speaker");
    // Untimed turns are text, not seek buttons, and stay off the timeline.
    expect(screen.getByText("Hi everybody.").closest("button")).toBeNull();
    expect(screen.getByText("2 identified")).toBeInTheDocument();
    expect(screen.getByText("Off")).toBeInTheDocument();
  });

  it("lets timed speaker turns seek and highlight during playback", () => {
    const { container } = render(
      <TranscriptionReviewWorkspace
        record={{
          ...record,
          segments: [],
          words: [],
          speaker_attributed_text: null,
          speaker_turns: [
            { speaker: "Speaker 1", text: "Question?", start: 0, end: 2 },
            { speaker: "Speaker 2", text: "Answer.", start: 2.5, end: 6 },
          ],
        }}
        audioUrl="/audio/meeting.wav"
      />,
    );

    const audio = container.querySelector("audio");
    Object.defineProperty(audio, "duration", { configurable: true, value: 6 });
    Object.defineProperty(audio, "currentTime", {
      configurable: true,
      writable: true,
      value: 0,
    });
    fireEvent.loadedMetadata(audio!);

    const answer = screen.getByText("Answer.").closest("button");
    expect(answer).not.toBeNull();
    fireEvent.click(answer!);
    expect(audio!.currentTime).toBe(2.5);

    audio!.currentTime = 3;
    fireEvent.timeUpdate(audio!);
    expect(answer).toHaveAttribute("data-active", "true");
    expect(screen.getByText("Turn-level")).toBeInTheDocument();
  });
});

