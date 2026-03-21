import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { VoicePicker } from "./VoicePicker";

describe("VoicePicker", () => {
  beforeEach(() => {
    Object.defineProperty(HTMLMediaElement.prototype, "play", {
      configurable: true,
      value: vi.fn().mockResolvedValue(undefined),
    });
    Object.defineProperty(HTMLMediaElement.prototype, "pause", {
      configurable: true,
      value: vi.fn(),
    });
  });

  it("renders the redesigned voice profile card structure", () => {
    render(
      <VoicePicker
        items={[
          {
            id: "balanced-21",
            name: "Balanced 21 yo",
            categoryLabel: "Designed voice",
            description:
              "Hello, this is Izwi. This short preview helps compare the voice.",
            meta: ["128 chars", "Mar 21, 2026"],
            previewUrl: "/voices/balanced-21.wav",
            actions: (
              <>
                <button type="button">Use in TTS</button>
                <button type="button">Delete</button>
              </>
            ),
          },
        ]}
        emptyTitle="No voices"
        emptyDescription="Nothing to show"
      />,
    );

    expect(screen.getByTestId("voice-card-balanced-21")).toBeInTheDocument();
    expect(screen.getByText("Designed voice")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Balanced 21 yo" })).toBeInTheDocument();
    expect(screen.getByText("Voice Notes")).toBeInTheDocument();
    expect(screen.getByText("Preview")).toBeInTheDocument();
    expect(screen.getByText("128 chars")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Play preview for Balanced 21 yo/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Use in TTS" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Delete" })).toBeInTheDocument();
  });

  it("updates the custom preview timeline from audio metadata", () => {
    const { container } = render(
      <VoicePicker
        items={[
          {
            id: "alloy",
            name: "Alloy",
            categoryLabel: "Built-in voice",
            description: "Warm and clear",
            previewUrl: "/voices/alloy.wav",
          },
        ]}
        emptyTitle="No voices"
        emptyDescription="Nothing to show"
      />,
    );

    const audio = container.querySelector("audio") as HTMLAudioElement | null;
    expect(audio).not.toBeNull();

    Object.defineProperty(audio!, "duration", {
      configurable: true,
      value: 15,
    });
    fireEvent.loadedMetadata(audio!);
    expect(screen.getByText("00:15")).toBeInTheDocument();

    Object.defineProperty(audio!, "currentTime", {
      configurable: true,
      writable: true,
      value: 4,
    });
    fireEvent.timeUpdate(audio!);

    expect(screen.getByText("00:04")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Play preview for Alloy/i }));
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalled();
  });
});
