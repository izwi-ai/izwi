import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { VoiceCaptureWorkspace } from "./VoiceCaptureWorkspace";

vi.mock("./VoiceClone", () => ({
  VoiceClone: ({
    onReferenceStateChange,
    onSavedVoiceCreated,
  }: {
    onReferenceStateChange?: (state: {
      mode: "upload" | "record" | "saved" | null;
      sampleReady: boolean;
      sampleDurationSecs: number | null;
      transcriptChars: number;
      activeSavedVoiceId: string | null;
      warnings: string[];
      canClone: boolean;
    }) => void;
    onSavedVoiceCreated?: (voiceId: string) => void;
  }) => (
    <div>
      <button
        type="button"
        onClick={() =>
          onReferenceStateChange?.({
            mode: "saved",
            sampleReady: true,
            sampleDurationSecs: 6.2,
            transcriptChars: 74,
            activeSavedVoiceId: "reference-voice",
            warnings: [],
            canClone: true,
          })
        }
      >
        Apply Reference
      </button>
      <button type="button" onClick={() => onSavedVoiceCreated?.("saved-voice")}>
        Save Voice
      </button>
    </div>
  ),
}));

describe("VoiceCaptureWorkspace", () => {
  it("shows guidance before a voice profile is available", () => {
    render(<VoiceCaptureWorkspace onUseInTts={vi.fn()} />);

    expect(
      screen.getByText(
        "Save a voice profile to enable one-click use in Text to Speech.",
      ),
    ).toBeInTheDocument();
  });

  it("requires rights confirmation before enabling TTS handoff", () => {
    const handleUseInTts = vi.fn();
    render(<VoiceCaptureWorkspace onUseInTts={handleUseInTts} />);

    fireEvent.click(screen.getByRole("button", { name: "Save Voice" }));

    expect(
      screen.getByText("Confirm rights to unlock direct handoff to Text to Speech."),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Use in TTS" }),
    ).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("checkbox", {
        name: /I have permission to clone this voice/i,
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Use in TTS" }));

    expect(handleUseInTts).toHaveBeenCalledWith("saved-voice");
  });
});
