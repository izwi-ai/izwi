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
  it("renders clone workflow guidance", () => {
    render(<VoiceCaptureWorkspace />);

    expect(
      screen.getByText(
        "Upload or record a clean reference with transcript, then save a reusable voice profile.",
      ),
    ).toBeInTheDocument();
  });

  it("notifies when a saved voice is created", () => {
    const handleVoiceSaved = vi.fn();
    render(<VoiceCaptureWorkspace onVoiceSaved={handleVoiceSaved} />);

    fireEvent.click(screen.getByRole("button", { name: "Save Voice" }));

    expect(
      screen.getByText("Voice profile saved successfully."),
    ).toBeInTheDocument();
    expect(handleVoiceSaved).toHaveBeenCalledWith("saved-voice");
  });
});
