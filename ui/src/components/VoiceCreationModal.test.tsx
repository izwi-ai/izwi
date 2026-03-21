import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { VoiceCreationModal } from "./VoiceCreationModal";

vi.mock("./VoiceCaptureWorkspace", () => ({
  VoiceCaptureWorkspace: ({
    onVoiceSaved,
  }: {
    onVoiceSaved?: (voiceId: string) => void;
  }) => (
    <div>
      <div data-testid="clone-workspace">Clone workspace</div>
      <button type="button" onClick={() => onVoiceSaved?.("saved-voice-1")}>
        Save mock voice
      </button>
    </div>
  ),
}));

function VoiceCreationModalHarness() {
  const [open, setOpen] = useState(true);
  return (
    <VoiceCreationModal open={open} onOpenChange={setOpen} />
  );
}

describe("VoiceCreationModal", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("moves from flow choice into clone and supports back navigation", () => {
    render(<VoiceCreationModalHarness />);

    fireEvent.click(screen.getByRole("button", { name: /Clone Voice/i }));

    expect(screen.getByRole("dialog", { name: "Clone Voice" })).toBeInTheDocument();
    expect(screen.getByTestId("clone-workspace")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Back" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Back" }));

    expect(screen.getByRole("dialog", { name: "New Voice" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Design Voice/i })).toBeInTheDocument();
  });

  it("protects in-progress draft when closing from clone step", () => {
    const confirmMock = vi.spyOn(window, "confirm").mockReturnValue(false);
    render(<VoiceCreationModalHarness />);

    fireEvent.click(screen.getByRole("button", { name: /Clone Voice/i }));
    fireEvent.click(screen.getByRole("button", { name: "Close" }));

    expect(confirmMock).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("dialog", { name: "Clone Voice" })).toBeInTheDocument();
  });

  it("shows a saved message after clone voice save callback", () => {
    render(<VoiceCreationModalHarness />);

    fireEvent.click(screen.getByRole("button", { name: /Clone Voice/i }));
    fireEvent.click(screen.getByRole("button", { name: "Save mock voice" }));

    expect(
      screen.getByText("Saved voice profile is ready in your library."),
    ).toBeInTheDocument();
  });
});
