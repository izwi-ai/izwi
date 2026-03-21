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

vi.mock("./VoiceDesignWorkspace", () => ({
  VoiceDesignWorkspace: ({
    onVoiceSaved,
  }: {
    onVoiceSaved?: (voiceId: string) => void;
  }) => (
    <div>
      <div data-testid="design-workspace">Design workspace</div>
      <button type="button" onClick={() => onVoiceSaved?.("designed-voice-1")}>
        Save designed voice
      </button>
    </div>
  ),
}));

function VoiceCreationModalHarness() {
  const [open, setOpen] = useState(true);
  return (
    <VoiceCreationModal
      open={open}
      onOpenChange={setOpen}
      onUseSavedVoiceInTts={vi.fn()}
      designModel={null}
      designModelReady={false}
      designModelOptions={[]}
      onDesignModelRequired={vi.fn()}
    />
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

    expect(screen.getByRole("dialog", { name: "Voice Saved" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Use in Text to Speech" })).toBeInTheDocument();
  });

  it("renders design workspace and shows a saved confirmation message", () => {
    render(<VoiceCreationModalHarness />);

    fireEvent.click(screen.getByRole("button", { name: /Design Voice/i }));
    expect(screen.getByTestId("design-workspace")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Save designed voice" }));
    expect(
      screen.getByText("Voice saved successfully"),
    ).toBeInTheDocument();
  });

  it("sends the saved voice to text to speech from success actions", () => {
    const onUseSavedVoiceInTts = vi.fn();
    function HarnessWithTtsCallback() {
      const [open, setOpen] = useState(true);
      return (
        <VoiceCreationModal
          open={open}
          onOpenChange={setOpen}
          onUseSavedVoiceInTts={onUseSavedVoiceInTts}
          designModel={null}
          designModelReady={false}
          designModelOptions={[]}
          onDesignModelRequired={vi.fn()}
        />
      );
    }

    render(<HarnessWithTtsCallback />);
    fireEvent.click(screen.getByRole("button", { name: /Clone Voice/i }));
    fireEvent.click(screen.getByRole("button", { name: "Save mock voice" }));
    fireEvent.click(screen.getByRole("button", { name: "Use in Text to Speech" }));

    expect(onUseSavedVoiceInTts).toHaveBeenCalledWith("saved-voice-1");
  });

  it("allows starting another creation flow from success state", () => {
    render(<VoiceCreationModalHarness />);
    fireEvent.click(screen.getByRole("button", { name: /Clone Voice/i }));
    fireEvent.click(screen.getByRole("button", { name: "Save mock voice" }));
    fireEvent.click(screen.getByRole("button", { name: "Create Another" }));

    expect(screen.getByRole("dialog", { name: "New Voice" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Design Voice/i })).toBeInTheDocument();
  });
});
