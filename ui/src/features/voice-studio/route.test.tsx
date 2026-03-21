import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import { VoiceStudioPage } from "./route";

vi.mock("@/features/voices/route", () => ({
  VoicesPage: ({ onAddNewVoice }: { onAddNewVoice?: () => void }) => (
    <div>
      <div data-testid="studio-library">Voice library</div>
      <button type="button" onClick={() => onAddNewVoice?.()}>
        Add Voice Shortcut
      </button>
    </div>
  ),
}));

vi.mock("@/components/VoiceCaptureWorkspace", () => ({
  VoiceCaptureWorkspace: () => (
    <div data-testid="studio-clone-capture">Capture workspace</div>
  ),
}));

vi.mock("@/components/VoiceDesignWorkspace", () => ({
  VoiceDesignWorkspace: () => (
    <div data-testid="studio-design">Design workspace</div>
  ),
}));

const baseProps = {
  models: [],
  selectedModel: null,
  loading: false,
  downloadProgress: {},
  onDownload: vi.fn(),
  onCancelDownload: vi.fn(),
  onLoad: vi.fn(),
  onUnload: vi.fn(),
  onDelete: vi.fn(),
  onSelect: vi.fn(),
  onError: vi.fn(),
};

function renderVoiceStudio(initialEntry: string) {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route path="/voice-studio" element={<VoiceStudioPage {...baseProps} />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("VoiceStudioPage", () => {
  it("shows the library content by default", () => {
    renderVoiceStudio("/voice-studio");

    expect(screen.getByTestId("studio-library")).toBeInTheDocument();
    expect(screen.queryByTestId("studio-clone-capture")).not.toBeInTheDocument();
    expect(screen.queryByTestId("studio-design")).not.toBeInTheDocument();
  });

  it("ignores legacy tab query and keeps library-first layout", () => {
    renderVoiceStudio("/voice-studio?tab=design");

    expect(screen.getByTestId("studio-library")).toBeInTheDocument();
    expect(screen.queryByTestId("studio-design")).not.toBeInTheDocument();
  });

  it("opens creation modal from the page header action", () => {
    renderVoiceStudio("/voice-studio");

    fireEvent.click(screen.getByRole("button", { name: "New Voice" }));

    expect(screen.getByRole("dialog", { name: "New Voice" })).toBeInTheDocument();
  });

  it("opens creation modal when library add shortcut is used", () => {
    renderVoiceStudio("/voice-studio");

    fireEvent.click(screen.getByRole("button", { name: "Add Voice Shortcut" }));

    expect(screen.getByRole("dialog", { name: "New Voice" })).toBeInTheDocument();
  });
});
