import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { VoiceStudioPage } from "./route";

const hookMocks = vi.hoisted(() => ({
  useRouteModelSelection: vi.fn(),
}));

vi.mock("@/features/models/hooks/useRouteModelSelection", () => ({
  useRouteModelSelection: hookMocks.useRouteModelSelection,
}));

vi.mock("@/features/models/components/RouteModelModal", () => ({
  RouteModelModal: () => null,
}));

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
  beforeEach(() => {
    hookMocks.useRouteModelSelection.mockReset();
    hookMocks.useRouteModelSelection.mockReturnValue({
      routeModels: [],
      resolvedSelectedModel: null,
      selectedModelReady: false,
      isModelModalOpen: false,
      intentVariant: null,
      closeModelModal: vi.fn(),
      openModelManager: vi.fn(),
      requestModel: vi.fn(),
      handleModelSelect: vi.fn(),
      modelOptions: [],
    });
  });

  it("defaults to the library tab", () => {
    renderVoiceStudio("/voice-studio");

    expect(screen.getByTestId("studio-library")).toBeInTheDocument();
    expect(screen.queryByTestId("studio-clone-capture")).not.toBeInTheDocument();
    expect(screen.queryByTestId("studio-design")).not.toBeInTheDocument();
  });

  it("renders clone tab content from query state", () => {
    renderVoiceStudio("/voice-studio?tab=clone");

    expect(screen.getByTestId("studio-clone-capture")).toBeInTheDocument();
    expect(
      screen.queryByTestId("page-header-history-slot"),
    ).not.toBeInTheDocument();
  });

  it("renders design tab content from query state", () => {
    renderVoiceStudio("/voice-studio?tab=design");

    expect(screen.getByTestId("studio-design")).toBeInTheDocument();
    expect(screen.getByTestId("page-header-history-slot")).toBeInTheDocument();
  });

  it("falls back to library for unknown tab query", () => {
    renderVoiceStudio("/voice-studio?tab=unexpected");

    expect(screen.getByTestId("studio-library")).toBeInTheDocument();
  });

  it("switches to design when add voice shortcut is used", () => {
    renderVoiceStudio("/voice-studio");

    fireEvent.click(screen.getByRole("button", { name: "Add Voice Shortcut" }));

    expect(screen.getByTestId("studio-design")).toBeInTheDocument();
  });
});
