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
        <Route path="/voices" element={<VoiceStudioPage {...baseProps} />} />
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

  it("shows the library content by default", () => {
    renderVoiceStudio("/voices");

    expect(screen.getByTestId("studio-library")).toBeInTheDocument();
  });

  it("ignores legacy tab query and keeps library-first layout", () => {
    renderVoiceStudio("/voices?tab=design");

    expect(screen.getByTestId("studio-library")).toBeInTheDocument();
  });

  it("opens creation modal from the page header action", () => {
    renderVoiceStudio("/voices");

    fireEvent.click(screen.getByRole("button", { name: "New Voice" }));

    expect(screen.getByRole("dialog", { name: "New Voice" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Clone Voice/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Design Voice/i })).toBeInTheDocument();
  });

  it("opens creation modal when library add shortcut is used", () => {
    renderVoiceStudio("/voices");

    fireEvent.click(screen.getByRole("button", { name: "Add Voice Shortcut" }));

    expect(screen.getByRole("dialog", { name: "New Voice" })).toBeInTheDocument();
  });
});
