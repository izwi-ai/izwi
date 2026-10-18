import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { VoiceStudioPage } from "./route";

const hookMocks = vi.hoisted(() => ({
  useRouteModelSelection: vi.fn(),
}));

const componentMocks = vi.hoisted(() => ({
  voicesPage: vi.fn(),
}));

vi.mock("@/features/models/hooks/useRouteModelSelection", () => ({
  useRouteModelSelection: hookMocks.useRouteModelSelection,
}));

vi.mock("@/features/models/components/RouteModelModal", () => ({
  RouteModelModal: () => null,
}));

vi.mock("@/features/voices/route", () => ({
  VoicesPage: (props: Record<string, unknown>) => {
    componentMocks.voicesPage(props);
    return (
      <div>
        <div data-testid="studio-library">Voice library</div>
      </div>
    );
  },
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
    componentMocks.voicesPage.mockReset();
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

  it("opens the design step for legacy ?tab=design links over the library", async () => {
    renderVoiceStudio("/voices?tab=design");

    expect(screen.getByTestId("studio-library")).toBeInTheDocument();
    expect(
      await screen.findByRole("dialog", { name: "Design Voice" }),
    ).toBeInTheDocument();
  });

  it("opens the clone step for legacy ?tab=clone links", async () => {
    renderVoiceStudio("/voices?tab=clone");

    expect(
      await screen.findByRole("dialog", { name: "Clone Voice" }),
    ).toBeInTheDocument();
  });

  it("ignores unknown tab values", () => {
    renderVoiceStudio("/voices?tab=library");

    expect(screen.getByTestId("studio-library")).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("opens creation modal from the page header action", () => {
    renderVoiceStudio("/voices");

    fireEvent.click(screen.getByRole("button", { name: "New Voice" }));

    expect(screen.getByRole("dialog", { name: "New Voice" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Clone Voice/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Design Voice/i })).toBeInTheDocument();
  });

  it("forwards a models-button trigger to the embedded voices workspace", () => {
    renderVoiceStudio("/voices");

    const initialCall =
      componentMocks.voicesPage.mock.calls[
        componentMocks.voicesPage.mock.calls.length - 1
      ]?.[0] as
      | { openModelManagerRequestKey?: number }
      | undefined;
    expect(initialCall?.openModelManagerRequestKey).toBe(0);

    fireEvent.click(screen.getByRole("button", { name: "Models" }));

    const afterClickCall =
      componentMocks.voicesPage.mock.calls[
        componentMocks.voicesPage.mock.calls.length - 1
      ]?.[0] as
      | { openModelManagerRequestKey?: number }
      | undefined;
    expect(afterClickCall?.openModelManagerRequestKey).toBe(1);
  });
});
