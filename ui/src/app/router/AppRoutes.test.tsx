import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Outlet, useLocation } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";

import { AppRoutes } from "@/app/router/AppRoutes";

vi.mock("@/app/layouts/AppLayout", () => ({
  AppLayout: () => <Outlet />,
}));

vi.mock("@/app/providers/ThemeProvider", () => ({
  useTheme: () => ({
    resolvedTheme: "dark",
    themePreference: "system",
    setThemePreference: vi.fn(),
  }),
}));

vi.mock("@/app/providers/ModelCatalogProvider", () => ({
  useModelCatalog: () => ({
    models: [],
    selectedModel: null,
    loading: false,
    downloadProgress: {},
    readyModelsCount: 0,
    selectModel: vi.fn(),
    reportError: vi.fn(),
    refreshModels: vi.fn(async () => undefined),
    downloadModel: vi.fn(),
    cancelModelDownload: vi.fn(),
    loadModel: vi.fn(),
    unloadModel: vi.fn(),
    deleteModel: vi.fn(),
  }),
}));

vi.mock("@/app/analytics/events", () => ({
  routeIdFromPathname: vi.fn(() => null),
  trackRouteViewed: vi.fn(),
}));

vi.mock("@/features/speech-text/route", () => ({
  SpeechTextPage: () => {
    const location = useLocation();
    return (
      <div data-testid="speech-text-location">
        {location.pathname}
        {location.search}
      </div>
    );
  },
}));

function renderRoutes(initialEntry: string) {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <AppRoutes />
    </MemoryRouter>,
  );
}

describe("AppRoutes recovery routes", () => {
  it("redirects the legacy diarization collection to its creation intent", async () => {
    renderRoutes("/diarization");

    expect(await screen.findByTestId("speech-text-location")).toHaveTextContent(
      "/transcription?create=diarization",
    );
    await waitFor(() => expect(document.title).toBe("New Diarization · Izwi"));
  });

  it("redirects legacy diarization details and preserves query state", async () => {
    renderRoutes("/diarization/diar-1?model=sortformer");

    expect(await screen.findByTestId("speech-text-location")).toHaveTextContent(
      "/transcription/diar-1?model=sortformer&mode=diarization",
    );
    await waitFor(() =>
      expect(document.title).toBe("Diarization Record · Izwi"),
    );
  });

  it("shows recovery choices instead of silently redirecting unknown paths", async () => {
    renderRoutes("/missing/workspace?from=bookmark");

    expect(
      screen.getByRole("heading", { name: "Page not found" }),
    ).toBeInTheDocument();
    expect(screen.getByText("/missing/workspace?from=bookmark")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open Voice" })).toHaveAttribute(
      "href",
      "/voice",
    );
    expect(
      screen.getByRole("link", { name: "Open Transcription" }),
    ).toHaveAttribute("href", "/transcription");
    expect(screen.getByRole("link", { name: "Open Models" })).toHaveAttribute(
      "href",
      "/models",
    );
    await waitFor(() => expect(document.title).toBe("Page Not Found · Izwi"));
  });
});
