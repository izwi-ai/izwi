import type { ReactElement } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AppLayout } from "@/app/layouts/AppLayout";

function BrokenRoute(): ReactElement {
  throw new Error("Route render failed");
}

vi.mock("@/app/onboarding/FirstRunOnboarding", () => ({
  FirstRunOnboarding: () => null,
}));

vi.mock("@/app/providers/AppUpdateProvider", () => ({
  useAppUpdates: () => ({
    availableUpdate: null,
    status: "idle",
    isPromptOpen: false,
    progressPercent: null,
    errorMessage: null,
    dismissPrompt: vi.fn(),
    installUpdate: vi.fn(),
    restartToApply: vi.fn(),
  }),
}));

vi.mock("@/app/analytics/events", () => ({
  trackThemePreferenceChanged: vi.fn(),
}));

describe("AppLayout model catalog recovery", () => {
  beforeEach(() => {
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: {
        getItem: vi.fn().mockReturnValue(null),
        setItem: vi.fn(),
        removeItem: vi.fn(),
        clear: vi.fn(),
      },
    });
  });

  it("shows a recoverable catalog failure without hiding route content", async () => {
    const onRetryModelCatalog = vi.fn().mockResolvedValue(undefined);

    render(
      <MemoryRouter initialEntries={["/models"]}>
        <Routes>
          <Route
            element={
              <AppLayout
                readyModelsCount={0}
                selectedModelLabel={null}
                catalogError="Local model service is offline"
                resolvedTheme="dark"
                themePreference="dark"
                onThemePreferenceChange={vi.fn()}
                onRetryModelCatalog={onRetryModelCatalog}
              />
            }
          >
            <Route path="/models" element={<div>Models route content</div>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );

    expect(screen.getByRole("alert")).toHaveTextContent(
      "Model service unavailable",
    );
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Local model service is offline",
    );
    expect(screen.getByText("Models route content")).toBeVisible();

    fireEvent.click(screen.getByRole("button", { name: "Retry models" }));

    await waitFor(() => expect(onRetryModelCatalog).toHaveBeenCalledTimes(1));
  });

  it("keeps shell navigation available when route content crashes", async () => {
    const preventExpectedWindowError = (event: ErrorEvent) => {
      event.preventDefault();
    };
    window.addEventListener("error", preventExpectedWindowError);
    vi.spyOn(console, "error").mockImplementation(() => {});

    try {
      render(
        <MemoryRouter initialEntries={["/broken"]}>
          <Routes>
            <Route
              element={
                <AppLayout
                  readyModelsCount={0}
                  selectedModelLabel={null}
                  catalogError={null}
                  resolvedTheme="dark"
                  themePreference="dark"
                  onThemePreferenceChange={vi.fn()}
                  onRetryModelCatalog={vi.fn().mockResolvedValue(undefined)}
                />
              }
            >
              <Route path="/broken" element={<BrokenRoute />} />
            </Route>
          </Routes>
        </MemoryRouter>,
      );

      expect(await screen.findByRole("alert")).toHaveTextContent(
        "This page could not be opened",
      );
      expect(screen.getByRole("link", { name: "Voice" })).toBeVisible();
    } finally {
      window.removeEventListener("error", preventExpectedWindowError);
      vi.restoreAllMocks();
    }
  });
});
