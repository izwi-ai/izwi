import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AppLayout } from "@/app/layouts/AppLayout";

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

function setDesktopNavigation(matches: boolean) {
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches,
      media: query,
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
}

function renderLayout() {
  return render(
    <MemoryRouter initialEntries={["/voice"]}>
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
          <Route path="/voice" element={<div>Voice content</div>} />
          <Route path="/chat" element={<div>Chat content</div>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

describe("AppLayout compact navigation accessibility", () => {
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

  it("removes the closed compact navigation from focus and names icon controls", () => {
    setDesktopNavigation(false);
    renderLayout();

    const sidebar = screen.getByTestId("app-sidebar");
    const menuTrigger = screen.getByTestId("mobile-navigation-trigger");

    expect(sidebar).toHaveAttribute("aria-hidden", "true");
    expect(sidebar).toHaveAttribute("inert");
    expect(menuTrigger).toHaveAccessibleName("Open navigation");
    expect(menuTrigger).toHaveAttribute("aria-expanded", "false");
    expect(menuTrigger).toHaveAttribute("aria-controls", "app-sidebar-navigation");
    expect(screen.getByTestId("mobile-theme-toggle")).toHaveAccessibleName(
      "Switch to light mode",
    );

    for (const link of within(sidebar).getAllByRole("link", { hidden: true })) {
      expect(link).toHaveAttribute("tabindex", "-1");
    }
    expect(screen.getByTestId("sidebar-collapse-toggle")).toHaveAttribute(
      "tabindex",
      "-1",
    );
  });

  it("moves focus into the opened drawer, traps Tab, and restores focus on Escape", async () => {
    setDesktopNavigation(false);
    renderLayout();

    const menuTrigger = screen.getByTestId("mobile-navigation-trigger");
    fireEvent.click(menuTrigger);

    const drawer = screen.getByRole("dialog", { name: "Main navigation" });
    const firstLink = within(drawer).getByRole("link", { name: "Voice" });
    const lastLink = within(drawer).getByRole("link", {
      name: "Izwi on GitHub",
    });

    await waitFor(() => expect(firstLink).toHaveFocus());
    expect(menuTrigger).toHaveAccessibleName("Close navigation");
    expect(menuTrigger).toHaveAttribute("aria-expanded", "true");
    expect(drawer).not.toHaveAttribute("inert");

    fireEvent.keyDown(firstLink, { key: "Tab", shiftKey: true });
    expect(lastLink).toHaveFocus();
    fireEvent.keyDown(lastLink, { key: "Tab" });
    expect(firstLink).toHaveFocus();

    fireEvent.keyDown(firstLink, { key: "Escape" });
    expect(menuTrigger).toHaveFocus();
    expect(menuTrigger).toHaveAccessibleName("Open navigation");
    expect(screen.getByTestId("app-sidebar")).toHaveAttribute("inert");
  });

  it("keeps desktop navigation operable and exposes collapse state", () => {
    setDesktopNavigation(true);
    renderLayout();

    const sidebar = screen.getByTestId("app-sidebar");
    const collapse = screen.getByTestId("sidebar-collapse-toggle");
    const voiceLink = within(sidebar).getByRole("link", { name: "Voice" });

    expect(sidebar).not.toHaveAttribute("aria-hidden");
    expect(sidebar).not.toHaveAttribute("inert");
    expect(voiceLink).not.toHaveAttribute("tabindex", "-1");
    expect(collapse).toHaveAccessibleName("Collapse sidebar");
    expect(collapse).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByTestId("desktop-theme-toggle")).toHaveAccessibleName(
      "Switch to light mode",
    );

    fireEvent.click(collapse);

    expect(collapse).toHaveAccessibleName("Expand sidebar");
    expect(collapse).toHaveAttribute("aria-expanded", "false");
  });
});
