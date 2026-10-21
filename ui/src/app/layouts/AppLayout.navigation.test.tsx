import type { ReactNode } from "react";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AppLayout } from "@/app/layouts/AppLayout";

vi.mock("@/api", () => ({
  api: { getHealth: vi.fn(() => new Promise(() => {})) },
}));

vi.mock("@/app/onboarding/FirstRunGate", () => ({
  FirstRunGate: ({ children }: { children: ReactNode }) => children,
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
              catalogError={null}
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
    expect(screen.queryByTestId("mobile-theme-toggle")).not.toBeInTheDocument();

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

    // The search button opens the drawer's Tab order; focus starts on the
    // first page link.
    const search = within(drawer).getByRole("button", {
      name: /^Search or jump to/,
    });
    search.focus();
    fireEvent.keyDown(search, { key: "Tab", shiftKey: true });
    expect(lastLink).toHaveFocus();
    fireEvent.keyDown(lastLink, { key: "Tab" });
    expect(search).toHaveFocus();

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
    expect(screen.queryByTestId("desktop-theme-toggle")).not.toBeInTheDocument();
    expect(
      within(sidebar)
        .getAllByRole("group")
        .map((group) => group.getAttribute("aria-label")),
    ).toEqual(["Converse", "Audio", "Library"]);
    expect(
      within(within(sidebar).getByRole("group", { name: "Audio" }))
        .getAllByRole("link")
        .map((link) => link.textContent),
    ).toEqual(["Transcription", "Text to Speech", "Studio"]);

    fireEvent.click(collapse);

    expect(collapse).toHaveAccessibleName("Expand sidebar");
    expect(collapse).toHaveAttribute("aria-expanded", "false");
  });
});

describe("AppLayout sidebar on Chat", () => {
  let savedPreference: string | null;

  beforeEach(() => {
    savedPreference = null;
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: {
        getItem: vi.fn(() => savedPreference),
        setItem: vi.fn((_key: string, value: string) => {
          savedPreference = value;
        }),
        removeItem: vi.fn(),
        clear: vi.fn(),
      },
    });
    setDesktopNavigation(true);
  });

  function navigate(name: string) {
    fireEvent.click(
      within(screen.getByTestId("app-sidebar")).getByRole("link", { name }),
    );
  }

  it("collapses for Chat without overwriting the saved preference", () => {
    renderLayout();
    const collapse = screen.getByTestId("sidebar-collapse-toggle");
    expect(collapse).toHaveAccessibleName("Collapse sidebar");

    navigate("Chat");
    expect(screen.getByText("Chat content")).toBeVisible();
    expect(collapse).toHaveAccessibleName("Expand sidebar");

    // Expanding it on Chat lasts while there, and is not saved.
    fireEvent.click(collapse);
    expect(collapse).toHaveAccessibleName("Collapse sidebar");

    navigate("Voice");
    expect(collapse).toHaveAccessibleName("Collapse sidebar");
    navigate("Chat");
    expect(collapse).toHaveAccessibleName("Expand sidebar");
    navigate("Voice");
    expect(collapse).toHaveAccessibleName("Collapse sidebar");
    expect(window.localStorage.setItem).not.toHaveBeenCalled();
  });

  it("restores a collapsed preference after Chat and saves changes made elsewhere", () => {
    savedPreference = "1";
    renderLayout();
    const collapse = screen.getByTestId("sidebar-collapse-toggle");
    expect(collapse).toHaveAccessibleName("Expand sidebar");

    navigate("Chat");
    fireEvent.click(collapse);
    expect(collapse).toHaveAccessibleName("Collapse sidebar");
    navigate("Voice");
    expect(collapse).toHaveAccessibleName("Expand sidebar");
    expect(savedPreference).toBe("1");

    fireEvent.click(collapse);
    expect(collapse).toHaveAccessibleName("Collapse sidebar");
    expect(window.localStorage.setItem).toHaveBeenCalledWith(
      "izwi.sidebar.collapsed",
      "0",
    );
  });
});
