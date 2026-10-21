import type { ReactNode } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AppLayout } from "@/app/layouts/AppLayout";

const themeMocks = vi.hoisted(() => ({ setThemePreference: vi.fn() }));

vi.mock("@/api", () => ({
  api: { getHealth: vi.fn(() => new Promise(() => {})) },
}));
vi.mock("@/app/onboarding/FirstRunGate", () => ({
  FirstRunGate: ({ children }: { children: ReactNode }) => children,
}));
vi.mock("@/app/providers/AppUpdateProvider", () => ({
  useAppUpdates: () => ({ availableUpdate: null, isPromptOpen: false }),
}));
vi.mock("@/app/analytics/events", () => ({
  trackThemePreferenceChanged: vi.fn(),
}));
vi.mock("@/app/providers/ThemeProvider", () => ({
  useTheme: () => ({ setThemePreference: themeMocks.setThemePreference }),
}));

function LocationProbe() {
  const location = useLocation();
  return (
    <div data-testid="location">{`${location.pathname}${location.search}`}</div>
  );
}

function renderLayout() {
  render(
    <MemoryRouter initialEntries={["/chat"]}>
      <Routes>
        <Route
          element={
            <AppLayout
              readyModelsCount={0}
              catalogError={null}
              onRetryModelCatalog={vi.fn().mockResolvedValue(true)}
            />
          }
        >
          <Route path="*" element={<LocationProbe />} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

describe("CommandPalette", () => {
  beforeEach(() => {
    themeMocks.setThemePreference.mockReset();
    vi.stubGlobal("localStorage", {
      getItem: vi.fn().mockReturnValue(null),
      setItem: vi.fn(),
    });
  });

  it("opens with the keyboard shortcut and jumps to a filtered page", () => {
    renderLayout();

    fireEvent.keyDown(window, { key: "k", ctrlKey: true });
    const input = screen.getByRole("combobox", {
      name: "Search pages and actions",
    });
    fireEvent.change(input, { target: { value: "spee" } });

    expect(
      screen.getAllByRole("option").map((option) => option.textContent),
    ).toEqual(["Text to Speech", "New speech"]);

    fireEvent.keyDown(input, { key: "ArrowDown" });
    fireEvent.keyDown(input, { key: "Enter" });

    expect(screen.getByTestId("location")).toHaveTextContent(
      "/text-to-speech?create=speech",
    );
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
  });

  it("opens from the sidebar and switches theme", () => {
    renderLayout();

    fireEvent.click(screen.getByRole("button", { name: /^Search or jump to/ }));
    fireEvent.click(screen.getByRole("option", { name: "Light theme" }));

    expect(themeMocks.setThemePreference).toHaveBeenCalledWith("light");
    expect(screen.getByTestId("location")).toHaveTextContent("/chat");
  });

  it("says when nothing matches", () => {
    renderLayout();

    fireEvent.keyDown(window, { key: "k", metaKey: true });
    fireEvent.change(screen.getByRole("combobox"), {
      target: { value: "zzz" },
    });

    expect(screen.getByText("No matching pages or actions.")).toBeVisible();
  });
});
