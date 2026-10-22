import { act, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AppRouter } from "@/app/router/AppRouter";

const updateMocks = vi.hoisted(() => ({ checkForUpdates: vi.fn() }));

vi.mock("@/app/providers/AppUpdateProvider", () => ({
  useAppUpdates: () => ({ checkForUpdates: updateMocks.checkForUpdates }),
}));

// The routes themselves are covered elsewhere; this probe shows where the
// real BrowserRouter is.
vi.mock("@/app/router/AppRoutes", async () => {
  const { useLocation } = await import("react-router");
  return {
    AppRoutes: function LocationProbe() {
      return <span data-testid="location">{useLocation().pathname}</span>;
    },
  };
});

function dispatchTrayRoute(path: unknown) {
  act(() => {
    window.dispatchEvent(new CustomEvent("izwi:tray-route", { detail: { path } }));
  });
}

describe("AppRouter", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.history.replaceState(null, "", "/voice");
  });

  it("navigates to the routes the tray menu picks", () => {
    render(<AppRouter />);
    expect(screen.getByTestId("location")).toHaveTextContent("/voice");

    dispatchTrayRoute("/settings");

    expect(screen.getByTestId("location")).toHaveTextContent("/settings");
    expect(window.location.pathname).toBe("/settings");
  });

  it("ignores tray routes that aren't app paths", () => {
    render(<AppRouter />);

    dispatchTrayRoute("settings");
    dispatchTrayRoute(42);
    dispatchTrayRoute(undefined);

    expect(screen.getByTestId("location")).toHaveTextContent("/voice");
  });

  it("checks for updates when the tray menu asks", () => {
    render(<AppRouter />);

    act(() => {
      window.dispatchEvent(new Event("izwi:tray-check-updates"));
    });

    expect(updateMocks.checkForUpdates).toHaveBeenCalledWith(true);
  });
});
