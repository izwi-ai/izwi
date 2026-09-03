import type { ReactNode } from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AppProviders } from "@/app/providers/AppProviders";

vi.mock("framer-motion", () => ({
  MotionConfig: ({
    children,
    reducedMotion,
  }: {
    children: ReactNode;
    reducedMotion: string;
  }) => <div data-reduced-motion={reducedMotion}>{children}</div>,
}));

vi.mock("@/app/analytics/AnalyticsBootstrapProvider", () => ({
  AnalyticsBootstrapProvider: ({ children }: { children: ReactNode }) => children,
}));

vi.mock("@/app/providers/AppUpdateProvider", () => ({
  AppUpdateProvider: ({ children }: { children: ReactNode }) => children,
}));

vi.mock("@/app/providers/ModelCatalogProvider", () => ({
  ModelCatalogProvider: ({ children }: { children: ReactNode }) => children,
}));

vi.mock("@/app/providers/NotificationProvider", () => ({
  NotificationProvider: ({ children }: { children: ReactNode }) => children,
}));

vi.mock("@/app/providers/ThemeProvider", () => ({
  ThemeProvider: ({ children }: { children: ReactNode }) => children,
}));

describe("AppProviders motion preferences", () => {
  it("delegates animation reduction to the operating-system preference", () => {
    render(
      <AppProviders>
        <span>Application</span>
      </AppProviders>,
    );

    expect(screen.getByText("Application").parentElement).toHaveAttribute(
      "data-reduced-motion",
      "user",
    );
  });
});
