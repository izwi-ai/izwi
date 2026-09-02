import { Suspense, lazy } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { RouteErrorBoundary } from "@/app/router/RouteErrorBoundary";

let shouldThrow = true;

function RecoverableRoute() {
  if (shouldThrow) {
    throw new Error("Route chunk failed");
  }
  return <div>Recovered route</div>;
}

describe("RouteErrorBoundary", () => {
  const preventExpectedWindowError = (event: ErrorEvent) => {
    event.preventDefault();
  };

  beforeEach(() => {
    shouldThrow = true;
    vi.spyOn(console, "error").mockImplementation(() => {});
    window.addEventListener("error", preventExpectedWindowError);
  });

  afterEach(() => {
    window.removeEventListener("error", preventExpectedWindowError);
    vi.restoreAllMocks();
  });

  it("keeps surrounding navigation visible and retries the failed route", () => {
    render(
      <div>
        <nav aria-label="Primary">Navigation remains available</nav>
        <RouteErrorBoundary>
          <RecoverableRoute />
        </RouteErrorBoundary>
      </div>,
    );

    expect(screen.getByRole("navigation", { name: "Primary" })).toBeVisible();
    expect(screen.getByRole("alert")).toHaveTextContent(
      "This page could not be opened",
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Route chunk failed");

    shouldThrow = false;
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));

    expect(screen.getByText("Recovered route")).toBeVisible();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("turns a rejected lazy route into recovery UI", async () => {
    const RejectedRoute = lazy(async () => {
      throw new Error("Route bundle could not be loaded");
    });

    render(
      <RouteErrorBoundary>
        <Suspense fallback={<div>Loading route</div>}>
          <RejectedRoute />
        </Suspense>
      </RouteErrorBoundary>,
    );

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Route bundle could not be loaded",
    );
  });
});
