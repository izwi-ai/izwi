import { Suspense, lazy, useEffect, useState } from "react";
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
      "This page hit a problem",
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Route chunk failed");
    expect(screen.getByRole("button", { name: "Reload page" })).toBeVisible();

    shouldThrow = false;
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));

    expect(screen.getByText("Recovered route")).toBeVisible();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("copies the error details for a bug report", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });

    render(
      <RouteErrorBoundary>
        <RecoverableRoute />
      </RouteErrorBoundary>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Copy details" }));

    expect(await screen.findByRole("button", { name: "Copied" })).toBeVisible();
    expect(writeText).toHaveBeenCalledWith(
      expect.stringContaining("Error: Route chunk failed\nRoute: /"),
    );
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

  it("preserves healthy child state and effects when the route reset key changes", () => {
    const cleanup = vi.fn();
    function StatefulRoute() {
      const [count, setCount] = useState(0);
      useEffect(() => cleanup, []);
      return <button onClick={() => setCount(count + 1)}>Count {count}</button>;
    }
    const view = render(
      <RouteErrorBoundary resetKey="/text-to-speech">
        <StatefulRoute />
      </RouteErrorBoundary>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Count 0" }));

    view.rerender(
      <RouteErrorBoundary resetKey="/text-to-speech/created-record">
        <StatefulRoute />
      </RouteErrorBoundary>,
    );

    expect(screen.getByRole("button", { name: "Count 1" })).toBeVisible();
    expect(cleanup).not.toHaveBeenCalled();
    view.unmount();
    expect(cleanup).toHaveBeenCalledOnce();
  });

  it("retains a route error until navigation changes the reset key", () => {
    const view = render(
      <RouteErrorBoundary resetKey="/broken">
        <RecoverableRoute />
      </RouteErrorBoundary>,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Route chunk failed");
    shouldThrow = false;
    view.rerender(
      <RouteErrorBoundary resetKey="/broken">
        <RecoverableRoute />
      </RouteErrorBoundary>,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Route chunk failed");

    view.rerender(
      <RouteErrorBoundary resetKey="/healthy">
        <RecoverableRoute />
      </RouteErrorBoundary>,
    );
    expect(screen.getByText("Recovered route")).toBeVisible();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("keeps recovery UI if the newly selected route also fails", () => {
    const view = render(
      <RouteErrorBoundary resetKey="/broken">
        <RecoverableRoute />
      </RouteErrorBoundary>,
    );
    view.rerender(
      <RouteErrorBoundary resetKey="/also-broken">
        <RecoverableRoute />
      </RouteErrorBoundary>,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Route chunk failed");
  });
});
