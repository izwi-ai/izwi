import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  NotificationProvider,
  useNotifications,
} from "@/app/providers/NotificationProvider";

function NotificationHarness() {
  const { notify } = useNotifications();

  return (
    <>
      <button
        type="button"
        onClick={() =>
          notify({
            title: "Export complete",
            description: "The transcript is ready.",
            tone: "success",
            durationMs: 60_000,
          })
        }
      >
        Show success
      </button>
      <button
        type="button"
        onClick={() =>
          notify({
            title: "Export failed",
            description: "Try the export again.",
            tone: "danger",
            durationMs: 60_000,
          })
        }
      >
        Show error
      </button>
    </>
  );
}

describe("NotificationProvider accessibility", () => {
  it("announces success politely and gives its dismiss action a useful name", async () => {
    render(
      <NotificationProvider>
        <NotificationHarness />
      </NotificationProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Show success" }));

    const notification = screen.getByRole("status");
    expect(notification).toHaveAttribute("aria-live", "polite");
    expect(notification).toHaveAttribute("aria-atomic", "true");
    expect(notification).toHaveTextContent("Export complete");

    const dismiss = screen.getByRole("button", {
      name: "Dismiss Export complete notification",
    });
    fireEvent.click(dismiss);
    await waitFor(() =>
      expect(screen.queryByRole("status")).not.toBeInTheDocument(),
    );
  });

  it("announces failures assertively", () => {
    render(
      <NotificationProvider>
        <NotificationHarness />
      </NotificationProvider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Show error" }));

    const notification = screen.getByRole("alert");
    expect(notification).toHaveAttribute("aria-live", "assertive");
    expect(notification).toHaveAttribute("aria-atomic", "true");
    expect(notification).toHaveTextContent("Export failed");
    expect(
      screen.getByRole("button", {
        name: "Dismiss Export failed notification",
      }),
    ).toBeInTheDocument();
  });
});
