import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  NotificationProvider,
  NotificationSlot,
  useNotifications,
  type NotificationInput,
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

let notifyRef: ((input: NotificationInput) => string) | null = null;
let dismissRef: ((id: string) => void) | null = null;

function ApiProbe() {
  const { notify, dismiss } = useNotifications();
  useEffect(() => {
    notifyRef = notify;
    dismissRef = dismiss;
  }, [notify, dismiss]);
  return null;
}

function renderApi() {
  render(
    <NotificationProvider>
      <ApiProbe />
    </NotificationProvider>,
  );
  return {
    notify: (input: NotificationInput) => {
      let id = "";
      act(() => {
        id = notifyRef!(input);
      });
      return id;
    },
    dismiss: (id: string) => act(() => dismissRef!(id)),
  };
}

function stack() {
  return screen.getByRole("region", { name: "Notifications" });
}

const downloads = {
  key: "downloads",
  title: (count: number) => `Downloading ${count} models`,
};

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

describe("NotificationProvider updates and progress", () => {
  it("updates a notification in place when notified again with its id", () => {
    const { notify } = renderApi();

    const id = notify({ id: "download:qwen", title: "Downloading Qwen3 4B", progress: 10 });
    expect(id).toBe("download:qwen");
    notify({
      id: "download:qwen",
      title: "Downloading Qwen3 4B",
      progress: 64,
      detail: "2.1 of 3.3 GB",
    });

    expect(within(stack()).getAllByRole("status")).toHaveLength(1);
    const bar = screen.getByRole("progressbar", { name: "Downloading Qwen3 4B" });
    expect(bar).toHaveAttribute("aria-valuenow", "64");
    expect(bar).toHaveAttribute("aria-valuetext", "2.1 of 3.3 GB");
    expect(stack()).toHaveTextContent("2.1 of 3.3 GB");
  });

  it("runs an action and closes the notification", async () => {
    const { notify } = renderApi();
    const onOpen = vi.fn();

    notify({
      title: "Transcript ready",
      description: "bughunt-long-2.mp3",
      tone: "success",
      action: { label: "Open", onClick: onOpen },
    });
    fireEvent.click(screen.getByRole("button", { name: "Open" }));

    expect(onOpen).toHaveBeenCalledOnce();
    await waitFor(() =>
      expect(screen.queryByText("Transcript ready")).not.toBeInTheDocument(),
    );
  });

  it("shows an error's full reason instead of truncating it", () => {
    const { notify } = renderApi();
    const reason =
      "Four models are already loaded and there is not enough memory left. Unload one to make room.";

    notify({ title: "Couldn't load Qwen3.5 4B", description: reason, tone: "danger" });

    const description = within(screen.getByRole("alert")).getByText(reason);
    expect(description).not.toHaveClass("truncate");
  });

  it("keeps progress the user closed hidden until its final update", async () => {
    const { notify } = renderApi();

    notify({ id: "dl", title: "Downloading Kokoro", progress: 20 });
    fireEvent.click(
      screen.getByRole("button", { name: "Dismiss Downloading Kokoro notification" }),
    );
    await waitFor(() =>
      expect(screen.queryByText("Downloading Kokoro")).not.toBeInTheDocument(),
    );

    notify({ id: "dl", title: "Downloading Kokoro", progress: 40 });
    expect(screen.queryByText("Downloading Kokoro")).not.toBeInTheDocument();

    notify({ id: "dl", title: "Kokoro downloaded", tone: "success" });
    expect(screen.getByText("Kokoro downloaded")).toBeInTheDocument();
  });

  it("removes a notification when the caller dismisses it", async () => {
    const { notify, dismiss } = renderApi();

    notify({ id: "dl", title: "Downloading Kokoro", progress: 20 });
    dismiss("dl");
    await waitFor(() =>
      expect(screen.queryByText("Downloading Kokoro")).not.toBeInTheDocument(),
    );

    // A caller's dismiss does not hide the id's next progress.
    notify({ id: "dl", title: "Downloading Kokoro", progress: 5 });
    expect(screen.getByText("Downloading Kokoro")).toBeInTheDocument();
  });
});

describe("NotificationProvider groups", () => {
  it("combines grouped notifications into one row with weighted progress", () => {
    const { notify } = renderApi();

    notify({
      id: "dl:a",
      title: "Downloading Qwen3 4B",
      progress: 50,
      weight: 4,
      group: downloads,
    });
    // One member shows as itself.
    expect(stack()).toHaveTextContent("Downloading Qwen3 4B");

    notify({
      id: "dl:b",
      title: "Downloading Kokoro",
      progress: 100,
      weight: 1,
      group: downloads,
    });
    notify({
      id: "dl:c",
      title: "Downloading Parakeet",
      progress: 0,
      weight: 5,
      group: downloads,
    });

    const rows = within(stack()).getAllByRole("status");
    expect(rows).toHaveLength(1);
    expect(rows[0]).toHaveTextContent("Downloading 3 models");
    // (50×4 + 100×1 + 0×5) / 10 = 30%
    expect(
      screen.getByRole("progressbar", { name: "Downloading 3 models" }),
    ).toHaveAttribute("aria-valuenow", "30");
    expect(rows[0]).toHaveTextContent("30%");

    // A member that finishes leaves the group and shows on its own.
    notify({ id: "dl:b", title: "Kokoro downloaded", tone: "success" });
    const updated = within(stack()).getAllByRole("status");
    expect(updated).toHaveLength(2);
    expect(updated[0]).toHaveTextContent("Downloading 2 models");
    expect(updated[1]).toHaveTextContent("Kokoro downloaded");
  });

  it("dismisses every member of a combined row", async () => {
    const { notify } = renderApi();

    notify({ id: "dl:a", title: "Downloading A", progress: 10, group: downloads });
    notify({ id: "dl:b", title: "Downloading B", progress: 30, group: downloads });
    fireEvent.click(
      screen.getByRole("button", { name: "Dismiss Downloading 2 models notification" }),
    );
    await waitFor(() =>
      expect(screen.queryByRole("status")).not.toBeInTheDocument(),
    );

    notify({ id: "dl:a", title: "Downloading A", progress: 60, group: downloads });
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });
});

describe("NotificationProvider durations", () => {
  // Closed rows animate out; only open ones count.
  const openText = () =>
    Array.from(stack().querySelectorAll('[data-state="open"]'))
      .map((row) => row.textContent)
      .join("\n");

  afterEach(() => {
    vi.useRealTimers();
  });

  it("closes success on its own but keeps errors and progress until dismissed", () => {
    vi.useFakeTimers();
    const { notify } = renderApi();

    notify({ title: "Model loaded", tone: "success" });
    notify({ title: "Model load failed", description: "Out of memory", tone: "danger" });
    notify({ title: "Downloading Kokoro", progress: 5 });
    act(() => {
      vi.advanceTimersByTime(4_000);
    });
    expect(openText()).toContain("Model loaded");

    act(() => {
      vi.advanceTimersByTime(60_000);
    });
    expect(openText()).not.toContain("Model loaded");
    expect(openText()).toContain("Model load failed");
    expect(openText()).toContain("Downloading Kokoro");
  });

  it("waits while the pointer is on a notification", () => {
    vi.useFakeTimers();
    const { notify } = renderApi();

    notify({ title: "Model unloaded", tone: "info" });
    fireEvent.pointerEnter(screen.getByRole("status"));
    act(() => {
      vi.advanceTimersByTime(30_000);
    });
    expect(openText()).toContain("Model unloaded");

    fireEvent.pointerLeave(screen.getByRole("status"));
    act(() => {
      vi.advanceTimersByTime(5_000);
    });
    expect(openText()).not.toContain("Model unloaded");
  });
});

describe("NotificationSlot", () => {
  it("renders persistent cards at the bottom of the notification stack", () => {
    render(
      <NotificationProvider>
        <ApiProbe />
        <NotificationSlot>
          <section aria-label="Update ready">Izwi 0.1.1 is ready</section>
        </NotificationSlot>
      </NotificationProvider>,
    );
    act(() => {
      notifyRef!({ title: "Model loaded", tone: "success" });
    });

    const region = stack();
    const card = within(region).getByRole("region", { name: "Update ready" });
    expect(card).toBeInTheDocument();
    expect(
      within(region).getByRole("status").compareDocumentPosition(card) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });
});
