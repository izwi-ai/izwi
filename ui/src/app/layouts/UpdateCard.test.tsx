import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { UpdateCard } from "@/app/layouts/UpdateCard";
import {
  AppUpdateProvider,
  useAppUpdates,
} from "@/app/providers/AppUpdateProvider";
import { NotificationProvider } from "@/app/providers/NotificationProvider";

const mocks = vi.hoisted(() => ({
  checkForBetaUpdate: vi.fn(),
  getUpdaterHealthSnapshot: vi.fn(),
  installBetaUpdate: vi.fn(),
  relaunchAfterUpdate: vi.fn(),
}));

vi.mock("@tauri-apps/api/core", () => ({
  isTauri: () => true,
}));

vi.mock("@/app/updates/client", () => mocks);

vi.mock("@/app/analytics/events", () => ({
  trackUpdateCheckCompleted: vi.fn(),
  trackUpdateCheckStarted: vi.fn(),
  trackUpdateInstallCompleted: vi.fn(),
  trackUpdateInstallFailed: vi.fn(),
  trackUpdateInstallStarted: vi.fn(),
}));

const update = {
  version: "0.1.1",
  currentVersion: "0.1.0",
  notes: "Faster transcription on long files.",
  publishedAt: null,
  releaseTag: "v0.1.1",
  manifestUrl: "https://example.com/update.json",
  platformBehavior: { appExitsDuringInstall: false, supportsRestartLater: true },
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((next) => {
    resolve = next;
  });
  return { promise, resolve };
}

function CheckButton() {
  const { checkForUpdates } = useAppUpdates();
  return (
    <button type="button" onClick={() => void checkForUpdates(true)}>
      Check for updates
    </button>
  );
}

function renderCard() {
  render(
    <NotificationProvider>
      <AppUpdateProvider>
        <CheckButton />
        <UpdateCard />
      </AppUpdateProvider>
    </NotificationProvider>,
  );
}

function card() {
  return screen.getByRole("region", { name: /Izwi 0\.1\.1/ });
}

describe("UpdateCard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.getUpdaterHealthSnapshot.mockResolvedValue({
      enabled: true,
      disableReason: null,
      requestTimeoutMs: 1000,
      maxCheckAttempts: 1,
      retryBackoffMs: 0,
      forcedManifestUrl: null,
    });
    mocks.checkForBetaUpdate.mockResolvedValue(update);
    mocks.installBetaUpdate.mockResolvedValue({
      appExitsDuringInstall: false,
      supportsRestartLater: true,
    });
  });

  it("installs an update, then keeps Restart now through a re-check", async () => {
    renderCard();

    expect(
      await screen.findByRole("heading", { name: "Izwi 0.1.1 is available" }),
    ).toBeVisible();
    // It sits in the notification stack, not on top of it.
    expect(
      within(screen.getByRole("region", { name: "Notifications" })).getByRole(
        "heading",
        { name: "Izwi 0.1.1 is available" },
      ),
    ).toBeVisible();
    fireEvent.click(within(card()).getByRole("button", { name: "Install" }));

    expect(
      await screen.findByRole("heading", { name: "Izwi 0.1.1 is ready to install" }),
    ).toBeVisible();
    expect(card()).toHaveTextContent(
      "Restart to finish. Your history and models stay as they are.",
    );
    expect(within(card()).getByRole("button", { name: "Later" })).toBeVisible();

    const check = deferred<typeof update>();
    mocks.checkForBetaUpdate.mockReturnValueOnce(check.promise);
    fireEvent.click(screen.getByRole("button", { name: "Check for updates" }));
    // Still ready while the check runs, and after it offers the same version.
    expect(within(card()).getByRole("button", { name: "Restart now" })).toBeVisible();
    await act(async () => {
      check.resolve(update);
      await check.promise;
    });
    expect(within(card()).getByRole("button", { name: "Restart now" })).toBeVisible();
    expect(
      within(card()).queryByRole("button", { name: "Install" }),
    ).not.toBeInTheDocument();

    fireEvent.click(within(card()).getByRole("button", { name: "Restart now" }));
    expect(mocks.relaunchAfterUpdate).toHaveBeenCalledOnce();
  });

  it("shows what's new on request and hides on Later", async () => {
    renderCard();

    const whatsNew = await screen.findByRole("button", { name: "What’s new" });
    expect(screen.queryByText(update.notes)).not.toBeInTheDocument();
    fireEvent.click(whatsNew);
    expect(whatsNew).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText(update.notes)).toBeVisible();

    fireEvent.click(within(card()).getByRole("button", { name: "Later" }));
    expect(
      screen.queryByRole("region", { name: /Izwi 0\.1\.1/ }),
    ).not.toBeInTheDocument();
  });

  it("explains a failed install in the card", async () => {
    mocks.installBetaUpdate.mockRejectedValue(
      new Error("Signature verification failed"),
    );
    renderCard();

    fireEvent.click(
      within(await screen.findByRole("region", { name: /Izwi 0\.1\.1/ })).getByRole(
        "button",
        { name: "Install" },
      ),
    );

    await waitFor(() =>
      expect(card()).toHaveTextContent("Signature verification failed"),
    );
    expect(within(card()).getByRole("button", { name: "Install" })).toBeEnabled();
  });

  it("says Izwi is up to date without beta wording", async () => {
    mocks.checkForBetaUpdate.mockResolvedValue(null);
    renderCard();
    await waitFor(() => expect(mocks.checkForBetaUpdate).toHaveBeenCalled());

    fireEvent.click(screen.getByRole("button", { name: "Check for updates" }));

    expect(await screen.findByText("Izwi is up to date")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Notifications" })).not.toHaveTextContent(
      /beta/i,
    );
  });
});
