import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  AppUpdateProvider,
  useAppUpdates,
} from "@/app/providers/AppUpdateProvider";

const coreMocks = vi.hoisted(() => ({
  isTauri: vi.fn(() => true),
}));

const updateMocks = vi.hoisted(() => ({
  checkForBetaUpdate: vi.fn(),
  getUpdaterHealthSnapshot: vi.fn(),
  installBetaUpdate: vi.fn(),
  relaunchAfterUpdate: vi.fn(),
  notify: vi.fn(),
}));

vi.mock("@tauri-apps/api/core", () => ({
  isTauri: coreMocks.isTauri,
}));

vi.mock("@/app/updates/client", () => ({
  checkForBetaUpdate: updateMocks.checkForBetaUpdate,
  getUpdaterHealthSnapshot: updateMocks.getUpdaterHealthSnapshot,
  installBetaUpdate: updateMocks.installBetaUpdate,
  relaunchAfterUpdate: updateMocks.relaunchAfterUpdate,
}));

vi.mock("@/app/providers/NotificationProvider", () => ({
  useNotifications: () => ({ notify: updateMocks.notify }),
}));

vi.mock("@/app/analytics/events", () => ({
  trackUpdateCheckCompleted: vi.fn(),
  trackUpdateCheckStarted: vi.fn(),
  trackUpdateInstallCompleted: vi.fn(),
  trackUpdateInstallFailed: vi.fn(),
  trackUpdateInstallStarted: vi.fn(),
}));

const health = {
  enabled: true,
  disableReason: null,
  requestTimeoutMs: 1000,
  maxCheckAttempts: 1,
  retryBackoffMs: 0,
  forcedManifestUrl: null,
};

const availableUpdate = {
  version: "0.1.1",
  currentVersion: "0.1.0",
  notes: null,
  publishedAt: null,
  releaseTag: "v0.1.1",
  manifestUrl: "https://example.com/update.json",
  platformBehavior: {
    appExitsDuringInstall: false,
    supportsRestartLater: true,
  },
};

function deferredPromise<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((nextResolve) => {
    resolve = nextResolve;
  });
  return { promise, resolve };
}

function UpdateProbe() {
  const {
    status,
    availableUpdate: offered,
    downloadedVersion,
    capabilityStatus,
    updaterSupported,
    checkForUpdates,
    installUpdate,
  } = useAppUpdates();
  return (
    <div>
      <span>{updaterSupported ? "supported" : "unsupported"}</span>
      <span data-testid="capability-status">{capabilityStatus}</span>
      <span data-testid="update-status">{status}</span>
      <span data-testid="offered-version">{offered?.version ?? "none"}</span>
      <span data-testid="downloaded-version">{downloadedVersion ?? "none"}</span>
      <button type="button" onClick={() => void checkForUpdates(true)}>
        Check
      </button>
      <button type="button" onClick={() => void installUpdate()}>
        Install
      </button>
    </div>
  );
}

function renderProvider() {
  return render(
    <AppUpdateProvider>
      <UpdateProbe />
    </AppUpdateProvider>,
  );
}

describe("AppUpdateProvider capabilities", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    coreMocks.isTauri.mockReturnValue(true);
    updateMocks.getUpdaterHealthSnapshot.mockResolvedValue(health);
    updateMocks.checkForBetaUpdate.mockResolvedValue(null);
  });

  it("does not check for updates before confirming updater health", async () => {
    const healthRead = deferredPromise<typeof health>();
    updateMocks.getUpdaterHealthSnapshot.mockReturnValue(healthRead.promise);

    renderProvider();

    expect(screen.getByTestId("capability-status")).toHaveTextContent("loading");
    expect(updateMocks.checkForBetaUpdate).not.toHaveBeenCalled();

    await act(async () => {
      healthRead.resolve(health);
      await healthRead.promise;
    });

    await waitFor(() =>
      expect(updateMocks.checkForBetaUpdate).toHaveBeenCalledTimes(1),
    );
    expect(screen.getByTestId("capability-status")).toHaveTextContent("ready");
  });

  it("suppresses rapid duplicate update checks", async () => {
    renderProvider();
    await waitFor(() =>
      expect(updateMocks.checkForBetaUpdate).toHaveBeenCalledTimes(1),
    );

    updateMocks.checkForBetaUpdate.mockClear();
    const check = deferredPromise<null>();
    updateMocks.checkForBetaUpdate.mockReturnValue(check.promise);

    fireEvent.click(screen.getByRole("button", { name: "Check" }));
    fireEvent.click(screen.getByRole("button", { name: "Check" }));

    expect(updateMocks.checkForBetaUpdate).toHaveBeenCalledTimes(1);
    await act(async () => {
      check.resolve(null);
      await check.promise;
    });
  });

  it("suppresses rapid duplicate update installs", async () => {
    updateMocks.checkForBetaUpdate.mockResolvedValue(availableUpdate);
    const install = deferredPromise<{
      appExitsDuringInstall: boolean;
      supportsRestartLater: boolean;
    }>();
    updateMocks.installBetaUpdate.mockReturnValue(install.promise);

    renderProvider();
    await waitFor(() =>
      expect(screen.getByTestId("update-status")).toHaveTextContent("available"),
    );

    fireEvent.click(screen.getByRole("button", { name: "Install" }));
    fireEvent.click(screen.getByRole("button", { name: "Install" }));

    expect(updateMocks.installBetaUpdate).toHaveBeenCalledTimes(1);
    await act(async () => {
      install.resolve({
        appExitsDuringInstall: false,
        supportsRestartLater: true,
      });
      await install.promise;
    });
  });
});

describe("AppUpdateProvider re-checks after a download", () => {
  const installed = { appExitsDuringInstall: false, supportsRestartLater: true };

  beforeEach(() => {
    vi.clearAllMocks();
    coreMocks.isTauri.mockReturnValue(true);
    updateMocks.getUpdaterHealthSnapshot.mockResolvedValue(health);
    updateMocks.checkForBetaUpdate.mockResolvedValue(availableUpdate);
    updateMocks.installBetaUpdate.mockResolvedValue(installed);
  });

  async function renderDownloaded() {
    renderProvider();
    await waitFor(() =>
      expect(screen.getByTestId("update-status")).toHaveTextContent("available"),
    );
    fireEvent.click(screen.getByRole("button", { name: "Install" }));
    await waitFor(() =>
      expect(screen.getByTestId("update-status")).toHaveTextContent("downloaded"),
    );
    expect(screen.getByTestId("downloaded-version")).toHaveTextContent(
      availableUpdate.version,
    );
  }

  async function checkAgain() {
    const calls = updateMocks.checkForBetaUpdate.mock.calls.length;
    fireEvent.click(screen.getByRole("button", { name: "Check" }));
    await waitFor(() =>
      expect(updateMocks.checkForBetaUpdate).toHaveBeenCalledTimes(calls + 1),
    );
    await waitFor(() =>
      expect(screen.getByTestId("update-status")).not.toHaveTextContent("checking"),
    );
  }

  it("stays ready to restart when the downloaded version is offered again", async () => {
    await renderDownloaded();

    await checkAgain();

    expect(screen.getByTestId("update-status")).toHaveTextContent("downloaded");
    expect(screen.getByTestId("offered-version")).toHaveTextContent(
      availableUpdate.version,
    );
    expect(updateMocks.installBetaUpdate).toHaveBeenCalledTimes(1);
  });

  it("does not download the same version twice", async () => {
    await renderDownloaded();

    fireEvent.click(screen.getByRole("button", { name: "Install" }));

    expect(updateMocks.installBetaUpdate).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("update-status")).toHaveTextContent("downloaded");
  });

  it("stays ready when a re-check finds nothing or fails", async () => {
    await renderDownloaded();

    updateMocks.checkForBetaUpdate.mockResolvedValueOnce(null);
    await checkAgain();
    expect(screen.getByTestId("update-status")).toHaveTextContent("downloaded");
    expect(screen.getByTestId("offered-version")).toHaveTextContent(
      availableUpdate.version,
    );
    expect(updateMocks.notify).not.toHaveBeenCalled();

    updateMocks.checkForBetaUpdate.mockRejectedValueOnce(new Error("Offline"));
    await checkAgain();
    expect(screen.getByTestId("update-status")).toHaveTextContent("downloaded");
  });

  it("offers a newer release over the one waiting for a restart", async () => {
    await renderDownloaded();

    updateMocks.checkForBetaUpdate.mockResolvedValueOnce({
      ...availableUpdate,
      version: "0.1.2",
      releaseTag: "v0.1.2",
    });
    await checkAgain();

    expect(screen.getByTestId("update-status")).toHaveTextContent("available");
    expect(screen.getByTestId("offered-version")).toHaveTextContent("0.1.2");
    expect(screen.getByTestId("downloaded-version")).toHaveTextContent(
      availableUpdate.version,
    );
  });
});
