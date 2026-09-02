import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { SettingsPage } from "@/features/settings/route";

const coreMocks = vi.hoisted(() => ({
  desktop: true,
  invoke: vi.fn(),
}));

const apiMocks = vi.hoisted(() => ({
  getPreferences: vi.fn(),
  updateAnalyticsPreference: vi.fn(),
  notify: vi.fn(),
}));

const updateState = vi.hoisted(() => ({
  value: {
    availableUpdate: null,
    status: "idle",
    lastCheckedAt: null,
    health: {
      enabled: true,
      disableReason: null,
      requestTimeoutMs: 1000,
      maxCheckAttempts: 1,
      retryBackoffMs: 0,
      forcedManifestUrl: null,
    },
    errorMessage: null,
    updaterSupported: true,
    capabilityStatus: "ready",
    refreshUpdaterCapability: vi.fn(async () => true),
    openPrompt: vi.fn(),
    checkForUpdates: vi.fn(),
  } as Record<string, unknown>,
}));

const availableUpdate = {
  version: "0.1.0-beta-20",
  platformBehavior: {
    appExitsDuringInstall: false,
    supportsRestartLater: true,
  },
};

vi.mock("@tauri-apps/api/core", () => ({
  isTauri: () => coreMocks.desktop,
  invoke: coreMocks.invoke,
}));

vi.mock("@/api", () => ({
  api: {
    getPreferences: apiMocks.getPreferences,
    updateAnalyticsPreference: apiMocks.updateAnalyticsPreference,
  },
}));

vi.mock("@/app/providers/AppUpdateProvider", () => ({
  useAppUpdates: () => updateState.value,
}));

vi.mock("@/app/providers/ThemeProvider", () => ({
  useTheme: () => ({
    themePreference: "system",
    resolvedTheme: "dark",
    setThemePreference: vi.fn(),
  }),
}));

vi.mock("@/app/providers/NotificationProvider", () => ({
  useNotifications: () => ({ notify: apiMocks.notify }),
}));

vi.mock("@/app/analytics/events", () => ({
  trackAnalyticsConsentChanged: vi.fn(),
  trackThemePreferenceChanged: vi.fn(),
}));

describe("SettingsPage capabilities", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, "error").mockImplementation(() => {});
    coreMocks.desktop = true;
    apiMocks.getPreferences.mockResolvedValue({ analytics_opt_in: false });
    apiMocks.updateAnalyticsPreference.mockImplementation(async ({ opt_in }) => ({
      analytics_opt_in: opt_in,
    }));
    coreMocks.invoke.mockImplementation(async (command: string) => {
      if (command === "tray_icon_visible") {
        return true;
      }
      if (command === "launch_at_login_enabled") {
        return false;
      }
      return undefined;
    });
    updateState.value = {
      availableUpdate: null,
      status: "idle",
      lastCheckedAt: null,
      health: {
        enabled: true,
        disableReason: null,
        requestTimeoutMs: 1000,
        maxCheckAttempts: 1,
        retryBackoffMs: 0,
        forcedManifestUrl: null,
      },
      errorMessage: null,
      updaterSupported: true,
      capabilityStatus: "ready",
      refreshUpdaterCapability: vi.fn(async () => true),
      openPrompt: vi.fn(),
      checkForUpdates: vi.fn(),
    };
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("labels desktop-only capabilities on web without rendering no-op controls", async () => {
    coreMocks.desktop = false;
    updateState.value = {
      ...updateState.value,
      health: null,
      updaterSupported: false,
      capabilityStatus: "unsupported",
    };

    render(<SettingsPage />);

    expect(screen.getByText("Desktop Only")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Check now" })).not.toBeInTheDocument();
    expect(screen.getByText(/Update checks and installation are available/i)).toBeVisible();
    expect(screen.getByText(/Tray behavior and launch-at-login settings/i)).toBeVisible();
    expect(
      screen.queryByRole("switch", { name: "Show tray icon" }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("switch", { name: "Launch Izwi when signing in" }),
    ).not.toBeInTheDocument();
    await screen.findByText("Disabled for this device.");
  });

  it("keeps failed preference reads unknown until each retry succeeds", async () => {
    apiMocks.getPreferences
      .mockRejectedValueOnce(new Error("preferences offline"))
      .mockResolvedValueOnce({ analytics_opt_in: true });

    const failedReads = new Set([
      "tray_icon_visible",
      "launch_at_login_enabled",
    ]);
    coreMocks.invoke.mockImplementation(async (command: string) => {
      if (failedReads.delete(command)) {
        throw new Error(`${command} unavailable`);
      }
      return command === "tray_icon_visible";
    });

    render(<SettingsPage />);

    const analyticsSwitch = await screen.findByRole("switch", {
      name: "Share anonymous usage data",
    });
    await screen.findByRole("button", { name: "Retry analytics preference" });
    expect(analyticsSwitch).toBeDisabled();
    expect(screen.getByRole("switch", { name: "Show tray icon" })).toBeDisabled();
    expect(
      screen.getByRole("switch", { name: "Launch Izwi when signing in" }),
    ).toBeDisabled();
    expect(screen.getAllByText("Unknown")).toHaveLength(3);

    fireEvent.click(
      screen.getByRole("button", { name: "Retry analytics preference" }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Retry tray setting" }));
    fireEvent.click(
      screen.getByRole("button", { name: "Retry startup setting" }),
    );

    await waitFor(() => expect(analyticsSwitch).toBeEnabled());
    expect(analyticsSwitch).toBeChecked();
    expect(screen.getByRole("switch", { name: "Show tray icon" })).toBeChecked();
    expect(
      screen.getByRole("switch", { name: "Launch Izwi when signing in" }),
    ).not.toBeChecked();
  });

  it("rolls a failed analytics write back to its last confirmed value", async () => {
    apiMocks.getPreferences.mockResolvedValueOnce({ analytics_opt_in: true });
    apiMocks.updateAnalyticsPreference.mockRejectedValueOnce(
      new Error("write failed"),
    );

    render(<SettingsPage />);

    const analyticsSwitch = await screen.findByRole("switch", {
      name: "Share anonymous usage data",
    });
    await waitFor(() => expect(analyticsSwitch).toBeChecked());

    fireEvent.click(analyticsSwitch);

    await waitFor(() => expect(apiMocks.updateAnalyticsPreference).toHaveBeenCalled());
    await waitFor(() => expect(analyticsSwitch).toBeChecked());
  });

  it.each([
    {
      name: "disabled",
      status: "error",
      health: {
        enabled: false,
        disableReason: "Disabled by policy",
        requestTimeoutMs: 1000,
        maxCheckAttempts: 1,
        retryBackoffMs: 0,
        forcedManifestUrl: null,
      },
      update: null,
      error: null,
      badge: "Updates Off",
      detail: "Disabled by policy",
    },
    {
      name: "checking",
      status: "checking",
      health: updateState.value.health,
      update: null,
      error: null,
      badge: "Checking",
      detail: null,
    },
    {
      name: "available",
      status: "available",
      health: updateState.value.health,
      update: availableUpdate,
      error: null,
      badge: "Update Available",
      detail: "View 0.1.0-beta-20",
    },
    {
      name: "downloading",
      status: "downloading",
      health: updateState.value.health,
      update: availableUpdate,
      error: null,
      badge: "Downloading",
      detail: null,
    },
    {
      name: "install failed",
      status: "available",
      health: updateState.value.health,
      update: availableUpdate,
      error: "Signature verification failed",
      badge: "Update Available",
      detail: "Last error: Signature verification failed",
    },
    {
      name: "restart ready",
      status: "downloaded",
      health: updateState.value.health,
      update: availableUpdate,
      error: null,
      badge: "Ready To Restart",
      detail: "View 0.1.0-beta-20",
    },
  ])("renders the $name updater state honestly", async (testCase) => {
    updateState.value = {
      ...updateState.value,
      status: testCase.status,
      health: testCase.health,
      availableUpdate: testCase.update,
      errorMessage: testCase.error,
    };

    render(<SettingsPage />);

    expect(screen.getByText(testCase.badge)).toBeVisible();
    if (testCase.detail) {
      expect(screen.getByText(testCase.detail)).toBeVisible();
    }
    await screen.findByText("Disabled for this device.");
  });

  it("offers capability retry instead of an updater action when support is unknown", () => {
    updateState.value = {
      ...updateState.value,
      health: null,
      capabilityStatus: "error",
    };

    render(<SettingsPage />);

    expect(screen.getByText("Support Unknown")).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Retry update support" }),
    ).toBeVisible();
    expect(screen.queryByRole("button", { name: "Check now" })).not.toBeInTheDocument();
  });
});
