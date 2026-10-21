import {
  Cpu,
  ExternalLink,
  Info,
  Keyboard,
  Loader2,
  Moon,
  Terminal,
  Power,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  Sun,
  SunMoon,
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { invoke, isTauri } from "@tauri-apps/api/core";

import { api } from "@/api";
import { setAnalyticsEnabled } from "@/app/analytics/client";
import {
  trackAnalyticsConsentChanged,
  trackThemePreferenceChanged,
} from "@/app/analytics/events";
import { useFirstRun } from "@/app/onboarding/FirstRunGate";
import { useNotifications } from "@/app/providers/NotificationProvider";
import { type UpdateStatus, useAppUpdates } from "@/app/providers/AppUpdateProvider";
import { useTheme } from "@/app/providers/ThemeProvider";
import { PageHeader, PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/ui/status-badge";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import { isMacPlatform } from "@/lib/platform";
import { APP_VERSION } from "@/shared/config/runtime";
import { useRuntimeHealth } from "@/shared/hooks/useRuntimeHealth";

type SettingReadStatus = "loading" | "ready" | "error" | "unsupported";

const THEME_OPTIONS: Array<{
  id: "system" | "light" | "dark";
  title: string;
  icon: typeof SunMoon;
}> = [
  {
    id: "system",
    title: "Auto",
    icon: SunMoon,
  },
  {
    id: "light",
    title: "Light",
    icon: Sun,
  },
  {
    id: "dark",
    title: "Dark",
    icon: Moon,
  },
];

function formatDateTime(value: number | string | null): string {
  if (value == null) {
    return "Not checked yet";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Unavailable";
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function getUpdateBadge(
  status: UpdateStatus,
  hasAvailableUpdate: boolean,
  requiredSystem: string | null,
  updatesEnabled: boolean,
  updaterSupported: boolean,
  capabilityStatus: "unsupported" | "loading" | "ready" | "error",
): { label: string; tone: "neutral" | "info" | "success" | "warning" } {
  if (!updaterSupported || capabilityStatus === "unsupported") {
    return { label: "Desktop only", tone: "neutral" };
  }

  if (capabilityStatus === "loading") {
    return { label: "Checking support", tone: "info" };
  }

  if (capabilityStatus === "error") {
    return { label: "Support unknown", tone: "warning" };
  }

  if (!updatesEnabled) {
    return { label: "Updates off", tone: "warning" };
  }

  if (status === "checking") {
    return { label: "Checking", tone: "info" };
  }

  if (status === "downloading") {
    return { label: "Downloading", tone: "info" };
  }

  if (status === "downloaded") {
    return { label: "Ready to restart", tone: "success" };
  }

  if (status === "error") {
    return { label: "Needs attention", tone: "warning" };
  }

  if (status === "available" || hasAvailableUpdate) {
    return { label: "Update available", tone: "info" };
  }

  if (requiredSystem) {
    return { label: `Needs ${requiredSystem}`, tone: "warning" };
  }

  return { label: "Current", tone: "success" };
}

const SETTINGS_SECTIONS: Array<{ id: string; label: string; icon: typeof SunMoon }> = [
  { id: "settings-appearance", label: "Appearance", icon: SunMoon },
  { id: "settings-system", label: "System", icon: Power },
  { id: "settings-runtime", label: "Runtime", icon: Cpu },
  { id: "settings-shortcuts", label: "Shortcuts", icon: Keyboard },
  { id: "settings-privacy", label: "Privacy", icon: ShieldCheck },
  { id: "settings-updates", label: "Updates", icon: RefreshCw },
  { id: "settings-about", label: "About", icon: Info },
];

const SHORTCUTS = [
  {
    label: "Search or jump to",
    detail: "Pages, new items and theme from anywhere.",
    mac: "⌘K",
    other: "Ctrl K",
  },
  {
    label: "New item",
    detail: "On Transcription, Text to Speech, Studio and Voices.",
    mac: "⌘N",
    other: "Ctrl N",
  },
  {
    label: "Create",
    detail: "Submits a New … dialog.",
    mac: "⌘↵",
    other: "Ctrl ↵",
  },
  { label: "Close", detail: "Closes the open dialog or menu.", mac: "Esc", other: "Esc" },
];

const BACKEND_LABELS: Record<string, string> = {
  cpu: "CPU",
  cuda: "CUDA",
  metal: "Metal",
};

function formatGibibytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes)) {
    return "Unknown";
  }
  const gigabytes = bytes / 1024 ** 3;
  return `${gigabytes >= 10 ? gigabytes.toFixed(0) : gigabytes.toFixed(1)} GB`;
}

/** Backend and memory from /v1/health. */
function RuntimeRows() {
  const health = useRuntimeHealth();
  if (health.status !== "ready") {
    return (
      <SettingsRow
        title="Local server"
        description={
          health.status === "loading" ? (
            <span className="inline-flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              Connecting…
            </span>
          ) : (
            "Izwi could not reach its local server."
          )
        }
      />
    );
  }
  const { runtime, version } = health.health;
  const value = (text: string) => (
    <span className="text-[13px] text-[var(--text-secondary)]">{text}</span>
  );
  return (
    <>
      <SettingsRow
        title="Backend"
        action={value(
          BACKEND_LABELS[runtime.selected_backend] ?? runtime.selected_backend,
        )}
      />
      <SettingsRow
        title="Memory"
        description="Reserved by loaded models and running work."
        action={value(
          runtime.memory
            ? `${formatGibibytes(runtime.memory.reserved_bytes)} of ${formatGibibytes(runtime.memory.system_total_bytes)}`
            : "Unknown",
        )}
      />
      <SettingsRow title="Server version" action={value(version)} />
    </>
  );
}

const SWATCH = {
  light: { bg: "#fafaf9", panel: "#ffffff", bar: "#e7e5e4" },
  dark: { bg: "#09090b", panel: "#17181d", bar: "#2a2c33" },
};

/** A tiny window preview for the theme picker. */
function ThemeSwatch({ theme }: { theme: "system" | "light" | "dark" }) {
  const half = (palette: (typeof SWATCH)["light"]) => (
    <div className="flex h-full flex-1 gap-1 p-1" style={{ background: palette.bg }}>
      <div className="w-1/4 rounded-[3px]" style={{ background: palette.bar }} />
      <div className="flex-1 rounded-[3px]" style={{ background: palette.panel }} />
    </div>
  );
  return (
    <div
      aria-hidden="true"
      className="flex h-12 overflow-hidden rounded-[8px] border border-[var(--border-subtle)]"
    >
      {theme === "system" ? (
        <>
          {half(SWATCH.dark)}
          {half(SWATCH.light)}
        </>
      ) : (
        half(SWATCH[theme])
      )}
    </div>
  );
}

function SettingsSection({
  id,
  title,
  description,
  children,
}: {
  id: string;
  title: string;
  description?: ReactNode;
  children: ReactNode;
}) {
  const headingId = `${id}-heading`;
  return (
    <section id={id} aria-labelledby={headingId} className="scroll-mt-6">
      <h2
        id={headingId}
        className="text-[15px] font-semibold text-[var(--text-primary)]"
      >
        {title}
      </h2>
      {description ? (
        <p className="mt-0.5 text-[13px] text-[var(--text-muted)]">
          {description}
        </p>
      ) : null}
      <div className="mt-3 rounded-[14px] border border-[var(--border-subtle)] bg-[var(--surface-panel)] px-4">
        {children}
      </div>
    </section>
  );
}

function SettingsRow({
  title,
  description,
  action,
  children,
  className,
}: {
  title?: ReactNode;
  description?: ReactNode;
  action?: ReactNode;
  children?: ReactNode;
  className?: string;
}) {
  const hasHeaderCopy = Boolean(title || description);

  return (
    <div
      className={cn(
        "border-b border-[var(--border-subtle)] py-3.5 last:border-b-0",
        className,
      )}
    >
      {hasHeaderCopy || action ? (
        <div
          className={cn(
            "flex flex-col md:flex-row md:items-center md:justify-between",
            hasHeaderCopy && action ? "gap-3" : "gap-0",
          )}
        >
          {hasHeaderCopy ? (
            <div className="min-w-0 max-w-2xl">
              {title ? (
                <div className="text-[13.5px] font-medium text-[var(--text-primary)]">
                  {title}
                </div>
              ) : null}
              {description ? (
                <div
                  className={cn(
                    "text-[12.5px] leading-5 text-[var(--text-subtle)]",
                    title ? "mt-0.5" : undefined,
                  )}
                >
                  {description}
                </div>
              ) : null}
            </div>
          ) : null}
          {action ? (
            // ml-auto keeps action-only rows right-aligned; justify-between
            // has no effect with a single child.
            <div className="shrink-0 md:ml-auto">{action}</div>
          ) : null}
        </div>
      ) : null}
      {children ? (
        <div
          className={cn(
            hasHeaderCopy ? "mt-3" : action ? "mt-1.5" : "mt-0",
          )}
        >
          {children}
        </div>
      ) : null}
    </div>
  );
}

type CliInstallStatus = {
  available: boolean;
  installed: boolean;
  location: string | null;
};

function CommandLineToolRow() {
  const { notify } = useNotifications();
  const [status, setStatus] = useState<CliInstallStatus | null>(null);
  const [readStatus, setReadStatus] = useState<SettingReadStatus>("loading");
  const [isInstalling, setIsInstalling] = useState(false);

  useEffect(() => {
    let cancelled = false;
    invoke<CliInstallStatus>("cli_install_status")
      .then((next) => {
        if (!cancelled) {
          setStatus(next);
          setReadStatus("ready");
        }
      })
      .catch((error) => {
        console.error("Failed to read command line tool status:", error);
        if (!cancelled) {
          setReadStatus("error");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleInstall = async () => {
    setIsInstalling(true);
    try {
      const next = await invoke<CliInstallStatus>("install_cli");
      setStatus(next);
      setReadStatus("ready");
      notify({
        title: "Command line tool installed",
        description: next.installed
          ? "Open a new terminal window to use izwi."
          : "Add the install folder to your PATH to use izwi.",
        tone: "success",
      });
    } catch (error) {
      console.error("Failed to install the command line tool:", error);
      notify({
        title: "Could not install the command line tool",
        description: error instanceof Error ? error.message : String(error),
        tone: "warning",
      });
    } finally {
      setIsInstalling(false);
    }
  };

  const unavailable = readStatus === "ready" && status?.available === false;
  const installed = status?.installed === true;

  return (
    <SettingsRow
      title="Command line tool"
      description={
        readStatus === "loading" ? (
          <span className="inline-flex items-center gap-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            Checking for izwi...
          </span>
        ) : readStatus === "error" ? (
          "Izwi could not check whether the izwi command is installed."
        ) : unavailable ? (
          "This build does not include the izwi command."
        ) : installed ? (
          <>
            Installed at{" "}
            <code className="font-mono text-xs text-[var(--text-secondary)]">
              {status?.location}
            </code>
          </>
        ) : (
          "Use Izwi from a terminal with the izwi command."
        )
      }
      action={
        unavailable ? null : (
          <Button
            type="button"
            size="sm"
            variant="outline"
            disabled={readStatus === "loading" || isInstalling}
            onClick={() => void handleInstall()}
          >
            {isInstalling ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Terminal className="h-4 w-4" />
            )}
            {installed ? "Reinstall" : "Install"}
          </Button>
        )
      }
    />
  );
}

/** First run again, in place of the app; downloaded models stay. */
function RunSetupAgainRow() {
  const { notify } = useNotifications();
  const { runSetupAgain } = useFirstRun();
  const [isStarting, setIsStarting] = useState(false);

  const handleRun = async () => {
    setIsStarting(true);
    try {
      await runSetupAgain();
    } catch (error) {
      console.error("Failed to reset onboarding:", error);
      notify({
        title: "Couldn't start setup",
        description: error instanceof Error ? error.message : String(error),
        tone: "danger",
      });
      setIsStarting(false);
    }
  };

  return (
    <SettingsRow
      title="First-run setup"
      description="Choose what to set up on this device again. Downloaded models stay."
      action={
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={isStarting}
          onClick={() => void handleRun()}
        >
          {isStarting ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <RotateCcw className="h-4 w-4" />
          )}
          Run setup again
        </Button>
      }
    />
  );
}

export function SettingsPage() {
  const { notify } = useNotifications();
  const desktopRuntime = isTauri();
  const {
    availableUpdate,
    unsupportedUpdate,
    status: updateStatus,
    lastCheckedAt,
    health: updaterHealth,
    errorMessage: updateErrorMessage,
    updaterSupported,
    capabilityStatus: updaterCapabilityStatus,
    refreshUpdaterCapability,
    openPrompt: openUpdatePrompt,
    checkForUpdates,
  } = useAppUpdates();
  const { themePreference, setThemePreference, resolvedTheme } = useTheme();
  const [analyticsOptIn, setAnalyticsOptIn] = useState<boolean | null>(null);
  const [preferencesReadStatus, setPreferencesReadStatus] =
    useState<SettingReadStatus>("loading");
  const [isSavingPreference, setIsSavingPreference] = useState(false);
  const [trayIconVisible, setTrayIconVisible] = useState<boolean | null>(null);
  const [trayIconReadStatus, setTrayIconReadStatus] = useState<SettingReadStatus>(
    desktopRuntime ? "loading" : "unsupported",
  );
  const [isSavingTrayIconVisibility, setIsSavingTrayIconVisibility] = useState(false);
  const [launchAtLoginEnabled, setLaunchAtLoginEnabled] = useState<
    boolean | null
  >(null);
  const [launchAtLoginReadStatus, setLaunchAtLoginReadStatus] =
    useState<SettingReadStatus>(desktopRuntime ? "loading" : "unsupported");
  const [isSavingLaunchAtLogin, setIsSavingLaunchAtLogin] = useState(false);
  const preferencesRequestRef = useRef(0);
  const trayIconRequestRef = useRef(0);
  const launchAtLoginRequestRef = useRef(0);

  const updatesEnabled = updaterHealth?.enabled === true;
  const updateBadge = useMemo(
    () =>
      getUpdateBadge(
        updateStatus,
        Boolean(availableUpdate),
        unsupportedUpdate?.requiredSystem ?? null,
        updatesEnabled,
        updaterSupported,
        updaterCapabilityStatus,
      ),
    [
      availableUpdate,
      unsupportedUpdate,
      updateStatus,
      updatesEnabled,
      updaterSupported,
      updaterCapabilityStatus,
    ],
  );

  const handleThemePreferenceChange = (
    nextPreference: "system" | "light" | "dark",
  ) => {
    if (themePreference === nextPreference) {
      return;
    }

    setThemePreference(nextPreference);
    void trackThemePreferenceChanged(nextPreference);
  };

  const loadPreferences = useCallback(async () => {
    const requestId = ++preferencesRequestRef.current;
    setAnalyticsOptIn(null);
    setPreferencesReadStatus("loading");
    try {
      const preferences = await api.getPreferences();
      if (requestId !== preferencesRequestRef.current) {
        return;
      }
      setAnalyticsOptIn(preferences.analytics_opt_in);
      setAnalyticsEnabled(preferences.analytics_opt_in);
      setPreferencesReadStatus("ready");
    } catch (error) {
      console.error("Failed to load user preferences:", error);
      if (requestId !== preferencesRequestRef.current) {
        return;
      }
      setAnalyticsOptIn(null);
      setPreferencesReadStatus("error");
      notify({
        title: "Could not load settings",
        description: "Retry to confirm your saved analytics preference.",
        tone: "warning",
      });
    }
  }, [notify]);

  const loadLaunchAtLogin = useCallback(async () => {
    if (!desktopRuntime) {
      setLaunchAtLoginEnabled(null);
      setLaunchAtLoginReadStatus("unsupported");
      return;
    }
    const requestId = ++launchAtLoginRequestRef.current;
    setLaunchAtLoginEnabled(null);
    setLaunchAtLoginReadStatus("loading");
    try {
      const enabled = await invoke<boolean>("launch_at_login_enabled");
      if (requestId !== launchAtLoginRequestRef.current) {
        return;
      }
      setLaunchAtLoginEnabled(enabled);
      setLaunchAtLoginReadStatus("ready");
    } catch (error) {
      console.error("Failed to load launch-at-login preference:", error);
      if (requestId !== launchAtLoginRequestRef.current) {
        return;
      }
      setLaunchAtLoginEnabled(null);
      setLaunchAtLoginReadStatus("error");
      notify({
        title: "Could not load startup preference",
        description: "Retry to confirm the current OS setting.",
        tone: "warning",
      });
    }
  }, [desktopRuntime, notify]);

  const loadTrayIconVisibility = useCallback(async () => {
    if (!desktopRuntime) {
      setTrayIconVisible(null);
      setTrayIconReadStatus("unsupported");
      return;
    }
    const requestId = ++trayIconRequestRef.current;
    setTrayIconVisible(null);
    setTrayIconReadStatus("loading");
    try {
      const visible = await invoke<boolean>("tray_icon_visible");
      if (requestId !== trayIconRequestRef.current) {
        return;
      }
      setTrayIconVisible(visible);
      setTrayIconReadStatus("ready");
    } catch (error) {
      console.error("Failed to load tray icon visibility:", error);
      if (requestId !== trayIconRequestRef.current) {
        return;
      }
      setTrayIconVisible(null);
      setTrayIconReadStatus("error");
      notify({
        title: "Could not load tray visibility",
        description: "Retry to confirm the current desktop setting.",
        tone: "warning",
      });
    }
  }, [desktopRuntime, notify]);

  useEffect(() => {
    void loadPreferences();
    return () => {
      preferencesRequestRef.current += 1;
    };
  }, [loadPreferences]);

  useEffect(() => {
    void loadLaunchAtLogin();
    return () => {
      launchAtLoginRequestRef.current += 1;
    };
  }, [loadLaunchAtLogin]);

  useEffect(() => {
    void loadTrayIconVisibility();
    return () => {
      trayIconRequestRef.current += 1;
    };
  }, [loadTrayIconVisibility]);

  const handleAnalyticsToggle = async (nextValue: boolean) => {
    if (
      preferencesReadStatus !== "ready" ||
      analyticsOptIn === null ||
      isSavingPreference
    ) {
      return;
    }

    const previousValue = analyticsOptIn;
    setAnalyticsOptIn(nextValue);
    setIsSavingPreference(true);

    try {
      const response = await api.updateAnalyticsPreference({ opt_in: nextValue });
      setAnalyticsOptIn(response.analytics_opt_in);
      setAnalyticsEnabled(response.analytics_opt_in);
      void trackAnalyticsConsentChanged(
        response.analytics_opt_in ? "opted_in" : "opted_out",
        "settings",
      );
      notify({
        title: response.analytics_opt_in
          ? "Anonymous analytics enabled"
          : "Anonymous analytics disabled",
        tone: "success",
      });
    } catch (error) {
      console.error("Failed to update analytics preference:", error);
      setAnalyticsOptIn(previousValue);
      notify({
        title: "Could not update analytics preference",
        description: "Please try again.",
        tone: "warning",
      });
    } finally {
      setIsSavingPreference(false);
    }
  };

  const handleLaunchAtLoginToggle = async (nextValue: boolean) => {
    if (
      !desktopRuntime ||
      launchAtLoginReadStatus !== "ready" ||
      launchAtLoginEnabled === null ||
      isSavingLaunchAtLogin
    ) {
      return;
    }

    const previousValue = launchAtLoginEnabled;
    setLaunchAtLoginEnabled(nextValue);
    setIsSavingLaunchAtLogin(true);

    try {
      await invoke("set_launch_at_login_enabled", { enabled: nextValue });
      notify({
        title: nextValue ? "Launch at login enabled" : "Launch at login disabled",
        tone: "success",
      });
    } catch (error) {
      console.error("Failed to update launch-at-login preference:", error);
      setLaunchAtLoginEnabled(previousValue);
      notify({
        title: "Could not update startup preference",
        description: error instanceof Error ? error.message : "Please try again.",
        tone: "warning",
      });
    } finally {
      setIsSavingLaunchAtLogin(false);
    }
  };

  const handleTrayIconVisibilityToggle = async (nextValue: boolean) => {
    if (
      !desktopRuntime ||
      trayIconReadStatus !== "ready" ||
      trayIconVisible === null ||
      isSavingTrayIconVisibility
    ) {
      return;
    }

    const previousValue = trayIconVisible;
    setTrayIconVisible(nextValue);
    setIsSavingTrayIconVisibility(true);

    try {
      await invoke("set_tray_icon_visible", { visible: nextValue });
      notify({
        title: nextValue ? "Tray icon shown" : "Tray icon hidden",
        description: nextValue
          ? "Izwi will keep running in the tray when the window is closed."
          : "Closing the Izwi window now exits the app.",
        tone: "success",
      });
    } catch (error) {
      console.error("Failed to update tray icon visibility:", error);
      setTrayIconVisible(previousValue);
      notify({
        title: "Could not update tray visibility",
        description: error instanceof Error ? error.message : "Please try again.",
        tone: "warning",
      });
    } finally {
      setIsSavingTrayIconVisibility(false);
    }
  };

  return (
    <PageShell className="pb-10">
      <PageHeader title="Settings" description="Preferences for this device." />

      <div className="grid gap-8 lg:grid-cols-[12rem_minmax(0,1fr)]">
        <nav aria-label="Settings sections" className="hidden lg:block">
          <ul className="sticky top-0 space-y-0.5">
            {SETTINGS_SECTIONS.map((section) => (
              <li key={section.id}>
                <button
                  type="button"
                  onClick={() =>
                    document
                      .getElementById(section.id)
                      ?.scrollIntoView({ behavior: "smooth", block: "start" })
                  }
                  className="flex h-8 w-full items-center gap-2.5 rounded-lg px-2.5 text-[13.5px] text-[var(--sidebar-link-title)] transition-colors hover:bg-[var(--sidebar-link-hover-bg)] hover:text-[var(--text-primary)]"
                >
                  <section.icon className="h-4 w-4 text-[var(--text-subtle)]" />
                  {section.label}
                </button>
              </li>
            ))}
          </ul>
        </nav>
        <div className="min-w-0 space-y-8">
          <SettingsSection
            id="settings-appearance"
            title="Appearance"
            description="Choose how Izwi looks. Auto follows your system."
          >
            <SettingsRow>
              <div
                role="group"
                aria-label="Theme"
                className="flex flex-wrap gap-3"
              >
                {THEME_OPTIONS.map((option) => {
                  const isActive = themePreference === option.id;
                  return (
                    <button
                      key={option.id}
                      type="button"
                      onClick={() => handleThemePreferenceChange(option.id)}
                      aria-pressed={isActive}
                      className={cn(
                        "flex w-[7.5rem] flex-col gap-2 rounded-[12px] border p-1.5 pb-2 text-[13px] transition-colors",
                        isActive
                          ? "border-[var(--text-primary)] text-[var(--text-primary)]"
                          : "border-[var(--border-default)] text-[var(--text-muted)] hover:border-[var(--border-strong)]",
                      )}
                    >
                      <ThemeSwatch theme={option.id} />
                      <span className="flex items-center justify-center gap-1.5">
                        {option.title}
                        {option.id === "system" && isActive ? (
                          <span className="text-[11px] text-[var(--text-subtle)]">
                            ({resolvedTheme === "dark" ? "Dark" : "Light"})
                          </span>
                        ) : null}
                      </span>
                    </button>
                  );
                })}
              </div>
            </SettingsRow>
          </SettingsSection>

          <SettingsSection
            id="settings-system"
            title="System"
            description="How the desktop app behaves on this computer."
          >
            {!desktopRuntime ? (
              <SettingsRow
                title="Desktop controls"
                description="Tray behavior, launch-at-login and command line tool settings are available in the packaged desktop app."
              />
            ) : (
              <>
                <SettingsRow
                  title="Show tray icon"
                  description={
                    trayIconReadStatus === "loading" ? (
                  <span className="inline-flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading tray visibility...
                  </span>
                    ) : trayIconReadStatus === "error" ? (
                      "Izwi could not confirm the current tray setting."
                    ) : trayIconVisible ? (
                      "Enabled. Closing the window keeps Izwi running in the tray."
                    ) : (
                      "Disabled. Closing the window exits Izwi."
                    )
                  }
                  action={
                    <div className="flex items-center gap-3">
                      <span className="text-sm font-medium text-[var(--text-secondary)]">
                        {trayIconReadStatus === "ready"
                          ? trayIconVisible
                            ? "On"
                            : "Off"
                          : "Unknown"}
                      </span>
                      <Switch
                        checked={trayIconVisible ?? false}
                        disabled={
                          trayIconReadStatus !== "ready" ||
                          isSavingTrayIconVisibility
                        }
                        onCheckedChange={(checked) =>
                          void handleTrayIconVisibilityToggle(checked)
                        }
                        aria-label="Show tray icon"
                      />
                    </div>
                  }
                >
                  {trayIconReadStatus === "error" ? (
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      onClick={() => void loadTrayIconVisibility()}
                    >
                      Retry tray setting
                    </Button>
                  ) : null}
                  {isSavingTrayIconVisibility ? (
                    <p className="text-sm text-[var(--text-muted)]">
                      Saving tray visibility...
                    </p>
                  ) : null}
                </SettingsRow>

                <SettingsRow
                  title="Launch at login"
                  description={
                    launchAtLoginReadStatus === "loading" ? (
                      <span className="inline-flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Loading startup preference...
                      </span>
                    ) : launchAtLoginReadStatus === "error" ? (
                      "Izwi could not confirm the current startup setting."
                    ) : launchAtLoginEnabled ? (
                      "Izwi opens automatically when you sign in."
                    ) : (
                      "Izwi only opens when launched manually."
                    )
                  }
                  action={
                    <div className="flex items-center gap-3">
                      <span className="text-sm font-medium text-[var(--text-secondary)]">
                        {launchAtLoginReadStatus === "ready"
                          ? launchAtLoginEnabled
                            ? "On"
                            : "Off"
                          : "Unknown"}
                      </span>
                      <Switch
                        checked={launchAtLoginEnabled ?? false}
                        disabled={
                          launchAtLoginReadStatus !== "ready" ||
                          isSavingLaunchAtLogin
                        }
                        onCheckedChange={(checked) =>
                          void handleLaunchAtLoginToggle(checked)
                        }
                        aria-label="Launch Izwi when signing in"
                      />
                    </div>
                  }
                >
                  {launchAtLoginReadStatus === "error" ? (
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      onClick={() => void loadLaunchAtLogin()}
                    >
                      Retry startup setting
                    </Button>
                  ) : null}
                  {isSavingLaunchAtLogin ? (
                    <p className="text-sm text-[var(--text-muted)]">
                      Saving startup preference...
                    </p>
                  ) : null}
                </SettingsRow>

                <CommandLineToolRow />
              </>
            )}
          </SettingsSection>

          <SettingsSection
            id="settings-runtime"
            title="Runtime"
            description="The local inference server on this device."
          >
            <RuntimeRows />
          </SettingsSection>

          <SettingsSection
            id="settings-shortcuts"
            title="Shortcuts"
            description="Keyboard shortcuts available across Izwi."
          >
            {SHORTCUTS.map((shortcut) => (
              <SettingsRow
                key={shortcut.label}
                title={shortcut.label}
                description={shortcut.detail}
                action={
                  <kbd className="rounded-[6px] border border-[var(--border-default)] px-1.5 py-0.5 font-sans text-xs text-[var(--text-secondary)]">
                    {isMacPlatform() ? shortcut.mac : shortcut.other}
                  </kbd>
                }
              />
            ))}
          </SettingsSection>

          <SettingsSection
            id="settings-privacy"
            title="Privacy"
            description="Audio, text and models never leave this device."
          >
            <SettingsRow
              title="Anonymous analytics"
              description={
                preferencesReadStatus === "loading" ? (
                  <span className="inline-flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading saved preference...
                  </span>
                ) : preferencesReadStatus === "error" ? (
                  "Izwi could not confirm your saved analytics preference."
                ) : analyticsOptIn ? (
                  "Enabled. Prompts, transcripts, audio, and personal identifiers are never sent."
                ) : (
                  "Disabled for this device."
                )
              }
              action={
                <div className="flex items-center gap-3">
                  <span className="text-sm font-medium text-[var(--text-secondary)]">
                    {preferencesReadStatus === "ready"
                      ? analyticsOptIn
                        ? "On"
                        : "Off"
                      : "Unknown"}
                  </span>
                  <Switch
                    checked={analyticsOptIn ?? false}
                    disabled={
                      preferencesReadStatus !== "ready" || isSavingPreference
                    }
                    onCheckedChange={(checked) => void handleAnalyticsToggle(checked)}
                    aria-label="Share anonymous usage data"
                  />
                </div>
              }
            >
              {preferencesReadStatus === "error" ? (
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => void loadPreferences()}
                >
                  Retry analytics preference
                </Button>
              ) : null}
              {isSavingPreference ? (
                <p className="text-sm text-[var(--text-muted)]">
                  Saving your preference...
                </p>
              ) : null}
            </SettingsRow>
          </SettingsSection>

          <SettingsSection
            id="settings-updates"
            title="Updates"
            description="Check for new releases and install them."
          >
            <SettingsRow
              description={
                !updaterSupported ? (
                  "Update checks and installation are available in the packaged desktop app."
                ) : updaterCapabilityStatus === "loading" ? (
                  "Checking update support for this installation..."
                ) : updaterCapabilityStatus === "error" ? (
                  "Izwi could not confirm whether updates are supported."
                ) : availableUpdate ? (
                  <>
                    Version{" "}
                    <span className="font-medium text-[var(--text-secondary)]">
                      {availableUpdate.version}
                    </span>{" "}
                    is available.
                  </>
                ) : unsupportedUpdate ? (
                  <>
                    Version{" "}
                    <span className="font-medium text-[var(--text-secondary)]">
                      {unsupportedUpdate.version}
                    </span>{" "}
                    needs {unsupportedUpdate.requiredSystem} or later. The version
                    you have keeps working.
                  </>
                ) : (
                  `Last checked ${formatDateTime(lastCheckedAt)}.`
                )
              }
              action={
                <div className="flex flex-wrap items-center gap-2">
                  <span className="inline-flex items-center gap-1 rounded-full border border-border/70 bg-[var(--bg-surface-0)] px-2.5 py-1 text-xs text-[var(--text-secondary)] whitespace-nowrap">
                    <span>Version</span>
                    <span className="font-mono font-semibold text-[var(--text-primary)]">
                      v{APP_VERSION}
                    </span>
                  </span>
                  <StatusBadge tone={updateBadge.tone}>{updateBadge.label}</StatusBadge>
                  {updaterSupported &&
                  updaterCapabilityStatus === "ready" &&
                  updatesEnabled ? (
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => void checkForUpdates(true)}
                      disabled={
                        updateStatus === "checking" ||
                        updateStatus === "downloading"
                      }
                    >
                      {updateStatus === "checking" ? "Checking..." : "Check now"}
                    </Button>
                  ) : null}
                  {updaterSupported && updaterCapabilityStatus === "error" ? (
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => void refreshUpdaterCapability()}
                    >
                      Retry update support
                    </Button>
                  ) : null}
                  {updaterSupported && availableUpdate ? (
                    <Button
                      type="button"
                      size="sm"
                      onClick={openUpdatePrompt}
                    >
                      View {availableUpdate.version}
                    </Button>
                  ) : null}
                </div>
              }
            >
              {updaterSupported &&
              updaterCapabilityStatus === "ready" &&
              !updatesEnabled ? (
                <p className="text-sm leading-6 text-[var(--status-warning-text)]">
                  {updaterHealth?.disableReason ?? "Updates are disabled."}
                </p>
              ) : null}
              {updateErrorMessage ? (
                <p className="text-sm leading-6 text-[var(--status-warning-text)]">
                  Last error: {updateErrorMessage}
                </p>
              ) : null}
            </SettingsRow>
          </SettingsSection>

          <SettingsSection id="settings-about" title="About">
            <SettingsRow
              title="Izwi"
              description="Local speech, transcription and chat."
              action={
                <span className="font-mono text-[13px] text-[var(--text-secondary)]">
                  v{APP_VERSION}
                </span>
              }
            />
            <RunSetupAgainRow />
            <SettingsRow
              title="Source code"
              action={
                <a
                  href="https://github.com/izwi-ai/izwi"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 text-[13px] text-[var(--text-secondary)] underline underline-offset-4 hover:text-[var(--text-primary)]"
                >
                  github.com/izwi-ai/izwi
                  <ExternalLink className="h-3.5 w-3.5" />
                </a>
              }
            />
          </SettingsSection>
        </div>
      </div>
    </PageShell>
  );
}
