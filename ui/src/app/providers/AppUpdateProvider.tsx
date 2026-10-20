import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { isTauri } from "@tauri-apps/api/core";

import {
  trackUpdateCheckCompleted,
  trackUpdateCheckStarted,
  trackUpdateInstallCompleted,
  trackUpdateInstallFailed,
  trackUpdateInstallStarted,
} from "@/app/analytics/events";
import { useNotifications } from "@/app/providers/NotificationProvider";
import {
  checkForBetaUpdate,
  getUpdaterHealthSnapshot,
  installBetaUpdate,
  relaunchAfterUpdate,
  type AppUpdateMetadata,
  type UpdaterHealthStatus,
} from "@/app/updates/client";

const UPDATE_CHECK_INTERVAL_MS = 6 * 60 * 60 * 1000;

export type UpdateStatus =
  | "idle"
  | "checking"
  | "available"
  | "downloading"
  | "downloaded"
  | "error";

export type UpdaterCapabilityStatus =
  | "unsupported"
  | "loading"
  | "ready"
  | "error";

interface AppUpdateContextValue {
  availableUpdate: AppUpdateMetadata | null;
  status: UpdateStatus;
  /** Version installed and waiting for a restart; it stays ready through re-checks. */
  downloadedVersion: string | null;
  lastCheckedAt: number | null;
  errorMessage: string | null;
  isPromptOpen: boolean;
  progressPercent: number | null;
  health: UpdaterHealthStatus | null;
  updaterSupported: boolean;
  capabilityStatus: UpdaterCapabilityStatus;
  refreshUpdaterCapability: () => Promise<boolean>;
  checkForUpdates: (manual?: boolean) => Promise<void>;
  installUpdate: () => Promise<void>;
  restartToApply: () => Promise<void>;
  openPrompt: () => void;
  dismissPrompt: () => void;
}

const AppUpdateContext = createContext<AppUpdateContextValue | null>(null);

interface AppUpdateProviderProps {
  children: ReactNode;
}

export function AppUpdateProvider({ children }: AppUpdateProviderProps) {
  const { notify } = useNotifications();
  const updaterSupported = isTauri();
  const [availableUpdate, setAvailableUpdate] = useState<AppUpdateMetadata | null>(
    null,
  );
  const [status, setStatus] = useState<UpdateStatus>("idle");
  const [downloadedVersion, setDownloadedVersion] = useState<string | null>(null);
  const [lastCheckedAt, setLastCheckedAt] = useState<number | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isPromptOpen, setIsPromptOpen] = useState(false);
  const [progressPercent, setProgressPercent] = useState<number | null>(null);
  const [health, setHealth] = useState<UpdaterHealthStatus | null>(null);
  const [capabilityStatus, setCapabilityStatus] =
    useState<UpdaterCapabilityStatus>(
      updaterSupported ? "loading" : "unsupported",
    );
  const statusRef = useRef<UpdateStatus>("idle");
  const downloadedVersionRef = useRef<string | null>(null);
  const healthRef = useRef<UpdaterHealthStatus | null>(null);

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  useEffect(() => {
    healthRef.current = health;
  }, [health]);

  const refreshUpdaterCapability = useCallback(async () => {
    if (!updaterSupported) {
      healthRef.current = null;
      setHealth(null);
      setCapabilityStatus("unsupported");
      return false;
    }

    setCapabilityStatus("loading");
    try {
      const snapshot = await getUpdaterHealthSnapshot();
      if (!snapshot) {
        healthRef.current = null;
        setHealth(null);
        setCapabilityStatus("error");
        return false;
      }
      healthRef.current = snapshot;
      setHealth(snapshot);
      setCapabilityStatus("ready");
      return true;
    } catch {
      healthRef.current = null;
      setHealth(null);
      setCapabilityStatus("error");
      return false;
    }
  }, [updaterSupported]);

  const checkForUpdates = useCallback(
    async (manual = false) => {
      if (!updaterSupported) {
        return;
      }
      const currentHealth = healthRef.current;
      if (currentHealth && !currentHealth.enabled) {
        statusRef.current = "error";
        setStatus("error");
        setErrorMessage(currentHealth.disableReason ?? "Updates are disabled.");
        if (manual) {
          notify({
            title: "Updates are disabled",
            description:
              currentHealth.disableReason ?? "Runtime policy has disabled updates.",
            tone: "warning",
          });
        }
        return;
      }
      const currentStatus = statusRef.current;
      if (currentStatus === "checking" || currentStatus === "downloading") {
        return;
      }

      statusRef.current = "checking";
      setStatus("checking");
      setErrorMessage(null);
      setProgressPercent(null);
      void trackUpdateCheckStarted(manual ? "manual" : "background");

      const downloaded = downloadedVersionRef.current;
      try {
        const update = await checkForBetaUpdate();
        setLastCheckedAt(Date.now());

        // An update waiting for a restart stays ready unless a newer one is out.
        if (downloaded && (!update || update.version === downloaded)) {
          if (update) {
            setAvailableUpdate(update);
          }
          statusRef.current = "downloaded";
          setStatus("downloaded");
          if (manual) {
            setIsPromptOpen(true);
          }
          void trackUpdateCheckCompleted(
            update ? "update_available" : "no_update",
            update?.version,
          );
          return;
        }

        if (!update) {
          setAvailableUpdate(null);
          statusRef.current = "idle";
          setStatus("idle");
          void trackUpdateCheckCompleted("no_update");
          if (manual) {
            notify({ title: "Izwi is up to date", tone: "success" });
          }
          return;
        }

        // The update card opens and says it; no separate notification.
        setAvailableUpdate(update);
        statusRef.current = "available";
        setStatus("available");
        setIsPromptOpen(true);
        void trackUpdateCheckCompleted("update_available", update.version);
      } catch (error) {
        const message =
          error instanceof Error ? error.message : "Unknown update-check error.";
        if (currentStatus === "downloaded") {
          statusRef.current = "downloaded";
          setStatus("downloaded");
        } else {
          statusRef.current = "error";
          setStatus("error");
          setErrorMessage(message);
        }
        void trackUpdateCheckCompleted("failed", undefined, message);
        if (manual) {
          notify({
            title: "Couldn’t check for updates",
            description: message,
            tone: "warning",
          });
        }
      }
    },
    [notify, updaterSupported],
  );

  const installUpdate = useCallback(async () => {
    if (!availableUpdate || !updaterSupported) {
      return;
    }
    if (
      statusRef.current === "downloading" ||
      downloadedVersionRef.current === availableUpdate.version
    ) {
      return;
    }

    statusRef.current = "downloading";
    setStatus("downloading");
    setErrorMessage(null);
    setProgressPercent(0);
    void trackUpdateInstallStarted(availableUpdate.version);

    let downloadedBytes = 0;
    let contentLength: number | null = null;

    try {
      const result = await installBetaUpdate((event) => {
        if (event.event === "Started") {
          contentLength = event.data.contentLength ?? null;
          setProgressPercent(contentLength ? 0 : null);
          return;
        }
        if (event.event === "Progress") {
          downloadedBytes += event.data.chunkLength;
          contentLength = event.data.contentLength ?? contentLength;
          if (contentLength && contentLength > 0) {
            const nextPercent = Math.min(
              100,
              Math.round((downloadedBytes / contentLength) * 100),
            );
            setProgressPercent(nextPercent);
          }
          return;
        }
        setProgressPercent(100);
      });

      void trackUpdateInstallCompleted(availableUpdate.version);

      if (result.appExitsDuringInstall) {
        notify({
          title: "Installing the update",
          description: "Izwi closes to run the installer.",
          tone: "info",
        });
        return;
      }

      downloadedVersionRef.current = availableUpdate.version;
      setDownloadedVersion(availableUpdate.version);
      statusRef.current = "downloaded";
      setStatus("downloaded");
      setIsPromptOpen(true);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Unknown update-install error.";
      statusRef.current = "available";
      setStatus("available");
      setErrorMessage(message);
      void trackUpdateInstallFailed(message);
      // The update card shows the reason.
      setIsPromptOpen(true);
    }
  }, [availableUpdate, notify, updaterSupported]);

  const restartToApply = useCallback(async () => {
    await relaunchAfterUpdate();
  }, []);

  const openPrompt = useCallback(() => {
    if (!availableUpdate) {
      return;
    }
    setIsPromptOpen(true);
  }, [availableUpdate]);

  const dismissPrompt = useCallback(() => {
    setIsPromptOpen(false);
  }, []);

  useEffect(() => {
    if (!updaterSupported) {
      return;
    }

    void refreshUpdaterCapability().then((available) => {
      if (available) {
        void checkForUpdates(false);
      }
    });
    const timer = window.setInterval(() => {
      void checkForUpdates(false);
    }, UPDATE_CHECK_INTERVAL_MS);

    return () => {
      window.clearInterval(timer);
    };
  }, [checkForUpdates, refreshUpdaterCapability, updaterSupported]);

  const value = useMemo<AppUpdateContextValue>(
    () => ({
      availableUpdate,
      status,
      downloadedVersion,
      lastCheckedAt,
      errorMessage,
      isPromptOpen,
      progressPercent,
      health,
      updaterSupported,
      capabilityStatus,
      refreshUpdaterCapability,
      checkForUpdates,
      installUpdate,
      restartToApply,
      openPrompt,
      dismissPrompt,
    }),
    [
      availableUpdate,
      status,
      downloadedVersion,
      lastCheckedAt,
      errorMessage,
      isPromptOpen,
      progressPercent,
      health,
      updaterSupported,
      capabilityStatus,
      refreshUpdaterCapability,
      checkForUpdates,
      installUpdate,
      restartToApply,
      openPrompt,
      dismissPrompt,
    ],
  );

  return (
    <AppUpdateContext.Provider value={value}>{children}</AppUpdateContext.Provider>
  );
}

export function useAppUpdates() {
  const context = useContext(AppUpdateContext);

  if (!context) {
    throw new Error("useAppUpdates must be used within AppUpdateProvider");
  }

  return context;
}
