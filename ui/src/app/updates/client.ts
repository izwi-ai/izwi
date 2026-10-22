import { Channel, invoke, isTauri } from "@tauri-apps/api/core";

export interface PlatformInstallBehavior {
  appExitsDuringInstall: boolean;
  supportsRestartLater: boolean;
}

export interface AppUpdateMetadata {
  version: string;
  currentVersion: string;
  notes: string | null;
  publishedAt: string | null;
  releaseTag: string;
  manifestUrl: string;
  platformBehavior: PlatformInstallBehavior;
  /** Set to the system it needs ("macOS 13.3") when this one is too old to install it. */
  requiredSystem: string | null;
}

export interface InstallResult {
  appExitsDuringInstall: boolean;
  supportsRestartLater: boolean;
}

export interface UpdaterHealthStatus {
  enabled: boolean;
  disableReason: string | null;
  requestTimeoutMs: number;
  maxCheckAttempts: number;
  retryBackoffMs: number;
  forcedManifestUrl: string | null;
}

export type DownloadEvent =
  | { event: "Started"; data: { contentLength: number | null } }
  | {
      event: "Progress";
      data: { chunkLength: number; contentLength: number | null };
    }
  | { event: "Finished" };

export async function checkForBetaUpdate(): Promise<AppUpdateMetadata | null> {
  if (!isTauri()) {
    return null;
  }

  return invoke<AppUpdateMetadata | null>("check_for_beta_update");
}

export async function installBetaUpdate(
  onEvent: (event: DownloadEvent) => void,
): Promise<InstallResult> {
  if (!isTauri()) {
    throw new Error("In-app update install is only available in desktop mode.");
  }

  const channel = new Channel<DownloadEvent>();
  channel.onmessage = (event) => {
    onEvent(event);
  };

  return invoke<InstallResult>("install_beta_update", { onEvent: channel });
}

export async function relaunchAfterUpdate(): Promise<void> {
  if (!isTauri()) {
    return;
  }

  await invoke("relaunch_after_update");
}

export async function getUpdaterHealthSnapshot(): Promise<UpdaterHealthStatus | null> {
  if (!isTauri()) {
    return null;
  }

  return invoke<UpdaterHealthStatus>("updater_health_snapshot");
}
