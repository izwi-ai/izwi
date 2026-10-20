import { invoke, isTauri } from "@tauri-apps/api/core";

/** Whether this runtime can show files in the OS file manager: the desktop app. */
export function canRevealPaths(): boolean {
  return isTauri();
}

/**
 * Show a path in Finder (Explorer, or the file manager on Linux). The
 * desktop app only shows paths inside the server's models folder and
 * rejects with the reason otherwise.
 */
export async function revealPath(path: string): Promise<void> {
  await invoke("reveal_path", { path });
}
