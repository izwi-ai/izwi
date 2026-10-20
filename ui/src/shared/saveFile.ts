import { invoke, isTauri } from "@tauri-apps/api/core";

/** Header the desktop app reads the suggested file name from. */
const FILENAME_HEADER = "x-izwi-filename";

/** Whether saving asks where to put the file: the desktop app's save dialog. */
export function savesWithDialog(): boolean {
  return isTauri();
}

/**
 * Save a file the user exported. The desktop app shows a native save dialog
 * and writes the file there, because its webview drops `<a download>`
 * clicks; a browser downloads it. Resolves to false when the save dialog
 * was cancelled.
 */
export async function saveFile(
  filename: string,
  data: Blob | string,
  type = "text/plain;charset=utf-8",
): Promise<boolean> {
  const blob = typeof data === "string" ? new Blob([data], { type }) : data;
  if (isTauri()) {
    const bytes = new Uint8Array(await blob.arrayBuffer());
    const savedPath = await invoke<string | null>("save_file", bytes, {
      // Header values are ASCII; names can be anything.
      headers: { [FILENAME_HEADER]: encodeURIComponent(filename) },
    });
    return savedPath !== null;
  }

  const url = URL.createObjectURL(blob);
  clickDownloadLink(url, filename);
  // Revoked later: some browsers read the file after the click returns.
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  return true;
}

/**
 * Save a file the local Izwi server serves, in the desktop app: it asks
 * where, then downloads the file there. Resolves to false when the save
 * dialog was cancelled.
 */
export async function saveServerFile(
  url: string,
  filename: string,
): Promise<boolean> {
  const savedPath = await invoke<string | null>("download_audio_file", {
    url,
    suggestedFilename: filename,
  });
  return savedPath !== null;
}

/** Start a browser download of `href`, named `filename`. */
export function clickDownloadLink(href: string, filename: string): void {
  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = filename;
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}
