export interface ModelDownloadProgressEntry {
  percent: number;
  currentFile: string;
  status: string;
  downloadedBytes: number;
  totalBytes: number;
}

export type ModelDownloadProgressMap = Record<string, ModelDownloadProgressEntry>;
