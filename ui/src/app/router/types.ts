import type { ModelInfo } from "@/api";

export interface DownloadProgressEntry {
  percent: number;
  currentFile: string;
  status: string;
  downloadedBytes: number;
  totalBytes: number;
}

export type DownloadProgressMap = Record<string, DownloadProgressEntry>;

export interface SharedPageProps {
  models: ModelInfo[];
  selectedModel: string | null;
  loading: boolean;
  downloadProgress: DownloadProgressMap;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onSelect: (variant: string) => void;
  onError: (message: string) => void;
  onRefresh: () => Promise<void>;
}

export interface VoiceRouteProps {
  models: ModelInfo[];
  loading: boolean;
  downloadProgress: DownloadProgressMap;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onError?: (message: string) => void;
}

export interface ModelsRouteProps {
  models: ModelInfo[];
  loading: boolean;
  downloadProgress: DownloadProgressMap;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onRefresh: () => Promise<void>;
}
