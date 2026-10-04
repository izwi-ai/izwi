import type { ModelInfo, ModelResidencySummary } from "@/api";
import type { ModelDownloadProgressMap } from "@/features/models/downloadProgress";

export interface SharedPageProps {
  models: ModelInfo[];
  selectedModel: string | null;
  loading: boolean;
  downloadProgress: ModelDownloadProgressMap;
  residencySummary?: ModelResidencySummary | null;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onSelect: (variant: string) => void;
  onError: (message: string) => void;
  onRefresh: () => Promise<boolean>;
}

export interface VoiceRouteProps {
  models: ModelInfo[];
  loading: boolean;
  downloadProgress: ModelDownloadProgressMap;
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
  downloadProgress: ModelDownloadProgressMap;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onRefresh: () => Promise<boolean>;
}
