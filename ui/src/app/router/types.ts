import type { ModelInfo, ModelResidencySummary, RemovedModelInfo } from "@/api";
import type { PipelineRoleKey } from "@/app/providers/ModelCatalogProvider";
import type { ModelDownloadProgressMap } from "@/features/models/downloadProgress";

export interface SharedPageProps {
  models: ModelInfo[];
  selectedModel: string | null;
  /** The page's own pick for a diarization-stack role (no fallback). */
  rolePickFor?: (role: PipelineRoleKey) => string | null;
  onPickRoleModel?: (role: PipelineRoleKey, variant: string | null) => void;
  /** Loads a model; with a role, it also becomes that role's pick. */
  onLoadForRole?: (variant: string, role: PipelineRoleKey | null) => void;
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
  removedModels?: RemovedModelInfo[];
  loading: boolean;
  downloadProgress: ModelDownloadProgressMap;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onSetPinned?: (variant: string, pinned: boolean) => void;
  onRefresh: () => Promise<boolean>;
}
