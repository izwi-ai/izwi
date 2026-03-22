import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
} from "react";
import { createPortal } from "react-dom";
import {
  AlertCircle,
  AlertTriangle,
  ChevronLeft,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  Download,
  FileAudio,
  FilePlus2,
  Library,
  Loader2,
  PencilLine,
  Play,
  Settings2,
  Trash2,
  Upload,
  Link2,
  History,
  Waves,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  api,
  type ModelInfo,
  type SavedVoiceSummary,
  type TtsProjectFolderRecord,
  type TtsProjectMetaRecord,
  type TtsProjectPronunciationRecord,
  type TtsProjectRecord,
  type TtsProjectSnapshotRecord,
  type TtsProjectSummary,
  type TtsProjectVoiceMode,
} from "@/api";
import {
  TEXT_TO_SPEECH_PREFERRED_MODELS,
  VOICE_CLONING_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import type { VoicePickerItem } from "@/components/VoicePicker";
import { VoiceSelect } from "@/components/VoiceSelect";
import { RouteModelSelect } from "@/components/RouteModelSelect";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Card } from "@/components/ui/card";
import { StudioWorkspaceScaffold } from "@/features/studio/components/StudioWorkspaceScaffold";
import { StudioProjectUtilities } from "@/features/studio/components/StudioProjectUtilities";
import { useDownloadIndicator } from "@/utils/useDownloadIndicator";
import { getSpeakerProfilesForVariant } from "@/types";

interface ModelOption {
  value: string;
  label: string;
  statusLabel: string;
  isReady: boolean;
}

interface StudioWorkspaceProps {
  activeProjectId?: string | null;
  selectedModel: string | null;
  selectedModelInfo: ModelInfo | null;
  availableModels: ModelInfo[];
  modelOptions?: ModelOption[];
  headerActionContainer?: HTMLElement | null;
  onNavigateProject?: (projectId: string | null) => void;
  onSelectModel?: (variant: string) => void;
  onOpenModelManager?: () => void;
  onModelRequired: () => void;
  onError: (message: string) => void;
}

interface StudioRenderQueueItem {
  id: string;
  projectId: string;
  segmentId: string;
  segmentLabel: string;
  status: "queued" | "running" | "failed";
  errorMessage?: string;
  jobId?: string;
}

const STUDIO_RENDER_QUEUE_STORAGE_KEY = "izwi.studio.render.queue.v1";
const STUDIO_UNFILED_FOLDER_VALUE = "__studio-unfiled-folder__";

const SAVED_VOICE_RENDERER_PREFERRED_MODELS = [
  ...VOICE_CLONING_PREFERRED_MODELS,
] as const;

function formatRelativeDate(timestampMs: number): string {
  const value = new Date(timestampMs);
  if (Number.isNaN(value.getTime())) {
    return "Unknown";
  }
  return value.toLocaleDateString([], {
    month: "short",
    day: "numeric",
  });
}

function sourceLabel(source: SavedVoiceSummary["source_route_kind"]): string {
  return source === "voice_cloning" ? "Cloned voice" : "Designed voice";
}

function downloadTextFile(filename: string, text: string): void {
  if (typeof window === "undefined") {
    return;
  }
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.URL.revokeObjectURL(url);
}

export function StudioWorkspace({
  activeProjectId,
  selectedModel,
  selectedModelInfo,
  availableModels,
  modelOptions = [],
  headerActionContainer,
  onNavigateProject,
  onSelectModel,
  onOpenModelManager,
  onModelRequired,
  onError,
}: StudioWorkspaceProps) {
  const [projects, setProjects] = useState<TtsProjectSummary[]>([]);
  const [projectsLoading, setProjectsLoading] = useState(false);
  const [projectFolders, setProjectFolders] = useState<TtsProjectFolderRecord[]>([]);
  const [projectMetaById, setProjectMetaById] = useState<
    Record<string, TtsProjectMetaRecord>
  >({});
  const [projectSearch, setProjectSearch] = useState("");
  const [projectStatusFilter, setProjectStatusFilter] = useState<
    "all" | "in_progress" | "ready"
  >("all");
  const [projectSort, setProjectSort] = useState<"recent" | "name" | "progress">(
    "recent",
  );
  const [projectFolderFilter, setProjectFolderFilter] = useState("all");
  const [selectedProjectIdState, setSelectedProjectIdState] = useState<
    string | null
  >(null);
  const [projectPronunciations, setProjectPronunciations] = useState<
    TtsProjectPronunciationRecord[]
  >([]);
  const [projectPronunciationsLoading, setProjectPronunciationsLoading] =
    useState(false);
  const [newPronunciationSource, setNewPronunciationSource] = useState("");
  const [newPronunciationReplacement, setNewPronunciationReplacement] =
    useState("");
  const [savingPronunciation, setSavingPronunciation] = useState(false);
  const [projectSnapshots, setProjectSnapshots] = useState<
    TtsProjectSnapshotRecord[]
  >([]);
  const [projectSnapshotsLoading, setProjectSnapshotsLoading] = useState(false);
  const [snapshotLabel, setSnapshotLabel] = useState("");
  const [savingSnapshot, setSavingSnapshot] = useState(false);
  const [restoringSnapshotId, setRestoringSnapshotId] = useState<string | null>(
    null,
  );
  const [selectedProject, setSelectedProject] = useState<TtsProjectRecord | null>(
    null,
  );
  const [projectLoading, setProjectLoading] = useState(false);
  const [savedVoices, setSavedVoices] = useState<SavedVoiceSummary[]>([]);
  const [savedVoicesLoading, setSavedVoicesLoading] = useState(false);
  const [savedVoicesError, setSavedVoicesError] = useState<string | null>(null);
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectText, setNewProjectText] = useState("");
  const [newProjectFilename, setNewProjectFilename] = useState("");
  const [newProjectSourceUrl, setNewProjectSourceUrl] = useState("");
  const [importingFromUrl, setImportingFromUrl] = useState(false);
  const [isCreateProjectDialogOpen, setIsCreateProjectDialogOpen] =
    useState(false);
  const [creatingProject, setCreatingProject] = useState(false);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);
  const [workspaceStatus, setWorkspaceStatus] = useState<{
    tone: "success" | "error";
    message: string;
  } | null>(null);
  const [projectName, setProjectName] = useState("");
  const [projectModelId, setProjectModelId] = useState(selectedModel ?? "");
  const [projectFolderId, setProjectFolderId] = useState("");
  const [projectVoiceMode, setProjectVoiceMode] =
    useState<TtsProjectVoiceMode>("built_in");
  const [projectSpeaker, setProjectSpeaker] = useState("Vivian");
  const [projectSavedVoiceId, setProjectSavedVoiceId] = useState("");
  const [projectSpeed, setProjectSpeed] = useState(1);
  const [savingProject, setSavingProject] = useState(false);
  const [savingSegmentId, setSavingSegmentId] = useState<string | null>(null);
  const [renderingSegmentId, setRenderingSegmentId] = useState<string | null>(
    null,
  );
  const [renderingAll, setRenderingAll] = useState(false);
  const [renderQueue, setRenderQueue] = useState<StudioRenderQueueItem[]>([]);
  const [renderQueueReady, setRenderQueueReady] = useState(false);
  const [processingRenderQueue, setProcessingRenderQueue] = useState(false);
  const [exportFormat, setExportFormat] = useState<"wav" | "raw_i16" | "raw_f32">(
    "wav",
  );
  const [exportScope, setExportScope] = useState<"all" | "selected">("all");
  const [exportIncludeScript, setExportIncludeScript] = useState(false);
  const [deletingProject, setDeletingProject] = useState(false);
  const [deleteProjectTarget, setDeleteProjectTarget] = useState<{
    id: string;
    name: string;
  } | null>(null);
  const [deleteProjectError, setDeleteProjectError] = useState<string | null>(null);
  const [deletingSegment, setDeletingSegment] = useState(false);
  const [deleteSegmentTarget, setDeleteSegmentTarget] = useState<{
    id: string;
    position: number;
    preview: string;
  } | null>(null);
  const [deleteSegmentError, setDeleteSegmentError] = useState<string | null>(null);
  const [segmentDrafts, setSegmentDrafts] = useState<Record<string, string>>({});
  const [segmentSelections, setSegmentSelections] = useState<
    Record<string, number | null>
  >({});
  const [selectedSegmentIds, setSelectedSegmentIds] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const {
    downloadState,
    downloadMessage,
    isDownloading,
    beginDownload,
    completeDownload,
    failDownload,
  } = useDownloadIndicator();
  const selectedProjectId =
    activeProjectId !== undefined ? activeProjectId : selectedProjectIdState;
  const setSelectedProjectId = useCallback(
    (nextProjectId: string | null | ((current: string | null) => string | null)) => {
      if (activeProjectId !== undefined) {
        const resolvedNextId =
          typeof nextProjectId === "function"
            ? nextProjectId(activeProjectId)
            : nextProjectId;
        onNavigateProject?.(resolvedNextId);
        return;
      }
      setSelectedProjectIdState(nextProjectId);
    },
    [activeProjectId, onNavigateProject],
  );

  const currentProjectModelInfo = useMemo(
    () =>
      availableModels.find((model) => model.variant === projectModelId) ?? null,
    [availableModels, projectModelId],
  );
  const currentProjectCapabilities =
    currentProjectModelInfo?.speech_capabilities ?? null;
  const supportsBuiltInVoices =
    currentProjectCapabilities?.supports_builtin_voices ?? false;
  const supportsSavedVoices =
    currentProjectCapabilities?.supports_reference_voice ?? false;
  const supportsSpeedControl =
    currentProjectCapabilities?.supports_speed_control ?? false;
  const projectCompatibleModels = useMemo(
    () =>
      availableModels.filter(
        (model) =>
          model.speech_capabilities?.supports_builtin_voices ||
          model.speech_capabilities?.supports_reference_voice,
      ),
    [availableModels],
  );
  const projectCompatibleVariantSet = useMemo(
    () => new Set(projectCompatibleModels.map((model) => model.variant)),
    [projectCompatibleModels],
  );
  const projectModelOptions = useMemo(
    () =>
      modelOptions.filter((option) =>
        projectCompatibleVariantSet.has(option.value),
      ),
    [modelOptions, projectCompatibleVariantSet],
  );

  const builtInCompatibleModels = useMemo(
    () =>
      projectCompatibleModels.filter(
        (model) => model.speech_capabilities?.supports_builtin_voices,
      ),
    [projectCompatibleModels],
  );
  const savedVoiceCompatibleModels = useMemo(
    () =>
      projectCompatibleModels.filter(
        (model) => model.speech_capabilities?.supports_reference_voice,
      ),
    [projectCompatibleModels],
  );

  const availableSpeakers = useMemo(
    () =>
      supportsBuiltInVoices ? getSpeakerProfilesForVariant(projectModelId) : [],
    [projectModelId, supportsBuiltInVoices],
  );
  const selectedVoiceName = useMemo(() => {
    if (projectVoiceMode === "saved") {
      return (
        savedVoices.find((voice) => voice.id === projectSavedVoiceId)?.name ?? null
      );
    }
    return availableSpeakers.find((voice) => voice.id === projectSpeaker)?.name ?? null;
  }, [
    availableSpeakers,
    projectSavedVoiceId,
    projectSpeaker,
    projectVoiceMode,
    savedVoices,
  ]);

  const selectedProjectSummary = useMemo(
    () =>
      projects.find((project) => project.id === selectedProjectId) ?? null,
    [projects, selectedProjectId],
  );

  const selectedProjectMeta = selectedProjectId
    ? projectMetaById[selectedProjectId] ?? null
    : null;

  const folderNameById = useMemo(
    () =>
      Object.fromEntries(projectFolders.map((folder) => [folder.id, folder.name] as const)),
    [projectFolders],
  );

  const visibleProjects = useMemo(() => {
    const search = projectSearch.trim().toLowerCase();
    const filtered = projects.filter((project) => {
      const meta = projectMetaById[project.id];
      const folderId = meta?.folder_id ?? null;
      const tags = meta?.tags ?? [];
      const completionReady =
        project.segment_count > 0 &&
        project.rendered_segment_count === project.segment_count;
      const statusMatch =
        projectStatusFilter === "all" ||
        (projectStatusFilter === "ready" && completionReady) ||
        (projectStatusFilter === "in_progress" && !completionReady);
      const folderMatch =
        projectFolderFilter === "all" || folderId === projectFolderFilter;
      const searchMatch =
        !search ||
        project.name.toLowerCase().includes(search) ||
        project.model_id?.toLowerCase().includes(search) ||
        tags.some((tag) => tag.toLowerCase().includes(search));
      return statusMatch && folderMatch && searchMatch;
    });

    const sorted = [...filtered];
    if (projectSort === "name") {
      sorted.sort((left, right) => left.name.localeCompare(right.name));
      return sorted;
    }
    if (projectSort === "progress") {
      sorted.sort((left, right) => {
        const leftRatio =
          left.segment_count > 0
            ? left.rendered_segment_count / left.segment_count
            : 0;
        const rightRatio =
          right.segment_count > 0
            ? right.rendered_segment_count / right.segment_count
            : 0;
        return rightRatio - leftRatio || right.updated_at - left.updated_at;
      });
      return sorted;
    }
    sorted.sort((left, right) => right.updated_at - left.updated_at);
    return sorted;
  }, [
    projectFolderFilter,
    projectMetaById,
    projectSearch,
    projectSort,
    projectStatusFilter,
    projects,
  ]);

  const projectDirty = useMemo(() => {
    if (!selectedProject) {
      return false;
    }
    return (
      projectName.trim() !== selectedProject.name ||
      projectModelId !== (selectedProject.model_id ?? "") ||
      projectVoiceMode !== selectedProject.voice_mode ||
      projectSpeaker !== (selectedProject.speaker ?? "") ||
      projectSavedVoiceId !== (selectedProject.saved_voice_id ?? "") ||
      (supportsSpeedControl
        ? Number(projectSpeed.toFixed(2)) !==
          Number((selectedProject.speed ?? 1).toFixed(2))
      : false)
    );
  }, [
    projectModelId,
    projectName,
    projectSavedVoiceId,
    projectSpeaker,
    projectSpeed,
    projectVoiceMode,
    selectedProject,
    supportsSpeedControl,
  ]);
  const projectFolderDirty =
    (selectedProjectMeta?.folder_id ?? "") !== projectFolderId;

  const loadProjects = useCallback(async () => {
    setProjectsLoading(true);
    try {
      const records = await api.listTtsProjects();
      setProjects(records);
      if (activeProjectId !== undefined) {
        if (
          activeProjectId &&
          !records.some((project) => project.id === activeProjectId)
        ) {
          onNavigateProject?.(null);
        }
        return;
      }
      setSelectedProjectId((current) => {
        if (current && records.some((project) => project.id === current)) {
          return current;
        }
        return records[0]?.id ?? null;
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to load TTS projects.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      setProjectsLoading(false);
    }
  }, [activeProjectId, onError, onNavigateProject, setSelectedProjectId]);

  const loadSavedVoices = useCallback(async () => {
    setSavedVoicesLoading(true);
    setSavedVoicesError(null);
    try {
      setSavedVoices(await api.listSavedVoices());
    } catch (err) {
      setSavedVoicesError(
        err instanceof Error ? err.message : "Failed to load saved voices.",
      );
    } finally {
      setSavedVoicesLoading(false);
    }
  }, []);

  const loadProjectFolders = useCallback(async () => {
    try {
      const folders = await api.listTtsProjectFolders();
      setProjectFolders(folders);
    } catch {
      setProjectFolders([]);
    }
  }, []);

  const loadProjectMeta = useCallback(async (records: TtsProjectSummary[]) => {
    if (records.length === 0) {
      setProjectMetaById({});
      return;
    }

    const entries = await Promise.all(
      records.map(async (project) => {
        try {
          const meta = await api.getTtsProjectMeta(project.id);
          return [project.id, meta] as const;
        } catch {
          return [
            project.id,
            {
              project_id: project.id,
              folder_id: null,
              tags: [],
              default_export_format: "wav",
              last_render_job_id: null,
              last_rendered_at: null,
            },
          ] as const;
        }
      }),
    );

    setProjectMetaById(Object.fromEntries(entries));
  }, []);

  const loadProject = useCallback(
    async (projectId: string) => {
      setProjectLoading(true);
      try {
        setSelectedProject(await api.getTtsProject(projectId));
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to load the TTS project.";
        setWorkspaceError(message);
        onError(message);
      } finally {
        setProjectLoading(false);
      }
    },
    [onError],
  );

  const loadProjectPronunciations = useCallback(async (projectId: string | null) => {
    if (!projectId) {
      setProjectPronunciations([]);
      return;
    }
    try {
      setProjectPronunciationsLoading(true);
      setProjectPronunciations(
        await api.listTtsProjectPronunciations(projectId),
      );
    } catch {
      setProjectPronunciations([]);
    } finally {
      setProjectPronunciationsLoading(false);
    }
  }, []);

  const loadProjectSnapshots = useCallback(async (projectId: string | null) => {
    if (!projectId) {
      setProjectSnapshots([]);
      return;
    }
    try {
      setProjectSnapshotsLoading(true);
      setProjectSnapshots(await api.listTtsProjectSnapshots(projectId));
    } catch {
      setProjectSnapshots([]);
    } finally {
      setProjectSnapshotsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadProjects();
    void loadSavedVoices();
    void loadProjectFolders();
  }, [loadProjectFolders, loadProjects, loadSavedVoices]);

  useEffect(() => {
    void loadProjectMeta(projects);
  }, [loadProjectMeta, projects]);

  useEffect(() => {
    if (!selectedProjectId) {
      setSelectedProject(null);
      return;
    }
    void loadProject(selectedProjectId);
  }, [loadProject, selectedProjectId]);

  useEffect(() => {
    void loadProjectPronunciations(selectedProjectId);
  }, [loadProjectPronunciations, selectedProjectId]);

  useEffect(() => {
    void loadProjectSnapshots(selectedProjectId);
  }, [loadProjectSnapshots, selectedProjectId]);

  useEffect(() => {
    if (!selectedProject) {
      setProjectName("");
      setProjectModelId(selectedModel ?? "");
      setProjectFolderId("");
      setProjectVoiceMode("built_in");
      setProjectSpeaker("Vivian");
      setProjectSavedVoiceId("");
      setProjectSpeed(1);
      setSegmentDrafts({});
      setSegmentSelections({});
      setSelectedSegmentIds([]);
      return;
    }

    setProjectName(selectedProject.name);
    setProjectModelId(selectedProject.model_id ?? selectedModel ?? "");
    setProjectFolderId(projectMetaById[selectedProject.id]?.folder_id ?? "");
    setProjectVoiceMode(selectedProject.voice_mode);
    setProjectSpeaker(selectedProject.speaker ?? "Vivian");
    setProjectSavedVoiceId(selectedProject.saved_voice_id ?? "");
    setProjectSpeed(selectedProject.speed ?? 1);
    setSegmentDrafts(
      Object.fromEntries(
        selectedProject.segments.map((segment) => [segment.id, segment.text]),
      ),
    );
    setSegmentSelections({});
    setSelectedSegmentIds([]);
  }, [projectMetaById, selectedModel, selectedProject]);

  useEffect(() => {
    if (!selectedProjectId) {
      return;
    }
    setProjectFolderId(projectMetaById[selectedProjectId]?.folder_id ?? "");
  }, [projectMetaById, selectedProjectId]);

  useEffect(() => {
    if (projectVoiceMode === "saved" && !supportsSavedVoices && supportsBuiltInVoices) {
      setProjectVoiceMode("built_in");
      return;
    }

    if (projectVoiceMode === "built_in" && !supportsBuiltInVoices && supportsSavedVoices) {
      setProjectVoiceMode("saved");
    }
  }, [
    projectVoiceMode,
    supportsBuiltInVoices,
    supportsSavedVoices,
  ]);

  useEffect(() => {
    if (selectedProject || !projectModelId) {
      return;
    }
    if (projectCompatibleVariantSet.has(projectModelId)) {
      return;
    }

    const fallback = resolvePreferredRouteModel({
      models: projectCompatibleModels,
      selectedModel,
      preferredVariants: TEXT_TO_SPEECH_PREFERRED_MODELS,
    });
    if (fallback && fallback !== projectModelId) {
      setProjectModelId(fallback);
    }
  }, [
    projectCompatibleModels,
    projectCompatibleVariantSet,
    projectModelId,
    selectedModel,
    selectedProject,
  ]);

  useEffect(() => {
    if (
      projectVoiceMode === "built_in" &&
      availableSpeakers.length > 0 &&
      !availableSpeakers.some((voice) => voice.id === projectSpeaker)
    ) {
      setProjectSpeaker(availableSpeakers[0]?.id ?? "Vivian");
    }
  }, [availableSpeakers, projectSpeaker, projectVoiceMode]);

  useEffect(() => {
    if (projectVoiceMode === "saved" && !projectSavedVoiceId && savedVoices.length > 0) {
      setProjectSavedVoiceId(savedVoices[0]?.id ?? "");
    }
  }, [projectSavedVoiceId, projectVoiceMode, savedVoices]);

  useEffect(() => {
    if (projectModelId && projectModelId !== selectedModel) {
      onSelectModel?.(projectModelId);
    }
  }, [onSelectModel, projectModelId, selectedModel]);

  useEffect(() => {
    if (
      typeof window === "undefined" ||
      typeof window.localStorage?.getItem !== "function"
    ) {
      setRenderQueueReady(true);
      return;
    }
    try {
      const raw = window.localStorage.getItem(STUDIO_RENDER_QUEUE_STORAGE_KEY);
      if (!raw) {
        setRenderQueue([]);
      } else {
        const parsed = JSON.parse(raw) as StudioRenderQueueItem[];
        setRenderQueue(
          parsed.map((item) => ({
            ...item,
            status: item.status === "running" ? "queued" : item.status,
          })),
        );
      }
    } catch {
      setRenderQueue([]);
    } finally {
      setRenderQueueReady(true);
    }
  }, []);

  useEffect(() => {
    if (
      !renderQueueReady ||
      typeof window === "undefined" ||
      typeof window.localStorage?.setItem !== "function"
    ) {
      return;
    }
    window.localStorage.setItem(
      STUDIO_RENDER_QUEUE_STORAGE_KEY,
      JSON.stringify(renderQueue),
    );
  }, [renderQueue, renderQueueReady]);

  const builtInVoiceItems: VoicePickerItem[] = availableSpeakers.map((voice) => ({
    id: voice.id,
    name: voice.name,
    categoryLabel: currentProjectModelInfo?.variant ?? "Built-in voice",
    description: voice.description,
    meta: [voice.language],
    selected: projectVoiceMode === "built_in" && projectSpeaker === voice.id,
    onSelect: () => {
      setProjectVoiceMode("built_in");
      setProjectSpeaker(voice.id);
      setWorkspaceStatus(null);
    },
  }));

  const projectVoiceNotice = useMemo(() => {
    if (!projectModelId) {
      return "Choose a render model before assigning a project voice.";
    }

    if (!supportsBuiltInVoices && !supportsSavedVoices) {
      return `${projectModelId} is not currently supported in Projects. Choose a model with built-in or saved-voice rendering.`;
    }

    if (projectVoiceMode === "saved") {
      if (!supportsSavedVoices) {
        return `${projectModelId} does not support reusable saved voices.`;
      }
      if (savedVoicesLoading) {
        return "Loading your saved voice library.";
      }
      return selectedVoiceName
        ? `Project voice set to "${selectedVoiceName}".`
        : "Choose a saved voice for this project.";
    }

    if (!supportsBuiltInVoices) {
      return `${projectModelId} does not expose built-in speakers on this route.`;
    }

    if (availableSpeakers.length === 0) {
      return `No built-in speakers are currently mapped for ${projectModelId}.`;
    }

    return selectedVoiceName
      ? `Project voice set to "${selectedVoiceName}".`
      : "Choose a built-in speaker for this project.";
  }, [
    availableSpeakers.length,
    projectModelId,
    projectVoiceMode,
    savedVoicesLoading,
    selectedVoiceName,
    supportsBuiltInVoices,
    supportsSavedVoices,
  ]);

  const savedVoiceItems: VoicePickerItem[] = savedVoices.map((voice) => ({
    id: voice.id,
    name: voice.name,
    categoryLabel: sourceLabel(voice.source_route_kind),
    description: voice.reference_text_preview,
    meta: [
      `${voice.reference_text_chars} chars`,
      formatRelativeDate(voice.updated_at || voice.created_at),
    ],
    previewUrl: api.savedVoiceAudioUrl(voice.id),
    selected: projectVoiceMode === "saved" && projectSavedVoiceId === voice.id,
    onSelect: () => {
      setProjectVoiceMode("saved");
      setProjectSavedVoiceId(voice.id);
      setWorkspaceStatus(null);
    },
  }));

  const selectedVoiceItem = useMemo(() => {
    if (projectVoiceMode === "saved") {
      return (
        savedVoiceItems.find((item) => item.id === projectSavedVoiceId) ?? null
      );
    }
    return builtInVoiceItems.find((item) => item.id === projectSpeaker) ?? null;
  }, [
    builtInVoiceItems,
    projectSavedVoiceId,
    projectSpeaker,
    projectVoiceMode,
    savedVoiceItems,
  ]);

  const normalizeImportedText = useCallback(
    (filename: string, text: string): string => {
      const lower = filename.toLowerCase();
      if (lower.endsWith(".json")) {
        try {
          const parsed = JSON.parse(text) as unknown;
          if (typeof parsed === "string") {
            return parsed;
          }
          return JSON.stringify(parsed, null, 2);
        } catch {
          return text;
        }
      }
      if (lower.endsWith(".html") || lower.endsWith(".htm")) {
        if (typeof DOMParser !== "undefined") {
          const parser = new DOMParser();
          const doc = parser.parseFromString(text, "text/html");
          return doc.body?.textContent?.trim() || text;
        }
      }
      if (lower.endsWith(".rtf")) {
        return text
          .replace(/\\par[d]?/g, "\n")
          .replace(/\\'[0-9a-fA-F]{2}/g, "")
          .replace(/[{}\\]/g, "")
          .trim();
      }
      return text;
    },
    [],
  );

  const handleImportFile = async (
    event: ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    try {
      const text = normalizeImportedText(file.name, await file.text());
      setNewProjectText(text);
      setNewProjectFilename(file.name);
      if (!newProjectName.trim()) {
        setNewProjectName(file.name.replace(/\.[^.]+$/, ""));
      }
      setWorkspaceStatus({
        tone: "success",
        message: `Imported ${file.name}. Review the text, then create the project.`,
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to read the selected file.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      event.target.value = "";
    }
  };

  const handleImportFromUrl = async () => {
    const sourceUrl = newProjectSourceUrl.trim();
    if (!sourceUrl) {
      const message = "Enter a URL to import a script.";
      setWorkspaceError(message);
      onError(message);
      return;
    }
    try {
      setImportingFromUrl(true);
      setWorkspaceError(null);
      const response = await fetch(sourceUrl);
      if (!response.ok) {
        throw new Error(`Failed to load URL (${response.status}).`);
      }
      const html = await response.text();
      const text = normalizeImportedText(`${sourceUrl}.html`, html);
      if (!text.trim()) {
        throw new Error("The URL did not return extractable text.");
      }
      setNewProjectText(text);
      if (!newProjectName.trim()) {
        try {
          const url = new URL(sourceUrl);
          setNewProjectName(url.hostname);
        } catch {
          setNewProjectName("Imported URL");
        }
      }
      setWorkspaceStatus({
        tone: "success",
        message: `Imported script content from ${sourceUrl}.`,
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to import script from URL.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      setImportingFromUrl(false);
    }
  };

  const resolveNewProjectDefaults = useCallback(() => {
    const currentVariant = selectedModelInfo?.variant ?? selectedModel ?? "";
    if (
      selectedModelInfo?.speech_capabilities?.supports_builtin_voices &&
      getSpeakerProfilesForVariant(currentVariant).length > 0
    ) {
      return {
        modelId: currentVariant,
        voiceMode: "built_in" as const,
        speaker: getSpeakerProfilesForVariant(currentVariant)[0]?.id ?? "Vivian",
        savedVoiceId: undefined,
      };
    }

    if (
      selectedModelInfo?.speech_capabilities?.supports_reference_voice &&
      savedVoices.length > 0
    ) {
      return {
        modelId: currentVariant,
        voiceMode: "saved" as const,
        speaker: undefined,
        savedVoiceId: savedVoices[0]?.id,
      };
    }

    const builtInModel = resolvePreferredRouteModel({
      models: builtInCompatibleModels,
      selectedModel,
      preferredVariants: TEXT_TO_SPEECH_PREFERRED_MODELS,
    });
    if (builtInModel) {
      return {
        modelId: builtInModel,
        voiceMode: "built_in" as const,
        speaker: getSpeakerProfilesForVariant(builtInModel)[0]?.id ?? "Vivian",
        savedVoiceId: undefined,
      };
    }

    const savedModel = resolvePreferredRouteModel({
      models: savedVoiceCompatibleModels,
      selectedModel,
      preferredVariants: SAVED_VOICE_RENDERER_PREFERRED_MODELS,
    });
    if (savedModel && savedVoices.length > 0) {
      return {
        modelId: savedModel,
        voiceMode: "saved" as const,
        speaker: undefined,
        savedVoiceId: savedVoices[0]?.id,
      };
    }

    return null;
  }, [
    builtInCompatibleModels,
    savedVoiceCompatibleModels,
    savedVoices,
    selectedModel,
    selectedModelInfo,
  ]);
  const newProjectDefaults = useMemo(
    () => resolveNewProjectDefaults(),
    [resolveNewProjectDefaults],
  );

  const openCreateProjectDialog = useCallback(() => {
    setWorkspaceError(null);
    setWorkspaceStatus(null);
    setIsCreateProjectDialogOpen(true);
  }, []);

  const handleCreateProject = async () => {
    if (!newProjectText.trim()) {
      const message = "Paste or import the script before creating a project.";
      setWorkspaceError(message);
      onError(message);
      return;
    }

    const defaults = resolveNewProjectDefaults();
    if (!defaults) {
      onModelRequired();
      const message =
        "Load a built-in voice model or saved-voice renderer before creating a TTS project.";
      setWorkspaceError(message);
      onError(message);
      return;
    }

    try {
      setCreatingProject(true);
      setWorkspaceError(null);
      setWorkspaceStatus(null);
      const defaultModelSupportsSpeed =
        availableModels.find((model) => model.variant === defaults.modelId)
          ?.speech_capabilities?.supports_speed_control ?? false;

      const project = await api.createTtsProject({
        name: newProjectName.trim() || undefined,
        source_filename: newProjectFilename || undefined,
        source_text: newProjectText,
        model_id: defaults.modelId,
        voice_mode: defaults.voiceMode,
        speaker: defaults.speaker,
        saved_voice_id: defaults.savedVoiceId,
        speed: defaultModelSupportsSpeed ? 1 : undefined,
      });

      setSelectedProjectId(project.id);
      setSelectedProject(project);
      setNewProjectName("");
      setNewProjectText("");
      setNewProjectFilename("");
      setNewProjectSourceUrl("");
      setIsCreateProjectDialogOpen(false);
      setWorkspaceStatus({
        tone: "success",
        message: `Created project "${project.name}" with ${project.segments.length} segments.`,
      });
      await loadProjects();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to create the TTS project.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      setCreatingProject(false);
    }
  };

  const persistSegmentDraft = useCallback(
    async (
      project: TtsProjectRecord,
      segmentId: string,
      options?: {
        requireChanges?: boolean;
      },
    ) => {
      const currentSegment = project.segments.find((segment) => segment.id === segmentId);
      if (!currentSegment) {
        return null;
      }

      const requireChanges = options?.requireChanges ?? false;
      const draft = segmentDrafts[segmentId] ?? currentSegment.text;
      const text = draft.trim();

      if (!text) {
        const message = "Segment text cannot be empty.";
        setWorkspaceError(message);
        onError(message);
        return null;
      }

      if (requireChanges && draft === currentSegment.text) {
        return project;
      }

      const updated = await api.updateTtsProjectSegment(project.id, segmentId, {
        text,
      });
      setSelectedProject(updated);
      return updated;
    },
    [onError, segmentDrafts],
  );

  const persistProjectSettings = useCallback(async () => {
    if (!selectedProject) {
      return null;
    }
    if (!projectModelId) {
      onModelRequired();
      return null;
    }
    if (projectVoiceMode === "built_in" && !projectSpeaker) {
      const message = "Choose a built-in voice before saving project settings.";
      setWorkspaceError(message);
      onError(message);
      return null;
    }
    if (projectVoiceMode === "saved" && !projectSavedVoiceId) {
      const message = "Choose a saved voice before saving project settings.";
      setWorkspaceError(message);
      onError(message);
      return null;
    }
    if (!projectDirty && !projectFolderDirty) {
      return selectedProject;
    }

    try {
      setSavingProject(true);
      let project = selectedProject;
      if (projectDirty) {
        project = await api.updateTtsProject(selectedProject.id, {
          name: projectName.trim(),
          model_id: projectModelId,
          voice_mode: projectVoiceMode,
          speaker: projectVoiceMode === "built_in" ? projectSpeaker : undefined,
          saved_voice_id:
            projectVoiceMode === "saved" ? projectSavedVoiceId : undefined,
          speed: supportsSpeedControl ? projectSpeed : undefined,
        });
      }
      if (projectFolderDirty) {
        const meta = await api.updateTtsProjectMeta(selectedProject.id, {
          folder_id: projectFolderId || undefined,
        });
        setProjectMetaById((current) => ({
          ...current,
          [selectedProject.id]: meta,
        }));
      }
      setSelectedProject(project);
      setWorkspaceStatus({
        tone: "success",
        message: "Project settings saved.",
      });
      await loadProjects();
      return project;
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to save project settings.";
      setWorkspaceError(message);
      onError(message);
      return null;
    } finally {
      setSavingProject(false);
    }
  }, [
    loadProjects,
    onError,
    onModelRequired,
    projectDirty,
    projectFolderDirty,
    projectFolderId,
    projectModelId,
    projectName,
    projectSavedVoiceId,
    projectSpeaker,
    projectSpeed,
    projectVoiceMode,
    selectedProject,
    supportsSpeedControl,
  ]);

  const handleSaveSegment = async (segmentId: string) => {
    if (!selectedProject) {
      return;
    }
    try {
      setSavingSegmentId(segmentId);
      const project = await persistSegmentDraft(selectedProject, segmentId, {
        requireChanges: true,
      });
      if (!project) {
        return;
      }
      setWorkspaceStatus({
        tone: "success",
        message: "Segment text saved. Re-render to refresh the audio.",
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to save the segment.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      setSavingSegmentId(null);
    }
  };

  const handleSplitSegment = async (segmentId: string) => {
    if (!selectedProject) {
      return;
    }

    const segment = selectedProject.segments.find((candidate) => candidate.id === segmentId);
    if (!segment) {
      return;
    }

    const draft = segmentDrafts[segmentId] ?? segment.text;
    const splitIndex = segmentSelections[segmentId];
    if (
      typeof splitIndex !== "number" ||
      splitIndex <= 0 ||
      splitIndex >= draft.length
    ) {
      const message = "Place the text cursor where the next segment should start.";
      setWorkspaceError(message);
      onError(message);
      return;
    }

    const beforeText = draft.slice(0, splitIndex).trim();
    const afterText = draft.slice(splitIndex).trim();
    if (!beforeText || !afterText) {
      const message =
        "Split the block so both resulting segments contain text.";
      setWorkspaceError(message);
      onError(message);
      return;
    }

    try {
      setWorkspaceError(null);
      const project = await api.splitTtsProjectSegment(selectedProject.id, segmentId, {
        before_text: beforeText,
        after_text: afterText,
      });
      setSelectedProject(project);
      setWorkspaceStatus({
        tone: "success",
        message: `Split segment ${segment.position + 1} into two blocks.`,
      });
      await loadProjects();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to split the segment.";
      setWorkspaceError(message);
      onError(message);
    }
  };

  const openDeleteSegmentConfirm = useCallback(
    (segmentId: string) => {
      if (deletingSegment) {
        return;
      }
      if (!selectedProject) {
        return;
      }
      if (selectedProject.segments.length <= 1) {
        const message = "A project must keep at least one segment.";
        setWorkspaceError(message);
        onError(message);
        return;
      }

      const segment = selectedProject.segments.find(
        (candidate) => candidate.id === segmentId,
      );
      if (!segment) {
        return;
      }

      setDeleteSegmentTarget({
        id: segment.id,
        position: segment.position + 1,
        preview: segment.text,
      });
      setDeleteSegmentError(null);
    },
    [deletingSegment, onError, selectedProject],
  );

  const closeDeleteSegmentConfirm = useCallback(() => {
    if (deletingSegment) {
      return;
    }
    setDeleteSegmentTarget(null);
    setDeleteSegmentError(null);
  }, [deletingSegment]);

  const confirmDeleteSegment = useCallback(async () => {
    if (!selectedProject || !deleteSegmentTarget || deletingSegment) {
      return;
    }

    const segment = selectedProject.segments.find(
      (candidate) => candidate.id === deleteSegmentTarget.id,
    );
    const deletedPosition = segment
      ? segment.position + 1
      : deleteSegmentTarget.position;

    try {
      setDeletingSegment(true);
      setDeleteSegmentError(null);
      setWorkspaceError(null);
      const project = await api.deleteTtsProjectSegment(
        selectedProject.id,
        deleteSegmentTarget.id,
      );
      setSelectedProject(project);
      setWorkspaceStatus({
        tone: "success",
        message: `Deleted segment ${deletedPosition}.`,
      });
      setDeleteSegmentTarget(null);
      await loadProjects();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to delete the segment.";
      setDeleteSegmentError(message);
      onError(message);
    } finally {
      setDeletingSegment(false);
    }
  }, [deleteSegmentTarget, deletingSegment, loadProjects, onError, selectedProject]);

  const handleDeleteSegment = async (segmentId: string) => {
    if (!selectedProject) {
      return;
    }
    if (selectedProject.segments.length <= 1) {
      const message = "A project must keep at least one segment.";
      setWorkspaceError(message);
      onError(message);
      return;
    }

    openDeleteSegmentConfirm(segmentId);
  };

  const queueSegmentsForRender = useCallback(
    async (project: TtsProjectRecord, segmentIds: string[]) => {
      const uniqueSegmentIds = Array.from(
        new Set(segmentIds.filter((id) => id.trim().length > 0)),
      );
      if (uniqueSegmentIds.length === 0) {
        return 0;
      }

      let jobId: string | undefined;
      try {
        const job = await api.createTtsProjectRenderJob(project.id, {
          queued_segment_ids: uniqueSegmentIds,
        });
        jobId = job.id;
      } catch {
        jobId = undefined;
      }

      setRenderQueue((current) => {
        const existingKeySet = new Set(
          current
            .filter((item) => item.status === "queued" || item.status === "running")
            .map((item) => `${item.projectId}:${item.segmentId}`),
        );
        const additions = uniqueSegmentIds
          .filter((segmentId) => !existingKeySet.has(`${project.id}:${segmentId}`))
          .map((segmentId) => {
            const segment = project.segments.find((entry) => entry.id === segmentId);
            return {
              id: `rq_${project.id}_${segmentId}_${Date.now()}`,
              projectId: project.id,
              segmentId,
              segmentLabel: segment
                ? `Segment ${segment.position + 1}`
                : "Segment",
              status: "queued" as const,
              jobId,
            };
          });
        return [...current, ...additions];
      });

      return uniqueSegmentIds.length;
    },
    [],
  );

  useEffect(() => {
    if (!renderQueueReady || processingRenderQueue) {
      return;
    }
    const nextItem = renderQueue.find((item) => item.status === "queued");
    if (!nextItem) {
      return;
    }

    setProcessingRenderQueue(true);
    setRenderingSegmentId(nextItem.segmentId);
    setRenderQueue((current) =>
      current.map((item) =>
        item.id === nextItem.id
          ? { ...item, status: "running", errorMessage: undefined }
          : item,
      ),
    );

    const process = async () => {
      try {
        const project = await api.renderTtsProjectSegment(
          nextItem.projectId,
          nextItem.segmentId,
        );
        if (selectedProjectId === nextItem.projectId) {
          setSelectedProject(project);
        }
        await loadProjects();

        setRenderQueue((current) =>
          current.filter((item) => item.id !== nextItem.id),
        );

        if (nextItem.jobId) {
          const hasPendingForJob = renderQueue.some(
            (item) =>
              item.jobId === nextItem.jobId &&
              item.id !== nextItem.id &&
              (item.status === "queued" || item.status === "running"),
          );
          if (!hasPendingForJob) {
            await api.updateTtsProjectRenderJob(nextItem.projectId, nextItem.jobId, {
              status: "completed",
            });
          }
        }
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to render queued segment.";
        setRenderQueue((current) =>
          current.map((item) =>
            item.id === nextItem.id
              ? { ...item, status: "failed", errorMessage: message }
              : item,
          ),
        );
        if (nextItem.jobId) {
          try {
            await api.updateTtsProjectRenderJob(nextItem.projectId, nextItem.jobId, {
              status: "failed",
              error_message: message,
            });
          } catch {
            // noop: queue failure state is already surfaced locally.
          }
        }
      } finally {
        setRenderingSegmentId(null);
        setProcessingRenderQueue(false);
      }
    };

    void process();
  }, [
    loadProjects,
    processingRenderQueue,
    renderQueue,
    renderQueueReady,
    selectedProjectId,
  ]);

  const retryRenderQueueItem = async (queueId: string) => {
    setRenderQueue((current) =>
      current.map((item) =>
        item.id === queueId
          ? { ...item, status: "queued", errorMessage: undefined }
          : item,
      ),
    );
  };

  const cancelRenderQueueItem = async (queueId: string) => {
    const item = renderQueue.find((entry) => entry.id === queueId);
    setRenderQueue((current) => current.filter((entry) => entry.id !== queueId));
    if (item?.jobId) {
      try {
        await api.updateTtsProjectRenderJob(item.projectId, item.jobId, {
          status: "cancelled",
        });
      } catch {
        // noop
      }
    }
  };

  const handleCreatePronunciation = async () => {
    if (!selectedProject) {
      return;
    }
    const source = newPronunciationSource.trim();
    const replacement = newPronunciationReplacement.trim();
    if (!source || !replacement) {
      const message = "Both source and replacement text are required.";
      setWorkspaceError(message);
      onError(message);
      return;
    }
    try {
      setSavingPronunciation(true);
      const created = await api.createTtsProjectPronunciation(selectedProject.id, {
        source_text: source,
        replacement_text: replacement,
      });
      setProjectPronunciations((current) => [created, ...current]);
      setNewPronunciationSource("");
      setNewPronunciationReplacement("");
      setWorkspaceStatus({
        tone: "success",
        message: `Added pronunciation rule for "${source}".`,
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to add pronunciation rule.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      setSavingPronunciation(false);
    }
  };

  const handleDeletePronunciation = async (entryId: string) => {
    if (!selectedProject) {
      return;
    }
    try {
      await api.deleteTtsProjectPronunciation(selectedProject.id, entryId);
      setProjectPronunciations((current) =>
        current.filter((entry) => entry.id !== entryId),
      );
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to delete pronunciation rule.";
      setWorkspaceError(message);
      onError(message);
    }
  };

  const handleCreateSnapshot = async () => {
    if (!selectedProject) {
      return;
    }
    try {
      setSavingSnapshot(true);
      const snapshot = await api.createTtsProjectSnapshot(selectedProject.id, {
        label: snapshotLabel.trim() || undefined,
      });
      setProjectSnapshots((current) => [snapshot, ...current]);
      setSnapshotLabel("");
      setWorkspaceStatus({
        tone: "success",
        message: `Saved snapshot${snapshot.label ? ` "${snapshot.label}"` : ""}.`,
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to save project snapshot.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      setSavingSnapshot(false);
    }
  };

  const handleRestoreSnapshot = async (snapshotId: string) => {
    if (!selectedProject) {
      return;
    }
    try {
      setRestoringSnapshotId(snapshotId);
      const restored = await api.restoreTtsProjectSnapshot(
        selectedProject.id,
        snapshotId,
      );
      setSelectedProject(restored);
      await loadProjects();
      setWorkspaceStatus({
        tone: "success",
        message: "Restored project snapshot.",
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to restore snapshot.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      setRestoringSnapshotId(null);
    }
  };

  const handleMergeSegmentWithNext = async (segmentId: string) => {
    if (!selectedProject) {
      return;
    }
    try {
      const project = await api.mergeTtsProjectSegmentWithNext(
        selectedProject.id,
        segmentId,
      );
      setSelectedProject(project);
      setWorkspaceStatus({
        tone: "success",
        message: "Merged segment with the next block.",
      });
      await loadProjects();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to merge project segments.";
      setWorkspaceError(message);
      onError(message);
    }
  };

  const handleMoveSegment = async (
    segmentId: string,
    direction: "up" | "down",
  ) => {
    if (!selectedProject) {
      return;
    }
    const ids = selectedProject.segments.map((segment) => segment.id);
    const index = ids.findIndex((id) => id === segmentId);
    if (index < 0) {
      return;
    }
    const targetIndex = direction === "up" ? index - 1 : index + 1;
    if (targetIndex < 0 || targetIndex >= ids.length) {
      return;
    }
    const reordered = [...ids];
    const [moved] = reordered.splice(index, 1);
    reordered.splice(targetIndex, 0, moved);
    try {
      const project = await api.reorderTtsProjectSegments(selectedProject.id, {
        ordered_segment_ids: reordered,
      });
      setSelectedProject(project);
      setWorkspaceStatus({
        tone: "success",
        message:
          direction === "up"
            ? "Moved segment up."
            : "Moved segment down.",
      });
      await loadProjects();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to reorder project segments.";
      setWorkspaceError(message);
      onError(message);
    }
  };

  const handleBulkDeleteSegments = async () => {
    if (!selectedProject || selectedSegmentIds.length === 0) {
      return;
    }
    if (selectedSegmentIds.length >= selectedProject.segments.length) {
      const message = "A project must keep at least one segment.";
      setWorkspaceError(message);
      onError(message);
      return;
    }
    try {
      const project = await api.bulkDeleteTtsProjectSegments(selectedProject.id, {
        segment_ids: selectedSegmentIds,
      });
      setSelectedProject(project);
      setSelectedSegmentIds([]);
      setWorkspaceStatus({
        tone: "success",
        message: `Deleted ${selectedSegmentIds.length} selected segment${selectedSegmentIds.length === 1 ? "" : "s"}.`,
      });
      await loadProjects();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to delete selected segments.";
      setWorkspaceError(message);
      onError(message);
    }
  };

  const handleRenderSelectedSegments = async () => {
    if (!selectedProject || selectedSegmentIds.length === 0) {
      return;
    }
    for (const segmentId of selectedSegmentIds) {
      await handleRenderSegment(segmentId);
    }
  };

  const handleRenderSegment = async (segmentId: string) => {
    if (!selectedProject) {
      return;
    }
    const project = await persistProjectSettings();
    if (!project) {
      return;
    }
    try {
      let currentProject = project;
      const currentSegment = currentProject.segments.find(
        (segment) => segment.id === segmentId,
      );
      const segmentDirty =
        currentSegment &&
        (segmentDrafts[segmentId] ?? currentSegment.text) !== currentSegment.text;

      if (segmentDirty) {
        const synced = await persistSegmentDraft(currentProject, segmentId);
        if (!synced) {
          return;
        }
        currentProject = synced;
      }
      const queuedCount = await queueSegmentsForRender(currentProject, [segmentId]);
      setWorkspaceStatus({
        tone: "success",
        message:
          queuedCount > 0
            ? segmentDirty
              ? "Saved latest edits and queued the segment for rendering."
              : "Segment queued for rendering."
            : "Segment is already in the render queue.",
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to render the segment.";
      setWorkspaceError(message);
      onError(message);
    }
  };

  const handleRenderAll = async () => {
    if (!selectedProject) {
      return;
    }
    const project = await persistProjectSettings();
    if (!project) {
      return;
    }
    try {
      setRenderingAll(true);
      let current = project;

      const segmentIdsToRender = new Set(
        current.segments
          .filter((segment) => {
            const draft = segmentDrafts[segment.id] ?? segment.text;
            return draft !== segment.text || !segment.speech_record_id;
          })
          .map((segment) => segment.id),
      );

      if (segmentIdsToRender.size === 0) {
        setWorkspaceStatus({
          tone: "success",
          message: "All segments already have current rendered audio.",
        });
        return;
      }

      for (const segment of current.segments) {
        const draft = segmentDrafts[segment.id] ?? segment.text;
        if (draft !== segment.text) {
          const synced = await persistSegmentDraft(current, segment.id);
          if (!synced) {
            return;
          }
          current = synced;
        }
      }

      const queuedCount = await queueSegmentsForRender(
        current,
        Array.from(segmentIdsToRender),
      );
      setWorkspaceStatus({
        tone: "success",
        message:
          queuedCount > 0
            ? `Queued ${queuedCount} segment${queuedCount === 1 ? "" : "s"} for background rendering.`
            : "All pending segments are already queued.",
      });
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed while rendering the project.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      setRenderingAll(false);
    }
  };

  const handleExport = async () => {
    if (!selectedProject || isDownloading) {
      return;
    }
    const segmentIds =
      exportScope === "selected" ? selectedSegmentIds : [];
    if (exportScope === "selected" && segmentIds.length === 0) {
      const message = "Select at least one segment to export a partial audio file.";
      setWorkspaceError(message);
      onError(message);
      return;
    }
    beginDownload();
    try {
      const extension =
        exportFormat === "wav"
          ? "wav"
          : exportFormat === "raw_i16"
            ? "pcm"
            : "f32";
      const baseSlug = selectedProject.name
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "");
      const filename = baseSlug ? `${baseSlug}.${extension}` : `tts-project.${extension}`;
      await api.downloadAudioFile(
        api.ttsProjectAudioUrl(selectedProject.id, {
          download: true,
          format: exportFormat,
          segment_ids: segmentIds,
        }),
        filename,
      );
      if (exportIncludeScript) {
        const scriptSource =
          exportScope === "selected"
            ? selectedProject.segments
                .filter((segment) => segmentIds.includes(segment.id))
                .map((segment) => segment.text)
                .join("\n\n")
            : selectedProject.source_text;
        downloadTextFile(
          `${baseSlug || "tts-project"}-script.txt`,
          scriptSource,
        );
      }
      completeDownload();
    } catch (err) {
      failDownload(err);
    }
  };

  const deleteProjectById = useCallback(
    async (projectId: string, projectName: string) => {
      if (deletingProject) {
        return;
      }
      try {
        setDeletingProject(true);
        setDeleteProjectError(null);
        await api.deleteTtsProject(projectId);
        setWorkspaceStatus({
          tone: "success",
          message: `Deleted project "${projectName}".`,
        });
        setSelectedProject((current) => (current?.id === projectId ? null : current));
        setSelectedProjectId((current) => (current === projectId ? null : current));
        setDeleteProjectTarget(null);
        await loadProjects();
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to delete the project.";
        setDeleteProjectError(message);
        onError(message);
      } finally {
        setDeletingProject(false);
      }
    },
    [deletingProject, loadProjects, onError],
  );

  const openDeleteProjectConfirm = useCallback(
    (projectId: string, projectName: string) => {
      if (deletingProject) {
        return;
      }
      setDeleteProjectTarget({ id: projectId, name: projectName });
      setDeleteProjectError(null);
    },
    [deletingProject],
  );

  const closeDeleteProjectConfirm = useCallback(() => {
    if (deletingProject) {
      return;
    }
    setDeleteProjectTarget(null);
    setDeleteProjectError(null);
  }, [deletingProject]);

  const confirmDeleteProject = useCallback(async () => {
    if (!deleteProjectTarget || deletingProject) {
      return;
    }
    await deleteProjectById(deleteProjectTarget.id, deleteProjectTarget.name);
  }, [deleteProjectTarget, deleteProjectById, deletingProject]);

  const handleDeleteProject = () => {
    if (!selectedProject) {
      return;
    }
    openDeleteProjectConfirm(selectedProject.id, selectedProject.name);
  };

  const selectedProjectRenderedCount =
    selectedProjectSummary?.rendered_segment_count ??
    selectedProject?.segments.filter((segment) => Boolean(segment.speech_record_id))
      .length ??
    0;
  const selectedProjectSegmentCount =
    selectedProjectSummary?.segment_count ?? selectedProject?.segments.length ?? 0;
  const selectedProjectTotalChars =
    selectedProjectSummary?.total_chars ??
    selectedProject?.segments.reduce((total, segment) => total + segment.input_chars, 0) ??
    0;
  const selectedProjectCompletionPercent =
    selectedProjectSegmentCount > 0
      ? Math.round(
          (selectedProjectRenderedCount / selectedProjectSegmentCount) * 100,
        )
      : 0;
  const editedSegmentCount =
    selectedProject?.segments.filter(
      (segment) => (segmentDrafts[segment.id] ?? segment.text) !== segment.text,
    ).length ?? 0;
  const pendingRenderSegmentCount =
    selectedProject?.segments.filter((segment) => {
      const draft = segmentDrafts[segment.id] ?? segment.text;
      return draft !== segment.text || !segment.speech_record_id;
    }).length ?? 0;
  const selectedSegmentCount = selectedSegmentIds.length;
  const readySegmentCount =
    selectedProjectSegmentCount > 0
      ? Math.max(0, selectedProjectSegmentCount - pendingRenderSegmentCount)
      : 0;
  const exportReady =
    selectedProjectSegmentCount > 0 &&
    pendingRenderSegmentCount === 0 &&
    selectedProjectRenderedCount === selectedProjectSegmentCount;
  const changedSinceLastRender =
    (selectedProject?.updated_at ?? 0) >
      (selectedProjectMeta?.last_rendered_at ?? 0) ||
    pendingRenderSegmentCount > 0;
  const primaryRenderLabel =
    pendingRenderSegmentCount > 0
      ? `Render ${pendingRenderSegmentCount} remaining`
      : "All blocks rendered";
  const activeProjectQueueItems = renderQueue.filter(
    (item) => selectedProject && item.projectId === selectedProject.id,
  );
  const queuedRenderCount = activeProjectQueueItems.filter(
    (item) => item.status === "queued" || item.status === "running",
  ).length;
  const failedRenderCount = activeProjectQueueItems.filter(
    (item) => item.status === "failed",
  ).length;
  const queuedSegmentIdSet = new Set(
    activeProjectQueueItems
      .filter((item) => item.status === "queued" || item.status === "running")
      .map((item) => item.segmentId),
  );
  const projectHeaderActions = (
    <Button
      size="sm"
      onClick={openCreateProjectDialog}
      className="h-9 gap-2 rounded-lg"
    >
      <FilePlus2 className="h-4 w-4" />
      New project
    </Button>
  );

  const projectLibraryFilters = (
    <Card className="rounded-2xl border-0 bg-[var(--bg-surface-0)] p-4 shadow-none sm:p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
            Project Library
          </div>
          <div className="mt-1 text-base font-semibold text-[var(--text-primary)]">
            {projects.length} project{projects.length === 1 ? "" : "s"}
          </div>
        </div>
        <div className="rounded-lg border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-2 text-[var(--text-muted)]">
          <Library className="h-4 w-4" />
        </div>
      </div>
      <div className="mt-4 space-y-2 rounded-xl bg-[var(--bg-surface-1)] p-3">
        <Input
          value={projectSearch}
          onChange={(event) => setProjectSearch(event.target.value)}
          placeholder="Search projects, models, or tags"
          className="bg-[var(--bg-surface-0)]"
        />
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          <Select
            value={projectStatusFilter}
            onValueChange={(value) =>
              setProjectStatusFilter(value as "all" | "in_progress" | "ready")
            }
          >
            <SelectTrigger className="h-9 bg-[var(--bg-surface-0)] px-2 text-xs">
              <SelectValue placeholder="All statuses" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All statuses</SelectItem>
              <SelectItem value="in_progress">In progress</SelectItem>
              <SelectItem value="ready">Ready to export</SelectItem>
            </SelectContent>
          </Select>
          <Select
            value={projectSort}
            onValueChange={(value) =>
              setProjectSort(value as "recent" | "name" | "progress")
            }
          >
            <SelectTrigger className="h-9 bg-[var(--bg-surface-0)] px-2 text-xs">
              <SelectValue placeholder="Sort: Recent" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="recent">Sort: Recent</SelectItem>
              <SelectItem value="name">Sort: Name</SelectItem>
              <SelectItem value="progress">Sort: Progress</SelectItem>
            </SelectContent>
          </Select>
          <Select value={projectFolderFilter} onValueChange={setProjectFolderFilter}>
            <SelectTrigger className="h-9 w-full bg-[var(--bg-surface-0)] px-2 text-xs">
              <SelectValue placeholder="All folders" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All folders</SelectItem>
              {projectFolders.map((folder) => (
                <SelectItem key={folder.id} value={folder.id}>
                  {folder.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
    </Card>
  );

  return (
    <>
      {headerActionContainer === undefined
        ? projectHeaderActions
        : headerActionContainer
          ? createPortal(projectHeaderActions, headerActionContainer)
          : null}

      <input
        ref={fileInputRef}
        type="file"
        accept=".txt,.md,.json,.csv,.rtf,.html,.htm,text/plain,text/markdown,text/csv,text/html,application/json"
        className="hidden"
        onChange={handleImportFile}
      />

      <Dialog
        open={isCreateProjectDialogOpen}
        onOpenChange={(open) => {
          if (!creatingProject) {
            setIsCreateProjectDialogOpen(open);
          }
        }}
      >
        <DialogContent className="max-w-4xl border-[var(--border-strong)] bg-[var(--bg-surface-0)] p-0">
          <DialogTitle className="sr-only">Create TTS project</DialogTitle>
          <div className="border-b border-[var(--border-muted)] px-5 py-4 sm:px-6">
            <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-muted)]">
              New Project
            </div>
            <h3 className="mt-2 text-lg font-semibold text-[var(--text-primary)]">
              Import and split a script
            </h3>
            <DialogDescription className="mt-1 max-w-2xl text-sm leading-relaxed text-[var(--text-muted)]">
              Start with a long-form script. Izwi splits it into editable
              narration segments, then keeps the shared voice, model, progress,
              and merged export together.
            </DialogDescription>
          </div>

          <div className="grid gap-5 p-5 sm:p-6 lg:grid-cols-[minmax(0,1fr)_280px]">
            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-wider text-[var(--text-primary)]">
                  Project name
                </label>
                <Input
                  value={newProjectName}
                  onChange={(event) => setNewProjectName(event.target.value)}
                  placeholder="Optional project name"
                />
              </div>

              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-wider text-[var(--text-primary)]">
                  Source script
                </label>
                <div className="flex flex-col gap-2 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3">
                  <label className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                    Import from URL
                  </label>
                  <div className="flex flex-wrap items-center gap-2">
                    <Input
                      value={newProjectSourceUrl}
                      onChange={(event) =>
                        setNewProjectSourceUrl(event.target.value)
                      }
                      placeholder="https://example.com/article"
                      className="flex-1 min-w-[200px] bg-[var(--bg-surface-0)]"
                    />
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => void handleImportFromUrl()}
                      disabled={importingFromUrl}
                      className="bg-[var(--bg-surface-0)]"
                    >
                      {importingFromUrl ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Upload className="h-4 w-4" />
                      )}
                      Import URL
                    </Button>
                  </div>
                </div>
                <Textarea
                  value={newProjectText}
                  onChange={(event) => setNewProjectText(event.target.value)}
                  rows={14}
                  placeholder="Paste the script you want to split into renderable segments..."
                  className="bg-[var(--bg-surface-1)] border-[var(--border-muted)]"
                />
              </div>

              {newProjectFilename ? (
                <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2 text-xs text-[var(--text-muted)]">
                  Imported file: {newProjectFilename}
                </div>
              ) : null}

              <div className="flex flex-wrap items-center justify-between gap-3 border-t border-[var(--border-muted)] pt-4">
                <Button
                  variant="outline"
                  onClick={() => fileInputRef.current?.click()}
                  className="bg-[var(--bg-surface-1)]"
                >
                  <Upload className="h-4 w-4" />
                  Import text file
                </Button>

                <div className="flex flex-wrap items-center gap-2">
                  <Button
                    variant="outline"
                    onClick={() => setIsCreateProjectDialogOpen(false)}
                    disabled={creatingProject}
                    className="bg-[var(--bg-surface-1)]"
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={handleCreateProject}
                    disabled={creatingProject}
                  >
                    {creatingProject ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Creating...
                      </>
                    ) : (
                      <>
                        <FilePlus2 className="h-4 w-4" />
                        Create project
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                  Workflow
                </div>
                <div className="mt-3 grid gap-2 text-xs">
                  <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-3 text-[var(--text-secondary)]">
                    1. Paste or import a full script
                  </div>
                  <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-3 text-[var(--text-secondary)]">
                    2. Izwi splits it into editable segments
                  </div>
                  <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-3 text-[var(--text-secondary)]">
                    3. Set one project voice and render/export
                  </div>
                </div>
              </div>

              <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
                <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                  Initial render profile
                </div>
                <div className="mt-3 text-sm text-[var(--text-secondary)]">
                  {newProjectDefaults ? (
                    <>
                      {newProjectDefaults.modelId}
                      {newProjectDefaults.voiceMode === "built_in" &&
                      newProjectDefaults.speaker
                        ? ` with built-in voice ${newProjectDefaults.speaker}`
                        : ""}
                      {newProjectDefaults.voiceMode === "saved" &&
                      newProjectDefaults.savedVoiceId
                        ? ` with your first saved voice`
                        : ""}
                    </>
                  ) : (
                    "Choose a compatible model before creating a TTS project."
                  )}
                </div>
                <p className="mt-2 text-xs leading-relaxed text-[var(--text-muted)]">
                  The project starts with the best available compatible render
                  profile, and you can refine model, voice, and speed after
                  creation.
                </p>
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog
        open={Boolean(deleteProjectTarget)}
        onOpenChange={(open) => {
          if (!open) {
            closeDeleteProjectConfirm();
          }
        }}
      >
        {deleteProjectTarget ? (
          <DialogContent className="max-w-md border-[var(--border-strong)] bg-[var(--bg-surface-1)] p-5">
            <DialogTitle className="sr-only">Delete project?</DialogTitle>
            <div className="flex items-start gap-3">
              <div className="mt-0.5 rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2 text-[var(--danger-text)]">
                <AlertTriangle className="h-4 w-4" />
              </div>
              <div className="min-w-0 flex-1">
                <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                  Delete project?
                </h3>
                <DialogDescription className="mt-1 text-sm text-[var(--text-muted)]">
                  This permanently removes project segments, pronunciation rules,
                  snapshots, and render queue history.
                </DialogDescription>
                <p className="mt-2 truncate text-xs text-[var(--text-subtle)]">
                  {deleteProjectTarget.name}
                </p>
              </div>
            </div>

            {deleteProjectError ? (
              <div className="mt-4 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]">
                {deleteProjectError}
              </div>
            ) : null}

            <div className="mt-5 flex items-center justify-end gap-2">
              <Button
                onClick={closeDeleteProjectConfirm}
                variant="outline"
                size="sm"
                className="h-8 border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)] hover:bg-[var(--bg-surface-3)]"
                disabled={deletingProject}
              >
                Cancel
              </Button>
              <Button
                onClick={() => void confirmDeleteProject()}
                variant="destructive"
                size="sm"
                className="h-8 gap-1.5"
                disabled={deletingProject}
              >
                {deletingProject ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Trash2 className="h-3.5 w-3.5" />
                )}
                Delete project
              </Button>
            </div>
          </DialogContent>
        ) : null}
      </Dialog>

      <Dialog
        open={Boolean(deleteSegmentTarget)}
        onOpenChange={(open) => {
          if (!open) {
            closeDeleteSegmentConfirm();
          }
        }}
      >
        {deleteSegmentTarget ? (
          <DialogContent className="max-w-md border-[var(--border-strong)] bg-[var(--bg-surface-1)] p-5">
            <DialogTitle className="sr-only">Delete segment?</DialogTitle>
            <div className="flex items-start gap-3">
              <div className="mt-0.5 rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2 text-[var(--danger-text)]">
                <AlertTriangle className="h-4 w-4" />
              </div>
              <div className="min-w-0 flex-1">
                <h3 className="text-sm font-semibold text-[var(--text-primary)]">
                  Delete segment {deleteSegmentTarget.position}?
                </h3>
                <DialogDescription className="mt-1 text-sm text-[var(--text-muted)]">
                  This permanently removes the segment from the project script.
                </DialogDescription>
                <p className="mt-2 line-clamp-3 text-xs text-[var(--text-subtle)]">
                  {deleteSegmentTarget.preview}
                </p>
              </div>
            </div>

            {deleteSegmentError ? (
              <div className="mt-4 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]">
                {deleteSegmentError}
              </div>
            ) : null}

            <div className="mt-5 flex items-center justify-end gap-2">
              <Button
                onClick={closeDeleteSegmentConfirm}
                variant="outline"
                size="sm"
                className="h-8 border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)] hover:bg-[var(--bg-surface-3)]"
                disabled={deletingSegment}
              >
                Cancel
              </Button>
              <Button
                onClick={() => void confirmDeleteSegment()}
                variant="destructive"
                size="sm"
                className="h-8 gap-1.5"
                disabled={deletingSegment}
              >
                {deletingSegment ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Trash2 className="h-3.5 w-3.5" />
                )}
                Delete segment
              </Button>
            </div>
          </DialogContent>
        ) : null}
      </Dialog>

      <div className="space-y-5">
        {workspaceError ? (
          <div className="flex items-start gap-2 rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-3 text-sm text-[var(--danger-text)]">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <p>{workspaceError}</p>
          </div>
        ) : null}

        {workspaceStatus ? (
          <div
            className={cn(
              "flex items-center gap-2 rounded-lg border px-3 py-2.5 text-sm",
              workspaceStatus.tone === "success"
                ? "bg-[var(--status-positive-bg)] border-[var(--status-positive-border)] text-[var(--status-positive-text)]"
                : "bg-[var(--danger-bg)] border-[var(--danger-border)] text-[var(--danger-text)]",
            )}
          >
            {workspaceStatus.tone === "success" ? (
              <CheckCircle2 className="h-4 w-4" />
            ) : (
              <AlertCircle className="h-4 w-4" />
            )}
            {workspaceStatus.message}
          </div>
        ) : null}

        {downloadState !== "idle" && downloadMessage ? (
          <div
            className={cn(
              "flex items-center gap-2 rounded-lg border px-3 py-2.5 text-sm",
              downloadState === "downloading" &&
                "bg-[var(--status-warning-bg)] border-[var(--status-warning-border)] text-[var(--status-warning-text)]",
              downloadState === "success" &&
                "bg-[var(--status-positive-bg)] border-[var(--status-positive-border)] text-[var(--status-positive-text)]",
              downloadState === "error" &&
                "bg-[var(--danger-bg)] border-[var(--danger-border)] text-[var(--danger-text)]",
            )}
          >
            {downloadState === "downloading" ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : downloadState === "success" ? (
              <CheckCircle2 className="h-4 w-4" />
            ) : (
              <AlertCircle className="h-4 w-4" />
            )}
            {downloadMessage}
          </div>
        ) : null}

        {!activeProjectId ? (
          <div className="space-y-5">
            {projectLibraryFilters}

            {projectsLoading ? (
              <div className="flex items-center gap-2 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2 text-xs text-[var(--text-muted)]">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Loading projects...
              </div>
            ) : projects.length === 0 ? (
              <div className="bg-[var(--bg-surface-0)] p-8 sm:p-10">
                <div className="flex min-h-[420px] flex-col items-center justify-center gap-6 text-center">
                  <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
                    <FileAudio className="h-6 w-6 text-[var(--text-muted)]" />
                  </div>
                  <div className="space-y-2">
                    <p className="text-xl font-semibold text-[var(--text-primary)]">
                      Create your first Studio project
                    </p>
                    <p className="max-w-xl text-sm leading-relaxed text-[var(--text-secondary)]">
                      Projects keep script segments, voice settings, render progress,
                      and export workflow in one place.
                    </p>
                  </div>
                  <Button onClick={openCreateProjectDialog}>
                    <FilePlus2 className="h-4 w-4" />
                    New project
                  </Button>
                </div>
              </div>
            ) : visibleProjects.length === 0 ? (
              <div className="rounded-xl border border-dashed border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-4 text-sm text-[var(--text-muted)]">
                No projects match the current search or filters.
              </div>
            ) : (
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                {visibleProjects.map((project) => {
                  const meta = projectMetaById[project.id];
                  const folderName = meta?.folder_id ? folderNameById[meta.folder_id] : null;
                  const progressPercent =
                    project.segment_count > 0
                      ? Math.round(
                          (project.rendered_segment_count / project.segment_count) * 100,
                        )
                      : 0;
                  const completionLabel = `${project.rendered_segment_count}/${project.segment_count} rendered`;
                  const projectStatus =
                    progressPercent >= 100
                      ? "Ready"
                      : project.rendered_segment_count > 0
                        ? "In progress"
                        : "Not rendered";
                  return (
                    <Card
                      key={project.id}
                      role="button"
                      tabIndex={0}
                      onClick={() => {
                        setSelectedProjectId(project.id);
                      }}
                      onKeyDown={(event) => {
                        if (event.currentTarget !== event.target) {
                          return;
                        }
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          setSelectedProjectId(project.id);
                        }
                      }}
                      className="group rounded-2xl border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-4 text-left shadow-none transition-colors hover:border-[var(--border-strong)] hover:bg-[var(--bg-surface-1)] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <h3 className="truncate text-base font-semibold text-[var(--text-primary)]">
                          {project.name}
                        </h3>
                        <button
                          type="button"
                          onPointerDown={(event) => {
                            event.stopPropagation();
                          }}
                          onClick={(event) => {
                            event.preventDefault();
                            event.stopPropagation();
                            openDeleteProjectConfirm(project.id, project.name);
                          }}
                          className="app-sidebar-delete-btn"
                          title="Delete project"
                          aria-label={`Delete project ${project.name}`}
                          disabled={deletingProject}
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </div>

                      <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[10px] font-medium uppercase tracking-[0.12em]">
                        <span
                          className={cn(
                            "rounded-full border px-2 py-0.5",
                            progressPercent >= 100
                              ? "border-[var(--status-positive-border)] bg-[var(--status-positive-bg)] text-[var(--status-positive-text)]"
                              : "border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] text-[var(--status-warning-text)]",
                          )}
                        >
                          {projectStatus}
                        </span>
                        {folderName ? (
                          <span className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2 py-0.5 text-[var(--text-muted)]">
                            {folderName}
                          </span>
                        ) : null}
                      </div>

                      <div className="mt-3 flex items-center gap-2">
                        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--bg-surface-3)]">
                          <div
                            className="h-full rounded-full bg-[var(--accent-solid)] transition-[width] duration-300"
                            style={{ width: `${progressPercent}%` }}
                          />
                        </div>
                        <span className="text-[11px] text-[var(--text-muted)]">
                          {progressPercent}%
                        </span>
                      </div>

                      <div className="mt-2 flex items-center justify-between text-xs text-[var(--text-muted)]">
                        <span>{completionLabel}</span>
                        <span>{project.total_chars} chars</span>
                      </div>

                      <p className="mt-2 line-clamp-2 text-xs text-[var(--text-secondary)]">
                        {project.model_id || "No model selected"}
                        {meta?.tags?.length ? ` · ${meta.tags.slice(0, 2).join(" · ")}` : ""}
                      </p>
                      <div className="mt-3 text-[11px] text-[var(--text-muted)]">
                        Updated {formatRelativeDate(project.updated_at)}
                      </div>
                    </Card>
                  );
                })}
              </div>
            )}
          </div>
        ) : !selectedProject ? (
          <Card className="rounded-2xl border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-6 shadow-none">
            <div className="flex flex-col items-start gap-3">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => onNavigateProject?.(null)}
                className="h-8 px-2 text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)]"
              >
                <ChevronLeft className="h-3.5 w-3.5" />
                Back to Studio
              </Button>
              <div className="text-sm text-[var(--text-muted)]">
                {projectLoading ? "Loading project..." : "Project not found."}
              </div>
            </div>
          </Card>
        ) : (
          <StudioWorkspaceScaffold
            overview={
              <>
              <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
                <div className="min-w-0 flex-1">
                  {onNavigateProject ? (
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => onNavigateProject(null)}
                      className="mb-3 h-8 px-2 text-xs text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                    >
                      <ChevronLeft className="h-3.5 w-3.5" />
                      Back to Studio
                    </Button>
                  ) : null}
                  <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                    Active Project
                  </div>
                  <h3 className="mt-2 text-2xl font-semibold tracking-tight text-[var(--text-primary)]">
                    {selectedProject.name}
                  </h3>
                  <p className="mt-2 max-w-3xl text-sm leading-relaxed text-[var(--text-secondary)]">
                    Keep the profile stable, review the split blocks, render only
                    the ones that still need updates, then export a merged narration
                    file when the project is ready.
                  </p>

                  <div className="mt-4 flex flex-wrap gap-2 text-xs">
                    <span className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-1 text-[var(--text-muted)]">
                      Updated {formatRelativeDate(selectedProject.updated_at)}
                    </span>
                    <span className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-1 text-[var(--text-muted)]">
                      {selectedProject.source_filename || "Manual paste"}
                    </span>
                    {projectModelId ? (
                      <span className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-1 text-[var(--text-muted)]">
                        {projectModelId}
                      </span>
                    ) : null}
                    {selectedVoiceItem?.name ? (
                      <span className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-1 text-[var(--text-muted)]">
                        {selectedVoiceItem.name}
                      </span>
                    ) : null}
                  </div>
                </div>

                <div className="grid gap-2 sm:grid-cols-2 xl:w-[360px] xl:grid-cols-1">
                  <Button
                    onClick={() => void handleRenderAll()}
                    disabled={renderingAll || pendingRenderSegmentCount === 0}
                    className="w-full justify-center"
                  >
                    {renderingAll ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Waves className="h-4 w-4" />
                    )}
                    {primaryRenderLabel}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={handleExport}
                    disabled={isDownloading}
                    className="w-full justify-center bg-[var(--bg-surface-1)]"
                  >
                    {isDownloading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Download className="h-4 w-4" />
                    )}
                    Export audio
                  </Button>
                </div>
              </div>

              <div className="mt-5 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-3">
                  <div className="text-[11px] uppercase tracking-wider text-[var(--text-muted)]">
                    Segments
                  </div>
                  <div className="mt-1 text-2xl font-semibold text-[var(--text-primary)]">
                    {selectedProjectSegmentCount}
                  </div>
                </div>
                <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-3">
                  <div className="text-[11px] uppercase tracking-wider text-[var(--text-muted)]">
                    Ready
                  </div>
                  <div className="mt-1 text-2xl font-semibold text-[var(--text-primary)]">
                    {readySegmentCount}
                  </div>
                </div>
                <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-3">
                  <div className="text-[11px] uppercase tracking-wider text-[var(--text-muted)]">
                    Needs Render
                  </div>
                  <div className="mt-1 text-2xl font-semibold text-[var(--text-primary)]">
                    {pendingRenderSegmentCount}
                  </div>
                </div>
                <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-3">
                  <div className="text-[11px] uppercase tracking-wider text-[var(--text-muted)]">
                    Script Size
                  </div>
                  <div className="mt-1 text-2xl font-semibold text-[var(--text-primary)]">
                    {selectedProjectTotalChars}
                  </div>
                  <div className="text-xs text-[var(--text-muted)]">chars</div>
                </div>
              </div>

              <div className="mt-5 rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div>
                    <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                      Workflow Status
                    </div>
                    <div className="mt-2 text-sm font-medium text-[var(--text-primary)]">
                      {exportReady && !changedSinceLastRender
                        ? "Merged export is current and ready."
                        : `${pendingRenderSegmentCount} segment${pendingRenderSegmentCount === 1 ? "" : "s"} still need rendering before the export is fully current.`}
                    </div>
                    <div className="mt-1 text-xs leading-relaxed text-[var(--text-muted)]">
                      Render actions use the latest saved project profile, and block
                      renders save edited text automatically before generating audio.
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2 text-[11px]">
                    {changedSinceLastRender ? (
                      <span className="rounded-full border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] px-2.5 py-1 text-[var(--status-warning-text)]">
                        Changed since last render
                      </span>
                    ) : (
                      <span className="rounded-full border border-[var(--status-positive-border)] bg-[var(--status-positive-bg)] px-2.5 py-1 text-[var(--status-positive-text)]">
                        Synced with last render
                      </span>
                    )}
                    <span className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-2.5 py-1 text-[var(--text-muted)]">
                      Prepare profile
                    </span>
                    <span className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-2.5 py-1 text-[var(--text-muted)]">
                      Review blocks
                    </span>
                    <span className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-2.5 py-1 text-[var(--text-muted)]">
                      Render remaining
                    </span>
                    <span className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-2.5 py-1 text-[var(--text-muted)]">
                      Export
                    </span>
                  </div>
                </div>

                <div className="mt-4 h-2 overflow-hidden rounded-full bg-[var(--bg-surface-2)]">
                  <div
                    className="h-full rounded-full bg-[var(--accent-solid)] transition-[width] duration-300"
                    style={{ width: `${selectedProjectCompletionPercent}%` }}
                  />
                </div>
                <div className="mt-2 text-xs text-[var(--text-muted)]">
                  {selectedProjectRenderedCount}/{selectedProjectSegmentCount} segments have attached audio in the project.
                </div>
              </div>
              </>
            }
            editor={
              <>
                <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                  <div>
                    <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                      Segments
                    </div>
                    <h3 className="mt-1 text-lg font-semibold text-[var(--text-primary)]">
                      Review and render script blocks
                    </h3>
                    <p className="mt-1 max-w-2xl text-sm text-[var(--text-secondary)]">
                      Edit the narration block, save a draft when needed, or render
                      immediately to refresh the attached audio.
                    </p>
                  </div>
                  {projectLoading ? (
                    <div className="inline-flex items-center gap-2 rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-1 text-xs text-[var(--text-muted)]">
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      Refreshing
                    </div>
                  ) : null}
                </div>

                <div className="mt-4 flex flex-wrap gap-2">
                  <span className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-1 text-[11px] text-[var(--text-muted)]">
                    {editedSegmentCount} edited
                  </span>
                  <span className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-1 text-[11px] text-[var(--text-muted)]">
                    {pendingRenderSegmentCount} awaiting render
                  </span>
                  <span className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-1 text-[11px] text-[var(--text-muted)]">
                    {readySegmentCount} ready
                  </span>
                  {selectedSegmentCount > 0 ? (
                    <span className="rounded-full border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] px-2.5 py-1 text-[11px] text-[var(--status-warning-text)]">
                      {selectedSegmentCount} selected
                    </span>
                  ) : null}
                </div>

                <div className="mt-4 flex flex-wrap items-center gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() =>
                      setSelectedSegmentIds(
                        selectedSegmentCount === selectedProject.segments.length
                          ? []
                          : selectedProject.segments.map((segment) => segment.id),
                      )
                    }
                    className="bg-[var(--bg-surface-1)]"
                  >
                    {selectedSegmentCount === selectedProject.segments.length
                      ? "Clear selection"
                      : "Select all"}
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => void handleRenderSelectedSegments()}
                    disabled={selectedSegmentCount === 0}
                    className="bg-[var(--bg-surface-1)]"
                  >
                    <Play className="h-4 w-4" />
                    Render selected
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => void handleBulkDeleteSegments()}
                    disabled={selectedSegmentCount === 0}
                    className="bg-[var(--bg-surface-1)]"
                  >
                    <Trash2 className="h-4 w-4" />
                    Delete selected
                  </Button>
                </div>

                <div className="mt-5 space-y-4">
                  {selectedProject.segments.map((segment) => {
                    const draft = segmentDrafts[segment.id] ?? segment.text;
                    const segmentDirty = draft !== segment.text;
                    const segmentNeedsRender = segmentDirty || !segment.speech_record_id;
                    const isSaving = savingSegmentId === segment.id;
                    const isRendering = renderingSegmentId === segment.id;
                    const segmentQueued = queuedSegmentIdSet.has(segment.id);
                    const splitIndex = segmentSelections[segment.id];
                    const canSplitSegment =
                      typeof splitIndex === "number" &&
                      splitIndex > 0 &&
                      splitIndex < draft.length;
                    const isSelected = selectedSegmentIds.includes(segment.id);
                    const isFirst = segment.position === 0;
                    const isLast =
                      segment.position === selectedProject.segments.length - 1;

                    return (
                      <div
                        key={segment.id}
                        className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4 sm:p-5"
                      >
                        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                          <div className="space-y-1.5">
                            <div className="flex flex-wrap items-center gap-2">
                              <label className="inline-flex items-center gap-1.5 text-xs text-[var(--text-muted)]">
                                <Switch
                                  checked={isSelected}
                                  onCheckedChange={(checked) =>
                                    setSelectedSegmentIds((current) => {
                                      if (checked) {
                                        return [...new Set([...current, segment.id])];
                                      }
                                      return current.filter((id) => id !== segment.id);
                                    })
                                  }
                                  aria-label={`Select segment ${segment.position + 1}`}
                                  className="h-5 w-9"
                                />
                                Select
                              </label>
                              <span className="text-xs font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                                Segment {segment.position + 1}
                              </span>
                              <span
                                className={cn(
                                  "rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.16em]",
                                  segmentDirty
                                    ? "border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] text-[var(--status-warning-text)]"
                                    : segment.speech_record_id
                                      ? "bg-[var(--status-positive-bg)] border-[var(--status-positive-border)] text-[var(--status-positive-text)]"
                                      : "bg-[var(--bg-surface-0)] border-[var(--border-muted)] text-[var(--text-muted)]",
                                )}
                              >
                                {segmentDirty
                                  ? "Edited"
                                  : segment.speech_record_id
                                    ? "Rendered"
                                    : "Needs render"}
                              </span>
                            </div>
                            <div className="text-sm text-[var(--text-secondary)]">
                              {segment.input_chars} chars
                              {segment.audio_duration_secs
                                ? ` · ${segment.audio_duration_secs.toFixed(1)}s audio`
                                : ""}
                            </div>
                          </div>

                          <div className="flex flex-wrap items-center gap-2">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => void handleSaveSegment(segment.id)}
                              disabled={!segmentDirty || isSaving}
                              className="bg-[var(--bg-surface-0)]"
                            >
                              {isSaving ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                <PencilLine className="h-4 w-4" />
                              )}
                              Save draft
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => void handleMoveSegment(segment.id, "up")}
                              disabled={isFirst}
                              className="bg-[var(--bg-surface-0)]"
                            >
                              <ChevronUp className="h-4 w-4" />
                              Move up
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => void handleMoveSegment(segment.id, "down")}
                              disabled={isLast}
                              className="bg-[var(--bg-surface-0)]"
                            >
                              <ChevronDown className="h-4 w-4" />
                              Move down
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => void handleMergeSegmentWithNext(segment.id)}
                              disabled={isLast}
                              className="bg-[var(--bg-surface-0)]"
                            >
                              <Link2 className="h-4 w-4" />
                              Merge next
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => void handleSplitSegment(segment.id)}
                              disabled={!canSplitSegment}
                              className="bg-[var(--bg-surface-0)]"
                            >
                              Split at cursor
                            </Button>
                            <Button
                              size="sm"
                              onClick={() => void handleRenderSegment(segment.id)}
                              disabled={isRendering || segmentQueued}
                            >
                              {isRendering ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                <Play className="h-4 w-4" />
                              )}
                              {segmentQueued
                                ? "Queued"
                                : segmentDirty
                                ? "Save & render"
                                : segmentNeedsRender
                                  ? "Render block"
                                  : "Re-render"}
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => void handleDeleteSegment(segment.id)}
                              disabled={selectedProject.segments.length <= 1}
                            >
                              <Trash2 className="h-4 w-4" />
                              Delete
                            </Button>
                          </div>
                        </div>

                        <Textarea
                          className="mt-4 bg-[var(--bg-surface-0)] border-[var(--border-muted)]"
                          value={draft}
                          onChange={(event) =>
                            setSegmentDrafts((current) => ({
                              ...current,
                              [segment.id]: event.target.value,
                            }))
                          }
                          onSelect={(event) =>
                            setSegmentSelections((current) => ({
                              ...current,
                              [segment.id]: event.currentTarget.selectionStart,
                            }))
                          }
                        />

                        <div className="mt-2 text-xs text-[var(--text-muted)]">
                          {canSplitSegment
                            ? "Split will create a new block where the cursor is placed."
                            : "Place the text cursor inside this block to enable splitting."}
                        </div>

                        {segmentDirty ? (
                          <div className="mt-4 rounded-xl border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] px-3 py-2 text-xs text-[var(--status-warning-text)]">
                            This block has local edits. Rendering will save the latest
                            text first and then refresh the audio.
                          </div>
                        ) : null}

                        {segment.speech_record_id ? (
                          <div className="mt-4 space-y-2 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-3">
                            <audio
                              src={api.textToSpeechRecordAudioUrl(segment.speech_record_id)}
                              controls
                              preload="none"
                              className="h-10 w-full"
                            />
                            <div className="text-xs text-[var(--text-muted)]">
                              {segmentDirty
                                ? "Preview reflects the last rendered audio until you render this edited block again."
                                : `Linked generation: ${segment.speech_record_id}`}
                            </div>
                          </div>
                        ) : (
                          <div className="mt-4 rounded-xl border border-dashed border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-2 text-xs text-[var(--text-muted)]">
                            Render this segment to attach audio and include it in the merged export.
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </>
            }
            actionRail={
              <>
                <Card className="rounded-2xl border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5 shadow-none">
                  <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                    Delivery
                  </div>
                  <div className="mt-2 text-sm font-semibold text-[var(--text-primary)]">
                    {exportReady && !changedSinceLastRender
                      ? "Everything is ready for merged export."
                      : pendingRenderSegmentCount > 0
                        ? `${pendingRenderSegmentCount} block${pendingRenderSegmentCount === 1 ? "" : "s"} still need rendering.`
                        : "Project changes are newer than the last complete render."}
                  </div>
                  <div className="mt-2 text-sm leading-relaxed text-[var(--text-secondary)]">
                    Use export once the project sounds right. If you changed text in a
                    block, re-render that block before downloading the merged file.
                  </div>

                  <div className="mt-4 grid gap-2">
                    <Select
                      value={exportFormat}
                      onValueChange={(value) =>
                        setExportFormat(
                          value as "wav" | "raw_i16" | "raw_f32",
                        )
                      }
                    >
                      <SelectTrigger className="h-9 bg-[var(--bg-surface-1)] px-2 text-xs">
                        <SelectValue placeholder="Export format: WAV" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="wav">Export format: WAV</SelectItem>
                        <SelectItem value="raw_i16">
                          Export format: PCM 16-bit
                        </SelectItem>
                        <SelectItem value="raw_f32">
                          Export format: Float 32-bit
                        </SelectItem>
                      </SelectContent>
                    </Select>
                    <Select
                      value={exportScope}
                      onValueChange={(value) =>
                        setExportScope(value as "all" | "selected")
                      }
                    >
                      <SelectTrigger className="h-9 bg-[var(--bg-surface-1)] px-2 text-xs">
                        <SelectValue placeholder="Scope: Full project" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">Scope: Full project</SelectItem>
                        <SelectItem value="selected">
                          Scope: Selected segments
                        </SelectItem>
                      </SelectContent>
                    </Select>
                    <label className="inline-flex items-center justify-between gap-2 text-xs text-[var(--text-secondary)]">
                      <span>Include script sidecar (.txt)</span>
                      <Switch
                        checked={exportIncludeScript}
                        onCheckedChange={setExportIncludeScript}
                        aria-label="Include script sidecar (.txt)"
                        className="h-5 w-9"
                      />
                    </label>
                  </div>

                  <div className="mt-5 grid gap-2">
                    {onOpenModelManager ? (
                      <Button
                        variant="outline"
                        size="sm"
                        className="w-full justify-center bg-[var(--bg-surface-1)]"
                        onClick={onOpenModelManager}
                      >
                        <Settings2 className="h-4 w-4" />
                        Open model manager
                      </Button>
                    ) : null}
                    <Button
                      variant="ghost"
                      onClick={() => void handleDeleteProject()}
                      disabled={deletingProject}
                      className="w-full justify-center"
                    >
                      {deletingProject ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Trash2 className="h-4 w-4" />
                      )}
                      Delete project
                    </Button>
                  </div>
                </Card>

                <Card className="rounded-2xl border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5 shadow-none">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                        Render Queue
                      </div>
                      <div className="mt-1 text-sm font-semibold text-[var(--text-primary)]">
                        {queuedRenderCount} queued · {failedRenderCount} failed
                      </div>
                    </div>
                  </div>

                  <div className="mt-3 space-y-2">
                    {activeProjectQueueItems.length === 0 ? (
                      <div className="rounded-xl border border-dashed border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2 text-xs text-[var(--text-muted)]">
                        No queued renders for this project.
                      </div>
                    ) : (
                      activeProjectQueueItems.map((item) => (
                        <div
                          key={item.id}
                          className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2"
                        >
                          <div className="flex items-center justify-between gap-2">
                            <div className="text-sm text-[var(--text-primary)]">
                              {item.segmentLabel}
                            </div>
                            <div className="text-xs uppercase tracking-wider text-[var(--text-muted)]">
                              {item.status}
                            </div>
                          </div>
                          {item.errorMessage ? (
                            <div className="mt-1 text-xs text-[var(--danger-text)]">
                              {item.errorMessage}
                            </div>
                          ) : null}
                          <div className="mt-2 flex items-center gap-2">
                            {item.status === "failed" ? (
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                className="h-7 bg-[var(--bg-surface-0)] px-2 text-xs"
                                onClick={() => void retryRenderQueueItem(item.id)}
                              >
                                Retry
                              </Button>
                            ) : null}
                            {item.status !== "running" ? (
                              <Button
                                type="button"
                                variant="ghost"
                                size="sm"
                                className="h-7 px-2 text-xs"
                                onClick={() => void cancelRenderQueueItem(item.id)}
                              >
                                Cancel
                              </Button>
                            ) : null}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </Card>
              </>
            }
            utilities={
              <StudioProjectUtilities
                profilePanel={
                  <Card className="rounded-2xl border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5 shadow-none">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                          Project Profile
                        </div>
                        <h3 className="mt-1 text-lg font-semibold text-[var(--text-primary)]">
                          Shared render settings
                        </h3>
                      </div>
                      {projectDirty || projectFolderDirty ? (
                        <div className="rounded-full border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--status-warning-text)]">
                          Unsaved
                        </div>
                      ) : (
                        <div className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
                          Synced
                        </div>
                      )}
                    </div>

                    <p className="mt-2 text-sm leading-relaxed text-[var(--text-secondary)]">
                      These defaults apply across the full project. Save them before
                      you start a long render pass, or let render actions sync them automatically.
                    </p>

                    <div className="mt-5 space-y-4">
                      <div className="space-y-2">
                        <label className="text-xs font-semibold uppercase tracking-wider text-[var(--text-primary)]">
                          Project name
                        </label>
                        <Input
                          value={projectName}
                          onChange={(event) => setProjectName(event.target.value)}
                        />
                      </div>

                      <div className="space-y-2">
                        <label className="text-xs font-semibold uppercase tracking-wider text-[var(--text-primary)]">
                          Render model
                        </label>
                        <RouteModelSelect
                          value={projectModelId}
                          options={projectModelOptions}
                          onSelect={(value) => {
                            setProjectModelId(value);
                            setWorkspaceStatus(null);
                          }}
                          className="w-full"
                        />
                      </div>

                      <div className="space-y-2">
                        <label className="text-xs font-semibold uppercase tracking-wider text-[var(--text-primary)]">
                          Folder
                        </label>
                        <Select
                          value={projectFolderId || STUDIO_UNFILED_FOLDER_VALUE}
                          onValueChange={(value) => {
                            setProjectFolderId(
                              value === STUDIO_UNFILED_FOLDER_VALUE ? "" : value,
                            );
                            setWorkspaceStatus(null);
                          }}
                        >
                          <SelectTrigger className="w-full bg-[var(--bg-surface-1)]">
                            <SelectValue placeholder="Unfiled" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value={STUDIO_UNFILED_FOLDER_VALUE}>
                              Unfiled
                            </SelectItem>
                            {projectFolders.map((folder) => (
                              <SelectItem key={folder.id} value={folder.id}>
                                {folder.name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <label className="text-xs font-semibold uppercase tracking-wider text-[var(--text-primary)]">
                          Project voice
                        </label>
                        <VoiceSelect
                          voiceMode={projectVoiceMode}
                          onVoiceModeChange={(value) => {
                            setProjectVoiceMode(value);
                            setWorkspaceStatus(null);
                          }}
                          savedVoiceItems={savedVoiceItems}
                          builtInVoiceItems={builtInVoiceItems}
                          selectedItem={selectedVoiceItem}
                          savedVoicesLoading={savedVoicesLoading}
                          savedVoicesError={savedVoicesError}
                          savedEnabled={supportsSavedVoices}
                          builtInEnabled={supportsBuiltInVoices}
                          disabled={!projectModelId}
                          modelLabel={currentProjectModelInfo?.variant ?? projectModelId}
                        />
                      </div>

                      <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-4 text-sm leading-relaxed text-[var(--text-secondary)]">
                        {projectVoiceNotice}
                      </div>

                      <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
                        <div className="flex items-center justify-between text-sm">
                          <span className="font-medium text-[var(--text-primary)]">
                            Speed
                          </span>
                          <span className="text-[var(--text-muted)]">
                            {supportsSpeedControl
                              ? `${projectSpeed.toFixed(2)}x`
                              : "Fixed by model"}
                          </span>
                        </div>
                        <Slider
                          value={[projectSpeed]}
                          min={0.5}
                          max={1.5}
                          step={0.05}
                          onValueChange={([value]) => setProjectSpeed(value ?? 1)}
                          disabled={!supportsSpeedControl}
                          className="mt-4"
                        />
                        <div className="mt-3 text-xs leading-relaxed text-[var(--text-muted)]">
                          {supportsSpeedControl
                            ? "This speed applies to every rendered segment in the project."
                            : "This model does not expose adjustable speed for project renders."}
                        </div>
                      </div>
                    </div>

                    <Button
                      variant="outline"
                      onClick={() => void persistProjectSettings()}
                      disabled={(!projectDirty && !projectFolderDirty) || savingProject}
                      className="mt-5 w-full justify-center bg-[var(--bg-surface-1)]"
                    >
                      {savingProject ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Settings2 className="h-4 w-4" />
                      )}
                      Save profile
                    </Button>
                  </Card>
                }
                pronunciationPanel={
                  <Card className="rounded-2xl border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5 shadow-none">
                    <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                      Pronunciation Rules
                    </div>
                    <div className="mt-2 text-sm leading-relaxed text-[var(--text-secondary)]">
                      Replace words or phrases before rendering to keep pronunciation
                      consistent across the project.
                    </div>

                    <div className="mt-4 grid gap-2">
                      <Input
                        value={newPronunciationSource}
                        onChange={(event) =>
                          setNewPronunciationSource(event.target.value)
                        }
                        placeholder="Source text (e.g. SQL)"
                      />
                      <Input
                        value={newPronunciationReplacement}
                        onChange={(event) =>
                          setNewPronunciationReplacement(event.target.value)
                        }
                        placeholder="Replacement text (e.g. sequel)"
                      />
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => void handleCreatePronunciation()}
                        disabled={savingPronunciation}
                        className="justify-center bg-[var(--bg-surface-1)]"
                      >
                        {savingPronunciation ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <PencilLine className="h-4 w-4" />
                        )}
                        Add rule
                      </Button>
                    </div>

                    <div className="mt-4 space-y-2">
                      {projectPronunciationsLoading ? (
                        <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2 text-xs text-[var(--text-muted)]">
                          Loading pronunciation rules...
                        </div>
                      ) : projectPronunciations.length === 0 ? (
                        <div className="rounded-xl border border-dashed border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2 text-xs text-[var(--text-muted)]">
                          No pronunciation rules yet.
                        </div>
                      ) : (
                        projectPronunciations.map((entry) => (
                          <div
                            key={entry.id}
                            className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2"
                          >
                            <div className="text-xs text-[var(--text-muted)]">
                              {entry.source_text}
                            </div>
                            <div className="text-sm text-[var(--text-primary)]">
                              {entry.replacement_text}
                            </div>
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              onClick={() => void handleDeletePronunciation(entry.id)}
                              className="mt-1 h-7 px-2 text-xs"
                            >
                              Remove
                            </Button>
                          </div>
                        ))
                      )}
                    </div>
                  </Card>
                }
                snapshotsPanel={
                  <Card className="rounded-2xl border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5 shadow-none">
                    <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                      Snapshots
                    </div>
                    <div className="mt-2 text-sm leading-relaxed text-[var(--text-secondary)]">
                      Save restore points before major edits and roll back when needed.
                    </div>

                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      <Input
                        value={snapshotLabel}
                        onChange={(event) => setSnapshotLabel(event.target.value)}
                        placeholder="Optional snapshot label"
                        className="flex-1 min-w-[180px] bg-[var(--bg-surface-1)]"
                      />
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => void handleCreateSnapshot()}
                        disabled={savingSnapshot}
                        className="bg-[var(--bg-surface-1)]"
                      >
                        {savingSnapshot ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <CheckCircle2 className="h-4 w-4" />
                        )}
                        Save snapshot
                      </Button>
                    </div>

                    <div className="mt-4 space-y-2">
                      {projectSnapshotsLoading ? (
                        <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2 text-xs text-[var(--text-muted)]">
                          Loading snapshots...
                        </div>
                      ) : projectSnapshots.length === 0 ? (
                        <div className="rounded-xl border border-dashed border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2 text-xs text-[var(--text-muted)]">
                          No snapshots yet.
                        </div>
                      ) : (
                        projectSnapshots.map((snapshot) => (
                          <div
                            key={snapshot.id}
                            className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2"
                          >
                            <div className="text-xs text-[var(--text-muted)]">
                              {snapshot.label || "Unnamed snapshot"} ·{" "}
                              {formatRelativeDate(snapshot.created_at)}
                            </div>
                            <Button
                              type="button"
                              variant="ghost"
                              size="sm"
                              onClick={() => void handleRestoreSnapshot(snapshot.id)}
                              disabled={restoringSnapshotId === snapshot.id}
                              className="mt-1 h-7 px-2 text-xs"
                            >
                              {restoringSnapshotId === snapshot.id ? (
                                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                              ) : (
                                <History className="h-3.5 w-3.5" />
                              )}
                              Restore
                            </Button>
                          </div>
                        ))
                      )}
                    </div>
                  </Card>
                }
              />
            }
          />
        )}
      </div>
    </>
  );
}
