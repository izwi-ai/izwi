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
  Waves,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  api,
  type ModelInfo,
  type SavedVoiceSummary,
  type TtsProjectFolderRecord,
  type TtsProjectMetaRecord,
  type TtsProjectRecord,
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
import { RouteHistoryDrawer } from "@/components/RouteHistoryDrawer";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { StudioWorkspaceScaffold } from "@/features/studio/components/StudioWorkspaceScaffold";
import { useDownloadIndicator } from "@/utils/useDownloadIndicator";
import { getSpeakerProfilesForVariant } from "@/types";

interface ModelOption {
  value: string;
  label: string;
  statusLabel: string;
  isReady: boolean;
}

interface StudioWorkspaceProps {
  selectedModel: string | null;
  selectedModelInfo: ModelInfo | null;
  availableModels: ModelInfo[];
  modelOptions?: ModelOption[];
  headerActionContainer?: HTMLElement | null;
  onSelectModel?: (variant: string) => void;
  onOpenModelManager?: () => void;
  onModelRequired: () => void;
  onError: (message: string) => void;
}

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

function projectAudioFilename(name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return slug ? `${slug}.wav` : "tts-project.wav";
}

export function StudioWorkspace({
  selectedModel,
  selectedModelInfo,
  availableModels,
  modelOptions = [],
  headerActionContainer,
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
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
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
  const [isCreateProjectDialogOpen, setIsCreateProjectDialogOpen] =
    useState(false);
  const [isProjectLibraryOpen, setIsProjectLibraryOpen] = useState(false);
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
  const [deletingProject, setDeletingProject] = useState(false);
  const [segmentDrafts, setSegmentDrafts] = useState<Record<string, string>>({});
  const [segmentSelections, setSegmentSelections] = useState<
    Record<string, number | null>
  >({});
  const fileInputRef = useRef<HTMLInputElement>(null);
  const {
    downloadState,
    downloadMessage,
    isDownloading,
    beginDownload,
    completeDownload,
    failDownload,
  } = useDownloadIndicator();

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
  }, [onError]);

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

  const handleImportFile = async (
    event: ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    try {
      const text = await file.text();
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
      setIsCreateProjectDialogOpen(false);
      setIsProjectLibraryOpen(false);
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

    const segment = selectedProject.segments.find((candidate) => candidate.id === segmentId);
    if (!segment) {
      return;
    }

    if (typeof window !== "undefined") {
      const confirmed = window.confirm(
        `Delete segment ${segment.position + 1} from "${selectedProject.name}"?`,
      );
      if (!confirmed) {
        return;
      }
    }

    try {
      setWorkspaceError(null);
      const project = await api.deleteTtsProjectSegment(selectedProject.id, segmentId);
      setSelectedProject(project);
      setWorkspaceStatus({
        tone: "success",
        message: `Deleted segment ${segment.position + 1}.`,
      });
      await loadProjects();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to delete the segment.";
      setWorkspaceError(message);
      onError(message);
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
      setRenderingSegmentId(segmentId);
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

      const updated = await api.renderTtsProjectSegment(currentProject.id, segmentId);
      setSelectedProject(updated);
      setWorkspaceStatus({
        tone: "success",
        message: segmentDirty
          ? "Saved the latest text and rendered the segment."
          : "Segment rendered and attached to the project.",
      });
      await loadProjects();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to render the segment.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      setRenderingSegmentId(null);
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

      for (const segment of current.segments) {
        if (!segmentIdsToRender.has(segment.id)) {
          continue;
        }

        current = await api.renderTtsProjectSegment(current.id, segment.id);
        setSelectedProject(current);
      }
      setWorkspaceStatus({
        tone: "success",
        message: `Rendered ${segmentIdsToRender.size} segment${segmentIdsToRender.size === 1 ? "" : "s"} that needed updates.`,
      });
      await loadProjects();
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
    beginDownload();
    try {
      await api.downloadAudioFile(
        api.ttsProjectAudioUrl(selectedProject.id, { download: true }),
        projectAudioFilename(selectedProject.name),
      );
      completeDownload();
    } catch (err) {
      failDownload(err);
    }
  };

  const handleDeleteProject = async () => {
    if (!selectedProject || deletingProject) {
      return;
    }
    if (typeof window !== "undefined") {
      const confirmed = window.confirm(
        `Delete the project "${selectedProject.name}"?`,
      );
      if (!confirmed) {
        return;
      }
    }
    try {
      setDeletingProject(true);
      await api.deleteTtsProject(selectedProject.id);
      setWorkspaceStatus({
        tone: "success",
        message: `Deleted project "${selectedProject.name}".`,
      });
      setSelectedProject(null);
      setSelectedProjectId(null);
      await loadProjects();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to delete the project.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      setDeletingProject(false);
    }
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
  const readySegmentCount =
    selectedProjectSegmentCount > 0
      ? Math.max(0, selectedProjectSegmentCount - pendingRenderSegmentCount)
      : 0;
  const exportReady =
    selectedProjectSegmentCount > 0 &&
    pendingRenderSegmentCount === 0 &&
    selectedProjectRenderedCount === selectedProjectSegmentCount;
  const primaryRenderLabel =
    pendingRenderSegmentCount > 0
      ? `Render ${pendingRenderSegmentCount} remaining`
      : "All blocks rendered";
  const projectLibraryActions = (
    <div className="flex items-center gap-2">
      <Button
        size="sm"
        onClick={openCreateProjectDialog}
        className="h-9 gap-2 rounded-lg"
      >
        <FilePlus2 className="h-4 w-4" />
        New project
      </Button>
      <RouteHistoryDrawer
        title="Studio"
        eyebrow="Project Library"
        headerIcon={Library}
        triggerLabel="Project Library"
        triggerCount={projects.length}
        countLabel={
          projectsLoading
            ? "Loading your reusable script projects."
            : projects.length === 0
              ? "No TTS projects yet."
              : `${visibleProjects.length} of ${projects.length} project${projects.length === 1 ? "" : "s"} shown.`
        }
        open={isProjectLibraryOpen}
        onOpenChange={setIsProjectLibraryOpen}
      >
        {({ close }) => (
          <div className="app-sidebar-list">
            {projectsLoading ? (
              <div className="app-sidebar-loading">
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Loading projects...
              </div>
            ) : projects.length === 0 ? (
              <div className="app-sidebar-empty">
                Create a project from the workspace to split a script into reusable
                renderable segments.
              </div>
            ) : (
              <div className="flex flex-col gap-2.5">
                <div className="space-y-2 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-3">
                  <Input
                    value={projectSearch}
                    onChange={(event) => setProjectSearch(event.target.value)}
                    placeholder="Search projects or tags"
                    className="bg-[var(--bg-surface-0)]"
                  />
                  <div className="grid gap-2 sm:grid-cols-2">
                    <select
                      value={projectStatusFilter}
                      onChange={(event) =>
                        setProjectStatusFilter(
                          event.target.value as "all" | "in_progress" | "ready",
                        )
                      }
                      className="h-9 rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-2 text-xs text-[var(--text-primary)]"
                    >
                      <option value="all">All statuses</option>
                      <option value="in_progress">In progress</option>
                      <option value="ready">Ready to export</option>
                    </select>
                    <select
                      value={projectSort}
                      onChange={(event) =>
                        setProjectSort(
                          event.target.value as "recent" | "name" | "progress",
                        )
                      }
                      className="h-9 rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-2 text-xs text-[var(--text-primary)]"
                    >
                      <option value="recent">Sort: Recent</option>
                      <option value="name">Sort: Name</option>
                      <option value="progress">Sort: Progress</option>
                    </select>
                  </div>
                  <select
                    value={projectFolderFilter}
                    onChange={(event) => setProjectFolderFilter(event.target.value)}
                    className="h-9 w-full rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-2 text-xs text-[var(--text-primary)]"
                  >
                    <option value="all">All folders</option>
                    {projectFolders.map((folder) => (
                      <option key={folder.id} value={folder.id}>
                        {folder.name}
                      </option>
                    ))}
                  </select>
                </div>
                {visibleProjects.length === 0 ? (
                  <div className="app-sidebar-empty">
                    No projects match the current search or filters.
                  </div>
                ) : null}
                {visibleProjects.map((project) => {
                  const isActive = project.id === selectedProjectId;
                  const completionLabel = `${project.rendered_segment_count}/${project.segment_count} segments rendered`;
                  const meta = projectMetaById[project.id];
                  const folderName = meta?.folder_id
                    ? folderNameById[meta.folder_id]
                    : null;
                  const previewParts = [
                    completionLabel,
                    `${project.total_chars} chars`,
                    project.model_id,
                    folderName ? `Folder: ${folderName}` : null,
                    ...(meta?.tags ?? []).slice(0, 2),
                  ].filter((value): value is string => Boolean(value));

                  return (
                    <button
                      key={project.id}
                      type="button"
                      onClick={() => {
                        setSelectedProjectId(project.id);
                        close();
                      }}
                      className={cn(
                        "group app-sidebar-row h-auto min-h-[110px] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                        isActive
                          ? "app-sidebar-row-active"
                          : "app-sidebar-row-idle",
                      )}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="app-sidebar-row-label truncate font-medium text-[var(--text-primary)]">
                          {project.name}
                        </span>
                        <span className="app-sidebar-row-meta shrink-0">
                          {formatRelativeDate(project.updated_at)}
                        </span>
                      </div>
                      <div className="mt-2 flex items-center gap-2">
                        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--bg-surface-3)]">
                          <div
                            className="h-full rounded-full bg-[var(--accent-solid)] transition-[width] duration-300"
                            style={{
                              width: `${
                                project.segment_count > 0
                                  ? Math.round(
                                      (project.rendered_segment_count /
                                        project.segment_count) *
                                        100,
                                    )
                                  : 0
                              }%`,
                            }}
                          />
                        </div>
                        <span className="app-sidebar-row-meta shrink-0">
                          {project.segment_count > 0
                            ? `${Math.round(
                                (project.rendered_segment_count /
                                  project.segment_count) *
                                  100,
                              )}%`
                            : "0%"}
                        </span>
                      </div>
                      <p
                        className="app-sidebar-row-preview"
                        style={{
                          display: "-webkit-box",
                          WebkitLineClamp: 3,
                          WebkitBoxOrient: "vertical",
                          overflow: "hidden",
                        }}
                      >
                        {previewParts.join(" · ")}
                      </p>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </RouteHistoryDrawer>
    </div>
  );

  return (
    <>
      {headerActionContainer === undefined
        ? projectLibraryActions
        : headerActionContainer
          ? createPortal(projectLibraryActions, headerActionContainer)
          : null}

      <input
        ref={fileInputRef}
        type="file"
        accept=".txt,.md,text/plain"
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

        {!selectedProject ? (
          <section className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-8 sm:p-10">
            <div className="flex min-h-[620px] flex-col items-center justify-center gap-6 text-center">
              <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4">
                <FileAudio className="h-6 w-6 text-[var(--text-muted)]" />
              </div>
              <div className="space-y-2">
                <p className="text-xl font-semibold text-[var(--text-primary)]">
                  Select or create a TTS project
                </p>
                <p className="max-w-xl text-sm leading-relaxed text-[var(--text-secondary)]">
                  Projects keep script segments, global render settings, per-segment
                  progress, and merged export in one place.
                </p>
              </div>
              <div className="flex flex-wrap items-center justify-center gap-3">
                <Button onClick={openCreateProjectDialog}>
                  <FilePlus2 className="h-4 w-4" />
                  New project
                </Button>
                {projects.length > 0 ? (
                  <Button
                    variant="outline"
                    onClick={() => setIsProjectLibraryOpen(true)}
                    className="bg-[var(--bg-surface-1)]"
                  >
                    <Library className="h-4 w-4" />
                    Open project library
                  </Button>
                ) : null}
              </div>
              <div className="grid w-full max-w-3xl gap-3 sm:grid-cols-3">
                <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-4 text-left">
                  <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                    Create
                  </div>
                  <div className="mt-2 text-sm text-[var(--text-secondary)]">
                    Import or paste a script and let Izwi split it into editable blocks.
                  </div>
                </div>
                <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-4 text-left">
                  <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                    Configure
                  </div>
                  <div className="mt-2 text-sm text-[var(--text-secondary)]">
                    Assign a project model, voice, and speed before rendering.
                  </div>
                </div>
                <div className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-4 py-4 text-left">
                  <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                    Export
                  </div>
                  <div className="mt-2 text-sm text-[var(--text-secondary)]">
                    Render individual segments or all at once, then export merged audio.
                  </div>
                </div>
              </div>
            </div>
          </section>
        ) : (
          <StudioWorkspaceScaffold
            library={
              <>
              <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
                <div className="min-w-0 flex-1">
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
                    Export merged WAV
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
                      {exportReady
                        ? "Merged export is current and ready."
                        : `${pendingRenderSegmentCount} segment${pendingRenderSegmentCount === 1 ? "" : "s"} still need rendering before the export is fully current.`}
                    </div>
                    <div className="mt-1 text-xs leading-relaxed text-[var(--text-muted)]">
                      Render actions use the latest saved project profile, and block
                      renders save edited text automatically before generating audio.
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2 text-[11px]">
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
                </div>

                <div className="mt-5 space-y-4">
                  {selectedProject.segments.map((segment) => {
                    const draft = segmentDrafts[segment.id] ?? segment.text;
                    const segmentDirty = draft !== segment.text;
                    const segmentNeedsRender = segmentDirty || !segment.speech_record_id;
                    const isSaving = savingSegmentId === segment.id;
                    const isRendering = renderingSegmentId === segment.id;
                    const splitIndex = segmentSelections[segment.id];
                    const canSplitSegment =
                      typeof splitIndex === "number" &&
                      splitIndex > 0 &&
                      splitIndex < draft.length;

                    return (
                      <div
                        key={segment.id}
                        className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-4 sm:p-5"
                      >
                        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                          <div className="space-y-1.5">
                            <div className="flex flex-wrap items-center gap-2">
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
                              onClick={() => void handleSplitSegment(segment.id)}
                              disabled={!canSplitSegment}
                              className="bg-[var(--bg-surface-0)]"
                            >
                              Split at cursor
                            </Button>
                            <Button
                              size="sm"
                              onClick={() => void handleRenderSegment(segment.id)}
                              disabled={isRendering}
                            >
                              {isRendering ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                <Play className="h-4 w-4" />
                              )}
                              {segmentDirty
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
            delivery={
              <>
                <section className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5">
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
                      <select
                        value={projectFolderId}
                        onChange={(event) => {
                          setProjectFolderId(event.target.value);
                          setWorkspaceStatus(null);
                        }}
                        className="h-10 w-full rounded-md border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 text-sm text-[var(--text-primary)]"
                      >
                        <option value="">Unfiled</option>
                        {projectFolders.map((folder) => (
                          <option key={folder.id} value={folder.id}>
                            {folder.name}
                          </option>
                        ))}
                      </select>
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
                </section>

                <section className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5">
                  <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                    Delivery
                  </div>
                  <div className="mt-2 text-sm font-semibold text-[var(--text-primary)]">
                    {exportReady
                      ? "Everything is ready for merged export."
                      : pendingRenderSegmentCount > 0
                        ? `${pendingRenderSegmentCount} block${pendingRenderSegmentCount === 1 ? "" : "s"} still need rendering.`
                        : "Render progress is in sync with the current project state."}
                  </div>
                  <div className="mt-2 text-sm leading-relaxed text-[var(--text-secondary)]">
                    Use export once the project sounds right. If you changed text in a
                    block, re-render that block before downloading the merged file.
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
                </section>
              </>
            }
          />
        )}
      </div>
    </>
  );
}
