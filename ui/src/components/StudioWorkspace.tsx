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
  ArrowLeft,
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  Download,
  FilePlus2,
  Loader2,
  PencilLine,
  Settings2,
  SlidersHorizontal,
  Trash2,
  Upload,
} from "lucide-react";
import { createUuid } from "@/lib/ids";
import { cn } from "@/lib/utils";
import {
  api,
  type ModelInfo,
  type SavedVoiceSummary,
  type StudioProjectMetaRecord,
  type StudioProjectPronunciationRecord,
  type StudioProjectRecord,
  type StudioProjectSummary,
  type StudioProjectVoiceMode,
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
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { StudioSegmentEditor } from "@/features/studio/components/StudioSegmentEditor";
import { StudioProjectHistoryTable } from "@/features/studio/components/StudioProjectHistoryTable";
import { StudioWorkspaceScaffold } from "@/features/studio/components/StudioWorkspaceScaffold";
import {
  clearStudioProjectDrafts,
  removeStudioSegmentDraft,
  removeStudioSegmentDrafts,
  restoreStudioSegmentDrafts,
  storeStudioSegmentDraft,
  type StudioSegmentDraftConflict,
} from "@/features/studio/segmentDrafts";
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
const STUDIO_SEGMENT_ADD_END_TARGET = "__end__";
const STUDIO_PROJECT_LIBRARY_PAGE_LIMIT = 24;

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
  const [projects, setProjects] = useState<StudioProjectSummary[]>([]);
  const [projectsNextCursor, setProjectsNextCursor] = useState<string | null>(
    null,
  );
  const [projectsHasMore, setProjectsHasMore] = useState(false);
  const [projectsLoading, setProjectsLoading] = useState(false);
  const [projectsLoadingMore, setProjectsLoadingMore] = useState(false);
  const [projectMetaById, setProjectMetaById] = useState<
    Record<string, StudioProjectMetaRecord>
  >({});
  const [selectedProjectIdState, setSelectedProjectIdState] = useState<
    string | null
  >(null);
  const [projectPronunciations, setProjectPronunciations] = useState<
    StudioProjectPronunciationRecord[]
  >([]);
  const [projectPronunciationsLoading, setProjectPronunciationsLoading] =
    useState(false);
  const [newPronunciationSource, setNewPronunciationSource] = useState("");
  const [newPronunciationReplacement, setNewPronunciationReplacement] =
    useState("");
  const [savingPronunciation, setSavingPronunciation] = useState(false);
  const [selectedProject, setSelectedProject] = useState<StudioProjectRecord | null>(
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
  const [isExportDialogOpen, setIsExportDialogOpen] = useState(false);
  const [creatingProject, setCreatingProject] = useState(false);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);
  const [workspaceStatus, setWorkspaceStatus] = useState<{
    tone: "success" | "error";
    message: string;
  } | null>(null);
  const [projectName, setProjectName] = useState("");
  const [projectModelId, setProjectModelId] = useState(selectedModel ?? "");
  const [projectVoiceMode, setProjectVoiceMode] =
    useState<StudioProjectVoiceMode>("built_in");
  const [projectSpeaker, setProjectSpeaker] = useState("Vivian");
  const [projectSavedVoiceId, setProjectSavedVoiceId] = useState("");
  const [projectSpeed, setProjectSpeed] = useState(1);
  const [savingProject, setSavingProject] = useState(false);
  const [savingSegmentId, setSavingSegmentId] = useState<string | null>(null);
  const [renderingSegmentId, setRenderingSegmentId] = useState<string | null>(
    null,
  );
  const [addingSegmentAfterSegmentId, setAddingSegmentAfterSegmentId] = useState<
    string | null
  >(null);
  const [focusSegmentId, setFocusSegmentId] = useState<string | null>(null);
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
  const [segmentDraftConflicts, setSegmentDraftConflicts] = useState<
    Record<string, StudioSegmentDraftConflict>
  >({});
  const [segmentSelections, setSegmentSelections] = useState<
    Record<string, number | null>
  >({});
  const [selectedSegmentIds, setSelectedSegmentIds] = useState<string[]>([]);
  const [segmentSettingsSegmentId, setSegmentSettingsSegmentId] = useState<
    string | null
  >(null);
  const [segmentSettingsModelId, setSegmentSettingsModelId] = useState("");
  const [segmentSettingsVoiceMode, setSegmentSettingsVoiceMode] =
    useState<StudioProjectVoiceMode>("built_in");
  const [segmentSettingsSpeaker, setSegmentSettingsSpeaker] = useState("");
  const [segmentSettingsSavedVoiceId, setSegmentSettingsSavedVoiceId] =
    useState("");
  const [savingSegmentSettings, setSavingSegmentSettings] = useState(false);
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

  const selectedProjectSummary = useMemo(
    () =>
      projects.find((project) => project.id === selectedProjectId) ?? null,
    [projects, selectedProjectId],
  );

  const visibleProjects = useMemo(
    () => [...projects].sort((left, right) => right.updated_at - left.updated_at),
    [projects],
  );

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
  const loadProjects = useCallback(async () => {
    setProjectsLoading(true);
    try {
      const page = await api.listStudioProjectPage({
        limit: STUDIO_PROJECT_LIBRARY_PAGE_LIMIT,
        cursor: null,
      });
      const records = page.items;
      setProjects(records);
      setProjectsNextCursor(page.pagination.next_cursor);
      setProjectsHasMore(page.pagination.has_more);
      if (activeProjectId !== undefined) {
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
        err instanceof Error ? err.message : "Failed to load Studio projects.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      setProjectsLoading(false);
    }
  }, [
    activeProjectId,
    onError,
    setSelectedProjectId,
  ]);

  const loadMoreProjects = useCallback(async () => {
    if (
      projectsLoading ||
      projectsLoadingMore ||
      !projectsHasMore ||
      !projectsNextCursor
    ) {
      return;
    }
    setProjectsLoadingMore(true);
    try {
      const page = await api.listStudioProjectPage({
        limit: STUDIO_PROJECT_LIBRARY_PAGE_LIMIT,
        cursor: projectsNextCursor,
      });
      setProjects((current) => {
        const seen = new Set(current.map((project) => project.id));
        const nextItems = page.items.filter((project) => !seen.has(project.id));
        return [...current, ...nextItems];
      });
      setProjectsNextCursor(page.pagination.next_cursor);
      setProjectsHasMore(page.pagination.has_more);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to load more Studio projects.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      setProjectsLoadingMore(false);
    }
  }, [
    onError,
    projectsHasMore,
    projectsLoading,
    projectsLoadingMore,
    projectsNextCursor,
  ]);

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

  const loadProjectMeta = useCallback(async (records: StudioProjectSummary[]) => {
    if (records.length === 0) {
      setProjectMetaById({});
      return;
    }

    const entries = await Promise.all(
      records.map(async (project) => {
        try {
          const meta = await api.getStudioProjectMeta(project.id);
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
        setSelectedProject(await api.getStudioProject(projectId));
      } catch (err) {
        const message =
          err instanceof Error ? err.message : "Failed to load the Studio project.";
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
        await api.listStudioProjectPronunciations(projectId),
      );
    } catch {
      setProjectPronunciations([]);
    } finally {
      setProjectPronunciationsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadProjects();
    void loadSavedVoices();
  }, [loadProjects, loadSavedVoices]);

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
    if (!selectedProject) {
      setProjectName("");
      setProjectModelId("");
      setProjectVoiceMode("built_in");
      setProjectSpeaker("Vivian");
      setProjectSavedVoiceId("");
      setProjectSpeed(1);
      setSegmentDrafts({});
      setSegmentDraftConflicts({});
      setSegmentSelections({});
      setSelectedSegmentIds([]);
      setAddingSegmentAfterSegmentId(null);
      setFocusSegmentId(null);
      setSegmentSettingsSegmentId(null);
      setSegmentSettingsModelId("");
      setSegmentSettingsVoiceMode("built_in");
      setSegmentSettingsSpeaker("");
      setSegmentSettingsSavedVoiceId("");
      return;
    }

    setProjectName(selectedProject.name);
    setProjectModelId(selectedProject.model_id ?? "");
    setProjectVoiceMode(selectedProject.voice_mode);
    setProjectSpeaker(selectedProject.speaker ?? "Vivian");
    setProjectSavedVoiceId(selectedProject.saved_voice_id ?? "");
    setProjectSpeed(selectedProject.speed ?? 1);
    const restoredDrafts = restoreStudioSegmentDrafts(
      selectedProject.id,
      selectedProject.segments,
    );
    setSegmentDrafts({
      ...Object.fromEntries(
        selectedProject.segments.map((segment) => [segment.id, segment.text]),
      ),
      ...restoredDrafts.drafts,
    });
    setSegmentDraftConflicts(restoredDrafts.conflicts);
    setSegmentSelections({});
    setSelectedSegmentIds([]);
  }, [selectedProject]);

  useEffect(() => {
    if (selectedProject) {
      return;
    }
    setProjectModelId(selectedModel ?? "");
  }, [selectedModel, selectedProject]);

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
    if (!workspaceStatus) {
      return;
    }
    const timeoutMs = workspaceStatus.tone === "success" ? 3500 : 6000;
    const timeoutId = window.setTimeout(() => {
      setWorkspaceStatus(null);
    }, timeoutMs);
    return () => window.clearTimeout(timeoutId);
  }, [workspaceStatus]);

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
  const segmentSettingsOpen = Boolean(segmentSettingsSegmentId);
  const segmentSettingsSegment = useMemo(() => {
    if (!selectedProject || !segmentSettingsSegmentId) {
      return null;
    }
    return (
      selectedProject.segments.find(
        (segment) => segment.id === segmentSettingsSegmentId,
      ) ?? null
    );
  }, [segmentSettingsSegmentId, selectedProject]);
  const segmentSettingsLabel = segmentSettingsSegment
    ? `Segment ${segmentSettingsSegment.position + 1}`
    : "Segment";

  const segmentSettingsModelInfo = useMemo(
    () =>
      availableModels.find((model) => model.variant === segmentSettingsModelId) ??
      null,
    [availableModels, segmentSettingsModelId],
  );
  const segmentSettingsCapabilities =
    segmentSettingsModelInfo?.speech_capabilities ?? null;
  const segmentSupportsBuiltInVoices =
    segmentSettingsCapabilities?.supports_builtin_voices ?? false;
  const segmentSupportsSavedVoices =
    segmentSettingsCapabilities?.supports_reference_voice ?? false;
  const segmentAvailableSpeakers = useMemo(
    () =>
      segmentSupportsBuiltInVoices
        ? getSpeakerProfilesForVariant(segmentSettingsModelId)
        : [],
    [segmentSettingsModelId, segmentSupportsBuiltInVoices],
  );

  const segmentBuiltInVoiceItems: VoicePickerItem[] = segmentAvailableSpeakers.map(
    (voice) => ({
      id: voice.id,
      name: voice.name,
      categoryLabel: segmentSettingsModelInfo?.variant ?? "Built-in voice",
      description: voice.description,
      meta: [voice.language],
      selected:
        segmentSettingsVoiceMode === "built_in" &&
        segmentSettingsSpeaker === voice.id,
      onSelect: () => {
        setSegmentSettingsVoiceMode("built_in");
        setSegmentSettingsSpeaker(voice.id);
        setWorkspaceStatus(null);
      },
    }),
  );
  const segmentSavedVoiceItems: VoicePickerItem[] = savedVoices.map((voice) => ({
    id: voice.id,
    name: voice.name,
    categoryLabel: sourceLabel(voice.source_route_kind),
    description: voice.reference_text_preview,
    meta: [
      `${voice.reference_text_chars} chars`,
      formatRelativeDate(voice.updated_at || voice.created_at),
    ],
    previewUrl: api.savedVoiceAudioUrl(voice.id),
    selected:
      segmentSettingsVoiceMode === "saved" &&
      segmentSettingsSavedVoiceId === voice.id,
    onSelect: () => {
      setSegmentSettingsVoiceMode("saved");
      setSegmentSettingsSavedVoiceId(voice.id);
      setWorkspaceStatus(null);
    },
  }));
  const segmentSelectedVoiceItem = useMemo(() => {
    if (segmentSettingsVoiceMode === "saved") {
      return (
        segmentSavedVoiceItems.find(
          (item) => item.id === segmentSettingsSavedVoiceId,
        ) ?? null
      );
    }
    return (
      segmentBuiltInVoiceItems.find((item) => item.id === segmentSettingsSpeaker) ??
      null
    );
  }, [
    segmentBuiltInVoiceItems,
    segmentSavedVoiceItems,
    segmentSettingsSavedVoiceId,
    segmentSettingsSpeaker,
    segmentSettingsVoiceMode,
  ]);
  const segmentSettingsDirty = useMemo(() => {
    if (!selectedProject || !segmentSettingsSegment) {
      return false;
    }

    const initialModelId =
      segmentSettingsSegment.model_id ?? selectedProject.model_id ?? "";
    const initialVoiceMode =
      segmentSettingsSegment.voice_mode ?? selectedProject.voice_mode;
    const initialSpeaker =
      segmentSettingsSegment.speaker ?? selectedProject.speaker ?? "";
    const initialSavedVoiceId =
      segmentSettingsSegment.saved_voice_id ?? selectedProject.saved_voice_id ?? "";

    return (
      segmentSettingsModelId !== initialModelId ||
      segmentSettingsVoiceMode !== initialVoiceMode ||
      (segmentSettingsVoiceMode === "built_in"
        ? segmentSettingsSpeaker !== initialSpeaker
        : segmentSettingsSavedVoiceId !== initialSavedVoiceId)
    );
  }, [
    segmentSettingsModelId,
    segmentSettingsSavedVoiceId,
    segmentSettingsSegment,
    segmentSettingsSpeaker,
    segmentSettingsVoiceMode,
    selectedProject,
  ]);
  const settingsFieldClass = "space-y-2.5";
  const settingsLabelClass =
    "text-xs font-semibold uppercase tracking-wider text-[var(--text-primary)]";
  const settingsControlHeightClass = "h-11";

  useEffect(() => {
    if (!segmentSettingsOpen) {
      return;
    }
    if (!selectedProject || !segmentSettingsSegmentId) {
      setSegmentSettingsSegmentId(null);
      return;
    }
    const stillExists = selectedProject.segments.some(
      (segment) => segment.id === segmentSettingsSegmentId,
    );
    if (!stillExists) {
      setSegmentSettingsSegmentId(null);
    }
  }, [segmentSettingsOpen, segmentSettingsSegmentId, selectedProject]);

  useEffect(() => {
    if (!segmentSettingsOpen) {
      return;
    }
    if (
      segmentSettingsVoiceMode === "saved" &&
      !segmentSupportsSavedVoices &&
      segmentSupportsBuiltInVoices
    ) {
      setSegmentSettingsVoiceMode("built_in");
      return;
    }
    if (
      segmentSettingsVoiceMode === "built_in" &&
      !segmentSupportsBuiltInVoices &&
      segmentSupportsSavedVoices
    ) {
      setSegmentSettingsVoiceMode("saved");
    }
  }, [
    segmentSettingsOpen,
    segmentSettingsVoiceMode,
    segmentSupportsBuiltInVoices,
    segmentSupportsSavedVoices,
  ]);

  useEffect(() => {
    if (!segmentSettingsOpen || segmentSettingsVoiceMode !== "built_in") {
      return;
    }
    if (
      segmentAvailableSpeakers.length > 0 &&
      !segmentAvailableSpeakers.some((voice) => voice.id === segmentSettingsSpeaker)
    ) {
      setSegmentSettingsSpeaker(segmentAvailableSpeakers[0]?.id ?? "");
    }
  }, [
    segmentAvailableSpeakers,
    segmentSettingsOpen,
    segmentSettingsSpeaker,
    segmentSettingsVoiceMode,
  ]);

  useEffect(() => {
    if (!segmentSettingsOpen || segmentSettingsVoiceMode !== "saved") {
      return;
    }
    if (!segmentSettingsSavedVoiceId && savedVoices.length > 0) {
      setSegmentSettingsSavedVoiceId(savedVoices[0]?.id ?? "");
    }
  }, [
    savedVoices,
    segmentSettingsOpen,
    segmentSettingsSavedVoiceId,
    segmentSettingsVoiceMode,
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
        "Load a built-in voice model or saved-voice renderer before creating a Studio project.";
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

      const project = await api.createStudioProject({
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
      setWorkspaceStatus({
        tone: "success",
        message: `Created project "${project.name}" with ${project.segments.length} segments.`,
      });
      await loadProjects();
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to create the Studio project.";
      setWorkspaceError(message);
      onError(message);
    } finally {
      setCreatingProject(false);
    }
  };

  const persistSegmentDraft = useCallback(
    async (
      project: StudioProjectRecord,
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

      const updated = await api.updateStudioProjectSegment(project.id, segmentId, {
        text,
      });
      removeStudioSegmentDraft(project.id, segmentId);
      setSegmentDraftConflicts((current) => {
        const next = { ...current };
        delete next[segmentId];
        return next;
      });
      setSelectedProject(updated);
      return updated;
    },
    [onError, segmentDrafts],
  );

  const closeSegmentSettingsDrawer = useCallback(() => {
    if (savingSegmentSettings) {
      return;
    }
    setSegmentSettingsSegmentId(null);
  }, [savingSegmentSettings]);

  const openSegmentSettingsDrawer = useCallback(
    (segmentId: string) => {
      if (!selectedProject) {
        return;
      }
      const segment = selectedProject.segments.find(
        (candidate) => candidate.id === segmentId,
      );
      if (!segment) {
        return;
      }

      const modelId = segment.model_id ?? selectedProject.model_id ?? "";
      if (!modelId) {
        onModelRequired();
        const message = "Select a model before configuring segment settings.";
        setWorkspaceError(message);
        onError(message);
        return;
      }

      const voiceMode = segment.voice_mode ?? selectedProject.voice_mode;
      const fallbackSpeaker = getSpeakerProfilesForVariant(modelId)[0]?.id ?? "Vivian";
      const speaker = segment.speaker ?? selectedProject.speaker ?? fallbackSpeaker;
      const savedVoiceId =
        segment.saved_voice_id ??
        selectedProject.saved_voice_id ??
        savedVoices[0]?.id ??
        "";

      setSegmentSettingsModelId(modelId);
      setSegmentSettingsVoiceMode(voiceMode);
      setSegmentSettingsSpeaker(speaker);
      setSegmentSettingsSavedVoiceId(savedVoiceId);
      setSegmentSettingsSegmentId(segmentId);
      setWorkspaceError(null);
      setWorkspaceStatus(null);
    },
    [onError, onModelRequired, savedVoices, selectedProject],
  );

  const persistSegmentSettings = useCallback(async () => {
    if (!selectedProject || !segmentSettingsSegmentId || !segmentSettingsSegment) {
      return null;
    }

    if (!segmentSettingsModelId) {
      const message = "Choose a model before saving segment settings.";
      setWorkspaceError(message);
      onError(message);
      return null;
    }
    if (segmentSettingsVoiceMode === "built_in" && !segmentSettingsSpeaker) {
      const message = "Choose a built-in voice for this segment.";
      setWorkspaceError(message);
      onError(message);
      return null;
    }
    if (segmentSettingsVoiceMode === "saved" && !segmentSettingsSavedVoiceId) {
      const message = "Choose a saved voice for this segment.";
      setWorkspaceError(message);
      onError(message);
      return null;
    }
    if (!segmentSettingsDirty) {
      setSegmentSettingsSegmentId(null);
      return selectedProject;
    }

    try {
      setSavingSegmentSettings(true);
      const project = await api.updateStudioProjectSegment(
        selectedProject.id,
        segmentSettingsSegmentId,
        {
          model_id: segmentSettingsModelId,
          voice_mode: segmentSettingsVoiceMode,
          speaker:
            segmentSettingsVoiceMode === "built_in"
              ? segmentSettingsSpeaker
              : undefined,
          saved_voice_id:
            segmentSettingsVoiceMode === "saved"
              ? segmentSettingsSavedVoiceId
              : undefined,
        },
      );
      setSelectedProject(project);
      setSegmentSettingsSegmentId(null);
      setWorkspaceStatus({
        tone: "success",
        message: `Saved settings for segment ${segmentSettingsSegment.position + 1}.`,
      });
      await loadProjects();
      return project;
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "Failed to save segment settings.";
      setWorkspaceError(message);
      onError(message);
      return null;
    } finally {
      setSavingSegmentSettings(false);
    }
  }, [
    loadProjects,
    onError,
    segmentSettingsDirty,
    segmentSettingsModelId,
    segmentSettingsSavedVoiceId,
    segmentSettingsSegment,
    segmentSettingsSegmentId,
    segmentSettingsSpeaker,
    segmentSettingsVoiceMode,
    selectedProject,
  ]);

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
    if (!projectDirty) {
      return selectedProject;
    }

    try {
      setSavingProject(true);
      let project = selectedProject;
      if (projectDirty) {
        project = await api.updateStudioProject(selectedProject.id, {
          name: projectName.trim(),
          model_id: projectModelId,
          voice_mode: projectVoiceMode,
          speaker: projectVoiceMode === "built_in" ? projectSpeaker : undefined,
          saved_voice_id:
            projectVoiceMode === "saved" ? projectSavedVoiceId : undefined,
          speed: supportsSpeedControl ? projectSpeed : undefined,
        });
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
    projectModelId,
    projectName,
    projectSavedVoiceId,
    projectSpeaker,
    projectSpeed,
    projectVoiceMode,
    selectedProject,
    supportsSpeedControl,
  ]);

  const runProjectMutation = async <T,>(
    mutation: () => Promise<T>,
    fallbackMessage: string,
  ): Promise<T | null> => {
    try {
      setWorkspaceError(null);
      return await mutation();
    } catch (err) {
      const message = err instanceof Error ? err.message : fallbackMessage;
      setWorkspaceError(message);
      onError(message);
      return null;
    }
  };

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

  const handleAddSegment = async (
    afterSegmentId: string | null,
    text: string,
  ): Promise<boolean> => {
    if (!selectedProject) {
      return false;
    }

    const targetKey = afterSegmentId ?? STUDIO_SEGMENT_ADD_END_TARGET;
    const existingSegmentIds = new Set(selectedProject.segments.map((segment) => segment.id));
    setAddingSegmentAfterSegmentId(targetKey);

    try {
      const project = await runProjectMutation(
        () =>
          api.createStudioProjectSegment(selectedProject.id, {
            text,
            after_segment_id: afterSegmentId ?? undefined,
          }),
        "Failed to add the segment.",
      );
      if (!project) {
        return false;
      }

      setSelectedProject(project);
      const insertedSegment = project.segments.find(
        (segment) => !existingSegmentIds.has(segment.id),
      );
      if (insertedSegment) {
        setFocusSegmentId(insertedSegment.id);
      }
      setWorkspaceStatus({
        tone: "success",
        message: insertedSegment
          ? `Added segment ${insertedSegment.position + 1}.`
          : "Added a new segment.",
      });
      await loadProjects();
      return true;
    } finally {
      setAddingSegmentAfterSegmentId(null);
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

    const project = await runProjectMutation(
      () =>
        api.splitStudioProjectSegment(selectedProject.id, segmentId, {
          before_text: beforeText,
          after_text: afterText,
        }),
      "Failed to split the segment.",
    );
    if (!project) {
      return;
    }
    removeStudioSegmentDraft(selectedProject.id, segmentId);
    setSelectedProject(project);
    setWorkspaceStatus({
      tone: "success",
      message: `Split segment ${segment.position + 1} into two blocks.`,
    });
    await loadProjects();
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
      const project = await api.deleteStudioProjectSegment(
        selectedProject.id,
        deleteSegmentTarget.id,
      );
      removeStudioSegmentDraft(selectedProject.id, deleteSegmentTarget.id);
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
    async (project: StudioProjectRecord, segmentIds: string[]) => {
      const uniqueSegmentIds = Array.from(
        new Set(segmentIds.filter((id) => id.trim().length > 0)),
      );
      if (uniqueSegmentIds.length === 0) {
        return 0;
      }

      let jobId: string | undefined;
      try {
        const job = await api.createStudioProjectRenderJob(project.id, {
          queued_segment_ids: uniqueSegmentIds,
        });
        jobId = job.id;
      } catch {
        jobId = undefined;
      }

      let addedCount = 0;
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
              id: createUuid(),
              projectId: project.id,
              segmentId,
              segmentLabel: segment
                ? `Segment ${segment.position + 1}`
                : "Segment",
              status: "queued" as const,
              jobId,
            };
          });
        addedCount = additions.length;
        return [...current, ...additions];
      });

      return addedCount;
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
        const project = await api.renderStudioProjectSegment(
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
            await api.updateStudioProjectRenderJob(nextItem.projectId, nextItem.jobId, {
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
            await api.updateStudioProjectRenderJob(nextItem.projectId, nextItem.jobId, {
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
        await api.updateStudioProjectRenderJob(item.projectId, item.jobId, {
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
      const created = await api.createStudioProjectPronunciation(selectedProject.id, {
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
      await api.deleteStudioProjectPronunciation(selectedProject.id, entryId);
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

  const handleMergeSegmentWithNext = async (segmentId: string) => {
    if (!selectedProject) {
      return;
    }
    const segmentIndex = selectedProject.segments.findIndex(
      (segment) => segment.id === segmentId,
    );
    if (segmentIndex < 0 || segmentIndex >= selectedProject.segments.length - 1) {
      return;
    }
    const mergedSegmentIds = [
      segmentId,
      selectedProject.segments[segmentIndex + 1]?.id,
    ].filter((id): id is string => Boolean(id));
    const project = await runProjectMutation(
      () => api.mergeStudioProjectSegmentWithNext(selectedProject.id, segmentId),
      "Failed to merge project segments.",
    );
    if (!project) {
      return;
    }
    removeStudioSegmentDrafts(selectedProject.id, mergedSegmentIds);
    setSelectedProject(project);
    setWorkspaceStatus({
      tone: "success",
      message: "Merged segment with the next block.",
    });
    await loadProjects();
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
    const project = await runProjectMutation(
      () =>
        api.reorderStudioProjectSegments(selectedProject.id, {
          ordered_segment_ids: reordered,
        }),
      "Failed to reorder project segments.",
    );
    if (!project) {
      return;
    }
    setSelectedProject(project);
    setWorkspaceStatus({
      tone: "success",
      message: direction === "up" ? "Moved segment up." : "Moved segment down.",
    });
    await loadProjects();
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
    const project = await runProjectMutation(
      () =>
        api.bulkDeleteStudioProjectSegments(selectedProject.id, {
          segment_ids: selectedSegmentIds,
        }),
      "Failed to delete selected segments.",
    );
    if (!project) {
      return;
    }
    removeStudioSegmentDrafts(selectedProject.id, selectedSegmentIds);
    setSelectedProject(project);
    setSelectedSegmentIds([]);
    setWorkspaceStatus({
      tone: "success",
      message: `Deleted ${selectedSegmentIds.length} selected segment${selectedSegmentIds.length === 1 ? "" : "s"}.`,
    });
    await loadProjects();
  };

  const queueRenderForSegments = async (
    segmentIds: string[],
  ): Promise<{ queuedCount: number; syncedDraftCount: number } | null> => {
    if (!selectedProject || segmentIds.length === 0) {
      return null;
    }

    const project = await persistProjectSettings();
    if (!project) {
      return null;
    }

    const dirtySegmentCount = segmentIds.filter((segmentId) => {
      const currentSegment = project.segments.find(
        (segment) => segment.id === segmentId,
      );
      if (!currentSegment) {
        return false;
      }
      return (
        (segmentDrafts[segmentId] ?? currentSegment.text) !== currentSegment.text
      );
    }).length;
    if (dirtySegmentCount > 0) {
      const message =
        dirtySegmentCount === 1
          ? "Save the edited segment before rendering."
          : `Save the ${dirtySegmentCount} edited segments before rendering.`;
      setWorkspaceError(message);
      onError(message);
      return null;
    }

    const queuedCount = await queueSegmentsForRender(project, segmentIds);
    return { queuedCount, syncedDraftCount: 0 };
  };

  const handleRenderSelectedSegments = async () => {
    if (!selectedProject || selectedSegmentIds.length === 0) {
      return;
    }
    const renderResult = await runProjectMutation(
      () => queueRenderForSegments(selectedSegmentIds),
      "Failed to queue selected segments for rendering.",
    );
    if (!renderResult) {
      return;
    }
    const { queuedCount, syncedDraftCount } = renderResult;
    const segmentLabel = `${queuedCount} selected segment${queuedCount === 1 ? "" : "s"}`;
    setWorkspaceStatus({
      tone: "success",
      message:
        queuedCount > 0
          ? syncedDraftCount > 0
            ? `Saved latest edits and queued ${segmentLabel}.`
            : `Queued ${segmentLabel} for rendering.`
          : "Selected segments are already in the render queue.",
    });
  };

  const handleRenderSegment = async (segmentId: string) => {
    if (!selectedProject) {
      return;
    }
    const renderResult = await runProjectMutation(
      () => queueRenderForSegments([segmentId]),
      "Failed to render the segment.",
    );
    if (!renderResult) {
      return;
    }
    const { queuedCount, syncedDraftCount } = renderResult;
    setWorkspaceStatus({
      tone: "success",
      message:
        queuedCount > 0
          ? syncedDraftCount > 0
            ? "Saved latest edits and queued the segment for rendering."
            : "Segment queued for rendering."
          : "Segment is already in the render queue.",
    });
  };

  const handleExport = async (): Promise<boolean> => {
    if (!selectedProject || isDownloading) {
      return false;
    }
    const segmentIds =
      exportScope === "selected" ? selectedSegmentIds : [];
    if (exportScope === "selected" && segmentIds.length === 0) {
      const message = "Select at least one segment to export a partial audio file.";
      setWorkspaceError(message);
      onError(message);
      return false;
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
      const filename = baseSlug ? `${baseSlug}.${extension}` : `studio-project.${extension}`;
      await api.downloadAudioFile(
        api.studioProjectAudioUrl(selectedProject.id, {
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
          `${baseSlug || "studio-project"}-script.txt`,
          scriptSource,
        );
      }
      completeDownload();
      return true;
    } catch (err) {
      failDownload(err);
      return false;
    }
  };

  const handleExportFromDialog = async () => {
    const didExport = await handleExport();
    if (didExport) {
      setIsExportDialogOpen(false);
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
        await api.deleteStudioProject(projectId);
        clearStudioProjectDrafts(projectId);
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
  const pendingRenderSegmentCount =
    selectedProject?.segments.filter((segment) => {
      const draft = segmentDrafts[segment.id] ?? segment.text;
      return draft !== segment.text || !segment.speech_record_id;
    }).length ?? 0;
  const selectedSegmentCount = selectedSegmentIds.length;
  const selectedSegmentIdSet = useMemo(
    () => new Set(selectedSegmentIds),
    [selectedSegmentIds],
  );
  const readySegmentCount =
    selectedProjectSegmentCount > 0
      ? Math.max(0, selectedProjectSegmentCount - pendingRenderSegmentCount)
      : 0;
  const activeProjectQueueItems = useMemo(
    () =>
      renderQueue.filter(
        (item) => selectedProject && item.projectId === selectedProject.id,
      ),
    [renderQueue, selectedProject],
  );
  const queuedRenderCount = activeProjectQueueItems.filter(
    (item) => item.status === "queued" || item.status === "running",
  ).length;
  const failedRenderCount = activeProjectQueueItems.filter(
    (item) => item.status === "failed",
  ).length;
  const queuedSegmentIdSet = useMemo(
    () =>
      new Set(
        activeProjectQueueItems
          .filter((item) => item.status === "queued" || item.status === "running")
          .map((item) => item.segmentId),
      ),
    [activeProjectQueueItems],
  );
  const projectHeaderActions = (
    <div className="flex items-center gap-2">
      <Button
        size="sm"
        onClick={openCreateProjectDialog}
        className="h-9 gap-2 rounded-lg"
      >
        <FilePlus2 className="h-4 w-4" />
        New project
      </Button>
      {activeProjectId ? (
        <Button
          size="sm"
          variant="outline"
          onClick={() => setIsExportDialogOpen(true)}
          className="h-9 gap-2 rounded-lg bg-[var(--bg-surface-1)]"
          disabled={!selectedProject}
        >
          <Download className="h-4 w-4" />
          Export
        </Button>
      ) : null}
    </div>
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
          <DialogTitle className="sr-only">Create Studio project</DialogTitle>
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
                  className="bg-[var(--bg-surface-1)] border-[var(--border-muted)]"
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
                    "Choose a compatible model before creating a Studio project."
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
        open={isExportDialogOpen}
        onOpenChange={(open) => {
          if (!isDownloading) {
            setIsExportDialogOpen(open);
          }
        }}
      >
        <DialogContent className="max-w-md border-[var(--border-strong)] bg-[var(--bg-surface-0)] p-5">
          <DialogTitle>Export audio</DialogTitle>
          <DialogDescription className="text-sm text-[var(--text-muted)]">
            {selectedProject
              ? `Choose export settings for "${selectedProject.name}".`
              : "Open a project to export audio."}
          </DialogDescription>

          <div className="mt-4 grid gap-3">
            <Select
              value={exportFormat}
              onValueChange={(value) =>
                setExportFormat(value as "wav" | "raw_i16" | "raw_f32")
              }
            >
              <SelectTrigger className="h-9 bg-[var(--bg-surface-1)] px-2 text-xs">
                <SelectValue placeholder="Export format: WAV" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="wav">Export format: WAV</SelectItem>
                <SelectItem value="raw_i16">Export format: PCM 16-bit</SelectItem>
                <SelectItem value="raw_f32">Export format: Float 32-bit</SelectItem>
              </SelectContent>
            </Select>

            <Select
              value={exportScope}
              onValueChange={(value) => setExportScope(value as "all" | "selected")}
            >
              <SelectTrigger className="h-9 bg-[var(--bg-surface-1)] px-2 text-xs">
                <SelectValue placeholder="Scope: Full project" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">Scope: Full project</SelectItem>
                <SelectItem value="selected">Scope: Selected segments</SelectItem>
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

            {exportScope === "selected" ? (
              <div className="rounded-md bg-[var(--bg-surface-1)] px-3 py-2 text-xs text-[var(--text-muted)]">
                {selectedSegmentIds.length > 0
                  ? `${selectedSegmentIds.length} selected segment${selectedSegmentIds.length === 1 ? "" : "s"} will be exported.`
                  : "Select one or more segments to export a partial file."}
              </div>
            ) : null}
          </div>

          <div className="mt-5 flex items-center justify-end gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setIsExportDialogOpen(false)}
              disabled={isDownloading}
            >
              Cancel
            </Button>
            <Button
              type="button"
              size="sm"
              onClick={() => void handleExportFromDialog()}
              disabled={
                !selectedProject ||
                isDownloading ||
                (exportScope === "selected" && selectedSegmentIds.length === 0)
              }
            >
              {isDownloading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Download className="h-4 w-4" />
              )}
              Export audio
            </Button>
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
                  and render queue history.
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

      <Sheet
        open={segmentSettingsOpen}
        onOpenChange={(open) => {
          if (!open) {
            closeSegmentSettingsDrawer();
          }
        }}
      >
        <SheetContent
          side="right"
          className="w-[min(92vw,30rem)] max-w-[30rem] gap-0 border-l border-[var(--border-muted)] bg-[var(--bg-surface-1)] p-0"
        >
          <SheetHeader className="gap-0 border-b border-[var(--border-muted)] px-5 py-5 pr-14 sm:px-6">
            <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--text-muted)]">
              Segment settings
            </div>
            <SheetTitle className="mt-2 text-base text-[var(--text-primary)]">
              {segmentSettingsLabel}
            </SheetTitle>
            <SheetDescription className="mt-1 text-sm text-[var(--text-muted)]">
              Set a custom model and voice for this segment. Leaving project
              defaults in place keeps behavior shared across all segments.
            </SheetDescription>
          </SheetHeader>

          <div className="flex min-h-0 flex-1 flex-col">
            <div className="space-y-4 px-5 py-4 sm:px-6 sm:py-5">
              <div className={settingsFieldClass}>
                <label className={settingsLabelClass}>Render model</label>
                <RouteModelSelect
                  value={segmentSettingsModelId}
                  options={projectModelOptions}
                  onSelect={(value) => {
                    setSegmentSettingsModelId(value);
                    setWorkspaceStatus(null);
                  }}
                  className="w-full"
                  triggerClassName={settingsControlHeightClass}
                />
              </div>

              <div className={settingsFieldClass}>
                <label className={settingsLabelClass}>Segment voice</label>
                <VoiceSelect
                  voiceMode={segmentSettingsVoiceMode}
                  onVoiceModeChange={(value) => {
                    setSegmentSettingsVoiceMode(value);
                    setWorkspaceStatus(null);
                  }}
                  savedVoiceItems={segmentSavedVoiceItems}
                  builtInVoiceItems={segmentBuiltInVoiceItems}
                  selectedItem={segmentSelectedVoiceItem}
                  savedVoicesLoading={savedVoicesLoading}
                  savedVoicesError={savedVoicesError}
                  savedEnabled={segmentSupportsSavedVoices}
                  builtInEnabled={segmentSupportsBuiltInVoices}
                  disabled={!segmentSettingsModelId}
                  modelLabel={
                    segmentSettingsModelInfo?.variant ?? segmentSettingsModelId
                  }
                  hideModelInTriggerSubtitle
                />
              </div>

              <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-2.5 text-xs leading-relaxed text-[var(--text-muted)]">
                Saving segment settings clears this segment audio link so the next
                render uses the updated model and voice.
              </div>
            </div>

            <SheetFooter className="mt-auto border-t border-[var(--border-muted)] px-5 py-4 sm:px-6 sm:space-x-0">
              <div className="flex w-full items-center justify-end gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={closeSegmentSettingsDrawer}
                  disabled={savingSegmentSettings}
                >
                  Cancel
                </Button>
                <Button
                  type="button"
                  size="sm"
                  onClick={() => void persistSegmentSettings()}
                  disabled={savingSegmentSettings || !segmentSettingsDirty}
                >
                  {savingSegmentSettings ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Settings2 className="h-4 w-4" />
                  )}
                  Save settings
                </Button>
              </div>
            </SheetFooter>
          </div>
        </SheetContent>
      </Sheet>

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
            <StudioProjectHistoryTable
              projects={visibleProjects}
              projectMetaById={projectMetaById}
              loading={projectsLoading}
              error={projects.length === 0 ? workspaceError : null}
              deletePending={deletingProject}
              loadMore={{
                canLoadMore: projectsHasMore && Boolean(projectsNextCursor),
                loading: projectsLoadingMore,
                onLoadMore: () => {
                  void loadMoreProjects();
                },
              }}
              onRefresh={() => {
                void loadProjects();
              }}
              onCreateProject={openCreateProjectDialog}
              onOpenProject={(projectId) => {
                setSelectedProjectId(projectId);
              }}
              onRequestDeleteProject={(project) => {
                openDeleteProjectConfirm(project.id, project.name);
              }}
            />
          </div>
        ) : !selectedProject ? (
          <Card className="rounded-2xl border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-6 shadow-none">
            <div className="flex flex-col items-start gap-3">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => onNavigateProject?.(null)}
                className="h-10 gap-2 rounded-full border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 text-sm font-medium text-[var(--text-secondary)] shadow-sm hover:bg-[var(--bg-surface-1)]"
              >
                <ArrowLeft className="h-4 w-4" />
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
              <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
                <div className="min-w-0">
                  {onNavigateProject ? (
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => onNavigateProject(null)}
                      className="mb-4 h-10 gap-2 rounded-full border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 text-sm font-medium text-[var(--text-secondary)] shadow-sm hover:bg-[var(--bg-surface-1)]"
                    >
                      <ArrowLeft className="h-4 w-4" />
                      Back to Studio
                    </Button>
                  ) : null}
                  <h3 className="truncate text-2xl font-semibold tracking-tight text-[var(--text-primary)]">
                    {selectedProject.name}
                  </h3>
                </div>
                <div className="text-xs text-[var(--text-muted)]">
                  Updated {formatRelativeDate(selectedProject.updated_at)}
                </div>
              </div>
            }
            statsRail={
              <div className="space-y-6">
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                    Status
                  </div>
                  <div className="mt-2 text-sm font-semibold text-[var(--text-primary)]">
                    {pendingRenderSegmentCount > 0
                      ? `${pendingRenderSegmentCount} block${pendingRenderSegmentCount === 1 ? "" : "s"} need rendering.`
                      : "All blocks are rendered and export-ready."}
                  </div>
                  <div className="mt-2 space-y-1 text-xs text-[var(--text-muted)]">
                    <div>{selectedProjectRenderedCount} rendered blocks</div>
                    <div>{queuedRenderCount} queued operations</div>
                    {failedRenderCount > 0 ? (
                      <div className="text-[var(--danger-text)]">
                        {failedRenderCount} queue failure{failedRenderCount === 1 ? "" : "s"}
                      </div>
                    ) : null}
                  </div>
                </div>

                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                    Project Stats
                  </div>
                  <div className="mt-2 space-y-2 text-sm">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[var(--text-muted)]">Segments</span>
                      <span className="font-semibold text-[var(--text-primary)]">
                        {selectedProjectSegmentCount}
                      </span>
                    </div>
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[var(--text-muted)]">Ready</span>
                      <span className="font-semibold text-[var(--text-primary)]">
                        {readySegmentCount}
                      </span>
                    </div>
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[var(--text-muted)]">Needs Render</span>
                      <span className="font-semibold text-[var(--text-primary)]">
                        {pendingRenderSegmentCount}
                      </span>
                    </div>
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[var(--text-muted)]">Script Size</span>
                      <span className="font-semibold text-[var(--text-primary)]">
                        {selectedProjectTotalChars} chars
                      </span>
                    </div>
                  </div>
                </div>

                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                    Completion
                  </div>
                  <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-[var(--bg-surface-2)]">
                    <div
                      className="h-full rounded-full bg-[var(--accent-solid)] transition-[width] duration-300"
                      style={{ width: `${selectedProjectCompletionPercent}%` }}
                    />
                  </div>
                  <div className="mt-2 text-xs text-[var(--text-muted)]">
                    {selectedProjectRenderedCount}/{selectedProjectSegmentCount} segments complete
                  </div>
                </div>

                <section className="space-y-3">
                  <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                    Render Queue
                  </div>
                  <div className="text-sm font-semibold text-[var(--text-primary)]">
                    {queuedRenderCount} queued · {failedRenderCount} failed
                  </div>

                  <div className="space-y-2 min-[1800px]:max-h-[320px] min-[1800px]:overflow-y-auto min-[1800px]:pr-1">
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
                </section>
              </div>
            }
            editor={
              <StudioSegmentEditor
                project={selectedProject}
                segmentDrafts={segmentDrafts}
                segmentDraftConflicts={segmentDraftConflicts}
                segmentSelections={segmentSelections}
                selectedSegmentIdSet={selectedSegmentIdSet}
                selectedSegmentCount={selectedSegmentCount}
                queuedSegmentIdSet={queuedSegmentIdSet}
                savingSegmentId={savingSegmentId}
                renderingSegmentId={renderingSegmentId}
                addingSegmentAfterSegmentId={addingSegmentAfterSegmentId}
                focusSegmentId={focusSegmentId}
                onToggleSelectAll={() =>
                  setSelectedSegmentIds(
                    selectedSegmentCount === selectedProject.segments.length
                      ? []
                      : selectedProject.segments.map((segment) => segment.id),
                  )
                }
                onRenderSelected={() => void handleRenderSelectedSegments()}
                onDeleteSelected={() => void handleBulkDeleteSegments()}
                onAddSegment={(afterSegmentId, text) =>
                  handleAddSegment(afterSegmentId, text)
                }
                onToggleSegmentSelection={(segmentId, checked) =>
                  setSelectedSegmentIds((current) => {
                    if (checked) {
                      return [...new Set([...current, segmentId])];
                    }
                    return current.filter((id) => id !== segmentId);
                  })
                }
                onSaveSegment={(segmentId) => void handleSaveSegment(segmentId)}
                onMoveSegment={(segmentId, direction) =>
                  void handleMoveSegment(segmentId, direction)
                }
                onMergeSegmentWithNext={(segmentId) =>
                  void handleMergeSegmentWithNext(segmentId)
                }
                onSplitSegment={(segmentId) => void handleSplitSegment(segmentId)}
                onRenderSegment={(segmentId) => void handleRenderSegment(segmentId)}
                onDeleteSegment={(segmentId) => void handleDeleteSegment(segmentId)}
                onOpenSegmentSettings={(segmentId) =>
                  openSegmentSettingsDrawer(segmentId)
                }
                onChangeSegmentDraft={(segmentId, value) => {
                  const segment = selectedProject.segments.find(
                    (candidate) => candidate.id === segmentId,
                  );
                  if (segment) {
                    storeStudioSegmentDraft(selectedProject.id, {
                      segmentId,
                      baseText: segment.text,
                      draftText: value,
                    });
                  }
                  setSegmentDrafts((current) => ({
                    ...current,
                    [segmentId]: value,
                  }));
                }}
                onRestoreSegmentDraftConflict={(segmentId) => {
                  const conflict = segmentDraftConflicts[segmentId];
                  const segment = selectedProject.segments.find(
                    (candidate) => candidate.id === segmentId,
                  );
                  if (!conflict || !segment) {
                    return;
                  }
                  storeStudioSegmentDraft(selectedProject.id, {
                    segmentId,
                    baseText: segment.text,
                    draftText: conflict.draftText,
                  });
                  setSegmentDrafts((current) => ({
                    ...current,
                    [segmentId]: conflict.draftText,
                  }));
                  setSegmentDraftConflicts((current) => {
                    const next = { ...current };
                    delete next[segmentId];
                    return next;
                  });
                }}
                onDiscardSegmentDraftConflict={(segmentId) => {
                  removeStudioSegmentDraft(selectedProject.id, segmentId);
                  const segment = selectedProject.segments.find(
                    (candidate) => candidate.id === segmentId,
                  );
                  if (segment) {
                    setSegmentDrafts((current) => ({
                      ...current,
                      [segmentId]: segment.text,
                    }));
                  }
                  setSegmentDraftConflicts((current) => {
                    const next = { ...current };
                    delete next[segmentId];
                    return next;
                  });
                }}
                onChangeSegmentCursor={(segmentId, cursor) =>
                  setSegmentSelections((current) => ({
                    ...current,
                    [segmentId]: cursor,
                  }))
                }
                onFocusSegmentHandled={(segmentId) =>
                  setFocusSegmentId((current) =>
                    current === segmentId ? null : current,
                  )
                }
                audioUrlForRecordId={(recordId) => api.textToSpeechRecordAudioUrl(recordId)}
              />
            }
            actionRail={
              <div className="space-y-8">
                <div className="space-y-1">
                  <div className="text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                    Configuration
                  </div>
                  <p className="text-sm text-[var(--text-secondary)]">
                    Configure and maintain this project.
                  </p>
                </div>

                <Accordion type="multiple" defaultValue={[]} className="space-y-3">
                  <AccordionItem
                    value="project-profile"
                    className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3"
                  >
                    <AccordionTrigger className="py-3 text-left hover:no-underline">
                      <div className="flex min-w-0 items-center gap-2">
                        <span className="text-sm font-semibold text-[var(--text-primary)]">
                          Project settings
                        </span>
                        {projectDirty ? (
                          <span className="rounded-full border border-[var(--status-warning-border)] bg-[var(--status-warning-bg)] px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--status-warning-text)]">
                            Unsaved
                          </span>
                        ) : null}
                      </div>
                    </AccordionTrigger>
                    <AccordionContent className="pb-3">
                      <p className="text-sm leading-relaxed text-[var(--text-secondary)]">
                        These defaults apply across the full project. Save them before
                        you start a long render pass, or let render actions sync them automatically.
                      </p>

                      <div className="mt-4 space-y-4">
                        <div className={settingsFieldClass}>
                          <label className={settingsLabelClass}>
                            Project name
                          </label>
                          <Input
                            className={settingsControlHeightClass}
                            value={projectName}
                            onChange={(event) => setProjectName(event.target.value)}
                          />
                        </div>

                        {onOpenModelManager ? (
                          <div className={settingsFieldClass}>
                            <label className={settingsLabelClass}>Models</label>
                            <Button
                              variant="outline"
                              size="lg"
                              className="w-full justify-center rounded-[14px] border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-4 text-[15px] shadow-none hover:bg-[var(--bg-surface-1)]"
                              onClick={onOpenModelManager}
                            >
                              <SlidersHorizontal className="h-4 w-4" />
                              Models
                            </Button>
                          </div>
                        ) : null}

                        <div className={settingsFieldClass}>
                          <label className={settingsLabelClass}>
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
                            triggerClassName={settingsControlHeightClass}
                          />
                        </div>

                        <div className={settingsFieldClass}>
                          <label className={settingsLabelClass}>
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
                            hideModelInTriggerSubtitle
                          />
                        </div>

                        <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-4">
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
                        size="lg"
                        onClick={() => void persistProjectSettings()}
                        disabled={!projectDirty || savingProject}
                        className="mt-4 w-full justify-center bg-[var(--bg-surface-0)]"
                      >
                        {savingProject ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Settings2 className="h-4 w-4" />
                        )}
                        Save profile
                      </Button>

                      <div className="mt-3">
                        <Button
                          variant="outline"
                          size="lg"
                          onClick={() => void handleDeleteProject()}
                          disabled={deletingProject}
                          className="w-full justify-center border-[var(--danger-border)] bg-[var(--danger-bg)] text-[var(--danger-text)] hover:bg-[var(--danger-bg-hover)] hover:text-[var(--danger-text)]"
                        >
                          {deletingProject ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Trash2 className="h-4 w-4" />
                          )}
                          Delete project
                        </Button>
                      </div>
                    </AccordionContent>
                  </AccordionItem>

                  <AccordionItem
                    value="pronunciation-rules"
                    className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3"
                  >
                    <AccordionTrigger className="py-3 text-left hover:no-underline">
                      <span className="text-sm font-semibold text-[var(--text-primary)]">
                        Pronunciation Rules
                      </span>
                    </AccordionTrigger>
                    <AccordionContent className="pb-3">
                      <p className="text-sm leading-relaxed text-[var(--text-secondary)]">
                        Replace words or phrases before rendering to keep pronunciation
                        consistent across the project.
                      </p>

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
                          className="justify-center bg-[var(--bg-surface-0)]"
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
                          <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-2 text-xs text-[var(--text-muted)]">
                            Loading pronunciation rules...
                          </div>
                        ) : projectPronunciations.length === 0 ? (
                          <div className="rounded-xl border border-dashed border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-2 text-xs text-[var(--text-muted)]">
                            No pronunciation rules yet.
                          </div>
                        ) : (
                          projectPronunciations.map((entry) => (
                            <div
                              key={entry.id}
                              className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-2"
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
                    </AccordionContent>
                  </AccordionItem>
                </Accordion>

              </div>
            }
          />
        )}
      </div>
    </>
  );
}
