import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  AlertTriangle,
  Loader2,
  Mic2,
  Settings2,
  Sparkles,
  Trash2,
} from "lucide-react";
import type { ModelInfo, SavedVoiceSummary } from "@/api";
import { api } from "@/api";
import { PageHeader, PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { StatePanel } from "@/components/ui/state-panel";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { getSpeakerProfilesForVariant, isLfm25AudioVariant } from "@/types";
import {
  TEXT_TO_SPEECH_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import type { ModelDownloadProgressMap } from "@/features/models/downloadProgress";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import { useRouteModelSelection } from "@/features/models/hooks/useRouteModelSelection";
import { VoiceLibraryTable } from "@/features/voices/components/VoiceLibraryTable";
import { type VoiceLibraryItem } from "@/features/voices/types";
import { cn } from "@/lib/utils";

const SAVED_VOICES_PAGE_LIMIT = 25;

interface VoicesPageProps {
  models: ModelInfo[];
  selectedModel: string | null;
  loading: boolean;
  downloadProgress: ModelDownloadProgressMap;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onSelect: (variant: string) => void;
  onError: (message: string) => void;
  embedded?: boolean;
  refreshKey?: number;
  openModelManagerRequestKey?: number;
}

type VoiceLibraryTab = "all" | "saved" | "built-in";

const BUILT_IN_PREVIEW_TEXT = {
  chinese: "你好，欢迎使用 Izwi 的内置语音预览。",
  japanese: "こんにちは。Izwi の音声プレビューです。",
  korean: "안녕하세요. Izwi 음성 미리보기입니다.",
  spanish: "Hola. Esta es una muestra de voz de Izwi.",
  french: "Bonjour. Ceci est un apercu vocal Izwi.",
  hindi: "Namaste. Yeh Izwi ki awaaz preview hai.",
  italian: "Ciao. Questa e una voce di anteprima Izwi.",
  portuguese: "Ola. Esta e uma amostra de voz do Izwi.",
  english: "Hello. This is an Izwi built-in voice preview.",
} as const;

function formatRelativeDate(timestampMs: number): string {
  const value = new Date(timestampMs);
  if (Number.isNaN(value.getTime())) {
    return "Unknown date";
  }
  return value.toLocaleDateString([], {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function savedVoiceSourceLabel(
  source: SavedVoiceSummary["source_route_kind"],
): string {
  switch (source) {
    case "voice_cloning":
      return "Cloned voice";
    case "voice_design":
      return "Designed voice";
    default:
      return "Saved voice";
  }
}

function shouldShowSavedVoiceCreatedDate(
  source: SavedVoiceSummary["source_route_kind"],
): boolean {
  return source === "voice_cloning" || source === "voice_design";
}

function previewTextForLanguage(language: string): string {
  const normalized = language.toLowerCase();
  if (normalized.includes("chinese")) return BUILT_IN_PREVIEW_TEXT.chinese;
  if (normalized.includes("japanese")) return BUILT_IN_PREVIEW_TEXT.japanese;
  if (normalized.includes("korean")) return BUILT_IN_PREVIEW_TEXT.korean;
  if (normalized.includes("spanish")) return BUILT_IN_PREVIEW_TEXT.spanish;
  if (normalized.includes("french")) return BUILT_IN_PREVIEW_TEXT.french;
  if (normalized.includes("hindi")) return BUILT_IN_PREVIEW_TEXT.hindi;
  if (normalized.includes("italian")) return BUILT_IN_PREVIEW_TEXT.italian;
  if (normalized.includes("portuguese"))
    return BUILT_IN_PREVIEW_TEXT.portuguese;
  return BUILT_IN_PREVIEW_TEXT.english;
}

export function VoicesPage({
  models,
  selectedModel,
  loading,
  downloadProgress,
  onDownload,
  onCancelDownload,
  onLoad,
  onUnload,
  onDelete,
  onSelect,
  onError,
  embedded = false,
  refreshKey = 0,
  openModelManagerRequestKey,
}: VoicesPageProps) {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<VoiceLibraryTab>("saved");
  const [savedVoices, setSavedVoices] = useState<SavedVoiceSummary[]>([]);
  const [savedVoicesLoading, setSavedVoicesLoading] = useState(true);
  const [savedVoicesLoadingMore, setSavedVoicesLoadingMore] = useState(false);
  const [savedVoicesError, setSavedVoicesError] = useState<string | null>(null);
  const [savedVoicesNextCursor, setSavedVoicesNextCursor] = useState<
    string | null
  >(null);
  const [savedVoicesHasMore, setSavedVoicesHasMore] = useState(false);
  const [deletingVoiceId, setDeletingVoiceId] = useState<string | null>(null);
  const [deleteConfirmVoiceId, setDeleteConfirmVoiceId] = useState<string | null>(
    null,
  );
  const [deleteConfirmError, setDeleteConfirmError] = useState<string | null>(
    null,
  );
  const [previewLoadingVoiceId, setPreviewLoadingVoiceId] = useState<
    string | null
  >(null);
  const [previewUrls, setPreviewUrls] = useState<Record<string, string>>({});
  const previewUrlsRef = useRef<Record<string, string>>({});
  const selectedBuiltInModelRef = useRef<string | null>(null);
  const previousOpenModelManagerRequestKeyRef = useRef<
    number | undefined
  >(openModelManagerRequestKey);

  const {
    routeModels,
    resolvedSelectedModel,
    selectedModelReady,
    isModelModalOpen,
    intentVariant,
    closeModelModal,
    openModelManager,
    requestModel,
    handleModelSelect,
  } = useRouteModelSelection({
    models,
    selectedModel,
    onSelect,
    modelFilter: (variant) => {
      const match = models.find((model) => model.variant === variant);
      return (
        match?.speech_capabilities?.supports_builtin_voices === true &&
        !isLfm25AudioVariant(variant) &&
        !variant.includes("Tokenizer")
      );
    },
    resolveSelectedModel: (routeModels, currentModel) =>
      resolvePreferredRouteModel({
        models: routeModels,
        selectedModel: currentModel,
        preferredVariants: TEXT_TO_SPEECH_PREFERRED_MODELS,
      }),
  });

  useEffect(() => {
    previewUrlsRef.current = previewUrls;
  }, [previewUrls]);

  useEffect(() => {
    selectedBuiltInModelRef.current = resolvedSelectedModel;
  }, [resolvedSelectedModel]);

  useEffect(() => {
    if (openModelManagerRequestKey === undefined) {
      return;
    }
    if (previousOpenModelManagerRequestKeyRef.current === openModelManagerRequestKey) {
      return;
    }
    previousOpenModelManagerRequestKeyRef.current = openModelManagerRequestKey;
    openModelManager();
  }, [openModelManager, openModelManagerRequestKey]);

  const buildPreviewKey = useCallback(
    (modelId: string | null, voiceId: string) => `${modelId ?? "__none__"}::${voiceId}`,
    [],
  );

  useEffect(() => {
    return () => {
      Object.values(previewUrlsRef.current).forEach((url) => {
        if (url.startsWith("blob:")) {
          URL.revokeObjectURL(url);
        }
      });
    };
  }, []);

  const loadSavedVoices = useCallback(async () => {
    setSavedVoicesLoading(true);
    setSavedVoicesError(null);
    try {
      const page = await api.listSavedVoicePage({
        limit: SAVED_VOICES_PAGE_LIMIT,
        cursor: null,
      });
      setSavedVoices(page.items);
      setSavedVoicesNextCursor(page.pagination.next_cursor);
      setSavedVoicesHasMore(page.pagination.has_more);
    } catch (error) {
      setSavedVoicesError(
        error instanceof Error ? error.message : "Failed to load saved voices.",
      );
    } finally {
      setSavedVoicesLoading(false);
    }
  }, []);

  const loadMoreSavedVoices = useCallback(async () => {
    if (
      savedVoicesLoading ||
      savedVoicesLoadingMore ||
      !savedVoicesHasMore ||
      !savedVoicesNextCursor
    ) {
      return;
    }
    setSavedVoicesLoadingMore(true);
    setSavedVoicesError(null);
    try {
      const page = await api.listSavedVoicePage({
        limit: SAVED_VOICES_PAGE_LIMIT,
        cursor: savedVoicesNextCursor,
      });
      setSavedVoices((current) => {
        const seen = new Set(current.map((voice) => voice.id));
        const nextItems = page.items.filter((voice) => !seen.has(voice.id));
        return [...current, ...nextItems];
      });
      setSavedVoicesNextCursor(page.pagination.next_cursor);
      setSavedVoicesHasMore(page.pagination.has_more);
    } catch (error) {
      setSavedVoicesError(
        error instanceof Error ? error.message : "Failed to load saved voices.",
      );
    } finally {
      setSavedVoicesLoadingMore(false);
    }
  }, [
    savedVoicesHasMore,
    savedVoicesLoading,
    savedVoicesLoadingMore,
    savedVoicesNextCursor,
  ]);

  useEffect(() => {
    setSavedVoices([]);
    setSavedVoicesNextCursor(null);
    setSavedVoicesHasMore(false);
    setSavedVoicesError(null);
    setSavedVoicesLoading(true);
    void loadSavedVoices();
  }, [loadSavedVoices, refreshKey]);

  const builtInVoices = useMemo(
    () => getSpeakerProfilesForVariant(resolvedSelectedModel),
    [resolvedSelectedModel],
  );

  const savedVoiceModels = useMemo(
    () =>
      models.filter(
        (model) =>
          !model.variant.includes("Tokenizer") &&
          model.speech_capabilities?.supports_reference_voice === true,
      ),
    [models],
  );

  const preferredSavedVoiceModel = useMemo(() => {
    if (savedVoiceModels.length === 0) {
      return null;
    }

    const selectedSavedVoiceModel =
      resolvedSelectedModel &&
      savedVoiceModels.some((model) => model.variant === resolvedSelectedModel)
        ? resolvedSelectedModel
        : null;

    return resolvePreferredRouteModel({
      models: savedVoiceModels,
      selectedModel: selectedSavedVoiceModel,
      preferredVariants: TEXT_TO_SPEECH_PREFERRED_MODELS,
    });
  }, [resolvedSelectedModel, savedVoiceModels]);

  const handleUseSavedVoice = (voiceId: string) => {
    const params = new URLSearchParams();
    params.set("voiceId", voiceId);
    if (preferredSavedVoiceModel) {
      params.set("model", preferredSavedVoiceModel);
    }
    navigate(`/text-to-speech?${params.toString()}`);
  };

  const handleUseBuiltInVoice = (speaker: string) => {
    const params = new URLSearchParams();
    params.set("speaker", speaker);
    if (resolvedSelectedModel) {
      params.set("model", resolvedSelectedModel);
    }
    navigate(`/text-to-speech?${params.toString()}`);
  };

  const handleDeleteVoice = async () => {
    if (!deleteConfirmVoiceId || deletingVoiceId) {
      return;
    }

    setDeletingVoiceId(deleteConfirmVoiceId);
    setDeleteConfirmError(null);
    try {
      await api.deleteSavedVoice(deleteConfirmVoiceId);
      setDeleteConfirmVoiceId(null);
      setDeleteConfirmError(null);
      await loadSavedVoices();
    } catch (error) {
      setDeleteConfirmError(
        error instanceof Error
          ? error.message
          : "Failed to delete saved voice.",
      );
    } finally {
      setDeletingVoiceId(null);
    }
  };

  const handlePreviewBuiltInVoice = async (
    voiceId: string,
    language: string,
  ) => {
    if (!resolvedSelectedModel) {
      requestModel();
      onError("Select and load a built-in voice model to generate previews.");
      return;
    }
    const requestModelId = resolvedSelectedModel;
    const previewKey = buildPreviewKey(requestModelId, voiceId);

    if (previewUrls[previewKey]) {
      return;
    }
    if (!selectedModelReady) {
      requestModel(resolvedSelectedModel);
      onError("Load the selected voice model before generating a preview.");
      return;
    }

    setPreviewLoadingVoiceId(previewKey);
    try {
      const result = await api.generateTTSWithStats({
        model_id: requestModelId,
        text: previewTextForLanguage(language),
        speaker: voiceId,
      });
      if (selectedBuiltInModelRef.current !== requestModelId) {
        return;
      }
      const url = URL.createObjectURL(result.audioBlob);
      setPreviewUrls((current) => ({ ...current, [previewKey]: url }));
    } catch (error) {
      onError(
        error instanceof Error
          ? error.message
          : "Failed to generate built-in voice preview.",
      );
    } finally {
      setPreviewLoadingVoiceId((current) =>
        current === previewKey ? null : current,
      );
    }
  };
  const tableActionButtonClass =
    "h-8 w-[7.5rem] rounded-[0.6rem] px-3 text-xs font-semibold";
  const deleteTargetVoice = savedVoices.find(
    (voice) => voice.id === deleteConfirmVoiceId,
  );

  const savedVoiceItems: VoiceLibraryItem[] = savedVoices.map(
    (voice) => ({
      id: voice.id,
      name: voice.name,
      secondaryLabel: shouldShowSavedVoiceCreatedDate(voice.source_route_kind)
        ? `Created ${formatRelativeDate(voice.created_at)}`
        : undefined,
      categoryLabel: savedVoiceSourceLabel(voice.source_route_kind),
      description: voice.reference_text_preview,
      previewUrl: api.savedVoiceAudioUrl(voice.id),
      actions: (
        <>
          <Button
            size="sm"
            className={tableActionButtonClass}
            onClick={(event) => {
              event.stopPropagation();
              handleUseSavedVoice(voice.id);
            }}
          >
            <Mic2 className="h-4 w-4" />
            Use in TTS
          </Button>
          <Button
            variant="outline"
            size="sm"
            className={cn(
              tableActionButtonClass,
              "border-[var(--border-strong)] bg-[var(--bg-surface-1)]/72",
            )}
            onClick={(event) => {
              event.stopPropagation();
              setDeleteConfirmVoiceId(voice.id);
              setDeleteConfirmError(null);
            }}
            disabled={deletingVoiceId === voice.id}
          >
            {deletingVoiceId === voice.id ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Trash2 className="h-4 w-4" />
            )}
            Delete
          </Button>
        </>
      ),
    }),
  );

  const builtInVoiceItems: VoiceLibraryItem[] = builtInVoices.map(
    (voice) => {
      const previewKey = buildPreviewKey(resolvedSelectedModel, voice.id);
      const previewUrl = previewUrls[previewKey] ?? null;
      const isPreviewLoading = previewLoadingVoiceId === previewKey;
      return {
        id: voice.id,
        name: voice.name,
        categoryLabel: "Built-in voice",
        description: voice.description,
        previewUrl,
        previewLoading: isPreviewLoading,
        previewMessage: previewUrl
          ? null
          : isPreviewLoading
            ? "Generating a preview sample for this speaker."
            : "Generate a preview sample to audition this built-in voice.",
        actions: (
          <>
            <Button
              size="sm"
              onClick={(event) => {
                event.stopPropagation();
                handleUseBuiltInVoice(voice.id);
              }}
              className={tableActionButtonClass}
            >
              <Mic2 className="h-4 w-4" />
              Use in TTS
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={(event) => {
                event.stopPropagation();
                void handlePreviewBuiltInVoice(voice.id, voice.language);
              }}
              disabled={isPreviewLoading}
              className={cn(
                tableActionButtonClass,
                "border-[var(--border-strong)] bg-[var(--bg-surface-1)]/72",
              )}
            >
              {isPreviewLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Sparkles className="h-4 w-4" />
              )}
              Preview
            </Button>
          </>
        ),
      };
    },
  );

  const totalSavedVoices = savedVoices.length;
  const totalBuiltInVoices = builtInVoices.length;
  const allVoiceItems = [...savedVoiceItems, ...builtInVoiceItems];
  const activeItems =
    activeTab === "all"
      ? allVoiceItems
      : activeTab === "saved"
        ? savedVoiceItems
        : builtInVoiceItems;
  const showSavedVoiceLoadMore =
    (activeTab === "saved" || activeTab === "all") &&
    savedVoices.length > 0 &&
    savedVoicesHasMore &&
    Boolean(savedVoicesNextCursor);
  const showSavedVoiceError =
    savedVoicesError && (activeTab === "saved" || activeTab === "all");

  const workspaceContent = (
    <div className="flex flex-col">
      <Tabs
        value={activeTab}
        onValueChange={(value) => setActiveTab(value as VoiceLibraryTab)}
        className="w-full"
      >
        <div className={cn("pb-4 sm:pb-5", embedded && "pb-3 sm:pb-4")}>
          <div className="flex flex-col gap-4 pb-4">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <TabsList className="grid h-10 w-full max-w-[30rem] grid-cols-3 overflow-hidden rounded-[var(--radius-pill)] border-[var(--border-strong)] bg-[var(--bg-surface-2)] p-[2px] shadow-none">
                <TabsTrigger
                  value="all"
                  className="h-full rounded-[var(--radius-pill)] px-3 text-[13px] font-semibold text-[var(--text-muted)] data-[state=active]:bg-[var(--bg-surface-1)] data-[state=active]:text-[var(--text-primary)] data-[state=active]:shadow-none"
                >
                  All Voices
                </TabsTrigger>
                <TabsTrigger
                  value="saved"
                  className="h-full gap-2 rounded-[var(--radius-pill)] px-3 text-[13px] font-semibold text-[var(--text-muted)] data-[state=active]:bg-[var(--bg-surface-1)] data-[state=active]:text-[var(--text-primary)] data-[state=active]:shadow-none"
                >
                  <span>My Voices</span>
                  <span className="rounded-full border border-current/20 px-2 py-0.5 text-[10px] font-semibold">
                    {totalSavedVoices}
                  </span>
                </TabsTrigger>
                <TabsTrigger
                  value="built-in"
                  className="h-full gap-2 rounded-[var(--radius-pill)] px-3 text-[13px] font-semibold text-[var(--text-muted)] data-[state=active]:bg-[var(--bg-surface-1)] data-[state=active]:text-[var(--text-primary)] data-[state=active]:shadow-none"
                >
                  <span>Built-in Voices</span>
                  <span className="rounded-full border border-current/20 px-2 py-0.5 text-[10px] font-semibold">
                    {totalBuiltInVoices}
                  </span>
                </TabsTrigger>
              </TabsList>

            </div>

          </div>

          {showSavedVoiceError ? (
            <div className="mt-4">
              <StatePanel
                title="Saved voices unavailable"
                description={savedVoicesError}
                tone="danger"
              />
            </div>
          ) : null}

          <VoiceLibraryTable
            items={activeItems}
            emptyTitle={
              activeTab === "all"
                ? savedVoicesLoading
                  ? "Loading voices"
                  : "No voices yet"
                : activeTab === "saved"
                  ? savedVoicesLoading
                    ? "Loading saved voices"
                    : "No saved voices yet"
                  : "No built-in voices available"
            }
            emptyDescription={
              activeTab === "all"
                ? routeModels.length === 0
                  ? "Save a voice or load a supported built-in voice model to populate the library."
                  : "Save a voice from voice cloning or voice design to build out your library."
                : activeTab === "saved"
                  ? savedVoicesLoading
                    ? "Fetching your reusable cloned and designed voices."
                    : "Save a result from voice cloning or voice design to build a reusable voice library."
                  : routeModels.length === 0
                    ? "Load a CustomVoice or Kokoro model to browse built-in voices."
                    : "Try a different built-in voice model."
            }
            className="mt-5"
            compact={embedded}
          />

          {showSavedVoiceLoadMore ? (
            <div className="mt-3 flex justify-center rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] px-3 py-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-9 gap-2"
                onClick={() => void loadMoreSavedVoices()}
                disabled={savedVoicesLoadingMore}
              >
                {savedVoicesLoadingMore ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : null}
                Load more
              </Button>
            </div>
          ) : null}
        </div>
      </Tabs>
    </div>
  );

  const deleteVoiceDialog = (
    <Dialog
      open={deleteConfirmVoiceId !== null}
      onOpenChange={(open) => {
        if (!deletingVoiceId && !open) {
          setDeleteConfirmVoiceId(null);
          setDeleteConfirmError(null);
        }
      }}
    >
      <DialogContent className="max-w-md border-[var(--border-strong)] bg-[var(--bg-surface-1)] p-5">
        <DialogTitle className="sr-only">Delete voice?</DialogTitle>
        <div className="flex items-start gap-3">
          <div className="mt-0.5 rounded-full border border-[var(--danger-border)] bg-[var(--danger-bg)] p-2 text-[var(--danger-text)]">
            <AlertTriangle className="h-4 w-4" />
          </div>
          <div className="min-w-0 flex-1">
            <h3 className="text-sm font-semibold text-[var(--text-primary)]">
              Delete voice?
            </h3>
            <DialogDescription className="mt-1 text-sm text-[var(--text-muted)]">
              This permanently removes the saved voice from your library.
            </DialogDescription>
            <p className="mt-2 truncate text-xs text-[var(--text-subtle)]">
              {deleteTargetVoice?.name || deleteConfirmVoiceId}
            </p>
          </div>
        </div>

        {deleteConfirmError ? (
          <div className="mt-4 rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] px-3 py-2 text-xs text-[var(--danger-text)]">
            {deleteConfirmError}
          </div>
        ) : null}

        <div className="mt-5 flex items-center justify-end gap-2">
          <Button
            type="button"
            variant="outline"
            onClick={() => setDeleteConfirmVoiceId(null)}
            size="sm"
            className="h-8 border-[var(--border-muted)] bg-[var(--bg-surface-2)] text-[var(--text-secondary)] hover:bg-[var(--bg-surface-3)]"
            disabled={deletingVoiceId !== null}
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="destructive"
            size="sm"
            className="h-8 gap-1.5"
            onClick={() => void handleDeleteVoice()}
            disabled={deletingVoiceId !== null}
          >
            {deletingVoiceId ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Deleting
              </>
            ) : (
              "Delete voice"
            )}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );

  const routeModelModal = (
    <RouteModelModal
      isOpen={isModelModalOpen}
      onClose={closeModelModal}
      title="Built-in Voice Models"
      description="Manage the voice models that expose built-in speaker libraries."
      models={routeModels}
      loading={loading}
      selectedVariant={resolvedSelectedModel}
      intentVariant={intentVariant}
      downloadProgress={downloadProgress}
      onDownload={onDownload}
      onCancelDownload={onCancelDownload}
      onLoad={onLoad}
      onUnload={onUnload}
      onDelete={onDelete}
      onUseModel={handleModelSelect}
      emptyMessage="Load a CustomVoice or Kokoro model to browse built-in voices."
    />
  );

  if (embedded) {
    return (
      <>
        {workspaceContent}
        {deleteVoiceDialog}
        {routeModelModal}
      </>
    );
  }

  return (
    <PageShell>
      <PageHeader
        title="Voices"
        description="Manage, browse, and use your saved, cloned, and designed voices for text-to-speech."
        actions={
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="h-9 gap-2"
            onClick={openModelManager}
          >
            <Settings2 className="h-4 w-4" />
            Models
          </Button>
        }
      />
      {workspaceContent}
      {deleteVoiceDialog}
      {routeModelModal}
    </PageShell>
  );
}
