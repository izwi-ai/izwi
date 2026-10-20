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
import { getSpeakerProfilesForVariant, isLfm25AudioVariant } from "@/types";
import {
  TEXT_TO_SPEECH_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import type { ModelDownloadProgressMap } from "@/features/models/downloadProgress";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import { useRouteModelSelection } from "@/features/models/hooks/useRouteModelSelection";
import { VoiceLibraryGrid } from "@/features/voices/components/VoiceLibraryGrid";
import {
  HistoryToolbar,
  SearchField,
  SegmentedFilter,
} from "@/components/history/HistoryToolbar";
import { getModelDisplayName } from "@/features/models/catalog/modelMetadata";
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
  const [voiceSearch, setVoiceSearch] = useState("");
  // Saved voices are searched on the server (all of them, not the loaded
  // pages), shortly after typing stops.
  const [savedVoiceQuery, setSavedVoiceQuery] = useState("");
  useEffect(() => {
    const timer = window.setTimeout(
      () => setSavedVoiceQuery(voiceSearch.trim()),
      250,
    );
    return () => window.clearTimeout(timer);
  }, [voiceSearch]);
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
        filters: savedVoiceQuery ? { q: savedVoiceQuery } : undefined,
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
  }, [savedVoiceQuery]);

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
        filters: savedVoiceQuery ? { q: savedVoiceQuery } : undefined,
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
    savedVoiceQuery,
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
  const deleteTargetVoice = savedVoices.find(
    (voice) => voice.id === deleteConfirmVoiceId,
  );
  const builtInModelLabel = resolvedSelectedModel
    ? getModelDisplayName(resolvedSelectedModel)
    : null;

  const savedVoiceItems: VoiceLibraryItem[] = savedVoices.map((voice) => {
    const kind = savedVoiceSourceLabel(voice.source_route_kind).replace(
      / voice$/,
      "",
    );
    return {
      id: voice.id,
      name: voice.name,
      meta: shouldShowSavedVoiceCreatedDate(voice.source_route_kind)
        ? `${kind} · ${formatRelativeDate(voice.created_at)}`
        : kind,
      kindLabel: kind,
      description: voice.reference_text_preview,
      previewUrl: api.savedVoiceAudioUrl(voice.id),
      primaryAction: {
        label: "Use in TTS",
        icon: Mic2,
        onClick: () => handleUseSavedVoice(voice.id),
      },
      menuItems: [
        {
          id: "use",
          label: "Use in TTS",
          icon: Mic2,
          onSelect: () => handleUseSavedVoice(voice.id),
        },
        {
          id: "delete",
          label: "Delete",
          icon: Trash2,
          danger: true,
          separatorBefore: true,
          disabled: deletingVoiceId === voice.id,
          onSelect: () => {
            setDeleteConfirmVoiceId(voice.id);
            setDeleteConfirmError(null);
          },
        },
      ],
    };
  });

  const builtInVoiceItems: VoiceLibraryItem[] = builtInVoices.map((voice) => {
    const previewKey = buildPreviewKey(resolvedSelectedModel, voice.id);
    const previewUrl = previewUrls[previewKey] ?? null;
    const isPreviewLoading = previewLoadingVoiceId === previewKey;
    return {
      id: voice.id,
      name: voice.name,
      meta: builtInModelLabel ? `Built-in · ${builtInModelLabel}` : "Built-in",
      kindLabel: "Built-in",
      description: voice.description,
      previewUrl,
      previewLoading: isPreviewLoading,
      onRequestPreview: () =>
        void handlePreviewBuiltInVoice(voice.id, voice.language),
      primaryAction: {
        label: "Use in TTS",
        icon: Mic2,
        onClick: () => handleUseBuiltInVoice(voice.id),
      },
      menuItems: [
        {
          id: "use",
          label: "Use in TTS",
          icon: Mic2,
          onSelect: () => handleUseBuiltInVoice(voice.id),
        },
        {
          id: "preview",
          label: previewUrl ? "Preview ready" : "Generate preview",
          icon: Sparkles,
          disabled: Boolean(previewUrl) || isPreviewLoading,
          onSelect: () =>
            void handlePreviewBuiltInVoice(voice.id, voice.language),
        },
      ],
    };
  });

  const totalSavedVoices = savedVoices.length;
  const totalBuiltInVoices = builtInVoices.length;
  const allVoiceItems = [...savedVoiceItems, ...builtInVoiceItems];
  const tabItems =
    activeTab === "all"
      ? allVoiceItems
      : activeTab === "saved"
        ? savedVoiceItems
        : builtInVoiceItems;
  // Saved voices come back already searched by the server; built-in voices
  // are searched here by name, notes and kind.
  const searchTerm = voiceSearch.trim().toLowerCase();
  const savedItemIds = new Set(savedVoiceItems.map((item) => item.id));
  const activeItems = searchTerm
    ? tabItems.filter(
        (item) =>
          savedItemIds.has(item.id) ||
          [item.name, item.description, item.meta]
            .filter(Boolean)
            .some((value) => value!.toLowerCase().includes(searchTerm)),
      )
    : tabItems;
  const noMatches = Boolean(searchTerm) && !savedVoicesLoading;
  const showSavedVoiceLoadMore =
    (activeTab === "saved" || activeTab === "all") &&
    savedVoices.length > 0 &&
    savedVoicesHasMore &&
    Boolean(savedVoicesNextCursor);
  const showSavedVoiceError =
    savedVoicesError && (activeTab === "saved" || activeTab === "all");

  const workspaceContent = (
    <div className="flex flex-col">
      <div>
        <div className={cn("pb-4 sm:pb-5", embedded && "pb-3 sm:pb-4")}>
          <HistoryToolbar>
            <SegmentedFilter
              label="Show"
              options={[
                {
                  id: "all",
                  label: "All",
                  count: totalSavedVoices + totalBuiltInVoices,
                },
                { id: "saved", label: "My voices", count: totalSavedVoices },
                {
                  id: "built-in",
                  label: "Built-in",
                  count: totalBuiltInVoices,
                },
              ]}
              value={activeTab}
              onChange={setActiveTab}
            />
            <SearchField
              value={voiceSearch}
              onChange={setVoiceSearch}
              placeholder="Search voices…"
            />
          </HistoryToolbar>

          {showSavedVoiceError ? (
            <div className="mt-4">
              <StatePanel
                title="Saved voices unavailable"
                description={savedVoicesError}
                tone="danger"
              />
            </div>
          ) : null}

          <VoiceLibraryGrid
            items={activeItems}
            emptyTitle={
              noMatches
                ? "No matches"
                : activeTab === "all"
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
              noMatches
                ? `No voice matches "${voiceSearch.trim()}".`
                : activeTab === "all"
                ? routeModels.length === 0
                  ? "Save a voice or load a supported built-in voice model to populate the library."
                  : "Save a voice from voice cloning or voice design to build out your library."
                : activeTab === "saved"
                  ? savedVoicesLoading
                    ? "Fetching your reusable cloned and designed voices."
                    : "Save a result from voice cloning or voice design to build a reusable voice library."
                  : routeModels.length === 0
                    ? "Load a CustomVoice, Kokoro, or Voxtral TTS model to browse built-in voices."
                    : "Try a different built-in voice model."
            }
            className="mt-1"
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
      </div>
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
      title="Built-in voice models"
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
      emptyMessage="Load a CustomVoice, Kokoro, or Voxtral TTS model to browse built-in voices."
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
