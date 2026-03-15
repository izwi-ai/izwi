import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Loader2,
  Library,
  Mic2,
  RefreshCw,
  Sparkles,
  Trash2,
} from "lucide-react";
import type { ModelInfo, SavedVoiceSummary } from "@/api";
import { api } from "@/api";
import { PageHeader, PageShell } from "@/components/PageShell";
import { VoicePicker, type VoicePickerItem } from "@/components/VoicePicker";
import {
  VOICE_ROUTE_BODY_COPY_CLASS,
  VOICE_ROUTE_META_COPY_CLASS,
  VOICE_ROUTE_PANEL_TITLE_CLASS,
  VOICE_ROUTE_SECTION_LABEL_CLASS,
  VOICE_ROUTE_WORKSPACE_DESCRIPTION_CLASS,
  VOICE_ROUTE_WORKSPACE_TITLE_CLASS,
} from "@/components/voiceRouteTypography";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { getSpeakerProfilesForVariant, isLfm25AudioVariant } from "@/types";
import {
  TEXT_TO_SPEECH_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import { useRouteModelSelection } from "@/features/models/hooks/useRouteModelSelection";
import { cn } from "@/lib/utils";

interface VoicesPageProps {
  models: ModelInfo[];
  selectedModel: string | null;
  loading: boolean;
  downloadProgress: Record<
    string,
    {
      percent: number;
      currentFile: string;
      status: string;
      downloadedBytes: number;
      totalBytes: number;
    }
  >;
  onDownload: (variant: string) => void;
  onCancelDownload?: (variant: string) => void;
  onLoad: (variant: string) => void;
  onUnload: (variant: string) => void;
  onDelete: (variant: string) => void;
  onSelect: (variant: string) => void;
  onError: (message: string) => void;
}

type SavedVoiceFilter = "all" | "voice_cloning" | "voice_design";

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
}: VoicesPageProps) {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<"saved" | "built-in">("saved");
  const [search, setSearch] = useState("");
  const [savedVoiceFilter, setSavedVoiceFilter] =
    useState<SavedVoiceFilter>("all");
  const [savedVoices, setSavedVoices] = useState<SavedVoiceSummary[]>([]);
  const [savedVoicesLoading, setSavedVoicesLoading] = useState(true);
  const [savedVoicesError, setSavedVoicesError] = useState<string | null>(null);
  const [deletingVoiceId, setDeletingVoiceId] = useState<string | null>(null);
  const [previewLoadingVoiceId, setPreviewLoadingVoiceId] = useState<
    string | null
  >(null);
  const [previewUrls, setPreviewUrls] = useState<Record<string, string>>({});
  const previewUrlsRef = useRef<Record<string, string>>({});

  const {
    routeModels,
    resolvedSelectedModel,
    selectedModelInfo,
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
    return () => {
      Object.values(previewUrlsRef.current).forEach((url) => {
        if (url.startsWith("blob:")) {
          URL.revokeObjectURL(url);
        }
      });
    };
  }, []);

  const loadSavedVoices = async () => {
    setSavedVoicesLoading(true);
    setSavedVoicesError(null);
    try {
      const records = await api.listSavedVoices();
      setSavedVoices(records);
    } catch (error) {
      setSavedVoicesError(
        error instanceof Error ? error.message : "Failed to load saved voices.",
      );
    } finally {
      setSavedVoicesLoading(false);
    }
  };

  useEffect(() => {
    void loadSavedVoices();
  }, []);

  const builtInVoices = useMemo(
    () => getSpeakerProfilesForVariant(resolvedSelectedModel),
    [resolvedSelectedModel],
  );

  const filteredSavedVoices = useMemo(() => {
    const normalizedQuery = search.trim().toLowerCase();
    return savedVoices.filter((voice) => {
      const matchesSource =
        savedVoiceFilter === "all" ||
        voice.source_route_kind === savedVoiceFilter;
      if (!matchesSource) {
        return false;
      }
      if (!normalizedQuery) {
        return true;
      }
      return (
        voice.name.toLowerCase().includes(normalizedQuery) ||
        voice.reference_text_preview.toLowerCase().includes(normalizedQuery)
      );
    });
  }, [savedVoices, savedVoiceFilter, search]);

  const filteredBuiltInVoices = useMemo(() => {
    const normalizedQuery = search.trim().toLowerCase();
    return builtInVoices.filter((voice) => {
      if (!normalizedQuery) {
        return true;
      }
      return (
        voice.name.toLowerCase().includes(normalizedQuery) ||
        voice.language.toLowerCase().includes(normalizedQuery) ||
        voice.description.toLowerCase().includes(normalizedQuery)
      );
    });
  }, [builtInVoices, search]);

  const handleUseSavedVoice = (voiceId: string) => {
    navigate(`/text-to-speech?voiceId=${encodeURIComponent(voiceId)}`);
  };

  const handleUseBuiltInVoice = (speaker: string) => {
    const params = new URLSearchParams();
    params.set("speaker", speaker);
    if (resolvedSelectedModel) {
      params.set("model", resolvedSelectedModel);
    }
    navigate(`/text-to-speech?${params.toString()}`);
  };

  const handleDeleteVoice = async (voiceId: string) => {
    setDeletingVoiceId(voiceId);
    try {
      await api.deleteSavedVoice(voiceId);
      setSavedVoices((current) =>
        current.filter((voice) => voice.id !== voiceId),
      );
    } catch (error) {
      onError(
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
    if (previewUrls[voiceId]) {
      return;
    }
    if (!selectedModelReady) {
      requestModel(resolvedSelectedModel);
      onError("Load the selected voice model before generating a preview.");
      return;
    }

    setPreviewLoadingVoiceId(voiceId);
    try {
      const result = await api.generateTTSWithStats({
        model_id: resolvedSelectedModel,
        text: previewTextForLanguage(language),
        speaker: voiceId,
      });
      const url = URL.createObjectURL(result.audioBlob);
      setPreviewUrls((current) => ({ ...current, [voiceId]: url }));
    } catch (error) {
      onError(
        error instanceof Error
          ? error.message
          : "Failed to generate built-in voice preview.",
      );
    } finally {
      setPreviewLoadingVoiceId(null);
    }
  };

  const savedVoiceItems: VoicePickerItem[] = filteredSavedVoices.map(
    (voice) => ({
      id: voice.id,
      name: voice.name,
      categoryLabel: savedVoiceSourceLabel(voice.source_route_kind),
      description: voice.reference_text_preview,
      meta: [
        `${voice.reference_text_chars} chars`,
        formatRelativeDate(voice.updated_at || voice.created_at),
      ],
      previewUrl: api.savedVoiceAudioUrl(voice.id),
      actions: (
        <>
          <Button
            size="sm"
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
            onClick={(event) => {
              event.stopPropagation();
              void handleDeleteVoice(voice.id);
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

  const builtInVoiceItems: VoicePickerItem[] = filteredBuiltInVoices.map(
    (voice) => ({
      id: voice.id,
      name: voice.name,
      categoryLabel: selectedModelInfo?.variant ?? "Built-in voice",
      description: voice.description,
      meta: [voice.language],
      previewUrl: previewUrls[voice.id] ?? null,
      previewMessage: previewUrls[voice.id]
        ? null
        : "Generate a preview sample to audition this built-in voice.",
      actions: (
        <>
          <Button
            size="sm"
            variant="outline"
            onClick={(event) => {
              event.stopPropagation();
              void handlePreviewBuiltInVoice(voice.id, voice.language);
            }}
            disabled={previewLoadingVoiceId === voice.id}
          >
            {previewLoadingVoiceId === voice.id ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Sparkles className="h-4 w-4" />
            )}
            Preview
          </Button>
          <Button
            size="sm"
            onClick={(event) => {
              event.stopPropagation();
              handleUseBuiltInVoice(voice.id);
            }}
          >
            <Mic2 className="h-4 w-4" />
            Use in TTS
          </Button>
        </>
      ),
    }),
  );

  const totalSavedVoices = savedVoices.length;
  const clonedVoiceCount = savedVoices.filter(
    (voice) => voice.source_route_kind === "voice_cloning",
  ).length;
  const designedVoiceCount = savedVoices.filter(
    (voice) => voice.source_route_kind === "voice_design",
  ).length;
  const activeResultCount =
    activeTab === "saved"
      ? filteredSavedVoices.length
      : filteredBuiltInVoices.length;
  const activeSearchPlaceholder =
    activeTab === "saved"
      ? "Search saved voices by name or transcript"
      : "Search built-in voices by name or language";
  const activeItems = activeTab === "saved" ? savedVoiceItems : builtInVoiceItems;
  const activeResultsLabel =
    activeResultCount === 1 ? "1 result" : `${activeResultCount} results`;
  const activePanelTitle =
    activeTab === "saved" ? "Reusable voice library" : "Built-in voice library";
  const activePanelDescription =
    activeTab === "saved"
      ? "Saved voices come from voice cloning and voice design and are ready to reuse in TTS."
      : "Built-in voices belong to the active model. Generate a preview first, then jump into TTS with the selected speaker.";

  return (
    <PageShell>
      <PageHeader
        title="Voices"
        description="Manage saved cloned and designed voices, browse built-in voice libraries, and hand them off into text-to-speech."
      />

      <div className="card flex min-h-0 flex-col p-4 xl:h-[calc(100dvh-11.75rem)]">
        <div className="mb-4 flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="flex items-center gap-3">
            <div className="rounded border border-[var(--border-muted)] bg-[var(--bg-surface-2)] p-2">
              <Library className="h-5 w-5 text-[var(--text-muted)]" />
            </div>
            <div>
              <h2 className={VOICE_ROUTE_WORKSPACE_TITLE_CLASS}>
                Voice browser
              </h2>
              <p className={VOICE_ROUTE_WORKSPACE_DESCRIPTION_CLASS}>
                Reusable saved voices stay ready for TTS. Built-in speakers come
                from the active model.
              </p>
            </div>
          </div>

          <Tabs
            value={activeTab}
            onValueChange={(value) => setActiveTab(value as "saved" | "built-in")}
            className="w-full max-w-md"
          >
            <TabsList className="grid w-full grid-cols-2 border-[var(--border-strong)] bg-[var(--bg-surface-2)] p-1 shadow-sm">
              <TabsTrigger
                value="saved"
                className="justify-between gap-2 text-[var(--text-muted)] hover:text-[var(--text-primary)] data-[state=active]:bg-[var(--accent-solid)] data-[state=active]:text-[var(--text-on-accent)] data-[state=active]:shadow-[0_8px_20px_-14px_rgba(17,17,17,0.55)]"
              >
                <span className="inline-flex items-center gap-2">
                  <Library className="h-4 w-4" />
                  My Voices
                </span>
                <span className="rounded-full border border-current/15 px-2 py-0.5 text-[10px] font-semibold">
                  {totalSavedVoices}
                </span>
              </TabsTrigger>
              <TabsTrigger
                value="built-in"
                className="justify-between gap-2 text-[var(--text-muted)] hover:text-[var(--text-primary)] data-[state=active]:bg-[var(--accent-solid)] data-[state=active]:text-[var(--text-on-accent)] data-[state=active]:shadow-[0_8px_20px_-14px_rgba(17,17,17,0.55)]"
              >
                <span className="inline-flex items-center gap-2">
                  <Sparkles className="h-4 w-4" />
                  Built-in Voices
                </span>
                <span className="rounded-full border border-current/15 px-2 py-0.5 text-[10px] font-semibold">
                  {builtInVoices.length}
                </span>
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </div>

        <div className="grid min-h-0 flex-1 gap-5 xl:grid-cols-[290px_minmax(0,1fr)]">
          <aside className="space-y-4">
            <section className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5">
              <div className={VOICE_ROUTE_SECTION_LABEL_CLASS}>Search</div>
              <Input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder={activeSearchPlaceholder}
                className="mt-3 bg-[var(--bg-surface-1)]"
              />
              <div
                className={cn(
                  VOICE_ROUTE_META_COPY_CLASS,
                  "mt-3 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-2",
                )}
              >
                {activeResultsLabel} matching the current tab.
              </div>
            </section>

            {activeTab === "saved" ? (
              <section className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className={VOICE_ROUTE_SECTION_LABEL_CLASS}>
                      Saved Library
                    </div>
                    <h3 className={cn(VOICE_ROUTE_PANEL_TITLE_CLASS, "mt-1")}>
                      Reusable saved voices
                    </h3>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => void loadSavedVoices()}
                    disabled={savedVoicesLoading}
                    className="bg-[var(--bg-surface-1)]"
                  >
                    {savedVoicesLoading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <RefreshCw className="h-4 w-4" />
                    )}
                    Refresh
                  </Button>
                </div>

                <div className="mt-4 grid gap-2 sm:grid-cols-3 xl:grid-cols-1">
                  <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-3">
                    <div className={VOICE_ROUTE_SECTION_LABEL_CLASS}>Total</div>
                    <div className="mt-1 text-xl font-semibold text-[var(--text-primary)]">
                      {totalSavedVoices}
                    </div>
                  </div>
                  <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-3">
                    <div className={VOICE_ROUTE_SECTION_LABEL_CLASS}>Cloned</div>
                    <div className="mt-1 text-xl font-semibold text-[var(--text-primary)]">
                      {clonedVoiceCount}
                    </div>
                  </div>
                  <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-3">
                    <div className={VOICE_ROUTE_SECTION_LABEL_CLASS}>Designed</div>
                    <div className="mt-1 text-xl font-semibold text-[var(--text-primary)]">
                      {designedVoiceCount}
                    </div>
                  </div>
                </div>

                <div className="mt-4 space-y-3">
                  <div className={VOICE_ROUTE_SECTION_LABEL_CLASS}>Filters</div>
                  <div className="flex flex-wrap gap-2">
                    {(
                      [
                        ["all", "All voices"],
                        ["voice_cloning", "Cloned"],
                        ["voice_design", "Designed"],
                      ] as const
                    ).map(([value, label]) => (
                      <Button
                        key={value}
                        type="button"
                        size="sm"
                        variant={savedVoiceFilter === value ? "default" : "outline"}
                        onClick={() => setSavedVoiceFilter(value)}
                      >
                        {label}
                      </Button>
                    ))}
                  </div>
                </div>
              </section>
            ) : (
              <section className="rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className={VOICE_ROUTE_SECTION_LABEL_CLASS}>
                      Built-in Model
                    </div>
                    <h3 className={cn(VOICE_ROUTE_PANEL_TITLE_CLASS, "mt-1")}>
                      {resolvedSelectedModel ?? "No model selected"}
                    </h3>
                    <div
                      className={cn(
                        "mt-2 text-xs",
                        selectedModelReady
                          ? "text-[var(--text-secondary)]"
                          : "text-amber-500",
                      )}
                    >
                      {selectedModelReady
                        ? "Loaded and ready for voice previews"
                        : "Choose and load a supported model before previewing"}
                    </div>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={openModelManager}
                    className="bg-[var(--bg-surface-1)]"
                  >
                    Models
                  </Button>
                </div>

                <div
                  className={cn(
                    VOICE_ROUTE_BODY_COPY_CLASS,
                    "mt-4 rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3.5 py-3",
                  )}
                >
                  {selectedModelInfo?.speech_capabilities?.built_in_voice_count
                    ? `${selectedModelInfo.speech_capabilities.built_in_voice_count} built-in speakers are available on the active model.`
                    : "Built-in voices appear when a CustomVoice or Kokoro model is selected."}
                </div>

                <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-1">
                  <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-3">
                    <div className={VOICE_ROUTE_SECTION_LABEL_CLASS}>
                      Visible voices
                    </div>
                    <div className="mt-1 text-xl font-semibold text-[var(--text-primary)]">
                      {filteredBuiltInVoices.length}
                    </div>
                  </div>
                  <div className="rounded-xl border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-3 py-3">
                    <div className={VOICE_ROUTE_SECTION_LABEL_CLASS}>
                      Route models
                    </div>
                    <div className="mt-1 text-xl font-semibold text-[var(--text-primary)]">
                      {routeModels.length}
                    </div>
                  </div>
                </div>
              </section>
            )}
          </aside>

          <section className="flex min-h-0 flex-col rounded-2xl border border-[var(--border-muted)] bg-[var(--bg-surface-0)] p-5 sm:p-6">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
              <div>
                <div className={VOICE_ROUTE_SECTION_LABEL_CLASS}>
                  {activeTab === "saved" ? "Saved voices" : "Built-in voices"}
                </div>
                <h3 className={cn(VOICE_ROUTE_PANEL_TITLE_CLASS, "mt-1")}>
                  {activePanelTitle}
                </h3>
                <p className={cn(VOICE_ROUTE_BODY_COPY_CLASS, "mt-1 max-w-3xl")}>
                  {activePanelDescription}
                </p>
              </div>

              <div className="flex flex-wrap gap-2 text-xs">
                <span className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-1 text-[var(--text-muted)]">
                  {activeResultsLabel}
                </span>
                {activeTab === "saved" && savedVoiceFilter !== "all" ? (
                  <span className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-1 text-[var(--text-muted)]">
                    {savedVoiceFilter === "voice_cloning" ? "Cloned only" : "Designed only"}
                  </span>
                ) : null}
                {activeTab === "built-in" && resolvedSelectedModel ? (
                  <span className="rounded-full border border-[var(--border-muted)] bg-[var(--bg-surface-1)] px-2.5 py-1 text-[var(--text-muted)]">
                    {resolvedSelectedModel}
                  </span>
                ) : null}
              </div>
            </div>

            {activeTab === "saved" && savedVoicesError ? (
              <div className="mt-4 rounded-xl border border-[var(--danger-border)] bg-[var(--danger-bg)] px-4 py-3 text-sm text-[var(--danger-text)]">
                {savedVoicesError}
              </div>
            ) : null}

            <div className="mt-5 min-h-0 flex-1 overflow-y-auto pr-1 scrollbar-thin">
              <VoicePicker
                items={activeItems}
                emptyTitle={
                  activeTab === "saved"
                    ? savedVoicesLoading
                      ? "Loading saved voices"
                      : "No saved voices yet"
                    : "No built-in voices available"
                }
                emptyDescription={
                  activeTab === "saved"
                    ? savedVoicesLoading
                      ? "Fetching your reusable cloned and designed voices."
                      : "Save a result from voice cloning or voice design to build a reusable voice library."
                    : routeModels.length === 0
                      ? "Load a CustomVoice or Kokoro model to browse built-in voices."
                      : "Try a different built-in voice model or search term."
                }
                className="grid-cols-[repeat(auto-fill,minmax(300px,1fr))]"
              />
            </div>
          </section>
        </div>
      </div>

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
    </PageShell>
  );
}
