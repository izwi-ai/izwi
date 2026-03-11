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
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { getSpeakerProfilesForVariant } from "@/types";
import {
  TEXT_TO_SPEECH_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import { useRouteModelSelection } from "@/features/models/hooks/useRouteModelSelection";

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

  return (
    <PageShell>
      <PageHeader
        title="Voices"
        description="Manage saved cloned and designed voices, browse built-in voice libraries, and hand them off into text-to-speech."
      />

      <div className="grid gap-6 xl:grid-cols-[340px_minmax(0,1fr)]">
        <Card>
          <CardContent className="space-y-6 p-6">
            <div className="space-y-1">
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                Voice Library
              </div>
              <h2 className="text-lg font-semibold text-foreground">
                Reusable voices first
              </h2>
              <p className="text-sm leading-relaxed text-muted-foreground">
                Saved voices are reusable assets. Built-in voices depend on the
                active model and can be previewed here before opening TTS.
              </p>
            </div>

            <div className="space-y-2.5">
              <label className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                Search voices
              </label>
              <Input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search by name, language, or transcript"
                className="bg-muted/20"
              />
            </div>

            <div className="space-y-4 rounded-2xl border border-border/60 bg-muted/20 p-5">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                    Built-in model
                  </div>
                  <div className="mt-1 text-sm font-semibold text-foreground">
                    {resolvedSelectedModel ?? "No model selected"}
                  </div>
                  <div className="mt-1 text-xs text-muted-foreground">
                    {selectedModelReady
                      ? "Loaded and ready for built-in previews"
                      : "Choose a built-in voice model before previewing"}
                  </div>
                </div>
                <Button variant="outline" size="sm" onClick={openModelManager}>
                  Models
                </Button>
              </div>
              <div className="rounded-xl border border-border/60 bg-background/50 px-3.5 py-3 text-xs leading-relaxed text-muted-foreground">
                {selectedModelInfo?.speech_capabilities?.built_in_voice_count
                  ? `${selectedModelInfo.speech_capabilities.built_in_voice_count} built-in voices available on the active model.`
                  : "Built-in voices appear when a CustomVoice, Kokoro, or LFM2 Audio model is selected."}
              </div>
            </div>

            <div className="space-y-3">
              <div className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                Saved voice filters
              </div>
              <div className="flex flex-wrap gap-2.5">
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

            <div className="flex flex-wrap gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => void loadSavedVoices()}
                disabled={savedVoicesLoading}
              >
                {savedVoicesLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <RefreshCw className="h-4 w-4" />
                )}
                Refresh voices
              </Button>
            </div>
          </CardContent>
        </Card>

        <Card className="min-h-[36rem]">
          <CardContent className="p-6">
            <Tabs
              value={activeTab}
              onValueChange={(value) =>
                setActiveTab(value as "saved" | "built-in")
              }
              className="space-y-5"
            >
              <TabsList>
                <TabsTrigger value="saved">
                  <Library className="h-4 w-4" />
                  My Voices
                </TabsTrigger>
                <TabsTrigger value="built-in">
                  <Sparkles className="h-4 w-4" />
                  Built-in Voices
                </TabsTrigger>
              </TabsList>

              <TabsContent value="saved" className="mt-0">
                {savedVoicesError ? (
                  <Card className="border-destructive/40 bg-destructive/5">
                    <CardContent className="p-5 text-sm text-destructive">
                      {savedVoicesError}
                    </CardContent>
                  </Card>
                ) : null}
                <VoicePicker
                  items={savedVoiceItems}
                  emptyTitle={
                    savedVoicesLoading
                      ? "Loading saved voices"
                      : "No saved voices yet"
                  }
                  emptyDescription={
                    savedVoicesLoading
                      ? "Fetching your reusable cloned and designed voices."
                      : "Save a result from voice cloning or voice design to build a reusable voice library."
                  }
                />
              </TabsContent>

              <TabsContent value="built-in" className="mt-0">
                <VoicePicker
                  items={builtInVoiceItems}
                  emptyTitle="No built-in voices available"
                  emptyDescription={
                    routeModels.length === 0
                      ? "Load a CustomVoice, Kokoro, or LFM2 Audio model to browse built-in voices."
                      : "Try a different built-in voice model or search term."
                  }
                />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
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
        emptyMessage="Load a CustomVoice, Kokoro, or LFM2 Audio model to browse built-in voices."
      />
    </PageShell>
  );
}
