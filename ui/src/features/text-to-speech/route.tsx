import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { Settings2 } from "lucide-react";

import { api, type ModelInfo, type SpeechHistoryRecord } from "@/api";
import { PageHeader, PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import {
  TEXT_TO_SPEECH_PREFERRED_MODELS,
  resolvePreferredRouteModel,
} from "@/features/models/catalog/routeModelCatalog";
import type { ModelDownloadProgressMap } from "@/features/models/downloadProgress";
import { RouteModelModal } from "@/features/models/components/RouteModelModal";
import { useRouteModelSelection } from "@/features/models/hooks/useRouteModelSelection";
import { TextToSpeechHistoryTable } from "@/features/text-to-speech/components/TextToSpeechHistoryTable";
import { NewTextToSpeechModal } from "@/features/text-to-speech/components/NewTextToSpeechModal";
import { TextToSpeechRecordDetail } from "@/features/text-to-speech/components/TextToSpeechRecordDetail";
import { useTextToSpeechHistory } from "@/features/text-to-speech/hooks/useTextToSpeechHistory";
import { useSpeechStreamPlayback } from "@/features/text-to-speech/hooks/useSpeechStreamPlayback";
import { useTextToSpeechRecord } from "@/features/text-to-speech/hooks/useTextToSpeechRecord";
import { useNewItemShortcut } from "@/hooks/useNewItemShortcut";
import {
  HistoryToolbar,
  SearchField,
  SegmentedFilter,
} from "@/components/history/HistoryToolbar";
import { NewItemButton } from "@/components/history/NewItemButton";
import {
  parseSpeechFilter,
  SPEECH_FILTER_OPTIONS,
  speechListFilters,
} from "@/features/text-to-speech/historyFilters";
import {
  normalizeSpeechProcessingStatus,
  resolveSpeechVoiceLabel,
} from "@/features/text-to-speech/support";

interface TextToSpeechPageProps {
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
}

export function TextToSpeechPage({
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
}: TextToSpeechPageProps) {
  const { recordId } = useParams<{ recordId: string }>();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const appliedQueryModelRef = useRef(false);
  const autoOpenCreateRef = useRef(false);
  const [isNewTextToSpeechModalOpen, setIsNewTextToSpeechModalOpen] =
    useState(false);
  const [streamingRecord, setStreamingRecord] =
    useState<SpeechHistoryRecord | null>(null);
  const [recordActionError, setRecordActionError] = useState<string | null>(
    null,
  );
  const [recordDeletePending, setRecordDeletePending] = useState(false);
  const [recordCancelPending, setRecordCancelPending] = useState(false);
  const [savedVoiceNameById, setSavedVoiceNameById] = useState<
    Record<string, string>
  >({});
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
  } = useRouteModelSelection({
    models,
    selectedModel,
    onSelect,
    modelFilter: (variant) => {
      const match = models.find((model) => model.variant === variant);
      const capabilities = match?.speech_capabilities;
      return Boolean(
        capabilities &&
          !variant.includes("Tokenizer") &&
          (capabilities.supports_builtin_voices ||
            capabilities.supports_reference_voice ||
            capabilities.supports_voice_description),
      );
    },
    resolveSelectedModel: (availableRouteModels, currentModel) =>
      resolvePreferredRouteModel({
        models: availableRouteModels,
        selectedModel: currentModel,
        preferredVariants: TEXT_TO_SPEECH_PREFERRED_MODELS,
      }),
  });

  // List search and filter live in the URL, so Back from a record keeps them.
  const historySearch = searchParams.get("q") ?? "";
  const historyFilter = parseSpeechFilter(searchParams.get("show"));
  const setHistoryParam = useCallback(
    (key: "q" | "show", value: string) => {
      const next = new URLSearchParams(searchParams);
      if (value && !(key === "show" && value === "all")) {
        next.set(key, value);
      } else {
        next.delete(key);
      }
      setSearchParams(next, { replace: true });
    },
    [searchParams, setSearchParams],
  );
  // "Today" starts at local midnight; fix it per filter choice so polling
  // keeps the same query.
  const historyFilters = useMemo(
    () => speechListFilters(historySearch, historyFilter),
    [historyFilter, historySearch],
  );
  const {
    records,
    total: historyTotal,
    loading: historyLoading,
    loadingMore: historyLoadingMore,
    error: historyError,
    hasMoreRecords,
    loadMoreRecords,
    refresh: refreshHistory,
  } = useTextToSpeechHistory(historyFilters);
  const {
    record,
    loading: recordLoading,
    error: recordError,
    refresh: refreshRecord,
  } = useTextToSpeechRecord(recordId);

  const playback = useSpeechStreamPlayback(recordId, {
    onStart: () => {
      setStreamingRecord((current) => current
        ? { ...current, processing_status: "processing" } : current);
    },
    onFinal: ({ record: finalRecord }) => {
      setStreamingRecord(finalRecord);
      void refreshHistory();
    },
    onError: () => {
      void refreshRecord();
      void refreshHistory();
    },
  });

  const refreshSavedVoiceNames = useCallback(async () => {
    try {
      const voices = await api.listSavedVoices();
      setSavedVoiceNameById(
        voices.reduce<Record<string, string>>((acc, voice) => {
          acc[voice.id] = voice.name;
          return acc;
        }, {}),
      );
    } catch {
      setSavedVoiceNameById({});
    }
  }, []);

  useEffect(() => {
    if (appliedQueryModelRef.current || recordId) {
      return;
    }

    const requestedModel = searchParams.get("model");
    const requestedSavedVoiceId = searchParams.get("voiceId");
    const requestedSpeaker = searchParams.get("speaker");

    if (!requestedModel && !requestedSavedVoiceId && !requestedSpeaker) {
      appliedQueryModelRef.current = true;
      return;
    }

    const requestedModelInfo = requestedModel
      ? routeModels.find((model) => model.variant === requestedModel) ?? null
      : null;

    const resolveModelForCapability = (
      capability: "supports_reference_voice" | "supports_builtin_voices",
    ) => {
      const capabilityModels = routeModels.filter(
        (model) => model.speech_capabilities?.[capability] === true,
      );
      if (capabilityModels.length === 0) {
        return null;
      }

      const selectedCapabilityModel =
        resolvedSelectedModel &&
        capabilityModels.some((model) => model.variant === resolvedSelectedModel)
          ? resolvedSelectedModel
          : null;

      return resolvePreferredRouteModel({
        models: capabilityModels,
        selectedModel: selectedCapabilityModel,
        preferredVariants: TEXT_TO_SPEECH_PREFERRED_MODELS,
      });
    };

    let targetModel: string | null = null;

    if (requestedSavedVoiceId) {
      targetModel =
        requestedModelInfo?.speech_capabilities?.supports_reference_voice
          ? requestedModelInfo.variant
          : resolveModelForCapability("supports_reference_voice");
    } else if (requestedSpeaker) {
      targetModel =
        requestedModelInfo?.speech_capabilities?.supports_builtin_voices
          ? requestedModelInfo.variant
          : resolveModelForCapability("supports_builtin_voices");
    } else if (requestedModelInfo) {
      targetModel = requestedModelInfo.variant;
    }

    if (!targetModel) {
      if (loading || routeModels.length === 0) {
        return;
      }
      if (!requestedSavedVoiceId && !requestedSpeaker) {
        appliedQueryModelRef.current = true;
      }
      return;
    }

    if (selectedModel !== targetModel) {
      onSelect(targetModel);
    }

    appliedQueryModelRef.current = true;
  }, [
    loading,
    onSelect,
    recordId,
    resolvedSelectedModel,
    routeModels,
    searchParams,
    selectedModel,
  ]);

  useEffect(() => {
    void refreshSavedVoiceNames();
  }, [refreshSavedVoiceNames]);

  useEffect(() => {
    if (!recordId) {
      setRecordActionError(null);
      return;
    }

    setRecordActionError(null);
    setStreamingRecord((current) =>
      current?.id === recordId ? current : null,
    );
  }, [recordId]);

  useEffect(() => {
    if (!record || !streamingRecord || record.id !== streamingRecord.id) {
      return;
    }

    const processingStatus = normalizeSpeechProcessingStatus(
      record.processing_status,
      record.processing_error,
    );
    if (processingStatus === "ready" || processingStatus === "failed") {
      setStreamingRecord(null);
    }
  }, [record, streamingRecord]);

  useNewItemShortcut(() => setIsNewTextToSpeechModalOpen(true));

  // ?create=speech (the command palette) opens the creation modal, then
  // drops the parameter so a refresh or Back does not reopen it.
  useEffect(() => {
    if (recordId || searchParams.get("create") !== "speech") {
      return;
    }
    setIsNewTextToSpeechModalOpen(true);
    const next = new URLSearchParams(searchParams);
    next.delete("create");
    setSearchParams(next, { replace: true });
  }, [recordId, searchParams, setSearchParams]);

  useEffect(() => {
    if (recordId || autoOpenCreateRef.current) {
      return;
    }

    if (searchParams.get("voiceId") || searchParams.get("speaker")) {
      setIsNewTextToSpeechModalOpen(true);
      autoOpenCreateRef.current = true;
    }
  }, [recordId, searchParams]);

  const visibleRecord = useMemo(() => {
    if (!recordId) {
      return null;
    }
    if (!streamingRecord || streamingRecord.id !== recordId) {
      return record;
    }
    if (!record) {
      return streamingRecord;
    }

    return {
      ...record,
      processing_status: streamingRecord.processing_status,
      processing_error:
        streamingRecord.processing_error ?? record.processing_error,
      model_id: streamingRecord.model_id || record.model_id,
      speaker: streamingRecord.speaker || record.speaker,
      language: streamingRecord.language || record.language,
      saved_voice_id: streamingRecord.saved_voice_id || record.saved_voice_id,
      speed: streamingRecord.speed ?? record.speed,
      input_text: streamingRecord.input_text || record.input_text,
      voice_description:
        streamingRecord.voice_description || record.voice_description,
      reference_text: streamingRecord.reference_text || record.reference_text,
      generation_time_ms:
        streamingRecord.generation_time_ms || record.generation_time_ms,
      audio_duration_secs:
        record.audio_duration_secs ?? streamingRecord.audio_duration_secs,
      rtf: record.rtf ?? streamingRecord.rtf,
      tokens_generated:
        record.tokens_generated ?? streamingRecord.tokens_generated,
      audio_filename: record.audio_filename || streamingRecord.audio_filename,
      audio_mime_type: record.audio_mime_type || streamingRecord.audio_mime_type,
    };
  }, [record, recordId, streamingRecord]);


  const visibleVoiceLabel = useMemo(
    () =>
      visibleRecord
        ? resolveSpeechVoiceLabel({
            savedVoiceId: visibleRecord.saved_voice_id,
            speaker: visibleRecord.speaker,
            modelId: visibleRecord.model_id,
            savedVoiceNameById,
          })
        : null,
    [savedVoiceNameById, visibleRecord],
  );

  const detailAudioUrl = useMemo(
    () =>
      recordId ? api.textToSpeechRecordAudioUrl(recordId) : null,
    [recordId],
  );

  const handleDetailDelete = async () => {
    if (!recordId || recordDeletePending) {
      return;
    }

    setRecordDeletePending(true);
    setRecordActionError(null);
    try {
      await api.deleteTextToSpeechRecord(recordId);
      setStreamingRecord(null);
      await refreshHistory();
      navigate("/text-to-speech", { replace: true });
    } catch (err) {
      setRecordActionError(
        err instanceof Error ? err.message : "Failed to delete generation.",
      );
    } finally {
      setRecordDeletePending(false);
    }
  };

  const handleDetailCancel = async () => {
    if (!recordId || recordCancelPending) {
      return;
    }

    setRecordCancelPending(true);
    setRecordActionError(null);
    try {
      playback.stop(false);
      const cancelled = await api.cancelTextToSpeechRecord(recordId);
      setStreamingRecord(cancelled.record);
      await Promise.all([refreshRecord(), refreshHistory()]);
    } catch (err) {
      setRecordActionError(
        err instanceof Error ? err.message : "Failed to cancel generation.",
      );
    } finally {
      setRecordCancelPending(false);
    }
  };

  const handleDeleteRecord = useCallback(
    async (targetRecordId: string) => {
      await api.deleteTextToSpeechRecord(targetRecordId);
      if (streamingRecord?.id === targetRecordId) {
        setStreamingRecord(null);
      }
      if (recordId === targetRecordId) {
        setRecordActionError(null);
      }
      await refreshHistory();
      if (recordId === targetRecordId) {
        navigate("/text-to-speech", { replace: true });
      }
    },
    [navigate, recordId, refreshHistory, streamingRecord?.id],
  );

  const handleCancelRecord = useCallback(
    async (targetRecordId: string) => {
      if (streamingRecord?.id === targetRecordId) playback.stop(false);
      await api.cancelTextToSpeechRecord(targetRecordId);
      if (streamingRecord?.id === targetRecordId) {
        setStreamingRecord(null);
      }
      await refreshHistory();
    },
    [playback, refreshHistory, streamingRecord?.id],
  );

  return (
    <PageShell>
      {playback.status !== "idle" && (
        <div role="status" className="flex items-center justify-between gap-3 rounded-md border p-3">
          <span>{playback.status === "playing" ? "Playing generated speech" : "Waiting for first audio"}</span>
          {playback.progress && <span role="status">{playback.progress.completedSegments} of {playback.progress.totalSegments} sections generated</span>}
          <Button variant="outline" size="sm" onClick={() => playback.stop()}>Stop generation and playback</Button>
        </div>
      )}
      {playback.error && <p role="alert" className="text-sm text-destructive">{playback.error}</p>}
      {playback.status === "idle" && visibleRecord?.model_id === "FishAudio-S2-Pro" &&
        (visibleRecord.processing_status === "pending" || visibleRecord.processing_status === "processing") && (
        <Button variant="outline" onClick={() => { void playback.listen(visibleRecord).catch(() => {}); }}>
          Listen from beginning
        </Button>
      )}
      {recordId ? (
        <>
          <TextToSpeechRecordDetail
            parentCrumb={{ label: "Text to Speech", to: "/text-to-speech" }}
            extraMenuItems={[
              {
                id: "models",
                label: "Models",
                icon: Settings2,
                onSelect: openModelManager,
              },
            ]}
            record={visibleRecord}
            audioUrl={detailAudioUrl}
            voiceLabel={visibleVoiceLabel}
            loading={recordLoading}
            error={recordError}
            deleteError={recordActionError}
            deletePending={recordDeletePending}
            cancelPending={recordCancelPending}
            onCancel={() => void handleDetailCancel()}
            onDelete={() => void handleDetailDelete()}
          />
        </>
      ) : (
        <>
          <PageHeader
            title="Text to Speech"
            description="Your generated speech. Play, reuse or export any take."
            actions={
              <>
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
                <NewItemButton
                  label="New generation"
                  onClick={() => setIsNewTextToSpeechModalOpen(true)}
                />
              </>
            }
          />

          <HistoryToolbar>
            <SearchField
              value={historySearch}
              onChange={(value) => setHistoryParam("q", value)}
              placeholder="Search text or voice…"
            />
            <SegmentedFilter
              label="Show"
              options={SPEECH_FILTER_OPTIONS}
              value={historyFilter}
              onChange={(value) => setHistoryParam("show", value)}
              selectedCount={historyLoading ? null : historyTotal}
            />
          </HistoryToolbar>

          <TextToSpeechHistoryTable
            records={records}
            total={historyTotal}
            filtered={Boolean(historySearch.trim()) || historyFilter !== "all"}
            savedVoiceNameById={savedVoiceNameById}
            loading={historyLoading}
            error={historyError}
            loadMore={{
              canLoadMore: hasMoreRecords,
              loading: historyLoadingMore,
              onLoadMore: () => {
                void loadMoreRecords();
              },
            }}
            onRefresh={() => void refreshHistory()}
            onOpenRecord={(nextRecordId) => {
              navigate(`/text-to-speech/${nextRecordId}`);
            }}
            onDeleteRecord={handleDeleteRecord}
            onCancelRecord={handleCancelRecord}
          />
        </>
      )}

      <NewTextToSpeechModal
        isOpen={isNewTextToSpeechModalOpen}
        onClose={() => setIsNewTextToSpeechModalOpen(false)}
        selectedModel={resolvedSelectedModel}
        selectedModelInfo={selectedModelInfo}
        selectedModelReady={selectedModelReady}
        isModelManagerOpen={isModelModalOpen}
        initialSavedVoiceId={searchParams.get("voiceId")}
        initialSpeaker={searchParams.get("speaker")}
        onLoadSelectedModel={(variant) => {
          onLoad(variant);
        }}
        onUnloadSelectedModel={(variant) => {
          onUnload(variant);
        }}
        onOpenModelManager={openModelManager}
        onModelRequired={() => {
          requestModel();
          onError(
            "Select and load a compatible TTS model before creating a text-to-speech job.",
          );
        }}
        onCreated={async (createdRecord) => {
          setRecordActionError(null);
          setStreamingRecord(createdRecord);
          navigate(`/text-to-speech/${createdRecord.id}`);
          void refreshHistory();
        }}
        onCreateStream={playback.create}
      />

      <RouteModelModal
        isOpen={isModelModalOpen}
        onClose={closeModelModal}
        title="Text-to-Speech Models"
        description="Manage built-in voice, saved-voice, and voice-direction TTS models for this route."
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
        onUseModel={onSelect}
        emptyMessage="Load a compatible TTS model with built-in voices, saved voices, or voice-direction prompts to generate speech."
        zIndexClassName={isNewTextToSpeechModalOpen ? "z-[70]" : "z-50"}
      />
    </PageShell>
  );
}
